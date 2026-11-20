
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import default_backend, lax

from clm_jax.physics.cn_zero import CnTransientFluxState, get_cn_flux_field
from clm_jax.physics.numerics import (
    fortran_divide,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.subgrid import patch_to_gridcell_mean

_USE_FUSED_PRODUCT_REDUCTION = False


class CnWoodProductsInputs(NamedTuple):

    soil_patch: jnp.ndarray
    patch_active: jnp.ndarray
    patch_gridcell_index: jnp.ndarray
    patch_weight_on_gridcell: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    dwt_wood_product_carbon_gain_patch: jnp.ndarray
    dwt_crop_product_carbon_gain_patch: jnp.ndarray
    dwt_wood_product_nitrogen_gain_patch: jnp.ndarray
    dwt_crop_product_nitrogen_gain_patch: jnp.ndarray


class CnProductSpeciesState(NamedTuple):

    crop_product_1year_gridcell: jnp.ndarray
    wood_product_10year_gridcell: jnp.ndarray
    wood_product_100year_gridcell: jnp.ndarray


class CnWoodProductsState(NamedTuple):

    carbon: CnProductSpeciesState
    nitrogen: CnProductSpeciesState


class CnWoodProductsParameters(NamedTuple):

    wood_harvest_10year_fraction_pft: jnp.ndarray
    dynamic_wood_10year_fraction_pft: jnp.ndarray
    dynamic_wood_100year_fraction_pft: jnp.ndarray
    time_step: float | jnp.ndarray = 1800.0


class CnProductSpeciesDiagnostics(NamedTuple):

    total_wood_product_gridcell: jnp.ndarray
    dwt_wood_product_10year_gain_gridcell: jnp.ndarray
    dwt_wood_product_100year_gain_gridcell: jnp.ndarray
    dwt_total_wood_product_gain_gridcell: jnp.ndarray
    dwt_crop_product_1year_gain_gridcell: jnp.ndarray
    harvest_deadstem_to_product_10year_patch: jnp.ndarray
    harvest_deadstem_to_product_10year_gridcell: jnp.ndarray
    harvest_deadstem_to_product_100year_patch: jnp.ndarray
    harvest_deadstem_to_product_100year_gridcell: jnp.ndarray
    grain_to_crop_product_1year_patch: jnp.ndarray
    grain_to_crop_product_1year_gridcell: jnp.ndarray
    crop_product_1year_loss_gridcell: jnp.ndarray
    wood_product_10year_loss_gridcell: jnp.ndarray
    wood_product_100year_loss_gridcell: jnp.ndarray
    total_wood_product_loss_gridcell: jnp.ndarray
    total_product_loss_gridcell: jnp.ndarray


class CnWoodProductsDiagnostics(NamedTuple):
    carbon: CnProductSpeciesDiagnostics
    nitrogen: CnProductSpeciesDiagnostics
    updated_gridcell_count: jnp.ndarray


class CnWoodProductsResult(NamedTuple):
    state: CnWoodProductsState
    cn_fluxes: CnTransientFluxState
    diagnostics: CnWoodProductsDiagnostics


_CROP_PRODUCT_1YEAR_DECAY_RATE = float.fromhex("0x1.353cd60000000p-24")
_WOOD_PRODUCT_10YEAR_DECAY_RATE = float.fromhex("0x1.eec7be0000000p-28")
_WOOD_PRODUCT_100YEAR_DECAY_RATE = float.fromhex("0x1.8bd2fe0000000p-31")

MAXIMUM_PATCHES_PER_GRIDCELL = 79

_CARBON_WOOD_HARVEST = "vegetation_carbon.wood_harvestc_patch"
_CARBON_GRAIN_PRODUCT = "vegetation_carbon.grainc_to_cropprodc_patch"
_NITROGEN_WOOD_HARVEST = "vegetation_nitrogen.wood_harvestn_patch"
_NITROGEN_GRAIN_PRODUCT = "vegetation_nitrogen.grainn_to_cropprodn_patch"


def _validate_shapes(
    inputs: CnWoodProductsInputs,
    state: CnWoodProductsState,
    cn_fluxes: CnTransientFluxState,
    parameters: CnWoodProductsParameters,
) -> tuple[int, int]:
    patch_count = inputs.soil_patch.shape[0]
    if jnp.ndim(inputs.soil_patch) != 1:
        raise ValueError("soil_patch must be a vector")
    for name in inputs._fields[1:]:
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")

    gridcell_count = state.carbon.crop_product_1year_gridcell.shape[0]
    for species in state:
        for pool in species:
            if jnp.shape(pool) != (gridcell_count,):
                raise ValueError("all product pools must contain every gridcell")

    for field_name in (
        _CARBON_WOOD_HARVEST,
        _CARBON_GRAIN_PRODUCT,
        _NITROGEN_WOOD_HARVEST,
        _NITROGEN_GRAIN_PRODUCT,
    ):
        if get_cn_flux_field(cn_fluxes, field_name).shape != (patch_count,):
            raise ValueError("product input fluxes must contain every patch")

    parameter_shapes = (
        jnp.shape(parameters.wood_harvest_10year_fraction_pft),
        jnp.shape(parameters.dynamic_wood_10year_fraction_pft),
        jnp.shape(parameters.dynamic_wood_100year_fraction_pft),
    )
    if len(set(parameter_shapes)) != 1:
        raise ValueError("product fraction parameters must be equally sized vectors")
    parameter_shape = parameter_shapes[0]
    if len(parameter_shape) != 1 or parameter_shape[0] == 0:
        raise ValueError("product fraction parameters must be nonempty vectors")
    if jnp.shape(parameters.time_step) != ():
        raise ValueError("time_step must be scalar")
    return patch_count, gridcell_count


def _contiguous_patch_to_gridcell_sum_reference(
    patch_values: jnp.ndarray,
    patch_gridcell: jnp.ndarray,
    number_of_gridcells: int,
) -> jnp.ndarray:

    number_of_patches = patch_values.shape[0]
    if number_of_patches == 0:
        return jnp.zeros(number_of_gridcells, dtype=jnp.float64)
    patch_index = jnp.arange(number_of_patches, dtype=jnp.int32)
    first_patch = jnp.full(
        number_of_gridcells,
        number_of_patches,
        dtype=jnp.int32,
    )
    first_patch = first_patch.at[patch_gridcell].min(patch_index)
    gridcell_index = jnp.arange(number_of_gridcells, dtype=jnp.int32)

    def accumulate_offset(offset: int, totals: jnp.ndarray) -> jnp.ndarray:
        unbounded_patch_index = first_patch + jnp.int32(offset)
        bounded_patch_index = jnp.clip(
            unbounded_patch_index,
            0,
            number_of_patches - 1,
        )
        belongs_to_gridcell = (
            (unbounded_patch_index < number_of_patches)
            & (patch_gridcell[bounded_patch_index] == gridcell_index)
        )
        candidate = fortran_add(totals, patch_values[bounded_patch_index])
        return jnp.where(belongs_to_gridcell, candidate, totals)

    return lax.fori_loop(
        0,
        MAXIMUM_PATCHES_PER_GRIDCELL,
        accumulate_offset,
        jnp.zeros(number_of_gridcells, dtype=jnp.float64),
    )


def _contiguous_patch_to_gridcell_sum(patch_values, patch_gridcell, number_of_gridcells):
    if _USE_FUSED_PRODUCT_REDUCTION and default_backend() == "gpu":
        from clm_jax.physics.ordered_gpu_reduction import ordered_unweighted_sum
        return ordered_unweighted_sum(
            patch_values, patch_gridcell, number_of_gridcells, MAXIMUM_PATCHES_PER_GRIDCELL,
        )
    return _contiguous_patch_to_gridcell_sum_reference(patch_values, patch_gridcell, number_of_gridcells)


def _partition_dynamic_wood(
    dynamic_wood_gain_patch: jnp.ndarray,
    patch_gridcell: jnp.ndarray,
    vegetation_type_patch: jnp.ndarray,
    dynamic_wood_10year_fraction_pft: jnp.ndarray,
    dynamic_wood_100year_fraction_pft: jnp.ndarray,
    number_of_gridcells: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    ten_fraction = dynamic_wood_10year_fraction_pft[vegetation_type_patch]
    hundred_fraction = dynamic_wood_100year_fraction_pft[vegetation_type_patch]
    total_fraction = fortran_add(ten_fraction, hundred_fraction)
    has_products = total_fraction > jnp.float64(0.0)
    safe_total = jnp.where(has_products, total_fraction, jnp.float64(1.0))
    normalized_ten = jnp.where(
        has_products,
        fortran_divide(ten_fraction, safe_total),
        jnp.float64(0.0),
    )
    normalized_hundred = jnp.where(
        has_products,
        fortran_divide(hundred_fraction, safe_total),
        jnp.float64(0.0),
    )
    ten_patch = fortran_multiply(dynamic_wood_gain_patch, normalized_ten)
    hundred_patch = fortran_multiply(
        dynamic_wood_gain_patch,
        normalized_hundred,
    )
    return (
        _contiguous_patch_to_gridcell_sum(
            ten_patch,
            patch_gridcell,
            number_of_gridcells,
        ),
        _contiguous_patch_to_gridcell_sum(
            hundred_patch,
            patch_gridcell,
            number_of_gridcells,
        ),
    )


def _update_species(
    state: CnProductSpeciesState,
    *,
    soil_patch: jnp.ndarray,
    patch_active: jnp.ndarray,
    patch_gridcell: jnp.ndarray,
    patch_weight: jnp.ndarray,
    vegetation_type_patch: jnp.ndarray,
    dynamic_wood_gain_patch: jnp.ndarray,
    wood_harvest_patch: jnp.ndarray,
    dynamic_crop_gain_patch: jnp.ndarray,
    grain_to_crop_product_patch: jnp.ndarray,
    parameters: CnWoodProductsParameters,
) -> tuple[CnProductSpeciesState, CnProductSpeciesDiagnostics]:
    gridcell_count = state.crop_product_1year_gridcell.shape[0]
    harvest_ten_fraction = parameters.wood_harvest_10year_fraction_pft[
        vegetation_type_patch
    ]
    harvest_hundred_fraction = fortran_add(
        jnp.float64(1.0),
        -harvest_ten_fraction,
    )
    harvest_ten_patch = jnp.where(
        soil_patch,
        fortran_multiply(wood_harvest_patch, harvest_ten_fraction),
        jnp.float64(0.0),
    )
    harvest_hundred_patch = jnp.where(
        soil_patch,
        fortran_multiply(wood_harvest_patch, harvest_hundred_fraction),
        jnp.float64(0.0),
    )
    grain_patch = jnp.where(
        soil_patch,
        grain_to_crop_product_patch,
        jnp.float64(0.0),
    )
    harvest_ten_gridcell = patch_to_gridcell_mean(
        harvest_ten_patch,
        patch_gridcell,
        patch_weight,
        patch_active,
        number_of_gridcells=gridcell_count,
        maximum_patches_per_gridcell=MAXIMUM_PATCHES_PER_GRIDCELL,
    )
    harvest_hundred_gridcell = patch_to_gridcell_mean(
        harvest_hundred_patch,
        patch_gridcell,
        patch_weight,
        patch_active,
        number_of_gridcells=gridcell_count,
        maximum_patches_per_gridcell=MAXIMUM_PATCHES_PER_GRIDCELL,
    )
    grain_gridcell = patch_to_gridcell_mean(
        grain_patch,
        patch_gridcell,
        patch_weight,
        patch_active,
        number_of_gridcells=gridcell_count,
        maximum_patches_per_gridcell=MAXIMUM_PATCHES_PER_GRIDCELL,
    )

    dynamic_ten_gridcell, dynamic_hundred_gridcell = _partition_dynamic_wood(
        dynamic_wood_gain_patch,
        patch_gridcell,
        vegetation_type_patch,
        parameters.dynamic_wood_10year_fraction_pft,
        parameters.dynamic_wood_100year_fraction_pft,
        gridcell_count,
    )
    dynamic_crop_gridcell = _contiguous_patch_to_gridcell_sum(
        dynamic_crop_gain_patch,
        patch_gridcell,
        gridcell_count,
    )

    crop_loss = fortran_multiply(
        state.crop_product_1year_gridcell,
        jnp.float64(_CROP_PRODUCT_1YEAR_DECAY_RATE),
    )
    ten_loss = fortran_multiply(
        state.wood_product_10year_gridcell,
        jnp.float64(_WOOD_PRODUCT_10YEAR_DECAY_RATE),
    )
    hundred_loss = fortran_multiply(
        state.wood_product_100year_gridcell,
        jnp.float64(_WOOD_PRODUCT_100YEAR_DECAY_RATE),
    )
    time_step = jnp.asarray(parameters.time_step, dtype=jnp.float64)

    crop_product = fortran_add(
        state.crop_product_1year_gridcell,
        fortran_multiply(dynamic_crop_gridcell, time_step),
    )
    ten_product = fortran_add(
        state.wood_product_10year_gridcell,
        fortran_multiply(dynamic_ten_gridcell, time_step),
    )
    hundred_product = fortran_add(
        state.wood_product_100year_gridcell,
        fortran_multiply(dynamic_hundred_gridcell, time_step),
    )
    crop_product = fortran_add(
        crop_product,
        fortran_multiply(grain_gridcell, time_step),
    )
    ten_product = fortran_add(
        ten_product,
        fortran_multiply(harvest_ten_gridcell, time_step),
    )
    hundred_product = fortran_add(
        hundred_product,
        fortran_multiply(harvest_hundred_gridcell, time_step),
    )
    crop_product = fortran_add(
        crop_product,
        -fortran_multiply(crop_loss, time_step),
    )
    ten_product = fortran_add(
        ten_product,
        -fortran_multiply(ten_loss, time_step),
    )
    hundred_product = fortran_add(
        hundred_product,
        -fortran_multiply(hundred_loss, time_step),
    )

    total_wood_product = fortran_add(ten_product, hundred_product)
    total_wood_loss = fortran_add(ten_loss, hundred_loss)
    total_product_loss = fortran_add(
        fortran_add(crop_loss, ten_loss),
        hundred_loss,
    )
    total_dynamic_wood_gain = fortran_add(
        dynamic_hundred_gridcell,
        dynamic_ten_gridcell,
    )
    return (
        CnProductSpeciesState(
            crop_product_1year_gridcell=crop_product,
            wood_product_10year_gridcell=ten_product,
            wood_product_100year_gridcell=hundred_product,
        ),
        CnProductSpeciesDiagnostics(
            total_wood_product_gridcell=total_wood_product,
            dwt_wood_product_10year_gain_gridcell=dynamic_ten_gridcell,
            dwt_wood_product_100year_gain_gridcell=dynamic_hundred_gridcell,
            dwt_total_wood_product_gain_gridcell=total_dynamic_wood_gain,
            dwt_crop_product_1year_gain_gridcell=dynamic_crop_gridcell,
            harvest_deadstem_to_product_10year_patch=harvest_ten_patch,
            harvest_deadstem_to_product_10year_gridcell=harvest_ten_gridcell,
            harvest_deadstem_to_product_100year_patch=harvest_hundred_patch,
            harvest_deadstem_to_product_100year_gridcell=(
                harvest_hundred_gridcell
            ),
            grain_to_crop_product_1year_patch=grain_patch,
            grain_to_crop_product_1year_gridcell=grain_gridcell,
            crop_product_1year_loss_gridcell=crop_loss,
            wood_product_10year_loss_gridcell=ten_loss,
            wood_product_100year_loss_gridcell=hundred_loss,
            total_wood_product_loss_gridcell=total_wood_loss,
            total_product_loss_gridcell=total_product_loss,
        ),
    )


def cn_wood_products(
    inputs: CnWoodProductsInputs,
    state: CnWoodProductsState,
    cn_fluxes: CnTransientFluxState,
    parameters: CnWoodProductsParameters,
) -> CnWoodProductsResult:

    _, gridcell_count = _validate_shapes(inputs, state, cn_fluxes, parameters)
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    patch_active = jnp.asarray(inputs.patch_active, dtype=jnp.bool_)
    patch_gridcell = jnp.asarray(inputs.patch_gridcell_index, dtype=jnp.int32)
    patch_weight = jnp.asarray(
        inputs.patch_weight_on_gridcell,
        dtype=jnp.float64,
    )
    vegetation_type_patch = jnp.asarray(
        inputs.vegetation_type_patch,
        dtype=jnp.int32,
    )

    carbon_state, carbon_diagnostics = _update_species(
        state.carbon,
        soil_patch=soil_patch,
        patch_active=patch_active,
        patch_gridcell=patch_gridcell,
        patch_weight=patch_weight,
        vegetation_type_patch=vegetation_type_patch,
        dynamic_wood_gain_patch=jnp.asarray(
            inputs.dwt_wood_product_carbon_gain_patch,
            dtype=jnp.float64,
        ),
        wood_harvest_patch=get_cn_flux_field(
            cn_fluxes,
            _CARBON_WOOD_HARVEST,
        ),
        dynamic_crop_gain_patch=jnp.asarray(
            inputs.dwt_crop_product_carbon_gain_patch,
            dtype=jnp.float64,
        ),
        grain_to_crop_product_patch=get_cn_flux_field(
            cn_fluxes,
            _CARBON_GRAIN_PRODUCT,
        ),
        parameters=parameters,
    )
    nitrogen_state, nitrogen_diagnostics = _update_species(
        state.nitrogen,
        soil_patch=soil_patch,
        patch_active=patch_active,
        patch_gridcell=patch_gridcell,
        patch_weight=patch_weight,
        vegetation_type_patch=vegetation_type_patch,
        dynamic_wood_gain_patch=jnp.asarray(
            inputs.dwt_wood_product_nitrogen_gain_patch,
            dtype=jnp.float64,
        ),
        wood_harvest_patch=get_cn_flux_field(
            cn_fluxes,
            _NITROGEN_WOOD_HARVEST,
        ),
        dynamic_crop_gain_patch=jnp.asarray(
            inputs.dwt_crop_product_nitrogen_gain_patch,
            dtype=jnp.float64,
        ),
        grain_to_crop_product_patch=get_cn_flux_field(
            cn_fluxes,
            _NITROGEN_GRAIN_PRODUCT,
        ),
        parameters=parameters,
    )
    return CnWoodProductsResult(
        state=CnWoodProductsState(
            carbon=carbon_state,
            nitrogen=nitrogen_state,
        ),
        cn_fluxes=cn_fluxes,
        diagnostics=CnWoodProductsDiagnostics(
            carbon=carbon_diagnostics,
            nitrogen=nitrogen_diagnostics,
            updated_gridcell_count=jnp.int32(gridcell_count),
        ),
    )


__all__ = [
    "MAXIMUM_PATCHES_PER_GRIDCELL",
    "CnProductSpeciesDiagnostics",
    "CnProductSpeciesState",
    "CnWoodProductsDiagnostics",
    "CnWoodProductsInputs",
    "CnWoodProductsParameters",
    "CnWoodProductsResult",
    "CnWoodProductsState",
    "cn_wood_products",
]
