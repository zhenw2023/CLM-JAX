
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    get_cn_flux_field,
    set_cn_flux_field,
)
from clm_jax.physics.cn_zero_manifest import NUMBER_OF_DECOMPOSITION_LEVELS
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)

_USE_FUSED_LITTER_GATHER = True


class LitterToColumnInputs(NamedTuple):

    soil_column: jnp.ndarray
    column_patch_indices: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    patch_weight_on_column: jnp.ndarray
    patch_active: jnp.ndarray
    leaf_profile_patch_level: jnp.ndarray
    fine_root_profile_patch_level: jnp.ndarray


class LitterToColumnParameters(NamedTuple):

    leaf_labile_fraction_pft: jnp.ndarray
    leaf_cellulose_fraction_pft: jnp.ndarray
    leaf_lignin_fraction_pft: jnp.ndarray
    fine_root_labile_fraction_pft: jnp.ndarray
    fine_root_cellulose_fraction_pft: jnp.ndarray
    fine_root_lignin_fraction_pft: jnp.ndarray
    first_prognostic_crop_pft: int | jnp.ndarray
    use_grain_product: bool | jnp.ndarray = True
    decomposition_level_count: int | jnp.ndarray = 20


class LitterToColumnDiagnostics(NamedTuple):
    updated_column_count: jnp.ndarray
    active_patch_visit_count: jnp.ndarray
    crop_patch_visit_count: jnp.ndarray


class LitterToColumnResult(NamedTuple):
    cn_fluxes: CnTransientFluxState
    diagnostics: LitterToColumnDiagnostics


_INPUT_CARBON_FIELDS = (
    "vegetation_carbon.leafc_to_litter_patch",
    "vegetation_carbon.frootc_to_litter_patch",
    "vegetation_carbon.livestemc_to_litter_patch",
    "vegetation_carbon.grainc_to_food_patch",
)
_INPUT_NITROGEN_FIELDS = (
    "vegetation_nitrogen.leafn_to_litter_patch",
    "vegetation_nitrogen.frootn_to_litter_patch",
    "vegetation_nitrogen.livestemn_to_litter_patch",
    "vegetation_nitrogen.grainn_to_food_patch",
)
_OUTPUT_CARBON_FIELDS = (
    "vegetation_carbon.phenology_c_to_litr_met_c_col",
    "vegetation_carbon.phenology_c_to_litr_cel_c_col",
    "vegetation_carbon.phenology_c_to_litr_lig_c_col",
)
_OUTPUT_NITROGEN_FIELDS = (
    "vegetation_nitrogen.phenology_n_to_litr_met_n_col",
    "vegetation_nitrogen.phenology_n_to_litr_cel_n_col",
    "vegetation_nitrogen.phenology_n_to_litr_lig_n_col",
)


def _validate_shapes(
    inputs: LitterToColumnInputs,
    cn_fluxes: CnTransientFluxState,
    parameters: LitterToColumnParameters,
) -> tuple[int, int]:
    if jnp.ndim(inputs.soil_column) != 1:
        raise ValueError("soil_column must be a vector")
    column_count = inputs.soil_column.shape[0]
    if (
        jnp.ndim(inputs.column_patch_indices) != 2
        or inputs.column_patch_indices.shape[0] != column_count
    ):
        raise ValueError("column_patch_indices must have shape (column, patch slot)")
    if jnp.ndim(inputs.vegetation_type_patch) != 1:
        raise ValueError("vegetation_type_patch must be a vector")
    patch_count = inputs.vegetation_type_patch.shape[0]
    for name in ("patch_weight_on_column", "patch_active"):
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    for name in (
        "leaf_profile_patch_level",
        "fine_root_profile_patch_level",
    ):
        if jnp.shape(getattr(inputs, name)) != (
            patch_count,
            NUMBER_OF_DECOMPOSITION_LEVELS,
        ):
            raise ValueError(f"{name} must have shape (patch, 25)")

    pft_count = None
    for name in parameters._fields[:6]:
        values = getattr(parameters, name)
        if jnp.ndim(values) != 1 or values.shape[0] < 1:
            raise ValueError(f"{name} must be a nonempty PFT vector")
        if pft_count is None:
            pft_count = values.shape[0]
        elif values.shape[0] != pft_count:
            raise ValueError("all litter-fraction vectors must have equal length")
    for name in parameters._fields[6:]:
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")

    for field_name in (*_INPUT_CARBON_FIELDS, *_INPUT_NITROGEN_FIELDS):
        if get_cn_flux_field(cn_fluxes, field_name).shape != (patch_count,):
            raise ValueError("litter input fluxes must match patch count")
    expected_output_shape = (
        column_count,
        NUMBER_OF_DECOMPOSITION_LEVELS,
    )
    for field_name in (*_OUTPUT_CARBON_FIELDS, *_OUTPUT_NITROGEN_FIELDS):
        if get_cn_flux_field(cn_fluxes, field_name).shape != expected_output_shape:
            raise ValueError("litter output fluxes must match column and level counts")
    return patch_count, column_count


def _multiply_litter_term(
    flux: jnp.ndarray,
    fraction: jnp.ndarray,
    weight: jnp.ndarray,
    profile: jnp.ndarray,
) -> jnp.ndarray:
    term = fortran_multiply(flux[:, None], fraction[:, None])
    term = fortran_multiply(term, weight[:, None])
    return fortran_multiply(term, profile)


def _add_organ(
    carry: tuple[jnp.ndarray, ...],
    carbon_flux: jnp.ndarray,
    nitrogen_flux: jnp.ndarray,
    fractions: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    weight: jnp.ndarray,
    profile: jnp.ndarray,
    update_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, ...]:
    updated = []
    for current, fraction in zip(carry[:3], fractions, strict=True):
        candidate = fortran_add(
            current,
            _multiply_litter_term(carbon_flux, fraction, weight, profile),
        )
        updated.append(jnp.where(update_mask, candidate, current))
    for current, fraction in zip(carry[3:], fractions, strict=True):
        candidate = fortran_add(
            current,
            _multiply_litter_term(nitrogen_flux, fraction, weight, profile),
        )
        updated.append(jnp.where(update_mask, candidate, current))
    return tuple(updated)


def _litter_to_column_reference(
    inputs: LitterToColumnInputs,
    cn_fluxes: CnTransientFluxState,
    parameters: LitterToColumnParameters,
) -> LitterToColumnResult:

    patch_count, _ = _validate_shapes(inputs, cn_fluxes, parameters)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    previous = tuple(
        get_cn_flux_field(cn_fluxes, field_name)
        for field_name in (*_OUTPUT_CARBON_FIELDS, *_OUTPUT_NITROGEN_FIELDS)
    )
    if patch_count == 0 or inputs.column_patch_indices.shape[1] == 0:
        return LitterToColumnResult(
            cn_fluxes=cn_fluxes,
            diagnostics=LitterToColumnDiagnostics(
                updated_column_count=jnp.sum(soil_column, dtype=jnp.int32),
                active_patch_visit_count=jnp.int32(0),
                crop_patch_visit_count=jnp.int32(0),
            ),
        )

    column_patch = jnp.asarray(inputs.column_patch_indices, dtype=jnp.int32)
    vegetation_type_patch = jnp.asarray(
        inputs.vegetation_type_patch,
        dtype=jnp.int32,
    )
    patch_weight = jnp.asarray(
        inputs.patch_weight_on_column,
        dtype=jnp.float64,
    )
    patch_active = jnp.asarray(inputs.patch_active, dtype=jnp.bool_)
    leaf_profile = jnp.asarray(
        inputs.leaf_profile_patch_level,
        dtype=jnp.float64,
    )
    fine_root_profile = jnp.asarray(
        inputs.fine_root_profile_patch_level,
        dtype=jnp.float64,
    )
    carbon_fluxes = tuple(
        get_cn_flux_field(cn_fluxes, field_name)
        for field_name in _INPUT_CARBON_FIELDS
    )
    nitrogen_fluxes = tuple(
        get_cn_flux_field(cn_fluxes, field_name)
        for field_name in _INPUT_NITROGEN_FIELDS
    )
    leaf_fraction_pft = tuple(
        jnp.asarray(getattr(parameters, name), dtype=jnp.float64)
        for name in parameters._fields[:3]
    )
    root_fraction_pft = tuple(
        jnp.asarray(getattr(parameters, name), dtype=jnp.float64)
        for name in parameters._fields[3:6]
    )
    first_crop = jnp.asarray(
        parameters.first_prognostic_crop_pft,
        dtype=jnp.int32,
    )
    grain_to_litter = ~jnp.asarray(
        parameters.use_grain_product,
        dtype=jnp.bool_,
    )
    active_level = jnp.arange(
        NUMBER_OF_DECOMPOSITION_LEVELS,
        dtype=jnp.int32,
    ) < jnp.asarray(parameters.decomposition_level_count, dtype=jnp.int32)

    def accumulate_patch_slot(
        slot: int,
        carry: tuple[jnp.ndarray, ...],
    ) -> tuple[jnp.ndarray, ...]:
        unbounded_patch = column_patch[:, slot]
        valid = unbounded_patch >= jnp.int32(0)
        bounded_patch = jnp.clip(unbounded_patch, 0, patch_count - 1)
        active_patch = soil_column & valid & patch_active[bounded_patch]
        vegetation_type = vegetation_type_patch[bounded_patch]
        crop_patch = active_patch & (vegetation_type >= first_crop)
        common_mask = active_patch[:, None] & active_level[None, :]
        crop_mask = crop_patch[:, None] & active_level[None, :]
        grain_mask = crop_mask & grain_to_litter
        weight = patch_weight[bounded_patch]
        selected_leaf_profile = leaf_profile[bounded_patch]
        selected_root_profile = fine_root_profile[bounded_patch]
        leaf_fractions = tuple(
            values[vegetation_type] for values in leaf_fraction_pft
        )
        root_fractions = tuple(
            values[vegetation_type] for values in root_fraction_pft
        )

        carry = _add_organ(
            carry,
            carbon_fluxes[0][bounded_patch],
            nitrogen_fluxes[0][bounded_patch],
            leaf_fractions,
            weight,
            selected_leaf_profile,
            common_mask,
        )
        carry = _add_organ(
            carry,
            carbon_fluxes[1][bounded_patch],
            nitrogen_fluxes[1][bounded_patch],
            root_fractions,
            weight,
            selected_root_profile,
            common_mask,
        )
        carry = _add_organ(
            carry,
            carbon_fluxes[2][bounded_patch],
            nitrogen_fluxes[2][bounded_patch],
            leaf_fractions,
            weight,
            selected_leaf_profile,
            crop_mask,
        )
        return _add_organ(
            carry,
            carbon_fluxes[3][bounded_patch],
            nitrogen_fluxes[3][bounded_patch],
            leaf_fractions,
            weight,
            selected_leaf_profile,
            grain_mask,
        )

    final = lax.fori_loop(
        0,
        column_patch.shape[1],
        accumulate_patch_slot,
        previous,
    )
    for field_name, values in zip(
        (*_OUTPUT_CARBON_FIELDS, *_OUTPUT_NITROGEN_FIELDS),
        final,
        strict=True,
    ):
        cn_fluxes = set_cn_flux_field(cn_fluxes, field_name, values)

    valid_patch = column_patch >= jnp.int32(0)
    bounded_patch = jnp.clip(column_patch, 0, patch_count - 1)
    active_visit = (
        soil_column[:, None] & valid_patch & patch_active[bounded_patch]
    )
    crop_visit = active_visit & (
        vegetation_type_patch[bounded_patch] >= first_crop
    )
    return LitterToColumnResult(
        cn_fluxes=cn_fluxes,
        diagnostics=LitterToColumnDiagnostics(
            updated_column_count=jnp.sum(soil_column, dtype=jnp.int32),
            active_patch_visit_count=jnp.sum(active_visit, dtype=jnp.int32),
            crop_patch_visit_count=jnp.sum(crop_visit, dtype=jnp.int32),
        ),
    )


def litter_to_column(inputs, cn_fluxes, parameters):
    if _USE_FUSED_LITTER_GATHER and jax.default_backend() == "gpu":
        from clm_jax.physics.litter_to_column_gpu import gather_litter_fluxes
        return gather_litter_fluxes(inputs, cn_fluxes, parameters)
    return _litter_to_column_reference(inputs, cn_fluxes, parameters)


__all__ = [
    "LitterToColumnDiagnostics",
    "LitterToColumnInputs",
    "LitterToColumnParameters",
    "LitterToColumnResult",
    "litter_to_column",
]
