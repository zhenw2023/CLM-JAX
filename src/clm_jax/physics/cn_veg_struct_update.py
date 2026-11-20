
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from clm_jax.constants import PI
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
    strict_exp,
)
from clm_jax.physics.native_power import native_power


class CnVegStructUpdateInputs(NamedTuple):

    soil_patch: jnp.ndarray
    patch_column_index: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    snow_depth_column: jnp.ndarray
    wind_forcing_height_patch: jnp.ndarray
    leaf_carbon_patch: jnp.ndarray
    dead_stem_carbon_patch: jnp.ndarray
    burned_area_fraction_column: jnp.ndarray
    harvest_date_patch: jnp.ndarray


class CnVegStructUpdateState(NamedTuple):

    maximum_crop_height_patch: jnp.ndarray
    peak_leaf_area_index_patch: jnp.ndarray
    total_leaf_area_index_patch: jnp.ndarray
    total_stem_area_index_patch: jnp.ndarray
    canopy_top_height_patch: jnp.ndarray
    canopy_bottom_height_patch: jnp.ndarray
    exposed_leaf_area_index_patch: jnp.ndarray
    exposed_stem_area_index_patch: jnp.ndarray
    fraction_vegetation_nosnow_albedo_patch: jnp.ndarray


class CnVegStructUpdateParameters(NamedTuple):

    woody_pft: jnp.ndarray
    specific_leaf_area_top_pft: jnp.ndarray
    specific_leaf_area_slope_pft: jnp.ndarray
    roughness_to_height_ratio_pft: jnp.ndarray
    displacement_to_height_ratio_pft: jnp.ndarray
    wood_density_pft: jnp.ndarray
    maximum_crop_height_pft: jnp.ndarray
    maximum_leaf_area_index_pft: jnp.ndarray
    radiation_step_seconds: float | jnp.ndarray = 1800.0
    spinup_state: int | jnp.ndarray = 0
    bare_ground_pft: int = 0
    generic_c3_crop_pft: int = 15
    generic_irrigated_c3_crop_pft: int = 16
    first_shrub_pft: int = 9
    last_shrub_pft: int = 11
    first_prognostic_crop_pft: int = 17
    temperate_corn_pft: int = 17
    irrigated_temperate_corn_pft: int = 18
    sugarcane_pft: int = 67
    irrigated_sugarcane_pft: int = 68
    tropical_corn_pft: int = 75
    irrigated_tropical_corn_pft: int = 76


class CnVegStructUpdateDiagnostics(NamedTuple):
    updated_patch_count: jnp.ndarray
    bare_patch_count: jnp.ndarray
    woody_patch_count: jnp.ndarray
    prognostic_crop_patch_count: jnp.ndarray
    short_vegetation_patch_count: jnp.ndarray
    harvested_stubble_patch_count: jnp.ndarray


class CnVegStructUpdateResult(NamedTuple):
    state: CnVegStructUpdateState
    diagnostics: CnVegStructUpdateDiagnostics


def _mul(*values: jnp.ndarray) -> jnp.ndarray:
    result = jnp.asarray(values[0], dtype=jnp.float64)
    for value in values[1:]:
        result = fortran_multiply(result, value)
    return result


def _sub(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return fortran_add(left, -jnp.asarray(right, dtype=jnp.float64))


@jax.custom_jvp
def _clamp_woody_height(height, wind_height_limit):
    return jnp.maximum(jnp.minimum(height, wind_height_limit), jnp.float64(0.01))


@_clamp_woody_height.defjvp
def _clamp_woody_height_jvp(primals, tangents):
    from clm_jax.physics.adjoint_scaling import sensitivity_product

    height, limit = primals
    result = _clamp_woody_height(height, limit)
    inner = jnp.minimum(height, limit)
    floor_weight = (inner == result) / (1. + (inner == .01))
    denominator = 1. + (height == limit)
    height_weight = floor_weight * ((height == inner) / denominator)
    limit_weight = floor_weight * ((limit == inner) / denominator)
    derivative = (sensitivity_product(height_weight, tangents[0])
                  + sensitivity_product(limit_weight, tangents[1]))
    return result, derivative


def _validate_shapes(
    inputs: CnVegStructUpdateInputs,
    state: CnVegStructUpdateState,
    parameters: CnVegStructUpdateParameters,
) -> tuple[int, int]:
    patch_count = inputs.soil_patch.shape[0]
    column_count = inputs.snow_depth_column.shape[0]
    if inputs.soil_patch.shape != (patch_count,):
        raise ValueError("soil_patch must be a vector")
    for name in (
        "patch_column_index",
        "vegetation_type_patch",
        "wind_forcing_height_patch",
        "leaf_carbon_patch",
        "dead_stem_carbon_patch",
        "harvest_date_patch",
    ):
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    if inputs.burned_area_fraction_column.shape != (column_count,):
        raise ValueError("burned_area_fraction_column must contain every column")
    for name in state._fields:
        if jnp.shape(getattr(state, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")

    pft_count = parameters.woody_pft.shape[0]
    for name in (
        "specific_leaf_area_top_pft",
        "specific_leaf_area_slope_pft",
        "roughness_to_height_ratio_pft",
        "displacement_to_height_ratio_pft",
        "wood_density_pft",
        "maximum_crop_height_pft",
        "maximum_leaf_area_index_pft",
    ):
        if jnp.shape(getattr(parameters, name)) != (pft_count,):
            raise ValueError(f"{name} must contain every PFT")
    return patch_count, column_count


def cn_veg_struct_update(
    inputs: CnVegStructUpdateInputs,
    state: CnVegStructUpdateState,
    parameters: CnVegStructUpdateParameters,
) -> CnVegStructUpdateResult:

    _validate_shapes(inputs, state, parameters)
    active = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    vegetation_type = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    column = jnp.asarray(inputs.patch_column_index, dtype=jnp.int32)

    woody = jnp.asarray(parameters.woody_pft, dtype=jnp.float64)[vegetation_type]
    slatop = jnp.asarray(
        parameters.specific_leaf_area_top_pft,
        dtype=jnp.float64,
    )[vegetation_type]
    dsladlai = jnp.asarray(
        parameters.specific_leaf_area_slope_pft,
        dtype=jnp.float64,
    )[vegetation_type]
    z0mr = jnp.asarray(
        parameters.roughness_to_height_ratio_pft,
        dtype=jnp.float64,
    )[vegetation_type]
    displar = jnp.asarray(
        parameters.displacement_to_height_ratio_pft,
        dtype=jnp.float64,
    )[vegetation_type]
    wood_density = jnp.asarray(parameters.wood_density_pft, dtype=jnp.float64)[
        vegetation_type
    ]
    maximum_crop_height = jnp.asarray(
        parameters.maximum_crop_height_pft,
        dtype=jnp.float64,
    )[vegetation_type]
    maximum_lai = jnp.asarray(
        parameters.maximum_leaf_area_index_pft,
        dtype=jnp.float64,
    )[vegetation_type]

    bare = vegetation_type == jnp.int32(parameters.bare_ground_pft)
    vegetated = ~bare
    woody_patch = vegetated & (woody == jnp.float64(1.0))
    prognostic_crop = vegetated & ~woody_patch & (
        vegetation_type >= jnp.int32(parameters.first_prognostic_crop_pft)
    )
    short_vegetation = vegetated & ~woody_patch & ~prognostic_crop

    old_tlai = jnp.asarray(state.total_leaf_area_index_patch, dtype=jnp.float64)
    old_tsai = jnp.asarray(state.total_stem_area_index_patch, dtype=jnp.float64)
    leaf_carbon = jnp.asarray(inputs.leaf_carbon_patch, dtype=jnp.float64)

    exponential_lai = fortran_divide(
        _mul(
            slatop,
            _sub(strict_exp(_mul(leaf_carbon, dsladlai)), jnp.float64(1.0)),
        ),
        jnp.where(dsladlai > 0.0, dsladlai, jnp.float64(1.0)),
    )
    linear_lai = _mul(slatop, leaf_carbon)
    diagnosed_lai = jnp.maximum(
        jnp.float64(0.0),
        jnp.where(dsladlai > 0.0, exponential_lai, linear_lai),
    )
    tlai = jnp.where(vegetated, diagnosed_lai, jnp.float64(0.0))

    generic_crop = (
        vegetation_type == jnp.int32(parameters.generic_c3_crop_pft)
    ) | (
        vegetation_type == jnp.int32(parameters.generic_irrigated_c3_crop_pft)
    )
    dt = jnp.asarray(parameters.radiation_step_seconds, dtype=jnp.float64)
    seconds_per_month = jnp.float64(2592000.0)
    crop_alpha = _sub(
        jnp.float64(1.0),
        fortran_divide(dt, seconds_per_month),
    )
    noncrop_alpha = _sub(
        jnp.float64(1.0),
        fortran_divide(_mul(jnp.float64(0.5), dt), seconds_per_month),
    )
    tsai_alpha = jnp.where(generic_crop, crop_alpha, noncrop_alpha)
    tsai_minimum = jnp.where(generic_crop, jnp.float64(0.1), jnp.float64(1.0))
    tsai_minimum = _mul(tsai_minimum, jnp.float64(0.5))
    zeng_tsai = jnp.maximum(
        fortran_add(
            _mul(tsai_alpha, old_tsai),
            jnp.maximum(_sub(old_tlai, tlai), jnp.float64(0.0)),
        ),
        tsai_minimum,
    )
    tsai = jnp.where(vegetated, zeng_tsai, jnp.float64(0.0))

    shrub = (
        vegetation_type >= jnp.int32(parameters.first_shrub_pft)
    ) & (
        vegetation_type <= jnp.int32(parameters.last_shrub_pft)
    )
    taper = jnp.where(shrub, jnp.float64(10.0), jnp.float64(200.0))
    dead_stem_carbon = jnp.asarray(inputs.dead_stem_carbon_patch, dtype=jnp.float64)
    accelerated_multiplier = jnp.where(
        jnp.asarray(parameters.spinup_state, dtype=jnp.int32) == 2,
        jnp.float64(10.0),
        jnp.float64(1.0),
    )
    woody_numerator = _mul(
        jnp.float64(3.0),
        dead_stem_carbon,
        accelerated_multiplier,
        taper,
        taper,
    )
    safe_wood_density = jnp.where(woody_patch, wood_density, jnp.float64(1.0))
    woody_denominator = _mul(
        jnp.float64(PI),
        jnp.float64(0.1),
        safe_wood_density,
    )
    woody_base = fortran_divide(
        jnp.where(woody_patch, woody_numerator, jnp.float64(1.0)),
        woody_denominator,
    )
    woody_height = native_power(
        jnp.maximum(woody_base, jnp.float64(0.0)),
        fortran_divide(jnp.float64(1.0), jnp.float64(3.0)),
    )

    wind_denominator = fortran_add(displar, z0mr)
    safe_wind_denominator = jnp.where(
        vegetated,
        wind_denominator,
        jnp.float64(1.0),
    )
    wind_height_limit = _sub(
        fortran_divide(
            jnp.asarray(inputs.wind_forcing_height_patch, dtype=jnp.float64),
            safe_wind_denominator,
        ),
        jnp.float64(3.0),
    )
    woody_height = _clamp_woody_height(woody_height, wind_height_limit)
    woody_bottom = jnp.maximum(
        jnp.float64(0.0),
        jnp.minimum(jnp.float64(3.0), _sub(woody_height, jnp.float64(1.0))),
    )

    peak_lai = jnp.asarray(state.peak_leaf_area_index_patch, dtype=jnp.int32)
    peak_lai = jnp.where(
        prognostic_crop & (tlai >= maximum_lai),
        jnp.int32(1),
        peak_lai,
    )
    corn_or_sugar = (
        (vegetation_type == jnp.int32(parameters.temperate_corn_pft))
        | (vegetation_type == jnp.int32(parameters.irrigated_temperate_corn_pft))
        | (vegetation_type == jnp.int32(parameters.sugarcane_pft))
        | (vegetation_type == jnp.int32(parameters.irrigated_sugarcane_pft))
        | (vegetation_type == jnp.int32(parameters.tropical_corn_pft))
        | (vegetation_type == jnp.int32(parameters.irrigated_tropical_corn_pft))
    )
    crop_tsai = _mul(
        jnp.where(corn_or_sugar, jnp.float64(0.1), jnp.float64(0.2)),
        tlai,
    )
    harvested_stubble = prognostic_crop & (
        jnp.asarray(inputs.harvest_date_patch, dtype=jnp.int32) < 999
    ) & (tlai == jnp.float64(0.0))
    stubble_tsai = _mul(
        jnp.float64(0.25),
        _sub(
            jnp.float64(1.0),
            _mul(
                jnp.asarray(inputs.burned_area_fraction_column, dtype=jnp.float64)[
                    column
                ],
                jnp.float64(0.90),
            ),
        ),
    )
    crop_tsai = jnp.where(harvested_stubble, stubble_tsai, crop_tsai)
    tsai = jnp.where(prognostic_crop, crop_tsai, tsai)

    htmx = jnp.asarray(state.maximum_crop_height_patch, dtype=jnp.float64)
    htmx = jnp.where(harvested_stubble, jnp.float64(0.0), htmx)
    peak_lai = jnp.where(harvested_stubble, jnp.int32(0), peak_lai)
    safe_crop_denominator = jnp.where(
        prognostic_crop,
        _sub(maximum_lai, jnp.float64(1.0)),
        jnp.float64(1.0),
    )
    crop_height_ratio = jnp.minimum(
        fortran_divide(tlai, safe_crop_denominator),
        jnp.float64(1.0),
    )
    crop_height = _mul(
        maximum_crop_height,
        _mul(crop_height_ratio, crop_height_ratio),
    )
    updated_htmx = jnp.maximum(htmx, crop_height)
    crop_height = jnp.maximum(
        jnp.float64(0.05),
        jnp.maximum(updated_htmx, crop_height),
    )
    htmx = jnp.where(prognostic_crop, updated_htmx, htmx)

    short_height = jnp.maximum(jnp.float64(0.25), _mul(tlai, jnp.float64(0.25)))
    short_height = jnp.maximum(
        jnp.minimum(short_height, wind_height_limit),
        jnp.float64(0.01),
    )
    short_bottom = jnp.maximum(
        jnp.float64(0.0),
        jnp.minimum(jnp.float64(0.05), _sub(short_height, jnp.float64(0.20))),
    )

    htop = jnp.where(
        woody_patch,
        woody_height,
        jnp.where(
            prognostic_crop,
            crop_height,
            jnp.where(short_vegetation, short_height, jnp.float64(0.0)),
        ),
    )
    hbot = jnp.where(
        woody_patch,
        woody_bottom,
        jnp.where(
            prognostic_crop,
            jnp.float64(0.02),
            jnp.where(short_vegetation, short_bottom, jnp.float64(0.0)),
        ),
    )

    snow_depth = jnp.asarray(inputs.snow_depth_column, dtype=jnp.float64)[column]
    tree_or_shrub = vegetated & (
        vegetation_type <= jnp.int32(parameters.last_shrub_pft)
    )
    canopy_thickness = _sub(htop, hbot)
    overlap = jnp.minimum(
        jnp.maximum(_sub(snow_depth, hbot), jnp.float64(0.0)),
        canopy_thickness,
    )
    woody_exposed_fraction = _sub(
        jnp.float64(1.0),
        fortran_divide(
            overlap,
            jnp.maximum(jnp.float64(1.0e-6), canopy_thickness),
        ),
    )
    short_exposed_fraction = _sub(
        jnp.float64(1.0),
        fortran_divide(
            jnp.maximum(
                jnp.minimum(snow_depth, jnp.float64(0.2)),
                jnp.float64(0.0),
            ),
            jnp.float64(0.2),
        ),
    )
    exposed_fraction = jnp.where(
        tree_or_shrub,
        woody_exposed_fraction,
        short_exposed_fraction,
    )
    elai = jnp.maximum(_mul(tlai, exposed_fraction), jnp.float64(0.0))
    esai = jnp.maximum(_mul(tsai, exposed_fraction), jnp.float64(0.0))
    fraction_nosnow = jnp.where(
        fortran_add(elai, esai) > jnp.float64(0.0),
        jnp.int32(1),
        jnp.int32(0),
    )

    def keep_active(updated: jnp.ndarray, previous: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(active, updated, previous)

    result_state = CnVegStructUpdateState(
        maximum_crop_height_patch=keep_active(
            htmx,
            state.maximum_crop_height_patch,
        ),
        peak_leaf_area_index_patch=keep_active(
            peak_lai,
            state.peak_leaf_area_index_patch,
        ),
        total_leaf_area_index_patch=keep_active(
            tlai,
            state.total_leaf_area_index_patch,
        ),
        total_stem_area_index_patch=keep_active(
            tsai,
            state.total_stem_area_index_patch,
        ),
        canopy_top_height_patch=keep_active(htop, state.canopy_top_height_patch),
        canopy_bottom_height_patch=keep_active(
            hbot,
            state.canopy_bottom_height_patch,
        ),
        exposed_leaf_area_index_patch=keep_active(
            elai,
            state.exposed_leaf_area_index_patch,
        ),
        exposed_stem_area_index_patch=keep_active(
            esai,
            state.exposed_stem_area_index_patch,
        ),
        fraction_vegetation_nosnow_albedo_patch=keep_active(
            fraction_nosnow,
            state.fraction_vegetation_nosnow_albedo_patch,
        ),
    )
    return CnVegStructUpdateResult(
        state=result_state,
        diagnostics=CnVegStructUpdateDiagnostics(
            updated_patch_count=jnp.sum(active, dtype=jnp.int32),
            bare_patch_count=jnp.sum(active & bare, dtype=jnp.int32),
            woody_patch_count=jnp.sum(active & woody_patch, dtype=jnp.int32),
            prognostic_crop_patch_count=jnp.sum(
                active & prognostic_crop,
                dtype=jnp.int32,
            ),
            short_vegetation_patch_count=jnp.sum(
                active & short_vegetation,
                dtype=jnp.int32,
            ),
            harvested_stubble_patch_count=jnp.sum(
                active & harvested_stubble,
                dtype=jnp.int32,
            ),
        ),
    )


__all__ = [
    "CnVegStructUpdateDiagnostics",
    "CnVegStructUpdateInputs",
    "CnVegStructUpdateParameters",
    "CnVegStructUpdateResult",
    "CnVegStructUpdateState",
    "cn_veg_struct_update",
]
