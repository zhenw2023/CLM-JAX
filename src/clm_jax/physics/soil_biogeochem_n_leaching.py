
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.physics.cn_zero_manifest import NUMBER_OF_DECOMPOSITION_LEVELS
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_masked_materialized_divide,
    fortran_multiply,
    fortran_ordered_sum,
)


class SoilBiogeochemNLeachingInputs(NamedTuple):

    soil_column: jnp.ndarray
    liquid_water_column_level: jnp.ndarray
    drainage_flux_column: jnp.ndarray
    surface_runoff_flux_column: jnp.ndarray
    nitrate_column_level: jnp.ndarray
    layer_thickness_column_level: jnp.ndarray
    soil_interface_depth: jnp.ndarray


class SoilBiogeochemNLeachingFluxState(NamedTuple):

    nitrate_leaching_column_level: jnp.ndarray
    nitrate_runoff_column_level: jnp.ndarray


class SoilBiogeochemNLeachingParameters(NamedTuple):

    time_step: float | jnp.ndarray = 1800.0
    soluble_nitrate_fraction: float | jnp.ndarray = 1.0
    soil_level_count: int | jnp.ndarray = 20
    decomposition_level_count: int | jnp.ndarray = 20
    runoff_mixing_depth: float | jnp.ndarray = 0.05000000074505806


DEFAULT_SOIL_BIOGEOCHEM_N_LEACHING_PARAMETERS = SoilBiogeochemNLeachingParameters()


class SoilBiogeochemNLeachingDiagnostics(NamedTuple):
    total_liquid_water_column: jnp.ndarray
    surface_liquid_water_column: jnp.ndarray
    updated_column_count: jnp.ndarray
    updated_column_level_count: jnp.ndarray


class SoilBiogeochemNLeachingResult(NamedTuple):
    flux_state: SoilBiogeochemNLeachingFluxState
    diagnostics: SoilBiogeochemNLeachingDiagnostics


def _validate_shapes(
    inputs: SoilBiogeochemNLeachingInputs,
    state: SoilBiogeochemNLeachingFluxState,
    parameters: SoilBiogeochemNLeachingParameters,
) -> tuple[int, int]:
    if jnp.ndim(inputs.soil_column) != 1:
        raise ValueError("soil_column must be a vector")
    column_count = inputs.soil_column.shape[0]
    level_shape = jnp.shape(inputs.liquid_water_column_level)
    if len(level_shape) != 2 or level_shape[0] != column_count:
        raise ValueError("liquid water must have shape (column, ground_level)")
    ground_level_count = level_shape[1]
    for name in ("nitrate_column_level", "layer_thickness_column_level"):
        if jnp.shape(getattr(inputs, name)) != level_shape:
            raise ValueError(f"{name} must match the liquid-water shape")
    for name in ("drainage_flux_column", "surface_runoff_flux_column"):
        if jnp.shape(getattr(inputs, name)) != (column_count,):
            raise ValueError(f"{name} must be a column vector")
    if jnp.shape(inputs.soil_interface_depth) != (ground_level_count + 1,):
        raise ValueError("soil_interface_depth must contain the surface and every lower interface")
    for name in state._fields:
        if jnp.shape(getattr(state, name)) != level_shape:
            raise ValueError(f"{name} must match the nitrate shape")
    for name in parameters._fields:
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    if ground_level_count < NUMBER_OF_DECOMPOSITION_LEVELS:
        raise ValueError("the configured kernel requires all 25 decomposition levels")
    return column_count, ground_level_count


def _ordered_level_sum(values: jnp.ndarray, level_count: jnp.ndarray) -> jnp.ndarray:
    level = jnp.arange(values.shape[1], dtype=jnp.int32)
    terms = jnp.where(level[None, :] < level_count, values, jnp.float64(0.0))
    return fortran_ordered_sum(
        jnp.swapaxes(terms, 0, 1),
        initial=jnp.zeros(values.shape[0], dtype=jnp.float64),
    )


def soil_biogeochem_n_leaching(
    inputs: SoilBiogeochemNLeachingInputs,
    state: SoilBiogeochemNLeachingFluxState,
    parameters: SoilBiogeochemNLeachingParameters = (
        DEFAULT_SOIL_BIOGEOCHEM_N_LEACHING_PARAMETERS
    ),
) -> SoilBiogeochemNLeachingResult:

    column_count, ground_level_count = _validate_shapes(inputs, state, parameters)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    liquid = jnp.asarray(inputs.liquid_water_column_level, dtype=jnp.float64)
    drainage = jnp.asarray(inputs.drainage_flux_column, dtype=jnp.float64)
    surface_runoff = jnp.asarray(inputs.surface_runoff_flux_column, dtype=jnp.float64)
    nitrate = jnp.asarray(inputs.nitrate_column_level, dtype=jnp.float64)
    thickness = jnp.asarray(inputs.layer_thickness_column_level, dtype=jnp.float64)
    interface = jnp.asarray(inputs.soil_interface_depth, dtype=jnp.float64)
    time_step = jnp.asarray(parameters.time_step, dtype=jnp.float64)
    soluble_fraction = jnp.asarray(parameters.soluble_nitrate_fraction, dtype=jnp.float64)
    soil_level_count = jnp.asarray(parameters.soil_level_count, dtype=jnp.int32)
    decomposition_level_count = jnp.asarray(
        parameters.decomposition_level_count, dtype=jnp.int32
    )
    mixing_depth = jnp.asarray(parameters.runoff_mixing_depth, dtype=jnp.float64)

    level = jnp.arange(ground_level_count, dtype=jnp.int32)
    total_water = _ordered_level_sum(liquid, soil_level_count)

    lower_interface = interface[1:]
    upper_interface = interface[:-1]
    full_mixing_layer = lower_interface <= mixing_depth
    partial_mixing_layer = (~full_mixing_layer) & (upper_interface < mixing_depth)
    partial_depth = fortran_add(mixing_depth, -upper_interface)
    partial_fraction = fortran_masked_materialized_divide(
        partial_depth[None, :],
        thickness,
        partial_mixing_layer[None, :],
        jnp.zeros_like(thickness),
    )
    surface_contribution = jnp.where(
        full_mixing_layer[None, :],
        liquid,
        jnp.where(
            partial_mixing_layer[None, :],
            fortran_multiply(liquid, partial_fraction),
            jnp.float64(0.0),
        ),
    )
    surface_water = _ordered_level_sum(surface_contribution, soil_level_count)

    active = soil_column[:, None] & (level[None, :] < decomposition_level_count)
    has_layer_water = active & (liquid > jnp.float64(0.0))

    dissolved_numerator = fortran_multiply(soluble_fraction, nitrate)
    dissolved_numerator = fortran_multiply(dissolved_numerator, thickness)
    dissolved_concentration = fortran_masked_materialized_divide(
        dissolved_numerator,
        liquid,
        has_layer_water,
        jnp.zeros_like(liquid),
    )

    leaching = fortran_multiply(dissolved_concentration, drainage[:, None])
    leaching = fortran_multiply(leaching, liquid)
    leaching_denominator = fortran_multiply(total_water[:, None], thickness)
    leaching = fortran_masked_materialized_divide(
        leaching,
        leaching_denominator,
        active & (leaching_denominator != jnp.float64(0.0)),
        jnp.zeros_like(leaching),
    )
    pool_per_step = fortran_divide(nitrate, time_step)
    leaching = jnp.maximum(jnp.minimum(leaching, pool_per_step), jnp.float64(0.0))

    runoff_numerator = fortran_multiply(dissolved_concentration, surface_runoff[:, None])
    runoff_numerator = fortran_multiply(runoff_numerator, liquid)
    full_denominator = fortran_multiply(surface_water[:, None], thickness)
    full_runoff = fortran_masked_materialized_divide(
        runoff_numerator,
        full_denominator,
        active & full_mixing_layer[None, :] & (full_denominator != jnp.float64(0.0)),
        jnp.zeros_like(runoff_numerator),
    )

    partial_numerator = fortran_multiply(runoff_numerator, partial_fraction)
    partial_denominator = fortran_multiply(surface_water[:, None], partial_depth[None, :])
    partial_runoff = fortran_masked_materialized_divide(
        partial_numerator,
        partial_denominator,
        active
        & partial_mixing_layer[None, :]
        & (partial_denominator != jnp.float64(0.0)),
        jnp.zeros_like(partial_numerator),
    )
    runoff = jnp.where(
        full_mixing_layer[None, :],
        full_runoff,
        jnp.where(partial_mixing_layer[None, :], partial_runoff, jnp.float64(0.0)),
    )
    remaining_pool_flux = fortran_add(pool_per_step, -leaching)
    runoff = jnp.maximum(jnp.minimum(runoff, remaining_pool_flux), jnp.float64(0.0))

    soluble_pool_flux = fortran_divide(
        fortran_multiply(soluble_fraction, nitrate),
        time_step,
    )
    leaching = jnp.maximum(jnp.minimum(leaching, soluble_pool_flux), jnp.float64(0.0))

    previous_leaching = jnp.asarray(state.nitrate_leaching_column_level, dtype=jnp.float64)
    previous_runoff = jnp.asarray(state.nitrate_runoff_column_level, dtype=jnp.float64)
    leaching = jnp.where(active, leaching, previous_leaching)
    runoff = jnp.where(active, runoff, previous_runoff)
    return SoilBiogeochemNLeachingResult(
        flux_state=SoilBiogeochemNLeachingFluxState(
            nitrate_leaching_column_level=leaching,
            nitrate_runoff_column_level=runoff,
        ),
        diagnostics=SoilBiogeochemNLeachingDiagnostics(
            total_liquid_water_column=jnp.where(
                soil_column, total_water, jnp.float64(0.0)
            ),
            surface_liquid_water_column=jnp.where(
                soil_column, surface_water, jnp.float64(0.0)
            ),
            updated_column_count=jnp.sum(soil_column, dtype=jnp.int32),
            updated_column_level_count=jnp.sum(active, dtype=jnp.int32),
        ),
    )


__all__ = [
    "DEFAULT_SOIL_BIOGEOCHEM_N_LEACHING_PARAMETERS",
    "SoilBiogeochemNLeachingDiagnostics",
    "SoilBiogeochemNLeachingFluxState",
    "SoilBiogeochemNLeachingInputs",
    "SoilBiogeochemNLeachingParameters",
    "SoilBiogeochemNLeachingResult",
    "soil_biogeochem_n_leaching",
]
