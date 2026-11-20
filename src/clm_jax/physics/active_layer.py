
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
)


class ActiveLayerInputs(NamedTuple):

    soil_column: jnp.ndarray
    column_gridcell_index: jnp.ndarray
    gridcell_latitude_radians: jnp.ndarray
    soil_temperature_column_level: jnp.ndarray
    soil_depth_level: jnp.ndarray
    month: int | jnp.ndarray
    day_of_month: int | jnp.ndarray
    seconds_of_day: int | jnp.ndarray


class ActiveLayerState(NamedTuple):

    thaw_depth_column: jnp.ndarray
    maximum_thaw_depth_column: jnp.ndarray
    prior_year_maximum_thaw_depth_column: jnp.ndarray
    thaw_depth_index_column: jnp.ndarray
    maximum_thaw_depth_index_column: jnp.ndarray
    prior_year_maximum_thaw_depth_index_column: jnp.ndarray


class ActiveLayerParameters(NamedTuple):

    freezing_temperature: float | jnp.ndarray = 273.15
    time_step_seconds: int | jnp.ndarray = 1800


DEFAULT_ACTIVE_LAYER_PARAMETERS = ActiveLayerParameters()


class ActiveLayerDiagnostics(NamedTuple):
    annual_reset_column: jnp.ndarray
    bottom_thawed_column: jnp.ndarray
    interpolated_thaw_front_column: jnp.ndarray
    updated_column_count: jnp.ndarray
    annual_reset_column_count: jnp.ndarray


class ActiveLayerResult(NamedTuple):
    state: ActiveLayerState
    diagnostics: ActiveLayerDiagnostics


def _validate_shapes(inputs: ActiveLayerInputs, state: ActiveLayerState) -> None:
    column_shape = state.thaw_depth_column.shape
    if len(column_shape) != 1:
        raise ValueError("active-layer column state must contain vectors")
    column_fields = (
        inputs.soil_column,
        inputs.column_gridcell_index,
        state.maximum_thaw_depth_column,
        state.prior_year_maximum_thaw_depth_column,
        state.thaw_depth_index_column,
        state.maximum_thaw_depth_index_column,
        state.prior_year_maximum_thaw_depth_index_column,
    )
    if any(field.shape != column_shape for field in column_fields):
        raise ValueError("active-layer column fields must have matching shapes")
    if (
        len(inputs.soil_temperature_column_level.shape) != 2
        or inputs.soil_temperature_column_level.shape[0] != column_shape[0]
    ):
        raise ValueError("soil temperature must be a column-by-level matrix")
    level_count = inputs.soil_temperature_column_level.shape[1]
    if level_count < 2 or inputs.soil_depth_level.shape != (level_count,):
        raise ValueError("soil depth must match at least two temperature levels")
    if len(inputs.gridcell_latitude_radians.shape) != 1:
        raise ValueError("gridcell latitude must be a vector")


def diagnose_active_layer(
    inputs: ActiveLayerInputs,
    state: ActiveLayerState,
    parameters: ActiveLayerParameters = DEFAULT_ACTIVE_LAYER_PARAMETERS,
) -> ActiveLayerResult:

    _validate_shapes(inputs, state)
    soil = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    temperatures = jnp.asarray(
        inputs.soil_temperature_column_level,
        dtype=jnp.float64,
    )
    depths = jnp.asarray(inputs.soil_depth_level, dtype=jnp.float64)
    freezing = jnp.asarray(parameters.freezing_temperature, dtype=jnp.float64)
    level_count = temperatures.shape[1]

    bottom_thawed = temperatures[:, -1] > freezing
    one_based_level = jnp.arange(1, level_count, dtype=jnp.int32)
    deepest_thawed_index = jnp.max(
        jnp.where(
            temperatures[:, :-1] > freezing,
            one_based_level[None, :],
            jnp.int32(0),
        ),
        axis=1,
    )
    safe_upper_index = jnp.maximum(deepest_thawed_index - 1, jnp.int32(0))
    column_index = jnp.arange(temperatures.shape[0], dtype=jnp.int32)
    t1 = temperatures[column_index, safe_upper_index]
    t2 = temperatures[column_index, safe_upper_index + 1]
    z1 = depths[safe_upper_index]
    z2 = depths[safe_upper_index + 1]
    valid_interpolation = (~bottom_thawed) & (deepest_thawed_index > 0)
    safe_temperature_difference = jnp.where(
        valid_interpolation,
        t1 - t2,
        jnp.float64(1.0),
    )
    interpolation_fraction = fortran_divide(
        fortran_multiply(t1 - freezing, z2 - z1),
        safe_temperature_difference,
    )
    interpolated_depth = fortran_add(z1, interpolation_fraction)
    diagnosed_depth = jnp.where(
        bottom_thawed,
        depths[-1],
        jnp.where(valid_interpolation, interpolated_depth, jnp.float64(0.0)),
    )
    diagnosed_index = jnp.where(
        bottom_thawed,
        jnp.int32(level_count),
        deepest_thawed_index,
    )

    time_step = jnp.asarray(parameters.time_step_seconds, dtype=jnp.int32)
    first_completed_step = (
        jnp.asarray(inputs.seconds_of_day, dtype=jnp.int32) // time_step
        == jnp.int32(1)
    )
    northern_turnover = (
        (jnp.asarray(inputs.month, dtype=jnp.int32) == jnp.int32(1))
        & (jnp.asarray(inputs.day_of_month, dtype=jnp.int32) == jnp.int32(1))
        & first_completed_step
    )
    southern_turnover = (
        (jnp.asarray(inputs.month, dtype=jnp.int32) == jnp.int32(7))
        & (jnp.asarray(inputs.day_of_month, dtype=jnp.int32) == jnp.int32(1))
        & first_completed_step
    )
    latitude = jnp.asarray(inputs.gridcell_latitude_radians, dtype=jnp.float64)[
        jnp.asarray(inputs.column_gridcell_index, dtype=jnp.int32)
    ]
    annual_reset = soil & (
        (northern_turnover & (latitude > jnp.float64(0.0)))
        | (southern_turnover & (latitude <= jnp.float64(0.0)))
    )

    reset_maximum_depth = jnp.where(
        annual_reset,
        jnp.float64(0.0),
        state.maximum_thaw_depth_column,
    )
    reset_maximum_index = jnp.where(
        annual_reset,
        jnp.int32(0),
        state.maximum_thaw_depth_index_column,
    )
    update_maximum = soil & (diagnosed_depth > reset_maximum_depth)
    final_maximum_depth = jnp.where(
        update_maximum,
        diagnosed_depth,
        reset_maximum_depth,
    )
    final_maximum_index = jnp.where(
        update_maximum,
        diagnosed_index,
        reset_maximum_index,
    )
    final_state = ActiveLayerState(
        thaw_depth_column=jnp.where(
            soil,
            diagnosed_depth,
            state.thaw_depth_column,
        ),
        maximum_thaw_depth_column=jnp.where(
            soil,
            final_maximum_depth,
            state.maximum_thaw_depth_column,
        ),
        prior_year_maximum_thaw_depth_column=jnp.where(
            annual_reset,
            state.maximum_thaw_depth_column,
            state.prior_year_maximum_thaw_depth_column,
        ),
        thaw_depth_index_column=jnp.where(
            soil,
            diagnosed_index,
            state.thaw_depth_index_column,
        ),
        maximum_thaw_depth_index_column=jnp.where(
            soil,
            final_maximum_index,
            state.maximum_thaw_depth_index_column,
        ),
        prior_year_maximum_thaw_depth_index_column=jnp.where(
            annual_reset,
            state.maximum_thaw_depth_index_column,
            state.prior_year_maximum_thaw_depth_index_column,
        ),
    )
    return ActiveLayerResult(
        state=final_state,
        diagnostics=ActiveLayerDiagnostics(
            annual_reset_column=annual_reset,
            bottom_thawed_column=soil & bottom_thawed,
            interpolated_thaw_front_column=soil & valid_interpolation,
            updated_column_count=jnp.sum(soil, dtype=jnp.int32),
            annual_reset_column_count=jnp.sum(annual_reset, dtype=jnp.int32),
        ),
    )
