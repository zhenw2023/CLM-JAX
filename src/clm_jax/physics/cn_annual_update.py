
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.subgrid import patch_to_column_average


class CnAnnualInputs(NamedTuple):

    soil_patch: jnp.ndarray
    soil_column: jnp.ndarray
    patch_column_index: jnp.ndarray
    patch_weight_on_column: jnp.ndarray
    patch_active: jnp.ndarray


class CnAnnualState(NamedTuple):

    annual_sum_counter_column: jnp.ndarray
    annual_average_air_temperature_column: jnp.ndarray
    annual_sum_npp_column: jnp.ndarray
    temporary_sum_potential_gpp_patch: jnp.ndarray
    annual_sum_potential_gpp_patch: jnp.ndarray
    temporary_maximum_retranslocated_nitrogen_patch: jnp.ndarray
    annual_maximum_retranslocated_nitrogen_patch: jnp.ndarray
    temporary_average_air_temperature_patch: jnp.ndarray
    annual_average_air_temperature_patch: jnp.ndarray
    temporary_sum_npp_patch: jnp.ndarray
    annual_sum_npp_patch: jnp.ndarray
    temporary_sum_litterfall_patch: jnp.ndarray
    annual_sum_litterfall_patch: jnp.ndarray


class CnAnnualParameters(NamedTuple):

    time_step: float | jnp.ndarray = 1800.0
    days_per_year: int | jnp.ndarray = 365
    seconds_per_day: float | jnp.ndarray = 86400.0
    maximum_patches_per_column: int | jnp.ndarray = 13


DEFAULT_CN_ANNUAL_PARAMETERS = CnAnnualParameters()


class CnAnnualDiagnostics(NamedTuple):
    end_of_year_column: jnp.ndarray
    turned_over_column_count: jnp.ndarray
    turned_over_patch_count: jnp.ndarray


class CnAnnualResult(NamedTuple):
    state: CnAnnualState
    diagnostics: CnAnnualDiagnostics


def _validate_shapes(inputs: CnAnnualInputs, state: CnAnnualState) -> None:
    column_shape = state.annual_sum_counter_column.shape
    if len(column_shape) != 1:
        raise ValueError("annual column fields must be vectors")
    if not (
        inputs.soil_column.shape
        == state.annual_average_air_temperature_column.shape
        == state.annual_sum_npp_column.shape
        == column_shape
    ):
        raise ValueError("annual column fields and soil-column filter must match")

    patch_shape = state.temporary_sum_potential_gpp_patch.shape
    patch_fields = (
        inputs.soil_patch,
        inputs.patch_column_index,
        inputs.patch_weight_on_column,
        inputs.patch_active,
        state.annual_sum_potential_gpp_patch,
        state.temporary_maximum_retranslocated_nitrogen_patch,
        state.annual_maximum_retranslocated_nitrogen_patch,
        state.temporary_average_air_temperature_patch,
        state.annual_average_air_temperature_patch,
        state.temporary_sum_npp_patch,
        state.annual_sum_npp_patch,
        state.temporary_sum_litterfall_patch,
        state.annual_sum_litterfall_patch,
    )
    if len(patch_shape) != 1 or any(field.shape != patch_shape for field in patch_fields):
        raise ValueError("annual patch fields and patch topology must be matching vectors")


def cn_annual_update(
    inputs: CnAnnualInputs,
    state: CnAnnualState,
    parameters: CnAnnualParameters = DEFAULT_CN_ANNUAL_PARAMETERS,
) -> CnAnnualResult:

    _validate_shapes(inputs, state)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    patch_column = jnp.asarray(inputs.patch_column_index, dtype=jnp.int32)
    patch_active = jnp.asarray(inputs.patch_active, dtype=jnp.bool_)
    patch_weight = jnp.asarray(inputs.patch_weight_on_column, dtype=jnp.float64)
    time_step = jnp.asarray(parameters.time_step, dtype=jnp.float64)
    seconds_per_year = fortran_multiply(
        jnp.asarray(parameters.days_per_year, dtype=jnp.float64),
        jnp.asarray(parameters.seconds_per_day, dtype=jnp.float64),
    )

    counter_candidate = fortran_add(
        jnp.asarray(state.annual_sum_counter_column, dtype=jnp.float64),
        time_step,
    )
    end_of_year = soil_column & (counter_candidate >= seconds_per_year)
    annual_sum_counter_column = jnp.where(
        end_of_year,
        jnp.float64(0.0),
        jnp.where(
            soil_column,
            counter_candidate,
            state.annual_sum_counter_column,
        ),
    )

    patch_end_of_year = soil_patch & end_of_year[patch_column]

    temporary_sum_potential_gpp_patch = jnp.where(
        patch_end_of_year,
        jnp.float64(0.0),
        state.temporary_sum_potential_gpp_patch,
    )
    annual_sum_potential_gpp_patch = jnp.where(
        patch_end_of_year,
        state.temporary_sum_potential_gpp_patch,
        state.annual_sum_potential_gpp_patch,
    )
    temporary_maximum_retranslocated_nitrogen_patch = jnp.where(
        patch_end_of_year,
        jnp.float64(0.0),
        state.temporary_maximum_retranslocated_nitrogen_patch,
    )
    annual_maximum_retranslocated_nitrogen_patch = jnp.where(
        patch_end_of_year,
        state.temporary_maximum_retranslocated_nitrogen_patch,
        state.annual_maximum_retranslocated_nitrogen_patch,
    )
    temporary_average_air_temperature_patch = jnp.where(
        patch_end_of_year,
        jnp.float64(0.0),
        state.temporary_average_air_temperature_patch,
    )
    annual_average_air_temperature_patch = jnp.where(
        patch_end_of_year,
        state.temporary_average_air_temperature_patch,
        state.annual_average_air_temperature_patch,
    )
    annual_sum_npp_patch = jnp.where(
        patch_end_of_year,
        fortran_multiply(state.temporary_sum_npp_patch, time_step),
        state.annual_sum_npp_patch,
    )
    temporary_sum_npp_patch = jnp.where(
        patch_end_of_year,
        jnp.float64(0.0),
        state.temporary_sum_npp_patch,
    )
    annual_sum_litterfall_patch = jnp.where(
        patch_end_of_year,
        fortran_multiply(state.temporary_sum_litterfall_patch, time_step),
        state.annual_sum_litterfall_patch,
    )
    temporary_sum_litterfall_patch = jnp.where(
        patch_end_of_year,
        jnp.float64(0.0),
        state.temporary_sum_litterfall_patch,
    )

    annual_sum_npp_aggregate = patch_to_column_average(
        annual_sum_npp_patch,
        patch_column,
        patch_weight,
        patch_active,
        end_of_year,
        maximum_patches_per_column=parameters.maximum_patches_per_column,
    )
    annual_average_air_temperature_aggregate = patch_to_column_average(
        annual_average_air_temperature_patch,
        patch_column,
        patch_weight,
        patch_active,
        end_of_year,
        maximum_patches_per_column=parameters.maximum_patches_per_column,
    )

    annual_sum_npp_column = jnp.where(
        end_of_year,
        annual_sum_npp_aggregate,
        state.annual_sum_npp_column,
    )
    annual_average_air_temperature_column = jnp.where(
        end_of_year,
        annual_average_air_temperature_aggregate,
        state.annual_average_air_temperature_column,
    )

    updated_state = CnAnnualState(
        annual_sum_counter_column=annual_sum_counter_column,
        annual_average_air_temperature_column=annual_average_air_temperature_column,
        annual_sum_npp_column=annual_sum_npp_column,
        temporary_sum_potential_gpp_patch=temporary_sum_potential_gpp_patch,
        annual_sum_potential_gpp_patch=annual_sum_potential_gpp_patch,
        temporary_maximum_retranslocated_nitrogen_patch=(
            temporary_maximum_retranslocated_nitrogen_patch
        ),
        annual_maximum_retranslocated_nitrogen_patch=(
            annual_maximum_retranslocated_nitrogen_patch
        ),
        temporary_average_air_temperature_patch=temporary_average_air_temperature_patch,
        annual_average_air_temperature_patch=annual_average_air_temperature_patch,
        temporary_sum_npp_patch=temporary_sum_npp_patch,
        annual_sum_npp_patch=annual_sum_npp_patch,
        temporary_sum_litterfall_patch=temporary_sum_litterfall_patch,
        annual_sum_litterfall_patch=annual_sum_litterfall_patch,
    )
    return CnAnnualResult(
        state=updated_state,
        diagnostics=CnAnnualDiagnostics(
            end_of_year_column=end_of_year,
            turned_over_column_count=jnp.sum(end_of_year, dtype=jnp.int32),
            turned_over_patch_count=jnp.sum(patch_end_of_year, dtype=jnp.int32),
        ),
    )
