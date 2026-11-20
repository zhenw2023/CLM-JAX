
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.floating_point import correctly_rounded_divide
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)


class PhenologyClimateInputs(NamedTuple):

    soil_patch: jnp.ndarray
    prognostic_crop_patch: jnp.ndarray
    reference_temperature_2m_patch: jnp.ndarray
    growing_degree_days_base_0_patch: jnp.ndarray
    growing_degree_days_base_8_patch: jnp.ndarray
    growing_degree_days_base_10_patch: jnp.ndarray
    years_crop_active_patch: jnp.ndarray
    current_date_yyyymmdd: int | jnp.ndarray
    current_time_of_day_seconds: int | jnp.ndarray


class PhenologyClimateState(NamedTuple):

    temperature_average_2m_patch: jnp.ndarray
    growing_degree_days_base_0_twenty_year_patch: jnp.ndarray
    growing_degree_days_base_8_twenty_year_patch: jnp.ndarray
    growing_degree_days_base_10_twenty_year_patch: jnp.ndarray


class PhenologyClimateParameters(NamedTuple):

    time_step_seconds: float | jnp.ndarray = 1800.0
    seconds_per_day: float | jnp.ndarray = 86_400.0
    days_per_year: float | jnp.ndarray = 365.0
    averaging_years: float | jnp.ndarray = 20.0


DEFAULT_PHENOLOGY_CLIMATE_PARAMETERS = PhenologyClimateParameters()


class PhenologyClimateDiagnostics(NamedTuple):
    updated_soil_patch_count: jnp.ndarray
    reset_crop_patch_count: jnp.ndarray
    averaged_crop_patch_count: jnp.ndarray


class PhenologyClimateResult(NamedTuple):
    state: PhenologyClimateState
    diagnostics: PhenologyClimateDiagnostics


def _validate_shapes(
    inputs: PhenologyClimateInputs,
    state: PhenologyClimateState,
) -> int:
    if jnp.ndim(inputs.soil_patch) != 1:
        raise ValueError("soil_patch must be a vector")
    patch_count = inputs.soil_patch.shape[0]
    expected = (patch_count,)
    for name in (
        "prognostic_crop_patch",
        "reference_temperature_2m_patch",
        "growing_degree_days_base_0_patch",
        "growing_degree_days_base_8_patch",
        "growing_degree_days_base_10_patch",
        "years_crop_active_patch",
    ):
        if jnp.shape(getattr(inputs, name)) != expected:
            raise ValueError(f"{name} must contain every patch")
    for name in PhenologyClimateState._fields:
        if jnp.shape(getattr(state, name)) != expected:
            raise ValueError(f"{name} must contain every patch")
    for name in ("current_date_yyyymmdd", "current_time_of_day_seconds"):
        if jnp.shape(getattr(inputs, name)) != ():
            raise ValueError(f"{name} must be scalar")
    return patch_count


def _validate_parameters(parameters: PhenologyClimateParameters) -> None:
    for name in PhenologyClimateParameters._fields:
        value = getattr(parameters, name)
        if jnp.shape(value) != ():
            raise ValueError(f"{name} must be scalar")
        if isinstance(value, (int, float, np.integer, np.floating)) and (
            not np.isfinite(value) or value <= 0.0
        ):
            raise ValueError(f"{name} must be finite and positive")


def phenology_climate(
    inputs: PhenologyClimateInputs,
    state: PhenologyClimateState,
    parameters: PhenologyClimateParameters = DEFAULT_PHENOLOGY_CLIMATE_PARAMETERS,
) -> PhenologyClimateResult:

    _validate_shapes(inputs, state)
    _validate_parameters(parameters)
    soil = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    crop = jnp.asarray(inputs.prognostic_crop_patch, dtype=jnp.bool_)

    def array(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(value, dtype=jnp.float64)

    time_step = jnp.asarray(parameters.time_step_seconds, dtype=jnp.float64)
    seconds_per_day = jnp.asarray(parameters.seconds_per_day, dtype=jnp.float64)
    days_per_year = jnp.asarray(parameters.days_per_year, dtype=jnp.float64)
    averaging_years = jnp.asarray(parameters.averaging_years, dtype=jnp.float64)
    fraction_of_day = correctly_rounded_divide(time_step, seconds_per_day)
    annual_weight = correctly_rounded_divide(fraction_of_day, days_per_year)

    temperature_average = fortran_add(
        array(state.temperature_average_2m_patch),
        fortran_multiply(array(inputs.reference_temperature_2m_patch), annual_weight),
    )
    temperature_average = jnp.where(
        soil,
        temperature_average,
        array(state.temperature_average_2m_patch),
    )

    current_date = jnp.asarray(inputs.current_date_yyyymmdd, dtype=jnp.int32)
    time_of_day = jnp.asarray(inputs.current_time_of_day_seconds, dtype=jnp.int32)
    month_day = current_date % jnp.int32(10_000)
    january_first = month_day == jnp.int32(101)
    years_active = jnp.asarray(inputs.years_crop_active_patch, dtype=jnp.int32)
    reset = crop & january_first & (years_active == jnp.int32(0))
    average = crop & january_first & (time_of_day == jnp.int32(0))
    first_active_year = average & (years_active == jnp.int32(1))
    years_minus_one = fortran_add(averaging_years, jnp.float64(-1.0))

    def update_mean(previous: jnp.ndarray, current: jnp.ndarray) -> jnp.ndarray:
        previous = array(previous)
        current = array(current)
        working = jnp.where(reset, jnp.float64(0.0), previous)
        working = jnp.where(first_active_year, current, working)
        mean_previous = jnp.where(average, working, jnp.float64(0.0))
        mean_current = jnp.where(average, current, jnp.float64(0.0))
        annual_mean = correctly_rounded_divide(
            fortran_add(fortran_multiply(years_minus_one, mean_previous), mean_current),
            averaging_years,
        )
        return jnp.where(average, annual_mean, working)

    updated = PhenologyClimateState(
        temperature_average_2m_patch=temperature_average,
        growing_degree_days_base_0_twenty_year_patch=update_mean(
            state.growing_degree_days_base_0_twenty_year_patch,
            inputs.growing_degree_days_base_0_patch,
        ),
        growing_degree_days_base_8_twenty_year_patch=update_mean(
            state.growing_degree_days_base_8_twenty_year_patch,
            inputs.growing_degree_days_base_8_patch,
        ),
        growing_degree_days_base_10_twenty_year_patch=update_mean(
            state.growing_degree_days_base_10_twenty_year_patch,
            inputs.growing_degree_days_base_10_patch,
        ),
    )
    return PhenologyClimateResult(
        state=updated,
        diagnostics=PhenologyClimateDiagnostics(
            updated_soil_patch_count=jnp.sum(soil, dtype=jnp.int32),
            reset_crop_patch_count=jnp.sum(reset, dtype=jnp.int32),
            averaged_crop_patch_count=jnp.sum(average, dtype=jnp.int32),
        ),
    )


__all__ = [
    "DEFAULT_PHENOLOGY_CLIMATE_PARAMETERS",
    "PhenologyClimateDiagnostics",
    "PhenologyClimateInputs",
    "PhenologyClimateParameters",
    "PhenologyClimateResult",
    "PhenologyClimateState",
    "phenology_climate",
]
