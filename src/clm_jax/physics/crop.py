
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class CropYearInputs(NamedTuple):

    prognostic_crop_patch: jnp.ndarray
    current_date_yyyymmdd: int | jnp.ndarray
    current_time_of_day_seconds: int | jnp.ndarray
    first_model_step: bool | jnp.ndarray


class CropYearState(NamedTuple):

    years_active_patch: jnp.ndarray


class CropYearDiagnostics(NamedTuple):

    increment_year: jnp.ndarray


class CropYearResult(NamedTuple):
    state: CropYearState
    diagnostics: CropYearDiagnostics


def increment_crop_year(inputs: CropYearInputs, state: CropYearState) -> CropYearResult:

    if state.years_active_patch.ndim != 1:
        raise ValueError("years_active_patch must be a vector")
    if inputs.prognostic_crop_patch.shape != state.years_active_patch.shape:
        raise ValueError("prognostic_crop_patch must match years_active_patch")

    years = jnp.asarray(state.years_active_patch, dtype=jnp.int32)
    crop_patch = jnp.asarray(inputs.prognostic_crop_patch, dtype=jnp.bool_)
    current_date = jnp.asarray(inputs.current_date_yyyymmdd, dtype=jnp.int32)
    time_of_day = jnp.asarray(inputs.current_time_of_day_seconds, dtype=jnp.int32)
    first_step = jnp.asarray(inputs.first_model_step, dtype=jnp.bool_)
    increment_year = (
        (current_date % jnp.int32(10000) == jnp.int32(101))
        & (time_of_day == jnp.int32(0))
        & ~first_step
    )
    updated = jnp.where(crop_patch & increment_year, years + jnp.int32(1), years)
    return CropYearResult(
        state=CropYearState(years_active_patch=updated),
        diagnostics=CropYearDiagnostics(increment_year=increment_year),
    )
