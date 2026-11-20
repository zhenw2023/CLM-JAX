
from __future__ import annotations

import calendar as calendar_module
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    advance_restart_clock,
    configured_land_restart,
    load_restart_clock,
)
from clm_jax.physics.nitrogen_dynamics import (
    FreeLivingFixationInputs,
    FreeLivingFixationParameters,
    NitrogenFertilizerInputs,
    NitrogenFertilizerParameters,
)


class FreeLivingFixationRestartData(NamedTuple):

    inputs: FreeLivingFixationInputs
    parameters: FreeLivingFixationParameters
    accumulator_steps_column: jnp.ndarray
    accumulator_period_steps: int
    restart_path: Path


class NitrogenFertilizerRestartData(NamedTuple):

    inputs: NitrogenFertilizerInputs
    parameters: NitrogenFertilizerParameters
    restart_path: Path


def _days_per_current_year(case: CimeCase, current_date: int) -> float:
    calendar_name = case.value("CALENDAR").strip().upper()
    if calendar_name == "NO_LEAP":
        return 365.0
    if calendar_name == "GREGORIAN":
        year = current_date // 10000
        return 366.0 if calendar_module.isleap(year) else 365.0
    raise ValueError(f"unsupported CLM calendar for nitrogen dynamics: {calendar_name!r}")


def load_free_living_fixation_restart(
    case: CimeCase,
    restart_path: str | Path | None = None,
) -> FreeLivingFixationRestartData:

    restart = (
        Path(restart_path).expanduser().resolve()
        if restart_path is not None
        else configured_land_restart(case)
    )
    values = load_native_variables(
        restart,
        [
            "AnnET_VALUE",
            "AnnET_NSTEPS",
            "AnnET_PERIOD",
            "cols1d_ityplun",
            "cols1d_active",
        ],
    )
    annual_et = jnp.asarray(values["AnnET_VALUE"], dtype=jnp.float64)
    accumulator_steps = jnp.asarray(values["AnnET_NSTEPS"], dtype=jnp.int32)
    landunit_type = jnp.asarray(values["cols1d_ityplun"], dtype=jnp.int32)
    active = jnp.asarray(values["cols1d_active"] == 1, dtype=jnp.bool_)
    if annual_et.ndim != 1:
        raise ValueError("AnnET_VALUE must be a native column vector")
    for name, value in {
        "AnnET_NSTEPS": accumulator_steps,
        "cols1d_ityplun": landunit_type,
        "cols1d_active": active,
    }.items():
        if value.shape != annual_et.shape:
            raise ValueError(f"restart variable {name!r} does not match AnnET_VALUE")
    period_value = values["AnnET_PERIOD"]
    if period_value.size != 1:
        raise ValueError("AnnET_PERIOD must contain one scalar value")
    period = int(period_value.reshape(()))
    if period <= 0:
        raise ValueError("AnnET_PERIOD must be positive")

    clock = advance_restart_clock(
        load_restart_clock(restart),
        case.value("CALENDAR"),
    )
    soil_column = active & (
        (landunit_type == jnp.int32(LANDUNIT_SOIL)) | (landunit_type == jnp.int32(LANDUNIT_CROP))
    )
    return FreeLivingFixationRestartData(
        inputs=FreeLivingFixationInputs(
            annual_average_evapotranspiration_column=annual_et,
            soil_column=soil_column,
            days_per_year=jnp.float64(_days_per_current_year(case, clock.current_date)),
        ),
        parameters=FreeLivingFixationParameters.from_case(case),
        accumulator_steps_column=accumulator_steps,
        accumulator_period_steps=period,
        restart_path=restart,
    )


def load_nitrogen_fertilizer_restart(
    case: CimeCase,
    restart_path: str | Path | None = None,
) -> NitrogenFertilizerRestartData:

    restart = (
        Path(restart_path).expanduser().resolve()
        if restart_path is not None
        else configured_land_restart(case)
    )
    values = load_native_variables(
        restart,
        [
            "fert",
            "pfts1d_column_index",
            "pfts1d_wtcol",
            "pfts1d_active",
            "cols1d_ityplun",
            "cols1d_active",
        ],
    )
    fertilizer = jnp.asarray(values["fert"], dtype=jnp.float64)
    patch_column = jnp.asarray(values["pfts1d_column_index"], dtype=jnp.int32) - 1
    patch_weight = jnp.asarray(values["pfts1d_wtcol"], dtype=jnp.float64)
    patch_active = jnp.asarray(values["pfts1d_active"] == 1, dtype=jnp.bool_)
    if fertilizer.ndim != 1:
        raise ValueError("fert must be a native patch vector")
    for name, value in {
        "pfts1d_column_index": patch_column,
        "pfts1d_wtcol": patch_weight,
        "pfts1d_active": patch_active,
    }.items():
        if value.shape != fertilizer.shape:
            raise ValueError(f"restart variable {name!r} does not match fert")

    landunit_type = jnp.asarray(values["cols1d_ityplun"], dtype=jnp.int32)
    column_active = jnp.asarray(values["cols1d_active"] == 1, dtype=jnp.bool_)
    if landunit_type.ndim != 1 or column_active.shape != landunit_type.shape:
        raise ValueError("native column topology has inconsistent shapes")
    soil_column = column_active & (
        (landunit_type == jnp.int32(LANDUNIT_SOIL)) | (landunit_type == jnp.int32(LANDUNIT_CROP))
    )
    return NitrogenFertilizerRestartData(
        inputs=NitrogenFertilizerInputs(
            fertilizer_patch=fertilizer,
            patch_column_index=patch_column,
            patch_weight_on_column=patch_weight,
            patch_active=patch_active,
            soil_column=soil_column,
        ),
        parameters=NitrogenFertilizerParameters.from_case(case),
        restart_path=restart,
    )
