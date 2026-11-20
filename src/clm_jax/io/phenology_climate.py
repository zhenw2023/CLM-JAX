
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.crop import FIRST_PROGNOSTIC_CROP_PFT
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    SubgridTopology,
    advance_restart_clock,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)
from clm_jax.physics.phenology_climate import (
    PhenologyClimateInputs,
    PhenologyClimateParameters,
    PhenologyClimateState,
)

_RESTART_VARIABLES = (
    "T_REF2M",
    "GDD0_VALUE",
    "GDD8_VALUE",
    "GDD10_VALUE",
    "gdd020",
    "gdd820",
    "gdd1020",
    "tempavg_t2m",
    "nyrs_crop_active",
)


class PhenologyClimateNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: PhenologyClimateInputs
    state: PhenologyClimateState
    parameters: PhenologyClimateParameters
    restart_path: Path


def load_phenology_climate_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
) -> PhenologyClimateNativeData:

    if case.value("CALENDAR").upper() != "NO_LEAP":
        raise ValueError("native phenology climate adapter currently requires NO_LEAP")
    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")

    topology = load_subgrid_topology(restart)
    patch_count = topology.patch.gridcell.shape[0]
    active = np.asarray(topology.patch.active, dtype=bool)
    landunit = np.asarray(topology.patch.landunit_type, dtype=np.int32)
    vegetation_type = np.asarray(topology.patch.vegetation_type, dtype=np.int32)
    soil_patch = active & (
        (landunit == LANDUNIT_SOIL) | (landunit == LANDUNIT_CROP)
    )
    crop_patch = active & (vegetation_type >= FIRST_PROGNOSTIC_CROP_PFT)

    values = load_native_variables(
        restart,
        list(_RESTART_VARIABLES),
        masked_integer_fill_value=-9999,
    )
    for name in _RESTART_VARIABLES:
        if np.shape(values[name]) != (patch_count,):
            raise ValueError(f"restart variable {name!r} must contain every patch")
    float_values = {
        name: np.asarray(values[name], dtype=np.float64)
        for name in _RESTART_VARIABLES
        if name != "nyrs_crop_active"
    }
    for name in ("T_REF2M", "tempavg_t2m"):
        if not np.all(np.isfinite(float_values[name][soil_patch])):
            raise ValueError(f"soil patches contain missing {name}")
    for name in (
        "GDD0_VALUE",
        "GDD8_VALUE",
        "GDD10_VALUE",
        "gdd020",
        "gdd820",
        "gdd1020",
    ):
        if not np.all(np.isfinite(float_values[name][crop_patch])):
            raise ValueError(f"prognostic crop patches contain missing {name}")
    years_active = np.asarray(values["nyrs_crop_active"], dtype=np.int32)
    if np.any(years_active[crop_patch] < 0):
        raise ValueError("prognostic crop patches contain invalid active-year state")

    clock = advance_restart_clock(
        load_restart_clock(restart),
        case.value("CALENDAR"),
    )
    parameters = PhenologyClimateParameters(
        time_step_seconds=np.float64(clock.step_seconds),
        seconds_per_day=np.float64(86_400.0),
        days_per_year=np.float64(365.0),
        averaging_years=np.float64(20.0),
    )
    return PhenologyClimateNativeData(
        topology=topology,
        inputs=PhenologyClimateInputs(
            soil_patch=jnp.asarray(soil_patch),
            prognostic_crop_patch=jnp.asarray(crop_patch),
            reference_temperature_2m_patch=jnp.asarray(float_values["T_REF2M"]),
            growing_degree_days_base_0_patch=jnp.asarray(float_values["GDD0_VALUE"]),
            growing_degree_days_base_8_patch=jnp.asarray(float_values["GDD8_VALUE"]),
            growing_degree_days_base_10_patch=jnp.asarray(
                float_values["GDD10_VALUE"]
            ),
            years_crop_active_patch=jnp.asarray(years_active),
            current_date_yyyymmdd=clock.current_date,
            current_time_of_day_seconds=clock.current_time_of_day,
        ),
        state=PhenologyClimateState(
            temperature_average_2m_patch=jnp.asarray(float_values["tempavg_t2m"]),
            growing_degree_days_base_0_twenty_year_patch=jnp.asarray(
                float_values["gdd020"]
            ),
            growing_degree_days_base_8_twenty_year_patch=jnp.asarray(
                float_values["gdd820"]
            ),
            growing_degree_days_base_10_twenty_year_patch=jnp.asarray(
                float_values["gdd1020"]
            ),
        ),
        parameters=parameters,
        restart_path=restart,
    )


__all__ = [
    "PhenologyClimateNativeData",
    "load_phenology_climate_native_data",
]
