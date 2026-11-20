
from __future__ import annotations

import calendar as calendar_module
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    RestartClock,
    advance_restart_clock,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)
from clm_jax.physics.cn_annual_update import (
    CnAnnualInputs,
    CnAnnualParameters,
    CnAnnualState,
)


class CnAnnualNativeData(NamedTuple):

    inputs: CnAnnualInputs
    restart_state: CnAnnualState
    parameters: CnAnnualParameters
    restart_clock: RestartClock
    calendar_name: str
    restart_path: Path


class CnAnnualCall(NamedTuple):

    inputs: CnAnnualInputs
    state: CnAnnualState
    parameters: CnAnnualParameters


_RESTART_FIELD_MAP = {
    "annual_sum_counter_column": "annsum_counter",
    "annual_average_air_temperature_column": "cannavg_t2m",
    "annual_sum_npp_column": "cannsum_npp",
    "temporary_sum_potential_gpp_patch": "tempsum_potential_gpp",
    "annual_sum_potential_gpp_patch": "annsum_potential_gpp",
    "temporary_maximum_retranslocated_nitrogen_patch": "tempmax_retransn",
    "annual_maximum_retranslocated_nitrogen_patch": "annmax_retransn",
    "temporary_average_air_temperature_patch": "tempavg_t2m",
    "annual_average_air_temperature_patch": "annavg_t2m",
    "temporary_sum_npp_patch": "tempsum_npp",
    "annual_sum_npp_patch": "annsum_npp",
    "temporary_sum_litterfall_patch": "tempsum_litfall",
    "annual_sum_litterfall_patch": "annsum_litfall",
}


def _days_per_year(calendar_name: str, current_date: int) -> int:
    year = current_date // 10_000
    if year < 1:
        raise ValueError("CLM restart current date must contain a positive year")
    if calendar_name == "NO_LEAP":
        return 365
    if calendar_name == "GREGORIAN":
        return 366 if calendar_module.isleap(year) else 365
    if calendar_name == "360_DAY":
        return 360
    raise ValueError(f"unsupported CLM annual-update calendar: {calendar_name!r}")


def _validate_case(case: CimeCase) -> None:
    general = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    required = {
        "use_cn": True,
        "use_crop": True,
        "use_fates": False,
    }
    for name, expected in required.items():
        if general.get(name) != expected:
            raise ValueError(f"native annual adapter requires {name}={expected!r}")


def _validate_topology(
    patch_column: np.ndarray,
    patch_weight: np.ndarray,
    patch_active: np.ndarray,
    soil_patch: np.ndarray,
    soil_column: np.ndarray,
) -> int:
    column_count = soil_column.size
    if patch_column.ndim != 1 or np.any(
        (patch_column < 0) | (patch_column >= column_count)
    ):
        raise ValueError("restart patch-to-column topology is out of bounds")
    if patch_column.size and np.any(patch_column[1:] < patch_column[:-1]):
        raise ValueError("annual aggregation requires contiguous ordered CLM patches")
    if not (
        patch_weight.shape
        == patch_active.shape
        == soil_patch.shape
        == patch_column.shape
    ):
        raise ValueError("restart annual patch topology has inconsistent extents")
    if not np.all(np.isfinite(patch_weight)) or np.any(patch_weight < 0.0):
        raise ValueError("restart patch weights must be finite and nonnegative")
    if np.any(~soil_column[patch_column[soil_patch]]):
        raise ValueError("filtered soil patches must map to filtered soil columns")
    counts = np.bincount(patch_column, minlength=column_count)
    return int(counts.max(initial=0))


def _restart_state(
    restart_path: Path,
    patch_count: int,
    column_count: int,
) -> CnAnnualState:
    values = load_native_variables(
        restart_path,
        list(_RESTART_FIELD_MAP.values()),
        dtype=np.float64,
        masked_floating_fill_value=float(CLM_SPECIAL_VALUE),
    )
    arrays: dict[str, jnp.ndarray] = {}
    for python_name, native_name in _RESTART_FIELD_MAP.items():
        host = np.asarray(values[native_name], dtype=np.float64)
        expected = (
            (column_count,)
            if python_name.endswith("_column")
            else (patch_count,)
        )
        if host.shape != expected:
            raise ValueError(
                f"restart annual field {native_name!r} has shape {host.shape}, expected {expected}"
            )
        if not np.all(np.isfinite(host)):
            raise ValueError(f"restart annual field {native_name!r} contains non-finite values")
        arrays[python_name] = jnp.asarray(host)
    return CnAnnualState(**arrays)


def load_cn_annual_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
) -> CnAnnualNativeData:

    _validate_case(case)
    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")

    topology = load_subgrid_topology(restart)
    column_active = np.asarray(topology.column.active, dtype=bool)
    patch_active = np.asarray(topology.patch.active, dtype=bool)
    soil_column = column_active & np.isin(
        np.asarray(topology.column.landunit_type, dtype=np.int32),
        (LANDUNIT_SOIL, LANDUNIT_CROP),
    )
    soil_patch = patch_active & np.isin(
        np.asarray(topology.patch.landunit_type, dtype=np.int32),
        (LANDUNIT_SOIL, LANDUNIT_CROP),
    )
    patch_column = np.asarray(topology.patch.column, dtype=np.int32)
    patch_weight = np.asarray(topology.patch.weight_on_column, dtype=np.float64)
    maximum_patches = _validate_topology(
        patch_column,
        patch_weight,
        patch_active,
        soil_patch,
        soil_column,
    )
    clock = load_restart_clock(restart)
    if clock.step_seconds <= 0:
        raise ValueError("restart CLM time step must be positive")
    calendar_name = case.value("CALENDAR").strip().upper()
    call_clock = advance_restart_clock(clock, calendar_name)

    inputs = CnAnnualInputs(
        soil_patch=jnp.asarray(soil_patch),
        soil_column=jnp.asarray(soil_column),
        patch_column_index=topology.patch.column,
        patch_weight_on_column=topology.patch.weight_on_column,
        patch_active=topology.patch.active,
    )
    return CnAnnualNativeData(
        inputs=inputs,
        restart_state=_restart_state(
            restart,
            patch_count=patch_column.size,
            column_count=soil_column.size,
        ),
        parameters=CnAnnualParameters(
            time_step=np.float64(clock.step_seconds),
            days_per_year=_days_per_year(calendar_name, call_clock.current_date),
            maximum_patches_per_column=maximum_patches,
        ),
        restart_clock=clock,
        calendar_name=calendar_name,
        restart_path=restart,
    )


def _validate_state(native: CnAnnualNativeData, state: CnAnnualState) -> None:
    patch_count = native.inputs.soil_patch.shape[0]
    column_count = native.inputs.soil_column.shape[0]
    for name, value in zip(state._fields, state, strict=True):
        expected = (column_count,) if name.endswith("_column") else (patch_count,)
        if jnp.shape(value) != expected:
            raise ValueError(f"current annual state field {name!r} has the wrong extent")


def initial_cn_annual_call(native: CnAnnualNativeData) -> CnAnnualCall:

    return CnAnnualCall(native.inputs, native.restart_state, native.parameters)


def assemble_cn_annual_call(
    native: CnAnnualNativeData,
    current_state: CnAnnualState,
) -> CnAnnualCall:

    _validate_state(native, current_state)
    return CnAnnualCall(native.inputs, current_state, native.parameters)


__all__ = [
    "CnAnnualCall",
    "CnAnnualNativeData",
    "assemble_cn_annual_call",
    "initial_cn_annual_call",
    "load_cn_annual_native_data",
]
