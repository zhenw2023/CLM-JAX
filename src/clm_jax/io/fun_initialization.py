
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import (
    RestartClock,
    SubgridTopology,
    advance_restart_clock,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
    model_step_from_restart_clock,
)
from clm_jax.physics.fun_initialization import (
    FunInitializationInputs,
    FunInitializationParameters,
    FunInitializationState,
)

_STATE_VARIABLES = (
    "leafcn_offset",
    "storage_cdemand",
    "storage_ndemand",
    "leafc_storage_xfer_acc",
    "leafn_storage_xfer_acc",
)
_PARAMETER_VARIABLES = ("leafcn", "ndays_on", "ndays_off")
_SECONDS_PER_DAY = 86_400


class FunInitializationNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: FunInitializationInputs
    state: FunInitializationState
    parameters: FunInitializationParameters
    clock: RestartClock
    restart_path: Path
    parameter_data_path: Path


def _selected_parameter_path(
    case: CimeCase,
    override: str | Path | None,
) -> Path:
    if override is None:
        group = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        declared = group.get("paramfile")
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError(f"clm_inparm in {case.case_docs / 'lnd_in'} lacks paramfile")
        path = Path(case.expand(declared)).expanduser()
        if not path.is_absolute():
            path = case.root / path
        path = path.resolve()
    else:
        path = Path(override).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM paramfile does not exist: {path}")
    return path


def _finite_scalar(name: str, value: jnp.ndarray) -> np.float64:
    host = np.asarray(value, dtype=np.float64)
    if host.size != 1 or not np.isfinite(host.reshape(())):
        raise ValueError(f"CLM parameter {name!r} must contain one finite scalar")
    return np.float64(host.reshape(()))


def load_fun_initialization_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> FunInitializationNativeData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    for key, expected in {
        "use_cn": True,
        "use_fun": True,
        "use_fates": False,
    }.items():
        if general.get(key) is not expected:
            raise ValueError(f"native FUN initialization requires {key}={expected!r}")
    if case.value("CALENDAR").upper() != "NO_LEAP":
        raise ValueError("native FUN initialization currently requires NO_LEAP")

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    parameter_data = _selected_parameter_path(case, parameter_data_path)
    topology = load_subgrid_topology(restart)
    patch_count = topology.patch.gridcell.shape[0]
    clock = advance_restart_clock(
        load_restart_clock(restart),
        case.value("CALENDAR"),
    )

    values = load_native_variables(
        restart,
        list(_STATE_VARIABLES),
        dtype=np.float64,
    )
    for name, array in values.items():
        if array.shape != (patch_count,):
            raise ValueError(f"restart variable {name!r} must contain every patch")
    parameter_values = load_native_variables(
        parameter_data,
        list(_PARAMETER_VARIABLES),
        dtype=np.float64,
    )
    leaf_cn_ratio = np.asarray(parameter_values["leafcn"], dtype=np.float64)
    if leaf_cn_ratio.ndim != 1 or leaf_cn_ratio.size < 1:
        raise ValueError("CLM leafcn must be a nonempty PFT vector")
    if not np.all(np.isfinite(leaf_cn_ratio)) or np.any(leaf_cn_ratio <= 0.0):
        raise ValueError("CLM leafcn must contain finite positive values")
    vegetation_type = np.asarray(topology.patch.vegetation_type, dtype=np.int32)
    if np.any((vegetation_type < 0) | (vegetation_type >= leaf_cn_ratio.size)):
        raise ValueError("restart vegetation type lies outside leafcn PFT vector")

    model_step = model_step_from_restart_clock(clock, case.value("CALENDAR"))
    return FunInitializationNativeData(
        topology=topology,
        inputs=FunInitializationInputs(
            vegetation_type_patch=jnp.asarray(vegetation_type),
            model_step=np.int32(model_step),
        ),
        state=FunInitializationState(
            leaf_cn_ratio_offset_patch=jnp.asarray(values["leafcn_offset"]),
            storage_carbon_demand_patch=jnp.asarray(values["storage_cdemand"]),
            storage_nitrogen_demand_patch=jnp.asarray(values["storage_ndemand"]),
            leaf_carbon_storage_transfer_accumulator_patch=jnp.asarray(
                values["leafc_storage_xfer_acc"]
            ),
            leaf_nitrogen_storage_transfer_accumulator_patch=jnp.asarray(
                values["leafn_storage_xfer_acc"]
            ),
        ),
        parameters=FunInitializationParameters(
            leaf_cn_ratio_pft=jnp.asarray(leaf_cn_ratio),
            time_step_seconds=np.float64(clock.step_seconds),
            days_per_year=np.float64(365.0),
            onset_period_days=_finite_scalar(
                "ndays_on",
                parameter_values["ndays_on"],
            ),
            offset_period_days=_finite_scalar(
                "ndays_off",
                parameter_values["ndays_off"],
            ),
            seconds_per_day=np.float64(_SECONDS_PER_DAY),
        ),
        clock=clock,
        restart_path=restart,
        parameter_data_path=parameter_data,
    )


__all__ = [
    "FunInitializationNativeData",
    "load_fun_initialization_native_data",
]
