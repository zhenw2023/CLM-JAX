
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.growth_respiration import (
    GrowthRespirationNativeData,
    load_growth_respiration_native_data,
)
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import load_restart_clock
from clm_jax.physics.c_state_update0 import (
    CStateUpdate0Inputs,
    CStateUpdate0Parameters,
    CStateUpdate0State,
)
from clm_jax.physics.cn_zero import CnTransientFluxState
from clm_jax.physics.growth_respiration import GrowthRespirationResult


class CStateUpdate0NativeData(NamedTuple):

    inputs: CStateUpdate0Inputs
    state: CStateUpdate0State
    parameters: CStateUpdate0Parameters
    number_of_columns: int
    restart_path: Path
    parameter_data_path: Path


class CStateUpdate0Call(NamedTuple):

    inputs: CStateUpdate0Inputs
    state: CStateUpdate0State
    cn_fluxes: CnTransientFluxState
    parameters: CStateUpdate0Parameters


def _require_same_selected_file(
    name: str,
    requested: str | Path | None,
    actual: Path,
) -> None:
    if requested is None:
        return
    if Path(requested).expanduser().resolve() != actual:
        raise ValueError(f"growth-respiration native data uses a different {name} file")


def load_c_state_update0_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    growth_respiration_native_data: GrowthRespirationNativeData | None = None,
) -> CStateUpdate0NativeData:

    growth = growth_respiration_native_data
    if growth is None:
        growth = load_growth_respiration_native_data(
            case,
            restart_path=restart_path,
            parameter_data_path=parameter_data_path,
        )
    else:
        _require_same_selected_file("restart", restart_path, growth.restart_path)
        _require_same_selected_file(
            "parameter-data",
            parameter_data_path,
            growth.parameter_data_path,
        )

    patch_count = growth.inputs.soil_patch.shape[0]
    values = load_native_variables(
        growth.restart_path,
        ["cpool"],
        dtype=np.float64,
        masked_floating_fill_value=float(CLM_SPECIAL_VALUE),
    )
    carbon_pool = np.asarray(values["cpool"], dtype=np.float64)
    if carbon_pool.shape != (patch_count,):
        raise ValueError("restart variable 'cpool' must contain every patch")
    soil_patch = np.asarray(growth.inputs.soil_patch, dtype=bool)
    if not np.all(np.isfinite(carbon_pool[soil_patch])):
        raise ValueError("active soil patches contain non-finite 'cpool'")
    clock = load_restart_clock(growth.restart_path)
    if clock.step_seconds <= 0:
        raise ValueError("restart radiation time step must be positive")

    return CStateUpdate0NativeData(
        inputs=CStateUpdate0Inputs(soil_patch=growth.inputs.soil_patch),
        state=CStateUpdate0State(carbon_pool_patch=jnp.asarray(carbon_pool)),
        parameters=CStateUpdate0Parameters(
            time_step_seconds=np.float64(clock.step_seconds),
        ),
        number_of_columns=growth.number_of_columns,
        restart_path=growth.restart_path,
        parameter_data_path=growth.parameter_data_path,
    )


def assemble_c_state_update0_call(
    native: CStateUpdate0NativeData,
    growth_respiration_result: GrowthRespirationResult,
) -> CStateUpdate0Call:

    cn_fluxes = growth_respiration_result.cn_fluxes
    patch_count = native.inputs.soil_patch.shape[0]
    if cn_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("growth-respiration result has the wrong patch count")
    if cn_fluxes.column_scalar.shape[0] != native.number_of_columns:
        raise ValueError("growth-respiration result has the wrong column count")
    if native.state.carbon_pool_patch.shape != (patch_count,):
        raise ValueError("native carbon state has the wrong patch count")
    return CStateUpdate0Call(
        inputs=native.inputs,
        state=native.state,
        cn_fluxes=cn_fluxes,
        parameters=native.parameters,
    )


__all__ = [
    "CStateUpdate0Call",
    "CStateUpdate0NativeData",
    "assemble_c_state_update0_call",
    "load_c_state_update0_native_data",
]
