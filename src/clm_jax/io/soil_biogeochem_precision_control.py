
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.cn_n_state_update3 import (
    CnNStateUpdate3NativeData,
    load_cn_n_state_update3_native_data,
)
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.physics.cn_n_state_update3 import CnNStateUpdate3Result
from clm_jax.physics.cn_state_update3 import CnStateUpdate3WithPrecisionResult
from clm_jax.physics.cn_zero_manifest import (
    NUMBER_OF_DECOMPOSITION_LEVELS,
    NUMBER_OF_DECOMPOSITION_POOLS,
)
from clm_jax.physics.soil_biogeochem_precision_control import (
    SoilBiogeochemPrecisionControlInputs,
    SoilBiogeochemPrecisionControlParameters,
    SoilBiogeochemPrecisionControlState,
)


class SoilBiogeochemPrecisionControlPersistentState(NamedTuple):

    carbon_truncation_column_level: jnp.ndarray
    nitrogen_truncation_column_level: jnp.ndarray


class SoilBiogeochemPrecisionControlNativeData(NamedTuple):

    inputs: SoilBiogeochemPrecisionControlInputs
    initial_persistent_state: SoilBiogeochemPrecisionControlPersistentState
    parameters: SoilBiogeochemPrecisionControlParameters
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path


class SoilBiogeochemPrecisionControlCall(NamedTuple):

    inputs: SoilBiogeochemPrecisionControlInputs
    state: SoilBiogeochemPrecisionControlState
    parameters: SoilBiogeochemPrecisionControlParameters


def _require_same_selected_file(
    name: str,
    requested: str | Path | None,
    actual: Path,
) -> None:
    if requested is None:
        return
    if Path(requested).expanduser().resolve() != actual:
        raise ValueError(
            f"native soil precision-control dependency uses a different {name} file"
        )


def load_soil_biogeochem_precision_control_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    n_state_update3_native_data: CnNStateUpdate3NativeData | None = None,
) -> SoilBiogeochemPrecisionControlNativeData:

    nitrogen = n_state_update3_native_data
    if nitrogen is None:
        nitrogen = load_cn_n_state_update3_native_data(
            case,
            restart_path=restart_path,
            surface_data_path=surface_data_path,
            parameter_data_path=parameter_data_path,
        )
    else:
        _require_same_selected_file("restart", restart_path, nitrogen.restart_path)
        _require_same_selected_file(
            "surface-data",
            surface_data_path,
            nitrogen.surface_data_path,
        )
        _require_same_selected_file(
            "parameter-data",
            parameter_data_path,
            nitrogen.parameter_data_path,
        )

    general = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    required = {
        "use_cn": True,
        "use_fun": True,
        "use_nitrif_denitrif": True,
        "use_vertsoilc": True,
        "use_c13": False,
        "use_c14": False,
    }
    for name, expected in required.items():
        if general.get(name, False) != expected:
            raise ValueError(
                "native soil precision-control adapter requires "
                f"{name}={expected!r}"
            )

    soil_column = np.asarray(nitrogen.inputs.soil_column, dtype=bool)
    column_count = soil_column.size
    raw = load_native_variables(
        nitrogen.restart_path,
        ["col_ctrunc_vr", "col_ntrunc_vr"],
        dtype=np.float64,
        masked_floating_fill_value=float(CLM_SPECIAL_VALUE),
    )
    values: dict[str, np.ndarray] = {}
    for name, array in raw.items():
        host = np.asarray(array, dtype=np.float64)
        if host.shape != (column_count, NUMBER_OF_DECOMPOSITION_LEVELS):
            raise ValueError(f"restart {name} must have shape (column, 25)")
        active = host[soil_column, :20]
        valid = np.isfinite(active) & (np.abs(active) < np.float64(CLM_SPECIAL_VALUE))
        if not np.all(valid):
            raise ValueError(f"active soil columns contain missing {name}")
        values[name] = host

    return SoilBiogeochemPrecisionControlNativeData(
        inputs=SoilBiogeochemPrecisionControlInputs(
            soil_column=nitrogen.inputs.soil_column
        ),
        initial_persistent_state=SoilBiogeochemPrecisionControlPersistentState(
            carbon_truncation_column_level=jnp.asarray(values["col_ctrunc_vr"]),
            nitrogen_truncation_column_level=jnp.asarray(values["col_ntrunc_vr"]),
        ),
        parameters=SoilBiogeochemPrecisionControlParameters(
            carbon_critical_value=np.float64(1.0e-8),
            decomposition_level_count=nitrogen.parameters.decomposition_level_count,
        ),
        restart_path=nitrogen.restart_path,
        surface_data_path=nitrogen.surface_data_path,
        parameter_data_path=nitrogen.parameter_data_path,
    )


def assemble_soil_biogeochem_precision_control_call(
    native: SoilBiogeochemPrecisionControlNativeData,
    carbon_state_update3_result: CnStateUpdate3WithPrecisionResult,
    nitrogen_state_update3_result: CnNStateUpdate3Result,
    persistent_state: SoilBiogeochemPrecisionControlPersistentState,
) -> SoilBiogeochemPrecisionControlCall:

    column_count = native.inputs.soil_column.shape[0]
    level_shape = (column_count, NUMBER_OF_DECOMPOSITION_LEVELS)
    pool_shape = (*level_shape, NUMBER_OF_DECOMPOSITION_POOLS)
    carbon_state = carbon_state_update3_result.state
    nitrogen_state = nitrogen_state_update3_result.state
    if jnp.shape(carbon_state.soil_carbon_pool_column_level_pool) != pool_shape:
        raise ValueError("same-step carbon pools have the wrong native shape")
    if jnp.shape(nitrogen_state.soil_nitrogen_pool_column_level_pool) != pool_shape:
        raise ValueError("same-step nitrogen pools have the wrong native shape")
    for name in (
        "mineral_nitrogen_column_level",
        "ammonium_column_level",
        "nitrate_column_level",
    ):
        if jnp.shape(getattr(nitrogen_state, name)) != level_shape:
            raise ValueError(f"same-step nitrogen {name} has the wrong native shape")
    for name in persistent_state._fields:
        if jnp.shape(getattr(persistent_state, name)) != level_shape:
            raise ValueError(f"current {name} has the wrong native shape")

    return SoilBiogeochemPrecisionControlCall(
        inputs=native.inputs,
        state=SoilBiogeochemPrecisionControlState(
            carbon_pool_column_level_pool=(
                carbon_state.soil_carbon_pool_column_level_pool
            ),
            nitrogen_pool_column_level_pool=(
                nitrogen_state.soil_nitrogen_pool_column_level_pool
            ),
            carbon_truncation_column_level=(
                persistent_state.carbon_truncation_column_level
            ),
            nitrogen_truncation_column_level=(
                persistent_state.nitrogen_truncation_column_level
            ),
            mineral_nitrogen_column_level=(
                nitrogen_state.mineral_nitrogen_column_level
            ),
            ammonium_column_level=nitrogen_state.ammonium_column_level,
            nitrate_column_level=nitrogen_state.nitrate_column_level,
        ),
        parameters=native.parameters,
    )


__all__ = [
    "SoilBiogeochemPrecisionControlCall",
    "SoilBiogeochemPrecisionControlNativeData",
    "SoilBiogeochemPrecisionControlPersistentState",
    "assemble_soil_biogeochem_precision_control_call",
    "load_soil_biogeochem_precision_control_native_data",
]
