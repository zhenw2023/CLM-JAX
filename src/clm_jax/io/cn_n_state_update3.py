
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.cn_state_update3 import (
    CnStateUpdate3NativeData,
    load_cn_state_update3_native_data,
)
from clm_jax.io.soil_biogeochem_n_state_update1 import (
    SoilBiogeochemNStateUpdate1NativeData,
    load_soil_biogeochem_n_state_update1_native_data,
)
from clm_jax.physics.cn_n_state_update3 import (
    CnNStateUpdate3Parameters,
    CnNStateUpdate3State,
)
from clm_jax.physics.cn_precision_control import (
    CnPrecisionControlInputs,
    CnPrecisionControlParameters,
)
from clm_jax.physics.cn_state_update2 import CnStateUpdate2Inputs
from clm_jax.physics.cn_state_update3 import CnStateUpdate3WithPrecisionResult
from clm_jax.physics.cn_zero import CnTransientFluxState, set_cn_flux_field
from clm_jax.physics.cn_zero_manifest import (
    NUMBER_OF_DECOMPOSITION_LEVELS,
    NUMBER_OF_DECOMPOSITION_POOLS,
)
from clm_jax.physics.soil_biogeochem_n_leaching import (
    SoilBiogeochemNLeachingResult,
)
from clm_jax.physics.soil_biogeochem_n_state_update1 import (
    SoilBiogeochemNStateUpdate1Result,
    SoilBiogeochemNStateUpdate1State,
)


class CnNStateUpdate3NativeData(NamedTuple):

    inputs: CnStateUpdate2Inputs
    initial_mineral_state: SoilBiogeochemNStateUpdate1State
    parameters: CnNStateUpdate3Parameters
    precision_inputs: CnPrecisionControlInputs
    precision_parameters: CnPrecisionControlParameters
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path


class CnNStateUpdate3Call(NamedTuple):

    inputs: CnStateUpdate2Inputs
    state: CnNStateUpdate3State
    cn_fluxes: CnTransientFluxState
    parameters: CnNStateUpdate3Parameters


def _require_same_selected_file(
    name: str,
    requested: str | Path | None,
    actual: Path,
) -> None:
    if requested is None:
        return
    if Path(requested).expanduser().resolve() != actual:
        raise ValueError(f"native NStateUpdate3 dependency uses a different {name} file")


def load_cn_n_state_update3_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    cn_state_update3_native_data: CnStateUpdate3NativeData | None = None,
    soil_n_state_update1_native_data: (
        SoilBiogeochemNStateUpdate1NativeData | None
    ) = None,
) -> CnNStateUpdate3NativeData:

    carbon = cn_state_update3_native_data
    if carbon is None:
        carbon = load_cn_state_update3_native_data(
            case,
            restart_path=restart_path,
            surface_data_path=surface_data_path,
            parameter_data_path=parameter_data_path,
        )
    else:
        _require_same_selected_file("restart", restart_path, carbon.restart_path)
        _require_same_selected_file(
            "surface-data",
            surface_data_path,
            carbon.surface_data_path,
        )
        _require_same_selected_file(
            "parameter-data",
            parameter_data_path,
            carbon.parameter_data_path,
        )

    mineral = soil_n_state_update1_native_data
    if mineral is None:
        mineral = load_soil_biogeochem_n_state_update1_native_data(
            case,
            restart_path=carbon.restart_path,
        )
    elif mineral.restart_path != carbon.restart_path:
        raise ValueError("carbon and mineral-N dependencies use different restarts")

    carbon_columns = np.asarray(carbon.inputs.soil_column, dtype=bool)
    mineral_columns = np.asarray(mineral.soil_column, dtype=bool)
    if not np.array_equal(carbon_columns, mineral_columns):
        raise ValueError("carbon and mineral-N dependencies disagree on soil columns")
    if float(carbon.parameters.time_step) != float(mineral.parameters.time_step):
        raise ValueError("carbon and mineral-N dependencies use different time steps")
    if int(carbon.parameters.decomposition_level_count) != 20:
        raise ValueError("native NStateUpdate3 requires 20 active decomposition levels")
    if int(mineral.parameters.decomposition_level_count) != 20:
        raise ValueError("mineral-N dependency requires 20 active decomposition levels")

    return CnNStateUpdate3NativeData(
        inputs=carbon.inputs,
        initial_mineral_state=mineral.restart_state,
        parameters=CnNStateUpdate3Parameters(
            time_step=carbon.parameters.time_step,
            decomposition_level_count=carbon.parameters.decomposition_level_count,
        ),
        precision_inputs=carbon.precision_inputs,
        precision_parameters=carbon.precision_parameters,
        restart_path=carbon.restart_path,
        surface_data_path=carbon.surface_data_path,
        parameter_data_path=carbon.parameter_data_path,
    )


def assemble_cn_n_state_update3_call(
    native: CnNStateUpdate3NativeData,
    carbon_state_update3_result: CnStateUpdate3WithPrecisionResult,
    mineral_state_update1_result: SoilBiogeochemNStateUpdate1Result,
    nitrate_leaching_result: SoilBiogeochemNLeachingResult,
) -> CnNStateUpdate3Call:

    patch_count = native.inputs.soil_patch.shape[0]
    column_count = native.inputs.soil_column.shape[0]
    level_shape = (column_count, NUMBER_OF_DECOMPOSITION_LEVELS)
    pool_shape = level_shape + (NUMBER_OF_DECOMPOSITION_POOLS,)

    carbon_state = carbon_state_update3_result.state
    if carbon_state.vegetation.paired_nitrogen_patch.shape[0] != patch_count:
        raise ValueError("post-carbon-Update3 vegetation has the wrong patch extent")
    if jnp.shape(carbon_state.soil_nitrogen_pool_column_level_pool) != pool_shape:
        raise ValueError("post-carbon-Update3 nitrogen pools have the wrong native shape")

    mineral_state = mineral_state_update1_result.state
    for name in mineral_state._fields:
        if jnp.shape(getattr(mineral_state, name)) != level_shape:
            raise ValueError(f"same-step mineral-N {name} has the wrong native shape")

    leaching = nitrate_leaching_result.flux_state
    for name in leaching._fields:
        if jnp.shape(getattr(leaching, name)) != level_shape:
            raise ValueError(f"same-step nitrate-loss {name} has the wrong native shape")

    fluxes = carbon_state_update3_result.cn_fluxes
    if fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("post-fire packed fluxes have the wrong patch extent")
    if fluxes.column_level.shape[:2] != level_shape:
        raise ValueError("post-fire packed fluxes have the wrong column-level extent")
    if fluxes.column_level_pool.shape[:3] != pool_shape:
        raise ValueError("post-fire packed fluxes have the wrong column-level-pool extent")
    fluxes = set_cn_flux_field(
        fluxes,
        "soil_nitrogen.smin_no3_leached_vr_col",
        leaching.nitrate_leaching_column_level,
    )
    fluxes = set_cn_flux_field(
        fluxes,
        "soil_nitrogen.smin_no3_runoff_vr_col",
        leaching.nitrate_runoff_column_level,
    )

    return CnNStateUpdate3Call(
        inputs=native.inputs,
        state=CnNStateUpdate3State(
            vegetation=carbon_state.vegetation,
            soil_nitrogen_pool_column_level_pool=(
                carbon_state.soil_nitrogen_pool_column_level_pool
            ),
            mineral_nitrogen_column_level=(
                mineral_state.mineral_nitrogen_column_level
            ),
            ammonium_column_level=mineral_state.ammonium_column_level,
            nitrate_column_level=mineral_state.nitrate_column_level,
        ),
        cn_fluxes=fluxes,
        parameters=native.parameters,
    )


__all__ = [
    "CnNStateUpdate3Call",
    "CnNStateUpdate3NativeData",
    "assemble_cn_n_state_update3_call",
    "load_cn_n_state_update3_native_data",
]
