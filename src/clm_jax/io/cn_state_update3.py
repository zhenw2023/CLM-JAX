
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.cn_fire_fluxes import (
    CnFireFluxNativeData,
    load_cn_fire_flux_native_data,
)
from clm_jax.io.cn_precision_control import (
    CnPrecisionControlNativeData,
    load_cn_precision_control_native_data,
)
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.physics.cn_fire_fluxes import CnFireFluxResult
from clm_jax.physics.cn_precision_control import (
    CnPrecisionControlInputs,
    CnPrecisionControlParameters,
)
from clm_jax.physics.cn_state_update2 import (
    CnStateUpdate2hWithPrecisionResult,
    CnStateUpdate2Inputs,
    CnStateUpdate2State,
)
from clm_jax.physics.cn_state_update3 import CnStateUpdate3Parameters
from clm_jax.physics.cn_zero import CnTransientFluxState
from clm_jax.physics.cn_zero_manifest import (
    NUMBER_OF_DECOMPOSITION_LEVELS,
    NUMBER_OF_DECOMPOSITION_POOLS,
)


class CnStateUpdate3NativeData(NamedTuple):

    inputs: CnStateUpdate2Inputs
    parameters: CnStateUpdate3Parameters
    precision_inputs: CnPrecisionControlInputs
    precision_parameters: CnPrecisionControlParameters
    use_guardrail: bool
    use_c13: bool
    use_c14: bool
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path


class CnStateUpdate3Call(NamedTuple):

    inputs: CnStateUpdate2Inputs
    state: CnStateUpdate2State
    cn_fluxes: CnTransientFluxState
    parameters: CnStateUpdate3Parameters


class CnStateUpdate3WithPrecisionCall(NamedTuple):

    inputs: CnStateUpdate2Inputs
    state: CnStateUpdate2State
    cn_fluxes: CnTransientFluxState
    parameters: CnStateUpdate3Parameters
    precision_inputs: CnPrecisionControlInputs
    precision_parameters: CnPrecisionControlParameters


def _logical(group: dict[str, object], name: str, default: bool) -> bool:
    value = group.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"CLM namelist value {name!r} must be logical")
    return value


def _require_same_selected_file(
    name: str,
    requested: str | Path | None,
    actual: Path,
) -> None:
    if requested is None:
        return
    if Path(requested).expanduser().resolve() != actual:
        raise ValueError(f"native Update3 dependency uses a different {name} file")


def load_cn_state_update3_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    fire_flux_native_data: CnFireFluxNativeData | None = None,
    precision_control_native_data: CnPrecisionControlNativeData | None = None,
) -> CnStateUpdate3NativeData:

    fire = fire_flux_native_data
    if fire is None:
        fire = load_cn_fire_flux_native_data(
            case,
            restart_path=restart_path,
            surface_data_path=surface_data_path,
            parameter_data_path=parameter_data_path,
        )
    else:
        _require_same_selected_file("restart", restart_path, fire.restart_path)
        _require_same_selected_file("surface-data", surface_data_path, fire.surface_data_path)
        _require_same_selected_file(
            "parameter-data",
            parameter_data_path,
            fire.parameter_data_path,
        )

    clm = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    use_c13 = _logical(clm, "use_c13", False)
    use_c14 = _logical(clm, "use_c14", False)
    if use_c13 or use_c14:
        raise NotImplementedError(
            "the normal-carbon Update3 adapter does not yet compose active C13/C14 updates"
        )

    precision = precision_control_native_data
    if precision is None:
        precision = load_cn_precision_control_native_data(
            case,
            restart_path=fire.restart_path,
            parameter_data_path=fire.parameter_data_path,
        )
    else:
        _require_same_selected_file("restart", restart_path, precision.restart_path)
        _require_same_selected_file(
            "parameter-data",
            parameter_data_path,
            precision.parameter_data_path,
        )

    persistent = fire.persistent_inputs
    if precision.restart_path != fire.restart_path:
        raise ValueError("fire and precision-control dependencies use different restarts")
    if precision.parameter_data_path != fire.parameter_data_path:
        raise ValueError("fire and precision-control dependencies use different parameter files")
    if not np.array_equal(
        np.asarray(precision.inputs.soil_patch, dtype=bool),
        np.asarray(persistent.soil_patch, dtype=bool),
    ):
        raise ValueError("fire and precision-control dependencies disagree on soil patches")
    if not np.array_equal(
        np.asarray(precision.inputs.vegetation_type_patch, dtype=np.int32),
        np.asarray(persistent.vegetation_type_patch, dtype=np.int32),
    ):
        raise ValueError("fire and precision-control dependencies disagree on PFT identity")
    use_guardrail = _logical(clm, "use_nguardrail", False)
    if not use_guardrail or not bool(precision.parameters.use_nitrogen_guardrail):
        raise ValueError("native Update3 composition requires use_nguardrail=.true.")
    if int(fire.parameters.decomposition_level_count) != 20:
        raise ValueError("native Update3 requires 20 active decomposition levels")
    return CnStateUpdate3NativeData(
        inputs=CnStateUpdate2Inputs(
            soil_patch=persistent.soil_patch,
            soil_column=persistent.soil_column,
        ),
        parameters=CnStateUpdate3Parameters(
            time_step=fire.parameters.time_step,
            decomposition_level_count=fire.parameters.decomposition_level_count,
        ),
        precision_inputs=precision.inputs,
        precision_parameters=precision.parameters,
        use_guardrail=use_guardrail,
        use_c13=use_c13,
        use_c14=use_c14,
        restart_path=fire.restart_path,
        surface_data_path=fire.surface_data_path,
        parameter_data_path=fire.parameter_data_path,
    )


def assemble_cn_state_update3_call(
    native: CnStateUpdate3NativeData,
    state_update2h_result: CnStateUpdate2hWithPrecisionResult,
    fire_flux_result: CnFireFluxResult,
) -> CnStateUpdate3Call:

    patch_count = native.inputs.soil_patch.shape[0]
    column_count = native.inputs.soil_column.shape[0]
    state = state_update2h_result.state
    if state.vegetation.paired_carbon_patch.shape[0] != patch_count:
        raise ValueError("post-Update2h vegetation state has the wrong patch extent")
    pool_shape = (
        column_count,
        NUMBER_OF_DECOMPOSITION_LEVELS,
        NUMBER_OF_DECOMPOSITION_POOLS,
    )
    if jnp.shape(state.soil_carbon_pool_column_level_pool) != pool_shape:
        raise ValueError("post-Update2h carbon pools have the wrong native shape")
    if jnp.shape(state.soil_nitrogen_pool_column_level_pool) != pool_shape:
        raise ValueError("post-Update2h nitrogen pools have the wrong native shape")

    fluxes = fire_flux_result.cn_fluxes
    if fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("post-fire packed fluxes have the wrong patch extent")
    if fluxes.column_level.shape[:2] != (
        column_count,
        NUMBER_OF_DECOMPOSITION_LEVELS,
    ):
        raise ValueError("post-fire packed fluxes have the wrong column-level extent")
    if fluxes.column_level_pool.shape[:3] != pool_shape:
        raise ValueError("post-fire packed fluxes have the wrong column-level-pool extent")

    return CnStateUpdate3Call(
        inputs=native.inputs,
        state=state,
        cn_fluxes=fluxes,
        parameters=native.parameters,
    )


def assemble_cn_state_update3_with_precision_call(
    native: CnStateUpdate3NativeData,
    state_update2h_result: CnStateUpdate2hWithPrecisionResult,
    fire_flux_result: CnFireFluxResult,
) -> CnStateUpdate3WithPrecisionCall:

    call = assemble_cn_state_update3_call(
        native,
        state_update2h_result,
        fire_flux_result,
    )
    if native.precision_inputs.soil_patch.shape != native.inputs.soil_patch.shape:
        raise ValueError("Update3 and precision-control patch extents differ")
    return CnStateUpdate3WithPrecisionCall(
        inputs=call.inputs,
        state=call.state,
        cn_fluxes=call.cn_fluxes,
        parameters=call.parameters,
        precision_inputs=native.precision_inputs,
        precision_parameters=native.precision_parameters,
    )


__all__ = [
    "CnStateUpdate3Call",
    "CnStateUpdate3NativeData",
    "CnStateUpdate3WithPrecisionCall",
    "assemble_cn_state_update3_call",
    "assemble_cn_state_update3_with_precision_call",
    "load_cn_state_update3_native_data",
]
