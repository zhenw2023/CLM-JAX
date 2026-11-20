
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.cn_gap_mortality import (
    CnGapMortalityNativeData,
    load_cn_gap_mortality_native_data,
)
from clm_jax.io.cn_state_update1 import (
    CnStateUpdate1NativeData,
    load_cn_state_update1_native_data,
)
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.soil_biogeochem_litter_vertical_transport import (
    SoilBiogeochemLitterVerticalTransportNativeData,
    load_soil_biogeochem_litter_vertical_transport_native_data,
)
from clm_jax.physics.cn_gap_mortality import CnGapMortalityResult
from clm_jax.physics.cn_precision_control import (
    CnPrecisionControlInputs,
    CnPrecisionControlParameters,
)
from clm_jax.physics.cn_state_update1 import CnStateUpdate1WithPrecisionResult
from clm_jax.physics.cn_state_update2 import (
    CN_STATE_UPDATE2H_FLUX_FIELDS,
    CnStateUpdate2Inputs,
    CnStateUpdate2Parameters,
    CnStateUpdate2Result,
    CnStateUpdate2State,
)
from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    get_cn_flux_field,
)
from clm_jax.physics.cn_zero_manifest import (
    NUMBER_OF_DECOMPOSITION_LEVELS,
    NUMBER_OF_DECOMPOSITION_POOLS,
)
from clm_jax.physics.soil_biogeochem_litter_vertical_transport import (
    SoilBiogeochemLitterVerticalTransportResult,
)


class CnStateUpdate2NativeData(NamedTuple):

    inputs: CnStateUpdate2Inputs
    parameters: CnStateUpdate2Parameters
    precision_inputs: CnPrecisionControlInputs
    precision_parameters: CnPrecisionControlParameters
    do_harvest: bool
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path


class CnStateUpdate2Call(NamedTuple):

    inputs: CnStateUpdate2Inputs
    state: CnStateUpdate2State
    cn_fluxes: CnTransientFluxState
    parameters: CnStateUpdate2Parameters


class CnStateUpdate2hWithPrecisionCall(NamedTuple):

    inputs: CnStateUpdate2Inputs
    state: CnStateUpdate2State
    cn_fluxes: CnTransientFluxState
    parameters: CnStateUpdate2Parameters
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
        raise ValueError(f"native Update2 dependency uses a different {name} file")


def _validate_dependencies(
    gap: CnGapMortalityNativeData,
    vertical: SoilBiogeochemLitterVerticalTransportNativeData,
    update1: CnStateUpdate1NativeData,
) -> None:
    if gap.restart_path != vertical.restart_path:
        raise ValueError("gap mortality and vertical transport use different restarts")
    if gap.surface_data_path != vertical.surface_data_path:
        raise ValueError("gap mortality and vertical transport use different surface data")
    if gap.parameter_data_path != vertical.parameter_data_path:
        raise ValueError("gap mortality and vertical transport use different parameter files")
    if gap.restart_path != update1.restart_path:
        raise ValueError("gap mortality and Update1 use different restarts")
    if gap.parameter_data_path != update1.parameter_data_path:
        raise ValueError("gap mortality and Update1 use different parameter files")
    if not np.array_equal(
        np.asarray(gap.persistent_inputs.soil_column, dtype=bool),
        np.asarray(vertical.soil_column, dtype=bool),
    ):
        raise ValueError("gap mortality and vertical transport disagree on soil columns")
    if not np.array_equal(
        np.asarray(gap.persistent_inputs.soil_patch, dtype=bool),
        np.asarray(update1.inputs.soil_patch, dtype=bool),
    ):
        raise ValueError("gap mortality and Update1 disagree on soil patches")
    if int(gap.parameters.decomposition_level_count) != 20:
        raise ValueError("native Update2 requires 20 active decomposition levels")
    if int(vertical.restart_state.carbon_pool_column_level_pool.shape[1]) != (
        NUMBER_OF_DECOMPOSITION_LEVELS
    ):
        raise ValueError("native Update2 requires all 25 ground levels")


def load_cn_state_update2_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    gap_mortality_native_data: CnGapMortalityNativeData | None = None,
    vertical_transport_native_data: (SoilBiogeochemLitterVerticalTransportNativeData | None) = None,
    state_update1_native_data: CnStateUpdate1NativeData | None = None,
) -> CnStateUpdate2NativeData:

    gap = gap_mortality_native_data
    vertical = vertical_transport_native_data
    update1 = state_update1_native_data
    if gap is None:
        gap = load_cn_gap_mortality_native_data(
            case,
            restart_path=restart_path,
            surface_data_path=surface_data_path,
            parameter_data_path=parameter_data_path,
        )
    else:
        _require_same_selected_file("restart", restart_path, gap.restart_path)
        _require_same_selected_file(
            "surface-data",
            surface_data_path,
            gap.surface_data_path,
        )
        _require_same_selected_file(
            "parameter-data",
            parameter_data_path,
            gap.parameter_data_path,
        )
    if vertical is None:
        vertical = load_soil_biogeochem_litter_vertical_transport_native_data(
            case,
            restart_path=gap.restart_path,
            surface_data_path=gap.surface_data_path,
            parameter_data_path=gap.parameter_data_path,
        )
    else:
        _require_same_selected_file("restart", restart_path, vertical.restart_path)
        _require_same_selected_file(
            "surface-data",
            surface_data_path,
            vertical.surface_data_path,
        )
        _require_same_selected_file(
            "parameter-data",
            parameter_data_path,
            vertical.parameter_data_path,
        )
    if update1 is None:
        update1 = load_cn_state_update1_native_data(
            case,
            restart_path=gap.restart_path,
            parameter_data_path=gap.parameter_data_path,
        )
    else:
        _require_same_selected_file("restart", restart_path, update1.restart_path)
        _require_same_selected_file(
            "parameter-data",
            parameter_data_path,
            update1.parameter_data_path,
        )
    _validate_dependencies(gap, vertical, update1)

    dynamic_subgrid = dict(
        read_case_namelist_group(case, "lnd_in", "dynamic_subgrid")
    )
    clm = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    do_harvest = _logical(dynamic_subgrid, "do_harvest", False)
    if do_harvest:
        raise NotImplementedError(
            "native Update2h currently requires inactive dynamic CNHarvest"
        )
    if not _logical(clm, "use_nguardrail", False):
        raise ValueError(
            "native Update2h composition requires the active third CN guardrail"
        )

    return CnStateUpdate2NativeData(
        inputs=CnStateUpdate2Inputs(
            soil_patch=gap.persistent_inputs.soil_patch,
            soil_column=gap.persistent_inputs.soil_column,
        ),
        parameters=CnStateUpdate2Parameters(
            time_step=vertical.parameters.time_step,
            decomposition_level_count=gap.parameters.decomposition_level_count,
        ),
        precision_inputs=CnPrecisionControlInputs(
            soil_patch=update1.inputs.soil_patch,
            vegetation_type_patch=update1.inputs.vegetation_type_patch,
        ),
        precision_parameters=update1.precision_parameters,
        do_harvest=do_harvest,
        restart_path=gap.restart_path,
        surface_data_path=gap.surface_data_path,
        parameter_data_path=gap.parameter_data_path,
    )


def assemble_cn_state_update2_call(
    native: CnStateUpdate2NativeData,
    state_update1_result: CnStateUpdate1WithPrecisionResult,
    vertical_transport_result: SoilBiogeochemLitterVerticalTransportResult,
    gap_mortality_result: CnGapMortalityResult,
) -> CnStateUpdate2Call:

    patch_count = native.inputs.soil_patch.shape[0]
    column_count = native.inputs.soil_column.shape[0]
    vegetation = state_update1_result.state.vegetation
    if vegetation.paired_carbon_patch.shape[0] != patch_count:
        raise ValueError("post-Update1 vegetation result has the wrong patch count")
    vertical_state = vertical_transport_result.state
    pool_shape = (
        column_count,
        NUMBER_OF_DECOMPOSITION_LEVELS,
        NUMBER_OF_DECOMPOSITION_POOLS,
    )
    for name in (
        "carbon_pool_column_level_pool",
        "nitrogen_pool_column_level_pool",
    ):
        if getattr(vertical_state, name).shape != pool_shape:
            raise ValueError(f"vertical-transport {name} has the wrong shape")
    fluxes = gap_mortality_result.cn_fluxes
    if fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("gap-mortality flux result has the wrong patch count")
    if fluxes.column_level.shape[:2] != (
        column_count,
        NUMBER_OF_DECOMPOSITION_LEVELS,
    ):
        raise ValueError("gap-mortality flux result has the wrong column extent")

    return CnStateUpdate2Call(
        inputs=native.inputs,
        state=CnStateUpdate2State(
            vegetation=vegetation,
            soil_carbon_pool_column_level_pool=(vertical_state.carbon_pool_column_level_pool),
            soil_nitrogen_pool_column_level_pool=(vertical_state.nitrogen_pool_column_level_pool),
        ),
        cn_fluxes=fluxes,
        parameters=native.parameters,
    )


def assemble_cn_state_update2h_with_precision_call(
    native: CnStateUpdate2NativeData,
    state_update2_result: CnStateUpdate2Result,
) -> CnStateUpdate2hWithPrecisionCall:

    patch_count = native.inputs.soil_patch.shape[0]
    column_count = native.inputs.soil_column.shape[0]
    state = state_update2_result.state
    if state.vegetation.paired_carbon_patch.shape[0] != patch_count:
        raise ValueError("post-Update2 vegetation result has the wrong patch count")
    expected_pool_shape = (
        column_count,
        NUMBER_OF_DECOMPOSITION_LEVELS,
        NUMBER_OF_DECOMPOSITION_POOLS,
    )
    if state.soil_carbon_pool_column_level_pool.shape != expected_pool_shape:
        raise ValueError("post-Update2 carbon pools have the wrong shape")
    if state.soil_nitrogen_pool_column_level_pool.shape != expected_pool_shape:
        raise ValueError("post-Update2 nitrogen pools have the wrong shape")
    fluxes = state_update2_result.cn_fluxes
    if fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("post-Update2 flux result has the wrong patch count")
    if fluxes.column_level.shape[:2] != (
        column_count,
        NUMBER_OF_DECOMPOSITION_LEVELS,
    ):
        raise ValueError("post-Update2 flux result has the wrong column extent")
    if not native.do_harvest:
        nonzero = tuple(
            name
            for name in CN_STATE_UPDATE2H_FLUX_FIELDS
            if np.any(np.asarray(get_cn_flux_field(fluxes, name)) != 0.0)
        )
        if nonzero:
            raise ValueError(
                "inactive CNHarvest boundary contains nonzero harvest fluxes: "
                + ", ".join(nonzero)
            )

    return CnStateUpdate2hWithPrecisionCall(
        inputs=native.inputs,
        state=state,
        cn_fluxes=fluxes,
        parameters=native.parameters,
        precision_inputs=native.precision_inputs,
        precision_parameters=native.precision_parameters,
    )


__all__ = [
    "CnStateUpdate2Call",
    "CnStateUpdate2NativeData",
    "CnStateUpdate2hWithPrecisionCall",
    "assemble_cn_state_update2_call",
    "assemble_cn_state_update2h_with_precision_call",
    "load_cn_state_update2_native_data",
]
