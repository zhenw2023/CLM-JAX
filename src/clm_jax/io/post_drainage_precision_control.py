
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.cn_n_state_update3 import (
    CnNStateUpdate3NativeData,
    load_cn_n_state_update3_native_data,
)
from clm_jax.io.soil_biogeochem_precision_control import (
    SoilBiogeochemPrecisionControlNativeData,
    SoilBiogeochemPrecisionControlPersistentState,
    assemble_soil_biogeochem_precision_control_call,
    load_soil_biogeochem_precision_control_native_data,
)
from clm_jax.physics.cn_n_state_update3 import CnNStateUpdate3Result
from clm_jax.physics.cn_state_update3 import CnStateUpdate3WithPrecisionResult
from clm_jax.physics.post_drainage_precision_control import (
    PostDrainagePrecisionControlInputs,
    PostDrainagePrecisionControlParameters,
    PostDrainagePrecisionControlState,
)


class PostDrainagePrecisionControlNativeData(NamedTuple):

    inputs: PostDrainagePrecisionControlInputs
    initial_soil_persistent_state: SoilBiogeochemPrecisionControlPersistentState
    parameters: PostDrainagePrecisionControlParameters
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path


class PostDrainagePrecisionControlCall(NamedTuple):
    inputs: PostDrainagePrecisionControlInputs
    state: PostDrainagePrecisionControlState
    parameters: PostDrainagePrecisionControlParameters


def load_post_drainage_precision_control_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    n_state_update3_native_data: CnNStateUpdate3NativeData | None = None,
    soil_precision_native_data: (
        SoilBiogeochemPrecisionControlNativeData | None
    ) = None,
) -> PostDrainagePrecisionControlNativeData:

    nitrogen = n_state_update3_native_data
    if nitrogen is None:
        nitrogen = load_cn_n_state_update3_native_data(
            case,
            restart_path=restart_path,
            surface_data_path=surface_data_path,
            parameter_data_path=parameter_data_path,
        )
    soil = soil_precision_native_data
    if soil is None:
        soil = load_soil_biogeochem_precision_control_native_data(
            case,
            restart_path=nitrogen.restart_path,
            surface_data_path=nitrogen.surface_data_path,
            parameter_data_path=nitrogen.parameter_data_path,
            n_state_update3_native_data=nitrogen,
        )
    if soil.restart_path != nitrogen.restart_path:
        raise ValueError("vegetation and soil precision controls use different restarts")
    if soil.surface_data_path != nitrogen.surface_data_path:
        raise ValueError("vegetation and soil precision controls use different surface data")
    if soil.parameter_data_path != nitrogen.parameter_data_path:
        raise ValueError("vegetation and soil precision controls use different parameter files")
    if not np.array_equal(
        np.asarray(nitrogen.precision_inputs.soil_patch, dtype=bool),
        np.asarray(nitrogen.inputs.soil_patch, dtype=bool),
    ):
        raise ValueError("post-drainage vegetation guardrail uses a different patch filter")
    if not np.array_equal(
        np.asarray(soil.inputs.soil_column, dtype=bool),
        np.asarray(nitrogen.inputs.soil_column, dtype=bool),
    ):
        raise ValueError("post-drainage soil guardrail uses a different column filter")

    return PostDrainagePrecisionControlNativeData(
        inputs=PostDrainagePrecisionControlInputs(
            vegetation=nitrogen.precision_inputs,
            soil=soil.inputs,
        ),
        initial_soil_persistent_state=soil.initial_persistent_state,
        parameters=PostDrainagePrecisionControlParameters(
            vegetation=nitrogen.precision_parameters,
            soil=soil.parameters,
        ),
        restart_path=nitrogen.restart_path,
        surface_data_path=nitrogen.surface_data_path,
        parameter_data_path=nitrogen.parameter_data_path,
    )


def assemble_post_drainage_precision_control_call(
    native: PostDrainagePrecisionControlNativeData,
    carbon_state_update3_result: CnStateUpdate3WithPrecisionResult,
    nitrogen_state_update3_result: CnNStateUpdate3Result,
    soil_persistent_state: SoilBiogeochemPrecisionControlPersistentState,
) -> PostDrainagePrecisionControlCall:

    patch_count = native.inputs.vegetation.soil_patch.shape[0]
    vegetation = nitrogen_state_update3_result.state.vegetation
    if vegetation.paired_carbon_patch.shape[0] != patch_count:
        raise ValueError("same-step post-NUpdate3 vegetation has the wrong patch extent")
    soil_native = SoilBiogeochemPrecisionControlNativeData(
        inputs=native.inputs.soil,
        initial_persistent_state=native.initial_soil_persistent_state,
        parameters=native.parameters.soil,
        restart_path=native.restart_path,
        surface_data_path=native.surface_data_path,
        parameter_data_path=native.parameter_data_path,
    )
    soil_call = assemble_soil_biogeochem_precision_control_call(
        soil_native,
        carbon_state_update3_result,
        nitrogen_state_update3_result,
        soil_persistent_state,
    )
    fluxes = nitrogen_state_update3_result.cn_fluxes
    if fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("same-step post-NUpdate3 fluxes have the wrong patch extent")
    if fluxes.column_scalar.shape[0] != native.inputs.soil.soil_column.shape[0]:
        raise ValueError("same-step post-NUpdate3 fluxes have the wrong column extent")

    return PostDrainagePrecisionControlCall(
        inputs=native.inputs,
        state=PostDrainagePrecisionControlState(
            vegetation=vegetation,
            soil=soil_call.state,
            cn_fluxes=fluxes,
        ),
        parameters=native.parameters,
    )


__all__ = [
    "PostDrainagePrecisionControlCall",
    "PostDrainagePrecisionControlNativeData",
    "assemble_post_drainage_precision_control_call",
    "load_post_drainage_precision_control_native_data",
]
