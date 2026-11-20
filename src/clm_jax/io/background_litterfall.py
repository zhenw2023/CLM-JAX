
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.offset_litterfall import (
    OffsetLitterfallNativeData,
    load_offset_litterfall_native_data,
)
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.background_litterfall import (
    BackgroundLitterfallInputs,
    BackgroundLitterfallParameters,
    BackgroundLitterfallState,
)
from clm_jax.physics.cn_zero import CnTransientFluxState
from clm_jax.physics.offset_litterfall import OffsetLitterfallResult


class BackgroundLitterfallNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: BackgroundLitterfallInputs
    state: BackgroundLitterfallState
    parameters: BackgroundLitterfallParameters
    restart_path: Path
    parameter_data_path: Path


class BackgroundLitterfallCall(NamedTuple):

    inputs: BackgroundLitterfallInputs
    state: BackgroundLitterfallState
    cn_fluxes: CnTransientFluxState
    parameters: BackgroundLitterfallParameters


def _require_same_selected_file(
    name: str,
    requested: str | Path | None,
    actual: Path,
) -> None:
    if requested is None:
        return
    resolved = Path(requested).expanduser().resolve()
    if resolved != actual:
        raise ValueError(f"offset native data uses a different {name} file")


def load_background_litterfall_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    offset_native_data: OffsetLitterfallNativeData | None = None,
) -> BackgroundLitterfallNativeData:

    offset = offset_native_data
    if offset is None:
        offset = load_offset_litterfall_native_data(
            case,
            restart_path=restart_path,
            parameter_data_path=parameter_data_path,
        )
    else:
        _require_same_selected_file("restart", restart_path, offset.restart_path)
        _require_same_selected_file(
            "parameter-data",
            parameter_data_path,
            offset.parameter_data_path,
        )

    patch_count = offset.inputs.soil_patch.shape[0]
    soil = np.asarray(offset.inputs.soil_patch, dtype=bool)
    restart_values = load_native_variables(
        offset.restart_path,
        ["bglfr"],
        dtype=np.float64,
        masked_floating_fill_value=float(CLM_SPECIAL_VALUE),
    )
    background_rate = np.asarray(restart_values["bglfr"], dtype=np.float64)
    if background_rate.shape != (patch_count,):
        raise ValueError("restart bglfr must contain every patch")
    if not np.all(np.isfinite(background_rate[soil])):
        raise ValueError("active soil patches contain non-finite bglfr")

    offset_inputs = offset.inputs
    offset_state = offset.state
    offset_parameters = offset.parameters
    return BackgroundLitterfallNativeData(
        topology=offset.topology,
        inputs=BackgroundLitterfallInputs(
            soil_patch=offset_inputs.soil_patch,
            vegetation_type_patch=offset_inputs.vegetation_type_patch,
            background_litterfall_rate_patch=jnp.asarray(background_rate),
            leaf_carbon_patch=offset_inputs.leaf_carbon_patch,
            fine_root_carbon_patch=offset_inputs.fine_root_carbon_patch,
            leaf_nitrogen_patch=offset_inputs.leaf_nitrogen_patch,
            fine_root_nitrogen_patch=offset_inputs.fine_root_nitrogen_patch,
        ),
        state=BackgroundLitterfallState(
            leaf_carbon_to_litter_fun_patch=(
                offset_state.leaf_carbon_to_litter_fun_patch
            ),
            leaf_carbon_nitrogen_offset_patch=(
                offset_state.leaf_carbon_nitrogen_offset_patch
            ),
        ),
        parameters=BackgroundLitterfallParameters(
            leaf_carbon_nitrogen_ratio_pft=(
                offset_parameters.leaf_carbon_nitrogen_ratio_pft
            ),
            leaf_litter_carbon_nitrogen_ratio_pft=(
                offset_parameters.leaf_litter_carbon_nitrogen_ratio_pft
            ),
            fine_root_carbon_nitrogen_ratio_pft=(
                offset_parameters.fine_root_carbon_nitrogen_ratio_pft
            ),
            use_fun=offset_parameters.use_fun,
            carbon_nitrogen_ratio_floating=(
                offset_parameters.carbon_nitrogen_ratio_floating
            ),
            time_step_seconds=offset_parameters.time_step_seconds,
        ),
        restart_path=offset.restart_path,
        parameter_data_path=offset.parameter_data_path,
    )


def assemble_background_litterfall_call(
    native: BackgroundLitterfallNativeData,
    offset_litterfall_result: OffsetLitterfallResult,
) -> BackgroundLitterfallCall:

    patch_count = native.inputs.soil_patch.shape[0]
    final_fluxes = offset_litterfall_result.cn_fluxes
    if final_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("offset-litterfall result has the wrong patch count")
    offset_state = offset_litterfall_result.state
    for name in (
        "leaf_carbon_to_litter_fun_patch",
        "leaf_carbon_nitrogen_offset_patch",
    ):
        if jnp.shape(getattr(offset_state, name)) != (patch_count,):
            raise ValueError("offset-litterfall state has the wrong patch count")
    return BackgroundLitterfallCall(
        inputs=native.inputs,
        state=BackgroundLitterfallState(
            leaf_carbon_to_litter_fun_patch=(
                offset_state.leaf_carbon_to_litter_fun_patch
            ),
            leaf_carbon_nitrogen_offset_patch=(
                offset_state.leaf_carbon_nitrogen_offset_patch
            ),
        ),
        cn_fluxes=final_fluxes,
        parameters=native.parameters,
    )


__all__ = [
    "BackgroundLitterfallCall",
    "BackgroundLitterfallNativeData",
    "assemble_background_litterfall_call",
    "load_background_litterfall_native_data",
]
