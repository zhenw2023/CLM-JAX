
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.background_litterfall import (
    BackgroundLitterfallNativeData,
    load_background_litterfall_native_data,
)
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.background_litterfall import BackgroundLitterfallResult
from clm_jax.physics.cn_zero import CnTransientFluxState
from clm_jax.physics.livewood_turnover import (
    LivewoodTurnoverInputs,
    LivewoodTurnoverParameters,
)

_SECONDS_PER_365_DAY_YEAR = np.float64(31_536_000.0)


class LivewoodTurnoverNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: LivewoodTurnoverInputs
    parameters: LivewoodTurnoverParameters
    restart_path: Path
    parameter_data_path: Path


class LivewoodTurnoverCall(NamedTuple):

    inputs: LivewoodTurnoverInputs
    cn_fluxes: CnTransientFluxState
    parameters: LivewoodTurnoverParameters


def _require_same_selected_file(
    name: str,
    requested: str | Path | None,
    actual: Path,
) -> None:
    if requested is None:
        return
    resolved = Path(requested).expanduser().resolve()
    if resolved != actual:
        raise ValueError(f"background native data uses a different {name} file")


def load_livewood_turnover_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    background_native_data: BackgroundLitterfallNativeData | None = None,
) -> LivewoodTurnoverNativeData:

    background = background_native_data
    if background is None:
        background = load_background_litterfall_native_data(
            case,
            restart_path=restart_path,
            parameter_data_path=parameter_data_path,
        )
    else:
        _require_same_selected_file(
            "restart",
            restart_path,
            background.restart_path,
        )
        _require_same_selected_file(
            "parameter-data",
            parameter_data_path,
            background.parameter_data_path,
        )

    patch_count = background.inputs.soil_patch.shape[0]
    soil = np.asarray(background.inputs.soil_patch, dtype=bool)
    restart_values = load_native_variables(
        background.restart_path,
        ["livestemc", "livecrootc", "livestemn", "livecrootn"],
        dtype=np.float64,
        masked_floating_fill_value=float(CLM_SPECIAL_VALUE),
    )
    for name in ("livestemc", "livecrootc", "livestemn", "livecrootn"):
        host = np.asarray(restart_values[name], dtype=np.float64)
        if host.shape != (patch_count,):
            raise ValueError(f"restart variable {name!r} must contain every patch")
        if not np.all(np.isfinite(host[soil])):
            raise ValueError(f"active soil patches contain non-finite {name!r}")

    parameter_values = load_native_variables(
        background.parameter_data_path,
        ["woody", "livewdcn", "deadwdcn", "lwtop_ann"],
        dtype=np.float64,
        masked_floating_fill_value=0.0,
    )
    pft_count = int(
        background.parameters.leaf_carbon_nitrogen_ratio_pft.shape[0]
    )
    woody = np.asarray(parameter_values["woody"], dtype=np.float64)
    live_ratio = np.asarray(parameter_values["livewdcn"], dtype=np.float64)
    dead_ratio = np.asarray(parameter_values["deadwdcn"], dtype=np.float64)
    for name, values in (
        ("woody", woody),
        ("livewdcn", live_ratio),
        ("deadwdcn", dead_ratio),
    ):
        if values.shape != (pft_count,) or not np.all(np.isfinite(values)):
            raise ValueError(f"CLM parameter {name!r} must be one finite PFT vector")
    if not np.all((woody == 0.0) | (woody == 1.0)):
        raise ValueError("CLM parameter 'woody' must contain only zero or one")
    woody_pft = woody > 0.0
    if np.any(live_ratio[woody_pft] <= 0.0):
        raise ValueError("CLM livewdcn must be positive for every woody PFT")
    if np.any(dead_ratio[woody_pft] <= 0.0):
        raise ValueError("CLM deadwdcn must be positive for every woody PFT")

    annual_turnover = np.asarray(
        parameter_values["lwtop_ann"],
        dtype=np.float64,
    )
    if annual_turnover.size != 1:
        raise ValueError("CLM parameter 'lwtop_ann' must contain one scalar")
    annual_turnover = np.float64(annual_turnover.reshape(()))
    if not np.isfinite(annual_turnover) or annual_turnover < 0.0:
        raise ValueError("CLM parameter 'lwtop_ann' must be finite and nonnegative")
    turnover_rate = np.float64(annual_turnover / _SECONDS_PER_365_DAY_YEAR)

    return LivewoodTurnoverNativeData(
        topology=background.topology,
        inputs=LivewoodTurnoverInputs(
            soil_patch=background.inputs.soil_patch,
            vegetation_type_patch=background.inputs.vegetation_type_patch,
            live_stem_carbon_patch=jnp.asarray(restart_values["livestemc"]),
            live_coarse_root_carbon_patch=jnp.asarray(
                restart_values["livecrootc"]
            ),
            live_stem_nitrogen_patch=jnp.asarray(restart_values["livestemn"]),
            live_coarse_root_nitrogen_patch=jnp.asarray(
                restart_values["livecrootn"]
            ),
        ),
        parameters=LivewoodTurnoverParameters(
            woody_pft=jnp.asarray(woody),
            live_wood_carbon_nitrogen_ratio_pft=jnp.asarray(live_ratio),
            dead_wood_carbon_nitrogen_ratio_pft=jnp.asarray(dead_ratio),
            live_wood_turnover_rate=turnover_rate,
            use_fun=background.parameters.use_fun,
            carbon_nitrogen_ratio_floating=(
                background.parameters.carbon_nitrogen_ratio_floating
            ),
        ),
        restart_path=background.restart_path,
        parameter_data_path=background.parameter_data_path,
    )


def assemble_livewood_turnover_call(
    native: LivewoodTurnoverNativeData,
    background_litterfall_result: BackgroundLitterfallResult,
) -> LivewoodTurnoverCall:

    patch_count = native.inputs.soil_patch.shape[0]
    final_fluxes = background_litterfall_result.cn_fluxes
    if final_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("background-litterfall result has the wrong patch count")
    return LivewoodTurnoverCall(
        inputs=native.inputs,
        cn_fluxes=final_fluxes,
        parameters=native.parameters,
    )


__all__ = [
    "LivewoodTurnoverCall",
    "LivewoodTurnoverNativeData",
    "assemble_livewood_turnover_call",
    "load_livewood_turnover_native_data",
]
