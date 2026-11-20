
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.litter_to_column import (
    LitterToColumnNativeData,
    load_litter_to_column_native_data,
)
from clm_jax.io.netcdf import load_native_variables
from clm_jax.physics.cn_zero import CnTransientFluxState
from clm_jax.physics.growth_respiration import (
    GrowthRespirationInputs,
    GrowthRespirationParameters,
)
from clm_jax.physics.litter_to_column import LitterToColumnResult


class GrowthRespirationNativeData(NamedTuple):

    inputs: GrowthRespirationInputs
    parameters: GrowthRespirationParameters
    number_of_columns: int
    restart_path: Path
    parameter_data_path: Path


class GrowthRespirationCall(NamedTuple):

    inputs: GrowthRespirationInputs
    cn_fluxes: CnTransientFluxState
    parameters: GrowthRespirationParameters


def _require_same_selected_file(
    name: str,
    requested: str | Path | None,
    actual: Path,
) -> None:
    if requested is None:
        return
    if Path(requested).expanduser().resolve() != actual:
        raise ValueError(f"litter native data uses a different {name} file")


def _pft_parameter(
    values: dict[str, jnp.ndarray],
    name: str,
    pft_count: int,
) -> np.ndarray:
    host = np.asarray(values[name], dtype=np.float64)
    if host.shape != (pft_count,) or not np.all(np.isfinite(host)):
        raise ValueError(f"CLM parameter {name!r} must be one finite PFT vector")
    return host


def _soil_patch_filter(
    litter: LitterToColumnNativeData,
) -> np.ndarray:
    persistent = litter.persistent_inputs
    matrix = np.asarray(persistent.column_patch_indices, dtype=np.int32)
    patch_active = np.asarray(persistent.patch_active, dtype=bool)
    soil_column = np.asarray(persistent.soil_column, dtype=bool)
    patch_count = persistent.vegetation_type_patch.shape[0]
    if matrix.ndim != 2 or matrix.shape[0] != soil_column.size:
        raise ValueError("litter topology has an invalid column-patch matrix")
    flat_patch = matrix.reshape(-1)
    valid = flat_patch >= 0
    represented = flat_patch[valid]
    if represented.size != patch_count or not np.array_equal(
        np.sort(represented),
        np.arange(patch_count, dtype=np.int32),
    ):
        raise ValueError("litter topology must represent every patch exactly once")
    flat_column = np.repeat(
        np.arange(matrix.shape[0], dtype=np.int32),
        matrix.shape[1],
    )
    soil_patch = np.zeros(patch_count, dtype=bool)
    soil_patch[represented] = (
        soil_column[flat_column[valid]] & patch_active[represented]
    )
    return soil_patch


def load_growth_respiration_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    litter_native_data: LitterToColumnNativeData | None = None,
) -> GrowthRespirationNativeData:

    litter = litter_native_data
    if litter is None:
        litter = load_litter_to_column_native_data(
            case,
            restart_path=restart_path,
            parameter_data_path=parameter_data_path,
        )
    else:
        _require_same_selected_file("restart", restart_path, litter.restart_path)
        _require_same_selected_file(
            "parameter-data",
            parameter_data_path,
            litter.parameter_data_path,
        )

    pft_count = litter.parameters.leaf_labile_fraction_pft.shape[0]
    values = load_native_variables(
        litter.parameter_data_path,
        ["woody", "grperc", "grpnow"],
        dtype=np.float64,
        masked_floating_fill_value=0.0,
    )
    woody = _pft_parameter(values, "woody", pft_count)
    growth_fraction = _pft_parameter(values, "grperc", pft_count)
    immediate_fraction = _pft_parameter(values, "grpnow", pft_count)
    if not np.all((woody == 0.0) | (woody == 1.0)):
        raise ValueError("CLM parameter 'woody' must contain only zero or one")
    if np.any(growth_fraction < 0.0):
        raise ValueError("CLM parameter 'grperc' must be nonnegative")
    if np.any((immediate_fraction < 0.0) | (immediate_fraction > 1.0)):
        raise ValueError("CLM parameter 'grpnow' must lie in [0, 1]")

    persistent = litter.persistent_inputs
    return GrowthRespirationNativeData(
        inputs=GrowthRespirationInputs(
            soil_patch=jnp.asarray(_soil_patch_filter(litter)),
            vegetation_type_patch=persistent.vegetation_type_patch,
        ),
        parameters=GrowthRespirationParameters(
            woody_pft=jnp.asarray(woody),
            growth_respiration_fraction_pft=jnp.asarray(growth_fraction),
            immediate_growth_respiration_fraction_pft=jnp.asarray(
                immediate_fraction
            ),
            first_prognostic_crop_pft=(
                litter.parameters.first_prognostic_crop_pft
            ),
        ),
        number_of_columns=persistent.soil_column.shape[0],
        restart_path=litter.restart_path,
        parameter_data_path=litter.parameter_data_path,
    )


def assemble_growth_respiration_call(
    native: GrowthRespirationNativeData,
    litter_to_column_result: LitterToColumnResult,
) -> GrowthRespirationCall:

    cn_fluxes = litter_to_column_result.cn_fluxes
    patch_count = native.inputs.soil_patch.shape[0]
    if cn_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("litter result has the wrong patch count")
    if cn_fluxes.column_scalar.shape[0] != native.number_of_columns:
        raise ValueError("litter result has the wrong column count")
    return GrowthRespirationCall(
        inputs=native.inputs,
        cn_fluxes=cn_fluxes,
        parameters=native.parameters,
    )


__all__ = [
    "GrowthRespirationCall",
    "GrowthRespirationNativeData",
    "assemble_growth_respiration_call",
    "load_growth_respiration_native_data",
]
