
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.livewood_turnover import (
    LivewoodTurnoverNativeData,
    load_livewood_turnover_native_data,
)
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.restart import LANDUNIT_CROP, LANDUNIT_SOIL, SubgridTopology
from clm_jax.physics.cn_zero import CnTransientFluxState
from clm_jax.physics.grain_to_product_pools import (
    GrainToProductPoolsInputs,
    GrainToProductPoolsParameters,
)
from clm_jax.physics.livewood_turnover import LivewoodTurnoverResult


class GrainToProductPoolsNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: GrainToProductPoolsInputs
    parameters: GrainToProductPoolsParameters
    restart_path: Path
    parameter_data_path: Path


class GrainToProductPoolsCall(NamedTuple):

    inputs: GrainToProductPoolsInputs
    cn_fluxes: CnTransientFluxState
    parameters: GrainToProductPoolsParameters


def _require_same_selected_file(
    name: str,
    requested: str | Path | None,
    actual: Path,
) -> None:
    if requested is None:
        return
    resolved = Path(requested).expanduser().resolve()
    if resolved != actual:
        raise ValueError(f"livewood native data uses a different {name} file")


def load_grain_to_product_pools_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    livewood_native_data: LivewoodTurnoverNativeData | None = None,
) -> GrainToProductPoolsNativeData:

    livewood = livewood_native_data
    if livewood is None:
        livewood = load_livewood_turnover_native_data(
            case,
            restart_path=restart_path,
            parameter_data_path=parameter_data_path,
        )
    else:
        _require_same_selected_file(
            "restart",
            restart_path,
            livewood.restart_path,
        )
        _require_same_selected_file(
            "parameter-data",
            parameter_data_path,
            livewood.parameter_data_path,
        )

    topology = livewood.topology
    patch_active = np.asarray(topology.patch.active, dtype=bool)
    patch_landunit = np.asarray(topology.patch.landunit_type, dtype=np.int32)
    soil_patch = patch_active & (
        (patch_landunit == LANDUNIT_SOIL) | (patch_landunit == LANDUNIT_CROP)
    )
    column_active = np.asarray(topology.column.active, dtype=bool)
    column_landunit = np.asarray(topology.column.landunit_type, dtype=np.int32)
    soil_column = column_active & (
        (column_landunit == LANDUNIT_SOIL) | (column_landunit == LANDUNIT_CROP)
    )
    patch_column = np.asarray(topology.patch.column, dtype=np.int32)
    patch_weight = np.asarray(topology.patch.weight_on_column, dtype=np.float64)
    column_count = column_active.shape[0]
    if np.any((patch_column < 0) | (patch_column >= column_count)):
        raise ValueError("restart patch-to-column topology is out of bounds")
    if not np.all(np.isfinite(patch_weight[patch_active])):
        raise ValueError("active restart patches contain non-finite column weights")

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    maximum_patches = int(general.get("maxpatch_pft", 0))
    if maximum_patches < 1:
        raise ValueError("native grain-product adapter requires maxpatch_pft >= 1")
    actual_patches = np.bincount(patch_column, minlength=column_count)
    if actual_patches.size and int(np.max(actual_patches)) > maximum_patches:
        raise ValueError("restart topology exceeds configured maxpatch_pft")

    return GrainToProductPoolsNativeData(
        topology=topology,
        inputs=GrainToProductPoolsInputs(
            soil_patch=jnp.asarray(soil_patch),
            soil_column=jnp.asarray(soil_column),
            patch_column_index=jnp.asarray(patch_column),
            patch_weight_on_column=jnp.asarray(patch_weight),
            patch_active=jnp.asarray(patch_active),
        ),
        parameters=GrainToProductPoolsParameters(
            use_crop=bool(general.get("use_crop", False)),
            use_grain_product=bool(general.get("use_grainproduct", False)),
            maximum_patches_per_column=maximum_patches,
        ),
        restart_path=livewood.restart_path,
        parameter_data_path=livewood.parameter_data_path,
    )


def assemble_grain_to_product_pools_call(
    native: GrainToProductPoolsNativeData,
    livewood_turnover_result: LivewoodTurnoverResult,
) -> GrainToProductPoolsCall:

    patch_count = native.inputs.soil_patch.shape[0]
    column_count = native.inputs.soil_column.shape[0]
    final_fluxes = livewood_turnover_result.cn_fluxes
    if final_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("livewood-turnover result has the wrong patch count")
    if final_fluxes.column_scalar.shape[0] != column_count:
        raise ValueError("livewood-turnover result has the wrong column count")
    return GrainToProductPoolsCall(
        inputs=native.inputs,
        cn_fluxes=final_fluxes,
        parameters=native.parameters,
    )


__all__ = [
    "GrainToProductPoolsCall",
    "GrainToProductPoolsNativeData",
    "assemble_grain_to_product_pools_call",
    "load_grain_to_product_pools_native_data",
]
