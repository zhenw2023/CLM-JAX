
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.cn_state_update2 import (
    CnStateUpdate2NativeData,
    load_cn_state_update2_native_data,
)
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import load_subgrid_topology
from clm_jax.physics.cn_state_update2 import CnStateUpdate2hWithPrecisionResult
from clm_jax.physics.cn_wood_products import (
    MAXIMUM_PATCHES_PER_GRIDCELL,
    CnProductSpeciesState,
    CnWoodProductsInputs,
    CnWoodProductsParameters,
    CnWoodProductsState,
)
from clm_jax.physics.cn_zero import CnTransientFluxState, get_cn_flux_field


class CnWoodProductsNativeData(NamedTuple):

    inputs: CnWoodProductsInputs
    restart_state: CnWoodProductsState
    parameters: CnWoodProductsParameters
    fixed_landcover: bool
    restart_path: Path
    parameter_data_path: Path


class CnWoodProductsCall(NamedTuple):

    inputs: CnWoodProductsInputs
    state: CnWoodProductsState
    cn_fluxes: CnTransientFluxState
    parameters: CnWoodProductsParameters


_RESTART_FIELDS = (
    "cropprod1c_g",
    "prod10c_g",
    "prod100c_g",
    "cropprod1n_g",
    "prod10n_g",
    "prod100n_g",
)
_PARAMETER_FIELDS = ("pprodharv10", "pprod10", "pprod100")
_SAME_STEP_PATCH_FLUX_FIELDS = (
    "vegetation_carbon.wood_harvestc_patch",
    "vegetation_carbon.grainc_to_cropprodc_patch",
    "vegetation_nitrogen.wood_harvestn_patch",
    "vegetation_nitrogen.grainn_to_cropprodn_patch",
)


def _require_same_selected_file(
    name: str,
    requested: str | Path | None,
    actual: Path,
) -> None:
    if requested is None:
        return
    if Path(requested).expanduser().resolve() != actual:
        raise ValueError(f"native product dependency uses a different {name} file")


def _validate_fixed_landcover(case: CimeCase, update2: CnStateUpdate2NativeData) -> None:
    clm = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    dynamic = dict(read_case_namelist_group(case, "lnd_in", "dynamic_subgrid"))
    required = {
        "use_cn": True,
        "use_crop": True,
        "use_grainproduct": True,
        "use_fates": False,
    }
    for name, expected in required.items():
        if clm.get(name) != expected:
            raise ValueError(
                f"native wood-products adapter requires {name}={expected!r}"
            )
    landuse = clm.get("flanduse_timeseries")
    if landuse not in (None, "", "null"):
        raise NotImplementedError(
            "transient land-use product gains require the same-step dynamic-subgrid result"
        )
    if dynamic.get("do_harvest", False) is not False or update2.do_harvest:
        raise ValueError("native product dependencies disagree on fixed CNHarvest")


def _validate_topology(
    patch_gridcell: np.ndarray,
    patch_active: np.ndarray,
    patch_weight: np.ndarray,
    gridcell_count: int,
) -> None:
    if patch_gridcell.ndim != 1:
        raise ValueError("restart patch gridcell indices must be a vector")
    if np.any((patch_gridcell < 0) | (patch_gridcell >= gridcell_count)):
        raise ValueError("restart patch gridcell index lies outside the grid")
    if np.any(patch_gridcell[1:] < patch_gridcell[:-1]):
        raise ValueError("native product reduction requires contiguous gridcell patches")
    if patch_active.shape != patch_gridcell.shape or patch_weight.shape != patch_gridcell.shape:
        raise ValueError("restart patch topology fields have inconsistent extents")
    if not np.all(np.isfinite(patch_weight)) or np.any(patch_weight < 0.0):
        raise ValueError("restart patch gridcell weights must be finite and nonnegative")

    patch_counts = np.bincount(patch_gridcell, minlength=gridcell_count)
    if int(np.max(patch_counts, initial=0)) > MAXIMUM_PATCHES_PER_GRIDCELL:
        raise ValueError(
            "restart exceeds the compiled product patch-per-gridcell bound"
        )
    contributing_weight = np.bincount(
        patch_gridcell,
        weights=np.where(patch_active, patch_weight, 0.0),
        minlength=gridcell_count,
    )
    if np.any(contributing_weight > 1.0 + 1.0e-6):
        raise ValueError("restart patch weights violate the production p2g bound")


def _species_state(
    values: dict[str, jnp.ndarray],
    species: str,
    gridcell_count: int,
) -> CnProductSpeciesState:
    names = (
        f"cropprod1{species}_g",
        f"prod10{species}_g",
        f"prod100{species}_g",
    )
    arrays = tuple(np.asarray(values[name], dtype=np.float64) for name in names)
    for name, array in zip(names, arrays, strict=True):
        if array.shape != (gridcell_count,) or not np.all(np.isfinite(array)):
            raise ValueError(f"restart product pool {name!r} is not one finite grid vector")
        if np.any(array < 0.0):
            raise ValueError(f"restart product pool {name!r} contains negative mass")
    return CnProductSpeciesState(*(jnp.asarray(array) for array in arrays))


def load_cn_wood_products_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    state_update2_native_data: CnStateUpdate2NativeData | None = None,
) -> CnWoodProductsNativeData:

    update2 = state_update2_native_data
    if update2 is None:
        update2 = load_cn_state_update2_native_data(
            case,
            restart_path=restart_path,
            parameter_data_path=parameter_data_path,
        )
    else:
        _require_same_selected_file("restart", restart_path, update2.restart_path)
        _require_same_selected_file(
            "parameter-data",
            parameter_data_path,
            update2.parameter_data_path,
        )
    _validate_fixed_landcover(case, update2)

    topology = load_subgrid_topology(update2.restart_path)
    patch_gridcell = np.asarray(topology.patch.gridcell, dtype=np.int32)
    patch_active = np.asarray(topology.patch.active, dtype=bool)
    patch_weight = np.asarray(topology.patch.weight_on_gridcell, dtype=np.float64)
    vegetation_type = np.asarray(topology.patch.vegetation_type, dtype=np.int32)
    gridcell_count = int(topology.grid.latitude_degrees.shape[0])
    patch_count = patch_gridcell.size
    _validate_topology(
        patch_gridcell,
        patch_active,
        patch_weight,
        gridcell_count,
    )
    if update2.inputs.soil_patch.shape != (patch_count,):
        raise ValueError("Update2 and product boundaries have different patch extents")

    restart_values = load_native_variables(
        update2.restart_path,
        list(_RESTART_FIELDS),
        dtype=np.float64,
    )
    parameter_values = load_native_variables(
        update2.parameter_data_path,
        list(_PARAMETER_FIELDS),
        dtype=np.float64,
        masked_floating_fill_value=0.0,
    )
    parameters: dict[str, jnp.ndarray] = {}
    pft_count: int | None = None
    for name in _PARAMETER_FIELDS:
        host = np.asarray(parameter_values[name], dtype=np.float64)
        if host.ndim != 1 or host.size == 0 or not np.all(np.isfinite(host)):
            raise ValueError(f"CLM product parameter {name!r} must be a finite PFT vector")
        if np.any((host < 0.0) | (host > 1.0)):
            raise ValueError(f"CLM product parameter {name!r} must lie in [0, 1]")
        if pft_count is None:
            pft_count = host.size
        elif host.size != pft_count:
            raise ValueError("CLM product parameter vectors have different extents")
        parameters[name] = jnp.asarray(host)
    if pft_count is None or np.any(
        (vegetation_type < 0) | (vegetation_type >= pft_count)
    ):
        raise ValueError("restart contains a PFT outside the product parameter file")

    zero_dynamic_flux = jnp.zeros(patch_count, dtype=jnp.float64)
    return CnWoodProductsNativeData(
        inputs=CnWoodProductsInputs(
            soil_patch=update2.inputs.soil_patch,
            patch_active=topology.patch.active,
            patch_gridcell_index=topology.patch.gridcell,
            patch_weight_on_gridcell=topology.patch.weight_on_gridcell,
            vegetation_type_patch=topology.patch.vegetation_type,
            dwt_wood_product_carbon_gain_patch=zero_dynamic_flux,
            dwt_crop_product_carbon_gain_patch=zero_dynamic_flux,
            dwt_wood_product_nitrogen_gain_patch=zero_dynamic_flux,
            dwt_crop_product_nitrogen_gain_patch=zero_dynamic_flux,
        ),
        restart_state=CnWoodProductsState(
            carbon=_species_state(restart_values, "c", gridcell_count),
            nitrogen=_species_state(restart_values, "n", gridcell_count),
        ),
        parameters=CnWoodProductsParameters(
            wood_harvest_10year_fraction_pft=parameters["pprodharv10"],
            dynamic_wood_10year_fraction_pft=parameters["pprod10"],
            dynamic_wood_100year_fraction_pft=parameters["pprod100"],
            time_step=update2.parameters.time_step,
        ),
        fixed_landcover=True,
        restart_path=update2.restart_path,
        parameter_data_path=update2.parameter_data_path,
    )


def assemble_cn_wood_products_call(
    native: CnWoodProductsNativeData,
    state_update2h_result: CnStateUpdate2hWithPrecisionResult,
    *,
    current_state: CnWoodProductsState | None = None,
) -> CnWoodProductsCall:

    patch_count = native.inputs.soil_patch.shape[0]
    gridcell_count = native.restart_state.carbon.crop_product_1year_gridcell.shape[0]
    if state_update2h_result.state.vegetation.paired_carbon_patch.shape[0] != patch_count:
        raise ValueError("post-Update2h vegetation result has the wrong patch count")
    fluxes = state_update2h_result.cn_fluxes
    if fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("post-Update2h flux result has the wrong patch count")
    for field_name in _SAME_STEP_PATCH_FLUX_FIELDS:
        if get_cn_flux_field(fluxes, field_name).shape != (patch_count,):
            raise ValueError(f"same-step product flux {field_name!r} has the wrong shape")

    state = native.restart_state if current_state is None else current_state
    for species in state:
        for pool in species:
            if jnp.shape(pool) != (gridcell_count,):
                raise ValueError("current product state has the wrong gridcell extent")
    return CnWoodProductsCall(
        inputs=native.inputs,
        state=state,
        cn_fluxes=fluxes,
        parameters=native.parameters,
    )


__all__ = [
    "CnWoodProductsCall",
    "CnWoodProductsNativeData",
    "assemble_cn_wood_products_call",
    "load_cn_wood_products_native_data",
]
