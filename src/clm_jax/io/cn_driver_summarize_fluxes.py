
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.cn_driver_summarize_states import (
    CnDriverSummarizeStatesNativeData,
    load_cn_driver_summarize_states_native_data,
)
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import load_restart_clock
from clm_jax.physics.bgc_timestep_initialization import (
    DynamicLandcoverCarbonFluxState,
)
from clm_jax.physics.cn_driver_summarize_fluxes import (
    CnDriverSummarizeFluxesInputs,
    CnDriverSummarizeFluxesParameters,
    CnDriverSummarizeFluxesState,
)
from clm_jax.physics.cn_wood_products import CnWoodProductsResult
from clm_jax.physics.post_drainage_precision_control import (
    PostDrainagePrecisionControlResult,
)


class CnDriverSummarizeFluxesPersistentState(NamedTuple):

    temporary_npp_sum_patch: jnp.ndarray
    lagged_npp_column: jnp.ndarray
    dynamic_landuse_amount_to_dribble_gridcell: jnp.ndarray
    harvest_xsmrpool_amount_to_dribble_gridcell: jnp.ndarray


class CnDriverSummarizeFluxesNativeData(NamedTuple):

    inputs: CnDriverSummarizeFluxesInputs
    persistent_state: CnDriverSummarizeFluxesPersistentState
    parameters: CnDriverSummarizeFluxesParameters
    restart_path: Path
    parameter_data_path: Path


class CnDriverSummarizeFluxesCall(NamedTuple):
    inputs: CnDriverSummarizeFluxesInputs
    state: CnDriverSummarizeFluxesState
    parameters: CnDriverSummarizeFluxesParameters


_RESTART_FIELDS = (
    "tempsum_npp",
    "col_lag_npp",
    "dwt_conv_flux_c_amt_to_dribble",
    "hrv_xsmrpool_to_atm_c_amt_to_dribble",
)


def _validate_configured_case(case: CimeCase) -> dict[str, object]:
    general = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    nitrogen = dict(read_case_namelist_group(case, "lnd_in", "clm_nitrogen"))
    required = {
        "use_cn": True,
        "use_crop": True,
        "use_century_decomp": True,
        "use_vertsoilc": True,
        "use_nitrif_denitrif": True,
        "use_fun": True,
        "use_grainproduct": True,
        "use_fates": False,
        "soil_layerstruct": "20SL_8.5m",
    }
    for name, expected in required.items():
        if general.get(name) != expected:
            raise ValueError(f"native flux-summary adapter requires {name}={expected!r}")
    if general.get("use_cndv", False) is not False:
        raise ValueError("configured flux summary requires use_cndv=.false.")
    if nitrogen.get("carbon_resp_opt") != 0:
        raise ValueError("native flux-summary adapter requires carbon_resp_opt=0")
    if case.value("CALENDAR").upper() != "NO_LEAP":
        raise ValueError("configured flux summary currently requires NO_LEAP")
    return general


def _validate_contiguous_topology(
    gridcell_index: np.ndarray,
    patch_column: np.ndarray,
) -> None:
    if np.any(gridcell_index[1:] < gridcell_index[:-1]):
        raise ValueError("native columns must be contiguous in gridcell order")
    if np.any(patch_column[1:] < patch_column[:-1]):
        raise ValueError("native patches must be contiguous in column order")
    column_counts = np.bincount(gridcell_index)
    patch_counts = np.bincount(patch_column)
    if int(np.max(column_counts, initial=0)) > 64:
        raise ValueError("native gridcell exceeds the 64-column summary bound")
    if int(np.max(patch_counts, initial=0)) > 79:
        raise ValueError("native column exceeds the 79-patch summary bound")


def _restart_vector(
    values: dict[str, jnp.ndarray],
    name: str,
    size: int,
) -> jnp.ndarray:
    host = np.asarray(values[name], dtype=np.float64)
    if host.shape != (size,) or not np.all(np.isfinite(host)):
        raise ValueError(f"restart {name!r} must be one finite native vector")
    return jnp.asarray(host)


def load_cn_driver_summarize_fluxes_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    state_summary_native_data: CnDriverSummarizeStatesNativeData | None = None,
) -> CnDriverSummarizeFluxesNativeData:

    general = _validate_configured_case(case)
    summary = state_summary_native_data
    if summary is None:
        summary = load_cn_driver_summarize_states_native_data(
            case,
            restart_path=restart_path,
            parameter_data_path=parameter_data_path,
        )
    else:
        if (
            restart_path is not None
            and Path(restart_path).expanduser().resolve() != summary.restart_path
        ):
            raise ValueError("flux-summary dependency uses a different restart")
        if (
            parameter_data_path is not None
            and Path(parameter_data_path).expanduser().resolve() != summary.parameter_data_path
        ):
            raise ValueError("flux-summary dependency uses a different parameter file")

    topology = summary.topology
    patch_count = topology.patch.column.shape[0]
    column_count = topology.column.gridcell.shape[0]
    gridcell_count = topology.grid.longitude_degrees.shape[0]
    _validate_contiguous_topology(
        np.asarray(topology.column.gridcell, dtype=np.int32),
        np.asarray(topology.patch.column, dtype=np.int32),
    )
    clock = load_restart_clock(summary.restart_path)
    time_step = int(general.get("dtime", 0))
    if time_step <= 0 or time_step != clock.step_seconds:
        raise ValueError("case dtime does not match the CLM restart time step")

    nfix_time_constant = general.get("nfix_timeconst")
    if nfix_time_constant is None:
        nfix_time_constant = 10.0
    nfix_time_constant = float(nfix_time_constant)
    if not 0.0 < nfix_time_constant < 500.0:
        raise ValueError("configured summary requires relaxed-NPP nitrogen fixation")

    values = load_native_variables(
        summary.restart_path,
        list(_RESTART_FIELDS),
        dtype=np.float64,
        masked_floating_fill_value=float(CLM_SPECIAL_VALUE),
    )
    persistent = CnDriverSummarizeFluxesPersistentState(
        temporary_npp_sum_patch=_restart_vector(
            values,
            "tempsum_npp",
            patch_count,
        ),
        lagged_npp_column=_restart_vector(
            values,
            "col_lag_npp",
            column_count,
        ),
        dynamic_landuse_amount_to_dribble_gridcell=_restart_vector(
            values,
            "dwt_conv_flux_c_amt_to_dribble",
            gridcell_count,
        ),
        harvest_xsmrpool_amount_to_dribble_gridcell=_restart_vector(
            values,
            "hrv_xsmrpool_to_atm_c_amt_to_dribble",
            gridcell_count,
        ),
    )
    zero_grid = jnp.zeros(gridcell_count, dtype=jnp.float64)
    return CnDriverSummarizeFluxesNativeData(
        inputs=CnDriverSummarizeFluxesInputs(
            soil_patch=summary.inputs.soil_patch,
            soil_column=summary.inputs.soil_column,
            vegetation_type_patch=summary.inputs.vegetation_type_patch,
            topology=topology,
            decomposition_layer_thickness=(summary.inputs.decomposition_layer_thickness),
            product_carbon_loss_gridcell=zero_grid,
            dynamic_landuse_carbon_flux_gridcell=zero_grid,
            is_beginning_current_year=jnp.asarray(False),
            is_first_step=jnp.asarray(False),
        ),
        persistent_state=persistent,
        parameters=CnDriverSummarizeFluxesParameters(
            time_step=np.float64(time_step),
            days_per_year=np.float64(365.0),
            nfix_time_constant_days=np.float64(nfix_time_constant),
            minimum_crop_vegetation_type=(summary.parameters.first_prognostic_crop_pft),
        ),
        restart_path=summary.restart_path,
        parameter_data_path=summary.parameter_data_path,
    )


def assemble_cn_driver_summarize_fluxes_call(
    native: CnDriverSummarizeFluxesNativeData,
    precision_result: PostDrainagePrecisionControlResult,
    product_result: CnWoodProductsResult,
    dynamic_landcover_carbon_flux: DynamicLandcoverCarbonFluxState,
    previous_state: CnDriverSummarizeFluxesState | None = None,
    *,
    is_beginning_current_year: bool | jnp.ndarray,
    is_first_step: bool | jnp.ndarray,
) -> CnDriverSummarizeFluxesCall:

    patch_count = native.inputs.soil_patch.shape[0]
    column_count = native.inputs.soil_column.shape[0]
    gridcell_count = native.inputs.topology.grid.longitude_degrees.shape[0]
    fluxes = precision_result.state.cn_fluxes
    if fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("post-precision fluxes have the wrong patch extent")
    if fluxes.column_scalar.shape[0] != column_count:
        raise ValueError("post-precision fluxes have the wrong column extent")

    product_loss = product_result.diagnostics.carbon.total_product_loss_gridcell
    dynamic_flux = dynamic_landcover_carbon_flux.dwt_conversion_carbon_flux_gridcell
    if jnp.shape(product_loss) != (gridcell_count,):
        raise ValueError("same-step product loss has the wrong gridcell extent")
    if jnp.shape(dynamic_flux) != (gridcell_count,):
        raise ValueError("same-step dynamic landuse flux has the wrong gridcell extent")

    if previous_state is None:
        persistent = native.persistent_state
    else:
        persistent = CnDriverSummarizeFluxesPersistentState(
            temporary_npp_sum_patch=previous_state.temporary_npp_sum_patch,
            lagged_npp_column=previous_state.lagged_npp_column,
            dynamic_landuse_amount_to_dribble_gridcell=(
                previous_state.dynamic_landuse_amount_to_dribble_gridcell
            ),
            harvest_xsmrpool_amount_to_dribble_gridcell=(
                previous_state.harvest_xsmrpool_amount_to_dribble_gridcell
            ),
        )
    expected_shapes = (
        (patch_count,),
        (column_count,),
        (gridcell_count,),
        (gridcell_count,),
    )
    for name, values, shape in zip(
        persistent._fields,
        persistent,
        expected_shapes,
        strict=True,
    ):
        if jnp.shape(values) != shape:
            raise ValueError(f"current {name} has the wrong native extent")

    return CnDriverSummarizeFluxesCall(
        inputs=native.inputs._replace(
            product_carbon_loss_gridcell=jnp.asarray(
                product_loss,
                dtype=jnp.float64,
            ),
            dynamic_landuse_carbon_flux_gridcell=jnp.asarray(
                dynamic_flux,
                dtype=jnp.float64,
            ),
            is_beginning_current_year=jnp.asarray(is_beginning_current_year),
            is_first_step=jnp.asarray(is_first_step),
        ),
        state=CnDriverSummarizeFluxesState(
            cn_fluxes=fluxes,
            temporary_npp_sum_patch=persistent.temporary_npp_sum_patch,
            lagged_npp_column=persistent.lagged_npp_column,
            dynamic_landuse_amount_to_dribble_gridcell=(
                persistent.dynamic_landuse_amount_to_dribble_gridcell
            ),
            harvest_xsmrpool_amount_to_dribble_gridcell=(
                persistent.harvest_xsmrpool_amount_to_dribble_gridcell
            ),
        ),
        parameters=native.parameters,
    )


__all__ = [
    "CnDriverSummarizeFluxesCall",
    "CnDriverSummarizeFluxesNativeData",
    "CnDriverSummarizeFluxesPersistentState",
    "assemble_cn_driver_summarize_fluxes_call",
    "load_cn_driver_summarize_fluxes_native_data",
]
