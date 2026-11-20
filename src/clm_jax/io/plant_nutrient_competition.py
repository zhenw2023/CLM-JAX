
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.plant_nutrient_demand import (
    PlantNutrientDemandNativeData,
    load_plant_nutrient_demand_native_data,
)
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.cn_fun_competition import FunSoilCompetitionResult
from clm_jax.physics.plant_nutrient_competition import (
    PlantNutrientCompetitionInputs,
    PlantNutrientCompetitionParameters,
    PlantNutrientCompetitionState,
)
from clm_jax.physics.plant_nutrient_demand import PlantNutrientDemandResult

_PARAMETER_VARIABLES = ("fcur", "evergreen")
_RESTART_VARIABLES = ("npool", "leafc_storage", "leafn_storage")


class PlantNutrientCompetitionPersistentInputs(NamedTuple):

    soil_patch: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    crop_live_patch: jnp.ndarray
    annual_net_primary_productivity_patch: jnp.ndarray
    plant_nitrogen_pool_patch: jnp.ndarray
    leaf_carbon_storage_patch: jnp.ndarray
    leaf_nitrogen_storage_patch: jnp.ndarray


class PlantNutrientCompetitionNativeData(NamedTuple):

    topology: SubgridTopology
    persistent_inputs: PlantNutrientCompetitionPersistentInputs
    state: PlantNutrientCompetitionState
    parameters: PlantNutrientCompetitionParameters
    restart_path: Path
    parameter_data_path: Path


class PlantNutrientCompetitionCall(NamedTuple):

    inputs: PlantNutrientCompetitionInputs
    state: PlantNutrientCompetitionState
    parameters: PlantNutrientCompetitionParameters


def _parameter_vector(
    values: dict[str, jnp.ndarray],
    name: str,
    pft_count: int,
    *,
    logical: bool = False,
) -> jnp.ndarray:
    host = np.asarray(values[name], dtype=np.float64)
    if host.shape != (pft_count,) or not np.all(np.isfinite(host)):
        raise ValueError(f"CLM parameter {name!r} must be one finite PFT vector")
    if logical and not np.all((host == 0.0) | (host == 1.0)):
        raise ValueError(f"CLM parameter {name!r} must contain only zero or one")
    return jnp.asarray(host)


def _require_active_controls(general: dict[str, object]) -> None:
    controls = {
        "use_fun": bool(general.get("use_fun", False)),
        "downreg_opt": bool(general.get("downreg_opt", False)),
        "cn_partition_opt": int(general.get("cn_partition_opt", 0)),
        "cn_residual_opt": int(general.get("cn_residual_opt", 0)),
        "carbon_resp_opt": int(general.get("carbon_resp_opt", 0)),
    }
    expected = {
        "use_fun": True,
        "downreg_opt": False,
        "cn_partition_opt": 0,
        "cn_residual_opt": 0,
        "carbon_resp_opt": 0,
    }
    for name, value in controls.items():
        if value != expected[name]:
            raise ValueError(
                f"native plant-competition adapter requires "
                f"{name}={expected[name]!r}"
            )


def load_plant_nutrient_competition_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    demand_native_data: PlantNutrientDemandNativeData | None = None,
) -> PlantNutrientCompetitionNativeData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    _require_active_controls(general)
    demand_native = (
        load_plant_nutrient_demand_native_data(
            case,
            restart_path=restart_path,
            parameter_data_path=parameter_data_path,
        )
        if demand_native_data is None
        else demand_native_data
    )
    restart = demand_native.restart_path
    parameter_data = demand_native.parameter_data_path
    if restart_path is not None and restart != Path(restart_path).expanduser().resolve():
        raise ValueError("demand_native_data uses a different restart")
    if parameter_data_path is not None and parameter_data != Path(
        parameter_data_path
    ).expanduser().resolve():
        raise ValueError("demand_native_data uses a different parameter file")

    patch_count = demand_native.topology.patch.gridcell.shape[0]
    pft_count = demand_native.parameters.woody_pft.shape[0]
    parameter_values = load_native_variables(
        parameter_data,
        list(_PARAMETER_VARIABLES),
        dtype=np.float64,
    )
    restart_values = load_native_variables(
        restart,
        list(_RESTART_VARIABLES),
        dtype=np.float64,
        masked_floating_fill_value=float(CLM_SPECIAL_VALUE),
    )
    soil_patch = np.asarray(
        demand_native.persistent_inputs.soil_patch, dtype=bool
    )
    for name in _RESTART_VARIABLES:
        host = np.asarray(restart_values[name], dtype=np.float64)
        if host.shape != (patch_count,):
            raise ValueError(f"restart {name!r} must contain every native patch")
        if not np.all(np.isfinite(host[soil_patch])):
            raise ValueError(f"active soil patches contain non-finite {name!r}")

    demand_parameters = demand_native.parameters
    parameters = PlantNutrientCompetitionParameters(
        woody_pft=demand_parameters.woody_pft,
        fine_root_to_leaf_carbon_pft=(
            demand_parameters.fine_root_to_leaf_carbon_pft
        ),
        coarse_root_to_stem_carbon_pft=(
            demand_parameters.coarse_root_to_stem_carbon_pft
        ),
        stem_to_leaf_carbon_pft=demand_parameters.stem_to_leaf_carbon_pft,
        live_wood_fraction_pft=demand_parameters.live_wood_fraction_pft,
        leaf_carbon_nitrogen_ratio_pft=(
            demand_parameters.leaf_carbon_nitrogen_ratio_pft
        ),
        fine_root_carbon_nitrogen_ratio_pft=(
            demand_parameters.fine_root_carbon_nitrogen_ratio_pft
        ),
        live_wood_carbon_nitrogen_ratio_pft=(
            demand_parameters.live_wood_carbon_nitrogen_ratio_pft
        ),
        dead_wood_carbon_nitrogen_ratio_pft=(
            demand_parameters.dead_wood_carbon_nitrogen_ratio_pft
        ),
        current_growth_fraction_pft=_parameter_vector(
            parameter_values, "fcur", pft_count
        ),
        grain_carbon_nitrogen_ratio_pft=(
            demand_parameters.grain_carbon_nitrogen_ratio_pft
        ),
        growth_respiration_fraction_pft=(
            demand_parameters.growth_respiration_fraction_pft
        ),
        current_growth_respiration_fraction_pft=(
            demand_parameters.current_growth_respiration_fraction_pft
        ),
        evergreen_pft=_parameter_vector(
            parameter_values,
            "evergreen",
            pft_count,
            logical=True,
        ),
        first_crop_pft=demand_parameters.first_crop_pft,
        time_step_seconds=demand_parameters.time_step_seconds,
        use_fun=True,
        downregulation=False,
        carbon_nitrogen_partition_option=0,
        carbon_nitrogen_residual_option=0,
        carbon_respiration_option=0,
        missing_value=np.float64(CLM_SPECIAL_VALUE),
    )
    zero_patch = jnp.zeros(patch_count, dtype=jnp.float64)
    persistent = demand_native.persistent_inputs
    return PlantNutrientCompetitionNativeData(
        topology=demand_native.topology,
        persistent_inputs=PlantNutrientCompetitionPersistentInputs(
            soil_patch=persistent.soil_patch,
            vegetation_type_patch=persistent.vegetation_type_patch,
            crop_live_patch=persistent.crop_live_patch,
            annual_net_primary_productivity_patch=(
                persistent.annual_net_primary_productivity_patch
            ),
            plant_nitrogen_pool_patch=jnp.asarray(restart_values["npool"]),
            leaf_carbon_storage_patch=jnp.asarray(
                restart_values["leafc_storage"]
            ),
            leaf_nitrogen_storage_patch=jnp.asarray(
                restart_values["leafn_storage"]
            ),
        ),
        state=PlantNutrientCompetitionState(
            cn_fluxes=demand_native.state.cn_fluxes,
            plant_carbon_allocation_patch=zero_patch,
            plant_nitrogen_allocation_patch=zero_patch,
            actual_storage_leaf_carbon_nitrogen_ratio_patch=jnp.full(
                patch_count,
                jnp.float64(CLM_SPECIAL_VALUE),
            ),
        ),
        parameters=parameters,
        restart_path=restart,
        parameter_data_path=parameter_data,
    )


def _patch_value(name: str, value: jnp.ndarray, patch_count: int) -> jnp.ndarray:
    result = jnp.asarray(value, dtype=jnp.float64)
    if result.shape != (patch_count,):
        raise ValueError(f"{name} must contain every native patch")
    return result


def assemble_plant_nutrient_competition_call(
    native: PlantNutrientCompetitionNativeData,
    plant_nutrient_demand_result: PlantNutrientDemandResult,
    fun_soil_competition_result: FunSoilCompetitionResult,
    *,
    sunlit_leaf_area_index_patch: jnp.ndarray,
    shaded_leaf_area_index_patch: jnp.ndarray,
    persistent_inputs: PlantNutrientCompetitionPersistentInputs | None = None,
    state: PlantNutrientCompetitionState | None = None,
) -> PlantNutrientCompetitionCall:

    patch_count = native.topology.patch.gridcell.shape[0]
    column_count = native.topology.column.gridcell.shape[0]
    persistent = native.persistent_inputs if persistent_inputs is None else persistent_inputs
    current_state = native.state if state is None else state
    for name in PlantNutrientCompetitionPersistentInputs._fields:
        if jnp.shape(getattr(persistent, name)) != (patch_count,):
            raise ValueError(f"persistent {name} must contain every native patch")
    for name in PlantNutrientCompetitionState._fields[1:]:
        if jnp.shape(getattr(current_state, name)) != (patch_count,):
            raise ValueError(f"state {name} must contain every native patch")

    demand = plant_nutrient_demand_result.state
    for name in (
        "leaf_allocation_patch",
        "stem_allocation_patch",
        "carbon_allometry_patch",
        "root_allocation_patch",
        "reproductive_allocation_patch",
    ):
        if jnp.shape(getattr(demand, name)) != (patch_count,):
            raise ValueError(f"same-step plant demand {name} has the wrong shape")
    cn_fluxes = fun_soil_competition_result.competition.cn_fluxes
    if cn_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("same-step FUN result has the wrong patch count")
    if cn_fluxes.column_scalar.shape[0] != column_count:
        raise ValueError("same-step FUN result has the wrong column count")
    fun_uptake = (
        fun_soil_competition_result.allocation.nitrogen_fluxes
        .soil_nitrogen_to_plant_fun_patch
    )

    return PlantNutrientCompetitionCall(
        inputs=PlantNutrientCompetitionInputs(
            soil_patch=persistent.soil_patch,
            vegetation_type_patch=persistent.vegetation_type_patch,
            crop_live_patch=persistent.crop_live_patch,
            annual_net_primary_productivity_patch=(
                persistent.annual_net_primary_productivity_patch
            ),
            leaf_allocation_patch=demand.leaf_allocation_patch,
            stem_allocation_patch=demand.stem_allocation_patch,
            carbon_allometry_patch=demand.carbon_allometry_patch,
            root_allocation_patch=demand.root_allocation_patch,
            reproductive_allocation_patch=demand.reproductive_allocation_patch,
            plant_nitrogen_pool_patch=persistent.plant_nitrogen_pool_patch,
            fun_soil_nitrogen_uptake_patch=_patch_value(
                "FUN soil nitrogen uptake", fun_uptake, patch_count
            ),
            sunlit_leaf_area_index_patch=_patch_value(
                "sunlit leaf area index",
                sunlit_leaf_area_index_patch,
                patch_count,
            ),
            shaded_leaf_area_index_patch=_patch_value(
                "shaded leaf area index",
                shaded_leaf_area_index_patch,
                patch_count,
            ),
            leaf_carbon_storage_patch=persistent.leaf_carbon_storage_patch,
            leaf_nitrogen_storage_patch=persistent.leaf_nitrogen_storage_patch,
        ),
        state=current_state._replace(cn_fluxes=cn_fluxes),
        parameters=native.parameters,
    )


__all__ = [
    "PlantNutrientCompetitionCall",
    "PlantNutrientCompetitionNativeData",
    "PlantNutrientCompetitionPersistentInputs",
    "assemble_plant_nutrient_competition_call",
    "load_plant_nutrient_competition_native_data",
]
