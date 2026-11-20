
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.canopy_temperature_restart import soil_layer_geometry_20sl_8p5m
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    SubgridTopology,
    configured_land_restart,
    load_restart_clock,
)
from clm_jax.io.soil_biogeochem_vertical_profile import (
    load_soil_biogeochem_vertical_profile_native_data,
)
from clm_jax.physics.cn_fun_allocation import (
    FunAllocationInputs,
    FunAllocationParameters,
    FunAllocationState,
)
from clm_jax.physics.cn_fun_competition import (
    FunCompetitionCouplingInputs,
    FunCompetitionCouplingParameters,
)
from clm_jax.physics.cn_zero import get_cn_flux_field
from clm_jax.physics.cn_zero_manifest import NUMBER_OF_DECOMPOSITION_LEVELS
from clm_jax.physics.plant_nutrient_demand import PlantNutrientDemandResult
from clm_jax.physics.soil_biogeochem_competition import (
    SoilBiogeochemCompetitionPreFunResult,
)

_PARAMETER_VARIABLES = (
    "leafcn",
    "season_decid",
    "stress_decid",
    "c3psn",
    "a_fix",
    "b_fix",
    "c_fix",
    "s_fix",
    "akc_active",
    "akn_active",
    "ekc_active",
    "ekn_active",
    "kc_nonmyc",
    "kn_nonmyc",
    "perecm",
    "grperc",
    "fun_cn_flex_a",
    "fun_cn_flex_b",
    "fun_cn_flex_c",
    "FUN_fracfixers",
    "ndays_off",
)
_RESTART_POOL_VARIABLES = (
    "leafc",
    "leafc_storage",
    "frootc",
    "leafn",
    "leafn_storage",
    "frootn",
    "livestemn",
    "livecrootn",
)
_RESTART_STATE_VARIABLES = (
    "leafcn_offset",
    "plantCN",
    "leafc_storage_xfer_acc",
    "leafn_storage_xfer_acc",
    "storage_cdemand",
    "storage_ndemand",
)
_PLANT_NO3_LEVEL = "soil_nitrogen.smin_no3_to_plant_vr_col"
_PLANT_NH4_LEVEL = "soil_nitrogen.smin_nh4_to_plant_vr_col"


class FunAllocationPersistentInputs(NamedTuple):

    leaf_carbon_patch: jnp.ndarray
    leaf_carbon_storage_patch: jnp.ndarray
    fine_root_carbon_patch: jnp.ndarray
    leaf_nitrogen_patch: jnp.ndarray
    leaf_nitrogen_storage_patch: jnp.ndarray
    fine_root_nitrogen_patch: jnp.ndarray
    live_stem_nitrogen_patch: jnp.ndarray
    live_coarse_root_nitrogen_patch: jnp.ndarray
    root_carbon_fraction_patch_level: jnp.ndarray
    soil_temperature_column_level: jnp.ndarray
    soil_liquid_water_column_level: jnp.ndarray


class FunAllocationNativeData(NamedTuple):

    topology: SubgridTopology
    soil_patch: jnp.ndarray
    soil_column: jnp.ndarray
    persistent_inputs: FunAllocationPersistentInputs
    state: FunAllocationState
    parameters: FunAllocationParameters
    coupling_parameters: FunCompetitionCouplingParameters
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path


class FunAllocationCall(NamedTuple):

    inputs: FunAllocationInputs
    state: FunAllocationState
    parameters: FunAllocationParameters


def _pft_names(parameter_data: Path) -> tuple[str, ...]:
    from netCDF4 import Dataset  

    with Dataset(parameter_data, mode="r") as dataset:
        if "pftname" not in dataset.variables:
            raise ValueError("CLM parameter file lacks pftname")
        raw = np.asarray(dataset.variables["pftname"][:])
    if raw.ndim != 2 or raw.dtype.kind not in {"S", "U"}:
        raise ValueError("CLM pftname must be a character matrix")
    if raw.dtype.kind == "S":
        return tuple(
            b"".join(np.asarray(row, dtype="S1").tolist()).decode("ascii").rstrip()
            for row in raw
        )
    return tuple("".join(np.asarray(row, dtype=str).tolist()).rstrip() for row in raw)


def _parameter_vector(
    values: dict[str, jnp.ndarray],
    name: str,
    pft_count: int,
) -> jnp.ndarray:
    host = np.asarray(values[name], dtype=np.float64)
    if host.shape != (pft_count,) or not np.all(np.isfinite(host)):
        raise ValueError(f"CLM parameter {name!r} must be one finite PFT vector")
    return jnp.asarray(host)


def _logical_parameter_vector(
    values: dict[str, jnp.ndarray],
    name: str,
    pft_count: int,
) -> jnp.ndarray:
    host = np.asarray(values[name], dtype=np.float64)
    if host.shape != (pft_count,) or not np.all((host == 0.0) | (host == 1.0)):
        raise ValueError(f"CLM parameter {name!r} must contain only zero or one")
    return jnp.asarray(host == 1.0)


def _finite_scalar(name: str, value: jnp.ndarray) -> np.float64:
    host = np.asarray(value, dtype=np.float64)
    if host.size != 1 or not np.isfinite(host.reshape(())):
        raise ValueError(f"CLM parameter {name!r} must contain one finite scalar")
    return np.float64(host.reshape(()))


def _ground_tail(
    name: str,
    value: jnp.ndarray,
    column_count: int,
) -> jnp.ndarray:
    host = np.asarray(value, dtype=np.float64)
    if host.ndim != 2 or host.shape[0] != column_count:
        raise ValueError(f"restart {name} must have native (column, level) dimensions")
    if host.shape[1] < NUMBER_OF_DECOMPOSITION_LEVELS:
        raise ValueError(f"restart {name} lacks the 25 ground-level tail")
    return jnp.asarray(host[:, -NUMBER_OF_DECOMPOSITION_LEVELS:])


def load_fun_allocation_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> FunAllocationNativeData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    for key, expected in {
        "use_cn": True,
        "use_crop": True,
        "use_fates": False,
        "use_flexiblecn": True,
        "use_fun": True,
        "use_nitrif_denitrif": True,
        "use_vertsoilc": True,
        "use_dynroot": False,
    }.items():
        if general.get(key) is not expected:
            raise ValueError(f"native FUN allocation adapter requires {key}={expected!r}")
    if general.get("soil_layerstruct") != "20SL_8.5m":
        raise ValueError(
            "native FUN allocation adapter requires soil_layerstruct='20SL_8.5m'"
        )
    if str(general.get("suplnitro", "NONE")).upper() != "NONE":
        raise ValueError("native FUN allocation adapter requires suplnitro='NONE'")

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    dimensions = load_dimension_sizes(restart)
    if dimensions.get("levgrnd") != NUMBER_OF_DECOMPOSITION_LEVELS:
        raise ValueError("restart must contain the native 25 ground levels")

    vertical = load_soil_biogeochem_vertical_profile_native_data(
        case,
        restart_path=restart,
        surface_data_path=surface_data_path,
        parameter_data_path=parameter_data_path,
    )
    topology = vertical.topology
    patch_count = topology.patch.gridcell.shape[0]
    column_count = topology.column.gridcell.shape[0]
    soil_patch = np.asarray(vertical.soil_patch, dtype=bool)
    soil_column = np.asarray(vertical.soil_column, dtype=bool)

    restart_values = load_native_variables(
        restart,
        [
            *_RESTART_POOL_VARIABLES,
            *_RESTART_STATE_VARIABLES,
            "T_SOISNO",
            "H2OSOI_LIQ",
        ],
        dtype=np.float64,
        masked_floating_fill_value=float(CLM_SPECIAL_VALUE),
    )
    for name in (*_RESTART_POOL_VARIABLES, *_RESTART_STATE_VARIABLES):
        host = np.asarray(restart_values[name], dtype=np.float64)
        if host.shape != (patch_count,):
            raise ValueError(f"restart {name!r} must contain every native patch")
        if not np.all(np.isfinite(host[soil_patch])):
            raise ValueError(f"active soil patches contain non-finite {name!r}")
    temperature = _ground_tail(
        "T_SOISNO", restart_values["T_SOISNO"], column_count
    )
    liquid_water = _ground_tail(
        "H2OSOI_LIQ", restart_values["H2OSOI_LIQ"], column_count
    )
    if not np.all(np.isfinite(np.asarray(temperature)[soil_column])):
        raise ValueError("active soil columns contain non-finite T_SOISNO")
    if not np.all(np.isfinite(np.asarray(liquid_water)[soil_column])):
        raise ValueError("active soil columns contain non-finite H2OSOI_LIQ")

    parameter_data = vertical.parameter_data_path
    parameter_values = load_native_variables(
        parameter_data,
        list(_PARAMETER_VARIABLES),
        dtype=np.float64,
    )
    pft_names = _pft_names(parameter_data)
    pft_count = len(pft_names)
    if pft_count < 1 or len(set(pft_names)) != pft_count:
        raise ValueError("CLM pftname entries must be nonempty and unique")
    try:
        first_crop_pft = pft_names.index("temperate_corn")
    except ValueError as error:
        raise ValueError("CLM parameter file lacks temperate_corn") from error
    vegetation_type = np.asarray(topology.patch.vegetation_type, dtype=np.int32)
    if np.any((vegetation_type < 0) | (vegetation_type >= pft_count)):
        raise ValueError("restart vegetation type lies outside native PFT parameters")

    _, _, layer_thickness = soil_layer_geometry_20sl_8p5m()
    clock = load_restart_clock(restart)
    parameters = FunAllocationParameters(
        leaf_carbon_nitrogen_ratio_pft=_parameter_vector(
            parameter_values, "leafcn", pft_count
        ),
        seasonal_deciduous_pft=_logical_parameter_vector(
            parameter_values, "season_decid", pft_count
        ),
        stress_deciduous_pft=_logical_parameter_vector(
            parameter_values, "stress_decid", pft_count
        ),
        c3_photosynthesis_pft=_parameter_vector(
            parameter_values, "c3psn", pft_count
        ),
        fixation_a_pft=_parameter_vector(parameter_values, "a_fix", pft_count),
        fixation_b_pft=_parameter_vector(parameter_values, "b_fix", pft_count),
        fixation_c_pft=_parameter_vector(parameter_values, "c_fix", pft_count),
        fixation_scale_pft=_parameter_vector(
            parameter_values, "s_fix", pft_count
        ),
        am_carbon_cost_pft=_parameter_vector(
            parameter_values, "akc_active", pft_count
        ),
        am_nitrogen_cost_pft=_parameter_vector(
            parameter_values, "akn_active", pft_count
        ),
        ecm_carbon_cost_pft=_parameter_vector(
            parameter_values, "ekc_active", pft_count
        ),
        ecm_nitrogen_cost_pft=_parameter_vector(
            parameter_values, "ekn_active", pft_count
        ),
        nonmycorrhizal_carbon_cost_pft=_parameter_vector(
            parameter_values, "kc_nonmyc", pft_count
        ),
        nonmycorrhizal_nitrogen_cost_pft=_parameter_vector(
            parameter_values, "kn_nonmyc", pft_count
        ),
        ecm_fraction_pft=_parameter_vector(parameter_values, "perecm", pft_count),
        growth_respiration_fraction_pft=_parameter_vector(
            parameter_values, "grperc", pft_count
        ),
        flexible_cn_cost_threshold_pft=_parameter_vector(
            parameter_values, "fun_cn_flex_a", pft_count
        ),
        flexible_cn_cost_shape_pft=_parameter_vector(
            parameter_values, "fun_cn_flex_b", pft_count
        ),
        flexible_cn_maximum_offset_pft=_parameter_vector(
            parameter_values, "fun_cn_flex_c", pft_count
        ),
        fixer_fraction_pft=_parameter_vector(
            parameter_values, "FUN_fracfixers", pft_count
        ),
        soil_layer_thickness=jnp.asarray(layer_thickness, dtype=jnp.float64),
        time_step_seconds=np.float64(clock.step_seconds),
        leaf_offset_days=_finite_scalar("ndays_off", parameter_values["ndays_off"]),
        first_crop_pft=np.int32(first_crop_pft),
        freezing_temperature=np.float64(273.15),
        small_value=np.float64(1.0e-12),
        special_value=np.float64(CLM_SPECIAL_VALUE),
        large_cost=np.float64(1.0e9),
        use_flexible_cn=True,
    )
    return FunAllocationNativeData(
        topology=topology,
        soil_patch=jnp.asarray(soil_patch),
        soil_column=jnp.asarray(soil_column),
        persistent_inputs=FunAllocationPersistentInputs(
            leaf_carbon_patch=jnp.asarray(restart_values["leafc"]),
            leaf_carbon_storage_patch=jnp.asarray(
                restart_values["leafc_storage"]
            ),
            fine_root_carbon_patch=jnp.asarray(restart_values["frootc"]),
            leaf_nitrogen_patch=jnp.asarray(restart_values["leafn"]),
            leaf_nitrogen_storage_patch=jnp.asarray(
                restart_values["leafn_storage"]
            ),
            fine_root_nitrogen_patch=jnp.asarray(restart_values["frootn"]),
            live_stem_nitrogen_patch=jnp.asarray(restart_values["livestemn"]),
            live_coarse_root_nitrogen_patch=jnp.asarray(
                restart_values["livecrootn"]
            ),
            root_carbon_fraction_patch_level=(
                vertical.carbon_root_fraction_patch_level
            ),
            soil_temperature_column_level=temperature,
            soil_liquid_water_column_level=liquid_water,
        ),
        state=FunAllocationState(
            leaf_carbon_nitrogen_offset_patch=jnp.asarray(
                restart_values["leafcn_offset"]
            ),
            plant_carbon_nitrogen_ratio_patch=jnp.asarray(
                restart_values["plantCN"]
            ),
            leaf_carbon_storage_transfer_accumulator_patch=jnp.asarray(
                restart_values["leafc_storage_xfer_acc"]
            ),
            leaf_nitrogen_storage_transfer_accumulator_patch=jnp.asarray(
                restart_values["leafn_storage_xfer_acc"]
            ),
            storage_carbon_demand_patch=jnp.asarray(
                restart_values["storage_cdemand"]
            ),
            storage_nitrogen_demand_patch=jnp.asarray(
                restart_values["storage_ndemand"]
            ),
        ),
        parameters=parameters,
        coupling_parameters=FunCompetitionCouplingParameters(
            maximum_patches_per_column=int(general["maxpatch_pft"]),
            special_value=float(CLM_SPECIAL_VALUE),
        ),
        restart_path=restart,
        surface_data_path=vertical.surface_data_path,
        parameter_data_path=parameter_data,
    )


def assemble_fun_competition_coupling_inputs(
    native: FunAllocationNativeData,
) -> FunCompetitionCouplingInputs:

    return FunCompetitionCouplingInputs(
        soil_patch=native.soil_patch,
        soil_column=native.soil_column,
        patch_column_index=native.topology.patch.column,
        patch_weight_on_column=native.topology.patch.weight_on_column,
        patch_active=native.topology.patch.active,
    )


def _patch_value(name: str, value: jnp.ndarray, patch_count: int) -> jnp.ndarray:
    result = jnp.asarray(value, dtype=jnp.float64)
    if result.shape != (patch_count,):
        raise ValueError(f"{name} must contain every native patch")
    return result


def _column_level_value(
    name: str,
    value: jnp.ndarray,
    column_count: int,
) -> jnp.ndarray:
    result = jnp.asarray(value, dtype=jnp.float64)
    if result.shape != (column_count, NUMBER_OF_DECOMPOSITION_LEVELS):
        raise ValueError(f"{name} must have shape (column, 25)")
    return result


def assemble_fun_allocation_call(
    native: FunAllocationNativeData,
    *,
    leaf_carbon_to_litter_fun_patch: jnp.ndarray,
    leaf_carbon_storage_to_transfer_patch: jnp.ndarray,
    leaf_nitrogen_storage_to_transfer_patch: jnp.ndarray,
    onset_flag_patch: jnp.ndarray,
    offset_flag_patch: jnp.ndarray,
    plant_nutrient_demand_result: PlantNutrientDemandResult | None = None,
    available_carbon_patch: jnp.ndarray | None = None,
    plant_nitrogen_demand_patch: jnp.ndarray | None = None,
    carbon_allometry_patch: jnp.ndarray | None = None,
    nitrogen_allometry_patch: jnp.ndarray | None = None,
    competition_pre_fun_result: SoilBiogeochemCompetitionPreFunResult | None = None,
    maximum_nitrate_uptake_column_level: jnp.ndarray | None = None,
    maximum_ammonium_uptake_column_level: jnp.ndarray | None = None,
    persistent_inputs: FunAllocationPersistentInputs | None = None,
    state: FunAllocationState | None = None,
) -> FunAllocationCall:

    patch_count = native.topology.patch.gridcell.shape[0]
    column_count = native.topology.column.gridcell.shape[0]
    persistent = native.persistent_inputs if persistent_inputs is None else persistent_inputs
    current_state = native.state if state is None else state

    if plant_nutrient_demand_result is not None:
        demand_state = plant_nutrient_demand_result.state
        direct = demand_state.direct_fluxes
        if available_carbon_patch is None:
            available_carbon_patch = direct.available_carbon_patch
        if plant_nitrogen_demand_patch is None:
            plant_nitrogen_demand_patch = direct.plant_nitrogen_demand_patch
        if carbon_allometry_patch is None:
            carbon_allometry_patch = demand_state.carbon_allometry_patch
        if nitrogen_allometry_patch is None:
            nitrogen_allometry_patch = demand_state.nitrogen_allometry_patch
    missing_demand = [
        name
        for name, value in {
            "available_carbon_patch": available_carbon_patch,
            "plant_nitrogen_demand_patch": plant_nitrogen_demand_patch,
            "carbon_allometry_patch": carbon_allometry_patch,
            "nitrogen_allometry_patch": nitrogen_allometry_patch,
        }.items()
        if value is None
    ]
    if missing_demand:
        raise ValueError(
            "same-step plant-demand output is required for " + ", ".join(missing_demand)
        )

    if competition_pre_fun_result is not None:
        if maximum_nitrate_uptake_column_level is None:
            maximum_nitrate_uptake_column_level = get_cn_flux_field(
                competition_pre_fun_result.cn_fluxes, _PLANT_NO3_LEVEL
            )
        if maximum_ammonium_uptake_column_level is None:
            maximum_ammonium_uptake_column_level = get_cn_flux_field(
                competition_pre_fun_result.cn_fluxes, _PLANT_NH4_LEVEL
            )
    if maximum_nitrate_uptake_column_level is None:
        raise ValueError("same-step maximum_nitrate_uptake_column_level is required")
    if maximum_ammonium_uptake_column_level is None:
        raise ValueError("same-step maximum_ammonium_uptake_column_level is required")

    patch_shape = (patch_count,)
    level_shape = (patch_count, NUMBER_OF_DECOMPOSITION_LEVELS)
    column_level_shape = (column_count, NUMBER_OF_DECOMPOSITION_LEVELS)
    for name in FunAllocationPersistentInputs._fields[:8]:
        if jnp.shape(getattr(persistent, name)) != patch_shape:
            raise ValueError(f"persistent {name} must contain every native patch")
    if jnp.shape(persistent.root_carbon_fraction_patch_level) != level_shape:
        raise ValueError("persistent root carbon fraction must have shape (patch, 25)")
    for name in FunAllocationPersistentInputs._fields[9:]:
        if jnp.shape(getattr(persistent, name)) != column_level_shape:
            raise ValueError(f"persistent {name} must have shape (column, 25)")
    for name in FunAllocationState._fields:
        if jnp.shape(getattr(current_state, name)) != patch_shape:
            raise ValueError(f"state {name} must contain every native patch")

    inputs = FunAllocationInputs(
        soil_patch=native.soil_patch,
        patch_column_index=native.topology.patch.column,
        vegetation_type_patch=native.topology.patch.vegetation_type,
        available_carbon_patch=_patch_value(
            "available_carbon_patch", available_carbon_patch, patch_count
        ),
        plant_nitrogen_demand_patch=_patch_value(
            "plant_nitrogen_demand_patch",
            plant_nitrogen_demand_patch,
            patch_count,
        ),
        leaf_carbon_patch=persistent.leaf_carbon_patch,
        leaf_carbon_storage_patch=persistent.leaf_carbon_storage_patch,
        fine_root_carbon_patch=persistent.fine_root_carbon_patch,
        leaf_nitrogen_patch=persistent.leaf_nitrogen_patch,
        leaf_nitrogen_storage_patch=persistent.leaf_nitrogen_storage_patch,
        leaf_nitrogen_storage_to_transfer_patch=_patch_value(
            "leaf_nitrogen_storage_to_transfer_patch",
            leaf_nitrogen_storage_to_transfer_patch,
            patch_count,
        ),
        fine_root_nitrogen_patch=persistent.fine_root_nitrogen_patch,
        live_stem_nitrogen_patch=persistent.live_stem_nitrogen_patch,
        live_coarse_root_nitrogen_patch=(
            persistent.live_coarse_root_nitrogen_patch
        ),
        leaf_carbon_to_litter_fun_patch=_patch_value(
            "leaf_carbon_to_litter_fun_patch",
            leaf_carbon_to_litter_fun_patch,
            patch_count,
        ),
        leaf_carbon_storage_to_transfer_patch=_patch_value(
            "leaf_carbon_storage_to_transfer_patch",
            leaf_carbon_storage_to_transfer_patch,
            patch_count,
        ),
        carbon_allometry_patch=_patch_value(
            "carbon_allometry_patch", carbon_allometry_patch, patch_count
        ),
        nitrogen_allometry_patch=_patch_value(
            "nitrogen_allometry_patch", nitrogen_allometry_patch, patch_count
        ),
        onset_flag_patch=_patch_value(
            "onset_flag_patch", onset_flag_patch, patch_count
        ),
        offset_flag_patch=_patch_value(
            "offset_flag_patch", offset_flag_patch, patch_count
        ),
        root_carbon_fraction_patch_level=(
            persistent.root_carbon_fraction_patch_level
        ),
        soil_temperature_column_level=persistent.soil_temperature_column_level,
        soil_liquid_water_column_level=persistent.soil_liquid_water_column_level,
        maximum_nitrate_uptake_column_level=_column_level_value(
            "maximum_nitrate_uptake_column_level",
            maximum_nitrate_uptake_column_level,
            column_count,
        ),
        maximum_ammonium_uptake_column_level=_column_level_value(
            "maximum_ammonium_uptake_column_level",
            maximum_ammonium_uptake_column_level,
            column_count,
        ),
    )
    return FunAllocationCall(inputs, current_state, native.parameters)


__all__ = [
    "FunAllocationCall",
    "FunAllocationNativeData",
    "FunAllocationPersistentInputs",
    "assemble_fun_competition_coupling_inputs",
    "assemble_fun_allocation_call",
    "load_fun_allocation_native_data",
]
