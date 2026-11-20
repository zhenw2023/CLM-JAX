
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.cn_driver_summarize_fluxes import (
    CnDriverSummarizeFluxesNativeData,
    load_cn_driver_summarize_fluxes_native_data,
)
from clm_jax.io.cn_driver_summarize_states import (
    CnDriverSummarizeStatesNativeData,
    load_cn_driver_summarize_states_native_data,
)
from clm_jax.io.cn_n_state_update3 import (
    CnNStateUpdate3NativeData,
    load_cn_n_state_update3_native_data,
)
from clm_jax.io.cn_veg_struct_update import (
    CnVegStructUpdateNativeData,
    load_cn_veg_struct_update_native_data,
)
from clm_jax.io.native_templates import native_template
from clm_jax.io.post_drainage_precision_control import (
    PostDrainagePrecisionControlNativeData,
    load_post_drainage_precision_control_native_data,
)
from clm_jax.io.soil_biogeochem_n_leaching import (
    SoilBiogeochemNLeachingNativeData,
    load_soil_biogeochem_n_leaching_native_data,
)
from clm_jax.physics.bgc_timestep_initialization import (
    DynamicLandcoverCarbonFluxState,
)
from clm_jax.physics.cn_fire_area import CnFireAreaResult
from clm_jax.physics.cn_state_update2 import CnStateUpdate2State
from clm_jax.physics.cn_state_update3 import CnStateUpdate3WithPrecisionResult
from clm_jax.physics.cn_veg_struct_update import CnVegStructUpdateInputs
from clm_jax.physics.cn_wood_products import CnWoodProductsResult
from clm_jax.physics.cn_zero import CnTransientFluxState
from clm_jax.physics.crop_phenology import CropPhenologyResult
from clm_jax.physics.ecosystem_dynamics_carry import (
    EcosystemDynamicsPostDrainageCarry,
)
from clm_jax.physics.ecosystem_dynamics_post_drainage import (
    EcosystemDynamicsPostDrainageFluxSummaryState,
    EcosystemDynamicsPostDrainageInputs,
    EcosystemDynamicsPostDrainageParameters,
    EcosystemDynamicsPostDrainageState,
)
from clm_jax.physics.post_drainage_precision_control import (
    PostDrainagePrecisionControlState,
)
from clm_jax.physics.soil_biogeochem_n_leaching import (
    SoilBiogeochemNLeachingInputs,
)
from clm_jax.physics.soil_biogeochem_n_state_update1 import (
    SoilBiogeochemNStateUpdate1Result,
    SoilBiogeochemNStateUpdate1State,
)
from clm_jax.physics.soil_biogeochem_precision_control import (
    SoilBiogeochemPrecisionControlState,
)


class EcosystemDynamicsPostDrainageNativeData(NamedTuple):

    leaching: SoilBiogeochemNLeachingNativeData
    nitrogen_state_update: CnNStateUpdate3NativeData
    precision: PostDrainagePrecisionControlNativeData
    state_summary: CnDriverSummarizeStatesNativeData
    flux_summary: CnDriverSummarizeFluxesNativeData
    vegetation_structure: CnVegStructUpdateNativeData
    restart_path: Path
    parameter_data_path: Path


class EcosystemDynamicsPostDrainageCall(NamedTuple):
    inputs: EcosystemDynamicsPostDrainageInputs
    state: EcosystemDynamicsPostDrainageState
    parameters: EcosystemDynamicsPostDrainageParameters


class EcosystemDynamicsPostDrainagePersistentState(NamedTuple):

    carbon_truncation_column_level: jnp.ndarray
    nitrogen_truncation_column_level: jnp.ndarray
    state_summaries: object
    flux_summary: EcosystemDynamicsPostDrainageFluxSummaryState
    vegetation_structure: object


def post_drainage_persistent_state(
    state: EcosystemDynamicsPostDrainageState,
) -> EcosystemDynamicsPostDrainagePersistentState:

    soil = state.biogeochemistry.soil
    return EcosystemDynamicsPostDrainagePersistentState(
        carbon_truncation_column_level=soil.carbon_truncation_column_level,
        nitrogen_truncation_column_level=soil.nitrogen_truncation_column_level,
        state_summaries=state.state_summaries,
        flux_summary=state.flux_summary,
        vegetation_structure=state.vegetation_structure,
    )


def _require_same_path(label: str, expected: Path, actual: Path) -> None:
    if expected != actual:
        raise ValueError(f"post-drainage children use different {label} files")


@native_template
def load_ecosystem_dynamics_post_drainage_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> EcosystemDynamicsPostDrainageNativeData:

    nitrogen = load_cn_n_state_update3_native_data(
        case,
        restart_path=restart_path,
        surface_data_path=surface_data_path,
        parameter_data_path=parameter_data_path,
    )
    leaching = load_soil_biogeochem_n_leaching_native_data(
        case,
        restart_path=nitrogen.restart_path,
        parameter_data_path=nitrogen.parameter_data_path,
    )
    precision = load_post_drainage_precision_control_native_data(
        case,
        restart_path=nitrogen.restart_path,
        surface_data_path=nitrogen.surface_data_path,
        parameter_data_path=nitrogen.parameter_data_path,
        n_state_update3_native_data=nitrogen,
    )
    state_summary = load_cn_driver_summarize_states_native_data(
        case,
        restart_path=nitrogen.restart_path,
        parameter_data_path=nitrogen.parameter_data_path,
    )
    flux_summary = load_cn_driver_summarize_fluxes_native_data(
        case,
        restart_path=nitrogen.restart_path,
        parameter_data_path=nitrogen.parameter_data_path,
        state_summary_native_data=state_summary,
    )
    vegetation_structure = load_cn_veg_struct_update_native_data(
        case,
        restart_path=nitrogen.restart_path,
        parameter_data_path=nitrogen.parameter_data_path,
    )

    for child in (leaching, precision, state_summary, flux_summary, vegetation_structure):
        _require_same_path("restart", nitrogen.restart_path, child.restart_path)
    for child in (leaching, precision, state_summary, flux_summary, vegetation_structure):
        _require_same_path(
            "parameter-data",
            nitrogen.parameter_data_path,
            child.parameter_data_path,
        )

    patch_filter = np.asarray(nitrogen.inputs.soil_patch, dtype=bool)
    column_filter = np.asarray(nitrogen.inputs.soil_column, dtype=bool)
    filter_pairs = (
        (precision.inputs.vegetation.soil_patch, patch_filter, "precision patch"),
        (state_summary.inputs.soil_patch, patch_filter, "state-summary patch"),
        (flux_summary.inputs.soil_patch, patch_filter, "flux-summary patch"),
        (vegetation_structure.soil_patch, patch_filter, "structure patch"),
        (leaching.soil_column, column_filter, "leaching column"),
        (precision.inputs.soil.soil_column, column_filter, "precision column"),
        (state_summary.inputs.soil_column, column_filter, "state-summary column"),
        (flux_summary.inputs.soil_column, column_filter, "flux-summary column"),
    )
    for actual, expected, label in filter_pairs:
        if not np.array_equal(np.asarray(actual, dtype=bool), expected):
            raise ValueError(f"post-drainage {label} filter is inconsistent")

    time_steps = (
        float(leaching.parameters.time_step),
        float(nitrogen.parameters.time_step),
        float(flux_summary.parameters.time_step),
        float(vegetation_structure.parameters.radiation_step_seconds),
    )
    if len(set(time_steps)) != 1:
        raise ValueError("post-drainage children use inconsistent radiation steps")
    return EcosystemDynamicsPostDrainageNativeData(
        leaching=leaching,
        nitrogen_state_update=nitrogen,
        precision=precision,
        state_summary=state_summary,
        flux_summary=flux_summary,
        vegetation_structure=vegetation_structure,
        restart_path=nitrogen.restart_path,
        parameter_data_path=nitrogen.parameter_data_path,
    )


def _assemble_ecosystem_dynamics_post_drainage_call(
    native: EcosystemDynamicsPostDrainageNativeData,
    carbon: CnStateUpdate2State,
    mineral: SoilBiogeochemNStateUpdate1State,
    cn_fluxes: CnTransientFluxState,
    liquid_water_column_level: jnp.ndarray,
    drainage_flux_column: jnp.ndarray,
    surface_runoff_flux_column: jnp.ndarray,
    total_product_loss_gridcell: jnp.ndarray,
    dynamic_landcover_carbon_flux: DynamicLandcoverCarbonFluxState,
    total_burned_area_fraction_column: jnp.ndarray,
    harvest_date_patch: jnp.ndarray,
    snow_depth_column: jnp.ndarray,
    wind_forcing_height_patch: jnp.ndarray,
    excess_maintenance_respiration_carbon_loss_patch: jnp.ndarray,
    *,
    do_albedo: bool | jnp.ndarray,
    is_beginning_current_year: bool | jnp.ndarray,
    is_first_step: bool | jnp.ndarray,
    temporary_sum_npp_patch: jnp.ndarray | None = None,
    previous_state: (
        EcosystemDynamicsPostDrainageState
        | EcosystemDynamicsPostDrainagePersistentState
        | None
    ) = None,
) -> EcosystemDynamicsPostDrainageCall:

    patch_count = native.nitrogen_state_update.inputs.soil_patch.shape[0]
    column_count = native.nitrogen_state_update.inputs.soil_column.shape[0]
    gridcell_count = native.flux_summary.inputs.topology.grid.longitude_degrees.shape[0]
    level_count = native.leaching.persistent.nitrate_column_level.shape[1]
    level_shape = (column_count, level_count)
    pool_shape = (*level_shape, 7)

    if carbon.vegetation.paired_carbon_patch.shape[0] != patch_count:
        raise ValueError("same-step vegetation has the wrong patch extent")
    for name in (
        "soil_carbon_pool_column_level_pool",
        "soil_nitrogen_pool_column_level_pool",
    ):
        if jnp.shape(getattr(carbon, name)) != pool_shape:
            raise ValueError(f"same-step {name} has the wrong native shape")
    for name in mineral._fields:
        if jnp.shape(getattr(mineral, name)) != level_shape:
            raise ValueError(f"same-step mineral-N {name} has the wrong native shape")
    if cn_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("same-step packed fluxes have the wrong patch extent")
    if cn_fluxes.column_scalar.shape[0] != column_count:
        raise ValueError("same-step packed fluxes have the wrong column extent")

    liquid = jnp.asarray(liquid_water_column_level, dtype=jnp.float64)
    drainage = jnp.asarray(drainage_flux_column, dtype=jnp.float64)
    runoff = jnp.asarray(surface_runoff_flux_column, dtype=jnp.float64)
    if liquid.shape != level_shape:
        raise ValueError("same-step liquid water has the wrong native shape")
    if drainage.shape != (column_count,) or runoff.shape != (column_count,):
        raise ValueError("same-step drainage/runoff must contain every column")

    if previous_state is None:
        carbon_truncation = (
            native.precision.initial_soil_persistent_state.carbon_truncation_column_level
        )
        nitrogen_truncation = (
            native.precision.initial_soil_persistent_state.nitrogen_truncation_column_level
        )
        state_summaries = native.state_summary.initial_state
        flux_persistent = EcosystemDynamicsPostDrainageFluxSummaryState(
            *native.flux_summary.persistent_state
        )
        vegetation_structure = native.vegetation_structure.restart_state
    elif isinstance(previous_state, EcosystemDynamicsPostDrainagePersistentState):
        carbon_truncation = previous_state.carbon_truncation_column_level
        nitrogen_truncation = previous_state.nitrogen_truncation_column_level
        state_summaries = previous_state.state_summaries
        flux_persistent = previous_state.flux_summary
        vegetation_structure = previous_state.vegetation_structure
    else:
        carbon_truncation = (
            previous_state.biogeochemistry.soil.carbon_truncation_column_level
        )
        nitrogen_truncation = (
            previous_state.biogeochemistry.soil.nitrogen_truncation_column_level
        )
        state_summaries = previous_state.state_summaries
        flux_persistent = previous_state.flux_summary
        vegetation_structure = previous_state.vegetation_structure
    if jnp.shape(carbon_truncation) != level_shape:
        raise ValueError("current carbon truncation has the wrong native shape")
    if jnp.shape(nitrogen_truncation) != level_shape:
        raise ValueError("current nitrogen truncation has the wrong native shape")
    if temporary_sum_npp_patch is not None:
        if jnp.shape(temporary_sum_npp_patch) != (patch_count,):
            raise ValueError("same-step temporary NPP sum has the wrong native shape")
        flux_persistent = flux_persistent._replace(
            temporary_npp_sum_patch=temporary_sum_npp_patch,
        )

    biogeochemistry = PostDrainagePrecisionControlState(
        vegetation=carbon.vegetation,
        soil=SoilBiogeochemPrecisionControlState(
            carbon_pool_column_level_pool=(
                carbon.soil_carbon_pool_column_level_pool
            ),
            nitrogen_pool_column_level_pool=(
                carbon.soil_nitrogen_pool_column_level_pool
            ),
            carbon_truncation_column_level=carbon_truncation,
            nitrogen_truncation_column_level=nitrogen_truncation,
            mineral_nitrogen_column_level=mineral.mineral_nitrogen_column_level,
            ammonium_column_level=mineral.ammonium_column_level,
            nitrate_column_level=mineral.nitrate_column_level,
        ),
        cn_fluxes=cn_fluxes,
    )

    product_loss = jnp.asarray(
        total_product_loss_gridcell,
        dtype=jnp.float64,
    )
    dynamic_flux = jnp.asarray(
        dynamic_landcover_carbon_flux.dwt_conversion_carbon_flux_gridcell,
        dtype=jnp.float64,
    )
    burned_area = jnp.asarray(
        total_burned_area_fraction_column,
        dtype=jnp.float64,
    )
    harvest_date = jnp.asarray(
        harvest_date_patch,
        dtype=jnp.int32,
    )
    snow_depth = jnp.asarray(snow_depth_column, dtype=jnp.float64)
    wind_height = jnp.asarray(wind_forcing_height_patch, dtype=jnp.float64)
    xsmrpool_loss = jnp.asarray(
        excess_maintenance_respiration_carbon_loss_patch,
        dtype=jnp.float64,
    )
    expected_shapes = (
        (product_loss, (gridcell_count,), "product loss"),
        (dynamic_flux, (gridcell_count,), "dynamic-landcover flux"),
        (burned_area, (column_count,), "burned area"),
        (harvest_date, (patch_count,), "harvest date"),
        (snow_depth, (column_count,), "snow depth"),
        (wind_height, (patch_count,), "wind forcing height"),
        (xsmrpool_loss, (patch_count,), "xsmrpool loss"),
    )
    for values, expected, label in expected_shapes:
        if values.shape != expected:
            raise ValueError(f"same-step {label} has the wrong native shape")

    inputs = EcosystemDynamicsPostDrainageInputs(
        leaching=SoilBiogeochemNLeachingInputs(
            soil_column=native.leaching.soil_column,
            liquid_water_column_level=liquid,
            drainage_flux_column=drainage,
            surface_runoff_flux_column=runoff,
            nitrate_column_level=mineral.nitrate_column_level,
            layer_thickness_column_level=(
                native.leaching.layer_thickness_column_level
            ),
            soil_interface_depth=native.leaching.soil_interface_depth,
        ),
        nitrogen_state_update=native.nitrogen_state_update.inputs,
        precision=native.precision.inputs,
        state_summary=native.state_summary.inputs,
        excess_maintenance_respiration_carbon_loss_patch=xsmrpool_loss,
        flux_summary=native.flux_summary.inputs._replace(
            product_carbon_loss_gridcell=product_loss,
            dynamic_landuse_carbon_flux_gridcell=dynamic_flux,
            is_beginning_current_year=jnp.asarray(is_beginning_current_year),
            is_first_step=jnp.asarray(is_first_step),
        ),
        vegetation_structure=CnVegStructUpdateInputs(
            soil_patch=native.vegetation_structure.soil_patch,
            patch_column_index=native.vegetation_structure.topology.patch.column,
            vegetation_type_patch=(
                native.vegetation_structure.topology.patch.vegetation_type
            ),
            snow_depth_column=snow_depth,
            wind_forcing_height_patch=wind_height,
            leaf_carbon_patch=jnp.zeros(patch_count, dtype=jnp.float64),
            dead_stem_carbon_patch=jnp.zeros(patch_count, dtype=jnp.float64),
            burned_area_fraction_column=burned_area,
            harvest_date_patch=harvest_date,
        ),
        do_albedo=jnp.asarray(do_albedo),
    )
    return EcosystemDynamicsPostDrainageCall(
        inputs=inputs,
        state=EcosystemDynamicsPostDrainageState(
            biogeochemistry=biogeochemistry,
            state_summaries=state_summaries,
            flux_summary=flux_persistent,
            vegetation_structure=vegetation_structure,
        ),
        parameters=EcosystemDynamicsPostDrainageParameters(
            leaching=native.leaching.parameters,
            nitrogen_state_update=native.nitrogen_state_update.parameters,
            precision=native.precision.parameters,
            state_summary=native.state_summary.parameters,
            flux_summary=native.flux_summary.parameters,
            vegetation_structure=native.vegetation_structure.parameters,
        ),
    )


def assemble_ecosystem_dynamics_post_drainage_call_from_carry(
    native: EcosystemDynamicsPostDrainageNativeData,
    carry: EcosystemDynamicsPostDrainageCarry,
    liquid_water_column_level: jnp.ndarray,
    drainage_flux_column: jnp.ndarray,
    surface_runoff_flux_column: jnp.ndarray,
    dynamic_landcover_carbon_flux: DynamicLandcoverCarbonFluxState,
    snow_depth_column: jnp.ndarray,
    wind_forcing_height_patch: jnp.ndarray,
    *,
    do_albedo: bool | jnp.ndarray,
    is_beginning_current_year: bool | jnp.ndarray,
    is_first_step: bool | jnp.ndarray,
    previous_state: (
        EcosystemDynamicsPostDrainageState
        | EcosystemDynamicsPostDrainagePersistentState
        | None
    ) = None,
) -> EcosystemDynamicsPostDrainageCall:

    return _assemble_ecosystem_dynamics_post_drainage_call(
        native,
        carry.carbon_nitrogen_state,
        carry.mineral_nitrogen_state,
        carry.cn_fluxes,
        liquid_water_column_level,
        drainage_flux_column,
        surface_runoff_flux_column,
        carry.total_product_loss_gridcell,
        dynamic_landcover_carbon_flux,
        carry.total_burned_area_fraction_column,
        carry.harvest_date_patch,
        snow_depth_column,
        wind_forcing_height_patch,
        carry.excess_maintenance_respiration_carbon_loss_patch,
        do_albedo=do_albedo,
        is_beginning_current_year=is_beginning_current_year,
        is_first_step=is_first_step,
        temporary_sum_npp_patch=carry.temporary_sum_npp_patch,
        previous_state=previous_state,
    )


def assemble_ecosystem_dynamics_post_drainage_call(
    native: EcosystemDynamicsPostDrainageNativeData,
    carbon_state_update3_result: CnStateUpdate3WithPrecisionResult,
    mineral_state_update1_result: SoilBiogeochemNStateUpdate1Result,
    liquid_water_column_level: jnp.ndarray,
    drainage_flux_column: jnp.ndarray,
    surface_runoff_flux_column: jnp.ndarray,
    product_result: CnWoodProductsResult,
    dynamic_landcover_carbon_flux: DynamicLandcoverCarbonFluxState,
    fire_area_result: CnFireAreaResult,
    crop_phenology_result: CropPhenologyResult,
    snow_depth_column: jnp.ndarray,
    wind_forcing_height_patch: jnp.ndarray,
    excess_maintenance_respiration_carbon_loss_patch: jnp.ndarray,
    *,
    do_albedo: bool | jnp.ndarray,
    is_beginning_current_year: bool | jnp.ndarray,
    is_first_step: bool | jnp.ndarray,
    previous_state: (
        EcosystemDynamicsPostDrainageState
        | EcosystemDynamicsPostDrainagePersistentState
        | None
    ) = None,
) -> EcosystemDynamicsPostDrainageCall:

    return _assemble_ecosystem_dynamics_post_drainage_call(
        native,
        carbon_state_update3_result.state,
        mineral_state_update1_result.state,
        carbon_state_update3_result.cn_fluxes,
        liquid_water_column_level,
        drainage_flux_column,
        surface_runoff_flux_column,
        product_result.diagnostics.carbon.total_product_loss_gridcell,
        dynamic_landcover_carbon_flux,
        fire_area_result.state.total_burned_area_fraction_column,
        crop_phenology_result.state.harvest_date_patch,
        snow_depth_column,
        wind_forcing_height_patch,
        excess_maintenance_respiration_carbon_loss_patch,
        do_albedo=do_albedo,
        is_beginning_current_year=is_beginning_current_year,
        is_first_step=is_first_step,
        previous_state=previous_state,
    )


__all__ = [
    "EcosystemDynamicsPostDrainageCall",
    "EcosystemDynamicsPostDrainageNativeData",
    "EcosystemDynamicsPostDrainagePersistentState",
    "assemble_ecosystem_dynamics_post_drainage_call",
    "assemble_ecosystem_dynamics_post_drainage_call_from_carry",
    "load_ecosystem_dynamics_post_drainage_native_data",
    "post_drainage_persistent_state",
]
