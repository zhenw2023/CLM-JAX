
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.physics.hydrology_drainage import (
    DEFAULT_HYDROLOGY_DRAINAGE_PARAMETERS,
    HydrologyDrainageInputs,
    HydrologyDrainageParameters,
    HydrologyDrainageResult,
    HydrologyDrainageState,
    hydrology_drainage_active_case,
)
from clm_jax.physics.hydrology_driver import HydrologyDriverResult
from clm_jax.physics.lake_hydrology_driver import LakeHydrologyDriverResult
from clm_jax.physics.physical_driver import (
    PhysicalDriverResult,
    PhysicalDriverRuntimeResult,
)
from clm_jax.physics.post_hydrology_no_drainage_driver import (
    PostHydrologyNoDrainageResult,
)
from clm_jax.physics.post_temperature_driver import PostTemperatureDriverResult
from clm_jax.physics.snow_water import NUMBER_OF_SNOW_LEVELS
from clm_jax.physics.surface_hydrology import NUMBER_OF_SOIL_LEVELS


class HydrologyDrainageDriverInputs(NamedTuple):

    flood_flux_gridcell: jnp.ndarray
    beginning_water_mass_column: jnp.ndarray
    total_plant_stored_water_column: jnp.ndarray
    do_surface_mass_balance_column: jnp.ndarray
    dynamic_runoff_routing_gridcell: jnp.ndarray


class HydrologyDrainageDriverResult(NamedTuple):

    inputs: HydrologyDrainageInputs
    entry_state: HydrologyDrainageState
    drainage: HydrologyDrainageResult


def assemble_hydrology_drainage_inputs(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    post_temperature: PostTemperatureDriverResult,
    hydrology: HydrologyDriverResult,
    post_hydrology: PostHydrologyNoDrainageResult,
    lake_hydrology: LakeHydrologyDriverResult,
    inputs: HydrologyDrainageDriverInputs,
) -> HydrologyDrainageInputs:

    del post_temperature  
    water = hydrology.hydrology.state
    lake = lake_hydrology.hydrology.state
    movement_inputs = hydrology.inputs.soil_water_movement
    surface_inputs = hydrology.inputs.surface_hydrology
    geometry = lake.geometry
    water_flux = lake.water_flux
    soil_interface_start = NUMBER_OF_SNOW_LEVELS + 1
    soil_interface_stop = soil_interface_start + NUMBER_OF_SOIL_LEVELS

    return HydrologyDrainageInputs(
        soil_layer_count_column=movement_inputs.soil_layer_count_column,
        snow_layer_count_column=geometry.snow_layer_count_column,
        layer_thickness_ground_column_level=(
            geometry.soil_snow_thickness_column_level[:, NUMBER_OF_SNOW_LEVELS:]
        ),
        lower_interface_depth_soil_column_level=(
            geometry.soil_snow_interface_depth_column_interface[
                :, soil_interface_start:soil_interface_stop
            ]
        ),
        topographic_slope_degrees_column=(
            surface_inputs.topographic_slope_degrees_column
        ),
        saturated_water_content_column_level=(
            movement_inputs.saturated_water_content_column_level
        ),
        saturated_hydraulic_conductivity_column_level=(
            movement_inputs.saturated_hydraulic_conductivity_column_level
        ),
        saturated_suction_column_level=(
            movement_inputs.saturated_suction_column_level
        ),
        clapp_hornberger_exponent_column_level=(
            movement_inputs.clapp_hornberger_exponent_column_level
        ),
        effective_porosity_column_level=(
            water.surface_hydrology.effective_porosity_column_level
        ),
        snow_water_equivalent_column=lake.water.snow_water_equivalent_column,
        canopy_water_patch=(
            physical.canopy_flux.state.active.postprocess.canopy_water_patch
        ),
        canopy_snow_patch=(
            physical.canopy_flux.state.active.postprocess.snow_canopy_water_patch
        ),
        total_plant_stored_water_column=inputs.total_plant_stored_water_column,
        aquifer_water_column=water.soil_water_movement.aquifer_water_column,
        rainfall_flux_column=physical.atmospheric.column_forcing.rain,
        snowfall_flux_column=physical.atmospheric.column_forcing.snow,
        flood_flux_gridcell=inputs.flood_flux_gridcell,
        total_evaporation_flux_column=water_flux.total_evaporation_flux_column,
        snow_capping_ice_flux_column=water_flux.snow_capping_ice_flux_column,
        snow_capping_liquid_flux_column=(
            water_flux.snow_capping_liquid_flux_column
        ),
        snow_capping_discarded_ice_flux_column=(
            water_flux.snow_capping_discarded_ice_flux_column
        ),
        snow_capping_discarded_liquid_flux_column=(
            water_flux.snow_capping_discarded_liquid_flux_column
        ),
        beginning_water_mass_column=inputs.beginning_water_mass_column,
        irrigation_flux_column=physical.canopy.irrigation.fluxes.applied_column,
        do_surface_mass_balance_column=inputs.do_surface_mass_balance_column,
        dynamic_runoff_routing_gridcell=inputs.dynamic_runoff_routing_gridcell,
        glacier_ice_growth_flux_column=(
            post_hydrology.glacier_surface_mass_balance.state.ice_growth_flux_column
        ),
        glacier_ice_melt_flux_column=(
            post_hydrology.glacier_surface_mass_balance.state.ice_melt_flux_column
        ),
    )


def assemble_hydrology_drainage_state(
    hydrology: HydrologyDriverResult,
    lake_hydrology: LakeHydrologyDriverResult,
    retained: HydrologyDrainageState,
) -> HydrologyDrainageState:

    water = hydrology.hydrology.state
    lake = lake_hydrology.hydrology.state
    flux = lake.water_flux
    return retained._replace(
        soil_snow_liquid_water_column_level=(
            lake.water.soil_snow_liquid_column_level
        ),
        soil_snow_ice_water_column_level=lake.water.soil_snow_ice_column_level,
        surface_water_column=water.surface_hydrology.surface_water_column,
        water_table_depth_column=(
            water.water_table_condensation.water_table_depth_column
        ),
        perched_water_table_depth_column=(
            water.water_table_condensation.perched_water_table_depth_column
        ),
        frost_table_depth_column=(
            water.water_table_condensation.frost_table_depth_column
        ),
        soil_ice_fraction_column_level=(
            water.surface_hydrology.soil_ice_fraction_column_level
        ),
        volumetric_water_column_level=(
            lake.water.soil_water_volume_fraction_column_level
        ),
        ending_water_mass_column=lake.water.end_water_balance_column,
        perched_drainage_flux_column=flux.perched_drainage_flux_column,
        saturated_excess_flux_column=flux.saturation_excess_flux_column,
        subsurface_drainage_flux_column=flux.drainage_flux_column,
        surface_runoff_flux_column=flux.surface_runoff_flux_column,
        surface_water_runoff_flux_column=flux.surface_water_runoff_flux_column,
        infiltration_flux_column=flux.infiltration_flux_column,
        wetland_glacier_runoff_flux_column=(
            flux.water_balance_runoff_flux_column
        ),
        total_runoff_flux_column=flux.total_runoff_flux_column,
        snow_capping_ice_runoff_flux_column=(
            flux.ice_runoff_snow_capping_flux_column
        ),
    )


def hydrology_drainage_driver_step(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    post_temperature: PostTemperatureDriverResult,
    hydrology: HydrologyDriverResult,
    post_hydrology: PostHydrologyNoDrainageResult,
    lake_hydrology: LakeHydrologyDriverResult,
    inputs: HydrologyDrainageDriverInputs,
    state: HydrologyDrainageState,
    parameters: HydrologyDrainageParameters = DEFAULT_HYDROLOGY_DRAINAGE_PARAMETERS,
) -> HydrologyDrainageDriverResult:

    entry_inputs = assemble_hydrology_drainage_inputs(
        physical,
        post_temperature,
        hydrology,
        post_hydrology,
        lake_hydrology,
        inputs,
    )
    entry_state = assemble_hydrology_drainage_state(
        hydrology,
        lake_hydrology,
        state,
    )
    result = hydrology_drainage_active_case(
        physical.soil_temperature.inputs.topology,
        entry_inputs,
        entry_state,
        parameters=parameters,
    )
    return HydrologyDrainageDriverResult(
        inputs=entry_inputs,
        entry_state=entry_state,
        drainage=result,
    )


__all__ = [
    "HydrologyDrainageDriverInputs",
    "HydrologyDrainageDriverResult",
    "assemble_hydrology_drainage_inputs",
    "assemble_hydrology_drainage_state",
    "hydrology_drainage_driver_step",
]
