
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.forcing.datm import DatmForcing
from clm_jax.physics.canopy_flux_driver import (
    CanopyFluxDriverResult,
    CanopyFluxRuntimeResult,
    CanopyFluxSoilTemperatureFields,
    canopy_flux_soil_temperature_fields,
)
from clm_jax.physics.hydrology_no_drainage import (
    DEFAULT_HYDROLOGY_NO_DRAINAGE_PARAMETERS,
    HydrologyNoDrainageInputs,
    HydrologyNoDrainageParameters,
    HydrologyNoDrainageResult,
    HydrologyNoDrainageState,
    hydrology_no_drainage_active_case,
)
from clm_jax.physics.physical_driver import (
    PhysicalDriverResult,
    PhysicalDriverRuntimeResult,
)
from clm_jax.physics.post_temperature_driver import PostTemperatureDriverResult
from clm_jax.physics.snow_layer_combining import SnowLayerCombiningState
from clm_jax.physics.snow_water import (
    NUMBER_OF_SNOW_LEVELS,
    SnowWaterAerosolState,
)
from clm_jax.physics.urban_fluxes import URBAN_SURFACE_PERVIOUS_ROAD


class HydrologyDriverInputs(NamedTuple):

    template: HydrologyNoDrainageInputs
    atmospheric_aerosol_flux_gridcell_species: jnp.ndarray
    urban_surface_patch_index: jnp.ndarray


class HydrologyDriverResult(NamedTuple):

    inputs: HydrologyNoDrainageInputs
    entry_state: HydrologyNoDrainageState
    hydrology: HydrologyNoDrainageResult


def _canopy_flux_projection(
    result: CanopyFluxDriverResult | CanopyFluxRuntimeResult,
) -> CanopyFluxSoilTemperatureFields:
    if isinstance(result, CanopyFluxRuntimeResult):
        return result.soil_temperature
    if isinstance(result, CanopyFluxDriverResult):
        return canopy_flux_soil_temperature_fields(result)
    raise TypeError("unsupported canopy flux result at hydrology boundary")


def _current_snow_aerosol(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    previous: SnowWaterAerosolState,
) -> SnowWaterAerosolState:

    aerosol = physical.canopy.state.canopy_hydrology.snow_aerosol
    return SnowWaterAerosolState(
        black_carbon_hydrophilic_layer=aerosol.hydrophilic_black_carbon_layer,
        black_carbon_hydrophobic_layer=aerosol.hydrophobic_black_carbon_layer,
        organic_carbon_hydrophilic_layer=aerosol.hydrophilic_organic_carbon_layer,
        organic_carbon_hydrophobic_layer=aerosol.hydrophobic_organic_carbon_layer,
        dust_species_1_layer=aerosol.dust_species_1_layer,
        dust_species_2_layer=aerosol.dust_species_2_layer,
        dust_species_3_layer=aerosol.dust_species_3_layer,
        dust_species_4_layer=aerosol.dust_species_4_layer,
        deposition=previous.deposition,
    )


def _current_old_ice_fraction(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    previous: jnp.ndarray,
) -> jnp.ndarray:
    current = jnp.asarray(
        physical.canopy.state.canopy_hydrology.snow.old_ice_fraction,
        dtype=jnp.float64,
    )
    retained = jnp.asarray(previous, dtype=jnp.float64)
    if current.ndim != 2 or retained.ndim != 2:
        raise ValueError("snow ice-fraction arrays must be two-dimensional")
    if current.shape[0] != retained.shape[0]:
        raise ValueError("snow ice-fraction column axes disagree")
    if current.shape[1] == retained.shape[1]:
        return current
    if current.shape[1] < NUMBER_OF_SNOW_LEVELS:
        raise ValueError("current ice fraction lacks the native snow axis")
    if retained.shape[1] < NUMBER_OF_SNOW_LEVELS:
        raise ValueError("compaction template lacks the native snow axis")
    return retained.at[:, :NUMBER_OF_SNOW_LEVELS].set(current[:, :NUMBER_OF_SNOW_LEVELS])


def _current_snow_structure(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    post: PostTemperatureDriverResult,
    previous: SnowLayerCombiningState,
) -> SnowLayerCombiningState:
    canopy = physical.canopy.state.canopy_hydrology
    snow = canopy.snow
    stage = physical.soil_temperature
    thermal = stage.inputs.thermal_properties
    temperature = stage.temperature
    phase = temperature.soil_snow_phase_change
    standing = temperature.standing_water_phase_change
    return previous._replace(
        snow_layer_count_column=snow.layer_count,
        snow_fraction_column=snow.snow_fraction,
        effective_snow_fraction_column=snow.effective_snow_fraction,
        snow_depth_column=phase.snow_depth,
        integrated_snow_column=standing.integrated_snowfall,
        snow_water_equivalent_column=phase.snow_water,
        soil_snow_thickness_column_level=thermal.layer_thickness,
        soil_snow_node_depth_column_level=thermal.node_depth,
        soil_snow_interface_depth_column_interface=thermal.interface_depth,
        soil_snow_temperature_column_level=phase.soil_snow_temperature,
        soil_snow_liquid_column_level=(post.soil_fluxes_inputs.liquid_water_column_level),
        soil_snow_ice_column_level=post.soil_fluxes_inputs.ice_water_column_level,
        snow_grain_radius_column_level=snow.snow_grain_radius,
        aerosol=_current_snow_aerosol(physical, previous.aerosol),
    )


def _current_effective_root_fraction(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    inputs: HydrologyDriverInputs,
) -> jnp.ndarray:

    retained = jnp.asarray(
        inputs.template.plant_sink.root_fraction_patch_level,
        dtype=jnp.float64,
    )
    if physical.urban_flux is None:
        return retained
    mapping = jnp.asarray(inputs.urban_surface_patch_index, dtype=jnp.int32)
    current = jnp.asarray(
        physical.urban_flux.surface.effective_root_fraction_surface_level,
        dtype=jnp.float64,
    )
    if mapping.ndim != 2 or current.ndim != 3:
        raise ValueError("urban root-fraction mapping must use native surface axes")
    if mapping.shape[:2] != current.shape[:2]:
        raise ValueError("urban patch mapping and effective root fractions disagree")
    patch = mapping[:, URBAN_SURFACE_PERVIOUS_ROAD]
    values = current[:, URBAN_SURFACE_PERVIOUS_ROAD, :]
    if values.shape[1] != retained.shape[1]:
        raise ValueError("urban and hydrology root-fraction level axes disagree")
    return retained.at[patch].set(values)


def assemble_hydrology_no_drainage_state(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    post: PostTemperatureDriverResult,
    state: HydrologyNoDrainageState,
) -> HydrologyNoDrainageState:

    snow = _current_snow_structure(physical, post, state.snow_structure)
    standing = physical.soil_temperature.temperature.standing_water_phase_change
    ground_slice = slice(NUMBER_OF_SNOW_LEVELS, None)

    snow_water = state.snow_water._replace(
        integrated_snow_column=snow.integrated_snow_column,
        snow_fraction_column=snow.snow_fraction_column,
        snow_depth_column=snow.snow_depth_column,
        soil_snow_liquid_column_level=snow.soil_snow_liquid_column_level,
        soil_snow_ice_column_level=snow.soil_snow_ice_column_level,
        soil_snow_thickness_column_level=snow.soil_snow_thickness_column_level,
        snow_drainage_flux_column=jnp.zeros_like(
            state.snow_water.snow_drainage_flux_column
        ),
        aerosol=snow.aerosol,
    )
    surface = state.surface_hydrology._replace(
        soil_snow_liquid_column_level=snow.soil_snow_liquid_column_level,
        surface_water_column=standing.standing_water,
    )
    movement = state.soil_water_movement._replace(
        liquid_water_mass_column_level=(snow.soil_snow_liquid_column_level[:, ground_slice]),
    )
    water_table = state.water_table_condensation._replace(
        soil_snow_liquid_water_column_level=snow.soil_snow_liquid_column_level,
        soil_snow_ice_water_column_level=snow.soil_snow_ice_column_level,
        snow_sublimation_flux_column=(post.patch_to_column.state.snow_sublimation_column),
    )
    capping = state.snow_capping._replace(
        soil_snow_liquid_column_level=snow.soil_snow_liquid_column_level,
        soil_snow_ice_column_level=snow.soil_snow_ice_column_level,
        soil_snow_thickness_column_level=snow.soil_snow_thickness_column_level,
        aerosol=snow.aerosol,
    )
    postprocess = state.snow_postprocess._replace(
        snow_layer_count_column=snow.snow_layer_count_column,
        soil_snow_thickness_column_level=snow.soil_snow_thickness_column_level,
        soil_snow_node_depth_column_level=snow.soil_snow_node_depth_column_level,
        soil_snow_interface_depth_column_interface=(
            snow.soil_snow_interface_depth_column_interface
        ),
        soil_snow_temperature_column_level=snow.soil_snow_temperature_column_level,
        soil_snow_ice_column_level=snow.soil_snow_ice_column_level,
        soil_snow_liquid_column_level=snow.soil_snow_liquid_column_level,
    )
    final = state.final_diagnostics._replace(
        ground_temperature_column=(
            physical.soil_temperature.temperature.final_diagnostics.ground_temperature
        ),
        snow_grain_radius_column_level=snow.snow_grain_radius_column_level,
    )
    return HydrologyNoDrainageState(
        snow_structure=snow,
        snow_water=snow_water,
        surface_hydrology=surface,
        plant_sink=state.plant_sink,
        soil_water_movement=movement,
        water_table_condensation=water_table,
        snow_capping=capping,
        snow_postprocess=postprocess,
        final_diagnostics=final,
    )


def assemble_hydrology_no_drainage_inputs(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    post: PostTemperatureDriverResult,
    forcing: DatmForcing,
    inputs: HydrologyDriverInputs,
    entry_state: HydrologyNoDrainageState,
) -> HydrologyNoDrainageInputs:

    template = inputs.template
    stage = physical.soil_temperature
    phase = stage.temperature.soil_snow_phase_change
    standing = stage.temperature.standing_water_phase_change
    snow = entry_state.snow_structure
    canopy = physical.canopy.state.canopy_hydrology
    columns = post.patch_to_column.state
    canopy_flux = _canopy_flux_projection(physical.canopy_flux)
    ground_slice = slice(NUMBER_OF_SNOW_LEVELS, None)

    snow_water = template.snow_water._replace(
        snow_layer_count_column=snow.snow_layer_count_column,
        effective_snow_fraction_column=snow.effective_snow_fraction_column,
        snow_water_equivalent_column=snow.snow_water_equivalent_column,
        snowmelt_flux_column=phase.snow_melt_flux,
        ground_rainfall_flux_column=columns.ground_rainfall_column,
        snow_sublimation_flux_column=columns.snow_sublimation_column,
        snow_dew_flux_column=columns.snow_dew_column,
        ground_evaporation_flux_column=columns.ground_liquid_evaporation_column,
        ground_dew_flux_column=columns.ground_dew_column,
        atmospheric_aerosol_flux_gridcell_species=jnp.asarray(
            inputs.atmospheric_aerosol_flux_gridcell_species,
            dtype=jnp.float64,
        ),
    )
    surface = template.surface_hydrology._replace(
        snow_layer_count_column=snow.snow_layer_count_column,
        soil_snow_thickness_column_level=snow.soil_snow_thickness_column_level,
        soil_snow_ice_column_level=snow.soil_snow_ice_column_level,
        snow_to_surface_water_flux_column=(
            physical.canopy.canopy_hydrology.fluxes.snow_on_surface_water_column
        ),
        flood_flux_column=physical.canopy.canopy_hydrology.fluxes.flood_column,
        ground_evaporation_flux_column=columns.ground_liquid_evaporation_column,
        soil_evaporation_flux_column=columns.bare_soil_evaporation_column,
        surface_water_evaporation_flux_column=(columns.surface_water_evaporation_column),
        effective_snow_fraction_column=snow.effective_snow_fraction_column,
        surface_water_fraction_column=canopy.surface_water_fraction_column,
        snow_free_surface_water_fraction_column=(canopy.surface_water_fraction_without_snow_column),
        water_table_depth_column=(entry_state.water_table_condensation.water_table_depth_column),
        frost_table_depth_column=(entry_state.water_table_condensation.frost_table_depth_column),
        perched_water_table_depth_column=(
            entry_state.water_table_condensation.perched_water_table_depth_column
        ),
    )
    plant = template.plant_sink._replace(
        soil_root_conductance_patch_level=(canopy_flux.soil_root_conductance_patch_level),
        root_fraction_patch_level=_current_effective_root_fraction(physical, inputs),
        vegetation_transpiration_patch=(post.soil_fluxes_inputs.vegetation_transpiration_patch),
        vegetation_transpiration_column=columns.vegetation_transpiration_column,
        soil_water_potential_column_level=(
            entry_state.final_diagnostics.soil_matrix_potential_column_level
        ),
        exposed_vegetation_flag_patch=(
            physical.atmospheric.driver_initialization.vegetation_exposed
        ),
        vegetation_water_potential_patch_segment=(
            physical.canopy_flux.state.plant.water_potential_segment
        ),
        soil_node_depth_column_level=(snow.soil_snow_node_depth_column_level[:, ground_slice]),
    )
    movement = template.soil_water_movement._replace(
        layer_thickness_column_level=(snow.soil_snow_thickness_column_level[:, ground_slice]),
        node_depth_column_level=(snow.soil_snow_node_depth_column_level[:, ground_slice]),
        ice_fraction_column_level=(entry_state.surface_hydrology.soil_ice_fraction_column_level),
        infiltration_flux_column=entry_state.surface_hydrology.infiltration_flux_column,
        root_water_sink_column_level=entry_state.plant_sink.root_soil_water_flux_column_level,
        ice_water_mass_column_level=(snow.soil_snow_ice_column_level[:, ground_slice]),
    )
    soil_interface_start = NUMBER_OF_SNOW_LEVELS + 1
    soil_interface_stop = soil_interface_start + 20
    water_table = template.water_table_condensation._replace(
        snow_layer_count_column=snow.snow_layer_count_column,
        layer_thickness_column_level=(snow.soil_snow_thickness_column_level[:, ground_slice]),
        node_depth_column_level=(snow.soil_snow_node_depth_column_level[:, ground_slice]),
        lower_interface_depth_column_level=(
            snow.soil_snow_interface_depth_column_interface[
                :, soil_interface_start:soil_interface_stop
            ]
        ),
        soil_temperature_column_level=(snow.soil_snow_temperature_column_level[:, ground_slice]),
        surface_water_fraction_column=canopy.surface_water_fraction_column,
        ground_dew_flux_column=columns.ground_dew_column,
        snow_dew_flux_column=columns.snow_dew_column,
    )
    old_ice_fraction = _current_old_ice_fraction(
        physical,
        template.snow_compaction.old_ice_fraction_column_level,
    )
    compaction = template.snow_compaction._replace(
        snow_layer_count_column=snow.snow_layer_count_column,
        wind_speed_gridcell=forcing.wind_speed,
        soil_snow_temperature_column_level=snow.soil_snow_temperature_column_level,
        soil_snow_melt_flag_column_level=phase.phase_flag,
        effective_snow_fraction_column=snow.effective_snow_fraction_column,
        previous_snow_water_column_level=(
            physical.canopy.canopy_hydrology.previous_snow_layer_water
        ),
        integrated_snow_column=standing.integrated_snowfall,
        old_ice_fraction_column_level=old_ice_fraction,
        soil_snow_ice_column_level=snow.soil_snow_ice_column_level,
        soil_snow_liquid_column_level=snow.soil_snow_liquid_column_level,
    )
    final = template.final_diagnostics._replace(
        snow_layer_count_column=snow.snow_layer_count_column,
        soil_snow_thickness_column_level=snow.soil_snow_thickness_column_level,
        soil_snow_node_depth_column_level=snow.soil_snow_node_depth_column_level,
        soil_snow_interface_depth_column_interface=(
            snow.soil_snow_interface_depth_column_interface
        ),
        soil_snow_temperature_column_level=snow.soil_snow_temperature_column_level,
        soil_snow_ice_column_level=snow.soil_snow_ice_column_level,
        soil_snow_liquid_column_level=snow.soil_snow_liquid_column_level,
        effective_snow_fraction_column=snow.effective_snow_fraction_column,
        surface_water_fraction_column=canopy.surface_water_fraction_column,
        surface_water_temperature_column=standing.standing_water_temperature,
    )
    return HydrologyNoDrainageInputs(
        snow_water=snow_water,
        surface_hydrology=surface,
        plant_sink=plant,
        soil_water_movement=movement,
        water_table_condensation=water_table,
        snow_compaction=compaction,
        final_diagnostics=final,
    )


def hydrology_driver_step(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    post: PostTemperatureDriverResult,
    forcing: DatmForcing,
    inputs: HydrologyDriverInputs,
    state: HydrologyNoDrainageState,
    parameters: HydrologyNoDrainageParameters = (DEFAULT_HYDROLOGY_NO_DRAINAGE_PARAMETERS),
) -> HydrologyDriverResult:

    topology = physical.soil_temperature.inputs.topology
    entry_state = assemble_hydrology_no_drainage_state(physical, post, state)
    assembled = assemble_hydrology_no_drainage_inputs(
        physical,
        post,
        forcing,
        inputs,
        entry_state,
    )
    result = hydrology_no_drainage_active_case(
        topology,
        assembled,
        entry_state,
        parameters=parameters,
    )
    return HydrologyDriverResult(
        inputs=assembled,
        entry_state=entry_state,
        hydrology=result,
    )


__all__ = [
    "HydrologyDriverInputs",
    "HydrologyDriverResult",
    "assemble_hydrology_no_drainage_inputs",
    "assemble_hydrology_no_drainage_state",
    "hydrology_driver_step",
]
