
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.forcing.datm import DatmForcing
from clm_jax.physics.hydrology_driver import HydrologyDriverResult
from clm_jax.physics.lake_hydrology import (
    DEFAULT_LAKE_HYDROLOGY_PARAMETERS,
    LakeHydrologyAerosolState,
    LakeHydrologyEnergyFluxState,
    LakeHydrologyGeometryState,
    LakeHydrologyInputs,
    LakeHydrologyParameters,
    LakeHydrologyResult,
    LakeHydrologyState,
    LakeHydrologyTemperatureState,
    LakeHydrologyWaterFluxState,
    LakeHydrologyWaterState,
    lake_hydrology,
)
from clm_jax.physics.physical_driver import (
    PhysicalDriverResult,
    PhysicalDriverRuntimeResult,
)
from clm_jax.physics.post_hydrology_no_drainage_driver import (
    PostHydrologyNoDrainageResult,
)
from clm_jax.physics.post_temperature_driver import PostTemperatureDriverResult
from clm_jax.physics.snow_water import NUMBER_OF_SNOW_LEVELS


class LakeHydrologyDriverInputs(NamedTuple):

    template: LakeHydrologyInputs
    lake_column_index: jnp.ndarray
    lake_patch_index: jnp.ndarray


class LakeHydrologyDriverResult(NamedTuple):

    inputs: LakeHydrologyInputs
    entry_state: LakeHydrologyState
    hydrology: LakeHydrologyResult


def _scatter_rows(
    base: jnp.ndarray,
    index: jnp.ndarray,
    values: jnp.ndarray,
    *,
    label: str,
) -> jnp.ndarray:
    full = jnp.asarray(base)
    mapped = jnp.asarray(index, dtype=jnp.int32)
    compact = jnp.asarray(values, dtype=full.dtype)
    if mapped.ndim != 1 or compact.shape != (mapped.shape[0], *full.shape[1:]):
        raise ValueError(f"{label} compact and native axes disagree")
    return full.at[mapped].set(compact)


def _validate_maps(
    inputs: LakeHydrologyDriverInputs,
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
) -> None:
    column = jnp.asarray(inputs.lake_column_index)
    patch = jnp.asarray(inputs.lake_patch_index)
    lake = physical.lake_temperature.state
    if column.ndim != 1 or patch.shape != column.shape:
        raise ValueError("lake column and patch maps must be equal-length vectors")
    if lake.lake_temperature.shape[0] != column.shape[0]:
        raise ValueError("LakeTemperature result and lake map disagree")


def assemble_lake_hydrology_inputs(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    post_temperature: PostTemperatureDriverResult,
    hydrology: HydrologyDriverResult,
    forcing: DatmForcing,
    inputs: LakeHydrologyDriverInputs,
) -> LakeHydrologyInputs:

    _validate_maps(inputs, physical)
    column = jnp.asarray(inputs.lake_column_index, dtype=jnp.int32)
    patch = jnp.asarray(inputs.lake_patch_index, dtype=jnp.int32)
    atmospheric = physical.atmospheric.column_forcing
    lake_flux = physical.lake_flux.fluxes
    lake_temperature = physical.lake_temperature.state
    soil_phase = physical.soil_temperature.temperature.soil_snow_phase_change
    post_water = post_temperature.soil_fluxes.state.water

    ground_temperature = _scatter_rows(
        hydrology.hydrology.state.final_diagnostics.ground_temperature_column,
        column,
        lake_flux.temperature.ground_temperature,
        label="lake ground temperature",
    )
    melt_flag = _scatter_rows(
        soil_phase.phase_flag,
        column,
        lake_temperature.soil_snow_phase_flag,
        label="lake soil/snow melt flag",
    )
    soil_evaporation = _scatter_rows(
        post_water.soil_evaporation_patch,
        patch,
        lake_flux.water.ground_evaporation,
        label="lake soil evaporation",
    )
    total_evaporation = _scatter_rows(
        post_water.total_evaporation_patch,
        patch,
        lake_flux.water.total_evaporation,
        label="lake total evaporation",
    )
    soil_saturation = _scatter_rows(
        inputs.template.soil_saturation_column_level,
        column,
        physical.lake_temperature.call.geometry.saturated_water_content,
        label="lake saturated water content",
    )
    return inputs.template._replace(
        atmospheric_temperature_column=atmospheric.temperature,
        rainfall_flux_column=atmospheric.rain,
        snowfall_flux_column=atmospheric.snow,
        wind_speed_gridcell=forcing.wind_speed,
        atmospheric_aerosol_flux_gridcell_species=(
            hydrology.inputs.snow_water.atmospheric_aerosol_flux_gridcell_species
        ),
        ground_temperature_column=ground_temperature,
        snow_cover_shape_column=(
            hydrology.inputs.snow_compaction.snow_cover_shape_column
        ),
        soil_snow_melt_flag_column_level=melt_flag,
        previous_snow_water_column_level=(
            physical.canopy.canopy_hydrology.previous_snow_layer_water
        ),
        soil_saturation_column_level=soil_saturation,
        soil_evaporation_flux_patch=soil_evaporation,
        total_evaporation_flux_patch=total_evaporation,
    )


def assemble_lake_hydrology_state(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    post_temperature: PostTemperatureDriverResult,
    hydrology: HydrologyDriverResult,
    post_hydrology: PostHydrologyNoDrainageResult,
    inputs: LakeHydrologyDriverInputs,
    state: LakeHydrologyState,
) -> LakeHydrologyState:

    _validate_maps(inputs, physical)
    column = jnp.asarray(inputs.lake_column_index, dtype=jnp.int32)
    patch = jnp.asarray(inputs.lake_patch_index, dtype=jnp.int32)
    water_stage = hydrology.hydrology.state
    snow = water_stage.snow_structure
    lake = physical.lake_temperature.state
    lake_geometry = physical.lake_temperature.call.geometry
    lake_phase = physical.lake_temperature.temperature
    soil_phase = physical.soil_temperature.temperature.soil_snow_phase_change
    post_water = post_temperature.soil_fluxes.state.water
    post_energy = post_temperature.soil_fluxes.state.energy

    soil_snow_temperature = _scatter_rows(
        snow.soil_snow_temperature_column_level,
        column,
        lake.soil_snow_temperature,
        label="lake soil/snow temperature",
    )
    soil_snow_liquid = _scatter_rows(
        snow.soil_snow_liquid_column_level,
        column,
        lake.soil_snow_liquid_water,
        label="lake soil/snow liquid water",
    )
    soil_snow_ice = _scatter_rows(
        snow.soil_snow_ice_column_level,
        column,
        lake.soil_snow_ice_water,
        label="lake soil/snow ice water",
    )
    layer_count = _scatter_rows(
        snow.snow_layer_count_column,
        column,
        lake.snow_layer_count,
        label="lake snow layer count",
    )
    snow_depth = _scatter_rows(
        snow.snow_depth_column,
        column,
        lake.snow_depth,
        label="lake snow depth",
    )
    snow_water = _scatter_rows(
        snow.snow_water_equivalent_column,
        column,
        lake.snow_water,
        label="lake snow water",
    )
    old_ice_fraction = _scatter_rows(
        state.water.old_ice_fraction_column_level,
        column,
        lake.soil_snow_old_ice_fraction,
        label="lake old ice fraction",
    )
    lake_temperature = _scatter_rows(
        state.temperature.lake_temperature_column_level,
        column,
        lake.lake_temperature,
        label="lake temperature",
    )
    lake_ice_fraction = _scatter_rows(
        state.lake_ice_fraction_column_level,
        column,
        lake.lake_ice_fraction,
        label="lake ice fraction",
    )
    soil_snow_thickness = _scatter_rows(
        snow.soil_snow_thickness_column_level,
        column,
        lake_geometry.soil_snow_layer_thickness,
        label="lake soil/snow layer thickness",
    )
    soil_snow_node_depth = _scatter_rows(
        snow.soil_snow_node_depth_column_level,
        column,
        lake_geometry.soil_snow_node_depth,
        label="lake soil/snow node depth",
    )
    soil_snow_interface_depth = (
        jnp.asarray(
            snow.soil_snow_interface_depth_column_interface,
            dtype=jnp.float64,
        )
        .at[column, NUMBER_OF_SNOW_LEVELS + 1 :]
        .set(
            lake_geometry.soil_snow_interface_depth[:, NUMBER_OF_SNOW_LEVELS:]
        )
    )

    patch_inputs = post_temperature.patch_to_column_inputs
    columns = post_temperature.patch_to_column.state
    snowmelt_flux = _scatter_rows(
        soil_phase.snow_melt_flux,
        column,
        lake_phase.water.snow_melt,
        label="lake snowmelt flux",
    )
    snowmelt_layer = _scatter_rows(
        soil_phase.snow_layer_melt_flux,
        column,
        lake_phase.water.snow_layer_melt,
        label="lake layer snowmelt flux",
    )
    snow_drainage = _scatter_rows(
        water_stage.snow_water.snow_drainage_flux_column,
        column,
        lake_phase.water.snow_drain,
        label="lake snow drainage",
    )
    snowmelt_energy = _scatter_rows(
        soil_phase.snow_melt_heat_flux,
        column,
        lake_phase.energy.snow_melt_heat,
        label="lake snowmelt energy",
    )

    sensible_total = _scatter_rows(
        post_energy.total_sensible_heat_patch,
        patch,
        lake_phase.energy.sensible_total,
        label="lake total sensible heat",
    )
    sensible_ground = _scatter_rows(
        post_energy.ground_sensible_heat_patch,
        patch,
        lake_phase.energy.sensible_ground,
        label="lake ground sensible heat",
    )
    soil_ground_heat = _scatter_rows(
        post_energy.soil_ground_heat_patch,
        patch,
        lake_phase.energy.soil_ground_heat,
        label="lake soil ground heat",
    )
    net_ground = _scatter_rows(
        physical.soil_temperature.temperature.ground_heat_flux.patch_state.net_ground_heat_patch,
        patch,
        lake_phase.energy.ground_net,
        label="lake net ground heat",
    )

    return LakeHydrologyState(
        geometry=LakeHydrologyGeometryState(
            snow_layer_count_column=layer_count,
            soil_snow_thickness_column_level=soil_snow_thickness,
            soil_snow_node_depth_column_level=soil_snow_node_depth,
            soil_snow_interface_depth_column_interface=soil_snow_interface_depth,
        ),
        temperature=LakeHydrologyTemperatureState(
            lake_temperature_column_level=lake_temperature,
            soil_snow_temperature_column_level=soil_snow_temperature,
            top_temperature_gradient_column=(
                water_stage.final_diagnostics.top_snow_temperature_gradient_column
            ),
            top_snow_temperature_column=(
                water_stage.final_diagnostics.top_snow_temperature_column
            ),
            snow_temperature_mass_sum_column=(
                water_stage.snow_postprocess.snow_temperature_mass_product_column
            ),
        ),
        water=LakeHydrologyWaterState(
            effective_snow_fraction_column=snow.effective_snow_fraction_column,
            snow_fraction_column=snow.snow_fraction_column,
            old_ice_fraction_column_level=old_ice_fraction,
            snow_depth_column=snow_depth,
            snow_water_equivalent_column=snow_water,
            integrated_snow_column=snow.integrated_snow_column,
            snow_ice_column=water_stage.snow_postprocess.snow_ice_column,
            snow_liquid_column=water_stage.snow_postprocess.snow_liquid_column,
            soil_snow_ice_column_level=soil_snow_ice,
            soil_snow_liquid_column_level=soil_snow_liquid,
            soil_water_volume_fraction_column_level=(
                water_stage.final_diagnostics.volumetric_soil_water_column_level
            ),
            snow_grain_radius_column_level=snow.snow_grain_radius_column_level,
            top_snow_grain_radius_column=(
                water_stage.final_diagnostics.top_snow_grain_radius_column
            ),
            top_snow_water_column=(
                water_stage.final_diagnostics.top_snow_water_column
            ),
            top_snow_liquid_fraction_column=(
                water_stage.final_diagnostics.top_snow_liquid_fraction_column
            ),
            end_water_balance_column=state.water.end_water_balance_column,
        ),
        water_flux=LakeHydrologyWaterFluxState(
            ground_precipitation_flux_patch=patch_inputs.ground_precipitation_patch,
            ground_snowfall_flux_patch=patch_inputs.ground_snowfall_patch,
            ground_rainfall_flux_patch=patch_inputs.ground_rainfall_patch,
            ground_evaporation_flux_patch=post_water.ground_liquid_evaporation_patch,
            snow_sublimation_flux_patch=post_water.snow_sublimation_patch,
            snow_dew_flux_patch=post_water.snow_dew_patch,
            ground_dew_flux_patch=post_water.ground_dew_patch,
            ground_rainfall_flux_column=columns.ground_rainfall_column,
            ground_snowfall_flux_column=columns.ground_snowfall_column,
            ground_precipitation_flux_column=columns.ground_precipitation_column,
            ground_evaporation_flux_column=(
                columns.ground_liquid_evaporation_column
            ),
            ground_dew_flux_column=columns.ground_dew_column,
            snow_dew_flux_column=columns.snow_dew_column,
            snow_sublimation_flux_column=columns.snow_sublimation_column,
            total_evaporation_flux_column=columns.total_evaporation_column,
            snowmelt_flux_column=snowmelt_flux,
            snowmelt_flux_column_level=snowmelt_layer,
            snow_capping_ice_flux_column=(
                water_stage.snow_capping.snow_capping_ice_flux_column
            ),
            snow_capping_liquid_flux_column=(
                water_stage.snow_capping.snow_capping_liquid_flux_column
            ),
            snow_capping_discarded_ice_flux_column=(
                water_stage.snow_capping.snow_capping_discarded_ice_flux_column
            ),
            snow_capping_discarded_liquid_flux_column=(
                water_stage.snow_capping.snow_capping_discarded_liquid_flux_column
            ),
            perched_drainage_flux_column=(
                state.water_flux.perched_drainage_flux_column
            ),
            surface_water_runoff_flux_column=(
                water_stage.surface_hydrology.surface_water_runoff_flux_column
            ),
            snow_drainage_flux_column=snow_drainage,
            saturation_excess_flux_column=(
                state.water_flux.saturation_excess_flux_column
            ),
            surface_runoff_flux_column=(
                water_stage.surface_hydrology.surface_runoff_flux_column
            ),
            drainage_flux_column=state.water_flux.drainage_flux_column,
            infiltration_flux_column=(
                water_stage.surface_hydrology.infiltration_flux_column
            ),
            water_balance_runoff_flux_column=(
                state.water_flux.water_balance_runoff_flux_column
            ),
            total_runoff_flux_column=state.water_flux.total_runoff_flux_column,
            ice_runoff_snow_capping_flux_column=(
                state.water_flux.ice_runoff_snow_capping_flux_column
            ),
            top_soil_water_flux_column=(
                water_stage.snow_water.top_soil_water_flux_column
            ),
            snow_layer_to_top_soil_flux_column=(
                snow.top_soil_water_flux_column
            ),
            flood_flux_column=physical.canopy.canopy_hydrology.fluxes.flood_column,
        ),
        energy_flux=LakeHydrologyEnergyFluxState(
            snowmelt_energy_flux_column=snowmelt_energy,
            sensible_heat_total_patch=sensible_total,
            sensible_heat_ground_patch=sensible_ground,
            soil_heat_flux_ground_patch=soil_ground_heat,
            net_ground_heat_flux_patch=net_ground,
            ground_lake_heat_flux_patch=net_ground,
        ),
        aerosol=LakeHydrologyAerosolState(
            mass=post_hydrology.aerosol_mass.state.aerosol,
            deposition=snow.aerosol.deposition,
        ),
        lake_ice_fraction_column_level=lake_ice_fraction,
    )


def lake_hydrology_driver_step(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    post_temperature: PostTemperatureDriverResult,
    hydrology: HydrologyDriverResult,
    post_hydrology: PostHydrologyNoDrainageResult,
    forcing: DatmForcing,
    inputs: LakeHydrologyDriverInputs,
    state: LakeHydrologyState,
    parameters: LakeHydrologyParameters = DEFAULT_LAKE_HYDROLOGY_PARAMETERS,
) -> LakeHydrologyDriverResult:

    entry_inputs = assemble_lake_hydrology_inputs(
        physical,
        post_temperature,
        hydrology,
        forcing,
        inputs,
    )
    entry_state = assemble_lake_hydrology_state(
        physical,
        post_temperature,
        hydrology,
        post_hydrology,
        inputs,
        state,
    )
    result = lake_hydrology(
        physical.soil_temperature.inputs.topology,
        entry_inputs,
        entry_state,
        parameters=parameters,
    )
    return LakeHydrologyDriverResult(
        inputs=entry_inputs,
        entry_state=entry_state,
        hydrology=result,
    )


__all__ = [
    "LakeHydrologyDriverInputs",
    "LakeHydrologyDriverResult",
    "assemble_lake_hydrology_inputs",
    "assemble_lake_hydrology_state",
    "lake_hydrology_driver_step",
]
