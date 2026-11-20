
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.physics.canopy_flux_driver import (
    CanopyFluxDriverResult,
    CanopyFluxRuntimeResult,
    CanopyFluxSoilTemperatureFields,
    canopy_flux_soil_temperature_fields,
)
from clm_jax.physics.canopy_hydrology import CanopyPrecipitationFluxes
from clm_jax.physics.glacier_surface_mass_balance import (
    GlacierIceMeltInputs,
    GlacierIceMeltResult,
    GlacierIceMeltState,
    handle_glacier_ice_melt,
)
from clm_jax.physics.patch_to_column_fluxes import (
    PatchToColumnFluxInputs,
    PatchToColumnFluxParameters,
    PatchToColumnFluxResult,
    PatchToColumnFluxState,
    aggregate_patch_fluxes_to_columns,
)
from clm_jax.physics.physical_driver import (
    PhysicalDriverResult,
    PhysicalDriverRuntimeResult,
)
from clm_jax.physics.soil_fluxes import (
    NUMBER_OF_SNOW_LEVELS,
    SoilFluxesInputs,
    SoilFluxesParameters,
    SoilFluxesResult,
    SoilFluxesState,
    soil_fluxes,
)
from clm_jax.physics.urban_flux_driver import urban_flux_soil_temperature_fields


class PostTemperatureNativeMap(NamedTuple):

    urban_surface_patch_index: jnp.ndarray
    lake_column_index: jnp.ndarray
    lake_patch_index: jnp.ndarray


class PostTemperatureDriverInputs(NamedTuple):

    native_map: PostTemperatureNativeMap
    do_surface_mass_balance_column: jnp.ndarray
    soil_fluxes_template: SoilFluxesInputs


class PostTemperatureDriverState(NamedTuple):

    glacier_ice_melt_flux_column: jnp.ndarray
    soil_fluxes: SoilFluxesState
    patch_to_column: PatchToColumnFluxState


class PostTemperatureDriverParameters(NamedTuple):

    soil_fluxes: SoilFluxesParameters = SoilFluxesParameters()
    patch_to_column: PatchToColumnFluxParameters = PatchToColumnFluxParameters()


DEFAULT_POST_TEMPERATURE_DRIVER_PARAMETERS = PostTemperatureDriverParameters()


class PostTemperatureDriverResult(NamedTuple):

    glacier_ice_melt: GlacierIceMeltResult
    soil_fluxes_inputs: SoilFluxesInputs
    soil_fluxes_entry_state: SoilFluxesState
    soil_fluxes: SoilFluxesResult
    patch_to_column_inputs: PatchToColumnFluxInputs
    patch_to_column: PatchToColumnFluxResult


def _scatter_compact(
    base: jnp.ndarray,
    index: jnp.ndarray,
    compact: jnp.ndarray,
    *,
    label: str,
) -> jnp.ndarray:

    full = jnp.asarray(base)
    mapped_index = jnp.asarray(index, dtype=jnp.int32)
    values = jnp.asarray(compact, dtype=full.dtype)
    if mapped_index.shape != values.shape:
        raise ValueError(f"{label} values and native patch map disagree")
    return full.at[jnp.reshape(mapped_index, (-1,))].set(jnp.reshape(values, (-1,)))


def _canopy_flux_boundary(
    result: CanopyFluxDriverResult | CanopyFluxRuntimeResult,
) -> tuple[object, CanopyFluxSoilTemperatureFields]:

    if isinstance(result, CanopyFluxRuntimeResult):
        return result.state.active.postprocess, result.soil_temperature
    if isinstance(result, CanopyFluxDriverResult):
        return result.active.state.postprocess, canopy_flux_soil_temperature_fields(result)
    raise TypeError("unsupported canopy flux result at post-temperature boundary")


def _merged_patch_fluxes(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    native_map: PostTemperatureNativeMap,
) -> dict[str, jnp.ndarray]:

    post, vegetation = _canopy_flux_boundary(physical.canopy_flux)
    urban = urban_flux_soil_temperature_fields(physical.urban_flux)
    urban_patch = native_map.urban_surface_patch_index
    zero_urban = jnp.zeros_like(urban.soil_evaporation_surface)

    def urban_overwrite(base: jnp.ndarray, values: jnp.ndarray, label: str) -> jnp.ndarray:
        return _scatter_compact(base, urban_patch, values, label=label)

    return {
        "snow_evaporation": urban_overwrite(
            post.snow_evaporation_patch,
            zero_urban,
            "urban snow evaporation",
        ),
        "bare_soil_evaporation": urban_overwrite(
            post.soil_evaporation_patch,
            zero_urban,
            "urban bare-soil evaporation",
        ),
        "surface_water_evaporation": urban_overwrite(
            post.surface_water_evaporation_patch,
            zero_urban,
            "urban surface-water evaporation",
        ),
        "soil_evaporation": urban_overwrite(
            post.ground_evaporation_patch,
            urban.soil_evaporation_surface,
            "urban soil evaporation",
        ),
        "ground_sensible_heat": urban_overwrite(
            post.sensible_heat_ground_patch,
            physical.urban_flux.surface.energy.sensible_heat_ground_surface,
            "urban ground sensible heat",
        ),
        "downward_longwave": urban_overwrite(
            post.downward_longwave_below_canopy_patch,
            urban.downward_longwave_below_canopy_surface,
            "urban downward longwave",
        ),
        "upward_longwave": urban_overwrite(
            post.upward_longwave_above_canopy_patch,
            urban.upward_longwave_above_canopy_surface,
            "urban upward longwave",
        ),
        "ground_sensible_derivative": urban_overwrite(
            post.ground_sensible_derivative_patch,
            urban.ground_sensible_derivative_surface,
            "urban ground sensible derivative",
        ),
        "ground_latent_derivative": urban_overwrite(
            post.ground_latent_derivative_patch,
            urban.ground_latent_derivative_surface,
            "urban ground latent derivative",
        ),
        "vegetation_transpiration": urban_overwrite(
            vegetation.vegetation_transpiration_patch,
            urban.vegetation_transpiration_surface,
            "urban vegetation transpiration",
        ),
        "vegetation_evaporation": urban_overwrite(
            vegetation.vegetation_evaporation_patch,
            urban.vegetation_evaporation_surface,
            "urban vegetation evaporation",
        ),
        "vegetation_sensible_heat": urban_overwrite(
            vegetation.vegetation_sensible_heat_patch,
            urban.vegetation_sensible_heat_surface,
            "urban vegetation sensible heat",
        ),
    }


def _merge_lake_flux_state(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    native_map: PostTemperatureNativeMap,
    state: SoilFluxesState,
) -> SoilFluxesState:

    lake_patch = native_map.lake_patch_index
    lake_flux = physical.lake_flux.fluxes
    lake_temperature = physical.lake_temperature.temperature

    def lake_overwrite(base: jnp.ndarray, values: jnp.ndarray, label: str) -> jnp.ndarray:
        return _scatter_compact(base, lake_patch, values, label=label)

    water = state.water._replace(
        soil_evaporation_patch=lake_overwrite(
            state.water.soil_evaporation_patch,
            lake_flux.water.ground_evaporation,
            "lake soil evaporation",
        ),
        total_evaporation_patch=lake_overwrite(
            state.water.total_evaporation_patch,
            lake_flux.water.total_evaporation,
            "lake total evaporation",
        ),
    )
    energy = state.energy._replace(
        soil_ground_heat_patch=lake_overwrite(
            state.energy.soil_ground_heat_patch,
            lake_temperature.energy.soil_ground_heat,
            "lake soil heat",
        ),
        total_sensible_heat_patch=lake_overwrite(
            state.energy.total_sensible_heat_patch,
            lake_temperature.energy.sensible_total,
            "lake total sensible heat",
        ),
        ground_sensible_heat_patch=lake_overwrite(
            state.energy.ground_sensible_heat_patch,
            lake_temperature.energy.sensible_ground,
            "lake ground sensible heat",
        ),
        total_latent_heat_patch=lake_overwrite(
            state.energy.total_latent_heat_patch,
            lake_flux.energy.latent_total,
            "lake total latent heat",
        ),
        ground_latent_heat_patch=lake_overwrite(
            state.energy.ground_latent_heat_patch,
            lake_flux.energy.latent_ground,
            "lake ground latent heat",
        ),
        outgoing_longwave_patch=lake_overwrite(
            state.energy.outgoing_longwave_patch,
            lake_flux.energy.outgoing_longwave,
            "lake outgoing longwave",
        ),
        net_longwave_patch=lake_overwrite(
            state.energy.net_longwave_patch,
            lake_flux.energy.net_longwave,
            "lake net longwave",
        ),
        soil_energy_error_column=_scatter_compact(
            state.energy.soil_energy_error_column,
            native_map.lake_column_index,
            lake_temperature.energy.conservation_error,
            label="lake column energy error",
        ),
    )
    skin_temperature = lake_overwrite(
        state.skin_temperature_patch,
        lake_flux.temperature.skin_temperature,
        "lake skin temperature",
    )
    return SoilFluxesState(
        skin_temperature_patch=skin_temperature,
        water=water,
        energy=energy,
    )


def assemble_soil_fluxes_entry_state(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    inputs: PostTemperatureDriverInputs,
    state: PostTemperatureDriverState,
) -> SoilFluxesState:

    patch = _merged_patch_fluxes(physical, inputs.native_map)
    template = state.soil_fluxes
    radiation = physical.radiation.state.urban.fluxes
    water = template.water._replace(
        snow_evaporation_patch=patch["snow_evaporation"],
        bare_soil_evaporation_patch=patch["bare_soil_evaporation"],
        surface_water_evaporation_patch=patch["surface_water_evaporation"],
        soil_evaporation_patch=patch["soil_evaporation"],
    )
    energy = template.energy._replace(
        ground_sensible_heat_patch=patch["ground_sensible_heat"],
        outgoing_longwave_patch=radiation.emitted_longwave_patch,
        net_longwave_patch=radiation.net_longwave_patch,
        urban_net_longwave_patch=radiation.urban_net_longwave_patch,
    )
    canopy_post, _ = _canopy_flux_boundary(physical.canopy_flux)
    assembled = SoilFluxesState(
        skin_temperature_patch=canopy_post.skin_temperature_patch,
        water=water,
        energy=energy,
    )
    return _merge_lake_flux_state(physical, inputs.native_map, assembled)


def assemble_soil_fluxes_inputs(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    inputs: PostTemperatureDriverInputs,
    glacier: GlacierIceMeltResult,
) -> SoilFluxesInputs:

    stage = physical.soil_temperature
    temperature = stage.temperature
    phase = temperature.soil_snow_phase_change
    standing_water = temperature.standing_water_phase_change
    full_liquid = phase.liquid_water.at[:, NUMBER_OF_SNOW_LEVELS:].set(
        glacier.state.liquid_water
    )
    full_ice = phase.ice_water.at[:, NUMBER_OF_SNOW_LEVELS:].set(
        glacier.state.ice_water
    )
    ground = stage.inputs.ground_heat_flux
    patch = _merged_patch_fluxes(physical, inputs.native_map)
    ground_patch = temperature.ground_heat_flux.patch_state
    return inputs.soil_fluxes_template._replace(
        snow_layer_count_column=stage.inputs.thermal_properties.snow_layer_count,
        downward_longwave_column=ground.downward_longwave_column,
        effective_snow_fraction_column=(
            stage.inputs.thermal_properties.effective_snow_fraction
        ),
        surface_water_fraction_column=stage.inputs.rhs_template.standing_water_fraction,
        liquid_water_column_level=full_liquid,
        ice_water_column_level=full_ice,
        ground_emissivity_column=ground.ground_emissivity_column,
        surface_water_temperature_column=standing_water.standing_water_temperature,
        previous_surface_water_temperature_column=(
            stage.inputs.solution_update_template.previous_standing_water_temperature
        ),
        previous_soil_snow_temperature_column_level=(
            stage.inputs.postsolve_flux_template.previous_soil_snow_temperature
        ),
        ground_temperature_column=temperature.final_diagnostics.ground_temperature,
        soil_snow_temperature_column_level=phase.soil_snow_temperature,
        phase_change_heat_flux_column=phase.phase_change_heat_flux,
        surface_water_phase_change_heat_flux_column=(
            standing_water.phase_change_heat_flux
        ),
        temperature_factor_column_level=(
            temperature.heat_diffusion.timestep_over_heat_capacity
        ),
        surface_water_heat_capacity_column=temperature.standing_water_heat_capacity,
        latent_heat_column=ground.latent_heat_column,
        surface_water_to_snow_heat_flux_column=standing_water.water_to_snow_heat_flux,
        building_heat_error_column=temperature.postsolve_flux.building_heat_flux,
        vegetation_exposed_patch=ground.vegetation_exposed_patch,
        absorbed_soil_patch=ground.absorbed_soil_patch,
        absorbed_snow_patch=ground.absorbed_snow_patch,
        absorbed_ground_patch=ground.absorbed_ground_patch,
        downward_longwave_below_canopy_patch=patch["downward_longwave"],
        upward_longwave_above_canopy_patch=patch["upward_longwave"],
        ground_sensible_derivative_patch=patch["ground_sensible_derivative"],
        ground_latent_derivative_patch=patch["ground_latent_derivative"],
        vegetation_transpiration_patch=patch["vegetation_transpiration"],
        vegetation_evaporation_patch=patch["vegetation_evaporation"],
        vegetation_sensible_heat_patch=patch["vegetation_sensible_heat"],
        waste_heat_patch=ground_patch.waste_heat_patch,
        air_conditioner_heat_patch=ground_patch.air_conditioner_heat_patch,
        traffic_heat_patch=ground_patch.traffic_heat_patch,
    )


def assemble_patch_to_column_inputs(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    soil_fluxes_result: SoilFluxesResult,
    soil_fluxes_inputs: SoilFluxesInputs,
) -> PatchToColumnFluxInputs:

    precipitation = physical.canopy.canopy_hydrology.fluxes.patch_precipitation
    return assemble_patch_to_column_inputs_from_precipitation(
        precipitation,
        soil_fluxes_result,
        soil_fluxes_inputs,
    )


def assemble_patch_to_column_inputs_from_precipitation(
    precipitation: CanopyPrecipitationFluxes,
    soil_fluxes_result: SoilFluxesResult,
    soil_fluxes_inputs: SoilFluxesInputs,
) -> PatchToColumnFluxInputs:

    water = soil_fluxes_result.state.water
    return PatchToColumnFluxInputs(
        snow_evaporation_patch=water.snow_evaporation_patch,
        bare_soil_evaporation_patch=water.bare_soil_evaporation_patch,
        surface_water_evaporation_patch=water.surface_water_evaporation_patch,
        soil_evaporation_patch=water.soil_evaporation_patch,
        total_evaporation_patch=water.total_evaporation_patch,
        ground_rainfall_patch=precipitation.ground_rain,
        ground_snowfall_patch=precipitation.ground_snow,
        vegetation_transpiration_patch=(
            soil_fluxes_inputs.vegetation_transpiration_patch
        ),
        ground_liquid_evaporation_patch=water.ground_liquid_evaporation_patch,
        ground_precipitation_patch=precipitation.ground_precipitation,
        ground_dew_patch=water.ground_dew_patch,
        snow_sublimation_patch=water.snow_sublimation_patch,
        snow_dew_patch=water.snow_dew_patch,
    )


def post_temperature_driver_step(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    inputs: PostTemperatureDriverInputs,
    state: PostTemperatureDriverState,
    parameters: PostTemperatureDriverParameters = (
        DEFAULT_POST_TEMPERATURE_DRIVER_PARAMETERS
    ),
) -> PostTemperatureDriverResult:

    topology = physical.soil_temperature.inputs.topology
    phase = physical.soil_temperature.temperature.soil_snow_phase_change
    glacier = handle_glacier_ice_melt(
        GlacierIceMeltInputs(
            do_surface_mass_balance=inputs.do_surface_mass_balance_column,
            landunit_type=topology.column.landunit_type,
            time_step_seconds=jnp.asarray(
                parameters.soil_fluxes.time_step_seconds,
                dtype=jnp.float64,
            ),
        ),
        GlacierIceMeltState(
            liquid_water=phase.liquid_water[:, NUMBER_OF_SNOW_LEVELS:],
            ice_water=phase.ice_water[:, NUMBER_OF_SNOW_LEVELS:],
            ice_melt_flux=state.glacier_ice_melt_flux_column,
        ),
    )
    soil_inputs = assemble_soil_fluxes_inputs(physical, inputs, glacier)
    soil_entry_state = assemble_soil_fluxes_entry_state(physical, inputs, state)
    soil_result = soil_fluxes(
        topology,
        soil_inputs,
        soil_entry_state,
        parameters=parameters.soil_fluxes,
    )
    patch_inputs = assemble_patch_to_column_inputs(
        physical,
        soil_result,
        soil_inputs,
    )
    patch_result = aggregate_patch_fluxes_to_columns(
        topology,
        patch_inputs,
        state.patch_to_column,
        parameters=parameters.patch_to_column,
    )
    return PostTemperatureDriverResult(
        glacier_ice_melt=glacier,
        soil_fluxes_inputs=soil_inputs,
        soil_fluxes_entry_state=soil_entry_state,
        soil_fluxes=soil_result,
        patch_to_column_inputs=patch_inputs,
        patch_to_column=patch_result,
    )


__all__ = [
    "PostTemperatureDriverInputs",
    "PostTemperatureDriverParameters",
    "PostTemperatureDriverResult",
    "PostTemperatureDriverState",
    "PostTemperatureNativeMap",
    "DEFAULT_POST_TEMPERATURE_DRIVER_PARAMETERS",
    "assemble_patch_to_column_inputs",
    "assemble_patch_to_column_inputs_from_precipitation",
    "assemble_soil_fluxes_entry_state",
    "assemble_soil_fluxes_inputs",
    "post_temperature_driver_step",
]
