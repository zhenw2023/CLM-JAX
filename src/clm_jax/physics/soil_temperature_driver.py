
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.io.restart import LANDUNIT_LAKE, SubgridTopology
from clm_jax.physics.atmospheric_driver import AtmosphericDriverResult
from clm_jax.physics.canopy_driver import CanopyDriverResult
from clm_jax.physics.canopy_flux_driver import (
    CanopyFluxDriverResult,
    CanopyFluxSoilTemperatureFields,
    canopy_flux_soil_temperature_fields,
)
from clm_jax.physics.canopy_temperature import NUMBER_OF_SNOW_LEVELS
from clm_jax.physics.canopy_temperature_driver import CanopyTemperatureDriverResult
from clm_jax.physics.radiation_sequence import RadiationSequenceResult
from clm_jax.physics.soil_temperature_step import (
    SoilTemperatureActiveInputs,
    SoilTemperatureActiveResult,
    SoilTemperatureRuntimeResult,
    soil_temperature_active_case,
)
from clm_jax.physics.urban_flux_driver import (
    UrbanFluxDriverInputs,
    UrbanFluxDriverResult,
    UrbanFluxSoilTemperatureFields,
    urban_flux_soil_temperature_fields,
)


class SoilTemperatureDriverInputs(NamedTuple):

    active_template: SoilTemperatureActiveInputs
    urban: UrbanFluxDriverInputs


class SoilTemperatureDriverResult(NamedTuple):

    inputs: SoilTemperatureActiveInputs
    temperature: SoilTemperatureActiveResult | SoilTemperatureRuntimeResult


def _scatter_surface(
    base: jnp.ndarray,
    patch_index: jnp.ndarray,
    compact: jnp.ndarray,
) -> jnp.ndarray:

    index = jnp.asarray(patch_index, dtype=jnp.int32)
    values = jnp.asarray(compact, dtype=jnp.asarray(base).dtype)
    if index.ndim != 2:
        raise ValueError("urban surface patch map must be two-dimensional")
    try:
        values = jnp.broadcast_to(values, index.shape)
    except ValueError as error:
        raise ValueError("urban surface values and native patch map disagree") from error
    return jnp.asarray(base).at[jnp.reshape(index, (-1,))].set(jnp.reshape(values, (-1,)))


def _scatter_landunit(
    base: jnp.ndarray,
    landunit_index: jnp.ndarray,
    compact: jnp.ndarray,
) -> jnp.ndarray:

    index = jnp.asarray(landunit_index, dtype=jnp.int32)
    values = jnp.asarray(compact, dtype=jnp.asarray(base).dtype)
    if index.ndim != 1 or values.shape != index.shape:
        raise ValueError("urban landunit values and native map disagree")
    return jnp.asarray(base).at[index].set(values)


def _current_soil_snow_geometry(
    template: SoilTemperatureActiveInputs,
    canopy: CanopyDriverResult,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    thermal = template.thermal_properties
    snow = canopy.state.canopy_hydrology.snow
    layer_thickness = jnp.concatenate(
        (
            jnp.asarray(snow.layer_thickness, dtype=jnp.float64),
            thermal.layer_thickness[:, NUMBER_OF_SNOW_LEVELS:],
        ),
        axis=1,
    )
    node_depth = jnp.concatenate(
        (
            jnp.asarray(snow.layer_depth, dtype=jnp.float64),
            thermal.node_depth[:, NUMBER_OF_SNOW_LEVELS:],
        ),
        axis=1,
    )
    interface_depth = jnp.concatenate(
        (
            jnp.asarray(snow.layer_interface_depth, dtype=jnp.float64),
            thermal.interface_depth[:, NUMBER_OF_SNOW_LEVELS + 1 :],
        ),
        axis=1,
    )
    if layer_thickness.shape != thermal.layer_thickness.shape:
        raise ValueError("current snow and static ground thicknesses do not join")
    if node_depth.shape != thermal.node_depth.shape:
        raise ValueError("current snow and static ground node depths do not join")
    if interface_depth.shape != thermal.interface_depth.shape:
        raise ValueError("current snow and static ground interfaces do not join")
    return layer_thickness, node_depth, interface_depth


def assemble_soil_temperature_driver_inputs(
    topology: SubgridTopology,
    atmospheric: AtmosphericDriverResult,
    canopy: CanopyDriverResult,
    radiation: RadiationSequenceResult,
    temperature: CanopyTemperatureDriverResult,
    canopy_flux: CanopyFluxDriverResult | CanopyFluxSoilTemperatureFields,
    urban: UrbanFluxDriverResult | UrbanFluxSoilTemperatureFields,
    inputs: SoilTemperatureDriverInputs,
) -> SoilTemperatureActiveInputs:

    template = inputs.active_template
    current_column = topology.column._replace(
        elevation=atmospheric.state.column_topography.elevation,
        column_type=atmospheric.state.column_topography.column_type,
    )
    current_topology = topology._replace(column=current_column)
    hydrology = canopy.state.canopy_hydrology
    snow = hydrology.snow
    temperature_inputs = temperature.inputs
    temperature_column = temperature.state.column
    layer_thickness, node_depth, interface_depth = _current_soil_snow_geometry(
        template,
        canopy,
    )

    active_nonlake = current_column.active & (
        current_column.landunit_type != jnp.int32(LANDUNIT_LAKE)
    )
    thermal = template.thermal_properties._replace(
        active_nonlake=active_nonlake,
        snow_layer_count=jnp.asarray(snow.layer_count, dtype=jnp.int32),
        landunit_type=jnp.asarray(current_column.landunit_type, dtype=jnp.int32),
        column_type=jnp.asarray(current_column.column_type, dtype=jnp.int32),
        layer_thickness=layer_thickness,
        node_depth=node_depth,
        interface_depth=interface_depth,
        soil_snow_temperature=jnp.asarray(
            temperature_inputs.soil_snow_temperature_column_level,
            dtype=jnp.float64,
        ),
        liquid_water=jnp.asarray(
            temperature_inputs.soil_liquid_water_column_level,
            dtype=jnp.float64,
        ),
        ice_water=jnp.asarray(
            temperature_inputs.soil_ice_water_column_level,
            dtype=jnp.float64,
        ),
        effective_snow_fraction=jnp.asarray(
            snow.effective_snow_fraction,
            dtype=jnp.float64,
        ),
        surface_water=jnp.asarray(
            hydrology.surface_water_column,
            dtype=jnp.float64,
        ),
        snow_water=jnp.asarray(snow.water_equivalent, dtype=jnp.float64),
    )

    canopy_fields = (
        canopy_flux
        if isinstance(canopy_flux, CanopyFluxSoilTemperatureFields)
        else canopy_flux_soil_temperature_fields(canopy_flux)
    )
    urban_fields = (
        urban
        if isinstance(urban, UrbanFluxSoilTemperatureFields)
        else urban_flux_soil_temperature_fields(urban)
    )
    urban_map = inputs.urban.native_map
    urban_patch = urban_map.surface_patch_index

    snow_evaporation = _scatter_surface(
        canopy_fields.snow_evaporation_patch,
        urban_patch,
        jnp.zeros_like(urban_fields.soil_evaporation_surface),
    )
    soil_evaporation = _scatter_surface(
        canopy_fields.soil_evaporation_patch,
        urban_patch,
        jnp.zeros_like(urban_fields.soil_evaporation_surface),
    )
    surface_water_evaporation = _scatter_surface(
        canopy_fields.surface_water_evaporation_patch,
        urban_patch,
        jnp.zeros_like(urban_fields.soil_evaporation_surface),
    )
    total_soil_evaporation = _scatter_surface(
        canopy_fields.ground_evaporation_patch,
        urban_patch,
        urban_fields.soil_evaporation_surface,
    )
    vegetation_transpiration = _scatter_surface(
        canopy_fields.vegetation_transpiration_patch,
        urban_patch,
        urban_fields.vegetation_transpiration_surface,
    )
    ground_derivative = _scatter_surface(
        canopy_fields.ground_energy_derivative_patch,
        urban_patch,
        urban_fields.ground_energy_derivative_surface,
    )
    canopy_longwave = _scatter_surface(
        canopy_fields.downward_longwave_below_canopy_patch,
        urban_patch,
        urban_fields.downward_longwave_below_canopy_surface,
    )
    snow_sensible = _scatter_surface(
        canopy_fields.sensible_heat_snow_patch,
        urban_patch,
        urban_fields.sensible_heat_snow_surface,
    )
    soil_sensible = _scatter_surface(
        canopy_fields.sensible_heat_soil_patch,
        urban_patch,
        urban_fields.sensible_heat_soil_surface,
    )
    water_sensible = _scatter_surface(
        canopy_fields.sensible_heat_surface_water_patch,
        urban_patch,
        urban_fields.sensible_heat_surface_water_surface,
    )
    ground_sensible = _scatter_surface(
        canopy_fields.sensible_heat_ground_patch,
        urban_patch,
        urban_fields.sensible_heat_ground_surface,
    )

    surface_radiation = radiation.state.canopy_surface.surface_radiation.fluxes
    urban_radiation = radiation.state.urban.fluxes
    landunit_index = urban_map.landunit_index
    urban_static = inputs.urban.static_parameters
    ground_template = template.ground_heat_flux
    ground = ground_template._replace(
        snow_layer_count_column=thermal.snow_layer_count,
        downward_longwave_column=jnp.asarray(
            atmospheric.column_forcing.longwave_down,
            dtype=jnp.float64,
        ),
        effective_snow_fraction_column=thermal.effective_snow_fraction,
        ground_emissivity_column=jnp.asarray(
            temperature_column.ground_emissivity_column,
            dtype=jnp.float64,
        ),
        surface_water_temperature_column=jnp.asarray(
            temperature_inputs.surface_water_temperature_column,
            dtype=jnp.float64,
        ),
        ground_temperature_column=jnp.asarray(
            temperature_column.ground_temperature_column,
            dtype=jnp.float64,
        ),
        soil_snow_temperature_column=thermal.soil_snow_temperature,
        latent_heat_column=jnp.asarray(
            temperature_column.latent_heat_column,
            dtype=jnp.float64,
        ),
        vegetation_exposed_patch=jnp.asarray(
            atmospheric.driver_initialization.vegetation_exposed,
            dtype=jnp.int32,
        ),
        snow_evaporation_patch=snow_evaporation,
        soil_evaporation_patch=soil_evaporation,
        surface_water_evaporation_patch=surface_water_evaporation,
        total_soil_evaporation_patch=total_soil_evaporation,
        vegetation_transpiration_patch=vegetation_transpiration,
        ground_energy_derivative_patch=ground_derivative,
        canopy_downward_longwave_patch=canopy_longwave,
        snow_sensible_heat_patch=snow_sensible,
        soil_sensible_heat_patch=soil_sensible,
        surface_water_sensible_heat_patch=water_sensible,
        ground_sensible_heat_patch=ground_sensible,
        net_longwave_patch=jnp.asarray(
            urban_radiation.net_longwave_patch,
            dtype=jnp.float64,
        ),
        absorbed_ground_patch=jnp.asarray(
            surface_radiation.absorbed_ground_patch,
            dtype=jnp.float64,
        ),
        absorbed_soil_patch=jnp.asarray(
            surface_radiation.absorbed_ground_soil_patch,
            dtype=jnp.float64,
        ),
        absorbed_snow_patch=jnp.asarray(
            surface_radiation.absorbed_ground_snow_patch,
            dtype=jnp.float64,
        ),
        absorbed_ground_layers_patch=jnp.asarray(
            surface_radiation.absorbed_ground_layers_patch,
            dtype=jnp.float64,
        ),
        waste_heat_landunit=_scatter_landunit(
            ground_template.waste_heat_landunit,
            landunit_index,
            urban_fields.waste_heat_landunit,
        ),
        air_conditioner_heat_landunit=_scatter_landunit(
            ground_template.air_conditioner_heat_landunit,
            landunit_index,
            urban_fields.heat_from_air_conditioning_landunit,
        ),
        traffic_heat_landunit=_scatter_landunit(
            ground_template.traffic_heat_landunit,
            landunit_index,
            urban_fields.traffic_heat_landunit,
        ),
        urban_roof_fraction_landunit=_scatter_landunit(
            ground_template.urban_roof_fraction_landunit,
            landunit_index,
            urban_static.roof_fraction_landunit,
        ),
    )

    water_fraction = jnp.asarray(
        hydrology.surface_water_fraction_column,
        dtype=jnp.float64,
    )
    water_temperature = jnp.asarray(
        temperature_inputs.surface_water_temperature_column,
        dtype=jnp.float64,
    )
    heat_diffusion = template.heat_diffusion_template._replace(
        active_nonlake=active_nonlake,
        snow_layer_count=thermal.snow_layer_count,
        column_type=thermal.column_type,
    )
    rhs = template.rhs_template._replace(
        active_nonlake=active_nonlake,
        snow_layer_count=thermal.snow_layer_count,
        landunit_type=thermal.landunit_type,
        column_type=thermal.column_type,
        standing_water_temperature=water_temperature,
        standing_water_fraction=water_fraction,
        effective_snow_fraction=thermal.effective_snow_fraction,
    )
    matrix = template.matrix_template._replace(
        active_nonlake=active_nonlake,
        snow_layer_count=thermal.snow_layer_count,
        landunit_type=thermal.landunit_type,
        column_type=thermal.column_type,
        standing_water_fraction=water_fraction,
        effective_snow_fraction=thermal.effective_snow_fraction,
    )
    linear_solve = template.linear_solve_template._replace(
        active_nonlake=active_nonlake,
        snow_layer_count=thermal.snow_layer_count,
        column_type=thermal.column_type,
    )
    solution_update = template.solution_update_template._replace(
        active_nonlake=active_nonlake,
        snow_layer_count=thermal.snow_layer_count,
        column_type=thermal.column_type,
        standing_water_fraction=water_fraction,
        previous_soil_snow_temperature=thermal.soil_snow_temperature,
        previous_standing_water_temperature=water_temperature,
    )
    postsolve = template.postsolve_flux_template._replace(
        active_nonlake=active_nonlake,
        snow_layer_count=thermal.snow_layer_count,
        urban_landunit=(thermal.landunit_type >= jnp.int32(7))
        & (thermal.landunit_type <= jnp.int32(9)),
        column_type=thermal.column_type,
        previous_soil_snow_temperature=jnp.asarray(
            temperature_column.temperature_before_column_level,
            dtype=jnp.float64,
        ),
    )
    standing_phase = template.standing_water_phase_change_template._replace(
        active_nonlake=active_nonlake,
        snow_layer_count=thermal.snow_layer_count,
        effective_snow_fraction=thermal.effective_snow_fraction,
        standing_water_fraction=water_fraction,
        snow_water=thermal.snow_water,
        standing_water=thermal.surface_water,
        integrated_snowfall=jnp.asarray(snow.integrated_snow, dtype=jnp.float64),
        snow_depth=jnp.asarray(snow.depth, dtype=jnp.float64),
        ice_water=thermal.ice_water,
    )
    soil_phase = template.soil_snow_phase_change_template._replace(
        active_nonlake=active_nonlake,
        snow_layer_count=thermal.snow_layer_count,
        landunit_type=thermal.landunit_type,
        column_type=thermal.column_type,
        layer_thickness=thermal.layer_thickness,
        effective_snow_fraction=thermal.effective_snow_fraction,
        standing_water_fraction=water_fraction,
        snow_depth=standing_phase.snow_depth,
        snow_water=thermal.snow_water,
        liquid_water=thermal.liquid_water,
        ice_water=thermal.ice_water,
    )

    building_template = template.building_temperature_template
    building = building_template._replace(
        active_urban=jnp.asarray(topology.landunit.active, dtype=jnp.bool_)
        & (jnp.asarray(topology.landunit.landunit_type) >= jnp.int32(7))
        & (jnp.asarray(topology.landunit.landunit_type) <= jnp.int32(9)),
        roof_height=_scatter_landunit(
            building_template.roof_height,
            landunit_index,
            urban_static.roof_height_landunit,
        ),
        canyon_height_width_ratio=_scatter_landunit(
            building_template.canyon_height_width_ratio,
            landunit_index,
            urban_static.canyon_height_width_ratio_landunit,
        ),
        roof_weight=_scatter_landunit(
            building_template.roof_weight,
            landunit_index,
            urban_static.roof_fraction_landunit,
        ),
        canyon_air_temperature=_scatter_landunit(
            building_template.canyon_air_temperature,
            landunit_index,
            urban_fields.canopy_air_temperature_landunit,
        ),
        building_temperature=_scatter_landunit(
            building_template.building_temperature,
            landunit_index,
            urban_fields.internal_building_temperature_landunit,
        ),
        previous_air_conditioning_flux=_scatter_landunit(
            building_template.previous_air_conditioning_flux,
            landunit_index,
            urban_fields.air_conditioning_flux_landunit,
        ),
        previous_heating_flux=_scatter_landunit(
            building_template.previous_heating_flux,
            landunit_index,
            urban_fields.heating_flux_landunit,
        ),
    )
    final = template.final_diagnostics_template._replace(
        active_nonlake=active_nonlake,
        snow_layer_count=thermal.snow_layer_count,
        landunit_type=thermal.landunit_type,
        effective_snow_fraction=thermal.effective_snow_fraction,
        standing_water_fraction=water_fraction,
        previous_ground_temperature=jnp.asarray(
            temperature_column.ground_temperature_column,
            dtype=jnp.float64,
        ),
    )
    return SoilTemperatureActiveInputs(
        topology=current_topology,
        urban_columns=template.urban_columns,
        thermal_properties=thermal,
        ground_heat_flux=ground,
        ground_heat_flux_patch_state=template.ground_heat_flux_patch_state,
        heat_diffusion_template=heat_diffusion,
        rhs_template=rhs,
        matrix_template=matrix,
        linear_solve_template=linear_solve,
        solution_update_template=solution_update,
        postsolve_flux_template=postsolve,
        standing_water_phase_change_template=standing_phase,
        soil_snow_phase_change_template=soil_phase,
        building_temperature_template=building,
        final_diagnostics_template=final,
    )


def soil_temperature_driver_step(
    topology: SubgridTopology,
    atmospheric: AtmosphericDriverResult,
    canopy: CanopyDriverResult,
    radiation: RadiationSequenceResult,
    temperature: CanopyTemperatureDriverResult,
    canopy_flux: CanopyFluxDriverResult | CanopyFluxSoilTemperatureFields,
    urban: UrbanFluxDriverResult | UrbanFluxSoilTemperatureFields,
    inputs: SoilTemperatureDriverInputs,
    *,
    time_step_seconds: float = 1800.0,
    maximum_patches_per_column: int = 79,
) -> SoilTemperatureDriverResult:

    assembled = assemble_soil_temperature_driver_inputs(
        topology,
        atmospheric,
        canopy,
        radiation,
        temperature,
        canopy_flux,
        urban,
        inputs,
    )
    result = soil_temperature_active_case(
        assembled,
        time_step_seconds=time_step_seconds,
        maximum_patches_per_column=maximum_patches_per_column,
    )
    return SoilTemperatureDriverResult(inputs=assembled, temperature=result)


__all__ = [
    "SoilTemperatureDriverInputs",
    "SoilTemperatureDriverResult",
    "assemble_soil_temperature_driver_inputs",
    "soil_temperature_driver_step",
]
