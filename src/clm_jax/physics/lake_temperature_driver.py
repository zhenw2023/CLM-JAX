
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.canopy_driver import CanopyDriverResult
from clm_jax.physics.canopy_temperature_driver import CanopyTemperatureDriverResult
from clm_jax.physics.lake_flux_driver import LakeFluxDriverResult, LakeFluxNativeMap
from clm_jax.physics.lake_temperature import (
    NUMBER_OF_SNOW_LEVELS,
    LakeColumnTemperatureState,
    LakeTemperatureGeometry,
    LakeTemperatureInputs,
    LakeTemperatureResult,
    lake_temperature,
)
from clm_jax.physics.radiation_sequence import RadiationSequenceResult


class LakeTemperatureDriverInputs(NamedTuple):

    native_map: LakeFluxNativeMap
    geometry: LakeTemperatureGeometry


class LakeTemperatureDriverCall(NamedTuple):

    inputs: LakeTemperatureInputs
    geometry: LakeTemperatureGeometry
    state: LakeColumnTemperatureState


class LakeTemperatureDriverResult(NamedTuple):

    call: LakeTemperatureDriverCall
    temperature: LakeTemperatureResult
    state: LakeColumnTemperatureState


def assemble_lake_temperature_driver_call(
    topology: SubgridTopology,
    canopy: CanopyDriverResult,
    radiation: RadiationSequenceResult,
    temperature: CanopyTemperatureDriverResult,
    lake_flux: LakeFluxDriverResult,
    inputs: LakeTemperatureDriverInputs,
    state: LakeColumnTemperatureState,
) -> LakeTemperatureDriverCall:

    native = inputs.native_map
    column = jnp.asarray(native.column_index, dtype=jnp.int32)
    patch = jnp.asarray(native.patch_index, dtype=jnp.int32)
    lake_count = column.shape[0]
    if patch.shape != (lake_count,):
        raise ValueError("lake column and patch maps must have identical lengths")
    if inputs.geometry.lake_depth.shape != (lake_count,):
        raise ValueError("lake-temperature geometry must use the compact native lake axis")

    hydrology = canopy.state.canopy_hydrology
    snow = hydrology.snow
    radiation_surface = radiation.state.canopy_surface.surface_radiation
    radiation_fluxes = radiation_surface.fluxes
    radiation_diagnostics = radiation_surface.diagnostics
    temperature_inputs = temperature.inputs
    surface = lake_flux.fluxes.surface
    energy = lake_flux.fluxes.energy

    dynamic_geometry = inputs.geometry._replace(
        soil_snow_layer_thickness=jnp.concatenate(
            (
                snow.layer_thickness[column],
                inputs.geometry.soil_snow_layer_thickness[:, NUMBER_OF_SNOW_LEVELS:],
            ),
            axis=1,
        ),
        soil_snow_node_depth=jnp.concatenate(
            (
                snow.layer_depth[column],
                inputs.geometry.soil_snow_node_depth[:, NUMBER_OF_SNOW_LEVELS:],
            ),
            axis=1,
        ),
        soil_snow_interface_depth=jnp.concatenate(
            (
                snow.layer_interface_depth[column, 1:],
                inputs.geometry.soil_snow_interface_depth[:, NUMBER_OF_SNOW_LEVELS:],
            ),
            axis=1,
        ),
    )

    current_state = state._replace(
        snow_layer_count=jnp.asarray(snow.layer_count[column], dtype=jnp.int32),
        snow_depth=jnp.asarray(snow.depth[column], dtype=jnp.float64),
        snow_water=jnp.asarray(snow.water_equivalent[column], dtype=jnp.float64),
        soil_snow_temperature=jnp.asarray(
            temperature_inputs.soil_snow_temperature_column_level[column],
            dtype=jnp.float64,
        ),
        soil_snow_liquid_water=jnp.asarray(
            temperature_inputs.soil_liquid_water_column_level[column],
            dtype=jnp.float64,
        ),
        soil_snow_ice_water=jnp.asarray(
            temperature_inputs.soil_ice_water_column_level[column],
            dtype=jnp.float64,
        ),
        soil_snow_old_ice_fraction=jnp.concatenate(
            (
                jnp.asarray(snow.old_ice_fraction[column], dtype=jnp.float64),
                state.soil_snow_old_ice_fraction[:, NUMBER_OF_SNOW_LEVELS:],
            ),
            axis=1,
        ),
        lake_temperature=jnp.asarray(
            lake_flux.state.lake_temperature_lake_level,
            dtype=jnp.float64,
        ),
        saved_top_eddy_conductivity=jnp.asarray(
            lake_flux.state.saved_top_eddy_conductivity_lake,
            dtype=jnp.float64,
        ),
        surface_absorption_fraction=jnp.asarray(
            surface.surface_absorption_fraction,
            dtype=jnp.float64,
        ),
    )

    dynamic_inputs = LakeTemperatureInputs(
        active_lake=jnp.ones((lake_count,), dtype=jnp.bool_),
        ground_temperature=jnp.asarray(
            lake_flux.fluxes.temperature.ground_temperature,
            dtype=jnp.float64,
        ),
        ground_absorbed_shortwave=jnp.asarray(
            radiation_fluxes.absorbed_ground_patch[patch],
            dtype=jnp.float64,
        ),
        layer_absorbed_shortwave=jnp.asarray(
            radiation_fluxes.absorbed_ground_layers_patch[patch],
            dtype=jnp.float64,
        ),
        incident_nir_direct=jnp.asarray(
            radiation_diagnostics.incident_direct_patch[patch, 1],
            dtype=jnp.float64,
        ),
        incident_nir_diffuse=jnp.asarray(
            radiation_diagnostics.incident_diffuse_patch[patch, 1],
            dtype=jnp.float64,
        ),
        reflected_nir_direct=jnp.asarray(
            radiation_diagnostics.reflected_direct_patch[patch, 1],
            dtype=jnp.float64,
        ),
        reflected_nir_diffuse=jnp.asarray(
            radiation_diagnostics.reflected_diffuse_patch[patch, 1],
            dtype=jnp.float64,
        ),
        wind_mixing_velocity=jnp.asarray(surface.wind_mixing_velocity, dtype=jnp.float64),
        mixing_decay_coefficient=jnp.asarray(
            surface.mixing_decay_coefficient,
            dtype=jnp.float64,
        ),
        lake_moisture_aerodynamic_resistance=jnp.asarray(
            surface.moisture_resistance,
            dtype=jnp.float64,
        ),
        soil_ground_heat_flux=jnp.asarray(energy.soil_ground_heat, dtype=jnp.float64),
        sensible_ground_heat_flux=jnp.asarray(energy.sensible_ground, dtype=jnp.float64),
        sensible_total_heat_flux=jnp.asarray(energy.sensible_total, dtype=jnp.float64),
        ground_net_heat_flux=jnp.asarray(energy.ground_net, dtype=jnp.float64),
    )
    return LakeTemperatureDriverCall(
        inputs=dynamic_inputs,
        geometry=dynamic_geometry,
        state=current_state,
    )


def lake_temperature_driver_step(
    topology: SubgridTopology,
    canopy: CanopyDriverResult,
    radiation: RadiationSequenceResult,
    temperature: CanopyTemperatureDriverResult,
    lake_flux: LakeFluxDriverResult,
    inputs: LakeTemperatureDriverInputs,
    state: LakeColumnTemperatureState,
    *,
    time_step_seconds: float = 1800.0,
) -> LakeTemperatureDriverResult:

    call = assemble_lake_temperature_driver_call(
        topology,
        canopy,
        radiation,
        temperature,
        lake_flux,
        inputs,
        state,
    )
    result = lake_temperature(
        call.inputs,
        call.geometry,
        call.state,
        time_step_seconds=time_step_seconds,
    )
    return LakeTemperatureDriverResult(
        call=call,
        temperature=result,
        state=result.state,
    )


__all__ = [
    "LakeTemperatureDriverCall",
    "LakeTemperatureDriverInputs",
    "LakeTemperatureDriverResult",
    "assemble_lake_temperature_driver_call",
    "lake_temperature_driver_step",
]
