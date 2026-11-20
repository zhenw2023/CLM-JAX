
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.forcing.datm import DatmForcing
from clm_jax.io.restart import SubgridTopology
from clm_jax.io.urban_fluxes_restart import (
    UrbanFluxesNativeMap,
    UrbanFluxesRestartState,
    UrbanFluxesStaticParameters,
    UrbanFluxesStepData,
    assemble_urban_fluxes_step_data,
)
from clm_jax.physics.canopy_driver import CanopyDriverResult
from clm_jax.physics.canopy_temperature import CanopyTemperatureParameters
from clm_jax.physics.canopy_temperature_driver import CanopyTemperatureDriverResult
from clm_jax.physics.urban_fluxes import (
    UrbanSurfaceFluxResult,
    UrbanTurbulenceResult,
    urban_surface_fluxes,
    urban_turbulent_fluxes,
)


class UrbanFluxDriverInputs(NamedTuple):

    native_map: UrbanFluxesNativeMap
    static_parameters: UrbanFluxesStaticParameters
    canopy_temperature_parameters: CanopyTemperatureParameters


class UrbanFluxDriverParameters(NamedTuple):

    maximum_stable_zeta: float | jnp.ndarray
    building_temperature_method: int | jnp.ndarray


UrbanFluxDriverState = UrbanFluxesRestartState


class UrbanFluxDriverCall(NamedTuple):

    step: UrbanFluxesStepData
    parameters: UrbanFluxDriverParameters
    previous_state: UrbanFluxDriverState


class UrbanFluxDriverResult(NamedTuple):

    step: UrbanFluxesStepData
    turbulence: UrbanTurbulenceResult
    surface: UrbanSurfaceFluxResult
    state: UrbanFluxDriverState


class UrbanFluxSoilTemperatureFields(NamedTuple):

    ground_energy_derivative_surface: jnp.ndarray
    downward_longwave_below_canopy_surface: jnp.ndarray
    upward_longwave_above_canopy_surface: jnp.ndarray
    ground_sensible_derivative_surface: jnp.ndarray
    ground_latent_derivative_surface: jnp.ndarray
    sensible_heat_snow_surface: jnp.ndarray
    sensible_heat_soil_surface: jnp.ndarray
    sensible_heat_surface_water_surface: jnp.ndarray
    sensible_heat_ground_surface: jnp.ndarray
    soil_evaporation_surface: jnp.ndarray
    vegetation_transpiration_surface: jnp.ndarray
    vegetation_evaporation_surface: jnp.ndarray
    vegetation_sensible_heat_surface: jnp.ndarray
    waste_heat_landunit: jnp.ndarray
    heat_from_air_conditioning_landunit: jnp.ndarray
    traffic_heat_landunit: jnp.ndarray
    canopy_air_temperature_landunit: jnp.ndarray
    internal_building_temperature_landunit: jnp.ndarray
    air_conditioning_flux_landunit: jnp.ndarray
    heating_flux_landunit: jnp.ndarray


def urban_flux_soil_temperature_fields(
    result: UrbanFluxDriverResult,
) -> UrbanFluxSoilTemperatureFields:

    energy = result.surface.energy
    water = result.surface.water
    diagnostics = result.turbulence.diagnostics
    return UrbanFluxSoilTemperatureFields(
        ground_energy_derivative_surface=energy.ground_energy_derivative_surface,
        downward_longwave_below_canopy_surface=(energy.downward_longwave_below_canopy_surface),
        upward_longwave_above_canopy_surface=(energy.upward_longwave_above_canopy_surface),
        ground_sensible_derivative_surface=(energy.ground_sensible_derivative_surface),
        ground_latent_derivative_surface=energy.ground_latent_derivative_surface,
        sensible_heat_snow_surface=energy.sensible_heat_snow_surface,
        sensible_heat_soil_surface=energy.sensible_heat_soil_surface,
        sensible_heat_surface_water_surface=(energy.sensible_heat_surface_water_surface),
        sensible_heat_ground_surface=energy.sensible_heat_ground_surface,
        soil_evaporation_surface=water.soil_evaporation_surface,
        vegetation_transpiration_surface=water.vegetation_transpiration_surface,
        vegetation_evaporation_surface=water.vegetation_evaporation_surface,
        vegetation_sensible_heat_surface=jnp.zeros_like(
            energy.sensible_heat_ground_surface
        ),
        waste_heat_landunit=diagnostics.waste_heat_landunit,
        heat_from_air_conditioning_landunit=(diagnostics.heat_from_air_conditioning_landunit),
        traffic_heat_landunit=diagnostics.traffic_heat_landunit,
        canopy_air_temperature_landunit=(result.turbulence.state.canopy_air_temperature_landunit),
        internal_building_temperature_landunit=(
            result.state.internal_building_temperature_landunit
        ),
        air_conditioning_flux_landunit=result.state.air_conditioning_flux_landunit,
        heating_flux_landunit=result.state.heating_flux_landunit,
    )


def assemble_urban_flux_driver_call(
    topology: SubgridTopology,
    grid_forcing: DatmForcing,
    canopy: CanopyDriverResult,
    temperature: CanopyTemperatureDriverResult,
    inputs: UrbanFluxDriverInputs,
    parameters: UrbanFluxDriverParameters,
    state: UrbanFluxDriverState,
) -> UrbanFluxDriverCall:

    step = assemble_urban_fluxes_step_data(
        topology,
        inputs.native_map,
        inputs.static_parameters,
        state,
        grid_forcing,
        temperature.inputs,
        inputs.canopy_temperature_parameters,
        temperature.state,
        canopy.state.canopy_hydrology,
    )
    return UrbanFluxDriverCall(
        step=step,
        parameters=parameters,
        previous_state=state,
    )


def solve_urban_flux_driver_call(
    call: UrbanFluxDriverCall,
) -> tuple[UrbanTurbulenceResult, UrbanSurfaceFluxResult]:

    turbulence = urban_turbulent_fluxes(
        call.step.turbulence_inputs,
        call.step.turbulence_state,
        call.parameters.maximum_stable_zeta,
        call.parameters.building_temperature_method,
    )
    surface = urban_surface_fluxes(
        call.step.turbulence_inputs,
        call.step.surface_inputs,
        turbulence,
        call.parameters.building_temperature_method,
    )
    return turbulence, surface


def complete_urban_flux_driver_call(
    call: UrbanFluxDriverCall,
    turbulence: UrbanTurbulenceResult,
    surface: UrbanSurfaceFluxResult,
) -> UrbanFluxDriverResult:

    state = call.previous_state._replace(
        turbulence=turbulence.state,
        internal_building_temperature_landunit=(surface.internal_building_temperature_landunit),
    )
    return UrbanFluxDriverResult(
        step=call.step,
        turbulence=turbulence,
        surface=surface,
        state=state,
    )


def urban_flux_driver_step(
    topology: SubgridTopology,
    grid_forcing: DatmForcing,
    canopy: CanopyDriverResult,
    temperature: CanopyTemperatureDriverResult,
    inputs: UrbanFluxDriverInputs,
    parameters: UrbanFluxDriverParameters,
    state: UrbanFluxDriverState,
) -> UrbanFluxDriverResult:

    call = assemble_urban_flux_driver_call(
        topology,
        grid_forcing,
        canopy,
        temperature,
        inputs,
        parameters,
        state,
    )
    turbulence, surface = solve_urban_flux_driver_call(call)
    return complete_urban_flux_driver_call(call, turbulence, surface)


__all__ = [
    "UrbanFluxDriverCall",
    "UrbanFluxDriverInputs",
    "UrbanFluxDriverParameters",
    "UrbanFluxDriverResult",
    "UrbanFluxDriverState",
    "UrbanFluxSoilTemperatureFields",
    "assemble_urban_flux_driver_call",
    "complete_urban_flux_driver_call",
    "solve_urban_flux_driver_call",
    "urban_flux_driver_step",
    "urban_flux_soil_temperature_fields",
]
