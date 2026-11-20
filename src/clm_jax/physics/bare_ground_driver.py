
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.forcing.datm import DatmForcing
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.atmospheric_driver import AtmosphericDriverResult
from clm_jax.physics.bare_ground_fluxes import (
    BareGroundFluxInputs,
    BareGroundFluxState,
    bare_ground_fluxes,
)
from clm_jax.physics.canopy_driver import CanopyDriverResult
from clm_jax.physics.canopy_temperature import (
    CanopyTemperatureParameters,
    CanopyTemperatureState,
)
from clm_jax.physics.canopy_temperature_driver import CanopyTemperatureDriverResult


class BareGroundDriverParameters(NamedTuple):

    canopy_temperature: CanopyTemperatureParameters
    maximum_stable_zeta: float


class BareGroundDriverResult(NamedTuple):

    inputs: BareGroundFluxInputs
    state: BareGroundFluxState


def synchronize_bare_ground_flux_state(
    state: BareGroundFluxState,
    temperature: CanopyTemperatureState,
) -> BareGroundFluxState:

    return state._replace(
        energy=state.energy._replace(
            sensible_heat_total_patch=(
                temperature.patch.sensible_heat_total_patch
            ),
            ground_sensible_derivative_patch=(
                temperature.patch.ground_sensible_derivative_patch
            ),
            ground_latent_derivative_patch=(
                temperature.patch.ground_latent_derivative_patch
            ),
            ground_energy_derivative_patch=(
                temperature.patch.ground_energy_derivative_patch
            ),
        ),
        water=state.water._replace(
            total_evaporation_patch=temperature.patch.total_evaporation_patch,
        ),
    )


def assemble_bare_ground_flux_inputs(
    atmospheric: AtmosphericDriverResult,
    grid_forcing: DatmForcing,
    canopy: CanopyDriverResult,
    temperature: CanopyTemperatureDriverResult,
    parameters: CanopyTemperatureParameters,
) -> BareGroundFluxInputs:

    forcing = atmospheric.column_forcing
    hydrology = canopy.state.canopy_hydrology
    temperature_inputs = temperature.inputs
    column = temperature.state.column
    patch = temperature.state.patch
    return BareGroundFluxInputs(
        vegetation_exposed_patch=(
            atmospheric.driver_initialization.vegetation_exposed
        ),
        eastward_wind_gridcell=grid_forcing.wind_u,
        northward_wind_gridcell=grid_forcing.wind_v,
        potential_temperature_column=forcing.potential_temperature,
        temperature_column=forcing.temperature,
        pressure_column=forcing.pressure,
        density_column=forcing.density,
        specific_humidity_column=forcing.specific_humidity,
        snow_layer_count_column=hydrology.snow.layer_count,
        ground_layer_thickness_column=parameters.ground_layer_thickness_column,
        saturated_water_content_column=parameters.saturated_water_content_column,
        soil_snow_temperature_column_level=(
            temperature_inputs.soil_snow_temperature_column_level
        ),
        soil_liquid_water_column_level=(
            temperature_inputs.soil_liquid_water_column_level
        ),
        soil_ice_water_column_level=(
            temperature_inputs.soil_ice_water_column_level
        ),
        surface_water_temperature_column=(
            temperature_inputs.surface_water_temperature_column
        ),
        ground_temperature_column=column.ground_temperature_column,
        virtual_potential_temperature_column=(
            column.virtual_potential_temperature_column
        ),
        convective_velocity_coefficient_column=(
            column.convective_velocity_coefficient_column
        ),
        convective_boundary_height_column=(
            column.convective_boundary_height_column
        ),
        soil_evaporation_resistance_column=(
            column.soil_evaporation_resistance_column
        ),
        snow_surface_specific_humidity_column=(
            column.snow_surface_specific_humidity_column
        ),
        soil_surface_specific_humidity_column=(
            column.soil_surface_specific_humidity_column
        ),
        surface_water_specific_humidity_column=(
            column.surface_water_specific_humidity_column
        ),
        ground_specific_humidity_column=column.ground_specific_humidity_column,
        ground_specific_humidity_temperature_derivative_column=(
            column.ground_specific_humidity_temperature_derivative_column
        ),
        latent_heat_column=column.latent_heat_column,
        ground_momentum_roughness_column=(
            column.ground_momentum_roughness_column
        ),
        ground_sensible_roughness_column=(
            column.ground_sensible_roughness_column
        ),
        ground_latent_roughness_column=column.ground_latent_roughness_column,
        forcing_height_wind_patch=patch.forcing_height_wind_patch,
        forcing_height_temperature_patch=patch.forcing_height_temperature_patch,
        forcing_height_humidity_patch=patch.forcing_height_humidity_patch,
        reference_temperature_patch=patch.reference_temperature_patch,
    )


def bare_ground_driver_step(
    topology: SubgridTopology,
    atmospheric: AtmosphericDriverResult,
    grid_forcing: DatmForcing,
    canopy: CanopyDriverResult,
    temperature: CanopyTemperatureDriverResult,
    parameters: BareGroundDriverParameters,
    state: BareGroundFluxState,
) -> BareGroundDriverResult:

    inputs = assemble_bare_ground_flux_inputs(
        atmospheric,
        grid_forcing,
        canopy,
        temperature,
        parameters.canopy_temperature,
    )
    column_topography = atmospheric.state.column_topography
    current_topology = topology._replace(
        column=topology.column._replace(
            elevation=column_topography.elevation,
            column_type=column_topography.column_type,
        )
    )
    synchronized_state = synchronize_bare_ground_flux_state(
        state,
        temperature.state,
    )
    state = bare_ground_fluxes(
        current_topology,
        inputs,
        synchronized_state,
        maximum_stable_zeta=jnp.asarray(
            parameters.maximum_stable_zeta,
            dtype=jnp.float64,
        ),
    )
    return BareGroundDriverResult(inputs=inputs, state=state)


__all__ = [
    "BareGroundDriverParameters",
    "BareGroundDriverResult",
    "assemble_bare_ground_flux_inputs",
    "bare_ground_driver_step",
    "synchronize_bare_ground_flux_state",
]
