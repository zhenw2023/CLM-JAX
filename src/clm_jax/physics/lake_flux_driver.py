
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.forcing.datm import DatmForcing
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.atmospheric_driver import AtmosphericDriverResult
from clm_jax.physics.canopy_driver import CanopyDriverResult
from clm_jax.physics.canopy_temperature_driver import CanopyTemperatureDriverResult
from clm_jax.physics.lake_fluxes import (
    LakeFluxInputs,
    LakeFluxResult,
    LakeFluxState,
    lake_fluxes,
)
from clm_jax.physics.numerics import fortran_divide, fortran_multiply
from clm_jax.physics.radiation_sequence import RadiationSequenceResult


class LakeFluxNativeMap(NamedTuple):

    column_index: jnp.ndarray
    patch_index: jnp.ndarray
    gridcell_index: jnp.ndarray


class LakeFluxDriverInputs(NamedTuple):

    native_map: LakeFluxNativeMap
    ground_layer_thickness_lake_level: jnp.ndarray
    lake_layer_thickness_lake_level: jnp.ndarray
    lake_depth: jnp.ndarray
    lake_fetch: jnp.ndarray


class LakeFluxDriverParameters(NamedTuple):
    maximum_stable_zeta: float | jnp.ndarray


class LakeFluxDriverState(NamedTuple):

    surface: LakeFluxState
    lake_temperature_lake_level: jnp.ndarray
    saved_top_eddy_conductivity_lake: jnp.ndarray


class LakeFluxDriverCall(NamedTuple):

    inputs: LakeFluxInputs
    parameters: LakeFluxDriverParameters
    previous_state: LakeFluxDriverState


class LakeFluxDriverResult(NamedTuple):

    inputs: LakeFluxInputs
    fluxes: LakeFluxResult
    state: LakeFluxDriverState


def _lake_latitude_radians_from_degrees(latitude_degrees: jnp.ndarray) -> jnp.ndarray:

    return fortran_divide(
        fortran_multiply(
            jnp.asarray(latitude_degrees, dtype=jnp.float64),
            jnp.float64(jnp.pi),
        ),
        jnp.float64(180.0),
    )


def assemble_lake_flux_driver_call(
    topology: SubgridTopology,
    atmospheric: AtmosphericDriverResult,
    grid_forcing: DatmForcing,
    canopy: CanopyDriverResult,
    radiation: RadiationSequenceResult,
    temperature: CanopyTemperatureDriverResult,
    inputs: LakeFluxDriverInputs,
    parameters: LakeFluxDriverParameters,
    state: LakeFluxDriverState,
) -> LakeFluxDriverCall:

    native = inputs.native_map
    column = jnp.asarray(native.column_index, dtype=jnp.int32)
    patch = jnp.asarray(native.patch_index, dtype=jnp.int32)
    gridcell = jnp.asarray(native.gridcell_index, dtype=jnp.int32)
    lake_count = column.shape[0]

    forcing = atmospheric.column_forcing
    hydrology = canopy.state.canopy_hydrology
    temperature_inputs = temperature.inputs
    temperature_column = temperature.state.column
    temperature_patch = temperature.state.patch
    surface_radiation = radiation.state.canopy_surface.surface_radiation
    radiation_fluxes = surface_radiation.fluxes
    radiation_diagnostics = surface_radiation.diagnostics

    soil_snow_layer_thickness = jnp.concatenate(
        (
            hydrology.snow.layer_thickness[column],
            inputs.ground_layer_thickness_lake_level,
        ),
        axis=1,
    )
    compact_inputs = LakeFluxInputs(
        active_lake=jnp.ones((lake_count,), dtype=jnp.bool_),
        latitude_radians=_lake_latitude_radians_from_degrees(
            topology.grid.latitude_degrees[gridcell]
        ),
        eastward_wind=grid_forcing.wind_u[gridcell],
        northward_wind=grid_forcing.wind_v[gridcell],
        atmospheric_temperature=forcing.temperature[column],
        atmospheric_potential_temperature=forcing.potential_temperature[column],
        atmospheric_pressure=forcing.pressure[column],
        atmospheric_density=forcing.density[column],
        atmospheric_specific_humidity=forcing.specific_humidity[column],
        downward_longwave=forcing.longwave_down[column],
        rain_flux=forcing.rain[column],
        snow_flux=forcing.snow[column],
        incident_nir_direct=radiation_diagnostics.incident_direct_patch[patch, 1],
        incident_nir_diffuse=radiation_diagnostics.incident_diffuse_patch[patch, 1],
        reflected_nir_direct=radiation_diagnostics.reflected_direct_patch[patch, 1],
        reflected_nir_diffuse=radiation_diagnostics.reflected_diffuse_patch[patch, 1],
        ground_absorbed_shortwave=radiation_fluxes.absorbed_ground_patch[patch],
        layer_absorbed_shortwave=(radiation_fluxes.absorbed_ground_layers_patch[patch]),
        snow_layer_count=hydrology.snow.layer_count[column],
        soil_snow_layer_thickness=soil_snow_layer_thickness,
        lake_layer_thickness=inputs.lake_layer_thickness_lake_level,
        lake_depth=inputs.lake_depth,
        lake_fetch=inputs.lake_fetch,
        soil_snow_temperature=(temperature_inputs.soil_snow_temperature_column_level[column]),
        soil_snow_liquid_water=(temperature_inputs.soil_liquid_water_column_level[column]),
        soil_snow_ice_water=(temperature_inputs.soil_ice_water_column_level[column]),
        lake_temperature=state.lake_temperature_lake_level,
        saved_top_eddy_conductivity=state.saved_top_eddy_conductivity_lake,
        forcing_height_wind=temperature_patch.forcing_height_wind_patch[patch],
        forcing_height_temperature=(temperature_patch.forcing_height_temperature_patch[patch]),
        forcing_height_humidity=(temperature_patch.forcing_height_humidity_patch[patch]),
    )
    synchronized_surface = state.surface._replace(
        ground_temperature=temperature_column.ground_temperature_column[column],
        momentum_roughness=(temperature_column.ground_momentum_roughness_column[column]),
        sensible_roughness=(temperature_column.ground_sensible_roughness_column[column]),
        latent_roughness=temperature_column.ground_latent_roughness_column[column],
    )
    return LakeFluxDriverCall(
        inputs=compact_inputs,
        parameters=parameters,
        previous_state=state._replace(surface=synchronized_surface),
    )


def solve_lake_flux_driver_call(call: LakeFluxDriverCall) -> LakeFluxResult:

    return lake_fluxes(
        call.inputs,
        call.previous_state.surface,
        call.parameters.maximum_stable_zeta,
    )


def complete_lake_flux_driver_call(
    call: LakeFluxDriverCall,
    fluxes: LakeFluxResult,
) -> LakeFluxDriverResult:

    return LakeFluxDriverResult(
        inputs=call.inputs,
        fluxes=fluxes,
        state=call.previous_state._replace(surface=fluxes.state),
    )


def lake_flux_driver_step(
    topology: SubgridTopology,
    atmospheric: AtmosphericDriverResult,
    grid_forcing: DatmForcing,
    canopy: CanopyDriverResult,
    radiation: RadiationSequenceResult,
    temperature: CanopyTemperatureDriverResult,
    inputs: LakeFluxDriverInputs,
    parameters: LakeFluxDriverParameters,
    state: LakeFluxDriverState,
) -> LakeFluxDriverResult:

    call = assemble_lake_flux_driver_call(
        topology,
        atmospheric,
        grid_forcing,
        canopy,
        radiation,
        temperature,
        inputs,
        parameters,
        state,
    )
    return complete_lake_flux_driver_call(call, solve_lake_flux_driver_call(call))


__all__ = [
    "LakeFluxDriverCall",
    "LakeFluxDriverInputs",
    "LakeFluxDriverParameters",
    "LakeFluxDriverResult",
    "LakeFluxDriverState",
    "LakeFluxNativeMap",
    "assemble_lake_flux_driver_call",
    "complete_lake_flux_driver_call",
    "lake_flux_driver_step",
    "solve_lake_flux_driver_call",
]
