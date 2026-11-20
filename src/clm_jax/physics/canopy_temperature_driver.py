
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.forcing.datm import DatmForcing
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.atmospheric_driver import AtmosphericDriverResult
from clm_jax.physics.canopy_driver import CanopyDriverResult
from clm_jax.physics.canopy_step import CanopyHydrologyState
from clm_jax.physics.canopy_temperature import (
    NUMBER_OF_SNOW_LEVELS,
    NUMBER_OF_TOTAL_LEVELS,
    CanopyTemperatureInputs,
    CanopyTemperatureParameters,
    CanopyTemperatureState,
    canopy_temperature,
)


class CanopyTemperatureDriverResult(NamedTuple):

    inputs: CanopyTemperatureInputs
    state: CanopyTemperatureState


def synchronize_canopy_temperature_water_state(
    inputs: CanopyTemperatureInputs,
    hydrology: CanopyHydrologyState,
) -> CanopyTemperatureInputs:

    number_of_columns = inputs.snow_layer_count_column.shape[0]
    total_shape = (number_of_columns, NUMBER_OF_TOTAL_LEVELS)
    snow_shape = (number_of_columns, NUMBER_OF_SNOW_LEVELS)
    if inputs.soil_snow_temperature_column_level.shape != total_shape:
        raise ValueError("CanopyTemperature temperature input has the wrong level shape")
    if inputs.soil_liquid_water_column_level.shape != total_shape:
        raise ValueError("CanopyTemperature liquid-water input has the wrong level shape")
    if inputs.soil_ice_water_column_level.shape != total_shape:
        raise ValueError("CanopyTemperature ice-water input has the wrong level shape")
    for name in ("layer_temperature", "layer_liquid", "layer_ice"):
        if getattr(hydrology.snow, name).shape != snow_shape:
            raise ValueError(f"CanopyHydrology snow {name} has the wrong level shape")
    if hydrology.top_soil_liquid_column.shape != (number_of_columns,):
        raise ValueError("CanopyHydrology top-soil liquid has the wrong column shape")

    temperature = (
        jnp.asarray(inputs.soil_snow_temperature_column_level, dtype=jnp.float64)
        .at[:, :NUMBER_OF_SNOW_LEVELS]
        .set(hydrology.snow.layer_temperature)
    )
    liquid = (
        jnp.asarray(inputs.soil_liquid_water_column_level, dtype=jnp.float64)
        .at[:, :NUMBER_OF_SNOW_LEVELS]
        .set(hydrology.snow.layer_liquid)
    )
    liquid = liquid.at[:, NUMBER_OF_SNOW_LEVELS].set(
        hydrology.top_soil_liquid_column
    )
    ice = (
        jnp.asarray(inputs.soil_ice_water_column_level, dtype=jnp.float64)
        .at[:, :NUMBER_OF_SNOW_LEVELS]
        .set(hydrology.snow.layer_ice)
    )
    return inputs._replace(
        snow_layer_count_column=jnp.asarray(hydrology.snow.layer_count, dtype=jnp.int32),
        soil_snow_temperature_column_level=temperature,
        surface_water_fraction_column=jnp.asarray(
            hydrology.surface_water_fraction_column,
            dtype=jnp.float64,
        ),
        effective_snow_fraction_column=jnp.asarray(
            hydrology.snow.effective_snow_fraction,
            dtype=jnp.float64,
        ),
        snow_fraction_column=jnp.asarray(
            hydrology.snow.snow_fraction,
            dtype=jnp.float64,
        ),
        soil_liquid_water_column_level=liquid,
        soil_ice_water_column_level=ice,
    )


def assemble_canopy_temperature_inputs(
    atmospheric: AtmosphericDriverResult,
    grid_forcing: DatmForcing,
    canopy: CanopyDriverResult,
    inputs: CanopyTemperatureInputs,
) -> CanopyTemperatureInputs:

    column_forcing = atmospheric.column_forcing
    synchronized = synchronize_canopy_temperature_water_state(
        inputs,
        canopy.state.canopy_hydrology,
    )
    return synchronized._replace(
        forcing_height_wind_gridcell=grid_forcing.reference_height,
        forcing_height_temperature_gridcell=grid_forcing.reference_height,
        forcing_height_humidity_gridcell=grid_forcing.reference_height,
        pressure_column=column_forcing.pressure,
        specific_humidity_column=column_forcing.specific_humidity,
        temperature_column=column_forcing.temperature,
        potential_temperature_column=column_forcing.potential_temperature,
        vegetation_exposed_patch=(
            atmospheric.driver_initialization.vegetation_exposed
        ),
    )


def canopy_temperature_driver_step(
    topology: SubgridTopology,
    atmospheric: AtmosphericDriverResult,
    grid_forcing: DatmForcing,
    canopy: CanopyDriverResult,
    inputs: CanopyTemperatureInputs,
    parameters: CanopyTemperatureParameters,
    state: CanopyTemperatureState,
) -> CanopyTemperatureDriverResult:

    dynamic_inputs = assemble_canopy_temperature_inputs(
        atmospheric,
        grid_forcing,
        canopy,
        inputs,
    )
    column_topography = atmospheric.state.column_topography
    current_topology = topology._replace(
        column=topology.column._replace(
            elevation=column_topography.elevation,
            column_type=column_topography.column_type,
        )
    )
    temperature = canopy_temperature(
        current_topology,
        dynamic_inputs,
        parameters,
        state,
    )
    return CanopyTemperatureDriverResult(
        inputs=dynamic_inputs,
        state=temperature,
    )


__all__ = [
    "CanopyTemperatureDriverResult",
    "assemble_canopy_temperature_inputs",
    "canopy_temperature_driver_step",
    "synchronize_canopy_temperature_water_state",
]
