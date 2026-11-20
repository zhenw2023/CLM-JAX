
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.forcing.datm import DatmForcing
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.atmospheric_driver import AtmosphericDriverResult
from clm_jax.physics.canopy_step import (
    CanopyHydrologyForcing,
    CanopyHydrologyState,
    CanopyHydrologyStepParameters,
    CanopyHydrologyStepResult,
    CanopyStructure,
    canopy_hydrology_step,
)
from clm_jax.physics.irrigation import (
    IrrigationApplicationParameters,
    IrrigationApplicationResult,
    IrrigationState,
    apply_irrigation,
)


class CanopyDriverParameters(NamedTuple):
    irrigation: IrrigationApplicationParameters
    canopy_hydrology: CanopyHydrologyStepParameters


class CanopyDriverInputs(NamedTuple):
    structure: CanopyStructure
    previous_snow_drainage_flux_column: jnp.ndarray
    flood_gridcell: jnp.ndarray


class CanopyDriverState(NamedTuple):
    irrigation: IrrigationState
    canopy_hydrology: CanopyHydrologyState


class CanopyDriverResult(NamedTuple):
    state: CanopyDriverState
    irrigation: IrrigationApplicationResult
    canopy_hydrology: CanopyHydrologyStepResult


def canopy_driver_step(
    topology: SubgridTopology,
    atmospheric: AtmosphericDriverResult,
    grid_forcing: DatmForcing,
    inputs: CanopyDriverInputs,
    state: CanopyDriverState,
    *,
    step_seconds: int | float,
    parameters: CanopyDriverParameters,
) -> CanopyDriverResult:

    column_topography = atmospheric.state.column_topography
    current_topology = topology._replace(
        column=topology.column._replace(
            elevation=column_topography.elevation,
            column_type=column_topography.column_type,
        )
    )
    irrigation = apply_irrigation(
        current_topology,
        state.irrigation,
        parameters=parameters.irrigation,
    )
    structure = inputs.structure._replace(
        vegetation_exposed_patch=(
            atmospheric.driver_initialization.vegetation_exposed
        )
    )
    snow = state.canopy_hydrology.snow._replace(
        old_ice_fraction=(
            atmospheric.driver_initialization.previous_snow_ice_fraction[
                :, : state.canopy_hydrology.snow.old_ice_fraction.shape[1]
            ]
        )
    )
    canopy_state = state.canopy_hydrology._replace(snow=snow)
    canopy = canopy_hydrology_step(
        current_topology,
        CanopyHydrologyForcing(
            rain_column=atmospheric.column_forcing.rain,
            snow_column=atmospheric.column_forcing.snow,
            temperature_column=atmospheric.column_forcing.temperature,
            wind_gridcell=grid_forcing.wind_speed,
            flood_gridcell=inputs.flood_gridcell,
            irrigation_patch=irrigation.fluxes.applied_patch,
        ),
        structure,
        canopy_state,
        inputs.previous_snow_drainage_flux_column,
        step_seconds=step_seconds,
        parameters=parameters.canopy_hydrology,
    )
    return CanopyDriverResult(
        state=CanopyDriverState(
            irrigation=irrigation.state,
            canopy_hydrology=canopy.state,
        ),
        irrigation=irrigation,
        canopy_hydrology=canopy,
    )


__all__ = [
    "CanopyDriverInputs",
    "CanopyDriverParameters",
    "CanopyDriverResult",
    "CanopyDriverState",
    "canopy_driver_step",
]
