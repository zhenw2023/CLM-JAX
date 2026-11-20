
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.forcing.datm import DatmForcing
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.canopy_driver import CanopyDriverResult
from clm_jax.physics.radiation_driver import (
    RadiationDriverInputs,
    RadiationDriverResult,
    RadiationDriverState,
    datm_shortwave_bands,
    radiation_driver_step,
)
from clm_jax.physics.urban_radiation import (
    UrbanRadiationInputs,
    UrbanRadiationResult,
    UrbanRadiationState,
    urban_radiation,
)


class RadiationSequenceInputs(NamedTuple):
    canopy_surface: RadiationDriverInputs
    urban: UrbanRadiationInputs


class RadiationSequenceState(NamedTuple):
    canopy_surface: RadiationDriverState
    urban: UrbanRadiationState


class RadiationSequenceResult(NamedTuple):
    state: RadiationSequenceState
    canopy_surface: RadiationDriverResult
    urban: UrbanRadiationResult


def radiation_sequence_step(
    topology: SubgridTopology,
    grid_forcing: DatmForcing,
    canopy: CanopyDriverResult,
    inputs: RadiationSequenceInputs,
    state: RadiationSequenceState,
    *,
    current_time_of_day_seconds: int | jnp.ndarray,
    step_seconds: int | jnp.ndarray,
    subgrid_fluxes_enabled: bool | jnp.ndarray = True,
) -> RadiationSequenceResult:

    canopy_surface = radiation_driver_step(
        topology,
        grid_forcing,
        canopy,
        inputs.canopy_surface,
        state.canopy_surface,
        current_time_of_day_seconds=current_time_of_day_seconds,
        step_seconds=step_seconds,
        subgrid_fluxes_enabled=subgrid_fluxes_enabled,
    )
    direct, diffuse = datm_shortwave_bands(grid_forcing)
    urban_inputs = inputs.urban._replace(
        direct_shortwave_gridcell=direct,
        diffuse_shortwave_gridcell=diffuse,
        downward_longwave_gridcell=grid_forcing.longwave_down,
        snow_fraction_column=canopy.state.canopy_hydrology.snow.snow_fraction,
    )

    surface_fluxes = canopy_surface.surface_radiation.fluxes
    urban_initial = state.urban._replace(
        fluxes=state.urban.fluxes._replace(
            absorbed_ground_patch=surface_fluxes.absorbed_ground_patch,
            absorbed_vegetation_patch=surface_fluxes.absorbed_vegetation_patch,
            absorbed_total_patch=surface_fluxes.absorbed_total_patch,
        )
    )
    urban = urban_radiation(topology, urban_inputs, urban_initial)

    synchronized_surface = canopy_surface.surface_radiation._replace(
        fluxes=surface_fluxes._replace(
            absorbed_ground_patch=urban.state.fluxes.absorbed_ground_patch,
            absorbed_vegetation_patch=urban.state.fluxes.absorbed_vegetation_patch,
            absorbed_total_patch=urban.state.fluxes.absorbed_total_patch,
        )
    )
    synchronized_canopy_surface = canopy_surface.state._replace(
        surface_radiation=synchronized_surface,
    )
    return RadiationSequenceResult(
        state=RadiationSequenceState(
            canopy_surface=synchronized_canopy_surface,
            urban=urban.state,
        ),
        canopy_surface=canopy_surface,
        urban=urban,
    )


__all__ = [
    "RadiationSequenceInputs",
    "RadiationSequenceResult",
    "RadiationSequenceState",
    "radiation_sequence_step",
]
