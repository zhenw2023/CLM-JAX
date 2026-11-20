
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.forcing.datm import DatmForcing
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.canopy_driver import CanopyDriverResult
from clm_jax.physics.canopy_radiation import (
    CanopySunShadeInputs,
    CanopySunShadeState,
    canopy_sunshade_fractions,
)
from clm_jax.physics.surface_radiation import (
    SurfaceRadiationInputs,
    SurfaceRadiationState,
    surface_radiation,
)


class RadiationDriverInputs(NamedTuple):

    canopy_sunshade: CanopySunShadeInputs
    surface_radiation: SurfaceRadiationInputs


class RadiationDriverState(NamedTuple):

    canopy_sunshade: CanopySunShadeState
    surface_radiation: SurfaceRadiationState


class RadiationDriverResult(NamedTuple):

    state: RadiationDriverState
    canopy_sunshade: CanopySunShadeState
    surface_radiation: SurfaceRadiationState


def datm_shortwave_bands(
    grid_forcing: DatmForcing,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    direct = jnp.stack(
        (
            jnp.asarray(grid_forcing.shortwave_visible_direct, dtype=jnp.float64),
            jnp.asarray(grid_forcing.shortwave_nir_direct, dtype=jnp.float64),
        ),
        axis=1,
    )
    diffuse = jnp.stack(
        (
            jnp.asarray(grid_forcing.shortwave_visible_diffuse, dtype=jnp.float64),
            jnp.asarray(grid_forcing.shortwave_nir_diffuse, dtype=jnp.float64),
        ),
        axis=1,
    )
    return direct, diffuse


def radiation_driver_step(
    topology: SubgridTopology,
    grid_forcing: DatmForcing,
    canopy: CanopyDriverResult,
    inputs: RadiationDriverInputs,
    state: RadiationDriverState,
    *,
    current_time_of_day_seconds: int | jnp.ndarray,
    step_seconds: int | jnp.ndarray,
    subgrid_fluxes_enabled: bool | jnp.ndarray = True,
) -> RadiationDriverResult:

    direct, diffuse = datm_shortwave_bands(grid_forcing)
    canopy_sunshade = canopy_sunshade_fractions(
        topology,
        direct[:, 0],
        diffuse[:, 0],
        inputs.canopy_sunshade,
        state.canopy_sunshade,
    )

    current_snow = canopy.state.canopy_hydrology.snow
    surface_inputs = inputs.surface_radiation._replace(
        direct_shortwave_gridcell=direct,
        diffuse_shortwave_gridcell=diffuse,
        snow_layer_count_column=current_snow.layer_count,
        snow_depth_column=current_snow.depth,
    )
    surface_initial_state = state.surface_radiation._replace(
        fluxes=state.surface_radiation.fluxes._replace(
            sunlit_fraction_patch=canopy_sunshade.sunlit_fraction_patch,
        )
    )
    surface = surface_radiation(
        topology,
        surface_inputs,
        surface_initial_state,
        current_time_of_day_seconds=current_time_of_day_seconds,
        time_step_seconds=step_seconds,
        subgrid_fluxes_enabled=subgrid_fluxes_enabled,
    )

    final_canopy_state = canopy_sunshade._replace(
        sunlit_fraction_patch=surface.fluxes.sunlit_fraction_patch,
    )
    return RadiationDriverResult(
        state=RadiationDriverState(
            canopy_sunshade=final_canopy_state,
            surface_radiation=surface,
        ),
        canopy_sunshade=canopy_sunshade,
        surface_radiation=surface,
    )


__all__ = [
    "RadiationDriverInputs",
    "RadiationDriverResult",
    "RadiationDriverState",
    "datm_shortwave_bands",
    "radiation_driver_step",
]
