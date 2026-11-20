
from __future__ import annotations

from typing import NamedTuple

from clm_jax.io.restart import SubgridTopology
from clm_jax.io.urban_albedo_restart import UrbanAlbedoRestartData
from clm_jax.physics.surface_albedo import SurfaceAlbedoState
from clm_jax.physics.surface_albedo_composition import SurfaceAlbedoCall
from clm_jax.physics.urban_albedo import (
    UrbanAlbedoInputs,
    UrbanAlbedoState,
    UrbanCanopyAlbedoState,
    UrbanSurfaceAlbedoState,
)


class UrbanAlbedoCall(NamedTuple):

    topology: SubgridTopology
    inputs: UrbanAlbedoInputs
    state: UrbanAlbedoState


def urban_surface_state_from_surface_albedo(
    state: SurfaceAlbedoState,
) -> UrbanSurfaceAlbedoState:

    canopy = state.canopy.two_stream
    return UrbanSurfaceAlbedoState(
        ground_direct_column=state.ground.ground_direct_albedo_column_band,
        ground_diffuse_column=state.ground.ground_diffuse_albedo_column_band,
        surface_direct_patch=canopy.direct_surface_albedo_patch_band,
        surface_diffuse_patch=canopy.diffuse_surface_albedo_patch_band,
        canopy=UrbanCanopyAlbedoState(
            absorbed_direct_patch=canopy.direct_absorption_patch_band,
            absorbed_sunlit_direct_patch=(canopy.direct_sunlit_absorption_patch_band),
            absorbed_shaded_direct_patch=(canopy.direct_shaded_absorption_patch_band),
            absorbed_diffuse_patch=canopy.diffuse_absorption_patch_band,
            absorbed_sunlit_diffuse_patch=(canopy.diffuse_sunlit_absorption_patch_band),
            absorbed_shaded_diffuse_patch=(canopy.diffuse_shaded_absorption_patch_band),
            transmitted_direct_patch=canopy.direct_beam_transmission_patch_band,
            transmitted_diffuse_from_direct_patch=(
                canopy.direct_to_diffuse_transmission_patch_band
            ),
            transmitted_diffuse_patch=canopy.diffuse_transmission_patch_band,
        ),
    )


def compose_urban_albedo_call(
    native: UrbanAlbedoRestartData,
    surface_call: SurfaceAlbedoCall,
    surface_state: SurfaceAlbedoState,
    state: UrbanAlbedoState | None = None,
) -> UrbanAlbedoCall:

    column_count = native.topology.column.gridcell.shape[0]
    patch_count = native.topology.patch.gridcell.shape[0]
    if surface_state.cosine_solar_zenith_column.shape != (column_count,):
        raise ValueError("stage-520 SurfaceAlbedo column axis disagrees with topology")
    if surface_call.inputs.snow_fraction_column.shape != (column_count,):
        raise ValueError("stage-520 snow-fraction column axis disagrees with topology")
    if surface_state.canopy.two_stream.direct_surface_albedo_patch_band.shape != (
        patch_count,
        2,
    ):
        raise ValueError("stage-520 SurfaceAlbedo patch axis disagrees with topology")
    previous = native.state if state is None else state
    return UrbanAlbedoCall(
        topology=native.topology,
        inputs=native.inputs._replace(
            cosine_solar_zenith_column=(surface_state.cosine_solar_zenith_column),
            snow_fraction_column=surface_call.inputs.snow_fraction_column,
            snow_water_column=surface_call.inputs.snow_water_equivalent_column,
        ),
        state=previous._replace(surface=urban_surface_state_from_surface_albedo(surface_state)),
    )


__all__ = [
    "UrbanAlbedoCall",
    "compose_urban_albedo_call",
    "urban_surface_state_from_surface_albedo",
]
