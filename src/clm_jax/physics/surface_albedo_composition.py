
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.io.surface_albedo import SurfaceAlbedoNativeData
from clm_jax.physics.ecosystem_dynamics_post_drainage import (
    EcosystemDynamicsPostDrainageResult,
)
from clm_jax.physics.hydrology_drainage import HydrologyDrainageState
from clm_jax.physics.lake_hydrology_driver import LakeHydrologyDriverResult
from clm_jax.physics.physical_driver import (
    PhysicalDriverResult,
    PhysicalDriverRuntimeResult,
)
from clm_jax.physics.post_lake_hydrology_driver import PostLakeHydrologyResult
from clm_jax.physics.solar_runtime import (
    native_grid_radians,
    raw_cosine_solar_zenith_native_runtime as raw_cosine_solar_zenith_runtime,
)
from clm_jax.physics.surface_albedo import (
    SurfaceAlbedoInputs,
    SurfaceAlbedoParameters,
    SurfaceAlbedoState,
)
from clm_jax.trajectory import NoleapModelClock, datm_next_shortwave_calendar_day

_NUMBER_OF_SNOW_LAYERS = 12


class SurfaceAlbedoCall(NamedTuple):

    inputs: SurfaceAlbedoInputs
    state: SurfaceAlbedoState
    parameters: SurfaceAlbedoParameters


class SurfaceAlbedoPostLakeCarry(NamedTuple):

    snow_fraction_column: jnp.ndarray
    snow_grain_radius_column_layer: jnp.ndarray
    aerosol_mass_concentration_column_layer_species: jnp.ndarray


def next_shortwave_cosine_gridcell(
    native: SurfaceAlbedoNativeData,
    clock: NoleapModelClock,
) -> jnp.ndarray:

    nextsw_cday = datm_next_shortwave_calendar_day(clock)
    return raw_cosine_solar_zenith_runtime(
        nextsw_cday,
        native_grid_radians(native.topology.grid.latitude_degrees),
        native_grid_radians(native.topology.grid.longitude_degrees),
    )


def snicar_aerosol_concentration_species_order(
    post_lake: PostLakeHydrologyResult,
) -> jnp.ndarray:

    concentration = post_lake.lake_aerosol_mass.state.concentration
    return jnp.stack(
        (
            concentration.hydrophilic_black_carbon_layer,
            concentration.hydrophobic_black_carbon_layer,
            concentration.hydrophilic_organic_carbon_layer,
            concentration.hydrophobic_organic_carbon_layer,
            concentration.dust_species_1_layer,
            concentration.dust_species_2_layer,
            concentration.dust_species_3_layer,
            concentration.dust_species_4_layer,
        ),
        axis=2,
    )


def surface_albedo_post_lake_carry(
    post_lake: PostLakeHydrologyResult,
) -> SurfaceAlbedoPostLakeCarry:

    return SurfaceAlbedoPostLakeCarry(
        snow_fraction_column=post_lake.snow_fraction_column,
        snow_grain_radius_column_layer=(
            post_lake.nonlake_snow_age.state.snow_grain_radius_column_level
        ),
        aerosol_mass_concentration_column_layer_species=(
            snicar_aerosol_concentration_species_order(post_lake)
        ),
    )


def compose_surface_albedo_inputs(
    native: SurfaceAlbedoNativeData,
    clock: NoleapModelClock,
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    lake_hydrology: LakeHydrologyDriverResult,
    post_lake: PostLakeHydrologyResult | SurfaceAlbedoPostLakeCarry,
    post_drainage: EcosystemDynamicsPostDrainageResult,
    *,
    drainage_state: HydrologyDrainageState | None = None,
) -> SurfaceAlbedoInputs:

    post_lake_carry = (
        post_lake
        if isinstance(post_lake, SurfaceAlbedoPostLakeCarry)
        else surface_albedo_post_lake_carry(post_lake)
    )
    lake = lake_hydrology.hydrology.state
    water = lake.water
    soil_water = (water.soil_water_volume_fraction_column_level
                  if drainage_state is None else drainage_state.volumetric_water_column_level)
    liquid = (water.soil_snow_liquid_column_level if drainage_state is None
              else drainage_state.soil_snow_liquid_water_column_level)
    ice = (water.soil_snow_ice_column_level if drainage_state is None
           else drainage_state.soil_snow_ice_water_column_level)
    canopy_water = physical.canopy.state.canopy_hydrology
    vegetation_structure = post_drainage.state.vegetation_structure
    aerosol = post_lake_carry.aerosol_mass_concentration_column_layer_species

    column_count = native.topology.column.gridcell.shape[0]
    patch_count = native.topology.patch.gridcell.shape[0]
    expected_column = (column_count,)
    expected_patch = (patch_count,)
    if water.snow_water_equivalent_column.shape != expected_column:
        raise ValueError("stage-510 hydrology column axis disagrees with native topology")
    if physical.canopy_flux.state.vegetation_temperature_patch.shape != expected_patch:
        raise ValueError("stage-510 canopy-flux patch axis disagrees with native topology")
    if aerosol.shape != (column_count, _NUMBER_OF_SNOW_LAYERS, 8):
        raise ValueError("stage-510 SNICAR concentration has an invalid native shape")

    return native.inputs._replace(
        cosine_solar_zenith_gridcell=next_shortwave_cosine_gridcell(native, clock),
        snow_layer_count_column=lake.geometry.snow_layer_count_column,
        ground_temperature_column=lake_hydrology.inputs.ground_temperature_column,
        top_soil_water_volume_fraction_column=(
            soil_water[:, 0]
        ),
        lake_ice_fraction_top_two_column=(
            lake.lake_ice_fraction_column_level[:, :2]
        ),
        snow_fraction_column=post_lake_carry.snow_fraction_column,
        snow_water_equivalent_column=water.snow_water_equivalent_column,
        snow_liquid_column_layer=(
            liquid[:, :_NUMBER_OF_SNOW_LAYERS]
        ),
        snow_ice_column_layer=(
            ice[:, :_NUMBER_OF_SNOW_LAYERS]
        ),
        snow_grain_radius_column_layer=(
            post_lake_carry.snow_grain_radius_column_layer
        ),
        aerosol_mass_concentration_column_layer_species=aerosol,
        vegetation_temperature_patch=(
            physical.canopy_flux.state.vegetation_temperature_patch
        ),
        wet_fraction_patch=canopy_water.canopy_wet_fractions.wet,
        canopy_snow_fraction_patch=(
            canopy_water.canopy_wet_fractions.snow_covered
        ),
        exposed_leaf_area_patch=(
            vegetation_structure.exposed_leaf_area_index_patch
        ),
        exposed_stem_area_patch=(
            vegetation_structure.exposed_stem_area_index_patch
        ),
    )


def compose_surface_albedo_call(
    native: SurfaceAlbedoNativeData,
    clock: NoleapModelClock,
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    lake_hydrology: LakeHydrologyDriverResult,
    post_lake: PostLakeHydrologyResult | SurfaceAlbedoPostLakeCarry,
    post_drainage: EcosystemDynamicsPostDrainageResult,
    state: SurfaceAlbedoState | None = None,
    *,
    drainage_state: HydrologyDrainageState | None = None,
) -> SurfaceAlbedoCall:

    return SurfaceAlbedoCall(
        inputs=compose_surface_albedo_inputs(
            native,
            clock,
            physical,
            lake_hydrology,
            post_lake,
            post_drainage,
            drainage_state=drainage_state,
        ),
        state=native.state if state is None else state,
        parameters=native.parameters,
    )


__all__ = [
    "SurfaceAlbedoCall",
    "SurfaceAlbedoPostLakeCarry",
    "compose_surface_albedo_call",
    "compose_surface_albedo_inputs",
    "next_shortwave_cosine_gridcell",
    "snicar_aerosol_concentration_species_order",
    "surface_albedo_post_lake_carry",
]
