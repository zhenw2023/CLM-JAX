
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.constants import DENSITY_FRESH_WATER, DENSITY_ICE
from clm_jax.io.restart import LANDUNIT_LAKE, SubgridTopology
from clm_jax.physics.atmospheric_driver import AtmosphericDriverResult
from clm_jax.physics.bare_ground_driver import BareGroundDriverResult
from clm_jax.physics.canopy_driver import CanopyDriverResult
from clm_jax.physics.canopy_flux_driver import (
    CanopyFluxDriverResult,
    CanopyFluxRuntimeResult,
)
from clm_jax.physics.canopy_temperature import NUMBER_OF_SNOW_LEVELS
from clm_jax.physics.canopy_temperature_driver import CanopyTemperatureDriverResult
from clm_jax.physics.dust import (
    DEFAULT_DUST_EXECUTION_PARAMETERS,
    DustDepositionInputs,
    DustEmissionInputs,
    DustExecutionParameters,
    DustParameters,
    DustState,
    DustStepResult,
    dust_emission_and_deposition,
)
from clm_jax.physics.lake_flux_driver import (
    LakeFluxDriverInputs,
    LakeFluxDriverResult,
)
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
)
from clm_jax.physics.urban_flux_driver import (
    UrbanFluxDriverInputs,
    UrbanFluxDriverResult,
)


class DustDriverInputs(NamedTuple):

    gravimetric_water_threshold_column: jnp.ndarray
    clay_mass_fraction_limited_column: jnp.ndarray
    saturated_water_content_top_column: jnp.ndarray
    top_soil_layer_thickness_column: jnp.ndarray
    total_leaf_area_index_patch: jnp.ndarray
    total_stem_area_index_patch: jnp.ndarray


class DustSurfaceBoundary(NamedTuple):

    aerodynamic_resistance_patch: jnp.ndarray
    dust_friction_velocity_patch: jnp.ndarray
    dust_wind_10m_patch: jnp.ndarray


class DustDriverResult(NamedTuple):

    surface: DustSurfaceBoundary
    emission_inputs: DustEmissionInputs
    deposition_inputs: DustDepositionInputs
    step: DustStepResult
    state: DustState


def _validate_patch_vector(
    name: str,
    value: jnp.ndarray,
    patch_count: int,
) -> jnp.ndarray:
    result = jnp.asarray(value, dtype=jnp.float64)
    if result.shape != (patch_count,):
        raise ValueError(f"{name} must contain every native patch")
    return result


def _scatter_compact_surface(
    base: jnp.ndarray,
    patch_index: jnp.ndarray,
    compact_value: jnp.ndarray,
) -> jnp.ndarray:

    patch = jnp.asarray(patch_index, dtype=jnp.int32)
    value = jnp.asarray(compact_value, dtype=jnp.float64)
    if patch.shape != value.shape:
        value = jnp.broadcast_to(value, patch.shape)
    return base.at[jnp.reshape(patch, (-1,))].set(jnp.reshape(value, (-1,)))


def assemble_dust_surface_boundary(
    topology: SubgridTopology,
    bare_ground: BareGroundDriverResult,
    canopy_flux: CanopyFluxDriverResult | CanopyFluxRuntimeResult,
    urban_inputs: UrbanFluxDriverInputs,
    urban: UrbanFluxDriverResult,
    lake_inputs: LakeFluxDriverInputs,
    lake: LakeFluxDriverResult,
) -> DustSurfaceBoundary:

    patch_count = topology.patch.column.shape[0]
    canopy_friction = canopy_flux.state.friction_velocity
    friction = _validate_patch_vector(
        "canopy dust friction velocity",
        canopy_friction.dust_friction_velocity_patch,
        patch_count,
    )
    wind_10m = _validate_patch_vector(
        "canopy dust 10-m wind",
        canopy_friction.wind_10m_dust_patch,
        patch_count,
    )
    runtime = isinstance(canopy_flux, CanopyFluxRuntimeResult)
    exposed = jnp.asarray(
        canopy_flux.dust.active_patch if runtime else canopy_flux.fast_inputs.energy.active_patch,
        dtype=jnp.bool_,
    )
    if exposed.shape != (patch_count,):
        raise ValueError("canopy active mask must contain every native patch")
    canopy_resistance = _validate_patch_vector(
        "canopy aerodynamic resistance",
        canopy_flux.dust.momentum_resistance_patch
        if runtime
        else canopy_flux.active.energy_hydraulics.energy.iteration.momentum_resistance_patch,
        patch_count,
    )
    bare_resistance = _validate_patch_vector(
        "bare-ground aerodynamic resistance",
        bare_ground.state.aerodynamic_resistance_patch,
        patch_count,
    )
    resistance = jnp.where(exposed, canopy_resistance, bare_resistance)

    urban_patch = jnp.asarray(
        urban_inputs.native_map.surface_patch_index,
        dtype=jnp.int32,
    )
    urban_friction = urban.turbulence.state.friction_velocity
    urban_shape = urban_patch.shape
    if urban_patch.ndim != 2:
        raise ValueError("urban surface map must be a landunit-by-surface matrix")
    urban_ram = jnp.asarray(
        urban.turbulence.diagnostics.momentum_resistance_landunit,
        dtype=jnp.float64,
    )[:, None]
    urban_fv = jnp.asarray(
        urban_friction.dust_friction_velocity_patch,
        dtype=jnp.float64,
    )[:, None]
    urban_u10 = jnp.asarray(
        urban_friction.wind_10m_dust_patch,
        dtype=jnp.float64,
    )[:, None]
    if any(value.shape[0] != urban_shape[0] for value in (urban_ram, urban_fv, urban_u10)):
        raise ValueError("urban friction result and native surface map disagree")
    resistance = _scatter_compact_surface(resistance, urban_patch, urban_ram)
    friction = _scatter_compact_surface(friction, urban_patch, urban_fv)
    wind_10m = _scatter_compact_surface(wind_10m, urban_patch, urban_u10)

    lake_patch = jnp.asarray(
        lake_inputs.native_map.patch_index,
        dtype=jnp.int32,
    )
    lake_friction = lake.fluxes.state.friction_velocity
    lake_ram = jnp.asarray(
        lake.fluxes.surface.aerodynamic_resistance,
        dtype=jnp.float64,
    )
    lake_fv = jnp.asarray(
        lake_friction.dust_friction_velocity_patch,
        dtype=jnp.float64,
    )
    lake_u10 = jnp.asarray(
        lake_friction.wind_10m_dust_patch,
        dtype=jnp.float64,
    )
    if not (lake_patch.shape == lake_ram.shape == lake_fv.shape == lake_u10.shape):
        raise ValueError("lake friction result and native patch map disagree")
    resistance = _scatter_compact_surface(resistance, lake_patch, lake_ram)
    friction = _scatter_compact_surface(friction, lake_patch, lake_fv)
    wind_10m = _scatter_compact_surface(wind_10m, lake_patch, lake_u10)

    return DustSurfaceBoundary(
        aerodynamic_resistance_patch=resistance,
        dust_friction_velocity_patch=friction,
        dust_wind_10m_patch=wind_10m,
    )


def _current_top_soil_volumetric_water(
    temperature: CanopyTemperatureDriverResult,
    top_soil_layer_thickness_column: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    layer = NUMBER_OF_SNOW_LEVELS
    liquid = jnp.asarray(
        temperature.inputs.soil_liquid_water_column_level[:, layer],
        dtype=jnp.float64,
    )
    ice = jnp.asarray(
        temperature.inputs.soil_ice_water_column_level[:, layer],
        dtype=jnp.float64,
    )
    thickness = jnp.asarray(top_soil_layer_thickness_column, dtype=jnp.float64)
    if not (liquid.shape == ice.shape == thickness.shape):
        raise ValueError("top-soil water and layer thickness dimensions disagree")
    liquid_volume = fortran_divide(
        liquid,
        fortran_multiply(thickness, jnp.float64(DENSITY_FRESH_WATER)),
    )
    ice_volume = fortran_divide(
        ice,
        fortran_multiply(thickness, jnp.float64(DENSITY_ICE)),
    )
    return liquid, ice, fortran_add(liquid_volume, ice_volume)


def assemble_dust_driver_inputs(
    topology: SubgridTopology,
    atmospheric: AtmosphericDriverResult,
    canopy: CanopyDriverResult,
    temperature: CanopyTemperatureDriverResult,
    inputs: DustDriverInputs,
    surface: DustSurfaceBoundary,
) -> tuple[DustEmissionInputs, DustDepositionInputs]:

    patch_count = topology.patch.column.shape[0]
    column_count = topology.column.gridcell.shape[0]
    liquid, ice, volumetric_water = _current_top_soil_volumetric_water(
        temperature,
        inputs.top_soil_layer_thickness_column,
    )
    column_fields = (
        inputs.gravimetric_water_threshold_column,
        inputs.clay_mass_fraction_limited_column,
        inputs.saturated_water_content_top_column,
        volumetric_water,
    )
    if any(jnp.shape(value) != (column_count,) for value in column_fields):
        raise ValueError("dust static soil fields must contain every native column")
    patch_fields = (
        inputs.total_leaf_area_index_patch,
        inputs.total_stem_area_index_patch,
        surface.aerodynamic_resistance_patch,
        surface.dust_friction_velocity_patch,
        surface.dust_wind_10m_patch,
    )
    if any(jnp.shape(value) != (patch_count,) for value in patch_fields):
        raise ValueError("dust vegetation and surface fields must contain every patch")

    forcing = atmospheric.column_forcing
    nonlake = jnp.asarray(topology.patch.active, dtype=jnp.bool_) & (
        jnp.asarray(topology.patch.landunit_type, dtype=jnp.int32) != jnp.int32(LANDUNIT_LAKE)
    )
    emission = DustEmissionInputs(
        nonlake_patch=nonlake,
        air_density_column=jnp.asarray(forcing.density, dtype=jnp.float64),
        gravimetric_water_threshold_column=jnp.asarray(
            inputs.gravimetric_water_threshold_column,
            dtype=jnp.float64,
        ),
        clay_mass_fraction_limited_column=jnp.asarray(
            inputs.clay_mass_fraction_limited_column,
            dtype=jnp.float64,
        ),
        saturated_water_content_top_column=jnp.asarray(
            inputs.saturated_water_content_top_column,
            dtype=jnp.float64,
        ),
        snow_cover_fraction_column=jnp.asarray(
            canopy.state.canopy_hydrology.snow.snow_fraction,
            dtype=jnp.float64,
        ),
        volumetric_soil_water_top_column=volumetric_water,
        soil_liquid_water_top_column=liquid,
        soil_ice_top_column=ice,
        leaf_area_index_patch=jnp.asarray(
            inputs.total_leaf_area_index_patch,
            dtype=jnp.float64,
        ),
        stem_area_index_patch=jnp.asarray(
            inputs.total_stem_area_index_patch,
            dtype=jnp.float64,
        ),
        dust_friction_velocity_patch=surface.dust_friction_velocity_patch,
        dust_wind_10m_patch=surface.dust_wind_10m_patch,
    )
    deposition = DustDepositionInputs(
        air_pressure_column=jnp.asarray(forcing.pressure, dtype=jnp.float64),
        air_density_column=jnp.asarray(forcing.density, dtype=jnp.float64),
        air_temperature_column=jnp.asarray(forcing.temperature, dtype=jnp.float64),
        aerodynamic_resistance_patch=surface.aerodynamic_resistance_patch,
        dust_friction_velocity_patch=surface.dust_friction_velocity_patch,
    )
    return emission, deposition


def dust_driver_step(
    topology: SubgridTopology,
    atmospheric: AtmosphericDriverResult,
    canopy: CanopyDriverResult,
    temperature: CanopyTemperatureDriverResult,
    bare_ground: BareGroundDriverResult,
    canopy_flux: CanopyFluxDriverResult | CanopyFluxRuntimeResult,
    urban_inputs: UrbanFluxDriverInputs,
    urban: UrbanFluxDriverResult,
    lake_inputs: LakeFluxDriverInputs,
    lake: LakeFluxDriverResult,
    inputs: DustDriverInputs,
    state: DustState,
    parameters: DustParameters,
    *,
    execution: DustExecutionParameters = DEFAULT_DUST_EXECUTION_PARAMETERS,
) -> DustDriverResult:

    surface = assemble_dust_surface_boundary(
        topology,
        bare_ground,
        canopy_flux,
        urban_inputs,
        urban,
        lake_inputs,
        lake,
    )
    emission, deposition = assemble_dust_driver_inputs(
        topology,
        atmospheric,
        canopy,
        temperature,
        inputs,
        surface,
    )
    step = dust_emission_and_deposition(
        topology,
        state,
        emission,
        deposition,
        parameters,
        execution=execution,
    )
    return DustDriverResult(
        surface=surface,
        emission_inputs=emission,
        deposition_inputs=deposition,
        step=step,
        state=step.state,
    )


__all__ = [
    "DustDriverInputs",
    "DustDriverResult",
    "DustSurfaceBoundary",
    "assemble_dust_driver_inputs",
    "assemble_dust_surface_boundary",
    "dust_driver_step",
]
