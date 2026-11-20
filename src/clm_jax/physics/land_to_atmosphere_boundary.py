
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp


class LandToAtmospherePhysicalFields(NamedTuple):

    reference_temperature_2m_patch: jnp.ndarray
    reference_specific_humidity_2m_patch: jnp.ndarray
    reference_wind_speed_10m_patch: jnp.ndarray
    eastward_stress_patch: jnp.ndarray
    northward_stress_patch: jnp.ndarray
    friction_velocity_patch: jnp.ndarray
    aerodynamic_resistance_patch: jnp.ndarray
    voc_flux_patch: jnp.ndarray
    dust_flux_patch: jnp.ndarray
    vegetation_temperature_patch: jnp.ndarray | None = None


def _patch_vector(name: str, value: Any, patch_count: int) -> jnp.ndarray:
    result = jnp.asarray(value, dtype=jnp.float64)
    if result.shape != (patch_count,):
        raise ValueError(
            f"{name} must contain every native patch; expected {(patch_count,)}, got {result.shape}"
        )
    return result


def _patch_matrix(name: str, value: Any, patch_count: int) -> jnp.ndarray:
    result = jnp.asarray(value, dtype=jnp.float64)
    if result.ndim != 2 or result.shape[0] != patch_count:
        raise ValueError(
            f"{name} must start with the native patch axis {patch_count}; got {result.shape}"
        )
    return result


def _scatter_compact(
    base: jnp.ndarray,
    patch_index: Any,
    compact_value: Any,
    *,
    name: str,
) -> jnp.ndarray:

    patch = jnp.asarray(patch_index, dtype=jnp.int32)
    value = jnp.asarray(compact_value, dtype=jnp.float64)
    try:
        value = jnp.broadcast_to(value, patch.shape)
    except ValueError as error:
        raise ValueError(
            f"{name} compact values {value.shape} cannot map to {patch.shape}"
        ) from error
    return base.at[jnp.reshape(patch, (-1,))].set(jnp.reshape(value, (-1,)))


def _broadcast_landunit_to_surface(
    value: Any,
    surface_shape: tuple[int, ...],
) -> jnp.ndarray:
    landunit = jnp.asarray(value, dtype=jnp.float64)
    if len(surface_shape) != 2 or landunit.shape != (surface_shape[0],):
        raise ValueError(
            "urban landunit field and compact surface mapping disagree: "
            f"{landunit.shape} versus {surface_shape}"
        )
    return jnp.broadcast_to(landunit[:, None], surface_shape)


def compose_land_to_atmosphere_physical_fields(
    canopy_flux: Any,
    urban_inputs: Any,
    urban_flux: Any,
    lake_inputs: Any,
    lake_flux: Any,
    dust: Any,
    voc: Any,
) -> LandToAtmospherePhysicalFields:

    canopy_state = canopy_flux.state
    post = canopy_state.active.postprocess
    friction = canopy_state.friction_velocity
    patch_count = int(jnp.shape(post.reference_temperature_2m_patch)[0])

    reference_temperature = _patch_vector(
        "canopy/bare reference temperature",
        post.reference_temperature_2m_patch,
        patch_count,
    )
    vegetation_temperature = _patch_vector(
        "canopy/bare vegetation temperature",
        canopy_state.vegetation_temperature_patch,
        patch_count,
    )
    reference_humidity = _patch_vector(
        "canopy/bare reference humidity",
        post.reference_specific_humidity_2m_patch,
        patch_count,
    )
    reference_wind = _patch_vector(
        "canopy/bare 10-m wind",
        friction.wind_10m_patch,
        patch_count,
    )
    eastward_stress = _patch_vector(
        "canopy/bare eastward stress",
        post.eastward_wind_stress_patch,
        patch_count,
    )
    northward_stress = _patch_vector(
        "canopy/bare northward stress",
        post.northward_wind_stress_patch,
        patch_count,
    )
    friction_velocity = _patch_vector(
        "canopy/bare friction velocity",
        friction.dust_friction_velocity_patch,
        patch_count,
    )

    urban_patch = jnp.asarray(
        urban_inputs.native_map.surface_patch_index,
        dtype=jnp.int32,
    )
    urban_atmosphere = urban_flux.surface.atmosphere
    urban_energy = urban_flux.surface.energy
    urban_friction = urban_flux.turbulence.state.friction_velocity
    reference_temperature = _scatter_compact(
        reference_temperature,
        urban_patch,
        urban_atmosphere.reference_temperature_surface,
        name="urban reference temperature",
    )
    vegetation_temperature = _scatter_compact(
        vegetation_temperature,
        urban_patch,
        urban_atmosphere.vegetation_temperature_surface,
        name="urban vegetation temperature",
    )
    reference_humidity = _scatter_compact(
        reference_humidity,
        urban_patch,
        urban_atmosphere.reference_specific_humidity_surface,
        name="urban reference humidity",
    )
    reference_wind = _scatter_compact(
        reference_wind,
        urban_patch,
        _broadcast_landunit_to_surface(urban_friction.wind_10m_patch, urban_patch.shape),
        name="urban 10-m wind",
    )
    eastward_stress = _scatter_compact(
        eastward_stress,
        urban_patch,
        urban_energy.eastward_wind_stress_surface,
        name="urban eastward stress",
    )
    northward_stress = _scatter_compact(
        northward_stress,
        urban_patch,
        urban_energy.northward_wind_stress_surface,
        name="urban northward stress",
    )
    friction_velocity = _scatter_compact(
        friction_velocity,
        urban_patch,
        _broadcast_landunit_to_surface(
            urban_friction.dust_friction_velocity_patch,
            urban_patch.shape,
        ),
        name="urban friction velocity",
    )

    lake_patch = jnp.asarray(lake_inputs.native_map.patch_index, dtype=jnp.int32)
    lake = lake_flux.fluxes
    reference_temperature = _scatter_compact(
        reference_temperature,
        lake_patch,
        lake.temperature.reference_temperature,
        name="lake reference temperature",
    )
    vegetation_temperature = _scatter_compact(
        vegetation_temperature,
        lake_patch,
        lake.temperature.vegetation_temperature,
        name="lake vegetation temperature",
    )
    reference_humidity = _scatter_compact(
        reference_humidity,
        lake_patch,
        lake.water.reference_specific_humidity,
        name="lake reference humidity",
    )
    reference_wind = _scatter_compact(
        reference_wind,
        lake_patch,
        lake.state.friction_velocity.wind_10m_patch,
        name="lake 10-m wind",
    )
    eastward_stress = _scatter_compact(
        eastward_stress,
        lake_patch,
        lake.energy.eastward_wind_stress,
        name="lake eastward stress",
    )
    northward_stress = _scatter_compact(
        northward_stress,
        lake_patch,
        lake.energy.northward_wind_stress,
        name="lake northward stress",
    )
    friction_velocity = _scatter_compact(
        friction_velocity,
        lake_patch,
        lake.state.friction_velocity.dust_friction_velocity_patch,
        name="lake friction velocity",
    )

    aerodynamic_resistance = _patch_vector(
        "final aerodynamic resistance",
        dust.surface.aerodynamic_resistance_patch,
        patch_count,
    )
    voc_flux = _patch_matrix(
        "VOC mechanism flux",
        voc.state.mechanism_flux_patch,
        patch_count,
    )
    dust_flux = _patch_matrix(
        "dust emission flux",
        dust.state.emission_flux_patch,
        patch_count,
    )
    return LandToAtmospherePhysicalFields(
        reference_temperature_2m_patch=reference_temperature,
        vegetation_temperature_patch=vegetation_temperature,
        reference_specific_humidity_2m_patch=reference_humidity,
        reference_wind_speed_10m_patch=reference_wind,
        eastward_stress_patch=eastward_stress,
        northward_stress_patch=northward_stress,
        friction_velocity_patch=friction_velocity,
        aerodynamic_resistance_patch=aerodynamic_resistance,
        voc_flux_patch=voc_flux,
        dust_flux_patch=dust_flux,
    )


__all__ = [
    "LandToAtmospherePhysicalFields",
    "compose_land_to_atmosphere_physical_fields",
]
