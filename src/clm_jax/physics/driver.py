
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.physics.numerics import fortran_add


class DriverInitializationResult(NamedTuple):
    intracellular_sunlit_co2: jnp.ndarray
    intracellular_shaded_co2: jnp.ndarray
    previous_snow_water_equivalent: jnp.ndarray
    bottom_heat_flux: jnp.ndarray
    vegetation_exposed: jnp.ndarray
    previous_snow_ice_fraction: jnp.ndarray


def initialize_driver_state(
    intracellular_sunlit_co2: jnp.ndarray,
    intracellular_shaded_co2: jnp.ndarray,
    snow_water_equivalent: jnp.ndarray,
    snow_layer_count: jnp.ndarray,
    snow_layer_ice: jnp.ndarray,
    snow_layer_liquid: jnp.ndarray,
    previous_snow_ice_fraction: jnp.ndarray,
    patch_active: jnp.ndarray,
    vegetation_exposed_for_albedo: jnp.ndarray,
    non_lake_column: jnp.ndarray,
    *,
    number_of_snow_levels: int | None = None,
) -> DriverInitializationResult:

    sunlit = jnp.asarray(intracellular_sunlit_co2, dtype=jnp.float64)
    shaded = jnp.asarray(intracellular_shaded_co2, dtype=jnp.float64)
    snow_water = jnp.asarray(snow_water_equivalent, dtype=jnp.float64)
    layer_count = jnp.asarray(snow_layer_count, dtype=jnp.int32)
    layer_ice = jnp.asarray(snow_layer_ice, dtype=jnp.float64)
    layer_liquid = jnp.asarray(snow_layer_liquid, dtype=jnp.float64)
    old_ice_fraction = jnp.asarray(previous_snow_ice_fraction, dtype=jnp.float64)
    active_patch = jnp.asarray(patch_active, dtype=jnp.bool_)
    albedo_exposed = jnp.asarray(vegetation_exposed_for_albedo, dtype=jnp.int32)
    non_lake = jnp.asarray(non_lake_column, dtype=jnp.bool_)

    if layer_ice.ndim != 2 or layer_liquid.shape != layer_ice.shape:
        raise ValueError("snow ice and liquid arrays must share (column, level) axes")
    if old_ice_fraction.shape != layer_ice.shape:
        raise ValueError("previous snow ice fraction must match the water arrays")
    number_of_columns, number_of_total_levels = layer_ice.shape
    snow_levels = (
        number_of_total_levels
        if number_of_snow_levels is None
        else number_of_snow_levels
    )
    if not isinstance(snow_levels, int) or not 1 <= snow_levels <= number_of_total_levels:
        raise ValueError("number_of_snow_levels must lie within the level axis")
    if snow_water.shape != (number_of_columns,):
        raise ValueError("snow water and layer arrays must share the column axis")
    if layer_count.shape != (number_of_columns,) or non_lake.shape != (
        number_of_columns,
    ):
        raise ValueError("column masks and snow counts must share the column axis")
    if active_patch.shape != albedo_exposed.shape:
        raise ValueError("patch activity and exposed-vegetation arrays must match")

    level_number = jnp.arange(-snow_levels + 1, 1, dtype=jnp.int32)
    active_snow_layer = level_number[None, :] >= layer_count[:, None] + jnp.int32(1)
    update_ice_fraction = non_lake[:, None] & active_snow_layer
    snow_ice = layer_ice[:, :snow_levels]
    snow_liquid = layer_liquid[:, :snow_levels]
    snow_layer_water = fortran_add(snow_liquid, snow_ice)
    safe_snow_layer_water = jnp.where(
        update_ice_fraction,
        snow_layer_water,
        jnp.float64(1.0),
    )
    current_ice_fraction = snow_ice / safe_snow_layer_water
    updated_snow_fraction = jnp.where(
        update_ice_fraction,
        current_ice_fraction,
        old_ice_fraction[:, :snow_levels],
    )
    old_ice_fraction = jnp.concatenate(
        (updated_snow_fraction, old_ice_fraction[:, snow_levels:]),
        axis=1,
    )

    return DriverInitializationResult(
        intracellular_sunlit_co2=jnp.full_like(sunlit, jnp.float64(-999.0)),
        intracellular_shaded_co2=jnp.full_like(shaded, jnp.float64(-999.0)),
        previous_snow_water_equivalent=snow_water,
        bottom_heat_flux=jnp.zeros_like(snow_water),
        vegetation_exposed=jnp.where(active_patch, albedo_exposed, jnp.int32(0)),
        previous_snow_ice_fraction=old_ice_fraction,
    )
