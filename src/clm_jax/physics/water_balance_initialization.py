
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from clm_jax.io.restart import (
    COLUMN_URBAN_IMPERVIOUS_ROAD,
    COLUMN_URBAN_PERVIOUS_ROAD,
    COLUMN_URBAN_ROOF,
    COLUMN_URBAN_SHADE_WALL,
    COLUMN_URBAN_SUN_WALL,
    LANDUNIT_CROP,
    LANDUNIT_LAKE,
    LANDUNIT_SOIL,
    SubgridTopology,
)
from clm_jax.physics.numerics import fortran_add
from clm_jax.physics.snow_water import (
    GROUND_SURFACE_INDEX,
    NUMBER_OF_GROUND_LEVELS,
    NUMBER_OF_SNOW_LEVELS,
    NUMBER_OF_SOIL_SNOW_LEVELS,
)
from clm_jax.physics.subgrid import patch_to_column_average


class WaterBalanceInitializationInputs(NamedTuple):
    snow_layer_count_column: jnp.ndarray
    water_table_depth_column: jnp.ndarray
    soil_bottom_interface_depth_column: jnp.ndarray
    surface_water_column: jnp.ndarray
    snow_water_equivalent_column: jnp.ndarray
    canopy_water_patch: jnp.ndarray
    canopy_snow_patch: jnp.ndarray
    total_plant_stored_water_column: jnp.ndarray
    soil_snow_liquid_water_column_level: jnp.ndarray
    soil_snow_ice_water_column_level: jnp.ndarray


class WaterBalanceInitializationState(NamedTuple):
    aquifer_water_column: jnp.ndarray
    beginning_water_mass_column: jnp.ndarray


class WaterBalanceInitializationParameters(NamedTuple):
    aquifer_water_baseline: float = 5000.0
    number_of_soil_levels: int = 20
    number_of_urban_levels: int = 5
    maximum_patches_per_column: int = 79


DEFAULT_WATER_BALANCE_INITIALIZATION_PARAMETERS = WaterBalanceInitializationParameters()


class WaterBalanceInitializationDiagnostics(NamedTuple):
    nonlake_column: jnp.ndarray
    lake_column: jnp.ndarray
    hydrologically_active_column: jnp.ndarray
    aquifer_reset_column: jnp.ndarray
    liquid_water_mass_column: jnp.ndarray
    ice_water_mass_column: jnp.ndarray


class WaterBalanceInitializationResult(NamedTuple):
    state: WaterBalanceInitializationState
    diagnostics: WaterBalanceInitializationDiagnostics


def _validate_shapes(
    topology: SubgridTopology,
    inputs: WaterBalanceInitializationInputs,
    state: WaterBalanceInitializationState,
    parameters: WaterBalanceInitializationParameters,
) -> None:
    column_shape = topology.column.gridcell.shape
    patch_shape = topology.patch.gridcell.shape
    if len(column_shape) != 1 or len(patch_shape) != 1:
        raise ValueError("native column and patch topology must be vectors")
    column_fields = (
        "snow_layer_count_column",
        "water_table_depth_column",
        "soil_bottom_interface_depth_column",
        "surface_water_column",
        "snow_water_equivalent_column",
        "total_plant_stored_water_column",
    )
    for name in column_fields:
        if getattr(inputs, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")
    for name in ("canopy_water_patch", "canopy_snow_patch"):
        if getattr(inputs, name).shape != patch_shape:
            raise ValueError(f"{name} must have shape {patch_shape}")
    level_shape = (column_shape[0], NUMBER_OF_SOIL_SNOW_LEVELS)
    for name in (
        "soil_snow_liquid_water_column_level",
        "soil_snow_ice_water_column_level",
    ):
        if getattr(inputs, name).shape != level_shape:
            raise ValueError(f"{name} must have shape {level_shape}")
    for name in WaterBalanceInitializationState._fields:
        if getattr(state, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")
    if parameters.number_of_soil_levels != 20:
        raise ValueError("reference case requires 20 soil levels")
    if parameters.number_of_urban_levels != 5:
        raise ValueError("reference case requires 5 urban levels")
    if parameters.maximum_patches_per_column < 1:
        raise ValueError("maximum_patches_per_column must be positive")


def begin_water_balance(
    topology: SubgridTopology,
    inputs: WaterBalanceInitializationInputs,
    state: WaterBalanceInitializationState,
    *,
    parameters: WaterBalanceInitializationParameters = (
        DEFAULT_WATER_BALANCE_INITIALIZATION_PARAMETERS
    ),
) -> WaterBalanceInitializationResult:

    _validate_shapes(topology, inputs, state, parameters)
    active = jnp.asarray(topology.column.active, dtype=jnp.bool_)
    landunit_type = jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
    column_type = jnp.asarray(topology.column.column_type, dtype=jnp.int32)
    lake = active & (landunit_type == jnp.int32(LANDUNIT_LAKE))
    nonlake = active & ~lake
    hydrologically_active = (
        (landunit_type == jnp.int32(LANDUNIT_SOIL))
        | (landunit_type == jnp.int32(LANDUNIT_CROP))
        | (column_type == jnp.int32(COLUMN_URBAN_PERVIOUS_ROAD))
    )

    water_table = jnp.asarray(inputs.water_table_depth_column, dtype=jnp.float64)
    soil_bottom = jnp.asarray(
        inputs.soil_bottom_interface_depth_column,
        dtype=jnp.float64,
    )
    reset_aquifer = nonlake & hydrologically_active & (water_table <= soil_bottom)
    aquifer = jnp.where(
        reset_aquifer,
        jnp.float64(parameters.aquifer_water_baseline),
        jnp.asarray(state.aquifer_water_column, dtype=jnp.float64),
    )

    canopy_water_column = patch_to_column_average(
        inputs.canopy_water_patch,
        topology.patch.column,
        topology.patch.weight_on_column,
        topology.patch.active,
        nonlake,
        maximum_patches_per_column=parameters.maximum_patches_per_column,
    )
    canopy_snow_column = patch_to_column_average(
        inputs.canopy_snow_patch,
        topology.patch.column,
        topology.patch.weight_on_column,
        topology.patch.active,
        nonlake,
        maximum_patches_per_column=parameters.maximum_patches_per_column,
    )
    plant_water = jnp.asarray(
        inputs.total_plant_stored_water_column,
        dtype=jnp.float64,
    )
    canopy_liquid = fortran_add(canopy_water_column, -canopy_snow_column)
    nonlake_liquid = jnp.where(
        nonlake,
        fortran_add(fortran_add(jnp.float64(0.0), canopy_liquid), plant_water),
        jnp.float64(0.0),
    )
    nonlake_ice = jnp.where(nonlake, canopy_snow_column, jnp.float64(0.0))
    lake_liquid = jnp.zeros_like(nonlake_liquid)
    lake_ice = jnp.zeros_like(nonlake_ice)

    layer_count = jnp.asarray(inputs.snow_layer_count_column, dtype=jnp.int32)
    liquid = jnp.asarray(
        inputs.soil_snow_liquid_water_column_level,
        dtype=jnp.float64,
    )
    ice = jnp.asarray(
        inputs.soil_snow_ice_water_column_level,
        dtype=jnp.float64,
    )
    snow_level_number = jnp.arange(
        -NUMBER_OF_SNOW_LEVELS + 1,
        1,
        dtype=jnp.int32,
    )

    def accumulate_snow(level: int, carry):
        nonlake_liquid_mass, nonlake_ice_mass, lake_liquid_mass, lake_ice_mass = carry
        resolved = snow_level_number[level] >= layer_count + jnp.int32(1)
        level_liquid = liquid[:, level]
        level_ice = ice[:, level]
        return (
            jnp.where(
                nonlake & resolved,
                fortran_add(nonlake_liquid_mass, level_liquid),
                nonlake_liquid_mass,
            ),
            jnp.where(
                nonlake & resolved,
                fortran_add(nonlake_ice_mass, level_ice),
                nonlake_ice_mass,
            ),
            jnp.where(
                lake & resolved,
                fortran_add(lake_liquid_mass, level_liquid),
                lake_liquid_mass,
            ),
            jnp.where(
                lake & resolved,
                fortran_add(lake_ice_mass, level_ice),
                lake_ice_mass,
            ),
        )

    nonlake_liquid, nonlake_ice, lake_liquid, lake_ice = lax.fori_loop(
        0,
        NUMBER_OF_SNOW_LEVELS,
        accumulate_snow,
        (nonlake_liquid, nonlake_ice, lake_liquid, lake_ice),
    )
    unresolved = (layer_count >= jnp.int32(0)) & (
        jnp.asarray(inputs.snow_water_equivalent_column, dtype=jnp.float64) != jnp.float64(0.0)
    )
    snow_water = jnp.asarray(
        inputs.snow_water_equivalent_column,
        dtype=jnp.float64,
    )
    nonlake_ice = jnp.where(
        nonlake & unresolved,
        fortran_add(nonlake_ice, snow_water),
        nonlake_ice,
    )
    lake_ice = jnp.where(
        lake & unresolved,
        fortran_add(lake_ice, snow_water),
        lake_ice,
    )

    nonlake_liquid = jnp.where(
        nonlake & hydrologically_active,
        fortran_add(
            nonlake_liquid,
            fortran_add(
                aquifer,
                -jnp.float64(parameters.aquifer_water_baseline),
            ),
        ),
        nonlake_liquid,
    )
    excludes_surface_water = (
        (column_type == jnp.int32(COLUMN_URBAN_ROOF))
        | (column_type == jnp.int32(COLUMN_URBAN_SUN_WALL))
        | (column_type == jnp.int32(COLUMN_URBAN_SHADE_WALL))
        | (column_type == jnp.int32(COLUMN_URBAN_IMPERVIOUS_ROAD))
    )
    nonlake_liquid = jnp.where(
        nonlake & ~excludes_surface_water,
        fortran_add(
            nonlake_liquid,
            jnp.asarray(inputs.surface_water_column, dtype=jnp.float64),
        ),
        nonlake_liquid,
    )

    def accumulate_ground(level: int, carry):
        nonlake_liquid_mass, nonlake_ice_mass, lake_liquid_mass, lake_ice_mass = carry
        native_index = level + GROUND_SURFACE_INDEX
        is_wall = (column_type == jnp.int32(COLUMN_URBAN_SUN_WALL)) | (
            column_type == jnp.int32(COLUMN_URBAN_SHADE_WALL)
        )
        roof_has_water = level < parameters.number_of_urban_levels
        nonlake_has_water = (
            nonlake & ~is_wall & ((column_type != jnp.int32(COLUMN_URBAN_ROOF)) | roof_has_water)
        )
        level_liquid = liquid[:, native_index]
        level_ice = ice[:, native_index]
        return (
            jnp.where(
                nonlake_has_water,
                fortran_add(nonlake_liquid_mass, level_liquid),
                nonlake_liquid_mass,
            ),
            jnp.where(
                nonlake_has_water,
                fortran_add(nonlake_ice_mass, level_ice),
                nonlake_ice_mass,
            ),
            jnp.where(
                lake,
                fortran_add(lake_liquid_mass, level_liquid),
                lake_liquid_mass,
            ),
            jnp.where(
                lake,
                fortran_add(lake_ice_mass, level_ice),
                lake_ice_mass,
            ),
        )

    nonlake_liquid, nonlake_ice, lake_liquid, lake_ice = lax.fori_loop(
        0,
        NUMBER_OF_GROUND_LEVELS,
        accumulate_ground,
        (nonlake_liquid, nonlake_ice, lake_liquid, lake_ice),
    )
    old_beginning_mass = jnp.asarray(
        state.beginning_water_mass_column,
        dtype=jnp.float64,
    )
    beginning_mass = jnp.where(
        nonlake,
        fortran_add(nonlake_liquid, nonlake_ice),
        old_beginning_mass,
    )
    beginning_mass = jnp.where(
        lake,
        fortran_add(lake_liquid, lake_ice),
        beginning_mass,
    )
    liquid_mass = jnp.where(
        nonlake,
        nonlake_liquid,
        jnp.where(lake, lake_liquid, jnp.float64(0.0)),
    )
    ice_mass = jnp.where(
        nonlake,
        nonlake_ice,
        jnp.where(lake, lake_ice, jnp.float64(0.0)),
    )
    return WaterBalanceInitializationResult(
        state=WaterBalanceInitializationState(
            aquifer_water_column=aquifer,
            beginning_water_mass_column=beginning_mass,
        ),
        diagnostics=WaterBalanceInitializationDiagnostics(
            nonlake_column=nonlake,
            lake_column=lake,
            hydrologically_active_column=hydrologically_active,
            aquifer_reset_column=reset_aquifer,
            liquid_water_mass_column=liquid_mass,
            ice_water_mass_column=ice_mass,
        ),
    )


__all__ = [
    "DEFAULT_WATER_BALANCE_INITIALIZATION_PARAMETERS",
    "WaterBalanceInitializationDiagnostics",
    "WaterBalanceInitializationInputs",
    "WaterBalanceInitializationParameters",
    "WaterBalanceInitializationResult",
    "WaterBalanceInitializationState",
    "begin_water_balance",
]
