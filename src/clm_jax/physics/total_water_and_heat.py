
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import lax

from clm_jax.constants import FREEZING_TEMPERATURE_FRESHWATER, LATENT_HEAT_FUSION
from clm_jax.io.restart import (
    COLUMN_URBAN_IMPERVIOUS_ROAD,
    COLUMN_URBAN_ROOF,
    COLUMN_URBAN_SHADE_WALL,
    COLUMN_URBAN_SUN_WALL,
    LANDUNIT_GLACIER,
    LANDUNIT_LAKE,
    LANDUNIT_WETLAND,
    SubgridTopology,
)
from clm_jax.physics.hydrology_drainage import AQUIFER_WATER_BASELINE_MILLIMETERS
from clm_jax.physics.numerics import fortran_add, fortran_multiply
from clm_jax.physics.snow_layer_combining import (
    ICE_SPECIFIC_HEAT,
    LIQUID_WATER_SPECIFIC_HEAT,
)
from clm_jax.physics.snow_water import (
    GROUND_SURFACE_INDEX,
    NUMBER_OF_GROUND_LEVELS,
    NUMBER_OF_SNOW_LEVELS,
    NUMBER_OF_SOIL_SNOW_LEVELS,
)
from clm_jax.physics.subgrid import patch_to_column_average

NUMBER_OF_URBAN_LEVELS = 5
DELTA_LIQUID_MINIMUM_TEMPERATURE = np.float64(FREEZING_TEMPERATURE_FRESHWATER)
DELTA_LIQUID_MAXIMUM_TEMPERATURE = np.float64(FREEZING_TEMPERATURE_FRESHWATER + 35.0)


class TotalWaterAndHeatInputs(NamedTuple):

    snow_layer_count_column: jnp.ndarray
    hydrologically_active_column: jnp.ndarray
    layer_thickness_column_ground_level: jnp.ndarray
    impervious_road_layer_count_landunit: jnp.ndarray
    wall_heat_capacity_landunit_level: jnp.ndarray
    roof_heat_capacity_landunit_level: jnp.ndarray
    impervious_road_heat_capacity_landunit_level: jnp.ndarray
    saturated_water_content_column_ground_level: jnp.ndarray
    soil_solid_heat_capacity_column_ground_level: jnp.ndarray
    soil_snow_temperature_column_level: jnp.ndarray
    surface_water_temperature_column: jnp.ndarray
    soil_snow_liquid_water_column_level: jnp.ndarray
    soil_snow_ice_water_column_level: jnp.ndarray
    snow_water_equivalent_column: jnp.ndarray
    surface_water_column: jnp.ndarray
    canopy_water_patch: jnp.ndarray
    canopy_snow_patch: jnp.ndarray
    total_plant_stored_water_column: jnp.ndarray
    aquifer_water_column: jnp.ndarray


class TotalWaterAndHeatResult(NamedTuple):
    liquid_water_mass_column: jnp.ndarray
    ice_water_mass_column: jnp.ndarray
    heat_content_column: jnp.ndarray
    liquid_sensible_heat_column: jnp.ndarray
    liquid_heat_capacity_column: jnp.ndarray


def _validate_shapes(topology: SubgridTopology, inputs: TotalWaterAndHeatInputs) -> None:
    column_count = topology.column.gridcell.shape[0]
    landunit_count = topology.landunit.gridcell.shape[0]
    patch_count = topology.patch.gridcell.shape[0]
    column = (column_count,)
    landunit = (landunit_count,)
    patch = (patch_count,)
    ground = (column_count, NUMBER_OF_GROUND_LEVELS)
    soil_snow = (column_count, NUMBER_OF_SOIL_SNOW_LEVELS)
    urban = (landunit_count, NUMBER_OF_URBAN_LEVELS)
    expected = {
        "snow_layer_count_column": column,
        "hydrologically_active_column": column,
        "layer_thickness_column_ground_level": ground,
        "impervious_road_layer_count_landunit": landunit,
        "wall_heat_capacity_landunit_level": urban,
        "roof_heat_capacity_landunit_level": urban,
        "impervious_road_heat_capacity_landunit_level": urban,
        "saturated_water_content_column_ground_level": ground,
        "soil_solid_heat_capacity_column_ground_level": ground,
        "soil_snow_temperature_column_level": soil_snow,
        "surface_water_temperature_column": column,
        "soil_snow_liquid_water_column_level": soil_snow,
        "soil_snow_ice_water_column_level": soil_snow,
        "snow_water_equivalent_column": column,
        "surface_water_column": column,
        "canopy_water_patch": patch,
        "canopy_snow_patch": patch,
        "total_plant_stored_water_column": column,
        "aquifer_water_column": column,
    }
    for name, shape in expected.items():
        if getattr(inputs, name).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")


def _temperature_to_heat(temperature: jnp.ndarray, heat_capacity: jnp.ndarray) -> jnp.ndarray:
    temperature_difference = fortran_add(
        jnp.asarray(temperature, dtype=jnp.float64),
        -jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
    )
    return fortran_multiply(heat_capacity, temperature_difference)


def _accumulate_liquid_water_heat(
    temperature: jnp.ndarray,
    water_mass: jnp.ndarray,
    heat_capacity: jnp.ndarray,
    sensible_heat: jnp.ndarray,
    latent_heat: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    local_capacity = fortran_multiply(
        water_mass,
        jnp.float64(LIQUID_WATER_SPECIFIC_HEAT),
    )
    return (
        fortran_add(heat_capacity, local_capacity),
        fortran_add(sensible_heat, _temperature_to_heat(temperature, local_capacity)),
        fortran_add(
            latent_heat,
            fortran_multiply(water_mass, jnp.float64(LATENT_HEAT_FUSION)),
        ),
    )


def liquid_water_heat(temperature: jnp.ndarray, water_mass: jnp.ndarray) -> jnp.ndarray:

    mass = jnp.asarray(water_mass, dtype=jnp.float64)
    zeros = jnp.zeros_like(mass)
    _, sensible, latent = _accumulate_liquid_water_heat(
        jnp.asarray(temperature, dtype=jnp.float64),
        mass,
        zeros,
        zeros,
        zeros,
    )
    return fortran_add(sensible, latent)


def adjust_delta_heat_for_delta_liquid(
    delta_liquid: jnp.ndarray,
    liquid_water_temperature_before: jnp.ndarray,
    liquid_water_temperature_after: jnp.ndarray,
    delta_heat: jnp.ndarray,
) -> jnp.ndarray:

    delta = jnp.asarray(delta_liquid, dtype=jnp.float64)
    before = jnp.asarray(liquid_water_temperature_before, dtype=jnp.float64)
    after = jnp.asarray(liquid_water_temperature_after, dtype=jnp.float64)
    heat = jnp.asarray(delta_heat, dtype=jnp.float64)
    if not (delta.shape == before.shape == after.shape == heat.shape):
        raise ValueError("all gridcell heat-adjustment arrays must have identical shapes")
    water_temperature = jnp.where(delta < jnp.float64(0.0), before, after)
    water_temperature = jnp.maximum(
        water_temperature,
        jnp.float64(DELTA_LIQUID_MINIMUM_TEMPERATURE),
    )
    water_temperature = jnp.minimum(
        water_temperature,
        jnp.float64(DELTA_LIQUID_MAXIMUM_TEMPERATURE),
    )
    adjusted = fortran_add(heat, -liquid_water_heat(water_temperature, delta))
    return jnp.where(delta != jnp.float64(0.0), adjusted, heat)


def compute_total_water_and_heat(
    topology: SubgridTopology,
    inputs: TotalWaterAndHeatInputs,
    *,
    maximum_patches_per_column: int = 79,
) -> TotalWaterAndHeatResult:

    _validate_shapes(topology, inputs)
    active = jnp.asarray(topology.column.active, dtype=jnp.bool_)
    landunit_type = jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
    column_type = jnp.asarray(topology.column.column_type, dtype=jnp.int32)
    landunit_index = jnp.asarray(topology.column.landunit, dtype=jnp.int32)
    nonlake = active & (landunit_type != jnp.int32(LANDUNIT_LAKE))
    lake = active & (landunit_type == jnp.int32(LANDUNIT_LAKE))
    snow_count = jnp.asarray(inputs.snow_layer_count_column, dtype=jnp.int32)
    liquid_water = jnp.asarray(inputs.soil_snow_liquid_water_column_level, dtype=jnp.float64)
    ice_water = jnp.asarray(inputs.soil_snow_ice_water_column_level, dtype=jnp.float64)
    temperature = jnp.asarray(inputs.soil_snow_temperature_column_level, dtype=jnp.float64)

    canopy_water = patch_to_column_average(
        inputs.canopy_water_patch,
        topology.patch.column,
        topology.patch.weight_on_column,
        topology.patch.active,
        nonlake,
        maximum_patches_per_column=maximum_patches_per_column,
    )
    canopy_snow = patch_to_column_average(
        inputs.canopy_snow_patch,
        topology.patch.column,
        topology.patch.weight_on_column,
        topology.patch.active,
        nonlake,
        maximum_patches_per_column=maximum_patches_per_column,
    )
    canopy_liquid_mass = fortran_add(canopy_water, -canopy_snow)
    liquid_mass = jnp.where(
        nonlake,
        fortran_add(canopy_liquid_mass, inputs.total_plant_stored_water_column),
        jnp.float64(0.0),
    )
    ice_mass = jnp.where(nonlake, canopy_snow, jnp.float64(0.0))

    liquid_heat_capacity = jnp.zeros_like(liquid_mass)
    liquid_sensible_heat = jnp.zeros_like(liquid_mass)
    liquid_latent_heat = jnp.zeros_like(liquid_mass)
    dry_heat = jnp.zeros_like(liquid_mass)
    ice_heat = jnp.zeros_like(liquid_mass)
    canopy_and_plant_water = fortran_add(
        canopy_liquid_mass,
        inputs.total_plant_stored_water_column,
    )
    canopy_cv, canopy_sensible, canopy_latent = _accumulate_liquid_water_heat(
        jnp.full_like(liquid_mass, jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)),
        canopy_and_plant_water,
        liquid_heat_capacity,
        liquid_sensible_heat,
        liquid_latent_heat,
    )
    liquid_heat_capacity = jnp.where(nonlake, canopy_cv, liquid_heat_capacity)
    liquid_sensible_heat = jnp.where(nonlake, canopy_sensible, liquid_sensible_heat)
    liquid_latent_heat = jnp.where(nonlake, canopy_latent, liquid_latent_heat)

    snow_level_number = jnp.arange(
        -NUMBER_OF_SNOW_LEVELS + 1,
        1,
        dtype=jnp.int32,
    )

    def accumulate_snow(level: int, carry):
        liq_mass, ice_mass_now, cv, sensible, latent, ice_sensible = carry
        explicit = active & (snow_count < jnp.int32(0)) & (
            snow_level_number[level] >= snow_count + jnp.int32(1)
        )
        liq_mass_candidate = fortran_add(liq_mass, liquid_water[:, level])
        ice_mass_candidate = fortran_add(ice_mass_now, ice_water[:, level])
        cv_candidate, sensible_candidate, latent_candidate = _accumulate_liquid_water_heat(
            temperature[:, level],
            liquid_water[:, level],
            cv,
            sensible,
            latent,
        )
        local_ice_capacity = fortran_multiply(
            ice_water[:, level],
            jnp.float64(ICE_SPECIFIC_HEAT),
        )
        ice_candidate = fortran_add(
            ice_sensible,
            _temperature_to_heat(temperature[:, level], local_ice_capacity),
        )
        return (
            jnp.where(explicit, liq_mass_candidate, liq_mass),
            jnp.where(explicit, ice_mass_candidate, ice_mass_now),
            jnp.where(explicit, cv_candidate, cv),
            jnp.where(explicit, sensible_candidate, sensible),
            jnp.where(explicit, latent_candidate, latent),
            jnp.where(explicit, ice_candidate, ice_sensible),
        )

    (
        liquid_mass,
        ice_mass,
        liquid_heat_capacity,
        liquid_sensible_heat,
        liquid_latent_heat,
        ice_heat,
    ) = lax.fori_loop(
        0,
        NUMBER_OF_SNOW_LEVELS,
        accumulate_snow,
        (
            liquid_mass,
            ice_mass,
            liquid_heat_capacity,
            liquid_sensible_heat,
            liquid_latent_heat,
            ice_heat,
        ),
    )

    unresolved = active & (snow_count >= jnp.int32(0)) & (
        inputs.snow_water_equivalent_column != jnp.float64(0.0)
    )
    ice_mass = jnp.where(
        unresolved,
        fortran_add(ice_mass, inputs.snow_water_equivalent_column),
        ice_mass,
    )
    unresolved_nonlake = unresolved & nonlake
    unresolved_ice_capacity = fortran_multiply(
        inputs.snow_water_equivalent_column,
        jnp.float64(ICE_SPECIFIC_HEAT),
    )
    unresolved_ice_heat = fortran_add(
        ice_heat,
        _temperature_to_heat(
            temperature[:, GROUND_SURFACE_INDEX],
            unresolved_ice_capacity,
        ),
    )
    ice_heat = jnp.where(unresolved_nonlake, unresolved_ice_heat, ice_heat)

    hydrology = nonlake & jnp.asarray(inputs.hydrologically_active_column, dtype=jnp.bool_)
    aquifer_excess = fortran_add(
        inputs.aquifer_water_column,
        -jnp.float64(AQUIFER_WATER_BASELINE_MILLIMETERS),
    )
    liquid_mass = jnp.where(
        hydrology,
        fortran_add(liquid_mass, aquifer_excess),
        liquid_mass,
    )
    aquifer_cv, aquifer_sensible, aquifer_latent = _accumulate_liquid_water_heat(
        jnp.full_like(liquid_mass, jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)),
        aquifer_excess,
        liquid_heat_capacity,
        liquid_sensible_heat,
        liquid_latent_heat,
    )
    liquid_heat_capacity = jnp.where(hydrology, aquifer_cv, liquid_heat_capacity)
    liquid_sensible_heat = jnp.where(hydrology, aquifer_sensible, liquid_sensible_heat)
    liquid_latent_heat = jnp.where(hydrology, aquifer_latent, liquid_latent_heat)

    walls = (column_type == jnp.int32(COLUMN_URBAN_SUN_WALL)) | (
        column_type == jnp.int32(COLUMN_URBAN_SHADE_WALL)
    )
    excludes_surface = (
        (column_type == jnp.int32(COLUMN_URBAN_ROOF))
        | walls
        | (column_type == jnp.int32(COLUMN_URBAN_IMPERVIOUS_ROAD))
    )
    includes_surface = nonlake & ~excludes_surface
    liquid_mass = jnp.where(
        includes_surface,
        fortran_add(liquid_mass, inputs.surface_water_column),
        liquid_mass,
    )
    surface_cv, surface_sensible, surface_latent = _accumulate_liquid_water_heat(
        inputs.surface_water_temperature_column,
        inputs.surface_water_column,
        liquid_heat_capacity,
        liquid_sensible_heat,
        liquid_latent_heat,
    )
    liquid_heat_capacity = jnp.where(includes_surface, surface_cv, liquid_heat_capacity)
    liquid_sensible_heat = jnp.where(includes_surface, surface_sensible, liquid_sensible_heat)
    liquid_latent_heat = jnp.where(includes_surface, surface_latent, liquid_latent_heat)

    road_layers = jnp.asarray(inputs.impervious_road_layer_count_landunit, dtype=jnp.int32)[
        landunit_index
    ]
    wall_capacity = jnp.asarray(inputs.wall_heat_capacity_landunit_level, dtype=jnp.float64)[
        landunit_index
    ]
    roof_capacity = jnp.asarray(inputs.roof_heat_capacity_landunit_level, dtype=jnp.float64)[
        landunit_index
    ]
    road_capacity = jnp.asarray(
        inputs.impervious_road_heat_capacity_landunit_level,
        dtype=jnp.float64,
    )[landunit_index]
    wetland_or_glacier = (landunit_type == jnp.int32(LANDUNIT_WETLAND)) | (
        landunit_type == jnp.int32(LANDUNIT_GLACIER)
    )
    roof = column_type == jnp.int32(COLUMN_URBAN_ROOF)
    impervious_road = column_type == jnp.int32(COLUMN_URBAN_IMPERVIOUS_ROAD)

    def accumulate_ground(level: int, carry):
        liq_mass, ice_mass_now, cv, sensible, latent, dry, ice_sensible = carry
        source_level = GROUND_SURFACE_INDEX + level
        urban_level = jnp.minimum(jnp.int32(level), jnp.int32(NUMBER_OF_URBAN_LEVELS - 1))
        roof_too_deep = roof & (level >= jnp.int32(NUMBER_OF_URBAN_LEVELS))
        has_water = active & ~walls & ~roof_too_deep
        liq_mass_candidate = fortran_add(liq_mass, liquid_water[:, source_level])
        ice_mass_candidate = fortran_add(ice_mass_now, ice_water[:, source_level])
        cv_candidate, sensible_candidate, latent_candidate = _accumulate_liquid_water_heat(
            temperature[:, source_level],
            liquid_water[:, source_level],
            cv,
            sensible,
            latent,
        )
        local_ice_capacity = fortran_multiply(
            ice_water[:, source_level],
            jnp.float64(ICE_SPECIFIC_HEAT),
        )
        ice_candidate = fortran_add(
            ice_sensible,
            _temperature_to_heat(temperature[:, source_level], local_ice_capacity),
        )

        thickness = inputs.layer_thickness_column_ground_level[:, level]
        wall_volume_capacity = fortran_multiply(wall_capacity[:, urban_level], thickness)
        roof_volume_capacity = fortran_multiply(roof_capacity[:, urban_level], thickness)
        road_volume_capacity = fortran_multiply(road_capacity[:, urban_level], thickness)
        soil_solid_fraction = fortran_add(
            jnp.float64(1.0),
            -inputs.saturated_water_content_column_ground_level[:, level],
        )
        soil_volume_capacity = fortran_multiply(
            inputs.soil_solid_heat_capacity_column_ground_level[:, level],
            soil_solid_fraction,
        )
        soil_volume_capacity = fortran_multiply(soil_volume_capacity, thickness)
        in_urban_depth = level < jnp.int32(NUMBER_OF_URBAN_LEVELS)
        use_wall = nonlake & walls & in_urban_depth
        use_roof = nonlake & roof & in_urban_depth
        use_road = (
            nonlake
            & impervious_road
            & (level < road_layers)
            & ~walls
            & ~roof
        )
        use_soil_nonlake = (
            nonlake
            & ~walls
            & ~roof
            & ~use_road
            & ~wetland_or_glacier
        )
        use_soil_lake = lake
        selected_dry_capacity = jnp.where(
            use_wall,
            wall_volume_capacity,
            jnp.where(
                use_roof,
                roof_volume_capacity,
                jnp.where(
                    use_road,
                    road_volume_capacity,
                    soil_volume_capacity,
                ),
            ),
        )
        add_dry = use_wall | use_roof | use_road | use_soil_nonlake | use_soil_lake
        dry_candidate = fortran_add(
            dry,
            _temperature_to_heat(temperature[:, source_level], selected_dry_capacity),
        )
        return (
            jnp.where(has_water, liq_mass_candidate, liq_mass),
            jnp.where(has_water, ice_mass_candidate, ice_mass_now),
            jnp.where(has_water, cv_candidate, cv),
            jnp.where(has_water, sensible_candidate, sensible),
            jnp.where(has_water, latent_candidate, latent),
            jnp.where(add_dry, dry_candidate, dry),
            jnp.where(has_water, ice_candidate, ice_sensible),
        )

    (
        liquid_mass,
        ice_mass,
        liquid_heat_capacity,
        liquid_sensible_heat,
        liquid_latent_heat,
        dry_heat,
        ice_heat,
    ) = lax.fori_loop(
        0,
        NUMBER_OF_GROUND_LEVELS,
        accumulate_ground,
        (
            liquid_mass,
            ice_mass,
            liquid_heat_capacity,
            liquid_sensible_heat,
            liquid_latent_heat,
            dry_heat,
            ice_heat,
        ),
    )
    heat = fortran_add(dry_heat, ice_heat)
    heat = fortran_add(heat, liquid_sensible_heat)
    heat = fortran_add(heat, liquid_latent_heat)
    return TotalWaterAndHeatResult(
        liquid_water_mass_column=liquid_mass,
        ice_water_mass_column=ice_mass,
        heat_content_column=heat,
        liquid_sensible_heat_column=liquid_sensible_heat,
        liquid_heat_capacity_column=liquid_heat_capacity,
    )
