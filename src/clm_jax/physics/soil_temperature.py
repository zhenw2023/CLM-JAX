
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.constants import (
    GRAVITY,
    LATENT_HEAT_FUSION,
    LATENT_HEAT_VAPORIZATION,
    STEFAN_BOLTZMANN,
)
from clm_jax.io.restart import LANDUNIT_URBAN_MAX, LANDUNIT_URBAN_MIN, SubgridTopology
from clm_jax.physics.band_diagonal import solve_five_band_system
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
    fortran_ordered_sum,
    fortran_power,
    strict_log10,
)
from clm_jax.physics.positive_power import positive_power

NUMBER_OF_SNOW_LEVELS = 12
NUMBER_OF_GROUND_LEVELS = 25
NUMBER_OF_URBAN_LEVELS = 5
NUMBER_OF_SOIL_SNOW_LEVELS = NUMBER_OF_SNOW_LEVELS + NUMBER_OF_GROUND_LEVELS
_PARALLEL_THERMAL_LEVELS: bool | None = None
_PARALLEL_SYSTEM_ASSEMBLY: bool | None = None


def _material_power(base, exponent):
    if jax.default_backend() == "gpu":
        return positive_power(base, exponent)
    return fortran_power(base, exponent)


def _parallel_thermal_levels() -> bool:
    return (
        jax.default_backend() == "gpu"
        if _PARALLEL_THERMAL_LEVELS is None else _PARALLEL_THERMAL_LEVELS
    )


def _parallel_system_assembly() -> bool:
    return (
        jax.default_backend() == "gpu"
        if _PARALLEL_SYSTEM_ASSEMBLY is None else _PARALLEL_SYSTEM_ASSEMBLY
    )

URBAN_ROOF = 71
URBAN_SUNWALL = 72
URBAN_SHADEWALL = 73
URBAN_IMPERVIOUS_ROAD = 74
URBAN_PERVIOUS_ROAD = 75

LANDUNIT_SOIL = 1
LANDUNIT_CROP = 2
LANDUNIT_ICE_MEC = 4
LANDUNIT_LAKE = 5
LANDUNIT_WETLAND = 6

LIQUID_WATER_DENSITY = 1000.0
ICE_DENSITY = 917.0
FREEZING_TEMPERATURE = 273.15
LIQUID_WATER_THERMAL_CONDUCTIVITY = 0.57
ICE_THERMAL_CONDUCTIVITY = 2.290
AIR_THERMAL_CONDUCTIVITY = 0.023
LIQUID_WATER_SPECIFIC_HEAT = 4188.0
ICE_SPECIFIC_HEAT = 2117.27
BEDROCK_THERMAL_CONDUCTIVITY = 3.0
BEDROCK_VOLUMETRIC_HEAT_CAPACITY = 2.0e6
THIN_SURFACE_LAYER = 1.0e-6
SURFACE_WATER_DEPTH_SCALE = 0.0010000000474974513

FIRST_FORTRAN_LEVEL = -NUMBER_OF_SNOW_LEVELS + 1
GROUND_FIRST_INDEX = NUMBER_OF_SNOW_LEVELS
GROUND_BOTTOM_INDEX = NUMBER_OF_SOIL_SNOW_LEVELS - 1
URBAN_BOTTOM_INDEX = NUMBER_OF_SNOW_LEVELS + NUMBER_OF_URBAN_LEVELS - 1

SURFACE_TEMPERATURE_TUNING_FACTOR = 0.34
CRANK_NICHOLSON_FACTOR = 0.5


class SoilThermalPropertyInputs(NamedTuple):

    active_nonlake: jnp.ndarray
    snow_layer_count: jnp.ndarray
    landunit_type: jnp.ndarray
    column_type: jnp.ndarray
    bedrock_layer_count: jnp.ndarray
    impervious_road_layer_count: jnp.ndarray
    layer_thickness: jnp.ndarray
    node_depth: jnp.ndarray
    interface_depth: jnp.ndarray
    soil_snow_temperature: jnp.ndarray
    liquid_water: jnp.ndarray
    ice_water: jnp.ndarray
    effective_snow_fraction: jnp.ndarray
    surface_water: jnp.ndarray
    snow_water: jnp.ndarray
    soil_mineral_conductivity: jnp.ndarray
    dry_soil_conductivity: jnp.ndarray
    soil_solid_heat_capacity: jnp.ndarray
    soil_porosity: jnp.ndarray
    urban_wall_conductivity: jnp.ndarray
    urban_roof_conductivity: jnp.ndarray
    urban_impervious_road_conductivity: jnp.ndarray
    urban_wall_volumetric_heat_capacity: jnp.ndarray
    urban_roof_volumetric_heat_capacity: jnp.ndarray
    urban_impervious_road_volumetric_heat_capacity: jnp.ndarray


class SoilThermalPropertyResult(NamedTuple):

    layer_thermal_conductivity: jnp.ndarray
    interface_thermal_conductivity: jnp.ndarray
    layer_heat_capacity: jnp.ndarray
    snow_partial_density: jnp.ndarray
    surface_water_interface_conductivity: jnp.ndarray


def _validate_thermal_property_shapes(inputs: SoilThermalPropertyInputs) -> None:
    if inputs.active_nonlake.ndim != 1:
        raise ValueError("active_nonlake must be a vector")

    number_of_columns = inputs.active_nonlake.shape[0]
    vector_shape = (number_of_columns,)
    layer_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS)
    interface_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS + 1)
    ground_shape = (number_of_columns, NUMBER_OF_GROUND_LEVELS)
    urban_shape = (number_of_columns, NUMBER_OF_URBAN_LEVELS)

    vector_fields = {
        "active_nonlake",
        "snow_layer_count",
        "landunit_type",
        "column_type",
        "bedrock_layer_count",
        "impervious_road_layer_count",
        "effective_snow_fraction",
        "surface_water",
        "snow_water",
    }
    layer_fields = {
        "layer_thickness",
        "node_depth",
        "soil_snow_temperature",
        "liquid_water",
        "ice_water",
    }
    ground_fields = {
        "soil_mineral_conductivity",
        "dry_soil_conductivity",
        "soil_solid_heat_capacity",
        "soil_porosity",
    }
    urban_fields = set(SoilThermalPropertyInputs._fields) - (
        vector_fields | layer_fields | ground_fields | {"interface_depth"}
    )

    expected_shapes = {
        **{name: vector_shape for name in vector_fields},
        **{name: layer_shape for name in layer_fields},
        **{name: ground_shape for name in ground_fields},
        **{name: urban_shape for name in urban_fields},
        "interface_depth": interface_shape,
    }
    for name, expected_shape in expected_shapes.items():
        if getattr(inputs, name).shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")


def _safe_fortran_divide(
    numerator: jnp.ndarray,
    denominator: jnp.ndarray,
) -> jnp.ndarray:

    safe_denominator = jnp.where(denominator != 0.0, denominator, jnp.float64(1.0))
    return fortran_divide(numerator, safe_denominator)


def _column_soil_thermal_properties(
    inputs: SoilThermalPropertyInputs,
    *, parallel_levels: bool = False,
) -> SoilThermalPropertyResult:

    active = jnp.asarray(inputs.active_nonlake, dtype=jnp.bool_)
    snow_layer_count = jnp.asarray(inputs.snow_layer_count, dtype=jnp.int32)
    landunit_type = jnp.asarray(inputs.landunit_type, dtype=jnp.int32)
    column_type = jnp.asarray(inputs.column_type, dtype=jnp.int32)
    bedrock_layer_count = jnp.asarray(inputs.bedrock_layer_count, dtype=jnp.int32)
    impervious_road_layer_count = jnp.asarray(
        inputs.impervious_road_layer_count,
        dtype=jnp.int32,
    )
    first_active_index = snow_layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS)
    enclosure = _is_urban_enclosure(column_type)

    conductivity = jnp.zeros((NUMBER_OF_SOIL_SNOW_LEVELS,), dtype=jnp.float64)
    snow_density = jnp.zeros((NUMBER_OF_SOIL_SNOW_LEVELS,), dtype=jnp.float64)

    def layer_conductivity(index):
        fortran_level = index + FIRST_FORTRAN_LEVEL
        ground_level = fortran_level - 1
        ground_index = jnp.clip(ground_level, 0, NUMBER_OF_GROUND_LEVELS - 1)
        urban_index = jnp.clip(ground_level, 0, NUMBER_OF_URBAN_LEVELS - 1)
        is_ground = fortran_level >= 1
        in_urban_material = fortran_level <= NUMBER_OF_URBAN_LEVELS

        is_wall = (column_type == URBAN_SUNWALL) | (column_type == URBAN_SHADEWALL)
        use_wall = is_wall & in_urban_material
        use_roof = (column_type == URBAN_ROOF) & in_urban_material
        use_impervious_road = (
            (column_type == URBAN_IMPERVIOUS_ROAD)
            & (fortran_level >= 1)
            & (fortran_level <= impervious_road_layer_count)
        )
        is_ordinary = (
            (landunit_type != LANDUNIT_WETLAND)
            & (landunit_type != LANDUNIT_ICE_MEC)
            & (~is_wall)
            & (column_type != URBAN_ROOF)
            & (~use_impervious_road)
        )

        thickness = inputs.layer_thickness[index]
        liquid = inputs.liquid_water[index]
        ice = inputs.ice_water[index]
        porosity = inputs.soil_porosity[ground_index]

        liquid_volume = _safe_fortran_divide(liquid, LIQUID_WATER_DENSITY)
        ice_volume = _safe_fortran_divide(ice, ICE_DENSITY)
        water_volume = fortran_add(liquid_volume, ice_volume)
        pore_volume = fortran_multiply(thickness, porosity)
        saturation = jnp.minimum(
            jnp.float64(1.0),
            _safe_fortran_divide(water_volume, pore_volume),
        )

        def moist_soil_conductivity(_: None) -> jnp.ndarray:
            safe_saturation = jnp.maximum(
                saturation,
                jnp.asarray(jnp.finfo(jnp.float64).tiny, dtype=jnp.float64),
            )
            unfrozen_kersten = jnp.maximum(
                jnp.float64(0.0),
                fortran_add(strict_log10(safe_saturation), jnp.float64(1.0)),
            )
            kersten = jnp.where(
                inputs.soil_snow_temperature[index] >= FREEZING_TEMPERATURE,
                unfrozen_kersten,
                saturation,
            )

            liquid_per_depth = _safe_fortran_divide(
                liquid,
                fortran_multiply(LIQUID_WATER_DENSITY, thickness),
            )
            ice_per_depth = _safe_fortran_divide(
                ice,
                fortran_multiply(ICE_DENSITY, thickness),
            )
            unfrozen_fraction = _safe_fortran_divide(
                liquid_per_depth,
                fortran_add(liquid_per_depth, ice_per_depth),
            )
            liquid_exponent = fortran_multiply(unfrozen_fraction, porosity)
            ice_exponent = fortran_multiply(
                fortran_add(jnp.float64(1.0), -unfrozen_fraction),
                porosity,
            )
            saturated_conductivity = fortran_multiply(
                fortran_multiply(
                    inputs.soil_mineral_conductivity[ground_index],
                    _material_power(LIQUID_WATER_THERMAL_CONDUCTIVITY, liquid_exponent),
                ),
                _material_power(ICE_THERMAL_CONDUCTIVITY, ice_exponent),
            )
            return fortran_add(
                fortran_multiply(kersten, saturated_conductivity),
                fortran_multiply(
                    fortran_add(jnp.float64(1.0), -kersten),
                    inputs.dry_soil_conductivity[ground_index],
                ),
            )

        ordinary_conductivity = lax.cond(
            saturation > jnp.float64(0.1e-6),
            moist_soil_conductivity,
            lambda _: inputs.dry_soil_conductivity[ground_index],
            operand=None,
        )
        ordinary_conductivity = jnp.where(
            fortran_level > bedrock_layer_count,
            jnp.float64(BEDROCK_THERMAL_CONDUCTIVITY),
            ordinary_conductivity,
        )
        water_or_ice_conductivity = jnp.where(
            inputs.soil_snow_temperature[index] < FREEZING_TEMPERATURE,
            jnp.float64(ICE_THERMAL_CONDUCTIVITY),
            jnp.float64(LIQUID_WATER_THERMAL_CONDUCTIVITY),
        )
        wetland_conductivity = jnp.where(
            fortran_level > 20,
            jnp.float64(BEDROCK_THERMAL_CONDUCTIVITY),
            water_or_ice_conductivity,
        )

        ground_conductivity = jnp.where(
            use_wall,
            inputs.urban_wall_conductivity[urban_index],
            jnp.where(
                use_roof,
                inputs.urban_roof_conductivity[urban_index],
                jnp.where(
                    use_impervious_road,
                    inputs.urban_impervious_road_conductivity[urban_index],
                    jnp.where(
                        is_ordinary,
                        ordinary_conductivity,
                        jnp.where(
                            landunit_type == LANDUNIT_ICE_MEC,
                            water_or_ice_conductivity,
                            jnp.where(
                                landunit_type == LANDUNIT_WETLAND,
                                wetland_conductivity,
                                jnp.float64(0.0),
                            ),
                        ),
                    ),
                ),
            ),
        )
        ground_defined = is_ground & (
            use_wall
            | use_roof
            | use_impervious_road
            | is_ordinary
            | (landunit_type == LANDUNIT_ICE_MEC)
            | (landunit_type == LANDUNIT_WETLAND)
        )
        selected_conductivity = jnp.where(ground_defined, ground_conductivity, 0.0)

        is_resolved_snow = (
            (snow_layer_count + 1 < 1)
            & (index >= first_active_index)
            & (fortran_level <= 0)
        )
        snow_mass = fortran_add(ice, liquid)
        snow_volume = fortran_multiply(inputs.effective_snow_fraction, thickness)
        layer_snow_density = _safe_fortran_divide(snow_mass, snow_volume)
        density_linear = fortran_multiply(jnp.float64(7.75e-5), layer_snow_density)
        density_quadratic = fortran_multiply(
            fortran_multiply(jnp.float64(1.105e-6), layer_snow_density),
            layer_snow_density,
        )
        snow_conductivity = fortran_add(
            jnp.float64(AIR_THERMAL_CONDUCTIVITY),
            fortran_multiply(
                fortran_add(density_linear, density_quadratic),
                fortran_add(
                    jnp.float64(ICE_THERMAL_CONDUCTIVITY),
                    -jnp.float64(AIR_THERMAL_CONDUCTIVITY),
                ),
            ),
        )
        selected_conductivity = jnp.where(
            is_resolved_snow,
            snow_conductivity,
            selected_conductivity,
        )
        selected_density = jnp.where(is_resolved_snow, layer_snow_density, 0.0)

        return (
            jnp.where(active, selected_conductivity, 0.0),
            jnp.where(active, selected_density, 0.0),
        )

    def set_layer_conductivity(index, values):
        heat, density = layer_conductivity(index)
        return values[0].at[index].set(heat), values[1].at[index].set(density)

    if parallel_levels:
        conductivity, snow_density = jax.vmap(layer_conductivity)(
            jnp.arange(NUMBER_OF_SOIL_SNOW_LEVELS),
        )
    else:
        conductivity, snow_density = lax.fori_loop(
            0, NUMBER_OF_SOIL_SNOW_LEVELS, set_layer_conductivity,
            (conductivity, snow_density),
        )

    interface_conductivity = jnp.zeros(
        (NUMBER_OF_SOIL_SNOW_LEVELS,),
        dtype=jnp.float64,
    )

    def interface_conductivity_at_level(index):
        fortran_level = index + FIRST_FORTRAN_LEVEL
        next_index = jnp.minimum(index + 1, GROUND_BOTTOM_INDEX)
        depth_difference = fortran_add(
            inputs.node_depth[next_index],
            -inputs.node_depth[index],
        )
        numerator = fortran_multiply(
            fortran_multiply(conductivity[index], conductivity[next_index]),
            depth_difference,
        )
        lower_node_to_interface = fortran_add(
            inputs.node_depth[next_index],
            -inputs.interface_depth[index + 1],
        )
        interface_to_upper_node = fortran_add(
            inputs.interface_depth[index + 1],
            -inputs.node_depth[index],
        )
        denominator = fortran_add(
            fortran_multiply(conductivity[index], lower_node_to_interface),
            fortran_multiply(conductivity[next_index], interface_to_upper_node),
        )
        harmonic = _safe_fortran_divide(numerator, denominator)

        enclosure_harmonic = (
            (index >= first_active_index)
            & (fortran_level <= NUMBER_OF_URBAN_LEVELS - 1)
        )
        enclosure_bottom = fortran_level == NUMBER_OF_URBAN_LEVELS
        ordinary_harmonic = (
            (index >= first_active_index)
            & (fortran_level <= NUMBER_OF_GROUND_LEVELS - 1)
        )
        selected = jnp.where(
            enclosure,
            jnp.where(
                enclosure_harmonic,
                harmonic,
                jnp.where(enclosure_bottom, conductivity[index], 0.0),
            ),
            jnp.where(ordinary_harmonic, harmonic, 0.0),
        )
        return jnp.where(active, selected, 0.0)

    if parallel_levels:
        interface_conductivity = jax.vmap(interface_conductivity_at_level)(
            jnp.arange(NUMBER_OF_SOIL_SNOW_LEVELS),
        )
    else:
        interface_conductivity = lax.fori_loop(
            0, NUMBER_OF_SOIL_SNOW_LEVELS,
            lambda index, current: current.at[index].set(interface_conductivity_at_level(index)),
            interface_conductivity,
        )

    ground_surface_index = NUMBER_OF_SNOW_LEVELS
    half_surface_water = fortran_multiply(jnp.float64(0.5), inputs.surface_water)
    surface_water_depth = fortran_multiply(
        jnp.float64(SURFACE_WATER_DEPTH_SCALE),
        half_surface_water,
    )
    surface_numerator = fortran_multiply(
        fortran_multiply(
            jnp.float64(LIQUID_WATER_THERMAL_CONDUCTIVITY),
            conductivity[ground_surface_index],
        ),
        fortran_add(inputs.node_depth[ground_surface_index], surface_water_depth),
    )
    surface_denominator = fortran_add(
        fortran_multiply(
            jnp.float64(LIQUID_WATER_THERMAL_CONDUCTIVITY),
            inputs.node_depth[ground_surface_index],
        ),
        fortran_multiply(conductivity[ground_surface_index], surface_water_depth),
    )
    surface_water_conductivity = jnp.where(
        active,
        _safe_fortran_divide(surface_numerator, surface_denominator),
        0.0,
    )

    heat_capacity = jnp.zeros((NUMBER_OF_SOIL_SNOW_LEVELS,), dtype=jnp.float64)

    def ground_heat_capacity(ground_index):
        index = ground_index + NUMBER_OF_SNOW_LEVELS
        fortran_level = ground_index + 1
        urban_index = jnp.minimum(ground_index, NUMBER_OF_URBAN_LEVELS - 1)
        in_urban_material = fortran_level <= NUMBER_OF_URBAN_LEVELS
        is_wall = (column_type == URBAN_SUNWALL) | (column_type == URBAN_SHADEWALL)
        use_wall = is_wall & in_urban_material
        use_roof = (column_type == URBAN_ROOF) & in_urban_material
        use_impervious_road = (
            (column_type == URBAN_IMPERVIOUS_ROAD)
            & (fortran_level <= impervious_road_layer_count)
        )
        is_ordinary = (
            (landunit_type != LANDUNIT_WETLAND)
            & (landunit_type != LANDUNIT_ICE_MEC)
            & (~is_wall)
            & (column_type != URBAN_ROOF)
            & (~use_impervious_road)
        )

        thickness = inputs.layer_thickness[index]
        ice_capacity = fortran_multiply(inputs.ice_water[index], ICE_SPECIFIC_HEAT)
        liquid_capacity = fortran_multiply(
            inputs.liquid_water[index],
            LIQUID_WATER_SPECIFIC_HEAT,
        )
        water_capacity = fortran_add(ice_capacity, liquid_capacity)
        solid_fraction = fortran_add(
            jnp.float64(1.0),
            -inputs.soil_porosity[ground_index],
        )
        solid_capacity = fortran_multiply(
            fortran_multiply(
                inputs.soil_solid_heat_capacity[ground_index],
                solid_fraction,
            ),
            thickness,
        )
        ordinary_capacity = fortran_add(solid_capacity, water_capacity)
        ordinary_capacity = jnp.where(
            fortran_level > bedrock_layer_count,
            fortran_multiply(BEDROCK_VOLUMETRIC_HEAT_CAPACITY, thickness),
            ordinary_capacity,
        )
        wetland_capacity = jnp.where(
            fortran_level > bedrock_layer_count,
            fortran_multiply(BEDROCK_VOLUMETRIC_HEAT_CAPACITY, thickness),
            water_capacity,
        )
        selected = jnp.where(
            use_wall,
            fortran_multiply(
                inputs.urban_wall_volumetric_heat_capacity[urban_index],
                thickness,
            ),
            jnp.where(
                use_roof,
                fortran_multiply(
                    inputs.urban_roof_volumetric_heat_capacity[urban_index],
                    thickness,
                ),
                jnp.where(
                    use_impervious_road,
                    fortran_multiply(
                        inputs.urban_impervious_road_volumetric_heat_capacity[urban_index],
                        thickness,
                    ),
                    jnp.where(
                        is_ordinary,
                        ordinary_capacity,
                        jnp.where(
                            landunit_type == LANDUNIT_WETLAND,
                            wetland_capacity,
                            jnp.where(
                                landunit_type == LANDUNIT_ICE_MEC,
                                water_capacity,
                                jnp.float64(0.0),
                            ),
                        ),
                    ),
                ),
            ),
        )
        defined = (
            use_wall
            | use_roof
            | use_impervious_road
            | is_ordinary
            | (landunit_type == LANDUNIT_WETLAND)
            | (landunit_type == LANDUNIT_ICE_MEC)
        )
        selected = jnp.where(defined, selected, 0.0)
        unresolved_snow = (
            (fortran_level == 1)
            & (snow_layer_count + 1 == 1)
            & (inputs.snow_water > 0.0)
        )
        selected = jnp.where(
            unresolved_snow,
            fortran_add(
                selected,
                fortran_multiply(ICE_SPECIFIC_HEAT, inputs.snow_water),
            ),
            selected,
        )
        return jnp.where(active, selected, 0.0)

    if parallel_levels:
        heat_capacity = heat_capacity.at[NUMBER_OF_SNOW_LEVELS:].set(
            jax.vmap(ground_heat_capacity)(jnp.arange(NUMBER_OF_GROUND_LEVELS)),
        )
    else:
        heat_capacity = lax.fori_loop(
            0, NUMBER_OF_GROUND_LEVELS,
            lambda index, current: current.at[index + NUMBER_OF_SNOW_LEVELS].set(
                ground_heat_capacity(index),
            ),
            heat_capacity,
        )

    def snow_heat_capacity(index):
        fortran_level = index + FIRST_FORTRAN_LEVEL
        is_resolved_snow = (
            (snow_layer_count + 1 < 1)
            & (index >= first_active_index)
            & (fortran_level <= 0)
        )
        liquid_capacity = fortran_multiply(
            LIQUID_WATER_SPECIFIC_HEAT,
            inputs.liquid_water[index],
        )
        ice_capacity = fortran_multiply(ICE_SPECIFIC_HEAT, inputs.ice_water[index])
        snow_capacity = jnp.maximum(
            jnp.float64(THIN_SURFACE_LAYER),
            _safe_fortran_divide(
                fortran_add(liquid_capacity, ice_capacity),
                inputs.effective_snow_fraction,
            ),
        )
        snow_capacity = jnp.where(
            inputs.effective_snow_fraction > 0.0,
            snow_capacity,
            jnp.float64(THIN_SURFACE_LAYER),
        )
        selected = jnp.where(is_resolved_snow, snow_capacity, jnp.float64(0.0))
        return jnp.where(active, selected, 0.0)

    if parallel_levels:
        heat_capacity = heat_capacity.at[:NUMBER_OF_SNOW_LEVELS].set(
            jax.vmap(snow_heat_capacity)(jnp.arange(NUMBER_OF_SNOW_LEVELS)),
        )
    else:
        heat_capacity = lax.fori_loop(
            0, NUMBER_OF_SNOW_LEVELS,
            lambda index, current: current.at[index].set(snow_heat_capacity(index)),
            heat_capacity,
        )

    return SoilThermalPropertyResult(
        layer_thermal_conductivity=conductivity,
        interface_thermal_conductivity=interface_conductivity,
        layer_heat_capacity=heat_capacity,
        snow_partial_density=snow_density,
        surface_water_interface_conductivity=surface_water_conductivity,
    )


def _soil_thermal_properties_reference(
    inputs: SoilThermalPropertyInputs,
) -> SoilThermalPropertyResult:

    _validate_thermal_property_shapes(inputs)
    return jax.vmap(lambda column: _column_soil_thermal_properties(
        column, parallel_levels=_parallel_thermal_levels(),
    ))(inputs)


@jax.custom_jvp
def soil_thermal_properties(inputs: SoilThermalPropertyInputs) -> SoilThermalPropertyResult:
    return _soil_thermal_properties_reference(inputs)


@soil_thermal_properties.defjvp
def _soil_thermal_properties_jvp(primals, tangents):
    def physical_storage(inputs):
        level = jnp.arange(NUMBER_OF_SOIL_SNOW_LEVELS)[None, :]
        used = inputs.active_nonlake[:, None] & (
            level >= inputs.snow_layer_count[:, None] + NUMBER_OF_SNOW_LEVELS
        )
        return _soil_thermal_properties_reference(inputs._replace(
            liquid_water=jnp.where(used, inputs.liquid_water, 0.),
            ice_water=jnp.where(used, inputs.ice_water, 0.),
            layer_thickness=jnp.where(used, inputs.layer_thickness, 1.),
            node_depth=jnp.where(used, inputs.node_depth, 0.),
            interface_depth=jnp.where(inputs.active_nonlake[:, None] & (
                jnp.arange(NUMBER_OF_SOIL_SNOW_LEVELS + 1)[None, :]
                >= inputs.snow_layer_count[:, None] + NUMBER_OF_SNOW_LEVELS
            ), inputs.interface_depth, 0.),
        ))

    return soil_thermal_properties(primals[0]), jax.jvp(physical_storage, primals, tangents)[1]


class GroundHeatFluxInputs(NamedTuple):

    snow_layer_count_column: jnp.ndarray
    downward_longwave_column: jnp.ndarray
    effective_snow_fraction_column: jnp.ndarray
    ground_emissivity_column: jnp.ndarray
    surface_water_temperature_column: jnp.ndarray
    ground_temperature_column: jnp.ndarray
    soil_snow_temperature_column: jnp.ndarray
    latent_heat_column: jnp.ndarray
    vegetation_exposed_patch: jnp.ndarray
    snow_evaporation_patch: jnp.ndarray
    soil_evaporation_patch: jnp.ndarray
    surface_water_evaporation_patch: jnp.ndarray
    total_soil_evaporation_patch: jnp.ndarray
    vegetation_transpiration_patch: jnp.ndarray
    ground_energy_derivative_patch: jnp.ndarray
    canopy_downward_longwave_patch: jnp.ndarray
    snow_sensible_heat_patch: jnp.ndarray
    soil_sensible_heat_patch: jnp.ndarray
    surface_water_sensible_heat_patch: jnp.ndarray
    ground_sensible_heat_patch: jnp.ndarray
    net_longwave_patch: jnp.ndarray
    absorbed_ground_patch: jnp.ndarray
    absorbed_soil_patch: jnp.ndarray
    absorbed_snow_patch: jnp.ndarray
    absorbed_ground_layers_patch: jnp.ndarray
    waste_heat_landunit: jnp.ndarray
    air_conditioner_heat_landunit: jnp.ndarray
    traffic_heat_landunit: jnp.ndarray
    urban_roof_fraction_landunit: jnp.ndarray


class GroundHeatFluxPatchState(NamedTuple):

    absorbed_ground_balance_patch: jnp.ndarray
    waste_heat_patch: jnp.ndarray
    air_conditioner_heat_patch: jnp.ndarray
    traffic_heat_patch: jnp.ndarray
    anthropogenic_heat_patch: jnp.ndarray
    net_ground_heat_patch: jnp.ndarray
    net_ground_heat_derivative_patch: jnp.ndarray


class GroundHeatFluxResult(NamedTuple):

    standing_water_heat_flux_column: jnp.ndarray
    top_snow_heat_flux_column: jnp.ndarray
    soil_heat_flux_column: jnp.ndarray
    top_layer_heat_flux_column: jnp.ndarray
    heat_flux_derivative_column: jnp.ndarray
    absorbed_ground_layers_column: jnp.ndarray
    patch_state: GroundHeatFluxPatchState


def _validate_ground_heat_flux_shapes(
    topology: SubgridTopology,
    inputs: GroundHeatFluxInputs,
    state: GroundHeatFluxPatchState,
) -> None:
    number_of_columns = topology.column.active.shape[0]
    number_of_patches = topology.patch.active.shape[0]
    number_of_landunits = topology.landunit.active.shape[0]
    column_shape = (number_of_columns,)
    patch_shape = (number_of_patches,)
    landunit_shape = (number_of_landunits,)

    topology_column_fields = (
        topology.column.landunit,
        topology.column.landunit_type,
        topology.column.column_type,
    )
    if any(field.shape != column_shape for field in topology_column_fields):
        raise ValueError("column topology fields must have one value per column")
    topology_patch_fields = (
        topology.patch.landunit,
        topology.patch.column,
        topology.patch.active,
        topology.patch.weight_on_column,
    )
    if any(field.shape != patch_shape for field in topology_patch_fields):
        raise ValueError("patch topology fields must have one value per patch")

    column_fields = {
        "snow_layer_count_column",
        "downward_longwave_column",
        "effective_snow_fraction_column",
        "ground_emissivity_column",
        "surface_water_temperature_column",
        "ground_temperature_column",
        "latent_heat_column",
    }
    landunit_fields = {
        "waste_heat_landunit",
        "air_conditioner_heat_landunit",
        "traffic_heat_landunit",
        "urban_roof_fraction_landunit",
    }
    layer_fields = {
        "soil_snow_temperature_column": (
            number_of_columns,
            NUMBER_OF_SOIL_SNOW_LEVELS,
        ),
        "absorbed_ground_layers_patch": (
            number_of_patches,
            NUMBER_OF_SNOW_LEVELS + 1,
        ),
    }
    for name in column_fields:
        if getattr(inputs, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")
    for name in landunit_fields:
        if getattr(inputs, name).shape != landunit_shape:
            raise ValueError(f"{name} must have shape {landunit_shape}")
    for name, expected_shape in layer_fields.items():
        if getattr(inputs, name).shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")

    patch_fields = set(GroundHeatFluxInputs._fields) - (
        column_fields | landunit_fields | set(layer_fields)
    )
    for name in patch_fields:
        if getattr(inputs, name).shape != patch_shape:
            raise ValueError(f"{name} must have shape {patch_shape}")
    for name in GroundHeatFluxPatchState._fields:
        if getattr(state, name).shape != patch_shape:
            raise ValueError(f"{name} must have shape {patch_shape}")


def _integer_power_three_and_four(value: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:

    squared = fortran_multiply(value, value)
    return (
        fortran_multiply(squared, value),
        fortran_multiply(squared, squared),
    )


def _contiguous_patch_to_column_sum(
    values: jnp.ndarray,
    topology: SubgridTopology,
    column_update_mask: jnp.ndarray,
    maximum_patches_per_column: int,
) -> jnp.ndarray:

    values = jnp.asarray(values, dtype=jnp.float64)
    patch_column = jnp.asarray(topology.patch.column, dtype=jnp.int32)
    patch_weight = jnp.asarray(topology.patch.weight_on_column, dtype=jnp.float64)
    patch_active = jnp.asarray(topology.patch.active, dtype=jnp.bool_)
    column_update_mask = jnp.asarray(column_update_mask, dtype=jnp.bool_)
    number_of_columns = column_update_mask.shape[0]
    number_of_patches = values.shape[0]
    result_shape = (number_of_columns, *values.shape[1:])
    result = jnp.zeros(result_shape, dtype=jnp.float64)
    if number_of_patches == 0:
        return result

    patch_index = jnp.arange(number_of_patches, dtype=jnp.int32)
    first_patch = jnp.full(number_of_columns, number_of_patches, dtype=jnp.int32)
    first_patch = first_patch.at[patch_column].min(patch_index)
    column_index = jnp.arange(number_of_columns, dtype=jnp.int32)
    trailing_dimensions = values.ndim - 1

    def accumulate(offset: int, column_values: jnp.ndarray) -> jnp.ndarray:
        unbounded_patch = first_patch + jnp.int32(offset)
        bounded_patch = jnp.clip(unbounded_patch, 0, number_of_patches - 1)
        belongs_to_column = (
            (unbounded_patch < number_of_patches)
            & (patch_column[bounded_patch] == column_index)
        )
        contributes = (
            belongs_to_column & patch_active[bounded_patch] & column_update_mask
        )
        selected_weight = patch_weight[bounded_patch].reshape(
            (number_of_columns, *([1] * trailing_dimensions))
        )
        contribution = fortran_multiply(values[bounded_patch], selected_weight)
        contribution = jnp.where(
            contributes.reshape((number_of_columns, *([1] * trailing_dimensions))),
            contribution,
            jnp.float64(0.0),
        )
        return fortran_add(column_values, contribution)

    return lax.fori_loop(
        0,
        maximum_patches_per_column,
        accumulate,
        result,
    )


def _rural_surface_flux(
    absorbed: jnp.ndarray,
    downward_longwave: jnp.ndarray,
    exposed_fraction: jnp.ndarray,
    emissivity: jnp.ndarray,
    atmospheric_longwave: jnp.ndarray,
    emitted_longwave: jnp.ndarray,
    sensible_heat: jnp.ndarray,
    evaporation: jnp.ndarray,
    latent_heat: jnp.ndarray,
) -> jnp.ndarray:
    intercepted_longwave = fortran_multiply(
        fortran_multiply(exposed_fraction, emissivity),
        atmospheric_longwave,
    )
    turbulent_loss = fortran_add(
        sensible_heat,
        fortran_multiply(evaporation, latent_heat),
    )
    return fortran_ordered_sum(
        jnp.stack(
            (
                absorbed,
                downward_longwave,
                intercepted_longwave,
                -emitted_longwave,
                -turbulent_loss,
            )
        )
    )


def compute_ground_heat_flux_and_derivative(
    topology: SubgridTopology,
    inputs: GroundHeatFluxInputs,
    state: GroundHeatFluxPatchState,
    *,
    maximum_patches_per_column: int = 79,
) -> GroundHeatFluxResult:

    _validate_ground_heat_flux_shapes(topology, inputs, state)
    if maximum_patches_per_column < 0:
        raise ValueError("maximum_patches_per_column must be nonnegative")

    column_active = jnp.asarray(topology.column.active, dtype=jnp.bool_)
    column_landunit_type = jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
    column_update = column_active & (column_landunit_type != LANDUNIT_LAKE)
    patch_column = jnp.asarray(topology.patch.column, dtype=jnp.int32)
    patch_landunit = jnp.asarray(topology.patch.landunit, dtype=jnp.int32)
    patch_active = jnp.asarray(topology.patch.active, dtype=jnp.bool_)
    patch_update = patch_active & column_update[patch_column]
    column_type = jnp.asarray(topology.column.column_type, dtype=jnp.int32)
    patch_column_type = column_type[patch_column]
    patch_is_road = (patch_column_type == URBAN_PERVIOUS_ROAD) | (
        patch_column_type == URBAN_IMPERVIOUS_ROAD
    )
    landunit_type = jnp.asarray(topology.landunit.landunit_type, dtype=jnp.int32)
    patch_is_urban = (landunit_type[patch_landunit] >= LANDUNIT_URBAN_MIN) & (
        landunit_type[patch_landunit] <= LANDUNIT_URBAN_MAX
    )

    snow_layer_count = jnp.asarray(inputs.snow_layer_count_column, dtype=jnp.int32)
    top_layer_index = snow_layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS)
    emissivity = jnp.asarray(inputs.ground_emissivity_column, dtype=jnp.float64)
    ground_temperature = jnp.asarray(inputs.ground_temperature_column, dtype=jnp.float64)
    soil_snow_temperature = jnp.asarray(
        inputs.soil_snow_temperature_column,
        dtype=jnp.float64,
    )
    surface_water_temperature = jnp.asarray(
        inputs.surface_water_temperature_column,
        dtype=jnp.float64,
    )
    column_index = jnp.arange(column_active.shape[0], dtype=jnp.int32)
    snow_surface_temperature = soil_snow_temperature[column_index, top_layer_index]
    soil_surface_temperature = soil_snow_temperature[:, NUMBER_OF_SNOW_LEVELS]

    ground_cube, ground_fourth = _integer_power_three_and_four(ground_temperature)
    _, snow_fourth = _integer_power_three_and_four(snow_surface_temperature)
    _, soil_fourth = _integer_power_three_and_four(soil_surface_temperature)
    _, water_fourth = _integer_power_three_and_four(surface_water_temperature)
    emissivity_sigma = fortran_multiply(emissivity, jnp.float64(STEFAN_BOLTZMANN))
    emitted_ground = fortran_multiply(emissivity_sigma, ground_fourth)
    emitted_derivative = fortran_multiply(
        fortran_multiply(jnp.float64(4.0), emissivity_sigma),
        ground_cube,
    )
    emitted_snow = fortran_multiply(emissivity_sigma, snow_fourth)
    emitted_soil = fortran_multiply(emissivity_sigma, soil_fourth)
    emitted_water = fortran_multiply(emissivity_sigma, water_fourth)

    atmospheric_longwave_patch = jnp.asarray(
        inputs.downward_longwave_column,
        dtype=jnp.float64,
    )[patch_column]
    emissivity_patch = emissivity[patch_column]
    latent_heat_patch = jnp.asarray(inputs.latent_heat_column, dtype=jnp.float64)[
        patch_column
    ]
    exposed_fraction_patch = jnp.asarray(
        jnp.int32(1)
        - jnp.asarray(inputs.vegetation_exposed_patch, dtype=jnp.int32),
        dtype=jnp.float64,
    )
    canopy_longwave = jnp.asarray(
        inputs.canopy_downward_longwave_patch,
        dtype=jnp.float64,
    )

    rural_ground = _rural_surface_flux(
        jnp.asarray(inputs.absorbed_ground_patch, dtype=jnp.float64),
        canopy_longwave,
        exposed_fraction_patch,
        emissivity_patch,
        atmospheric_longwave_patch,
        emitted_ground[patch_column],
        jnp.asarray(inputs.ground_sensible_heat_patch, dtype=jnp.float64),
        jnp.asarray(inputs.total_soil_evaporation_patch, dtype=jnp.float64),
        latent_heat_patch,
    )
    rural_soil = _rural_surface_flux(
        jnp.asarray(inputs.absorbed_soil_patch, dtype=jnp.float64),
        canopy_longwave,
        exposed_fraction_patch,
        emissivity_patch,
        atmospheric_longwave_patch,
        emitted_soil[patch_column],
        jnp.asarray(inputs.soil_sensible_heat_patch, dtype=jnp.float64),
        jnp.asarray(inputs.soil_evaporation_patch, dtype=jnp.float64),
        latent_heat_patch,
    )
    rural_water = _rural_surface_flux(
        jnp.asarray(inputs.absorbed_soil_patch, dtype=jnp.float64),
        canopy_longwave,
        exposed_fraction_patch,
        emissivity_patch,
        atmospheric_longwave_patch,
        emitted_water[patch_column],
        jnp.asarray(inputs.surface_water_sensible_heat_patch, dtype=jnp.float64),
        jnp.asarray(inputs.surface_water_evaporation_patch, dtype=jnp.float64),
        latent_heat_patch,
    )

    roof_fraction = jnp.asarray(
        inputs.urban_roof_fraction_landunit,
        dtype=jnp.float64,
    )[patch_landunit]
    canyon_fraction = fortran_add(jnp.float64(1.0), -roof_fraction)
    safe_canyon_fraction = jnp.where(
        patch_is_urban & patch_is_road,
        canyon_fraction,
        jnp.float64(1.0),
    )
    urban_waste = fortran_divide(
        jnp.asarray(inputs.waste_heat_landunit, dtype=jnp.float64)[patch_landunit],
        safe_canyon_fraction,
    )
    urban_air_conditioner = fortran_divide(
        jnp.asarray(inputs.air_conditioner_heat_landunit, dtype=jnp.float64)[
            patch_landunit
        ],
        safe_canyon_fraction,
    )
    urban_traffic = fortran_divide(
        jnp.asarray(inputs.traffic_heat_landunit, dtype=jnp.float64)[patch_landunit],
        safe_canyon_fraction,
    )
    urban_waste = jnp.where(patch_is_road, urban_waste, jnp.float64(0.0))
    urban_air_conditioner = jnp.where(
        patch_is_road,
        urban_air_conditioner,
        jnp.float64(0.0),
    )
    urban_traffic = jnp.where(patch_is_road, urban_traffic, jnp.float64(0.0))
    urban_turbulent_loss = fortran_ordered_sum(
        jnp.stack(
            (
                jnp.asarray(inputs.ground_sensible_heat_patch, dtype=jnp.float64),
                fortran_multiply(
                    jnp.asarray(inputs.total_soil_evaporation_patch, dtype=jnp.float64),
                    latent_heat_patch,
                ),
                fortran_multiply(
                    jnp.asarray(inputs.vegetation_transpiration_patch, dtype=jnp.float64),
                    jnp.float64(LATENT_HEAT_VAPORIZATION),
                ),
            )
        )
    )
    urban_ground = fortran_ordered_sum(
        jnp.stack(
            (
                jnp.asarray(inputs.absorbed_ground_patch, dtype=jnp.float64),
                canopy_longwave,
                -jnp.asarray(inputs.net_longwave_patch, dtype=jnp.float64),
                -urban_turbulent_loss,
                urban_waste,
                urban_air_conditioner,
                urban_traffic,
            )
        )
    )

    ground_flux = jnp.where(patch_is_urban, urban_ground, rural_ground)
    soil_flux = jnp.where(patch_is_urban, urban_ground, rural_soil)
    water_flux = jnp.where(patch_is_urban, urban_ground, rural_water)
    derivative = fortran_add(
        -jnp.asarray(inputs.ground_energy_derivative_patch, dtype=jnp.float64),
        -emitted_derivative[patch_column],
    )

    snow_fraction_patch = jnp.asarray(
        inputs.effective_snow_fraction_column,
        dtype=jnp.float64,
    )[patch_column]
    ground_balance = fortran_add(
        fortran_multiply(
            snow_fraction_patch,
            jnp.asarray(inputs.absorbed_snow_patch, dtype=jnp.float64),
        ),
        fortran_multiply(
            fortran_add(jnp.float64(1.0), -snow_fraction_patch),
            jnp.asarray(inputs.absorbed_soil_patch, dtype=jnp.float64),
        ),
    )

    updated_state = GroundHeatFluxPatchState(
        absorbed_ground_balance_patch=jnp.where(
            patch_update & ~patch_is_urban,
            ground_balance,
            jnp.asarray(state.absorbed_ground_balance_patch, dtype=jnp.float64),
        ),
        waste_heat_patch=jnp.where(
            patch_update & patch_is_urban,
            urban_waste,
            jnp.asarray(state.waste_heat_patch, dtype=jnp.float64),
        ),
        air_conditioner_heat_patch=jnp.where(
            patch_update & patch_is_urban,
            urban_air_conditioner,
            jnp.asarray(state.air_conditioner_heat_patch, dtype=jnp.float64),
        ),
        traffic_heat_patch=jnp.where(
            patch_update & patch_is_urban,
            urban_traffic,
            jnp.asarray(state.traffic_heat_patch, dtype=jnp.float64),
        ),
        anthropogenic_heat_patch=jnp.asarray(
            state.anthropogenic_heat_patch,
            dtype=jnp.float64,
        ),
        net_ground_heat_patch=jnp.where(
            patch_update,
            ground_flux,
            jnp.asarray(state.net_ground_heat_patch, dtype=jnp.float64),
        ),
        net_ground_heat_derivative_patch=jnp.where(
            patch_update,
            derivative,
            jnp.asarray(state.net_ground_heat_derivative_patch, dtype=jnp.float64),
        ),
    )

    absorbed_layers = jnp.asarray(inputs.absorbed_ground_layers_patch, dtype=jnp.float64)
    layer_index = jnp.arange(NUMBER_OF_SNOW_LEVELS + 1, dtype=jnp.int32)[None, :]
    patch_top_index = top_layer_index[patch_column, None]
    rural_layers = jnp.where(
        layer_index >= patch_top_index,
        absorbed_layers,
        jnp.float64(0.0),
    )
    urban_layers = jnp.where(
        layer_index == patch_top_index,
        jnp.asarray(inputs.absorbed_ground_patch, dtype=jnp.float64)[:, None],
        jnp.float64(0.0),
    )
    patch_layers = jnp.where(patch_is_urban[:, None], urban_layers, rural_layers)

    top_absorbed = absorbed_layers[
        jnp.arange(patch_active.shape[0], dtype=jnp.int32),
        top_layer_index[patch_column],
    ]
    rural_top = _rural_surface_flux(
        top_absorbed,
        canopy_longwave,
        exposed_fraction_patch,
        emissivity_patch,
        atmospheric_longwave_patch,
        emitted_ground[patch_column],
        jnp.asarray(inputs.ground_sensible_heat_patch, dtype=jnp.float64),
        jnp.asarray(inputs.total_soil_evaporation_patch, dtype=jnp.float64),
        latent_heat_patch,
    )
    rural_top_snow = _rural_surface_flux(
        top_absorbed,
        canopy_longwave,
        exposed_fraction_patch,
        emissivity_patch,
        atmospheric_longwave_patch,
        emitted_snow[patch_column],
        jnp.asarray(inputs.snow_sensible_heat_patch, dtype=jnp.float64),
        jnp.asarray(inputs.snow_evaporation_patch, dtype=jnp.float64),
        latent_heat_patch,
    )
    top_flux = jnp.where(patch_is_urban, ground_flux, rural_top)
    top_snow_flux = jnp.where(patch_is_urban, ground_flux, rural_top_snow)

    def aggregate(values: jnp.ndarray) -> jnp.ndarray:
        return _contiguous_patch_to_column_sum(
            values,
            topology,
            column_update,
            maximum_patches_per_column,
        )

    return GroundHeatFluxResult(
        standing_water_heat_flux_column=aggregate(water_flux),
        top_snow_heat_flux_column=aggregate(top_snow_flux),
        soil_heat_flux_column=aggregate(soil_flux),
        top_layer_heat_flux_column=aggregate(top_flux),
        heat_flux_derivative_column=aggregate(derivative),
        absorbed_ground_layers_column=aggregate(patch_layers),
        patch_state=updated_state,
    )


class SoilHeatDiffusionInputs(NamedTuple):

    active_nonlake: jnp.ndarray
    snow_layer_count: jnp.ndarray
    column_type: jnp.ndarray
    layer_thickness: jnp.ndarray
    node_depth: jnp.ndarray
    interface_depth: jnp.ndarray
    interface_thermal_conductivity: jnp.ndarray
    layer_heat_capacity: jnp.ndarray
    soil_snow_temperature: jnp.ndarray
    bottom_heat_flux: jnp.ndarray
    building_temperature: jnp.ndarray
    roof_inner_temperature: jnp.ndarray
    sunwall_inner_temperature: jnp.ndarray
    shadewall_inner_temperature: jnp.ndarray


class SoilHeatDiffusionResult(NamedTuple):

    interface_heat_flux: jnp.ndarray
    timestep_over_heat_capacity: jnp.ndarray


def _validate_heat_diffusion_shapes(inputs: SoilHeatDiffusionInputs) -> None:
    if inputs.active_nonlake.ndim != 1:
        raise ValueError("active_nonlake must be a vector")

    number_of_columns = inputs.active_nonlake.shape[0]
    vector_shape = (number_of_columns,)
    layer_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS)
    interface_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS + 1)

    vector_fields = {
        "active_nonlake",
        "snow_layer_count",
        "column_type",
        "bottom_heat_flux",
        "building_temperature",
        "roof_inner_temperature",
        "sunwall_inner_temperature",
        "shadewall_inner_temperature",
    }
    for name in vector_fields:
        if getattr(inputs, name).shape != vector_shape:
            raise ValueError(f"{name} must have shape {vector_shape}")

    if inputs.interface_depth.shape != interface_shape:
        raise ValueError(f"interface_depth must have shape {interface_shape}")

    for name in set(SoilHeatDiffusionInputs._fields) - vector_fields - {"interface_depth"}:
        if getattr(inputs, name).shape != layer_shape:
            raise ValueError(f"{name} must have shape {layer_shape}")


def _is_urban_enclosure(column_type: jnp.ndarray) -> jnp.ndarray:
    return (
        (column_type == URBAN_ROOF)
        | (column_type == URBAN_SUNWALL)
        | (column_type == URBAN_SHADEWALL)
    )


def _column_heat_diffusion_flux_and_factor(
    inputs: SoilHeatDiffusionInputs,
    time_step_seconds: jnp.ndarray,
    *, parallel_levels: bool = False,
) -> SoilHeatDiffusionResult:

    active = jnp.asarray(inputs.active_nonlake, dtype=jnp.bool_)
    snow_layer_count = jnp.asarray(inputs.snow_layer_count, dtype=jnp.int32)
    column_type = jnp.asarray(inputs.column_type, dtype=jnp.int32)
    top_index = snow_layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS)
    urban_enclosure = _is_urban_enclosure(column_type)
    bottom_index = jnp.where(urban_enclosure, URBAN_BOTTOM_INDEX, GROUND_BOTTOM_INDEX)

    flux = jnp.zeros((NUMBER_OF_SOIL_SNOW_LEVELS,), dtype=jnp.float64)
    factor = jnp.zeros((NUMBER_OF_SOIL_SNOW_LEVELS,), dtype=jnp.float64)

    def evaluate_level(index):
        valid = active & (index >= top_index) & (index <= bottom_index)
        at_top = index == top_index
        at_bottom = index == bottom_index

        capacity_factor = fortran_divide(
            time_step_seconds,
            jnp.where(valid, inputs.layer_heat_capacity[index], jnp.float64(1.0)),
        )

        def neighboring_flux(_: None) -> jnp.ndarray:
            next_index = jnp.minimum(index + 1, bottom_index)
            temperature_difference = fortran_add(
                inputs.soil_snow_temperature[next_index],
                -inputs.soil_snow_temperature[index],
            )
            depth_difference = fortran_add(
                inputs.node_depth[next_index],
                -inputs.node_depth[index],
            )
            return fortran_divide(
                fortran_multiply(
                    inputs.interface_thermal_conductivity[index],
                    temperature_difference,
                ),
                jnp.where(valid & (~at_bottom), depth_difference, jnp.float64(1.0)),
            )

        def urban_bottom_flux(_: None) -> jnp.ndarray:
            inner_temperature = jnp.where(
                column_type == URBAN_SUNWALL,
                inputs.sunwall_inner_temperature,
                jnp.where(
                    column_type == URBAN_SHADEWALL,
                    inputs.shadewall_inner_temperature,
                    inputs.roof_inner_temperature,
                ),
            )
            temperature_difference = fortran_add(
                inner_temperature,
                -fortran_multiply(
                    jnp.float64(CRANK_NICHOLSON_FACTOR),
                    inputs.soil_snow_temperature[index],
                ),
            )
            depth_difference = fortran_add(
                inputs.interface_depth[index + 1],
                -inputs.node_depth[index],
            )
            return fortran_divide(
                fortran_multiply(
                    inputs.interface_thermal_conductivity[index],
                    temperature_difference,
                ),
                jnp.where(valid & urban_enclosure & at_bottom,
                          depth_difference, jnp.float64(1.0)),
            )

        def nonurban_top_factor(_: None) -> jnp.ndarray:
            top_interval = fortran_add(
                inputs.node_depth[index],
                -inputs.interface_depth[index],
            )
            extended_interval = fortran_add(
                inputs.node_depth[index + 1],
                -inputs.interface_depth[index],
            )
            effective_depth = fortran_multiply(
                jnp.float64(0.5),
                fortran_add(
                    top_interval,
                    fortran_multiply(
                        jnp.float64(SURFACE_TEMPERATURE_TUNING_FACTOR),
                        extended_interval,
                    ),
                ),
            )
            return fortran_divide(
                fortran_multiply(capacity_factor, inputs.layer_thickness[index]),
                jnp.where(valid & (~urban_enclosure) & at_top,
                          effective_depth, jnp.float64(1.0)),
            )

        selected_factor = lax.cond(
            (~urban_enclosure) & at_top,
            nonurban_top_factor,
            lambda _: capacity_factor,
            operand=None,
        )
        selected_flux = lax.cond(
            urban_enclosure & at_bottom,
            urban_bottom_flux,
            lambda _: lax.cond(
                (~urban_enclosure) & at_bottom,
                lambda __: jnp.asarray(inputs.bottom_heat_flux, dtype=jnp.float64),
                neighboring_flux,
                operand=None,
            ),
            operand=None,
        )

        return jnp.where(valid, selected_flux, 0.0), jnp.where(valid, selected_factor, 0.0)

    def set_level(index, values):
        heat, capacity = evaluate_level(index)
        return values[0].at[index].set(heat), values[1].at[index].set(capacity)

    if parallel_levels:
        flux, factor = jax.vmap(evaluate_level)(jnp.arange(NUMBER_OF_SOIL_SNOW_LEVELS))
    else:
        flux, factor = lax.fori_loop(
            0, NUMBER_OF_SOIL_SNOW_LEVELS, set_level, (flux, factor),
        )
    return SoilHeatDiffusionResult(
        interface_heat_flux=flux,
        timestep_over_heat_capacity=factor,
    )


def compute_heat_diffusion_flux_and_factor(
    inputs: SoilHeatDiffusionInputs,
    *,
    time_step_seconds: float = 1800.0,
) -> SoilHeatDiffusionResult:

    _validate_heat_diffusion_shapes(inputs)
    if time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")

    return jax.vmap(
        lambda column_inputs: _column_heat_diffusion_flux_and_factor(
            column_inputs,
            jnp.float64(time_step_seconds),
            parallel_levels=_parallel_thermal_levels(),
        )
    )(inputs)


class SoilTemperatureRHSInputs(NamedTuple):

    active_nonlake: jnp.ndarray
    snow_layer_count: jnp.ndarray
    landunit_type: jnp.ndarray
    column_type: jnp.ndarray
    standing_water_heat_flux: jnp.ndarray
    top_snow_heat_flux: jnp.ndarray
    soil_heat_flux: jnp.ndarray
    top_layer_heat_flux: jnp.ndarray
    heat_flux_derivative: jnp.ndarray
    absorbed_ground_layers: jnp.ndarray
    layer_thermal_conductivity: jnp.ndarray
    standing_water_thermal_conductivity: jnp.ndarray
    timestep_over_heat_capacity: jnp.ndarray
    interface_heat_flux: jnp.ndarray
    standing_water_heat_capacity: jnp.ndarray
    standing_water_thickness: jnp.ndarray
    node_depth: jnp.ndarray
    soil_snow_temperature: jnp.ndarray
    standing_water_temperature: jnp.ndarray
    standing_water_fraction: jnp.ndarray
    effective_snow_fraction: jnp.ndarray


class SoilTemperatureRHSResult(NamedTuple):

    right_hand_side: jnp.ndarray
    standing_water_interface_heat_flux: jnp.ndarray


def _validate_soil_temperature_rhs_shapes(inputs: SoilTemperatureRHSInputs) -> None:
    if inputs.active_nonlake.ndim != 1:
        raise ValueError("active_nonlake must be a vector")
    number_of_columns = inputs.active_nonlake.shape[0]
    vector_shape = (number_of_columns,)
    layer_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS)
    radiation_shape = (number_of_columns, NUMBER_OF_SNOW_LEVELS + 1)
    layer_fields = {
        "layer_thermal_conductivity",
        "timestep_over_heat_capacity",
        "interface_heat_flux",
        "node_depth",
        "soil_snow_temperature",
    }
    radiation_fields = {"absorbed_ground_layers"}
    for name in set(SoilTemperatureRHSInputs._fields) - layer_fields - radiation_fields:
        if getattr(inputs, name).shape != vector_shape:
            raise ValueError(f"{name} must have shape {vector_shape}")
    for name in layer_fields:
        if getattr(inputs, name).shape != layer_shape:
            raise ValueError(f"{name} must have shape {layer_shape}")
    for name in radiation_fields:
        if getattr(inputs, name).shape != radiation_shape:
            raise ValueError(f"{name} must have shape {radiation_shape}")


def _rhs_surface_entry(
    temperature: jnp.ndarray,
    factor: jnp.ndarray,
    surface_flux: jnp.ndarray,
    flux_derivative: jnp.ndarray,
    interface_flux: jnp.ndarray,
) -> jnp.ndarray:
    forcing = fortran_ordered_sum(
        jnp.stack(
            (
                surface_flux,
                -fortran_multiply(flux_derivative, temperature),
                fortran_multiply(
                    jnp.float64(CRANK_NICHOLSON_FACTOR),
                    interface_flux,
                ),
            )
        )
    )
    return fortran_add(temperature, fortran_multiply(factor, forcing))


def _rhs_internal_entry(
    temperature: jnp.ndarray,
    factor: jnp.ndarray,
    lower_interface_flux: jnp.ndarray,
    upper_interface_flux: jnp.ndarray,
) -> jnp.ndarray:
    flux_difference = fortran_add(lower_interface_flux, -upper_interface_flux)
    coefficient = fortran_multiply(
        jnp.float64(CRANK_NICHOLSON_FACTOR),
        factor,
    )
    return fortran_add(
        temperature,
        fortran_multiply(coefficient, flux_difference),
    )


def _column_soil_temperature_rhs(
    inputs: SoilTemperatureRHSInputs,
    time_step_seconds: jnp.ndarray,
) -> SoilTemperatureRHSResult:
    active = jnp.asarray(inputs.active_nonlake, dtype=jnp.bool_)
    snow_layer_count = jnp.asarray(inputs.snow_layer_count, dtype=jnp.int32)
    landunit_type = jnp.asarray(inputs.landunit_type, dtype=jnp.int32)
    column_type = jnp.asarray(inputs.column_type, dtype=jnp.int32)
    urban = (landunit_type >= LANDUNIT_URBAN_MIN) & (landunit_type <= LANDUNIT_URBAN_MAX)
    urban_enclosure = urban & _is_urban_enclosure(column_type)
    urban_road = urban & (
        (column_type == URBAN_IMPERVIOUS_ROAD) | (column_type == URBAN_PERVIOUS_ROAD)
    )
    first_active_index = snow_layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS)

    temperature = jnp.asarray(inputs.soil_snow_temperature, dtype=jnp.float64)
    factor = jnp.asarray(inputs.timestep_over_heat_capacity, dtype=jnp.float64)
    interface_flux = jnp.asarray(inputs.interface_heat_flux, dtype=jnp.float64)
    absorbed_layers = jnp.asarray(inputs.absorbed_ground_layers, dtype=jnp.float64)
    right_hand_side = jnp.zeros(
        (NUMBER_OF_SOIL_SNOW_LEVELS + 1,),
        dtype=jnp.float64,
    )

    def snow_entry(index: int) -> jnp.ndarray:
        resolved_snow = active & (index >= first_active_index) & (
            index < NUMBER_OF_SNOW_LEVELS
        )
        at_surface = index == first_active_index
        previous_index = jnp.maximum(index - 1, 0)
        surface_flux = jnp.where(
            urban_enclosure,
            inputs.top_layer_heat_flux,
            inputs.top_snow_heat_flux,
        )
        surface_value = _rhs_surface_entry(
            temperature[index],
            factor[index],
            surface_flux,
            inputs.heat_flux_derivative,
            interface_flux[index],
        )
        internal_value = _rhs_internal_entry(
            temperature[index],
            factor[index],
            interface_flux[index],
            interface_flux[previous_index],
        )
        internal_value = fortran_add(
            internal_value,
            fortran_multiply(factor[index], absorbed_layers[index]),
        )
        selected = jnp.where(at_surface, surface_value, internal_value)
        return jnp.where(resolved_snow, selected, jnp.float64(0.0))

    if _parallel_system_assembly():
        snow_values = jax.vmap(snow_entry)(jnp.arange(NUMBER_OF_SNOW_LEVELS))
        right_hand_side = right_hand_side.at[:NUMBER_OF_SNOW_LEVELS].set(snow_values)
    else:
        right_hand_side = lax.fori_loop(
            0, NUMBER_OF_SNOW_LEVELS,
            lambda index, current: current.at[index].set(snow_entry(index)),
            right_hand_side,
        )

    soil_surface_index = NUMBER_OF_SNOW_LEVELS
    water_depth_interval = fortran_add(
        fortran_multiply(
            jnp.float64(0.5),
            inputs.standing_water_thickness,
        ),
        inputs.node_depth[soil_surface_index],
    )
    safe_water_depth_interval = jnp.where(
        active,
        water_depth_interval,
        jnp.float64(1.0),
    )
    water_interface_flux = fortran_divide(
        fortran_multiply(
            inputs.standing_water_thermal_conductivity,
            fortran_add(
                temperature[soil_surface_index],
                -inputs.standing_water_temperature,
            ),
        ),
        safe_water_depth_interval,
    )
    safe_water_capacity = jnp.where(
        active,
        inputs.standing_water_heat_capacity,
        jnp.float64(1.0),
    )
    water_factor = fortran_divide(time_step_seconds, safe_water_capacity)
    water_forcing = fortran_ordered_sum(
        jnp.stack(
            (
                inputs.standing_water_heat_flux,
                -fortran_multiply(
                    inputs.heat_flux_derivative,
                    inputs.standing_water_temperature,
                ),
                fortran_multiply(
                    jnp.float64(CRANK_NICHOLSON_FACTOR),
                    water_interface_flux,
                ),
            )
        )
    )
    water_rhs = fortran_add(
        inputs.standing_water_temperature,
        fortran_multiply(water_factor, water_forcing),
    )
    right_hand_side = right_hand_side.at[NUMBER_OF_SNOW_LEVELS].set(
        jnp.where(active, water_rhs, jnp.float64(0.0))
    )

    snow_present = snow_layer_count < 0

    def ground_entry(ground_index: int) -> jnp.ndarray:
        layer_index = NUMBER_OF_SNOW_LEVELS + ground_index
        fortran_level = ground_index + 1
        enclosure_valid = fortran_level <= NUMBER_OF_URBAN_LEVELS
        valid = active & jnp.where(urban_enclosure, enclosure_valid, True)
        at_surface = (fortran_level == 1) & ~snow_present
        previous_index = layer_index - 1

        enclosure_surface = _rhs_surface_entry(
            temperature[layer_index],
            factor[layer_index],
            inputs.top_layer_heat_flux,
            inputs.heat_flux_derivative,
            interface_flux[layer_index],
        )
        ordinary_surface = _rhs_surface_entry(
            temperature[layer_index],
            factor[layer_index],
            inputs.top_snow_heat_flux,
            inputs.heat_flux_derivative,
            interface_flux[layer_index],
        )
        exposed_soil_forcing = fortran_multiply(
            fortran_add(jnp.float64(1.0), -inputs.effective_snow_fraction),
            fortran_add(
                inputs.soil_heat_flux,
                -fortran_multiply(
                    inputs.heat_flux_derivative,
                    temperature[layer_index],
                ),
            ),
        )
        weighted_diffusion = fortran_multiply(
            jnp.float64(CRANK_NICHOLSON_FACTOR),
            fortran_add(
                interface_flux[layer_index],
                -fortran_multiply(
                    inputs.effective_snow_fraction,
                    interface_flux[previous_index],
                ),
            ),
        )
        ordinary_first = fortran_add(
            temperature[layer_index],
            fortran_multiply(
                factor[layer_index],
                fortran_add(exposed_soil_forcing, weighted_diffusion),
            ),
        )
        ordinary_first = fortran_add(
            ordinary_first,
            fortran_multiply(
                fortran_multiply(
                    inputs.effective_snow_fraction,
                    factor[layer_index],
                ),
                absorbed_layers[NUMBER_OF_SNOW_LEVELS],
            ),
        )

        internal = _rhs_internal_entry(
            temperature[layer_index],
            factor[layer_index],
            interface_flux[layer_index],
            interface_flux[previous_index],
        )
        enclosure_first = fortran_add(
            internal,
            fortran_multiply(
                factor[layer_index],
                absorbed_layers[NUMBER_OF_SNOW_LEVELS],
            ),
        )
        enclosure_bottom = fortran_add(
            temperature[layer_index],
            fortran_multiply(
                factor[layer_index],
                fortran_add(
                    interface_flux[layer_index],
                    -fortran_multiply(
                        jnp.float64(CRANK_NICHOLSON_FACTOR),
                        interface_flux[previous_index],
                    ),
                ),
            ),
        )
        ordinary_bottom = fortran_ordered_sum(
            jnp.stack(
                (
                    temperature[layer_index],
                    -fortran_multiply(
                        fortran_multiply(
                            jnp.float64(CRANK_NICHOLSON_FACTOR),
                            factor[layer_index],
                        ),
                        interface_flux[previous_index],
                    ),
                    fortran_multiply(
                        factor[layer_index],
                        interface_flux[layer_index],
                    ),
                )
            )
        )

        enclosure_value = jnp.where(
            at_surface,
            enclosure_surface,
            jnp.where(
                fortran_level == 1,
                enclosure_first,
                jnp.where(
                    fortran_level == NUMBER_OF_URBAN_LEVELS,
                    enclosure_bottom,
                    internal,
                ),
            ),
        )
        ordinary_value = jnp.where(
            at_surface,
            ordinary_surface,
            jnp.where(
                fortran_level == 1,
                ordinary_first,
                jnp.where(
                    fortran_level == NUMBER_OF_GROUND_LEVELS,
                    ordinary_bottom,
                    internal,
                ),
            ),
        )
        selected = jnp.where(
            urban_enclosure,
            enclosure_value,
            ordinary_value,
        )
        return jnp.where(valid, selected, jnp.float64(0.0))

    if _parallel_system_assembly():
        ground_values = jax.vmap(ground_entry)(jnp.arange(NUMBER_OF_GROUND_LEVELS))
        right_hand_side = right_hand_side.at[NUMBER_OF_SNOW_LEVELS + 1:].set(ground_values)
    else:
        right_hand_side = lax.fori_loop(
            0, NUMBER_OF_GROUND_LEVELS,
            lambda index, current: current.at[NUMBER_OF_SNOW_LEVELS + 1 + index].set(
                ground_entry(index),
            ),
            right_hand_side,
        )

    first_soil_output_index = NUMBER_OF_SNOW_LEVELS + 1
    soil_water_coupling = fortran_multiply(
        fortran_multiply(
            inputs.standing_water_fraction,
            factor[soil_surface_index],
        ),
        fortran_add(
            fortran_add(
                inputs.soil_heat_flux,
                -fortran_multiply(
                    inputs.heat_flux_derivative,
                    temperature[soil_surface_index],
                ),
            ),
            fortran_multiply(
                jnp.float64(CRANK_NICHOLSON_FACTOR),
                water_interface_flux,
            ),
        ),
    )
    coupled_soil_rhs = fortran_add(
        right_hand_side[first_soil_output_index],
        -soil_water_coupling,
    )
    right_hand_side = right_hand_side.at[first_soil_output_index].set(
        jnp.where(
            active & (inputs.standing_water_fraction != 0.0),
            coupled_soil_rhs,
            right_hand_side[first_soil_output_index],
        )
    )

    supported = (~urban) | urban_enclosure | urban_road
    right_hand_side = jnp.where(
        active & supported,
        right_hand_side,
        jnp.zeros_like(right_hand_side),
    )
    return SoilTemperatureRHSResult(
        right_hand_side=right_hand_side,
        standing_water_interface_heat_flux=jnp.where(
            active & supported,
            water_interface_flux,
            jnp.float64(0.0),
        ),
    )


def _set_soil_temperature_rhs_reference(
    inputs: SoilTemperatureRHSInputs,
    *,
    time_step_seconds: float = 1800.0,
) -> SoilTemperatureRHSResult:

    _validate_soil_temperature_rhs_shapes(inputs)
    if time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")
    return jax.vmap(
        lambda column_inputs: _column_soil_temperature_rhs(
            column_inputs,
            jnp.float64(time_step_seconds),
        )
    )(inputs)


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _rhs_with_storage_adjoint(inputs, time_step_seconds):
    return _set_soil_temperature_rhs_reference(inputs, time_step_seconds=time_step_seconds)


@_rhs_with_storage_adjoint.defjvp
def _rhs_storage_jvp(time_step_seconds, primals, tangents):
    def physical_storage(inputs):
        level = jnp.arange(NUMBER_OF_SOIL_SNOW_LEVELS)[None, :]
        bottom = jnp.where(_is_urban_enclosure(inputs.column_type),
                           URBAN_BOTTOM_INDEX, GROUND_BOTTOM_INDEX)
        used = inputs.active_nonlake[:, None] & (
            level >= inputs.snow_layer_count[:, None] + NUMBER_OF_SNOW_LEVELS
        ) & (level <= bottom[:, None])
        return _set_soil_temperature_rhs_reference(inputs._replace(
            soil_snow_temperature=jnp.where(used, inputs.soil_snow_temperature, 0.),
        ), time_step_seconds=time_step_seconds)

    return (_rhs_with_storage_adjoint(primals[0], time_step_seconds),
            jax.jvp(physical_storage, primals, tangents)[1])


def set_soil_temperature_rhs(inputs: SoilTemperatureRHSInputs, *, time_step_seconds=1800.):
    return _rhs_with_storage_adjoint(inputs, time_step_seconds)


NUMBER_OF_TEMPERATURE_SOLVE_LEVELS = (
    NUMBER_OF_SNOW_LEVELS + 1 + NUMBER_OF_GROUND_LEVELS
)
NUMBER_OF_TEMPERATURE_MATRIX_BANDS = 5


class SoilTemperatureMatrixInputs(NamedTuple):

    active_nonlake: jnp.ndarray
    snow_layer_count: jnp.ndarray
    landunit_type: jnp.ndarray
    column_type: jnp.ndarray
    heat_flux_derivative: jnp.ndarray
    layer_thermal_conductivity: jnp.ndarray
    standing_water_thermal_conductivity: jnp.ndarray
    timestep_over_heat_capacity: jnp.ndarray
    standing_water_heat_capacity: jnp.ndarray
    standing_water_thickness: jnp.ndarray
    node_depth: jnp.ndarray
    interface_depth: jnp.ndarray
    standing_water_fraction: jnp.ndarray
    effective_snow_fraction: jnp.ndarray


class SoilTemperatureMatrixResult(NamedTuple):

    band_matrix: jnp.ndarray


def _validate_soil_temperature_matrix_shapes(
    inputs: SoilTemperatureMatrixInputs,
) -> None:
    if inputs.active_nonlake.ndim != 1:
        raise ValueError("active_nonlake must be a vector")
    number_of_columns = inputs.active_nonlake.shape[0]
    vector_shape = (number_of_columns,)
    layer_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS)
    interface_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS + 1)
    layer_fields = {
        "layer_thermal_conductivity",
        "timestep_over_heat_capacity",
        "node_depth",
    }
    interface_fields = {"interface_depth"}
    vector_fields = (
        set(SoilTemperatureMatrixInputs._fields) - layer_fields - interface_fields
    )
    for name in vector_fields:
        if getattr(inputs, name).shape != vector_shape:
            raise ValueError(f"{name} must have shape {vector_shape}")
    for name in layer_fields:
        if getattr(inputs, name).shape != layer_shape:
            raise ValueError(f"{name} must have shape {layer_shape}")
    for name in interface_fields:
        if getattr(inputs, name).shape != interface_shape:
            raise ValueError(f"{name} must have shape {interface_shape}")


def _matrix_product_over_depth(
    factors: tuple[jnp.ndarray, ...],
    depth_interval: jnp.ndarray,
) -> jnp.ndarray:

    product = factors[0]
    for factor in factors[1:]:
        product = fortran_multiply(product, factor)
    return _safe_fortran_divide(product, depth_interval)


def _matrix_interface_coefficient(
    factor: jnp.ndarray,
    conductivity: jnp.ndarray,
    depth_interval: jnp.ndarray,
) -> jnp.ndarray:
    return -_matrix_product_over_depth(
        (
            jnp.float64(1.0 - CRANK_NICHOLSON_FACTOR),
            factor,
            conductivity,
        ),
        depth_interval,
    )


def _matrix_surface_diagonal(
    factor: jnp.ndarray,
    conductivity: jnp.ndarray,
    depth_interval: jnp.ndarray,
    flux_derivative: jnp.ndarray,
) -> jnp.ndarray:
    conductive = _matrix_product_over_depth(
        (
            jnp.float64(1.0 - CRANK_NICHOLSON_FACTOR),
            factor,
            conductivity,
        ),
        depth_interval,
    )
    return fortran_add(
        fortran_add(jnp.float64(1.0), conductive),
        -fortran_multiply(factor, flux_derivative),
    )


def _matrix_internal_diagonal(
    factor: jnp.ndarray,
    upper_conductivity: jnp.ndarray,
    upper_depth_interval: jnp.ndarray,
    lower_conductivity: jnp.ndarray,
    lower_depth_interval: jnp.ndarray,
) -> jnp.ndarray:
    conductivity_sum = fortran_add(
        _safe_fortran_divide(upper_conductivity, upper_depth_interval),
        _safe_fortran_divide(lower_conductivity, lower_depth_interval),
    )
    implicit_factor = fortran_multiply(
        jnp.float64(1.0 - CRANK_NICHOLSON_FACTOR),
        factor,
    )
    return fortran_add(
        jnp.float64(1.0),
        fortran_multiply(implicit_factor, conductivity_sum),
    )


def _matrix_snow_soil_diagonal(
    factor: jnp.ndarray,
    upper_conductivity: jnp.ndarray,
    upper_depth_interval: jnp.ndarray,
    snow_conductivity: jnp.ndarray,
    snow_depth_interval: jnp.ndarray,
    snow_fraction: jnp.ndarray,
    flux_derivative: jnp.ndarray,
) -> jnp.ndarray:
    weighted_snow_conductivity = _matrix_product_over_depth(
        (snow_fraction, snow_conductivity),
        snow_depth_interval,
    )
    conductivity_sum = fortran_add(
        _safe_fortran_divide(upper_conductivity, upper_depth_interval),
        weighted_snow_conductivity,
    )
    conduction = fortran_multiply(
        fortran_multiply(
            jnp.float64(1.0 - CRANK_NICHOLSON_FACTOR),
            factor,
        ),
        conductivity_sum,
    )
    exposed_fraction = fortran_add(jnp.float64(1.0), -snow_fraction)
    surface_derivative = fortran_multiply(
        fortran_multiply(exposed_fraction, factor),
        flux_derivative,
    )
    return fortran_add(
        fortran_add(jnp.float64(1.0), conduction),
        -surface_derivative,
    )


def _column_soil_temperature_matrix(
    inputs: SoilTemperatureMatrixInputs,
    time_step_seconds: jnp.ndarray,
) -> SoilTemperatureMatrixResult:
    active = jnp.asarray(inputs.active_nonlake, dtype=jnp.bool_)
    snow_layer_count = jnp.asarray(inputs.snow_layer_count, dtype=jnp.int32)
    landunit_type = jnp.asarray(inputs.landunit_type, dtype=jnp.int32)
    column_type = jnp.asarray(inputs.column_type, dtype=jnp.int32)
    urban = (landunit_type >= LANDUNIT_URBAN_MIN) & (
        landunit_type <= LANDUNIT_URBAN_MAX
    )
    urban_enclosure = urban & _is_urban_enclosure(column_type)
    urban_road = urban & (
        (column_type == URBAN_IMPERVIOUS_ROAD)
        | (column_type == URBAN_PERVIOUS_ROAD)
    )
    supported = (~urban) | urban_enclosure | urban_road
    active_supported = active & supported
    snow_present = snow_layer_count < 0
    first_active_index = snow_layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS)

    conductivity = jnp.asarray(
        inputs.layer_thermal_conductivity,
        dtype=jnp.float64,
    )
    factor = jnp.asarray(inputs.timestep_over_heat_capacity, dtype=jnp.float64)
    node_depth = jnp.asarray(inputs.node_depth, dtype=jnp.float64)
    interface_depth = jnp.asarray(inputs.interface_depth, dtype=jnp.float64)
    matrix = jnp.zeros(
        (
            NUMBER_OF_TEMPERATURE_MATRIX_BANDS,
            NUMBER_OF_TEMPERATURE_SOLVE_LEVELS,
        ),
        dtype=jnp.float64,
    )

    def snow_entries(index: int):
        next_index = jnp.minimum(index + 1, NUMBER_OF_SOIL_SNOW_LEVELS - 1)
        previous_index = jnp.maximum(index - 1, 0)
        upper_interval = fortran_add(
            node_depth[next_index],
            -node_depth[index],
        )
        lower_interval = fortran_add(
            node_depth[index],
            -node_depth[previous_index],
        )
        top = index == first_active_index
        layer_active = active_supported & (index >= first_active_index)

        diagonal = jnp.where(
            top,
            _matrix_surface_diagonal(
                factor[index],
                conductivity[index],
                upper_interval,
                inputs.heat_flux_derivative,
            ),
            _matrix_internal_diagonal(
                factor[index],
                conductivity[index],
                upper_interval,
                conductivity[previous_index],
                lower_interval,
            ),
        )
        first_superdiagonal = _matrix_interface_coefficient(
            factor[index],
            conductivity[index],
            upper_interval,
        )
        first_subdiagonal = _matrix_interface_coefficient(
            factor[index],
            conductivity[previous_index],
            lower_interval,
        )
        return (
            jnp.where(layer_active, diagonal, jnp.float64(0.0)),
            jnp.where(
                layer_active & (index != NUMBER_OF_SNOW_LEVELS - 1),
                first_superdiagonal,
                jnp.float64(0.0),
            ),
            jnp.where(
                layer_active & ~top,
                first_subdiagonal,
                jnp.float64(0.0),
            ),
        )

    def set_snow_layer(index, current):
        diagonal, upper, lower = snow_entries(index)
        return current.at[2, index].set(diagonal).at[1, index].set(upper).at[3, index].set(lower)

    if _parallel_system_assembly():
        diagonal, upper, lower = jax.vmap(snow_entries)(jnp.arange(NUMBER_OF_SNOW_LEVELS))
        matrix = matrix.at[2, :NUMBER_OF_SNOW_LEVELS].set(diagonal)
        matrix = matrix.at[1, :NUMBER_OF_SNOW_LEVELS].set(upper)
        matrix = matrix.at[3, :NUMBER_OF_SNOW_LEVELS].set(lower)
    else:
        matrix = lax.fori_loop(0, NUMBER_OF_SNOW_LEVELS, set_snow_layer, matrix)

    snow_surface_index = NUMBER_OF_SNOW_LEVELS - 1
    first_soil_layer_index = NUMBER_OF_SNOW_LEVELS
    first_soil_output_index = NUMBER_OF_SNOW_LEVELS + 1
    snow_soil_interval = fortran_add(
        node_depth[first_soil_layer_index],
        -node_depth[snow_surface_index],
    )
    snow_to_soil = _matrix_interface_coefficient(
        factor[snow_surface_index],
        conductivity[snow_surface_index],
        snow_soil_interval,
    )
    matrix = matrix.at[0, snow_surface_index].set(
        jnp.where(
            active_supported & snow_present,
            snow_to_soil,
            jnp.float64(0.0),
        )
    )

    def ground_entries(ground_index: int):
        layer_index = NUMBER_OF_SNOW_LEVELS + ground_index
        fortran_level = ground_index + 1
        previous_index = layer_index - 1
        next_index = jnp.minimum(
            layer_index + 1,
            NUMBER_OF_SOIL_SNOW_LEVELS - 1,
        )
        lower_interval = fortran_add(
            node_depth[layer_index],
            -node_depth[previous_index],
        )
        upper_interval = fortran_add(
            node_depth[next_index],
            -node_depth[layer_index],
        )
        enclosure_valid = fortran_level <= NUMBER_OF_URBAN_LEVELS
        layer_active = active_supported & jnp.where(
            urban_enclosure,
            enclosure_valid,
            True,
        )
        at_surface = (fortran_level == 1) & ~snow_present
        at_snow_interface = (fortran_level == 1) & snow_present
        at_enclosure_bottom = fortran_level == NUMBER_OF_URBAN_LEVELS
        at_ground_bottom = fortran_level == NUMBER_OF_GROUND_LEVELS

        surface_diagonal = _matrix_surface_diagonal(
            factor[layer_index],
            conductivity[layer_index],
            upper_interval,
            inputs.heat_flux_derivative,
        )
        ordinary_snow_diagonal = _matrix_snow_soil_diagonal(
            factor[layer_index],
            conductivity[layer_index],
            upper_interval,
            conductivity[previous_index],
            lower_interval,
            inputs.effective_snow_fraction,
            inputs.heat_flux_derivative,
        )
        internal_diagonal = _matrix_internal_diagonal(
            factor[layer_index],
            conductivity[layer_index],
            upper_interval,
            conductivity[previous_index],
            lower_interval,
        )

        bottom_interface_index = fortran_level + NUMBER_OF_SNOW_LEVELS
        enclosure_bottom_interval = fortran_add(
            interface_depth[bottom_interface_index],
            -node_depth[layer_index],
        )
        enclosure_bottom_diagonal = _matrix_internal_diagonal(
            factor[layer_index],
            conductivity[layer_index],
            enclosure_bottom_interval,
            conductivity[previous_index],
            lower_interval,
        )
        ground_bottom_conduction = _matrix_product_over_depth(
            (
                jnp.float64(1.0 - CRANK_NICHOLSON_FACTOR),
                factor[layer_index],
                conductivity[previous_index],
            ),
            lower_interval,
        )
        ground_bottom_diagonal = fortran_add(
            jnp.float64(1.0),
            ground_bottom_conduction,
        )

        enclosure_diagonal = jnp.where(
            at_surface,
            surface_diagonal,
            jnp.where(
                at_enclosure_bottom,
                enclosure_bottom_diagonal,
                internal_diagonal,
            ),
        )
        ordinary_diagonal = jnp.where(
            at_surface,
            surface_diagonal,
            jnp.where(
                at_snow_interface,
                ordinary_snow_diagonal,
                jnp.where(
                    at_ground_bottom,
                    ground_bottom_diagonal,
                    internal_diagonal,
                ),
            ),
        )
        diagonal = jnp.where(
            urban_enclosure,
            enclosure_diagonal,
            ordinary_diagonal,
        )

        first_superdiagonal = _matrix_interface_coefficient(
            factor[layer_index],
            conductivity[layer_index],
            upper_interval,
        )
        first_subdiagonal = _matrix_interface_coefficient(
            factor[layer_index],
            conductivity[previous_index],
            lower_interval,
        )
        enclosure_bottom_subdiagonal = -fortran_multiply(
            fortran_multiply(
                jnp.float64(1.0 - CRANK_NICHOLSON_FACTOR),
                factor[layer_index],
            ),
            _safe_fortran_divide(
                conductivity[previous_index],
                lower_interval,
            ),
        )
        is_bottom = jnp.where(
            urban_enclosure,
            at_enclosure_bottom,
            at_ground_bottom,
        )
        return (
            jnp.where(layer_active, diagonal, jnp.float64(0.0)),
            jnp.where(
                layer_active & ~is_bottom,
                first_superdiagonal,
                jnp.float64(0.0),
            ),
            jnp.where(
                layer_active & (fortran_level != 1),
                jnp.where(
                    urban_enclosure & at_enclosure_bottom,
                    enclosure_bottom_subdiagonal,
                    first_subdiagonal,
                ),
                jnp.float64(0.0),
            ),
        )

    def set_ground_layer(index, current):
        diagonal, upper, lower = ground_entries(index)
        output = NUMBER_OF_SNOW_LEVELS + 1 + index
        return current.at[2, output].set(diagonal).at[1, output].set(upper).at[3, output].set(lower)

    if _parallel_system_assembly():
        diagonal, upper, lower = jax.vmap(ground_entries)(jnp.arange(NUMBER_OF_GROUND_LEVELS))
        matrix = matrix.at[2, NUMBER_OF_SNOW_LEVELS + 1:].set(diagonal)
        matrix = matrix.at[1, NUMBER_OF_SNOW_LEVELS + 1:].set(upper)
        matrix = matrix.at[3, NUMBER_OF_SNOW_LEVELS + 1:].set(lower)
    else:
        matrix = lax.fori_loop(0, NUMBER_OF_GROUND_LEVELS, set_ground_layer, matrix)

    snow_to_soil_weight = jnp.where(
        urban_enclosure,
        jnp.float64(1.0),
        inputs.effective_snow_fraction,
    )
    soil_to_snow = -_matrix_product_over_depth(
        (
            snow_to_soil_weight,
            jnp.float64(1.0 - CRANK_NICHOLSON_FACTOR),
            factor[first_soil_layer_index],
            conductivity[snow_surface_index],
        ),
        snow_soil_interval,
    )
    matrix = matrix.at[4, first_soil_output_index].set(
        jnp.where(
            active_supported & snow_present,
            soil_to_snow,
            jnp.float64(0.0),
        )
    )

    water_interval = fortran_add(
        fortran_multiply(
            jnp.float64(0.5),
            inputs.standing_water_thickness,
        ),
        node_depth[first_soil_layer_index],
    )
    water_factor = _safe_fortran_divide(
        time_step_seconds,
        inputs.standing_water_heat_capacity,
    )
    water_conduction = _matrix_product_over_depth(
        (
            jnp.float64(1.0 - CRANK_NICHOLSON_FACTOR),
            water_factor,
            inputs.standing_water_thermal_conductivity,
        ),
        water_interval,
    )
    water_diagonal = fortran_add(
        fortran_add(jnp.float64(1.0), water_conduction),
        -fortran_multiply(water_factor, inputs.heat_flux_derivative),
    )
    water_output_index = NUMBER_OF_SNOW_LEVELS
    matrix = matrix.at[2, water_output_index].set(
        jnp.where(active_supported, water_diagonal, jnp.float64(0.0))
    )
    matrix = matrix.at[1, water_output_index].set(
        jnp.where(active_supported, -water_conduction, jnp.float64(0.0))
    )

    soil_water_conduction = _matrix_product_over_depth(
        (
            inputs.standing_water_fraction,
            jnp.float64(1.0 - CRANK_NICHOLSON_FACTOR),
            factor[first_soil_layer_index],
            inputs.standing_water_thermal_conductivity,
        ),
        water_interval,
    )
    matrix = matrix.at[3, first_soil_output_index].set(
        jnp.where(
            active_supported & (inputs.standing_water_fraction != 0.0),
            -soil_water_conduction,
            jnp.float64(0.0),
        )
    )

    soil_water_diagonal_conduction = _matrix_product_over_depth(
        (
            jnp.float64(1.0 - CRANK_NICHOLSON_FACTOR),
            factor[first_soil_layer_index],
            inputs.standing_water_thermal_conductivity,
        ),
        water_interval,
    )
    soil_water_diagonal_derivative = fortran_multiply(
        factor[first_soil_layer_index],
        inputs.heat_flux_derivative,
    )
    soil_water_diagonal_correction = fortran_multiply(
        inputs.standing_water_fraction,
        fortran_add(
            soil_water_diagonal_conduction,
            soil_water_diagonal_derivative,
        ),
    )
    corrected_soil_diagonal = fortran_add(
        matrix[2, first_soil_output_index],
        soil_water_diagonal_correction,
    )
    matrix = matrix.at[2, first_soil_output_index].set(
        jnp.where(
            active_supported & (inputs.standing_water_fraction != 0.0),
            corrected_soil_diagonal,
            matrix[2, first_soil_output_index],
        )
    )

    return SoilTemperatureMatrixResult(
        band_matrix=jnp.where(active_supported, matrix, jnp.zeros_like(matrix))
    )


def _set_soil_temperature_matrix_reference(
    inputs: SoilTemperatureMatrixInputs,
    *,
    time_step_seconds: float = 1800.0,
) -> SoilTemperatureMatrixResult:

    _validate_soil_temperature_matrix_shapes(inputs)
    if time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")
    return jax.vmap(
        lambda column_inputs: _column_soil_temperature_matrix(
            column_inputs,
            jnp.float64(time_step_seconds),
        )
    )(inputs)


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _matrix_with_storage_adjoint(inputs, time_step_seconds):
    return _set_soil_temperature_matrix_reference(inputs, time_step_seconds=time_step_seconds)


@_matrix_with_storage_adjoint.defjvp
def _matrix_storage_jvp(time_step_seconds, primals, tangents):
    def physical_storage(inputs):
        first = inputs.snow_layer_count[:, None] + NUMBER_OF_SNOW_LEVELS
        used = inputs.active_nonlake[:, None] & (
            jnp.arange(NUMBER_OF_SOIL_SNOW_LEVELS)[None, :] >= first)
        interface_used = inputs.active_nonlake[:, None] & (
            jnp.arange(NUMBER_OF_SOIL_SNOW_LEVELS + 1)[None, :] >= first)
        return _set_soil_temperature_matrix_reference(inputs._replace(
            node_depth=jnp.where(used, inputs.node_depth, 0.),
            interface_depth=jnp.where(interface_used, inputs.interface_depth, 0.),
        ), time_step_seconds=time_step_seconds)

    return (_matrix_with_storage_adjoint(primals[0], time_step_seconds),
            jax.jvp(physical_storage, primals, tangents)[1])


def set_soil_temperature_matrix(inputs: SoilTemperatureMatrixInputs, *, time_step_seconds=1800.):
    return _matrix_with_storage_adjoint(inputs, time_step_seconds)


class SoilTemperatureLinearSolveInputs(NamedTuple):

    active_nonlake: jnp.ndarray
    snow_layer_count: jnp.ndarray
    column_type: jnp.ndarray
    band_matrix: jnp.ndarray
    right_hand_side: jnp.ndarray


class SoilTemperatureLinearSolveResult(NamedTuple):

    solution_vector: jnp.ndarray
    all_diagonal_pivots: jnp.ndarray


def _validate_soil_temperature_linear_solve_shapes(
    inputs: SoilTemperatureLinearSolveInputs,
) -> None:
    if inputs.active_nonlake.ndim != 1:
        raise ValueError("active_nonlake must be a vector")
    number_of_columns = inputs.active_nonlake.shape[0]
    vector_shape = (number_of_columns,)
    solve_shape = (number_of_columns, NUMBER_OF_TEMPERATURE_SOLVE_LEVELS)
    matrix_shape = (
        number_of_columns,
        NUMBER_OF_TEMPERATURE_MATRIX_BANDS,
        NUMBER_OF_TEMPERATURE_SOLVE_LEVELS,
    )
    for name in ("active_nonlake", "snow_layer_count", "column_type"):
        if getattr(inputs, name).shape != vector_shape:
            raise ValueError(f"{name} must have shape {vector_shape}")
    if inputs.right_hand_side.shape != solve_shape:
        raise ValueError(f"right_hand_side must have shape {solve_shape}")
    if inputs.band_matrix.shape != matrix_shape:
        raise ValueError(f"band_matrix must have shape {matrix_shape}")


def solve_soil_temperature_banded_system(
    inputs: SoilTemperatureLinearSolveInputs,
) -> SoilTemperatureLinearSolveResult:

    _validate_soil_temperature_linear_solve_shapes(inputs)
    top_index = (
        jnp.asarray(inputs.snow_layer_count, dtype=jnp.int32)
        + NUMBER_OF_SNOW_LEVELS
    )
    enclosure = (
        (inputs.column_type == URBAN_ROOF)
        | (inputs.column_type == URBAN_SUNWALL)
        | (inputs.column_type == URBAN_SHADEWALL)
    )
    bottom_index = jnp.where(
        enclosure,
        NUMBER_OF_SNOW_LEVELS + NUMBER_OF_URBAN_LEVELS,
        NUMBER_OF_TEMPERATURE_SOLVE_LEVELS - 1,
    )
    result = solve_five_band_system(
        inputs.band_matrix,
        inputs.right_hand_side,
        top_index,
        bottom_index,
        inputs.active_nonlake,
    )
    return SoilTemperatureLinearSolveResult(
        solution_vector=result.solution,
        all_diagonal_pivots=result.all_diagonal_pivots,
    )


class SoilTemperatureSolutionUpdateInputs(NamedTuple):

    active_nonlake: jnp.ndarray
    snow_layer_count: jnp.ndarray
    column_type: jnp.ndarray
    standing_water_fraction: jnp.ndarray
    previous_soil_snow_temperature: jnp.ndarray
    previous_standing_water_temperature: jnp.ndarray
    solution_vector: jnp.ndarray


class SoilTemperatureSolutionUpdateResult(NamedTuple):

    soil_snow_temperature: jnp.ndarray
    standing_water_temperature: jnp.ndarray


def _validate_soil_temperature_solution_update_shapes(
    inputs: SoilTemperatureSolutionUpdateInputs,
) -> None:
    if inputs.active_nonlake.ndim != 1:
        raise ValueError("active_nonlake must be a vector")
    number_of_columns = inputs.active_nonlake.shape[0]
    vector_shape = (number_of_columns,)
    layer_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS)
    solve_shape = (number_of_columns, NUMBER_OF_TEMPERATURE_SOLVE_LEVELS)
    vector_fields = {
        "active_nonlake",
        "snow_layer_count",
        "column_type",
        "standing_water_fraction",
        "previous_standing_water_temperature",
    }
    for name in vector_fields:
        if getattr(inputs, name).shape != vector_shape:
            raise ValueError(f"{name} must have shape {vector_shape}")
    if inputs.previous_soil_snow_temperature.shape != layer_shape:
        raise ValueError(
            f"previous_soil_snow_temperature must have shape {layer_shape}"
        )
    if inputs.solution_vector.shape != solve_shape:
        raise ValueError(f"solution_vector must have shape {solve_shape}")


def update_soil_temperature_from_solution(
    inputs: SoilTemperatureSolutionUpdateInputs,
) -> SoilTemperatureSolutionUpdateResult:

    _validate_soil_temperature_solution_update_shapes(inputs)
    active = jnp.asarray(inputs.active_nonlake, dtype=jnp.bool_)[:, None]
    layer_indices = jnp.arange(NUMBER_OF_SOIL_SNOW_LEVELS, dtype=jnp.int32)[
        None, :
    ]
    first_active_snow = (
        jnp.asarray(inputs.snow_layer_count, dtype=jnp.int32)
        + NUMBER_OF_SNOW_LEVELS
    )[:, None]
    snow_layer = layer_indices < NUMBER_OF_SNOW_LEVELS
    active_snow = snow_layer & (layer_indices >= first_active_snow)
    ground_level = layer_indices - NUMBER_OF_SNOW_LEVELS + 1
    enclosure = (
        (inputs.column_type == URBAN_ROOF)
        | (inputs.column_type == URBAN_SUNWALL)
        | (inputs.column_type == URBAN_SHADEWALL)
    )[:, None]
    active_ground = (~snow_layer) & jnp.where(
        enclosure,
        ground_level <= NUMBER_OF_URBAN_LEVELS,
        ground_level <= NUMBER_OF_GROUND_LEVELS,
    )
    solution_indices = jnp.where(
        snow_layer,
        layer_indices,
        layer_indices + 1,
    )
    solved_layers = jnp.take_along_axis(
        inputs.solution_vector,
        solution_indices,
        axis=1,
    )
    update_mask = active & (active_snow | active_ground)
    soil_snow_temperature = jnp.where(
        update_mask,
        solved_layers,
        inputs.previous_soil_snow_temperature,
    )

    first_soil_temperature = soil_snow_temperature[:, NUMBER_OF_SNOW_LEVELS]
    solved_water_temperature = inputs.solution_vector[:, NUMBER_OF_SNOW_LEVELS]
    standing_water_temperature = jnp.where(
        inputs.standing_water_fraction == 0.0,
        first_soil_temperature,
        solved_water_temperature,
    )
    standing_water_temperature = jnp.where(
        inputs.active_nonlake,
        standing_water_temperature,
        inputs.previous_standing_water_temperature,
    )
    return SoilTemperatureSolutionUpdateResult(
        soil_snow_temperature=soil_snow_temperature,
        standing_water_temperature=standing_water_temperature,
    )


class SoilTemperaturePostSolveFluxInputs(NamedTuple):

    active_nonlake: jnp.ndarray
    snow_layer_count: jnp.ndarray
    urban_landunit: jnp.ndarray
    column_type: jnp.ndarray
    node_depth: jnp.ndarray
    interface_depth: jnp.ndarray
    interface_thermal_conductivity: jnp.ndarray
    previous_interface_heat_flux: jnp.ndarray
    previous_updated_interface_heat_flux: jnp.ndarray
    previous_soil_snow_temperature: jnp.ndarray
    soil_snow_temperature: jnp.ndarray
    building_temperature: jnp.ndarray
    roof_inner_temperature: jnp.ndarray
    sunwall_inner_temperature: jnp.ndarray
    shadewall_inner_temperature: jnp.ndarray
    cooling_on: jnp.ndarray
    heating_on: jnp.ndarray
    previous_building_heat_flux: jnp.ndarray
    previous_urban_air_conditioning_flux: jnp.ndarray
    previous_urban_heating_flux: jnp.ndarray


class SoilTemperaturePostSolveFluxResult(NamedTuple):

    interface_heat_flux: jnp.ndarray
    updated_interface_heat_flux: jnp.ndarray
    building_heat_flux: jnp.ndarray
    urban_air_conditioning_flux: jnp.ndarray
    urban_heating_flux: jnp.ndarray


def _validate_soil_temperature_postsolve_flux_shapes(
    inputs: SoilTemperaturePostSolveFluxInputs,
) -> None:
    if inputs.active_nonlake.ndim != 1:
        raise ValueError("active_nonlake must be a vector")
    number_of_columns = inputs.active_nonlake.shape[0]
    vector_shape = (number_of_columns,)
    layer_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS)
    interface_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS + 1)
    layer_fields = {
        "node_depth",
        "interface_thermal_conductivity",
        "previous_interface_heat_flux",
        "previous_updated_interface_heat_flux",
        "previous_soil_snow_temperature",
        "soil_snow_temperature",
    }
    for name in set(SoilTemperaturePostSolveFluxInputs._fields) - (
        layer_fields | {"interface_depth"}
    ):
        if getattr(inputs, name).shape != vector_shape:
            raise ValueError(f"{name} must have shape {vector_shape}")
    for name in layer_fields:
        if getattr(inputs, name).shape != layer_shape:
            raise ValueError(f"{name} must have shape {layer_shape}")
    if inputs.interface_depth.shape != interface_shape:
        raise ValueError(f"interface_depth must have shape {interface_shape}")


@partial(jax.custom_jvp, nondiff_argnums=(4,))
def _postsolve_conductive_flux(
    conductivity, temperature_difference, distance, level_flags, inner_boundary,
):
    return _safe_fortran_divide(fortran_multiply(conductivity, temperature_difference), distance)


@_postsolve_conductive_flux.defjvp
def _postsolve_conductive_flux_jvp(inner_boundary, primals, tangents):
    valid, enclosure_bottom, ground_bottom = primals[3]
    if inner_boundary:
        used = valid & enclosure_bottom
    else:
        used = valid & ~enclosure_bottom & ~ground_bottom

    def active_flux(k, delta_t, delta_z):
        return _safe_fortran_divide(
            fortran_multiply(jnp.where(used, k, 0.), jnp.where(used, delta_t, 0.)),
            jnp.where(used, delta_z, 1.),
        )

    value = _postsolve_conductive_flux(*primals, inner_boundary)
    return value, jax.jvp(active_flux, primals[:3], tangents[:3])[1]


def _column_soil_temperature_postsolve_flux(
    inputs: SoilTemperaturePostSolveFluxInputs,
    *,
    simple_building_temperature: bool,
    parallel_levels: bool = False,
) -> SoilTemperaturePostSolveFluxResult:

    active = jnp.asarray(inputs.active_nonlake, dtype=jnp.bool_)
    first_active_index = (
        jnp.asarray(inputs.snow_layer_count, dtype=jnp.int32)
        + jnp.int32(NUMBER_OF_SNOW_LEVELS)
    )
    column_type = jnp.asarray(inputs.column_type, dtype=jnp.int32)
    enclosure = _is_urban_enclosure(column_type)

    def enclosure_inner_temperature() -> jnp.ndarray:
        if simple_building_temperature:
            return inputs.building_temperature
        return jnp.where(
            column_type == URBAN_SUNWALL,
            inputs.sunwall_inner_temperature,
            jnp.where(
                column_type == URBAN_SHADEWALL,
                inputs.shadewall_inner_temperature,
                inputs.roof_inner_temperature,
            ),
        )

    inner_temperature = enclosure_inner_temperature()

    def evaluate_level(index: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        fortran_level = index + jnp.int32(FIRST_FORTRAN_LEVEL)
        enclosure_level = enclosure & (fortran_level <= NUMBER_OF_URBAN_LEVELS)
        valid = active & (index >= first_active_index) & (
            enclosure_level | (~enclosure)
        )
        enclosure_bottom = enclosure & (fortran_level == NUMBER_OF_URBAN_LEVELS)
        ground_bottom = (~enclosure) & (fortran_level == NUMBER_OF_GROUND_LEVELS)
        level_flags = valid, enclosure_bottom, ground_bottom

        next_index = jnp.minimum(index + 1, GROUND_BOTTOM_INDEX)
        neighboring_temperature_difference = fortran_add(
            inputs.soil_snow_temperature[next_index],
            -inputs.soil_snow_temperature[index],
        )
        neighboring_depth_difference = fortran_add(
            inputs.node_depth[next_index],
            -inputs.node_depth[index],
        )
        neighboring_flux = _postsolve_conductive_flux(
            inputs.interface_thermal_conductivity[index],
            neighboring_temperature_difference,
            neighboring_depth_difference,
            level_flags, False,
        )

        inner_depth_difference = fortran_add(
            inputs.interface_depth[index + 1],
            -inputs.node_depth[index],
        )
        new_inner_flux = _postsolve_conductive_flux(
            inputs.interface_thermal_conductivity[index],
            fortran_add(inner_temperature, -inputs.soil_snow_temperature[index]),
            inner_depth_difference,
            level_flags, True,
        )
        old_inner_flux = _postsolve_conductive_flux(
            inputs.interface_thermal_conductivity[index],
            fortran_add(inner_temperature, -inputs.previous_soil_snow_temperature[index]),
            inner_depth_difference,
            level_flags, True,
        )
        selected_updated_flux = jnp.where(
            enclosure_bottom,
            new_inner_flux,
            jnp.where(ground_bottom, jnp.float64(0.0), neighboring_flux),
        )
        return (
            jnp.where(
                valid & enclosure_bottom,
                old_inner_flux,
                inputs.previous_interface_heat_flux[index],
            ),
            jnp.where(
                valid,
                selected_updated_flux,
                inputs.previous_updated_interface_heat_flux[index],
            ),
        )

    def update_level(index, fluxes):
        original, updated = evaluate_level(index)
        return fluxes[0].at[index].set(original), fluxes[1].at[index].set(updated)

    if parallel_levels:
        interface_heat_flux, updated_interface_heat_flux = jax.vmap(evaluate_level)(
            jnp.arange(NUMBER_OF_SOIL_SNOW_LEVELS),
        )
    else:
        interface_heat_flux, updated_interface_heat_flux = lax.fori_loop(
            0, NUMBER_OF_SOIL_SNOW_LEVELS, update_level,
            (inputs.previous_interface_heat_flux, inputs.previous_updated_interface_heat_flux),
        )

    building_heat_flux = fortran_add(
        fortran_multiply(
            jnp.float64(CRANK_NICHOLSON_FACTOR),
            interface_heat_flux[URBAN_BOTTOM_INDEX],
        ),
        fortran_multiply(
            fortran_add(
                jnp.float64(1.0),
                -jnp.float64(CRANK_NICHOLSON_FACTOR),
            ),
            updated_interface_heat_flux[URBAN_BOTTOM_INDEX],
        ),
    )
    update_urban = active & jnp.asarray(inputs.urban_landunit, dtype=jnp.bool_)
    building_heat_flux = jnp.where(
        update_urban,
        jnp.where(enclosure, building_heat_flux, jnp.float64(0.0)),
        inputs.previous_building_heat_flux,
    )

    urban_air_conditioning_flux = inputs.previous_urban_air_conditioning_flux
    urban_heating_flux = inputs.previous_urban_heating_flux
    if simple_building_temperature:
        air_conditioning = jnp.where(
            inputs.cooling_on,
            jnp.abs(building_heat_flux),
            jnp.float64(0.0),
        )
        heating = jnp.where(
            inputs.cooling_on,
            jnp.float64(0.0),
            jnp.where(
                inputs.heating_on,
                jnp.abs(building_heat_flux),
                jnp.float64(0.0),
            ),
        )
        urban_air_conditioning_flux = jnp.where(
            update_urban,
            air_conditioning,
            urban_air_conditioning_flux,
        )
        urban_heating_flux = jnp.where(
            update_urban,
            heating,
            urban_heating_flux,
        )

    return SoilTemperaturePostSolveFluxResult(
        interface_heat_flux=interface_heat_flux,
        updated_interface_heat_flux=updated_interface_heat_flux,
        building_heat_flux=building_heat_flux,
        urban_air_conditioning_flux=urban_air_conditioning_flux,
        urban_heating_flux=urban_heating_flux,
    )


def recompute_soil_temperature_interface_fluxes(
    inputs: SoilTemperaturePostSolveFluxInputs,
    *,
    simple_building_temperature: bool = False,
) -> SoilTemperaturePostSolveFluxResult:

    _validate_soil_temperature_postsolve_flux_shapes(inputs)
    return jax.vmap(
        lambda column_inputs: _column_soil_temperature_postsolve_flux(
            column_inputs,
            simple_building_temperature=simple_building_temperature,
            parallel_levels=_parallel_thermal_levels(),
        )
    )(inputs)


class StandingWaterPhaseChangeInputs(NamedTuple):

    active_nonlake: jnp.ndarray
    snow_layer_count: jnp.ndarray
    surface_flux_temperature_derivative: jnp.ndarray
    effective_snow_fraction: jnp.ndarray
    standing_water_fraction: jnp.ndarray
    snow_water: jnp.ndarray
    standing_water: jnp.ndarray
    integrated_snowfall: jnp.ndarray
    snow_depth: jnp.ndarray
    ice_water: jnp.ndarray
    timestep_over_heat_capacity: jnp.ndarray
    standing_water_heat_capacity: jnp.ndarray
    soil_snow_temperature: jnp.ndarray
    standing_water_temperature: jnp.ndarray
    previous_phase_change_heat_flux: jnp.ndarray
    previous_water_to_ice_flux: jnp.ndarray
    previous_water_to_snow_heat_flux: jnp.ndarray


class StandingWaterPhaseChangeResult(NamedTuple):

    snow_water: jnp.ndarray
    standing_water: jnp.ndarray
    integrated_snowfall: jnp.ndarray
    snow_depth: jnp.ndarray
    ice_water: jnp.ndarray
    soil_snow_temperature: jnp.ndarray
    standing_water_temperature: jnp.ndarray
    phase_change_heat_flux: jnp.ndarray
    water_to_ice_flux: jnp.ndarray
    water_to_snow_heat_flux: jnp.ndarray


def _validate_standing_water_phase_change_shapes(
    inputs: StandingWaterPhaseChangeInputs,
) -> None:
    if inputs.active_nonlake.ndim != 1:
        raise ValueError("active_nonlake must be a vector")
    number_of_columns = inputs.active_nonlake.shape[0]
    vector_shape = (number_of_columns,)
    layer_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS)
    layer_fields = {
        "ice_water",
        "timestep_over_heat_capacity",
        "soil_snow_temperature",
    }
    for name in set(StandingWaterPhaseChangeInputs._fields) - layer_fields:
        if getattr(inputs, name).shape != vector_shape:
            raise ValueError(f"{name} must have shape {vector_shape}")
    for name in layer_fields:
        if getattr(inputs, name).shape != layer_shape:
            raise ValueError(f"{name} must have shape {layer_shape}")


def _standing_water_phase_change_c1(
    snow_layer_count: jnp.ndarray,
    snow_fraction: jnp.ndarray,
    bottom_snow_factor: jnp.ndarray,
    heat_flux_derivative: jnp.ndarray,
    time_step_seconds: jnp.ndarray,
) -> jnp.ndarray:

    one_layer_weight = fortran_multiply(
        snow_fraction,
        fortran_add(
            _safe_fortran_divide(time_step_seconds, bottom_snow_factor),
            -fortran_multiply(heat_flux_derivative, time_step_seconds),
        ),
    )
    multilayer_weight = fortran_multiply(
        _safe_fortran_divide(snow_fraction, bottom_snow_factor),
        time_step_seconds,
    )
    return jnp.where(
        snow_layer_count == jnp.int32(-1),
        one_layer_weight,
        multilayer_weight,
    )


def _column_standing_water_phase_change(
    inputs: StandingWaterPhaseChangeInputs,
    time_step_seconds: jnp.ndarray,
) -> StandingWaterPhaseChangeResult:

    active_result = StandingWaterPhaseChangeResult(
        snow_water=inputs.snow_water,
        standing_water=inputs.standing_water,
        integrated_snowfall=inputs.integrated_snowfall,
        snow_depth=inputs.snow_depth,
        ice_water=inputs.ice_water,
        soil_snow_temperature=inputs.soil_snow_temperature,
        standing_water_temperature=inputs.standing_water_temperature,
        phase_change_heat_flux=jnp.float64(0.0),
        water_to_ice_flux=jnp.float64(0.0),
        water_to_snow_heat_flux=jnp.float64(0.0),
    )
    inactive_result = StandingWaterPhaseChangeResult(
        snow_water=inputs.snow_water,
        standing_water=inputs.standing_water,
        integrated_snowfall=inputs.integrated_snowfall,
        snow_depth=inputs.snow_depth,
        ice_water=inputs.ice_water,
        soil_snow_temperature=inputs.soil_snow_temperature,
        standing_water_temperature=inputs.standing_water_temperature,
        phase_change_heat_flux=inputs.previous_phase_change_heat_flux,
        water_to_ice_flux=inputs.previous_water_to_ice_flux,
        water_to_snow_heat_flux=inputs.previous_water_to_snow_heat_flux,
    )

    def freeze_surface_water(
        initialized: StandingWaterPhaseChangeResult,
    ) -> StandingWaterPhaseChangeResult:
        temperature_increment = fortran_add(
            jnp.float64(FREEZING_TEMPERATURE),
            -initialized.standing_water_temperature,
        )
        standing_water_temperature = jnp.float64(FREEZING_TEMPERATURE)
        temperature_capacity_loss = _safe_fortran_divide(
            fortran_multiply(
                temperature_increment,
                inputs.standing_water_heat_capacity,
            ),
            time_step_seconds,
        )
        energy_residual = fortran_multiply(
            inputs.standing_water_fraction,
            fortran_add(
                fortran_multiply(
                    inputs.surface_flux_temperature_derivative,
                    temperature_increment,
                ),
                -temperature_capacity_loss,
            ),
        )
        energy_residual = jnp.where(
            temperature_increment == 0.0,
            jnp.float64(-0.0),
            energy_residual,
        )
        phase_mass = _safe_fortran_divide(
            fortran_multiply(energy_residual, time_step_seconds),
            jnp.float64(LATENT_HEAT_FUSION),
        )
        remaining_water = fortran_add(initialized.standing_water, phase_mass)
        snow_area_depth = fortran_multiply(
            inputs.effective_snow_fraction,
            initialized.snow_depth,
        )
        average_snow_density = lax.cond(
            snow_area_depth > 0.0,
            lambda _: jnp.minimum(
                jnp.float64(800.0),
                _safe_fortran_divide(initialized.snow_water, snow_area_depth),
            ),
            lambda _: jnp.float64(200.0),
            operand=None,
        )

        def update_snow_depth(
            snow_water: jnp.ndarray,
            density: jnp.ndarray,
        ) -> jnp.ndarray:
            return lax.cond(
                (inputs.effective_snow_fraction > 0.0)
                & (inputs.snow_layer_count < 0),
                lambda _: _safe_fortran_divide(
                    snow_water,
                    fortran_multiply(
                        density,
                        inputs.effective_snow_fraction,
                    ),
                ),
                lambda _: _safe_fortran_divide(
                    snow_water,
                    jnp.float64(ICE_DENSITY),
                ),
                operand=None,
            )

        def partially_freeze(
            _: None,
        ) -> StandingWaterPhaseChangeResult:
            snow_water = fortran_add(initialized.snow_water, -phase_mass)
            integrated_snowfall = fortran_add(
                initialized.integrated_snowfall,
                -phase_mass,
            )
            bottom_snow_index = NUMBER_OF_SNOW_LEVELS - 1
            bottom_snow_ice = fortran_add(
                initialized.ice_water[bottom_snow_index],
                -phase_mass,
            )
            ice_water = initialized.ice_water.at[bottom_snow_index].set(
                jnp.where(
                    inputs.snow_layer_count < 0,
                    bottom_snow_ice,
                    initialized.ice_water[bottom_snow_index],
                )
            )
            standing_water = remaining_water
            snow_depth = update_snow_depth(snow_water, average_snow_density)
            water_to_ice_flux = _safe_fortran_divide(
                -phase_mass,
                time_step_seconds,
            )

            def no_resolved_snow(_: None) -> tuple[jnp.ndarray, jnp.ndarray]:
                return standing_water_temperature, jnp.float64(0.0)

            def mix_lowest_snow(_: None) -> tuple[jnp.ndarray, jnp.ndarray]:
                bottom_factor = inputs.timestep_over_heat_capacity[
                    bottom_snow_index
                ]
                snow_weight = _standing_water_phase_change_c1(
                    inputs.snow_layer_count,
                    inputs.effective_snow_fraction,
                    bottom_factor,
                    inputs.surface_flux_temperature_derivative,
                    time_step_seconds,
                )
                water_weight = jnp.where(
                    inputs.standing_water_fraction != 0.0,
                    fortran_add(
                        fortran_multiply(
                            -jnp.float64(LIQUID_WATER_SPECIFIC_HEAT),
                            phase_mass,
                        ),
                        -fortran_multiply(
                            fortran_multiply(
                                inputs.standing_water_fraction,
                                inputs.surface_flux_temperature_derivative,
                            ),
                            time_step_seconds,
                        ),
                    ),
                    jnp.float64(0.0),
                )
                mixed_temperature = _safe_fortran_divide(
                    fortran_add(
                        fortran_multiply(
                            snow_weight,
                            initialized.soil_snow_temperature[
                                bottom_snow_index
                            ],
                        ),
                        fortran_multiply(
                            water_weight,
                            standing_water_temperature,
                        ),
                    ),
                    fortran_add(snow_weight, water_weight),
                )
                water_to_snow_heat_flux = _safe_fortran_divide(
                    fortran_multiply(
                        fortran_add(
                            standing_water_temperature,
                            -mixed_temperature,
                        ),
                        water_weight,
                    ),
                    time_step_seconds,
                )
                return mixed_temperature, water_to_snow_heat_flux

            bottom_snow_temperature, water_to_snow_heat_flux = lax.cond(
                inputs.snow_layer_count == 0,
                no_resolved_snow,
                mix_lowest_snow,
                operand=None,
            )
            soil_snow_temperature = initialized.soil_snow_temperature.at[
                bottom_snow_index
            ].set(bottom_snow_temperature)
            return StandingWaterPhaseChangeResult(
                snow_water=snow_water,
                standing_water=standing_water,
                integrated_snowfall=integrated_snowfall,
                snow_depth=snow_depth,
                ice_water=ice_water,
                soil_snow_temperature=soil_snow_temperature,
                standing_water_temperature=standing_water_temperature,
                phase_change_heat_flux=energy_residual,
                water_to_ice_flux=water_to_ice_flux,
                water_to_snow_heat_flux=water_to_snow_heat_flux,
            )

        def completely_freeze(
            _: None,
        ) -> StandingWaterPhaseChangeResult:
            density_numerator = fortran_add(
                fortran_multiply(initialized.snow_water, average_snow_density),
                fortran_multiply(
                    initialized.standing_water,
                    jnp.float64(ICE_DENSITY),
                ),
            )
            average_combined_density = _safe_fortran_divide(
                density_numerator,
                fortran_add(initialized.snow_water, initialized.standing_water),
            )
            snow_water = fortran_add(
                initialized.snow_water,
                initialized.standing_water,
            )
            integrated_snowfall = fortran_add(
                initialized.integrated_snowfall,
                initialized.standing_water,
            )
            water_to_ice_flux = _safe_fortran_divide(
                initialized.standing_water,
                time_step_seconds,
            )
            bottom_snow_index = NUMBER_OF_SNOW_LEVELS - 1
            combined_bottom_ice = fortran_add(
                initialized.ice_water[bottom_snow_index],
                initialized.standing_water,
            )
            ice_water = initialized.ice_water.at[bottom_snow_index].set(
                jnp.where(
                    inputs.snow_layer_count < 0,
                    combined_bottom_ice,
                    initialized.ice_water[bottom_snow_index],
                )
            )
            cooled_water_temperature = fortran_add(
                standing_water_temperature,
                -_safe_fortran_divide(
                    fortran_multiply(
                        remaining_water,
                        jnp.float64(LATENT_HEAT_FUSION),
                    ),
                    fortran_add(
                        fortran_multiply(
                            time_step_seconds,
                            inputs.surface_flux_temperature_derivative,
                        ),
                        -inputs.standing_water_heat_capacity,
                    ),
                ),
            )
            excess_latent_flux = _safe_fortran_divide(
                fortran_multiply(
                    fortran_multiply(
                        inputs.standing_water_fraction,
                        remaining_water,
                    ),
                    jnp.float64(LATENT_HEAT_FUSION),
                ),
                time_step_seconds,
            )
            phase_change_heat_flux = fortran_add(
                energy_residual,
                -excess_latent_flux,
            )

            def no_resolved_snow(_: None) -> tuple[jnp.ndarray, jnp.ndarray]:
                return cooled_water_temperature, cooled_water_temperature

            def mix_lowest_snow(_: None) -> tuple[jnp.ndarray, jnp.ndarray]:
                bottom_factor = inputs.timestep_over_heat_capacity[
                    bottom_snow_index
                ]
                snow_weight = _standing_water_phase_change_c1(
                    inputs.snow_layer_count,
                    inputs.effective_snow_fraction,
                    bottom_factor,
                    inputs.surface_flux_temperature_derivative,
                    time_step_seconds,
                )
                water_weight = jnp.where(
                    inputs.standing_water_fraction != 0.0,
                    fortran_multiply(
                        inputs.standing_water_fraction,
                        fortran_add(
                            inputs.standing_water_heat_capacity,
                            -fortran_multiply(
                                time_step_seconds,
                                inputs.surface_flux_temperature_derivative,
                            ),
                        ),
                    ),
                    jnp.float64(0.0),
                )
                equilibrium_temperature = _safe_fortran_divide(
                    fortran_add(
                        fortran_multiply(
                            snow_weight,
                            initialized.soil_snow_temperature[
                                bottom_snow_index
                            ],
                        ),
                        fortran_multiply(
                            water_weight,
                            cooled_water_temperature,
                        ),
                    ),
                    fortran_add(snow_weight, water_weight),
                )
                return equilibrium_temperature, equilibrium_temperature

            bottom_snow_temperature, final_water_temperature = lax.cond(
                inputs.snow_layer_count == 0,
                no_resolved_snow,
                mix_lowest_snow,
                operand=None,
            )
            soil_snow_temperature = initialized.soil_snow_temperature.at[
                bottom_snow_index
            ].set(bottom_snow_temperature)
            snow_depth = update_snow_depth(
                snow_water,
                average_combined_density,
            )
            return StandingWaterPhaseChangeResult(
                snow_water=snow_water,
                standing_water=jnp.float64(0.0),
                integrated_snowfall=integrated_snowfall,
                snow_depth=snow_depth,
                ice_water=ice_water,
                soil_snow_temperature=soil_snow_temperature,
                standing_water_temperature=final_water_temperature,
                phase_change_heat_flux=phase_change_heat_flux,
                water_to_ice_flux=water_to_ice_flux,
                water_to_snow_heat_flux=jnp.float64(0.0),
            )

        return lax.cond(
            remaining_water >= 0.0,
            partially_freeze,
            completely_freeze,
            operand=None,
        )

    def update_active(_: None) -> StandingWaterPhaseChangeResult:
        freezing = (
            inputs.standing_water_fraction > 0.0
        ) & (
            inputs.standing_water_temperature
            <= jnp.float64(FREEZING_TEMPERATURE)
        )
        return lax.cond(
            freezing,
            freeze_surface_water,
            lambda initialized: initialized,
            active_result,
        )

    return lax.cond(
        jnp.asarray(inputs.active_nonlake, dtype=jnp.bool_),
        update_active,
        lambda _: inactive_result,
        operand=None,
    )


def phase_change_standing_surface_water(
    inputs: StandingWaterPhaseChangeInputs,
    *,
    time_step_seconds: float = 1800.0,
) -> StandingWaterPhaseChangeResult:

    _validate_standing_water_phase_change_shapes(inputs)
    if time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")
    return jax.vmap(
        lambda column_inputs: _column_standing_water_phase_change(
            column_inputs,
            jnp.float64(time_step_seconds),
        )
    )(inputs)


class SoilSnowPhaseChangeInputs(NamedTuple):

    active_nonlake: jnp.ndarray
    snow_layer_count: jnp.ndarray
    landunit_type: jnp.ndarray
    column_type: jnp.ndarray
    layer_thickness: jnp.ndarray
    soil_clapp_hornberger_exponent: jnp.ndarray
    soil_saturated_suction: jnp.ndarray
    soil_porosity: jnp.ndarray
    surface_flux_temperature_derivative: jnp.ndarray
    effective_snow_fraction: jnp.ndarray
    standing_water_fraction: jnp.ndarray
    snow_depth: jnp.ndarray
    snow_water: jnp.ndarray
    liquid_water: jnp.ndarray
    ice_water: jnp.ndarray
    timestep_over_heat_capacity: jnp.ndarray
    soil_snow_temperature: jnp.ndarray
    previous_phase_flag: jnp.ndarray
    previous_phase_change_heat_flux: jnp.ndarray
    previous_snow_melt_flux: jnp.ndarray
    previous_snow_freeze_flux: jnp.ndarray
    previous_snow_drain_flux: jnp.ndarray
    previous_snow_layer_melt_flux: jnp.ndarray
    previous_snow_layer_freeze_flux: jnp.ndarray
    previous_snow_melt_heat_flux: jnp.ndarray
    previous_rural_snow_melt_heat_flux: jnp.ndarray
    previous_urban_snow_melt_heat_flux: jnp.ndarray


class SoilSnowPhaseChangeResult(NamedTuple):

    snow_depth: jnp.ndarray
    snow_water: jnp.ndarray
    liquid_water: jnp.ndarray
    ice_water: jnp.ndarray
    soil_snow_temperature: jnp.ndarray
    phase_flag: jnp.ndarray
    phase_change_heat_flux: jnp.ndarray
    snow_melt_flux: jnp.ndarray
    snow_freeze_flux: jnp.ndarray
    snow_drain_flux: jnp.ndarray
    snow_layer_melt_flux: jnp.ndarray
    snow_layer_freeze_flux: jnp.ndarray
    snow_melt_heat_flux: jnp.ndarray
    rural_snow_melt_heat_flux: jnp.ndarray
    urban_snow_melt_heat_flux: jnp.ndarray


class _SoilSnowPhaseLoopState(NamedTuple):
    temperature: jnp.ndarray
    liquid_water: jnp.ndarray
    ice_water: jnp.ndarray
    phase_flag: jnp.ndarray
    snow_depth: jnp.ndarray
    snow_water: jnp.ndarray
    phase_change_heat_flux: jnp.ndarray
    snow_melt_flux: jnp.ndarray
    snow_freeze_flux: jnp.ndarray
    snow_drain_flux: jnp.ndarray
    snow_layer_melt_flux: jnp.ndarray
    snow_layer_freeze_flux: jnp.ndarray


def _validate_soil_snow_phase_change_shapes(
    inputs: SoilSnowPhaseChangeInputs,
) -> None:
    if inputs.active_nonlake.ndim != 1:
        raise ValueError("active_nonlake must be a vector")
    number_of_columns = inputs.active_nonlake.shape[0]
    vector_shape = (number_of_columns,)
    layer_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS)
    ground_shape = (number_of_columns, NUMBER_OF_GROUND_LEVELS)
    snow_shape = (number_of_columns, NUMBER_OF_SNOW_LEVELS)
    layer_fields = {
        "layer_thickness",
        "liquid_water",
        "ice_water",
        "timestep_over_heat_capacity",
        "soil_snow_temperature",
        "previous_phase_flag",
    }
    ground_fields = {
        "soil_clapp_hornberger_exponent",
        "soil_saturated_suction",
        "soil_porosity",
    }
    snow_fields = {
        "previous_snow_layer_melt_flux",
        "previous_snow_layer_freeze_flux",
    }
    vector_fields = (
        set(SoilSnowPhaseChangeInputs._fields)
        - layer_fields
        - ground_fields
        - snow_fields
    )
    for name in vector_fields:
        if getattr(inputs, name).shape != vector_shape:
            raise ValueError(f"{name} must have shape {vector_shape}")
    for name in layer_fields:
        if getattr(inputs, name).shape != layer_shape:
            raise ValueError(f"{name} must have shape {layer_shape}")
    for name in ground_fields:
        if getattr(inputs, name).shape != ground_shape:
            raise ValueError(f"{name} must have shape {ground_shape}")
    for name in snow_fields:
        if getattr(inputs, name).shape != snow_shape:
            raise ValueError(f"{name} must have shape {snow_shape}")


def _soil_supercooled_liquid_water(
    temperature: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    porosity: jnp.ndarray,
    saturated_suction: jnp.ndarray,
    clapp_hornberger_exponent: jnp.ndarray,
    eligible: jnp.ndarray,
) -> jnp.ndarray:

    tfrz = jnp.float64(FREEZING_TEMPERATURE)
    cold = eligible & (temperature < tfrz)
    temperature_depression = fortran_add(tfrz, -temperature)
    suction_numerator = fortran_multiply(
        jnp.float64(LATENT_HEAT_FUSION),
        temperature_depression,
    )
    suction_denominator = fortran_multiply(
        jnp.float64(GRAVITY),
        temperature,
    )
    frozen_suction = fortran_multiply(
        _safe_fortran_divide(
            jnp.where(cold, suction_numerator, jnp.float64(1.0)),
            jnp.where(cold, suction_denominator, jnp.float64(1.0)),
        ),
        jnp.float64(1000.0),
    )
    suction_ratio = _safe_fortran_divide(
        jnp.where(cold, frozen_suction, jnp.float64(1.0)),
        jnp.where(cold, saturated_suction, jnp.float64(1.0)),
    )
    exponent = _safe_fortran_divide(
        jnp.float64(-1.0),
        jnp.where(cold, clapp_hornberger_exponent, jnp.float64(1.0)),
    )
    retention_fraction = _material_power(suction_ratio, exponent)
    supercooled = fortran_multiply(porosity, retention_fraction)
    supercooled = fortran_multiply(supercooled, layer_thickness)
    supercooled = fortran_multiply(supercooled, jnp.float64(1000.0))
    return jnp.where(cold, supercooled, jnp.float64(0.0))


@jax.custom_jvp
def _phase_temperature_with_latent_balance(rounded_temperature, phase_temperature, heat_absorbed):
    return rounded_temperature


@_phase_temperature_with_latent_balance.defjvp
def _phase_temperature_with_latent_balance_jvp(primals, tangents):
    rounded_temperature, _, heat_absorbed = primals
    rounded_dot, phase_dot, _ = tangents
    return rounded_temperature, jnp.where(heat_absorbed, phase_dot, rounded_dot)


def _column_soil_snow_phase_change(
    inputs: SoilSnowPhaseChangeInputs,
    time_step_seconds: jnp.ndarray,
    *,
    linearize_latent_balance: bool = False,
) -> SoilSnowPhaseChangeResult:

    inactive_result = SoilSnowPhaseChangeResult(
        snow_depth=inputs.snow_depth,
        snow_water=inputs.snow_water,
        liquid_water=inputs.liquid_water,
        ice_water=inputs.ice_water,
        soil_snow_temperature=inputs.soil_snow_temperature,
        phase_flag=inputs.previous_phase_flag,
        phase_change_heat_flux=inputs.previous_phase_change_heat_flux,
        snow_melt_flux=inputs.previous_snow_melt_flux,
        snow_freeze_flux=inputs.previous_snow_freeze_flux,
        snow_drain_flux=inputs.previous_snow_drain_flux,
        snow_layer_melt_flux=inputs.previous_snow_layer_melt_flux,
        snow_layer_freeze_flux=inputs.previous_snow_layer_freeze_flux,
        snow_melt_heat_flux=inputs.previous_snow_melt_heat_flux,
        rural_snow_melt_heat_flux=inputs.previous_rural_snow_melt_heat_flux,
        urban_snow_melt_heat_flux=inputs.previous_urban_snow_melt_heat_flux,
    )

    def update_active(_: None) -> SoilSnowPhaseChangeResult:
        tfrz = jnp.float64(FREEZING_TEMPERATURE)
        hfus = jnp.float64(LATENT_HEAT_FUSION)
        top_index = (
            jnp.asarray(inputs.snow_layer_count, dtype=jnp.int32)
            + jnp.int32(NUMBER_OF_SNOW_LEVELS)
        )
        indices = jnp.arange(NUMBER_OF_SOIL_SNOW_LEVELS, dtype=jnp.int32)
        valid = indices >= top_index
        snow = indices < NUMBER_OF_SNOW_LEVELS
        enclosure = _is_urban_enclosure(inputs.column_type)
        ground_level = indices - NUMBER_OF_SNOW_LEVELS + 1
        processable = (~enclosure) | (ground_level <= NUMBER_OF_URBAN_LEVELS)
        processable = valid & (snow | processable)

        snow_melting = (
            processable
            & snow
            & (inputs.ice_water > 0.0)
            & (inputs.soil_snow_temperature > tfrz)
        )
        snow_freezing = (
            processable
            & snow
            & (inputs.liquid_water > 0.0)
            & (inputs.soil_snow_temperature < tfrz)
        )

        ground_temperature = inputs.soil_snow_temperature[
            NUMBER_OF_SNOW_LEVELS:
        ]
        ground_ice = inputs.ice_water[NUMBER_OF_SNOW_LEVELS:]
        ground_liquid = inputs.liquid_water[NUMBER_OF_SNOW_LEVELS:]
        ground_processable = processable[NUMBER_OF_SNOW_LEVELS:]
        ground_melting = (
            ground_processable & (ground_ice > 0.0) & (ground_temperature > tfrz)
        )
        temperature_after_ground_melt = jnp.where(
            ground_melting,
            tfrz,
            ground_temperature,
        )
        supercool_eligible = (
            (inputs.landunit_type == LANDUNIT_SOIL)
            | (inputs.landunit_type == LANDUNIT_CROP)
            | (inputs.column_type == URBAN_PERVIOUS_ROAD)
        )
        supercooled_water = _soil_supercooled_liquid_water(
            temperature_after_ground_melt,
            inputs.layer_thickness[NUMBER_OF_SNOW_LEVELS:],
            inputs.soil_porosity,
            inputs.soil_saturated_suction,
            inputs.soil_clapp_hornberger_exponent,
            ground_processable & supercool_eligible,
        )
        ground_freezing = (
            ground_processable
            & (ground_liquid > supercooled_water)
            & (temperature_after_ground_melt < tfrz)
        )
        thin_snow_melting = (
            ground_processable
            & (ground_level[NUMBER_OF_SNOW_LEVELS:] == 1)
            & (inputs.snow_layer_count == 0)
            & (inputs.snow_water > 0.0)
            & (~ground_melting)
            & (~ground_freezing)
            & (temperature_after_ground_melt > tfrz)
        )

        identified_phase_flag = jnp.zeros(
            (NUMBER_OF_SOIL_SNOW_LEVELS,),
            dtype=jnp.int32,
        )
        identified_phase_flag = jnp.where(
            snow_melting,
            jnp.int32(1),
            identified_phase_flag,
        )
        identified_phase_flag = jnp.where(
            snow_freezing,
            jnp.int32(2),
            identified_phase_flag,
        )
        ground_flags = jnp.where(
            ground_melting | thin_snow_melting,
            jnp.int32(1),
            jnp.where(ground_freezing, jnp.int32(2), jnp.int32(0)),
        )
        identified_phase_flag = identified_phase_flag.at[
            NUMBER_OF_SNOW_LEVELS:
        ].set(ground_flags)
        phase_flag = jnp.where(
            valid,
            identified_phase_flag,
            inputs.previous_phase_flag,
        )

        phase_identified = processable & (identified_phase_flag > 0)
        temperature_increment = jnp.where(
            phase_identified,
            fortran_add(tfrz, -inputs.soil_snow_temperature),
            jnp.float64(0.0),
        )
        temperature = jnp.where(
            phase_identified,
            tfrz,
            inputs.soil_snow_temperature,
        )
        snow_layer_melt_flux = jnp.zeros(
            (NUMBER_OF_SNOW_LEVELS,),
            dtype=jnp.float64,
        )
        snow_layer_freeze_flux = jnp.zeros(
            (NUMBER_OF_SNOW_LEVELS,),
            dtype=jnp.float64,
        )
        initial_state = _SoilSnowPhaseLoopState(
            temperature=temperature,
            liquid_water=inputs.liquid_water,
            ice_water=inputs.ice_water,
            phase_flag=phase_flag,
            snow_depth=inputs.snow_depth,
            snow_water=inputs.snow_water,
            phase_change_heat_flux=jnp.float64(0.0),
            snow_melt_flux=jnp.float64(0.0),
            snow_freeze_flux=jnp.float64(0.0),
            snow_drain_flux=jnp.float64(0.0),
            snow_layer_melt_flux=snow_layer_melt_flux,
            snow_layer_freeze_flux=snow_layer_freeze_flux,
        )

        def update_level(
            index: int,
            state: _SoilSnowPhaseLoopState,
        ) -> _SoilSnowPhaseLoopState:
            fortran_level = index + FIRST_FORTRAN_LEVEL
            factor = inputs.timestep_over_heat_capacity[index]
            increment = temperature_increment[index]
            derivative_energy = fortran_multiply(
                inputs.surface_flux_temperature_derivative,
                increment,
            )
            capacity_energy = _safe_fortran_divide(increment, factor)
            top_energy = fortran_add(derivative_energy, -capacity_energy)
            top_snow_energy = fortran_multiply(
                inputs.effective_snow_fraction,
                top_energy,
            )
            top_energy = jnp.where(fortran_level > 0, top_energy, top_snow_energy)
            standing_water_correction = fortran_multiply(
                inputs.standing_water_fraction,
                derivative_energy,
            )
            top_energy = jnp.where(
                (fortran_level == 1) & (inputs.standing_water_fraction != 0.0),
                fortran_add(top_energy, -standing_water_correction),
                top_energy,
            )
            exposed_soil_fraction = fortran_add(
                fortran_add(
                    jnp.float64(1.0),
                    -inputs.effective_snow_fraction,
                ),
                -inputs.standing_water_fraction,
            )
            first_soil_energy = fortran_add(
                fortran_multiply(
                    fortran_multiply(
                        exposed_soil_fraction,
                        inputs.surface_flux_temperature_derivative,
                    ),
                    increment,
                ),
                -capacity_energy,
            )
            noninterface_energy = jnp.where(
                fortran_level < 1,
                -fortran_multiply(
                    inputs.effective_snow_fraction,
                    _safe_fortran_divide(increment, factor),
                ),
                -_safe_fortran_divide(increment, factor),
            )
            energy_residual = jnp.where(
                index == top_index,
                top_energy,
                jnp.where(
                    fortran_level == 1,
                    first_soil_energy,
                    noninterface_energy,
                ),
            )
            energy_residual = jnp.where(
                state.phase_flag[index] > 0,
                energy_residual,
                jnp.float64(0.0),
            )
            invalid_melting = (
                (state.phase_flag[index] == 1) & (energy_residual < 0.0)
            )
            invalid_freezing = (
                (state.phase_flag[index] == 2) & (energy_residual > 0.0)
            )
            invalid_phase = invalid_melting | invalid_freezing
            energy_residual = jnp.where(
                invalid_phase,
                jnp.float64(0.0),
                energy_residual,
            )
            corrected_flag = jnp.where(
                invalid_phase,
                jnp.int32(0),
                state.phase_flag[index],
            )
            state = state._replace(
                phase_flag=state.phase_flag.at[index].set(corrected_flag)
            )
            do_phase_change = (
                processable[index]
                & (corrected_flag > 0)
                & (jnp.abs(energy_residual) > 0.0)
            )

            def apply_phase_change(
                current: _SoilSnowPhaseLoopState,
            ) -> _SoilSnowPhaseLoopState:
                phase_mass = _safe_fortran_divide(
                    fortran_multiply(energy_residual, time_step_seconds),
                    hfus,
                )

                def melt_unresolved_snow(
                    values: tuple[
                        jnp.ndarray,
                        jnp.ndarray,
                        jnp.ndarray,
                        jnp.ndarray,
                        jnp.ndarray,
                        jnp.ndarray,
                    ],
                ) -> tuple[
                    jnp.ndarray,
                    jnp.ndarray,
                    jnp.ndarray,
                    jnp.ndarray,
                    jnp.ndarray,
                    jnp.ndarray,
                ]:
                    (
                        current_phase_mass,
                        current_energy,
                        current_snow_water,
                        current_snow_depth,
                        current_melt_flux,
                        current_drain_flux,
                    ) = values
                    initial_snow_water = current_snow_water
                    current_snow_water = jnp.maximum(
                        jnp.float64(0.0),
                        fortran_add(initial_snow_water, -current_phase_mass),
                    )
                    proportion = _safe_fortran_divide(
                        current_snow_water,
                        initial_snow_water,
                    )
                    current_snow_depth = fortran_multiply(
                        proportion,
                        current_snow_depth,
                    )
                    melted_snow = fortran_add(
                        initial_snow_water,
                        -current_snow_water,
                    )
                    remaining_energy = fortran_add(
                        current_energy,
                        -_safe_fortran_divide(
                            fortran_multiply(hfus, melted_snow),
                            time_step_seconds,
                        ),
                    )
                    has_remaining_energy = remaining_energy > 0.0
                    current_phase_mass = jnp.where(
                        has_remaining_energy,
                        _safe_fortran_divide(
                            fortran_multiply(
                                remaining_energy,
                                time_step_seconds,
                            ),
                            hfus,
                        ),
                        jnp.float64(0.0),
                    )
                    current_energy = jnp.where(
                        has_remaining_energy,
                        remaining_energy,
                        jnp.float64(0.0),
                    )
                    current_melt_flux = _safe_fortran_divide(
                        jnp.maximum(jnp.float64(0.0), melted_snow),
                        time_step_seconds,
                    )
                    current_drain_flux = current_melt_flux
                    return (
                        current_phase_mass,
                        current_energy,
                        current_snow_water,
                        current_snow_depth,
                        current_melt_flux,
                        current_drain_flux,
                    )

                thin_snow_phase = (
                    (fortran_level == 1)
                    & (inputs.snow_layer_count == 0)
                    & (current.snow_water > 0.0)
                    & (phase_mass > 0.0)
                )
                (
                    phase_mass,
                    adjusted_energy,
                    snow_water,
                    snow_depth,
                    thin_snow_melt_flux,
                    snow_drain_flux,
                ) = lax.cond(
                    thin_snow_phase,
                    melt_unresolved_snow,
                    lambda values: values,
                    (
                        phase_mass,
                        energy_residual,
                        current.snow_water,
                        current.snow_depth,
                        current.snow_melt_flux,
                        current.snow_drain_flux,
                    ),
                )
                phase_change_heat_flux = jnp.where(
                    thin_snow_phase,
                    fortran_multiply(hfus, thin_snow_melt_flux),
                    current.phase_change_heat_flux,
                )

                initial_ice = inputs.ice_water[index]
                initial_liquid = inputs.liquid_water[index]
                initial_water_mass = fortran_add(initial_ice, initial_liquid)

                def melt_ice(_: None) -> tuple[jnp.ndarray, jnp.ndarray]:
                    updated_ice = jnp.maximum(
                        jnp.float64(0.0),
                        fortran_add(initial_ice, -phase_mass),
                    )
                    residual = fortran_add(
                        adjusted_energy,
                        -_safe_fortran_divide(
                            fortran_multiply(
                                hfus,
                                fortran_add(initial_ice, -updated_ice),
                            ),
                            time_step_seconds,
                        ),
                    )
                    return updated_ice, residual

                def freeze_liquid(_: None) -> tuple[jnp.ndarray, jnp.ndarray]:
                    ground_index = jnp.clip(
                        index - NUMBER_OF_SNOW_LEVELS,
                        0,
                        NUMBER_OF_GROUND_LEVELS - 1,
                    )
                    soil_ice = jnp.where(
                        initial_water_mass < supercooled_water[ground_index],
                        jnp.float64(0.0),
                        jnp.minimum(
                            fortran_add(
                                initial_water_mass,
                                -supercooled_water[ground_index],
                            ),
                            fortran_add(initial_ice, -phase_mass),
                        ),
                    )
                    snow_ice = jnp.minimum(
                        initial_water_mass,
                        fortran_add(initial_ice, -phase_mass),
                    )
                    updated_ice = jnp.where(
                        fortran_level <= 0,
                        snow_ice,
                        soil_ice,
                    )
                    residual = fortran_add(
                        adjusted_energy,
                        -_safe_fortran_divide(
                            fortran_multiply(
                                hfus,
                                fortran_add(initial_ice, -updated_ice),
                            ),
                            time_step_seconds,
                        ),
                    )
                    return updated_ice, residual

                updated_ice, remaining_energy = lax.cond(
                    phase_mass > 0.0,
                    melt_ice,
                    lambda _: lax.cond(
                        phase_mass < 0.0,
                        freeze_liquid,
                        lambda __: (
                            current.ice_water[index],
                            jnp.float64(0.0),
                        ),
                        operand=None,
                    ),
                    operand=None,
                )
                updated_liquid = jnp.maximum(
                    jnp.float64(0.0),
                    fortran_add(initial_water_mass, -updated_ice),
                )
                temperature_value = current.temperature[index]

                def apply_remaining_energy(_: None) -> jnp.ndarray:
                    top_first_soil_denominator = fortran_add(
                        jnp.float64(1.0),
                        -fortran_multiply(
                            fortran_multiply(
                                fortran_add(
                                    jnp.float64(1.0),
                                    -inputs.standing_water_fraction,
                                ),
                                factor,
                            ),
                            inputs.surface_flux_temperature_derivative,
                        ),
                    )
                    top_first_soil_increment = _safe_fortran_divide(
                        fortran_multiply(factor, remaining_energy),
                        top_first_soil_denominator,
                    )
                    top_snow_denominator = fortran_add(
                        jnp.float64(1.0),
                        -fortran_multiply(
                            factor,
                            inputs.surface_flux_temperature_derivative,
                        ),
                    )
                    top_snow_increment = _safe_fortran_divide(
                        fortran_multiply(
                            _safe_fortran_divide(
                                factor,
                                inputs.effective_snow_fraction,
                            ),
                            remaining_energy,
                        ),
                        top_snow_denominator,
                    )
                    top_increment = jnp.where(
                        fortran_level == 1,
                        top_first_soil_increment,
                        top_snow_increment,
                    )
                    first_soil_denominator = fortran_add(
                        jnp.float64(1.0),
                        -fortran_multiply(
                            fortran_multiply(exposed_soil_fraction, factor),
                            inputs.surface_flux_temperature_derivative,
                        ),
                    )
                    first_soil_increment = _safe_fortran_divide(
                        fortran_multiply(factor, remaining_energy),
                        first_soil_denominator,
                    )
                    ordinary_increment = jnp.where(
                        fortran_level > 0,
                        fortran_multiply(factor, remaining_energy),
                        jnp.where(
                            inputs.effective_snow_fraction > 0.0,
                            fortran_multiply(
                                _safe_fortran_divide(
                                    factor,
                                    inputs.effective_snow_fraction,
                                ),
                                remaining_energy,
                            ),
                            jnp.float64(0.0),
                        ),
                    )
                    temperature_increment_value = jnp.where(
                        index == top_index,
                        top_increment,
                        jnp.where(
                            fortran_level == 1,
                            first_soil_increment,
                            ordinary_increment,
                        ),
                    )
                    updated_temperature = fortran_add(
                        temperature_value,
                        temperature_increment_value,
                    )
                    snow_mixture = (
                        (fortran_level <= 0)
                        & (fortran_multiply(updated_liquid, updated_ice) > 0.0)
                    )
                    return jnp.where(snow_mixture, tfrz, updated_temperature)

                temperature_value = lax.cond(
                    jnp.abs(remaining_energy) > 0.0,
                    apply_remaining_energy,
                    lambda _: temperature_value,
                    operand=None,
                )
                if linearize_latent_balance:
                    ice_from_energy = fortran_add(initial_ice, -phase_mass)
                    soil_index = jnp.clip(index - NUMBER_OF_SNOW_LEVELS, 0, NUMBER_OF_GROUND_LEVELS - 1)
                    supercooled = supercooled_water[soil_index]
                    freeze_absorbs_heat = jnp.where(
                        fortran_level <= 0,
                        ice_from_energy < initial_water_mass,
                        (initial_water_mass > supercooled)
                        & (ice_from_energy < fortran_add(initial_water_mass, -supercooled)),
                    )
                    heat_absorbed = ((phase_mass > 0.0) & (ice_from_energy > 0.0)) | (
                        (phase_mass < 0.0) & freeze_absorbs_heat
                    )
                    temperature_value = _phase_temperature_with_latent_balance(
                        temperature_value, current.temperature[index], heat_absorbed
                    )
                ice_melted = fortran_add(initial_ice, -updated_ice)
                layer_latent_flux = _safe_fortran_divide(
                    fortran_multiply(hfus, ice_melted),
                    time_step_seconds,
                )
                phase_change_heat_flux = fortran_add(
                    phase_change_heat_flux,
                    layer_latent_flux,
                )

                snow_index = jnp.clip(index, 0, NUMBER_OF_SNOW_LEVELS - 1)
                snow_layer_melt = _safe_fortran_divide(
                    jnp.maximum(jnp.float64(0.0), ice_melted),
                    time_step_seconds,
                )
                snow_layer_freeze = _safe_fortran_divide(
                    jnp.maximum(jnp.float64(0.0), -ice_melted),
                    time_step_seconds,
                )
                is_snow_melting = (corrected_flag == 1) & (fortran_level < 1)
                is_snow_freezing = (corrected_flag == 2) & (fortran_level < 1)
                snow_layer_melt_flux = current.snow_layer_melt_flux.at[
                    snow_index
                ].set(
                    jnp.where(
                        is_snow_melting,
                        snow_layer_melt,
                        current.snow_layer_melt_flux[snow_index],
                    )
                )
                snow_layer_freeze_flux = current.snow_layer_freeze_flux.at[
                    snow_index
                ].set(
                    jnp.where(
                        is_snow_freezing,
                        snow_layer_freeze,
                        current.snow_layer_freeze_flux[snow_index],
                    )
                )
                snow_melt_flux = jnp.where(
                    is_snow_melting,
                    fortran_add(thin_snow_melt_flux, snow_layer_melt),
                    thin_snow_melt_flux,
                )
                snow_freeze_flux = jnp.where(
                    is_snow_freezing,
                    fortran_add(current.snow_freeze_flux, snow_layer_freeze),
                    current.snow_freeze_flux,
                )
                return current._replace(
                    temperature=current.temperature.at[index].set(
                        temperature_value
                    ),
                    liquid_water=current.liquid_water.at[index].set(
                        updated_liquid
                    ),
                    ice_water=current.ice_water.at[index].set(updated_ice),
                    snow_depth=snow_depth,
                    snow_water=snow_water,
                    phase_change_heat_flux=phase_change_heat_flux,
                    snow_melt_flux=snow_melt_flux,
                    snow_freeze_flux=snow_freeze_flux,
                    snow_drain_flux=snow_drain_flux,
                    snow_layer_melt_flux=snow_layer_melt_flux,
                    snow_layer_freeze_flux=snow_layer_freeze_flux,
                )

            return lax.cond(
                do_phase_change,
                apply_phase_change,
                lambda unchanged: unchanged,
                state,
            )

        final_state = lax.fori_loop(
            0,
            NUMBER_OF_SOIL_SNOW_LEVELS,
            update_level,
            initial_state,
        )
        snow_melt_heat_flux = fortran_multiply(
            final_state.snow_melt_flux,
            hfus,
        )
        urban = (inputs.landunit_type >= LANDUNIT_URBAN_MIN) & (
            inputs.landunit_type <= LANDUNIT_URBAN_MAX
        )
        rural = (inputs.landunit_type == LANDUNIT_SOIL) | (
            inputs.landunit_type == LANDUNIT_CROP
        )
        rural_snow_melt_heat_flux = jnp.where(
            rural,
            snow_melt_heat_flux,
            inputs.previous_rural_snow_melt_heat_flux,
        )
        urban_snow_melt_heat_flux = jnp.where(
            urban,
            snow_melt_heat_flux,
            inputs.previous_urban_snow_melt_heat_flux,
        )
        return SoilSnowPhaseChangeResult(
            snow_depth=final_state.snow_depth,
            snow_water=final_state.snow_water,
            liquid_water=final_state.liquid_water,
            ice_water=final_state.ice_water,
            soil_snow_temperature=final_state.temperature,
            phase_flag=final_state.phase_flag,
            phase_change_heat_flux=final_state.phase_change_heat_flux,
            snow_melt_flux=final_state.snow_melt_flux,
            snow_freeze_flux=final_state.snow_freeze_flux,
            snow_drain_flux=final_state.snow_drain_flux,
            snow_layer_melt_flux=final_state.snow_layer_melt_flux,
            snow_layer_freeze_flux=final_state.snow_layer_freeze_flux,
            snow_melt_heat_flux=snow_melt_heat_flux,
            rural_snow_melt_heat_flux=rural_snow_melt_heat_flux,
            urban_snow_melt_heat_flux=urban_snow_melt_heat_flux,
        )

    return lax.cond(
        jnp.asarray(inputs.active_nonlake, dtype=jnp.bool_),
        update_active,
        lambda _: inactive_result,
        operand=None,
    )


def phase_change_soil_snow_layers(
    inputs: SoilSnowPhaseChangeInputs,
    *,
    time_step_seconds: float = 1800.0,
) -> SoilSnowPhaseChangeResult:

    _validate_soil_snow_phase_change_shapes(inputs)
    if time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")
    from clm_jax.physics.solver_scalarization import _native_tile_trace

    if jax.default_backend() == "gpu" and not _native_tile_trace.get():
        from clm_jax.physics.soil_snow_phase_change_gpu import phase_change

        return phase_change(inputs, float(time_step_seconds))
    return jax.vmap(
        lambda column_inputs: _column_soil_snow_phase_change(
            column_inputs,
            jnp.float64(time_step_seconds),
        )
    )(inputs)


class SoilTemperatureFinalDiagnosticsInputs(NamedTuple):

    active_nonlake: jnp.ndarray
    snow_layer_count: jnp.ndarray
    landunit_type: jnp.ndarray
    effective_snow_fraction: jnp.ndarray
    standing_water_fraction: jnp.ndarray
    soil_snow_temperature: jnp.ndarray
    standing_water_temperature: jnp.ndarray
    interface_heat_flux: jnp.ndarray
    updated_interface_heat_flux: jnp.ndarray
    previous_ground_temperature: jnp.ndarray
    previous_layer_one_two_heat_flux: jnp.ndarray
    previous_ground_layer_heat_flux: jnp.ndarray


class SoilTemperatureFinalDiagnosticsResult(NamedTuple):

    ground_temperature: jnp.ndarray
    layer_one_two_heat_flux: jnp.ndarray
    ground_layer_heat_flux: jnp.ndarray


def _validate_soil_temperature_final_diagnostic_shapes(
    inputs: SoilTemperatureFinalDiagnosticsInputs,
) -> None:
    if inputs.active_nonlake.ndim != 1:
        raise ValueError("active_nonlake must be a vector")
    number_of_columns = inputs.active_nonlake.shape[0]
    vector_shape = (number_of_columns,)
    layer_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS)
    ground_shape = (number_of_columns, NUMBER_OF_GROUND_LEVELS)
    layer_fields = {
        "soil_snow_temperature",
        "interface_heat_flux",
        "updated_interface_heat_flux",
    }
    for name in set(SoilTemperatureFinalDiagnosticsInputs._fields) - (
        layer_fields | {"previous_ground_layer_heat_flux"}
    ):
        if getattr(inputs, name).shape != vector_shape:
            raise ValueError(f"{name} must have shape {vector_shape}")
    for name in layer_fields:
        if getattr(inputs, name).shape != layer_shape:
            raise ValueError(f"{name} must have shape {layer_shape}")
    if inputs.previous_ground_layer_heat_flux.shape != ground_shape:
        raise ValueError(
            f"previous_ground_layer_heat_flux must have shape {ground_shape}"
        )


def _column_soil_temperature_final_diagnostics(
    inputs: SoilTemperatureFinalDiagnosticsInputs,
) -> SoilTemperatureFinalDiagnosticsResult:

    active = jnp.asarray(inputs.active_nonlake, dtype=jnp.bool_)
    first_active_index = (
        jnp.asarray(inputs.snow_layer_count, dtype=jnp.int32)
        + jnp.int32(NUMBER_OF_SNOW_LEVELS)
    )
    first_soil_temperature = inputs.soil_snow_temperature[GROUND_FIRST_INDEX]
    top_snow_temperature = inputs.soil_snow_temperature[first_active_index]
    snow_fraction = inputs.effective_snow_fraction
    water_fraction = inputs.standing_water_fraction

    snow_and_water_ground_temperature = fortran_add(
        fortran_add(
            fortran_multiply(snow_fraction, top_snow_temperature),
            fortran_multiply(
                fortran_add(
                    fortran_add(jnp.float64(1.0), -snow_fraction),
                    -water_fraction,
                ),
                first_soil_temperature,
            ),
        ),
        fortran_multiply(water_fraction, inputs.standing_water_temperature),
    )
    snow_ground_temperature = fortran_add(
        fortran_multiply(snow_fraction, top_snow_temperature),
        fortran_multiply(
            fortran_add(jnp.float64(1.0), -snow_fraction),
            first_soil_temperature,
        ),
    )
    water_ground_temperature = fortran_add(
        fortran_multiply(
            fortran_add(jnp.float64(1.0), -water_fraction),
            first_soil_temperature,
        ),
        fortran_multiply(water_fraction, inputs.standing_water_temperature),
    )
    has_snow = inputs.snow_layer_count < 0
    has_standing_water = water_fraction != 0.0
    ground_temperature = jnp.where(
        has_snow,
        jnp.where(
            has_standing_water,
            snow_and_water_ground_temperature,
            snow_ground_temperature,
        ),
        jnp.where(
            has_standing_water,
            water_ground_temperature,
            first_soil_temperature,
        ),
    )
    ground_temperature = jnp.where(
        active,
        ground_temperature,
        inputs.previous_ground_temperature,
    )

    old_weighted_flux = fortran_multiply(
        -jnp.float64(CRANK_NICHOLSON_FACTOR),
        inputs.interface_heat_flux[GROUND_FIRST_INDEX],
    )
    new_weighted_flux = fortran_multiply(
        -fortran_add(
            jnp.float64(1.0),
            -jnp.float64(CRANK_NICHOLSON_FACTOR),
        ),
        inputs.updated_interface_heat_flux[GROUND_FIRST_INDEX],
    )
    layer_one_two_heat_flux = fortran_add(old_weighted_flux, new_weighted_flux)
    layer_one_two_heat_flux = jnp.where(
        active,
        layer_one_two_heat_flux,
        inputs.previous_layer_one_two_heat_flux,
    )

    rural = (inputs.landunit_type == LANDUNIT_SOIL) | (
        inputs.landunit_type == LANDUNIT_CROP
    )
    ground_level_indices = (
        jnp.arange(NUMBER_OF_GROUND_LEVELS, dtype=jnp.int32)
        + jnp.int32(GROUND_FIRST_INDEX)
    )
    old_ground_flux = fortran_multiply(
        -jnp.float64(CRANK_NICHOLSON_FACTOR),
        inputs.interface_heat_flux[ground_level_indices],
    )
    new_ground_flux = fortran_multiply(
        -fortran_add(
            jnp.float64(1.0),
            -jnp.float64(CRANK_NICHOLSON_FACTOR),
        ),
        inputs.updated_interface_heat_flux[ground_level_indices],
    )
    ground_layer_heat_flux = fortran_add(old_ground_flux, new_ground_flux)
    bottom_ground_offset = GROUND_BOTTOM_INDEX - GROUND_FIRST_INDEX
    ground_layer_heat_flux = ground_layer_heat_flux.at[bottom_ground_offset].set(
        jnp.float64(0.0)
    )
    ground_layer_heat_flux = jnp.where(
        active & rural,
        ground_layer_heat_flux,
        inputs.previous_ground_layer_heat_flux,
    )
    return SoilTemperatureFinalDiagnosticsResult(
        ground_temperature=ground_temperature,
        layer_one_two_heat_flux=layer_one_two_heat_flux,
        ground_layer_heat_flux=ground_layer_heat_flux,
    )


def finalize_soil_temperature_diagnostics(
    inputs: SoilTemperatureFinalDiagnosticsInputs,
) -> SoilTemperatureFinalDiagnosticsResult:

    _validate_soil_temperature_final_diagnostic_shapes(inputs)
    return jax.vmap(_column_soil_temperature_final_diagnostics)(inputs)


_native_phase_change_soil_snow_layers = phase_change_soil_snow_layers


@partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def _mixed_snow_phase_transition(inputs, seconds, point_local):
    from clm_jax.physics.solver_scalarization import _native_tile_trace
    token = _native_tile_trace.set(point_local)
    try:
        return _native_phase_change_soil_snow_layers(inputs, time_step_seconds=seconds)
    finally:
        _native_tile_trace.reset(token)


@_mixed_snow_phase_transition.defjvp
def _mixed_snow_equilibrium_jvp(seconds, point_local, primals, tangents):
    inputs, = primals
    input_dot, = tangents
    native = _mixed_snow_phase_transition(inputs, seconds, point_local)
    reference = jax.vmap(lambda column: _column_soil_snow_phase_change(
        column, jnp.float64(seconds), linearize_latent_balance=True
    ))
    dot = jax.jvp(reference, primals, tangents)[1]
    return native, _mixed_snow_equilibrium_dot(inputs, seconds, input_dot, dot)


def _mixed_snow_equilibrium_dot(inputs, seconds, input_dot, dot):
    level = jnp.arange(NUMBER_OF_SOIL_SNOW_LEVELS)[None,:]
    first = inputs.snow_layer_count[:,None] + NUMBER_OF_SNOW_LEVELS
    equilibrium = (inputs.active_nonlake[:,None] & (level >= first)
        & (level < NUMBER_OF_SNOW_LEVELS)
        & (inputs.soil_snow_temperature == jnp.float64(FREEZING_TEMPERATURE))
        & (inputs.ice_water > 0.) & (inputs.liquid_water > 0.))
    temperature_dot = jnp.where(equilibrium, input_dot.soil_snow_temperature, 0.)
    factor = jnp.where(equilibrium, inputs.timestep_over_heat_capacity, 1.)
    surface_derivative = jnp.where(equilibrium & (level == first),
        inputs.surface_flux_temperature_derivative[:,None], 0.)
    fraction = jnp.where(equilibrium, inputs.effective_snow_fraction[:,None], 0.)
    capacity = jnp.maximum((1./factor - surface_derivative) * fraction, 0.)
    heat_dot = capacity * temperature_dot
    mass_dot = heat_dot * seconds / jnp.float64(LATENT_HEAT_FUSION)
    melt_dot = .5 * mass_dot[:,:NUMBER_OF_SNOW_LEVELS] / seconds
    freeze_dot = -melt_dot
    melt_flux_dot, freeze_flux_dot = jnp.sum(melt_dot,axis=1), jnp.sum(freeze_dot,axis=1)
    melt_heat_dot = melt_flux_dot * jnp.float64(LATENT_HEAT_FUSION)
    rural = (inputs.landunit_type == LANDUNIT_SOIL) | (inputs.landunit_type == LANDUNIT_CROP)
    urban = (inputs.landunit_type >= LANDUNIT_URBAN_MIN) & (inputs.landunit_type <= LANDUNIT_URBAN_MAX)
    return dot._replace(
        soil_snow_temperature=dot.soil_snow_temperature-temperature_dot,
        liquid_water=dot.liquid_water+mass_dot,
        ice_water=dot.ice_water-mass_dot,
        phase_change_heat_flux=dot.phase_change_heat_flux+jnp.sum(heat_dot,axis=1),
        snow_melt_flux=dot.snow_melt_flux+melt_flux_dot,
        snow_freeze_flux=dot.snow_freeze_flux+freeze_flux_dot,
        snow_layer_melt_flux=dot.snow_layer_melt_flux+melt_dot,
        snow_layer_freeze_flux=dot.snow_layer_freeze_flux+freeze_dot,
        snow_melt_heat_flux=dot.snow_melt_heat_flux+melt_heat_dot,
        rural_snow_melt_heat_flux=dot.rural_snow_melt_heat_flux+jnp.where(rural,melt_heat_dot,0.),
        urban_snow_melt_heat_flux=dot.urban_snow_melt_heat_flux+jnp.where(urban,melt_heat_dot,0.),
    )


def phase_change_soil_snow_layers(inputs, *, time_step_seconds=1800.):
    from clm_jax.physics.solver_scalarization import _native_tile_trace
    return _mixed_snow_phase_transition(inputs, time_step_seconds, _native_tile_trace.get())
