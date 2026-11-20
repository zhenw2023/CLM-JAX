
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from clm_jax.constants import (
    CLM_SPECIAL_VALUE,
    DENSITY_FRESH_WATER,
    DENSITY_ICE,
    FREEZING_TEMPERATURE_FRESHWATER,
    LATENT_HEAT_SUBLIMATION,
    LATENT_HEAT_VAPORIZATION,
    WATER_VAPOR_GAS_CONSTANT_OVER_GRAVITY,
)
from clm_jax.io.restart import (
    COLUMN_URBAN_IMPERVIOUS_ROAD,
    COLUMN_URBAN_PERVIOUS_ROAD,
    COLUMN_URBAN_ROOF,
    COLUMN_URBAN_SHADE_WALL,
    COLUMN_URBAN_SUN_WALL,
    LANDUNIT_CROP,
    LANDUNIT_GLACIER,
    LANDUNIT_LAKE,
    LANDUNIT_SOIL,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    LANDUNIT_WETLAND,
    SubgridTopology,
)
from clm_jax.physics.numerics import (
    fortran_divide,
    fortran_hardware_add,
    fortran_hardware_multiply,
    fortran_linear_combination,
    fortran_ordered_sum,
    fortran_product,
    strict_exp,
)
from clm_jax.physics.positive_power import positive_power
from clm_jax.physics.qsat import qsat

NUMBER_OF_SNOW_LEVELS = 12
NUMBER_OF_SOIL_LEVELS = 20
NUMBER_OF_GROUND_LEVELS = 25
NUMBER_OF_TOTAL_LEVELS = NUMBER_OF_SNOW_LEVELS + NUMBER_OF_GROUND_LEVELS
NUMBER_OF_URBAN_LEVELS = 5

SOIL_POTENTIAL_MINIMUM = -1.0e8
SOIL_ROUGHNESS_LENGTH = 0.01
SNOW_ROUGHNESS_LENGTH = 0.0024


class CanopyTemperatureInputs(NamedTuple):

    forcing_height_wind_gridcell: jnp.ndarray
    forcing_height_temperature_gridcell: jnp.ndarray
    forcing_height_humidity_gridcell: jnp.ndarray
    pressure_column: jnp.ndarray
    specific_humidity_column: jnp.ndarray
    temperature_column: jnp.ndarray
    potential_temperature_column: jnp.ndarray
    snow_layer_count_column: jnp.ndarray
    soil_snow_temperature_column_level: jnp.ndarray
    surface_water_temperature_column: jnp.ndarray
    surface_water_fraction_column: jnp.ndarray
    effective_snow_fraction_column: jnp.ndarray
    snow_fraction_column: jnp.ndarray
    soil_liquid_water_column_level: jnp.ndarray
    soil_ice_water_column_level: jnp.ndarray
    vegetation_exposed_patch: jnp.ndarray
    exposed_leaf_area_index_patch: jnp.ndarray
    exposed_stem_area_index_patch: jnp.ndarray
    canopy_top_height_patch: jnp.ndarray


class CanopyTemperatureParameters(NamedTuple):

    ground_layer_thickness_column: jnp.ndarray
    saturated_water_content_column: jnp.ndarray
    clapp_hornberger_b_column: jnp.ndarray
    saturated_suction_column: jnp.ndarray
    dry_water_content_column: jnp.ndarray
    optimal_water_content_column: jnp.ndarray
    saturated_hydraulic_conductivity_column: jnp.ndarray
    pervious_road_root_fraction_column: jnp.ndarray
    momentum_roughness_ratio_patch: jnp.ndarray
    displacement_ratio_patch: jnp.ndarray
    urban_roughness_length_landunit: jnp.ndarray
    urban_displacement_height_landunit: jnp.ndarray


class CanopyTemperatureColumnState(NamedTuple):
    temperature_before_column_level: jnp.ndarray
    surface_water_temperature_before_column: jnp.ndarray
    dry_surface_layer_thickness_column: jnp.ndarray
    soil_evaporation_resistance_column: jnp.ndarray
    ground_temperature_column: jnp.ndarray
    snow_surface_specific_humidity_column: jnp.ndarray
    soil_surface_specific_humidity_column: jnp.ndarray
    surface_water_specific_humidity_column: jnp.ndarray
    ground_specific_humidity_column: jnp.ndarray
    ground_specific_humidity_temperature_derivative_column: jnp.ndarray
    soil_evaporation_alpha_column: jnp.ndarray
    urban_soil_evaporation_alpha_column: jnp.ndarray
    pervious_road_root_resistance_column: jnp.ndarray
    latent_heat_column: jnp.ndarray
    ground_emissivity_column: jnp.ndarray
    ground_momentum_roughness_column: jnp.ndarray
    ground_sensible_roughness_column: jnp.ndarray
    ground_latent_roughness_column: jnp.ndarray
    convective_velocity_coefficient_column: jnp.ndarray
    convective_boundary_height_column: jnp.ndarray
    virtual_potential_temperature_column: jnp.ndarray


class CanopyTemperaturePatchState(NamedTuple):
    sensible_heat_total_patch: jnp.ndarray
    rural_sensible_heat_total_patch: jnp.ndarray
    urban_sensible_heat_total_patch: jnp.ndarray
    latent_heat_total_patch: jnp.ndarray
    rural_latent_heat_total_patch: jnp.ndarray
    urban_latent_heat_total_patch: jnp.ndarray
    vegetation_sensible_heat_patch: jnp.ndarray
    total_evaporation_patch: jnp.ndarray
    vegetation_evaporation_patch: jnp.ndarray
    vegetation_transpiration_patch: jnp.ndarray
    ground_energy_derivative_patch: jnp.ndarray
    ground_sensible_derivative_patch: jnp.ndarray
    ground_latent_derivative_patch: jnp.ndarray
    vegetation_emissivity_patch: jnp.ndarray
    momentum_roughness_patch: jnp.ndarray
    displacement_height_patch: jnp.ndarray
    vegetation_momentum_roughness_patch: jnp.ndarray
    vegetation_sensible_roughness_patch: jnp.ndarray
    vegetation_latent_roughness_patch: jnp.ndarray
    forcing_height_wind_patch: jnp.ndarray
    forcing_height_temperature_patch: jnp.ndarray
    forcing_height_humidity_patch: jnp.ndarray
    reference_temperature_patch: jnp.ndarray


class CanopyTemperatureState(NamedTuple):
    column: CanopyTemperatureColumnState
    patch: CanopyTemperaturePatchState


def _add(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    left, right = jnp.broadcast_arrays(left, right)
    terms = jnp.stack((left, right), axis=0)
    return fortran_linear_combination(terms, jnp.ones_like(terms))


def _subtract(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return _add(left, -right)


def _multiply(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return lax.optimization_barrier(fortran_product(left, right))


def _soil_surface_relative_humidity(liquid, ice, thickness, porosity, suction, b, temperature):
    volume = fortran_divide(
        fortran_hardware_add(
            fortran_divide(liquid, jnp.float64(DENSITY_FRESH_WATER)),
            fortran_divide(ice, jnp.float64(DENSITY_ICE)),
        ),
        thickness,
    )
    wetness = jnp.clip(fortran_divide(volume, porosity), jnp.float64(0.01), jnp.float64(1.0))
    potential = jnp.maximum(
        jnp.float64(SOIL_POTENTIAL_MINIMUM),
        fortran_hardware_multiply(-suction, positive_power(wetness, -b)),
    )
    return strict_exp(fortran_divide(
        fortran_divide(potential, jnp.float64(WATER_VAPOR_GAS_CONSTANT_OVER_GRAVITY)),
        temperature,
    ))


def _urban_mask(landunit_type: jnp.ndarray) -> jnp.ndarray:
    return (landunit_type >= jnp.int32(LANDUNIT_URBAN_MIN)) & (
        landunit_type <= jnp.int32(LANDUNIT_URBAN_MAX)
    )


def _where_update(
    mask: jnp.ndarray,
    value: jnp.ndarray,
    previous: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.where(mask, value, previous)


def _validate_shapes(
    topology: SubgridTopology,
    inputs: CanopyTemperatureInputs,
    parameters: CanopyTemperatureParameters,
    state: CanopyTemperatureState,
) -> None:
    ng = topology.grid.longitude_degrees.shape[0]
    nl = topology.landunit.gridcell.shape[0]
    nc = topology.column.gridcell.shape[0]
    npatch = topology.patch.gridcell.shape[0]
    for name in (
        "forcing_height_wind_gridcell",
        "forcing_height_temperature_gridcell",
        "forcing_height_humidity_gridcell",
    ):
        if getattr(inputs, name).shape != (ng,):
            raise ValueError(f"{name} must have shape (gridcell,)")
    for name in (
        "pressure_column",
        "specific_humidity_column",
        "temperature_column",
        "potential_temperature_column",
        "snow_layer_count_column",
        "surface_water_temperature_column",
        "surface_water_fraction_column",
        "effective_snow_fraction_column",
        "snow_fraction_column",
    ):
        if getattr(inputs, name).shape != (nc,):
            raise ValueError(f"{name} must have shape (column,)")
    for name in (
        "soil_snow_temperature_column_level",
        "soil_liquid_water_column_level",
        "soil_ice_water_column_level",
    ):
        if getattr(inputs, name).shape != (nc, NUMBER_OF_TOTAL_LEVELS):
            raise ValueError(f"{name} must have shape (column, {NUMBER_OF_TOTAL_LEVELS})")
    for name in (
        "vegetation_exposed_patch",
        "exposed_leaf_area_index_patch",
        "exposed_stem_area_index_patch",
        "canopy_top_height_patch",
    ):
        if getattr(inputs, name).shape != (npatch,):
            raise ValueError(f"{name} must have shape (pft,)")
    for name in CanopyTemperatureParameters._fields[:7]:
        if getattr(parameters, name).shape != (nc, NUMBER_OF_GROUND_LEVELS):
            raise ValueError(f"{name} must have shape (column, {NUMBER_OF_GROUND_LEVELS})")
    if parameters.pervious_road_root_fraction_column.shape != (
        nc,
        NUMBER_OF_SOIL_LEVELS,
    ):
        raise ValueError("pervious road root fraction must have shape (column, 20)")
    for name in ("momentum_roughness_ratio_patch", "displacement_ratio_patch"):
        if getattr(parameters, name).shape != (npatch,):
            raise ValueError(f"{name} must have shape (pft,)")
    for name in ("urban_roughness_length_landunit", "urban_displacement_height_landunit"):
        if getattr(parameters, name).shape != (nl,):
            raise ValueError(f"{name} must have shape (landunit,)")
    for name, value in zip(CanopyTemperatureColumnState._fields, state.column, strict=True):
        expected = (
            (nc, NUMBER_OF_TOTAL_LEVELS)
            if name == "temperature_before_column_level"
            else (nc, NUMBER_OF_SOIL_LEVELS)
            if name == "pervious_road_root_resistance_column"
            else (nc,)
        )
        if value.shape != expected:
            raise ValueError(f"column state {name} must have shape {expected}")
    for name, value in zip(CanopyTemperaturePatchState._fields, state.patch, strict=True):
        if value.shape != (npatch,):
            raise ValueError(f"patch state {name} must have shape (pft,)")


def soil_surface_resistance_swenson_lawrence_2014(
    soil_mask: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    saturated_water_content: jnp.ndarray,
    clapp_hornberger_b: jnp.ndarray,
    saturated_suction: jnp.ndarray,
    soil_temperature: jnp.ndarray,
    liquid_water: jnp.ndarray,
    ice_water: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    dz = jnp.where(soil_mask, layer_thickness, jnp.float64(1.0))
    watsat = jnp.where(soil_mask, saturated_water_content, jnp.float64(0.5))
    bsw = jnp.where(soil_mask, clapp_hornberger_b, jnp.float64(4.0))
    sucsat = jnp.where(soil_mask, saturated_suction, jnp.float64(100.0))
    temperature = jnp.where(soil_mask, soil_temperature, jnp.float64(273.15))
    liquid = jnp.where(soil_mask, liquid_water, jnp.float64(1.0))
    ice = jnp.where(soil_mask, ice_water, jnp.float64(0.0))

    vwc_liquid = fortran_divide(
        jnp.maximum(liquid, jnp.float64(1.0e-6)),
        _multiply(dz, jnp.float64(DENSITY_FRESH_WATER)),
    )
    ice_porosity = fortran_divide(
        ice, _multiply(dz, jnp.float64(DENSITY_ICE)),
    )
    effective_porosity_top = jnp.maximum(
        jnp.float64(0.01),
        _subtract(watsat, jnp.minimum(watsat, ice_porosity)),
    )
    air_dry_water = _multiply(
        watsat,
        positive_power(
            fortran_divide(sucsat, jnp.float64(1.0e7)),
            fortran_divide(jnp.float64(1.0), bsw),
        ),
    )
    default_real_diffusivity = jnp.float64(2.119999953720253e-5)
    default_real_reference_temperature = jnp.float64(273.149993896484375)
    default_real_eight_tenths = jnp.float64(0.800000011920929)
    free_air_diffusivity = _multiply(
        default_real_diffusivity,
        positive_power(
            fortran_divide(temperature, default_real_reference_temperature),
            jnp.float64(1.75),
        ),
    )
    air_space = _subtract(watsat, air_dry_water)
    soil_diffusivity = _multiply(
        _multiply(air_space, free_air_diffusivity),
        positive_power(
            fortran_divide(air_space, watsat),
            fortran_divide(jnp.float64(3.0), jnp.maximum(jnp.float64(3.0), bsw)),
        ),
    )
    dry_layer = fortran_divide(
        _multiply(
            jnp.float64(15.0),
            jnp.maximum(
                jnp.float64(0.001),
                _subtract(
                    _multiply(default_real_eight_tenths, effective_porosity_top),
                    vwc_liquid,
                ),
            ),
        ),
        jnp.maximum(
            jnp.float64(0.001),
            _subtract(_multiply(default_real_eight_tenths, watsat), air_dry_water),
        ),
    )
    dry_layer = jnp.minimum(jnp.float64(200.0), jnp.maximum(dry_layer, 0.0))
    resistance_denominator = _multiply(_multiply(soil_diffusivity, air_space), jnp.float64(1.0e3))
    resistance = _add(
        fortran_divide(dry_layer, resistance_denominator),
        jnp.float64(20.0),
    )
    resistance = jnp.minimum(jnp.float64(1.0e6), resistance)
    return dry_layer, resistance


def canopy_temperature(
    topology: SubgridTopology,
    inputs: CanopyTemperatureInputs,
    parameters: CanopyTemperatureParameters,
    state: CanopyTemperatureState,
) -> CanopyTemperatureState:

    inputs = CanopyTemperatureInputs(
        *(
            jnp.asarray(value, dtype=jnp.int32)
            if name in ("snow_layer_count_column", "vegetation_exposed_patch")
            else jnp.asarray(value, dtype=jnp.float64)
            for name, value in zip(CanopyTemperatureInputs._fields, inputs, strict=True)
        )
    )
    parameters = CanopyTemperatureParameters(
        *(jnp.asarray(value, dtype=jnp.float64) for value in parameters)
    )
    state = CanopyTemperatureState(
        column=CanopyTemperatureColumnState(
            *(jnp.asarray(value, dtype=jnp.float64) for value in state.column)
        ),
        patch=CanopyTemperaturePatchState(
            *(jnp.asarray(value, dtype=jnp.float64) for value in state.patch)
        ),
    )
    _validate_shapes(topology, inputs, parameters, state)

    column_active = jnp.asarray(topology.column.active, dtype=jnp.bool_)
    column_landunit_type = jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
    column_type = jnp.asarray(topology.column.column_type, dtype=jnp.int32)
    nonlake_column = column_active & (column_landunit_type != LANDUNIT_LAKE)
    soil_column = nonlake_column & (
        (column_landunit_type == LANDUNIT_SOIL) | (column_landunit_type == LANDUNIT_CROP)
    )
    wet_or_glacier = nonlake_column & (
        (column_landunit_type == LANDUNIT_WETLAND) | (column_landunit_type == LANDUNIT_GLACIER)
    )
    urban_column = nonlake_column & _urban_mask(column_landunit_type)
    pervious_road = urban_column & (column_type == COLUMN_URBAN_PERVIOUS_ROAD)
    wall = urban_column & (
        (column_type == COLUMN_URBAN_SUN_WALL) | (column_type == COLUMN_URBAN_SHADE_WALL)
    )
    sealed_urban = urban_column & (
        (column_type == COLUMN_URBAN_ROOF) | (column_type == COLUMN_URBAN_IMPERVIOUS_ROAD)
    )

    levels = jnp.arange(
        -NUMBER_OF_SNOW_LEVELS + 1,
        NUMBER_OF_GROUND_LEVELS + 1,
        dtype=jnp.int32,
    )
    urban_structure = urban_column & (
        (column_type == COLUMN_URBAN_ROOF)
        | (column_type == COLUMN_URBAN_SUN_WALL)
        | (column_type == COLUMN_URBAN_SHADE_WALL)
    )
    before = jnp.where(
        urban_structure[:, None] & (levels[None, :] > NUMBER_OF_URBAN_LEVELS),
        jnp.float64(CLM_SPECIAL_VALUE),
        inputs.soil_snow_temperature_column_level,
    )
    before = _where_update(
        nonlake_column[:, None],
        before,
        state.column.temperature_before_column_level,
    )
    surface_water_before = _where_update(
        nonlake_column,
        inputs.surface_water_temperature_column,
        state.column.surface_water_temperature_before_column,
    )

    soil_index = NUMBER_OF_SNOW_LEVELS
    top_layer_index = jnp.clip(
        inputs.snow_layer_count_column + jnp.int32(NUMBER_OF_SNOW_LEVELS),
        jnp.int32(0),
        jnp.int32(NUMBER_OF_TOTAL_LEVELS - 1),
    )
    row = jnp.arange(column_active.shape[0], dtype=jnp.int32)
    top_temperature = inputs.soil_snow_temperature_column_level[row, top_layer_index]
    top_liquid = inputs.soil_liquid_water_column_level[row, top_layer_index]
    top_ice = inputs.soil_ice_water_column_level[row, top_layer_index]
    soil_temperature = inputs.soil_snow_temperature_column_level[:, soil_index]
    soil_liquid = inputs.soil_liquid_water_column_level[:, soil_index]
    soil_ice = inputs.soil_ice_water_column_level[:, soil_index]
    dz_top = parameters.ground_layer_thickness_column[:, 0]

    dry_layer_new, soil_resistance_new = soil_surface_resistance_swenson_lawrence_2014(
        soil_column,
        dz_top,
        parameters.saturated_water_content_column[:, 0],
        parameters.clapp_hornberger_b_column[:, 0],
        parameters.saturated_suction_column[:, 0],
        soil_temperature,
        soil_liquid,
        soil_ice,
    )
    dry_layer = _where_update(
        soil_column,
        dry_layer_new,
        state.column.dry_surface_layer_thickness_column,
    )
    soil_resistance_value = jnp.where(
        soil_column,
        soil_resistance_new,
        jnp.where(wet_or_glacier, jnp.float64(0.0), jnp.float64(1.0e6)),
    )
    soil_resistance = _where_update(
        nonlake_column,
        soil_resistance_value,
        state.column.soil_evaporation_resistance_column,
    )

    snow_ground_temperature = fortran_linear_combination(
        jnp.stack(
            (
                top_temperature,
                soil_temperature,
                inputs.surface_water_temperature_column,
            ),
            axis=0,
        ),
        jnp.stack(
            (
                inputs.effective_snow_fraction_column,
                jnp.float64(1.0)
                - inputs.effective_snow_fraction_column
                - inputs.surface_water_fraction_column,
                inputs.surface_water_fraction_column,
            ),
            axis=0,
        ),
    )
    no_snow_ground_temperature = fortran_linear_combination(
        jnp.stack((soil_temperature, inputs.surface_water_temperature_column), axis=0),
        jnp.stack(
            (
                jnp.float64(1.0) - inputs.surface_water_fraction_column,
                inputs.surface_water_fraction_column,
            ),
            axis=0,
        ),
    )
    ground_temperature_new = jnp.where(
        inputs.snow_layer_count_column < 0,
        snow_ground_temperature,
        no_snow_ground_temperature,
    )
    ground_temperature = _where_update(
        nonlake_column,
        ground_temperature_new,
        state.column.ground_temperature_column,
    )

    safe_dz = jnp.where(soil_column, dz_top, jnp.float64(1.0))
    safe_watsat = jnp.where(
        soil_column,
        parameters.saturated_water_content_column[:, 0],
        jnp.float64(0.5),
    )
    safe_sucsat = jnp.where(
        soil_column,
        parameters.saturated_suction_column[:, 0],
        jnp.float64(100.0),
    )
    safe_bsw = jnp.where(
        soil_column,
        parameters.clapp_hornberger_b_column[:, 0],
        jnp.float64(4.0),
    )
    safe_soil_temperature = jnp.where(soil_column, soil_temperature, jnp.float64(273.15))
    relative_humidity = _soil_surface_relative_humidity(
        jnp.where(soil_column, soil_liquid, jnp.float64(0.0)),
        jnp.where(soil_column, soil_ice, jnp.float64(0.0)),
        safe_dz, safe_watsat, safe_sucsat, safe_bsw, safe_soil_temperature,
    )
    soil_qred = fortran_linear_combination(
        jnp.stack(
            (
                relative_humidity,
                jnp.ones_like(relative_humidity),
                jnp.ones_like(relative_humidity),
            ),
            axis=0,
        ),
        jnp.stack(
            (
                jnp.float64(1.0)
                - inputs.snow_fraction_column
                - inputs.surface_water_fraction_column,
                inputs.snow_fraction_column,
                inputs.surface_water_fraction_column,
            ),
            axis=0,
        ),
    )

    soil_slice = slice(soil_index, soil_index + NUMBER_OF_SOIL_LEVELS)
    road_temperature = inputs.soil_snow_temperature_column_level[:, soil_slice]
    road_liquid = inputs.soil_liquid_water_column_level[:, soil_slice]
    road_ice = inputs.soil_ice_water_column_level[:, soil_slice]
    road_dz = parameters.ground_layer_thickness_column[:, :NUMBER_OF_SOIL_LEVELS]
    road_watsat = parameters.saturated_water_content_column[:, :NUMBER_OF_SOIL_LEVELS]
    road_watdry = parameters.dry_water_content_column[:, :NUMBER_OF_SOIL_LEVELS]
    road_watopt = parameters.optimal_water_content_column[:, :NUMBER_OF_SOIL_LEVELS]
    safe_road_dz = jnp.where(pervious_road[:, None], road_dz, jnp.float64(1.0))
    safe_road_watsat = jnp.where(pervious_road[:, None], road_watsat, jnp.float64(0.5))
    safe_road_watdry = jnp.where(pervious_road[:, None], road_watdry, jnp.float64(0.1))
    safe_road_watopt = jnp.where(pervious_road[:, None], road_watopt, jnp.float64(0.3))
    road_ice_volume = jnp.minimum(
        safe_road_watsat,
        fortran_divide(
            jnp.where(pervious_road[:, None], road_ice, jnp.float64(0.0)),
            _multiply(safe_road_dz, jnp.float64(DENSITY_ICE)),
        ),
    )
    effective_road_porosity = _subtract(safe_road_watsat, road_ice_volume)
    road_liquid_volume = jnp.minimum(
        effective_road_porosity,
        fortran_divide(
            jnp.where(pervious_road[:, None], road_liquid, jnp.float64(0.0)),
            _multiply(safe_road_dz, jnp.float64(DENSITY_FRESH_WATER)),
        ),
    )
    road_fac = jnp.minimum(
        fortran_divide(
            jnp.maximum(_subtract(road_liquid_volume, safe_road_watdry), jnp.float64(0.0)),
            _subtract(safe_road_watopt, safe_road_watdry),
        ),
        jnp.float64(1.0),
    )
    road_fac = jnp.where(
        road_temperature >= jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
        road_fac,
        jnp.float64(0.0),
    )
    road_root_resistance_raw = _multiply(parameters.pervious_road_root_fraction_column, road_fac)
    road_humidity = fortran_ordered_sum(road_root_resistance_raw.T)
    road_qred = _add(
        _multiply(
            _subtract(jnp.float64(1.0), inputs.snow_fraction_column),
            road_humidity,
        ),
        inputs.snow_fraction_column,
    )
    normalized_road_root = jnp.where(
        road_humidity[:, None] > jnp.float64(0.0),
        fortran_divide(
            road_root_resistance_raw,
            jnp.where(
                road_humidity[:, None] > 0.0,
                road_humidity[:, None],
                jnp.float64(1.0),
            ),
        ),
        road_root_resistance_raw,
    )
    road_root_resistance = _where_update(
        pervious_road[:, None],
        normalized_road_root,
        state.column.pervious_road_root_resistance_column,
    )

    qred = jnp.ones_like(ground_temperature)
    qred = jnp.where(soil_column, soil_qred, qred)
    qred = jnp.where(pervious_road, road_qred, qred)
    qred = jnp.where(wall, jnp.float64(0.0), qred)
    qred = jnp.where(sealed_urban, jnp.float64(1.0), qred)
    soil_alpha = _where_update(
        soil_column,
        soil_qred,
        state.column.soil_evaporation_alpha_column,
    )
    soil_alpha = _where_update(
        wet_or_glacier,
        jnp.float64(CLM_SPECIAL_VALUE),
        soil_alpha,
    )
    urban_soil_alpha = _where_update(
        pervious_road,
        road_qred,
        state.column.urban_soil_evaporation_alpha_column,
    )
    urban_soil_alpha = _where_update(
        wall | sealed_urban,
        jnp.float64(CLM_SPECIAL_VALUE),
        urban_soil_alpha,
    )

    safe_pressure = jnp.where(nonlake_column, inputs.pressure_column, jnp.float64(101325.0))
    snow_saturation = qsat(
        jnp.where(nonlake_column, top_temperature, jnp.float64(273.15)),
        safe_pressure,
    )
    soil_saturation = qsat(safe_soil_temperature, safe_pressure)
    surface_water_saturation = qsat(
        jnp.where(
            nonlake_column,
            inputs.surface_water_temperature_column,
            jnp.float64(273.15),
        ),
        safe_pressure,
    )
    ground_saturation = qsat(
        jnp.where(nonlake_column, ground_temperature_new, jnp.float64(273.15)),
        safe_pressure,
    )

    snow_q = snow_saturation.specific_humidity
    snow_dq = snow_saturation.specific_humidity_temperature_derivative
    impossible_snow_clip = (snow_q > inputs.specific_humidity_column) & (
        inputs.specific_humidity_column > snow_q
    )
    snow_q = jnp.where(impossible_snow_clip, inputs.specific_humidity_column, snow_q)
    snow_dq = jnp.where(impossible_snow_clip, jnp.float64(0.0), snow_dq)

    soil_qsat = soil_saturation.specific_humidity
    soil_dq = soil_saturation.specific_humidity_temperature_derivative
    soil_clip = (soil_qsat > inputs.specific_humidity_column) & (
        inputs.specific_humidity_column > relative_humidity * soil_qsat
    )
    soil_qsat = jnp.where(soil_clip, inputs.specific_humidity_column, soil_qsat)
    soil_dq = jnp.where(soil_clip, jnp.float64(0.0), soil_dq)
    soil_surface_q = relative_humidity * soil_qsat
    exposed_fraction = _subtract(
        _subtract(jnp.float64(1.0), inputs.snow_fraction_column),
        inputs.surface_water_fraction_column,
    )
    combined_dq = fortran_hardware_add(
        fortran_hardware_multiply(inputs.snow_fraction_column, snow_dq),
        fortran_hardware_multiply(
            fortran_hardware_multiply(exposed_fraction, relative_humidity), soil_dq,
        ),
    )
    no_snow_layers = inputs.snow_layer_count_column >= 0
    snow_q = jnp.where(no_snow_layers, soil_surface_q, snow_q)
    combined_dq = jnp.where(
        no_snow_layers,
        fortran_hardware_multiply(
            fortran_hardware_multiply(
                _subtract(jnp.float64(1.0), inputs.surface_water_fraction_column),
                relative_humidity,
            ),
            combined_dq,
        ),
        combined_dq,
    )

    surface_water_q = surface_water_saturation.specific_humidity
    surface_water_dq = surface_water_saturation.specific_humidity_temperature_derivative
    impossible_water_clip = (surface_water_q > inputs.specific_humidity_column) & (
        inputs.specific_humidity_column > surface_water_q
    )
    surface_water_q = jnp.where(
        impossible_water_clip, inputs.specific_humidity_column, surface_water_q
    )
    surface_water_dq = jnp.where(impossible_water_clip, jnp.float64(0.0), surface_water_dq)
    combined_dq = fortran_hardware_add(
        combined_dq,
        fortran_hardware_multiply(inputs.surface_water_fraction_column, surface_water_dq),
    )
    exposed_soil_fraction = _subtract(
        _subtract(jnp.float64(1.0), inputs.effective_snow_fraction_column),
        inputs.surface_water_fraction_column,
    )
    soil_ground_q = fortran_linear_combination(
        jnp.stack((snow_q, soil_surface_q, surface_water_q), axis=0),
        jnp.stack(
            (
                inputs.effective_snow_fraction_column,
                exposed_soil_fraction,
                inputs.surface_water_fraction_column,
            ),
            axis=0,
        ),
    )

    general_qsat = ground_saturation.specific_humidity
    general_qsat_dq = ground_saturation.specific_humidity_temperature_derivative
    general_q = _multiply(qred, general_qsat)
    general_dq = _multiply(qred, general_qsat_dq)
    general_clip = (general_qsat > inputs.specific_humidity_column) & (
        inputs.specific_humidity_column > general_q
    )
    general_q = jnp.where(general_clip, inputs.specific_humidity_column, general_q)
    general_dq = jnp.where(general_clip, jnp.float64(0.0), general_dq)

    qg_snow_new = jnp.where(soil_column, snow_q, general_q)
    qg_soil_new = jnp.where(soil_column, soil_surface_q, general_q)
    qg_surface_water_new = jnp.where(soil_column, surface_water_q, general_q)
    qg_new = jnp.where(soil_column, soil_ground_q, general_q)
    dqg_new = jnp.where(soil_column, combined_dq, general_dq)
    qg_snow = _where_update(
        nonlake_column,
        qg_snow_new,
        state.column.snow_surface_specific_humidity_column,
    )
    qg_soil = _where_update(
        nonlake_column,
        qg_soil_new,
        state.column.soil_surface_specific_humidity_column,
    )
    qg_surface_water = _where_update(
        nonlake_column,
        qg_surface_water_new,
        state.column.surface_water_specific_humidity_column,
    )
    qg = _where_update(
        nonlake_column,
        qg_new,
        state.column.ground_specific_humidity_column,
    )
    dqg = _where_update(
        nonlake_column,
        dqg_new,
        state.column.ground_specific_humidity_temperature_derivative_column,
    )

    nonurban_emissivity = jnp.where(
        column_landunit_type == LANDUNIT_GLACIER,
        jnp.float64(0.97),
        fortran_hardware_add(
            fortran_hardware_multiply(
                fortran_hardware_add(jnp.float64(1.0), -inputs.snow_fraction_column),
                jnp.float64(0.96),
            ),
            fortran_hardware_multiply(inputs.snow_fraction_column, jnp.float64(0.97)),
        ),
    )
    ground_emissivity = _where_update(
        nonlake_column & ~_urban_mask(column_landunit_type),
        nonurban_emissivity,
        state.column.ground_emissivity_column,
    )
    latent_heat_new = jnp.where(
        (top_liquid <= jnp.float64(0.0)) & (top_ice > jnp.float64(0.0)),
        jnp.float64(LATENT_HEAT_SUBLIMATION),
        jnp.float64(LATENT_HEAT_VAPORIZATION),
    )
    latent_heat = _where_update(
        nonlake_column,
        latent_heat_new,
        state.column.latent_heat_column,
    )
    ground_roughness_new = jnp.where(
        inputs.snow_fraction_column > jnp.float64(0.0),
        jnp.float64(SNOW_ROUGHNESS_LENGTH),
        jnp.float64(SOIL_ROUGHNESS_LENGTH),
    )
    ground_momentum_roughness = _where_update(
        nonlake_column,
        ground_roughness_new,
        state.column.ground_momentum_roughness_column,
    )
    ground_sensible_roughness = _where_update(
        nonlake_column,
        ground_roughness_new,
        state.column.ground_sensible_roughness_column,
    )
    ground_latent_roughness = _where_update(
        nonlake_column,
        ground_roughness_new,
        state.column.ground_latent_roughness_column,
    )
    beta = _where_update(
        nonlake_column,
        jnp.float64(1.0),
        state.column.convective_velocity_coefficient_column,
    )
    boundary_height = _where_update(
        nonlake_column,
        jnp.float64(1000.0),
        state.column.convective_boundary_height_column,
    )
    virtual_potential_temperature = _where_update(
        nonlake_column,
        inputs.potential_temperature_column
        * (jnp.float64(1.0) + jnp.float64(0.61) * inputs.specific_humidity_column),
        state.column.virtual_potential_temperature_column,
    )

    patch_active = jnp.asarray(topology.patch.active, dtype=jnp.bool_)
    patch_landunit_type = jnp.asarray(topology.patch.landunit_type, dtype=jnp.int32)
    nonlake_patch = patch_active & (patch_landunit_type != LANDUNIT_LAKE)
    urban_patch = nonlake_patch & _urban_mask(patch_landunit_type)
    rural_patch = nonlake_patch & (
        (patch_landunit_type == LANDUNIT_SOIL) | (patch_landunit_type == LANDUNIT_CROP)
    )
    momentum_roughness_new = (
        parameters.momentum_roughness_ratio_patch * inputs.canopy_top_height_patch
    )
    displacement_new = parameters.displacement_ratio_patch * inputs.canopy_top_height_patch
    momentum_roughness = _where_update(
        nonlake_patch,
        momentum_roughness_new,
        state.patch.momentum_roughness_patch,
    )
    displacement = _where_update(
        nonlake_patch,
        displacement_new,
        state.patch.displacement_height_patch,
    )
    zero = jnp.float64(0.0)
    exposed_area = _add(
        inputs.exposed_leaf_area_index_patch,
        inputs.exposed_stem_area_index_patch,
    )
    vegetation_emissivity_new = _subtract(jnp.float64(1.0), strict_exp(-exposed_area))

    patch_column = jnp.asarray(topology.patch.column, dtype=jnp.int32)
    patch_landunit = jnp.asarray(topology.patch.landunit, dtype=jnp.int32)
    patch_gridcell = jnp.asarray(topology.patch.gridcell, dtype=jnp.int32)
    nc = column_active.shape[0]
    nl = topology.landunit.gridcell.shape[0]
    ng = topology.grid.longitude_degrees.shape[0]
    safe_column = jnp.clip(patch_column, 0, max(nc - 1, 0))
    safe_landunit = jnp.clip(patch_landunit, 0, max(nl - 1, 0))
    safe_gridcell = jnp.clip(patch_gridcell, 0, max(ng - 1, 0))
    bare_ground = inputs.vegetation_exposed_patch == 0
    wet_glacier_patch = patch_active & (
        (patch_landunit_type == LANDUNIT_WETLAND) | (patch_landunit_type == LANDUNIT_GLACIER)
    )
    lake_patch = patch_active & (patch_landunit_type == LANDUNIT_LAKE)

    def assign_patch_forcing_height(gridcell_height: jnp.ndarray) -> jnp.ndarray:
        base = gridcell_height[safe_gridcell]
        rural_bare = _add(_add(base, ground_momentum_roughness[safe_column]), displacement)
        rural_vegetated = _add(_add(base, momentum_roughness), displacement)
        rural_height = jnp.where(bare_ground, rural_bare, rural_vegetated)
        wet_glacier_height = _add(base, ground_momentum_roughness[safe_column])
        urban_height = _add(
            _add(
                base,
                parameters.urban_roughness_length_landunit[safe_landunit],
            ),
            parameters.urban_displacement_height_landunit[safe_landunit],
        )
        result = base
        result = jnp.where(rural_patch, rural_height, result)
        result = jnp.where(wet_glacier_patch, wet_glacier_height, result)
        result = jnp.where(lake_patch, base, result)
        return jnp.where(urban_patch, urban_height, result)

    forcing_height_wind_new = assign_patch_forcing_height(inputs.forcing_height_wind_gridcell)
    forcing_height_temperature_new = assign_patch_forcing_height(
        inputs.forcing_height_temperature_gridcell
    )
    forcing_height_humidity_new = assign_patch_forcing_height(
        inputs.forcing_height_humidity_gridcell
    )
    forcing_height_wind = _where_update(
        patch_active,
        forcing_height_wind_new,
        state.patch.forcing_height_wind_patch,
    )
    forcing_height_temperature = _where_update(
        patch_active,
        forcing_height_temperature_new,
        state.patch.forcing_height_temperature_patch,
    )
    forcing_height_humidity = _where_update(
        patch_active,
        forcing_height_humidity_new,
        state.patch.forcing_height_humidity_patch,
    )
    reference_temperature_new = (
        inputs.temperature_column[safe_column] + jnp.float64(0.0098) * forcing_height_temperature
    )
    reference_temperature = _where_update(
        nonlake_patch,
        reference_temperature_new,
        state.patch.reference_temperature_patch,
    )

    def reset_nonlake(previous: jnp.ndarray) -> jnp.ndarray:
        return _where_update(nonlake_patch, zero, previous)

    patch_state = CanopyTemperaturePatchState(
        sensible_heat_total_patch=reset_nonlake(state.patch.sensible_heat_total_patch),
        rural_sensible_heat_total_patch=_where_update(
            rural_patch, zero, state.patch.rural_sensible_heat_total_patch
        ),
        urban_sensible_heat_total_patch=_where_update(
            urban_patch, zero, state.patch.urban_sensible_heat_total_patch
        ),
        latent_heat_total_patch=reset_nonlake(state.patch.latent_heat_total_patch),
        rural_latent_heat_total_patch=_where_update(
            rural_patch, zero, state.patch.rural_latent_heat_total_patch
        ),
        urban_latent_heat_total_patch=_where_update(
            urban_patch, zero, state.patch.urban_latent_heat_total_patch
        ),
        vegetation_sensible_heat_patch=reset_nonlake(state.patch.vegetation_sensible_heat_patch),
        total_evaporation_patch=reset_nonlake(state.patch.total_evaporation_patch),
        vegetation_evaporation_patch=reset_nonlake(state.patch.vegetation_evaporation_patch),
        vegetation_transpiration_patch=reset_nonlake(state.patch.vegetation_transpiration_patch),
        ground_energy_derivative_patch=reset_nonlake(state.patch.ground_energy_derivative_patch),
        ground_sensible_derivative_patch=reset_nonlake(
            state.patch.ground_sensible_derivative_patch
        ),
        ground_latent_derivative_patch=reset_nonlake(state.patch.ground_latent_derivative_patch),
        vegetation_emissivity_patch=_where_update(
            nonlake_patch,
            vegetation_emissivity_new,
            state.patch.vegetation_emissivity_patch,
        ),
        momentum_roughness_patch=momentum_roughness,
        displacement_height_patch=displacement,
        vegetation_momentum_roughness_patch=_where_update(
            nonlake_patch,
            momentum_roughness,
            state.patch.vegetation_momentum_roughness_patch,
        ),
        vegetation_sensible_roughness_patch=_where_update(
            nonlake_patch,
            momentum_roughness,
            state.patch.vegetation_sensible_roughness_patch,
        ),
        vegetation_latent_roughness_patch=_where_update(
            nonlake_patch,
            momentum_roughness,
            state.patch.vegetation_latent_roughness_patch,
        ),
        forcing_height_wind_patch=forcing_height_wind,
        forcing_height_temperature_patch=forcing_height_temperature,
        forcing_height_humidity_patch=forcing_height_humidity,
        reference_temperature_patch=reference_temperature,
    )
    column_state = CanopyTemperatureColumnState(
        temperature_before_column_level=before,
        surface_water_temperature_before_column=surface_water_before,
        dry_surface_layer_thickness_column=dry_layer,
        soil_evaporation_resistance_column=soil_resistance,
        ground_temperature_column=ground_temperature,
        snow_surface_specific_humidity_column=qg_snow,
        soil_surface_specific_humidity_column=qg_soil,
        surface_water_specific_humidity_column=qg_surface_water,
        ground_specific_humidity_column=qg,
        ground_specific_humidity_temperature_derivative_column=dqg,
        soil_evaporation_alpha_column=soil_alpha,
        urban_soil_evaporation_alpha_column=urban_soil_alpha,
        pervious_road_root_resistance_column=road_root_resistance,
        latent_heat_column=latent_heat,
        ground_emissivity_column=ground_emissivity,
        ground_momentum_roughness_column=ground_momentum_roughness,
        ground_sensible_roughness_column=ground_sensible_roughness,
        ground_latent_roughness_column=ground_latent_roughness,
        convective_velocity_coefficient_column=beta,
        convective_boundary_height_column=boundary_height,
        virtual_potential_temperature_column=virtual_potential_temperature,
    )
    return CanopyTemperatureState(column=column_state, patch=patch_state)
