
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from clm_jax.constants import DENSITY_ICE, PI
from clm_jax.io.restart import (
    COLUMN_URBAN_IMPERVIOUS_ROAD,
    COLUMN_URBAN_PERVIOUS_ROAD,
    COLUMN_URBAN_ROOF,
    COLUMN_URBAN_SHADE_WALL,
    COLUMN_URBAN_SUN_WALL,
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    SubgridTopology,
)
from clm_jax.physics.native_power import native_power
from clm_jax.physics.numerics import (
    fortran_divide,
    strict_exp,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.numerics import fortran_hardware_power as fortran_power
from clm_jax.physics.pointwise import scalar_map
from clm_jax.physics.snow_water import (
    GROUND_SURFACE_INDEX,
    NUMBER_OF_SOIL_SNOW_LEVELS,
)

NUMBER_OF_GROUND_LEVELS = 25
NUMBER_OF_SOIL_LEVELS = 20
SOIL_LEVEL_STOP = GROUND_SURFACE_INDEX + NUMBER_OF_SOIL_LEVELS

SOIL_ICE_IMPEDANCE_EXPONENT = np.float64(np.float32(6.0))
SURFACE_WATER_CONNECTIVITY_THRESHOLD = np.float64(np.float32(0.4))
SURFACE_WATER_CONNECTIVITY_EXPONENT = np.float64(np.float32(0.13889))
URBAN_PONDING_DEPTH_MILLIMETERS = np.float64(1.0)


def _ice_impedance_power(exponent):
    power = native_power if jax.default_backend() == "gpu" else fortran_power
    return power(jnp.float64(10.0), exponent)


def _surface_connectivity_power(connectivity_delta):
    base = jnp.where(connectivity_delta > 0.0, connectivity_delta, 1.0)
    power = native_power if jax.default_backend() == "gpu" else fortran_power
    return power(base, jnp.float64(SURFACE_WATER_CONNECTIVITY_EXPONENT))


class SurfaceHydrologyMasks(NamedTuple):
    hydrology_column: jnp.ndarray
    soil_crop_column: jnp.ndarray
    urban_column: jnp.ndarray


class SurfaceHydrologyInputs(NamedTuple):
    time_step_seconds: jnp.ndarray
    snow_layer_count_column: jnp.ndarray
    soil_snow_thickness_column_level: jnp.ndarray
    soil_snow_ice_column_level: jnp.ndarray
    snow_to_surface_water_flux_column: jnp.ndarray
    flood_flux_column: jnp.ndarray
    ground_evaporation_flux_column: jnp.ndarray
    soil_evaporation_flux_column: jnp.ndarray
    surface_water_evaporation_flux_column: jnp.ndarray
    effective_snow_fraction_column: jnp.ndarray
    surface_water_fraction_column: jnp.ndarray
    snow_free_surface_water_fraction_column: jnp.ndarray
    water_table_depth_column: jnp.ndarray
    frost_table_depth_column: jnp.ndarray
    perched_water_table_depth_column: jnp.ndarray
    maximum_saturated_fraction_column: jnp.ndarray
    surface_water_threshold_column: jnp.ndarray
    topographic_slope_degrees_column: jnp.ndarray
    soil_porosity_column_level: jnp.ndarray
    saturated_hydraulic_conductivity_column_level: jnp.ndarray


class SurfaceHydrologyState(NamedTuple):
    soil_snow_liquid_column_level: jnp.ndarray
    surface_water_column: jnp.ndarray
    top_soil_water_flux_column: jnp.ndarray
    surface_runoff_flux_column: jnp.ndarray
    surface_water_runoff_flux_column: jnp.ndarray
    infiltration_flux_column: jnp.ndarray
    saturated_fraction_column: jnp.ndarray
    impermeable_fraction_column: jnp.ndarray
    fractional_impermeability_column_level: jnp.ndarray
    soil_ice_fraction_column_level: jnp.ndarray
    effective_porosity_column_level: jnp.ndarray


class SurfaceHydrologyDiagnostics(NamedTuple):
    masks: SurfaceHydrologyMasks
    maximum_infiltration_flux_column: jnp.ndarray
    infiltration_excess_flux_column: jnp.ndarray
    soil_input_flux_column: jnp.ndarray
    surface_water_input_flux_column: jnp.ndarray
    surface_water_drainage_flux_column: jnp.ndarray


class SurfaceHydrologyResult(NamedTuple):
    state: SurfaceHydrologyState
    diagnostics: SurfaceHydrologyDiagnostics


def build_surface_hydrology_masks(topology: SubgridTopology) -> SurfaceHydrologyMasks:

    active = jnp.asarray(topology.column.active, dtype=jnp.bool_)
    landunit_type = jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
    column_type = jnp.asarray(topology.column.column_type, dtype=jnp.int32)
    soil_crop = active & (
        (landunit_type == jnp.int32(LANDUNIT_SOIL))
        | (landunit_type == jnp.int32(LANDUNIT_CROP))
    )
    urban = active & (
        (landunit_type >= jnp.int32(LANDUNIT_URBAN_MIN))
        & (landunit_type <= jnp.int32(LANDUNIT_URBAN_MAX))
    )
    hydrology = soil_crop | (
        active & (column_type == jnp.int32(COLUMN_URBAN_PERVIOUS_ROAD))
    )
    return SurfaceHydrologyMasks(
        hydrology_column=hydrology,
        soil_crop_column=soil_crop,
        urban_column=urban,
    )


def _validate_shapes(
    topology: SubgridTopology,
    inputs: SurfaceHydrologyInputs,
    state: SurfaceHydrologyState,
) -> None:
    column_shape = topology.column.gridcell.shape
    if len(column_shape) != 1:
        raise ValueError("column topology must be a vector")
    soil_snow_shape = (column_shape[0], NUMBER_OF_SOIL_SNOW_LEVELS)
    ground_shape = (column_shape[0], NUMBER_OF_GROUND_LEVELS)
    if jnp.shape(inputs.time_step_seconds) != ():
        raise ValueError("time_step_seconds must be scalar")
    input_matrices = {
        "soil_snow_thickness_column_level": soil_snow_shape,
        "soil_snow_ice_column_level": soil_snow_shape,
        "soil_porosity_column_level": ground_shape,
        "saturated_hydraulic_conductivity_column_level": ground_shape,
    }
    for name, expected in input_matrices.items():
        if getattr(inputs, name).shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    input_vectors = set(SurfaceHydrologyInputs._fields) - {
        "time_step_seconds",
        *input_matrices,
    }
    for name in input_vectors:
        if getattr(inputs, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")
    state_matrices = {
        "soil_snow_liquid_column_level": soil_snow_shape,
        "fractional_impermeability_column_level": ground_shape,
        "soil_ice_fraction_column_level": ground_shape,
        "effective_porosity_column_level": ground_shape,
    }
    for name, expected in state_matrices.items():
        if getattr(state, name).shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    for name in set(SurfaceHydrologyState._fields) - set(state_matrices):
        if getattr(state, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")


def _scalar_sin(value: jnp.ndarray) -> jnp.ndarray:
    value = jnp.asarray(value, dtype=jnp.float64)
    shape = value.shape
    return jnp.reshape(scalar_map(jnp.sin, jnp.reshape(value, (-1,))), shape)


def _soil_ice_properties(
    inputs: SurfaceHydrologyInputs,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    soil_slice = slice(GROUND_SURFACE_INDEX, SOIL_LEVEL_STOP)
    porosity = jnp.asarray(inputs.soil_porosity_column_level[:, :NUMBER_OF_SOIL_LEVELS])
    thickness = jnp.asarray(inputs.soil_snow_thickness_column_level[:, soil_slice])
    ice = jnp.asarray(inputs.soil_snow_ice_column_level[:, soil_slice])
    ice_volume = jnp.minimum(
        porosity,
        fortran_divide(
            ice,
            fortran_multiply(thickness, jnp.float64(DENSITY_ICE)),
        ),
    )
    ice_fraction = jnp.minimum(
        jnp.float64(1.0),
        fortran_divide(ice_volume, porosity),
    )
    effective_porosity = jnp.maximum(
        jnp.float64(0.01),
        fortran_add(porosity, -ice_volume),
    )
    return ice_volume, ice_fraction, effective_porosity


def _surface_runoff(
    topology: SubgridTopology,
    inputs: SurfaceHydrologyInputs,
    state: SurfaceHydrologyState,
    masks: SurfaceHydrologyMasks,
) -> SurfaceHydrologyState:
    hydrology = masks.hydrology_column
    urban = masks.urban_column
    column_type = jnp.asarray(topology.column.column_type, dtype=jnp.int32)
    time_step = jnp.asarray(inputs.time_step_seconds, dtype=jnp.float64)

    _, ice_fraction_soil, _ = _soil_ice_properties(inputs)
    exp_minus_three = strict_exp(jnp.float64(-3.0))
    exponent = fortran_multiply(
        jnp.float64(-3.0),
        fortran_add(jnp.float64(1.0), -ice_fraction_soil),
    )
    fractional_impermeability_soil = jnp.maximum(
        jnp.float64(0.0),
        fortran_divide(
            fortran_add(strict_exp(exponent), -exp_minus_three),
            fortran_add(jnp.float64(1.0), -exp_minus_three),
        ),
    )
    fractional_impermeability = state.fractional_impermeability_column_level.at[
        :, :NUMBER_OF_SOIL_LEVELS
    ].set(
        jnp.where(
            hydrology[:, None],
            fractional_impermeability_soil,
            state.fractional_impermeability_column_level[:, :NUMBER_OF_SOIL_LEVELS],
        )
    )
    soil_ice_fraction = state.soil_ice_fraction_column_level.at[
        :, :NUMBER_OF_SOIL_LEVELS
    ].set(
        jnp.where(
            hydrology[:, None],
            ice_fraction_soil,
            state.soil_ice_fraction_column_level[:, :NUMBER_OF_SOIL_LEVELS],
        )
    )

    normal_exponent = fortran_multiply(
        fortran_multiply(jnp.float64(-0.5), jnp.float64(0.5)),
        inputs.water_table_depth_column,
    )
    normal_saturated_fraction = fortran_multiply(
        inputs.maximum_saturated_fraction_column,
        strict_exp(normal_exponent),
    )
    perched_exponent = fortran_multiply(
        fortran_multiply(jnp.float64(-0.5), jnp.float64(0.5)),
        inputs.perched_water_table_depth_column,
    )
    perched_saturated_fraction = fortran_multiply(
        inputs.maximum_saturated_fraction_column,
        strict_exp(perched_exponent),
    )
    use_perched = (
        inputs.frost_table_depth_column <= inputs.water_table_depth_column
    ) & (inputs.frost_table_depth_column > inputs.perched_water_table_depth_column)
    calculated_saturated_fraction = jnp.where(
        use_perched,
        perched_saturated_fraction,
        normal_saturated_fraction,
    )
    saturated_fraction = jnp.where(
        hydrology,
        calculated_saturated_fraction,
        state.saturated_fraction_column,
    )
    impermeable_fraction = jnp.where(
        hydrology,
        calculated_saturated_fraction,
        state.impermeable_fraction_column,
    )

    top_soil_flux_before_surface_additions = jnp.asarray(
        state.top_soil_water_flux_column,
        dtype=jnp.float64,
    )
    surface_runoff = jnp.where(
        hydrology,
        fortran_multiply(
            calculated_saturated_fraction,
            top_soil_flux_before_surface_additions,
        ),
        state.surface_runoff_flux_column,
    )
    liquid = jnp.asarray(state.soil_snow_liquid_column_level, dtype=jnp.float64)
    top_liquid = liquid[:, GROUND_SURFACE_INDEX]

    roof_or_impervious = urban & (
        (column_type == jnp.int32(COLUMN_URBAN_ROOF))
        | (column_type == jnp.int32(COLUMN_URBAN_IMPERVIOUS_ROAD))
    )
    roof_or_impervious_snow = roof_or_impervious & (
        inputs.snow_layer_count_column < jnp.int32(0)
    )
    roof_or_impervious_no_snow = roof_or_impervious & ~roof_or_impervious_snow
    ponding_rate = fortran_divide(
        jnp.float64(URBAN_PONDING_DEPTH_MILLIMETERS),
        time_step,
    )
    urban_excess = jnp.maximum(
        jnp.float64(0.0),
        fortran_add(
            fortran_add(
                fortran_add(
                    fortran_divide(top_liquid, time_step),
                    top_soil_flux_before_surface_additions,
                ),
                -inputs.ground_evaporation_flux_column,
            ),
            -ponding_rate,
        ),
    )
    stored_top_liquid = jnp.maximum(
        jnp.float64(0.0),
        fortran_add(
            top_liquid,
            fortran_multiply(
                fortran_add(
                    top_soil_flux_before_surface_additions,
                    -inputs.ground_evaporation_flux_column,
                ),
                time_step,
            ),
        ),
    )
    updated_top_liquid = jnp.where(
        urban_excess > jnp.float64(0.0),
        jnp.float64(URBAN_PONDING_DEPTH_MILLIMETERS),
        stored_top_liquid,
    )
    liquid = liquid.at[:, GROUND_SURFACE_INDEX].set(
        jnp.where(roof_or_impervious_no_snow, updated_top_liquid, top_liquid)
    )
    surface_runoff = jnp.where(
        roof_or_impervious_snow,
        jnp.maximum(jnp.float64(0.0), top_soil_flux_before_surface_additions),
        surface_runoff,
    )
    surface_runoff = jnp.where(
        roof_or_impervious_no_snow,
        urban_excess,
        surface_runoff,
    )
    wall = urban & (
        (column_type == jnp.int32(COLUMN_URBAN_SUN_WALL))
        | (column_type == jnp.int32(COLUMN_URBAN_SHADE_WALL))
    )
    surface_runoff = jnp.where(wall, jnp.float64(0.0), surface_runoff)
    surface_runoff = jnp.where(
        urban,
        fortran_add(surface_runoff, inputs.flood_flux_column),
        surface_runoff,
    )

    top_soil_flux_after_surface_additions = fortran_add(
        fortran_add(
            top_soil_flux_before_surface_additions,
            inputs.snow_to_surface_water_flux_column,
        ),
        inputs.flood_flux_column,
    )
    top_soil_flux = jnp.where(
        hydrology,
        top_soil_flux_after_surface_additions,
        top_soil_flux_before_surface_additions,
    )
    return SurfaceHydrologyState(
        soil_snow_liquid_column_level=liquid,
        surface_water_column=state.surface_water_column,
        top_soil_water_flux_column=top_soil_flux,
        surface_runoff_flux_column=surface_runoff,
        surface_water_runoff_flux_column=state.surface_water_runoff_flux_column,
        infiltration_flux_column=state.infiltration_flux_column,
        saturated_fraction_column=saturated_fraction,
        impermeable_fraction_column=impermeable_fraction,
        fractional_impermeability_column_level=fractional_impermeability,
        soil_ice_fraction_column_level=soil_ice_fraction,
        effective_porosity_column_level=state.effective_porosity_column_level,
    )


def _infiltration(
    topology: SubgridTopology,
    inputs: SurfaceHydrologyInputs,
    state: SurfaceHydrologyState,
    masks: SurfaceHydrologyMasks,
) -> SurfaceHydrologyResult:
    hydrology = masks.hydrology_column
    soil_crop = masks.soil_crop_column
    urban = masks.urban_column
    column_type = jnp.asarray(topology.column.column_type, dtype=jnp.int32)
    time_step = jnp.asarray(inputs.time_step_seconds, dtype=jnp.float64)
    _, ice_fraction_soil, effective_porosity_soil = _soil_ice_properties(inputs)

    soil_ice_fraction = state.soil_ice_fraction_column_level.at[
        :, :NUMBER_OF_SOIL_LEVELS
    ].set(
        jnp.where(
            hydrology[:, None],
            ice_fraction_soil,
            state.soil_ice_fraction_column_level[:, :NUMBER_OF_SOIL_LEVELS],
        )
    )
    effective_porosity = state.effective_porosity_column_level.at[
        :, :NUMBER_OF_SOIL_LEVELS
    ].set(
        jnp.where(
            hydrology[:, None],
            effective_porosity_soil,
            state.effective_porosity_column_level[:, :NUMBER_OF_SOIL_LEVELS],
        )
    )

    no_snow = inputs.snow_layer_count_column >= jnp.int32(0)
    effective_snow_fraction = jnp.where(
        no_snow,
        jnp.float64(0.0),
        inputs.effective_snow_fraction_column,
    )
    evaporation = jnp.where(
        no_snow,
        inputs.ground_evaporation_flux_column,
        inputs.soil_evaporation_flux_column,
    )
    net_surface_input = fortran_add(
        state.top_soil_water_flux_column,
        -state.surface_runoff_flux_column,
    )
    one_minus_surface_water = fortran_add(
        jnp.float64(1.0),
        -inputs.surface_water_fraction_column,
    )
    soil_input = fortran_multiply(one_minus_surface_water, net_surface_input)
    surface_water_input = fortran_multiply(
        inputs.surface_water_fraction_column,
        net_surface_input,
    )
    evaporation_soil_fraction = fortran_add(
        fortran_add(jnp.float64(1.0), -effective_snow_fraction),
        -inputs.surface_water_fraction_column,
    )
    soil_input = fortran_add(
        soil_input,
        -fortran_multiply(evaporation_soil_fraction, evaporation),
    )
    surface_water_input = fortran_add(
        surface_water_input,
        -fortran_multiply(
            inputs.surface_water_fraction_column,
            inputs.surface_water_evaporation_flux_column,
        ),
    )

    ice_impedance = _ice_impedance_power(
        fortran_multiply(
            jnp.float64(-SOIL_ICE_IMPEDANCE_EXPONENT),
            ice_fraction_soil[:, :3],
        ),
    )
    conductivity_with_ice = fortran_multiply(
        ice_impedance,
        inputs.saturated_hydraulic_conductivity_column_level[:, :3],
    )
    minimum_conductivity = jnp.minimum(
        jnp.minimum(conductivity_with_ice[:, 0], conductivity_with_ice[:, 1]),
        conductivity_with_ice[:, 2],
    )
    maximum_infiltration = fortran_multiply(
        fortran_add(jnp.float64(1.0), -state.saturated_fraction_column),
        minimum_conductivity,
    )
    infiltration_excess = jnp.maximum(
        jnp.float64(0.0),
        fortran_add(
            soil_input,
            -fortran_multiply(one_minus_surface_water, maximum_infiltration),
        ),
    )
    infiltration = fortran_add(soil_input, -infiltration_excess)
    surface_water_input = fortran_add(surface_water_input, infiltration_excess)

    connectivity_delta = fortran_add(
        inputs.snow_free_surface_water_fraction_column,
        -jnp.float64(SURFACE_WATER_CONNECTIVITY_THRESHOLD),
    )
    connected_fraction = jnp.where(
        inputs.snow_free_surface_water_fraction_column
        <= jnp.float64(SURFACE_WATER_CONNECTIVITY_THRESHOLD),
        jnp.float64(0.0),
        _surface_connectivity_power(connectivity_delta),
    )
    surface_water_above_threshold = fortran_add(
        state.surface_water_column,
        -inputs.surface_water_threshold_column,
    )
    slope_radians = fortran_multiply(
        fortran_divide(jnp.float64(PI), jnp.float64(180.0)),
        inputs.topographic_slope_degrees_column,
    )
    wet_reservoir_coefficient = fortran_multiply(
        jnp.float64(1.0e-4),
        _scalar_sin(slope_radians),
    )
    potential_surface_water_runoff = fortran_multiply(
        fortran_multiply(wet_reservoir_coefficient, connected_fraction),
        surface_water_above_threshold,
    )
    storage_limited_surface_water_runoff = jnp.minimum(
        potential_surface_water_runoff,
        fortran_divide(surface_water_above_threshold, time_step),
    )
    surface_water_runoff = jnp.where(
        state.surface_water_column >= inputs.surface_water_threshold_column,
        storage_limited_surface_water_runoff,
        jnp.float64(0.0),
    )
    surface_water_runoff = jnp.where(
        surface_water_runoff < jnp.float64(1.0e-8),
        jnp.float64(0.0),
        surface_water_runoff,
    )
    surface_water_input_after_runoff = fortran_add(
        surface_water_input,
        -surface_water_runoff,
    )
    surface_water_before_drainage = fortran_add(
        state.surface_water_column,
        fortran_multiply(surface_water_input_after_runoff, time_step),
    )
    evaporated_all_surface_water = surface_water_before_drainage < jnp.float64(0.0)
    infiltration = jnp.where(
        evaporated_all_surface_water,
        fortran_add(
            infiltration,
            fortran_divide(surface_water_before_drainage, time_step),
        ),
        infiltration,
    )
    nonnegative_surface_water = jnp.where(
        evaporated_all_surface_water,
        jnp.float64(0.0),
        surface_water_before_drainage,
    )
    surface_water_drainage = jnp.where(
        evaporated_all_surface_water,
        jnp.float64(0.0),
        jnp.minimum(
            fortran_multiply(
                inputs.surface_water_fraction_column,
                maximum_infiltration,
            ),
            fortran_divide(nonnegative_surface_water, time_step),
        ),
    )
    updated_surface_water = fortran_add(
        nonnegative_surface_water,
        -fortran_multiply(surface_water_drainage, time_step),
    )
    infiltration = fortran_add(infiltration, surface_water_drainage)

    final_surface_water = jnp.where(
        soil_crop,
        updated_surface_water,
        state.surface_water_column,
    )
    final_surface_water_runoff = jnp.where(
        soil_crop,
        surface_water_runoff,
        state.surface_water_runoff_flux_column,
    )
    final_infiltration = jnp.where(
        soil_crop,
        infiltration,
        state.infiltration_flux_column,
    )

    urban_pervious_no_snow_infiltration = fortran_add(
        fortran_add(
            state.top_soil_water_flux_column,
            -state.surface_runoff_flux_column,
        ),
        -inputs.ground_evaporation_flux_column,
    )
    urban_pervious_snow_infiltration = fortran_add(
        fortran_add(
            state.top_soil_water_flux_column,
            -state.surface_runoff_flux_column,
        ),
        -fortran_multiply(
            fortran_add(
                jnp.float64(1.0),
                -inputs.effective_snow_fraction_column,
            ),
            inputs.soil_evaporation_flux_column,
        ),
    )
    pervious_road = hydrology & ~soil_crop
    final_infiltration = jnp.where(
        pervious_road,
        jnp.where(
            no_snow,
            urban_pervious_no_snow_infiltration,
            urban_pervious_snow_infiltration,
        ),
        final_infiltration,
    )
    final_surface_water_runoff = jnp.where(
        pervious_road,
        jnp.float64(0.0),
        final_surface_water_runoff,
    )

    impervious_urban = urban & (
        column_type != jnp.int32(COLUMN_URBAN_PERVIOUS_ROAD)
    )
    final_infiltration = jnp.where(
        impervious_urban,
        jnp.float64(0.0),
        final_infiltration,
    )

    final_state = SurfaceHydrologyState(
        soil_snow_liquid_column_level=state.soil_snow_liquid_column_level,
        surface_water_column=final_surface_water,
        top_soil_water_flux_column=state.top_soil_water_flux_column,
        surface_runoff_flux_column=state.surface_runoff_flux_column,
        surface_water_runoff_flux_column=final_surface_water_runoff,
        infiltration_flux_column=final_infiltration,
        saturated_fraction_column=state.saturated_fraction_column,
        impermeable_fraction_column=state.impermeable_fraction_column,
        fractional_impermeability_column_level=(
            state.fractional_impermeability_column_level
        ),
        soil_ice_fraction_column_level=soil_ice_fraction,
        effective_porosity_column_level=effective_porosity,
    )
    zeros = jnp.zeros_like(state.surface_water_column)
    return SurfaceHydrologyResult(
        state=final_state,
        diagnostics=SurfaceHydrologyDiagnostics(
            masks=masks,
            maximum_infiltration_flux_column=jnp.where(
                soil_crop,
                maximum_infiltration,
                zeros,
            ),
            infiltration_excess_flux_column=jnp.where(
                soil_crop,
                infiltration_excess,
                zeros,
            ),
            soil_input_flux_column=jnp.where(soil_crop, soil_input, zeros),
            surface_water_input_flux_column=jnp.where(
                soil_crop,
                surface_water_input_after_runoff,
                zeros,
            ),
            surface_water_drainage_flux_column=jnp.where(
                soil_crop,
                surface_water_drainage,
                zeros,
            ),
        ),
    )


def surface_runoff_and_infiltration(
    topology: SubgridTopology,
    inputs: SurfaceHydrologyInputs,
    state: SurfaceHydrologyState,
) -> SurfaceHydrologyResult:

    _validate_shapes(topology, inputs, state)
    masks = build_surface_hydrology_masks(topology)
    runoff_state = _surface_runoff(topology, inputs, state, masks)
    return _infiltration(topology, inputs, runoff_state, masks)


__all__ = [
    "NUMBER_OF_GROUND_LEVELS",
    "NUMBER_OF_SOIL_LEVELS",
    "SurfaceHydrologyDiagnostics",
    "SurfaceHydrologyInputs",
    "SurfaceHydrologyMasks",
    "SurfaceHydrologyResult",
    "SurfaceHydrologyState",
    "build_surface_hydrology_masks",
    "surface_runoff_and_infiltration",
]
