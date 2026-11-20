
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from clm_jax.constants import (
    CLM_SPECIAL_VALUE,
    DENSITY_FRESH_WATER,
    DENSITY_ICE,
    PI,
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
    LANDUNIT_WETLAND,
    SubgridTopology,
)
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
    fortran_power,
)
from clm_jax.physics.snow_water import (
    GROUND_SURFACE_INDEX,
    NUMBER_OF_GROUND_LEVELS,
    NUMBER_OF_SNOW_LEVELS,
    NUMBER_OF_SOIL_SNOW_LEVELS,
)
from clm_jax.physics.subgrid import patch_to_column_average
from clm_jax.physics.surface_hydrology import (
    NUMBER_OF_SOIL_LEVELS,
    build_surface_hydrology_masks,
)

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase


SOIL_ICE_IMPEDANCE_EXPONENT = np.float64(np.float32(6.0))
MINIMUM_SOIL_WATER_MILLIMETERS = np.float64(0.01)
MAXIMUM_PONDING_MILLIMETERS = np.float64(0.0)
AQUIFER_WATER_BASELINE_MILLIMETERS = np.float64(5000.0)
PERCHED_MAXIMUM_DRAINAGE_SCALE = np.float64(1.0e-5)
MINIMUM_SPECIFIC_YIELD = np.float64(0.02)
MAXIMUM_WATER_TABLE_DEPTH_METERS = np.float64(80.0)


class HydrologyDrainageParameters(NamedTuple):

    time_step_seconds: float = 1800.0
    baseflow_scalar: float = 0.001
    original_hydrology_flag: int = 0
    maximum_patches_per_column: int = 79

    @classmethod
    def from_case(cls, case: CimeCase) -> HydrologyDrainageParameters:

        from clm_jax.io.namelist import read_case_namelist_group  

        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        drainage = read_case_namelist_group(
            case,
            "lnd_in",
            "soilhydrology_inparm",
        )
        movement = read_case_namelist_group(
            case,
            "lnd_in",
            "soilwater_movement_inparm",
        )
        lower_boundary = int(movement.get("lower_boundary_condition", 4))
        if lower_boundary in (3, 4):
            raise ValueError(
                "CLM-JAX active drainage requires the configured non-aquifer lower boundary"
            )
        original_flag = int(drainage.get("origflag", 0))
        if original_flag != 0:
            raise ValueError("CLM-JAX active drainage requires origflag=0")
        return cls(
            time_step_seconds=float(general["dtime"]),
            baseflow_scalar=float(drainage.get("baseflow_scalar", 1.0e-2)),
            original_hydrology_flag=original_flag,
        )


DEFAULT_HYDROLOGY_DRAINAGE_PARAMETERS = HydrologyDrainageParameters()


def _drainage_power(base, exponent):
    if jax.default_backend() == "gpu":
        from clm_jax.physics.native_power import native_power
        return native_power(base, exponent)
    return fortran_power(base, exponent)


class HydrologyDrainageInputs(NamedTuple):

    soil_layer_count_column: jnp.ndarray
    snow_layer_count_column: jnp.ndarray
    layer_thickness_ground_column_level: jnp.ndarray
    lower_interface_depth_soil_column_level: jnp.ndarray
    topographic_slope_degrees_column: jnp.ndarray
    saturated_water_content_column_level: jnp.ndarray
    saturated_hydraulic_conductivity_column_level: jnp.ndarray
    saturated_suction_column_level: jnp.ndarray
    clapp_hornberger_exponent_column_level: jnp.ndarray
    effective_porosity_column_level: jnp.ndarray
    snow_water_equivalent_column: jnp.ndarray
    canopy_water_patch: jnp.ndarray
    canopy_snow_patch: jnp.ndarray
    total_plant_stored_water_column: jnp.ndarray
    aquifer_water_column: jnp.ndarray
    rainfall_flux_column: jnp.ndarray
    snowfall_flux_column: jnp.ndarray
    flood_flux_gridcell: jnp.ndarray
    total_evaporation_flux_column: jnp.ndarray
    snow_capping_ice_flux_column: jnp.ndarray
    snow_capping_liquid_flux_column: jnp.ndarray
    snow_capping_discarded_ice_flux_column: jnp.ndarray
    snow_capping_discarded_liquid_flux_column: jnp.ndarray
    beginning_water_mass_column: jnp.ndarray
    irrigation_flux_column: jnp.ndarray
    do_surface_mass_balance_column: jnp.ndarray
    dynamic_runoff_routing_gridcell: jnp.ndarray
    glacier_ice_growth_flux_column: jnp.ndarray
    glacier_ice_melt_flux_column: jnp.ndarray


class HydrologyDrainageState(NamedTuple):

    soil_snow_liquid_water_column_level: jnp.ndarray
    soil_snow_ice_water_column_level: jnp.ndarray
    surface_water_column: jnp.ndarray
    water_table_depth_column: jnp.ndarray
    perched_water_table_depth_column: jnp.ndarray
    frost_table_depth_column: jnp.ndarray
    soil_ice_fraction_column_level: jnp.ndarray
    volumetric_water_column_level: jnp.ndarray
    ending_water_mass_column: jnp.ndarray
    perched_drainage_flux_column: jnp.ndarray
    saturated_excess_flux_column: jnp.ndarray
    subsurface_drainage_flux_column: jnp.ndarray
    surface_runoff_flux_column: jnp.ndarray
    surface_water_runoff_flux_column: jnp.ndarray
    infiltration_flux_column: jnp.ndarray
    wetland_glacier_runoff_flux_column: jnp.ndarray
    total_runoff_flux_column: jnp.ndarray
    urban_runoff_flux_column: jnp.ndarray
    rural_runoff_flux_column: jnp.ndarray
    excess_soil_ice_runoff_flux_column: jnp.ndarray
    snow_capping_ice_runoff_flux_column: jnp.ndarray


class HydrologyDrainageDiagnostics(NamedTuple):
    nonlake_column: jnp.ndarray
    hydrology_column: jnp.ndarray
    urban_column: jnp.ndarray
    wetland_or_glacier_column: jnp.ndarray
    do_surface_mass_balance_column: jnp.ndarray
    frost_interface_found_column: jnp.ndarray
    perched_interface_found_column: jnp.ndarray
    frost_level_column: jnp.ndarray
    perched_level_column: jnp.ndarray
    layer_above_water_table_column: jnp.ndarray
    topographic_subsurface_runoff_flux_column: jnp.ndarray
    liquid_water_mass_column: jnp.ndarray
    ice_water_mass_column: jnp.ndarray


class HydrologyDrainageResult(NamedTuple):
    state: HydrologyDrainageState
    diagnostics: HydrologyDrainageDiagnostics


def _validate_shapes(
    topology: SubgridTopology,
    inputs: HydrologyDrainageInputs,
    state: HydrologyDrainageState,
    parameters: HydrologyDrainageParameters,
) -> None:
    column_shape = topology.column.gridcell.shape
    patch_shape = topology.patch.gridcell.shape
    gridcell_shape = topology.grid.longitude_degrees.shape
    if len(column_shape) != 1 or len(patch_shape) != 1 or len(gridcell_shape) != 1:
        raise ValueError("native gridcell, column and patch topology must be vectors")
    number_of_columns = column_shape[0]
    ground_shape = (number_of_columns, NUMBER_OF_GROUND_LEVELS)
    soil_shape = (number_of_columns, NUMBER_OF_SOIL_LEVELS)
    soil_snow_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS)

    input_shapes = {
        "soil_layer_count_column": column_shape,
        "snow_layer_count_column": column_shape,
        "layer_thickness_ground_column_level": ground_shape,
        "lower_interface_depth_soil_column_level": soil_shape,
        "topographic_slope_degrees_column": column_shape,
        "saturated_water_content_column_level": ground_shape,
        "saturated_hydraulic_conductivity_column_level": ground_shape,
        "saturated_suction_column_level": ground_shape,
        "clapp_hornberger_exponent_column_level": ground_shape,
        "effective_porosity_column_level": ground_shape,
        "snow_water_equivalent_column": column_shape,
        "canopy_water_patch": patch_shape,
        "canopy_snow_patch": patch_shape,
        "total_plant_stored_water_column": column_shape,
        "aquifer_water_column": column_shape,
        "rainfall_flux_column": column_shape,
        "snowfall_flux_column": column_shape,
        "flood_flux_gridcell": gridcell_shape,
        "total_evaporation_flux_column": column_shape,
        "snow_capping_ice_flux_column": column_shape,
        "snow_capping_liquid_flux_column": column_shape,
        "snow_capping_discarded_ice_flux_column": column_shape,
        "snow_capping_discarded_liquid_flux_column": column_shape,
        "beginning_water_mass_column": column_shape,
        "irrigation_flux_column": column_shape,
        "do_surface_mass_balance_column": column_shape,
        "dynamic_runoff_routing_gridcell": gridcell_shape,
        "glacier_ice_growth_flux_column": column_shape,
        "glacier_ice_melt_flux_column": column_shape,
    }
    for name, expected in input_shapes.items():
        if getattr(inputs, name).shape != expected:
            raise ValueError(f"{name} must have shape {expected}")

    state_matrix_shapes = {
        "soil_snow_liquid_water_column_level": soil_snow_shape,
        "soil_snow_ice_water_column_level": soil_snow_shape,
        "soil_ice_fraction_column_level": ground_shape,
        "volumetric_water_column_level": ground_shape,
    }
    for name, expected in state_matrix_shapes.items():
        if getattr(state, name).shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    for name in set(HydrologyDrainageState._fields) - set(state_matrix_shapes):
        if getattr(state, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")

    if parameters.time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")
    if parameters.baseflow_scalar < 0.0:
        raise ValueError("baseflow_scalar must be nonnegative")
    if parameters.original_hydrology_flag != 0:
        raise ValueError("the configured active branch requires origflag=0")
    if parameters.maximum_patches_per_column < 1:
        raise ValueError("maximum_patches_per_column must be positive")


def _soil_ice_fraction(
    ice_water: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    saturated_water_content: jnp.ndarray,
) -> jnp.ndarray:
    ice_volume = jnp.minimum(
        saturated_water_content,
        fortran_divide(
            ice_water,
            fortran_multiply(layer_thickness, jnp.float64(DENSITY_ICE)),
        ),
    )
    return jnp.minimum(
        jnp.float64(1.0),
        fortran_divide(ice_volume, saturated_water_content),
    )


def _find_interface_level(
    depth: jnp.ndarray,
    lower_interface_depth: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    upper = jnp.concatenate((jnp.zeros((1,), dtype=jnp.float64), lower_interface_depth[:-1]))

    def visit(level: int, carry: tuple[jnp.ndarray, jnp.ndarray]):
        selected, found = carry
        contains = (depth >= upper[level]) & (depth <= lower_interface_depth[level])
        choose = contains & ~found
        return jnp.where(choose, jnp.int32(level), selected), found | contains

    return lax.fori_loop(
        0,
        NUMBER_OF_SOIL_LEVELS,
        visit,
        (jnp.int32(0), jnp.bool_(False)),
    )


def _water_table_increment(water, storage_fraction):
    return fortran_divide(
        fortran_divide(water, storage_fraction), jnp.float64(1000.0),
    )


def _perched_lateral_flow_column(
    active: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    lower_interface_depth: jnp.ndarray,
    slope_degrees: jnp.ndarray,
    saturated_water_content: jnp.ndarray,
    saturated_hydraulic_conductivity: jnp.ndarray,
    saturated_suction: jnp.ndarray,
    clapp_hornberger_exponent: jnp.ndarray,
    ice_fraction: jnp.ndarray,
    frost_table_depth: jnp.ndarray,
    water_table_depth: jnp.ndarray,
    perched_water_table_depth: jnp.ndarray,
    liquid_water: jnp.ndarray,
    previous_flux: jnp.ndarray,
    time_step_seconds: jnp.ndarray,
) -> tuple[jnp.ndarray, ...]:

    def execute(_):
        frost_level, frost_found = _find_interface_level(
            frost_table_depth,
            lower_interface_depth,
        )
        perched_level, perched_found = _find_interface_level(
            perched_water_table_depth,
            lower_interface_depth,
        )
        radians = fortran_multiply(
            fortran_divide(jnp.float64(PI), jnp.float64(180.0)),
            slope_degrees,
        )
        maximum_flux = fortran_multiply(
            jnp.float64(PERCHED_MAXIMUM_DRAINAGE_SCALE),
            jnp.sin(radians),
        )
        thickness_millimeters = fortran_multiply(
            layer_thickness,
            jnp.float64(1000.0),
        )

        def accumulate(level: int, carry: tuple[jnp.ndarray, jnp.ndarray]):
            conductivity_sum, thickness_sum = carry
            included = (level >= perched_level) & (level <= frost_level)
            next_level = jnp.minimum(level + 1, NUMBER_OF_SOIL_LEVELS - 1)
            mean_ice = fortran_multiply(
                jnp.float64(0.5),
                fortran_add(ice_fraction[level], ice_fraction[next_level]),
            )
            impedance = _drainage_power(
                jnp.float64(10.0),
                fortran_multiply(
                    -jnp.float64(SOIL_ICE_IMPEDANCE_EXPONENT),
                    mean_ice,
                ),
            )
            conductivity_term = fortran_multiply(
                fortran_multiply(
                    impedance,
                    saturated_hydraulic_conductivity[level],
                ),
                thickness_millimeters[level],
            )
            return (
                jnp.where(
                    included,
                    fortran_add(conductivity_sum, conductivity_term),
                    conductivity_sum,
                ),
                jnp.where(
                    included,
                    fortran_add(thickness_sum, thickness_millimeters[level]),
                    thickness_sum,
                ),
            )

        conductivity_sum, thickness_sum = lax.fori_loop(
            0,
            NUMBER_OF_SOIL_LEVELS,
            accumulate,
            (jnp.float64(0.0), jnp.float64(0.0)),
        )
        mean_conductivity = jnp.where(
            thickness_sum > jnp.float64(0.0),
            fortran_divide(conductivity_sum, thickness_sum),
            conductivity_sum,
        )
        flux = fortran_multiply(
            fortran_multiply(maximum_flux, mean_conductivity),
            fortran_add(frost_table_depth, -perched_water_table_depth),
        )
        exists = (
            (frost_table_depth > perched_water_table_depth)
            & frost_found
            & perched_found
            & (frost_level > perched_level)
        )
        flux = jnp.where(exists, flux, jnp.float64(0.0))
        drainage_total = -fortran_multiply(flux, time_step_seconds)

        def remove(level: int, carry):
            current_liquid, current_perched, remaining, done = carry
            included = exists & (level > perched_level) & (level <= frost_level) & ~done
            drainage_layer = jnp.maximum(
                remaining,
                -(current_liquid[level] - jnp.float64(MINIMUM_SOIL_WATER_MILLIMETERS)),
            )
            drainage_layer = jnp.minimum(drainage_layer, jnp.float64(0.0))
            updated_remaining = fortran_add(remaining, -drainage_layer)
            updated_liquid = current_liquid.at[level].set(
                jnp.where(
                    included,
                    fortran_add(current_liquid[level], drainage_layer),
                    current_liquid[level],
                )
            )
            suction_ratio = fortran_add(
                jnp.float64(1.0),
                fortran_divide(
                    fortran_multiply(jnp.float64(1000.0), water_table_depth),
                    saturated_suction[level],
                ),
            )
            inverse_exponent = fortran_divide(
                -jnp.float64(1.0),
                clapp_hornberger_exponent[level],
            )
            specific_yield = fortran_multiply(
                saturated_water_content[level],
                fortran_add(
                    jnp.float64(1.0),
                    -_drainage_power(suction_ratio, inverse_exponent),
                ),
            )
            specific_yield = jnp.maximum(
                specific_yield,
                jnp.float64(MINIMUM_SPECIFIC_YIELD),
            )
            exhausted = included & (updated_remaining >= jnp.float64(0.0))
            depleted_depth = lower_interface_depth[level]
            partial_depth = fortran_add(
                current_perched,
                -_water_table_increment(drainage_layer, specific_yield),
            )
            updated_perched = jnp.where(
                exhausted,
                partial_depth,
                jnp.where(included, depleted_depth, current_perched),
            )
            return (
                updated_liquid,
                updated_perched,
                jnp.where(included, updated_remaining, remaining),
                done | exhausted,
            )

        liquid, perched_depth, residual, _ = lax.fori_loop(
            0,
            NUMBER_OF_SOIL_LEVELS,
            remove,
            (
                liquid_water,
                perched_water_table_depth,
                drainage_total,
                jnp.bool_(False),
            ),
        )
        adjusted_flux = fortran_add(
            flux,
            fortran_divide(residual, time_step_seconds),
        )
        adjusted_flux = jnp.where(exists, adjusted_flux, jnp.float64(0.0))
        return (
            liquid,
            perched_depth,
            adjusted_flux,
            frost_level,
            perched_level,
            frost_found,
            perched_found,
        )

    def preserve(_):
        return (
            liquid_water,
            perched_water_table_depth,
            previous_flux,
            jnp.int32(-1),
            jnp.int32(-1),
            jnp.bool_(False),
            jnp.bool_(False),
        )

    return lax.cond(active, execute, preserve, operand=None)


def _lateral_flow_power_law_column(
    active: jnp.ndarray,
    urban: jnp.ndarray,
    urban_nonpervious: jnp.ndarray,
    soil_layer_count: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    lower_interface_depth: jnp.ndarray,
    slope_degrees: jnp.ndarray,
    saturated_water_content: jnp.ndarray,
    saturated_suction: jnp.ndarray,
    clapp_hornberger_exponent: jnp.ndarray,
    effective_porosity: jnp.ndarray,
    ice_fraction: jnp.ndarray,
    water_table_depth: jnp.ndarray,
    liquid_water: jnp.ndarray,
    ice_water: jnp.ndarray,
    surface_water: jnp.ndarray,
    snow_capping_liquid_flux: jnp.ndarray,
    previous_drainage_flux: jnp.ndarray,
    previous_saturated_excess_flux: jnp.ndarray,
    previous_qrgwl_flux: jnp.ndarray,
    previous_ice_excess_flux: jnp.ndarray,
    time_step_seconds: jnp.ndarray,
    baseflow_scalar: jnp.ndarray,
) -> tuple[jnp.ndarray, ...]:

    def execute(_):
        last_soil_level = jnp.int32(NUMBER_OF_SOIL_LEVELS - 1)
        bedrock_count = jnp.clip(
            jnp.asarray(soil_layer_count, dtype=jnp.int32),
            1,
            NUMBER_OF_SOIL_LEVELS,
        )
        bedrock_level = bedrock_count - jnp.int32(1)
        thickness_millimeters = fortran_multiply(
            layer_thickness,
            jnp.float64(1000.0),
        )

        def find_water_table(level: int, carry):
            layer_above, found = carry
            contains = water_table_depth <= lower_interface_depth[level]
            choose = contains & ~found
            return (
                jnp.where(choose, jnp.int32(level), layer_above),
                found | contains,
            )

        layer_above_water_table, _ = lax.fori_loop(
            0,
            NUMBER_OF_SOIL_LEVELS,
            find_water_table,
            (jnp.int32(NUMBER_OF_SOIL_LEVELS), jnp.bool_(False)),
        )

        def sum_ice(level: int, carry):
            thickness_sum, ice_sum = carry
            start = jnp.maximum(layer_above_water_table, jnp.int32(1)) - jnp.int32(1)
            included = level >= start
            return (
                jnp.where(
                    included,
                    fortran_add(thickness_sum, thickness_millimeters[level]),
                    thickness_sum,
                ),
                jnp.where(
                    included,
                    fortran_add(
                        ice_sum,
                        fortran_multiply(
                            ice_fraction[level],
                            thickness_millimeters[level],
                        ),
                    ),
                    ice_sum,
                ),
            )

        thickness_sum, ice_sum = lax.fori_loop(
            0,
            NUMBER_OF_SOIL_LEVELS,
            sum_ice,
            (jnp.float64(0.0), jnp.float64(0.0)),
        )
        impedance = _drainage_power(
            jnp.float64(10.0),
            fortran_multiply(
                -jnp.float64(SOIL_ICE_IMPEDANCE_EXPONENT),
                fortran_divide(ice_sum, thickness_sum),
            ),
        )
        radians = fortran_multiply(
            fortran_divide(jnp.float64(PI), jnp.float64(180.0)),
            slope_degrees,
        )
        saturated_depth = fortran_add(
            lower_interface_depth[bedrock_level],
            -water_table_depth,
        )
        topographic_flux = fortran_multiply(
            fortran_multiply(
                fortran_multiply(impedance, baseflow_scalar),
                jnp.tan(radians),
            ),
            saturated_depth,
        )
        topographic_flux = jnp.where(
            water_table_depth <= lower_interface_depth[bedrock_level],
            topographic_flux,
            jnp.float64(0.0),
        )
        drainage_total = -fortran_multiply(topographic_flux, time_step_seconds)

        def remove_topographic(level: int, carry):
            current_liquid, current_water_table, remaining, done = carry
            included = (level >= layer_above_water_table) & (level < bedrock_count) & ~done
            suction_ratio = fortran_add(
                jnp.float64(1.0),
                fortran_divide(
                    fortran_multiply(jnp.float64(1000.0), current_water_table),
                    saturated_suction[level],
                ),
            )
            inverse_exponent = fortran_divide(
                -jnp.float64(1.0),
                clapp_hornberger_exponent[level],
            )
            specific_yield = fortran_multiply(
                saturated_water_content[level],
                fortran_add(
                    jnp.float64(1.0),
                    -_drainage_power(suction_ratio, inverse_exponent),
                ),
            )
            specific_yield = jnp.maximum(
                specific_yield,
                jnp.float64(MINIMUM_SPECIFIC_YIELD),
            )
            available = fortran_multiply(
                fortran_multiply(
                    specific_yield,
                    fortran_add(
                        lower_interface_depth[level],
                        -current_water_table,
                    ),
                ),
                jnp.float64(1000.0),
            )
            drainage_layer = jnp.maximum(remaining, -available)
            drainage_layer = jnp.minimum(drainage_layer, jnp.float64(0.0))
            updated_remaining = fortran_add(remaining, -drainage_layer)
            updated_liquid = current_liquid.at[level].set(
                jnp.where(
                    included,
                    fortran_add(current_liquid[level], drainage_layer),
                    current_liquid[level],
                )
            )
            exhausted = included & (updated_remaining >= jnp.float64(0.0))
            partial_depth = fortran_add(
                current_water_table,
                -_water_table_increment(drainage_layer, specific_yield),
            )
            updated_water_table = jnp.where(
                exhausted,
                partial_depth,
                jnp.where(
                    included,
                    lower_interface_depth[level],
                    current_water_table,
                ),
            )
            return (
                updated_liquid,
                updated_water_table,
                jnp.where(included, updated_remaining, remaining),
                done | exhausted,
            )

        liquid, water_table, residual, _ = lax.fori_loop(
            0,
            NUMBER_OF_SOIL_LEVELS,
            remove_topographic,
            (
                liquid_water,
                water_table_depth,
                drainage_total,
                jnp.bool_(False),
            ),
        )
        topographic_flux = fortran_add(
            topographic_flux,
            -fortran_divide(residual, time_step_seconds),
        )
        water_table = jnp.clip(
            water_table,
            jnp.float64(0.0),
            jnp.float64(MAXIMUM_WATER_TABLE_DEPTH_METERS),
        )

        def move_excess(offset: int, current_liquid: jnp.ndarray):
            level = last_soil_level - jnp.int32(offset)
            executes = level >= jnp.int32(1)
            capacity = fortran_multiply(
                effective_porosity[level],
                thickness_millimeters[level],
            )
            excess = jnp.maximum(
                fortran_add(current_liquid[level], -capacity),
                jnp.float64(0.0),
            )
            level_water = jnp.minimum(capacity, current_liquid[level])
            current_liquid = current_liquid.at[level].set(
                jnp.where(executes, level_water, current_liquid[level])
            )
            current_liquid = current_liquid.at[level - 1].set(
                jnp.where(
                    executes,
                    fortran_add(current_liquid[level - 1], excess),
                    current_liquid[level - 1],
                )
            )
            return current_liquid

        liquid = lax.fori_loop(
            0,
            NUMBER_OF_SOIL_LEVELS - 1,
            move_excess,
            liquid,
        )

        top_capacity = fortran_add(
            fortran_multiply(
                saturated_water_content[0],
                thickness_millimeters[0],
            ),
            -ice_water[0],
        )
        liquid_above_minimum = jnp.maximum(
            fortran_add(
                liquid[0],
                -jnp.float64(MINIMUM_SOIL_WATER_MILLIMETERS),
            ),
            jnp.float64(0.0),
        )
        allowed_above_minimum = jnp.maximum(
            fortran_add(
                fortran_add(
                    jnp.float64(MAXIMUM_PONDING_MILLIMETERS),
                    top_capacity,
                ),
                -jnp.float64(MINIMUM_SOIL_WATER_MILLIMETERS),
            ),
            jnp.float64(0.0),
        )
        liquid_excess = jnp.maximum(
            fortran_add(liquid_above_minimum, -allowed_above_minimum),
            jnp.float64(0.0),
        )
        liquid = liquid.at[0].set(fortran_add(liquid[0], -liquid_excess))
        saturated_excess = jnp.where(
            urban,
            fortran_divide(liquid_excess, time_step_seconds),
            jnp.float64(0.0),
        )
        surface = jnp.where(
            urban,
            surface_water,
            fortran_add(surface_water, liquid_excess),
        )

        ice_capacity = jnp.maximum(
            fortran_add(
                fortran_add(
                    jnp.float64(MAXIMUM_PONDING_MILLIMETERS),
                    fortran_multiply(
                        saturated_water_content[0],
                        thickness_millimeters[0],
                    ),
                ),
                -liquid[0],
            ),
            jnp.float64(0.0),
        )
        ice_excess = jnp.maximum(
            fortran_add(jnp.maximum(ice_water[0], jnp.float64(0.0)), -ice_capacity),
            jnp.float64(0.0),
        )
        ice = ice_water.at[0].set(jnp.minimum(ice_capacity, ice_water[0]))
        ice_excess_flux = fortran_divide(ice_excess, time_step_seconds)

        def restore_minimum(level: int, carry):
            current_liquid, current_water_table = carry
            deficit = jnp.where(
                current_liquid[level] < jnp.float64(MINIMUM_SOIL_WATER_MILLIMETERS),
                fortran_add(
                    jnp.float64(MINIMUM_SOIL_WATER_MILLIMETERS),
                    -current_liquid[level],
                ),
                jnp.float64(0.0),
            )
            deepens = jnp.int32(level + 1) == layer_above_water_table
            updated_water_table = jnp.where(
                deepens,
                fortran_add(
                    current_water_table,
                    _water_table_increment(deficit, effective_porosity[level]),
                ),
                current_water_table,
            )
            current_liquid = current_liquid.at[level].set(
                fortran_add(current_liquid[level], deficit)
            )
            current_liquid = current_liquid.at[level + 1].set(
                fortran_add(current_liquid[level + 1], -deficit)
            )
            return current_liquid, updated_water_table

        liquid, water_table = lax.fori_loop(
            0,
            NUMBER_OF_SOIL_LEVELS - 1,
            restore_minimum,
            (liquid, water_table),
        )

        bottom = NUMBER_OF_SOIL_LEVELS - 1
        bottom_deficit = jnp.where(
            liquid[bottom] < jnp.float64(MINIMUM_SOIL_WATER_MILLIMETERS),
            fortran_add(
                jnp.float64(MINIMUM_SOIL_WATER_MILLIMETERS),
                -liquid[bottom],
            ),
            jnp.float64(0.0),
        )

        def find_bottom_water(offset: int, carry):
            current_liquid, remaining, done = carry
            level = NUMBER_OF_SOIL_LEVELS - 2 - jnp.int32(offset)
            available = jnp.maximum(
                fortran_add(
                    fortran_add(
                        current_liquid[level],
                        -jnp.float64(MINIMUM_SOIL_WATER_MILLIMETERS),
                    ),
                    -remaining,
                ),
                jnp.float64(0.0),
            )
            enough = available >= remaining
            transfer = jnp.where(enough, remaining, available)
            executes = ~done & (remaining > jnp.float64(0.0))
            current_liquid = current_liquid.at[bottom].set(
                jnp.where(
                    executes,
                    fortran_add(current_liquid[bottom], transfer),
                    current_liquid[bottom],
                )
            )
            current_liquid = current_liquid.at[level].set(
                jnp.where(
                    executes,
                    fortran_add(current_liquid[level], -transfer),
                    current_liquid[level],
                )
            )
            updated_remaining = fortran_add(remaining, -transfer)
            return (
                current_liquid,
                jnp.where(executes, updated_remaining, remaining),
                done | (executes & enough),
            )

        liquid, residual_bottom, _ = lax.fori_loop(
            0,
            NUMBER_OF_SOIL_LEVELS - 1,
            find_bottom_water,
            (liquid, bottom_deficit, jnp.bool_(False)),
        )
        liquid = liquid.at[bottom].set(fortran_add(liquid[bottom], residual_bottom))
        saturated_excess = fortran_add(
            saturated_excess,
            -fortran_divide(residual_bottom, time_step_seconds),
        )
        drainage = fortran_add(saturated_excess, topographic_flux)
        drainage = jnp.where(urban_nonpervious, jnp.float64(0.0), drainage)
        return (
            liquid,
            ice,
            surface,
            water_table,
            saturated_excess,
            drainage,
            snow_capping_liquid_flux,
            ice_excess_flux,
            layer_above_water_table,
            topographic_flux,
        )

    def preserve(_):
        return (
            liquid_water,
            ice_water,
            surface_water,
            water_table_depth,
            previous_saturated_excess_flux,
            previous_drainage_flux,
            previous_qrgwl_flux,
            previous_ice_excess_flux,
            jnp.int32(-1),
            jnp.float64(0.0),
        )

    return lax.cond(active, execute, preserve, operand=None)


def _compute_nonlake_water_mass(
    topology: SubgridTopology,
    inputs: HydrologyDrainageInputs,
    liquid_water: jnp.ndarray,
    ice_water: jnp.ndarray,
    surface_water: jnp.ndarray,
    nonlake: jnp.ndarray,
    hydrology: jnp.ndarray,
    *,
    maximum_patches_per_column: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:

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
    liquid_mass = jnp.where(
        nonlake,
        fortran_add(
            fortran_add(canopy_water, -canopy_snow),
            inputs.total_plant_stored_water_column,
        ),
        jnp.float64(0.0),
    )
    ice_mass = jnp.where(nonlake, canopy_snow, jnp.float64(0.0))
    snow_level_number = jnp.arange(
        -NUMBER_OF_SNOW_LEVELS + 1,
        1,
        dtype=jnp.int32,
    )

    def accumulate_snow(level: int, carry):
        current_liquid, current_ice = carry
        explicit = (
            nonlake
            & (inputs.snow_layer_count_column < jnp.int32(0))
            & (snow_level_number[level] >= inputs.snow_layer_count_column + jnp.int32(1))
        )
        return (
            jnp.where(
                explicit,
                fortran_add(current_liquid, liquid_water[:, level]),
                current_liquid,
            ),
            jnp.where(
                explicit,
                fortran_add(current_ice, ice_water[:, level]),
                current_ice,
            ),
        )

    liquid_mass, ice_mass = lax.fori_loop(
        0,
        NUMBER_OF_SNOW_LEVELS,
        accumulate_snow,
        (liquid_mass, ice_mass),
    )
    unresolved_snow = (
        nonlake
        & (inputs.snow_layer_count_column >= jnp.int32(0))
        & (inputs.snow_water_equivalent_column != jnp.float64(0.0))
    )
    ice_mass = jnp.where(
        unresolved_snow,
        fortran_add(ice_mass, inputs.snow_water_equivalent_column),
        ice_mass,
    )
    liquid_mass = jnp.where(
        nonlake & hydrology,
        fortran_add(
            liquid_mass,
            fortran_add(
                inputs.aquifer_water_column,
                -jnp.float64(AQUIFER_WATER_BASELINE_MILLIMETERS),
            ),
        ),
        liquid_mass,
    )
    column_type = jnp.asarray(topology.column.column_type, dtype=jnp.int32)
    excludes_surface_water = (
        (column_type == jnp.int32(COLUMN_URBAN_ROOF))
        | (column_type == jnp.int32(COLUMN_URBAN_SUN_WALL))
        | (column_type == jnp.int32(COLUMN_URBAN_SHADE_WALL))
        | (column_type == jnp.int32(COLUMN_URBAN_IMPERVIOUS_ROAD))
    )
    liquid_mass = jnp.where(
        nonlake & ~excludes_surface_water,
        fortran_add(liquid_mass, surface_water),
        liquid_mass,
    )

    def accumulate_ground(level: int, carry):
        current_liquid, current_ice = carry
        walls = (column_type == jnp.int32(COLUMN_URBAN_SUN_WALL)) | (
            column_type == jnp.int32(COLUMN_URBAN_SHADE_WALL)
        )
        roof_too_deep = (column_type == jnp.int32(COLUMN_URBAN_ROOF)) & (level >= jnp.int32(5))
        includes = nonlake & ~walls & ~roof_too_deep
        source_level = GROUND_SURFACE_INDEX + level
        return (
            jnp.where(
                includes,
                fortran_add(current_liquid, liquid_water[:, source_level]),
                current_liquid,
            ),
            jnp.where(
                includes,
                fortran_add(current_ice, ice_water[:, source_level]),
                current_ice,
            ),
        )

    return lax.fori_loop(
        0,
        NUMBER_OF_GROUND_LEVELS,
        accumulate_ground,
        (liquid_mass, ice_mass),
    )


def hydrology_drainage_active_case(
    topology: SubgridTopology,
    inputs: HydrologyDrainageInputs,
    state: HydrologyDrainageState,
    *,
    parameters: HydrologyDrainageParameters = DEFAULT_HYDROLOGY_DRAINAGE_PARAMETERS,
) -> HydrologyDrainageResult:

    _validate_shapes(topology, inputs, state, parameters)
    active = jnp.asarray(topology.column.active, dtype=jnp.bool_)
    landunit_type = jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
    column_type = jnp.asarray(topology.column.column_type, dtype=jnp.int32)
    masks = build_surface_hydrology_masks(topology)
    hydrology = masks.hydrology_column
    urban = masks.urban_column
    nonlake = active & (landunit_type != jnp.int32(LANDUNIT_LAKE))
    urban_nonpervious = urban & (column_type != jnp.int32(COLUMN_URBAN_PERVIOUS_ROAD))
    wetland_or_glacier = nonlake & (
        (landunit_type == jnp.int32(LANDUNIT_WETLAND))
        | (landunit_type == jnp.int32(LANDUNIT_GLACIER))
    )
    do_surface_mass_balance = jnp.asarray(
        inputs.do_surface_mass_balance_column,
        dtype=jnp.bool_,
    )
    time_step = jnp.float64(parameters.time_step_seconds)
    ground_slice = slice(GROUND_SURFACE_INDEX, NUMBER_OF_SOIL_SNOW_LEVELS)
    soil_slice = slice(0, NUMBER_OF_SOIL_LEVELS)

    liquid_ground = jnp.asarray(
        state.soil_snow_liquid_water_column_level[:, ground_slice],
        dtype=jnp.float64,
    )
    ice_ground = jnp.asarray(
        state.soil_snow_ice_water_column_level[:, ground_slice],
        dtype=jnp.float64,
    )
    ice_fraction_candidate = _soil_ice_fraction(
        ice_ground[:, soil_slice],
        inputs.layer_thickness_ground_column_level[:, soil_slice],
        inputs.saturated_water_content_column_level[:, soil_slice],
    )
    ice_fraction = (
        jnp.asarray(
            state.soil_ice_fraction_column_level,
            dtype=jnp.float64,
        )
        .at[:, soil_slice]
        .set(
            jnp.where(
                hydrology[:, None],
                ice_fraction_candidate,
                state.soil_ice_fraction_column_level[:, soil_slice],
            )
        )
    )

    perched_results = jax.vmap(_perched_lateral_flow_column)(
        hydrology,
        inputs.layer_thickness_ground_column_level[:, soil_slice],
        inputs.lower_interface_depth_soil_column_level,
        inputs.topographic_slope_degrees_column,
        inputs.saturated_water_content_column_level[:, soil_slice],
        inputs.saturated_hydraulic_conductivity_column_level[:, soil_slice],
        inputs.saturated_suction_column_level[:, soil_slice],
        inputs.clapp_hornberger_exponent_column_level[:, soil_slice],
        ice_fraction[:, soil_slice],
        state.frost_table_depth_column,
        state.water_table_depth_column,
        state.perched_water_table_depth_column,
        liquid_ground[:, soil_slice],
        state.perched_drainage_flux_column,
        jnp.broadcast_to(time_step, state.water_table_depth_column.shape),
    )
    (
        liquid_soil,
        perched_water_table,
        perched_flux,
        frost_level,
        perched_level,
        frost_found,
        perched_found,
    ) = perched_results
    liquid_ground = liquid_ground.at[:, soil_slice].set(liquid_soil)

    lateral_results = jax.vmap(_lateral_flow_power_law_column)(
        hydrology,
        urban,
        urban_nonpervious,
        inputs.soil_layer_count_column,
        inputs.layer_thickness_ground_column_level[:, soil_slice],
        inputs.lower_interface_depth_soil_column_level,
        inputs.topographic_slope_degrees_column,
        inputs.saturated_water_content_column_level[:, soil_slice],
        inputs.saturated_suction_column_level[:, soil_slice],
        inputs.clapp_hornberger_exponent_column_level[:, soil_slice],
        inputs.effective_porosity_column_level[:, soil_slice],
        ice_fraction[:, soil_slice],
        state.water_table_depth_column,
        liquid_ground[:, soil_slice],
        ice_ground[:, soil_slice],
        state.surface_water_column,
        inputs.snow_capping_liquid_flux_column,
        state.subsurface_drainage_flux_column,
        state.saturated_excess_flux_column,
        state.wetland_glacier_runoff_flux_column,
        state.excess_soil_ice_runoff_flux_column,
        jnp.broadcast_to(time_step, state.water_table_depth_column.shape),
        jnp.full_like(
            state.water_table_depth_column,
            jnp.float64(parameters.baseflow_scalar),
        ),
    )
    (
        liquid_soil,
        ice_soil,
        surface_water,
        water_table,
        saturated_excess_flux,
        drainage_flux,
        qrgwl_flux,
        ice_excess_flux,
        layer_above_water_table,
        topographic_flux,
    ) = lateral_results
    drainage_flux = jnp.where(
        urban_nonpervious,
        jnp.float64(0.0),
        drainage_flux,
    )
    qrgwl_flux = jnp.where(
        urban_nonpervious,
        inputs.snow_capping_liquid_flux_column,
        qrgwl_flux,
    )
    liquid_ground = liquid_ground.at[:, soil_slice].set(liquid_soil)
    ice_ground = ice_ground.at[:, soil_slice].set(ice_soil)
    liquid_water = state.soil_snow_liquid_water_column_level.at[:, ground_slice].set(liquid_ground)
    ice_water = state.soil_snow_ice_water_column_level.at[:, ground_slice].set(ice_ground)

    volumetric_candidate = fortran_add(
        fortran_divide(
            liquid_ground,
            fortran_multiply(
                inputs.layer_thickness_ground_column_level,
                jnp.float64(DENSITY_FRESH_WATER),
            ),
        ),
        fortran_divide(
            ice_ground,
            fortran_multiply(
                inputs.layer_thickness_ground_column_level,
                jnp.float64(DENSITY_ICE),
            ),
        ),
    )
    walls = (column_type == jnp.int32(COLUMN_URBAN_SUN_WALL)) | (
        column_type == jnp.int32(COLUMN_URBAN_SHADE_WALL)
    )
    roof_below_urban = (column_type[:, None] == jnp.int32(COLUMN_URBAN_ROOF)) & (
        jnp.arange(NUMBER_OF_GROUND_LEVELS, dtype=jnp.int32)[None, :] >= jnp.int32(5)
    )
    update_volumetric = nonlake[:, None] & ~walls[:, None] & ~roof_below_urban
    volumetric_water = jnp.where(
        update_volumetric,
        volumetric_candidate,
        state.volumetric_water_column_level,
    )

    liquid_mass, ice_mass = _compute_nonlake_water_mass(
        topology,
        inputs,
        liquid_water,
        ice_water,
        surface_water,
        nonlake,
        hydrology,
        maximum_patches_per_column=parameters.maximum_patches_per_column,
    )
    ending_water_mass = jnp.where(
        nonlake,
        fortran_add(liquid_mass, ice_mass),
        state.ending_water_mass_column,
    )

    drainage_flux = jnp.where(wetland_or_glacier, jnp.float64(0.0), drainage_flux)
    perched_flux = jnp.where(
        wetland_or_glacier | urban_nonpervious,
        jnp.float64(0.0),
        perched_flux,
    )
    surface_water_runoff_flux = jnp.where(
        wetland_or_glacier | urban_nonpervious,
        jnp.float64(0.0),
        state.surface_water_runoff_flux_column,
    )
    surface_runoff_flux = jnp.where(
        wetland_or_glacier,
        jnp.float64(0.0),
        state.surface_runoff_flux_column,
    )
    infiltration_flux = jnp.where(
        wetland_or_glacier,
        jnp.float64(0.0),
        state.infiltration_flux_column,
    )
    saturated_excess_flux = jnp.where(
        urban_nonpervious,
        jnp.float64(CLM_SPECIAL_VALUE),
        saturated_excess_flux,
    )

    gridcell = jnp.asarray(topology.column.gridcell, dtype=jnp.int32)
    qrgwl_wetland_glacier = fortran_add(
        inputs.rainfall_flux_column,
        inputs.snowfall_flux_column,
    )
    qrgwl_wetland_glacier = fortran_add(
        qrgwl_wetland_glacier,
        inputs.flood_flux_gridcell[gridcell],
    )
    qrgwl_wetland_glacier = fortran_add(
        qrgwl_wetland_glacier,
        -inputs.total_evaporation_flux_column,
    )
    qrgwl_wetland_glacier = fortran_add(
        qrgwl_wetland_glacier,
        -inputs.snow_capping_ice_flux_column,
    )
    qrgwl_wetland_glacier = fortran_add(
        qrgwl_wetland_glacier,
        -inputs.snow_capping_discarded_ice_flux_column,
    )
    qrgwl_wetland_glacier = fortran_add(
        qrgwl_wetland_glacier,
        -inputs.snow_capping_discarded_liquid_flux_column,
    )
    qrgwl_wetland_glacier = fortran_add(
        qrgwl_wetland_glacier,
        -fortran_divide(
            fortran_add(ending_water_mass, -inputs.beginning_water_mass_column),
            time_step,
        ),
    )
    qrgwl_flux = jnp.where(wetland_or_glacier, qrgwl_wetland_glacier, qrgwl_flux)
    snow_capping_ice_runoff = jnp.where(
        nonlake,
        inputs.snow_capping_ice_flux_column,
        state.snow_capping_ice_runoff_flux_column,
    )

    routing = inputs.dynamic_runoff_routing_gridcell[gridcell]
    qrgwl_flux = jnp.where(
        do_surface_mass_balance,
        fortran_add(qrgwl_flux, inputs.glacier_ice_melt_flux_column),
        qrgwl_flux,
    )
    adjusted_ice_runoff = fortran_add(
        snow_capping_ice_runoff,
        -fortran_multiply(routing, inputs.glacier_ice_growth_flux_column),
    )
    adjusted_ice_runoff = fortran_add(
        adjusted_ice_runoff,
        -fortran_multiply(
            fortran_add(jnp.float64(1.0), -routing),
            inputs.glacier_ice_melt_flux_column,
        ),
    )
    snow_capping_ice_runoff = jnp.where(
        do_surface_mass_balance,
        adjusted_ice_runoff,
        snow_capping_ice_runoff,
    )

    total_runoff = fortran_add(drainage_flux, surface_runoff_flux)
    total_runoff = fortran_add(total_runoff, surface_water_runoff_flux)
    total_runoff = fortran_add(total_runoff, qrgwl_flux)
    total_runoff = fortran_add(total_runoff, perched_flux)
    soil_crop = active & (
        (landunit_type == jnp.int32(LANDUNIT_SOIL)) | (landunit_type == jnp.int32(LANDUNIT_CROP))
    )
    total_runoff = jnp.where(
        soil_crop,
        fortran_add(total_runoff, -inputs.irrigation_flux_column),
        total_runoff,
    )
    total_runoff = jnp.where(
        nonlake,
        total_runoff,
        state.total_runoff_flux_column,
    )
    urban_runoff = jnp.where(
        nonlake & urban,
        total_runoff,
        state.urban_runoff_flux_column,
    )
    rural_runoff = jnp.where(
        nonlake & soil_crop,
        total_runoff,
        state.rural_runoff_flux_column,
    )

    return HydrologyDrainageResult(
        state=HydrologyDrainageState(
            soil_snow_liquid_water_column_level=liquid_water,
            soil_snow_ice_water_column_level=ice_water,
            surface_water_column=surface_water,
            water_table_depth_column=water_table,
            perched_water_table_depth_column=perched_water_table,
            frost_table_depth_column=state.frost_table_depth_column,
            soil_ice_fraction_column_level=ice_fraction,
            volumetric_water_column_level=volumetric_water,
            ending_water_mass_column=ending_water_mass,
            perched_drainage_flux_column=perched_flux,
            saturated_excess_flux_column=saturated_excess_flux,
            subsurface_drainage_flux_column=drainage_flux,
            surface_runoff_flux_column=surface_runoff_flux,
            surface_water_runoff_flux_column=surface_water_runoff_flux,
            infiltration_flux_column=infiltration_flux,
            wetland_glacier_runoff_flux_column=qrgwl_flux,
            total_runoff_flux_column=total_runoff,
            urban_runoff_flux_column=urban_runoff,
            rural_runoff_flux_column=rural_runoff,
            excess_soil_ice_runoff_flux_column=ice_excess_flux,
            snow_capping_ice_runoff_flux_column=snow_capping_ice_runoff,
        ),
        diagnostics=HydrologyDrainageDiagnostics(
            nonlake_column=nonlake,
            hydrology_column=hydrology,
            urban_column=urban,
            wetland_or_glacier_column=wetland_or_glacier,
            do_surface_mass_balance_column=do_surface_mass_balance,
            frost_interface_found_column=frost_found,
            perched_interface_found_column=perched_found,
            frost_level_column=frost_level,
            perched_level_column=perched_level,
            layer_above_water_table_column=layer_above_water_table,
            topographic_subsurface_runoff_flux_column=topographic_flux,
            liquid_water_mass_column=liquid_mass,
            ice_water_mass_column=ice_mass,
        ),
    )
