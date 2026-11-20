
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.constants import (
    GAS_CONSTANT_DRY_AIR,
    SPECIFIC_HEAT_DRY_AIR,
    STANDARD_ATMOSPHERIC_PRESSURE,
    STEFAN_BOLTZMANN,
)
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
    fortran_ordered_sum,
    fortran_power,
)

NUMBER_OF_BUILDING_EQUATIONS = 5

ROOF_CONVECTION = 0.948
ENHANCED_ROOF_CONVECTION = 4.040
FLOOR_CONVECTION = 0.948
ENHANCED_FLOOR_CONVECTION = 4.040
SUNWALL_CONVECTION = 3.076
SHADEWALL_CONVECTION = 3.076
INTERIOR_ROOF_EMISSIVITY = 0.9
INTERIOR_SUNWALL_EMISSIVITY = 0.9
INTERIOR_SHADEWALL_EMISSIVITY = 0.9
INTERIOR_FLOOR_EMISSIVITY = 0.9
FLOOR_THICKNESS = 0.1
FLOOR_VOLUMETRIC_HEAT_CAPACITY = 2350.0 * 880.0
VENTILATION_AIR_CHANGES_PER_HOUR = 0.30000001192092896


class UrbanBuildingTemperatureParameters(NamedTuple):

    roof_convection: jnp.ndarray
    enhanced_roof_convection: jnp.ndarray
    floor_convection: jnp.ndarray
    enhanced_floor_convection: jnp.ndarray
    sunwall_convection: jnp.ndarray
    shadewall_convection: jnp.ndarray
    roof_emissivity: jnp.ndarray
    sunwall_emissivity: jnp.ndarray
    shadewall_emissivity: jnp.ndarray
    floor_emissivity: jnp.ndarray
    floor_thickness: jnp.ndarray
    floor_volumetric_heat_capacity: jnp.ndarray
    ventilation_air_changes_per_hour: jnp.ndarray
    stefan_boltzmann: jnp.ndarray
    standard_pressure: jnp.ndarray
    dry_air_gas_constant: jnp.ndarray
    dry_air_specific_heat: jnp.ndarray


class UrbanBuildingTemperatureInputs(NamedTuple):

    active_urban: jnp.ndarray
    roof_height: jnp.ndarray
    canyon_height_width_ratio: jnp.ndarray
    roof_weight: jnp.ndarray
    canyon_air_temperature: jnp.ndarray
    minimum_building_temperature: jnp.ndarray
    maximum_building_temperature: jnp.ndarray
    roof_inner_temperature: jnp.ndarray
    sunwall_inner_temperature: jnp.ndarray
    shadewall_inner_temperature: jnp.ndarray
    floor_temperature: jnp.ndarray
    building_temperature: jnp.ndarray
    previous_roof_boundary_temperature: jnp.ndarray
    previous_sunwall_boundary_temperature: jnp.ndarray
    previous_shadewall_boundary_temperature: jnp.ndarray
    roof_boundary_temperature: jnp.ndarray
    sunwall_boundary_temperature: jnp.ndarray
    shadewall_boundary_temperature: jnp.ndarray
    roof_node_depth: jnp.ndarray
    sunwall_node_depth: jnp.ndarray
    shadewall_node_depth: jnp.ndarray
    roof_interface_depth: jnp.ndarray
    sunwall_interface_depth: jnp.ndarray
    shadewall_interface_depth: jnp.ndarray
    roof_thermal_conductivity: jnp.ndarray
    sunwall_thermal_conductivity: jnp.ndarray
    shadewall_thermal_conductivity: jnp.ndarray
    previous_building_heat_flux: jnp.ndarray
    previous_air_conditioning_flux: jnp.ndarray
    previous_heating_flux: jnp.ndarray


class UrbanBuildingTemperatureResult(NamedTuple):

    roof_inner_temperature: jnp.ndarray
    sunwall_inner_temperature: jnp.ndarray
    shadewall_inner_temperature: jnp.ndarray
    floor_temperature: jnp.ndarray
    building_temperature: jnp.ndarray
    unconstrained_building_temperature: jnp.ndarray
    building_heat_flux: jnp.ndarray
    air_conditioning_flux: jnp.ndarray
    heating_flux: jnp.ndarray
    coefficient_matrix: jnp.ndarray
    right_hand_side: jnp.ndarray
    pivot_indices: jnp.ndarray
    nonsingular: jnp.ndarray


def default_urban_building_temperature_parameters() -> UrbanBuildingTemperatureParameters:

    return UrbanBuildingTemperatureParameters(
        roof_convection=jnp.float64(ROOF_CONVECTION),
        enhanced_roof_convection=jnp.float64(ENHANCED_ROOF_CONVECTION),
        floor_convection=jnp.float64(FLOOR_CONVECTION),
        enhanced_floor_convection=jnp.float64(ENHANCED_FLOOR_CONVECTION),
        sunwall_convection=jnp.float64(SUNWALL_CONVECTION),
        shadewall_convection=jnp.float64(SHADEWALL_CONVECTION),
        roof_emissivity=jnp.float64(INTERIOR_ROOF_EMISSIVITY),
        sunwall_emissivity=jnp.float64(INTERIOR_SUNWALL_EMISSIVITY),
        shadewall_emissivity=jnp.float64(INTERIOR_SHADEWALL_EMISSIVITY),
        floor_emissivity=jnp.float64(INTERIOR_FLOOR_EMISSIVITY),
        floor_thickness=jnp.float64(FLOOR_THICKNESS),
        floor_volumetric_heat_capacity=jnp.float64(
            FLOOR_VOLUMETRIC_HEAT_CAPACITY
        ),
        ventilation_air_changes_per_hour=jnp.float64(
            VENTILATION_AIR_CHANGES_PER_HOUR
        ),
        stefan_boltzmann=jnp.float64(STEFAN_BOLTZMANN),
        standard_pressure=jnp.float64(STANDARD_ATMOSPHERIC_PRESSURE),
        dry_air_gas_constant=jnp.float64(GAS_CONSTANT_DRY_AIR),
        dry_air_specific_heat=jnp.float64(SPECIFIC_HEAT_DRY_AIR),
    )


def _validate_inputs(inputs: UrbanBuildingTemperatureInputs) -> None:
    if inputs.active_urban.ndim != 1:
        raise ValueError("active_urban must be a vector")
    expected_shape = inputs.active_urban.shape
    for name in UrbanBuildingTemperatureInputs._fields:
        if getattr(inputs, name).shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")


def _product(*factors: jnp.ndarray | float) -> jnp.ndarray:
    if not factors:
        raise ValueError("ordered product requires at least one factor")
    result = jnp.asarray(factors[0], dtype=jnp.float64)
    for factor in factors[1:]:
        result = fortran_multiply(result, jnp.asarray(factor, dtype=jnp.float64))
    return result


def _sum(*terms: jnp.ndarray | float) -> jnp.ndarray:
    if not terms:
        raise ValueError("ordered sum requires at least one term")
    return fortran_ordered_sum(
        jnp.stack(tuple(jnp.asarray(term, dtype=jnp.float64) for term in terms))
    )


def _power3(value: jnp.ndarray) -> jnp.ndarray:
    if jax.default_backend() == "gpu":
        from clm_jax.physics.native_power import native_power
        return native_power(value, jnp.float64(3.0))
    exponent = lax.optimization_barrier(jnp.float64(3.0))
    return fortran_power(value, exponent)


def _power4(value: jnp.ndarray) -> jnp.ndarray:
    if jax.default_backend() == "gpu":
        from clm_jax.physics.native_power import native_power
        return native_power(value, jnp.float64(4.0))
    return fortran_power(value, jnp.float64(4.0))


def _swap_rows(
    matrix: jnp.ndarray,
    first: jnp.ndarray,
    second: jnp.ndarray,
    column_start: int,
    column_stop: int,
) -> jnp.ndarray:
    columns = jnp.arange(column_start, column_stop, dtype=jnp.int32)
    first_values = matrix[first, columns]
    second_values = matrix[second, columns]
    matrix = matrix.at[first, columns].set(second_values)
    return matrix.at[second, columns].set(first_values)


def _recursive_dgetrf2(
    matrix: jnp.ndarray,
    pivots: jnp.ndarray,
    row_start: int,
    column_start: int,
    rows: int,
    columns: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    if rows == 1 and columns == 1:
        pivot = jnp.int32(row_start)
        pivots = pivots.at[column_start].set(pivot)
        return matrix, pivots, matrix[row_start, column_start] != 0.0

    if columns == 1:
        candidates = jnp.abs(
            matrix[row_start : row_start + rows, column_start]
        )
        pivot = jnp.int32(row_start) + jnp.argmax(candidates).astype(jnp.int32)
        matrix = _swap_rows(
            matrix,
            jnp.int32(row_start),
            pivot,
            column_start,
            column_start + 1,
        )
        pivots = pivots.at[column_start].set(pivot)
        diagonal = matrix[row_start, column_start]
        nonsingular = diagonal != 0.0
        safe_diagonal = jnp.where(nonsingular, diagonal, jnp.float64(1.0))
        inverse = fortran_divide(jnp.float64(1.0), safe_diagonal)
        for row in range(row_start + 1, row_start + rows):
            scaled = fortran_multiply(matrix[row, column_start], inverse)
            matrix = matrix.at[row, column_start].set(
                jnp.where(nonsingular, scaled, matrix[row, column_start])
            )
        return matrix, pivots, nonsingular

    first_columns = min(rows, columns) // 2
    second_columns = columns - first_columns
    matrix, pivots, first_nonsingular = _recursive_dgetrf2(
        matrix,
        pivots,
        row_start,
        column_start,
        rows,
        first_columns,
    )

    for pivot_column in range(column_start, column_start + first_columns):
        matrix = _swap_rows(
            matrix,
            jnp.int32(pivot_column - column_start + row_start),
            pivots[pivot_column],
            column_start + first_columns,
            column_start + columns,
        )

    for right_column in range(
        column_start + first_columns,
        column_start + columns,
    ):
        for lower_column in range(first_columns):
            source_row = row_start + lower_column
            source = matrix[source_row, right_column]
            for target_row in range(source_row + 1, row_start + first_columns):
                updated = fortran_add(
                    matrix[target_row, right_column],
                    -fortran_multiply(
                        source,
                        matrix[target_row, column_start + lower_column],
                    ),
                )
                matrix = matrix.at[target_row, right_column].set(updated)

    for right_column in range(
        column_start + first_columns,
        column_start + columns,
    ):
        for inner in range(first_columns):
            temporary = fortran_multiply(
                jnp.float64(-1.0),
                matrix[row_start + inner, right_column],
            )
            for target_row in range(
                row_start + first_columns,
                row_start + rows,
            ):
                updated = fortran_add(
                    matrix[target_row, right_column],
                    fortran_multiply(
                        temporary,
                        matrix[target_row, column_start + inner],
                    ),
                )
                matrix = matrix.at[target_row, right_column].set(updated)

    matrix, pivots, second_nonsingular = _recursive_dgetrf2(
        matrix,
        pivots,
        row_start + first_columns,
        column_start + first_columns,
        rows - first_columns,
        second_columns,
    )

    for pivot_column in range(
        column_start + first_columns,
        column_start + min(rows, columns),
    ):
        matrix = _swap_rows(
            matrix,
            jnp.int32(pivot_column - column_start + row_start),
            pivots[pivot_column],
            column_start,
            column_start + first_columns,
        )
    return matrix, pivots, first_nonsingular & second_nonsingular


def _dgesv_five(
    coefficient_matrix: jnp.ndarray,
    right_hand_side: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    pivots = jnp.arange(NUMBER_OF_BUILDING_EQUATIONS, dtype=jnp.int32)
    factored, pivots, nonsingular = _recursive_dgetrf2(
        coefficient_matrix,
        pivots,
        0,
        0,
        NUMBER_OF_BUILDING_EQUATIONS,
        NUMBER_OF_BUILDING_EQUATIONS,
    )
    solution = right_hand_side

    for row in range(NUMBER_OF_BUILDING_EQUATIONS):
        pivot = pivots[row]
        first = solution[row]
        second = solution[pivot]
        solution = solution.at[row].set(second)
        solution = solution.at[pivot].set(first)

    for column in range(NUMBER_OF_BUILDING_EQUATIONS):
        source = solution[column]
        for row in range(column + 1, NUMBER_OF_BUILDING_EQUATIONS):
            solution = solution.at[row].set(
                fortran_add(
                    solution[row],
                    -fortran_multiply(source, factored[row, column]),
                )
            )

    for column in range(NUMBER_OF_BUILDING_EQUATIONS - 1, -1, -1):
        quotient = fortran_divide(solution[column], factored[column, column])
        solution = solution.at[column].set(quotient)
        temporary = -quotient
        for row in range(column):
            solution = solution.at[row].set(
                fortran_add(
                    solution[row],
                    fortran_multiply(temporary, factored[row, column]),
                )
            )
    return solution, pivots, nonsingular


def _difference(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return fortran_add(left, -right)


def _half_conduction(
    conductivity: jnp.ndarray,
    temperature: jnp.ndarray,
    depth_interval: jnp.ndarray,
) -> jnp.ndarray:
    return fortran_divide(
        _product(jnp.float64(0.5), conductivity, temperature),
        depth_interval,
    )


def _assemble_building_system(
    inputs: UrbanBuildingTemperatureInputs,
    parameters: UrbanBuildingTemperatureParameters,
    time_step_seconds: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    hwr = inputs.canyon_height_width_ratio
    roof_temperature = inputs.roof_inner_temperature
    sunwall_temperature = inputs.sunwall_inner_temperature
    shadewall_temperature = inputs.shadewall_inner_temperature
    floor_temperature = inputs.floor_temperature
    building_temperature = inputs.building_temperature

    roof_convection = jnp.where(
        roof_temperature <= building_temperature,
        parameters.enhanced_roof_convection,
        parameters.roof_convection,
    )
    floor_convection = jnp.where(
        floor_temperature >= building_temperature,
        parameters.enhanced_floor_convection,
        parameters.floor_convection,
    )
    sunwall_convection = parameters.sunwall_convection
    shadewall_convection = parameters.shadewall_convection

    roof_emissivity = parameters.roof_emissivity
    sunwall_emissivity = parameters.sunwall_emissivity
    shadewall_emissivity = parameters.shadewall_emissivity
    floor_emissivity = parameters.floor_emissivity
    one_minus_roof_emissivity = _difference(1.0, roof_emissivity)
    one_minus_sunwall_emissivity = _difference(1.0, sunwall_emissivity)
    one_minus_shadewall_emissivity = _difference(
        1.0, shadewall_emissivity
    )
    one_minus_floor_emissivity = _difference(1.0, floor_emissivity)

    floor_capacity = fortran_divide(
        _product(
            parameters.floor_thickness,
            parameters.floor_volumetric_heat_capacity,
        ),
        time_step_seconds,
    )
    dry_air_density = fortran_divide(
        parameters.standard_pressure,
        _product(parameters.dry_air_gas_constant, building_temperature),
    )

    roof_interval = _difference(
        inputs.roof_interface_depth, inputs.roof_node_depth
    )
    sunwall_interval = _difference(
        inputs.sunwall_interface_depth, inputs.sunwall_node_depth
    )
    shadewall_interval = _difference(
        inputs.shadewall_interface_depth, inputs.shadewall_node_depth
    )

    roof_for_floor = _difference(
        jnp.sqrt(fortran_add(1.0, fortran_power(hwr, jnp.float64(2.0)))),
        hwr,
    )
    floor_for_roof = roof_for_floor
    wall_for_floor = _product(0.5, _difference(1.0, roof_for_floor))
    floor_for_wall = fortran_divide(wall_for_floor, hwr)
    roof_for_wall = floor_for_wall
    wall_for_roof = wall_for_floor
    wall_for_wall = _sum(1.0, -roof_for_wall, -floor_for_wall)

    sb = parameters.stefan_boltzmann
    roof_cube = _power3(roof_temperature)
    sunwall_cube = _power3(sunwall_temperature)
    shadewall_cube = _power3(shadewall_temperature)
    floor_cube = _power3(floor_temperature)
    roof_fourth = _power4(roof_temperature)
    sunwall_fourth = _power4(sunwall_temperature)
    shadewall_fourth = _power4(shadewall_temperature)
    floor_fourth = _power4(floor_temperature)

    a11 = _sum(
        _product(0.5, roof_convection),
        fortran_divide(
            _product(0.5, inputs.roof_thermal_conductivity), roof_interval
        ),
        _product(4.0, roof_emissivity, sb, roof_cube),
        -_product(
            4.0,
            roof_emissivity,
            sb,
            roof_cube,
            roof_for_wall,
            one_minus_sunwall_emissivity,
            wall_for_roof,
        ),
        -_product(
            4.0,
            roof_emissivity,
            sb,
            roof_cube,
            roof_for_wall,
            one_minus_shadewall_emissivity,
            wall_for_roof,
        ),
        -_product(
            4.0,
            roof_emissivity,
            sb,
            roof_cube,
            roof_for_floor,
            one_minus_floor_emissivity,
            floor_for_roof,
        ),
    )
    a12 = _sum(
        -_product(
            4.0,
            roof_emissivity,
            sunwall_emissivity,
            sb,
            sunwall_cube,
            wall_for_roof,
        ),
        -_product(
            4.0,
            sunwall_emissivity,
            sb,
            sunwall_cube,
            wall_for_wall,
            one_minus_shadewall_emissivity,
            wall_for_roof,
        ),
        -_product(
            4.0,
            sunwall_emissivity,
            sb,
            sunwall_cube,
            wall_for_floor,
            one_minus_floor_emissivity,
            floor_for_roof,
        ),
    )
    a13 = _sum(
        -_product(
            4.0,
            roof_emissivity,
            shadewall_emissivity,
            sb,
            shadewall_cube,
            wall_for_roof,
        ),
        -_product(
            4.0,
            shadewall_emissivity,
            sb,
            shadewall_cube,
            wall_for_wall,
            one_minus_sunwall_emissivity,
            wall_for_roof,
        ),
        -_product(
            4.0,
            shadewall_emissivity,
            sb,
            shadewall_cube,
            wall_for_floor,
            one_minus_floor_emissivity,
            floor_for_roof,
        ),
    )
    a14 = _sum(
        -_product(
            4.0,
            roof_emissivity,
            floor_emissivity,
            sb,
            floor_cube,
            floor_for_roof,
        ),
        -_product(
            4.0,
            floor_emissivity,
            sb,
            floor_cube,
            floor_for_wall,
            one_minus_sunwall_emissivity,
            wall_for_roof,
        ),
        -_product(
            4.0,
            floor_emissivity,
            sb,
            floor_cube,
            floor_for_wall,
            one_minus_shadewall_emissivity,
            wall_for_roof,
        ),
    )
    a15 = -_product(0.5, roof_convection)
    rhs1 = _sum(
        _half_conduction(
            inputs.roof_thermal_conductivity,
            inputs.roof_boundary_temperature,
            roof_interval,
        ),
        -_half_conduction(
            inputs.roof_thermal_conductivity,
            _difference(
                roof_temperature,
                inputs.previous_roof_boundary_temperature,
            ),
            roof_interval,
        ),
        -_product(
            3.0,
            roof_emissivity,
            sunwall_emissivity,
            sb,
            sunwall_fourth,
            wall_for_roof,
        ),
        -_product(
            3.0,
            roof_emissivity,
            shadewall_emissivity,
            sb,
            shadewall_fourth,
            wall_for_roof,
        ),
        -_product(
            3.0,
            roof_emissivity,
            floor_emissivity,
            sb,
            floor_fourth,
            floor_for_roof,
        ),
        _product(3.0, roof_emissivity, sb, roof_fourth),
        -_product(
            3.0,
            roof_emissivity,
            sb,
            roof_fourth,
            roof_for_wall,
            one_minus_sunwall_emissivity,
            wall_for_roof,
        ),
        -_product(
            3.0,
            roof_emissivity,
            sb,
            roof_fourth,
            roof_for_wall,
            one_minus_shadewall_emissivity,
            wall_for_roof,
        ),
        -_product(
            3.0,
            roof_emissivity,
            sb,
            roof_fourth,
            roof_for_floor,
            one_minus_floor_emissivity,
            floor_for_roof,
        ),
        -_product(
            3.0,
            sunwall_emissivity,
            sb,
            sunwall_fourth,
            wall_for_wall,
            one_minus_shadewall_emissivity,
            wall_for_roof,
        ),
        -_product(
            3.0,
            sunwall_emissivity,
            sb,
            sunwall_fourth,
            wall_for_floor,
            one_minus_floor_emissivity,
            floor_for_roof,
        ),
        -_product(
            3.0,
            shadewall_emissivity,
            sb,
            shadewall_fourth,
            wall_for_wall,
            one_minus_sunwall_emissivity,
            wall_for_roof,
        ),
        -_product(
            3.0,
            shadewall_emissivity,
            sb,
            shadewall_fourth,
            wall_for_floor,
            one_minus_floor_emissivity,
            floor_for_roof,
        ),
        -_product(
            3.0,
            floor_emissivity,
            sb,
            floor_fourth,
            floor_for_wall,
            one_minus_sunwall_emissivity,
            wall_for_roof,
        ),
        -_product(
            3.0,
            floor_emissivity,
            sb,
            floor_fourth,
            floor_for_wall,
            one_minus_shadewall_emissivity,
            wall_for_roof,
        ),
        -_product(
            0.5,
            roof_convection,
            _difference(roof_temperature, building_temperature),
        ),
    )

    a21 = _sum(
        -_product(
            4.0,
            sunwall_emissivity,
            roof_emissivity,
            sb,
            roof_cube,
            roof_for_wall,
        ),
        -_product(
            4.0,
            roof_emissivity,
            sb,
            roof_cube,
            roof_for_wall,
            one_minus_shadewall_emissivity,
            wall_for_wall,
        ),
        -_product(
            4.0,
            roof_emissivity,
            sb,
            roof_cube,
            roof_for_floor,
            one_minus_floor_emissivity,
            floor_for_wall,
        ),
    )
    a22 = _sum(
        _product(0.5, sunwall_convection, hwr),
        _product(
            fortran_divide(
                _product(0.5, inputs.sunwall_thermal_conductivity),
                sunwall_interval,
            ),
            hwr,
        ),
        _product(4.0, sunwall_emissivity, sb, sunwall_cube),
        -_product(
            4.0,
            sunwall_emissivity,
            sb,
            sunwall_cube,
            wall_for_roof,
            one_minus_roof_emissivity,
            roof_for_wall,
        ),
        -_product(
            4.0,
            sunwall_emissivity,
            sb,
            sunwall_cube,
            wall_for_wall,
            one_minus_shadewall_emissivity,
            wall_for_wall,
        ),
        -_product(
            4.0,
            sunwall_emissivity,
            sb,
            sunwall_cube,
            wall_for_floor,
            one_minus_floor_emissivity,
            floor_for_wall,
        ),
    )
    a23 = _sum(
        -_product(
            4.0,
            sunwall_emissivity,
            shadewall_emissivity,
            sb,
            shadewall_cube,
            wall_for_wall,
        ),
        -_product(
            4.0,
            shadewall_emissivity,
            sb,
            shadewall_cube,
            wall_for_floor,
            one_minus_floor_emissivity,
            floor_for_wall,
        ),
        -_product(
            4.0,
            shadewall_emissivity,
            sb,
            shadewall_cube,
            wall_for_roof,
            one_minus_roof_emissivity,
            roof_for_wall,
        ),
    )
    a24 = _sum(
        -_product(
            4.0,
            sunwall_emissivity,
            floor_emissivity,
            sb,
            floor_cube,
            floor_for_wall,
        ),
        -_product(
            4.0,
            floor_emissivity,
            sb,
            floor_cube,
            floor_for_roof,
            one_minus_roof_emissivity,
            roof_for_wall,
        ),
        -_product(
            4.0,
            floor_emissivity,
            sb,
            floor_cube,
            floor_for_wall,
            one_minus_shadewall_emissivity,
            wall_for_wall,
        ),
    )
    a25 = -_product(0.5, sunwall_convection, hwr)
    rhs2 = _sum(
        _product(
            _half_conduction(
                inputs.sunwall_thermal_conductivity,
                inputs.sunwall_boundary_temperature,
                sunwall_interval,
            ),
            hwr,
        ),
        -_product(
            _half_conduction(
                inputs.sunwall_thermal_conductivity,
                _difference(
                    sunwall_temperature,
                    inputs.previous_sunwall_boundary_temperature,
                ),
                sunwall_interval,
            ),
            hwr,
        ),
        -_product(
            3.0,
            sunwall_emissivity,
            roof_emissivity,
            sb,
            roof_fourth,
            roof_for_wall,
        ),
        -_product(
            3.0,
            sunwall_emissivity,
            shadewall_emissivity,
            sb,
            shadewall_fourth,
            wall_for_wall,
        ),
        -_product(
            3.0,
            sunwall_emissivity,
            floor_emissivity,
            sb,
            floor_fourth,
            floor_for_wall,
        ),
        _product(3.0, sunwall_emissivity, sb, sunwall_fourth),
        -_product(
            3.0,
            sunwall_emissivity,
            sb,
            sunwall_fourth,
            wall_for_roof,
            one_minus_roof_emissivity,
            roof_for_wall,
        ),
        -_product(
            3.0,
            sunwall_emissivity,
            sb,
            sunwall_fourth,
            wall_for_wall,
            one_minus_shadewall_emissivity,
            wall_for_wall,
        ),
        -_product(
            3.0,
            sunwall_emissivity,
            sb,
            sunwall_fourth,
            wall_for_floor,
            one_minus_floor_emissivity,
            floor_for_wall,
        ),
        -_product(
            3.0,
            shadewall_emissivity,
            sb,
            shadewall_fourth,
            wall_for_floor,
            one_minus_floor_emissivity,
            floor_for_wall,
        ),
        -_product(
            3.0,
            shadewall_emissivity,
            sb,
            shadewall_fourth,
            wall_for_roof,
            one_minus_roof_emissivity,
            roof_for_wall,
        ),
        -_product(
            3.0,
            roof_emissivity,
            sb,
            roof_fourth,
            roof_for_wall,
            one_minus_shadewall_emissivity,
            wall_for_wall,
        ),
        -_product(
            3.0,
            roof_emissivity,
            sb,
            roof_fourth,
            roof_for_floor,
            one_minus_floor_emissivity,
            floor_for_wall,
        ),
        -_product(
            3.0,
            floor_emissivity,
            sb,
            floor_fourth,
            floor_for_roof,
            one_minus_roof_emissivity,
            roof_for_wall,
        ),
        -_product(
            3.0,
            floor_emissivity,
            sb,
            floor_fourth,
            floor_for_wall,
            one_minus_shadewall_emissivity,
            wall_for_wall,
        ),
        -_product(
            0.5,
            sunwall_convection,
            _difference(sunwall_temperature, building_temperature),
            hwr,
        ),
    )

    a31 = _sum(
        -_product(
            4.0,
            shadewall_emissivity,
            roof_emissivity,
            sb,
            roof_cube,
            roof_for_wall,
        ),
        -_product(
            4.0,
            roof_emissivity,
            sb,
            roof_cube,
            roof_for_wall,
            one_minus_sunwall_emissivity,
            wall_for_wall,
        ),
        -_product(
            4.0,
            roof_emissivity,
            sb,
            roof_cube,
            roof_for_floor,
            one_minus_floor_emissivity,
            floor_for_wall,
        ),
    )
    a32 = _sum(
        -_product(
            4.0,
            shadewall_emissivity,
            sunwall_emissivity,
            sb,
            sunwall_cube,
            wall_for_wall,
        ),
        -_product(
            4.0,
            sunwall_emissivity,
            sb,
            sunwall_cube,
            wall_for_floor,
            one_minus_floor_emissivity,
            floor_for_wall,
        ),
        -_product(
            4.0,
            sunwall_emissivity,
            sb,
            sunwall_cube,
            wall_for_roof,
            one_minus_roof_emissivity,
            roof_for_wall,
        ),
    )
    a33 = _sum(
        _product(0.5, shadewall_convection, hwr),
        _product(
            fortran_divide(
                _product(0.5, inputs.shadewall_thermal_conductivity),
                shadewall_interval,
            ),
            hwr,
        ),
        _product(4.0, shadewall_emissivity, sb, shadewall_cube),
        -_product(
            4.0,
            shadewall_emissivity,
            sb,
            shadewall_cube,
            wall_for_roof,
            one_minus_roof_emissivity,
            roof_for_wall,
        ),
        -_product(
            4.0,
            shadewall_emissivity,
            sb,
            shadewall_cube,
            wall_for_wall,
            one_minus_sunwall_emissivity,
            wall_for_wall,
        ),
        -_product(
            4.0,
            shadewall_emissivity,
            sb,
            shadewall_cube,
            wall_for_floor,
            one_minus_floor_emissivity,
            floor_for_wall,
        ),
    )
    a34 = _sum(
        -_product(
            4.0,
            shadewall_emissivity,
            floor_emissivity,
            sb,
            floor_cube,
            floor_for_wall,
        ),
        -_product(
            4.0,
            floor_emissivity,
            sb,
            floor_cube,
            floor_for_roof,
            one_minus_roof_emissivity,
            roof_for_wall,
        ),
        -_product(
            4.0,
            floor_emissivity,
            sb,
            floor_cube,
            floor_for_wall,
            one_minus_sunwall_emissivity,
            wall_for_wall,
        ),
    )
    a35 = -_product(0.5, shadewall_convection, hwr)
    rhs3 = _sum(
        _product(
            _half_conduction(
                inputs.shadewall_thermal_conductivity,
                inputs.shadewall_boundary_temperature,
                shadewall_interval,
            ),
            hwr,
        ),
        -_product(
            _half_conduction(
                inputs.shadewall_thermal_conductivity,
                _difference(
                    shadewall_temperature,
                    inputs.previous_shadewall_boundary_temperature,
                ),
                shadewall_interval,
            ),
            hwr,
        ),
        -_product(
            3.0,
            shadewall_emissivity,
            roof_emissivity,
            sb,
            roof_fourth,
            roof_for_wall,
        ),
        -_product(
            3.0,
            shadewall_emissivity,
            sunwall_emissivity,
            sb,
            sunwall_fourth,
            wall_for_wall,
        ),
        -_product(
            3.0,
            shadewall_emissivity,
            floor_emissivity,
            sb,
            floor_fourth,
            floor_for_wall,
        ),
        _product(3.0, shadewall_emissivity, sb, shadewall_fourth),
        -_product(
            3.0,
            shadewall_emissivity,
            sb,
            shadewall_fourth,
            wall_for_roof,
            one_minus_roof_emissivity,
            roof_for_wall,
        ),
        -_product(
            3.0,
            shadewall_emissivity,
            sb,
            shadewall_fourth,
            wall_for_wall,
            one_minus_sunwall_emissivity,
            wall_for_wall,
        ),
        -_product(
            3.0,
            shadewall_emissivity,
            sb,
            shadewall_fourth,
            wall_for_floor,
            one_minus_floor_emissivity,
            floor_for_wall,
        ),
        -_product(
            3.0,
            sunwall_emissivity,
            sb,
            sunwall_fourth,
            wall_for_floor,
            one_minus_floor_emissivity,
            floor_for_wall,
        ),
        -_product(
            3.0,
            sunwall_emissivity,
            sb,
            sunwall_fourth,
            wall_for_roof,
            one_minus_roof_emissivity,
            roof_for_wall,
        ),
        -_product(
            3.0,
            roof_emissivity,
            sb,
            roof_fourth,
            roof_for_wall,
            one_minus_sunwall_emissivity,
            wall_for_wall,
        ),
        -_product(
            3.0,
            roof_emissivity,
            sb,
            roof_fourth,
            roof_for_floor,
            one_minus_floor_emissivity,
            floor_for_wall,
        ),
        -_product(
            3.0,
            floor_emissivity,
            sb,
            floor_fourth,
            floor_for_roof,
            one_minus_roof_emissivity,
            roof_for_wall,
        ),
        -_product(
            3.0,
            floor_emissivity,
            sb,
            floor_fourth,
            floor_for_wall,
            one_minus_sunwall_emissivity,
            wall_for_wall,
        ),
        -_product(
            0.5,
            shadewall_convection,
            _difference(shadewall_temperature, building_temperature),
            hwr,
        ),
    )

    a41 = _sum(
        -_product(
            4.0,
            floor_emissivity,
            roof_emissivity,
            sb,
            roof_cube,
            roof_for_floor,
        ),
        -_product(
            4.0,
            roof_emissivity,
            sb,
            roof_cube,
            roof_for_wall,
            one_minus_sunwall_emissivity,
            wall_for_floor,
        ),
        -_product(
            4.0,
            roof_emissivity,
            sb,
            roof_cube,
            roof_for_wall,
            one_minus_shadewall_emissivity,
            wall_for_floor,
        ),
    )
    a42 = _sum(
        -_product(
            4.0,
            floor_emissivity,
            sunwall_emissivity,
            sb,
            sunwall_cube,
            wall_for_floor,
        ),
        -_product(
            4.0,
            sunwall_emissivity,
            sb,
            sunwall_cube,
            wall_for_wall,
            one_minus_shadewall_emissivity,
            wall_for_floor,
        ),
        -_product(
            4.0,
            sunwall_emissivity,
            sb,
            sunwall_cube,
            wall_for_roof,
            one_minus_roof_emissivity,
            roof_for_floor,
        ),
    )
    a43 = _sum(
        -_product(
            4.0,
            floor_emissivity,
            shadewall_emissivity,
            sb,
            shadewall_cube,
            wall_for_floor,
        ),
        -_product(
            4.0,
            shadewall_emissivity,
            sb,
            shadewall_cube,
            wall_for_roof,
            one_minus_roof_emissivity,
            roof_for_floor,
        ),
        -_product(
            4.0,
            shadewall_emissivity,
            sb,
            shadewall_cube,
            wall_for_wall,
            one_minus_sunwall_emissivity,
            wall_for_floor,
        ),
    )
    a44 = _sum(
        fortran_add(floor_capacity, _product(0.5, floor_convection)),
        _product(4.0, floor_emissivity, sb, floor_cube),
        -_product(
            4.0,
            floor_emissivity,
            sb,
            floor_cube,
            floor_for_roof,
            one_minus_roof_emissivity,
            roof_for_floor,
        ),
        -_product(
            4.0,
            floor_emissivity,
            sb,
            floor_cube,
            floor_for_wall,
            one_minus_sunwall_emissivity,
            wall_for_floor,
        ),
        -_product(
            4.0,
            floor_emissivity,
            sb,
            floor_cube,
            floor_for_wall,
            one_minus_shadewall_emissivity,
            wall_for_floor,
        ),
    )
    a45 = -_product(0.5, floor_convection)
    rhs4 = _sum(
        _product(floor_capacity, floor_temperature),
        -_product(
            3.0,
            floor_emissivity,
            roof_emissivity,
            sb,
            roof_fourth,
            roof_for_floor,
        ),
        -_product(
            3.0,
            floor_emissivity,
            sunwall_emissivity,
            sb,
            sunwall_fourth,
            wall_for_floor,
        ),
        -_product(
            3.0,
            floor_emissivity,
            shadewall_emissivity,
            sb,
            shadewall_fourth,
            wall_for_floor,
        ),
        _product(3.0, floor_emissivity, sb, floor_fourth),
        -_product(
            3.0,
            floor_emissivity,
            sb,
            floor_fourth,
            floor_for_roof,
            one_minus_roof_emissivity,
            roof_for_floor,
        ),
        -_product(
            3.0,
            floor_emissivity,
            sb,
            floor_fourth,
            floor_for_wall,
            one_minus_sunwall_emissivity,
            wall_for_floor,
        ),
        -_product(
            3.0,
            floor_emissivity,
            sb,
            floor_fourth,
            floor_for_wall,
            one_minus_shadewall_emissivity,
            wall_for_floor,
        ),
        -_product(
            3.0,
            sunwall_emissivity,
            sb,
            sunwall_fourth,
            wall_for_wall,
            one_minus_shadewall_emissivity,
            wall_for_floor,
        ),
        -_product(
            3.0,
            sunwall_emissivity,
            sb,
            sunwall_fourth,
            wall_for_roof,
            one_minus_roof_emissivity,
            roof_for_floor,
        ),
        -_product(
            3.0,
            shadewall_emissivity,
            sb,
            shadewall_fourth,
            wall_for_roof,
            one_minus_roof_emissivity,
            roof_for_floor,
        ),
        -_product(
            3.0,
            shadewall_emissivity,
            sb,
            shadewall_fourth,
            wall_for_wall,
            one_minus_sunwall_emissivity,
            wall_for_floor,
        ),
        -_product(
            3.0,
            roof_emissivity,
            sb,
            roof_fourth,
            roof_for_wall,
            one_minus_sunwall_emissivity,
            wall_for_floor,
        ),
        -_product(
            3.0,
            roof_emissivity,
            sb,
            roof_fourth,
            roof_for_wall,
            one_minus_shadewall_emissivity,
            wall_for_floor,
        ),
        -_product(
            0.5,
            floor_convection,
            _difference(floor_temperature, building_temperature),
        ),
    )

    a51 = -_product(0.5, roof_convection)
    a52 = -_product(0.5, sunwall_convection, hwr)
    a53 = -_product(0.5, shadewall_convection, hwr)
    a54 = -_product(0.5, floor_convection)
    building_capacity = fortran_divide(
        _product(
            inputs.roof_height,
            dry_air_density,
            parameters.dry_air_specific_heat,
        ),
        time_step_seconds,
    )
    ventilation_coefficient = _product(
        fortran_divide(
            _product(
                inputs.roof_height,
                parameters.ventilation_air_changes_per_hour,
            ),
            3600.0,
        ),
        dry_air_density,
        parameters.dry_air_specific_heat,
    )
    a55 = _sum(
        building_capacity,
        ventilation_coefficient,
        _product(0.5, roof_convection),
        _product(0.5, sunwall_convection, hwr),
        _product(0.5, shadewall_convection, hwr),
        _product(0.5, floor_convection),
    )
    rhs5 = _sum(
        _product(building_capacity, building_temperature),
        _product(
            ventilation_coefficient,
            inputs.canyon_air_temperature,
        ),
        _product(
            0.5,
            roof_convection,
            _difference(roof_temperature, building_temperature),
        ),
        _product(
            0.5,
            sunwall_convection,
            _difference(sunwall_temperature, building_temperature),
            hwr,
        ),
        _product(
            0.5,
            shadewall_convection,
            _difference(shadewall_temperature, building_temperature),
            hwr,
        ),
        _product(
            0.5,
            floor_convection,
            _difference(floor_temperature, building_temperature),
        ),
    )

    matrix = jnp.stack(
        (
            jnp.stack((a11, a12, a13, a14, a15)),
            jnp.stack((a21, a22, a23, a24, a25)),
            jnp.stack((a31, a32, a33, a34, a35)),
            jnp.stack((a41, a42, a43, a44, a45)),
            jnp.stack((a51, a52, a53, a54, a55)),
        )
    )
    right_hand_side = jnp.stack((rhs1, rhs2, rhs3, rhs4, rhs5))
    return matrix, right_hand_side, dry_air_density


def _active_building_temperature(
    inputs: UrbanBuildingTemperatureInputs,
    parameters: UrbanBuildingTemperatureParameters,
    time_step_seconds: jnp.ndarray,
    heating_air_conditioning_enabled: jnp.ndarray,
) -> UrbanBuildingTemperatureResult:
    matrix, right_hand_side, dry_air_density = _assemble_building_system(
        inputs,
        parameters,
        time_step_seconds,
    )
    solution, pivots, nonsingular = _dgesv_five(matrix, right_hand_side)
    unconstrained_temperature = solution[4]

    building_capacity = fortran_divide(
        _product(
            inputs.roof_height,
            dry_air_density,
            parameters.dry_air_specific_heat,
        ),
        time_step_seconds,
    )

    def apply_hac(_: None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        above_maximum = (
            unconstrained_temperature > inputs.maximum_building_temperature
        )
        below_minimum = (
            unconstrained_temperature < inputs.minimum_building_temperature
        )
        final_temperature = jnp.where(
            above_maximum,
            inputs.maximum_building_temperature,
            jnp.where(
                below_minimum,
                inputs.minimum_building_temperature,
                unconstrained_temperature,
            ),
        )
        hac_flux = _product(
            inputs.roof_weight,
            jnp.abs(
                _difference(
                    _product(building_capacity, final_temperature),
                    _product(building_capacity, unconstrained_temperature),
                )
            ),
        )
        air_conditioning_flux = jnp.where(
            above_maximum,
            hac_flux,
            jnp.where(
                below_minimum,
                inputs.previous_air_conditioning_flux,
                jnp.float64(0.0),
            ),
        )
        heating_flux = jnp.where(
            below_minimum,
            hac_flux,
            jnp.where(
                above_maximum,
                inputs.previous_heating_flux,
                jnp.float64(0.0),
            ),
        )
        return final_temperature, air_conditioning_flux, heating_flux

    def disable_hac(_: None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return unconstrained_temperature, jnp.float64(0.0), jnp.float64(0.0)

    final_temperature, air_conditioning_flux, heating_flux = lax.cond(
        heating_air_conditioning_enabled,
        apply_hac,
        disable_hac,
        operand=None,
    )
    building_heat_flux = _product(
        inputs.roof_weight,
        building_capacity,
        _difference(final_temperature, inputs.building_temperature),
    )
    return UrbanBuildingTemperatureResult(
        roof_inner_temperature=solution[0],
        sunwall_inner_temperature=solution[1],
        shadewall_inner_temperature=solution[2],
        floor_temperature=solution[3],
        building_temperature=final_temperature,
        unconstrained_building_temperature=unconstrained_temperature,
        building_heat_flux=building_heat_flux,
        air_conditioning_flux=air_conditioning_flux,
        heating_flux=heating_flux,
        coefficient_matrix=matrix,
        right_hand_side=right_hand_side,
        pivot_indices=pivots,
        nonsingular=nonsingular,
    )


def _building_temperature_one(
    inputs: UrbanBuildingTemperatureInputs,
    parameters: UrbanBuildingTemperatureParameters,
    time_step_seconds: jnp.ndarray,
    heating_air_conditioning_enabled: jnp.ndarray,
) -> UrbanBuildingTemperatureResult:
    inactive_result = UrbanBuildingTemperatureResult(
        roof_inner_temperature=inputs.roof_inner_temperature,
        sunwall_inner_temperature=inputs.sunwall_inner_temperature,
        shadewall_inner_temperature=inputs.shadewall_inner_temperature,
        floor_temperature=inputs.floor_temperature,
        building_temperature=inputs.building_temperature,
        unconstrained_building_temperature=inputs.building_temperature,
        building_heat_flux=inputs.previous_building_heat_flux,
        air_conditioning_flux=inputs.previous_air_conditioning_flux,
        heating_flux=inputs.previous_heating_flux,
        coefficient_matrix=jnp.zeros(
            (NUMBER_OF_BUILDING_EQUATIONS, NUMBER_OF_BUILDING_EQUATIONS),
            dtype=jnp.float64,
        ),
        right_hand_side=jnp.zeros(
            (NUMBER_OF_BUILDING_EQUATIONS,), dtype=jnp.float64
        ),
        pivot_indices=jnp.arange(
            NUMBER_OF_BUILDING_EQUATIONS, dtype=jnp.int32
        ),
        nonsingular=jnp.bool_(False),
    )
    return lax.cond(
        jnp.asarray(inputs.active_urban, dtype=jnp.bool_),
        lambda _: _active_building_temperature(
            inputs,
            parameters,
            time_step_seconds,
            heating_air_conditioning_enabled,
        ),
        lambda _: inactive_result,
        operand=None,
    )


def building_temperature(
    inputs: UrbanBuildingTemperatureInputs,
    parameters: UrbanBuildingTemperatureParameters | None = None,
    *,
    time_step_seconds: float = 1800.0,
    heating_air_conditioning_enabled: bool = True,
) -> UrbanBuildingTemperatureResult:

    _validate_inputs(inputs)
    if time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")
    if parameters is None:
        parameters = default_urban_building_temperature_parameters()
    return jax.vmap(
        _building_temperature_one,
        in_axes=(0, None, None, None),
    )(
        inputs,
        parameters,
        jnp.float64(time_step_seconds),
        jnp.bool_(heating_air_conditioning_enabled),
    )


def active_building_temperature(
    inputs: UrbanBuildingTemperatureInputs,
    parameters: UrbanBuildingTemperatureParameters | None = None,
    *,
    time_step_seconds: float = 1800.0,
    heating_air_conditioning_enabled: bool = True,
) -> UrbanBuildingTemperatureResult:

    _validate_inputs(inputs)
    if time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")
    if parameters is None:
        parameters = default_urban_building_temperature_parameters()
    return jax.vmap(
        _active_building_temperature,
        in_axes=(0, None, None, None),
    )(
        inputs,
        parameters,
        jnp.float64(time_step_seconds),
        jnp.bool_(heating_air_conditioning_enabled),
    )
