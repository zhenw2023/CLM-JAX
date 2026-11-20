

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.constants import (
    DENSITY_FRESH_WATER,
    DENSITY_ICE,
    FREEZING_TEMPERATURE_FRESHWATER,
    GRAVITY,
    LATENT_HEAT_FUSION,
    VON_KARMAN,
)
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
    fortran_power,
    strict_negative_exp,
)

NUMBER_OF_SNOW_LEVELS = 12
NUMBER_OF_LAKE_LEVELS = 10
NUMBER_OF_SOIL_LEVELS = 20
NUMBER_OF_GROUND_LEVELS = 25
NUMBER_OF_SOIL_SNOW_LEVELS = NUMBER_OF_SNOW_LEVELS + NUMBER_OF_GROUND_LEVELS
NUMBER_OF_COUPLED_LEVELS = NUMBER_OF_SNOW_LEVELS + NUMBER_OF_LAKE_LEVELS + NUMBER_OF_GROUND_LEVELS
NUMBER_OF_SHORTWAVE_LEVELS = NUMBER_OF_SNOW_LEVELS + 1

SPECIFIC_HEAT_WATER = 4.188e3
SPECIFIC_HEAT_ICE = 2.11727e3
THERMAL_CONDUCTIVITY_AIR = 0.023
THERMAL_CONDUCTIVITY_ICE = 2.290
THERMAL_CONDUCTIVITY_WATER = 0.57
THERMAL_CONDUCTIVITY_BEDROCK = 3.0
VOLUMETRIC_HEAT_CAPACITY_BEDROCK = 2.0e6
MAXIMUM_DENSITY_TEMPERATURE = 277.0
SURFACE_ABSORPTION_DEPTH = 0.6
VISIBLE_SURFACE_ABSORPTION_FRACTION = 0.0
MINIMUM_BRUNT_VAISALA_FREQUENCY_SQUARED = 7.5e-5
DEEP_MIXING_DEPTH = 25.0
DEEP_MIXING_FACTOR = 10.0
CRANK_NICHOLSON_FACTOR = 0.5
PHASE_CHANGE_RESIDUAL = 1.0e-12

_PARALLEL_LEVEL_ENTRIES: bool | None = None


def _map_level_entries(entry, initial, start, stop, *, offset=0):
    parallel = (
        jax.default_backend() == "gpu"
        if _PARALLEL_LEVEL_ENTRIES is None else _PARALLEL_LEVEL_ENTRIES
    )
    if parallel:
        values = jax.vmap(entry)(jnp.arange(start, stop, dtype=jnp.int32))
        return jax.tree.map(
            lambda array, value: array.at[start + offset:stop + offset].set(value),
            initial, values,
        )

    def advance(index, arrays):
        return jax.tree.map(
            lambda array, value: array.at[index + offset].set(value), arrays, entry(index),
        )

    return lax.fori_loop(start, stop, advance, initial)


class LakeTemperatureGeometry(NamedTuple):

    soil_snow_layer_thickness: jnp.ndarray
    soil_snow_node_depth: jnp.ndarray
    soil_snow_interface_depth: jnp.ndarray
    lake_layer_thickness: jnp.ndarray
    lake_node_depth: jnp.ndarray
    lake_depth: jnp.ndarray
    light_extinction_coefficient: jnp.ndarray
    saturated_water_content: jnp.ndarray
    saturated_thermal_conductivity: jnp.ndarray
    mineral_thermal_conductivity: jnp.ndarray
    dry_thermal_conductivity: jnp.ndarray
    solid_volumetric_heat_capacity: jnp.ndarray


class LakeTemperatureInputs(NamedTuple):

    active_lake: jnp.ndarray
    ground_temperature: jnp.ndarray
    ground_absorbed_shortwave: jnp.ndarray
    layer_absorbed_shortwave: jnp.ndarray
    incident_nir_direct: jnp.ndarray
    incident_nir_diffuse: jnp.ndarray
    reflected_nir_direct: jnp.ndarray
    reflected_nir_diffuse: jnp.ndarray
    wind_mixing_velocity: jnp.ndarray
    mixing_decay_coefficient: jnp.ndarray
    lake_moisture_aerodynamic_resistance: jnp.ndarray
    soil_ground_heat_flux: jnp.ndarray
    sensible_ground_heat_flux: jnp.ndarray
    sensible_total_heat_flux: jnp.ndarray
    ground_net_heat_flux: jnp.ndarray


class LakeColumnTemperatureState(NamedTuple):

    snow_layer_count: jnp.ndarray
    snow_depth: jnp.ndarray
    snow_water: jnp.ndarray
    soil_snow_temperature: jnp.ndarray
    soil_snow_liquid_water: jnp.ndarray
    soil_snow_ice_water: jnp.ndarray
    soil_snow_old_ice_fraction: jnp.ndarray
    soil_snow_phase_flag: jnp.ndarray
    lake_temperature: jnp.ndarray
    lake_ice_fraction: jnp.ndarray
    saved_top_eddy_conductivity: jnp.ndarray
    surface_absorption_fraction: jnp.ndarray
    lake_surface_ice_fraction: jnp.ndarray
    lake_ice_thickness: jnp.ndarray
    lake_resistance: jnp.ndarray
    ground_methane_conductance: jnp.ndarray


class LakeTemperatureEnergyFluxes(NamedTuple):
    soil_ground_heat: jnp.ndarray
    sensible_ground: jnp.ndarray
    sensible_total: jnp.ndarray
    ground_net: jnp.ndarray
    snow_melt_heat: jnp.ndarray
    conservation_error: jnp.ndarray


class LakeTemperatureWaterFluxes(NamedTuple):
    snow_melt: jnp.ndarray
    snow_freeze: jnp.ndarray
    snow_drain: jnp.ndarray
    snow_layer_melt: jnp.ndarray
    snow_layer_freeze: jnp.ndarray
    snow_layer_phase_flag: jnp.ndarray


class LakeTemperatureDiagnostics(NamedTuple):
    lake_density_before_diffusion: jnp.ndarray
    lake_diffusivity: jnp.ndarray
    lake_thermal_conductivity: jnp.ndarray
    lake_heat_capacity_before_phase_change: jnp.ndarray
    soil_snow_heat_capacity_before_phase_change: jnp.ndarray
    coupled_interface_conductivity: jnp.ndarray
    coupled_timestep_over_capacity: jnp.ndarray
    coupled_diffusion_flux: jnp.ndarray
    coupled_lower_diagonal: jnp.ndarray
    coupled_diagonal: jnp.ndarray
    coupled_upper_diagonal: jnp.ndarray
    coupled_right_hand_side: jnp.ndarray
    coupled_temperature_after_diffusion: jnp.ndarray
    lake_temperature_after_phase_change: jnp.ndarray
    lake_ice_fraction_after_phase_change: jnp.ndarray
    soil_snow_temperature_after_phase_change: jnp.ndarray
    soil_snow_liquid_after_phase_change: jnp.ndarray
    soil_snow_ice_after_phase_change: jnp.ndarray
    lake_temperature_after_convection: jnp.ndarray
    lake_ice_fraction_after_convection: jnp.ndarray
    lake_absorbed_shortwave: jnp.ndarray
    soil_absorbed_shortwave: jnp.ndarray
    latent_heat_absorbed: jnp.ndarray
    initial_column_energy: jnp.ndarray
    final_column_energy: jnp.ndarray
    applied_column_energy_flux: jnp.ndarray
    raw_conservation_error: jnp.ndarray
    deepest_top_convection_level: jnp.ndarray
    shallowest_bottom_convection_level: jnp.ndarray


class LakeTemperatureResult(NamedTuple):
    state: LakeColumnTemperatureState
    energy: LakeTemperatureEnergyFluxes
    water: LakeTemperatureWaterFluxes
    diagnostics: LakeTemperatureDiagnostics


def _add(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(left, dtype=jnp.float64) + jnp.asarray(right, dtype=jnp.float64)


def _subtract(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return _add(left, -jnp.asarray(right, dtype=jnp.float64))


def _multiply(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(left, dtype=jnp.float64) * jnp.asarray(right, dtype=jnp.float64)


def _divide(numerator: jnp.ndarray, denominator: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(numerator, dtype=jnp.float64) / jnp.asarray(
        denominator,
        dtype=jnp.float64,
    )


def _power(base: jnp.ndarray, exponent: float | jnp.ndarray) -> jnp.ndarray:
    from clm_jax.physics.solver_scalarization import _native_tile_trace

    if _native_tile_trace.get() and isinstance(exponent, (int, float)) and exponent == 2.:
        return fortran_multiply(base, base)
    return jnp.power(
        jnp.asarray(base, dtype=jnp.float64),
        jnp.asarray(exponent, dtype=jnp.float64),
    )


@jax.custom_jvp
def _nonnegative_square_root(value: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.maximum(value, jnp.float64(0.0)))


@_nonnegative_square_root.defjvp
def _nonnegative_square_root_jvp(primals, tangents):
    def guarded(value):
        negative = value < jnp.float64(0.0)
        clamped = jnp.maximum(value, jnp.float64(0.0))
        root = jnp.sqrt(jnp.where(negative, jnp.float64(1.0), clamped))
        return jnp.where(negative, jnp.float64(0.0), root)

    return _nonnegative_square_root(*primals), jax.jvp(guarded, primals, tangents)[1]


def _sum_terms(*terms: jnp.ndarray) -> jnp.ndarray:
    result = jnp.asarray(terms[0], dtype=jnp.float64)
    for term in terms[1:]:
        result = _add(result, term)
    return result


def _lake_density(
    temperature: jnp.ndarray,
    ice_fraction: jnp.ndarray,
) -> jnp.ndarray:
    return _strict_lake_density(temperature, ice_fraction)


def _strict_lake_density(
    temperature: jnp.ndarray,
    ice_fraction: jnp.ndarray,
) -> jnp.ndarray:

    temperature_distance = jnp.abs(
        fortran_add(temperature, jnp.float64(-MAXIMUM_DENSITY_TEMPERATURE))
    )
    liquid_weight = fortran_multiply(
        fortran_add(jnp.float64(1.0), -ice_fraction),
        jnp.float64(1000.0),
    )
    liquid_correction = fortran_add(
        jnp.float64(1.0),
        -fortran_multiply(
            jnp.float64(1.9549e-5),
            _power(temperature_distance, 1.68),
        ),
    )
    return fortran_add(
        fortran_multiply(
            liquid_weight,
            liquid_correction,
        ),
        fortran_multiply(ice_fraction, jnp.float64(DENSITY_ICE)),
    )


def _validate_shapes(
    inputs: LakeTemperatureInputs,
    geometry: LakeTemperatureGeometry,
    state: LakeColumnTemperatureState,
) -> None:
    number_of_columns = inputs.active_lake.shape[0]
    vector_shape = (number_of_columns,)
    soil_snow_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS)
    lake_shape = (number_of_columns, NUMBER_OF_LAKE_LEVELS)
    ground_shape = (number_of_columns, NUMBER_OF_GROUND_LEVELS)
    shortwave_shape = (number_of_columns, NUMBER_OF_SHORTWAVE_LEVELS)

    if inputs.active_lake.ndim != 1:
        raise ValueError("active_lake must be a vector")
    for name in set(LakeTemperatureInputs._fields) - {"layer_absorbed_shortwave"}:
        if getattr(inputs, name).shape != vector_shape:
            raise ValueError(f"{name} must have shape {vector_shape}")
    if inputs.layer_absorbed_shortwave.shape != shortwave_shape:
        raise ValueError(f"layer_absorbed_shortwave must have shape {shortwave_shape}")
    for name in (
        "soil_snow_layer_thickness",
        "soil_snow_node_depth",
        "soil_snow_interface_depth",
    ):
        if getattr(geometry, name).shape != soil_snow_shape:
            raise ValueError(f"{name} must have shape {soil_snow_shape}")
    for name in ("lake_layer_thickness", "lake_node_depth"):
        if getattr(geometry, name).shape != lake_shape:
            raise ValueError(f"{name} must have shape {lake_shape}")
    for name in (
        "saturated_water_content",
        "saturated_thermal_conductivity",
        "mineral_thermal_conductivity",
        "dry_thermal_conductivity",
        "solid_volumetric_heat_capacity",
    ):
        if getattr(geometry, name).shape != ground_shape:
            raise ValueError(f"{name} must have shape {ground_shape}")
    for name in ("lake_depth", "light_extinction_coefficient"):
        if getattr(geometry, name).shape != vector_shape:
            raise ValueError(f"{name} must have shape {vector_shape}")
    for name in (
        "soil_snow_temperature",
        "soil_snow_liquid_water",
        "soil_snow_ice_water",
        "soil_snow_old_ice_fraction",
        "soil_snow_phase_flag",
    ):
        if getattr(state, name).shape != soil_snow_shape:
            raise ValueError(f"{name} must have shape {soil_snow_shape}")
    for name in ("lake_temperature", "lake_ice_fraction"):
        if getattr(state, name).shape != lake_shape:
            raise ValueError(f"{name} must have shape {lake_shape}")
    for name in set(LakeColumnTemperatureState._fields) - {
        "soil_snow_temperature",
        "soil_snow_liquid_water",
        "soil_snow_ice_water",
        "soil_snow_old_ice_fraction",
        "soil_snow_phase_flag",
        "lake_temperature",
        "lake_ice_fraction",
    }:
        if getattr(state, name).shape != vector_shape:
            raise ValueError(f"{name} must have shape {vector_shape}")


def _soil_thermal_properties(
    snow_layer_count: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    node_depth: jnp.ndarray,
    interface_depth: jnp.ndarray,
    saturated_water_content: jnp.ndarray,
    saturated_thermal_conductivity: jnp.ndarray,
    mineral_thermal_conductivity: jnp.ndarray,
    dry_thermal_conductivity: jnp.ndarray,
    solid_heat_capacity: jnp.ndarray,
    liquid_water: jnp.ndarray,
    ice_water: jnp.ndarray,
    temperature: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    tfrz = jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)
    denh2o = jnp.float64(DENSITY_FRESH_WATER)
    denice = jnp.float64(DENSITY_ICE)
    top_index = snow_layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS)
    layer_conductivity = jnp.zeros((NUMBER_OF_SOIL_SNOW_LEVELS,), dtype=jnp.float64)

    def soil_conductivity_entry(level):
        index = NUMBER_OF_SNOW_LEVELS + level
        total_water = fortran_add(ice_water[index], liquid_water[index])
        liquid_fraction = fortran_divide(liquid_water[index], total_water)

        def unfrozen(_: None) -> jnp.ndarray:
            return saturated_thermal_conductivity[level]

        def frozen(_: None) -> jnp.ndarray:
            liquid_exponent = fortran_multiply(
                liquid_fraction,
                saturated_water_content[level],
            )
            return fortran_multiply(
                fortran_multiply(
                    mineral_thermal_conductivity[level],
                    _power(jnp.float64(0.249), liquid_exponent),
                ),
                _power(
                    jnp.float64(2.29),
                    saturated_water_content[level],
                ),
            )

        saturated_conductivity = lax.cond(
            temperature[index] >= tfrz,
            unfrozen,
            frozen,
            operand=None,
        )
        conductivity = fortran_add(
            saturated_conductivity,
            fortran_multiply(jnp.float64(0.0), dry_thermal_conductivity[level]),
        )
        relative_water = fortran_divide(
            fortran_add(
                fortran_divide(liquid_water[index], denh2o),
                fortran_divide(ice_water[index], denice),
            ),
            fortran_multiply(
                layer_thickness[index],
                saturated_water_content[level],
            ),
        )

        def with_excess_ice(_: None) -> jnp.ndarray:
            excess_volume = fortran_multiply(
                fortran_add(relative_water, jnp.float64(-1.0)),
                saturated_water_content[level],
            )
            numerator = fortran_add(
                conductivity,
                fortran_multiply(
                    excess_volume,
                    jnp.float64(THERMAL_CONDUCTIVITY_ICE),
                ),
            )
            scale = fortran_add(jnp.float64(1.0), excess_volume)
            return fortran_divide(fortran_divide(numerator, scale), scale)

        conductivity = lax.cond(
            relative_water > jnp.float64(1.0),
            with_excess_ice,
            lambda _: conductivity,
            operand=None,
        )
        return conductivity

    layer_conductivity = _map_level_entries(
        soil_conductivity_entry, layer_conductivity, 0, NUMBER_OF_SOIL_LEVELS,
        offset=NUMBER_OF_SNOW_LEVELS,
    )

    layer_conductivity = layer_conductivity.at[
        NUMBER_OF_SNOW_LEVELS + NUMBER_OF_SOIL_LEVELS:
    ].set(jnp.float64(THERMAL_CONDUCTIVITY_BEDROCK))

    def snow_conductivity_entry(index):
        def active_snow(_):
            bulk_density = fortran_divide(
                fortran_add(ice_water[index], liquid_water[index]),
                layer_thickness[index],
            )
            snow_factor = fortran_add(
                fortran_multiply(jnp.float64(7.75e-5), bulk_density),
                fortran_multiply(
                    fortran_multiply(jnp.float64(1.105e-6), bulk_density),
                    bulk_density,
                ),
            )
            snow_conductivity = fortran_add(
                jnp.float64(THERMAL_CONDUCTIVITY_AIR),
                fortran_multiply(
                    snow_factor,
                    fortran_add(
                        jnp.float64(THERMAL_CONDUCTIVITY_ICE),
                        -jnp.float64(THERMAL_CONDUCTIVITY_AIR),
                    ),
                ),
            )
            return snow_conductivity

        return lax.cond(
            index >= top_index,
            active_snow,
            lambda _: jnp.float64(0.0),
            None,
        )

    layer_conductivity = _map_level_entries(
        snow_conductivity_entry, layer_conductivity, 0, NUMBER_OF_SNOW_LEVELS,
    )

    interface_conductivity = jnp.zeros_like(layer_conductivity)

    def interface_conductivity_entry(index):
        fortran_level = index - (NUMBER_OF_SNOW_LEVELS - 1)

        def active_level(_):
            def layer_interface(_):
                next_index = index + 1
                distance = fortran_add(
                    node_depth[next_index],
                    -node_depth[index],
                )
                numerator = fortran_multiply(
                    fortran_multiply(
                        layer_conductivity[index],
                        layer_conductivity[next_index],
                    ),
                    distance,
                )
                denominator = fortran_add(
                    fortran_multiply(
                        layer_conductivity[index],
                        fortran_add(
                            node_depth[next_index],
                            -interface_depth[index],
                        ),
                    ),
                    fortran_multiply(
                        layer_conductivity[next_index],
                        fortran_add(interface_depth[index], -node_depth[index]),
                    ),
                )
                value = fortran_divide(numerator, denominator)
                return value

            def bottom_snow(_):
                return layer_conductivity[index]

            return lax.cond(
                fortran_level == 0,
                bottom_snow,
                lambda _: lax.cond(
                    fortran_level <= NUMBER_OF_GROUND_LEVELS - 1,
                    layer_interface,
                    lambda _: jnp.float64(0.0),
                    None,
                ),
                None,
            )

        return lax.cond(
            index >= top_index,
            active_level,
            lambda _: jnp.float64(0.0),
            None,
        )

    interface_conductivity = _map_level_entries(
        interface_conductivity_entry, interface_conductivity, 0, NUMBER_OF_SOIL_SNOW_LEVELS,
    )

    heat_capacity = jnp.zeros_like(layer_conductivity)

    def ground_capacity_entry(level):
        index = NUMBER_OF_SNOW_LEVELS + level

        def soil_capacity(_: None):
            solid = fortran_multiply(
                fortran_multiply(
                    solid_heat_capacity[level],
                    fortran_add(
                        jnp.float64(1.0),
                        -saturated_water_content[level],
                    ),
                ),
                layer_thickness[index],
            )
            water = fortran_add(
                fortran_multiply(
                    ice_water[index],
                    jnp.float64(SPECIFIC_HEAT_ICE),
                ),
                fortran_multiply(
                    liquid_water[index],
                    jnp.float64(SPECIFIC_HEAT_WATER),
                ),
            )
            return fortran_add(solid, water)

        def bedrock_capacity(_: None):
            return fortran_multiply(
                jnp.float64(VOLUMETRIC_HEAT_CAPACITY_BEDROCK),
                layer_thickness[index],
            )

        value = lax.cond(
            level < NUMBER_OF_SOIL_LEVELS,
            soil_capacity,
            bedrock_capacity,
            operand=None,
        )
        return value

    heat_capacity = _map_level_entries(
        ground_capacity_entry, heat_capacity, 0, NUMBER_OF_GROUND_LEVELS,
        offset=NUMBER_OF_SNOW_LEVELS,
    )

    def snow_capacity_entry(index):
        def active_snow(_):
            value = fortran_add(
                fortran_multiply(
                    liquid_water[index],
                    jnp.float64(SPECIFIC_HEAT_WATER),
                ),
                fortran_multiply(
                    ice_water[index],
                    jnp.float64(SPECIFIC_HEAT_ICE),
                ),
            )
            return value

        return lax.cond(
            index >= top_index,
            active_snow,
            lambda _: jnp.float64(0.0),
            None,
        )

    heat_capacity = _map_level_entries(
        snow_capacity_entry, heat_capacity, 0, NUMBER_OF_SNOW_LEVELS,
    )

    return (
        interface_conductivity,
        heat_capacity,
        layer_conductivity[NUMBER_OF_SNOW_LEVELS],
    )


def _tridiagonal_solve_reference(
    lower: jnp.ndarray,
    diagonal: jnp.ndarray,
    upper: jnp.ndarray,
    right_hand_side: jnp.ndarray,
    initial_solution: jnp.ndarray,
    top_index: jnp.ndarray,
) -> jnp.ndarray:

    def forward_step(carry, row):
        beta, previous_solution = carry
        index, lower_value, diagonal_value, upper_value, rhs, original = row
        interior = index > top_index
        active = index >= top_index
        gamma_value = fortran_divide(
            jnp.where(interior, upper_value, 0.0), jnp.where(interior, beta, 1.0),
        )
        lower_value = jnp.where(interior, lower_value, 0.0)
        diagonal_value = jnp.where(interior, diagonal_value, 0.0)
        rhs = jnp.where(active, rhs, 0.0)
        next_beta = jnp.where(interior, fortran_add(
            diagonal_value, -fortran_multiply(lower_value, gamma_value),
        ), beta)
        numerator = jnp.where(interior, fortran_add(
            rhs, -fortran_multiply(lower_value, previous_solution),
        ), rhs)
        value = fortran_divide(
            jnp.where(active, numerator, 0.0), jnp.where(active, next_beta, 1.0),
        )
        return (next_beta, jnp.where(active, value, previous_solution)), (
            jnp.where(active, value, original), gamma_value,
        )

    _, (solution, gamma) = lax.scan(
        forward_step,
        (diagonal[top_index], jnp.float64(0.0)),
        (
            jnp.arange(NUMBER_OF_COUPLED_LEVELS, dtype=jnp.int32),
            lower, diagonal, jnp.concatenate((jnp.zeros((1,), upper.dtype), upper[:-1])),
            right_hand_side, initial_solution,
        ),
    )

    def backward_step(next_solution, row):
        index, value, next_gamma = row
        active = index >= top_index
        update = fortran_add(
            jnp.where(active, value, 0.0),
            -fortran_multiply(next_gamma, jnp.where(active, next_solution, 0.0)),
        )
        corrected = jnp.where(active, update, value)
        return corrected, corrected

    _, corrected = lax.scan(
        backward_step, solution[-1],
        (jnp.arange(NUMBER_OF_COUPLED_LEVELS-1, dtype=jnp.int32), solution[:-1], gamma[1:]),
        reverse=True,
    )
    return jnp.concatenate((corrected, solution[-1:]))


def _tridiagonal_solve(*arguments):
    from clm_jax.physics.solver_scalarization import _native_tile_trace

    if jax.default_backend() == "gpu" and not _native_tile_trace.get():
        from clm_jax.physics.lake_temperature_gpu import lake_tridiagonal_solve

        return lake_tridiagonal_solve(*arguments)
    return _tridiagonal_solve_reference(*arguments)


def _inactive_column_result(
    inputs: LakeTemperatureInputs,
    state: LakeColumnTemperatureState,
) -> LakeTemperatureResult:
    zero = jnp.float64(0.0)
    zero_lake = jnp.zeros((NUMBER_OF_LAKE_LEVELS,), dtype=jnp.float64)
    zero_soil_snow = jnp.zeros((NUMBER_OF_SOIL_SNOW_LEVELS,), dtype=jnp.float64)
    zero_coupled = jnp.zeros((NUMBER_OF_COUPLED_LEVELS,), dtype=jnp.float64)
    zero_snow = jnp.zeros((NUMBER_OF_SNOW_LEVELS,), dtype=jnp.float64)
    return LakeTemperatureResult(
        state=state,
        energy=LakeTemperatureEnergyFluxes(
            soil_ground_heat=inputs.soil_ground_heat_flux,
            sensible_ground=inputs.sensible_ground_heat_flux,
            sensible_total=inputs.sensible_total_heat_flux,
            ground_net=inputs.ground_net_heat_flux,
            snow_melt_heat=zero,
            conservation_error=zero,
        ),
        water=LakeTemperatureWaterFluxes(
            snow_melt=zero,
            snow_freeze=zero,
            snow_drain=zero,
            snow_layer_melt=zero_snow,
            snow_layer_freeze=zero_snow,
            snow_layer_phase_flag=jnp.zeros((NUMBER_OF_SNOW_LEVELS,), dtype=jnp.int32),
        ),
        diagnostics=LakeTemperatureDiagnostics(
            lake_density_before_diffusion=zero_lake,
            lake_diffusivity=zero_lake,
            lake_thermal_conductivity=zero_lake,
            lake_heat_capacity_before_phase_change=zero_lake,
            soil_snow_heat_capacity_before_phase_change=zero_soil_snow,
            coupled_interface_conductivity=zero_coupled,
            coupled_timestep_over_capacity=zero_coupled,
            coupled_diffusion_flux=zero_coupled,
            coupled_lower_diagonal=zero_coupled,
            coupled_diagonal=zero_coupled,
            coupled_upper_diagonal=zero_coupled,
            coupled_right_hand_side=zero_coupled,
            coupled_temperature_after_diffusion=zero_coupled,
            lake_temperature_after_phase_change=zero_lake,
            lake_ice_fraction_after_phase_change=zero_lake,
            soil_snow_temperature_after_phase_change=zero_soil_snow,
            soil_snow_liquid_after_phase_change=zero_soil_snow,
            soil_snow_ice_after_phase_change=zero_soil_snow,
            lake_temperature_after_convection=zero_lake,
            lake_ice_fraction_after_convection=zero_lake,
            lake_absorbed_shortwave=zero_lake,
            soil_absorbed_shortwave=zero,
            latent_heat_absorbed=zero,
            initial_column_energy=zero,
            final_column_energy=zero,
            applied_column_energy_flux=zero,
            raw_conservation_error=zero,
            deepest_top_convection_level=jnp.int32(0),
            shallowest_bottom_convection_level=jnp.int32(NUMBER_OF_LAKE_LEVELS + 1),
        ),
    )


def _active_column_step(
    inputs: LakeTemperatureInputs,
    geometry: LakeTemperatureGeometry,
    state: LakeColumnTemperatureState,
    time_step_seconds: float,
) -> LakeTemperatureResult:
    tfrz = jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)
    denh2o = jnp.float64(DENSITY_FRESH_WATER)
    denice = jnp.float64(DENSITY_ICE)
    hfus = jnp.float64(LATENT_HEAT_FUSION)
    cpliq = jnp.float64(SPECIFIC_HEAT_WATER)
    cpice = jnp.float64(SPECIFIC_HEAT_ICE)
    dtime = jnp.float64(time_step_seconds)
    cwat = fortran_multiply(cpliq, denh2o)
    cice_effective = fortran_multiply(cpice, denh2o)
    fusion_per_volume = fortran_multiply(hfus, denh2o)
    ice_conductivity_effective = fortran_divide(
        fortran_multiply(jnp.float64(THERMAL_CONDUCTIVITY_ICE), denice),
        denh2o,
    )
    molecular_diffusivity = fortran_divide(
        jnp.float64(THERMAL_CONDUCTIVITY_WATER),
        cwat,
    )

    snow_layer_count = state.snow_layer_count
    top_index = snow_layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS)
    soil_temperature = state.soil_snow_temperature
    soil_liquid = state.soil_snow_liquid_water
    soil_ice = state.soil_snow_ice_water
    old_ice_fraction = state.soil_snow_old_ice_fraction
    soil_snow_phase_flag = state.soil_snow_phase_flag
    lake_temperature = state.lake_temperature
    lake_ice_fraction = state.lake_ice_fraction
    snow_depth = state.snow_depth
    snow_water = state.snow_water

    for index in range(NUMBER_OF_SNOW_LEVELS):
        valid = index >= top_index
        total = _add(soil_liquid[index], soil_ice[index])
        safe_total = jnp.where(valid, total, jnp.float64(1.0))
        fraction = _divide(soil_ice[index], safe_total)
        old_ice_fraction = old_ice_fraction.at[index].set(
            jnp.where(valid, fraction, old_ice_fraction[index])
        )

    nir_absorbed = _sum_terms(
        inputs.incident_nir_direct,
        inputs.incident_nir_diffuse,
        -inputs.reflected_nir_direct,
        -inputs.reflected_nir_diffuse,
    )
    nir_absorbed = jnp.minimum(nir_absorbed, inputs.ground_absorbed_shortwave)
    beta = _divide(
        nir_absorbed,
        jnp.maximum(jnp.float64(1.0e-5), inputs.ground_absorbed_shortwave),
    )
    beta = _add(
        beta,
        _multiply(
            _subtract(jnp.float64(1.0), beta),
            jnp.float64(VISIBLE_SURFACE_ABSORPTION_FRACTION),
        ),
    )

    density_before = _strict_lake_density(lake_temperature, lake_ice_fraction)
    lake_diffusivity = jnp.zeros((NUMBER_OF_LAKE_LEVELS,), dtype=jnp.float64)
    lake_conductivity = jnp.zeros((NUMBER_OF_LAKE_LEVELS,), dtype=jnp.float64)
    warm_surface = (
        (inputs.ground_temperature > tfrz)
        & (lake_temperature[0] > tfrz)
        & (snow_layer_count == jnp.int32(0))
    )
    deep_lake = geometry.lake_depth >= jnp.float64(DEEP_MIXING_DEPTH)

    def lake_diffusivity_step(level, carry):
        diffusivity_values, conductivity_values = carry
        density_gradient = fortran_divide(
            fortran_add(density_before[level + 1], -density_before[level]),
            fortran_add(
                geometry.lake_node_depth[level + 1],
                -geometry.lake_node_depth[level],
            ),
        )
        n_squared = fortran_multiply(
            fortran_divide(jnp.float64(GRAVITY), density_before[level]),
            density_gradient,
        )
        mixing_length = fortran_multiply(
            jnp.float64(VON_KARMAN),
            geometry.lake_node_depth[level],
        )
        numerator = fortran_multiply(
            fortran_multiply(jnp.float64(40.0), n_squared),
            _power(mixing_length, 2.0),
        )
        wind_squared = _power(inputs.wind_mixing_velocity, 2.0)
        attenuation = strict_negative_exp(
            fortran_multiply(
                fortran_multiply(jnp.float64(-2.0), inputs.mixing_decay_coefficient),
                geometry.lake_node_depth[level],
            )
        )
        denominator = jnp.maximum(
            fortran_multiply(wind_squared, attenuation),
            jnp.float64(1.0e-10),
        )
        richardson = fortran_divide(
            fortran_add(
                _nonnegative_square_root(
                    fortran_add(
                        jnp.float64(1.0),
                        fortran_divide(numerator, denominator),
                    )
                ),
                jnp.float64(-1.0),
            ),
            jnp.float64(20.0),
        )
        fang_diffusivity = fortran_multiply(
            jnp.float64(1.039e-8),
            _power(
                jnp.maximum(
                    n_squared,
                    jnp.float64(MINIMUM_BRUNT_VAISALA_FREQUENCY_SQUARED),
                ),
                -0.43,
            ),
        )

        def warm_diffusion(_: None) -> tuple[jnp.ndarray, jnp.ndarray]:
            eddy_numerator = fortran_multiply(
                fortran_multiply(
                    fortran_multiply(
                        jnp.float64(VON_KARMAN),
                        inputs.wind_mixing_velocity,
                    ),
                    geometry.lake_node_depth[level],
                ),
                strict_negative_exp(
                    -fortran_multiply(
                        inputs.mixing_decay_coefficient,
                        geometry.lake_node_depth[level],
                    ),
                ),
            )
            eddy = fortran_divide(
                eddy_numerator,
                fortran_add(
                    jnp.float64(1.0),
                    fortran_multiply(
                        fortran_multiply(jnp.float64(37.0), richardson),
                        richardson,
                    ),
                ),
            )
            diffusivity = fortran_add(
                fortran_add(molecular_diffusivity, eddy),
                fang_diffusivity,
            )
            diffusivity = jnp.where(
                deep_lake,
                fortran_multiply(diffusivity, jnp.float64(DEEP_MIXING_FACTOR)),
                diffusivity,
            )
            return diffusivity, fortran_multiply(diffusivity, cwat)

        def cold_diffusion(_: None) -> tuple[jnp.ndarray, jnp.ndarray]:
            diffusivity = fortran_add(molecular_diffusivity, fang_diffusivity)
            diffusivity = jnp.where(
                deep_lake,
                fortran_multiply(diffusivity, jnp.float64(DEEP_MIXING_FACTOR)),
                diffusivity,
            )
            liquid_conductivity = fortran_multiply(diffusivity, cwat)
            conductivity = fortran_divide(
                fortran_multiply(liquid_conductivity, ice_conductivity_effective),
                fortran_add(
                    fortran_multiply(
                        fortran_add(jnp.float64(1.0), -lake_ice_fraction[level]),
                        ice_conductivity_effective,
                    ),
                    fortran_multiply(liquid_conductivity, lake_ice_fraction[level]),
                ),
            )
            return diffusivity, conductivity

        diffusivity, conductivity = lax.cond(
            warm_surface,
            warm_diffusion,
            cold_diffusion,
            operand=None,
        )
        return (
            diffusivity_values.at[level].set(diffusivity),
            conductivity_values.at[level].set(conductivity),
        )

    lake_diffusivity, lake_conductivity = lax.fori_loop(
        0,
        NUMBER_OF_LAKE_LEVELS - 1,
        lake_diffusivity_step,
        (lake_diffusivity, lake_conductivity),
    )

    last_level = NUMBER_OF_LAKE_LEVELS - 1
    lake_diffusivity = lake_diffusivity.at[last_level].set(lake_diffusivity[last_level - 1])

    def warm_last(_: None) -> jnp.ndarray:
        return lake_conductivity[last_level - 1]

    def cold_last(_: None) -> jnp.ndarray:
        liquid_conductivity = fortran_multiply(lake_diffusivity[last_level], cwat)
        return fortran_divide(
            fortran_multiply(liquid_conductivity, ice_conductivity_effective),
            fortran_add(
                fortran_multiply(
                    fortran_add(jnp.float64(1.0), -lake_ice_fraction[last_level]),
                    ice_conductivity_effective,
                ),
                fortran_multiply(liquid_conductivity, lake_ice_fraction[last_level]),
            ),
        )

    lake_conductivity = lake_conductivity.at[last_level].set(
        lax.cond(warm_surface, warm_last, cold_last, operand=None)
    )
    saved_top_eddy_conductivity = fortran_multiply(lake_diffusivity[0], cwat)

    extinction = jnp.where(
        geometry.light_extinction_coefficient > jnp.float64(0.0),
        geometry.light_extinction_coefficient,
        fortran_multiply(
            jnp.float64(1.1925),
            fortran_power(
                jnp.maximum(geometry.lake_depth, jnp.float64(1.0)),
                jnp.float64(-0.424),
            ),
        ),
    )
    no_snow = snow_layer_count == jnp.int32(0)
    one_minus_beta = fortran_add(jnp.float64(1.0), -beta)

    def accumulate_absorbed_shortwave(level, values):
        lake_values, soil_value = values
        half_thickness = fortran_multiply(jnp.float64(0.5), geometry.lake_layer_thickness[level])
        upper_depth = fortran_add(geometry.lake_node_depth[level], -half_thickness)
        lower_depth = fortran_add(geometry.lake_node_depth[level], half_thickness)
        relative_in = strict_negative_exp(
            -fortran_multiply(
                extinction,
                jnp.maximum(
                    fortran_add(upper_depth, -jnp.float64(SURFACE_ABSORPTION_DEPTH)),
                    jnp.float64(0.0),
                ),
            )
        )
        relative_out = strict_negative_exp(
            -fortran_multiply(
                extinction,
                jnp.maximum(
                    fortran_add(lower_depth, -jnp.float64(SURFACE_ABSORPTION_DEPTH)),
                    jnp.float64(0.0),
                ),
            )
        )
        warm_value = fortran_multiply(
            fortran_multiply(
                fortran_add(relative_in, -relative_out),
                inputs.ground_absorbed_shortwave,
            ),
            one_minus_beta,
        )
        frozen_first = fortran_multiply(inputs.ground_absorbed_shortwave, one_minus_beta)
        snow_first = inputs.layer_absorbed_shortwave[-1]
        value = jnp.where(
            warm_surface,
            warm_value,
            jnp.where(
                (level == 0) & no_snow,
                frozen_first,
                jnp.where(level == 0, snow_first, jnp.float64(0.0)),
            ),
        )
        lake_values = lake_values.at[level].set(value)
        bottom_soil_value = fortran_multiply(
            fortran_multiply(relative_out, inputs.ground_absorbed_shortwave),
            one_minus_beta,
        )
        soil_value = jnp.where(
            (level == NUMBER_OF_LAKE_LEVELS - 1) & warm_surface,
            bottom_soil_value,
            soil_value,
        )
        return lake_values, soil_value

    lake_solar, soil_solar = lax.fori_loop(
        0,
        NUMBER_OF_LAKE_LEVELS,
        accumulate_absorbed_shortwave,
        (
            jnp.zeros((NUMBER_OF_LAKE_LEVELS,), dtype=jnp.float64),
            jnp.float64(0.0),
        ),
    )

    lake_heat_capacity = fortran_multiply(
        geometry.lake_layer_thickness,
        fortran_add(
            fortran_multiply(
                cwat,
                fortran_add(jnp.float64(1.0), -lake_ice_fraction),
            ),
            fortran_multiply(cice_effective, lake_ice_fraction),
        ),
    )
    soil_conductivity, soil_heat_capacity, top_soil_conductivity = _soil_thermal_properties(
        snow_layer_count,
        geometry.soil_snow_layer_thickness,
        geometry.soil_snow_node_depth,
        geometry.soil_snow_interface_depth,
        geometry.saturated_water_content,
        geometry.saturated_thermal_conductivity,
        geometry.mineral_thermal_conductivity,
        geometry.dry_thermal_conductivity,
        geometry.solid_volumetric_heat_capacity,
        soil_liquid,
        soil_ice,
        soil_temperature,
    )
    initial_soil_heat_capacity = soil_heat_capacity
    initial_lake_heat_capacity = lake_heat_capacity

    def accumulate_initial_lake_energy(level, accumulated_energy):
        sensible = fortran_multiply(
            lake_heat_capacity[level],
            fortran_add(lake_temperature[level], -tfrz),
        )
        latent = fortran_multiply(
            fortran_multiply(fusion_per_volume, geometry.lake_layer_thickness[level]),
            fortran_add(jnp.float64(1.0), -lake_ice_fraction[level]),
        )
        return fortran_add(fortran_add(accumulated_energy, sensible), latent)

    initial_energy = lax.fori_loop(
        0,
        NUMBER_OF_LAKE_LEVELS,
        accumulate_initial_lake_energy,
        jnp.float64(0.0),
    )

    def accumulate_initial_soil_snow_energy(index, accumulated_energy):
        valid = index >= top_index
        sensible = fortran_multiply(
            soil_heat_capacity[index],
            fortran_add(soil_temperature[index], -tfrz),
        )
        latent = fortran_multiply(hfus, soil_liquid[index])
        contribution = fortran_add(fortran_add(accumulated_energy, sensible), latent)
        updated_energy = jnp.where(
            valid,
            contribution,
            accumulated_energy,
        )
        thin_snow = (
            (index == NUMBER_OF_SNOW_LEVELS)
            & (snow_water > jnp.float64(0.0))
            & (index == top_index)
        )
        return jnp.where(
            thin_snow,
            fortran_add(updated_energy, -fortran_multiply(snow_water, hfus)),
            updated_energy,
        )

    initial_energy = lax.fori_loop(
        0,
        NUMBER_OF_SOIL_SNOW_LEVELS,
        accumulate_initial_soil_snow_energy,
        initial_energy,
    )

    coupled_depth = jnp.zeros((NUMBER_OF_COUPLED_LEVELS,), dtype=jnp.float64)
    coupled_capacity = jnp.zeros_like(coupled_depth)
    coupled_solar = jnp.zeros_like(coupled_depth)
    coupled_temperature = jnp.zeros_like(coupled_depth)
    for index in range(NUMBER_OF_SNOW_LEVELS):
        valid = index >= top_index
        coupled_depth = coupled_depth.at[index].set(
            jnp.where(valid, geometry.soil_snow_node_depth[index], jnp.float64(0.0))
        )
        coupled_capacity = coupled_capacity.at[index].set(
            jnp.where(valid, soil_heat_capacity[index], jnp.float64(1.0))
        )
        snow_solar = jnp.where(
            index == top_index,
            jnp.float64(0.0),
            inputs.layer_absorbed_shortwave[index],
        )
        coupled_solar = coupled_solar.at[index].set(jnp.where(valid, snow_solar, jnp.float64(0.0)))
        coupled_temperature = coupled_temperature.at[index].set(
            jnp.where(valid, soil_temperature[index], jnp.float64(0.0))
        )
    for level in range(NUMBER_OF_LAKE_LEVELS):
        index = NUMBER_OF_SNOW_LEVELS + level
        coupled_depth = coupled_depth.at[index].set(geometry.lake_node_depth[level])
        coupled_capacity = coupled_capacity.at[index].set(lake_heat_capacity[level])
        coupled_solar = coupled_solar.at[index].set(lake_solar[level])
        coupled_temperature = coupled_temperature.at[index].set(lake_temperature[level])
    lake_bottom_interface = fortran_add(
        geometry.lake_node_depth[-1],
        fortran_divide(
            geometry.lake_layer_thickness[-1],
            jnp.float64(2.0),
        ),
    )
    for level in range(NUMBER_OF_GROUND_LEVELS):
        index = NUMBER_OF_SNOW_LEVELS + NUMBER_OF_LAKE_LEVELS + level
        soil_index = NUMBER_OF_SNOW_LEVELS + level
        coupled_depth = coupled_depth.at[index].set(
            fortran_add(
                lake_bottom_interface,
                geometry.soil_snow_node_depth[soil_index],
            )
        )
        coupled_capacity = coupled_capacity.at[index].set(soil_heat_capacity[soil_index])
        coupled_solar = coupled_solar.at[index].set(
            jnp.where(level == 0, soil_solar, jnp.float64(0.0))
        )
        coupled_temperature = coupled_temperature.at[index].set(soil_temperature[soil_index])

    interface_conductivity = jnp.zeros_like(coupled_depth)

    def coupled_interface_entry(index):
        fortran_level = index - (NUMBER_OF_SNOW_LEVELS - 1)

        def active_level(_):
            def snow_interface(_):
                return soil_conductivity[index]

            def bottom_snow_interface(_):
                distance = fortran_add(
                    coupled_depth[index + 1],
                    -coupled_depth[index],
                )
                numerator = fortran_multiply(
                    fortran_multiply(
                        lake_conductivity[0],
                        soil_conductivity[index],
                    ),
                    distance,
                )
                denominator = fortran_add(
                    fortran_multiply(
                        soil_conductivity[index],
                        geometry.lake_node_depth[0],
                    ),
                    fortran_multiply(
                        lake_conductivity[0],
                        -geometry.soil_snow_node_depth[index],
                    ),
                )
                return fortran_divide(numerator, denominator)

            def lake_interface(_):
                level = fortran_level - 1
                numerator = fortran_multiply(
                    fortran_multiply(
                        lake_conductivity[level],
                        lake_conductivity[level + 1],
                    ),
                    fortran_add(
                        geometry.lake_layer_thickness[level + 1],
                        geometry.lake_layer_thickness[level],
                    ),
                )
                denominator = fortran_add(
                    fortran_multiply(
                        lake_conductivity[level],
                        geometry.lake_layer_thickness[level + 1],
                    ),
                    fortran_multiply(
                        lake_conductivity[level + 1],
                        geometry.lake_layer_thickness[level],
                    ),
                )
                return fortran_divide(numerator, denominator)

            def lake_bottom_interface_value(_):
                distance = fortran_add(
                    coupled_depth[index + 1],
                    -coupled_depth[index],
                )
                numerator = fortran_multiply(
                    fortran_multiply(
                        top_soil_conductivity,
                        lake_conductivity[-1],
                    ),
                    distance,
                )
                denominator = fortran_add(
                    fortran_multiply(
                        top_soil_conductivity,
                        fortran_divide(
                            geometry.lake_layer_thickness[-1],
                            jnp.float64(2.0),
                        ),
                    ),
                    fortran_multiply(
                        lake_conductivity[-1],
                        geometry.soil_snow_node_depth[NUMBER_OF_SNOW_LEVELS],
                    ),
                )
                return fortran_divide(numerator, denominator)

            def soil_interface(_):
                soil_index = index - NUMBER_OF_LAKE_LEVELS
                return soil_conductivity[soil_index]

            return lax.cond(
                fortran_level < 0,
                snow_interface,
                lambda _: lax.cond(
                    fortran_level == 0,
                    bottom_snow_interface,
                    lambda _: lax.cond(
                        fortran_level < NUMBER_OF_LAKE_LEVELS,
                        lake_interface,
                        lambda _: lax.cond(
                            fortran_level == NUMBER_OF_LAKE_LEVELS,
                            lake_bottom_interface_value,
                            soil_interface,
                            None,
                        ),
                        None,
                    ),
                    None,
                ),
                None,
            )

        return lax.cond(
            index >= top_index,
            active_level,
            lambda _: jnp.float64(0.0),
            None,
        )

    interface_conductivity = _map_level_entries(
        coupled_interface_entry, interface_conductivity, 0, NUMBER_OF_COUPLED_LEVELS,
    )

    diffusion_flux = jnp.zeros_like(coupled_depth)
    timestep_over_capacity = jnp.zeros_like(coupled_depth)

    def diffusion_entry(index):
        def active_level(_):
            factor = fortran_divide(dtime, coupled_capacity[index])

            def interior_flux(_):
                temperature_difference = fortran_add(
                    coupled_temperature[index + 1],
                    -coupled_temperature[index],
                )
                depth_difference = fortran_add(
                    coupled_depth[index + 1],
                    -coupled_depth[index],
                )
                flux = fortran_divide(
                    fortran_multiply(
                        interface_conductivity[index],
                        temperature_difference,
                    ),
                    depth_difference,
                )
                return flux

            level_flux = lax.cond(
                index < NUMBER_OF_COUPLED_LEVELS - 1,
                interior_flux,
                lambda _: jnp.float64(0.0),
                None,
            )
            return level_flux, factor

        return lax.cond(
            index >= top_index,
            active_level,
            lambda _: (jnp.float64(0.0), jnp.float64(0.0)),
            None,
        )

    diffusion_flux, timestep_over_capacity = _map_level_entries(
        diffusion_entry, (diffusion_flux, timestep_over_capacity), 0, NUMBER_OF_COUPLED_LEVELS,
    )

    lower = jnp.zeros_like(coupled_depth)
    diagonal = jnp.ones_like(coupled_depth)
    upper = jnp.zeros_like(coupled_depth)
    rhs = coupled_temperature
    one_minus_cn = fortran_add(
        jnp.float64(1.0),
        -jnp.float64(CRANK_NICHOLSON_FACTOR),
    )
    cn = jnp.float64(CRANK_NICHOLSON_FACTOR)

    def coefficient(
        factor: jnp.ndarray,
        conductivity: jnp.ndarray,
        distance: jnp.ndarray,
    ) -> jnp.ndarray:
        return fortran_divide(
            fortran_multiply(
                fortran_multiply(one_minus_cn, factor),
                conductivity,
            ),
            distance,
        )

    def matrix_entry(index):
        def top_level(_):
            distance_below = fortran_add(
                coupled_depth[index + 1],
                -coupled_depth[index],
            )
            below = coefficient(
                timestep_over_capacity[index],
                interface_conductivity[index],
                distance_below,
            )
            diagonal_value = fortran_add(jnp.float64(1.0), below)
            incoming = fortran_add(
                fortran_add(
                    inputs.ground_net_heat_flux,
                    coupled_solar[index],
                ),
                fortran_multiply(cn, diffusion_flux[index]),
            )
            rhs_value = fortran_add(
                coupled_temperature[index],
                fortran_multiply(timestep_over_capacity[index], incoming),
            )
            return (
                jnp.float64(0.0), diagonal_value, -below, rhs_value,
            )

        def middle_level(_):
            distance_above = fortran_add(
                coupled_depth[index],
                -coupled_depth[index - 1],
            )
            distance_below = fortran_add(
                coupled_depth[index + 1],
                -coupled_depth[index],
            )
            above = coefficient(
                timestep_over_capacity[index],
                interface_conductivity[index - 1],
                distance_above,
            )
            below = coefficient(
                timestep_over_capacity[index],
                interface_conductivity[index],
                distance_below,
            )
            conductance_sum = fortran_add(
                fortran_divide(interface_conductivity[index], distance_below),
                fortran_divide(interface_conductivity[index - 1], distance_above),
            )
            diagonal_value = fortran_add(
                jnp.float64(1.0),
                fortran_multiply(
                    fortran_multiply(
                        one_minus_cn,
                        timestep_over_capacity[index],
                    ),
                    conductance_sum,
                ),
            )
            flux_difference = fortran_add(
                diffusion_flux[index],
                -diffusion_flux[index - 1],
            )
            rhs_value = fortran_add(
                fortran_add(
                    coupled_temperature[index],
                    fortran_multiply(
                        fortran_multiply(
                            cn,
                            timestep_over_capacity[index],
                        ),
                        flux_difference,
                    ),
                ),
                fortran_multiply(
                    timestep_over_capacity[index],
                    coupled_solar[index],
                ),
            )
            return (
                -above, diagonal_value, -below, rhs_value,
            )

        def bottom_level(_):
            distance_above = fortran_add(
                coupled_depth[index],
                -coupled_depth[index - 1],
            )
            above = coefficient(
                timestep_over_capacity[index],
                interface_conductivity[index - 1],
                distance_above,
            )
            diagonal_value = fortran_add(jnp.float64(1.0), above)
            outgoing = fortran_multiply(
                fortran_multiply(cn, timestep_over_capacity[index]),
                diffusion_flux[index - 1],
            )
            rhs_value = fortran_add(coupled_temperature[index], -outgoing)
            return (
                -above, diagonal_value, jnp.float64(0.0), rhs_value,
            )

        def active_level(_):
            return lax.cond(
                index == top_index,
                top_level,
                lambda _: lax.cond(
                    index == NUMBER_OF_COUPLED_LEVELS - 1,
                    bottom_level,
                    middle_level,
                    None,
                ),
                None,
            )

        return lax.cond(
            index >= top_index,
            active_level,
            lambda _: (jnp.float64(0.0), jnp.float64(1.0), jnp.float64(0.0),
                       coupled_temperature[index]),
            None,
        )

    lower, diagonal, upper, rhs = _map_level_entries(
        matrix_entry, (lower, diagonal, upper, rhs), 0, NUMBER_OF_COUPLED_LEVELS,
    )

    solution = _tridiagonal_solve(
        lower,
        diagonal,
        upper,
        rhs,
        coupled_temperature,
        top_index,
    )
    for index in range(NUMBER_OF_SNOW_LEVELS):
        soil_temperature = soil_temperature.at[index].set(
            jnp.where(index >= top_index, solution[index], soil_temperature[index])
        )
    lake_temperature = solution[
        NUMBER_OF_SNOW_LEVELS : NUMBER_OF_SNOW_LEVELS + NUMBER_OF_LAKE_LEVELS
    ]
    soil_temperature = soil_temperature.at[NUMBER_OF_SNOW_LEVELS:].set(
        solution[NUMBER_OF_SNOW_LEVELS + NUMBER_OF_LAKE_LEVELS :]
    )

    snow_melt_flux = jnp.float64(0.0)
    snow_freeze_flux = jnp.float64(0.0)
    snow_drain_flux = jnp.float64(0.0)
    snow_layer_melt = jnp.zeros((NUMBER_OF_SNOW_LEVELS,), dtype=jnp.float64)
    snow_layer_freeze = jnp.zeros((NUMBER_OF_SNOW_LEVELS,), dtype=jnp.float64)
    snow_phase_flag = jnp.zeros((NUMBER_OF_SNOW_LEVELS,), dtype=jnp.int32)
    soil_snow_phase_flag = soil_snow_phase_flag.at[:NUMBER_OF_SNOW_LEVELS].set(snow_phase_flag)
    latent_heat_absorbed = jnp.float64(0.0)

    thin_snow_melts = (
        (snow_layer_count == jnp.int32(0))
        & (snow_water > jnp.float64(0.0))
        & (lake_temperature[0] > tfrz)
    )

    def melt_thin_snow(values):
        (
            current_temperature,
            current_depth,
            current_water,
            current_latent,
            current_melt,
            current_drain,
        ) = values
        available = _multiply(
            _subtract(current_temperature[0], tfrz),
            lake_heat_capacity[0],
        )
        melt = jnp.minimum(current_water, _divide(available, hfus))
        remaining = jnp.maximum(
            _subtract(available, _multiply(melt, hfus)),
            jnp.float64(0.0),
        )
        updated_temperature = current_temperature.at[0].set(
            _add(tfrz, _divide(remaining, lake_heat_capacity[0]))
        )
        updated_depth = _multiply(
            current_depth,
            _subtract(jnp.float64(1.0), _divide(melt, current_water)),
        )
        updated_water = _subtract(current_water, melt)
        updated_latent = _add(current_latent, _multiply(melt, hfus))
        updated_flux = _add(current_melt, _divide(melt, dtime))
        updated_drain = _add(current_drain, _divide(melt, dtime))
        updated_water = jnp.where(
            updated_water < jnp.float64(PHASE_CHANGE_RESIDUAL),
            jnp.float64(0.0),
            updated_water,
        )
        updated_depth = jnp.where(
            updated_depth < jnp.float64(PHASE_CHANGE_RESIDUAL),
            jnp.float64(0.0),
            updated_depth,
        )
        return (
            updated_temperature,
            updated_depth,
            updated_water,
            updated_latent,
            updated_flux,
            updated_drain,
        )

    (
        lake_temperature,
        snow_depth,
        snow_water,
        latent_heat_absorbed,
        snow_melt_flux,
        snow_drain_flux,
    ) = lax.cond(
        thin_snow_melts,
        melt_thin_snow,
        lambda values: values,
        (
            lake_temperature,
            snow_depth,
            snow_water,
            latent_heat_absorbed,
            snow_melt_flux,
            snow_drain_flux,
        ),
    )

    for level in range(NUMBER_OF_LAKE_LEVELS):
        melting = (lake_temperature[level] > tfrz) & (lake_ice_fraction[level] > jnp.float64(0.0))
        freezing = (lake_temperature[level] < tfrz) & (lake_ice_fraction[level] < jnp.float64(1.0))

        def melt_or_freeze(values):
            temperature_value, fraction_value, capacity_value, latent_value = values
            available = _multiply(
                _subtract(temperature_value, tfrz),
                capacity_value,
            )
            melt = jnp.where(
                melting,
                jnp.minimum(
                    _multiply(
                        _multiply(fraction_value, denh2o),
                        geometry.lake_layer_thickness[level],
                    ),
                    _divide(available, hfus),
                ),
                jnp.maximum(
                    -_multiply(
                        _multiply(
                            _subtract(jnp.float64(1.0), fraction_value),
                            denh2o,
                        ),
                        geometry.lake_layer_thickness[level],
                    ),
                    _divide(available, hfus),
                ),
            )
            remaining = jnp.where(
                melting,
                jnp.maximum(
                    _subtract(available, _multiply(melt, hfus)),
                    jnp.float64(0.0),
                ),
                jnp.minimum(
                    _subtract(available, _multiply(melt, hfus)),
                    jnp.float64(0.0),
                ),
            )
            updated_fraction = _subtract(
                fraction_value,
                _divide(
                    melt,
                    _multiply(denh2o, geometry.lake_layer_thickness[level]),
                ),
            )
            updated_latent = _add(latent_value, _multiply(melt, hfus))
            updated_capacity = _add(
                capacity_value,
                _multiply(melt, _subtract(cpliq, cpice)),
            )
            updated_temperature = _add(
                tfrz,
                _divide(remaining, updated_capacity),
            )
            updated_fraction = jnp.where(
                updated_fraction > jnp.float64(1.0 - PHASE_CHANGE_RESIDUAL),
                jnp.float64(1.0),
                updated_fraction,
            )
            updated_fraction = jnp.where(
                updated_fraction < jnp.float64(PHASE_CHANGE_RESIDUAL),
                jnp.float64(0.0),
                updated_fraction,
            )
            return (
                updated_temperature,
                updated_fraction,
                updated_capacity,
                updated_latent,
            )

        (
            updated_temperature,
            updated_fraction,
            updated_capacity,
            latent_heat_absorbed,
        ) = lax.cond(
            melting | freezing,
            melt_or_freeze,
            lambda values: values,
            (
                lake_temperature[level],
                lake_ice_fraction[level],
                lake_heat_capacity[level],
                latent_heat_absorbed,
            ),
        )
        lake_temperature = lake_temperature.at[level].set(updated_temperature)
        lake_ice_fraction = lake_ice_fraction.at[level].set(updated_fraction)
        lake_heat_capacity = lake_heat_capacity.at[level].set(updated_capacity)

    soil_available_energy = fortran_multiply(
        fortran_add(soil_temperature, -tfrz), soil_heat_capacity
    )
    soil_energy_limited_melt = fortran_divide(soil_available_energy, hfus)

    for index in range(NUMBER_OF_SOIL_SNOW_LEVELS):
        valid = index >= top_index
        melting = valid & (soil_temperature[index] > tfrz) & (soil_ice[index] > jnp.float64(0.0))
        freezing = (
            valid & (soil_temperature[index] < tfrz) & (soil_liquid[index] > jnp.float64(0.0))
        )

        def soil_phase_change(values):
            (
                temperature_value,
                liquid_value,
                ice_value,
                capacity_value,
                latent_value,
                melt_flux_value,
                freeze_flux_value,
                layer_melt_value,
                layer_freeze_value,
                flag_value,
            ) = values
            available = soil_available_energy[index]
            melt = jnp.where(
                melting,
                jnp.minimum(ice_value, soil_energy_limited_melt[index]),
                jnp.maximum(-liquid_value, soil_energy_limited_melt[index]),
            )
            remaining = jnp.where(
                melting,
                jnp.maximum(
                    _subtract(available, _multiply(melt, hfus)),
                    jnp.float64(0.0),
                ),
                jnp.minimum(
                    _subtract(available, _multiply(melt, hfus)),
                    jnp.float64(0.0),
                ),
            )
            is_snow = index < NUMBER_OF_SNOW_LEVELS
            melt_rate = _divide(melt, dtime)
            freeze_rate = -melt_rate
            if is_snow:
                layer_melt_value = jnp.where(melting, melt_rate, layer_melt_value)
                layer_freeze_value = jnp.where(
                    freezing,
                    freeze_rate,
                    layer_freeze_value,
                )
                melt_flux_value = jnp.where(
                    melting,
                    _add(melt_flux_value, melt_rate),
                    melt_flux_value,
                )
                freeze_flux_value = jnp.where(
                    freezing,
                    _add(freeze_flux_value, freeze_rate),
                    freeze_flux_value,
                )
                flag_value = jnp.where(
                    melting,
                    jnp.int32(1),
                    jnp.where(freezing, jnp.int32(2), flag_value),
                )
            updated_ice = _subtract(ice_value, melt)
            updated_liquid = fortran_add(liquid_value, melt)
            updated_latent = _add(latent_value, _multiply(melt, hfus))
            updated_capacity = _add(
                capacity_value,
                _multiply(melt, _subtract(cpliq, cpice)),
            )
            updated_temperature = _add(
                tfrz,
                _divide(remaining, updated_capacity),
            )
            updated_ice = jnp.where(
                updated_ice < jnp.float64(PHASE_CHANGE_RESIDUAL),
                jnp.float64(0.0),
                updated_ice,
            )
            updated_liquid = jnp.where(
                updated_liquid < jnp.float64(PHASE_CHANGE_RESIDUAL),
                jnp.float64(0.0),
                updated_liquid,
            )
            return (
                updated_temperature,
                updated_liquid,
                updated_ice,
                updated_capacity,
                updated_latent,
                melt_flux_value,
                freeze_flux_value,
                layer_melt_value,
                layer_freeze_value,
                flag_value,
            )

        if index < NUMBER_OF_SNOW_LEVELS:
            layer_melt_input = snow_layer_melt[index]
            layer_freeze_input = snow_layer_freeze[index]
            flag_input = snow_phase_flag[index]
        else:
            layer_melt_input = jnp.float64(0.0)
            layer_freeze_input = jnp.float64(0.0)
            flag_input = jnp.int32(0)
        (
            temperature_value,
            liquid_value,
            ice_value,
            capacity_value,
            latent_heat_absorbed,
            snow_melt_flux,
            snow_freeze_flux,
            layer_melt_value,
            layer_freeze_value,
            flag_value,
        ) = lax.cond(
            melting | freezing,
            soil_phase_change,
            lambda values: values,
            (
                soil_temperature[index],
                soil_liquid[index],
                soil_ice[index],
                soil_heat_capacity[index],
                latent_heat_absorbed,
                snow_melt_flux,
                snow_freeze_flux,
                layer_melt_input,
                layer_freeze_input,
                flag_input,
            ),
        )
        soil_temperature = soil_temperature.at[index].set(temperature_value)
        soil_liquid = soil_liquid.at[index].set(liquid_value)
        soil_ice = soil_ice.at[index].set(ice_value)
        soil_heat_capacity = soil_heat_capacity.at[index].set(capacity_value)
        if index < NUMBER_OF_SNOW_LEVELS:
            snow_layer_melt = snow_layer_melt.at[index].set(layer_melt_value)
            snow_layer_freeze = snow_layer_freeze.at[index].set(layer_freeze_value)
            snow_phase_flag = snow_phase_flag.at[index].set(flag_value)
            soil_snow_phase_flag = soil_snow_phase_flag.at[index].set(flag_value)

    lake_temperature_after_phase_change = lake_temperature
    lake_ice_fraction_after_phase_change = lake_ice_fraction
    soil_snow_temperature_after_phase_change = soil_temperature
    soil_snow_liquid_after_phase_change = soil_liquid
    soil_snow_ice_after_phase_change = soil_ice

    density = _lake_density(lake_temperature, lake_ice_fraction)
    deepest_top_convection = jnp.int32(0)
    shallowest_bottom_convection = jnp.int32(NUMBER_OF_LAKE_LEVELS + 1)

    def mix_convective_segment(start_level, stop_level, values):
        temperature_values, fraction_values, density_values = values

        def accumulate_segment(level, sums):
            heat_sum, ice_sum, depth_sum = sums
            liquid_fraction = fortran_add(jnp.float64(1.0), -fraction_values[level])
            volumetric_capacity = fortran_add(
                fortran_multiply(liquid_fraction, cwat),
                fortran_multiply(fraction_values[level], cice_effective),
            )
            layer_heat = fortran_multiply(
                fortran_multiply(
                    geometry.lake_layer_thickness[level],
                    fortran_add(temperature_values[level], -tfrz),
                ),
                volumetric_capacity,
            )
            layer_ice = fortran_multiply(
                fraction_values[level], geometry.lake_layer_thickness[level]
            )
            return (
                fortran_add(heat_sum, layer_heat),
                fortran_add(ice_sum, layer_ice),
                fortran_add(depth_sum, geometry.lake_layer_thickness[level]),
            )

        def accumulate_active_segment(level, sums):
            active_level = (level >= start_level) & (level < stop_level)
            return lax.cond(
                active_level,
                lambda current: accumulate_segment(level, current),
                lambda current: current,
                sums,
            )

        heat_sum, ice_sum, depth_sum = lax.fori_loop(
            0,
            NUMBER_OF_LAKE_LEVELS,
            accumulate_active_segment,
            (jnp.float64(0.0), jnp.float64(0.0), jnp.float64(0.0)),
        )
        average_heat = fortran_divide(heat_sum, depth_sum)
        average_ice = fortran_divide(ice_sum, depth_sum)

        def positive_average(_: None):
            return (
                jnp.float64(0.0),
                fortran_divide(
                    average_heat,
                    fortran_multiply(fortran_add(jnp.float64(1.0), -average_ice), cwat),
                ),
            )

        def nonpositive_average(_: None):
            return lax.cond(
                average_heat < jnp.float64(0.0),
                lambda __: (
                    fortran_divide(
                        average_heat,
                        fortran_multiply(average_ice, cice_effective),
                    ),
                    jnp.float64(0.0),
                ),
                lambda __: (jnp.float64(0.0), jnp.float64(0.0)),
                operand=None,
            )

        frozen_temperature, liquid_temperature = lax.cond(
            average_heat > jnp.float64(0.0),
            positive_average,
            nonpositive_average,
            operand=None,
        )

        def redistribute_segment(level, carry):
            current_temperature, current_fraction, current_density, depth_seen = carry
            next_depth = fortran_add(depth_seen, geometry.lake_layer_thickness[level])
            fully_frozen = fortran_divide(next_depth, depth_sum) <= average_ice
            partly_frozen = fortran_divide(depth_seen, depth_sum) < average_ice
            partial_fraction = fortran_divide(
                fortran_add(fortran_multiply(average_ice, depth_sum), -depth_seen),
                geometry.lake_layer_thickness[level],
            )
            partial_liquid_fraction = fortran_add(jnp.float64(1.0), -partial_fraction)
            mixed_heat = fortran_add(
                fortran_multiply(
                    fortran_multiply(partial_fraction, frozen_temperature),
                    cice_effective,
                ),
                fortran_multiply(
                    fortran_multiply(partial_liquid_fraction, liquid_temperature),
                    cwat,
                ),
            )
            mixed_capacity = fortran_add(
                fortran_multiply(partial_fraction, cice_effective),
                fortran_multiply(partial_liquid_fraction, cwat),
            )
            mixed_temperature = fortran_add(fortran_divide(mixed_heat, mixed_capacity), tfrz)
            new_fraction = jnp.where(
                fully_frozen,
                jnp.float64(1.0),
                jnp.where(partly_frozen, partial_fraction, jnp.float64(0.0)),
            )
            new_temperature = jnp.where(
                fully_frozen,
                fortran_add(frozen_temperature, tfrz),
                jnp.where(
                    partly_frozen,
                    mixed_temperature,
                    fortran_add(liquid_temperature, tfrz),
                ),
            )
            current_fraction = current_fraction.at[level].set(new_fraction)
            current_temperature = current_temperature.at[level].set(new_temperature)
            current_density = current_density.at[level].set(
                _lake_density(new_temperature, new_fraction)
            )
            return (
                current_temperature,
                current_fraction,
                current_density,
                next_depth,
            )

        def redistribute_active_segment(level, carry):
            active_level = (level >= start_level) & (level < stop_level)
            return lax.cond(
                active_level,
                lambda current: redistribute_segment(level, current),
                lambda current: current,
                carry,
            )

        temperature_values, fraction_values, density_values, _ = lax.fori_loop(
            0,
            NUMBER_OF_LAKE_LEVELS,
            redistribute_active_segment,
            (
                temperature_values,
                fraction_values,
                density_values,
                jnp.float64(0.0),
            ),
        )
        return temperature_values, fraction_values, density_values

    def mix_from_top_interface(interface, values):
        temperature_values, fraction_values, density_values, _ = values
        unstable = (density_values[interface] > density_values[interface + 1]) | (
            (fraction_values[interface] < jnp.float64(1.0))
            & (fraction_values[interface + 1] > jnp.float64(0.0))
        )

        def mix_segment(current_values):
            current_temperature, current_fraction, current_density, _ = current_values
            temperature_values, fraction_values, density_values = mix_convective_segment(
                0,
                interface + 2,
                (current_temperature, current_fraction, current_density),
            )
            return (
                temperature_values,
                fraction_values,
                density_values,
                interface + jnp.int32(2),
            )

        return lax.cond(
            unstable,
            mix_segment,
            lambda current_values: current_values,
            values,
        )

    (
        lake_temperature,
        lake_ice_fraction,
        density,
        deepest_top_convection,
    ) = lax.fori_loop(
        0,
        NUMBER_OF_LAKE_LEVELS - 2,
        mix_from_top_interface,
        (
            lake_temperature,
            lake_ice_fraction,
            density,
            deepest_top_convection,
        ),
    )

    bottom_convection = (density[-2] > density[-1]) | (
        (lake_ice_fraction[-2] < jnp.float64(1.0)) & (lake_ice_fraction[-1] > jnp.float64(0.0))
    )

    def mix_from_bottom_interface(iteration, values):
        temperature_values, fraction_values, density_values, _ = values
        interface = jnp.int32(NUMBER_OF_LAKE_LEVELS - 2) - iteration
        unstable = bottom_convection & (
            (density_values[interface] > density_values[interface + 1])
            | (
                (fraction_values[interface] < jnp.float64(1.0))
                & (fraction_values[interface + 1] > jnp.float64(0.0))
            )
        )

        def mix_segment(current_values):
            current_temperature, current_fraction, current_density, _ = current_values
            temperature_values, fraction_values, density_values = mix_convective_segment(
                interface,
                NUMBER_OF_LAKE_LEVELS,
                (current_temperature, current_fraction, current_density),
            )
            return (
                temperature_values,
                fraction_values,
                density_values,
                interface + jnp.int32(1),
            )

        return lax.cond(
            unstable,
            mix_segment,
            lambda current_values: current_values,
            values,
        )

    (
        lake_temperature,
        lake_ice_fraction,
        density,
        shallowest_bottom_convection,
    ) = lax.fori_loop(
        0,
        NUMBER_OF_LAKE_LEVELS - 1,
        mix_from_bottom_interface,
        (
            lake_temperature,
            lake_ice_fraction,
            density,
            shallowest_bottom_convection,
        ),
    )

    lake_temperature_after_convection = lake_temperature
    lake_ice_fraction_after_convection = lake_ice_fraction

    lake_resistance = jnp.float64(0.0)
    for level in range(NUMBER_OF_LAKE_LEVELS):
        fortran_level = level + 1
        contributes = (fortran_level > deepest_top_convection) & (
            fortran_level < shallowest_bottom_convection
        )
        ground_index = NUMBER_OF_SNOW_LEVELS + level
        term = _divide(
            geometry.soil_snow_layer_thickness[ground_index],
            lake_diffusivity[level],
        )
        lake_resistance = jnp.where(
            contributes,
            _add(lake_resistance, term),
            lake_resistance,
        )
    methane_conductance = _divide(
        jnp.float64(1.0),
        _add(lake_resistance, inputs.lake_moisture_aerodynamic_resistance),
    )
    methane_conductance = jnp.where(
        lake_ice_fraction[0] > jnp.float64(0.1),
        jnp.float64(0.0),
        methane_conductance,
    )

    lake_heat_capacity = fortran_multiply(
        geometry.lake_layer_thickness,
        fortran_add(
            fortran_multiply(
                cwat,
                fortran_add(jnp.float64(1.0), -lake_ice_fraction),
            ),
            fortran_multiply(cice_effective, lake_ice_fraction),
        ),
    )
    _, soil_heat_capacity, _ = _soil_thermal_properties(
        snow_layer_count,
        geometry.soil_snow_layer_thickness,
        geometry.soil_snow_node_depth,
        geometry.soil_snow_interface_depth,
        geometry.saturated_water_content,
        geometry.saturated_thermal_conductivity,
        geometry.mineral_thermal_conductivity,
        geometry.dry_thermal_conductivity,
        geometry.solid_volumetric_heat_capacity,
        soil_liquid,
        soil_ice,
        soil_temperature,
    )

    def accumulate_final_lake_energy(level, carry):
        accumulated_energy, accumulated_flux = carry
        sensible = fortran_multiply(
            lake_heat_capacity[level],
            fortran_add(lake_temperature[level], -tfrz),
        )
        latent = fortran_multiply(
            fortran_multiply(fusion_per_volume, geometry.lake_layer_thickness[level]),
            fortran_add(jnp.float64(1.0), -lake_ice_fraction[level]),
        )
        return (
            fortran_add(fortran_add(accumulated_energy, sensible), latent),
            fortran_add(accumulated_flux, lake_solar[level]),
        )

    final_energy, applied_flux = lax.fori_loop(
        0,
        NUMBER_OF_LAKE_LEVELS,
        accumulate_final_lake_energy,
        (jnp.float64(0.0), inputs.ground_net_heat_flux),
    )

    def accumulate_final_soil_snow_energy(index, carry):
        accumulated_energy, accumulated_flux = carry
        valid = index >= top_index
        contribution = fortran_add(
            fortran_add(
                accumulated_energy,
                fortran_multiply(
                    soil_heat_capacity[index],
                    fortran_add(soil_temperature[index], -tfrz),
                ),
            ),
            fortran_multiply(hfus, soil_liquid[index]),
        )
        updated_energy = jnp.where(
            valid,
            contribution,
            accumulated_energy,
        )
        snow_flux = jnp.where(
            valid & (index < NUMBER_OF_SNOW_LEVELS),
            coupled_solar[index],
            jnp.float64(0.0),
        )
        updated_flux = fortran_add(accumulated_flux, snow_flux)
        thin_snow = (
            (index == NUMBER_OF_SNOW_LEVELS)
            & (snow_water > jnp.float64(0.0))
            & (index == top_index)
        )
        updated_energy = jnp.where(
            thin_snow,
            fortran_add(updated_energy, -fortran_multiply(snow_water, hfus)),
            updated_energy,
        )
        updated_flux = jnp.where(
            index == NUMBER_OF_SNOW_LEVELS,
            fortran_add(updated_flux, soil_solar),
            updated_flux,
        )
        return updated_energy, updated_flux

    final_energy, applied_flux = lax.fori_loop(
        0,
        NUMBER_OF_SOIL_SNOW_LEVELS,
        accumulate_final_soil_snow_energy,
        (final_energy, applied_flux),
    )

    raw_error = fortran_add(
        fortran_divide(
            fortran_add(final_energy, -initial_energy),
            dtime,
        ),
        -applied_flux,
    )
    incorporate_error = jnp.abs(raw_error) < jnp.float64(0.10)
    soil_ground_heat = jnp.where(
        incorporate_error,
        fortran_add(inputs.soil_ground_heat_flux, raw_error),
        inputs.soil_ground_heat_flux,
    )
    sensible_ground = jnp.where(
        incorporate_error,
        fortran_add(inputs.sensible_ground_heat_flux, -raw_error),
        inputs.sensible_ground_heat_flux,
    )
    sensible_total = jnp.where(
        incorporate_error,
        fortran_add(inputs.sensible_total_heat_flux, -raw_error),
        inputs.sensible_total_heat_flux,
    )
    ground_net = jnp.where(
        incorporate_error,
        fortran_add(inputs.ground_net_heat_flux, raw_error),
        inputs.ground_net_heat_flux,
    )
    conservation_error = jnp.where(
        incorporate_error,
        jnp.float64(0.0),
        raw_error,
    )

    def accumulate_lake_ice_thickness(level, accumulated_thickness):
        nominal_ice_thickness = fortran_multiply(
            lake_ice_fraction[level],
            geometry.lake_layer_thickness[level],
        )
        physical_ice_thickness = fortran_divide(
            fortran_multiply(nominal_ice_thickness, denh2o),
            denice,
        )
        return fortran_add(accumulated_thickness, physical_ice_thickness)

    lake_ice_thickness = lax.fori_loop(
        0,
        NUMBER_OF_LAKE_LEVELS,
        accumulate_lake_ice_thickness,
        jnp.float64(0.0),
    )

    updated_state = LakeColumnTemperatureState(
        snow_layer_count=snow_layer_count,
        snow_depth=snow_depth,
        snow_water=snow_water,
        soil_snow_temperature=soil_temperature,
        soil_snow_liquid_water=soil_liquid,
        soil_snow_ice_water=soil_ice,
        soil_snow_old_ice_fraction=old_ice_fraction,
        soil_snow_phase_flag=soil_snow_phase_flag,
        lake_temperature=lake_temperature,
        lake_ice_fraction=lake_ice_fraction,
        saved_top_eddy_conductivity=saved_top_eddy_conductivity,
        surface_absorption_fraction=beta,
        lake_surface_ice_fraction=lake_ice_fraction[0],
        lake_ice_thickness=lake_ice_thickness,
        lake_resistance=lake_resistance,
        ground_methane_conductance=methane_conductance,
    )
    return LakeTemperatureResult(
        state=updated_state,
        energy=LakeTemperatureEnergyFluxes(
            soil_ground_heat=soil_ground_heat,
            sensible_ground=sensible_ground,
            sensible_total=sensible_total,
            ground_net=ground_net,
            snow_melt_heat=_multiply(snow_melt_flux, hfus),
            conservation_error=conservation_error,
        ),
        water=LakeTemperatureWaterFluxes(
            snow_melt=snow_melt_flux,
            snow_freeze=snow_freeze_flux,
            snow_drain=snow_drain_flux,
            snow_layer_melt=snow_layer_melt,
            snow_layer_freeze=snow_layer_freeze,
            snow_layer_phase_flag=snow_phase_flag,
        ),
        diagnostics=LakeTemperatureDiagnostics(
            lake_density_before_diffusion=density_before,
            lake_diffusivity=lake_diffusivity,
            lake_thermal_conductivity=lake_conductivity,
            lake_heat_capacity_before_phase_change=initial_lake_heat_capacity,
            soil_snow_heat_capacity_before_phase_change=initial_soil_heat_capacity,
            coupled_interface_conductivity=interface_conductivity,
            coupled_timestep_over_capacity=timestep_over_capacity,
            coupled_diffusion_flux=diffusion_flux,
            coupled_lower_diagonal=lower,
            coupled_diagonal=diagonal,
            coupled_upper_diagonal=upper,
            coupled_right_hand_side=rhs,
            coupled_temperature_after_diffusion=solution,
            lake_temperature_after_phase_change=lake_temperature_after_phase_change,
            lake_ice_fraction_after_phase_change=lake_ice_fraction_after_phase_change,
            soil_snow_temperature_after_phase_change=(soil_snow_temperature_after_phase_change),
            soil_snow_liquid_after_phase_change=soil_snow_liquid_after_phase_change,
            soil_snow_ice_after_phase_change=soil_snow_ice_after_phase_change,
            lake_temperature_after_convection=lake_temperature_after_convection,
            lake_ice_fraction_after_convection=lake_ice_fraction_after_convection,
            lake_absorbed_shortwave=lake_solar,
            soil_absorbed_shortwave=soil_solar,
            latent_heat_absorbed=latent_heat_absorbed,
            initial_column_energy=initial_energy,
            final_column_energy=final_energy,
            applied_column_energy_flux=applied_flux,
            raw_conservation_error=raw_error,
            deepest_top_convection_level=deepest_top_convection,
            shallowest_bottom_convection_level=shallowest_bottom_convection,
        ),
    )


def lake_temperature(
    inputs: LakeTemperatureInputs,
    geometry: LakeTemperatureGeometry,
    state: LakeColumnTemperatureState,
    *,
    time_step_seconds: float = 1800.0,
) -> LakeTemperatureResult:

    _validate_shapes(inputs, geometry, state)
    if time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")

    from clm_jax.physics.solver_scalarization import _native_tile_trace

    if jax.default_backend() == "gpu" and not _native_tile_trace.get():
        from clm_jax.physics.lake_temperature_gpu import lake_temperature_columns

        return lake_temperature_columns(
            inputs, geometry, state, time_step_seconds=time_step_seconds)
    return _lake_temperature_reference(
        inputs, geometry, state, time_step_seconds=time_step_seconds)


def _lake_temperature_reference(
    inputs, geometry, state, *, time_step_seconds=1800.,
):

    def one_column(column_inputs, column_geometry, column_state):
        return lax.cond(
            jnp.asarray(column_inputs.active_lake, dtype=jnp.bool_),
            lambda operands: _active_column_step(
                operands[0],
                operands[1],
                operands[2],
                time_step_seconds,
            ),
            lambda operands: _inactive_column_result(operands[0], operands[2]),
            (column_inputs, column_geometry, column_state),
        )

    return jax.vmap(one_column)(inputs, geometry, state)
