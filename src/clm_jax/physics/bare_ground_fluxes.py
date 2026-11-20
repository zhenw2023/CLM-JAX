
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import default_backend, jit, lax, tree_util

from clm_jax.constants import (
    GRAVITY,
    SPECIFIC_HEAT_DRY_AIR,
    UNIVERSAL_GAS_CONSTANT,
    VON_KARMAN,
)
from clm_jax.io.restart import LANDUNIT_CROP, LANDUNIT_SOIL, SubgridTopology
from clm_jax.physics.canopy_temperature import (
    NUMBER_OF_GROUND_LEVELS,
    NUMBER_OF_SNOW_LEVELS,
    NUMBER_OF_TOTAL_LEVELS,
)
from clm_jax.physics.filters import exposed_vegetation_masks
from clm_jax.physics.friction_velocity import (
    FrictionVelocityState,
    friction_velocity,
    monin_obukhov_initialize,
)
from clm_jax.physics.human_index import FastHumanStressIndices, fast_human_stress_indices
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
    fortran_ordered_sum,
    fortran_product,
    strict_exp,
)
from clm_jax.physics.pointwise import scalar_map
from clm_jax.physics.qsat import qsat

NUMBER_OF_STABILITY_ITERATIONS = 3
MOLECULAR_VISCOSITY_AIR = 1.5e-5


class BareGroundFluxInputs(NamedTuple):
    vegetation_exposed_patch: jnp.ndarray
    eastward_wind_gridcell: jnp.ndarray
    northward_wind_gridcell: jnp.ndarray
    potential_temperature_column: jnp.ndarray
    temperature_column: jnp.ndarray
    pressure_column: jnp.ndarray
    density_column: jnp.ndarray
    specific_humidity_column: jnp.ndarray
    snow_layer_count_column: jnp.ndarray
    ground_layer_thickness_column: jnp.ndarray
    saturated_water_content_column: jnp.ndarray
    soil_snow_temperature_column_level: jnp.ndarray
    soil_liquid_water_column_level: jnp.ndarray
    soil_ice_water_column_level: jnp.ndarray
    surface_water_temperature_column: jnp.ndarray
    ground_temperature_column: jnp.ndarray
    virtual_potential_temperature_column: jnp.ndarray
    convective_velocity_coefficient_column: jnp.ndarray
    convective_boundary_height_column: jnp.ndarray
    soil_evaporation_resistance_column: jnp.ndarray
    snow_surface_specific_humidity_column: jnp.ndarray
    soil_surface_specific_humidity_column: jnp.ndarray
    surface_water_specific_humidity_column: jnp.ndarray
    ground_specific_humidity_column: jnp.ndarray
    ground_specific_humidity_temperature_derivative_column: jnp.ndarray
    latent_heat_column: jnp.ndarray
    ground_momentum_roughness_column: jnp.ndarray
    ground_sensible_roughness_column: jnp.ndarray
    ground_latent_roughness_column: jnp.ndarray
    forcing_height_wind_patch: jnp.ndarray
    forcing_height_temperature_patch: jnp.ndarray
    forcing_height_humidity_patch: jnp.ndarray
    reference_temperature_patch: jnp.ndarray


class BareGroundEnergyState(NamedTuple):
    sensible_heat_snow_patch: jnp.ndarray
    sensible_heat_soil_patch: jnp.ndarray
    sensible_heat_surface_water_patch: jnp.ndarray
    sensible_heat_ground_patch: jnp.ndarray
    sensible_heat_total_patch: jnp.ndarray
    eastward_wind_stress_patch: jnp.ndarray
    northward_wind_stress_patch: jnp.ndarray
    downward_longwave_below_canopy_patch: jnp.ndarray
    upward_longwave_above_canopy_patch: jnp.ndarray
    ground_sensible_derivative_patch: jnp.ndarray
    ground_latent_derivative_patch: jnp.ndarray
    ground_energy_derivative_patch: jnp.ndarray
    transpiration_wetness_patch: jnp.ndarray
    root_resistance_patch_level: jnp.ndarray


class BareGroundWaterState(NamedTuple):
    evaporation_snow_patch: jnp.ndarray
    evaporation_soil_patch: jnp.ndarray
    evaporation_surface_water_patch: jnp.ndarray
    ground_evaporation_patch: jnp.ndarray
    total_evaporation_patch: jnp.ndarray
    reference_specific_humidity_patch: jnp.ndarray
    reference_relative_humidity_patch: jnp.ndarray
    rural_reference_relative_humidity_patch: jnp.ndarray


class BareGroundTemperatureState(NamedTuple):
    reference_temperature_patch: jnp.ndarray
    rural_reference_temperature_patch: jnp.ndarray
    vegetation_temperature_patch: jnp.ndarray


class BareGroundPhotosynthesisState(NamedTuple):
    sunlit_stomatal_resistance_patch: jnp.ndarray
    shaded_stomatal_resistance_patch: jnp.ndarray


class BareGroundHumanIndexState(NamedTuple):
    temperature_celsius_patch: jnp.ndarray
    vapor_pressure_patch: jnp.ndarray
    wet_bulb_stull_patch: jnp.ndarray
    rural_wet_bulb_stull_patch: jnp.ndarray
    heat_index_patch: jnp.ndarray
    rural_heat_index_patch: jnp.ndarray
    apparent_temperature_patch: jnp.ndarray
    rural_apparent_temperature_patch: jnp.ndarray
    simplified_wet_bulb_globe_temperature_patch: jnp.ndarray
    rural_simplified_wet_bulb_globe_temperature_patch: jnp.ndarray
    humidex_patch: jnp.ndarray
    rural_humidex_patch: jnp.ndarray
    discomfort_index_stull_patch: jnp.ndarray
    rural_discomfort_index_stull_patch: jnp.ndarray


class BareGroundFluxState(NamedTuple):
    energy: BareGroundEnergyState
    water: BareGroundWaterState
    temperature: BareGroundTemperatureState
    photosynthesis: BareGroundPhotosynthesisState
    human_index: BareGroundHumanIndexState
    friction_velocity: FrictionVelocityState
    effective_root_fraction_patch_level: jnp.ndarray
    ground_methane_conductance_patch: jnp.ndarray
    aerodynamic_resistance_patch: jnp.ndarray


def _add(*terms: jnp.ndarray) -> jnp.ndarray:
    values = jnp.broadcast_arrays(*(jnp.asarray(term, dtype=jnp.float64) for term in terms))
    return lax.optimization_barrier(fortran_ordered_sum(jnp.stack(values, axis=0)))


def _multiply(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    left, right = jnp.broadcast_arrays(
        jnp.asarray(left, dtype=jnp.float64),
        jnp.asarray(right, dtype=jnp.float64),
    )
    return lax.optimization_barrier(fortran_product(left, right))


def _multiply_boundary(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return fortran_multiply(left, right)


def _add_boundary(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return fortran_add(left, right)


def _divide(numerator: jnp.ndarray, denominator: jnp.ndarray) -> jnp.ndarray:
    return fortran_divide(numerator, denominator)


def _roughness_reynolds(friction: jnp.ndarray, roughness: jnp.ndarray) -> jnp.ndarray:
    return _divide(_multiply(friction, roughness), jnp.float64(MOLECULAR_VISCOSITY_AIR))


def _sensible_heat_conductance(
    density: jnp.ndarray,
    temperature_profile: jnp.ndarray,
    friction: jnp.ndarray,
) -> jnp.ndarray:
    resistance = _divide(
        jnp.float64(1.0), _multiply(temperature_profile, friction),
    )
    return _divide(
        _multiply(density, jnp.float64(SPECIFIC_HEAT_DRY_AIR)), resistance,
    )


@jit
def _reference_humidity_at_two_meters(
    forcing_humidity: jnp.ndarray,
    humidity_difference: jnp.ndarray,
    humidity_profile: jnp.ndarray,
    humidity_profile_2m: jnp.ndarray,
) -> jnp.ndarray:
    def calculate_point(operands):
        forcing, difference, profile, profile_2m = operands
        profile_difference = _add(
            _divide(jnp.float64(1.0), profile_2m),
            -_divide(jnp.float64(1.0), profile),
        )
        return _add_boundary(
            forcing,
            _multiply_boundary(
                _multiply_boundary(profile, difference),
                profile_difference,
            ),
        )

    return scalar_map(
        calculate_point,
        (
            forcing_humidity,
            humidity_difference,
            humidity_profile,
            humidity_profile_2m,
        ),
    )


def _scalar_exp(value: jnp.ndarray) -> jnp.ndarray:
    return strict_exp(value)


def _scalar_power(value: jnp.ndarray, exponent: float) -> jnp.ndarray:
    if default_backend() == "gpu":
        from clm_jax.physics.native_power import native_power
        return native_power(value, jnp.float64(exponent))
    shape = value.shape
    exponent_array = jnp.full((value.size,), exponent, dtype=jnp.float64)
    flattened = jnp.reshape(value, (-1,))
    powered = scalar_map(
        lambda operands: operands[0] ** operands[1],
        (flattened, exponent_array),
    )
    return jnp.reshape(powered, shape)


def _validate_shapes(
    topology: SubgridTopology,
    inputs: BareGroundFluxInputs,
    state: BareGroundFluxState,
) -> None:
    number_of_gridcells = topology.grid.longitude_degrees.shape[0]
    number_of_columns = topology.column.gridcell.shape[0]
    number_of_patches = topology.patch.gridcell.shape[0]
    if inputs.vegetation_exposed_patch.shape != (number_of_patches,):
        raise ValueError("vegetation_exposed_patch must have shape (pft,)")
    for name in ("eastward_wind_gridcell", "northward_wind_gridcell"):
        if getattr(inputs, name).shape != (number_of_gridcells,):
            raise ValueError(f"{name} must have shape (gridcell,)")
    column_level_names = (
        "soil_snow_temperature_column_level",
        "soil_liquid_water_column_level",
        "soil_ice_water_column_level",
    )
    ground_level_names = (
        "ground_layer_thickness_column",
        "saturated_water_content_column",
    )
    patch_names = (
        "forcing_height_wind_patch",
        "forcing_height_temperature_patch",
        "forcing_height_humidity_patch",
        "reference_temperature_patch",
    )
    excluded = {"vegetation_exposed_patch", *column_level_names, *ground_level_names, *patch_names}
    for name in BareGroundFluxInputs._fields:
        if name in excluded or name.endswith("_gridcell"):
            continue
        if getattr(inputs, name).shape != (number_of_columns,):
            raise ValueError(f"{name} must have shape (column,)")
    for name in column_level_names:
        if getattr(inputs, name).shape != (number_of_columns, NUMBER_OF_TOTAL_LEVELS):
            raise ValueError(f"{name} must have shape (column, {NUMBER_OF_TOTAL_LEVELS})")
    for name in ground_level_names:
        if getattr(inputs, name).shape != (number_of_columns, NUMBER_OF_GROUND_LEVELS):
            raise ValueError(f"{name} must have shape (column, {NUMBER_OF_GROUND_LEVELS})")
    for name in patch_names:
        if getattr(inputs, name).shape != (number_of_patches,):
            raise ValueError(f"{name} must have shape (pft,)")
    for group in (
        state.energy,
        state.water,
        state.temperature,
        state.photosynthesis,
        state.human_index,
        state.friction_velocity,
    ):
        for name, value in zip(group._fields, group, strict=True):
            expected = (
                (number_of_patches, NUMBER_OF_GROUND_LEVELS)
                if name == "root_resistance_patch_level"
                else (number_of_patches,)
            )
            if value.shape != expected:
                raise ValueError(f"state field {name} must have shape {expected}")
    if state.effective_root_fraction_patch_level.shape != (
        number_of_patches,
        NUMBER_OF_GROUND_LEVELS,
    ):
        raise ValueError("effective_root_fraction_patch_level must have shape (pft, 25)")
    for name in ("ground_methane_conductance_patch", "aerodynamic_resistance_patch"):
        if getattr(state, name).shape != (number_of_patches,):
            raise ValueError(f"{name} must have shape (pft,)")


def _update(mask: jnp.ndarray, value: jnp.ndarray, previous: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(mask, value, jnp.asarray(previous, dtype=jnp.float64))


def _human_state_update(
    active: jnp.ndarray,
    rural: jnp.ndarray,
    calculated: FastHumanStressIndices,
    previous: BareGroundHumanIndexState,
) -> BareGroundHumanIndexState:
    return BareGroundHumanIndexState(
        temperature_celsius_patch=_update(
            active, calculated.temperature_celsius, previous.temperature_celsius_patch
        ),
        vapor_pressure_patch=_update(
            active, calculated.vapor_pressure, previous.vapor_pressure_patch
        ),
        wet_bulb_stull_patch=_update(
            active, calculated.wet_bulb_stull, previous.wet_bulb_stull_patch
        ),
        rural_wet_bulb_stull_patch=_update(
            rural, calculated.wet_bulb_stull, previous.rural_wet_bulb_stull_patch
        ),
        heat_index_patch=_update(active, calculated.heat_index, previous.heat_index_patch),
        rural_heat_index_patch=_update(
            rural, calculated.heat_index, previous.rural_heat_index_patch
        ),
        apparent_temperature_patch=_update(
            active, calculated.apparent_temperature, previous.apparent_temperature_patch
        ),
        rural_apparent_temperature_patch=_update(
            rural, calculated.apparent_temperature, previous.rural_apparent_temperature_patch
        ),
        simplified_wet_bulb_globe_temperature_patch=_update(
            active,
            calculated.simplified_wet_bulb_globe_temperature,
            previous.simplified_wet_bulb_globe_temperature_patch,
        ),
        rural_simplified_wet_bulb_globe_temperature_patch=_update(
            rural,
            calculated.simplified_wet_bulb_globe_temperature,
            previous.rural_simplified_wet_bulb_globe_temperature_patch,
        ),
        humidex_patch=_update(active, calculated.humidex, previous.humidex_patch),
        rural_humidex_patch=_update(rural, calculated.humidex, previous.rural_humidex_patch),
        discomfort_index_stull_patch=_update(
            active,
            calculated.discomfort_index_stull,
            previous.discomfort_index_stull_patch,
        ),
        rural_discomfort_index_stull_patch=_update(
            rural,
            calculated.discomfort_index_stull,
            previous.rural_discomfort_index_stull_patch,
        ),
    )


def bare_ground_fluxes(
    topology: SubgridTopology,
    inputs: BareGroundFluxInputs,
    state: BareGroundFluxState,
    maximum_stable_zeta: float | jnp.ndarray = 0.5,
) -> BareGroundFluxState:

    _validate_shapes(topology, inputs, state)
    inputs = BareGroundFluxInputs(
        *(
            jnp.asarray(value, dtype=jnp.int32)
            if name in ("vegetation_exposed_patch", "snow_layer_count_column")
            else jnp.asarray(value, dtype=jnp.float64)
            for name, value in zip(BareGroundFluxInputs._fields, inputs, strict=True)
        )
    )
    patch_column = topology.patch.column
    patch_gridcell = topology.patch.gridcell
    masks = exposed_vegetation_masks(
        topology.patch.landunit_type,
        topology.patch.active,
        inputs.vegetation_exposed_patch,
    )
    active = masks.not_exposed
    rural = active & (
        (topology.patch.landunit_type == jnp.int32(LANDUNIT_SOIL))
        | (topology.patch.landunit_type == jnp.int32(LANDUNIT_CROP))
    )

    def column(values: jnp.ndarray) -> jnp.ndarray:
        return values[patch_column]

    forcing_u = inputs.eastward_wind_gridcell[patch_gridcell]
    forcing_v = inputs.northward_wind_gridcell[patch_gridcell]
    forcing_theta = column(inputs.potential_temperature_column)
    forcing_temperature = column(inputs.temperature_column)
    forcing_pressure = column(inputs.pressure_column)
    forcing_density = column(inputs.density_column)
    forcing_humidity = column(inputs.specific_humidity_column)
    ground_temperature = column(inputs.ground_temperature_column)
    ground_humidity = column(inputs.ground_specific_humidity_column)
    virtual_temperature = column(inputs.virtual_potential_temperature_column)
    reference_temperature = inputs.reference_temperature_patch

    safe_active = active.astype(jnp.float64)
    forcing_u_safe = jnp.where(active, forcing_u, 1.0)
    forcing_v_safe = jnp.where(active, forcing_v, 0.0)
    forcing_theta_safe = jnp.where(active, forcing_theta, 280.0)
    forcing_temperature_safe = jnp.where(active, forcing_temperature, 280.0)
    forcing_pressure_safe = jnp.where(active, forcing_pressure, 100000.0)
    forcing_density_safe = jnp.where(active, forcing_density, 1.2)
    forcing_humidity_safe = jnp.where(active, forcing_humidity, 0.005)
    ground_temperature_safe = jnp.where(active, ground_temperature, 280.0)
    ground_humidity_safe = jnp.where(active, ground_humidity, 0.005)
    virtual_temperature_safe = jnp.where(active, virtual_temperature, 280.0)
    reference_temperature_safe = jnp.where(active, reference_temperature, 280.0)

    wind = jnp.maximum(
        jnp.float64(1.0),
        jnp.sqrt(
            _add(
                _multiply(forcing_u_safe, forcing_u_safe),
                _multiply(forcing_v_safe, forcing_v_safe),
            )
        ),
    )
    temperature_difference = _add(reference_temperature_safe, -ground_temperature_safe)
    humidity_difference = _add(forcing_humidity_safe, -ground_humidity_safe)
    virtual_temperature_difference = _add(
        _multiply(
            temperature_difference,
            _add(jnp.float64(1.0), _multiply(jnp.float64(0.61), forcing_humidity_safe)),
        ),
        _multiply(
            _multiply(jnp.float64(0.61), forcing_theta_safe),
            humidity_difference,
        ),
    )
    height = jnp.where(active, inputs.forcing_height_wind_patch, 30.0)
    momentum_roughness = jnp.where(
        active,
        column(inputs.ground_momentum_roughness_column),
        0.01,
    )
    initial_sensible_roughness = jnp.where(
        active,
        column(inputs.ground_sensible_roughness_column),
        0.01,
    )
    initial_latent_roughness = jnp.where(
        active,
        column(inputs.ground_latent_roughness_column),
        0.01,
    )
    initialized = monin_obukhov_initialize(
        wind,
        virtual_temperature_safe,
        virtual_temperature_difference,
        height,
        momentum_roughness,
        maximum_stable_zeta,
    )

    initial_iteration_state = (
        initialized.stability_wind_speed,
        initialized.monin_obukhov_length,
        initial_sensible_roughness,
        initial_latent_roughness,
        state.friction_velocity,
        jnp.zeros_like(wind),
        jnp.zeros_like(wind),
        jnp.zeros_like(wind),
        jnp.zeros_like(wind),
        jnp.zeros_like(wind),
    )

    def stability_iteration(iteration_state, zero_based_iteration):
        (
            stability_wind,
            monin_obukhov_length,
            sensible_roughness,
            latent_roughness,
            friction_state,
            _friction,
            _temperature_profile,
            _humidity_profile,
            _temperature_profile_2m,
            _humidity_profile_2m,
        ) = iteration_state
        friction_result = friction_velocity(
            active,
            jnp.zeros_like(wind),
            momentum_roughness,
            sensible_roughness,
            latent_roughness,
            monin_obukhov_length,
            zero_based_iteration + jnp.int32(1),
            wind,
            stability_wind,
            inputs.forcing_height_wind_patch,
            inputs.forcing_height_temperature_patch,
            inputs.forcing_height_humidity_patch,
            friction_state,
        )
        friction = friction_result.friction_velocity_patch
        temperature_profile = friction_result.temperature_profile_patch
        humidity_profile = friction_result.humidity_profile_patch
        friction_calculation = jnp.where(active, friction, jnp.float64(0.1))
        temperature_profile_calculation = jnp.where(
            active, temperature_profile, jnp.float64(0.1)
        )
        humidity_profile_calculation = jnp.where(
            active, humidity_profile, jnp.float64(0.1)
        )
        temperature_scale = _multiply(
            temperature_profile_calculation, temperature_difference
        )
        humidity_scale = _multiply(humidity_profile_calculation, humidity_difference)
        roughness_reynolds = _roughness_reynolds(friction_calculation, momentum_roughness)
        roughness_exponent = _multiply(
            jnp.float64(0.13),
            _scalar_power(jnp.maximum(roughness_reynolds, 0.0), 0.45),
        )
        roughness_exponential = _scalar_exp(roughness_exponent)
        new_sensible_roughness = momentum_roughness / roughness_exponential
        new_sensible_roughness = jnp.where(
            active, new_sensible_roughness, sensible_roughness
        )
        new_latent_roughness = jnp.where(active, new_sensible_roughness, latent_roughness)
        virtual_humidity_factor = _add(
            jnp.float64(1.0),
            _multiply(jnp.float64(0.61), forcing_humidity_safe),
        )
        virtual_temperature_term = _multiply(
            temperature_scale,
            virtual_humidity_factor,
        )
        virtual_humidity_term = _multiply(
            _multiply(jnp.float64(0.61), forcing_theta_safe),
            humidity_scale,
        )
        virtual_temperature_scale = _add_boundary(
            virtual_temperature_term,
            virtual_humidity_term,
        )
        zeta_numerator = _multiply(
            _multiply(_multiply(height, jnp.float64(VON_KARMAN)), jnp.float64(GRAVITY)),
            virtual_temperature_scale,
        )
        zeta_denominator = _multiply(
            _multiply(friction_calculation, friction_calculation),
            virtual_temperature_safe,
        )
        zeta = zeta_numerator / zeta_denominator
        stable = zeta >= jnp.float64(0.0)
        bounded_stable_zeta = jnp.minimum(
            jnp.asarray(maximum_stable_zeta, dtype=jnp.float64),
            jnp.maximum(zeta, jnp.float64(0.01)),
        )
        bounded_unstable_zeta = jnp.maximum(
            jnp.float64(-100.0),
            jnp.minimum(zeta, jnp.float64(-0.01)),
        )
        bounded_zeta = jnp.where(stable, bounded_stable_zeta, bounded_unstable_zeta)
        convective_argument = (
            -_multiply(
                _multiply(
                    _multiply(jnp.float64(GRAVITY), friction_calculation),
                    virtual_temperature_scale,
                ),
                column(inputs.convective_boundary_height_column),
            )
            / virtual_temperature_safe
        )
        safe_convective_argument = jnp.where(
            active & ~stable,
            jnp.maximum(convective_argument, jnp.float64(0.0)),
            jnp.float64(1.0),
        )
        convective_velocity = _multiply(
            column(inputs.convective_velocity_coefficient_column),
            _scalar_power(safe_convective_argument, 0.333),
        )
        unstable_wind = jnp.sqrt(
            _add(
                _multiply(wind, wind),
                _multiply(convective_velocity, convective_velocity),
            )
        )
        new_stability_wind = jnp.where(stable, jnp.maximum(wind, 0.1), unstable_wind)
        new_stability_wind = jnp.where(active, new_stability_wind, stability_wind)
        new_monin_length = jnp.where(
            active,
            height / bounded_zeta,
            monin_obukhov_length,
        )
        return (
            new_stability_wind,
            new_monin_length,
            new_sensible_roughness,
            new_latent_roughness,
            friction_result.state,
            friction,
            temperature_profile,
            humidity_profile,
            friction_result.temperature_profile_2m_patch,
            friction_result.humidity_profile_2m_patch,
        ), None

    final_iteration_state = initial_iteration_state
    for zero_based_iteration in range(NUMBER_OF_STABILITY_ITERATIONS):
        final_iteration_state, _ = stability_iteration(
            final_iteration_state,
            jnp.int32(zero_based_iteration),
        )
        final_iteration_state = tree_util.tree_map(
            lax.optimization_barrier,
            final_iteration_state,
        )
    (
        stability_wind,
        _monin_obukhov_length,
        _sensible_roughness,
        _latent_roughness,
        friction_state,
        friction,
        temperature_profile,
        humidity_profile,
        temperature_profile_2m,
        humidity_profile_2m,
    ) = final_iteration_state

    friction_calculation = jnp.where(active, friction, jnp.float64(0.1))
    temperature_profile_calculation = jnp.where(
        active, temperature_profile, jnp.float64(0.1)
    )
    humidity_profile_calculation = jnp.where(
        active, humidity_profile, jnp.float64(0.1)
    )
    temperature_profile_2m_calculation = jnp.where(
        active, temperature_profile_2m, jnp.float64(0.1)
    )
    humidity_profile_2m_calculation = jnp.where(
        active, humidity_profile_2m, jnp.float64(0.1)
    )
    momentum_resistance = _divide(
        jnp.float64(1.0),
        _divide(_multiply(friction_calculation, friction_calculation), stability_wind),
    )
    moisture_resistance = _divide(
        jnp.float64(1.0),
        _multiply(humidity_profile_calculation, friction_calculation),
    )
    heat_conductance = _sensible_heat_conductance(
        forcing_density_safe, temperature_profile_calculation, friction_calculation,
    )
    dew = humidity_difference > jnp.float64(0.0)
    evaporation_conductance = _divide(
        forcing_density_safe,
        _add(
            moisture_resistance,
            jnp.where(
                dew,
                jnp.float64(0.0),
                column(inputs.soil_evaporation_resistance_column),
            ),
        ),
    )

    snow_level = jnp.clip(
        inputs.snow_layer_count_column[patch_column] + jnp.int32(NUMBER_OF_SNOW_LEVELS),
        min=jnp.int32(0),
        max=jnp.int32(NUMBER_OF_TOTAL_LEVELS - 1),
    )
    snow_surface_temperature = inputs.soil_snow_temperature_column_level[
        patch_column, snow_level
    ]
    soil_surface_temperature = inputs.soil_snow_temperature_column_level[
        patch_column, jnp.int32(NUMBER_OF_SNOW_LEVELS)
    ]
    surface_water_temperature = column(inputs.surface_water_temperature_column)

    ground_sensible_derivative = heat_conductance
    ground_latent_derivative = _multiply(
        evaporation_conductance,
        column(inputs.ground_specific_humidity_temperature_derivative_column),
    )
    ground_energy_derivative = _add_boundary(
        ground_sensible_derivative,
        _multiply_boundary(column(inputs.latent_heat_column), ground_latent_derivative),
    )
    eastward_stress = -_divide(
        _multiply(forcing_density_safe, forcing_u_safe),
        momentum_resistance,
    )
    northward_stress = -_divide(
        _multiply(forcing_density_safe, forcing_v_safe),
        momentum_resistance,
    )
    sensible_ground = -_multiply(heat_conductance, temperature_difference)
    sensible_snow = -_multiply(
        heat_conductance,
        _add(reference_temperature_safe, -snow_surface_temperature),
    )
    sensible_soil = -_multiply(
        heat_conductance,
        _add(reference_temperature_safe, -soil_surface_temperature),
    )
    sensible_surface_water = -_multiply(
        heat_conductance,
        _add(reference_temperature_safe, -surface_water_temperature),
    )
    ground_evaporation = -_multiply(evaporation_conductance, humidity_difference)
    evaporation_snow = -_multiply(
        evaporation_conductance,
        _add(forcing_humidity_safe, -column(inputs.snow_surface_specific_humidity_column)),
    )
    evaporation_soil = -_multiply(
        evaporation_conductance,
        _add(forcing_humidity_safe, -column(inputs.soil_surface_specific_humidity_column)),
    )
    evaporation_surface_water = -_multiply(
        evaporation_conductance,
        _add(
            forcing_humidity_safe,
            -column(inputs.surface_water_specific_humidity_column),
        ),
    )

    reference_temperature_2m = _add(
        reference_temperature_safe,
        _multiply(
            _multiply(temperature_profile_calculation, temperature_difference),
            _add(
                _divide(jnp.float64(1.0), temperature_profile_2m_calculation),
                -_divide(jnp.float64(1.0), temperature_profile_calculation),
            ),
        ),
    )
    reference_humidity_2m = _reference_humidity_at_two_meters(
        forcing_humidity_safe,
        humidity_difference,
        humidity_profile_calculation,
        humidity_profile_2m_calculation,
    )
    saturation_2m = qsat(reference_temperature_2m, forcing_pressure_safe)
    relative_humidity_2m = jnp.minimum(
        jnp.float64(100.0),
        _multiply_boundary(
            _divide(reference_humidity_2m, saturation_2m.specific_humidity),
            jnp.float64(100.0),
        ),
    )
    safe_relative_humidity = jnp.where(active, relative_humidity_2m, 50.0)
    human_indices = fast_human_stress_indices(
        reference_temperature_2m,
        safe_relative_humidity,
        saturation_2m.vapor_pressure,
        jnp.where(active, friction_state.wind_10m_patch, jnp.float64(1.0)),
    )

    stomatal_denominator = _multiply_boundary(
        _multiply_boundary(jnp.float64(UNIVERSAL_GAS_CONSTANT), jnp.float64(0.001)),
        reference_temperature_safe,
    )
    stomatal_conversion = _multiply_boundary(
        _divide(forcing_pressure_safe, stomatal_denominator),
        jnp.float64(1.0e6),
    )
    stomatal_factor = _multiply_boundary(
        _divide(jnp.float64(1.0), jnp.float64(1.0e15)),
        stomatal_conversion,
    )

    energy = BareGroundEnergyState(
        sensible_heat_snow_patch=_update(
            active, sensible_snow, state.energy.sensible_heat_snow_patch
        ),
        sensible_heat_soil_patch=_update(
            active, sensible_soil, state.energy.sensible_heat_soil_patch
        ),
        sensible_heat_surface_water_patch=_update(
            active, sensible_surface_water, state.energy.sensible_heat_surface_water_patch
        ),
        sensible_heat_ground_patch=_update(
            active, sensible_ground, state.energy.sensible_heat_ground_patch
        ),
        sensible_heat_total_patch=_update(
            active, sensible_ground, state.energy.sensible_heat_total_patch
        ),
        eastward_wind_stress_patch=_update(
            active, eastward_stress, state.energy.eastward_wind_stress_patch
        ),
        northward_wind_stress_patch=_update(
            active, northward_stress, state.energy.northward_wind_stress_patch
        ),
        downward_longwave_below_canopy_patch=_update(
            active, jnp.zeros_like(safe_active), state.energy.downward_longwave_below_canopy_patch
        ),
        upward_longwave_above_canopy_patch=_update(
            active, jnp.zeros_like(safe_active), state.energy.upward_longwave_above_canopy_patch
        ),
        ground_sensible_derivative_patch=_update(
            active, ground_sensible_derivative, state.energy.ground_sensible_derivative_patch
        ),
        ground_latent_derivative_patch=_update(
            active, ground_latent_derivative, state.energy.ground_latent_derivative_patch
        ),
        ground_energy_derivative_patch=_update(
            active, ground_energy_derivative, state.energy.ground_energy_derivative_patch
        ),
        transpiration_wetness_patch=_update(
            active, jnp.zeros_like(safe_active), state.energy.transpiration_wetness_patch
        ),
        root_resistance_patch_level=jnp.where(
            active[:, None],
            jnp.zeros_like(state.energy.root_resistance_patch_level),
            state.energy.root_resistance_patch_level,
        ),
    )
    water = BareGroundWaterState(
        evaporation_snow_patch=_update(
            active, evaporation_snow, state.water.evaporation_snow_patch
        ),
        evaporation_soil_patch=_update(
            active, evaporation_soil, state.water.evaporation_soil_patch
        ),
        evaporation_surface_water_patch=_update(
            active, evaporation_surface_water, state.water.evaporation_surface_water_patch
        ),
        ground_evaporation_patch=_update(
            active, ground_evaporation, state.water.ground_evaporation_patch
        ),
        total_evaporation_patch=_update(
            active, ground_evaporation, state.water.total_evaporation_patch
        ),
        reference_specific_humidity_patch=_update(
            active, reference_humidity_2m, state.water.reference_specific_humidity_patch
        ),
        reference_relative_humidity_patch=_update(
            active, relative_humidity_2m, state.water.reference_relative_humidity_patch
        ),
        rural_reference_relative_humidity_patch=_update(
            rural, relative_humidity_2m, state.water.rural_reference_relative_humidity_patch
        ),
    )
    temperature = BareGroundTemperatureState(
        reference_temperature_patch=_update(
            active, reference_temperature_2m, state.temperature.reference_temperature_patch
        ),
        rural_reference_temperature_patch=_update(
            rural, reference_temperature_2m, state.temperature.rural_reference_temperature_patch
        ),
        vegetation_temperature_patch=_update(
            active, forcing_temperature_safe, state.temperature.vegetation_temperature_patch
        ),
    )
    photosynthesis = BareGroundPhotosynthesisState(
        sunlit_stomatal_resistance_patch=_update(
            active,
            stomatal_factor,
            state.photosynthesis.sunlit_stomatal_resistance_patch,
        ),
        shaded_stomatal_resistance_patch=_update(
            active,
            stomatal_factor,
            state.photosynthesis.shaded_stomatal_resistance_patch,
        ),
    )
    return BareGroundFluxState(
        energy=energy,
        water=water,
        temperature=temperature,
        photosynthesis=photosynthesis,
        human_index=_human_state_update(active, rural, human_indices, state.human_index),
        friction_velocity=friction_state,
        effective_root_fraction_patch_level=jnp.where(
            active[:, None],
            jnp.zeros_like(state.effective_root_fraction_patch_level),
            state.effective_root_fraction_patch_level,
        ),
        ground_methane_conductance_patch=_update(
            active,
            _divide(jnp.float64(1.0), moisture_resistance),
            state.ground_methane_conductance_patch,
        ),
        aerodynamic_resistance_patch=_update(
            active,
            momentum_resistance,
            state.aerodynamic_resistance_patch,
        ),
    )
