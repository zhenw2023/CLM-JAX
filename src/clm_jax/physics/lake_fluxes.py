
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax, tree_util

from clm_jax.constants import (
    FREEZING_TEMPERATURE_FRESHWATER,
    GRAVITY,
    LATENT_HEAT_SUBLIMATION,
    LATENT_HEAT_VAPORIZATION,
    SPECIFIC_HEAT_DRY_AIR,
    STEFAN_BOLTZMANN,
    VON_KARMAN,
)
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
    fortran_power,
    strict_exp,
    strict_negative_exp,
)
from clm_jax.physics.native_power import native_power
from clm_jax.physics.pointwise import scalar_map
from clm_jax.physics.qsat import qsat

NUMBER_OF_LAKE_LEVELS = 10
NUMBER_OF_SNOW_LEVELS = 12
NUMBER_OF_GROUND_LEVELS = 25
NUMBER_OF_TOTAL_LEVELS = NUMBER_OF_SNOW_LEVELS + NUMBER_OF_GROUND_LEVELS
NUMBER_OF_STABILITY_ITERATIONS = 4

LAKE_EMISSIVITY = 0.97
VISIBLE_SURFACE_ABSORPTION_FRACTION = 0.0
FROZEN_LAKE_ROUGHNESS = 0.001
SNOW_ROUGHNESS = 0.0024
MAXIMUM_DENSITY_TEMPERATURE = 277.0
MINIMUM_OPEN_WATER_ROUGHNESS = 1.0e-10
BASE_CHARNOCK_PARAMETER = 0.01
SMOOTH_FLOW_ROUGHNESS_COEFFICIENT = 0.1
MAXIMUM_CHARNOCK_INCREMENT = 0.1
CRITICAL_DIMENSIONLESS_FETCH = 100.0
AIR_KINEMATIC_VISCOSITY = 1.51e-5
AIR_KINEMATIC_VISCOSITY_TEMPERATURE = 293.15
AIR_KINEMATIC_VISCOSITY_PRESSURE = 1.013e5
AIR_PRANDTL_NUMBER = float.fromhex("0x1.6d0e560000000p-1")
WATER_VAPOR_SCHMIDT_NUMBER = float.fromhex("0x1.51eb860000000p-1")
AIR_THERMAL_CONDUCTIVITY = 0.023
ICE_THERMAL_CONDUCTIVITY = 2.290


class LakeFluxInputs(NamedTuple):
    active_lake: jnp.ndarray
    latitude_radians: jnp.ndarray
    eastward_wind: jnp.ndarray
    northward_wind: jnp.ndarray
    atmospheric_temperature: jnp.ndarray
    atmospheric_potential_temperature: jnp.ndarray
    atmospheric_pressure: jnp.ndarray
    atmospheric_density: jnp.ndarray
    atmospheric_specific_humidity: jnp.ndarray
    downward_longwave: jnp.ndarray
    rain_flux: jnp.ndarray
    snow_flux: jnp.ndarray
    incident_nir_direct: jnp.ndarray
    incident_nir_diffuse: jnp.ndarray
    reflected_nir_direct: jnp.ndarray
    reflected_nir_diffuse: jnp.ndarray
    ground_absorbed_shortwave: jnp.ndarray
    layer_absorbed_shortwave: jnp.ndarray
    snow_layer_count: jnp.ndarray
    soil_snow_layer_thickness: jnp.ndarray
    lake_layer_thickness: jnp.ndarray
    lake_depth: jnp.ndarray
    lake_fetch: jnp.ndarray
    soil_snow_temperature: jnp.ndarray
    soil_snow_liquid_water: jnp.ndarray
    soil_snow_ice_water: jnp.ndarray
    lake_temperature: jnp.ndarray
    saved_top_eddy_conductivity: jnp.ndarray
    forcing_height_wind: jnp.ndarray
    forcing_height_temperature: jnp.ndarray
    forcing_height_humidity: jnp.ndarray


class LakeFluxState(NamedTuple):
    ground_temperature: jnp.ndarray
    momentum_roughness: jnp.ndarray
    sensible_roughness: jnp.ndarray
    latent_roughness: jnp.ndarray
    lake_friction_velocity: jnp.ndarray
    friction_velocity: FrictionVelocityState


class LakeEnergyFluxes(NamedTuple):
    outgoing_longwave: jnp.ndarray
    net_longwave: jnp.ndarray
    soil_ground_heat: jnp.ndarray
    sensible_ground: jnp.ndarray
    sensible_total: jnp.ndarray
    latent_ground: jnp.ndarray
    latent_total: jnp.ndarray
    ground_net: jnp.ndarray
    eastward_wind_stress: jnp.ndarray
    northward_wind_stress: jnp.ndarray


class LakeWaterFluxes(NamedTuple):
    ground_evaporation: jnp.ndarray
    total_evaporation: jnp.ndarray
    ground_precipitation: jnp.ndarray
    reference_specific_humidity: jnp.ndarray
    reference_relative_humidity: jnp.ndarray


class LakeTemperatureState(NamedTuple):
    ground_temperature: jnp.ndarray
    vegetation_temperature: jnp.ndarray
    reference_temperature: jnp.ndarray
    skin_temperature: jnp.ndarray


class LakeSurfaceState(NamedTuple):
    surface_absorption_fraction: jnp.ndarray
    checked_ground_absorbed_shortwave: jnp.ndarray
    aerodynamic_resistance: jnp.ndarray
    lake_aerodynamic_resistance: jnp.ndarray
    moisture_resistance: jnp.ndarray
    wind_mixing_velocity: jnp.ndarray
    mixing_decay_coefficient: jnp.ndarray
    forcing_height_wind: jnp.ndarray
    forcing_height_temperature: jnp.ndarray
    forcing_height_humidity: jnp.ndarray


class LakeFluxDiagnostics(NamedTuple):
    active_iteration: jnp.ndarray
    sign_change_count_iteration: jnp.ndarray
    ground_temperature_iteration: jnp.ndarray
    stability_wind_iteration: jnp.ndarray
    monin_obukhov_length_iteration: jnp.ndarray
    friction_velocity_iteration: jnp.ndarray
    momentum_resistance_iteration: jnp.ndarray
    thermal_resistance_iteration: jnp.ndarray
    moisture_resistance_iteration: jnp.ndarray
    momentum_roughness_iteration: jnp.ndarray
    sensible_roughness_iteration: jnp.ndarray
    latent_roughness_iteration: jnp.ndarray


class LakeFluxResult(NamedTuple):
    state: LakeFluxState
    energy: LakeEnergyFluxes
    water: LakeWaterFluxes
    temperature: LakeTemperatureState
    human_index: FastHumanStressIndices
    surface: LakeSurfaceState
    diagnostics: LakeFluxDiagnostics


def _sum(*terms: jnp.ndarray) -> jnp.ndarray:
    result = jnp.asarray(terms[0], dtype=jnp.float64)
    for term in terms[1:]:
        result = fortran_add(result, jnp.asarray(term, dtype=jnp.float64))
    return result


def _multiply(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return fortran_multiply(
        jnp.asarray(left, dtype=jnp.float64),
        jnp.asarray(right, dtype=jnp.float64),
    )


def _divide(numerator: jnp.ndarray, denominator: jnp.ndarray) -> jnp.ndarray:
    return fortran_divide(
        jnp.asarray(numerator, dtype=jnp.float64),
        jnp.asarray(denominator, dtype=jnp.float64),
    )


def _power(base: jnp.ndarray, exponent: float | jnp.ndarray) -> jnp.ndarray:
    return fortran_power(
        jnp.asarray(base, dtype=jnp.float64),
        jnp.asarray(exponent, dtype=jnp.float64),
    )


def _scalar_log(value: jnp.ndarray) -> jnp.ndarray:
    value = jnp.asarray(value, dtype=jnp.float64)
    shape = value.shape
    return jnp.reshape(scalar_map(jnp.log, jnp.reshape(value, (-1,))), shape)


def _scalar_sin(value: jnp.ndarray) -> jnp.ndarray:
    value = jnp.asarray(value, dtype=jnp.float64)
    shape = value.shape
    return jnp.reshape(scalar_map(jnp.sin, jnp.reshape(value, (-1,))), shape)


def _mixing_decay(wind_two_meters: jnp.ndarray, latitude_radians: jnp.ndarray) -> jnp.ndarray:
    sine_magnitude = jnp.abs(_scalar_sin(latitude_radians))
    latitude_factor = jnp.where(
        sine_magnitude > jnp.float64(0.0),
        jnp.sqrt(jnp.where(sine_magnitude > 0.0, sine_magnitude, jnp.float64(1.0))),
        jnp.float64(0.0),
    )
    return _multiply(
        _multiply(jnp.float64(6.6), latitude_factor),
        native_power(wind_two_meters, jnp.float64(-1.84)),
    )


def _validate_shapes(inputs: LakeFluxInputs, state: LakeFluxState) -> None:
    size = inputs.active_lake.shape[0]
    vector_fields = set(LakeFluxInputs._fields) - {
        "layer_absorbed_shortwave",
        "soil_snow_layer_thickness",
        "lake_layer_thickness",
        "soil_snow_temperature",
        "soil_snow_liquid_water",
        "soil_snow_ice_water",
        "lake_temperature",
    }
    for name in vector_fields:
        if getattr(inputs, name).shape != (size,):
            raise ValueError(f"{name} must have shape ({size},)")
    expected_matrices = {
        "layer_absorbed_shortwave": (size, NUMBER_OF_SNOW_LEVELS + 1),
        "soil_snow_layer_thickness": (size, NUMBER_OF_TOTAL_LEVELS),
        "lake_layer_thickness": (size, NUMBER_OF_LAKE_LEVELS),
        "soil_snow_temperature": (size, NUMBER_OF_TOTAL_LEVELS),
        "soil_snow_liquid_water": (size, NUMBER_OF_TOTAL_LEVELS),
        "soil_snow_ice_water": (size, NUMBER_OF_TOTAL_LEVELS),
        "lake_temperature": (size, NUMBER_OF_LAKE_LEVELS),
    }
    for name, expected in expected_matrices.items():
        if getattr(inputs, name).shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    for name in LakeFluxState._fields[:-1]:
        if getattr(state, name).shape != (size,):
            raise ValueError(f"{name} must have shape ({size},)")
    for name, value in zip(
        FrictionVelocityState._fields,
        state.friction_velocity,
        strict=True,
    ):
        if value.shape != (size,):
            raise ValueError(f"friction_velocity.{name} must have shape ({size},)")


def _thermal_roughness(
    momentum_roughness: jnp.ndarray,
    friction_velocity_value: jnp.ndarray,
    kinematic_viscosity: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    reynolds_root = jnp.sqrt(
        jnp.maximum(
            _divide(
                _multiply(momentum_roughness, friction_velocity_value),
                kinematic_viscosity,
            ),
            jnp.float64(0.1),
        )
    )
    sensible_exponent = _multiply(
        _divide(-jnp.float64(VON_KARMAN), jnp.float64(AIR_PRANDTL_NUMBER)),
        _sum(_multiply(jnp.float64(4.0), reynolds_root), jnp.float64(-3.2)),
    )
    latent_exponent = _multiply(
        _divide(
            -jnp.float64(VON_KARMAN),
            jnp.float64(WATER_VAPOR_SCHMIDT_NUMBER),
        ),
        _sum(_multiply(jnp.float64(4.0), reynolds_root), jnp.float64(-4.2)),
    )
    sensible = _multiply(momentum_roughness, strict_exp(sensible_exponent))
    latent = _multiply(momentum_roughness, strict_exp(latent_exponent))
    return (
        jnp.maximum(sensible, jnp.float64(MINIMUM_OPEN_WATER_ROUGHNESS)),
        jnp.maximum(latent, jnp.float64(MINIMUM_OPEN_WATER_ROUGHNESS)),
    )


def _frozen_thermal_roughness(
    momentum_roughness: jnp.ndarray,
    friction_velocity_value: jnp.ndarray,
) -> jnp.ndarray:
    reynolds = _divide(
        _multiply(friction_velocity_value, momentum_roughness),
        jnp.float64(1.5e-5),
    )
    positive_reynolds = reynolds > jnp.float64(0.0)
    reynolds_for_power = jnp.where(positive_reynolds, reynolds, jnp.float64(1.0))
    exponent = jnp.where(
        positive_reynolds,
        _multiply(jnp.float64(0.13), _power(reynolds_for_power, 0.45)),
        jnp.float64(0.0),
    )
    return _divide(momentum_roughness, strict_exp(exponent))


def _kinematic_viscosity(
    ground_temperature: jnp.ndarray,
    atmospheric_pressure: jnp.ndarray,
) -> jnp.ndarray:
    temperature_ratio = _divide(
        ground_temperature,
        jnp.float64(AIR_KINEMATIC_VISCOSITY_TEMPERATURE),
    )
    value = _multiply(
        jnp.float64(AIR_KINEMATIC_VISCOSITY),
        _power(temperature_ratio, 1.5),
    )
    value = _multiply(value, jnp.float64(AIR_KINEMATIC_VISCOSITY_PRESSURE))
    return _divide(value, atmospheric_pressure)


def _roughness_for_surface(
    ground_temperature: jnp.ndarray,
    snow_layer_count: jnp.ndarray,
    previous_momentum_roughness: jnp.ndarray,
    previous_lake_friction_velocity: jnp.ndarray,
    atmospheric_pressure: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    open_water = ground_temperature > jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)
    snow_free = snow_layer_count == jnp.int32(0)
    frozen_momentum = jnp.where(
        snow_free,
        jnp.float64(FROZEN_LAKE_ROUGHNESS),
        jnp.float64(SNOW_ROUGHNESS),
    )
    momentum = jnp.where(open_water, previous_momentum_roughness, frozen_momentum)
    viscosity = _kinematic_viscosity(ground_temperature, atmospheric_pressure)
    open_sensible, open_latent = _thermal_roughness(
        momentum,
        previous_lake_friction_velocity,
        viscosity,
    )
    frozen_thermal = _frozen_thermal_roughness(
        momentum,
        previous_lake_friction_velocity,
    )
    sensible = jnp.where(open_water, open_sensible, frozen_thermal)
    latent = jnp.where(open_water, open_latent, frozen_thermal)
    return momentum, sensible, latent


def _updated_roughness(
    ground_temperature: jnp.ndarray,
    snow_layer_count: jnp.ndarray,
    lake_depth: jnp.ndarray,
    fetch: jnp.ndarray,
    reference_wind: jnp.ndarray,
    friction_velocity_value: jnp.ndarray,
    atmospheric_pressure: jnp.ndarray,
    previous_momentum: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    open_water = (ground_temperature > jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)) & (
        snow_layer_count == jnp.int32(0)
    )
    frozen_no_snow = snow_layer_count == jnp.int32(0)
    calculation_friction = jnp.where(
        friction_velocity_value > jnp.float64(0.0),
        friction_velocity_value,
        jnp.float64(1.0e-4),
    )
    one_third = _divide(jnp.float64(1.0), jnp.float64(3.0))
    fetch_scale = _divide(
        _divide(
            _multiply(fetch, jnp.float64(GRAVITY)),
            calculation_friction,
        ),
        calculation_friction,
    )
    depth_scale = _divide(
        _divide(
            _multiply(lake_depth, jnp.float64(GRAVITY)),
            reference_wind,
        ),
        reference_wind,
    )
    fetch_limited = -_divide(
        _power(fetch_scale, one_third),
        jnp.float64(CRITICAL_DIMENSIONLESS_FETCH),
    )
    depth_limited = -_power(depth_scale, 0.5)
    charnock = _sum(
        jnp.float64(BASE_CHARNOCK_PARAMETER),
        _multiply(
            jnp.float64(MAXIMUM_CHARNOCK_INCREMENT),
            strict_negative_exp(jnp.maximum(fetch_limited, depth_limited)),
        ),
    )
    viscosity = _kinematic_viscosity(ground_temperature, atmospheric_pressure)
    smooth_roughness = _divide(
        _multiply(jnp.float64(SMOOTH_FLOW_ROUGHNESS_COEFFICIENT), viscosity),
        jnp.maximum(calculation_friction, jnp.float64(1.0e-4)),
    )
    wave_roughness = _divide(
        _multiply(
            _multiply(charnock, calculation_friction),
            calculation_friction,
        ),
        jnp.float64(GRAVITY),
    )
    open_momentum = jnp.maximum(
        jnp.maximum(smooth_roughness, wave_roughness),
        jnp.float64(MINIMUM_OPEN_WATER_ROUGHNESS),
    )
    momentum = jnp.where(
        open_water,
        open_momentum,
        jnp.where(
            frozen_no_snow,
            jnp.float64(FROZEN_LAKE_ROUGHNESS),
            previous_momentum,
        ),
    )
    open_sensible, open_latent = _thermal_roughness(
        momentum,
        calculation_friction,
        viscosity,
    )
    frozen_thermal = _frozen_thermal_roughness(momentum, calculation_friction)
    sensible = jnp.where(open_water, open_sensible, frozen_thermal)
    latent = jnp.where(open_water, open_latent, frozen_thermal)
    return (
        jnp.where(jnp.isfinite(momentum), momentum, previous_momentum),
        sensible,
        latent,
    )


def lake_fluxes(
    inputs: LakeFluxInputs,
    state: LakeFluxState,
    maximum_stable_zeta: float | jnp.ndarray = 0.5,
) -> LakeFluxResult:

    _validate_shapes(inputs, state)
    active = jnp.asarray(inputs.active_lake, dtype=jnp.bool_)
    size = active.shape[0]

    def vector(name: str, inactive: float) -> jnp.ndarray:
        value = jnp.asarray(getattr(inputs, name), dtype=jnp.float64)
        return jnp.where(active, value, jnp.float64(inactive))

    snow_layer_count = jnp.where(
        active,
        jnp.asarray(inputs.snow_layer_count, dtype=jnp.int32),
        jnp.int32(0),
    )
    atmospheric_pressure = vector("atmospheric_pressure", 100000.0)
    atmospheric_temperature = vector("atmospheric_temperature", 280.0)
    atmospheric_theta = vector("atmospheric_potential_temperature", 280.0)
    atmospheric_density = vector("atmospheric_density", 1.2)
    atmospheric_humidity = vector("atmospheric_specific_humidity", 0.005)
    downward_longwave = vector("downward_longwave", 300.0)
    ground_shortwave = vector("ground_absorbed_shortwave", 0.0)
    lake_depth = vector("lake_depth", 10.0)
    lake_fetch = vector("lake_fetch", -1.0)
    layer_absorbed_shortwave = jnp.where(
        active[:, None],
        jnp.asarray(inputs.layer_absorbed_shortwave, dtype=jnp.float64),
        jnp.float64(0.0),
    )
    soil_snow_layer_thickness = jnp.where(
        active[:, None],
        jnp.asarray(inputs.soil_snow_layer_thickness, dtype=jnp.float64),
        jnp.float64(1.0),
    )
    lake_layer_thickness = jnp.where(
        active[:, None],
        jnp.asarray(inputs.lake_layer_thickness, dtype=jnp.float64),
        jnp.float64(1.0),
    )
    soil_snow_temperature = jnp.where(
        active[:, None],
        jnp.asarray(inputs.soil_snow_temperature, dtype=jnp.float64),
        jnp.float64(273.15),
    )
    soil_snow_liquid_water = jnp.where(
        active[:, None],
        jnp.asarray(inputs.soil_snow_liquid_water, dtype=jnp.float64),
        jnp.float64(0.0),
    )
    soil_snow_ice_water = jnp.where(
        active[:, None],
        jnp.asarray(inputs.soil_snow_ice_water, dtype=jnp.float64),
        jnp.float64(0.0),
    )
    lake_temperature = jnp.where(
        active[:, None],
        jnp.asarray(inputs.lake_temperature, dtype=jnp.float64),
        jnp.float64(273.15),
    )
    ground_temperature = jnp.where(
        active,
        jnp.asarray(state.ground_temperature, dtype=jnp.float64),
        jnp.float64(280.0),
    )
    prior_momentum = jnp.where(
        active,
        jnp.asarray(state.momentum_roughness, dtype=jnp.float64),
        jnp.float64(0.01),
    )
    prior_lake_friction = jnp.where(
        active,
        jnp.asarray(state.lake_friction_velocity, dtype=jnp.float64),
        jnp.float64(0.1),
    )
    momentum, sensible, latent = _roughness_for_surface(
        ground_temperature,
        snow_layer_count,
        prior_momentum,
        prior_lake_friction,
        atmospheric_pressure,
    )
    momentum = jnp.where(active, momentum, prior_momentum)
    sensible = jnp.where(
        active,
        sensible,
        jnp.asarray(state.sensible_roughness, dtype=jnp.float64),
    )
    latent = jnp.where(
        active,
        latent,
        jnp.asarray(state.latent_roughness, dtype=jnp.float64),
    )

    forcing_height_wind = jnp.where(
        active,
        _sum(vector("forcing_height_wind", 30.0), momentum),
        jnp.asarray(inputs.forcing_height_wind, dtype=jnp.float64),
    )
    forcing_height_temperature = jnp.where(
        active,
        _sum(vector("forcing_height_temperature", 30.0), momentum),
        jnp.asarray(inputs.forcing_height_temperature, dtype=jnp.float64),
    )
    forcing_height_humidity = jnp.where(
        active,
        _sum(vector("forcing_height_humidity", 30.0), momentum),
        jnp.asarray(inputs.forcing_height_humidity, dtype=jnp.float64),
    )

    top_index = jnp.clip(
        snow_layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS),
        jnp.int32(0),
        jnp.int32(NUMBER_OF_TOTAL_LEVELS - 1),
    )
    top_layer_absorption = jnp.take_along_axis(
        layer_absorbed_shortwave,
        jnp.minimum(top_index, jnp.int32(NUMBER_OF_SNOW_LEVELS))[:, None],
        axis=1,
    )[:, 0]
    top_layer_thickness = jnp.take_along_axis(
        soil_snow_layer_thickness,
        top_index[:, None],
        axis=1,
    )[:, 0]
    nir_absorbed = _sum(
        vector("incident_nir_direct", 0.0),
        vector("incident_nir_diffuse", 0.0),
        -vector("reflected_nir_direct", 0.0),
        -vector("reflected_nir_diffuse", 0.0),
    )
    nir_absorbed = jnp.minimum(nir_absorbed, ground_shortwave)
    shortwave_denominator = jnp.maximum(jnp.float64(1.0e-5), ground_shortwave)
    snow_absorption_fraction = _divide(top_layer_absorption, shortwave_denominator)
    open_absorption_fraction = _divide(nir_absorbed, shortwave_denominator)
    open_absorption_fraction = _sum(
        open_absorption_fraction,
        _multiply(
            _sum(jnp.float64(1.0), -open_absorption_fraction),
            jnp.float64(VISIBLE_SURFACE_ABSORPTION_FRACTION),
        ),
    )
    has_snow = snow_layer_count < jnp.int32(0)
    surface_absorption_fraction = jnp.where(
        has_snow,
        snow_absorption_fraction,
        open_absorption_fraction,
    )
    surface_layer_half_thickness = jnp.where(
        has_snow,
        _divide(top_layer_thickness, jnp.float64(2.0)),
        _divide(
            lake_layer_thickness[:, 0],
            jnp.float64(2.0),
        ),
    )
    initial_saturation = qsat(ground_temperature, atmospheric_pressure)
    surface_humidity = initial_saturation.specific_humidity
    surface_humidity_derivative = initial_saturation.specific_humidity_temperature_derivative
    atmospheric_reference_temperature = _sum(
        atmospheric_temperature,
        _multiply(jnp.float64(0.0098), forcing_height_temperature),
    )
    atmospheric_virtual_temperature = _multiply(
        atmospheric_theta,
        _sum(
            jnp.float64(1.0),
            _multiply(jnp.float64(0.61), atmospheric_humidity),
        ),
    )
    eastward_wind = vector("eastward_wind", 1.0)
    northward_wind = vector("northward_wind", 0.0)
    reference_wind = jnp.maximum(
        jnp.float64(1.0),
        jnp.sqrt(
            _sum(
                _multiply(eastward_wind, eastward_wind),
                _multiply(northward_wind, northward_wind),
            )
        ),
    )
    temperature_difference = _sum(
        atmospheric_reference_temperature,
        -ground_temperature,
    )
    humidity_difference = _sum(atmospheric_humidity, -surface_humidity)
    virtual_temperature_difference = _sum(
        _multiply(
            temperature_difference,
            _sum(
                jnp.float64(1.0),
                _multiply(jnp.float64(0.61), atmospheric_humidity),
            ),
        ),
        _multiply(
            _multiply(jnp.float64(0.61), atmospheric_theta),
            humidity_difference,
        ),
    )
    initialized = monin_obukhov_initialize(
        reference_wind,
        atmospheric_virtual_temperature,
        virtual_temperature_difference,
        forcing_height_wind,
        momentum,
        maximum_stable_zeta,
    )
    stability_wind = initialized.stability_wind_speed
    monin_length = initialized.monin_obukhov_length
    latent_heat = jnp.where(
        ground_temperature > jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
        jnp.float64(LATENT_HEAT_VAPORIZATION),
        jnp.float64(LATENT_HEAT_SUBLIMATION),
    )
    fetch = jnp.where(
        lake_fetch > jnp.float64(0.0),
        lake_fetch,
        jnp.where(lake_depth < jnp.float64(4.0), 100.0, _multiply(25.0, lake_depth)),
    )

    zeros = jnp.zeros(size, dtype=jnp.float64)
    initial_iteration_state = (
        active,
        jnp.zeros(size, dtype=jnp.int32),
        zeros,
        stability_wind,
        monin_length,
        momentum,
        sensible,
        latent,
        ground_temperature,
        surface_humidity,
        surface_humidity_derivative,
        latent_heat,
        state.friction_velocity,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        temperature_difference,
        humidity_difference,
        zeros,
        zeros,
    )

    def iteration_body(iteration_state, zero_based_iteration):
        (
            iteration_active,
            sign_change_count,
            previous_monin_length,
            stability_wind,
            monin_length,
            momentum,
            sensible,
            latent,
            ground_temperature,
            surface_humidity,
            surface_humidity_derivative,
            latent_heat,
            friction_state,
            _friction_value,
            _temperature_profile,
            _humidity_profile,
            _temperature_profile_2m,
            _humidity_profile_2m,
            _momentum_resistance,
            _thermal_resistance,
            _temperature_difference,
            _humidity_difference,
            _sensible_heat,
            _evaporation,
        ) = iteration_state
        friction_result = friction_velocity(
            iteration_active,
            jnp.zeros(size, dtype=jnp.float64),
            momentum,
            sensible,
            latent,
            monin_length,
            jnp.int32(zero_based_iteration + 1),
            reference_wind,
            stability_wind,
            forcing_height_wind,
            forcing_height_temperature,
            forcing_height_humidity,
            friction_state,
        )
        friction_value = jnp.where(
            iteration_active,
            friction_result.friction_velocity_patch,
            _friction_value,
        )
        temperature_profile = jnp.where(
            iteration_active,
            friction_result.temperature_profile_patch,
            _temperature_profile,
        )
        humidity_profile = jnp.where(
            iteration_active,
            friction_result.humidity_profile_patch,
            _humidity_profile,
        )
        temperature_profile_2m = jnp.where(
            iteration_active,
            friction_result.temperature_profile_2m_patch,
            _temperature_profile_2m,
        )
        humidity_profile_2m = jnp.where(
            iteration_active,
            friction_result.humidity_profile_2m_patch,
            _humidity_profile_2m,
        )
        top_temperature = jnp.take_along_axis(
            soil_snow_temperature,
            top_index[:, None],
            axis=1,
        )[:, 0]
        top_liquid = jnp.take_along_axis(
            soil_snow_liquid_water,
            top_index[:, None],
            axis=1,
        )[:, 0]
        top_ice = jnp.take_along_axis(
            soil_snow_ice_water,
            top_index[:, None],
            axis=1,
        )[:, 0]
        open_surface = (
            (ground_temperature > jnp.float64(FREEZING_TEMPERATURE_FRESHWATER))
            & (lake_temperature[:, 0] > jnp.float64(FREEZING_TEMPERATURE_FRESHWATER))
            & (snow_layer_count == jnp.int32(0))
        )
        snow_free = snow_layer_count == jnp.int32(0)
        snow_bulk_density = _divide(
            _sum(top_ice, top_liquid),
            top_layer_thickness,
        )
        snow_conductivity = _sum(
            jnp.float64(AIR_THERMAL_CONDUCTIVITY),
            _multiply(
                _sum(
                    _multiply(jnp.float64(7.75e-5), snow_bulk_density),
                    _multiply(
                        jnp.float64(1.105e-6),
                        _multiply(snow_bulk_density, snow_bulk_density),
                    ),
                ),
                jnp.float64(ICE_THERMAL_CONDUCTIVITY - AIR_THERMAL_CONDUCTIVITY),
            ),
        )
        surface_conductivity = jnp.where(
            open_surface,
            vector("saved_top_eddy_conductivity", 0.57),
            jnp.where(snow_free, jnp.float64(ICE_THERMAL_CONDUCTIVITY), snow_conductivity),
        )
        surface_layer_temperature = jnp.where(
            snow_free,
            lake_temperature[:, 0],
            top_temperature,
        )
        momentum_resistance = _divide(
            jnp.float64(1.0),
            _divide(_multiply(friction_value, friction_value), stability_wind),
        )
        thermal_resistance = _divide(
            jnp.float64(1.0),
            _multiply(temperature_profile, friction_value),
        )
        moisture_resistance = _divide(
            jnp.float64(1.0),
            _multiply(humidity_profile, friction_value),
        )
        previous_ground_temperature = ground_temperature
        longwave_derivative_quarter = _multiply(
            jnp.float64(LAKE_EMISSIVITY),
            jnp.float64(STEFAN_BOLTZMANN),
        )
        longwave_derivative_quarter = _multiply(
            longwave_derivative_quarter,
            previous_ground_temperature,
        )
        longwave_derivative_quarter = _multiply(
            longwave_derivative_quarter,
            previous_ground_temperature,
        )
        longwave_derivative_quarter = _multiply(
            longwave_derivative_quarter,
            previous_ground_temperature,
        )
        sensible_numerator = _multiply(
            _divide(
                _multiply(atmospheric_density, jnp.float64(SPECIFIC_HEAT_DRY_AIR)),
                thermal_resistance,
            ),
            atmospheric_reference_temperature,
        )
        humidity_linearization = _sum(
            surface_humidity,
            -_multiply(surface_humidity_derivative, previous_ground_temperature),
            -atmospheric_humidity,
        )
        latent_numerator = _multiply(
            _divide(_multiply(latent_heat, atmospheric_density), moisture_resistance),
            humidity_linearization,
        )
        conductive_numerator = _divide(
            _multiply(surface_conductivity, surface_layer_temperature),
            surface_layer_half_thickness,
        )
        surface_temperature_numerator = _sum(
            _multiply(surface_absorption_fraction, ground_shortwave),
            _multiply(jnp.float64(LAKE_EMISSIVITY), downward_longwave),
            _multiply(
                _multiply(jnp.float64(3.0), longwave_derivative_quarter),
                previous_ground_temperature,
            ),
            sensible_numerator,
            -latent_numerator,
            conductive_numerator,
        )
        surface_temperature_denominator = _sum(
            _multiply(jnp.float64(4.0), longwave_derivative_quarter),
            _divide(
                _multiply(atmospheric_density, jnp.float64(SPECIFIC_HEAT_DRY_AIR)),
                thermal_resistance,
            ),
            _multiply(
                _divide(_multiply(latent_heat, atmospheric_density), moisture_resistance),
                surface_humidity_derivative,
            ),
            _divide(surface_conductivity, surface_layer_half_thickness),
        )
        updated_ground_temperature = _divide(
            surface_temperature_numerator,
            surface_temperature_denominator,
        )
        ground_temperature = jnp.where(
            iteration_active,
            updated_ground_temperature,
            ground_temperature,
        )
        latent_heat = jnp.where(
            iteration_active,
            jnp.where(
                ground_temperature > jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
                jnp.float64(LATENT_HEAT_VAPORIZATION),
                jnp.float64(LATENT_HEAT_SUBLIMATION),
            ),
            latent_heat,
        )
        sensible_heat = _divide(
            _multiply(
                _multiply(atmospheric_density, jnp.float64(SPECIFIC_HEAT_DRY_AIR)),
                _sum(ground_temperature, -atmospheric_reference_temperature),
            ),
            thermal_resistance,
        )
        evaporation_inner = _sum(
            surface_humidity,
            _multiply(
                surface_humidity_derivative,
                _sum(ground_temperature, -previous_ground_temperature),
            ),
            -atmospheric_humidity,
        )
        evaporation = _divide(
            _multiply(atmospheric_density, evaporation_inner),
            moisture_resistance,
        )
        saturation = qsat(ground_temperature, atmospheric_pressure)
        surface_humidity = jnp.where(
            iteration_active,
            saturation.specific_humidity,
            surface_humidity,
        )
        surface_humidity_derivative = jnp.where(
            iteration_active,
            saturation.specific_humidity_temperature_derivative,
            surface_humidity_derivative,
        )
        temperature_difference = jnp.where(
            iteration_active,
            _sum(atmospheric_reference_temperature, -ground_temperature),
            _temperature_difference,
        )
        humidity_difference = jnp.where(
            iteration_active,
            _sum(atmospheric_humidity, -surface_humidity),
            _humidity_difference,
        )
        temperature_scale = _multiply(temperature_profile, temperature_difference)
        humidity_scale = _multiply(humidity_profile, humidity_difference)
        virtual_temperature_scale = _sum(
            _multiply(
                temperature_scale,
                _sum(
                    jnp.float64(1.0),
                    _multiply(jnp.float64(0.61), atmospheric_humidity),
                ),
            ),
            _multiply(
                _multiply(jnp.float64(0.61), atmospheric_theta),
                humidity_scale,
            ),
        )
        zeta = _divide(
            _multiply(
                _multiply(
                    _multiply(
                        forcing_height_wind,
                        jnp.float64(VON_KARMAN),
                    ),
                    jnp.float64(GRAVITY),
                ),
                virtual_temperature_scale,
            ),
            _multiply(
                _multiply(friction_value, friction_value),
                atmospheric_virtual_temperature,
            ),
        )
        stable = zeta >= jnp.float64(0.0)
        bounded_zeta = jnp.where(
            stable,
            jnp.minimum(
                jnp.asarray(maximum_stable_zeta, dtype=jnp.float64),
                jnp.maximum(zeta, jnp.float64(0.01)),
            ),
            jnp.maximum(jnp.float64(-100.0), jnp.minimum(zeta, jnp.float64(-0.01))),
        )
        convective_argument = _divide(
            -_multiply(
                _multiply(
                    _multiply(jnp.float64(GRAVITY), friction_value),
                    virtual_temperature_scale,
                ),
                jnp.float64(1000.0),
            ),
            atmospheric_virtual_temperature,
        )
        convective_base = jnp.where(
            stable,
            jnp.float64(1.0),
            jnp.maximum(convective_argument, jnp.float64(1.0e-300)),
        )
        convective_velocity = jnp.where(
            stable,
            jnp.float64(0.0),
            _power(convective_base, 0.333),
        )
        updated_stability_wind = jnp.where(
            stable,
            jnp.maximum(reference_wind, jnp.float64(0.1)),
            jnp.sqrt(
                _sum(
                    _multiply(reference_wind, reference_wind),
                    _multiply(convective_velocity, convective_velocity),
                )
            ),
        )
        updated_monin_length = _divide(forcing_height_wind, bounded_zeta)
        sign_changed = (
            _multiply(previous_monin_length, updated_monin_length) < jnp.float64(0.0)
        ) & iteration_active
        sign_change_count = sign_change_count + sign_changed.astype(jnp.int32)
        new_momentum, new_sensible, new_latent = _updated_roughness(
            ground_temperature,
            snow_layer_count,
            lake_depth,
            fetch,
            reference_wind,
            friction_value,
            atmospheric_pressure,
            momentum,
        )
        momentum = jnp.where(iteration_active, new_momentum, momentum)
        sensible = jnp.where(iteration_active, new_sensible, sensible)
        latent = jnp.where(iteration_active, new_latent, latent)
        stability_wind = jnp.where(
            iteration_active,
            updated_stability_wind,
            stability_wind,
        )
        monin_length = jnp.where(
            iteration_active,
            updated_monin_length,
            monin_length,
        )
        next_active = iteration_active & (sign_change_count < jnp.int32(3))
        iteration_state = (
            next_active,
            sign_change_count,
            monin_length,
            stability_wind,
            monin_length,
            momentum,
            sensible,
            latent,
            ground_temperature,
            surface_humidity,
            surface_humidity_derivative,
            latent_heat,
            friction_result.state,
            friction_value,
            temperature_profile,
            humidity_profile,
            temperature_profile_2m,
            humidity_profile_2m,
            momentum_resistance,
            thermal_resistance,
            temperature_difference,
            humidity_difference,
            sensible_heat,
            evaporation,
        )
        iteration_state = tree_util.tree_map(lax.optimization_barrier, iteration_state)
        iteration_diagnostics = (
            iteration_active,
            sign_change_count,
            ground_temperature,
            stability_wind,
            monin_length,
            friction_value,
            momentum_resistance,
            thermal_resistance,
            moisture_resistance,
            momentum,
            sensible,
            latent,
        )
        return iteration_state, iteration_diagnostics

    iteration_state, iteration_history = lax.scan(
        iteration_body,
        initial_iteration_state,
        jnp.arange(NUMBER_OF_STABILITY_ITERATIONS, dtype=jnp.int32),
    )

    (
        _iteration_active,
        _sign_change_count,
        _previous_monin_length,
        _stability_wind,
        _monin_length,
        momentum,
        sensible,
        latent,
        ground_temperature,
        surface_humidity,
        surface_humidity_derivative,
        latent_heat,
        friction_state,
        friction_value,
        temperature_profile,
        humidity_profile,
        temperature_profile_2m,
        humidity_profile_2m,
        momentum_resistance,
        thermal_resistance,
        temperature_difference,
        humidity_difference,
        sensible_heat,
        evaporation,
    ) = iteration_state
    moisture_resistance = iteration_history[8][-1]

    correction_source_temperature = ground_temperature
    frozen_or_snow = has_snow | (
        lake_temperature[:, 0] <= jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)
    )
    freeze_surface = frozen_or_snow & (
        ground_temperature > jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)
    )
    convective_surface = (
        (lake_temperature[:, 0] > ground_temperature)
        & (ground_temperature > jnp.float64(MAXIMUM_DENSITY_TEMPERATURE))
    ) | (
        (lake_temperature[:, 0] < ground_temperature)
        & (lake_temperature[:, 0] > jnp.float64(FREEZING_TEMPERATURE_FRESHWATER))
        & (ground_temperature < jnp.float64(MAXIMUM_DENSITY_TEMPERATURE))
    )
    corrected_ground_temperature = jnp.where(
        freeze_surface,
        jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
        jnp.where(convective_surface, lake_temperature[:, 0], ground_temperature),
    )
    corrected = freeze_surface | convective_surface
    corrected_sensible_heat = _divide(
        _multiply(
            _multiply(atmospheric_density, jnp.float64(SPECIFIC_HEAT_DRY_AIR)),
            _sum(corrected_ground_temperature, -atmospheric_reference_temperature),
        ),
        thermal_resistance,
    )
    corrected_evaporation = _divide(
        _multiply(
            atmospheric_density,
            _sum(
                surface_humidity,
                _multiply(
                    surface_humidity_derivative,
                    _sum(corrected_ground_temperature, -correction_source_temperature),
                ),
                -atmospheric_humidity,
            ),
        ),
        moisture_resistance,
    )
    ground_temperature = jnp.where(corrected, corrected_ground_temperature, ground_temperature)
    sensible_heat = jnp.where(corrected, corrected_sensible_heat, sensible_heat)
    evaporation = jnp.where(corrected, corrected_evaporation, evaporation)
    latent_heat = jnp.where(
        ground_temperature > jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
        jnp.float64(LATENT_HEAT_VAPORIZATION),
        jnp.float64(LATENT_HEAT_SUBLIMATION),
    )

    outgoing_longwave = _sum(
        _multiply(
            _sum(jnp.float64(1.0), -jnp.float64(LAKE_EMISSIVITY)),
            downward_longwave,
        ),
        _multiply(
            _multiply(
                jnp.float64(LAKE_EMISSIVITY),
                jnp.float64(STEFAN_BOLTZMANN),
            ),
            _power(ground_temperature, 4.0),
        ),
    )
    latent_flux = _multiply(latent_heat, evaporation)
    soil_ground_heat = _sum(
        ground_shortwave,
        downward_longwave,
        -outgoing_longwave,
        -sensible_heat,
        -latent_flux,
    )
    eastward_stress = -_divide(
        _multiply(atmospheric_density, eastward_wind),
        momentum_resistance,
    )
    northward_stress = -_divide(
        _multiply(atmospheric_density, northward_wind),
        momentum_resistance,
    )
    reference_temperature = _sum(
        atmospheric_reference_temperature,
        _multiply(
            _multiply(temperature_profile, temperature_difference),
            _sum(
                _divide(jnp.float64(1.0), temperature_profile_2m),
                -_divide(jnp.float64(1.0), temperature_profile),
            ),
        ),
    )
    reference_humidity = _sum(
        atmospheric_humidity,
        _multiply(
            _multiply(humidity_profile, humidity_difference),
            _sum(
                _divide(jnp.float64(1.0), humidity_profile_2m),
                -_divide(jnp.float64(1.0), humidity_profile),
            ),
        ),
    )
    reference_saturation = qsat(reference_temperature, atmospheric_pressure)
    reference_relative_humidity = jnp.minimum(
        jnp.float64(100.0),
        _multiply(
            _divide(reference_humidity, reference_saturation.specific_humidity),
            jnp.float64(100.0),
        ),
    )
    human_index = fast_human_stress_indices(
        reference_temperature,
        reference_relative_humidity,
        reference_saturation.vapor_pressure,
        friction_state.wind_10m_patch,
    )
    ground_net = _sum(
        _sum(
            _multiply(surface_absorption_fraction, ground_shortwave),
            downward_longwave,
        ),
        -_sum(outgoing_longwave, sensible_heat, latent_flux),
    )
    wind_two_meters = jnp.maximum(
        jnp.float64(0.1),
        _multiply(
            _divide(friction_value, jnp.float64(VON_KARMAN)),
            _scalar_log(_divide(jnp.float64(2.0), momentum)),
        ),
    )
    wind_mixing_velocity = _multiply(jnp.float64(1.2e-3), wind_two_meters)
    mixing_decay = _mixing_decay(wind_two_meters, vector("latitude_radians", 0.5))

    def retain(value: jnp.ndarray, previous: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(active, value, jnp.asarray(previous, dtype=jnp.float64))

    updated_state = LakeFluxState(
        ground_temperature=retain(ground_temperature, state.ground_temperature),
        momentum_roughness=retain(momentum, state.momentum_roughness),
        sensible_roughness=retain(sensible, state.sensible_roughness),
        latent_roughness=retain(latent, state.latent_roughness),
        lake_friction_velocity=retain(friction_value, state.lake_friction_velocity),
        friction_velocity=friction_state,
    )

    def active_value(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(active, value, jnp.zeros_like(value))

    diagnostics = LakeFluxDiagnostics(*iteration_history)
    return LakeFluxResult(
        state=updated_state,
        energy=LakeEnergyFluxes(
            *(
                active_value(value)
                for value in (
                    outgoing_longwave,
                    _sum(outgoing_longwave, -downward_longwave),
                    soil_ground_heat,
                    sensible_heat,
                    sensible_heat,
                    latent_flux,
                    latent_flux,
                    ground_net,
                    eastward_stress,
                    northward_stress,
                )
            )
        ),
        water=LakeWaterFluxes(
            *(
                active_value(value)
                for value in (
                    evaporation,
                    evaporation,
                    _sum(vector("rain_flux", 0.0), vector("snow_flux", 0.0)),
                    reference_humidity,
                    reference_relative_humidity,
                )
            )
        ),
        temperature=LakeTemperatureState(
            ground_temperature=active_value(ground_temperature),
            vegetation_temperature=active_value(atmospheric_temperature),
            reference_temperature=active_value(reference_temperature),
            skin_temperature=active_value(atmospheric_temperature),
        ),
        human_index=FastHumanStressIndices(*(active_value(value) for value in human_index)),
        surface=LakeSurfaceState(
            surface_absorption_fraction=active_value(surface_absorption_fraction),
            checked_ground_absorbed_shortwave=active_value(ground_shortwave),
            aerodynamic_resistance=active_value(momentum_resistance),
            lake_aerodynamic_resistance=active_value(momentum_resistance),
            moisture_resistance=active_value(moisture_resistance),
            wind_mixing_velocity=active_value(wind_mixing_velocity),
            mixing_decay_coefficient=active_value(mixing_decay),
            forcing_height_wind=retain(forcing_height_wind, inputs.forcing_height_wind),
            forcing_height_temperature=retain(
                forcing_height_temperature,
                inputs.forcing_height_temperature,
            ),
            forcing_height_humidity=retain(
                forcing_height_humidity,
                inputs.forcing_height_humidity,
            ),
        ),
        diagnostics=diagnostics,
    )
