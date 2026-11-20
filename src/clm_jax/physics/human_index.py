
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import default_backend, lax

from clm_jax.constants import FREEZING_TEMPERATURE_FRESHWATER
from clm_jax.physics.numerics import fortran_divide, fortran_ordered_sum, fortran_product
from clm_jax.physics.pointwise import scalar_map


class FastHumanStressIndices(NamedTuple):
    temperature_celsius: jnp.ndarray
    vapor_pressure: jnp.ndarray
    wet_bulb_stull: jnp.ndarray
    heat_index: jnp.ndarray
    apparent_temperature: jnp.ndarray
    simplified_wet_bulb_globe_temperature: jnp.ndarray
    humidex: jnp.ndarray
    discomfort_index_stull: jnp.ndarray


def _scalar_atan(value: jnp.ndarray) -> jnp.ndarray:

    if default_backend() == "gpu":
        from clm_jax.physics.native_arctangent import native_atan
        return native_atan(value)

    shape = value.shape
    flat = jnp.reshape(value, (-1,))
    return jnp.reshape(scalar_map(jnp.arctan, flat), shape)


def _add(*terms: jnp.ndarray) -> jnp.ndarray:
    broadcast = jnp.broadcast_arrays(*(jnp.asarray(term, dtype=jnp.float64) for term in terms))
    return lax.optimization_barrier(fortran_ordered_sum(jnp.stack(broadcast, axis=0)))


def _multiply(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    left, right = jnp.broadcast_arrays(
        jnp.asarray(left, dtype=jnp.float64),
        jnp.asarray(right, dtype=jnp.float64),
    )
    return lax.optimization_barrier(fortran_product(left, right))


def _multiply_boundary(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    left, right = jnp.broadcast_arrays(
        jnp.asarray(left, dtype=jnp.float64),
        jnp.asarray(right, dtype=jnp.float64),
    )
    factors = jnp.stack((right, jnp.ones_like(right)), axis=0)

    def multiply_step(running: jnp.ndarray, factor: jnp.ndarray):
        return running * factor, None

    result, _ = lax.scan(multiply_step, left, factors)
    return lax.optimization_barrier(result)


def _divide(numerator: jnp.ndarray, denominator: jnp.ndarray) -> jnp.ndarray:
    return fortran_divide(numerator, denominator)


def kelvin_to_celsius(temperature: jnp.ndarray) -> jnp.ndarray:
    return _add(temperature, -jnp.float64(FREEZING_TEMPERATURE_FRESHWATER))


def vapor_pressure(
    relative_humidity: jnp.ndarray,
    saturated_vapor_pressure: jnp.ndarray,
) -> jnp.ndarray:
    humidity, saturation = jnp.broadcast_arrays(
        jnp.asarray(relative_humidity, dtype=jnp.float64),
        jnp.asarray(saturated_vapor_pressure, dtype=jnp.float64),
    )
    return _multiply_boundary(_divide(humidity, jnp.float64(100.0)), saturation)


def wet_bulb_stull(
    temperature_celsius: jnp.ndarray,
    relative_humidity: jnp.ndarray,
) -> jnp.ndarray:

    temperature, humidity = jnp.broadcast_arrays(
        jnp.asarray(temperature_celsius, dtype=jnp.float64),
        jnp.asarray(relative_humidity, dtype=jnp.float64),
    )
    first_angle = _scalar_atan(
        _multiply(jnp.float64(0.151977), jnp.sqrt(_add(humidity, jnp.float64(8.313659))))
    )
    second_angle = _scalar_atan(_add(temperature, humidity))
    third_angle = _scalar_atan(_add(humidity, jnp.float64(-1.676331)))
    fourth_angle = _scalar_atan(_multiply(jnp.float64(0.023101), humidity))
    if default_backend() == "gpu":
        from clm_jax.physics.native_power import native_power
        humidity_power = native_power(humidity, jnp.float64(1.5))
    else:
        humidity_power = humidity ** jnp.float64(1.5)
    return _add(
        _multiply(temperature, first_angle),
        second_angle,
        -third_angle,
        _multiply(_multiply(jnp.float64(0.00391838), humidity_power), fourth_angle),
        jnp.float64(-4.686035),
    )


def heat_index(
    temperature_celsius: jnp.ndarray,
    relative_humidity: jnp.ndarray,
) -> jnp.ndarray:

    temperature, humidity = jnp.broadcast_arrays(
        jnp.asarray(temperature_celsius, dtype=jnp.float64),
        jnp.asarray(relative_humidity, dtype=jnp.float64),
    )
    temperature_fahrenheit = _add(
        _divide(_multiply_boundary(temperature, jnp.float64(9.0)), jnp.float64(5.0)),
        jnp.float64(32.0),
    )
    temperature_squared = temperature_fahrenheit**jnp.float64(2.0)
    humidity_squared = humidity**jnp.float64(2.0)
    polynomial = _add(
        jnp.float64(-42.379),
        _multiply(jnp.float64(2.04901523), temperature_fahrenheit),
        _multiply(jnp.float64(10.14333127), humidity),
        -_multiply(_multiply(jnp.float64(0.22475541), temperature_fahrenheit), humidity),
        -_multiply(jnp.float64(6.83783e-3), temperature_squared),
        -_multiply(jnp.float64(5.481717e-2), humidity_squared),
        _multiply(_multiply(jnp.float64(1.22874e-3), temperature_squared), humidity),
        _multiply(_multiply(jnp.float64(8.5282e-4), temperature_fahrenheit), humidity_squared),
        -_multiply(
            _multiply(jnp.float64(1.99e-6), temperature_squared),
            humidity_squared,
        ),
    )
    fahrenheit_index = jnp.where(
        temperature_fahrenheit < jnp.float64(68.0),
        temperature_fahrenheit,
        polynomial,
    )
    return _divide(
        _multiply_boundary(
            _add(fahrenheit_index, jnp.float64(-32.0)),
            jnp.float64(5.0),
        ),
        jnp.float64(9.0),
    )


def apparent_temperature(
    temperature_celsius: jnp.ndarray,
    vapor_pressure_pascal: jnp.ndarray,
    wind_10m: jnp.ndarray,
) -> jnp.ndarray:
    temperature, pressure, wind = jnp.broadcast_arrays(
        jnp.asarray(temperature_celsius, dtype=jnp.float64),
        jnp.asarray(vapor_pressure_pascal, dtype=jnp.float64),
        jnp.asarray(wind_10m, dtype=jnp.float64),
    )
    return _add(
        temperature,
        _divide(
            _multiply_boundary(jnp.float64(3.30), pressure),
            jnp.float64(1000.0),
        ),
        -_multiply_boundary(jnp.float64(0.70), wind),
        jnp.float64(-4.0),
    )


def simplified_wet_bulb_globe_temperature(
    temperature_celsius: jnp.ndarray,
    vapor_pressure_pascal: jnp.ndarray,
) -> jnp.ndarray:
    temperature, pressure = jnp.broadcast_arrays(
        jnp.asarray(temperature_celsius, dtype=jnp.float64),
        jnp.asarray(vapor_pressure_pascal, dtype=jnp.float64),
    )
    return _add(
        _multiply(jnp.float64(0.567), temperature),
        _divide(_multiply(jnp.float64(0.393), pressure), jnp.float64(100.0)),
        jnp.float64(3.94),
    )


def humidex(
    temperature_celsius: jnp.ndarray,
    vapor_pressure_pascal: jnp.ndarray,
) -> jnp.ndarray:
    temperature, pressure = jnp.broadcast_arrays(
        jnp.asarray(temperature_celsius, dtype=jnp.float64),
        jnp.asarray(vapor_pressure_pascal, dtype=jnp.float64),
    )
    correction = _multiply_boundary(
        _divide(jnp.float64(5.0), jnp.float64(9.0)),
        _add(_divide(pressure, jnp.float64(100.0)), jnp.float64(-10.0)),
    )
    return _add(temperature, correction)


def discomfort_index_stull(
    temperature_celsius: jnp.ndarray,
    relative_humidity: jnp.ndarray,
    wet_bulb_temperature: jnp.ndarray,
) -> jnp.ndarray:
    temperature, humidity, wet_bulb = jnp.broadcast_arrays(
        jnp.asarray(temperature_celsius, dtype=jnp.float64),
        jnp.asarray(relative_humidity, dtype=jnp.float64),
        jnp.asarray(wet_bulb_temperature, dtype=jnp.float64),
    )
    limited_temperature = jnp.minimum(temperature, jnp.float64(50.0))
    limited_humidity = jnp.maximum(
        jnp.minimum(humidity, jnp.float64(99.0)),
        jnp.float64(5.0),
    )
    minimum_humidity = _add(
        _multiply(limited_temperature, jnp.float64(-2.27)),
        jnp.float64(27.7),
    )
    valid = (limited_temperature >= jnp.float64(-20.0)) & (
        limited_humidity >= minimum_humidity
    )
    valid_index = _add(
        _multiply(jnp.float64(0.5), wet_bulb),
        _multiply(jnp.float64(0.5), limited_temperature),
    )
    return jnp.where(valid, valid_index, limited_temperature)


def fast_human_stress_indices(
    reference_temperature_kelvin: jnp.ndarray,
    relative_humidity: jnp.ndarray,
    saturated_vapor_pressure: jnp.ndarray,
    wind_10m: jnp.ndarray,
) -> FastHumanStressIndices:

    temperature = kelvin_to_celsius(reference_temperature_kelvin)
    pressure = vapor_pressure(relative_humidity, saturated_vapor_pressure)
    wet_bulb = wet_bulb_stull(temperature, relative_humidity)
    nws_heat_index = heat_index(temperature, relative_humidity)
    apparent = apparent_temperature(temperature, pressure, wind_10m)
    wet_bulb_globe = simplified_wet_bulb_globe_temperature(temperature, pressure)
    humidex_value = humidex(temperature, pressure)
    discomfort = discomfort_index_stull(temperature, relative_humidity, wet_bulb)
    return FastHumanStressIndices(
        temperature,
        pressure,
        wet_bulb,
        nws_heat_index,
        apparent,
        wet_bulb_globe,
        humidex_value,
        discomfort,
    )
