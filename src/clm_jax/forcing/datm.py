
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from clm_jax.constants import (
    FREEZING_TEMPERATURE_FRESHWATER,
    GAS_CONSTANT_DRY_AIR,
)
from clm_jax.floating_point import correctly_rounded_divide


def _add_step(running, term):
    return running + term, None


def _multiply_step(running, factor):
    return running * factor, None


def _fortran_add(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:

    left, right = jnp.broadcast_arrays(
        jnp.asarray(left, dtype=jnp.float64),
        jnp.asarray(right, dtype=jnp.float64),
    )
    terms = jnp.stack((right, jnp.zeros_like(right)), axis=0)
    result, _ = lax.scan(_add_step, left, terms)
    return lax.optimization_barrier(result)


def _fortran_multiply(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:

    left, right = jnp.broadcast_arrays(
        jnp.asarray(left, dtype=jnp.float64),
        jnp.asarray(right, dtype=jnp.float64),
    )
    factors = jnp.stack((right, jnp.ones_like(right)), axis=0)
    result, _ = lax.scan(_multiply_step, left, factors)
    return lax.optimization_barrier(result)


def _fortran_divide(numerator: jnp.ndarray, denominator: jnp.ndarray) -> jnp.ndarray:

    return correctly_rounded_divide(numerator, denominator)


class DatmForcing(NamedTuple):
    reference_height: jnp.ndarray
    temperature: jnp.ndarray
    potential_temperature: jnp.ndarray
    specific_humidity: jnp.ndarray
    density: jnp.ndarray
    pressure: jnp.ndarray
    longwave_down: jnp.ndarray
    wind_u: jnp.ndarray
    wind_v: jnp.ndarray
    shortwave_nir_direct: jnp.ndarray
    shortwave_nir_diffuse: jnp.ndarray
    shortwave_visible_direct: jnp.ndarray
    shortwave_visible_diffuse: jnp.ndarray
    rain_convective: jnp.ndarray
    rain_large_scale: jnp.ndarray
    snow_convective: jnp.ndarray
    snow_large_scale: jnp.ndarray

    @property
    def wind_speed(self) -> jnp.ndarray:
        return jnp.sqrt(self.wind_u**2 + self.wind_v**2)

    @property
    def relative_humidity(self) -> jnp.ndarray:

        return clm_relative_humidity(
            self.temperature,
            self.specific_humidity,
            self.pressure,
        )

    @property
    def shortwave_down(self) -> jnp.ndarray:
        return (
            self.shortwave_nir_direct
            + self.shortwave_nir_diffuse
            + self.shortwave_visible_direct
            + self.shortwave_visible_diffuse
        )

    @property
    def rain(self) -> jnp.ndarray:
        return self.rain_convective + self.rain_large_scale

    @property
    def snow(self) -> jnp.ndarray:
        return self.snow_convective + self.snow_large_scale


class ClmncepShortwave(NamedTuple):

    nir_direct: jnp.ndarray
    nir_diffuse: jnp.ndarray
    visible_direct: jnp.ndarray
    visible_diffuse: jnp.ndarray


def datm_saturation_vapor_pressure(
    polynomial_temperature: jnp.ndarray,
    bottom_temperature: jnp.ndarray,
) -> jnp.ndarray:

    temperature = jnp.asarray(polynomial_temperature, dtype=jnp.float64)
    bottom_temperature = jnp.asarray(bottom_temperature, dtype=jnp.float64)
    celsius = jnp.clip(temperature - FREEZING_TEMPERATURE_FRESHWATER, -50.0, 50.0)

    water = jnp.float64(6.136820929e-11)
    for coefficient in (
        2.034080948e-08,
        3.031240396e-06,
        2.650648471e-04,
        1.428945805e-02,
        4.436518521e-01,
        6.107799961,
    ):
        water = jnp.float64(coefficient) + celsius * water

    ice = jnp.float64(1.838826904e-10)
    for coefficient in (
        4.838803174e-08,
        5.824720280e-06,
        4.176223716e-04,
        1.886013408e-02,
        5.034698970e-01,
        6.109177956,
    ):
        ice = jnp.float64(coefficient) + celsius * ice

    return jnp.float64(100.0) * jnp.where(
        bottom_temperature < FREEZING_TEMPERATURE_FRESHWATER,
        ice,
        water,
    )


def clm_relative_humidity(
    temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
) -> jnp.ndarray:

    vapor_pressure = datm_saturation_vapor_pressure(temperature, temperature)
    saturation_humidity = _fortran_divide(
        _fortran_multiply(jnp.float64(0.622), vapor_pressure),
        _fortran_add(
            pressure,
            -_fortran_multiply(jnp.float64(0.378), vapor_pressure),
        ),
    )
    return _fortran_multiply(
        jnp.float64(100.0),
        _fortran_divide(specific_humidity, saturation_humidity),
    )


def datm_rain_fraction(temperature: jnp.ndarray) -> jnp.ndarray:

    temperature = jnp.asarray(temperature, dtype=jnp.float64)
    return jnp.clip(
        (temperature - FREEZING_TEMPERATURE_FRESHWATER) * jnp.float64(0.5),
        0.0,
        1.0,
    )


def clmncep_shortwave_partition(shortwave_down: jnp.ndarray) -> ClmncepShortwave:
    from clm_jax.physics.numerics import (
        fortran_hardware_add as add, fortran_hardware_multiply as multiply,
    )

    half = multiply(jnp.asarray(shortwave_down, jnp.float64), jnp.float64(.5))
    square = multiply(half, half)
    cube = multiply(square, half)

    def fraction(offset, linear, quadratic, cubic):
        value = add(jnp.float64(offset), multiply(jnp.float64(linear), half))
        value = add(value, -multiply(jnp.float64(quadratic), square))
        value = add(value, multiply(jnp.float64(cubic), cube))
        return jnp.clip(value, .01, .99)

    nir = fraction(.29548, .00504, 1.4957e-5, 1.4881e-8)
    visible = fraction(.17639, .00380, 9.0039e-6, 8.1351e-9)
    return ClmncepShortwave(
        nir_direct=multiply(nir, half),
        nir_diffuse=multiply(add(jnp.float64(1.), -nir), half),
        visible_direct=multiply(visible, half),
        visible_diffuse=multiply(add(jnp.float64(1.), -visible), half),
    )


native_clmncep_shortwave_partition = clmncep_shortwave_partition


def clmncep_forcing(
    temperature: jnp.ndarray,
    wind_speed: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    longwave_down: jnp.ndarray,
    shortwave_down: jnp.ndarray,
    precipitation: jnp.ndarray,
) -> DatmForcing:

    temperature = jnp.maximum(jnp.float64(180.0), jnp.asarray(temperature, jnp.float64))
    wind_speed = jnp.asarray(wind_speed, dtype=jnp.float64)
    pressure = jnp.asarray(pressure, dtype=jnp.float64)
    longwave_down = jnp.asarray(longwave_down, dtype=jnp.float64)
    precipitation = jnp.asarray(precipitation, dtype=jnp.float64)

    saturation_pressure = datm_saturation_vapor_pressure(temperature, temperature)
    saturation_humidity = _fortran_divide(
        _fortran_multiply(jnp.float64(0.622), saturation_pressure),
        _fortran_add(
            pressure,
            -_fortran_multiply(jnp.float64(0.378), saturation_pressure),
        ),
    )
    specific_humidity = jnp.minimum(
        jnp.asarray(specific_humidity, dtype=jnp.float64), saturation_humidity
    )
    vapor_pressure = _fortran_divide(
        _fortran_multiply(specific_humidity, pressure),
        _fortran_add(
            jnp.float64(0.622),
            _fortran_multiply(jnp.float64(0.378), specific_humidity),
        ),
    )
    density = _fortran_divide(
        _fortran_add(
            pressure,
            -_fortran_multiply(jnp.float64(0.378), vapor_pressure),
        ),
        _fortran_multiply(temperature, jnp.float64(GAS_CONSTANT_DRY_AIR)),
    )

    wind_component = _fortran_divide(wind_speed, jnp.sqrt(jnp.float64(2.0)))
    shortwave = clmncep_shortwave_partition(shortwave_down)

    rain_fraction = datm_rain_fraction(temperature)
    rain_convective_unpartitioned = precipitation * jnp.float64(0.1)
    rain_large_scale_unpartitioned = precipitation * jnp.float64(0.9)
    snow_fraction = jnp.float64(1.0) - rain_fraction

    return DatmForcing(
        reference_height=jnp.full_like(temperature, jnp.float64(30.0)),
        temperature=temperature,
        potential_temperature=temperature,
        specific_humidity=specific_humidity,
        density=density,
        pressure=pressure,
        longwave_down=longwave_down,
        wind_u=wind_component,
        wind_v=wind_component,
        shortwave_nir_direct=shortwave.nir_direct,
        shortwave_nir_diffuse=shortwave.nir_diffuse,
        shortwave_visible_direct=shortwave.visible_direct,
        shortwave_visible_diffuse=shortwave.visible_diffuse,
        rain_convective=jnp.maximum(0.0, rain_convective_unpartitioned * rain_fraction),
        rain_large_scale=jnp.maximum(0.0, rain_large_scale_unpartitioned * rain_fraction),
        snow_convective=jnp.maximum(0.0, rain_convective_unpartitioned * snow_fraction),
        snow_large_scale=jnp.maximum(0.0, rain_large_scale_unpartitioned * snow_fraction),
    )
