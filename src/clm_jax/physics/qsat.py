
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
from jax import custom_jvp, default_backend, jvp, lax

from clm_jax.constants import FREEZING_TEMPERATURE_FRESHWATER, GAS_CONSTANT_WATER_VAPOR
from clm_jax.physics.numerics import (
    fortran_multiply,
    fortran_ordered_sum,
)


class QSatResult(NamedTuple):
    vapor_pressure: jnp.ndarray
    vapor_pressure_temperature_derivative: jnp.ndarray
    specific_humidity: jnp.ndarray
    specific_humidity_temperature_derivative: jnp.ndarray


_WATER_PRESSURE = (
    6.11213476,
    0.444007856,
    0.143064234e-01,
    0.264461437e-03,
    0.305903558e-05,
    0.196237241e-07,
    0.892344772e-10,
    -0.373208410e-12,
    0.209339997e-15,
)
_WATER_DERIVATIVE = (
    0.444017302,
    0.286064092e-01,
    0.794683137e-03,
    0.121211669e-04,
    0.103354611e-06,
    0.404125005e-09,
    -0.788037859e-12,
    -0.114596802e-13,
    0.381294516e-16,
)
_ICE_PRESSURE = (
    6.11123516,
    0.503109514,
    0.188369801e-01,
    0.420547422e-03,
    0.614396778e-05,
    0.602780717e-07,
    0.387940929e-09,
    0.149436277e-11,
    0.262655803e-14,
)
_ICE_DERIVATIVE = (
    0.503277922,
    0.377289173e-01,
    0.126801703e-02,
    0.249468427e-04,
    0.313703411e-06,
    0.257180651e-08,
    0.133268878e-10,
    0.394116744e-13,
    0.498070196e-16,
)


def _multiply_scan(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    left, right = jnp.broadcast_arrays(left, right)
    factors = jnp.stack((right, jnp.ones_like(right)), axis=0)

    def multiply_step(running: jnp.ndarray, factor: jnp.ndarray):
        return running * factor, None

    result, _ = lax.scan(multiply_step, left, factors)
    return result


def _multiply(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    if default_backend() == "gpu":
        return fortran_multiply(left, right)
    return _multiply_scan(left, right)


def _add(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    left, right = jnp.broadcast_arrays(left, right)
    return lax.optimization_barrier(fortran_ordered_sum(jnp.stack((left, right), axis=0)))


def _horner_loop(
    value: jnp.ndarray,
    coefficients: tuple[float, ...],
    *,
    unroll=False,
    native_arithmetic=False,
) -> jnp.ndarray:
    result = jnp.asarray(coefficients[-1], dtype=jnp.float64) + jnp.zeros_like(value)
    remaining = jnp.asarray(coefficients[-2::-1], dtype=jnp.float64)

    def horner_step(running: jnp.ndarray, coefficient: jnp.ndarray):
        if native_arithmetic:
            product = _multiply_scan(value, running)
            updated, _ = lax.scan(
                lambda carry, term: (carry + term, None),
                coefficient + jnp.zeros_like(value),
                product[None],
            )
        else:
            updated = _add(coefficient + jnp.zeros_like(value), _multiply(value, running))
        return updated, None

    result, _ = lax.scan(horner_step, result, remaining, unroll=unroll)
    return result


@partial(custom_jvp, nondiff_argnums=(1,))
def _gpu_horner(value, coefficients):
    return _horner_loop(value, coefficients, unroll=True)


@_gpu_horner.defjvp
def _gpu_horner_jvp(coefficients, primals, tangents):
    return jvp(
        lambda value: _horner_loop(value, coefficients, native_arithmetic=True),
        primals,
        tangents,
    )


def _horner(value: jnp.ndarray, coefficients: tuple[float, ...]) -> jnp.ndarray:
    if default_backend() == "gpu":
        return _gpu_horner(value, coefficients)
    return _horner_loop(value, coefficients)


def _vapor_pressure_polynomials(
    temperature: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    temperature_celsius = jnp.clip(
        jnp.asarray(temperature, dtype=jnp.float64) - FREEZING_TEMPERATURE_FRESHWATER,
        min=jnp.float64(-75.0),
        max=jnp.float64(100.0),
    )
    liquid = temperature_celsius >= 0.0
    pressure_hpa = jnp.where(
        liquid,
        _horner(temperature_celsius, _WATER_PRESSURE),
        _horner(temperature_celsius, _ICE_PRESSURE),
    )
    derivative_hpa_per_kelvin = jnp.where(
        liquid,
        _horner(temperature_celsius, _WATER_DERIVATIVE),
        _horner(temperature_celsius, _ICE_DERIVATIVE),
    )
    return temperature_celsius, pressure_hpa, derivative_hpa_per_kelvin


def qsat(temperature: jnp.ndarray, pressure: jnp.ndarray) -> QSatResult:

    _, pressure_hpa, derivative_hpa_per_kelvin = _vapor_pressure_polynomials(temperature)
    vapor_pressure = _multiply(pressure_hpa, jnp.float64(100.0))
    vapor_pressure_derivative = _multiply(derivative_hpa_per_kelvin, jnp.float64(100.0))
    atmospheric_pressure = jnp.asarray(pressure, dtype=jnp.float64)

    inverse_denominator = jnp.float64(1.0) / (
        _add(
            atmospheric_pressure,
            -_multiply(jnp.float64(0.378), vapor_pressure),
        )
    )
    vapor_pressure_factor = _multiply(jnp.float64(0.622), inverse_denominator)
    derivative_factor = _multiply(vapor_pressure_factor, inverse_denominator)
    specific_humidity = _multiply(vapor_pressure, vapor_pressure_factor)
    specific_humidity_derivative = _multiply(
        _multiply(vapor_pressure_derivative, derivative_factor),
        atmospheric_pressure,
    )
    return QSatResult(
        vapor_pressure,
        vapor_pressure_derivative,
        specific_humidity,
        specific_humidity_derivative,
    )


def rho_saturated(temperature: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:

    temperature = jnp.asarray(temperature, dtype=jnp.float64)
    _, pressure_hpa, derivative_hpa_per_kelvin = _vapor_pressure_polynomials(temperature)
    vapor_pressure = _multiply(pressure_hpa, jnp.float64(100.0))
    gas_temperature = _multiply(GAS_CONSTANT_WATER_VAPOR, temperature)
    density = vapor_pressure / gas_temperature

    density_derivative = _add(
        derivative_hpa_per_kelvin / gas_temperature,
        -(density / temperature),
    )
    return density, density_derivative
