
from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.physics.numerics import (
    _LN2_HIGH,
    _LN2_LOW,
    _double_double_add,
    _double_double_multiply,
    _lookup_double_double_table,
    _quick_two_sum,
    _refined_divide,
    _rounded_add,
    _strict_exp_reference,
    _transcendental_primal,
    _two_sum,
)


def _log_coefficients():
    with localcontext() as context:
        context.prec = 100
        values = [Decimal(1) / Decimal(2*k + 1) for k in range(5)]
        return tuple((float(value), float(value - Decimal.from_float(float(value))))
                     for value in values)


_LOG_COEFFICIENTS = _log_coefficients()


def _center_logarithms():
    with localcontext() as context:
        context.prec = 100
        values = [(Decimal(index) / 256).ln() for index in range(128, 385)]
        return tuple((float(value), float(value - Decimal.from_float(float(value))))
                     for value in values)


_CENTER_LOGARITHMS = _center_logarithms()


@_transcendental_primal
def _positive_power_reference(base, exponent):
    base, exponent = jnp.broadcast_arrays(jnp.asarray(base, jnp.float64),
                                         jnp.asarray(exponent, jnp.float64))
    valid = (base > 0) & jnp.isfinite(base) & jnp.isfinite(exponent)
    mantissa, scale = jnp.frexp(jnp.where(valid, base, 1.0))
    below = mantissa < jnp.float64(0.7071067811865476)
    mantissa = jnp.where(below, mantissa * 2, mantissa)
    scale = scale - below.astype(jnp.int32)
    zero = jnp.zeros_like(base)
    center_index = jnp.floor(mantissa * 256.0 + 0.5).astype(jnp.int32)
    center = center_index.astype(jnp.float64) * jnp.float64(1.0 / 256.0)
    center_log = _lookup_double_double_table(_CENTER_LOGARITHMS, center_index - 128)
    numerator = _rounded_add(mantissa, -center)
    denominator = _two_sum(mantissa, center)
    quotient = _refined_divide(numerator, denominator[0])
    product = _double_double_multiply((quotient, zero), denominator)
    remainder = _double_double_add((numerator, zero), (-product[0], -product[1]))
    ratio = _quick_two_sum(quotient, _refined_divide(
        _rounded_add(remainder[0], remainder[1]), denominator[0]))
    squared = _double_double_multiply(ratio, ratio)
    initial = tuple(jnp.full_like(base, value) for value in _LOG_COEFFICIENTS[-1])
    from clm_jax.physics.numerics import _ordered_compensated_horner
    polynomial = _ordered_compensated_horner(
        initial, squared, _LOG_COEFFICIENTS[:-1][::-1],
        unroll=jax.default_backend() == "gpu",
    )
    logarithm = _double_double_multiply(polynomial, ratio)
    logarithm = _double_double_add(logarithm, logarithm)
    logarithm = _double_double_add(center_log, logarithm)
    binary = _double_double_multiply((scale.astype(jnp.float64), zero),
                                     (jnp.full_like(base, _LN2_HIGH),
                                      jnp.full_like(base, _LN2_LOW)))
    logarithm = _double_double_add(binary, logarithm)
    argument = _double_double_multiply(logarithm, (exponent, zero))
    finite_result = valid & (argument[0] > -745.0) & (argument[0] < 709.0)
    result = _strict_exp_reference(jnp.where(finite_result, argument[0], 0.0),
                                   jnp.where(finite_result, argument[1], 0.0))
    return jnp.where(finite_result, result, jnp.power(base, exponent))


@jax.custom_jvp
def positive_power(base, exponent):
    base, exponent = jnp.broadcast_arrays(jnp.asarray(base, jnp.float64),
                                         jnp.asarray(exponent, jnp.float64))
    if jax.default_backend() == "gpu":
        from clm_jax.physics.solver_scalarization import _native_tile_trace
        if not _native_tile_trace.get():
            from clm_jax.physics.positive_power_gpu import positive_power_gpu
            return positive_power_gpu(base, exponent)
    return _positive_power_reference(base, exponent)


@positive_power.defjvp
def _positive_power_jvp(primals, tangents):
    return positive_power(*primals), jax.jvp(jnp.power, primals, tangents)[1]
