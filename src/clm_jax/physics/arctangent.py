
from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.physics.numerics import (
    _double_double_add,
    _double_double_multiply,
    _quick_two_sum,
    _refined_divide,
    _rounded_add,
    _two_sum,
)


def _coefficients():
    with localcontext() as context:
        context.prec = 100

        def atan_series(x):
            term, total = x, x
            for k in range(1, 512):
                term *= -x*x
                updated = total + term / Decimal(2*k+1)
                if updated == total:
                    return total
                total = updated
            raise ArithmeticError("arctangent constant did not converge")

        quarter_pi = 4*atan_series(Decimal(1)/5) - atan_series(Decimal(1)/239)
        angles = []
        for k in range(17):
            x = Decimal(k)/16
            angle = (atan_series(x) if x <= Decimal('0.5') else
                     quarter_pi + atan_series((x-1)/(x+1)))
            angles.append(angle)
        pair = lambda x: (float(x), float(x-Decimal.from_float(float(x))))
        values = [Decimal((-1)**k) / Decimal(2*k + 1) for k in range(12)]
        return tuple(map(pair, values)), tuple(map(pair, angles))


_COEFFICIENTS, _ANGLES = _coefficients()
_PI_OVER_FOUR = (0.7853981633974483, 3.061616997868383e-17)


def _arctangent_reference(value):
    value = jnp.asarray(value, jnp.float64)
    x = jnp.abs(value)
    finite = jnp.isfinite(x)
    asymptotic = x >= jnp.float64(2.0**54)
    safe = jnp.where(finite & ~asymptotic, x, 1.)
    large = safe > 1.
    zero, one = jnp.zeros_like(x), jnp.ones_like(x)
    reduced = jnp.where(large, _refined_divide(one, jnp.maximum(safe, 1.)), safe)
    index = jnp.floor(reduced*16 + .5).astype(jnp.int32)
    center = index.astype(jnp.float64)*.0625
    product = _double_double_multiply((safe, zero), (center, zero))
    difference = _two_sum(safe, -center)
    inverse_difference = _double_double_add((one, zero), (-product[0], -product[1]))
    total = _double_double_add((one, zero), product)
    inverse_total = _two_sum(safe, center)
    numerator = tuple(jnp.where(large, b, a) for a, b in zip(difference, inverse_difference))
    denominator = tuple(jnp.where(large, b, a) for a, b in zip(total, inverse_total))
    quotient = _refined_divide(numerator[0], denominator[0])
    product = _double_double_multiply((quotient, zero), denominator)
    remainder = _double_double_add(numerator, (-product[0], -product[1]))
    ratio = _quick_two_sum(quotient, _refined_divide(
        _rounded_add(remainder[0], remainder[1]), denominator[0]))
    squared = _double_double_multiply(ratio, ratio)
    initial = tuple(jnp.full_like(x, c) for c in _COEFFICIENTS[-1])

    def horner(total, coefficient):
        return _double_double_add(_double_double_multiply(total, squared),
                                  (coefficient[0], coefficient[1])), None

    polynomial, _ = lax.scan(horner, initial,
                              jnp.asarray(_COEFFICIENTS[:-1][::-1], jnp.float64),
                              unroll=jax.default_backend() == "gpu")
    result = _double_double_multiply(polynomial, ratio)
    angle = tuple(jnp.asarray([row[k] for row in _ANGLES], jnp.float64)[index]
                  for k in (0, 1))
    result = _double_double_add(result, angle)
    reflected = _double_double_add(tuple(2*c for c in _PI_OVER_FOUR),
                                   (-result[0], -result[1]))
    result = tuple(jnp.where(large, b, a) for a, b in zip(result, reflected))
    answer = _rounded_add(*result)
    answer = jnp.where(asymptotic, jnp.float64(1.5707963267948966), answer)
    answer = jnp.copysign(answer, value)
    return jnp.where(jnp.isnan(value) | (x < jnp.float64(2.0**-28)), value, answer)


@jax.custom_jvp
def arctangent(value):
    value = jnp.asarray(value, jnp.float64)
    if jax.default_backend() == "gpu":
        from clm_jax.physics.solver_scalarization import _native_tile_trace
        if not _native_tile_trace.get():
            from clm_jax.physics.arctangent_gpu import arctangent_gpu
            return arctangent_gpu(value)
    return _arctangent_reference(value)


@arctangent.defjvp
def _arctangent_jvp(primals, tangents):
    value, = primals
    return arctangent(value), tangents[0] / (1. + value*value)
