
from decimal import Decimal, localcontext
from math import factorial

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.physics.numerics import (
    _double_double_add,
    _double_double_multiply,
    _transcendental_primal,
)


def _coefficients():
    with localcontext() as context:
        context.prec = 100
        values = [Decimal((-1) ** k) / Decimal(factorial(2 * k)) for k in range(19)]
        return tuple((float(x), float(x - Decimal.from_float(float(x)))) for x in values)


_COEFFICIENTS = _coefficients()


@_transcendental_primal
def _reference(value):
    value = jnp.asarray(value, jnp.float64)
    zero = jnp.zeros_like(value)
    valid = jnp.isfinite(value) & (jnp.abs(value) <= jnp.float64(3.141592653589793))
    x = jnp.where(valid, value, 0.)
    squared = _double_double_multiply((x, zero), (x, zero))
    initial = tuple(jnp.full_like(x, c) for c in _COEFFICIENTS[-1])

    def horner(total, coefficient):
        return _double_double_add(
            _double_double_multiply(total, squared), (coefficient[0], coefficient[1]),
        ), None

    result, _ = lax.scan(
        horner, initial, jnp.asarray(_COEFFICIENTS[:-1][::-1], jnp.float64),
        unroll=True,
    )
    return jnp.where(valid, result[0] + result[1], jnp.cos(value))


@jax.custom_jvp
def snow_cover_cosine(value):
    return _reference(value)


@snow_cover_cosine.defjvp
def _jvp(primals, tangents):
    value, = primals
    tangent, = tangents
    return snow_cover_cosine(value), -jnp.sin(value) * tangent
