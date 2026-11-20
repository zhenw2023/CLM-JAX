
import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.physics.native_acos import (
    _HALF_PI, _HALF_PI_TAIL, _ROOT_COLUMNS, _TABLE_COLUMNS, _asin_dd,
)
from clm_jax.physics.native_exponential import fma
from clm_jax.physics.native_solar_math import (
    _asin_reference, _elementwise, _lookup, _two_sum,
)
from clm_jax.physics.numerics import (
    _rounded_add as add, _rounded_multiply as mul, fortran_divide,
)


def _refine(value):
    z = mul(jnp.float64(.5), add(jnp.float64(1.), -value))
    root = jnp.sqrt(z)
    tail = fortran_divide(fma(-root, root, z), add(root, root))
    high, low = _asin_dd(*_two_sum(root, tail))
    leading = add(jnp.float64(_HALF_PI), -add(high, high))
    tail = add(add(add(jnp.float64(_HALF_PI), -leading), -add(high, high)),
               add(jnp.float64(_HALF_PI_TAIL), -add(low, low)))
    return add(leading, tail)


def _middle(value):
    safe = jnp.where((value >= .5) & (value < .96875), value, .75)
    k = (lax.bitcast_convert_type(safe, jnp.uint64) >> jnp.uint64(32)).astype(jnp.uint32)
    row = ((k & jnp.uint32(0xfe000)) >> jnp.uint32(13)).astype(jnp.int32) + 96
    c = tuple(_lookup(_TABLE_COLUMNS, row, i) for i in range(15))
    offset = add(safe, -c[0])
    p = c[10]
    for column in range(9, 1, -1):
        p = add(c[column], mul(offset, p))
    p = add(mul(mul(offset, offset), p), c[11])
    t = add(mul(c[1], offset), p)
    result = add(c[12], t)
    error = add(add(c[12], -result), t)
    accepted = result == add(result, mul(jnp.float64(1.01), error))
    slope = mul(offset, c[13])
    leading = add(c[12], slope)
    tail = add(add(add(c[12], -leading), slope), add(p, mul(offset, c[14])))
    refined = add(leading, tail)
    error = add(add(leading, -refined), tail)
    epsilon = jnp.where(safe < .75, 1.0005,
                        jnp.where(safe < .921875, 1.0008, 1.0007))
    refined_accepted = refined == add(refined, mul(epsilon, error))
    result = jnp.where(accepted, result, refined)
    return lax.cond(jnp.all(accepted | refined_accepted), lambda: result,
        lambda: jnp.where(accepted | refined_accepted, result, _refine(safe)))


def _endpoint(value):
    safe = jnp.where((value >= .96875) & (value < 1.), value, .984375)
    z = mul(jnp.float64(.5), add(jnp.float64(1.), -safe))
    k = (lax.bitcast_convert_type(z, jnp.uint64) >> jnp.uint64(32)).astype(jnp.uint32)
    row = ((k & jnp.uint32(0x1fffff)) >> jnp.uint32(14)).astype(jnp.int32)
    scale = jnp.int32(511) - (k >> jnp.uint32(21)).astype(jnp.int32)
    t = mul(_lookup(_ROOT_COLUMNS, row, 0), jnp.ldexp(jnp.ones_like(z), scale))
    residual = add(jnp.float64(1.), -mul(mul(t, t), z))
    p = jnp.float64(.312523626554518656309172508769531)
    for coefficient in (.375017500867345182581453026130850,
                        .499999999495955425917856814202739,
                        .999999999859990725855365213134618):
        p = add(jnp.float64(coefficient), mul(residual, p))
    t = mul(t, p)
    c = mul(t, z)
    t = mul(c, add(jnp.float64(1.5), -mul(mul(jnp.float64(.5), t), c)))
    split = jnp.float64(16777216.)
    y = add(add(c, split), -split)
    cc = fortran_divide(add(z, -mul(y, y)), add(t, y))
    p = jnp.float64(.0181382903404565056280372531963613)
    for coefficient in (.0223551211026525610742786300334557,
                        .0303821268582119319911193410625235,
                        .0446428561421059750978517350006940,
                        .0750000000026122686814431784722623,
                        .166666666666664110590506577996662):
        p = add(mul(p, z), jnp.float64(coefficient))
    p = mul(p, z)
    correction = add(add(jnp.float64(_HALF_PI_TAIL), -mul(jnp.float64(2.), cc)),
                     -mul(mul(jnp.float64(2.), add(y, cc)), p))
    leading = add(jnp.float64(_HALF_PI), -mul(jnp.float64(2.), y))
    result = add(leading, correction)
    error = add(add(leading, -result), correction)
    accepted = result == add(result, mul(jnp.float64(1.003), error))
    return lax.cond(jnp.all(accepted), lambda: result,
                    lambda: jnp.where(accepted, result, _refine(safe)))


def _reference(value):
    magnitude = jnp.abs(value)
    result = jnp.where(magnitude < .5, jnp.abs(_asin_reference(value)),
        jnp.where(magnitude < .96875, _middle(magnitude), _endpoint(magnitude)))
    result = jnp.copysign(result, value)
    return jnp.where(magnitude < 1., result, jnp.arcsin(value))


native_asin = jax.custom_jvp(_elementwise("clm_native_full_asin", _reference))


@native_asin.defjvp
def _jvp(primals, tangents):
    value, = primals
    tangent, = tangents
    return native_asin(value), tangent / jnp.sqrt((1. - value) * (1. + value))
