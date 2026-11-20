
import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.physics.native_acos_tables import ROOT, MIDDLE
from clm_jax.physics.native_exponential import fma
from clm_jax.physics.native_solar_math import (
    _dd_multiply, _elementwise, _lookup, _two_sum,
)
from clm_jax.physics.native_solar_tables import ASIN, ASIN_SERIES
from clm_jax.physics.numerics import (
    _rounded_add as add, _rounded_multiply as mul, fortran_divide,
)

_ROOT_COLUMNS = (ROOT,)
_SERIES_COLUMNS = tuple(tuple(row[i] for row in ASIN_SERIES) for i in range(2))
_TABLE_ROWS = tuple(row[:7] + (0.,) * 4 + row[7:] for row in ASIN) + MIDDLE
_TABLE_COLUMNS = tuple(tuple(row[i] for row in _TABLE_ROWS) for i in range(15))
_HALF_PI = float.fromhex("0x1.921fb54442d18p+0")
_HALF_PI_TAIL = float.fromhex("0x1.1a62633145c07p-54")


def _asin_dd(high, low):
    square = _dd_multiply((high, low), (high, low))
    initial = tuple(jnp.full_like(high, v) for v in ASIN_SERIES[-1])

    def horner(index, result):
        row = len(ASIN_SERIES) - 2 - index
        result = _dd_multiply(result, square)
        leading, tail = _two_sum(result[0], _lookup(_SERIES_COLUMNS, row, 0))
        return _two_sum(leading, add(add(tail, result[1]),
                                    _lookup(_SERIES_COLUMNS, row, 1)))

    series = lax.fori_loop(0, len(ASIN_SERIES) - 1, horner, initial)
    return _dd_multiply(series, (high, low))


def _half_pi_minus(high, low):
    leading = add(jnp.float64(_HALF_PI), -high)
    tail = add(add(add(jnp.float64(_HALF_PI), -leading), -high),
               add(jnp.float64(_HALF_PI_TAIL), -low))
    return add(leading, tail)


def _refine_acos(value):
    magnitude = jnp.abs(value)
    half_angle = magnitude > .5
    z = mul(jnp.float64(.5), add(jnp.float64(1.), -magnitude))
    root = jnp.sqrt(z)
    root_tail = fortran_divide(fma(-root, root, z), add(root, root))
    high, low = _two_sum(root, root_tail)
    high = jnp.where(half_angle, high, value)
    low = jnp.where(half_angle, low, jnp.float64(0.))
    high, low = _asin_dd(high, low)
    endpoint = jnp.where(jnp.signbit(value), _half_pi_minus(high, low), add(high, low))
    return jnp.where(half_angle, add(endpoint, endpoint), _half_pi_minus(high, low))


def _small_reference(value):
    safe = jnp.where(jnp.abs(value) < .125, value, jnp.float64(0.))
    square = mul(safe, safe)
    p = jnp.float64(.0181382903404565056280372531963613)
    for coefficient in (.0223551211026525610742786300334557,
                        .0303821268582119319911193410625235,
                        .0446428561421059750978517350006940,
                        .0750000000026122686814431784722623,
                        .166666666666664110590506577996662):
        p = add(mul(p, square), jnp.float64(coefficient))
    correction = mul(p, mul(square, safe))
    leading = add(jnp.float64(_HALF_PI), -safe)
    tail = add(add(add(add(jnp.float64(_HALF_PI), -leading), -safe),
                   jnp.float64(_HALF_PI_TAIL)), -correction)
    result = add(leading, tail)
    residual = add(add(leading, -result), tail)
    accepted = result == add(result, mul(jnp.float64(1.004), residual))
    return lax.cond(jnp.all(accepted), lambda: result,
                    lambda: jnp.where(accepted, result, _refine_acos(safe)))


def _table_reference(value):
    magnitude = jnp.abs(value)
    safe = jnp.where((magnitude >= .125) & (magnitude < .96875), value, jnp.float64(.25))
    k = ((lax.bitcast_convert_type(jnp.abs(safe), jnp.uint64) >> jnp.uint64(32))
         .astype(jnp.uint32))
    coarse = ((k & jnp.uint32(0xfe000)) >> jnp.uint32(13)).astype(jnp.int32)
    small_row = jnp.where(k < 0x3fd00000, (k & 0xfffff) >> 15,
                         ((k & 0xfffff) >> 14) + 32).astype(jnp.int32)
    row = jnp.where(k < 0x3fe00000, small_row, coarse + 96)
    coefficients = tuple(_lookup(_TABLE_COLUMNS, row, i) for i in range(15))
    offset = add(jnp.abs(safe), -coefficients[0])
    polynomial = coefficients[10]
    for column in range(9, 1, -1):
        polynomial = add(coefficients[column], mul(offset, polynomial))
    polynomial = add(mul(mul(offset, offset), polynomial), coefficients[11])
    t = add(mul(coefficients[1], offset), polynomial)
    negative = jnp.signbit(safe)
    center = add(jnp.float64(_HALF_PI), jnp.where(negative, coefficients[12], -coefficients[12]))
    tail = add(jnp.float64(_HALF_PI_TAIL), jnp.where(negative, t, -t))
    result = add(center, tail)
    error = add(add(center, -result), tail)
    epsilon = jnp.where(k < 0x3fe00000, 1.02,
                        jnp.where(negative, jnp.where(k < 0x3fe80000, 1.02,
                            jnp.where(k < 0x3fed8000, 1.01, 1.005)), 1.04))
    accepted = result == add(result, mul(epsilon, error))
    slope = mul(offset, coefficients[13])
    refined_center = add(coefficients[12], slope)
    refined_tail = add(add(add(coefficients[12], -refined_center), slope),
                       add(polynomial, mul(offset, coefficients[14])))
    positive_center = add(jnp.float64(_HALF_PI), -refined_center)
    positive_tail = add(add(add(add(jnp.float64(_HALF_PI), -positive_center),
                               -refined_center), -refined_tail), jnp.float64(_HALF_PI_TAIL))
    negative_center = add(jnp.float64(_HALF_PI), refined_center)
    negative_tail = add(add(add(jnp.float64(_HALF_PI), -negative_center), refined_center),
                        add(jnp.float64(_HALF_PI_TAIL), refined_tail))
    refined_center = jnp.where(negative, negative_center, positive_center)
    refined_tail = jnp.where(negative, negative_tail, positive_tail)
    refined = add(refined_center, refined_tail)
    refined_error = add(add(refined_center, -refined), refined_tail)
    epsilon = jnp.where(k < 0x3fe00000, 1.0002,
        jnp.where(k < 0x3fe80000, jnp.where(negative, 1.0002, 1.0004),
            jnp.where(k < 0x3fed8000, jnp.where(negative, 1.0008, 1.0032),
                jnp.where(negative, 1.0005, 1.003))))
    refined_accepted = refined == add(refined, mul(epsilon, refined_error))
    result = jnp.where(accepted, result, refined)
    return lax.cond(jnp.all(accepted | refined_accepted), lambda: result,
        lambda: jnp.where(accepted | refined_accepted, result, _refine_acos(safe)))


def _endpoint_reference(value):
    magnitude = jnp.abs(value)
    selected = (magnitude >= .96875) & (magnitude < 1.)
    safe = jnp.where(selected, value, jnp.float64(.984375))
    negative = jnp.signbit(safe)
    z = mul(jnp.float64(.5), add(jnp.float64(1.), -jnp.abs(safe)))
    high_word = (lax.bitcast_convert_type(z, jnp.uint64) >> jnp.uint64(32)).astype(jnp.uint32)
    row = ((high_word & jnp.uint32(0x1fffff)) >> jnp.uint32(14)).astype(jnp.int32)
    scale = jnp.int32(511) - (high_word >> jnp.uint32(21)).astype(jnp.int32)
    t = mul(_lookup(_ROOT_COLUMNS, row, 0), jnp.ldexp(jnp.ones_like(z), scale))
    residual = add(jnp.float64(1.), -mul(mul(t, t), z))
    polynomial = add(jnp.float64(.375017500867345182581453026130850),
                     mul(residual, jnp.float64(.312523626554518656309172508769531)))
    polynomial = add(jnp.float64(.499999999495955425917856814202739),
                     mul(residual, polynomial))
    polynomial = add(jnp.float64(.999999999859990725855365213134618),
                     mul(residual, polynomial))
    t = mul(t, polynomial)
    c = mul(t, z)
    t = mul(c, add(jnp.float64(1.5), -mul(mul(jnp.float64(.5), t), c)))
    split = mul(jnp.float64(134217728.), c)
    y = add(add(split, c), -split)
    cc = fortran_divide(add(z, -mul(y, y)), add(t, y))
    p = jnp.float64(.0181382903404565056280372531963613)
    for coefficient in (.0223551211026525610742786300334557,
                        .0303821268582119319911193410625235,
                        .0446428561421059750978517350006940,
                        .0750000000026122686814431784722623,
                        .166666666666664110590506577996662):
        p = add(mul(p, z), jnp.float64(coefficient))
    p = mul(p, z)
    positive_tail = add(cc, mul(p, add(y, cc)))
    negative_tail = add(add(jnp.float64(_HALF_PI_TAIL), -cc), -mul(add(y, cc), p))
    leading = jnp.where(negative, add(jnp.float64(_HALF_PI), -y), y)
    tail = jnp.where(negative, negative_tail, positive_tail)
    result = add(leading, tail)
    error = add(add(leading, -result), tail)
    epsilon = jnp.where(negative, jnp.float64(1.002), jnp.float64(1.03))
    accepted = result == add(result, mul(epsilon, error))

    def refine():
        root_high, root_low = _two_sum(y, cc)
        high, low = _asin_dd(root_high, root_low)
        negative_leading = add(jnp.float64(_HALF_PI), -high)
        negative_tail = add(add(add(jnp.float64(_HALF_PI), -negative_leading), -high),
                            add(jnp.float64(_HALF_PI_TAIL), -low))
        refined = jnp.where(negative, add(negative_leading, negative_tail), add(high, low))
        return jnp.where(accepted, result, refined)

    result = lax.cond(jnp.all(accepted), lambda: result, refine)
    return add(result, result)


def _reference(value):
    magnitude = jnp.abs(value)
    safe = jnp.where(magnitude < 1., value, jnp.float64(0.))
    small = _small_reference(safe)
    table = _table_reference(safe)
    endpoint = _endpoint_reference(safe)
    result = jnp.where(magnitude < .125, small,
                       jnp.where(magnitude < .96875, table, endpoint))
    return jnp.where(magnitude < 1., result, jnp.arccos(value))


native_acos = _elementwise("clm_native_acos", _reference)
