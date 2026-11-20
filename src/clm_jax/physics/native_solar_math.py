
import jax
import jax.numpy as jnp
from jax import lax
from jax.extend import core
from jax.interpreters import batching, mlir

from clm_jax.physics import native_solar_tables as tables
from clm_jax.physics.native_exponential import fma
from clm_jax.physics.numerics import _rounded_add as add, _rounded_multiply as mul
from clm_jax.physics.solver_scalarization import _native_tile_trace

_SIN_COLUMNS = tuple(tuple(row[i] for row in tables.SINCOS) for i in range(4))
_ASIN_COLUMNS = tuple(tuple(row[i] for row in tables.ASIN) for i in range(11))
_SERIES_COLUMNS = tuple(tuple(row[i] for row in tables.ASIN_SERIES) for i in range(2))


def _lookup(columns, row, column):
    if _native_tile_trace.get():
        from clm_jax.physics.solver_constant_lookup import constant_lookup
        return constant_lookup(columns[column], row)
    return jnp.asarray(columns[column], jnp.float64)[row]


def _hex(value):
    return jnp.float64(float.fromhex(value))


def _table_coordinates(x):
    magnitude = abs(x)
    shifted = add(_hex("0x1.8p45"), magnitude)
    row = (lax.bitcast_convert_type(shifted, jnp.uint64) & jnp.uint64(0xffffffff)).astype(jnp.int32)
    return jnp.clip(row, 0, 109), add(magnitude, -add(shifted, -_hex("0x1.8p45")))


def _cos_reduced(x, dx):
    dx = jnp.where(x < 0., -dx, dx)
    row, residual = _table_coordinates(x)
    residual = add(residual, dx)
    squared = mul(residual, residual)
    sine = fma(mul(residual, squared),
               fma(squared, jnp.float64(8.33333214285722277379541354343671e-3),
                   jnp.float64(-1.66666666666664880952546298448555e-1)), residual)
    cosine = mul(squared, fma(squared,
        fma(squared, jnp.float64(1.38888874007937613028114285595617e-3),
            jnp.float64(-4.16666666666664434524222570944589e-2)), jnp.float64(.5)))
    sn, ssn, cs, ccs = (_lookup(_SIN_COLUMNS, row, i) for i in range(4))
    correction = fma(-sn, sine, fma(-cs, cosine, fma(-sine, ssn, ccs)))
    return add(cs, correction)


def _sin_reduced(x, dx):
    row, residual = _table_coordinates(x)
    signed_dx = jnp.where(x <= 0., -dx, dx)
    squared = mul(residual, residual)
    sine = add(residual, fma(mul(residual, squared),
        fma(squared, jnp.float64(8.33333214285722277379541354343671e-3),
            jnp.float64(-1.66666666666664880952546298448555e-1)), signed_dx))
    cosine = fma(residual, signed_dx, mul(squared,
        fma(squared, fma(squared,
            jnp.float64(1.38888874007937613028114285595617e-3),
            jnp.float64(-4.16666666666664434524222570944589e-2)), jnp.float64(.5))))
    sn, ssn, cs, ccs = (_lookup(_SIN_COLUMNS, row, i) for i in range(4))
    correction = fma(cs, sine, fma(-sn, cosine, fma(sine, ccs, ssn)))
    table_value = jnp.copysign(add(sn, correction), x)
    xx = mul(x, x)
    polynomial = _hex("-0x1.ADDFFC2FCDF59p-26")
    for coefficient in ("0x1.71DE27B9A7ED9p-19", "-0x1.A01A019DB08B8p-13",
                        "0x1.1111111110ECEp-7", "-0x1.5555555555555p-3"):
        polynomial = fma(xx, polynomial, _hex(coefficient))
    small = add(x, fma(fma(polynomial, x, -mul(jnp.float64(.5), dx)), xx, dx))
    return jnp.where(abs(x) < .126, small, table_value)


def _reduce(x):
    shifted = fma(x, _hex("0x1.45F306DC9C883p-1"), _hex("0x1.8p52"))
    integer = add(shifted, -_hex("0x1.8p52"))
    quadrant = (lax.bitcast_convert_type(shifted, jnp.uint64) & jnp.uint64(3)).astype(jnp.int32)
    y = fma(-integer, _hex("-0x1.DDE973C000000p-27"),
            fma(-integer, _hex("0x1.921FB58000000p0"), x))
    first = _hex("-0x1.CB3B398000000p-55")
    a = fma(-integer, first, y)
    tail = fma(-integer, first, add(y, -a))
    second = _hex("-0x1.d747f23e32ed7p-83")
    b = fma(-integer, second, a)
    tail = add(tail, fma(-integer, second, add(a, -b)))
    return b, tail, quadrant


def _sincos_reference(x, cosine):
    safe = jnp.where(jnp.isfinite(x) & (abs(x) < 32.), x, jnp.float64(0.))
    a, da, quadrant = _reduce(safe)
    quadrant = quadrant + int(cosine)
    reduced = jnp.where((quadrant & 1) != 0, _cos_reduced(a, da), _sin_reduced(a, da))
    reduced = jnp.where((quadrant & 2) != 0, -reduced, reduced)
    complement = add(_hex("0x1.921FB54442D18p0"), -abs(safe))
    if cosine:
        angle = add(complement, _hex("0x1.1A62633145C07p-54"))
        tail = add(add(complement, -angle), _hex("0x1.1A62633145C07p-54"))
        middle = _sin_reduced(angle, tail)
        small = _cos_reduced(safe, jnp.zeros_like(safe))
        small = jnp.where(abs(safe) < _hex("0x1p-27"), jnp.float64(1.), small)
        general = jnp.cos(x)
    else:
        middle = jnp.copysign(_cos_reduced(complement, _hex("0x1.1A62633145C07p-54")), safe)
        small = _sin_reduced(safe, jnp.zeros_like(safe))
        small = jnp.where(abs(safe) < _hex("0x1p-26"), safe, small)
        general = jnp.sin(x)
    high = ((lax.bitcast_convert_type(safe, jnp.uint64) >> jnp.uint64(32)) & jnp.uint64(0x7fffffff))
    value = jnp.where(high < 0x3feb6000, small,
                      jnp.where(high < 0x400368fd, middle, reduced))
    return jnp.where(jnp.isfinite(x) & (abs(x) < 32.), value, general)


def _sincos_sse_reference(x, cosine):
    magnitude = abs(x)
    complement = add(_hex("0x1.921FB54442D18p0"), -magnitude)
    angle = add(complement, _hex("0x1.1A62633145C07p-54"))
    tail = add(add(complement, -angle), _hex("0x1.1A62633145C07p-54"))
    high = ((lax.bitcast_convert_type(x, jnp.uint64) >> jnp.uint64(32))
            & jnp.uint64(0x7fffffff))
    middle = high >= 0x3feb6000
    a = jnp.where(middle, angle, x)
    da = jnp.where(middle, tail, jnp.zeros_like(x))

    row, residual = _table_coordinates(a)
    signed_da = jnp.where(a <= 0., -da, da)
    square = mul(residual, residual)
    sn, ssn, cs, ccs = (_lookup(_SIN_COLUMNS, row, i) for i in range(4))
    sine_coefficient = add(jnp.float64(-1.66666666666664880952546298448555e-1),
                           mul(square, jnp.float64(8.33333214285722277379541354343671e-3)))
    cosine_coefficient = add(jnp.float64(.5), mul(square, add(
        jnp.float64(-4.16666666666664434524222570944589e-2),
        mul(square, jnp.float64(1.38888874007937613028114285595617e-3)),
    )))
    sine = add(residual, add(signed_da, mul(mul(residual, square), sine_coefficient)))
    cosine_term = add(mul(residual, signed_da), mul(square, cosine_coefficient))
    correction = add(add(add(ssn, mul(sine, ccs)), -mul(sn, cosine_term)), mul(cs, sine))
    sine = jnp.copysign(add(sn, correction), a)

    xx = mul(a, a)
    polynomial = _hex("-0x1.ADDFFC2FCDF59p-26")
    for coefficient in ("0x1.71DE27B9A7ED9p-19", "-0x1.A01A019DB08B8p-13",
                        "0x1.1111111110ECEp-7", "-0x1.5555555555555p-3"):
        polynomial = add(mul(polynomial, xx), _hex(coefficient))
    small_sine = add(a, add(mul(add(mul(polynomial, a), -mul(jnp.float64(.5), da)), xx), da))
    sine = jnp.where(abs(a) < .126, small_sine, sine)

    cos_da = jnp.where(a < 0., -da, da)
    cos_residual = add(residual, cos_da)
    cos_square = mul(cos_residual, cos_residual)
    sine_term = add(cos_residual, mul(mul(cos_residual, cos_square), add(
        jnp.float64(-1.66666666666664880952546298448555e-1),
        mul(cos_square, jnp.float64(8.33333214285722277379541354343671e-3)),
    )))
    cos_term = mul(cos_square, add(jnp.float64(.5), mul(cos_square, add(
        jnp.float64(-4.16666666666664434524222570944589e-2),
        mul(cos_square, jnp.float64(1.38888874007937613028114285595617e-3)),
    ))))
    cos_correction = add(add(add(ccs, -mul(sine_term, ssn)), -mul(cs, cos_term)),
                         -mul(sn, sine_term))
    cosine_value = add(cs, cos_correction)
    if cosine:
        result = jnp.where(middle, sine, cosine_value)
        result = jnp.where(high < 0x3e400000, jnp.float64(1.), result)
    else:
        result = jnp.where(middle, jnp.copysign(cosine_value, x), sine)
        result = jnp.where(high < 0x3e400000, x, result)
    return jnp.where(magnitude < 2.426265, result, _sincos_reference(x, cosine))


def _two_sum(a, b):
    leading = add(a, b)
    z = add(leading, -a)
    return leading, add(add(a, -add(leading, -z)), add(b, -z))


def _dd_multiply(a, b):
    leading = mul(a[0], b[0])
    tail = fma(a[0], b[0], -leading)
    tail = fma(a[0], b[1], fma(a[1], b[0], tail))
    return _two_sum(leading, tail)


def _asin_series(x):
    squared = mul(x, x)
    square = squared, fma(x, x, -squared)
    result = tuple(jnp.full_like(x, v) for v in tables.ASIN_SERIES[-1])
    def horner(index, result):
        row = 63 - index
        result = _dd_multiply(result, square)
        leading, tail = _two_sum(result[0], _lookup(_SERIES_COLUMNS, row, 0))
        return _two_sum(leading, add(add(tail, result[1]), _lookup(_SERIES_COLUMNS, row, 1)))
    result = lax.fori_loop(0, 64, horner, result)
    leading, tail = _dd_multiply(result, (x, jnp.zeros_like(x)))
    return add(leading, tail)


def _asin_reference(x):
    magnitude = abs(x)
    safe = jnp.where(magnitude <= .5, magnitude, jnp.float64(0.))
    squared = mul(safe, safe)
    polynomial = jnp.float64(1.81382903404565056280372531963613e-2)
    for coefficient in (2.23551211026525610742786300334557e-2,
                        3.03821268582119319911193410625235e-2,
                        4.46428561421059750978517350006940e-2,
                        7.50000000026122686814431784722623e-2,
                        1.66666666666664110590506577996662e-1):
        polynomial = fma(polynomial, squared, jnp.float64(coefficient))
    correction = mul(polynomial, mul(squared, safe))
    small = add(safe, correction)
    small_tail = add(add(safe, -small), correction)
    small_valid = small == fma(jnp.float64(1.025), small_tail, small)
    high = (lax.bitcast_convert_type(safe, jnp.uint64) >> jnp.uint64(32)).astype(jnp.uint32)
    row = jnp.where(high < 0x3fd00000, (high & 0xfffff) >> 15,
                    ((high & 0xfffff) >> 14) + 32).astype(jnp.int32)
    row = jnp.clip(row, 0, 95)
    xx = add(safe, -_lookup(_ASIN_COLUMNS, row, 0))
    p = _lookup(_ASIN_COLUMNS, row, 6)
    for column in (5, 4, 3, 2):
        p = fma(xx, p, _lookup(_ASIN_COLUMNS, row, column))
    p = fma(mul(xx, xx), p, _lookup(_ASIN_COLUMNS, row, 7))
    t = add(mul(_lookup(_ASIN_COLUMNS, row, 1), xx), p)
    leading = _lookup(_ASIN_COLUMNS, row, 8)
    result = add(leading, t)
    tail = add(add(leading, -result), t)
    valid = result == fma(jnp.float64(1.05), tail, result)
    refined = fma(xx, _lookup(_ASIN_COLUMNS, row, 9), leading)
    t = add(fma(xx, _lookup(_ASIN_COLUMNS, row, 9), add(leading, -refined)),
            fma(xx, _lookup(_ASIN_COLUMNS, row, 10), p))
    refined_result = add(refined, t)
    refined_tail = add(add(refined, -refined_result), t)
    refined_valid = refined_result == fma(jnp.float64(1.0005), refined_tail, refined_result)
    result = jnp.where(valid, result, refined_result)
    valid = valid | refined_valid
    result = jnp.where(safe < .125, small, result)
    valid = jnp.where(safe < .125, small_valid, valid)
    result = lax.cond(jnp.all(valid), lambda: result,
                      lambda: jnp.where(valid, result, _asin_series(safe)))
    result = jnp.copysign(result, x)
    return jnp.where(magnitude <= .5, result, jnp.arcsin(x))


def _elementwise(name, function):
    def gpu(value):
        from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch
        return fused_native_batch(function, value.reshape(-1), block_size=128,
            direct_io=True, fold_literals=True, preserve_literal_barriers=True,
            guard_branches=True).reshape(value.shape)
    primitive = core.Primitive(name)
    primitive.def_impl(gpu)
    primitive.def_abstract_eval(lambda value: value)
    mlir.register_lowering(primitive, mlir.lower_fun(gpu, multiple_results=False))
    batching.primitive_batchers[primitive] = lambda values, axes: (primitive.bind(values[0]), axes[0])
    def evaluate(value):
        value = jnp.asarray(value, jnp.float64)
        if jax.default_backend() == "gpu" and not _native_tile_trace.get():
            return primitive.bind(value)
        return function(value)
    return evaluate


native_sin = jax.custom_jvp(_elementwise("clm_native_solar_sin", lambda x: _sincos_reference(x, False)))
native_cos = jax.custom_jvp(_elementwise("clm_native_solar_cos", lambda x: _sincos_reference(x, True)))
native_asin = jax.custom_jvp(_elementwise("clm_native_solar_asin", _asin_reference))
native_sincos_sin = jax.custom_jvp(_elementwise(
    "clm_native_solar_sincos_sin", lambda x: _sincos_sse_reference(x, False)))
native_sincos_cos = jax.custom_jvp(_elementwise(
    "clm_native_solar_sincos_cos", lambda x: _sincos_sse_reference(x, True)))


@native_sin.defjvp
def _sin_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return native_sin(x), jnp.cos(x) * dx


@native_cos.defjvp
def _cos_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return native_cos(x), -jnp.sin(x) * dx


@native_sincos_sin.defjvp
def _sincos_sin_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return native_sincos_sin(x), jnp.cos(x) * dx


@native_sincos_cos.defjvp
def _sincos_cos_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return native_sincos_cos(x), -jnp.sin(x) * dx


@native_asin.defjvp
def _asin_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return native_asin(x), dx / jnp.sqrt(1. - x * x)
