
import jax
import jax.numpy as jnp
from jax import lax
from jax.extend import core
from jax.interpreters import ad, batching, mlir

from clm_jax.physics.native_exponential_tables import COAR, FINE, SMALL
from clm_jax.physics.numerics import (
    _lookup_double_double_table,
    _rounded_add as add,
    _rounded_multiply as mul,
    _rounded_fused_multiply_add,
)
from clm_jax.physics.solver_scalarization import _native_tile_trace


def fma(a, b, c):
    a, b, c = jnp.broadcast_arrays(jnp.float64(a), jnp.float64(b), jnp.float64(c))
    if _native_tile_trace.get():
        from clm_jax.physics.solver_fma import fma_p
        return fma_p.bind(a, b, c)
    return _rounded_fused_multiply_add(a, b, c)


def _polynomial(x):
    squared = mul(x, x)
    second = fma(squared, jnp.float64(float.fromhex("0x1.6c16c16c16c17p-10")),
                 fma(x, jnp.float64(float.fromhex("0x1.1111111111111p-7")),
                     jnp.float64(float.fromhex("0x1.5555555555555p-5"))))
    first = fma(x, jnp.float64(float.fromhex("0x1.5555555555555p-3")), jnp.float64(.5))
    return add(x, fma(squared, first, mul(mul(squared, squared), second)))


def _high_word(value):
    bits = lax.bitcast_convert_type(value, jnp.uint64)
    return ((bits >> jnp.uint64(32)) & jnp.uint64(0x7fffffff)).astype(jnp.uint32)


def _small(value, high):
    safe_high = jnp.where(high >= jnp.uint32(0x3f862e42), high, jnp.uint32(0x3fe00000))
    shift = jnp.uint32(0x40c) - (safe_high >> jnp.uint32(20))
    coordinate = ((jnp.uint32(0x100000) | (safe_high & jnp.uint32(0xfffff))) >> shift)
    row = (((coordinate - jnp.uint32(1)) & jnp.uint32(0xfffffffe)) >> jnp.uint32(1)).astype(jnp.int32)
    row = jnp.clip(row, 0, 66) + jnp.where(jnp.signbit(value), 67, 0)
    base, exponential = _lookup_double_double_table(SMALL, row)
    small = fma(exponential, _polynomial(add(value, -base)), exponential)
    small = jnp.where(high < jnp.uint32(0x3f862e42), add(jnp.float64(1.), _polynomial(value)), small)
    quadratic = fma(value, fma(value, jnp.float64(.5), jnp.float64(1.)), jnp.float64(1.))
    small = jnp.where(high < jnp.uint32(0x3ed00000), quadratic, small)
    small = jnp.where(high < jnp.uint32(0x3e300000), add(jnp.float64(1.), value), small)
    return small


def _range_reduced(value):
    three51 = jnp.float64(float.fromhex("0x1.8000000000000p+52"))
    y = fma(value, jnp.float64(float.fromhex("0x1.71547652b82fep+0")), three51)
    power = add(y, -three51)
    t = fma(-power, jnp.float64(float.fromhex("0x1.62e42fefa3800p-1")), value)
    three33 = jnp.float64(float.fromhex("0x1.8000000000000p+34"))
    y = add(t, three33)
    base = add(y, -three33)
    delta = fma(-power, jnp.float64(float.fromhex("0x1.ef35793c76730p-45")), add(t, -base))
    polynomial = fma(delta, jnp.float64(float.fromhex("0x1.5555555555a0fp-3")),
                     jnp.float64(float.fromhex("0x1.00000000004dcp-1")))
    epsilon = fma(mul(delta, delta), polynomial, delta)
    low = (lax.bitcast_convert_type(y, jnp.uint64) & jnp.uint64(0xffffffff)).astype(jnp.uint32)
    coarse_index = (((low.astype(jnp.int32) >> jnp.int32(8)) & jnp.int32(-2)) + 356) // 2
    fine_index = (low & jnp.uint32(511)).astype(jnp.int32)
    ch, cl = _lookup_double_double_table(COAR, jnp.clip(coarse_index, 0, 355))
    fh, fl = _lookup_double_double_table(FINE, fine_index)
    high_product = mul(ch, fh)
    low_product = fma(cl, fl, fma(ch, fl, mul(cl, fh)))
    remainder = fma(high_product, epsilon, fma(low_product, epsilon, low_product))
    result = add(high_product, remainder)
    return power.astype(jnp.int64), high_product, remainder, result


def _power_of_two(exponent):
    bits = (exponent + jnp.int64(1023)) << jnp.int64(52)
    return lax.bitcast_convert_type(bits, jnp.float64)


def _scaled_result(value, high, power, leading, remainder, result):
    normal = mul(result, _power_of_two(power))
    correction = add(add(leading, -result), remainder)
    below_one = result < jnp.float64(1.)
    negative_result = jnp.where(below_one, add(result, result), result)
    correction = jnp.where(below_one, add(correction, correction), correction)
    negative_power = power - below_one.astype(jnp.int64)
    shift = jnp.clip(-(jnp.int64(1022) + negative_power), 0, 53)
    scaled = mul(negative_result, _power_of_two(-shift))
    correction = mul(correction, _power_of_two(-shift))
    total = add(jnp.float64(1.), scaled)
    residual = add(add(add(jnp.float64(1.), -total), scaled), correction)
    total = add(total, residual)
    significand = lax.bitcast_convert_type(total, jnp.uint64) - jnp.uint64(0x3ff0000000000000)
    subnormal = lax.bitcast_convert_type(significand, jnp.float64)
    negative = jnp.where(negative_power >= -1022,
                         mul(negative_result, _power_of_two(negative_power)), subnormal)
    positive = mul(mul(result, _power_of_two(power - 256)),
                   jnp.float64(float.fromhex("0x1p+256")))
    extended = jnp.where(jnp.signbit(value), negative, positive)
    return jnp.where(high < jnp.uint32(0x40862002), normal, extended)


def _reference(value):
    safe = jnp.where(jnp.isfinite(value) & (abs(value) < 746.), value, jnp.float64(0.))
    high = _high_word(safe)

    def general(_):
        power, leading, remainder, result = _range_reduced(safe)
        return _scaled_result(safe, high, power, leading, remainder, result)

    if _native_tile_trace.get() and safe.size == 1:
        result = lax.cond(jnp.all(high < jnp.uint32(0x3ff0a2b2)),
                          lambda _: _small(safe, high), general, ())
    else:
        result = jnp.where(high < jnp.uint32(0x3ff0a2b2), _small(safe, high), general(()))
    result = jnp.where(jnp.isposinf(value), jnp.float64(jnp.inf), result)
    result = jnp.where(jnp.isneginf(value), jnp.float64(0.), result)
    result = jnp.where(value > jnp.float64(709.782712893384), jnp.float64(jnp.inf), result)
    result = jnp.where(value < jnp.float64(float.fromhex("-0x1.74910d52d3051p+9")),
                       jnp.float64(0.), result)
    nan = lax.bitcast_convert_type(
        lax.bitcast_convert_type(value, jnp.uint64) | jnp.uint64(0x8000000000000),
        jnp.float64,
    )
    return jnp.where(jnp.isnan(value), nan, result)


def _gpu_implementation(value):
    from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch
    return fused_native_batch(
        _reference, value.reshape(-1), block_size=128, direct_io=True,
        fold_literals=True, preserve_literal_barriers=True, guard_branches=True,
    ).reshape(value.shape)


_exp_p = core.Primitive("clm_native_gnu_exp")
_exp_p.def_impl(_gpu_implementation)
_exp_p.def_abstract_eval(lambda value: value)
mlir.register_lowering(_exp_p, mlir.lower_fun(_gpu_implementation, multiple_results=False))
batching.primitive_batchers[_exp_p] = (
    lambda values, axes: (_exp_p.bind(values[0]), axes[0])
)


def _primitive_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    result = _exp_p.bind(value)
    if isinstance(tangent, ad.Zero):
        return result, ad.Zero.from_primal_value(result)
    return result, result * tangent


ad.primitive_jvps[_exp_p] = _primitive_jvp


@jax.custom_jvp
def native_exp(value):
    value = jnp.asarray(value, jnp.float64)
    if jax.default_backend() == "gpu" and not _native_tile_trace.get():
        return _exp_p.bind(value)
    return _reference(value)


@native_exp.defjvp
def _jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    result = native_exp(x)
    return result, result * dx
