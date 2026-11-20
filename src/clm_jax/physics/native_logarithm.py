
import jax
import jax.numpy as jnp
from jax import lax
from jax.extend import core
from jax.interpreters import batching, mlir

from clm_jax.physics.native_exponential import fma
from clm_jax.physics.native_logarithm_tables import IU, IV, LU, LV
from clm_jax.physics.numerics import (
    _lookup_double_double_table,
    _rounded_add as add,
    _rounded_multiply as mul,
)
from clm_jax.physics.solver_scalarization import _native_tile_trace


def _lookup(table, index):
    if _native_tile_trace.get():
        from clm_jax.physics.solver_constant_lookup import constant_lookup
        return constant_lookup(table, index)
    return jnp.asarray(table, dtype=jnp.float64)[index]


def _exact_add(x, y):
    high = add(x, y)
    low = jnp.where(
        jnp.abs(x) > jnp.abs(y),
        add(add(x, -high), y),
        add(add(y, -high), x),
    )
    return high, low


def _general(safe_bits):
    exponent = ((safe_bits >> jnp.uint64(52)) & jnp.uint64(0x7ff)).astype(jnp.int32)
    subnormal = exponent == 0

    def normalize_subnormals():
        leading = jnp.zeros_like(exponent)
        remaining = safe_bits
        for amount in (32, 16, 8, 4, 2, 1):
            upper = remaining >> jnp.uint64(amount)
            present = upper != 0
            remaining = jnp.where(present, upper, remaining)
            leading += jnp.where(present, jnp.int32(amount), jnp.int32(0))
        shift = jnp.int32(52) - leading
        return (jnp.where(subnormal, safe_bits << shift.astype(jnp.uint64), safe_bits),
                jnp.where(subnormal, -shift - jnp.int32(1022), exponent - jnp.int32(1023)))

    mantissa, n = lax.cond(jnp.any(subnormal), normalize_subnormals,
                           lambda: (safe_bits, exponent - jnp.int32(1023)))
    mantissa = (mantissa & jnp.uint64(0xfffffffffffff)) | jnp.uint64(0x3ff0000000000000)
    u = lax.bitcast_convert_type(mantissa, jnp.float64)
    upper = u > jnp.float64(float.fromhex("0x1.6a09e667f3bccp+0"))
    u = jnp.where(upper, mul(u, jnp.float64(.5)), u)
    n += upper.astype(jnp.int32)
    word = lax.bitcast_convert_type(add(u, jnp.float64(151. / 512.)), jnp.uint64)
    i = ((word >> jnp.uint64(44)) & jnp.uint64(255)).astype(jnp.int32)
    i = jnp.clip(i, 0, 181)
    inverse_u = _lookup(IU, i)
    word = lax.bitcast_convert_type(
        fma(u, inverse_u, jnp.float64(361. / 131072.)), jnp.uint64
    )
    j = ((word >> jnp.uint64(36)) & jnp.uint64(65535)).astype(jnp.int32)
    j = jnp.clip(j, 0, 361)
    p0 = mul(
        add(jnp.float64(1.), mul((i - 75).astype(jnp.float64), jnp.float64(1. / 256.))),
        add(jnp.float64(1.), mul((j - 180).astype(jnp.float64), jnp.float64(1. / 65536.))),
    )
    w = mul(add(u, -p0), mul(inverse_u, _lookup(IV, j)))
    polynomial = fma(w, jnp.float64(float.fromhex("0x1.5555555588d2ep-2")),
                     jnp.float64(float.fromhex("-0x1.000000001aa8fp-1")))
    polynomial = fma(mul(polynomial, w), w, w)
    lu, lut = _lookup_double_double_table(LU, i)
    lv, lvt = _lookup_double_double_table(LV, j)
    sij, ssij = _exact_add(lu, lv)
    leading, tail = _exact_add(mul(n.astype(jnp.float64), jnp.float64(float.fromhex("0x1.62e42fefa3800p-1"))), sij)
    remainder = add(add(add(lut, lvt), ssij), tail)
    remainder = fma(n.astype(jnp.float64), jnp.float64(float.fromhex("0x1.ef35793c76730p-45")), remainder)
    return add(leading, add(polynomial, remainder))


def _small(w):
    t = mul(jnp.float64(-.5), w)
    a = mul(t, w)
    aa = fma(t, w, -a)
    b, bb = _exact_add(w, a)
    coefficients = (
        "0x1.5555555555555p-2", "-0x1.fffffffffffbbp-3",
        "0x1.999999999992fp-3", "-0x1.55555556503fdp-3",
        "0x1.24924925b3d62p-3", "-0x1.ffffe160472fcp-4",
        "0x1.c71c525db58acp-4", "-0x1.9a4ac11a2a61cp-4",
        "0x1.750770df2b591p-4",
    )
    polynomial = jnp.full_like(w, float.fromhex(coefficients[-1]))
    for coefficient in reversed(coefficients[:-1]):
        polynomial = fma(w, polynomial, jnp.float64(float.fromhex(coefficient)))
    return add(b, fma(polynomial, mul(mul(w, w), w), add(aa, bb)))


def _reference(value):
    bits = lax.bitcast_convert_type(value, jnp.uint64)
    magnitude = bits & jnp.uint64(0x7fffffffffffffff)
    valid = ((bits >> jnp.uint64(63)) == 0) & (magnitude != 0) & (magnitude < jnp.uint64(0x7ff0000000000000))
    safe_bits = jnp.where(valid, bits, jnp.uint64(0x3ff0000000000000))
    w = add(jnp.where(valid, value, jnp.float64(1.)), jnp.float64(-1.))
    close = jnp.abs(w) <= jnp.float64(.03)

    def general_or_mixed():
        general = _general(safe_bits)
        return lax.cond(jnp.any(close), lambda: jnp.where(close, _small(w), general),
                        lambda: general)

    result = lax.cond(jnp.all(close), lambda: _small(w), general_or_mixed)
    result = jnp.where(magnitude == 0, jnp.float64(-jnp.inf), result)
    result = jnp.where((bits >> jnp.uint64(63) != 0) & (magnitude != 0), jnp.float64(jnp.nan), result)
    result = jnp.where(jnp.isposinf(value), value, result)
    return jnp.where(jnp.isnan(value), add(value, value), result)


def _gpu_implementation(value):
    from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch
    return fused_native_batch(
        _reference, value.reshape(-1), block_size=128, direct_io=True,
        fold_literals=True, preserve_literal_barriers=True, guard_branches=True,
    ).reshape(value.shape)


_log_p = core.Primitive("clm_native_gnu_log")
_log_p.def_impl(_gpu_implementation)
_log_p.def_abstract_eval(lambda value: value)
mlir.register_lowering(_log_p, mlir.lower_fun(_gpu_implementation, multiple_results=False))
batching.primitive_batchers[_log_p] = lambda values, axes: (_log_p.bind(values[0]), axes[0])


@jax.custom_jvp
def native_log(value):
    value = jnp.asarray(value, dtype=jnp.float64)
    if jax.default_backend() == "gpu" and not _native_tile_trace.get():
        return _log_p.bind(value)
    return _reference(value)


@native_log.defjvp
def _native_log_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    return native_log(value), tangent / value
