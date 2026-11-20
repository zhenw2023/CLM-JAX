
import jax
import jax.numpy as jnp
from jax import lax
from jax.extend import core
from jax.interpreters import ad, batching, mlir

from clm_jax.physics.native_exponential import fma
from clm_jax.physics.native_exponential_tables import COAR, FINE
from clm_jax.physics.native_power_tables import UI, VJ
from clm_jax.physics.numerics import (
    _lookup_double_double_table,
    _rounded_add as add,
    _rounded_multiply as mul,
    fortran_divide,
    fortran_hardware_power,
)
from clm_jax.physics.solver_scalarization import _native_tile_trace

_UI = tuple(tuple(row[c] for row in UI) for c in range(4))
_VJ = tuple(tuple(row[c] for row in VJ) for c in range(3))
_LN2_HIGH = float.fromhex("0x1.62e42fefa3800p-1")
_LN2_LOW = float.fromhex("0x1.ef35793c76730p-45")


def _lookup(table, index):
    if _native_tile_trace.get():
        from clm_jax.physics.solver_constant_lookup import constant_lookup
        return constant_lookup(table, index)
    return jnp.asarray(table, jnp.float64)[index]


def _polynomial(value, coefficients):
    result = jnp.full_like(value, coefficients[-1])
    for coefficient in coefficients[-2::-1]:
        result = add(mul(value, result), jnp.float64(coefficient))
    return result


def _general_log_pair(base):
    bits = lax.bitcast_convert_type(base, jnp.uint64)
    high = (bits >> jnp.uint64(32)).astype(jnp.uint32)
    upper = (high & jnp.uint32(0xfffff)) >= jnp.uint32(0x6a09e)
    mantissa_bits = (bits & jnp.uint64(0xfffffffffffff)) | jnp.where(
        upper, jnp.uint64(0x3fe0000000000000), jnp.uint64(0x3ff0000000000000)
    )
    u = lax.bitcast_convert_type(mantissa_bits, jnp.float64)
    scale = (high >> jnp.uint32(20)).astype(jnp.int32) + upper.astype(jnp.int32) - 1023
    bigu = jnp.float64(float.fromhex("0x1.7fffffffffd2cp+42"))
    v = add(u, bigu)
    uu = add(v, -bigu)
    index = (lax.bitcast_convert_type(v, jnp.uint64) & jnp.uint64(0x3ff)).astype(jnp.int32)
    ih, il, lh, ll = tuple(_lookup(table, index) for table in _UI)

    eps = add(u, -uu)
    e1, e2 = mul(eps, ih), mul(eps, il)
    e = add(e1, e2)
    e2 = add(add(e1, -e), e2)
    n = scale.astype(jnp.float64)
    t = add(mul(n, jnp.float64(_LN2_HIGH)), lh)
    t1 = add(t, e)
    t2 = add(add(add(add(add(t, -t1), e), mul(n, jnp.float64(_LN2_LOW))), ll), e2)
    q = _polynomial(e, (-.5, 3.3333333333331404e-1, -2.4999999999996436e-1,
                        2.0000010500004459e-1, -1.6666678916688004e-1))
    t2 = add(t2, mul(mul(e, e), q))
    general = add(t1, t2)
    general_tail = add(add(t1, -general), t2)

    def unit_scale_pair():
        bigv = jnp.float64(float.fromhex("0x1.7fffffff8016ap+33"))
        v = add(mul(u, add(ih, il)), bigv)
        vv = add(v, -bigv)
        index2 = (lax.bitcast_convert_type(v, jnp.uint64) & jnp.uint64(0x7ffff)).astype(jnp.int32)
        inverse, log_high, log_low = tuple(_lookup(table, index2) for table in _VJ)
        eps = add(u, -mul(uu, vv))
        e1 = mul(eps, ih)
        e2 = mul(eps, add(il, mul(inverse, add(ih, il))))
        e = add(e1, e2)
        e2 = add(add(e1, -e), e2)
        t = add(lh, log_high)
        t1 = add(t, e)
        t2 = add(add(add(add(t, -t1), e), add(ll, log_low)), e2)
        t2 = add(t2, mul(mul(e, e), _polynomial(e, (-.5, 1. / 3., -.25))))
        unit_scale = add(t1, t2)
        unit_scale_tail = add(add(t1, -unit_scale), t2)

        def near_pair():
            t = add(base, jnp.float64(-1.))
            t1 = add(add(t, jnp.float64(5.e6)), jnp.float64(-5.e6))
            t2 = add(t, -t1)
            e1 = add(t, -mul(mul(jnp.float64(.5), t1), t1))
            r = _polynomial(t, (3.33333333333333333372884096563030e-1,
                                -2.50000000000000000213574153875908e-1,
                                1.99999999999683593814072199830603e-1,
                                -1.66666666666065494878165510225378e-1,
                                1.42857517857114380606360005067609e-1,
                                -1.25000449999974370683775964001702e-1))
            e2 = add(mul(mul(mul(t, t), t), r), -mul(mul(jnp.float64(.5), t2), add(t, t1)))
            near = add(e1, e2)
            near_tail = add(add(e1, -near), e2)
            close = (index > 298) & (index < 302)
            return (jnp.where(close, near, unit_scale),
                    jnp.where(close, near_tail, unit_scale_tail))

        close = (index > 298) & (index < 302)
        return lax.cond(jnp.any(close), near_pair, lambda: (unit_scale, unit_scale_tail))

    def selected_pair():
        unit_scale, unit_scale_tail = unit_scale_pair()
        return (jnp.where(scale == 0, unit_scale, general),
                jnp.where(scale == 0, unit_scale_tail, general_tail))

    return lax.cond(jnp.any(scale == 0), selected_pair, lambda: (general, general_tail))


def _log_pair(base):
    return lax.cond(
        jnp.all(base == jnp.float64(2.)),
        lambda: (
            jnp.full_like(base, float.fromhex("0x1.62e42fefa39efp-1")),
            jnp.full_like(base, float.fromhex("0x1.abc9e3b398000p-56")),
        ),
        lambda: _general_log_pair(base),
    )


def _exp_pair(value, tail):
    three51 = jnp.float64(float.fromhex("0x1.8000000000000p+52"))
    y = fma(value, jnp.float64(float.fromhex("0x1.71547652b82fep+0")), three51)
    power = add(y, -three51)
    t = fma(-power, jnp.float64(_LN2_HIGH), value)
    residual = fma(-power, jnp.float64(_LN2_LOW), tail)
    three33 = jnp.float64(float.fromhex("0x1.8000000000000p+34"))
    y = add(t, three33)
    base = add(y, -three33)
    delta = add(residual, add(t, -base))
    polynomial = fma(delta, jnp.float64(float.fromhex("0x1.5555555555a0fp-3")),
                     jnp.float64(float.fromhex("0x1.00000000004dcp-1")))
    epsilon = fma(mul(delta, delta), polynomial, delta)
    low = (lax.bitcast_convert_type(y, jnp.uint64) & jnp.uint64(0xffffffff)).astype(jnp.uint32)
    i = (((low.astype(jnp.int32) >> jnp.int32(8)) & jnp.int32(-2)) + 356) // 2
    j = (low & jnp.uint32(511)).astype(jnp.int32)
    ch, cl = _lookup_double_double_table(COAR, jnp.clip(i, 0, 355))
    fh, fl = _lookup_double_double_table(FINE, j)
    al = mul(ch, fh)
    bet = fma(cl, fl, fma(ch, fl, mul(cl, fh)))
    rem = fma(al, epsilon, fma(bet, epsilon, bet))
    result = mul(add(al, rem), lax.bitcast_convert_type(
        (power.astype(jnp.int64) + jnp.int64(1023)) << jnp.int64(52), jnp.float64
    ))
    high = ((lax.bitcast_convert_type(value, jnp.uint64) >> jnp.uint64(32))
            & jnp.uint64(0x7fffffff))
    return jnp.where(high <= jnp.uint64(0x3c900000), jnp.float64(1.), result)


def _normal(base, exponent):
    logarithm, tail = _log_pair(base)
    return _normal_with_log_pair(exponent, logarithm, tail)


def _normal_with_log_pair(exponent, logarithm, tail):
    split = jnp.float64(134217729.)
    t = mul(exponent, split)
    y1 = add(t, -add(t, -exponent))
    y2 = add(exponent, -y1)
    t = mul(logarithm, split)
    a1 = add(t, -add(t, -logarithm))
    a2 = add(add(logarithm, -a1), tail)
    a = mul(y1, a1)
    aa = add(mul(y2, a1), mul(exponent, a2))
    high = add(a, aa)
    low = add(add(a, -high), aa)
    return _exp_pair(high, low)


def _reference(base, exponent):
    weibull_inner = (exponent == 3.95) & (base >= 1.e-12) & (base <= 16.)
    weibull_outer = (base == 2.) & (exponent >= -1022.) & (exponent <= 0.)
    ice_impedance = (base == 10.) & (exponent >= -6.) & (exponent <= 0.)
    building_temperature = (
        ((exponent == 3.) | (exponent == 4.)) & (base >= 128.) & (base <= 512.)
    )
    richards_hydraulics = (
        (base >= .01) & (base <= 1.) & (abs(exponent) <= 128.)
    )
    temperature_response = (
        (base >= 1.) & (base <= 8.) & (exponent >= -32.) & (exponent <= 16.)
    )
    valid = ((base >= 1.e-100) & (base <= 1.e100) & jnp.isfinite(base)
             & jnp.isfinite(exponent)
             & ((abs(exponent) <= 2.)
                | weibull_inner | weibull_outer | ice_impedance | building_temperature
                | richards_hydraulics | temperature_response))
    underflow = (base == 2.) & (exponent <= -1075.)
    least_subnormal = (base == 2.) & (exponent > -1075.) & (exponent <= -1074.)
    exact_tiny = underflow | least_subnormal
    normal = _normal(jnp.where(valid, base, 1.), jnp.where(valid, exponent, .333))
    normal = jnp.where(underflow, jnp.float64(0.), normal)
    normal = jnp.where(least_subnormal,
                       lax.bitcast_convert_type(jnp.uint64(1), jnp.float64), normal)
    normal = jnp.where(exponent == 1., base, normal)
    normal = jnp.where(exponent == -1., fortran_divide(jnp.ones_like(base), base), normal)

    def exceptional():
        from clm_jax.physics.positive_power import positive_power
        return jnp.where(valid | exact_tiny, normal, positive_power(base, exponent))

    return lax.cond(jnp.all(valid | exact_tiny), lambda: normal, exceptional)


def _power_of_two_reference(exponent):
    valid = ((exponent >= -1022.) & (exponent <= 1.)) | (exponent == 1.5)
    safe = jnp.where(valid, exponent, jnp.float64(0.))
    normal = _normal_with_log_pair(
        safe, jnp.full_like(safe, float.fromhex("0x1.62e42fefa39efp-1")),
        jnp.full_like(safe, float.fromhex("0x1.abc9e3b398000p-56")),
    )
    underflow = exponent <= -1075.
    least_subnormal = (exponent > -1075.) & (exponent <= -1074.)
    normal = jnp.where(underflow, jnp.float64(0.), normal)
    normal = jnp.where(least_subnormal,
                       lax.bitcast_convert_type(jnp.uint64(1), jnp.float64), normal)
    selected = valid | underflow | least_subnormal

    def exceptional():
        from clm_jax.physics.positive_power import positive_power
        return jnp.where(selected, normal, positive_power(jnp.float64(2.), exponent))

    return lax.cond(jnp.all(selected), lambda: normal, exceptional)


def _gpu_implementation(base, exponent):
    from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch
    return fused_native_batch(
        _reference, base.reshape(-1), exponent.reshape(-1), block_size=128,
        direct_io=True, fold_literals=True, preserve_literal_barriers=True,
        guard_branches=True,
    ).reshape(base.shape)


_power_p = core.Primitive("clm_native_gnu_power")
_power_p.def_impl(
    lambda base, exponent: (_gpu_implementation(base, exponent)
                            if jax.default_backend() == "gpu"
                            else _reference(base, exponent))
)
_power_p.def_abstract_eval(lambda base, exponent: base)
mlir.register_lowering(_power_p, mlir.lower_fun(_gpu_implementation, multiple_results=False))
mlir.register_lowering(
    _power_p, mlir.lower_fun(_reference, multiple_results=False), platform="cpu",
)


def _batch(values, axes):
    if all(a is None for a in axes):
        return _power_p.bind(*values), None
    size = next(v.shape[a] for v, a in zip(values, axes, strict=True) if a is not None)
    arrays = (batching.bdim_at_front(v, a, size)
              for v, a in zip(values, axes, strict=True))
    return _power_p.bind(*arrays), 0


batching.primitive_batchers[_power_p] = _batch


def _primitive_jvp(primals, tangents):
    result = _power_p.bind(*primals)
    if all(isinstance(tangent, ad.Zero) for tangent in tangents):
        return result, ad.Zero.from_primal_value(result)
    tangents = tuple(ad.instantiate_zeros(tangent) for tangent in tangents)
    return result, jax.jvp(fortran_hardware_power, primals, tangents)[1]


ad.primitive_jvps[_power_p] = _primitive_jvp


@jax.custom_jvp
def native_power(base, exponent):
    base, exponent = jnp.broadcast_arrays(jnp.asarray(base, jnp.float64),
                                         jnp.asarray(exponent, jnp.float64))
    if not _native_tile_trace.get():
        return _power_p.bind(base, exponent)
    return _reference(base, exponent)


@native_power.defjvp
def _jvp(primals, tangents):
    return native_power(*primals), jax.jvp(fortran_hardware_power, primals, tangents)[1]


@jax.custom_jvp
def native_power_of_two(exponent):
    exponent = jnp.asarray(exponent, jnp.float64)
    if _native_tile_trace.get():
        return _power_of_two_reference(exponent)
    return native_power(jnp.float64(2.), exponent)


@native_power_of_two.defjvp
def _power_of_two_jvp(primals, tangents):
    return native_power_of_two(primals[0]), jax.jvp(
        lambda exponent: jnp.power(jnp.float64(2.), exponent), primals, tangents,
    )[1]


def _diffusion_reference(base, exponent):
    normal_domain = ((base >= float.fromhex("0x1p-53")) & (base <= 1.)
                     & (exponent >= 0.) & (exponent <= 10. / 3.))
    zero = (base == 0.) & ~jnp.signbit(base) & (exponent > 0.)
    safe_base = jnp.where(normal_domain, base, jnp.float64(1.))
    safe_exponent = jnp.where(normal_domain, exponent, jnp.float64(1.))
    normal = jnp.where(zero, jnp.float64(0.), _normal(safe_base, safe_exponent))
    return lax.cond(jnp.all(normal_domain | zero), lambda: normal,
                    lambda: jnp.where(normal_domain | zero, normal, _reference(base, exponent)))


def _diffusion_gpu(base, exponent):
    from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch
    return fused_native_batch(
        _diffusion_reference, base.reshape(-1), exponent.reshape(-1), block_size=128,
        direct_io=True, fold_literals=True, preserve_literal_barriers=True,
        guard_branches=True,
    ).reshape(base.shape)


_diffusion_p = core.Primitive("clm_native_gnu_diffusion_power")
_diffusion_p.def_impl(lambda base, exponent: (
    _diffusion_gpu(base, exponent) if jax.default_backend() == "gpu"
    else _diffusion_reference(base, exponent)))
_diffusion_p.def_abstract_eval(lambda base, exponent: base)
mlir.register_lowering(_diffusion_p, mlir.lower_fun(_diffusion_gpu, multiple_results=False))
mlir.register_lowering(_diffusion_p, mlir.lower_fun(_diffusion_reference, multiple_results=False),
                      platform="cpu")


def _diffusion_batch(values, axes):
    if all(a is None for a in axes):
        return _diffusion_p.bind(*values), None
    size = next(v.shape[a] for v, a in zip(values, axes, strict=True) if a is not None)
    return _diffusion_p.bind(*(batching.bdim_at_front(v, a, size)
                             for v, a in zip(values, axes, strict=True))), 0


def _diffusion_primitive_jvp(primals, tangents):
    result = _diffusion_p.bind(*primals)
    if all(isinstance(tangent, ad.Zero) for tangent in tangents):
        return result, ad.Zero.from_primal_value(result)
    return result, jax.jvp(fortran_hardware_power, primals,
                           tuple(ad.instantiate_zeros(t) for t in tangents))[1]


batching.primitive_batchers[_diffusion_p] = _diffusion_batch
ad.primitive_jvps[_diffusion_p] = _diffusion_primitive_jvp


@jax.custom_jvp
def native_diffusion_power(base, exponent):
    base, exponent = jnp.broadcast_arrays(jnp.asarray(base, jnp.float64),
                                         jnp.asarray(exponent, jnp.float64))
    return (_diffusion_reference(base, exponent) if _native_tile_trace.get()
            else _diffusion_p.bind(base, exponent))


@native_diffusion_power.defjvp
def _diffusion_jvp(primals, tangents):
    return native_diffusion_power(*primals), jax.jvp(fortran_hardware_power, primals, tangents)[1]
