
import jax
import jax.numpy as jnp
from jax import lax
from jax.extend import core
from jax.interpreters import ad, batching, mlir

from clm_jax.physics import native_arctangent_tables as tables
from clm_jax.physics.arctangent import _arctangent_reference
from clm_jax.physics.native_exponential import fma
from clm_jax.physics.numerics import (
    _rounded_add as add,
    _rounded_multiply as mul,
    fortran_divide,
)
from clm_jax.physics.solver_scalarization import _native_tile_trace


_COLUMNS = tuple(tuple(row[i] for row in tables.CIJ) for i in range(7))


def _lookup(index, column):
    if _native_tile_trace.get():
        from clm_jax.physics.solver_constant_lookup import constant_lookup
        return constant_lookup(_COLUMNS[column], index)
    return jnp.asarray(_COLUMNS[column], jnp.float64)[index]


def _small_correction(x):
    squared = mul(x, x)
    polynomial = jnp.full_like(x, tables.D13)
    for coefficient in (tables.D11, tables.D9, tables.D7, tables.D5, tables.D3):
        polynomial = fma(squared, polynomial, jnp.float64(coefficient))
    return mul(polynomial, mul(x, squared))


def _certificate(leading, correction, error):
    lower = add(leading, add(correction, -error))
    upper = add(leading, add(correction, error))
    return lower, lower == upper


def _fast(value):
    magnitude = jnp.abs(value)
    safe = jnp.where(jnp.isfinite(magnitude), magnitude, jnp.float64(1.))
    inverse = fortran_divide(jnp.float64(1.), jnp.maximum(safe, jnp.float64(1.)))
    product = mul(inverse, safe)
    product_tail = fma(inverse, safe, -product)
    inverse_tail = mul(inverse, add(add(jnp.float64(1.), -product), -product_tail))
    reflected = safe >= jnp.float64(1.)
    reduced = jnp.where(reflected, inverse, safe)
    index = jnp.clip(jnp.rint(mul(reduced, jnp.float64(256.))).astype(jnp.int32) - 16, 0, 240)
    z = add(reduced, -_lookup(index, 0))
    z = jnp.where(reflected, add(z, inverse_tail), z)
    polynomial = _lookup(index, 6)
    for column in (5, 4, 3, 2):
        polynomial = fma(z, polynomial, _lookup(index, column))
    angle = _lookup(index, 1)
    correction = mul(z, polynomial)
    small_bound = jnp.where(index < 48, tables.U21,
                           jnp.where(index < 112, tables.U22,
                                     jnp.where(index < 176, tables.U23, tables.U24)))
    table_value, table_valid = _certificate(angle, correction, mul(small_bound, angle))
    reflected_value, reflected_valid = _certificate(
        add(jnp.float64(tables.HPI), -angle),
        fma(-z, polynomial, jnp.float64(tables.HPI1)),
        jnp.where(index < 112, tables.U31, tables.U32),
    )
    result = jnp.where(reflected, reflected_value, table_value)
    valid = jnp.where(reflected, reflected_valid, table_valid)

    small_value, small_valid = _certificate(
        safe, _small_correction(safe), mul(jnp.float64(tables.U1), safe),
    )
    high = add(jnp.float64(tables.HPI), -inverse)
    high_tail = add(add(jnp.float64(tables.HPI), -high), -inverse)
    large_value, large_valid = _certificate(
        high,
        add(add(add(jnp.float64(tables.HPI1), high_tail), -inverse_tail),
            -_small_correction(inverse)),
        jnp.float64(tables.U4),
    )
    result = jnp.where(safe < .0625, small_value, result)
    valid = jnp.where(safe < .0625, small_valid, valid)
    result = jnp.where(safe >= 16., large_value, result)
    valid = jnp.where(safe >= 16., large_valid, valid)
    result = jnp.where(magnitude >= tables.E, jnp.float64(tables.HPI), result)
    valid = jnp.where(magnitude >= tables.E, True, valid)
    result = jnp.copysign(result, value)
    result = jnp.where(magnitude < tables.A, value, result)
    valid = jnp.where(magnitude < tables.A, True, valid)
    result = jnp.where(jnp.isnan(value), add(value, value), result)
    return result, valid | jnp.isnan(value)


def _positive_surface_angle(value):
    inverse = fortran_divide(jnp.float64(1.), value)
    product = mul(inverse, value)
    tail = mul(inverse, add(add(jnp.float64(1.), -product),
                           -fma(inverse, value, -product)))
    index = jnp.clip(jnp.rint(mul(inverse, jnp.float64(256.))).astype(jnp.int32) - 16,
                     0, 240)
    z = add(add(inverse, -_lookup(index, 0)), tail)
    polynomial = _lookup(index, 6)
    for column in (5, 4, 3, 2):
        polynomial = fma(z, polynomial, _lookup(index, column))
    return _certificate(
        add(jnp.float64(tables.HPI), -_lookup(index, 1)),
        fma(-z, polynomial, jnp.float64(tables.HPI1)),
        jnp.where(index < 112, tables.U31, tables.U32),
    )


def _general_reference(value):
    fast, certified = _fast(value)
    return lax.cond(
        jnp.all(certified),
        lambda: fast,
        lambda: jnp.where(certified, fast, _arctangent_reference(value)),
    )


def _reference(value):
    def surface():
        fast, certified = _positive_surface_angle(value)
        return lax.cond(jnp.all(certified), lambda: fast,
                        lambda: jnp.where(certified, fast, _arctangent_reference(value)))

    return lax.cond(jnp.all((value >= 1.) & (value <= 16.)),
                    surface, lambda: _general_reference(value))


def _gpu_implementation(value):
    from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch
    from clm_jax.physics.solver_fma import native_exp_residual

    def point(x):
        token = native_exp_residual.set(True)
        try:
            return _reference(x)
        finally:
            native_exp_residual.reset(token)

    return fused_native_batch(
        point, value.reshape(-1), block_size=128, direct_io=True,
        fold_literals=True, preserve_literal_barriers=True, guard_branches=True,
    ).reshape(value.shape)


_atan_p = core.Primitive("clm_native_certified_atan")
_atan_p.def_impl(_gpu_implementation)
_atan_p.def_abstract_eval(lambda value: value)
mlir.register_lowering(_atan_p, mlir.lower_fun(_gpu_implementation, multiple_results=False))
batching.primitive_batchers[_atan_p] = lambda values, axes: (_atan_p.bind(values[0]), axes[0])


def _primitive_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    result = _atan_p.bind(value)
    if isinstance(tangent, ad.Zero):
        return result, ad.Zero.from_primal_value(result)
    return result, tangent / (1.0 + value * value)


ad.primitive_jvps[_atan_p] = _primitive_jvp


@jax.custom_jvp
def native_atan(value):
    value = jnp.asarray(value, jnp.float64)
    if jax.default_backend() == "gpu" and not _native_tile_trace.get():
        return _atan_p.bind(value)
    return _reference(value)


@native_atan.defjvp
def _native_atan_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    return native_atan(value), tangent / (1. + value * value)
