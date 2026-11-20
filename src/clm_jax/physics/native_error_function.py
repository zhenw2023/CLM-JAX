
import jax
import jax.numpy as jnp
from jax import lax
from jax.extend import core
from jax.interpreters import ad, batching, mlir

from clm_jax.physics.native_exponential import native_exp
from clm_jax.physics.numerics import (
    _rounded_add as add,
    _rounded_multiply as mul,
    fortran_divide as div,
)
from clm_jax.physics.solver_scalarization import _native_tile_trace


_PP = (1.28379167095512558561e-1, -3.25042107247001499370e-1,
       -2.84817495755985104766e-2, -5.77027029648944159157e-3,
       -2.37630166566501626084e-5)
_QQ = (1.0, 3.97917223959155352819e-1, 6.50222499887672944485e-2,
       5.08130628187576562776e-3, 1.32494738004321644526e-4,
       -3.96022827877536812320e-6)
_PA = (-2.36211856075265944077e-3, 4.14856118683748331666e-1,
       -3.72207876035701323847e-1, 3.18346619901161753674e-1,
       -1.10894694282396677476e-1, 3.54783043256182359371e-2,
       -2.16637559486879084300e-3)
_QA = (1.0, 1.06420880400844228286e-1, 5.40397917702171048937e-1,
       7.18286544141962662868e-2, 1.26171219808761642112e-1,
       1.36370839120290507362e-2, 1.19844998467991074170e-2)
_RA = (-9.86494403484714822705e-3, -6.93858572707181764372e-1,
       -1.05586262253232909814e1, -6.23753324503260060396e1,
       -1.62396669462573470355e2, -1.84605092906711035994e2,
       -8.12874355063065934246e1, -9.81432934416914548592)
_SA = (1.0, 1.96512716674392571292e1, 1.37657754143519042600e2,
       4.34565877475229228821e2, 6.45387271733267880336e2,
       4.29008140027567833386e2, 1.08635005541779435134e2,
       6.57024977031928170135, -6.04244152148580987438e-2)
_RB = (-9.86494292470009928597e-3, -7.99283237680523006574e-1,
       -1.77579549177547519889e1, -1.60636384855821916062e2,
       -6.37566443368389627722e2, -1.02509513161107724954e3,
       -4.83519191608651397019e2)
_SB = (1.0, 3.03380607434824582924e1, 3.25792512996573918826e2,
       1.53672958608443695994e3, 3.19985821950859553908e3,
       2.55305040643316442583e3, 4.74528541206955367215e2,
       -2.24409524465858183362e1)


def _paired_polynomial(x, coefficients):
    squared = mul(x, x)
    power = jnp.ones_like(x)
    total = jnp.zeros_like(x)
    for i in range(0, len(coefficients), 2):
        term = jnp.full_like(x, coefficients[i])
        if i + 1 < len(coefficients):
            term = add(term, mul(x, jnp.float64(coefficients[i + 1])))
        product = term if i == 0 else mul(power, term)
        total = product if i == 0 else add(total, product)
        power = squared if i == 0 else (
            mul(squared, squared) if i == 2 else (
                mul(mul(squared, squared), squared) if i == 4
                else mul(mul(squared, squared), mul(squared, squared))
            )
        )
    return total


def _small(value):
    z = mul(value, value)
    ratio = div(_paired_polynomial(z, _PP), _paired_polynomial(z, _QQ))
    result = add(value, mul(value, ratio))
    efx = jnp.float64(1.28379167095512586316e-1)
    tiny = add(value, mul(efx, value))
    subnormal = mul(jnp.float64(0.0625),
                    add(mul(jnp.float64(16.0), value),
                        mul(mul(jnp.float64(16.0), efx), value)))
    high = (lax.bitcast_convert_type(jnp.abs(value), jnp.uint64) >> jnp.uint64(32)).astype(jnp.uint32)
    tiny = jnp.where(high < jnp.uint32(0x00800000), subnormal, tiny)
    return jnp.where(high < jnp.uint32(0x3e300000), tiny, result)


def _middle(value):
    shifted = add(jnp.abs(value), jnp.float64(-1.0))
    ratio = div(_paired_polynomial(shifted, _PA), _paired_polynomial(shifted, _QA))
    constant = jnp.float64(8.45062911510467529297e-1)
    return jnp.where(jnp.signbit(value), add(-constant, -ratio), add(constant, ratio))


def _tail(value):
    magnitude = jnp.maximum(jnp.abs(value), jnp.float64(1.25))
    coordinate = div(jnp.float64(1.0), mul(magnitude, magnitude))
    ratio_a = div(_paired_polynomial(coordinate, _RA), _paired_polynomial(coordinate, _SA))
    ratio_b = div(_paired_polynomial(coordinate, _RB), _paired_polynomial(coordinate, _SB))
    high = (lax.bitcast_convert_type(magnitude, jnp.uint64) >> jnp.uint64(32)).astype(jnp.uint32)
    ratio = jnp.where(high < jnp.uint32(0x4006db6e), ratio_a, ratio_b)
    truncated = lax.bitcast_convert_type(
        (lax.bitcast_convert_type(magnitude, jnp.uint64) >> jnp.uint64(32)) << jnp.uint64(32),
        jnp.float64,
    )
    leading = native_exp(add(-mul(truncated, truncated), jnp.float64(-0.5625)))
    correction = native_exp(add(mul(add(truncated, -magnitude), add(truncated, magnitude)), ratio))
    remainder = div(mul(leading, correction), magnitude)
    return jnp.where(jnp.signbit(value), add(remainder, jnp.float64(-1.0)),
                     add(jnp.float64(1.0), -remainder))


def _reference(value):
    magnitude = jnp.abs(value)
    return lax.cond(
        jnp.all(magnitude < 0.84375), lambda: _small(value),
        lambda: lax.cond(
            jnp.all((magnitude >= 0.84375) & (magnitude < 1.25)), lambda: _middle(value),
            lambda: jnp.where(
                jnp.isnan(value), value + value,
                jnp.where(magnitude >= 6.0, jnp.copysign(jnp.float64(1.0), value),
                          jnp.where(magnitude < 0.84375, _small(value),
                                    jnp.where(magnitude < 1.25, _middle(value), _tail(value)))),
            ),
        ),
    )


def _gpu_implementation(value):
    from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch
    return fused_native_batch(
        _reference, value.reshape(-1), block_size=128, direct_io=True,
        fold_literals=True, preserve_literal_barriers=True, guard_branches=True,
    ).reshape(value.shape)


_erf_p = core.Primitive("clm_native_gnu_erf")
_erf_p.def_impl(_gpu_implementation)
_erf_p.def_abstract_eval(lambda value: value)
mlir.register_lowering(_erf_p, mlir.lower_fun(_gpu_implementation, multiple_results=False))
batching.primitive_batchers[_erf_p] = lambda values, axes: (_erf_p.bind(values[0]), axes[0])


def _derivative(value):
    return jnp.float64(1.12837916709551257390) * native_exp(-value * value)


def _primitive_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    result = _erf_p.bind(value)
    if isinstance(tangent, ad.Zero):
        return result, ad.Zero.from_primal_value(result)
    return result, _derivative(value) * tangent


ad.primitive_jvps[_erf_p] = _primitive_jvp


@jax.custom_jvp
def native_erf(value):
    value = jnp.asarray(value, jnp.float64)
    if jax.default_backend() == "gpu" and not _native_tile_trace.get():
        return _erf_p.bind(value)
    if jax.default_backend() == "cpu":
        from jax.scipy.special import erf
        return erf(value)
    return _reference(value)


@native_erf.defjvp
def _native_erf_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    return native_erf(value), _derivative(value) * tangent
