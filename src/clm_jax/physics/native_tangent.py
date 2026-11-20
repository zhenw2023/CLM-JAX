
import jax
import jax.numpy as jnp

from clm_jax.physics.native_solar_math import _dd_multiply, _elementwise, _two_sum
from clm_jax.physics.numerics import (
    _double_double_add as dd_add,
    _rounded_add as add,
    fortran_divide,
)

_SIN = (
    (1.0, 0.0),
    (-0.16666666666666666, -9.25185853854297e-18),
    (0.008333333333333333, 1.1564823173178714e-19),
    (-0.0001984126984126984, -1.7209558293420705e-22),
    (2.7557319223985893e-06, -1.858393274046472e-22),
    (-2.505210838544172e-08, 1.448814070935912e-24),
    (1.6059043836821613e-10, 1.2585294588752098e-26),
    (-7.647163731819816e-13, -7.03872877733453e-30),
    (2.8114572543455206e-15, 1.6508842730861433e-31),
    (-8.22063524662433e-18, -2.2141894119604265e-34),
    (1.9572941063391263e-20, -1.3643503830087908e-36),
    (-3.868170170630684e-23, 8.843177655482344e-40),
    (6.446950284384474e-26, -1.9330404233703465e-42),
    (-9.183689863795546e-29, -1.4303150396787322e-45),
    (1.1309962886447716e-31, 1.0498015412959506e-47),
)
_COS = (
    (1.0, 0.0),
    (-0.5, 0.0),
    (0.041666666666666664, 2.3129646346357427e-18),
    (-0.001388888888888889, 5.300543954373577e-20),
    (2.48015873015873e-05, 2.1511947866775882e-23),
    (-2.755731922398589e-07, -2.3767714622250297e-23),
    (2.08767569878681e-09, -1.20734505911326e-25),
    (-1.1470745597729725e-11, -2.0655512752830745e-28),
    (4.779477332387385e-14, 4.399205485834081e-31),
    (-1.5619206968586225e-16, -1.1910679660273754e-32),
    (4.110317623312165e-19, 1.4412973378659527e-36),
    (-8.896791392450574e-22, 7.911402614872376e-38),
    (1.6117375710961184e-24, -3.6846573564509766e-41),
    (-2.4795962632247976e-27, 1.2953730964765229e-43),
    (3.279889237069838e-30, 1.5117542744029879e-46),
)
_HALF_PI = float.fromhex("0x1.921fb54442d18p0")
_HALF_PI_TAIL = float.fromhex("0x1.1a62633145c07p-54")
_HALF_PI_THIRD = float.fromhex("-0x1.f1976b7ed8fbcp-110")


def _polynomial(coefficients, square):
    result = tuple(jnp.float64(v) for v in coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        result = dd_add(_dd_multiply(result, square),
                        tuple(jnp.float64(v) for v in coefficient))
    return result


def _reference(x):
    magnitude = jnp.abs(x)
    bounded = jnp.isfinite(x) & (magnitude <= _HALF_PI)
    safe = jnp.where(bounded, magnitude, jnp.float64(.5))
    complement = safe > jnp.float64(_HALF_PI / 2.)
    reduced = _two_sum(safe, -jnp.float64(_HALF_PI))
    reduced = dd_add(reduced, (-jnp.float64(_HALF_PI_TAIL),
                              -jnp.float64(_HALF_PI_THIRD)))
    reduced = (jnp.where(complement, reduced[0], safe),
               jnp.where(complement, reduced[1], jnp.float64(0.)))
    square = _dd_multiply(reduced, reduced)
    sine = _dd_multiply(reduced, _polynomial(_SIN, square))
    cosine = _polynomial(_COS, square)
    numerator = (jnp.where(complement, -cosine[0], sine[0]),
                 jnp.where(complement, -cosine[1], sine[1]))
    denominator = (jnp.where(complement, sine[0], cosine[0]),
                   jnp.where(complement, sine[1], cosine[1]))
    quotient = fortran_divide(numerator[0], denominator[0])
    product = _dd_multiply((quotient, jnp.zeros_like(x)), denominator)
    remainder = dd_add(numerator, (-product[0], -product[1]))
    correction = fortran_divide(add(remainder[0], remainder[1]), denominator[0])
    value = jnp.copysign(add(quotient, correction), x)
    value = jnp.where(magnitude <= float.fromhex("0x1.b096cp-27"), x, value)
    return jnp.where(bounded, value, jnp.tan(x))


_device_tan = _elementwise("clm_native_solar_tan", _reference)


@jax.custom_jvp
def native_tan(x):
    x = jnp.asarray(x, jnp.float64)
    if jax.default_backend() == "cpu":
        return jnp.tan(x)
    return _device_tan(x)


@native_tan.defjvp
def _jvp(primals, tangents):
    x, = primals
    dx, = tangents
    value = native_tan(x)
    return value, (1. + value * value) * dx
