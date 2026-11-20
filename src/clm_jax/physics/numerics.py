
from __future__ import annotations

from contextvars import ContextVar
from functools import partial, wraps

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.floating_point import _native_tile_rounding, binary64_ldexp, correctly_rounded_divide
from clm_jax.physics.exponential_coefficients import (
    EXP_COEFFICIENTS,
    EXP_REDUCTION_TABLE,
)
from clm_jax.physics.pointwise import scalar_map
from clm_jax.physics.solver_fma import native_exp_residual
from clm_jax.physics.adjoint_scaling import sensitivity_product
_native_transcendental_zero = ContextVar("clm_native_transcendental_zero", default=False)


def _transcendental_primal(function):
    @wraps(function)
    def evaluate(*args, **kwargs):
        if not _native_tile_rounding.get():
            return function(*args, **kwargs)
        token = _native_transcendental_zero.set(True)
        try:
            return function(*args, **kwargs)
        finally:
            _native_transcendental_zero.reset(token)

    return evaluate

_DEKKER_SPLITTER = 134217729.0
_LN2_HIGH = 0.6931471805599453
_LN2_LOW = 2.3190468138462996e-17
_INVERSE_LN2 = 1.4426950408889634
_EXPM1_OVERFLOW_THRESHOLD = 7.09782712893383973096e02
_EXPM1_LN2_HIGH = 6.93147180369123816490e-01
_EXPM1_LN2_LOW = 1.90821492927058770002e-10
_EXPM1_INVERSE_LN2 = 1.44269504088896338700e00
_EXPM1_Q1 = -3.33333333333331316428e-02
_EXPM1_Q2 = 1.58730158725481460165e-03
_EXPM1_Q3 = -7.93650757867487942473e-05
_EXPM1_Q4 = 4.00821782732936239552e-06
_EXPM1_Q5 = -2.01099218183624371326e-07


@jax.custom_jvp
def _rounded_add(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:

    left, right = jnp.broadcast_arrays(
        jnp.asarray(left, dtype=jnp.float64),
        jnp.asarray(right, dtype=jnp.float64),
    )
    value = left + right
    if _native_tile_rounding.get():
        return value
    return jnp.copysign(jnp.abs(value), value)


def _real_arithmetic_tangent(tangent, primal):
    if getattr(tangent, "dtype", None) == jax.dtypes.float0:
        return jnp.zeros_like(primal)
    return jnp.asarray(tangent, dtype=jnp.float64)


@_rounded_add.defjvp
def _rounded_add_jvp(primals, tangents):
    left, right = jnp.broadcast_arrays(
        *(jnp.asarray(value, dtype=jnp.float64) for value in primals)
    )
    left_dot, right_dot = (
        _real_arithmetic_tangent(tangent, primal)
        for tangent, primal in zip(tangents, (left, right), strict=True)
    )
    return _rounded_add(left, right), left_dot + right_dot


@jax.custom_jvp
def _rounded_multiply(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:

    left, right = jnp.broadcast_arrays(
        jnp.asarray(left, dtype=jnp.float64),
        jnp.asarray(right, dtype=jnp.float64),
    )
    value = left * right
    if _native_tile_rounding.get():
        return value
    return jnp.copysign(jnp.abs(value), value)


@partial(_rounded_multiply.defjvp, symbolic_zeros=True)
def _rounded_multiply_jvp(primals, tangents):
    left, right = jnp.broadcast_arrays(
        *(jnp.asarray(value, dtype=jnp.float64) for value in primals)
    )
    result = _rounded_multiply(left, right)
    tangent = _product_direction((left, right), tangents, result)
    return result, tangent


def _refined_divide(numerator: jnp.ndarray, denominator: jnp.ndarray) -> jnp.ndarray:

    return correctly_rounded_divide(numerator, denominator)


def _two_sum(left: jnp.ndarray, right: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:

    total = _rounded_add(left, right)
    right_virtual = _rounded_add(total, -left)
    left_error = _rounded_add(
        left,
        -_rounded_add(total, -right_virtual),
    )
    right_error = _rounded_add(right, -right_virtual)
    return total, _rounded_add(left_error, right_error)


def _quick_two_sum(
    larger: jnp.ndarray,
    smaller: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    total = _rounded_add(larger, smaller)
    error = _rounded_add(smaller, -_rounded_add(total, -larger))
    return total, error


def _split(value: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:

    scaled = _rounded_multiply(jnp.float64(_DEKKER_SPLITTER), value)
    large = _rounded_add(scaled, -value)
    high = _rounded_add(scaled, -large)
    return high, _rounded_add(value, -high)


def _two_product(
    left: jnp.ndarray,
    right: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    product = _rounded_multiply(left, right)
    if native_exp_residual.get():
        from clm_jax.physics.solver_fma import product_residual
        return product, product_residual(left, right, product)
    left_high, left_low = _split(left)
    right_high, right_low = _split(right)
    error = _rounded_add(_rounded_multiply(left_high, right_high), -product)
    error = _rounded_add(error, _rounded_multiply(left_high, right_low))
    error = _rounded_add(error, _rounded_multiply(left_low, right_high))
    error = _rounded_add(error, _rounded_multiply(left_low, right_low))
    return product, error


def _rounded_fused_multiply_add(
    left: jnp.ndarray,
    right: jnp.ndarray,
    addend: jnp.ndarray,
) -> jnp.ndarray:

    product = _two_product(left, right)
    result, _ = _double_double_add(
        product,
        (jnp.asarray(addend, dtype=jnp.float64), jnp.float64(0.0)),
    )
    return result


def _double_double_add(
    left: tuple[jnp.ndarray, jnp.ndarray],
    right: tuple[jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:

    high, error = _two_sum(left[0], right[0])
    low, low_error = _two_sum(left[1], right[1])
    error = _rounded_add(error, low)
    high, error = _quick_two_sum(high, error)
    error = _rounded_add(error, low_error)
    return _quick_two_sum(high, error)


def _double_double_multiply(
    left: tuple[jnp.ndarray, jnp.ndarray],
    right: tuple[jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:

    high, error = _two_product(left[0], right[0])
    error = _rounded_add(error, _rounded_multiply(left[0], right[1]))
    error = _rounded_add(error, _rounded_multiply(left[1], right[0]))
    error = _rounded_add(error, _rounded_multiply(left[1], right[1]))
    return _quick_two_sum(high, error)


def _double_double_divide_by_float(
    numerator: tuple[jnp.ndarray, jnp.ndarray],
    denominator: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    quotient_high = _refined_divide(numerator[0], denominator)
    product = _two_product(quotient_high, denominator)
    remainder = _double_double_add(numerator, (-product[0], -product[1]))
    quotient_low = _refined_divide(
        _rounded_add(remainder[0], remainder[1]),
        denominator,
    )
    return _quick_two_sum(quotient_high, quotient_low)


@jax.custom_jvp
def strict_exp(value: jnp.ndarray) -> jnp.ndarray:
    from clm_jax.physics.native_exponential import native_exp
    return native_exp(value)


def _lookup_double_double_table(values, index):
    from clm_jax.physics.solver_scalarization import _native_tile_trace

    if _native_tile_trace.get():
        from clm_jax.physics.solver_constant_lookup import constant_lookup

        return tuple(constant_lookup(tuple(row[column] for row in values), index)
                     for column in (0, 1))
    table = jnp.asarray(values, dtype=jnp.float64)[index]
    return table[..., 0], table[..., 1]


def _ordered_compensated_horner(initial, argument, coefficients, *, unroll):
    from clm_jax.physics.solver_scalarization import _native_tile_trace

    if _native_tile_trace.get():
        def condition(carry):
            return carry[0] < len(coefficients)

        def body(carry):
            index, high, low = carry
            coefficient = _lookup_double_double_table(coefficients, index)
            high, low = _double_double_add(
                _double_double_multiply((high, low), argument), coefficient,
            )
            return index + jnp.int32(1), high, low

        _, high, low = lax.while_loop(condition, body, (jnp.int32(0), *initial))
        return high, low

    def step(total, coefficient):
        return _double_double_add(
            _double_double_multiply(total, argument),
            (coefficient[0], coefficient[1]),
        ), None

    return lax.scan(
        step, initial, jnp.asarray(coefficients, jnp.float64),
        unroll=unroll,
    )[0]


@_transcendental_primal
def _strict_exp_reference(value: jnp.ndarray, value_tail: jnp.ndarray | None = None) -> jnp.ndarray:

    value = jnp.asarray(value, dtype=jnp.float64)
    nearest_power = jnp.floor(
        _rounded_add(
            _rounded_multiply(value, jnp.float64(_INVERSE_LN2)),
            jnp.float64(0.5),
        )
    )
    logarithm_multiple = _double_double_multiply(
        (nearest_power, jnp.zeros_like(value)),
        (
            jnp.full_like(value, _LN2_HIGH),
            jnp.full_like(value, _LN2_LOW),
        ),
    )
    reduced = _double_double_add(
        (value, jnp.zeros_like(value) if value_tail is None else value_tail),
        (-logarithm_multiple[0], -logarithm_multiple[1]),
    )
    table_coordinate = jnp.rint(_rounded_multiply(reduced[0], jnp.float64(256.0)))
    table_index = jnp.clip(table_coordinate.astype(jnp.int32) + 90, 0, 180)
    abscissa = _rounded_multiply(table_coordinate, jnp.float64(1.0 / 256.0))
    local = _double_double_add(reduced, (-abscissa, jnp.zeros_like(value)))
    initial = (
        jnp.full_like(value, EXP_COEFFICIENTS[9][0]),
        jnp.full_like(value, EXP_COEFFICIENTS[9][1]),
    )

    polynomial = _ordered_compensated_horner(
        initial, local, EXP_COEFFICIENTS[:9][::-1],
        unroll=value.size >= 131072 or jax.default_backend() == "gpu",
    )
    table = _lookup_double_double_table(EXP_REDUCTION_TABLE, table_index)
    total = _double_double_multiply(
        polynomial,
        table,
    )
    exponent = nearest_power.astype(jnp.int32)
    scaled_high = binary64_ldexp(total[0], exponent)
    scaled_low = binary64_ldexp(total[1], exponent)
    result, _ = _quick_two_sum(scaled_high, scaled_low)

    subnormal_shift = jnp.clip(
        exponent + jnp.int32(1074),
        jnp.int32(-1),
        jnp.int32(52),
    )
    subnormal_high = binary64_ldexp(total[0], subnormal_shift)
    subnormal_low = binary64_ldexp(total[1], subnormal_shift)
    floor_count = jnp.floor(subnormal_high)
    fraction = _rounded_add(
        _rounded_add(subnormal_high, -floor_count),
        subnormal_low,
    )
    below_zero = fraction < jnp.float64(0.0)
    at_least_one = fraction >= jnp.float64(1.0)
    floor_count = floor_count + jnp.where(
        below_zero,
        jnp.float64(-1.0),
        jnp.where(at_least_one, jnp.float64(1.0), jnp.float64(0.0)),
    )
    fraction = fraction + jnp.where(
        below_zero,
        jnp.float64(1.0),
        jnp.where(at_least_one, jnp.float64(-1.0), jnp.float64(0.0)),
    )
    floor_bits = floor_count.astype(jnp.uint64)
    round_up = (fraction > jnp.float64(0.5)) | (
        (fraction == jnp.float64(0.5))
        & ((floor_bits & jnp.uint64(1)) == jnp.uint64(1))
    )
    subnormal_bits = floor_bits + round_up.astype(jnp.uint64)
    subnormal_bits = jnp.where(
        exponent < jnp.int32(-1075),
        jnp.uint64(0),
        subnormal_bits,
    )
    subnormal_result = lax.bitcast_convert_type(subnormal_bits, jnp.float64)
    return jnp.where(exponent <= jnp.int32(-1022), subnormal_result, result)


@strict_exp.defjvp
def _strict_exp_jvp(primals, tangents):

    (value,) = primals
    (tangent,) = tangents
    result = strict_exp(value)
    return result, result * tangent


def _absolute_high_word(value: jnp.ndarray) -> jnp.ndarray:

    bits = lax.bitcast_convert_type(jnp.asarray(value, dtype=jnp.float64), jnp.uint64)
    return jnp.asarray(
        (bits >> jnp.uint64(32)) & jnp.uint64(0x7FFFFFFF),
        dtype=jnp.uint32,
    )


@jax.custom_jvp
def strict_negative_exp(value: jnp.ndarray) -> jnp.ndarray:

    from clm_jax.physics.native_exponential import native_exp
    return native_exp(value)


@strict_negative_exp.defjvp
def _strict_negative_exp_jvp(primals, tangents):

    (value,) = primals
    (tangent,) = tangents
    result = strict_negative_exp(value)
    return result, result * tangent


@jax.custom_jvp
def strict_acos(value: jnp.ndarray) -> jnp.ndarray:
    from clm_jax.physics.native_acos import native_acos
    return native_acos(jnp.asarray(value, dtype=jnp.float64))


@strict_acos.defjvp
def _strict_acos_jvp(primals, tangents):
    (value,) = primals
    (tangent,) = tangents
    result = strict_acos(value)
    derivative = -jnp.reciprocal(jnp.sqrt(jnp.float64(1.0) - value * value))
    return result, derivative * tangent


@jax.custom_jvp
def strict_expm1(value: jnp.ndarray) -> jnp.ndarray:

    value = jnp.asarray(value, dtype=jnp.float64)
    absolute_high_word = _absolute_high_word(value)
    negative = jnp.signbit(value)
    finite = jnp.isfinite(value)
    very_negative = negative & (absolute_high_word >= jnp.uint32(0x4043687A))
    positive_overflow = (~negative) & (value > jnp.float64(_EXPM1_OVERFLOW_THRESHOLD))

    work_value = jnp.where(
        finite & ~very_negative & ~positive_overflow,
        value,
        jnp.float64(0.0),
    )
    needs_reduction = absolute_high_word > jnp.uint32(0x3FD62E42)
    near_ln2 = absolute_high_word < jnp.uint32(0x3FF0A2B2)

    near_high = jnp.where(
        negative,
        _rounded_add(work_value, jnp.float64(_EXPM1_LN2_HIGH)),
        _rounded_add(work_value, -jnp.float64(_EXPM1_LN2_HIGH)),
    )
    near_low = jnp.where(
        negative,
        -jnp.float64(_EXPM1_LN2_LOW),
        jnp.float64(_EXPM1_LN2_LOW),
    )
    near_k = jnp.where(negative, jnp.int32(-1), jnp.int32(1))

    signed_half = jnp.where(negative, jnp.float64(-0.5), jnp.float64(0.5))
    far_k_float = jnp.trunc(
        _rounded_add(
            _rounded_multiply(jnp.float64(_EXPM1_INVERSE_LN2), work_value),
            signed_half,
        )
    )
    far_k = far_k_float.astype(jnp.int32)
    far_high = _rounded_add(
        work_value,
        -_rounded_multiply(far_k_float, jnp.float64(_EXPM1_LN2_HIGH)),
    )
    far_low = _rounded_multiply(far_k_float, jnp.float64(_EXPM1_LN2_LOW))

    high = jnp.where(near_ln2, near_high, far_high)
    low = jnp.where(near_ln2, near_low, far_low)
    reduction_k = jnp.where(near_ln2, near_k, far_k)
    reduced = _rounded_add(high, -low)
    correction = _rounded_add(_rounded_add(high, -reduced), -low)
    reduced = jnp.where(needs_reduction, reduced, work_value)
    correction = jnp.where(needs_reduction, correction, jnp.float64(0.0))
    k = jnp.where(needs_reduction, reduction_k, jnp.int32(0))

    half_x = _rounded_multiply(jnp.float64(0.5), reduced)
    half_x_squared = _rounded_multiply(reduced, half_x)

    polynomial_1 = _rounded_add(
        jnp.float64(1.0),
        _rounded_multiply(half_x_squared, jnp.float64(_EXPM1_Q1)),
    )
    half_x_fourth = _rounded_multiply(half_x_squared, half_x_squared)
    polynomial_2 = _rounded_add(
        jnp.float64(_EXPM1_Q2),
        _rounded_multiply(half_x_squared, jnp.float64(_EXPM1_Q3)),
    )
    half_x_eighth = _rounded_multiply(half_x_fourth, half_x_fourth)
    polynomial_3 = _rounded_add(
        jnp.float64(_EXPM1_Q4),
        _rounded_multiply(half_x_squared, jnp.float64(_EXPM1_Q5)),
    )
    r1 = _rounded_add(
        _rounded_add(
            polynomial_1,
            _rounded_multiply(half_x_fourth, polynomial_2),
        ),
        _rounded_multiply(half_x_eighth, polynomial_3),
    )
    temporary = _rounded_add(
        jnp.float64(3.0),
        -_rounded_multiply(r1, half_x),
    )
    rational_numerator = _rounded_add(r1, -temporary)
    rational_denominator = _rounded_add(
        jnp.float64(6.0),
        -_rounded_multiply(reduced, temporary),
    )
    error = _rounded_multiply(
        half_x_squared,
        _refined_divide(rational_numerator, rational_denominator),
    )

    zero_k_result = _rounded_add(
        reduced,
        -_rounded_add(
            _rounded_multiply(reduced, error),
            -half_x_squared,
        ),
    )
    scaled_error = _rounded_add(error, -correction)
    scaled_error = _rounded_add(
        _rounded_multiply(reduced, scaled_error),
        -correction,
    )
    scaled_error = _rounded_add(scaled_error, -half_x_squared)

    minus_one_result = _rounded_add(
        _rounded_multiply(
            jnp.float64(0.5),
            _rounded_add(reduced, -scaled_error),
        ),
        jnp.float64(-0.5),
    )
    one_result_low = _rounded_multiply(
        jnp.float64(-2.0),
        _rounded_add(scaled_error, -_rounded_add(reduced, jnp.float64(0.5))),
    )
    one_result_high = _rounded_add(
        jnp.float64(1.0),
        _rounded_multiply(
            jnp.float64(2.0),
            _rounded_add(reduced, -scaled_error),
        ),
    )
    one_result = jnp.where(reduced < jnp.float64(-0.25), one_result_low, one_result_high)

    power_of_two = binary64_ldexp(jnp.ones_like(reduced), k)
    outside_central_result = _rounded_add(
        _rounded_multiply(
            _rounded_add(
                jnp.float64(1.0),
                -_rounded_add(scaled_error, -reduced),
            ),
            power_of_two,
        ),
        jnp.float64(-1.0),
    )
    lower_scale_term = _rounded_add(
        jnp.float64(1.0),
        -binary64_ldexp(jnp.ones_like(reduced), -k),
    )
    lower_scale_result = _rounded_multiply(
        _rounded_add(
            lower_scale_term,
            -_rounded_add(scaled_error, -reduced),
        ),
        power_of_two,
    )
    upper_scale_result = _rounded_multiply(
        _rounded_add(
            _rounded_add(
                reduced,
                -_rounded_add(
                    scaled_error,
                    binary64_ldexp(jnp.ones_like(reduced), -k),
                ),
            ),
            jnp.float64(1.0),
        ),
        power_of_two,
    )
    general_result = jnp.where(
        (k <= jnp.int32(-2)) | (k > jnp.int32(56)),
        outside_central_result,
        jnp.where(k < jnp.int32(20), lower_scale_result, upper_scale_result),
    )
    result = jnp.where(
        k == jnp.int32(0),
        zero_k_result,
        jnp.where(
            k == jnp.int32(-1),
            minus_one_result,
            jnp.where(k == jnp.int32(1), one_result, general_result),
        ),
    )

    tiny_input = absolute_high_word < jnp.uint32(0x3C900000)
    result = jnp.where(tiny_input, value, result)
    result = jnp.where(very_negative, jnp.float64(-1.0), result)
    result = jnp.where(positive_overflow, jnp.float64(jnp.inf), result)
    result = jnp.where(jnp.isposinf(value), value, result)
    result = jnp.where(jnp.isneginf(value), jnp.float64(-1.0), result)
    return jnp.where(jnp.isnan(value), _rounded_add(value, value), result)


@strict_expm1.defjvp
def _strict_expm1_jvp(primals, tangents):
    (value,) = primals
    (tangent,) = tangents
    result = strict_expm1(value)
    return result, _rounded_add(result, jnp.float64(1.0)) * tangent


@jax.custom_jvp
def strict_tanh(value: jnp.ndarray) -> jnp.ndarray:

    value = jnp.asarray(value, dtype=jnp.float64)
    absolute_high_word = _absolute_high_word(value)
    finite_absolute_value = jnp.where(jnp.isfinite(value), jnp.abs(value), jnp.float64(0.0))
    at_least_one = absolute_high_word >= jnp.uint32(0x3FF00000)
    expm1_argument = jnp.where(
        at_least_one,
        _rounded_multiply(jnp.float64(2.0), finite_absolute_value),
        -_rounded_multiply(jnp.float64(2.0), finite_absolute_value),
    )
    exponential_minus_one = strict_expm1(expm1_argument)
    denominator = _rounded_add(exponential_minus_one, jnp.float64(2.0))
    below_one = _refined_divide(-exponential_minus_one, denominator)
    at_or_above_one = _rounded_add(
        jnp.float64(1.0),
        -_refined_divide(jnp.float64(2.0), denominator),
    )
    magnitude = jnp.where(at_least_one, at_or_above_one, below_one)
    magnitude = jnp.where(
        absolute_high_word >= jnp.uint32(0x40360000),
        _rounded_add(jnp.float64(1.0), -jnp.float64(1.0e-300)),
        magnitude,
    )
    signed = jnp.where(jnp.signbit(value), -magnitude, magnitude)
    tiny = absolute_high_word < jnp.uint32(0x3C800000)
    tiny_result = _rounded_multiply(value, _rounded_add(jnp.float64(1.0), value))
    result = jnp.where(tiny, tiny_result, signed)
    result = jnp.where(value == jnp.float64(0.0), value, result)
    result = jnp.where(jnp.isposinf(value), jnp.float64(1.0), result)
    result = jnp.where(jnp.isneginf(value), jnp.float64(-1.0), result)
    return jnp.where(jnp.isnan(value), _rounded_add(value, value), result)


@strict_tanh.defjvp
def _strict_tanh_jvp(primals, tangents):
    (value,) = primals
    (tangent,) = tangents
    result = strict_tanh(value)
    derivative = jnp.float64(1.0) - result * result
    return result, derivative * tangent


def _product_step(_, operands):
    return None, operands[0] * operands[1]


def _add_step(running, term):
    return running + term, None


def _multiply_step(running, factor):
    return running * factor, None


def fortran_products(
    factors: jnp.ndarray,
    coefficients: jnp.ndarray,
) -> jnp.ndarray:

    factors = jnp.asarray(factors, dtype=jnp.float64)
    coefficients = jnp.asarray(coefficients, dtype=jnp.float64)
    factors, coefficients = jnp.broadcast_arrays(factors, coefficients)
    if factors.ndim < 1 or factors.shape[0] < 1:
        raise ValueError("Fortran product sequence must contain at least one term")

    if jax.default_backend() == "gpu":
        return _gpu_rounded_product(factors, coefficients)

    _, products = lax.scan(_product_step, None, (factors, coefficients))
    return products


@jax.custom_jvp
def _gpu_rounded_product(left, right):
    return _rounded_multiply(left, right)


@partial(_gpu_rounded_product.defjvp, symbolic_zeros=True)
def _gpu_rounded_product_jvp(primals, tangents):
    result = _gpu_rounded_product(*primals)
    return result, _product_direction(primals, tangents, result)


@jax.custom_jvp
def _gpu_rounded_sum(left, right):
    return _rounded_add(left, right)


@_gpu_rounded_sum.defjvp
def _gpu_rounded_sum_jvp(primals, tangents):
    return _gpu_rounded_sum(*primals), tangents[0] + tangents[1]


@jax.custom_jvp
def _gpu_assigned_sum(left, right):
    value = _rounded_add(left, right)
    return jnp.where(value == 0, jnp.float64(0), value)


@_gpu_assigned_sum.defjvp
def _gpu_assigned_sum_jvp(primals, tangents):
    return _gpu_assigned_sum(*primals), tangents[0] + tangents[1]


def fortran_product(
    factor: jnp.ndarray,
    coefficient: jnp.ndarray,
) -> jnp.ndarray:

    return fortran_products(factor[None, ...], coefficient[None, ...])[0]


def fortran_add(
    left: jnp.ndarray,
    right: jnp.ndarray,
) -> jnp.ndarray:

    left, right = jnp.broadcast_arrays(
        jnp.asarray(left, dtype=jnp.float64),
        jnp.asarray(right, dtype=jnp.float64),
    )
    if jax.default_backend() == "gpu":
        return _gpu_assigned_sum(left, right)
    terms = jnp.stack((right, jnp.zeros_like(right)), axis=0)

    result, _ = lax.scan(_add_step, left, terms)
    return lax.optimization_barrier(result)


def fortran_multiply(
    left: jnp.ndarray,
    right: jnp.ndarray,
) -> jnp.ndarray:

    left, right = jnp.broadcast_arrays(
        jnp.asarray(left, dtype=jnp.float64),
        jnp.asarray(right, dtype=jnp.float64),
    )
    if jax.default_backend() == "gpu":
        return _gpu_rounded_product(left, right)
    factors = jnp.stack((right, jnp.ones_like(right)), axis=0)

    result, _ = lax.scan(_multiply_step, left, factors)
    return lax.optimization_barrier(result)


def fortran_hardware_multiply(
    left: jnp.ndarray,
    right: jnp.ndarray,
) -> jnp.ndarray:

    return _rounded_multiply(left, right)


def fortran_hardware_add(
    left: jnp.ndarray,
    right: jnp.ndarray,
) -> jnp.ndarray:

    return _rounded_add(left, right)


@jax.custom_jvp
def fortran_materialize(value: jnp.ndarray) -> jnp.ndarray:

    value = jnp.asarray(value, dtype=jnp.float64)
    return lax.optimization_barrier(value)


@fortran_materialize.defjvp
def _fortran_materialize_jvp(primals, tangents):
    (value,) = primals
    (tangent,) = tangents
    return fortran_materialize(value), tangent


@jax.custom_jvp
def fortran_rounding_barrier(value: jnp.ndarray) -> jnp.ndarray:

    value = jnp.asarray(value, dtype=jnp.float64)
    return lax.optimization_barrier(value)


@fortran_rounding_barrier.defjvp
def _fortran_rounding_barrier_jvp(primals, tangents):
    (value,) = primals
    (tangent,) = tangents
    return fortran_rounding_barrier(value), tangent


def fortran_divide(
    numerator: jnp.ndarray,
    denominator: jnp.ndarray,
) -> jnp.ndarray:

    numerator, denominator = jnp.broadcast_arrays(
        jnp.asarray(numerator, dtype=jnp.float64),
        jnp.asarray(denominator, dtype=jnp.float64),
    )
    numerator, denominator = lax.optimization_barrier((numerator, denominator))
    return _refined_divide(numerator, denominator)


@jax.custom_jvp
def fortran_hardware_divide(
    numerator: jnp.ndarray,
    denominator: jnp.ndarray,
) -> jnp.ndarray:

    numerator, denominator = jnp.broadcast_arrays(
        jnp.asarray(numerator, dtype=jnp.float64),
        jnp.asarray(denominator, dtype=jnp.float64),
    )
    value = numerator / denominator
    return jnp.copysign(jnp.abs(value), value)


@fortran_hardware_divide.defjvp
def _fortran_hardware_divide_jvp(primals, tangents):
    numerator, denominator = primals
    numerator_tangent, denominator_tangent = tangents
    quotient = fortran_hardware_divide(numerator, denominator)
    tangent = (numerator_tangent - quotient * denominator_tangent) / denominator
    return quotient, tangent


def fortran_power(base: jnp.ndarray, exponent: jnp.ndarray) -> jnp.ndarray:

    base, exponent = jnp.broadcast_arrays(
        jnp.asarray(base, dtype=jnp.float64),
        jnp.asarray(exponent, dtype=jnp.float64),
    )
    shape = base.shape
    powers = scalar_map(
        lambda operands: jnp.power(operands[0], operands[1]),
        (jnp.reshape(base, (-1,)), jnp.reshape(exponent, (-1,))),
    )
    return jnp.reshape(powers, shape)


@jax.custom_jvp
def fortran_hardware_power(
    base: jnp.ndarray,
    exponent: jnp.ndarray,
) -> jnp.ndarray:

    base, exponent = jnp.broadcast_arrays(
        jnp.asarray(base, dtype=jnp.float64),
        jnp.asarray(exponent, dtype=jnp.float64),
    )
    value = jnp.power(base, exponent)
    return jnp.copysign(jnp.abs(value), value)


def _fortran_hardware_power_jvp(primals, tangents):
    from jax.custom_derivatives import SymbolicZero
    from clm_jax.physics.adjoint_scaling import sensitivity_product

    base, exponent = jnp.broadcast_arrays(
        *(jnp.asarray(value,dtype=jnp.float64) for value in primals))
    result = fortran_hardware_power(*primals)
    terms = []
    if not isinstance(tangents[0],SymbolicZero):
        safe_base = jnp.where(exponent == 0.,1.,base)
        slope = jnp.where(exponent == 0.,0.,exponent*jnp.power(safe_base,exponent-1.))
        terms.append(sensitivity_product(slope,_real_arithmetic_tangent(tangents[0],base)))
    if not isinstance(tangents[1],SymbolicZero):
        slope = result*jnp.log(jnp.where(base == 0.,1.,base))
        terms.append(sensitivity_product(slope,_real_arithmetic_tangent(tangents[1],exponent)))
    tangent = sum(terms,jnp.zeros_like(result))
    return result,tangent


fortran_hardware_power.defjvp(_fortran_hardware_power_jvp,symbolic_zeros=True)


@jax.custom_jvp
def strict_log10(value: jnp.ndarray) -> jnp.ndarray:

    value = jnp.asarray(value, dtype=jnp.float64)
    bits = lax.bitcast_convert_type(value, jnp.uint64)
    magnitude_bits = bits & jnp.uint64(0x7FFFFFFFFFFFFFFF)
    subnormal = (magnitude_bits != 0) & (magnitude_bits < jnp.uint64(0x0010000000000000))
    scaled = jnp.where(
        subnormal,
        _rounded_multiply(value, jnp.float64(1.8014398509481984e16)),
        value,
    )
    scaled_bits = lax.bitcast_convert_type(scaled, jnp.uint64)
    exponent_bits = (scaled_bits >> jnp.uint64(52)) & jnp.uint64(0x7FF)
    exponent = exponent_bits.astype(jnp.int32) - jnp.int32(1023)
    exponent = exponent - jnp.where(subnormal, jnp.int32(54), jnp.int32(0))
    negative_exponent = (exponent < 0).astype(jnp.int32)
    normalized_exponent = (jnp.int32(1023) - negative_exponent).astype(jnp.uint64)
    normalized_bits = (scaled_bits & jnp.uint64(0x000FFFFFFFFFFFFF)) | (
        normalized_exponent << jnp.uint64(52)
    )
    normalized = lax.bitcast_convert_type(normalized_bits, jnp.float64)
    exponent_value = (exponent + negative_exponent).astype(jnp.float64)

    low_term = _rounded_multiply(
        exponent_value,
        jnp.float64(3.69423907715893078616e-13),
    )
    if jax.default_backend() == "gpu":
        from clm_jax.physics.native_logarithm import native_log
        logarithm = native_log(normalized)
    else:
        logarithm = jnp.log(normalized)
    logarithm_term = _rounded_multiply(
        jnp.float64(4.34294481903251816668e-01),
        logarithm,
    )
    partial = _rounded_add(low_term, logarithm_term)
    result = _rounded_add(
        partial,
        _rounded_multiply(
            exponent_value,
            jnp.float64(3.01029995663611771306e-01),
        ),
    )

    result = jnp.where(value == 0.0, -jnp.inf, result)
    result = jnp.where(value < 0.0, jnp.nan, result)
    result = jnp.where(jnp.isposinf(value), value, result)
    return jnp.where(jnp.isnan(value), value, result)


@strict_log10.defjvp
def _strict_log10_jvp(primals, tangents):

    (value,) = primals
    (tangent,) = tangents
    result = strict_log10(value)
    derivative = tangent / (value * jnp.float64(2.302585092994046))
    return result, derivative


def strict_divide(
    numerator: jnp.ndarray,
    denominator: jnp.ndarray,
) -> jnp.ndarray:

    return _refined_divide(numerator, denominator)


@jax.custom_jvp
def fortran_masked_materialized_divide(
    numerator: jnp.ndarray,
    denominator: jnp.ndarray,
    active: jnp.ndarray,
    inactive_value: jnp.ndarray,
) -> jnp.ndarray:

    numerator, denominator, active, inactive_value = jnp.broadcast_arrays(
        jnp.asarray(numerator, dtype=jnp.float64),
        jnp.asarray(denominator, dtype=jnp.float64),
        jnp.asarray(active, dtype=jnp.bool_),
        jnp.asarray(inactive_value, dtype=jnp.float64),
    )
    safe_numerator, safe_denominator = lax.optimization_barrier(
        (
            jnp.where(active, numerator, jnp.float64(0.0)),
            jnp.where(active, denominator, jnp.float64(1.0)),
        )
    )
    quotient = fortran_divide(safe_numerator, safe_denominator)
    return lax.optimization_barrier(jnp.where(active, quotient, inactive_value))


@fortran_masked_materialized_divide.defjvp
def _fortran_masked_materialized_divide_jvp(primals, tangents):
    numerator, denominator, active, inactive_value = primals
    numerator_tangent, denominator_tangent, _, inactive_tangent = tangents
    numerator, denominator, active, inactive_value = jnp.broadcast_arrays(
        numerator, denominator, active, inactive_value
    )
    numerator_tangent = jnp.broadcast_to(numerator_tangent, numerator.shape)
    denominator_tangent = jnp.broadcast_to(denominator_tangent, denominator.shape)
    inactive_tangent = jnp.broadcast_to(inactive_tangent, inactive_value.shape)
    quotient = fortran_masked_materialized_divide(numerator, denominator, active, inactive_value)
    safe_denominator = jnp.where(active, denominator, jnp.float64(1.0))
    active_tangent = (numerator_tangent - quotient * denominator_tangent) / safe_denominator
    quotient_tangent = jnp.where(active, active_tangent, inactive_tangent)
    return quotient, quotient_tangent


_UNROLL_INDEXED_FOLD: bool | None = None


def ordered_indexed_fold(function, initial, count):
    unroll = (
        jax.default_backend() == "gpu" if _UNROLL_INDEXED_FOLD is None else _UNROLL_INDEXED_FOLD
    )
    if unroll:
        for index in range(count):
            initial, _ = function(initial, index)
        return initial
    return lax.scan(function, initial, jnp.arange(count, dtype=jnp.int32))[0]


def fortran_ordered_sum(
    terms: jnp.ndarray,
    *,
    initial: jnp.ndarray | None = None,
) -> jnp.ndarray:

    terms = jnp.asarray(terms, dtype=jnp.float64)
    if terms.ndim < 1 or terms.shape[0] < 1:
        raise ValueError("Fortran sum sequence must contain at least one term")
    if initial is None:
        carry = terms[0]
        remaining = terms[1:]
    else:
        carry = jnp.asarray(initial, dtype=jnp.float64)
        remaining = terms

    if jax.default_backend() == "gpu" and remaining.shape[0] <= 32:
        for index in range(remaining.shape[0]):
            carry = _gpu_rounded_sum(carry, remaining[index])
        return carry
    carry, _ = lax.scan(_add_step, carry, remaining)
    return carry


def fortran_linear_combination(
    factors: jnp.ndarray,
    coefficients: jnp.ndarray,
) -> jnp.ndarray:

    return fortran_ordered_sum(fortran_products(factors, coefficients))


_product_step = jax.custom_jvp(_product_step)
_multiply_step = jax.custom_jvp(_multiply_step)


def _product_direction(primals, tangents, result):
    from jax.custom_derivatives import SymbolicZero
    terms = []
    for coefficient, direction in ((primals[1], tangents[0]), (primals[0], tangents[1])):
        if (isinstance(direction, SymbolicZero)
                or getattr(direction, 'dtype', None) == jax.dtypes.float0):
            continue
        terms.append(sensitivity_product(coefficient, _real_arithmetic_tangent(direction, coefficient)))
    if not terms:
        return jnp.zeros_like(result)
    return terms[0] if len(terms) == 1 else terms[0] + terms[1]


@partial(_product_step.defjvp, symbolic_zeros=True)
def _product_step_jvp(primals, tangents):
    result = _product_step(*primals)
    tangent = _product_direction(primals[1], tangents[1], result[1])
    return result, (None, tangent)


@partial(_multiply_step.defjvp, symbolic_zeros=True)
def _multiply_step_jvp(primals, tangents):
    result = _multiply_step(*primals)
    tangent = _product_direction(primals, tangents, result[0])
    return result, (tangent, None)
