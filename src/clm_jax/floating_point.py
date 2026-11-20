
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from contextvars import ContextVar

_native_tile_rounding = ContextVar("clm_native_solver_ptx_rounding", default=False)

_DEKKER_SPLITTER = 134217729.0


@jax.custom_jvp
def binary64_ldexp(value: jnp.ndarray, exponent: jnp.ndarray) -> jnp.ndarray:
    value, exponent = jnp.broadcast_arrays(jnp.asarray(value, jnp.float64),
                                           jnp.asarray(exponent, jnp.int32))
    bits = lax.bitcast_convert_type(value, jnp.uint64)
    sign = (bits >> jnp.uint64(63)) << jnp.uint64(63)
    fraction = bits & jnp.uint64(0x000fffffffffffff)
    stored_exponent = ((bits >> jnp.uint64(52)) & jnp.uint64(0x7ff)).astype(jnp.int32)
    subnormal = stored_exponent == 0
    mantissa, sub_exponent = jnp.frexp(fraction.astype(jnp.float64))
    sub_bits = lax.bitcast_convert_type(mantissa * 2., jnp.uint64)
    significand = jnp.where(subnormal, sub_bits, bits) & jnp.uint64(0x000fffffffffffff)
    significand |= jnp.uint64(0x0010000000000000)
    unbiased = jnp.where(subnormal, sub_exponent - 1075, stored_exponent - 1023)
    target = unbiased + jnp.clip(exponent, -4096, 4096)
    normal_bits = ((jnp.clip(target + 1023, 1, 2046).astype(jnp.uint64) << jnp.uint64(52))
                   | (significand & jnp.uint64(0x000fffffffffffff)))
    shift = jnp.clip(-1022 - target, 1, 53).astype(jnp.uint64)
    quotient = significand >> shift
    remainder = significand & ((jnp.uint64(1) << shift) - jnp.uint64(1))
    half = jnp.uint64(1) << (shift - jnp.uint64(1))
    rounded = quotient + ((remainder > half) | ((remainder == half)
                            & ((quotient & jnp.uint64(1)) != 0))).astype(jnp.uint64)
    magnitude = jnp.where(target < -1022, rounded, normal_bits)
    magnitude = jnp.where(target < -1075, jnp.uint64(0), magnitude)
    magnitude = jnp.where(target > 1023, jnp.uint64(0x7ff0000000000000), magnitude)
    special = (stored_exponent == 2047) | ((stored_exponent == 0) & (fraction == 0))
    return lax.bitcast_convert_type(jnp.where(special, bits, sign | magnitude), jnp.float64)


@binary64_ldexp.defjvp
def _binary64_ldexp_jvp(primals, tangents):
    value, exponent = primals
    return binary64_ldexp(value, exponent), tangents[0] * binary64_ldexp(jnp.ones_like(value), exponent)


def _rounded_add(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    left, right = jnp.broadcast_arrays(
        jnp.asarray(left, dtype=jnp.float64),
        jnp.asarray(right, dtype=jnp.float64),
    )
    result = left + right
    return jnp.copysign(jnp.abs(result), result)


def _rounded_multiply(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    left, right = jnp.broadcast_arrays(
        jnp.asarray(left, dtype=jnp.float64),
        jnp.asarray(right, dtype=jnp.float64),
    )
    result = left * right
    return jnp.copysign(jnp.abs(result), result)


def _two_sum(left: jnp.ndarray, right: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    total = _rounded_add(left, right)
    right_virtual = _rounded_add(total, -left)
    left_error = _rounded_add(left, -_rounded_add(total, -right_virtual))
    right_error = _rounded_add(right, -right_virtual)
    return total, _rounded_add(left_error, right_error)


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
    left_high, left_low = _split(left)
    right_high, right_low = _split(right)
    error = _rounded_add(_rounded_multiply(left_high, right_high), -product)
    error = _rounded_add(error, _rounded_multiply(left_high, right_low))
    error = _rounded_add(error, _rounded_multiply(left_low, right_high))
    error = _rounded_add(error, _rounded_multiply(left_low, right_low))
    return product, error


@jax.custom_jvp
def correctly_rounded_divide(
    numerator: jnp.ndarray,
    denominator: jnp.ndarray,
) -> jnp.ndarray:

    numerator, denominator = jnp.broadcast_arrays(
        jnp.asarray(numerator, dtype=jnp.float64),
        jnp.asarray(denominator, dtype=jnp.float64),
    )
    if jax.default_backend() == "gpu":
        numerator, denominator = lax.optimization_barrier((numerator, denominator))
        quotient = numerator / denominator
        if _native_tile_rounding.get():
            return quotient
        return jnp.copysign(jnp.abs(quotient), quotient)
    quotient = lax.optimization_barrier(numerator / denominator)
    product, product_error = _two_product(quotient, denominator)
    residual_high, residual_error = _two_sum(numerator, -product)
    residual = _rounded_add(
        residual_high,
        _rounded_add(residual_error, -product_error),
    )
    correction = lax.optimization_barrier(residual / denominator)
    corrected = _rounded_add(quotient, correction)
    return jnp.where(jnp.isfinite(quotient), corrected, quotient)


@correctly_rounded_divide.defjvp
def _correctly_rounded_divide_jvp(primals, tangents):
    numerator, denominator = primals
    numerator_tangent, denominator_tangent = tangents
    quotient = correctly_rounded_divide(numerator, denominator)
    tangent = (numerator_tangent - quotient * denominator_tangent) / denominator
    return quotient, tangent


def _round_nonnegative_count_to_uint64(
    high: jnp.ndarray,
    low: jnp.ndarray,
) -> jnp.ndarray:

    floor_count = jnp.floor(high)
    fraction = _rounded_add(_rounded_add(high, -floor_count), low)
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
    return floor_bits + round_up.astype(jnp.uint64)


@jax.custom_jvp
def nonnegative_divide_preserving_subnormal(
    numerator: jnp.ndarray,
    denominator: jnp.ndarray,
) -> jnp.ndarray:

    numerator, denominator = jnp.broadcast_arrays(
        jnp.asarray(numerator, dtype=jnp.float64),
        jnp.asarray(denominator, dtype=jnp.float64),
    )
    if jax.default_backend() == "gpu":
        return correctly_rounded_divide(numerator, denominator)
    ordinary = correctly_rounded_divide(numerator, denominator)

    numerator_bits = lax.bitcast_convert_type(numerator, jnp.uint64)
    denominator_bits = lax.bitcast_convert_type(denominator, jnp.uint64)
    exponent_mask = jnp.uint64(0x7FF0000000000000)
    fraction_mask = jnp.uint64(0x000FFFFFFFFFFFFF)
    sign_mask = jnp.uint64(0x8000000000000000)
    normalized_exponent = jnp.uint64(1023) << jnp.uint64(52)
    numerator_exponent_bits = (
        (numerator_bits & exponent_mask) >> jnp.uint64(52)
    )
    denominator_exponent_bits = (
        (denominator_bits & exponent_mask) >> jnp.uint64(52)
    )
    normal_positive_operands = (
        (numerator_exponent_bits > jnp.uint64(0))
        & (numerator_exponent_bits < jnp.uint64(0x7FF))
        & (denominator_exponent_bits > jnp.uint64(0))
        & (denominator_exponent_bits < jnp.uint64(0x7FF))
        & ((numerator_bits & sign_mask) == jnp.uint64(0))
        & ((denominator_bits & sign_mask) == jnp.uint64(0))
    )
    numerator_fraction = numerator_bits & fraction_mask
    subnormal_positive_numerator = (
        (numerator_exponent_bits == jnp.uint64(0))
        & (numerator_fraction != jnp.uint64(0))
        & (denominator_exponent_bits > jnp.uint64(0))
        & (denominator_exponent_bits < jnp.uint64(0x7FF))
        & ((numerator_bits & sign_mask) == jnp.uint64(0))
        & ((denominator_bits & sign_mask) == jnp.uint64(0))
    )

    numerator_mantissa = lax.bitcast_convert_type(
        numerator_fraction | normalized_exponent,
        jnp.float64,
    )
    denominator_mantissa = lax.bitcast_convert_type(
        (denominator_bits & fraction_mask) | normalized_exponent,
        jnp.float64,
    )
    safe_numerator_mantissa = jnp.where(
        normal_positive_operands,
        numerator_mantissa,
        jnp.float64(1.0),
    )
    safe_denominator_mantissa = jnp.where(
        normal_positive_operands,
        denominator_mantissa,
        jnp.float64(1.0),
    )

    quotient_high = correctly_rounded_divide(
        safe_numerator_mantissa,
        safe_denominator_mantissa,
    )
    product, product_error = _two_product(
        quotient_high,
        safe_denominator_mantissa,
    )
    residual_high, residual_error = _two_sum(
        safe_numerator_mantissa,
        -product,
    )
    residual = _rounded_add(
        residual_high,
        _rounded_add(residual_error, -product_error),
    )
    quotient_low = correctly_rounded_divide(
        residual,
        safe_denominator_mantissa,
    )

    exponent_difference = (
        numerator_exponent_bits.astype(jnp.int32)
        - denominator_exponent_bits.astype(jnp.int32)
    )
    subnormal_shift = exponent_difference + jnp.int32(1074)
    possible_subnormal = (
        normal_positive_operands
        & (subnormal_shift >= jnp.int32(-2))
        & (subnormal_shift <= jnp.int32(53))
    )
    safe_shift = jnp.where(possible_subnormal, subnormal_shift, jnp.int32(0))
    scaled_high = jnp.ldexp(quotient_high, safe_shift)
    scaled_low = jnp.ldexp(quotient_low, safe_shift)
    result_bits = _round_nonnegative_count_to_uint64(scaled_high, scaled_low)
    minimum_normal_bits = jnp.uint64(0x0010000000000000)
    reconstructed = lax.bitcast_convert_type(result_bits, jnp.float64)
    use_reconstructed = (
        possible_subnormal & (result_bits <= minimum_normal_bits)
    )

    safe_subnormal_count = jnp.where(
        subnormal_positive_numerator,
        numerator_fraction.astype(jnp.float64),
        jnp.float64(1.0),
    )
    safe_subnormal_denominator = jnp.where(
        subnormal_positive_numerator,
        denominator_mantissa,
        jnp.float64(1.0),
    )
    subnormal_high = correctly_rounded_divide(
        safe_subnormal_count,
        safe_subnormal_denominator,
    )
    subnormal_product, subnormal_product_error = _two_product(
        subnormal_high,
        safe_subnormal_denominator,
    )
    subnormal_residual_high, subnormal_residual_error = _two_sum(
        safe_subnormal_count,
        -subnormal_product,
    )
    subnormal_residual = _rounded_add(
        subnormal_residual_high,
        _rounded_add(subnormal_residual_error, -subnormal_product_error),
    )
    subnormal_low = correctly_rounded_divide(
        subnormal_residual,
        safe_subnormal_denominator,
    )
    denominator_exponent = (
        denominator_exponent_bits.astype(jnp.int32) - jnp.int32(1023)
    )
    subnormal_scale = -denominator_exponent
    possible_subnormal_numerator = (
        subnormal_positive_numerator & (subnormal_scale <= jnp.int32(53))
    )
    safe_subnormal_scale = jnp.where(
        possible_subnormal_numerator,
        subnormal_scale,
        jnp.int32(0),
    )
    subnormal_high = jnp.ldexp(subnormal_high, safe_subnormal_scale)
    subnormal_low = jnp.ldexp(subnormal_low, safe_subnormal_scale)
    subnormal_result_bits = _round_nonnegative_count_to_uint64(
        subnormal_high,
        subnormal_low,
    )
    subnormal_reconstructed = lax.bitcast_convert_type(
        subnormal_result_bits,
        jnp.float64,
    )
    use_subnormal_numerator = (
        possible_subnormal_numerator
        & (subnormal_result_bits <= minimum_normal_bits)
    )
    result = jnp.where(use_reconstructed, reconstructed, ordinary)
    return jnp.where(
        use_subnormal_numerator,
        subnormal_reconstructed,
        result,
    )


@nonnegative_divide_preserving_subnormal.defjvp
def _nonnegative_divide_preserving_subnormal_jvp(primals, tangents):
    numerator, denominator = primals
    numerator_tangent, denominator_tangent = tangents
    quotient = nonnegative_divide_preserving_subnormal(numerator, denominator)
    tangent = (
        numerator_tangent - quotient * denominator_tangent
    ) / denominator
    return quotient, tangent
