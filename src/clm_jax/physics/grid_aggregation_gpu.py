
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from clm_jax.physics.ordered_gpu_reduction import _first, _instruction


@partial(jax.custom_jvp, nondiff_argnums=(5, 6))
def ordered_scaled_mean(values, weights, active, scale, groups, count, limit, special):
    rows, fields = values.shape
    if rows == 0 or count == 0 or fields == 0:
        return jnp.full((count, fields), special, jnp.float64)
    first = _first(groups, count)
    row_tile, field_tile = 32, 4

    def kernel(vref, wref, aref, sref, gref, fref, special_ref, out):
        group = pl.program_id(0) * row_tile + jnp.arange(row_tile)
        field = pl.program_id(1) * field_tile + jnp.arange(field_tile)
        start = plt.load(fref.at[group], mask=group < count, other=rows)
        sentinel = special_ref[()]

        def accumulate(offset, carry):
            index = start + offset
            in_bounds = (group < count) & (index < rows)
            owner = plt.load(gref.at[index], mask=in_bounds, other=-1)
            enabled = in_bounds & (owner == group)
            enabled &= plt.load(aref.at[index], mask=in_bounds, other=False)
            weight = plt.load(wref.at[index], mask=in_bounds, other=0.0)
            factor = plt.load(sref.at[index], mask=in_bounds, other=0.0)
            value = plt.load(
                vref.at[index[:, None], field[None, :]],
                mask=in_bounds[:, None] & (field[None, :] < fields), other=0.0,
            )
            contributes = (enabled & (weight != 0.0) & (factor != sentinel))[:, None]
            contributes &= (field[None, :] < fields) & (value != sentinel)
            factor = jnp.broadcast_to(factor[:, None], value.shape)
            weight = jnp.broadcast_to(weight[:, None], value.shape)
            product = _instruction("mul", _instruction("mul", value, factor), weight)
            total = _instruction("add", carry[0], product)
            sum_weights = _instruction("add", carry[1], weight)
            return (jnp.where(contributes, total, carry[0]),
                    jnp.where(contributes, sum_weights, carry[1]))

        zero = jnp.zeros((row_tile, field_tile), jnp.float64)
        total, denominator = lax.fori_loop(0, limit, accumulate, (zero, zero))
        contributes = denominator != 0.0
        mean = _instruction("div", total, jnp.where(contributes, denominator, 1.0))
        result = jnp.where(contributes, mean, sentinel)
        plt.store(out.at[group[:, None], field[None, :]], result,
                  mask=(group[:, None] < count) & (field[None, :] < fields))

    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((count, fields), jnp.float64),
        grid=(pl.cdiv(count, row_tile), pl.cdiv(fields, field_tile)),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
    )(values, weights, active, scale, groups, first, jnp.asarray(special, jnp.float64))


@ordered_scaled_mean.defjvp
def _ordered_scaled_mean_jvp(count, limit, primals, tangents):
    values, weights, active, scale, groups, special = primals
    dv, dw, _, ds, _, dsp = tangents
    result = ordered_scaled_mean(values, weights, active, scale, groups, count, limit, special)
    if values.size == 0 or count == 0:
        return result, jnp.full_like(result, dsp)
    from clm_jax.physics.ordered_linear_reduction import linear_ordered_segment_sum
    from clm_jax.physics.numerics import fortran_multiply

    index = jnp.arange(values.shape[0], dtype=jnp.int32)
    member = index < _first(groups, count)[groups] + limit
    included = (member & active & (weights != 0.) & (scale != special))[:, None]
    included &= values != special
    safe = tuple(jnp.where(included, value, 0.)
                 for value in (values, weights[:, None], scale[:, None]))
    directions = tuple(jnp.where(included, value, 0.)
                       for value in (dv, dw[:, None], ds[:, None]))
    _, product_dot = jax.jvp(lambda v,w,s: fortran_multiply(fortran_multiply(v,s),w),
                             safe, directions)
    denominator = linear_ordered_segment_sum(safe[1], groups, count)
    denominator_dot = linear_ordered_segment_sum(directions[1], groups, count)
    total_dot = linear_ordered_segment_sum(product_dot, groups, count)
    contributes = denominator != 0.
    safe_denominator = jnp.where(contributes, denominator, 1.)
    mean = jnp.where(contributes, result, 0.)
    derivative = (total_dot - mean * denominator_dot) / safe_denominator
    return result, jnp.where(contributes, derivative, dsp)
