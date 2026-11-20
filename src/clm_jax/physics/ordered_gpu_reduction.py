
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from clm_jax.physics.numerics import fortran_add, fortran_multiply


def _first(groups, count):
    return (
        jnp.full((count,), groups.size, jnp.int32)
        .at[groups]
        .min(jnp.arange(groups.size, dtype=jnp.int32))
    )


def reference(values, weights, valid, groups, count, limit):
    if values.size == 0 or count == 0:
        return jnp.zeros((count,), jnp.float64), jnp.zeros((count,), jnp.float64)
    first = _first(groups, count)
    group_index = jnp.arange(count, dtype=jnp.int32)

    def step(offset, carry):
        index = first + offset
        safe = jnp.minimum(index, values.size - 1)
        enabled = (index < values.size) & (groups[safe] == group_index) & valid[safe]
        product = jnp.where(enabled, fortran_multiply(values[safe], weights[safe]), 0.0)
        weight = jnp.where(enabled, weights[safe], 0.0)
        return fortran_add(carry[0], product), fortran_add(carry[1], weight)

    zeros = jnp.zeros((count,), jnp.float64)
    return lax.fori_loop(0, limit, step, (zeros, zeros))


def _instruction(name, left, right):
    return plt.elementwise_inline_asm(
        f"{name}.rn.f64 $0, $1, $2;",
        args=[left, right],
        constraints="=d,d,d",
        pack=1,
        result_shape_dtypes=[jax.ShapeDtypeStruct(left.shape, jnp.float64)],
    )[0]


@partial(jax.custom_jvp, nondiff_argnums=(4, 5))
def ordered_sums(values, weights, valid, groups, count, limit):
    if values.size == 0 or count == 0:
        return jnp.zeros((count,), jnp.float64), jnp.zeros((count,), jnp.float64)
    first = _first(groups, count)
    block = 128

    def kernel(vref, wref, mref, gref, fref, total_ref, weight_ref):
        group = pl.program_id(0) * block + jnp.arange(block)
        first_index = plt.load(fref.at[group], mask=group < count, other=values.size)

        def step(offset, carry):
            index = first_index + offset
            in_bounds = (group < count) & (index < values.size)
            owns = plt.load(gref.at[index], mask=in_bounds, other=-1) == group
            active = plt.load(mref.at[index], mask=in_bounds, other=False)
            enabled = in_bounds & owns & active
            value = plt.load(vref.at[index], mask=enabled, other=0.0)
            weight = plt.load(wref.at[index], mask=enabled, other=0.0)
            product = _instruction("mul", value, weight)
            total = _instruction("add", carry[0], product)
            sum_weight = _instruction("add", carry[1], weight)
            return jnp.where(total == 0.0, 0.0, total), jnp.where(
                sum_weight == 0.0, 0.0, sum_weight
            )

        zeros = jnp.zeros((block,), jnp.float64)
        totals, sum_weights = lax.fori_loop(0, limit, step, (zeros, zeros))
        plt.store(total_ref.at[group], totals, mask=group < count)
        plt.store(weight_ref.at[group], sum_weights, mask=group < count)

    shape = jax.ShapeDtypeStruct((count,), jnp.float64)
    return pl.pallas_call(
        kernel,
        out_shape=(shape, shape),
        grid=(pl.cdiv(count, block),),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
    )(values, weights, valid, groups, first)


@ordered_sums.defjvp
def ordered_sums_jvp(count, limit, primals, tangents):
    values, weights, valid, groups = primals
    result = ordered_sums(values, weights, valid, groups, count, limit)
    if values.size == 0 or count == 0:
        return result, tuple(jnp.zeros_like(value) for value in result)
    from clm_jax.physics.ordered_linear_reduction import linear_ordered_segment_sum

    index = jnp.arange(values.size, dtype=jnp.int32)
    included = valid & (index < _first(groups, count)[groups] + limit)
    safe_values, safe_weights = (jnp.where(included, value, 0.) for value in (values, weights))
    safe_tangents = tuple(jnp.where(included, tangent, 0.) for tangent in tangents[:2])
    _, product_dot = jax.jvp(fortran_multiply, (safe_values, safe_weights), safe_tangents)
    total_dot = linear_ordered_segment_sum(jnp.where(included, product_dot, 0.), groups, count)
    weight_dot = linear_ordered_segment_sum(jnp.where(included, tangents[1], 0.), groups, count)
    return result, (total_dot, weight_dot)


@partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def ordered_unweighted_sum(values, groups, count, limit):
    if values.size == 0 or count == 0:
        return jnp.zeros((count,), jnp.float64)
    first, block = _first(groups, count), 128

    def kernel(vref, gref, fref, out):
        group = pl.program_id(0) * block + jnp.arange(block)
        start = plt.load(fref.at[group], mask=group < count, other=values.size)

        def step(offset, total):
            index = start + offset
            valid = (group < count) & (index < values.size)
            owner = plt.load(gref.at[index], mask=valid, other=-1)
            valid &= owner == group
            value = plt.load(vref.at[index], mask=valid, other=0.0)
            return jnp.where(valid, _instruction("add", total, value), total)

        result = lax.fori_loop(0, limit, step, jnp.zeros((block,), jnp.float64))
        plt.store(out.at[group], result, mask=group < count)

    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((count,), jnp.float64),
        grid=(pl.cdiv(count, block),),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
    )(values, groups, first)


@ordered_unweighted_sum.defjvp
def _unweighted_jvp(count, limit, primals, tangents):
    values, groups = primals
    result = ordered_unweighted_sum(values, groups, count, limit)
    if values.size == 0 or count == 0:
        return result, jnp.zeros_like(result)
    from clm_jax.physics.ordered_linear_reduction import linear_ordered_segment_sum
    included = jnp.arange(values.size) < _first(groups, count)[groups] + limit
    return result, linear_ordered_segment_sum(jnp.where(included, tangents[0], 0.), groups, count)
