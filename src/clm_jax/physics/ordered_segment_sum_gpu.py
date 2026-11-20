
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt


def ordered_segment_sum_gpu(values, segment_ids, number_of_segments):
    width = math.prod(values.shape[1:])
    size = number_of_segments * width
    if size == 0 or values.shape[0] == 0:
        return jnp.zeros((number_of_segments, *values.shape[1:]), values.dtype)
    order = jnp.argsort(segment_ids, stable=True)
    counts = jnp.bincount(segment_ids, length=number_of_segments)
    starts = jnp.cumsum(counts) - counts
    block = 128

    def kernel(vref, order_ref, count_ref, start_ref, output_ref):
        lane = pl.program_id(0) * block + jnp.arange(block)
        valid_lane = lane < size
        group, feature = lane // width, lane % width
        count = plt.load(count_ref.at[group], mask=valid_lane, other=0)
        start = plt.load(start_ref.at[group], mask=valid_lane, other=0)
        limit = jnp.max(count)

        def accumulate(carry):
            offset, total = carry
            valid = valid_lane & (offset < count)
            index = plt.load(order_ref.at[start + offset], mask=valid, other=0)
            value = plt.load(vref.at[index * width + feature], mask=valid, other=0.0)
            updated = plt.elementwise_inline_asm(
                "add.rn.f64 $0, $1, $2;", args=[total, value],
                constraints="=d,d,d", pack=1,
                result_shape_dtypes=[jax.ShapeDtypeStruct((block,), jnp.float64)],
            )[0]
            return offset + 1, jnp.where(valid, updated, total)

        _, result = lax.while_loop(
            lambda carry: carry[0] < limit, accumulate,
            (jnp.int32(0), jnp.zeros((block,), jnp.float64)),
        )
        plt.store(output_ref.at[lane], result, mask=valid_lane)

    result = pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((size,), jnp.float64),
        grid=(pl.cdiv(size, block),),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
    )(values.reshape(-1), order, counts, starts)
    return result.reshape((number_of_segments, *values.shape[1:]))
