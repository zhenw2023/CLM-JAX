
import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from clm_jax.physics.numerics import fortran_add, fortran_multiply


def _reference(values, thickness, initial, column_mask, level_count):
    mask = column_mask.reshape((-1, *(1 for _ in initial.shape[1:])))

    def step(level, total):
        term = fortran_multiply(values[:, level, ...], thickness[level])
        return jnp.where(mask, fortran_add(total, term), total)

    return lax.fori_loop(0, level_count, step, initial)


@partial(jax.custom_jvp, nondiff_argnums=(4,))
def integrate_levels(values, thickness, initial, column_mask, level_count):
    if jax.default_backend() != "gpu" or not initial.size:
        return _reference(values, thickness, initial, column_mask, level_count)
    width, levels = math.prod(initial.shape[1:]), values.shape[1]
    size, block = initial.size, 128

    def kernel(values_ref, thickness_ref, initial_ref, mask_ref, output_ref):
        index = pl.program_id(0) * block + jnp.arange(block)
        column, component = index // width, index % width
        enabled = (index < size) & plt.load(
            mask_ref.at[column], mask=index < size, other=False,
        )
        start = plt.load(initial_ref.at[index], mask=index < size, other=0.)

        def rounded(op, left, right):
            result = plt.elementwise_inline_asm(
                op + ".rn.f64 $0, $1, $2;", args=[left, right],
                constraints="=d,d,d", pack=1,
                result_shape_dtypes=[jax.ShapeDtypeStruct((block,), jnp.float64)],
            )[0]
            return jnp.where(result == 0., jnp.float64(0.), result) if op == "add" else result

        def step(level, total):
            flux = plt.load(
                values_ref.at[(column * levels + level) * width + component],
                mask=enabled, other=0.,
            )
            thickness = jnp.broadcast_to(plt.load(thickness_ref.at[level]), (block,))
            term = rounded("mul", flux, thickness)
            return jnp.where(enabled, rounded("add", total, term), total)

        result = lax.fori_loop(0, level_count, step, start)
        plt.store(output_ref.at[index], result, mask=index < size)

    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((size,), initial.dtype),
        grid=(pl.cdiv(size, block),),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
    )(values.reshape(-1), thickness, initial.reshape(-1), column_mask).reshape(initial.shape)


@integrate_levels.defjvp
def _derivative(level_count, primals, tangents):
    def reference(*args):
        return _reference(*args, level_count)

    return integrate_levels(*primals, level_count), jax.jvp(reference, primals, tangents)[1]
