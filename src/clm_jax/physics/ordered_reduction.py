
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.physics.numerics import fortran_hardware_add


def _ordered_segment_sum_reference(values, segment_ids, number_of_segments):
    result = jnp.zeros((number_of_segments, *values.shape[1:]), dtype=values.dtype)
    if values.shape[0] == 0:
        return result
    order = jnp.argsort(segment_ids, stable=True)
    counts = jnp.bincount(segment_ids, length=number_of_segments)
    starts = jnp.cumsum(counts) - counts
    maximum_count = jnp.max(counts, initial=0)

    def accumulate(carry):
        offset, total = carry
        index = order[jnp.minimum(starts + offset, values.shape[0] - 1)]
        valid = (offset < counts).reshape((number_of_segments, *((1,) * (values.ndim - 1))))
        contribution = jnp.where(valid, values[index], jnp.zeros_like(total))
        updated = fortran_hardware_add(total, contribution)
        return offset + 1, jnp.where(valid, updated, total)

    return lax.while_loop(
        lambda carry: carry[0] < maximum_count, accumulate, (jnp.int32(0), result)
    )[1]


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def ordered_segment_sum(values, segment_ids, number_of_segments):
    if jax.default_backend() == "gpu" and values.dtype == jnp.float64:
        from clm_jax.physics.ordered_segment_sum_gpu import ordered_segment_sum_gpu

        return ordered_segment_sum_gpu(values, segment_ids, number_of_segments)
    return _ordered_segment_sum_reference(values, segment_ids, number_of_segments)


@ordered_segment_sum.defjvp
def _ordered_segment_sum_jvp(number_of_segments, primals, tangents):
    values, segment_ids = primals
    derivative, _ = tangents
    result = ordered_segment_sum(values, segment_ids, number_of_segments)
    tangent = jax.ops.segment_sum(derivative, segment_ids, number_of_segments)
    return result, tangent
