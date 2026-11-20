from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from clm_jax.physics.numerics import fortran_add, fortran_multiply


class OrderedMapping(NamedTuple):
    weights: jax.Array
    source_index: jax.Array
    row_offsets: jax.Array
    source_size: int
    destination_size: int
    maximum_row_length: int


class PartitionedMapping(NamedTuple):

    partials: OrderedMapping
    reduction: OrderedMapping
    source_size: int
    destination_size: int


def prepare_partitioned_mapping(mapping, source_owner):
    from clm_jax.forcing.remap import SparseMapping

    owner = np.asarray(source_owner, dtype=np.int64)
    if owner.shape != (mapping.source_size,) or np.any(owner < 0):
        raise ValueError("source owner must contain one nonnegative rank per source")
    rows = np.asarray(mapping.destination_index, dtype=np.int64)
    source = np.asarray(mapping.source_index, dtype=np.int32)
    pairs, partial_rows = np.unique(
        owner[source] * mapping.destination_size + rows, return_inverse=True,
    )
    partials = prepare_ordered_mapping(SparseMapping(
        mapping.weights, jnp.asarray(partial_rows, jnp.int32), mapping.source_index,
        mapping.source_size, len(pairs),
    ))
    reduction = prepare_ordered_mapping(SparseMapping(
        jnp.ones(len(pairs), jnp.float64),
        jnp.asarray(pairs % mapping.destination_size, jnp.int32),
        jnp.arange(len(pairs), dtype=jnp.int32), len(pairs), mapping.destination_size,
    ))
    return PartitionedMapping(partials, reduction, mapping.source_size,
                              mapping.destination_size)


def prepare_ordered_mapping(mapping):
    if isinstance(mapping, OrderedMapping):
        return mapping
    rows = np.asarray(mapping.destination_index, dtype=np.int32)
    if np.any(rows < 0) or np.any(rows >= mapping.destination_size):
        raise ValueError("sparse mapping destination index outside domain")
    order = np.argsort(rows, kind="stable").astype(np.int32)
    counts = np.bincount(rows, minlength=mapping.destination_size)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    return OrderedMapping(
        mapping.weights[order], mapping.source_index[order], jnp.asarray(offsets),
        mapping.source_size, mapping.destination_size, int(counts.max(initial=0)),
    )


def _reference(source, weights, indices, offsets, limit):
    count = offsets.shape[0] - 1
    initial = jnp.zeros((*source.shape[:-1], count), dtype=jnp.float64)
    if weights.size == 0:
        return initial

    def step(offset, total):
        position = offsets[:-1] + offset
        valid = position < offsets[1:]
        safe = jnp.minimum(position, weights.size - 1)
        product = fortran_multiply(source[..., indices[safe]], weights[safe])
        updated = fortran_add(total, product)
        return jnp.where(valid, updated, total)

    return lax.fori_loop(0, limit, step, initial)


@partial(jax.custom_jvp, nondiff_argnums=(4,))
def _ordered_gpu(source, weights, indices, offsets, limit):
    count = offsets.shape[0] - 1
    shape = (*source.shape[:-1], count)
    if weights.size == 0 or count == 0:
        return jnp.zeros(shape, jnp.float64)
    fields = source.reshape((-1, source.shape[-1]))
    block = 128

    def kernel(xref, wref, iref, oref, result):
        row = pl.program_id(0) * block + jnp.arange(block)
        field = pl.program_id(1)
        valid = row < count
        first = plt.load(oref.at[row], mask=valid, other=0)
        end = plt.load(oref.at[row + 1], mask=valid, other=0)

        def step(offset, total):
            position = first + offset
            enabled = valid & (position < end)
            column = plt.load(iref.at[position], mask=enabled, other=0)
            value = plt.load(xref.at[field, column], mask=enabled, other=0.0)
            weight = plt.load(wref.at[position], mask=enabled, other=0.0)
            product = plt.elementwise_inline_asm(
                "mul.rn.f64 $0, $1, $2;", args=[value, weight], constraints="=d,d,d",
                pack=1, result_shape_dtypes=[jax.ShapeDtypeStruct((block,), jnp.float64)],
            )[0]
            updated = plt.elementwise_inline_asm(
                "add.rn.f64 $0, $1, $2;", args=[total, product], constraints="=d,d,d",
                pack=1, result_shape_dtypes=[jax.ShapeDtypeStruct((block,), jnp.float64)],
            )[0]
            return jnp.where(enabled, updated, total)

        total = lax.fori_loop(0, jnp.max(end - first), step, jnp.zeros(block, jnp.float64))
        plt.store(result.at[field, row], total, mask=valid)

    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((fields.shape[0], count), jnp.float64),
        grid=(pl.cdiv(count, block), fields.shape[0]),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
    )(fields, weights, indices, offsets).reshape(shape)


@_ordered_gpu.defjvp
def _ordered_gpu_jvp(limit, primals, tangents):
    source, weights, indices, offsets = primals
    return jax.jvp(
        lambda x, w: _reference(x, w, indices, offsets, limit),
        (source, weights), (tangents[0], tangents[1]),
    )


@partial(jax.jit, static_argnums=(4,))
def _apply(source, weights, indices, offsets, limit):
    function = _ordered_gpu if jax.default_backend() == "gpu" else _reference
    return function(source, weights, indices, offsets, limit)


def apply_ordered_mapping(mapping, source):
    source = jnp.asarray(source, dtype=jnp.float64)
    if source.shape[-1] != mapping.source_size:
        raise ValueError("source shape differs from prepared sparse mapping")
    return _apply(source, mapping.weights, mapping.source_index,
                  mapping.row_offsets, mapping.maximum_row_length)
