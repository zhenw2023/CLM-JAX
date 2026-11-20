from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from clm_jax.physics.converged_iteration import (
    SPATIAL_SOLVE_AXIS, _dynamic_primal, gpu_convergence_scope,
)
from clm_jax.physics.solver_scalarization import _native_tile_trace

BLOCK_SIZE = 64


def _reference(template, stress, first, second, first_result, second_result, tolerance, enabled):
    from clm_jax.physics.photosynthesis import _solve_ci_secant_brent
    return _solve_ci_secant_brent(
        template, stress, first, second, first_result, second_result, tolerance,
        enabled=enabled,
    )


@lru_cache(maxsize=8)
def _tile_program(tree, signatures, block):
    from clm_jax.physics.solver_kernel_lowering import native_solver_jaxpr
    inputs = tree.unflatten([
        jax.ShapeDtypeStruct((block, *shape), dtype) for shape, dtype in signatures
    ])
    function = gpu_convergence_scope(jax.vmap(_reference, axis_name=SPATIAL_SOLVE_AXIS))
    closed = jax.make_jaxpr(function)(*inputs)
    outputs, output_tree = jax.tree.flatten(jax.eval_shape(function, *inputs))
    if closed.consts:
        raise ValueError("native Ci solver unexpectedly captured array constants")
    return native_solver_jaxpr(closed), outputs, output_tree


def fused_ci_batch(*arguments, block_size=BLOCK_SIZE):
    leaves, tree = jax.tree.flatten(arguments)
    count = leaves[0].shape[0]
    if not count:
        return jax.vmap(_reference)(*arguments)
    signatures = tuple((value.shape[1:], value.dtype) for value in leaves)
    closed, outputs, output_tree = _tile_program(tree, signatures, block_size)
    shapes = tuple(jax.ShapeDtypeStruct((count, *out.shape[1:]), out.dtype) for out in outputs)

    def kernel(*refs):
        row = pl.program_id(0) * block_size + jnp.arange(block_size)
        safe = jnp.minimum(row, count - 1)

        def indices(value, rows):
            if len(value.shape) == 1:
                return (rows,)
            if len(value.shape) != 2:
                raise ValueError("Ci solver expects patch scalars or two leaf classes")
            return rows[:, None], jnp.arange(value.shape[1])[None, :]

        values = [plt.load(ref.at[indices(ref, safe)]) for ref in refs[:len(leaves)]]
        results = jax.core.eval_jaxpr(closed.jaxpr, (), *values)
        for ref, result in zip(refs[len(leaves):], results, strict=True):
            mask = row < count
            if len(ref.shape) == 2:
                mask = mask[:, None]
            plt.store(ref.at[indices(ref, row)], result, mask=mask)

    result = pl.pallas_call(
        kernel, out_shape=shapes, grid=(pl.cdiv(count, block_size),),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
    )(*leaves)
    return output_tree.unflatten(result)


@custom_vmap
def _dispatch(*arguments):
    return _reference(*arguments)


@_dispatch.def_vmap
def _dispatch_batch(axis_size, in_batched, *arguments):
    if not all(jax.tree.leaves(in_batched)):
        axes = jax.tree.map(lambda batched: 0 if batched else None, tuple(in_batched))
        result = jax.vmap(_reference, in_axes=axes, axis_name=SPATIAL_SOLVE_AXIS)(*arguments)
        return result, jax.tree.map(lambda _: True, result)
    batch = jax.tree.map(
        lambda value, batched: value if batched else jnp.broadcast_to(
            value, (axis_size, *jnp.shape(value)),
        ), arguments, tuple(in_batched),
    )
    result = (fused_ci_batch(*batch) if jax.default_backend() == "gpu"
              else jax.vmap(_reference)(*batch))
    return result, jax.tree.map(lambda _: True, result)


def solve_ci_gpu(*arguments):
    if (_dynamic_primal.get() and jax.default_backend() == "gpu"
            and not _native_tile_trace.get()):
        return _dispatch(*arguments)
    return _reference(*arguments)
