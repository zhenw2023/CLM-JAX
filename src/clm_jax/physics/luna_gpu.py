
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from clm_jax.physics.converged_iteration import SPATIAL_SOLVE_AXIS, gpu_convergence_scope
from clm_jax.physics.luna import (
    LunaNitrogenAllocationInputs, _AllocationContext, _allocation_context,
    _allocation_iterations, luna_nitrogen_allocation,
)
from clm_jax.physics.solver_scalarization import _native_tile_trace


def _reference(inputs):
    shape = inputs.functional_nitrogen.shape
    flat = jax.tree.map(lambda value: value.reshape(-1), inputs)
    result = _reference_map(flat)
    return jax.tree.map(lambda value: value.reshape(shape), result)


@lru_cache(maxsize=4)
def _program(block):
    from clm_jax.physics.solver_kernel_lowering import native_solver_jaxpr

    inputs = LunaNitrogenAllocationInputs(*(
        jax.ShapeDtypeStruct((block,), jnp.float64) for _ in LunaNitrogenAllocationInputs._fields
    ))
    context = _AllocationContext(inputs, *(
        jax.ShapeDtypeStruct((block,), jnp.float64) for _ in _AllocationContext._fields[1:]
    ))
    scoped = gpu_convergence_scope(jax.vmap(
        _allocation_iterations, axis_name=SPATIAL_SOLVE_AXIS,
    ))
    token = _native_tile_trace.set(True)
    try:
        program = jax.make_jaxpr(scoped)(context)
        shapes = jax.eval_shape(scoped, context)
    finally:
        _native_tile_trace.reset(token)
    if program.consts:
        raise ValueError("LUNA iteration unexpectedly captured non-scalar constants")
    return native_solver_jaxpr(program), shapes


@jax.custom_jvp
def allocation_batch(inputs):
    if jax.default_backend() != "gpu":
        return _reference(inputs)
    shape = inputs.functional_nitrogen.shape
    count = inputs.functional_nitrogen.size
    if count == 0:
        return _reference(inputs)
    block = 64
    program, outputs = _program(block)
    flat = jax.tree.map(lambda value: value.reshape(-1), inputs)
    context = jax.vmap(_allocation_context)(flat)
    leaves = jax.tree.leaves(context)

    def kernel(*refs):
        index = pl.program_id(0) * block + jnp.arange(block)
        safe = jnp.minimum(index, count - 1)
        values = [plt.load(ref.at[safe]) for ref in refs[:len(leaves)]]
        results = jax.core.eval_jaxpr(program.jaxpr, (), *values)
        for ref, value in zip(refs[len(leaves):], results, strict=True):
            plt.store(ref.at[index], value, mask=index < count)

    out_shape = jax.tree.map(lambda value: jax.ShapeDtypeStruct((count,), value.dtype), outputs)
    result = pl.pallas_call(
        kernel, out_shape=out_shape, grid=(pl.cdiv(count, block),),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
    )(*leaves)
    return jax.tree.map(lambda value: value.reshape(shape), result)


@allocation_batch.defjvp
def _derivative(primals, tangents):
    return jax.jvp(_reference, primals, tangents)


def _reference_map(flat):
    from clm_jax.physics.converged_iteration import iteration_ad_checkpointing

    if not iteration_ad_checkpointing() or jax.default_backend() == "cpu":
        return jax.vmap(luna_nitrogen_allocation)(flat)
    from clm_jax.physics.pointwise import active_scalar_map

    scalar_shapes = jax.tree.map(lambda value: jax.ShapeDtypeStruct((), value.dtype), flat)
    shapes = jax.eval_shape(luna_nitrogen_allocation, scalar_shapes)
    empty = jax.tree.map(lambda value: jnp.zeros(value.shape, value.dtype), shapes)
    return active_scalar_map(
        luna_nitrogen_allocation, jnp.ones(flat.functional_nitrogen.shape, dtype=jnp.bool_),
        flat, empty,
    )
