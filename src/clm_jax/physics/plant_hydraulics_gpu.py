
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_batching import custom_vmap
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from clm_jax.physics.converged_iteration import _dynamic_primal, gpu_convergence_scope
from clm_jax.physics.solver_scalarization import (
    _fold_tile_literals,
    _guard_tile_branches,
    _native_tile_rounding,
    _native_tile_trace,
    _preserve_tile_literal_barriers,
)

BLOCK_SIZE = 64
_retained_tile_programs = ContextVar("clm_retained_native_tile_programs", default=None)


@contextmanager
def retain_native_tile_programs(*, maximum_entries=64):
    if (isinstance(maximum_entries, bool) or not isinstance(maximum_entries, int)
            or maximum_entries < 1):
        raise ValueError("native program cache size must be a positive integer")
    programs = OrderedDict()
    state = dict(programs=programs, maximum_entries=maximum_entries,
                 hits=0, misses=0, evictions=0)
    token = _retained_tile_programs.set(state)
    try:
        yield
    finally:
        _retained_tile_programs.reset(token)
        programs.clear()


def native_tile_program_cache_statistics():
    state = _retained_tile_programs.get()
    if state is None:
        return None
    return dict(native_tile_program_cache_entries=len(state["programs"]),
                native_tile_program_cache_hits=state["hits"],
                native_tile_program_cache_misses=state["misses"],
                native_tile_program_cache_evictions=state["evictions"])


def _reference(initial, inputs, parameters, transpiration, enabled):
    from clm_jax.physics.plant_hydraulics import _newton_spac_solution_reference
    return _newton_spac_solution_reference(
        initial, inputs, parameters, transpiration, enabled=enabled,
    )


@lru_cache(maxsize=8)
def _tile_program(reference, tree, signatures, block, hardware_rounding=False,
                  fold_literals=False, guard_branches=False, preserve_literal_barriers=False):
    from clm_jax.physics.solver_kernel_lowering import native_solver_jaxpr
    from clm_jax.physics.solver_scalarization import _flatten_storage, _restore_storage, evaluate

    scalar_arguments = tree.unflatten([
        jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in signatures
    ])
    function = gpu_convergence_scope(reference)
    token = _native_tile_trace.set(True)
    rounding_token = _native_tile_rounding.set(hardware_rounding)
    branch_token = _guard_tile_branches.set(guard_branches)
    try:
        scalar_program, output_shapes = jax.make_jaxpr(
            function, return_shape=True,
        )(*scalar_arguments)
        outputs, output_tree = jax.tree.flatten(output_shapes)
    finally:
        _native_tile_rounding.reset(rounding_token)
        _guard_tile_branches.reset(branch_token)
        _native_tile_trace.reset(token)
    storage_shapes = tuple(shape for shape, _ in signatures)

    def tile(*values):
        return _flatten_storage(evaluate(scalar_program, _restore_storage(storage_shapes, values)))

    tile_arguments = [
        jax.ShapeDtypeStruct((block,), dtype)
        for shape, dtype in signatures for _ in np.ndindex(shape)
    ]
    literal_token = _fold_tile_literals.set(fold_literals)
    barrier_token = _preserve_tile_literal_barriers.set(preserve_literal_barriers)
    branch_token = _guard_tile_branches.set(guard_branches)
    try:
        tiled = jax.make_jaxpr(tile)(*tile_arguments)
    finally:
        _fold_tile_literals.reset(literal_token)
        _preserve_tile_literal_barriers.reset(barrier_token)
        _guard_tile_branches.reset(branch_token)
    if tiled.consts:
        raise ValueError("native Newton tile unexpectedly captured array constants")
    return native_solver_jaxpr(tiled), outputs, output_tree


def _retained_tile_program(reference, tree, signatures, block, hardware_rounding=False,
                           fold_literals=False, guard_branches=False,
                           preserve_literal_barriers=False):
    state = _retained_tile_programs.get()
    key = (reference, tree, signatures, block, hardware_rounding, fold_literals,
           guard_branches, preserve_literal_barriers)
    if state is None:
        return _tile_program(*key)
    programs = state["programs"]
    if key in programs:
        state["hits"] += 1
        programs.move_to_end(key)
        return programs[key]
    state["misses"] += 1
    program = _tile_program(*key)
    programs[key] = program
    if len(programs) > state["maximum_entries"]:
        programs.popitem(last=False)
        state["evictions"] += 1
    return program


def fused_newton_batch(*arguments, block_size=BLOCK_SIZE):
    return fused_native_batch(
        _reference, *arguments, block_size=block_size,
        direct_io=True, fold_literals=True, guard_branches=True,
        preserve_literal_barriers=True,
    )


def fused_native_batch(reference, *arguments, block_size=BLOCK_SIZE, hardware_rounding=True,
                       shared_argnums=(), fold_literals=False, direct_io=False,
                       guard_branches=False, preserve_literal_barriers=False):
    leaves, tree = jax.tree.flatten(arguments)
    batched = jax.tree.leaves(tuple(
        jax.tree.map(lambda _, index=index: index not in shared_argnums, value)
        for index, value in enumerate(arguments)
    ))
    count = next(value.shape[0] for value, batch in zip(leaves, batched, strict=True) if batch)
    if not count:
        return jax.vmap(reference, in_axes=tuple(
            None if index in shared_argnums else 0 for index in range(len(arguments))
        ))(*arguments)
    leaves = [value if batch else jnp.asarray(value)
              for value, batch in zip(leaves, batched, strict=True)]
    signatures = tuple((value.shape[1:] if batch else value.shape, value.dtype)
                       for value, batch in zip(leaves, batched, strict=True))
    closed, outputs, output_tree = _retained_tile_program(
        reference, tree, signatures, block_size, hardware_rounding=hardware_rounding,
        fold_literals=fold_literals,
        guard_branches=guard_branches,
        preserve_literal_barriers=preserve_literal_barriers,
    )
    if direct_io:
        from clm_jax.physics.native_tile_call import native_tile_call
        return output_tree.unflatten(native_tile_call(
            leaves, closed, signatures, batched, outputs, count, block_size,
        ))
    shapes = tuple(jax.ShapeDtypeStruct((count, *output.shape), output.dtype) for output in outputs)
    tile_io = plt

    def kernel(*refs):
        row = pl.program_id(0) * block_size + jnp.arange(block_size)
        safe = jnp.minimum(row, count - 1)
        values = []
        for ref, (shape, _), batch in zip(refs[:len(leaves)], signatures, batched, strict=True):
            for index in np.ndindex(shape):
                value = tile_io.load(ref.at[(safe, *index)] if batch else ref.at[index])
                values.append(value if batch else jnp.broadcast_to(value, (block_size,)))
        results = iter(jax.core.eval_jaxpr(closed.jaxpr, (), *values))
        for ref, output in zip(refs[len(leaves):], outputs, strict=True):
            for index in np.ndindex(output.shape):
                tile_io.store(ref.at[(row, *index)], next(results), mask=row < count)

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
        result = jax.vmap(_reference, in_axes=axes)(*arguments)
        return result, jax.tree.map(lambda _: True, result)
    result = fused_newton_batch(*arguments)
    return result, jax.tree.map(lambda _: True, result)


def solve_newton_gpu(*arguments):
    if (_dynamic_primal.get() and jax.default_backend() == "gpu"
            and not _native_tile_trace.get()):
        token = _native_tile_trace.set(True)
        try:
            return _dispatch(*arguments)
        finally:
            _native_tile_trace.reset(token)
    return _reference(*arguments)
