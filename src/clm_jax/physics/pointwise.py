
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.physics.converged_iteration import (
    SPATIAL_SOLVE_AXIS,
    checkpoint_iteration,
    iteration_ad_checkpointing,
)

_GPU_ACTIVE_CAPACITIES = (2048, 32768)
_AD_ACTIVE_BATCH_SIZE = 4096


def scalar_map(function, operands):

    if jax.default_backend() == "cpu":
        return lax.map(function, operands)
    return jax.vmap(function)(operands)


def _sparse_active_map(function, active, operands, capacity):
    count = active.shape[0]
    indices = jnp.nonzero(active, size=capacity, fill_value=count)[0]
    valid = indices < count
    safe_indices = jnp.where(valid, indices, indices[0])
    packed = jax.tree.map(lambda field: field[safe_indices], operands)
    result = jax.vmap(function, axis_name=SPATIAL_SOLVE_AXIS)(packed)

    def unpack(field):
        mask = valid.reshape((capacity, *((1,) * (field.ndim - 1))))
        values = jnp.where(mask, field, jnp.zeros_like(field))
        output = jnp.zeros((count, *field.shape[1:]), field.dtype)
        return output.at[indices].set(values, mode="drop")

    return jax.tree.map(unpack, result)


def active_scalar_map(function, active, operands, empty, *, batch_size=None):

    def scalar(arguments):
        enabled, values = arguments
        return lax.cond(enabled, function, lambda _: empty, values)

    from clm_jax.physics.solver_scalarization import _guard_tile_branches

    if _guard_tile_branches.get() and active.shape == (1,):
        values = jax.tree.map(lambda value: value[0], operands)
        result = scalar((active[0], values))
        return jax.tree.map(lambda value: value[None], result)

    if jax.default_backend() == "cpu":
        return lax.map(scalar, (active, operands))
    count = active.shape[0]
    if count == 0:
        return jax.vmap(scalar)((active, operands))
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive or None")
    if iteration_ad_checkpointing():
        batch_size = (min(_AD_ACTIVE_BATCH_SIZE, batch_size)
                      if batch_size is not None else _AD_ACTIVE_BATCH_SIZE)
    if batch_size is None and count > 4096:
        active_count = jnp.count_nonzero(active)
        capacities = tuple(size for size in _GPU_ACTIVE_CAPACITIES if size < count)

        def dispatch(position, arguments):
            if position == len(capacities):
                return active_scalar_map(
                    function, arguments[0], arguments[1], empty, batch_size=count,
                )
            capacity = capacities[position]
            return lax.cond(
                (active_count > 0) & (active_count <= capacity),
                lambda args: _sparse_active_map(function, args[0], args[1], capacity),
                lambda args: dispatch(position + 1, args),
                arguments,
            )

        return dispatch(0, (active, operands))
    width = count if batch_size is None else min(count, batch_size)
    padding = (-count) % width

    def pack(value):
        padded = jnp.pad(value, ((0, padding), *((0, 0) for _ in value.shape[1:])))
        return padded.reshape((-1, width, *value.shape[1:]))

    zeros = jax.tree.map(lambda value: jnp.zeros((width, *value.shape), value.dtype), empty)

    def block(arguments):
        enabled, values = arguments

        def solve(values):
            first = jnp.argmax(enabled)

            def sanitize(value):
                mask = enabled.reshape((width, *((1,) * (value.ndim - 1))))
                return jnp.where(mask, value, value[first])

            calculated = jax.vmap(function, axis_name=SPATIAL_SOLVE_AXIS)(
                jax.tree.map(sanitize, values)
            )

            def retain(value):
                mask = enabled.reshape((width, *((1,) * (value.ndim - 1))))
                return jnp.where(mask, value, jnp.zeros_like(value))

            return jax.tree.map(retain, calculated)

        return lax.cond(jnp.any(enabled), solve, lambda _: zeros, values)

    result = lax.map(checkpoint_iteration(block), (pack(active), jax.tree.map(pack, operands)))
    return jax.tree.map(lambda value: value.reshape((-1, *value.shape[2:]))[:count], result)
