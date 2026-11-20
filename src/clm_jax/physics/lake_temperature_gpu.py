
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap

from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch


def _reference(*arguments):
    from clm_jax.physics.lake_temperature import _tridiagonal_solve_reference

    return _tridiagonal_solve_reference(*arguments)


@custom_vmap
def _dispatch(*arguments):
    return _reference(*arguments)


@_dispatch.def_vmap
def _batch(axis_size, in_batched, *arguments):
    values = tuple(
        value if mapped else jnp.broadcast_to(value, (axis_size, *jnp.shape(value)))
        for mapped, value in zip(in_batched, arguments, strict=True)
    )
    return fused_native_batch(
        _reference, *values, block_size=32, hardware_rounding=True, direct_io=True,
    ), True


@jax.custom_jvp
def lake_tridiagonal_solve(*arguments):
    return _dispatch(*arguments)


@partial(lake_tridiagonal_solve.defjvp, symbolic_zeros=True)
def _derivative(primals, tangents):
    from clm_jax.compilation import _symbolic_zero_jvp
    return _symbolic_zero_jvp(_reference, primals, tangents)


@lru_cache(maxsize=8)
def _column_program(time_step_seconds):
    from clm_jax.physics.lake_temperature import (
        _active_column_step,
        _inactive_column_result,
        _lake_temperature_reference,
    )
    from clm_jax.physics.solver_fma import native_exp_residual

    def column(inputs, geometry, state):
        token = native_exp_residual.set(True)
        try:
            return jax.lax.cond(
                jnp.asarray(inputs.active_lake, jnp.bool_),
                lambda: _active_column_step(inputs, geometry, state, time_step_seconds),
                lambda: _inactive_column_result(inputs, state),
            )
        finally:
            native_exp_residual.reset(token)

    def reference(*arguments):
        return _lake_temperature_reference(*arguments, time_step_seconds=time_step_seconds)

    @jax.custom_jvp
    def execute(*arguments):
        return fused_native_batch(
            column, *arguments, block_size=32, fold_literals=True,
            hardware_rounding=True, direct_io=True, guard_branches=True,
            preserve_literal_barriers=True,
        )

    @partial(execute.defjvp, symbolic_zeros=True)
    def derivative(primals, tangents):
        from clm_jax.compilation import _symbolic_zero_jvp
        return _symbolic_zero_jvp(reference, primals, tangents)

    return execute


def lake_temperature_columns(inputs, geometry, state, *, time_step_seconds=1800.):
    return _column_program(time_step_seconds)(inputs, geometry, state)
