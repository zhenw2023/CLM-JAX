
from functools import cache

import jax
import jax.numpy as jnp

from clm_jax.physics.numerics import fortran_add, fortran_divide, fortran_multiply


def pivoted_dgtsv(matrix, layer_count):
    count = matrix.diagonal_level.shape[0]
    active = jnp.arange(count) < layer_count
    lower = matrix.lower_diagonal_level
    diagonal = list(jnp.where(active, matrix.diagonal_level, 1.0))
    upper = list(matrix.upper_diagonal_level)
    rhs = list(matrix.right_hand_side_level)
    second_upper = []
    singular = jnp.bool_(False)
    for i in range(count - 1):
        swap = jnp.abs(diagonal[i]) < jnp.abs(lower[i + 1])
        denominator = jnp.where(swap, lower[i + 1], diagonal[i])
        singular = singular | (denominator == 0)
        factor = fortran_divide(
            jnp.where(swap, diagonal[i], lower[i + 1]),
            jnp.where(denominator != 0, denominator, 1.0),
        )
        following = diagonal[i + 1]
        diagonal[i + 1] = fortran_add(
            jnp.where(swap, upper[i], following),
            -fortran_multiply(factor, jnp.where(swap, following, upper[i])),
        )
        diagonal[i] = jnp.where(swap, lower[i + 1], diagonal[i])
        upper[i] = jnp.where(swap, following, upper[i])
        previous_rhs = rhs[i]
        rhs[i] = jnp.where(swap, rhs[i + 1], rhs[i])
        rhs[i + 1] = fortran_add(
            jnp.where(swap, previous_rhs, rhs[i + 1]),
            -fortran_multiply(factor, rhs[i]),
        )
        if i < count - 2:
            second_upper.append(jnp.where(swap, upper[i + 1], 0.0))
            upper[i + 1] = jnp.where(
                swap, fortran_multiply(-factor, upper[i + 1]), upper[i + 1],
            )
    singular = singular | (diagonal[-1] == 0)
    rhs[-1] = fortran_divide(rhs[-1], diagonal[-1])
    for i in reversed(range(count - 1)):
        numerator = fortran_add(rhs[i], -fortran_multiply(upper[i], rhs[i + 1]))
        if i < count - 2:
            numerator = fortran_add(
                numerator, -fortran_multiply(second_upper[i], rhs[i + 2]),
            )
        rhs[i] = fortran_divide(numerator, diagonal[i])
    return jnp.where(active, jnp.where(singular, jnp.nan, jnp.stack(rhs)), 0.0)


@cache
def _column_function(parameters):
    from clm_jax.physics import soil_water_movement as movement
    from clm_jax.physics.solver_fma import native_exp_residual

    def column(*values):
        token = movement._adaptive_early_exit.set(True)
        residual_token = native_exp_residual.set(True)
        try:
            result = movement._adaptive_moisture_columns_reference(
                *(value[None] for value in values), parameters,
                precompute_invariants=True,
            )
            return jax.tree.map(lambda value: value[0], result)
        finally:
            native_exp_residual.reset(residual_token)
            movement._adaptive_early_exit.reset(token)

    return column


def adaptive_columns(values, parameters, work_hint):
    from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch

    key = values[0]
    if work_hint is not None:
        hint = jnp.where(jnp.isfinite(work_hint) & (work_hint > 0), work_hint, 0.0)
        work = jnp.clip(hint, 0, parameters.maximum_adaptive_trials).astype(jnp.int32)
        key = (work << 5) + key
    order = jnp.argsort(key, stable=True)
    result = fused_native_batch(
        _column_function(parameters), *(value[order] for value in values),
        block_size=32, hardware_rounding=True, fold_literals=True,
        direct_io=True, guard_branches=True,
        preserve_literal_barriers=True,
    )
    return jax.tree.map(
        lambda value: jnp.empty_like(value).at[order].set(value, unique_indices=True),
        result,
    )
