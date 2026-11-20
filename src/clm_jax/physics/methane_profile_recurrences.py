from functools import wraps

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.physics.methane_balance_initialization import NUMBER_OF_METHANE_SOIL_LEVELS
from clm_jax.physics.numerics import fortran_divide
from clm_jax.physics.numerics import fortran_hardware_add as add
from clm_jax.physics.numerics import fortran_hardware_multiply as multiply


def _ordered_columns(reference, *, shared_count=0):
    def column(*args):
        batched_count = len(args) - shared_count
        expanded = tuple(value[None] if i < batched_count else value
                         for i, value in enumerate(args))
        return jax.tree.map(lambda value: value[0], reference(*expanded))

    @jax.custom_jvp
    def gpu(*args):
        from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch

        return fused_native_batch(
            column, *args, block_size=64,
            shared_argnums=tuple(range(len(args) - shared_count, len(args))),
            hardware_rounding=True, direct_io=True,
        )

    @gpu.defjvp
    def derivative(primals, tangents):
        return gpu(*primals), jax.jvp(reference, primals, tangents)[1]

    @wraps(reference)
    def execute(*args):
        if jax.default_backend() == "gpu" and args[0].shape[0] > 0:
            return gpu(*args)
        return reference(*args)

    return execute


def _serial_column_integral(rate, thickness):
    def level_sum(level, total):
        return add(total, multiply(rate[:, level], thickness[:, level]))

    return lax.fori_loop(0, NUMBER_OF_METHANE_SOIL_LEVELS, level_sum,
                        jnp.zeros((rate.shape[0],), dtype=jnp.float64))


def _clamp_negative_methane(relative, storage, thickness, surface, dt):
    def correct_level(level, carry):
        values, flux = carry
        old = values[:, level + 1]
        negative = old < jnp.float64(0)
        deficit = multiply(-old, storage[:, level])
        deficit = multiply(deficit, thickness[:, level])
        corrected = add(flux, -fortran_divide(deficit, dt))
        return (
            values.at[:, level + 1].set(jnp.where(negative, jnp.float64(0), old)),
            jnp.where(negative, corrected, flux),
        )

    return lax.fori_loop(0, NUMBER_OF_METHANE_SOIL_LEVELS, correct_level,
                        (relative, surface))


def _methane_balance(concentration, before, production, oxidation, thickness, dt):
    def balance_level(level, error):
        delta = add(concentration[:, level], -before[:, level])
        error = add(error, multiply(delta, thickness[:, level]))
        amount = multiply(production[:, level], thickness[:, level])
        amount = multiply(amount, dt)
        error = add(error, -amount)
        amount = multiply(oxidation[:, level], thickness[:, level])
        amount = multiply(amount, dt)
        return add(error, amount)

    return lax.fori_loop(0, NUMBER_OF_METHANE_SOIL_LEVELS, balance_level,
                        jnp.zeros_like(concentration[:, 0]))


serial_column_integral = _ordered_columns(_serial_column_integral)
clamp_negative_methane = _ordered_columns(_clamp_negative_methane, shared_count=1)
methane_balance = _ordered_columns(_methane_balance, shared_count=1)
