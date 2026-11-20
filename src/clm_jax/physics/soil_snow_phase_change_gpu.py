
from functools import lru_cache, partial

import jax
import jax.numpy as jnp

from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch


@lru_cache(maxsize=8)
def _column_step(seconds, *, linearize_latent_balance=False):
    from clm_jax.physics.soil_temperature import _column_soil_snow_phase_change

    def step(inputs):
        return _column_soil_snow_phase_change(
            inputs, jnp.float64(seconds), linearize_latent_balance=linearize_latent_balance
        )

    return step


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def phase_change(inputs, time_step_seconds):
    return fused_native_batch(
        _column_step(time_step_seconds), inputs,
        block_size=32, hardware_rounding=True, direct_io=True,
    )


@phase_change.defjvp
def _phase_change_jvp(time_step_seconds, primals, tangents):
    from clm_jax.physics.soil_temperature import _mixed_snow_equilibrium_dot
    step = _column_step(time_step_seconds, linearize_latent_balance=True)
    value, dot = jax.jvp(jax.vmap(step), primals, tangents)
    return value, _mixed_snow_equilibrium_dot(primals[0],time_step_seconds,tangents[0],dot)
