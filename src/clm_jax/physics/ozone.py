
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class OzoneStressState(NamedTuple):

    shaded_photosynthesis_coefficient_patch: jnp.ndarray
    sunlit_photosynthesis_coefficient_patch: jnp.ndarray
    shaded_conductance_coefficient_patch: jnp.ndarray
    sunlit_conductance_coefficient_patch: jnp.ndarray


def ozone_stress_off(state: OzoneStressState) -> OzoneStressState:

    arrays = tuple(jnp.asarray(value, dtype=jnp.float64) for value in state)
    if not arrays or arrays[0].ndim != 1:
        raise ValueError("ozone coefficient state must have shape (patch,)")
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("all ozone coefficient fields must have the same patch shape")
    return OzoneStressState(*(jnp.ones_like(value) for value in arrays))
