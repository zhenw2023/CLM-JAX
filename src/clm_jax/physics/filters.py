
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.io.restart import LANDUNIT_LAKE, LANDUNIT_URBAN_MAX, LANDUNIT_URBAN_MIN


class ExposedVegetationMasks(NamedTuple):
    exposed: jnp.ndarray
    not_exposed: jnp.ndarray


def exposed_vegetation_masks(
    landunit_type: jnp.ndarray,
    patch_active: jnp.ndarray,
    vegetation_exposed: jnp.ndarray,
) -> ExposedVegetationMasks:

    if not (landunit_type.shape == patch_active.shape == vegetation_exposed.shape):
        raise ValueError("all exposed-vegetation filter inputs must share the patch axis")
    landunit_type = jnp.asarray(landunit_type, dtype=jnp.int32)
    active = jnp.asarray(patch_active, dtype=jnp.bool_)
    exposed_fraction = jnp.asarray(vegetation_exposed, dtype=jnp.int32)
    urban = (landunit_type >= LANDUNIT_URBAN_MIN) & (landunit_type <= LANDUNIT_URBAN_MAX)
    eligible = active & (landunit_type != LANDUNIT_LAKE) & ~urban
    exposed = eligible & (exposed_fraction > jnp.int32(0))
    return ExposedVegetationMasks(
        exposed=exposed,
        not_exposed=eligible & ~exposed,
    )
