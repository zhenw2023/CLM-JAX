
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.driver import load_driver_initialization_native_data


class ExposedVegetationFilterNativeData(NamedTuple):

    landunit_type: jnp.ndarray
    patch_active: jnp.ndarray
    vegetation_exposed: jnp.ndarray
    restart_path: Path


def load_exposed_vegetation_filter_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
) -> ExposedVegetationFilterNativeData:

    driver = load_driver_initialization_native_data(case, restart_path)
    vegetation_exposed = jnp.where(
        driver.patch_active,
        driver.vegetation_exposed_for_albedo,
        jnp.int32(0),
    )
    return ExposedVegetationFilterNativeData(
        landunit_type=driver.topology.patch.landunit_type,
        patch_active=driver.patch_active,
        vegetation_exposed=vegetation_exposed,
        restart_path=driver.restart_path,
    )


__all__ = [
    "ExposedVegetationFilterNativeData",
    "load_exposed_vegetation_filter_native_data",
]
