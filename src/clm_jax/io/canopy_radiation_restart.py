
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    SubgridTopology,
    configured_land_restart,
    load_subgrid_topology,
)
from clm_jax.physics.canopy_radiation import (
    CanopySunShadeInputs,
    CanopySunShadeState,
)


class CanopyRadiationRestartData(NamedTuple):
    topology: SubgridTopology
    inputs: CanopySunShadeInputs
    state: CanopySunShadeState


_RESTART_VARIABLES = (
    "elai",
    "fsun",
    "parsun",
    "parsha",
    "tlai_z",
    "nrad",
    "fsun_z",
    "fabd_sun_z",
    "fabd_sha_z",
    "fabi_sun_z",
    "fabi_sha_z",
)


def _finite_restart_float(value: jnp.ndarray) -> jnp.ndarray:
    value = jnp.asarray(value, dtype=jnp.float64)
    return jnp.where(jnp.isnan(value), jnp.float64(CLM_SPECIAL_VALUE), value)


def load_canopy_radiation_restart(
    case: CimeCase,
    restart_path: str | Path | None = None,
) -> CanopyRadiationRestartData:

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")

    dimensions = load_dimension_sizes(restart)
    if "pft" not in dimensions or "levcan" not in dimensions:
        raise ValueError(f"restart lacks native pft/levcan dimensions: {restart}")
    number_of_patches = dimensions["pft"]
    number_of_layers = dimensions["levcan"]
    if number_of_layers < 1:
        raise ValueError("restart must contain at least one canopy layer")

    topology = load_subgrid_topology(restart)
    if topology.patch.gridcell.shape != (number_of_patches,):
        raise ValueError("restart topology and pft dimension disagree")
    values = load_native_variables(
        restart,
        list(_RESTART_VARIABLES),
        masked_integer_fill_value=-9999,
    )

    scalar_shape = (number_of_patches,)
    layer_shape = (number_of_patches, number_of_layers)
    for name in ("elai", "fsun", "nrad"):
        if values[name].shape != scalar_shape:
            raise ValueError(f"restart variable {name!r} has unexpected shape {values[name].shape}")
    for name in _RESTART_VARIABLES:
        if name not in ("elai", "fsun", "nrad") and values[name].shape != layer_shape:
            raise ValueError(f"restart variable {name!r} has unexpected shape {values[name].shape}")

    scalar_initialized = (topology.patch.landunit_type == LANDUNIT_SOIL) | (
        topology.patch.landunit_type == LANDUNIT_CROP
    )
    initial_scalar_leaf_area = jnp.where(
        scalar_initialized,
        jnp.float64(0.0),
        jnp.float64(CLM_SPECIAL_VALUE),
    )
    initial_layer_leaf_area = jnp.full(
        layer_shape,
        jnp.float64(CLM_SPECIAL_VALUE),
        dtype=jnp.float64,
    )

    return CanopyRadiationRestartData(
        topology=topology,
        inputs=CanopySunShadeInputs(
            exposed_leaf_area_index_patch=_finite_restart_float(values["elai"]),
            total_leaf_area_by_layer_patch=_finite_restart_float(values["tlai_z"]),
            sunlit_fraction_by_layer_patch=_finite_restart_float(values["fsun_z"]),
            radiative_layer_count_patch=jnp.asarray(values["nrad"], dtype=jnp.int32),
            direct_absorption_sunlit_by_layer_patch=_finite_restart_float(values["fabd_sun_z"]),
            direct_absorption_shaded_by_layer_patch=_finite_restart_float(values["fabd_sha_z"]),
            diffuse_absorption_sunlit_by_layer_patch=_finite_restart_float(values["fabi_sun_z"]),
            diffuse_absorption_shaded_by_layer_patch=_finite_restart_float(values["fabi_sha_z"]),
        ),
        state=CanopySunShadeState(
            sunlit_leaf_area_patch=initial_scalar_leaf_area,
            shaded_leaf_area_patch=initial_scalar_leaf_area,
            sunlit_leaf_area_by_layer_patch=initial_layer_leaf_area,
            shaded_leaf_area_by_layer_patch=initial_layer_leaf_area,
            sunlit_fraction_patch=_finite_restart_float(values["fsun"]),
            absorbed_par_sunlit_by_layer_patch=_finite_restart_float(values["parsun"]),
            absorbed_par_shaded_by_layer_patch=_finite_restart_float(values["parsha"]),
        ),
    )
