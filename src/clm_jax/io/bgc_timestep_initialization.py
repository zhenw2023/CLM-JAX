
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.native_templates import native_template
from clm_jax.io.netcdf import load_dimension_sizes
from clm_jax.io.restart import (
    SubgridTopology,
    configured_land_restart,
    load_subgrid_topology,
)
from clm_jax.physics.bgc_timestep_initialization import (
    BgcTimestepInitializationState,
    DynamicLandcoverCarbonFluxState,
    DynamicLandcoverCarbonSummaryState,
    DynamicLandcoverNitrogenFluxState,
    DynamicLandcoverNitrogenSummaryState,
)


class BgcTimestepInitializationNativeData(NamedTuple):

    topology: SubgridTopology
    decomposition_full_level_count: int
    restart_path: Path


@native_template
def load_bgc_timestep_initialization_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
) -> BgcTimestepInitializationNativeData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    if general.get("use_cn") is not True:
        raise ValueError("BGC timestep initialization requires use_cn=.true.")
    if general.get("use_vertsoilc") is not True:
        raise ValueError("this native adapter requires use_vertsoilc=.true. for full levels")

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    dimensions = load_dimension_sizes(restart)
    required = ("gridcell", "column", "pft", "levgrnd")
    missing = [name for name in required if name not in dimensions]
    if missing:
        raise ValueError(f"CLM restart lacks dimensions: {', '.join(missing)}")

    topology = load_subgrid_topology(restart)
    native_counts = {
        "gridcell": topology.grid.longitude_degrees.shape[0],
        "column": topology.column.gridcell.shape[0],
        "pft": topology.patch.gridcell.shape[0],
    }
    for name, count in native_counts.items():
        if dimensions[name] != count:
            raise ValueError(f"restart {name} dimension disagrees with topology")
    return BgcTimestepInitializationNativeData(
        topology=topology,
        decomposition_full_level_count=int(dimensions["levgrnd"]),
        restart_path=restart,
    )


def assemble_bgc_timestep_initialization_state(
    native: BgcTimestepInitializationNativeData,
    *,
    fill_value: float = 0.0,
) -> BgcTimestepInitializationState:

    gridcell_count = native.topology.grid.longitude_degrees.shape[0]
    column_count = native.topology.column.gridcell.shape[0]
    patch_count = native.topology.patch.gridcell.shape[0]
    level_count = native.decomposition_full_level_count
    value = np.float64(fill_value)

    grid_value = jnp.full((gridcell_count,), value, dtype=jnp.float64)
    column_value = jnp.full((column_count,), value, dtype=jnp.float64)
    column_level_value = jnp.full((column_count, level_count), value, dtype=jnp.float64)
    patch_value = jnp.full((patch_count,), value, dtype=jnp.float64)

    def grid() -> jnp.ndarray:
        return grid_value

    def column() -> jnp.ndarray:
        return column_value

    def column_level() -> jnp.ndarray:
        return column_level_value

    def patch() -> jnp.ndarray:
        return patch_value

    return BgcTimestepInitializationState(
        carbon_flux=DynamicLandcoverCarbonFluxState(
            grid(),
            grid(),
            grid(),
            column(),
            column_level(),
            column_level(),
            column_level(),
            column_level(),
            column_level(),
        ),
        nitrogen_flux=DynamicLandcoverNitrogenFluxState(
            grid(),
            grid(),
            grid(),
            column_level(),
            column_level(),
            column_level(),
            column_level(),
            column_level(),
        ),
        carbon_summary=DynamicLandcoverCarbonSummaryState(
            patch(),
            patch(),
            patch(),
        ),
        nitrogen_summary=DynamicLandcoverNitrogenSummaryState(
            patch(),
            patch(),
            patch(),
            patch(),
        ),
    )


__all__ = [
    "BgcTimestepInitializationNativeData",
    "assemble_bgc_timestep_initialization_state",
    "load_bgc_timestep_initialization_native_data",
]
