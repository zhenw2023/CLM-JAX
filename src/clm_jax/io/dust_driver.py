
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.canopy_temperature_driver import (
    load_canopy_temperature_driver_native_data,
)
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.dust import (
    DustExecutionParameters,
    DustParameters,
    DustState,
    initialize_dust_parameters,
    initialize_dust_state,
)
from clm_jax.physics.dust_driver import DustDriverInputs


class DustDriverNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: DustDriverInputs
    state: DustState
    parameters: DustParameters
    execution: DustExecutionParameters
    restart_path: Path
    surface_data_path: Path


def _configured_clm_input_path(case: CimeCase, name: str) -> Path:
    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    declared = general.get(name)
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(f"CLM clm_inparm lacks a scalar {name} path")
    path = Path(case.expand(declared)).expanduser()
    if not path.is_absolute():
        path = case.root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM {name} is absent: {path}")
    return path


def _top_texture_on_gridcells(
    topology: SubgridTopology,
    surface_data_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    values = load_native_variables(
        surface_data_path,
        ["PCT_SAND", "PCT_CLAY"],
        dtype=np.float64,
    )
    sand = np.asarray(values["PCT_SAND"], dtype=np.float64)
    clay = np.asarray(values["PCT_CLAY"], dtype=np.float64)
    if sand.ndim != 3 or sand.shape != clay.shape or sand.shape[0] != 10:
        raise ValueError("fsurdat PCT_SAND/PCT_CLAY must contain ten soil levels")
    latitude = np.asarray(topology.grid.latitude_index, dtype=np.int64)
    longitude = np.asarray(topology.grid.longitude_index, dtype=np.int64)
    sand_grid = sand[:, latitude, longitude].copy()
    clay_grid = clay[:, latitude, longitude].copy()
    zero_surface = sand_grid[0] + clay_grid[0] == np.float64(0.0)
    inconsistent = zero_surface & np.any(
        sand_grid + clay_grid != np.float64(0.0),
        axis=0,
    )
    if np.any(inconsistent):
        raise ValueError("soil texture has zero surface but nonzero deeper levels")
    sand_grid[:, zero_surface] = np.float64(1.0)
    clay_grid[:, zero_surface] = np.float64(1.0)
    if np.any(sand_grid + clay_grid == np.float64(0.0)):
        raise ValueError("soil texture contains a zero sand-plus-clay level")
    return sand_grid[0], clay_grid[0]


def _dust_soil_constants(
    topology: SubgridTopology,
    surface_data_path: Path,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    _, clay_grid = _top_texture_on_gridcells(topology, surface_data_path)
    column_gridcell = np.asarray(topology.column.gridcell, dtype=np.int64)
    clay = clay_grid[column_gridcell]
    threshold = (
        np.float64(0.17)
        + (np.float64(0.14) * clay) * np.float64(0.01)
    )
    limited = np.minimum(
        clay * np.float64(0.01),
        np.float64(0.20),
    )
    return (
        jnp.asarray(threshold, dtype=jnp.float64),
        jnp.asarray(limited, dtype=jnp.float64),
    )


def load_dust_driver_native_data(
    case: CimeCase,
    restart_path: str | Path,
) -> DustDriverNativeData:

    restart = Path(restart_path).expanduser().resolve()
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    temperature = load_canopy_temperature_driver_native_data(case, restart)
    topology = temperature.topology
    surface_data = _configured_clm_input_path(case, "fsurdat")
    threshold, limited_clay = _dust_soil_constants(topology, surface_data)
    vegetation = load_native_variables(
        restart,
        ["tlai", "tsai"],
        dtype=np.float64,
    )
    patch_count = topology.patch.column.shape[0]
    total_leaf = np.asarray(vegetation["tlai"], dtype=np.float64)
    total_stem = np.asarray(vegetation["tsai"], dtype=np.float64)
    if total_leaf.shape != (patch_count,) or total_stem.shape != (patch_count,):
        raise ValueError("restart tlai/tsai must contain every native patch")
    if not np.all(np.isfinite(total_leaf)) or not np.all(np.isfinite(total_stem)):
        raise ValueError("restart tlai/tsai contains missing values")

    saturated = jnp.asarray(
        temperature.parameters.saturated_water_content_column[:, 0],
        dtype=jnp.float64,
    )
    thickness = jnp.asarray(
        temperature.parameters.ground_layer_thickness_column[:, 0],
        dtype=jnp.float64,
    )
    return DustDriverNativeData(
        topology=topology,
        inputs=DustDriverInputs(
            gravimetric_water_threshold_column=threshold,
            clay_mass_fraction_limited_column=limited_clay,
            saturated_water_content_top_column=saturated,
            top_soil_layer_thickness_column=thickness,
            total_leaf_area_index_patch=jnp.asarray(total_leaf, dtype=jnp.float64),
            total_stem_area_index_patch=jnp.asarray(total_stem, dtype=jnp.float64),
        ),
        state=initialize_dust_state(topology),
        parameters=initialize_dust_parameters(),
        execution=DustExecutionParameters.from_case(case),
        restart_path=restart,
        surface_data_path=surface_data,
    )


__all__ = [
    "DustDriverNativeData",
    "load_dust_driver_native_data",
]
