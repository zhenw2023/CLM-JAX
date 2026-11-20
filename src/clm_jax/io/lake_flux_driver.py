
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.canopy_temperature_driver import (
    load_canopy_temperature_driver_native_data,
)
from clm_jax.io.canopy_temperature_restart import soil_layer_geometry_20sl_8p5m
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import LANDUNIT_LAKE, SubgridTopology
from clm_jax.physics.friction_velocity import FrictionVelocityState
from clm_jax.physics.lake_flux_driver import (
    LakeFluxDriverInputs,
    LakeFluxDriverParameters,
    LakeFluxDriverState,
    LakeFluxNativeMap,
)
from clm_jax.physics.lake_fluxes import (
    NUMBER_OF_GROUND_LEVELS,
    NUMBER_OF_LAKE_LEVELS,
    LakeFluxState,
)


class LakeFluxDriverNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: LakeFluxDriverInputs
    parameters: LakeFluxDriverParameters
    state: LakeFluxDriverState
    restart_path: Path
    surface_data_path: Path


def _configured_surface_data(case: CimeCase) -> Path:
    values = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    declared = values.get("fsurdat")
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError("CLM clm_inparm lacks a scalar fsurdat path")
    path = Path(case.expand(declared)).expanduser()
    if not path.is_absolute():
        path = case.root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM surface dataset is absent: {path}")
    return path


def _lake_native_map(topology: SubgridTopology) -> LakeFluxNativeMap:
    column_active = np.asarray(topology.column.active, dtype=bool)
    column_type = np.asarray(topology.column.landunit_type, dtype=np.int32)
    patch_active = np.asarray(topology.patch.active, dtype=bool)
    patch_type = np.asarray(topology.patch.landunit_type, dtype=np.int32)
    column = np.flatnonzero(column_active & (column_type == LANDUNIT_LAKE))
    patch = np.flatnonzero(patch_active & (patch_type == LANDUNIT_LAKE))
    if column.size == 0 or patch.size == 0:
        raise ValueError("native case contains no active lake column/patch")
    if column.size != patch.size:
        raise ValueError("LakeFluxes requires exactly one patch per lake column")
    patch_column = np.asarray(topology.patch.column, dtype=np.int64)[patch]
    if not np.array_equal(patch_column, column):
        raise ValueError("native lake patch order does not match lake column order")
    gridcell = np.asarray(topology.column.gridcell, dtype=np.int64)[column]
    return LakeFluxNativeMap(
        column_index=jnp.asarray(column, dtype=jnp.int32),
        patch_index=jnp.asarray(patch, dtype=jnp.int32),
        gridcell_index=jnp.asarray(gridcell, dtype=jnp.int32),
    )


def _surface_lake_values(
    topology: SubgridTopology,
    native_map: LakeFluxNativeMap,
    surface_data: Path,
) -> tuple[np.ndarray, np.ndarray]:
    depth_grid = np.asarray(
        load_native_variables(surface_data, ["LAKEDEPTH"])["LAKEDEPTH"],
        dtype=np.float64,
    )
    if depth_grid.ndim != 2:
        raise ValueError("fsurdat LAKEDEPTH must have dimensions (lat, lon)")
    try:
        fetch_grid = np.asarray(
            load_native_variables(surface_data, ["LAKEFETCH"])["LAKEFETCH"],
            dtype=np.float64,
        )
    except (IndexError, KeyError):
        fetch_grid = np.full(depth_grid.shape, -1.0, dtype=np.float64)
    if fetch_grid.shape != depth_grid.shape:
        raise ValueError("fsurdat LAKEFETCH and LAKEDEPTH grids disagree")

    gridcell = np.asarray(native_map.gridcell_index, dtype=np.int64)
    latitude = np.asarray(topology.grid.latitude_index, dtype=np.int64)[gridcell]
    longitude = np.asarray(topology.grid.longitude_index, dtype=np.int64)[gridcell]
    if (
        np.any(latitude < 0)
        or np.any(latitude >= depth_grid.shape[0])
        or np.any(longitude < 0)
        or np.any(longitude >= depth_grid.shape[1])
    ):
        raise ValueError("native lake grid indices lie outside fsurdat")
    depth = depth_grid[latitude, longitude]
    fetch = fetch_grid[latitude, longitude]
    if np.any(~np.isfinite(depth)):
        raise ValueError("active lake columns contain missing LAKEDEPTH")
    if np.any((depth <= 0.0) | (depth >= 5000.0)):
        raise ValueError("active LAKEDEPTH lies outside CLM's supported range")
    fetch = np.where(np.isfinite(fetch), fetch, np.float64(-1.0))
    return depth, fetch


def _lake_layer_thickness(depth: np.ndarray) -> np.ndarray:

    standard = np.asarray(
        [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 7.0, 10.45, 10.45],
        dtype=np.float64,
    )
    standard_depth = np.sum(standard, dtype=np.float64)
    result = np.empty((depth.size, NUMBER_OF_LAKE_LEVELS), dtype=np.float64)
    shallow = depth <= np.float64(1.0)
    result[shallow] = depth[shallow, None] / np.float64(NUMBER_OF_LAKE_LEVELS)
    ratio = depth[~shallow] / standard_depth
    result[~shallow, 0] = standard[0]
    result[~shallow, 1:-1] = standard[None, 1:-1] * ratio[:, None]
    result[~shallow, -1] = standard[-1] * ratio - (standard[0] - standard[0] * ratio)
    if not np.allclose(
        np.sum(result, axis=1),
        depth,
        rtol=0.0,
        atol=4.0 * np.finfo(np.float64).eps * np.maximum(depth, 1.0),
    ):
        raise ValueError("reconstructed lake layers do not close to LAKEDEPTH")
    return result


def lake_node_depth_from_thickness(layer_thickness: np.ndarray) -> np.ndarray:

    thickness = np.asarray(layer_thickness, dtype=np.float64)
    if thickness.ndim != 2 or thickness.shape[1] == 0:
        raise ValueError("lake layer thickness must be a nonempty two-dimensional array")
    node = np.empty_like(thickness)
    node[:, 0] = np.divide(thickness[:, 0], np.float64(2.0))
    for level in range(1, thickness.shape[1]):
        spacing = np.divide(
            np.add(thickness[:, level - 1], thickness[:, level]),
            np.float64(2.0),
        )
        node[:, level] = np.add(node[:, level - 1], spacing)
    return node


def load_lake_flux_driver_native_data(
    case: CimeCase,
    restart_path: str | Path,
) -> LakeFluxDriverNativeData:

    restart = Path(restart_path).expanduser().resolve()
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    dimensions = load_dimension_sizes(restart)
    if dimensions.get("levlak") != NUMBER_OF_LAKE_LEVELS:
        raise ValueError(
            f"LakeFluxes requires levlak={NUMBER_OF_LAKE_LEVELS}, got {dimensions.get('levlak')}"
        )
    temperature = load_canopy_temperature_driver_native_data(case, restart)
    topology = temperature.topology
    native_map = _lake_native_map(topology)
    column = np.asarray(native_map.column_index, dtype=np.int64)
    lake_count = column.size
    surface_data = _configured_surface_data(case)
    depth, fetch = _surface_lake_values(
        topology,
        native_map,
        surface_data,
    )
    lake_layers = _lake_layer_thickness(depth)
    _, _, ground_thickness = soil_layer_geometry_20sl_8p5m()
    ground_layers = np.broadcast_to(
        ground_thickness[None, :],
        (lake_count, NUMBER_OF_GROUND_LEVELS),
    ).copy()

    restart_values = load_native_variables(
        restart,
        ["T_LAKE", "SAVEDTKE1", "USTLAKE"],
    )
    lake_temperature = np.asarray(restart_values["T_LAKE"], dtype=np.float64)[column]
    saved_tke = np.asarray(restart_values["SAVEDTKE1"], dtype=np.float64)[column]
    lake_friction = np.asarray(restart_values["USTLAKE"], dtype=np.float64)[column]
    for name, value in (
        ("T_LAKE", lake_temperature),
        ("SAVEDTKE1", saved_tke),
        ("USTLAKE", lake_friction),
    ):
        if np.any(~np.isfinite(value)):
            raise ValueError(f"active lake restart field {name} contains missing values")

    column_state = temperature.state.column
    special = jnp.full(
        (lake_count,),
        jnp.float64(CLM_SPECIAL_VALUE),
        dtype=jnp.float64,
    )
    surface_state = LakeFluxState(
        ground_temperature=column_state.ground_temperature_column[column],
        momentum_roughness=column_state.ground_momentum_roughness_column[column],
        sensible_roughness=column_state.ground_sensible_roughness_column[column],
        latent_roughness=column_state.ground_latent_roughness_column[column],
        lake_friction_velocity=jnp.asarray(lake_friction),
        friction_velocity=FrictionVelocityState(*(special for _ in FrictionVelocityState._fields)),
    )
    friction = read_case_namelist_group(case, "lnd_in", "friction_velocity")
    maximum_stable_zeta = float(friction.get("zetamaxstable", 0.5))
    if not 0.0 < maximum_stable_zeta <= 1.0:
        raise ValueError("zetamaxstable must lie in (0, 1]")

    return LakeFluxDriverNativeData(
        topology=topology,
        inputs=LakeFluxDriverInputs(
            native_map=native_map,
            ground_layer_thickness_lake_level=jnp.asarray(ground_layers),
            lake_layer_thickness_lake_level=jnp.asarray(lake_layers),
            lake_depth=jnp.asarray(depth),
            lake_fetch=jnp.asarray(fetch),
        ),
        parameters=LakeFluxDriverParameters(
            maximum_stable_zeta=maximum_stable_zeta,
        ),
        state=LakeFluxDriverState(
            surface=surface_state,
            lake_temperature_lake_level=jnp.asarray(lake_temperature),
            saved_top_eddy_conductivity_lake=jnp.asarray(saved_tke),
        ),
        restart_path=restart,
        surface_data_path=surface_data,
    )


__all__ = [
    "LakeFluxDriverNativeData",
    "lake_node_depth_from_thickness",
    "load_lake_flux_driver_native_data",
]
