
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.canopy_temperature_restart import (
    soil_layer_geometry_20sl_8p5m,
)
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    SubgridTopology,
    configured_land_restart,
    load_subgrid_topology,
)
from clm_jax.physics.cn_zero_manifest import (
    NUMBER_OF_DECOMPOSITION_LEVELS,
    NUMBER_OF_DECOMPOSITION_POOLS,
    NUMBER_OF_DECOMPOSITION_TRANSITIONS,
)
from clm_jax.physics.soil_biogeochem_potential import (
    SoilBiogeochemPotentialInputs,
    SoilBiogeochemPotentialParameters,
)

_CARBON_RESTART_VARIABLES = (
    "litr1c_vr",
    "litr2c_vr",
    "litr3c_vr",
    "cwdc_vr",
    "soil1c_vr",
    "soil2c_vr",
    "soil3c_vr",
)
_NITROGEN_RESTART_VARIABLES = (
    "litr1n_vr",
    "litr2n_vr",
    "litr3n_vr",
    "cwdn_vr",
    "soil1n_vr",
    "soil2n_vr",
    "soil3n_vr",
)
_PARAMETER_VARIABLES = (
    "cn_s1_bgc",
    "cn_s2_bgc",
    "cn_s3_bgc",
    "rf_l1s1_bgc",
    "rf_l2s1_bgc",
    "rf_l3s2_bgc",
    "rf_s2s1_bgc",
    "rf_s2s3_bgc",
    "rf_s3s1_bgc",
    "rf_cwdl2_bgc",
    "rf_cwdl3_bgc",
    "cwd_fcel",
    "cwd_flig",
    "dnp",
)


class SoilBiogeochemPotentialPoolState(NamedTuple):

    decomposition_carbon_pool_column_level_pool: jnp.ndarray
    decomposition_nitrogen_pool_column_level_pool: jnp.ndarray


class SoilBiogeochemPotentialNativeData(NamedTuple):

    topology: SubgridTopology
    soil_column: jnp.ndarray
    pools: SoilBiogeochemPotentialPoolState
    respiration_fraction_column_level_transition: jnp.ndarray
    path_fraction_column_level_transition: jnp.ndarray
    parameters: SoilBiogeochemPotentialParameters
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path


def _selected_input_path(
    case: CimeCase,
    key: str,
    override: str | Path | None,
) -> Path:
    if override is None:
        group = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        declared = group.get(key)
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError(f"clm_inparm in {case.case_docs / 'lnd_in'} lacks {key}")
        path = Path(case.expand(declared)).expanduser()
        if not path.is_absolute():
            path = case.root / path
        path = path.resolve()
    else:
        path = Path(override).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM {key} file does not exist: {path}")
    return path


def _finite_scalar(name: str, value: jnp.ndarray) -> np.float64:
    host = np.asarray(value, dtype=np.float64)
    if host.size != 1 or not np.isfinite(host.reshape(())):
        raise ValueError(f"CLM parameter {name!r} must contain one finite scalar")
    return np.float64(host.reshape(()))


def _original_soil_interfaces() -> np.ndarray:
    node = np.empty(10, dtype=np.float64)
    for level in range(1, 11):
        node[level - 1] = np.float64(0.025) * (
            np.exp(np.float64(0.5) * (np.float64(level) - np.float64(0.5)))
            - np.float64(1.0)
        )
    thickness = np.empty(10, dtype=np.float64)
    thickness[0] = np.float64(0.5) * (node[0] + node[1])
    for level in range(1, 9):
        thickness[level] = np.float64(0.5) * (node[level + 1] - node[level - 1])
    thickness[-1] = node[-1] - node[-2]
    interface = np.empty(11, dtype=np.float64)
    interface[0] = np.float64(0.0)
    for level in range(1, 10):
        interface[level] = np.float64(0.5) * (node[level - 1] + node[level])
    interface[-1] = node[-1] + np.float64(0.5) * thickness[-1]
    return interface


def _surface_soil_sand(
    topology: SubgridTopology,
    surface_data_path: Path,
) -> np.ndarray:
    values = load_native_variables(surface_data_path, ["PCT_SAND", "PCT_CLAY"])
    longitude = np.asarray(topology.grid.longitude_index, dtype=np.int64)
    latitude = np.asarray(topology.grid.latitude_index, dtype=np.int64)
    sand = np.asarray(values["PCT_SAND"], dtype=np.float64)[:, latitude, longitude]
    clay = np.asarray(values["PCT_CLAY"], dtype=np.float64)[:, latitude, longitude]
    if sand.shape[0] != 10 or clay.shape != sand.shape:
        raise ValueError("fsurdat PCT_SAND/PCT_CLAY must contain ten texture levels")
    zero_surface = sand[0] + clay[0] == np.float64(0.0)
    inconsistent = zero_surface & np.any(sand + clay != np.float64(0.0), axis=0)
    if np.any(inconsistent):
        raise ValueError("soil texture has zero surface but nonzero deeper levels")
    sand[:, zero_surface] = np.float64(1.0)
    clay[:, zero_surface] = np.float64(1.0)
    if np.any(sand + clay == np.float64(0.0)):
        raise ValueError("soil texture contains a zero sand-plus-clay level")

    _, interface, _ = soil_layer_geometry_20sl_8p5m()
    original_interface = _original_soil_interfaces()
    texture_level = np.empty(NUMBER_OF_DECOMPOSITION_LEVELS, dtype=np.int64)
    texture_level[0] = 0
    for level in range(1, 20):
        texture_level[level] = np.searchsorted(
            original_interface[1:-1],
            interface[level + 1],
            side="right",
        )
    texture_level[20:] = 9
    column_gridcell = np.asarray(topology.column.gridcell, dtype=np.int64)
    return sand[texture_level][:, column_gridcell].T


def _cascade_arrays(
    column_sand: np.ndarray,
    parameter: dict[str, np.float64],
) -> tuple[np.ndarray, np.ndarray, SoilBiogeochemPotentialParameters]:
    column_count = column_sand.shape[0]
    respiration = np.zeros(
        (
            column_count,
            NUMBER_OF_DECOMPOSITION_LEVELS,
            NUMBER_OF_DECOMPOSITION_TRANSITIONS,
        ),
        dtype=np.float64,
    )
    path = np.zeros_like(respiration)

    respiration[:, :, 0] = parameter["rf_l1s1_bgc"]
    respiration[:, :, 1] = parameter["rf_l2s1_bgc"]
    respiration[:, :, 2] = parameter["rf_l3s2_bgc"]
    path[:, :, :3] = np.float64(1.0)

    texture_term = np.float64(0.68) * np.float64(0.01)
    texture_term = texture_term * (np.float64(100.0) - column_sand)
    soil1_respiration = np.float64(0.85) - texture_term
    denominator = np.float64(1.0) - soil1_respiration
    soil1_to_soil3 = np.float64(0.004) / denominator
    soil1_to_soil2 = np.float64(1.0) - soil1_to_soil3
    respiration[:, :, 3] = soil1_respiration
    respiration[:, :, 4] = soil1_respiration
    path[:, :, 3] = soil1_to_soil2
    path[:, :, 4] = soil1_to_soil3

    respiration[:, :, 5] = parameter["rf_s2s1_bgc"]
    respiration[:, :, 6] = parameter["rf_s2s3_bgc"]
    respiration[:, :, 7] = parameter["rf_s3s1_bgc"]
    path[:, :, 5] = np.float64(0.42) / np.float64(0.45)
    path[:, :, 6] = np.float64(0.03) / np.float64(0.45)
    path[:, :, 7] = np.float64(1.0)

    respiration[:, :, 8] = parameter["rf_cwdl2_bgc"]
    respiration[:, :, 9] = parameter["rf_cwdl3_bgc"]
    path[:, :, 8] = parameter["cwd_fcel"]
    path[:, :, 9] = parameter["cwd_flig"]
    parameters = SoilBiogeochemPotentialParameters(
        decomposition_level_count=20,
        decomposition_pool_count=7,
        decomposition_transition_count=10,
        use_fates=False,
        cascade_donor_pool=jnp.asarray(
            [0, 1, 2, 4, 4, 5, 5, 6, 3, 3],
            dtype=jnp.int32,
        ),
        cascade_receiver_pool=jnp.asarray(
            [4, 4, 5, 5, 6, 4, 6, 4, 1, 2],
            dtype=jnp.int32,
        ),
        floating_cn_ratio_pool=jnp.asarray(
            [True, True, True, True, False, False, False],
            dtype=jnp.bool_,
        ),
        initial_cn_ratio_pool=jnp.asarray(
            [
                90.0,
                90.0,
                90.0,
                90.0,
                parameter["cn_s1_bgc"],
                parameter["cn_s2_bgc"],
                parameter["cn_s3_bgc"],
            ],
            dtype=jnp.float64,
        ),
        denitrification_proportion=parameter["dnp"],
    )
    return respiration, path, parameters


def load_soil_biogeochem_potential_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> SoilBiogeochemPotentialNativeData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    required = {
        "use_cn": True,
        "use_century_decomp": True,
        "use_vertsoilc": True,
        "use_fates": False,
        "soil_layerstruct": "20SL_8.5m",
    }
    for key, expected in required.items():
        if general.get(key) != expected:
            raise ValueError(
                f"native SoilBiogeochemPotential adapter requires {key}={expected!r}"
            )
    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    surface_data = _selected_input_path(case, "fsurdat", surface_data_path)
    parameter_data = _selected_input_path(case, "paramfile", parameter_data_path)
    dimensions = load_dimension_sizes(restart)
    if dimensions.get("levgrnd") != NUMBER_OF_DECOMPOSITION_LEVELS:
        raise ValueError("restart must contain 25 native ground levels")
    topology = load_subgrid_topology(restart)
    column_count = topology.column.gridcell.shape[0]
    active = np.asarray(topology.column.active, dtype=bool)
    landunit_type = np.asarray(topology.column.landunit_type, dtype=np.int32)
    soil_column = active & (
        (landunit_type == LANDUNIT_SOIL) | (landunit_type == LANDUNIT_CROP)
    )

    restart_values = load_native_variables(
        restart,
        [*_CARBON_RESTART_VARIABLES, *_NITROGEN_RESTART_VARIABLES],
        dtype=np.float64,
    )
    for name, value in restart_values.items():
        if value.shape != (column_count, NUMBER_OF_DECOMPOSITION_LEVELS):
            raise ValueError(f"restart variable {name!r} must have shape (column, 25)")
    carbon = np.stack(
        [np.asarray(restart_values[name]) for name in _CARBON_RESTART_VARIABLES],
        axis=2,
    )
    nitrogen = np.stack(
        [np.asarray(restart_values[name]) for name in _NITROGEN_RESTART_VARIABLES],
        axis=2,
    )
    active_pool_values = np.concatenate(
        (carbon[soil_column, :20, :].reshape(-1), nitrogen[soil_column, :20, :].reshape(-1))
    )
    if not np.all(np.isfinite(active_pool_values)):
        raise ValueError("filtered soil columns contain missing decomposition C/N pools")

    raw_parameters = load_native_variables(
        parameter_data,
        list(_PARAMETER_VARIABLES),
        dtype=np.float64,
    )
    parameter = {
        name: _finite_scalar(name, raw_parameters[name])
        for name in _PARAMETER_VARIABLES
    }
    sand = _surface_soil_sand(topology, surface_data)
    respiration, path, parameters = _cascade_arrays(sand, parameter)
    return SoilBiogeochemPotentialNativeData(
        topology=topology,
        soil_column=jnp.asarray(soil_column),
        pools=SoilBiogeochemPotentialPoolState(
            decomposition_carbon_pool_column_level_pool=jnp.asarray(carbon),
            decomposition_nitrogen_pool_column_level_pool=jnp.asarray(nitrogen),
        ),
        respiration_fraction_column_level_transition=jnp.asarray(respiration),
        path_fraction_column_level_transition=jnp.asarray(path),
        parameters=parameters,
        restart_path=restart,
        surface_data_path=surface_data,
        parameter_data_path=parameter_data,
    )


def assemble_soil_biogeochem_potential_inputs(
    native: SoilBiogeochemPotentialNativeData,
    pools: SoilBiogeochemPotentialPoolState | None = None,
) -> SoilBiogeochemPotentialInputs:

    current = native.pools if pools is None else pools
    column_count = native.soil_column.shape[0]
    expected = (
        column_count,
        NUMBER_OF_DECOMPOSITION_LEVELS,
        NUMBER_OF_DECOMPOSITION_POOLS,
    )
    for name in SoilBiogeochemPotentialPoolState._fields:
        if jnp.shape(getattr(current, name)) != expected:
            raise ValueError(f"{name} must have shape (column, 25, 7)")
    return SoilBiogeochemPotentialInputs(
        soil_column=native.soil_column,
        decomposition_carbon_pool_column_level_pool=(
            current.decomposition_carbon_pool_column_level_pool
        ),
        decomposition_nitrogen_pool_column_level_pool=(
            current.decomposition_nitrogen_pool_column_level_pool
        ),
        respiration_fraction_column_level_transition=(
            native.respiration_fraction_column_level_transition
        ),
        path_fraction_column_level_transition=(
            native.path_fraction_column_level_transition
        ),
    )


__all__ = [
    "SoilBiogeochemPotentialNativeData",
    "SoilBiogeochemPotentialPoolState",
    "assemble_soil_biogeochem_potential_inputs",
    "load_soil_biogeochem_potential_native_data",
]
