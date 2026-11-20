
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.native_templates import native_template

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    RestartClock,
    SubgridTopology,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)
from clm_jax.physics.urban_albedo import (
    UrbanAlbedoInputs,
    UrbanAlbedoState,
    UrbanCanopyAlbedoState,
    UrbanSurfaceAlbedoState,
)
from clm_jax.physics.urban_radiation import UrbanShortwaveAbsorptionState


class UrbanAlbedoRestartData(NamedTuple):
    topology: SubgridTopology
    clock: RestartClock
    inputs: UrbanAlbedoInputs
    state: UrbanAlbedoState
    surface_data_path: Path


_RESTART_VARIABLES = (
    "coszen",
    "frac_sno",
    "H2OSNO",
    "sabs_roof_dir",
    "sabs_roof_dif",
    "sabs_sunwall_dir",
    "sabs_sunwall_dif",
    "sabs_shadewall_dir",
    "sabs_shadewall_dif",
    "sabs_improad_dir",
    "sabs_improad_dif",
    "sabs_perroad_dir",
    "sabs_perroad_dif",
)

_SCALAR_SURFACE_VARIABLES = ("CANYON_HWR", "WTROAD_PERV")
_BAND_SURFACE_VARIABLES = (
    "ALB_ROOF_DIR",
    "ALB_ROOF_DIF",
    "ALB_IMPROAD_DIR",
    "ALB_IMPROAD_DIF",
    "ALB_PERROAD_DIR",
    "ALB_PERROAD_DIF",
    "ALB_WALL_DIR",
    "ALB_WALL_DIF",
)


def _finite(value: jnp.ndarray) -> jnp.ndarray:
    value = jnp.asarray(value, dtype=jnp.float64)
    return jnp.where(jnp.isnan(value), jnp.float64(CLM_SPECIAL_VALUE), value)


def _configured_surface_data(case: CimeCase) -> Path:
    group = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    declared = group.get("fsurdat")
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(f"clm_inparm in {case.case_docs / 'lnd_in'} lacks fsurdat")
    path = Path(case.expand(declared)).expanduser()
    if not path.is_absolute():
        path = case.root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM surface data does not exist: {path}")
    return path


def _select_urban_surface_parameters(
    topology: SubgridTopology,
    surface: dict[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    landunit_type = np.asarray(topology.landunit.landunit_type, dtype=np.int64)
    urban = (landunit_type >= LANDUNIT_URBAN_MIN) & (landunit_type <= LANDUNIT_URBAN_MAX)
    number_of_landunits = landunit_type.size
    result: dict[str, jnp.ndarray] = {}
    if not np.any(urban):
        for name in _SCALAR_SURFACE_VARIABLES:
            result[name] = jnp.full((number_of_landunits,), CLM_SPECIAL_VALUE, dtype=jnp.float64)
        for name in _BAND_SURFACE_VARIABLES:
            result[name] = jnp.full((number_of_landunits, 2), CLM_SPECIAL_VALUE, dtype=jnp.float64)
        return result

    parent = np.asarray(topology.landunit.gridcell, dtype=np.int64)[urban]
    longitude = np.asarray(topology.grid.longitude_index, dtype=np.int64)[parent]
    latitude = np.asarray(topology.grid.latitude_index, dtype=np.int64)[parent]
    density = landunit_type[urban] - LANDUNIT_URBAN_MIN
    for name in _SCALAR_SURFACE_VARIABLES:
        source = np.asarray(surface[name], dtype=np.float64)
        if source.ndim != 3 or source.shape[0] != 3:
            raise ValueError(f"surface variable {name} must be (numurbl, lat, lon)")
        selected = source[density, latitude, longitude]
        if not np.all(np.isfinite(selected)):
            raise ValueError(f"surface variable {name} is missing on an urban landunit")
        values = np.full(number_of_landunits, CLM_SPECIAL_VALUE, dtype=np.float64)
        values[urban] = selected
        result[name] = jnp.asarray(values)
    for name in _BAND_SURFACE_VARIABLES:
        source = np.asarray(surface[name], dtype=np.float64)
        if source.ndim != 4 or source.shape[:2] != (2, 3):
            raise ValueError(f"surface variable {name} must be (numrad, numurbl, lat, lon)")
        selected = source[:, density, latitude, longitude].T
        if not np.all(np.isfinite(selected)):
            raise ValueError(f"surface variable {name} is missing on an urban landunit")
        values = np.full((number_of_landunits, 2), CLM_SPECIAL_VALUE, dtype=np.float64)
        values[urban] = selected
        result[name] = jnp.asarray(values)
    return result


def _empty_surface_state(
    dimensions: dict[str, int],
) -> UrbanSurfaceAlbedoState:
    column = jnp.full((dimensions["column"], 2), CLM_SPECIAL_VALUE, dtype=jnp.float64)
    patch = jnp.full((dimensions["pft"], 2), CLM_SPECIAL_VALUE, dtype=jnp.float64)
    canopy = UrbanCanopyAlbedoState(*(patch for _ in UrbanCanopyAlbedoState._fields))
    return UrbanSurfaceAlbedoState(column, column, patch, patch, canopy)


def _validate_surface_state(
    value: UrbanSurfaceAlbedoState,
    dimensions: dict[str, int],
) -> UrbanSurfaceAlbedoState:
    column_shape = (dimensions["column"], 2)
    patch_shape = (dimensions["pft"], 2)
    fields = {
        name: jnp.asarray(getattr(value, name), dtype=jnp.float64)
        for name in UrbanSurfaceAlbedoState._fields[:-1]
    }
    for name in ("ground_direct_column", "ground_diffuse_column"):
        if fields[name].shape != column_shape:
            raise ValueError(f"preceding {name} must have shape {column_shape}")
    for name in ("surface_direct_patch", "surface_diffuse_patch"):
        if fields[name].shape != patch_shape:
            raise ValueError(f"preceding {name} must have shape {patch_shape}")
    canopy = {
        name: jnp.asarray(getattr(value.canopy, name), dtype=jnp.float64)
        for name in UrbanCanopyAlbedoState._fields
    }
    for name, array in canopy.items():
        if array.shape != patch_shape:
            raise ValueError(f"preceding canopy {name} must have shape {patch_shape}")
    return UrbanSurfaceAlbedoState(**fields, canopy=UrbanCanopyAlbedoState(**canopy))


@native_template
def load_urban_albedo_template(case, restart_path=None, surface_data_path=None):
    return load_urban_albedo_restart(case, restart_path, surface_data_path)


def load_urban_albedo_restart(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    cosine_solar_zenith_column: jnp.ndarray | None = None,
    preceding_surface_state: UrbanSurfaceAlbedoState | None = None,
) -> UrbanAlbedoRestartData:

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    surface_path = (
        _configured_surface_data(case)
        if surface_data_path is None
        else Path(surface_data_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    if not surface_path.is_file():
        raise FileNotFoundError(f"CLM surface data does not exist: {surface_path}")
    dimensions = load_dimension_sizes(restart)
    for name in ("landunit", "column", "pft", "numrad"):
        if name not in dimensions:
            raise ValueError(f"restart lacks dimension {name}: {restart}")
    if dimensions["numrad"] != 2:
        raise ValueError("UrbanAlbedo requires the two native CLM solar bands")

    topology = load_subgrid_topology(restart)
    restart_values = load_native_variables(restart, list(_RESTART_VARIABLES))
    surface_values = load_native_variables(
        surface_path,
        list(_SCALAR_SURFACE_VARIABLES + _BAND_SURFACE_VARIABLES),
    )
    parameters = _select_urban_surface_parameters(topology, surface_values)
    cosine = (
        _finite(restart_values["coszen"])
        if cosine_solar_zenith_column is None
        else jnp.asarray(cosine_solar_zenith_column, dtype=jnp.float64)
    )
    if cosine.shape != (dimensions["column"],):
        raise ValueError(f"cosine_solar_zenith_column must have shape ({dimensions['column']},)")
    inputs = UrbanAlbedoInputs(
        cosine_solar_zenith_column=cosine,
        snow_fraction_column=_finite(restart_values["frac_sno"]),
        snow_water_column=_finite(restart_values["H2OSNO"]),
        canyon_height_width_ratio_landunit=parameters["CANYON_HWR"],
        pervious_road_fraction_landunit=parameters["WTROAD_PERV"],
        roof_direct_albedo_landunit=parameters["ALB_ROOF_DIR"],
        roof_diffuse_albedo_landunit=parameters["ALB_ROOF_DIF"],
        impervious_road_direct_albedo_landunit=parameters["ALB_IMPROAD_DIR"],
        impervious_road_diffuse_albedo_landunit=parameters["ALB_IMPROAD_DIF"],
        pervious_road_direct_albedo_landunit=parameters["ALB_PERROAD_DIR"],
        pervious_road_diffuse_albedo_landunit=parameters["ALB_PERROAD_DIF"],
        wall_direct_albedo_landunit=parameters["ALB_WALL_DIR"],
        wall_diffuse_albedo_landunit=parameters["ALB_WALL_DIF"],
    )
    absorption = UrbanShortwaveAbsorptionState(
        roof_direct_landunit=_finite(restart_values["sabs_roof_dir"]),
        roof_diffuse_landunit=_finite(restart_values["sabs_roof_dif"]),
        sun_wall_direct_landunit=_finite(restart_values["sabs_sunwall_dir"]),
        sun_wall_diffuse_landunit=_finite(restart_values["sabs_sunwall_dif"]),
        shade_wall_direct_landunit=_finite(restart_values["sabs_shadewall_dir"]),
        shade_wall_diffuse_landunit=_finite(restart_values["sabs_shadewall_dif"]),
        impervious_road_direct_landunit=_finite(restart_values["sabs_improad_dir"]),
        impervious_road_diffuse_landunit=_finite(restart_values["sabs_improad_dif"]),
        pervious_road_direct_landunit=_finite(restart_values["sabs_perroad_dir"]),
        pervious_road_diffuse_landunit=_finite(restart_values["sabs_perroad_dif"]),
    )
    surface_state = (
        _empty_surface_state(dimensions)
        if preceding_surface_state is None
        else _validate_surface_state(preceding_surface_state, dimensions)
    )
    return UrbanAlbedoRestartData(
        topology=topology,
        clock=load_restart_clock(restart),
        inputs=inputs,
        state=UrbanAlbedoState(absorption, surface_state),
        surface_data_path=surface_path,
    )
