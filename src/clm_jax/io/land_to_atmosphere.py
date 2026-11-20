
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.forcing.area import load_land_flux_area_factor
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.cn_fire_emissions import (
    load_cn_fire_emissions_configuration,
    require_disabled_cn_fire_emissions,
)
from clm_jax.io.coupler import load_river_volume_coupler_state
from clm_jax.io.dry_deposition import load_dry_deposition_configuration
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.native_templates import native_template
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import (
    configured_land_restart,
    load_subgrid_topology,
)
from clm_jax.io.urban_albedo_restart import load_urban_albedo_restart
from clm_jax.physics.land_to_atmosphere import (
    LandToAtmosphereParameters,
    LandToAtmosphereState,
    LandToAtmosphereTopology,
)
from clm_jax.physics.land_to_atmosphere_composition import (
    LandToAtmosphereExternalFields,
)

_EARTH_RADIUS_KILOMETERS = np.float64(6371.22)


class LandToAtmosphereNativeData(NamedTuple):

    topology: LandToAtmosphereTopology
    state: LandToAtmosphereState
    parameters: LandToAtmosphereParameters
    external: LandToAtmosphereExternalFields
    land_restart_path: Path
    coupler_restart_path: Path
    land_domain_path: Path
    surface_data_path: Path
    runoff_mapping_path: Path


def _configured_clm_path(case: CimeCase, name: str) -> Path:
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


def _configured_case_path(case: CimeCase, path_key: str, file_key: str) -> Path:
    parent = Path(case.expand(case.value(path_key))).expanduser()
    if not parent.is_absolute():
        parent = case.root / parent
    path = (parent / case.expand(case.value(file_key))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured {file_key} is absent: {path}")
    return path


def _one_cpl_input(case: CimeCase, key: str) -> Path:
    matches = {
        entry.path.resolve()
        for entry in case.input_file_entries()
        if entry.component == "cpl" and entry.key == key and entry.path.is_file()
    }
    if len(matches) != 1:
        raise ValueError(f"expected one existing cpl:{key} input, found {tuple(matches)}")
    return next(iter(matches))


def _companion_coupler_restart(land_restart: Path) -> Path:
    marker = ".clm2.r."
    if marker not in land_restart.name:
        raise ValueError(f"cannot derive CPL companion from {land_restart.name}")
    candidate = land_restart.with_name(land_restart.name.replace(marker, ".cpl.r.", 1))
    if not candidate.is_file():
        raise FileNotFoundError(f"companion CPL restart does not exist: {candidate}")
    return candidate.resolve()


def _gridcell_area_from_domain(
    topology,
    domain_path: Path,
) -> jnp.ndarray:
    area_radians_squared = np.asarray(
        load_native_variables(domain_path, ["area"], dtype=np.float64)["area"],
        dtype=np.float64,
    )
    if area_radians_squared.ndim != 2:
        raise ValueError("land-domain area must have dimensions (lat, lon)")
    latitude = np.asarray(topology.grid.latitude_index, dtype=np.int64)
    longitude = np.asarray(topology.grid.longitude_index, dtype=np.int64)
    if (
        np.any(latitude < 0)
        or np.any(latitude >= area_radians_squared.shape[0])
        or np.any(longitude < 0)
        or np.any(longitude >= area_radians_squared.shape[1])
    ):
        raise ValueError("native grid indices lie outside the configured land domain")
    selected = area_radians_squared[latitude, longitude]
    area = selected * (_EARTH_RADIUS_KILOMETERS**2)
    if np.any(~np.isfinite(area)) or np.any(area <= np.float64(0.0)):
        raise ValueError("active native land grid contains invalid domain area")
    return jnp.asarray(area, dtype=jnp.float64)


def _ice_runoff_melted_gridcell(
    case: CimeCase,
    topology,
    surface_data_path: Path,
) -> jnp.ndarray:
    region_source = np.asarray(
        load_native_variables(
            surface_data_path,
            ["GLACIER_REGION"],
        )["GLACIER_REGION"],
        dtype=np.int64,
    )
    if region_source.ndim != 2:
        raise ValueError("surface GLACIER_REGION must have dimensions (lat, lon)")
    latitude = np.asarray(topology.grid.latitude_index, dtype=np.int64)
    longitude = np.asarray(topology.grid.longitude_index, dtype=np.int64)
    if (
        np.any(latitude < 0)
        or np.any(latitude >= region_source.shape[0])
        or np.any(longitude < 0)
        or np.any(longitude >= region_source.shape[1])
    ):
        raise ValueError("native grid indices lie outside surface GLACIER_REGION")
    region = region_source[latitude, longitude]
    behavior = dict(
        read_case_namelist_group(case, "lnd_in", "clm_glacier_behavior")
    ).get("glacier_region_ice_runoff_behavior")
    if not isinstance(behavior, tuple):
        raise ValueError("glacier_region_ice_runoff_behavior must be an indexed array")
    normalized = tuple(str(value).strip().lower() for value in behavior)
    allowed = frozenset(("melted", "remains_ice"))
    if any(value not in allowed for value in normalized):
        raise ValueError(f"unsupported glacier ice-runoff behavior: {normalized}")
    if np.any(region < 0) or np.any(region >= len(normalized)):
        raise ValueError("surface GLACIER_REGION is not covered by ice-runoff behavior")
    return jnp.asarray(
        np.asarray([normalized[index] == "melted" for index in region], dtype=bool)
    )


def _restart_state(restart_path: Path, column_count: int, gridcell_count: int):
    values = load_native_variables(restart_path, ["TWS"], dtype=np.float64)
    total_water = jnp.asarray(values["TWS"], dtype=jnp.float64)
    if total_water.shape != (gridcell_count,):
        raise ValueError("restart TWS does not match the native gridcell axis")
    zero_column = jnp.zeros(column_count, dtype=jnp.float64)
    return LandToAtmosphereState(
        groundwater_runoff_column=zero_column,
        total_runoff_column=zero_column,
        liquid_from_ice_column=zero_column,
        ice_to_liquid_sensible_heat_column=zero_column,
        total_water_storage_gridcell=total_water,
    )


def load_dynamic_landcover_fluxes_from_restart(
    restart_path: str | Path,
    *,
    days_per_year: float = 365.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    names = (
        "eflx_dynbal_amt_to_dribble",
        "qflx_liq_dynbal_amt_to_dribble",
        "qflx_ice_dynbal_amt_to_dribble",
    )
    values = load_native_variables(restart_path, list(names), dtype=np.float64)
    seconds_per_year = jnp.float64(days_per_year) * jnp.float64(86_400.0)
    arrays = tuple(jnp.asarray(values[name], dtype=jnp.float64) for name in names)
    if len({array.shape for array in arrays}) != 1 or arrays[0].ndim != 1:
        raise ValueError("dynamic-landcover dribbler pools must share one gridcell axis")
    return tuple(array / seconds_per_year for array in arrays)


@native_template
def load_land_to_atmosphere_native_data(
    case: CimeCase,
    land_restart_path: str | Path | None = None,
    coupler_restart_path: str | Path | None = None,
) -> LandToAtmosphereNativeData:

    land_restart = (
        configured_land_restart(case)
        if land_restart_path is None
        else Path(land_restart_path).expanduser().resolve()
    )
    if not land_restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {land_restart}")
    coupler_restart = (
        _companion_coupler_restart(land_restart)
        if coupler_restart_path is None
        else Path(coupler_restart_path).expanduser().resolve()
    )
    if not coupler_restart.is_file():
        raise FileNotFoundError(f"CPL restart does not exist: {coupler_restart}")

    subgrid = load_subgrid_topology(land_restart)
    surface_data = _configured_clm_path(case, "fsurdat")
    land_domain = _configured_case_path(case, "LND_DOMAIN_PATH", "LND_DOMAIN_FILE")
    runoff_mapping = _one_cpl_input(case, "rof2lnd_fmapname")
    area = _gridcell_area_from_domain(subgrid, land_domain)
    urban = load_urban_albedo_restart(
        case,
        restart_path=land_restart,
        surface_data_path=surface_data,
    )
    topology = LandToAtmosphereTopology(
        subgrid=subgrid,
        canyon_height_width_ratio_landunit=(
            urban.inputs.canyon_height_width_ratio_landunit
        ),
        ice_runoff_melted_gridcell=_ice_runoff_melted_gridcell(
            case,
            subgrid,
            surface_data,
        ),
        gridcell_area_square_kilometers=area,
    )

    dry_deposition = load_dry_deposition_configuration(case)
    if dry_deposition.number_of_species:
        raise NotImplementedError(
            "active dry-deposition fluxes must be supplied by the same-step "
            "DryDepVelocity result"
        )
    fire = load_cn_fire_emissions_configuration(case)
    require_disabled_cn_fire_emissions(fire)
    patch_count = subgrid.patch.gridcell.shape[0]
    gridcell_count = subgrid.grid.longitude_degrees.shape[0]
    energy_dynamic, liquid_dynamic, ice_dynamic = (
        load_dynamic_landcover_fluxes_from_restart(land_restart)
    )
    if energy_dynamic.shape != (gridcell_count,):
        raise ValueError("dynamic-landcover restart pools disagree with native topology")
    river_volume = load_river_volume_coupler_state(
        coupler_restart,
        subgrid,
        runoff_mapping,
        area,
        land_flux_area_factor=load_land_flux_area_factor(case, subgrid),
    )
    external = LandToAtmosphereExternalFields(
        dry_deposition_velocity_patch=jnp.zeros(
            (patch_count, dry_deposition.number_of_species),
            dtype=jnp.float64,
        ),
        fire_flux_patch=jnp.zeros(
            (patch_count, fire.mechanism_component_count),
            dtype=jnp.float64,
        ),
        fire_injection_top_patch=jnp.zeros(patch_count, dtype=jnp.float64),
        dynamic_sensible_heat_gridcell=energy_dynamic,
        dynamic_liquid_water_gridcell=liquid_dynamic,
        dynamic_ice_water_gridcell=ice_dynamic,
        river_volume_gridcell=river_volume,
    )
    return LandToAtmosphereNativeData(
        topology=topology,
        state=_restart_state(
            land_restart,
            subgrid.column.gridcell.shape[0],
            gridcell_count,
        ),
        parameters=LandToAtmosphereParameters.from_case(case),
        external=external,
        land_restart_path=land_restart,
        coupler_restart_path=coupler_restart,
        land_domain_path=land_domain,
        surface_data_path=surface_data,
        runoff_mapping_path=runoff_mapping,
    )


__all__ = [
    "LandToAtmosphereNativeData",
    "load_dynamic_landcover_fluxes_from_restart",
    "load_land_to_atmosphere_native_data",
]
