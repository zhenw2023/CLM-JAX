
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.forcing.atmosphere import Atm2LndParameters
from clm_jax.forcing.remap import load_sparse_mapping
from clm_jax.io.cime_case import CimeCase, InputFileEntry
from clm_jax.io.coupler import load_glacier_coupler_state
from clm_jax.io.driver import load_driver_initialization_native_data
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_variables
from clm_jax.io.restart import load_restart_clock
from clm_jax.physics.atmospheric_driver import (
    NUMBER_OF_SNOW_LEVELS,
    AtmosphericDriverInputs,
    AtmosphericDriverState,
    AtmosphericDriverStatic,
)
from clm_jax.physics.daylength import initialize_daylength
from clm_jax.physics.solar import FIXED_YEAR_2000_ORBIT, solar_declination
from clm_jax.physics.solar_runtime import native_grid_radians
from clm_jax.physics.topography import (
    ColumnTopography,
    collapse_mask_from_region_behavior,
    map_glacier_state_to_land,
    select_land_gridcells,
)
from clm_jax.trajectory import noleap_calendar_day_from_clock, noleap_clock_from_restart


class AtmosphericDriverNativeData(NamedTuple):

    static: AtmosphericDriverStatic
    state: AtmosphericDriverState
    inputs: AtmosphericDriverInputs
    land_restart_path: Path
    coupler_restart_path: Path


def _one_entry(
    case: CimeCase,
    *,
    component: str,
    key: str,
    parent_name: str | None = None,
) -> InputFileEntry:
    matches = [
        entry
        for entry in case.input_file_entries()
        if entry.component == component
        and entry.key == key
        and (parent_name is None or parent_name in entry.path.parts)
    ]
    existing = [entry for entry in matches if entry.path.is_file()]
    unique = {entry.path.resolve(): entry for entry in existing}
    if len(unique) != 1:
        raise ValueError(
            f"expected one existing {component}:{key} input"
            f"{f' below {parent_name}' if parent_name else ''}, found {tuple(unique)}"
        )
    return next(iter(unique.values()))


def _companion_coupler_restart(land_restart: Path) -> Path:
    marker = ".clm2.r."
    if marker not in land_restart.name:
        raise ValueError(f"cannot derive CPL companion from {land_restart.name}")
    candidate = land_restart.with_name(land_restart.name.replace(marker, ".cpl.r.", 1))
    if not candidate.is_file():
        raise FileNotFoundError(f"companion CPL restart does not exist: {candidate}")
    return candidate.resolve()


def _region_flag(
    glacier_region: np.ndarray,
    values: tuple[str, ...],
    *,
    enabled_value: str,
    allowed: frozenset[str],
) -> jnp.ndarray:
    normalized = tuple(value.strip() for value in values)
    invalid = tuple(value for value in normalized if value not in allowed)
    if invalid:
        raise ValueError(f"unsupported glacier behavior values: {invalid}")
    flags = np.asarray([value == enabled_value for value in normalized], dtype=bool)
    if glacier_region.size and (
        np.min(glacier_region) < 0 or np.max(glacier_region) >= flags.size
    ):
        raise ValueError("surface-data GLACIER_REGION lies outside namelist behavior array")
    return jnp.asarray(flags[glacier_region])


def load_atmospheric_driver_native_data(
    case: CimeCase,
    land_restart_path: str | Path,
    coupler_restart_path: str | Path | None = None,
) -> AtmosphericDriverNativeData:

    land_restart = Path(land_restart_path).expanduser().resolve()
    driver = load_driver_initialization_native_data(case, land_restart)
    if driver.number_of_snow_levels != NUMBER_OF_SNOW_LEVELS:
        raise ValueError(
            f"active driver requires nlevsno={NUMBER_OF_SNOW_LEVELS}, "
            f"got {driver.number_of_snow_levels}"
        )
    coupler_restart = (
        _companion_coupler_restart(land_restart)
        if coupler_restart_path is None
        else Path(coupler_restart_path).expanduser().resolve()
    )
    if not coupler_restart.is_file():
        raise FileNotFoundError(f"CPL restart does not exist: {coupler_restart}")

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    glacier_configuration = read_case_namelist_group(
        case, "lnd_in", "clm_glacier_behavior"
    )
    if general.get("glc_do_dynglacier") is not True:
        raise ValueError("active atmospheric driver requires glc_do_dynglacier=.true.")
    if int(general.get("maxpatch_glcmec", -1)) != 10:
        raise ValueError("active atmospheric driver requires maxpatch_glcmec=10")

    glacier_mapping_path = _one_entry(
        case,
        component="cpl",
        key="glc2lnd_fmapname",
    ).path
    surface_path = _one_entry(case, component="clm", key="fsurdat").path
    atmospheric_topography_path = _one_entry(
        case,
        component="datm",
        key="file1",
        parent_name="topo_forcing",
    ).path

    topology = driver.topology
    full_grid_index = np.asarray(topology.grid.latitude_index, dtype=np.int64) * 288 + np.asarray(
        topology.grid.longitude_index, dtype=np.int64
    )
    mapping = load_sparse_mapping(glacier_mapping_path)
    from clm_jax.io.glacier_mapping import prepare_glacier_mapping

    mapping = prepare_glacier_mapping(case, mapping, "GLC2LND_FMAPTYPE")
    full_glacier = map_glacier_state_to_land(
        load_glacier_coupler_state(coupler_restart),
        mapping,
    )
    glacier = select_land_gridcells(full_glacier, jnp.asarray(full_grid_index))

    atmospheric_elevation_full = load_variables(
        atmospheric_topography_path,
        ["TOPO"],
        indexers={"time": 0},
        dtype=np.float64,
    )["TOPO"].reshape(-1)
    if atmospheric_elevation_full.size != 192 * 288:
        raise ValueError("configured atmospheric topography is not f09 (192x288)")
    atmospheric_elevation = jnp.asarray(atmospheric_elevation_full[full_grid_index])

    glacier_region_full = load_variables(surface_path, ["GLACIER_REGION"])[
        "GLACIER_REGION"
    ].reshape(-1)
    glacier_region = np.asarray(glacier_region_full[full_grid_index], dtype=np.int32)
    region_behavior = glacier_configuration["glacier_region_behavior"]
    rain_behavior = glacier_configuration["glacier_region_rain_to_snow_behavior"]
    if not isinstance(region_behavior, tuple) or not isinstance(rain_behavior, tuple):
        raise ValueError("glacier behavior namelist values must be indexed arrays")
    collapse = collapse_mask_from_region_behavior(glacier_region, region_behavior)
    rain_to_snow_runs_off = _region_flag(
        glacier_region,
        rain_behavior,
        enabled_value="runs_off",
        allowed=frozenset(("runs_off", "converted_to_snow")),
    )

    restart_clock = load_restart_clock(land_restart)
    model_clock = noleap_clock_from_restart(restart_clock)
    calendar_day = noleap_calendar_day_from_clock(model_clock)
    previous_calendar_day = calendar_day - jnp.float64(restart_clock.step_seconds) / jnp.float64(
        86_400.0
    )
    declination, _ = solar_declination(calendar_day, FIXED_YEAR_2000_ORBIT)
    previous_declination, _ = solar_declination(
        previous_calendar_day,
        FIXED_YEAR_2000_ORBIT,
    )
    latitude_radians = native_grid_radians(topology.grid.latitude_degrees)
    daylength = initialize_daylength(
        latitude_radians,
        declination,
        previous_declination,
        FIXED_YEAR_2000_ORBIT.obliquity_radians,
    )
    column_topography = ColumnTopography(
        elevation=topology.column.elevation,
        needs_downscaling=jnp.zeros_like(topology.column.active),
        column_type=topology.column.column_type,
    )
    return AtmosphericDriverNativeData(
        static=AtmosphericDriverStatic(
            topology=topology,
            latitude_radians=latitude_radians,
            atmospheric_elevation=atmospheric_elevation,
            glacier=glacier,
            collapse_to_atmospheric_topography=collapse,
            rain_to_snow_runs_off=rain_to_snow_runs_off,
            parameters=Atm2LndParameters.from_case(case),
        ),
        state=AtmosphericDriverState(
            daylength=daylength,
            previous_snow_ice_fraction=driver.previous_snow_ice_fraction,
            column_topography=column_topography,
            first_step_of_run_segment=jnp.asarray(True, dtype=jnp.bool_),
        ),
        inputs=AtmosphericDriverInputs(
            intracellular_sunlit_co2=driver.intracellular_sunlit_co2,
            intracellular_shaded_co2=driver.intracellular_shaded_co2,
            snow_water_equivalent=driver.snow_water_equivalent,
            snow_layer_count=driver.snow_layer_count,
            snow_layer_ice=driver.snow_layer_ice,
            snow_layer_liquid=driver.snow_layer_liquid,
            vegetation_exposed_for_albedo=driver.vegetation_exposed_for_albedo,
        ),
        land_restart_path=land_restart,
        coupler_restart_path=coupler_restart,
    )


__all__ = [
    "AtmosphericDriverNativeData",
    "load_atmospheric_driver_native_data",
]
