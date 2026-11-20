
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.forcing.urban import UrbanTimeVaryingInputs
from clm_jax.io.bgc_external_forcing import (
    StreamBracket,
    _advance_noleap,
    _integer,
    _path,
    _read_records,
    _stream_bracket,
    _string,
)
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    RestartClock,
    SubgridTopology,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)

_FIELD_NAMES = ("tbuildmax_TBD", "tbuildmax_HD", "tbuildmax_MD")
_SECONDS_PER_DAY = 86_400


class UrbanTimeVaryingNativeData(NamedTuple):

    inputs: UrbanTimeVaryingInputs
    topology: SubgridTopology
    clock: RestartClock
    call_date: int
    call_time_of_day: int
    bracket: StreamBracket
    stream_path: Path
    surface_data_path: Path
    restart_path: Path


def _configured_surface_data(case: CimeCase) -> Path:
    group = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    declared = group.get("fsurdat")
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError("CLM namelist lacks required fsurdat")
    path = Path(case.expand(declared)).expanduser()
    if not path.is_absolute():
        path = case.root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM surface data does not exist: {path}")
    return path


def _native_grid_records(
    lower: np.ndarray,
    upper: np.ndarray,
    source_longitude: np.ndarray,
    source_latitude: np.ndarray,
    topology: SubgridTopology,
) -> tuple[np.ndarray, np.ndarray]:
    longitude_index = np.asarray(topology.grid.longitude_index, dtype=np.int64)
    latitude_index = np.asarray(topology.grid.latitude_index, dtype=np.int64)
    expected_longitude = source_longitude[longitude_index]
    expected_latitude = source_latitude[latitude_index]
    if not np.allclose(
        expected_longitude,
        np.asarray(topology.grid.longitude_degrees),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("urban stream longitude grid differs from the CLM grid")
    if not np.allclose(
        expected_latitude,
        np.asarray(topology.grid.latitude_degrees),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("urban stream latitude grid differs from the CLM grid")
    return (
        lower[latitude_index, longitude_index],
        upper[latitude_index, longitude_index],
    )


def load_urban_time_varying_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    *,
    call_date: int | None = None,
    call_time_of_day: int | None = None,
) -> UrbanTimeVaryingNativeData:

    if case.value("CALENDAR", "NO_LEAP").strip().upper() != "NO_LEAP":
        raise NotImplementedError("native urban stream adapter currently requires NO_LEAP")
    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    topology = load_subgrid_topology(restart)
    clock = load_restart_clock(restart)
    default_date, default_time = _advance_noleap(
        clock.current_date,
        clock.current_time_of_day,
        clock.step_seconds,
    )
    selected_date = default_date if call_date is None else call_date
    selected_time = default_time if call_time_of_day is None else call_time_of_day
    if not 0 <= selected_time < _SECONDS_PER_DAY:
        raise ValueError("call_time_of_day must lie in [0, 86400)")

    group = dict(read_case_namelist_group(case, "lnd_in", "urbantv_streams"))
    if _string(group, "urbantvmapalgo", "nn") != "nn":
        raise NotImplementedError("native urban stream adapter currently requires mapalgo='nn'")
    stream_path = _path(group, "stream_fldfilename_urbantv", case)
    bracket = _stream_bracket(
        stream_path,
        call_date=selected_date,
        call_time_of_day=selected_time,
        year_first=_integer(group, "stream_year_first_urbantv", 1),
        year_last=_integer(group, "stream_year_last_urbantv", 1),
        year_align=_integer(group, "model_year_align_urbantv", 1),
        time_axis_mode="extend",
        interpolation=_string(group, "urbantv_tintalgo", "linear"),
    )

    lower_fields: list[np.ndarray] = []
    upper_fields: list[np.ndarray] = []
    for field_name in _FIELD_NAMES:
        lower, upper, longitude, latitude, units = _read_records(
            stream_path,
            field_name,
            bracket.lower_index,
            bracket.upper_index,
        )
        if units != "K":
            raise ValueError(f"urban stream variable {field_name} must use K")
        mapped_lower, mapped_upper = _native_grid_records(
            lower,
            upper,
            longitude,
            latitude,
            topology,
        )
        lower_fields.append(mapped_lower)
        upper_fields.append(mapped_upper)

    surface_path = _configured_surface_data(case)
    region = np.asarray(
        load_native_variables(surface_path, ["URBAN_REGION_ID"])["URBAN_REGION_ID"],
        dtype=np.int64,
    )
    longitude_index = np.asarray(topology.grid.longitude_index, dtype=np.int64)
    latitude_index = np.asarray(topology.grid.latitude_index, dtype=np.int64)
    if region.ndim != 2 or np.any(latitude_index >= region.shape[0]) or np.any(
        longitude_index >= region.shape[1]
    ):
        raise ValueError("URBAN_REGION_ID is incompatible with the CLM grid")
    urban_valid = region[latitude_index, longitude_index] != 0

    lower_gridcell_density = np.stack(lower_fields, axis=1)
    upper_gridcell_density = np.stack(upper_fields, axis=1)
    landunit_gridcell = np.asarray(topology.landunit.gridcell, dtype=np.int64)
    landunit_type = np.asarray(topology.landunit.landunit_type, dtype=np.int64)
    urban = (landunit_type >= LANDUNIT_URBAN_MIN) & (
        landunit_type <= LANDUNIT_URBAN_MAX
    )
    if np.any((landunit_gridcell[urban] < 0) | (landunit_gridcell[urban] >= urban_valid.size)):
        raise ValueError("an urban landunit has an invalid parent gridcell")
    density = landunit_type[urban] - LANDUNIT_URBAN_MIN
    selected = lower_gridcell_density[landunit_gridcell[urban], density]
    if np.any(~urban_valid[landunit_gridcell[urban]]) or np.any(selected <= 0.0):
        raise ValueError("configured inputdata contains an invalid urban landunit")

    return UrbanTimeVaryingNativeData(
        inputs=UrbanTimeVaryingInputs(
            building_maximum_temperature_lower_gridcell_density=jnp.asarray(
                lower_gridcell_density
            ),
            building_maximum_temperature_upper_gridcell_density=jnp.asarray(
                upper_gridcell_density
            ),
            lower_weight=bracket.lower_weight,
            landunit_gridcell_index=topology.landunit.gridcell,
            landunit_type=topology.landunit.landunit_type,
            urban_valid_gridcell=jnp.asarray(urban_valid),
        ),
        topology=topology,
        clock=clock,
        call_date=selected_date,
        call_time_of_day=selected_time,
        bracket=bracket,
        stream_path=stream_path,
        surface_data_path=surface_path,
        restart_path=restart,
    )


__all__ = [
    "UrbanTimeVaryingNativeData",
    "load_urban_time_varying_native_data",
]
