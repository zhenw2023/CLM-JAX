
from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.netcdf import load_native_variables

_MONTH_LENGTHS_NOLEAP = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_SECONDS_PER_DAY = 86_400


class RestartClock(NamedTuple):
    step_seconds: int
    start_date: int
    start_time_of_day: int
    reference_date: int
    reference_time_of_day: int
    current_date: int
    current_time_of_day: int


class GridTopology(NamedTuple):
    longitude_degrees: jnp.ndarray
    latitude_degrees: jnp.ndarray
    longitude_index: jnp.ndarray
    latitude_index: jnp.ndarray


class LandunitTopology(NamedTuple):
    gridcell: jnp.ndarray
    landunit_type: jnp.ndarray
    active: jnp.ndarray
    weight_on_gridcell: jnp.ndarray


class ColumnTopology(NamedTuple):
    gridcell: jnp.ndarray
    landunit: jnp.ndarray
    landunit_type: jnp.ndarray
    column_type: jnp.ndarray
    active: jnp.ndarray
    weight_on_gridcell: jnp.ndarray
    weight_on_landunit: jnp.ndarray
    elevation: jnp.ndarray


class PatchTopology(NamedTuple):
    gridcell: jnp.ndarray
    landunit: jnp.ndarray
    column: jnp.ndarray
    vegetation_type: jnp.ndarray
    column_type: jnp.ndarray
    landunit_type: jnp.ndarray
    active: jnp.ndarray
    weight_on_gridcell: jnp.ndarray
    weight_on_landunit: jnp.ndarray
    weight_on_column: jnp.ndarray


class SubgridTopology(NamedTuple):
    grid: GridTopology
    landunit: LandunitTopology
    column: ColumnTopology
    patch: PatchTopology


LANDUNIT_SOIL = 1
LANDUNIT_CROP = 2
LANDUNIT_GLACIER = 4
LANDUNIT_LAKE = 5
LANDUNIT_WETLAND = 6
LANDUNIT_URBAN_MIN = 7
LANDUNIT_URBAN_MAX = 9

COLUMN_URBAN_ROOF = 71
COLUMN_URBAN_SUN_WALL = 72
COLUMN_URBAN_SHADE_WALL = 73
COLUMN_URBAN_IMPERVIOUS_ROAD = 74
COLUMN_URBAN_PERVIOUS_ROAD = 75

_TOPOLOGY_VARIABLES = (
    "grid1d_lon",
    "grid1d_lat",
    "grid1d_ixy",
    "grid1d_jxy",
    "land1d_gridcell_index",
    "land1d_ityplun",
    "land1d_active",
    "land1d_wtxy",
    "cols1d_gridcell_index",
    "cols1d_landunit_index",
    "cols1d_ityplun",
    "cols1d_ityp",
    "cols1d_active",
    "cols1d_wtxy",
    "cols1d_wtlnd",
    "cols1d_topoglc",
    "pfts1d_gridcell_index",
    "pfts1d_landunit_index",
    "pfts1d_column_index",
    "pfts1d_itypveg",
    "pfts1d_itypcol",
    "pfts1d_ityplun",
    "pfts1d_active",
    "pfts1d_wtxy",
    "pfts1d_wtlnd",
    "pfts1d_wtcol",
)


def configured_land_restart(case: CimeCase) -> Path:

    entries = tuple(case.input_file_entries())
    for key in ("nrevsn", "finidat"):
        for entry in entries:
            if entry.component == "clm" and entry.key == key and entry.path.exists():
                return entry.path.resolve()

    rpointer = case.path_value("RUNDIR") / "rpointer.lnd"
    if rpointer.is_file():
        for line in rpointer.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name:
                path = Path(name)
                if not path.is_absolute():
                    path = rpointer.parent / path
                if path.exists():
                    return path.resolve()
                break
    raise FileNotFoundError(f"no readable CLM restart is configured by {case.root}")


@cache
def load_restart_clock(path: str | Path) -> RestartClock:
    names = (
        "timemgr_rst_step_sec",
        "timemgr_rst_start_ymd",
        "timemgr_rst_start_tod",
        "timemgr_rst_ref_ymd",
        "timemgr_rst_ref_tod",
        "timemgr_rst_curr_ymd",
        "timemgr_rst_curr_tod",
    )
    values = load_native_variables(path, names)
    integers = [int(np.asarray(values[name]).item()) for name in names]
    return RestartClock(*integers)


def _calendar_day_number(ymd: int, calendar_name: str) -> int:
    year = ymd // 10_000
    month = (ymd // 100) % 100
    day = ymd % 100
    normalized = calendar_name.strip().upper()
    if normalized == "NO_LEAP":
        month_lengths = _MONTH_LENGTHS_NOLEAP
        days_before_year = year * 365
    elif normalized == "360_DAY":
        month_lengths = (30,) * 12
        days_before_year = year * 360
    elif normalized == "GREGORIAN":
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        month_lengths = (31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
        prior_year = year - 1
        days_before_year = (
            prior_year * 365
            + prior_year // 4
            - prior_year // 100
            + prior_year // 400
        )
    else:
        raise ValueError(f"unsupported CLM calendar {normalized!r}")
    if (
        year < 1
        or month < 1
        or month > 12
        or day < 1
        or day > month_lengths[month - 1]
    ):
        raise ValueError(f"invalid {normalized} date {ymd}")
    return days_before_year + sum(month_lengths[: month - 1]) + day - 1


def model_step_from_restart_clock(
    clock: RestartClock,
    calendar_name: str,
) -> int:

    if clock.step_seconds <= 0:
        raise ValueError("restart time step must be positive")
    for name, value in (
        ("start_time_of_day", clock.start_time_of_day),
        ("current_time_of_day", clock.current_time_of_day),
    ):
        if value < 0 or value >= _SECONDS_PER_DAY:
            raise ValueError(f"restart {name} must lie in [0, 86400)")
    elapsed_seconds = (
        _calendar_day_number(clock.current_date, calendar_name)
        - _calendar_day_number(clock.start_date, calendar_name)
    ) * _SECONDS_PER_DAY + (
        clock.current_time_of_day - clock.start_time_of_day
    )
    if elapsed_seconds < 0 or elapsed_seconds % clock.step_seconds != 0:
        raise ValueError("restart clock does not lie on its declared model time step")
    return elapsed_seconds // clock.step_seconds


def advance_restart_clock(
    clock: RestartClock,
    calendar_name: str,
    *,
    steps: int = 1,
) -> RestartClock:

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("model-clock steps must be a nonnegative integer")
    if clock.step_seconds <= 0:
        raise ValueError("restart time step must be positive")
    if clock.current_time_of_day < 0 or clock.current_time_of_day >= _SECONDS_PER_DAY:
        raise ValueError("restart current_time_of_day must lie in [0, 86400)")

    normalized = calendar_name.strip().upper()

    def month_lengths(year: int) -> tuple[int, ...]:
        if normalized == "NO_LEAP":
            return _MONTH_LENGTHS_NOLEAP
        if normalized == "360_DAY":
            return (30,) * 12
        if normalized == "GREGORIAN":
            leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
            return (31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
        raise ValueError(f"unsupported CLM calendar {normalized!r}")

    year = clock.current_date // 10_000
    month = (clock.current_date // 100) % 100
    day = clock.current_date % 100
    lengths = month_lengths(year)
    if year < 1 or month < 1 or month > 12 or day < 1 or day > lengths[month - 1]:
        raise ValueError(f"invalid {normalized} date {clock.current_date}")

    total_time = clock.current_time_of_day + steps * clock.step_seconds
    days_to_advance, time_of_day = divmod(total_time, _SECONDS_PER_DAY)
    while days_to_advance:
        remaining_in_month = lengths[month - 1] - day
        if days_to_advance <= remaining_in_month:
            day += days_to_advance
            days_to_advance = 0
        else:
            days_to_advance -= remaining_in_month + 1
            day = 1
            month += 1
            if month > 12:
                year += 1
                month = 1
                lengths = month_lengths(year)

    return clock._replace(
        current_date=year * 10_000 + month * 100 + day,
        current_time_of_day=time_of_day,
    )


@cache
def load_subgrid_topology(path: str | Path) -> SubgridTopology:

    values = load_native_variables(path, _TOPOLOGY_VARIABLES)

    def index(name: str) -> jnp.ndarray:
        return jnp.asarray(values[name], dtype=jnp.int32) - jnp.int32(1)

    def integer(name: str) -> jnp.ndarray:
        return jnp.asarray(values[name], dtype=jnp.int32)

    def boolean(name: str) -> jnp.ndarray:
        return jnp.asarray(values[name] == 1, dtype=jnp.bool_)

    def real(name: str) -> jnp.ndarray:
        return jnp.asarray(values[name], dtype=jnp.float64)

    return SubgridTopology(
        grid=GridTopology(
            longitude_degrees=real("grid1d_lon"),
            latitude_degrees=real("grid1d_lat"),
            longitude_index=index("grid1d_ixy"),
            latitude_index=index("grid1d_jxy"),
        ),
        landunit=LandunitTopology(
            gridcell=index("land1d_gridcell_index"),
            landunit_type=integer("land1d_ityplun"),
            active=boolean("land1d_active"),
            weight_on_gridcell=real("land1d_wtxy"),
        ),
        column=ColumnTopology(
            gridcell=index("cols1d_gridcell_index"),
            landunit=index("cols1d_landunit_index"),
            landunit_type=integer("cols1d_ityplun"),
            column_type=integer("cols1d_ityp"),
            active=boolean("cols1d_active"),
            weight_on_gridcell=real("cols1d_wtxy"),
            weight_on_landunit=real("cols1d_wtlnd"),
            elevation=real("cols1d_topoglc"),
        ),
        patch=PatchTopology(
            gridcell=index("pfts1d_gridcell_index"),
            landunit=index("pfts1d_landunit_index"),
            column=index("pfts1d_column_index"),
            vegetation_type=integer("pfts1d_itypveg"),
            column_type=integer("pfts1d_itypcol"),
            landunit_type=integer("pfts1d_ityplun"),
            active=boolean("pfts1d_active"),
            weight_on_gridcell=real("pfts1d_wtxy"),
            weight_on_landunit=real("pfts1d_wtlnd"),
            weight_on_column=real("pfts1d_wtcol"),
        ),
    )
