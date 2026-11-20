
from __future__ import annotations

import warnings
from functools import cache, lru_cache, wraps
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from clm_jax.forcing.bgc import (
    BgcExternalForcingInputs,
    BgcExternalForcingParameters,
)
from clm_jax.forcing.remap import (
    apply_sparse_mapping,
    build_rectilinear_bilinear_mapping,
)
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.restart import (
    RestartClock,
    SubgridTopology,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)

_MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_SECONDS_PER_DAY = 86_400


def _cached_source_arrays(maxsize):
    def decorate(function):
        @lru_cache(maxsize=maxsize)
        @wraps(function)
        def construct(*arguments,**keywords):
            with jax.core.eval_context():
                result=function(*arguments,**keywords)
            if any(isinstance(value,jax.core.Tracer) for value in jax.tree.leaves(result)):
                raise ValueError('external stream cache cannot contain traced model values')
            return result
        construct._clm_concrete_source_arrays=True
        return construct
    return decorate


class StreamBracket(NamedTuple):

    lower_index: int
    upper_index: int
    lower_weight: np.float64
    lower_date: int
    lower_time_of_day: int
    upper_date: int
    upper_time_of_day: int
    interpolation: str
    time_axis_mode: str


class BgcExternalForcingNativeData(NamedTuple):

    inputs: BgcExternalForcingInputs
    parameters: BgcExternalForcingParameters
    topology: SubgridTopology
    clock: RestartClock
    call_date: int
    call_time_of_day: int
    nitrogen_deposition_bracket: StreamBracket
    population_density_bracket: StreamBracket
    lightning_frequency_bracket: StreamBracket
    nitrogen_deposition_path: Path
    population_density_path: Path
    lightning_frequency_path: Path
    restart_path: Path


def _netcdf_types():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="numpy.ndarray size changed.*",
            category=RuntimeWarning,
        )
        from netCDF4 import Dataset, num2date  

    return Dataset, num2date


def _integer(group: dict[str, object], name: str, default: int) -> int:
    value = group.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"CLM namelist value {name!r} must be integer")
    return value


def _string(group: dict[str, object], name: str, default: str) -> str:
    value = group.get(name, default)
    if not isinstance(value, str):
        raise TypeError(f"CLM namelist value {name!r} must be a string")
    return value.strip().lower()


def _path(group: dict[str, object], name: str, case: CimeCase) -> Path:
    declared = group.get(name)
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(f"CLM namelist lacks required {name}")
    path = Path(case.expand(declared)).expanduser()
    if not path.is_absolute():
        path = case.root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM stream file does not exist: {path}")
    return path


def _split_date(date: int) -> tuple[int, int, int]:
    return date // 10_000, (date // 100) % 100, date % 100


def _date(year: int, month: int, day: int) -> int:
    return year * 10_000 + month * 100 + day


def _noleap_ordinal(date: int) -> int:
    year, month, day = _split_date(date)
    if year < 1 or not 1 <= month <= 12 or not 1 <= day <= _MONTH_LENGTHS[month - 1]:
        raise ValueError(f"invalid NO_LEAP date: {date}")
    return (year - 1) * 365 + sum(_MONTH_LENGTHS[: month - 1]) + day - 1


def _advance_noleap(date: int, time_of_day: int, seconds: int) -> tuple[int, int]:
    total = _noleap_ordinal(date) * _SECONDS_PER_DAY + time_of_day + seconds
    ordinal, result_time = divmod(total, _SECONDS_PER_DAY)
    year, day_of_year = divmod(ordinal, 365)
    year += 1
    month = 1
    while day_of_year >= _MONTH_LENGTHS[month - 1]:
        day_of_year -= _MONTH_LENGTHS[month - 1]
        month += 1
    return _date(year, month, day_of_year + 1), result_time


def _replace_year(date: int, year: int) -> int:
    _, month, day = _split_date(date)
    return _date(year, month, day)


@cache
def _time_metadata(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Dataset, num2date = _netcdf_types()
    with Dataset(path, mode="r") as dataset:
        time = dataset.variables["time"]
        units = time.getncattr("units")
        calendar = time.getncattr("calendar") if "calendar" in time.ncattrs() else "noleap"
        if str(calendar).strip().lower() not in {"noleap", "no_leap", "365_day"}:
            raise NotImplementedError("BGC stream adapter currently requires NO_LEAP inputdata")
        decoded = num2date(
            np.asarray(time[:], dtype=np.float64),
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=True,
        )
    dates = np.fromiter(
        (_date(value.year, value.month, value.day) for value in decoded),
        dtype=np.int64,
        count=len(decoded),
    )
    seconds = np.fromiter(
        (
            int(
                np.floor(
                    value.hour * 3600
                    + value.minute * 60
                    + value.second
                    + value.microsecond / 1_000_000.0
                    + 0.5
                )
            )
            for value in decoded
        ),
        dtype=np.int64,
        count=len(decoded),
    )
    ordinals = np.asarray(
        [_noleap_ordinal(int(date)) * _SECONDS_PER_DAY + int(time)
         for date, time in zip(dates, seconds, strict=True)],
        dtype=np.int64,
    )
    return dates, seconds, ordinals


def _interpolation_weight(
    lower_seconds: int,
    upper_seconds: int,
    target_seconds: int,
    algorithm: str,
) -> np.float64:
    if algorithm == "lower":
        return np.float64(1.0 if lower_seconds < upper_seconds else 0.0)
    if algorithm == "upper":
        return np.float64(0.0 if lower_seconds < upper_seconds else 1.0)
    if algorithm == "nearest":
        return np.float64(
            1.0
            if abs(lower_seconds - target_seconds) <= abs(upper_seconds - target_seconds)
            else 0.0
        )
    if algorithm == "linear":
        if lower_seconds == upper_seconds:
            return np.float64(0.5)
        return np.float64(upper_seconds - target_seconds) / np.float64(
            upper_seconds - lower_seconds
        )
    raise ValueError(f"unsupported CLM stream interpolation algorithm: {algorithm!r}")


def _stream_bracket(
    path: Path,
    *,
    call_date: int,
    call_time_of_day: int,
    year_first: int,
    year_last: int,
    year_align: int,
    time_axis_mode: str,
    interpolation: str,
) -> StreamBracket:
    dates, times, ordinals = _time_metadata(path)
    years = dates // 10_000
    selected = np.flatnonzero((years >= year_first) & (years <= year_last))
    if selected.size == 0:
        raise ValueError(f"stream {path} has no records in {year_first}-{year_last}")
    selected_dates = dates[selected]
    selected_times = times[selected]
    source_seconds = ordinals[selected]
    if np.any(np.diff(source_seconds) <= 0):
        raise ValueError(f"stream time coordinate is not strictly increasing: {path}")

    model_year = call_date // 10_000
    year_count = year_last - year_first + 1
    if year_count < 1:
        raise ValueError("stream_year_last must not precede stream_year_first")
    mode = time_axis_mode.strip().lower()
    if mode == "cycle":
        data_year = year_first + ((model_year - year_align) % year_count)
    elif mode in {"extend", "limit"}:
        data_year = year_first + model_year - year_align
    else:
        raise ValueError(f"unsupported CLM stream time-axis mode: {time_axis_mode!r}")
    data_date = _replace_year(call_date, data_year)
    target = _noleap_ordinal(data_date) * _SECONDS_PER_DAY + call_time_of_day
    model_target = _noleap_ordinal(call_date) * _SECONDS_PER_DAY + call_time_of_day
    base_year_shift = model_year - data_year

    if target < source_seconds[0]:
        if mode == "limit":
            raise ValueError("model date precedes a taxmode='limit' stream")
        if mode == "extend":
            lower_position = upper_position = 0
            lower_date = 101
            lower_time_of_day = 0
            upper_date = _replace_year(
                int(selected_dates[0]),
                int(years[selected[0]] + base_year_shift),
            )
            upper_time_of_day = int(selected_times[0])
        else:
            lower_position = selected.size - 1
            upper_position = 0
            lower_date = _replace_year(
                int(selected_dates[-1]),
                int(years[selected[-1]] + base_year_shift - year_count),
            )
            lower_time_of_day = int(selected_times[-1])
            upper_date = _replace_year(
                int(selected_dates[0]),
                int(years[selected[0]] + base_year_shift),
            )
            upper_time_of_day = int(selected_times[0])
    elif target >= source_seconds[-1]:
        if mode == "limit":
            raise ValueError("model date follows a taxmode='limit' stream")
        if mode == "extend":
            lower_position = upper_position = selected.size - 1
            lower_date = _replace_year(
                int(selected_dates[-1]),
                int(years[selected[-1]] + base_year_shift),
            )
            lower_time_of_day = int(selected_times[-1])
            upper_date = 99991231
            upper_time_of_day = 0
        else:
            lower_position = selected.size - 1
            upper_position = 0
            lower_date = _replace_year(
                int(selected_dates[-1]),
                int(years[selected[-1]] + base_year_shift),
            )
            lower_time_of_day = int(selected_times[-1])
            upper_date = _replace_year(
                int(selected_dates[0]),
                int(years[selected[0]] + base_year_shift + year_count),
            )
            upper_time_of_day = int(selected_times[0])
    else:
        upper_position = int(np.searchsorted(source_seconds, target, side="right"))
        lower_position = upper_position - 1
        lower_date = _replace_year(
            int(selected_dates[lower_position]),
            int(years[selected[lower_position]] + base_year_shift),
        )
        lower_time_of_day = int(selected_times[lower_position])
        upper_date = _replace_year(
            int(selected_dates[upper_position]),
            int(years[selected[upper_position]] + base_year_shift),
        )
        upper_time_of_day = int(selected_times[upper_position])

    lower_time = _noleap_ordinal(lower_date) * _SECONDS_PER_DAY + lower_time_of_day
    upper_time = _noleap_ordinal(upper_date) * _SECONDS_PER_DAY + upper_time_of_day
    weight = _interpolation_weight(lower_time, upper_time, model_target, interpolation)
    lower_index = int(selected[lower_position])
    upper_index = int(selected[upper_position])
    return StreamBracket(
        lower_index=lower_index,
        upper_index=upper_index,
        lower_weight=weight,
        lower_date=lower_date,
        lower_time_of_day=lower_time_of_day,
        upper_date=upper_date,
        upper_time_of_day=upper_time_of_day,
        interpolation=interpolation,
        time_axis_mode=mode,
    )


@lru_cache(maxsize=16)
def _read_records(
    path: Path,
    variable_name: str,
    lower_index: int,
    upper_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    Dataset, _ = _netcdf_types()
    with Dataset(path, mode="r") as dataset:
        variable = dataset.variables[variable_name]
        lower = np.asarray(variable[lower_index, ...], dtype=np.float64)
        upper = (
            lower.copy()
            if upper_index == lower_index
            else np.asarray(variable[upper_index, ...], dtype=np.float64)
        )
        latitude = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
        longitude = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
        units = str(variable.getncattr("units")).strip() if "units" in variable.ncattrs() else ""
    if lower.shape != (latitude.size, longitude.size) or upper.shape != lower.shape:
        raise ValueError(f"stream variable {variable_name!r} must have (time,lat,lon) axes")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError(f"stream variable {variable_name!r} contains non-finite values")
    return lower, upper, longitude, latitude, units


@_cached_source_arrays(maxsize=8)
def _mapped_grid_geometry(
    source_longitude_bytes: bytes,
    source_latitude_bytes: bytes,
    destination_longitude_bytes: bytes,
    destination_latitude_bytes: bytes,
    backend: str,
):
    source_longitude = np.frombuffer(source_longitude_bytes, dtype=np.float64)
    source_latitude = np.frombuffer(source_latitude_bytes, dtype=np.float64)
    destination_longitude = np.frombuffer(destination_longitude_bytes, dtype=np.float64)
    destination_latitude = np.frombuffer(destination_latitude_bytes, dtype=np.float64)
    longitude = np.where(source_longitude > 180.0, source_longitude - 360.0, source_longitude)
    longitude_order = np.argsort(longitude)
    latitude_order = np.argsort(source_latitude)
    longitude = longitude[longitude_order]
    latitude = source_latitude[latitude_order]
    destination_longitude = np.where(
        destination_longitude > 180.0,
        destination_longitude - 360.0,
        destination_longitude,
    )
    mapping = build_rectilinear_bilinear_mapping(
        longitude,
        latitude,
        destination_longitude,
        destination_latitude,
    )
    if backend == "gpu":
        from clm_jax.forcing.ordered_mapping import prepare_ordered_mapping

        mapping = prepare_ordered_mapping(mapping)
    return latitude_order, longitude_order, mapping


def _mapped_records(
    lower: np.ndarray,
    upper: np.ndarray,
    source_longitude: np.ndarray,
    source_latitude: np.ndarray,
    topology: SubgridTopology,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    coordinates = tuple(np.asarray(value, dtype=np.float64) for value in (
        source_longitude, source_latitude, topology.grid.longitude_degrees,
        topology.grid.latitude_degrees,
    ))
    if any(value.ndim != 1 for value in coordinates):
        raise ValueError("BGC mapping coordinates must be vectors")
    latitude_order, longitude_order, mapping = _mapped_grid_geometry(
        *(value.tobytes() for value in coordinates), jax.default_backend(),
    )
    lower = lower[np.ix_(latitude_order, longitude_order)]
    upper = upper[np.ix_(latitude_order, longitude_order)]
    records = jnp.asarray(np.stack((lower.reshape(-1), upper.reshape(-1))))
    mapped = apply_sparse_mapping(mapping, records)
    return mapped[0], mapped[1]


def _native_grid_records(
    lower: np.ndarray,
    upper: np.ndarray,
    source_longitude: np.ndarray,
    source_latitude: np.ndarray,
    topology: SubgridTopology,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    longitude_index = np.asarray(topology.grid.longitude_index, dtype=np.int64)
    latitude_index = np.asarray(topology.grid.latitude_index, dtype=np.int64)
    expected_longitude = source_longitude[longitude_index]
    actual_longitude = np.asarray(topology.grid.longitude_degrees, dtype=np.float64)
    expected_latitude = source_latitude[latitude_index]
    actual_latitude = np.asarray(topology.grid.latitude_degrees, dtype=np.float64)
    if not np.allclose(expected_longitude, actual_longitude, rtol=0.0, atol=1.0e-12):
        raise ValueError("nitrogen-deposition longitude grid differs from the CLM grid")
    if not np.allclose(expected_latitude, actual_latitude, rtol=0.0, atol=1.0e-12):
        raise ValueError("nitrogen-deposition latitude grid differs from the CLM grid")
    return (
        jnp.asarray(lower[latitude_index, longitude_index]),
        jnp.asarray(upper[latitude_index, longitude_index]),
    )


@_cached_source_arrays(maxsize=16)
def _stream_records_on_land_grid(
    path: Path,
    variable_name: str,
    lower_index: int,
    upper_index: int,
    restart_path: Path,
    native_grid: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, str]:

    records = _read_records(path, variable_name, lower_index, upper_index)
    topology = load_subgrid_topology(restart_path)
    remap = _native_grid_records if native_grid else _mapped_records
    lower, upper = remap(*records[:4], topology)
    return lower, upper, records[4]


def load_bgc_external_forcing_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    *,
    call_date: int | None = None,
    call_time_of_day: int | None = None,
) -> BgcExternalForcingNativeData:

    general = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    if general.get("use_cn") is not True:
        raise ValueError("BGC external forcing requires use_cn=.true.")
    calendar = case.value("CALENDAR", "NO_LEAP").strip().upper()
    if calendar != "NO_LEAP":
        raise NotImplementedError("native BGC forcing adapter currently requires NO_LEAP")
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

    ndep = dict(read_case_namelist_group(case, "lnd_in", "ndepdyn_nml"))
    population = dict(read_case_namelist_group(case, "lnd_in", "popd_streams"))
    lightning = dict(read_case_namelist_group(case, "lnd_in", "light_streams"))
    ndep_path = _path(ndep, "stream_fldfilename_ndep", case)
    population_path = _path(population, "stream_fldfilename_popdens", case)
    lightning_path = _path(lightning, "stream_fldfilename_lightng", case)

    ndep_bracket = _stream_bracket(
        ndep_path,
        call_date=selected_date,
        call_time_of_day=selected_time,
        year_first=_integer(ndep, "stream_year_first_ndep", 1),
        year_last=_integer(ndep, "stream_year_last_ndep", 1),
        year_align=_integer(ndep, "model_year_align_ndep", 1),
        time_axis_mode=_string(ndep, "ndep_taxmode", "extend"),
        interpolation=_string(ndep, "ndep_tintalgo", "linear"),
    )
    population_bracket = _stream_bracket(
        population_path,
        call_date=selected_date,
        call_time_of_day=selected_time,
        year_first=_integer(population, "stream_year_first_popdens", 1),
        year_last=_integer(population, "stream_year_last_popdens", 1),
        year_align=_integer(population, "model_year_align_popdens", 1),
        time_axis_mode="extend",
        interpolation=_string(population, "popdens_tintalgo", "nearest"),
    )
    lightning_bracket = _stream_bracket(
        lightning_path,
        call_date=selected_date,
        call_time_of_day=selected_time,
        year_first=_integer(lightning, "stream_year_first_lightng", 1),
        year_last=_integer(lightning, "stream_year_last_lightng", 1),
        year_align=_integer(lightning, "model_year_align_lightng", 1),
        time_axis_mode="cycle",
        interpolation=_string(lightning, "lightng_tintalgo", "linear"),
    )

    ndep_variable = ndep.get("ndep_varlist", "NDEP_year")
    if not isinstance(ndep_variable, str) or "," in ndep_variable:
        raise NotImplementedError("native BGC adapter currently requires one NDEP variable")
    ndep_lower, ndep_upper, ndep_units = _stream_records_on_land_grid(
        ndep_path,
        ndep_variable.strip(),
        ndep_bracket.lower_index,
        ndep_bracket.upper_index,
        restart,
        True,
    )
    population_lower, population_upper, _ = _stream_records_on_land_grid(
        population_path,
        "hdm",
        population_bracket.lower_index,
        population_bracket.upper_index,
        restart,
        False,
    )
    lightning_lower, lightning_upper, _ = _stream_records_on_land_grid(
        lightning_path,
        "lnfm",
        lightning_bracket.lower_index,
        lightning_bracket.upper_index,
        restart,
        False,
    )
    if ndep_units == "g(N)/m2/s":
        annual = False
    elif ndep_units == "g(N)/m2/yr":
        annual = True
    else:
        raise ValueError(f"unsupported nitrogen-deposition units: {ndep_units!r}")
    return BgcExternalForcingNativeData(
        inputs=BgcExternalForcingInputs(
            nitrogen_deposition_lower_gridcell=ndep_lower,
            nitrogen_deposition_upper_gridcell=ndep_upper,
            population_density_lower_gridcell=population_lower,
            population_density_upper_gridcell=population_upper,
            lightning_frequency_lower_gridcell=lightning_lower,
            lightning_frequency_upper_gridcell=lightning_upper,
            nitrogen_deposition_lower_weight=ndep_bracket.lower_weight,
            population_density_lower_weight=population_bracket.lower_weight,
            lightning_frequency_lower_weight=lightning_bracket.lower_weight,
        ),
        parameters=BgcExternalForcingParameters(
            nitrogen_deposition_is_annual=annual,
            days_per_year=365.0,
        ),
        topology=topology,
        clock=clock,
        call_date=selected_date,
        call_time_of_day=selected_time,
        nitrogen_deposition_bracket=ndep_bracket,
        population_density_bracket=population_bracket,
        lightning_frequency_bracket=lightning_bracket,
        nitrogen_deposition_path=ndep_path,
        population_density_path=population_path,
        lightning_frequency_path=lightning_path,
        restart_path=restart,
    )


__all__ = [
    "BgcExternalForcingNativeData",
    "StreamBracket",
    "load_bgc_external_forcing_native_data",
]
