
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from clm_jax.calendar import noleap_seconds_between
from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.canopy_temperature_restart import soil_layer_geometry_20sl_8p5m
from clm_jax.physics.history_output import H1_HISTORY_VARIABLES, H2_HISTORY_VARIABLES

_TIME_ORIGIN = datetime(1979, 1, 1)
_SECONDS_PER_DAY = np.float64(86_400.0)
_FILL_VALUE = np.float64(CLM_SPECIAL_VALUE)

_FIELD_UNITS = {
    **dict.fromkeys(
        ("AR", "GPP", "GRAINC_TO_FOOD", "GRAINC_TO_SEED", "HR", "NBP", "NEE", "NEP", "NPP"),
        "gC/m^2/s",
    ),
    **dict.fromkeys(
        ("CPOOL", "CROPPROD1C", "GRAINC", "TOTECOSYSC", "TOTLITC", "TOTSOMC", "TOTVEGC"),
        "gC/m^2",
    ),
    **dict.fromkeys(("GRAINN", "NPOOL", "SMINN"), "gN/m^2"),
    "PLANT_NDEMAND": "gN/m^2/s",
    **dict.fromkeys(
        ("EFLX_LH_TOT", "ERRSEB", "ERRSOI", "FGR", "FIRA", "FLDS", "FSA", "FSDS", "FSH"),
        "W/m^2",
    ),
    **dict.fromkeys(("ERRH2O", "ERRH2OSNO", "H2OCAN", "H2OSFC", "H2OSNO", "TWS"), "mm"),
    **dict.fromkeys(
        (
            "QDRAI",
            "QFLX_EVAP_TOT",
            "QIRRIG",
            "QOVER",
            "QRUNOFF",
            "QSOIL",
            "QVEGE",
            "QVEGT",
            "RAIN",
            "RAIN_FROM_ATM",
            "SNOW",
            "SNOW_FROM_ATM",
        ),
        "mm/s",
    ),
    **dict.fromkeys(("SOILICE", "SOILLIQ"), "kg/m2"),
    "H2OSOI": "mm3/mm3",
    **dict.fromkeys(("TBOT", "TG", "TSA", "TSOI"), "K"),
    "PBOT": "Pa",
    "QBOT": "kg/kg",
    "WIND": "m/s",
    "ZWT": "m",
    "TLAI": "m^2/m^2",
    "FPSN": "umol/m^2/s",
    "BTRAN2": "unitless",
}

_LEVEL_DIMENSION = {
    "H2OSOI": "levsoi",
    "SOILICE": "levsoi",
    "SOILLIQ": "levsoi",
    "TSOI": "levgrnd",
}


class HistoryWriteRecord(NamedTuple):

    h1_path: Path
    h2_path: Path
    index: int


def _dataset_type():
    from netCDF4 import Dataset  

    return Dataset


def _open_history_append(path: Path, *, timeout_seconds: float = 300.0):
    if not path.is_file():
        raise FileNotFoundError(path)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            return _dataset_type()(path, mode="r+")
        except OSError as error:
            remaining = deadline - time.monotonic()
            if error.errno != -101 or remaining <= 0 or not path.is_file():
                raise
            time.sleep(min(0.1, remaining))


def _configured_domain_path(case: Any) -> Path:
    parent = Path(case.expand(case.value("LND_DOMAIN_PATH"))).expanduser()
    if not parent.is_absolute():
        parent = case.root / parent
    path = (parent / case.expand(case.value("LND_DOMAIN_FILE"))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured land domain does not exist: {path}")
    return path


def _clock_datetime(clock: Any) -> datetime:
    hour, remainder = divmod(int(np.asarray(clock.seconds_into_day)), 3600)
    minute, second = divmod(remainder, 60)
    return datetime(
        int(np.asarray(clock.year)),
        int(np.asarray(clock.month)),
        int(np.asarray(clock.day)),
        hour=hour,
        minute=minute,
        second=second,
    )


class ClmHistoryWriter:

    def __init__(
        self,
        case: Any,
        output_directory: str | Path,
        topology: Any,
        *,
        overwrite: bool,
        records_per_file: int | None = None,
    ) -> None:
        self._case_name = case.root.name
        self._directory = Path(output_directory).expanduser().resolve()
        self._directory.mkdir(parents=True, exist_ok=True)
        self._overwrite = bool(overwrite)
        self._record_count = 0
        if records_per_file is not None and records_per_file < 1:
            raise ValueError("records_per_file must be positive")
        self._records_per_file = records_per_file
        self._h1_path: Path | None = None
        self._h2_path: Path | None = None

        grid = topology.grid
        self._latitude_index = np.asarray(grid.latitude_index, dtype=np.int64)
        self._longitude_index = np.asarray(grid.longitude_index, dtype=np.int64)
        self._gridcell_count = int(self._latitude_index.size)
        self._landunit_count = int(topology.landunit.gridcell.shape[0])
        self._column_count = int(topology.column.gridcell.shape[0])
        self._patch_count = int(topology.patch.gridcell.shape[0])

        Dataset = _dataset_type()
        with Dataset(_configured_domain_path(case), mode="r") as domain:
            longitude = np.asarray(domain.variables["xc"][:], dtype=np.float64)
            latitude = np.asarray(domain.variables["yc"][:], dtype=np.float64)
        if longitude.ndim != 2 or latitude.shape != longitude.shape:
            raise ValueError("land-domain xc/yc must be matching two-dimensional arrays")
        self._latitude = np.ascontiguousarray(latitude[:, 0])
        self._longitude = np.ascontiguousarray(longitude[0, :])
        self._latitude_count, self._longitude_count = longitude.shape
        if (
            np.any(self._latitude_index < 0)
            or np.any(self._latitude_index >= self._latitude_count)
            or np.any(self._longitude_index < 0)
            or np.any(self._longitude_index >= self._longitude_count)
        ):
            raise ValueError("native grid indices lie outside the configured land domain")
        flat_indices = self._latitude_index * self._longitude_count + self._longitude_index
        if np.unique(flat_indices).size != self._gridcell_count:
            raise ValueError("native grid topology contains duplicate rectangular cells")

    @property
    def paths(self) -> tuple[Path, Path] | None:
        if self._h1_path is None or self._h2_path is None:
            return None
        return self._h1_path, self._h2_path

    @property
    def all_paths(self) -> tuple[Path, ...]:
        if self._records_per_file is not None:
            return tuple(sorted(self._directory.glob(f"{self._case_name}.clm2.h[12].*.nc")))
        return () if self.paths is None else self.paths

    def _initialize_paths(self, clock: Any) -> None:
        current = _clock_datetime(clock)
        seconds = current.hour * 3600 + current.minute * 60 + current.second
        suffix = current.strftime("%Y-%m-%d-") + f"{seconds:05d}"
        self._h1_path = self._directory / f"{self._case_name}.clm2.h1.{suffix}.nc"
        self._h2_path = self._directory / f"{self._case_name}.clm2.h2.{suffix}.nc"
        for path, tape in ((self._h1_path, 1), (self._h2_path, 2)):
            if path.exists():
                if not self._overwrite:
                    raise FileExistsError(f"history output already exists: {path}")
                path.unlink()
            self._create_file(path, tape=tape)

    def _create_file(self, path: Path, *, tape: int) -> None:
        Dataset = _dataset_type()
        node_depth, _, _ = soil_layer_geometry_20sl_8p5m()
        with Dataset(path, mode="w", format="NETCDF4") as dataset:
            for name, size in (
                ("lon", self._longitude_count),
                ("lat", self._latitude_count),
                ("gridcell", self._gridcell_count),
                ("landunit", self._landunit_count),
                ("column", self._column_count),
                ("pft", self._patch_count),
                ("levgrnd", 25),
                ("levsoi", 20),
                ("hist_interval", 2),
            ):
                dataset.createDimension(name, size)
            dataset.createDimension("time", None)

            latitude = dataset.createVariable("lat", "f8", ("lat",), fill_value=_FILL_VALUE)
            latitude.long_name = "coordinate latitude"
            latitude.units = "degrees_north"
            latitude.missing_value = _FILL_VALUE
            latitude[:] = self._latitude
            longitude = dataset.createVariable("lon", "f8", ("lon",), fill_value=_FILL_VALUE)
            longitude.long_name = "coordinate longitude"
            longitude.units = "degrees_east"
            longitude.missing_value = _FILL_VALUE
            longitude[:] = self._longitude
            for name, values in (("levgrnd", node_depth), ("levsoi", node_depth[:20])):
                coordinate = dataset.createVariable(name, "f8", (name,))
                coordinate.long_name = "coordinate soil levels"
                coordinate.units = "m"
                coordinate[:] = values

            time = dataset.createVariable("time", "f8", ("time",))
            time.long_name = "time"
            time.units = "days since 1979-01-01 00:00:00"
            time.calendar = "noleap"
            time.bounds = "time_bounds"
            bounds = dataset.createVariable("time_bounds", "f8", ("time", "hist_interval"))
            bounds.long_name = "history time interval endpoints"
            for name, long_name, units in (
                ("mcdate", "current date (YYYYMMDD)", None),
                ("mcsec", "current seconds of current date", "s"),
                ("mdcur", "current day (from base day)", None),
                ("mscur", "current seconds of current day", None),
                ("nstep", "time step", None),
            ):
                variable = dataset.createVariable(name, "i4", ("time",))
                variable.long_name = long_name
                if units is not None:
                    variable.units = units

            fields = H1_HISTORY_VARIABLES if tape == 1 else H2_HISTORY_VARIABLES
            for name in fields:
                if tape == 1:
                    level = _LEVEL_DIMENSION.get(name)
                    dimensions = (
                        ("time", "lat", "lon") if level is None else ("time", level, "lat", "lon")
                    )
                else:
                    dimensions = ("time", "pft")
                variable = dataset.createVariable(
                    name,
                    "f8",
                    dimensions,
                    fill_value=_FILL_VALUE,
                )
                variable.long_name = (
                    "volumetric soil water (vegetated landunits only)" if name == "H2OSOI" else name
                )
                variable.units = _FIELD_UNITS[name]
                variable.cell_methods = "time: instantaneous"
                if tape == 2 and name == "SMINN":
                    variable.source_level = "column"
                    variable.comment = (
                        "Live soil mineral nitrogen in each PFT's parent column. "
                        "The original CLM forced-PFT SMINN history buffer is not "
                        "a native PFT observable; retain this distinction in validation."
                    )

    def _scatter_gridcell(self, name: str, values: np.ndarray) -> np.ndarray:
        expected_levels = 20 if _LEVEL_DIMENSION.get(name) == "levsoi" else 25
        if name not in _LEVEL_DIMENSION:
            if values.shape != (self._gridcell_count,):
                raise ValueError(
                    f"h1 {name} must have shape {(self._gridcell_count,)}, got {values.shape}"
                )
            result = np.full(
                (self._latitude_count, self._longitude_count),
                _FILL_VALUE,
                dtype=np.float64,
            )
            result[self._latitude_index, self._longitude_index] = values
            return result
        if values.shape != (self._gridcell_count, expected_levels):
            raise ValueError(
                f"h1 {name} must have shape "
                f"{(self._gridcell_count, expected_levels)}, got {values.shape}"
            )
        result = np.full(
            (expected_levels, self._latitude_count, self._longitude_count),
            _FILL_VALUE,
            dtype=np.float64,
        )
        result[:, self._latitude_index, self._longitude_index] = values.T
        return result

    def _write_time(self, dataset: Any, index: int, clock: Any) -> None:
        current = _clock_datetime(clock)
        elapsed_seconds = noleap_seconds_between(_TIME_ORIGIN, current)
        current_days = np.float64(elapsed_seconds) / _SECONDS_PER_DAY
        step_seconds = int(np.asarray(clock.step_seconds))
        dataset.variables["time"][index] = current_days
        dataset.variables["time_bounds"][index, :] = (
            current_days - np.float64(step_seconds) / _SECONDS_PER_DAY,
            current_days,
        )
        seconds = int(np.asarray(clock.seconds_into_day))
        dataset.variables["mcdate"][index] = (
            current.year * 10_000 + current.month * 100 + current.day
        )
        dataset.variables["mcsec"][index] = seconds
        dataset.variables["mdcur"][index] = elapsed_seconds // 86_400
        dataset.variables["mscur"][index] = seconds
        dataset.variables["nstep"][index] = int(np.asarray(clock.model_step))

    def append(self, snapshot: Any, clock: Any) -> HistoryWriteRecord:

        if self._records_per_file is not None and self._record_count == self._records_per_file:
            assert self.paths is not None
            self._h1_path = self._h2_path = None
            self._record_count = 0
        if self._h1_path is None:
            self._initialize_paths(clock)
        assert self._h1_path is not None and self._h2_path is not None
        if tuple(snapshot.gridcell) != H1_HISTORY_VARIABLES:
            raise ValueError("h1 snapshot schema or ordering does not match its contract")
        if tuple(snapshot.pft) != H2_HISTORY_VARIABLES:
            raise ValueError("h2 snapshot schema or ordering does not match its contract")

        index = self._record_count
        with _open_history_append(self._h1_path) as dataset:
            self._write_time(dataset, index, clock)
            for name in H1_HISTORY_VARIABLES:
                values = np.asarray(snapshot.gridcell[name], dtype=np.float64)
                dataset.variables[name][index] = self._scatter_gridcell(name, values)
        with _open_history_append(self._h2_path) as dataset:
            self._write_time(dataset, index, clock)
            for name in H2_HISTORY_VARIABLES:
                values = np.asarray(snapshot.pft[name], dtype=np.float64)
                if values.shape != (self._patch_count,):
                    raise ValueError(
                        f"h2 {name} must have shape {(self._patch_count,)}, got {values.shape}"
                    )
                dataset.variables[name][index] = values
        self._record_count += 1
        if self._records_per_file is not None and self._record_count == self._records_per_file:
            import json  

            marker = self._h1_path.with_suffix(".complete.json")
            temporary = marker.with_suffix(".json.partial")
            temporary.write_text(
                json.dumps(
                    {
                        "h1_path": str(self._h1_path),
                        "h2_path": str(self._h2_path),
                        "records": self._record_count,
                        "last_time": _clock_datetime(clock).isoformat(),
                    }
                )
                + "\n"
            )
            temporary.replace(marker)
        return HistoryWriteRecord(self._h1_path, self._h2_path, index)


__all__ = ["ClmHistoryWriter", "HistoryWriteRecord"]
