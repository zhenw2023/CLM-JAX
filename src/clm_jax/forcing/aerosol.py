
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import cftime
import netCDF4
import numpy as np

from clm_jax.calendar import NOLEAP_MONTH_LENGTHS
from clm_jax.forcing.area import load_land_flux_area_factor
from clm_jax.io.datm_stream import DatmStream
from clm_jax.io.namelist import read_case_namelist_group


SPECIES = (
    "bcphidry", "bcphodry", "bcphiwet", "ocphidry", "ocphodry", "ocphiwet",
    "dstwet1", "dstdry1", "dstwet2", "dstdry2", "dstwet3", "dstdry3",
    "dstwet4", "dstdry4",
)
YEAR_SECONDS = 365 * 86400


def _cycle_index_path(directory, path, time, calendar, year):
    stat = path.stat()
    key = (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
           stat.st_ctime_ns, time.size, time.units, calendar, year)
    return Path(directory) / (hashlib.sha256(repr(key).encode()).hexdigest() + ".npz")


def _write_cycle_index(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez(stream, records=records, source_key=path.stem)
            stream.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _cycle_records(path, time, calendar, year, directory):
    if directory is not None:
        index = _cycle_index_path(directory, path, time, calendar, year)
        try:
            with np.load(index, allow_pickle=False) as saved:
                records = saved["records"]
                same_source = saved["source_key"].item() == index.stem
            if (same_source and records.dtype.kind == "i" and records.ndim == 1
                    and len(records) >= 2 and 0 <= records[0] and records[-1] < time.size
                    and np.all(records[1:] > records[:-1])):
                return records
        except (OSError, ValueError, TypeError, EOFError, KeyError, BadZipFile):
            pass
    dates = cftime.num2date(time[:], time.units, calendar=calendar)
    records = np.array([i for i, date in enumerate(dates) if date.year == year], dtype=np.int64)
    if directory is not None and len(records) >= 2:
        try:
            _write_cycle_index(index, records)
        except OSError:
            pass
    return records


@dataclass
class AerosolDepositionForcing:

    path: Path
    variables: tuple[str, ...]
    records: np.ndarray
    seconds: np.ndarray
    latitude_indices: np.ndarray
    longitude_indices: np.ndarray
    cache: OrderedDict = field(default_factory=OrderedDict)
    land_flux_area_factor: np.ndarray | None = None

    @classmethod
    def from_case(cls, case, topology, *, time_index_directory=None):
        if not read_case_namelist_group(case, "datm_in", "datm_nml").get("presaero", False):
            raise ValueError("configured CLM trajectory requires prescribed aerosol deposition")
        group = read_case_namelist_group(case, "datm_in", "shr_strdata_nml")

        def values(name):
            value = group[name]
            return value if isinstance(value, tuple) else (value,)

        matches = []
        for index, declaration in enumerate(values("streams")):
            filename, _align, first, last = declaration.split()
            stream = DatmStream.load(case.case_docs / filename)
            mapping = {v.model_variable: v.file_variable for v in stream.field_variables}
            if set(SPECIES) <= mapping.keys():
                matches.append((index, int(first), int(last), stream, mapping))
        if len(matches) != 1:
            raise ValueError("expected one DATM stream containing all 14 aerosol species")
        index, first, last, stream, mapping = matches[0]
        if (first != last or values("taxmode")[index] != "cycle"
                or values("tintalgo")[index] != "linear" or stream.offset != 0
                or len(stream.field_files) != 1):
            raise ValueError("aerosol forcing requires the configured cyclic single-year linear stream")
        path = stream.field_files[0]
        iy = np.asarray(topology.grid.latitude_index, dtype=np.int64)
        ix = np.asarray(topology.grid.longitude_index, dtype=np.int64)
        with netCDF4.Dataset(path) as data:
            time = data["time"]
            calendar = getattr(time, "calendar", "standard")
            if calendar not in {"noleap", "365_day"}:
                raise ValueError("aerosol stream requires a noleap calendar")
            records = _cycle_records(path, time, calendar, first, time_index_directory)
            epoch = cftime.date2num(cftime.DatetimeNoLeap(first, 1, 1), time.units,
                                   calendar=calendar)
            end = cftime.date2num(cftime.DatetimeNoLeap(first + 1, 1, 1), time.units,
                                 calendar=calendar)
            seconds = np.rint((np.asarray(time[records]) - epoch) * YEAR_SECONDS / (end - epoch))
            if (len(records) < 2 or np.any(np.diff(seconds) <= 0)
                    or seconds[0] < 0 or seconds[-1] >= YEAR_SECONDS):
                raise ValueError("invalid cyclic aerosol time coordinate")
            for name, indices, target in (
                ("lat", iy, topology.grid.latitude_degrees),
                ("lon", ix, topology.grid.longitude_degrees),
            ):
                coordinate = np.asarray(data[name][:])[indices]
                if not np.allclose(coordinate, np.asarray(target), rtol=0, atol=1e-10):
                    raise ValueError(f"aerosol {name} grid requires an explicit remapping")
            for name in SPECIES:
                variable = data[mapping[name]]
                if variable.dimensions != ("time", "lat", "lon") or variable.units != "kg m-2 s-1":
                    raise ValueError(f"invalid aerosol deposition dimensions or units: {name}")
        return cls(path, tuple(mapping[name] for name in SPECIES), records,
                   seconds, iy, ix, land_flux_area_factor=load_land_flux_area_factor(case, topology))

    def _record(self, position):
        position %= len(self.records)
        if position not in self.cache:
            with netCDF4.Dataset(self.path) as data:
                data.set_auto_mask(False)
                record = np.stack([
                    np.asarray(data[name][int(self.records[position]), :, :], np.float64)[
                        self.latitude_indices, self.longitude_indices]
                    for name in self.variables
                ], axis=-1)
            if not np.all(np.isfinite(record)) or np.any(abs(record) > 1e30):
                raise ValueError("missing/nonfinite deposition on the configured CLM land grid")
            self.cache[position] = record
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(position)
        return self.cache[position]

    def sample(self, seconds):
        seconds = int(seconds) % YEAR_SECONDS
        lower = int(np.searchsorted(self.seconds, seconds, side="right")) - 1
        upper = lower + 1
        low_time = self.seconds[lower] if lower >= 0 else self.seconds[-1] - YEAR_SECONDS
        high_time = (self.seconds[upper] if upper < len(self.seconds)
                     else self.seconds[0] + YEAR_SECONDS)
        weight = np.float64(high_time - seconds) / np.float64(high_time - low_time)
        return self._record(lower) * weight + self._record(upper) * (np.float64(1) - weight)

    def load_chunk(self, descriptors, step_seconds):
        values = []
        for descriptor in descriptors:
            date = descriptor.call_date
            month, day = date // 100 % 100, date % 100
            elapsed = (sum(NOLEAP_MONTH_LENGTHS[:month - 1]) + day - 1) * 86400
            elapsed += int(descriptor.call_time_of_day) - step_seconds
            values.append(self.sample(elapsed))
        import jax
        chunk = np.stack(values)
        if self.land_flux_area_factor is not None:
            chunk = chunk * self.land_flux_area_factor[None, :, None]
        return jax.device_put(chunk)
