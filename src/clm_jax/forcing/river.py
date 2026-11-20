
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import cftime
import netCDF4
import numpy as np


@dataclass(frozen=True)
class RiverVolumeForcing:
    bounds: np.ndarray
    volume: np.ndarray
    time_units: str

    @classmethod
    def load(cls, path: Path, topology, *, time_window=None):
        with netCDF4.Dataset(path) as data:
            data.set_auto_mask(False)
            time = data["time"]
            if getattr(time, "calendar", "") not in {"noleap", "365_day"}:
                raise ValueError("river forcing requires a noleap calendar")
            if not time.units.startswith("seconds since "):
                raise ValueError("river forcing time must use seconds since an epoch")
            bounds = np.asarray(data[time.bounds][:], dtype=np.float64)
            times = np.asarray(time[:], dtype=np.float64)
            field = data["river_volume"]
            if field.dimensions != ("time", "gridcell") or field.units != "m3":
                raise ValueError("river_volume must be (time, gridcell) in m3")
            if field.dtype != np.dtype("float64"):
                raise ValueError("river volume must retain native double precision")
            count, gridcells = field.shape
            if count == 0 or bounds.shape != (count, 2) or not np.all(np.isfinite(bounds)):
                raise ValueError("invalid or missing river coupling records")
            if np.any(bounds[:, 1] <= bounds[:, 0]) or np.any(bounds[1:, 0] != bounds[:-1, 1]):
                raise ValueError("river forcing intervals must be positive, contiguous and ordered")
            if times.shape != (count,) or not np.array_equal(times, bounds[:, 0]):
                raise ValueError("river time must mark the beginning of each exchange interval")
            if gridcells != len(topology.grid.longitude_index):
                raise ValueError("river coupling field does not cover the CLM land grid")
            for name in ("longitude_index", "latitude_index"):
                coordinate = data[name]
                if getattr(coordinate, "index_base", None) != 0:
                    raise ValueError("river grid indices must explicitly be zero based")
                if not np.array_equal(coordinate[:], np.asarray(getattr(topology.grid, name))):
                    raise ValueError(f"river {name} differs from the configured CLM grid")
            units = time.units
            records = slice(None)
            if time_window is not None:
                start, stop = cftime.date2num(time_window, units, calendar="noleap")
                if not bounds[0, 0] <= start < stop <= bounds[-1, 1]:
                    raise ValueError("river forcing does not cover the requested interval")
                first = np.searchsorted(times, start, side="right") - 1
                end = np.searchsorted(times, stop, side="left")
                records = slice(first, end)
            volume = np.asarray(field[records, :], dtype=np.float64)
            bounds = bounds[records]
        if not np.all(np.isfinite(volume)) or np.any(np.abs(volume) > 1e30):
            raise ValueError("invalid or missing river coupling records")
        return cls(bounds, volume, units)

    def load_chunk(self, descriptors, step_seconds):
        starts = []
        for descriptor in descriptors:
            date = descriptor.call_date
            end = cftime.DatetimeNoLeap(date // 10000, date // 100 % 100, date % 100)
            end += timedelta(seconds=int(descriptor.call_time_of_day))
            starts.append(end - timedelta(seconds=step_seconds))
        return self.load_steps(starts, step_seconds)

    def load_steps(self, starts, step_seconds):
        if step_seconds <= 0 or not len(starts):
            raise ValueError("river forcing requires nonempty, positive-duration steps")
        times = np.asarray(cftime.date2num(starts, self.time_units, calendar="noleap"))
        indices = np.searchsorted(self.bounds[:, 0], times, side="right") - 1
        if np.any(indices < 0) or np.any(times + step_seconds > self.bounds[indices, 1]):
            raise ValueError("river forcing does not cover every complete CLM timestep")
        import jax
        return jax.device_put(self.volume[indices])
