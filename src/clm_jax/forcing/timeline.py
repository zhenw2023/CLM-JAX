
from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from clm_jax.calendar import (
    add_noleap_seconds,
    integer_date_and_time,
    noleap_absolute_seconds,
    noleap_calendar_day,
    noleap_seconds_between,
    validate_noleap_datetime,
)
from clm_jax.forcing.area import correct_atmospheric_land_fluxes, load_land_flux_area_factor
from clm_jax.forcing.atmosphere import compress_datm_forcing_to_land_grid
from clm_jax.forcing.remap import SparseMapping, load_sparse_mapping
from clm_jax.forcing.wfde5 import (
    Wfde5Record,
    Wfde5StreamSet,
    map_wfde5_record,
    prepare_mapped_wfde5_datm_step,
)
from clm_jax.io.cime_case import CimeCase

if TYPE_CHECKING:
    from clm_jax.forcing.datm import DatmForcing
from clm_jax.io.datm_stream import DatmStream
from clm_jax.io.netcdf import load_variables
from clm_jax.io.restart import load_subgrid_topology


class Wfde5StepDescriptor(NamedTuple):

    lower_index: int
    upper_index: int
    lower_weight: np.float64
    model_calendar_day: np.float64
    lower_calendar_day: np.float64
    call_date: int
    call_time_of_day: int


@dataclass(frozen=True)
class Wfde5TimeAxis:

    instants: tuple[datetime, ...]
    absolute_seconds: np.ndarray
    forcing_interval_seconds: int

    @classmethod
    def from_instants(cls, instants: Sequence[datetime]) -> Wfde5TimeAxis:
        values = tuple(instants)
        if len(values) < 2:
            raise ValueError("WFDE5 forcing needs at least two time records")
        for index, value in enumerate(values):
            validate_noleap_datetime(value, name=f"WFDE5 time[{index}]")
        seconds = np.fromiter(
            (noleap_absolute_seconds(value) for value in values),
            dtype=np.int64,
            count=len(values),
        )
        intervals = np.diff(seconds)
        if np.any(intervals <= 0) or not np.all(intervals == intervals[0]):
            raise ValueError("WFDE5 time coordinate must be strictly increasing and regular")
        return cls(values, seconds, int(intervals[0]))

    @classmethod
    def from_streams(cls, streams: Wfde5StreamSet) -> Wfde5TimeAxis:
        axes = tuple(
            _load_stream_instants(stream)
            for stream in (streams.meteorology, streams.solar, streams.precipitation)
        )
        return cls._common_axis(axes)

    @classmethod
    def _common_axis(cls, axes) -> Wfde5TimeAxis:
        if axes[1] != axes[0] or axes[2] != axes[0]:
            raise ValueError("WFDE5 meteorology, solar and precipitation times differ")
        return cls.from_instants(axes[0])

    @classmethod
    def through_sample(
        cls, streams: Wfde5StreamSet, last_sample: datetime
    ) -> tuple[Wfde5StreamSet, Wfde5TimeAxis]:
        prefixes, axes = [], []
        for stream in (streams.meteorology, streams.solar, streams.precipitation):
            instants = []
            files = []
            for path in stream.field_files:
                files.append(path)
                instants.extend(_load_stream_instants(replace(stream, field_files=(path,))))
                if len(instants) >= 2 and instants[-1] >= last_sample:
                    break
            axis = cls.from_instants(instants)
            upper = axis.describe(last_sample).upper_index
            axes.append(axis.instants[:upper + 1])
            prefixes.append(replace(stream, field_files=tuple(files)))
        return Wfde5StreamSet(*prefixes), cls._common_axis(axes)

    def describe(
        self,
        sampling_time: datetime,
        *,
        call_time: datetime | None = None,
    ) -> Wfde5StepDescriptor:

        target = noleap_absolute_seconds(sampling_time)
        source = self.absolute_seconds
        if target < int(source[0]) or target > int(source[-1]):
            raise ValueError(
                f"DATM sampling time {sampling_time.isoformat()} lies outside WFDE5 coverage "
                f"{self.instants[0].isoformat()}..{self.instants[-1].isoformat()}"
            )
        upper_index = max(1, int(np.searchsorted(source, target, side="left")))
        lower_index = upper_index - 1
        lower_seconds = int(source[lower_index])
        upper_seconds = int(source[upper_index])
        lower_weight = np.float64(upper_seconds - target) / np.float64(
            upper_seconds - lower_seconds
        )
        boundary = sampling_time if call_time is None else call_time
        call_date, call_seconds = integer_date_and_time(boundary)
        return Wfde5StepDescriptor(
            lower_index=lower_index,
            upper_index=upper_index,
            lower_weight=lower_weight,
            model_calendar_day=np.float64(noleap_calendar_day(sampling_time)),
            lower_calendar_day=np.float64(noleap_calendar_day(self.instants[lower_index])),
            call_date=call_date,
            call_time_of_day=call_seconds,
        )

    def iter_descriptors(
        self,
        start: datetime,
        stop: datetime,
        step_seconds: int,
    ) -> Iterator[Wfde5StepDescriptor]:

        if isinstance(step_seconds, bool) or not isinstance(step_seconds, int):
            raise TypeError("model forcing step must be an integer number of seconds")
        if step_seconds <= 0:
            raise ValueError("model forcing step must be positive")
        duration = noleap_seconds_between(start, stop)
        if duration <= 0 or duration % step_seconds:
            raise ValueError("forcing interval must be a positive whole number of steps")
        for step in range(1, duration // step_seconds + 1):
            call = add_noleap_seconds(start, step * step_seconds)
            sampling_time = add_noleap_seconds(start, (step - 1) * step_seconds)
            yield self.describe(sampling_time, call_time=call)


class Wfde5ChunkLoader:

    def __init__(
        self,
        streams: Wfde5StreamSet,
        time_axis: Wfde5TimeAxis,
        fill_mapping: SparseMapping,
        remapping: SparseMapping,
        destination_latitude_radians: jax.Array,
        destination_longitude_radians: jax.Array,
        land_latitude_index: jax.Array,
        land_longitude_index: jax.Array,
        longitude_count: int,
        land_flux_area_factor=None,
    ) -> None:
        self.streams = streams
        self.time_axis = time_axis
        if jax.default_backend() == "gpu":
            from clm_jax.forcing.ordered_mapping import prepare_ordered_mapping

            fill_mapping = prepare_ordered_mapping(fill_mapping)
            remapping = prepare_ordered_mapping(remapping)
        self.fill_mapping = fill_mapping
        self.remapping = remapping
        self.destination_latitude_radians = destination_latitude_radians
        self.destination_longitude_radians = destination_longitude_radians
        self.land_latitude_index = land_latitude_index
        self.land_longitude_index = land_longitude_index
        self.longitude_count = longitude_count
        self.land_flux_area_factor = (
            None if land_flux_area_factor is None
            else jnp.asarray(land_flux_area_factor, dtype=jnp.float64)
        )
        self._mapped_records: OrderedDict[int, Wfde5Record] = OrderedDict()

    def _load_mapped_record(self, index: int) -> Wfde5Record:
        if index not in self._mapped_records:
            self._mapped_records[index] = map_wfde5_record(
                self.streams.load_record(index), self.fill_mapping, self.remapping,
            )
            if len(self._mapped_records) > 3:
                self._mapped_records.popitem(last=False)
        self._mapped_records.move_to_end(index)
        return self._mapped_records[index]

    @classmethod
    def from_case(
        cls,
        case: CimeCase,
        restart_file: str | Path,
        fill_mapping_file: str | Path,
        remapping_file: str | Path,
        *,
        last_sampling_time: datetime | None = None,
    ) -> Wfde5ChunkLoader:
        streams = Wfde5StreamSet.from_case(case)
        if last_sampling_time is None:
            time_axis = Wfde5TimeAxis.from_streams(streams)
        else:
            streams, time_axis = Wfde5TimeAxis.through_sample(streams, last_sampling_time)
        fill_mapping = load_sparse_mapping(fill_mapping_file)
        remapping = load_sparse_mapping(remapping_file)
        domain = case.path_value("ATM_DOMAIN_PATH") / case.value("ATM_DOMAIN_FILE")
        coordinates = load_variables(domain, ["xc", "yc"], dtype=np.float64)
        longitude = jnp.asarray(coordinates["xc"], dtype=jnp.float64)
        latitude = jnp.asarray(coordinates["yc"], dtype=jnp.float64)
        if longitude.shape != latitude.shape or longitude.ndim != 2:
            raise ValueError("configured DATM destination domain must have 2-D xc/yc")
        if remapping.destination_size != longitude.size:
            raise ValueError("WFDE5 remapping and configured DATM domain sizes differ")
        topology = load_subgrid_topology(restart_file)
        radians = jnp.float64(np.pi / 180.0)
        return cls(
            streams=streams,
            time_axis=time_axis,
            fill_mapping=fill_mapping,
            remapping=remapping,
            destination_latitude_radians=latitude.reshape(-1) * radians,
            destination_longitude_radians=longitude.reshape(-1) * radians,
            land_latitude_index=topology.grid.latitude_index,
            land_longitude_index=topology.grid.longitude_index,
            longitude_count=longitude.shape[1],
            land_flux_area_factor=load_land_flux_area_factor(case, topology),
        )

    def load_chunk(
        self,
        descriptors: Sequence[Wfde5StepDescriptor],
        *,
        model_step_seconds: int,
    ):

        requested = tuple(descriptors)
        if not requested:
            raise ValueError("cannot load an empty WFDE5 forcing chunk")
        record_indices = sorted({index for descriptor in requested for index in descriptor[:2]})
        mapped: dict[int, Wfde5Record] = {}
        for index in record_indices:
            mapped[index] = self._load_mapped_record(index)

        steps = []
        for descriptor in requested:
            forcing = prepare_mapped_wfde5_datm_step(
                mapped[descriptor.lower_index],
                mapped[descriptor.upper_index],
                descriptor.lower_weight,
                model_calendar_day=descriptor.model_calendar_day,
                lower_calendar_day=descriptor.lower_calendar_day,
                destination_latitude_radians=self.destination_latitude_radians,
                destination_longitude_radians=self.destination_longitude_radians,
                forcing_interval_seconds=self.time_axis.forcing_interval_seconds,
                model_step_seconds=model_step_seconds,
            )
            steps.append(
                compress_datm_forcing_to_land_grid(
                    forcing,
                    self.land_latitude_index,
                    self.land_longitude_index,
                    longitude_count=self.longitude_count,
                )
            )
        chunk = jax.tree.map(lambda *values: jnp.stack(values), *steps)
        if self.land_flux_area_factor is not None:
            chunk = correct_atmospheric_land_fluxes(chunk, self.land_flux_area_factor)
        return chunk


@dataclass(frozen=True)
class RestartAwareWfde5ChunkLoader:

    wfde5: Wfde5ChunkLoader
    initial_coupler_forcing: DatmForcing | None

    @classmethod
    def from_case(
        cls,
        case: CimeCase,
        restart_file: str | Path,
        fill_mapping_file: str | Path,
        remapping_file: str | Path,
        *,
        coupler_restart_file: str | Path | None = None,
        last_sampling_time: datetime | None = None,
    ) -> RestartAwareWfde5ChunkLoader:

        wfde5 = Wfde5ChunkLoader.from_case(
            case,
            restart_file,
            fill_mapping_file,
            remapping_file,
            last_sampling_time=last_sampling_time,
        )
        initial = None
        if coupler_restart_file is not None:
            from clm_jax.io.coupler import (  
                load_atmospheric_coupler_forcing,
            )

            initial = load_atmospheric_coupler_forcing(
                coupler_restart_file,
                load_subgrid_topology(restart_file),
                land_flux_area_factor=wfde5.land_flux_area_factor,
            )
        return cls(wfde5=wfde5, initial_coupler_forcing=initial)

    @property
    def time_axis(self) -> Wfde5TimeAxis:
        return self.wfde5.time_axis

    def load_chunk(
        self,
        descriptors: Sequence[Wfde5StepDescriptor],
        *,
        model_step_seconds: int,
        first_model_step_index: int,
    ) -> DatmForcing:

        if (
            isinstance(first_model_step_index, bool)
            or not isinstance(first_model_step_index, int)
            or first_model_step_index < 0
        ):
            raise ValueError("first_model_step_index must be a nonnegative integer")
        requested = tuple(descriptors)
        chunk = self.wfde5.load_chunk(
            requested,
            model_step_seconds=model_step_seconds,
        )
        if self.initial_coupler_forcing is None or first_model_step_index != 0:
            return chunk
        if not requested:
            raise ValueError("cannot restore CPL forcing into an empty chunk")
        return jax.tree.map(
            lambda values, initial: values.at[0].set(initial),
            chunk,
            self.initial_coupler_forcing,
        )


def _load_stream_instants(stream: DatmStream) -> tuple[datetime, ...]:
    if not stream.field_files:
        raise ValueError("WFDE5 stream has no field files")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="numpy.ndarray size changed.*",
            category=RuntimeWarning,
        )
        from netCDF4 import Dataset, num2date  

    return tuple(
        instant
        for path in stream.field_files
        for instant in _load_file_instants(path, Dataset, num2date)
    )


def _load_file_instants(path, Dataset, num2date) -> tuple[datetime, ...]:
    with Dataset(path, mode="r") as dataset:
        time = dataset.variables["time"]
        units = str(time.getncattr("units"))
        calendar = str(time.getncattr("calendar")) if "calendar" in time.ncattrs() else "noleap"
        if calendar.strip().lower() not in {"noleap", "no_leap", "365_day"}:
            raise ValueError(f"WFDE5 stream must use a NO_LEAP calendar: {path}")
        decoded = num2date(
            np.asarray(time[:], dtype=np.float64),
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=True,
        )
    result = tuple(
        datetime(
            value.year,
            value.month,
            value.day,
            value.hour,
            value.minute,
            value.second,
            value.microsecond,
        )
        for value in decoded
    )
    return result


__all__ = [
    "RestartAwareWfde5ChunkLoader",
    "Wfde5ChunkLoader",
    "Wfde5StepDescriptor",
    "Wfde5TimeAxis",
]
