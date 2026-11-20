
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.forcing.datm import DatmForcing, clmncep_forcing
from clm_jax.forcing.remap import SparseMapping, fill_and_remap, temporal_interpolate
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.datm_stream import DatmStream
from clm_jax.io.netcdf import load_datm_time_slice
from clm_jax.physics.solar import (
    FIXED_YEAR_2000_ORBIT,
    OrbitalParameters,
)
from clm_jax.physics.solar_runtime import (
    native_datm_coszen_temporal_interpolate as coszen_temporal_interpolate,
)


class Wfde5Record(NamedTuple):
    temperature: jnp.ndarray
    wind_speed: jnp.ndarray
    specific_humidity: jnp.ndarray
    pressure: jnp.ndarray
    longwave_down: jnp.ndarray
    shortwave_interval_mean: jnp.ndarray
    precipitation: jnp.ndarray


@dataclass(frozen=True)
class Wfde5StreamSet:

    meteorology: DatmStream
    solar: DatmStream
    precipitation: DatmStream

    @classmethod
    def from_case(cls, case: CimeCase) -> Wfde5StreamSet:
        streams = tuple(DatmStream.load(path) for path in case.datm_stream_paths())
        return cls(
            meteorology=_select_stream(
                streams,
                frozenset(("tbot", "wind", "shum", "pbot", "lwdn")),
                "WFDE5 meteorology",
            ),
            solar=_select_stream(streams, frozenset(("swdn",)), "WFDE5 solar"),
            precipitation=_select_stream(streams, frozenset(("precn",)), "WFDE5 precipitation"),
        )

    def load_record(self, time_index: int) -> Wfde5Record:

        meteorology = load_datm_time_slice(self.meteorology, time_index)
        solar = load_datm_time_slice(self.solar, time_index)
        precipitation = load_datm_time_slice(self.precipitation, time_index)
        return Wfde5Record(
            temperature=meteorology["tbot"],
            wind_speed=meteorology["wind"],
            specific_humidity=meteorology["shum"],
            pressure=meteorology["pbot"],
            longwave_down=meteorology["lwdn"],
            shortwave_interval_mean=solar["swdn"],
            precipitation=precipitation["precn"],
        )


def _select_stream(
    streams: tuple[DatmStream, ...],
    required_model_variables: frozenset[str],
    description: str,
) -> DatmStream:
    matching = tuple(
        stream
        for stream in streams
        if required_model_variables.issubset(
            mapping.model_variable.lower() for mapping in stream.field_variables
        )
    )
    if len(matching) != 1:
        raise ValueError(
            f"expected exactly one {description} stream with variables "
            f"{sorted(required_model_variables)}, found {len(matching)}"
        )
    return matching[0]


def map_wfde5_record(
    record: Wfde5Record,
    fill_mapping: SparseMapping,
    remapping: SparseMapping,
) -> Wfde5Record:

    native = jnp.stack(tuple(jnp.asarray(field).reshape(-1) for field in record))
    mapped = fill_and_remap(native, fill_mapping, remapping)
    return Wfde5Record(*(mapped[index] for index in range(len(Wfde5Record._fields))))


def prepare_wfde5_datm_step(
    lower_record: Wfde5Record,
    upper_record: Wfde5Record,
    lower_weight: jnp.ndarray | float,
    fill_mapping: SparseMapping,
    remapping: SparseMapping,
    *,
    model_calendar_day: jnp.ndarray | float,
    lower_calendar_day: jnp.ndarray | float,
    destination_latitude_radians: jnp.ndarray,
    destination_longitude_radians: jnp.ndarray,
    forcing_interval_seconds: int,
    model_step_seconds: int,
    orbit: OrbitalParameters = FIXED_YEAR_2000_ORBIT,
) -> DatmForcing:

    lower = map_wfde5_record(lower_record, fill_mapping, remapping)
    upper = map_wfde5_record(upper_record, fill_mapping, remapping)
    return prepare_mapped_wfde5_datm_step(
        lower,
        upper,
        lower_weight,
        model_calendar_day=model_calendar_day,
        lower_calendar_day=lower_calendar_day,
        destination_latitude_radians=destination_latitude_radians,
        destination_longitude_radians=destination_longitude_radians,
        forcing_interval_seconds=forcing_interval_seconds,
        model_step_seconds=model_step_seconds,
        orbit=orbit,
    )


def prepare_mapped_wfde5_datm_step(
    lower: Wfde5Record,
    upper: Wfde5Record,
    lower_weight: jnp.ndarray | float,
    *,
    model_calendar_day: jnp.ndarray | float,
    lower_calendar_day: jnp.ndarray | float,
    destination_latitude_radians: jnp.ndarray,
    destination_longitude_radians: jnp.ndarray,
    forcing_interval_seconds: int,
    model_step_seconds: int,
    orbit: OrbitalParameters = FIXED_YEAR_2000_ORBIT,
) -> DatmForcing:

    lower_weight = jnp.asarray(lower_weight, dtype=jnp.float64)

    temperature = temporal_interpolate(lower.temperature, upper.temperature, lower_weight)
    wind_speed = temporal_interpolate(lower.wind_speed, upper.wind_speed, lower_weight)
    specific_humidity = temporal_interpolate(
        lower.specific_humidity, upper.specific_humidity, lower_weight
    )
    pressure = temporal_interpolate(lower.pressure, upper.pressure, lower_weight)
    longwave_down = temporal_interpolate(lower.longwave_down, upper.longwave_down, lower_weight)
    precipitation = jnp.where(
        lower_weight >= jnp.float64(0.5), lower.precipitation, upper.precipitation
    )
    shortwave_down = coszen_temporal_interpolate(
        lower.shortwave_interval_mean,
        model_calendar_day,
        lower_calendar_day,
        destination_latitude_radians,
        destination_longitude_radians,
        interval_seconds=forcing_interval_seconds,
        model_step_seconds=model_step_seconds,
        orbit=orbit,
    )
    return clmncep_forcing(
        temperature,
        wind_speed,
        specific_humidity,
        pressure,
        longwave_down,
        shortwave_down,
        precipitation,
    )
