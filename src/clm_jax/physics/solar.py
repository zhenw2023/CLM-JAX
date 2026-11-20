
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import PI, SECONDS_PER_DAY


class OrbitalParameters(NamedTuple):
    eccentricity: jnp.ndarray
    obliquity_radians: jnp.ndarray
    mean_longitude_vernal_equinox: jnp.ndarray
    perihelion_longitude_plus_pi: jnp.ndarray


FIXED_YEAR_2000_ORBIT = OrbitalParameters(
    eccentricity=np.float64(1.67036603927656034e-2),
    obliquity_radians=np.float64(4.09101122579778931e-1),
    mean_longitude_vernal_equinox=np.float64(-3.24724956615293284e-2),
    perihelion_longitude_plus_pi=np.float64(4.93745779048815780),
)

SOLAR_ZENITH_MINIMUM = 0.001


def solar_declination(
    calendar_day: jnp.ndarray,
    orbit: OrbitalParameters = FIXED_YEAR_2000_ORBIT,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    calendar_day = jnp.asarray(calendar_day, dtype=jnp.float64)
    eccentricity = jnp.asarray(orbit.eccentricity, dtype=jnp.float64)
    mean_longitude = jnp.asarray(orbit.mean_longitude_vernal_equinox, dtype=jnp.float64) + (
        calendar_day - jnp.float64(80.5)
    ) * jnp.float64(2.0) * jnp.float64(PI) / jnp.float64(365.0)
    relative_longitude = mean_longitude - orbit.perihelion_longitude_plus_pi
    sine_relative_longitude = jnp.sin(relative_longitude)
    true_longitude = mean_longitude + eccentricity * (
        jnp.float64(2.0) * sine_relative_longitude
        + eccentricity
        * (
            jnp.float64(1.25) * jnp.sin(jnp.float64(2.0) * relative_longitude)
            + eccentricity
            * (
                (jnp.float64(13.0) / jnp.float64(12.0))
                * jnp.sin(jnp.float64(3.0) * relative_longitude)
                - jnp.float64(0.25) * sine_relative_longitude
            )
        )
    )
    inverse_distance = (
        jnp.float64(1.0)
        + eccentricity * jnp.cos(true_longitude - orbit.perihelion_longitude_plus_pi)
    ) / (jnp.float64(1.0) - eccentricity * eccentricity)
    declination = jnp.arcsin(jnp.sin(orbit.obliquity_radians) * jnp.sin(true_longitude))
    return declination, inverse_distance * inverse_distance


def cosine_solar_zenith(
    calendar_day: jnp.ndarray,
    latitude_radians: jnp.ndarray,
    longitude_radians: jnp.ndarray,
    orbit: OrbitalParameters = FIXED_YEAR_2000_ORBIT,
) -> jnp.ndarray:

    return jnp.maximum(
        jnp.float64(SOLAR_ZENITH_MINIMUM),
        raw_cosine_solar_zenith(
            calendar_day,
            latitude_radians,
            longitude_radians,
            orbit,
        ),
    )


def raw_cosine_solar_zenith(
    calendar_day: jnp.ndarray,
    latitude_radians: jnp.ndarray,
    longitude_radians: jnp.ndarray,
    orbit: OrbitalParameters = FIXED_YEAR_2000_ORBIT,
) -> jnp.ndarray:

    calendar_day = jnp.asarray(calendar_day, dtype=jnp.float64)
    latitude = jnp.asarray(latitude_radians, dtype=jnp.float64)
    longitude = jnp.asarray(longitude_radians, dtype=jnp.float64)
    declination, _ = solar_declination(calendar_day, orbit)
    hour_angle = (calendar_day - jnp.floor(calendar_day)) * jnp.float64(2.0) * jnp.float64(
        PI
    ) + longitude
    cosine = jnp.sin(latitude) * jnp.sin(declination) - jnp.cos(latitude) * jnp.cos(
        declination
    ) * jnp.cos(hour_angle)
    return cosine


def average_cosine_solar_zenith(
    lower_calendar_day: jnp.ndarray,
    latitude_radians: jnp.ndarray,
    longitude_radians: jnp.ndarray,
    *,
    interval_seconds: int,
    model_step_seconds: int,
    orbit: OrbitalParameters = FIXED_YEAR_2000_ORBIT,
) -> jnp.ndarray:

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if model_step_seconds <= 0:
        raise ValueError("model_step_seconds must be positive")

    local_step = model_step_seconds
    if interval_seconds % local_step != 0:
        local_step = interval_seconds // (interval_seconds // local_step + 1)

    elapsed = 0
    accumulated_seconds = 0
    total = jnp.zeros_like(
        jnp.broadcast_arrays(latitude_radians, longitude_radians)[0], dtype=jnp.float64
    )
    while elapsed < interval_seconds:
        step = min(local_step, interval_seconds - elapsed)
        elapsed += step
        sample_day = jnp.asarray(lower_calendar_day, dtype=jnp.float64) + jnp.float64(
            elapsed
        ) / jnp.float64(SECONDS_PER_DAY)
        total = total + cosine_solar_zenith(
            sample_day,
            latitude_radians,
            longitude_radians,
            orbit,
        ) * jnp.float64(step)
        accumulated_seconds += step
    return total / jnp.float64(accumulated_seconds)


def coszen_temporal_interpolate(
    interval_mean_shortwave: jnp.ndarray,
    model_calendar_day: jnp.ndarray,
    lower_calendar_day: jnp.ndarray,
    latitude_radians: jnp.ndarray,
    longitude_radians: jnp.ndarray,
    *,
    interval_seconds: int,
    model_step_seconds: int,
    orbit: OrbitalParameters = FIXED_YEAR_2000_ORBIT,
) -> jnp.ndarray:

    instantaneous_cosine = cosine_solar_zenith(
        model_calendar_day,
        latitude_radians,
        longitude_radians,
        orbit,
    )
    average_cosine = average_cosine_solar_zenith(
        lower_calendar_day,
        latitude_radians,
        longitude_radians,
        interval_seconds=interval_seconds,
        model_step_seconds=model_step_seconds,
        orbit=orbit,
    )
    daytime = instantaneous_cosine > jnp.float64(SOLAR_ZENITH_MINIMUM)
    return jnp.where(
        daytime,
        jnp.asarray(interval_mean_shortwave, dtype=jnp.float64)
        * instantaneous_cosine
        / average_cosine,
        jnp.float64(0.0),
    )
