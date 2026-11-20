
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import PI
from clm_jax.physics.native_solar_math import native_sincos_cos, native_sincos_sin
from clm_jax.physics.numerics import fortran_divide, fortran_multiply, strict_acos


class DaylengthState(NamedTuple):
    current: jnp.ndarray
    previous: jnp.ndarray
    maximum: jnp.ndarray


_SECONDS_PER_RADIAN = np.float64(13750.9871)
MAXIMUM_DAYLENGTH_SECONDS = (
    np.float64(np.float64(2.0) * _SECONDS_PER_RADIAN) * PI
)
_LATITUDE_EPSILON = np.float64(10.0) * np.finfo(np.float64).eps
_POLE = PI / np.float64(2.0)
_OFFSET_POLE = _POLE - _LATITUDE_EPSILON


def daylength(latitude: jnp.ndarray, declination: jnp.ndarray) -> jnp.ndarray:

    latitude = jnp.asarray(latitude, dtype=jnp.float64)
    declination = jnp.asarray(declination, dtype=jnp.float64)
    invalid = (jnp.abs(latitude) >= (_POLE + _LATITUDE_EPSILON)) | (jnp.abs(declination) >= _POLE)
    adjusted_latitude = jnp.clip(latitude, min=-_OFFSET_POLE, max=_OFFSET_POLE)
    cosine_hour_angle = fortran_divide(
        -fortran_multiply(
            native_sincos_sin(adjusted_latitude), native_sincos_sin(declination)
        ),
        fortran_multiply(
            native_sincos_cos(adjusted_latitude), native_sincos_cos(declination)
        ),
    )
    cosine_hour_angle = jnp.clip(cosine_hour_angle, min=-1.0, max=1.0)
    result = fortran_multiply(
        jnp.float64(np.float64(2.0) * _SECONDS_PER_RADIAN),
        strict_acos(cosine_hour_angle),
    )
    return jnp.where(invalid, jnp.float64(jnp.nan), result)


def maximum_daylength(latitude: jnp.ndarray, obliquity: jnp.ndarray) -> jnp.ndarray:

    latitude = jnp.asarray(latitude, dtype=jnp.float64)
    obliquity = jnp.asarray(obliquity, dtype=jnp.float64)
    maximum_declination = jnp.where(latitude < 0.0, -obliquity, obliquity)
    return daylength(latitude, maximum_declination)


def initialize_daylength(
    latitude: jnp.ndarray,
    declination: jnp.ndarray,
    previous_declination: jnp.ndarray,
    obliquity: jnp.ndarray,
) -> DaylengthState:

    return DaylengthState(
        current=daylength(latitude, declination),
        previous=daylength(latitude, previous_declination),
        maximum=maximum_daylength(latitude, obliquity),
    )


def update_daylength(
    state: DaylengthState,
    latitude: jnp.ndarray,
    declination: jnp.ndarray,
    obliquity: jnp.ndarray,
    *,
    first_step_of_run_segment: bool | jnp.ndarray,
) -> DaylengthState:

    first_step = jnp.asarray(first_step_of_run_segment, dtype=jnp.bool_)
    new_current = daylength(latitude, declination)
    return DaylengthState(
        current=jnp.where(first_step, state.current, new_current),
        previous=jnp.where(first_step, state.previous, state.current),
        maximum=maximum_daylength(latitude, obliquity),
    )
