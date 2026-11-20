
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.restart import LANDUNIT_URBAN_MAX, LANDUNIT_URBAN_MIN
from clm_jax.physics.numerics import fortran_add, fortran_multiply


class UrbanTimeVaryingInputs(NamedTuple):

    building_maximum_temperature_lower_gridcell_density: jnp.ndarray
    building_maximum_temperature_upper_gridcell_density: jnp.ndarray
    lower_weight: float | jnp.ndarray
    landunit_gridcell_index: jnp.ndarray
    landunit_type: jnp.ndarray
    urban_valid_gridcell: jnp.ndarray


class UrbanTimeVaryingState(NamedTuple):

    building_maximum_temperature_landunit: jnp.ndarray


class UrbanTimeVaryingDiagnostics(NamedTuple):

    lower_weight: jnp.ndarray
    urban_landunit_count: jnp.ndarray
    invalid_urban_landunit_count: jnp.ndarray


class UrbanTimeVaryingResult(NamedTuple):
    state: UrbanTimeVaryingState
    diagnostics: UrbanTimeVaryingDiagnostics


def _validate(inputs: UrbanTimeVaryingInputs) -> tuple[int, int]:
    lower_shape = jnp.shape(
        inputs.building_maximum_temperature_lower_gridcell_density
    )
    upper_shape = jnp.shape(
        inputs.building_maximum_temperature_upper_gridcell_density
    )
    if len(lower_shape) != 2 or lower_shape[1] != 3:
        raise ValueError("urban stream records must have shape (gridcell, 3)")
    if upper_shape != lower_shape:
        raise ValueError("lower and upper urban stream records must have equal shapes")
    landunit_shape = jnp.shape(inputs.landunit_gridcell_index)
    if len(landunit_shape) != 1 or jnp.shape(inputs.landunit_type) != landunit_shape:
        raise ValueError("urban landunit lookup arrays must share one landunit axis")
    if jnp.shape(inputs.urban_valid_gridcell) != (lower_shape[0],):
        raise ValueError("urban_valid_gridcell must use the native gridcell axis")
    return lower_shape[0], landunit_shape[0]


def interpolate_urban_time_varying(
    inputs: UrbanTimeVaryingInputs,
) -> UrbanTimeVaryingResult:

    number_of_gridcells, _ = _validate(inputs)
    lower = jnp.asarray(
        inputs.building_maximum_temperature_lower_gridcell_density,
        dtype=jnp.float64,
    )
    upper = jnp.asarray(
        inputs.building_maximum_temperature_upper_gridcell_density,
        dtype=jnp.float64,
    )
    lower_weight = jnp.asarray(inputs.lower_weight, dtype=jnp.float64)
    mapped = fortran_add(
        fortran_multiply(lower, lower_weight),
        fortran_multiply(upper, jnp.float64(1.0) - lower_weight),
    )

    landunit_gridcell = jnp.asarray(inputs.landunit_gridcell_index, dtype=jnp.int32)
    landunit_type = jnp.asarray(inputs.landunit_type, dtype=jnp.int32)
    urban = (landunit_type >= jnp.int32(LANDUNIT_URBAN_MIN)) & (
        landunit_type <= jnp.int32(LANDUNIT_URBAN_MAX)
    )
    density = jnp.clip(
        landunit_type - jnp.int32(LANDUNIT_URBAN_MIN),
        jnp.int32(0),
        jnp.int32(2),
    )
    safe_gridcell = jnp.clip(
        landunit_gridcell,
        jnp.int32(0),
        jnp.int32(number_of_gridcells - 1),
    )
    selected = mapped[safe_gridcell, density]
    result = jnp.where(urban, selected, jnp.float64(CLM_SPECIAL_VALUE))
    urban_valid = jnp.asarray(inputs.urban_valid_gridcell, dtype=jnp.bool_)[safe_gridcell]
    invalid = urban & (
        ~urban_valid
        | (landunit_gridcell < 0)
        | (landunit_gridcell >= number_of_gridcells)
        | (selected <= jnp.float64(0.0))
    )
    return UrbanTimeVaryingResult(
        state=UrbanTimeVaryingState(
            building_maximum_temperature_landunit=result,
        ),
        diagnostics=UrbanTimeVaryingDiagnostics(
            lower_weight=lower_weight,
            urban_landunit_count=jnp.sum(urban, dtype=jnp.int32),
            invalid_urban_landunit_count=jnp.sum(invalid, dtype=jnp.int32),
        ),
    )


__all__ = [
    "UrbanTimeVaryingDiagnostics",
    "UrbanTimeVaryingInputs",
    "UrbanTimeVaryingResult",
    "UrbanTimeVaryingState",
    "interpolate_urban_time_varying",
]
