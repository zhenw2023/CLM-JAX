
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from clm_jax.constants import FREEZING_TEMPERATURE_FRESHWATER
from clm_jax.io.restart import LANDUNIT_GLACIER, LANDUNIT_SOIL
from clm_jax.physics.numerics import fortran_divide, fortran_multiply

_GLACIER_COLUMN_TYPE_BASE = LANDUNIT_GLACIER * 100
_BARELAND_NORMALIZATION_TOLERANCE = 1.0e-13


class LandToGlacierInputs(NamedTuple):

    column_gridcell: jnp.ndarray
    column_landunit_type: jnp.ndarray
    column_type: jnp.ndarray
    column_weight_on_gridcell: jnp.ndarray
    landunit_gridcell: jnp.ndarray
    landunit_type: jnp.ndarray
    landunit_weight_on_gridcell: jnp.ndarray
    do_surface_mass_balance: jnp.ndarray
    surface_temperature: jnp.ndarray
    elevation: jnp.ndarray
    ice_flux: jnp.ndarray
    initialization: jnp.ndarray


class LandToGlacierResult(NamedTuple):
    surface_temperature_gridcell_class: jnp.ndarray
    elevation_gridcell_class: jnp.ndarray
    ice_flux_gridcell_class: jnp.ndarray
    assignment_count_gridcell_class: jnp.ndarray
    bareland_normalization_column: jnp.ndarray


def _validate_shapes(
    inputs: LandToGlacierInputs,
    *,
    number_of_gridcells: int,
    maximum_glacier_classes: int,
) -> None:
    column_shape = inputs.column_gridcell.shape
    landunit_shape = inputs.landunit_gridcell.shape
    if len(column_shape) != 1 or len(landunit_shape) != 1:
        raise ValueError("land-to-glacier topology fields must be vectors")
    column_fields = (
        "column_landunit_type",
        "column_type",
        "column_weight_on_gridcell",
        "do_surface_mass_balance",
        "surface_temperature",
        "elevation",
        "ice_flux",
    )
    if any(getattr(inputs, name).shape != column_shape for name in column_fields):
        raise ValueError("all land-to-glacier column fields must share one axis")
    landunit_fields = ("landunit_type", "landunit_weight_on_gridcell")
    if any(getattr(inputs, name).shape != landunit_shape for name in landunit_fields):
        raise ValueError("all land-to-glacier landunit fields must share one axis")
    if inputs.initialization.shape != ():
        raise ValueError("initialization must be a scalar")
    if number_of_gridcells <= 0 or maximum_glacier_classes <= 0:
        raise ValueError("coupling dimensions must be positive")


def land_to_glacier_coupling(
    inputs: LandToGlacierInputs,
    *,
    number_of_gridcells: int,
    maximum_glacier_classes: int,
) -> LandToGlacierResult:

    _validate_shapes(
        inputs,
        number_of_gridcells=number_of_gridcells,
        maximum_glacier_classes=maximum_glacier_classes,
    )
    column_gridcell = jnp.asarray(inputs.column_gridcell, dtype=jnp.int32)
    column_landunit_type = jnp.asarray(inputs.column_landunit_type, dtype=jnp.int32)
    column_type = jnp.asarray(inputs.column_type, dtype=jnp.int32)
    column_weight = jnp.asarray(inputs.column_weight_on_gridcell, dtype=jnp.float64)
    landunit_gridcell = jnp.asarray(inputs.landunit_gridcell, dtype=jnp.int32)
    landunit_type = jnp.asarray(inputs.landunit_type, dtype=jnp.int32)
    landunit_weight = jnp.asarray(inputs.landunit_weight_on_gridcell, dtype=jnp.float64)
    do_smb = jnp.asarray(inputs.do_surface_mass_balance, dtype=jnp.bool_)
    surface_temperature = jnp.asarray(inputs.surface_temperature, dtype=jnp.float64)
    elevation = jnp.asarray(inputs.elevation, dtype=jnp.float64)
    ice_flux = jnp.asarray(inputs.ice_flux, dtype=jnp.float64)
    initialization = jnp.asarray(inputs.initialization, dtype=jnp.bool_)

    glacier_landunit_weight = jnp.where(
        landunit_type == jnp.int32(LANDUNIT_GLACIER),
        landunit_weight,
        jnp.float64(0.0),
    )
    glacier_area = jax.ops.segment_sum(
        glacier_landunit_weight,
        landunit_gridcell,
        number_of_gridcells,
    )
    column_glacier_area = glacier_area[column_gridcell]
    all_glacier = (
        jnp.abs(column_glacier_area - jnp.float64(1.0))
        < jnp.float64(_BARELAND_NORMALIZATION_TOLERANCE)
    )
    safe_denominator = jnp.where(
        all_glacier,
        jnp.float64(1.0),
        jnp.float64(1.0) - column_glacier_area,
    )
    bareland_normalization = jnp.where(
        all_glacier,
        jnp.float64(1.0),
        fortran_divide(column_weight, safe_denominator),
    )

    glacier_column = column_landunit_type == jnp.int32(LANDUNIT_GLACIER)
    soil_column = column_landunit_type == jnp.int32(LANDUNIT_SOIL)
    eligible = do_smb & (glacier_column | soil_column)
    elevation_class = jnp.where(
        glacier_column,
        column_type - jnp.int32(_GLACIER_COLUMN_TYPE_BASE),
        jnp.int32(0),
    )
    class_count = maximum_glacier_classes + 1
    flat_size = number_of_gridcells * class_count
    flat_index = column_gridcell * jnp.int32(class_count) + elevation_class
    safe_index = jnp.where(eligible, flat_index, jnp.int32(0))
    assignment_count = jax.ops.segment_sum(
        eligible.astype(jnp.int32),
        safe_index,
        flat_size,
    )

    def assigned_sum(values: jnp.ndarray) -> jnp.ndarray:
        return jax.ops.segment_sum(
            jnp.where(eligible, values, jnp.float64(0.0)),
            safe_index,
            flat_size,
        )

    temperature_sum = assigned_sum(surface_temperature)
    elevation_sum = assigned_sum(elevation)
    normalization = jnp.where(glacier_column, jnp.float64(1.0), bareland_normalization)
    normalized_ice_flux = fortran_multiply(ice_flux, normalization)
    flux_sum = assigned_sum(jnp.where(initialization, jnp.float64(0.0), normalized_ice_flux))
    assigned = assignment_count > jnp.int32(0)
    output_shape = (number_of_gridcells, class_count)
    return LandToGlacierResult(
        surface_temperature_gridcell_class=jnp.where(
            assigned,
            temperature_sum,
            jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
        ).reshape(output_shape),
        elevation_gridcell_class=jnp.where(
            assigned,
            elevation_sum,
            jnp.float64(0.0),
        ).reshape(output_shape),
        ice_flux_gridcell_class=jnp.where(
            assigned & ~initialization,
            flux_sum,
            jnp.float64(0.0),
        ).reshape(output_shape),
        assignment_count_gridcell_class=assignment_count.reshape(output_shape),
        bareland_normalization_column=bareland_normalization,
    )


__all__ = ["LandToGlacierInputs", "LandToGlacierResult", "land_to_glacier_coupling"]
