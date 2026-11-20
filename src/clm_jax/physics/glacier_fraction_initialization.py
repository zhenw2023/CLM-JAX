
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_GLACIER,
    LANDUNIT_LAKE,
    LANDUNIT_SOIL,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    LANDUNIT_WETLAND,
)
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
    fortran_ordered_sum,
)

MAXIMUM_LANDUNIT_TYPE = 9
GLACIER_COLUMN_TYPE_BASE = LANDUNIT_GLACIER * 100

LANDUNIT_SUM_TOLERANCE = float.fromhex("0x1.6849b80000000p-47")

_DECREASE_ORDER = (
    LANDUNIT_SOIL - 1,
    LANDUNIT_CROP - 1,
    LANDUNIT_URBAN_MAX - 1,  
    LANDUNIT_URBAN_MAX - 2,  
    LANDUNIT_URBAN_MIN - 1,  
    LANDUNIT_WETLAND - 1,
    LANDUNIT_LAKE - 1,
)


class GlacierFractionInitializationInputs(NamedTuple):

    landunit_gridcell: jnp.ndarray
    landunit_type: jnp.ndarray
    landunit_weight_on_gridcell: jnp.ndarray
    column_gridcell: jnp.ndarray
    column_landunit: jnp.ndarray
    column_landunit_type: jnp.ndarray
    column_type: jnp.ndarray
    column_weight_on_landunit: jnp.ndarray
    patch_column: jnp.ndarray
    patch_weight_on_column: jnp.ndarray
    glacier_ice_mask: jnp.ndarray
    glacier_fraction_by_class: jnp.ndarray
    has_virtual_columns_gridcell: jnp.ndarray


class GlacierFractionInitializationResult(NamedTuple):

    landunit_weight_on_gridcell: jnp.ndarray
    landunit_active: jnp.ndarray
    column_weight_on_gridcell: jnp.ndarray
    column_weight_on_landunit: jnp.ndarray
    column_active: jnp.ndarray
    patch_weight_on_gridcell: jnp.ndarray
    patch_weight_on_landunit: jnp.ndarray
    patch_weight_on_column: jnp.ndarray
    patch_active: jnp.ndarray
    landunit_sum_gridcell: jnp.ndarray
    missing_landunit_assignment_gridcell_type: jnp.ndarray
    missing_glacier_class_assignment_gridcell_class: jnp.ndarray


def _validate_shapes(inputs: GlacierFractionInitializationInputs) -> None:
    landunit_shape = inputs.landunit_gridcell.shape
    column_shape = inputs.column_gridcell.shape
    patch_shape = inputs.patch_column.shape
    grid_shape = inputs.glacier_ice_mask.shape
    if any(len(shape) != 1 for shape in (landunit_shape, column_shape, patch_shape, grid_shape)):
        raise ValueError("subgrid topology and ice-mask fields must be vectors")
    if any(
        getattr(inputs, name).shape != landunit_shape
        for name in ("landunit_type", "landunit_weight_on_gridcell")
    ):
        raise ValueError("all landunit fields must share one axis")
    if any(
        getattr(inputs, name).shape != column_shape
        for name in (
            "column_landunit",
            "column_landunit_type",
            "column_type",
            "column_weight_on_landunit",
        )
    ):
        raise ValueError("all column fields must share one axis")
    if inputs.patch_weight_on_column.shape != patch_shape:
        raise ValueError("all patch fields must share one axis")
    if inputs.has_virtual_columns_gridcell.shape != grid_shape:
        raise ValueError("virtual-column and ice-mask fields must share one grid")
    fraction_shape = inputs.glacier_fraction_by_class.shape
    if len(fraction_shape) != 2 or fraction_shape[0] != grid_shape[0]:
        raise ValueError("glacier fractions must have shape (gridcell, class)")
    if fraction_shape[1] < 2:
        raise ValueError("glacier fractions require bare-land and at least one ice class")


def _landunit_sum(weights: jnp.ndarray) -> jnp.ndarray:
    return fortran_ordered_sum(jnp.moveaxis(weights, -1, 0))


def reconcile_landunit_weights(weights: jnp.ndarray) -> jnp.ndarray:

    weights = jnp.asarray(weights, dtype=jnp.float64)
    if weights.shape[-1:] != (MAXIMUM_LANDUNIT_TYPE,):
        raise ValueError("the final axis must contain all nine CLM landunit types")

    one = jnp.float64(1.0)
    zero = jnp.float64(0.0)
    tolerance = jnp.float64(LANDUNIT_SUM_TOLERANCE)
    initial_sum = _landunit_sum(weights)
    balanced = jnp.abs(fortran_add(initial_sum, -one)) <= tolerance
    grow = (~balanced) & (initial_sum < one)
    soil_increment = fortran_add(one, -initial_sum)
    soil = jnp.where(
        grow,
        fortran_add(weights[..., LANDUNIT_SOIL - 1], soil_increment),
        weights[..., LANDUNIT_SOIL - 1],
    )
    result = weights.at[..., LANDUNIT_SOIL - 1].set(soil)

    shrink = (~balanced) & (~grow)
    excess = fortran_add(initial_sum, -one)
    for landunit_index in _DECREASE_ORDER:
        decrease = shrink & (excess > tolerance)
        candidate = fortran_add(result[..., landunit_index], -excess)
        updated = jnp.where(decrease, jnp.maximum(candidate, zero), result[..., landunit_index])
        result = result.at[..., landunit_index].set(updated)
        current_sum = _landunit_sum(result)
        excess = fortran_add(current_sum, -one)
    return result


def initialize_glacier_fractions(
    inputs: GlacierFractionInitializationInputs,
    *,
    glc_do_dynglacier: bool,
    run_zero_weight_urban: bool,
    all_active: bool = False,
) -> GlacierFractionInitializationResult:

    _validate_shapes(inputs)
    landunit_gridcell = jnp.asarray(inputs.landunit_gridcell, dtype=jnp.int32)
    landunit_type = jnp.asarray(inputs.landunit_type, dtype=jnp.int32)
    landunit_weight = jnp.asarray(inputs.landunit_weight_on_gridcell, dtype=jnp.float64)
    column_gridcell = jnp.asarray(inputs.column_gridcell, dtype=jnp.int32)
    column_landunit = jnp.asarray(inputs.column_landunit, dtype=jnp.int32)
    column_landunit_type = jnp.asarray(inputs.column_landunit_type, dtype=jnp.int32)
    column_type = jnp.asarray(inputs.column_type, dtype=jnp.int32)
    column_weight_on_landunit = jnp.asarray(
        inputs.column_weight_on_landunit,
        dtype=jnp.float64,
    )
    patch_column = jnp.asarray(inputs.patch_column, dtype=jnp.int32)
    patch_weight_on_column = jnp.asarray(inputs.patch_weight_on_column, dtype=jnp.float64)
    ice_mask = jnp.asarray(inputs.glacier_ice_mask, dtype=jnp.float64)
    fraction = jnp.asarray(inputs.glacier_fraction_by_class, dtype=jnp.float64)
    virtual = jnp.asarray(inputs.has_virtual_columns_gridcell, dtype=jnp.bool_)
    number_of_gridcells = ice_mask.shape[0]
    number_of_classes = fraction.shape[1] - 1

    safe_landunit_type = jnp.clip(landunit_type - jnp.int32(1), 0, MAXIMUM_LANDUNIT_TYPE - 1)
    dense_weight = (
        jnp.zeros(
            (number_of_gridcells, MAXIMUM_LANDUNIT_TYPE),
            dtype=jnp.float64,
        )
        .at[landunit_gridcell, safe_landunit_type]
        .set(landunit_weight)
    )
    landunit_assignment_count = (
        jnp.zeros(
            (number_of_gridcells, MAXIMUM_LANDUNIT_TYPE),
            dtype=jnp.int32,
        )
        .at[landunit_gridcell, safe_landunit_type]
        .add(1)
    )

    glacier_area = fortran_ordered_sum(jnp.moveaxis(fraction[:, 1:], -1, 0))
    update_gridcell = jnp.asarray(glc_do_dynglacier) & (ice_mask > jnp.float64(0.0))
    dense_weight = dense_weight.at[:, LANDUNIT_GLACIER - 1].set(
        jnp.where(
            update_gridcell,
            glacier_area,
            dense_weight[:, LANDUNIT_GLACIER - 1],
        )
    )
    dense_weight = reconcile_landunit_weights(dense_weight)
    updated_landunit_weight = dense_weight[landunit_gridcell, safe_landunit_type]

    glacier_column = column_landunit_type == jnp.int32(LANDUNIT_GLACIER)
    elevation_class = column_type - jnp.int32(GLACIER_COLUMN_TYPE_BASE)
    valid_class = (
        glacier_column
        & (elevation_class >= jnp.int32(1))
        & (elevation_class <= jnp.int32(number_of_classes))
    )
    safe_class = jnp.clip(elevation_class, 1, number_of_classes)
    class_assignment_count = (
        jnp.zeros(
            (number_of_gridcells, number_of_classes + 1),
            dtype=jnp.int32,
        )
        .at[column_gridcell, safe_class]
        .add(valid_class.astype(jnp.int32))
    )

    update_column = (
        update_gridcell[column_gridcell]
        & glacier_column
        & valid_class
        & (glacier_area[column_gridcell] > jnp.float64(0.0))
    )
    safe_area = jnp.where(
        update_column,
        glacier_area[column_gridcell],
        jnp.float64(1.0),
    )
    mapped_column_weight = fortran_divide(
        fraction[column_gridcell, safe_class],
        safe_area,
    )
    updated_column_weight_on_landunit = jnp.where(
        update_column,
        mapped_column_weight,
        column_weight_on_landunit,
    )

    updated_column_weight_on_gridcell = fortran_multiply(
        updated_column_weight_on_landunit,
        updated_landunit_weight[column_landunit],
    )
    updated_patch_weight_on_landunit = fortran_multiply(
        patch_weight_on_column,
        updated_column_weight_on_landunit[patch_column],
    )
    updated_patch_weight_on_gridcell = fortran_multiply(
        patch_weight_on_column,
        updated_column_weight_on_gridcell[patch_column],
    )

    urban_landunit = (landunit_type >= jnp.int32(LANDUNIT_URBAN_MIN)) & (
        landunit_type <= jnp.int32(LANDUNIT_URBAN_MAX)
    )
    landunit_active = (
        (updated_landunit_weight > jnp.float64(0.0))
        | (landunit_type == jnp.int32(LANDUNIT_SOIL))
        | ((landunit_type == jnp.int32(LANDUNIT_GLACIER)) & virtual[landunit_gridcell])
        | (urban_landunit & jnp.asarray(run_zero_weight_urban))
    )
    landunit_active = jnp.where(jnp.asarray(all_active), True, landunit_active)

    urban_column = (column_landunit_type >= jnp.int32(LANDUNIT_URBAN_MIN)) & (
        column_landunit_type <= jnp.int32(LANDUNIT_URBAN_MAX)
    )
    column_active = (
        (landunit_active[column_landunit] & (updated_column_weight_on_landunit > 0.0))
        | (glacier_column & virtual[column_gridcell])
        | (landunit_active[column_landunit] & urban_column)
    )
    column_active = jnp.where(jnp.asarray(all_active), True, column_active)
    patch_active = column_active[patch_column] & (patch_weight_on_column > jnp.float64(0.0))
    patch_active = jnp.where(jnp.asarray(all_active), True, patch_active)

    missing_landunit = (landunit_assignment_count == jnp.int32(0)) & (
        dense_weight > jnp.float64(0.0)
    )
    missing_glacier_class = (
        update_gridcell[:, None]
        & (fraction[:, 1:] > jnp.float64(0.0))
        & (class_assignment_count[:, 1:] == jnp.int32(0))
    )
    return GlacierFractionInitializationResult(
        landunit_weight_on_gridcell=updated_landunit_weight,
        landunit_active=landunit_active,
        column_weight_on_gridcell=updated_column_weight_on_gridcell,
        column_weight_on_landunit=updated_column_weight_on_landunit,
        column_active=column_active,
        patch_weight_on_gridcell=updated_patch_weight_on_gridcell,
        patch_weight_on_landunit=updated_patch_weight_on_landunit,
        patch_weight_on_column=patch_weight_on_column,
        patch_active=patch_active,
        landunit_sum_gridcell=_landunit_sum(dense_weight),
        missing_landunit_assignment_gridcell_type=missing_landunit,
        missing_glacier_class_assignment_gridcell_class=missing_glacier_class,
    )


__all__ = [
    "GlacierFractionInitializationInputs",
    "GlacierFractionInitializationResult",
    "LANDUNIT_SUM_TOLERANCE",
    "initialize_glacier_fractions",
    "reconcile_landunit_weights",
]
