
from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.forcing.remap import SparseMapping, apply_normalized_sparse_mapping
from clm_jax.io.coupler import GlacierCouplerState
from clm_jax.io.restart import (
    LANDUNIT_GLACIER,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    ColumnTopology,
)

DEFAULT_GLACIER_ELEVATION_BOUNDS = (
    0.0,
    200.0,
    400.0,
    700.0,
    1000.0,
    1300.0,
    1600.0,
    2000.0,
    2500.0,
    3000.0,
    10000.0,
)
GLACIER_COLUMN_TYPE_BASE = LANDUNIT_GLACIER * 100
SINGLE_AT_ATMOSPHERIC_TOPOGRAPHY = "single_at_atm_topo"


class GlacierLandFields(NamedTuple):

    ice_mask: jnp.ndarray
    fraction_by_class: jnp.ndarray
    elevation_by_class: jnp.ndarray


class ColumnTopography(NamedTuple):

    elevation: jnp.ndarray
    needs_downscaling: jnp.ndarray
    column_type: jnp.ndarray


def glacier_elevation_class(
    elevation: jnp.ndarray | float,
    bounds: Sequence[float] = DEFAULT_GLACIER_ELEVATION_BOUNDS,
) -> jnp.ndarray:

    boundaries = jnp.asarray(bounds, dtype=jnp.float64)
    if boundaries.ndim != 1 or boundaries.shape[0] < 2:
        raise ValueError("glacier elevation bounds must contain at least two values")
    elevation = jnp.asarray(elevation, dtype=jnp.float64)
    elevation_class = jnp.searchsorted(boundaries, elevation, side="right")
    return jnp.clip(elevation_class, 1, boundaries.shape[0] - 1).astype(jnp.int32)


def virtual_elevation_by_class(
    bounds: Sequence[float] = DEFAULT_GLACIER_ELEVATION_BOUNDS,
) -> jnp.ndarray:

    boundaries = jnp.asarray(bounds, dtype=jnp.float64)
    if boundaries.ndim != 1 or boundaries.shape[0] < 3:
        raise ValueError("at least two glacier elevation classes are required")
    middle = (boundaries[:-2] + boundaries[1:-1]) / jnp.float64(2.0)
    highest = jnp.float64(2.0) * boundaries[-2] - boundaries[-3]
    return jnp.concatenate((jnp.zeros(1, dtype=jnp.float64), middle, highest[None]))


def map_glacier_state_to_land(
    state: GlacierCouplerState,
    mapping: SparseMapping,
    bounds: Sequence[float] = DEFAULT_GLACIER_ELEVATION_BOUNDS,
) -> GlacierLandFields:

    if state.elevation.shape != (mapping.source_size,):
        raise ValueError("coupler elevation size does not match mapping source grid")
    if state.ice_fraction.shape != state.elevation.shape:
        raise ValueError("coupler ice fraction and elevation must have identical shapes")
    if state.ice_mask.shape != state.elevation.shape:
        raise ValueError("coupler ice mask and elevation must have identical shapes")

    number_of_classes = len(bounds) - 1
    source_class = glacier_elevation_class(state.elevation, bounds)
    class_ids = jnp.arange(1, number_of_classes + 1, dtype=jnp.int32)[:, None]
    glacier_fraction = jnp.where(
        source_class[None, :] == class_ids,
        state.ice_fraction[None, :],
        jnp.float64(0.0),
    )
    source_fraction = jnp.concatenate(
        ((jnp.float64(1.0) - state.ice_fraction)[None, :], glacier_fraction), axis=0
    )

    destination_fraction = apply_normalized_sparse_mapping(
        mapping,
        source_fraction,
        state.ice_mask,
    )
    source_elevation = jnp.broadcast_to(state.elevation, source_fraction.shape)
    destination_elevation = apply_normalized_sparse_mapping(
        mapping,
        source_elevation,
        source_fraction * state.ice_mask[None, :],
    )
    virtual_elevation = virtual_elevation_by_class(bounds)[:, None]
    destination_elevation = jnp.where(
        destination_fraction <= jnp.float64(0.0),
        virtual_elevation,
        destination_elevation,
    )
    destination_ice_mask = apply_normalized_sparse_mapping(mapping, state.ice_mask)
    return GlacierLandFields(
        ice_mask=destination_ice_mask,
        fraction_by_class=jnp.swapaxes(destination_fraction, 0, 1),
        elevation_by_class=jnp.swapaxes(destination_elevation, 0, 1),
    )


def select_land_gridcells(
    fields: GlacierLandFields,
    destination_index: jnp.ndarray,
) -> GlacierLandFields:

    destination_index = jnp.asarray(destination_index, dtype=jnp.int32)
    return GlacierLandFields(
        ice_mask=fields.ice_mask[destination_index],
        fraction_by_class=fields.fraction_by_class[destination_index],
        elevation_by_class=fields.elevation_by_class[destination_index],
    )


def collapse_mask_from_region_behavior(
    glacier_region: jnp.ndarray,
    region_behavior: Sequence[str],
) -> jnp.ndarray:

    collapse_by_region = jnp.asarray(
        [behavior.strip() == SINGLE_AT_ATMOSPHERIC_TOPOGRAPHY for behavior in region_behavior],
        dtype=jnp.bool_,
    )
    region = jnp.asarray(glacier_region, dtype=jnp.int32)
    if collapse_by_region.shape[0] == 0:
        raise ValueError("glacier region behavior must not be empty")
    return collapse_by_region[region]


def update_column_topography(
    initial_column_elevation: jnp.ndarray,
    atmospheric_elevation: jnp.ndarray,
    column: ColumnTopology,
    glacier: GlacierLandFields,
    collapse_to_atmospheric_topography: jnp.ndarray,
    *,
    glc_do_dynglacier: bool,
    bounds: Sequence[float] = DEFAULT_GLACIER_ELEVATION_BOUNDS,
) -> ColumnTopography:

    initial_column_elevation = jnp.asarray(initial_column_elevation, dtype=jnp.float64)
    atmospheric_elevation = jnp.asarray(atmospheric_elevation, dtype=jnp.float64)
    collapse = jnp.asarray(collapse_to_atmospheric_topography, dtype=jnp.bool_)
    if initial_column_elevation.shape != column.gridcell.shape:
        raise ValueError("initial elevation must contain one value per CLM column")
    if glacier.ice_mask.shape != atmospheric_elevation.shape:
        raise ValueError("mapped glacier fields and atmospheric elevation must share a grid")
    if collapse.shape != atmospheric_elevation.shape:
        raise ValueError("collapse mask and atmospheric elevation must share a grid")

    gridcell = column.gridcell
    is_glacier = column.landunit_type == LANDUNIT_GLACIER
    is_active_glacier = is_glacier & column.active
    collapse_column = collapse[gridcell]
    needs_downscaling = is_active_glacier & ~collapse_column
    elevation = initial_column_elevation

    if glc_do_dynglacier:
        is_urban = (column.landunit_type >= LANDUNIT_URBAN_MIN) & (
            column.landunit_type <= LANDUNIT_URBAN_MAX
        )
        dynamic_update = (glacier.ice_mask[gridcell] > jnp.float64(0.0)) & ~is_urban
        elevation_class = jnp.where(
            is_glacier,
            column.column_type - jnp.int32(GLACIER_COLUMN_TYPE_BASE),
            jnp.int32(0),
        )
        mapped_elevation = glacier.elevation_by_class[gridcell, elevation_class]
        elevation = jnp.where(dynamic_update, mapped_elevation, elevation)
        needs_downscaling = needs_downscaling | dynamic_update

    elevation = jnp.where(
        needs_downscaling,
        elevation,
        atmospheric_elevation[gridcell],
    )
    updated_glacier_type = (
        jnp.int32(GLACIER_COLUMN_TYPE_BASE) + glacier_elevation_class(elevation, bounds)
    )
    column_type = jnp.where(
        is_glacier & collapse_column,
        updated_glacier_type,
        column.column_type,
    )
    return ColumnTopography(elevation, needs_downscaling, column_type)
