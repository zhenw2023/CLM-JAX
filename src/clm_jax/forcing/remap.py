
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class SparseMapping(NamedTuple):

    weights: jnp.ndarray
    destination_index: jnp.ndarray
    source_index: jnp.ndarray
    source_size: int
    destination_size: int


def load_sparse_mapping(path: str | Path) -> SparseMapping:

    from clm_jax.io.netcdf import (  
        load_dimension_sizes,
        load_native_variables,
    )

    sizes = load_dimension_sizes(path)
    values = load_native_variables(path, ("S", "row", "col"))
    return SparseMapping(
        weights=jnp.asarray(values["S"], dtype=jnp.float64),
        destination_index=jnp.asarray(values["row"], dtype=jnp.int32) - jnp.int32(1),
        source_index=jnp.asarray(values["col"], dtype=jnp.int32) - jnp.int32(1),
        source_size=sizes["n_a"],
        destination_size=sizes["n_b"],
    )


def build_rectilinear_bilinear_mapping(
    source_longitude_degrees: np.ndarray,
    source_latitude_degrees: np.ndarray,
    destination_longitude_degrees: np.ndarray,
    destination_latitude_degrees: np.ndarray,
) -> SparseMapping:

    source_longitude = np.asarray(source_longitude_degrees, dtype=np.float64)
    source_latitude = np.asarray(source_latitude_degrees, dtype=np.float64)
    destination_longitude = np.asarray(
        destination_longitude_degrees,
        dtype=np.float64,
    )
    destination_latitude = np.asarray(
        destination_latitude_degrees,
        dtype=np.float64,
    )
    if source_longitude.ndim != 1 or source_latitude.ndim != 1:
        raise ValueError("source longitude and latitude must be vectors")
    if destination_longitude.shape != destination_latitude.shape:
        raise ValueError("destination longitude and latitude shapes must match")
    if destination_longitude.ndim != 1:
        raise ValueError("destination longitude and latitude must be vectors")
    if source_longitude.size < 2 or source_latitude.size < 2:
        raise ValueError("bilinear source grid requires at least two points per axis")
    if np.any(np.diff(source_longitude) <= 0.0) or np.any(np.diff(source_latitude) <= 0.0):
        raise ValueError("source axes must be strictly increasing")
    if np.any(np.abs(destination_latitude) > 90.001):
        raise ValueError("destination latitude lies outside the production tolerance")

    longitude_count = source_longitude.size
    latitude_count = source_latitude.size
    source_chunks: list[np.ndarray] = []
    destination_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    one = np.float64(1.0)
    circle = np.float64(360.0)
    north_pole = np.float64(90.0)

    for destination, (longitude, latitude) in enumerate(
        zip(destination_longitude, destination_latitude, strict=True)
    ):
        longitude_outside = longitude < source_longitude[0] or longitude > source_longitude[-1]
        if longitude_outside:
            left = longitude_count - 1
            right = 0
        elif longitude >= source_longitude[-1]:
            left = longitude_count - 2
            right = longitude_count - 1
        else:
            left = int(np.searchsorted(source_longitude, longitude, side="right") - 1)
            right = left + 1

        south = latitude < source_latitude[0]
        north = latitude > source_latitude[-1]
        if south:
            lower = upper = 0
        elif north:
            lower = upper = latitude_count - 1
        elif latitude >= source_latitude[-1]:
            lower = latitude_count - 2
            upper = latitude_count - 1
        else:
            lower = int(np.searchsorted(source_latitude, latitude, side="right") - 1)
            upper = lower + 1

        left_longitude = source_longitude[left]
        right_longitude = source_longitude[right]
        mapped_longitude = longitude
        if longitude_outside:
            left_longitude = np.fmod(left_longitude, circle)
            right_longitude = np.fmod(right_longitude, circle)
            mapped_longitude = np.fmod(mapped_longitude, circle)
            if left_longitude > mapped_longitude:
                left_longitude -= circle
            if right_longitude < mapped_longitude:
                right_longitude += circle

        mapped_latitude = np.clip(latitude, -north_pole, north_pole)
        lower_latitude = source_latitude[lower]
        upper_latitude = source_latitude[upper]
        if south:
            lower_latitude = -north_pole
        elif north:
            upper_latitude = north_pole

        longitude_fraction = (mapped_longitude - left_longitude) / (
            right_longitude - left_longitude
        )
        latitude_fraction = (mapped_latitude - lower_latitude) / (upper_latitude - lower_latitude)
        if not (0.0 <= longitude_fraction <= 1.0 and 0.0 <= latitude_fraction <= 1.0):
            raise ValueError("destination does not lie inside its production map cell")

        if north:
            source_index = np.concatenate(
                (
                    np.asarray([lower * longitude_count + left, lower * longitude_count + right]),
                    upper * longitude_count + np.arange(longitude_count),
                )
            )
            weight = np.concatenate(
                (
                    np.asarray(
                        [
                            (one - longitude_fraction) * (one - latitude_fraction),
                            longitude_fraction * (one - latitude_fraction),
                        ]
                    ),
                    np.full(
                        longitude_count,
                        latitude_fraction / np.float64(longitude_count),
                    ),
                )
            )
        elif south:
            source_index = np.concatenate(
                (
                    np.asarray([upper * longitude_count + left, upper * longitude_count + right]),
                    lower * longitude_count + np.arange(longitude_count),
                )
            )
            weight = np.concatenate(
                (
                    np.asarray(
                        [
                            (one - longitude_fraction) * latitude_fraction,
                            longitude_fraction * latitude_fraction,
                        ]
                    ),
                    np.full(
                        longitude_count,
                        (one - latitude_fraction) / np.float64(longitude_count),
                    ),
                )
            )
        else:
            source_index = np.asarray(
                [
                    lower * longitude_count + left,
                    lower * longitude_count + right,
                    upper * longitude_count + left,
                    upper * longitude_count + right,
                ]
            )
            weight = np.asarray(
                [
                    (one - longitude_fraction) * (one - latitude_fraction),
                    longitude_fraction * (one - latitude_fraction),
                    (one - longitude_fraction) * latitude_fraction,
                    longitude_fraction * latitude_fraction,
                ]
            )
        normalization = np.float64(0.0)
        for value in weight:
            normalization = np.float64(normalization + value)
        weight = weight / normalization
        source_chunks.append(source_index.astype(np.int32, copy=False))
        destination_chunks.append(np.full(source_index.size, destination, dtype=np.int32))
        weight_chunks.append(weight.astype(np.float64, copy=False))

    return SparseMapping(
        weights=jnp.asarray(np.concatenate(weight_chunks)),
        destination_index=jnp.asarray(np.concatenate(destination_chunks)),
        source_index=jnp.asarray(np.concatenate(source_chunks)),
        source_size=latitude_count * longitude_count,
        destination_size=destination_longitude.size,
    )


def apply_sparse_mapping(mapping: SparseMapping, source: jnp.ndarray) -> jnp.ndarray:

    from clm_jax.forcing.ordered_mapping import (
        OrderedMapping, PartitionedMapping, apply_ordered_mapping,
    )

    if isinstance(mapping, PartitionedMapping):
        partials = apply_ordered_mapping(mapping.partials, source)
        return apply_ordered_mapping(mapping.reduction, partials)

    if isinstance(mapping, OrderedMapping):
        return apply_ordered_mapping(mapping, source)

    source = jnp.asarray(source, dtype=jnp.float64)
    if source.shape[-1] != mapping.source_size:
        raise ValueError(
            f"mapping expects source size {mapping.source_size}, got {source.shape[-1]}"
        )
    leading_shape = source.shape[:-1]
    source_2d = source.reshape((-1, mapping.source_size))

    def apply_one(field: jnp.ndarray) -> jnp.ndarray:
        contributions = mapping.weights * field[mapping.source_index]
        return jax.ops.segment_sum(
            contributions,
            mapping.destination_index,
            num_segments=mapping.destination_size,
            indices_are_sorted=False,
        )

    destination = jax.vmap(apply_one)(source_2d)
    return destination.reshape((*leading_shape, mapping.destination_size))


def apply_normalized_sparse_mapping(
    mapping: SparseMapping,
    source: jnp.ndarray,
    source_weight: jnp.ndarray | None = None,
) -> jnp.ndarray:

    from clm_jax.physics.numerics import fortran_divide, fortran_multiply  

    source = jnp.asarray(source, dtype=jnp.float64)
    if source_weight is None:
        weight = jnp.ones_like(source)
    else:
        weight = jnp.broadcast_to(jnp.asarray(source_weight, dtype=jnp.float64), source.shape)
    mapped_weight = apply_sparse_mapping(mapping, weight)
    mapped_source = apply_sparse_mapping(mapping, fortran_multiply(source, weight))
    safe_weight = jnp.where(mapped_weight != 0.0, mapped_weight, 1.0)
    normalized = fortran_multiply(
        mapped_source, fortran_divide(jnp.float64(1.0), safe_weight),
    )
    return jnp.where(mapped_weight != 0.0, normalized, 0.0)


def fill_and_remap(
    source: jnp.ndarray,
    fill_mapping: SparseMapping,
    remapping: SparseMapping,
) -> jnp.ndarray:

    filled = apply_sparse_mapping(fill_mapping, source)
    return apply_sparse_mapping(remapping, filled)


def temporal_interpolate(
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    lower_weight: jnp.ndarray | float,
) -> jnp.ndarray:

    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)
    lower_weight = jnp.asarray(lower_weight, dtype=jnp.float64)
    upper_weight = jnp.float64(1.0) - lower_weight
    return lower * lower_weight + upper * upper_weight
