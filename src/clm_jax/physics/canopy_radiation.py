
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.io.restart import (
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    SubgridTopology,
)
from clm_jax.physics.numerics import fortran_linear_combination


class CanopySunShadeInputs(NamedTuple):

    exposed_leaf_area_index_patch: jnp.ndarray
    total_leaf_area_by_layer_patch: jnp.ndarray
    sunlit_fraction_by_layer_patch: jnp.ndarray
    radiative_layer_count_patch: jnp.ndarray
    direct_absorption_sunlit_by_layer_patch: jnp.ndarray
    direct_absorption_shaded_by_layer_patch: jnp.ndarray
    diffuse_absorption_sunlit_by_layer_patch: jnp.ndarray
    diffuse_absorption_shaded_by_layer_patch: jnp.ndarray


class CanopySunShadeState(NamedTuple):

    sunlit_leaf_area_patch: jnp.ndarray
    shaded_leaf_area_patch: jnp.ndarray
    sunlit_leaf_area_by_layer_patch: jnp.ndarray
    shaded_leaf_area_by_layer_patch: jnp.ndarray
    sunlit_fraction_patch: jnp.ndarray
    absorbed_par_sunlit_by_layer_patch: jnp.ndarray
    absorbed_par_shaded_by_layer_patch: jnp.ndarray


def _validated_arrays(
    topology: SubgridTopology,
    visible_direct_gridcell: jnp.ndarray,
    visible_diffuse_gridcell: jnp.ndarray,
    inputs: CanopySunShadeInputs,
    state: CanopySunShadeState,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    tuple[jnp.ndarray, ...],
    tuple[jnp.ndarray, ...],
]:

    number_of_patches = topology.patch.gridcell.shape[0]
    number_of_gridcells = topology.grid.longitude_degrees.shape[0]
    visible_direct = jnp.asarray(visible_direct_gridcell, dtype=jnp.float64)
    visible_diffuse = jnp.asarray(visible_diffuse_gridcell, dtype=jnp.float64)
    if visible_direct.shape != (number_of_gridcells,) or visible_diffuse.shape != (
        number_of_gridcells,
    ):
        raise ValueError("visible shortwave forcing must contain one value per gridcell")

    input_arrays = (
        jnp.asarray(inputs.exposed_leaf_area_index_patch, dtype=jnp.float64),
        jnp.asarray(inputs.total_leaf_area_by_layer_patch, dtype=jnp.float64),
        jnp.asarray(inputs.sunlit_fraction_by_layer_patch, dtype=jnp.float64),
        jnp.asarray(inputs.radiative_layer_count_patch, dtype=jnp.int32),
        jnp.asarray(inputs.direct_absorption_sunlit_by_layer_patch, dtype=jnp.float64),
        jnp.asarray(inputs.direct_absorption_shaded_by_layer_patch, dtype=jnp.float64),
        jnp.asarray(inputs.diffuse_absorption_sunlit_by_layer_patch, dtype=jnp.float64),
        jnp.asarray(inputs.diffuse_absorption_shaded_by_layer_patch, dtype=jnp.float64),
    )
    layer_shape = input_arrays[1].shape
    if len(layer_shape) != 2 or layer_shape[0] != number_of_patches:
        raise ValueError("canopy layer inputs must have shape (patch, canopy_layer)")
    if input_arrays[0].shape != (number_of_patches,) or input_arrays[3].shape != (
        number_of_patches,
    ):
        raise ValueError("canopy scalar inputs must contain one value per patch")
    if any(value.shape != layer_shape for value in input_arrays[2:3] + input_arrays[4:]):
        raise ValueError("all canopy layer inputs must have the same shape")

    state_arrays = tuple(jnp.asarray(value, dtype=jnp.float64) for value in state)
    scalar_state = (state_arrays[0], state_arrays[1], state_arrays[4])
    layered_state = state_arrays[2:4] + state_arrays[5:]
    if any(value.shape != (number_of_patches,) for value in scalar_state):
        raise ValueError("canopy scalar state must contain one value per patch")
    if any(value.shape != layer_shape for value in layered_state):
        raise ValueError("canopy layer state must match the input canopy-layer shape")
    return visible_direct, visible_diffuse, input_arrays, state_arrays


def canopy_sunshade_fractions(
    topology: SubgridTopology,
    visible_direct_gridcell: jnp.ndarray,
    visible_diffuse_gridcell: jnp.ndarray,
    inputs: CanopySunShadeInputs,
    state: CanopySunShadeState,
) -> CanopySunShadeState:

    visible_direct, visible_diffuse, input_arrays, state_arrays = _validated_arrays(
        topology,
        visible_direct_gridcell,
        visible_diffuse_gridcell,
        inputs,
        state,
    )
    (
        exposed_leaf_area,
        total_leaf_area_by_layer,
        sunlit_fraction_by_layer,
        radiative_layer_count,
        direct_absorption_sunlit,
        direct_absorption_shaded,
        diffuse_absorption_sunlit,
        diffuse_absorption_shaded,
    ) = input_arrays
    (
        previous_sunlit_leaf_area,
        previous_shaded_leaf_area,
        previous_sunlit_leaf_area_by_layer,
        previous_shaded_leaf_area_by_layer,
        previous_sunlit_fraction,
        previous_absorbed_par_sunlit,
        previous_absorbed_par_shaded,
    ) = state_arrays

    landunit_type = topology.patch.landunit_type
    urban = (landunit_type >= LANDUNIT_URBAN_MIN) & (
        landunit_type <= LANDUNIT_URBAN_MAX
    )
    update_patch = topology.patch.active & ~urban
    number_of_layers = total_leaf_area_by_layer.shape[1]
    layer_number = jnp.arange(number_of_layers, dtype=jnp.int32)
    update_layer = update_patch[:, None] & (
        layer_number[None, :] < radiative_layer_count[:, None]
    )

    sunlit_leaf_area_candidate = total_leaf_area_by_layer * sunlit_fraction_by_layer
    shaded_leaf_area_candidate = total_leaf_area_by_layer * (
        jnp.float64(1.0) - sunlit_fraction_by_layer
    )
    sunlit_leaf_area_by_layer = jnp.where(
        update_layer,
        sunlit_leaf_area_candidate,
        previous_sunlit_leaf_area_by_layer,
    )
    shaded_leaf_area_by_layer = jnp.where(
        update_layer,
        shaded_leaf_area_candidate,
        previous_shaded_leaf_area_by_layer,
    )

    sunlit_leaf_area_candidate_total = jnp.zeros_like(exposed_leaf_area)
    shaded_leaf_area_candidate_total = jnp.zeros_like(exposed_leaf_area)
    for layer_index in range(number_of_layers):
        include_layer = layer_index < radiative_layer_count
        sunlit_leaf_area_candidate_total = sunlit_leaf_area_candidate_total + jnp.where(
            include_layer,
            sunlit_leaf_area_candidate[:, layer_index],
            jnp.float64(0.0),
        )
        shaded_leaf_area_candidate_total = shaded_leaf_area_candidate_total + jnp.where(
            include_layer,
            shaded_leaf_area_candidate[:, layer_index],
            jnp.float64(0.0),
        )

    sunlit_leaf_area = jnp.where(
        update_patch,
        sunlit_leaf_area_candidate_total,
        previous_sunlit_leaf_area,
    )
    shaded_leaf_area = jnp.where(
        update_patch,
        shaded_leaf_area_candidate_total,
        previous_shaded_leaf_area,
    )
    has_exposed_leaf = exposed_leaf_area > jnp.float64(0.0)
    safe_exposed_leaf_area = jnp.where(
        has_exposed_leaf,
        exposed_leaf_area,
        jnp.float64(1.0),
    )
    sunlit_fraction_candidate = jnp.where(
        has_exposed_leaf,
        sunlit_leaf_area_candidate_total / safe_exposed_leaf_area,
        jnp.float64(0.0),
    )
    sunlit_fraction = jnp.where(
        update_patch,
        sunlit_fraction_candidate,
        previous_sunlit_fraction,
    )

    patch_gridcell = topology.patch.gridcell
    visible_direct_patch = visible_direct[patch_gridcell, None]
    visible_diffuse_patch = visible_diffuse[patch_gridcell, None]
    absorbed_par_sunlit_candidate = fortran_linear_combination(
        jnp.stack((visible_direct_patch, visible_diffuse_patch)),
        jnp.stack((direct_absorption_sunlit, diffuse_absorption_sunlit)),
    )
    absorbed_par_shaded_candidate = fortran_linear_combination(
        jnp.stack((visible_direct_patch, visible_diffuse_patch)),
        jnp.stack((direct_absorption_shaded, diffuse_absorption_shaded)),
    )
    absorbed_par_sunlit = jnp.where(
        update_layer,
        absorbed_par_sunlit_candidate,
        previous_absorbed_par_sunlit,
    )
    absorbed_par_shaded = jnp.where(
        update_layer,
        absorbed_par_shaded_candidate,
        previous_absorbed_par_shaded,
    )

    return CanopySunShadeState(
        sunlit_leaf_area_patch=sunlit_leaf_area,
        shaded_leaf_area_patch=shaded_leaf_area,
        sunlit_leaf_area_by_layer_patch=sunlit_leaf_area_by_layer,
        shaded_leaf_area_by_layer_patch=shaded_leaf_area_by_layer,
        sunlit_fraction_patch=sunlit_fraction,
        absorbed_par_sunlit_by_layer_patch=absorbed_par_sunlit,
        absorbed_par_shaded_by_layer_patch=absorbed_par_shaded,
    )
