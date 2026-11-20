
from __future__ import annotations

from collections.abc import Callable
from functools import cache, partial, wraps
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from clm_jax.compilation import jit_unique, process_lifetime_jit
from clm_jax.execution import _distributed_vmap, _reshard_tree_for_pmap
from clm_jax.execution_observer import stage_barrier
from clm_jax.physics.snow_snicar import MINIMUM_SNOW_WATER_EQUIVALENT, SnicarRuntimeResult
from clm_jax.physics.surface_albedo import (
    NUMBER_OF_SNOW_ABSORPTION_LEVELS,
    CanopySurfaceAlbedoDiagnostics,
    CanopySurfaceAlbedoInputs,
    CanopySurfaceAlbedoResult,
    CanopySurfaceAlbedoState,
    GroundSnowAlbedoDiagnostics,
    GroundSnowAlbedoInputs,
    GroundSnowAlbedoState,
    SoilAlbedoDiagnostics,
    SoilAlbedoInputs,
    SoilAlbedoState,
    SurfaceAlbedoColumnInputs,
    SurfaceAlbedoColumnParameters,
    SurfaceAlbedoColumnState,
    SurfaceAlbedoInputs,
    SurfaceAlbedoParameters,
    SurfaceAlbedoResult,
    SurfaceAlbedoState,
    canopy_surface_albedo_big_leaf,
    soil_albedo,
    surface_albedo_canopy_inputs,
    surface_albedo_column_inputs,
    surface_albedo_column_parameters,
    surface_albedo_column_state,
    surface_albedo_columns_configured,
    surface_albedo_configured,
    weight_ground_snow_albedo,
)
from clm_jax.physics.surface_albedo_composition import SurfaceAlbedoCall
from clm_jax.runtime import co_locate_array_leaves


class SurfaceAlbedoShardPlan(NamedTuple):

    column_count: int
    patch_count: int
    device_count: int
    columns_per_device: int
    column_padding: int
    patches_per_device: int
    patch_indices: np.ndarray
    valid_patch_slots: np.ndarray


class SurfaceAlbedoRuntimeSnowDiagnostics(NamedTuple):
    fatal_retry_failure_count: jax.Array
    energy_conservation_failure_count: jax.Array
    invalid_active_snow_grain_radius_count: jax.Array
    maximum_attempt_count: jax.Array
    maximum_absolute_energy_residual: jax.Array


class SurfaceAlbedoRuntimeDiagnostics(NamedTuple):

    soil: SoilAlbedoDiagnostics
    direct_snow: SurfaceAlbedoRuntimeSnowDiagnostics
    diffuse_snow: SurfaceAlbedoRuntimeSnowDiagnostics
    ground: GroundSnowAlbedoDiagnostics
    canopy: CanopySurfaceAlbedoDiagnostics


class SurfaceAlbedoRuntimeResult(NamedTuple):

    state: SurfaceAlbedoState
    diagnostics: SurfaceAlbedoRuntimeDiagnostics
    all_finite: jax.Array | bool = True


class EqualAxisShardPlan(NamedTuple):
    item_count: int
    device_count: int
    items_per_device: int
    padding: int
    item_indices: np.ndarray | None = None
    valid_slots: np.ndarray | None = None


class SurfaceAlbedoColumnRuntimeDiagnostics(NamedTuple):
    soil: SoilAlbedoDiagnostics
    direct_snow: SurfaceAlbedoRuntimeSnowDiagnostics
    diffuse_snow: SurfaceAlbedoRuntimeSnowDiagnostics
    ground: GroundSnowAlbedoDiagnostics


class SurfaceAlbedoColumnRuntimeResult(NamedTuple):
    state: SurfaceAlbedoColumnState
    diagnostics: SurfaceAlbedoColumnRuntimeDiagnostics


_COLUMN_INPUT_FIELDS = {
    "nonurban_column",
    "column_gridcell",
    "landunit_type_column",
    "snow_layer_count_column",
    "ground_temperature_column",
    "top_soil_water_volume_fraction_column",
    "lake_ice_fraction_top_two_column",
    "soil_color_column",
    "snow_fraction_column",
    "snow_water_equivalent_column",
    "snow_liquid_column_layer",
    "snow_ice_column_layer",
    "snow_grain_radius_column_layer",
    "aerosol_mass_concentration_column_layer_species",
}
_PATCH_INPUT_FIELDS = {
    "nonurban_patch",
    "patch_gridcell",
    "patch_column",
    "pft_type_patch",
    "vegetation_temperature_patch",
    "wet_fraction_patch",
    "canopy_snow_fraction_patch",
    "exposed_leaf_area_patch",
    "exposed_stem_area_patch",
}
_PFT_INPUT_FIELDS = {
    "leaf_orientation_pft",
    "leaf_reflectance_pft_band",
    "stem_reflectance_pft_band",
    "leaf_transmittance_pft_band",
    "stem_transmittance_pft_band",
}
_GRIDCELL_INPUT_FIELDS = {"cosine_solar_zenith_gridcell"}
_ALL_INPUT_FIELDS = (
    _COLUMN_INPUT_FIELDS
    | _PATCH_INPUT_FIELDS
    | _PFT_INPUT_FIELDS
    | _GRIDCELL_INPUT_FIELDS
)
if _ALL_INPUT_FIELDS != set(SurfaceAlbedoInputs._fields):
    raise AssertionError("SurfaceAlbedo sharding fields do not cover the public input")

_SURFACE_ALBEDO_INPUT_AXES = SurfaceAlbedoInputs(
    *(
        None if name in _PFT_INPUT_FIELDS else 0
        for name in SurfaceAlbedoInputs._fields
    )
)

_SURFACE_ALBEDO_COLUMN_INPUT_AXES = SurfaceAlbedoColumnInputs(
    *(0 for _ in SurfaceAlbedoColumnInputs._fields)
)
_CANOPY_PFT_INPUT_FIELDS = {
    "leaf_orientation_pft",
    "leaf_reflectance_pft_band",
    "stem_reflectance_pft_band",
    "leaf_transmittance_pft_band",
    "stem_transmittance_pft_band",
}
_SURFACE_ALBEDO_CANOPY_INPUT_AXES = CanopySurfaceAlbedoInputs(
    *(
        None if name in _CANOPY_PFT_INPUT_FIELDS else 0
        for name in CanopySurfaceAlbedoInputs._fields
    )
)

_MINIMUM_SELECTED_ITEMS_PER_DEVICE = 64


def _plan_equal_axis(
    item_count: int,
    device_count: int,
) -> EqualAxisShardPlan:
    if item_count < 1 or device_count < 1:
        raise ValueError("equal-axis sharding requires positive sizes")
    items_per_device = (item_count + device_count - 1) // device_count
    return EqualAxisShardPlan(
        item_count=item_count,
        device_count=device_count,
        items_per_device=items_per_device,
        padding=items_per_device * device_count - item_count,
    )


def _plan_selected_axis(
    selection_mask: np.ndarray,
    device_count: int,
) -> EqualAxisShardPlan | None:

    if device_count < 1:
        raise ValueError("selected-axis sharding requires a positive device count")
    selection = np.asarray(selection_mask, dtype=np.bool_)
    if selection.ndim != 1:
        raise ValueError(
            f"selected-axis mask must be a vector, got {selection.shape}"
        )
    selected = np.flatnonzero(selection).astype(np.int32, copy=False)
    if selected.size == 0:
        return None
    required_per_device = (
        int(selected.size) + device_count - 1
    ) // device_count
    items_per_device = max(
        _MINIMUM_SELECTED_ITEMS_PER_DEVICE,
        1 << (required_per_device - 1).bit_length(),
    )
    indices = np.zeros((device_count, items_per_device), dtype=np.int32)
    valid = np.zeros_like(indices, dtype=np.bool_)
    for device, owned in enumerate(np.array_split(selected, device_count)):
        count = int(owned.size)
        if count:
            indices[device, :count] = owned
            valid[device, :count] = True
    return EqualAxisShardPlan(
        item_count=int(selection.size),
        device_count=device_count,
        items_per_device=items_per_device,
        padding=items_per_device * device_count - int(selected.size),
        item_indices=indices,
        valid_slots=valid,
    )


def _pack_equal_axis(value, plan: EqualAxisShardPlan) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim < 1 or array.shape[0] != plan.item_count:
        raise ValueError(
            f"equal-axis leaf must start with {plan.item_count}, got {array.shape}"
        )
    if plan.item_indices is not None:
        return array[jnp.asarray(plan.item_indices, dtype=jnp.int32)]
    if plan.padding:
        array = jnp.concatenate(
            (array, jnp.repeat(array[-1:], plan.padding, axis=0)),
            axis=0,
        )
    return jnp.reshape(
        array,
        (plan.device_count, plan.items_per_device, *array.shape[1:]),
    )


def _unpack_equal_axis(value, plan: EqualAxisShardPlan) -> jax.Array:
    array = jnp.asarray(value)
    expected = (plan.device_count, plan.items_per_device)
    if array.ndim < 2 or array.shape[:2] != expected:
        raise ValueError(
            f"mapped equal-axis leaf must start with {expected}, got {array.shape}"
        )
    flattened = jnp.reshape(
        array,
        (plan.device_count * plan.items_per_device, *array.shape[2:]),
    )
    if plan.item_indices is None:
        return flattened[: plan.item_count]
    if plan.valid_slots is None:
        raise AssertionError("reordered equal-axis plan requires valid slots")
    valid = jnp.asarray(plan.valid_slots.reshape(-1), dtype=jnp.bool_)
    index = jnp.asarray(plan.item_indices.reshape(-1), dtype=jnp.int32)
    output = jnp.zeros((plan.item_count, *array.shape[2:]), dtype=array.dtype)
    index = jnp.where(valid, index, jnp.int32(plan.item_count))
    return output.at[index].set(flattened, mode="drop")


def plan_surface_albedo_shards(
    column_count: int,
    patch_column,
    device_count: int,
) -> SurfaceAlbedoShardPlan:

    if isinstance(column_count, bool) or not isinstance(column_count, int):
        raise TypeError("SurfaceAlbedo column_count must be an integer")
    if isinstance(device_count, bool) or not isinstance(device_count, int):
        raise TypeError("SurfaceAlbedo device_count must be an integer")
    if column_count < 1:
        raise ValueError("SurfaceAlbedo requires at least one column")
    if device_count < 1:
        raise ValueError("SurfaceAlbedo device_count must be positive")
    patch_column_array = np.asarray(patch_column, dtype=np.int64)
    if patch_column_array.ndim != 1 or patch_column_array.size < 1:
        raise ValueError("SurfaceAlbedo patch_column must be a nonempty vector")
    if np.any((patch_column_array < 0) | (patch_column_array >= column_count)):
        raise ValueError("SurfaceAlbedo patch_column is out of bounds")

    columns_per_device = (column_count + device_count - 1) // device_count
    owner = patch_column_array // columns_per_device
    owned_patch = [
        np.flatnonzero(owner == device).astype(np.int32, copy=False)
        for device in range(device_count)
    ]
    patches_per_device = max(1, *(indices.size for indices in owned_patch))
    patch_indices = np.zeros((device_count, patches_per_device), dtype=np.int32)
    valid_patch_slots = np.zeros_like(patch_indices, dtype=np.bool_)
    for device, indices in enumerate(owned_patch):
        count = int(indices.size)
        patch_indices[device, :count] = indices
        valid_patch_slots[device, :count] = True
    padded_column_count = columns_per_device * device_count
    return SurfaceAlbedoShardPlan(
        column_count=column_count,
        patch_count=int(patch_column_array.size),
        device_count=device_count,
        columns_per_device=columns_per_device,
        column_padding=padded_column_count - column_count,
        patches_per_device=patches_per_device,
        patch_indices=patch_indices,
        valid_patch_slots=valid_patch_slots,
    )


def _pack_column(value, plan: SurfaceAlbedoShardPlan) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim < 1 or array.shape[0] != plan.column_count:
        raise ValueError(
            "SurfaceAlbedo column leaf must start with "
            f"{plan.column_count}, got {array.shape}"
        )
    if plan.column_padding:
        array = jnp.concatenate(
            (array, jnp.repeat(array[-1:], plan.column_padding, axis=0)),
            axis=0,
        )
    return jnp.reshape(
        array,
        (plan.device_count, plan.columns_per_device, *array.shape[1:]),
    )


def _unpack_column(value, plan: SurfaceAlbedoShardPlan) -> jax.Array:
    array = jnp.asarray(value)
    expected = (plan.device_count, plan.columns_per_device)
    if array.ndim < 2 or array.shape[:2] != expected:
        raise ValueError(
            "mapped SurfaceAlbedo column leaf must start with "
            f"{expected}, got {array.shape}"
        )
    flattened = jnp.reshape(
        array,
        (plan.device_count * plan.columns_per_device, *array.shape[2:]),
    )
    return flattened[: plan.column_count]


def _pack_patch(value, plan: SurfaceAlbedoShardPlan) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim < 1 or array.shape[0] != plan.patch_count:
        raise ValueError(
            "SurfaceAlbedo patch leaf must start with "
            f"{plan.patch_count}, got {array.shape}"
        )
    return array[jnp.asarray(plan.patch_indices, dtype=jnp.int32)]


def _unpack_patch(value, plan: SurfaceAlbedoShardPlan) -> jax.Array:
    array = jnp.asarray(value)
    expected = (plan.device_count, plan.patches_per_device)
    if array.ndim < 2 or array.shape[:2] != expected:
        raise ValueError(
            "mapped SurfaceAlbedo patch leaf must start with "
            f"{expected}, got {array.shape}"
        )
    flattened = jnp.reshape(
        array,
        (plan.device_count * plan.patches_per_device, *array.shape[2:]),
    )
    valid = jnp.asarray(plan.valid_patch_slots.reshape(-1), dtype=jnp.bool_)
    index = jnp.asarray(plan.patch_indices.reshape(-1), dtype=jnp.int32)
    output = jnp.zeros((plan.patch_count, *array.shape[2:]), dtype=array.dtype)
    return output.at[index[valid]].set(flattened[valid])


def pack_surface_albedo_inputs(
    inputs: SurfaceAlbedoInputs,
    plan: SurfaceAlbedoShardPlan,
) -> SurfaceAlbedoInputs:

    packed_values = []
    for name, value in zip(SurfaceAlbedoInputs._fields, inputs, strict=True):
        if name in _PFT_INPUT_FIELDS:
            packed_values.append(value)
        elif name == "cosine_solar_zenith_gridcell":
            column_cosine = jnp.asarray(value)[
                jnp.asarray(inputs.column_gridcell, dtype=jnp.int32)
            ]
            packed_values.append(_pack_column(column_cosine, plan))
        elif name in _COLUMN_INPUT_FIELDS:
            packed_values.append(_pack_column(value, plan))
        elif name in _PATCH_INPUT_FIELDS:
            packed_values.append(_pack_patch(value, plan))
        else:  
            raise AssertionError(f"unclassified SurfaceAlbedo input {name}")
    packed = SurfaceAlbedoInputs(*packed_values)

    padded_column_count = plan.device_count * plan.columns_per_device
    valid_column = jnp.reshape(
        jnp.arange(padded_column_count, dtype=jnp.int32)
        < jnp.int32(plan.column_count),
        (plan.device_count, plan.columns_per_device),
    )
    valid_patch = jnp.asarray(plan.valid_patch_slots, dtype=jnp.bool_)
    column_start = (
        jnp.arange(plan.device_count, dtype=jnp.int32)
        * jnp.int32(plan.columns_per_device)
    )[:, None]
    local_patch_column = packed.patch_column - column_start
    local_patch_column = jnp.where(valid_patch, local_patch_column, jnp.int32(0))
    local_column = jnp.broadcast_to(
        jnp.arange(plan.columns_per_device, dtype=jnp.int32)[None, :],
        (plan.device_count, plan.columns_per_device),
    )
    return packed._replace(
        nonurban_column=packed.nonurban_column & valid_column,
        nonurban_patch=packed.nonurban_patch & valid_patch,
        column_gridcell=local_column,
        patch_column=local_patch_column,
        patch_gridcell=local_patch_column,
    )


def pack_surface_albedo_state(
    state: SurfaceAlbedoState,
    plan: SurfaceAlbedoShardPlan,
) -> SurfaceAlbedoState:

    column_tree = lambda tree: jax.tree_util.tree_map(  
        lambda value: _pack_column(value, plan),
        tree,
    )
    patch_tree = lambda tree: jax.tree_util.tree_map(  
        lambda value: _pack_patch(value, plan),
        tree,
    )
    return SurfaceAlbedoState(
        cosine_solar_zenith_column=_pack_column(
            state.cosine_solar_zenith_column,
            plan,
        ),
        soil=column_tree(state.soil),
        ground=column_tree(state.ground),
        canopy=patch_tree(state.canopy),
        pure_snow_ground_direct_albedo_column_band=_pack_column(
            state.pure_snow_ground_direct_albedo_column_band,
            plan,
        ),
        pure_snow_ground_diffuse_albedo_column_band=_pack_column(
            state.pure_snow_ground_diffuse_albedo_column_band,
            plan,
        ),
        no_black_carbon_ground_direct_albedo_column_band=_pack_column(
            state.no_black_carbon_ground_direct_albedo_column_band,
            plan,
        ),
        no_black_carbon_ground_diffuse_albedo_column_band=_pack_column(
            state.no_black_carbon_ground_diffuse_albedo_column_band,
            plan,
        ),
        no_organic_carbon_ground_direct_albedo_column_band=_pack_column(
            state.no_organic_carbon_ground_direct_albedo_column_band,
            plan,
        ),
        no_organic_carbon_ground_diffuse_albedo_column_band=_pack_column(
            state.no_organic_carbon_ground_diffuse_albedo_column_band,
            plan,
        ),
        no_dust_ground_direct_albedo_column_band=_pack_column(
            state.no_dust_ground_direct_albedo_column_band,
            plan,
        ),
        no_dust_ground_diffuse_albedo_column_band=_pack_column(
            state.no_dust_ground_diffuse_albedo_column_band,
            plan,
        ),
    )


def unpack_surface_albedo_state(
    mapped: SurfaceAlbedoState,
    plan: SurfaceAlbedoShardPlan,
) -> SurfaceAlbedoState:

    column_tree = lambda tree: jax.tree_util.tree_map(  
        lambda value: _unpack_column(value, plan),
        tree,
    )
    patch_tree = lambda tree: jax.tree_util.tree_map(  
        lambda value: _unpack_patch(value, plan),
        tree,
    )
    return SurfaceAlbedoState(
        cosine_solar_zenith_column=_unpack_column(
            mapped.cosine_solar_zenith_column,
            plan,
        ),
        soil=column_tree(mapped.soil),
        ground=column_tree(mapped.ground),
        canopy=patch_tree(mapped.canopy),
        pure_snow_ground_direct_albedo_column_band=_unpack_column(
            mapped.pure_snow_ground_direct_albedo_column_band,
            plan,
        ),
        pure_snow_ground_diffuse_albedo_column_band=_unpack_column(
            mapped.pure_snow_ground_diffuse_albedo_column_band,
            plan,
        ),
        no_black_carbon_ground_direct_albedo_column_band=_unpack_column(
            mapped.no_black_carbon_ground_direct_albedo_column_band,
            plan,
        ),
        no_black_carbon_ground_diffuse_albedo_column_band=_unpack_column(
            mapped.no_black_carbon_ground_diffuse_albedo_column_band,
            plan,
        ),
        no_organic_carbon_ground_direct_albedo_column_band=_unpack_column(
            mapped.no_organic_carbon_ground_direct_albedo_column_band,
            plan,
        ),
        no_organic_carbon_ground_diffuse_albedo_column_band=_unpack_column(
            mapped.no_organic_carbon_ground_diffuse_albedo_column_band,
            plan,
        ),
        no_dust_ground_direct_albedo_column_band=_unpack_column(
            mapped.no_dust_ground_direct_albedo_column_band,
            plan,
        ),
        no_dust_ground_diffuse_albedo_column_band=_unpack_column(
            mapped.no_dust_ground_diffuse_albedo_column_band,
            plan,
        ),
    )


def _snow_runtime_diagnostics(result) -> SurfaceAlbedoRuntimeSnowDiagnostics:
    if isinstance(result, SnicarRuntimeResult):
        return SurfaceAlbedoRuntimeSnowDiagnostics(*result.diagnostics)
    diagnostics = result.diagnostics
    invalid_radius = result.optical.diagnostics.invalid_active_snow_grain_radius_column_layer
    return SurfaceAlbedoRuntimeSnowDiagnostics(
        fatal_retry_failure_count=jnp.sum(
            diagnostics.fatal_retry_failure_column_band,
            dtype=jnp.int32,
        ),
        energy_conservation_failure_count=jnp.sum(
            diagnostics.energy_conservation_failure_column_band,
            dtype=jnp.int32,
        ),
        invalid_active_snow_grain_radius_count=jnp.sum(
            invalid_radius,
            dtype=jnp.int32,
        ),
        maximum_attempt_count=jnp.max(diagnostics.attempt_count_column_band),
        maximum_absolute_energy_residual=jnp.max(
            jnp.abs(diagnostics.energy_residual_column_band)
        ),
    )


def surface_albedo_runtime_step(
    inputs: SurfaceAlbedoInputs,
    state: SurfaceAlbedoState,
    parameters: SurfaceAlbedoParameters,
) -> SurfaceAlbedoRuntimeResult:

    return _project_surface_albedo_runtime(surface_albedo_configured(inputs, state, parameters))


def _project_surface_albedo_runtime(result, *, check_dense_fields=False):
    finite = jnp.asarray(True)
    if check_dense_fields:
        for snow in (result.direct_snow, result.diffuse_snow):
            if isinstance(snow, SnicarRuntimeResult):
                finite = finite & snow.all_finite
        seen = set()
        for value in jax.tree.leaves(result):
            if id(value) in seen:
                continue
            seen.add(id(value))
            if jnp.issubdtype(value.dtype, jnp.inexact):
                finite = finite & jnp.all(jnp.isfinite(value))
    return SurfaceAlbedoRuntimeResult(
        state=result.state,
        diagnostics=SurfaceAlbedoRuntimeDiagnostics(
            soil=result.diagnostics.soil,
            direct_snow=_snow_runtime_diagnostics(result.direct_snow),
            diffuse_snow=_snow_runtime_diagnostics(result.diffuse_snow),
            ground=result.diagnostics.ground,
            canopy=result.diagnostics.canopy,
        ),
        all_finite=finite,
    )


def surface_albedo_column_runtime_step(
    inputs: SurfaceAlbedoColumnInputs,
    state: SurfaceAlbedoColumnState,
    parameters: SurfaceAlbedoColumnParameters,
) -> SurfaceAlbedoColumnRuntimeResult:

    result = surface_albedo_columns_configured(
        inputs,
        state,
        parameters,
        early_exit_snicar_retry_loop=True,
    )
    return SurfaceAlbedoColumnRuntimeResult(
        state=result.state,
        diagnostics=SurfaceAlbedoColumnRuntimeDiagnostics(
            soil=result.diagnostics.soil,
            direct_snow=_snow_runtime_diagnostics(result.direct_snow),
            diffuse_snow=_snow_runtime_diagnostics(result.diffuse_snow),
            ground=result.diagnostics.ground,
        ),
    )


def surface_albedo_unqualified_column_runtime_step(
    inputs: SurfaceAlbedoColumnInputs,
    state: SurfaceAlbedoColumnState,
    parameters: SurfaceAlbedoColumnParameters,
) -> SurfaceAlbedoColumnRuntimeResult:

    zero = jnp.float64(0.0)
    nonurban = jnp.asarray(inputs.nonurban_column, dtype=jnp.bool_)
    cosine = jnp.asarray(inputs.cosine_solar_zenith_column, dtype=jnp.float64)

    def initialize_column_band(value: jax.Array) -> jax.Array:
        return jnp.where(
            nonurban[:, None],
            zero,
            jnp.asarray(value, dtype=jnp.float64),
        )

    soil = soil_albedo(
        SoilAlbedoInputs(
            nonurban_column=inputs.nonurban_column,
            landunit_type_column=inputs.landunit_type_column,
            cosine_solar_zenith_column=cosine,
            snow_layer_count_column=inputs.snow_layer_count_column,
            ground_temperature_column=inputs.ground_temperature_column,
            top_soil_water_volume_fraction_column=(
                inputs.top_soil_water_volume_fraction_column
            ),
            lake_ice_fraction_top_two_column=(
                inputs.lake_ice_fraction_top_two_column
            ),
            soil_color_column=inputs.soil_color_column,
        ),
        SoilAlbedoState(
            direct_column_band=initialize_column_band(
                state.soil.direct_column_band
            ),
            diffuse_column_band=initialize_column_band(
                state.soil.diffuse_column_band
            ),
        ),
        parameters.soil,
    )

    snow_water = jnp.asarray(
        inputs.snow_water_equivalent_column,
        dtype=jnp.float64,
    )
    tiny_snow = (
        nonurban
        & (cosine > zero)
        & (snow_water < jnp.float64(MINIMUM_SNOW_WATER_EQUIVALENT))
        & (snow_water > zero)
    )
    snow_albedo = jnp.where(
        tiny_snow[:, None],
        soil.state.diffuse_column_band,
        zero,
    )
    snow_absorption = jnp.zeros(
        (
            nonurban.shape[0],
            NUMBER_OF_SNOW_ABSORPTION_LEVELS,
            2,
        ),
        dtype=jnp.float64,
    )
    initialized_ground = GroundSnowAlbedoState(
        ground_direct_albedo_column_band=initialize_column_band(
            state.ground.ground_direct_albedo_column_band
        ),
        ground_diffuse_albedo_column_band=initialize_column_band(
            state.ground.ground_diffuse_albedo_column_band
        ),
        snow_direct_albedo_history_column_band=(
            state.ground.snow_direct_albedo_history_column_band
        ),
        snow_diffuse_albedo_history_column_band=(
            state.ground.snow_diffuse_albedo_history_column_band
        ),
        direct_visible_absorption_column_level=jnp.where(
            nonurban[:, None],
            zero,
            jnp.asarray(
                state.ground.direct_visible_absorption_column_level,
                dtype=jnp.float64,
            ),
        ),
        direct_nir_absorption_column_level=jnp.where(
            nonurban[:, None],
            zero,
            jnp.asarray(
                state.ground.direct_nir_absorption_column_level,
                dtype=jnp.float64,
            ),
        ),
        diffuse_visible_absorption_column_level=jnp.where(
            nonurban[:, None],
            zero,
            jnp.asarray(
                state.ground.diffuse_visible_absorption_column_level,
                dtype=jnp.float64,
            ),
        ),
        diffuse_nir_absorption_column_level=jnp.where(
            nonurban[:, None],
            zero,
            jnp.asarray(
                state.ground.diffuse_nir_absorption_column_level,
                dtype=jnp.float64,
            ),
        ),
    )
    ground = weight_ground_snow_albedo(
        GroundSnowAlbedoInputs(
            nonurban_column=inputs.nonurban_column,
            landunit_type_column=inputs.landunit_type_column,
            cosine_solar_zenith_column=cosine,
            snow_fraction_column=inputs.snow_fraction_column,
            snow_water_equivalent_column=inputs.snow_water_equivalent_column,
            soil_direct_albedo_column_band=soil.state.direct_column_band,
            soil_diffuse_albedo_column_band=soil.state.diffuse_column_band,
            snow_direct_albedo_column_band=snow_albedo,
            snow_diffuse_albedo_column_band=snow_albedo,
            snow_direct_absorption_column_level_band=snow_absorption,
            snow_diffuse_absorption_column_level_band=snow_absorption,
        ),
        initialized_ground,
        subgrid_fluxes_enabled=parameters.subgrid_fluxes_enabled,
    )
    final_state = SurfaceAlbedoColumnState(
        cosine_solar_zenith_column=cosine,
        soil=soil.state,
        ground=ground.state,
        pure_snow_ground_direct_albedo_column_band=initialize_column_band(
            state.pure_snow_ground_direct_albedo_column_band
        ),
        pure_snow_ground_diffuse_albedo_column_band=initialize_column_band(
            state.pure_snow_ground_diffuse_albedo_column_band
        ),
        no_black_carbon_ground_direct_albedo_column_band=initialize_column_band(
            state.no_black_carbon_ground_direct_albedo_column_band
        ),
        no_black_carbon_ground_diffuse_albedo_column_band=initialize_column_band(
            state.no_black_carbon_ground_diffuse_albedo_column_band
        ),
        no_organic_carbon_ground_direct_albedo_column_band=initialize_column_band(
            state.no_organic_carbon_ground_direct_albedo_column_band
        ),
        no_organic_carbon_ground_diffuse_albedo_column_band=initialize_column_band(
            state.no_organic_carbon_ground_diffuse_albedo_column_band
        ),
        no_dust_ground_direct_albedo_column_band=initialize_column_band(
            state.no_dust_ground_direct_albedo_column_band
        ),
        no_dust_ground_diffuse_albedo_column_band=initialize_column_band(
            state.no_dust_ground_diffuse_albedo_column_band
        ),
    )
    empty_snow = SurfaceAlbedoRuntimeSnowDiagnostics(
        fatal_retry_failure_count=jnp.int32(0),
        energy_conservation_failure_count=jnp.int32(0),
        invalid_active_snow_grain_radius_count=jnp.int32(0),
        maximum_attempt_count=jnp.int32(0),
        maximum_absolute_energy_residual=zero,
    )
    return SurfaceAlbedoColumnRuntimeResult(
        state=final_state,
        diagnostics=SurfaceAlbedoColumnRuntimeDiagnostics(
            soil=soil.diagnostics,
            direct_snow=empty_snow,
            diffuse_snow=empty_snow,
            ground=ground.diagnostics,
        ),
    )


def _pack_column_kernel(
    inputs: SurfaceAlbedoColumnInputs,
    state: SurfaceAlbedoColumnState,
    plan: EqualAxisShardPlan,
) -> tuple[SurfaceAlbedoColumnInputs, SurfaceAlbedoColumnState]:
    pack = lambda tree: jax.tree_util.tree_map(  
        lambda value: _pack_equal_axis(value, plan),
        tree,
    )
    packed_inputs = pack(inputs)
    valid = (
        jnp.asarray(plan.valid_slots, dtype=jnp.bool_)
        if plan.valid_slots is not None
        else jnp.reshape(
            jnp.arange(
                plan.device_count * plan.items_per_device,
                dtype=jnp.int32,
            )
            < jnp.int32(plan.item_count),
            (plan.device_count, plan.items_per_device),
        )
    )
    return (
        packed_inputs._replace(
            nonurban_column=packed_inputs.nonurban_column & valid,
        ),
        pack(state),
    )


def _unpack_column_kernel_state(
    state: SurfaceAlbedoColumnState,
    plan: EqualAxisShardPlan,
) -> SurfaceAlbedoColumnState:
    return jax.tree_util.tree_map(
        lambda value: _unpack_equal_axis(value, plan),
        state,
    )


def _patch_local_canopy_inputs(
    inputs: CanopySurfaceAlbedoInputs,
) -> CanopySurfaceAlbedoInputs:

    column = jnp.asarray(inputs.patch_column, dtype=jnp.int32)
    patch_count = int(column.shape[0])
    return inputs._replace(
        patch_column=jnp.arange(patch_count, dtype=jnp.int32),
        ground_direct_albedo_column_band=(
            inputs.ground_direct_albedo_column_band[column]
        ),
        ground_diffuse_albedo_column_band=(
            inputs.ground_diffuse_albedo_column_band[column]
        ),
        snow_free_soil_direct_albedo_column_band=(
            inputs.snow_free_soil_direct_albedo_column_band[column]
        ),
        snow_free_soil_diffuse_albedo_column_band=(
            inputs.snow_free_soil_diffuse_albedo_column_band[column]
        ),
    )


def _pack_canopy_kernel(
    inputs: CanopySurfaceAlbedoInputs,
    state: CanopySurfaceAlbedoState,
    plan: EqualAxisShardPlan,
) -> tuple[CanopySurfaceAlbedoInputs, CanopySurfaceAlbedoState]:
    local = _patch_local_canopy_inputs(inputs)
    values = []
    for name, value in zip(CanopySurfaceAlbedoInputs._fields, local, strict=True):
        values.append(
            value
            if name in _CANOPY_PFT_INPUT_FIELDS
            else _pack_equal_axis(value, plan)
        )
    packed = CanopySurfaceAlbedoInputs(*values)
    valid = jnp.reshape(
        jnp.arange(plan.device_count * plan.items_per_device, dtype=jnp.int32)
        < jnp.int32(plan.item_count),
        (plan.device_count, plan.items_per_device),
    )
    local_patch = jnp.broadcast_to(
        jnp.arange(plan.items_per_device, dtype=jnp.int32)[None, :],
        (plan.device_count, plan.items_per_device),
    )
    packed = packed._replace(
        nonurban_patch=packed.nonurban_patch & valid,
        patch_column=local_patch,
    )
    packed_state = jax.tree_util.tree_map(
        lambda value: _pack_equal_axis(value, plan),
        state,
    )
    return packed, packed_state


def _unpack_canopy_kernel_state(
    state: CanopySurfaceAlbedoState,
    plan: EqualAxisShardPlan,
) -> CanopySurfaceAlbedoState:
    return jax.tree_util.tree_map(
        lambda value: _unpack_equal_axis(value, plan),
        state,
    )


def _aggregate_count_tuple(mapped):
    return type(mapped)(
        *(jnp.sum(value, dtype=jnp.int32) for value in mapped)
    )


def _aggregate_snow_runtime_diagnostics(
    mapped: SurfaceAlbedoRuntimeSnowDiagnostics,
) -> SurfaceAlbedoRuntimeSnowDiagnostics:
    return SurfaceAlbedoRuntimeSnowDiagnostics(
        fatal_retry_failure_count=jnp.sum(
            mapped.fatal_retry_failure_count,
            dtype=jnp.int32,
        ),
        energy_conservation_failure_count=jnp.sum(
            mapped.energy_conservation_failure_count,
            dtype=jnp.int32,
        ),
        invalid_active_snow_grain_radius_count=jnp.sum(
            mapped.invalid_active_snow_grain_radius_count,
            dtype=jnp.int32,
        ),
        maximum_attempt_count=jnp.max(mapped.maximum_attempt_count),
        maximum_absolute_energy_residual=jnp.max(
            mapped.maximum_absolute_energy_residual
        ),
    )


def _aggregate_runtime_result(
    mapped: SurfaceAlbedoRuntimeResult,
    plan: SurfaceAlbedoShardPlan,
) -> SurfaceAlbedoRuntimeResult:
    canopy = mapped.diagnostics.canopy
    canopy_counts = CanopySurfaceAlbedoDiagnostics(
        *(jnp.sum(value, dtype=jnp.int32) for value in canopy[:-1]),
        snow_free_diagnostic=canopy.snow_free_diagnostic[0],
    )
    return SurfaceAlbedoRuntimeResult(
        state=unpack_surface_albedo_state(mapped.state, plan),
        diagnostics=SurfaceAlbedoRuntimeDiagnostics(
            soil=_aggregate_count_tuple(mapped.diagnostics.soil),
            direct_snow=_aggregate_snow_runtime_diagnostics(
                mapped.diagnostics.direct_snow
            ),
            diffuse_snow=_aggregate_snow_runtime_diagnostics(
                mapped.diagnostics.diffuse_snow
            ),
            ground=_aggregate_count_tuple(mapped.diagnostics.ground),
            canopy=canopy_counts,
        ),
    )


def _aggregate_column_runtime_result(
    mapped: SurfaceAlbedoColumnRuntimeResult,
    plan: EqualAxisShardPlan,
) -> SurfaceAlbedoColumnRuntimeResult:
    return SurfaceAlbedoColumnRuntimeResult(
        state=_unpack_column_kernel_state(mapped.state, plan),
        diagnostics=SurfaceAlbedoColumnRuntimeDiagnostics(
            soil=_aggregate_count_tuple(mapped.diagnostics.soil),
            direct_snow=_aggregate_snow_runtime_diagnostics(
                mapped.diagnostics.direct_snow
            ),
            diffuse_snow=_aggregate_snow_runtime_diagnostics(
                mapped.diagnostics.diffuse_snow
            ),
            ground=_aggregate_count_tuple(mapped.diagnostics.ground),
        ),
    )


def _combine_column_runtime_results(
    selected: SurfaceAlbedoColumnRuntimeResult,
    ordinary: SurfaceAlbedoColumnRuntimeResult,
    selection_mask: np.ndarray,
) -> SurfaceAlbedoColumnRuntimeResult:

    selection = jnp.asarray(selection_mask, dtype=jnp.bool_)

    def choose(selected_value, ordinary_value):
        selected_array = jnp.asarray(selected_value)
        mask = jnp.reshape(
            selection,
            (selection.shape[0],) + (1,) * (selected_array.ndim - 1),
        )
        return jnp.where(mask, selected_array, jnp.asarray(ordinary_value))

    def add_counts(left, right):
        return type(left)(
            *(jnp.asarray(a) + jnp.asarray(b) for a, b in zip(left, right, strict=True))
        )

    def combine_snow(left, right):
        return SurfaceAlbedoRuntimeSnowDiagnostics(
            fatal_retry_failure_count=(
                left.fatal_retry_failure_count + right.fatal_retry_failure_count
            ),
            energy_conservation_failure_count=(
                left.energy_conservation_failure_count
                + right.energy_conservation_failure_count
            ),
            invalid_active_snow_grain_radius_count=(
                left.invalid_active_snow_grain_radius_count
                + right.invalid_active_snow_grain_radius_count
            ),
            maximum_attempt_count=jnp.maximum(
                left.maximum_attempt_count,
                right.maximum_attempt_count,
            ),
            maximum_absolute_energy_residual=jnp.maximum(
                left.maximum_absolute_energy_residual,
                right.maximum_absolute_energy_residual,
            ),
        )

    return SurfaceAlbedoColumnRuntimeResult(
        state=jax.tree_util.tree_map(choose, selected.state, ordinary.state),
        diagnostics=SurfaceAlbedoColumnRuntimeDiagnostics(
            soil=add_counts(selected.diagnostics.soil, ordinary.diagnostics.soil),
            direct_snow=combine_snow(
                selected.diagnostics.direct_snow,
                ordinary.diagnostics.direct_snow,
            ),
            diffuse_snow=combine_snow(
                selected.diagnostics.diffuse_snow,
                ordinary.diagnostics.diffuse_snow,
            ),
            ground=add_counts(
                selected.diagnostics.ground,
                ordinary.diagnostics.ground,
            ),
        ),
    )


def _aggregate_canopy_result(
    mapped: CanopySurfaceAlbedoResult,
    plan: EqualAxisShardPlan,
) -> CanopySurfaceAlbedoResult:
    diagnostics = mapped.diagnostics
    return CanopySurfaceAlbedoResult(
        state=_unpack_canopy_kernel_state(mapped.state, plan),
        diagnostics=CanopySurfaceAlbedoDiagnostics(
            *(jnp.sum(value, dtype=jnp.int32) for value in diagnostics[:-1]),
            snow_free_diagnostic=diagnostics.snow_free_diagnostic[0],
        ),
    )


@cache
def _mapped_surface_albedo(device_count: int):
    compiler_options = (
        {"xla_cpu_parallel_codegen_split_count": 1}
        if jax.default_backend() == "cpu"
        else None
    )
    return _distributed_vmap(
        surface_albedo_runtime_step,
        device_count=device_count,
        axis_name="clm_surface_albedo_device",
        in_axes=(_SURFACE_ALBEDO_INPUT_AXES, 0, None),
        compiler_options=compiler_options,
    )


def _compiled_axis_layout(function):
    @cache
    def kernel(dimensions):
        @jax.jit
        def execute(arguments, indices, valid):
            return function(*arguments, EqualAxisShardPlan(*dimensions, indices, valid))

        return execute

    @wraps(function)
    def execute(*arguments):
        *values, plan = arguments
        dimensions = (
            plan.item_count, plan.device_count, plan.items_per_device,
            plan.padding if plan.item_indices is None else 0,
        )
        return kernel(dimensions)(tuple(values), plan.item_indices, plan.valid_slots)

    return execute


_pack_column_kernel = _compiled_axis_layout(_pack_column_kernel)
_pack_canopy_kernel = _compiled_axis_layout(_pack_canopy_kernel)
_aggregate_column_runtime_result = _compiled_axis_layout(_aggregate_column_runtime_result)
_aggregate_canopy_result = _compiled_axis_layout(_aggregate_canopy_result)
_combine_column_runtime_results = jax.jit(_combine_column_runtime_results)


@cache
def _mapped_surface_albedo_columns(device_count: int):
    compiler_options = (
        {"xla_cpu_parallel_codegen_split_count": 1}
        if jax.default_backend() == "cpu"
        else None
    )
    return _distributed_vmap(
        surface_albedo_column_runtime_step,
        device_count=device_count,
        axis_name="clm_surface_albedo_column_device",
        in_axes=(_SURFACE_ALBEDO_COLUMN_INPUT_AXES, 0, None),
        compiler_options=compiler_options,
    )


@cache
def _compiled_unqualified_surface_albedo_columns():
    return process_lifetime_jit(
        surface_albedo_unqualified_column_runtime_step,
        cpu_parallel_codegen_split_count=(
            1 if jax.default_backend() == "cpu" else None
        ),
    )


@cache
def _mapped_surface_albedo_canopy(device_count: int):
    compiler_options = (
        {"xla_cpu_parallel_codegen_split_count": 1}
        if jax.default_backend() == "cpu"
        else None
    )
    return _distributed_vmap(
        canopy_surface_albedo_big_leaf,
        device_count=device_count,
        axis_name="clm_surface_albedo_canopy_device",
        in_axes=(_SURFACE_ALBEDO_CANOPY_INPUT_AXES, 0, None),
        compiler_options=compiler_options,
    )


def _execute_column_partition(
    inputs: SurfaceAlbedoColumnInputs,
    state: SurfaceAlbedoColumnState,
    parameters: SurfaceAlbedoColumnParameters,
    plan: EqualAxisShardPlan,
    devices: tuple[jax.Device, ...],
) -> SurfaceAlbedoColumnRuntimeResult:

    packed_inputs, packed_state = _pack_column_kernel(inputs, state, plan)
    arguments = _reshard_tree_for_pmap(
        (packed_inputs, packed_state, parameters),
        (_SURFACE_ALBEDO_COLUMN_INPUT_AXES, 0, None),
        devices,
        "clm_surface_albedo_column_device",
    )
    mapped = _mapped_surface_albedo_columns(len(devices))(*arguments)
    mapped = co_locate_array_leaves(mapped, devices[0])
    return _aggregate_column_runtime_result(mapped, plan)


def surface_albedo_parallel(
    call: SurfaceAlbedoCall,
    *,
    device_count: int,
) -> SurfaceAlbedoRuntimeResult:

    if jax.local_device_count() < device_count:
        raise ValueError(
            f"requested {device_count} SurfaceAlbedo devices but JAX exposes "
            f"only {jax.local_device_count()}"
        )
    devices = tuple(jax.local_devices()[:device_count])

    column_inputs = surface_albedo_column_inputs(call.inputs)
    column_state = surface_albedo_column_state(call.state)
    column_parameters = surface_albedo_column_parameters(call.parameters)
    qualified_snow = (
        np.asarray(jax.device_get(column_inputs.nonurban_column), dtype=np.bool_)
        & (
            np.asarray(
                jax.device_get(column_inputs.cosine_solar_zenith_column),
                dtype=np.float64,
            )
            > 0.0
        )
        & (
            np.asarray(
                jax.device_get(column_inputs.snow_water_equivalent_column),
                dtype=np.float64,
            )
            > MINIMUM_SNOW_WATER_EQUIVALENT
        )
    )
    qualified_plan = _plan_selected_axis(qualified_snow, device_count)
    qualified_columns = (
        None
        if qualified_plan is None
        else _execute_column_partition(
            column_inputs,
            column_state,
            column_parameters,
            qualified_plan,
            devices,
        )
    )
    ordinary_columns = None
    if not np.all(qualified_snow):
        ordinary_inputs = column_inputs._replace(
            nonurban_column=(
                column_inputs.nonurban_column
                & ~jnp.asarray(qualified_snow, dtype=jnp.bool_)
            ),
        )
        ordinary_columns = _compiled_unqualified_surface_albedo_columns()(
            ordinary_inputs,
            column_state,
            column_parameters,
        )
    if qualified_columns is None:
        if ordinary_columns is None:  
            raise AssertionError("SurfaceAlbedo has no column partition")
        columns = ordinary_columns
    elif ordinary_columns is None:
        columns = qualified_columns
    else:
        columns = _combine_column_runtime_results(
            qualified_columns,
            ordinary_columns,
            qualified_snow,
        )

    canopy_inputs = surface_albedo_canopy_inputs(call.inputs, columns.state)
    canopy_plan = _plan_equal_axis(
        int(canopy_inputs.nonurban_patch.shape[0]),
        device_count,
    )
    packed_canopy_inputs, packed_canopy_state = _pack_canopy_kernel(
        canopy_inputs,
        call.state.canopy,
        canopy_plan,
    )
    canopy_arguments = _reshard_tree_for_pmap(
        (packed_canopy_inputs, packed_canopy_state, call.parameters.canopy),
        (_SURFACE_ALBEDO_CANOPY_INPUT_AXES, 0, None),
        devices,
        "clm_surface_albedo_canopy_device",
    )
    mapped_canopy = _mapped_surface_albedo_canopy(device_count)(
        *canopy_arguments
    )
    mapped_canopy = co_locate_array_leaves(mapped_canopy, devices[0])
    canopy = _aggregate_canopy_result(mapped_canopy, canopy_plan)

    column_state = columns.state
    state = SurfaceAlbedoState(
        cosine_solar_zenith_column=column_state.cosine_solar_zenith_column,
        soil=column_state.soil,
        ground=column_state.ground,
        canopy=canopy.state,
        pure_snow_ground_direct_albedo_column_band=(
            column_state.pure_snow_ground_direct_albedo_column_band
        ),
        pure_snow_ground_diffuse_albedo_column_band=(
            column_state.pure_snow_ground_diffuse_albedo_column_band
        ),
        no_black_carbon_ground_direct_albedo_column_band=(
            column_state.no_black_carbon_ground_direct_albedo_column_band
        ),
        no_black_carbon_ground_diffuse_albedo_column_band=(
            column_state.no_black_carbon_ground_diffuse_albedo_column_band
        ),
        no_organic_carbon_ground_direct_albedo_column_band=(
            column_state.no_organic_carbon_ground_direct_albedo_column_band
        ),
        no_organic_carbon_ground_diffuse_albedo_column_band=(
            column_state.no_organic_carbon_ground_diffuse_albedo_column_band
        ),
        no_dust_ground_direct_albedo_column_band=(
            column_state.no_dust_ground_direct_albedo_column_band
        ),
        no_dust_ground_diffuse_albedo_column_band=(
            column_state.no_dust_ground_diffuse_albedo_column_band
        ),
    )
    return SurfaceAlbedoRuntimeResult(
        state=state,
        diagnostics=SurfaceAlbedoRuntimeDiagnostics(
            soil=columns.diagnostics.soil,
            direct_snow=columns.diagnostics.direct_snow,
            diffuse_snow=columns.diagnostics.diffuse_snow,
            ground=columns.diagnostics.ground,
            canopy=canopy.diagnostics,
        ),
    )


_USE_GPU_CONVERGED_ALBEDO = True
_USE_GPU_COMPACT_ALBEDO = True


@jax.custom_jvp
def _gpu_surface_albedo(inputs, state, parameters):
    return surface_albedo_configured(
        inputs, state, parameters, early_exit_snicar_retry_loop=True,
        compact_active_snow_columns=True,
    )


@partial(_gpu_surface_albedo.defjvp, symbolic_zeros=True)
def _gpu_surface_albedo_jvp(primals, tangents):
    from clm_jax.compilation import _symbolic_zero_jvp
    return _symbolic_zero_jvp(surface_albedo_configured, primals, tangents)


_compiled_gpu_surface_albedo = jit_unique(_gpu_surface_albedo, forward_inputs=True)


@jax.custom_jvp
def _gpu_surface_albedo_runtime(inputs, state, parameters):
    return _project_surface_albedo_runtime(
        surface_albedo_configured(
            inputs, state, parameters, early_exit_snicar_retry_loop=True,
            compact_active_snow_columns=True, compact_snow_diagnostics=True,
        ), check_dense_fields=True,
    )


@partial(_gpu_surface_albedo_runtime.defjvp, symbolic_zeros=True)
def _gpu_surface_albedo_runtime_jvp(primals, tangents):
    def reference(*arguments):
        return _project_surface_albedo_runtime(
            surface_albedo_configured(*arguments), check_dense_fields=True)
    from clm_jax.compilation import _symbolic_zero_jvp
    return _symbolic_zero_jvp(reference, primals, tangents)


_compiled_gpu_surface_albedo_runtime = jit_unique(
    _gpu_surface_albedo_runtime, forward_inputs=True,
)


class SurfaceAlbedoExecutor:

    def __init__(
        self,
        *,
        jit: bool = True,
        device_count: int = 1,
        stage_completed: Callable[[str], None] | None = None,
    ) -> None:
        if isinstance(device_count, bool) or not isinstance(device_count, int):
            raise TypeError("SurfaceAlbedo device_count must be an integer")
        if device_count < 1:
            raise ValueError("SurfaceAlbedo device_count must be positive")
        if not jit and device_count != 1:
            raise ValueError("mapped SurfaceAlbedo execution requires jit=True")
        self.stage_completed = stage_completed
        self.device_count = device_count
        self.step = process_lifetime_jit(surface_albedo_configured, enabled=jit)
        if (
            _USE_GPU_CONVERGED_ALBEDO and jit and device_count == 1
            and jax.default_backend() == "gpu"
        ):
            self.step = (
                _compiled_gpu_surface_albedo_runtime if _USE_GPU_COMPACT_ALBEDO
                else _compiled_gpu_surface_albedo
            )

    def __call__(
        self,
        call: SurfaceAlbedoCall,
    ) -> SurfaceAlbedoResult | SurfaceAlbedoRuntimeResult:
        call = co_locate_array_leaves(call, jax.devices()[0])
        stage_barrier(call)
        result = (
            self.step(call.inputs, call.state, call.parameters)
            if self.device_count == 1
            else surface_albedo_parallel(call, device_count=self.device_count)
        )
        if self.stage_completed is not None:
            stage_barrier(result)
            self.stage_completed("stage510_surface_albedo")
        return result


__all__ = [
    "SurfaceAlbedoExecutor",
    "SurfaceAlbedoRuntimeDiagnostics",
    "SurfaceAlbedoRuntimeResult",
    "SurfaceAlbedoRuntimeSnowDiagnostics",
    "SurfaceAlbedoShardPlan",
    "pack_surface_albedo_inputs",
    "pack_surface_albedo_state",
    "plan_surface_albedo_shards",
    "surface_albedo_parallel",
    "surface_albedo_runtime_step",
    "unpack_surface_albedo_state",
]
