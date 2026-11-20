
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class DynamicLandcoverCarbonFluxState(NamedTuple):
    dwt_seed_carbon_to_leaf_gridcell: jnp.ndarray
    dwt_seed_carbon_to_dead_stem_gridcell: jnp.ndarray
    dwt_conversion_carbon_flux_gridcell: jnp.ndarray
    dwt_slash_carbon_flux_column: jnp.ndarray
    dwt_fine_root_carbon_to_metabolic_litter_column_level: jnp.ndarray
    dwt_fine_root_carbon_to_cellulose_litter_column_level: jnp.ndarray
    dwt_fine_root_carbon_to_lignin_litter_column_level: jnp.ndarray
    dwt_live_coarse_root_carbon_to_cwd_column_level: jnp.ndarray
    dwt_dead_coarse_root_carbon_to_cwd_column_level: jnp.ndarray


class DynamicLandcoverNitrogenFluxState(NamedTuple):
    dwt_seed_nitrogen_to_leaf_gridcell: jnp.ndarray
    dwt_seed_nitrogen_to_dead_stem_gridcell: jnp.ndarray
    dwt_conversion_nitrogen_flux_gridcell: jnp.ndarray
    dwt_fine_root_nitrogen_to_metabolic_litter_column_level: jnp.ndarray
    dwt_fine_root_nitrogen_to_cellulose_litter_column_level: jnp.ndarray
    dwt_fine_root_nitrogen_to_lignin_litter_column_level: jnp.ndarray
    dwt_live_coarse_root_nitrogen_to_cwd_column_level: jnp.ndarray
    dwt_dead_coarse_root_nitrogen_to_cwd_column_level: jnp.ndarray


class DynamicLandcoverCarbonSummaryState(NamedTuple):
    displayed_vegetation_carbon_patch: jnp.ndarray
    stored_vegetation_carbon_patch: jnp.ndarray
    total_carbon_patch: jnp.ndarray


class DynamicLandcoverNitrogenSummaryState(NamedTuple):
    displayed_vegetation_nitrogen_patch: jnp.ndarray
    stored_vegetation_nitrogen_patch: jnp.ndarray
    total_vegetation_nitrogen_patch: jnp.ndarray
    total_nitrogen_patch: jnp.ndarray


class BgcTimestepInitializationState(NamedTuple):
    carbon_flux: DynamicLandcoverCarbonFluxState
    nitrogen_flux: DynamicLandcoverNitrogenFluxState
    carbon_summary: DynamicLandcoverCarbonSummaryState
    nitrogen_summary: DynamicLandcoverNitrogenSummaryState


class BgcTimestepInitializationDiagnostics(NamedTuple):
    zeroed_gridcell_value_count: jnp.ndarray
    zeroed_column_value_count: jnp.ndarray
    zeroed_column_level_value_count: jnp.ndarray
    zeroed_patch_value_count: jnp.ndarray
    zeroed_value_count: jnp.ndarray


class BgcTimestepInitializationResult(NamedTuple):
    state: BgcTimestepInitializationState
    diagnostics: BgcTimestepInitializationDiagnostics


def _validate_shapes(state: BgcTimestepInitializationState) -> tuple[int, int, int, int]:
    carbon_flux = state.carbon_flux
    nitrogen_flux = state.nitrogen_flux
    grid_fields = (
        carbon_flux.dwt_seed_carbon_to_leaf_gridcell,
        carbon_flux.dwt_seed_carbon_to_dead_stem_gridcell,
        carbon_flux.dwt_conversion_carbon_flux_gridcell,
        nitrogen_flux.dwt_seed_nitrogen_to_leaf_gridcell,
        nitrogen_flux.dwt_seed_nitrogen_to_dead_stem_gridcell,
        nitrogen_flux.dwt_conversion_nitrogen_flux_gridcell,
    )
    if any(len(field.shape) != 1 for field in grid_fields):
        raise ValueError("dynamic-land-cover gridcell fields must be vectors")
    gridcell_count = grid_fields[0].shape[0]
    if any(field.shape != (gridcell_count,) for field in grid_fields):
        raise ValueError("dynamic-land-cover gridcell fields must match")

    carbon_column_level = carbon_flux[4:]
    nitrogen_column_level = nitrogen_flux[3:]
    column_level_fields = (*carbon_column_level, *nitrogen_column_level)
    if any(len(field.shape) != 2 for field in column_level_fields):
        raise ValueError("dynamic-land-cover column-level fields must be matrices")
    column_count, level_count = column_level_fields[0].shape
    if any(field.shape != (column_count, level_count) for field in column_level_fields):
        raise ValueError("dynamic-land-cover column-level fields must match")
    if carbon_flux.dwt_slash_carbon_flux_column.shape != (column_count,):
        raise ValueError("dynamic-land-cover column fields must match level matrices")

    patch_fields = (*state.carbon_summary, *state.nitrogen_summary)
    if any(len(field.shape) != 1 for field in patch_fields):
        raise ValueError("dynamic-land-cover patch fields must be vectors")
    patch_count = patch_fields[0].shape[0]
    if any(field.shape != (patch_count,) for field in patch_fields):
        raise ValueError("dynamic-land-cover patch fields must match")
    return gridcell_count, column_count, level_count, patch_count


def initialize_bgc_timestep(
    state: BgcTimestepInitializationState,
) -> BgcTimestepInitializationResult:

    gridcell_count, column_count, level_count, patch_count = _validate_shapes(state)
    zeroed_state = jax.tree.map(
        lambda value: jnp.zeros_like(value, dtype=jnp.float64),
        state,
    )
    gridcell_values = jnp.int32(6 * gridcell_count)
    column_values = jnp.int32(column_count)
    column_level_values = jnp.int32(10 * column_count * level_count)
    patch_values = jnp.int32(7 * patch_count)
    return BgcTimestepInitializationResult(
        state=zeroed_state,
        diagnostics=BgcTimestepInitializationDiagnostics(
            zeroed_gridcell_value_count=gridcell_values,
            zeroed_column_value_count=column_values,
            zeroed_column_level_value_count=column_level_values,
            zeroed_patch_value_count=patch_values,
            zeroed_value_count=(
                gridcell_values + column_values + column_level_values + patch_values
            ),
        ),
    )
