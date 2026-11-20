
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.physics.numerics import (
    fortran_add,
    fortran_multiply,
    fortran_ordered_sum,
)
from clm_jax.physics.subgrid import patch_to_column_average

NUMBER_OF_DECOMPOSITION_LEVELS = 20
NUMBER_OF_DECOMPOSITION_POOLS = 7

LITTER_POOL_INDICES = (0, 1, 2)
CWD_POOL_INDICES = (3,)
SOIL_POOL_INDICES = (4, 5, 6)

CARBON_DISPLAYED_COMPONENTS = (
    "leafc",
    "frootc",
    "livestemc",
    "deadstemc",
    "livecrootc",
    "deadcrootc",
)
CARBON_STORED_COMPONENTS = (
    "cpool",
    "leafc_storage",
    "frootc_storage",
    "livestemc_storage",
    "deadstemc_storage",
    "livecrootc_storage",
    "deadcrootc_storage",
    "leafc_xfer",
    "frootc_xfer",
    "livestemc_xfer",
    "deadstemc_xfer",
    "livecrootc_xfer",
    "deadcrootc_xfer",
    "gresp_storage",
    "gresp_xfer",
)
NITROGEN_DISPLAYED_COMPONENTS = (
    "leafn",
    "frootn",
    "livestemn",
    "deadstemn",
    "livecrootn",
    "deadcrootn",
)
NITROGEN_STORED_COMPONENTS = (
    "leafn_storage",
    "frootn_storage",
    "livestemn_storage",
    "deadstemn_storage",
    "livecrootn_storage",
    "deadcrootn_storage",
    "leafn_xfer",
    "frootn_xfer",
    "livestemn_xfer",
    "deadstemn_xfer",
    "livecrootn_xfer",
    "deadcrootn_xfer",
    "npool",
    "retransn",
)


class CarbonNitrogenBalanceInitializationInputs(NamedTuple):
    all_column: jnp.ndarray
    soil_column: jnp.ndarray
    soil_patch: jnp.ndarray
    patch_active: jnp.ndarray
    patch_column_index: jnp.ndarray
    patch_weight_on_column: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    decomposition_layer_thickness: jnp.ndarray
    decomposition_interface_depth: jnp.ndarray
    decomposition_carbon_column_level_pool: jnp.ndarray
    carbon_truncation_column_level: jnp.ndarray
    decomposition_nitrogen_column_level_pool: jnp.ndarray
    mineral_nitrogen_column_level: jnp.ndarray
    nitrate_nitrogen_column_level: jnp.ndarray
    ammonium_nitrogen_column_level: jnp.ndarray
    nitrogen_truncation_column_level: jnp.ndarray
    displayed_carbon_patch_component: jnp.ndarray
    stored_carbon_patch_component: jnp.ndarray
    grain_carbon_patch: jnp.ndarray
    grain_carbon_storage_patch: jnp.ndarray
    grain_carbon_transfer_patch: jnp.ndarray
    crop_seed_carbon_deficit_patch: jnp.ndarray
    excess_maintenance_respiration_carbon_patch: jnp.ndarray
    excess_maintenance_respiration_carbon_loss_patch: jnp.ndarray
    patch_carbon_truncation: jnp.ndarray
    displayed_nitrogen_patch_component: jnp.ndarray
    stored_nitrogen_patch_component: jnp.ndarray
    grain_nitrogen_patch: jnp.ndarray
    grain_nitrogen_storage_patch: jnp.ndarray
    grain_nitrogen_transfer_patch: jnp.ndarray
    crop_seed_nitrogen_deficit_patch: jnp.ndarray
    patch_nitrogen_truncation: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CarbonNitrogenBalanceInitializationParameters:
    use_crop: bool = True
    use_nitrification_denitrification: bool = True
    first_prognostic_crop_pft: int = 17
    maximum_patches_per_column: int = 79

    def tree_flatten(self):
        return (), (
            self.use_crop,
            self.use_nitrification_denitrification,
            self.first_prognostic_crop_pft,
            self.maximum_patches_per_column,
        )

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del children
        return cls(*auxiliary)


DEFAULT_CARBON_NITROGEN_BALANCE_INITIALIZATION_PARAMETERS = (
    CarbonNitrogenBalanceInitializationParameters()
)


class CarbonNitrogenStateSummaries(NamedTuple):
    decomposition_carbon_column_pool: jnp.ndarray
    decomposition_carbon_one_meter_column_pool: jnp.ndarray
    decomposition_soil_carbon_column_level: jnp.ndarray
    carbon_truncation_column: jnp.ndarray
    total_litter_carbon_column: jnp.ndarray
    total_litter_carbon_one_meter_column: jnp.ndarray
    total_soil_carbon_column: jnp.ndarray
    total_soil_carbon_one_meter_column: jnp.ndarray
    coarse_woody_debris_carbon_column: jnp.ndarray
    nitrate_nitrogen_column: jnp.ndarray
    ammonium_nitrogen_column: jnp.ndarray
    decomposition_nitrogen_column_pool: jnp.ndarray
    decomposition_nitrogen_one_meter_column_pool: jnp.ndarray
    decomposition_soil_nitrogen_column_level: jnp.ndarray
    mineral_nitrogen_column: jnp.ndarray
    nitrogen_truncation_column: jnp.ndarray
    total_litter_nitrogen_column: jnp.ndarray
    total_litter_nitrogen_one_meter_column: jnp.ndarray
    total_soil_nitrogen_column: jnp.ndarray
    total_soil_nitrogen_one_meter_column: jnp.ndarray
    coarse_woody_debris_nitrogen_column: jnp.ndarray
    displayed_vegetation_carbon_patch: jnp.ndarray
    stored_vegetation_carbon_patch: jnp.ndarray
    total_vegetation_carbon_patch: jnp.ndarray
    total_carbon_patch: jnp.ndarray
    wood_carbon_patch: jnp.ndarray
    total_vegetation_carbon_column: jnp.ndarray
    total_patch_carbon_column: jnp.ndarray
    total_ecosystem_carbon_column: jnp.ndarray
    total_carbon_column: jnp.ndarray
    displayed_vegetation_nitrogen_patch: jnp.ndarray
    stored_vegetation_nitrogen_patch: jnp.ndarray
    total_vegetation_nitrogen_patch: jnp.ndarray
    total_nitrogen_patch: jnp.ndarray
    total_vegetation_nitrogen_column: jnp.ndarray
    total_patch_nitrogen_column: jnp.ndarray
    total_ecosystem_nitrogen_column: jnp.ndarray
    total_nitrogen_column: jnp.ndarray


class CarbonNitrogenBalanceInitializationState(NamedTuple):
    summaries: CarbonNitrogenStateSummaries
    beginning_carbon_balance_column: jnp.ndarray
    beginning_nitrogen_balance_column: jnp.ndarray


class CarbonNitrogenBalanceInitializationDiagnostics(NamedTuple):
    summarized_column_count: jnp.ndarray
    balanced_column_count: jnp.ndarray
    summarized_patch_count: jnp.ndarray


class CarbonNitrogenBalanceInitializationResult(NamedTuple):
    state: CarbonNitrogenBalanceInitializationState
    diagnostics: CarbonNitrogenBalanceInitializationDiagnostics


def _validate_shapes(
    inputs: CarbonNitrogenBalanceInitializationInputs,
    state: CarbonNitrogenBalanceInitializationState,
    parameters: CarbonNitrogenBalanceInitializationParameters,
) -> tuple[int, int]:
    if jnp.ndim(inputs.all_column) != 1 or jnp.ndim(inputs.soil_patch) != 1:
        raise ValueError("column and patch filters must be vectors")
    column_count = inputs.all_column.shape[0]
    patch_count = inputs.soil_patch.shape[0]
    for name in ("soil_column",):
        if jnp.shape(getattr(inputs, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    for name in (
        "patch_active",
        "patch_column_index",
        "patch_weight_on_column",
        "vegetation_type_patch",
        "grain_carbon_patch",
        "grain_carbon_storage_patch",
        "grain_carbon_transfer_patch",
        "crop_seed_carbon_deficit_patch",
        "excess_maintenance_respiration_carbon_patch",
        "excess_maintenance_respiration_carbon_loss_patch",
        "patch_carbon_truncation",
        "grain_nitrogen_patch",
        "grain_nitrogen_storage_patch",
        "grain_nitrogen_transfer_patch",
        "crop_seed_nitrogen_deficit_patch",
        "patch_nitrogen_truncation",
    ):
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    if jnp.shape(inputs.decomposition_layer_thickness) != (NUMBER_OF_DECOMPOSITION_LEVELS,):
        raise ValueError("decomposition_layer_thickness must contain 20 levels")
    if jnp.shape(inputs.decomposition_interface_depth) != (NUMBER_OF_DECOMPOSITION_LEVELS,):
        raise ValueError("decomposition_interface_depth must contain 20 level bottoms")
    pool_shape = (
        column_count,
        NUMBER_OF_DECOMPOSITION_LEVELS,
        NUMBER_OF_DECOMPOSITION_POOLS,
    )
    for name in (
        "decomposition_carbon_column_level_pool",
        "decomposition_nitrogen_column_level_pool",
    ):
        if jnp.shape(getattr(inputs, name)) != pool_shape:
            raise ValueError(f"{name} must have shape {pool_shape}")
    level_shape = (column_count, NUMBER_OF_DECOMPOSITION_LEVELS)
    for name in (
        "carbon_truncation_column_level",
        "mineral_nitrogen_column_level",
        "nitrate_nitrogen_column_level",
        "ammonium_nitrogen_column_level",
        "nitrogen_truncation_column_level",
    ):
        if jnp.shape(getattr(inputs, name)) != level_shape:
            raise ValueError(f"{name} must have shape {level_shape}")
    component_shapes = {
        "displayed_carbon_patch_component": (patch_count, len(CARBON_DISPLAYED_COMPONENTS)),
        "stored_carbon_patch_component": (patch_count, len(CARBON_STORED_COMPONENTS)),
        "displayed_nitrogen_patch_component": (
            patch_count,
            len(NITROGEN_DISPLAYED_COMPONENTS),
        ),
        "stored_nitrogen_patch_component": (
            patch_count,
            len(NITROGEN_STORED_COMPONENTS),
        ),
    }
    for name, expected in component_shapes.items():
        if jnp.shape(getattr(inputs, name)) != expected:
            raise ValueError(f"{name} must have shape {expected}")
    summary_shapes = {
        "decomposition_carbon_column_pool": (column_count, NUMBER_OF_DECOMPOSITION_POOLS),
        "decomposition_carbon_one_meter_column_pool": (
            column_count,
            NUMBER_OF_DECOMPOSITION_POOLS,
        ),
        "decomposition_soil_carbon_column_level": (
            column_count,
            NUMBER_OF_DECOMPOSITION_LEVELS,
        ),
        "decomposition_nitrogen_column_pool": (
            column_count,
            NUMBER_OF_DECOMPOSITION_POOLS,
        ),
        "decomposition_nitrogen_one_meter_column_pool": (
            column_count,
            NUMBER_OF_DECOMPOSITION_POOLS,
        ),
        "decomposition_soil_nitrogen_column_level": (
            column_count,
            NUMBER_OF_DECOMPOSITION_LEVELS,
        ),
    }
    patch_summaries = {
        "displayed_vegetation_carbon_patch",
        "stored_vegetation_carbon_patch",
        "total_vegetation_carbon_patch",
        "total_carbon_patch",
        "wood_carbon_patch",
        "displayed_vegetation_nitrogen_patch",
        "stored_vegetation_nitrogen_patch",
        "total_vegetation_nitrogen_patch",
        "total_nitrogen_patch",
    }
    for name in CarbonNitrogenStateSummaries._fields:
        expected = summary_shapes.get(
            name,
            (patch_count,) if name in patch_summaries else (column_count,),
        )
        if jnp.shape(getattr(state.summaries, name)) != expected:
            raise ValueError(f"summary {name} must have shape {expected}")
    for name in (
        "beginning_carbon_balance_column",
        "beginning_nitrogen_balance_column",
    ):
        if jnp.shape(getattr(state, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    if parameters.maximum_patches_per_column < 1:
        raise ValueError("maximum_patches_per_column must be positive")
    return column_count, patch_count


def _integrate_levels(
    values: jnp.ndarray,
    thickness: jnp.ndarray,
    column_mask: jnp.ndarray,
) -> jnp.ndarray:
    result = jnp.zeros((values.shape[0], *values.shape[2:]), dtype=jnp.float64)
    update = column_mask.reshape((values.shape[0],) + (1,) * (values.ndim - 2))

    def add_level(level: int, running: jnp.ndarray) -> jnp.ndarray:
        contribution = fortran_multiply(values[:, level, ...], thickness[level])
        return jnp.where(update, fortran_add(running, contribution), running)

    return lax.fori_loop(0, NUMBER_OF_DECOMPOSITION_LEVELS, add_level, result)


def _integrate_one_meter(
    values: jnp.ndarray,
    thickness: jnp.ndarray,
    interface_depth: jnp.ndarray,
    column_mask: jnp.ndarray,
) -> jnp.ndarray:
    top = jnp.concatenate((jnp.zeros(1, dtype=jnp.float64), interface_depth[:-1]))
    extent = jnp.where(
        interface_depth <= jnp.float64(1.0),
        thickness,
        jnp.where(top < jnp.float64(1.0), fortran_add(jnp.float64(1.0), -top), 0.0),
    )
    return _integrate_levels(values, extent, column_mask)


def _sum_selected_pools(values: jnp.ndarray, indices: tuple[int, ...]) -> jnp.ndarray:
    return fortran_ordered_sum(jnp.stack([values[..., index] for index in indices], axis=0))


def _masked_component_sum(components: jnp.ndarray, update: jnp.ndarray) -> jnp.ndarray:
    total = fortran_ordered_sum(jnp.swapaxes(components, 0, 1))
    return jnp.where(update, total, jnp.float64(0.0))


def _preserve_filtered(
    candidate: jnp.ndarray,
    previous: jnp.ndarray,
    update: jnp.ndarray,
) -> jnp.ndarray:
    shaped_update = update.reshape((update.shape[0],) + (1,) * (candidate.ndim - 1))
    return jnp.where(shaped_update, candidate, previous)


def initialize_carbon_nitrogen_balance(
    inputs: CarbonNitrogenBalanceInitializationInputs,
    state: CarbonNitrogenBalanceInitializationState,
    parameters: CarbonNitrogenBalanceInitializationParameters = (
        DEFAULT_CARBON_NITROGEN_BALANCE_INITIALIZATION_PARAMETERS
    ),
) -> CarbonNitrogenBalanceInitializationResult:

    _validate_shapes(inputs, state, parameters)
    all_column = jnp.asarray(inputs.all_column, dtype=jnp.bool_)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    previous = state.summaries
    thickness = jnp.asarray(inputs.decomposition_layer_thickness, dtype=jnp.float64)
    interface = jnp.asarray(inputs.decomposition_interface_depth, dtype=jnp.float64)
    carbon = jnp.asarray(inputs.decomposition_carbon_column_level_pool, dtype=jnp.float64)
    nitrogen = jnp.asarray(
        inputs.decomposition_nitrogen_column_level_pool,
        dtype=jnp.float64,
    )

    carbon_pool = _preserve_filtered(
        _integrate_levels(carbon, thickness, all_column),
        previous.decomposition_carbon_column_pool,
        all_column,
    )
    carbon_pool_1m = _preserve_filtered(
        _integrate_one_meter(carbon, thickness, interface, all_column),
        previous.decomposition_carbon_one_meter_column_pool,
        all_column,
    )
    soil_carbon_level = _preserve_filtered(
        jnp.where(
            all_column[:, None],
            _sum_selected_pools(carbon, SOIL_POOL_INDICES),
            jnp.float64(0.0),
        ),
        previous.decomposition_soil_carbon_column_level,
        all_column,
    )
    carbon_truncation = _preserve_filtered(
        _integrate_levels(
            jnp.asarray(inputs.carbon_truncation_column_level, dtype=jnp.float64),
            thickness,
            all_column,
        ),
        previous.carbon_truncation_column,
        all_column,
    )
    litter_carbon = _preserve_filtered(
        _sum_selected_pools(carbon_pool, LITTER_POOL_INDICES),
        previous.total_litter_carbon_column,
        all_column,
    )
    litter_carbon_1m = _preserve_filtered(
        _sum_selected_pools(carbon_pool_1m, LITTER_POOL_INDICES),
        previous.total_litter_carbon_one_meter_column,
        all_column,
    )
    soil_carbon = _preserve_filtered(
        _sum_selected_pools(carbon_pool, SOIL_POOL_INDICES),
        previous.total_soil_carbon_column,
        all_column,
    )
    soil_carbon_1m = _preserve_filtered(
        _sum_selected_pools(carbon_pool_1m, SOIL_POOL_INDICES),
        previous.total_soil_carbon_one_meter_column,
        all_column,
    )
    cwd_carbon = _preserve_filtered(
        _sum_selected_pools(carbon_pool, CWD_POOL_INDICES),
        previous.coarse_woody_debris_carbon_column,
        all_column,
    )

    nitrate = _preserve_filtered(
        _integrate_levels(
            jnp.asarray(inputs.nitrate_nitrogen_column_level, dtype=jnp.float64),
            thickness,
            all_column,
        ),
        previous.nitrate_nitrogen_column,
        all_column,
    )
    ammonium = _preserve_filtered(
        _integrate_levels(
            jnp.asarray(inputs.ammonium_nitrogen_column_level, dtype=jnp.float64),
            thickness,
            all_column,
        ),
        previous.ammonium_nitrogen_column,
        all_column,
    )
    if not parameters.use_nitrification_denitrification:
        nitrate = previous.nitrate_nitrogen_column
        ammonium = previous.ammonium_nitrogen_column
    nitrogen_pool = _preserve_filtered(
        _integrate_levels(nitrogen, thickness, all_column),
        previous.decomposition_nitrogen_column_pool,
        all_column,
    )
    nitrogen_pool_1m = _preserve_filtered(
        _integrate_one_meter(nitrogen, thickness, interface, all_column),
        previous.decomposition_nitrogen_one_meter_column_pool,
        all_column,
    )
    soil_nitrogen_level = _preserve_filtered(
        _sum_selected_pools(nitrogen, SOIL_POOL_INDICES),
        previous.decomposition_soil_nitrogen_column_level,
        all_column,
    )
    litter_nitrogen = _preserve_filtered(
        _sum_selected_pools(nitrogen_pool, LITTER_POOL_INDICES),
        previous.total_litter_nitrogen_column,
        all_column,
    )
    litter_nitrogen_1m = _preserve_filtered(
        _sum_selected_pools(nitrogen_pool_1m, LITTER_POOL_INDICES),
        previous.total_litter_nitrogen_one_meter_column,
        all_column,
    )
    soil_nitrogen = _preserve_filtered(
        _sum_selected_pools(nitrogen_pool, SOIL_POOL_INDICES),
        previous.total_soil_nitrogen_column,
        all_column,
    )
    soil_nitrogen_1m = _preserve_filtered(
        _sum_selected_pools(nitrogen_pool_1m, SOIL_POOL_INDICES),
        previous.total_soil_nitrogen_one_meter_column,
        all_column,
    )
    cwd_nitrogen = _preserve_filtered(
        _sum_selected_pools(nitrogen_pool, CWD_POOL_INDICES),
        previous.coarse_woody_debris_nitrogen_column,
        all_column,
    )
    mineral_nitrogen = _preserve_filtered(
        _integrate_levels(
            jnp.asarray(inputs.mineral_nitrogen_column_level, dtype=jnp.float64),
            thickness,
            all_column,
        ),
        previous.mineral_nitrogen_column,
        all_column,
    )
    nitrogen_truncation = _preserve_filtered(
        _integrate_levels(
            jnp.asarray(inputs.nitrogen_truncation_column_level, dtype=jnp.float64),
            thickness,
            all_column,
        ),
        previous.nitrogen_truncation_column,
        all_column,
    )

    displayed_carbon = _masked_component_sum(
        jnp.asarray(inputs.displayed_carbon_patch_component, dtype=jnp.float64),
        soil_patch,
    )
    stored_carbon = _masked_component_sum(
        jnp.asarray(inputs.stored_carbon_patch_component, dtype=jnp.float64),
        soil_patch,
    )
    crop_patch = soil_patch & (
        jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
        >= jnp.int32(parameters.first_prognostic_crop_pft)
    )
    if parameters.use_crop:
        displayed_carbon = jnp.where(
            crop_patch,
            fortran_add(displayed_carbon, inputs.grain_carbon_patch),
            displayed_carbon,
        )
        crop_stored_carbon = fortran_ordered_sum(
            jnp.stack(
                (
                    stored_carbon,
                    jnp.asarray(inputs.grain_carbon_storage_patch, dtype=jnp.float64),
                    jnp.asarray(inputs.grain_carbon_transfer_patch, dtype=jnp.float64),
                ),
                axis=0,
            )
        )
        stored_carbon = jnp.where(
            crop_patch,
            crop_stored_carbon,
            stored_carbon,
        )
    displayed_carbon = _preserve_filtered(
        displayed_carbon,
        previous.displayed_vegetation_carbon_patch,
        soil_patch,
    )
    stored_carbon = _preserve_filtered(
        stored_carbon,
        previous.stored_vegetation_carbon_patch,
        soil_patch,
    )
    total_vegetation_carbon_patch = jnp.where(
        soil_patch,
        fortran_add(displayed_carbon, stored_carbon),
        previous.total_vegetation_carbon_patch,
    )
    total_carbon_patch = fortran_ordered_sum(
        jnp.stack(
            (
                total_vegetation_carbon_patch,
                jnp.asarray(
                    inputs.excess_maintenance_respiration_carbon_patch,
                    dtype=jnp.float64,
                ),
                jnp.asarray(inputs.patch_carbon_truncation, dtype=jnp.float64),
            ),
            axis=0,
        )
    )
    if parameters.use_crop:
        total_carbon_patch = fortran_ordered_sum(
            jnp.stack(
                (
                    total_carbon_patch,
                    jnp.asarray(inputs.crop_seed_carbon_deficit_patch, dtype=jnp.float64),
                    jnp.asarray(
                        inputs.excess_maintenance_respiration_carbon_loss_patch,
                        dtype=jnp.float64,
                    ),
                ),
                axis=0,
            )
        )
    total_carbon_patch = jnp.where(
        soil_patch,
        total_carbon_patch,
        previous.total_carbon_patch,
    )
    displayed_components = jnp.asarray(
        inputs.displayed_carbon_patch_component,
        dtype=jnp.float64,
    )
    wood_carbon = fortran_ordered_sum(
        jnp.stack(
            (
                displayed_components[:, 3],
                displayed_components[:, 2],
                displayed_components[:, 5],
                displayed_components[:, 4],
            ),
            axis=0,
        )
    )
    wood_carbon = jnp.where(
        soil_patch,
        wood_carbon,
        previous.wood_carbon_patch,
    )

    p2c_arguments = (
        jnp.asarray(inputs.patch_column_index, dtype=jnp.int32),
        jnp.asarray(inputs.patch_weight_on_column, dtype=jnp.float64),
        jnp.asarray(inputs.patch_active, dtype=jnp.bool_),
        soil_column,
    )
    p2c_keywords = {"maximum_patches_per_column": parameters.maximum_patches_per_column}
    total_vegetation_carbon_column = _preserve_filtered(
        patch_to_column_average(
            total_vegetation_carbon_patch,
            *p2c_arguments,
            **p2c_keywords,
        ),
        previous.total_vegetation_carbon_column,
        soil_column,
    )
    total_patch_carbon_column = _preserve_filtered(
        patch_to_column_average(
            total_carbon_patch,
            *p2c_arguments,
            **p2c_keywords,
        ),
        previous.total_patch_carbon_column,
        soil_column,
    )
    total_ecosystem_carbon = jnp.where(
        all_column,
        fortran_ordered_sum(
            jnp.stack(
                (
                    cwd_carbon,
                    litter_carbon,
                    soil_carbon,
                    total_vegetation_carbon_column,
                ),
                axis=0,
            )
        ),
        previous.total_ecosystem_carbon_column,
    )
    total_carbon = jnp.where(
        all_column,
        fortran_ordered_sum(
            jnp.stack(
                (
                    total_patch_carbon_column,
                    cwd_carbon,
                    litter_carbon,
                    soil_carbon,
                    carbon_truncation,
                ),
                axis=0,
            )
        ),
        previous.total_carbon_column,
    )

    displayed_nitrogen = _masked_component_sum(
        jnp.asarray(inputs.displayed_nitrogen_patch_component, dtype=jnp.float64),
        soil_patch,
    )
    stored_nitrogen = _masked_component_sum(
        jnp.asarray(inputs.stored_nitrogen_patch_component, dtype=jnp.float64),
        soil_patch,
    )
    if parameters.use_crop:
        displayed_nitrogen = jnp.where(
            crop_patch,
            fortran_add(displayed_nitrogen, inputs.grain_nitrogen_patch),
            displayed_nitrogen,
        )
        crop_stored_nitrogen = fortran_ordered_sum(
            jnp.stack(
                (
                    stored_nitrogen,
                    jnp.asarray(inputs.grain_nitrogen_storage_patch, dtype=jnp.float64),
                    jnp.asarray(inputs.grain_nitrogen_transfer_patch, dtype=jnp.float64),
                    jnp.asarray(inputs.crop_seed_nitrogen_deficit_patch, dtype=jnp.float64),
                ),
                axis=0,
            )
        )
        stored_nitrogen = jnp.where(
            crop_patch,
            crop_stored_nitrogen,
            stored_nitrogen,
        )
    displayed_nitrogen = _preserve_filtered(
        displayed_nitrogen,
        previous.displayed_vegetation_nitrogen_patch,
        soil_patch,
    )
    stored_nitrogen = _preserve_filtered(
        stored_nitrogen,
        previous.stored_vegetation_nitrogen_patch,
        soil_patch,
    )
    total_vegetation_nitrogen_patch = jnp.where(
        soil_patch,
        fortran_add(displayed_nitrogen, stored_nitrogen),
        previous.total_vegetation_nitrogen_patch,
    )
    total_nitrogen_patch = jnp.where(
        soil_patch,
        fortran_add(
            total_vegetation_nitrogen_patch,
            jnp.asarray(inputs.patch_nitrogen_truncation, dtype=jnp.float64),
        ),
        previous.total_nitrogen_patch,
    )
    total_vegetation_nitrogen_column = _preserve_filtered(
        patch_to_column_average(
            total_vegetation_nitrogen_patch,
            *p2c_arguments,
            **p2c_keywords,
        ),
        previous.total_vegetation_nitrogen_column,
        soil_column,
    )
    total_patch_nitrogen_column = _preserve_filtered(
        patch_to_column_average(
            total_nitrogen_patch,
            *p2c_arguments,
            **p2c_keywords,
        ),
        previous.total_patch_nitrogen_column,
        soil_column,
    )
    total_ecosystem_nitrogen = jnp.where(
        all_column,
        fortran_ordered_sum(
            jnp.stack(
                (
                    cwd_nitrogen,
                    litter_nitrogen,
                    soil_nitrogen,
                    mineral_nitrogen,
                    total_vegetation_nitrogen_column,
                ),
                axis=0,
            )
        ),
        previous.total_ecosystem_nitrogen_column,
    )
    total_nitrogen = jnp.where(
        all_column,
        fortran_ordered_sum(
            jnp.stack(
                (
                    total_patch_nitrogen_column,
                    cwd_nitrogen,
                    litter_nitrogen,
                    soil_nitrogen,
                    mineral_nitrogen,
                    nitrogen_truncation,
                ),
                axis=0,
            )
        ),
        previous.total_nitrogen_column,
    )

    beginning_carbon = jnp.where(
        soil_column,
        total_carbon,
        jnp.asarray(state.beginning_carbon_balance_column, dtype=jnp.float64),
    )
    beginning_nitrogen = jnp.where(
        soil_column,
        total_nitrogen,
        jnp.asarray(state.beginning_nitrogen_balance_column, dtype=jnp.float64),
    )
    summaries = CarbonNitrogenStateSummaries(
        carbon_pool,
        carbon_pool_1m,
        soil_carbon_level,
        carbon_truncation,
        litter_carbon,
        litter_carbon_1m,
        soil_carbon,
        soil_carbon_1m,
        cwd_carbon,
        nitrate,
        ammonium,
        nitrogen_pool,
        nitrogen_pool_1m,
        soil_nitrogen_level,
        mineral_nitrogen,
        nitrogen_truncation,
        litter_nitrogen,
        litter_nitrogen_1m,
        soil_nitrogen,
        soil_nitrogen_1m,
        cwd_nitrogen,
        displayed_carbon,
        stored_carbon,
        total_vegetation_carbon_patch,
        total_carbon_patch,
        wood_carbon,
        total_vegetation_carbon_column,
        total_patch_carbon_column,
        total_ecosystem_carbon,
        total_carbon,
        displayed_nitrogen,
        stored_nitrogen,
        total_vegetation_nitrogen_patch,
        total_nitrogen_patch,
        total_vegetation_nitrogen_column,
        total_patch_nitrogen_column,
        total_ecosystem_nitrogen,
        total_nitrogen,
    )
    return CarbonNitrogenBalanceInitializationResult(
        state=CarbonNitrogenBalanceInitializationState(
            summaries,
            beginning_carbon,
            beginning_nitrogen,
        ),
        diagnostics=CarbonNitrogenBalanceInitializationDiagnostics(
            summarized_column_count=jnp.sum(all_column, dtype=jnp.int32),
            balanced_column_count=jnp.sum(soil_column, dtype=jnp.int32),
            summarized_patch_count=jnp.sum(soil_patch, dtype=jnp.int32),
        ),
    )


__all__ = [
    "CARBON_DISPLAYED_COMPONENTS",
    "CARBON_STORED_COMPONENTS",
    "CWD_POOL_INDICES",
    "DEFAULT_CARBON_NITROGEN_BALANCE_INITIALIZATION_PARAMETERS",
    "LITTER_POOL_INDICES",
    "NITROGEN_DISPLAYED_COMPONENTS",
    "NITROGEN_STORED_COMPONENTS",
    "NUMBER_OF_DECOMPOSITION_LEVELS",
    "NUMBER_OF_DECOMPOSITION_POOLS",
    "SOIL_POOL_INDICES",
    "CarbonNitrogenBalanceInitializationDiagnostics",
    "CarbonNitrogenBalanceInitializationInputs",
    "CarbonNitrogenBalanceInitializationParameters",
    "CarbonNitrogenBalanceInitializationResult",
    "CarbonNitrogenBalanceInitializationState",
    "CarbonNitrogenStateSummaries",
    "initialize_carbon_nitrogen_balance",
]
