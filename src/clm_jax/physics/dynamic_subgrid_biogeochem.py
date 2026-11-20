
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.io.restart import LANDUNIT_CROP, LANDUNIT_SOIL, SubgridTopology
from clm_jax.physics.annual_flux_dribbler import (
    AnnualFluxDribblerState,
    annual_flux_dribbler_step,
)
from clm_jax.physics.bgc_timestep_initialization import (
    DynamicLandcoverCarbonFluxState,
    DynamicLandcoverNitrogenFluxState,
)
from clm_jax.physics.cn_precision_control import CnPrecisionControlState
from clm_jax.physics.cn_zero_manifest import (
    NUMBER_OF_DECOMPOSITION_LEVELS,
    NUMBER_OF_DECOMPOSITION_POOLS,
)
from clm_jax.physics.dynamic_subgrid_weights import (
    ColumnWeightTransition,
    PatchWeightTransition,
    patch_grew,
    patch_initiating,
    patch_old_weight_was_zero,
    update_column_state_no_special_handling,
    update_patch_state,
    update_patch_state_partition_flux_by_type,
)
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
)
from clm_jax.physics.soil_biogeochem_precision_control import (
    SoilBiogeochemPrecisionControlState,
)

CONFIGURED_ACTIVE_DECOMPOSITION_LEVELS = 20
LEAF_CARBON_SEED = 1.0
DEAD_STEM_CARBON_SEED = 0.1
METABOLIC_LITTER_POOL_INDEX = 0
CELLULOSE_LITTER_POOL_INDEX = 1
LIGNIN_LITTER_POOL_INDEX = 2
COARSE_WOODY_DEBRIS_POOL_INDEX = 3
DYNAMIC_INITIATING_CANOPY_FIELDS = (
    "sunlit_leaf_area_index",
    "shaded_leaf_area_index",
)
DYNAMIC_INITIATING_CN_VEGETATION_FIELDS = (
    "dormant_flag",
    "days_active",
    "onset_flag",
    "onset_counter",
    "onset_gddflag",
    "onset_fdd",
    "onset_gdd",
    "onset_swi",
    "offset_flag",
    "offset_counter",
    "offset_fdd",
    "offset_swi",
    "long_growing_season_factor",
    "background_litterfall_rate",
    "background_transfer_growth_rate",
    "annual_average_2m_temperature",
    "temporary_average_2m_temperature",
    "carbon_allometry",
    "nitrogen_allometry",
    "temporary_sum_potential_gpp",
    "annual_sum_potential_gpp",
    "temporary_maximum_retranslocated_nitrogen",
    "annual_maximum_retranslocated_nitrogen",
    "downregulation",
)
DYNAMIC_INITIATING_CARBON_FLUX_FIELDS = (
    "excess_maintenance_respiration_pool_recovery",
    "plant_carbon_allocation",
    "excess_carbon_flux",
    "previous_leaf_carbon_to_litter",
    "previous_fine_root_carbon_to_litter",
    "available_carbon",
    "gpp_before_downregulation",
    "temporary_sum_npp",
    "annual_sum_npp",
)
DYNAMIC_INITIATING_NITROGEN_FLUX_FIELDS = (
    "plant_nitrogen_demand",
    "available_retranslocated_nitrogen",
    "plant_nitrogen_allocation",
)
DYNAMIC_INITIATING_PHOTOSYNTHESIS_FIELDS = (
    "sunlit_leaf_photosynthesis",
    "shaded_leaf_photosynthesis",
)


class DynamicVegetationPatchInputs(NamedTuple):
    soil_patch_with_inactive: jnp.ndarray
    vegetation_type_patch: jnp.ndarray


class DynamicVegetationPatchParameters(NamedTuple):
    conversion_fraction_pft: jnp.ndarray
    woody_pft: jnp.ndarray
    evergreen_pft: jnp.ndarray
    leaf_carbon_nitrogen_ratio_pft: jnp.ndarray
    deadwood_carbon_nitrogen_ratio_pft: jnp.ndarray
    no_vegetation_type: int | jnp.ndarray = 0


class DynamicVegetationPatchState(NamedTuple):
    carbon_nitrogen: CnPrecisionControlState
    excess_maintenance_respiration_loss_carbon_patch: jnp.ndarray


class DynamicVegetationPatchFluxes(NamedTuple):
    conversion_carbon_gridcell_area_patch: jnp.ndarray
    wood_product_carbon_gridcell_area_patch: jnp.ndarray
    crop_product_carbon_gridcell_area_patch: jnp.ndarray
    fine_root_carbon_to_litter_column_area_patch: jnp.ndarray
    live_coarse_root_carbon_to_litter_column_area_patch: jnp.ndarray
    dead_coarse_root_carbon_to_litter_column_area_patch: jnp.ndarray
    leaf_carbon_seed_gridcell_area_patch: jnp.ndarray
    dead_stem_carbon_seed_gridcell_area_patch: jnp.ndarray
    conversion_nitrogen_gridcell_area_patch: jnp.ndarray
    wood_product_nitrogen_gridcell_area_patch: jnp.ndarray
    crop_product_nitrogen_gridcell_area_patch: jnp.ndarray
    fine_root_nitrogen_to_litter_column_area_patch: jnp.ndarray
    live_coarse_root_nitrogen_to_litter_column_area_patch: jnp.ndarray
    dead_coarse_root_nitrogen_to_litter_column_area_patch: jnp.ndarray
    leaf_nitrogen_seed_gridcell_area_patch: jnp.ndarray
    dead_stem_nitrogen_seed_gridcell_area_patch: jnp.ndarray


class DynamicVegetationPatchAdjustmentResult(NamedTuple):
    state: DynamicVegetationPatchState
    fluxes: DynamicVegetationPatchFluxes


class DynamicPatchStateUpdateInputs(NamedTuple):
    soil_column_with_inactive: jnp.ndarray
    timestep_seconds: float | jnp.ndarray


class DynamicPatchStateUpdateState(NamedTuple):
    soil_carbon_nitrogen: SoilBiogeochemPrecisionControlState
    seed_carbon_gridcell: jnp.ndarray
    seed_nitrogen_gridcell: jnp.ndarray


class DynamicPatchStateUpdateResult(NamedTuple):
    state: DynamicPatchStateUpdateState
    updated_column_level_count: jnp.ndarray
    updated_gridcell_count: jnp.ndarray


class DynamicPatchFluxAggregationInputs(NamedTuple):
    timestep_seconds: float | jnp.ndarray
    fine_root_profile_patch_level: jnp.ndarray
    coarse_root_profile_patch_level: jnp.ndarray


class DynamicPatchFluxAggregationParameters(NamedTuple):
    fine_root_metabolic_fraction_pft: jnp.ndarray
    fine_root_cellulose_fraction_pft: jnp.ndarray
    fine_root_lignin_fraction_pft: jnp.ndarray


class DynamicVegetationPatchRateFluxes(NamedTuple):
    seed_carbon_to_leaf_patch: jnp.ndarray
    seed_carbon_to_dead_stem_patch: jnp.ndarray
    conversion_carbon_patch: jnp.ndarray
    wood_product_carbon_gain_patch: jnp.ndarray
    crop_product_carbon_gain_patch: jnp.ndarray
    seed_nitrogen_to_leaf_patch: jnp.ndarray
    seed_nitrogen_to_dead_stem_patch: jnp.ndarray
    conversion_nitrogen_patch: jnp.ndarray
    wood_product_nitrogen_gain_patch: jnp.ndarray
    crop_product_nitrogen_gain_patch: jnp.ndarray


class DynamicPatchFluxAggregationResult(NamedTuple):
    carbon_flux: DynamicLandcoverCarbonFluxState
    nitrogen_flux: DynamicLandcoverNitrogenFluxState
    patch_flux: DynamicVegetationPatchRateFluxes


class DynamicPatchInitializationClock(NamedTuple):
    is_beginning_current_year: bool | jnp.ndarray
    is_first_step: bool | jnp.ndarray
    previous_year_fraction: float | jnp.ndarray
    current_year_fraction: float | jnp.ndarray
    timestep_seconds: float | jnp.ndarray
    days_per_year: float | jnp.ndarray
    allows_nonannual_weight_change: bool | jnp.ndarray = False


class DynamicPatchInitializationState(NamedTuple):
    canopy_patch_field: jnp.ndarray
    carbon_nitrogen_vegetation_patch_field: jnp.ndarray
    annual_average_2m_temperature_column: jnp.ndarray
    carbon_flux_patch_field: jnp.ndarray
    nitrogen_flux_patch_field: jnp.ndarray
    photosynthesis_patch_field: jnp.ndarray
    weight_change_dribbler: AnnualFluxDribblerState
    smoothed_weight_change_patch: jnp.ndarray


class DynamicPatchInitializationResult(NamedTuple):
    state: DynamicPatchInitializationState
    raw_weight_change_patch: jnp.ndarray
    initiating_patch: jnp.ndarray
    illegal_nonannual_weight_change_patch: jnp.ndarray


class _SeedAmounts(NamedTuple):
    leaf: jnp.ndarray
    leaf_storage: jnp.ndarray
    leaf_transfer: jnp.ndarray
    dead_stem: jnp.ndarray


class DynamicSoilColumnAdjustmentResult(NamedTuple):
    state: SoilBiogeochemPrecisionControlState
    carbon_balance_adjustment_column: jnp.ndarray
    nitrogen_balance_adjustment_column: jnp.ndarray
    nitrate_balance_adjustment_column: jnp.ndarray
    ammonium_balance_adjustment_column: jnp.ndarray


def _compute_seed_amounts(
    inputs: DynamicVegetationPatchInputs,
    transition: PatchWeightTransition,
    parameters: DynamicVegetationPatchParameters,
    leaf: jnp.ndarray,
    leaf_storage: jnp.ndarray,
    leaf_transfer: jnp.ndarray,
    *,
    nitrogen: bool,
) -> _SeedAmounts:

    vegetation_type = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    selected = jnp.asarray(inputs.soil_patch_with_inactive, dtype=jnp.bool_)
    growing = selected & patch_grew(transition)
    ignore_current = patch_old_weight_was_zero(transition)
    leaf_value = jnp.asarray(leaf, dtype=jnp.float64)
    storage_value = jnp.asarray(leaf_storage, dtype=jnp.float64)
    transfer_value = jnp.asarray(leaf_transfer, dtype=jnp.float64)
    total = fortran_add(fortran_add(leaf_value, storage_value), transfer_value)
    use_default = (total == jnp.float64(0.0)) | ignore_current
    safe_total = jnp.where(use_default, jnp.float64(1.0), total)
    evergreen = jnp.asarray(parameters.evergreen_pft, dtype=jnp.float64)[
        vegetation_type
    ]
    proportion_leaf = jnp.where(
        use_default,
        jnp.where(evergreen == jnp.float64(1.0), 1.0, 0.0),
        fortran_divide(leaf_value, safe_total),
    )
    proportion_storage = jnp.where(
        use_default,
        jnp.where(evergreen == jnp.float64(1.0), 0.0, 1.0),
        fortran_divide(storage_value, safe_total),
    )
    proportion_transfer = jnp.where(
        use_default,
        jnp.float64(0.0),
        fortran_divide(transfer_value, safe_total),
    )
    vegetated = vegetation_type != jnp.asarray(
        parameters.no_vegetation_type,
        dtype=jnp.int32,
    )
    woody = (
        jnp.asarray(parameters.woody_pft, dtype=jnp.float64)[vegetation_type]
        == jnp.float64(1.0)
    )
    if nitrogen:
        leaf_multiplier = fortran_divide(
            jnp.float64(1.0),
            jnp.asarray(
                parameters.leaf_carbon_nitrogen_ratio_pft,
                dtype=jnp.float64,
            )[vegetation_type],
        )
        dead_stem_multiplier = fortran_divide(
            jnp.float64(1.0),
            jnp.asarray(
                parameters.deadwood_carbon_nitrogen_ratio_pft,
                dtype=jnp.float64,
            )[vegetation_type],
        )
    else:
        leaf_multiplier = jnp.ones_like(leaf_value)
        dead_stem_multiplier = jnp.ones_like(leaf_value)
    leaf_seed = jnp.where(
        vegetated,
        fortran_multiply(jnp.float64(LEAF_CARBON_SEED), leaf_multiplier),
        jnp.float64(0.0),
    )
    dead_stem_seed = jnp.where(
        vegetated & woody,
        fortran_multiply(
            jnp.float64(DEAD_STEM_CARBON_SEED),
            dead_stem_multiplier,
        ),
        jnp.float64(0.0),
    )
    return _SeedAmounts(
        leaf=jnp.where(
            growing,
            fortran_multiply(leaf_seed, proportion_leaf),
            jnp.float64(0.0),
        ),
        leaf_storage=jnp.where(
            growing,
            fortran_multiply(leaf_seed, proportion_storage),
            jnp.float64(0.0),
        ),
        leaf_transfer=jnp.where(
            growing,
            fortran_multiply(leaf_seed, proportion_transfer),
            jnp.float64(0.0),
        ),
        dead_stem=jnp.where(growing, dead_stem_seed, jnp.float64(0.0)),
    )


def _validate_patch_inputs(
    transition: PatchWeightTransition,
    inputs: DynamicVegetationPatchInputs,
    parameters: DynamicVegetationPatchParameters,
    state: DynamicVegetationPatchState,
) -> int:
    number_of_patches = transition.weight_change.shape[0]
    patch_shape = (number_of_patches,)
    if inputs.soil_patch_with_inactive.shape != patch_shape:
        raise ValueError("soil_patch_with_inactive must match patch weights")
    if inputs.vegetation_type_patch.shape != patch_shape:
        raise ValueError("vegetation_type_patch must match patch weights")
    carbon_nitrogen = state.carbon_nitrogen
    expected_shapes = (
        (carbon_nitrogen.paired_carbon_patch, (number_of_patches, 22)),
        (carbon_nitrogen.paired_nitrogen_patch, (number_of_patches, 22)),
        (carbon_nitrogen.carbon_only_patch, (number_of_patches, 4)),
        (carbon_nitrogen.nitrogen_only_patch, (number_of_patches, 2)),
        (carbon_nitrogen.carbon_truncation_patch, patch_shape),
        (carbon_nitrogen.nitrogen_truncation_patch, patch_shape),
        (state.excess_maintenance_respiration_loss_carbon_patch, patch_shape),
    )
    if any(value.shape != shape for value, shape in expected_shapes):
        raise ValueError("dynamic vegetation C/N state has incompatible patch shapes")
    parameter_arrays = (
        parameters.conversion_fraction_pft,
        parameters.woody_pft,
        parameters.evergreen_pft,
        parameters.leaf_carbon_nitrogen_ratio_pft,
        parameters.deadwood_carbon_nitrogen_ratio_pft,
    )
    if any(len(value.shape) != 1 for value in parameter_arrays):
        raise ValueError("dynamic vegetation PFT parameters must be vectors")
    parameter_count = parameter_arrays[0].shape[0]
    if any(value.shape != (parameter_count,) for value in parameter_arrays):
        raise ValueError("dynamic vegetation PFT parameter vectors must match")
    return number_of_patches


def initialize_dynamic_vegetation_patches(
    topology: SubgridTopology,
    transition: PatchWeightTransition,
    clock: DynamicPatchInitializationClock,
    state: DynamicPatchInitializationState,
) -> DynamicPatchInitializationResult:

    patch_landunit_type = jnp.asarray(
        topology.patch.landunit_type,
        dtype=jnp.int32,
    )
    patch_column = jnp.asarray(topology.patch.column, dtype=jnp.int32)
    number_of_patches = patch_landunit_type.shape[0]
    number_of_columns = topology.column.gridcell.shape[0]
    if patch_column.shape != (number_of_patches,):
        raise ValueError("patch column topology must match patches")
    if transition.weight_change.shape != (number_of_patches,):
        raise ValueError("patch transition must match topology")
    expected_patch_shapes = (
        (
            state.canopy_patch_field,
            (number_of_patches, len(DYNAMIC_INITIATING_CANOPY_FIELDS)),
        ),
        (
            state.carbon_nitrogen_vegetation_patch_field,
            (number_of_patches, len(DYNAMIC_INITIATING_CN_VEGETATION_FIELDS)),
        ),
        (
            state.carbon_flux_patch_field,
            (number_of_patches, len(DYNAMIC_INITIATING_CARBON_FLUX_FIELDS)),
        ),
        (
            state.nitrogen_flux_patch_field,
            (number_of_patches, len(DYNAMIC_INITIATING_NITROGEN_FLUX_FIELDS)),
        ),
        (
            state.photosynthesis_patch_field,
            (number_of_patches, len(DYNAMIC_INITIATING_PHOTOSYNTHESIS_FIELDS)),
        ),
        (state.smoothed_weight_change_patch, (number_of_patches,)),
        (state.weight_change_dribbler.amount_to_dribble, (number_of_patches,)),
        (
            state.weight_change_dribbler.amount_from_this_timestep,
            (number_of_patches,),
        ),
    )
    if any(value.shape != shape for value, shape in expected_patch_shapes):
        raise ValueError("dynamic initiating-patch state has incompatible shapes")
    if state.annual_average_2m_temperature_column.shape != (number_of_columns,):
        raise ValueError("annual column temperature must match column topology")

    soil_or_crop = (patch_landunit_type == jnp.int32(LANDUNIT_SOIL)) | (
        patch_landunit_type == jnp.int32(LANDUNIT_CROP)
    )
    initiating = soil_or_crop & patch_initiating(transition)
    initiating_field = initiating[:, None]
    canopy = jnp.where(
        initiating_field,
        jnp.zeros_like(state.canopy_patch_field, dtype=jnp.float64),
        jnp.asarray(state.canopy_patch_field, dtype=jnp.float64),
    )
    vegetation_reset = jnp.zeros_like(
        state.carbon_nitrogen_vegetation_patch_field,
        dtype=jnp.float64,
    )
    vegetation_reset = vegetation_reset.at[:, 0].set(jnp.float64(1.0))
    vegetation_reset = vegetation_reset.at[:, 15].set(
        jnp.asarray(
            state.annual_average_2m_temperature_column,
            dtype=jnp.float64,
        )[patch_column]
    )
    vegetation = jnp.where(
        initiating_field,
        vegetation_reset,
        jnp.asarray(
            state.carbon_nitrogen_vegetation_patch_field,
            dtype=jnp.float64,
        ),
    )
    carbon_flux = jnp.where(
        initiating_field,
        jnp.zeros_like(state.carbon_flux_patch_field, dtype=jnp.float64),
        jnp.asarray(state.carbon_flux_patch_field, dtype=jnp.float64),
    )
    nitrogen_flux = jnp.where(
        initiating_field,
        jnp.zeros_like(state.nitrogen_flux_patch_field, dtype=jnp.float64),
        jnp.asarray(state.nitrogen_flux_patch_field, dtype=jnp.float64),
    )
    photosynthesis = jnp.where(
        initiating_field,
        jnp.zeros_like(state.photosynthesis_patch_field, dtype=jnp.float64),
        jnp.asarray(state.photosynthesis_patch_field, dtype=jnp.float64),
    )

    raw_weight_change = jnp.where(
        soil_or_crop,
        transition.weight_change,
        jnp.float64(0.0),
    )
    dribbler = annual_flux_dribbler_step(
        state.weight_change_dribbler,
        raw_weight_change,
        is_beginning_current_year=clock.is_beginning_current_year,
        is_first_step=clock.is_first_step,
        previous_year_fraction=clock.previous_year_fraction,
        current_year_fraction=clock.current_year_fraction,
        time_step_seconds=clock.timestep_seconds,
        days_per_year=clock.days_per_year,
        allows_nonannual_delta=clock.allows_nonannual_weight_change,
    )
    return DynamicPatchInitializationResult(
        state=DynamicPatchInitializationState(
            canopy_patch_field=canopy,
            carbon_nitrogen_vegetation_patch_field=vegetation,
            annual_average_2m_temperature_column=(
                state.annual_average_2m_temperature_column
            ),
            carbon_flux_patch_field=carbon_flux,
            nitrogen_flux_patch_field=nitrogen_flux,
            photosynthesis_patch_field=photosynthesis,
            weight_change_dribbler=dribbler.state,
            smoothed_weight_change_patch=dribbler.dribbled_delta,
        ),
        raw_weight_change_patch=raw_weight_change,
        initiating_patch=initiating,
        illegal_nonannual_weight_change_patch=dribbler.illegal_nonannual_delta,
    )


def dynamic_vegetation_patch_adjustments(
    transition: PatchWeightTransition,
    inputs: DynamicVegetationPatchInputs,
    parameters: DynamicVegetationPatchParameters,
    state: DynamicVegetationPatchState,
) -> DynamicVegetationPatchAdjustmentResult:

    number_of_patches = _validate_patch_inputs(
        transition,
        inputs,
        parameters,
        state,
    )
    selected = jnp.asarray(inputs.soil_patch_with_inactive, dtype=jnp.bool_)
    vegetation_type = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    carbon = jnp.asarray(state.carbon_nitrogen.paired_carbon_patch, dtype=jnp.float64)
    nitrogen = jnp.asarray(
        state.carbon_nitrogen.paired_nitrogen_patch,
        dtype=jnp.float64,
    )
    carbon_only = jnp.asarray(state.carbon_nitrogen.carbon_only_patch, dtype=jnp.float64)
    nitrogen_only = jnp.asarray(
        state.carbon_nitrogen.nitrogen_only_patch,
        dtype=jnp.float64,
    )
    carbon_truncation = jnp.asarray(
        state.carbon_nitrogen.carbon_truncation_patch,
        dtype=jnp.float64,
    )
    nitrogen_truncation = jnp.asarray(
        state.carbon_nitrogen.nitrogen_truncation_patch,
        dtype=jnp.float64,
    )
    xsmrpool_loss = jnp.asarray(
        state.excess_maintenance_respiration_loss_carbon_patch,
        dtype=jnp.float64,
    )
    zeros = jnp.zeros(number_of_patches, dtype=jnp.float64)

    carbon_seed = _compute_seed_amounts(
        inputs,
        transition,
        parameters,
        carbon[:, 0],
        carbon[:, 1],
        carbon[:, 2],
        nitrogen=False,
    )
    conversion_carbon = zeros
    wood_product_carbon = zeros
    crop_product_carbon = zeros
    fine_root_carbon = zeros
    live_coarse_root_carbon = zeros
    dead_coarse_root_carbon = zeros
    leaf_carbon_seed = zeros
    dead_stem_carbon_seed = zeros

    for index, seed in zip((0, 1, 2), carbon_seed[:3], strict=True):
        update = update_patch_state(
            transition,
            selected,
            carbon[:, index],
            flux_out_gridcell_area=conversion_carbon,
            seed=seed,
            seed_addition_gridcell_area=leaf_carbon_seed,
        )
        carbon = carbon.at[:, index].set(update.state)
        conversion_carbon = update.flux_out_gridcell_area
        leaf_carbon_seed = update.seed_addition_gridcell_area

    update = update_patch_state(
        transition,
        selected,
        carbon[:, 3],
        flux_out_column_area=fine_root_carbon,
    )
    carbon = carbon.at[:, 3].set(update.state)
    fine_root_carbon = update.flux_out_column_area
    for index in (4, 5, 10, 11, 12):
        update = update_patch_state(
            transition,
            selected,
            carbon[:, index],
            flux_out_gridcell_area=conversion_carbon,
        )
        carbon = carbon.at[:, index].set(update.state)
        conversion_carbon = update.flux_out_gridcell_area

    partition = update_patch_state_partition_flux_by_type(
        transition,
        selected,
        vegetation_type,
        jnp.asarray(parameters.conversion_fraction_pft, dtype=jnp.float64),
        carbon[:, 13],
        conversion_carbon,
        wood_product_carbon,
        seed=carbon_seed.dead_stem,
        seed_addition_gridcell_area=dead_stem_carbon_seed,
    )
    carbon = carbon.at[:, 13].set(partition.state)
    conversion_carbon = partition.flux_one_gridcell_area
    wood_product_carbon = partition.flux_two_gridcell_area
    dead_stem_carbon_seed = partition.seed_addition_gridcell_area
    for index in (14, 15):
        update = update_patch_state(
            transition,
            selected,
            carbon[:, index],
            flux_out_gridcell_area=conversion_carbon,
        )
        carbon = carbon.at[:, index].set(update.state)
        conversion_carbon = update.flux_out_gridcell_area

    for index, root_flux_name in ((16, "live"), (19, "dead")):
        root_flux = (
            live_coarse_root_carbon
            if root_flux_name == "live"
            else dead_coarse_root_carbon
        )
        update = update_patch_state(
            transition,
            selected,
            carbon[:, index],
            flux_out_column_area=root_flux,
        )
        carbon = carbon.at[:, index].set(update.state)
        if root_flux_name == "live":
            live_coarse_root_carbon = update.flux_out_column_area
        else:
            dead_coarse_root_carbon = update.flux_out_column_area
        for storage_index in (index + 1, index + 2):
            update = update_patch_state(
                transition,
                selected,
                carbon[:, storage_index],
                flux_out_gridcell_area=conversion_carbon,
            )
            carbon = carbon.at[:, storage_index].set(update.state)
            conversion_carbon = update.flux_out_gridcell_area

    for index in range(4):
        update = update_patch_state(
            transition,
            selected,
            carbon_only[:, index],
            flux_out_gridcell_area=conversion_carbon,
        )
        carbon_only = carbon_only.at[:, index].set(update.state)
        conversion_carbon = update.flux_out_gridcell_area
    update = update_patch_state(
        transition,
        selected,
        carbon_truncation,
        flux_out_gridcell_area=conversion_carbon,
    )
    carbon_truncation = update.state
    conversion_carbon = update.flux_out_gridcell_area
    update = update_patch_state(
        transition,
        selected,
        carbon[:, 6],
        flux_out_gridcell_area=crop_product_carbon,
    )
    carbon = carbon.at[:, 6].set(update.state)
    crop_product_carbon = update.flux_out_gridcell_area
    for index in (7, 8, 9):
        update = update_patch_state(
            transition,
            selected,
            carbon[:, index],
            flux_out_gridcell_area=conversion_carbon,
        )
        carbon = carbon.at[:, index].set(update.state)
        conversion_carbon = update.flux_out_gridcell_area
    update = update_patch_state(
        transition,
        selected,
        xsmrpool_loss,
        flux_out_gridcell_area=conversion_carbon,
    )
    xsmrpool_loss = update.state
    conversion_carbon = update.flux_out_gridcell_area

    nitrogen_seed = _compute_seed_amounts(
        inputs,
        transition,
        parameters,
        nitrogen[:, 0],
        nitrogen[:, 1],
        nitrogen[:, 2],
        nitrogen=True,
    )
    conversion_nitrogen = zeros
    wood_product_nitrogen = zeros
    crop_product_nitrogen = zeros
    fine_root_nitrogen = zeros
    live_coarse_root_nitrogen = zeros
    dead_coarse_root_nitrogen = zeros
    leaf_nitrogen_seed = zeros
    dead_stem_nitrogen_seed = zeros

    for index, seed in zip((0, 1, 2), nitrogen_seed[:3], strict=True):
        update = update_patch_state(
            transition,
            selected,
            nitrogen[:, index],
            flux_out_gridcell_area=conversion_nitrogen,
            seed=seed,
            seed_addition_gridcell_area=leaf_nitrogen_seed,
        )
        nitrogen = nitrogen.at[:, index].set(update.state)
        conversion_nitrogen = update.flux_out_gridcell_area
        leaf_nitrogen_seed = update.seed_addition_gridcell_area
    update = update_patch_state(
        transition,
        selected,
        nitrogen[:, 3],
        flux_out_column_area=fine_root_nitrogen,
    )
    nitrogen = nitrogen.at[:, 3].set(update.state)
    fine_root_nitrogen = update.flux_out_column_area
    for index in (4, 5, 10, 11, 12):
        update = update_patch_state(
            transition,
            selected,
            nitrogen[:, index],
            flux_out_gridcell_area=conversion_nitrogen,
        )
        nitrogen = nitrogen.at[:, index].set(update.state)
        conversion_nitrogen = update.flux_out_gridcell_area
    partition = update_patch_state_partition_flux_by_type(
        transition,
        selected,
        vegetation_type,
        jnp.asarray(parameters.conversion_fraction_pft, dtype=jnp.float64),
        nitrogen[:, 13],
        conversion_nitrogen,
        wood_product_nitrogen,
        seed=nitrogen_seed.dead_stem,
        seed_addition_gridcell_area=dead_stem_nitrogen_seed,
    )
    nitrogen = nitrogen.at[:, 13].set(partition.state)
    conversion_nitrogen = partition.flux_one_gridcell_area
    wood_product_nitrogen = partition.flux_two_gridcell_area
    dead_stem_nitrogen_seed = partition.seed_addition_gridcell_area
    for index in (14, 15):
        update = update_patch_state(
            transition,
            selected,
            nitrogen[:, index],
            flux_out_gridcell_area=conversion_nitrogen,
        )
        nitrogen = nitrogen.at[:, index].set(update.state)
        conversion_nitrogen = update.flux_out_gridcell_area
    for index, root_flux_name in ((16, "live"), (19, "dead")):
        root_flux = (
            live_coarse_root_nitrogen
            if root_flux_name == "live"
            else dead_coarse_root_nitrogen
        )
        update = update_patch_state(
            transition,
            selected,
            nitrogen[:, index],
            flux_out_column_area=root_flux,
        )
        nitrogen = nitrogen.at[:, index].set(update.state)
        if root_flux_name == "live":
            live_coarse_root_nitrogen = update.flux_out_column_area
        else:
            dead_coarse_root_nitrogen = update.flux_out_column_area
        for storage_index in (index + 1, index + 2):
            update = update_patch_state(
                transition,
                selected,
                nitrogen[:, storage_index],
                flux_out_gridcell_area=conversion_nitrogen,
            )
            nitrogen = nitrogen.at[:, storage_index].set(update.state)
            conversion_nitrogen = update.flux_out_gridcell_area
    for index in range(2):
        update = update_patch_state(
            transition,
            selected,
            nitrogen_only[:, index],
            flux_out_gridcell_area=conversion_nitrogen,
        )
        nitrogen_only = nitrogen_only.at[:, index].set(update.state)
        conversion_nitrogen = update.flux_out_gridcell_area
    update = update_patch_state(
        transition,
        selected,
        nitrogen_truncation,
        flux_out_gridcell_area=conversion_nitrogen,
    )
    nitrogen_truncation = update.state
    conversion_nitrogen = update.flux_out_gridcell_area
    update = update_patch_state(
        transition,
        selected,
        nitrogen[:, 6],
        flux_out_gridcell_area=crop_product_nitrogen,
    )
    nitrogen = nitrogen.at[:, 6].set(update.state)
    crop_product_nitrogen = update.flux_out_gridcell_area
    for index in (7, 8, 9):
        update = update_patch_state(
            transition,
            selected,
            nitrogen[:, index],
            flux_out_gridcell_area=conversion_nitrogen,
        )
        nitrogen = nitrogen.at[:, index].set(update.state)
        conversion_nitrogen = update.flux_out_gridcell_area

    return DynamicVegetationPatchAdjustmentResult(
        state=DynamicVegetationPatchState(
            carbon_nitrogen=CnPrecisionControlState(
                paired_carbon_patch=carbon,
                paired_nitrogen_patch=nitrogen,
                carbon_only_patch=carbon_only,
                nitrogen_only_patch=nitrogen_only,
                carbon_truncation_patch=carbon_truncation,
                nitrogen_truncation_patch=nitrogen_truncation,
            ),
            excess_maintenance_respiration_loss_carbon_patch=xsmrpool_loss,
        ),
        fluxes=DynamicVegetationPatchFluxes(
            conversion_carbon_gridcell_area_patch=conversion_carbon,
            wood_product_carbon_gridcell_area_patch=wood_product_carbon,
            crop_product_carbon_gridcell_area_patch=crop_product_carbon,
            fine_root_carbon_to_litter_column_area_patch=fine_root_carbon,
            live_coarse_root_carbon_to_litter_column_area_patch=(
                live_coarse_root_carbon
            ),
            dead_coarse_root_carbon_to_litter_column_area_patch=(
                dead_coarse_root_carbon
            ),
            leaf_carbon_seed_gridcell_area_patch=leaf_carbon_seed,
            dead_stem_carbon_seed_gridcell_area_patch=dead_stem_carbon_seed,
            conversion_nitrogen_gridcell_area_patch=conversion_nitrogen,
            wood_product_nitrogen_gridcell_area_patch=wood_product_nitrogen,
            crop_product_nitrogen_gridcell_area_patch=crop_product_nitrogen,
            fine_root_nitrogen_to_litter_column_area_patch=fine_root_nitrogen,
            live_coarse_root_nitrogen_to_litter_column_area_patch=(
                live_coarse_root_nitrogen
            ),
            dead_coarse_root_nitrogen_to_litter_column_area_patch=(
                dead_coarse_root_nitrogen
            ),
            leaf_nitrogen_seed_gridcell_area_patch=leaf_nitrogen_seed,
            dead_stem_nitrogen_seed_gridcell_area_patch=dead_stem_nitrogen_seed,
        ),
    )


def apply_dynamic_patch_carbon_nitrogen_fluxes(
    inputs: DynamicPatchStateUpdateInputs,
    carbon_flux: DynamicLandcoverCarbonFluxState,
    nitrogen_flux: DynamicLandcoverNitrogenFluxState,
    state: DynamicPatchStateUpdateState,
    *,
    active_level_count: int = CONFIGURED_ACTIVE_DECOMPOSITION_LEVELS,
) -> DynamicPatchStateUpdateResult:

    soil = state.soil_carbon_nitrogen
    carbon_pool = jnp.asarray(
        soil.carbon_pool_column_level_pool,
        dtype=jnp.float64,
    )
    nitrogen_pool = jnp.asarray(
        soil.nitrogen_pool_column_level_pool,
        dtype=jnp.float64,
    )
    if carbon_pool.shape != nitrogen_pool.shape:
        raise ValueError("carbon and nitrogen pool shapes must match")
    if len(carbon_pool.shape) != 3 or carbon_pool.shape[2] != (
        NUMBER_OF_DECOMPOSITION_POOLS
    ):
        raise ValueError("soil pools must have shape (column, level, 7)")
    number_of_columns, number_of_levels, _ = carbon_pool.shape
    soil_column = jnp.asarray(inputs.soil_column_with_inactive, dtype=jnp.bool_)
    if soil_column.shape != (number_of_columns,):
        raise ValueError("soil_column_with_inactive must match soil columns")
    if not 0 <= active_level_count <= number_of_levels:
        raise ValueError("active_level_count must fit the soil level dimension")

    carbon_level_fluxes = carbon_flux[4:]
    nitrogen_level_fluxes = nitrogen_flux[3:]
    expected_level_shape = (number_of_columns, number_of_levels)
    if any(value.shape != expected_level_shape for value in carbon_level_fluxes):
        raise ValueError("dynamic carbon level fluxes must match soil pools")
    if any(value.shape != expected_level_shape for value in nitrogen_level_fluxes):
        raise ValueError("dynamic nitrogen level fluxes must match soil pools")
    number_of_gridcells = state.seed_carbon_gridcell.shape[0]
    if state.seed_nitrogen_gridcell.shape != (number_of_gridcells,):
        raise ValueError("carbon and nitrogen seed pools must match")
    carbon_seed_fluxes = carbon_flux[:2]
    nitrogen_seed_fluxes = nitrogen_flux[:2]
    if any(value.shape != (number_of_gridcells,) for value in carbon_seed_fluxes):
        raise ValueError("dynamic carbon seed fluxes must match seed pools")
    if any(value.shape != (number_of_gridcells,) for value in nitrogen_seed_fluxes):
        raise ValueError("dynamic nitrogen seed fluxes must match seed pools")
    if jnp.shape(inputs.timestep_seconds) != ():
        raise ValueError("timestep_seconds must be scalar")

    timestep = jnp.asarray(inputs.timestep_seconds, dtype=jnp.float64)
    active = soil_column[:, None] & (
        jnp.arange(number_of_levels, dtype=jnp.int32)[None, :]
        < jnp.int32(active_level_count)
    )

    def add_pool_flux(
        pools: jnp.ndarray,
        pool_index: int,
        flux: jnp.ndarray,
    ) -> jnp.ndarray:
        previous = pools[:, :, pool_index]
        updated = fortran_add(
            previous,
            fortran_multiply(jnp.asarray(flux, dtype=jnp.float64), timestep),
        )
        return pools.at[:, :, pool_index].set(jnp.where(active, updated, previous))

    carbon_pool = add_pool_flux(
        carbon_pool,
        METABOLIC_LITTER_POOL_INDEX,
        carbon_flux.dwt_fine_root_carbon_to_metabolic_litter_column_level,
    )
    carbon_pool = add_pool_flux(
        carbon_pool,
        CELLULOSE_LITTER_POOL_INDEX,
        carbon_flux.dwt_fine_root_carbon_to_cellulose_litter_column_level,
    )
    carbon_pool = add_pool_flux(
        carbon_pool,
        LIGNIN_LITTER_POOL_INDEX,
        carbon_flux.dwt_fine_root_carbon_to_lignin_litter_column_level,
    )
    carbon_pool = add_pool_flux(
        carbon_pool,
        COARSE_WOODY_DEBRIS_POOL_INDEX,
        fortran_add(
            carbon_flux.dwt_live_coarse_root_carbon_to_cwd_column_level,
            carbon_flux.dwt_dead_coarse_root_carbon_to_cwd_column_level,
        ),
    )

    carbon_seed = jnp.asarray(state.seed_carbon_gridcell, dtype=jnp.float64)
    carbon_seed = fortran_add(
        carbon_seed,
        -fortran_multiply(
            carbon_flux.dwt_seed_carbon_to_leaf_gridcell,
            timestep,
        ),
    )
    carbon_seed = fortran_add(
        carbon_seed,
        -fortran_multiply(
            carbon_flux.dwt_seed_carbon_to_dead_stem_gridcell,
            timestep,
        ),
    )

    nitrogen_pool = add_pool_flux(
        nitrogen_pool,
        METABOLIC_LITTER_POOL_INDEX,
        nitrogen_flux.dwt_fine_root_nitrogen_to_metabolic_litter_column_level,
    )
    nitrogen_pool = add_pool_flux(
        nitrogen_pool,
        CELLULOSE_LITTER_POOL_INDEX,
        nitrogen_flux.dwt_fine_root_nitrogen_to_cellulose_litter_column_level,
    )
    nitrogen_pool = add_pool_flux(
        nitrogen_pool,
        LIGNIN_LITTER_POOL_INDEX,
        nitrogen_flux.dwt_fine_root_nitrogen_to_lignin_litter_column_level,
    )
    nitrogen_pool = add_pool_flux(
        nitrogen_pool,
        COARSE_WOODY_DEBRIS_POOL_INDEX,
        fortran_add(
            nitrogen_flux.dwt_live_coarse_root_nitrogen_to_cwd_column_level,
            nitrogen_flux.dwt_dead_coarse_root_nitrogen_to_cwd_column_level,
        ),
    )

    nitrogen_seed = jnp.asarray(state.seed_nitrogen_gridcell, dtype=jnp.float64)
    nitrogen_seed = fortran_add(
        nitrogen_seed,
        -fortran_multiply(
            nitrogen_flux.dwt_seed_nitrogen_to_leaf_gridcell,
            timestep,
        ),
    )
    nitrogen_seed = fortran_add(
        nitrogen_seed,
        -fortran_multiply(
            nitrogen_flux.dwt_seed_nitrogen_to_dead_stem_gridcell,
            timestep,
        ),
    )

    return DynamicPatchStateUpdateResult(
        state=DynamicPatchStateUpdateState(
            soil_carbon_nitrogen=soil._replace(
                carbon_pool_column_level_pool=carbon_pool,
                nitrogen_pool_column_level_pool=nitrogen_pool,
            ),
            seed_carbon_gridcell=carbon_seed,
            seed_nitrogen_gridcell=nitrogen_seed,
        ),
        updated_column_level_count=(
            jnp.sum(soil_column, dtype=jnp.int32) * jnp.int32(active_level_count)
        ),
        updated_gridcell_count=jnp.int32(number_of_gridcells),
    )


def aggregate_dynamic_vegetation_patch_fluxes(
    topology: SubgridTopology,
    inputs: DynamicPatchFluxAggregationInputs,
    parameters: DynamicPatchFluxAggregationParameters,
    patch_amounts: DynamicVegetationPatchFluxes,
    carbon_flux: DynamicLandcoverCarbonFluxState,
    nitrogen_flux: DynamicLandcoverNitrogenFluxState,
    *,
    active_level_count: int = CONFIGURED_ACTIVE_DECOMPOSITION_LEVELS,
) -> DynamicPatchFluxAggregationResult:

    patch_gridcell = jnp.asarray(topology.patch.gridcell, dtype=jnp.int32)
    patch_column = jnp.asarray(topology.patch.column, dtype=jnp.int32)
    vegetation_type = jnp.asarray(topology.patch.vegetation_type, dtype=jnp.int32)
    number_of_patches = patch_gridcell.shape[0]
    if patch_column.shape != (number_of_patches,) or vegetation_type.shape != (
        number_of_patches,
    ):
        raise ValueError("patch topology vectors must match")
    number_of_gridcells = topology.grid.longitude_degrees.shape[0]
    number_of_columns = topology.column.gridcell.shape[0]
    number_of_levels = carbon_flux[4].shape[1]
    if jnp.shape(inputs.timestep_seconds) != ():
        raise ValueError("timestep_seconds must be scalar")
    profile_shape = (number_of_patches, number_of_levels)
    if inputs.fine_root_profile_patch_level.shape != profile_shape:
        raise ValueError("fine-root profile must have shape (patch, level)")
    if inputs.coarse_root_profile_patch_level.shape != profile_shape:
        raise ValueError("coarse-root profile must have shape (patch, level)")
    if not 0 <= active_level_count <= number_of_levels:
        raise ValueError("active_level_count must fit the profile level dimension")
    patch_shape = (number_of_patches,)
    if any(value.shape != patch_shape for value in patch_amounts):
        raise ValueError("all dynamic vegetation patch amounts must match topology")
    pft_parameters = tuple(parameters)
    if any(len(value.shape) != 1 for value in pft_parameters):
        raise ValueError("fine-root litter parameters must be PFT vectors")
    pft_count = pft_parameters[0].shape[0]
    if any(value.shape != (pft_count,) for value in pft_parameters):
        raise ValueError("fine-root litter parameter vectors must match")

    grid_shape = (number_of_gridcells,)
    column_shape = (number_of_columns,)
    column_level_shape = (number_of_columns, number_of_levels)
    if any(value.shape != grid_shape for value in carbon_flux[:3]):
        raise ValueError("dynamic carbon gridcell fluxes must match topology")
    if carbon_flux[3].shape != column_shape:
        raise ValueError("dynamic slash carbon flux must match columns")
    if any(value.shape != column_level_shape for value in carbon_flux[4:]):
        raise ValueError("dynamic carbon level fluxes must match topology")
    if any(value.shape != grid_shape for value in nitrogen_flux[:3]):
        raise ValueError("dynamic nitrogen gridcell fluxes must match topology")
    if any(value.shape != column_level_shape for value in nitrogen_flux[3:]):
        raise ValueError("dynamic nitrogen level fluxes must match topology")

    timestep = jnp.asarray(inputs.timestep_seconds, dtype=jnp.float64)

    def accumulate_patch_vector(
        initial: jnp.ndarray,
        target_index: jnp.ndarray,
        contribution: jnp.ndarray,
    ) -> jnp.ndarray:
        def add_patch(index: int, values: jnp.ndarray) -> jnp.ndarray:
            target = target_index[index]
            return values.at[target].set(
                fortran_add(values[target], contribution[index])
            )

        return lax.fori_loop(0, number_of_patches, add_patch, initial)

    leaf_seed_carbon_patch = fortran_divide(
        patch_amounts.leaf_carbon_seed_gridcell_area_patch,
        timestep,
    )
    dead_stem_seed_carbon_patch = fortran_divide(
        patch_amounts.dead_stem_carbon_seed_gridcell_area_patch,
        timestep,
    )
    leaf_seed_nitrogen_patch = fortran_divide(
        patch_amounts.leaf_nitrogen_seed_gridcell_area_patch,
        timestep,
    )
    dead_stem_seed_nitrogen_patch = fortran_divide(
        patch_amounts.dead_stem_nitrogen_seed_gridcell_area_patch,
        timestep,
    )
    seed_carbon_leaf_gridcell = accumulate_patch_vector(
        jnp.asarray(carbon_flux.dwt_seed_carbon_to_leaf_gridcell, dtype=jnp.float64),
        patch_gridcell,
        leaf_seed_carbon_patch,
    )
    seed_carbon_dead_stem_gridcell = accumulate_patch_vector(
        jnp.asarray(
            carbon_flux.dwt_seed_carbon_to_dead_stem_gridcell,
            dtype=jnp.float64,
        ),
        patch_gridcell,
        dead_stem_seed_carbon_patch,
    )
    seed_nitrogen_leaf_gridcell = accumulate_patch_vector(
        jnp.asarray(
            nitrogen_flux.dwt_seed_nitrogen_to_leaf_gridcell,
            dtype=jnp.float64,
        ),
        patch_gridcell,
        leaf_seed_nitrogen_patch,
    )
    seed_nitrogen_dead_stem_gridcell = accumulate_patch_vector(
        jnp.asarray(
            nitrogen_flux.dwt_seed_nitrogen_to_dead_stem_gridcell,
            dtype=jnp.float64,
        ),
        patch_gridcell,
        dead_stem_seed_nitrogen_patch,
    )

    fine_root_carbon = fortran_multiply(
        jnp.float64(-1.0),
        patch_amounts.fine_root_carbon_to_litter_column_area_patch,
    )
    live_coarse_root_carbon = fortran_multiply(
        jnp.float64(-1.0),
        patch_amounts.live_coarse_root_carbon_to_litter_column_area_patch,
    )
    dead_coarse_root_carbon = fortran_multiply(
        jnp.float64(-1.0),
        patch_amounts.dead_coarse_root_carbon_to_litter_column_area_patch,
    )
    fine_root_nitrogen = fortran_multiply(
        jnp.float64(-1.0),
        patch_amounts.fine_root_nitrogen_to_litter_column_area_patch,
    )
    live_coarse_root_nitrogen = fortran_multiply(
        jnp.float64(-1.0),
        patch_amounts.live_coarse_root_nitrogen_to_litter_column_area_patch,
    )
    dead_coarse_root_nitrogen = fortran_multiply(
        jnp.float64(-1.0),
        patch_amounts.dead_coarse_root_nitrogen_to_litter_column_area_patch,
    )

    slash_carbon = jnp.asarray(
        carbon_flux.dwt_slash_carbon_flux_column,
        dtype=jnp.float64,
    )

    def add_slash(index: int, values: jnp.ndarray) -> jnp.ndarray:
        column = patch_column[index]
        updated = fortran_add(
            values[column],
            fortran_divide(fine_root_carbon[index], timestep),
        )
        updated = fortran_add(
            updated,
            fortran_divide(live_coarse_root_carbon[index], timestep),
        )
        updated = fortran_add(
            updated,
            fortran_divide(dead_coarse_root_carbon[index], timestep),
        )
        return values.at[column].set(updated)

    slash_carbon = lax.fori_loop(0, number_of_patches, add_slash, slash_carbon)

    carbon_level = tuple(
        jnp.asarray(value, dtype=jnp.float64) for value in carbon_flux[4:]
    )
    nitrogen_level = tuple(
        jnp.asarray(value, dtype=jnp.float64) for value in nitrogen_flux[3:]
    )
    fine_root_profile = jnp.asarray(
        inputs.fine_root_profile_patch_level,
        dtype=jnp.float64,
    )
    coarse_root_profile = jnp.asarray(
        inputs.coarse_root_profile_patch_level,
        dtype=jnp.float64,
    )
    metabolic_fraction = jnp.asarray(
        parameters.fine_root_metabolic_fraction_pft,
        dtype=jnp.float64,
    )[vegetation_type]
    cellulose_fraction = jnp.asarray(
        parameters.fine_root_cellulose_fraction_pft,
        dtype=jnp.float64,
    )[vegetation_type]
    lignin_fraction = jnp.asarray(
        parameters.fine_root_lignin_fraction_pft,
        dtype=jnp.float64,
    )[vegetation_type]

    def vertical_contribution(
        amount: jnp.ndarray,
        fraction: jnp.ndarray | None,
        profile: jnp.ndarray,
        patch_index: int,
        level_index: int,
    ) -> jnp.ndarray:
        value = amount[patch_index]
        if fraction is not None:
            value = fortran_multiply(value, fraction[patch_index])
        value = fortran_divide(value, timestep)
        return fortran_multiply(value, profile[patch_index, level_index])

    def add_vertical_level(
        level_index: int,
        level_carry: tuple[jnp.ndarray, ...],
    ) -> tuple[jnp.ndarray, ...]:
        def add_vertical(
            patch_index: int,
            carry: tuple[jnp.ndarray, ...],
        ) -> tuple[jnp.ndarray, ...]:
            c_met, c_cel, c_lig, c_live, c_dead = carry[:5]
            n_met, n_cel, n_lig, n_live, n_dead = carry[5:]
            column = patch_column[patch_index]

            def add_at_column(matrix: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
                return matrix.at[column, level_index].set(
                    fortran_add(matrix[column, level_index], value)
                )

            c_met = add_at_column(
                c_met,
                vertical_contribution(
                    fine_root_carbon,
                    metabolic_fraction,
                    fine_root_profile,
                    patch_index,
                    level_index,
                ),
            )
            c_cel = add_at_column(
                c_cel,
                vertical_contribution(
                    fine_root_carbon,
                    cellulose_fraction,
                    fine_root_profile,
                    patch_index,
                    level_index,
                ),
            )
            c_lig = add_at_column(
                c_lig,
                vertical_contribution(
                    fine_root_carbon,
                    lignin_fraction,
                    fine_root_profile,
                    patch_index,
                    level_index,
                ),
            )
            n_met = add_at_column(
                n_met,
                vertical_contribution(
                    fine_root_nitrogen,
                    metabolic_fraction,
                    fine_root_profile,
                    patch_index,
                    level_index,
                ),
            )
            n_cel = add_at_column(
                n_cel,
                vertical_contribution(
                    fine_root_nitrogen,
                    cellulose_fraction,
                    fine_root_profile,
                    patch_index,
                    level_index,
                ),
            )
            n_lig = add_at_column(
                n_lig,
                vertical_contribution(
                    fine_root_nitrogen,
                    lignin_fraction,
                    fine_root_profile,
                    patch_index,
                    level_index,
                ),
            )
            c_live = add_at_column(
                c_live,
                vertical_contribution(
                    live_coarse_root_carbon,
                    None,
                    coarse_root_profile,
                    patch_index,
                    level_index,
                ),
            )
            n_live = add_at_column(
                n_live,
                vertical_contribution(
                    live_coarse_root_nitrogen,
                    None,
                    coarse_root_profile,
                    patch_index,
                    level_index,
                ),
            )
            c_dead = add_at_column(
                c_dead,
                vertical_contribution(
                    dead_coarse_root_carbon,
                    None,
                    coarse_root_profile,
                    patch_index,
                    level_index,
                ),
            )
            n_dead = add_at_column(
                n_dead,
                vertical_contribution(
                    dead_coarse_root_nitrogen,
                    None,
                    coarse_root_profile,
                    patch_index,
                    level_index,
                ),
            )
            return (
                c_met,
                c_cel,
                c_lig,
                c_live,
                c_dead,
                n_met,
                n_cel,
                n_lig,
                n_live,
                n_dead,
            )

        return lax.fori_loop(
            0,
            number_of_patches,
            add_vertical,
            level_carry,
        )

    combined_level = lax.fori_loop(
        0,
        active_level_count,
        add_vertical_level,
        (*carbon_level, *nitrogen_level),
    )
    carbon_level = combined_level[:5]
    nitrogen_level = combined_level[5:]

    wood_product_carbon_patch = fortran_divide(
        -patch_amounts.wood_product_carbon_gridcell_area_patch,
        timestep,
    )
    crop_product_carbon_patch = fortran_divide(
        -patch_amounts.crop_product_carbon_gridcell_area_patch,
        timestep,
    )
    wood_product_nitrogen_patch = fortran_divide(
        -patch_amounts.wood_product_nitrogen_gridcell_area_patch,
        timestep,
    )
    crop_product_nitrogen_patch = fortran_divide(
        -patch_amounts.crop_product_nitrogen_gridcell_area_patch,
        timestep,
    )
    conversion_carbon_patch = fortran_divide(
        -patch_amounts.conversion_carbon_gridcell_area_patch,
        timestep,
    )
    conversion_nitrogen_patch = fortran_divide(
        -patch_amounts.conversion_nitrogen_gridcell_area_patch,
        timestep,
    )
    conversion_carbon_gridcell = accumulate_patch_vector(
        jnp.asarray(carbon_flux.dwt_conversion_carbon_flux_gridcell, dtype=jnp.float64),
        patch_gridcell,
        conversion_carbon_patch,
    )
    conversion_nitrogen_gridcell = accumulate_patch_vector(
        jnp.asarray(
            nitrogen_flux.dwt_conversion_nitrogen_flux_gridcell,
            dtype=jnp.float64,
        ),
        patch_gridcell,
        conversion_nitrogen_patch,
    )

    return DynamicPatchFluxAggregationResult(
        carbon_flux=DynamicLandcoverCarbonFluxState(
            dwt_seed_carbon_to_leaf_gridcell=seed_carbon_leaf_gridcell,
            dwt_seed_carbon_to_dead_stem_gridcell=(
                seed_carbon_dead_stem_gridcell
            ),
            dwt_conversion_carbon_flux_gridcell=conversion_carbon_gridcell,
            dwt_slash_carbon_flux_column=slash_carbon,
            dwt_fine_root_carbon_to_metabolic_litter_column_level=carbon_level[0],
            dwt_fine_root_carbon_to_cellulose_litter_column_level=carbon_level[1],
            dwt_fine_root_carbon_to_lignin_litter_column_level=carbon_level[2],
            dwt_live_coarse_root_carbon_to_cwd_column_level=carbon_level[3],
            dwt_dead_coarse_root_carbon_to_cwd_column_level=carbon_level[4],
        ),
        nitrogen_flux=DynamicLandcoverNitrogenFluxState(
            dwt_seed_nitrogen_to_leaf_gridcell=seed_nitrogen_leaf_gridcell,
            dwt_seed_nitrogen_to_dead_stem_gridcell=(
                seed_nitrogen_dead_stem_gridcell
            ),
            dwt_conversion_nitrogen_flux_gridcell=conversion_nitrogen_gridcell,
            dwt_fine_root_nitrogen_to_metabolic_litter_column_level=nitrogen_level[0],
            dwt_fine_root_nitrogen_to_cellulose_litter_column_level=(
                nitrogen_level[1]
            ),
            dwt_fine_root_nitrogen_to_lignin_litter_column_level=nitrogen_level[2],
            dwt_live_coarse_root_nitrogen_to_cwd_column_level=nitrogen_level[3],
            dwt_dead_coarse_root_nitrogen_to_cwd_column_level=nitrogen_level[4],
        ),
        patch_flux=DynamicVegetationPatchRateFluxes(
            seed_carbon_to_leaf_patch=leaf_seed_carbon_patch,
            seed_carbon_to_dead_stem_patch=dead_stem_seed_carbon_patch,
            conversion_carbon_patch=conversion_carbon_patch,
            wood_product_carbon_gain_patch=wood_product_carbon_patch,
            crop_product_carbon_gain_patch=crop_product_carbon_patch,
            seed_nitrogen_to_leaf_patch=leaf_seed_nitrogen_patch,
            seed_nitrogen_to_dead_stem_patch=dead_stem_seed_nitrogen_patch,
            conversion_nitrogen_patch=conversion_nitrogen_patch,
            wood_product_nitrogen_gain_patch=wood_product_nitrogen_patch,
            crop_product_nitrogen_gain_patch=crop_product_nitrogen_patch,
        ),
    )


def _validate_inputs(
    topology: SubgridTopology,
    state: SoilBiogeochemPrecisionControlState,
    decomposition_layer_thickness: jnp.ndarray,
    active_level_count: int,
) -> int:
    number_of_columns = topology.column.gridcell.shape[0]
    pool_shape = (
        number_of_columns,
        NUMBER_OF_DECOMPOSITION_LEVELS,
        NUMBER_OF_DECOMPOSITION_POOLS,
    )
    level_shape = (number_of_columns, NUMBER_OF_DECOMPOSITION_LEVELS)
    if state.carbon_pool_column_level_pool.shape != pool_shape:
        raise ValueError(f"carbon pools must have shape {pool_shape}")
    if state.nitrogen_pool_column_level_pool.shape != pool_shape:
        raise ValueError(f"nitrogen pools must have shape {pool_shape}")
    for name in (
        "carbon_truncation_column_level",
        "nitrogen_truncation_column_level",
        "mineral_nitrogen_column_level",
        "ammonium_column_level",
        "nitrate_column_level",
    ):
        if getattr(state, name).shape != level_shape:
            raise ValueError(f"{name} must have shape {level_shape}")
    if decomposition_layer_thickness.shape != (NUMBER_OF_DECOMPOSITION_LEVELS,):
        raise ValueError(
            "decomposition_layer_thickness must have shape "
            f"({NUMBER_OF_DECOMPOSITION_LEVELS},)"
        )
    if not 0 <= active_level_count <= NUMBER_OF_DECOMPOSITION_LEVELS:
        raise ValueError("active_level_count must be within the decomposition grid")
    return number_of_columns


def _update_fields(
    transition: ColumnWeightTransition,
    column_gridcell: jnp.ndarray,
    fields_by_update_column: jnp.ndarray,
    *,
    number_of_gridcells: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    def update_one(field: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        result = update_column_state_no_special_handling(
            transition,
            column_gridcell,
            field,
            number_of_gridcells=number_of_gridcells,
        )
        return result.state, result.adjustment_column

    return jax.vmap(update_one)(fields_by_update_column)


def dynamic_soil_column_adjustments(
    transition: ColumnWeightTransition,
    topology: SubgridTopology,
    state: SoilBiogeochemPrecisionControlState,
    decomposition_layer_thickness: jnp.ndarray,
    *,
    active_level_count: int = CONFIGURED_ACTIVE_DECOMPOSITION_LEVELS,
) -> DynamicSoilColumnAdjustmentResult:

    thickness = jnp.asarray(decomposition_layer_thickness, dtype=jnp.float64)
    number_of_columns = _validate_inputs(
        topology,
        state,
        thickness,
        active_level_count,
    )
    number_of_gridcells = topology.grid.longitude_degrees.shape[0]
    column_gridcell = jnp.asarray(topology.column.gridcell, dtype=jnp.int32)

    def update_pool_cube(values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        original = jnp.asarray(values, dtype=jnp.float64)
        ordered = jnp.transpose(
            original[:, :active_level_count, :],
            (2, 1, 0),
        )
        flat = ordered.reshape((-1, number_of_columns))
        updated_flat, adjustments_flat = _update_fields(
            transition,
            column_gridcell,
            flat,
            number_of_gridcells=number_of_gridcells,
        )
        updated_ordered = updated_flat.reshape(ordered.shape)
        adjustments = adjustments_flat.reshape(ordered.shape)
        updated_active = jnp.transpose(updated_ordered, (2, 1, 0))
        return original.at[:, :active_level_count, :].set(updated_active), adjustments

    def update_level_matrix(values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        original = jnp.asarray(values, dtype=jnp.float64)
        ordered = original[:, :active_level_count].T
        updated, adjustments = _update_fields(
            transition,
            column_gridcell,
            ordered,
            number_of_gridcells=number_of_gridcells,
        )
        return original.at[:, :active_level_count].set(updated.T), adjustments

    carbon_pool, carbon_pool_adjustment = update_pool_cube(
        state.carbon_pool_column_level_pool
    )
    carbon_truncation, carbon_truncation_adjustment = update_level_matrix(
        state.carbon_truncation_column_level
    )
    carbon_balance = jnp.zeros(number_of_columns, dtype=jnp.float64)
    for pool_index in range(NUMBER_OF_DECOMPOSITION_POOLS):
        for level_index in range(active_level_count):
            carbon_balance = fortran_add(
                carbon_balance,
                fortran_multiply(
                    carbon_pool_adjustment[pool_index, level_index],
                    thickness[level_index],
                ),
            )
    for level_index in range(active_level_count):
        carbon_balance = fortran_add(
            carbon_balance,
            fortran_multiply(
                carbon_truncation_adjustment[level_index],
                thickness[level_index],
            ),
        )

    nitrogen_pool, nitrogen_pool_adjustment = update_pool_cube(
        state.nitrogen_pool_column_level_pool
    )
    nitrogen_truncation, nitrogen_truncation_adjustment = update_level_matrix(
        state.nitrogen_truncation_column_level
    )
    mineral_nitrogen, mineral_adjustment = update_level_matrix(
        state.mineral_nitrogen_column_level
    )
    nitrogen_balance = jnp.zeros(number_of_columns, dtype=jnp.float64)
    for pool_index in range(NUMBER_OF_DECOMPOSITION_POOLS):
        for level_index in range(active_level_count):
            nitrogen_balance = fortran_add(
                nitrogen_balance,
                fortran_multiply(
                    nitrogen_pool_adjustment[pool_index, level_index],
                    thickness[level_index],
                ),
            )
    for level_index in range(active_level_count):
        nitrogen_balance = fortran_add(
            nitrogen_balance,
            fortran_multiply(
                nitrogen_truncation_adjustment[level_index],
                thickness[level_index],
            ),
        )
        nitrogen_balance = fortran_add(
            nitrogen_balance,
            fortran_multiply(
                mineral_adjustment[level_index],
                thickness[level_index],
            ),
        )

    nitrate, nitrate_adjustment = update_level_matrix(state.nitrate_column_level)
    ammonium, ammonium_adjustment = update_level_matrix(state.ammonium_column_level)
    zero_adjustment = jnp.zeros(number_of_columns, dtype=jnp.float64)
    if active_level_count:
        final_level = active_level_count - 1
        nitrate_balance = fortran_add(
            zero_adjustment,
            fortran_multiply(
                nitrate_adjustment[final_level],
                thickness[final_level],
            ),
        )
        ammonium_balance = fortran_add(
            zero_adjustment,
            fortran_multiply(
                ammonium_adjustment[final_level],
                thickness[final_level],
            ),
        )
    else:
        nitrate_balance = zero_adjustment
        ammonium_balance = zero_adjustment

    return DynamicSoilColumnAdjustmentResult(
        state=SoilBiogeochemPrecisionControlState(
            carbon_pool_column_level_pool=carbon_pool,
            nitrogen_pool_column_level_pool=nitrogen_pool,
            carbon_truncation_column_level=carbon_truncation,
            nitrogen_truncation_column_level=nitrogen_truncation,
            mineral_nitrogen_column_level=mineral_nitrogen,
            ammonium_column_level=ammonium,
            nitrate_column_level=nitrate,
        ),
        carbon_balance_adjustment_column=carbon_balance,
        nitrogen_balance_adjustment_column=nitrogen_balance,
        nitrate_balance_adjustment_column=nitrate_balance,
        ammonium_balance_adjustment_column=ammonium_balance,
    )


__all__ = [
    "CELLULOSE_LITTER_POOL_INDEX",
    "COARSE_WOODY_DEBRIS_POOL_INDEX",
    "CONFIGURED_ACTIVE_DECOMPOSITION_LEVELS",
    "DEAD_STEM_CARBON_SEED",
    "DYNAMIC_INITIATING_CANOPY_FIELDS",
    "DYNAMIC_INITIATING_CARBON_FLUX_FIELDS",
    "DYNAMIC_INITIATING_CN_VEGETATION_FIELDS",
    "DYNAMIC_INITIATING_NITROGEN_FLUX_FIELDS",
    "DYNAMIC_INITIATING_PHOTOSYNTHESIS_FIELDS",
    "DynamicPatchInitializationClock",
    "DynamicPatchInitializationResult",
    "DynamicPatchInitializationState",
    "DynamicPatchFluxAggregationInputs",
    "DynamicPatchFluxAggregationParameters",
    "DynamicPatchFluxAggregationResult",
    "DynamicPatchStateUpdateInputs",
    "DynamicPatchStateUpdateResult",
    "DynamicPatchStateUpdateState",
    "DynamicSoilColumnAdjustmentResult",
    "DynamicVegetationPatchAdjustmentResult",
    "DynamicVegetationPatchFluxes",
    "DynamicVegetationPatchInputs",
    "DynamicVegetationPatchParameters",
    "DynamicVegetationPatchRateFluxes",
    "DynamicVegetationPatchState",
    "LEAF_CARBON_SEED",
    "LIGNIN_LITTER_POOL_INDEX",
    "METABOLIC_LITTER_POOL_INDEX",
    "apply_dynamic_patch_carbon_nitrogen_fluxes",
    "aggregate_dynamic_vegetation_patch_fluxes",
    "dynamic_soil_column_adjustments",
    "dynamic_vegetation_patch_adjustments",
    "initialize_dynamic_vegetation_patches",
]
