
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import lax

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.annual_flux_dribbler import AnnualFluxDribblerState
from clm_jax.physics.annual_flux_dribbler import current_flux as annual_flux_current_flux
from clm_jax.physics.annual_flux_dribbler import (
    set_current_delta as set_annual_flux_current_delta,
)
from clm_jax.physics.cn_flux_integrals_gpu import integrate_levels
from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    get_cn_flux_field,
    set_cn_flux_field,
)
from clm_jax.physics.cn_zero_manifest import (
    NUMBER_OF_DECOMPOSITION_POOLS,
    NUMBER_OF_DECOMPOSITION_TRANSITIONS,
)
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
    fortran_ordered_sum,
    strict_exp,
)
from clm_jax.physics.subgrid import (
    column_to_gridcell_mean,
    patch_to_column_average,
)

_MAXIMUM_PATCHES_PER_COLUMN = 79
_MAXIMUM_COLUMNS_PER_GRIDCELL = 64
_ACTIVE_DECOMPOSITION_LEVELS = 20
_SECONDS_PER_DAY = 86400.0


class CnDriverSummarizeFluxesInputs(NamedTuple):
    soil_patch: jnp.ndarray
    soil_column: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    topology: SubgridTopology
    decomposition_layer_thickness: jnp.ndarray
    product_carbon_loss_gridcell: jnp.ndarray
    dynamic_landuse_carbon_flux_gridcell: jnp.ndarray
    is_beginning_current_year: jnp.ndarray
    is_first_step: jnp.ndarray


class CnDriverSummarizeFluxesState(NamedTuple):
    cn_fluxes: CnTransientFluxState
    temporary_npp_sum_patch: jnp.ndarray
    lagged_npp_column: jnp.ndarray
    dynamic_landuse_amount_to_dribble_gridcell: jnp.ndarray
    harvest_xsmrpool_amount_to_dribble_gridcell: jnp.ndarray


class CnDriverSummarizeFluxesParameters(NamedTuple):
    time_step: float | jnp.ndarray = 1800.0
    days_per_year: float | jnp.ndarray = 365.0
    nfix_time_constant_days: float | jnp.ndarray = 10.0
    minimum_crop_vegetation_type: int | jnp.ndarray = 17
    cascade_donor_pool: jnp.ndarray = np.asarray(
        [0, 1, 2, 4, 4, 5, 5, 6, 3, 3],
        dtype=np.int32,
    )
    pool_is_litter: jnp.ndarray = np.asarray(
        [True, True, True, False, False, False, False],
        dtype=np.bool_,
    )
    pool_is_soil: jnp.ndarray = np.asarray(
        [False, False, False, False, True, True, True],
        dtype=np.bool_,
    )
    pool_is_cwd: jnp.ndarray = np.asarray(
        [False, False, False, True, False, False, False],
        dtype=np.bool_,
    )
    special_value: float | jnp.ndarray = float(CLM_SPECIAL_VALUE)


class CnDriverSummarizeFluxesGridcellFluxes(NamedTuple):
    net_ecosystem_production: jnp.ndarray
    fire_carbon_loss: jnp.ndarray
    harvest_xsmrpool_carbon_flux: jnp.ndarray
    harvest_xsmrpool_carbon_flux_dribbled: jnp.ndarray
    dynamic_landuse_carbon_flux_dribbled: jnp.ndarray
    net_ecosystem_exchange: jnp.ndarray
    landuse_carbon_flux: jnp.ndarray
    net_biome_production: jnp.ndarray


class CnDriverSummarizeFluxesDiagnostics(NamedTuple):
    summarized_patch_count: jnp.ndarray
    summarized_column_count: jnp.ndarray
    unexpected_nonannual_delta_count: jnp.ndarray


class CnDriverSummarizeFluxesResult(NamedTuple):
    state: CnDriverSummarizeFluxesState
    gridcell_fluxes: CnDriverSummarizeFluxesGridcellFluxes
    diagnostics: CnDriverSummarizeFluxesDiagnostics


def _field(state: CnTransientFluxState, qualified_name: str) -> jnp.ndarray:
    return get_cn_flux_field(state, qualified_name)


def _set_masked(
    state: CnTransientFluxState,
    qualified_name: str,
    values: jnp.ndarray,
    mask: jnp.ndarray,
) -> CnTransientFluxState:
    previous = _field(state, qualified_name)
    broadcast = jnp.reshape(mask, mask.shape + (1,) * (previous.ndim - 1))
    return set_cn_flux_field(
        state,
        qualified_name,
        jnp.where(broadcast, values, previous),
    )


def _sum_fields(
    state: CnTransientFluxState,
    names: tuple[str, ...],
) -> jnp.ndarray:
    return fortran_ordered_sum(jnp.stack(tuple(_field(state, name) for name in names), axis=0))


def _integrate_levels(
    values: jnp.ndarray,
    thickness: jnp.ndarray,
    initial: jnp.ndarray,
    column_mask: jnp.ndarray,
) -> jnp.ndarray:

    return integrate_levels(values, thickness, initial, column_mask, _ACTIVE_DECOMPOSITION_LEVELS)


def _soil_carbon_summary(
    state: CnTransientFluxState,
    column_mask: jnp.ndarray,
    thickness: jnp.ndarray,
    parameters: CnDriverSummarizeFluxesParameters,
) -> CnTransientFluxState:
    transition_shape = (
        state.column_transition.shape[0],
        NUMBER_OF_DECOMPOSITION_TRANSITIONS,
    )
    pool_shape = (
        state.column_pool.shape[0],
        NUMBER_OF_DECOMPOSITION_POOLS,
    )

    hr_transition = _integrate_levels(
        _field(state, "soil_carbon.decomp_cascade_hr_vr_col"),
        thickness,
        _field(state, "soil_carbon.decomp_cascade_hr_col"),
        column_mask,
    )
    state = _set_masked(
        state,
        "soil_carbon.decomp_cascade_hr_col",
        hr_transition,
        column_mask,
    )
    carbon_transfer = _integrate_levels(
        _field(state, "soil_carbon.decomp_cascade_ctransfer_vr_col"),
        thickness,
        _field(state, "soil_carbon.decomp_cascade_ctransfer_col"),
        column_mask,
    )
    state = _set_masked(
        state,
        "soil_carbon.decomp_cascade_ctransfer_col",
        carbon_transfer,
        column_mask,
    )

    active_vertical_hr = jnp.zeros(
        (transition_shape[0], _ACTIVE_DECOMPOSITION_LEVELS),
        dtype=jnp.float64,
    )
    transition_hr_vertical = _field(
        state,
        "soil_carbon.decomp_cascade_hr_vr_col",
    )

    def sum_transition(transition: int, running: jnp.ndarray) -> jnp.ndarray:
        return fortran_add(
            running,
            transition_hr_vertical[:, :_ACTIVE_DECOMPOSITION_LEVELS, transition],
        )

    active_vertical_hr = lax.fori_loop(
        0,
        NUMBER_OF_DECOMPOSITION_TRANSITIONS,
        sum_transition,
        active_vertical_hr,
    )
    vertical_hr = (
        _field(state, "soil_carbon.hr_vr_col")
        .at[:, :_ACTIVE_DECOMPOSITION_LEVELS]
        .set(active_vertical_hr)
    )
    state = _set_masked(state, "soil_carbon.hr_vr_col", vertical_hr, column_mask)

    carbon_leached_pool = _integrate_levels(
        _field(state, "soil_carbon.decomp_cpools_transport_tendency_col"),
        thickness,
        jnp.zeros(pool_shape, dtype=jnp.float64),
        column_mask,
    )
    state = _set_masked(
        state,
        "soil_carbon.decomp_cpools_leached_col",
        carbon_leached_pool,
        column_mask,
    )
    carbon_leached = fortran_ordered_sum(
        jnp.moveaxis(carbon_leached_pool, 1, 0),
        initial=jnp.zeros(pool_shape[0], dtype=jnp.float64),
    )
    state = _set_masked(
        state,
        "soil_carbon.som_c_leached_col",
        carbon_leached,
        column_mask,
    )

    donor = jnp.asarray(parameters.cascade_donor_pool, dtype=jnp.int32)
    is_soil_transition = jnp.asarray(parameters.pool_is_soil, dtype=jnp.bool_)[donor]
    is_litter_transition = jnp.asarray(parameters.pool_is_litter, dtype=jnp.bool_)[donor]
    somhr = _field(state, "soil_carbon.somhr_col")
    lithr = _field(state, "soil_carbon.lithr_col")

    def classify_transition(
        transition: int,
        running: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        soil_running, litter_running = running
        flux = hr_transition[:, transition]
        return (
            jnp.where(
                column_mask & is_soil_transition[transition],
                fortran_add(soil_running, flux),
                soil_running,
            ),
            jnp.where(
                column_mask & is_litter_transition[transition],
                fortran_add(litter_running, flux),
                litter_running,
            ),
        )

    somhr, lithr = lax.fori_loop(
        0,
        NUMBER_OF_DECOMPOSITION_TRANSITIONS,
        classify_transition,
        (somhr, lithr),
    )
    state = _set_masked(state, "soil_carbon.somhr_col", somhr, column_mask)
    state = _set_masked(state, "soil_carbon.lithr_col", lithr, column_mask)
    state = _set_masked(
        state,
        "soil_carbon.hr_col",
        fortran_add(lithr, somhr),
        column_mask,
    )
    return state


def _soil_nitrogen_summary(
    state: CnTransientFluxState,
    column_mask: jnp.ndarray,
    thickness: jnp.ndarray,
) -> CnTransientFluxState:
    column_count = state.column_scalar.shape[0]
    transition_shape = (column_count, NUMBER_OF_DECOMPOSITION_TRANSITIONS)
    pool_shape = (column_count, NUMBER_OF_DECOMPOSITION_POOLS)

    for integrated_name, vertical_name in (
        (
            "soil_nitrogen.decomp_cascade_ntransfer_col",
            "soil_nitrogen.decomp_cascade_ntransfer_vr_col",
        ),
        (
            "soil_nitrogen.decomp_cascade_sminn_flux_col",
            "soil_nitrogen.decomp_cascade_sminn_flux_vr_col",
        ),
    ):
        integrated = _integrate_levels(
            _field(state, vertical_name),
            thickness,
            _field(state, integrated_name).reshape(transition_shape),
            column_mask,
        )
        state = _set_masked(state, integrated_name, integrated, column_mask)

    for integrated_name, vertical_name in (
        ("soil_nitrogen.f_nit_col", "soil_nitrogen.f_nit_vr_col"),
        ("soil_nitrogen.f_denit_col", "soil_nitrogen.f_denit_vr_col"),
        ("soil_nitrogen.pot_f_nit_col", "soil_nitrogen.pot_f_nit_vr_col"),
        ("soil_nitrogen.pot_f_denit_col", "soil_nitrogen.pot_f_denit_vr_col"),
        ("soil_nitrogen.f_n2o_nit_col", "soil_nitrogen.f_n2o_nit_vr_col"),
        (
            "soil_nitrogen.f_n2o_denit_col",
            "soil_nitrogen.f_n2o_denit_vr_col",
        ),
        (
            "soil_nitrogen.smin_no3_leached_col",
            "soil_nitrogen.smin_no3_leached_vr_col",
        ),
        (
            "soil_nitrogen.smin_no3_runoff_col",
            "soil_nitrogen.smin_no3_runoff_vr_col",
        ),
    ):
        integrated = _integrate_levels(
            _field(state, vertical_name),
            thickness,
            _field(state, integrated_name),
            column_mask,
        )
        state = _set_masked(state, integrated_name, integrated, column_mask)

    state = _set_masked(
        state,
        "soil_nitrogen.denit_col",
        _field(state, "soil_nitrogen.f_denit_col"),
        column_mask,
    )
    supplement = _integrate_levels(
        _field(state, "soil_nitrogen.supplement_to_sminn_vr_col"),
        thickness,
        jnp.zeros(column_count, dtype=jnp.float64),
        column_mask,
    )
    state = _set_masked(
        state,
        "soil_nitrogen.supplement_to_sminn_col",
        supplement,
        column_mask,
    )
    nitrogen_leached_pool = _integrate_levels(
        _field(state, "soil_nitrogen.decomp_npools_transport_tendency_col"),
        thickness,
        jnp.zeros(pool_shape, dtype=jnp.float64),
        column_mask,
    )
    state = _set_masked(
        state,
        "soil_nitrogen.decomp_npools_leached_col",
        nitrogen_leached_pool,
        column_mask,
    )
    nitrogen_leached = fortran_ordered_sum(
        jnp.moveaxis(nitrogen_leached_pool, 1, 0),
        initial=jnp.zeros(column_count, dtype=jnp.float64),
    )
    return _set_masked(
        state,
        "soil_nitrogen.som_n_leached_col",
        nitrogen_leached,
        column_mask,
    )


_FIRE_NITROGEN_FIELDS = tuple(
    f"vegetation_nitrogen.{name}_patch"
    for name in (
        "m_leafn_to_fire",
        "m_leafn_storage_to_fire",
        "m_leafn_xfer_to_fire",
        "m_frootn_to_fire",
        "m_frootn_storage_to_fire",
        "m_frootn_xfer_to_fire",
        "m_livestemn_to_fire",
        "m_livestemn_storage_to_fire",
        "m_livestemn_xfer_to_fire",
        "m_deadstemn_to_fire",
        "m_deadstemn_storage_to_fire",
        "m_deadstemn_xfer_to_fire",
        "m_livecrootn_to_fire",
        "m_livecrootn_storage_to_fire",
        "m_livecrootn_xfer_to_fire",
        "m_deadcrootn_to_fire",
        "m_deadcrootn_storage_to_fire",
        "m_deadcrootn_xfer_to_fire",
        "m_retransn_to_fire",
    )
)


def _patch_to_column(
    values: jnp.ndarray,
    inputs: CnDriverSummarizeFluxesInputs,
) -> jnp.ndarray:
    return patch_to_column_average(
        values,
        inputs.topology.patch.column,
        inputs.topology.patch.weight_on_column,
        inputs.topology.patch.active,
        inputs.soil_column,
        maximum_patches_per_column=_MAXIMUM_PATCHES_PER_COLUMN,
    )


def _vegetation_nitrogen_summary(
    state: CnTransientFluxState,
    inputs: CnDriverSummarizeFluxesInputs,
) -> CnTransientFluxState:
    patch_mask = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    column_mask = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    deploy = _sum_fields(
        state,
        (
            "vegetation_nitrogen.sminn_to_npool_patch",
            "vegetation_nitrogen.retransn_to_npool_patch",
            "vegetation_nitrogen.free_retransn_to_npool_patch",
        ),
    )
    state = _set_masked(
        state,
        "vegetation_nitrogen.ndeploy_patch",
        deploy,
        patch_mask,
    )
    fire_patch = _sum_fields(state, _FIRE_NITROGEN_FIELDS)
    state = _set_masked(
        state,
        "vegetation_nitrogen.fire_nloss_patch",
        fire_patch,
        patch_mask,
    )
    fire_patch_column = _patch_to_column(fire_patch, inputs)
    fire_pool = _integrate_levels(
        _field(state, "vegetation_nitrogen.m_decomp_npools_to_fire_vr_col"),
        inputs.decomposition_layer_thickness,
        _field(state, "vegetation_nitrogen.m_decomp_npools_to_fire_col"),
        column_mask,
    )
    state = _set_masked(
        state,
        "vegetation_nitrogen.m_decomp_npools_to_fire_col",
        fire_pool,
        column_mask,
    )
    fire_column = fortran_ordered_sum(
        jnp.moveaxis(fire_pool, 1, 0),
        initial=fire_patch_column,
    )
    return _set_masked(
        state,
        "vegetation_nitrogen.fire_nloss_col",
        fire_column,
        column_mask,
    )


_CURRENT_GROWTH_RESPIRATION_FIELDS = tuple(
    f"vegetation_carbon.cpool_{organ}_gr_patch"
    for organ in ("leaf", "froot", "livestem", "deadstem", "livecroot", "deadcroot")
)
_TRANSFER_GROWTH_RESPIRATION_FIELDS = tuple(
    f"vegetation_carbon.transfer_{organ}_gr_patch"
    for organ in ("leaf", "froot", "livestem", "deadstem", "livecroot", "deadcroot")
)
_STORAGE_GROWTH_RESPIRATION_FIELDS = tuple(
    f"vegetation_carbon.cpool_{organ}_storage_gr_patch"
    for organ in ("leaf", "froot", "livestem", "deadstem", "livecroot", "deadcroot")
)
_ROOT_RESPIRATION_FIELDS = tuple(
    f"vegetation_carbon.{name}_patch"
    for name in (
        "froot_mr",
        "cpool_froot_gr",
        "cpool_livecroot_gr",
        "cpool_deadcroot_gr",
        "transfer_froot_gr",
        "transfer_livecroot_gr",
        "transfer_deadcroot_gr",
        "cpool_froot_storage_gr",
        "cpool_livecroot_storage_gr",
        "cpool_deadcroot_storage_gr",
    )
)
_ABOVEGROUND_NPP_FIELDS = tuple(
    f"vegetation_carbon.{name}_patch"
    for name in (
        "cpool_to_leafc",
        "leafc_xfer_to_leafc",
        "cpool_to_livestemc",
        "livestemc_xfer_to_livestemc",
        "cpool_to_deadstemc",
        "deadstemc_xfer_to_deadstemc",
    )
)
_BELOWGROUND_NPP_FIELDS = tuple(
    f"vegetation_carbon.{name}_patch"
    for name in (
        "cpool_to_frootc",
        "frootc_xfer_to_frootc",
        "cpool_to_livecrootc",
        "livecrootc_xfer_to_livecrootc",
        "cpool_to_deadcrootc",
        "deadcrootc_xfer_to_deadcrootc",
    )
)
_LITTERFALL_FIELDS = tuple(
    f"vegetation_carbon.{name}_patch"
    for name in (
        "leafc_to_litter",
        "frootc_to_litter",
        "m_leafc_to_litter",
        "m_leafc_storage_to_litter",
        "m_leafc_xfer_to_litter",
        "m_frootc_to_litter",
        "m_frootc_storage_to_litter",
        "m_frootc_xfer_to_litter",
        "m_livestemc_to_litter",
        "m_livestemc_storage_to_litter",
        "m_livestemc_xfer_to_litter",
        "m_deadstemc_to_litter",
        "m_deadstemc_storage_to_litter",
        "m_deadstemc_xfer_to_litter",
        "m_livecrootc_to_litter",
        "m_livecrootc_storage_to_litter",
        "m_livecrootc_xfer_to_litter",
        "m_deadcrootc_to_litter",
        "m_deadcrootc_storage_to_litter",
        "m_deadcrootc_xfer_to_litter",
        "m_gresp_storage_to_litter",
        "m_gresp_xfer_to_litter",
        "m_leafc_to_litter_fire",
        "m_leafc_storage_to_litter_fire",
        "m_leafc_xfer_to_litter_fire",
        "m_livestemc_to_litter_fire",
        "m_livestemc_storage_to_litter_fire",
        "m_livestemc_xfer_to_litter_fire",
        "m_deadstemc_to_litter_fire",
        "m_deadstemc_storage_to_litter_fire",
        "m_deadstemc_xfer_to_litter_fire",
        "m_frootc_to_litter_fire",
        "m_frootc_storage_to_litter_fire",
        "m_frootc_xfer_to_litter_fire",
        "m_livecrootc_to_litter_fire",
        "m_livecrootc_storage_to_litter_fire",
        "m_livecrootc_xfer_to_litter_fire",
        "m_deadcrootc_to_litter_fire",
        "m_deadcrootc_storage_to_litter_fire",
        "m_deadcrootc_xfer_to_litter_fire",
        "m_gresp_storage_to_litter_fire",
        "m_gresp_xfer_to_litter_fire",
        "hrv_leafc_to_litter",
        "hrv_leafc_storage_to_litter",
        "hrv_leafc_xfer_to_litter",
        "hrv_frootc_to_litter",
        "hrv_frootc_storage_to_litter",
        "hrv_frootc_xfer_to_litter",
        "hrv_livestemc_to_litter",
        "hrv_livestemc_storage_to_litter",
        "hrv_livestemc_xfer_to_litter",
        "hrv_deadstemc_storage_to_litter",
        "hrv_deadstemc_xfer_to_litter",
        "hrv_livecrootc_to_litter",
        "hrv_livecrootc_storage_to_litter",
        "hrv_livecrootc_xfer_to_litter",
        "hrv_deadcrootc_to_litter",
        "hrv_deadcrootc_storage_to_litter",
        "hrv_deadcrootc_xfer_to_litter",
        "hrv_gresp_storage_to_litter",
        "hrv_gresp_xfer_to_litter",
    )
)
_FIRE_CARBON_FIELDS = tuple(
    f"vegetation_carbon.{name}_patch"
    for name in (
        "m_leafc_to_fire",
        "m_leafc_storage_to_fire",
        "m_leafc_xfer_to_fire",
        "m_frootc_to_fire",
        "m_frootc_storage_to_fire",
        "m_frootc_xfer_to_fire",
        "m_livestemc_to_fire",
        "m_livestemc_storage_to_fire",
        "m_livestemc_xfer_to_fire",
        "m_deadstemc_to_fire",
        "m_deadstemc_storage_to_fire",
        "m_deadstemc_xfer_to_fire",
        "m_livecrootc_to_fire",
        "m_livecrootc_storage_to_fire",
        "m_livecrootc_xfer_to_fire",
        "m_deadcrootc_to_fire",
        "m_deadcrootc_storage_to_fire",
        "m_deadcrootc_xfer_to_fire",
        "m_gresp_storage_to_fire",
        "m_gresp_xfer_to_fire",
    )
)
_WOOD_CARBON_ALLOCATION_FIELDS = tuple(
    f"vegetation_carbon.{name}_patch"
    for name in (
        "livestemc_xfer_to_livestemc",
        "deadstemc_xfer_to_deadstemc",
        "livecrootc_xfer_to_livecrootc",
        "deadcrootc_xfer_to_deadcrootc",
        "cpool_to_livestemc",
        "cpool_to_deadstemc",
        "cpool_to_livecrootc",
        "cpool_to_deadcrootc",
    )
)
_WOOD_CARBON_LOSS_FIELDS = tuple(
    f"vegetation_carbon.{name}_patch"
    for name in (
        "m_livestemc_to_litter",
        "m_deadstemc_to_litter",
        "m_livecrootc_to_litter",
        "m_deadcrootc_to_litter",
        "m_livestemc_to_fire",
        "m_deadstemc_to_fire",
        "m_livecrootc_to_fire",
        "m_deadcrootc_to_fire",
        "hrv_livestemc_to_litter",
        "hrv_livestemc_storage_to_litter",
        "hrv_livestemc_xfer_to_litter",
        "wood_harvestc",
        "hrv_deadstemc_storage_to_litter",
        "hrv_deadstemc_xfer_to_litter",
        "hrv_livecrootc_to_litter",
        "hrv_livecrootc_storage_to_litter",
        "hrv_livecrootc_xfer_to_litter",
        "hrv_deadcrootc_to_litter",
        "hrv_deadcrootc_storage_to_litter",
        "hrv_deadcrootc_xfer_to_litter",
    )
)
_SLASH_HARVEST_CARBON_FIELDS = tuple(
    f"vegetation_carbon.{name}_patch"
    for name in (
        "hrv_leafc_to_litter",
        "hrv_leafc_storage_to_litter",
        "hrv_leafc_xfer_to_litter",
        "hrv_frootc_to_litter",
        "hrv_frootc_storage_to_litter",
        "hrv_frootc_xfer_to_litter",
        "hrv_livestemc_to_litter",
        "hrv_livestemc_storage_to_litter",
        "hrv_livestemc_xfer_to_litter",
        "hrv_deadstemc_storage_to_litter",
        "hrv_deadstemc_xfer_to_litter",
        "hrv_livecrootc_to_litter",
        "hrv_livecrootc_storage_to_litter",
        "hrv_livecrootc_xfer_to_litter",
        "hrv_deadcrootc_to_litter",
        "hrv_deadcrootc_storage_to_litter",
        "hrv_deadcrootc_xfer_to_litter",
        "hrv_xsmrpool_to_atm",
        "hrv_gresp_storage_to_litter",
        "hrv_gresp_xfer_to_litter",
    )
)


def _dribble_flux(
    current_flux: jnp.ndarray,
    amount_to_dribble: jnp.ndarray,
    inputs: CnDriverSummarizeFluxesInputs,
    parameters: CnDriverSummarizeFluxesParameters,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    time_step = jnp.asarray(parameters.time_step, dtype=jnp.float64)
    current_delta = fortran_multiply(current_flux, time_step)
    beginning_year = jnp.asarray(inputs.is_beginning_current_year, dtype=jnp.bool_)
    dribbler_state = set_annual_flux_current_delta(
        AnnualFluxDribblerState(
            amount_to_dribble=amount_to_dribble,
            amount_from_this_timestep=jnp.zeros_like(current_delta),
        ),
        current_delta,
        is_beginning_current_year=beginning_year,
    )
    flux = annual_flux_current_flux(
        dribbler_state,
        time_step_seconds=time_step,
        days_per_year=parameters.days_per_year,
    )
    illegal = (
        (~beginning_year)
        & (~jnp.asarray(inputs.is_first_step, dtype=jnp.bool_))
        & (current_delta != jnp.float64(0.0))
    )
    return flux, dribbler_state.amount_to_dribble, illegal


def _vegetation_carbon_summary(
    state: CnDriverSummarizeFluxesState,
    inputs: CnDriverSummarizeFluxesInputs,
    parameters: CnDriverSummarizeFluxesParameters,
) -> tuple[
    CnDriverSummarizeFluxesState,
    CnDriverSummarizeFluxesGridcellFluxes,
    jnp.ndarray,
]:
    fluxes = state.cn_fluxes
    patch_mask = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    column_mask = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    crop = patch_mask & (
        jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
        >= jnp.asarray(parameters.minimum_crop_vegetation_type, dtype=jnp.int32)
    )

    maintenance = _sum_fields(
        fluxes,
        tuple(
            f"vegetation_carbon.{name}_mr_patch"
            for name in ("leaf", "froot", "livestem", "livecroot")
        ),
    )
    maintenance = jnp.where(
        crop,
        fortran_add(
            maintenance,
            _field(fluxes, "vegetation_carbon.grain_mr_patch"),
        ),
        maintenance,
    )
    fluxes = _set_masked(
        fluxes,
        "vegetation_carbon.mr_patch",
        maintenance,
        patch_mask,
    )

    current_growth = _sum_fields(fluxes, _CURRENT_GROWTH_RESPIRATION_FIELDS)
    current_growth = jnp.where(
        crop,
        fortran_add(
            current_growth,
            _field(fluxes, "vegetation_carbon.cpool_grain_gr_patch"),
        ),
        current_growth,
    )
    fluxes = _set_masked(
        fluxes,
        "vegetation_carbon.current_gr_patch",
        current_growth,
        patch_mask,
    )
    transfer_growth = _sum_fields(fluxes, _TRANSFER_GROWTH_RESPIRATION_FIELDS)
    transfer_growth = jnp.where(
        crop,
        fortran_add(
            transfer_growth,
            _field(fluxes, "vegetation_carbon.transfer_grain_gr_patch"),
        ),
        transfer_growth,
    )
    fluxes = _set_masked(
        fluxes,
        "vegetation_carbon.transfer_gr_patch",
        transfer_growth,
        patch_mask,
    )
    storage_growth = _sum_fields(fluxes, _STORAGE_GROWTH_RESPIRATION_FIELDS)
    storage_growth = jnp.where(
        crop,
        fortran_add(
            storage_growth,
            _field(fluxes, "vegetation_carbon.cpool_grain_storage_gr_patch"),
        ),
        storage_growth,
    )
    fluxes = _set_masked(
        fluxes,
        "vegetation_carbon.storage_gr_patch",
        storage_growth,
        patch_mask,
    )
    total_growth = fortran_ordered_sum(
        jnp.stack((current_growth, transfer_growth, storage_growth), axis=0)
    )
    fluxes = _set_masked(
        fluxes,
        "vegetation_carbon.gr_patch",
        total_growth,
        patch_mask,
    )

    autotrophic = fortran_add(maintenance, total_growth)
    autotrophic = jnp.where(
        crop,
        fortran_add(
            autotrophic,
            _field(fluxes, "vegetation_carbon.xsmrpool_to_atm_patch"),
        ),
        autotrophic,
    )
    autotrophic = fortran_add(
        autotrophic,
        _field(fluxes, "vegetation_carbon.soilc_change_patch"),
    )
    fluxes = _set_masked(
        fluxes,
        "vegetation_carbon.ar_patch",
        autotrophic,
        patch_mask,
    )
    gross_primary = _sum_fields(
        fluxes,
        (
            "vegetation_carbon.psnsun_to_cpool_patch",
            "vegetation_carbon.psnshade_to_cpool_patch",
        ),
    )
    fluxes = _set_masked(
        fluxes,
        "vegetation_carbon.gpp_patch",
        gross_primary,
        patch_mask,
    )
    net_primary = fortran_add(gross_primary, -autotrophic)
    fluxes = _set_masked(
        fluxes,
        "vegetation_carbon.npp_patch",
        net_primary,
        patch_mask,
    )
    root_respiration = _sum_fields(fluxes, _ROOT_RESPIRATION_FIELDS)
    fluxes = _set_masked(
        fluxes,
        "vegetation_carbon.rr_patch",
        root_respiration,
        patch_mask,
    )
    temporary_npp = jnp.where(
        patch_mask,
        fortran_add(state.temporary_npp_sum_patch, net_primary),
        state.temporary_npp_sum_patch,
    )

    aboveground = _sum_fields(fluxes, _ABOVEGROUND_NPP_FIELDS)
    aboveground = jnp.where(
        crop,
        fortran_add(
            fortran_add(
                aboveground,
                _field(fluxes, "vegetation_carbon.cpool_to_grainc_patch"),
            ),
            _field(fluxes, "vegetation_carbon.grainc_xfer_to_grainc_patch"),
        ),
        aboveground,
    )
    fluxes = _set_masked(
        fluxes,
        "vegetation_carbon.agnpp_patch",
        aboveground,
        patch_mask,
    )
    belowground = _sum_fields(fluxes, _BELOWGROUND_NPP_FIELDS)
    fluxes = _set_masked(
        fluxes,
        "vegetation_carbon.bgnpp_patch",
        belowground,
        patch_mask,
    )
    litterfall = _sum_fields(fluxes, _LITTERFALL_FIELDS)
    litterfall = jnp.where(
        crop,
        fortran_add(
            litterfall,
            _field(fluxes, "vegetation_carbon.livestemc_to_litter_patch"),
        ),
        litterfall,
    )
    fluxes = _set_masked(
        fluxes,
        "vegetation_carbon.litfall_patch",
        litterfall,
        patch_mask,
    )
    fire_patch = _sum_fields(fluxes, _FIRE_CARBON_FIELDS)
    fluxes = _set_masked(
        fluxes,
        "vegetation_carbon.fire_closs_patch",
        fire_patch,
        patch_mask,
    )

    derived_patch_fields = (
        (
            "vegetation_carbon.frootc_alloc_patch",
            (
                "vegetation_carbon.frootc_xfer_to_frootc_patch",
                "vegetation_carbon.cpool_to_frootc_patch",
            ),
        ),
        (
            "vegetation_carbon.frootc_loss_patch",
            tuple(
                f"vegetation_carbon.{name}_patch"
                for name in (
                    "m_frootc_to_litter",
                    "m_frootc_to_fire",
                    "m_frootc_to_litter_fire",
                    "hrv_frootc_to_litter",
                    "frootc_to_litter",
                )
            ),
        ),
        (
            "vegetation_carbon.leafc_alloc_patch",
            (
                "vegetation_carbon.leafc_xfer_to_leafc_patch",
                "vegetation_carbon.cpool_to_leafc_patch",
            ),
        ),
        (
            "vegetation_carbon.leafc_loss_patch",
            tuple(
                f"vegetation_carbon.{name}_patch"
                for name in (
                    "m_leafc_to_litter",
                    "m_leafc_to_fire",
                    "m_leafc_to_litter_fire",
                    "hrv_leafc_to_litter",
                    "leafc_to_litter",
                )
            ),
        ),
        ("vegetation_carbon.woodc_alloc_patch", _WOOD_CARBON_ALLOCATION_FIELDS),
        ("vegetation_carbon.woodc_loss_patch", _WOOD_CARBON_LOSS_FIELDS),
        (
            "vegetation_carbon.slash_harvestc_patch",
            _SLASH_HARVEST_CARBON_FIELDS,
        ),
    )
    for output_name, source_names in derived_patch_fields:
        fluxes = _set_masked(
            fluxes,
            output_name,
            _sum_fields(fluxes, source_names),
            patch_mask,
        )

    patch_to_column_fields = (
        (
            "vegetation_carbon.hrv_xsmrpool_to_atm_patch",
            "vegetation_carbon.hrv_xsmrpool_to_atm_col",
        ),
        ("vegetation_carbon.npp_patch", "vegetation_carbon.npp_col"),
        ("vegetation_carbon.rr_patch", "vegetation_carbon.rr_col"),
        ("vegetation_carbon.ar_patch", "vegetation_carbon.ar_col"),
        ("vegetation_carbon.gpp_patch", "vegetation_carbon.gpp_col"),
    )
    for patch_name, column_name in patch_to_column_fields:
        fluxes = _set_masked(
            fluxes,
            column_name,
            _patch_to_column(_field(fluxes, patch_name), inputs),
            column_mask,
        )
    fire_patch_column = _patch_to_column(fire_patch, inputs)

    lag = state.lagged_npp_column
    nfix_seconds = fortran_multiply(
        jnp.asarray(parameters.nfix_time_constant_days, dtype=jnp.float64),
        jnp.float64(_SECONDS_PER_DAY),
    )
    relaxation = strict_exp(
        -fortran_divide(
            jnp.asarray(parameters.time_step, dtype=jnp.float64),
            nfix_seconds,
        )
    )
    updated_lag = fortran_add(
        fortran_multiply(lag, relaxation),
        fortran_multiply(
            _field(fluxes, "vegetation_carbon.npp_col"),
            fortran_add(jnp.float64(1.0), -relaxation),
        ),
    )
    first_lag = lag == jnp.asarray(parameters.special_value, dtype=jnp.float64)
    lag = jnp.where(
        column_mask,
        jnp.where(
            first_lag,
            _field(fluxes, "vegetation_carbon.npp_col"),
            updated_lag,
        ),
        lag,
    )

    carbon_fire_pool = _integrate_levels(
        _field(fluxes, "vegetation_carbon.m_decomp_cpools_to_fire_vr_col"),
        inputs.decomposition_layer_thickness,
        _field(fluxes, "vegetation_carbon.m_decomp_cpools_to_fire_col"),
        column_mask,
    )
    fluxes = _set_masked(
        fluxes,
        "vegetation_carbon.m_decomp_cpools_to_fire_col",
        carbon_fire_pool,
        column_mask,
    )
    fire_column = fortran_ordered_sum(
        jnp.moveaxis(carbon_fire_pool, 1, 0),
        initial=fire_patch_column,
    )
    fluxes = _set_masked(
        fluxes,
        "vegetation_carbon.fire_closs_col",
        fire_column,
        column_mask,
    )
    zero_column = jnp.zeros_like(fire_column)
    for name in (
        "vegetation_carbon.litfire_col",
        "vegetation_carbon.somfire_col",
        "vegetation_carbon.totfire_col",
        "vegetation_carbon.cwdc_hr_col",
    ):
        fluxes = _set_masked(fluxes, name, zero_column, column_mask)
    soil_hr = _field(fluxes, "soil_carbon.hr_col")
    soil_lithr = _field(fluxes, "soil_carbon.lithr_col")
    root_column = _field(fluxes, "vegetation_carbon.rr_col")
    autotrophic_column = _field(fluxes, "vegetation_carbon.ar_col")
    gross_primary_column = _field(fluxes, "vegetation_carbon.gpp_col")
    soil_respiration = fortran_add(root_column, soil_hr)
    ecosystem_respiration = fortran_add(autotrophic_column, soil_hr)
    net_ecosystem_column = fortran_add(gross_primary_column, -ecosystem_respiration)
    for name, values in (
        ("vegetation_carbon.sr_col", soil_respiration),
        ("vegetation_carbon.er_col", ecosystem_respiration),
        ("vegetation_carbon.nep_col", net_ecosystem_column),
    ):
        fluxes = _set_masked(fluxes, name, values, column_mask)

    topology = inputs.topology
    number_of_gridcells = topology.grid.longitude_degrees.shape[0]

    def c2g(values: jnp.ndarray) -> jnp.ndarray:
        return column_to_gridcell_mean(
            values,
            topology.column.gridcell,
            topology.column.weight_on_gridcell,
            topology.column.active,
            number_of_gridcells=number_of_gridcells,
            maximum_columns_per_gridcell=_MAXIMUM_COLUMNS_PER_GRIDCELL,
            special_value=float(CLM_SPECIAL_VALUE),
        )

    net_ecosystem_gridcell = c2g(_field(fluxes, "vegetation_carbon.nep_col"))
    fire_gridcell = c2g(_field(fluxes, "vegetation_carbon.fire_closs_col"))
    harvest_xsmrpool_gridcell = c2g(_field(fluxes, "vegetation_carbon.hrv_xsmrpool_to_atm_col"))
    harvest_dribbled, harvest_amount, harvest_illegal = _dribble_flux(
        harvest_xsmrpool_gridcell,
        state.harvest_xsmrpool_amount_to_dribble_gridcell,
        inputs,
        parameters,
    )
    dynamic_dribbled, dynamic_amount, dynamic_illegal = _dribble_flux(
        inputs.dynamic_landuse_carbon_flux_gridcell,
        state.dynamic_landuse_amount_to_dribble_gridcell,
        inputs,
        parameters,
    )
    net_ecosystem_exchange = fortran_add(
        fortran_add(-net_ecosystem_gridcell, fire_gridcell),
        harvest_dribbled,
    )
    landuse_flux = fortran_add(
        dynamic_dribbled,
        inputs.product_carbon_loss_gridcell,
    )
    net_biome = fortran_add(-net_ecosystem_exchange, -landuse_flux)

    donor = jnp.asarray(parameters.cascade_donor_pool, dtype=jnp.int32)
    pool_is_cwd = jnp.asarray(parameters.pool_is_cwd, dtype=jnp.bool_)
    pool_is_litter = jnp.asarray(parameters.pool_is_litter, dtype=jnp.bool_)
    transition_transfer = _field(
        fluxes,
        "soil_carbon.decomp_cascade_ctransfer_col",
    )
    cwd_loss = jnp.zeros_like(fire_column)
    litter_loss = soil_lithr

    def add_pool_loss(
        pool: int,
        running: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        cwd_running, litter_running = running
        loss = carbon_fire_pool[:, pool]
        return (
            jnp.where(
                column_mask & pool_is_cwd[pool],
                fortran_add(cwd_running, loss),
                cwd_running,
            ),
            jnp.where(
                column_mask & pool_is_litter[pool],
                fortran_add(litter_running, loss),
                litter_running,
            ),
        )

    cwd_loss, litter_loss = lax.fori_loop(
        0,
        NUMBER_OF_DECOMPOSITION_POOLS,
        add_pool_loss,
        (cwd_loss, litter_loss),
    )

    def add_transition_loss(
        transition: int,
        running: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        cwd_running, litter_running = running
        loss = transition_transfer[:, transition]
        donor_pool = donor[transition]
        return (
            jnp.where(
                column_mask & pool_is_cwd[donor_pool],
                fortran_add(cwd_running, loss),
                cwd_running,
            ),
            jnp.where(
                column_mask & pool_is_litter[donor_pool],
                fortran_add(litter_running, loss),
                litter_running,
            ),
        )

    cwd_loss, litter_loss = lax.fori_loop(
        0,
        NUMBER_OF_DECOMPOSITION_TRANSITIONS,
        add_transition_loss,
        (cwd_loss, litter_loss),
    )
    fluxes = _set_masked(
        fluxes,
        "vegetation_carbon.cwdc_loss_col",
        cwd_loss,
        column_mask,
    )
    fluxes = _set_masked(
        fluxes,
        "vegetation_carbon.litterc_loss_col",
        litter_loss,
        column_mask,
    )

    next_state = CnDriverSummarizeFluxesState(
        cn_fluxes=fluxes,
        temporary_npp_sum_patch=temporary_npp,
        lagged_npp_column=lag,
        dynamic_landuse_amount_to_dribble_gridcell=dynamic_amount,
        harvest_xsmrpool_amount_to_dribble_gridcell=harvest_amount,
    )
    gridcell_fluxes = CnDriverSummarizeFluxesGridcellFluxes(
        net_ecosystem_production=net_ecosystem_gridcell,
        fire_carbon_loss=fire_gridcell,
        harvest_xsmrpool_carbon_flux=harvest_xsmrpool_gridcell,
        harvest_xsmrpool_carbon_flux_dribbled=harvest_dribbled,
        dynamic_landuse_carbon_flux_dribbled=dynamic_dribbled,
        net_ecosystem_exchange=net_ecosystem_exchange,
        landuse_carbon_flux=landuse_flux,
        net_biome_production=net_biome,
    )
    illegal = jnp.sum(harvest_illegal, dtype=jnp.int32) + jnp.sum(
        dynamic_illegal,
        dtype=jnp.int32,
    )
    return next_state, gridcell_fluxes, illegal


def _validate_shapes(
    inputs: CnDriverSummarizeFluxesInputs,
    state: CnDriverSummarizeFluxesState,
) -> None:
    patch_shape = inputs.topology.patch.column.shape
    column_shape = inputs.topology.column.gridcell.shape
    grid_shape = inputs.topology.grid.longitude_degrees.shape
    if inputs.soil_patch.shape != patch_shape:
        raise ValueError("soil_patch must contain every native patch")
    if inputs.vegetation_type_patch.shape != patch_shape:
        raise ValueError("vegetation_type_patch must contain every native patch")
    if inputs.soil_column.shape != column_shape:
        raise ValueError("soil_column must contain every native column")
    if inputs.decomposition_layer_thickness.shape != (_ACTIVE_DECOMPOSITION_LEVELS,):
        raise ValueError("decomposition thickness must contain 20 active levels")
    if inputs.product_carbon_loss_gridcell.shape != grid_shape:
        raise ValueError("product carbon loss must contain every gridcell")
    if inputs.dynamic_landuse_carbon_flux_gridcell.shape != grid_shape:
        raise ValueError("dynamic landuse carbon flux must contain every gridcell")
    if state.cn_fluxes.patch_scalar.shape[0] != patch_shape[0]:
        raise ValueError("packed CN fluxes have the wrong patch extent")
    if state.cn_fluxes.column_scalar.shape[0] != column_shape[0]:
        raise ValueError("packed CN fluxes have the wrong column extent")
    if state.temporary_npp_sum_patch.shape != patch_shape:
        raise ValueError("temporary NPP sum must contain every patch")
    if state.lagged_npp_column.shape != column_shape:
        raise ValueError("lagged NPP must contain every column")
    for name in (
        "dynamic_landuse_amount_to_dribble_gridcell",
        "harvest_xsmrpool_amount_to_dribble_gridcell",
    ):
        if getattr(state, name).shape != grid_shape:
            raise ValueError(f"{name} must contain every gridcell")


def cn_driver_summarize_fluxes(
    inputs: CnDriverSummarizeFluxesInputs,
    state: CnDriverSummarizeFluxesState,
    parameters: CnDriverSummarizeFluxesParameters,
) -> CnDriverSummarizeFluxesResult:

    _validate_shapes(inputs, state)
    column_mask = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    thickness = jnp.asarray(inputs.decomposition_layer_thickness, dtype=jnp.float64)
    fluxes = _soil_carbon_summary(
        state.cn_fluxes,
        column_mask,
        thickness,
        parameters,
    )
    fluxes = _soil_nitrogen_summary(fluxes, column_mask, thickness)
    state, gridcell_fluxes, illegal = _vegetation_carbon_summary(
        state._replace(cn_fluxes=fluxes),
        inputs,
        parameters,
    )
    fluxes = _vegetation_nitrogen_summary(state.cn_fluxes, inputs)
    state = state._replace(cn_fluxes=fluxes)
    return CnDriverSummarizeFluxesResult(
        state=state,
        gridcell_fluxes=gridcell_fluxes,
        diagnostics=CnDriverSummarizeFluxesDiagnostics(
            summarized_patch_count=jnp.sum(inputs.soil_patch, dtype=jnp.int32),
            summarized_column_count=jnp.sum(inputs.soil_column, dtype=jnp.int32),
            unexpected_nonannual_delta_count=illegal,
        ),
    )


__all__ = [
    "CnDriverSummarizeFluxesDiagnostics",
    "CnDriverSummarizeFluxesGridcellFluxes",
    "CnDriverSummarizeFluxesInputs",
    "CnDriverSummarizeFluxesParameters",
    "CnDriverSummarizeFluxesResult",
    "CnDriverSummarizeFluxesState",
    "cn_driver_summarize_fluxes",
]
