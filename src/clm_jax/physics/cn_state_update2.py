
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.physics.cn_precision_control import (
    CARBON_ONLY_FIELDS,
    DEFAULT_CN_PRECISION_CONTROL_PARAMETERS,
    NITROGEN_ONLY_FIELDS,
    PAIRED_CARBON_FIELDS,
    PAIRED_NITROGEN_FIELDS,
    CnPrecisionControlDiagnostics,
    CnPrecisionControlInputs,
    CnPrecisionControlParameters,
    CnPrecisionControlState,
    cn_precision_control,
    get_cn_precision_control_state_field,
    replace_cn_precision_control_state_fields,
)
from clm_jax.physics.cn_zero import CnTransientFluxState, get_cn_flux_field
from clm_jax.physics.cn_zero_manifest import (
    NUMBER_OF_DECOMPOSITION_LEVELS,
    NUMBER_OF_DECOMPOSITION_POOLS,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)


class CnStateUpdate2Inputs(NamedTuple):

    soil_patch: jnp.ndarray
    soil_column: jnp.ndarray


class CnStateUpdate2State(NamedTuple):

    vegetation: CnPrecisionControlState
    soil_carbon_pool_column_level_pool: jnp.ndarray
    soil_nitrogen_pool_column_level_pool: jnp.ndarray


class CnStateUpdate2Parameters(NamedTuple):

    time_step: float | jnp.ndarray = 1800.0
    decomposition_level_count: int | jnp.ndarray = 20


DEFAULT_CN_STATE_UPDATE2_PARAMETERS = CnStateUpdate2Parameters()


class CnStateUpdate2Diagnostics(NamedTuple):
    updated_patch_count: jnp.ndarray
    updated_column_count: jnp.ndarray
    updated_column_level_count: jnp.ndarray
    vegetation_carbon_assignment_count: jnp.ndarray
    vegetation_nitrogen_assignment_count: jnp.ndarray


class CnStateUpdate2Result(NamedTuple):
    state: CnStateUpdate2State
    cn_fluxes: CnTransientFluxState
    diagnostics: CnStateUpdate2Diagnostics


class CnStateUpdate2hWithPrecisionResult(NamedTuple):

    state: CnStateUpdate2State
    cn_fluxes: CnTransientFluxState
    update_diagnostics: CnStateUpdate2Diagnostics
    precision_diagnostics: CnPrecisionControlDiagnostics


_CARBON_PATCH_UPDATES = (
    ("vegetation_carbon.leafc_patch", "vegetation_carbon.m_leafc_to_litter_patch"),
    ("vegetation_carbon.frootc_patch", "vegetation_carbon.m_frootc_to_litter_patch"),
    (
        "vegetation_carbon.livestemc_patch",
        "vegetation_carbon.m_livestemc_to_litter_patch",
    ),
    (
        "vegetation_carbon.deadstemc_patch",
        "vegetation_carbon.m_deadstemc_to_litter_patch",
    ),
    (
        "vegetation_carbon.livecrootc_patch",
        "vegetation_carbon.m_livecrootc_to_litter_patch",
    ),
    (
        "vegetation_carbon.deadcrootc_patch",
        "vegetation_carbon.m_deadcrootc_to_litter_patch",
    ),
    (
        "vegetation_carbon.leafc_storage_patch",
        "vegetation_carbon.m_leafc_storage_to_litter_patch",
    ),
    (
        "vegetation_carbon.frootc_storage_patch",
        "vegetation_carbon.m_frootc_storage_to_litter_patch",
    ),
    (
        "vegetation_carbon.livestemc_storage_patch",
        "vegetation_carbon.m_livestemc_storage_to_litter_patch",
    ),
    (
        "vegetation_carbon.deadstemc_storage_patch",
        "vegetation_carbon.m_deadstemc_storage_to_litter_patch",
    ),
    (
        "vegetation_carbon.livecrootc_storage_patch",
        "vegetation_carbon.m_livecrootc_storage_to_litter_patch",
    ),
    (
        "vegetation_carbon.deadcrootc_storage_patch",
        "vegetation_carbon.m_deadcrootc_storage_to_litter_patch",
    ),
    (
        "vegetation_carbon.gresp_storage_patch",
        "vegetation_carbon.m_gresp_storage_to_litter_patch",
    ),
    (
        "vegetation_carbon.leafc_xfer_patch",
        "vegetation_carbon.m_leafc_xfer_to_litter_patch",
    ),
    (
        "vegetation_carbon.frootc_xfer_patch",
        "vegetation_carbon.m_frootc_xfer_to_litter_patch",
    ),
    (
        "vegetation_carbon.livestemc_xfer_patch",
        "vegetation_carbon.m_livestemc_xfer_to_litter_patch",
    ),
    (
        "vegetation_carbon.deadstemc_xfer_patch",
        "vegetation_carbon.m_deadstemc_xfer_to_litter_patch",
    ),
    (
        "vegetation_carbon.livecrootc_xfer_patch",
        "vegetation_carbon.m_livecrootc_xfer_to_litter_patch",
    ),
    (
        "vegetation_carbon.deadcrootc_xfer_patch",
        "vegetation_carbon.m_deadcrootc_xfer_to_litter_patch",
    ),
    (
        "vegetation_carbon.gresp_xfer_patch",
        "vegetation_carbon.m_gresp_xfer_to_litter_patch",
    ),
)

_NITROGEN_PATCH_UPDATES = (
    ("vegetation_nitrogen.leafn_patch", "vegetation_nitrogen.m_leafn_to_litter_patch"),
    (
        "vegetation_nitrogen.frootn_patch",
        "vegetation_nitrogen.m_frootn_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.livestemn_patch",
        "vegetation_nitrogen.m_livestemn_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.deadstemn_patch",
        "vegetation_nitrogen.m_deadstemn_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.livecrootn_patch",
        "vegetation_nitrogen.m_livecrootn_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.deadcrootn_patch",
        "vegetation_nitrogen.m_deadcrootn_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.retransn_patch",
        "vegetation_nitrogen.m_retransn_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.leafn_storage_patch",
        "vegetation_nitrogen.m_leafn_storage_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.frootn_storage_patch",
        "vegetation_nitrogen.m_frootn_storage_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.livestemn_storage_patch",
        "vegetation_nitrogen.m_livestemn_storage_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.deadstemn_storage_patch",
        "vegetation_nitrogen.m_deadstemn_storage_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.livecrootn_storage_patch",
        "vegetation_nitrogen.m_livecrootn_storage_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.deadcrootn_storage_patch",
        "vegetation_nitrogen.m_deadcrootn_storage_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.leafn_xfer_patch",
        "vegetation_nitrogen.m_leafn_xfer_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.frootn_xfer_patch",
        "vegetation_nitrogen.m_frootn_xfer_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.livestemn_xfer_patch",
        "vegetation_nitrogen.m_livestemn_xfer_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.deadstemn_xfer_patch",
        "vegetation_nitrogen.m_deadstemn_xfer_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.livecrootn_xfer_patch",
        "vegetation_nitrogen.m_livecrootn_xfer_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.deadcrootn_xfer_patch",
        "vegetation_nitrogen.m_deadcrootn_xfer_to_litter_patch",
    ),
)

_CARBON_COLUMN_UPDATES = (
    "vegetation_carbon.gap_mortality_c_to_litr_met_c_col",
    "vegetation_carbon.gap_mortality_c_to_litr_cel_c_col",
    "vegetation_carbon.gap_mortality_c_to_litr_lig_c_col",
    "vegetation_carbon.gap_mortality_c_to_cwdc_col",
)
_NITROGEN_COLUMN_UPDATES = (
    "vegetation_nitrogen.gap_mortality_n_to_litr_met_n_col",
    "vegetation_nitrogen.gap_mortality_n_to_litr_cel_n_col",
    "vegetation_nitrogen.gap_mortality_n_to_litr_lig_n_col",
    "vegetation_nitrogen.gap_mortality_n_to_cwdn_col",
)

_HARVEST_CARBON_PATCH_UPDATES = (
    ("vegetation_carbon.leafc_patch", "vegetation_carbon.hrv_leafc_to_litter_patch"),
    (
        "vegetation_carbon.frootc_patch",
        "vegetation_carbon.hrv_frootc_to_litter_patch",
    ),
    (
        "vegetation_carbon.livestemc_patch",
        "vegetation_carbon.hrv_livestemc_to_litter_patch",
    ),
    (
        "vegetation_carbon.deadstemc_patch",
        "vegetation_carbon.wood_harvestc_patch",
    ),
    (
        "vegetation_carbon.livecrootc_patch",
        "vegetation_carbon.hrv_livecrootc_to_litter_patch",
    ),
    (
        "vegetation_carbon.deadcrootc_patch",
        "vegetation_carbon.hrv_deadcrootc_to_litter_patch",
    ),
    (
        "vegetation_carbon.xsmrpool_patch",
        "vegetation_carbon.hrv_xsmrpool_to_atm_patch",
    ),
    (
        "vegetation_carbon.leafc_storage_patch",
        "vegetation_carbon.hrv_leafc_storage_to_litter_patch",
    ),
    (
        "vegetation_carbon.frootc_storage_patch",
        "vegetation_carbon.hrv_frootc_storage_to_litter_patch",
    ),
    (
        "vegetation_carbon.livestemc_storage_patch",
        "vegetation_carbon.hrv_livestemc_storage_to_litter_patch",
    ),
    (
        "vegetation_carbon.deadstemc_storage_patch",
        "vegetation_carbon.hrv_deadstemc_storage_to_litter_patch",
    ),
    (
        "vegetation_carbon.livecrootc_storage_patch",
        "vegetation_carbon.hrv_livecrootc_storage_to_litter_patch",
    ),
    (
        "vegetation_carbon.deadcrootc_storage_patch",
        "vegetation_carbon.hrv_deadcrootc_storage_to_litter_patch",
    ),
    (
        "vegetation_carbon.gresp_storage_patch",
        "vegetation_carbon.hrv_gresp_storage_to_litter_patch",
    ),
    (
        "vegetation_carbon.leafc_xfer_patch",
        "vegetation_carbon.hrv_leafc_xfer_to_litter_patch",
    ),
    (
        "vegetation_carbon.frootc_xfer_patch",
        "vegetation_carbon.hrv_frootc_xfer_to_litter_patch",
    ),
    (
        "vegetation_carbon.livestemc_xfer_patch",
        "vegetation_carbon.hrv_livestemc_xfer_to_litter_patch",
    ),
    (
        "vegetation_carbon.deadstemc_xfer_patch",
        "vegetation_carbon.hrv_deadstemc_xfer_to_litter_patch",
    ),
    (
        "vegetation_carbon.livecrootc_xfer_patch",
        "vegetation_carbon.hrv_livecrootc_xfer_to_litter_patch",
    ),
    (
        "vegetation_carbon.deadcrootc_xfer_patch",
        "vegetation_carbon.hrv_deadcrootc_xfer_to_litter_patch",
    ),
    (
        "vegetation_carbon.gresp_xfer_patch",
        "vegetation_carbon.hrv_gresp_xfer_to_litter_patch",
    ),
)
_HARVEST_NITROGEN_PATCH_UPDATES = (
    (
        "vegetation_nitrogen.leafn_patch",
        "vegetation_nitrogen.hrv_leafn_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.frootn_patch",
        "vegetation_nitrogen.hrv_frootn_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.livestemn_patch",
        "vegetation_nitrogen.hrv_livestemn_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.deadstemn_patch",
        "vegetation_nitrogen.wood_harvestn_patch",
    ),
    (
        "vegetation_nitrogen.livecrootn_patch",
        "vegetation_nitrogen.hrv_livecrootn_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.deadcrootn_patch",
        "vegetation_nitrogen.hrv_deadcrootn_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.retransn_patch",
        "vegetation_nitrogen.hrv_retransn_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.leafn_storage_patch",
        "vegetation_nitrogen.hrv_leafn_storage_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.frootn_storage_patch",
        "vegetation_nitrogen.hrv_frootn_storage_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.livestemn_storage_patch",
        "vegetation_nitrogen.hrv_livestemn_storage_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.deadstemn_storage_patch",
        "vegetation_nitrogen.hrv_deadstemn_storage_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.livecrootn_storage_patch",
        "vegetation_nitrogen.hrv_livecrootn_storage_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.deadcrootn_storage_patch",
        "vegetation_nitrogen.hrv_deadcrootn_storage_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.leafn_xfer_patch",
        "vegetation_nitrogen.hrv_leafn_xfer_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.frootn_xfer_patch",
        "vegetation_nitrogen.hrv_frootn_xfer_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.livestemn_xfer_patch",
        "vegetation_nitrogen.hrv_livestemn_xfer_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.deadstemn_xfer_patch",
        "vegetation_nitrogen.hrv_deadstemn_xfer_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.livecrootn_xfer_patch",
        "vegetation_nitrogen.hrv_livecrootn_xfer_to_litter_patch",
    ),
    (
        "vegetation_nitrogen.deadcrootn_xfer_patch",
        "vegetation_nitrogen.hrv_deadcrootn_xfer_to_litter_patch",
    ),
)
_HARVEST_CARBON_COLUMN_UPDATES = (
    "vegetation_carbon.harvest_c_to_litr_met_c_col",
    "vegetation_carbon.harvest_c_to_litr_cel_c_col",
    "vegetation_carbon.harvest_c_to_litr_lig_c_col",
    "vegetation_carbon.harvest_c_to_cwdc_col",
)
_HARVEST_NITROGEN_COLUMN_UPDATES = (
    "vegetation_nitrogen.harvest_n_to_litr_met_n_col",
    "vegetation_nitrogen.harvest_n_to_litr_cel_n_col",
    "vegetation_nitrogen.harvest_n_to_litr_lig_n_col",
    "vegetation_nitrogen.harvest_n_to_cwdn_col",
)
CN_STATE_UPDATE2H_FLUX_FIELDS = (
    *(flux_name for _, flux_name in _HARVEST_CARBON_PATCH_UPDATES),
    *(flux_name for _, flux_name in _HARVEST_NITROGEN_PATCH_UPDATES),
    *_HARVEST_CARBON_COLUMN_UPDATES,
    *_HARVEST_NITROGEN_COLUMN_UPDATES,
)


def _validate_shapes(
    inputs: CnStateUpdate2Inputs,
    state: CnStateUpdate2State,
    cn_fluxes: CnTransientFluxState,
    parameters: CnStateUpdate2Parameters,
) -> tuple[int, int]:
    if jnp.ndim(inputs.soil_patch) != 1 or jnp.ndim(inputs.soil_column) != 1:
        raise ValueError("soil_patch and soil_column must be vectors")
    patch_count = inputs.soil_patch.shape[0]
    column_count = inputs.soil_column.shape[0]
    vegetation_shapes = {
        "paired_carbon_patch": (patch_count, len(PAIRED_CARBON_FIELDS)),
        "paired_nitrogen_patch": (patch_count, len(PAIRED_NITROGEN_FIELDS)),
        "carbon_only_patch": (patch_count, len(CARBON_ONLY_FIELDS)),
        "nitrogen_only_patch": (patch_count, len(NITROGEN_ONLY_FIELDS)),
        "carbon_truncation_patch": (patch_count,),
        "nitrogen_truncation_patch": (patch_count,),
    }
    for name, expected in vegetation_shapes.items():
        if jnp.shape(getattr(state.vegetation, name)) != expected:
            raise ValueError(f"vegetation {name} has an incompatible shape")
    pool_shape = (
        column_count,
        NUMBER_OF_DECOMPOSITION_LEVELS,
        NUMBER_OF_DECOMPOSITION_POOLS,
    )
    for name in (
        "soil_carbon_pool_column_level_pool",
        "soil_nitrogen_pool_column_level_pool",
    ):
        if jnp.shape(getattr(state, name)) != pool_shape:
            raise ValueError(f"{name} must have shape (column, 25, 7)")
    if cn_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("CN patch fluxes must match patch count")
    if cn_fluxes.column_level.shape[:2] != (
        column_count,
        NUMBER_OF_DECOMPOSITION_LEVELS,
    ):
        raise ValueError("CN column-level fluxes have incompatible dimensions")
    for name in parameters._fields:
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    return patch_count, column_count


def _update_vegetation(
    vegetation: CnPrecisionControlState,
    updates: tuple[tuple[str, str], ...],
    cn_fluxes: CnTransientFluxState,
    time_step: jnp.ndarray,
    soil_patch: jnp.ndarray,
) -> CnPrecisionControlState:
    changed = {}
    for state_name, flux_name in updates:
        previous = (
            changed[state_name] if state_name in changed
            else get_cn_precision_control_state_field(vegetation, state_name)
        )
        loss = fortran_multiply(get_cn_flux_field(cn_fluxes, flux_name), time_step)
        candidate = fortran_add(previous, -loss)
        changed[state_name] = jnp.where(soil_patch, candidate, previous)
    return replace_cn_precision_control_state_fields(vegetation, changed)


def _update_soil_pools(
    pools: jnp.ndarray,
    flux_names: tuple[str, str, str, str],
    cn_fluxes: CnTransientFluxState,
    time_step: jnp.ndarray,
    update: jnp.ndarray,
) -> jnp.ndarray:
    updated = [pools[:, :, index] for index in range(pools.shape[-1])]
    for pool_index, flux_name in enumerate(flux_names):
        previous = updated[pool_index]
        gain = fortran_multiply(get_cn_flux_field(cn_fluxes, flux_name), time_step)
        candidate = fortran_add(previous, gain)
        updated[pool_index] = jnp.where(update, candidate, previous)
    return jnp.stack(updated, axis=-1)


def _cn_state_update2_with_flux_maps(
    inputs: CnStateUpdate2Inputs,
    state: CnStateUpdate2State,
    cn_fluxes: CnTransientFluxState,
    parameters: CnStateUpdate2Parameters,
    carbon_patch_updates: tuple[tuple[str, str], ...],
    nitrogen_patch_updates: tuple[tuple[str, str], ...],
    carbon_column_updates: tuple[str, str, str, str],
    nitrogen_column_updates: tuple[str, str, str, str],
) -> CnStateUpdate2Result:
    _validate_shapes(inputs, state, cn_fluxes, parameters)
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    time_step = jnp.asarray(parameters.time_step, dtype=jnp.float64)
    decomposition_level_count = jnp.asarray(
        parameters.decomposition_level_count,
        dtype=jnp.int32,
    )
    level_active = (
        jnp.arange(
            NUMBER_OF_DECOMPOSITION_LEVELS,
            dtype=jnp.int32,
        )
        < decomposition_level_count
    )
    column_level_update = soil_column[:, None] & level_active[None, :]

    carbon_pool = _update_soil_pools(
        jnp.asarray(state.soil_carbon_pool_column_level_pool, dtype=jnp.float64),
        carbon_column_updates,
        cn_fluxes,
        time_step,
        column_level_update,
    )
    vegetation = _update_vegetation(
        state.vegetation,
        carbon_patch_updates,
        cn_fluxes,
        time_step,
        soil_patch,
    )
    nitrogen_pool = _update_soil_pools(
        jnp.asarray(state.soil_nitrogen_pool_column_level_pool, dtype=jnp.float64),
        nitrogen_column_updates,
        cn_fluxes,
        time_step,
        column_level_update,
    )
    vegetation = _update_vegetation(
        vegetation,
        nitrogen_patch_updates,
        cn_fluxes,
        time_step,
        soil_patch,
    )

    updated_patch_count = jnp.sum(soil_patch, dtype=jnp.int32)
    updated_column_count = jnp.sum(soil_column, dtype=jnp.int32)
    return CnStateUpdate2Result(
        state=CnStateUpdate2State(
            vegetation=vegetation,
            soil_carbon_pool_column_level_pool=carbon_pool,
            soil_nitrogen_pool_column_level_pool=nitrogen_pool,
        ),
        cn_fluxes=cn_fluxes,
        diagnostics=CnStateUpdate2Diagnostics(
            updated_patch_count=updated_patch_count,
            updated_column_count=updated_column_count,
            updated_column_level_count=(updated_column_count * decomposition_level_count),
            vegetation_carbon_assignment_count=(
                updated_patch_count * jnp.int32(len(carbon_patch_updates))
            ),
            vegetation_nitrogen_assignment_count=(
                updated_patch_count * jnp.int32(len(nitrogen_patch_updates))
            ),
        ),
    )


def cn_state_update2(
    inputs: CnStateUpdate2Inputs,
    state: CnStateUpdate2State,
    cn_fluxes: CnTransientFluxState,
    parameters: CnStateUpdate2Parameters = DEFAULT_CN_STATE_UPDATE2_PARAMETERS,
) -> CnStateUpdate2Result:

    return _cn_state_update2_with_flux_maps(
        inputs,
        state,
        cn_fluxes,
        parameters,
        _CARBON_PATCH_UPDATES,
        _NITROGEN_PATCH_UPDATES,
        _CARBON_COLUMN_UPDATES,
        _NITROGEN_COLUMN_UPDATES,
    )


def cn_state_update2h(
    inputs: CnStateUpdate2Inputs,
    state: CnStateUpdate2State,
    cn_fluxes: CnTransientFluxState,
    parameters: CnStateUpdate2Parameters = DEFAULT_CN_STATE_UPDATE2_PARAMETERS,
) -> CnStateUpdate2Result:

    return _cn_state_update2_with_flux_maps(
        inputs,
        state,
        cn_fluxes,
        parameters,
        _HARVEST_CARBON_PATCH_UPDATES,
        _HARVEST_NITROGEN_PATCH_UPDATES,
        _HARVEST_CARBON_COLUMN_UPDATES,
        _HARVEST_NITROGEN_COLUMN_UPDATES,
    )


def cn_state_update2h_with_precision_control(
    inputs: CnStateUpdate2Inputs,
    state: CnStateUpdate2State,
    cn_fluxes: CnTransientFluxState,
    parameters: CnStateUpdate2Parameters,
    precision_inputs: CnPrecisionControlInputs,
    precision_parameters: CnPrecisionControlParameters = (
        DEFAULT_CN_PRECISION_CONTROL_PARAMETERS
    ),
) -> CnStateUpdate2hWithPrecisionResult:

    if jnp.shape(precision_inputs.soil_patch) != jnp.shape(inputs.soil_patch):
        raise ValueError("Update2h and precision-control patch filters differ")
    update = cn_state_update2h(inputs, state, cn_fluxes, parameters)
    precision = cn_precision_control(
        precision_inputs,
        update.state.vegetation,
        precision_parameters,
    )
    return CnStateUpdate2hWithPrecisionResult(
        state=update.state._replace(vegetation=precision.state),
        cn_fluxes=update.cn_fluxes,
        update_diagnostics=update.diagnostics,
        precision_diagnostics=precision.diagnostics,
    )


__all__ = [
    "CN_STATE_UPDATE2H_FLUX_FIELDS",
    "CnStateUpdate2Diagnostics",
    "CnStateUpdate2Inputs",
    "CnStateUpdate2Parameters",
    "CnStateUpdate2Result",
    "CnStateUpdate2State",
    "CnStateUpdate2hWithPrecisionResult",
    "DEFAULT_CN_STATE_UPDATE2_PARAMETERS",
    "cn_state_update2",
    "cn_state_update2h",
    "cn_state_update2h_with_precision_control",
]
