
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.physics.cn_precision_control import (
    CARBON_ONLY_FIELDS,
    NITROGEN_ONLY_FIELDS,
    PAIRED_CARBON_FIELDS,
    PAIRED_NITROGEN_FIELDS,
    CnPrecisionControlState,
    get_cn_precision_control_state_field,
    set_cn_precision_control_state_field,
)
from clm_jax.physics.cn_state_update2 import CnStateUpdate2Inputs
from clm_jax.physics.cn_zero import CnTransientFluxState, get_cn_flux_field
from clm_jax.physics.cn_zero_manifest import (
    NUMBER_OF_DECOMPOSITION_LEVELS,
    NUMBER_OF_DECOMPOSITION_POOLS,
)
from clm_jax.physics.numerics import fortran_add, fortran_multiply


class CnNStateUpdate3State(NamedTuple):

    vegetation: CnPrecisionControlState
    soil_nitrogen_pool_column_level_pool: jnp.ndarray
    mineral_nitrogen_column_level: jnp.ndarray
    ammonium_column_level: jnp.ndarray
    nitrate_column_level: jnp.ndarray


class CnNStateUpdate3Parameters(NamedTuple):

    time_step: float | jnp.ndarray = 1800.0
    decomposition_level_count: int | jnp.ndarray = 20


DEFAULT_CN_N_STATE_UPDATE3_PARAMETERS = CnNStateUpdate3Parameters()


class CnNStateUpdate3Diagnostics(NamedTuple):
    updated_patch_count: jnp.ndarray
    updated_column_count: jnp.ndarray
    updated_column_level_count: jnp.ndarray
    mineral_state_assignment_count: jnp.ndarray
    vegetation_state_assignment_count: jnp.ndarray
    soil_gain_assignment_count: jnp.ndarray
    soil_loss_assignment_count: jnp.ndarray


class CnNStateUpdate3Result(NamedTuple):
    state: CnNStateUpdate3State
    cn_fluxes: CnTransientFluxState
    diagnostics: CnNStateUpdate3Diagnostics


_PATCH_UPDATE_SEQUENCES = (
    (
        "vegetation_nitrogen.leafn_patch",
        (
            (-1, "vegetation_nitrogen.m_leafn_to_fire_patch"),
            (-1, "vegetation_nitrogen.m_leafn_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_nitrogen.frootn_patch",
        (
            (-1, "vegetation_nitrogen.m_frootn_to_fire_patch"),
            (-1, "vegetation_nitrogen.m_frootn_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_nitrogen.livestemn_patch",
        (
            (-1, "vegetation_nitrogen.m_livestemn_to_fire_patch"),
            (-1, "vegetation_nitrogen.m_livestemn_to_litter_fire_patch"),
            (-1, "vegetation_nitrogen.m_livestemn_to_deadstemn_fire_patch"),
        ),
    ),
    (
        "vegetation_nitrogen.deadstemn_patch",
        (
            (-1, "vegetation_nitrogen.m_deadstemn_to_fire_patch"),
            (-1, "vegetation_nitrogen.m_deadstemn_to_litter_fire_patch"),
            (+1, "vegetation_nitrogen.m_livestemn_to_deadstemn_fire_patch"),
        ),
    ),
    (
        "vegetation_nitrogen.livecrootn_patch",
        (
            (-1, "vegetation_nitrogen.m_livecrootn_to_fire_patch"),
            (-1, "vegetation_nitrogen.m_livecrootn_to_litter_fire_patch"),
            (-1, "vegetation_nitrogen.m_livecrootn_to_deadcrootn_fire_patch"),
        ),
    ),
    (
        "vegetation_nitrogen.deadcrootn_patch",
        (
            (-1, "vegetation_nitrogen.m_deadcrootn_to_fire_patch"),
            (-1, "vegetation_nitrogen.m_deadcrootn_to_litter_fire_patch"),
            (+1, "vegetation_nitrogen.m_livecrootn_to_deadcrootn_fire_patch"),
        ),
    ),
    (
        "vegetation_nitrogen.leafn_storage_patch",
        (
            (-1, "vegetation_nitrogen.m_leafn_storage_to_fire_patch"),
            (-1, "vegetation_nitrogen.m_leafn_storage_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_nitrogen.frootn_storage_patch",
        (
            (-1, "vegetation_nitrogen.m_frootn_storage_to_fire_patch"),
            (-1, "vegetation_nitrogen.m_frootn_storage_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_nitrogen.livestemn_storage_patch",
        (
            (-1, "vegetation_nitrogen.m_livestemn_storage_to_fire_patch"),
            (-1, "vegetation_nitrogen.m_livestemn_storage_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_nitrogen.deadstemn_storage_patch",
        (
            (-1, "vegetation_nitrogen.m_deadstemn_storage_to_fire_patch"),
            (-1, "vegetation_nitrogen.m_deadstemn_storage_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_nitrogen.livecrootn_storage_patch",
        (
            (-1, "vegetation_nitrogen.m_livecrootn_storage_to_fire_patch"),
            (-1, "vegetation_nitrogen.m_livecrootn_storage_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_nitrogen.deadcrootn_storage_patch",
        (
            (-1, "vegetation_nitrogen.m_deadcrootn_storage_to_fire_patch"),
            (-1, "vegetation_nitrogen.m_deadcrootn_storage_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_nitrogen.leafn_xfer_patch",
        (
            (-1, "vegetation_nitrogen.m_leafn_xfer_to_fire_patch"),
            (-1, "vegetation_nitrogen.m_leafn_xfer_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_nitrogen.frootn_xfer_patch",
        (
            (-1, "vegetation_nitrogen.m_frootn_xfer_to_fire_patch"),
            (-1, "vegetation_nitrogen.m_frootn_xfer_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_nitrogen.livestemn_xfer_patch",
        (
            (-1, "vegetation_nitrogen.m_livestemn_xfer_to_fire_patch"),
            (-1, "vegetation_nitrogen.m_livestemn_xfer_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_nitrogen.deadstemn_xfer_patch",
        (
            (-1, "vegetation_nitrogen.m_deadstemn_xfer_to_fire_patch"),
            (-1, "vegetation_nitrogen.m_deadstemn_xfer_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_nitrogen.livecrootn_xfer_patch",
        (
            (-1, "vegetation_nitrogen.m_livecrootn_xfer_to_fire_patch"),
            (-1, "vegetation_nitrogen.m_livecrootn_xfer_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_nitrogen.deadcrootn_xfer_patch",
        (
            (-1, "vegetation_nitrogen.m_deadcrootn_xfer_to_fire_patch"),
            (-1, "vegetation_nitrogen.m_deadcrootn_xfer_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_nitrogen.retransn_patch",
        (
            (-1, "vegetation_nitrogen.m_retransn_to_fire_patch"),
            (-1, "vegetation_nitrogen.m_retransn_to_litter_fire_patch"),
        ),
    ),
)

_SOIL_GAIN_ASSIGNMENTS = (
    (3, "vegetation_nitrogen.fire_mortality_n_to_cwdn_col"),
    (0, "vegetation_nitrogen.m_n_to_litr_met_fire_col"),
    (1, "vegetation_nitrogen.m_n_to_litr_cel_fire_col"),
    (2, "vegetation_nitrogen.m_n_to_litr_lig_fire_col"),
)
_SOIL_FIRE_LOSS = "vegetation_nitrogen.m_decomp_npools_to_fire_vr_col"
_NITRATE_LEACHING = "soil_nitrogen.smin_no3_leached_vr_col"
_NITRATE_RUNOFF = "soil_nitrogen.smin_no3_runoff_vr_col"


def _validate_shapes(
    inputs: CnStateUpdate2Inputs,
    state: CnNStateUpdate3State,
    cn_fluxes: CnTransientFluxState,
    parameters: CnNStateUpdate3Parameters,
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
    if jnp.shape(state.soil_nitrogen_pool_column_level_pool) != pool_shape:
        raise ValueError("soil nitrogen pools must have shape (column, 25, 7)")
    level_shape = (column_count, NUMBER_OF_DECOMPOSITION_LEVELS)
    for name in (
        "mineral_nitrogen_column_level",
        "ammonium_column_level",
        "nitrate_column_level",
    ):
        if jnp.shape(getattr(state, name)) != level_shape:
            raise ValueError(f"{name} must have shape (column, 25)")
    if cn_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("CN patch fluxes must match patch count")
    if cn_fluxes.column_level.shape[:2] != level_shape:
        raise ValueError("CN column-level fluxes have incompatible dimensions")
    if cn_fluxes.column_level_pool.shape[:3] != pool_shape:
        raise ValueError("CN column-level-pool fluxes have incompatible dimensions")
    for name in parameters._fields:
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    return patch_count, column_count


def _update_vegetation_nitrogen(
    vegetation: CnPrecisionControlState,
    cn_fluxes: CnTransientFluxState,
    time_step: jnp.ndarray,
    soil_patch: jnp.ndarray,
) -> CnPrecisionControlState:
    for state_name, operations in _PATCH_UPDATE_SEQUENCES:
        previous = get_cn_precision_control_state_field(vegetation, state_name)
        candidate = previous
        for sign, flux_name in operations:
            increment = fortran_multiply(
                get_cn_flux_field(cn_fluxes, flux_name),
                time_step,
            )
            candidate = fortran_add(candidate, increment if sign > 0 else -increment)
        vegetation = set_cn_precision_control_state_field(
            vegetation,
            state_name,
            jnp.where(soil_patch, candidate, previous),
        )
    return vegetation


def cn_n_state_update3(
    inputs: CnStateUpdate2Inputs,
    state: CnNStateUpdate3State,
    cn_fluxes: CnTransientFluxState,
    parameters: CnNStateUpdate3Parameters = DEFAULT_CN_N_STATE_UPDATE3_PARAMETERS,
) -> CnNStateUpdate3Result:

    _validate_shapes(inputs, state, cn_fluxes, parameters)
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    time_step = jnp.asarray(parameters.time_step, dtype=jnp.float64)
    level_count = jnp.asarray(parameters.decomposition_level_count, dtype=jnp.int32)
    level_active = (
        jnp.arange(NUMBER_OF_DECOMPOSITION_LEVELS, dtype=jnp.int32) < level_count
    )
    update = soil_column[:, None] & level_active[None, :]

    nitrate_previous = jnp.asarray(state.nitrate_column_level, dtype=jnp.float64)
    loss_rate = fortran_add(
        get_cn_flux_field(cn_fluxes, _NITRATE_LEACHING),
        get_cn_flux_field(cn_fluxes, _NITRATE_RUNOFF),
    )
    loss = fortran_multiply(loss_rate, time_step)
    nitrate_candidate = jnp.maximum(
        fortran_add(nitrate_previous, -loss),
        jnp.float64(0.0),
    )
    nitrate = jnp.where(update, nitrate_candidate, nitrate_previous)
    mineral_previous = jnp.asarray(
        state.mineral_nitrogen_column_level,
        dtype=jnp.float64,
    )
    mineral_candidate = fortran_add(
        nitrate,
        jnp.asarray(state.ammonium_column_level, dtype=jnp.float64),
    )
    mineral = jnp.where(update, mineral_candidate, mineral_previous)

    pools = jnp.asarray(
        state.soil_nitrogen_pool_column_level_pool,
        dtype=jnp.float64,
    )
    for pool_index, flux_name in _SOIL_GAIN_ASSIGNMENTS:
        previous = pools[:, :, pool_index]
        gain = fortran_multiply(get_cn_flux_field(cn_fluxes, flux_name), time_step)
        candidate = fortran_add(previous, gain)
        pools = pools.at[:, :, pool_index].set(jnp.where(update, candidate, previous))
    previous_pools = pools
    fire_loss = fortran_multiply(
        get_cn_flux_field(cn_fluxes, _SOIL_FIRE_LOSS),
        time_step,
    )
    pools = jnp.where(
        update[:, :, None],
        fortran_add(previous_pools, -fire_loss),
        previous_pools,
    )

    vegetation = _update_vegetation_nitrogen(
        state.vegetation,
        cn_fluxes,
        time_step,
        soil_patch,
    )
    updated_patch_count = jnp.sum(soil_patch, dtype=jnp.int32)
    updated_column_count = jnp.sum(soil_column, dtype=jnp.int32)
    updated_column_level_count = updated_column_count * level_count
    return CnNStateUpdate3Result(
        state=CnNStateUpdate3State(
            vegetation=vegetation,
            soil_nitrogen_pool_column_level_pool=pools,
            mineral_nitrogen_column_level=mineral,
            ammonium_column_level=state.ammonium_column_level,
            nitrate_column_level=nitrate,
        ),
        cn_fluxes=cn_fluxes,
        diagnostics=CnNStateUpdate3Diagnostics(
            updated_patch_count=updated_patch_count,
            updated_column_count=updated_column_count,
            updated_column_level_count=updated_column_level_count,
            mineral_state_assignment_count=(updated_column_level_count * jnp.int32(2)),
            vegetation_state_assignment_count=(updated_patch_count * jnp.int32(38)),
            soil_gain_assignment_count=(updated_column_level_count * jnp.int32(4)),
            soil_loss_assignment_count=(
                updated_column_level_count
                * jnp.int32(NUMBER_OF_DECOMPOSITION_POOLS)
            ),
        ),
    )


__all__ = [
    "DEFAULT_CN_N_STATE_UPDATE3_PARAMETERS",
    "CnNStateUpdate3Diagnostics",
    "CnNStateUpdate3Parameters",
    "CnNStateUpdate3Result",
    "CnNStateUpdate3State",
    "cn_n_state_update3",
]
