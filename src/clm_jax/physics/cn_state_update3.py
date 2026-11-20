
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
    cn_precision_control,
    get_cn_precision_control_state_field,
    replace_cn_precision_control_state_fields,
)
from clm_jax.physics.cn_state_update2 import CnStateUpdate2Inputs, CnStateUpdate2State
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


class CnStateUpdate3Parameters(NamedTuple):

    time_step: float | jnp.ndarray = 1800.0
    decomposition_level_count: int | jnp.ndarray = 20


DEFAULT_CN_STATE_UPDATE3_PARAMETERS = CnStateUpdate3Parameters()


class CnStateUpdate3Diagnostics(NamedTuple):
    updated_patch_count: jnp.ndarray
    updated_column_count: jnp.ndarray
    updated_column_level_count: jnp.ndarray
    vegetation_state_assignment_count: jnp.ndarray
    soil_gain_assignment_count: jnp.ndarray
    soil_loss_assignment_count: jnp.ndarray


class CnStateUpdate3Result(NamedTuple):
    state: CnStateUpdate2State
    cn_fluxes: CnTransientFluxState
    diagnostics: CnStateUpdate3Diagnostics


class CnStateUpdate3WithPrecisionResult(NamedTuple):

    state: CnStateUpdate2State
    cn_fluxes: CnTransientFluxState
    update_diagnostics: CnStateUpdate3Diagnostics
    precision_diagnostics: CnPrecisionControlDiagnostics


_PATCH_UPDATE_SEQUENCES = (
    (
        "vegetation_carbon.leafc_patch",
        (
            (-1, "vegetation_carbon.m_leafc_to_fire_patch"),
            (-1, "vegetation_carbon.m_leafc_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_carbon.frootc_patch",
        (
            (-1, "vegetation_carbon.m_frootc_to_fire_patch"),
            (-1, "vegetation_carbon.m_frootc_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_carbon.livestemc_patch",
        (
            (-1, "vegetation_carbon.m_livestemc_to_fire_patch"),
            (-1, "vegetation_carbon.m_livestemc_to_litter_fire_patch"),
            (-1, "vegetation_carbon.m_livestemc_to_deadstemc_fire_patch"),
        ),
    ),
    (
        "vegetation_carbon.deadstemc_patch",
        (
            (-1, "vegetation_carbon.m_deadstemc_to_fire_patch"),
            (-1, "vegetation_carbon.m_deadstemc_to_litter_fire_patch"),
            (+1, "vegetation_carbon.m_livestemc_to_deadstemc_fire_patch"),
        ),
    ),
    (
        "vegetation_carbon.livecrootc_patch",
        (
            (-1, "vegetation_carbon.m_livecrootc_to_fire_patch"),
            (-1, "vegetation_carbon.m_livecrootc_to_litter_fire_patch"),
            (-1, "vegetation_carbon.m_livecrootc_to_deadcrootc_fire_patch"),
        ),
    ),
    (
        "vegetation_carbon.deadcrootc_patch",
        (
            (-1, "vegetation_carbon.m_deadcrootc_to_fire_patch"),
            (-1, "vegetation_carbon.m_deadcrootc_to_litter_fire_patch"),
            (+1, "vegetation_carbon.m_livecrootc_to_deadcrootc_fire_patch"),
        ),
    ),
    (
        "vegetation_carbon.leafc_storage_patch",
        (
            (-1, "vegetation_carbon.m_leafc_storage_to_fire_patch"),
            (-1, "vegetation_carbon.m_leafc_storage_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_carbon.frootc_storage_patch",
        (
            (-1, "vegetation_carbon.m_frootc_storage_to_fire_patch"),
            (-1, "vegetation_carbon.m_frootc_storage_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_carbon.livestemc_storage_patch",
        (
            (-1, "vegetation_carbon.m_livestemc_storage_to_fire_patch"),
            (-1, "vegetation_carbon.m_livestemc_storage_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_carbon.deadstemc_storage_patch",
        (
            (-1, "vegetation_carbon.m_deadstemc_storage_to_fire_patch"),
            (-1, "vegetation_carbon.m_deadstemc_storage_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_carbon.livecrootc_storage_patch",
        (
            (-1, "vegetation_carbon.m_livecrootc_storage_to_fire_patch"),
            (-1, "vegetation_carbon.m_livecrootc_storage_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_carbon.deadcrootc_storage_patch",
        (
            (-1, "vegetation_carbon.m_deadcrootc_storage_to_fire_patch"),
            (-1, "vegetation_carbon.m_deadcrootc_storage_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_carbon.gresp_storage_patch",
        (
            (-1, "vegetation_carbon.m_gresp_storage_to_fire_patch"),
            (-1, "vegetation_carbon.m_gresp_storage_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_carbon.leafc_xfer_patch",
        (
            (-1, "vegetation_carbon.m_leafc_xfer_to_fire_patch"),
            (-1, "vegetation_carbon.m_leafc_xfer_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_carbon.frootc_xfer_patch",
        (
            (-1, "vegetation_carbon.m_frootc_xfer_to_fire_patch"),
            (-1, "vegetation_carbon.m_frootc_xfer_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_carbon.livestemc_xfer_patch",
        (
            (-1, "vegetation_carbon.m_livestemc_xfer_to_fire_patch"),
            (-1, "vegetation_carbon.m_livestemc_xfer_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_carbon.deadstemc_xfer_patch",
        (
            (-1, "vegetation_carbon.m_deadstemc_xfer_to_fire_patch"),
            (-1, "vegetation_carbon.m_deadstemc_xfer_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_carbon.livecrootc_xfer_patch",
        (
            (-1, "vegetation_carbon.m_livecrootc_xfer_to_fire_patch"),
            (-1, "vegetation_carbon.m_livecrootc_xfer_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_carbon.deadcrootc_xfer_patch",
        (
            (-1, "vegetation_carbon.m_deadcrootc_xfer_to_fire_patch"),
            (-1, "vegetation_carbon.m_deadcrootc_xfer_to_litter_fire_patch"),
        ),
    ),
    (
        "vegetation_carbon.gresp_xfer_patch",
        (
            (-1, "vegetation_carbon.m_gresp_xfer_to_fire_patch"),
            (-1, "vegetation_carbon.m_gresp_xfer_to_litter_fire_patch"),
        ),
    ),
)

_SOIL_GAIN_ASSIGNMENTS = (
    (3, "vegetation_carbon.fire_mortality_c_to_cwdc_col"),
    (0, "vegetation_carbon.m_c_to_litr_met_fire_col"),
    (1, "vegetation_carbon.m_c_to_litr_cel_fire_col"),
    (2, "vegetation_carbon.m_c_to_litr_lig_fire_col"),
)
_SOIL_FIRE_LOSS = "vegetation_carbon.m_decomp_cpools_to_fire_vr_col"


def _validate_shapes(
    inputs: CnStateUpdate2Inputs,
    state: CnStateUpdate2State,
    cn_fluxes: CnTransientFluxState,
    parameters: CnStateUpdate3Parameters,
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
    if jnp.shape(state.soil_carbon_pool_column_level_pool) != pool_shape:
        raise ValueError("soil carbon pools must have shape (column, 25, 7)")
    if jnp.shape(state.soil_nitrogen_pool_column_level_pool) != pool_shape:
        raise ValueError("soil nitrogen pools must have shape (column, 25, 7)")
    if cn_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("CN patch fluxes must match patch count")
    if cn_fluxes.column_level.shape[:2] != (
        column_count,
        NUMBER_OF_DECOMPOSITION_LEVELS,
    ):
        raise ValueError("CN column-level fluxes have incompatible dimensions")
    if cn_fluxes.column_level_pool.shape[:3] != pool_shape:
        raise ValueError("CN column-level-pool fluxes have incompatible dimensions")
    for name in parameters._fields:
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    return patch_count, column_count


def _update_soil_carbon(
    pools: jnp.ndarray,
    cn_fluxes: CnTransientFluxState,
    time_step: jnp.ndarray,
    update: jnp.ndarray,
) -> jnp.ndarray:
    updated = [pools[:, :, index] for index in range(pools.shape[-1])]
    for pool_index, flux_name in _SOIL_GAIN_ASSIGNMENTS:
        previous = updated[pool_index]
        gain = fortran_multiply(get_cn_flux_field(cn_fluxes, flux_name), time_step)
        candidate = fortran_add(previous, gain)
        updated[pool_index] = jnp.where(update, candidate, previous)
    pools = jnp.stack(updated, axis=-1)

    previous = pools
    loss = fortran_multiply(get_cn_flux_field(cn_fluxes, _SOIL_FIRE_LOSS), time_step)
    candidate = fortran_add(previous, -loss)
    return jnp.where(update[:, :, None], candidate, previous)


def _update_vegetation_carbon(
    vegetation,
    cn_fluxes: CnTransientFluxState,
    time_step: jnp.ndarray,
    soil_patch: jnp.ndarray,
):
    changed = {}
    for state_name, operations in _PATCH_UPDATE_SEQUENCES:
        previous = (
            changed[state_name] if state_name in changed
            else get_cn_precision_control_state_field(vegetation, state_name)
        )
        candidate = previous
        for sign, flux_name in operations:
            increment = fortran_multiply(get_cn_flux_field(cn_fluxes, flux_name), time_step)
            candidate = fortran_add(candidate, increment if sign > 0 else -increment)
        changed[state_name] = jnp.where(soil_patch, candidate, previous)
    return replace_cn_precision_control_state_fields(vegetation, changed)


def cn_state_update3(
    inputs: CnStateUpdate2Inputs,
    state: CnStateUpdate2State,
    cn_fluxes: CnTransientFluxState,
    parameters: CnStateUpdate3Parameters = DEFAULT_CN_STATE_UPDATE3_PARAMETERS,
) -> CnStateUpdate3Result:

    _validate_shapes(inputs, state, cn_fluxes, parameters)
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    time_step = jnp.asarray(parameters.time_step, dtype=jnp.float64)
    level_count = jnp.asarray(parameters.decomposition_level_count, dtype=jnp.int32)
    level_active = jnp.arange(NUMBER_OF_DECOMPOSITION_LEVELS, dtype=jnp.int32) < level_count
    column_level_update = soil_column[:, None] & level_active[None, :]

    soil_carbon = _update_soil_carbon(
        jnp.asarray(state.soil_carbon_pool_column_level_pool, dtype=jnp.float64),
        cn_fluxes,
        time_step,
        column_level_update,
    )
    vegetation = _update_vegetation_carbon(
        state.vegetation,
        cn_fluxes,
        time_step,
        soil_patch,
    )
    updated_patch_count = jnp.sum(soil_patch, dtype=jnp.int32)
    updated_column_count = jnp.sum(soil_column, dtype=jnp.int32)
    updated_column_level_count = updated_column_count * level_count
    return CnStateUpdate3Result(
        state=CnStateUpdate2State(
            vegetation=vegetation,
            soil_carbon_pool_column_level_pool=soil_carbon,
            soil_nitrogen_pool_column_level_pool=state.soil_nitrogen_pool_column_level_pool,
        ),
        cn_fluxes=cn_fluxes,
        diagnostics=CnStateUpdate3Diagnostics(
            updated_patch_count=updated_patch_count,
            updated_column_count=updated_column_count,
            updated_column_level_count=updated_column_level_count,
            vegetation_state_assignment_count=(updated_patch_count * jnp.int32(40)),
            soil_gain_assignment_count=(updated_column_level_count * jnp.int32(4)),
            soil_loss_assignment_count=(
                updated_column_level_count * jnp.int32(NUMBER_OF_DECOMPOSITION_POOLS)
            ),
        ),
    )


def cn_state_update3_with_precision_control(
    inputs: CnStateUpdate2Inputs,
    state: CnStateUpdate2State,
    cn_fluxes: CnTransientFluxState,
    parameters: CnStateUpdate3Parameters,
    precision_inputs: CnPrecisionControlInputs,
    precision_parameters: CnPrecisionControlParameters = (DEFAULT_CN_PRECISION_CONTROL_PARAMETERS),
) -> CnStateUpdate3WithPrecisionResult:

    if jnp.shape(precision_inputs.soil_patch) != jnp.shape(inputs.soil_patch):
        raise ValueError("Update3 and precision-control patch filters differ")
    update = cn_state_update3(inputs, state, cn_fluxes, parameters)
    precision = cn_precision_control(
        precision_inputs,
        update.state.vegetation,
        precision_parameters,
    )
    return CnStateUpdate3WithPrecisionResult(
        state=update.state._replace(vegetation=precision.state),
        cn_fluxes=update.cn_fluxes,
        update_diagnostics=update.diagnostics,
        precision_diagnostics=precision.diagnostics,
    )


__all__ = [
    "DEFAULT_CN_STATE_UPDATE3_PARAMETERS",
    "CnStateUpdate3Diagnostics",
    "CnStateUpdate3Parameters",
    "CnStateUpdate3Result",
    "CnStateUpdate3WithPrecisionResult",
    "cn_state_update3",
    "cn_state_update3_with_precision_control",
]
