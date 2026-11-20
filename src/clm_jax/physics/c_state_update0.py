
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.physics.cn_zero import CnTransientFluxState, get_cn_flux_field
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)


class CStateUpdate0Inputs(NamedTuple):

    soil_patch: jnp.ndarray


class CStateUpdate0State(NamedTuple):

    carbon_pool_patch: jnp.ndarray


class CStateUpdate0Parameters(NamedTuple):

    time_step_seconds: float | jnp.ndarray = 1800.0


DEFAULT_C_STATE_UPDATE0_PARAMETERS = CStateUpdate0Parameters()


class CStateUpdate0Diagnostics(NamedTuple):
    updated_patch_count: jnp.ndarray


class CStateUpdate0Result(NamedTuple):
    state: CStateUpdate0State
    diagnostics: CStateUpdate0Diagnostics


_SUNLIT_PHOTOSYNTHESIS = "vegetation_carbon.psnsun_to_cpool_patch"
_SHADED_PHOTOSYNTHESIS = "vegetation_carbon.psnshade_to_cpool_patch"


def _validate_shapes(
    inputs: CStateUpdate0Inputs,
    state: CStateUpdate0State,
    cn_fluxes: CnTransientFluxState,
    parameters: CStateUpdate0Parameters,
) -> int:
    if jnp.ndim(inputs.soil_patch) != 1:
        raise ValueError("soil_patch must be a vector")
    patch_count = inputs.soil_patch.shape[0]
    if jnp.shape(state.carbon_pool_patch) != (patch_count,):
        raise ValueError("carbon_pool_patch must contain every patch")
    for field_name in (_SUNLIT_PHOTOSYNTHESIS, _SHADED_PHOTOSYNTHESIS):
        if get_cn_flux_field(cn_fluxes, field_name).shape != (patch_count,):
            raise ValueError("photosynthesis fluxes must match patch count")
    if jnp.shape(parameters.time_step_seconds) != ():
        raise ValueError("time_step_seconds must be scalar")
    return patch_count


def c_state_update0(
    inputs: CStateUpdate0Inputs,
    state: CStateUpdate0State,
    cn_fluxes: CnTransientFluxState,
    parameters: CStateUpdate0Parameters = DEFAULT_C_STATE_UPDATE0_PARAMETERS,
) -> CStateUpdate0Result:

    _validate_shapes(inputs, state, cn_fluxes, parameters)
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    carbon_pool = jnp.asarray(state.carbon_pool_patch, dtype=jnp.float64)
    time_step = jnp.asarray(parameters.time_step_seconds, dtype=jnp.float64)
    sunlit = get_cn_flux_field(cn_fluxes, _SUNLIT_PHOTOSYNTHESIS)
    shaded = get_cn_flux_field(cn_fluxes, _SHADED_PHOTOSYNTHESIS)

    updated = fortran_add(
        carbon_pool,
        fortran_multiply(sunlit, time_step),
    )
    updated = fortran_add(
        updated,
        fortran_multiply(shaded, time_step),
    )
    carbon_pool = jnp.where(soil_patch, updated, carbon_pool)

    return CStateUpdate0Result(
        state=CStateUpdate0State(carbon_pool_patch=carbon_pool),
        diagnostics=CStateUpdate0Diagnostics(
            updated_patch_count=jnp.sum(soil_patch, dtype=jnp.int32),
        ),
    )


__all__ = [
    "CStateUpdate0Diagnostics",
    "CStateUpdate0Inputs",
    "CStateUpdate0Parameters",
    "CStateUpdate0Result",
    "CStateUpdate0State",
    "DEFAULT_C_STATE_UPDATE0_PARAMETERS",
    "c_state_update0",
]
