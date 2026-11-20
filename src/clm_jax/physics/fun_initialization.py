
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.physics.numerics import (
    fortran_divide,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)


class FunInitializationInputs(NamedTuple):

    vegetation_type_patch: jnp.ndarray
    model_step: int | jnp.ndarray


class FunInitializationState(NamedTuple):

    leaf_cn_ratio_offset_patch: jnp.ndarray
    storage_carbon_demand_patch: jnp.ndarray
    storage_nitrogen_demand_patch: jnp.ndarray
    leaf_carbon_storage_transfer_accumulator_patch: jnp.ndarray
    leaf_nitrogen_storage_transfer_accumulator_patch: jnp.ndarray


class FunInitializationParameters(NamedTuple):

    leaf_cn_ratio_pft: jnp.ndarray
    time_step_seconds: float | jnp.ndarray = 1800.0
    days_per_year: float | jnp.ndarray = 365.0
    onset_period_days: float | jnp.ndarray = 30.0
    offset_period_days: float | jnp.ndarray = 15.0
    seconds_per_day: float | jnp.ndarray = 86_400.0


_REFERENCE_LEAF_CN_RATIO = np.full(79, 25.0, dtype=np.float64)
DEFAULT_FUN_INITIALIZATION_PARAMETERS = FunInitializationParameters(
    leaf_cn_ratio_pft=_REFERENCE_LEAF_CN_RATIO,
)


class FunInitializationRuntimeState(NamedTuple):

    time_step_seconds: jnp.ndarray
    onset_period_days: jnp.ndarray
    offset_period_days: jnp.ndarray
    steps_per_year: jnp.ndarray


class FunInitializationDiagnostics(NamedTuple):
    annual_reset: jnp.ndarray
    reset_patch_count: jnp.ndarray


class FunInitializationResult(NamedTuple):
    state: FunInitializationState
    runtime: FunInitializationRuntimeState
    diagnostics: FunInitializationDiagnostics


def _validate_shapes(
    inputs: FunInitializationInputs,
    state: FunInitializationState,
    parameters: FunInitializationParameters,
) -> int:
    if jnp.ndim(inputs.vegetation_type_patch) != 1:
        raise ValueError("vegetation_type_patch must be a vector")
    patch_count = inputs.vegetation_type_patch.shape[0]
    if jnp.shape(inputs.model_step) != ():
        raise ValueError("model_step must be scalar")
    for name in FunInitializationState._fields:
        if jnp.shape(getattr(state, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    if (
        jnp.ndim(parameters.leaf_cn_ratio_pft) != 1
        or parameters.leaf_cn_ratio_pft.shape[0] < 1
    ):
        raise ValueError("leaf_cn_ratio_pft must be a nonempty PFT vector")
    for name in (
        "time_step_seconds",
        "days_per_year",
        "onset_period_days",
        "offset_period_days",
        "seconds_per_day",
    ):
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    return patch_count


def fun_initialization(
    inputs: FunInitializationInputs,
    state: FunInitializationState,
    parameters: FunInitializationParameters = DEFAULT_FUN_INITIALIZATION_PARAMETERS,
) -> FunInitializationResult:

    patch_count = _validate_shapes(inputs, state, parameters)
    vegetation_type = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    model_step = jnp.asarray(inputs.model_step, dtype=jnp.int32)
    leaf_cn_ratio_pft = jnp.asarray(
        parameters.leaf_cn_ratio_pft,
        dtype=jnp.float64,
    )
    time_step = jnp.asarray(parameters.time_step_seconds, dtype=jnp.float64)
    days_per_year = jnp.asarray(parameters.days_per_year, dtype=jnp.float64)
    seconds_per_day = jnp.asarray(parameters.seconds_per_day, dtype=jnp.float64)
    onset_period = jnp.asarray(parameters.onset_period_days, dtype=jnp.float64)
    offset_period = jnp.asarray(parameters.offset_period_days, dtype=jnp.float64)

    seconds_per_year = fortran_multiply(seconds_per_day, days_per_year)
    steps_per_year = jnp.asarray(
        fortran_divide(seconds_per_year, time_step),
        dtype=jnp.int32,
    )
    annual_reset = jnp.mod(model_step, steps_per_year) == jnp.int32(0)
    selected_leaf_cn = leaf_cn_ratio_pft[vegetation_type]

    return FunInitializationResult(
        state=FunInitializationState(
            leaf_cn_ratio_offset_patch=jnp.where(
                annual_reset,
                selected_leaf_cn,
                jnp.asarray(state.leaf_cn_ratio_offset_patch, dtype=jnp.float64),
            ),
            storage_carbon_demand_patch=jnp.where(
                annual_reset,
                jnp.float64(0.0),
                jnp.asarray(state.storage_carbon_demand_patch, dtype=jnp.float64),
            ),
            storage_nitrogen_demand_patch=jnp.where(
                annual_reset,
                jnp.float64(0.0),
                jnp.asarray(state.storage_nitrogen_demand_patch, dtype=jnp.float64),
            ),
            leaf_carbon_storage_transfer_accumulator_patch=jnp.where(
                annual_reset,
                jnp.float64(0.0),
                jnp.asarray(
                    state.leaf_carbon_storage_transfer_accumulator_patch,
                    dtype=jnp.float64,
                ),
            ),
            leaf_nitrogen_storage_transfer_accumulator_patch=jnp.where(
                annual_reset,
                jnp.float64(0.0),
                jnp.asarray(
                    state.leaf_nitrogen_storage_transfer_accumulator_patch,
                    dtype=jnp.float64,
                ),
            ),
        ),
        runtime=FunInitializationRuntimeState(
            time_step_seconds=time_step,
            onset_period_days=onset_period,
            offset_period_days=offset_period,
            steps_per_year=steps_per_year,
        ),
        diagnostics=FunInitializationDiagnostics(
            annual_reset=annual_reset,
            reset_patch_count=jnp.where(
                annual_reset,
                jnp.int32(patch_count),
                jnp.int32(0),
            ),
        ),
    )


__all__ = [
    "DEFAULT_FUN_INITIALIZATION_PARAMETERS",
    "FunInitializationDiagnostics",
    "FunInitializationInputs",
    "FunInitializationParameters",
    "FunInitializationResult",
    "FunInitializationRuntimeState",
    "FunInitializationState",
    "fun_initialization",
]
