
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.physics.cn_zero_manifest import (
    NUMBER_OF_DECOMPOSITION_LEVELS,
    NUMBER_OF_DECOMPOSITION_POOLS,
)
from clm_jax.physics.numerics import fortran_add


class SoilBiogeochemPrecisionControlInputs(NamedTuple):

    soil_column: jnp.ndarray


class SoilBiogeochemPrecisionControlState(NamedTuple):

    carbon_pool_column_level_pool: jnp.ndarray
    nitrogen_pool_column_level_pool: jnp.ndarray
    carbon_truncation_column_level: jnp.ndarray
    nitrogen_truncation_column_level: jnp.ndarray
    mineral_nitrogen_column_level: jnp.ndarray
    ammonium_column_level: jnp.ndarray
    nitrate_column_level: jnp.ndarray


class SoilBiogeochemPrecisionControlParameters(NamedTuple):

    carbon_critical_value: float | jnp.ndarray = 1.0e-8
    decomposition_level_count: int | jnp.ndarray = 20


DEFAULT_SOIL_BIOGEOCHEM_PRECISION_CONTROL_PARAMETERS = (
    SoilBiogeochemPrecisionControlParameters()
)


class SoilBiogeochemPrecisionControlDiagnostics(NamedTuple):
    updated_column_count: jnp.ndarray
    updated_column_level_count: jnp.ndarray
    truncated_pool_count: jnp.ndarray


class SoilBiogeochemPrecisionControlResult(NamedTuple):
    state: SoilBiogeochemPrecisionControlState
    diagnostics: SoilBiogeochemPrecisionControlDiagnostics


def _validate_shapes(
    inputs: SoilBiogeochemPrecisionControlInputs,
    state: SoilBiogeochemPrecisionControlState,
    parameters: SoilBiogeochemPrecisionControlParameters,
) -> int:
    if jnp.ndim(inputs.soil_column) != 1:
        raise ValueError("soil_column must be a vector")
    column_count = inputs.soil_column.shape[0]
    level_shape = (column_count, NUMBER_OF_DECOMPOSITION_LEVELS)
    pool_shape = (*level_shape, NUMBER_OF_DECOMPOSITION_POOLS)
    for name in (
        "carbon_pool_column_level_pool",
        "nitrogen_pool_column_level_pool",
    ):
        if jnp.shape(getattr(state, name)) != pool_shape:
            raise ValueError(f"{name} must have shape (column, 25, 7)")
    for name in (
        "carbon_truncation_column_level",
        "nitrogen_truncation_column_level",
        "mineral_nitrogen_column_level",
        "ammonium_column_level",
        "nitrate_column_level",
    ):
        if jnp.shape(getattr(state, name)) != level_shape:
            raise ValueError(f"{name} must have shape (column, 25)")
    for name in parameters._fields:
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    return column_count


def soil_biogeochem_precision_control(
    inputs: SoilBiogeochemPrecisionControlInputs,
    state: SoilBiogeochemPrecisionControlState,
    parameters: SoilBiogeochemPrecisionControlParameters = (
        DEFAULT_SOIL_BIOGEOCHEM_PRECISION_CONTROL_PARAMETERS
    ),
) -> SoilBiogeochemPrecisionControlResult:

    _validate_shapes(inputs, state, parameters)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    level_count = jnp.asarray(parameters.decomposition_level_count, dtype=jnp.int32)
    active = soil_column[:, None] & (
        jnp.arange(NUMBER_OF_DECOMPOSITION_LEVELS, dtype=jnp.int32)[None, :]
        < level_count
    )
    threshold = jnp.asarray(parameters.carbon_critical_value, dtype=jnp.float64)
    carbon = jnp.asarray(state.carbon_pool_column_level_pool, dtype=jnp.float64)
    nitrogen = jnp.asarray(
        state.nitrogen_pool_column_level_pool,
        dtype=jnp.float64,
    )
    carbon_correction = jnp.zeros(active.shape, dtype=jnp.float64)
    nitrogen_correction = jnp.zeros(active.shape, dtype=jnp.float64)
    truncated_pool_count = jnp.int32(0)
    carbon_pools = []
    nitrogen_pools = []

    for pool_index in range(NUMBER_OF_DECOMPOSITION_POOLS):
        carbon_previous = carbon[:, :, pool_index]
        nitrogen_previous = nitrogen[:, :, pool_index]
        truncate = active & (jnp.abs(carbon_previous) < threshold)
        carbon_correction = fortran_add(
            carbon_correction,
            jnp.where(truncate, carbon_previous, jnp.float64(0.0)),
        )
        nitrogen_correction = fortran_add(
            nitrogen_correction,
            jnp.where(truncate, nitrogen_previous, jnp.float64(0.0)),
        )
        carbon_pools.append(jnp.where(truncate, jnp.float64(0.0), carbon_previous))
        nitrogen_pools.append(jnp.where(truncate, jnp.float64(0.0), nitrogen_previous))
        truncated_pool_count = truncated_pool_count + jnp.sum(
            truncate,
            dtype=jnp.int32,
        )

    carbon = jnp.stack(carbon_pools, axis=-1)
    nitrogen = jnp.stack(nitrogen_pools, axis=-1)

    carbon_truncation_previous = jnp.asarray(
        state.carbon_truncation_column_level,
        dtype=jnp.float64,
    )
    nitrogen_truncation_previous = jnp.asarray(
        state.nitrogen_truncation_column_level,
        dtype=jnp.float64,
    )
    carbon_truncation = jnp.where(
        active,
        fortran_add(carbon_truncation_previous, carbon_correction),
        carbon_truncation_previous,
    )
    nitrogen_truncation = jnp.where(
        active,
        fortran_add(nitrogen_truncation_previous, nitrogen_correction),
        nitrogen_truncation_previous,
    )
    updated_column_count = jnp.sum(soil_column, dtype=jnp.int32)
    return SoilBiogeochemPrecisionControlResult(
        state=SoilBiogeochemPrecisionControlState(
            carbon_pool_column_level_pool=carbon,
            nitrogen_pool_column_level_pool=nitrogen,
            carbon_truncation_column_level=carbon_truncation,
            nitrogen_truncation_column_level=nitrogen_truncation,
            mineral_nitrogen_column_level=state.mineral_nitrogen_column_level,
            ammonium_column_level=state.ammonium_column_level,
            nitrate_column_level=state.nitrate_column_level,
        ),
        diagnostics=SoilBiogeochemPrecisionControlDiagnostics(
            updated_column_count=updated_column_count,
            updated_column_level_count=updated_column_count * level_count,
            truncated_pool_count=truncated_pool_count,
        ),
    )


__all__ = [
    "DEFAULT_SOIL_BIOGEOCHEM_PRECISION_CONTROL_PARAMETERS",
    "SoilBiogeochemPrecisionControlDiagnostics",
    "SoilBiogeochemPrecisionControlInputs",
    "SoilBiogeochemPrecisionControlParameters",
    "SoilBiogeochemPrecisionControlResult",
    "SoilBiogeochemPrecisionControlState",
    "soil_biogeochem_precision_control",
]
