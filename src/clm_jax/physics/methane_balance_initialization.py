
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.physics.numerics import fortran_hardware_add as fortran_add
from clm_jax.physics.numerics import fortran_hardware_multiply as fortran_multiply

NUMBER_OF_METHANE_SOIL_LEVELS = 20


class MethaneBalanceInitializationInputs(NamedTuple):
    nonlake_column: jnp.ndarray
    lake_column: jnp.ndarray
    inundated_fraction_column: jnp.ndarray
    soil_layer_thickness_column_level: jnp.ndarray
    saturated_methane_concentration_column_level: jnp.ndarray
    unsaturated_methane_concentration_column_level: jnp.ndarray


class MethaneBalanceInitializationState(NamedTuple):
    beginning_methane_carbon_column: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MethaneBalanceInitializationParameters:
    allow_lake_production: bool = False
    carbon_atomic_weight: float = 12.011

    def tree_flatten(self):
        return (), (self.allow_lake_production, self.carbon_atomic_weight)

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del children
        return cls(*auxiliary)


DEFAULT_METHANE_BALANCE_INITIALIZATION_PARAMETERS = MethaneBalanceInitializationParameters()


class MethaneBalanceInitializationDiagnostics(NamedTuple):
    nonlake_column_count: jnp.ndarray
    lake_column_count: jnp.ndarray
    initialized_column_count: jnp.ndarray


class MethaneBalanceInitializationResult(NamedTuple):
    state: MethaneBalanceInitializationState
    diagnostics: MethaneBalanceInitializationDiagnostics


def _validate_shapes(
    inputs: MethaneBalanceInitializationInputs,
    state: MethaneBalanceInitializationState,
) -> int:
    if jnp.ndim(inputs.nonlake_column) != 1:
        raise ValueError("nonlake_column must be a vector")
    column_count = inputs.nonlake_column.shape[0]
    for name in (
        "lake_column",
        "inundated_fraction_column",
    ):
        if jnp.shape(getattr(inputs, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    level_shape = (column_count, NUMBER_OF_METHANE_SOIL_LEVELS)
    for name in (
        "soil_layer_thickness_column_level",
        "saturated_methane_concentration_column_level",
        "unsaturated_methane_concentration_column_level",
    ):
        if jnp.shape(getattr(inputs, name)) != level_shape:
            raise ValueError(f"{name} must have shape {level_shape}")
    if jnp.shape(state.beginning_methane_carbon_column) != (column_count,):
        raise ValueError("beginning_methane_carbon_column must contain every column")
    return column_count


def initialize_methane_balance(
    inputs: MethaneBalanceInitializationInputs,
    state: MethaneBalanceInitializationState,
    parameters: MethaneBalanceInitializationParameters = (
        DEFAULT_METHANE_BALANCE_INITIALIZATION_PARAMETERS
    ),
) -> MethaneBalanceInitializationResult:

    _validate_shapes(inputs, state)
    nonlake = jnp.asarray(inputs.nonlake_column, dtype=jnp.bool_)
    lake = jnp.asarray(inputs.lake_column, dtype=jnp.bool_)
    initialized = nonlake | lake
    inundated = jnp.asarray(inputs.inundated_fraction_column, dtype=jnp.float64)
    thickness = jnp.asarray(
        inputs.soil_layer_thickness_column_level,
        dtype=jnp.float64,
    )
    saturated = jnp.asarray(
        inputs.saturated_methane_concentration_column_level,
        dtype=jnp.float64,
    )
    unsaturated = jnp.asarray(
        inputs.unsaturated_methane_concentration_column_level,
        dtype=jnp.float64,
    )
    carbon_atomic_weight = jnp.float64(parameters.carbon_atomic_weight)
    beginning = jnp.where(
        initialized,
        jnp.float64(0.0),
        jnp.asarray(state.beginning_methane_carbon_column, dtype=jnp.float64),
    )

    def add_level(level: int, running: jnp.ndarray) -> jnp.ndarray:
        dry_fraction = fortran_add(jnp.float64(1.0), -inundated)
        mixed_concentration = fortran_add(
            fortran_multiply(inundated, saturated[:, level]),
            fortran_multiply(dry_fraction, unsaturated[:, level]),
        )
        nonlake_contribution = fortran_multiply(
            fortran_multiply(mixed_concentration, thickness[:, level]),
            carbon_atomic_weight,
        )
        updated = jnp.where(
            nonlake,
            fortran_add(running, nonlake_contribution),
            running,
        )
        if parameters.allow_lake_production:
            lake_contribution = fortran_multiply(
                fortran_multiply(saturated[:, level], thickness[:, level]),
                carbon_atomic_weight,
            )
            updated = jnp.where(
                lake,
                fortran_add(updated, lake_contribution),
                updated,
            )
        return updated

    beginning = lax.fori_loop(
        0,
        NUMBER_OF_METHANE_SOIL_LEVELS,
        add_level,
        beginning,
    )
    return MethaneBalanceInitializationResult(
        state=MethaneBalanceInitializationState(beginning),
        diagnostics=MethaneBalanceInitializationDiagnostics(
            nonlake_column_count=jnp.sum(nonlake, dtype=jnp.int32),
            lake_column_count=jnp.sum(lake, dtype=jnp.int32),
            initialized_column_count=jnp.sum(initialized, dtype=jnp.int32),
        ),
    )


__all__ = [
    "DEFAULT_METHANE_BALANCE_INITIALIZATION_PARAMETERS",
    "NUMBER_OF_METHANE_SOIL_LEVELS",
    "MethaneBalanceInitializationDiagnostics",
    "MethaneBalanceInitializationInputs",
    "MethaneBalanceInitializationParameters",
    "MethaneBalanceInitializationResult",
    "MethaneBalanceInitializationState",
    "initialize_methane_balance",
]
