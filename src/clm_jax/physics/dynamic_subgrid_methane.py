
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.dynamic_subgrid_weights import (
    ColumnWeightTransition,
    update_column_state_no_special_handling,
)
from clm_jax.physics.methane_balance_initialization import (
    NUMBER_OF_METHANE_SOIL_LEVELS,
)
from clm_jax.physics.numerics import fortran_add, fortran_multiply


class DynamicMethaneColumnState(NamedTuple):
    inundated_fraction_column: jnp.ndarray
    lagged_inundated_fraction_column: jnp.ndarray
    saturated_methane_concentration_column_level: jnp.ndarray
    unsaturated_methane_concentration_column_level: jnp.ndarray
    lagged_layer_saturation_column_level: jnp.ndarray
    saturated_oxygen_concentration_column_level: jnp.ndarray
    unsaturated_oxygen_concentration_column_level: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DynamicMethaneColumnParameters:
    carbon_atomic_weight: float = 12.011

    def tree_flatten(self):
        return (), (self.carbon_atomic_weight,)

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del children
        return cls(*auxiliary)


DEFAULT_DYNAMIC_METHANE_COLUMN_PARAMETERS = DynamicMethaneColumnParameters()


class DynamicMethaneColumnAdjustmentResult(NamedTuple):
    state: DynamicMethaneColumnState
    methane_balance_adjustment_column: jnp.ndarray


def _validate_dynamic_methane_shapes(
    topology: SubgridTopology,
    transition: ColumnWeightTransition,
    state: DynamicMethaneColumnState,
    soil_layer_thickness_column_level: jnp.ndarray,
) -> tuple[int, int]:
    number_of_columns = topology.column.gridcell.shape[0]
    if transition.old_column_weight_on_gridcell.shape != (number_of_columns,):
        raise ValueError("column transition must match topology")
    vector_shape = (number_of_columns,)
    for name in (
        "inundated_fraction_column",
        "lagged_inundated_fraction_column",
    ):
        if getattr(state, name).shape != vector_shape:
            raise ValueError(f"{name} must have shape {vector_shape}")

    level_fields = (
        "saturated_methane_concentration_column_level",
        "unsaturated_methane_concentration_column_level",
        "lagged_layer_saturation_column_level",
        "saturated_oxygen_concentration_column_level",
        "unsaturated_oxygen_concentration_column_level",
    )
    level_shape = getattr(state, level_fields[0]).shape
    if len(level_shape) != 2 or level_shape[0] != number_of_columns:
        raise ValueError("methane column-level fields must be column-by-level matrices")
    if level_shape[1] < NUMBER_OF_METHANE_SOIL_LEVELS:
        raise ValueError(
            "methane column state must contain all configured soil levels"
        )
    for name in level_fields[1:]:
        if getattr(state, name).shape != level_shape:
            raise ValueError("all methane column-level fields must have matching shapes")
    if soil_layer_thickness_column_level.shape != level_shape:
        raise ValueError("soil layer thickness must match methane column-level fields")
    return number_of_columns, topology.grid.longitude_degrees.shape[0]


def dynamic_methane_column_adjustments(
    transition: ColumnWeightTransition,
    topology: SubgridTopology,
    state: DynamicMethaneColumnState,
    soil_layer_thickness_column_level: jnp.ndarray,
    parameters: DynamicMethaneColumnParameters = (
        DEFAULT_DYNAMIC_METHANE_COLUMN_PARAMETERS
    ),
) -> DynamicMethaneColumnAdjustmentResult:

    _, number_of_gridcells = _validate_dynamic_methane_shapes(
        topology,
        transition,
        state,
        soil_layer_thickness_column_level,
    )
    column_gridcell = jnp.asarray(topology.column.gridcell, dtype=jnp.int32)
    old_inundated = jnp.asarray(
        state.inundated_fraction_column,
        dtype=jnp.float64,
    )
    thickness = jnp.asarray(
        soil_layer_thickness_column_level,
        dtype=jnp.float64,
    )

    def update(
        values: jnp.ndarray,
        old_fraction: jnp.ndarray | None = None,
        new_fraction: jnp.ndarray | None = None,
    ):
        return update_column_state_no_special_handling(
            transition,
            column_gridcell,
            values,
            number_of_gridcells=number_of_gridcells,
            fractional_area_old=old_fraction,
            fractional_area_new=new_fraction,
        )

    new_inundated = update(old_inundated).state
    old_uninundated = fortran_add(jnp.float64(1.0), -old_inundated)
    new_uninundated = update(old_uninundated).state
    lagged_inundated = update(
        jnp.asarray(
            state.lagged_inundated_fraction_column,
            dtype=jnp.float64,
        )
    ).state

    saturated_methane = jnp.asarray(
        state.saturated_methane_concentration_column_level,
        dtype=jnp.float64,
    )
    unsaturated_methane = jnp.asarray(
        state.unsaturated_methane_concentration_column_level,
        dtype=jnp.float64,
    )
    lagged_layer_saturation = jnp.asarray(
        state.lagged_layer_saturation_column_level,
        dtype=jnp.float64,
    )
    saturated_oxygen = jnp.asarray(
        state.saturated_oxygen_concentration_column_level,
        dtype=jnp.float64,
    )
    unsaturated_oxygen = jnp.asarray(
        state.unsaturated_oxygen_concentration_column_level,
        dtype=jnp.float64,
    )
    methane_balance = jnp.zeros_like(old_inundated, dtype=jnp.float64)
    carbon_atomic_weight = jnp.float64(parameters.carbon_atomic_weight)

    def update_level(level: int, carry):
        (
            saturated_ch4,
            unsaturated_ch4,
            lagged_saturation,
            saturated_o2,
            unsaturated_o2,
            balance,
        ) = carry

        saturated_result = update(
            saturated_ch4[:, level],
            old_inundated,
            new_inundated,
        )
        saturated_contribution = fortran_multiply(
            fortran_multiply(
                saturated_result.adjustment_column,
                thickness[:, level],
            ),
            carbon_atomic_weight,
        )
        balance = fortran_add(balance, saturated_contribution)
        saturated_ch4 = saturated_ch4.at[:, level].set(saturated_result.state)

        unsaturated_result = update(
            unsaturated_ch4[:, level],
            old_uninundated,
            new_uninundated,
        )
        unsaturated_contribution = fortran_multiply(
            fortran_multiply(
                unsaturated_result.adjustment_column,
                thickness[:, level],
            ),
            carbon_atomic_weight,
        )
        balance = fortran_add(balance, unsaturated_contribution)
        unsaturated_ch4 = unsaturated_ch4.at[:, level].set(
            unsaturated_result.state
        )

        lagged_result = update(
            lagged_saturation[:, level],
            old_uninundated,
            new_uninundated,
        )
        lagged_saturation = lagged_saturation.at[:, level].set(
            lagged_result.state
        )
        saturated_o2_result = update(
            saturated_o2[:, level],
            old_inundated,
            new_inundated,
        )
        saturated_o2 = saturated_o2.at[:, level].set(
            saturated_o2_result.state
        )
        unsaturated_o2_result = update(
            unsaturated_o2[:, level],
            old_uninundated,
            new_uninundated,
        )
        unsaturated_o2 = unsaturated_o2.at[:, level].set(
            unsaturated_o2_result.state
        )
        return (
            saturated_ch4,
            unsaturated_ch4,
            lagged_saturation,
            saturated_o2,
            unsaturated_o2,
            balance,
        )

    (
        saturated_methane,
        unsaturated_methane,
        lagged_layer_saturation,
        saturated_oxygen,
        unsaturated_oxygen,
        methane_balance,
    ) = lax.fori_loop(
        0,
        NUMBER_OF_METHANE_SOIL_LEVELS,
        update_level,
        (
            saturated_methane,
            unsaturated_methane,
            lagged_layer_saturation,
            saturated_oxygen,
            unsaturated_oxygen,
            methane_balance,
        ),
    )

    return DynamicMethaneColumnAdjustmentResult(
        state=DynamicMethaneColumnState(
            inundated_fraction_column=new_inundated,
            lagged_inundated_fraction_column=lagged_inundated,
            saturated_methane_concentration_column_level=saturated_methane,
            unsaturated_methane_concentration_column_level=unsaturated_methane,
            lagged_layer_saturation_column_level=lagged_layer_saturation,
            saturated_oxygen_concentration_column_level=saturated_oxygen,
            unsaturated_oxygen_concentration_column_level=unsaturated_oxygen,
        ),
        methane_balance_adjustment_column=methane_balance,
    )


__all__ = [
    "DEFAULT_DYNAMIC_METHANE_COLUMN_PARAMETERS",
    "DynamicMethaneColumnAdjustmentResult",
    "DynamicMethaneColumnParameters",
    "DynamicMethaneColumnState",
    "dynamic_methane_column_adjustments",
]
