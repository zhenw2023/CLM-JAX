
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from clm_jax.physics.subgrid import patch_to_column_average


class PlantNutrientDemandAggregationInputs(NamedTuple):

    patch_column_index: jnp.ndarray
    patch_weight_on_column: jnp.ndarray
    patch_active: jnp.ndarray
    soil_column: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PlantNutrientDemandAggregationParameters:

    maximum_patches_per_column: int = 79

    def tree_flatten(self):

        return (), self.maximum_patches_per_column

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del children
        return cls(auxiliary)


DEFAULT_PLANT_NUTRIENT_DEMAND_AGGREGATION_PARAMETERS = PlantNutrientDemandAggregationParameters()


class PlantNutrientDemandAggregationDiagnostics(NamedTuple):
    updated_column_count: jnp.ndarray


class PlantNutrientDemandAggregationResult(NamedTuple):
    plant_nitrogen_demand_column: jnp.ndarray
    diagnostics: PlantNutrientDemandAggregationDiagnostics


def aggregate_plant_nutrient_demand(
    plant_nitrogen_demand_patch: jnp.ndarray,
    inputs: PlantNutrientDemandAggregationInputs,
    parameters: PlantNutrientDemandAggregationParameters = (
        DEFAULT_PLANT_NUTRIENT_DEMAND_AGGREGATION_PARAMETERS
    ),
) -> PlantNutrientDemandAggregationResult:

    values = jnp.asarray(plant_nitrogen_demand_patch, dtype=jnp.float64)
    column = jnp.asarray(inputs.patch_column_index, dtype=jnp.int32)
    weight = jnp.asarray(inputs.patch_weight_on_column, dtype=jnp.float64)
    active = jnp.asarray(inputs.patch_active, dtype=jnp.bool_)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    if not (values.ndim == 1 and values.shape == column.shape == weight.shape == active.shape):
        raise ValueError("plant demand and patch topology must be identically sized vectors")
    if soil_column.ndim != 1:
        raise ValueError("soil_column must be a vector")
    column_demand = patch_to_column_average(
        values,
        column,
        weight,
        active,
        soil_column,
        maximum_patches_per_column=parameters.maximum_patches_per_column,
    )
    return PlantNutrientDemandAggregationResult(
        plant_nitrogen_demand_column=column_demand,
        diagnostics=PlantNutrientDemandAggregationDiagnostics(
            updated_column_count=jnp.sum(soil_column, dtype=jnp.int32),
        ),
    )
