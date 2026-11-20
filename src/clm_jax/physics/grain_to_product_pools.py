
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    get_cn_flux_field,
    set_cn_flux_field,
)
from clm_jax.physics.subgrid import patch_to_column_average


class GrainToProductPoolsInputs(NamedTuple):

    soil_patch: jnp.ndarray
    soil_column: jnp.ndarray
    patch_column_index: jnp.ndarray
    patch_weight_on_column: jnp.ndarray
    patch_active: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GrainToProductPoolsParameters:

    use_crop: bool = True
    use_grain_product: bool = True
    maximum_patches_per_column: int = 79

    def tree_flatten(self):

        return (), (
            self.use_crop,
            self.use_grain_product,
            self.maximum_patches_per_column,
        )

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del children
        return cls(*auxiliary)


DEFAULT_GRAIN_TO_PRODUCT_POOLS_PARAMETERS = GrainToProductPoolsParameters()


class GrainToProductPoolsDiagnostics(NamedTuple):
    updated_patch_count: jnp.ndarray
    updated_column_count: jnp.ndarray


class GrainToProductPoolsResult(NamedTuple):
    cn_fluxes: CnTransientFluxState
    diagnostics: GrainToProductPoolsDiagnostics


_CARBON_FOOD = "vegetation_carbon.grainc_to_food_patch"
_CARBON_PRODUCT_PATCH = "vegetation_carbon.grainc_to_cropprodc_patch"
_CARBON_PRODUCT_COLUMN = "vegetation_carbon.grainc_to_cropprodc_col"
_NITROGEN_FOOD = "vegetation_nitrogen.grainn_to_food_patch"
_NITROGEN_PRODUCT_PATCH = "vegetation_nitrogen.grainn_to_cropprodn_patch"
_NITROGEN_PRODUCT_COLUMN = "vegetation_nitrogen.grainn_to_cropprodn_col"


def _validate_shapes(
    inputs: GrainToProductPoolsInputs,
    cn_fluxes: CnTransientFluxState,
    parameters: GrainToProductPoolsParameters,
) -> tuple[int, int]:
    if jnp.ndim(inputs.soil_patch) != 1 or jnp.ndim(inputs.soil_column) != 1:
        raise ValueError("soil_patch and soil_column must be vectors")
    patch_count = inputs.soil_patch.shape[0]
    column_count = inputs.soil_column.shape[0]
    for name in (
        "patch_column_index",
        "patch_weight_on_column",
        "patch_active",
    ):
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    for field_name in (
        _CARBON_FOOD,
        _CARBON_PRODUCT_PATCH,
        _NITROGEN_FOOD,
        _NITROGEN_PRODUCT_PATCH,
    ):
        if get_cn_flux_field(cn_fluxes, field_name).shape != (patch_count,):
            raise ValueError("grain product patch fluxes must match patch count")
    for field_name in (_CARBON_PRODUCT_COLUMN, _NITROGEN_PRODUCT_COLUMN):
        if get_cn_flux_field(cn_fluxes, field_name).shape != (column_count,):
            raise ValueError("grain product column fluxes must match column count")
    for name in (
        "use_crop",
        "use_grain_product",
        "maximum_patches_per_column",
    ):
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    return patch_count, column_count


def grain_to_product_pools(
    inputs: GrainToProductPoolsInputs,
    cn_fluxes: CnTransientFluxState,
    parameters: GrainToProductPoolsParameters = (DEFAULT_GRAIN_TO_PRODUCT_POOLS_PARAMETERS),
) -> GrainToProductPoolsResult:

    _validate_shapes(inputs, cn_fluxes, parameters)
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    patch_column = jnp.asarray(inputs.patch_column_index, dtype=jnp.int32)
    patch_weight = jnp.asarray(
        inputs.patch_weight_on_column,
        dtype=jnp.float64,
    )
    patch_active = jnp.asarray(inputs.patch_active, dtype=jnp.bool_)
    enabled = jnp.asarray(parameters.use_crop, dtype=jnp.bool_) & jnp.asarray(
        parameters.use_grain_product,
        dtype=jnp.bool_,
    )
    update_patch = enabled & soil_patch
    update_column = enabled & soil_column

    carbon_food = get_cn_flux_field(cn_fluxes, _CARBON_FOOD)
    nitrogen_food = get_cn_flux_field(cn_fluxes, _NITROGEN_FOOD)
    previous_carbon_patch = get_cn_flux_field(
        cn_fluxes,
        _CARBON_PRODUCT_PATCH,
    )
    previous_nitrogen_patch = get_cn_flux_field(
        cn_fluxes,
        _NITROGEN_PRODUCT_PATCH,
    )
    carbon_patch_candidate = jnp.where(
        soil_patch,
        carbon_food,
        previous_carbon_patch,
    )
    nitrogen_patch_candidate = jnp.where(
        soil_patch,
        nitrogen_food,
        previous_nitrogen_patch,
    )

    carbon_column_candidate = patch_to_column_average(
        carbon_patch_candidate,
        patch_column,
        patch_weight,
        patch_active,
        soil_column,
        maximum_patches_per_column=parameters.maximum_patches_per_column,
    )
    nitrogen_column_candidate = patch_to_column_average(
        nitrogen_patch_candidate,
        patch_column,
        patch_weight,
        patch_active,
        soil_column,
        maximum_patches_per_column=parameters.maximum_patches_per_column,
    )

    cn_fluxes = set_cn_flux_field(
        cn_fluxes,
        _CARBON_PRODUCT_PATCH,
        jnp.where(update_patch, carbon_food, previous_carbon_patch),
    )
    cn_fluxes = set_cn_flux_field(
        cn_fluxes,
        _NITROGEN_PRODUCT_PATCH,
        jnp.where(update_patch, nitrogen_food, previous_nitrogen_patch),
    )
    previous_carbon_column = get_cn_flux_field(
        cn_fluxes,
        _CARBON_PRODUCT_COLUMN,
    )
    previous_nitrogen_column = get_cn_flux_field(
        cn_fluxes,
        _NITROGEN_PRODUCT_COLUMN,
    )
    cn_fluxes = set_cn_flux_field(
        cn_fluxes,
        _CARBON_PRODUCT_COLUMN,
        jnp.where(
            update_column,
            carbon_column_candidate,
            previous_carbon_column,
        ),
    )
    cn_fluxes = set_cn_flux_field(
        cn_fluxes,
        _NITROGEN_PRODUCT_COLUMN,
        jnp.where(
            update_column,
            nitrogen_column_candidate,
            previous_nitrogen_column,
        ),
    )

    return GrainToProductPoolsResult(
        cn_fluxes=cn_fluxes,
        diagnostics=GrainToProductPoolsDiagnostics(
            updated_patch_count=jnp.sum(update_patch, dtype=jnp.int32),
            updated_column_count=jnp.sum(update_column, dtype=jnp.int32),
        ),
    )


__all__ = [
    "DEFAULT_GRAIN_TO_PRODUCT_POOLS_PARAMETERS",
    "GrainToProductPoolsDiagnostics",
    "GrainToProductPoolsInputs",
    "GrainToProductPoolsParameters",
    "GrainToProductPoolsResult",
    "grain_to_product_pools",
]
