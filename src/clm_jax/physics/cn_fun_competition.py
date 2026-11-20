
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.physics.cn_fun_allocation import (
    FunAllocationInputs,
    FunAllocationParameters,
    FunAllocationResult,
    FunAllocationState,
    carbon_use_efficiency_nitrogen_uptake,
)
from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    get_cn_flux_field,
    set_cn_flux_field,
)
from clm_jax.physics.cn_zero_manifest import NUMBER_OF_DECOMPOSITION_LEVELS
from clm_jax.physics.soil_biogeochem_competition import (
    SoilBiogeochemCompetitionDiagnostics,
    SoilBiogeochemCompetitionInputs,
    SoilBiogeochemCompetitionParameters,
    SoilBiogeochemCompetitionResult,
    SoilBiogeochemCompetitionState,
    finalize_soil_biogeochem_competition_fun,
    soil_biogeochem_competition_pre_fun,
)
from clm_jax.physics.subgrid import (
    patch_to_column_average,
    patch_to_column_mean,
)

NUMBER_OF_FUN_LEVELS = 20

_MAXIMUM_NITRATE = "soil_nitrogen.smin_no3_to_plant_vr_col"
_MAXIMUM_AMMONIUM = "soil_nitrogen.smin_nh4_to_plant_vr_col"
_FUN_NITRATE_COLUMN_LEVEL = "soil_nitrogen.sminn_to_plant_fun_no3_vr_col"
_FUN_AMMONIUM_COLUMN_LEVEL = "soil_nitrogen.sminn_to_plant_fun_nh4_vr_col"
_SOIL_CARBON_CHANGE_COLUMN = "soil_carbon.soilc_change_col"
_NITROGEN_FIXATION_COLUMN = "soil_nitrogen.nfix_to_sminn_col"

_CARBON_PATCH_FIELDS = {
    "vegetation_carbon.npp_Nactive_no3_patch": "active_nitrate_patch",
    "vegetation_carbon.npp_Nactive_nh4_patch": "active_ammonium_patch",
    "vegetation_carbon.npp_Nactive_patch": "active_total_patch",
    "vegetation_carbon.npp_Nnonmyc_no3_patch": ("nonmycorrhizal_nitrate_patch"),
    "vegetation_carbon.npp_Nnonmyc_nh4_patch": ("nonmycorrhizal_ammonium_patch"),
    "vegetation_carbon.npp_Nnonmyc_patch": "nonmycorrhizal_total_patch",
    "vegetation_carbon.npp_Nfix_patch": "fixation_patch",
    "vegetation_carbon.npp_Nretrans_patch": "retranslocation_patch",
    "vegetation_carbon.npp_Nuptake_patch": "nitrogen_uptake_patch",
    "vegetation_carbon.npp_growth_patch": "growth_patch",
    "vegetation_carbon.npp_burnedoff_patch": "burned_off_patch",
    "vegetation_carbon.soilc_change_patch": "soil_carbon_change_patch",
}
_NITROGEN_PATCH_FIELDS = {
    "vegetation_nitrogen.retransn_to_npool_patch": ("paid_retranslocation_to_pool_patch"),
    "vegetation_nitrogen.free_retransn_to_npool_patch": ("free_retranslocation_to_pool_patch"),
}


class FunCompetitionCouplingInputs(NamedTuple):

    soil_patch: jnp.ndarray
    soil_column: jnp.ndarray
    patch_column_index: jnp.ndarray
    patch_weight_on_column: jnp.ndarray
    patch_active: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FunCompetitionCouplingParameters:

    maximum_patches_per_column: int = 79
    special_value: float = CLM_SPECIAL_VALUE

    def tree_flatten(self):

        return (), (self.maximum_patches_per_column, self.special_value)

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del children
        return cls(*auxiliary)


DEFAULT_FUN_COMPETITION_COUPLING_PARAMETERS = FunCompetitionCouplingParameters()


class FunCompetitionFluxIntegrationResult(NamedTuple):

    cn_fluxes: CnTransientFluxState
    nitrate_uptake_column_level: jnp.ndarray
    ammonium_uptake_column_level: jnp.ndarray


class FunSoilCompetitionResult(NamedTuple):

    pre_fun_diagnostics: SoilBiogeochemCompetitionDiagnostics
    allocation: FunAllocationResult
    competition: SoilBiogeochemCompetitionResult


def _validate_coupling_shapes(
    inputs: FunCompetitionCouplingInputs,
    allocation: FunAllocationResult,
    cn_fluxes: CnTransientFluxState,
    parameters: FunCompetitionCouplingParameters,
) -> tuple[int, int]:
    patch_count = inputs.soil_patch.shape[0]
    column_count = inputs.soil_column.shape[0]
    patch_shape = (patch_count,)
    if not (
        jnp.shape(inputs.soil_patch)
        == jnp.shape(inputs.patch_column_index)
        == jnp.shape(inputs.patch_weight_on_column)
        == jnp.shape(inputs.patch_active)
        == patch_shape
    ):
        raise ValueError("FUN coupling patch topology must be identically sized vectors")
    if jnp.ndim(inputs.soil_column) != 1:
        raise ValueError("FUN coupling soil_column must be a vector")
    if parameters.maximum_patches_per_column < 0:
        raise ValueError("maximum_patches_per_column must be nonnegative")
    if cn_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("packed CN flux state has an incompatible patch count")
    if cn_fluxes.column_scalar.shape[0] != column_count:
        raise ValueError("packed CN flux state has an incompatible column count")
    for name in allocation.carbon_fluxes._fields:
        if jnp.shape(getattr(allocation.carbon_fluxes, name)) != patch_shape:
            raise ValueError(f"FUN carbon flux {name} must contain every patch")
    for name in allocation.nitrogen_fluxes._fields:
        expected = (
            (patch_count, NUMBER_OF_DECOMPOSITION_LEVELS)
            if name.endswith("_patch_level")
            else patch_shape
        )
        if jnp.shape(getattr(allocation.nitrogen_fluxes, name)) != expected:
            raise ValueError(f"FUN nitrogen flux {name} must have shape {expected!r}")
    return patch_count, column_count


def integrate_fun_allocation_fluxes(
    inputs: FunCompetitionCouplingInputs,
    allocation: FunAllocationResult,
    cn_fluxes: CnTransientFluxState,
    parameters: FunCompetitionCouplingParameters = (DEFAULT_FUN_COMPETITION_COUPLING_PARAMETERS),
) -> FunCompetitionFluxIntegrationResult:

    _, column_count = _validate_coupling_shapes(inputs, allocation, cn_fluxes, parameters)
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    patch_column = jnp.asarray(inputs.patch_column_index, dtype=jnp.int32)
    patch_weight = jnp.asarray(inputs.patch_weight_on_column, dtype=jnp.float64)
    patch_active = jnp.asarray(inputs.patch_active, dtype=jnp.bool_)

    updated = cn_fluxes
    for field, attribute in _CARBON_PATCH_FIELDS.items():
        old = get_cn_flux_field(updated, field)
        value = jnp.asarray(getattr(allocation.carbon_fluxes, attribute), dtype=jnp.float64)
        updated = set_cn_flux_field(
            updated,
            field,
            jnp.where(soil_patch, value, old),
        )
    for field, attribute in _NITROGEN_PATCH_FIELDS.items():
        old = get_cn_flux_field(updated, field)
        value = jnp.asarray(getattr(allocation.nitrogen_fluxes, attribute), dtype=jnp.float64)
        updated = set_cn_flux_field(
            updated,
            field,
            jnp.where(soil_patch, value, old),
        )

    def filtered_column_sum(patch_values: jnp.ndarray) -> jnp.ndarray:
        return patch_to_column_average(
            patch_values,
            patch_column,
            patch_weight,
            patch_active,
            soil_column,
            maximum_patches_per_column=parameters.maximum_patches_per_column,
        )

    for field, patch_values in (
        (
            _SOIL_CARBON_CHANGE_COLUMN,
            allocation.carbon_fluxes.soil_carbon_change_patch,
        ),
        (
            _NITROGEN_FIXATION_COLUMN,
            allocation.nitrogen_fluxes.fixation_patch,
        ),
    ):
        old = get_cn_flux_field(updated, field)
        reduced = filtered_column_sum(patch_values)
        updated = set_cn_flux_field(
            updated,
            field,
            jnp.where(soil_column, reduced, old),
        )

    def unfiltered_column_mean(patch_values: jnp.ndarray) -> jnp.ndarray:
        return patch_to_column_mean(
            patch_values,
            patch_column,
            patch_weight,
            patch_active,
            number_of_columns=column_count,
            maximum_patches_per_column=parameters.maximum_patches_per_column,
            special_value=parameters.special_value,
        )

    nitrate_active = jax.vmap(
        unfiltered_column_mean,
        in_axes=1,
        out_axes=1,
    )(allocation.nitrogen_fluxes.soil_nitrate_to_plant_fun_patch_level[:, :NUMBER_OF_FUN_LEVELS])
    ammonium_active = jax.vmap(
        unfiltered_column_mean,
        in_axes=1,
        out_axes=1,
    )(allocation.nitrogen_fluxes.soil_ammonium_to_plant_fun_patch_level[:, :NUMBER_OF_FUN_LEVELS])
    nitrate_old = get_cn_flux_field(updated, _FUN_NITRATE_COLUMN_LEVEL)
    ammonium_old = get_cn_flux_field(updated, _FUN_AMMONIUM_COLUMN_LEVEL)
    nitrate_column = nitrate_old.at[:, :NUMBER_OF_FUN_LEVELS].set(nitrate_active)
    ammonium_column = ammonium_old.at[:, :NUMBER_OF_FUN_LEVELS].set(ammonium_active)
    updated = set_cn_flux_field(
        updated,
        _FUN_NITRATE_COLUMN_LEVEL,
        nitrate_column,
    )
    updated = set_cn_flux_field(
        updated,
        _FUN_AMMONIUM_COLUMN_LEVEL,
        ammonium_column,
    )
    return FunCompetitionFluxIntegrationResult(
        cn_fluxes=updated,
        nitrate_uptake_column_level=nitrate_column,
        ammonium_uptake_column_level=ammonium_column,
    )


def fun_soil_biogeochem_competition(
    coupling_inputs: FunCompetitionCouplingInputs,
    allocation_inputs: FunAllocationInputs,
    allocation_state: FunAllocationState,
    allocation_parameters: FunAllocationParameters,
    competition_inputs: SoilBiogeochemCompetitionInputs,
    competition_state: SoilBiogeochemCompetitionState,
    cn_fluxes: CnTransientFluxState,
    competition_parameters: SoilBiogeochemCompetitionParameters,
    coupling_parameters: FunCompetitionCouplingParameters = (
        DEFAULT_FUN_COMPETITION_COUPLING_PARAMETERS
    ),
) -> FunSoilCompetitionResult:

    pre_fun = soil_biogeochem_competition_pre_fun(
        competition_inputs,
        competition_state,
        cn_fluxes,
        competition_parameters,
    )
    allocation = carbon_use_efficiency_nitrogen_uptake(
        allocation_inputs._replace(
            maximum_nitrate_uptake_column_level=get_cn_flux_field(
                pre_fun.cn_fluxes, _MAXIMUM_NITRATE
            ),
            maximum_ammonium_uptake_column_level=get_cn_flux_field(
                pre_fun.cn_fluxes, _MAXIMUM_AMMONIUM
            ),
        ),
        allocation_state,
        allocation_parameters,
    )
    integrated = integrate_fun_allocation_fluxes(
        coupling_inputs,
        allocation,
        pre_fun.cn_fluxes,
        coupling_parameters,
    )
    competition = finalize_soil_biogeochem_competition_fun(
        competition_inputs,
        pre_fun.state,
        integrated.cn_fluxes,
        integrated.nitrate_uptake_column_level,
        integrated.ammonium_uptake_column_level,
        competition_parameters,
    )
    return FunSoilCompetitionResult(
        pre_fun_diagnostics=pre_fun.diagnostics,
        allocation=allocation,
        competition=competition,
    )


__all__ = [
    "DEFAULT_FUN_COMPETITION_COUPLING_PARAMETERS",
    "FunCompetitionCouplingInputs",
    "FunCompetitionCouplingParameters",
    "FunCompetitionFluxIntegrationResult",
    "FunSoilCompetitionResult",
    "fun_soil_biogeochem_competition",
    "integrate_fun_allocation_fluxes",
]
