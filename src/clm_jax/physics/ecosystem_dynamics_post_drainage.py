
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from clm_jax.physics.carbon_nitrogen_balance_initialization import (
    CARBON_DISPLAYED_COMPONENTS,
    CARBON_STORED_COMPONENTS,
    NITROGEN_DISPLAYED_COMPONENTS,
    NITROGEN_STORED_COMPONENTS,
)
from clm_jax.physics.cn_driver_summarize_fluxes import (
    CnDriverSummarizeFluxesDiagnostics,
    CnDriverSummarizeFluxesGridcellFluxes,
    CnDriverSummarizeFluxesInputs,
    CnDriverSummarizeFluxesParameters,
    CnDriverSummarizeFluxesState,
    cn_driver_summarize_fluxes,
)
from clm_jax.physics.cn_driver_summarize_states import (
    CnDriverSummarizeStatesDiagnostics,
    CnDriverSummarizeStatesInputs,
    CnDriverSummarizeStatesParameters,
    CnDriverSummarizeStatesState,
    cn_driver_summarize_states,
)
from clm_jax.physics.cn_n_state_update3 import (
    CnNStateUpdate3Diagnostics,
    CnNStateUpdate3Parameters,
    CnNStateUpdate3State,
    cn_n_state_update3,
)
from clm_jax.physics.cn_precision_control import (
    CnPrecisionControlState,
    get_cn_precision_control_state_field,
)
from clm_jax.physics.cn_state_update2 import CnStateUpdate2Inputs
from clm_jax.physics.cn_veg_struct_update import (
    CnVegStructUpdateDiagnostics,
    CnVegStructUpdateInputs,
    CnVegStructUpdateParameters,
    CnVegStructUpdateResult,
    CnVegStructUpdateState,
    cn_veg_struct_update,
)
from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    get_cn_flux_field,
    set_cn_flux_field,
)
from clm_jax.physics.post_drainage_precision_control import (
    PostDrainagePrecisionControlDiagnostics,
    PostDrainagePrecisionControlInputs,
    PostDrainagePrecisionControlParameters,
    PostDrainagePrecisionControlState,
    post_drainage_precision_control,
)
from clm_jax.physics.soil_biogeochem_n_leaching import (
    SoilBiogeochemNLeachingDiagnostics,
    SoilBiogeochemNLeachingFluxState,
    SoilBiogeochemNLeachingInputs,
    SoilBiogeochemNLeachingParameters,
    soil_biogeochem_n_leaching,
)

_LEACHING_FLUX = "soil_nitrogen.smin_no3_leached_vr_col"
_RUNOFF_FLUX = "soil_nitrogen.smin_no3_runoff_vr_col"
_ACTIVE_DECOMPOSITION_LEVELS = 20


class EcosystemDynamicsPostDrainageInputs(NamedTuple):

    leaching: SoilBiogeochemNLeachingInputs
    nitrogen_state_update: CnStateUpdate2Inputs
    precision: PostDrainagePrecisionControlInputs
    state_summary: CnDriverSummarizeStatesInputs
    excess_maintenance_respiration_carbon_loss_patch: jnp.ndarray
    flux_summary: CnDriverSummarizeFluxesInputs
    vegetation_structure: CnVegStructUpdateInputs
    do_albedo: bool | jnp.ndarray


class EcosystemDynamicsPostDrainageFluxSummaryState(NamedTuple):

    temporary_npp_sum_patch: jnp.ndarray
    lagged_npp_column: jnp.ndarray
    dynamic_landuse_amount_to_dribble_gridcell: jnp.ndarray
    harvest_xsmrpool_amount_to_dribble_gridcell: jnp.ndarray


class EcosystemDynamicsPostDrainageState(NamedTuple):

    biogeochemistry: PostDrainagePrecisionControlState
    state_summaries: CnDriverSummarizeStatesState
    flux_summary: EcosystemDynamicsPostDrainageFluxSummaryState
    vegetation_structure: CnVegStructUpdateState


class EcosystemDynamicsPostDrainageParameters(NamedTuple):
    leaching: SoilBiogeochemNLeachingParameters
    nitrogen_state_update: CnNStateUpdate3Parameters
    precision: PostDrainagePrecisionControlParameters
    state_summary: CnDriverSummarizeStatesParameters
    flux_summary: CnDriverSummarizeFluxesParameters
    vegetation_structure: CnVegStructUpdateParameters


class EcosystemDynamicsPostDrainageDiagnostics(NamedTuple):
    leaching: SoilBiogeochemNLeachingDiagnostics
    nitrogen_state_update: CnNStateUpdate3Diagnostics
    precision: PostDrainagePrecisionControlDiagnostics
    state_summary: CnDriverSummarizeStatesDiagnostics
    flux_summary: CnDriverSummarizeFluxesDiagnostics
    vegetation_structure: CnVegStructUpdateDiagnostics


class EcosystemDynamicsPostDrainageResult(NamedTuple):
    state: EcosystemDynamicsPostDrainageState
    gridcell_fluxes: CnDriverSummarizeFluxesGridcellFluxes
    diagnostics: EcosystemDynamicsPostDrainageDiagnostics


def _qualified_components(
    state: CnPrecisionControlState,
    species: str,
    names: tuple[str, ...],
) -> jnp.ndarray:
    return jnp.stack(
        tuple(
            get_cn_precision_control_state_field(
                state,
                f"vegetation_{species}.{name}_patch",
            )
            for name in names
        ),
        axis=1,
    )


def _vegetation_field(
    state: CnPrecisionControlState,
    species: str,
    name: str,
) -> jnp.ndarray:
    return get_cn_precision_control_state_field(
        state,
        f"vegetation_{species}.{name}_patch",
    )


def _state_summary_inputs(
    template: CnDriverSummarizeStatesInputs,
    biogeochemistry: PostDrainagePrecisionControlState,
    excess_respiration_loss: jnp.ndarray,
) -> CnDriverSummarizeStatesInputs:

    vegetation = biogeochemistry.vegetation
    soil = biogeochemistry.soil
    active = slice(0, _ACTIVE_DECOMPOSITION_LEVELS)
    return template._replace(
        decomposition_carbon_column_level_pool=(
            soil.carbon_pool_column_level_pool[:, active, :]
        ),
        carbon_truncation_column_level=(
            soil.carbon_truncation_column_level[:, active]
        ),
        decomposition_nitrogen_column_level_pool=(
            soil.nitrogen_pool_column_level_pool[:, active, :]
        ),
        mineral_nitrogen_column_level=(
            soil.mineral_nitrogen_column_level[:, active]
        ),
        nitrate_nitrogen_column_level=soil.nitrate_column_level[:, active],
        ammonium_nitrogen_column_level=soil.ammonium_column_level[:, active],
        nitrogen_truncation_column_level=(
            soil.nitrogen_truncation_column_level[:, active]
        ),
        displayed_carbon_patch_component=_qualified_components(
            vegetation,
            "carbon",
            CARBON_DISPLAYED_COMPONENTS,
        ),
        stored_carbon_patch_component=_qualified_components(
            vegetation,
            "carbon",
            CARBON_STORED_COMPONENTS,
        ),
        grain_carbon_patch=_vegetation_field(vegetation, "carbon", "grainc"),
        grain_carbon_storage_patch=_vegetation_field(
            vegetation,
            "carbon",
            "grainc_storage",
        ),
        grain_carbon_transfer_patch=_vegetation_field(
            vegetation,
            "carbon",
            "grainc_xfer",
        ),
        crop_seed_carbon_deficit_patch=_vegetation_field(
            vegetation,
            "carbon",
            "cropseedc_deficit",
        ),
        excess_maintenance_respiration_carbon_patch=_vegetation_field(
            vegetation,
            "carbon",
            "xsmrpool",
        ),
        excess_maintenance_respiration_carbon_loss_patch=jnp.asarray(
            excess_respiration_loss,
            dtype=jnp.float64,
        ),
        patch_carbon_truncation=vegetation.carbon_truncation_patch,
        displayed_nitrogen_patch_component=_qualified_components(
            vegetation,
            "nitrogen",
            NITROGEN_DISPLAYED_COMPONENTS,
        ),
        stored_nitrogen_patch_component=_qualified_components(
            vegetation,
            "nitrogen",
            NITROGEN_STORED_COMPONENTS,
        ),
        grain_nitrogen_patch=_vegetation_field(
            vegetation,
            "nitrogen",
            "grainn",
        ),
        grain_nitrogen_storage_patch=_vegetation_field(
            vegetation,
            "nitrogen",
            "grainn_storage",
        ),
        grain_nitrogen_transfer_patch=_vegetation_field(
            vegetation,
            "nitrogen",
            "grainn_xfer",
        ),
        crop_seed_nitrogen_deficit_patch=_vegetation_field(
            vegetation,
            "nitrogen",
            "cropseedn_deficit",
        ),
        patch_nitrogen_truncation=vegetation.nitrogen_truncation_patch,
    )


def _no_vegetation_structure_update(
    state: CnVegStructUpdateState,
) -> CnVegStructUpdateResult:
    zero = jnp.int32(0)
    return CnVegStructUpdateResult(
        state=state,
        diagnostics=CnVegStructUpdateDiagnostics(
            updated_patch_count=zero,
            bare_patch_count=zero,
            woody_patch_count=zero,
            prognostic_crop_patch_count=zero,
            short_vegetation_patch_count=zero,
            harvested_stubble_patch_count=zero,
        ),
    )


def ecosystem_dynamics_post_drainage(
    inputs: EcosystemDynamicsPostDrainageInputs,
    state: EcosystemDynamicsPostDrainageState,
    parameters: EcosystemDynamicsPostDrainageParameters,
) -> EcosystemDynamicsPostDrainageResult:

    if jnp.shape(inputs.do_albedo) != ():
        raise ValueError("do_albedo must be scalar")

    current = state.biogeochemistry
    leaching_inputs = inputs.leaching._replace(
        nitrate_column_level=current.soil.nitrate_column_level,
    )
    leaching = soil_biogeochem_n_leaching(
        leaching_inputs,
        SoilBiogeochemNLeachingFluxState(
            nitrate_leaching_column_level=get_cn_flux_field(
                current.cn_fluxes,
                _LEACHING_FLUX,
            ),
            nitrate_runoff_column_level=get_cn_flux_field(
                current.cn_fluxes,
                _RUNOFF_FLUX,
            ),
        ),
        parameters.leaching,
    )
    fluxes: CnTransientFluxState = set_cn_flux_field(
        current.cn_fluxes,
        _LEACHING_FLUX,
        leaching.flux_state.nitrate_leaching_column_level,
    )
    fluxes = set_cn_flux_field(
        fluxes,
        _RUNOFF_FLUX,
        leaching.flux_state.nitrate_runoff_column_level,
    )

    nitrogen = cn_n_state_update3(
        inputs.nitrogen_state_update,
        CnNStateUpdate3State(
            vegetation=current.vegetation,
            soil_nitrogen_pool_column_level_pool=(
                current.soil.nitrogen_pool_column_level_pool
            ),
            mineral_nitrogen_column_level=(
                current.soil.mineral_nitrogen_column_level
            ),
            ammonium_column_level=current.soil.ammonium_column_level,
            nitrate_column_level=current.soil.nitrate_column_level,
        ),
        fluxes,
        parameters.nitrogen_state_update,
    )
    before_precision = PostDrainagePrecisionControlState(
        vegetation=nitrogen.state.vegetation,
        soil=current.soil._replace(
            nitrogen_pool_column_level_pool=(
                nitrogen.state.soil_nitrogen_pool_column_level_pool
            ),
            mineral_nitrogen_column_level=(
                nitrogen.state.mineral_nitrogen_column_level
            ),
            ammonium_column_level=nitrogen.state.ammonium_column_level,
            nitrate_column_level=nitrogen.state.nitrate_column_level,
        ),
        cn_fluxes=nitrogen.cn_fluxes,
    )
    precision = post_drainage_precision_control(
        inputs.precision,
        before_precision,
        parameters.precision,
    )

    summary = cn_driver_summarize_states(
        _state_summary_inputs(
            inputs.state_summary,
            precision.state,
            inputs.excess_maintenance_respiration_carbon_loss_patch,
        ),
        state.state_summaries,
        parameters.state_summary,
    )

    flux_summary = cn_driver_summarize_fluxes(
        inputs.flux_summary,
        CnDriverSummarizeFluxesState(
            cn_fluxes=precision.state.cn_fluxes,
            temporary_npp_sum_patch=state.flux_summary.temporary_npp_sum_patch,
            lagged_npp_column=state.flux_summary.lagged_npp_column,
            dynamic_landuse_amount_to_dribble_gridcell=(
                state.flux_summary.dynamic_landuse_amount_to_dribble_gridcell
            ),
            harvest_xsmrpool_amount_to_dribble_gridcell=(
                state.flux_summary.harvest_xsmrpool_amount_to_dribble_gridcell
            ),
        ),
        parameters.flux_summary,
    )

    vegetation = precision.state.vegetation
    vegetation_inputs = inputs.vegetation_structure._replace(
        leaf_carbon_patch=_vegetation_field(vegetation, "carbon", "leafc"),
        dead_stem_carbon_patch=_vegetation_field(
            vegetation,
            "carbon",
            "deadstemc",
        ),
    )
    vegetation_structure = lax.cond(
        jnp.asarray(inputs.do_albedo, dtype=jnp.bool_),
        lambda _: cn_veg_struct_update(
            vegetation_inputs,
            state.vegetation_structure,
            parameters.vegetation_structure,
        ),
        lambda _: _no_vegetation_structure_update(state.vegetation_structure),
        operand=None,
    )

    final_biogeochemistry = precision.state._replace(
        cn_fluxes=flux_summary.state.cn_fluxes,
    )
    return EcosystemDynamicsPostDrainageResult(
        state=EcosystemDynamicsPostDrainageState(
            biogeochemistry=final_biogeochemistry,
            state_summaries=summary.state,
            flux_summary=EcosystemDynamicsPostDrainageFluxSummaryState(
                temporary_npp_sum_patch=(
                    flux_summary.state.temporary_npp_sum_patch
                ),
                lagged_npp_column=flux_summary.state.lagged_npp_column,
                dynamic_landuse_amount_to_dribble_gridcell=(
                    flux_summary.state.dynamic_landuse_amount_to_dribble_gridcell
                ),
                harvest_xsmrpool_amount_to_dribble_gridcell=(
                    flux_summary.state.harvest_xsmrpool_amount_to_dribble_gridcell
                ),
            ),
            vegetation_structure=vegetation_structure.state,
        ),
        gridcell_fluxes=flux_summary.gridcell_fluxes,
        diagnostics=EcosystemDynamicsPostDrainageDiagnostics(
            leaching=leaching.diagnostics,
            nitrogen_state_update=nitrogen.diagnostics,
            precision=precision.diagnostics,
            state_summary=summary.diagnostics,
            flux_summary=flux_summary.diagnostics,
            vegetation_structure=vegetation_structure.diagnostics,
        ),
    )


__all__ = [
    "EcosystemDynamicsPostDrainageDiagnostics",
    "EcosystemDynamicsPostDrainageFluxSummaryState",
    "EcosystemDynamicsPostDrainageInputs",
    "EcosystemDynamicsPostDrainageParameters",
    "EcosystemDynamicsPostDrainageResult",
    "EcosystemDynamicsPostDrainageState",
    "ecosystem_dynamics_post_drainage",
]
