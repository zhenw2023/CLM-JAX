
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.bgc_timestep_initialization import (
    DynamicLandcoverCarbonFluxState,
    DynamicLandcoverNitrogenFluxState,
)
from clm_jax.physics.dynamic_subgrid_biogeochem import (
    DynamicPatchFluxAggregationInputs,
    DynamicPatchFluxAggregationParameters,
    DynamicPatchInitializationClock,
    DynamicPatchInitializationState,
    DynamicPatchStateUpdateInputs,
    DynamicPatchStateUpdateState,
    DynamicVegetationPatchFluxes,
    DynamicVegetationPatchInputs,
    DynamicVegetationPatchParameters,
    DynamicVegetationPatchRateFluxes,
    DynamicVegetationPatchState,
    aggregate_dynamic_vegetation_patch_fluxes,
    apply_dynamic_patch_carbon_nitrogen_fluxes,
    dynamic_soil_column_adjustments,
    dynamic_vegetation_patch_adjustments,
    initialize_dynamic_vegetation_patches,
)
from clm_jax.physics.dynamic_subgrid_methane import (
    DEFAULT_DYNAMIC_METHANE_COLUMN_PARAMETERS,
    DynamicMethaneColumnParameters,
    DynamicMethaneColumnState,
    dynamic_methane_column_adjustments,
)
from clm_jax.physics.dynamic_subgrid_weights import (
    ColumnWeightTransition,
    PatchWeightTransition,
)
from clm_jax.physics.soil_biogeochem_precision_control import (
    DEFAULT_SOIL_BIOGEOCHEM_PRECISION_CONTROL_PARAMETERS,
    SoilBiogeochemPrecisionControlDiagnostics,
    SoilBiogeochemPrecisionControlInputs,
    SoilBiogeochemPrecisionControlParameters,
    soil_biogeochem_precision_control,
)


class DynamicAreaConservationInputs(NamedTuple):
    clock: DynamicPatchInitializationClock
    vegetation_patch: DynamicVegetationPatchInputs
    patch_flux_aggregation: DynamicPatchFluxAggregationInputs
    patch_state_update: DynamicPatchStateUpdateInputs
    decomposition_layer_thickness: jnp.ndarray
    methane_layer_thickness_column_level: jnp.ndarray


class DynamicAreaConservationParameters(NamedTuple):
    vegetation_patch: DynamicVegetationPatchParameters
    patch_flux_aggregation: DynamicPatchFluxAggregationParameters
    soil_precision_control: SoilBiogeochemPrecisionControlParameters = (
        DEFAULT_SOIL_BIOGEOCHEM_PRECISION_CONTROL_PARAMETERS
    )
    methane_column: DynamicMethaneColumnParameters = (
        DEFAULT_DYNAMIC_METHANE_COLUMN_PARAMETERS
    )


class DynamicAreaConservationState(NamedTuple):
    patch_initialization: DynamicPatchInitializationState
    vegetation_patch: DynamicVegetationPatchState
    patch_state_update: DynamicPatchStateUpdateState
    carbon_flux: DynamicLandcoverCarbonFluxState
    nitrogen_flux: DynamicLandcoverNitrogenFluxState
    methane_column: DynamicMethaneColumnState


class DynamicAreaConservationDiagnostics(NamedTuple):
    initiating_patch: jnp.ndarray
    raw_weight_change_patch: jnp.ndarray
    illegal_nonannual_weight_change_patch: jnp.ndarray
    soil_precision_control: SoilBiogeochemPrecisionControlDiagnostics
    updated_column_level_count: jnp.ndarray
    updated_gridcell_count: jnp.ndarray
    carbon_balance_adjustment_column: jnp.ndarray
    nitrogen_balance_adjustment_column: jnp.ndarray
    nitrate_balance_adjustment_column: jnp.ndarray
    ammonium_balance_adjustment_column: jnp.ndarray
    methane_balance_adjustment_column: jnp.ndarray


class DynamicAreaConservationResult(NamedTuple):
    state: DynamicAreaConservationState
    patch_amounts: DynamicVegetationPatchFluxes
    patch_rate_fluxes: DynamicVegetationPatchRateFluxes
    diagnostics: DynamicAreaConservationDiagnostics


def dynamic_area_conservation(
    topology: SubgridTopology,
    patch_transition: PatchWeightTransition,
    column_transition: ColumnWeightTransition,
    inputs: DynamicAreaConservationInputs,
    parameters: DynamicAreaConservationParameters,
    state: DynamicAreaConservationState,
) -> DynamicAreaConservationResult:

    patch_initialization = initialize_dynamic_vegetation_patches(
        topology,
        patch_transition,
        inputs.clock,
        state.patch_initialization,
    )
    vegetation_patch = dynamic_vegetation_patch_adjustments(
        patch_transition,
        inputs.vegetation_patch,
        parameters.vegetation_patch,
        state.vegetation_patch,
    )
    patch_fluxes = aggregate_dynamic_vegetation_patch_fluxes(
        topology,
        inputs.patch_flux_aggregation,
        parameters.patch_flux_aggregation,
        vegetation_patch.fluxes,
        state.carbon_flux,
        state.nitrogen_flux,
    )

    patch_state_update = apply_dynamic_patch_carbon_nitrogen_fluxes(
        inputs.patch_state_update,
        patch_fluxes.carbon_flux,
        patch_fluxes.nitrogen_flux,
        state.patch_state_update,
    )
    precision_control = soil_biogeochem_precision_control(
        SoilBiogeochemPrecisionControlInputs(
            soil_column=inputs.patch_state_update.soil_column_with_inactive,
        ),
        patch_state_update.state.soil_carbon_nitrogen,
        parameters.soil_precision_control,
    )
    soil_column = dynamic_soil_column_adjustments(
        column_transition,
        topology,
        precision_control.state,
        inputs.decomposition_layer_thickness,
    )
    methane_column = dynamic_methane_column_adjustments(
        column_transition,
        topology,
        state.methane_column,
        inputs.methane_layer_thickness_column_level,
        parameters.methane_column,
    )

    final_patch_state = patch_state_update.state._replace(
        soil_carbon_nitrogen=soil_column.state,
    )
    return DynamicAreaConservationResult(
        state=DynamicAreaConservationState(
            patch_initialization=patch_initialization.state,
            vegetation_patch=vegetation_patch.state,
            patch_state_update=final_patch_state,
            carbon_flux=patch_fluxes.carbon_flux,
            nitrogen_flux=patch_fluxes.nitrogen_flux,
            methane_column=methane_column.state,
        ),
        patch_amounts=vegetation_patch.fluxes,
        patch_rate_fluxes=patch_fluxes.patch_flux,
        diagnostics=DynamicAreaConservationDiagnostics(
            initiating_patch=patch_initialization.initiating_patch,
            raw_weight_change_patch=patch_initialization.raw_weight_change_patch,
            illegal_nonannual_weight_change_patch=(
                patch_initialization.illegal_nonannual_weight_change_patch
            ),
            soil_precision_control=precision_control.diagnostics,
            updated_column_level_count=(
                patch_state_update.updated_column_level_count
            ),
            updated_gridcell_count=patch_state_update.updated_gridcell_count,
            carbon_balance_adjustment_column=(
                soil_column.carbon_balance_adjustment_column
            ),
            nitrogen_balance_adjustment_column=(
                soil_column.nitrogen_balance_adjustment_column
            ),
            nitrate_balance_adjustment_column=(
                soil_column.nitrate_balance_adjustment_column
            ),
            ammonium_balance_adjustment_column=(
                soil_column.ammonium_balance_adjustment_column
            ),
            methane_balance_adjustment_column=(
                methane_column.methane_balance_adjustment_column
            ),
        ),
    )


__all__ = [
    "DynamicAreaConservationDiagnostics",
    "DynamicAreaConservationInputs",
    "DynamicAreaConservationParameters",
    "DynamicAreaConservationResult",
    "DynamicAreaConservationState",
    "dynamic_area_conservation",
]
