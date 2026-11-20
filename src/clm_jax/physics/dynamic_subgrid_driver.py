
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.dynamic_area_conservation import (
    DynamicAreaConservationInputs,
    DynamicAreaConservationParameters,
    DynamicAreaConservationResult,
    DynamicAreaConservationState,
    dynamic_area_conservation,
)
from clm_jax.physics.dynamic_subgrid import (
    DynamicBiogeophysicalBalanceResult,
    DynamicBiogeophysicalDribblerState,
    DynamicHeatWaterContent,
    DynamicSubgridWeightDiagnostics,
    dynamic_biogeophysical_balance_step,
    dynamic_heat_water_content,
    dynamic_subgrid_weight_diagnostics,
)
from clm_jax.physics.dynamic_subgrid_weights import (
    ColumnWeightTransition,
    DynamicPriorWeights,
    NewColumnInitializationResult,
    NewColumnState,
    PatchWeightTransition,
    initialize_new_columns,
    prepare_column_weight_transition,
    prepare_patch_weight_transition,
    save_prior_weights,
)
from clm_jax.physics.total_water_and_heat import TotalWaterAndHeatInputs


class DynamicSubgridDriverInputs(NamedTuple):
    total_water_and_heat_before: TotalWaterAndHeatInputs
    total_water_and_heat_after: TotalWaterAndHeatInputs
    canyon_height_to_width_ratio_landunit_before: jnp.ndarray
    canyon_height_to_width_ratio_landunit_after: jnp.ndarray
    area_conservation: DynamicAreaConservationInputs


class DynamicSubgridDriverParameters(NamedTuple):
    area_conservation: DynamicAreaConservationParameters
    for_testing_zero_dynamic_balance_fluxes: bool | jnp.ndarray = False


class DynamicSubgridDriverState(NamedTuple):
    new_column: NewColumnState
    biogeophysical_dribbler: DynamicBiogeophysicalDribblerState
    area_conservation: DynamicAreaConservationState


class DynamicSubgridDriverDiagnostics(NamedTuple):
    content_before: DynamicHeatWaterContent
    prior_weights: DynamicPriorWeights
    patch_transition: PatchWeightTransition
    column_transition: ColumnWeightTransition
    weight_diagnostics: DynamicSubgridWeightDiagnostics
    new_column_initialization: NewColumnInitializationResult


class DynamicSubgridDriverResult(NamedTuple):
    state: DynamicSubgridDriverState
    synchronized_total_water_and_heat: TotalWaterAndHeatInputs
    biogeophysical_balance: DynamicBiogeophysicalBalanceResult
    area_conservation: DynamicAreaConservationResult
    diagnostics: DynamicSubgridDriverDiagnostics


_COMPILED_DYNAMIC_HEAT_WATER_CONTENT = jax.jit(
    dynamic_heat_water_content,
    inline=False,
)
_COMPILED_SAVE_PRIOR_WEIGHTS = jax.jit(
    save_prior_weights,
    inline=False,
)
_COMPILED_PREPARE_PATCH_WEIGHT_TRANSITION = jax.jit(
    prepare_patch_weight_transition,
    inline=False,
)
_COMPILED_PREPARE_COLUMN_WEIGHT_TRANSITION = jax.jit(
    prepare_column_weight_transition,
    inline=False,
)
_COMPILED_DYNAMIC_SUBGRID_WEIGHT_DIAGNOSTICS = jax.jit(
    dynamic_subgrid_weight_diagnostics,
    inline=False,
)
_COMPILED_INITIALIZE_NEW_COLUMNS = jax.jit(
    initialize_new_columns,
    inline=False,
)
_COMPILED_DYNAMIC_BIOGEOPHYSICAL_BALANCE_STEP = jax.jit(
    dynamic_biogeophysical_balance_step,
    inline=False,
)
_COMPILED_DYNAMIC_AREA_CONSERVATION = jax.jit(
    dynamic_area_conservation,
    inline=False,
)


def _dynamic_subgrid_driver_active_case(
    topology_before: SubgridTopology,
    topology_after: SubgridTopology,
    inputs: DynamicSubgridDriverInputs,
    state: DynamicSubgridDriverState,
    parameters: DynamicSubgridDriverParameters,
    *,
    compiled_children: bool,
) -> DynamicSubgridDriverResult:

    heat_water_kernel = (
        _COMPILED_DYNAMIC_HEAT_WATER_CONTENT if compiled_children else dynamic_heat_water_content
    )
    prior_weight_kernel = _COMPILED_SAVE_PRIOR_WEIGHTS if compiled_children else save_prior_weights
    patch_transition_kernel = (
        _COMPILED_PREPARE_PATCH_WEIGHT_TRANSITION
        if compiled_children
        else prepare_patch_weight_transition
    )
    column_transition_kernel = (
        _COMPILED_PREPARE_COLUMN_WEIGHT_TRANSITION
        if compiled_children
        else prepare_column_weight_transition
    )
    diagnostic_kernel = (
        _COMPILED_DYNAMIC_SUBGRID_WEIGHT_DIAGNOSTICS
        if compiled_children
        else dynamic_subgrid_weight_diagnostics
    )
    initialize_column_kernel = (
        _COMPILED_INITIALIZE_NEW_COLUMNS if compiled_children else initialize_new_columns
    )
    biogeophysical_kernel = (
        _COMPILED_DYNAMIC_BIOGEOPHYSICAL_BALANCE_STEP
        if compiled_children
        else dynamic_biogeophysical_balance_step
    )
    area_conservation_kernel = (
        _COMPILED_DYNAMIC_AREA_CONSERVATION if compiled_children else dynamic_area_conservation
    )

    content_before = heat_water_kernel(
        topology_before,
        inputs.total_water_and_heat_before,
        inputs.canyon_height_to_width_ratio_landunit_before,
    )
    prior_weights = prior_weight_kernel(topology_before)
    patch_transition = patch_transition_kernel(topology_before, topology_after)
    column_transition = column_transition_kernel(topology_before, topology_after)

    weight_diagnostics = diagnostic_kernel(topology_after)
    new_column = initialize_column_kernel(
        topology_after,
        prior_weights.column_active,
        state.new_column,
    )

    total_after = inputs.total_water_and_heat_after._replace(
        soil_snow_temperature_column_level=(new_column.state.soil_snow_temperature_column_level),
        soil_snow_liquid_water_column_level=(new_column.state.soil_snow_liquid_water_column_level),
        soil_snow_ice_water_column_level=(new_column.state.soil_snow_ice_water_column_level),
        aquifer_water_column=new_column.state.aquifer_water_column,
    )
    clock = inputs.area_conservation.clock
    biogeophysical = biogeophysical_kernel(
        content_before,
        topology_after,
        total_after,
        inputs.canyon_height_to_width_ratio_landunit_after,
        state.biogeophysical_dribbler,
        is_beginning_current_year=clock.is_beginning_current_year,
        is_first_step=clock.is_first_step,
        previous_year_fraction=clock.previous_year_fraction,
        current_year_fraction=clock.current_year_fraction,
        time_step_seconds=clock.timestep_seconds,
        days_per_year=clock.days_per_year,
        for_testing_zero_dynamic_balance_fluxes=(
            parameters.for_testing_zero_dynamic_balance_fluxes
        ),
    )
    area = area_conservation_kernel(
        topology_after,
        patch_transition,
        column_transition,
        inputs.area_conservation,
        parameters.area_conservation,
        state.area_conservation,
    )
    updated_dribblers = DynamicBiogeophysicalDribblerState(
        liquid_water=biogeophysical.liquid_water_flux.state,
        ice_water=biogeophysical.ice_water_flux.state,
        energy=biogeophysical.energy_flux.state,
    )
    return DynamicSubgridDriverResult(
        state=DynamicSubgridDriverState(
            new_column=new_column.state,
            biogeophysical_dribbler=updated_dribblers,
            area_conservation=area.state,
        ),
        synchronized_total_water_and_heat=total_after,
        biogeophysical_balance=biogeophysical,
        area_conservation=area,
        diagnostics=DynamicSubgridDriverDiagnostics(
            content_before=content_before,
            prior_weights=prior_weights,
            patch_transition=patch_transition,
            column_transition=column_transition,
            weight_diagnostics=weight_diagnostics,
            new_column_initialization=new_column,
        ),
    )


def dynamic_subgrid_driver_active_case(
    topology_before: SubgridTopology,
    topology_after: SubgridTopology,
    inputs: DynamicSubgridDriverInputs,
    state: DynamicSubgridDriverState,
    parameters: DynamicSubgridDriverParameters,
) -> DynamicSubgridDriverResult:

    return _dynamic_subgrid_driver_active_case(
        topology_before,
        topology_after,
        inputs,
        state,
        parameters,
        compiled_children=False,
    )


def dynamic_subgrid_driver_active_case_staged(
    topology_before: SubgridTopology,
    topology_after: SubgridTopology,
    inputs: DynamicSubgridDriverInputs,
    state: DynamicSubgridDriverState,
    parameters: DynamicSubgridDriverParameters,
) -> DynamicSubgridDriverResult:

    return _dynamic_subgrid_driver_active_case(
        topology_before,
        topology_after,
        inputs,
        state,
        parameters,
        compiled_children=True,
    )


__all__ = [
    "DynamicSubgridDriverDiagnostics",
    "DynamicSubgridDriverInputs",
    "DynamicSubgridDriverParameters",
    "DynamicSubgridDriverResult",
    "DynamicSubgridDriverState",
    "dynamic_subgrid_driver_active_case",
    "dynamic_subgrid_driver_active_case_staged",
]
