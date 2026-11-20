
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.hydrology_diagnostics import (
    DEFAULT_HYDROLOGY_DIAGNOSTICS_PARAMETERS,
    NUMBER_OF_SOIL_LEVELS,
    HydrologyDiagnosticsInputs,
    HydrologyDiagnosticsMasks,
    HydrologyDiagnosticsParameters,
    HydrologyDiagnosticsState,
    hydrology_no_drainage_diagnostics_active_case,
)
from clm_jax.physics.snow_capping import (
    DEFAULT_SNOW_CAPPING_PARAMETERS,
    SnowCappingDiagnostics,
    SnowCappingInputs,
    SnowCappingParameters,
    SnowCappingState,
    snow_capping_active_case,
)
from clm_jax.physics.snow_compaction import (
    DEFAULT_SNOW_COMPACTION_PARAMETERS,
    SnowCompactionDiagnostics,
    SnowCompactionInputs,
    SnowCompactionParameters,
    SnowCompactionState,
    snow_compaction_active_case,
)
from clm_jax.physics.snow_layer_combining import (
    DEFAULT_SNOW_LAYER_COMBINING_PARAMETERS,
    SnowLayerCombiningDiagnostics,
    SnowLayerCombiningParameters,
    SnowLayerCombiningState,
    combine_snow_layers_active_case,
)
from clm_jax.physics.snow_layer_division import (
    DEFAULT_SNOW_LAYER_DIVISION_PARAMETERS,
    SnowLayerDivisionDiagnostics,
    SnowLayerDivisionParameters,
    SnowLayerDivisionState,
    divide_snow_layers_active_case,
)
from clm_jax.physics.snow_layer_postprocess import (
    DEFAULT_SNOW_LAYER_POSTPROCESS_PARAMETERS,
    SnowLayerPostprocessDiagnostics,
    SnowLayerPostprocessInputs,
    SnowLayerPostprocessParameters,
    SnowLayerPostprocessState,
    snow_layer_postprocess_active_case,
)
from clm_jax.physics.snow_water import (
    NUMBER_OF_SNOW_LEVELS,
    SnowWaterDiagnostics,
    SnowWaterInputs,
    SnowWaterState,
    build_snow_column_masks,
    snow_water,
)
from clm_jax.physics.soil_water_movement import (
    DEFAULT_SOIL_WATER_MOVEMENT_PARAMETERS,
    SoilWaterMovementDiagnostics,
    SoilWaterMovementInputs,
    SoilWaterMovementParameters,
    SoilWaterMovementState,
    soil_water_movement_active_case,
)
from clm_jax.physics.soil_water_plant_sink import (
    DEFAULT_SOIL_WATER_PLANT_SINK_PARAMETERS,
    SoilWaterPlantSinkDiagnostics,
    SoilWaterPlantSinkInputs,
    SoilWaterPlantSinkParameters,
    SoilWaterPlantSinkState,
    compute_effective_root_fraction_and_vertical_sink,
)
from clm_jax.physics.surface_hydrology import (
    SurfaceHydrologyDiagnostics,
    SurfaceHydrologyInputs,
    SurfaceHydrologyState,
    surface_runoff_and_infiltration,
)
from clm_jax.physics.water_table import (
    DEFAULT_WATER_TABLE_CONDENSATION_PARAMETERS,
    WaterTableCondensationDiagnostics,
    WaterTableCondensationInputs,
    WaterTableCondensationParameters,
    WaterTableCondensationState,
    water_tables_and_condensation_active_case,
)

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase


class HydrologyNoDrainageParameters(NamedTuple):
    plant_sink: SoilWaterPlantSinkParameters = (
        DEFAULT_SOIL_WATER_PLANT_SINK_PARAMETERS
    )
    soil_water_movement: SoilWaterMovementParameters = (
        DEFAULT_SOIL_WATER_MOVEMENT_PARAMETERS
    )
    water_table_condensation: WaterTableCondensationParameters = (
        DEFAULT_WATER_TABLE_CONDENSATION_PARAMETERS
    )
    snow_capping: SnowCappingParameters = DEFAULT_SNOW_CAPPING_PARAMETERS
    snow_compaction: SnowCompactionParameters = DEFAULT_SNOW_COMPACTION_PARAMETERS
    snow_layer_combining: SnowLayerCombiningParameters = (
        DEFAULT_SNOW_LAYER_COMBINING_PARAMETERS
    )
    snow_layer_division: SnowLayerDivisionParameters = (
        DEFAULT_SNOW_LAYER_DIVISION_PARAMETERS
    )
    snow_layer_postprocess: SnowLayerPostprocessParameters = (
        DEFAULT_SNOW_LAYER_POSTPROCESS_PARAMETERS
    )
    final_diagnostics: HydrologyDiagnosticsParameters = (
        DEFAULT_HYDROLOGY_DIAGNOSTICS_PARAMETERS
    )

    @classmethod
    def from_case(cls, case: CimeCase) -> HydrologyNoDrainageParameters:

        return cls(
            plant_sink=SoilWaterPlantSinkParameters.from_case(case),
            soil_water_movement=SoilWaterMovementParameters.from_case(case),
            water_table_condensation=WaterTableCondensationParameters.from_case(case),
            snow_capping=SnowCappingParameters.from_case(case),
            snow_compaction=SnowCompactionParameters.from_case(case),
            snow_layer_combining=SnowLayerCombiningParameters.from_case(case),
            snow_layer_postprocess=SnowLayerPostprocessParameters.from_case(case),
            final_diagnostics=HydrologyDiagnosticsParameters.from_case(case),
        )


DEFAULT_HYDROLOGY_NO_DRAINAGE_PARAMETERS = HydrologyNoDrainageParameters()


class HydrologyNoDrainageInputs(NamedTuple):

    snow_water: SnowWaterInputs
    surface_hydrology: SurfaceHydrologyInputs
    plant_sink: SoilWaterPlantSinkInputs
    soil_water_movement: SoilWaterMovementInputs
    water_table_condensation: WaterTableCondensationInputs
    snow_compaction: SnowCompactionInputs
    final_diagnostics: HydrologyDiagnosticsInputs


class HydrologyNoDrainageState(NamedTuple):

    snow_structure: SnowLayerCombiningState
    snow_water: SnowWaterState
    surface_hydrology: SurfaceHydrologyState
    plant_sink: SoilWaterPlantSinkState
    soil_water_movement: SoilWaterMovementState
    water_table_condensation: WaterTableCondensationState
    snow_capping: SnowCappingState
    snow_postprocess: SnowLayerPostprocessState
    final_diagnostics: HydrologyDiagnosticsState


class HydrologyNoDrainageDiagnostics(NamedTuple):
    snow_water: SnowWaterDiagnostics
    surface_hydrology: SurfaceHydrologyDiagnostics
    plant_sink: SoilWaterPlantSinkDiagnostics
    soil_water_movement: SoilWaterMovementDiagnostics
    water_table_condensation: WaterTableCondensationDiagnostics
    snow_capping: SnowCappingDiagnostics
    snow_compaction: SnowCompactionDiagnostics
    snow_layer_combining: SnowLayerCombiningDiagnostics
    snow_layer_division: SnowLayerDivisionDiagnostics
    snow_layer_postprocess: SnowLayerPostprocessDiagnostics
    final_diagnostics: HydrologyDiagnosticsMasks


class HydrologyNoDrainageResult(NamedTuple):
    state: HydrologyNoDrainageState
    diagnostics: HydrologyNoDrainageDiagnostics


class HydrologyNoDrainageStageFunctions(NamedTuple):

    snow_water: Callable
    surface_hydrology: Callable
    plant_sink: Callable
    soil_water_movement: Callable
    water_table_condensation: Callable
    snow_capping: Callable
    snow_compaction: Callable
    snow_layer_combining: Callable
    snow_layer_division: Callable
    snow_layer_postprocess: Callable
    final_diagnostics: Callable


def _plant_sink_stage(topology, inputs, state, parameters):
    return compute_effective_root_fraction_and_vertical_sink(
        topology,
        inputs,
        state,
        parameters=parameters,
    )


def _soil_water_movement_stage(topology, inputs, state, parameters):
    return soil_water_movement_active_case(
        topology,
        inputs,
        state,
        parameters=parameters,
    )


def _water_table_condensation_stage(topology, inputs, state, parameters):
    return water_tables_and_condensation_active_case(
        topology,
        inputs,
        state,
        parameters=parameters,
    )


def _snow_capping_stage(topology, inputs, state, parameters):
    return snow_capping_active_case(
        topology,
        inputs,
        state,
        parameters=parameters,
    )


def _snow_compaction_stage(topology, inputs, state, parameters):
    return snow_compaction_active_case(
        topology,
        inputs,
        state,
        parameters=parameters,
    )


def _snow_layer_combining_stage(topology, state, parameters):
    return combine_snow_layers_active_case(
        topology,
        state,
        parameters=parameters,
    )


def _snow_layer_division_stage(topology, state, parameters):
    return divide_snow_layers_active_case(
        topology,
        state,
        parameters=parameters,
    )


def _snow_layer_postprocess_stage(topology, inputs, state, parameters):
    return snow_layer_postprocess_active_case(
        topology,
        inputs,
        state,
        parameters=parameters,
    )


def _final_diagnostics_stage(topology, inputs, state, parameters):
    return hydrology_no_drainage_diagnostics_active_case(
        topology,
        inputs,
        state,
        parameters=parameters,
    )


DIRECT_HYDROLOGY_NO_DRAINAGE_STAGES = HydrologyNoDrainageStageFunctions(
    snow_water=snow_water,
    surface_hydrology=surface_runoff_and_infiltration,
    plant_sink=_plant_sink_stage,
    soil_water_movement=_soil_water_movement_stage,
    water_table_condensation=_water_table_condensation_stage,
    snow_capping=_snow_capping_stage,
    snow_compaction=_snow_compaction_stage,
    snow_layer_combining=_snow_layer_combining_stage,
    snow_layer_division=_snow_layer_division_stage,
    snow_layer_postprocess=_snow_layer_postprocess_stage,
    final_diagnostics=_final_diagnostics_stage,
)


def hydrology_no_drainage_with_stages(
    topology: SubgridTopology,
    inputs: HydrologyNoDrainageInputs,
    state: HydrologyNoDrainageState,
    *,
    parameters: HydrologyNoDrainageParameters,
    stages: HydrologyNoDrainageStageFunctions,
) -> HydrologyNoDrainageResult:

    snow = state.snow_structure
    initial_snow_masks = build_snow_column_masks(
        topology,
        snow.snow_layer_count_column,
    )

    snow_water_inputs = inputs.snow_water._replace(
        snow_layer_count_column=snow.snow_layer_count_column,
        effective_snow_fraction_column=snow.effective_snow_fraction_column,
        snow_water_equivalent_column=snow.snow_water_equivalent_column,
    )
    snow_water_state = state.snow_water._replace(
        integrated_snow_column=snow.integrated_snow_column,
        snow_fraction_column=snow.snow_fraction_column,
        snow_depth_column=snow.snow_depth_column,
        soil_snow_liquid_column_level=snow.soil_snow_liquid_column_level,
        soil_snow_ice_column_level=snow.soil_snow_ice_column_level,
        soil_snow_thickness_column_level=snow.soil_snow_thickness_column_level,
        top_soil_water_flux_column=state.snow_water.top_soil_water_flux_column,
        aerosol=snow.aerosol,
    )
    snow_water_result = stages.snow_water(
        topology,
        snow_water_inputs,
        snow_water_state,
    )
    current_integrated_snow = snow_water_result.state.integrated_snow_column
    current_snow_fraction = snow_water_result.state.snow_fraction_column
    current_snow_depth = snow_water_result.state.snow_depth_column
    current_liquid = snow_water_result.state.soil_snow_liquid_column_level
    current_ice = snow_water_result.state.soil_snow_ice_column_level
    current_thickness = snow_water_result.state.soil_snow_thickness_column_level
    current_top_soil_flux = snow_water_result.state.top_soil_water_flux_column
    current_aerosol = snow_water_result.state.aerosol

    surface_inputs = inputs.surface_hydrology._replace(
        snow_layer_count_column=snow.snow_layer_count_column,
        soil_snow_thickness_column_level=current_thickness,
        soil_snow_ice_column_level=current_ice,
        effective_snow_fraction_column=snow.effective_snow_fraction_column,
        water_table_depth_column=(
            state.water_table_condensation.water_table_depth_column
        ),
        frost_table_depth_column=(
            state.water_table_condensation.frost_table_depth_column
        ),
        perched_water_table_depth_column=(
            state.water_table_condensation.perched_water_table_depth_column
        ),
    )
    surface_state = state.surface_hydrology._replace(
        soil_snow_liquid_column_level=current_liquid,
        top_soil_water_flux_column=current_top_soil_flux,
    )
    surface_result = stages.surface_hydrology(
        topology,
        surface_inputs,
        surface_state,
    )
    current_liquid = surface_result.state.soil_snow_liquid_column_level
    current_top_soil_flux = surface_result.state.top_soil_water_flux_column

    ground_slice = slice(NUMBER_OF_SNOW_LEVELS, None)
    plant_sink_inputs = inputs.plant_sink._replace(
        soil_water_potential_column_level=(
            state.final_diagnostics.soil_matrix_potential_column_level
        ),
        soil_node_depth_column_level=(
            snow.soil_snow_node_depth_column_level[:, ground_slice]
        ),
    )
    plant_sink_result = stages.plant_sink(
        topology,
        plant_sink_inputs,
        state.plant_sink,
        parameters.plant_sink,
    )

    movement_inputs = inputs.soil_water_movement._replace(
        layer_thickness_column_level=current_thickness[:, ground_slice],
        node_depth_column_level=snow.soil_snow_node_depth_column_level[
            :, ground_slice
        ],
        ice_fraction_column_level=(
            surface_result.state.soil_ice_fraction_column_level
        ),
        infiltration_flux_column=surface_result.state.infiltration_flux_column,
        root_water_sink_column_level=(
            plant_sink_result.state.root_soil_water_flux_column_level
        ),
        ice_water_mass_column_level=current_ice[:, ground_slice],
    )
    movement_state = state.soil_water_movement._replace(
        liquid_water_mass_column_level=current_liquid[:, ground_slice],
        soil_water_potential_column_level=(
            state.final_diagnostics.soil_matrix_potential_column_level
        ),
    )
    movement_result = stages.soil_water_movement(
        topology,
        movement_inputs,
        movement_state,
        parameters.soil_water_movement,
    )
    current_liquid = current_liquid.at[:, ground_slice].set(
        movement_result.state.liquid_water_mass_column_level
    )

    soil_interface_start = NUMBER_OF_SNOW_LEVELS + 1
    soil_interface_stop = soil_interface_start + NUMBER_OF_SOIL_LEVELS
    water_table_inputs = inputs.water_table_condensation._replace(
        soil_layer_count_column=movement_inputs.soil_layer_count_column,
        snow_layer_count_column=snow.snow_layer_count_column,
        layer_thickness_column_level=current_thickness[:, ground_slice],
        node_depth_column_level=snow.soil_snow_node_depth_column_level[
            :, ground_slice
        ],
        lower_interface_depth_column_level=(
            snow.soil_snow_interface_depth_column_interface[
                :,
                soil_interface_start:soil_interface_stop,
            ]
        ),
        soil_temperature_column_level=snow.soil_snow_temperature_column_level[
            :, ground_slice
        ],
        saturated_water_content_column_level=(
            movement_inputs.saturated_water_content_column_level
        ),
        surface_water_fraction_column=(
            surface_inputs.surface_water_fraction_column
        ),
        ground_dew_flux_column=snow_water_inputs.ground_dew_flux_column,
        snow_dew_flux_column=snow_water_inputs.snow_dew_flux_column,
    )
    water_table_state = state.water_table_condensation._replace(
        soil_snow_liquid_water_column_level=current_liquid,
        soil_snow_ice_water_column_level=current_ice,
        volumetric_water_column_level=(
            movement_result.state.volumetric_water_column_level
        ),
        snow_sublimation_flux_column=snow_water_inputs.snow_sublimation_flux_column,
    )
    water_table_result = stages.water_table_condensation(
        topology,
        water_table_inputs,
        water_table_state,
        parameters.water_table_condensation,
    )
    current_liquid = (
        water_table_result.state.soil_snow_liquid_water_column_level
    )
    current_ice = water_table_result.state.soil_snow_ice_water_column_level

    capping_inputs = SnowCappingInputs(
        snow_layer_count_column=snow.snow_layer_count_column,
        snow_water_equivalent_column=snow.snow_water_equivalent_column,
    )
    capping_state = state.snow_capping._replace(
        soil_snow_liquid_column_level=current_liquid,
        soil_snow_ice_column_level=current_ice,
        soil_snow_thickness_column_level=current_thickness,
        aerosol=current_aerosol,
    )
    capping_result = stages.snow_capping(
        topology,
        capping_inputs,
        capping_state,
        parameters.snow_capping,
    )
    current_liquid = capping_result.state.soil_snow_liquid_column_level
    current_ice = capping_result.state.soil_snow_ice_column_level
    current_thickness = capping_result.state.soil_snow_thickness_column_level
    current_aerosol = capping_result.state.aerosol

    compaction_inputs = inputs.snow_compaction._replace(
        snow_layer_count_column=snow.snow_layer_count_column,
        soil_snow_temperature_column_level=snow.soil_snow_temperature_column_level,
        effective_snow_fraction_column=snow.effective_snow_fraction_column,
        integrated_snow_column=current_integrated_snow,
        soil_snow_ice_column_level=current_ice,
        soil_snow_liquid_column_level=current_liquid,
    )
    compaction_result = stages.snow_compaction(
        topology,
        compaction_inputs,
        SnowCompactionState(
            soil_snow_thickness_column_level=current_thickness,
        ),
        parameters.snow_compaction,
    )
    current_thickness = (
        compaction_result.state.soil_snow_thickness_column_level
    )

    combining_state = SnowLayerCombiningState(
        snow_layer_count_column=snow.snow_layer_count_column,
        snow_fraction_column=current_snow_fraction,
        effective_snow_fraction_column=snow.effective_snow_fraction_column,
        snow_depth_column=current_snow_depth,
        integrated_snow_column=current_integrated_snow,
        snow_water_equivalent_column=snow.snow_water_equivalent_column,
        soil_snow_thickness_column_level=current_thickness,
        soil_snow_node_depth_column_level=(
            snow.soil_snow_node_depth_column_level
        ),
        soil_snow_interface_depth_column_interface=(
            snow.soil_snow_interface_depth_column_interface
        ),
        soil_snow_temperature_column_level=(
            snow.soil_snow_temperature_column_level
        ),
        soil_snow_ice_column_level=current_ice,
        soil_snow_liquid_column_level=current_liquid,
        snow_grain_radius_column_level=snow.snow_grain_radius_column_level,
        top_soil_water_flux_column=snow.top_soil_water_flux_column,
        aerosol=current_aerosol,
    )
    combining_result = stages.snow_layer_combining(
        topology,
        combining_state,
        parameters.snow_layer_combining,
    )
    combined = combining_result.state

    division_result = stages.snow_layer_division(
        topology,
        SnowLayerDivisionState(
            snow_layer_count_column=combined.snow_layer_count_column,
            effective_snow_fraction_column=(
                combined.effective_snow_fraction_column
            ),
            soil_snow_thickness_column_level=(
                combined.soil_snow_thickness_column_level
            ),
            soil_snow_node_depth_column_level=(
                combined.soil_snow_node_depth_column_level
            ),
            soil_snow_interface_depth_column_interface=(
                combined.soil_snow_interface_depth_column_interface
            ),
            soil_snow_temperature_column_level=(
                combined.soil_snow_temperature_column_level
            ),
            soil_snow_ice_column_level=combined.soil_snow_ice_column_level,
            soil_snow_liquid_column_level=combined.soil_snow_liquid_column_level,
            snow_grain_radius_column_level=(
                combined.snow_grain_radius_column_level
            ),
            aerosol=combined.aerosol,
        ),
        parameters.snow_layer_division,
    )
    divided = division_result.state

    postprocess_result = stages.snow_layer_postprocess(
        topology,
        SnowLayerPostprocessInputs(
            previous_snow_column=initial_snow_masks.snow_column,
            snow_depth_column=combined.snow_depth_column,
            effective_snow_fraction_column=(
                divided.effective_snow_fraction_column
            ),
        ),
        state.snow_postprocess._replace(
            snow_layer_count_column=divided.snow_layer_count_column,
            soil_snow_thickness_column_level=(
                divided.soil_snow_thickness_column_level
            ),
            soil_snow_node_depth_column_level=(
                divided.soil_snow_node_depth_column_level
            ),
            soil_snow_interface_depth_column_interface=(
                divided.soil_snow_interface_depth_column_interface
            ),
            soil_snow_temperature_column_level=(
                divided.soil_snow_temperature_column_level
            ),
            soil_snow_ice_column_level=divided.soil_snow_ice_column_level,
            soil_snow_liquid_column_level=divided.soil_snow_liquid_column_level,
        ),
        parameters.snow_layer_postprocess,
    )
    postprocessed = postprocess_result.state

    final_inputs = inputs.final_diagnostics._replace(
        snow_layer_count_column=postprocessed.snow_layer_count_column,
        soil_snow_thickness_column_level=(
            postprocessed.soil_snow_thickness_column_level
        ),
        soil_snow_node_depth_column_level=(
            postprocessed.soil_snow_node_depth_column_level
        ),
        soil_snow_interface_depth_column_interface=(
            postprocessed.soil_snow_interface_depth_column_interface
        ),
        soil_snow_temperature_column_level=(
            postprocessed.soil_snow_temperature_column_level
        ),
        soil_snow_ice_column_level=postprocessed.soil_snow_ice_column_level,
        soil_snow_liquid_column_level=postprocessed.soil_snow_liquid_column_level,
        effective_snow_fraction_column=(
            divided.effective_snow_fraction_column
        ),
        surface_water_fraction_column=surface_inputs.surface_water_fraction_column,
        soil_porosity_column_level=(
            movement_inputs.saturated_water_content_column_level
        ),
        saturated_suction_column_level=(
            movement_inputs.saturated_suction_column_level
        ),
        clapp_hornberger_exponent_column_level=(
            movement_inputs.clapp_hornberger_exponent_column_level
        ),
    )
    final_diagnostics_state = state.final_diagnostics._replace(
        volumetric_soil_water_column_level=(
            water_table_result.state.volumetric_water_column_level
        ),
        soil_matrix_potential_column_level=(
            movement_result.state.soil_water_potential_column_level
        ),
        top_snow_water_column=postprocessed.top_snow_water_column,
        snow_grain_radius_column_level=divided.snow_grain_radius_column_level,
    )
    final_result = stages.final_diagnostics(
        topology,
        final_inputs,
        final_diagnostics_state,
        parameters.final_diagnostics,
    )

    final_snow = combined._replace(
        snow_layer_count_column=postprocessed.snow_layer_count_column,
        effective_snow_fraction_column=divided.effective_snow_fraction_column,
        soil_snow_thickness_column_level=(
            postprocessed.soil_snow_thickness_column_level
        ),
        soil_snow_node_depth_column_level=(
            postprocessed.soil_snow_node_depth_column_level
        ),
        soil_snow_interface_depth_column_interface=(
            postprocessed.soil_snow_interface_depth_column_interface
        ),
        soil_snow_temperature_column_level=(
            postprocessed.soil_snow_temperature_column_level
        ),
        soil_snow_ice_column_level=postprocessed.soil_snow_ice_column_level,
        soil_snow_liquid_column_level=postprocessed.soil_snow_liquid_column_level,
        snow_grain_radius_column_level=final_result.state.snow_grain_radius_column_level,
        aerosol=divided.aerosol,
    )
    synchronized_snow_water = snow_water_result.state._replace(
        integrated_snow_column=final_snow.integrated_snow_column,
        snow_fraction_column=final_snow.snow_fraction_column,
        snow_depth_column=final_snow.snow_depth_column,
        soil_snow_liquid_column_level=final_snow.soil_snow_liquid_column_level,
        soil_snow_ice_column_level=final_snow.soil_snow_ice_column_level,
        soil_snow_thickness_column_level=final_snow.soil_snow_thickness_column_level,
        top_soil_water_flux_column=current_top_soil_flux,
        aerosol=final_snow.aerosol,
    )
    synchronized_surface = surface_result.state._replace(
        soil_snow_liquid_column_level=final_snow.soil_snow_liquid_column_level,
        top_soil_water_flux_column=current_top_soil_flux,
    )
    synchronized_movement = movement_result.state._replace(
        liquid_water_mass_column_level=final_snow.soil_snow_liquid_column_level[
            :, ground_slice
        ],
        volumetric_water_column_level=(
            final_result.state.volumetric_soil_water_column_level
        ),
        soil_water_potential_column_level=(
            final_result.state.soil_matrix_potential_column_level
        ),
    )
    synchronized_water_table = water_table_result.state._replace(
        soil_snow_liquid_water_column_level=(
            final_snow.soil_snow_liquid_column_level
        ),
        soil_snow_ice_water_column_level=final_snow.soil_snow_ice_column_level,
        volumetric_water_column_level=(
            final_result.state.volumetric_soil_water_column_level
        ),
    )
    synchronized_capping = capping_result.state._replace(
        soil_snow_liquid_column_level=final_snow.soil_snow_liquid_column_level,
        soil_snow_ice_column_level=final_snow.soil_snow_ice_column_level,
        soil_snow_thickness_column_level=final_snow.soil_snow_thickness_column_level,
        aerosol=final_snow.aerosol,
    )

    return HydrologyNoDrainageResult(
        state=HydrologyNoDrainageState(
            snow_structure=final_snow,
            snow_water=synchronized_snow_water,
            surface_hydrology=synchronized_surface,
            plant_sink=plant_sink_result.state,
            soil_water_movement=synchronized_movement,
            water_table_condensation=synchronized_water_table,
            snow_capping=synchronized_capping,
            snow_postprocess=postprocessed,
            final_diagnostics=final_result.state,
        ),
        diagnostics=HydrologyNoDrainageDiagnostics(
            snow_water=snow_water_result.diagnostics,
            surface_hydrology=surface_result.diagnostics,
            plant_sink=plant_sink_result.diagnostics,
            soil_water_movement=movement_result.diagnostics,
            water_table_condensation=water_table_result.diagnostics,
            snow_capping=capping_result.diagnostics,
            snow_compaction=compaction_result.diagnostics,
            snow_layer_combining=combining_result.diagnostics,
            snow_layer_division=division_result.diagnostics,
            snow_layer_postprocess=postprocess_result.diagnostics,
            final_diagnostics=final_result.masks,
        ),
    )


def hydrology_no_drainage_active_case(
    topology: SubgridTopology,
    inputs: HydrologyNoDrainageInputs,
    state: HydrologyNoDrainageState,
    *,
    parameters: HydrologyNoDrainageParameters = (
        DEFAULT_HYDROLOGY_NO_DRAINAGE_PARAMETERS
    ),
) -> HydrologyNoDrainageResult:

    return hydrology_no_drainage_with_stages(
        topology,
        inputs,
        state,
        parameters=parameters,
        stages=DIRECT_HYDROLOGY_NO_DRAINAGE_STAGES,
    )
