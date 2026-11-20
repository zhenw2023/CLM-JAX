
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from clm_jax.io.restart import LANDUNIT_URBAN_MAX, LANDUNIT_URBAN_MIN, SubgridTopology
from clm_jax.physics.numerics import fortran_divide, fortran_multiply
from clm_jax.physics.soil_temperature import (
    LIQUID_WATER_SPECIFIC_HEAT,
    THIN_SURFACE_LAYER,
    URBAN_BOTTOM_INDEX,
    URBAN_ROOF,
    URBAN_SHADEWALL,
    URBAN_SUNWALL,
    GroundHeatFluxInputs,
    GroundHeatFluxPatchState,
    GroundHeatFluxResult,
    SoilHeatDiffusionInputs,
    SoilHeatDiffusionResult,
    SoilSnowPhaseChangeInputs,
    SoilSnowPhaseChangeResult,
    SoilTemperatureFinalDiagnosticsInputs,
    SoilTemperatureFinalDiagnosticsResult,
    SoilTemperatureLinearSolveInputs,
    SoilTemperatureLinearSolveResult,
    SoilTemperatureMatrixInputs,
    SoilTemperatureMatrixResult,
    SoilTemperaturePostSolveFluxInputs,
    SoilTemperaturePostSolveFluxResult,
    SoilTemperatureRHSInputs,
    SoilTemperatureRHSResult,
    SoilTemperatureSolutionUpdateInputs,
    SoilTemperatureSolutionUpdateResult,
    SoilThermalPropertyInputs,
    SoilThermalPropertyResult,
    StandingWaterPhaseChangeInputs,
    StandingWaterPhaseChangeResult,
    compute_ground_heat_flux_and_derivative,
    compute_heat_diffusion_flux_and_factor,
    finalize_soil_temperature_diagnostics,
    phase_change_soil_snow_layers,
    phase_change_standing_surface_water,
    recompute_soil_temperature_interface_fluxes,
    set_soil_temperature_matrix,
    set_soil_temperature_rhs,
    soil_thermal_properties,
    solve_soil_temperature_banded_system,
    update_soil_temperature_from_solution,
)
from clm_jax.physics.urban_building_temperature import (
    UrbanBuildingTemperatureInputs,
    UrbanBuildingTemperatureResult,
    building_temperature,
)


class UrbanConstructionColumnMap(NamedTuple):

    roof_column: jnp.ndarray
    sunwall_column: jnp.ndarray
    shadewall_column: jnp.ndarray


class ColumnBuildingTemperatures(NamedTuple):

    building_temperature: jnp.ndarray
    roof_inner_temperature: jnp.ndarray
    sunwall_inner_temperature: jnp.ndarray
    shadewall_inner_temperature: jnp.ndarray


class SoilTemperatureActiveInputs(NamedTuple):

    topology: SubgridTopology
    urban_columns: UrbanConstructionColumnMap
    thermal_properties: SoilThermalPropertyInputs
    ground_heat_flux: GroundHeatFluxInputs
    ground_heat_flux_patch_state: GroundHeatFluxPatchState
    heat_diffusion_template: SoilHeatDiffusionInputs
    rhs_template: SoilTemperatureRHSInputs
    matrix_template: SoilTemperatureMatrixInputs
    linear_solve_template: SoilTemperatureLinearSolveInputs
    solution_update_template: SoilTemperatureSolutionUpdateInputs
    postsolve_flux_template: SoilTemperaturePostSolveFluxInputs
    standing_water_phase_change_template: StandingWaterPhaseChangeInputs
    soil_snow_phase_change_template: SoilSnowPhaseChangeInputs
    building_temperature_template: UrbanBuildingTemperatureInputs
    final_diagnostics_template: SoilTemperatureFinalDiagnosticsInputs


class SoilTemperatureColumnResult(NamedTuple):

    thermal_properties: SoilThermalPropertyResult
    ground_heat_flux: GroundHeatFluxResult
    heat_diffusion: SoilHeatDiffusionResult
    standing_water_heat_capacity: jnp.ndarray
    standing_water_thickness: jnp.ndarray
    rhs: SoilTemperatureRHSResult
    matrix: SoilTemperatureMatrixResult
    linear_solve: SoilTemperatureLinearSolveResult
    solution_update: SoilTemperatureSolutionUpdateResult
    postsolve_flux: SoilTemperaturePostSolveFluxResult
    standing_water_phase_change: StandingWaterPhaseChangeResult
    soil_snow_phase_change: SoilSnowPhaseChangeResult
    final_diagnostics: SoilTemperatureFinalDiagnosticsResult


class SoilTemperatureRuntimeColumnResult(NamedTuple):

    thermal_properties: SoilThermalPropertyResult
    ground_heat_flux: GroundHeatFluxResult
    heat_diffusion: SoilHeatDiffusionResult
    standing_water_heat_capacity: jnp.ndarray
    postsolve_flux: SoilTemperaturePostSolveFluxResult
    standing_water_phase_change: StandingWaterPhaseChangeResult
    soil_snow_phase_change: SoilSnowPhaseChangeResult
    final_diagnostics: SoilTemperatureFinalDiagnosticsResult


class SoilTemperatureActiveResult(NamedTuple):

    thermal_properties: SoilThermalPropertyResult
    ground_heat_flux: GroundHeatFluxResult
    heat_diffusion: SoilHeatDiffusionResult
    standing_water_heat_capacity: jnp.ndarray
    standing_water_thickness: jnp.ndarray
    rhs: SoilTemperatureRHSResult
    matrix: SoilTemperatureMatrixResult
    linear_solve: SoilTemperatureLinearSolveResult
    solution_update: SoilTemperatureSolutionUpdateResult
    postsolve_flux: SoilTemperaturePostSolveFluxResult
    standing_water_phase_change: StandingWaterPhaseChangeResult
    soil_snow_phase_change: SoilSnowPhaseChangeResult
    building_temperature: UrbanBuildingTemperatureResult
    final_diagnostics: SoilTemperatureFinalDiagnosticsResult


class SoilTemperatureRuntimeResult(NamedTuple):

    thermal_properties: SoilThermalPropertyResult
    ground_heat_flux: GroundHeatFluxResult
    heat_diffusion: SoilHeatDiffusionResult
    standing_water_heat_capacity: jnp.ndarray
    postsolve_flux: SoilTemperaturePostSolveFluxResult
    standing_water_phase_change: StandingWaterPhaseChangeResult
    soil_snow_phase_change: SoilSnowPhaseChangeResult
    building_temperature: UrbanBuildingTemperatureResult
    final_diagnostics: SoilTemperatureFinalDiagnosticsResult


_URBAN_CONSTRUCTION_TYPES = (
    ("roof_column", URBAN_ROOF),
    ("sunwall_column", URBAN_SUNWALL),
    ("shadewall_column", URBAN_SHADEWALL),
)


def make_urban_construction_column_map(
    topology: SubgridTopology,
) -> UrbanConstructionColumnMap:

    landunit_type = np.asarray(topology.landunit.landunit_type, dtype=np.int64)
    landunit_active = np.asarray(topology.landunit.active, dtype=bool)
    column_landunit = np.asarray(topology.column.landunit, dtype=np.int64)
    column_type = np.asarray(topology.column.column_type, dtype=np.int64)
    column_active = np.asarray(topology.column.active, dtype=bool)
    number_of_landunits = landunit_type.size
    number_of_columns = column_type.size

    if column_landunit.shape != (number_of_columns,):
        raise ValueError("column landunit and type arrays must have equal length")
    if number_of_landunits and number_of_columns == 0:
        raise ValueError("a landunit topology cannot be mapped without columns")

    urban = (
        landunit_active
        & (landunit_type >= LANDUNIT_URBAN_MIN)
        & (landunit_type <= LANDUNIT_URBAN_MAX)
    )
    urban_landunits = np.flatnonzero(urban)
    mapped: dict[str, jnp.ndarray] = {}
    for field_name, selected_type in _URBAN_CONSTRUCTION_TYPES:
        values = np.zeros(number_of_landunits, dtype=np.int32)
        for landunit in urban_landunits:
            matches = np.flatnonzero(
                column_active & (column_landunit == landunit) & (column_type == selected_type)
            )
            if matches.size != 1:
                raise ValueError(
                    f"active urban landunit {landunit} requires exactly one "
                    f"type-{selected_type} construction column; found {matches.size}"
                )
            values[landunit] = np.int32(matches[0])
        mapped[field_name] = jnp.asarray(values, dtype=jnp.int32)
    return UrbanConstructionColumnMap(**mapped)


def _validate_composition_shapes(inputs: SoilTemperatureActiveInputs) -> None:
    number_of_columns = inputs.thermal_properties.active_nonlake.shape[0]
    number_of_landunits = inputs.building_temperature_template.active_urban.shape[0]
    if inputs.topology.column.active.shape != (number_of_columns,):
        raise ValueError("topology and thermal-property column counts differ")
    if inputs.topology.landunit.active.shape != (number_of_landunits,):
        raise ValueError("topology and building-temperature landunit counts differ")
    for name in UrbanConstructionColumnMap._fields:
        if getattr(inputs.urban_columns, name).shape != (number_of_landunits,):
            raise ValueError(f"urban_columns.{name} must have shape {(number_of_landunits,)}")


def _surface_water_thermal_properties(
    active_nonlake: jnp.ndarray,
    standing_water: jnp.ndarray,
    standing_water_fraction: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    active = jnp.asarray(active_nonlake, dtype=jnp.bool_)
    water = jnp.asarray(standing_water, dtype=jnp.float64)
    fraction = jnp.asarray(standing_water_fraction, dtype=jnp.float64)
    resolved = (water > THIN_SURFACE_LAYER) & (fraction > THIN_SURFACE_LAYER)
    safe_fraction = jnp.where(resolved, fraction, jnp.float64(1.0))
    heat_capacity = jnp.maximum(
        jnp.float64(THIN_SURFACE_LAYER),
        fortran_divide(
            fortran_multiply(jnp.float64(LIQUID_WATER_SPECIFIC_HEAT), water),
            safe_fraction,
        ),
    )
    depth_scale = jnp.float64(0.0010000000474974513)
    thickness = jnp.maximum(
        jnp.float64(THIN_SURFACE_LAYER),
        fortran_divide(fortran_multiply(depth_scale, water), safe_fraction),
    )
    heat_capacity = jnp.where(resolved, heat_capacity, jnp.float64(THIN_SURFACE_LAYER))
    thickness = jnp.where(resolved, thickness, jnp.float64(THIN_SURFACE_LAYER))
    return (
        jnp.where(active, heat_capacity, jnp.float64(0.0)),
        jnp.where(active, thickness, jnp.float64(0.0)),
    )


def _landunit_to_column(
    topology: SubgridTopology,
    landunit_values: jnp.ndarray,
) -> jnp.ndarray:
    return landunit_values[jnp.asarray(topology.column.landunit, dtype=jnp.int32)]


def _mapped_construction_value(
    values: jnp.ndarray,
    column_indices: jnp.ndarray,
    active_urban: jnp.ndarray,
    inactive_value: jnp.ndarray,
) -> jnp.ndarray:
    safe_indices = jnp.clip(
        jnp.asarray(column_indices, dtype=jnp.int32),
        0,
        values.shape[0] - 1,
    )
    gathered = values[safe_indices]
    return jnp.where(active_urban, gathered, inactive_value)


def assemble_coupled_building_temperature_inputs(
    inputs: SoilTemperatureActiveInputs,
    interface_thermal_conductivity: jnp.ndarray,
    post_phase_temperature: jnp.ndarray,
) -> UrbanBuildingTemperatureInputs:
    template = inputs.building_temperature_template
    active = jnp.asarray(template.active_urban, dtype=jnp.bool_)
    previous_temperature = inputs.postsolve_flux_template.previous_soil_snow_temperature
    geometry = inputs.thermal_properties
    bottom_interface_index = URBAN_BOTTOM_INDEX + 1

    def gather(
        values: jnp.ndarray,
        column_indices: jnp.ndarray,
        inactive_value: jnp.ndarray,
    ) -> jnp.ndarray:
        return _mapped_construction_value(
            values,
            column_indices,
            active,
            inactive_value,
        )

    roof = inputs.urban_columns.roof_column
    sunwall = inputs.urban_columns.sunwall_column
    shadewall = inputs.urban_columns.shadewall_column
    return template._replace(
        previous_roof_boundary_temperature=gather(
            previous_temperature[:, URBAN_BOTTOM_INDEX],
            roof,
            template.previous_roof_boundary_temperature,
        ),
        previous_sunwall_boundary_temperature=gather(
            previous_temperature[:, URBAN_BOTTOM_INDEX],
            sunwall,
            template.previous_sunwall_boundary_temperature,
        ),
        previous_shadewall_boundary_temperature=gather(
            previous_temperature[:, URBAN_BOTTOM_INDEX],
            shadewall,
            template.previous_shadewall_boundary_temperature,
        ),
        roof_boundary_temperature=gather(
            post_phase_temperature[:, URBAN_BOTTOM_INDEX],
            roof,
            template.roof_boundary_temperature,
        ),
        sunwall_boundary_temperature=gather(
            post_phase_temperature[:, URBAN_BOTTOM_INDEX],
            sunwall,
            template.sunwall_boundary_temperature,
        ),
        shadewall_boundary_temperature=gather(
            post_phase_temperature[:, URBAN_BOTTOM_INDEX],
            shadewall,
            template.shadewall_boundary_temperature,
        ),
        roof_node_depth=gather(
            geometry.node_depth[:, URBAN_BOTTOM_INDEX],
            roof,
            template.roof_node_depth,
        ),
        sunwall_node_depth=gather(
            geometry.node_depth[:, URBAN_BOTTOM_INDEX],
            sunwall,
            template.sunwall_node_depth,
        ),
        shadewall_node_depth=gather(
            geometry.node_depth[:, URBAN_BOTTOM_INDEX],
            shadewall,
            template.shadewall_node_depth,
        ),
        roof_interface_depth=gather(
            geometry.interface_depth[:, bottom_interface_index],
            roof,
            template.roof_interface_depth,
        ),
        sunwall_interface_depth=gather(
            geometry.interface_depth[:, bottom_interface_index],
            sunwall,
            template.sunwall_interface_depth,
        ),
        shadewall_interface_depth=gather(
            geometry.interface_depth[:, bottom_interface_index],
            shadewall,
            template.shadewall_interface_depth,
        ),
        roof_thermal_conductivity=gather(
            interface_thermal_conductivity[:, URBAN_BOTTOM_INDEX],
            roof,
            template.roof_thermal_conductivity,
        ),
        sunwall_thermal_conductivity=gather(
            interface_thermal_conductivity[:, URBAN_BOTTOM_INDEX],
            sunwall,
            template.sunwall_thermal_conductivity,
        ),
        shadewall_thermal_conductivity=gather(
            interface_thermal_conductivity[:, URBAN_BOTTOM_INDEX],
            shadewall,
            template.shadewall_thermal_conductivity,
        ),
    )


def soil_temperature_column_case(
    inputs: SoilTemperatureActiveInputs,
    *,
    time_step_seconds: float = 1800.0,
    maximum_patches_per_column: int = 79,
) -> SoilTemperatureColumnResult:

    _validate_composition_shapes(inputs)
    if time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")

    ground, building, landunit_type = _soil_temperature_column_boundary(
        inputs, maximum_patches_per_column,
    )
    return _soil_temperature_column_transition(
        inputs, ground, building, landunit_type, time_step_seconds=time_step_seconds,
    )


def _soil_temperature_column_boundary(inputs, maximum_patches_per_column):

    state = inputs.thermal_properties
    water_temperature = inputs.solution_update_template.previous_standing_water_temperature
    ground_template = inputs.ground_heat_flux
    ground = compute_ground_heat_flux_and_derivative(
        inputs.topology,
        ground_template._replace(
            snow_layer_count_column=state.snow_layer_count,
            effective_snow_fraction_column=state.effective_snow_fraction,
            surface_water_temperature_column=water_temperature,
            ground_temperature_column=(
                inputs.final_diagnostics_template.previous_ground_temperature
            ),
            soil_snow_temperature_column=state.soil_snow_temperature,
        ),
        inputs.ground_heat_flux_patch_state,
        maximum_patches_per_column=maximum_patches_per_column,
    )
    building = ColumnBuildingTemperatures(*(
        _landunit_to_column(inputs.topology, getattr(inputs.building_temperature_template, name))
        for name in ColumnBuildingTemperatures._fields
    ))
    return ground, building, inputs.topology.column.landunit_type


def _soil_temperature_column_transition(
    inputs, ground, building, landunit_type, *, time_step_seconds=1800.,
):

    state = inputs.thermal_properties
    thermal = soil_thermal_properties(state)
    water_temperature = inputs.solution_update_template.previous_standing_water_temperature
    diffusion = compute_heat_diffusion_flux_and_factor(
        inputs.heat_diffusion_template._replace(
            active_nonlake=state.active_nonlake,
            snow_layer_count=state.snow_layer_count,
            column_type=state.column_type,
            layer_thickness=state.layer_thickness,
            node_depth=state.node_depth,
            interface_depth=state.interface_depth,
            interface_thermal_conductivity=thermal.interface_thermal_conductivity,
            layer_heat_capacity=thermal.layer_heat_capacity,
            soil_snow_temperature=state.soil_snow_temperature,
            building_temperature=building.building_temperature,
            roof_inner_temperature=building.roof_inner_temperature,
            sunwall_inner_temperature=building.sunwall_inner_temperature,
            shadewall_inner_temperature=building.shadewall_inner_temperature,
        ),
        time_step_seconds=time_step_seconds,
    )

    water_fraction = inputs.rhs_template.standing_water_fraction
    water_heat_capacity, water_thickness = _surface_water_thermal_properties(
        state.active_nonlake,
        state.surface_water,
        water_fraction,
    )
    rhs = set_soil_temperature_rhs(
        inputs.rhs_template._replace(
            active_nonlake=state.active_nonlake,
            snow_layer_count=state.snow_layer_count,
            landunit_type=state.landunit_type,
            column_type=state.column_type,
            standing_water_heat_flux=ground.standing_water_heat_flux_column,
            top_snow_heat_flux=ground.top_snow_heat_flux_column,
            soil_heat_flux=ground.soil_heat_flux_column,
            top_layer_heat_flux=ground.top_layer_heat_flux_column,
            heat_flux_derivative=ground.heat_flux_derivative_column,
            absorbed_ground_layers=ground.absorbed_ground_layers_column,
            layer_thermal_conductivity=thermal.interface_thermal_conductivity,
            standing_water_thermal_conductivity=(thermal.surface_water_interface_conductivity),
            timestep_over_heat_capacity=diffusion.timestep_over_heat_capacity,
            interface_heat_flux=diffusion.interface_heat_flux,
            standing_water_heat_capacity=water_heat_capacity,
            standing_water_thickness=water_thickness,
            node_depth=state.node_depth,
            soil_snow_temperature=state.soil_snow_temperature,
            standing_water_temperature=water_temperature,
            effective_snow_fraction=state.effective_snow_fraction,
        ),
        time_step_seconds=time_step_seconds,
    )

    matrix = set_soil_temperature_matrix(
        inputs.matrix_template._replace(
            active_nonlake=state.active_nonlake,
            snow_layer_count=state.snow_layer_count,
            landunit_type=state.landunit_type,
            column_type=state.column_type,
            heat_flux_derivative=ground.heat_flux_derivative_column,
            layer_thermal_conductivity=thermal.interface_thermal_conductivity,
            standing_water_thermal_conductivity=(thermal.surface_water_interface_conductivity),
            timestep_over_heat_capacity=diffusion.timestep_over_heat_capacity,
            standing_water_heat_capacity=water_heat_capacity,
            standing_water_thickness=water_thickness,
            node_depth=state.node_depth,
            interface_depth=state.interface_depth,
            standing_water_fraction=water_fraction,
            effective_snow_fraction=state.effective_snow_fraction,
        ),
        time_step_seconds=time_step_seconds,
    )

    linear_solve = solve_soil_temperature_banded_system(
        inputs.linear_solve_template._replace(
            active_nonlake=state.active_nonlake,
            snow_layer_count=state.snow_layer_count,
            column_type=state.column_type,
            band_matrix=matrix.band_matrix,
            right_hand_side=rhs.right_hand_side,
        )
    )
    solution_update = update_soil_temperature_from_solution(
        inputs.solution_update_template._replace(
            active_nonlake=state.active_nonlake,
            snow_layer_count=state.snow_layer_count,
            column_type=state.column_type,
            standing_water_fraction=water_fraction,
            previous_soil_snow_temperature=state.soil_snow_temperature,
            solution_vector=linear_solve.solution_vector,
        )
    )

    landunit_type = jnp.asarray(landunit_type, dtype=jnp.int32)
    urban_column = (landunit_type >= LANDUNIT_URBAN_MIN) & (landunit_type <= LANDUNIT_URBAN_MAX)
    postsolve = recompute_soil_temperature_interface_fluxes(
        inputs.postsolve_flux_template._replace(
            active_nonlake=state.active_nonlake,
            snow_layer_count=state.snow_layer_count,
            urban_landunit=urban_column,
            column_type=state.column_type,
            node_depth=state.node_depth,
            interface_depth=state.interface_depth,
            interface_thermal_conductivity=thermal.interface_thermal_conductivity,
            previous_interface_heat_flux=diffusion.interface_heat_flux,
            soil_snow_temperature=solution_update.soil_snow_temperature,
            building_temperature=building.building_temperature,
            roof_inner_temperature=building.roof_inner_temperature,
            sunwall_inner_temperature=building.sunwall_inner_temperature,
            shadewall_inner_temperature=building.shadewall_inner_temperature,
        ),
        simple_building_temperature=False,
    )

    standing_phase_template = inputs.standing_water_phase_change_template
    standing_phase = phase_change_standing_surface_water(
        standing_phase_template._replace(
            active_nonlake=state.active_nonlake,
            snow_layer_count=state.snow_layer_count,
            surface_flux_temperature_derivative=ground.heat_flux_derivative_column,
            effective_snow_fraction=state.effective_snow_fraction,
            standing_water_fraction=water_fraction,
            snow_water=state.snow_water,
            standing_water=state.surface_water,
            ice_water=state.ice_water,
            timestep_over_heat_capacity=diffusion.timestep_over_heat_capacity,
            standing_water_heat_capacity=water_heat_capacity,
            soil_snow_temperature=solution_update.soil_snow_temperature,
            standing_water_temperature=solution_update.standing_water_temperature,
        ),
        time_step_seconds=time_step_seconds,
    )

    soil_phase_template = inputs.soil_snow_phase_change_template
    soil_phase = phase_change_soil_snow_layers(
        soil_phase_template._replace(
            active_nonlake=state.active_nonlake,
            snow_layer_count=state.snow_layer_count,
            landunit_type=state.landunit_type,
            column_type=state.column_type,
            layer_thickness=state.layer_thickness,
            surface_flux_temperature_derivative=ground.heat_flux_derivative_column,
            effective_snow_fraction=state.effective_snow_fraction,
            standing_water_fraction=water_fraction,
            snow_depth=standing_phase.snow_depth,
            snow_water=standing_phase.snow_water,
            liquid_water=state.liquid_water,
            ice_water=standing_phase.ice_water,
            timestep_over_heat_capacity=diffusion.timestep_over_heat_capacity,
            soil_snow_temperature=standing_phase.soil_snow_temperature,
        ),
        time_step_seconds=time_step_seconds,
    )

    final = finalize_soil_temperature_diagnostics(
        inputs.final_diagnostics_template._replace(
            active_nonlake=state.active_nonlake,
            snow_layer_count=state.snow_layer_count,
            landunit_type=state.landunit_type,
            effective_snow_fraction=state.effective_snow_fraction,
            standing_water_fraction=water_fraction,
            soil_snow_temperature=soil_phase.soil_snow_temperature,
            standing_water_temperature=standing_phase.standing_water_temperature,
            interface_heat_flux=postsolve.interface_heat_flux,
            updated_interface_heat_flux=postsolve.updated_interface_heat_flux,
        )
    )

    return SoilTemperatureColumnResult(
        thermal_properties=thermal,
        ground_heat_flux=ground,
        heat_diffusion=diffusion,
        standing_water_heat_capacity=water_heat_capacity,
        standing_water_thickness=water_thickness,
        rhs=rhs,
        matrix=matrix,
        linear_solve=linear_solve,
        solution_update=solution_update,
        postsolve_flux=postsolve,
        standing_water_phase_change=standing_phase,
        soil_snow_phase_change=soil_phase,
        final_diagnostics=final,
    )


def soil_temperature_runtime_column_case(
    inputs: SoilTemperatureActiveInputs,
    *,
    time_step_seconds: float = 1800.0,
    maximum_patches_per_column: int = 79,
) -> SoilTemperatureRuntimeColumnResult:

    from clm_jax.physics.solver_scalarization import _native_tile_trace

    if jax.default_backend() == "gpu" and not _native_tile_trace.get():
        from clm_jax.physics.soil_temperature_gpu import soil_temperature_columns

        _validate_composition_shapes(inputs)
        if time_step_seconds <= 0.0:
            raise ValueError("time_step_seconds must be positive")
        return soil_temperature_columns(
            inputs, float(time_step_seconds), maximum_patches_per_column,
        )
    return _soil_temperature_runtime_column_reference(
        inputs, time_step_seconds=time_step_seconds,
        maximum_patches_per_column=maximum_patches_per_column,
    )


def _soil_temperature_runtime_column_reference(
    inputs, *, time_step_seconds=1800., maximum_patches_per_column=79,
):

    complete = soil_temperature_column_case(
        inputs,
        time_step_seconds=time_step_seconds,
        maximum_patches_per_column=maximum_patches_per_column,
    )
    return SoilTemperatureRuntimeColumnResult(
        thermal_properties=complete.thermal_properties,
        ground_heat_flux=complete.ground_heat_flux,
        heat_diffusion=complete.heat_diffusion,
        standing_water_heat_capacity=complete.standing_water_heat_capacity,
        postsolve_flux=complete.postsolve_flux,
        standing_water_phase_change=complete.standing_water_phase_change,
        soil_snow_phase_change=complete.soil_snow_phase_change,
        final_diagnostics=complete.final_diagnostics,
    )


def soil_temperature_active_case(
    inputs: SoilTemperatureActiveInputs,
    *,
    time_step_seconds: float = 1800.0,
    maximum_patches_per_column: int = 79,
    heating_air_conditioning_enabled: bool = True,
) -> SoilTemperatureActiveResult:

    columns = soil_temperature_column_case(
        inputs,
        time_step_seconds=time_step_seconds,
        maximum_patches_per_column=maximum_patches_per_column,
    )
    coupled_building = assemble_coupled_building_temperature_inputs(
        inputs,
        columns.thermal_properties.interface_thermal_conductivity,
        columns.soil_snow_phase_change.soil_snow_temperature,
    )
    building = building_temperature(
        coupled_building,
        time_step_seconds=time_step_seconds,
        heating_air_conditioning_enabled=heating_air_conditioning_enabled,
    )
    return SoilTemperatureActiveResult(
        thermal_properties=columns.thermal_properties,
        ground_heat_flux=columns.ground_heat_flux,
        heat_diffusion=columns.heat_diffusion,
        standing_water_heat_capacity=columns.standing_water_heat_capacity,
        standing_water_thickness=columns.standing_water_thickness,
        rhs=columns.rhs,
        matrix=columns.matrix,
        linear_solve=columns.linear_solve,
        solution_update=columns.solution_update,
        postsolve_flux=columns.postsolve_flux,
        standing_water_phase_change=columns.standing_water_phase_change,
        soil_snow_phase_change=columns.soil_snow_phase_change,
        building_temperature=building,
        final_diagnostics=columns.final_diagnostics,
    )
