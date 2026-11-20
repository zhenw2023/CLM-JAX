
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.forcing.datm import DatmForcing
from clm_jax.io.canopy_fluxes_inputs import (
    CanopyFluxesFastInputs,
    CanopyFluxesPftParameters,
    assemble_canopy_fluxes_fast_inputs,
    assemble_canopy_fluxes_luna_inputs,
    synchronize_canopy_fluxes_water_state,
)
from clm_jax.io.canopy_hydraulic_inputs import (
    CanopyBiochemicalPftParameters,
    CanopyHydraulicNativeInputs,
    assemble_canopy_hydraulic_coupling,
)
from clm_jax.io.luna import LunaRestartData
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.atmospheric_driver import AtmosphericDriverResult
from clm_jax.physics.bare_ground_driver import BareGroundDriverResult
from clm_jax.physics.canopy_driver import CanopyDriverResult
from clm_jax.physics.canopy_fluxes import (
    CanopyEnergyInitialState,
    CanopyEnergyParameters,
    CanopyFastHumanStressState,
    CanopyFluxesActiveParameters,
    CanopyFluxesActiveResult,
    CanopyFluxesActiveState,
    CanopyFluxesLunaInputs,
    CanopyHydraulicCouplingInputs,
    canopy_fluxes_active_case,
    canopy_iteration_work,
)
from clm_jax.physics.canopy_temperature import CanopyTemperatureParameters
from clm_jax.physics.canopy_temperature_driver import CanopyTemperatureDriverResult
from clm_jax.physics.friction_velocity import FrictionVelocityState
from clm_jax.physics.luna import LunaCapacityParameters
from clm_jax.physics.plant_hydraulics import (
    PlantHydraulicParameters,
    PlantHydraulicPatchState,
)
from clm_jax.physics.radiation_sequence import RadiationSequenceResult
from clm_jax.trajectory import (
    NoleapModelClock,
    model_time_flags,
    noleap_calendar_day_from_clock,
)


class CanopyFluxDriverInputs(NamedTuple):

    pft: CanopyFluxesPftParameters
    hydraulic: CanopyHydraulicNativeInputs
    plant_parameters: PlantHydraulicParameters
    biochemical_parameters: CanopyBiochemicalPftParameters
    temperature_parameters: CanopyTemperatureParameters
    luna_template: LunaRestartData


class CanopyFluxDriverParameters(NamedTuple):

    maximum_stable_zeta: float | jnp.ndarray
    use_undercanopy_stability: bool | jnp.ndarray
    use_hydraulic_stress: bool | jnp.ndarray
    update_snow_canopy_water: bool | jnp.ndarray
    luna_capacity: LunaCapacityParameters


class CanopyFluxDriverState(NamedTuple):

    active: CanopyFluxesActiveState
    plant: PlantHydraulicPatchState
    vegetation_temperature_patch: jnp.ndarray
    root_zone_wetness_patch: jnp.ndarray
    friction_velocity: FrictionVelocityState
    iteration_work_patch: jnp.ndarray | None = None


class CanopyFluxDriverResult(NamedTuple):

    fast_inputs: CanopyFluxesFastInputs
    energy_initial: CanopyEnergyInitialState
    hydraulic_coupling: CanopyHydraulicCouplingInputs
    luna_inputs: CanopyFluxesLunaInputs
    active: CanopyFluxesActiveResult
    state: CanopyFluxDriverState


class CanopyFluxVOCFields(NamedTuple):

    active_patch: jnp.ndarray
    vegetation_temperature_patch: jnp.ndarray
    intracellular_co2_patch_leaf: jnp.ndarray


class CanopyFluxSoilTemperatureFields(NamedTuple):

    snow_evaporation_patch: jnp.ndarray
    soil_evaporation_patch: jnp.ndarray
    surface_water_evaporation_patch: jnp.ndarray
    ground_evaporation_patch: jnp.ndarray
    vegetation_transpiration_patch: jnp.ndarray
    vegetation_evaporation_patch: jnp.ndarray
    vegetation_sensible_heat_patch: jnp.ndarray
    soil_root_conductance_patch_level: jnp.ndarray
    ground_methane_conductance_patch: jnp.ndarray
    ground_energy_derivative_patch: jnp.ndarray
    downward_longwave_below_canopy_patch: jnp.ndarray
    sensible_heat_snow_patch: jnp.ndarray
    sensible_heat_soil_patch: jnp.ndarray
    sensible_heat_surface_water_patch: jnp.ndarray
    sensible_heat_ground_patch: jnp.ndarray


class CanopyFluxDustFields(NamedTuple):

    active_patch: jnp.ndarray
    momentum_resistance_patch: jnp.ndarray


class CanopyFluxEcosystemFields(NamedTuple):

    vegetation_exposed_patch: jnp.ndarray
    sunlit_leaf_area_patch: jnp.ndarray
    shaded_leaf_area_patch: jnp.ndarray
    total_leaf_area_patch: jnp.ndarray
    sunlit_leaf_respiration_patch: jnp.ndarray
    shaded_leaf_respiration_patch: jnp.ndarray
    sunlit_photosynthesis_patch: jnp.ndarray
    shaded_photosynthesis_patch: jnp.ndarray
    transpiration_wetness_patch: jnp.ndarray
    root_zone_wetness_patch: jnp.ndarray


class CanopyFluxRuntimeResult(NamedTuple):

    state: CanopyFluxDriverState
    voc: CanopyFluxVOCFields
    soil_temperature: CanopyFluxSoilTemperatureFields
    dust: CanopyFluxDustFields
    ecosystem: CanopyFluxEcosystemFields


def canopy_flux_voc_fields(result: CanopyFluxDriverResult) -> CanopyFluxVOCFields:

    return CanopyFluxVOCFields(
        active_patch=result.fast_inputs.energy.active_patch,
        vegetation_temperature_patch=result.state.vegetation_temperature_patch,
        intracellular_co2_patch_leaf=(
            result.active.energy_hydraulics.hydraulic.intracellular_co2_patch_leaf
        ),
    )


def canopy_flux_soil_temperature_fields(
    result: CanopyFluxDriverResult,
) -> CanopyFluxSoilTemperatureFields:

    post = result.active.state.postprocess
    iteration = result.active.energy_hydraulics.energy.iteration
    return CanopyFluxSoilTemperatureFields(
        snow_evaporation_patch=post.snow_evaporation_patch,
        soil_evaporation_patch=post.soil_evaporation_patch,
        surface_water_evaporation_patch=post.surface_water_evaporation_patch,
        ground_evaporation_patch=post.ground_evaporation_patch,
        vegetation_transpiration_patch=iteration.vegetation_transpiration_patch,
        vegetation_evaporation_patch=iteration.vegetation_evaporation_patch,
        vegetation_sensible_heat_patch=(iteration.vegetation_sensible_heat_flux_patch),
        soil_root_conductance_patch_level=(
            result.hydraulic_coupling.plant_inputs.soil_root_conductance_level
        ),
        ground_methane_conductance_patch=(
            iteration.ground_methane_conductance_patch
        ),
        ground_energy_derivative_patch=post.ground_energy_derivative_patch,
        downward_longwave_below_canopy_patch=(post.downward_longwave_below_canopy_patch),
        sensible_heat_snow_patch=post.sensible_heat_snow_patch,
        sensible_heat_soil_patch=post.sensible_heat_soil_patch,
        sensible_heat_surface_water_patch=(post.sensible_heat_surface_water_patch),
        sensible_heat_ground_patch=post.sensible_heat_ground_patch,
    )


def canopy_flux_ecosystem_fields(
    result: CanopyFluxDriverResult,
) -> CanopyFluxEcosystemFields:

    energy = result.fast_inputs.energy
    hydraulic = result.active.energy_hydraulics.hydraulic
    respiration = hydraulic.leaf_maintenance_respiration_patch_leaf
    photosynthesis = hydraulic.gross_photosynthesis_patch_leaf
    return CanopyFluxEcosystemFields(
        vegetation_exposed_patch=jnp.asarray(
            energy.vegetation_fraction_without_snow_patch,
            dtype=jnp.int32,
        ),
        sunlit_leaf_area_patch=energy.sunlit_leaf_area_index_patch,
        shaded_leaf_area_patch=energy.shaded_leaf_area_index_patch,
        total_leaf_area_patch=(
            result.luna_inputs.capacity_inputs.total_leaf_area_patch
        ),
        sunlit_leaf_respiration_patch=respiration[:, 0],
        shaded_leaf_respiration_patch=respiration[:, 1],
        sunlit_photosynthesis_patch=photosynthesis[:, 0],
        shaded_photosynthesis_patch=photosynthesis[:, 1],
        transpiration_wetness_patch=(
            hydraulic.photosynthesis.transpiration_wetness_patch
        ),
        root_zone_wetness_patch=(
            result.state.root_zone_wetness_patch
        ),
    )


def canopy_flux_runtime_result(
    result: CanopyFluxDriverResult,
) -> CanopyFluxRuntimeResult:

    return CanopyFluxRuntimeResult(
        state=result.state,
        voc=canopy_flux_voc_fields(result),
        soil_temperature=canopy_flux_soil_temperature_fields(result),
        dust=CanopyFluxDustFields(
            active_patch=result.fast_inputs.energy.active_patch,
            momentum_resistance_patch=(
                result.active.energy_hydraulics.energy.iteration.momentum_resistance_patch
            ),
        ),
        ecosystem=canopy_flux_ecosystem_fields(result),
    )


class CanopyFluxDriverCall(NamedTuple):

    topology: SubgridTopology
    fast_inputs: CanopyFluxesFastInputs
    energy_initial: CanopyEnergyInitialState
    hydraulic_coupling: CanopyHydraulicCouplingInputs
    luna_inputs: CanopyFluxesLunaInputs
    synchronized_state: CanopyFluxesActiveState
    active_parameters: CanopyFluxesActiveParameters
    previous_state: CanopyFluxDriverState


def assemble_canopy_flux_energy_initial_state(
    state: CanopyFluxDriverState,
    temperature: CanopyTemperatureDriverResult,
) -> CanopyEnergyInitialState:

    patch_shape = temperature.state.patch.displacement_height_patch.shape
    values = (
        state.vegetation_temperature_patch,
        temperature.state.patch.displacement_height_patch,
        temperature.state.patch.vegetation_momentum_roughness_patch,
    )
    if any(jnp.shape(value) != patch_shape for value in values):
        raise ValueError("stage-260 energy state and stage-240 patch dimensions disagree")
    if any(jnp.shape(value) != patch_shape for value in state.friction_velocity):
        raise ValueError("stage-260 friction state must contain every patch")
    if (state.iteration_work_patch is not None
            and jnp.shape(state.iteration_work_patch) != patch_shape):
        raise ValueError("stage-260 scheduling hint must contain every patch")
    return CanopyEnergyInitialState(
        vegetation_temperature_patch=jnp.asarray(values[0], dtype=jnp.float64),
        displacement_height_patch=jnp.asarray(values[1], dtype=jnp.float64),
        vegetation_momentum_roughness_patch=jnp.asarray(values[2], dtype=jnp.float64),
        friction_velocity=state.friction_velocity,
        iteration_work_patch=state.iteration_work_patch,
    )


def synchronize_canopy_flux_driver_state(
    state: CanopyFluxDriverState,
    canopy: CanopyDriverResult,
    temperature: CanopyTemperatureDriverResult,
    bare_ground: BareGroundDriverResult,
) -> CanopyFluxDriverState:

    bare = bare_ground.state
    post = state.active.postprocess._replace(
        eastward_wind_stress_patch=bare.energy.eastward_wind_stress_patch,
        northward_wind_stress_patch=bare.energy.northward_wind_stress_patch,
        sensible_heat_ground_patch=bare.energy.sensible_heat_ground_patch,
        sensible_heat_snow_patch=bare.energy.sensible_heat_snow_patch,
        sensible_heat_soil_patch=bare.energy.sensible_heat_soil_patch,
        sensible_heat_surface_water_patch=(bare.energy.sensible_heat_surface_water_patch),
        ground_evaporation_patch=bare.water.ground_evaporation_patch,
        snow_evaporation_patch=bare.water.evaporation_snow_patch,
        soil_evaporation_patch=bare.water.evaporation_soil_patch,
        surface_water_evaporation_patch=(bare.water.evaporation_surface_water_patch),
        reference_temperature_2m_patch=bare.temperature.reference_temperature_patch,
        rural_reference_temperature_2m_patch=(bare.temperature.rural_reference_temperature_patch),
        reference_specific_humidity_2m_patch=(bare.water.reference_specific_humidity_patch),
        reference_relative_humidity_2m_patch=(bare.water.reference_relative_humidity_patch),
        rural_reference_relative_humidity_2m_patch=(
            bare.water.rural_reference_relative_humidity_patch
        ),
        human_stress=CanopyFastHumanStressState(*bare.human_index),
        downward_longwave_below_canopy_patch=(bare.energy.downward_longwave_below_canopy_patch),
        upward_longwave_above_canopy_patch=(bare.energy.upward_longwave_above_canopy_patch),
        ground_sensible_derivative_patch=(bare.energy.ground_sensible_derivative_patch),
        ground_latent_derivative_patch=(bare.energy.ground_latent_derivative_patch),
        ground_energy_derivative_patch=bare.energy.ground_energy_derivative_patch,
    )
    active = state.active._replace(postprocess=post)
    active = synchronize_canopy_fluxes_water_state(
        active,
        canopy.state.canopy_hydrology.canopy_water,
    )
    return state._replace(
        active=active,
        plant=state.plant._replace(
            vegetation_transpiration=(
                temperature.state.patch.vegetation_transpiration_patch
            ),
        ),
        vegetation_temperature_patch=(bare.temperature.vegetation_temperature_patch),
        friction_velocity=bare.friction_velocity,
    )


def _current_luna_restart(
    template: LunaRestartData,
    state: CanopyFluxDriverState,
) -> LunaRestartData:

    return template._replace(
        daily=state.active.luna_daily,
        ten_day=state.active.luna_ten_day,
        capacity=state.active.luna_capacity,
    )


def _current_hydraulic_inputs(
    inputs: CanopyHydraulicNativeInputs,
    state: CanopyFluxDriverState,
    clock: NoleapModelClock,
) -> CanopyHydraulicNativeInputs:

    capacity = state.active.luna_capacity
    return inputs._replace(
        luna_vcmax25_by_layer_patch=(capacity.maximum_carboxylation_rate_by_layer_patch),
        luna_jmax25_by_layer_patch=(capacity.maximum_electron_transport_rate_by_layer_patch),
        calendar_day=noleap_calendar_day_from_clock(clock),
    )


def assemble_canopy_flux_driver_call(
    topology: SubgridTopology,
    atmospheric: AtmosphericDriverResult,
    grid_forcing: DatmForcing,
    canopy: CanopyDriverResult,
    radiation: RadiationSequenceResult,
    temperature: CanopyTemperatureDriverResult,
    bare_ground: BareGroundDriverResult,
    clock: NoleapModelClock,
    inputs: CanopyFluxDriverInputs,
    parameters: CanopyFluxDriverParameters,
    state: CanopyFluxDriverState,
    root_vulnerability_power_base: jnp.ndarray | float = 2.0,
    solar_declination_radians: jnp.ndarray | None = None,
) -> CanopyFluxDriverCall:

    state = synchronize_canopy_flux_driver_state(
        state,
        canopy,
        temperature,
        bare_ground,
    )

    column_topography = atmospheric.state.column_topography
    current_topology = topology._replace(
        column=topology.column._replace(
            elevation=column_topography.elevation,
            column_type=column_topography.column_type,
        )
    )
    radiation_state = radiation.state.canopy_surface
    fast = assemble_canopy_fluxes_fast_inputs(
        current_topology,
        grid_forcing,
        atmospheric.column_forcing,
        temperature.inputs,
        temperature.state,
        canopy.state.canopy_hydrology,
        radiation_state.surface_radiation,
        radiation_state.canopy_sunshade,
        inputs.pft,
    )
    energy_initial = assemble_canopy_flux_energy_initial_state(state, temperature)
    hydraulic = assemble_canopy_hydraulic_coupling(
        current_topology,
        _current_hydraulic_inputs(inputs.hydraulic, state, clock),
        inputs.plant_parameters,
        inputs.biochemical_parameters,
        state.plant,
        fast.energy,
        energy_initial,
        inputs.temperature_parameters,
        temperature.inputs.soil_liquid_water_column_level,
        temperature.inputs.soil_ice_water_column_level,
        radiation_state.canopy_sunshade,
        grid_forcing.pressure,
        root_vulnerability_power_base=root_vulnerability_power_base,
        solar_declination_radians=solar_declination_radians,
    )
    luna = assemble_canopy_fluxes_luna_inputs(
        current_topology,
        noleap_calendar_day_from_clock(clock),
        radiation_state.canopy_sunshade,
        _current_luna_restart(inputs.luna_template, state),
        solar_declination_radians=solar_declination_radians,
    )
    return CanopyFluxDriverCall(
        topology=current_topology,
        fast_inputs=fast,
        energy_initial=energy_initial,
        hydraulic_coupling=hydraulic,
        luna_inputs=luna,
        synchronized_state=state.active,
        active_parameters=CanopyFluxesActiveParameters(
            energy=CanopyEnergyParameters(
                step_seconds=jnp.asarray(clock.step_seconds, dtype=jnp.float64),
                maximum_stable_zeta=jnp.asarray(
                    parameters.maximum_stable_zeta,
                    dtype=jnp.float64,
                ),
                use_undercanopy_stability=jnp.asarray(
                    parameters.use_undercanopy_stability,
                    dtype=jnp.bool_,
                ),
                use_hydraulic_stress=jnp.asarray(
                    parameters.use_hydraulic_stress,
                    dtype=jnp.bool_,
                ),
            ),
            luna_capacity=parameters.luna_capacity,
            is_end_day=model_time_flags(clock).end_current_day,
            update_snow_canopy_water=jnp.asarray(
                parameters.update_snow_canopy_water,
                dtype=jnp.bool_,
            ),
        ),
        previous_state=state,
    )


def retain_inactive_plant_state(active_patch, solved, previous) -> PlantHydraulicPatchState:
    active_patch = jnp.asarray(active_patch, dtype=jnp.bool_)
    return PlantHydraulicPatchState(
        water_potential_segment=jnp.where(
            active_patch[:, None], solved.water_potential_segment, previous.water_potential_segment,
        ),
        vegetation_transpiration=jnp.where(
            active_patch, solved.vegetation_transpiration, previous.vegetation_transpiration,
        ),
    )


def complete_canopy_flux_driver_call(
    call: CanopyFluxDriverCall,
    active: CanopyFluxesActiveResult,
) -> CanopyFluxDriverResult:

    solved = active.energy_hydraulics
    active_patch = jnp.asarray(call.fast_inputs.energy.active_patch, dtype=jnp.bool_)
    plant = retain_inactive_plant_state(
        active_patch, solved.hydraulic.plant_state, call.previous_state.plant,
    )
    solved = solved._replace(hydraulic=solved.hydraulic._replace(plant_state=plant))
    active = active._replace(energy_hydraulics=solved)
    updated_state = CanopyFluxDriverState(
        active=active.state,
        plant=plant,
        vegetation_temperature_patch=(solved.energy.iteration.vegetation_temperature_patch),
        root_zone_wetness_patch=jnp.where(
            active_patch,
            call.hydraulic_coupling.root_zone_wetness_patch,
            call.previous_state.root_zone_wetness_patch,
        ),
        friction_velocity=solved.energy.iteration.friction_velocity,
        iteration_work_patch=(
            None if call.previous_state.iteration_work_patch is None
            else canopy_iteration_work(solved)
        ),
    )
    return CanopyFluxDriverResult(
        fast_inputs=call.fast_inputs,
        energy_initial=call.energy_initial,
        hydraulic_coupling=call.hydraulic_coupling,
        luna_inputs=call.luna_inputs,
        active=active,
        state=updated_state,
    )


def canopy_flux_driver_step(
    topology: SubgridTopology,
    atmospheric: AtmosphericDriverResult,
    grid_forcing: DatmForcing,
    canopy: CanopyDriverResult,
    radiation: RadiationSequenceResult,
    temperature: CanopyTemperatureDriverResult,
    bare_ground: BareGroundDriverResult,
    clock: NoleapModelClock,
    inputs: CanopyFluxDriverInputs,
    parameters: CanopyFluxDriverParameters,
    state: CanopyFluxDriverState,
) -> CanopyFluxDriverResult:

    call = assemble_canopy_flux_driver_call(
        topology,
        atmospheric,
        grid_forcing,
        canopy,
        radiation,
        temperature,
        bare_ground,
        clock,
        inputs,
        parameters,
        state,
    )
    active = canopy_fluxes_active_case(
        call.topology,
        call.fast_inputs.energy,
        call.hydraulic_coupling,
        call.energy_initial,
        call.fast_inputs.postprocess,
        call.luna_inputs,
        call.synchronized_state,
        call.active_parameters,
    )
    return complete_canopy_flux_driver_call(call, active)


__all__ = [
    "CanopyFluxDriverInputs",
    "CanopyFluxDriverCall",
    "CanopyFluxDriverParameters",
    "CanopyFluxDriverResult",
    "CanopyFluxDriverState",
    "CanopyFluxDustFields",
    "CanopyFluxEcosystemFields",
    "CanopyFluxRuntimeResult",
    "CanopyFluxSoilTemperatureFields",
    "CanopyFluxVOCFields",
    "assemble_canopy_flux_energy_initial_state",
    "assemble_canopy_flux_driver_call",
    "canopy_flux_driver_step",
    "canopy_flux_ecosystem_fields",
    "canopy_flux_soil_temperature_fields",
    "canopy_flux_runtime_result",
    "canopy_flux_voc_fields",
    "complete_canopy_flux_driver_call",
    "synchronize_canopy_flux_driver_state",
]
