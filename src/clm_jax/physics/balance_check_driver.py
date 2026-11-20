
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.forcing.datm import DatmForcing
from clm_jax.io.restart import LANDUNIT_LAKE, SubgridTopology
from clm_jax.physics.balance_check import (
    DEFAULT_BALANCE_CHECK_PARAMETERS,
    BalanceCheckInputs,
    BalanceCheckParameters,
    BalanceCheckResult,
    BalanceCheckState,
    balance_check,
)
from clm_jax.physics.hydrology_drainage_driver import (
    HydrologyDrainageDriverResult,
)
from clm_jax.physics.hydrology_driver import HydrologyDriverResult
from clm_jax.physics.lake_hydrology_driver import LakeHydrologyDriverResult
from clm_jax.physics.physical_driver import (
    PhysicalDriverResult,
    PhysicalDriverRuntimeResult,
)
from clm_jax.physics.post_hydrology_no_drainage_driver import (
    PostHydrologyNoDrainageResult,
)
from clm_jax.physics.post_temperature_driver import PostTemperatureDriverResult
from clm_jax.physics.radiation_driver import datm_shortwave_bands


class BalanceCheckRadiationFields(NamedTuple):

    absorbed_shortwave_patch: jnp.ndarray
    reflected_shortwave_patch: jnp.ndarray
    absorbed_vegetation_shortwave_patch: jnp.ndarray
    absorbed_ground_shortwave_patch: jnp.ndarray


class BalanceCheckDriverResult(NamedTuple):

    inputs: BalanceCheckInputs
    balance: BalanceCheckResult


def balance_check_radiation_fields(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
) -> BalanceCheckRadiationFields:

    if physical.radiation is None:
        raise ValueError("physical radiation state was released before BalanceCheck projection")
    flux = physical.radiation.state.canopy_surface.surface_radiation.fluxes
    return BalanceCheckRadiationFields(
        absorbed_shortwave_patch=flux.absorbed_total_patch,
        reflected_shortwave_patch=flux.reflected_total_patch,
        absorbed_vegetation_shortwave_patch=flux.absorbed_vegetation_patch,
        absorbed_ground_shortwave_patch=flux.absorbed_ground_patch,
    )


def initial_balance_check_state(topology: SubgridTopology) -> BalanceCheckState:

    column = jnp.full(
        topology.column.gridcell.shape,
        jnp.float64(CLM_SPECIAL_VALUE),
        dtype=jnp.float64,
    )
    patch = jnp.full(
        topology.patch.gridcell.shape,
        jnp.float64(CLM_SPECIAL_VALUE),
        dtype=jnp.float64,
    )
    return BalanceCheckState(
        water_balance_error_column=column,
        snow_balance_error_column=column,
        snow_sources_column=column,
        snow_sinks_column=column,
        solar_balance_error_patch=patch,
        longwave_balance_error_patch=patch,
        surface_energy_balance_error_patch=patch,
        net_radiation_patch=patch,
    )


def assemble_balance_check_inputs(
    forcing: DatmForcing,
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    post_temperature: PostTemperatureDriverResult,
    hydrology: HydrologyDriverResult,
    post_hydrology: PostHydrologyNoDrainageResult,
    lake_hydrology: LakeHydrologyDriverResult,
    drainage: HydrologyDrainageDriverResult,
    radiation: BalanceCheckRadiationFields,
    *,
    steps_since_restart: int | jnp.ndarray,
) -> BalanceCheckInputs:

    direct, diffuse = datm_shortwave_bands(forcing)
    atmospheric = physical.atmospheric.column_forcing
    lake_state = lake_hydrology.hydrology.state
    water = lake_state.water
    water_flux = lake_state.water_flux
    lake_energy = lake_state.energy_flux
    drained = drainage.drainage.state
    energy = post_temperature.soil_fluxes.state.energy
    soil_temperature = physical.soil_temperature.temperature
    ground_patch = soil_temperature.ground_heat_flux.patch_state
    patch_is_lake = (
        physical.soil_temperature.inputs.topology.patch.landunit_type
        == jnp.int32(LANDUNIT_LAKE)
    )
    checked_absorbed_ground = jnp.where(
        patch_is_lake,
        radiation.absorbed_ground_shortwave_patch,
        ground_patch.absorbed_ground_balance_patch,
    )

    return BalanceCheckInputs(
        steps_since_restart=jnp.asarray(steps_since_restart, dtype=jnp.int32),
        snow_layer_count_column=lake_state.geometry.snow_layer_count_column,
        direct_shortwave_gridcell_band=direct,
        diffuse_shortwave_gridcell_band=diffuse,
        rainfall_flux_column=atmospheric.rain,
        snowfall_flux_column=atmospheric.snow,
        downward_longwave_column=atmospheric.longwave_down,
        snow_water_equivalent_column=water.snow_water_equivalent_column,
        old_snow_water_equivalent_column=(
            physical.atmospheric.driver_initialization.previous_snow_water_equivalent
        ),
        effective_snow_fraction_column=water.effective_snow_fraction_column,
        beginning_water_mass_column=drainage.inputs.beginning_water_mass_column,
        ending_water_mass_column=drained.ending_water_mass_column,
        flood_flux_column=water_flux.flood_flux_column,
        irrigation_flux_column=drainage.inputs.irrigation_flux_column,
        glacier_dynamic_water_flux_column=(
            post_hydrology.glacier_surface_mass_balance.state.dynamic_water_flux_column
        ),
        total_evaporation_flux_column=water_flux.total_evaporation_flux_column,
        surface_runoff_flux_column=drained.surface_runoff_flux_column,
        surface_water_runoff_flux_column=drained.surface_water_runoff_flux_column,
        wetland_glacier_runoff_flux_column=(
            drained.wetland_glacier_runoff_flux_column
        ),
        drainage_flux_column=drained.subsurface_drainage_flux_column,
        perched_drainage_flux_column=drained.perched_drainage_flux_column,
        snow_capping_ice_runoff_flux_column=(
            drained.snow_capping_ice_runoff_flux_column
        ),
        excess_soil_ice_runoff_flux_column=(
            drained.excess_soil_ice_runoff_flux_column
        ),
        snow_capping_discarded_liquid_flux_column=(
            water_flux.snow_capping_discarded_liquid_flux_column
        ),
        snow_capping_discarded_ice_flux_column=(
            water_flux.snow_capping_discarded_ice_flux_column
        ),
        ground_precipitation_flux_column=(
            water_flux.ground_precipitation_flux_column
        ),
        ground_snow_flux_column=water_flux.ground_snowfall_flux_column,
        ground_rain_flux_column=water_flux.ground_rainfall_flux_column,
        snow_dew_flux_column=water_flux.snow_dew_flux_column,
        ground_dew_flux_column=water_flux.ground_dew_flux_column,
        snow_sublimation_flux_column=water_flux.snow_sublimation_flux_column,
        ground_evaporation_flux_column=water_flux.ground_evaporation_flux_column,
        snow_drainage_flux_column=water_flux.snow_drainage_flux_column,
        snow_capping_ice_flux_column=water_flux.snow_capping_ice_flux_column,
        snow_capping_liquid_flux_column=water_flux.snow_capping_liquid_flux_column,
        snow_to_surface_water_flux_column=(
            hydrology.inputs.surface_hydrology.snow_to_surface_water_flux_column
        ),
        surface_water_to_ice_flux_column=(
            soil_temperature.standing_water_phase_change.water_to_ice_flux
        ),
        snow_layer_to_top_soil_flux_column=(
            water_flux.snow_layer_to_top_soil_flux_column
        ),
        absorbed_shortwave_patch=radiation.absorbed_shortwave_patch,
        reflected_shortwave_patch=radiation.reflected_shortwave_patch,
        absorbed_vegetation_shortwave_patch=(
            radiation.absorbed_vegetation_shortwave_patch
        ),
        absorbed_ground_shortwave_patch=radiation.absorbed_ground_shortwave_patch,
        checked_absorbed_ground_shortwave_patch=(
            checked_absorbed_ground
        ),
        outgoing_longwave_patch=energy.outgoing_longwave_patch,
        net_longwave_patch=energy.net_longwave_patch,
        sensible_heat_flux_patch=lake_energy.sensible_heat_total_patch,
        latent_heat_flux_patch=energy.total_latent_heat_patch,
        ground_heat_flux_patch=lake_energy.soil_heat_flux_ground_patch,
        waste_heat_flux_patch=ground_patch.waste_heat_patch,
        air_conditioning_heat_flux_patch=ground_patch.air_conditioner_heat_patch,
        traffic_heat_flux_patch=ground_patch.traffic_heat_patch,
        soil_energy_balance_error_column=energy.soil_energy_error_column,
    )


def balance_check_driver_step(
    forcing: DatmForcing,
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    post_temperature: PostTemperatureDriverResult,
    hydrology: HydrologyDriverResult,
    post_hydrology: PostHydrologyNoDrainageResult,
    lake_hydrology: LakeHydrologyDriverResult,
    drainage: HydrologyDrainageDriverResult,
    radiation: BalanceCheckRadiationFields,
    state: BalanceCheckState,
    *,
    steps_since_restart: int | jnp.ndarray,
    parameters: BalanceCheckParameters = DEFAULT_BALANCE_CHECK_PARAMETERS,
) -> BalanceCheckDriverResult:

    inputs = assemble_balance_check_inputs(
        forcing,
        physical,
        post_temperature,
        hydrology,
        post_hydrology,
        lake_hydrology,
        drainage,
        radiation,
        steps_since_restart=steps_since_restart,
    )
    result = balance_check(
        physical.soil_temperature.inputs.topology,
        inputs,
        state,
        parameters=parameters,
    )
    return BalanceCheckDriverResult(inputs=inputs, balance=result)


__all__ = [
    "BalanceCheckDriverResult",
    "BalanceCheckRadiationFields",
    "assemble_balance_check_inputs",
    "balance_check_driver_step",
    "balance_check_radiation_fields",
    "initial_balance_check_state",
]
