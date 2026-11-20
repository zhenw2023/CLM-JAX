
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from clm_jax.physics.active_layer import ActiveLayerState
from clm_jax.physics.aerosol import SnowAerosolMassState
from clm_jax.physics.balance_check import BalanceCheckState
from clm_jax.physics.carbon_nitrogen_balance_check import (
    CarbonNitrogenBalanceCheckState,
)
from clm_jax.physics.ecosystem_dynamics import EcosystemDynamicsAnnualUpdateState
from clm_jax.physics.ecosystem_dynamics_post_drainage import (
    EcosystemDynamicsPostDrainageState,
)
from clm_jax.physics.end_of_timestep import EndOfTimestepState
from clm_jax.physics.hydrology_drainage import HydrologyDrainageState
from clm_jax.physics.hydrology_driver import HydrologyDriverResult
from clm_jax.physics.hydrology_no_drainage import HydrologyNoDrainageState
from clm_jax.physics.lake_hydrology import LakeHydrologyState
from clm_jax.physics.lake_hydrology_driver import LakeHydrologyDriverResult
from clm_jax.physics.land_to_atmosphere import LandToAtmosphereState
from clm_jax.physics.methane_driver_facade import MethaneDriverActiveState
from clm_jax.physics.physical_driver import (
    PhysicalDriverResult,
    PhysicalDriverRuntimeResult,
    PhysicalDriverState,
)
from clm_jax.physics.post_hydrology_no_drainage_driver import (
    PostHydrologyNoDrainageResult,
    PostHydrologyNoDrainageState,
)
from clm_jax.physics.post_lake_hydrology_driver import PostLakeHydrologyResult
from clm_jax.physics.post_temperature_driver import (
    PostTemperatureDriverResult,
    PostTemperatureDriverState,
)
from clm_jax.physics.snow_age_grain import SnowAgeGrainState
from clm_jax.physics.surface_albedo import SurfaceAlbedoState
from clm_jax.physics.urban_albedo import UrbanAlbedoState


class PostLakeHydrologyState(NamedTuple):

    aerosol_mass: SnowAerosolMassState
    snow_age: SnowAgeGrainState
    snow_fraction_column: jnp.ndarray


class UrbanBuildingTimestepState(NamedTuple):

    roof_inner_temperature: jnp.ndarray
    sunwall_inner_temperature: jnp.ndarray
    shadewall_inner_temperature: jnp.ndarray
    floor_temperature: jnp.ndarray
    building_temperature: jnp.ndarray
    building_heat_flux: jnp.ndarray
    air_conditioning_flux: jnp.ndarray
    heating_flux: jnp.ndarray


class SoilTemperaturePostsolveTimestepState(NamedTuple):

    updated_interface_heat_flux: jnp.ndarray
    building_heat_flux: jnp.ndarray
    urban_air_conditioning_flux: jnp.ndarray
    urban_heating_flux: jnp.ndarray


class StandingWaterPhaseTimestepState(NamedTuple):

    phase_change_heat_flux: jnp.ndarray
    water_to_ice_flux: jnp.ndarray
    water_to_snow_heat_flux: jnp.ndarray


class SoilSnowPhaseTimestepState(NamedTuple):

    phase_flag: jnp.ndarray
    phase_change_heat_flux: jnp.ndarray
    snow_melt_flux: jnp.ndarray
    snow_freeze_flux: jnp.ndarray
    snow_drain_flux: jnp.ndarray
    snow_layer_melt_flux: jnp.ndarray
    snow_layer_freeze_flux: jnp.ndarray
    snow_melt_heat_flux: jnp.ndarray
    rural_snow_melt_heat_flux: jnp.ndarray
    urban_snow_melt_heat_flux: jnp.ndarray


class SoilTemperatureFinalTimestepState(NamedTuple):

    layer_one_two_heat_flux: jnp.ndarray
    ground_layer_heat_flux: jnp.ndarray


class SoilTemperatureTimestepState(NamedTuple):

    surface_water_temperature_column: jnp.ndarray
    nonlake_soil_snow_phase_flag_column_level: jnp.ndarray
    ground_heat_flux_patch_state: Any
    postsolve: SoilTemperaturePostsolveTimestepState
    standing_water_phase: StandingWaterPhaseTimestepState
    soil_snow_phase: SoilSnowPhaseTimestepState
    final: SoilTemperatureFinalTimestepState
    building: UrbanBuildingTimestepState


class LandTimestepCarry(NamedTuple):

    physical: PhysicalDriverState
    soil_temperature: SoilTemperatureTimestepState
    post_temperature: PostTemperatureDriverState
    hydrology: HydrologyNoDrainageState
    post_hydrology: PostHydrologyNoDrainageState
    lake_hydrology: LakeHydrologyState
    post_lake_hydrology: PostLakeHydrologyState
    active_layer: ActiveLayerState
    drainage: HydrologyDrainageState
    ecosystem_pre_drainage: EcosystemDynamicsAnnualUpdateState
    ecosystem_post_drainage: EcosystemDynamicsPostDrainageState
    methane: MethaneDriverActiveState
    surface_albedo: SurfaceAlbedoState
    urban_albedo: UrbanAlbedoState
    balance: BalanceCheckState
    carbon_nitrogen_balance: CarbonNitrogenBalanceCheckState
    land_to_atmosphere: LandToAtmosphereState
    end_of_timestep: EndOfTimestepState


def physical_driver_state(
    result: PhysicalDriverResult | PhysicalDriverRuntimeResult,
) -> PhysicalDriverState:

    required = (*PhysicalDriverState._fields, "irrigation_demand")
    missing = tuple(name for name in required if getattr(result, name, None) is None)
    if missing:
        raise ValueError(
            f"physical result was pruned before its state carry was projected: {missing}"
        )
    canopy_flux_state = result.canopy_flux.state
    bare_ground_state = result.bare_ground.state
    final_physical = getattr(result, "land_to_atmosphere", None)
    final_vegetation_temperature = getattr(
        final_physical,
        "vegetation_temperature_patch",
        None,
    )
    if final_vegetation_temperature is not None:
        canopy_flux_state = canopy_flux_state._replace(
            vegetation_temperature_patch=final_vegetation_temperature,
        )
        bare_ground_state = bare_ground_state._replace(
            temperature=bare_ground_state.temperature._replace(
                vegetation_temperature_patch=final_vegetation_temperature,
            ),
            friction_velocity=result.canopy_flux.state.friction_velocity,
        )
    return PhysicalDriverState(
        atmospheric=result.atmospheric.state,
        canopy=result.canopy.state._replace(
            irrigation=result.irrigation_demand.state,
        ),
        radiation=result.radiation.state,
        canopy_temperature=result.canopy_temperature.state,
        bare_ground=bare_ground_state,
        canopy_flux=canopy_flux_state,
        urban_flux=result.urban_flux.state,
        lake_flux=result.lake_flux.state,
        dust=result.dust.state,
        voc=result.voc.state,
        lake_temperature=result.lake_temperature.state,
    )


def soil_temperature_timestep_state(
    result: PhysicalDriverResult | PhysicalDriverRuntimeResult,
) -> SoilTemperatureTimestepState:

    soil_temperature = getattr(result, "soil_temperature", None)
    if soil_temperature is None:
        raise ValueError("physical result was pruned before soil-temperature carry projection")
    temperature = soil_temperature.temperature
    standing = temperature.standing_water_phase_change
    soil_phase = temperature.soil_snow_phase_change
    postsolve = temperature.postsolve_flux
    final = temperature.final_diagnostics
    building = temperature.building_temperature
    return SoilTemperatureTimestepState(
        surface_water_temperature_column=standing.standing_water_temperature,
        nonlake_soil_snow_phase_flag_column_level=soil_phase.phase_flag,
        ground_heat_flux_patch_state=temperature.ground_heat_flux.patch_state,
        postsolve=SoilTemperaturePostsolveTimestepState(
            updated_interface_heat_flux=postsolve.updated_interface_heat_flux,
            building_heat_flux=postsolve.building_heat_flux,
            urban_air_conditioning_flux=postsolve.urban_air_conditioning_flux,
            urban_heating_flux=postsolve.urban_heating_flux,
        ),
        standing_water_phase=StandingWaterPhaseTimestepState(
            phase_change_heat_flux=standing.phase_change_heat_flux,
            water_to_ice_flux=standing.water_to_ice_flux,
            water_to_snow_heat_flux=standing.water_to_snow_heat_flux,
        ),
        soil_snow_phase=SoilSnowPhaseTimestepState(
            phase_flag=soil_phase.phase_flag,
            phase_change_heat_flux=soil_phase.phase_change_heat_flux,
            snow_melt_flux=soil_phase.snow_melt_flux,
            snow_freeze_flux=soil_phase.snow_freeze_flux,
            snow_drain_flux=soil_phase.snow_drain_flux,
            snow_layer_melt_flux=soil_phase.snow_layer_melt_flux,
            snow_layer_freeze_flux=soil_phase.snow_layer_freeze_flux,
            snow_melt_heat_flux=soil_phase.snow_melt_heat_flux,
            rural_snow_melt_heat_flux=soil_phase.rural_snow_melt_heat_flux,
            urban_snow_melt_heat_flux=soil_phase.urban_snow_melt_heat_flux,
        ),
        final=SoilTemperatureFinalTimestepState(
            layer_one_two_heat_flux=final.layer_one_two_heat_flux,
            ground_layer_heat_flux=final.ground_layer_heat_flux,
        ),
        building=UrbanBuildingTimestepState(
            roof_inner_temperature=building.roof_inner_temperature,
            sunwall_inner_temperature=building.sunwall_inner_temperature,
            shadewall_inner_temperature=building.shadewall_inner_temperature,
            floor_temperature=building.floor_temperature,
            building_temperature=building.building_temperature,
            building_heat_flux=building.building_heat_flux,
            air_conditioning_flux=building.air_conditioning_flux,
            heating_flux=building.heating_flux,
        ),
    )


def post_temperature_driver_state(
    result: PostTemperatureDriverResult,
) -> PostTemperatureDriverState:

    return PostTemperatureDriverState(
        glacier_ice_melt_flux_column=(result.glacier_ice_melt.state.ice_melt_flux),
        soil_fluxes=result.soil_fluxes.state,
        patch_to_column=result.patch_to_column.state,
    )


def post_hydrology_driver_state(
    result: PostHydrologyNoDrainageResult,
) -> PostHydrologyNoDrainageState:

    return PostHydrologyNoDrainageState(
        glacier_surface_mass_balance=result.glacier_surface_mass_balance.state,
        aerosol_mass=result.aerosol_mass.state,
    )


def post_lake_hydrology_state(
    result: PostLakeHydrologyResult,
) -> PostLakeHydrologyState:

    return PostLakeHydrologyState(
        aerosol_mass=result.lake_aerosol_mass.state,
        snow_age=result.nonlake_snow_age.state,
        snow_fraction_column=result.snow_fraction_column,
    )


def compose_land_timestep_carry(
    *,
    physical: PhysicalDriverState,
    soil_temperature: SoilTemperatureTimestepState,
    post_temperature: PostTemperatureDriverResult,
    hydrology: HydrologyDriverResult,
    post_hydrology: PostHydrologyNoDrainageResult,
    lake_hydrology: LakeHydrologyDriverResult,
    post_lake_hydrology: PostLakeHydrologyResult,
    active_layer: Any,
    drainage: Any,
    ecosystem_pre_drainage: EcosystemDynamicsAnnualUpdateState,
    ecosystem_post_drainage: Any,
    methane: Any,
    surface_albedo: Any,
    urban_albedo: Any,
    balance: Any,
    carbon_nitrogen_balance: Any,
    land_to_atmosphere: Any,
    end_of_timestep: Any,
) -> LandTimestepCarry:

    return LandTimestepCarry(
        physical=physical,
        soil_temperature=soil_temperature,
        post_temperature=post_temperature_driver_state(post_temperature),
        hydrology=hydrology.hydrology.state,
        post_hydrology=post_hydrology_driver_state(post_hydrology),
        lake_hydrology=lake_hydrology.hydrology.state,
        post_lake_hydrology=(
            post_lake_hydrology
            if isinstance(post_lake_hydrology, PostLakeHydrologyState)
            else post_lake_hydrology_state(post_lake_hydrology)
        ),
        active_layer=active_layer.state,
        drainage=drainage.drainage.state,
        ecosystem_pre_drainage=ecosystem_pre_drainage,
        ecosystem_post_drainage=ecosystem_post_drainage.state,
        methane=methane.methane.state,
        surface_albedo=surface_albedo.state,
        urban_albedo=urban_albedo.state,
        balance=balance.balance.state,
        carbon_nitrogen_balance=carbon_nitrogen_balance.balance.state,
        land_to_atmosphere=land_to_atmosphere.state,
        end_of_timestep=end_of_timestep.state,
    )


__all__ = [
    "LandTimestepCarry",
    "PostLakeHydrologyState",
    "SoilSnowPhaseTimestepState",
    "SoilTemperatureFinalTimestepState",
    "SoilTemperaturePostsolveTimestepState",
    "SoilTemperatureTimestepState",
    "StandingWaterPhaseTimestepState",
    "UrbanBuildingTimestepState",
    "compose_land_timestep_carry",
    "physical_driver_state",
    "post_hydrology_driver_state",
    "post_lake_hydrology_state",
    "post_temperature_driver_state",
    "soil_temperature_timestep_state",
]
