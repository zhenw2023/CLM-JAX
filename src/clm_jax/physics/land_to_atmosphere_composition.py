
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp

from clm_jax.physics.balance_check_driver import BalanceCheckRadiationFields
from clm_jax.physics.ecosystem_dynamics_post_drainage import (
    EcosystemDynamicsPostDrainageResult,
)
from clm_jax.physics.hydrology_drainage_driver import (
    HydrologyDrainageDriverResult,
)
from clm_jax.physics.lake_hydrology_driver import LakeHydrologyDriverResult
from clm_jax.physics.land_to_atmosphere import (
    LandToAtmosphereInputs,
    LandToAtmosphereParameters,
    LandToAtmosphereState,
    LandToAtmosphereTopology,
)
from clm_jax.physics.land_to_atmosphere_boundary import (
    LandToAtmospherePhysicalFields,
)
from clm_jax.physics.methane_driver_facade import MethaneDriverActiveResult
from clm_jax.physics.physical_driver import (
    PhysicalDriverResult,
    PhysicalDriverRuntimeResult,
)
from clm_jax.physics.post_temperature_driver import PostTemperatureDriverResult
from clm_jax.physics.urban_albedo import UrbanAlbedoResult, UrbanAlbedoState

if TYPE_CHECKING:
    from clm_jax.io.restart import SubgridTopology
    from clm_jax.physics.dynamic_subgrid_driver import DynamicSubgridDriverResult


class LandToAtmosphereExternalFields(NamedTuple):

    dry_deposition_velocity_patch: jnp.ndarray
    fire_flux_patch: jnp.ndarray
    fire_injection_top_patch: jnp.ndarray
    dynamic_sensible_heat_gridcell: jnp.ndarray
    dynamic_liquid_water_gridcell: jnp.ndarray
    dynamic_ice_water_gridcell: jnp.ndarray
    river_volume_gridcell: jnp.ndarray


class LandToAtmosphereCall(NamedTuple):

    topology: LandToAtmosphereTopology
    inputs: LandToAtmosphereInputs
    state: LandToAtmosphereState
    parameters: LandToAtmosphereParameters


class LandToAtmospherePhysicalCarry(NamedTuple):

    fields: LandToAtmospherePhysicalFields
    precipitation_conversion_sensible_heat_column: jnp.ndarray
    rain_to_snow_runoff_conversion_column: jnp.ndarray


class LandToAtmospherePostTemperatureCarry(NamedTuple):

    outgoing_longwave_patch: jnp.ndarray
    total_evaporation_patch: jnp.ndarray
    sensible_heat_patch: jnp.ndarray
    latent_heat_patch: jnp.ndarray


class LandToAtmosphereUpstreamCarry(NamedTuple):

    topology: SubgridTopology
    physical: LandToAtmospherePhysicalCarry
    radiation: BalanceCheckRadiationFields
    post_temperature: LandToAtmospherePostTemperatureCarry


def land_to_atmosphere_physical_carry(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
) -> LandToAtmospherePhysicalCarry:

    if physical.land_to_atmosphere is None:
        raise ValueError(
            "physical result lacks its same-step lnd2atm projection; rerun stage 330"
        )
    forcing = physical.atmospheric.column_forcing
    return LandToAtmospherePhysicalCarry(
        fields=physical.land_to_atmosphere,
        precipitation_conversion_sensible_heat_column=forcing.sensible_heat_flux,
        rain_to_snow_runoff_conversion_column=forcing.rain_to_snow_runoff,
    )


def land_to_atmosphere_post_temperature_carry(
    post_temperature: PostTemperatureDriverResult,
) -> LandToAtmospherePostTemperatureCarry:

    state = post_temperature.soil_fluxes.state
    return LandToAtmospherePostTemperatureCarry(
        outgoing_longwave_patch=state.energy.outgoing_longwave_patch,
        total_evaporation_patch=state.water.total_evaporation_patch,
        sensible_heat_patch=state.energy.total_sensible_heat_patch,
        latent_heat_patch=state.energy.total_latent_heat_patch,
    )


def external_fields_after_dynamic_subgrid(
    previous: LandToAtmosphereExternalFields,
    dynamic: DynamicSubgridDriverResult,
) -> LandToAtmosphereExternalFields:

    balance = dynamic.biogeophysical_balance
    return previous._replace(
        dynamic_sensible_heat_gridcell=balance.energy_flux.current_flux,
        dynamic_liquid_water_gridcell=balance.liquid_water_flux.current_flux,
        dynamic_ice_water_gridcell=balance.ice_water_flux.current_flux,
    )


def _leading_axis(name: str, value: jnp.ndarray, expected: int) -> jnp.ndarray:
    array = jnp.asarray(value, dtype=jnp.float64)
    if array.ndim not in (1, 2) or array.shape[0] != expected:
        raise ValueError(f"{name} must have leading native axis {expected}; got {array.shape}")
    return array


def _vector(name: str, value: jnp.ndarray, expected: int) -> jnp.ndarray:
    array = jnp.asarray(value, dtype=jnp.float64)
    if array.shape != (expected,):
        raise ValueError(f"{name} must have shape {(expected,)}; got {array.shape}")
    return array


def _albedo_state(albedo: UrbanAlbedoResult | UrbanAlbedoState) -> UrbanAlbedoState:
    return albedo.state if isinstance(albedo, UrbanAlbedoResult) else albedo


def compose_land_to_atmosphere_inputs(
    topology: LandToAtmosphereTopology,
    physical: (
        PhysicalDriverResult
        | PhysicalDriverRuntimeResult
        | LandToAtmospherePhysicalCarry
    ),
    radiation: BalanceCheckRadiationFields,
    post_temperature: (
        PostTemperatureDriverResult | LandToAtmospherePostTemperatureCarry
    ),
    lake_hydrology: LakeHydrologyDriverResult,
    drainage: HydrologyDrainageDriverResult,
    post_drainage: EcosystemDynamicsPostDrainageResult,
    methane: MethaneDriverActiveResult,
    albedo: UrbanAlbedoResult | UrbanAlbedoState,
    external: LandToAtmosphereExternalFields,
) -> LandToAtmosphereInputs:

    subgrid = topology.subgrid
    column_count = subgrid.column.gridcell.shape[0]
    patch_count = subgrid.patch.gridcell.shape[0]
    gridcell_count = subgrid.grid.longitude_degrees.shape[0]

    physical_carry = (
        physical
        if isinstance(physical, LandToAtmospherePhysicalCarry)
        else land_to_atmosphere_physical_carry(physical)
    )
    post_temperature_carry = (
        post_temperature
        if isinstance(post_temperature, LandToAtmospherePostTemperatureCarry)
        else land_to_atmosphere_post_temperature_carry(post_temperature)
    )
    physical_fields = physical_carry.fields
    lake_state = lake_hydrology.hydrology.state
    water = lake_state.water
    drainage_inputs = drainage.inputs
    drainage_state = drainage.drainage.state
    methane_finalize = methane.state.finalize
    final_albedo = _albedo_state(albedo).surface

    snow_water = _vector(
        "snow_water_column",
        water.snow_water_equivalent_column,
        column_count,
    )
    volumetric_water = _leading_axis(
        "volumetric_soil_water_column",
        drainage_state.volumetric_water_column_level,
        column_count,
    )
    direct_albedo = _leading_axis(
        "direct_albedo_patch",
        final_albedo.surface_direct_patch,
        patch_count,
    )
    diffuse_albedo = _leading_axis(
        "diffuse_albedo_patch",
        final_albedo.surface_diffuse_patch,
        patch_count,
    )
    sensible_heat = _vector(
        "sensible_heat_patch",
        lake_state.energy_flux.sensible_heat_total_patch,
        patch_count,
    )

    patch_fields = {
        "outgoing_longwave_patch": post_temperature_carry.outgoing_longwave_patch,
        "reference_temperature_2m_patch": (
            physical_fields.reference_temperature_2m_patch
        ),
        "reference_specific_humidity_2m_patch": (
            physical_fields.reference_specific_humidity_2m_patch
        ),
        "reference_wind_speed_10m_patch": (
            physical_fields.reference_wind_speed_10m_patch
        ),
        "eastward_stress_patch": physical_fields.eastward_stress_patch,
        "northward_stress_patch": physical_fields.northward_stress_patch,
        "total_evaporation_patch": post_temperature_carry.total_evaporation_patch,
        "absorbed_solar_patch": radiation.absorbed_shortwave_patch,
        "friction_velocity_patch": physical_fields.friction_velocity_patch,
        "aerodynamic_resistance_patch": (
            physical_fields.aerodynamic_resistance_patch
        ),
        "sensible_heat_patch": sensible_heat,
        "latent_heat_patch": post_temperature_carry.latent_heat_patch,
        "dry_deposition_velocity_patch": external.dry_deposition_velocity_patch,
        "voc_flux_patch": physical_fields.voc_flux_patch,
        "fire_flux_patch": external.fire_flux_patch,
        "fire_injection_top_patch": external.fire_injection_top_patch,
        "dust_flux_patch": physical_fields.dust_flux_patch,
    }
    patch = {
        name: _leading_axis(name, value, patch_count)
        for name, value in patch_fields.items()
    }

    column_fields = {
        "ice_runoff_snow_capping_column": (
            drainage_state.snow_capping_ice_runoff_flux_column
        ),
        "excess_ice_runoff_column": (
            drainage_state.excess_soil_ice_runoff_flux_column
        ),
        "precipitation_conversion_sensible_heat_column": (
            physical_carry.precipitation_conversion_sensible_heat_column
        ),
        "surface_runoff_column": drainage_state.surface_runoff_flux_column,
        "drainage_column": drainage_state.subsurface_drainage_flux_column,
        "rain_to_snow_runoff_conversion_column": (
            physical_carry.rain_to_snow_runoff_conversion_column
        ),
        "surface_water_runoff_column": (
            drainage_state.surface_water_runoff_flux_column
        ),
        "perched_drainage_column": drainage_state.perched_drainage_flux_column,
        "irrigation_column": drainage_inputs.irrigation_flux_column,
        "methane_surface_flux_column": methane_finalize.surface_methane_flux_column,
        "ending_water_balance_column": drainage_state.ending_water_mass_column,
    }
    column = {
        name: _leading_axis(name, value, column_count)
        for name, value in column_fields.items()
    }

    gridcell_fields = {
        "dynamic_sensible_heat_gridcell": external.dynamic_sensible_heat_gridcell,
        "net_carbon_exchange_gridcell": (
            -post_drainage.gridcell_fluxes.net_biome_production
        ),
        "net_methane_carbon_correction_gridcell": (
            methane_finalize.net_methane_carbon_correction_gridcell
        ),
        "dynamic_liquid_water_gridcell": external.dynamic_liquid_water_gridcell,
        "dynamic_ice_water_gridcell": external.dynamic_ice_water_gridcell,
        "river_volume_gridcell": external.river_volume_gridcell,
    }
    gridcell = {
        name: _vector(name, value, gridcell_count)
        for name, value in gridcell_fields.items()
    }

    return LandToAtmosphereInputs(
        snow_water_column=snow_water,
        volumetric_soil_water_column=volumetric_water,
        direct_albedo_patch=direct_albedo,
        diffuse_albedo_patch=diffuse_albedo,
        **patch,
        **column,
        **gridcell,
    )


def compose_land_to_atmosphere_state(
    previous: LandToAtmosphereState,
    drainage: HydrologyDrainageDriverResult,
) -> LandToAtmosphereState:

    current = drainage.drainage.state
    column_count = current.total_runoff_flux_column.shape[0]
    groundwater = _vector(
        "groundwater_runoff_column",
        current.wetland_glacier_runoff_flux_column,
        column_count,
    )
    runoff = _vector(
        "total_runoff_column",
        current.total_runoff_flux_column,
        column_count,
    )
    return previous._replace(
        groundwater_runoff_column=groundwater,
        total_runoff_column=runoff,
    )


def compose_land_to_atmosphere_call(
    topology: LandToAtmosphereTopology,
    previous_state: LandToAtmosphereState,
    parameters: LandToAtmosphereParameters,
    physical: (
        PhysicalDriverResult
        | PhysicalDriverRuntimeResult
        | LandToAtmospherePhysicalCarry
    ),
    radiation: BalanceCheckRadiationFields,
    post_temperature: (
        PostTemperatureDriverResult | LandToAtmospherePostTemperatureCarry
    ),
    lake_hydrology: LakeHydrologyDriverResult,
    drainage: HydrologyDrainageDriverResult,
    post_drainage: EcosystemDynamicsPostDrainageResult,
    methane: MethaneDriverActiveResult,
    albedo: UrbanAlbedoResult | UrbanAlbedoState,
    external: LandToAtmosphereExternalFields,
) -> LandToAtmosphereCall:

    return LandToAtmosphereCall(
        topology=topology,
        inputs=compose_land_to_atmosphere_inputs(
            topology,
            physical,
            radiation,
            post_temperature,
            lake_hydrology,
            drainage,
            post_drainage,
            methane,
            albedo,
            external,
        ),
        state=compose_land_to_atmosphere_state(previous_state, drainage),
        parameters=parameters,
    )


__all__ = [
    "LandToAtmosphereCall",
    "LandToAtmosphereExternalFields",
    "LandToAtmospherePhysicalCarry",
    "LandToAtmospherePostTemperatureCarry",
    "LandToAtmosphereUpstreamCarry",
    "compose_land_to_atmosphere_call",
    "compose_land_to_atmosphere_inputs",
    "compose_land_to_atmosphere_state",
    "external_fields_after_dynamic_subgrid",
    "land_to_atmosphere_physical_carry",
    "land_to_atmosphere_post_temperature_carry",
]
