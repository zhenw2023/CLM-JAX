
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.aerosol import (
    SnowAerosolMassInputs,
    SnowAerosolMassResult,
    SnowAerosolMassState,
    SnowAerosolState,
    snow_aerosol_masses,
)
from clm_jax.physics.glacier_surface_mass_balance import (
    DEFAULT_GLACIER_SURFACE_MASS_BALANCE_PARAMETERS,
    GlacierSurfaceMassBalanceInputs,
    GlacierSurfaceMassBalanceParameters,
    GlacierSurfaceMassBalanceResult,
    GlacierSurfaceMassBalanceState,
    compute_glacier_surface_mass_balance,
)
from clm_jax.physics.hydrology_driver import HydrologyDriverResult
from clm_jax.physics.post_temperature_driver import PostTemperatureDriverResult
from clm_jax.physics.snow_water import NUMBER_OF_SNOW_LEVELS, build_snow_column_masks


class PostHydrologyNoDrainageInputs(NamedTuple):

    do_surface_mass_balance_column: jnp.ndarray
    dynamic_runoff_routing_gridcell: jnp.ndarray


class PostHydrologyNoDrainageState(NamedTuple):

    glacier_surface_mass_balance: GlacierSurfaceMassBalanceState
    aerosol_mass: SnowAerosolMassState


class PostHydrologyNoDrainageResult(NamedTuple):

    glacier_surface_mass_balance: GlacierSurfaceMassBalanceResult
    aerosol_mass: SnowAerosolMassResult


def _current_full_aerosol_state(
    retained: SnowAerosolState,
    hydrology: HydrologyDriverResult,
) -> SnowAerosolState:

    primary = hydrology.hydrology.state.snow_structure.aerosol
    return retained._replace(
        hydrophobic_black_carbon_layer=(primary.black_carbon_hydrophobic_layer),
        hydrophilic_black_carbon_layer=(primary.black_carbon_hydrophilic_layer),
        hydrophobic_organic_carbon_layer=(primary.organic_carbon_hydrophobic_layer),
        hydrophilic_organic_carbon_layer=(primary.organic_carbon_hydrophilic_layer),
        dust_species_1_layer=primary.dust_species_1_layer,
        dust_species_2_layer=primary.dust_species_2_layer,
        dust_species_3_layer=primary.dust_species_3_layer,
        dust_species_4_layer=primary.dust_species_4_layer,
    )


def assemble_glacier_surface_mass_balance_boundary(
    post_temperature: PostTemperatureDriverResult,
    hydrology: HydrologyDriverResult,
    inputs: PostHydrologyNoDrainageInputs,
    state: PostHydrologyNoDrainageState,
) -> tuple[GlacierSurfaceMassBalanceInputs, GlacierSurfaceMassBalanceState]:

    water = hydrology.hydrology.state
    return (
        GlacierSurfaceMassBalanceInputs(
            do_surface_mass_balance_column=(
                inputs.do_surface_mass_balance_column
            ),
            dynamic_runoff_routing_gridcell=(
                inputs.dynamic_runoff_routing_gridcell
            ),
            snow_persistence_column=(
                water.snow_postprocess.snow_persistence_column
            ),
            snow_capping_ice_flux_column=(
                water.snow_capping.snow_capping_ice_flux_column
            ),
        ),
        state.glacier_surface_mass_balance._replace(
            ice_melt_flux_column=(
                post_temperature.glacier_ice_melt.state.ice_melt_flux
            )
        ),
    )


def assemble_nonlake_aerosol_mass_boundary(
    topology: SubgridTopology,
    hydrology: HydrologyDriverResult,
    state: PostHydrologyNoDrainageState,
) -> tuple[SnowAerosolMassInputs, SnowAerosolMassState]:

    water = hydrology.hydrology.state
    snow = water.snow_structure
    masks = build_snow_column_masks(topology, snow.snow_layer_count_column)
    retained = state.aerosol_mass
    return (
        SnowAerosolMassInputs(
            snow_column=masks.snow_column,
            no_snow_column=masks.no_snow_column,
            snow_layer_count_column=snow.snow_layer_count_column,
            snow_ice_column_level=(
                snow.soil_snow_ice_column_level[:, :NUMBER_OF_SNOW_LEVELS]
            ),
            snow_liquid_column_level=(
                snow.soil_snow_liquid_column_level[:, :NUMBER_OF_SNOW_LEVELS]
            ),
        ),
        retained._replace(
            top_snow_water_column=(
                water.final_diagnostics.top_snow_water_column
            ),
            snow_grain_radius_column_level=snow.snow_grain_radius_column_level,
            aerosol=_current_full_aerosol_state(retained.aerosol, hydrology),
        ),
    )


def post_hydrology_no_drainage_step(
    topology: SubgridTopology,
    post_temperature: PostTemperatureDriverResult,
    hydrology: HydrologyDriverResult,
    inputs: PostHydrologyNoDrainageInputs,
    state: PostHydrologyNoDrainageState,
    parameters: GlacierSurfaceMassBalanceParameters = (
        DEFAULT_GLACIER_SURFACE_MASS_BALANCE_PARAMETERS
    ),
) -> PostHydrologyNoDrainageResult:

    glacier_inputs, glacier_state = (
        assemble_glacier_surface_mass_balance_boundary(
            post_temperature,
            hydrology,
            inputs,
            state,
        )
    )
    glacier = compute_glacier_surface_mass_balance(
        topology,
        glacier_inputs,
        glacier_state,
        parameters,
    )

    aerosol_inputs, aerosol_state = assemble_nonlake_aerosol_mass_boundary(
        topology,
        hydrology,
        state,
    )
    aerosol = snow_aerosol_masses(aerosol_inputs, aerosol_state)
    return PostHydrologyNoDrainageResult(
        glacier_surface_mass_balance=glacier,
        aerosol_mass=aerosol,
    )


__all__ = [
    "PostHydrologyNoDrainageInputs",
    "PostHydrologyNoDrainageResult",
    "PostHydrologyNoDrainageState",
    "assemble_glacier_surface_mass_balance_boundary",
    "assemble_nonlake_aerosol_mass_boundary",
    "post_hydrology_no_drainage_step",
]
