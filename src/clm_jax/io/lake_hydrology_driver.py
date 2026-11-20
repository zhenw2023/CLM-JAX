
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.constants import FREEZING_TEMPERATURE_FRESHWATER
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.hydrology_driver import HydrologyDriverNativeData
from clm_jax.io.physical_driver import PhysicalDriverNativeData
from clm_jax.io.post_hydrology_no_drainage_driver import (
    PostHydrologyNoDrainageNativeData,
)
from clm_jax.io.radiation_driver import assert_same_topology
from clm_jax.io.restart import SubgridTopology
from clm_jax.io.water_balance_initialization import (
    load_water_balance_initialization_native_data,
)
from clm_jax.physics.lake_hydrology import (
    LakeHydrologyAerosolState,
    LakeHydrologyEnergyFluxState,
    LakeHydrologyGeometryState,
    LakeHydrologyInputs,
    LakeHydrologyParameters,
    LakeHydrologyState,
    LakeHydrologyTemperatureState,
    LakeHydrologyWaterFluxState,
    LakeHydrologyWaterState,
)
from clm_jax.physics.lake_hydrology_driver import LakeHydrologyDriverInputs
from clm_jax.physics.snow_water import NUMBER_OF_SNOW_LEVELS
from clm_jax.physics.water_balance_initialization import begin_water_balance


class LakeHydrologyDriverNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: LakeHydrologyDriverInputs
    state: LakeHydrologyState
    parameters: LakeHydrologyParameters
    land_restart_path: Path


def load_lake_hydrology_driver_native_data(
    case: CimeCase,
    physical: PhysicalDriverNativeData,
    hydrology: HydrologyDriverNativeData,
    post_hydrology: PostHydrologyNoDrainageNativeData,
) -> LakeHydrologyDriverNativeData:

    topology = physical.static.topology
    assert_same_topology(topology, hydrology.topology)
    assert_same_topology(topology, post_hydrology.topology)
    water_balance = load_water_balance_initialization_native_data(
        case,
        physical.land_restart_path,
    )
    assert_same_topology(topology, water_balance.topology)
    beginning = begin_water_balance(
        topology,
        water_balance.inputs,
        water_balance.state,
        parameters=water_balance.parameters,
    ).state.beginning_water_mass_column

    number_of_gridcells = int(topology.grid.longitude_degrees.shape[0])
    number_of_columns = int(topology.column.gridcell.shape[0])
    number_of_patches = int(topology.patch.gridcell.shape[0])
    lake_map = physical.inputs.lake_temperature.native_map
    lake_column = jnp.asarray(lake_map.column_index, dtype=jnp.int32)
    lake_patch = jnp.asarray(lake_map.patch_index, dtype=jnp.int32)
    lake_count = int(lake_column.shape[0])
    if lake_patch.shape != (lake_count,):
        raise ValueError("native lake column and patch maps disagree")

    lake_compact = physical.state.lake_temperature
    lake_thickness_compact = (
        physical.inputs.lake_temperature.geometry.lake_layer_thickness
    )
    number_of_lake_levels = int(lake_thickness_compact.shape[1])
    lake_thickness = jnp.ones(
        (number_of_columns, number_of_lake_levels),
        dtype=jnp.float64,
    ).at[lake_column].set(lake_thickness_compact)
    lake_temperature = jnp.full(
        (number_of_columns, number_of_lake_levels),
        jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
        dtype=jnp.float64,
    ).at[lake_column].set(lake_compact.lake_temperature)
    lake_ice_fraction = jnp.zeros_like(lake_temperature).at[lake_column].set(
        lake_compact.lake_ice_fraction
    )

    snow = hydrology.state.snow_structure
    final = hydrology.state.final_diagnostics
    postprocess = hydrology.state.snow_postprocess
    zero_column = jnp.zeros((number_of_columns,), dtype=jnp.float64)
    zero_patch = jnp.zeros((number_of_patches,), dtype=jnp.float64)
    zero_snow = jnp.zeros(
        (number_of_columns, NUMBER_OF_SNOW_LEVELS),
        dtype=jnp.float64,
    )
    zero_grid = jnp.zeros((number_of_gridcells,), dtype=jnp.float64)
    zero_water_flux = LakeHydrologyWaterFluxState(
        **{
            name: (
                zero_patch
                if name in LakeHydrologyWaterFluxState._fields[:7]
                else zero_snow
                if name == "snowmelt_flux_column_level"
                else zero_column
            )
            for name in LakeHydrologyWaterFluxState._fields
        }
    )
    zero_energy_flux = LakeHydrologyEnergyFluxState(
        snowmelt_energy_flux_column=zero_column,
        sensible_heat_total_patch=zero_patch,
        sensible_heat_ground_patch=zero_patch,
        soil_heat_flux_ground_patch=zero_patch,
        net_ground_heat_flux_patch=zero_patch,
        ground_lake_heat_flux_patch=zero_patch,
    )

    inputs = LakeHydrologyInputs(
        atmospheric_temperature_column=zero_column,
        rainfall_flux_column=zero_column,
        snowfall_flux_column=zero_column,
        wind_speed_gridcell=zero_grid,
        flood_flux_gridcell=physical.inputs.canopy.flood_gridcell,
        atmospheric_aerosol_flux_gridcell_species=(
            hydrology.inputs.atmospheric_aerosol_flux_gridcell_species
        ),
        soil_saturation_column_level=(
            hydrology.inputs.template.soil_water_movement.saturated_water_content_column_level
        ),
        lake_layer_thickness_column_level=lake_thickness,
        ground_temperature_column=final.ground_temperature_column,
        snow_cover_shape_column=(
            hydrology.inputs.template.snow_compaction.snow_cover_shape_column
        ),
        soil_snow_melt_flag_column_level=jnp.zeros_like(
            snow.soil_snow_temperature_column_level,
            dtype=jnp.int32,
        ),
        previous_snow_water_column_level=(
            physical.state.canopy.canopy_hydrology.snow.layer_ice
            + physical.state.canopy.canopy_hydrology.snow.layer_liquid
        ),
        begin_water_balance_column=beginning,
        soil_evaporation_flux_patch=zero_patch,
        total_evaporation_flux_patch=zero_patch,
    )
    state = LakeHydrologyState(
        geometry=LakeHydrologyGeometryState(
            snow_layer_count_column=snow.snow_layer_count_column,
            soil_snow_thickness_column_level=(
                snow.soil_snow_thickness_column_level
            ),
            soil_snow_node_depth_column_level=(
                snow.soil_snow_node_depth_column_level
            ),
            soil_snow_interface_depth_column_interface=(
                snow.soil_snow_interface_depth_column_interface
            ),
        ),
        temperature=LakeHydrologyTemperatureState(
            lake_temperature_column_level=lake_temperature,
            soil_snow_temperature_column_level=(
                snow.soil_snow_temperature_column_level
            ),
            top_temperature_gradient_column=(
                final.top_snow_temperature_gradient_column
            ),
            top_snow_temperature_column=final.top_snow_temperature_column,
            snow_temperature_mass_sum_column=(
                postprocess.snow_temperature_mass_product_column
            ),
        ),
        water=LakeHydrologyWaterState(
            effective_snow_fraction_column=snow.effective_snow_fraction_column,
            snow_fraction_column=snow.snow_fraction_column,
            old_ice_fraction_column_level=(
                hydrology.inputs.template.snow_compaction.old_ice_fraction_column_level
            ),
            snow_depth_column=snow.snow_depth_column,
            snow_water_equivalent_column=snow.snow_water_equivalent_column,
            integrated_snow_column=snow.integrated_snow_column,
            snow_ice_column=postprocess.snow_ice_column,
            snow_liquid_column=postprocess.snow_liquid_column,
            soil_snow_ice_column_level=snow.soil_snow_ice_column_level,
            soil_snow_liquid_column_level=snow.soil_snow_liquid_column_level,
            soil_water_volume_fraction_column_level=(
                final.volumetric_soil_water_column_level
            ),
            snow_grain_radius_column_level=snow.snow_grain_radius_column_level,
            top_snow_grain_radius_column=final.top_snow_grain_radius_column,
            top_snow_water_column=final.top_snow_water_column,
            top_snow_liquid_fraction_column=(
                final.top_snow_liquid_fraction_column
            ),
            end_water_balance_column=zero_column,
        ),
        water_flux=zero_water_flux,
        energy_flux=zero_energy_flux,
        aerosol=LakeHydrologyAerosolState(
            mass=post_hydrology.state.aerosol_mass.aerosol,
            deposition=snow.aerosol.deposition,
        ),
        lake_ice_fraction_column_level=lake_ice_fraction,
    )
    return LakeHydrologyDriverNativeData(
        topology=topology,
        inputs=LakeHydrologyDriverInputs(
            template=inputs,
            lake_column_index=lake_column,
            lake_patch_index=lake_patch,
        ),
        state=state,
        parameters=LakeHydrologyParameters.from_case(case),
        land_restart_path=physical.land_restart_path,
    )


__all__ = [
    "LakeHydrologyDriverNativeData",
    "load_lake_hydrology_driver_native_data",
]
