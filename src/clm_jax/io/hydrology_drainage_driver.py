
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.hydrology_driver import HydrologyDriverNativeData
from clm_jax.io.lake_hydrology_driver import LakeHydrologyDriverNativeData
from clm_jax.io.physical_driver import PhysicalDriverNativeData
from clm_jax.io.post_hydrology_no_drainage_driver import (
    PostHydrologyNoDrainageNativeData,
)
from clm_jax.io.radiation_driver import assert_same_topology
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.hydrology_drainage import (
    HydrologyDrainageParameters,
    HydrologyDrainageState,
)
from clm_jax.physics.hydrology_drainage_driver import (
    HydrologyDrainageDriverInputs,
)
from clm_jax.physics.snow_water import (
    NUMBER_OF_GROUND_LEVELS,
    NUMBER_OF_SOIL_SNOW_LEVELS,
)


class HydrologyDrainageDriverNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: HydrologyDrainageDriverInputs
    state: HydrologyDrainageState
    parameters: HydrologyDrainageParameters
    land_restart_path: Path


def load_hydrology_drainage_driver_native_data(
    case: CimeCase,
    physical: PhysicalDriverNativeData,
    hydrology: HydrologyDriverNativeData,
    post_hydrology: PostHydrologyNoDrainageNativeData,
    lake_hydrology: LakeHydrologyDriverNativeData,
) -> HydrologyDrainageDriverNativeData:

    topology = physical.static.topology
    assert_same_topology(topology, hydrology.topology)
    assert_same_topology(topology, post_hydrology.topology)
    assert_same_topology(topology, lake_hydrology.topology)

    number_of_columns = int(topology.column.gridcell.shape[0])
    zero_column = jnp.zeros((number_of_columns,), dtype=jnp.float64)
    zero_ground = jnp.zeros(
        (number_of_columns, NUMBER_OF_GROUND_LEVELS),
        dtype=jnp.float64,
    )
    zero_soil_snow = jnp.zeros(
        (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS),
        dtype=jnp.float64,
    )

    retained_state = HydrologyDrainageState(
        soil_snow_liquid_water_column_level=zero_soil_snow,
        soil_snow_ice_water_column_level=zero_soil_snow,
        surface_water_column=zero_column,
        water_table_depth_column=zero_column,
        perched_water_table_depth_column=zero_column,
        frost_table_depth_column=zero_column,
        soil_ice_fraction_column_level=zero_ground,
        volumetric_water_column_level=zero_ground,
        ending_water_mass_column=zero_column,
        perched_drainage_flux_column=zero_column,
        saturated_excess_flux_column=zero_column,
        subsurface_drainage_flux_column=zero_column,
        surface_runoff_flux_column=zero_column,
        surface_water_runoff_flux_column=zero_column,
        infiltration_flux_column=zero_column,
        wetland_glacier_runoff_flux_column=zero_column,
        total_runoff_flux_column=zero_column,
        urban_runoff_flux_column=zero_column,
        rural_runoff_flux_column=zero_column,
        excess_soil_ice_runoff_flux_column=zero_column,
        snow_capping_ice_runoff_flux_column=zero_column,
    )
    return HydrologyDrainageDriverNativeData(
        topology=topology,
        inputs=HydrologyDrainageDriverInputs(
            flood_flux_gridcell=physical.inputs.canopy.flood_gridcell,
            beginning_water_mass_column=(
                lake_hydrology.inputs.template.begin_water_balance_column
            ),
            total_plant_stored_water_column=zero_column,
            do_surface_mass_balance_column=(
                post_hydrology.inputs.do_surface_mass_balance_column
            ),
            dynamic_runoff_routing_gridcell=(
                post_hydrology.inputs.dynamic_runoff_routing_gridcell
            ),
        ),
        state=retained_state,
        parameters=HydrologyDrainageParameters.from_case(case),
        land_restart_path=physical.land_restart_path,
    )


__all__ = [
    "HydrologyDrainageDriverNativeData",
    "load_hydrology_drainage_driver_native_data",
]
