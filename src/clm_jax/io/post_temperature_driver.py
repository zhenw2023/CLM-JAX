
from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.lake_flux_driver import load_lake_flux_driver_native_data
from clm_jax.io.lnd2glc import load_land_to_glacier_native_data
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.radiation_driver import assert_same_topology
from clm_jax.io.restart import SubgridTopology, load_restart_clock
from clm_jax.io.urban_flux_driver import load_urban_flux_driver_native_data
from clm_jax.physics.patch_to_column_fluxes import (
    PatchToColumnFluxParameters,
    PatchToColumnFluxState,
)
from clm_jax.physics.post_temperature_driver import (
    PostTemperatureDriverInputs,
    PostTemperatureDriverParameters,
    PostTemperatureDriverState,
    PostTemperatureNativeMap,
)
from clm_jax.physics.soil_fluxes import (
    NUMBER_OF_SOIL_SNOW_LEVELS,
    SoilFluxesEnergyState,
    SoilFluxesInputs,
    SoilFluxesParameters,
    SoilFluxesState,
    SoilFluxesWaterState,
)


class PostTemperatureDriverNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: PostTemperatureDriverInputs
    state: PostTemperatureDriverState
    parameters: PostTemperatureDriverParameters
    restart_path: Path


def _zero_soil_fluxes_inputs(
    number_of_columns: int,
    number_of_patches: int,
) -> SoilFluxesInputs:

    column = jnp.zeros((number_of_columns,), dtype=jnp.float64)
    patch = jnp.zeros((number_of_patches,), dtype=jnp.float64)
    layer = jnp.zeros(
        (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS),
        dtype=jnp.float64,
    )
    return SoilFluxesInputs(
        snow_layer_count_column=jnp.zeros((number_of_columns,), dtype=jnp.int32),
        downward_longwave_column=column,
        effective_snow_fraction_column=column,
        surface_water_fraction_column=column,
        liquid_water_column_level=layer,
        ice_water_column_level=layer,
        ground_emissivity_column=column,
        surface_water_temperature_column=column,
        previous_surface_water_temperature_column=column,
        previous_soil_snow_temperature_column_level=layer,
        ground_temperature_column=column,
        soil_snow_temperature_column_level=layer,
        phase_change_heat_flux_column=column,
        surface_water_phase_change_heat_flux_column=column,
        temperature_factor_column_level=layer,
        surface_water_heat_capacity_column=column,
        latent_heat_column=column,
        surface_water_to_snow_heat_flux_column=column,
        building_heat_error_column=column,
        vegetation_exposed_patch=jnp.zeros((number_of_patches,), dtype=jnp.int32),
        absorbed_soil_patch=patch,
        absorbed_snow_patch=patch,
        absorbed_ground_patch=patch,
        downward_longwave_below_canopy_patch=patch,
        upward_longwave_above_canopy_patch=patch,
        ground_sensible_derivative_patch=patch,
        ground_latent_derivative_patch=patch,
        vegetation_transpiration_patch=patch,
        vegetation_evaporation_patch=patch,
        vegetation_sensible_heat_patch=patch,
        waste_heat_patch=patch,
        air_conditioner_heat_patch=patch,
        traffic_heat_patch=patch,
    )


def _zero_soil_fluxes_state(
    number_of_columns: int,
    number_of_patches: int,
) -> SoilFluxesState:

    column = jnp.zeros((number_of_columns,), dtype=jnp.float64)
    patch = jnp.zeros((number_of_patches,), dtype=jnp.float64)
    return SoilFluxesState(
        skin_temperature_patch=patch,
        water=SoilFluxesWaterState(
            **{name: patch for name in SoilFluxesWaterState._fields}
        ),
        energy=SoilFluxesEnergyState(
            **{
                name: column if name == "soil_energy_error_column" else patch
                for name in SoilFluxesEnergyState._fields
            }
        ),
    )


def _zero_patch_to_column_state(number_of_columns: int) -> PatchToColumnFluxState:
    column = jnp.zeros((number_of_columns,), dtype=jnp.float64)
    return PatchToColumnFluxState(
        **{name: column for name in PatchToColumnFluxState._fields}
    )


@cache
def load_post_temperature_driver_native_data(
    case: CimeCase,
    restart_path: str | Path,
) -> PostTemperatureDriverNativeData:

    restart = Path(restart_path).expanduser().resolve()
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")

    glacier = load_land_to_glacier_native_data(case, restart)
    urban = load_urban_flux_driver_native_data(case, restart)
    lake = load_lake_flux_driver_native_data(case, restart)
    topology = urban.topology
    assert_same_topology(topology, lake.topology)
    do_smb = glacier.inputs.do_surface_mass_balance
    number_of_columns = int(topology.column.gridcell.shape[0])
    number_of_patches = int(topology.patch.gridcell.shape[0])
    if do_smb.shape != (number_of_columns,):
        raise ValueError("SMB filter and native column topology disagree")

    clock = load_restart_clock(restart)
    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    maximum_patches = int(general["maxpatch_pft"])
    soil_parameters = SoilFluxesParameters(
        time_step_seconds=float(clock.step_seconds),
        maximum_patches_per_column=maximum_patches,
    )
    patch_parameters = PatchToColumnFluxParameters(
        maximum_patches_per_column=maximum_patches
    )
    native_map = PostTemperatureNativeMap(
        urban_surface_patch_index=urban.inputs.native_map.surface_patch_index,
        lake_column_index=lake.inputs.native_map.column_index,
        lake_patch_index=lake.inputs.native_map.patch_index,
    )
    return PostTemperatureDriverNativeData(
        topology=topology,
        inputs=PostTemperatureDriverInputs(
            native_map=native_map,
            do_surface_mass_balance_column=jnp.asarray(do_smb, dtype=jnp.bool_),
            soil_fluxes_template=_zero_soil_fluxes_inputs(
                number_of_columns,
                number_of_patches,
            ),
        ),
        state=PostTemperatureDriverState(
            glacier_ice_melt_flux_column=jnp.zeros(
                (number_of_columns,),
                dtype=jnp.float64,
            ),
            soil_fluxes=_zero_soil_fluxes_state(
                number_of_columns,
                number_of_patches,
            ),
            patch_to_column=_zero_patch_to_column_state(number_of_columns),
        ),
        parameters=PostTemperatureDriverParameters(
            soil_fluxes=soil_parameters,
            patch_to_column=patch_parameters,
        ),
        restart_path=restart,
    )


__all__ = [
    "PostTemperatureDriverNativeData",
    "load_post_temperature_driver_native_data",
]
