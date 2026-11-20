
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.physical_driver import PhysicalDriverNativeData
from clm_jax.io.post_temperature_driver import PostTemperatureDriverNativeData
from clm_jax.io.radiation_driver import assert_same_topology
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.aerosol import (
    SnowAerosolConcentrationState,
    SnowAerosolMassState,
)
from clm_jax.physics.glacier_surface_mass_balance import (
    GlacierSurfaceMassBalanceParameters,
    GlacierSurfaceMassBalanceState,
)
from clm_jax.physics.post_hydrology_no_drainage_driver import (
    PostHydrologyNoDrainageInputs,
    PostHydrologyNoDrainageState,
)


class PostHydrologyNoDrainageNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: PostHydrologyNoDrainageInputs
    state: PostHydrologyNoDrainageState
    parameters: GlacierSurfaceMassBalanceParameters
    coupler_restart_path: Path


def _dynamic_runoff_routing_for_case(
    coupler_restart: Path,
    number_of_gridcells: int,
) -> jnp.ndarray:

    name = "g2x_gx_Sg_icemask_coupled_fluxes"
    coupled = np.asarray(
        load_native_variables(coupler_restart, [name], dtype=np.float64)[name],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(coupled)):
        raise ValueError("CISM coupled-flux ice mask contains non-finite values")
    if np.any(coupled != np.float64(0.0)):
        raise NotImplementedError(
            "nonzero dynamic-CISM runoff routing requires the GLC-to-land remap"
        )
    return jnp.zeros((number_of_gridcells,), dtype=jnp.float64)


def load_post_hydrology_no_drainage_native_data(
    case: CimeCase,
    physical: PhysicalDriverNativeData,
    post_temperature: PostTemperatureDriverNativeData,
) -> PostHydrologyNoDrainageNativeData:

    topology = physical.static.topology
    assert_same_topology(topology, post_temperature.topology)
    coupler_restart = physical.coupler_restart_path.resolve()
    number_of_gridcells = int(topology.grid.longitude_degrees.shape[0])
    number_of_columns = int(topology.column.gridcell.shape[0])
    snow = physical.state.canopy.canopy_hydrology.snow
    aerosol = physical.state.canopy.canopy_hydrology.snow_aerosol
    zero_column = jnp.zeros((number_of_columns,), dtype=jnp.float64)
    zero_snow = jnp.zeros_like(snow.snow_grain_radius, dtype=jnp.float64)
    concentration = SnowAerosolConcentrationState(
        **{name: zero_snow for name in SnowAerosolConcentrationState._fields}
    )
    return PostHydrologyNoDrainageNativeData(
        topology=topology,
        inputs=PostHydrologyNoDrainageInputs(
            do_surface_mass_balance_column=(
                post_temperature.inputs.do_surface_mass_balance_column
            ),
            dynamic_runoff_routing_gridcell=_dynamic_runoff_routing_for_case(
                coupler_restart,
                number_of_gridcells,
            ),
        ),
        state=PostHydrologyNoDrainageState(
            glacier_surface_mass_balance=GlacierSurfaceMassBalanceState(
                ice_growth_flux_column=zero_column,
                ice_melt_flux_column=zero_column,
                net_ice_flux_column=zero_column,
                dynamic_water_flux_column=zero_column,
            ),
            aerosol_mass=SnowAerosolMassState(
                top_snow_water_column=zero_column,
                snow_grain_radius_column_level=snow.snow_grain_radius,
                aerosol=aerosol,
                concentration=concentration,
            ),
        ),
        parameters=GlacierSurfaceMassBalanceParameters.from_case(case),
        coupler_restart_path=coupler_restart,
    )


__all__ = [
    "PostHydrologyNoDrainageNativeData",
    "load_post_hydrology_no_drainage_native_data",
]
