
from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import NamedTuple

from clm_jax.io.atmospheric_driver import load_atmospheric_driver_native_data
from clm_jax.io.bare_ground_driver import load_bare_ground_driver_native_data
from clm_jax.io.canopy_driver import load_canopy_driver_native_data
from clm_jax.io.canopy_flux_driver import load_canopy_flux_driver_native_data
from clm_jax.io.canopy_temperature_driver import (
    load_canopy_temperature_driver_native_data,
)
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.dust_driver import load_dust_driver_native_data
from clm_jax.io.irrigation_demand_driver import (
    load_irrigation_demand_driver_native_data,
)
from clm_jax.io.lake_flux_driver import load_lake_flux_driver_native_data
from clm_jax.io.lake_temperature_driver import (
    load_lake_temperature_driver_native_data,
)
from clm_jax.io.radiation_driver import assert_same_topology
from clm_jax.io.radiation_sequence import load_radiation_sequence_native_data
from clm_jax.io.soil_temperature_driver import (
    load_soil_temperature_driver_native_data,
)
from clm_jax.io.urban_flux_driver import load_urban_flux_driver_native_data
from clm_jax.io.voc_driver import load_voc_driver_native_data
from clm_jax.physics.physical_driver import (
    PhysicalDriverInputs,
    PhysicalDriverParameters,
    PhysicalDriverState,
    PhysicalDriverStatic,
)


class PhysicalDriverNativeData(NamedTuple):

    static: PhysicalDriverStatic
    inputs: PhysicalDriverInputs
    state: PhysicalDriverState
    parameters: PhysicalDriverParameters
    land_restart_path: Path
    coupler_restart_path: Path


@cache
def load_physical_driver_native_data(
    case: CimeCase,
    land_restart_path: str | Path,
    coupler_restart_path: str | Path | None = None,
) -> PhysicalDriverNativeData:

    restart = Path(land_restart_path).expanduser().resolve()
    atmospheric = load_atmospheric_driver_native_data(
        case,
        restart,
        coupler_restart_path,
    )
    canopy = load_canopy_driver_native_data(case, restart)
    radiation = load_radiation_sequence_native_data(case, restart)
    canopy_temperature = load_canopy_temperature_driver_native_data(case, restart)
    bare_ground = load_bare_ground_driver_native_data(case, restart)
    canopy_flux = load_canopy_flux_driver_native_data(case, restart)
    urban_flux = load_urban_flux_driver_native_data(case, restart)
    lake_flux = load_lake_flux_driver_native_data(case, restart)
    irrigation_demand = load_irrigation_demand_driver_native_data(case, restart)
    dust = load_dust_driver_native_data(case, restart)
    voc = load_voc_driver_native_data(case, restart)
    lake_temperature = load_lake_temperature_driver_native_data(case, restart)
    soil_temperature = load_soil_temperature_driver_native_data(case, restart)

    topology = atmospheric.static.topology
    for other in (
        canopy.topology,
        radiation.topology,
        canopy_temperature.topology,
        bare_ground.topology,
        canopy_flux.topology,
        urban_flux.topology,
        lake_flux.topology,
        irrigation_demand.topology,
        dust.topology,
        voc.topology,
        lake_temperature.topology,
        soil_temperature.topology,
    ):
        assert_same_topology(topology, other)

    return PhysicalDriverNativeData(
        static=PhysicalDriverStatic(
            topology=topology,
            atmospheric=atmospheric.static,
        ),
        inputs=PhysicalDriverInputs(
            atmospheric=atmospheric.inputs,
            canopy=canopy.inputs,
            radiation=radiation.inputs,
            canopy_temperature=canopy_temperature.inputs,
            canopy_flux=canopy_flux.inputs,
            urban_flux=urban_flux.inputs,
            lake_flux=lake_flux.inputs,
            irrigation_demand=irrigation_demand.inputs,
            dust=dust.inputs,
            voc=voc.inputs,
            voc_accumulators=voc.accumulators,
            lake_temperature=lake_temperature.inputs,
            soil_temperature=soil_temperature.inputs,
        ),
        state=PhysicalDriverState(
            atmospheric=atmospheric.state,
            canopy=canopy.state,
            radiation=radiation.state,
            canopy_temperature=canopy_temperature.state,
            bare_ground=bare_ground.state,
            canopy_flux=canopy_flux.state,
            urban_flux=urban_flux.state,
            lake_flux=lake_flux.state,
            dust=dust.state,
            voc=voc.state,
            lake_temperature=lake_temperature.state,
        ),
        parameters=PhysicalDriverParameters(
            canopy=canopy.parameters,
            canopy_temperature=canopy_temperature.parameters,
            bare_ground=bare_ground.parameters,
            canopy_flux=canopy_flux.parameters,
            urban_flux=urban_flux.parameters,
            lake_flux=lake_flux.parameters,
            irrigation_demand=irrigation_demand.parameters,
            dust=dust.parameters,
            dust_execution=dust.execution,
            voc=voc.parameters,
        ),
        land_restart_path=restart,
        coupler_restart_path=atmospheric.coupler_restart_path,
    )


__all__ = [
    "PhysicalDriverNativeData",
    "load_physical_driver_native_data",
]
