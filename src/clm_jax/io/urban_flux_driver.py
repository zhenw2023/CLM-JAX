
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from clm_jax.io.canopy_temperature_driver import (
    load_canopy_temperature_driver_native_data,
)
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.radiation_driver import assert_same_topology
from clm_jax.io.restart import SubgridTopology
from clm_jax.io.urban_fluxes_restart import load_urban_fluxes_restart
from clm_jax.physics.urban_flux_driver import (
    UrbanFluxDriverInputs,
    UrbanFluxDriverParameters,
    UrbanFluxDriverState,
)


class UrbanFluxDriverNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: UrbanFluxDriverInputs
    parameters: UrbanFluxDriverParameters
    state: UrbanFluxDriverState
    restart_path: Path
    surface_data_path: Path


def load_urban_flux_driver_native_data(
    case: CimeCase,
    restart_path: str | Path,
) -> UrbanFluxDriverNativeData:

    urban = load_urban_fluxes_restart(case, restart_path)
    temperature = load_canopy_temperature_driver_native_data(
        case,
        urban.restart_path,
    )
    assert_same_topology(urban.topology, temperature.topology)
    return UrbanFluxDriverNativeData(
        topology=urban.topology,
        inputs=UrbanFluxDriverInputs(
            native_map=urban.native_map,
            static_parameters=urban.parameters,
            canopy_temperature_parameters=temperature.parameters,
        ),
        parameters=UrbanFluxDriverParameters(
            maximum_stable_zeta=urban.maximum_stable_zeta,
            building_temperature_method=urban.building_temperature_method,
        ),
        state=urban.state,
        restart_path=urban.restart_path,
        surface_data_path=urban.surface_data_path,
    )


__all__ = [
    "UrbanFluxDriverNativeData",
    "load_urban_flux_driver_native_data",
]
