
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.io.canopy_restart import load_canopy_hydrology_restart
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.irrigation_restart import load_irrigation_restart
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.canopy_driver import (
    CanopyDriverInputs,
    CanopyDriverParameters,
    CanopyDriverState,
)
from clm_jax.physics.canopy_step import CanopyHydrologyStepParameters
from clm_jax.physics.irrigation import IrrigationApplicationParameters


class CanopyDriverNativeData(NamedTuple):
    topology: SubgridTopology
    inputs: CanopyDriverInputs
    state: CanopyDriverState
    parameters: CanopyDriverParameters
    restart_path: Path


def load_canopy_driver_native_data(
    case: CimeCase,
    restart_path: str | Path,
) -> CanopyDriverNativeData:

    restart = Path(restart_path).expanduser().resolve()
    canopy = load_canopy_hydrology_restart(case, restart)
    irrigation = load_irrigation_restart(case, restart)
    if canopy.topology.patch.gridcell.shape != irrigation.topology.patch.gridcell.shape:
        raise ValueError("canopy and irrigation adapters loaded different patch topologies")
    grid_count = canopy.topology.grid.longitude_degrees.shape[0]
    return CanopyDriverNativeData(
        topology=canopy.topology,
        inputs=CanopyDriverInputs(
            structure=canopy.structure,
            previous_snow_drainage_flux_column=(
                canopy.previous_snow_drainage_flux_column
            ),
            flood_gridcell=jnp.zeros(grid_count, dtype=jnp.float64),
        ),
        state=CanopyDriverState(
            irrigation=irrigation.state,
            canopy_hydrology=canopy.state,
        ),
        parameters=CanopyDriverParameters(
            irrigation=IrrigationApplicationParameters.from_case(case),
            canopy_hydrology=CanopyHydrologyStepParameters.from_case(case),
        ),
        restart_path=restart,
    )


__all__ = ["CanopyDriverNativeData", "load_canopy_driver_native_data"]
