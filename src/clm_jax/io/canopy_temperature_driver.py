
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.io.canopy_temperature_restart import load_canopy_temperature_restart
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.restart import SubgridTopology, load_subgrid_topology
from clm_jax.physics.canopy_temperature import (
    CanopyTemperatureInputs,
    CanopyTemperatureParameters,
    CanopyTemperatureState,
)


class CanopyTemperatureDriverNativeData(NamedTuple):
    topology: SubgridTopology
    inputs: CanopyTemperatureInputs
    parameters: CanopyTemperatureParameters
    state: CanopyTemperatureState
    restart_path: Path


def load_canopy_temperature_driver_native_data(
    case: CimeCase,
    restart_path: str | Path,
) -> CanopyTemperatureDriverNativeData:

    restart = Path(restart_path).expanduser().resolve()
    topology = load_subgrid_topology(restart)
    grid_placeholder = jnp.zeros(
        topology.grid.longitude_degrees.shape,
        dtype=jnp.float64,
    )
    column_placeholder = jnp.zeros(
        topology.column.gridcell.shape,
        dtype=jnp.float64,
    )
    native = load_canopy_temperature_restart(
        case,
        pressure_column=column_placeholder,
        specific_humidity_column=column_placeholder,
        temperature_column=column_placeholder,
        potential_temperature_column=column_placeholder,
        forcing_height_wind_gridcell=grid_placeholder,
        forcing_height_temperature_gridcell=grid_placeholder,
        forcing_height_humidity_gridcell=grid_placeholder,
        restart_path=restart,
    )
    return CanopyTemperatureDriverNativeData(
        topology=native.topology,
        inputs=native.inputs,
        parameters=native.parameters,
        state=native.state,
        restart_path=restart,
    )


__all__ = [
    "CanopyTemperatureDriverNativeData",
    "load_canopy_temperature_driver_native_data",
]
