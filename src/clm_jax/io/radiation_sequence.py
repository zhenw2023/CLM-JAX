
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.radiation_driver import (
    assert_same_topology,
    load_radiation_driver_native_data,
)
from clm_jax.io.restart import SubgridTopology
from clm_jax.io.urban_radiation_restart import load_urban_radiation_restart
from clm_jax.physics.radiation_sequence import (
    RadiationSequenceInputs,
    RadiationSequenceState,
)


class RadiationSequenceNativeData(NamedTuple):
    topology: SubgridTopology
    inputs: RadiationSequenceInputs
    state: RadiationSequenceState
    restart_path: Path


def load_radiation_sequence_native_data(
    case: CimeCase,
    restart_path: str | Path,
) -> RadiationSequenceNativeData:

    restart = Path(restart_path).expanduser().resolve()
    surface = load_radiation_driver_native_data(case, restart)
    number_of_gridcells = surface.topology.grid.longitude_degrees.shape[0]
    band_placeholder = jnp.zeros((number_of_gridcells, 2), dtype=jnp.float64)
    scalar_placeholder = jnp.zeros((number_of_gridcells,), dtype=jnp.float64)
    urban = load_urban_radiation_restart(
        case,
        band_placeholder,
        band_placeholder,
        scalar_placeholder,
        restart,
        preceding_surface_state=surface.state.surface_radiation,
    )
    assert_same_topology(surface.topology, urban.topology)
    return RadiationSequenceNativeData(
        topology=surface.topology,
        inputs=RadiationSequenceInputs(
            canopy_surface=surface.inputs,
            urban=urban.inputs,
        ),
        state=RadiationSequenceState(
            canopy_surface=surface.state,
            urban=urban.state,
        ),
        restart_path=restart,
    )


__all__ = ["RadiationSequenceNativeData", "load_radiation_sequence_native_data"]
