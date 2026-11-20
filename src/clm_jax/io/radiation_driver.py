
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
from weakref import ref

import jax
import jax.numpy as jnp
import numpy as np

from clm_jax.io.canopy_radiation_restart import load_canopy_radiation_restart
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.restart import SubgridTopology
from clm_jax.io.surface_radiation_restart import load_surface_radiation_restart
from clm_jax.physics.radiation_driver import (
    RadiationDriverInputs,
    RadiationDriverState,
)


class RadiationDriverNativeData(NamedTuple):
    topology: SubgridTopology
    inputs: RadiationDriverInputs
    state: RadiationDriverState
    restart_path: Path


_VALIDATED_TOPOLOGY_FIELDS = {}


def assert_same_topology(left: SubgridTopology, right: SubgridTopology) -> None:

    for left_group, right_group in zip(left, right, strict=True):
        for left_value, right_value in zip(left_group, right_group, strict=True):
            immutable = isinstance(left_value, jax.Array) and isinstance(right_value, jax.Array)
            key = (id(left_value), id(right_value))
            cached = _VALIDATED_TOPOLOGY_FIELDS.get(key) if immutable else None
            if (cached is not None and cached[0]() is left_value and cached[1]() is right_value
                    and not left_value.is_deleted() and not right_value.is_deleted()):
                continue
            if not np.array_equal(np.asarray(left_value), np.asarray(right_value)):
                raise ValueError("radiation adapters loaded different subgrid topologies")
            if immutable:
                def forget(_, key=key):
                    _VALIDATED_TOPOLOGY_FIELDS.pop(key, None)
                _VALIDATED_TOPOLOGY_FIELDS[key] = (
                    ref(left_value, forget), ref(right_value, forget),
                )


def load_radiation_driver_native_data(
    case: CimeCase,
    restart_path: str | Path,
) -> RadiationDriverNativeData:

    restart = Path(restart_path).expanduser().resolve()
    canopy = load_canopy_radiation_restart(case, restart)
    number_of_gridcells = canopy.topology.grid.longitude_degrees.shape[0]
    placeholder_shortwave = jnp.zeros(
        (number_of_gridcells, 2),
        dtype=jnp.float64,
    )
    surface = load_surface_radiation_restart(
        case,
        placeholder_shortwave,
        placeholder_shortwave,
        restart,
    )
    assert_same_topology(canopy.topology, surface.topology)
    return RadiationDriverNativeData(
        topology=canopy.topology,
        inputs=RadiationDriverInputs(
            canopy_sunshade=canopy.inputs,
            surface_radiation=surface.inputs,
        ),
        state=RadiationDriverState(
            canopy_sunshade=canopy.state,
            surface_radiation=surface.state,
        ),
        restart_path=restart,
    )


__all__ = [
    "RadiationDriverNativeData",
    "assert_same_topology",
    "load_radiation_driver_native_data",
]
