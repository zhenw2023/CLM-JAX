
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.canopy_temperature_driver import (
    load_canopy_temperature_driver_native_data,
)
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.bare_ground_driver import BareGroundDriverParameters
from clm_jax.physics.bare_ground_fluxes import (
    BareGroundEnergyState,
    BareGroundFluxState,
    BareGroundHumanIndexState,
    BareGroundPhotosynthesisState,
    BareGroundTemperatureState,
    BareGroundWaterState,
)
from clm_jax.physics.canopy_temperature import NUMBER_OF_GROUND_LEVELS
from clm_jax.physics.friction_velocity import FrictionVelocityState


class BareGroundDriverNativeData(NamedTuple):
    topology: SubgridTopology
    parameters: BareGroundDriverParameters
    state: BareGroundFluxState
    restart_path: Path


def _special(shape: tuple[int, ...]) -> jnp.ndarray:
    return jnp.full(shape, jnp.float64(CLM_SPECIAL_VALUE), dtype=jnp.float64)


def _initial_state(
    topology: SubgridTopology,
    vegetation_temperature_patch: jnp.ndarray,
) -> BareGroundFluxState:

    number_of_patches = topology.patch.gridcell.shape[0]
    patch_shape = (number_of_patches,)
    patch_level_shape = (number_of_patches, NUMBER_OF_GROUND_LEVELS)
    if vegetation_temperature_patch.shape != patch_shape:
        raise ValueError("restart T_VEG has an unexpected patch shape")
    vector = _special(patch_shape)
    level = _special(patch_level_shape)

    energy_values = []
    for name in BareGroundEnergyState._fields:
        energy_values.append(level if name == "root_resistance_patch_level" else vector)
    temperature_values = []
    for name in BareGroundTemperatureState._fields:
        temperature_values.append(
            vegetation_temperature_patch if name == "vegetation_temperature_patch" else vector
        )
    return BareGroundFluxState(
        energy=BareGroundEnergyState(*energy_values),
        water=BareGroundWaterState(*(vector for _ in BareGroundWaterState._fields)),
        temperature=BareGroundTemperatureState(*temperature_values),
        photosynthesis=BareGroundPhotosynthesisState(
            *(vector for _ in BareGroundPhotosynthesisState._fields)
        ),
        human_index=BareGroundHumanIndexState(
            *(vector for _ in BareGroundHumanIndexState._fields)
        ),
        friction_velocity=FrictionVelocityState(
            *(vector for _ in FrictionVelocityState._fields)
        ),
        effective_root_fraction_patch_level=level,
        ground_methane_conductance_patch=vector,
        aerodynamic_resistance_patch=vector,
    )


def load_bare_ground_driver_native_data(
    case: CimeCase,
    restart_path: str | Path,
) -> BareGroundDriverNativeData:

    restart = Path(restart_path).expanduser().resolve()
    temperature = load_canopy_temperature_driver_native_data(case, restart)
    restart_values = load_native_variables(restart, ["T_VEG"])
    vegetation_temperature = jnp.asarray(
        restart_values["T_VEG"],
        dtype=jnp.float64,
    )
    friction = read_case_namelist_group(case, "lnd_in", "friction_velocity")
    maximum_stable_zeta = float(friction.get("zetamaxstable", 0.5))
    if not 0.0 < maximum_stable_zeta <= 1.0:
        raise ValueError("zetamaxstable must be in (0, 1]")
    return BareGroundDriverNativeData(
        topology=temperature.topology,
        parameters=BareGroundDriverParameters(
            canopy_temperature=temperature.parameters,
            maximum_stable_zeta=maximum_stable_zeta,
        ),
        state=_initial_state(temperature.topology, vegetation_temperature),
        restart_path=restart,
    )


__all__ = [
    "BareGroundDriverNativeData",
    "load_bare_ground_driver_native_data",
]
