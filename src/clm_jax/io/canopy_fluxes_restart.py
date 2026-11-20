
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.luna import load_luna_restart
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import SubgridTopology, configured_land_restart
from clm_jax.physics.canopy_fluxes import (
    CanopyEnergyInitialState,
    CanopyFastHumanStressState,
    CanopyFluxesActiveState,
    CanopyFluxesPostprocessState,
)
from clm_jax.physics.canopy_temperature import CanopyTemperatureState
from clm_jax.physics.friction_velocity import FrictionVelocityState
from clm_jax.physics.ozone import OzoneStressState
from clm_jax.physics.photosynthesis import PhotosynthesisTotalState


class CanopyFluxesRestartData(NamedTuple):

    topology: SubgridTopology
    state: CanopyFluxesActiveState
    vegetation_temperature_patch: jnp.ndarray


def _finite_restart_float(value: jnp.ndarray) -> jnp.ndarray:
    value = jnp.asarray(value, dtype=jnp.float64)
    return jnp.where(jnp.isnan(value), jnp.float64(CLM_SPECIAL_VALUE), value)


def load_canopy_fluxes_restart(
    case: CimeCase,
    restart_path: str | Path | None = None,
) -> CanopyFluxesRestartData:

    luna = load_luna_restart(case, restart_path)
    restart = Path(restart_path).expanduser().resolve() if restart_path is not None else None
    if restart is None:
        restart = configured_land_restart(case)
    restart_state = load_native_variables(
        restart,
        ["H2OCAN", "LIQCAN", "SNOCAN", "T_VEG"],
    )

    patch_count = luna.topology.patch.gridcell.shape[0]
    patch_shape = (patch_count,)
    for name, value in restart_state.items():
        if value.shape != patch_shape:
            raise ValueError(f"restart variable {name!r} has unexpected shape {value.shape}")
    special = jnp.full(patch_shape, jnp.float64(CLM_SPECIAL_VALUE))
    human_stress = CanopyFastHumanStressState(
        *(special for _ in CanopyFastHumanStressState._fields)
    )
    postprocess_values = []
    for name in CanopyFluxesPostprocessState._fields:
        if name == "human_stress":
            postprocess_values.append(human_stress)
        elif name == "canopy_water_patch":
            postprocess_values.append(_finite_restart_float(restart_state["H2OCAN"]))
        elif name == "liquid_canopy_water_patch":
            postprocess_values.append(_finite_restart_float(restart_state["LIQCAN"]))
        elif name == "snow_canopy_water_patch":
            postprocess_values.append(_finite_restart_float(restart_state["SNOCAN"]))
        else:
            postprocess_values.append(special)

    photosynthesis = PhotosynthesisTotalState(*(special for _ in PhotosynthesisTotalState._fields))
    ozone = OzoneStressState(
        *(jnp.ones(patch_shape, dtype=jnp.float64) for _ in OzoneStressState._fields)
    )
    return CanopyFluxesRestartData(
        topology=luna.topology,
        state=CanopyFluxesActiveState(
            postprocess=CanopyFluxesPostprocessState(*postprocess_values),
            photosynthesis_total=photosynthesis,
            ozone_stress=ozone,
            luna_daily=luna.daily,
            luna_ten_day=luna.ten_day,
            luna_capacity=luna.capacity,
        ),
        vegetation_temperature_patch=_finite_restart_float(restart_state["T_VEG"]),
    )


def assemble_canopy_energy_initial_state(
    restart: CanopyFluxesRestartData,
    canopy_temperature: CanopyTemperatureState,
) -> CanopyEnergyInitialState:

    patch_count = restart.topology.patch.gridcell.shape[0]
    expected_shape = (patch_count,)
    dynamic = (
        restart.vegetation_temperature_patch,
        canopy_temperature.patch.displacement_height_patch,
        canopy_temperature.patch.vegetation_momentum_roughness_patch,
    )
    if any(jnp.shape(value) != expected_shape for value in dynamic):
        raise ValueError(
            "CanopyFluxes energy entry fields must share the restart patch dimension"
        )
    special = jnp.full(expected_shape, jnp.float64(CLM_SPECIAL_VALUE))
    return CanopyEnergyInitialState(
        vegetation_temperature_patch=jnp.asarray(dynamic[0], dtype=jnp.float64),
        displacement_height_patch=jnp.asarray(dynamic[1], dtype=jnp.float64),
        vegetation_momentum_roughness_patch=jnp.asarray(dynamic[2], dtype=jnp.float64),
        friction_velocity=FrictionVelocityState(
            *(special for _ in FrictionVelocityState._fields)
        ),
    )
