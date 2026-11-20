
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.canopy_fluxes_inputs import load_canopy_fluxes_pft_parameters
from clm_jax.io.canopy_fluxes_restart import load_canopy_fluxes_restart
from clm_jax.io.canopy_hydraulic_inputs import load_canopy_hydraulic_native_data
from clm_jax.io.canopy_temperature_driver import (
    load_canopy_temperature_driver_native_data,
)
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.luna import load_luna_capacity_parameters, load_luna_restart
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.radiation_driver import assert_same_topology
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.canopy_flux_driver import (
    CanopyFluxDriverInputs,
    CanopyFluxDriverParameters,
    CanopyFluxDriverState,
)
from clm_jax.physics.friction_velocity import FrictionVelocityState


class CanopyFluxDriverNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: CanopyFluxDriverInputs
    parameters: CanopyFluxDriverParameters
    state: CanopyFluxDriverState
    restart_path: Path


def _configured_boolean(group: dict[str, object], name: str, expected: bool) -> bool:
    value = group.get(name)
    if value is not expected:
        raise ValueError(f"stage-260 native case requires {name}={expected!r}")
    return expected


def load_canopy_flux_driver_native_data(
    case: CimeCase,
    restart_path: str | Path,
) -> CanopyFluxDriverNativeData:

    restart = Path(restart_path).expanduser().resolve()
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")

    temperature = load_canopy_temperature_driver_native_data(case, restart)
    flux = load_canopy_fluxes_restart(case, restart)
    hydraulic = load_canopy_hydraulic_native_data(case, restart)
    luna = load_luna_restart(case, restart)
    for other in (flux.topology, hydraulic.topology, luna.topology):
        assert_same_topology(temperature.topology, other)

    friction_group = read_case_namelist_group(case, "lnd_in", "friction_velocity")
    maximum_stable_zeta = float(friction_group.get("zetamaxstable", 0.5))
    if not 0.0 < maximum_stable_zeta <= 1.0:
        raise ValueError("zetamaxstable must be in (0, 1]")
    canopy_group = read_case_namelist_group(case, "lnd_in", "canopyfluxes_inparm")
    use_undercanopy_stability = _configured_boolean(
        canopy_group,
        "use_undercanopy_stability",
        False,
    )
    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    use_hydraulic_stress = _configured_boolean(general, "use_hydrstress", True)

    patch_shape = flux.vegetation_temperature_patch.shape
    special = jnp.full(patch_shape, jnp.float64(CLM_SPECIAL_VALUE))
    root_zone_wetness = load_native_variables(
        restart,
        ["btran2"],
        dtype=None,
        masked_floating_fill_value=float(CLM_SPECIAL_VALUE),
    )["btran2"]
    if root_zone_wetness.shape != patch_shape:
        raise ValueError("restart btran2 must contain every patch")
    return CanopyFluxDriverNativeData(
        topology=temperature.topology,
        inputs=CanopyFluxDriverInputs(
            pft=load_canopy_fluxes_pft_parameters(
                case,
                hydraulic.parameter_data_path,
            ),
            hydraulic=hydraulic.inputs,
            plant_parameters=hydraulic.plant_parameters,
            biochemical_parameters=hydraulic.biochemical_parameters,
            temperature_parameters=temperature.parameters,
            luna_template=luna,
        ),
        parameters=CanopyFluxDriverParameters(
            maximum_stable_zeta=maximum_stable_zeta,
            use_undercanopy_stability=use_undercanopy_stability,
            use_hydraulic_stress=use_hydraulic_stress,
            update_snow_canopy_water=True,
            luna_capacity=load_luna_capacity_parameters(
                case,
                hydraulic.parameter_data_path,
            ),
        ),
        state=CanopyFluxDriverState(
            active=flux.state,
            plant=hydraulic.initial_plant_state,
            vegetation_temperature_patch=flux.vegetation_temperature_patch,
            root_zone_wetness_patch=root_zone_wetness,
            friction_velocity=FrictionVelocityState(
                *(special for _ in FrictionVelocityState._fields)
            ),
            iteration_work_patch=jnp.zeros_like(
                flux.vegetation_temperature_patch, dtype=jnp.int32,
            ),
        ),
        restart_path=restart,
    )


__all__ = [
    "CanopyFluxDriverNativeData",
    "load_canopy_flux_driver_native_data",
]
