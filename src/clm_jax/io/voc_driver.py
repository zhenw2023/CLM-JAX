
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.diagnostics_and_accumulators import (
    load_diagnostics_and_accumulators_restart,
)
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.restart import SubgridTopology
from clm_jax.io.voc import (
    VOCConfiguration,
    load_mapped_isoprene_emission_factors_for_grid,
)
from clm_jax.physics.diagnostics_and_accumulators import ConfiguredAccumulatorState
from clm_jax.physics.voc import VOCParameters, VOCState, initialize_voc_state
from clm_jax.physics.voc_driver import VOCDriverInputs


class VOCDriverNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: VOCDriverInputs
    state: VOCState
    parameters: VOCParameters
    accumulators: ConfiguredAccumulatorState
    configuration: VOCConfiguration
    restart_path: Path


def load_voc_driver_native_data(
    case: CimeCase,
    restart_path: str | Path,
) -> VOCDriverNativeData:

    restart = Path(restart_path).expanduser().resolve()
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    accumulated = load_diagnostics_and_accumulators_restart(case, restart)
    topology = accumulated.topology
    configuration = VOCConfiguration.from_case(case)
    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    co2_type = str(general.get("co2_type", "")).strip().lower()
    if co2_type != "constant":
        raise ValueError("stage-310 adapter currently requires native co2_type='constant'")
    co2_ppmv = float(general.get("co2_ppmv", 0.0))
    if not 10.0 <= co2_ppmv <= 15_000.0:
        raise ValueError("configured CO2 mixing ratio lies outside native validity bounds")
    inputs = VOCDriverInputs(
        co2_volume_mixing_ratio_ppmv=jnp.asarray(co2_ppmv, dtype=jnp.float64),
        mapped_isoprene_emission_factor_grid=(
            load_mapped_isoprene_emission_factors_for_grid(case, topology)
        ),
    )
    return VOCDriverNativeData(
        topology=topology,
        inputs=inputs,
        state=initialize_voc_state(topology, configuration.parameters),
        parameters=configuration.parameters,
        accumulators=accumulated.state,
        configuration=configuration,
        restart_path=restart,
    )


__all__ = [
    "VOCDriverNativeData",
    "load_voc_driver_native_data",
]
