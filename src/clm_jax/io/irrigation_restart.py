
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import (
    SubgridTopology,
    configured_land_restart,
    load_subgrid_topology,
)
from clm_jax.physics.irrigation import IrrigationState


class IrrigationRestartData(NamedTuple):
    topology: SubgridTopology
    state: IrrigationState


def _optional_patch_variable(
    restart: Path,
    name: str,
    number_of_patches: int,
) -> jnp.ndarray | None:
    try:
        value = load_native_variables(restart, [name])[name]
    except KeyError:
        return None
    if value.shape != (number_of_patches,):
        return None
    return value


def load_irrigation_restart(
    case: CimeCase,
    restart_path: str | Path | None = None,
) -> IrrigationRestartData:

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")

    topology = load_subgrid_topology(restart)
    number_of_patches = topology.patch.column.shape[0]
    steps = _optional_patch_variable(
        restart,
        "n_irrig_steps_left",
        number_of_patches,
    )
    rate = _optional_patch_variable(restart, "irrig_rate", number_of_patches)
    demand_rate = _optional_patch_variable(
        restart,
        "irrig_rate_demand",
        number_of_patches,
    )

    if steps is None:
        steps = jnp.zeros(number_of_patches, dtype=jnp.int32)
    else:
        steps = jnp.asarray(steps, dtype=jnp.int32)
    if rate is None:
        rate = jnp.zeros(number_of_patches, dtype=jnp.float64)
    else:
        rate = jnp.asarray(rate, dtype=jnp.float64)
        rate = jnp.where(jnp.isnan(rate), jnp.float64(CLM_SPECIAL_VALUE), rate)
    if demand_rate is None:
        demand_rate = rate
    else:
        demand_rate = jnp.asarray(demand_rate, dtype=jnp.float64)
        demand_rate = jnp.where(
            jnp.isnan(demand_rate),
            jnp.float64(CLM_SPECIAL_VALUE),
            demand_rate,
        )

    return IrrigationRestartData(
        topology=topology,
        state=IrrigationState(
            rate_patch=rate,
            demand_rate_patch=demand_rate,
            steps_left_patch=steps,
        ),
    )
