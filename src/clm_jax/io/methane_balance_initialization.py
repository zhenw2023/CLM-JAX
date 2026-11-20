
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.canopy_temperature_restart import soil_layer_geometry_20sl_8p5m
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_LAKE,
    SubgridTopology,
    configured_land_restart,
    load_subgrid_topology,
)
from clm_jax.physics.methane_balance_initialization import (
    NUMBER_OF_METHANE_SOIL_LEVELS,
    MethaneBalanceInitializationInputs,
    MethaneBalanceInitializationParameters,
    MethaneBalanceInitializationState,
)

_RESTART_VARIABLES = (
    "FINUNDATED",
    "CONC_CH4_SAT",
    "CONC_CH4_UNSAT",
)


class MethaneBalanceInitializationNativeData(NamedTuple):
    topology: SubgridTopology
    inputs: MethaneBalanceInitializationInputs
    state: MethaneBalanceInitializationState
    parameters: MethaneBalanceInitializationParameters
    restart_path: Path


def load_methane_balance_initialization_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
) -> MethaneBalanceInitializationNativeData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    methane_namelist = read_case_namelist_group(case, "lnd_in", "ch4par_in")
    if general.get("use_lch4") is not True:
        raise ValueError("methane balance initialization requires use_lch4=.true.")
    if general.get("soil_layerstruct") != "20SL_8.5m":
        raise ValueError("methane balance adapter requires soil_layerstruct='20SL_8.5m'")
    allow_lake_production = methane_namelist.get("allowlakeprod", False)
    if not isinstance(allow_lake_production, bool):
        raise TypeError("ch4par_in allowlakeprod must be logical")

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    dimensions = load_dimension_sizes(restart)
    if dimensions.get("levgrnd") != 25:
        raise ValueError("native methane balance adapter requires 25 ground levels")

    topology = load_subgrid_topology(restart)
    column_count = topology.column.gridcell.shape[0]
    if dimensions.get("column") != column_count:
        raise ValueError("restart column dimension disagrees with native topology")
    active = np.asarray(topology.column.active, dtype=bool)
    lake = active & (np.asarray(topology.column.landunit_type, dtype=np.int32) == LANDUNIT_LAKE)
    nonlake = active & ~lake

    raw = load_native_variables(
        restart,
        list(_RESTART_VARIABLES),
        dtype=np.float64,
    )
    inundated = np.asarray(raw["FINUNDATED"], dtype=np.float64)
    level_shape = (column_count, dimensions["levgrnd"])
    saturated = np.asarray(raw["CONC_CH4_SAT"], dtype=np.float64)
    unsaturated = np.asarray(raw["CONC_CH4_UNSAT"], dtype=np.float64)
    if inundated.shape != (column_count,):
        raise ValueError("restart FINUNDATED must contain every column")
    for name, value in (
        ("CONC_CH4_SAT", saturated),
        ("CONC_CH4_UNSAT", unsaturated),
    ):
        if value.shape != level_shape:
            raise ValueError(f"restart {name} must have shape {level_shape}")
        if not np.all(np.isfinite(value[active, :NUMBER_OF_METHANE_SOIL_LEVELS])):
            raise ValueError(f"restart {name} is non-finite on active methane columns")
        if np.any(value[active, :NUMBER_OF_METHANE_SOIL_LEVELS] < 0.0):
            raise ValueError(f"restart {name} is negative on active methane columns")
    if not np.all(np.isfinite(inundated[active])) or np.any(
        (inundated[nonlake] < 0.0) | (inundated[nonlake] > 1.0)
    ):
        raise ValueError("restart FINUNDATED lies outside [0, 1]")

    _, _, thickness = soil_layer_geometry_20sl_8p5m()
    thickness_column_level = np.broadcast_to(
        thickness[:NUMBER_OF_METHANE_SOIL_LEVELS],
        (column_count, NUMBER_OF_METHANE_SOIL_LEVELS),
    )
    return MethaneBalanceInitializationNativeData(
        topology=topology,
        inputs=MethaneBalanceInitializationInputs(
            nonlake_column=jnp.asarray(nonlake),
            lake_column=jnp.asarray(lake),
            inundated_fraction_column=jnp.asarray(inundated),
            soil_layer_thickness_column_level=jnp.asarray(thickness_column_level),
            saturated_methane_concentration_column_level=jnp.asarray(
                saturated[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
            ),
            unsaturated_methane_concentration_column_level=jnp.asarray(
                unsaturated[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
            ),
        ),
        state=MethaneBalanceInitializationState(
            beginning_methane_carbon_column=jnp.zeros(
                column_count,
                dtype=jnp.float64,
            )
        ),
        parameters=MethaneBalanceInitializationParameters(
            allow_lake_production=allow_lake_production
        ),
        restart_path=restart,
    )


__all__ = [
    "MethaneBalanceInitializationNativeData",
    "load_methane_balance_initialization_native_data",
]
