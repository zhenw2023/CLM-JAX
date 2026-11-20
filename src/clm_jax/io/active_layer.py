
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.canopy_temperature_restart import soil_layer_geometry_20sl_8p5m
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.native_templates import native_template
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    RestartClock,
    SubgridTopology,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)
from clm_jax.physics.active_layer import (
    ActiveLayerInputs,
    ActiveLayerParameters,
    ActiveLayerState,
)
from clm_jax.physics.numerics import fortran_multiply

NUMBER_OF_GROUND_LEVELS = 25


class ActiveLayerNativeData(NamedTuple):

    topology: SubgridTopology
    soil_column: jnp.ndarray
    restart_soil_temperature_column_level: jnp.ndarray
    soil_depth_level: jnp.ndarray
    state: ActiveLayerState
    parameters: ActiveLayerParameters
    clock: RestartClock
    restart_path: Path


@native_template
def load_active_layer_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
) -> ActiveLayerNativeData:

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    dimensions = load_dimension_sizes(restart)
    if dimensions.get("levgrnd") != NUMBER_OF_GROUND_LEVELS:
        raise ValueError("active-layer adapter requires 25 ground levels")
    topology = load_subgrid_topology(restart)
    column_count = topology.column.gridcell.shape[0]
    landunit_type = np.asarray(topology.column.landunit_type, dtype=np.int32)
    soil_column = (landunit_type == LANDUNIT_SOIL) | (
        landunit_type == LANDUNIT_CROP
    )
    values = load_native_variables(
        restart,
        [
            "T_SOISNO",
            "altmax",
            "altmax_lastyear",
            "altmax_indx",
            "altmax_lastyear_indx",
        ],
        masked_integer_fill_value=-9999,
    )
    temperature = np.asarray(values["T_SOISNO"], dtype=np.float64)
    if temperature.ndim != 2 or temperature.shape[0] != column_count:
        raise ValueError("restart T_SOISNO must use the native column axis")
    if temperature.shape[1] < NUMBER_OF_GROUND_LEVELS:
        raise ValueError("restart T_SOISNO lacks the 25-level ground tail")
    ground_temperature = temperature[:, -NUMBER_OF_GROUND_LEVELS:]

    def real_vector(name: str) -> np.ndarray:
        value = np.asarray(values[name], dtype=np.float64)
        if value.shape != (column_count,):
            raise ValueError(f"restart {name} must contain every column")
        if not np.all(np.isfinite(value[soil_column])):
            raise ValueError(f"active soil columns contain invalid {name}")
        return value

    def index_vector(name: str) -> np.ndarray:
        value = np.asarray(values[name], dtype=np.int32)
        if value.shape != (column_count,):
            raise ValueError(f"restart {name} must contain every column")
        if np.any(
            (value[soil_column] < 0)
            | (value[soil_column] > NUMBER_OF_GROUND_LEVELS)
        ):
            raise ValueError(f"active soil columns contain invalid {name}")
        return value

    if not np.all(np.isfinite(ground_temperature[soil_column])):
        raise ValueError("active soil columns contain invalid ground temperature")
    maximum_depth = real_vector("altmax")
    prior_maximum_depth = real_vector("altmax_lastyear")
    maximum_index = index_vector("altmax_indx")
    prior_maximum_index = index_vector("altmax_lastyear_indx")
    clock = load_restart_clock(restart)
    soil_depth, _, _ = soil_layer_geometry_20sl_8p5m()
    return ActiveLayerNativeData(
        topology=topology,
        soil_column=jnp.asarray(soil_column),
        restart_soil_temperature_column_level=jnp.asarray(ground_temperature),
        soil_depth_level=jnp.asarray(soil_depth, dtype=jnp.float64),
        state=ActiveLayerState(
            thaw_depth_column=jnp.zeros(column_count, dtype=jnp.float64),
            maximum_thaw_depth_column=jnp.asarray(maximum_depth),
            prior_year_maximum_thaw_depth_column=jnp.asarray(prior_maximum_depth),
            thaw_depth_index_column=jnp.zeros(column_count, dtype=jnp.int32),
            maximum_thaw_depth_index_column=jnp.asarray(maximum_index),
            prior_year_maximum_thaw_depth_index_column=jnp.asarray(
                prior_maximum_index,
            ),
        ),
        parameters=ActiveLayerParameters(time_step_seconds=clock.step_seconds),
        clock=clock,
        restart_path=restart,
    )


def assemble_active_layer_inputs(
    native: ActiveLayerNativeData,
    *,
    soil_temperature_column_level: jnp.ndarray | None = None,
    month: int | jnp.ndarray | None = None,
    day_of_month: int | jnp.ndarray | None = None,
    seconds_of_day: int | jnp.ndarray | None = None,
) -> ActiveLayerInputs:

    temperature = (
        native.restart_soil_temperature_column_level
        if soil_temperature_column_level is None
        else jnp.asarray(soil_temperature_column_level, dtype=jnp.float64)
    )
    column_count = native.soil_column.shape[0]
    if temperature.shape != (column_count, NUMBER_OF_GROUND_LEVELS):
        raise ValueError("same-step ground temperature has the wrong shape")
    date = native.clock.current_date
    default_month = (date // 100) % 100
    default_day = date % 100
    latitude_radians = fortran_multiply(
        jnp.asarray(native.topology.grid.latitude_degrees, dtype=jnp.float64),
        jnp.float64(np.pi / np.float64(180.0)),
    )
    return ActiveLayerInputs(
        soil_column=native.soil_column,
        column_gridcell_index=native.topology.column.gridcell,
        gridcell_latitude_radians=latitude_radians,
        soil_temperature_column_level=temperature,
        soil_depth_level=native.soil_depth_level,
        month=default_month if month is None else month,
        day_of_month=default_day if day_of_month is None else day_of_month,
        seconds_of_day=(
            native.clock.current_time_of_day
            if seconds_of_day is None
            else seconds_of_day
        ),
    )
