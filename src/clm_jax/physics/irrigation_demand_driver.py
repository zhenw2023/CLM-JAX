
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.atmospheric_driver import AtmosphericDriverResult
from clm_jax.physics.canopy_driver import (
    CanopyDriverInputs,
    CanopyDriverResult,
)
from clm_jax.physics.canopy_temperature import (
    NUMBER_OF_SNOW_LEVELS,
    NUMBER_OF_SOIL_LEVELS,
)
from clm_jax.physics.canopy_temperature_driver import CanopyTemperatureDriverResult
from clm_jax.physics.irrigation import (
    IrrigationDemandInputs,
    IrrigationDemandParameters,
    IrrigationDemandResult,
    IrrigationState,
    calculate_irrigation_demand,
)
from clm_jax.physics.numerics import fortran_divide, fortran_power
from clm_jax.physics.soil_moisture_stress import effective_soil_porosity
from clm_jax.trajectory import NoleapModelClock

WILTING_POINT_SOIL_MATRIC_POTENTIAL = -150_000.0


class IrrigationDemandDriverInputs(NamedTuple):

    irrigated_patch: jnp.ndarray
    saturated_water_content_column_level: jnp.ndarray
    relative_saturation_wilting_point_column_level: jnp.ndarray
    relative_saturation_target_column_level: jnp.ndarray
    soil_layer_depth_column_level: jnp.ndarray
    soil_layer_thickness_column_level: jnp.ndarray
    bedrock_soil_layer_count_column: jnp.ndarray
    river_water_volume_gridcell: jnp.ndarray
    gridcell_area_square_kilometers: jnp.ndarray
    runoff_prognostic: jnp.ndarray


class IrrigationDemandDriverResult(NamedTuple):

    inputs: IrrigationDemandInputs
    demand: IrrigationDemandResult
    state: IrrigationState


def clapp_hornberger_relative_saturation(
    soil_matric_potential: float | jnp.ndarray,
    saturated_suction_column_level: jnp.ndarray,
    clapp_hornberger_b_column_level: jnp.ndarray,
) -> jnp.ndarray:

    suction = jnp.asarray(saturated_suction_column_level, dtype=jnp.float64)
    exponent = jnp.asarray(clapp_hornberger_b_column_level, dtype=jnp.float64)
    if suction.shape != exponent.shape or suction.ndim != 2:
        raise ValueError("soil suction and Clapp-Hornberger b must be matching matrices")
    base = fortran_divide(-jnp.float64(soil_matric_potential), suction)
    inverse_exponent = fortran_divide(-jnp.float64(1.0), exponent)
    relative = fortran_power(base, inverse_exponent)
    relative = jnp.minimum(relative, jnp.float64(1.0))
    return jnp.maximum(relative, jnp.float64(0.0))


def _exposed_column_mask(
    topology: SubgridTopology,
    exposed_patch: jnp.ndarray,
) -> jnp.ndarray:
    patch = jnp.asarray(exposed_patch, dtype=jnp.bool_)
    if patch.shape != topology.patch.column.shape:
        raise ValueError("exposed vegetation mask must contain every native patch")
    number_of_columns = topology.column.gridcell.shape[0]
    return (
        jnp.zeros(number_of_columns, dtype=jnp.int32)
        .at[jnp.asarray(topology.patch.column, dtype=jnp.int32)]
        .max(patch.astype(jnp.int32))
        > jnp.int32(0)
    )


def assemble_irrigation_demand_inputs(
    topology: SubgridTopology,
    atmospheric: AtmosphericDriverResult,
    canopy_inputs: CanopyDriverInputs,
    canopy: CanopyDriverResult,
    temperature: CanopyTemperatureDriverResult,
    clock: NoleapModelClock,
    inputs: IrrigationDemandDriverInputs,
) -> IrrigationDemandInputs:

    exposed_patch = jnp.asarray(atmospheric.exposed_vegetation.exposed, dtype=jnp.bool_)
    exposed_column = _exposed_column_mask(topology, exposed_patch)
    temperature_inputs = temperature.inputs
    soil_slice = slice(
        NUMBER_OF_SNOW_LEVELS,
        NUMBER_OF_SNOW_LEVELS + NUMBER_OF_SOIL_LEVELS,
    )
    soil_temperature = jnp.asarray(
        temperature_inputs.soil_snow_temperature_column_level[:, soil_slice],
        dtype=jnp.float64,
    )
    soil_liquid = jnp.asarray(
        temperature_inputs.soil_liquid_water_column_level[:, soil_slice],
        dtype=jnp.float64,
    )
    soil_ice = jnp.asarray(
        temperature_inputs.soil_ice_water_column_level[:, soil_slice],
        dtype=jnp.float64,
    )
    thickness = jnp.asarray(inputs.soil_layer_thickness_column_level, dtype=jnp.float64)
    saturated = jnp.asarray(
        inputs.saturated_water_content_column_level,
        dtype=jnp.float64,
    )
    effective_porosity = effective_soil_porosity(
        thickness,
        saturated,
        soil_ice,
        exposed_column,
        saturated,
    )
    return IrrigationDemandInputs(
        exposed_vegetation_patch=exposed_patch,
        irrigated_patch=jnp.asarray(inputs.irrigated_patch, dtype=jnp.bool_),
        leaf_area_index=jnp.asarray(
            canopy_inputs.structure.exposed_leaf_area_index_patch,
            dtype=jnp.float64,
        ),
        soil_temperature=soil_temperature,
        effective_porosity=effective_porosity,
        soil_liquid_water=soil_liquid,
        relative_saturation_wilting_point=jnp.asarray(
            inputs.relative_saturation_wilting_point_column_level,
            dtype=jnp.float64,
        ),
        relative_saturation_target=jnp.asarray(
            inputs.relative_saturation_target_column_level,
            dtype=jnp.float64,
        ),
        soil_layer_depth=jnp.asarray(
            inputs.soil_layer_depth_column_level,
            dtype=jnp.float64,
        ),
        soil_layer_thickness=thickness,
        bedrock_soil_layer_count=jnp.asarray(
            inputs.bedrock_soil_layer_count_column,
            dtype=jnp.int32,
        ),
        river_water_volume=jnp.asarray(
            inputs.river_water_volume_gridcell,
            dtype=jnp.float64,
        ),
        gridcell_area_square_kilometers=jnp.asarray(
            inputs.gridcell_area_square_kilometers,
            dtype=jnp.float64,
        ),
        current_time_of_day_seconds=jnp.asarray(
            clock.seconds_into_day,
            dtype=jnp.int32,
        ),
        runoff_prognostic=jnp.asarray(inputs.runoff_prognostic, dtype=jnp.bool_),
    )


def irrigation_demand_driver_step(
    topology: SubgridTopology,
    atmospheric: AtmosphericDriverResult,
    canopy_inputs: CanopyDriverInputs,
    canopy: CanopyDriverResult,
    temperature: CanopyTemperatureDriverResult,
    clock: NoleapModelClock,
    inputs: IrrigationDemandDriverInputs,
    parameters: IrrigationDemandParameters,
) -> IrrigationDemandDriverResult:

    dynamic_inputs = assemble_irrigation_demand_inputs(
        topology,
        atmospheric,
        canopy_inputs,
        canopy,
        temperature,
        clock,
        inputs,
    )
    demand = calculate_irrigation_demand(
        topology,
        canopy.state.irrigation,
        dynamic_inputs,
        parameters=parameters,
    )
    return IrrigationDemandDriverResult(
        inputs=dynamic_inputs,
        demand=demand,
        state=demand.state,
    )


__all__ = [
    "IrrigationDemandDriverInputs",
    "IrrigationDemandDriverResult",
    "WILTING_POINT_SOIL_MATRIC_POTENTIAL",
    "assemble_irrigation_demand_inputs",
    "clapp_hornberger_relative_saturation",
    "irrigation_demand_driver_step",
]
