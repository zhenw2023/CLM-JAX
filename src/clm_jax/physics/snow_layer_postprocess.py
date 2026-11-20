
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import lax

from clm_jax.constants import FREEZING_TEMPERATURE_FRESHWATER
from clm_jax.io.restart import LANDUNIT_LAKE, SubgridTopology
from clm_jax.physics.numerics import fortran_add, fortran_multiply
from clm_jax.physics.snow_layer_combining import NUMBER_OF_SOIL_SNOW_INTERFACES
from clm_jax.physics.snow_water import (
    NUMBER_OF_SNOW_LEVELS,
    NUMBER_OF_SOIL_SNOW_LEVELS,
    SnowColumnMasks,
    build_snow_column_masks,
)

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase


class SnowLayerPostprocessParameters(NamedTuple):
    time_step_seconds: float = 1800.0

    @classmethod
    def from_case(cls, case: CimeCase) -> SnowLayerPostprocessParameters:
        from clm_jax.io.namelist import read_case_namelist_group  

        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        return cls(time_step_seconds=float(general["dtime"]))


DEFAULT_SNOW_LAYER_POSTPROCESS_PARAMETERS = SnowLayerPostprocessParameters()


class SnowLayerPostprocessInputs(NamedTuple):
    previous_snow_column: jnp.ndarray
    snow_depth_column: jnp.ndarray
    effective_snow_fraction_column: jnp.ndarray


class SnowLayerPostprocessState(NamedTuple):
    snow_layer_count_column: jnp.ndarray
    snow_persistence_column: jnp.ndarray
    soil_snow_thickness_column_level: jnp.ndarray
    soil_snow_node_depth_column_level: jnp.ndarray
    soil_snow_interface_depth_column_interface: jnp.ndarray
    soil_snow_temperature_column_level: jnp.ndarray
    soil_snow_ice_column_level: jnp.ndarray
    soil_snow_liquid_column_level: jnp.ndarray
    snow_ice_column: jnp.ndarray
    snow_liquid_column: jnp.ndarray
    column_average_snow_depth_column: jnp.ndarray
    snow_temperature_mass_product_column: jnp.ndarray
    top_snow_water_column: jnp.ndarray


class SnowLayerPostprocessDiagnostics(NamedTuple):
    rebuilt_masks: SnowColumnMasks
    active_nonlake_column: jnp.ndarray
    cleared_empty_layer: jnp.ndarray


class SnowLayerPostprocessResult(NamedTuple):
    state: SnowLayerPostprocessState
    diagnostics: SnowLayerPostprocessDiagnostics


def _validate_shapes(
    topology: SubgridTopology,
    inputs: SnowLayerPostprocessInputs,
    state: SnowLayerPostprocessState,
    parameters: SnowLayerPostprocessParameters,
) -> None:
    column_shape = topology.column.gridcell.shape
    if len(column_shape) != 1:
        raise ValueError("column topology must be a vector")
    number_of_columns = column_shape[0]
    level_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS)
    interface_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_INTERFACES)
    for name in SnowLayerPostprocessInputs._fields:
        if getattr(inputs, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")
    vector_fields = set(SnowLayerPostprocessState._fields) - {
        "soil_snow_thickness_column_level",
        "soil_snow_node_depth_column_level",
        "soil_snow_interface_depth_column_interface",
        "soil_snow_temperature_column_level",
        "soil_snow_ice_column_level",
        "soil_snow_liquid_column_level",
    }
    for name in vector_fields:
        if getattr(state, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")
    for name in (
        "soil_snow_thickness_column_level",
        "soil_snow_node_depth_column_level",
        "soil_snow_temperature_column_level",
        "soil_snow_ice_column_level",
        "soil_snow_liquid_column_level",
    ):
        if getattr(state, name).shape != level_shape:
            raise ValueError(f"{name} must have shape {level_shape}")
    if state.soil_snow_interface_depth_column_interface.shape != interface_shape:
        raise ValueError(
            "soil_snow_interface_depth_column_interface must have shape "
            f"{interface_shape}"
        )
    if parameters.time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")


def snow_layer_postprocess_active_case(
    topology: SubgridTopology,
    inputs: SnowLayerPostprocessInputs,
    state: SnowLayerPostprocessState,
    *,
    parameters: SnowLayerPostprocessParameters = (
        DEFAULT_SNOW_LAYER_POSTPROCESS_PARAMETERS
    ),
) -> SnowLayerPostprocessResult:

    _validate_shapes(topology, inputs, state, parameters)
    number_of_columns = topology.column.gridcell.shape[0]
    layer_count = jnp.asarray(state.snow_layer_count_column, dtype=jnp.int32)
    previous_snow = jnp.asarray(inputs.previous_snow_column, dtype=jnp.bool_)
    level_index = jnp.arange(NUMBER_OF_SNOW_LEVELS, dtype=jnp.int32)
    cleared_empty_layer = (
        previous_snow[:, None]
        & (layer_count[:, None] > jnp.int32(-NUMBER_OF_SNOW_LEVELS))
        & (level_index[None, :] <= layer_count[:, None] + jnp.int32(11))
    )

    def clear_snow_levels(values: jnp.ndarray) -> jnp.ndarray:
        snow_values = jnp.where(
            cleared_empty_layer,
            jnp.float64(0.0),
            values[:, :NUMBER_OF_SNOW_LEVELS],
        )
        return values.at[:, :NUMBER_OF_SNOW_LEVELS].set(snow_values)

    thickness = clear_snow_levels(
        jnp.asarray(state.soil_snow_thickness_column_level, dtype=jnp.float64)
    )
    node_depth = clear_snow_levels(
        jnp.asarray(state.soil_snow_node_depth_column_level, dtype=jnp.float64)
    )
    temperature = clear_snow_levels(
        jnp.asarray(state.soil_snow_temperature_column_level, dtype=jnp.float64)
    )
    ice = clear_snow_levels(
        jnp.asarray(state.soil_snow_ice_column_level, dtype=jnp.float64)
    )
    liquid = clear_snow_levels(
        jnp.asarray(state.soil_snow_liquid_column_level, dtype=jnp.float64)
    )
    interface_depth = jnp.asarray(
        state.soil_snow_interface_depth_column_interface,
        dtype=jnp.float64,
    )
    interface_depth = interface_depth.at[:, :NUMBER_OF_SNOW_LEVELS].set(
        jnp.where(
            cleared_empty_layer,
            jnp.float64(0.0),
            interface_depth[:, :NUMBER_OF_SNOW_LEVELS],
        )
    )

    rebuilt_masks = build_snow_column_masks(topology, layer_count)
    active_nonlake = jnp.asarray(topology.column.active, dtype=jnp.bool_) & (
        jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
        != jnp.int32(LANDUNIT_LAKE)
    )
    snow_persistence = jnp.asarray(
        state.snow_persistence_column,
        dtype=jnp.float64,
    )
    snow_persistence = jnp.where(
        rebuilt_masks.snow_column,
        fortran_add(
            snow_persistence,
            jnp.float64(parameters.time_step_seconds),
        ),
        jnp.where(
            rebuilt_masks.no_snow_column,
            jnp.float64(0.0),
            snow_persistence,
        ),
    )

    snow_ice = jnp.where(
        active_nonlake,
        jnp.float64(0.0),
        jnp.asarray(state.snow_ice_column, dtype=jnp.float64),
    )
    snow_liquid = jnp.where(
        active_nonlake,
        jnp.float64(0.0),
        jnp.asarray(state.snow_liquid_column, dtype=jnp.float64),
    )
    temperature_mass_product = jnp.where(
        active_nonlake,
        jnp.float64(0.0),
        jnp.asarray(
            state.snow_temperature_mass_product_column,
            dtype=jnp.float64,
        ),
    )
    top_index = layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS)

    def accumulate_level(
        snow_level: int,
        totals: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ):
        ice_total, liquid_total, temperature_total = totals
        active_level = rebuilt_masks.snow_column & (snow_level >= top_index)
        layer_ice = jnp.where(active_level, ice[:, snow_level], jnp.float64(0.0))
        layer_liquid = jnp.where(active_level, liquid[:, snow_level], jnp.float64(0.0))
        layer_temperature = jnp.where(active_level, temperature[:, snow_level], jnp.float64(0.0))
        updated_ice = fortran_add(ice_total, layer_ice)
        updated_liquid = fortran_add(liquid_total, layer_liquid)
        updated_temperature = fortran_add(
            temperature_total,
            fortran_multiply(layer_ice, layer_temperature),
        )
        updated_temperature = fortran_add(
            updated_temperature,
            fortran_multiply(
                layer_liquid,
                jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
            ),
        )
        return (
            jnp.where(active_level, updated_ice, ice_total),
            jnp.where(active_level, updated_liquid, liquid_total),
            jnp.where(active_level, updated_temperature, temperature_total),
        )

    snow_ice, snow_liquid, temperature_mass_product = lax.fori_loop(
        0,
        NUMBER_OF_SNOW_LEVELS,
        accumulate_level,
        (snow_ice, snow_liquid, temperature_mass_product),
    )
    column_average_snow_depth = fortran_multiply(
        jnp.asarray(inputs.snow_depth_column, dtype=jnp.float64),
        jnp.asarray(inputs.effective_snow_fraction_column, dtype=jnp.float64),
    )

    rows = jnp.arange(number_of_columns, dtype=jnp.int32)
    safe_top_index = jnp.clip(
        top_index,
        jnp.int32(0),
        jnp.int32(NUMBER_OF_SNOW_LEVELS - 1),
    )
    top_water = fortran_add(
        ice[rows, safe_top_index],
        liquid[rows, safe_top_index],
    )
    top_snow_water = jnp.where(
        rebuilt_masks.snow_column,
        top_water,
        jnp.where(
            rebuilt_masks.no_snow_column,
            jnp.float64(0.0),
            jnp.asarray(state.top_snow_water_column, dtype=jnp.float64),
        ),
    )
    return SnowLayerPostprocessResult(
        state=SnowLayerPostprocessState(
            snow_layer_count_column=layer_count,
            snow_persistence_column=snow_persistence,
            soil_snow_thickness_column_level=thickness,
            soil_snow_node_depth_column_level=node_depth,
            soil_snow_interface_depth_column_interface=interface_depth,
            soil_snow_temperature_column_level=temperature,
            soil_snow_ice_column_level=ice,
            soil_snow_liquid_column_level=liquid,
            snow_ice_column=snow_ice,
            snow_liquid_column=snow_liquid,
            column_average_snow_depth_column=column_average_snow_depth,
            snow_temperature_mass_product_column=temperature_mass_product,
            top_snow_water_column=top_snow_water,
        ),
        diagnostics=SnowLayerPostprocessDiagnostics(
            rebuilt_masks=rebuilt_masks,
            active_nonlake_column=active_nonlake,
            cleared_empty_layer=cleared_empty_layer,
        ),
    )
