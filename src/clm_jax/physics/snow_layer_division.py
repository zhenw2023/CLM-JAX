
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from clm_jax.constants import FREEZING_TEMPERATURE_FRESHWATER
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.numerics import (
    fortran_divide,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.snow_layer_combining import (
    NUMBER_OF_SOIL_SNOW_INTERFACES,
    _combine_mass_and_enthalpy,
)
from clm_jax.physics.snow_water import (
    NUMBER_OF_SNOW_LEVELS,
    NUMBER_OF_SOIL_SNOW_LEVELS,
    SnowWaterAerosolState,
    build_snow_column_masks,
)

MAXIMUM_BOTTOM_LAYER_THICKNESS = (
    0.03,
    0.07,
    0.18,
    0.41,
    0.88,
    1.83,
    3.74,
    7.57,
    15.24,
    30.59,
    61.3,
    float(np.finfo(np.float64).max),
)
MAXIMUM_UPPER_LAYER_THICKNESS = (
    0.02,
    0.05,
    0.11,
    0.23,
    0.47,
    0.95,
    1.91,
    3.83,
    7.67,
    15.35,
    30.71,
    float(np.finfo(np.float64).max),
)
MINIMUM_SNOW_GRAIN_RADIUS = 54.526
MAXIMUM_SNOW_GRAIN_RADIUS = 1500.0


class SnowLayerDivisionParameters(NamedTuple):

    maximum_bottom_layer_thickness: tuple[float, ...] = (
        MAXIMUM_BOTTOM_LAYER_THICKNESS
    )
    maximum_upper_layer_thickness: tuple[float, ...] = (
        MAXIMUM_UPPER_LAYER_THICKNESS
    )
    minimum_snow_grain_radius: float = MINIMUM_SNOW_GRAIN_RADIUS
    maximum_snow_grain_radius: float = MAXIMUM_SNOW_GRAIN_RADIUS
    lake_snow_layer_offset: float = 0.03


DEFAULT_SNOW_LAYER_DIVISION_PARAMETERS = SnowLayerDivisionParameters()


class SnowLayerDivisionState(NamedTuple):
    snow_layer_count_column: jnp.ndarray
    effective_snow_fraction_column: jnp.ndarray
    soil_snow_thickness_column_level: jnp.ndarray
    soil_snow_node_depth_column_level: jnp.ndarray
    soil_snow_interface_depth_column_interface: jnp.ndarray
    soil_snow_temperature_column_level: jnp.ndarray
    soil_snow_ice_column_level: jnp.ndarray
    soil_snow_liquid_column_level: jnp.ndarray
    snow_grain_radius_column_level: jnp.ndarray
    aerosol: SnowWaterAerosolState


class SnowLayerDivisionDiagnostics(NamedTuple):
    snow_column: jnp.ndarray
    bottom_split_count_column: jnp.ndarray
    excess_transfer_count_column: jnp.ndarray


class SnowLayerDivisionResult(NamedTuple):
    state: SnowLayerDivisionState
    diagnostics: SnowLayerDivisionDiagnostics


def _validate_shapes(
    topology: SubgridTopology,
    state: SnowLayerDivisionState,
    parameters: SnowLayerDivisionParameters,
) -> None:
    column_shape = topology.column.gridcell.shape
    if len(column_shape) != 1:
        raise ValueError("column topology must be a vector")
    number_of_columns = column_shape[0]
    level_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS)
    interface_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_INTERFACES)
    snow_shape = (number_of_columns, NUMBER_OF_SNOW_LEVELS)
    for name in (
        "snow_layer_count_column",
        "effective_snow_fraction_column",
    ):
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
    if state.snow_grain_radius_column_level.shape != snow_shape:
        raise ValueError(
            f"snow_grain_radius_column_level must have shape {snow_shape}"
        )
    for name in SnowWaterAerosolState._fields[:-1]:
        if getattr(state.aerosol, name).shape != snow_shape:
            raise ValueError(f"aerosol.{name} must have shape {snow_shape}")
    for name in state.aerosol.deposition._fields:
        if getattr(state.aerosol.deposition, name).shape != column_shape:
            raise ValueError(f"aerosol.deposition.{name} must have shape {column_shape}")
    if len(parameters.maximum_bottom_layer_thickness) != NUMBER_OF_SNOW_LEVELS:
        raise ValueError(
            "maximum_bottom_layer_thickness must contain "
            f"{NUMBER_OF_SNOW_LEVELS} values"
        )
    if len(parameters.maximum_upper_layer_thickness) != NUMBER_OF_SNOW_LEVELS:
        raise ValueError(
            "maximum_upper_layer_thickness must contain "
            f"{NUMBER_OF_SNOW_LEVELS} values"
        )
    if parameters.minimum_snow_grain_radius <= 0.0:
        raise ValueError("minimum_snow_grain_radius must be positive")
    if (
        parameters.maximum_snow_grain_radius
        <= parameters.minimum_snow_grain_radius
    ):
        raise ValueError("maximum_snow_grain_radius must exceed the minimum")
    if parameters.lake_snow_layer_offset < 0.0:
        raise ValueError("lake_snow_layer_offset must be nonnegative")


def _gather(values: jnp.ndarray, index: jnp.ndarray) -> jnp.ndarray:
    rows = jnp.arange(values.shape[0], dtype=jnp.int32)
    return values[rows, index]


def _write(
    values: jnp.ndarray,
    index: jnp.ndarray,
    replacement: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    from clm_jax.physics.solver_scalarization import _native_tile_trace
    if _native_tile_trace.get():
        index = jnp.where(index < 0, index + values.shape[1], index)
        levels = jnp.arange(values.shape[1], dtype=jnp.int32)
        chosen = (levels[None, :] == index[:, None]) & mask[:, None]
        chosen = chosen.reshape((*chosen.shape, *((1,) * (values.ndim - 2))))
        return jnp.where(chosen, replacement[:, None, ...], values)
    rows = jnp.arange(values.shape[0], dtype=jnp.int32)
    current = values[rows, index]
    while mask.ndim < replacement.ndim:
        mask = mask[..., None]
    return values.at[rows, index].set(jnp.where(mask, replacement, current))


def divide_snow_layers_active_case(
    topology: SubgridTopology,
    state: SnowLayerDivisionState,
    *,
    parameters: SnowLayerDivisionParameters = DEFAULT_SNOW_LAYER_DIVISION_PARAMETERS,
    is_lake: bool = False,
) -> SnowLayerDivisionResult:
    if jax.default_backend() == 'gpu' and state.snow_layer_count_column.size > 16384:
        from clm_jax.physics.snow_column_filter import filtered_snow_transition

        def transition(local_topology, local_state):
            from clm_jax.physics.snow_layers_gpu import snow_layers_gpu

            return snow_layers_gpu(
                'division', local_topology, local_state, parameters,
                lake_only=is_lake,
            )

        return filtered_snow_transition(transition, topology, state, lake_only=is_lake)
    return _divide_snow_layers_active_case(
        topology, state, parameters=parameters, is_lake=is_lake,
    )


def _divide_snow_layers_active_case(
    topology: SubgridTopology,
    state: SnowLayerDivisionState,
    *,
    parameters: SnowLayerDivisionParameters = DEFAULT_SNOW_LAYER_DIVISION_PARAMETERS,
    is_lake: bool = False,
) -> SnowLayerDivisionResult:

    _validate_shapes(topology, state, parameters)
    number_of_columns = topology.column.gridcell.shape[0]
    snow_column = build_snow_column_masks(
        topology,
        state.snow_layer_count_column,
        lake_only=is_lake,
    ).snow_column
    layer_count = jnp.asarray(state.snow_layer_count_column, dtype=jnp.int32)
    original_number_of_layers = jnp.abs(layer_count)
    original_top_index = layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS)
    snow_fraction = jnp.asarray(
        state.effective_snow_fraction_column,
        dtype=jnp.float64,
    )
    thickness = jnp.asarray(
        state.soil_snow_thickness_column_level,
        dtype=jnp.float64,
    )
    node_depth = jnp.asarray(
        state.soil_snow_node_depth_column_level,
        dtype=jnp.float64,
    )
    interface_depth = jnp.asarray(
        state.soil_snow_interface_depth_column_interface,
        dtype=jnp.float64,
    )
    temperature = jnp.asarray(
        state.soil_snow_temperature_column_level,
        dtype=jnp.float64,
    )
    ice = jnp.asarray(state.soil_snow_ice_column_level, dtype=jnp.float64)
    liquid = jnp.asarray(state.soil_snow_liquid_column_level, dtype=jnp.float64)
    radius = jnp.asarray(
        state.snow_grain_radius_column_level,
        dtype=jnp.float64,
    )
    aerosol = jnp.stack(
        [
            jnp.asarray(getattr(state.aerosol, name), dtype=jnp.float64)
            for name in SnowWaterAerosolState._fields[:-1]
        ],
        axis=-1,
    )

    compact_shape = (number_of_columns, NUMBER_OF_SNOW_LEVELS)
    compact_thickness = jnp.zeros(compact_shape, dtype=jnp.float64)
    compact_temperature = jnp.zeros(compact_shape, dtype=jnp.float64)
    compact_ice = jnp.zeros(compact_shape, dtype=jnp.float64)
    compact_liquid = jnp.zeros(compact_shape, dtype=jnp.float64)
    compact_radius = jnp.zeros(compact_shape, dtype=jnp.float64)
    compact_aerosol = jnp.zeros(
        (*compact_shape, len(SnowWaterAerosolState._fields) - 1),
        dtype=jnp.float64,
    )

    def load_compact_level(
        compact_index: int,
        compact: tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ],
    ):
        (
            work_thickness,
            work_temperature,
            work_ice,
            work_liquid,
            work_radius,
            work_aerosol,
        ) = compact
        native_index = jnp.clip(
            original_top_index + compact_index,
            jnp.int32(0),
            jnp.int32(NUMBER_OF_SNOW_LEVELS - 1),
        )
        active = snow_column & (compact_index < original_number_of_layers)
        native_thickness = jnp.where(active, _gather(thickness, native_index), 0.)
        compact_layer_thickness = (
            native_thickness
            if is_lake
            else fortran_multiply(snow_fraction, native_thickness)
        )
        work_thickness = work_thickness.at[:, compact_index].set(
            jnp.where(
                active,
                compact_layer_thickness,
                work_thickness[:, compact_index],
            )
        )
        work_temperature = work_temperature.at[:, compact_index].set(
            jnp.where(
                active,
                _gather(temperature, native_index),
                work_temperature[:, compact_index],
            )
        )
        work_ice = work_ice.at[:, compact_index].set(
            jnp.where(
                active,
                _gather(ice, native_index),
                work_ice[:, compact_index],
            )
        )
        work_liquid = work_liquid.at[:, compact_index].set(
            jnp.where(
                active,
                _gather(liquid, native_index),
                work_liquid[:, compact_index],
            )
        )
        work_radius = work_radius.at[:, compact_index].set(
            jnp.where(
                active,
                _gather(radius, native_index),
                work_radius[:, compact_index],
            )
        )
        work_aerosol = work_aerosol.at[:, compact_index, :].set(
            jnp.where(
                active[:, None],
                _gather(aerosol, native_index),
                work_aerosol[:, compact_index, :],
            )
        )
        return (
            work_thickness,
            work_temperature,
            work_ice,
            work_liquid,
            work_radius,
            work_aerosol,
        )

    (
        compact_thickness,
        compact_temperature,
        compact_ice,
        compact_liquid,
        compact_radius,
        compact_aerosol,
    ) = lax.fori_loop(
        0,
        NUMBER_OF_SNOW_LEVELS,
        load_compact_level,
        (
            compact_thickness,
            compact_temperature,
            compact_ice,
            compact_liquid,
            compact_radius,
            compact_aerosol,
        ),
    )

    maximum_bottom = jnp.asarray(
        parameters.maximum_bottom_layer_thickness,
        dtype=jnp.float64,
    )
    maximum_upper = jnp.asarray(
        parameters.maximum_upper_layer_thickness,
        dtype=jnp.float64,
    )
    lake_offset = (
        jnp.float64(parameters.lake_snow_layer_offset)
        if is_lake
        else jnp.float64(0.0)
    )
    bottom_offset = (
        fortran_multiply(jnp.float64(2.0), lake_offset)
        if is_lake
        else jnp.float64(0.0)
    )
    zero_count = jnp.zeros(number_of_columns, dtype=jnp.int32)

    def process_compact_level(
        compact_index: int,
        carry: tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ],
    ):
        (
            number_of_layers,
            work_thickness,
            work_temperature,
            work_ice,
            work_liquid,
            work_radius,
            work_aerosol,
            split_count,
            transfer_count,
        ) = carry
        loop_active = snow_column & (compact_index < number_of_layers)
        bottom = loop_active & (compact_index == number_of_layers - jnp.int32(1))
        bottom_limit = fortran_add(
            maximum_bottom[compact_index],
            bottom_offset,
        )
        split = bottom & (
            work_thickness[:, compact_index] > bottom_limit
        )

        old_thickness = work_thickness[:, compact_index]
        old_temperature = work_temperature[:, compact_index]
        old_ice = work_ice[:, compact_index]
        old_liquid = work_liquid[:, compact_index]
        old_aerosol = work_aerosol[:, compact_index, :]
        half_thickness = fortran_divide(old_thickness, jnp.float64(2.0))
        half_ice = fortran_divide(old_ice, jnp.float64(2.0))
        half_liquid = fortran_divide(old_liquid, jnp.float64(2.0))
        half_aerosol = fortran_divide(old_aerosol, jnp.float64(2.0))
        work_thickness = work_thickness.at[:, compact_index].set(
            jnp.where(split, half_thickness, old_thickness)
        )
        work_thickness = work_thickness.at[:, compact_index + 1].set(
            jnp.where(
                split,
                half_thickness,
                work_thickness[:, compact_index + 1],
            )
        )
        work_ice = work_ice.at[:, compact_index].set(
            jnp.where(split, half_ice, old_ice)
        )
        work_ice = work_ice.at[:, compact_index + 1].set(
            jnp.where(split, half_ice, work_ice[:, compact_index + 1])
        )
        work_liquid = work_liquid.at[:, compact_index].set(
            jnp.where(split, half_liquid, old_liquid)
        )
        work_liquid = work_liquid.at[:, compact_index + 1].set(
            jnp.where(split, half_liquid, work_liquid[:, compact_index + 1])
        )
        work_aerosol = work_aerosol.at[:, compact_index, :].set(
            jnp.where(split[:, None], half_aerosol, old_aerosol)
        )
        work_aerosol = work_aerosol.at[:, compact_index + 1, :].set(
            jnp.where(
                split[:, None],
                half_aerosol,
                work_aerosol[:, compact_index + 1, :],
            )
        )
        work_radius = work_radius.at[:, compact_index + 1].set(
            jnp.where(
                split,
                work_radius[:, compact_index],
                work_radius[:, compact_index + 1],
            )
        )

        upper_index = jnp.maximum(compact_index - jnp.int32(1), jnp.int32(0))
        upper_temperature = work_temperature[:, upper_index]
        upper_thickness = work_thickness[:, upper_index]
        gradient_denominator = fortran_divide(
            fortran_add(
                upper_thickness,
                fortran_multiply(
                    jnp.float64(2.0),
                    work_thickness[:, compact_index],
                ),
            ),
            jnp.float64(2.0),
        )
        safe_gradient_denominator = jnp.where(
            gradient_denominator != jnp.float64(0.0),
            gradient_denominator,
            jnp.float64(1.0),
        )
        temperature_gradient = fortran_divide(
            fortran_add(upper_temperature, -old_temperature),
            safe_gradient_denominator,
        )
        temperature_delta = fortran_divide(
            fortran_multiply(
                temperature_gradient,
                work_thickness[:, compact_index],
            ),
            jnp.float64(2.0),
        )
        proposed_lower_temperature = fortran_add(
            old_temperature,
            -temperature_delta,
        )
        first_layer = compact_index == jnp.int32(0)
        bounded_lower_temperature = jnp.where(
            first_layer
            | (
                proposed_lower_temperature
                >= jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)
            ),
            old_temperature,
            proposed_lower_temperature,
        )
        adjusted_upper_temperature = jnp.where(
            (~first_layer)
            & (
                proposed_lower_temperature
                < jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)
            ),
            fortran_add(old_temperature, temperature_delta),
            old_temperature,
        )
        work_temperature = work_temperature.at[:, compact_index + 1].set(
            jnp.where(
                split,
                bounded_lower_temperature,
                work_temperature[:, compact_index + 1],
            )
        )
        work_temperature = work_temperature.at[:, compact_index].set(
            jnp.where(split, adjusted_upper_temperature, old_temperature)
        )
        number_of_layers = jnp.where(
            split,
            number_of_layers + jnp.int32(1),
            number_of_layers,
        )
        split_count = split_count + split.astype(jnp.int32)

        has_below = loop_active & (
            compact_index < number_of_layers - jnp.int32(1)
        )
        upper_limit = fortran_add(maximum_upper[compact_index], lake_offset)
        transfer = has_below & (
            work_thickness[:, compact_index] > upper_limit
        )
        current_thickness = work_thickness[:, compact_index]
        excess_thickness = fortran_add(
            fortran_add(current_thickness, -maximum_upper[compact_index]),
            -lake_offset,
        )
        safe_current_thickness = jnp.where(
            current_thickness != jnp.float64(0.0),
            current_thickness,
            jnp.float64(1.0),
        )
        excess_fraction = fortran_divide(
            excess_thickness,
            safe_current_thickness,
        )
        retained_fraction = fortran_divide(
            upper_limit,
            safe_current_thickness,
        )
        current_ice = work_ice[:, compact_index]
        current_liquid = work_liquid[:, compact_index]
        current_aerosol = work_aerosol[:, compact_index, :]
        excess_ice = fortran_multiply(excess_fraction, current_ice)
        excess_liquid = fortran_multiply(excess_fraction, current_liquid)
        excess_aerosol = fortran_multiply(
            excess_fraction[:, None],
            current_aerosol,
        )
        retained_ice = fortran_multiply(retained_fraction, current_ice)
        retained_liquid = fortran_multiply(retained_fraction, current_liquid)
        retained_aerosol = fortran_multiply(
            retained_fraction[:, None],
            current_aerosol,
        )
        work_ice = work_ice.at[:, compact_index].set(
            jnp.where(transfer, retained_ice, current_ice)
        )
        work_liquid = work_liquid.at[:, compact_index].set(
            jnp.where(transfer, retained_liquid, current_liquid)
        )
        work_aerosol = work_aerosol.at[:, compact_index, :].set(
            jnp.where(transfer[:, None], retained_aerosol, current_aerosol)
        )
        work_thickness = work_thickness.at[:, compact_index].set(
            jnp.where(
                transfer,
                upper_limit,
                current_thickness,
            )
        )

        lower_index = compact_index + jnp.int32(1)
        lower_thickness = work_thickness[:, lower_index]
        lower_liquid = work_liquid[:, lower_index]
        lower_ice = work_ice[:, lower_index]
        lower_temperature = work_temperature[:, lower_index]
        lower_aerosol = work_aerosol[:, lower_index, :]
        lower_radius = work_radius[:, lower_index]
        current_radius = work_radius[:, compact_index]
        lower_mass = fortran_add(lower_liquid, lower_ice)
        excess_mass = fortran_add(excess_liquid, excess_ice)
        radius_numerator = fortran_add(
            fortran_multiply(lower_radius, lower_mass),
            fortran_multiply(current_radius, excess_mass),
        )
        radius_denominator = fortran_add(lower_mass, excess_mass)
        safe_radius_denominator = jnp.where(
            radius_denominator != jnp.float64(0.0),
            radius_denominator,
            jnp.float64(1.0),
        )
        combined_radius = fortran_divide(
            radius_numerator,
            safe_radius_denominator,
        )
        combined_radius = jnp.where(
            combined_radius > jnp.float64(parameters.maximum_snow_grain_radius),
            jnp.float64(parameters.maximum_snow_grain_radius),
            jnp.where(
                combined_radius
                < jnp.float64(parameters.minimum_snow_grain_radius),
                jnp.float64(parameters.minimum_snow_grain_radius),
                combined_radius,
            ),
        )
        (
            combined_thickness,
            combined_liquid,
            combined_ice,
            combined_temperature,
        ) = _combine_mass_and_enthalpy(
            lower_thickness,
            lower_liquid,
            lower_ice,
            lower_temperature,
            excess_thickness,
            excess_liquid,
            excess_ice,
            work_temperature[:, compact_index],
        )
        combined_aerosol = fortran_add(lower_aerosol, excess_aerosol)
        work_thickness = work_thickness.at[:, lower_index].set(
            jnp.where(transfer, combined_thickness, lower_thickness)
        )
        work_liquid = work_liquid.at[:, lower_index].set(
            jnp.where(transfer, combined_liquid, lower_liquid)
        )
        work_ice = work_ice.at[:, lower_index].set(
            jnp.where(transfer, combined_ice, lower_ice)
        )
        work_temperature = work_temperature.at[:, lower_index].set(
            jnp.where(transfer, combined_temperature, lower_temperature)
        )
        work_radius = work_radius.at[:, lower_index].set(
            jnp.where(transfer, combined_radius, lower_radius)
        )
        work_aerosol = work_aerosol.at[:, lower_index, :].set(
            jnp.where(transfer[:, None], combined_aerosol, lower_aerosol)
        )
        transfer_count = transfer_count + transfer.astype(jnp.int32)
        return (
            number_of_layers,
            work_thickness,
            work_temperature,
            work_ice,
            work_liquid,
            work_radius,
            work_aerosol,
            split_count,
            transfer_count,
        )

    (
        final_number_of_layers,
        compact_thickness,
        compact_temperature,
        compact_ice,
        compact_liquid,
        compact_radius,
        compact_aerosol,
        bottom_split_count,
        excess_transfer_count,
    ) = lax.fori_loop(
        0,
        NUMBER_OF_SNOW_LEVELS - 1,
        process_compact_level,
        (
            original_number_of_layers,
            compact_thickness,
            compact_temperature,
            compact_ice,
            compact_liquid,
            compact_radius,
            compact_aerosol,
            zero_count,
            zero_count,
        ),
    )
    final_layer_count = jnp.where(
        snow_column,
        -final_number_of_layers,
        layer_count,
    )
    final_top_index = final_layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS)

    def store_native_level(
        native_index: int,
        output: tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ],
    ):
        (
            output_thickness,
            output_temperature,
            output_ice,
            output_liquid,
            output_radius,
            output_aerosol,
        ) = output
        active = snow_column & (native_index >= final_top_index)
        compact_index = jnp.clip(
            native_index - final_top_index,
            jnp.int32(0),
            jnp.int32(NUMBER_OF_SNOW_LEVELS - 1),
        )
        covered_thickness = _gather(compact_thickness, compact_index)
        safe_snow_fraction = jnp.where(
            snow_fraction != jnp.float64(0.0),
            snow_fraction,
            jnp.float64(1.0),
        )
        native_thickness = (
            covered_thickness
            if is_lake
            else fortran_divide(covered_thickness, safe_snow_fraction)
        )
        output_thickness = output_thickness.at[:, native_index].set(
            jnp.where(active, native_thickness, output_thickness[:, native_index])
        )
        output_temperature = output_temperature.at[:, native_index].set(
            jnp.where(
                active,
                _gather(compact_temperature, compact_index),
                output_temperature[:, native_index],
            )
        )
        output_ice = output_ice.at[:, native_index].set(
            jnp.where(
                active,
                _gather(compact_ice, compact_index),
                output_ice[:, native_index],
            )
        )
        output_liquid = output_liquid.at[:, native_index].set(
            jnp.where(
                active,
                _gather(compact_liquid, compact_index),
                output_liquid[:, native_index],
            )
        )
        output_radius = output_radius.at[:, native_index].set(
            jnp.where(
                active,
                _gather(compact_radius, compact_index),
                output_radius[:, native_index],
            )
        )
        output_aerosol = output_aerosol.at[:, native_index, :].set(
            jnp.where(
                active[:, None],
                _gather(compact_aerosol, compact_index),
                output_aerosol[:, native_index, :],
            )
        )
        return (
            output_thickness,
            output_temperature,
            output_ice,
            output_liquid,
            output_radius,
            output_aerosol,
        )

    thickness, temperature, ice, liquid, radius, aerosol = lax.fori_loop(
        0,
        NUMBER_OF_SNOW_LEVELS,
        store_native_level,
        (thickness, temperature, ice, liquid, radius, aerosol),
    )

    def rebuild_geometry(
        reverse_step: int,
        geometry: tuple[jnp.ndarray, jnp.ndarray],
    ):
        rebuilt_node_depth, rebuilt_interface_depth = geometry
        native_index = jnp.int32(NUMBER_OF_SNOW_LEVELS - 1) - reverse_step
        active = snow_column & (native_index >= final_top_index)
        lower_interface = rebuilt_interface_depth[:, native_index + jnp.int32(1)]
        half_thickness = fortran_multiply(
            jnp.float64(0.5),
            thickness[:, native_index],
        )
        new_node_depth = fortran_add(lower_interface, -half_thickness)
        new_upper_interface = fortran_add(
            lower_interface,
            -thickness[:, native_index],
        )
        rebuilt_node_depth = rebuilt_node_depth.at[:, native_index].set(
            jnp.where(
                active,
                new_node_depth,
                rebuilt_node_depth[:, native_index],
            )
        )
        rebuilt_interface_depth = rebuilt_interface_depth.at[:, native_index].set(
            jnp.where(
                active,
                new_upper_interface,
                rebuilt_interface_depth[:, native_index],
            )
        )
        return rebuilt_node_depth, rebuilt_interface_depth

    node_depth, interface_depth = lax.fori_loop(
        0,
        NUMBER_OF_SNOW_LEVELS,
        rebuild_geometry,
        (node_depth, interface_depth),
    )
    aerosol_fields = {
        name: aerosol[:, :, index]
        for index, name in enumerate(SnowWaterAerosolState._fields[:-1])
    }
    final_aerosol = SnowWaterAerosolState(
        **aerosol_fields,
        deposition=state.aerosol.deposition,
    )
    return SnowLayerDivisionResult(
        state=SnowLayerDivisionState(
            snow_layer_count_column=final_layer_count,
            effective_snow_fraction_column=snow_fraction,
            soil_snow_thickness_column_level=thickness,
            soil_snow_node_depth_column_level=node_depth,
            soil_snow_interface_depth_column_interface=interface_depth,
            soil_snow_temperature_column_level=temperature,
            soil_snow_ice_column_level=ice,
            soil_snow_liquid_column_level=liquid,
            snow_grain_radius_column_level=radius,
            aerosol=final_aerosol,
        ),
        diagnostics=SnowLayerDivisionDiagnostics(
            snow_column=snow_column,
            bottom_split_count_column=bottom_split_count,
            excess_transfer_count_column=excess_transfer_count,
        ),
    )
