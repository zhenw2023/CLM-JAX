
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.constants import (
    FREEZING_TEMPERATURE_FRESHWATER,
    LATENT_HEAT_FUSION,
)
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_GLACIER,
    LANDUNIT_LAKE,
    LANDUNIT_SOIL,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    LANDUNIT_WETLAND,
    SubgridTopology,
)
from clm_jax.physics.numerics import (
    fortran_divide,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.snow_water import (
    NUMBER_OF_SNOW_LEVELS,
    NUMBER_OF_SOIL_SNOW_LEVELS,
    SnowWaterAerosolState,
    build_snow_column_masks,
)

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase


NUMBER_OF_SOIL_SNOW_INTERFACES = NUMBER_OF_SOIL_SNOW_LEVELS + 1
GROUND_SURFACE_INDEX = NUMBER_OF_SNOW_LEVELS
MINIMUM_SNOW_ICE_MASS = 0.01
MINIMUM_SNOW_DEPTH = 0.01
MINIMUM_SNOW_BULK_DENSITY = 50.0
_USE_BULK_LAYER_SHIFT = True
ICE_SPECIFIC_HEAT = 2117.27
LIQUID_WATER_SPECIFIC_HEAT = 4188.0
MINIMUM_LAYER_THICKNESS = (
    0.010,
    0.015,
    0.025,
    0.055,
    0.115,
    0.235,
    0.475,
    0.955,
    1.915,
    3.835,
    7.675,
    15.355,
)


class SnowLayerCombiningParameters(NamedTuple):

    time_step_seconds: float = 1800.0
    minimum_snow_ice_mass: float = MINIMUM_SNOW_ICE_MASS
    minimum_snow_depth: float = MINIMUM_SNOW_DEPTH
    minimum_snow_bulk_density: float = MINIMUM_SNOW_BULK_DENSITY
    minimum_layer_thickness: tuple[float, ...] = MINIMUM_LAYER_THICKNESS
    lake_snow_layer_offset: float = 0.03

    @classmethod
    def from_case(cls, case: CimeCase) -> SnowLayerCombiningParameters:

        from clm_jax.io.namelist import read_case_namelist_group  

        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        return cls(time_step_seconds=float(general["dtime"]))


DEFAULT_SNOW_LAYER_COMBINING_PARAMETERS = SnowLayerCombiningParameters()


class SnowLayerCombiningState(NamedTuple):

    snow_layer_count_column: jnp.ndarray
    snow_fraction_column: jnp.ndarray
    effective_snow_fraction_column: jnp.ndarray
    snow_depth_column: jnp.ndarray
    integrated_snow_column: jnp.ndarray
    snow_water_equivalent_column: jnp.ndarray
    soil_snow_thickness_column_level: jnp.ndarray
    soil_snow_node_depth_column_level: jnp.ndarray
    soil_snow_interface_depth_column_interface: jnp.ndarray
    soil_snow_temperature_column_level: jnp.ndarray
    soil_snow_ice_column_level: jnp.ndarray
    soil_snow_liquid_column_level: jnp.ndarray
    snow_grain_radius_column_level: jnp.ndarray
    top_soil_water_flux_column: jnp.ndarray
    aerosol: SnowWaterAerosolState


class SnowLayerCombiningDiagnostics(NamedTuple):
    snow_column: jnp.ndarray
    low_ice_removal_count_column: jnp.ndarray
    bulk_snow_collapsed_column: jnp.ndarray
    pair_combination_count_column: jnp.ndarray
    final_snow_ice_column: jnp.ndarray
    final_snow_liquid_column: jnp.ndarray


class SnowLayerCombiningResult(NamedTuple):
    state: SnowLayerCombiningState
    diagnostics: SnowLayerCombiningDiagnostics


def _validate_shapes(
    topology: SubgridTopology,
    state: SnowLayerCombiningState,
    parameters: SnowLayerCombiningParameters,
) -> None:
    column_shape = topology.column.gridcell.shape
    if len(column_shape) != 1:
        raise ValueError("column topology must be a vector")
    number_of_columns = column_shape[0]
    level_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS)
    interface_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_INTERFACES)
    snow_shape = (number_of_columns, NUMBER_OF_SNOW_LEVELS)

    scalar_fields = (
        "snow_layer_count_column",
        "snow_fraction_column",
        "effective_snow_fraction_column",
        "snow_depth_column",
        "integrated_snow_column",
        "snow_water_equivalent_column",
        "top_soil_water_flux_column",
    )
    for name in scalar_fields:
        if getattr(state, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")
    level_fields = (
        "soil_snow_thickness_column_level",
        "soil_snow_node_depth_column_level",
        "soil_snow_temperature_column_level",
        "soil_snow_ice_column_level",
        "soil_snow_liquid_column_level",
    )
    for name in level_fields:
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

    if parameters.time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")
    if parameters.minimum_snow_ice_mass < 0.0:
        raise ValueError("minimum_snow_ice_mass must be nonnegative")
    if parameters.minimum_snow_depth <= 0.0:
        raise ValueError("minimum_snow_depth must be positive")
    if parameters.minimum_snow_bulk_density <= 0.0:
        raise ValueError("minimum_snow_bulk_density must be positive")
    if parameters.lake_snow_layer_offset < 0.0:
        raise ValueError("lake_snow_layer_offset must be nonnegative")
    if len(parameters.minimum_layer_thickness) != NUMBER_OF_SNOW_LEVELS:
        raise ValueError(
            f"minimum_layer_thickness must contain {NUMBER_OF_SNOW_LEVELS} values"
        )


def _gather_column_level(values: jnp.ndarray, index: jnp.ndarray) -> jnp.ndarray:
    rows = jnp.arange(values.shape[0], dtype=jnp.int32)
    return values[rows, index]


def _restore_soil_tail(snow: jnp.ndarray, original: jnp.ndarray) -> jnp.ndarray:
    tail = jnp.asarray(original, dtype=snow.dtype)[:, snow.shape[1]:]
    return jnp.concatenate((snow, tail), axis=1)


def _set_column_level(
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


def _shift_snow_arrays(values, enabled, upper_index, top_index):
    if not _USE_BULK_LAYER_SHIFT:
        def shift(step, carry):
            index = jnp.int32(NUMBER_OF_SNOW_LEVELS - 1) - step
            source = jnp.maximum(index - jnp.int32(1), jnp.int32(0))
            active = enabled & (index <= upper_index) & (index >= top_index + 1)
            return tuple(value.at[:, index].set(jnp.where(
                active.reshape((active.shape[0],) + (1,) * (value.ndim - 2)),
                value[:, source], value[:, index],
            )) for value in carry)
        return lax.fori_loop(0, NUMBER_OF_SNOW_LEVELS, shift, values)

    shifted = []
    for value in values:
        index = jnp.arange(value.shape[1], dtype=jnp.int32)
        active = (enabled[:, None] & (index[None, :] <= upper_index)
                  if jnp.ndim(upper_index) == 0 else
                  enabled[:, None] & (index[None, :] <= upper_index[:, None]))
        active = (active & (index[None, :] >= top_index[:, None] + 1)
                  & (index[None, :] < NUMBER_OF_SNOW_LEVELS))
        source = jnp.maximum(index - jnp.int32(1), jnp.int32(0))
        shifted.append(jnp.where(
            active.reshape((*active.shape, *((1,) * (value.ndim - 2)))),
            jnp.take(value, source, axis=1), value,
        ))
    return tuple(shifted)


def _combine_mass_and_enthalpy(
    thickness: jnp.ndarray,
    liquid: jnp.ndarray,
    ice: jnp.ndarray,
    temperature: jnp.ndarray,
    thickness_two: jnp.ndarray,
    liquid_two: jnp.ndarray,
    ice_two: jnp.ndarray,
    temperature_two: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    cpice = jnp.float64(ICE_SPECIFIC_HEAT)
    cpliq = jnp.float64(LIQUID_WATER_SPECIFIC_HEAT)
    freezing = jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)
    fusion = jnp.float64(LATENT_HEAT_FUSION)

    combined_thickness = fortran_add(thickness, thickness_two)
    combined_ice = fortran_add(ice, ice_two)
    combined_liquid = fortran_add(liquid, liquid_two)

    heat_capacity = fortran_add(
        fortran_multiply(cpice, ice),
        fortran_multiply(cpliq, liquid),
    )
    enthalpy = fortran_add(
        fortran_multiply(
            heat_capacity,
            fortran_add(temperature, -freezing),
        ),
        fortran_multiply(fusion, liquid),
    )
    heat_capacity_two = fortran_add(
        fortran_multiply(cpice, ice_two),
        fortran_multiply(cpliq, liquid_two),
    )
    enthalpy_two = fortran_add(
        fortran_multiply(
            heat_capacity_two,
            fortran_add(temperature_two, -freezing),
        ),
        fortran_multiply(fusion, liquid_two),
    )
    combined_enthalpy = fortran_add(enthalpy, enthalpy_two)
    combined_capacity = fortran_add(
        fortran_multiply(cpice, combined_ice),
        fortran_multiply(cpliq, combined_liquid),
    )
    safe_combined_capacity = jnp.where(
        combined_capacity != jnp.float64(0.0),
        combined_capacity,
        jnp.float64(1.0),
    )
    temperature_numerator = fortran_add(
        combined_enthalpy,
        -fortran_multiply(fusion, combined_liquid),
    )
    combined_temperature = fortran_add(
        freezing,
        fortran_divide(temperature_numerator, safe_combined_capacity),
    )
    return (
        combined_thickness,
        combined_liquid,
        combined_ice,
        combined_temperature,
    )


def combine_snow_layers_active_case(
    topology: SubgridTopology,
    state: SnowLayerCombiningState,
    *,
    parameters: SnowLayerCombiningParameters = DEFAULT_SNOW_LAYER_COMBINING_PARAMETERS,
    lake_only: bool = False,
) -> SnowLayerCombiningResult:
    if jax.default_backend() == 'gpu' and state.snow_layer_count_column.size > 16384:
        from clm_jax.physics.snow_column_filter import filtered_snow_transition

        def transition(local_topology, local_state):
            from clm_jax.physics.snow_layers_gpu import snow_layers_gpu

            return snow_layers_gpu(
                'combining', local_topology, local_state, parameters,
                lake_only=lake_only,
            )

        return filtered_snow_transition(transition, topology, state, lake_only=lake_only)
    return _combine_snow_layers_active_case(
        topology, state, parameters=parameters, lake_only=lake_only,
    )


class _CombiningColumns(NamedTuple):
    gridcell: jax.Array
    landunit_type: jax.Array
    active: jax.Array


class _CombiningTopology(NamedTuple):
    column: _CombiningColumns


def _combine_snow_layers_active_case(
    topology: SubgridTopology,
    state: SnowLayerCombiningState,
    *,
    parameters: SnowLayerCombiningParameters = (
        DEFAULT_SNOW_LAYER_COMBINING_PARAMETERS
    ),
    lake_only: bool = False,
) -> SnowLayerCombiningResult:
    columns = topology.column
    local = _CombiningTopology(_CombiningColumns(
        columns.gridcell, columns.landunit_type, columns.active,
    ))
    return _combining_with_storage_adjoint(local, state, parameters, lake_only)


@partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def _combining_with_storage_adjoint(topology, state, parameters, lake_only):
    return _combine_snow_layers_reference(
        topology, state, parameters=parameters, lake_only=lake_only,
    )


@_combining_with_storage_adjoint.defjvp
def _combining_storage_jvp(parameters, lake_only, primals, tangents):
    def guarded(topology, state):
        active = build_snow_column_masks(
            topology, state.snow_layer_count_column, lake_only=lake_only,
        ).snow_column
        top = state.snow_layer_count_column + jnp.int32(NUMBER_OF_SNOW_LEVELS)

        def used_slots(value):
            if value.ndim == 1:
                return active
            return active[:, None] & (jnp.arange(value.shape[1])[None, :] >= top[:, None])

        safe = jax.tree.map(lambda value: jnp.where(
            used_slots(value), value, jnp.zeros_like(value),
        ), state)
        result = _combine_snow_layers_reference(
            topology, safe, parameters=parameters, lake_only=lake_only,
        )
        restored = jax.tree.map(lambda value, original: jnp.where(
            used_slots(original), value, original,
        ), result.state, state)
        return result._replace(state=restored)

    return (_combining_with_storage_adjoint(*primals, parameters, lake_only),
            jax.jvp(guarded, primals, tangents)[1])


def _combine_snow_layers_reference(
    topology: SubgridTopology,
    state: SnowLayerCombiningState,
    *,
    parameters: SnowLayerCombiningParameters = DEFAULT_SNOW_LAYER_COMBINING_PARAMETERS,
    lake_only: bool = False,
) -> SnowLayerCombiningResult:

    _validate_shapes(topology, state, parameters)
    number_of_columns = topology.column.gridcell.shape[0]
    masks = build_snow_column_masks(
        topology,
        state.snow_layer_count_column,
        lake_only=lake_only,
    )
    snow_column = masks.snow_column
    landunit_type = jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
    lake = landunit_type == jnp.int32(LANDUNIT_LAKE)
    urban = (landunit_type >= jnp.int32(LANDUNIT_URBAN_MIN)) & (
        landunit_type <= jnp.int32(LANDUNIT_URBAN_MAX)
    )
    soil_crop_or_urban = (
        (landunit_type == jnp.int32(LANDUNIT_SOIL))
        | (landunit_type == jnp.int32(LANDUNIT_CROP))
        | urban
    )
    wetland_or_glacier = (
        (landunit_type == jnp.int32(LANDUNIT_WETLAND))
        | (landunit_type == jnp.int32(LANDUNIT_GLACIER))
    )

    snow_layer_count = jnp.asarray(
        state.snow_layer_count_column,
        dtype=jnp.int32,
    )
    original_top_index = snow_layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS)
    snow_fraction = jnp.asarray(state.snow_fraction_column, dtype=jnp.float64)
    effective_snow_fraction = jnp.asarray(
        state.effective_snow_fraction_column,
        dtype=jnp.float64,
    )
    snow_depth = jnp.asarray(state.snow_depth_column, dtype=jnp.float64)
    integrated_snow = jnp.asarray(state.integrated_snow_column, dtype=jnp.float64)
    snow_water_equivalent = jnp.asarray(
        state.snow_water_equivalent_column,
        dtype=jnp.float64,
    )
    thickness = jnp.asarray(
        state.soil_snow_thickness_column_level,
        dtype=jnp.float64,
    )[:, :GROUND_SURFACE_INDEX + 1]
    node_depth = jnp.asarray(
        state.soil_snow_node_depth_column_level,
        dtype=jnp.float64,
    )[:, :GROUND_SURFACE_INDEX + 1]
    interface_depth = jnp.asarray(
        state.soil_snow_interface_depth_column_interface,
        dtype=jnp.float64,
    )[:, :GROUND_SURFACE_INDEX + 1]
    temperature = jnp.asarray(
        state.soil_snow_temperature_column_level,
        dtype=jnp.float64,
    )[:, :GROUND_SURFACE_INDEX + 1]
    ice = jnp.asarray(state.soil_snow_ice_column_level, dtype=jnp.float64)[
        :, :GROUND_SURFACE_INDEX + 1
    ]
    liquid = jnp.asarray(state.soil_snow_liquid_column_level, dtype=jnp.float64)[
        :, :GROUND_SURFACE_INDEX + 1
    ]
    snow_grain_radius = jnp.asarray(
        state.snow_grain_radius_column_level,
        dtype=jnp.float64,
    )
    top_soil_water_flux = jnp.where(
        snow_column,
        jnp.float64(0.0),
        jnp.asarray(state.top_soil_water_flux_column, dtype=jnp.float64),
    )
    aerosol = jnp.stack(
        [
            jnp.asarray(getattr(state.aerosol, name), dtype=jnp.float64)
            for name in SnowWaterAerosolState._fields[:-1]
        ],
        axis=-1,
    )
    zero_count = jnp.zeros(number_of_columns, dtype=jnp.int32)
    time_step = jnp.float64(parameters.time_step_seconds)
    minimum_ice = jnp.float64(parameters.minimum_snow_ice_mass)

    def remove_low_ice_layer(
        level_index: int,
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
            layer_count,
            water_flux,
            layer_temperature,
            layer_ice,
            layer_liquid,
            layer_thickness,
            grain_radius,
            layer_aerosol,
            removal_count,
        ) = carry
        source_ice = layer_ice[:, level_index]
        source_liquid = layer_liquid[:, level_index]
        source_mass = fortran_add(source_liquid, source_ice)
        in_original_loop = level_index >= original_top_index
        remove = snow_column & in_original_loop & (source_ice <= minimum_ice)
        current_top_index = layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS)
        shift = (
            remove
            & (level_index > current_top_index)
            & (layer_count < jnp.int32(-1))
        )

        destination = level_index + jnp.int32(1)
        transfer_mass = remove & (
            (level_index < jnp.int32(NUMBER_OF_SNOW_LEVELS - 1))
            | soil_crop_or_urban
        )
        destination_ice = layer_ice[:, destination]
        destination_liquid = layer_liquid[:, destination]
        layer_ice = layer_ice.at[:, destination].set(
            jnp.where(
                transfer_mass,
                fortran_add(destination_ice, source_ice),
                destination_ice,
            )
        )
        layer_liquid = layer_liquid.at[:, destination].set(
            jnp.where(
                transfer_mass,
                fortran_add(destination_liquid, source_liquid),
                destination_liquid,
            )
        )

        snow_destination = jnp.minimum(
            destination,
            jnp.int32(NUMBER_OF_SNOW_LEVELS - 1),
        )
        transfer_snow_geometry = remove & (
            level_index < jnp.int32(NUMBER_OF_SNOW_LEVELS - 1)
        )
        destination_thickness = layer_thickness[:, destination]
        layer_thickness = layer_thickness.at[:, destination].set(
            jnp.where(
                transfer_snow_geometry,
                fortran_add(
                    destination_thickness,
                    layer_thickness[:, level_index],
                ),
                destination_thickness,
            )
        )
        destination_aerosol = layer_aerosol[:, snow_destination, :]
        layer_aerosol = layer_aerosol.at[:, snow_destination, :].set(
            jnp.where(
                transfer_snow_geometry[:, None],
                fortran_add(
                    destination_aerosol,
                    layer_aerosol[:, level_index, :],
                ),
                destination_aerosol,
            )
        )

        bottom = level_index == jnp.int32(NUMBER_OF_SNOW_LEVELS - 1)
        report_flux = remove & bottom & (soil_crop_or_urban | shift)
        removal_flux = fortran_divide(source_mass, time_step)
        water_flux = jnp.where(report_flux, removal_flux, water_flux)

        (
            layer_temperature,
            layer_ice,
            layer_liquid,
            layer_thickness,
            grain_radius,
            layer_aerosol,
        ) = _shift_snow_arrays(
            (
                layer_temperature,
                layer_ice,
                layer_liquid,
                layer_thickness,
                grain_radius,
                layer_aerosol,
            ),
            shift, level_index, current_top_index,
        )
        layer_count = jnp.where(remove, layer_count + jnp.int32(1), layer_count)
        removal_count = removal_count + remove.astype(jnp.int32)
        return (
            layer_count,
            water_flux,
            layer_temperature,
            layer_ice,
            layer_liquid,
            layer_thickness,
            grain_radius,
            layer_aerosol,
            removal_count,
        )

    (
        snow_layer_count,
        top_soil_water_flux,
        temperature,
        ice,
        liquid,
        thickness,
        snow_grain_radius,
        aerosol,
        low_ice_removal_count,
    ) = lax.fori_loop(
        0,
        NUMBER_OF_SNOW_LEVELS,
        remove_low_ice_layer,
        (
            snow_layer_count,
            top_soil_water_flux,
            temperature,
            ice,
            liquid,
            thickness,
            snow_grain_radius,
            aerosol,
            zero_count,
        ),
    )

    zero = jnp.zeros(number_of_columns, dtype=jnp.float64)

    def sum_snow_level(
        level_index: int,
        totals: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ):
        water_total, depth_total, ice_total, liquid_total = totals
        active_level = snow_column & (
            level_index
            >= snow_layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS)
        )
        water_with_ice = fortran_add(water_total, ice[:, level_index])
        water_with_liquid = fortran_add(
            water_with_ice,
            liquid[:, level_index],
        )
        return (
            jnp.where(active_level, water_with_liquid, water_total),
            jnp.where(
                active_level,
                fortran_add(depth_total, thickness[:, level_index]),
                depth_total,
            ),
            jnp.where(
                active_level,
                fortran_add(ice_total, ice[:, level_index]),
                ice_total,
            ),
            jnp.where(
                active_level,
                fortran_add(liquid_total, liquid[:, level_index]),
                liquid_total,
            ),
        )

    snow_water_sum, snow_depth_sum, snow_ice_sum, snow_liquid_sum = lax.fori_loop(
        0,
        NUMBER_OF_SNOW_LEVELS,
        sum_snow_level,
        (zero, zero, zero, zero),
    )
    snow_water_equivalent = jnp.where(
        snow_column,
        snow_water_sum,
        snow_water_equivalent,
    )
    snow_depth = jnp.where(snow_column, snow_depth_sum, snow_depth)

    fractional_depth = fortran_multiply(effective_snow_fraction, snow_depth)
    safe_fractional_depth = jnp.where(
        fractional_depth != jnp.float64(0.0),
        fractional_depth,
        jnp.float64(1.0),
    )
    bulk_density = fortran_divide(
        snow_water_equivalent,
        safe_fractional_depth,
    )
    nonlake_bulk_snow_collapsed = (~lake) & (
        (fractional_depth < jnp.float64(parameters.minimum_snow_depth))
        | (bulk_density < jnp.float64(parameters.minimum_snow_bulk_density))
    )
    lake_bulk_snow_collapsed = lake & (
        snow_depth
        < fortran_add(
            jnp.float64(parameters.minimum_snow_depth),
            jnp.float64(parameters.lake_snow_layer_offset),
        )
    )
    bulk_snow_collapsed = (
        snow_column
        & (snow_depth > jnp.float64(0.0))
        & (lake_bulk_snow_collapsed | nonlake_bulk_snow_collapsed)
    )
    snow_layer_count = jnp.where(
        bulk_snow_collapsed,
        jnp.int32(0),
        snow_layer_count,
    )
    snow_water_equivalent = jnp.where(
        bulk_snow_collapsed,
        snow_ice_sum,
        snow_water_equivalent,
    )
    aerosol = jnp.where(
        bulk_snow_collapsed[:, None, None],
        jnp.float64(0.0),
        aerosol,
    )
    snow_depth = jnp.where(
        bulk_snow_collapsed & (snow_water_equivalent <= jnp.float64(0.0)),
        jnp.float64(0.0),
        snow_depth,
    )

    bottom_snow_index = jnp.int32(NUMBER_OF_SNOW_LEVELS - 1)
    top_soil_index = jnp.int32(GROUND_SURFACE_INDEX)
    transfer_collapsed_liquid = bulk_snow_collapsed & soil_crop_or_urban
    top_soil_liquid = liquid[:, top_soil_index]
    liquid = liquid.at[:, top_soil_index].set(
        jnp.where(
            transfer_collapsed_liquid,
            fortran_add(top_soil_liquid, snow_liquid_sum),
            top_soil_liquid,
        )
    )
    clear_bottom_liquid = bulk_snow_collapsed & (
        soil_crop_or_urban | wetland_or_glacier
    )
    liquid = liquid.at[:, bottom_snow_index].set(
        jnp.where(
            clear_bottom_liquid,
            jnp.float64(0.0),
            liquid[:, bottom_snow_index],
        )
    )

    snow_water_gone = snow_column & (
        snow_water_equivalent <= jnp.float64(0.0)
    )
    snow_depth = jnp.where(snow_water_gone, jnp.float64(0.0), snow_depth)
    snow_fraction = jnp.where(
        snow_water_gone,
        jnp.float64(0.0),
        snow_fraction,
    )
    effective_snow_fraction = jnp.where(
        snow_water_gone,
        jnp.float64(0.0),
        effective_snow_fraction,
    )
    integrated_snow = jnp.where(
        snow_water_gone,
        jnp.float64(0.0),
        integrated_snow,
    )

    entered_multilayer = snow_column & (snow_layer_count < jnp.int32(-1))
    second_loop_top_index = snow_layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS)
    minimum_thickness = jnp.asarray(
        parameters.minimum_layer_thickness,
        dtype=jnp.float64,
    )

    def combine_thin_layer(
        level_index: int,
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
            layer_count,
            layer_temperature,
            layer_ice,
            layer_liquid,
            layer_thickness,
            grain_radius,
            layer_aerosol,
            minimum_index,
            combination_count,
        ) = carry
        current_top_index = layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS)
        loop_active = (
            entered_multilayer
            & (level_index >= second_loop_top_index)
            & (layer_count < jnp.int32(-1))
        )
        clipped_minimum_index = jnp.clip(
            minimum_index,
            jnp.int32(0),
            jnp.int32(NUMBER_OF_SNOW_LEVELS - 1),
        )
        local_minimum = fortran_add(
            minimum_thickness[clipped_minimum_index],
            jnp.where(
                lake,
                jnp.float64(parameters.lake_snow_layer_offset),
                jnp.float64(0.0),
            ),
        )
        fractional_layer_depth = fortran_multiply(
            effective_snow_fraction,
            layer_thickness[:, level_index],
        )
        layer_mass = fortran_add(
            layer_liquid[:, level_index],
            layer_ice[:, level_index],
        )
        safe_layer_depth = jnp.where(
            fractional_layer_depth != jnp.float64(0.0),
            fractional_layer_depth,
            jnp.float64(1.0),
        )
        layer_density = fortran_divide(layer_mass, safe_layer_depth)
        thin = (
            (fractional_layer_depth < local_minimum)
            | (layer_density < jnp.float64(parameters.minimum_snow_bulk_density))
        )
        combine = loop_active & thin

        upper_index = jnp.maximum(level_index - jnp.int32(1), jnp.int32(0))
        lower_index = jnp.minimum(
            level_index + jnp.int32(1),
            jnp.int32(NUMBER_OF_SNOW_LEVELS - 1),
        )
        upper_pair_thickness = fortran_add(
            layer_thickness[:, upper_index],
            layer_thickness[:, level_index],
        )
        lower_pair_thickness = fortran_add(
            layer_thickness[:, lower_index],
            layer_thickness[:, level_index],
        )
        interior_neighbor = jnp.where(
            upper_pair_thickness < lower_pair_thickness,
            upper_index,
            lower_index,
        )
        neighbor = jnp.where(
            level_index == current_top_index,
            lower_index,
            jnp.where(
                level_index == jnp.int32(NUMBER_OF_SNOW_LEVELS - 1),
                upper_index,
                interior_neighbor,
            ),
        )
        destination_index = jnp.maximum(level_index, neighbor)
        source_index = jnp.minimum(level_index, neighbor)

        destination_liquid = _gather_column_level(
            layer_liquid,
            destination_index,
        )
        destination_ice = _gather_column_level(layer_ice, destination_index)
        destination_thickness = _gather_column_level(
            layer_thickness,
            destination_index,
        )
        destination_temperature = _gather_column_level(
            layer_temperature,
            destination_index,
        )
        destination_radius = _gather_column_level(
            grain_radius,
            destination_index,
        )
        destination_aerosol = _gather_column_level(
            layer_aerosol,
            destination_index,
        )
        source_liquid = _gather_column_level(layer_liquid, source_index)
        source_ice = _gather_column_level(layer_ice, source_index)
        source_thickness = _gather_column_level(layer_thickness, source_index)
        source_temperature = _gather_column_level(
            layer_temperature,
            source_index,
        )
        source_radius = _gather_column_level(grain_radius, source_index)
        source_aerosol = _gather_column_level(layer_aerosol, source_index)

        destination_mass = fortran_add(destination_liquid, destination_ice)
        source_mass = fortran_add(source_liquid, source_ice)
        radius_numerator = fortran_add(
            fortran_multiply(destination_radius, destination_mass),
            fortran_multiply(source_radius, source_mass),
        )
        radius_denominator = fortran_add(
            fortran_add(
                fortran_add(destination_liquid, destination_ice),
                source_liquid,
            ),
            source_ice,
        )
        safe_radius_denominator = jnp.where(
            radius_denominator != jnp.float64(0.0),
            radius_denominator,
            jnp.float64(1.0),
        )
        combined_radius = fortran_divide(
            radius_numerator,
            safe_radius_denominator,
        )
        (
            combined_thickness,
            combined_liquid,
            combined_ice,
            combined_temperature,
        ) = _combine_mass_and_enthalpy(
            destination_thickness,
            destination_liquid,
            destination_ice,
            destination_temperature,
            source_thickness,
            source_liquid,
            source_ice,
            source_temperature,
        )
        combined_aerosol = fortran_add(destination_aerosol, source_aerosol)

        layer_thickness = _set_column_level(
            layer_thickness,
            destination_index,
            combined_thickness,
            combine,
        )
        layer_liquid = _set_column_level(
            layer_liquid,
            destination_index,
            combined_liquid,
            combine,
        )
        layer_ice = _set_column_level(
            layer_ice,
            destination_index,
            combined_ice,
            combine,
        )
        layer_temperature = _set_column_level(
            layer_temperature,
            destination_index,
            combined_temperature,
            combine,
        )
        grain_radius = _set_column_level(
            grain_radius,
            destination_index,
            combined_radius,
            combine,
        )
        layer_aerosol = _set_column_level(
            layer_aerosol,
            destination_index,
            combined_aerosol,
            combine,
        )

        (
            layer_temperature,
            layer_ice,
            layer_liquid,
            layer_thickness,
            grain_radius,
            layer_aerosol,
        ) = _shift_snow_arrays(
            (
                layer_temperature,
                layer_ice,
                layer_liquid,
                layer_thickness,
                grain_radius,
                layer_aerosol,
            ),
            combine, destination_index - jnp.int32(1), current_top_index,
        )
        layer_count = jnp.where(
            combine,
            layer_count + jnp.int32(1),
            layer_count,
        )
        minimum_index = jnp.where(
            loop_active & ~thin,
            minimum_index + jnp.int32(1),
            minimum_index,
        )
        combination_count = combination_count + combine.astype(jnp.int32)
        return (
            layer_count,
            layer_temperature,
            layer_ice,
            layer_liquid,
            layer_thickness,
            grain_radius,
            layer_aerosol,
            minimum_index,
            combination_count,
        )

    (
        snow_layer_count,
        temperature,
        ice,
        liquid,
        thickness,
        snow_grain_radius,
        aerosol,
        _,
        pair_combination_count,
    ) = lax.fori_loop(
        0,
        NUMBER_OF_SNOW_LEVELS,
        combine_thin_layer,
        (
            snow_layer_count,
            temperature,
            ice,
            liquid,
            thickness,
            snow_grain_radius,
            aerosol,
            zero_count,
            zero_count,
        ),
    )

    def rebuild_geometry(
        reverse_step: int,
        geometry: tuple[jnp.ndarray, jnp.ndarray],
    ):
        rebuilt_node_depth, rebuilt_interface_depth = geometry
        level_index = jnp.int32(NUMBER_OF_SNOW_LEVELS - 1) - reverse_step
        active_level = snow_column & (
            level_index
            >= snow_layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS)
        )
        lower_interface = rebuilt_interface_depth[:, level_index + jnp.int32(1)]
        half_thickness = fortran_multiply(
            jnp.float64(0.5),
            thickness[:, level_index],
        )
        new_node_depth = fortran_add(lower_interface, -half_thickness)
        new_upper_interface = fortran_add(
            lower_interface,
            -thickness[:, level_index],
        )
        rebuilt_node_depth = rebuilt_node_depth.at[:, level_index].set(
            jnp.where(
                active_level,
                new_node_depth,
                rebuilt_node_depth[:, level_index],
            )
        )
        rebuilt_interface_depth = rebuilt_interface_depth.at[:, level_index].set(
            jnp.where(
                active_level,
                new_upper_interface,
                rebuilt_interface_depth[:, level_index],
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
    return SnowLayerCombiningResult(
        state=SnowLayerCombiningState(
            snow_layer_count_column=snow_layer_count,
            snow_fraction_column=snow_fraction,
            effective_snow_fraction_column=effective_snow_fraction,
            snow_depth_column=snow_depth,
            integrated_snow_column=integrated_snow,
            snow_water_equivalent_column=snow_water_equivalent,
            soil_snow_thickness_column_level=_restore_soil_tail(
                thickness, state.soil_snow_thickness_column_level,
            ),
            soil_snow_node_depth_column_level=_restore_soil_tail(
                node_depth, state.soil_snow_node_depth_column_level,
            ),
            soil_snow_interface_depth_column_interface=_restore_soil_tail(
                interface_depth, state.soil_snow_interface_depth_column_interface,
            ),
            soil_snow_temperature_column_level=_restore_soil_tail(
                temperature, state.soil_snow_temperature_column_level,
            ),
            soil_snow_ice_column_level=_restore_soil_tail(
                ice, state.soil_snow_ice_column_level,
            ),
            soil_snow_liquid_column_level=_restore_soil_tail(
                liquid, state.soil_snow_liquid_column_level,
            ),
            snow_grain_radius_column_level=snow_grain_radius,
            top_soil_water_flux_column=top_soil_water_flux,
            aerosol=final_aerosol,
        ),
        diagnostics=SnowLayerCombiningDiagnostics(
            snow_column=snow_column,
            low_ice_removal_count_column=low_ice_removal_count,
            bulk_snow_collapsed_column=bulk_snow_collapsed,
            pair_combination_count_column=pair_combination_count,
            final_snow_ice_column=snow_ice_sum,
            final_snow_liquid_column=snow_liquid_sum,
        ),
    )
