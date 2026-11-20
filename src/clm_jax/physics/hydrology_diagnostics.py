
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.constants import (
    CLM_SPECIAL_VALUE,
    DENSITY_FRESH_WATER,
    DENSITY_ICE,
)
from clm_jax.io.restart import (
    COLUMN_URBAN_ROOF,
    COLUMN_URBAN_SHADE_WALL,
    COLUMN_URBAN_SUN_WALL,
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
from clm_jax.physics.numerics import fortran_hardware_power as fortran_power
from clm_jax.physics.positive_power import positive_power
from clm_jax.physics.snow_layer_combining import NUMBER_OF_SOIL_SNOW_INTERFACES
from clm_jax.physics.snow_water import (
    NUMBER_OF_GROUND_LEVELS,
    NUMBER_OF_SNOW_LEVELS,
    NUMBER_OF_SOIL_SNOW_LEVELS,
    build_snow_column_masks,
)
from clm_jax.physics.surface_hydrology import build_surface_hydrology_masks

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase


NUMBER_OF_SOIL_LEVELS = 20
NUMBER_OF_URBAN_LEVELS = 5
SOIL_WATER_POTENTIAL_MINIMUM_MPA = -15.0
MINIMUM_SOILPSI_FRACTIONAL_SATURATION = 0.001
MINIMUM_MATRIX_POTENTIAL_SATURATION = 0.01
DRY_SOIL_POTENTIAL_MILLIMETERS = 316230.0


def _retention_power(saturation, exponent):
    if jax.default_backend() == "gpu":
        return positive_power(saturation, exponent)
    return fortran_power(saturation, exponent)


class HydrologyDiagnosticsParameters(NamedTuple):
    number_of_soil_levels: int = NUMBER_OF_SOIL_LEVELS
    number_of_urban_levels: int = NUMBER_OF_URBAN_LEVELS

    @classmethod
    def from_case(cls, case: CimeCase) -> HydrologyDiagnosticsParameters:

        from clm_jax.io.namelist import read_case_namelist_group  

        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        layer_structure = str(general["soil_layerstruct"]).strip()
        if layer_structure != "20SL_8.5m":
            raise ValueError(
                "CLM-JAX hydrology diagnostics currently require the reference "
                f"20SL_8.5m structure, got {layer_structure!r}"
            )
        return cls()


DEFAULT_HYDROLOGY_DIAGNOSTICS_PARAMETERS = HydrologyDiagnosticsParameters()


class HydrologyDiagnosticsInputs(NamedTuple):
    snow_layer_count_column: jnp.ndarray
    soil_snow_thickness_column_level: jnp.ndarray
    soil_snow_node_depth_column_level: jnp.ndarray
    soil_snow_interface_depth_column_interface: jnp.ndarray
    soil_snow_temperature_column_level: jnp.ndarray
    soil_snow_ice_column_level: jnp.ndarray
    soil_snow_liquid_column_level: jnp.ndarray
    effective_snow_fraction_column: jnp.ndarray
    surface_water_fraction_column: jnp.ndarray
    surface_water_temperature_column: jnp.ndarray
    soil_porosity_column_level: jnp.ndarray
    saturated_suction_column_level: jnp.ndarray
    clapp_hornberger_exponent_column_level: jnp.ndarray
    minimum_soil_matrix_potential_column: jnp.ndarray


class HydrologyDiagnosticsState(NamedTuple):
    near_surface_soil_temperature_column: jnp.ndarray
    top_10cm_soil_temperature_column: jnp.ndarray
    top_17cm_soil_temperature_column: jnp.ndarray
    top_10cm_soil_water_column: jnp.ndarray
    total_soil_liquid_water_column: jnp.ndarray
    total_soil_ice_water_column: jnp.ndarray
    ground_temperature_column: jnp.ndarray
    urban_ground_temperature_column: jnp.ndarray
    rural_ground_temperature_column: jnp.ndarray
    volumetric_soil_water_column_level: jnp.ndarray
    soil_water_potential_column_level: jnp.ndarray
    soil_matrix_potential_column_level: jnp.ndarray
    top_5cm_water_availability_column: jnp.ndarray
    top_17cm_water_availability_column: jnp.ndarray
    top_snow_water_column: jnp.ndarray
    snow_grain_radius_column_level: jnp.ndarray
    top_snow_temperature_column: jnp.ndarray
    top_snow_temperature_gradient_column: jnp.ndarray
    top_snow_grain_radius_column: jnp.ndarray
    top_snow_liquid_fraction_column: jnp.ndarray


class HydrologyDiagnosticsMasks(NamedTuple):
    nonlake_column: jnp.ndarray
    rural_nonlake_column: jnp.ndarray
    urban_column: jnp.ndarray
    soil_crop_column: jnp.ndarray
    hydrology_column: jnp.ndarray
    snow_column: jnp.ndarray
    no_snow_column: jnp.ndarray
    volumetric_update_column_level: jnp.ndarray


class HydrologyDiagnosticsResult(NamedTuple):
    state: HydrologyDiagnosticsState
    masks: HydrologyDiagnosticsMasks


def _validate_shapes(
    topology: SubgridTopology,
    inputs: HydrologyDiagnosticsInputs,
    state: HydrologyDiagnosticsState,
    parameters: HydrologyDiagnosticsParameters,
) -> None:
    column_shape = topology.column.gridcell.shape
    if len(column_shape) != 1:
        raise ValueError("column topology must be a vector")
    number_of_columns = column_shape[0]
    soil_snow_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS)
    interface_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_INTERFACES)
    ground_shape = (number_of_columns, NUMBER_OF_GROUND_LEVELS)
    snow_shape = (number_of_columns, NUMBER_OF_SNOW_LEVELS)
    vector_inputs = {
        "snow_layer_count_column",
        "effective_snow_fraction_column",
        "surface_water_fraction_column",
        "surface_water_temperature_column",
        "minimum_soil_matrix_potential_column",
    }
    soil_snow_inputs = {
        "soil_snow_thickness_column_level",
        "soil_snow_node_depth_column_level",
        "soil_snow_temperature_column_level",
        "soil_snow_ice_column_level",
        "soil_snow_liquid_column_level",
    }
    ground_inputs = {
        "soil_porosity_column_level",
        "saturated_suction_column_level",
        "clapp_hornberger_exponent_column_level",
    }
    for name in vector_inputs:
        if getattr(inputs, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")
    for name in soil_snow_inputs:
        if getattr(inputs, name).shape != soil_snow_shape:
            raise ValueError(f"{name} must have shape {soil_snow_shape}")
    if inputs.soil_snow_interface_depth_column_interface.shape != interface_shape:
        raise ValueError(
            "soil_snow_interface_depth_column_interface must have shape "
            f"{interface_shape}"
        )
    for name in ground_inputs:
        if getattr(inputs, name).shape != ground_shape:
            raise ValueError(f"{name} must have shape {ground_shape}")
    matrix_state_shapes = {
        "volumetric_soil_water_column_level": ground_shape,
        "soil_water_potential_column_level": ground_shape,
        "soil_matrix_potential_column_level": ground_shape,
        "snow_grain_radius_column_level": snow_shape,
    }
    for name in HydrologyDiagnosticsState._fields:
        expected = matrix_state_shapes.get(name, column_shape)
        if getattr(state, name).shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if parameters.number_of_soil_levels != NUMBER_OF_SOIL_LEVELS:
        raise ValueError(
            f"number_of_soil_levels must be {NUMBER_OF_SOIL_LEVELS}"
        )
    if parameters.number_of_urban_levels != NUMBER_OF_URBAN_LEVELS:
        raise ValueError(
            f"number_of_urban_levels must be {NUMBER_OF_URBAN_LEVELS}"
        )


def _fraction_within_depth(
    upper_interface: jnp.ndarray,
    lower_interface: jnp.ndarray,
    thickness: jnp.ndarray,
    depth: float,
) -> jnp.ndarray:
    target = jnp.float64(depth)
    partial = fortran_divide(
        fortran_add(target, -upper_interface),
        thickness,
    )
    return jnp.where(
        lower_interface <= target,
        jnp.float64(1.0),
        jnp.where(
            (lower_interface > target) & (upper_interface < target),
            partial,
            jnp.float64(0.0),
        ),
    )


def _dry_water_content(
    porosity: jnp.ndarray,
    saturated_suction: jnp.ndarray,
    exponent: jnp.ndarray,
) -> jnp.ndarray:
    suction_ratio = fortran_divide(
        jnp.float64(DRY_SOIL_POTENTIAL_MILLIMETERS),
        saturated_suction,
    )
    inverse_exponent = fortran_divide(jnp.float64(-1.0), exponent)
    return fortran_multiply(
        porosity,
        _retention_power(suction_ratio, inverse_exponent),
    )


def hydrology_no_drainage_diagnostics_active_case(
    topology: SubgridTopology,
    inputs: HydrologyDiagnosticsInputs,
    state: HydrologyDiagnosticsState,
    *,
    parameters: HydrologyDiagnosticsParameters = (
        DEFAULT_HYDROLOGY_DIAGNOSTICS_PARAMETERS
    ),
) -> HydrologyDiagnosticsResult:

    _validate_shapes(topology, inputs, state, parameters)
    number_of_columns = topology.column.gridcell.shape[0]
    column_type = jnp.asarray(topology.column.column_type, dtype=jnp.int32)
    surface_masks = build_surface_hydrology_masks(topology)
    snow_masks = build_snow_column_masks(
        topology,
        inputs.snow_layer_count_column,
    )
    nonlake = snow_masks.snow_column | snow_masks.no_snow_column
    urban = surface_masks.urban_column
    rural_nonlake = nonlake & ~urban
    hydrology = surface_masks.hydrology_column

    thickness = jnp.asarray(
        inputs.soil_snow_thickness_column_level,
        dtype=jnp.float64,
    )
    node_depth = jnp.asarray(
        inputs.soil_snow_node_depth_column_level,
        dtype=jnp.float64,
    )
    interface_depth = jnp.asarray(
        inputs.soil_snow_interface_depth_column_interface,
        dtype=jnp.float64,
    )
    temperature = jnp.asarray(
        inputs.soil_snow_temperature_column_level,
        dtype=jnp.float64,
    )
    ice = jnp.asarray(inputs.soil_snow_ice_column_level, dtype=jnp.float64)
    liquid = jnp.asarray(inputs.soil_snow_liquid_column_level, dtype=jnp.float64)
    ground_thickness = thickness[:, NUMBER_OF_SNOW_LEVELS:]
    ground_node_depth = node_depth[:, NUMBER_OF_SNOW_LEVELS:]
    ground_temperature = temperature[:, NUMBER_OF_SNOW_LEVELS:]
    ground_ice = ice[:, NUMBER_OF_SNOW_LEVELS:]
    ground_liquid = liquid[:, NUMBER_OF_SNOW_LEVELS:]
    ground_interfaces = interface_depth[:, NUMBER_OF_SNOW_LEVELS:]

    top_10cm_temperature = jnp.where(
        rural_nonlake,
        jnp.float64(0.0),
        jnp.asarray(state.top_10cm_soil_temperature_column, dtype=jnp.float64),
    )
    top_17cm_temperature = jnp.where(
        rural_nonlake,
        jnp.float64(0.0),
        jnp.asarray(state.top_17cm_soil_temperature_column, dtype=jnp.float64),
    )
    top_10cm_water = jnp.where(
        rural_nonlake,
        jnp.float64(0.0),
        jnp.asarray(state.top_10cm_soil_water_column, dtype=jnp.float64),
    )
    total_liquid = jnp.where(
        rural_nonlake,
        jnp.float64(0.0),
        jnp.asarray(state.total_soil_liquid_water_column, dtype=jnp.float64),
    )
    total_ice = jnp.where(
        rural_nonlake,
        jnp.float64(0.0),
        jnp.asarray(state.total_soil_ice_water_column, dtype=jnp.float64),
    )

    def accumulate_soil_layer(
        level: int,
        totals: tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ],
    ):
        temperature_10cm, temperature_17cm, water_10cm, liquid_total, ice_total = (
            totals
        )
        layer_temperature = ground_temperature[:, level]
        layer_thickness = ground_thickness[:, level]
        layer_liquid = ground_liquid[:, level]
        layer_ice = ground_ice[:, level]
        upper_interface = ground_interfaces[:, level]
        lower_interface = ground_interfaces[:, level + 1]
        fraction_10cm = _fraction_within_depth(
            upper_interface,
            lower_interface,
            layer_thickness,
            0.1,
        )
        fraction_17cm = _fraction_within_depth(
            upper_interface,
            lower_interface,
            layer_thickness,
            0.17,
        )
        contribution_10cm = fortran_multiply(
            fortran_multiply(layer_temperature, layer_thickness),
            fraction_10cm,
        )
        contribution_17cm = fortran_multiply(
            fortran_multiply(layer_temperature, layer_thickness),
            fraction_17cm,
        )
        water_contribution = fortran_multiply(
            fortran_add(layer_liquid, layer_ice),
            fraction_10cm,
        )
        return (
            jnp.where(
                rural_nonlake,
                fortran_add(temperature_10cm, contribution_10cm),
                temperature_10cm,
            ),
            jnp.where(
                rural_nonlake,
                fortran_add(temperature_17cm, contribution_17cm),
                temperature_17cm,
            ),
            jnp.where(
                rural_nonlake,
                fortran_add(water_10cm, water_contribution),
                water_10cm,
            ),
            jnp.where(
                rural_nonlake,
                fortran_add(liquid_total, layer_liquid),
                liquid_total,
            ),
            jnp.where(
                rural_nonlake,
                fortran_add(ice_total, layer_ice),
                ice_total,
            ),
        )

    (
        top_10cm_temperature,
        top_17cm_temperature,
        top_10cm_water,
        total_liquid,
        total_ice,
    ) = lax.fori_loop(
        0,
        NUMBER_OF_SOIL_LEVELS,
        accumulate_soil_layer,
        (
            top_10cm_temperature,
            top_17cm_temperature,
            top_10cm_water,
            total_liquid,
            total_ice,
        ),
    )
    near_surface_temperature = jnp.where(
        rural_nonlake,
        ground_temperature[:, 0],
        jnp.asarray(state.near_surface_soil_temperature_column, dtype=jnp.float64),
    )

    layer_count = jnp.asarray(inputs.snow_layer_count_column, dtype=jnp.int32)
    top_index = jnp.clip(
        layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS),
        jnp.int32(0),
        jnp.int32(NUMBER_OF_SOIL_SNOW_LEVELS - 1),
    )
    rows = jnp.arange(number_of_columns, dtype=jnp.int32)
    top_temperature = temperature[rows, top_index]
    snow_fraction = jnp.asarray(
        inputs.effective_snow_fraction_column,
        dtype=jnp.float64,
    )
    surface_water_fraction = jnp.asarray(
        inputs.surface_water_fraction_column,
        dtype=jnp.float64,
    )
    surface_water_temperature = jnp.asarray(
        inputs.surface_water_temperature_column,
        dtype=jnp.float64,
    )
    exposed_fraction = fortran_add(
        fortran_add(jnp.float64(1.0), -snow_fraction),
        -surface_water_fraction,
    )
    snow_ground_temperature = fortran_add(
        fortran_add(
            fortran_multiply(snow_fraction, top_temperature),
            fortran_multiply(exposed_fraction, ground_temperature[:, 0]),
        ),
        fortran_multiply(surface_water_fraction, surface_water_temperature),
    )
    no_snow_ground_temperature = fortran_add(
        fortran_multiply(
            fortran_add(jnp.float64(1.0), -surface_water_fraction),
            ground_temperature[:, 0],
        ),
        fortran_multiply(surface_water_fraction, surface_water_temperature),
    )
    diagnosed_ground_temperature = jnp.where(
        layer_count < jnp.int32(0),
        snow_ground_temperature,
        no_snow_ground_temperature,
    )
    diagnosed_ground_temperature = jnp.where(
        nonlake,
        diagnosed_ground_temperature,
        jnp.asarray(state.ground_temperature_column, dtype=jnp.float64),
    )
    urban_ground_temperature = jnp.where(
        urban,
        top_temperature,
        jnp.asarray(state.urban_ground_temperature_column, dtype=jnp.float64),
    )
    top_10cm_temperature = jnp.where(
        rural_nonlake,
        fortran_divide(top_10cm_temperature, jnp.float64(0.1)),
        top_10cm_temperature,
    )
    top_17cm_temperature = jnp.where(
        rural_nonlake,
        fortran_divide(top_17cm_temperature, jnp.float64(0.17)),
        top_17cm_temperature,
    )
    rural_ground_temperature = jnp.where(
        surface_masks.soil_crop_column,
        top_temperature,
        jnp.asarray(state.rural_ground_temperature_column, dtype=jnp.float64),
    )

    urban_restricted = urban & (
        (column_type == jnp.int32(COLUMN_URBAN_SUN_WALL))
        | (column_type == jnp.int32(COLUMN_URBAN_SHADE_WALL))
        | (column_type == jnp.int32(COLUMN_URBAN_ROOF))
    )
    ground_level_index = jnp.arange(NUMBER_OF_GROUND_LEVELS, dtype=jnp.int32)
    volumetric_update = nonlake[:, None] & ~(
        urban_restricted[:, None]
        & (ground_level_index[None, :] >= jnp.int32(NUMBER_OF_URBAN_LEVELS))
    )
    safe_thickness = jnp.where(
        volumetric_update,
        ground_thickness,
        jnp.float64(1.0),
    )
    volumetric_candidate = fortran_add(
        fortran_divide(
            ground_liquid,
            fortran_multiply(safe_thickness, jnp.float64(DENSITY_FRESH_WATER)),
        ),
        fortran_divide(
            ground_ice,
            fortran_multiply(safe_thickness, jnp.float64(DENSITY_ICE)),
        ),
    )
    volumetric_water = jnp.where(
        volumetric_update,
        volumetric_candidate,
        jnp.asarray(state.volumetric_soil_water_column_level, dtype=jnp.float64),
    )

    porosity = jnp.asarray(inputs.soil_porosity_column_level, dtype=jnp.float64)
    saturated_suction = jnp.asarray(
        inputs.saturated_suction_column_level,
        dtype=jnp.float64,
    )
    exponent = jnp.asarray(
        inputs.clapp_hornberger_exponent_column_level,
        dtype=jnp.float64,
    )
    safe_porosity = jnp.where(hydrology[:, None], porosity, jnp.float64(1.0))
    safe_suction = jnp.where(
        hydrology[:, None],
        saturated_suction,
        jnp.float64(1.0),
    )
    safe_exponent = jnp.where(hydrology[:, None], exponent, jnp.float64(1.0))
    safe_ground_thickness = jnp.where(
        hydrology[:, None],
        ground_thickness,
        jnp.float64(1.0),
    )
    volumetric_liquid = fortran_divide(
        ground_liquid,
        fortran_multiply(
            safe_ground_thickness,
            jnp.float64(DENSITY_FRESH_WATER),
        ),
    )
    fractional_saturation = jnp.maximum(
        fortran_divide(volumetric_liquid, safe_porosity),
        jnp.float64(MINIMUM_SOILPSI_FRACTIONAL_SATURATION),
    )
    potential = fortran_multiply(
        fortran_multiply(safe_suction, jnp.float64(-9.8e-6)),
        _retention_power(fractional_saturation, -safe_exponent),
    )
    potential = jnp.minimum(
        jnp.maximum(potential, jnp.float64(SOIL_WATER_POTENTIAL_MINIMUM_MPA)),
        jnp.float64(0.0),
    )
    positive_liquid = ground_liquid > jnp.float64(0.0)
    potential = jnp.where(
        positive_liquid,
        potential,
        jnp.float64(SOIL_WATER_POTENTIAL_MINIMUM_MPA),
    )
    soil_water_potential = jnp.where(
        hydrology[:, None],
        potential,
        jnp.asarray(state.soil_water_potential_column_level, dtype=jnp.float64),
    )

    matrix_saturation = jnp.minimum(
        jnp.float64(1.0),
        jnp.maximum(
            fortran_divide(volumetric_water, safe_porosity),
            jnp.float64(MINIMUM_MATRIX_POTENTIAL_SATURATION),
        ),
    )
    matrix_potential_candidate = fortran_multiply(
        -safe_suction,
        _retention_power(matrix_saturation, -safe_exponent),
    )
    matrix_potential_candidate = jnp.maximum(
        jnp.asarray(inputs.minimum_soil_matrix_potential_column, dtype=jnp.float64)[
            :, None
        ],
        matrix_potential_candidate,
    )
    matrix_potential = jnp.where(
        hydrology[:, None],
        matrix_potential_candidate,
        jnp.asarray(state.soil_matrix_potential_column_level, dtype=jnp.float64),
    )

    dry_water = _dry_water_content(safe_porosity, safe_suction, safe_exponent)

    def accumulate_water_availability(
        maximum_center_depth: float,
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def accumulate_level(
            level: int,
            values: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ):
            available_water, capacity, depth_sum = values
            center_depth = fortran_add(
                ground_node_depth[:, level],
                fortran_multiply(
                    jnp.float64(0.5),
                    ground_thickness[:, level],
                ),
            )
            selected = hydrology & (
                center_depth <= jnp.float64(maximum_center_depth)
            )
            water_increment = fortran_multiply(
                fortran_add(
                    volumetric_water[:, level],
                    -dry_water[:, level],
                ),
                ground_thickness[:, level],
            )
            capacity_increment = fortran_multiply(
                fortran_add(safe_porosity[:, level], -dry_water[:, level]),
                ground_thickness[:, level],
            )
            return (
                jnp.where(
                    selected,
                    fortran_add(available_water, water_increment),
                    available_water,
                ),
                jnp.where(
                    selected,
                    fortran_add(capacity, capacity_increment),
                    capacity,
                ),
                jnp.where(
                    selected,
                    fortran_add(depth_sum, ground_thickness[:, level]),
                    depth_sum,
                ),
            )

        return lax.fori_loop(
            0,
            NUMBER_OF_GROUND_LEVELS,
            accumulate_level,
            carry,
        )

    zero = jnp.zeros(number_of_columns, dtype=jnp.float64)
    available_water, available_capacity, selected_depth = (
        accumulate_water_availability(0.05, (zero, zero, zero))
    )

    def availability_fraction(
        available: jnp.ndarray,
        capacity: jnp.ndarray,
        depth_sum: jnp.ndarray,
    ) -> jnp.ndarray:
        nonzero_depth = depth_sum != jnp.float64(0.0)
        fallback_available = fortran_add(
            volumetric_water[:, 0],
            -dry_water[:, 0],
        )
        fallback_capacity = fortran_add(
            safe_porosity[:, 0],
            -dry_water[:, 0],
        )
        mean_available = jnp.where(
            nonzero_depth,
            fortran_divide(available, jnp.where(nonzero_depth, depth_sum, 1.0)),
            fallback_available,
        )
        mean_capacity = jnp.where(
            nonzero_depth,
            fortran_divide(capacity, jnp.where(nonzero_depth, depth_sum, 1.0)),
            fallback_capacity,
        )
        return fortran_divide(mean_available, mean_capacity)

    availability_5cm_candidate = availability_fraction(
        available_water,
        available_capacity,
        selected_depth,
    )
    availability_5cm = jnp.where(
        hydrology,
        availability_5cm_candidate,
        jnp.asarray(state.top_5cm_water_availability_column, dtype=jnp.float64),
    )
    available_water, available_capacity, selected_depth = (
        accumulate_water_availability(
            0.17,
            (available_water, available_capacity, selected_depth),
        )
    )
    availability_17cm_candidate = availability_fraction(
        available_water,
        available_capacity,
        selected_depth,
    )
    availability_17cm = jnp.where(
        hydrology,
        availability_17cm_candidate,
        jnp.asarray(state.top_17cm_water_availability_column, dtype=jnp.float64),
    )

    top_snow_water_candidate = fortran_add(
        ice[rows, top_index],
        liquid[rows, top_index],
    )
    top_snow_water = jnp.where(
        snow_masks.snow_column,
        top_snow_water_candidate,
        jnp.where(
            snow_masks.no_snow_column,
            jnp.float64(0.0),
            jnp.asarray(state.top_snow_water_column, dtype=jnp.float64),
        ),
    )
    snow_grain_radius = jnp.where(
        snow_masks.no_snow_column[:, None],
        jnp.float64(0.0),
        jnp.asarray(state.snow_grain_radius_column_level, dtype=jnp.float64),
    )
    special = jnp.float64(CLM_SPECIAL_VALUE)

    def clear_no_snow(values: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(snow_masks.no_snow_column, special, values)

    return HydrologyDiagnosticsResult(
        state=HydrologyDiagnosticsState(
            near_surface_soil_temperature_column=near_surface_temperature,
            top_10cm_soil_temperature_column=top_10cm_temperature,
            top_17cm_soil_temperature_column=top_17cm_temperature,
            top_10cm_soil_water_column=top_10cm_water,
            total_soil_liquid_water_column=total_liquid,
            total_soil_ice_water_column=total_ice,
            ground_temperature_column=diagnosed_ground_temperature,
            urban_ground_temperature_column=urban_ground_temperature,
            rural_ground_temperature_column=rural_ground_temperature,
            volumetric_soil_water_column_level=volumetric_water,
            soil_water_potential_column_level=soil_water_potential,
            soil_matrix_potential_column_level=matrix_potential,
            top_5cm_water_availability_column=availability_5cm,
            top_17cm_water_availability_column=availability_17cm,
            top_snow_water_column=top_snow_water,
            snow_grain_radius_column_level=snow_grain_radius,
            top_snow_temperature_column=clear_no_snow(
                jnp.asarray(state.top_snow_temperature_column, dtype=jnp.float64)
            ),
            top_snow_temperature_gradient_column=clear_no_snow(
                jnp.asarray(
                    state.top_snow_temperature_gradient_column,
                    dtype=jnp.float64,
                )
            ),
            top_snow_grain_radius_column=clear_no_snow(
                jnp.asarray(state.top_snow_grain_radius_column, dtype=jnp.float64)
            ),
            top_snow_liquid_fraction_column=clear_no_snow(
                jnp.asarray(state.top_snow_liquid_fraction_column, dtype=jnp.float64)
            ),
        ),
        masks=HydrologyDiagnosticsMasks(
            nonlake_column=nonlake,
            rural_nonlake_column=rural_nonlake,
            urban_column=urban,
            soil_crop_column=surface_masks.soil_crop_column,
            hydrology_column=hydrology,
            snow_column=snow_masks.snow_column,
            no_snow_column=snow_masks.no_snow_column,
            volumetric_update_column_level=volumetric_update,
        ),
    )
