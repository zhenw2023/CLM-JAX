
from __future__ import annotations

from contextvars import ContextVar
from functools import cache
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.constants import DENSITY_ICE
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import fortran_hardware_divide as fortran_divide
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.numerics import fortran_hardware_power
from clm_jax.physics.native_power import native_power
from clm_jax.physics.surface_hydrology import (
    NUMBER_OF_GROUND_LEVELS,
    NUMBER_OF_SOIL_LEVELS,
    build_surface_hydrology_masks,
)

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase

METER_TO_MILLIMETER = 1000.0
LIQUID_WATER_DENSITY = 1000.0
SOIL_ICE_IMPEDANCE_EXPONENT = 6.0
_USE_GPU_ADAPTIVE_EXIT = True
_USE_GPU_COMPACT_RICHARDS = True
_USE_GPU_HYDRAULIC_INVARIANTS = True
_adaptive_early_exit = ContextVar("clm_richards_early_exit", default=False)


def _hydraulic_power(base, exponent):
    if jax.default_backend() == "gpu":
        return native_power(base, exponent)
    return fortran_hardware_power(base, exponent)


class SoilWaterMovementParameters(NamedTuple):

    time_step_seconds: float = 1800.0
    minimum_substep_seconds: float = 60.0
    completion_tolerance: float = 1.0e-8
    upper_error_tolerance: float = 1.0e-1
    lower_error_tolerance: float = 1.0e-2
    maximum_adaptive_trials: int = 256
    use_flexible_cn: bool = True
    minimum_liquid_water_mass: float = 0.001

    @classmethod
    def from_case(cls, case: CimeCase) -> SoilWaterMovementParameters:

        from clm_jax.io.namelist import read_case_namelist_group  
        from clm_jax.io.restart import (  
            configured_land_restart,
            load_restart_clock,
        )

        movement = read_case_namelist_group(
            case,
            "lnd_in",
            "soilwater_movement_inparm",
        )
        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        if int(movement["soilwater_movement_method"]) != 1:
            raise ValueError("CLM-JAX currently requires moisture-form soil water method 1")
        if int(movement["upper_boundary_condition"]) != 1:
            raise ValueError("CLM-JAX currently requires the upper flux boundary")
        if int(movement["lower_boundary_condition"]) != 2:
            raise ValueError("CLM-JAX currently requires the lower zero-flux boundary")
        clock = load_restart_clock(configured_land_restart(case))
        return cls(
            time_step_seconds=float(clock.step_seconds),
            minimum_substep_seconds=float(movement["dtmin"]),
            completion_tolerance=float(movement["verysmall"]),
            upper_error_tolerance=float(movement["xtolerupper"]),
            lower_error_tolerance=float(movement["xtolerlower"]),
            use_flexible_cn=bool(general["use_flexiblecn"]),
        )


DEFAULT_SOIL_WATER_MOVEMENT_PARAMETERS = SoilWaterMovementParameters()


class SoilWaterMovementInputs(NamedTuple):

    soil_layer_count_column: jnp.ndarray
    layer_thickness_column_level: jnp.ndarray
    node_depth_column_level: jnp.ndarray
    ice_fraction_column_level: jnp.ndarray
    saturated_water_content_column_level: jnp.ndarray
    saturated_hydraulic_conductivity_column_level: jnp.ndarray
    clapp_hornberger_exponent_column_level: jnp.ndarray
    saturated_suction_column_level: jnp.ndarray
    infiltration_flux_column: jnp.ndarray
    root_water_sink_column_level: jnp.ndarray
    ice_water_mass_column_level: jnp.ndarray


class SoilWaterMovementState(NamedTuple):

    liquid_water_mass_column_level: jnp.ndarray
    volumetric_water_column_level: jnp.ndarray
    soil_water_potential_column_level: jnp.ndarray
    hydraulic_conductivity_column_level: jnp.ndarray
    aquifer_water_column: jnp.ndarray
    recharge_flux_column: jnp.ndarray
    adaptive_substep_count_column: jnp.ndarray


class SoilWaterMovementDiagnostics(NamedTuple):

    hydrology_column: jnp.ndarray
    adaptive_loop_completed_column: jnp.ndarray
    final_maximum_flux_error_column: jnp.ndarray


class SoilWaterMovementResult(NamedTuple):
    state: SoilWaterMovementState
    diagnostics: SoilWaterMovementDiagnostics


class RichardsMoistureTrialInputs(NamedTuple):

    soil_layer_count: jnp.ndarray
    substep_seconds: jnp.ndarray
    liquid_water_mass_level: jnp.ndarray
    layer_thickness_level: jnp.ndarray
    node_depth_level: jnp.ndarray
    ice_fraction_level: jnp.ndarray
    saturated_water_content_level: jnp.ndarray
    saturated_hydraulic_conductivity_level: jnp.ndarray
    clapp_hornberger_exponent_level: jnp.ndarray
    saturated_suction_level: jnp.ndarray
    infiltration_flux: jnp.ndarray
    root_water_sink_level: jnp.ndarray


class RichardsHydraulicProperties(NamedTuple):

    volumetric_liquid_water_level: jnp.ndarray
    relative_saturation_level: jnp.ndarray
    hydraulic_conductivity_level: jnp.ndarray
    soil_water_potential_level: jnp.ndarray
    conductivity_derivative_interface_level: jnp.ndarray
    potential_derivative_water_level: jnp.ndarray
    ice_impedance_level: jnp.ndarray


class RichardsMoistureFluxes(NamedTuple):

    incoming_flux_level: jnp.ndarray
    outgoing_flux_level: jnp.ndarray
    incoming_derivative_above_level: jnp.ndarray
    incoming_derivative_level: jnp.ndarray
    outgoing_derivative_level: jnp.ndarray
    outgoing_derivative_below_level: jnp.ndarray


class RichardsMoistureMatrix(NamedTuple):

    lower_diagonal_level: jnp.ndarray
    diagonal_level: jnp.ndarray
    upper_diagonal_level: jnp.ndarray
    right_hand_side_level: jnp.ndarray


class RichardsMoistureTrialResult(NamedTuple):

    hydraulic: RichardsHydraulicProperties
    fluxes: RichardsMoistureFluxes
    matrix: RichardsMoistureMatrix
    water_content_increment_level: jnp.ndarray
    flux_error_level: jnp.ndarray
    maximum_flux_error: jnp.ndarray
    time_over_depth_level: jnp.ndarray


def _validate_trial_inputs(inputs: RichardsMoistureTrialInputs) -> None:
    level_fields = (
        "liquid_water_mass_level",
        "layer_thickness_level",
        "node_depth_level",
        "ice_fraction_level",
        "saturated_water_content_level",
        "saturated_hydraulic_conductivity_level",
        "clapp_hornberger_exponent_level",
        "saturated_suction_level",
        "root_water_sink_level",
    )
    expected = (NUMBER_OF_SOIL_LEVELS,)
    for name in level_fields:
        if getattr(inputs, name).shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    for name in ("soil_layer_count", "substep_seconds", "infiltration_flux"):
        if jnp.shape(getattr(inputs, name)) != ():
            raise ValueError(f"{name} must be scalar")


def _active_level_mask(soil_layer_count: jnp.ndarray) -> jnp.ndarray:
    levels = jnp.arange(NUMBER_OF_SOIL_LEVELS, dtype=jnp.int32)
    return levels < jnp.asarray(soil_layer_count, dtype=jnp.int32)


class _HydraulicInvariants(NamedTuple):
    water_mass_capacity: jnp.ndarray
    millimeter_depth: jnp.ndarray
    ice_impedance: jnp.ndarray
    conductivity_exponent: jnp.ndarray
    conductivity_scale: jnp.ndarray


def _hydraulic_invariants(layer_count, thickness, ice_fraction, exponent, conductivity):
    last = jnp.arange(NUMBER_OF_SOIL_LEVELS, dtype=jnp.int32) == layer_count - 1
    ice_fraction = jnp.asarray(ice_fraction, dtype=jnp.float64)
    next_ice = jnp.concatenate((ice_fraction[1:], ice_fraction[-1:]))
    interface_ice = fortran_multiply(jnp.float64(0.5), fortran_add(ice_fraction, next_ice))
    interface_ice = jnp.where(last, ice_fraction, interface_ice)
    impedance = _hydraulic_power(
        jnp.float64(10.0),
        fortran_multiply(jnp.float64(-SOIL_ICE_IMPEDANCE_EXPONENT), interface_ice),
    )
    return _HydraulicInvariants(
        fortran_multiply(thickness, jnp.float64(LIQUID_WATER_DENSITY)),
        fortran_multiply(jnp.float64(METER_TO_MILLIMETER), thickness),
        impedance,
        fortran_add(fortran_multiply(jnp.float64(2.0), exponent), jnp.float64(3.0)),
        fortran_multiply(impedance, conductivity),
    )


def _hydraulic_properties(
    inputs: RichardsMoistureTrialInputs,
    fixed: _HydraulicInvariants | None = None,
) -> tuple[RichardsHydraulicProperties, jnp.ndarray]:

    layer_count = jnp.asarray(inputs.soil_layer_count, dtype=jnp.int32)
    active = _active_level_mask(layer_count)
    last = jnp.arange(NUMBER_OF_SOIL_LEVELS, dtype=jnp.int32) == layer_count - 1
    zero = jnp.float64(0.0)

    bounded_water_mass = jnp.maximum(
        jnp.asarray(inputs.liquid_water_mass_level, dtype=jnp.float64),
        jnp.float64(1.0e-6),
    )
    if fixed is None:
        fixed = _hydraulic_invariants(
            layer_count, inputs.layer_thickness_level, inputs.ice_fraction_level,
            inputs.clapp_hornberger_exponent_level,
            inputs.saturated_hydraulic_conductivity_level,
        )
    water_mass_capacity = fixed.water_mass_capacity
    volumetric_water = fortran_divide(bounded_water_mass, water_mass_capacity)
    time_over_depth = fortran_divide(
        jnp.asarray(inputs.substep_seconds, dtype=jnp.float64),
        fixed.millimeter_depth,
    )

    relative_saturation = fortran_divide(
        volumetric_water,
        inputs.saturated_water_content_level,
    )
    relative_saturation = jnp.minimum(relative_saturation, jnp.float64(1.0))
    relative_saturation = jnp.maximum(relative_saturation, jnp.float64(0.01))

    next_saturation = jnp.concatenate(
        (relative_saturation[1:], relative_saturation[-1:])
    )
    interface_saturation = fortran_multiply(
        jnp.float64(0.5),
        fortran_add(relative_saturation, next_saturation),
    )
    interface_saturation = jnp.where(
        last,
        relative_saturation,
        interface_saturation,
    )
    interface_saturation = jnp.minimum(interface_saturation, jnp.float64(1.0))
    interface_saturation = jnp.maximum(interface_saturation, jnp.float64(0.01))

    ice_impedance = fixed.ice_impedance
    conductivity_exponent = fixed.conductivity_exponent
    conductivity = fortran_multiply(
        fixed.conductivity_scale,
        _hydraulic_power(interface_saturation, conductivity_exponent),
    )
    conductivity_derivative = fortran_divide(
        fortran_multiply(conductivity_exponent, conductivity),
        interface_saturation,
    )

    suction_power = _hydraulic_power(
        relative_saturation,
        -jnp.asarray(inputs.clapp_hornberger_exponent_level, dtype=jnp.float64),
    )
    water_potential = fortran_multiply(
        -jnp.asarray(inputs.saturated_suction_level, dtype=jnp.float64),
        suction_power,
    )
    potential_derivative_saturation = fortran_divide(
        fortran_multiply(
            -jnp.asarray(inputs.clapp_hornberger_exponent_level, dtype=jnp.float64),
            water_potential,
        ),
        relative_saturation,
    )
    potential_derivative_water = fortran_divide(
        potential_derivative_saturation,
        inputs.saturated_water_content_level,
    )

    def mask(values: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(active, values, zero)

    return (
        RichardsHydraulicProperties(
            volumetric_liquid_water_level=mask(volumetric_water),
            relative_saturation_level=mask(relative_saturation),
            hydraulic_conductivity_level=mask(conductivity),
            soil_water_potential_level=mask(water_potential),
            conductivity_derivative_interface_level=mask(conductivity_derivative),
            potential_derivative_water_level=mask(potential_derivative_water),
            ice_impedance_level=mask(ice_impedance),
        ),
        mask(time_over_depth),
    )


def _moisture_fluxes(
    inputs: RichardsMoistureTrialInputs,
    hydraulic: RichardsHydraulicProperties,
) -> RichardsMoistureFluxes:

    layer_count = jnp.asarray(inputs.soil_layer_count, dtype=jnp.int32)
    level = jnp.arange(NUMBER_OF_SOIL_LEVELS, dtype=jnp.int32)
    active = level < layer_count
    interface_active = level < layer_count - 1
    zero = jnp.float64(0.0)

    next_potential = jnp.concatenate(
        (
            hydraulic.soil_water_potential_level[1:],
            hydraulic.soil_water_potential_level[-1:],
        )
    )
    next_potential_derivative = jnp.concatenate(
        (
            hydraulic.potential_derivative_water_level[1:],
            hydraulic.potential_derivative_water_level[-1:],
        )
    )
    next_porosity = jnp.concatenate(
        (
            inputs.saturated_water_content_level[1:],
            inputs.saturated_water_content_level[-1:],
        )
    )
    next_depth = jnp.concatenate(
        (inputs.node_depth_level[1:], inputs.node_depth_level[-1:])
    )

    conductivity_derivative_above = fortran_divide(
        fortran_multiply(
            jnp.float64(0.5),
            hydraulic.conductivity_derivative_interface_level,
        ),
        inputs.saturated_water_content_level,
    )
    conductivity_derivative_below = fortran_divide(
        fortran_multiply(
            jnp.float64(0.5),
            hydraulic.conductivity_derivative_interface_level,
        ),
        next_porosity,
    )
    potential_difference = fortran_add(
        next_potential,
        -hydraulic.soil_water_potential_level,
    )
    depth_difference = fortran_add(next_depth, -inputs.node_depth_level)
    denominator = fortran_multiply(
        jnp.float64(METER_TO_MILLIMETER),
        depth_difference,
    )
    safe_denominator = jnp.where(
        interface_active,
        denominator,
        jnp.float64(1.0),
    )

    potential_flux = fortran_divide(
        fortran_multiply(-hydraulic.hydraulic_conductivity_level, potential_difference),
        safe_denominator,
    )
    outgoing = fortran_add(
        potential_flux,
        hydraulic.hydraulic_conductivity_level,
    )

    derivative_above_numerator = fortran_add(
        fortran_multiply(
            hydraulic.hydraulic_conductivity_level,
            hydraulic.potential_derivative_water_level,
        ),
        -fortran_multiply(conductivity_derivative_above, potential_difference),
    )
    outgoing_derivative = fortran_add(
        fortran_divide(derivative_above_numerator, safe_denominator),
        conductivity_derivative_above,
    )

    derivative_below_numerator = fortran_add(
        fortran_multiply(
            -hydraulic.hydraulic_conductivity_level,
            next_potential_derivative,
        ),
        -fortran_multiply(conductivity_derivative_below, potential_difference),
    )
    outgoing_derivative_below = fortran_add(
        fortran_divide(derivative_below_numerator, safe_denominator),
        conductivity_derivative_below,
    )

    outgoing = jnp.where(interface_active, outgoing, zero)
    outgoing_derivative = jnp.where(interface_active, outgoing_derivative, zero)
    outgoing_derivative_below = jnp.where(
        interface_active,
        outgoing_derivative_below,
        zero,
    )

    previous_outgoing = jnp.concatenate((jnp.zeros((1,), dtype=jnp.float64), outgoing[:-1]))
    previous_derivative = jnp.concatenate(
        (jnp.zeros((1,), dtype=jnp.float64), outgoing_derivative[:-1])
    )
    previous_derivative_below = jnp.concatenate(
        (jnp.zeros((1,), dtype=jnp.float64), outgoing_derivative_below[:-1])
    )
    incoming = jnp.where(
        level == 0,
        jnp.asarray(inputs.infiltration_flux, dtype=jnp.float64),
        previous_outgoing,
    )
    incoming_derivative_above = jnp.where(level == 0, zero, previous_derivative)
    incoming_derivative = jnp.where(level == 0, zero, previous_derivative_below)

    return RichardsMoistureFluxes(
        incoming_flux_level=jnp.where(active, incoming, zero),
        outgoing_flux_level=outgoing,
        incoming_derivative_above_level=jnp.where(
            active,
            incoming_derivative_above,
            zero,
        ),
        incoming_derivative_level=jnp.where(active, incoming_derivative, zero),
        outgoing_derivative_level=outgoing_derivative,
        outgoing_derivative_below_level=outgoing_derivative_below,
    )


def _assemble_moisture_matrix(
    inputs: RichardsMoistureTrialInputs,
    fluxes: RichardsMoistureFluxes,
    time_over_depth: jnp.ndarray,
) -> RichardsMoistureMatrix:

    layer_count = jnp.asarray(inputs.soil_layer_count, dtype=jnp.int32)
    level = jnp.arange(NUMBER_OF_SOIL_LEVELS, dtype=jnp.int32)
    active = level < layer_count
    interior_or_bottom = (level > 0) & active
    not_bottom = level < layer_count - 1
    zero = jnp.float64(0.0)

    net_flux = fortran_add(
        fortran_add(fluxes.incoming_flux_level, -fluxes.outgoing_flux_level),
        -inputs.root_water_sink_level,
    )
    right_hand_side = fortran_multiply(-net_flux, time_over_depth)

    derivative_difference = fortran_add(
        -fluxes.incoming_derivative_level,
        fluxes.outgoing_derivative_level,
    )
    diagonal = fortran_add(
        jnp.float64(-1.0),
        -fortran_multiply(derivative_difference, time_over_depth),
    )
    lower = fortran_multiply(
        fluxes.incoming_derivative_above_level,
        time_over_depth,
    )
    upper = -fortran_multiply(
        fluxes.outgoing_derivative_below_level,
        time_over_depth,
    )

    return RichardsMoistureMatrix(
        lower_diagonal_level=jnp.where(interior_or_bottom, lower, zero),
        diagonal_level=jnp.where(active, diagonal, zero),
        upper_diagonal_level=jnp.where(active & not_bottom, upper, zero),
        right_hand_side_level=jnp.where(active, right_hand_side, zero),
    )


def _solve_active_dgtsv(
    matrix: RichardsMoistureMatrix,
    soil_layer_count: jnp.ndarray,
) -> jnp.ndarray:

    if jax.default_backend() == "gpu":
        from clm_jax.physics.richards_gpu import pivoted_dgtsv

        return pivoted_dgtsv(matrix, soil_layer_count)

    active = jnp.arange(NUMBER_OF_SOIL_LEVELS, dtype=jnp.int32) < soil_layer_count
    diagonal = jnp.where(active, matrix.diagonal_level, jnp.float64(1.0))
    solution = lax.linalg.tridiagonal_solve(
        matrix.lower_diagonal_level,
        diagonal,
        matrix.upper_diagonal_level,
        matrix.right_hand_side_level[:, None],
    )[:, 0]
    return jnp.where(active, solution, jnp.float64(0.0))


def richards_moisture_trial(
    inputs: RichardsMoistureTrialInputs,
    *,
    fixed: _HydraulicInvariants | None = None,
) -> RichardsMoistureTrialResult:

    _validate_trial_inputs(inputs)
    hydraulic, time_over_depth = _hydraulic_properties(inputs, fixed)
    fluxes = _moisture_fluxes(inputs, hydraulic)
    matrix = _assemble_moisture_matrix(inputs, fluxes, time_over_depth)
    water_increment = _solve_active_dgtsv(matrix, inputs.soil_layer_count)

    active = _active_level_mask(inputs.soil_layer_count)
    safe_time_over_depth = jnp.where(
        active,
        time_over_depth,
        jnp.float64(1.0),
    )
    flux_from_increment = fortran_divide(water_increment, safe_time_over_depth)
    flux_at_start = fortran_add(
        fortran_add(fluxes.incoming_flux_level, -fluxes.outgoing_flux_level),
        -inputs.root_water_sink_level,
    )
    flux_error = fortran_multiply(
        fortran_multiply(
            jnp.abs(fortran_add(flux_at_start, -flux_from_increment)),
            jnp.asarray(inputs.substep_seconds, dtype=jnp.float64),
        ),
        jnp.float64(0.5),
    )
    flux_error = jnp.where(active, flux_error, jnp.float64(0.0))
    maximum_error = jnp.max(
        jnp.where(active, flux_error, jnp.float64(-jnp.inf))
    )
    return RichardsMoistureTrialResult(
        hydraulic=hydraulic,
        fluxes=fluxes,
        matrix=matrix,
        water_content_increment_level=water_increment,
        flux_error_level=flux_error,
        maximum_flux_error=maximum_error,
        time_over_depth_level=time_over_depth,
    )


class _AdaptiveColumnCarry(NamedTuple):
    liquid_water_mass_level: jnp.ndarray
    substep_seconds: jnp.ndarray
    completed_seconds: jnp.ndarray
    trial_count: jnp.ndarray
    completed: jnp.ndarray
    soil_water_potential_level: jnp.ndarray
    hydraulic_conductivity_level: jnp.ndarray
    maximum_flux_error: jnp.ndarray


class _AdaptiveColumnResult(NamedTuple):
    liquid_water_mass_level: jnp.ndarray
    soil_water_potential_level: jnp.ndarray
    hydraulic_conductivity_level: jnp.ndarray
    trial_count: jnp.ndarray
    completed: jnp.ndarray
    maximum_flux_error: jnp.ndarray


def _advance_unfinished_columns(trial, current, *column_inputs, capacities=(2048, 16384)):
    if any(size < 2 for size in capacities):
        raise ValueError("Richards packed capacities must be at least two columns")
    count = current.completed.size
    capacities = tuple(size for size in capacities if size < count)
    if not capacities:
        return jax.vmap(trial)(current, *column_inputs)
    running = ~current.completed
    active_count = jnp.sum(running, dtype=jnp.int32)

    def packed(size):
        def execute(_):
            index = jnp.nonzero(running, size=size, fill_value=count)[0]
            safe_index = jnp.where(index < count, index, index[0])
            values = jax.tree.map(lambda value: value[safe_index], (current, column_inputs))
            result = jax.vmap(trial)(values[0], *values[1])
            return jax.tree.map(
                lambda original, updated: original.at[index].set(updated, mode="drop"),
                current, result,
            )
        return execute

    selector = jnp.int32(len(capacities))
    for branch in reversed(range(len(capacities))):
        selector = jnp.where((active_count > 0) & (active_count <= capacities[branch]),
                             jnp.int32(branch), selector)
    branches = tuple(packed(size) for size in capacities)
    return lax.switch(
        selector, (*branches, lambda _: jax.vmap(trial)(current, *column_inputs)), None,
    )


def _adaptive_moisture_columns_reference(
    soil_layer_count: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    node_depth: jnp.ndarray,
    ice_fraction: jnp.ndarray,
    saturated_water_content: jnp.ndarray,
    saturated_hydraulic_conductivity: jnp.ndarray,
    clapp_hornberger_exponent: jnp.ndarray,
    saturated_suction: jnp.ndarray,
    infiltration_flux: jnp.ndarray,
    root_water_sink: jnp.ndarray,
    liquid_water_mass: jnp.ndarray,
    initial_soil_water_potential: jnp.ndarray,
    initial_hydraulic_conductivity: jnp.ndarray,
    parameters: SoilWaterMovementParameters,
    *,
    compact_unfinished_columns: bool = False,
    precompute_invariants: bool = False,
) -> _AdaptiveColumnResult:

    time_step = jnp.float64(parameters.time_step_seconds)
    minimum_substep = jnp.float64(parameters.minimum_substep_seconds)
    completion_tolerance = jnp.float64(parameters.completion_tolerance)
    upper_tolerance = jnp.float64(parameters.upper_error_tolerance)
    lower_tolerance = jnp.float64(parameters.lower_error_tolerance)
    fixed = (
        jax.vmap(_hydraulic_invariants)(
            soil_layer_count, layer_thickness, ice_fraction,
            clapp_hornberger_exponent, saturated_hydraulic_conductivity,
        ) if precompute_invariants else None
    )
    initial = _AdaptiveColumnCarry(
        liquid_water_mass_level=liquid_water_mass,
        substep_seconds=jnp.full_like(infiltration_flux, time_step),
        completed_seconds=jnp.zeros_like(infiltration_flux),
        trial_count=jnp.zeros_like(soil_layer_count, dtype=jnp.int32),
        completed=jnp.zeros_like(soil_layer_count, dtype=jnp.bool_),
        soil_water_potential_level=initial_soil_water_potential,
        hydraulic_conductivity_level=initial_hydraulic_conductivity,
        maximum_flux_error=jnp.zeros_like(infiltration_flux),
    )

    def adaptive_trial(
        carry: _AdaptiveColumnCarry,
        soil_layer_count,
        layer_thickness,
        node_depth,
        ice_fraction,
        saturated_water_content,
        saturated_hydraulic_conductivity,
        clapp_hornberger_exponent,
        saturated_suction,
        infiltration_flux,
        root_water_sink,
        fixed,
    ) -> _AdaptiveColumnCarry:
        active_level = _active_level_mask(soil_layer_count)

        def execute(current: _AdaptiveColumnCarry) -> _AdaptiveColumnCarry:
            trial = richards_moisture_trial(
                RichardsMoistureTrialInputs(
                    soil_layer_count=soil_layer_count,
                    substep_seconds=current.substep_seconds,
                    liquid_water_mass_level=current.liquid_water_mass_level,
                    layer_thickness_level=layer_thickness,
                    node_depth_level=node_depth,
                    ice_fraction_level=ice_fraction,
                    saturated_water_content_level=saturated_water_content,
                    saturated_hydraulic_conductivity_level=(
                        saturated_hydraulic_conductivity
                    ),
                    clapp_hornberger_exponent_level=clapp_hornberger_exponent,
                    saturated_suction_level=saturated_suction,
                    infiltration_flux=infiltration_flux,
                    root_water_sink_level=root_water_sink,
                ), fixed=fixed,
            )
            trial_count = current.trial_count + jnp.int32(1)
            potential = jnp.where(
                active_level,
                trial.hydraulic.soil_water_potential_level,
                current.soil_water_potential_level,
            )
            conductivity = jnp.where(
                active_level,
                trial.hydraulic.hydraulic_conductivity_level,
                current.hydraulic_conductivity_level,
            )
            reject = (
                trial.maximum_flux_error > upper_tolerance
            ) & (current.substep_seconds > minimum_substep)

            def reject_trial(_: None) -> _AdaptiveColumnCarry:
                next_substep = jnp.maximum(
                    fortran_divide(current.substep_seconds, jnp.float64(2.0)),
                    minimum_substep,
                )
                return _AdaptiveColumnCarry(
                    liquid_water_mass_level=current.liquid_water_mass_level,
                    substep_seconds=next_substep,
                    completed_seconds=current.completed_seconds,
                    trial_count=trial_count,
                    completed=jnp.bool_(False),
                    soil_water_potential_level=potential,
                    hydraulic_conductivity_level=conductivity,
                    maximum_flux_error=trial.maximum_flux_error,
                )

            def accept_trial(_: None) -> _AdaptiveColumnCarry:
                layer_mass_scale = fortran_multiply(
                    jnp.float64(METER_TO_MILLIMETER),
                    layer_thickness,
                )
                mass_increment = fortran_multiply(
                    trial.water_content_increment_level,
                    layer_mass_scale,
                )
                next_liquid = fortran_add(
                    current.liquid_water_mass_level,
                    mass_increment,
                )
                completed_seconds = fortran_add(
                    current.completed_seconds,
                    current.substep_seconds,
                )
                completed = (
                    jnp.abs(fortran_add(time_step, -completed_seconds))
                    < completion_tolerance
                )
                doubled_substep = fortran_multiply(
                    current.substep_seconds,
                    jnp.float64(2.0),
                )
                candidate_substep = jnp.where(
                    trial.maximum_flux_error < lower_tolerance,
                    doubled_substep,
                    current.substep_seconds,
                )
                remaining = fortran_add(time_step, -completed_seconds)
                next_substep = jnp.minimum(candidate_substep, remaining)
                next_substep = jnp.where(
                    completed,
                    current.substep_seconds,
                    next_substep,
                )
                return _AdaptiveColumnCarry(
                    liquid_water_mass_level=next_liquid,
                    substep_seconds=next_substep,
                    completed_seconds=completed_seconds,
                    trial_count=trial_count,
                    completed=completed,
                    soil_water_potential_level=potential,
                    hydraulic_conductivity_level=conductivity,
                    maximum_flux_error=trial.maximum_flux_error,
                )

            return lax.cond(reject, reject_trial, accept_trial, operand=None)

        return lax.cond(carry.completed, lambda value: value, execute, carry)

    def iteration(_, carry):
        def advance(current):
            if compact_unfinished_columns:
                return _advance_unfinished_columns(
                    adaptive_trial, current, soil_layer_count, layer_thickness,
                    node_depth, ice_fraction, saturated_water_content,
                    saturated_hydraulic_conductivity, clapp_hornberger_exponent,
                    saturated_suction, infiltration_flux, root_water_sink, fixed,
                )
            return jax.vmap(adaptive_trial)(
                current,
                soil_layer_count,
                layer_thickness,
                node_depth,
                ice_fraction,
                saturated_water_content,
                saturated_hydraulic_conductivity,
                clapp_hornberger_exponent,
                saturated_suction,
                infiltration_flux,
                root_water_sink,
                fixed,
            )

        return lax.cond(jnp.all(carry.completed), lambda current: current, advance, carry)

    if _adaptive_early_exit.get():
        def condition(carry):
            index, current = carry
            return (index < parameters.maximum_adaptive_trials) & ~jnp.all(current.completed)

        def advance(carry):
            index, current = carry
            return index + jnp.int32(1), iteration(index, current)

        final = lax.while_loop(condition, advance, (jnp.int32(0), initial))[1]
    else:
        from clm_jax.physics.converged_iteration import checkpointed_fori_loop

        final = checkpointed_fori_loop(
            0,
            parameters.maximum_adaptive_trials,
            iteration,
            initial,
        )
    return _AdaptiveColumnResult(
        liquid_water_mass_level=final.liquid_water_mass_level,
        soil_water_potential_level=final.soil_water_potential_level,
        hydraulic_conductivity_level=final.hydraulic_conductivity_level,
        trial_count=final.trial_count,
        completed=final.completed,
        maximum_flux_error=final.maximum_flux_error,
    )


@cache
def _adaptive_gpu_program(parameters):
    def reference(arguments, work_hint):
        return _adaptive_moisture_columns_reference(*arguments, parameters)

    @jax.custom_jvp
    def execute(arguments, work_hint):
        if jax.default_backend() != "gpu":
            token = _adaptive_early_exit.set(True)
            try:
                return _adaptive_moisture_columns_reference(
                    *arguments, parameters,
                    compact_unfinished_columns=_USE_GPU_COMPACT_RICHARDS,
                    precompute_invariants=_USE_GPU_HYDRAULIC_INVARIANTS,
                )
            finally:
                _adaptive_early_exit.reset(token)
        from clm_jax.physics.richards_gpu import adaptive_columns

        return adaptive_columns(arguments, parameters, work_hint)

    @execute.defjvp
    def derivative(primals, tangents):
        token = _adaptive_early_exit.set(False)
        try:
            return jax.jvp(reference, primals, tangents)
        finally:
            _adaptive_early_exit.reset(token)

    def dispatch(*arguments, work_hint=None):
        return execute(arguments, work_hint)

    return dispatch


def _adaptive_moisture_columns(*arguments, work_hint=None):
    *values, parameters = arguments
    if _USE_GPU_ADAPTIVE_EXIT and jax.default_backend() == "gpu":
        return _adaptive_gpu_program(parameters)(*values, work_hint=work_hint)
    return _adaptive_moisture_columns_reference(*values, parameters)


def _flexible_cn_liquid_cleanup(
    liquid_water_mass: jnp.ndarray,
    aquifer_water: jnp.ndarray,
    minimum_water_mass: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    def move_deficit(level: int, liquid: jnp.ndarray) -> jnp.ndarray:
        deficit = jnp.where(
            liquid[level] < jnp.float64(0.0),
            fortran_add(minimum_water_mass, -liquid[level]),
            jnp.float64(0.0),
        )
        liquid = liquid.at[level].set(fortran_add(liquid[level], deficit))
        return liquid.at[level + 1].set(
            fortran_add(liquid[level + 1], -deficit)
        )

    liquid_water_mass = lax.fori_loop(
        0,
        NUMBER_OF_SOIL_LEVELS - 1,
        move_deficit,
        liquid_water_mass,
    )
    bottom = NUMBER_OF_SOIL_LEVELS - 1
    bottom_deficit = jnp.where(
        liquid_water_mass[bottom] < minimum_water_mass,
        fortran_add(minimum_water_mass, -liquid_water_mass[bottom]),
        jnp.float64(0.0),
    )
    liquid_water_mass = liquid_water_mass.at[bottom].set(
        fortran_add(liquid_water_mass[bottom], bottom_deficit)
    )
    aquifer_water = fortran_add(aquifer_water, -bottom_deficit)
    return liquid_water_mass, aquifer_water


def _validate_movement_shapes(
    topology: SubgridTopology,
    inputs: SoilWaterMovementInputs,
    state: SoilWaterMovementState,
) -> None:
    column_shape = topology.column.gridcell.shape
    if len(column_shape) != 1:
        raise ValueError("column topology must be a vector")
    ground_shape = (column_shape[0], NUMBER_OF_GROUND_LEVELS)
    soil_shape = (column_shape[0], NUMBER_OF_SOIL_LEVELS)
    input_ground_fields = {
        "layer_thickness_column_level",
        "node_depth_column_level",
        "ice_fraction_column_level",
        "saturated_water_content_column_level",
        "saturated_hydraulic_conductivity_column_level",
        "clapp_hornberger_exponent_column_level",
        "saturated_suction_column_level",
        "ice_water_mass_column_level",
    }
    for name in input_ground_fields:
        if getattr(inputs, name).shape != ground_shape:
            raise ValueError(f"{name} must have shape {ground_shape}")
    if inputs.root_water_sink_column_level.shape != soil_shape:
        raise ValueError(f"root_water_sink_column_level must have shape {soil_shape}")
    for name in {"soil_layer_count_column", "infiltration_flux_column"}:
        if getattr(inputs, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")

    state_ground_fields = {
        "liquid_water_mass_column_level",
        "volumetric_water_column_level",
        "soil_water_potential_column_level",
        "hydraulic_conductivity_column_level",
    }
    for name in state_ground_fields:
        if getattr(state, name).shape != ground_shape:
            raise ValueError(f"{name} must have shape {ground_shape}")
    for name in set(SoilWaterMovementState._fields) - state_ground_fields:
        if getattr(state, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")


def soil_water_movement_active_case(
    topology: SubgridTopology,
    inputs: SoilWaterMovementInputs,
    state: SoilWaterMovementState,
    *,
    parameters: SoilWaterMovementParameters = DEFAULT_SOIL_WATER_MOVEMENT_PARAMETERS,
) -> SoilWaterMovementResult:

    _validate_movement_shapes(topology, inputs, state)
    if parameters.maximum_adaptive_trials < 1:
        raise ValueError("maximum_adaptive_trials must be positive")
    if parameters.minimum_substep_seconds <= 0.0:
        raise ValueError("minimum_substep_seconds must be positive")
    if parameters.time_step_seconds < parameters.minimum_substep_seconds:
        raise ValueError("time_step_seconds must be at least minimum_substep_seconds")

    hydrology = build_surface_hydrology_masks(topology).hydrology_column
    soil_slice = slice(0, NUMBER_OF_SOIL_LEVELS)
    soil_level = jnp.arange(NUMBER_OF_SOIL_LEVELS, dtype=jnp.float64)
    safe_thickness = jnp.full(
        (NUMBER_OF_SOIL_LEVELS,),
        jnp.float64(0.1),
    )
    safe_depth = fortran_add(
        jnp.float64(0.05),
        fortran_multiply(jnp.float64(0.1), soil_level),
    )
    safe_liquid = jnp.full(
        (NUMBER_OF_SOIL_LEVELS,),
        jnp.float64(20.0),
    )

    def active_rows(values: jnp.ndarray, fallback: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(hydrology[:, None], values, fallback[None, :])

    layer_count = jnp.where(
        hydrology,
        inputs.soil_layer_count_column,
        jnp.int32(2),
    )
    layer_thickness = active_rows(
        inputs.layer_thickness_column_level[:, soil_slice],
        safe_thickness,
    )
    node_depth = active_rows(
        inputs.node_depth_column_level[:, soil_slice],
        safe_depth,
    )
    ice_fraction = active_rows(
        inputs.ice_fraction_column_level[:, soil_slice],
        jnp.zeros((NUMBER_OF_SOIL_LEVELS,), dtype=jnp.float64),
    )
    saturated_water_content = active_rows(
        inputs.saturated_water_content_column_level[:, soil_slice],
        jnp.full((NUMBER_OF_SOIL_LEVELS,), jnp.float64(0.4)),
    )
    saturated_hydraulic_conductivity = active_rows(
        inputs.saturated_hydraulic_conductivity_column_level[:, soil_slice],
        jnp.full((NUMBER_OF_SOIL_LEVELS,), jnp.float64(1.0e-4)),
    )
    clapp_hornberger_exponent = active_rows(
        inputs.clapp_hornberger_exponent_column_level[:, soil_slice],
        jnp.full((NUMBER_OF_SOIL_LEVELS,), jnp.float64(4.0)),
    )
    saturated_suction = active_rows(
        inputs.saturated_suction_column_level[:, soil_slice],
        jnp.full((NUMBER_OF_SOIL_LEVELS,), jnp.float64(100.0)),
    )
    infiltration_flux = jnp.where(
        hydrology,
        inputs.infiltration_flux_column,
        jnp.float64(0.0),
    )
    root_water_sink = active_rows(
        inputs.root_water_sink_column_level,
        jnp.zeros((NUMBER_OF_SOIL_LEVELS,), dtype=jnp.float64),
    )
    initial_liquid = active_rows(
        state.liquid_water_mass_column_level[:, soil_slice],
        safe_liquid,
    )
    initial_potential = active_rows(
        state.soil_water_potential_column_level[:, soil_slice],
        jnp.full((NUMBER_OF_SOIL_LEVELS,), jnp.float64(-100.0)),
    )
    initial_conductivity = active_rows(
        state.hydraulic_conductivity_column_level[:, soil_slice],
        jnp.full((NUMBER_OF_SOIL_LEVELS,), jnp.float64(1.0e-4)),
    )
    adaptive = _adaptive_moisture_columns(
        layer_count,
        layer_thickness,
        node_depth,
        ice_fraction,
        saturated_water_content,
        saturated_hydraulic_conductivity,
        clapp_hornberger_exponent,
        saturated_suction,
        infiltration_flux,
        root_water_sink,
        initial_liquid,
        initial_potential,
        initial_conductivity,
        parameters,
        work_hint=state.adaptive_substep_count_column,
    )

    candidate_liquid = adaptive.liquid_water_mass_level
    candidate_aquifer = jnp.where(
        hydrology,
        state.aquifer_water_column,
        jnp.float64(0.0),
    )
    if parameters.use_flexible_cn:
        cleanup = jax.vmap(
            _flexible_cn_liquid_cleanup,
            in_axes=(0, 0, None),
        )
        if jax.default_backend() == "gpu":
            from clm_jax.physics.soil_water_cleanup_gpu import liquid_cleanup_columns

            cleanup = liquid_cleanup_columns
        candidate_liquid, candidate_aquifer = cleanup(
            candidate_liquid,
            candidate_aquifer,
            jnp.float64(parameters.minimum_liquid_water_mass),
        )

    liquid_soil = jnp.where(
        hydrology[:, None],
        candidate_liquid,
        state.liquid_water_mass_column_level[:, soil_slice],
    )
    aquifer_water = jnp.where(
        hydrology,
        candidate_aquifer,
        state.aquifer_water_column,
    )

    liquid_water = state.liquid_water_mass_column_level.at[:, soil_slice].set(
        liquid_soil
    )
    potential = state.soil_water_potential_column_level.at[:, soil_slice].set(
        jnp.where(
            hydrology[:, None],
            adaptive.soil_water_potential_level,
            state.soil_water_potential_column_level[:, soil_slice],
        )
    )
    conductivity = state.hydraulic_conductivity_column_level.at[:, soil_slice].set(
        jnp.where(
            hydrology[:, None],
            adaptive.hydraulic_conductivity_level,
            state.hydraulic_conductivity_column_level[:, soil_slice],
        )
    )

    volumetric_water = state.volumetric_water_column_level
    if parameters.use_flexible_cn:
        liquid_volume = fortran_divide(
            candidate_liquid,
            fortran_multiply(
                layer_thickness,
                jnp.float64(LIQUID_WATER_DENSITY),
            ),
        )
        ice_volume = fortran_divide(
            active_rows(
                inputs.ice_water_mass_column_level[:, soil_slice],
                jnp.zeros((NUMBER_OF_SOIL_LEVELS,), dtype=jnp.float64),
            ),
            fortran_multiply(layer_thickness, jnp.float64(DENSITY_ICE)),
        )
        updated_volume = fortran_add(liquid_volume, ice_volume)
        volumetric_water = volumetric_water.at[:, soil_slice].set(
            jnp.where(
                hydrology[:, None],
                updated_volume,
                volumetric_water[:, soil_slice],
            )
        )

    result_state = SoilWaterMovementState(
        liquid_water_mass_column_level=liquid_water,
        volumetric_water_column_level=volumetric_water,
        soil_water_potential_column_level=potential,
        hydraulic_conductivity_column_level=conductivity,
        aquifer_water_column=aquifer_water,
        recharge_flux_column=jnp.where(
            hydrology,
            jnp.float64(0.0),
            state.recharge_flux_column,
        ),
        adaptive_substep_count_column=jnp.where(
            hydrology,
            adaptive.trial_count.astype(jnp.float64),
            state.adaptive_substep_count_column,
        ),
    )
    return SoilWaterMovementResult(
        state=result_state,
        diagnostics=SoilWaterMovementDiagnostics(
            hydrology_column=hydrology,
            adaptive_loop_completed_column=jnp.where(
                hydrology,
                adaptive.completed,
                jnp.bool_(True),
            ),
            final_maximum_flux_error_column=jnp.where(
                hydrology,
                adaptive.maximum_flux_error,
                jnp.float64(0.0),
            ),
        ),
    )
