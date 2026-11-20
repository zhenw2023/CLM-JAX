
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.physics.methane_balance_initialization import (
    NUMBER_OF_METHANE_SOIL_LEVELS,
)
from clm_jax.physics.numerics import (
    fortran_divide,
    strict_exp,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.subgrid import column_to_gridcell_mean

INUNDATION_METHOD_SURFACE_WATER = 0
INUNDATION_METHOD_WATER_TABLE = 1
INUNDATION_METHOD_TOTAL_WATER_STORAGE = 2


class MethaneDriverPreprocessInputs(NamedTuple):
    methane_column: jnp.ndarray
    column_gridcell: jnp.ndarray
    latitude_degrees_gridcell: jnp.ndarray
    atmospheric_temperature_gridcell: jnp.ndarray
    atmospheric_bottom_pressure_gridcell: jnp.ndarray
    atmospheric_oxygen_partial_pressure_gridcell: jnp.ndarray
    atmospheric_carbon_dioxide_partial_pressure_gridcell: jnp.ndarray
    atmospheric_methane_partial_pressure_gridcell: jnp.ndarray
    surface_runoff_column: jnp.ndarray
    surface_water_fraction_column: jnp.ndarray
    water_table_depth_column: jnp.ndarray
    perched_water_table_depth_column: jnp.ndarray
    soil_bottom_depth_column: jnp.ndarray
    total_water_storage_gridcell: jnp.ndarray
    water_table_scale_gridcell: jnp.ndarray
    water_table_inundation_scale_gridcell: jnp.ndarray
    runoff_inundation_scale_gridcell: jnp.ndarray
    total_water_storage_slope_gridcell: jnp.ndarray
    total_water_storage_intercept_gridcell: jnp.ndarray
    snow_depth_column: jnp.ndarray
    soil_layer_thickness_column_level: jnp.ndarray
    unsaturated_methane_concentration_column_level: jnp.ndarray


class MethaneDriverPreprocessState(NamedTuple):
    lagged_surface_runoff_column: jnp.ndarray
    inundated_fraction_column: jnp.ndarray
    presnow_inundated_fraction_column: jnp.ndarray
    lagged_inundated_fraction_column: jnp.ndarray
    saturated_methane_concentration_column_level: jnp.ndarray
    inundation_change_surface_flux_column: jnp.ndarray
    methane_first_time_column: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MethaneDriverPreprocessParameters:
    time_step_seconds: float = 1800.0
    runoff_lag_days: float = 30.0
    high_latitude_lag_factor: float = 2.0
    redox_lag_days: float = 30.0
    atmospheric_methane_mole_fraction: float = 1.7e-6
    carbon_atomic_weight: float = 12.011
    universal_gas_constant_joule_per_kelvin_kmol: float = 8314.467591
    seconds_per_day: float = 86400.0
    methane_offline: bool = True
    inundation_method: int = INUNDATION_METHOD_TOTAL_WATER_STORAGE

    def tree_flatten(self):
        return (), (
            self.time_step_seconds,
            self.runoff_lag_days,
            self.high_latitude_lag_factor,
            self.redox_lag_days,
            self.atmospheric_methane_mole_fraction,
            self.carbon_atomic_weight,
            self.universal_gas_constant_joule_per_kelvin_kmol,
            self.seconds_per_day,
            self.methane_offline,
            self.inundation_method,
        )

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del children
        return cls(*auxiliary)


DEFAULT_METHANE_DRIVER_PREPROCESS_PARAMETERS = MethaneDriverPreprocessParameters()


class MethaneDriverPreprocessDiagnostics(NamedTuple):
    atmospheric_concentration_gridcell_species: jnp.ndarray
    atmospheric_methane_partial_pressure_gridcell: jnp.ndarray
    previous_inundated_fraction_column: jnp.ndarray
    diagnosed_inundated_fraction_before_snow_hold_column: jnp.ndarray
    runoff_lag_seconds_column: jnp.ndarray
    redox_lag_seconds: jnp.ndarray
    increasing_inundation_column: jnp.ndarray
    decreasing_inundation_column: jnp.ndarray
    methane_column_count: jnp.ndarray


class MethaneDriverPreprocessResult(NamedTuple):
    state: MethaneDriverPreprocessState
    diagnostics: MethaneDriverPreprocessDiagnostics


class MethaneDriverFinalizeInputs(NamedTuple):
    methane_column: jnp.ndarray
    nonlake_column: jnp.ndarray
    lake_column: jnp.ndarray
    column_active: jnp.ndarray
    column_gridcell: jnp.ndarray
    column_weight_on_gridcell: jnp.ndarray
    inundated_fraction_column: jnp.ndarray
    soil_layer_thickness_column_level: jnp.ndarray
    saturated_methane_concentration_column_level: jnp.ndarray
    unsaturated_methane_concentration_column_level: jnp.ndarray
    saturated_surface_diffusion_column: jnp.ndarray
    unsaturated_surface_diffusion_column: jnp.ndarray
    saturated_surface_aerenchyma_column: jnp.ndarray
    unsaturated_surface_aerenchyma_column: jnp.ndarray
    saturated_surface_ebullition_column: jnp.ndarray
    unsaturated_surface_ebullition_column: jnp.ndarray
    saturated_methane_oxidation_column_level: jnp.ndarray
    unsaturated_methane_oxidation_column_level: jnp.ndarray
    saturated_methane_production_column_level: jnp.ndarray
    unsaturated_methane_production_column_level: jnp.ndarray
    inundation_change_surface_flux_column: jnp.ndarray
    beginning_methane_carbon_column: jnp.ndarray


class MethaneDriverFinalizeState(NamedTuple):
    total_methane_carbon_column: jnp.ndarray
    surface_methane_flux_column: jnp.ndarray
    methane_oxidation_gridcell: jnp.ndarray
    methane_production_gridcell: jnp.ndarray
    net_methane_carbon_correction_gridcell: jnp.ndarray
    methane_first_time_column: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MethaneDriverFinalizeParameters:
    time_step_seconds: float = 1800.0
    carbon_atomic_weight: float = 12.011
    conservation_tolerance_g_c_m2_timestep: float = 1.0e-7
    allow_lake_production: bool = False
    maximum_columns_per_gridcell: int | None = None

    def tree_flatten(self):
        return (), (
            self.time_step_seconds,
            self.carbon_atomic_weight,
            self.conservation_tolerance_g_c_m2_timestep,
            self.allow_lake_production,
            self.maximum_columns_per_gridcell,
        )

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del children
        return cls(*auxiliary)


DEFAULT_METHANE_DRIVER_FINALIZE_PARAMETERS = MethaneDriverFinalizeParameters()


class MethaneDriverFinalizeDiagnostics(NamedTuple):
    saturated_surface_flux_mol_m2_s_column: jnp.ndarray
    unsaturated_surface_flux_mol_m2_s_column: jnp.ndarray
    methane_oxidation_g_c_m2_s_column: jnp.ndarray
    methane_production_g_c_m2_s_column: jnp.ndarray
    net_methane_carbon_correction_g_c_m2_s_column: jnp.ndarray
    methane_conservation_error_g_c_m2_timestep_column: jnp.ndarray
    methane_conservation_violation_column: jnp.ndarray
    methane_column_count: jnp.ndarray


class MethaneDriverFinalizeResult(NamedTuple):
    state: MethaneDriverFinalizeState
    diagnostics: MethaneDriverFinalizeDiagnostics


def _validate_preprocess_shapes(
    inputs: MethaneDriverPreprocessInputs,
    state: MethaneDriverPreprocessState,
) -> tuple[int, int, int]:
    if jnp.ndim(inputs.methane_column) != 1:
        raise ValueError("methane_column must be a vector")
    column_count = inputs.methane_column.shape[0]
    if jnp.shape(inputs.column_gridcell) != (column_count,):
        raise ValueError("column_gridcell must contain every column")
    if jnp.ndim(inputs.latitude_degrees_gridcell) != 1:
        raise ValueError("latitude_degrees_gridcell must be a vector")
    gridcell_count = inputs.latitude_degrees_gridcell.shape[0]
    gridcell_fields = (
        "atmospheric_temperature_gridcell",
        "atmospheric_bottom_pressure_gridcell",
        "atmospheric_oxygen_partial_pressure_gridcell",
        "atmospheric_carbon_dioxide_partial_pressure_gridcell",
        "atmospheric_methane_partial_pressure_gridcell",
        "total_water_storage_gridcell",
        "water_table_scale_gridcell",
        "water_table_inundation_scale_gridcell",
        "runoff_inundation_scale_gridcell",
        "total_water_storage_slope_gridcell",
        "total_water_storage_intercept_gridcell",
    )
    for name in gridcell_fields:
        if jnp.shape(getattr(inputs, name)) != (gridcell_count,):
            raise ValueError(f"{name} must contain every gridcell")
    column_fields = (
        "surface_runoff_column",
        "surface_water_fraction_column",
        "water_table_depth_column",
        "perched_water_table_depth_column",
        "soil_bottom_depth_column",
        "snow_depth_column",
    )
    for name in column_fields:
        if jnp.shape(getattr(inputs, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    if jnp.ndim(inputs.soil_layer_thickness_column_level) != 2:
        raise ValueError("soil_layer_thickness_column_level must be a matrix")
    level_shape = jnp.shape(inputs.soil_layer_thickness_column_level)
    if level_shape[0] != column_count or level_shape[1] < NUMBER_OF_METHANE_SOIL_LEVELS:
        raise ValueError("methane preprocessing requires at least 20 soil levels")
    level_count = level_shape[1]
    if jnp.shape(inputs.unsaturated_methane_concentration_column_level) != level_shape:
        raise ValueError("unsaturated methane concentration must match soil geometry")
    state_column_fields = (
        "lagged_surface_runoff_column",
        "inundated_fraction_column",
        "presnow_inundated_fraction_column",
        "lagged_inundated_fraction_column",
        "inundation_change_surface_flux_column",
        "methane_first_time_column",
    )
    for name in state_column_fields:
        if jnp.shape(getattr(state, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    if jnp.shape(state.saturated_methane_concentration_column_level) != level_shape:
        raise ValueError("saturated methane concentration must match soil geometry")
    return column_count, gridcell_count, level_count


def _diagnose_inundated_fraction(
    inputs: MethaneDriverPreprocessInputs,
    column_gridcell: jnp.ndarray,
    lagged_surface_runoff: jnp.ndarray,
    parameters: MethaneDriverPreprocessParameters,
) -> jnp.ndarray:

    if parameters.inundation_method == INUNDATION_METHOD_SURFACE_WATER:
        diagnosed = jnp.asarray(
            inputs.surface_water_fraction_column,
            dtype=jnp.float64,
        )
    elif parameters.inundation_method == INUNDATION_METHOD_WATER_TABLE:
        water_table_scale = jnp.take(
            jnp.asarray(inputs.water_table_scale_gridcell, dtype=jnp.float64),
            column_gridcell,
        )
        base_scale = jnp.take(
            jnp.asarray(
                inputs.water_table_inundation_scale_gridcell,
                dtype=jnp.float64,
            ),
            column_gridcell,
        )
        runoff_scale = jnp.take(
            jnp.asarray(inputs.runoff_inundation_scale_gridcell, dtype=jnp.float64),
            column_gridcell,
        )
        water_table = jnp.asarray(inputs.water_table_depth_column, dtype=jnp.float64)
        perched = jnp.asarray(
            inputs.perched_water_table_depth_column,
            dtype=jnp.float64,
        )
        bottom = jnp.asarray(inputs.soil_bottom_depth_column, dtype=jnp.float64)
        use_perched = (perched < fortran_add(bottom, jnp.float64(-1.0e-5))) & (
            perched < water_table
        )
        actual_water_table = jnp.where(use_perched, perched, water_table)
        runoff_term = fortran_multiply(runoff_scale, lagged_surface_runoff)
        exponential_term = fortran_multiply(
            base_scale,
            strict_exp(-fortran_divide(actual_water_table, water_table_scale)),
        )
        diagnosed = jnp.where(
            water_table_scale > jnp.float64(0.0),
            fortran_add(exponential_term, runoff_term),
            runoff_term,
        )
    elif parameters.inundation_method == INUNDATION_METHOD_TOTAL_WATER_STORAGE:
        total_water_storage = jnp.take(
            jnp.asarray(inputs.total_water_storage_gridcell, dtype=jnp.float64),
            column_gridcell,
        )
        slope = jnp.take(
            jnp.asarray(inputs.total_water_storage_slope_gridcell, dtype=jnp.float64),
            column_gridcell,
        )
        intercept = jnp.take(
            jnp.asarray(
                inputs.total_water_storage_intercept_gridcell,
                dtype=jnp.float64,
            ),
            column_gridcell,
        )
        diagnosed = fortran_add(
            fortran_multiply(slope, total_water_storage),
            intercept,
        )
    else:
        raise ValueError(f"unknown methane inundation method {parameters.inundation_method}")
    return jnp.minimum(jnp.float64(1.0), jnp.maximum(jnp.float64(0.0), diagnosed))


def methane_driver_preprocess_active_case(
    inputs: MethaneDriverPreprocessInputs,
    state: MethaneDriverPreprocessState,
    parameters: MethaneDriverPreprocessParameters = (DEFAULT_METHANE_DRIVER_PREPROCESS_PARAMETERS),
) -> MethaneDriverPreprocessResult:

    column_count, _, level_count = _validate_preprocess_shapes(inputs, state)
    active = jnp.asarray(inputs.methane_column, dtype=jnp.bool_)
    column_gridcell = jnp.asarray(inputs.column_gridcell, dtype=jnp.int32)
    dtime = jnp.float64(parameters.time_step_seconds)
    seconds_per_day = jnp.float64(parameters.seconds_per_day)

    bottom_pressure = jnp.asarray(
        inputs.atmospheric_bottom_pressure_gridcell,
        dtype=jnp.float64,
    )
    supplied_methane_pressure = jnp.asarray(
        inputs.atmospheric_methane_partial_pressure_gridcell,
        dtype=jnp.float64,
    )
    if parameters.methane_offline:
        methane_pressure = fortran_multiply(
            jnp.float64(parameters.atmospheric_methane_mole_fraction),
            bottom_pressure,
        )
    else:
        methane_pressure = supplied_methane_pressure
    gas_constant_molar = fortran_divide(
        jnp.float64(parameters.universal_gas_constant_joule_per_kelvin_kmol),
        jnp.float64(1000.0),
    )
    atmospheric_temperature = jnp.asarray(
        inputs.atmospheric_temperature_gridcell,
        dtype=jnp.float64,
    )

    def concentration(partial_pressure: jnp.ndarray) -> jnp.ndarray:
        return fortran_divide(
            fortran_divide(partial_pressure, gas_constant_molar),
            atmospheric_temperature,
        )

    atmospheric_concentration = jnp.stack(
        (
            concentration(methane_pressure),
            concentration(
                jnp.asarray(
                    inputs.atmospheric_oxygen_partial_pressure_gridcell,
                    dtype=jnp.float64,
                )
            ),
            concentration(
                jnp.asarray(
                    inputs.atmospheric_carbon_dioxide_partial_pressure_gridcell,
                    dtype=jnp.float64,
                )
            ),
        ),
        axis=1,
    )

    previous_inundated = jnp.asarray(
        state.inundated_fraction_column,
        dtype=jnp.float64,
    )
    latitude = jnp.take(
        jnp.asarray(inputs.latitude_degrees_gridcell, dtype=jnp.float64),
        column_gridcell,
    )
    base_runoff_lag_seconds = fortran_multiply(
        jnp.float64(parameters.runoff_lag_days),
        seconds_per_day,
    )
    high_runoff_lag_seconds = fortran_multiply(
        base_runoff_lag_seconds,
        jnp.float64(parameters.high_latitude_lag_factor),
    )
    runoff_lag_seconds = jnp.where(
        latitude < jnp.float64(45.0),
        base_runoff_lag_seconds,
        high_runoff_lag_seconds,
    )
    runoff_decay = strict_exp(-fortran_divide(dtime, runoff_lag_seconds))
    one_minus_runoff_decay = fortran_add(jnp.float64(1.0), -runoff_decay)
    old_lagged_runoff = jnp.asarray(
        state.lagged_surface_runoff_column,
        dtype=jnp.float64,
    )
    updated_lagged_runoff = fortran_add(
        fortran_multiply(old_lagged_runoff, runoff_decay),
        fortran_multiply(
            jnp.asarray(inputs.surface_runoff_column, dtype=jnp.float64),
            one_minus_runoff_decay,
        ),
    )
    lagged_runoff = jnp.where(active, updated_lagged_runoff, old_lagged_runoff)

    diagnosed = _diagnose_inundated_fraction(
        inputs,
        column_gridcell,
        lagged_runoff,
        parameters,
    )
    diagnosed = jnp.where(active, diagnosed, previous_inundated)
    old_presnow = jnp.asarray(
        state.presnow_inundated_fraction_column,
        dtype=jnp.float64,
    )
    snow_free = jnp.asarray(inputs.snow_depth_column, dtype=jnp.float64) <= jnp.float64(0.0)
    snow_held_inundated = jnp.where(snow_free, diagnosed, old_presnow)
    inundated = jnp.where(active, snow_held_inundated, previous_inundated)
    presnow = jnp.where(active & snow_free, inundated, old_presnow)

    redox_lag_seconds = fortran_multiply(
        jnp.float64(parameters.redox_lag_days),
        seconds_per_day,
    )
    old_lagged_inundated = jnp.asarray(
        state.lagged_inundated_fraction_column,
        dtype=jnp.float64,
    )
    if parameters.redox_lag_days > 0.0:
        redox_decay = strict_exp(-fortran_divide(dtime, redox_lag_seconds))
        updated_lagged_inundated = fortran_add(
            fortran_multiply(old_lagged_inundated, redox_decay),
            fortran_multiply(
                inundated,
                fortran_add(jnp.float64(1.0), -redox_decay),
            ),
        )
    else:
        updated_lagged_inundated = inundated
    lagged_inundated = jnp.where(
        active,
        updated_lagged_inundated,
        old_lagged_inundated,
    )

    first_time = jnp.asarray(state.methane_first_time_column, dtype=jnp.bool_)
    increasing = active & ~first_time & (inundated > previous_inundated)
    decreasing = active & ~first_time & (inundated < previous_inundated)
    unsaturated = jnp.asarray(
        inputs.unsaturated_methane_concentration_column_level,
        dtype=jnp.float64,
    )
    old_saturated = jnp.asarray(
        state.saturated_methane_concentration_column_level,
        dtype=jnp.float64,
    )
    saturated = old_saturated
    old_change_flux = jnp.asarray(
        state.inundation_change_surface_flux_column,
        dtype=jnp.float64,
    )
    change_flux = jnp.where(active, jnp.float64(0.0), old_change_flux)
    thickness = jnp.asarray(
        inputs.soil_layer_thickness_column_level,
        dtype=jnp.float64,
    )
    inundation_delta = fortran_add(inundated, -previous_inundated)
    dryout_delta = fortran_add(previous_inundated, -inundated)
    carbon_atomic_weight = jnp.float64(parameters.carbon_atomic_weight)

    def redistribute_level(level: int, carry):
        concentration_matrix, flux = carry
        old_level = concentration_matrix[:, level]
        mixed_numerator = fortran_add(
            fortran_multiply(previous_inundated, old_level),
            fortran_multiply(inundation_delta, unsaturated[:, level]),
        )
        increased_level = fortran_divide(
            mixed_numerator,
            jnp.where(increasing, inundated, jnp.float64(1.0)),
        )
        new_level = jnp.where(increasing, increased_level, old_level)
        concentration_matrix = concentration_matrix.at[:, level].set(new_level)

        contribution = fortran_multiply(
            dryout_delta,
            fortran_add(old_level, -unsaturated[:, level]),
        )
        contribution = fortran_multiply(contribution, thickness[:, level])
        contribution = fortran_divide(contribution, dtime)
        contribution = fortran_multiply(contribution, carbon_atomic_weight)
        contribution = fortran_divide(contribution, jnp.float64(1000.0))
        flux = jnp.where(
            decreasing,
            fortran_add(flux, contribution),
            flux,
        )
        return concentration_matrix, flux

    saturated, change_flux = lax.fori_loop(
        0,
        NUMBER_OF_METHANE_SOIL_LEVELS,
        redistribute_level,
        (saturated, change_flux),
    )
    saturated = jnp.where(active[:, None], saturated, old_saturated)
    if level_count > NUMBER_OF_METHANE_SOIL_LEVELS:
        saturated = saturated.at[:, NUMBER_OF_METHANE_SOIL_LEVELS:].set(
            old_saturated[:, NUMBER_OF_METHANE_SOIL_LEVELS:]
        )

    return MethaneDriverPreprocessResult(
        state=MethaneDriverPreprocessState(
            lagged_surface_runoff_column=lagged_runoff,
            inundated_fraction_column=inundated,
            presnow_inundated_fraction_column=presnow,
            lagged_inundated_fraction_column=lagged_inundated,
            saturated_methane_concentration_column_level=saturated,
            inundation_change_surface_flux_column=change_flux,
            methane_first_time_column=first_time,
        ),
        diagnostics=MethaneDriverPreprocessDiagnostics(
            atmospheric_concentration_gridcell_species=atmospheric_concentration,
            atmospheric_methane_partial_pressure_gridcell=methane_pressure,
            previous_inundated_fraction_column=previous_inundated,
            diagnosed_inundated_fraction_before_snow_hold_column=diagnosed,
            runoff_lag_seconds_column=runoff_lag_seconds,
            redox_lag_seconds=redox_lag_seconds,
            increasing_inundation_column=increasing,
            decreasing_inundation_column=decreasing,
            methane_column_count=jnp.sum(active, dtype=jnp.int32),
        ),
    )


def _validate_finalize_shapes(
    inputs: MethaneDriverFinalizeInputs,
    state: MethaneDriverFinalizeState,
) -> tuple[int, int, int]:
    if jnp.ndim(inputs.methane_column) != 1:
        raise ValueError("methane_column must be a vector")
    column_count = inputs.methane_column.shape[0]
    column_fields = (
        "nonlake_column",
        "lake_column",
        "column_active",
        "column_gridcell",
        "column_weight_on_gridcell",
        "inundated_fraction_column",
        "saturated_surface_diffusion_column",
        "unsaturated_surface_diffusion_column",
        "saturated_surface_aerenchyma_column",
        "unsaturated_surface_aerenchyma_column",
        "saturated_surface_ebullition_column",
        "unsaturated_surface_ebullition_column",
        "inundation_change_surface_flux_column",
        "beginning_methane_carbon_column",
    )
    for name in column_fields:
        if jnp.shape(getattr(inputs, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    level_shape = jnp.shape(inputs.soil_layer_thickness_column_level)
    if (
        len(level_shape) != 2
        or level_shape[0] != column_count
        or level_shape[1] < NUMBER_OF_METHANE_SOIL_LEVELS
    ):
        raise ValueError("methane finalization requires at least 20 soil levels")
    level_count = level_shape[1]
    level_fields = (
        "saturated_methane_concentration_column_level",
        "unsaturated_methane_concentration_column_level",
        "saturated_methane_oxidation_column_level",
        "unsaturated_methane_oxidation_column_level",
        "saturated_methane_production_column_level",
        "unsaturated_methane_production_column_level",
    )
    for name in level_fields:
        if jnp.shape(getattr(inputs, name)) != level_shape:
            raise ValueError(f"{name} must match the soil-level shape")
    state_column_fields = (
        "total_methane_carbon_column",
        "surface_methane_flux_column",
        "methane_first_time_column",
    )
    for name in state_column_fields:
        if jnp.shape(getattr(state, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    if jnp.ndim(state.methane_oxidation_gridcell) != 1:
        raise ValueError("methane_oxidation_gridcell must be a vector")
    gridcell_count = state.methane_oxidation_gridcell.shape[0]
    for name in (
        "methane_production_gridcell",
        "net_methane_carbon_correction_gridcell",
    ):
        if jnp.shape(getattr(state, name)) != (gridcell_count,):
            raise ValueError(f"{name} must contain every gridcell")
    return column_count, gridcell_count, level_count


def methane_driver_finalize_active_case(
    inputs: MethaneDriverFinalizeInputs,
    state: MethaneDriverFinalizeState,
    parameters: MethaneDriverFinalizeParameters = (DEFAULT_METHANE_DRIVER_FINALIZE_PARAMETERS),
) -> MethaneDriverFinalizeResult:

    column_count, gridcell_count, level_count = _validate_finalize_shapes(inputs, state)
    if parameters.allow_lake_production:
        raise ValueError("configured methane finalization excludes lake production")

    methane_column = jnp.asarray(inputs.methane_column, dtype=jnp.bool_)
    nonlake_column = jnp.asarray(inputs.nonlake_column, dtype=jnp.bool_)
    lake_column = jnp.asarray(inputs.lake_column, dtype=jnp.bool_)
    inundated = jnp.asarray(inputs.inundated_fraction_column, dtype=jnp.float64)
    dry_fraction = fortran_add(jnp.float64(1.0), -inundated)
    carbon_atomic_weight = jnp.float64(parameters.carbon_atomic_weight)

    saturated_surface = fortran_add(
        jnp.asarray(inputs.saturated_surface_diffusion_column, dtype=jnp.float64),
        jnp.asarray(inputs.saturated_surface_aerenchyma_column, dtype=jnp.float64),
    )
    saturated_surface = fortran_add(
        saturated_surface,
        jnp.asarray(inputs.saturated_surface_ebullition_column, dtype=jnp.float64),
    )
    unsaturated_surface = fortran_add(
        jnp.asarray(inputs.unsaturated_surface_diffusion_column, dtype=jnp.float64),
        jnp.asarray(inputs.unsaturated_surface_aerenchyma_column, dtype=jnp.float64),
    )
    unsaturated_surface = fortran_add(
        unsaturated_surface,
        jnp.asarray(inputs.unsaturated_surface_ebullition_column, dtype=jnp.float64),
    )
    mixed_surface = fortran_add(
        fortran_multiply(inundated, saturated_surface),
        fortran_multiply(dry_fraction, unsaturated_surface),
    )
    mixed_surface = fortran_multiply(mixed_surface, carbon_atomic_weight)
    mixed_surface = fortran_divide(mixed_surface, jnp.float64(1000.0))
    surface_flux = jnp.where(methane_column, mixed_surface, jnp.float64(0.0))

    thickness = jnp.asarray(
        inputs.soil_layer_thickness_column_level,
        dtype=jnp.float64,
    )
    saturated_oxidation = jnp.asarray(
        inputs.saturated_methane_oxidation_column_level,
        dtype=jnp.float64,
    )
    unsaturated_oxidation = jnp.asarray(
        inputs.unsaturated_methane_oxidation_column_level,
        dtype=jnp.float64,
    )
    saturated_production = jnp.asarray(
        inputs.saturated_methane_production_column_level,
        dtype=jnp.float64,
    )
    unsaturated_production = jnp.asarray(
        inputs.unsaturated_methane_production_column_level,
        dtype=jnp.float64,
    )
    oxidation_total = jnp.zeros((column_count,), dtype=jnp.float64)
    production_total = jnp.zeros((column_count,), dtype=jnp.float64)

    def integrate_level(level: int, carry):
        oxidation, production = carry
        mixed_oxidation = fortran_add(
            fortran_multiply(inundated, saturated_oxidation[:, level]),
            fortran_multiply(dry_fraction, unsaturated_oxidation[:, level]),
        )
        oxidation_increment = fortran_multiply(mixed_oxidation, thickness[:, level])
        oxidation_increment = fortran_multiply(
            oxidation_increment,
            carbon_atomic_weight,
        )
        mixed_production = fortran_add(
            fortran_multiply(inundated, saturated_production[:, level]),
            fortran_multiply(dry_fraction, unsaturated_production[:, level]),
        )
        production_increment = fortran_multiply(mixed_production, thickness[:, level])
        production_increment = fortran_multiply(
            production_increment,
            carbon_atomic_weight,
        )
        return (
            jnp.where(
                methane_column,
                fortran_add(oxidation, oxidation_increment),
                oxidation,
            ),
            jnp.where(
                methane_column,
                fortran_add(production, production_increment),
                production,
            ),
        )

    oxidation_total, production_total = lax.fori_loop(
        0,
        NUMBER_OF_METHANE_SOIL_LEVELS,
        integrate_level,
        (oxidation_total, production_total),
    )
    net_carbon_correction = jnp.zeros((column_count,), dtype=jnp.float64)
    net_carbon_correction = jnp.where(
        methane_column,
        fortran_add(-production_total, oxidation_total),
        net_carbon_correction,
    )
    surface_flux = jnp.where(
        methane_column,
        fortran_add(
            surface_flux,
            jnp.asarray(
                inputs.inundation_change_surface_flux_column,
                dtype=jnp.float64,
            ),
        ),
        surface_flux,
    )

    old_total_methane = jnp.asarray(
        state.total_methane_carbon_column,
        dtype=jnp.float64,
    )
    inventory_mask = nonlake_column | lake_column
    total_methane = jnp.where(inventory_mask, jnp.float64(0.0), old_total_methane)
    saturated_concentration = jnp.asarray(
        inputs.saturated_methane_concentration_column_level,
        dtype=jnp.float64,
    )
    unsaturated_concentration = jnp.asarray(
        inputs.unsaturated_methane_concentration_column_level,
        dtype=jnp.float64,
    )

    def inventory_level(level: int, inventory: jnp.ndarray) -> jnp.ndarray:
        mixed_concentration = fortran_add(
            fortran_multiply(inundated, saturated_concentration[:, level]),
            fortran_multiply(dry_fraction, unsaturated_concentration[:, level]),
        )
        contribution = fortran_multiply(mixed_concentration, thickness[:, level])
        contribution = fortran_multiply(contribution, carbon_atomic_weight)
        return jnp.where(
            nonlake_column,
            fortran_add(inventory, contribution),
            inventory,
        )

    total_methane = lax.fori_loop(
        0,
        NUMBER_OF_METHANE_SOIL_LEVELS,
        inventory_level,
        total_methane,
    )

    beginning = jnp.asarray(inputs.beginning_methane_carbon_column, dtype=jnp.float64)
    tendency = fortran_add(production_total, -oxidation_total)
    tendency = fortran_add(
        tendency,
        -fortran_multiply(surface_flux, jnp.float64(1000.0)),
    )
    expected_change = fortran_multiply(
        jnp.float64(parameters.time_step_seconds),
        tendency,
    )
    conservation_error = fortran_add(total_methane, -beginning)
    conservation_error = fortran_add(conservation_error, -expected_change)
    first_time = jnp.asarray(state.methane_first_time_column, dtype=jnp.bool_)
    checked = methane_column & ~first_time
    violation = checked & (
        jnp.abs(conservation_error) > jnp.float64(parameters.conservation_tolerance_g_c_m2_timestep)
    )

    column_gridcell = jnp.asarray(inputs.column_gridcell, dtype=jnp.int32)
    column_weight = jnp.asarray(inputs.column_weight_on_gridcell, dtype=jnp.float64)
    column_active = jnp.asarray(inputs.column_active, dtype=jnp.bool_)

    def c2g(values: jnp.ndarray) -> jnp.ndarray:
        return column_to_gridcell_mean(
            values,
            column_gridcell,
            column_weight,
            column_active,
            number_of_gridcells=gridcell_count,
            maximum_columns_per_gridcell=parameters.maximum_columns_per_gridcell,
        )

    oxidation_gridcell = c2g(oxidation_total)
    production_gridcell = c2g(production_total)
    net_correction_gridcell = c2g(net_carbon_correction)

    next_first_time = jnp.zeros_like(first_time)
    return MethaneDriverFinalizeResult(
        state=MethaneDriverFinalizeState(
            total_methane_carbon_column=total_methane,
            surface_methane_flux_column=surface_flux,
            methane_oxidation_gridcell=oxidation_gridcell,
            methane_production_gridcell=production_gridcell,
            net_methane_carbon_correction_gridcell=net_correction_gridcell,
            methane_first_time_column=next_first_time,
        ),
        diagnostics=MethaneDriverFinalizeDiagnostics(
            saturated_surface_flux_mol_m2_s_column=jnp.where(
                methane_column,
                saturated_surface,
                jnp.float64(0.0),
            ),
            unsaturated_surface_flux_mol_m2_s_column=jnp.where(
                methane_column,
                unsaturated_surface,
                jnp.float64(0.0),
            ),
            methane_oxidation_g_c_m2_s_column=oxidation_total,
            methane_production_g_c_m2_s_column=production_total,
            net_methane_carbon_correction_g_c_m2_s_column=net_carbon_correction,
            methane_conservation_error_g_c_m2_timestep_column=conservation_error,
            methane_conservation_violation_column=violation,
            methane_column_count=jnp.sum(methane_column, dtype=jnp.int32),
        ),
    )


__all__ = [
    "DEFAULT_METHANE_DRIVER_FINALIZE_PARAMETERS",
    "DEFAULT_METHANE_DRIVER_PREPROCESS_PARAMETERS",
    "INUNDATION_METHOD_SURFACE_WATER",
    "INUNDATION_METHOD_TOTAL_WATER_STORAGE",
    "INUNDATION_METHOD_WATER_TABLE",
    "MethaneDriverFinalizeDiagnostics",
    "MethaneDriverFinalizeInputs",
    "MethaneDriverFinalizeParameters",
    "MethaneDriverFinalizeResult",
    "MethaneDriverFinalizeState",
    "MethaneDriverPreprocessDiagnostics",
    "MethaneDriverPreprocessInputs",
    "MethaneDriverPreprocessParameters",
    "MethaneDriverPreprocessResult",
    "MethaneDriverPreprocessState",
    "methane_driver_finalize_active_case",
    "methane_driver_preprocess_active_case",
]
