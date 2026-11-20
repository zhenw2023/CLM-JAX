
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.constants import UNIVERSAL_GAS_CONSTANT
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
from clm_jax.physics.numerics import (
    fortran_hardware_power as fortran_power,
)
from clm_jax.physics.native_power import native_power
from clm_jax.physics.ordered_reduction import ordered_segment_sum
from clm_jax.physics.pointwise import scalar_map


class MethaneWaterTableInputs(NamedTuple):
    methane_column: jnp.ndarray
    soil_porosity_column_level: jnp.ndarray
    volumetric_soil_water_column_level: jnp.ndarray
    soil_temperature_column_level: jnp.ndarray


class MethaneWaterTableState(NamedTuple):
    layer_above_water_table_column: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MethaneWaterTableParameters:
    saturation_fraction: float = 0.95
    freezing_temperature_kelvin: float = 273.15

    def tree_flatten(self):
        return (), (
            self.saturation_fraction,
            self.freezing_temperature_kelvin,
        )

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del children
        return cls(*auxiliary)


DEFAULT_METHANE_WATER_TABLE_PARAMETERS = MethaneWaterTableParameters()


class MethaneWaterTableDiagnostics(NamedTuple):
    perched_search_limit_column: jnp.ndarray
    saturated_transition_found_column: jnp.ndarray
    methane_column_count: jnp.ndarray


class MethaneWaterTableResult(NamedTuple):
    state: MethaneWaterTableState
    diagnostics: MethaneWaterTableDiagnostics


def _validate_water_table_shapes(
    inputs: MethaneWaterTableInputs,
    state: MethaneWaterTableState,
) -> int:
    if jnp.ndim(inputs.methane_column) != 1:
        raise ValueError("methane_column must be a vector")
    column_count = inputs.methane_column.shape[0]
    level_shape = jnp.shape(inputs.soil_porosity_column_level)
    if len(level_shape) != 2 or level_shape[0] != column_count:
        raise ValueError("soil porosity must be a column-by-level matrix")
    if level_shape[1] < NUMBER_OF_METHANE_SOIL_LEVELS:
        raise ValueError("methane water-table diagnosis requires 20 soil levels")
    for name in (
        "volumetric_soil_water_column_level",
        "soil_temperature_column_level",
    ):
        if jnp.shape(getattr(inputs, name)) != level_shape:
            raise ValueError("methane water-table level fields must have matching shapes")
    if jnp.shape(state.layer_above_water_table_column) != (column_count,):
        raise ValueError("water-table state must contain every column")
    return column_count


def diagnose_methane_water_table_layer(
    inputs: MethaneWaterTableInputs,
    state: MethaneWaterTableState,
    parameters: MethaneWaterTableParameters = (
        DEFAULT_METHANE_WATER_TABLE_PARAMETERS
    ),
) -> MethaneWaterTableResult:

    column_count = _validate_water_table_shapes(inputs, state)
    methane_column = jnp.asarray(inputs.methane_column, dtype=jnp.bool_)
    porosity = jnp.asarray(
        inputs.soil_porosity_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    water = jnp.asarray(
        inputs.volumetric_soil_water_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    temperature = jnp.asarray(
        inputs.soil_temperature_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    threshold = fortran_multiply(
        jnp.float64(parameters.saturation_fraction),
        porosity,
    )
    saturated = water > threshold
    frozen_saturated = (
        temperature < jnp.float64(parameters.freezing_temperature_kelvin)
    ) & saturated

    def find_perched_limit(offset: int, perch: jnp.ndarray) -> jnp.ndarray:
        level = NUMBER_OF_METHANE_SOIL_LEVELS - 1 - offset
        return jnp.where(
            methane_column & frozen_saturated[:, level],
            jnp.int32(level),
            perch,
        )

    perch = lax.fori_loop(
        0,
        NUMBER_OF_METHANE_SOIL_LEVELS,
        find_perched_limit,
        jnp.full((column_count,), NUMBER_OF_METHANE_SOIL_LEVELS, dtype=jnp.int32),
    )

    def find_transition(offset: int, carry):
        water_table_layer, found = carry
        layer_number = perch - jnp.int32(offset)
        current_index = jnp.clip(
            layer_number - jnp.int32(1),
            jnp.int32(0),
            jnp.int32(NUMBER_OF_METHANE_SOIL_LEVELS - 1),
        )
        previous_index = jnp.clip(
            layer_number - jnp.int32(2),
            jnp.int32(0),
            jnp.int32(NUMBER_OF_METHANE_SOIL_LEVELS - 1),
        )
        current_saturated = jnp.take_along_axis(
            saturated,
            current_index[:, None],
            axis=1,
        )[:, 0]
        previous_unsaturated = ~jnp.take_along_axis(
            saturated,
            previous_index[:, None],
            axis=1,
        )[:, 0]
        transition = (
            methane_column
            & ~found
            & (layer_number >= jnp.int32(2))
            & current_saturated
            & previous_unsaturated
        )
        return (
            jnp.where(transition, layer_number - jnp.int32(1), water_table_layer),
            found | transition,
        )

    water_table_layer, transition_found = lax.fori_loop(
        0,
        NUMBER_OF_METHANE_SOIL_LEVELS,
        find_transition,
        (perch, jnp.zeros((column_count,), dtype=jnp.bool_)),
    )
    top_layer_saturated = saturated[:, 0]
    water_table_layer = jnp.where(
        methane_column
        & (water_table_layer == perch)
        & top_layer_saturated,
        jnp.int32(0),
        water_table_layer,
    )
    previous = jnp.asarray(
        state.layer_above_water_table_column,
        dtype=jnp.int32,
    )
    water_table_layer = jnp.where(
        methane_column,
        water_table_layer,
        previous,
    )
    return MethaneWaterTableResult(
        state=MethaneWaterTableState(water_table_layer),
        diagnostics=MethaneWaterTableDiagnostics(
            perched_search_limit_column=jnp.where(
                methane_column,
                perch,
                previous,
            ),
            saturated_transition_found_column=(
                methane_column & transition_found
            ),
            methane_column_count=jnp.sum(methane_column, dtype=jnp.int32),
        ),
    )


class MethaneAnnualUpdateInputs(NamedTuple):
    methane_column: jnp.ndarray
    methane_patch: jnp.ndarray
    patch_column: jnp.ndarray
    soil_heterotrophic_respiration_column: jnp.ndarray
    inundated_fraction_column: jnp.ndarray
    aboveground_npp_patch: jnp.ndarray
    belowground_npp_patch: jnp.ndarray


class MethaneAnnualUpdateState(NamedTuple):
    annual_accumulator_seconds_column: jnp.ndarray
    temporary_soil_respiration_mean_column: jnp.ndarray
    annual_soil_respiration_mean_column: jnp.ndarray
    temporary_respiration_weighted_inundation_column: jnp.ndarray
    annual_respiration_weighted_inundation_column: jnp.ndarray
    temporary_aboveground_npp_mean_patch: jnp.ndarray
    annual_aboveground_npp_mean_patch: jnp.ndarray
    temporary_belowground_npp_mean_patch: jnp.ndarray
    annual_belowground_npp_mean_patch: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MethaneAnnualUpdateParameters:
    time_step_seconds: float = 1800.0
    days_per_year: int = 365

    def tree_flatten(self):
        return (), (self.time_step_seconds, self.days_per_year)

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del children
        return cls(*auxiliary)


DEFAULT_METHANE_ANNUAL_UPDATE_PARAMETERS = MethaneAnnualUpdateParameters()


class MethaneAnnualUpdateDiagnostics(NamedTuple):
    rollover_column: jnp.ndarray
    methane_column_count: jnp.ndarray
    methane_patch_count: jnp.ndarray
    seconds_per_year: jnp.ndarray


class MethaneAnnualUpdateResult(NamedTuple):
    state: MethaneAnnualUpdateState
    diagnostics: MethaneAnnualUpdateDiagnostics


def _validate_annual_update_shapes(
    inputs: MethaneAnnualUpdateInputs,
    state: MethaneAnnualUpdateState,
) -> tuple[int, int]:
    if jnp.ndim(inputs.methane_column) != 1:
        raise ValueError("methane_column must be a vector")
    if jnp.ndim(inputs.methane_patch) != 1:
        raise ValueError("methane_patch must be a vector")
    column_count = inputs.methane_column.shape[0]
    patch_count = inputs.methane_patch.shape[0]
    for name in (
        "soil_heterotrophic_respiration_column",
        "inundated_fraction_column",
    ):
        if jnp.shape(getattr(inputs, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    for name in (
        "patch_column",
        "aboveground_npp_patch",
        "belowground_npp_patch",
    ):
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    for name in (
        "annual_accumulator_seconds_column",
        "temporary_soil_respiration_mean_column",
        "annual_soil_respiration_mean_column",
        "temporary_respiration_weighted_inundation_column",
        "annual_respiration_weighted_inundation_column",
    ):
        if jnp.shape(getattr(state, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    for name in (
        "temporary_aboveground_npp_mean_patch",
        "annual_aboveground_npp_mean_patch",
        "temporary_belowground_npp_mean_patch",
        "annual_belowground_npp_mean_patch",
    ):
        if jnp.shape(getattr(state, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    return column_count, patch_count


def update_methane_annual_means(
    inputs: MethaneAnnualUpdateInputs,
    state: MethaneAnnualUpdateState,
    parameters: MethaneAnnualUpdateParameters = (
        DEFAULT_METHANE_ANNUAL_UPDATE_PARAMETERS
    ),
) -> MethaneAnnualUpdateResult:

    _validate_annual_update_shapes(inputs, state)
    methane_column = jnp.asarray(inputs.methane_column, dtype=jnp.bool_)
    methane_patch = jnp.asarray(inputs.methane_patch, dtype=jnp.bool_)
    patch_column = jnp.asarray(inputs.patch_column, dtype=jnp.int32)
    dt = jnp.float64(parameters.time_step_seconds)
    seconds_per_year = fortran_multiply(
        jnp.float64(parameters.days_per_year),
        jnp.float64(86400.0),
    )
    time_fraction = fortran_divide(dt, seconds_per_year)

    counter_old = jnp.asarray(
        state.annual_accumulator_seconds_column,
        dtype=jnp.float64,
    )
    counter_incremented = fortran_add(counter_old, dt)
    counter = jnp.where(methane_column, counter_incremented, counter_old)
    rollover = methane_column & (counter >= seconds_per_year)

    somhr = jnp.asarray(
        inputs.soil_heterotrophic_respiration_column,
        dtype=jnp.float64,
    )
    inundated = jnp.asarray(
        inputs.inundated_fraction_column,
        dtype=jnp.float64,
    )
    temporary_somhr_old = jnp.asarray(
        state.temporary_soil_respiration_mean_column,
        dtype=jnp.float64,
    )
    annual_somhr_old = jnp.asarray(
        state.annual_soil_respiration_mean_column,
        dtype=jnp.float64,
    )
    temporary_finrw_old = jnp.asarray(
        state.temporary_respiration_weighted_inundation_column,
        dtype=jnp.float64,
    )
    annual_finrw_old = jnp.asarray(
        state.annual_respiration_weighted_inundation_column,
        dtype=jnp.float64,
    )

    somhr_increment = fortran_multiply(time_fraction, somhr)
    finrw_increment = fortran_multiply(
        fortran_multiply(time_fraction, inundated),
        somhr,
    )
    accumulated_somhr = fortran_add(temporary_somhr_old, somhr_increment)
    accumulated_finrw = fortran_add(temporary_finrw_old, finrw_increment)
    active_nonrollover = methane_column & ~rollover
    annual_somhr = jnp.where(rollover, temporary_somhr_old, annual_somhr_old)
    temporary_somhr = jnp.where(
        rollover,
        jnp.float64(0.0),
        jnp.where(active_nonrollover, accumulated_somhr, temporary_somhr_old),
    )
    positive_rollover_somhr = temporary_somhr_old > jnp.float64(0.0)
    rollover_finrw = jnp.where(
        positive_rollover_somhr,
        fortran_divide(
            temporary_finrw_old,
            jnp.where(
                positive_rollover_somhr,
                temporary_somhr_old,
                jnp.float64(1.0),
            ),
        ),
        jnp.float64(0.0),
    )
    annual_finrw = jnp.where(rollover, rollover_finrw, annual_finrw_old)
    temporary_finrw = jnp.where(
        rollover,
        jnp.float64(0.0),
        jnp.where(active_nonrollover, accumulated_finrw, temporary_finrw_old),
    )

    patch_rollover = methane_patch & rollover[patch_column]
    patch_nonrollover = methane_patch & ~patch_rollover
    agnpp = jnp.asarray(inputs.aboveground_npp_patch, dtype=jnp.float64)
    bgnpp = jnp.asarray(inputs.belowground_npp_patch, dtype=jnp.float64)
    temporary_agnpp_old = jnp.asarray(
        state.temporary_aboveground_npp_mean_patch,
        dtype=jnp.float64,
    )
    annual_agnpp_old = jnp.asarray(
        state.annual_aboveground_npp_mean_patch,
        dtype=jnp.float64,
    )
    temporary_bgnpp_old = jnp.asarray(
        state.temporary_belowground_npp_mean_patch,
        dtype=jnp.float64,
    )
    annual_bgnpp_old = jnp.asarray(
        state.annual_belowground_npp_mean_patch,
        dtype=jnp.float64,
    )
    accumulated_agnpp = fortran_add(
        temporary_agnpp_old,
        fortran_multiply(time_fraction, agnpp),
    )
    accumulated_bgnpp = fortran_add(
        temporary_bgnpp_old,
        fortran_multiply(time_fraction, bgnpp),
    )
    annual_agnpp = jnp.where(
        patch_rollover,
        temporary_agnpp_old,
        annual_agnpp_old,
    )
    temporary_agnpp = jnp.where(
        patch_rollover,
        jnp.float64(0.0),
        jnp.where(patch_nonrollover, accumulated_agnpp, temporary_agnpp_old),
    )
    annual_bgnpp = jnp.where(
        patch_rollover,
        temporary_bgnpp_old,
        annual_bgnpp_old,
    )
    temporary_bgnpp = jnp.where(
        patch_rollover,
        jnp.float64(0.0),
        jnp.where(patch_nonrollover, accumulated_bgnpp, temporary_bgnpp_old),
    )

    counter = jnp.where(rollover, jnp.float64(0.0), counter)
    return MethaneAnnualUpdateResult(
        state=MethaneAnnualUpdateState(
            annual_accumulator_seconds_column=counter,
            temporary_soil_respiration_mean_column=temporary_somhr,
            annual_soil_respiration_mean_column=annual_somhr,
            temporary_respiration_weighted_inundation_column=temporary_finrw,
            annual_respiration_weighted_inundation_column=annual_finrw,
            temporary_aboveground_npp_mean_patch=temporary_agnpp,
            annual_aboveground_npp_mean_patch=annual_agnpp,
            temporary_belowground_npp_mean_patch=temporary_bgnpp,
            annual_belowground_npp_mean_patch=annual_bgnpp,
        ),
        diagnostics=MethaneAnnualUpdateDiagnostics(
            rollover_column=rollover,
            methane_column_count=jnp.sum(methane_column, dtype=jnp.int32),
            methane_patch_count=jnp.sum(methane_patch, dtype=jnp.int32),
            seconds_per_year=seconds_per_year,
        ),
    )


class MethaneProductionInputs(NamedTuple):
    methane_column: jnp.ndarray
    methane_patch: jnp.ndarray
    patch_column: jnp.ndarray
    patch_weight_on_column: jnp.ndarray
    vegetated_patch: jnp.ndarray
    root_respiration_patch: jnp.ndarray
    carbon_root_fraction_patch_level: jnp.ndarray
    water_table_layer_column: jnp.ndarray
    soil_layer_thickness_column_level: jnp.ndarray
    soil_temperature_column_level: jnp.ndarray
    soil_heterotrophic_respiration_column: jnp.ndarray
    litter_heterotrophic_respiration_column: jnp.ndarray
    vertically_resolved_heterotrophic_respiration_column_level: jnp.ndarray
    oxygen_decomposition_scalar_column_level: jnp.ndarray
    potential_nitrification_column_level: jnp.ndarray
    inundated_fraction_column: jnp.ndarray
    lagged_inundated_fraction_column: jnp.ndarray
    lagged_layer_saturation_column_level: jnp.ndarray


class MethaneProductionState(NamedTuple):
    unsaturated_methane_production_column_level: jnp.ndarray
    saturated_methane_production_column_level: jnp.ndarray
    unsaturated_oxygen_decomposition_demand_column_level: jnp.ndarray
    saturated_oxygen_decomposition_demand_column_level: jnp.ndarray
    seasonal_inundation_factor_column: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MethaneProductionParameters:
    methane_temperature_q10: float = 1.33
    methane_temperature_reference_kelvin: float = 295.0
    methane_fraction_of_mineralization: float = 0.2
    carbon_nitrogen_decomposition_scale: float = 1.0
    carbon_atomic_weight: float = 12.011

    def tree_flatten(self):
        return (), (
            self.methane_temperature_q10,
            self.methane_temperature_reference_kelvin,
            self.methane_fraction_of_mineralization,
            self.carbon_nitrogen_decomposition_scale,
            self.carbon_atomic_weight,
        )

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del children
        return cls(*auxiliary)


DEFAULT_METHANE_PRODUCTION_PARAMETERS = MethaneProductionParameters()


class MethaneProductionDiagnostics(NamedTuple):
    vertically_resolved_root_respiration_column_level: jnp.ndarray
    base_decomposition_column: jnp.ndarray
    decomposition_partition_column_level: jnp.ndarray
    methane_column_count: jnp.ndarray
    methane_patch_count: jnp.ndarray


class MethaneProductionResult(NamedTuple):
    state: MethaneProductionState
    diagnostics: MethaneProductionDiagnostics


def _validate_methane_production_shapes(
    inputs: MethaneProductionInputs,
    state: MethaneProductionState,
) -> tuple[int, int, int]:
    if jnp.ndim(inputs.methane_column) != 1:
        raise ValueError("methane_column must be a vector")
    if jnp.ndim(inputs.methane_patch) != 1:
        raise ValueError("methane_patch must be a vector")
    column_count = inputs.methane_column.shape[0]
    patch_count = inputs.methane_patch.shape[0]
    column_level_shape = jnp.shape(inputs.soil_layer_thickness_column_level)
    if (
        len(column_level_shape) != 2
        or column_level_shape[0] != column_count
        or column_level_shape[1] < NUMBER_OF_METHANE_SOIL_LEVELS
    ):
        raise ValueError("methane production requires at least 20 column levels")
    level_count = column_level_shape[1]
    for name in (
        "soil_temperature_column_level",
        "vertically_resolved_heterotrophic_respiration_column_level",
        "oxygen_decomposition_scalar_column_level",
        "potential_nitrification_column_level",
        "lagged_layer_saturation_column_level",
    ):
        if jnp.shape(getattr(inputs, name)) != column_level_shape:
            raise ValueError(f"{name} must match the column-level shape")
    for name in (
        "water_table_layer_column",
        "soil_heterotrophic_respiration_column",
        "litter_heterotrophic_respiration_column",
        "inundated_fraction_column",
        "lagged_inundated_fraction_column",
    ):
        if jnp.shape(getattr(inputs, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    for name in (
        "patch_column",
        "patch_weight_on_column",
        "vegetated_patch",
        "root_respiration_patch",
    ):
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    if jnp.shape(inputs.carbon_root_fraction_patch_level) != (
        patch_count,
        level_count,
    ):
        raise ValueError("carbon root fractions must have shape (patch, level)")
    for name in MethaneProductionState._fields[:4]:
        if jnp.shape(getattr(state, name)) != column_level_shape:
            raise ValueError(f"{name} must match the column-level shape")
    if jnp.shape(state.seasonal_inundation_factor_column) != (column_count,):
        raise ValueError("seasonal inundation factor must contain every column")
    return column_count, patch_count, level_count


def methane_production_active_case(
    inputs: MethaneProductionInputs,
    state: MethaneProductionState,
    parameters: MethaneProductionParameters = DEFAULT_METHANE_PRODUCTION_PARAMETERS,
) -> MethaneProductionResult:

    column_count, patch_count, level_count = _validate_methane_production_shapes(
        inputs,
        state,
    )
    methane_column = jnp.asarray(inputs.methane_column, dtype=jnp.bool_)
    methane_patch = jnp.asarray(inputs.methane_patch, dtype=jnp.bool_)
    patch_column = jnp.asarray(inputs.patch_column, dtype=jnp.int32)
    patch_weight = jnp.asarray(inputs.patch_weight_on_column, dtype=jnp.float64)
    vegetated_patch = jnp.asarray(inputs.vegetated_patch, dtype=jnp.bool_)
    root_respiration_patch = jnp.asarray(
        inputs.root_respiration_patch,
        dtype=jnp.float64,
    )
    carbon_root_fraction = jnp.asarray(
        inputs.carbon_root_fraction_patch_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]

    root_respiration = jnp.zeros(
        (column_count, NUMBER_OF_METHANE_SOIL_LEVELS),
        dtype=jnp.float64,
    )

    def aggregate_level(level: int, matrix: jnp.ndarray) -> jnp.ndarray:
        level_values = matrix[:, level]

        def aggregate_patch(patch_index: int, values: jnp.ndarray) -> jnp.ndarray:
            column = patch_column[patch_index]
            contribution = fortran_multiply(
                fortran_multiply(
                    root_respiration_patch[patch_index],
                    carbon_root_fraction[patch_index, level],
                ),
                patch_weight[patch_index],
            )
            include = (
                methane_patch[patch_index]
                & vegetated_patch[patch_index]
                & (patch_weight[patch_index] > jnp.float64(0.0))
            )
            previous = values[column]
            updated = fortran_add(previous, contribution)
            return values.at[column].set(jnp.where(include, updated, previous))

        level_values = lax.fori_loop(0, patch_count, aggregate_patch, level_values)
        return matrix.at[:, level].set(level_values)

    if jax.default_backend() == "cpu":
        root_respiration = lax.fori_loop(
            0,
            NUMBER_OF_METHANE_SOIL_LEVELS,
            aggregate_level,
            root_respiration,
        )
    else:
        contributions = fortran_multiply(
            fortran_multiply(root_respiration_patch[:, None], carbon_root_fraction),
            patch_weight[:, None],
        )
        include = methane_patch & vegetated_patch & (patch_weight > jnp.float64(0.0))
        root_respiration = ordered_segment_sum(
            jnp.where(include[:, None], contributions, jnp.float64(0.0)),
            patch_column,
            column_count,
        )

    thickness = jnp.asarray(
        inputs.soil_layer_thickness_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    temperature = jnp.asarray(
        inputs.soil_temperature_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    somhr = jnp.asarray(
        inputs.soil_heterotrophic_respiration_column,
        dtype=jnp.float64,
    )
    lithr = jnp.asarray(
        inputs.litter_heterotrophic_respiration_column,
        dtype=jnp.float64,
    )
    total_heterotrophic_respiration = fortran_add(somhr, lithr)
    base_decomposition = fortran_multiply(
        fortran_divide(
            total_heterotrophic_respiration,
            jnp.float64(parameters.carbon_atomic_weight),
        ),
        jnp.float64(parameters.carbon_nitrogen_decomposition_scale),
    )
    hr_vr = jnp.asarray(
        inputs.vertically_resolved_heterotrophic_respiration_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    positive_respiration = total_heterotrophic_respiration > jnp.float64(0.0)
    safe_total_respiration = jnp.where(
        positive_respiration,
        total_heterotrophic_respiration,
        jnp.float64(1.0),
    )
    partition = jnp.where(
        positive_respiration[:, None],
        fortran_divide(
            fortran_multiply(hr_vr, thickness),
            safe_total_respiration[:, None],
        ),
        jnp.float64(1.0),
    )
    base_partition_per_volume = fortran_divide(
        fortran_multiply(base_decomposition[:, None], partition),
        thickness,
    )

    oxygen_scalar = jnp.asarray(
        inputs.oxygen_decomposition_scalar_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    positive_oxygen_scalar = oxygen_scalar > jnp.float64(0.0)
    oxygen_demand = jnp.where(
        positive_oxygen_scalar,
        fortran_divide(
            base_partition_per_volume,
            jnp.where(positive_oxygen_scalar, oxygen_scalar, jnp.float64(1.0)),
        ),
        base_partition_per_volume,
    )
    root_oxygen_demand = fortran_divide(
        fortran_divide(
            root_respiration,
            jnp.float64(parameters.carbon_atomic_weight),
        ),
        thickness,
    )
    oxygen_demand = fortran_add(oxygen_demand, root_oxygen_demand)
    potential_nitrification = jnp.asarray(
        inputs.potential_nitrification_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    nitrification_oxygen_demand = fortran_divide(
        fortran_multiply(potential_nitrification, jnp.float64(2.0)),
        jnp.float64(14.0),
    )
    oxygen_demand = fortran_add(oxygen_demand, nitrification_oxygen_demand)

    exponent = fortran_divide(
        fortran_add(
            temperature,
            -jnp.float64(parameters.methane_temperature_reference_kelvin),
        ),
        jnp.float64(10.0),
    )
    q10 = jnp.float64(parameters.methane_temperature_q10)
    if jax.default_backend() == "gpu":
        temperature_factor = native_power(
            jnp.where(methane_column[:, None], q10, jnp.float64(1.0)),
            jnp.where(methane_column[:, None], exponent, jnp.float64(0.0)),
        )
    else:
        temperature_factor = fortran_power(q10, exponent)
    base_methane_fraction = fortran_multiply(
        jnp.float64(parameters.methane_fraction_of_mineralization),
        temperature_factor,
    )
    water_table = jnp.asarray(inputs.water_table_layer_column, dtype=jnp.int32)
    layer_number = jnp.arange(
        1,
        NUMBER_OF_METHANE_SOIL_LEVELS + 1,
        dtype=jnp.int32,
    )[None, :]
    below_unsaturated_water_table = layer_number > water_table[:, None]
    lagged_layer_saturation = jnp.asarray(
        inputs.lagged_layer_saturation_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    unsaturated_fraction = jnp.where(
        below_unsaturated_water_table,
        fortran_multiply(base_methane_fraction, lagged_layer_saturation),
        base_methane_fraction,
    )
    unsaturated_fraction = jnp.minimum(unsaturated_fraction, jnp.float64(0.5))
    unsaturated_production = jnp.where(
        below_unsaturated_water_table,
        fortran_divide(
            fortran_multiply(
                fortran_multiply(
                    unsaturated_fraction,
                    base_decomposition[:, None],
                ),
                partition,
            ),
            thickness,
        ),
        jnp.float64(0.0),
    )

    inundated = jnp.asarray(
        inputs.inundated_fraction_column,
        dtype=jnp.float64,
    )
    lagged_inundated = jnp.asarray(
        inputs.lagged_inundated_fraction_column,
        dtype=jnp.float64,
    )
    rising_inundation = lagged_inundated < inundated
    safe_inundated = jnp.where(rising_inundation, inundated, jnp.float64(1.0))
    saturated_fraction = jnp.where(
        rising_inundation[:, None],
        fortran_divide(
            fortran_multiply(base_methane_fraction, lagged_inundated[:, None]),
            safe_inundated[:, None],
        ),
        base_methane_fraction,
    )
    saturated_fraction = jnp.minimum(saturated_fraction, jnp.float64(0.5))
    saturated_production = fortran_divide(
        fortran_multiply(
            fortran_multiply(
                saturated_fraction,
                base_decomposition[:, None],
            ),
            partition,
        ),
        thickness,
    )

    active_level = methane_column[:, None] & (
        layer_number <= jnp.int32(NUMBER_OF_METHANE_SOIL_LEVELS)
    )

    def update_level_state(previous: jnp.ndarray, updated: jnp.ndarray) -> jnp.ndarray:
        previous = jnp.asarray(previous, dtype=jnp.float64)
        active_previous = previous[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
        result_active = jnp.where(active_level, updated, active_previous)
        return jnp.concatenate(
            (result_active, previous[:, NUMBER_OF_METHANE_SOIL_LEVELS:]),
            axis=1,
        )

    unsaturated_oxygen = update_level_state(
        state.unsaturated_oxygen_decomposition_demand_column_level,
        oxygen_demand,
    )
    saturated_oxygen = update_level_state(
        state.saturated_oxygen_decomposition_demand_column_level,
        oxygen_demand,
    )
    unsaturated_methane = update_level_state(
        state.unsaturated_methane_production_column_level,
        unsaturated_production,
    )
    saturated_methane = update_level_state(
        state.saturated_methane_production_column_level,
        saturated_production,
    )
    seasonal_inundation_factor = jnp.where(
        methane_column,
        jnp.float64(1.0),
        jnp.asarray(state.seasonal_inundation_factor_column, dtype=jnp.float64),
    )
    return MethaneProductionResult(
        state=MethaneProductionState(
            unsaturated_methane_production_column_level=unsaturated_methane,
            saturated_methane_production_column_level=saturated_methane,
            unsaturated_oxygen_decomposition_demand_column_level=unsaturated_oxygen,
            saturated_oxygen_decomposition_demand_column_level=saturated_oxygen,
            seasonal_inundation_factor_column=seasonal_inundation_factor,
        ),
        diagnostics=MethaneProductionDiagnostics(
            vertically_resolved_root_respiration_column_level=root_respiration,
            base_decomposition_column=base_decomposition,
            decomposition_partition_column_level=partition,
            methane_column_count=jnp.sum(methane_column, dtype=jnp.int32),
            methane_patch_count=jnp.sum(methane_patch, dtype=jnp.int32),
        ),
    )


class MethaneOxidationInputs(NamedTuple):
    methane_column: jnp.ndarray
    water_table_layer_column: jnp.ndarray
    volumetric_soil_water_column_level: jnp.ndarray
    soil_porosity_column_level: jnp.ndarray
    soil_matrix_potential_column_level: jnp.ndarray
    soil_temperature_column_level: jnp.ndarray
    unsaturated_methane_concentration_column_level: jnp.ndarray
    saturated_methane_concentration_column_level: jnp.ndarray
    unsaturated_oxygen_concentration_column_level: jnp.ndarray
    saturated_oxygen_concentration_column_level: jnp.ndarray


class MethaneOxidationState(NamedTuple):
    unsaturated_methane_oxidation_column_level: jnp.ndarray
    saturated_methane_oxidation_column_level: jnp.ndarray
    unsaturated_oxygen_oxidation_demand_column_level: jnp.ndarray
    saturated_oxygen_oxidation_demand_column_level: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MethaneOxidationParameters:
    saturated_maximum_oxidation_rate: float = 1.25e-5
    saturated_methane_half_saturation: float = 5.0e-3
    oxidation_temperature_q10: float = 1.9
    critical_soil_matrix_potential_mm: float = -240000.0
    oxygen_half_saturation: float = 2.0e-2
    unsaturated_methane_half_saturation: float = 5.0e-4
    unsaturated_maximum_oxidation_rate: float = 1.25e-6
    freezing_temperature_kelvin: float = 273.15
    temperature_reference_offset_kelvin: float = 12.0
    henry_reference_temperature_kelvin: float = 298.0
    methane_henry_inverse_temperature_kelvin: float = 1600.0
    oxygen_henry_inverse_temperature_kelvin: float = 1500.0
    methane_henry_constant_litre_atm_per_mol: float = 714.29
    oxygen_henry_constant_litre_atm_per_mol: float = 769.23
    gas_constant_litre_atm_per_mol_kelvin: float = 0.0821

    def tree_flatten(self):
        return (), tuple(getattr(self, name) for name in self.__dataclass_fields__)

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del children
        return cls(*auxiliary)


DEFAULT_METHANE_OXIDATION_PARAMETERS = MethaneOxidationParameters()


class MethaneOxidationDiagnostics(NamedTuple):
    soil_water_moisture_factor_column_level: jnp.ndarray
    unsaturated_relative_methane_concentration_column_level: jnp.ndarray
    saturated_relative_methane_concentration_column_level: jnp.ndarray
    methane_column_count: jnp.ndarray


class MethaneOxidationResult(NamedTuple):
    state: MethaneOxidationState
    diagnostics: MethaneOxidationDiagnostics


def _validate_methane_oxidation_shapes(
    inputs: MethaneOxidationInputs,
    state: MethaneOxidationState,
) -> tuple[int, int]:
    if jnp.ndim(inputs.methane_column) != 1:
        raise ValueError("methane_column must be a vector")
    column_count = inputs.methane_column.shape[0]
    level_shape = jnp.shape(inputs.volumetric_soil_water_column_level)
    if (
        len(level_shape) != 2
        or level_shape[0] != column_count
        or level_shape[1] < NUMBER_OF_METHANE_SOIL_LEVELS
    ):
        raise ValueError("methane oxidation requires at least 20 column levels")
    for name in MethaneOxidationInputs._fields[2:]:
        if jnp.shape(getattr(inputs, name)) != level_shape:
            raise ValueError(f"{name} must match the column-level shape")
    if jnp.shape(inputs.water_table_layer_column) != (column_count,):
        raise ValueError("water table state must contain every column")
    for name in MethaneOxidationState._fields:
        if jnp.shape(getattr(state, name)) != level_shape:
            raise ValueError(f"{name} must match the column-level shape")
    return column_count, level_shape[1]


def _elementwise_log(value: jnp.ndarray) -> jnp.ndarray:
    value = jnp.asarray(value, dtype=jnp.float64)
    shape = value.shape
    return jnp.reshape(scalar_map(jnp.log, jnp.reshape(value, (-1,))), shape)


def methane_oxidation_active_case(
    inputs: MethaneOxidationInputs,
    state: MethaneOxidationState,
    parameters: MethaneOxidationParameters = DEFAULT_METHANE_OXIDATION_PARAMETERS,
) -> MethaneOxidationResult:

    column_count, _ = _validate_methane_oxidation_shapes(inputs, state)
    methane_column = jnp.asarray(inputs.methane_column, dtype=jnp.bool_)
    water_table = jnp.asarray(inputs.water_table_layer_column, dtype=jnp.int32)
    layer_number = jnp.arange(
        1,
        NUMBER_OF_METHANE_SOIL_LEVELS + 1,
        dtype=jnp.int32,
    )[None, :]
    above_unsaturated_water_table = layer_number <= water_table[:, None]
    water = jnp.asarray(
        inputs.volumetric_soil_water_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    porosity = jnp.asarray(
        inputs.soil_porosity_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    soil_matrix_potential = jnp.asarray(
        inputs.soil_matrix_potential_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    temperature = jnp.asarray(
        inputs.soil_temperature_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    pore_volume = jnp.maximum(fortran_add(porosity, -water), jnp.float64(0.0))
    water_volume = jnp.minimum(porosity, water)
    moisture_limited = above_unsaturated_water_table & (
        soil_matrix_potential < jnp.float64(0.0)
    )
    moisture_exponent = fortran_divide(
        -soil_matrix_potential,
        jnp.float64(parameters.critical_soil_matrix_potential_mm),
    )
    moisture_factor = jnp.where(
        moisture_limited,
        strict_exp(moisture_exponent),
        jnp.float64(1.0),
    )

    def relative_concentration(
        concentration: jnp.ndarray,
        henry_inverse_temperature: float,
        henry_constant: float,
        above_water_table: jnp.ndarray,
    ) -> jnp.ndarray:
        inverse_temperature_difference = fortran_add(
            fortran_divide(jnp.float64(1.0), temperature),
            -fortran_divide(
                jnp.float64(1.0),
                jnp.float64(parameters.henry_reference_temperature_kelvin),
            ),
        )
        henry_exponent = fortran_add(
            fortran_multiply(
                -jnp.float64(henry_inverse_temperature),
                inverse_temperature_difference,
            ),
            _elementwise_log(jnp.full_like(temperature, jnp.float64(henry_constant))),
        )
        inverse_henry = strict_exp(henry_exponent)
        concentration_conversion = fortran_multiply(
            fortran_divide(temperature, inverse_henry),
            jnp.float64(parameters.gas_constant_litre_atm_per_mol_kelvin),
        )
        above_denominator = fortran_add(
            water_volume,
            fortran_divide(pore_volume, concentration_conversion),
        )
        safe_above_denominator = jnp.where(
            above_water_table,
            above_denominator,
            jnp.float64(1.0),
        )
        safe_porosity = jnp.where(
            above_water_table,
            jnp.float64(1.0),
            porosity,
        )
        return jnp.where(
            above_water_table,
            fortran_divide(concentration, safe_above_denominator),
            fortran_divide(concentration, safe_porosity),
        )

    unsaturated_methane_concentration = jnp.asarray(
        inputs.unsaturated_methane_concentration_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    saturated_methane_concentration = jnp.asarray(
        inputs.saturated_methane_concentration_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    unsaturated_oxygen_concentration = jnp.asarray(
        inputs.unsaturated_oxygen_concentration_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    saturated_oxygen_concentration = jnp.asarray(
        inputs.saturated_oxygen_concentration_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    unsaturated_relative_methane = relative_concentration(
        unsaturated_methane_concentration,
        parameters.methane_henry_inverse_temperature_kelvin,
        parameters.methane_henry_constant_litre_atm_per_mol,
        above_unsaturated_water_table,
    )
    unsaturated_relative_oxygen = relative_concentration(
        unsaturated_oxygen_concentration,
        parameters.oxygen_henry_inverse_temperature_kelvin,
        parameters.oxygen_henry_constant_litre_atm_per_mol,
        above_unsaturated_water_table,
    )
    saturated_relative_methane = fortran_divide(
        saturated_methane_concentration,
        porosity,
    )
    saturated_relative_oxygen = fortran_divide(
        saturated_oxygen_concentration,
        porosity,
    )
    high_affinity = ~above_unsaturated_water_table
    unsaturated_half_saturation = jnp.where(
        high_affinity,
        jnp.float64(parameters.saturated_methane_half_saturation),
        jnp.float64(parameters.unsaturated_methane_half_saturation),
    )
    unsaturated_maximum_rate = jnp.where(
        high_affinity,
        jnp.float64(parameters.saturated_maximum_oxidation_rate),
        jnp.float64(parameters.unsaturated_maximum_oxidation_rate),
    )
    temperature_reference = fortran_add(
        jnp.float64(parameters.freezing_temperature_kelvin),
        jnp.float64(parameters.temperature_reference_offset_kelvin),
    )
    temperature_exponent = fortran_divide(
        fortran_add(temperature, -temperature_reference),
        jnp.float64(10.0),
    )
    if jax.default_backend() == "gpu":
        temperature_factor = native_power(
            jnp.where(methane_column[:, None],
                      jnp.float64(parameters.oxidation_temperature_q10), jnp.float64(1.0)),
            jnp.where(methane_column[:, None], temperature_exponent, jnp.float64(0.0)),
        )
    else:
        temperature_factor = fortran_power(
            jnp.float64(parameters.oxidation_temperature_q10), temperature_exponent,
        )

    def oxidation_rate(
        relative_methane: jnp.ndarray,
        relative_oxygen: jnp.ndarray,
        maximum_rate: jnp.ndarray,
        methane_half_saturation: jnp.ndarray,
        water_factor: jnp.ndarray,
    ) -> jnp.ndarray:
        rate = fortran_multiply(maximum_rate, water_volume)
        rate = fortran_multiply(rate, relative_methane)
        rate = fortran_divide(
            rate,
            fortran_add(methane_half_saturation, relative_methane),
        )
        rate = fortran_multiply(rate, relative_oxygen)
        rate = fortran_divide(
            rate,
            fortran_add(
                jnp.float64(parameters.oxygen_half_saturation),
                relative_oxygen,
            ),
        )
        rate = fortran_multiply(rate, temperature_factor)
        rate = fortran_multiply(rate, water_factor)
        return jnp.where(
            temperature <= jnp.float64(parameters.freezing_temperature_kelvin),
            jnp.float64(0.0),
            rate,
        )

    unsaturated_oxidation = oxidation_rate(
        unsaturated_relative_methane,
        unsaturated_relative_oxygen,
        unsaturated_maximum_rate,
        unsaturated_half_saturation,
        moisture_factor,
    )
    saturated_oxidation = oxidation_rate(
        saturated_relative_methane,
        saturated_relative_oxygen,
        jnp.float64(parameters.saturated_maximum_oxidation_rate),
        jnp.float64(parameters.saturated_methane_half_saturation),
        jnp.float64(1.0),
    )
    unsaturated_oxygen_demand = fortran_multiply(
        unsaturated_oxidation,
        jnp.float64(2.0),
    )
    saturated_oxygen_demand = fortran_multiply(
        saturated_oxidation,
        jnp.float64(2.0),
    )
    active_level = methane_column[:, None]

    def update_level_state(previous: jnp.ndarray, updated: jnp.ndarray) -> jnp.ndarray:
        previous = jnp.asarray(previous, dtype=jnp.float64)
        active_previous = previous[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
        result_active = jnp.where(active_level, updated, active_previous)
        return jnp.concatenate(
            (result_active, previous[:, NUMBER_OF_METHANE_SOIL_LEVELS:]),
            axis=1,
        )

    return MethaneOxidationResult(
        state=MethaneOxidationState(
            unsaturated_methane_oxidation_column_level=update_level_state(
                state.unsaturated_methane_oxidation_column_level,
                unsaturated_oxidation,
            ),
            saturated_methane_oxidation_column_level=update_level_state(
                state.saturated_methane_oxidation_column_level,
                saturated_oxidation,
            ),
            unsaturated_oxygen_oxidation_demand_column_level=update_level_state(
                state.unsaturated_oxygen_oxidation_demand_column_level,
                unsaturated_oxygen_demand,
            ),
            saturated_oxygen_oxidation_demand_column_level=update_level_state(
                state.saturated_oxygen_oxidation_demand_column_level,
                saturated_oxygen_demand,
            ),
        ),
        diagnostics=MethaneOxidationDiagnostics(
            soil_water_moisture_factor_column_level=moisture_factor,
            unsaturated_relative_methane_concentration_column_level=(
                unsaturated_relative_methane
            ),
            saturated_relative_methane_concentration_column_level=(
                saturated_relative_methane
            ),
            methane_column_count=jnp.sum(methane_column, dtype=jnp.int32),
        ),
    )


class MethaneAerenchymaInputs(NamedTuple):
    methane_column: jnp.ndarray
    methane_patch: jnp.ndarray
    patch_column: jnp.ndarray
    column_gridcell: jnp.ndarray
    patch_weight_on_column: jnp.ndarray
    vegetated_patch: jnp.ndarray
    grass_or_crop_patch: jnp.ndarray
    annual_npp_patch: jnp.ndarray
    annual_average_aboveground_npp_patch: jnp.ndarray
    annual_average_belowground_npp_patch: jnp.ndarray
    effective_root_fraction_patch_level: jnp.ndarray
    root_fraction_patch_level: jnp.ndarray
    transpiration_patch: jnp.ndarray
    ground_methane_conductance_patch: jnp.ndarray
    water_table_layer_column: jnp.ndarray
    layer_depth_column_level: jnp.ndarray
    layer_thickness_column_level: jnp.ndarray
    soil_temperature_column_level: jnp.ndarray
    soil_porosity_column_level: jnp.ndarray
    volumetric_soil_water_column_level: jnp.ndarray
    atmospheric_methane_concentration_gridcell: jnp.ndarray
    atmospheric_oxygen_concentration_gridcell: jnp.ndarray
    unsaturated_methane_concentration_column_level: jnp.ndarray
    saturated_methane_concentration_column_level: jnp.ndarray
    unsaturated_oxygen_concentration_column_level: jnp.ndarray
    saturated_oxygen_concentration_column_level: jnp.ndarray
    unsaturated_methane_production_column_level: jnp.ndarray
    saturated_methane_production_column_level: jnp.ndarray


class MethaneAerenchymaState(NamedTuple):
    unsaturated_methane_loss_column_level: jnp.ndarray
    saturated_methane_loss_column_level: jnp.ndarray
    unsaturated_transpiration_loss_column_level: jnp.ndarray
    saturated_transpiration_loss_column_level: jnp.ndarray
    unsaturated_oxygen_gain_column_level: jnp.ndarray
    saturated_oxygen_gain_column_level: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MethaneAerenchymaParameters:
    time_step_seconds: float = 1800.0
    aerenchyma_area_scale: float = 1.0
    nongrass_porosity_ratio: float = 0.33
    unsaturated_porosity_ratio: float = 0.05 / 0.3
    minimum_aerenchyma_porosity: float = 0.05
    root_obliquity: float = 3.0
    freezing_temperature_kelvin: float = 273.15
    special_value: float = 1.0e36
    methane_gas_diffusivity_constant: float = 0.1875
    oxygen_gas_diffusivity_constant: float = 0.1759
    methane_henry_inverse_temperature_kelvin: float = 1600.0
    oxygen_henry_inverse_temperature_kelvin: float = 1500.0
    methane_henry_constant_litre_atm_per_mol: float = 714.29
    oxygen_henry_constant_litre_atm_per_mol: float = 769.23
    henry_reference_temperature_kelvin: float = 298.0
    gas_constant_litre_atm_per_mol_kelvin: float = 0.0821
    pi: float = 3.141592653589793
    small_number: float = 1.0e-12

    def tree_flatten(self):
        return (), tuple(getattr(self, name) for name in self.__dataclass_fields__)

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del children
        return cls(*auxiliary)


DEFAULT_METHANE_AERENCHYMA_PARAMETERS = MethaneAerenchymaParameters()


class MethaneAerenchymaDiagnostics(NamedTuple):
    unsaturated_tiller_area_patch: jnp.ndarray
    saturated_tiller_area_patch: jnp.ndarray
    npp_root_fraction_patch: jnp.ndarray
    methane_column_count: jnp.ndarray
    methane_patch_count: jnp.ndarray


class MethaneAerenchymaResult(NamedTuple):
    state: MethaneAerenchymaState
    diagnostics: MethaneAerenchymaDiagnostics


def _validate_methane_aerenchyma_shapes(
    inputs: MethaneAerenchymaInputs,
    state: MethaneAerenchymaState,
) -> tuple[int, int, int]:
    if jnp.ndim(inputs.methane_column) != 1:
        raise ValueError("methane_column must be a vector")
    if jnp.ndim(inputs.methane_patch) != 1:
        raise ValueError("methane_patch must be a vector")
    column_count = inputs.methane_column.shape[0]
    patch_count = inputs.methane_patch.shape[0]
    level_shape = jnp.shape(inputs.layer_depth_column_level)
    if (
        len(level_shape) != 2
        or level_shape[0] != column_count
        or level_shape[1] < NUMBER_OF_METHANE_SOIL_LEVELS
    ):
        raise ValueError("methane aerenchyma requires at least 20 column levels")
    level_count = level_shape[1]
    for name in (
        "layer_thickness_column_level",
        "soil_temperature_column_level",
        "soil_porosity_column_level",
        "volumetric_soil_water_column_level",
        "unsaturated_methane_concentration_column_level",
        "saturated_methane_concentration_column_level",
        "unsaturated_oxygen_concentration_column_level",
        "saturated_oxygen_concentration_column_level",
        "unsaturated_methane_production_column_level",
        "saturated_methane_production_column_level",
    ):
        if jnp.shape(getattr(inputs, name)) != level_shape:
            raise ValueError(f"{name} must match the column-level shape")
    if jnp.shape(inputs.water_table_layer_column) != (column_count,):
        raise ValueError("water table state must contain every column")
    if jnp.shape(inputs.column_gridcell) != (column_count,):
        raise ValueError("column_gridcell must contain every column")
    for name in (
        "patch_column",
        "patch_weight_on_column",
        "vegetated_patch",
        "grass_or_crop_patch",
        "annual_npp_patch",
        "annual_average_aboveground_npp_patch",
        "annual_average_belowground_npp_patch",
        "transpiration_patch",
        "ground_methane_conductance_patch",
    ):
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    for name in (
        "effective_root_fraction_patch_level",
        "root_fraction_patch_level",
    ):
        if jnp.shape(getattr(inputs, name)) != (patch_count, level_count):
            raise ValueError(f"{name} must have shape (patch, level)")
    gridcell_count = inputs.atmospheric_methane_concentration_gridcell.shape[0]
    if jnp.shape(inputs.atmospheric_methane_concentration_gridcell) != (
        gridcell_count,
    ) or jnp.shape(inputs.atmospheric_oxygen_concentration_gridcell) != (
        gridcell_count,
    ):
        raise ValueError("atmospheric concentrations must share a gridcell vector")
    for name in MethaneAerenchymaState._fields:
        if jnp.shape(getattr(state, name)) != level_shape:
            raise ValueError(f"{name} must match the column-level shape")
    return column_count, patch_count, level_count


def methane_aerenchyma_active_case(
    inputs: MethaneAerenchymaInputs,
    state: MethaneAerenchymaState,
    parameters: MethaneAerenchymaParameters = DEFAULT_METHANE_AERENCHYMA_PARAMETERS,
) -> MethaneAerenchymaResult:

    column_count, patch_count, _ = _validate_methane_aerenchyma_shapes(
        inputs,
        state,
    )
    methane_column = jnp.asarray(inputs.methane_column, dtype=jnp.bool_)
    methane_patch = jnp.asarray(inputs.methane_patch, dtype=jnp.bool_)
    patch_column = jnp.asarray(inputs.patch_column, dtype=jnp.int32)
    column_gridcell = jnp.asarray(inputs.column_gridcell, dtype=jnp.int32)
    patch_weight = jnp.asarray(inputs.patch_weight_on_column, dtype=jnp.float64)
    vegetated = jnp.asarray(inputs.vegetated_patch, dtype=jnp.bool_)
    grass_or_crop = jnp.asarray(inputs.grass_or_crop_patch, dtype=jnp.bool_)
    annual_npp = jnp.maximum(
        jnp.asarray(inputs.annual_npp_patch, dtype=jnp.float64),
        jnp.float64(0.0),
    )
    annual_aboveground_npp = jnp.asarray(
        inputs.annual_average_aboveground_npp_patch,
        dtype=jnp.float64,
    )
    annual_belowground_npp = jnp.asarray(
        inputs.annual_average_belowground_npp_patch,
        dtype=jnp.float64,
    )
    valid_npp_means = (
        (annual_aboveground_npp != jnp.float64(parameters.special_value))
        & (annual_belowground_npp != jnp.float64(parameters.special_value))
        & (annual_aboveground_npp > jnp.float64(0.0))
        & (annual_belowground_npp > jnp.float64(0.0))
    )
    npp_sum = fortran_add(annual_aboveground_npp, annual_belowground_npp)
    npp_ratio = jnp.where(
        valid_npp_means,
        fortran_divide(
            annual_belowground_npp,
            jnp.where(valid_npp_means, npp_sum, jnp.float64(1.0)),
        ),
        jnp.float64(0.5),
    )
    tiller_mass = fortran_multiply(annual_npp, npp_ratio)
    tiller_mass = fortran_multiply(tiller_mass, jnp.float64(4.0))
    tiller_count = fortran_divide(tiller_mass, jnp.float64(0.22))
    base_tiller_porosity = jnp.where(
        grass_or_crop,
        jnp.float64(0.3),
        fortran_multiply(
            jnp.float64(0.3),
            jnp.float64(parameters.nongrass_porosity_ratio),
        ),
    )

    def tiller_area(saturated: bool) -> jnp.ndarray:
        porosity = base_tiller_porosity
        if not saturated:
            porosity = fortran_multiply(
                porosity,
                jnp.float64(parameters.unsaturated_porosity_ratio),
            )
        porosity = jnp.maximum(
            porosity,
            jnp.float64(parameters.minimum_aerenchyma_porosity),
        )
        area = fortran_multiply(
            jnp.float64(parameters.aerenchyma_area_scale),
            tiller_count,
        )
        area = fortran_multiply(area, porosity)
        area = fortran_multiply(area, jnp.float64(parameters.pi))
        tiller_radius_squared = fortran_power(
            jnp.float64(2.9e-3),
            jnp.float64(2.0),
        )
        return fortran_multiply(area, tiller_radius_squared)

    unsaturated_tiller_area = tiller_area(False)
    saturated_tiller_area = tiller_area(True)
    effective_root_fraction = jnp.asarray(
        inputs.effective_root_fraction_patch_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    root_fraction = jnp.asarray(
        inputs.root_fraction_patch_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    transpiration = jnp.asarray(inputs.transpiration_patch, dtype=jnp.float64)
    ground_conductance = jnp.asarray(
        inputs.ground_methane_conductance_patch,
        dtype=jnp.float64,
    )
    water_table = jnp.asarray(inputs.water_table_layer_column, dtype=jnp.int32)
    depth = jnp.asarray(
        inputs.layer_depth_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    thickness = jnp.asarray(
        inputs.layer_thickness_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    temperature = jnp.asarray(
        inputs.soil_temperature_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    porosity = jnp.asarray(
        inputs.soil_porosity_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    water = jnp.asarray(
        inputs.volumetric_soil_water_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    atmospheric_methane = jnp.asarray(
        inputs.atmospheric_methane_concentration_gridcell,
        dtype=jnp.float64,
    )
    atmospheric_oxygen = jnp.asarray(
        inputs.atmospheric_oxygen_concentration_gridcell,
        dtype=jnp.float64,
    )
    methane_diffusivity = fortran_multiply(
        jnp.float64(parameters.methane_gas_diffusivity_constant),
        jnp.float64(1.0e-4),
    )
    oxygen_diffusivity = fortran_multiply(
        methane_diffusivity,
        jnp.float64(parameters.oxygen_gas_diffusivity_constant),
    )
    oxygen_diffusivity = fortran_divide(
        oxygen_diffusivity,
        jnp.float64(parameters.methane_gas_diffusivity_constant),
    )

    def henry_conversion(
        layer_temperature: jnp.ndarray,
        inverse_temperature: float,
        standard_constant: float,
    ) -> jnp.ndarray:
        inverse_difference = fortran_add(
            fortran_divide(jnp.float64(1.0), layer_temperature),
            -fortran_divide(
                jnp.float64(1.0),
                jnp.float64(parameters.henry_reference_temperature_kelvin),
            ),
        )
        exponent = fortran_multiply(
            -jnp.float64(inverse_temperature),
            inverse_difference,
        )
        exponent = fortran_add(
            exponent,
            jnp.log(jnp.float64(standard_constant)),
        )
        inverse_henry = strict_exp(exponent)
        conversion = fortran_divide(layer_temperature, inverse_henry)
        return fortran_multiply(
            conversion,
            jnp.float64(parameters.gas_constant_litre_atm_per_mol_kelvin),
        )

    column_henry = None
    if jax.default_backend() == "gpu":
        column_henry = lax.optimization_barrier((
            henry_conversion(
                temperature,
                parameters.methane_henry_inverse_temperature_kelvin,
                parameters.methane_henry_constant_litre_atm_per_mol,
            ),
            henry_conversion(
                temperature,
                parameters.oxygen_henry_inverse_temperature_kelvin,
                parameters.oxygen_henry_constant_litre_atm_per_mol,
            ),
        ))

    def transport_branch(
        saturated: bool,
        methane_concentration: jnp.ndarray,
        oxygen_concentration: jnp.ndarray,
        methane_production: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        methane_concentration = jnp.asarray(
            methane_concentration,
            dtype=jnp.float64,
        )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
        oxygen_concentration = jnp.asarray(
            oxygen_concentration,
            dtype=jnp.float64,
        )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
        methane_production = jnp.asarray(
            methane_production,
            dtype=jnp.float64,
        )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
        area = saturated_tiller_area if saturated else unsaturated_tiller_area
        methane_loss = jnp.zeros(
            (column_count, NUMBER_OF_METHANE_SOIL_LEVELS),
            dtype=jnp.float64,
        )
        transpiration_loss = jnp.zeros_like(methane_loss)
        oxygen_gain = jnp.zeros_like(methane_loss)

        level_number = jnp.arange(
            1,
            NUMBER_OF_METHANE_SOIL_LEVELS + 1,
            dtype=jnp.int32,
        )

        def patch_contributions(patch_index):
            column = patch_column[patch_index]
            gridcell = column_gridcell[column]
            layer_temperature = temperature[column]
            layer_porosity = porosity[column]
            layer_water = water[column]
            layer_thickness = thickness[column]
            layer_depth = depth[column]
            methane = methane_concentration[column]
            oxygen = oxygen_concentration[column]
            water_minimum = jnp.minimum(layer_porosity, layer_water)
            methane_henry = (
                column_henry[0][column] if column_henry is not None else henry_conversion(
                    layer_temperature,
                    parameters.methane_henry_inverse_temperature_kelvin,
                    parameters.methane_henry_constant_litre_atm_per_mol,
                )
            )
            gas_volume = fortran_add(layer_porosity, -water_minimum)
            water_denominator = fortran_divide(gas_volume, methane_henry)
            water_denominator = fortran_add(water_denominator, water_minimum)
            dissolved_methane = fortran_divide(methane, water_denominator)
            patch_transpiration_loss = fortran_multiply(
                dissolved_methane,
                effective_root_fraction[patch_index],
            )
            patch_transpiration_loss = fortran_multiply(
                patch_transpiration_loss,
                transpiration[patch_index],
            )
            patch_transpiration_loss = fortran_divide(
                patch_transpiration_loss,
                layer_thickness,
            )
            patch_transpiration_loss = fortran_divide(
                patch_transpiration_loss,
                jnp.float64(1000.0),
            )
            patch_transpiration_loss = jnp.maximum(
                patch_transpiration_loss,
                jnp.float64(0.0),
            )
            patch_transpiration_loss = jnp.where(
                vegetated[patch_index],
                patch_transpiration_loss,
                jnp.float64(0.0),
            )

            below_water_table = jnp.bool_(saturated) | (
                level_number > water_table[column]
            )
            transport_active = (
                vegetated[patch_index]
                & below_water_table
                & (
                    layer_temperature
                    > jnp.float64(parameters.freezing_temperature_kelvin)
                )
            )
            conductance = fortran_multiply(
                area[patch_index],
                root_fraction[patch_index],
            )
            conductance = fortran_multiply(conductance, methane_diffusivity)
            conductance = fortran_divide(
                conductance,
                fortran_multiply(
                    layer_depth,
                    jnp.float64(parameters.root_obliquity),
                ),
            )
            resistance = fortran_divide(
                jnp.float64(1.0),
                fortran_add(conductance, jnp.float64(parameters.small_number)),
            )
            resistance = fortran_add(
                resistance,
                fortran_divide(
                    jnp.float64(1.0),
                    fortran_add(
                        ground_conductance[patch_index],
                        jnp.float64(parameters.small_number),
                    ),
                ),
            )
            conductance = fortran_divide(jnp.float64(1.0), resistance)
            methane_gradient = fortran_divide(methane, layer_porosity)
            methane_gradient = fortran_divide(methane_gradient, methane_henry)
            methane_gradient = fortran_add(
                methane_gradient,
                -atmospheric_methane[gridcell],
            )
            aerenchyma_loss = fortran_multiply(conductance, methane_gradient)
            aerenchyma_loss = fortran_divide(aerenchyma_loss, layer_thickness)
            aerenchyma_loss = jnp.maximum(aerenchyma_loss, jnp.float64(0.0))
            aerenchyma_loss = jnp.where(
                transport_active,
                aerenchyma_loss,
                jnp.float64(0.0),
            )

            oxygen_henry = (
                column_henry[1][column] if column_henry is not None else henry_conversion(
                    layer_temperature,
                    parameters.oxygen_henry_inverse_temperature_kelvin,
                    parameters.oxygen_henry_constant_litre_atm_per_mol,
                )
            )
            oxygen_conductance = fortran_multiply(
                area[patch_index],
                root_fraction[patch_index],
            )
            oxygen_conductance = fortran_multiply(
                oxygen_conductance,
                oxygen_diffusivity,
            )
            oxygen_conductance = fortran_divide(
                oxygen_conductance,
                fortran_multiply(
                    layer_depth,
                    jnp.float64(parameters.root_obliquity),
                ),
            )
            oxygen_resistance = fortran_divide(
                jnp.float64(1.0),
                fortran_add(
                    oxygen_conductance,
                    jnp.float64(parameters.small_number),
                ),
            )
            oxygen_resistance = fortran_add(
                oxygen_resistance,
                fortran_divide(
                    jnp.float64(1.0),
                    fortran_add(
                        ground_conductance[patch_index],
                        jnp.float64(parameters.small_number),
                    ),
                ),
            )
            oxygen_conductance = fortran_divide(
                jnp.float64(1.0),
                oxygen_resistance,
            )
            oxygen_gradient = fortran_divide(oxygen, layer_porosity)
            oxygen_gradient = fortran_divide(oxygen_gradient, oxygen_henry)
            oxygen_gradient = fortran_add(
                oxygen_gradient,
                -atmospheric_oxygen[gridcell],
            )
            aerenchyma_oxygen = fortran_multiply(
                -oxygen_conductance,
                oxygen_gradient,
            )
            aerenchyma_oxygen = fortran_divide(
                aerenchyma_oxygen,
                layer_thickness,
            )
            aerenchyma_oxygen = jnp.maximum(
                aerenchyma_oxygen,
                jnp.float64(0.0),
            )
            aerenchyma_oxygen = jnp.where(
                transport_active,
                aerenchyma_oxygen,
                jnp.float64(0.0),
            )

            combined_loss = fortran_add(
                aerenchyma_loss,
                patch_transpiration_loss,
            )
            available_methane = fortran_divide(
                methane,
                jnp.float64(parameters.time_step_seconds),
            )
            available_methane = fortran_add(
                available_methane,
                methane_production[column],
            )
            transported = jnp.minimum(combined_loss, available_methane)
            transpired = jnp.minimum(patch_transpiration_loss, transported)
            return (
                fortran_multiply(transported, patch_weight[patch_index]),
                fortran_multiply(transpired, patch_weight[patch_index]),
                fortran_multiply(aerenchyma_oxygen, patch_weight[patch_index]),
            )

        def aggregate_patch(patch_index: int, carry):
            methane_values, transpiration_values, oxygen_values = carry
            column = patch_column[patch_index]
            methane_contribution, transpiration_contribution, oxygen_contribution = (
                patch_contributions(patch_index)
            )
            include = methane_patch[patch_index]
            methane_previous = methane_values[column]
            methane_updated = fortran_add(
                methane_previous,
                methane_contribution,
            )
            transpiration_previous = transpiration_values[column]
            transpiration_updated = fortran_add(
                transpiration_previous,
                transpiration_contribution,
            )
            oxygen_previous = oxygen_values[column]
            oxygen_updated = fortran_add(
                oxygen_previous,
                oxygen_contribution,
            )
            methane_values = methane_values.at[column].set(
                jnp.where(include, methane_updated, methane_previous)
            )
            transpiration_values = transpiration_values.at[column].set(
                jnp.where(include, transpiration_updated, transpiration_previous)
            )
            oxygen_values = oxygen_values.at[column].set(
                jnp.where(include, oxygen_updated, oxygen_previous)
            )
            return methane_values, transpiration_values, oxygen_values

        if jax.default_backend() != "cpu":
            contributions = jax.vmap(patch_contributions)(
                jnp.arange(patch_count, dtype=jnp.int32)
            )
            return tuple(
                ordered_segment_sum(
                    jnp.where(methane_patch[:, None], values, jnp.float64(0.0)),
                    patch_column,
                    column_count,
                )
                for values in contributions
            )

        return lax.fori_loop(
            0,
            patch_count,
            aggregate_patch,
            (methane_loss, transpiration_loss, oxygen_gain),
        )

    unsaturated_outputs = transport_branch(
        False,
        inputs.unsaturated_methane_concentration_column_level,
        inputs.unsaturated_oxygen_concentration_column_level,
        inputs.unsaturated_methane_production_column_level,
    )
    saturated_outputs = transport_branch(
        True,
        inputs.saturated_methane_concentration_column_level,
        inputs.saturated_oxygen_concentration_column_level,
        inputs.saturated_methane_production_column_level,
    )

    def update_level_state(previous: jnp.ndarray, updated: jnp.ndarray) -> jnp.ndarray:
        previous = jnp.asarray(previous, dtype=jnp.float64)
        result_active = jnp.where(
            methane_column[:, None],
            updated,
            previous[:, :NUMBER_OF_METHANE_SOIL_LEVELS],
        )
        return jnp.concatenate(
            (result_active, previous[:, NUMBER_OF_METHANE_SOIL_LEVELS:]),
            axis=1,
        )

    return MethaneAerenchymaResult(
        state=MethaneAerenchymaState(
            unsaturated_methane_loss_column_level=update_level_state(
                state.unsaturated_methane_loss_column_level,
                unsaturated_outputs[0],
            ),
            saturated_methane_loss_column_level=update_level_state(
                state.saturated_methane_loss_column_level,
                saturated_outputs[0],
            ),
            unsaturated_transpiration_loss_column_level=update_level_state(
                state.unsaturated_transpiration_loss_column_level,
                unsaturated_outputs[1],
            ),
            saturated_transpiration_loss_column_level=update_level_state(
                state.saturated_transpiration_loss_column_level,
                saturated_outputs[1],
            ),
            unsaturated_oxygen_gain_column_level=update_level_state(
                state.unsaturated_oxygen_gain_column_level,
                unsaturated_outputs[2],
            ),
            saturated_oxygen_gain_column_level=update_level_state(
                state.saturated_oxygen_gain_column_level,
                saturated_outputs[2],
            ),
        ),
        diagnostics=MethaneAerenchymaDiagnostics(
            unsaturated_tiller_area_patch=unsaturated_tiller_area,
            saturated_tiller_area_patch=saturated_tiller_area,
            npp_root_fraction_patch=npp_ratio,
            methane_column_count=jnp.sum(methane_column, dtype=jnp.int32),
            methane_patch_count=jnp.sum(methane_patch, dtype=jnp.int32),
        ),
    )


class MethaneEbullitionInputs(NamedTuple):
    methane_column: jnp.ndarray
    water_table_layer_column: jnp.ndarray
    layer_depth_column_level: jnp.ndarray
    interface_depth_column_interface: jnp.ndarray
    soil_temperature_column_level: jnp.ndarray
    soil_porosity_column_level: jnp.ndarray
    atmospheric_pressure_column: jnp.ndarray
    surface_water_depth_mm_column: jnp.ndarray
    surface_water_fraction_column: jnp.ndarray
    unsaturated_methane_concentration_column_level: jnp.ndarray
    saturated_methane_concentration_column_level: jnp.ndarray


class MethaneEbullitionState(NamedTuple):
    unsaturated_ebullition_column_level: jnp.ndarray
    saturated_ebullition_column_level: jnp.ndarray
    unsaturated_ebullition_total_column: jnp.ndarray
    saturated_ebullition_total_column: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MethaneEbullitionParameters:
    time_step_seconds: float = 1800.0
    maximum_bubble_saturation_ratio: float = 0.15
    bubble_methane_fraction: float = 0.57
    freezing_temperature_kelvin: float = 273.15
    methane_henry_inverse_temperature_kelvin: float = 1600.0
    methane_henry_constant_litre_atm_per_mol: float = 714.29
    henry_reference_temperature_kelvin: float = 298.0
    gas_constant_litre_atm_per_mol_kelvin: float = 0.0821
    molar_gas_constant_joule_per_mol_kelvin: float = float(UNIVERSAL_GAS_CONSTANT / 1000.0)
    water_density_kg_per_m3: float = 1000.0
    gravity_m_per_s2: float = 9.80616

    def tree_flatten(self):
        return (), tuple(getattr(self, name) for name in self.__dataclass_fields__)

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del children
        return cls(*auxiliary)


DEFAULT_METHANE_EBULLITION_PARAMETERS = MethaneEbullitionParameters()


class MethaneEbullitionDiagnostics(NamedTuple):
    unsaturated_volumetric_gas_content_column_level: jnp.ndarray
    saturated_volumetric_gas_content_column_level: jnp.ndarray
    unsaturated_pressure_column_level: jnp.ndarray
    saturated_pressure_column_level: jnp.ndarray
    methane_column_count: jnp.ndarray


class MethaneEbullitionResult(NamedTuple):
    state: MethaneEbullitionState
    diagnostics: MethaneEbullitionDiagnostics


def _validate_methane_ebullition_shapes(
    inputs: MethaneEbullitionInputs,
    state: MethaneEbullitionState,
) -> tuple[int, int]:
    if jnp.ndim(inputs.methane_column) != 1:
        raise ValueError("methane_column must be a vector")
    column_count = inputs.methane_column.shape[0]
    level_shape = jnp.shape(inputs.layer_depth_column_level)
    if (
        len(level_shape) != 2
        or level_shape[0] != column_count
        or level_shape[1] < NUMBER_OF_METHANE_SOIL_LEVELS
    ):
        raise ValueError("methane ebullition requires at least 20 column levels")
    for name in (
        "soil_temperature_column_level",
        "soil_porosity_column_level",
        "unsaturated_methane_concentration_column_level",
        "saturated_methane_concentration_column_level",
    ):
        if jnp.shape(getattr(inputs, name)) != level_shape:
            raise ValueError(f"{name} must match the column-level shape")
    interface_shape = jnp.shape(inputs.interface_depth_column_interface)
    if (
        len(interface_shape) != 2
        or interface_shape[0] != column_count
        or interface_shape[1] < NUMBER_OF_METHANE_SOIL_LEVELS + 1
    ):
        raise ValueError("interface depths must cover native indices 0 through 20")
    for name in (
        "water_table_layer_column",
        "atmospheric_pressure_column",
        "surface_water_depth_mm_column",
        "surface_water_fraction_column",
    ):
        if jnp.shape(getattr(inputs, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    for name in MethaneEbullitionState._fields[:2]:
        if jnp.shape(getattr(state, name)) != level_shape:
            raise ValueError(f"{name} must match the column-level shape")
    for name in MethaneEbullitionState._fields[2:]:
        if jnp.shape(getattr(state, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    return column_count, level_shape[1]


def methane_ebullition_active_case(
    inputs: MethaneEbullitionInputs,
    state: MethaneEbullitionState,
    parameters: MethaneEbullitionParameters = DEFAULT_METHANE_EBULLITION_PARAMETERS,
) -> MethaneEbullitionResult:

    _validate_methane_ebullition_shapes(inputs, state)
    methane_column = jnp.asarray(inputs.methane_column, dtype=jnp.bool_)
    water_table = jnp.asarray(inputs.water_table_layer_column, dtype=jnp.int32)
    depth = jnp.asarray(
        inputs.layer_depth_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    interface_depth = jnp.asarray(
        inputs.interface_depth_column_interface,
        dtype=jnp.float64,
    )
    temperature = jnp.asarray(
        inputs.soil_temperature_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    porosity = jnp.asarray(
        inputs.soil_porosity_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    atmospheric_pressure = jnp.asarray(
        inputs.atmospheric_pressure_column,
        dtype=jnp.float64,
    )
    surface_water = jnp.asarray(
        inputs.surface_water_depth_mm_column,
        dtype=jnp.float64,
    )
    surface_water_fraction = jnp.asarray(
        inputs.surface_water_fraction_column,
        dtype=jnp.float64,
    )

    inverse_temperature_difference = fortran_add(
        fortran_divide(jnp.float64(1.0), temperature),
        -fortran_divide(
            jnp.float64(1.0),
            jnp.float64(parameters.henry_reference_temperature_kelvin),
        ),
    )
    henry_exponent = fortran_multiply(
        -jnp.float64(parameters.methane_henry_inverse_temperature_kelvin),
        inverse_temperature_difference,
    )
    henry_exponent = fortran_add(
        henry_exponent,
        jnp.log(
            jnp.float64(parameters.methane_henry_constant_litre_atm_per_mol)
        ),
    )
    inverse_henry = strict_exp(henry_exponent)
    henry = fortran_divide(jnp.float64(1.0), inverse_henry)
    henry_conversion = fortran_multiply(temperature, henry)
    henry_conversion = fortran_multiply(
        henry_conversion,
        jnp.float64(parameters.gas_constant_litre_atm_per_mol_kelvin),
    )
    layer_number = jnp.arange(
        1,
        NUMBER_OF_METHANE_SOIL_LEVELS + 1,
        dtype=jnp.int32,
    )[None, :]

    def branch(
        saturated: bool,
        concentration: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        concentration = jnp.asarray(concentration, dtype=jnp.float64)[
            :, :NUMBER_OF_METHANE_SOIL_LEVELS
        ]
        branch_water_table = (
            jnp.zeros_like(water_table) if saturated else water_table
        )
        water_table_interface = jnp.take_along_axis(
            interface_depth,
            branch_water_table[:, None],
            axis=1,
        )[:, 0]
        hydrostatic_head = fortran_add(
            depth,
            -water_table_interface[:, None],
        )
        hydrostatic_pressure = fortran_multiply(
            jnp.float64(parameters.water_density_kg_per_m3),
            jnp.float64(parameters.gravity_m_per_s2),
        )
        hydrostatic_pressure = fortran_multiply(
            hydrostatic_pressure,
            hydrostatic_head,
        )
        pressure = fortran_add(
            atmospheric_pressure[:, None],
            hydrostatic_pressure,
        )
        ponded = saturated & (surface_water_fraction > jnp.float64(0.0))
        ponding_pressure = fortran_multiply(
            jnp.float64(parameters.water_density_kg_per_m3),
            jnp.float64(parameters.gravity_m_per_s2),
        )
        ponding_pressure = fortran_multiply(
            ponding_pressure,
            surface_water,
        )
        ponding_pressure = fortran_divide(
            ponding_pressure,
            jnp.float64(1000.0),
        )
        ponding_pressure = fortran_divide(
            ponding_pressure,
            jnp.where(
                ponded,
                surface_water_fraction,
                jnp.float64(1.0),
            ),
        )
        pressure = jnp.where(
            ponded[:, None],
            fortran_add(pressure, ponding_pressure[:, None]),
            pressure,
        )
        gas_content = fortran_divide(concentration, porosity)
        gas_content = fortran_divide(gas_content, henry_conversion)
        gas_content = fortran_multiply(
            gas_content,
            jnp.float64(parameters.molar_gas_constant_joule_per_mol_kelvin),
        )
        gas_content = fortran_multiply(gas_content, temperature)
        gas_content = fortran_divide(gas_content, pressure)
        threshold = fortran_multiply(
            jnp.float64(parameters.maximum_bubble_saturation_ratio),
            jnp.float64(parameters.bubble_methane_fraction),
        )
        rate = fortran_add(gas_content, -threshold)
        rate = fortran_multiply(rate, concentration)
        rate = fortran_divide(
            rate,
            jnp.float64(parameters.time_step_seconds),
        )
        active = (
            (layer_number > branch_water_table[:, None])
            & (
                temperature
                > jnp.float64(parameters.freezing_temperature_kelvin)
            )
            & (gas_content > threshold)
        )
        return jnp.where(active, rate, jnp.float64(0.0)), gas_content, pressure

    unsaturated_rate, unsaturated_gas_content, unsaturated_pressure = branch(
        False,
        inputs.unsaturated_methane_concentration_column_level,
    )
    saturated_rate, saturated_gas_content, saturated_pressure = branch(
        True,
        inputs.saturated_methane_concentration_column_level,
    )

    def update_level_state(previous: jnp.ndarray, updated: jnp.ndarray) -> jnp.ndarray:
        previous = jnp.asarray(previous, dtype=jnp.float64)
        active_result = jnp.where(
            methane_column[:, None],
            updated,
            previous[:, :NUMBER_OF_METHANE_SOIL_LEVELS],
        )
        return jnp.concatenate(
            (active_result, previous[:, NUMBER_OF_METHANE_SOIL_LEVELS:]),
            axis=1,
        )

    return MethaneEbullitionResult(
        state=MethaneEbullitionState(
            unsaturated_ebullition_column_level=update_level_state(
                state.unsaturated_ebullition_column_level,
                unsaturated_rate,
            ),
            saturated_ebullition_column_level=update_level_state(
                state.saturated_ebullition_column_level,
                saturated_rate,
            ),
            unsaturated_ebullition_total_column=jnp.asarray(
                state.unsaturated_ebullition_total_column,
                dtype=jnp.float64,
            ),
            saturated_ebullition_total_column=jnp.asarray(
                state.saturated_ebullition_total_column,
                dtype=jnp.float64,
            ),
        ),
        diagnostics=MethaneEbullitionDiagnostics(
            unsaturated_volumetric_gas_content_column_level=(
                unsaturated_gas_content
            ),
            saturated_volumetric_gas_content_column_level=saturated_gas_content,
            unsaturated_pressure_column_level=unsaturated_pressure,
            saturated_pressure_column_level=saturated_pressure,
            methane_column_count=jnp.sum(methane_column, dtype=jnp.int32),
        ),
    )


class MethaneGasCompetitionInputs(NamedTuple):
    methane_column: jnp.ndarray
    unsaturated_methane_concentration_column_level: jnp.ndarray
    saturated_methane_concentration_column_level: jnp.ndarray
    unsaturated_oxygen_concentration_column_level: jnp.ndarray
    saturated_oxygen_concentration_column_level: jnp.ndarray
    unsaturated_methane_production_column_level: jnp.ndarray
    saturated_methane_production_column_level: jnp.ndarray
    unsaturated_oxygen_aerenchyma_column_level: jnp.ndarray
    saturated_oxygen_aerenchyma_column_level: jnp.ndarray


class MethaneGasCompetitionState(NamedTuple):
    unsaturated_methane_oxidation_column_level: jnp.ndarray
    saturated_methane_oxidation_column_level: jnp.ndarray
    unsaturated_oxygen_oxidation_column_level: jnp.ndarray
    saturated_oxygen_oxidation_column_level: jnp.ndarray
    unsaturated_methane_aerenchyma_column_level: jnp.ndarray
    saturated_methane_aerenchyma_column_level: jnp.ndarray
    unsaturated_methane_ebullition_column_level: jnp.ndarray
    saturated_methane_ebullition_column_level: jnp.ndarray
    unsaturated_oxygen_decomposition_column_level: jnp.ndarray
    saturated_oxygen_decomposition_column_level: jnp.ndarray
    unsaturated_oxygen_stress_column_level: jnp.ndarray
    saturated_oxygen_stress_column_level: jnp.ndarray
    unsaturated_methane_stress_column_level: jnp.ndarray
    saturated_methane_stress_column_level: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MethaneGasCompetitionParameters:
    time_step_seconds: float = 1800.0

    def tree_flatten(self):
        return (), (self.time_step_seconds,)

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del children
        return cls(*auxiliary)


DEFAULT_METHANE_GAS_COMPETITION_PARAMETERS = MethaneGasCompetitionParameters()


class MethaneGasCompetitionDiagnostics(NamedTuple):
    unsaturated_initial_oxygen_demand_column_level: jnp.ndarray
    saturated_initial_oxygen_demand_column_level: jnp.ndarray
    unsaturated_initial_methane_demand_column_level: jnp.ndarray
    saturated_initial_methane_demand_column_level: jnp.ndarray
    methane_column_count: jnp.ndarray


class MethaneGasCompetitionResult(NamedTuple):
    state: MethaneGasCompetitionState
    diagnostics: MethaneGasCompetitionDiagnostics


def _validate_methane_gas_competition_shapes(
    inputs: MethaneGasCompetitionInputs,
    state: MethaneGasCompetitionState,
) -> tuple[int, int]:
    if jnp.ndim(inputs.methane_column) != 1:
        raise ValueError("methane_column must be a vector")
    column_count = inputs.methane_column.shape[0]
    level_shape = jnp.shape(
        inputs.unsaturated_methane_concentration_column_level
    )
    if len(level_shape) != 2 or level_shape[0] != column_count:
        raise ValueError("methane concentration must be a column-by-level matrix")
    if level_shape[1] < NUMBER_OF_METHANE_SOIL_LEVELS:
        raise ValueError("methane gas competition requires 20 soil levels")
    for name in MethaneGasCompetitionInputs._fields[2:]:
        if jnp.shape(getattr(inputs, name)) != level_shape:
            raise ValueError(f"{name} must match the column-level shape")
    for name in MethaneGasCompetitionState._fields:
        if jnp.shape(getattr(state, name)) != level_shape:
            raise ValueError(f"{name} must match the column-level shape")
    return column_count, level_shape[1]


def methane_gas_competition_active_case(
    inputs: MethaneGasCompetitionInputs,
    state: MethaneGasCompetitionState,
    parameters: MethaneGasCompetitionParameters = (
        DEFAULT_METHANE_GAS_COMPETITION_PARAMETERS
    ),
) -> MethaneGasCompetitionResult:

    _validate_methane_gas_competition_shapes(inputs, state)
    methane_column = jnp.asarray(inputs.methane_column, dtype=jnp.bool_)
    time_step = jnp.float64(parameters.time_step_seconds)
    one = jnp.float64(1.0)

    def active_levels(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(value, dtype=jnp.float64)[
            :, :NUMBER_OF_METHANE_SOIL_LEVELS
        ]

    def compete(
        methane_concentration: jnp.ndarray,
        oxygen_concentration: jnp.ndarray,
        methane_production: jnp.ndarray,
        oxygen_aerenchyma: jnp.ndarray,
        methane_oxidation: jnp.ndarray,
        oxygen_oxidation: jnp.ndarray,
        methane_aerenchyma: jnp.ndarray,
        methane_ebullition: jnp.ndarray,
        oxygen_decomposition: jnp.ndarray,
    ) -> tuple[jnp.ndarray, ...]:
        methane_concentration = active_levels(methane_concentration)
        oxygen_concentration = active_levels(oxygen_concentration)
        methane_production = active_levels(methane_production)
        oxygen_aerenchyma = active_levels(oxygen_aerenchyma)
        methane_oxidation = active_levels(methane_oxidation)
        oxygen_oxidation = active_levels(oxygen_oxidation)
        methane_aerenchyma = active_levels(methane_aerenchyma)
        methane_ebullition = active_levels(methane_ebullition)
        oxygen_decomposition = active_levels(oxygen_decomposition)

        oxygen_demand = fortran_add(
            oxygen_decomposition,
            oxygen_oxidation,
        )
        oxygen_available = fortran_add(
            fortran_divide(oxygen_concentration, time_step),
            oxygen_aerenchyma,
        )
        positive_oxygen_demand = oxygen_demand > jnp.float64(0.0)
        oxygen_shortfall = ~(oxygen_available > oxygen_demand)
        oxygen_stress = jnp.where(
            positive_oxygen_demand,
            jnp.where(
                ~oxygen_shortfall,
                one,
                fortran_divide(
                    oxygen_available,
                    jnp.where(positive_oxygen_demand & oxygen_shortfall,
                              oxygen_demand, one),
                ),
            ),
            one,
        )

        methane_demand = fortran_add(
            fortran_add(methane_oxidation, methane_aerenchyma),
            methane_ebullition,
        )
        methane_available = fortran_add(
            fortran_divide(methane_concentration, time_step),
            methane_production,
        )
        positive_methane_demand = methane_demand > jnp.float64(0.0)
        methane_stress = jnp.where(
            positive_methane_demand,
            jnp.minimum(
                fortran_divide(
                    methane_available,
                    jnp.where(positive_methane_demand, methane_demand, one),
                ),
                one,
            ),
            one,
        )

        limited = (oxygen_stress < one) | (methane_stress < one)
        methane_limited = methane_stress <= oxygen_stress

        positive_decomposition = oxygen_decomposition > jnp.float64(0.0)
        oxygen_after_oxidation = fortran_add(
            oxygen_available,
            -fortran_multiply(methane_stress, oxygen_oxidation),
        )
        methane_limited_oxygen_stress = jnp.where(
            (oxygen_stress < one) & positive_decomposition,
            jnp.minimum(
                fortran_divide(
                    oxygen_after_oxidation,
                    jnp.where(
                        (oxygen_stress < one) & positive_decomposition,
                        oxygen_decomposition,
                        one,
                    ),
                ),
                one,
            ),
            jnp.where(oxygen_stress < one, one, oxygen_stress),
        )

        nonoxidation_methane_demand = fortran_add(
            methane_aerenchyma,
            methane_ebullition,
        )
        positive_nonoxidation_demand = (
            nonoxidation_methane_demand > jnp.float64(0.0)
        )
        methane_after_oxidation = fortran_add(
            methane_available,
            -fortran_multiply(oxygen_stress, methane_oxidation),
        )
        oxygen_limited_methane_stress = jnp.where(
            (methane_stress < one) & positive_nonoxidation_demand,
            jnp.minimum(
                fortran_divide(
                    methane_after_oxidation,
                    jnp.where(
                        (methane_stress < one) & positive_nonoxidation_demand,
                        nonoxidation_methane_demand,
                        one,
                    ),
                ),
                one,
            ),
            jnp.where(methane_stress < one, one, methane_stress),
        )

        resolved_oxygen_stress = jnp.where(
            limited & methane_limited,
            methane_limited_oxygen_stress,
            oxygen_stress,
        )
        resolved_methane_stress = jnp.where(
            limited & ~methane_limited,
            oxygen_limited_methane_stress,
            methane_stress,
        )
        oxidation_scale = jnp.where(
            limited,
            jnp.where(methane_limited, methane_stress, oxygen_stress),
            one,
        )
        resolved_methane_oxidation = fortran_multiply(
            methane_oxidation,
            oxidation_scale,
        )
        resolved_oxygen_oxidation = fortran_multiply(
            oxygen_oxidation,
            oxidation_scale,
        )
        resolved_methane_aerenchyma = fortran_multiply(
            methane_aerenchyma,
            resolved_methane_stress,
        )
        resolved_methane_ebullition = fortran_multiply(
            methane_ebullition,
            resolved_methane_stress,
        )
        resolved_oxygen_decomposition = fortran_multiply(
            oxygen_decomposition,
            resolved_oxygen_stress,
        )
        return (
            resolved_methane_oxidation,
            resolved_oxygen_oxidation,
            resolved_methane_aerenchyma,
            resolved_methane_ebullition,
            resolved_oxygen_decomposition,
            resolved_oxygen_stress,
            resolved_methane_stress,
            oxygen_demand,
            methane_demand,
        )

    unsaturated = compete(
        inputs.unsaturated_methane_concentration_column_level,
        inputs.unsaturated_oxygen_concentration_column_level,
        inputs.unsaturated_methane_production_column_level,
        inputs.unsaturated_oxygen_aerenchyma_column_level,
        state.unsaturated_methane_oxidation_column_level,
        state.unsaturated_oxygen_oxidation_column_level,
        state.unsaturated_methane_aerenchyma_column_level,
        state.unsaturated_methane_ebullition_column_level,
        state.unsaturated_oxygen_decomposition_column_level,
    )
    saturated = compete(
        inputs.saturated_methane_concentration_column_level,
        inputs.saturated_oxygen_concentration_column_level,
        inputs.saturated_methane_production_column_level,
        inputs.saturated_oxygen_aerenchyma_column_level,
        state.saturated_methane_oxidation_column_level,
        state.saturated_oxygen_oxidation_column_level,
        state.saturated_methane_aerenchyma_column_level,
        state.saturated_methane_ebullition_column_level,
        state.saturated_oxygen_decomposition_column_level,
    )

    def preserve_filtered_and_deep(
        previous: jnp.ndarray,
        updated: jnp.ndarray,
    ) -> jnp.ndarray:
        previous = jnp.asarray(previous, dtype=jnp.float64)
        active_result = jnp.where(
            methane_column[:, None],
            updated,
            previous[:, :NUMBER_OF_METHANE_SOIL_LEVELS],
        )
        return jnp.concatenate(
            (active_result, previous[:, NUMBER_OF_METHANE_SOIL_LEVELS:]),
            axis=1,
        )

    unsat_indices = (0, 1, 2, 3, 4, 5, 6)
    sat_indices = (0, 1, 2, 3, 4, 5, 6)
    previous_pairs = (
        (
            state.unsaturated_methane_oxidation_column_level,
            state.saturated_methane_oxidation_column_level,
        ),
        (
            state.unsaturated_oxygen_oxidation_column_level,
            state.saturated_oxygen_oxidation_column_level,
        ),
        (
            state.unsaturated_methane_aerenchyma_column_level,
            state.saturated_methane_aerenchyma_column_level,
        ),
        (
            state.unsaturated_methane_ebullition_column_level,
            state.saturated_methane_ebullition_column_level,
        ),
        (
            state.unsaturated_oxygen_decomposition_column_level,
            state.saturated_oxygen_decomposition_column_level,
        ),
        (
            state.unsaturated_oxygen_stress_column_level,
            state.saturated_oxygen_stress_column_level,
        ),
        (
            state.unsaturated_methane_stress_column_level,
            state.saturated_methane_stress_column_level,
        ),
    )
    resolved_pairs = tuple(
        (
            preserve_filtered_and_deep(previous[0], unsaturated[unsat_index]),
            preserve_filtered_and_deep(previous[1], saturated[sat_index]),
        )
        for previous, unsat_index, sat_index in zip(
            previous_pairs,
            unsat_indices,
            sat_indices,
            strict=True,
        )
    )
    return MethaneGasCompetitionResult(
        state=MethaneGasCompetitionState(
            unsaturated_methane_oxidation_column_level=resolved_pairs[0][0],
            saturated_methane_oxidation_column_level=resolved_pairs[0][1],
            unsaturated_oxygen_oxidation_column_level=resolved_pairs[1][0],
            saturated_oxygen_oxidation_column_level=resolved_pairs[1][1],
            unsaturated_methane_aerenchyma_column_level=resolved_pairs[2][0],
            saturated_methane_aerenchyma_column_level=resolved_pairs[2][1],
            unsaturated_methane_ebullition_column_level=resolved_pairs[3][0],
            saturated_methane_ebullition_column_level=resolved_pairs[3][1],
            unsaturated_oxygen_decomposition_column_level=resolved_pairs[4][0],
            saturated_oxygen_decomposition_column_level=resolved_pairs[4][1],
            unsaturated_oxygen_stress_column_level=resolved_pairs[5][0],
            saturated_oxygen_stress_column_level=resolved_pairs[5][1],
            unsaturated_methane_stress_column_level=resolved_pairs[6][0],
            saturated_methane_stress_column_level=resolved_pairs[6][1],
        ),
        diagnostics=MethaneGasCompetitionDiagnostics(
            unsaturated_initial_oxygen_demand_column_level=unsaturated[7],
            saturated_initial_oxygen_demand_column_level=saturated[7],
            unsaturated_initial_methane_demand_column_level=unsaturated[8],
            saturated_initial_methane_demand_column_level=saturated[8],
            methane_column_count=jnp.sum(methane_column, dtype=jnp.int32),
        ),
    )


__all__ = [
    "DEFAULT_METHANE_AERENCHYMA_PARAMETERS",
    "DEFAULT_METHANE_ANNUAL_UPDATE_PARAMETERS",
    "DEFAULT_METHANE_EBULLITION_PARAMETERS",
    "DEFAULT_METHANE_GAS_COMPETITION_PARAMETERS",
    "DEFAULT_METHANE_OXIDATION_PARAMETERS",
    "DEFAULT_METHANE_PRODUCTION_PARAMETERS",
    "DEFAULT_METHANE_WATER_TABLE_PARAMETERS",
    "MethaneAerenchymaDiagnostics",
    "MethaneAerenchymaInputs",
    "MethaneAerenchymaParameters",
    "MethaneAerenchymaResult",
    "MethaneAerenchymaState",
    "MethaneAnnualUpdateDiagnostics",
    "MethaneAnnualUpdateInputs",
    "MethaneAnnualUpdateParameters",
    "MethaneAnnualUpdateResult",
    "MethaneAnnualUpdateState",
    "MethaneEbullitionDiagnostics",
    "MethaneEbullitionInputs",
    "MethaneEbullitionParameters",
    "MethaneEbullitionResult",
    "MethaneEbullitionState",
    "MethaneGasCompetitionDiagnostics",
    "MethaneGasCompetitionInputs",
    "MethaneGasCompetitionParameters",
    "MethaneGasCompetitionResult",
    "MethaneGasCompetitionState",
    "MethaneOxidationDiagnostics",
    "MethaneOxidationInputs",
    "MethaneOxidationParameters",
    "MethaneOxidationResult",
    "MethaneOxidationState",
    "MethaneProductionDiagnostics",
    "MethaneProductionInputs",
    "MethaneProductionParameters",
    "MethaneProductionResult",
    "MethaneProductionState",
    "MethaneWaterTableDiagnostics",
    "MethaneWaterTableInputs",
    "MethaneWaterTableParameters",
    "MethaneWaterTableResult",
    "MethaneWaterTableState",
    "diagnose_methane_water_table_layer",
    "methane_aerenchyma_active_case",
    "methane_ebullition_active_case",
    "methane_gas_competition_active_case",
    "methane_production_active_case",
    "methane_oxidation_active_case",
    "update_methane_annual_means",
]
