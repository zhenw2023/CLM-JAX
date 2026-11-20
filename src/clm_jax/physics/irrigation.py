
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import lax

from clm_jax.constants import (
    CLM_SPECIAL_VALUE,
    DENSITY_FRESH_WATER,
    FREEZING_TEMPERATURE_FRESHWATER,
)
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.numerics import fortran_add, fortran_divide, fortran_multiply
from clm_jax.physics.subgrid import patch_to_column_mean

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase


class IrrigationState(NamedTuple):

    rate_patch: jnp.ndarray
    demand_rate_patch: jnp.ndarray
    steps_left_patch: jnp.ndarray


class IrrigationApplicationParameters(NamedTuple):

    maximum_patches_per_column: int = 79

    @classmethod
    def from_case(cls, case: CimeCase) -> IrrigationApplicationParameters:
        from clm_jax.io.namelist import read_case_namelist_group  

        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        return cls(maximum_patches_per_column=int(general["maxpatch_pft"]))


DEFAULT_IRRIGATION_APPLICATION_PARAMETERS = IrrigationApplicationParameters()


class IrrigationDemandParameters(NamedTuple):

    minimum_leaf_area_index: float = 0.0
    start_time_seconds: int = 21600
    irrigation_length_seconds: int = 14400
    target_soil_matric_potential: float = -3400.0
    irrigation_depth: float = 0.6
    threshold_fraction: float = 1.0
    river_volume_threshold_fraction: float = 0.0
    limit_irrigation_if_runoff_prognostic: bool = False
    time_step_seconds: int = 1800
    maximum_columns_per_gridcell: int = 79

    @classmethod
    def from_case(
        cls,
        case: CimeCase,
        *,
        time_step_seconds: int | None = None,
    ) -> IrrigationDemandParameters:

        from clm_jax.io.namelist import read_case_namelist_group  
        from clm_jax.io.restart import (  
            configured_land_restart,
            load_restart_clock,
        )

        irrigation = read_case_namelist_group(case, "lnd_in", "irrigation_inparm")
        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        if time_step_seconds is None:
            restart = configured_land_restart(case)
            time_step_seconds = load_restart_clock(restart).step_seconds
        limit = bool(irrigation["limit_irrigation_if_rof_enabled"])
        river_threshold = irrigation.get("irrig_river_volume_threshold", 0.0)
        if not limit:
            river_threshold = 0.0
        return cls(
            minimum_leaf_area_index=float(irrigation["irrig_min_lai"]),
            start_time_seconds=int(irrigation["irrig_start_time"]),
            irrigation_length_seconds=int(irrigation["irrig_length"]),
            target_soil_matric_potential=float(irrigation["irrig_target_smp"]),
            irrigation_depth=float(irrigation["irrig_depth"]),
            threshold_fraction=float(irrigation["irrig_threshold_fraction"]),
            river_volume_threshold_fraction=float(river_threshold),
            limit_irrigation_if_runoff_prognostic=limit,
            time_step_seconds=int(time_step_seconds),
            maximum_columns_per_gridcell=int(general["maxpatch_pft"]),
        )


DEFAULT_IRRIGATION_DEMAND_PARAMETERS = IrrigationDemandParameters()


class IrrigationDemandInputs(NamedTuple):

    exposed_vegetation_patch: jnp.ndarray
    irrigated_patch: jnp.ndarray
    leaf_area_index: jnp.ndarray
    soil_temperature: jnp.ndarray
    effective_porosity: jnp.ndarray
    soil_liquid_water: jnp.ndarray
    relative_saturation_wilting_point: jnp.ndarray
    relative_saturation_target: jnp.ndarray
    soil_layer_depth: jnp.ndarray
    soil_layer_thickness: jnp.ndarray
    bedrock_soil_layer_count: jnp.ndarray
    river_water_volume: jnp.ndarray
    gridcell_area_square_kilometers: jnp.ndarray
    current_time_of_day_seconds: jnp.ndarray
    runoff_prognostic: jnp.ndarray


class IrrigationDemandDiagnostics(NamedTuple):
    check_patch: jnp.ndarray
    check_column: jnp.ndarray
    included_soil_layer: jnp.ndarray
    liquid_water_total: jnp.ndarray
    target_water_total: jnp.ndarray
    wilting_water_total: jnp.ndarray
    threshold_water_total: jnp.ndarray
    deficit: jnp.ndarray
    gridcell_deficit: jnp.ndarray
    gridcell_limit_ratio: jnp.ndarray
    limited_deficit: jnp.ndarray
    negative_deficit_error: jnp.ndarray


class IrrigationDemandResult(NamedTuple):
    state: IrrigationState
    diagnostics: IrrigationDemandDiagnostics


def _fortran_nint(value: jnp.ndarray) -> jnp.ndarray:

    value = jnp.asarray(value, dtype=jnp.float64)
    rounded = jnp.where(
        value >= jnp.float64(0.0),
        jnp.floor(value + jnp.float64(0.5)),
        jnp.ceil(value - jnp.float64(0.5)),
    )
    return rounded.astype(jnp.int32)


def _column_to_gridcell_mean(
    topology: SubgridTopology,
    column_values: jnp.ndarray,
    *,
    maximum_columns_per_gridcell: int,
) -> jnp.ndarray:

    values = jnp.asarray(column_values, dtype=jnp.float64)
    gridcell = jnp.asarray(topology.column.gridcell, dtype=jnp.int32)
    weight = jnp.asarray(topology.column.weight_on_gridcell, dtype=jnp.float64)
    active = jnp.asarray(topology.column.active, dtype=jnp.bool_)
    number_of_columns = values.shape[0]
    number_of_gridcells = topology.grid.longitude_degrees.shape[0]
    grid_shape = (number_of_gridcells,)
    if number_of_columns == 0:
        return jnp.full(grid_shape, jnp.float64(CLM_SPECIAL_VALUE))

    column_index = jnp.arange(number_of_columns, dtype=jnp.int32)
    first_column = jnp.full(grid_shape, number_of_columns, dtype=jnp.int32)
    first_column = first_column.at[gridcell].min(column_index)
    grid_index = jnp.arange(number_of_gridcells, dtype=jnp.int32)

    initial = (
        jnp.zeros(grid_shape, dtype=jnp.float64),
        jnp.zeros(grid_shape, dtype=jnp.float64),
    )

    def accumulate_offset(
        offset: int,
        carry: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        totals, sum_weights = carry
        unbounded_column = first_column + jnp.int32(offset)
        bounded_column = jnp.clip(unbounded_column, 0, number_of_columns - 1)
        selected_value = values[bounded_column]
        selected_weight = weight[bounded_column]
        belongs_to_gridcell = (unbounded_column < number_of_columns) & (
            gridcell[bounded_column] == grid_index
        )
        contributes = (
            belongs_to_gridcell
            & active[bounded_column]
            & (selected_weight != jnp.float64(0.0))
            & (selected_value != jnp.float64(CLM_SPECIAL_VALUE))
        )
        contribution = jnp.where(
            contributes,
            fortran_multiply(selected_value, selected_weight),
            jnp.float64(0.0),
        )
        contribution_weight = jnp.where(
            contributes,
            selected_weight,
            jnp.float64(0.0),
        )
        return (
            fortran_add(totals, contribution),
            fortran_add(sum_weights, contribution_weight),
        )

    totals, sum_weights = lax.fori_loop(
        0,
        maximum_columns_per_gridcell,
        accumulate_offset,
        initial,
    )
    has_contributor = sum_weights != jnp.float64(0.0)
    safe_weights = jnp.where(has_contributor, sum_weights, jnp.float64(1.0))
    return jnp.where(
        has_contributor,
        fortran_divide(totals, safe_weights),
        jnp.float64(CLM_SPECIAL_VALUE),
    )


def _validate_irrigation_demand_shapes(
    topology: SubgridTopology,
    state: IrrigationState,
    inputs: IrrigationDemandInputs,
    parameters: IrrigationDemandParameters,
) -> None:
    patch_shape = topology.patch.column.shape
    column_shape = topology.column.gridcell.shape
    grid_shape = topology.grid.longitude_degrees.shape
    state_fields = (state.rate_patch, state.demand_rate_patch, state.steps_left_patch)
    patch_fields = (
        inputs.exposed_vegetation_patch,
        inputs.irrigated_patch,
        inputs.leaf_area_index,
    )
    soil_fields = (
        inputs.soil_temperature,
        inputs.effective_porosity,
        inputs.soil_liquid_water,
        inputs.relative_saturation_wilting_point,
        inputs.relative_saturation_target,
        inputs.soil_layer_depth,
        inputs.soil_layer_thickness,
    )
    if patch_shape != topology.patch.gridcell.shape or len(patch_shape) != 1:
        raise ValueError("patch topology fields must be identically sized vectors")
    if column_shape != topology.column.active.shape or len(column_shape) != 1:
        raise ValueError("column topology fields must be identically sized vectors")
    if any(jnp.shape(field) != patch_shape for field in (*state_fields, *patch_fields)):
        raise ValueError("irrigation state and patch inputs must contain one value per patch")
    if not soil_fields or jnp.ndim(soil_fields[0]) != 2:
        raise ValueError("soil inputs must use column-by-level matrices")
    soil_shape = jnp.shape(soil_fields[0])
    if soil_shape[0] != column_shape[0] or any(
        jnp.shape(field) != soil_shape for field in soil_fields
    ):
        raise ValueError("all soil inputs must use the same native column-by-level shape")
    if jnp.shape(inputs.bedrock_soil_layer_count) != column_shape:
        raise ValueError("bedrock layer count must contain one value per column")
    if not (
        jnp.shape(inputs.river_water_volume)
        == jnp.shape(inputs.gridcell_area_square_kilometers)
        == grid_shape
    ):
        raise ValueError("river volume and area must contain one value per gridcell")
    if jnp.shape(inputs.current_time_of_day_seconds) != ():
        raise ValueError("current time of day must be scalar")
    if jnp.shape(inputs.runoff_prognostic) != ():
        raise ValueError("runoff-prognostic flag must be scalar")
    if parameters.time_step_seconds <= 0 or parameters.time_step_seconds > 86400:
        raise ValueError("time_step_seconds must lie in [1, 86400]")
    if parameters.irrigation_length_seconds <= 0 or parameters.irrigation_length_seconds > 86400:
        raise ValueError("irrigation_length_seconds must lie in [1, 86400]")
    if not 0 <= parameters.start_time_seconds < 86400:
        raise ValueError("start_time_seconds must lie in [0, 86400)")
    if parameters.minimum_leaf_area_index < 0.0:
        raise ValueError("minimum_leaf_area_index must be nonnegative")
    if parameters.irrigation_depth < 0.0:
        raise ValueError("irrigation_depth must be nonnegative")
    if not 0.0 <= parameters.threshold_fraction <= 1.0:
        raise ValueError("threshold_fraction must lie in [0, 1]")
    if parameters.maximum_columns_per_gridcell <= 0:
        raise ValueError("maximum_columns_per_gridcell must be positive")


def calculate_irrigation_demand(
    topology: SubgridTopology,
    state: IrrigationState,
    inputs: IrrigationDemandInputs,
    *,
    parameters: IrrigationDemandParameters = DEFAULT_IRRIGATION_DEMAND_PARAMETERS,
) -> IrrigationDemandResult:

    _validate_irrigation_demand_shapes(topology, state, inputs, parameters)
    rate_patch = jnp.asarray(state.rate_patch, dtype=jnp.float64)
    demand_rate_patch = jnp.asarray(state.demand_rate_patch, dtype=jnp.float64)
    steps_left_patch = jnp.asarray(state.steps_left_patch, dtype=jnp.int32)
    exposed = jnp.asarray(inputs.exposed_vegetation_patch, dtype=jnp.bool_)
    irrigated = jnp.asarray(inputs.irrigated_patch, dtype=jnp.bool_)
    leaf_area_index = jnp.asarray(inputs.leaf_area_index, dtype=jnp.float64)
    soil_temperature = jnp.asarray(inputs.soil_temperature, dtype=jnp.float64)
    effective_porosity = jnp.asarray(inputs.effective_porosity, dtype=jnp.float64)
    soil_liquid_water = jnp.asarray(inputs.soil_liquid_water, dtype=jnp.float64)
    relative_saturation_wilting = jnp.asarray(
        inputs.relative_saturation_wilting_point,
        dtype=jnp.float64,
    )
    relative_saturation_target = jnp.asarray(
        inputs.relative_saturation_target,
        dtype=jnp.float64,
    )
    soil_layer_depth = jnp.asarray(inputs.soil_layer_depth, dtype=jnp.float64)
    soil_layer_thickness = jnp.asarray(inputs.soil_layer_thickness, dtype=jnp.float64)
    bedrock_layer_count = jnp.asarray(inputs.bedrock_soil_layer_count, dtype=jnp.int32)

    time_step = jnp.int32(parameters.time_step_seconds)
    previous_time_of_day = (
        jnp.asarray(inputs.current_time_of_day_seconds, dtype=jnp.int32) - time_step
    )
    degrees_per_second = fortran_divide(jnp.float64(15.0), jnp.float64(3600.0))
    longitude_seconds = _fortran_nint(
        fortran_divide(
            jnp.asarray(topology.grid.longitude_degrees, dtype=jnp.float64),
            degrees_per_second,
        )
    )
    patch_local_time = jnp.mod(
        previous_time_of_day + longitude_seconds[topology.patch.gridcell],
        jnp.int32(86400),
    )
    seconds_since_start = jnp.mod(
        patch_local_time - jnp.int32(parameters.start_time_seconds),
        jnp.int32(86400),
    )
    check_patch = (
        exposed
        & irrigated
        & (leaf_area_index > jnp.float64(parameters.minimum_leaf_area_index))
        & (seconds_since_start < time_step)
    )

    number_of_columns = topology.column.gridcell.shape[0]
    check_column_count = (
        jnp.zeros(number_of_columns, dtype=jnp.int32)
        .at[topology.patch.column]
        .max(check_patch.astype(jnp.int32))
    )
    check_column = check_column_count > jnp.int32(0)
    included_soil_layer = jnp.zeros(soil_liquid_water.shape, dtype=jnp.bool_)
    initial_soil_state = (
        jnp.zeros(number_of_columns, dtype=jnp.bool_),
        included_soil_layer,
        jnp.zeros(number_of_columns, dtype=jnp.float64),
        jnp.zeros(number_of_columns, dtype=jnp.float64),
        jnp.zeros(number_of_columns, dtype=jnp.float64),
    )

    def accumulate_soil_layer(
        level: int,
        carry: tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        reached_depth, included, liquid_total, target_total, wilting_total = carry
        stops_here = (
            (soil_layer_depth[:, level] > jnp.float64(parameters.irrigation_depth))
            | (jnp.int32(level + 1) > bedrock_layer_count)
            | (soil_temperature[:, level] <= jnp.float64(FREEZING_TEMPERATURE_FRESHWATER))
        )
        include = check_column & ~reached_depth & ~stops_here
        reached_depth = reached_depth | (check_column & ~reached_depth & stops_here)
        included = included.at[:, level].set(include)
        liquid = jnp.where(include, soil_liquid_water[:, level], jnp.float64(0.0))
        porosity = jnp.where(include, effective_porosity[:, level], jnp.float64(0.0))
        thickness = jnp.where(include, soil_layer_thickness[:, level], jnp.float64(0.0))
        target_saturation = jnp.where(
            include,
            relative_saturation_target[:, level],
            jnp.float64(0.0),
        )
        wilting_saturation = jnp.where(
            include,
            relative_saturation_wilting[:, level],
            jnp.float64(0.0),
        )
        target_layer = fortran_multiply(
            fortran_multiply(
                fortran_multiply(porosity, target_saturation),
                jnp.float64(DENSITY_FRESH_WATER),
            ),
            thickness,
        )
        wilting_layer = fortran_multiply(
            fortran_multiply(
                fortran_multiply(porosity, wilting_saturation),
                jnp.float64(DENSITY_FRESH_WATER),
            ),
            thickness,
        )
        return (
            reached_depth,
            included,
            fortran_add(liquid_total, liquid),
            fortran_add(target_total, target_layer),
            fortran_add(wilting_total, wilting_layer),
        )

    (
        _,
        included_soil_layer,
        liquid_water_total,
        target_water_total,
        wilting_water_total,
    ) = lax.fori_loop(
        0,
        soil_liquid_water.shape[1],
        accumulate_soil_layer,
        initial_soil_state,
    )
    threshold_water_total = fortran_add(
        wilting_water_total,
        fortran_multiply(
            jnp.float64(parameters.threshold_fraction),
            fortran_add(target_water_total, -wilting_water_total),
        ),
    )
    calculated_deficit = fortran_add(target_water_total, -liquid_water_total)
    needs_irrigation = check_column & (liquid_water_total < threshold_water_total)
    deficit = jnp.where(needs_irrigation, calculated_deficit, jnp.float64(0.0))
    negative_deficit_error = needs_irrigation & (calculated_deficit < jnp.float64(0.0))

    gridcell_deficit = _column_to_gridcell_mean(
        topology,
        deficit,
        maximum_columns_per_gridcell=parameters.maximum_columns_per_gridcell,
    )
    river_water_volume = jnp.asarray(inputs.river_water_volume, dtype=jnp.float64)
    gridcell_area = jnp.asarray(
        inputs.gridcell_area_square_kilometers,
        dtype=jnp.float64,
    )
    available_river_water = fortran_multiply(
        river_water_volume,
        fortran_add(
            jnp.float64(1.0),
            -jnp.float64(parameters.river_volume_threshold_fraction),
        ),
    )
    safe_gridcell_area = jnp.where(
        gridcell_area != jnp.float64(0.0),
        gridcell_area,
        jnp.float64(1.0),
    )
    maximum_supported_deficit = jnp.where(
        river_water_volume > jnp.float64(0.0),
        fortran_multiply(
            fortran_divide(available_river_water, safe_gridcell_area),
            jnp.float64(1.0e-3),
        ),
        jnp.float64(0.0),
    )
    needs_limiting = gridcell_deficit > maximum_supported_deficit
    safe_gridcell_deficit = jnp.where(
        needs_limiting,
        gridcell_deficit,
        jnp.float64(1.0),
    )
    gridcell_limit_ratio = jnp.where(
        needs_limiting,
        fortran_divide(maximum_supported_deficit, safe_gridcell_deficit),
        jnp.float64(1.0),
    )
    volume_limited_deficit = jnp.where(
        check_column,
        fortran_multiply(
            deficit,
            gridcell_limit_ratio[topology.column.gridcell],
        ),
        jnp.float64(0.0),
    )
    limit_irrigation = jnp.asarray(
        parameters.limit_irrigation_if_runoff_prognostic,
        dtype=jnp.bool_,
    ) & jnp.asarray(inputs.runoff_prognostic, dtype=jnp.bool_)
    limited_deficit = jnp.where(limit_irrigation, volume_limited_deficit, deficit)

    irrigation_steps = (
        parameters.irrigation_length_seconds + parameters.time_step_seconds - 1
    ) // parameters.time_step_seconds
    rate_denominator = jnp.float64(parameters.time_step_seconds * irrigation_steps)
    patch_column = topology.patch.column
    calculated_rate = fortran_divide(limited_deficit[patch_column], rate_denominator)
    calculated_demand_rate = fortran_divide(deficit[patch_column], rate_denominator)
    next_state = IrrigationState(
        rate_patch=jnp.where(check_patch, calculated_rate, rate_patch),
        demand_rate_patch=jnp.where(
            check_patch,
            calculated_demand_rate,
            demand_rate_patch,
        ),
        steps_left_patch=jnp.where(
            check_patch,
            jnp.int32(irrigation_steps),
            steps_left_patch,
        ),
    )
    return IrrigationDemandResult(
        state=next_state,
        diagnostics=IrrigationDemandDiagnostics(
            check_patch=check_patch,
            check_column=check_column,
            included_soil_layer=included_soil_layer,
            liquid_water_total=liquid_water_total,
            target_water_total=target_water_total,
            wilting_water_total=wilting_water_total,
            threshold_water_total=threshold_water_total,
            deficit=deficit,
            gridcell_deficit=gridcell_deficit,
            gridcell_limit_ratio=gridcell_limit_ratio,
            limited_deficit=limited_deficit,
            negative_deficit_error=negative_deficit_error,
        ),
    )


class IrrigationApplicationFluxes(NamedTuple):

    applied_patch: jnp.ndarray
    demand_patch: jnp.ndarray
    applied_column: jnp.ndarray


class IrrigationApplicationResult(NamedTuple):
    state: IrrigationState
    fluxes: IrrigationApplicationFluxes


def apply_irrigation(
    topology: SubgridTopology,
    state: IrrigationState,
    *,
    parameters: IrrigationApplicationParameters = DEFAULT_IRRIGATION_APPLICATION_PARAMETERS,
) -> IrrigationApplicationResult:

    patch_shape = topology.patch.column.shape
    rate = jnp.asarray(state.rate_patch, dtype=jnp.float64)
    demand_rate = jnp.asarray(state.demand_rate_patch, dtype=jnp.float64)
    steps_left = jnp.asarray(state.steps_left_patch, dtype=jnp.int32)
    if not (rate.shape == demand_rate.shape == steps_left.shape == patch_shape and rate.ndim == 1):
        raise ValueError("irrigation state must contain one value per native patch")

    apply_now = steps_left > jnp.int32(0)
    zeros = jnp.zeros(patch_shape, dtype=jnp.float64)
    applied_patch = jnp.where(apply_now, rate, zeros)
    demand_patch = jnp.where(apply_now, demand_rate, zeros)
    next_steps_left = jnp.where(
        apply_now,
        steps_left - jnp.int32(1),
        steps_left,
    )
    applied_column = patch_to_column_mean(
        patch_values=applied_patch,
        patch_column=topology.patch.column,
        patch_weight_on_column=topology.patch.weight_on_column,
        patch_active=topology.patch.active,
        number_of_columns=topology.column.active.shape[0],
        maximum_patches_per_column=parameters.maximum_patches_per_column,
    )
    return IrrigationApplicationResult(
        state=IrrigationState(
            rate_patch=rate,
            demand_rate_patch=demand_rate,
            steps_left_patch=next_steps_left,
        ),
        fluxes=IrrigationApplicationFluxes(
            applied_patch=applied_patch,
            demand_patch=demand_patch,
            applied_column=applied_column,
        ),
    )
