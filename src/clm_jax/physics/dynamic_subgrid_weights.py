
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.restart import LANDUNIT_CROP, LANDUNIT_SOIL, SubgridTopology
from clm_jax.physics.numerics import fortran_add, fortran_divide, fortran_multiply
from clm_jax.physics.snow_water import (
    NUMBER_OF_GROUND_LEVELS,
    NUMBER_OF_SNOW_LEVELS,
    NUMBER_OF_SOIL_SNOW_LEVELS,
)

TEMPLATE_NONE_FOUND = -9999


class DynamicPriorWeights(NamedTuple):

    patch_weight_on_gridcell: jnp.ndarray
    column_active: jnp.ndarray


class PatchWeightTransition(NamedTuple):

    old_patch_weight_on_gridcell: jnp.ndarray
    new_patch_weight_on_gridcell: jnp.ndarray
    old_column_weight_on_gridcell: jnp.ndarray
    weight_change: jnp.ndarray
    growing_old_fraction: jnp.ndarray
    growing_new_fraction: jnp.ndarray


class PatchStateUpdateResult(NamedTuple):
    state: jnp.ndarray
    flux_out_column_area: jnp.ndarray
    flux_out_gridcell_area: jnp.ndarray
    seed_addition_gridcell_area: jnp.ndarray


class PatchPartitionedFluxResult(NamedTuple):
    state: jnp.ndarray
    flux_one_gridcell_area: jnp.ndarray
    flux_two_gridcell_area: jnp.ndarray
    seed_addition_gridcell_area: jnp.ndarray


class ColumnWeightTransition(NamedTuple):

    old_column_weight_on_gridcell: jnp.ndarray
    new_column_weight_on_gridcell: jnp.ndarray
    area_gained_column: jnp.ndarray
    natural_vegetation_template_column: jnp.ndarray
    any_changes: jnp.ndarray


class ColumnStateUpdateResult(NamedTuple):
    state: jnp.ndarray
    non_conserved_mass_gridcell: jnp.ndarray
    adjustment_column: jnp.ndarray
    invalid_shrinking_column: jnp.ndarray
    inconsistent_input_column: jnp.ndarray


class NewColumnState(NamedTuple):

    soil_snow_temperature_column_level: jnp.ndarray
    soil_snow_liquid_water_column_level: jnp.ndarray
    soil_snow_ice_water_column_level: jnp.ndarray
    volumetric_water_column_ground_level: jnp.ndarray
    aquifer_water_column: jnp.ndarray


class NewColumnInitializationResult(NamedTuple):
    state: NewColumnState
    template_column: jnp.ndarray
    newly_active_column: jnp.ndarray
    template_missing_column: jnp.ndarray
    unsupported_new_column: jnp.ndarray
    invalid_positive_weight_soil_column: jnp.ndarray


def save_prior_weights(topology: SubgridTopology) -> DynamicPriorWeights:

    return DynamicPriorWeights(
        patch_weight_on_gridcell=jnp.asarray(
            topology.patch.weight_on_gridcell,
            dtype=jnp.float64,
        ),
        column_active=jnp.asarray(topology.column.active, dtype=jnp.bool_),
    )


def prepare_patch_weight_transition(
    topology_before: SubgridTopology,
    topology_after: SubgridTopology,
) -> PatchWeightTransition:

    old_patch = jnp.asarray(topology_before.patch.weight_on_gridcell, dtype=jnp.float64)
    new_patch = jnp.asarray(topology_after.patch.weight_on_gridcell, dtype=jnp.float64)
    if old_patch.shape != new_patch.shape:
        raise ValueError("old and new patch weights must have identical shapes")
    patch_column = jnp.asarray(topology_before.patch.column, dtype=jnp.int32)
    if patch_column.shape != old_patch.shape:
        raise ValueError("patch column indices must match patch weights")
    if topology_before.column.weight_on_gridcell.shape != (
        topology_before.column.gridcell.shape[0],
    ):
        raise ValueError("old column weights must match column topology")

    old_column = jnp.asarray(
        topology_before.column.weight_on_gridcell,
        dtype=jnp.float64,
    )[patch_column]
    change = fortran_add(new_patch, -old_patch)
    growing = change > jnp.float64(0.0)
    safe_new_patch = jnp.where(growing, new_patch, jnp.float64(1.0))
    old_fraction = jnp.where(
        growing,
        fortran_divide(old_patch, safe_new_patch),
        jnp.float64(1.0),
    )
    new_fraction = jnp.where(
        growing,
        fortran_divide(change, safe_new_patch),
        jnp.float64(0.0),
    )
    return PatchWeightTransition(
        old_patch_weight_on_gridcell=old_patch,
        new_patch_weight_on_gridcell=new_patch,
        old_column_weight_on_gridcell=old_column,
        weight_change=change,
        growing_old_fraction=old_fraction,
        growing_new_fraction=new_fraction,
    )


def update_patch_state(
    transition: PatchWeightTransition,
    filter_with_inactive: jnp.ndarray,
    state: jnp.ndarray,
    *,
    flux_out_column_area: jnp.ndarray | None = None,
    flux_out_gridcell_area: jnp.ndarray | None = None,
    seed: jnp.ndarray | None = None,
    seed_addition_gridcell_area: jnp.ndarray | None = None,
) -> PatchStateUpdateResult:

    values = jnp.asarray(state, dtype=jnp.float64)
    selected = jnp.asarray(filter_with_inactive, dtype=jnp.bool_)
    shape = transition.weight_change.shape
    if values.shape != shape or selected.shape != shape:
        raise ValueError("patch state, transition and filter must have identical shapes")

    zeros = jnp.zeros_like(values)
    column_flux = (
        zeros
        if flux_out_column_area is None
        else jnp.asarray(flux_out_column_area, dtype=jnp.float64)
    )
    gridcell_flux = (
        zeros
        if flux_out_gridcell_area is None
        else jnp.asarray(flux_out_gridcell_area, dtype=jnp.float64)
    )
    seed_addition = (
        zeros
        if seed_addition_gridcell_area is None
        else jnp.asarray(seed_addition_gridcell_area, dtype=jnp.float64)
    )
    if not (column_flux.shape == gridcell_flux.shape == seed_addition.shape == shape):
        raise ValueError("all optional patch accumulators must match patch state")

    change = transition.weight_change
    growing = selected & (change > jnp.float64(0.0))
    shrinking = selected & (change < jnp.float64(0.0))
    grown_state = fortran_multiply(values, transition.growing_old_fraction)
    if seed is not None:
        seed_value = jnp.asarray(seed, dtype=jnp.float64)
        if seed_value.shape != shape:
            raise ValueError("seed must match patch state")
        grown_state = fortran_add(
            grown_state,
            fortran_multiply(seed_value, transition.growing_new_fraction),
        )
        seed_candidate = fortran_add(
            seed_addition,
            fortran_multiply(seed_value, change),
        )
        seed_addition = jnp.where(growing, seed_candidate, seed_addition)

    gridcell_candidate = fortran_add(
        gridcell_flux,
        fortran_multiply(values, change),
    )
    relative_column_change = fortran_divide(
        change,
        jnp.where(
            shrinking,
            transition.old_column_weight_on_gridcell,
            jnp.float64(1.0),
        ),
    )
    column_candidate = fortran_add(
        column_flux,
        fortran_multiply(values, relative_column_change),
    )
    return PatchStateUpdateResult(
        state=jnp.where(growing, grown_state, values),
        flux_out_column_area=jnp.where(shrinking, column_candidate, column_flux),
        flux_out_gridcell_area=jnp.where(shrinking, gridcell_candidate, gridcell_flux),
        seed_addition_gridcell_area=seed_addition,
    )


def update_patch_state_partition_flux_by_type(
    transition: PatchWeightTransition,
    filter_with_inactive: jnp.ndarray,
    vegetation_type_patch: jnp.ndarray,
    flux_one_fraction_by_vegetation_type: jnp.ndarray,
    state: jnp.ndarray,
    flux_one_gridcell_area: jnp.ndarray,
    flux_two_gridcell_area: jnp.ndarray,
    *,
    seed: jnp.ndarray | None = None,
    seed_addition_gridcell_area: jnp.ndarray | None = None,
) -> PatchPartitionedFluxResult:

    values = jnp.asarray(state, dtype=jnp.float64)
    zeros = jnp.zeros_like(values)
    updated = update_patch_state(
        transition,
        filter_with_inactive,
        values,
        flux_out_gridcell_area=zeros,
        seed=seed,
        seed_addition_gridcell_area=seed_addition_gridcell_area,
    )
    vegetation_type = jnp.asarray(vegetation_type_patch, dtype=jnp.int32)
    fractions = jnp.asarray(
        flux_one_fraction_by_vegetation_type,
        dtype=jnp.float64,
    )
    if vegetation_type.shape != values.shape:
        raise ValueError("vegetation types must match patch state")
    fraction_one = fractions[vegetation_type]
    selected = jnp.asarray(filter_with_inactive, dtype=jnp.bool_)
    first = jnp.asarray(flux_one_gridcell_area, dtype=jnp.float64)
    second = jnp.asarray(flux_two_gridcell_area, dtype=jnp.float64)
    if first.shape != values.shape or second.shape != values.shape:
        raise ValueError("partitioned patch fluxes must match patch state")
    first_candidate = fortran_add(
        first,
        fortran_multiply(updated.flux_out_gridcell_area, fraction_one),
    )
    second_fraction = fortran_add(jnp.float64(1.0), -fraction_one)
    second_candidate = fortran_add(
        second,
        fortran_multiply(updated.flux_out_gridcell_area, second_fraction),
    )
    return PatchPartitionedFluxResult(
        state=updated.state,
        flux_one_gridcell_area=jnp.where(selected, first_candidate, first),
        flux_two_gridcell_area=jnp.where(selected, second_candidate, second),
        seed_addition_gridcell_area=updated.seed_addition_gridcell_area,
    )


def _first_prior_active_column(
    topology: SubgridTopology,
    prior_column_active: jnp.ndarray,
    landunit_type: int,
) -> jnp.ndarray:
    number_of_columns = topology.column.gridcell.shape[0]
    number_of_gridcells = topology.grid.longitude_degrees.shape[0]
    gridcell = jnp.asarray(topology.column.gridcell, dtype=jnp.int32)
    column_landunit_type = jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
    prior_active = jnp.asarray(prior_column_active, dtype=jnp.bool_)
    if prior_active.shape != (number_of_columns,):
        raise ValueError("prior column activity must match column topology")
    candidate = jnp.where(
        prior_active & (column_landunit_type == jnp.int32(landunit_type)),
        jnp.arange(number_of_columns, dtype=jnp.int32),
        jnp.int32(number_of_columns),
    )
    first = jnp.full(number_of_gridcells, number_of_columns, dtype=jnp.int32)
    first = first.at[gridcell].min(candidate)
    return jnp.where(
        first[gridcell] < jnp.int32(number_of_columns),
        first[gridcell],
        jnp.int32(TEMPLATE_NONE_FOUND),
    )


def prepare_column_weight_transition(
    topology_before: SubgridTopology,
    topology_after: SubgridTopology,
) -> ColumnWeightTransition:

    old = jnp.asarray(topology_before.column.weight_on_gridcell, dtype=jnp.float64)
    new = jnp.asarray(topology_after.column.weight_on_gridcell, dtype=jnp.float64)
    if old.shape != new.shape:
        raise ValueError("old and new column weights must have identical shapes")
    area_gained = fortran_add(new, -old)
    template = _first_prior_active_column(
        topology_before,
        topology_before.column.active,
        LANDUNIT_SOIL,
    )
    return ColumnWeightTransition(
        old_column_weight_on_gridcell=old,
        new_column_weight_on_gridcell=new,
        area_gained_column=area_gained,
        natural_vegetation_template_column=template,
        any_changes=jnp.any(area_gained != jnp.float64(0.0)),
    )


def update_column_state(
    transition: ColumnWeightTransition,
    column_gridcell: jnp.ndarray,
    values_input: jnp.ndarray,
    values_input_valid: jnp.ndarray,
    has_prognostic_state: jnp.ndarray,
    fractional_area_old: jnp.ndarray,
    fractional_area_new: jnp.ndarray,
    state: jnp.ndarray,
    non_conserved_mass_gridcell: jnp.ndarray,
) -> ColumnStateUpdateResult:

    values = jnp.asarray(state, dtype=jnp.float64)
    inputs = jnp.asarray(values_input, dtype=jnp.float64)
    inputs_valid = jnp.asarray(values_input_valid, dtype=jnp.bool_)
    prognostic = jnp.asarray(has_prognostic_state, dtype=jnp.bool_)
    fraction_old = jnp.asarray(fractional_area_old, dtype=jnp.float64)
    fraction_new = jnp.asarray(fractional_area_new, dtype=jnp.float64)
    gridcell = jnp.asarray(column_gridcell, dtype=jnp.int32)
    column_shape = transition.area_gained_column.shape
    if not (
        values.shape
        == inputs.shape
        == inputs_valid.shape
        == prognostic.shape
        == fraction_old.shape
        == fraction_new.shape
        == gridcell.shape
        == column_shape
    ):
        raise ValueError("all column-state inputs must have identical column shape")
    non_conserved_initial = jnp.asarray(
        non_conserved_mass_gridcell,
        dtype=jnp.float64,
    )
    number_of_gridcells = non_conserved_initial.shape[0]
    if number_of_gridcells == 0 and values.shape[0] != 0:
        raise ValueError("column state requires at least one gridcell")

    change = transition.area_gained_column
    shrinking = change < jnp.float64(0.0)
    growing = change > jnp.float64(0.0)
    invalid_shrinking = shrinking & ~inputs_valid
    inconsistent = prognostic & inputs_valid & (inputs != values)
    total_loss = jnp.zeros(number_of_gridcells, dtype=jnp.float64)
    total_area_lost = jnp.zeros(number_of_gridcells, dtype=jnp.float64)

    def accumulate_loss(index: int, carry):
        loss_gridcell, area_gridcell, non_conserved = carry
        target = gridcell[index]
        area_lost = fortran_multiply(jnp.float64(-1.0), change[index])
        weighted_loss = fortran_multiply(area_lost, inputs[index])
        weighted_loss = fortran_multiply(weighted_loss, fraction_old[index])
        do_loss = shrinking[index] & inputs_valid[index]
        new_loss = fortran_add(loss_gridcell[target], weighted_loss)
        new_area = fortran_add(area_gridcell[target], area_lost)
        new_non_conserved = fortran_add(non_conserved[target], -weighted_loss)
        return (
            loss_gridcell.at[target].set(
                jnp.where(do_loss, new_loss, loss_gridcell[target])
            ),
            area_gridcell.at[target].set(
                jnp.where(do_loss, new_area, area_gridcell[target])
            ),
            non_conserved.at[target].set(
                jnp.where(
                    do_loss & ~prognostic[index],
                    new_non_conserved,
                    non_conserved[target],
                )
            ),
        )

    total_loss, total_area_lost, non_conserved = lax.fori_loop(
        0,
        values.shape[0],
        accumulate_loss,
        (total_loss, total_area_lost, non_conserved_initial),
    )
    has_loss = total_area_lost > jnp.float64(0.0)
    gain_per_area = jnp.where(
        has_loss,
        fortran_divide(
            total_loss,
            jnp.where(has_loss, total_area_lost, jnp.float64(1.0)),
        ),
        jnp.float64(0.0),
    )
    adjustment = jnp.zeros_like(values)

    def distribute_gain(index: int, carry):
        updated_state, updated_non_conserved, updated_adjustment = carry
        target = gridcell[index]
        mass_gained = fortran_multiply(change[index], gain_per_area[target])
        denominator = fortran_multiply(
            transition.new_column_weight_on_gridcell[index],
            fraction_new[index],
        )
        can_update = growing[index] & prognostic[index] & (
            fraction_new[index] != jnp.float64(0.0)
        )
        numerator = fortran_multiply(
            transition.old_column_weight_on_gridcell[index],
            updated_state[index],
        )
        numerator = fortran_multiply(numerator, fraction_old[index])
        numerator = fortran_add(numerator, mass_gained)
        candidate = fortran_divide(
            numerator,
            jnp.where(can_update, denominator, jnp.float64(1.0)),
        )
        new_value = jnp.where(can_update, candidate, updated_state[index])
        new_adjustment = fortran_multiply(new_value, fraction_new[index])
        old_apparent = fortran_multiply(updated_state[index], fraction_old[index])
        new_adjustment = fortran_add(new_adjustment, -old_apparent)
        new_non_conserved = fortran_add(
            updated_non_conserved[target],
            mass_gained,
        )
        return (
            updated_state.at[index].set(new_value),
            updated_non_conserved.at[target].set(
                jnp.where(
                    growing[index] & ~prognostic[index],
                    new_non_conserved,
                    updated_non_conserved[target],
                )
            ),
            updated_adjustment.at[index].set(
                jnp.where(growing[index] & prognostic[index], new_adjustment, 0.0)
            ),
        )

    updated_state, non_conserved, adjustment = lax.fori_loop(
        0,
        values.shape[0],
        distribute_gain,
        (values, non_conserved, adjustment),
    )
    any_changes = jnp.asarray(transition.any_changes, dtype=jnp.bool_)
    return ColumnStateUpdateResult(
        state=jnp.where(any_changes, updated_state, values),
        non_conserved_mass_gridcell=jnp.where(
            any_changes,
            non_conserved,
            non_conserved_initial,
        ),
        adjustment_column=jnp.where(any_changes, adjustment, jnp.zeros_like(values)),
        invalid_shrinking_column=invalid_shrinking,
        inconsistent_input_column=inconsistent,
    )


def update_column_state_no_special_handling(
    transition: ColumnWeightTransition,
    column_gridcell: jnp.ndarray,
    state: jnp.ndarray,
    *,
    number_of_gridcells: int,
    fractional_area_old: jnp.ndarray | None = None,
    fractional_area_new: jnp.ndarray | None = None,
) -> ColumnStateUpdateResult:

    values = jnp.asarray(state, dtype=jnp.float64)
    if (fractional_area_old is None) != (fractional_area_new is None):
        raise ValueError("old and new fractional areas must be provided together")
    old_fraction = (
        jnp.ones_like(values)
        if fractional_area_old is None
        else jnp.asarray(fractional_area_old, dtype=jnp.float64)
    )
    new_fraction = (
        jnp.ones_like(values)
        if fractional_area_new is None
        else jnp.asarray(fractional_area_new, dtype=jnp.float64)
    )
    gridcell = jnp.asarray(column_gridcell, dtype=jnp.int32)
    if number_of_gridcells < 0:
        raise ValueError("number_of_gridcells must be nonnegative")
    return update_column_state(
        transition,
        gridcell,
        values,
        jnp.ones_like(values, dtype=jnp.bool_),
        jnp.ones_like(values, dtype=jnp.bool_),
        old_fraction,
        new_fraction,
        values,
        jnp.zeros(number_of_gridcells, dtype=jnp.float64),
    )


def update_column_state_fill_special_using_natural_vegetation(
    transition: ColumnWeightTransition,
    topology: SubgridTopology,
    state: jnp.ndarray,
    non_conserved_mass_gridcell: jnp.ndarray,
    *,
    fractional_area_old: jnp.ndarray | None = None,
    fractional_area_new: jnp.ndarray | None = None,
) -> ColumnStateUpdateResult:

    values = jnp.asarray(state, dtype=jnp.float64)
    if (fractional_area_old is None) != (fractional_area_new is None):
        raise ValueError("old and new fractional areas must be provided together")
    old_fraction = (
        jnp.ones_like(values)
        if fractional_area_old is None
        else jnp.asarray(fractional_area_old, dtype=jnp.float64)
    )
    new_fraction = (
        jnp.ones_like(values)
        if fractional_area_new is None
        else jnp.asarray(fractional_area_new, dtype=jnp.float64)
    )
    special = landunit_is_special(topology.column.landunit_type)
    template = transition.natural_vegetation_template_column
    template_found = template != jnp.int32(TEMPLATE_NONE_FOUND)
    safe_template = jnp.clip(template, 0, max(values.shape[0] - 1, 0))
    template_values = jnp.where(
        template_found,
        values[safe_template],
        jnp.float64(CLM_SPECIAL_VALUE),
    )
    values_input = jnp.where(special, template_values, values)
    values_valid = (~special) | template_found
    return update_column_state(
        transition,
        topology.column.gridcell,
        values_input,
        values_valid,
        ~special,
        old_fraction,
        new_fraction,
        values,
        non_conserved_mass_gridcell,
    )


def update_column_state_fill_using_fixed_values(
    transition: ColumnWeightTransition,
    topology: SubgridTopology,
    state: jnp.ndarray,
    landunit_values: jnp.ndarray,
    non_conserved_mass_gridcell: jnp.ndarray,
    *,
    fractional_area_old: jnp.ndarray | None = None,
    fractional_area_new: jnp.ndarray | None = None,
) -> ColumnStateUpdateResult:

    values = jnp.asarray(state, dtype=jnp.float64)
    fills = jnp.asarray(landunit_values, dtype=jnp.float64)
    if fills.shape != (9,):
        raise ValueError("landunit_values must contain the nine CLM landunit types")
    if (fractional_area_old is None) != (fractional_area_new is None):
        raise ValueError("old and new fractional areas must be provided together")
    old_fraction = (
        jnp.ones_like(values)
        if fractional_area_old is None
        else jnp.asarray(fractional_area_old, dtype=jnp.float64)
    )
    new_fraction = (
        jnp.ones_like(values)
        if fractional_area_new is None
        else jnp.asarray(fractional_area_new, dtype=jnp.float64)
    )
    column_landunit_type = jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
    selected_fill = fills[column_landunit_type - jnp.int32(1)]
    use_existing = selected_fill == jnp.float64(CLM_SPECIAL_VALUE)
    values_input = jnp.where(use_existing, values, selected_fill)
    return update_column_state(
        transition,
        topology.column.gridcell,
        values_input,
        jnp.ones_like(values, dtype=jnp.bool_),
        use_existing,
        old_fraction,
        new_fraction,
        values,
        non_conserved_mass_gridcell,
    )


def update_column_state_fill_special_using_fixed_value(
    transition: ColumnWeightTransition,
    topology: SubgridTopology,
    state: jnp.ndarray,
    special_value: jnp.ndarray | float,
    non_conserved_mass_gridcell: jnp.ndarray,
    *,
    fractional_area_old: jnp.ndarray | None = None,
    fractional_area_new: jnp.ndarray | None = None,
) -> ColumnStateUpdateResult:

    landunit_types = jnp.arange(1, 10, dtype=jnp.int32)
    fills = jnp.where(
        landunit_is_special(landunit_types),
        jnp.asarray(special_value, dtype=jnp.float64),
        jnp.float64(CLM_SPECIAL_VALUE),
    )
    return update_column_state_fill_using_fixed_values(
        transition,
        topology,
        state,
        fills,
        non_conserved_mass_gridcell,
        fractional_area_old=fractional_area_old,
        fractional_area_new=fractional_area_new,
    )


def patch_old_weight_was_zero(transition: PatchWeightTransition) -> jnp.ndarray:
    return transition.old_patch_weight_on_gridcell == jnp.float64(0.0)


def patch_grew(transition: PatchWeightTransition) -> jnp.ndarray:
    return transition.weight_change > jnp.float64(0.0)


def patch_initiating(transition: PatchWeightTransition) -> jnp.ndarray:
    return patch_old_weight_was_zero(transition) & (
        transition.new_patch_weight_on_gridcell > jnp.float64(0.0)
    )


def first_prior_active_column_by_landunit(
    topology: SubgridTopology,
    prior_column_active: jnp.ndarray,
    landunit_type: int,
) -> jnp.ndarray:

    return _first_prior_active_column(topology, prior_column_active, landunit_type)


def initial_crop_template_columns(
    topology: SubgridTopology,
    prior_column_active: jnp.ndarray,
) -> jnp.ndarray:

    soil = _first_prior_active_column(topology, prior_column_active, LANDUNIT_SOIL)
    crop = _first_prior_active_column(topology, prior_column_active, LANDUNIT_CROP)
    return jnp.where(soil != jnp.int32(TEMPLATE_NONE_FOUND), soil, crop)


def initialize_new_columns(
    topology: SubgridTopology,
    prior_column_active: jnp.ndarray,
    state: NewColumnState,
) -> NewColumnInitializationResult:

    number_of_columns = topology.column.gridcell.shape[0]
    prior_active = jnp.asarray(prior_column_active, dtype=jnp.bool_)
    active = jnp.asarray(topology.column.active, dtype=jnp.bool_)
    landunit_type = jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
    weight = jnp.asarray(topology.column.weight_on_gridcell, dtype=jnp.float64)
    column_shape = (number_of_columns,)
    soil_snow_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS)
    ground_shape = (number_of_columns, NUMBER_OF_GROUND_LEVELS)
    if prior_active.shape != column_shape or active.shape != column_shape:
        raise ValueError("column activity must match column topology")
    if landunit_type.shape != column_shape or weight.shape != column_shape:
        raise ValueError("column landunit types and weights must match topology")

    temperature = jnp.asarray(
        state.soil_snow_temperature_column_level,
        dtype=jnp.float64,
    )
    liquid = jnp.asarray(
        state.soil_snow_liquid_water_column_level,
        dtype=jnp.float64,
    )
    ice = jnp.asarray(
        state.soil_snow_ice_water_column_level,
        dtype=jnp.float64,
    )
    volumetric = jnp.asarray(
        state.volumetric_water_column_ground_level,
        dtype=jnp.float64,
    )
    aquifer = jnp.asarray(state.aquifer_water_column, dtype=jnp.float64)
    if temperature.shape != soil_snow_shape:
        raise ValueError(
            "soil_snow_temperature_column_level must have shape "
            f"{soil_snow_shape}"
        )
    if liquid.shape != soil_snow_shape or ice.shape != soil_snow_shape:
        raise ValueError("soil/snow liquid and ice water must match temperature")
    if volumetric.shape != ground_shape:
        raise ValueError(
            "volumetric_water_column_ground_level must have shape "
            f"{ground_shape}"
        )
    if aquifer.shape != column_shape:
        raise ValueError("aquifer_water_column must match column topology")

    newly_active = active & ~prior_active
    soil = landunit_type == jnp.int32(LANDUNIT_SOIL)
    crop = landunit_type == jnp.int32(LANDUNIT_CROP)
    crop_template = initial_crop_template_columns(topology, prior_active)
    template_found = crop_template != jnp.int32(TEMPLATE_NONE_FOUND)
    copy_from_template = newly_active & crop & template_found
    template = jnp.where(
        newly_active & crop,
        crop_template,
        jnp.int32(TEMPLATE_NONE_FOUND),
    )

    safe_template = jnp.clip(crop_template, 0, max(number_of_columns - 1, 0))
    copy_soil_snow = copy_from_template[:, None] & (
        jnp.arange(NUMBER_OF_SOIL_SNOW_LEVELS, dtype=jnp.int32)[None, :]
        >= jnp.int32(NUMBER_OF_SNOW_LEVELS)
    )
    temperature = jnp.where(
        copy_soil_snow,
        temperature[safe_template],
        temperature,
    )
    liquid = jnp.where(copy_soil_snow, liquid[safe_template], liquid)
    ice = jnp.where(copy_soil_snow, ice[safe_template], ice)
    volumetric = jnp.where(
        copy_from_template[:, None],
        volumetric[safe_template],
        volumetric,
    )
    aquifer = jnp.where(
        copy_from_template,
        aquifer[safe_template],
        aquifer,
    )
    return NewColumnInitializationResult(
        state=NewColumnState(
            soil_snow_temperature_column_level=temperature,
            soil_snow_liquid_water_column_level=liquid,
            soil_snow_ice_water_column_level=ice,
            volumetric_water_column_ground_level=volumetric,
            aquifer_water_column=aquifer,
        ),
        template_column=template,
        newly_active_column=newly_active,
        template_missing_column=newly_active & crop & ~template_found,
        unsupported_new_column=newly_active & ~soil & ~crop,
        invalid_positive_weight_soil_column=(
            newly_active & soil & (weight > jnp.float64(0.0))
        ),
    )


def landunit_is_special(landunit_type: jnp.ndarray) -> jnp.ndarray:

    types = jnp.asarray(landunit_type, dtype=jnp.int32)
    return (types != jnp.int32(LANDUNIT_SOIL)) & (
        types != jnp.int32(LANDUNIT_CROP)
    )


def fill_value_use_existing() -> float:

    return CLM_SPECIAL_VALUE
