
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
)


def patch_to_gridcell_mean(
    patch_values: jnp.ndarray,
    patch_gridcell: jnp.ndarray,
    patch_weight_on_gridcell: jnp.ndarray,
    patch_active: jnp.ndarray,
    *,
    number_of_gridcells: int,
    maximum_patches_per_gridcell: int | None = None,
    special_value: float = CLM_SPECIAL_VALUE,
) -> jnp.ndarray:

    values = jnp.asarray(patch_values, dtype=jnp.float64)
    gridcell = jnp.asarray(patch_gridcell, dtype=jnp.int32)
    weight = jnp.asarray(patch_weight_on_gridcell, dtype=jnp.float64)
    active = jnp.asarray(patch_active, dtype=jnp.bool_)
    if not (values.shape == gridcell.shape == weight.shape == active.shape and values.ndim == 1):
        raise ValueError("patch values and gridcell topology must be identically sized vectors")
    if number_of_gridcells < 0:
        raise ValueError("number_of_gridcells must be nonnegative")

    totals = jnp.zeros(number_of_gridcells, dtype=jnp.float64)
    sum_weights = jnp.zeros(number_of_gridcells, dtype=jnp.float64)
    if values.shape[0] == 0:
        return jnp.full(
            number_of_gridcells,
            jnp.float64(special_value),
            dtype=jnp.float64,
        )

    if maximum_patches_per_gridcell is not None:
        if maximum_patches_per_gridcell < 0:
            raise ValueError("maximum_patches_per_gridcell must be nonnegative")
        number_of_patches = values.shape[0]
        patch_index = jnp.arange(number_of_patches, dtype=jnp.int32)
        first_patch = jnp.full(
            number_of_gridcells,
            number_of_patches,
            dtype=jnp.int32,
        )
        first_patch = first_patch.at[gridcell].min(patch_index)
        gridcell_index = jnp.arange(number_of_gridcells, dtype=jnp.int32)

        def accumulate_offset(
            offset: int,
            carry: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            gridcell_totals, gridcell_weights = carry
            unbounded_patch_index = first_patch + jnp.int32(offset)
            bounded_patch_index = jnp.clip(
                unbounded_patch_index,
                0,
                number_of_patches - 1,
            )
            belongs_to_gridcell = (unbounded_patch_index < number_of_patches) & (
                gridcell[bounded_patch_index] == gridcell_index
            )
            selected_weight = weight[bounded_patch_index]
            selected_value = values[bounded_patch_index]
            contributes = (
                belongs_to_gridcell
                & active[bounded_patch_index]
                & (selected_weight != jnp.float64(0.0))
                & (selected_value != jnp.float64(special_value))
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
                jnp.where(
                    contributes,
                    fortran_add(gridcell_totals, contribution),
                    gridcell_totals,
                ),
                jnp.where(
                    contributes,
                    fortran_add(gridcell_weights, contribution_weight),
                    gridcell_weights,
                ),
            )

        totals, sum_weights = lax.fori_loop(
            0,
            maximum_patches_per_gridcell,
            accumulate_offset,
            (totals, sum_weights),
        )
    else:

        def accumulate_patch(
            index: int,
            carry: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            gridcell_totals, gridcell_weights = carry
            gridcell_index = gridcell[index]
            selected_weight = weight[index]
            selected_value = values[index]
            contributes = (
                active[index]
                & (selected_weight != jnp.float64(0.0))
                & (selected_value != jnp.float64(special_value))
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
            updated_total = fortran_add(
                gridcell_totals[gridcell_index],
                contribution,
            )
            updated_weight = fortran_add(
                gridcell_weights[gridcell_index],
                contribution_weight,
            )
            return (
                gridcell_totals.at[gridcell_index].set(
                    jnp.where(
                        contributes,
                        updated_total,
                        gridcell_totals[gridcell_index],
                    )
                ),
                gridcell_weights.at[gridcell_index].set(
                    jnp.where(
                        contributes,
                        updated_weight,
                        gridcell_weights[gridcell_index],
                    )
                ),
            )

        totals, sum_weights = lax.fori_loop(
            0,
            values.shape[0],
            accumulate_patch,
            (totals, sum_weights),
        )

    has_contributor = sum_weights != jnp.float64(0.0)
    safe_weights = jnp.where(has_contributor, sum_weights, jnp.float64(1.0))
    return jnp.where(
        has_contributor,
        fortran_divide(totals, safe_weights),
        jnp.float64(special_value),
    )


def column_to_gridcell_mean(
    column_values: jnp.ndarray,
    column_gridcell: jnp.ndarray,
    column_weight_on_gridcell: jnp.ndarray,
    column_active: jnp.ndarray,
    *,
    number_of_gridcells: int,
    maximum_columns_per_gridcell: int | None = None,
    special_value: float = CLM_SPECIAL_VALUE,
) -> jnp.ndarray:

    values = jnp.asarray(column_values, dtype=jnp.float64)
    gridcell = jnp.asarray(column_gridcell, dtype=jnp.int32)
    weight = jnp.asarray(column_weight_on_gridcell, dtype=jnp.float64)
    active = jnp.asarray(column_active, dtype=jnp.bool_)
    if not (values.shape == gridcell.shape == weight.shape == active.shape and values.ndim == 1):
        raise ValueError("column values and gridcell topology must be identically sized vectors")
    if number_of_gridcells < 0:
        raise ValueError("number_of_gridcells must be nonnegative")

    totals = jnp.zeros(number_of_gridcells, dtype=jnp.float64)
    sum_weights = jnp.zeros(number_of_gridcells, dtype=jnp.float64)
    number_of_columns = values.shape[0]
    if number_of_columns == 0:
        return jnp.full(
            number_of_gridcells,
            jnp.float64(special_value),
            dtype=jnp.float64,
        )

    def contribution_at(index: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        selected_weight = weight[index]
        selected_value = values[index]
        contributes = (
            active[index]
            & (selected_weight != jnp.float64(0.0))
            & (selected_value != jnp.float64(special_value))
        )
        return (
            jnp.where(
                contributes,
                fortran_multiply(selected_value, selected_weight),
                jnp.float64(0.0),
            ),
            jnp.where(contributes, selected_weight, jnp.float64(0.0)),
        )

    if maximum_columns_per_gridcell is not None:
        if maximum_columns_per_gridcell < 0:
            raise ValueError("maximum_columns_per_gridcell must be nonnegative")
        if jax.default_backend() == "gpu" and number_of_columns >= 4096:
            from clm_jax.physics.ordered_gpu_reduction import ordered_sums

            totals, sum_weights = ordered_sums(
                values, weight, active & (weight != 0.0) & (values != special_value),
                gridcell, number_of_gridcells, maximum_columns_per_gridcell,
            )
            has_contributor = sum_weights != 0.0
            return jnp.where(
                has_contributor,
                fortran_divide(totals, jnp.where(has_contributor, sum_weights, 1.0)),
                jnp.float64(special_value),
            )
        column_index = jnp.arange(number_of_columns, dtype=jnp.int32)
        first_column = jnp.full(
            number_of_gridcells,
            number_of_columns,
            dtype=jnp.int32,
        )
        first_column = first_column.at[gridcell].min(column_index)
        gridcell_index = jnp.arange(number_of_gridcells, dtype=jnp.int32)

        def accumulate_offset(
            offset: int,
            carry: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            gridcell_totals, gridcell_weights = carry
            unbounded_column = first_column + jnp.int32(offset)
            bounded_column = jnp.clip(unbounded_column, 0, number_of_columns - 1)
            contribution, contribution_weight = contribution_at(bounded_column)
            belongs = (unbounded_column < number_of_columns) & (
                gridcell[bounded_column] == gridcell_index
            )
            contribution = jnp.where(belongs, contribution, jnp.float64(0.0))
            contribution_weight = jnp.where(
                belongs,
                contribution_weight,
                jnp.float64(0.0),
            )
            return (
                fortran_add(gridcell_totals, contribution),
                fortran_add(gridcell_weights, contribution_weight),
            )

        totals, sum_weights = lax.fori_loop(
            0,
            maximum_columns_per_gridcell,
            accumulate_offset,
            (totals, sum_weights),
        )
    else:

        def accumulate_column(
            index: int,
            carry: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            gridcell_totals, gridcell_weights = carry
            target = gridcell[index]
            contribution, contribution_weight = contribution_at(index)
            return (
                gridcell_totals.at[target].set(fortran_add(gridcell_totals[target], contribution)),
                gridcell_weights.at[target].set(
                    fortran_add(gridcell_weights[target], contribution_weight)
                ),
            )

        totals, sum_weights = lax.fori_loop(
            0,
            number_of_columns,
            accumulate_column,
            (totals, sum_weights),
        )

    has_contributor = sum_weights != jnp.float64(0.0)
    safe_weights = jnp.where(has_contributor, sum_weights, jnp.float64(1.0))
    return jnp.where(
        has_contributor,
        fortran_divide(totals, safe_weights),
        jnp.float64(special_value),
    )


def patch_to_column_average(
    patch_values: jnp.ndarray,
    patch_column: jnp.ndarray,
    patch_weight_on_column: jnp.ndarray,
    patch_active: jnp.ndarray,
    column_update_mask: jnp.ndarray,
    *,
    maximum_patches_per_column: int | None = None,
) -> jnp.ndarray:

    values = jnp.asarray(patch_values, dtype=jnp.float64)
    column = jnp.asarray(patch_column, dtype=jnp.int32)
    weight = jnp.asarray(patch_weight_on_column, dtype=jnp.float64)
    active = jnp.asarray(patch_active, dtype=jnp.bool_)
    update = jnp.asarray(column_update_mask, dtype=jnp.bool_)
    result = jnp.zeros(update.shape, dtype=jnp.float64)
    if values.shape[0] == 0:
        return result

    if maximum_patches_per_column is not None:
        if jax.default_backend() == "gpu" and values.size >= 4096:
            from clm_jax.physics.ordered_gpu_reduction import ordered_sums

            totals, _ = ordered_sums(
                values, weight, active & update[column], column, update.size,
                maximum_patches_per_column,
            )
            return totals
        number_of_patches = values.shape[0]
        patch_index = jnp.arange(number_of_patches, dtype=jnp.int32)
        first_patch = jnp.full(update.shape, number_of_patches, dtype=jnp.int32)
        first_patch = first_patch.at[column].min(patch_index)
        column_index = jnp.arange(update.shape[0], dtype=jnp.int32)

        def accumulate_offset(
            offset: int,
            column_values: jnp.ndarray,
        ) -> jnp.ndarray:
            unbounded_patch_index = first_patch + jnp.int32(offset)
            bounded_patch_index = jnp.clip(
                unbounded_patch_index,
                0,
                number_of_patches - 1,
            )
            belongs_to_column = (unbounded_patch_index < number_of_patches) & (
                column[bounded_patch_index] == column_index
            )
            contribution = jnp.where(
                belongs_to_column & active[bounded_patch_index] & update,
                fortran_multiply(
                    values[bounded_patch_index],
                    weight[bounded_patch_index],
                ),
                jnp.float64(0.0),
            )
            return fortran_add(column_values, contribution)

        stop = jnp.max(jnp.bincount(column, length=update.shape[0]), initial=0)

        def active_offset(offset, carry):
            return lax.cond(
                offset < stop,
                lambda value: accumulate_offset(offset, value),
                lambda value: value,
                carry,
            )

        return lax.fori_loop(0, maximum_patches_per_column, active_offset, result)

    def accumulate(patch_index: int, column_values: jnp.ndarray) -> jnp.ndarray:
        column_index = column[patch_index]
        contribution = jnp.where(
            active[patch_index] & update[column_index],
            fortran_multiply(values[patch_index], weight[patch_index]),
            jnp.float64(0.0),
        )
        updated = fortran_add(column_values[column_index], contribution)
        return column_values.at[column_index].set(updated)

    return lax.fori_loop(0, values.shape[0], accumulate, result)


def patch_to_column_mean(
    patch_values: jnp.ndarray,
    patch_column: jnp.ndarray,
    patch_weight_on_column: jnp.ndarray,
    patch_active: jnp.ndarray,
    *,
    number_of_columns: int,
    maximum_patches_per_column: int | None = None,
    special_value: float = CLM_SPECIAL_VALUE,
) -> jnp.ndarray:

    values = jnp.asarray(patch_values, dtype=jnp.float64)
    column = jnp.asarray(patch_column, dtype=jnp.int32)
    weight = jnp.asarray(patch_weight_on_column, dtype=jnp.float64)
    active = jnp.asarray(patch_active, dtype=jnp.bool_)
    if not (values.shape == column.shape == weight.shape == active.shape and values.ndim == 1):
        raise ValueError("patch values and topology fields must be identically sized vectors")
    if number_of_columns < 0:
        raise ValueError("number_of_columns must be nonnegative")

    totals = jnp.zeros(number_of_columns, dtype=jnp.float64)
    sum_weights = jnp.zeros(number_of_columns, dtype=jnp.float64)
    if values.shape[0] == 0:
        return jnp.full(
            number_of_columns,
            jnp.float64(special_value),
            dtype=jnp.float64,
        )

    if maximum_patches_per_column is not None:
        if maximum_patches_per_column < 0:
            raise ValueError("maximum_patches_per_column must be nonnegative")
        if jax.default_backend() == "gpu" and values.size >= 4096:
            from clm_jax.physics.ordered_gpu_reduction import ordered_sums

            totals, sum_weights = ordered_sums(
                values, weight, active & (weight != 0.0) & (values != special_value),
                column, number_of_columns, maximum_patches_per_column,
            )
            has_contributor = sum_weights != 0.0
            return jnp.where(
                has_contributor,
                totals / jnp.where(has_contributor, sum_weights, 1.0),
                jnp.float64(special_value),
            )
        number_of_patches = values.shape[0]
        patch_index = jnp.arange(number_of_patches, dtype=jnp.int32)
        first_patch = jnp.full(
            number_of_columns,
            number_of_patches,
            dtype=jnp.int32,
        )
        first_patch = first_patch.at[column].min(patch_index)
        column_index = jnp.arange(number_of_columns, dtype=jnp.int32)

        def accumulate_offset(
            offset: int,
            carry: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            column_totals, column_weights = carry
            unbounded_patch_index = first_patch + jnp.int32(offset)
            bounded_patch_index = jnp.clip(
                unbounded_patch_index,
                0,
                number_of_patches - 1,
            )
            belongs_to_column = (unbounded_patch_index < number_of_patches) & (
                column[bounded_patch_index] == column_index
            )
            selected_weight = weight[bounded_patch_index]
            selected_value = values[bounded_patch_index]
            contributes = (
                belongs_to_column
                & active[bounded_patch_index]
                & (selected_weight != jnp.float64(0.0))
                & (selected_value != jnp.float64(special_value))
            )
            contribution_weight = jnp.where(
                contributes,
                selected_weight,
                jnp.float64(0.0),
            )
            contribution_value = jnp.where(
                contributes,
                selected_value * selected_weight,
                jnp.float64(0.0),
            )
            return (
                column_totals + contribution_value,
                column_weights + contribution_weight,
            )

        totals, sum_weights = lax.fori_loop(
            0,
            maximum_patches_per_column,
            accumulate_offset,
            (totals, sum_weights),
        )
    else:

        def accumulate_patch(
            index: int,
            carry: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            column_totals, column_weights = carry
            column_index = column[index]
            selected_weight = weight[index]
            selected_value = values[index]
            contributes = (
                active[index]
                & (selected_weight != jnp.float64(0.0))
                & (selected_value != jnp.float64(special_value))
            )
            contribution_weight = jnp.where(
                contributes,
                selected_weight,
                jnp.float64(0.0),
            )
            contribution_value = jnp.where(
                contributes,
                selected_value * selected_weight,
                jnp.float64(0.0),
            )
            return (
                column_totals.at[column_index].add(contribution_value),
                column_weights.at[column_index].add(contribution_weight),
            )

        totals, sum_weights = lax.fori_loop(
            0,
            values.shape[0],
            accumulate_patch,
            (totals, sum_weights),
        )

    has_contributor = sum_weights != jnp.float64(0.0)
    safe_weights = jnp.where(has_contributor, sum_weights, jnp.float64(1.0))
    return jnp.where(
        has_contributor,
        totals / safe_weights,
        jnp.float64(special_value),
    )
