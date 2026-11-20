
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
)

FIVE_BAND_WIDTH = 5
FIVE_BAND_SUBDIAGONALS = 2
FIVE_BAND_SUPERDIAGONALS = 2
LAPACK_BAND_STORAGE_ROWS = 7
LAPACK_BAND_DIAGONAL_ROW = 4


class BandDiagonalResult(NamedTuple):

    solution: jnp.ndarray
    all_diagonal_pivots: jnp.ndarray


def _validate_band_system_shapes(
    band_matrix: jnp.ndarray,
    right_hand_side: jnp.ndarray,
    top_index: jnp.ndarray,
    bottom_index: jnp.ndarray,
    active: jnp.ndarray,
) -> None:
    if right_hand_side.ndim != 2:
        raise ValueError("right_hand_side must be a column-by-level matrix")
    number_of_columns, number_of_levels = right_hand_side.shape
    if band_matrix.shape != (number_of_columns, FIVE_BAND_WIDTH, number_of_levels):
        raise ValueError(
            "band_matrix must have shape "
            f"{(number_of_columns, FIVE_BAND_WIDTH, number_of_levels)}"
        )
    vector_shape = (number_of_columns,)
    for name, values in {
        "top_index": top_index,
        "bottom_index": bottom_index,
        "active": active,
    }.items():
        if values.shape != vector_shape:
            raise ValueError(f"{name} must have shape {vector_shape}")


def _column_lapack_storage(
    band_matrix: jnp.ndarray,
    right_hand_side: jnp.ndarray,
    top_index: jnp.ndarray,
    bottom_index: jnp.ndarray,
    active: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    number_of_levels = right_hand_side.shape[0]
    local_indices = jnp.arange(number_of_levels, dtype=jnp.int32)
    top = jnp.clip(jnp.asarray(top_index, dtype=jnp.int32), 0, number_of_levels - 1)
    bottom = jnp.clip(
        jnp.asarray(bottom_index, dtype=jnp.int32),
        0,
        number_of_levels - 1,
    )
    number_of_equations = jnp.where(
        active,
        jnp.maximum(jnp.int32(0), bottom - top + jnp.int32(1)),
        jnp.int32(0),
    )

    def gather_band(band: int, shift: int) -> jnp.ndarray:
        indices = jnp.clip(
            top + local_indices + jnp.int32(shift),
            0,
            number_of_levels - 1,
        )
        return jnp.take(band_matrix[band], indices)

    valid = local_indices < number_of_equations
    storage = jnp.zeros(
        (LAPACK_BAND_STORAGE_ROWS, number_of_levels),
        dtype=jnp.float64,
    )
    storage = storage.at[2].set(
        jnp.where(valid & (local_indices >= 2), gather_band(0, -2), 0.0)
    )
    storage = storage.at[3].set(
        jnp.where(valid & (local_indices >= 1), gather_band(1, -1), 0.0)
    )
    storage = storage.at[4].set(
        jnp.where(valid, gather_band(2, 0), jnp.float64(1.0))
    )
    storage = storage.at[5].set(
        jnp.where(
            valid & (local_indices + 1 < number_of_equations),
            gather_band(3, 1),
            0.0,
        )
    )
    storage = storage.at[6].set(
        jnp.where(
            valid & (local_indices + 2 < number_of_equations),
            gather_band(4, 2),
            0.0,
        )
    )
    rhs_indices = jnp.clip(top + local_indices, 0, number_of_levels - 1)
    local_rhs = jnp.where(
        valid,
        jnp.take(right_hand_side, rhs_indices),
        jnp.float64(0.0),
    )
    return storage, local_rhs, number_of_equations


def _dgbtf2_diagonal_pivot_path(
    storage: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    number_of_levels = storage.shape[1]

    def factor_column(
        column: int,
        state: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        current, diagonal_pivots = state
        has_first_subdiagonal = column + 1 < number_of_levels
        has_second_subdiagonal = column + 2 < number_of_levels
        candidates = jnp.stack(
            (
                jnp.abs(current[4, column]),
                jnp.where(
                    has_first_subdiagonal,
                    jnp.abs(current[5, column]),
                    jnp.float64(-1.0),
                ),
                jnp.where(
                    has_second_subdiagonal,
                    jnp.abs(current[6, column]),
                    jnp.float64(-1.0),
                ),
            )
        )
        diagonal_pivots = diagonal_pivots & (
            (jnp.argmax(candidates) == 0) & (current[4, column] != 0.0)
        )
        safe_pivot = jnp.where(
            current[4, column] != 0.0,
            current[4, column],
            jnp.float64(1.0),
        )
        inverse_pivot = fortran_divide(jnp.float64(1.0), safe_pivot)

        for subdiagonal in range(FIVE_BAND_SUBDIAGONALS):
            row = LAPACK_BAND_DIAGONAL_ROW + 1 + subdiagonal
            valid = column + subdiagonal + 1 < number_of_levels
            multiplier = fortran_multiply(inverse_pivot, current[row, column])
            current = current.at[row, column].set(
                jnp.where(valid, multiplier, current[row, column])
            )

        for column_offset in range(1, FIVE_BAND_SUPERDIAGONALS + 1):
            target_column = jnp.minimum(
                column + column_offset,
                number_of_levels - 1,
            )
            pivot_row = LAPACK_BAND_DIAGONAL_ROW - column_offset
            temporary = -current[pivot_row, target_column]
            valid_column = column + column_offset < number_of_levels
            for subdiagonal in range(FIVE_BAND_SUBDIAGONALS):
                target_row = (
                    LAPACK_BAND_DIAGONAL_ROW
                    + 1
                    + subdiagonal
                    - column_offset
                )
                valid_row = column + subdiagonal + 1 < number_of_levels
                updated = fortran_add(
                    current[target_row, target_column],
                    fortran_multiply(
                        current[
                            LAPACK_BAND_DIAGONAL_ROW + 1 + subdiagonal,
                            column,
                        ],
                        temporary,
                    ),
                )
                current = current.at[target_row, target_column].set(
                    jnp.where(
                        valid_column & valid_row,
                        updated,
                        current[target_row, target_column],
                    )
                )
        return current, diagonal_pivots

    return lax.fori_loop(
        0,
        number_of_levels,
        factor_column,
        (storage, jnp.asarray(True)),
    )


def _dgbtrs_diagonal_pivot_path(
    factored_storage: jnp.ndarray,
    right_hand_side: jnp.ndarray,
) -> jnp.ndarray:

    number_of_levels = right_hand_side.shape[0]

    def forward_column(column: int, solution: jnp.ndarray) -> jnp.ndarray:
        temporary = -solution[column]
        for subdiagonal in range(FIVE_BAND_SUBDIAGONALS):
            target = jnp.minimum(
                column + subdiagonal + 1,
                number_of_levels - 1,
            )
            valid = column + subdiagonal + 1 < number_of_levels
            updated = fortran_add(
                solution[target],
                fortran_multiply(
                    factored_storage[
                        LAPACK_BAND_DIAGONAL_ROW + 1 + subdiagonal,
                        column,
                    ],
                    temporary,
                ),
            )
            solution = solution.at[target].set(
                jnp.where(valid, updated, solution[target])
            )
        return solution

    solution = lax.fori_loop(
        0,
        number_of_levels - 1,
        forward_column,
        right_hand_side,
    )

    def backward_column(reverse_index: int, current: jnp.ndarray) -> jnp.ndarray:
        column = number_of_levels - 1 - reverse_index

        def solve_nonzero(solution: jnp.ndarray) -> jnp.ndarray:
            quotient = fortran_divide(
                solution[column],
                factored_storage[LAPACK_BAND_DIAGONAL_ROW, column],
            )
            solution = solution.at[column].set(quotient)
            for superdiagonal in range(1, 5):
                target = jnp.maximum(column - superdiagonal, 0)
                valid = column - superdiagonal >= 0
                updated = fortran_add(
                    solution[target],
                    -fortran_multiply(
                        quotient,
                        factored_storage[
                            LAPACK_BAND_DIAGONAL_ROW - superdiagonal,
                            column,
                        ],
                    ),
                )
                solution = solution.at[target].set(
                    jnp.where(valid, updated, solution[target])
                )
            return solution

        return lax.cond(
            current[column] != 0.0,
            solve_nonzero,
            lambda solution: solution,
            current,
        )

    return lax.fori_loop(
        0,
        number_of_levels,
        backward_column,
        solution,
    )


def _column_solve_five_band_system(
    band_matrix: jnp.ndarray,
    right_hand_side: jnp.ndarray,
    top_index: jnp.ndarray,
    bottom_index: jnp.ndarray,
    active: jnp.ndarray,
) -> BandDiagonalResult:
    number_of_levels = right_hand_side.shape[0]
    storage, local_rhs, number_of_equations = _column_lapack_storage(
        band_matrix,
        right_hand_side,
        top_index,
        bottom_index,
        active,
    )
    factored, diagonal_pivots = _dgbtf2_diagonal_pivot_path(storage)
    local_solution = _dgbtrs_diagonal_pivot_path(factored, local_rhs)

    global_indices = jnp.arange(number_of_levels, dtype=jnp.int32)
    top = jnp.clip(jnp.asarray(top_index, dtype=jnp.int32), 0, number_of_levels - 1)
    local_indices = global_indices - top
    valid = (
        active
        & (local_indices >= 0)
        & (local_indices < number_of_equations)
    )
    aligned = jnp.take(
        local_solution,
        jnp.clip(local_indices, 0, number_of_levels - 1),
    )
    return BandDiagonalResult(
        solution=jnp.where(valid, aligned, jnp.float64(0.0)),
        all_diagonal_pivots=jnp.where(active, diagonal_pivots, True),
    )


def _solve_column_arguments(arguments) -> BandDiagonalResult:
    return _column_solve_five_band_system(*arguments)


def solve_five_band_system(
    band_matrix: jnp.ndarray,
    right_hand_side: jnp.ndarray,
    top_index: jnp.ndarray,
    bottom_index: jnp.ndarray,
    active: jnp.ndarray,
) -> BandDiagonalResult:

    _validate_band_system_shapes(
        band_matrix,
        right_hand_side,
        top_index,
        bottom_index,
        active,
    )
    arguments = (band_matrix, right_hand_side, top_index, bottom_index, active)
    from clm_jax.physics.solver_scalarization import _native_tile_trace

    if jax.default_backend() == "gpu" and not _native_tile_trace.get():
        from clm_jax.physics.band_diagonal_gpu import solve_five_band_system_gpu

        return solve_five_band_system_gpu(*arguments)
    if jax.default_backend() == "cpu" and right_hand_side.shape[0] >= 32:
        return lax.map(_solve_column_arguments, arguments, batch_size=32)
    return jax.vmap(_column_solve_five_band_system)(*arguments)
