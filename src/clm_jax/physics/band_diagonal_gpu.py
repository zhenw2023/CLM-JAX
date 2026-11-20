
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from clm_jax.physics.band_diagonal import (
    BandDiagonalResult,
    _column_lapack_storage,
    _column_solve_five_band_system,
    _validate_band_system_shapes,
)
from clm_jax.physics.numerics import fortran_add, fortran_divide, fortran_multiply


def _reference(matrix, rhs, top, bottom, active):
    return jax.vmap(_column_solve_five_band_system)(matrix, rhs, top, bottom, active)


def _kernel_solve(matrix, rhs, top, bottom, active, *, interpret=False):
    _validate_band_system_shapes(matrix, rhs, top, bottom, active)
    count, levels = rhs.shape
    if count == 0:
        return BandDiagonalResult(rhs, jnp.ones((0,), jnp.bool_))
    storage, local_rhs, size = jax.vmap(_column_lapack_storage)(
        matrix, rhs, top, bottom, active,
    )
    block = 128

    def instruction(name, left, right):
        if interpret:
            return {"add": fortran_add, "mul": fortran_multiply,
                    "div": fortran_divide}[name](left, right)
        value = plt.elementwise_inline_asm(
            f"{name}.rn.f64 $0, $1, $2;", args=[left, right],
            constraints="=d,d,d", pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct(left.shape, jnp.float64)],
        )[0]
        return jnp.where(value == 0.0, 0.0, value) if name == "add" else value

    def kernel(ab_ref, rhs_ref, work_ref, solution_ref, pivot_ref):
        column = pl.program_id(0) * block + jnp.arange(block)
        valid = column < count

        def load_ab(row, level):
            return plt.load(work_ref.at[row, level, column], mask=valid, other=0.0)

        def store_ab(row, level, value):
            plt.store(work_ref.at[row, level, column], value, mask=valid)

        def load_rhs(level):
            return plt.load(solution_ref.at[level, column], mask=valid, other=0.0)

        def store_rhs(level, value):
            plt.store(solution_ref.at[level, column], value, mask=valid)

        def initialize(level, _):
            for row in range(7):
                store_ab(row, level, plt.load(
                    ab_ref.at[row, level, column], mask=valid, other=0.0,
                ))
            store_rhs(level, plt.load(rhs_ref.at[level, column], mask=valid, other=0.0))
            return None

        lax.fori_loop(0, levels, initialize, None)

        def factor(level, pivots):
            diagonal = load_ab(4, level)
            first = jnp.where(level + 1 < levels, jnp.abs(load_ab(5, level)), -1.0)
            second = jnp.where(level + 2 < levels, jnp.abs(load_ab(6, level)), -1.0)
            diagonal_selected = jnp.isnan(diagonal) | (
                (jnp.abs(diagonal) >= first) & (jnp.abs(diagonal) >= second)
            )
            pivots = pivots & (diagonal != 0.0) & diagonal_selected
            inverse = instruction(
                "div", jnp.ones((block,), jnp.float64),
                jnp.where(diagonal != 0.0, diagonal, 1.0),
            )
            for sub in range(2):
                original = load_ab(5 + sub, level)
                multiplier = instruction("mul", inverse, original)
                store_ab(5 + sub, level, jnp.where(
                    level + sub + 1 < levels, multiplier, original,
                ))
            for offset in range(1, 3):
                target = jnp.minimum(level + offset, levels - 1)
                temporary = -load_ab(4 - offset, target)
                for sub in range(2):
                    row = 5 + sub - offset
                    original = load_ab(row, target)
                    updated = instruction(
                        "add", original, instruction("mul", load_ab(5 + sub, level), temporary),
                    )
                    enabled = (level + offset < levels) & (level + sub + 1 < levels)
                    store_ab(row, target, jnp.where(enabled, updated, original))
            return pivots

        pivots = lax.fori_loop(0, levels, factor, jnp.ones((block,), jnp.bool_))

        def forward(level, _):
            temporary = -load_rhs(level)
            for sub in range(2):
                target = jnp.minimum(level + sub + 1, levels - 1)
                original = load_rhs(target)
                updated = instruction(
                    "add", original, instruction("mul", load_ab(5 + sub, level), temporary),
                )
                store_rhs(target, jnp.where(level + sub + 1 < levels, updated, original))
            return None

        lax.fori_loop(0, levels - 1, forward, None)

        def backward(reverse, _):
            level = levels - 1 - reverse
            original = load_rhs(level)
            nonzero = original != 0.0
            quotient = instruction("div", original, load_ab(4, level))
            store_rhs(level, jnp.where(nonzero, quotient, original))
            for offset in range(1, 5):
                target = jnp.maximum(level - offset, 0)
                before = load_rhs(target)
                updated = instruction(
                    "add", before, -instruction("mul", quotient, load_ab(4 - offset, level)),
                )
                store_rhs(target, jnp.where(nonzero & (level >= offset), updated, before))
            return None

        lax.fori_loop(0, levels, backward, None)
        plt.store(pivot_ref.at[column], pivots, mask=valid)

    _, solved, pivots = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((7, levels, count), jnp.float64),
                   jax.ShapeDtypeStruct((levels, count), jnp.float64),
                   jax.ShapeDtypeStruct((count,), jnp.bool_)),
        grid=(pl.cdiv(count, block),), interpret=interpret,
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
    )(jnp.transpose(storage, (1, 2, 0)), local_rhs.T)
    local_index = jnp.arange(levels)[None, :] - jnp.clip(top, 0, levels - 1)[:, None]
    aligned = jnp.take_along_axis(solved.T, jnp.clip(local_index, 0, levels - 1), axis=1)
    enabled = active[:, None] & (local_index >= 0) & (local_index < size[:, None])
    return BandDiagonalResult(
        jnp.where(enabled, aligned, 0.0), jnp.where(active, pivots, True),
    )


@jax.custom_jvp
def solve_five_band_system_gpu(matrix, rhs, top, bottom, active):
    return _kernel_solve(matrix, rhs, top, bottom, active)


@solve_five_band_system_gpu.defjvp
def _solve_jvp(primals, tangents):
    matrix, rhs, top, bottom, active = primals
    return jax.jvp(
        lambda a, b: _reference(a, b, top, bottom, active),
        (matrix, rhs), (tangents[0], tangents[1]),
    )
