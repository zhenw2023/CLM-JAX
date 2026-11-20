
import math

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.physics.numerics import (
    fortran_divide,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)


def ordered_thomas_solve(
    lower: jnp.ndarray,
    diagonal: jnp.ndarray,
    upper: jnp.ndarray,
    right_hand_side: jnp.ndarray,
    initial_solution: jnp.ndarray,
) -> jnp.ndarray:

    del initial_solution
    beta = diagonal[..., 0]
    first_solution = fortran_divide(right_hand_side[..., 0], beta)

    def forward_step(carry, row):
        previous_solution, previous_beta = carry
        lower_value, diagonal_value, previous_upper, rhs = row
        gamma_value = fortran_divide(previous_upper, previous_beta)
        next_beta = fortran_add(
            diagonal_value,
            -fortran_multiply(lower_value, gamma_value),
        )
        numerator = fortran_add(
            rhs,
            -fortran_multiply(lower_value, previous_solution),
        )
        next_solution = fortran_divide(numerator, next_beta)
        return (next_solution, next_beta), (next_solution, gamma_value)

    (last_solution, _), (forward, gamma) = lax.scan(
        forward_step,
        (first_solution, beta),
        tuple(
            jnp.moveaxis(value, -1, 0)
            for value in (
                lower[..., 1:], diagonal[..., 1:], upper[..., :-1], right_hand_side[..., 1:]
            )
        ),
    )
    forward = jnp.concatenate((first_solution[None, ...], forward), axis=0)

    def backward_step(next_solution, row):
        current_solution, next_gamma = row
        solution = fortran_add(
            current_solution, -fortran_multiply(next_gamma, next_solution)
        )
        return solution, solution

    _, backward = lax.scan(
        backward_step,
        last_solution,
        (forward[:-1], gamma),
        reverse=True,
    )
    return jnp.moveaxis(jnp.concatenate((backward, last_solution[None, ...]), axis=0), 0, -1)



def _column(lower, diagonal, upper, rhs):
    return ordered_thomas_solve(
        lower[None, :], diagonal[None, :], upper[None, :], rhs[None, :], rhs[None, :],
    )[0]


@jax.custom_jvp
def tiled_thomas_solve(lower, diagonal, upper, rhs):
    from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch
    return fused_native_batch(
        _column, lower, diagonal, upper, rhs, block_size=64, hardware_rounding=True,
    )


@tiled_thomas_solve.defjvp
def _tiled_jvp(primals, tangents):
    def native(lower, diagonal, upper, rhs):
        return ordered_thomas_solve(lower, diagonal, upper, rhs, rhs)
    return tiled_thomas_solve(*primals), jax.jvp(native, primals, tangents)[1]


@jax.custom_jvp
def tiled_broadcast_thomas_solve(lower, diagonal, upper, rhs):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt
    from clm_jax.physics.plant_hydraulics_gpu import _tile_program

    fields = (lower, diagonal, upper, rhs)
    dimensions = rhs.shape[:-1]
    count, levels, block = math.prod(dimensions), rhs.shape[-1], 128
    if count == 0:
        return ordered_thomas_solve(lower, diagonal, upper, rhs, rhs)
    leaves, tree = jax.tree.flatten(fields)
    shapes = tuple(value.shape[:-1] for value in fields)
    for shape in shapes:
        if jnp.broadcast_shapes(shape, dimensions) != dimensions:
            raise ValueError("matrix batch dimensions must broadcast to the RHS")
    signatures = tuple(((levels,), value.dtype) for value in leaves)
    program, _, _ = _tile_program(
        _column, tree, signatures, block, hardware_rounding=True,
    )
    strides = tuple(math.prod(dimensions[i+1:]) for i in range(len(dimensions)))

    def kernel(*refs):
        row = pl.program_id(0) * block + jnp.arange(block)
        safe = jnp.minimum(row, count - 1)
        values = []
        for ref, shape in zip(refs[:4], shapes, strict=True):
            shape = (1,) * (len(dimensions) - len(shape)) + shape
            index = jnp.zeros((block,), jnp.int32)
            for size, stride in zip(shape, strides, strict=True):
                index = index * size
                if size != 1:
                    index = index + (safe // stride) % size
            for level in range(levels):
                values.append(plt.load(ref.at[level, index]))
        result = jax.core.eval_jaxpr(program.jaxpr, (), *values)
        for level, value in enumerate(result):
            plt.store(refs[4].at[level, row], value, mask=row < count)

    result = pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((levels, count), rhs.dtype),
        grid=(pl.cdiv(count, block),),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
    )(*(jnp.moveaxis(value, -1, 0).reshape(levels, -1) for value in fields))
    return jnp.moveaxis(result.reshape((levels, *dimensions)), 0, -1)


@tiled_broadcast_thomas_solve.defjvp
def _broadcast_tiled_jvp(primals, tangents):
    def reference(lower, diagonal, upper, rhs):
        return ordered_thomas_solve(lower, diagonal, upper, rhs, rhs)
    return tiled_broadcast_thomas_solve(*primals), jax.jvp(reference, primals, tangents)[1]


def transport_thomas_solve(lower, diagonal, upper, rhs, initial):
    if jax.default_backend() == "gpu":
        return tiled_broadcast_thomas_solve(lower, diagonal, upper, rhs)
    return ordered_thomas_solve(lower, diagonal, upper, rhs, initial)
