
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt


def _reference(values, base, inverse):
    return tuple(
        old if not value.shape[0] else jnp.where(
            (inverse >= 0).reshape((-1, *(1 for _ in value.shape[1:]))),
            value[jnp.maximum(inverse, 0)], old,
        )
        for value, old in zip(values, base, strict=True)
    )


@jax.custom_jvp
def restore_selected_rows(values, base, inverse):
    if (not values or not values[0].size or not inverse.size
            or jax.default_backend() != "gpu"):
        return _reference(values, base, inverse)
    width, block = math.prod(values[0].shape[1:]), 256
    size, nfields = inverse.size * width, len(values)

    def kernel(*refs):
        offset = pl.program_id(0) * block + jnp.arange(block)
        selected = plt.load(
            refs[2 * nfields].at[offset // width], mask=offset < size, other=-1,
        )
        source = jnp.maximum(selected, 0) * width + offset % width
        for i in range(nfields):
            new = plt.load(
                refs[i].at[source], mask=(offset < size) & (selected >= 0), other=0,
            )
            old = plt.load(
                refs[nfields + i].at[offset],
                mask=(offset < size) & (selected < 0), other=0,
            )
            plt.store(
                refs[2 * nfields + 1 + i].at[offset],
                jnp.where(selected >= 0, new, old), mask=offset < size,
            )

    result = pl.pallas_call(
        kernel,
        out_shape=tuple(jax.ShapeDtypeStruct((size,), x.dtype) for x in values),
        grid=(pl.cdiv(size, block),),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
    )(*(x.reshape(-1) for x in values), *(x.reshape(-1) for x in base), inverse)
    return tuple(x.reshape(old.shape) for x, old in zip(result, base, strict=True))


@restore_selected_rows.defjvp
def _derivative(primals, tangents):
    return restore_selected_rows(*primals), jax.jvp(_reference, primals, tangents)[1]
