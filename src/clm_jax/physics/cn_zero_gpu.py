
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt


def _reference(values, mask):
    return tuple(
        jnp.where(mask.reshape((mask.size,) + (1,) * (value.ndim - 1)), 0.0, value)
        for value in values
    )


def _copy_fields(values, mask):
    shape = values[0].shape
    size, width, block = math.prod(shape), math.prod(shape[1:]), 256
    nfields = len(values)

    def kernel(*refs):
        index = pl.program_id(0) * block + jnp.arange(block)
        enabled = plt.load(refs[nfields].at[index // width], mask=index < size, other=False)
        for source, destination in zip(refs[:nfields], refs[nfields + 1:], strict=True):
            value = plt.load(source.at[index], mask=(index < size) & ~enabled, other=0.0)
            plt.store(destination.at[index], value, mask=index < size)

    result = pl.pallas_call(
        kernel,
        out_shape=tuple(jax.ShapeDtypeStruct((size,), value.dtype) for value in values),
        grid=(pl.cdiv(size, block),),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
    )(*(value.reshape(-1) for value in values), mask)
    return tuple(value.reshape(shape) for value in result)


@jax.custom_jvp
def masked_reset(values, mask):
    if not values or not values[0].size:
        return _reference(values, mask)
    outputs = []
    for start in range(0, len(values), 8):
        outputs.extend(_copy_fields(values[start:start + 8], mask))
    return tuple(outputs)


@masked_reset.defjvp
def _masked_reset_jvp(primals, tangents):
    return masked_reset(*primals), jax.jvp(_reference, primals, tangents)[1]
