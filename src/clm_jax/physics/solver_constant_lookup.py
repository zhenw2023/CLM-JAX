
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core
from jax.interpreters import batching, mlir

constant_lookup_p = core.Primitive("clm_solver_constant_lookup")


def _reference(index, *, data, dtype):
    table = jnp.asarray(np.frombuffer(data, dtype=np.dtype(dtype)))
    return table[jnp.clip(index, 0, table.size - 1)]


constant_lookup_p.def_impl(_reference)
constant_lookup_p.def_abstract_eval(
    lambda index, *, data, dtype: jax.core.ShapedArray(index.shape, np.dtype(dtype)),
)


def _lower(ctx, index, *, data, dtype):
    return mlir.lower_fun(partial(_reference, data=data, dtype=dtype),
                          multiple_results=False)(ctx, index)


mlir.register_lowering(constant_lookup_p, _lower)
batching.primitive_batchers[constant_lookup_p] = (
    lambda values, axes, **params: (constant_lookup_p.bind(*values, **params), axes[0])
)


def constant_lookup(values, index):
    values = np.ascontiguousarray(values)
    if values.ndim != 1 or not 1 <= values.size <= 1024:
        raise ValueError("native solver lookup requires a small nonempty vector")
    if isinstance(index, (int, np.integer)):
        return values[np.clip(index, 0, values.size - 1)]
    index = jnp.clip(index, 0, values.size - 1)
    return constant_lookup_p.bind(index, data=values.tobytes(), dtype=values.dtype.str)
