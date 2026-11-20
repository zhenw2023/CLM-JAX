
from contextvars import ContextVar

import jax.numpy as jnp
from jax.extend import core
from jax.interpreters import batching

native_exp_residual = ContextVar("clm_native_exp_product_residual", default=False)
fma_p = core.Primitive("clm_solver_exp_fma")
fma_p.def_abstract_eval(lambda left, right, addend: left)


def _batch_fma(values, axes):
    if all(axis is None for axis in axes):
        return fma_p.bind(*values), None
    size = next(value.shape[axis] for value, axis in zip(values, axes, strict=True)
                if axis is not None)
    values = [batching.bdim_at_front(value, axis, size)
              for value, axis in zip(values, axes, strict=True)]
    rank = max(value.ndim for value in values)
    values = [
        value.reshape((size, *(1,) * (rank - value.ndim), *value.shape[1:]))
        for value in values
    ]
    return fma_p.bind(*jnp.broadcast_arrays(*values)), 0


batching.primitive_batchers[fma_p] = _batch_fma


def product_residual(left, right, product):
    left, right, product = jnp.broadcast_arrays(left, right, product)
    residual = fma_p.bind(left, right, -product)
    return jnp.where(residual == 0., 0., residual)
