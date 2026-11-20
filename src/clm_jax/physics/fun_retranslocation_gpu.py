
import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap

from clm_jax.physics.cn_fun import (
    _retranslocation_early_exit, _retranslocation_fused_gpu, fun_retranslocation,
)
from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch


def _reference(*arguments):
    token = _retranslocation_fused_gpu.set(False)
    try:
        return fun_retranslocation(*arguments)
    finally:
        _retranslocation_fused_gpu.reset(token)


def _converged_reference(*arguments):
    token = _retranslocation_early_exit.set(True)
    try:
        return _reference(*arguments)
    finally:
        _retranslocation_early_exit.reset(token)


@jax.custom_jvp
def fused_retranslocation_batch(*arguments):
    if jax.default_backend() != "gpu":
        return jax.vmap(_reference)(*arguments)
    return fused_native_batch(_converged_reference, *arguments)


@fused_retranslocation_batch.defjvp
def _derivative(primals, tangents):
    return jax.jvp(jax.vmap(_reference), primals, tangents)


def retranslocation_batch(*arguments):
    values = jnp.broadcast_arrays(*(jnp.asarray(value, jnp.float64) for value in arguments))
    if values[0].ndim != 1:
        raise ValueError("FUN tile candidate requires one spatial batch axis")
    return fused_retranslocation_batch(*values)


@custom_vmap
def _scalar_dispatch(*arguments):
    return _reference(*arguments)


@_scalar_dispatch.def_vmap
def _batch_dispatch(axis_size, in_batched, *arguments):
    values = tuple(value if mapped else jnp.broadcast_to(value, (axis_size, *jnp.shape(value)))
                   for mapped, value in zip(in_batched, arguments, strict=True))
    result = fused_native_batch(_converged_reference, *values)
    return result, jax.tree.map(lambda _: True, result)


@jax.custom_jvp
def scalar_retranslocation(*arguments):
    return _scalar_dispatch(*arguments)


@scalar_retranslocation.defjvp
def _scalar_derivative(primals, tangents):
    return jax.jvp(_reference, primals, tangents)


def fused_fun_allocation(inputs, state, parameters):
    from clm_jax.physics.cn_fun_allocation import carbon_use_efficiency_nitrogen_uptake
    token = _retranslocation_fused_gpu.set(True)
    try:
        return carbon_use_efficiency_nitrogen_uptake(inputs, state, parameters)
    finally:
        _retranslocation_fused_gpu.reset(token)
