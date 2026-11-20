
import jax
import jax.numpy as jnp

from clm_jax.physics import cn_fun_allocation as fun
from clm_jax.physics.cn_fun import (
    _retranslocation_early_exit,
    _retranslocation_fused_gpu,
)
from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch
from clm_jax.physics.solver_fma import native_exp_residual


def _patch(*arguments):
    contexts = (
        (fun._materialized_fun, True),
        (_retranslocation_fused_gpu, False),
        (_retranslocation_early_exit, True),
        (native_exp_residual, True),
    )
    tokens = [(context, context.set(value)) for context, value in contexts]
    try:
        return fun._allocate_fun_patch(*arguments)
    finally:
        for context, token in reversed(tokens):
            context.reset(token)


def sorted_fun_allocation(inputs, state, parameters):
    key = (
        inputs.soil_patch.astype(jnp.int32) * 512
        + inputs.vegetation_type_patch.astype(jnp.int32) * 2
        + (inputs.available_carbon_patch > 0).astype(jnp.int32)
    )
    order = jnp.argsort(key, stable=True)
    packed = inputs._replace(**{
        name: getattr(inputs, name)[order]
        for name in fun.FunAllocationInputs._fields[:21]
    })
    packed_state = jax.tree.map(lambda value: value[order], state)
    operands, scalars = fun._prepare_fun_patches(packed, packed_state, parameters)
    allocated = fused_native_batch(
        _patch, *operands, scalars, shared_argnums=(5,), block_size=32,
        fold_literals=True, direct_io=True, guard_branches=True,
    )
    result = fun._assemble_fun_allocation(
        allocated, jnp.sum(operands[0], dtype=jnp.int32),
    )

    def restore(value):
        if value.ndim == 0:
            return value
        return jnp.empty_like(value).at[order].set(value, unique_indices=True)

    return jax.tree.map(restore, result)
