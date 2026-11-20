
import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap

from clm_jax.physics.converged_iteration import SPATIAL_SOLVE_AXIS, _dynamic_primal
from clm_jax.physics.photosynthesis import solve_hydraulic_photosynthesis_medlyn as _native_hybrid
from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch
from clm_jax.physics.solver_scalarization import _native_tile_trace


def _reference(photosynthesis, inputs, parameters, state, enabled):
    return _native_hybrid(
        photosynthesis, inputs, parameters, state, enabled=enabled,
    )


def fused_hybrid_batch(*arguments, block_size=64):
    return fused_native_batch(_reference, *arguments, block_size=block_size)


@custom_vmap
def _dispatch(*arguments):
    return _reference(*arguments)


@_dispatch.def_vmap
def _dispatch_batch(axis_size, in_batched, *arguments):
    scientific_batched = list(in_batched)
    scientific_batched[0] = scientific_batched[0]._replace(hydraulic_stress=True)
    if not all(jax.tree.leaves(scientific_batched)):
        axes = jax.tree.map(lambda batched: 0 if batched else None, tuple(in_batched))
        result = jax.vmap(_reference, in_axes=axes, axis_name=SPATIAL_SOLVE_AXIS)(*arguments)
        return result, jax.tree.map(lambda _: True, result)
    batched = jax.tree.map(
        lambda value, mapped: value if mapped else jnp.broadcast_to(value, (axis_size, *jnp.shape(value))),
        arguments, tuple(in_batched),
    )
    result = fused_hybrid_batch(*batched)
    return result, jax.tree.map(lambda _: True, result)


def solve_hybrid_gpu(*arguments):
    if (_dynamic_primal.get() and jax.default_backend() == "gpu"
            and not _native_tile_trace.get()):
        token = _native_tile_trace.set(True)
        try:
            return _dispatch(*arguments)
        finally:
            _native_tile_trace.reset(token)
    return _reference(*arguments)
