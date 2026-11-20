
import jax
import jax.numpy as jnp

from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch
from clm_jax.physics.solver_fma import native_exp_residual
from clm_jax.physics.native_tile_call import transpose_native_storage


def _point(inputs, coupling, initial, parameters):
    from clm_jax.physics.canopy_fluxes import _canopy_hydraulic_energy_reference

    inputs, coupling, initial = jax.tree.map(
        lambda value: value[None],
        (inputs, coupling, initial),
    )
    token = native_exp_residual.set(True)
    try:
        result = _canopy_hydraulic_energy_reference(inputs, coupling, initial, parameters)
    finally:
        native_exp_residual.reset(token)
    return jax.tree.map(lambda value: value[0], result)


def canopy_hydraulic_energy_gpu(inputs, coupling, initial, parameters, *, block_size=32,
                                group_leafless=True):
    count = inputs.active_patch.size
    c3 = coupling.biochemistry.c3_pathway.reshape((count, -1))[:, 0]
    daylight = jnp.any(coupling.biochemistry.absorbed_par.reshape((count, -1)) > 0, axis=1)
    branch = (
        inputs.active_patch.astype(jnp.int32) * 4
        + daylight.astype(jnp.int32) * 2
        + c3.astype(jnp.int32)
    )
    if group_leafless:
        leafless = ((inputs.exposed_leaf_area_index_patch == 0.)
                    & (inputs.dry_foliage_fraction_patch == 0.))
        branch = (branch << 1) + leafless.astype(jnp.int32)
    temperature = jnp.clip((initial.vegetation_temperature_patch / 5).astype(jnp.int32), 0, 127)
    water = jnp.maximum(jnp.abs(coupling.initial_plant_state.water_potential_segment[:, 0]), 1.0)
    water_bin = jnp.clip(jnp.frexp(water)[1], 0, 63)
    state_key = (temperature << 6) + water_bin
    if initial.iteration_work_patch is None:
        key = (branch << 14) + state_key
    else:
        work = jnp.clip(initial.iteration_work_patch, 0, (1 << 18) - 1)
        key = (branch << 20) + jnp.where(work > 0, work, state_key)
    order = jnp.argsort(key, stable=True)
    packed = jax.tree.map(lambda value: value[order], (inputs, coupling, initial))
    token = transpose_native_storage.set(True)
    try:
        result = fused_native_batch(
            _point,
            *packed,
            parameters,
            shared_argnums=(3,),
            block_size=block_size,
            fold_literals=True,
            direct_io=True,
            guard_branches=True,
            preserve_literal_barriers=True,
        )
    finally:
        transpose_native_storage.reset(token)
    return jax.tree.map(
        lambda value: jnp.empty_like(value).at[order].set(value, unique_indices=True),
        result,
    )
