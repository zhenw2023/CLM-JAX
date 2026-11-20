
import jax
import jax.numpy as jnp

from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch


def _reference(*arguments):
    from clm_jax.physics.soil_biogeochem_potential import _non_fates_arrays
    return _non_fates_arrays(*arguments)


def _cell(inputs, work, rate, gross, immobilization, active, shared):
    parameters, pool, transition = shared
    inputs = inputs._replace(
        soil_column=inputs.soil_column[None],
        **{name: getattr(inputs, name)[None, None, :] for name in inputs._fields[1:]},
    )
    work = type(work)(*(value[None, None, ...] for value in work))
    result = _reference(
        inputs, work, parameters, inputs.soil_column, active[None], pool, transition,
        rate[None, None, :], gross[None, None], immobilization[None, None],
    )
    return jax.tree.map(lambda value: value[0, 0], result)


@jax.custom_jvp
def non_fates_arrays(inputs, work, parameters, soil, active_level, active_pool,
                     active_transition, rate, gross, immobilization):
    columns, levels = gross.shape
    cells = columns * levels
    if not cells:
        return _reference(inputs, work, parameters, soil, active_level, active_pool,
                          active_transition, rate, gross, immobilization)
    flat_inputs = inputs._replace(
        soil_column=jnp.broadcast_to(soil[:, None], (columns, levels)).reshape(-1),
        **{name: getattr(inputs, name).reshape(cells, -1) for name in inputs._fields[1:]},
    )
    flat_work = type(work)(*(value.reshape(cells, *value.shape[2:]) for value in work))
    active = jnp.broadcast_to(active_level[None, :], (columns, levels)).reshape(-1)
    result = fused_native_batch(
        _cell, flat_inputs, flat_work, rate.reshape(cells, -1), gross.reshape(-1),
        immobilization.reshape(-1), active, (parameters, active_pool, active_transition),
        block_size=128, hardware_rounding=True, shared_argnums=(6,),
    )
    return tuple(value.reshape(columns, levels, *value.shape[1:]) for value in result)


@non_fates_arrays.defjvp
def _non_fates_jvp(primals, tangents):
    return non_fates_arrays(*primals), jax.jvp(_reference, primals, tangents)[1]
