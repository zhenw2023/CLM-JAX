
from functools import lru_cache, partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch


@lru_cache(maxsize=2)
def _column_step(lake_only):
    from clm_jax.physics.snow_water import _snow_water_reference

    def step(landunit_type, active, inputs, state, time_step):
        topology = SimpleNamespace(column=SimpleNamespace(
            gridcell=jnp.zeros((1,), jnp.int32),
            landunit_type=landunit_type[None], active=active[None],
        ))
        inputs = jax.tree.map(lambda value: value[None], inputs)
        inputs = inputs._replace(time_step_seconds=time_step)
        state = jax.tree.map(lambda value: value[None], state)
        result = _snow_water_reference(topology, inputs, state, lake_only=lake_only)
        return jax.tree.map(lambda value: value[0], result)

    return step


@partial(jax.custom_jvp, nondiff_argnums=(5,))
def _execute(landunit_type, active, inputs, state, time_step, lake_only):
    return fused_native_batch(
        _column_step(lake_only), landunit_type, active, inputs, state, time_step,
        shared_argnums=(4,), block_size=64, direct_io=True,
        fold_literals=False,
    )


@_execute.defjvp
def _execute_jvp(lake_only, primals, tangents):
    reference = jax.vmap(_column_step(lake_only), in_axes=(0, 0, 0, 0, None))
    return _execute(*primals, lake_only), jax.jvp(reference, primals, tangents)[1]


def snow_water_gpu(topology, inputs, state, *, lake_only=False):
    batched = inputs._replace(
        time_step_seconds=None,
        atmospheric_aerosol_flux_gridcell_species=(
            inputs.atmospheric_aerosol_flux_gridcell_species[topology.column.gridcell]
        ),
    )
    return _execute(
        topology.column.landunit_type, topology.column.active, batched, state,
        inputs.time_step_seconds, lake_only,
    )
