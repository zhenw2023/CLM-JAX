
from functools import lru_cache
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from clm_jax.physics.native_tile_call import transpose_native_storage
from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch


@lru_cache(maxsize=8)
def _program(operation, parameters, lake_only):
    if operation == "compaction":
        from clm_jax.physics.snow_compaction import _snow_compaction_active_case as reference
    elif operation == "combining":
        from clm_jax.physics.snow_layer_combining import _combine_snow_layers_active_case as reference
    elif operation == "division":
        from clm_jax.physics.snow_layer_division import _divide_snow_layers_active_case as reference
    else:
        raise ValueError(operation)

    def point(landunit_type, active, inputs, state):
        topology = SimpleNamespace(
            column=SimpleNamespace(gridcell=jnp.zeros((1,), jnp.int32),
                                   landunit_type=landunit_type[None], active=active[None]),
            grid=SimpleNamespace(longitude_degrees=jnp.zeros((1,), jnp.float64)),
        )
        local_state = jax.tree.map(lambda value: value[None], state)
        if operation == "compaction":
            result = reference(topology, jax.tree.map(lambda value: value[None], inputs),
                               local_state, parameters=parameters, lake_only=lake_only)
        elif operation == "combining":
            result = reference(topology, local_state, parameters=parameters, lake_only=lake_only)
        else:
            result = reference(topology, local_state, parameters=parameters, is_lake=lake_only)
        return jax.tree.map(lambda value: value[0], result)

    @jax.custom_jvp
    def execute(*arguments):
        token = transpose_native_storage.set(True)
        try:
            return fused_native_batch(point, *arguments, block_size=32, direct_io=True,
                                      fold_literals=False)
        finally:
            transpose_native_storage.reset(token)

    @execute.defjvp
    def derivative(primals, tangents):
        return execute(*primals), jax.jvp(jax.vmap(point), primals, tangents)[1]

    return execute


def snow_layers_gpu(operation, topology, state, parameters, *, inputs=None, lake_only=False):
    if inputs is not None:
        inputs = inputs._replace(wind_speed_gridcell=(
            inputs.wind_speed_gridcell[topology.column.gridcell]))
    return _program(operation, parameters, lake_only)(
        topology.column.landunit_type, topology.column.active, inputs, state)
