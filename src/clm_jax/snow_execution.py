
from functools import cache

import jax
import jax.numpy as jnp

from clm_jax.execution import _distributed_vmap, _reshard_tree_for_pmap
from clm_jax.io.restart import GridTopology, SubgridTopology
from clm_jax.physics.snow_compaction import snow_compaction_active_case
from clm_jax.physics.snow_layer_combining import combine_snow_layers_active_case
from clm_jax.runtime import gather_mapped_result


@cache
def parallel_snow_stage(kind: str, workers: int):
    if kind not in {"compaction", "combining"}:
        raise ValueError(f"unknown snow stage: {kind}")
    if not 1 <= workers <= jax.local_device_count():
        raise ValueError("snow workers must fit available JAX devices")
    devices = tuple(jax.local_devices()[:workers])
    axis = "snow_column"

    @jax.jit
    def pack(column, inputs, state):
        count = column.active.size
        width = (count + workers - 1) // workers
        if kind == "compaction":
            inputs = inputs._replace(
                wind_speed_gridcell=inputs.wind_speed_gridcell[column.gridcell]
            )

        def packed(value):
            return jnp.pad(
                value, ((0, workers * width - count),) + ((0, 0),) * (value.ndim - 1)
            ).reshape(workers, width, *value.shape[1:])

        return jax.tree.map(packed, (column, inputs, state))

    def compute(column, inputs, state, parameters):
        if kind == "compaction":
            local_index = jnp.arange(column.active.size, dtype=jnp.int32)
            zero = jnp.zeros(column.active.shape, dtype=jnp.float64)
            grid = GridTopology(zero, zero, local_index, local_index)
            topology = SubgridTopology(grid, None, column._replace(gridcell=local_index), None)
            return snow_compaction_active_case(topology, inputs, state, parameters=parameters)
        return combine_snow_layers_active_case(
            SubgridTopology(None, None, column, None), state, parameters=parameters
        )

    def compiled(options):
        return _distributed_vmap(
            compute,
            device_count=workers,
            axis_name=axis,
            in_axes=(0, 0, 0, None),
            static_argnums=(3,),
            compiler_options=options,
        )

    forward = compiled({"xla_cpu_parallel_codegen_split_count": 1})
    transformed = compiled(None)

    @jax.jit(static_argnums=(1,))
    def finish(result, count):
        return jax.tree.map(lambda value: value.reshape((-1, *value.shape[2:]))[:count], result)

    def execute(*arguments):
        if kind == "compaction":
            topology, inputs, state, parameters = arguments
        else:
            topology, state, parameters = arguments
            inputs = ()
        packed = _reshard_tree_for_pmap(
            pack(topology.column, inputs, state), (0, 0, 0), devices, axis
        )
        traced = any(isinstance(value, jax.core.Tracer) for value in jax.tree.leaves(packed))
        result = (transformed if traced else forward)(*packed, parameters)
        return finish(gather_mapped_result(result, devices[0]), topology.column.active.size)

    return execute
