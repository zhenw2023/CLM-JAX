
import jax
import jax.numpy as jnp

from clm_jax.physics.snow_water import build_snow_column_masks


def filtered_snow_transition(function, topology, state, *, lake_only=False,
                             capacities=(8192, 49152, 65536), inputs=None,
                             shared_input_fields=(), diagnostic_defaults=None):
    mask = build_snow_column_masks(
        topology, (state if inputs is None else inputs).snow_layer_count_column,
        lake_only=lake_only,
    ).snow_column
    count = mask.size
    capacities = tuple(size for size in capacities if size < count // 2)
    active_count = jnp.count_nonzero(mask)

    def complete():
        if inputs is None:
            return function(topology, state)
        return function(topology, inputs, state)

    def compact(capacity):
        indices = jnp.nonzero(mask, size=capacity, fill_value=count)[0]
        safe = jnp.minimum(indices, count - 1)
        column = jax.tree.map(lambda value: value[safe], topology.column)
        column = column._replace(active=column.active & (indices < count))
        packed_state = jax.tree.map(lambda value: value[safe], state)
        packed_topology = topology._replace(column=column)
        if inputs is None:
            result = function(packed_topology, packed_state)
        else:
            packed_inputs = type(inputs)(*(
                value if name in shared_input_fields else value[safe]
                for name, value in zip(inputs._fields, inputs, strict=True)
            ))
            result = function(packed_topology, packed_inputs, packed_state)

        def restore(base, packed, updated):
            if updated is packed:
                return base
            return jnp.asarray(base, updated.dtype).at[indices].set(updated, mode='drop')

        restored = jax.tree.map(restore, state, packed_state, result.state)
        defaults = {} if diagnostic_defaults is None else diagnostic_defaults
        diagnostics = jax.tree_util.tree_map_with_path(
            lambda path, value: jnp.full(
                (count, *value.shape[1:]), defaults.get(path[-1].name, 0), value.dtype)
                .at[indices].set(value, mode='drop'),
            result.diagnostics,
        )
        return result._replace(state=restored, diagnostics=diagnostics)

    def dispatch(position):
        if position == len(capacities):
            return complete()
        capacity = capacities[position]
        return jax.lax.cond(active_count <= capacity,
                            lambda: compact(capacity), lambda: dispatch(position + 1))

    return dispatch(0)
