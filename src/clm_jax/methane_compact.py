
from weakref import ref

import jax
import jax.numpy as jnp
import numpy as np

from clm_jax.compilation import jit_unique
from clm_jax.physics.methane_driver_facade import _methane_driver_active_case
from clm_jax.physics.subgrid import column_to_gridcell_mean

_PLANS = {}


def _make_plan(column, patch, patch_column):
    if (np.any((patch_column < 0) | (patch_column >= column.size))
            or not np.array_equal(column[patch_column], patch)):
        return None
    plans = []
    for active in (True, False):
        ci, pi = (np.flatnonzero(mask == active).astype(np.int32) for mask in (column, patch))
        if not ci.size or not pi.size:
            return None
        mapping = np.full(column.size, -1, np.int32)
        mapping[ci] = np.arange(ci.size)
        plans.append((ci, pi, mapping))
    inverse = tuple(np.argsort(np.concatenate([plan[axis] for plan in plans])).astype(np.int32)
                    for axis in (0, 1))
    return jax.tree.map(jnp.asarray, (tuple(plans), inverse))


def methane_filter_plan(inputs):
    sources = (inputs.preprocess.methane_column, inputs.production.methane_patch,
               inputs.production.patch_column)
    if any(isinstance(value, jax.core.Tracer) for value in sources):
        return None
    nc, npatch = sources[0].size, sources[1].size
    ng = inputs.preprocess.latitude_degrees_gridcell.size
    if min(nc, npatch) <= 256 or len({nc, npatch, ng}) != 3:
        return None
    key = tuple(map(id, sources))
    cacheable = all(isinstance(value, jax.Array) for value in sources)
    entry = _PLANS.get(key) if cacheable else None
    if entry is not None and all(
        source() is value for source, value in zip(entry[0], sources, strict=True)
    ):
        return entry[1]
    plan = _make_plan(*map(np.asarray, sources))
    if cacheable:
        def forget(reference):
            current = _PLANS.get(key)
            if current is not None and any(item is reference for item in current[0]):
                _PLANS.pop(key, None)

        _PLANS[key] = (tuple(ref(value, forget) for value in sources), plan)
    return plan


def _select(tree, plan, active, columns, patches):
    ci, pi, mapping = plan
    memo = {}

    def take(path, value):
        if (not hasattr(value, "shape") or not value.ndim
                or value.shape[0] not in (columns, patches)):
            return value
        name = getattr(path[-1], "name", "")
        remap = name if name in {"patch_column", "methane_column", "methane_patch"} else ""
        key = id(value), remap
        if key not in memo:
            selected = value[ci if value.shape[0] == columns else pi]
            if remap == "patch_column":
                selected = mapping[selected]
            elif remap:
                selected = jnp.full(selected.shape, active, jnp.bool_)
            memo[key] = selected
        return memo[key]

    return jax.tree.map_with_path(take, tree)


def _partitioned_methane(inputs, state, parameters, plan):
    plans, inverses = plan
    columns, patches = inputs.preprocess.methane_column.size, inputs.production.methane_patch.size
    results = []
    for active, partition in zip((True, False), plans, strict=True):
        selected = _select((inputs, state), partition, active, columns, patches)
        result = _methane_driver_active_case(
            *selected, parameters, compiled_children=False, materialized_children=True,
        )
        final = result.state.finalize._replace(
            methane_oxidation_gridcell=None, methane_production_gridcell=None,
            net_methane_carbon_correction_gridcell=None,
        )
        results.append(result._replace(state=result.state._replace(finalize=final)))
    memo = {}

    def merge(path, a, b):
        name = getattr(path[-1], "name", "")
        if not a.ndim:
            return a + b if name.endswith("_count") else a
        if "gridcell" in name:
            return a
        if "patch" in name:
            axis = 1
        elif "column" in name:
            axis = 0
        else:
            raise ValueError(f"unclassified methane output axis: {name}")
        key = id(a), id(b)
        if key not in memo:
            memo[key] = jnp.concatenate((a, b), axis=0)[inverses[axis]]
        return memo[key]

    result = jax.tree.map_with_path(merge, *results)

    def mean(value):
        return column_to_gridcell_mean(
            value, inputs.preprocess.column_gridcell, inputs.finalize.column_weight_on_gridcell,
            inputs.finalize.column_active,
            number_of_gridcells=state.finalize.methane_oxidation_gridcell.size,
            maximum_columns_per_gridcell=parameters.finalize.maximum_columns_per_gridcell,
        )

    diagnostics = result.diagnostics.finalize
    final = result.state.finalize._replace(
        methane_oxidation_gridcell=mean(diagnostics.methane_oxidation_g_c_m2_s_column),
        methane_production_gridcell=mean(diagnostics.methane_production_g_c_m2_s_column),
        net_methane_carbon_correction_gridcell=mean(
            diagnostics.net_methane_carbon_correction_g_c_m2_s_column),
    )
    return result._replace(state=result.state._replace(finalize=final))


partitioned_methane = jit_unique(_partitioned_methane, forward_inputs=True)
