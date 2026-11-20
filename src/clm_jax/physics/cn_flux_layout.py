
from __future__ import annotations

from functools import wraps

import jax
import jax.numpy as jnp

from clm_jax.physics.cn_zero import CnFluxFields, CnTransientFluxState


def _map_flux_groups(tree, transform):
    transformed = {}

    def change(state):
        if not isinstance(state, CnTransientFluxState):
            return state
        groups = []
        for group in state:
            identity = id(group)
            if identity not in transformed:
                transformed[identity] = transform(group)
            groups.append(transformed[identity])
        return CnTransientFluxState(*groups)

    return jax.tree.map(
        change, tree, is_leaf=lambda value: isinstance(value, CnTransientFluxState)
    )


def split_cn_fluxes(tree):

    return _map_flux_groups(
        tree,
        lambda group: group if isinstance(group, CnFluxFields) else CnFluxFields(
            tuple(group[..., index] for index in range(group.shape[-1]))
        ),
    )


@jax.jit
def _split_packed_group(group):
    return CnFluxFields(tuple(group[..., index] for index in range(group.shape[-1])))


def prepare_cn_flux_fields(tree):

    return _map_flux_groups(
        tree,
        lambda group: group if isinstance(group, CnFluxFields)
        else _split_packed_group(group),
    )


def pack_cn_fluxes(tree):

    return _map_flux_groups(
        tree,
        lambda group: jnp.stack(group.values, axis=-1)
        if isinstance(group, CnFluxFields) else group,
    )


def with_fieldwise_cn_fluxes(function, *, pack_output=True):

    @wraps(function)
    def execute(*arguments):
        result = function(*split_cn_fluxes(arguments))
        return pack_cn_fluxes(result) if pack_output else result

    return execute


@jax.jit
def _stack_fields(fields):
    return jnp.stack(fields, axis=-1)


def restore_cn_fluxes(tree):

    restored = {}

    def group(value):
        if not isinstance(value, CnFluxFields):
            return value
        key = tuple(id(field) for field in value.values)
        if key not in restored:
            restored[key] = _stack_fields(value.values)
        return restored[key]

    return jax.tree.map(group, tree, is_leaf=lambda value: isinstance(value, CnFluxFields))
