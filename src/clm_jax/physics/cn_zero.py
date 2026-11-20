
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import NamedTuple

import jax
import jax.numpy as jnp

from clm_jax.physics.cn_zero_manifest import (
    CN_ZERO_FIELD_COUNT,
    COLUMN_LEVEL_FIELDS,
    COLUMN_LEVEL_POOL_FIELDS,
    COLUMN_LEVEL_TRANSITION_FIELDS,
    COLUMN_POOL_FIELDS,
    COLUMN_SCALAR_FIELDS,
    COLUMN_TRANSITION_FIELDS,
    NUMBER_OF_DECOMPOSITION_LEVELS,
    NUMBER_OF_DECOMPOSITION_POOLS,
    NUMBER_OF_DECOMPOSITION_TRANSITIONS,
    PATCH_SCALAR_FIELDS,
)


class CnZeroInputs(NamedTuple):

    soil_patch: jnp.ndarray
    soil_column: jnp.ndarray


class CnFluxFields(NamedTuple):

    values: tuple[jnp.ndarray, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return (*self.values[0].shape, len(self.values))


class CnTransientFluxState(NamedTuple):

    patch_scalar: jnp.ndarray
    column_scalar: jnp.ndarray
    column_level: jnp.ndarray
    column_pool: jnp.ndarray
    column_transition: jnp.ndarray
    column_level_pool: jnp.ndarray
    column_level_transition: jnp.ndarray


class CnZeroDiagnostics(NamedTuple):
    zeroed_patch_count: jnp.ndarray
    zeroed_column_count: jnp.ndarray


class CnZeroResult(NamedTuple):
    state: CnTransientFluxState
    diagnostics: CnZeroDiagnostics


_FIELD_GROUPS = {
    "patch_scalar": PATCH_SCALAR_FIELDS,
    "column_scalar": COLUMN_SCALAR_FIELDS,
    "column_level": COLUMN_LEVEL_FIELDS,
    "column_pool": COLUMN_POOL_FIELDS,
    "column_transition": COLUMN_TRANSITION_FIELDS,
    "column_level_pool": COLUMN_LEVEL_POOL_FIELDS,
    "column_level_transition": COLUMN_LEVEL_TRANSITION_FIELDS,
}
_FIELD_LOCATIONS = {
    name: (group, index)
    for group, names in _FIELD_GROUPS.items()
    for index, name in enumerate(names)
}
if len(_FIELD_LOCATIONS) != CN_ZERO_FIELD_COUNT:
    raise RuntimeError("CNZero field names must be globally unique")


_soil_only_reset = ContextVar("clm_cn_zero_soil_only", default=False)


@contextmanager
def soil_only_reset_scope():
    token = _soil_only_reset.set(True)
    try:
        yield
    finally:
        _soil_only_reset.reset(token)


def initialize_cn_transient_flux_state(
    number_of_patches: int,
    number_of_columns: int,
    *,
    fill_value: float = 0.0,
    fieldwise: bool = False,
) -> CnTransientFluxState:

    if number_of_patches < 0 or number_of_columns < 0:
        raise ValueError("patch and column counts cannot be negative")
    shapes = (
        (number_of_patches, len(PATCH_SCALAR_FIELDS)),
        (number_of_columns, len(COLUMN_SCALAR_FIELDS)),
        (number_of_columns, NUMBER_OF_DECOMPOSITION_LEVELS, len(COLUMN_LEVEL_FIELDS)),
        (number_of_columns, NUMBER_OF_DECOMPOSITION_POOLS, len(COLUMN_POOL_FIELDS)),
        (number_of_columns, NUMBER_OF_DECOMPOSITION_TRANSITIONS, len(COLUMN_TRANSITION_FIELDS)),
        (number_of_columns, NUMBER_OF_DECOMPOSITION_LEVELS,
         NUMBER_OF_DECOMPOSITION_POOLS, len(COLUMN_LEVEL_POOL_FIELDS)),
        (number_of_columns, NUMBER_OF_DECOMPOSITION_LEVELS,
         NUMBER_OF_DECOMPOSITION_TRANSITIONS, len(COLUMN_LEVEL_TRANSITION_FIELDS)),
    )
    fill = jnp.float64(fill_value)

    def allocate(shape):
        if fieldwise:
            value = jnp.full(shape[:-1], fill, dtype=jnp.float64)
            return CnFluxFields((value,) * shape[-1])
        return jnp.full(shape, fill, dtype=jnp.float64)

    return CnTransientFluxState(*(allocate(shape) for shape in shapes))


def _validate_shapes(inputs: CnZeroInputs, state: CnTransientFluxState) -> None:
    if inputs.soil_patch.ndim != 1 or inputs.soil_column.ndim != 1:
        raise ValueError("soil_patch and soil_column must be vectors")
    number_of_patches = inputs.soil_patch.shape[0]
    number_of_columns = inputs.soil_column.shape[0]
    expected_shapes = {
        "patch_scalar": (number_of_patches, len(PATCH_SCALAR_FIELDS)),
        "column_scalar": (number_of_columns, len(COLUMN_SCALAR_FIELDS)),
        "column_level": (
            number_of_columns,
            NUMBER_OF_DECOMPOSITION_LEVELS,
            len(COLUMN_LEVEL_FIELDS),
        ),
        "column_pool": (
            number_of_columns,
            NUMBER_OF_DECOMPOSITION_POOLS,
            len(COLUMN_POOL_FIELDS),
        ),
        "column_transition": (
            number_of_columns,
            NUMBER_OF_DECOMPOSITION_TRANSITIONS,
            len(COLUMN_TRANSITION_FIELDS),
        ),
        "column_level_pool": (
            number_of_columns,
            NUMBER_OF_DECOMPOSITION_LEVELS,
            NUMBER_OF_DECOMPOSITION_POOLS,
            len(COLUMN_LEVEL_POOL_FIELDS),
        ),
        "column_level_transition": (
            number_of_columns,
            NUMBER_OF_DECOMPOSITION_LEVELS,
            NUMBER_OF_DECOMPOSITION_TRANSITIONS,
            len(COLUMN_LEVEL_TRANSITION_FIELDS),
        ),
    }
    for name, expected_shape in expected_shapes.items():
        if getattr(state, name).shape != expected_shape:
            raise ValueError(f"{name} has an incompatible shape")


def _zero_on_filter(
    values: jnp.ndarray | CnFluxFields, mask: jnp.ndarray
) -> jnp.ndarray | CnFluxFields:
    fields = values.values if isinstance(values, CnFluxFields) else (values,)
    if _soil_only_reset.get() and all(value.dtype == jnp.float64 for value in fields):
        if isinstance(values, CnFluxFields):
            return CnFluxFields(tuple(
                jnp.zeros(value.shape, dtype=value.dtype)
                for value in values.values
            ))
        return jnp.zeros(values.shape, dtype=values.dtype)
    if isinstance(values, CnFluxFields):
        if values.values[0].shape[0] >= 4096 and jax.default_backend() == "gpu":
            from clm_jax.physics.cn_zero_gpu import masked_reset

            return CnFluxFields(masked_reset(values.values, mask))
        return CnFluxFields(tuple(_zero_on_filter(field, mask) for field in values.values))
    broadcast_shape = (mask.shape[0],) + (1,) * (values.ndim - 1)
    return jnp.where(mask.reshape(broadcast_shape), jnp.float64(0.0), values)


def zero_cn_transient_fluxes(
    inputs: CnZeroInputs,
    state: CnTransientFluxState,
) -> CnZeroResult:

    _validate_shapes(inputs, state)
    patch_mask = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    column_mask = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    return CnZeroResult(
        state=CnTransientFluxState(
            patch_scalar=_zero_on_filter(state.patch_scalar, patch_mask),
            column_scalar=_zero_on_filter(state.column_scalar, column_mask),
            column_level=_zero_on_filter(state.column_level, column_mask),
            column_pool=_zero_on_filter(state.column_pool, column_mask),
            column_transition=_zero_on_filter(state.column_transition, column_mask),
            column_level_pool=_zero_on_filter(state.column_level_pool, column_mask),
            column_level_transition=_zero_on_filter(
                state.column_level_transition,
                column_mask,
            ),
        ),
        diagnostics=CnZeroDiagnostics(
            zeroed_patch_count=jnp.sum(patch_mask, dtype=jnp.int32),
            zeroed_column_count=jnp.sum(column_mask, dtype=jnp.int32),
        ),
    )


def get_cn_flux_field(state: CnTransientFluxState, qualified_name: str) -> jnp.ndarray:

    try:
        group, index = _FIELD_LOCATIONS[qualified_name]
    except KeyError as error:
        raise KeyError(f"unknown CN transient flux field: {qualified_name}") from error
    packed = getattr(state, group)
    return packed.values[index] if isinstance(packed, CnFluxFields) else packed[..., index]


def set_cn_flux_field(
    state: CnTransientFluxState,
    qualified_name: str,
    values: jnp.ndarray,
) -> CnTransientFluxState:

    try:
        group, index = _FIELD_LOCATIONS[qualified_name]
    except KeyError as error:
        raise KeyError(f"unknown CN transient flux field: {qualified_name}") from error
    packed = getattr(state, group)
    values = jnp.asarray(values, dtype=jnp.float64)
    if values.shape != packed.shape[:-1]:
        raise ValueError(f"{qualified_name} must have shape {packed.shape[:-1]}")
    if isinstance(packed, CnFluxFields):
        fields = packed.values
        return state._replace(
            **{group: CnFluxFields((*fields[:index], values, *fields[index + 1:]))}
        )
    return state._replace(**{group: packed.at[..., index].set(values)})
