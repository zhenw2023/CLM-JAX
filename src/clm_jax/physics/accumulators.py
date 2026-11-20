
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.physics.numerics import fortran_add, fortran_divide, fortran_multiply

ACCUMULATOR_RESET_VALUE = -99999.0
ACCUMULATOR_MAXIMUM_VALUE = 99999.0


class AccumulatorState(NamedTuple):

    value: jnp.ndarray
    nsteps: jnp.ndarray
    period: int | jnp.ndarray


class AccumulatorResult(NamedTuple):
    state: AccumulatorState
    extracted: jnp.ndarray


def _validate(
    state: AccumulatorState,
    sample: jnp.ndarray,
    active: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    value = jnp.asarray(state.value, dtype=jnp.float64)
    nsteps = jnp.asarray(state.nsteps, dtype=jnp.int32)
    sample = jnp.asarray(sample, dtype=jnp.float64)
    active = jnp.asarray(active, dtype=jnp.bool_)
    if value.shape != nsteps.shape or value.shape != sample.shape or value.shape != active.shape:
        raise ValueError("value, nsteps, sample and active must have identical shapes")
    period = jnp.asarray(state.period, dtype=jnp.int32)
    if period.ndim != 0:
        raise ValueError("period must be scalar")
    return value, nsteps, sample, active


def _fortran_nint(value: jnp.ndarray) -> jnp.ndarray:

    return jnp.where(
        value >= jnp.float64(0.0),
        jnp.floor(value + jnp.float64(0.5)),
        jnp.ceil(value - jnp.float64(0.5)),
    ).astype(jnp.int32)


def update_time_average(
    state: AccumulatorState,
    sample: jnp.ndarray,
    active: jnp.ndarray,
    nstep: int | jnp.ndarray,
    *,
    special_value: float = CLM_SPECIAL_VALUE,
) -> AccumulatorResult:

    value, steps, sample, active = _validate(state, sample, active)
    period = jnp.asarray(state.period, dtype=jnp.int32)
    step = jnp.asarray(nstep, dtype=jnp.int32)
    reset = (((step % period) == jnp.int32(1)) | (period == jnp.int32(1))) & (step != jnp.int32(0))
    reset_mask = active & reset
    value = jnp.where(reset_mask, jnp.float64(0.0), value)
    steps = jnp.where(reset_mask, jnp.int32(0), steps)
    value = jnp.where(active, fortran_add(value, sample), value)
    steps = jnp.where(active, steps + jnp.int32(1), steps)
    interval_end = (step % period) == jnp.int32(0)
    normalized = fortran_divide(value, steps.astype(jnp.float64))
    value = jnp.where(active & interval_end, normalized, value)
    extracted = jnp.where(interval_end, value, jnp.float64(special_value))
    return AccumulatorResult(AccumulatorState(value, steps, period), extracted)


def update_running_mean(
    state: AccumulatorState,
    sample: jnp.ndarray,
    active: jnp.ndarray,
) -> AccumulatorResult:

    value, steps, sample, active = _validate(state, sample, active)
    period = jnp.asarray(state.period, dtype=jnp.int32)
    next_steps = jnp.minimum(steps + jnp.int32(1), period)
    accumulation_period = next_steps.astype(jnp.float64)
    weighted_previous = fortran_multiply(
        (next_steps - jnp.int32(1)).astype(jnp.float64),
        value,
    )
    numerator = fortran_add(weighted_previous, sample)
    candidate = fortran_divide(numerator, accumulation_period)
    value = jnp.where(active, candidate, value)
    steps = jnp.where(active, next_steps, steps)
    return AccumulatorResult(AccumulatorState(value, steps, period), value)


def update_running_accumulation(
    state: AccumulatorState,
    sample: jnp.ndarray,
    active: jnp.ndarray,
) -> AccumulatorResult:

    value, steps, sample, active = _validate(state, sample, active)
    period = jnp.asarray(state.period, dtype=jnp.int32)
    reset = _fortran_nint(sample) == jnp.int32(ACCUMULATOR_RESET_VALUE)
    candidate = jnp.minimum(
        jnp.maximum(fortran_add(value, sample), jnp.float64(0.0)),
        jnp.float64(ACCUMULATOR_MAXIMUM_VALUE),
    )
    updated_value = jnp.where(reset, jnp.float64(0.0), candidate)
    updated_steps = jnp.where(reset, jnp.int32(0), steps + jnp.int32(1))
    value = jnp.where(active, updated_value, value)
    steps = jnp.where(active, updated_steps, steps)
    return AccumulatorResult(AccumulatorState(value, steps, period), value)
