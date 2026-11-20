
from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
from jax import lax
from jax.extend.core import get_opaque_trace_state

from clm_jax.compilation import _EAGER_TRACE
from clm_jax.io.restart import RestartClock, model_step_from_restart_clock

_NOLEAP_MONTH_LENGTHS = jnp.asarray(
    (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
    dtype=jnp.int32,
)
_SECONDS_PER_DAY = 86_400
_NOLEAP_DAYS_BEFORE_MONTH = jnp.asarray(
    (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334),
    dtype=jnp.int32,
)

State = TypeVar("State")
Forcing = TypeVar("Forcing")
Output = TypeVar("Output")


class NoleapModelClock(NamedTuple):

    year: jax.Array
    month: jax.Array
    day: jax.Array
    seconds_into_day: jax.Array
    model_step: jax.Array
    step_seconds: jax.Array


class ModelTimeFlags(NamedTuple):

    beginning_current_day: jax.Array
    end_current_day: jax.Array
    beginning_current_month: jax.Array
    beginning_current_year: jax.Array
    end_current_year: jax.Array


class ModelStepResult(NamedTuple):

    state: Any
    output: Any


class TrajectoryResult(NamedTuple):

    state: Any
    clock: NoleapModelClock
    history: Any


def noleap_clock_from_restart(clock: RestartClock) -> NoleapModelClock:

    date = int(clock.current_date)
    return NoleapModelClock(
        year=jnp.asarray(date // 10_000, dtype=jnp.int32),
        month=jnp.asarray((date // 100) % 100, dtype=jnp.int32),
        day=jnp.asarray(date % 100, dtype=jnp.int32),
        seconds_into_day=jnp.asarray(clock.current_time_of_day, dtype=jnp.int32),
        model_step=jnp.asarray(
            model_step_from_restart_clock(clock, "NO_LEAP"),
            dtype=jnp.int64,
        ),
        step_seconds=jnp.asarray(clock.step_seconds, dtype=jnp.int32),
    )


def _validate_clock(clock: NoleapModelClock) -> None:

    for name, value in zip(NoleapModelClock._fields, clock, strict=True):
        if jnp.shape(value) != ():
            raise ValueError(f"model clock {name} must be scalar")


@jax.jit
def advance_noleap_clock(clock: NoleapModelClock) -> NoleapModelClock:

    _validate_clock(clock)
    step_seconds = jnp.asarray(clock.step_seconds, dtype=jnp.int32)
    total_seconds = jnp.asarray(clock.seconds_into_day, dtype=jnp.int32) + step_seconds
    days_advanced = total_seconds // jnp.int32(_SECONDS_PER_DAY)
    seconds_into_day = total_seconds % jnp.int32(_SECONDS_PER_DAY)

    def advance_one_day(_, date):
        year, month, day = date
        month_length = _NOLEAP_MONTH_LENGTHS[month - jnp.int32(1)]
        end_month = day == month_length
        next_day = jnp.where(end_month, jnp.int32(1), day + jnp.int32(1))
        next_month = jnp.where(end_month, month + jnp.int32(1), month)
        end_year = next_month == jnp.int32(13)
        next_year = jnp.where(end_year, year + jnp.int32(1), year)
        next_month = jnp.where(end_year, jnp.int32(1), next_month)
        return next_year, next_month, next_day

    year, month, day = lax.fori_loop(
        jnp.int32(0),
        days_advanced,
        advance_one_day,
        (
            jnp.asarray(clock.year, dtype=jnp.int32),
            jnp.asarray(clock.month, dtype=jnp.int32),
            jnp.asarray(clock.day, dtype=jnp.int32),
        ),
    )
    return NoleapModelClock(
        year=year,
        month=month,
        day=day,
        seconds_into_day=seconds_into_day,
        model_step=jnp.asarray(clock.model_step, dtype=jnp.int64) + jnp.int64(1),
        step_seconds=step_seconds,
    )


@jax.jit
def model_time_flags(clock: NoleapModelClock) -> ModelTimeFlags:

    _validate_clock(clock)
    month = jnp.asarray(clock.month, dtype=jnp.int32)
    day = jnp.asarray(clock.day, dtype=jnp.int32)
    seconds = jnp.asarray(clock.seconds_into_day, dtype=jnp.int32)
    step_seconds = jnp.asarray(clock.step_seconds, dtype=jnp.int32)
    start_day = seconds == step_seconds
    end_day = seconds == jnp.int32(0)
    return ModelTimeFlags(
        beginning_current_day=start_day,
        end_current_day=end_day,
        beginning_current_month=start_day & (day == jnp.int32(1)),
        beginning_current_year=(start_day & (month == jnp.int32(1)) & (day == jnp.int32(1))),
        end_current_year=(end_day & (month == jnp.int32(1)) & (day == jnp.int32(1))),
    )


@jax.jit
def _eager_calendar_day(clock: NoleapModelClock) -> jax.Array:
    month = jnp.asarray(clock.month, jnp.int32)
    day = jnp.asarray(clock.day, jnp.int32)
    denominator = lax.optimization_barrier(jnp.float64(_SECONDS_PER_DAY))
    fraction = lax.optimization_barrier(
        jnp.asarray(clock.seconds_into_day, jnp.float64) / denominator,
    )
    return (
        jnp.asarray(_NOLEAP_DAYS_BEFORE_MONTH[month - 1], jnp.float64)
        + jnp.asarray(day, jnp.float64)
        + fraction
    )


def noleap_calendar_day_from_clock(clock: NoleapModelClock) -> jax.Array:

    _validate_clock(clock)
    if get_opaque_trace_state() == _EAGER_TRACE:
        return _eager_calendar_day(clock)
    month = jnp.asarray(clock.month, dtype=jnp.int32)
    day = jnp.asarray(clock.day, dtype=jnp.int32)
    seconds = jnp.asarray(clock.seconds_into_day, dtype=jnp.float64)
    return (
        jnp.asarray(_NOLEAP_DAYS_BEFORE_MONTH[month - jnp.int32(1)], dtype=jnp.float64)
        + jnp.asarray(day, dtype=jnp.float64)
        + seconds / jnp.float64(_SECONDS_PER_DAY)
    )


@jax.jit
def datm_next_shortwave_calendar_day(clock: NoleapModelClock) -> jax.Array:
    from clm_jax.floating_point import correctly_rounded_divide
    from clm_jax.physics.numerics import fortran_hardware_add

    _validate_clock(clock)
    month = jnp.asarray(clock.month, jnp.int32)
    day = _NOLEAP_DAYS_BEFORE_MONTH[month - 1] + jnp.asarray(clock.day, jnp.int32)
    step = jnp.asarray(clock.step_seconds, jnp.int32)
    previous_seconds = jnp.asarray(clock.seconds_into_day, jnp.int32) - step
    previous_day = day + previous_seconds // jnp.int32(_SECONDS_PER_DAY)
    previous_day = (previous_day - 1) % jnp.int32(365) + 1
    previous_seconds = previous_seconds % jnp.int32(_SECONDS_PER_DAY)
    seconds_per_day = jnp.float64(_SECONDS_PER_DAY)
    previous_julian = fortran_hardware_add(
        previous_day.astype(jnp.float64),
        correctly_rounded_divide(previous_seconds.astype(jnp.float64), seconds_per_day),
    )
    return fortran_hardware_add(
        previous_julian, correctly_rounded_divide(step.astype(jnp.float64), seconds_per_day),
    )


@jax.jit
def prepare_trajectory_step(forcing: Forcing, index, clock: NoleapModelClock):
    call_clock = advance_noleap_clock(clock)
    forcing_step = jax.tree.map(lambda value: value[index], forcing)
    host_clock = jnp.stack((
        call_clock.year * 10_000 + call_clock.month * 100 + call_clock.day,
        call_clock.seconds_into_day,
        call_clock.model_step,
    ))
    return forcing_step, call_clock, host_clock


def integrate_trajectory(
    step: Callable[[State, Forcing, NoleapModelClock, ModelTimeFlags], ModelStepResult],
    initial_state: State,
    initial_clock: NoleapModelClock,
    forcing: Forcing,
) -> TrajectoryResult:

    _validate_clock(initial_clock)

    def scan_step(carry, forcing_step):
        state, clock = carry
        call_clock = advance_noleap_clock(clock)
        result = step(
            state,
            forcing_step,
            call_clock,
            model_time_flags(call_clock),
        )
        if not isinstance(result, ModelStepResult):
            raise TypeError("complete model step must return ModelStepResult")
        return (result.state, call_clock), result.output

    (state, clock), history = lax.scan(
        scan_step,
        (initial_state, initial_clock),
        forcing,
    )
    return TrajectoryResult(state=state, clock=clock, history=history)


__all__ = [
    "ModelStepResult",
    "ModelTimeFlags",
    "NoleapModelClock",
    "TrajectoryResult",
    "advance_noleap_clock",
    "datm_next_shortwave_calendar_day",
    "integrate_trajectory",
    "model_time_flags",
    "noleap_calendar_day_from_clock",
    "noleap_clock_from_restart",
    "prepare_trajectory_step",
]
