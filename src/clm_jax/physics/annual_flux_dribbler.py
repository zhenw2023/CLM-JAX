
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
)

SECONDS_PER_DAY = 86400.0
YEAR_FRACTION_WRAP_TOLERANCE = 1.0e-15


class AnnualFluxDribblerState(NamedTuple):

    amount_to_dribble: jnp.ndarray
    amount_from_this_timestep: jnp.ndarray


class AnnualFluxDribblerResult(NamedTuple):

    state: AnnualFluxDribblerState
    current_flux: jnp.ndarray
    dribbled_delta: jnp.ndarray
    amount_left_beginning: jnp.ndarray
    amount_left_end: jnp.ndarray
    illegal_nonannual_delta: jnp.ndarray


def set_current_delta(
    state: AnnualFluxDribblerState,
    delta: jnp.ndarray,
    *,
    is_beginning_current_year: jnp.ndarray | bool,
) -> AnnualFluxDribblerState:

    amount = jnp.asarray(state.amount_to_dribble, dtype=jnp.float64)
    passthrough = jnp.asarray(state.amount_from_this_timestep, dtype=jnp.float64)
    current_delta = jnp.asarray(delta, dtype=jnp.float64)
    if amount.shape != current_delta.shape or passthrough.shape != current_delta.shape:
        raise ValueError("dribbler state and delta must have identical shapes")

    beginning = jnp.asarray(is_beginning_current_year, dtype=jnp.bool_)
    return AnnualFluxDribblerState(
        amount_to_dribble=jnp.where(beginning, current_delta, amount),
        amount_from_this_timestep=jnp.where(
            beginning,
            jnp.zeros_like(current_delta),
            current_delta,
        ),
    )


def current_flux(
    state: AnnualFluxDribblerState,
    *,
    time_step_seconds: jnp.ndarray | float,
    days_per_year: jnp.ndarray | float,
) -> jnp.ndarray:

    time_step = jnp.asarray(time_step_seconds, dtype=jnp.float64)
    days = jnp.asarray(days_per_year, dtype=jnp.float64)
    seconds_per_year = fortran_multiply(days, jnp.float64(SECONDS_PER_DAY))
    flux_from_dribbling = fortran_divide(state.amount_to_dribble, seconds_per_year)
    flux_from_this_timestep = fortran_divide(
        state.amount_from_this_timestep,
        time_step,
    )
    return fortran_add(flux_from_dribbling, flux_from_this_timestep)


def dribbled_delta(
    state: AnnualFluxDribblerState,
    *,
    time_step_seconds: jnp.ndarray | float,
    days_per_year: jnp.ndarray | float,
) -> jnp.ndarray:

    flux = current_flux(
        state,
        time_step_seconds=time_step_seconds,
        days_per_year=days_per_year,
    )
    return fortran_multiply(
        flux,
        jnp.asarray(time_step_seconds, dtype=jnp.float64),
    )


def amount_left_to_dribble(
    amount_to_dribble: jnp.ndarray,
    year_fraction: jnp.ndarray | float,
) -> jnp.ndarray:

    amount = jnp.asarray(amount_to_dribble, dtype=jnp.float64)
    fraction = jnp.asarray(year_fraction, dtype=jnp.float64)
    remaining_fraction = fortran_add(jnp.float64(1.0), -fraction)
    remaining = fortran_multiply(amount, remaining_fraction)
    return jnp.where(
        fraction < jnp.float64(YEAR_FRACTION_WRAP_TOLERANCE),
        jnp.zeros_like(amount),
        remaining,
    )


def annual_flux_dribbler_step(
    state: AnnualFluxDribblerState,
    delta: jnp.ndarray,
    *,
    is_beginning_current_year: jnp.ndarray | bool,
    is_first_step: jnp.ndarray | bool,
    previous_year_fraction: jnp.ndarray | float,
    current_year_fraction: jnp.ndarray | float,
    time_step_seconds: jnp.ndarray | float,
    days_per_year: jnp.ndarray | float,
    allows_nonannual_delta: jnp.ndarray | bool = True,
) -> AnnualFluxDribblerResult:

    current_delta = jnp.asarray(delta, dtype=jnp.float64)
    beginning = jnp.asarray(is_beginning_current_year, dtype=jnp.bool_)
    first_step = jnp.asarray(is_first_step, dtype=jnp.bool_)
    allows_nonannual = jnp.asarray(allows_nonannual_delta, dtype=jnp.bool_)
    updated = set_current_delta(
        state,
        current_delta,
        is_beginning_current_year=beginning,
    )
    flux = current_flux(
        updated,
        time_step_seconds=time_step_seconds,
        days_per_year=days_per_year,
    )
    delta_out = fortran_multiply(
        flux,
        jnp.asarray(time_step_seconds, dtype=jnp.float64),
    )
    amount_beginning = amount_left_to_dribble(
        updated.amount_to_dribble,
        previous_year_fraction,
    )
    amount_end = amount_left_to_dribble(
        updated.amount_to_dribble,
        current_year_fraction,
    )
    illegal = (
        (~beginning)
        & (~allows_nonannual)
        & (~first_step)
        & (updated.amount_from_this_timestep != jnp.float64(0.0))
    )
    return AnnualFluxDribblerResult(
        state=updated,
        current_flux=flux,
        dribbled_delta=delta_out,
        amount_left_beginning=amount_beginning,
        amount_left_end=amount_end,
        illegal_nonannual_delta=illegal,
    )
