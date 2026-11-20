
from __future__ import annotations

from collections.abc import Callable
from functools import cache, partial
from typing import TypeVar

import jax

from clm_jax.execution_observer import stage_barrier
from clm_jax.io.dry_deposition import DryDepositionConfiguration
from clm_jax.physics.dry_deposition import dry_deposition_velocity

State = TypeVar("State")


def _identity(state: State) -> State:
    return state


@cache
def _configured_step(
    species: tuple[str, ...],
    method: str,
    *,
    jit: bool,
):

    if not species or method != "xactive_lnd":
        return _identity
    step = partial(dry_deposition_velocity, species=species, method=method)
    return jax.jit(step) if jit else step


class DryDepositionExecutor:

    def __init__(
        self,
        configuration: DryDepositionConfiguration,
        *,
        jit: bool = True,
        stage_completed: Callable[[str], None] | None = None,
    ) -> None:
        self.configuration = configuration
        self.stage_completed = stage_completed
        self.step = _configured_step(
            configuration.species,
            configuration.method,
            jit=jit,
        )

    def __call__(self, state: State) -> State:
        result = self.step(state)
        if self.stage_completed is not None:
            stage_barrier(result)
            self.stage_completed("stage440_dry_deposition_velocity")
        return result


__all__ = ["DryDepositionExecutor"]
