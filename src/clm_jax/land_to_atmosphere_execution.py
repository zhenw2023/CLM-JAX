
from __future__ import annotations

from collections.abc import Callable

import jax

from clm_jax.compilation import process_lifetime_jit
from clm_jax.execution_observer import stage_barrier
from clm_jax.physics.land_to_atmosphere import (
    LandToAtmosphereResult,
    land_to_atmosphere_coupling,
)
from clm_jax.physics.land_to_atmosphere_composition import LandToAtmosphereCall
from clm_jax.runtime import co_locate_array_leaves


class LandToAtmosphereExecutor:

    def __init__(
        self,
        *,
        jit: bool = True,
        stage_completed: Callable[[str], None] | None = None,
    ) -> None:
        self.stage_completed = stage_completed
        self.step = (
            process_lifetime_jit(
                land_to_atmosphere_coupling,
                static_argnames=("parameters",),
            )
            if jit
            else land_to_atmosphere_coupling
        )

    def __call__(self, call: LandToAtmosphereCall) -> LandToAtmosphereResult:
        call = co_locate_array_leaves(call, jax.devices()[0])
        result = self.step(
            call.topology,
            call.inputs,
            call.state,
            parameters=call.parameters,
        )
        if self.stage_completed is not None:
            stage_barrier(result)
            self.stage_completed("stage530_land_to_atmosphere")
        return result


__all__ = ["LandToAtmosphereExecutor"]
