
from __future__ import annotations

from collections.abc import Callable

import jax

from clm_jax.compilation import process_lifetime_jit
from clm_jax.execution_observer import stage_barrier
from clm_jax.physics.urban_albedo import UrbanAlbedoResult, urban_albedo
from clm_jax.physics.urban_albedo_composition import UrbanAlbedoCall
from clm_jax.runtime import co_locate_array_leaves


class UrbanAlbedoExecutor:

    def __init__(
        self,
        *,
        jit: bool = True,
        stage_completed: Callable[[str], None] | None = None,
    ) -> None:
        self.stage_completed = stage_completed
        self.step = process_lifetime_jit(urban_albedo, enabled=jit)

    def __call__(self, call: UrbanAlbedoCall) -> UrbanAlbedoResult:
        call = co_locate_array_leaves(call, jax.devices()[0])
        result = self.step(call.topology, call.inputs, call.state)
        if self.stage_completed is not None:
            stage_barrier(result)
            self.stage_completed("stage520_urban_albedo")
        return result


__all__ = ["UrbanAlbedoExecutor"]
