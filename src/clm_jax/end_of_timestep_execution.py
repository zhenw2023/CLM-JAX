
from __future__ import annotations

from collections.abc import Callable
from functools import cache, partial

import jax

from clm_jax.compilation import jit_unique
from clm_jax.execution_observer import stage_barrier
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.diagnostics_and_accumulators import (
    ConfiguredAccumulatorParameters,
)
from clm_jax.physics.end_of_timestep import (
    EndOfTimestepInputs,
    EndOfTimestepResult,
    EndOfTimestepState,
    end_of_timestep_step,
)
from clm_jax.runtime import co_locate_array_leaves
from clm_jax.trajectory import NoleapModelClock


@cache
def _configured_end_of_timestep_step(
    number_of_gridcells: int,
    maximum_glacier_classes: int,
    use_cndv: bool,
    *,
    jit: bool,
):

    step = partial(
        end_of_timestep_step,
        number_of_gridcells=number_of_gridcells,
        maximum_glacier_classes=maximum_glacier_classes,
        use_cndv=use_cndv,
    )
    return jit_unique(step, forward_inputs=True) if jit else step


class EndOfTimestepExecutor:

    def __init__(
        self,
        *,
        number_of_gridcells: int,
        maximum_glacier_classes: int,
        use_cndv: bool,
        jit: bool = True,
        stage_completed: Callable[[str], None] | None = None,
    ) -> None:
        if number_of_gridcells <= 0 or maximum_glacier_classes <= 0:
            raise ValueError("end-of-timestep coupling dimensions must be positive")
        if use_cndv:
            raise NotImplementedError(
                "CNDV end-of-step vegetation dynamics are not active in "
                "I2000Clm50BgcCrop"
            )
        self.stage_completed = stage_completed
        self.step = _configured_end_of_timestep_step(
            number_of_gridcells,
            maximum_glacier_classes,
            use_cndv,
            jit=jit,
        )

    def __call__(
        self,
        topology: SubgridTopology,
        inputs: EndOfTimestepInputs,
        state: EndOfTimestepState,
        clock: NoleapModelClock,
        accumulator_parameters: ConfiguredAccumulatorParameters,
    ) -> EndOfTimestepResult:
        topology, inputs, state, clock, accumulator_parameters = co_locate_array_leaves(
            (topology, inputs, state, clock, accumulator_parameters),
            jax.devices()[0],
        )
        result = self.step(
            topology,
            inputs,
            state,
            clock,
            accumulator_parameters,
        )
        if self.stage_completed is not None:
            stage_barrier(result)
            for stage in (
                "stage540_land_to_glacier",
                "stage550_diagnostics_and_accumulators",
                "stage570_end_of_step_vegetation",
            ):
                self.stage_completed(stage)
        return result


__all__ = ["EndOfTimestepExecutor"]
