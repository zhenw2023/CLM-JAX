
from __future__ import annotations

from collections.abc import Callable
from functools import cache
import jax

from clm_jax.compilation import process_lifetime_jit, jit_unique, materialize_outputs
from clm_jax.execution_observer import stage_barrier
from clm_jax.physics.hydrology_drainage import (
    HydrologyDrainageParameters,
    HydrologyDrainageState,
    hydrology_drainage_active_case,
)
from clm_jax.physics.hydrology_drainage_driver import (
    HydrologyDrainageDriverInputs,
    HydrologyDrainageDriverResult,
    assemble_hydrology_drainage_inputs,
    assemble_hydrology_drainage_state,
)
from clm_jax.physics.hydrology_driver import HydrologyDriverResult
from clm_jax.physics.lake_hydrology_driver import LakeHydrologyDriverResult
from clm_jax.physics.physical_driver import (
    PhysicalDriverResult,
    PhysicalDriverRuntimeResult,
)
from clm_jax.physics.post_hydrology_no_drainage_driver import (
    PostHydrologyNoDrainageResult,
)
from clm_jax.physics.post_temperature_driver import PostTemperatureDriverResult


def _drainage_entry(physical, post_temperature, hydrology, post_hydrology,
                    lake_hydrology, inputs, state):
    return (
        assemble_hydrology_drainage_inputs(
            physical, post_temperature, hydrology, post_hydrology, lake_hydrology, inputs,
        ),
        assemble_hydrology_drainage_state(hydrology, lake_hydrology, state),
    )


@cache
def _compiled_drainage_transition(parameters):
    def transition(physical, *arguments):
        entry_inputs, entry_state = materialize_outputs(_drainage_entry)(
            physical, *arguments,
        )
        result = hydrology_drainage_active_case(
            physical.soil_temperature.inputs.topology, entry_inputs, entry_state,
            parameters=parameters,
        )
        return HydrologyDrainageDriverResult(entry_inputs, entry_state, result)

    return jit_unique(transition, forward_inputs=True)


class HydrologyDrainageDriverExecutor:

    def __init__(
        self,
        *,
        jit: bool = True,
        stage_completed: Callable[[str], None] | None = None,
    ) -> None:
        self.stage_completed = stage_completed
        self.gpu = jit and jax.default_backend() == "gpu"
        if jit:
            self.core_step = process_lifetime_jit(
                hydrology_drainage_active_case,
                static_argnames=("parameters",),
            )
        else:
            self.core_step = hydrology_drainage_active_case

    def __call__(
        self,
        physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
        post_temperature: PostTemperatureDriverResult,
        hydrology: HydrologyDriverResult,
        post_hydrology: PostHydrologyNoDrainageResult,
        lake_hydrology: LakeHydrologyDriverResult,
        inputs: HydrologyDrainageDriverInputs,
        state: HydrologyDrainageState,
        parameters: HydrologyDrainageParameters,
    ) -> HydrologyDrainageDriverResult:
        if self.gpu:
            result = _compiled_drainage_transition(parameters)(
                physical, post_temperature, hydrology, post_hydrology,
                lake_hydrology, inputs, state,
            )
            if self.stage_completed is not None:
                stage_barrier(result)
                self.stage_completed("stage450_hydrology_drainage")
            return result
        entry_inputs = assemble_hydrology_drainage_inputs(
            physical,
            post_temperature,
            hydrology,
            post_hydrology,
            lake_hydrology,
            inputs,
        )
        entry_state = assemble_hydrology_drainage_state(
            hydrology,
            lake_hydrology,
            state,
        )
        drainage = self.core_step(
            physical.soil_temperature.inputs.topology,
            entry_inputs,
            entry_state,
            parameters=parameters,
        )
        result = HydrologyDrainageDriverResult(
            inputs=entry_inputs,
            entry_state=entry_state,
            drainage=drainage,
        )
        if self.stage_completed is not None:
            stage_barrier(result)
            self.stage_completed("stage450_hydrology_drainage")
        return result


__all__ = ["HydrologyDrainageDriverExecutor"]
