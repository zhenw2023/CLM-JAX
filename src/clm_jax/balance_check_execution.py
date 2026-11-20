
from __future__ import annotations

from collections.abc import Callable

import jax

from clm_jax.compilation import process_lifetime_jit
from clm_jax.execution_observer import stage_barrier
from clm_jax.forcing.datm import DatmForcing
from clm_jax.physics.balance_check import (
    BalanceCheckParameters,
    BalanceCheckState,
    balance_check,
)
from clm_jax.physics.balance_check_driver import (
    BalanceCheckDriverResult,
    BalanceCheckRadiationFields,
    assemble_balance_check_inputs,
)
from clm_jax.physics.hydrology_drainage_driver import HydrologyDrainageDriverResult
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
from clm_jax.runtime import co_locate_array_leaves


class BalanceCheckExecutor:

    def __init__(
        self,
        *,
        jit: bool = True,
        stage_completed: Callable[[str], None] | None = None,
    ) -> None:
        self.stage_completed = stage_completed
        self.core_step = (
            process_lifetime_jit(
                balance_check,
                static_argnames=("parameters",),
            )
            if jit
            else balance_check
        )

    def __call__(
        self,
        forcing: DatmForcing,
        physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
        post_temperature: PostTemperatureDriverResult,
        hydrology: HydrologyDriverResult,
        post_hydrology: PostHydrologyNoDrainageResult,
        lake_hydrology: LakeHydrologyDriverResult,
        drainage: HydrologyDrainageDriverResult,
        radiation: BalanceCheckRadiationFields,
        state: BalanceCheckState,
        parameters: BalanceCheckParameters,
        *,
        steps_since_restart: int,
    ) -> BalanceCheckDriverResult:
        inputs = assemble_balance_check_inputs(
            forcing,
            physical,
            post_temperature,
            hydrology,
            post_hydrology,
            lake_hydrology,
            drainage,
            radiation,
            steps_since_restart=steps_since_restart,
        )
        topology, inputs, state = co_locate_array_leaves(
            (physical.soil_temperature.inputs.topology, inputs, state),
            jax.devices()[0],
        )
        balance = self.core_step(
            topology,
            inputs,
            state,
            parameters=parameters,
        )
        result = BalanceCheckDriverResult(inputs=inputs, balance=balance)
        if self.stage_completed is not None:
            stage_barrier(result)
            self.stage_completed("stage480_balance_check")
        return result


__all__ = ["BalanceCheckExecutor"]
