
from __future__ import annotations

from collections.abc import Callable

import jax

from clm_jax.compilation import process_lifetime_jit
from clm_jax.execution_observer import stage_barrier
from clm_jax.physics.carbon_nitrogen_balance_check import (
    CarbonNitrogenBalanceCheckParameters,
    carbon_nitrogen_balance_check,
)
from clm_jax.physics.carbon_nitrogen_balance_driver import (
    CarbonNitrogenBalanceCarry,
    CarbonNitrogenBalanceDriverResult,
    assemble_carbon_nitrogen_balance_inputs,
)
from clm_jax.physics.ecosystem_dynamics_post_drainage import (
    EcosystemDynamicsPostDrainageResult,
)
from clm_jax.runtime import co_locate_array_leaves


class CarbonNitrogenBalanceExecutor:

    def __init__(
        self,
        *,
        jit: bool = True,
        stage_completed: Callable[[str], None] | None = None,
    ) -> None:
        self.stage_completed = stage_completed
        self.core_step = (
            process_lifetime_jit(
                carbon_nitrogen_balance_check,
                static_argnames=("parameters",),
            )
            if jit
            else carbon_nitrogen_balance_check
        )

    def __call__(
        self,
        carry: CarbonNitrogenBalanceCarry,
        post_drainage: EcosystemDynamicsPostDrainageResult,
        parameters: CarbonNitrogenBalanceCheckParameters,
        *,
        steps_since_restart: int,
    ) -> CarbonNitrogenBalanceDriverResult:
        inputs = assemble_carbon_nitrogen_balance_inputs(
            carry,
            post_drainage,
            steps_since_restart=steps_since_restart,
        )
        inputs, state = co_locate_array_leaves(
            (inputs, carry.state),
            jax.devices()[0],
        )
        balance = self.core_step(
            inputs,
            state,
            parameters=parameters,
        )
        result = CarbonNitrogenBalanceDriverResult(inputs=inputs, balance=balance)
        if self.stage_completed is not None:
            stage_barrier(result)
            self.stage_completed("stage490_carbon_nitrogen_balance_check")
        return result


__all__ = ["CarbonNitrogenBalanceExecutor"]
