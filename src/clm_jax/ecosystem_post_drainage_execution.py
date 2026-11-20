
from __future__ import annotations

from collections.abc import Callable

from clm_jax.compilation import jit_unique, process_lifetime_jit
from clm_jax.execution_observer import stage_barrier
from clm_jax.physics.cn_flux_layout import with_fieldwise_cn_fluxes
from clm_jax.physics.ecosystem_dynamics_post_drainage import (
    EcosystemDynamicsPostDrainageInputs,
    EcosystemDynamicsPostDrainageParameters,
    EcosystemDynamicsPostDrainageResult,
    EcosystemDynamicsPostDrainageState,
    ecosystem_dynamics_post_drainage,
)

_fieldwise_post_drainage = with_fieldwise_cn_fluxes(
    ecosystem_dynamics_post_drainage
)
_persistent_fieldwise_post_drainage = jit_unique(
    with_fieldwise_cn_fluxes(ecosystem_dynamics_post_drainage, pack_output=False),
    forward_inputs=True,
)


class EcosystemDynamicsPostDrainageExecutor:

    def __init__(
        self,
        *,
        jit: bool = True,
        stage_completed: Callable[[str], None] | None = None,
        preserve_flux_fields: bool = False,
    ) -> None:
        self.stage_completed = stage_completed
        self.step = (
            (
                _persistent_fieldwise_post_drainage
                if preserve_flux_fields
                else process_lifetime_jit(_fieldwise_post_drainage)
            )
            if jit
            else ecosystem_dynamics_post_drainage
        )

    def __call__(
        self,
        inputs: EcosystemDynamicsPostDrainageInputs,
        state: EcosystemDynamicsPostDrainageState,
        parameters: EcosystemDynamicsPostDrainageParameters,
    ) -> EcosystemDynamicsPostDrainageResult:
        result = self.step(inputs, state, parameters)
        if self.stage_completed is not None:
            stage_barrier(result)
            self.stage_completed("stage460_ecosystem_dynamics_post_drainage")
        return result


__all__ = ["EcosystemDynamicsPostDrainageExecutor"]
