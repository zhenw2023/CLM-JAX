
from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, replace
from functools import cache
from typing import NamedTuple

import jax
from jax.extend.core import get_opaque_trace_state

from clm_jax.compilation import (
    _EAGER_TRACE,
    _inside_ad_stage,
    _stage_reverse_ad,
    jit_discrete_parameters,
    materialize_outputs,
)
from clm_jax.ecosystem_execution import (
    EcosystemCallOwner,
    EcosystemDynamicsCarryObserver,
    EcosystemDynamicsStageExecutor,
    _execute,
    _stage_functions,
)
from clm_jax.execution_observer import stage_barrier
from clm_jax.physics.cn_flux_layout import (
    prepare_cn_flux_fields,
    restore_cn_fluxes,
    with_fieldwise_cn_fluxes,
)
from clm_jax.physics.cn_fun_allocation import _materialized_fun
from clm_jax.physics.ecosystem_dynamics import (
    EcosystemDynamicsAnnualUpdateState,
    EcosystemDynamicsCompleteInputs,
    EcosystemDynamicsCompleteResult,
)
from clm_jax.physics.ecosystem_dynamics_carry import (
    EcosystemDynamicsPostDrainageCarry,
)
from clm_jax.runtime import release_completed_stage_memory


class EcosystemDynamicsDriverResult(NamedTuple):

    ecosystem: EcosystemDynamicsCompleteResult
    post_drainage_carry: EcosystemDynamicsPostDrainageCarry


@cache
def _combined_gpu_executor(*, share_compact_inputs=False):

    stages = _stage_functions(
        jit=True, retain_diagnostics=False,
        persistent_flux_fields=True, fuse_state_updates=True,
    )
    stages = replace(stages, **{
        field.name: materialize_outputs(getattr(stages, field.name))
        for field in fields(stages) if callable(getattr(stages, field.name))
    })

    def run(inputs, state, stage_functions):
        token = _materialized_fun.set(True)
        try:
            observer = EcosystemDynamicsCarryObserver()
            ecosystem = _execute(
                EcosystemCallOwner(inputs, state), stage_functions,
                stage_observer=observer, retain_diagnostics=False,
            )
            return EcosystemDynamicsDriverResult(ecosystem, observer.carry())
        finally:
            _materialized_fun.reset(token)

    def combined(inputs, state):
        return run(inputs, state, stages)

    fieldwise = with_fieldwise_cn_fluxes(combined, pack_output=False)
    full = jit_discrete_parameters(fieldwise, forward_inputs=True)
    from clm_jax.ecosystem_compact import compact_executor, soil_filter_plan

    compact = compact_executor(run, stages, share_inputs=share_compact_inputs)

    def execute(inputs, state):
        if get_opaque_trace_state() != _EAGER_TRACE:
            if _stage_reverse_ad.get() and not _inside_ad_stage.get():
                return fieldwise(inputs, state)
            return full(inputs, state)
        plan = soil_filter_plan(inputs)
        return full(inputs, state) if plan is None else compact(inputs, state, plan)

    return execute


class EcosystemDynamicsDriverExecutor:

    def __init__(
        self,
        *,
        jit: bool = True,
        stage_completed: Callable[[str], None] | None = None,
        memory_barrier: Callable[[], object] | None = None,
        retain_diagnostics: bool = False,
        collect_between_stages: bool = True,
        preserve_flux_fields: bool = False,
    ) -> None:
        self.stage_completed = stage_completed
        self._preserve_flux_fields = preserve_flux_fields
        self._combine_gpu_stages = (
            jit and preserve_flux_fields and not retain_diagnostics
            and not collect_between_stages and jax.default_backend() == "gpu"
        )
        self._carry_observer = EcosystemDynamicsCarryObserver()
        stage_memory_barrier = None
        if jit and collect_between_stages:
            stage_memory_barrier = (
                release_completed_stage_memory
                if memory_barrier is None
                else memory_barrier
            )
        self._executor = EcosystemDynamicsStageExecutor(
            jit=jit,
            stage_observer=self._carry_observer,
            memory_barrier=stage_memory_barrier,
            retain_diagnostics=retain_diagnostics,
            persistent_flux_fields=jit and not retain_diagnostics,
        )

    def __call__(
        self,
        inputs: EcosystemDynamicsCompleteInputs,
        state: EcosystemDynamicsAnnualUpdateState,
    ) -> EcosystemDynamicsDriverResult:
        return self.execute_prepared(self.prepare(inputs, state))

    def prepare(
        self,
        inputs: EcosystemDynamicsCompleteInputs,
        state: EcosystemDynamicsAnnualUpdateState,
    ) -> EcosystemCallOwner:

        if self._preserve_flux_fields:
            inputs, state = prepare_cn_flux_fields((inputs, state))
        return EcosystemCallOwner(inputs, state)

    def execute_prepared(
        self,
        prepared: EcosystemCallOwner,
    ) -> EcosystemDynamicsDriverResult:

        self._carry_observer.reset()
        if self._combine_gpu_stages:
            ecosystem, carry = _combined_gpu_executor(share_compact_inputs=True)(*prepared.take())
        else:
            ecosystem = self._executor.execute_prepared(prepared)
            carry = self._carry_observer.carry()
        if not self._preserve_flux_fields:
            ecosystem, carry = restore_cn_fluxes((ecosystem, carry))
        stage_barrier(carry)
        self._carry_observer.reset()
        result = EcosystemDynamicsDriverResult(
            ecosystem=ecosystem,
            post_drainage_carry=carry,
        )
        if self.stage_completed is not None:
            self.stage_completed("stage420_ecosystem_dynamics_pre_drainage")
        return result


__all__ = [
    "EcosystemDynamicsDriverExecutor",
    "EcosystemDynamicsDriverResult",
]
