
from __future__ import annotations

from collections.abc import Callable
from functools import cache
import jax

from clm_jax.compilation import process_lifetime_jit, jit_unique
from clm_jax.execution_observer import stage_barrier
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.aerosol import snow_aerosol_masses
from clm_jax.physics.glacier_surface_mass_balance import (
    GlacierSurfaceMassBalanceParameters,
    compute_glacier_surface_mass_balance,
)
from clm_jax.physics.hydrology_driver import HydrologyDriverResult
from clm_jax.physics.post_hydrology_no_drainage_driver import (
    PostHydrologyNoDrainageInputs,
    PostHydrologyNoDrainageResult,
    PostHydrologyNoDrainageState,
    assemble_glacier_surface_mass_balance_boundary,
    assemble_nonlake_aerosol_mass_boundary,
    post_hydrology_no_drainage_step,
)
from clm_jax.physics.post_temperature_driver import PostTemperatureDriverResult


@cache
def _compiled_post_hydrology(parameters):
    def step(*arguments):
        return post_hydrology_no_drainage_step(*arguments, parameters=parameters)

    return jit_unique(step, forward_inputs=True)


class PostHydrologyNoDrainageExecutor:

    def __init__(
        self,
        *,
        jit: bool = True,
        stage_completed: Callable[[str], None] | None = None,
    ) -> None:
        self.stage_completed = stage_completed
        self.gpu = jit and jax.default_backend() == "gpu"
        if jit:
            self.glacier_step = process_lifetime_jit(
                compute_glacier_surface_mass_balance,
                static_argnames=("parameters",),
            )
            self.aerosol_step = process_lifetime_jit(snow_aerosol_masses)
        else:
            self.glacier_step = compute_glacier_surface_mass_balance
            self.aerosol_step = snow_aerosol_masses

    def _completed(self, name: str, result):
        if self.stage_completed is not None:
            stage_barrier(result)
            self.stage_completed(name)
        return result

    def __call__(
        self,
        topology: SubgridTopology,
        post_temperature: PostTemperatureDriverResult,
        hydrology: HydrologyDriverResult,
        inputs: PostHydrologyNoDrainageInputs,
        state: PostHydrologyNoDrainageState,
        parameters: GlacierSurfaceMassBalanceParameters,
    ) -> PostHydrologyNoDrainageResult:
        if self.gpu:
            result = _compiled_post_hydrology(parameters)(
                topology, post_temperature, hydrology, inputs, state,
            )
            if self.stage_completed is not None:
                stage_barrier(result)
                self.stage_completed("stage365_glacier_surface_mass_balance")
                self.stage_completed("stage370_nonlake_aerosol_masses")
            return result
        glacier_inputs, glacier_state = (
            assemble_glacier_surface_mass_balance_boundary(
                post_temperature,
                hydrology,
                inputs,
                state,
            )
        )
        glacier = self._completed(
            "stage365_glacier_surface_mass_balance",
            self.glacier_step(
                topology,
                glacier_inputs,
                glacier_state,
                parameters=parameters,
            ),
        )
        aerosol_inputs, aerosol_state = assemble_nonlake_aerosol_mass_boundary(
            topology,
            hydrology,
            state,
        )
        aerosol = self._completed(
            "stage370_nonlake_aerosol_masses",
            self.aerosol_step(aerosol_inputs, aerosol_state),
        )
        return PostHydrologyNoDrainageResult(
            glacier_surface_mass_balance=glacier,
            aerosol_mass=aerosol,
        )


__all__ = ["PostHydrologyNoDrainageExecutor"]
