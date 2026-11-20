
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from clm_jax.compilation import jit_unique, process_lifetime_jit
from clm_jax.execution_observer import stage_barrier
from clm_jax.physics.glacier_surface_mass_balance import (
    GlacierIceMeltInputs,
    GlacierIceMeltState,
    handle_glacier_ice_melt,
)
from clm_jax.physics.patch_to_column_fluxes import (
    aggregate_patch_fluxes_to_columns,
)
from clm_jax.physics.physical_driver import (
    PhysicalDriverResult,
    PhysicalDriverRuntimeResult,
)
from clm_jax.physics.post_temperature_driver import (
    PostTemperatureDriverInputs,
    PostTemperatureDriverParameters,
    PostTemperatureDriverResult,
    PostTemperatureDriverState,
    assemble_patch_to_column_inputs_from_precipitation,
    assemble_soil_fluxes_entry_state,
    assemble_soil_fluxes_inputs,
)
from clm_jax.physics.soil_fluxes import NUMBER_OF_SNOW_LEVELS, soil_fluxes
from clm_jax.runtime import release_completed_stage_memory


def _assemble_soil_fluxes_boundary(physical, inputs, state, glacier):
    return (
        assemble_soil_fluxes_inputs(physical, inputs, glacier),
        assemble_soil_fluxes_entry_state(physical, inputs, state),
    )


_compiled_soil_fluxes_boundary = jit_unique(
    _assemble_soil_fluxes_boundary, forward_inputs=True,
)


class PostTemperatureDriverExecutor:

    def __init__(
        self,
        *,
        jit: bool = True,
        stage_completed: Callable[[str], None] | None = None,
        memory_barrier: Callable[[], object] | None = None,
        sources_materialized: Callable[[], None] | None = None,
    ) -> None:
        self.stage_completed = stage_completed
        self.assemble_boundary = (
            _compiled_soil_fluxes_boundary
            if jit and jax.default_backend() == "gpu"
            else _assemble_soil_fluxes_boundary
        )
        self.sources_materialized = sources_materialized
        self.single_owner = jit and jax.default_backend() == "cpu"
        self.memory_barrier = (
            release_completed_stage_memory
            if memory_barrier is None
            else memory_barrier
        )
        if jit:
            self.glacier_step = process_lifetime_jit(handle_glacier_ice_melt)
            self.soil_step = process_lifetime_jit(
                soil_fluxes,
                static_argnames=("parameters",),
                cpu_parallel_codegen_split_count=(1 if self.single_owner else None),
            )
            self.patch_step = process_lifetime_jit(
                aggregate_patch_fluxes_to_columns,
                static_argnames=("parameters",),
            )
        else:
            self.glacier_step = handle_glacier_ice_melt
            self.soil_step = soil_fluxes
            self.patch_step = aggregate_patch_fluxes_to_columns

    def _completed(self, name: str, result):
        if self.stage_completed is not None:
            stage_barrier(result)
            self.stage_completed(name)
        return result

    def _release_completed_stage(self) -> None:
        self.memory_barrier()

    def __call__(
        self,
        physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
        inputs: PostTemperatureDriverInputs,
        state: PostTemperatureDriverState,
        parameters: PostTemperatureDriverParameters,
    ) -> PostTemperatureDriverResult:
        topology = physical.soil_temperature.inputs.topology
        phase = physical.soil_temperature.temperature.soil_snow_phase_change
        precipitation = physical.canopy.canopy_hydrology.fluxes.patch_precipitation
        glacier = self._completed(
            "stage340_glacier_ice_melt",
            self.glacier_step(
                GlacierIceMeltInputs(
                    do_surface_mass_balance=(
                        inputs.do_surface_mass_balance_column
                    ),
                    landunit_type=topology.column.landunit_type,
                    time_step_seconds=jnp.asarray(
                        parameters.soil_fluxes.time_step_seconds,
                        dtype=jnp.float64,
                    ),
                ),
                GlacierIceMeltState(
                    liquid_water=phase.liquid_water[
                        :, NUMBER_OF_SNOW_LEVELS:
                    ],
                    ice_water=phase.ice_water[:, NUMBER_OF_SNOW_LEVELS:],
                    ice_melt_flux=state.glacier_ice_melt_flux_column,
                ),
            ),
        )
        self._release_completed_stage()
        soil_inputs, soil_entry_state = self.assemble_boundary(
            physical, inputs, state, glacier,
        )
        stage_barrier((soil_inputs, soil_entry_state, precipitation))
        if self.single_owner:
            topology, soil_inputs, soil_entry_state = jax.device_put(
                (topology, soil_inputs, soil_entry_state),
                jax.devices()[0],
            )
            stage_barrier((topology, soil_inputs, soil_entry_state))
        patch_to_column_state = state.patch_to_column
        del physical, inputs, state, phase
        if self.sources_materialized is not None:
            self.sources_materialized()
        self._release_completed_stage()
        soil_result = self._completed(
            "stage345_soil_fluxes",
            self.soil_step(
                topology,
                soil_inputs,
                soil_entry_state,
                parameters=parameters.soil_fluxes,
            ),
        )
        self._release_completed_stage()
        patch_inputs = assemble_patch_to_column_inputs_from_precipitation(
            precipitation,
            soil_result,
            soil_inputs,
        )
        patch_result = self._completed(
            "stage350_patch_to_column",
            self.patch_step(
                topology,
                patch_inputs,
                patch_to_column_state,
                parameters=parameters.patch_to_column,
            ),
        )
        self._release_completed_stage()
        return PostTemperatureDriverResult(
            glacier_ice_melt=glacier,
            soil_fluxes_inputs=soil_inputs,
            soil_fluxes_entry_state=soil_entry_state,
            soil_fluxes=soil_result,
            patch_to_column_inputs=patch_inputs,
            patch_to_column=patch_result,
        )


__all__ = ["PostTemperatureDriverExecutor"]
