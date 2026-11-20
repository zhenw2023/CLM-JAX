
from __future__ import annotations

from collections.abc import Callable
from functools import cache, partial

import jax

from clm_jax.compilation import jit_unique, materialize_outputs, process_lifetime_jit
from clm_jax.execution import (
    canopy_flux_runtime_driver_step_parallel,
    lake_flux_driver_step_parallel,
    lake_temperature_driver_step_parallel,
    soil_temperature_driver_step_parallel,
    urban_flux_driver_step_parallel,
    voc_driver_step_parallel,
)
from clm_jax.execution_observer import stage_barrier, stage_timing_synchronized
from clm_jax.physics.atmospheric_driver import atmospheric_driver_step
from clm_jax.physics.bare_ground_driver import bare_ground_driver_step
from clm_jax.physics.canopy_driver import canopy_driver_step
from clm_jax.physics.canopy_temperature_driver import canopy_temperature_driver_step
from clm_jax.physics.dust import dust_emission_and_deposition
from clm_jax.physics.dust_driver import (
    DustDriverResult,
    assemble_dust_driver_inputs,
    assemble_dust_surface_boundary,
)
from clm_jax.physics.irrigation_demand_driver import irrigation_demand_driver_step
from clm_jax.physics.land_to_atmosphere_boundary import (
    compose_land_to_atmosphere_physical_fields,
)
from clm_jax.physics.physical_driver import (
    PhysicalDriverInputs,
    PhysicalDriverParameters,
    PhysicalDriverRuntimeResult,
    PhysicalDriverState,
    PhysicalDriverStatic,
)
from clm_jax.physics.radiation_sequence import radiation_sequence_step
from clm_jax.runtime import release_completed_stage_memory
from clm_jax.trajectory import NoleapModelClock

_compiled_lnd2atm_boundary = jit_unique(
    compose_land_to_atmosphere_physical_fields, forward_inputs=True,
)


def _physical_prefix(topology, atmospheric_static, state, inputs, parameters, forcing, clock,
                     *, canopy_parameters):
    atmospheric = materialize_outputs(atmospheric_driver_step)(
        atmospheric_static, state[0], inputs[0], forcing, clock,
    )
    canopy = materialize_outputs(canopy_driver_step)(
        topology, atmospheric, forcing, inputs[1], state[1],
        step_seconds=clock.step_seconds, parameters=canopy_parameters,
    )
    radiation = materialize_outputs(radiation_sequence_step)(
        topology, forcing, canopy, inputs[2], state[2],
        current_time_of_day_seconds=clock.seconds_into_day, step_seconds=clock.step_seconds,
    )
    temperature = materialize_outputs(canopy_temperature_driver_step)(
        topology, atmospheric, forcing, canopy, inputs[3], parameters[0], state[3],
    )
    bare_ground = bare_ground_driver_step(
        topology, atmospheric, forcing, canopy, temperature, parameters[1], state[4],
    )
    return atmospheric, canopy, radiation, temperature, bare_ground


@cache
def _compiled_physical_prefix(canopy_parameters):
    return jit_unique(partial(_physical_prefix, canopy_parameters=canopy_parameters),
                      forward_inputs=True)


def _dust_driver_owners(
    topology,
    atmospheric,
    canopy,
    temperature,
    bare_ground,
    canopy_flux,
    urban_inputs,
    urban_flux,
    lake_inputs,
    lake_flux,
    inputs,
):
    surface = assemble_dust_surface_boundary(
        topology,
        bare_ground,
        canopy_flux,
        urban_inputs,
        urban_flux,
        lake_inputs,
        lake_flux,
    )
    emission, deposition = assemble_dust_driver_inputs(
        topology,
        atmospheric,
        canopy,
        temperature,
        inputs,
        surface,
    )
    return surface, emission, deposition


_compiled_dust_owners = jit_unique(_dust_driver_owners, forward_inputs=True)


def _execute_dust_driver_stage(
    dust_core,
    topology,
    atmospheric,
    canopy,
    temperature,
    bare_ground,
    canopy_flux,
    urban_inputs,
    urban_flux,
    lake_inputs,
    lake_flux,
    inputs,
    state,
    parameters,
    execution,
    *,
    assemble=_dust_driver_owners,
) -> DustDriverResult:
    surface, emission, deposition = assemble(
        topology, atmospheric, canopy, temperature, bare_ground, canopy_flux,
        urban_inputs, urban_flux, lake_inputs, lake_flux, inputs,
    )
    step = dust_core(
        topology,
        state,
        emission,
        deposition,
        parameters,
        execution=execution,
    )
    return DustDriverResult(
        surface=surface,
        emission_inputs=emission,
        deposition_inputs=deposition,
        step=step,
        state=step.state,
    )


class PhysicalDriverExecutor:

    def __init__(
        self,
        *,
        device_count: int,
        soil_temperature_device_count: int | None = None,
        jit: bool = True,
        stage_completed: Callable[[str], None] | None = None,
        memory_barrier: Callable[[], object] | None = None,
        retain_diagnostics: bool = True,
        fuse_gpu_prefix: bool = True,
    ) -> None:
        if isinstance(device_count, bool) or not isinstance(device_count, int):
            raise TypeError("physical driver device_count must be an integer")
        if device_count < 1:
            raise ValueError("physical driver device_count must be positive")
        if soil_temperature_device_count is None:
            soil_temperature_device_count = device_count
        if (
            isinstance(soil_temperature_device_count, bool)
            or not isinstance(soil_temperature_device_count, int)
        ):
            raise TypeError("soil-temperature device_count must be an integer")
        if soil_temperature_device_count < 1:
            raise ValueError("soil-temperature device_count must be positive")
        self.device_count = device_count
        self.soil_temperature_device_count = soil_temperature_device_count
        self.urban_flux_device_count = min(device_count, 4)
        self.lake_flux_device_count = min(device_count, 16)
        self.lake_temperature_device_count = min(device_count, 16)
        self.stage_completed = stage_completed
        self.retain_diagnostics = bool(retain_diagnostics)
        self.fuse_gpu_prefix = jit and fuse_gpu_prefix and jax.default_backend() == "gpu"
        self.dust_owners = (
            _compiled_dust_owners if jit and jax.default_backend() == "gpu"
            else _dust_driver_owners
        )
        self.land_to_atmosphere_boundary = (
            _compiled_lnd2atm_boundary
            if jit and jax.default_backend() == "gpu"
            else compose_land_to_atmosphere_physical_fields
        )
        self.memory_barrier = (
            release_completed_stage_memory
            if memory_barrier is None
            else memory_barrier
        )
        if jit:
            self.atmospheric = process_lifetime_jit(atmospheric_driver_step)
            self.canopy = process_lifetime_jit(
                canopy_driver_step,
                static_argnames=("parameters",),
            )
            self.radiation = process_lifetime_jit(radiation_sequence_step)
            self.canopy_temperature = process_lifetime_jit(
                canopy_temperature_driver_step
            )
            self.bare_ground = process_lifetime_jit(bare_ground_driver_step)
            self.irrigation_demand = process_lifetime_jit(
                irrigation_demand_driver_step,
                static_argnames=("parameters",),
            )
            self.dust_core = process_lifetime_jit(
                dust_emission_and_deposition,
                static_argnames=("execution",),
            )
        else:
            self.atmospheric = atmospheric_driver_step
            self.canopy = canopy_driver_step
            self.radiation = radiation_sequence_step
            self.canopy_temperature = canopy_temperature_driver_step
            self.bare_ground = bare_ground_driver_step
            self.irrigation_demand = irrigation_demand_driver_step
            self.dust_core = dust_emission_and_deposition

    def _completed(self, name: str, result):
        if self.stage_completed is not None:
            stage_barrier(result)
            self.stage_completed(name)
        return result

    def _release_completed_stages(self) -> None:

        self.memory_barrier()

    def __call__(
        self,
        static: PhysicalDriverStatic,
        state: PhysicalDriverState,
        inputs: PhysicalDriverInputs,
        parameters: PhysicalDriverParameters,
        forcing,
        clock: NoleapModelClock,
    ) -> PhysicalDriverRuntimeResult:

        topology = static.topology
        if self.fuse_gpu_prefix and not stage_timing_synchronized():
            results = _compiled_physical_prefix(parameters.canopy)(
                topology, static.atmospheric,
                (state.atmospheric, state.canopy, state.radiation,
                 state.canopy_temperature, state.bare_ground),
                (inputs.atmospheric, inputs.canopy, inputs.radiation, inputs.canopy_temperature),
                (parameters.canopy_temperature, parameters.bare_ground),
                forcing, clock,
            )
            atmospheric, canopy, radiation, canopy_temperature, bare_ground = (
                self._completed(name, result) for name, result in zip((
                    "stage140_atmospheric", "stage200_canopy", "stage230_radiation",
                    "stage240_canopy_temperature", "stage250_bare_ground",
                ), results, strict=True)
            )
        else:
            atmospheric = self._completed(
                "stage140_atmospheric",
                self.atmospheric(
                    static.atmospheric,
                    state.atmospheric,
                    inputs.atmospheric,
                    forcing,
                    clock,
                ),
            )
            canopy = self._completed(
                "stage200_canopy",
                self.canopy(
                    topology,
                    atmospheric,
                    forcing,
                    inputs.canopy,
                    state.canopy,
                    step_seconds=clock.step_seconds,
                    parameters=parameters.canopy,
                ),
            )
            radiation = self._completed(
                "stage230_radiation",
                self.radiation(
                    topology,
                    forcing,
                    canopy,
                    inputs.radiation,
                    state.radiation,
                    current_time_of_day_seconds=clock.seconds_into_day,
                    step_seconds=clock.step_seconds,
                ),
            )
            canopy_temperature = self._completed(
                "stage240_canopy_temperature",
                self.canopy_temperature(
                    topology,
                    atmospheric,
                    forcing,
                    canopy,
                    inputs.canopy_temperature,
                    parameters.canopy_temperature,
                    state.canopy_temperature,
                ),
            )
            bare_ground = self._completed(
                "stage250_bare_ground",
                self.bare_ground(
                    topology,
                    atmospheric,
                    forcing,
                    canopy,
                    canopy_temperature,
                    parameters.bare_ground,
                    state.bare_ground,
                ),
            )
        canopy_flux = self._completed(
            "stage260_canopy_flux",
            canopy_flux_runtime_driver_step_parallel(
                topology,
                atmospheric,
                forcing,
                canopy,
                radiation,
                canopy_temperature,
                bare_ground,
                clock,
                inputs.canopy_flux,
                parameters.canopy_flux,
                state.canopy_flux,
                device_count=self.device_count,
            ),
        )
        urban_flux = self._completed(
            "stage270_urban_flux",
            urban_flux_driver_step_parallel(
                topology,
                forcing,
                canopy,
                canopy_temperature,
                inputs.urban_flux,
                parameters.urban_flux,
                state.urban_flux,
                device_count=self.urban_flux_device_count,
            ),
        )
        lake_flux = self._completed(
            "stage280_lake_flux",
            lake_flux_driver_step_parallel(
                topology,
                atmospheric,
                forcing,
                canopy,
                radiation,
                canopy_temperature,
                inputs.lake_flux,
                parameters.lake_flux,
                state.lake_flux,
                device_count=self.lake_flux_device_count,
            ),
        )
        irrigation_demand = self._completed(
            "stage290_irrigation_demand",
            self.irrigation_demand(
                topology,
                atmospheric,
                inputs.canopy,
                canopy,
                canopy_temperature,
                clock,
                inputs.irrigation_demand,
                parameters=parameters.irrigation_demand,
            ),
        )
        self._release_completed_stages()
        dust = self._completed(
            "stage300_dust",
            _execute_dust_driver_stage(
                self.dust_core,
                topology,
                atmospheric,
                canopy,
                canopy_temperature,
                bare_ground,
                canopy_flux,
                inputs.urban_flux,
                urban_flux,
                inputs.lake_flux,
                lake_flux,
                inputs.dust,
                state.dust,
                parameters.dust,
                parameters.dust_execution,
                assemble=self.dust_owners,
            ),
        )
        voc = self._completed(
            "stage310_voc",
            voc_driver_step_parallel(
                topology,
                forcing,
                atmospheric,
                radiation,
                canopy_temperature,
                canopy_flux.voc,
                inputs.voc_accumulators,
                inputs.voc,
                state.voc,
                parameters.voc,
                device_count=self.device_count,
            ),
        )
        lake_temperature = self._completed(
            "stage320_lake_temperature",
            lake_temperature_driver_step_parallel(
                topology,
                canopy,
                radiation,
                canopy_temperature,
                lake_flux,
                inputs.lake_temperature,
                state.lake_temperature,
                time_step_seconds=clock.step_seconds,
                device_count=self.lake_temperature_device_count,
            ),
        )
        self._release_completed_stages()
        soil_temperature = self._completed(
            "stage330_soil_temperature",
            soil_temperature_driver_step_parallel(
                topology,
                atmospheric,
                canopy,
                radiation,
                canopy_temperature,
                canopy_flux.soil_temperature,
                urban_flux,
                inputs.soil_temperature,
                time_step_seconds=clock.step_seconds,
                device_count=self.soil_temperature_device_count,
                memory_barrier=self.memory_barrier,
                retain_diagnostics=self.retain_diagnostics,
            ),
        )
        self._release_completed_stages()
        return PhysicalDriverRuntimeResult(
            atmospheric=atmospheric,
            canopy=canopy,
            radiation=radiation,
            canopy_temperature=canopy_temperature,
            bare_ground=bare_ground,
            canopy_flux=canopy_flux,
            urban_flux=urban_flux,
            lake_flux=lake_flux,
            irrigation_demand=irrigation_demand,
            dust=dust,
            voc=voc,
            lake_temperature=lake_temperature,
            soil_temperature=soil_temperature,
            land_to_atmosphere=self.land_to_atmosphere_boundary(
                canopy_flux,
                inputs.urban_flux,
                urban_flux,
                inputs.lake_flux,
                lake_flux,
                dust,
                voc,
            ),
        )


__all__ = ["PhysicalDriverExecutor"]
