
from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any
from weakref import ref

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import get_opaque_trace_state

from clm_jax.compilation import (
    _EAGER_TRACE, _inside_ad_stage, _stage_reverse_ad, jit_unique, materialize_outputs,
    process_lifetime_jit,
)
from clm_jax.execution import (
    ColumnPatchShardPlan,
    _distributed_vmap,
    _pack_column_array,
    _pack_owned_patch_array,
    _reshard_tree_for_pmap,
    _unpack_column_array,
    plan_column_patch_shards,
)
from clm_jax.execution_observer import stage_barrier
from clm_jax.forcing.datm import DatmForcing
from clm_jax.io.restart import ColumnTopology, PatchTopology, SubgridTopology
from clm_jax.physics.hydrology_driver import (
    HydrologyDriverInputs,
    HydrologyDriverResult,
    assemble_hydrology_no_drainage_inputs,
    assemble_hydrology_no_drainage_state,
)
from clm_jax.physics.hydrology_no_drainage import (
    DIRECT_HYDROLOGY_NO_DRAINAGE_STAGES,
    HydrologyNoDrainageInputs,
    HydrologyNoDrainageParameters,
    HydrologyNoDrainageResult,
    HydrologyNoDrainageStageFunctions,
    HydrologyNoDrainageState,
    hydrology_no_drainage_active_case,
    hydrology_no_drainage_with_stages,
)
from clm_jax.physics.physical_driver import (
    PhysicalDriverResult,
    PhysicalDriverRuntimeResult,
)
from clm_jax.physics.post_temperature_driver import PostTemperatureDriverResult
from clm_jax.physics.soil_water_movement import (
    SoilWaterMovementDiagnostics,
    SoilWaterMovementResult,
    soil_water_movement_active_case,
)
from clm_jax.physics.soil_water_plant_sink import SoilWaterPlantSinkInputs
from clm_jax.physics.surface_hydrology import build_surface_hydrology_masks
from clm_jax.runtime import co_locate_array_leaves, gather_mapped_result
from clm_jax.snow_execution import parallel_snow_stage

_HYDROLOGY_AXIS_NAME = "clm_hydrology_device"
_RICHARDS_AXIS_NAME = "clm_richards_column"
_USE_FUSED_GPU_HYDROLOGY = True


def _hydrology_owners(physical, post, forcing, inputs, state):
    entry = assemble_hydrology_no_drainage_state(physical, post, state)
    assembled = assemble_hydrology_no_drainage_inputs(physical, post, forcing, inputs, entry)
    return assembled, entry


def _hydrology_transition(topology, inputs, state, parameters):
    return hydrology_no_drainage_active_case(topology, inputs, state, parameters=parameters)


_compiled_hydrology_owners = jit_unique(_hydrology_owners, forward_inputs=True)
@cache
def _gpu_hydrology_program(parameters, compact=False):
    def transition(topology, inputs, state, indices, mask):
        def richards(topology, inputs, state, parameters):
            if compact:
                return _compact_soil_water_movement.__wrapped__(
                    topology, inputs, state, parameters, indices, mask,
                )
            return soil_water_movement_active_case(
                topology, inputs, state, parameters=parameters,
            )

        direct = DIRECT_HYDROLOGY_NO_DRAINAGE_STAGES._replace(soil_water_movement=richards)
        stages = type(direct)(*(materialize_outputs(function) for function in direct))
        return hydrology_no_drainage_with_stages(
            topology, inputs, state, parameters=parameters, stages=stages,
        )

    return jit_unique(transition, forward_inputs=True)


def _compiled_hydrology_transition(topology, inputs, state, parameters, filter_topology=None):
    plan = None
    if jax.default_backend() == "gpu" and get_opaque_trace_state() == _EAGER_TRACE:
        plan = _static_hydrology_indices(topology if filter_topology is None else filter_topology)
    return _gpu_hydrology_program(parameters, plan is not None)(
        topology, inputs, state, *(plan if plan is not None else (None, None)),
    )
_PATCH_PLANT_SINK_FIELDS = frozenset(
    {
        "soil_root_conductance_patch_level",
        "root_fraction_patch_level",
        "vegetation_transpiration_patch",
        "exposed_vegetation_flag_patch",
        "vegetation_water_potential_patch_segment",
    }
)
_TOPOLOGY_AXES = SubgridTopology(
    grid=None,
    landunit=None,
    column=ColumnTopology(*(0 for _ in ColumnTopology._fields)),
    patch=PatchTopology(*(0 for _ in PatchTopology._fields)),
)


def _pack_hydrology_topology(
    topology: SubgridTopology,
    plan: ColumnPatchShardPlan,
) -> SubgridTopology:
    column_valid = jnp.arange(
        plan.device_count * plan.columns_per_device,
        dtype=jnp.int32,
    ) < jnp.int32(plan.column_count)
    column_valid = jnp.reshape(
        column_valid,
        (plan.device_count, plan.columns_per_device),
    )
    column = jax.tree.map(lambda value: _pack_column_array(value, plan), topology.column)
    column = column._replace(active=column.active & column_valid)
    patch = jax.tree.map(lambda value: _pack_owned_patch_array(value, plan), topology.patch)
    column_start = jnp.arange(plan.device_count, dtype=jnp.int32)[:, None] * jnp.int32(
        plan.columns_per_device
    )
    patch = patch._replace(
        column=jnp.where(plan.patch_valid, patch.column - column_start, jnp.int32(0)),
        active=patch.active & plan.patch_valid,
    )
    return topology._replace(column=column, patch=patch)


def _pack_hydrology_inputs(
    inputs: HydrologyNoDrainageInputs,
    plan: ColumnPatchShardPlan,
) -> HydrologyNoDrainageInputs:
    def replicated(value: Any) -> jax.Array:
        array = jnp.asarray(value)
        return jnp.broadcast_to(array, (plan.device_count, *array.shape))

    def columns(tree: Any, *, replicated_fields: frozenset[str] = frozenset()) -> Any:
        return type(tree)(
            *(
                replicated(value)
                if name in replicated_fields
                else _pack_column_array(value, plan)
                for name, value in zip(tree._fields, tree, strict=True)
            )
        )

    plant_sink = SoilWaterPlantSinkInputs(
        *(
            (
                _pack_owned_patch_array(value, plan)
                if name in _PATCH_PLANT_SINK_FIELDS
                else _pack_column_array(value, plan)
            )
            for name, value in zip(
                SoilWaterPlantSinkInputs._fields,
                inputs.plant_sink,
                strict=True,
            )
        )
    )
    packed = {
        name: columns(getattr(inputs, name))
        for name in HydrologyNoDrainageInputs._fields
        if name not in {"plant_sink", "snow_water", "surface_hydrology", "snow_compaction"}
    }
    packed["snow_water"] = columns(
        inputs.snow_water,
        replicated_fields=frozenset(
            {
                "time_step_seconds",
                "atmospheric_aerosol_flux_gridcell_species",
            }
        ),
    )
    packed["surface_hydrology"] = columns(
        inputs.surface_hydrology,
        replicated_fields=frozenset({"time_step_seconds"}),
    )
    packed["snow_compaction"] = columns(
        inputs.snow_compaction,
        replicated_fields=frozenset({"wind_speed_gridcell"}),
    )
    return inputs._replace(
        **packed,
        plant_sink=plant_sink,
    )


def _hydrology_shard(
    topology: SubgridTopology,
    inputs: HydrologyNoDrainageInputs,
    state: HydrologyNoDrainageState,
    parameters: HydrologyNoDrainageParameters,
) -> HydrologyNoDrainageResult:
    return hydrology_no_drainage_with_stages(
        topology,
        inputs,
        state,
        parameters=parameters,
        stages=DIRECT_HYDROLOGY_NO_DRAINAGE_STAGES,
    )


@cache
def _richards_programs(device_count: int):

    @jax.jit
    def pack(column, inputs, state):
        count = column.active.size
        width = (count + device_count - 1) // device_count

        def padded(value):
            values = jnp.pad(
                value,
                ((0, width * device_count - count),) + ((0, 0),) * (value.ndim - 1),
            )
            return values.reshape(device_count, width, *value.shape[1:])

        return jax.tree.map(padded, (column, inputs, state))

    def solve_columns(column, inputs, state, parameters):
        return soil_water_movement_active_case(
            SubgridTopology(None, None, column, None), inputs, state, parameters=parameters,
        )

    def solve(column, inputs, state, parameters):
        count = column.active.size
        width = min(64, count)
        blocks = (count + width - 1) // width

        def tiles(value):
            padded = jnp.pad(
                value,
                ((0, blocks * width - count),) + ((0, 0),) * (value.ndim - 1),
            )
            return padded.reshape(blocks, width, *value.shape[1:])

        def solve_tile(arguments):
            return solve_columns(*arguments, parameters)

        solved = jax.lax.map(solve_tile, jax.tree.map(tiles, (column, inputs, state)))
        return jax.tree.map(
            lambda value: value.reshape((-1, *value.shape[2:]))[:count], solved,
        )

    mapped = jax.pmap(
        solve, axis_name=_RICHARDS_AXIS_NAME, in_axes=(0, 0, 0, None),
        static_broadcasted_argnums=(3,), devices=jax.local_devices()[:device_count],
    )
    transformed = jax.pmap(
        solve_columns, axis_name=_RICHARDS_AXIS_NAME, in_axes=(0, 0, 0, None),
        static_broadcasted_argnums=(3,), devices=jax.local_devices()[:device_count],
    )

    def execute(*arguments):
        traced = any(isinstance(value, jax.core.Tracer) for value in jax.tree.leaves(arguments))
        return (transformed if traced else mapped)(*arguments)

    return pack, execute


def _parallel_soil_water_movement(topology, inputs, state, parameters):
    count = min(16, jax.local_device_count())
    pack, solve = _richards_programs(count)
    devices = tuple(jax.local_devices()[:count])
    packed = _reshard_tree_for_pmap(
        pack(topology.column, inputs, state), (0, 0, 0), devices, _RICHARDS_AXIS_NAME,
    )
    solved = gather_mapped_result(solve(*packed, parameters), devices[0])
    columns = topology.column.active.size
    return jax.tree.map(lambda value: value.reshape((-1, *value.shape[2:]))[:columns], solved)


_HYDROLOGY_STATIC_INDICES: dict[tuple[int, ...], tuple[Any, Any]] = {}


def _static_hydrology_indices(topology):
    column = topology.column
    sources = (column.active, column.landunit_type, column.column_type)
    cacheable = all(isinstance(value, jax.Array) for value in sources)
    key = tuple(id(value) for value in sources)
    entry = _HYDROLOGY_STATIC_INDICES.get(key) if cacheable else None
    if entry is not None and all(
        reference() is value and not value.is_deleted()
        for reference, value in zip(entry[0], sources, strict=True)
    ):
        return entry[1]
    mask = build_surface_hydrology_masks(topology).hydrology_column
    indices = np.flatnonzero(np.asarray(mask)).astype(np.int32)
    plan = None
    if 0 < indices.size < mask.size:
        indices = np.pad(indices, (0, max(0, 2 - indices.size)), constant_values=mask.size)
        plan = (jnp.asarray(indices, jnp.int32), mask)
    if cacheable:
        def forget(reference):
            current = _HYDROLOGY_STATIC_INDICES.get(key)
            if current is not None and any(item is reference for item in current[0]):
                _HYDROLOGY_STATIC_INDICES.pop(key, None)

        _HYDROLOGY_STATIC_INDICES[key] = (tuple(ref(value, forget) for value in sources), plan)
    return plan


@jax.jit(static_argnums=(3,))
def _compact_soil_water_movement(topology, inputs, state, parameters, indices, mask):
    count = mask.size
    safe = jnp.minimum(indices, count - 1)
    column, selected_inputs, selected_state = jax.tree.map(
        lambda value: value[safe], (topology.column, inputs, state),
    )
    column = column._replace(active=column.active & (indices < count))
    solved = soil_water_movement_active_case(
        topology._replace(column=column), selected_inputs, selected_state, parameters=parameters,
    )
    restored = jax.tree.map(
        lambda original, updated: jnp.asarray(original, updated.dtype)
            .at[indices].set(updated, mode="drop"),
        state, solved.state,
    )
    diagnostics = SoilWaterMovementDiagnostics(
        hydrology_column=mask,
        adaptive_loop_completed_column=jnp.ones(mask.shape, bool).at[indices].set(
            solved.diagnostics.adaptive_loop_completed_column, mode="drop"),
        final_maximum_flux_error_column=jnp.zeros(mask.shape, jnp.float64).at[indices].set(
            solved.diagnostics.final_maximum_flux_error_column, mode="drop"),
    )
    return SoilWaterMovementResult(restored, diagnostics)


_complete_soil_water_movement = jax.jit(
    DIRECT_HYDROLOGY_NO_DRAINAGE_STAGES.soil_water_movement, static_argnums=(3,),
)


def _gpu_soil_water_movement(topology, inputs, state, parameters):
    if get_opaque_trace_state() == _EAGER_TRACE:
        plan = _static_hydrology_indices(topology)
        if plan is not None:
            return _compact_soil_water_movement(topology, inputs, state, parameters, *plan)
    return _complete_soil_water_movement(topology, inputs, state, parameters)


@cache
def _nested_hydrology_stages(*, for_ad=False) -> HydrologyNoDrainageStageFunctions:

    compiler_options = (
        {"xla_cpu_parallel_codegen_split_count": 1}
        if jax.default_backend() == "cpu" and not for_ad
        else None
    )

    def compiled(function, *, static_argnums=()):
        if for_ad:
            return process_lifetime_jit(function, static_argnums=static_argnums)
        return jax.jit(
            function,
            static_argnums=static_argnums,
            inline=False,
            compiler_options=compiler_options,
        )

    direct = DIRECT_HYDROLOGY_NO_DRAINAGE_STAGES
    snow_workers = (min(64, jax.local_device_count())
                    if jax.default_backend() == "cpu" and not for_ad else 1)
    return HydrologyNoDrainageStageFunctions(
        snow_water=compiled(direct.snow_water),
        surface_hydrology=compiled(direct.surface_hydrology),
        plant_sink=compiled(direct.plant_sink, static_argnums=(3,)),
        soil_water_movement=(
            compiled(direct.soil_water_movement, static_argnums=(3,)) if for_ad
            else _parallel_soil_water_movement
            if jax.default_backend() == "cpu" and jax.local_device_count() > 1
            else _gpu_soil_water_movement if jax.default_backend() == "gpu"
            else compiled(direct.soil_water_movement, static_argnums=(3,))
        ),
        water_table_condensation=compiled(
            direct.water_table_condensation,
            static_argnums=(3,),
        ),
        snow_capping=compiled(direct.snow_capping, static_argnums=(3,)),
        snow_compaction=(
            parallel_snow_stage("compaction", snow_workers)
            if snow_workers > 1 else compiled(direct.snow_compaction, static_argnums=(3,))
        ),
        snow_layer_combining=(
            parallel_snow_stage("combining", snow_workers)
            if snow_workers > 1 else compiled(direct.snow_layer_combining, static_argnums=(2,))
        ),
        snow_layer_division=compiled(
            direct.snow_layer_division,
            static_argnums=(2,),
        ),
        snow_layer_postprocess=compiled(
            direct.snow_layer_postprocess,
            static_argnums=(3,),
        ),
        final_diagnostics=compiled(
            direct.final_diagnostics,
            static_argnums=(3,),
        ),
    )


@cache
def _mapped_hydrology(device_count: int):
    compiler_options = (
        {"xla_cpu_parallel_codegen_split_count": 1}
        if jax.default_backend() == "cpu"
        else None
    )
    return _distributed_vmap(
        _hydrology_shard,
        device_count=device_count,
        axis_name=_HYDROLOGY_AXIS_NAME,
        in_axes=(_TOPOLOGY_AXES, 0, 0, None),
        static_argnums=(3,),
        compiler_options=compiler_options,
    )


def hydrology_no_drainage_parallel(
    topology: SubgridTopology,
    inputs: HydrologyNoDrainageInputs,
    state: HydrologyNoDrainageState,
    parameters: HydrologyNoDrainageParameters,
    *,
    device_count: int,
) -> HydrologyNoDrainageResult:

    plan = plan_column_patch_shards(topology, device_count)
    devices = tuple(jax.local_devices()[:device_count])
    packed = (
        _pack_hydrology_topology(topology, plan),
        _pack_hydrology_inputs(inputs, plan),
        jax.tree.map(lambda value: _pack_column_array(value, plan), state),
    )
    packed = _reshard_tree_for_pmap(
        packed,
        (_TOPOLOGY_AXES, 0, 0),
        devices,
        _HYDROLOGY_AXIS_NAME,
    )
    mapped = _mapped_hydrology(device_count)(*packed, parameters)
    mapped = gather_mapped_result(mapped, devices[0])
    return jax.tree.map(
        lambda value: _unpack_column_array(value, plan),
        mapped,
    )


class HydrologyDriverExecutor:

    def __init__(
        self,
        *,
        jit: bool = True,
        device_count: int = 1,
        stage_completed: Callable[[str], None] | None = None,
    ) -> None:
        if isinstance(device_count, bool) or not isinstance(device_count, int):
            raise TypeError("hydrology device_count must be an integer")
        if device_count < 1:
            raise ValueError("hydrology device_count must be positive")
        if not jit and device_count != 1:
            raise ValueError("parallel hydrology execution requires jit=true")
        self.stage_completed = stage_completed
        self.device_count = device_count
        self.compiled_owners = jit and device_count == 1 and jax.default_backend() == "gpu"
        self.fused_gpu = (
            jit and _USE_FUSED_GPU_HYDROLOGY and device_count == 1
            and jax.default_backend() == "gpu"
        )
        if jit:
            self.core_stages = _nested_hydrology_stages()
            self.core_step = None
        else:
            self.core_stages = None
            self.core_step = hydrology_no_drainage_active_case

    def _advance(self, topology, inputs, state, parameters, filter_topology=None):
        staged_ad = _stage_reverse_ad.get() and not _inside_ad_stage.get()
        if self.fused_gpu and not staged_ad:
            return _compiled_hydrology_transition(
                topology, inputs, state, parameters, filter_topology,
            )
        if self.core_stages is not None and (
            self.device_count == 1 or jax.default_backend() == "cpu"
        ):
            return hydrology_no_drainage_with_stages(
                topology, inputs, state, parameters=parameters,
                stages=_nested_hydrology_stages(for_ad=True) if staged_ad else self.core_stages,
            )
        if self.device_count == 1:
            return self.core_step(topology, inputs, state, parameters=parameters)
        return hydrology_no_drainage_parallel(
            topology, inputs, state, parameters, device_count=self.device_count,
        )

    def __call__(
        self,
        physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
        post: PostTemperatureDriverResult,
        forcing: DatmForcing,
        inputs: HydrologyDriverInputs,
        state: HydrologyNoDrainageState,
        parameters: HydrologyNoDrainageParameters,
        *,
        filter_topology: SubgridTopology | None = None,
    ) -> HydrologyDriverResult:
        topology = physical.soil_temperature.inputs.topology
        assemble = _compiled_hydrology_owners if self.compiled_owners else _hydrology_owners
        assembled, entry_state = assemble(physical, post, forcing, inputs, state)
        topology, assembled, entry_state, parameters = co_locate_array_leaves(
            (topology, assembled, entry_state, parameters),
            jax.devices()[0],
        )
        stage_barrier((topology, assembled, entry_state, parameters))
        hydrology = self._advance(
            topology, assembled, entry_state, parameters, filter_topology,
        )
        result = HydrologyDriverResult(
            inputs=assembled,
            entry_state=entry_state,
            hydrology=hydrology,
        )
        if self.stage_completed is not None:
            stage_barrier(result)
            self.stage_completed("stage360_hydrology_no_drainage")
        return result


__all__ = ["HydrologyDriverExecutor", "hydrology_no_drainage_parallel"]
