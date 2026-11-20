
from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from clm_jax.compilation import jit_unique
from clm_jax.execution import _distributed_vmap, _reshard_tree_for_pmap
from clm_jax.execution_observer import stage_barrier
from clm_jax.forcing.datm import DatmForcing
from clm_jax.io.methane_driver import MethaneDriverNativeInputs
from clm_jax.methane_compact import methane_filter_plan, partitioned_methane
from clm_jax.physics.ecosystem_dynamics_carry import (
    EcosystemDynamicsPostDrainageCarry,
)
from clm_jax.physics.ecosystem_dynamics_post_drainage import (
    EcosystemDynamicsPostDrainageResult,
)
from clm_jax.physics.hydrology_drainage_driver import (
    HydrologyDrainageDriverResult,
)
from clm_jax.physics.hydrology_driver import HydrologyDriverResult
from clm_jax.physics.lake_hydrology_driver import LakeHydrologyDriverResult
from clm_jax.physics.methane_biogeochemistry import (
    MethaneAerenchymaInputs,
    MethaneAerenchymaParameters,
    MethaneAerenchymaResult,
    MethaneAerenchymaState,
    methane_aerenchyma_active_case,
)
from clm_jax.physics.methane_driver_composition import (
    MethaneDriverCarry,
    assemble_methane_driver_inputs,
)
from clm_jax.physics.methane_driver_facade import (
    MethaneDriverActiveInputs,
    MethaneDriverActiveParameters,
    MethaneDriverActiveResult,
    MethaneDriverActiveState,
    methane_driver_active_case,
    methane_driver_active_case_staged,
)
from clm_jax.physics.methane_soil_gas_transport import (
    MethaneSoilGasTransportInputs,
    MethaneSoilGasTransportParameters,
    MethaneSoilGasTransportResult,
    MethaneSoilGasTransportState,
    compose_methane_soil_gas_transport_result,
    methane_soil_gas_transport_competition_active_case,
    methane_soil_gas_transport_saturated_branch_active_case,
    methane_soil_gas_transport_unsaturated_branch_active_case,
)
from clm_jax.physics.physical_driver import (
    PhysicalDriverResult,
    PhysicalDriverRuntimeResult,
)
from clm_jax.runtime import co_locate_array_leaves


def _methane_input_owners(*arguments):
    return assemble_methane_driver_inputs(
        *arguments[:-1], total_water_storage_gridcell=arguments[-1],
    )


_compiled_methane_input_owners = jit_unique(_methane_input_owners, forward_inputs=True)


class MethaneDriverResult(NamedTuple):

    inputs: MethaneDriverActiveInputs
    methane: MethaneDriverActiveResult


class MethaneTransportShardPlan(NamedTuple):

    column_count: int
    device_count: int
    columns_per_device: int
    padding: int


class MethaneAerenchymaShardPlan(NamedTuple):

    column_count: int
    patch_count: int
    device_count: int
    columns_per_device: int
    column_padding: int
    patches_per_device: int
    patch_indices: np.ndarray
    valid_patch_slots: np.ndarray


_AERENCHYMA_GRIDCELL_FIELDS = {
    "atmospheric_methane_concentration_gridcell",
    "atmospheric_oxygen_concentration_gridcell",
}
_AERENCHYMA_PATCH_FIELDS = {
    "methane_patch",
    "patch_column",
    "patch_weight_on_column",
    "vegetated_patch",
    "grass_or_crop_patch",
    "annual_npp_patch",
    "annual_average_aboveground_npp_patch",
    "annual_average_belowground_npp_patch",
    "effective_root_fraction_patch_level",
    "root_fraction_patch_level",
    "transpiration_patch",
    "ground_methane_conductance_patch",
}
_AERENCHYMA_INPUT_AXES = MethaneAerenchymaInputs(
    *(
        None if name in _AERENCHYMA_GRIDCELL_FIELDS else 0
        for name in MethaneAerenchymaInputs._fields
    )
)


_TRANSPORT_GRIDCELL_FIELDS = {
    "atmospheric_methane_gridcell",
    "atmospheric_oxygen_gridcell",
}
_TRANSPORT_INPUT_AXES = MethaneSoilGasTransportInputs(
    *(
        None if name in _TRANSPORT_GRIDCELL_FIELDS else 0
        for name in MethaneSoilGasTransportInputs._fields
    )
)


def plan_methane_aerenchyma_shards(
    column_count: int,
    patch_column,
    device_count: int,
) -> MethaneAerenchymaShardPlan:

    if isinstance(column_count, bool) or not isinstance(column_count, int):
        raise TypeError("methane aerenchyma column_count must be an integer")
    if isinstance(device_count, bool) or not isinstance(device_count, int):
        raise TypeError("methane aerenchyma device_count must be an integer")
    if column_count < 1:
        raise ValueError("methane aerenchyma requires at least one column")
    if device_count < 1:
        raise ValueError("methane aerenchyma device_count must be positive")
    patch_column_array = np.asarray(patch_column, dtype=np.int64)
    if patch_column_array.ndim != 1:
        raise ValueError("methane aerenchyma patch_column must be a vector")
    patch_count = int(patch_column_array.size)
    if patch_count < 1:
        raise ValueError("methane aerenchyma requires at least one patch")
    if np.any((patch_column_array < 0) | (patch_column_array >= column_count)):
        raise ValueError("methane aerenchyma patch_column is out of bounds")

    columns_per_device = (column_count + device_count - 1) // device_count
    owners = patch_column_array // columns_per_device
    owned_patches = [
        np.flatnonzero(owners == device).astype(np.int32, copy=False)
        for device in range(device_count)
    ]
    patches_per_device = max(1, *(indices.size for indices in owned_patches))
    patch_indices = np.zeros(
        (device_count, patches_per_device),
        dtype=np.int32,
    )
    valid_patch_slots = np.zeros_like(patch_indices, dtype=np.bool_)
    for device, indices in enumerate(owned_patches):
        count = int(indices.size)
        patch_indices[device, :count] = indices
        valid_patch_slots[device, :count] = True

    padded_column_count = columns_per_device * device_count
    return MethaneAerenchymaShardPlan(
        column_count=column_count,
        patch_count=patch_count,
        device_count=device_count,
        columns_per_device=columns_per_device,
        column_padding=padded_column_count - column_count,
        patches_per_device=patches_per_device,
        patch_indices=patch_indices,
        valid_patch_slots=valid_patch_slots,
    )


def _pack_aerenchyma_column(
    value,
    plan: MethaneAerenchymaShardPlan,
) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim < 1 or array.shape[0] != plan.column_count:
        raise ValueError(
            f"methane aerenchyma column leaf must start with {plan.column_count}, got {array.shape}"
        )
    if plan.column_padding:
        array = jnp.concatenate(
            (array, jnp.repeat(array[-1:], plan.column_padding, axis=0)),
            axis=0,
        )
    return jnp.reshape(
        array,
        (plan.device_count, plan.columns_per_device, *array.shape[1:]),
    )


def _unpack_aerenchyma_column(
    value,
    plan: MethaneAerenchymaShardPlan,
) -> jax.Array:
    array = jnp.asarray(value)
    expected = (plan.device_count, plan.columns_per_device)
    if array.ndim < 2 or array.shape[:2] != expected:
        raise ValueError(
            f"mapped methane aerenchyma column leaf must start with {expected}, got {array.shape}"
        )
    flattened = jnp.reshape(
        array,
        (plan.device_count * plan.columns_per_device, *array.shape[2:]),
    )
    return flattened[: plan.column_count]


def _pack_aerenchyma_patch(
    value,
    plan: MethaneAerenchymaShardPlan,
) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim < 1 or array.shape[0] != plan.patch_count:
        raise ValueError(
            f"methane aerenchyma patch leaf must start with {plan.patch_count}, got {array.shape}"
        )
    return array[jnp.asarray(plan.patch_indices, dtype=jnp.int32)]


def _unpack_aerenchyma_patch(
    value,
    plan: MethaneAerenchymaShardPlan,
) -> jax.Array:
    array = jnp.asarray(value)
    expected = (plan.device_count, plan.patches_per_device)
    if array.ndim < 2 or array.shape[:2] != expected:
        raise ValueError(
            f"mapped methane aerenchyma patch leaf must start with {expected}, got {array.shape}"
        )
    flattened = jnp.reshape(
        array,
        (plan.device_count * plan.patches_per_device, *array.shape[2:]),
    )
    valid = jnp.asarray(plan.valid_patch_slots.reshape(-1), dtype=jnp.bool_)
    indices = jnp.asarray(plan.patch_indices.reshape(-1), dtype=jnp.int32)
    output = jnp.zeros(
        (plan.patch_count, *array.shape[2:]),
        dtype=array.dtype,
    )
    return output.at[indices[valid]].set(flattened[valid])


def _pack_aerenchyma_inputs(
    inputs: MethaneAerenchymaInputs,
    plan: MethaneAerenchymaShardPlan,
) -> MethaneAerenchymaInputs:
    packed_values = []
    for name, value in zip(
        MethaneAerenchymaInputs._fields,
        inputs,
        strict=True,
    ):
        if name in _AERENCHYMA_GRIDCELL_FIELDS:
            packed_values.append(value)
        elif name in _AERENCHYMA_PATCH_FIELDS:
            packed_values.append(_pack_aerenchyma_patch(value, plan))
        else:
            packed_values.append(_pack_aerenchyma_column(value, plan))
    packed = MethaneAerenchymaInputs(*packed_values)

    valid_columns = jnp.arange(
        plan.device_count * plan.columns_per_device,
        dtype=jnp.int32,
    ) < jnp.int32(plan.column_count)
    valid_columns = jnp.reshape(
        valid_columns,
        (plan.device_count, plan.columns_per_device),
    )
    valid_patches = jnp.asarray(plan.valid_patch_slots, dtype=jnp.bool_)
    column_starts = (
        jnp.arange(plan.device_count, dtype=jnp.int32) * jnp.int32(plan.columns_per_device)
    )[:, None]
    local_patch_column = packed.patch_column - column_starts
    local_patch_column = jnp.where(
        valid_patches,
        local_patch_column,
        jnp.int32(0),
    )
    return packed._replace(
        methane_column=packed.methane_column & valid_columns,
        methane_patch=packed.methane_patch & valid_patches,
        patch_column=local_patch_column,
    )


def _unpack_aerenchyma_result(
    mapped: MethaneAerenchymaResult,
    plan: MethaneAerenchymaShardPlan,
) -> MethaneAerenchymaResult:
    state = jax.tree_util.tree_map(
        lambda value: _unpack_aerenchyma_column(value, plan),
        mapped.state,
    )
    diagnostics = mapped.diagnostics._replace(
        unsaturated_tiller_area_patch=_unpack_aerenchyma_patch(
            mapped.diagnostics.unsaturated_tiller_area_patch,
            plan,
        ),
        saturated_tiller_area_patch=_unpack_aerenchyma_patch(
            mapped.diagnostics.saturated_tiller_area_patch,
            plan,
        ),
        npp_root_fraction_patch=_unpack_aerenchyma_patch(
            mapped.diagnostics.npp_root_fraction_patch,
            plan,
        ),
        methane_column_count=jnp.sum(
            mapped.diagnostics.methane_column_count,
            dtype=jnp.int32,
        ),
        methane_patch_count=jnp.sum(
            mapped.diagnostics.methane_patch_count,
            dtype=jnp.int32,
        ),
    )
    return mapped._replace(state=state, diagnostics=diagnostics)


@cache
def _mapped_methane_aerenchyma(device_count: int):
    compiler_options = (
        {"xla_cpu_parallel_codegen_split_count": 1}
        if jax.default_backend() == "cpu"
        else None
    )
    return _distributed_vmap(
        methane_aerenchyma_active_case,
        device_count=device_count,
        axis_name="clm_methane_aerenchyma_device",
        in_axes=(_AERENCHYMA_INPUT_AXES, 0, None),
        compiler_options=compiler_options,
    )


def methane_aerenchyma_parallel(
    inputs: MethaneAerenchymaInputs,
    state: MethaneAerenchymaState,
    parameters: MethaneAerenchymaParameters,
    *,
    device_count: int,
) -> MethaneAerenchymaResult:

    plan = plan_methane_aerenchyma_shards(
        int(inputs.methane_column.shape[0]),
        inputs.patch_column,
        device_count,
    )
    devices = tuple(jax.local_devices()[: plan.device_count])
    if len(devices) != plan.device_count:
        raise ValueError(
            f"requested {plan.device_count} methane devices but JAX exposes "
            f"only {jax.local_device_count()}"
        )
    packed_inputs = _pack_aerenchyma_inputs(inputs, plan)
    packed_state = jax.tree_util.tree_map(
        lambda value: _pack_aerenchyma_column(value, plan),
        state,
    )
    packed_inputs, packed_state, parameters = _reshard_tree_for_pmap(
        (packed_inputs, packed_state, parameters),
        (_AERENCHYMA_INPUT_AXES, 0, None),
        devices,
        "clm_methane_aerenchyma_device",
    )
    mapped = _mapped_methane_aerenchyma(plan.device_count)(
        packed_inputs,
        packed_state,
        parameters,
    )
    mapped = co_locate_array_leaves(mapped, devices[0])
    return _unpack_aerenchyma_result(mapped, plan)


def plan_methane_transport_shards(
    column_count: int,
    device_count: int,
) -> MethaneTransportShardPlan:

    if isinstance(column_count, bool) or not isinstance(column_count, int):
        raise TypeError("methane transport column_count must be an integer")
    if isinstance(device_count, bool) or not isinstance(device_count, int):
        raise TypeError("methane transport device_count must be an integer")
    if column_count < 1:
        raise ValueError("methane transport requires at least one column")
    if device_count < 1:
        raise ValueError("methane transport device_count must be positive")
    columns_per_device = (column_count + device_count - 1) // device_count
    padded_count = columns_per_device * device_count
    return MethaneTransportShardPlan(
        column_count=column_count,
        device_count=device_count,
        columns_per_device=columns_per_device,
        padding=padded_count - column_count,
    )


def _pack_transport_column(value, plan: MethaneTransportShardPlan) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim < 1 or array.shape[0] != plan.column_count:
        raise ValueError(
            f"methane transport column leaf must start with {plan.column_count}, got {array.shape}"
        )
    if plan.padding:
        array = jnp.concatenate(
            (array, jnp.repeat(array[-1:], plan.padding, axis=0)),
            axis=0,
        )
    return jnp.reshape(
        array,
        (plan.device_count, plan.columns_per_device, *array.shape[1:]),
    )


def _unpack_transport_column(value, plan: MethaneTransportShardPlan) -> jax.Array:
    array = jnp.asarray(value)
    expected = (plan.device_count, plan.columns_per_device)
    if array.ndim < 2 or array.shape[:2] != expected:
        raise ValueError(
            f"mapped methane column leaf must start with {expected}, got {array.shape}"
        )
    flattened = jnp.reshape(
        array,
        (plan.device_count * plan.columns_per_device, *array.shape[2:]),
    )
    return flattened[: plan.column_count]


def _pack_transport_inputs(
    inputs: MethaneSoilGasTransportInputs,
    plan: MethaneTransportShardPlan,
) -> MethaneSoilGasTransportInputs:
    packed = MethaneSoilGasTransportInputs(
        *(
            value if name in _TRANSPORT_GRIDCELL_FIELDS else _pack_transport_column(value, plan)
            for name, value in zip(
                MethaneSoilGasTransportInputs._fields,
                inputs,
                strict=True,
            )
        )
    )
    valid = jnp.arange(
        plan.device_count * plan.columns_per_device,
        dtype=jnp.int32,
    ) < jnp.int32(plan.column_count)
    valid = jnp.reshape(valid, (plan.device_count, plan.columns_per_device))
    return packed._replace(methane_column=packed.methane_column & valid)


def _transport_unmap_helpers(
    plan: MethaneTransportShardPlan,
    memory_barrier: Callable[[], object] | None,
):
    def complete_group(tree):
        placed = co_locate_array_leaves(tree, jax.devices()[0])
        if memory_barrier is not None:
            stage_barrier(placed)
            memory_barrier()
        return placed

    def columns(tree):
        placed = complete_group(tree)
        return jax.tree_util.tree_map(
            lambda value: _unpack_transport_column(value, plan),
            placed,
        )
    return complete_group, columns


def _unpack_transport_competition(
    mapped,
    plan: MethaneTransportShardPlan,
    memory_barrier: Callable[[], object] | None = None,
):

    complete_group, columns = _transport_unmap_helpers(plan, memory_barrier)
    mapped = complete_group(mapped)
    diagnostics = mapped.diagnostics
    competition = mapped._replace(
        state=columns(mapped.state),
        diagnostics=diagnostics._replace(
            unsaturated_initial_oxygen_demand_column_level=(
                _unpack_transport_column(
                    diagnostics.unsaturated_initial_oxygen_demand_column_level,
                    plan,
                )
            ),
            saturated_initial_oxygen_demand_column_level=(
                _unpack_transport_column(
                    diagnostics.saturated_initial_oxygen_demand_column_level,
                    plan,
                )
            ),
            unsaturated_initial_methane_demand_column_level=(
                _unpack_transport_column(
                    diagnostics.unsaturated_initial_methane_demand_column_level,
                    plan,
                )
            ),
            saturated_initial_methane_demand_column_level=(
                _unpack_transport_column(
                    diagnostics.saturated_initial_methane_demand_column_level,
                    plan,
                )
            ),
            methane_column_count=jnp.sum(
                diagnostics.methane_column_count,
                dtype=jnp.int32,
            ),
        ),
    )
    return complete_group(competition)


def _unpack_transport_branch(
    mapped,
    plan: MethaneTransportShardPlan,
    memory_barrier: Callable[[], object] | None = None,
):

    complete_group, columns = _transport_unmap_helpers(plan, memory_barrier)
    mapped = complete_group(mapped)
    return complete_group(
        mapped._replace(
            state=columns(mapped.state),
            ground_methane_conductance_column=_unpack_transport_column(
                mapped.ground_methane_conductance_column,
                plan,
            ),
            diagnostics=columns(mapped.diagnostics),
        )
    )


def _unpack_transport_result(
    mapped: MethaneSoilGasTransportResult,
    plan: MethaneTransportShardPlan,
    memory_barrier: Callable[[], object] | None = None,
) -> MethaneSoilGasTransportResult:

    complete_group, columns = _transport_unmap_helpers(plan, memory_barrier)
    mapped = complete_group(mapped)

    competition = _unpack_transport_competition(
        mapped.diagnostics.competition,
        plan,
        memory_barrier,
    )
    diagnostics = complete_group(
        mapped.diagnostics._replace(
            competition=competition,
            unsaturated_henry_ratio_column_level_species=_unpack_transport_column(
                mapped.diagnostics.unsaturated_henry_ratio_column_level_species,
                plan,
            ),
            saturated_henry_ratio_column_level_species=_unpack_transport_column(
                mapped.diagnostics.saturated_henry_ratio_column_level_species,
                plan,
            ),
            unsaturated_effective_storage_column_level_species=(
                _unpack_transport_column(
                    mapped.diagnostics.unsaturated_effective_storage_column_level_species,
                    plan,
                )
            ),
            saturated_effective_storage_column_level_species=(
                _unpack_transport_column(
                    mapped.diagnostics.saturated_effective_storage_column_level_species,
                    plan,
                )
            ),
            unsaturated_diffusivity_column_level_species=_unpack_transport_column(
                mapped.diagnostics.unsaturated_diffusivity_column_level_species,
                plan,
            ),
            saturated_diffusivity_column_level_species=_unpack_transport_column(
                mapped.diagnostics.saturated_diffusivity_column_level_species,
                plan,
            ),
            unsaturated_species_ground_conductance_column_species=(
                _unpack_transport_column(
                    mapped.diagnostics.unsaturated_species_ground_conductance_column_species,
                    plan,
                )
            ),
            saturated_species_ground_conductance_column_species=(
                _unpack_transport_column(
                    mapped.diagnostics.saturated_species_ground_conductance_column_species,
                    plan,
                )
            ),
            unsaturated_raw_methane_conservation_error_column=(
                _unpack_transport_column(
                    mapped.diagnostics.unsaturated_raw_methane_conservation_error_column,
                    plan,
                )
            ),
            saturated_raw_methane_conservation_error_column=(
                _unpack_transport_column(
                    mapped.diagnostics.saturated_raw_methane_conservation_error_column,
                    plan,
                )
            ),
            methane_conservation_violation_column=_unpack_transport_column(
                mapped.diagnostics.methane_conservation_violation_column,
                plan,
            ),
            methane_column_count=jnp.sum(
                mapped.diagnostics.methane_column_count,
                dtype=jnp.int32,
            ),
        )
    )
    return mapped._replace(state=columns(mapped.state), diagnostics=diagnostics)


def _methane_transport_devices(device_count: int) -> tuple[jax.Device, ...]:
    devices = tuple(jax.local_devices()[:device_count])
    if len(devices) != device_count:
        raise ValueError(
            f"requested {device_count} methane devices but JAX exposes "
            f"only {jax.local_device_count()}"
        )
    return devices


@cache
def _mapped_methane_transport_competition(device_count: int):
    compiler_options = (
        {"xla_cpu_parallel_codegen_split_count": 1}
        if jax.default_backend() == "cpu"
        else None
    )
    return _distributed_vmap(
        methane_soil_gas_transport_competition_active_case,
        device_count=device_count,
        axis_name="clm_methane_column_device",
        in_axes=(_TRANSPORT_INPUT_AXES, 0, None),
        compiler_options=compiler_options,
    )


@cache
def _mapped_methane_transport_unsaturated(device_count: int):
    compiler_options = (
        {"xla_cpu_parallel_codegen_split_count": 1}
        if jax.default_backend() == "cpu"
        else None
    )
    return _distributed_vmap(
        methane_soil_gas_transport_unsaturated_branch_active_case,
        device_count=device_count,
        axis_name="clm_methane_column_device",
        in_axes=(_TRANSPORT_INPUT_AXES, 0, 0, None),
        compiler_options=compiler_options,
    )


@cache
def _mapped_methane_transport_saturated(device_count: int):
    compiler_options = (
        {"xla_cpu_parallel_codegen_split_count": 1}
        if jax.default_backend() == "cpu"
        else None
    )
    return _distributed_vmap(
        methane_soil_gas_transport_saturated_branch_active_case,
        device_count=device_count,
        axis_name="clm_methane_column_device",
        in_axes=(_TRANSPORT_INPUT_AXES, 0, 0, 0, None),
        compiler_options=compiler_options,
    )


def methane_soil_gas_transport_parallel(
    inputs: MethaneSoilGasTransportInputs,
    state: MethaneSoilGasTransportState,
    parameters: MethaneSoilGasTransportParameters,
    *,
    device_count: int,
    memory_barrier: Callable[[], object] | None = None,
) -> MethaneSoilGasTransportResult:

    plan = plan_methane_transport_shards(
        int(inputs.methane_column.shape[0]),
        device_count,
    )
    devices = tuple(jax.local_devices()[: plan.device_count])
    if len(devices) != plan.device_count:
        raise ValueError(
            f"requested {plan.device_count} methane devices but JAX exposes "
            f"only {jax.local_device_count()}"
        )
    packed_inputs = _pack_transport_inputs(inputs, plan)
    packed_state = jax.tree_util.tree_map(
        lambda value: _pack_transport_column(value, plan),
        state,
    )
    packed_inputs, packed_state, parameters = _reshard_tree_for_pmap(
        (packed_inputs, packed_state, parameters),
        (_TRANSPORT_INPUT_AXES, 0, None),
        devices,
        "clm_methane_column_device",
    )

    def complete_stage(result):
        if memory_barrier is not None:
            stage_barrier(result)
            memory_barrier()
        return result

    competition = complete_stage(
        _mapped_methane_transport_competition(plan.device_count)(
            packed_inputs,
            packed_state,
            parameters,
        )
    )
    unsaturated = complete_stage(
        _mapped_methane_transport_unsaturated(plan.device_count)(
            packed_inputs,
            packed_state,
            competition.state,
            parameters,
        )
    )
    saturated = complete_stage(
        _mapped_methane_transport_saturated(plan.device_count)(
            packed_inputs,
            packed_state,
            competition.state,
            unsaturated.ground_methane_conductance_column,
            parameters,
        )
    )
    del packed_inputs, packed_state, parameters
    if memory_barrier is not None:
        memory_barrier()

    competition = _unpack_transport_competition(
        competition,
        plan,
        memory_barrier,
    )
    unsaturated = _unpack_transport_branch(
        unsaturated,
        plan,
        memory_barrier,
    )
    saturated = _unpack_transport_branch(
        saturated,
        plan,
        memory_barrier,
    )
    result = compose_methane_soil_gas_transport_result(
        inputs,
        competition,
        unsaturated,
        saturated,
    )
    return complete_stage(result)


class MethaneDriverExecutor:

    def __init__(
        self,
        *,
        jit: bool = True,
        device_count: int = 1,
        combine_gpu_stages: bool = False,
        stage_completed: Callable[[str], None] | None = None,
        memory_barrier: Callable[[], object] | None = None,
    ) -> None:
        if isinstance(device_count, bool) or not isinstance(device_count, int):
            raise TypeError("methane driver device_count must be an integer")
        if device_count < 1:
            raise ValueError("methane driver device_count must be positive")
        self.stage_completed = stage_completed
        self.memory_barrier = memory_barrier
        self.device_count = device_count
        self.jit = jit
        self.compiled_owners = jit and device_count == 1 and jax.default_backend() == "gpu"
        self.combine_gpu_stages = combine_gpu_stages and self.compiled_owners
        self.aerenchyma_kernel = None
        self.transport_kernel = None
        self.step = methane_driver_active_case_staged if jit else methane_driver_active_case
        if jit and device_count > 1:
            self.aerenchyma_kernel = lambda inputs, state, parameters: methane_aerenchyma_parallel(
                inputs,
                state,
                parameters,
                device_count=self.device_count,
            )
            self.transport_kernel = lambda inputs, state, parameters: (
                methane_soil_gas_transport_parallel(
                    inputs,
                    state,
                    parameters,
                    device_count=self.device_count,
                    memory_barrier=self.memory_barrier,
                )
            )

    def __call__(
        self,
        native: MethaneDriverNativeInputs,
        state: MethaneDriverActiveState,
        parameters: MethaneDriverActiveParameters,
        forcing: DatmForcing,
        physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
        hydrology: HydrologyDriverResult,
        lake_hydrology: LakeHydrologyDriverResult,
        drainage: HydrologyDrainageDriverResult,
        post_drainage: EcosystemDynamicsPostDrainageResult,
        ecosystem_carry: EcosystemDynamicsPostDrainageCarry,
        methane_carry: MethaneDriverCarry,
        *,
        total_water_storage_gridcell: jax.Array | None = None,
    ) -> MethaneDriverResult:
        assemble = _compiled_methane_input_owners if self.compiled_owners else _methane_input_owners
        inputs = assemble(
            native,
            state,
            forcing,
            physical,
            hydrology,
            lake_hydrology,
            drainage,
            post_drainage,
            ecosystem_carry,
            methane_carry,
            total_water_storage_gridcell,
        )
        inputs, state, parameters = co_locate_array_leaves(
            (inputs, state, parameters),
            jax.devices()[0],
        )
        stage_barrier((inputs, state, parameters))
        plan = (methane_filter_plan(inputs)
                if self.combine_gpu_stages and self.memory_barrier is None else None)
        if plan is not None:
            methane = partitioned_methane(inputs, state, parameters, plan)
        elif not self.jit:
            methane = self.step(inputs, state, parameters)
        else:
            methane = self.step(
                inputs,
                state,
                parameters,
                aerenchyma_kernel=self.aerenchyma_kernel,
                transport_kernel=self.transport_kernel,
                stage_completed=self.stage_completed,
                memory_barrier=self.memory_barrier,
            )
        result = MethaneDriverResult(inputs=inputs, methane=methane)
        if self.stage_completed is not None:
            stage_barrier(result)
            self.stage_completed("stage500_methane")
        return result


__all__ = [
    "MethaneAerenchymaShardPlan",
    "MethaneDriverExecutor",
    "MethaneDriverResult",
    "MethaneTransportShardPlan",
    "methane_aerenchyma_parallel",
    "methane_soil_gas_transport_parallel",
    "plan_methane_aerenchyma_shards",
    "plan_methane_transport_shards",
]
