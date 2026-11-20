
from __future__ import annotations

from collections.abc import Callable
from functools import cache

import jax
import jax.numpy as jnp

from clm_jax.compilation import jit_unique, process_lifetime_jit, materialize_outputs
from clm_jax.execution_observer import stage_barrier
from clm_jax.forcing.datm import DatmForcing
from clm_jax.physics.hydrology_driver import HydrologyDriverResult
from clm_jax.physics.lake_hydrology import (
    LakeHydrologyDiagnostics,
    LakeHydrologyParameters,
    LakeHydrologyResult,
    LakeHydrologyState,
    lake_hydrology,
)
from clm_jax.physics.lake_hydrology_driver import (
    LakeHydrologyDriverInputs,
    LakeHydrologyDriverResult,
    assemble_lake_hydrology_inputs,
    assemble_lake_hydrology_state,
)
from clm_jax.physics.physical_driver import (
    PhysicalDriverResult,
    PhysicalDriverRuntimeResult,
)
from clm_jax.physics.post_hydrology_no_drainage_driver import (
    PostHydrologyNoDrainageResult,
)
from clm_jax.physics.post_temperature_driver import PostTemperatureDriverResult


def compact_lake_hydrology(
    topology,
    inputs,
    state,
    lake_columns,
    lake_patches,
    *,
    parameters,
):
    if lake_columns.size == 0:
        no_column = jnp.zeros_like(topology.column.active, dtype=jnp.bool_)
        no_patch = jnp.zeros_like(topology.patch.active, dtype=jnp.bool_)
        zero_column = jnp.zeros_like(state.water.snow_depth_column)
        return LakeHydrologyResult(
            state,
            LakeHydrologyDiagnostics(
                no_column,
                no_patch,
                no_column,
                no_column,
                no_column,
                no_column,
                no_column,
                no_column,
                no_column,
                zero_column,
                zero_column,
            ),
        )

    def field_name(path):
        return path[-1].name

    def indices(path):
        return lake_patches if field_name(path).endswith("_patch") else lake_columns

    def gather_input(path, value):
        return value if "_gridcell" in field_name(path) else value[indices(path)]

    compact_inputs = jax.tree_util.tree_map_with_path(gather_input, inputs)
    compact_state = jax.tree_util.tree_map_with_path(
        lambda path, value: value[indices(path)],
        state,
    )
    local_column = jnp.full(topology.column.active.shape, -1, dtype=jnp.int32)
    local_column = local_column.at[lake_columns].set(
        jnp.arange(lake_columns.size, dtype=jnp.int32),
    )
    patch = jax.tree.map(lambda value: value[lake_patches], topology.patch)
    compact_topology = topology._replace(
        column=jax.tree.map(lambda value: value[lake_columns], topology.column),
        patch=patch._replace(column=local_column[patch.column]),
    )
    result = lake_hydrology(
        compact_topology,
        compact_inputs,
        compact_state,
        parameters=parameters,
    )
    restored_state = jax.tree_util.tree_map_with_path(
        lambda path, previous, updated: previous.at[indices(path)].set(updated),
        state,
        result.state,
    )

    def restore_diagnostic(path, updated):
        count = (
            topology.patch.active.size
            if field_name(path).endswith("_patch")
            else topology.column.active.size
        )
        return (
            jnp.zeros((count, *updated.shape[1:]), dtype=updated.dtype)
            .at[indices(path)]
            .set(updated)
        )

    diagnostics: LakeHydrologyDiagnostics = jax.tree_util.tree_map_with_path(
        restore_diagnostic,
        result.diagnostics,
    )
    return LakeHydrologyResult(restored_state, diagnostics)


def _assemble_lake_entry(
    physical, post_temperature, hydrology, post_hydrology, forcing, inputs, state,
):
    return (
        assemble_lake_hydrology_inputs(
            physical, post_temperature, hydrology, forcing, inputs,
        ),
        assemble_lake_hydrology_state(
            physical, post_temperature, hydrology, post_hydrology, inputs, state,
        ),
    )


_compiled_lake_entry = jit_unique(_assemble_lake_entry, forward_inputs=True)


def _lake_transition(
    physical, post_temperature, hydrology, post_hydrology, forcing, inputs, state,
    *, parameters,
):
    entry_inputs, entry_state = materialize_outputs(_assemble_lake_entry)(
        physical, post_temperature, hydrology, post_hydrology, forcing, inputs, state,
    )
    result = compact_lake_hydrology(
        physical.soil_temperature.inputs.topology, entry_inputs, entry_state,
        inputs.lake_column_index, inputs.lake_patch_index, parameters=parameters,
    )
    return LakeHydrologyDriverResult(entry_inputs, entry_state, result)


@cache
def _compiled_lake_transition(parameters):
    def transition(*arguments):
        return _lake_transition(*arguments, parameters=parameters)

    return jit_unique(transition, forward_inputs=True)


class LakeHydrologyDriverExecutor:

    def __init__(
        self,
        *,
        jit: bool = True,
        stage_completed: Callable[[str], None] | None = None,
    ) -> None:
        self.stage_completed = stage_completed
        self.compact = jit
        self.assemble_entry = _compiled_lake_entry if jit else _assemble_lake_entry
        if jit:
            self.core_step = process_lifetime_jit(
                compact_lake_hydrology,
                static_argnames=("parameters",),
            )
        else:
            self.core_step = lake_hydrology

    def __call__(
        self,
        physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
        post_temperature: PostTemperatureDriverResult,
        hydrology: HydrologyDriverResult,
        post_hydrology: PostHydrologyNoDrainageResult,
        forcing: DatmForcing,
        inputs: LakeHydrologyDriverInputs,
        state: LakeHydrologyState,
        parameters: LakeHydrologyParameters,
    ) -> LakeHydrologyDriverResult:
        if self.compact and jax.default_backend() == "gpu":
            result = _compiled_lake_transition(parameters)(
                physical, post_temperature, hydrology, post_hydrology,
                forcing, inputs, state,
            )
            if self.stage_completed is not None:
                stage_barrier(result)
                self.stage_completed("stage380_lake_hydrology")
            return result
        entry_inputs, entry_state = self.assemble_entry(
            physical, post_temperature, hydrology, post_hydrology, forcing, inputs, state,
        )
        lake = self.core_step(
            physical.soil_temperature.inputs.topology,
            entry_inputs,
            entry_state,
            *((inputs.lake_column_index, inputs.lake_patch_index) if self.compact else ()),
            parameters=parameters,
        )
        result = LakeHydrologyDriverResult(
            inputs=entry_inputs,
            entry_state=entry_state,
            hydrology=lake,
        )
        if self.stage_completed is not None:
            stage_barrier(result)
            self.stage_completed("stage380_lake_hydrology")
        return result


__all__ = ["LakeHydrologyDriverExecutor"]
