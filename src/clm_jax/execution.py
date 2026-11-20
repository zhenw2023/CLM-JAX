
from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any, NamedTuple
from weakref import ref

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from clm_jax.compilation import jit_discrete_parameters, jit_unique, process_lifetime_jit
from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.execution_observer import stage_barrier
from clm_jax.io.restart import (
    LANDUNIT_LAKE,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    ColumnTopology,
    PatchTopology,
    SubgridTopology,
)
from clm_jax.physics.canopy_flux_driver import (
    CanopyFluxDriverCall,
    CanopyFluxDriverResult,
    CanopyFluxDriverState,
    CanopyFluxDustFields,
    CanopyFluxEcosystemFields,
    CanopyFluxRuntimeResult,
    CanopyFluxSoilTemperatureFields,
    CanopyFluxVOCFields,
    assemble_canopy_flux_driver_call,
    complete_canopy_flux_driver_call,
    retain_inactive_plant_state,
)
from clm_jax.physics.canopy_fluxes import (
    CanopyFluxesActiveParameters,
    CanopyFluxesActiveResult,
    CanopyFluxesLunaInputs,
    canopy_fluxes_active_case,
    canopy_iteration_work,
)
from clm_jax.physics.converged_iteration import gpu_convergence_scope
from clm_jax.physics.lake_flux_driver import (
    LakeFluxDriverCall,
    LakeFluxDriverResult,
    assemble_lake_flux_driver_call,
    complete_lake_flux_driver_call,
)
from clm_jax.physics.lake_fluxes import LakeFluxResult, lake_fluxes
from clm_jax.physics.lake_temperature import LakeTemperatureResult, lake_temperature
from clm_jax.physics.lake_temperature_driver import (
    LakeTemperatureDriverCall,
    LakeTemperatureDriverResult,
    assemble_lake_temperature_driver_call,
)
from clm_jax.physics.luna import LunaCapacityInputs
from clm_jax.physics.soil_temperature import (
    GroundHeatFluxInputs,
    GroundHeatFluxPatchState,
    GroundHeatFluxResult,
    SoilHeatDiffusionInputs,
    SoilSnowPhaseChangeInputs,
    SoilTemperatureFinalDiagnosticsInputs,
    SoilTemperatureLinearSolveInputs,
    SoilTemperatureMatrixInputs,
    SoilTemperaturePostSolveFluxInputs,
    SoilTemperatureRHSInputs,
    SoilTemperatureSolutionUpdateInputs,
    SoilThermalPropertyInputs,
    StandingWaterPhaseChangeInputs,
)
from clm_jax.physics.soil_temperature_driver import (
    SoilTemperatureDriverInputs,
    SoilTemperatureDriverResult,
    assemble_soil_temperature_driver_inputs,
)
from clm_jax.physics.soil_temperature_step import (
    SoilTemperatureActiveInputs,
    SoilTemperatureActiveResult,
    SoilTemperatureColumnResult,
    SoilTemperatureRuntimeColumnResult,
    SoilTemperatureRuntimeResult,
    assemble_coupled_building_temperature_inputs,
    soil_temperature_column_case,
    soil_temperature_runtime_column_case,
)
from clm_jax.physics.solar_runtime import (
    solar_declination_native_runtime as solar_declination_runtime,
)
from clm_jax.physics.urban_building_temperature import (
    NUMBER_OF_BUILDING_EQUATIONS,
    UrbanBuildingTemperatureResult,
    active_building_temperature,
    building_temperature,
)
from clm_jax.physics.urban_flux_driver import (
    UrbanFluxDriverCall,
    UrbanFluxDriverResult,
    UrbanFluxSoilTemperatureFields,
    assemble_urban_flux_driver_call,
    complete_urban_flux_driver_call,
)
from clm_jax.physics.urban_fluxes import (
    UrbanSurfaceFluxResult,
    UrbanTurbulenceResult,
    urban_surface_fluxes,
    urban_turbulent_fluxes,
)
from clm_jax.physics.voc import (
    VOCEmissionResult,
    VOCInputs,
    VOCParameters,
    VOCState,
    voc_emission,
)
from clm_jax.physics.voc_driver import (
    VOCDriverInputs,
    VOCDriverResult,
    assemble_voc_driver_inputs,
)
from clm_jax.runtime import gather_mapped_result
from clm_jax.trajectory import noleap_calendar_day_from_clock


class PatchShardPlan(NamedTuple):

    patch_count: int
    device_count: int
    patches_per_device: int
    padding: int


class ColumnPatchShardPlan(NamedTuple):

    column_count: int
    patch_count: int
    device_count: int
    columns_per_device: int
    column_padding: int
    patches_per_device: int
    patch_indices: jax.Array
    patch_valid: jax.Array
    valid_patch_slots: jax.Array
    valid_patch_indices: jax.Array


def plan_patch_shards(patch_count: int, device_count: int) -> PatchShardPlan:

    if patch_count < 1:
        raise ValueError("patch-parallel execution requires at least one patch")
    if device_count < 1:
        raise ValueError("device_count must be positive")
    patches_per_device = (patch_count + device_count - 1) // device_count
    padded_count = patches_per_device * device_count
    return PatchShardPlan(
        patch_count=patch_count,
        device_count=device_count,
        patches_per_device=patches_per_device,
        padding=padded_count - patch_count,
    )


def plan_column_patch_shards(
    topology: SubgridTopology,
    device_count: int,
) -> ColumnPatchShardPlan:

    column_count = int(topology.column.active.shape[0])
    patch_count = int(topology.patch.active.shape[0])
    if column_count < 1 or patch_count < 1:
        raise ValueError("column/patch execution requires nonempty native topology")
    if device_count < 1:
        raise ValueError("device_count must be positive")
    columns_per_device = (column_count + device_count - 1) // device_count
    padded_column_count = columns_per_device * device_count
    patch_column = np.asarray(topology.patch.column, dtype=np.int64)
    if patch_column.shape != (patch_count,):
        raise ValueError("patch column map does not match the native patch count")
    if np.any((patch_column < 0) | (patch_column >= column_count)):
        raise ValueError("patch topology refers to a column outside the native domain")

    owner = np.minimum(patch_column // columns_per_device, device_count - 1)
    owned = [np.flatnonzero(owner == device) for device in range(device_count)]
    patches_per_device = max(indices.size for indices in owned)
    patch_indices = np.zeros((device_count, patches_per_device), dtype=np.int32)
    patch_valid = np.zeros((device_count, patches_per_device), dtype=bool)
    for device, indices in enumerate(owned):
        if indices.size:
            patch_indices[device, : indices.size] = indices.astype(np.int32)
            patch_indices[device, indices.size :] = indices[-1]
            patch_valid[device, : indices.size] = True

    flat_valid = np.flatnonzero(patch_valid.reshape(-1)).astype(np.int32)
    flat_native = patch_indices.reshape(-1)[flat_valid]
    if flat_native.size != patch_count or np.unique(flat_native).size != patch_count:
        raise ValueError("column shards must own every native patch exactly once")
    return ColumnPatchShardPlan(
        column_count=column_count,
        patch_count=patch_count,
        device_count=device_count,
        columns_per_device=columns_per_device,
        column_padding=padded_column_count - column_count,
        patches_per_device=patches_per_device,
        patch_indices=jnp.asarray(patch_indices, dtype=jnp.int32),
        patch_valid=jnp.asarray(patch_valid, dtype=jnp.bool_),
        valid_patch_slots=jnp.asarray(flat_valid, dtype=jnp.int32),
        valid_patch_indices=jnp.asarray(flat_native, dtype=jnp.int32),
    )


def _pack_column_array(value: Any, plan: ColumnPatchShardPlan) -> jax.Array:

    array = jnp.asarray(value)
    if array.ndim < 1 or array.shape[0] != plan.column_count:
        raise ValueError(
            "every column-sharded leaf must have native leading dimension "
            f"{plan.column_count}, got {array.shape}"
        )
    if plan.column_padding:
        array = jnp.concatenate(
            (array, jnp.repeat(array[-1:], plan.column_padding, axis=0)),
            axis=0,
        )
    return jnp.reshape(
        array,
        (
            plan.device_count,
            plan.columns_per_device,
            *array.shape[1:],
        ),
    )


def _unpack_column_array(value: Any, plan: ColumnPatchShardPlan) -> jax.Array:

    array = jnp.asarray(value)
    expected = (plan.device_count, plan.columns_per_device)
    if array.ndim < 2 or array.shape[:2] != expected:
        raise ValueError(f"mapped column result must start with {expected}, got {array.shape}")
    flattened = jnp.reshape(
        array,
        (plan.device_count * plan.columns_per_device, *array.shape[2:]),
    )
    return flattened[: plan.column_count]


def _pack_owned_patch_array(value: Any, plan: ColumnPatchShardPlan) -> jax.Array:

    array = jnp.asarray(value)
    if array.ndim < 1 or array.shape[0] != plan.patch_count:
        raise ValueError(
            "every owned-patch leaf must have native leading dimension "
            f"{plan.patch_count}, got {array.shape}"
        )
    return array[plan.patch_indices]


def _unpack_owned_patch_array(value: Any, plan: ColumnPatchShardPlan) -> jax.Array:

    array = jnp.asarray(value)
    expected = (plan.device_count, plan.patches_per_device)
    if array.ndim < 2 or array.shape[:2] != expected:
        raise ValueError(f"mapped patch result must start with {expected}, got {array.shape}")
    flattened = jnp.reshape(
        array,
        (plan.device_count * plan.patches_per_device, *array.shape[2:]),
    )
    selected = flattened[plan.valid_patch_slots]
    result = jnp.zeros((plan.patch_count, *array.shape[2:]), dtype=array.dtype)
    return result.at[plan.valid_patch_indices].set(selected)


def _pack_patch_array(value: Any, plan: PatchShardPlan) -> jax.Array:

    array = jnp.asarray(value)
    if array.ndim < 1 or array.shape[0] != plan.patch_count:
        raise ValueError(
            "every sharded leaf must have native patch leading dimension "
            f"{plan.patch_count}, got {array.shape}"
        )
    if plan.padding:
        extension = jnp.repeat(array[-1:], plan.padding, axis=0)
        array = jnp.concatenate((array, extension), axis=0)
    return jnp.reshape(
        array,
        (plan.device_count, plan.patches_per_device, *array.shape[1:]),
    )


@jax.jit(static_argnums=(1,))
def _pack_patch_tree(tree: Any, plan: PatchShardPlan) -> Any:
    return jax.tree_util.tree_map(lambda value: _pack_patch_array(value, plan), tree)


def _reshard_tree_leading_axis(
    tree: Any,
    devices: tuple[jax.Device, ...],
    axis_name: str,
) -> Any:

    mesh = Mesh(np.asarray(devices, dtype=object), (axis_name,))
    sharding = NamedSharding(mesh, PartitionSpec(axis_name))
    return jax.device_put(tree, sharding)


def _reshard_tree_for_pmap(
    tree: Any,
    axes: Any,
    devices: tuple[jax.Device, ...],
    axis_name: str,
) -> Any:

    mesh = Mesh(np.asarray(devices, dtype=object), (axis_name,))
    mapped = NamedSharding(mesh, PartitionSpec(axis_name))
    replicated = NamedSharding(mesh, PartitionSpec())

    def target_sharding(axis: int | None) -> NamedSharding:
        if axis not in (0, None):
            raise ValueError(f"only pmap axes 0 and None are supported, got {axis}")
        return mapped if axis == 0 else replicated

    targets = jax.tree_util.tree_map(
        target_sharding,
        axes,
        is_leaf=lambda value: value is None,
    )
    return jax.device_put(tree, targets)


def _unpack_patch_array(value: Any, plan: PatchShardPlan) -> jax.Array:

    array = jnp.asarray(value)
    expected = (plan.device_count, plan.patches_per_device)
    if array.ndim < 2 or array.shape[:2] != expected:
        raise ValueError(
            f"mapped result must start with device/patch shape {expected}, got {array.shape}"
        )
    flattened = jnp.reshape(array, (plan.device_count * plan.patches_per_device, *array.shape[2:]))
    return flattened[: plan.patch_count]


@jax.jit(static_argnums=(1,))
def _unpack_patch_tree(tree: Any, plan: PatchShardPlan) -> Any:
    return jax.tree_util.tree_map(lambda value: _unpack_patch_array(value, plan), tree)


def _unpack_patch_tree_on_device(
    tree: Any,
    plan: PatchShardPlan,
    device: jax.Device,
) -> Any:

    gathered = gather_mapped_result(tree, device)
    return _unpack_patch_tree(gathered, plan)


_CANOPY_STATIC_INDICES: dict[tuple[int, ...], tuple[Any, jax.Array]] = {}


def _static_canopy_patch_indices(topology: SubgridTopology) -> jax.Array:

    patch = topology.patch
    sources = (patch.active, patch.vegetation_type, patch.landunit_type)
    cacheable = all(isinstance(value, jax.Array) for value in sources)
    key = tuple(id(value) for value in sources)
    cached = _CANOPY_STATIC_INDICES.get(key) if cacheable else None
    if cached is not None and all(
        reference() is value and not value.is_deleted()
        for reference, value in zip(cached[0], sources, strict=True)
    ):
        return cached[1]
    landunit = np.asarray(patch.landunit_type, dtype=np.int32)
    eligible = (
        np.asarray(patch.active, dtype=bool)
        & (np.asarray(patch.vegetation_type, dtype=np.int32) > 0)
        & (landunit != LANDUNIT_LAKE)
        & ~((landunit >= LANDUNIT_URBAN_MIN) & (landunit <= LANDUNIT_URBAN_MAX))
    )
    indices = np.flatnonzero(eligible).astype(np.int32)
    if indices.size == 0:
        raise ValueError("native topology contains no possible active canopy patches")
    with jax.core.eval_context():
        result = jnp.asarray(indices, dtype=jnp.int32)
    if cacheable:
        def forget(reference):
            current = _CANOPY_STATIC_INDICES.get(key)
            if current is not None and any(item is reference for item in current[0]):
                _CANOPY_STATIC_INDICES.pop(key, None)

        _CANOPY_STATIC_INDICES[key] = (tuple(ref(value, forget) for value in sources), result)
    return result


@jax.jit(static_argnums=(2,))
def _select_patch_tree(tree: Any, indices: jax.Array, patch_count: int) -> Any:
    def select(value: Any) -> jax.Array:
        array = jnp.asarray(value)
        if array.ndim < 1 or array.shape[0] != patch_count:
            raise ValueError(
                "every compacted leaf must have native patch leading dimension "
                f"{patch_count}, got {array.shape}"
            )
        return array[indices]

    return jax.tree_util.tree_map(select, tree)


@jax.jit(static_argnums=(2,))
def _select_luna_inputs(
    inputs: CanopyFluxesLunaInputs,
    indices: jax.Array,
    patch_count: int,
) -> CanopyFluxesLunaInputs:
    capacity = LunaCapacityInputs(
        *(
            value
            if name == "daylength_seconds_gridcell"
            else _select_patch_tree(value, indices, patch_count)
            for name, value in zip(
                LunaCapacityInputs._fields,
                inputs.capacity_inputs,
                strict=True,
            )
        )
    )
    return CanopyFluxesLunaInputs(
        *(_select_patch_tree(value, indices, patch_count) for value in inputs[:5]),
        capacity_inputs=capacity,
    )


@jax.jit(static_argnums=(2,))
def _scatter_patch_tree(
    selected: Any,
    indices: jax.Array,
    patch_count: int,
    base: Any | None = None,
) -> Any:

    selected_count = indices.shape[0]

    def scatter(value: Any, base_value: Any | None = None) -> jax.Array:
        array = jnp.asarray(value)
        if array.ndim < 1 or array.shape[0] != selected_count:
            raise ValueError(
                "every compact result leaf must have selected-patch leading dimension "
                f"{selected_count}, got {array.shape}"
            )
        shape = (patch_count, *array.shape[1:])
        if base_value is None:
            if jnp.issubdtype(array.dtype, jnp.floating):
                output = jnp.full(shape, jnp.asarray(CLM_SPECIAL_VALUE, dtype=array.dtype))
            else:
                output = jnp.zeros(shape, dtype=array.dtype)
        else:
            output = jnp.asarray(base_value, dtype=array.dtype)
            if output.shape != shape:
                raise ValueError(
                    f"compact result base must have native shape {shape}, got {output.shape}"
                )
        return output.at[indices].set(array)

    if base is None:
        return jax.tree_util.tree_map(scatter, selected)
    return jax.tree_util.tree_map(scatter, selected, base)


def _expand_compact_canopy_result(
    selected: CanopyFluxesActiveResult,
    indices: jax.Array,
    patch_count: int,
    initial_energy_state: Any,
    hydraulic_coupling: Any,
    state: Any,
) -> CanopyFluxesActiveResult:

    expanded = _scatter_patch_tree(selected, indices, patch_count)
    expanded_state = _scatter_patch_tree(selected.state, indices, patch_count, state)
    selected_solved = selected.energy_hydraulics
    expanded_solved = expanded.energy_hydraulics
    expanded_hydraulic = expanded_solved.hydraulic._replace(
        plant_state=_scatter_patch_tree(
            selected_solved.hydraulic.plant_state,
            indices,
            patch_count,
            hydraulic_coupling.initial_plant_state,
        )
    )
    selected_iteration = selected_solved.energy.iteration
    expanded_iteration = expanded_solved.energy.iteration._replace(
        vegetation_temperature_patch=_scatter_patch_tree(
            selected_iteration.vegetation_temperature_patch,
            indices,
            patch_count,
            initial_energy_state.vegetation_temperature_patch,
        ),
        friction_velocity=_scatter_patch_tree(
            selected_iteration.friction_velocity,
            indices,
            patch_count,
            initial_energy_state.friction_velocity,
        ),
    )
    expanded_energy = expanded_solved.energy._replace(iteration=expanded_iteration)
    return expanded._replace(
        energy_hydraulics=expanded_solved._replace(
            energy=expanded_energy,
            hydraulic=expanded_hydraulic,
        ),
        state=expanded_state,
    )


def _pack_luna_inputs(
    inputs: CanopyFluxesLunaInputs,
    plan: PatchShardPlan,
) -> CanopyFluxesLunaInputs:
    capacity = LunaCapacityInputs(
        *(
            value if name == "daylength_seconds_gridcell" else _pack_patch_array(value, plan)
            for name, value in zip(
                LunaCapacityInputs._fields,
                inputs.capacity_inputs,
                strict=True,
            )
        )
    )
    return CanopyFluxesLunaInputs(
        *(_pack_patch_array(value, plan) for value in inputs[:5]),
        capacity_inputs=capacity,
    )


_PATCH_TOPOLOGY_AXES = PatchTopology(*(0 for _ in PatchTopology._fields))
_TOPOLOGY_AXES = SubgridTopology(
    grid=None,
    landunit=None,
    column=None,
    patch=_PATCH_TOPOLOGY_AXES,
)
_LUNA_CAPACITY_AXES = LunaCapacityInputs(
    *(None if name == "daylength_seconds_gridcell" else 0 for name in LunaCapacityInputs._fields)
)
_LUNA_INPUT_AXES = CanopyFluxesLunaInputs(
    absorbed_par_sunlit_by_layer_patch=0,
    absorbed_par_shaded_by_layer_patch=0,
    sunlit_leaf_area_by_layer_patch=0,
    shaded_leaf_area_by_layer_patch=0,
    radiative_layer_count_patch=0,
    capacity_inputs=_LUNA_CAPACITY_AXES,
)
_CANOPY_RUNTIME_INPUT_AXES = (
    _TOPOLOGY_AXES,
    0,
    0,
    0,
    0,
    _LUNA_INPUT_AXES,
    0,
    0,
    None,
)


def _distributed_vmap(
    function: Callable[..., Any],
    *,
    device_count: int,
    axis_name: str,
    in_axes: Any,
    static_argnums: tuple[int, ...] = (),
    compiler_options: dict[str, Any] | None = None,
):

    devices = tuple(jax.local_devices()[:device_count])
    if len(devices) != device_count:
        raise ValueError(
            f"requested {device_count} devices but JAX exposes only "
            f"{jax.local_device_count()}"
        )
    mesh = Mesh(np.asarray(devices, dtype=object), (axis_name,))
    mapped = NamedSharding(mesh, PartitionSpec(axis_name))
    replicated = NamedSharding(mesh, PartitionSpec())
    all_input_shardings = jax.tree_util.tree_map(
        lambda axis: mapped if axis == 0 else replicated,
        in_axes,
        is_leaf=lambda value: value is None,
    )
    input_shardings = tuple(
        sharding
        for index, sharding in enumerate(all_input_shardings)
        if index not in static_argnums
    )
    vectorized = jax.vmap(
        function,
        in_axes=in_axes,
        out_axes=0,
        axis_size=device_count,
        spmd_axis_name=axis_name,
    )
    return jax.jit(
        vectorized,
        in_shardings=input_shardings,
        out_shardings=mapped,
        static_argnums=static_argnums,
        compiler_options=compiler_options,
    )

_VOC_PATCH_INPUT_FIELDS = frozenset(
    {
        "soil_patch",
        "direct_visible_radiation_24h_patch",
        "direct_visible_radiation_240h_patch",
        "diffuse_visible_radiation_24h_patch",
        "diffuse_visible_radiation_240h_patch",
        "sunlit_fraction_patch",
        "sunlit_fraction_24h_patch",
        "sunlit_fraction_240h_patch",
        "exposed_leaf_area_index_patch",
        "exposed_leaf_area_index_240h_patch",
        "sunlit_intracellular_co2_patch",
        "shaded_intracellular_co2_patch",
        "vegetation_temperature_patch",
        "vegetation_temperature_24h_patch",
        "vegetation_temperature_240h_patch",
    }
)
_VOC_INPUT_AXES = VOCInputs(
    *(0 if name in _VOC_PATCH_INPUT_FIELDS else None for name in VOCInputs._fields)
)

_GROUND_HEAT_LANDUNIT_FIELDS = frozenset(
    {
        "waste_heat_landunit",
        "air_conditioner_heat_landunit",
        "traffic_heat_landunit",
        "urban_roof_fraction_landunit",
    }
)
_GROUND_HEAT_COLUMN_FIELDS = frozenset(
    {
        "snow_layer_count_column",
        "downward_longwave_column",
        "effective_snow_fraction_column",
        "ground_emissivity_column",
        "surface_water_temperature_column",
        "ground_temperature_column",
        "soil_snow_temperature_column",
        "latent_heat_column",
    }
)
_SOIL_COLUMN_TOPOLOGY_AXES = ColumnTopology(*(0 for _ in ColumnTopology._fields))
_SOIL_PATCH_TOPOLOGY_AXES = PatchTopology(*(0 for _ in PatchTopology._fields))
_SOIL_TOPOLOGY_AXES = SubgridTopology(
    grid=None,
    landunit=None,
    column=_SOIL_COLUMN_TOPOLOGY_AXES,
    patch=_SOIL_PATCH_TOPOLOGY_AXES,
)
_SOIL_GROUND_INPUT_AXES = GroundHeatFluxInputs(
    *(None if name in _GROUND_HEAT_LANDUNIT_FIELDS else 0 for name in GroundHeatFluxInputs._fields)
)
_SOIL_ACTIVE_INPUT_AXES = SoilTemperatureActiveInputs(
    topology=_SOIL_TOPOLOGY_AXES,
    urban_columns=None,
    thermal_properties=SoilThermalPropertyInputs(*(0 for _ in SoilThermalPropertyInputs._fields)),
    ground_heat_flux=_SOIL_GROUND_INPUT_AXES,
    ground_heat_flux_patch_state=GroundHeatFluxPatchState(
        *(0 for _ in GroundHeatFluxPatchState._fields)
    ),
    heat_diffusion_template=SoilHeatDiffusionInputs(*(0 for _ in SoilHeatDiffusionInputs._fields)),
    rhs_template=SoilTemperatureRHSInputs(*(0 for _ in SoilTemperatureRHSInputs._fields)),
    matrix_template=SoilTemperatureMatrixInputs(*(0 for _ in SoilTemperatureMatrixInputs._fields)),
    linear_solve_template=SoilTemperatureLinearSolveInputs(
        *(0 for _ in SoilTemperatureLinearSolveInputs._fields)
    ),
    solution_update_template=SoilTemperatureSolutionUpdateInputs(
        *(0 for _ in SoilTemperatureSolutionUpdateInputs._fields)
    ),
    postsolve_flux_template=SoilTemperaturePostSolveFluxInputs(
        *(0 for _ in SoilTemperaturePostSolveFluxInputs._fields)
    ),
    standing_water_phase_change_template=StandingWaterPhaseChangeInputs(
        *(0 for _ in StandingWaterPhaseChangeInputs._fields)
    ),
    soil_snow_phase_change_template=SoilSnowPhaseChangeInputs(
        *(0 for _ in SoilSnowPhaseChangeInputs._fields)
    ),
    building_temperature_template=None,
    final_diagnostics_template=SoilTemperatureFinalDiagnosticsInputs(
        *(0 for _ in SoilTemperatureFinalDiagnosticsInputs._fields)
    ),
)


def _pack_soil_temperature_active_inputs(
    inputs: SoilTemperatureActiveInputs,
    plan: ColumnPatchShardPlan,
) -> SoilTemperatureActiveInputs:

    column_valid = jnp.arange(
        plan.device_count * plan.columns_per_device,
        dtype=jnp.int32,
    ) < jnp.int32(plan.column_count)
    column_valid = jnp.reshape(
        column_valid,
        (plan.device_count, plan.columns_per_device),
    )
    column = jax.tree_util.tree_map(
        lambda value: _pack_column_array(value, plan),
        inputs.topology.column,
    )
    column = column._replace(active=column.active & column_valid)
    patch = jax.tree_util.tree_map(
        lambda value: _pack_owned_patch_array(value, plan),
        inputs.topology.patch,
    )
    column_start = jnp.arange(plan.device_count, dtype=jnp.int32)[:, None] * jnp.int32(
        plan.columns_per_device
    )
    local_patch_column = patch.column - column_start
    patch = patch._replace(
        column=jnp.where(plan.patch_valid, local_patch_column, jnp.int32(0)),
        active=patch.active & plan.patch_valid,
    )
    topology = inputs.topology._replace(column=column, patch=patch)

    thermal = jax.tree_util.tree_map(
        lambda value: _pack_column_array(value, plan),
        inputs.thermal_properties,
    )
    thermal = thermal._replace(active_nonlake=thermal.active_nonlake & column_valid)
    ground = GroundHeatFluxInputs(
        *(
            value
            if name in _GROUND_HEAT_LANDUNIT_FIELDS
            else (
                _pack_column_array(value, plan)
                if name in _GROUND_HEAT_COLUMN_FIELDS
                else _pack_owned_patch_array(value, plan)
            )
            for name, value in zip(
                GroundHeatFluxInputs._fields,
                inputs.ground_heat_flux,
                strict=True,
            )
        )
    )

    def columns(tree: Any) -> Any:
        return jax.tree_util.tree_map(
            lambda value: _pack_column_array(value, plan),
            tree,
        )

    return SoilTemperatureActiveInputs(
        topology=topology,
        urban_columns=inputs.urban_columns,
        thermal_properties=thermal,
        ground_heat_flux=ground,
        ground_heat_flux_patch_state=jax.tree_util.tree_map(
            lambda value: _pack_owned_patch_array(value, plan),
            inputs.ground_heat_flux_patch_state,
        ),
        heat_diffusion_template=columns(inputs.heat_diffusion_template),
        rhs_template=columns(inputs.rhs_template),
        matrix_template=columns(inputs.matrix_template),
        linear_solve_template=columns(inputs.linear_solve_template),
        solution_update_template=columns(inputs.solution_update_template),
        postsolve_flux_template=columns(inputs.postsolve_flux_template),
        standing_water_phase_change_template=columns(inputs.standing_water_phase_change_template),
        soil_snow_phase_change_template=columns(inputs.soil_snow_phase_change_template),
        building_temperature_template=inputs.building_temperature_template,
        final_diagnostics_template=columns(inputs.final_diagnostics_template),
    )


def _unpack_soil_temperature_column_result(
    mapped: SoilTemperatureColumnResult,
    plan: ColumnPatchShardPlan,
) -> SoilTemperatureColumnResult:

    def columns(tree: Any) -> Any:
        return jax.tree_util.tree_map(
            lambda value: _unpack_column_array(value, plan),
            tree,
        )

    ground = mapped.ground_heat_flux
    unpacked_ground = GroundHeatFluxResult(
        *(columns(value) for value in ground[:6]),
        patch_state=jax.tree_util.tree_map(
            lambda value: _unpack_owned_patch_array(value, plan),
            ground.patch_state,
        ),
    )
    return SoilTemperatureColumnResult(
        thermal_properties=columns(mapped.thermal_properties),
        ground_heat_flux=unpacked_ground,
        heat_diffusion=columns(mapped.heat_diffusion),
        standing_water_heat_capacity=_unpack_column_array(
            mapped.standing_water_heat_capacity,
            plan,
        ),
        standing_water_thickness=_unpack_column_array(
            mapped.standing_water_thickness,
            plan,
        ),
        rhs=columns(mapped.rhs),
        matrix=columns(mapped.matrix),
        linear_solve=columns(mapped.linear_solve),
        solution_update=columns(mapped.solution_update),
        postsolve_flux=columns(mapped.postsolve_flux),
        standing_water_phase_change=columns(mapped.standing_water_phase_change),
        soil_snow_phase_change=columns(mapped.soil_snow_phase_change),
        final_diagnostics=columns(mapped.final_diagnostics),
    )


def _unpack_soil_temperature_runtime_column_result(
    mapped: SoilTemperatureRuntimeColumnResult,
    plan: ColumnPatchShardPlan,
) -> SoilTemperatureRuntimeColumnResult:

    def columns(tree: Any) -> Any:
        return jax.tree_util.tree_map(
            lambda value: _unpack_column_array(value, plan),
            tree,
        )

    ground = mapped.ground_heat_flux
    unpacked_ground = GroundHeatFluxResult(
        *(columns(value) for value in ground[:6]),
        patch_state=jax.tree_util.tree_map(
            lambda value: _unpack_owned_patch_array(value, plan),
            ground.patch_state,
        ),
    )
    return SoilTemperatureRuntimeColumnResult(
        thermal_properties=columns(mapped.thermal_properties),
        ground_heat_flux=unpacked_ground,
        heat_diffusion=columns(mapped.heat_diffusion),
        standing_water_heat_capacity=_unpack_column_array(
            mapped.standing_water_heat_capacity,
            plan,
        ),
        postsolve_flux=columns(mapped.postsolve_flux),
        standing_water_phase_change=columns(mapped.standing_water_phase_change),
        soil_snow_phase_change=columns(mapped.soil_snow_phase_change),
        final_diagnostics=columns(mapped.final_diagnostics),
    )


@cache
def _soil_temperature_layout_kernels(dimensions: tuple[int, ...]):
    pack_function = _pack_soil_temperature_active_inputs
    unpack_function = _unpack_soil_temperature_runtime_column_result

    @jax.jit
    def pack(inputs, indices):
        return pack_function(inputs, ColumnPatchShardPlan(*dimensions, *indices))

    @jax.jit
    def unpack(result, indices):
        return unpack_function(result, ColumnPatchShardPlan(*dimensions, *indices))

    return pack, unpack


def _pack_soil_temperature_active_inputs_compiled(inputs, plan):
    return _soil_temperature_layout_kernels(tuple(plan[:6]))[0](inputs, plan[6:])


def _unpack_soil_temperature_runtime_column_result_compiled(result, plan):
    return _soil_temperature_layout_kernels(tuple(plan[:6]))[1](result, plan[6:])


@cache
def _mapped_canopy_fluxes_active_case(device_count: int):
    devices = tuple(jax.local_devices()[:device_count])
    if len(devices) != device_count:
        raise ValueError(
            f"requested {device_count} devices but JAX exposes only {jax.local_device_count()}"
        )
    return jax.pmap(
        canopy_fluxes_active_case,
        axis_name="clm_patch_device",
        in_axes=(
            _TOPOLOGY_AXES,
            0,
            0,
            0,
            0,
            _LUNA_INPUT_AXES,
            0,
            None,
        ),
        out_axes=0,
        devices=devices,
    )


def _canopy_fluxes_voc_fields_core(
    topology: SubgridTopology,
    energy_inputs: Any,
    hydraulic_coupling: Any,
    initial_energy_state: Any,
    postprocess_inputs: Any,
    luna_inputs: CanopyFluxesLunaInputs,
    state: Any,
    parameters: CanopyFluxesActiveParameters,
) -> CanopyFluxVOCFields:

    solved = canopy_fluxes_active_case(
        topology,
        energy_inputs,
        hydraulic_coupling,
        initial_energy_state,
        postprocess_inputs,
        luna_inputs,
        state,
        parameters,
    )
    return CanopyFluxVOCFields(
        active_patch=energy_inputs.active_patch,
        vegetation_temperature_patch=(
            solved.energy_hydraulics.energy.iteration.vegetation_temperature_patch
        ),
        intracellular_co2_patch_leaf=(
            solved.energy_hydraulics.hydraulic.intracellular_co2_patch_leaf
        ),
    )


def _canopy_fluxes_soil_temperature_fields_core(
    topology: SubgridTopology,
    energy_inputs: Any,
    hydraulic_coupling: Any,
    initial_energy_state: Any,
    postprocess_inputs: Any,
    luna_inputs: CanopyFluxesLunaInputs,
    state: Any,
    parameters: CanopyFluxesActiveParameters,
) -> CanopyFluxSoilTemperatureFields:

    solved = canopy_fluxes_active_case(
        topology,
        energy_inputs,
        hydraulic_coupling,
        initial_energy_state,
        postprocess_inputs,
        luna_inputs,
        state,
        parameters,
    )
    post = solved.state.postprocess
    iteration = solved.energy_hydraulics.energy.iteration
    return CanopyFluxSoilTemperatureFields(
        snow_evaporation_patch=post.snow_evaporation_patch,
        soil_evaporation_patch=post.soil_evaporation_patch,
        surface_water_evaporation_patch=post.surface_water_evaporation_patch,
        ground_evaporation_patch=post.ground_evaporation_patch,
        vegetation_transpiration_patch=iteration.vegetation_transpiration_patch,
        vegetation_evaporation_patch=iteration.vegetation_evaporation_patch,
        vegetation_sensible_heat_patch=(iteration.vegetation_sensible_heat_flux_patch),
        soil_root_conductance_patch_level=(
            hydraulic_coupling.plant_inputs.soil_root_conductance_level
        ),
        ground_methane_conductance_patch=(
            iteration.ground_methane_conductance_patch
        ),
        ground_energy_derivative_patch=post.ground_energy_derivative_patch,
        downward_longwave_below_canopy_patch=(post.downward_longwave_below_canopy_patch),
        sensible_heat_snow_patch=post.sensible_heat_snow_patch,
        sensible_heat_soil_patch=post.sensible_heat_soil_patch,
        sensible_heat_surface_water_patch=(post.sensible_heat_surface_water_patch),
        sensible_heat_ground_patch=post.sensible_heat_ground_patch,
    )


@gpu_convergence_scope
def _canopy_fluxes_runtime_core(
    topology: SubgridTopology,
    energy_inputs: Any,
    hydraulic_coupling: Any,
    initial_energy_state: Any,
    postprocess_inputs: Any,
    luna_inputs: CanopyFluxesLunaInputs,
    state: Any,
    previous_root_zone_wetness_patch: Any,
    parameters: CanopyFluxesActiveParameters,
) -> CanopyFluxRuntimeResult:

    solved = canopy_fluxes_active_case(
        topology,
        energy_inputs,
        hydraulic_coupling,
        initial_energy_state,
        postprocess_inputs,
        luna_inputs,
        state,
        parameters,
    )
    coupled = solved.energy_hydraulics
    iteration = coupled.energy.iteration
    active_patch = jnp.asarray(energy_inputs.active_patch, dtype=jnp.bool_)
    initial_plant = hydraulic_coupling.initial_plant_state
    solved_plant = coupled.hydraulic.plant_state
    plant = solved_plant._replace(
        water_potential_segment=jnp.where(
            active_patch[:, None],
            solved_plant.water_potential_segment,
            initial_plant.water_potential_segment,
        ),
        vegetation_transpiration=jnp.where(
            active_patch,
            solved_plant.vegetation_transpiration,
            initial_plant.vegetation_transpiration,
        ),
    )
    root_zone_wetness = jnp.where(
        active_patch,
        hydraulic_coupling.root_zone_wetness_patch,
        jnp.asarray(previous_root_zone_wetness_patch, dtype=jnp.float64),
    )
    runtime_state = CanopyFluxDriverState(
        active=solved.state,
        plant=plant,
        vegetation_temperature_patch=iteration.vegetation_temperature_patch,
        root_zone_wetness_patch=root_zone_wetness,
        friction_velocity=iteration.friction_velocity,
        iteration_work_patch=(
            None if initial_energy_state.iteration_work_patch is None
            else canopy_iteration_work(coupled)
        ),
    )
    post = solved.state.postprocess
    return CanopyFluxRuntimeResult(
        state=runtime_state,
        voc=CanopyFluxVOCFields(
            active_patch=active_patch,
            vegetation_temperature_patch=iteration.vegetation_temperature_patch,
            intracellular_co2_patch_leaf=(coupled.hydraulic.intracellular_co2_patch_leaf),
        ),
        soil_temperature=CanopyFluxSoilTemperatureFields(
            snow_evaporation_patch=post.snow_evaporation_patch,
            soil_evaporation_patch=post.soil_evaporation_patch,
            surface_water_evaporation_patch=post.surface_water_evaporation_patch,
            ground_evaporation_patch=post.ground_evaporation_patch,
            vegetation_transpiration_patch=iteration.vegetation_transpiration_patch,
            vegetation_evaporation_patch=iteration.vegetation_evaporation_patch,
            vegetation_sensible_heat_patch=(iteration.vegetation_sensible_heat_flux_patch),
            soil_root_conductance_patch_level=(
                hydraulic_coupling.plant_inputs.soil_root_conductance_level
            ),
            ground_methane_conductance_patch=(
                iteration.ground_methane_conductance_patch
            ),
            ground_energy_derivative_patch=post.ground_energy_derivative_patch,
            downward_longwave_below_canopy_patch=(post.downward_longwave_below_canopy_patch),
            sensible_heat_snow_patch=post.sensible_heat_snow_patch,
            sensible_heat_soil_patch=post.sensible_heat_soil_patch,
            sensible_heat_surface_water_patch=(post.sensible_heat_surface_water_patch),
            sensible_heat_ground_patch=post.sensible_heat_ground_patch,
        ),
        dust=CanopyFluxDustFields(
            active_patch=active_patch,
            momentum_resistance_patch=iteration.momentum_resistance_patch,
        ),
        ecosystem=CanopyFluxEcosystemFields(
            vegetation_exposed_patch=jnp.asarray(
                energy_inputs.vegetation_fraction_without_snow_patch,
                dtype=jnp.int32,
            ),
            sunlit_leaf_area_patch=energy_inputs.sunlit_leaf_area_index_patch,
            shaded_leaf_area_patch=energy_inputs.shaded_leaf_area_index_patch,
            total_leaf_area_patch=(
                luna_inputs.capacity_inputs.total_leaf_area_patch
            ),
            sunlit_leaf_respiration_patch=(
                coupled.hydraulic.leaf_maintenance_respiration_patch_leaf[:, 0]
            ),
            shaded_leaf_respiration_patch=(
                coupled.hydraulic.leaf_maintenance_respiration_patch_leaf[:, 1]
            ),
            sunlit_photosynthesis_patch=(
                coupled.hydraulic.gross_photosynthesis_patch_leaf[:, 0]
            ),
            shaded_photosynthesis_patch=(
                coupled.hydraulic.gross_photosynthesis_patch_leaf[:, 1]
            ),
            transpiration_wetness_patch=(
                coupled.hydraulic.photosynthesis.transpiration_wetness_patch
            ),
            root_zone_wetness_patch=(
                root_zone_wetness
            ),
        ),
    )


@cache
def _mapped_canopy_fluxes_voc_fields(device_count: int):
    devices = tuple(jax.local_devices()[:device_count])
    if len(devices) != device_count:
        raise ValueError(
            f"requested {device_count} devices but JAX exposes only {jax.local_device_count()}"
        )
    return jax.pmap(
        _canopy_fluxes_voc_fields_core,
        axis_name="clm_patch_device",
        in_axes=(
            _TOPOLOGY_AXES,
            0,
            0,
            0,
            0,
            _LUNA_INPUT_AXES,
            0,
            None,
        ),
        out_axes=0,
        devices=devices,
    )


@cache
def _mapped_canopy_fluxes_soil_temperature_fields(device_count: int):
    devices = tuple(jax.local_devices()[:device_count])
    if len(devices) != device_count:
        raise ValueError(
            f"requested {device_count} devices but JAX exposes only {jax.local_device_count()}"
        )
    return jax.pmap(
        _canopy_fluxes_soil_temperature_fields_core,
        axis_name="clm_patch_device",
        in_axes=(
            _TOPOLOGY_AXES,
            0,
            0,
            0,
            0,
            _LUNA_INPUT_AXES,
            0,
            None,
        ),
        out_axes=0,
        devices=devices,
    )


@cache
def _mapped_canopy_fluxes_runtime(device_count: int):
    devices = tuple(jax.local_devices()[:device_count])
    if len(devices) != device_count:
        raise ValueError(
            f"requested {device_count} devices but JAX exposes only {jax.local_device_count()}"
        )
    return jax.pmap(
        _canopy_fluxes_runtime_core,
        axis_name="clm_patch_device",
        in_axes=_CANOPY_RUNTIME_INPUT_AXES,
        out_axes=0,
        devices=devices,
    )


def canopy_fluxes_active_case_parallel(
    topology: SubgridTopology,
    energy_inputs: Any,
    hydraulic_coupling: Any,
    initial_energy_state: Any,
    postprocess_inputs: Any,
    luna_inputs: CanopyFluxesLunaInputs,
    state: Any,
    parameters: CanopyFluxesActiveParameters,
    *,
    device_count: int | None = None,
    compact_static_filter: bool = False,
) -> CanopyFluxesActiveResult:

    patch_count = int(topology.patch.gridcell.shape[0])
    if compact_static_filter:
        indices = _static_canopy_patch_indices(topology)
        selected_topology = topology._replace(
            patch=_select_patch_tree(topology.patch, indices, patch_count)
        )
        selected = canopy_fluxes_active_case_parallel(
            selected_topology,
            _select_patch_tree(energy_inputs, indices, patch_count),
            _select_patch_tree(hydraulic_coupling, indices, patch_count),
            _select_patch_tree(initial_energy_state, indices, patch_count),
            _select_patch_tree(postprocess_inputs, indices, patch_count),
            _select_luna_inputs(luna_inputs, indices, patch_count),
            _select_patch_tree(state, indices, patch_count),
            parameters,
            device_count=device_count,
            compact_static_filter=False,
        )
        return _expand_compact_canopy_result(
            selected,
            indices,
            patch_count,
            initial_energy_state,
            hydraulic_coupling,
            state,
        )
    available = jax.local_device_count()
    selected = available if device_count is None else int(device_count)
    if selected > available:
        raise ValueError(f"requested {selected} devices but JAX exposes only {available}")
    plan = plan_patch_shards(patch_count, selected)

    patch_topology = _pack_patch_tree(topology.patch, plan)
    valid = jnp.arange(plan.device_count * plan.patches_per_device) < plan.patch_count
    valid = jnp.reshape(valid, (plan.device_count, plan.patches_per_device))
    patch_topology = patch_topology._replace(active=patch_topology.active & valid)
    mapped_topology = topology._replace(patch=patch_topology)

    mapped_energy = _pack_patch_tree(energy_inputs, plan)
    mapped_energy = mapped_energy._replace(active_patch=mapped_energy.active_patch & valid)
    mapped_luna = _pack_luna_inputs(luna_inputs, plan)

    mapped = _mapped_canopy_fluxes_active_case(plan.device_count)(
        mapped_topology,
        mapped_energy,
        _pack_patch_tree(hydraulic_coupling, plan),
        _pack_patch_tree(initial_energy_state, plan),
        _pack_patch_tree(postprocess_inputs, plan),
        mapped_luna,
        _pack_patch_tree(state, plan),
        parameters,
    )
    return _unpack_patch_tree_on_device(mapped, plan, jax.local_devices()[0])


def canopy_flux_driver_step_parallel(
    topology: SubgridTopology,
    atmospheric: Any,
    grid_forcing: Any,
    canopy: Any,
    radiation: Any,
    temperature: Any,
    bare_ground: Any,
    clock: Any,
    inputs: Any,
    parameters: Any,
    state: Any,
    *,
    device_count: int | None = None,
    compact_static_filter: bool = True,
) -> CanopyFluxDriverResult:

    call = assemble_canopy_flux_driver_call(
        topology,
        atmospheric,
        grid_forcing,
        canopy,
        radiation,
        temperature,
        bare_ground,
        clock,
        inputs,
        parameters,
        state,
    )
    active = canopy_fluxes_active_case_parallel(
        call.topology,
        call.fast_inputs.energy,
        call.hydraulic_coupling,
        call.energy_initial,
        call.fast_inputs.postprocess,
        call.luna_inputs,
        call.synchronized_state,
        call.active_parameters,
        device_count=device_count,
        compact_static_filter=compact_static_filter,
    )
    return complete_canopy_flux_driver_call(call, active)


def canopy_fluxes_voc_fields_parallel(
    topology: SubgridTopology,
    energy_inputs: Any,
    hydraulic_coupling: Any,
    initial_energy_state: Any,
    postprocess_inputs: Any,
    luna_inputs: CanopyFluxesLunaInputs,
    state: Any,
    parameters: CanopyFluxesActiveParameters,
    *,
    device_count: int | None = None,
    compact_static_filter: bool = True,
) -> CanopyFluxVOCFields:

    patch_count = int(topology.patch.gridcell.shape[0])
    if compact_static_filter:
        indices = _static_canopy_patch_indices(topology)
        selected_topology = topology._replace(
            patch=_select_patch_tree(topology.patch, indices, patch_count)
        )
        selected = canopy_fluxes_voc_fields_parallel(
            selected_topology,
            _select_patch_tree(energy_inputs, indices, patch_count),
            _select_patch_tree(hydraulic_coupling, indices, patch_count),
            _select_patch_tree(initial_energy_state, indices, patch_count),
            _select_patch_tree(postprocess_inputs, indices, patch_count),
            _select_luna_inputs(luna_inputs, indices, patch_count),
            _select_patch_tree(state, indices, patch_count),
            parameters,
            device_count=device_count,
            compact_static_filter=False,
        )
        base = CanopyFluxVOCFields(
            active_patch=jnp.asarray(energy_inputs.active_patch, dtype=jnp.bool_),
            vegetation_temperature_patch=jnp.asarray(
                initial_energy_state.vegetation_temperature_patch,
                dtype=jnp.float64,
            ),
            intracellular_co2_patch_leaf=jnp.zeros(
                (patch_count, *selected.intracellular_co2_patch_leaf.shape[1:]),
                dtype=jnp.float64,
            ),
        )
        return _scatter_patch_tree(selected, indices, patch_count, base)

    available = jax.local_device_count()
    selected_device_count = available if device_count is None else int(device_count)
    if selected_device_count < 1:
        raise ValueError("device_count must be positive")
    if selected_device_count > available:
        raise ValueError(
            f"requested {selected_device_count} devices but JAX exposes only {available}"
        )
    plan = plan_patch_shards(patch_count, selected_device_count)
    patch_topology = _pack_patch_tree(topology.patch, plan)
    valid = jnp.arange(plan.device_count * plan.patches_per_device) < plan.patch_count
    valid = jnp.reshape(valid, (plan.device_count, plan.patches_per_device))
    patch_topology = patch_topology._replace(active=patch_topology.active & valid)
    mapped_topology = topology._replace(patch=patch_topology)
    mapped_energy = _pack_patch_tree(energy_inputs, plan)
    mapped_energy = mapped_energy._replace(active_patch=mapped_energy.active_patch & valid)
    mapped = _mapped_canopy_fluxes_voc_fields(plan.device_count)(
        mapped_topology,
        mapped_energy,
        _pack_patch_tree(hydraulic_coupling, plan),
        _pack_patch_tree(initial_energy_state, plan),
        _pack_patch_tree(postprocess_inputs, plan),
        _pack_luna_inputs(luna_inputs, plan),
        _pack_patch_tree(state, plan),
        parameters,
    )
    return _unpack_patch_tree_on_device(mapped, plan, jax.local_devices()[0])


def canopy_fluxes_soil_temperature_fields_parallel(
    topology: SubgridTopology,
    energy_inputs: Any,
    hydraulic_coupling: Any,
    initial_energy_state: Any,
    postprocess_inputs: Any,
    luna_inputs: CanopyFluxesLunaInputs,
    state: Any,
    parameters: CanopyFluxesActiveParameters,
    *,
    device_count: int | None = None,
    compact_static_filter: bool = True,
) -> CanopyFluxSoilTemperatureFields:

    patch_count = int(topology.patch.gridcell.shape[0])
    if compact_static_filter:
        indices = _static_canopy_patch_indices(topology)
        selected_topology = topology._replace(
            patch=_select_patch_tree(topology.patch, indices, patch_count)
        )
        selected = canopy_fluxes_soil_temperature_fields_parallel(
            selected_topology,
            _select_patch_tree(energy_inputs, indices, patch_count),
            _select_patch_tree(hydraulic_coupling, indices, patch_count),
            _select_patch_tree(initial_energy_state, indices, patch_count),
            _select_patch_tree(postprocess_inputs, indices, patch_count),
            _select_luna_inputs(luna_inputs, indices, patch_count),
            _select_patch_tree(state, indices, patch_count),
            parameters,
            device_count=device_count,
            compact_static_filter=False,
        )
        post = state.postprocess
        base = CanopyFluxSoilTemperatureFields(
            snow_evaporation_patch=post.snow_evaporation_patch,
            soil_evaporation_patch=post.soil_evaporation_patch,
            surface_water_evaporation_patch=post.surface_water_evaporation_patch,
            ground_evaporation_patch=post.ground_evaporation_patch,
            vegetation_transpiration_patch=jnp.zeros(
                (patch_count,),
                dtype=jnp.float64,
            ),
            vegetation_evaporation_patch=jnp.zeros(
                (patch_count,),
                dtype=jnp.float64,
            ),
            vegetation_sensible_heat_patch=jnp.zeros(
                (patch_count,),
                dtype=jnp.float64,
            ),
            soil_root_conductance_patch_level=(
                hydraulic_coupling.plant_inputs.soil_root_conductance_level
            ),
            ground_methane_conductance_patch=jnp.zeros(
                (patch_count,),
                dtype=jnp.float64,
            ),
            ground_energy_derivative_patch=post.ground_energy_derivative_patch,
            downward_longwave_below_canopy_patch=(post.downward_longwave_below_canopy_patch),
            sensible_heat_snow_patch=post.sensible_heat_snow_patch,
            sensible_heat_soil_patch=post.sensible_heat_soil_patch,
            sensible_heat_surface_water_patch=(post.sensible_heat_surface_water_patch),
            sensible_heat_ground_patch=post.sensible_heat_ground_patch,
        )
        return _scatter_patch_tree(selected, indices, patch_count, base)

    available = jax.local_device_count()
    selected_device_count = available if device_count is None else int(device_count)
    if selected_device_count < 1:
        raise ValueError("device_count must be positive")
    if selected_device_count > available:
        raise ValueError(
            f"requested {selected_device_count} devices but JAX exposes only {available}"
        )
    plan = plan_patch_shards(patch_count, selected_device_count)
    patch_topology = _pack_patch_tree(topology.patch, plan)
    valid = jnp.arange(plan.device_count * plan.patches_per_device) < plan.patch_count
    valid = jnp.reshape(valid, (plan.device_count, plan.patches_per_device))
    patch_topology = patch_topology._replace(active=patch_topology.active & valid)
    mapped_topology = topology._replace(patch=patch_topology)
    mapped_energy = _pack_patch_tree(energy_inputs, plan)
    mapped_energy = mapped_energy._replace(active_patch=mapped_energy.active_patch & valid)
    mapped = _mapped_canopy_fluxes_soil_temperature_fields(plan.device_count)(
        mapped_topology,
        mapped_energy,
        _pack_patch_tree(hydraulic_coupling, plan),
        _pack_patch_tree(initial_energy_state, plan),
        _pack_patch_tree(postprocess_inputs, plan),
        _pack_luna_inputs(luna_inputs, plan),
        _pack_patch_tree(state, plan),
        parameters,
    )
    return _unpack_patch_tree_on_device(mapped, plan, jax.local_devices()[0])


def canopy_fluxes_runtime_parallel(
    topology: SubgridTopology,
    energy_inputs: Any,
    hydraulic_coupling: Any,
    initial_energy_state: Any,
    postprocess_inputs: Any,
    luna_inputs: CanopyFluxesLunaInputs,
    state: Any,
    previous_root_zone_wetness_patch: Any,
    parameters: CanopyFluxesActiveParameters,
    *,
    device_count: int | None = None,
    compact_static_filter: bool = True,
) -> CanopyFluxRuntimeResult:

    patch_count = int(topology.patch.gridcell.shape[0])
    if compact_static_filter:
        indices = _static_canopy_patch_indices(topology)
        selected_topology = topology._replace(
            patch=_select_patch_tree(topology.patch, indices, patch_count)
        )
        selected = canopy_fluxes_runtime_parallel(
            selected_topology,
            _select_patch_tree(energy_inputs, indices, patch_count),
            _select_patch_tree(hydraulic_coupling, indices, patch_count),
            _select_patch_tree(initial_energy_state, indices, patch_count),
            _select_patch_tree(postprocess_inputs, indices, patch_count),
            _select_luna_inputs(luna_inputs, indices, patch_count),
            _select_patch_tree(state, indices, patch_count),
            _select_patch_tree(
                previous_root_zone_wetness_patch,
                indices,
                patch_count,
            ),
            parameters,
            device_count=device_count,
            compact_static_filter=False,
        )
        post = state.postprocess
        base_state = CanopyFluxDriverState(
            active=state,
            plant=hydraulic_coupling.initial_plant_state,
            vegetation_temperature_patch=(initial_energy_state.vegetation_temperature_patch),
            root_zone_wetness_patch=previous_root_zone_wetness_patch,
            friction_velocity=initial_energy_state.friction_velocity,
            iteration_work_patch=initial_energy_state.iteration_work_patch,
        )
        base = CanopyFluxRuntimeResult(
            state=base_state,
            voc=CanopyFluxVOCFields(
                active_patch=jnp.asarray(energy_inputs.active_patch, dtype=jnp.bool_),
                vegetation_temperature_patch=(initial_energy_state.vegetation_temperature_patch),
                intracellular_co2_patch_leaf=jnp.zeros(
                    (
                        patch_count,
                        *selected.voc.intracellular_co2_patch_leaf.shape[1:],
                    ),
                    dtype=jnp.float64,
                ),
            ),
            soil_temperature=CanopyFluxSoilTemperatureFields(
                snow_evaporation_patch=post.snow_evaporation_patch,
                soil_evaporation_patch=post.soil_evaporation_patch,
                surface_water_evaporation_patch=(post.surface_water_evaporation_patch),
                ground_evaporation_patch=post.ground_evaporation_patch,
                vegetation_transpiration_patch=jnp.zeros(
                    (patch_count,),
                    dtype=jnp.float64,
                ),
                vegetation_evaporation_patch=jnp.zeros(
                    (patch_count,),
                    dtype=jnp.float64,
                ),
                vegetation_sensible_heat_patch=jnp.zeros(
                    (patch_count,),
                    dtype=jnp.float64,
                ),
                soil_root_conductance_patch_level=(
                    hydraulic_coupling.plant_inputs.soil_root_conductance_level
                ),
                ground_methane_conductance_patch=jnp.zeros(
                    (patch_count,),
                    dtype=jnp.float64,
                ),
                ground_energy_derivative_patch=(post.ground_energy_derivative_patch),
                downward_longwave_below_canopy_patch=(post.downward_longwave_below_canopy_patch),
                sensible_heat_snow_patch=post.sensible_heat_snow_patch,
                sensible_heat_soil_patch=post.sensible_heat_soil_patch,
                sensible_heat_surface_water_patch=(post.sensible_heat_surface_water_patch),
                sensible_heat_ground_patch=post.sensible_heat_ground_patch,
            ),
            dust=CanopyFluxDustFields(
                active_patch=jnp.asarray(energy_inputs.active_patch, dtype=jnp.bool_),
                momentum_resistance_patch=jnp.zeros(
                    (patch_count,),
                    dtype=jnp.float64,
                ),
            ),
            ecosystem=CanopyFluxEcosystemFields(
                vegetation_exposed_patch=jnp.asarray(
                    energy_inputs.vegetation_fraction_without_snow_patch,
                    dtype=jnp.int32,
                ),
                sunlit_leaf_area_patch=(
                    energy_inputs.sunlit_leaf_area_index_patch
                ),
                shaded_leaf_area_patch=(
                    energy_inputs.shaded_leaf_area_index_patch
                ),
                total_leaf_area_patch=(
                    luna_inputs.capacity_inputs.total_leaf_area_patch
                ),
                sunlit_leaf_respiration_patch=jnp.zeros(
                    (patch_count,), dtype=jnp.float64
                ),
                shaded_leaf_respiration_patch=jnp.zeros(
                    (patch_count,), dtype=jnp.float64
                ),
                sunlit_photosynthesis_patch=jnp.zeros(
                    (patch_count,), dtype=jnp.float64
                ),
                shaded_photosynthesis_patch=jnp.zeros(
                    (patch_count,), dtype=jnp.float64
                ),
                transpiration_wetness_patch=jnp.ones(
                    (patch_count,), dtype=jnp.float64
                ),
                root_zone_wetness_patch=(
                    previous_root_zone_wetness_patch
                ),
            ),
        )
        return _scatter_patch_tree(selected, indices, patch_count, base)

    available = jax.local_device_count()
    selected_device_count = available if device_count is None else int(device_count)
    if selected_device_count < 1:
        raise ValueError("device_count must be positive")
    if selected_device_count > available:
        raise ValueError(
            f"requested {selected_device_count} devices but JAX exposes only {available}"
        )
    if selected_device_count == 1 and jax.local_devices()[0].platform == "gpu":
        return process_lifetime_jit(_canopy_fluxes_runtime_core)(
            topology, energy_inputs, hydraulic_coupling, initial_energy_state,
            postprocess_inputs, luna_inputs, state, previous_root_zone_wetness_patch,
            parameters,
        )
    plan = plan_patch_shards(patch_count, selected_device_count)
    patch_topology = _pack_patch_tree(topology.patch, plan)
    valid = jnp.arange(plan.device_count * plan.patches_per_device) < plan.patch_count
    valid = jnp.reshape(valid, (plan.device_count, plan.patches_per_device))
    patch_topology = patch_topology._replace(active=patch_topology.active & valid)
    mapped_topology = topology._replace(patch=patch_topology)
    mapped_energy = _pack_patch_tree(energy_inputs, plan)
    mapped_energy = mapped_energy._replace(active_patch=mapped_energy.active_patch & valid)
    arguments = (
        mapped_topology,
        mapped_energy,
        _pack_patch_tree(hydraulic_coupling, plan),
        _pack_patch_tree(initial_energy_state, plan),
        _pack_patch_tree(postprocess_inputs, plan),
        _pack_luna_inputs(luna_inputs, plan),
        _pack_patch_tree(state, plan),
        _pack_patch_tree(previous_root_zone_wetness_patch, plan),
        parameters,
    )
    devices = tuple(jax.local_devices()[:selected_device_count])
    arguments = _reshard_tree_for_pmap(
        arguments,
        _CANOPY_RUNTIME_INPUT_AXES,
        devices,
        "clm_patch_device",
    )
    mapped = _mapped_canopy_fluxes_runtime(selected_device_count)(*arguments)
    return _unpack_patch_tree_on_device(
        mapped,
        plan,
        jax.local_devices()[0],
    )


def canopy_flux_voc_driver_step_parallel(
    topology: Any,
    atmospheric: Any,
    grid_forcing: Any,
    canopy: Any,
    radiation: Any,
    temperature: Any,
    bare_ground: Any,
    clock: Any,
    inputs: Any,
    parameters: Any,
    state: Any,
    *,
    device_count: int | None = None,
) -> CanopyFluxVOCFields:

    call: CanopyFluxDriverCall = assemble_canopy_flux_driver_call(
        topology,
        atmospheric,
        grid_forcing,
        canopy,
        radiation,
        temperature,
        bare_ground,
        clock,
        inputs,
        parameters,
        state,
    )
    return canopy_fluxes_voc_fields_parallel(
        call.topology,
        call.fast_inputs.energy,
        call.hydraulic_coupling,
        call.energy_initial,
        call.fast_inputs.postprocess,
        call.luna_inputs,
        call.synchronized_state,
        call.active_parameters,
        device_count=device_count,
        compact_static_filter=True,
    )


def canopy_flux_soil_temperature_driver_step_parallel(
    topology: Any,
    atmospheric: Any,
    grid_forcing: Any,
    canopy: Any,
    radiation: Any,
    temperature: Any,
    bare_ground: Any,
    clock: Any,
    inputs: Any,
    parameters: Any,
    state: Any,
    *,
    device_count: int | None = None,
) -> CanopyFluxSoilTemperatureFields:

    call: CanopyFluxDriverCall = assemble_canopy_flux_driver_call(
        topology,
        atmospheric,
        grid_forcing,
        canopy,
        radiation,
        temperature,
        bare_ground,
        clock,
        inputs,
        parameters,
        state,
    )
    return canopy_fluxes_soil_temperature_fields_parallel(
        call.topology,
        call.fast_inputs.energy,
        call.hydraulic_coupling,
        call.energy_initial,
        call.fast_inputs.postprocess,
        call.luna_inputs,
        call.synchronized_state,
        call.active_parameters,
        device_count=device_count,
        compact_static_filter=True,
    )


_compiled_canopy_flux_driver_call = jit_unique(
    assemble_canopy_flux_driver_call, forward_inputs=True,
)


def canopy_flux_runtime_driver_step_parallel(
    topology: Any,
    atmospheric: Any,
    grid_forcing: Any,
    canopy: Any,
    radiation: Any,
    temperature: Any,
    bare_ground: Any,
    clock: Any,
    inputs: Any,
    parameters: Any,
    state: Any,
    *,
    device_count: int | None = None,
) -> CanopyFluxRuntimeResult:

    declination, _ = solar_declination_runtime(noleap_calendar_day_from_clock(clock))
    call: CanopyFluxDriverCall = _compiled_canopy_flux_driver_call(
        topology,
        atmospheric,
        grid_forcing,
        canopy,
        radiation,
        temperature,
        bare_ground,
        clock,
        inputs,
        parameters,
        state,
        jnp.float64(2.0),
        declination,
    )
    result = canopy_fluxes_runtime_parallel(
        call.topology,
        call.fast_inputs.energy,
        call.hydraulic_coupling,
        call.energy_initial,
        call.fast_inputs.postprocess,
        call.luna_inputs,
        call.synchronized_state,
        call.previous_state.root_zone_wetness_patch,
        call.active_parameters,
        device_count=device_count,
        compact_static_filter=True,
    )
    plant = process_lifetime_jit(retain_inactive_plant_state)(
        call.fast_inputs.energy.active_patch, result.state.plant, call.previous_state.plant,
    )
    return result._replace(state=result.state._replace(plant=plant))


def _pack_voc_inputs(inputs: VOCInputs, plan: PatchShardPlan) -> VOCInputs:

    return VOCInputs(
        *(
            _pack_patch_array(value, plan) if name in _VOC_PATCH_INPUT_FIELDS else value
            for name, value in zip(VOCInputs._fields, inputs, strict=True)
        )
    )


def _reshard_voc_inputs(
    inputs: VOCInputs,
    devices: tuple[jax.Device, ...],
) -> VOCInputs:

    return VOCInputs(
        *(
            _reshard_tree_leading_axis(value, devices, "clm_voc_device")
            if name in _VOC_PATCH_INPUT_FIELDS
            else value
            for name, value in zip(VOCInputs._fields, inputs, strict=True)
        )
    )


@cache
def _mapped_voc_emission(device_count: int):
    devices = tuple(jax.local_devices()[:device_count])
    if len(devices) != device_count:
        raise ValueError(
            f"requested {device_count} devices but JAX exposes only {jax.local_device_count()}"
        )
    return jax.pmap(
        voc_emission,
        axis_name="clm_voc_device",
        in_axes=(
            _TOPOLOGY_AXES,
            0,
            _VOC_INPUT_AXES,
            None,
        ),
        out_axes=0,
        devices=devices,
    )


_compiled_voc_emission = jit_unique(voc_emission, forward_inputs=True)
_compiled_voc_driver_inputs = jit_unique(assemble_voc_driver_inputs, forward_inputs=True)


def solve_voc_emission_parallel(
    topology: SubgridTopology,
    state: VOCState,
    inputs: VOCInputs,
    parameters: VOCParameters,
    *,
    device_count: int | None = None,
) -> VOCEmissionResult:

    patch_count = int(topology.patch.gridcell.shape[0])
    available = jax.local_device_count()
    selected = available if device_count is None else int(device_count)
    if selected < 1:
        raise ValueError("device_count must be positive")
    if selected > available:
        raise ValueError(f"requested {selected} devices but JAX exposes only {available}")
    if _single_gpu_direct_physics_enabled(selected):
        return _compiled_voc_emission(topology, state, inputs, parameters)
    plan = plan_patch_shards(patch_count, selected)
    devices = tuple(jax.local_devices()[:selected])

    patch_topology = _pack_patch_tree(topology.patch, plan)
    valid = jnp.arange(plan.device_count * plan.patches_per_device) < plan.patch_count
    valid = jnp.reshape(valid, (plan.device_count, plan.patches_per_device))
    patch_topology = patch_topology._replace(active=patch_topology.active & valid)
    patch_topology = _reshard_tree_leading_axis(
        patch_topology,
        devices,
        "clm_voc_device",
    )
    mapped_topology = topology._replace(patch=patch_topology)
    mapped_state = _reshard_tree_leading_axis(
        _pack_patch_tree(state, plan),
        devices,
        "clm_voc_device",
    )
    mapped_inputs = _pack_voc_inputs(inputs, plan)
    mapped_inputs = mapped_inputs._replace(
        soil_patch=mapped_inputs.soil_patch & valid,
    )
    mapped_inputs = _reshard_voc_inputs(mapped_inputs, devices)
    mapped = _mapped_voc_emission(selected)(
        mapped_topology,
        mapped_state,
        mapped_inputs,
        parameters,
    )
    return _unpack_patch_tree_on_device(mapped, plan, jax.local_devices()[0])


def voc_driver_step_parallel(
    topology: SubgridTopology,
    grid_forcing: Any,
    atmospheric: Any,
    radiation: Any,
    temperature: Any,
    canopy_flux: Any,
    accumulators: Any,
    inputs: VOCDriverInputs,
    state: VOCState,
    parameters: VOCParameters,
    *,
    device_count: int | None = None,
) -> VOCDriverResult:

    assemble = (
        _compiled_voc_driver_inputs
        if _single_gpu_direct_physics_enabled(
            jax.local_device_count() if device_count is None else int(device_count)
        )
        else assemble_voc_driver_inputs
    )
    dynamic_inputs = assemble(
        topology,
        grid_forcing,
        atmospheric,
        radiation,
        temperature,
        canopy_flux,
        accumulators,
        inputs,
    )
    emission = solve_voc_emission_parallel(
        topology,
        state,
        dynamic_inputs,
        parameters,
        device_count=device_count,
    )
    return VOCDriverResult(
        inputs=dynamic_inputs,
        emission=emission,
        state=emission.state,
    )


def _single_gpu_direct_physics_enabled(device_count: int) -> bool:
    return device_count == 1 and jax.default_backend() == "gpu"


def _urban_flux_driver_core(
    call: UrbanFluxDriverCall,
) -> tuple[UrbanTurbulenceResult, UrbanSurfaceFluxResult]:

    turbulence = urban_turbulent_fluxes(
        call.step.turbulence_inputs,
        call.step.turbulence_state,
        call.parameters.maximum_stable_zeta,
        call.parameters.building_temperature_method,
    )
    surface = urban_surface_fluxes(
        call.step.turbulence_inputs,
        call.step.surface_inputs,
        turbulence,
        call.parameters.building_temperature_method,
    )
    diagnostics = turbulence.diagnostics
    for name in _URBAN_ITERATION_DIAGNOSTICS:
        diagnostics = diagnostics._replace(**{name: jnp.moveaxis(getattr(diagnostics, name), 1, 0)})
    return turbulence._replace(diagnostics=diagnostics), surface


def _urban_flux_soil_temperature_fields_core(
    call: UrbanFluxDriverCall,
) -> UrbanFluxSoilTemperatureFields:

    turbulence = urban_turbulent_fluxes(
        call.step.turbulence_inputs,
        call.step.turbulence_state,
        call.parameters.maximum_stable_zeta,
        call.parameters.building_temperature_method,
    )
    surface = urban_surface_fluxes(
        call.step.turbulence_inputs,
        call.step.surface_inputs,
        turbulence,
        call.parameters.building_temperature_method,
    )
    energy = surface.energy
    water = surface.water
    diagnostics = turbulence.diagnostics
    return UrbanFluxSoilTemperatureFields(
        ground_energy_derivative_surface=energy.ground_energy_derivative_surface,
        downward_longwave_below_canopy_surface=(energy.downward_longwave_below_canopy_surface),
        upward_longwave_above_canopy_surface=(energy.upward_longwave_above_canopy_surface),
        ground_sensible_derivative_surface=(energy.ground_sensible_derivative_surface),
        ground_latent_derivative_surface=energy.ground_latent_derivative_surface,
        sensible_heat_snow_surface=energy.sensible_heat_snow_surface,
        sensible_heat_soil_surface=energy.sensible_heat_soil_surface,
        sensible_heat_surface_water_surface=(energy.sensible_heat_surface_water_surface),
        sensible_heat_ground_surface=energy.sensible_heat_ground_surface,
        soil_evaporation_surface=water.soil_evaporation_surface,
        vegetation_transpiration_surface=water.vegetation_transpiration_surface,
        vegetation_evaporation_surface=water.vegetation_evaporation_surface,
        vegetation_sensible_heat_surface=jnp.zeros_like(energy.sensible_heat_ground_surface),
        waste_heat_landunit=diagnostics.waste_heat_landunit,
        heat_from_air_conditioning_landunit=(diagnostics.heat_from_air_conditioning_landunit),
        traffic_heat_landunit=diagnostics.traffic_heat_landunit,
        canopy_air_temperature_landunit=(turbulence.state.canopy_air_temperature_landunit),
        internal_building_temperature_landunit=(surface.internal_building_temperature_landunit),
        air_conditioning_flux_landunit=(call.previous_state.air_conditioning_flux_landunit),
        heating_flux_landunit=call.previous_state.heating_flux_landunit,
    )


_URBAN_ITERATION_DIAGNOSTICS = (
    "stability_wind_iteration_landunit",
    "monin_obukhov_length_iteration_landunit",
    "friction_velocity_iteration_landunit",
    "thermal_resistance_iteration_landunit",
    "canopy_air_temperature_iteration_landunit",
    "canopy_air_specific_humidity_iteration_landunit",
    "surface_latent_conductance_iteration",
    "surface_wet_fraction_iteration",
)
_URBAN_FLUX_CALL_AXES = UrbanFluxDriverCall(
    step=0,
    parameters=None,
    previous_state=0,
)


def _restore_urban_iteration_axes(
    turbulence: UrbanTurbulenceResult,
) -> UrbanTurbulenceResult:
    diagnostics = turbulence.diagnostics
    for name in _URBAN_ITERATION_DIAGNOSTICS:
        diagnostics = diagnostics._replace(**{name: jnp.moveaxis(getattr(diagnostics, name), 0, 1)})
    return turbulence._replace(diagnostics=diagnostics)


def _urban_flux_driver_direct_core(call: UrbanFluxDriverCall):
    turbulence, surface = _urban_flux_driver_core(call)
    return _restore_urban_iteration_axes(turbulence), surface


@cache
def _mapped_urban_flux_driver_core(device_count: int):
    devices = tuple(jax.local_devices()[:device_count])
    if len(devices) != device_count:
        raise ValueError(
            f"requested {device_count} devices but JAX exposes only {jax.local_device_count()}"
        )
    return jax.pmap(
        _urban_flux_driver_core,
        axis_name="clm_urban_device",
        in_axes=(_URBAN_FLUX_CALL_AXES,),
        out_axes=0,
        devices=devices,
    )


@cache
def _mapped_urban_flux_soil_temperature_fields(device_count: int):
    devices = tuple(jax.local_devices()[:device_count])
    if len(devices) != device_count:
        raise ValueError(
            f"requested {device_count} devices but JAX exposes only {jax.local_device_count()}"
        )
    return jax.pmap(
        _urban_flux_soil_temperature_fields_core,
        axis_name="clm_urban_device",
        in_axes=(_URBAN_FLUX_CALL_AXES,),
        out_axes=0,
        devices=devices,
    )


def solve_urban_flux_driver_call_parallel(
    call: UrbanFluxDriverCall,
    *,
    device_count: int | None = None,
) -> tuple[UrbanTurbulenceResult, UrbanSurfaceFluxResult]:

    landunit_count = int(call.step.turbulence_inputs.active_landunit.shape[0])
    if landunit_count < 1:
        raise ValueError("parallel UrbanFluxes requires at least one urban landunit")
    available = jax.local_device_count()
    selected = available if device_count is None else int(device_count)
    if selected < 1:
        raise ValueError("device_count must be positive")
    if selected > available:
        raise ValueError(f"requested {selected} devices but JAX exposes only {available}")
    if _single_gpu_direct_physics_enabled(selected):
        return process_lifetime_jit(_urban_flux_driver_direct_core)(call)
    plan = plan_patch_shards(landunit_count, selected)
    packed_step = _pack_patch_tree(call.step, plan)
    valid = jnp.arange(plan.device_count * plan.patches_per_device) < plan.patch_count
    valid = jnp.reshape(valid, (plan.device_count, plan.patches_per_device))
    packed_step = packed_step._replace(
        turbulence_inputs=packed_step.turbulence_inputs._replace(
            active_landunit=(packed_step.turbulence_inputs.active_landunit & valid)
        )
    )
    packed_call = UrbanFluxDriverCall(
        step=packed_step,
        parameters=call.parameters,
        previous_state=_pack_patch_tree(call.previous_state, plan),
    )
    packed_call = _reshard_tree_for_pmap(
        packed_call,
        _URBAN_FLUX_CALL_AXES,
        tuple(jax.local_devices()[:selected]),
        "clm_urban_device",
    )
    turbulence, surface = _mapped_urban_flux_driver_core(selected)(packed_call)
    turbulence, surface = _unpack_patch_tree_on_device(
        (turbulence, surface),
        plan,
        jax.local_devices()[0],
    )
    return (
        _restore_urban_iteration_axes(turbulence),
        surface,
    )


@cache
def _compiled_compact_assembly(function):
    return jit_discrete_parameters(function, forward_inputs=True)


def _compact_assembly(function, *arguments):
    if jax.default_backend() == "gpu":
        return _compiled_compact_assembly(function)(*arguments)
    return function(*arguments)


def urban_flux_driver_step_parallel(
    topology: Any,
    grid_forcing: Any,
    canopy: Any,
    temperature: Any,
    inputs: Any,
    parameters: Any,
    state: Any,
    *,
    device_count: int | None = None,
) -> UrbanFluxDriverResult:

    call = _compact_assembly(assemble_urban_flux_driver_call,
        topology,
        grid_forcing,
        canopy,
        temperature,
        inputs,
        parameters,
        state,
    )
    turbulence, surface = solve_urban_flux_driver_call_parallel(
        call,
        device_count=device_count,
    )
    return complete_urban_flux_driver_call(call, turbulence, surface)


def solve_urban_flux_soil_temperature_fields_parallel(
    call: UrbanFluxDriverCall,
    *,
    device_count: int | None = None,
) -> UrbanFluxSoilTemperatureFields:

    landunit_count = int(call.step.turbulence_inputs.active_landunit.shape[0])
    if landunit_count < 1:
        raise ValueError("parallel UrbanFluxes requires at least one urban landunit")
    available = jax.local_device_count()
    selected = available if device_count is None else int(device_count)
    if selected < 1:
        raise ValueError("device_count must be positive")
    if selected > available:
        raise ValueError(f"requested {selected} devices but JAX exposes only {available}")
    if _single_gpu_direct_physics_enabled(selected):
        return process_lifetime_jit(_urban_flux_soil_temperature_fields_core)(call)
    plan = plan_patch_shards(landunit_count, selected)
    packed_step = _pack_patch_tree(call.step, plan)
    valid = jnp.arange(plan.device_count * plan.patches_per_device) < plan.patch_count
    valid = jnp.reshape(valid, (plan.device_count, plan.patches_per_device))
    packed_step = packed_step._replace(
        turbulence_inputs=packed_step.turbulence_inputs._replace(
            active_landunit=packed_step.turbulence_inputs.active_landunit & valid
        )
    )
    packed_call = UrbanFluxDriverCall(
        step=packed_step,
        parameters=call.parameters,
        previous_state=_pack_patch_tree(call.previous_state, plan),
    )
    packed_call = _reshard_tree_for_pmap(
        packed_call,
        _URBAN_FLUX_CALL_AXES,
        tuple(jax.local_devices()[:selected]),
        "clm_urban_device",
    )
    projected = _mapped_urban_flux_soil_temperature_fields(selected)(packed_call)
    return _unpack_patch_tree_on_device(
        projected,
        plan,
        jax.local_devices()[0],
    )


def urban_flux_soil_temperature_driver_step_parallel(
    topology: Any,
    grid_forcing: Any,
    canopy: Any,
    temperature: Any,
    inputs: Any,
    parameters: Any,
    state: Any,
    *,
    device_count: int | None = None,
) -> UrbanFluxSoilTemperatureFields:

    call = _compact_assembly(assemble_urban_flux_driver_call,
        topology,
        grid_forcing,
        canopy,
        temperature,
        inputs,
        parameters,
        state,
    )
    return solve_urban_flux_soil_temperature_fields_parallel(
        call,
        device_count=device_count,
    )


def _lake_flux_driver_core(call: LakeFluxDriverCall) -> LakeFluxResult:

    result = lake_fluxes(
        call.inputs,
        call.previous_state.surface,
        call.parameters.maximum_stable_zeta,
    )
    diagnostics = result.diagnostics
    for name in diagnostics._fields:
        diagnostics = diagnostics._replace(**{name: jnp.moveaxis(getattr(diagnostics, name), 1, 0)})
    return result._replace(diagnostics=diagnostics)


def _restore_lake_iteration_axes(result: LakeFluxResult) -> LakeFluxResult:
    diagnostics = result.diagnostics
    for name in diagnostics._fields:
        diagnostics = diagnostics._replace(**{name: jnp.moveaxis(getattr(diagnostics, name), 0, 1)})
    return result._replace(diagnostics=diagnostics)


def _lake_flux_driver_direct_core(call: LakeFluxDriverCall) -> LakeFluxResult:
    return _restore_lake_iteration_axes(_lake_flux_driver_core(call))


_LAKE_FLUX_CALL_AXES = LakeFluxDriverCall(
    inputs=0,
    parameters=None,
    previous_state=0,
)


@cache
def _mapped_lake_flux_driver_core(device_count: int):
    devices = tuple(jax.local_devices()[:device_count])
    if len(devices) != device_count:
        raise ValueError(
            f"requested {device_count} devices but JAX exposes only {jax.local_device_count()}"
        )
    return jax.pmap(
        _lake_flux_driver_core,
        axis_name="clm_lake_device",
        in_axes=(_LAKE_FLUX_CALL_AXES,),
        out_axes=0,
        devices=devices,
    )


def solve_lake_flux_driver_call_parallel(
    call: LakeFluxDriverCall,
    *,
    device_count: int | None = None,
) -> LakeFluxResult:

    lake_count = int(call.inputs.active_lake.shape[0])
    if lake_count < 1:
        raise ValueError("parallel LakeFluxes requires at least one active lake")
    available = jax.local_device_count()
    selected = available if device_count is None else int(device_count)
    if selected < 1:
        raise ValueError("device_count must be positive")
    if selected > available:
        raise ValueError(f"requested {selected} devices but JAX exposes only {available}")
    if _single_gpu_direct_physics_enabled(selected):
        return process_lifetime_jit(_lake_flux_driver_direct_core)(call)
    plan = plan_patch_shards(lake_count, selected)
    packed_inputs = _pack_patch_tree(call.inputs, plan)
    valid = jnp.arange(plan.device_count * plan.patches_per_device) < plan.patch_count
    valid = jnp.reshape(valid, (plan.device_count, plan.patches_per_device))
    packed_inputs = packed_inputs._replace(
        active_lake=packed_inputs.active_lake & valid,
    )
    packed_call = LakeFluxDriverCall(
        inputs=packed_inputs,
        parameters=call.parameters,
        previous_state=_pack_patch_tree(call.previous_state, plan),
    )
    packed_call = _reshard_tree_for_pmap(
        packed_call,
        _LAKE_FLUX_CALL_AXES,
        tuple(jax.local_devices()[:selected]),
        "clm_lake_device",
    )
    result = _mapped_lake_flux_driver_core(selected)(packed_call)
    result = _unpack_patch_tree_on_device(
        result,
        plan,
        jax.local_devices()[0],
    )
    return _restore_lake_iteration_axes(result)


def lake_flux_driver_step_parallel(
    topology: Any,
    atmospheric: Any,
    grid_forcing: Any,
    canopy: Any,
    radiation: Any,
    temperature: Any,
    inputs: Any,
    parameters: Any,
    state: Any,
    *,
    device_count: int | None = None,
) -> LakeFluxDriverResult:

    call = _compact_assembly(assemble_lake_flux_driver_call,
        topology,
        atmospheric,
        grid_forcing,
        canopy,
        radiation,
        temperature,
        inputs,
        parameters,
        state,
    )
    fluxes = solve_lake_flux_driver_call_parallel(
        call,
        device_count=device_count,
    )
    return complete_lake_flux_driver_call(call, fluxes)


@cache
def _mapped_lake_temperature_driver_core(
    device_count: int,
    time_step_seconds: float,
):
    devices = tuple(jax.local_devices()[:device_count])
    if len(devices) != device_count:
        raise ValueError(
            f"requested {device_count} devices but JAX exposes only {jax.local_device_count()}"
        )

    def core(call: LakeTemperatureDriverCall) -> LakeTemperatureResult:
        return lake_temperature(
            call.inputs,
            call.geometry,
            call.state,
            time_step_seconds=time_step_seconds,
        )

    call_axes = LakeTemperatureDriverCall(inputs=0, geometry=0, state=0)
    return jax.pmap(
        core,
        axis_name="clm_lake_device",
        in_axes=(call_axes,),
        out_axes=0,
        devices=devices,
    )


def solve_lake_temperature_driver_call_parallel(
    call: LakeTemperatureDriverCall,
    *,
    time_step_seconds: float = 1800.0,
    device_count: int | None = None,
) -> LakeTemperatureResult:

    lake_count = int(call.inputs.active_lake.shape[0])
    if lake_count < 1:
        raise ValueError("parallel LakeTemperature requires at least one active lake")
    available = jax.local_device_count()
    selected = available if device_count is None else int(device_count)
    if selected < 1:
        raise ValueError("device_count must be positive")
    if selected > available:
        raise ValueError(f"requested {selected} devices but JAX exposes only {available}")
    step_seconds = float(time_step_seconds)
    if step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")
    if _single_gpu_direct_physics_enabled(selected):
        return process_lifetime_jit(
            lake_temperature, static_argnames=("time_step_seconds",),
        )(call.inputs, call.geometry, call.state, time_step_seconds=step_seconds)
    plan = plan_patch_shards(lake_count, selected)
    devices = tuple(jax.local_devices()[:selected])
    packed = _pack_patch_tree(call, plan)
    valid = jnp.arange(plan.device_count * plan.patches_per_device) < plan.patch_count
    valid = jnp.reshape(valid, (plan.device_count, plan.patches_per_device))
    packed = packed._replace(
        inputs=packed.inputs._replace(active_lake=packed.inputs.active_lake & valid)
    )
    packed = _reshard_tree_leading_axis(packed, devices, "clm_lake_device")
    mapped = _mapped_lake_temperature_driver_core(selected, step_seconds)(packed)
    return _unpack_patch_tree_on_device(mapped, plan, jax.local_devices()[0])


def lake_temperature_driver_step_parallel(
    topology: Any,
    canopy: Any,
    radiation: Any,
    temperature: Any,
    lake_flux: Any,
    inputs: Any,
    state: Any,
    *,
    time_step_seconds: float = 1800.0,
    device_count: int | None = None,
) -> LakeTemperatureDriverResult:

    call = _compact_assembly(assemble_lake_temperature_driver_call,
        topology,
        canopy,
        radiation,
        temperature,
        lake_flux,
        inputs,
        state,
    )
    result = solve_lake_temperature_driver_call_parallel(
        call,
        time_step_seconds=time_step_seconds,
        device_count=device_count,
    )
    return LakeTemperatureDriverResult(
        call=call,
        temperature=result,
        state=result.state,
    )


@cache
def _mapped_soil_temperature_column_case(
    device_count: int,
    time_step_seconds: float,
    maximum_patches_per_column: int,
):

    devices = tuple(jax.local_devices()[:device_count])
    if len(devices) != device_count:
        raise ValueError(
            f"requested {device_count} devices but JAX exposes only {jax.local_device_count()}"
        )

    def core(inputs: SoilTemperatureActiveInputs) -> SoilTemperatureColumnResult:
        return soil_temperature_column_case(
            inputs,
            time_step_seconds=time_step_seconds,
            maximum_patches_per_column=maximum_patches_per_column,
        )

    return jax.pmap(
        core,
        axis_name="clm_soil_temperature_device",
        in_axes=(_SOIL_ACTIVE_INPUT_AXES,),
        out_axes=0,
        devices=devices,
    )


@cache
def _mapped_soil_temperature_runtime_column_case(
    device_count: int,
    time_step_seconds: float,
    maximum_patches_per_column: int,
):

    devices = tuple(jax.local_devices()[:device_count])
    if len(devices) != device_count:
        raise ValueError(
            f"requested {device_count} devices but JAX exposes only {jax.local_device_count()}"
        )

    def core(inputs: SoilTemperatureActiveInputs) -> SoilTemperatureRuntimeColumnResult:
        return soil_temperature_runtime_column_case(
            inputs,
            time_step_seconds=time_step_seconds,
            maximum_patches_per_column=maximum_patches_per_column,
        )

    return jax.pmap(
        core,
        axis_name="clm_soil_temperature_device",
        in_axes=(_SOIL_ACTIVE_INPUT_AXES,),
        out_axes=0,
        devices=devices,
    )


@cache
def _compiled_building_temperature(
    time_step_seconds: float,
    heating_air_conditioning_enabled: bool,
):

    def core(inputs):
        return active_building_temperature(
            inputs,
            time_step_seconds=time_step_seconds,
            heating_air_conditioning_enabled=heating_air_conditioning_enabled,
        )

    return jit_unique(
        core,
        compiler_options={
            "xla_cpu_enable_fast_math": False,
        },
    )


def _solve_compact_building_temperature(
    inputs,
    *,
    time_step_seconds: float,
    heating_air_conditioning_enabled: bool,
) -> UrbanBuildingTemperatureResult:

    if isinstance(inputs.active_urban, jax.core.Tracer):
        return process_lifetime_jit(
            building_temperature,
            static_argnames=("time_step_seconds", "heating_air_conditioning_enabled"),
        )(
            inputs,
            time_step_seconds=float(time_step_seconds),
            heating_air_conditioning_enabled=bool(heating_air_conditioning_enabled),
        )
    active_indices = np.flatnonzero(np.asarray(inputs.active_urban, dtype=np.bool_))
    compact = _gather_building_temperature_inputs(inputs, active_indices)
    active = _compiled_building_temperature(
        float(time_step_seconds),
        bool(heating_air_conditioning_enabled),
    )(compact)
    return _restore_building_temperature(inputs, active_indices, active)


@jax.jit
def _gather_building_temperature_inputs(inputs, active_indices):
    return jax.tree.map(lambda value: value[active_indices], inputs)


@jax.jit
def _restore_building_temperature(inputs, active_indices, active):

    def restore(previous, updated):
        return jnp.asarray(previous).at[active_indices].set(updated)

    number_of_landunits = int(inputs.active_urban.shape[0])
    matrix = jnp.zeros(
        (
            number_of_landunits,
            NUMBER_OF_BUILDING_EQUATIONS,
            NUMBER_OF_BUILDING_EQUATIONS,
        ),
        dtype=jnp.float64,
    ).at[active_indices].set(active.coefficient_matrix)
    right_hand_side = jnp.zeros(
        (number_of_landunits, NUMBER_OF_BUILDING_EQUATIONS),
        dtype=jnp.float64,
    ).at[active_indices].set(active.right_hand_side)
    pivot_indices = jnp.broadcast_to(
        jnp.arange(NUMBER_OF_BUILDING_EQUATIONS, dtype=jnp.int32),
        (number_of_landunits, NUMBER_OF_BUILDING_EQUATIONS),
    ).at[active_indices].set(active.pivot_indices)
    nonsingular = jnp.zeros((number_of_landunits,), dtype=jnp.bool_).at[
        active_indices
    ].set(active.nonsingular)
    return UrbanBuildingTemperatureResult(
        roof_inner_temperature=restore(
            inputs.roof_inner_temperature, active.roof_inner_temperature
        ),
        sunwall_inner_temperature=restore(
            inputs.sunwall_inner_temperature, active.sunwall_inner_temperature
        ),
        shadewall_inner_temperature=restore(
            inputs.shadewall_inner_temperature, active.shadewall_inner_temperature
        ),
        floor_temperature=restore(inputs.floor_temperature, active.floor_temperature),
        building_temperature=restore(
            inputs.building_temperature, active.building_temperature
        ),
        unconstrained_building_temperature=restore(
            inputs.building_temperature, active.unconstrained_building_temperature
        ),
        building_heat_flux=restore(
            inputs.previous_building_heat_flux, active.building_heat_flux
        ),
        air_conditioning_flux=restore(
            inputs.previous_air_conditioning_flux, active.air_conditioning_flux
        ),
        heating_flux=restore(inputs.previous_heating_flux, active.heating_flux),
        coefficient_matrix=matrix,
        right_hand_side=right_hand_side,
        pivot_indices=pivot_indices,
        nonsingular=nonsingular,
    )


_compiled_coupled_building_inputs = jit_unique(
    assemble_coupled_building_temperature_inputs, forward_inputs=True,
)
_compiled_soil_temperature_driver_inputs = jit_unique(
    assemble_soil_temperature_driver_inputs, forward_inputs=True,
)


def _single_gpu_direct_soil_enabled(device_count: int) -> bool:
    return device_count == 1 and jax.default_backend() == "gpu"


_SOIL_DIRECT_LAYOUT_METADATA: dict[int, tuple[Any, int, int, int]] = {}


def _direct_soil_patch_bound(topology: SubgridTopology, requested: int) -> int:
    column_count = int(topology.column.active.shape[0])
    patch_count = int(topology.patch.active.shape[0])
    if column_count < 1 or patch_count < 1:
        raise ValueError("column/patch execution requires nonempty native topology")
    column_map = topology.patch.column
    if column_map.shape != (patch_count,):
        raise ValueError("patch column map does not match the native patch count")
    if isinstance(column_map, jax.core.Tracer):
        return requested
    key = id(column_map)
    cached = _SOIL_DIRECT_LAYOUT_METADATA.get(key)
    if cached is not None and cached[0]() is column_map and cached[1:3] == (
        column_count, patch_count,
    ):
        return min(requested, cached[3])

    native = np.asarray(column_map, dtype=np.int64)
    if np.any((native < 0) | (native >= column_count)):
        raise ValueError("patch topology refers to a column outside the native domain")
    maximum = int(np.bincount(native, minlength=column_count).max())
    if isinstance(column_map, jax.Array):
        def forget(reference):
            current = _SOIL_DIRECT_LAYOUT_METADATA.get(key)
            if current is not None and current[0] is reference:
                _SOIL_DIRECT_LAYOUT_METADATA.pop(key, None)

        _SOIL_DIRECT_LAYOUT_METADATA[key] = (
            ref(column_map, forget), column_count, patch_count, maximum,
        )
    return min(requested, maximum)


def solve_soil_temperature_active_case_parallel(
    inputs: SoilTemperatureActiveInputs,
    *,
    time_step_seconds: float = 1800.0,
    maximum_patches_per_column: int = 79,
    heating_air_conditioning_enabled: bool = True,
    device_count: int | None = None,
    memory_barrier: Callable[[], object] | None = None,
    retain_diagnostics: bool = True,
) -> SoilTemperatureActiveResult | SoilTemperatureRuntimeResult:

    available = jax.local_device_count()
    selected = available if device_count is None else int(device_count)
    if selected < 1:
        raise ValueError("device_count must be positive")
    if selected > available:
        raise ValueError(f"requested {selected} devices but JAX exposes only {available}")
    step_seconds = float(time_step_seconds)
    if step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")
    if maximum_patches_per_column < 0:
        raise ValueError("maximum_patches_per_column must be nonnegative")

    direct = _single_gpu_direct_soil_enabled(selected)
    mapped = packed = None
    if direct:
        maximum_patches_per_column = _direct_soil_patch_bound(
            inputs.topology, maximum_patches_per_column,
        )
    else:
        plan = plan_column_patch_shards(inputs.topology, selected)
        patch_counts = np.bincount(
            np.asarray(inputs.topology.patch.column, dtype=np.int64),
            minlength=plan.column_count,
        )
        maximum_patches_per_column = min(maximum_patches_per_column, int(patch_counts.max()))
        packed = _pack_soil_temperature_active_inputs_compiled(inputs, plan)
        devices = tuple(jax.local_devices()[:selected])
        packed = _reshard_tree_for_pmap(
            packed,
            _SOIL_ACTIVE_INPUT_AXES,
            devices,
            "clm_soil_temperature_device",
        )
    if retain_diagnostics:
        if direct:
            columns = process_lifetime_jit(
                soil_temperature_column_case,
                static_argnames=("time_step_seconds", "maximum_patches_per_column"),
            )(inputs, time_step_seconds=step_seconds,
              maximum_patches_per_column=int(maximum_patches_per_column))
        else:
            mapped = _mapped_soil_temperature_column_case(
                selected,
                step_seconds,
                int(maximum_patches_per_column),
            )(packed)
            mapped = gather_mapped_result(mapped, devices[0])
            columns = _unpack_soil_temperature_column_result(mapped, plan)
        coupled_building = _compiled_coupled_building_inputs(
            inputs,
            columns.thermal_properties.interface_thermal_conductivity,
            columns.soil_snow_phase_change.soil_snow_temperature,
        )
        if memory_barrier is not None:
            stage_barrier((columns, coupled_building))
            del mapped, packed
            memory_barrier()
    else:
        if direct:
            columns = process_lifetime_jit(
                soil_temperature_runtime_column_case,
                static_argnames=("time_step_seconds", "maximum_patches_per_column"),
            )(inputs, time_step_seconds=step_seconds,
              maximum_patches_per_column=int(maximum_patches_per_column))
        else:
            mapped = _mapped_soil_temperature_runtime_column_case(
                selected,
                step_seconds,
                int(maximum_patches_per_column),
            )(packed)
            mapped = gather_mapped_result(mapped, devices[0])
            columns = _unpack_soil_temperature_runtime_column_result_compiled(mapped, plan)
        coupled_building = _compiled_coupled_building_inputs(
            inputs,
            columns.thermal_properties.interface_thermal_conductivity,
            columns.soil_snow_phase_change.soil_snow_temperature,
        )
        thermal_properties = columns.thermal_properties
        ground_heat_flux = columns.ground_heat_flux
        heat_diffusion = columns.heat_diffusion
        standing_water_heat_capacity = columns.standing_water_heat_capacity
        postsolve_flux = columns.postsolve_flux
        standing_water_phase_change = columns.standing_water_phase_change
        soil_snow_phase_change = columns.soil_snow_phase_change
        final_diagnostics = columns.final_diagnostics
        if memory_barrier is not None:
            stage_barrier(
                (
                    coupled_building,
                    thermal_properties,
                    ground_heat_flux,
                    heat_diffusion,
                    standing_water_heat_capacity,
                    postsolve_flux,
                    standing_water_phase_change,
                    soil_snow_phase_change,
                    final_diagnostics,
                )
            )
            del mapped, packed, columns
            memory_barrier()
    building = _solve_compact_building_temperature(
        coupled_building,
        time_step_seconds=step_seconds,
        heating_air_conditioning_enabled=bool(heating_air_conditioning_enabled),
    )
    if not retain_diagnostics:
        return SoilTemperatureRuntimeResult(
            thermal_properties=thermal_properties,
            ground_heat_flux=ground_heat_flux,
            heat_diffusion=heat_diffusion,
            standing_water_heat_capacity=standing_water_heat_capacity,
            postsolve_flux=postsolve_flux,
            standing_water_phase_change=standing_water_phase_change,
            soil_snow_phase_change=soil_snow_phase_change,
            building_temperature=building,
            final_diagnostics=final_diagnostics,
        )
    return SoilTemperatureActiveResult(
        thermal_properties=columns.thermal_properties,
        ground_heat_flux=columns.ground_heat_flux,
        heat_diffusion=columns.heat_diffusion,
        standing_water_heat_capacity=columns.standing_water_heat_capacity,
        standing_water_thickness=columns.standing_water_thickness,
        rhs=columns.rhs,
        matrix=columns.matrix,
        linear_solve=columns.linear_solve,
        solution_update=columns.solution_update,
        postsolve_flux=columns.postsolve_flux,
        standing_water_phase_change=columns.standing_water_phase_change,
        soil_snow_phase_change=columns.soil_snow_phase_change,
        building_temperature=building,
        final_diagnostics=columns.final_diagnostics,
    )


def soil_temperature_driver_step_parallel(
    topology: SubgridTopology,
    atmospheric: Any,
    canopy: Any,
    radiation: Any,
    temperature: Any,
    canopy_flux: Any,
    urban: Any,
    inputs: SoilTemperatureDriverInputs,
    *,
    time_step_seconds: float = 1800.0,
    maximum_patches_per_column: int = 79,
    heating_air_conditioning_enabled: bool = True,
    device_count: int | None = None,
    memory_barrier: Callable[[], object] | None = None,
    retain_diagnostics: bool = True,
) -> SoilTemperatureDriverResult:

    assemble = (
        _compiled_soil_temperature_driver_inputs
        if jax.local_devices()[0].platform == "gpu"
        else assemble_soil_temperature_driver_inputs
    )
    assembled = assemble(
        topology,
        atmospheric,
        canopy,
        radiation,
        temperature,
        canopy_flux,
        urban,
        inputs,
    )
    result = solve_soil_temperature_active_case_parallel(
        assembled,
        time_step_seconds=time_step_seconds,
        maximum_patches_per_column=maximum_patches_per_column,
        heating_air_conditioning_enabled=heating_air_conditioning_enabled,
        device_count=device_count,
        memory_barrier=memory_barrier,
        retain_diagnostics=retain_diagnostics,
    )
    return SoilTemperatureDriverResult(inputs=assembled, temperature=result)
