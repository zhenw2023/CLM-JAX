
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from clm_jax.balance_check_execution import BalanceCheckExecutor
from clm_jax.carbon_nitrogen_balance_execution import (
    CarbonNitrogenBalanceExecutor,
)
from clm_jax.compilation import jit_unique, process_lifetime_jit
from clm_jax.dry_deposition_execution import DryDepositionExecutor
from clm_jax.ecosystem_driver_execution import EcosystemDynamicsDriverExecutor
from clm_jax.ecosystem_post_drainage_execution import (
    EcosystemDynamicsPostDrainageExecutor,
)
from clm_jax.end_of_timestep_execution import EndOfTimestepExecutor
from clm_jax.execution_observer import stage_barrier
from clm_jax.forcing.bgc import interpolate_bgc_external_forcing
from clm_jax.hydrology_drainage_execution import HydrologyDrainageDriverExecutor
from clm_jax.io.active_layer import (
    assemble_active_layer_inputs,
    load_active_layer_native_data,
)
from clm_jax.io.bgc_timestep_initialization import (
    assemble_bgc_timestep_initialization_state,
    load_bgc_timestep_initialization_native_data,
)
from clm_jax.io.carbon_nitrogen_balance_initialization import (
    load_carbon_nitrogen_balance_initialization_native_data,
)
from clm_jax.io.decomposition_rates import decomposition_feedback_from_methane
from clm_jax.io.dry_deposition import load_dry_deposition_configuration
from clm_jax.io.ecosystem_dynamics_inputs import (
    assemble_ecosystem_dynamics_complete_inputs,
)
from clm_jax.io.ecosystem_dynamics_post_drainage import (
    assemble_ecosystem_dynamics_post_drainage_call_from_carry,
    load_ecosystem_dynamics_post_drainage_native_data,
    post_drainage_persistent_state,
)
from clm_jax.io.ecosystem_dynamics_state import (
    load_ecosystem_dynamics_initial_state_native_data,
    load_ecosystem_dynamics_native_children,
)
from clm_jax.io.ecosystem_fast_boundary import (
    apply_early_ecosystem_vertical_profile,
    assemble_ecosystem_fast_boundary,
    synchronize_crop_growing_degree_days,
)
from clm_jax.io.end_of_timestep import load_end_of_timestep_native_data
from clm_jax.io.land_to_atmosphere import load_land_to_atmosphere_native_data
from clm_jax.io.methane_driver import load_methane_driver_native_data
from clm_jax.io.soil_biogeochem_nitrif_denitrif import nitrification_feedback_from_methane
from clm_jax.io.surface_albedo import load_surface_albedo_native_data
from clm_jax.io.urban_albedo_restart import load_urban_albedo_template
from clm_jax.land_to_atmosphere_execution import LandToAtmosphereExecutor
from clm_jax.methane_execution import MethaneDriverExecutor
from clm_jax.physics.active_layer import diagnose_active_layer
from clm_jax.physics.balance_check import BalanceCheckParameters
from clm_jax.physics.balance_check_driver import initial_balance_check_state
from clm_jax.physics.bgc_timestep_initialization import initialize_bgc_timestep
from clm_jax.physics.carbon_nitrogen_balance_check import (
    CarbonNitrogenBalanceCheckParameters,
)
from clm_jax.physics.carbon_nitrogen_balance_driver import (
    carbon_nitrogen_balance_carry,
    carbon_nitrogen_balance_carry_from_summaries,
)
from clm_jax.physics.carbon_nitrogen_balance_initialization import (
    initialize_carbon_nitrogen_balance,
)
from clm_jax.physics.end_of_timestep import (
    EndOfTimestepState,
    compose_end_of_timestep_inputs,
)
from clm_jax.physics.history_output import (
    compose_instantaneous_history,
    project_early_history_fields,
)
from clm_jax.physics.land_to_atmosphere_composition import (
    compose_land_to_atmosphere_call,
    land_to_atmosphere_post_temperature_carry,
)
from clm_jax.physics.methane_balance_initialization import (
    initialize_methane_balance,
)
from clm_jax.physics.methane_driver_composition import (
    methane_driver_carry,
    methane_driver_carry_from_state,
)
from clm_jax.physics.next_timestep import next_ecosystem_state_from_parts
from clm_jax.physics.snow_water import NUMBER_OF_SNOW_LEVELS
from clm_jax.physics.surface_albedo_composition import (
    compose_surface_albedo_call,
    surface_albedo_post_lake_carry,
)
from clm_jax.physics.timestep_carry import (
    compose_land_timestep_carry,
    post_lake_hydrology_state,
)
from clm_jax.physics.urban_albedo_composition import compose_urban_albedo_call
from clm_jax.runtime import (
    co_locate_array_leaves,
    release_completed_stage_memory,
)
from clm_jax.surface_albedo_execution import SurfaceAlbedoExecutor, SurfaceAlbedoRuntimeResult
from clm_jax.trajectory import model_time_flags
from clm_jax.urban_albedo_execution import UrbanAlbedoExecutor

_compiled_fast_boundary = jit_unique(assemble_ecosystem_fast_boundary, forward_inputs=True)
_compiled_tail_inputs = jit_unique(compose_end_of_timestep_inputs, forward_inputs=True)
_compiled_active_layer = jit_unique(diagnose_active_layer, forward_inputs=True)
_compiled_bgc_initialization = process_lifetime_jit(initialize_bgc_timestep)


def _retain_active_layer_history(result, previous, date, seconds, time_step):
    if date % 10_000 in (101, 701) and seconds // time_step == 1:
        return result
    return _retain_previous_active_layer_history(result, previous)


def _retain_previous_active_layer_history(result, previous):
    retained = {}
    for field in ("prior_year_maximum_thaw_depth_column",
                  "prior_year_maximum_thaw_depth_index_column"):
        before, after = getattr(previous, field), getattr(result.state, field)
        if before.dtype == after.dtype:
            retained[field] = before
    return result._replace(state=result.state._replace(**retained))


class LandTimestepTemporalInputs(NamedTuple):

    children: Any
    retain_previous_active_layer_history: bool


class LandTimestepResult(NamedTuple):

    carry: Any
    land_to_atmosphere: Any
    end_of_timestep: Any
    post_drainage: Any
    methane: Any
    surface_albedo: Any
    urban_albedo: Any
    history: Any | None
    diagnostics: dict[str, Any]


@dataclass
class LandTimestepBoundaryOwner:

    _boundary: Any

    def take(self):
        if self._boundary is None:
            raise RuntimeError("land timestep boundary already consumed")
        boundary, self._boundary = self._boundary, None
        return boundary


@dataclass
class PriorLandTimestepState:

    active_layer: Any
    ecosystem_pre_drainage: Any
    ecosystem_post_drainage: Any
    end_of_timestep: Any
    drainage: Any
    balance: Any
    methane: Any
    land_to_atmosphere: Any
    surface_albedo: Any
    urban_albedo: Any

    def take(self, name: str) -> Any:

        value = getattr(self, name)
        if value is None:
            raise RuntimeError(f"preceding-step state owner already consumed: {name}")
        setattr(self, name, None)
        return value

    def assert_consumed(self) -> None:
        remaining = tuple(
            name for name in self.__dataclass_fields__ if getattr(self, name) is not None
        )
        if remaining:
            raise RuntimeError(
                "preceding-step state owners were not consumed: " + ", ".join(remaining)
            )


def project_prior_land_timestep_state(carry: Any) -> PriorLandTimestepState:

    return PriorLandTimestepState(
        active_layer=carry.active_layer,
        ecosystem_pre_drainage=carry.ecosystem_pre_drainage,
        ecosystem_post_drainage=carry.ecosystem_post_drainage,
        end_of_timestep=carry.end_of_timestep,
        drainage=carry.drainage,
        balance=carry.balance,
        methane=carry.methane,
        land_to_atmosphere=carry.land_to_atmosphere,
        surface_albedo=carry.surface_albedo,
        urban_albedo=carry.urban_albedo,
    )


def _retain_record_fields(record: Any, names: set[str]) -> Any:

    return record._replace(
        **{name: None for name in record._fields if name not in names}
    )


def _compact_late_physical_sources(physical: Any) -> Any:

    atmospheric = physical.atmospheric
    atmospheric = atmospheric._replace(
        state=_retain_record_fields(atmospheric.state, {"column_topography"}),
        driver_initialization=_retain_record_fields(
            atmospheric.driver_initialization,
            {"previous_snow_water_equivalent"},
        ),
        column_forcing=_retain_record_fields(
            atmospheric.column_forcing,
            {
                "longwave_down",
                "pressure",
                "rain",
                "snow",
                "specific_humidity",
                "temperature",
            },
        ),
        exposed_vegetation=None,
    )

    canopy = physical.canopy
    canopy_hydrology = _retain_record_fields(
        canopy.state.canopy_hydrology,
        {"canopy_wet_fractions"},
    )
    canopy_state = _retain_record_fields(canopy.state, {"canopy_hydrology"})
    canopy_state = canopy_state._replace(canopy_hydrology=canopy_hydrology)
    irrigation_fluxes = _retain_record_fields(
        canopy.irrigation.fluxes,
        {"applied_column", "applied_patch"},
    )
    irrigation = _retain_record_fields(canopy.irrigation, {"fluxes"})
    irrigation = irrigation._replace(fluxes=irrigation_fluxes)
    canopy = canopy._replace(
        state=canopy_state,
        irrigation=irrigation,
        canopy_hydrology=None,
    )

    canopy_flux = physical.canopy_flux
    active = canopy_flux.state.active
    postprocess = _retain_record_fields(
        active.postprocess,
        {"canopy_water_patch", "snow_canopy_water_patch"},
    )
    photosynthesis = _retain_record_fields(
        active.photosynthesis_total,
        {"photosynthesis_patch"},
    )
    active = _retain_record_fields(
        active,
        {"photosynthesis_total", "postprocess"},
    )._replace(
        postprocess=postprocess,
        photosynthesis_total=photosynthesis,
    )
    canopy_flux_state = _retain_record_fields(
        canopy_flux.state,
        {"active", "vegetation_temperature_patch"},
    )._replace(active=active)
    canopy_soil_temperature = _retain_record_fields(
        canopy_flux.soil_temperature,
        {"ground_methane_conductance_patch", "vegetation_transpiration_patch"},
    )
    canopy_dust = _retain_record_fields(canopy_flux.dust, {"active_patch"})
    canopy_ecosystem = _retain_record_fields(
        canopy_flux.ecosystem,
        {
            "root_zone_wetness_patch",
            "transpiration_wetness_patch",
            "vegetation_exposed_patch",
        },
    )
    canopy_flux = canopy_flux._replace(
        state=canopy_flux_state,
        voc=None,
        soil_temperature=canopy_soil_temperature,
        dust=canopy_dust,
        ecosystem=canopy_ecosystem,
    )

    soil_temperature = physical.soil_temperature
    soil_inputs = _retain_record_fields(soil_temperature.inputs, {"topology"})
    temperature = soil_temperature.temperature
    ground_heat_flux = _retain_record_fields(
        temperature.ground_heat_flux,
        {"patch_state"},
    )
    ground_patch = _retain_record_fields(
        ground_heat_flux.patch_state,
        {
            "absorbed_ground_balance_patch",
            "air_conditioner_heat_patch",
            "traffic_heat_patch",
            "waste_heat_patch",
        },
    )
    ground_heat_flux = ground_heat_flux._replace(patch_state=ground_patch)
    standing_water = _retain_record_fields(
        temperature.standing_water_phase_change,
        {"water_to_ice_flux"},
    )
    temperature = _retain_record_fields(
        temperature,
        {"ground_heat_flux", "standing_water_phase_change"},
    )._replace(
        ground_heat_flux=ground_heat_flux,
        standing_water_phase_change=standing_water,
    )
    soil_temperature = soil_temperature._replace(
        inputs=soil_inputs,
        temperature=temperature,
    )

    land_to_atmosphere = _retain_record_fields(
        physical.land_to_atmosphere,
        {"reference_temperature_2m_patch", "vegetation_temperature_patch"},
    )
    return physical._replace(
        atmospheric=atmospheric,
        canopy=canopy,
        canopy_flux=canopy_flux,
        urban_flux=None,
        lake_flux=None,
        lake_temperature=None,
        soil_temperature=soil_temperature,
        land_to_atmosphere=land_to_atmosphere,
    )


def _all_finite_reduction(tree: Any) -> jax.Array:

    finite = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(tree):
        value = jnp.asarray(leaf)
        if jnp.issubdtype(value.dtype, jnp.inexact):
            finite = jnp.logical_and(finite, jnp.all(jnp.isfinite(value)))
    return finite


def _grouped_finite_reduction(tree: Any) -> jax.Array:

    groups = {}
    for leaf in jax.tree.leaves(tree):
        value = jnp.asarray(leaf)
        if jnp.issubdtype(value.dtype, jnp.inexact):
            key = (value.shape, value.dtype)
            current = jnp.isfinite(value)
            groups[key] = current if key not in groups else groups[key] & current
    finite = jnp.asarray(True, dtype=jnp.bool_)
    for mask in groups.values():
        finite = finite & jnp.all(mask)
    return finite


_compiled_unique_finite_reduction = process_lifetime_jit(_grouped_finite_reduction)


def _compiled_all_finite_reduction(tree: Any) -> jax.Array:
    seen = set()
    unique = []
    for leaf in jax.tree.leaves(tree):
        identity = id(leaf)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(leaf)
    return _compiled_unique_finite_reduction(tuple(unique))


def _all_finite(tree: Any) -> bool:

    return bool(jax.device_get(_compiled_all_finite_reduction(tree)))


def _methane_finite_reduction(
    methane: Any,
    column_mask: Any,
    patch_mask: Any,
) -> jax.Array:

    column = jnp.asarray(column_mask, dtype=jnp.bool_)
    patch = jnp.asarray(patch_mask, dtype=jnp.bool_)
    finite = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(methane):
        value = jnp.asarray(leaf)
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            continue
        if value.ndim >= 2 and value.shape[:2] == (column.shape[0], 25):
            value = value[:, :20]
        if value.ndim and value.shape[0] == column.shape[0]:
            mask = column.reshape((column.shape[0],) + (1,) * (value.ndim - 1))
            check = jnp.all(~mask | jnp.isfinite(value))
        elif value.ndim and value.shape[0] == patch.shape[0]:
            mask = patch.reshape((patch.shape[0],) + (1,) * (value.ndim - 1))
            check = jnp.all(~mask | jnp.isfinite(value))
        else:
            check = jnp.all(jnp.isfinite(value))
        finite = jnp.logical_and(finite, check)
    return finite


_compiled_methane_finite_reduction = process_lifetime_jit(
    _methane_finite_reduction
)


def _methane_all_finite(result: Any, column_mask: Any, patch_mask: Any) -> bool:

    return bool(
        jax.device_get(
            _compiled_methane_finite_reduction(
                result.methane,
                column_mask,
                patch_mask,
            )
        )
    )


def _drainage_finite_reduction(result: Any) -> jax.Array:

    layer_count = jnp.asarray(result.inputs.snow_layer_count_column)
    if layer_count.ndim != 1:
        return jnp.asarray(False, dtype=jnp.bool_)
    level = jnp.arange(NUMBER_OF_SNOW_LEVELS, dtype=jnp.int32)[None, :]
    active_snow = (layer_count[:, None] < 0) & (
        level >= layer_count[:, None] + NUMBER_OF_SNOW_LEVELS
    )
    entry = result.entry_state
    output = result.drainage.state
    sanitized_entry = entry
    sanitized_output = output
    finite = jnp.asarray(True, dtype=jnp.bool_)
    for name in (
        "soil_snow_liquid_water_column_level",
        "soil_snow_ice_water_column_level",
    ):
        before = jnp.asarray(getattr(entry, name))
        after = jnp.asarray(getattr(output, name))
        if before.ndim != 2 or after.shape != before.shape:
            return jnp.asarray(False, dtype=jnp.bool_)
        applicable = jnp.concatenate(
            (
                active_snow,
                jnp.ones(
                    (before.shape[0], before.shape[1] - NUMBER_OF_SNOW_LEVELS),
                    dtype=jnp.bool_,
                ),
            ),
            axis=1,
        )
        for check in (
            jnp.all(~jnp.isinf(before)),
            jnp.all(~jnp.isinf(after)),
            jnp.all(jnp.where(applicable, jnp.isfinite(before), True)),
            jnp.all(jnp.where(applicable, jnp.isfinite(after), True)),
        ):
            finite = jnp.logical_and(finite, check)
        inactive_before = before[:, :NUMBER_OF_SNOW_LEVELS]
        inactive_after = after[:, :NUMBER_OF_SNOW_LEVELS]
        preserved = (inactive_before == inactive_after) | (
            jnp.isnan(inactive_before) & jnp.isnan(inactive_after)
        )
        finite = jnp.logical_and(
            finite,
            jnp.all(jnp.where(~active_snow, preserved, True)),
        )
        sanitized_entry = sanitized_entry._replace(**{name: jnp.where(applicable, before, 0.0)})
        sanitized_output = sanitized_output._replace(**{name: jnp.where(applicable, after, 0.0)})
    sanitized = result._replace(
        entry_state=sanitized_entry,
        drainage=result.drainage._replace(state=sanitized_output),
    )
    return jnp.logical_and(finite, _all_finite_reduction(sanitized))


_compiled_drainage_finite_reduction = process_lifetime_jit(
    _drainage_finite_reduction
)


def _drainage_all_finite(result: Any) -> bool:

    return bool(jax.device_get(_compiled_drainage_finite_reduction(result)))


def _fatal_count_reduction(diagnostics: Any) -> jax.Array:
    values = [
        jnp.asarray(getattr(diagnostics, name), dtype=jnp.int32)
        for name in diagnostics._fields
        if name.endswith("_fatal")
    ]
    return jnp.sum(jnp.stack(values)) if values else jnp.asarray(0)


_compiled_fatal_count_reduction = process_lifetime_jit(_fatal_count_reduction)


def _fatal_count(diagnostics: Any) -> int:
    return int(jax.device_get(_compiled_fatal_count_reduction(diagnostics)))


def _timestep_science_checks(
    balance_diagnostics, cn_diagnostics, drainage, methane,
    column_mask, patch_mask, late_state,
):

    late_finite = _compiled_all_finite_reduction(late_state)
    for result in late_state:
        if isinstance(result, SurfaceAlbedoRuntimeResult):
            late_finite = late_finite & result.all_finite
    values = (
        _compiled_fatal_count_reduction(balance_diagnostics),
        _compiled_fatal_count_reduction(cn_diagnostics),
        jnp.count_nonzero(
            methane.methane.diagnostics.finalize.methane_conservation_violation_column
        ),
        _compiled_drainage_finite_reduction(drainage),
        _compiled_methane_finite_reduction(methane.methane, column_mask, patch_mask),
        late_finite,
        cn_diagnostics.skipped,
    )
    if any(isinstance(value, jax.core.Tracer) for value in values):
        return values
    fatal, cn_fatal, violations, drainage_ok, methane_ok, late_ok, skipped = jax.device_get(values)
    return (
        int(fatal), int(cn_fatal), int(violations), bool(drainage_ok),
        bool(methane_ok), bool(late_ok), bool(skipped),
    )


def execute_stage400_to_570(
    configuration: Any,
    case: Any,
    boundary: Any,
    *,
    stage_completed: Callable[[str], None],
    prior_state: PriorLandTimestepState | None = None,
    steps_since_restart: int = 1,
    capture_history: bool = False,
    river_volume_gridcell: jax.Array | None = None,
    temporal_inputs: LandTimestepTemporalInputs | None = None,
) -> LandTimestepResult:

    if temporal_inputs is None:
        if (isinstance(steps_since_restart, bool)
                or not isinstance(steps_since_restart, int) or steps_since_restart < 1):
            raise ValueError("steps_since_restart must be a positive integer")
    else:
        if type(temporal_inputs.retain_previous_active_layer_history) is not bool:
            raise TypeError("prepared active-layer retention must be a static Boolean")
        step_aval=jax.typeof(steps_since_restart)
        if step_aval.shape or step_aval.dtype.kind not in 'iu':
            raise ValueError("prepared steps_since_restart must be a scalar integer")
        if not isinstance(steps_since_restart,jax.core.Tracer) and int(steps_since_restart)<1:
            raise ValueError("steps_since_restart must be positive")
    if isinstance(boundary, LandTimestepBoundaryOwner):
        boundary = boundary.take()
    has_prior_state = prior_state is not None

    topology = boundary.topology
    forcing = boundary.forcing
    clock = boundary.clock
    physical = boundary.physical
    post_temperature = boundary.post_temperature
    hydrology = boundary.hydrology
    post_hydrology = boundary.post_hydrology
    lake = boundary.lake_hydrology

    tail_physical = boundary.tail_physical_carry
    lnd2atm_physical = boundary.land_to_atmosphere_physical_carry
    lnd2atm_post_temperature = land_to_atmosphere_post_temperature_carry(post_temperature)
    balance_radiation = boundary.balance_radiation
    surface_post_lake = surface_albedo_post_lake_carry(boundary.post_lake_hydrology)
    post_lake_state = post_lake_hydrology_state(boundary.post_lake_hydrology)
    post_temperature_map = boundary.post_temperature_map
    wind_forcing_height_patch = boundary.wind_forcing_height_patch
    drainage_native = boundary.drainage_native
    physical_state = boundary.physical_state
    soil_temperature_state = boundary.soil_temperature_state
    del boundary
    flags = model_time_flags(clock)
    if temporal_inputs is None:
        call_date = (
            int(jax.device_get(clock.year)) * 10_000
            + int(jax.device_get(clock.month)) * 100
            + int(jax.device_get(clock.day))
        )
        call_time_of_day = int(jax.device_get(clock.seconds_into_day))
    active_layer_native = load_active_layer_native_data(
        case,
        configuration.case.restart_file,
    )
    prior_active_layer = None if prior_state is None else prior_state.take("active_layer")
    active_layer_state = (
        active_layer_native.state if prior_active_layer is None else prior_active_layer
    )
    active_layer = (
        _compiled_active_layer
        if configuration.execution.jit and jax.default_backend() == "gpu"
        else diagnose_active_layer
    )(
        assemble_active_layer_inputs(
            active_layer_native,
            soil_temperature_column_level=(
                lake.hydrology.state.temperature.soil_snow_temperature_column_level[
                    :, NUMBER_OF_SNOW_LEVELS:
                ]
            ),
            month=clock.month,
            day_of_month=clock.day,
            seconds_of_day=clock.seconds_into_day,
        ),
        active_layer_state,
        active_layer_native.parameters,
    )
    if temporal_inputs is None:
        active_layer = _retain_active_layer_history(
            active_layer, active_layer_state, call_date, call_time_of_day,
            configuration.time.time_step_seconds,
        )
    elif temporal_inputs.retain_previous_active_layer_history:
        active_layer = _retain_previous_active_layer_history(active_layer, active_layer_state)
    del prior_active_layer, active_layer_state
    stage_completed("stage410_active_layer")

    children = (load_ecosystem_dynamics_native_children(
        case,
        restart_path=configuration.case.restart_file,
        call_date=call_date,
        call_time_of_day=call_time_of_day,
    ) if temporal_inputs is None else temporal_inputs.children)
    children = children._replace(nitrif_denitrif=children.nitrif_denitrif._replace(
        persistent=nitrification_feedback_from_methane(
            children.nitrif_denitrif.persistent,
            None if prior_state is None else prior_state.methane,
        ),
    ))
    parameter_data_path = children.parameter_data_path
    tail_native = load_end_of_timestep_native_data(
        case,
        restart_path=configuration.case.restart_file,
        parameter_data_path=parameter_data_path,
    )
    prior_ecosystem_pre = (
        None if prior_state is None else prior_state.take("ecosystem_pre_drainage")
    )
    prior_ecosystem_post = (
        None if prior_state is None else prior_state.take("ecosystem_post_drainage")
    )
    initial_ecosystem = (
        load_ecosystem_dynamics_initial_state_native_data(
            case,
            native_children=children,
        )
        if not has_prior_state
        else None
    )
    initial_ecosystem_state = (
        initial_ecosystem.state
        if initial_ecosystem is not None
        else next_ecosystem_state_from_parts(
            prior_ecosystem_pre,
            prior_ecosystem_post,
        )
    )
    prior_post_drainage_persistent = (
        None
        if prior_ecosystem_post is None
        else post_drainage_persistent_state(prior_ecosystem_post)
    )
    prior_end_of_timestep = None if prior_state is None else prior_state.take("end_of_timestep")
    accumulator_state = (
        tail_native.accumulators.state
        if prior_end_of_timestep is None
        else prior_end_of_timestep.accumulators
    )
    if has_prior_state:
        initial_ecosystem_state = synchronize_crop_growing_degree_days(
            initial_ecosystem_state,
            accumulator_state,
        )
    del prior_end_of_timestep
    ecosystem_state = apply_early_ecosystem_vertical_profile(
        initial_ecosystem_state,
        children.vertical_profile,
        active_layer.state.prior_year_maximum_thaw_depth_index_column,
        jit=configuration.execution.jit,
    )
    cn_native = load_carbon_nitrogen_balance_initialization_native_data(
        case,
        restart_path=configuration.case.restart_file,
        parameter_data_path=parameter_data_path,
    )
    cn_initialization = (
        initialize_carbon_nitrogen_balance(
            cn_native.inputs,
            cn_native.state,
            cn_native.parameters,
        )
        if not has_prior_state
        else None
    )
    if cn_initialization is None:
        cn_summaries = prior_ecosystem_post.state_summaries
        cn_carry = carbon_nitrogen_balance_carry_from_summaries(
            cn_native.inputs,
            cn_summaries,
        )
    else:
        cn_summaries = cn_initialization.state.summaries
        cn_carry = carbon_nitrogen_balance_carry(
            cn_native.inputs,
            cn_initialization,
        )
    carbon_root_fraction_patch_level = children.vertical_profile.carbon_root_fraction_patch_level
    external_forcing_state = interpolate_bgc_external_forcing(
        children.external_forcing.inputs,
        children.external_forcing.parameters,
    ).state
    nitrogen_deposition = external_forcing_state.nitrogen_deposition_gridcell
    fast_boundary = (
        _compiled_fast_boundary
        if configuration.execution.jit
        else assemble_ecosystem_fast_boundary
    )(
        topology,
        forcing,
        nitrogen_deposition,
        physical,
        post_temperature_map,
        hydrology,
        lake,
        accumulator_state,
        active_layer,
        decomposition_feedback_from_methane(
            children.decomposition.persistent,
            None if prior_state is None else prior_state.methane,
        ),
        cn_summaries,
        ecosystem_state,
        children.crop.inputs.fertilizer_nitrogen_patch,
        carbon_root_fraction_patch_level,
        children.vertical_profile.carbon_root_profile_beta_patch,
    )
    ecosystem_inputs = assemble_ecosystem_dynamics_complete_inputs(
        children,
        fast_boundary,
        external_forcing_state=external_forcing_state,
        state=ecosystem_state,
        state_summaries=cn_summaries,
        peak_leaf_area_index_patch=(
            children.crop.inputs.peak_leaf_area_index_patch
            if prior_ecosystem_post is None
            else prior_ecosystem_post.vegetation_structure.peak_leaf_area_index_patch
        ),
    )
    physical = _compact_late_physical_sources(physical)
    ecosystem_inputs, ecosystem_state = co_locate_array_leaves(
        (ecosystem_inputs, ecosystem_state),
        jax.local_devices()[0],
    )
    stage_barrier((ecosystem_inputs, ecosystem_state))
    del (
        active_layer_native,
        children,
        cn_native,
        cn_initialization,
        cn_summaries,
        nitrogen_deposition,
        fast_boundary,
        initial_ecosystem,
        initial_ecosystem_state,
        prior_ecosystem_post,
        prior_ecosystem_pre,
    )
    release_completed_stage_memory()
    ecosystem_executor = EcosystemDynamicsDriverExecutor(
        jit=configuration.execution.jit,
        stage_completed=stage_completed,
        retain_diagnostics=False,
        collect_between_stages=False,
        preserve_flux_fields=True,
    )
    prepared_ecosystem = ecosystem_executor.prepare(
        ecosystem_inputs,
        ecosystem_state,
    )
    stage_barrier((prepared_ecosystem.inputs, prepared_ecosystem.state))
    del ecosystem_inputs, ecosystem_state
    release_completed_stage_memory()
    ecosystem = ecosystem_executor.execute_prepared(prepared_ecosystem)
    ecosystem_carry = ecosystem.post_drainage_carry
    ecosystem_state_next = ecosystem.ecosystem.state
    early_history = project_early_history_fields(ecosystem_state_next) if capture_history else None
    stage_barrier((ecosystem_carry, early_history))
    del (
        ecosystem,
        ecosystem_executor,
        prepared_ecosystem,
    )
    release_completed_stage_memory()

    dry_configuration = load_dry_deposition_configuration(case)
    dry_deposition = DryDepositionExecutor(
        dry_configuration,
        jit=configuration.execution.jit,
        stage_completed=stage_completed,
    )(
        jnp.empty(
            (topology.patch.gridcell.shape[0], dry_configuration.number_of_species),
            dtype=jnp.float64,
        )
    )

    prior_drainage = None if prior_state is None else prior_state.take("drainage")
    drainage_inputs = (
        drainage_native.inputs
        if prior_drainage is None
        else drainage_native.inputs._replace(
            beginning_water_mass_column=(prior_drainage.ending_water_mass_column),
        )
    )
    drainage_state = drainage_native.state if prior_drainage is None else prior_drainage
    drainage = HydrologyDrainageDriverExecutor(
        jit=configuration.execution.jit,
        stage_completed=stage_completed,
    )(
        physical,
        post_temperature,
        hydrology,
        post_hydrology,
        lake,
        drainage_inputs,
        drainage_state,
        drainage_native.parameters,
    )
    del prior_drainage
    bgc_native = load_bgc_timestep_initialization_native_data(
        case,
        restart_path=configuration.case.restart_file,
    )
    initialize_bgc = (
        _compiled_bgc_initialization
        if configuration.execution.jit
        else initialize_bgc_timestep
    )
    dynamic_landcover_carbon_flux = initialize_bgc(
        assemble_bgc_timestep_initialization_state(bgc_native)
    ).state.carbon_flux
    water = drainage.drainage.state
    post_drainage_native = load_ecosystem_dynamics_post_drainage_native_data(
        case,
        restart_path=configuration.case.restart_file,
        parameter_data_path=parameter_data_path,
    )
    post_drainage_call = assemble_ecosystem_dynamics_post_drainage_call_from_carry(
        post_drainage_native,
        ecosystem_carry,
        water.soil_snow_liquid_water_column_level[:, NUMBER_OF_SNOW_LEVELS:],
        water.subsurface_drainage_flux_column,
        water.surface_runoff_flux_column,
        dynamic_landcover_carbon_flux,
        lake.hydrology.state.water.snow_depth_column,
        wind_forcing_height_patch,
        do_albedo=True,
        is_beginning_current_year=flags.beginning_current_year,
        is_first_step=False,
        previous_state=prior_post_drainage_persistent,
    )
    post_drainage_call = co_locate_array_leaves(
        post_drainage_call,
        jax.devices()[0],
    )
    stage_barrier(post_drainage_call)
    del bgc_native, post_drainage_native
    release_completed_stage_memory()
    post_drainage = EcosystemDynamicsPostDrainageExecutor(
        jit=configuration.execution.jit,
        preserve_flux_fields=True,
        stage_completed=stage_completed,
    )(*post_drainage_call)

    prior_balance = None if prior_state is None else prior_state.take("balance")
    balance = BalanceCheckExecutor(
        jit=configuration.execution.jit,
        stage_completed=stage_completed,
    )(
        forcing,
        physical,
        post_temperature,
        hydrology,
        post_hydrology,
        lake,
        drainage,
        balance_radiation,
        (initial_balance_check_state(topology) if prior_balance is None else prior_balance),
        BalanceCheckParameters.from_case(case),
        steps_since_restart=steps_since_restart,
    )
    del prior_balance, prior_post_drainage_persistent
    cn_balance = CarbonNitrogenBalanceExecutor(
        jit=configuration.execution.jit,
        stage_completed=stage_completed,
    )(
        cn_carry,
        post_drainage,
        CarbonNitrogenBalanceCheckParameters.from_case(case),
        steps_since_restart=steps_since_restart,
    )
    methane_native = load_methane_driver_native_data(
        case,
        restart_path=configuration.case.restart_file,
        parameter_data_path=parameter_data_path,
    )
    prior_methane = None if prior_state is None else prior_state.take("methane")
    prior_land_to_atmosphere = (
        None if prior_state is None else prior_state.take("land_to_atmosphere")
    )
    if prior_methane is None:
        methane_initialization = process_lifetime_jit(initialize_methane_balance)(
            methane_native.balance_initialization.inputs,
            methane_native.balance_initialization.state,
            methane_native.balance_initialization.parameters,
        )
        methane_carry = methane_driver_carry(
            methane_initialization,
            carbon_root_fraction_patch_level,
        )
        methane_state = methane_native.state
        prior_total_water_storage = None
    else:
        methane_initialization = None
        methane_carry = methane_driver_carry_from_state(
            prior_methane,
            carbon_root_fraction_patch_level,
        )
        methane_state = prior_methane
        prior_total_water_storage = prior_land_to_atmosphere.total_water_storage_gridcell
    methane = MethaneDriverExecutor(
        jit=configuration.execution.jit,
        device_count=jax.local_device_count(),
        combine_gpu_stages=True,
        stage_completed=stage_completed,
    )(
        methane_native.inputs,
        methane_state,
        methane_native.parameters,
        forcing,
        physical,
        hydrology,
        lake,
        drainage,
        post_drainage,
        ecosystem_carry,
        methane_carry,
        total_water_storage_gridcell=prior_total_water_storage,
    )
    del prior_methane
    release_completed_stage_memory()

    prior_surface_albedo = None if prior_state is None else prior_state.take("surface_albedo")
    surface_native = load_surface_albedo_native_data(
        case,
        restart_path=configuration.case.restart_file,
    )
    surface_call = compose_surface_albedo_call(
        surface_native,
        clock,
        physical,
        lake,
        surface_post_lake,
        post_drainage,
        state=prior_surface_albedo,
        drainage_state=drainage.drainage.state,
    )
    surface_albedo_devices = jax.local_device_count()
    if jax.default_backend() == "cpu":
        surface_albedo_devices = min(surface_albedo_devices, 16)
    surface_albedo = SurfaceAlbedoExecutor(
        jit=configuration.execution.jit,
        device_count=surface_albedo_devices,
        stage_completed=stage_completed,
    )(surface_call)
    del prior_surface_albedo
    prior_urban_albedo = None if prior_state is None else prior_state.take("urban_albedo")
    urban_native = load_urban_albedo_template(
        case,
        restart_path=configuration.case.restart_file,
    )
    urban_call = compose_urban_albedo_call(
        urban_native,
        surface_call,
        surface_albedo.state,
        prior_urban_albedo,
    )
    urban_albedo = UrbanAlbedoExecutor(
        jit=configuration.execution.jit,
        stage_completed=stage_completed,
    )(urban_call)
    del prior_urban_albedo

    lnd2atm_native = load_land_to_atmosphere_native_data(
        case,
        land_restart_path=configuration.case.restart_file,
        coupler_restart_path=configuration.case.coupler_restart_file,
    )
    external = lnd2atm_native.external._replace(dry_deposition_velocity_patch=dry_deposition)
    if river_volume_gridcell is not None:
        if river_volume_gridcell.shape != external.river_volume_gridcell.shape:
            raise ValueError("river coupling input must match the CLM land grid")
        external = external._replace(river_volume_gridcell=river_volume_gridcell)
    lnd2atm_call = compose_land_to_atmosphere_call(
        lnd2atm_native.topology._replace(subgrid=topology),
        (lnd2atm_native.state if prior_land_to_atmosphere is None else prior_land_to_atmosphere),
        lnd2atm_native.parameters,
        lnd2atm_physical,
        balance_radiation,
        lnd2atm_post_temperature,
        lake,
        drainage,
        post_drainage,
        methane.methane,
        urban_albedo,
        external,
    )
    land_to_atmosphere = LandToAtmosphereExecutor(
        jit=configuration.execution.jit,
        stage_completed=stage_completed,
    )(lnd2atm_call)
    del prior_land_to_atmosphere
    if prior_state is not None:
        prior_state.assert_consumed()

    tail_inputs = (
        _compiled_tail_inputs
        if configuration.execution.jit and jax.default_backend() == "gpu"
        else compose_end_of_timestep_inputs
    )(
        topology,
        forcing,
        physical,
        tail_physical,
        lake,
        drainage,
        post_drainage,
        ecosystem_carry,
        tail_native.land_to_glacier.inputs,
        tail_native.accumulators.parameters,
        tail_native.composition_parameters,
    )
    tail_state = EndOfTimestepState(
        accumulators=accumulator_state,
        vegetation=post_drainage.state.vegetation_structure,
    )
    end_of_timestep = EndOfTimestepExecutor(
        number_of_gridcells=tail_native.land_to_glacier.number_of_gridcells,
        maximum_glacier_classes=(tail_native.land_to_glacier.maximum_glacier_classes),
        use_cndv=tail_native.vegetation.use_cndv,
        jit=configuration.execution.jit,
        stage_completed=stage_completed,
    )(
        topology,
        tail_inputs,
        tail_state,
        clock,
        tail_native.accumulators.parameters,
    )

    history_snapshot = None
    if capture_history:
        if early_history is None:
            raise RuntimeError("history capture lost its stage-420 projection")
        history_snapshot = compose_instantaneous_history(
            topology=topology,
            forcing=forcing,
            physical=physical,
            post_temperature=post_temperature,
            lake=lake,
            drainage=drainage,
            post_drainage=post_drainage,
            balance=balance,
            balance_radiation=balance_radiation,
            land_to_atmosphere_call=lnd2atm_call,
            land_to_atmosphere=land_to_atmosphere,
            early=early_history,
        )
    del early_history

    if history_snapshot is not None and jax.default_backend() == "gpu":
        history_arrays = {
            id(value): value for value in jax.tree.leaves(history_snapshot)
            if isinstance(value, jax.Array)
        }
        for value in history_arrays.values():
            value.copy_to_host_async()
        del history_arrays

    science_validation_started = perf_counter()
    (
        balance_fatal_count, cn_fatal_count, methane_conservation_violations,
        drainage_finite, methane_finite, late_state_finite, cn_check_skipped,
    ) = _timestep_science_checks(
        balance.balance.diagnostics,
        cn_balance.balance.diagnostics,
        drainage,
        methane,
        methane_native.inputs.methane_column,
        methane_native.inputs.methane_patch,
        (
            post_drainage,
            surface_albedo,
            urban_albedo,
            land_to_atmosphere,
            end_of_timestep,
        )
    )
    all_finite = (
        drainage_finite
        & methane_finite
        & late_state_finite
    )
    science_clean = (
        all_finite
        & (balance_fatal_count == 0)
        & (cn_fatal_count == 0)
        & (methane_conservation_violations == 0)
    )
    if not isinstance(science_clean, jax.core.Tracer):
        science_clean = bool(science_clean)
    science_validation_wall_seconds = perf_counter() - science_validation_started
    diagnostics = {
        "all_finite": all_finite,
        "balance_fatal_count": balance_fatal_count,
        "carbon_nitrogen_fatal_count": cn_fatal_count,
        "carbon_nitrogen_check_skipped": cn_check_skipped,
        "drainage_all_finite": drainage_finite,
        "dry_deposition_species": dry_configuration.number_of_species,
        "methane_all_finite": methane_finite,
        "methane_conservation_violation_count": methane_conservation_violations,
        "science_validation_wall_seconds": science_validation_wall_seconds,
        "science_clean": science_clean,
    }
    carry = compose_land_timestep_carry(
        physical=physical_state,
        soil_temperature=soil_temperature_state,
        post_temperature=post_temperature,
        hydrology=hydrology,
        post_hydrology=post_hydrology,
        lake_hydrology=lake,
        post_lake_hydrology=post_lake_state,
        active_layer=active_layer,
        drainage=drainage,
        ecosystem_pre_drainage=ecosystem_state_next,
        ecosystem_post_drainage=post_drainage,
        methane=methane,
        surface_albedo=surface_albedo,
        urban_albedo=urban_albedo,
        balance=balance,
        carbon_nitrogen_balance=cn_balance,
        land_to_atmosphere=land_to_atmosphere,
        end_of_timestep=end_of_timestep,
    )
    return LandTimestepResult(
        carry=carry,
        land_to_atmosphere=land_to_atmosphere,
        end_of_timestep=end_of_timestep,
        post_drainage=post_drainage,
        methane=methane,
        surface_albedo=surface_albedo,
        urban_albedo=urban_albedo,
        history=history_snapshot,
        diagnostics=diagnostics,
    )


__all__ = [
    "LandTimestepResult",
    "PriorLandTimestepState",
    "execute_stage400_to_570",
    "project_prior_land_timestep_state",
]
