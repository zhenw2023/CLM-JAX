
from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from clm_jax.calendar import add_noleap_seconds, integer_date_and_time
from clm_jax.compilation import jit_unique, process_lifetime_jit
from clm_jax.execution_observer import (
    defer_stage_barriers,
    stage_barrier,
    stage_timing_synchronized,
)
from clm_jax.io.native_templates import native_template
from clm_jax.model_config import ModelConfiguration, execution_shard_count
from clm_jax.runtime import (
    co_locate_array_leaves,
    configure_memory_reclamation,
    configure_runtime,
    current_memory_map_count,
    current_resident_set_size_gib,
    defer_full_collections_to_memory_policy,
    gpu_dispatch_affinity,
    maximum_resident_set_size_gib,
    memory_reclamation_statistics,
    release_completed_stage_memory,
    release_startup_executables,
)


class _Stage400Boundary(NamedTuple):

    topology: Any
    forcing: Any
    clock: Any
    post_temperature_map: Any
    drainage_native: Any
    tail_physical_carry: Any
    land_to_atmosphere_physical_carry: Any
    balance_radiation: Any
    wind_forcing_height_patch: Any
    physical_state: Any
    soil_temperature_state: Any
    physical: Any
    post_temperature: Any
    hydrology: Any
    post_hydrology: Any
    lake_hydrology: Any
    post_lake_hydrology: Any
    completed_stages: list[str]


class _DownstreamNativeTemplates(NamedTuple):

    hydrology: Any
    post_hydrology: Any
    lake_hydrology: Any
    drainage: Any


_DOWNSTREAM_NATIVE_TEMPLATE_CACHE: dict[
    tuple[Path, Path, Path], _DownstreamNativeTemplates
] = {}


@native_template
def _load_initialized_glacier_fractions(case, restart_path, coupler_restart_path):
    from clm_jax.io.glacier_fraction_initialization import (
        load_glacier_fraction_initialization_native_data,
    )
    from clm_jax.physics.glacier_fraction_initialization import initialize_glacier_fractions

    native = load_glacier_fraction_initialization_native_data(
        case, restart_path=restart_path, coupler_restart_path=coupler_restart_path,
    )
    result = process_lifetime_jit(
        initialize_glacier_fractions,
        static_argnames=("glc_do_dynglacier", "run_zero_weight_urban", "all_active"),
    )(
        native.inputs, glc_do_dynglacier=native.glc_do_dynglacier,
        run_zero_weight_urban=native.run_zero_weight_urban, all_active=native.all_active,
    )
    return co_locate_array_leaves(result, jax.local_devices()[0])


@native_template
def _load_physical_restart_templates(case, restart_path, coupler_restart_path):
    from clm_jax.io.glacier_fraction_initialization import (
        apply_glacier_fraction_initialization,
    )
    from clm_jax.io.physical_driver import load_physical_driver_native_data
    from clm_jax.io.post_temperature_driver import (
        load_post_temperature_driver_native_data,
    )
    from clm_jax.io.radiation_driver import assert_same_topology

    native = load_physical_driver_native_data(case, restart_path, coupler_restart_path)
    post_native = load_post_temperature_driver_native_data(case, restart_path)
    assert_same_topology(native.static.topology, post_native.topology)
    glacier = _load_initialized_glacier_fractions(case, restart_path, coupler_restart_path)
    topology = apply_glacier_fraction_initialization(native.static.topology, glacier)
    native = native._replace(static=native.static._replace(
        topology=topology,
        atmospheric=native.static.atmospheric._replace(topology=topology),
    ))
    return co_locate_array_leaves((native, post_native), jax.local_devices()[0])


@jax.core.eval_context()
def _load_downstream_native_templates(
    case,
    land_restart_path: str | Path,
    coupler_restart_path: str | Path,
) -> _DownstreamNativeTemplates:

    cache_key = (
        Path(case.root).expanduser().resolve(),
        Path(land_restart_path).expanduser().resolve(),
        Path(coupler_restart_path).expanduser().resolve(),
    )
    cached = _DOWNSTREAM_NATIVE_TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from clm_jax.io.hydrology_drainage_driver import (
        load_hydrology_drainage_driver_native_data,
    )
    from clm_jax.io.hydrology_driver import load_hydrology_driver_native_data
    from clm_jax.io.lake_hydrology_driver import (
        load_lake_hydrology_driver_native_data,
    )
    from clm_jax.io.physical_driver import load_physical_driver_native_data
    from clm_jax.io.post_hydrology_no_drainage_driver import (
        load_post_hydrology_no_drainage_native_data,
    )
    from clm_jax.io.post_temperature_driver import (
        load_post_temperature_driver_native_data,
    )

    physical = load_physical_driver_native_data(
        case,
        land_restart_path,
        coupler_restart_path,
    )
    post_temperature = load_post_temperature_driver_native_data(
        case,
        land_restart_path,
    )
    hydrology = load_hydrology_driver_native_data(case, physical)
    post_hydrology = load_post_hydrology_no_drainage_native_data(
        case,
        physical,
        post_temperature,
    )
    lake_hydrology = load_lake_hydrology_driver_native_data(
        case,
        physical,
        hydrology,
        post_hydrology,
    )
    drainage = load_hydrology_drainage_driver_native_data(
        case,
        physical,
        hydrology,
        post_hydrology,
        lake_hydrology,
    )
    templates = _DownstreamNativeTemplates(
        hydrology=hydrology,
        post_hydrology=post_hydrology,
        lake_hydrology=lake_hydrology,
        drainage=drainage,
    )
    templates = co_locate_array_leaves(templates, jax.local_devices()[0])
    _DOWNSTREAM_NATIVE_TEMPLATE_CACHE[cache_key] = templates
    return templates


def _compact_post_temperature_physical_sources(physical):

    stage = physical.soil_temperature
    source_inputs = stage.inputs
    source_thermal = source_inputs.thermal_properties
    retained_thermal_fields = {
        "bedrock_layer_count",
        "interface_depth",
        "layer_thickness",
        "node_depth",
    }
    thermal = source_thermal._replace(
        **{name: None for name in source_thermal._fields if name not in retained_thermal_fields}
    )
    inputs = source_inputs._replace(
        **{
            name: None
            for name in source_inputs._fields
            if name not in {"topology", "thermal_properties"}
        },
        thermal_properties=thermal,
    )
    return physical._replace(
        radiation=None,
        soil_temperature=stage._replace(inputs=inputs),
    )


def _next_physical_owners(inputs, carry):

    from clm_jax.physics.next_timestep import (
        next_atmospheric_inputs,
        next_canopy_flux_inputs,
        next_canopy_inputs,
        next_canopy_temperature_inputs,
        next_dust_inputs,
        next_physical_state,
        next_radiation_inputs,
        next_soil_temperature_inputs,
    )

    next_inputs = inputs._replace(
        atmospheric=next_atmospheric_inputs(inputs.atmospheric, carry),
        canopy=next_canopy_inputs(inputs.canopy, carry),
        canopy_flux=next_canopy_flux_inputs(inputs.canopy_flux, carry),
        radiation=next_radiation_inputs(inputs.radiation, carry),
        canopy_temperature=next_canopy_temperature_inputs(
            inputs.canopy_temperature,
            carry,
        ),
        voc_accumulators=carry.end_of_timestep.accumulators,
        dust=next_dust_inputs(inputs.dust, carry),
        soil_temperature=next_soil_temperature_inputs(
            inputs.soil_temperature,
            carry,
        ),
    )
    return next_inputs, next_physical_state(next_inputs, carry)


_compiled_next_physical_owners = jit_unique(_next_physical_owners, forward_inputs=True)
_USE_COMPILED_PHYSICAL_CARRY = True


def _apply_prior_physical_carry(native, carry):
    assemble = (_compiled_next_physical_owners
                if _USE_COMPILED_PHYSICAL_CARRY and jax.default_backend() == "gpu"
                else _next_physical_owners)
    inputs, state = assemble(native.inputs, carry)
    return native._replace(inputs=inputs, state=state)


def _write_json(path: Path, value: dict, *, overwrite: bool) -> None:

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"configured output already exists and output.overwrite=false: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _compare_configured_history(configuration, history_files) -> dict[str, Any] | None:

    if configuration.validation is None:
        return None
    from clm_jax.physics.history_output import H1_HISTORY_VARIABLES, H2_HISTORY_VARIABLES
    from clm_jax.validation.history import compare_history_collection

    candidates = tuple(Path(path) for path in history_files)

    def candidate(tape: str) -> Path:
        matches = tuple(path for path in candidates if f".{tape}." in path.name)
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one candidate {tape} history file; got {matches}"
            )
        return matches[0]

    h1 = compare_history_collection(
        configuration.validation.reference_h1,
        candidate("h1"),
        included_variables=H1_HISTORY_VARIABLES,
    )
    h2 = compare_history_collection(
        configuration.validation.reference_h2,
        candidate("h2"),
        included_variables=H2_HISTORY_VARIABLES,
    )
    report_path = configuration.output.directory / "history_comparison.json"
    report = {
        "all_exact": h1["exact"] and h2["exact"],
        "h1": h1,
        "h2": h2,
    }
    _write_json(report_path, report, overwrite=configuration.output.overwrite)
    return {
        "all_exact": report["all_exact"],
        "h1_compared_variables": h1["compared_variables"],
        "h1_exact_variables": h1["exact_variables"],
        "h2_compared_variables": h2["compared_variables"],
        "h2_exact_variables": h2["exact_variables"],
        "report": str(report_path),
    }


def _normalize_trajectory_carry_layout(carry: Any) -> Any:

    normalized = co_locate_array_leaves(carry, jax.devices()[0])
    jax.block_until_ready(normalized)
    return normalized


def _run_single_physical_boundary(
    configuration,
    case,
    runtime,
    *,
    return_boundary: bool = False,
    forcing_override: Any | None = None,
    aerosol_forcing_override: Any | None = None,
    call_clock_override: Any | None = None,
    prior_carry: Any | None = None,
    allow_configured_multistep: bool = False,
    initial_physical_inputs: Any | None = None,
) -> dict | _Stage400Boundary:

    if configuration.time.step_count != 1 and not allow_configured_multistep:
        raise RuntimeError(
            "the direct physical-boundary helper requires exactly one time step; "
            "multi-step integration must use the trajectory driver"
        )

    from clm_jax.forcing.area import load_land_flux_area_factor
    from clm_jax.hydrology_execution import HydrologyDriverExecutor
    from clm_jax.io.coupler import load_atmospheric_coupler_forcing
    from clm_jax.io.radiation_driver import assert_same_topology
    from clm_jax.io.restart import load_restart_clock
    from clm_jax.io.snow_aging import load_snow_aging_lookup_table
    from clm_jax.lake_hydrology_execution import LakeHydrologyDriverExecutor
    from clm_jax.physical_execution import PhysicalDriverExecutor
    from clm_jax.physics.balance_check_driver import (
        balance_check_radiation_fields,
    )
    from clm_jax.physics.end_of_timestep import end_of_timestep_physical_carry
    from clm_jax.physics.land_to_atmosphere_composition import (
        land_to_atmosphere_physical_carry,
    )
    from clm_jax.physics.snow_age_grain import SnowAgeGrainParameters
    from clm_jax.physics.timestep_carry import (
        physical_driver_state,
        soil_temperature_timestep_state,
    )
    from clm_jax.post_hydrology_no_drainage_execution import (
        PostHydrologyNoDrainageExecutor,
    )
    from clm_jax.post_lake_hydrology_execution import PostLakeHydrologyExecutor
    from clm_jax.post_temperature_execution import PostTemperatureDriverExecutor
    from clm_jax.trajectory import advance_noleap_clock, noleap_clock_from_restart

    restart_clock = load_restart_clock(configuration.case.restart_file)
    configured_date, configured_seconds = integer_date_and_time(configuration.time.start)
    if (
        restart_clock.current_date != configured_date
        or restart_clock.current_time_of_day != configured_seconds
    ):
        raise ValueError(
            "configured time.start does not match the completed CLM restart boundary: "
            f"config={configured_date:08d}/{configured_seconds:05d}, "
            f"restart={restart_clock.current_date:08d}/"
            f"{restart_clock.current_time_of_day:05d}"
        )
    if restart_clock.step_seconds != configuration.time.time_step_seconds:
        raise ValueError(
            "configured time step disagrees with the CLM restart: "
            f"config={configuration.time.time_step_seconds}, "
            f"restart={restart_clock.step_seconds}"
        )

    completed_stages: list[str] = []
    boundary_started = perf_counter()
    previous_stage_completed = boundary_started
    log_stage_progress = prior_carry is None or stage_timing_synchronized()

    def stage_completed(name: str) -> None:
        nonlocal previous_stage_completed
        completed_stages.append(name)
        now = perf_counter()
        peak_rss_gib = maximum_resident_set_size_gib()
        if log_stage_progress:
            print(
                json.dumps(
                    {
                        "boundary_wall_seconds": now - boundary_started,
                        "current_rss_gib": current_resident_set_size_gib(),
                        "memory_map_count": (
                            current_memory_map_count() if stage_timing_synchronized() else None
                        ),
                        "maximum_rss_gib": peak_rss_gib,
                        "stage": name,
                        (
                            "stage_wall_seconds" if stage_timing_synchronized()
                            else "stage_dispatch_seconds"
                        ): now - previous_stage_completed,
                        "status": "complete" if stage_timing_synchronized() else "dispatched",
                    }
                ),
                flush=True,
            )
        previous_stage_completed = now
        if peak_rss_gib > configuration.execution.maximum_rss_gib:
            raise MemoryError(
                f"{name} exceeded the configured peak RSS limit: "
                f"{peak_rss_gib:.3f} > "
                f"{configuration.execution.maximum_rss_gib:.3f} GiB"
            )

    native, post_native = _load_physical_restart_templates(
        case,
        configuration.case.restart_file,
        configuration.case.coupler_restart_file,
    )
    if initial_physical_inputs is not None:
        if prior_carry is not None:
            raise ValueError("initial physical controls apply only at the restart boundary")
        native = native._replace(inputs=initial_physical_inputs)
    if prior_carry is not None:
        post_native = post_native._replace(state=prior_carry.post_temperature)
    topology = native.static.topology
    if prior_carry is not None:
        native = _apply_prior_physical_carry(native, prior_carry)
    resident_trajectory = (
        allow_configured_multistep and runtime.backend == "gpu"
        and len(runtime.devices) == 1
    )
    if not resident_trajectory:
        native = co_locate_array_leaves(native, jax.devices()[0])
    stage_barrier(native)
    release_completed_stage_memory()
    stage_completed("stage010_glacier_fraction_initialization")
    forcing = (
        load_atmospheric_coupler_forcing(
            configuration.case.coupler_restart_file,
            topology,
            land_flux_area_factor=load_land_flux_area_factor(case, topology),
        )
        if forcing_override is None
        else forcing_override
    )
    call_clock = (
        advance_noleap_clock(noleap_clock_from_restart(restart_clock))
        if call_clock_override is None
        else call_clock_override
    )

    physical = PhysicalDriverExecutor(
        device_count=len(runtime.devices),
        soil_temperature_device_count=min(len(runtime.devices), 32),
        jit=configuration.execution.jit,
        stage_completed=stage_completed,
        retain_diagnostics=False,
    )(
        native.static,
        native.state,
        native.inputs,
        native.parameters,
        forcing,
        call_clock,
    )
    stage_barrier(getattr(physical, "land_to_atmosphere", physical))
    physical_state = physical_driver_state(physical)
    soil_temperature_state = soil_temperature_timestep_state(physical)
    del native
    release_completed_stage_memory()

    tail_physical_carry = end_of_timestep_physical_carry(physical)
    lnd2atm_physical_carry = land_to_atmosphere_physical_carry(physical)
    balance_radiation = balance_check_radiation_fields(physical)
    wind_forcing_height_patch = physical.canopy_temperature.state.patch.forcing_height_wind_patch
    physical = physical._replace(
        canopy_temperature=None,
        bare_ground=physical.bare_ground._replace(
            inputs=physical.bare_ground.inputs._replace(**{
                name: None for name in physical.bare_ground.inputs._fields
                if name != "vegetation_exposed_patch"
            }),
            state=physical.bare_ground.state._replace(**{
                name: None for name in physical.bare_ground.state._fields
                if name != "ground_methane_conductance_patch"
            }),
        ),
        irrigation_demand=None,
        dust=None,
        voc=None,
    )
    release_completed_stage_memory()

    if not resident_trajectory:
        post_native = co_locate_array_leaves(post_native, jax.devices()[0])
    stage_barrier(post_native)
    post_temperature_map = post_native.inputs.native_map

    def release_post_temperature_sources() -> None:
        nonlocal physical, post_native
        physical = _compact_post_temperature_physical_sources(physical)
        post_native = None

    post_temperature = PostTemperatureDriverExecutor(
        jit=configuration.execution.jit,
        stage_completed=stage_completed,
        sources_materialized=release_post_temperature_sources,
    )(
        physical,
        post_native.inputs,
        post_native.state,
        post_native.parameters,
    )

    release_completed_stage_memory()

    downstream_templates = _load_downstream_native_templates(
        case,
        configuration.case.restart_file,
        configuration.case.coupler_restart_file,
    )
    hydrology_native = downstream_templates.hydrology
    post_hydrology_native = downstream_templates.post_hydrology
    lake_hydrology_native = downstream_templates.lake_hydrology
    drainage_native = downstream_templates.drainage
    if aerosol_forcing_override is not None:
        hydrology_native = hydrology_native._replace(inputs=hydrology_native.inputs._replace(
            atmospheric_aerosol_flux_gridcell_species=aerosol_forcing_override))
    if prior_carry is not None:
        from clm_jax.physics.next_timestep import next_hydrology_state

        hydrology_native = hydrology_native._replace(
            state=next_hydrology_state(prior_carry)
        )
        post_hydrology_native = post_hydrology_native._replace(
            state=prior_carry.post_hydrology,
        )
        lake_hydrology_native = lake_hydrology_native._replace(
            inputs=lake_hydrology_native.inputs._replace(
                template=lake_hydrology_native.inputs.template._replace(
                    begin_water_balance_column=(prior_carry.drainage.ending_water_mass_column),
                ),
            ),
            state=prior_carry.lake_hydrology,
        )
        drainage_native = drainage_native._replace(
            inputs=drainage_native.inputs._replace(
                beginning_water_mass_column=(
                    prior_carry.drainage.ending_water_mass_column
                ),
            ),
        )
    snow_aging_lookup = load_snow_aging_lookup_table(case)
    snow_age_parameters = SnowAgeGrainParameters.from_case(case)
    assert_same_topology(
        hydrology_native.topology,
        post_hydrology_native.topology,
    )
    assert_same_topology(
        hydrology_native.topology,
        lake_hydrology_native.topology,
    )
    if not resident_trajectory:
        (
            hydrology_native,
            post_hydrology_native,
            lake_hydrology_native,
            drainage_native,
        ) = co_locate_array_leaves(
            (
                hydrology_native,
                post_hydrology_native,
                lake_hydrology_native,
                drainage_native,
            ),
            jax.devices()[0],
        )
    stage_barrier(
        (
            hydrology_native,
            post_hydrology_native,
            lake_hydrology_native,
            drainage_native,
        )
    )
    del downstream_templates
    release_completed_stage_memory()

    hydrology = HydrologyDriverExecutor(
        jit=configuration.execution.jit,
        device_count=len(runtime.devices),
        stage_completed=stage_completed,
    )(
        physical,
        post_temperature,
        forcing,
        hydrology_native.inputs,
        hydrology_native.state,
        hydrology_native.parameters,
        filter_topology=topology,
    )
    del hydrology_native

    execution_device = jax.devices()[0]
    (
        topology,
        forcing,
        physical,
        post_temperature,
        tail_physical_carry,
        lnd2atm_physical_carry,
        balance_radiation,
        wind_forcing_height_patch,
        physical_state,
        soil_temperature_state,
    ) = co_locate_array_leaves(
        (
            topology,
            forcing,
            physical,
            post_temperature,
            tail_physical_carry,
            lnd2atm_physical_carry,
            balance_radiation,
            wind_forcing_height_patch,
            physical_state,
            soil_temperature_state,
        ),
        execution_device,
    )
    stage_barrier(
        (
            topology,
            forcing,
            physical,
            post_temperature,
            tail_physical_carry,
            lnd2atm_physical_carry,
            balance_radiation,
            wind_forcing_height_patch,
            physical_state,
            soil_temperature_state,
        )
    )
    release_completed_stage_memory()
    post_hydrology = PostHydrologyNoDrainageExecutor(
        jit=configuration.execution.jit,
        stage_completed=stage_completed,
    )(
        topology,
        post_temperature,
        hydrology,
        post_hydrology_native.inputs,
        post_hydrology_native.state,
        post_hydrology_native.parameters,
    )
    del post_hydrology_native
    release_completed_stage_memory()
    lake_column_index = lake_hydrology_native.inputs.lake_column_index
    lake_hydrology = LakeHydrologyDriverExecutor(
        jit=configuration.execution.jit,
        stage_completed=stage_completed,
    )(
        physical,
        post_temperature,
        hydrology,
        post_hydrology,
        forcing,
        lake_hydrology_native.inputs,
        lake_hydrology_native.state,
        lake_hydrology_native.parameters,
    )
    post_lake_hydrology = PostLakeHydrologyExecutor(
        jit=configuration.execution.jit,
        stage_completed=stage_completed,
    )(
        topology,
        physical,
        post_hydrology,
        lake_hydrology,
        lake_column_index,
        snow_aging_lookup,
        snow_age_parameters,
    )
    del lake_hydrology_native, lake_column_index, snow_aging_lookup
    release_completed_stage_memory()
    boundary = _Stage400Boundary(
        topology=topology,
        forcing=forcing,
        clock=call_clock,
        post_temperature_map=post_temperature_map,
        drainage_native=drainage_native,
        tail_physical_carry=tail_physical_carry,
        land_to_atmosphere_physical_carry=lnd2atm_physical_carry,
        balance_radiation=balance_radiation,
        wind_forcing_height_patch=wind_forcing_height_patch,
        physical_state=physical_state,
        soil_temperature_state=soil_temperature_state,
        physical=physical,
        post_temperature=post_temperature,
        hydrology=hydrology,
        post_hydrology=post_hydrology,
        lake_hydrology=lake_hydrology,
        post_lake_hydrology=post_lake_hydrology,
        completed_stages=completed_stages,
    )
    if return_boundary:
        return boundary
    converted_layers = int(
        jax.device_get(jnp.count_nonzero(post_temperature.glacier_ice_melt.converted_layer))
    )
    patch_to_column_finite = bool(
        jax.device_get(
            jnp.all(
                jnp.stack(
                    [
                        jnp.all(jnp.isfinite(field))
                        for field in post_temperature.patch_to_column.state
                    ]
                )
            )
        )
    )
    hydrology_state = hydrology.hydrology.state
    hydrology_water_finite = bool(
        jax.device_get(
            jnp.all(
                jnp.stack(
                    [
                        jnp.all(
                            jnp.isfinite(
                                hydrology_state.snow_structure.soil_snow_liquid_column_level
                            )
                        ),
                        jnp.all(
                            jnp.isfinite(hydrology_state.snow_structure.soil_snow_ice_column_level)
                        ),
                        jnp.all(
                            jnp.isfinite(
                                hydrology_state.soil_water_movement.volumetric_water_column_level
                            )
                        ),
                    ]
                )
            )
        )
    )
    maximum_hydrology_substeps = int(
        jax.device_get(jnp.max(hydrology_state.soil_water_movement.adaptive_substep_count_column))
    )
    glacier_net_ice_all_finite = bool(
        jax.device_get(
            jnp.all(
                jnp.isfinite(post_hydrology.glacier_surface_mass_balance.state.net_ice_flux_column)
            )
        )
    )
    nonlake_snow_columns = int(
        jax.device_get(jnp.count_nonzero(post_hydrology.aerosol_mass.diagnostics.snow_column))
    )
    lake_columns = int(
        jax.device_get(jnp.count_nonzero(lake_hydrology.hydrology.diagnostics.lake_column))
    )
    lake_water_all_finite = bool(
        jax.device_get(
            jnp.all(
                jnp.stack(
                    [
                        jnp.all(
                            jnp.isfinite(
                                lake_hydrology.hydrology.state.water.soil_snow_liquid_column_level
                            )
                        ),
                        jnp.all(
                            jnp.isfinite(
                                lake_hydrology.hydrology.state.water.soil_snow_ice_column_level
                            )
                        ),
                        jnp.all(
                            jnp.isfinite(
                                lake_hydrology.hydrology.state.temperature.lake_temperature_column_level
                            )
                        ),
                    ]
                )
            )
        )
    )
    snow_age_all_finite = bool(
        jax.device_get(
            jnp.all(
                jnp.stack(
                    [
                        jnp.all(jnp.isfinite(field))
                        for field in post_lake_hydrology.nonlake_snow_age.state
                    ]
                )
            )
        )
    )
    summary = {
        "backend": runtime.backend,
        "boundary": "stage400_nonlake_snow_age",
        "call_time": configuration.time.stop.isoformat(),
        "case": str(case.root),
        "columns": int(topology.column.gridcell.shape[0]),
        "complete_land_time_step": False,
        "completed_stages": completed_stages,
        "coupler_restart_file": str(configuration.case.coupler_restart_file),
        "devices": len(runtime.devices),
        "gridcells": int(topology.grid.longitude_degrees.shape[0]),
        "glacier_converted_layers": converted_layers,
        "glacier_net_ice_all_finite": glacier_net_ice_all_finite,
        "hydrology_water_all_finite": hydrology_water_finite,
        "land_restart_file": str(configuration.case.restart_file),
        "landunits": int(topology.landunit.gridcell.shape[0]),
        "lake_columns": lake_columns,
        "lake_water_all_finite": lake_water_all_finite,
        "maximum_rss_gib": maximum_resident_set_size_gib(),
        "maximum_rss_limit_gib": configuration.execution.maximum_rss_gib,
        "model_step": int(jax.device_get(call_clock.model_step)),
        "maximum_hydrology_substeps": maximum_hydrology_substeps,
        "nonlake_snow_columns": nonlake_snow_columns,
        "patches": int(topology.patch.gridcell.shape[0]),
        "patch_to_column_all_finite": patch_to_column_finite,
        "start": configuration.time.start.isoformat(),
        "snow_age_all_finite": snow_age_all_finite,
        "status": "post_lake_hydrology_boundary_complete",
        "x64_enabled": runtime.x64_enabled,
    }
    _write_json(
        configuration.output.directory / "physical_driver_boundary.json",
        summary,
        overwrite=configuration.output.overwrite,
    )
    return summary


def _run_single_land_timestep(configuration, case, runtime) -> dict:

    boundary = _run_single_physical_boundary(
        configuration,
        case,
        runtime,
        return_boundary=True,
    )
    if not isinstance(boundary, _Stage400Boundary):
        raise TypeError("stage-400 production boundary returned a summary artifact")
    release_completed_stage_memory()
    completed_stages = boundary.completed_stages

    def stage_completed(name: str) -> None:
        completed_stages.append(name)
        peak_rss_gib = maximum_resident_set_size_gib()
        print(
            json.dumps(
                {
                    "current_rss_gib": current_resident_set_size_gib(),
                    "memory_map_count": (
                        current_memory_map_count() if stage_timing_synchronized() else None
                    ),
                    "maximum_rss_gib": peak_rss_gib,
                    "stage": name,
                    "status": "complete" if stage_timing_synchronized() else "dispatched",
                }
            ),
            flush=True,
        )
        if peak_rss_gib > configuration.execution.maximum_rss_gib:
            raise MemoryError(
                f"{name} exceeded the configured peak RSS limit: "
                f"{peak_rss_gib:.3f} > "
                f"{configuration.execution.maximum_rss_gib:.3f} GiB"
            )

    from clm_jax.io.history import ClmHistoryWriter
    from clm_jax.land_timestep import execute_stage400_to_570

    river_volume = None
    if configuration.forcing.river_volume_file is not None:
        from clm_jax.forcing.river import RiverVolumeForcing
        river = RiverVolumeForcing.load(configuration.forcing.river_volume_file, boundary.topology)
        river_volume = river.load_steps(
            [configuration.time.start], configuration.time.time_step_seconds,
        )[0]
    result = execute_stage400_to_570(
        configuration,
        case,
        boundary,
        stage_completed=stage_completed,
        capture_history=True,
        river_volume_gridcell=river_volume,
    )
    if result.history is None:
        raise RuntimeError("single-step integration did not return its final history snapshot")
    jax.block_until_ready((result.carry, result.history))
    history_writer = ClmHistoryWriter(
        case,
        configuration.output.directory,
        boundary.topology,
        overwrite=configuration.output.overwrite,
    )
    history_record = history_writer.append(jax.device_get(result.history), boundary.clock)
    history_files = (history_record.h1_path, history_record.h2_path)
    history_validation = _compare_configured_history(configuration, history_files)
    maximum_rss_gib = maximum_resident_set_size_gib()
    memory_within_limit = maximum_rss_gib <= configuration.execution.maximum_rss_gib
    science_clean = bool(result.diagnostics["science_clean"])
    run_clean = science_clean and memory_within_limit
    summary = {
        "backend": runtime.backend,
        "boundary": "stage570_end_of_step_vegetation",
        "call_time": configuration.time.stop.isoformat(),
        "case": str(case.root),
        "columns": int(boundary.topology.column.gridcell.shape[0]),
        "complete_land_time_step": True,
        "completed_stages": completed_stages,
        "coupler_restart_file": str(configuration.case.coupler_restart_file),
        "devices": len(runtime.devices),
        "diagnostics": result.diagnostics,
        "gridcells": int(boundary.topology.grid.longitude_degrees.shape[0]),
        "history_files": [str(path) for path in history_files],
        "history_validation": history_validation,
        "land_restart_file": str(configuration.case.restart_file),
        "river_volume_boundary": _river_boundary_provenance(configuration),
        "landunits": int(boundary.topology.landunit.gridcell.shape[0]),
        "maximum_rss_gib": maximum_rss_gib,
        "maximum_rss_limit_gib": configuration.execution.maximum_rss_gib,
        "memory_within_limit": memory_within_limit,
        "model_step": int(jax.device_get(boundary.clock.model_step)),
        "patches": int(boundary.topology.patch.gridcell.shape[0]),
        "start": configuration.time.start.isoformat(),
        "run_clean": run_clean,
        "status": (
            "land_timestep_complete"
            if run_clean
            else (
                "land_timestep_failed_science_gate"
                if not science_clean
                else "land_timestep_failed_memory_gate"
            )
        ),
        "x64_enabled": runtime.x64_enabled,
    }
    _write_json(
        configuration.output.directory / "land_timestep_summary.json",
        summary,
        overwrite=configuration.output.overwrite,
    )
    return summary


def _run_land_trajectory(configuration, case, runtime, *, pin_dispatch=None) -> dict:

    from clm_jax.forcing.aerosol import AerosolDepositionForcing
    from clm_jax.forcing.timeline import RestartAwareWfde5ChunkLoader
    from clm_jax.io.history import ClmHistoryWriter
    from clm_jax.io.restart import load_restart_clock, load_subgrid_topology
    from clm_jax.land_timestep import (
        execute_stage400_to_570,
        project_prior_land_timestep_state,
    )
    from clm_jax.trajectory import noleap_clock_from_restart, prepare_trajectory_step

    trajectory_started = perf_counter()
    loader = RestartAwareWfde5ChunkLoader.from_case(
        case,
        configuration.case.restart_file,
        configuration.forcing.nearest_fill_mapping,
        configuration.forcing.bilinear_remapping,
        coupler_restart_file=configuration.case.coupler_restart_file,
    )
    descriptors = iter(
        loader.time_axis.iter_descriptors(
            configuration.time.start,
            configuration.time.stop,
            configuration.time.time_step_seconds,
        )
    )
    clock = noleap_clock_from_restart(load_restart_clock(configuration.case.restart_file))
    river_forcing = None
    topology = load_subgrid_topology(configuration.case.restart_file)
    aerosol_forcing = AerosolDepositionForcing.from_case(
        case, topology)
    if configuration.forcing.river_volume_file is not None:
        from clm_jax.forcing.river import RiverVolumeForcing
        river_forcing = RiverVolumeForcing.load(
            configuration.forcing.river_volume_file,
            topology,
        )
    carry = None
    completed_steps = 0
    completed_stage_count = 0
    last_completed_stages: list[str] = []
    history: list[dict[str, Any]] = []
    step_wall_seconds: list[float] = []
    step_nonwriting_seconds: list[float] = []
    step_history_write_seconds: list[float] = []
    step_history_transfer_seconds: list[float] = []
    forcing_chunk_seconds: list[float] = []
    science_clean = True
    dimensions: dict[str, int] | None = None
    history_writer: ClmHistoryWriter | None = None
    gpu_dispatch_cpu = None
    single_gpu = runtime.backend == "gpu" and len(runtime.devices) == 1
    resume_start_step = 0
    checkpoint_write_seconds: list[float] = []
    checkpoint_contract = None
    checkpoint_directory = configuration.output.directory.parent / "checkpoints"
    if configuration.output.checkpointing:
        from clm_jax.trajectory_checkpoint import (
            load_checkpoint,
            model_contract,
            recover_uncommitted_history,
        )

        segment_steps = (
            configuration.output.history_interval_steps
            * configuration.output.history_records_per_file
        )
        if configuration.time.step_count % segment_steps:
            raise ValueError("checkpointed trajectories must end on a closed history segment")
        checkpoint_contract = model_contract(configuration)
        saved = load_checkpoint(checkpoint_directory, checkpoint_contract)
        if saved is not None:
            resume_start_step = int(saved["completed_steps"])
            if (
                resume_start_step >= configuration.time.step_count
                or resume_start_step % segment_steps
            ):
                raise ValueError("checkpoint does not precede this stop on a closed segment")
            carry = co_locate_array_leaves(saved["carry"], jax.devices()[0])
            clock = jax.device_put(saved["clock"])
            completed_steps = resume_start_step
            completed_stage_count = int(saved["completed_stage_count"])
            dimensions = saved["dimensions"]
            science_clean = bool(saved["science_clean"])
            descriptors = islice(descriptors, resume_start_step, None)
            archived = recover_uncommitted_history(
                configuration.output.directory, case.root.name, saved["clock"],
            )
            print(json.dumps({"stage": "checkpoint_restore", "status": "complete",
                              "completed_steps": resume_start_step,
                              "archived_uncommitted_files": archived}), flush=True)
            del saved

    for chunk_start in range(
        resume_start_step,
        configuration.time.step_count,
        configuration.forcing.chunk_steps,
    ):
        chunk_descriptors = tuple(islice(descriptors, configuration.forcing.chunk_steps))
        if not chunk_descriptors:
            raise RuntimeError("WFDE5 descriptor stream ended before the configured stop")
        forcing_started = perf_counter()
        forcing_chunk = loader.load_chunk(
            chunk_descriptors,
            model_step_seconds=configuration.time.time_step_seconds,
            first_model_step_index=chunk_start,
        )
        river_chunk = (None if river_forcing is None else river_forcing.load_chunk(
            chunk_descriptors, configuration.time.time_step_seconds,
        ))
        aerosol_chunk = aerosol_forcing.load_chunk(
            chunk_descriptors, configuration.time.time_step_seconds)
        jax.block_until_ready((forcing_chunk, river_chunk, aerosol_chunk))
        forcing_chunk_seconds.append(perf_counter() - forcing_started)
        print(json.dumps({
            "stage": "forcing_chunk", "status": "complete",
            "first_model_step_index": chunk_start + 1,
            "steps": len(chunk_descriptors),
            "wall_seconds": forcing_chunk_seconds[-1],
        }), flush=True)

        for chunk_index, descriptor in enumerate(chunk_descriptors):
            step_index = chunk_start + chunk_index
            step_started = perf_counter()
            if step_index == resume_start_step + 2 and pin_dispatch is not None:
                gpu_dispatch_cpu = pin_dispatch()
            forcing_step, call_clock, host_clock = prepare_trajectory_step(
                forcing_chunk, chunk_index, clock,
            )
            call_date, call_time_of_day, call_model_step = map(int, jax.device_get(host_clock))
            if call_date != descriptor.call_date or call_time_of_day != descriptor.call_time_of_day:
                raise RuntimeError(
                    "WFDE5 descriptor and model clock diverged at step "
                    f"{step_index + 1}: descriptor="
                    f"{descriptor.call_date:08d}/{descriptor.call_time_of_day:05d}, "
                    f"clock={call_date:08d}/{call_time_of_day:05d}"
                )

            boundary = _run_single_physical_boundary(
                configuration,
                case,
                runtime,
                return_boundary=True,
                forcing_override=forcing_step,
                aerosol_forcing_override=aerosol_chunk[chunk_index],
                call_clock_override=call_clock,
                prior_carry=carry,
                allow_configured_multistep=True,
            )
            if not isinstance(boundary, _Stage400Boundary):
                raise TypeError("stage-400 trajectory boundary returned a summary artifact")
            if dimensions is None:
                dimensions = {
                    "columns": int(boundary.topology.column.gridcell.shape[0]),
                    "gridcells": int(boundary.topology.grid.longitude_degrees.shape[0]),
                    "landunits": int(boundary.topology.landunit.gridcell.shape[0]),
                    "patches": int(boundary.topology.patch.gridcell.shape[0]),
                }
            prior_state = None if carry is None else project_prior_land_timestep_state(carry)
            carry = None
            release_completed_stage_memory()
            previous_tail_stage_completed = perf_counter()

            def stage_completed(
                name: str,
                *,
                completed_stages=boundary.completed_stages,
                current_step=step_index + 1,
            ) -> None:
                nonlocal previous_tail_stage_completed
                completed_stages.append(name)
                now = perf_counter()
                peak_rss_gib = maximum_resident_set_size_gib()
                if current_step == 1 or stage_timing_synchronized():
                    print(
                        json.dumps(
                            {
                                "current_rss_gib": current_resident_set_size_gib(),
                                "memory_map_count": (
                                    current_memory_map_count()
                                    if stage_timing_synchronized() else None
                                ),
                                "maximum_rss_gib": peak_rss_gib,
                                "model_step_index": current_step,
                                "stage": name,
                                (
                                    "stage_wall_seconds" if stage_timing_synchronized()
                                    else "stage_dispatch_seconds"
                                ): now - previous_tail_stage_completed,
                                "status": (
                                    "complete" if stage_timing_synchronized() else "dispatched"
                                ),
                            }
                        ),
                        flush=True,
                    )
                previous_tail_stage_completed = now
                if peak_rss_gib > configuration.execution.maximum_rss_gib:
                    raise MemoryError(
                        f"step {current_step} {name} exceeded the configured "
                        f"peak RSS limit: {peak_rss_gib:.3f} > "
                        f"{configuration.execution.maximum_rss_gib:.3f} GiB"
                    )

            write_history = (
                (step_index + 1) % configuration.output.history_interval_steps == 0
                or step_index + 1 == configuration.time.step_count
            )
            result = execute_stage400_to_570(
                configuration,
                case,
                boundary,
                stage_completed=stage_completed,
                prior_state=prior_state,
                steps_since_restart=step_index + 1,
                capture_history=write_history,
                river_volume_gridcell=(None if river_chunk is None else river_chunk[chunk_index]),
            )
            del prior_state
            diagnostics = dict(result.diagnostics)
            science_clean = science_clean and bool(diagnostics["science_clean"])
            history_write_elapsed = 0.0
            history_transfer_elapsed = 0.0
            checkpoint_elapsed = 0.0
            carry = (
                co_locate_array_leaves(result.carry, jax.devices()[0])
                if single_gpu else result.carry
            )
            jax.block_until_ready((carry, result.history))
            if write_history:
                if result.history is None:
                    raise RuntimeError(
                        f"step {step_index + 1} did not return its requested history snapshot"
                    )
                transfer_started = perf_counter()
                host_history = jax.device_get(result.history)
                history_transfer_elapsed = perf_counter() - transfer_started
                history_write_started = perf_counter()
                if history_writer is None:
                    history_writer = ClmHistoryWriter(
                        case,
                        configuration.output.directory,
                        boundary.topology,
                        overwrite=configuration.output.overwrite,
                        records_per_file=configuration.output.history_records_per_file,
                    )
                history_writer.append(host_history, call_clock)
                history_write_elapsed = perf_counter() - history_write_started
                del host_history
            clock = call_clock
            completed_steps = step_index + 1
            completed_stage_count += len(boundary.completed_stages)
            last_completed_stages = list(boundary.completed_stages)
            if write_history:
                history.append(
                    {
                        "call_date": call_date,
                        "call_time_of_day": call_time_of_day,
                        "diagnostics": diagnostics,
                        "model_step": call_model_step,
                        "step_index": completed_steps,
                    }
                )

            del boundary, diagnostics, forcing_step, result
            release_completed_stage_memory()
            if not single_gpu:
                carry = _normalize_trajectory_carry_layout(carry)
            release_completed_stage_memory()
            if configuration.output.checkpointing and (
                completed_steps == segment_steps
                or completed_steps % configuration.output.restart_interval_steps == 0
                or completed_steps == configuration.time.step_count
            ):
                from clm_jax.trajectory_checkpoint import save_checkpoint

                checkpoint_started = perf_counter()
                manifest = save_checkpoint(checkpoint_directory, {
                    "carry": carry, "clock": clock, "completed_steps": completed_steps,
                    "completed_stage_count": completed_stage_count,
                    "dimensions": dimensions, "science_clean": science_clean,
                }, checkpoint_contract)
                checkpoint_elapsed = perf_counter() - checkpoint_started
                checkpoint_write_seconds.append(checkpoint_elapsed)
                print(
                    json.dumps({"stage": "checkpoint", "status": "complete", **manifest}),
                    flush=True,
                )
            if completed_steps == resume_start_step + 1 and configuration.time.step_count > 1:
                mappings_before_release = current_memory_map_count()
                release_startup_executables()
                print(
                    json.dumps(
                        {
                            "memory_map_count_after": current_memory_map_count(),
                            "memory_map_count_before": mappings_before_release,
                            "model_step_index": completed_steps,
                            "stage": "startup_executable_release",
                            "status": "complete",
                        }
                    ),
                    flush=True,
                )
            step_elapsed = perf_counter() - step_started
            step_wall_seconds.append(step_elapsed)
            nonwriting_elapsed = step_elapsed - history_write_elapsed - checkpoint_elapsed
            step_nonwriting_seconds.append(nonwriting_elapsed)
            step_history_write_seconds.append(history_write_elapsed)
            step_history_transfer_seconds.append(history_transfer_elapsed)
            if write_history:
                history[-1]["wall_seconds"] = step_elapsed
            print(
                json.dumps(
                    {
                        "current_rss_gib": current_resident_set_size_gib(),
                        "maximum_rss_gib": maximum_resident_set_size_gib(),
                        "model_step_index": completed_steps,
                        "stage": "model_step",
                        "status": "complete",
                        "wall_seconds": step_elapsed,
                        "nonwriting_seconds": nonwriting_elapsed,
                        "history_write_seconds": history_write_elapsed,
                        "history_transfer_seconds": history_transfer_elapsed,
                        "checkpoint_write_seconds": checkpoint_elapsed,
                        "nonwriting_seconds_per_model_hour": (
                            nonwriting_elapsed * 3600 / configuration.time.time_step_seconds
                        ),
                    }
                ),
                flush=True,
            )

        del forcing_chunk, river_chunk, aerosol_chunk
        release_completed_stage_memory()

    if next(descriptors, None) is not None:
        raise RuntimeError("WFDE5 descriptor stream exceeded the configured stop")
    if completed_steps != configuration.time.step_count or dimensions is None:
        raise RuntimeError(
            f"trajectory completed {completed_steps} of "
            f"{configuration.time.step_count} configured steps"
        )

    history_files = (
        ()
        if history_writer is None or history_writer.paths is None
        else history_writer.all_paths
    )
    history_validation = _compare_configured_history(configuration, history_files)
    maximum_rss_gib = maximum_resident_set_size_gib()
    memory_within_limit = maximum_rss_gib <= configuration.execution.maximum_rss_gib
    run_clean = science_clean and memory_within_limit
    summary = {
        "backend": runtime.backend,
        "boundary": "stage570_end_of_step_vegetation",
        "call_time": configuration.time.stop.isoformat(),
        "case": str(case.root),
        "complete_land_trajectory": True,
        "gpu_dispatch_cpu": gpu_dispatch_cpu,
        "completed_stage_count": completed_stage_count,
        "completed_steps": completed_steps,
        "resumed_from_step": resume_start_step,
        "coupler_restart_file": str(configuration.case.coupler_restart_file),
        "devices": len(runtime.devices),
        **dimensions,
        "history": history,
        "history_files": [str(path) for path in history_files],
        "history_interval_steps": configuration.output.history_interval_steps,
        "history_validation": history_validation,
        "land_restart_file": str(configuration.case.restart_file),
        "river_volume_boundary": _river_boundary_provenance(configuration),
        "last_completed_stages": last_completed_stages,
        "aerosol_deposition_boundary": {
            "source": str(aerosol_forcing.path),
            "variables": list(aerosol_forcing.variables),
            "time_sampling": "linear cyclic climatology at the beginning of each CLM step",
        },
        "maximum_rss_gib": maximum_rss_gib,
        "maximum_rss_limit_gib": configuration.execution.maximum_rss_gib,
        "memory_within_limit": memory_within_limit,
        "model_step": int(jax.device_get(clock.model_step)),
        "run_clean": run_clean,
        "science_clean": science_clean,
        "start": configuration.time.start.isoformat(),
        "step_wall_seconds": step_wall_seconds,
        "step_nonwriting_seconds": step_nonwriting_seconds,
        "step_history_write_seconds": step_history_write_seconds,
        "step_history_transfer_seconds": step_history_transfer_seconds,
        "forcing_chunk_seconds": forcing_chunk_seconds,
        "checkpoint_write_seconds": checkpoint_write_seconds,
        "stage_timings_synchronized": stage_timing_synchronized(),
        "timing_definition": (
            "nonwriting_seconds = synchronized complete step wall time minus checkpoint and "
            "history "
            "file creation/append/close; includes host orchestration, diagnostics, device "
            "transfers and any compilation; forcing chunk load/map is recorded separately"
        ),
        "status": (
            "land_trajectory_complete"
            if run_clean
            else (
                "land_trajectory_failed_science_gate"
                if not science_clean
                else "land_trajectory_failed_memory_gate"
            )
        ),
        "stop": configuration.time.stop.isoformat(),
        "trajectory_wall_seconds": perf_counter() - trajectory_started,
        "memory_reclamation": memory_reclamation_statistics(),
        "x64_enabled": runtime.x64_enabled,
    }
    _write_json(
        configuration.output.directory / "land_trajectory_summary.json",
        summary,
        overwrite=configuration.output.overwrite,
    )
    return summary


def _river_boundary_provenance(configuration):
    path = configuration.forcing.river_volume_file
    return {
        "kind": "prescribed_mosart_exchange" if path is not None else "fixed_coupler_restart",
        "file": str(path if path is not None else configuration.case.coupler_restart_file),
    }


@defer_full_collections_to_memory_policy()
def run_configured_model(configuration: ModelConfiguration) -> int:

    runtime = configure_runtime(enable_x64=configuration.execution.enable_x64)
    configure_memory_reclamation(configuration.execution.maximum_rss_gib)
    from clm_jax.io.cime_case import CimeCase

    case = CimeCase.load(configuration.case.directory)
    if not configuration.case.inputdata_root.is_dir():
        raise FileNotFoundError(
            f"configured inputdata root does not exist: {configuration.case.inputdata_root}"
        )
    for description, path in (
        ("land restart", configuration.case.restart_file),
        ("coupler restart", configuration.case.coupler_restart_file),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"configured {description} does not exist: {path}")
    river_path = configuration.forcing.river_volume_file
    if river_path is not None and not river_path.is_file():
        raise FileNotFoundError(f"configured river volume forcing does not exist: {river_path}")
    for description, path in (
        ("nearest-fill mapping", configuration.forcing.nearest_fill_mapping),
        ("bilinear remapping", configuration.forcing.bilinear_remapping),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"configured {description} does not exist: {path}")
    if configuration.validation is not None:
        from clm_jax.validation.history import history_reference_files

        for description, path in (
            ("reference h1 history", configuration.validation.reference_h1),
            ("reference h2 history", configuration.validation.reference_h2),
        ):
            try:
                history_reference_files(path)
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"configured {description} does not exist: {path}"
                ) from error
    case_inputdata = case.path_value("DIN_LOC_ROOT")
    if case_inputdata != configuration.case.inputdata_root:
        raise ValueError(
            f"config.toml inputdata root {configuration.case.inputdata_root} "
            f"disagrees with the CIME case value {case_inputdata}"
        )
    expected_shards = execution_shard_count(configuration.execution)
    if len(runtime.devices) != expected_shards:
        raise RuntimeError(
            f"configuration requires {expected_shards} execution shards, "
            f"but JAX initialized {len(runtime.devices)}"
        )
    from clm_jax.forcing.timeline import Wfde5TimeAxis
    from clm_jax.forcing.wfde5 import Wfde5StreamSet

    forcing_axis = Wfde5TimeAxis.from_streams(Wfde5StreamSet.from_case(case))
    first_sampling_time = configuration.time.start
    last_sampling_time = add_noleap_seconds(
        configuration.time.stop,
        -configuration.time.time_step_seconds,
    )
    forcing_axis.describe(
        first_sampling_time,
        call_time=add_noleap_seconds(
            first_sampling_time,
            configuration.time.time_step_seconds,
        ),
    )
    forcing_axis.describe(last_sampling_time, call_time=configuration.time.stop)
    if not configuration.development.preflight_only:
        from clm_jax.io.native_templates import native_template_scope

        with (
            native_template_scope(enabled=runtime.backend == "gpu"),
            defer_stage_barriers(
                enabled=runtime.backend == "gpu" and not configuration.execution.profile_stages,
            ),
            gpu_dispatch_affinity(
                enabled=runtime.backend == "gpu" and len(runtime.devices) == 1
                and configuration.execution.gpu_pin_dispatch_thread,
            ) as pin_dispatch,
        ):
            summary = (
                _run_single_land_timestep(configuration, case, runtime)
                if configuration.time.step_count == 1
                else _run_land_trajectory(configuration, case, runtime, pin_dispatch=pin_dispatch)
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["run_clean"] else 1
    summary = {
        "backend": runtime.backend,
        "case": str(case.root),
        "compilation_cache_directory": str(configuration.execution.compilation_cache_directory),
        "devices": len(runtime.devices),
        "forcing_chunk_steps": configuration.forcing.chunk_steps,
        "forcing_interval_seconds": forcing_axis.forcing_interval_seconds,
        "forcing_records": len(forcing_axis.instants),
        "inputdata_root": str(configuration.case.inputdata_root),
        "restart_file": str(configuration.case.restart_file),
        "coupler_restart_file": str(configuration.case.coupler_restart_file),
        "start": configuration.time.start.isoformat(),
        "status": "preflight_complete",
        "steps": configuration.time.step_count,
        "stop": configuration.time.stop.isoformat(),
        "x64_enabled": runtime.x64_enabled,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


__all__ = ["run_configured_model"]
