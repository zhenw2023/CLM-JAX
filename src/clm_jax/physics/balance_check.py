
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.restart import (
    COLUMN_URBAN_PERVIOUS_ROAD,
    COLUMN_URBAN_SHADE_WALL,
    COLUMN_URBAN_SUN_WALL,
    LANDUNIT_CROP,
    LANDUNIT_GLACIER,
    LANDUNIT_LAKE,
    LANDUNIT_SOIL,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    LANDUNIT_WETLAND,
    SubgridTopology,
)
from clm_jax.physics.numerics import fortran_add, fortran_multiply

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase


class BalanceCheckParameters(NamedTuple):

    time_step_seconds: float = 1800.0
    skip_steps: int = 2

    @classmethod
    def from_case(cls, case: CimeCase) -> BalanceCheckParameters:

        from clm_jax.io.namelist import read_case_namelist_group  

        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        time_step = float(general["dtime"])
        if time_step <= 0.0:
            raise ValueError("dtime must be positive")
        skip_steps = max(2, int(np.floor(3600.0 / time_step + 0.5)))
        return cls(time_step_seconds=time_step, skip_steps=skip_steps)


DEFAULT_BALANCE_CHECK_PARAMETERS = BalanceCheckParameters()


class BalanceCheckInputs(NamedTuple):

    steps_since_restart: jnp.ndarray
    snow_layer_count_column: jnp.ndarray
    direct_shortwave_gridcell_band: jnp.ndarray
    diffuse_shortwave_gridcell_band: jnp.ndarray
    rainfall_flux_column: jnp.ndarray
    snowfall_flux_column: jnp.ndarray
    downward_longwave_column: jnp.ndarray
    snow_water_equivalent_column: jnp.ndarray
    old_snow_water_equivalent_column: jnp.ndarray
    effective_snow_fraction_column: jnp.ndarray
    beginning_water_mass_column: jnp.ndarray
    ending_water_mass_column: jnp.ndarray
    flood_flux_column: jnp.ndarray
    irrigation_flux_column: jnp.ndarray
    glacier_dynamic_water_flux_column: jnp.ndarray
    total_evaporation_flux_column: jnp.ndarray
    surface_runoff_flux_column: jnp.ndarray
    surface_water_runoff_flux_column: jnp.ndarray
    wetland_glacier_runoff_flux_column: jnp.ndarray
    drainage_flux_column: jnp.ndarray
    perched_drainage_flux_column: jnp.ndarray
    snow_capping_ice_runoff_flux_column: jnp.ndarray
    excess_soil_ice_runoff_flux_column: jnp.ndarray
    snow_capping_discarded_liquid_flux_column: jnp.ndarray
    snow_capping_discarded_ice_flux_column: jnp.ndarray
    ground_precipitation_flux_column: jnp.ndarray
    ground_snow_flux_column: jnp.ndarray
    ground_rain_flux_column: jnp.ndarray
    snow_dew_flux_column: jnp.ndarray
    ground_dew_flux_column: jnp.ndarray
    snow_sublimation_flux_column: jnp.ndarray
    ground_evaporation_flux_column: jnp.ndarray
    snow_drainage_flux_column: jnp.ndarray
    snow_capping_ice_flux_column: jnp.ndarray
    snow_capping_liquid_flux_column: jnp.ndarray
    snow_to_surface_water_flux_column: jnp.ndarray
    surface_water_to_ice_flux_column: jnp.ndarray
    snow_layer_to_top_soil_flux_column: jnp.ndarray
    absorbed_shortwave_patch: jnp.ndarray
    reflected_shortwave_patch: jnp.ndarray
    absorbed_vegetation_shortwave_patch: jnp.ndarray
    absorbed_ground_shortwave_patch: jnp.ndarray
    checked_absorbed_ground_shortwave_patch: jnp.ndarray
    outgoing_longwave_patch: jnp.ndarray
    net_longwave_patch: jnp.ndarray
    sensible_heat_flux_patch: jnp.ndarray
    latent_heat_flux_patch: jnp.ndarray
    ground_heat_flux_patch: jnp.ndarray
    waste_heat_flux_patch: jnp.ndarray
    air_conditioning_heat_flux_patch: jnp.ndarray
    traffic_heat_flux_patch: jnp.ndarray
    soil_energy_balance_error_column: jnp.ndarray


class BalanceCheckState(NamedTuple):

    water_balance_error_column: jnp.ndarray
    snow_balance_error_column: jnp.ndarray
    snow_sources_column: jnp.ndarray
    snow_sinks_column: jnp.ndarray
    solar_balance_error_patch: jnp.ndarray
    longwave_balance_error_patch: jnp.ndarray
    surface_energy_balance_error_patch: jnp.ndarray
    net_radiation_patch: jnp.ndarray


class BalanceCheckDiagnostics(NamedTuple):

    after_skip_steps: jnp.ndarray
    water_warning_column: jnp.ndarray
    snow_warning_column: jnp.ndarray
    solar_warning_patch: jnp.ndarray
    longwave_warning_patch: jnp.ndarray
    surface_energy_warning_patch: jnp.ndarray
    soil_energy_warning_column: jnp.ndarray
    last_water_warning_index: jnp.ndarray
    last_snow_warning_index: jnp.ndarray
    last_solar_warning_index: jnp.ndarray
    last_longwave_warning_index: jnp.ndarray
    last_surface_energy_warning_index: jnp.ndarray
    last_soil_energy_warning_index: jnp.ndarray
    water_fatal: jnp.ndarray
    snow_fatal: jnp.ndarray
    solar_fatal: jnp.ndarray
    longwave_fatal: jnp.ndarray
    surface_energy_fatal: jnp.ndarray
    soil_energy_fatal: jnp.ndarray


class BalanceCheckResult(NamedTuple):
    state: BalanceCheckState
    diagnostics: BalanceCheckDiagnostics


_COLUMN_FLOAT_FIELDS = tuple(
    field
    for field in BalanceCheckInputs._fields
    if field
    not in {
        "steps_since_restart",
        "snow_layer_count_column",
        "direct_shortwave_gridcell_band",
        "diffuse_shortwave_gridcell_band",
        "absorbed_shortwave_patch",
        "reflected_shortwave_patch",
        "absorbed_vegetation_shortwave_patch",
        "absorbed_ground_shortwave_patch",
        "checked_absorbed_ground_shortwave_patch",
        "outgoing_longwave_patch",
        "net_longwave_patch",
        "sensible_heat_flux_patch",
        "latent_heat_flux_patch",
        "ground_heat_flux_patch",
        "waste_heat_flux_patch",
        "air_conditioning_heat_flux_patch",
        "traffic_heat_flux_patch",
    }
)

_PATCH_FLOAT_FIELDS = (
    "absorbed_shortwave_patch",
    "reflected_shortwave_patch",
    "absorbed_vegetation_shortwave_patch",
    "absorbed_ground_shortwave_patch",
    "checked_absorbed_ground_shortwave_patch",
    "outgoing_longwave_patch",
    "net_longwave_patch",
    "sensible_heat_flux_patch",
    "latent_heat_flux_patch",
    "ground_heat_flux_patch",
    "waste_heat_flux_patch",
    "air_conditioning_heat_flux_patch",
    "traffic_heat_flux_patch",
)


def _cast_and_validate(
    topology: SubgridTopology,
    inputs: BalanceCheckInputs,
    state: BalanceCheckState,
    parameters: BalanceCheckParameters,
) -> tuple[BalanceCheckInputs, BalanceCheckState]:
    number_of_gridcells = topology.grid.longitude_degrees.shape[0]
    number_of_columns = topology.column.gridcell.shape[0]
    number_of_patches = topology.patch.gridcell.shape[0]
    column_shape = (number_of_columns,)
    patch_shape = (number_of_patches,)

    if parameters.time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")
    if parameters.skip_steps < 2:
        raise ValueError("skip_steps must be at least two")
    steps = jnp.asarray(inputs.steps_since_restart, dtype=jnp.int32)
    if steps.shape != ():
        raise ValueError("steps_since_restart must be scalar")
    snow_layers = jnp.asarray(inputs.snow_layer_count_column, dtype=jnp.int32)
    if snow_layers.shape != column_shape:
        raise ValueError("snow_layer_count_column must contain one value per column")

    direct = jnp.asarray(inputs.direct_shortwave_gridcell_band, dtype=jnp.float64)
    diffuse = jnp.asarray(inputs.diffuse_shortwave_gridcell_band, dtype=jnp.float64)
    if direct.shape != (number_of_gridcells, 2) or diffuse.shape != (
        number_of_gridcells,
        2,
    ):
        raise ValueError("shortwave forcing must have shape (gridcell, 2)")

    replacements: dict[str, jnp.ndarray] = {
        "steps_since_restart": steps,
        "snow_layer_count_column": snow_layers,
        "direct_shortwave_gridcell_band": direct,
        "diffuse_shortwave_gridcell_band": diffuse,
    }
    for field in _COLUMN_FLOAT_FIELDS:
        value = jnp.asarray(getattr(inputs, field), dtype=jnp.float64)
        if value.shape != column_shape:
            raise ValueError(f"{field} must have shape {column_shape}")
        replacements[field] = value
    for field in _PATCH_FLOAT_FIELDS:
        value = jnp.asarray(getattr(inputs, field), dtype=jnp.float64)
        if value.shape != patch_shape:
            raise ValueError(f"{field} must have shape {patch_shape}")
        replacements[field] = value

    cast_state: dict[str, jnp.ndarray] = {}
    for field in BalanceCheckState._fields:
        expected = column_shape if field.endswith("_column") else patch_shape
        value = jnp.asarray(getattr(state, field), dtype=jnp.float64)
        if value.shape != expected:
            raise ValueError(f"{field} must have shape {expected}")
        cast_state[field] = value
    return BalanceCheckInputs(**replacements), BalanceCheckState(**cast_state)


def _ordered_sum(*terms: jnp.ndarray) -> jnp.ndarray:
    value = terms[0]
    for term in terms[1:]:
        value = fortran_add(value, term)
    return value


def _last_true_index(mask: jnp.ndarray) -> jnp.ndarray:
    indices = jnp.where(mask, jnp.arange(mask.size, dtype=jnp.int32), jnp.int32(-1))
    return jnp.max(jnp.concatenate((jnp.array([-1], dtype=jnp.int32), indices)))


def _fatal_at_last(
    values: jnp.ndarray,
    last_index: jnp.ndarray,
    threshold: float,
    after_skip: jnp.ndarray,
) -> jnp.ndarray:
    safe_index = jnp.maximum(last_index, jnp.int32(0))
    return (
        after_skip
        & (last_index >= jnp.int32(0))
        & (jnp.abs(values[safe_index]) > jnp.float64(threshold))
    )


def balance_check(
    topology: SubgridTopology,
    inputs: BalanceCheckInputs,
    state: BalanceCheckState,
    *,
    parameters: BalanceCheckParameters = DEFAULT_BALANCE_CHECK_PARAMETERS,
) -> BalanceCheckResult:

    inputs, state = _cast_and_validate(topology, inputs, state, parameters)
    active_column = jnp.asarray(topology.column.active, dtype=jnp.bool_)
    active_patch = jnp.asarray(topology.patch.active, dtype=jnp.bool_)
    column_type = jnp.asarray(topology.column.column_type, dtype=jnp.int32)
    column_landunit_type = jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
    patch_landunit_type = jnp.asarray(topology.patch.landunit_type, dtype=jnp.int32)
    patch_column = jnp.asarray(topology.patch.column, dtype=jnp.int32)
    patch_gridcell = jnp.asarray(topology.patch.gridcell, dtype=jnp.int32)

    wall_column = (column_type == jnp.int32(COLUMN_URBAN_SUN_WALL)) | (
        column_type == jnp.int32(COLUMN_URBAN_SHADE_WALL)
    )
    rain = jnp.where(wall_column, jnp.float64(0.0), inputs.rainfall_flux_column)
    snow = jnp.where(wall_column, jnp.float64(0.0), inputs.snowfall_flux_column)

    water_flux_sum = _ordered_sum(
        rain,
        snow,
        inputs.flood_flux_column,
        inputs.irrigation_flux_column,
        inputs.glacier_dynamic_water_flux_column,
        -inputs.total_evaporation_flux_column,
        -inputs.surface_runoff_flux_column,
        -inputs.surface_water_runoff_flux_column,
        -inputs.wetland_glacier_runoff_flux_column,
        -inputs.drainage_flux_column,
        -inputs.perched_drainage_flux_column,
        -inputs.snow_capping_ice_runoff_flux_column,
        -inputs.excess_soil_ice_runoff_flux_column,
        -inputs.snow_capping_discarded_liquid_flux_column,
        -inputs.snow_capping_discarded_ice_flux_column,
    )
    water_storage_change = fortran_add(
        inputs.ending_water_mass_column,
        -inputs.beginning_water_mass_column,
    )
    water_error = fortran_add(
        water_storage_change,
        -fortran_multiply(water_flux_sum, jnp.float64(parameters.time_step_seconds)),
    )
    water_error = jnp.where(active_column, water_error, jnp.float64(0.0))

    default_snow_sources = _ordered_sum(
        inputs.ground_precipitation_flux_column,
        inputs.snow_dew_flux_column,
        inputs.ground_dew_flux_column,
    )
    default_snow_sinks = _ordered_sum(
        inputs.snow_sublimation_flux_column,
        inputs.ground_evaporation_flux_column,
        inputs.snow_drainage_flux_column,
        inputs.snow_capping_ice_flux_column,
        inputs.snow_capping_liquid_flux_column,
        inputs.snow_capping_discarded_ice_flux_column,
        inputs.snow_capping_discarded_liquid_flux_column,
        inputs.snow_layer_to_top_soil_flux_column,
    )
    rain_dew_sum = _ordered_sum(
        inputs.ground_rain_flux_column,
        inputs.snow_dew_flux_column,
        inputs.ground_dew_flux_column,
    )
    fractional_rain_dew = fortran_multiply(
        inputs.effective_snow_fraction_column,
        rain_dew_sum,
    )
    fractional_evaporation = fortran_multiply(
        inputs.effective_snow_fraction_column,
        fortran_add(
            inputs.snow_sublimation_flux_column,
            inputs.ground_evaporation_flux_column,
        ),
    )
    overridden_snow_sinks = _ordered_sum(
        fractional_evaporation,
        inputs.snow_capping_ice_flux_column,
        inputs.snow_capping_liquid_flux_column,
        inputs.snow_capping_discarded_ice_flux_column,
        inputs.snow_capping_discarded_liquid_flux_column,
        inputs.snow_drainage_flux_column,
        inputs.snow_layer_to_top_soil_flux_column,
    )
    lake_column = column_landunit_type == jnp.int32(LANDUNIT_LAKE)
    lake_snow_sources = fortran_add(
        inputs.ground_snow_flux_column,
        fractional_rain_dew,
    )
    natural_or_pervious = (
        (column_type == jnp.int32(COLUMN_URBAN_PERVIOUS_ROAD))
        | (column_landunit_type == jnp.int32(LANDUNIT_SOIL))
        | (column_landunit_type == jnp.int32(LANDUNIT_CROP))
        | (column_landunit_type == jnp.int32(LANDUNIT_WETLAND))
        | (column_landunit_type == jnp.int32(LANDUNIT_GLACIER))
    )
    natural_snow_sources = _ordered_sum(
        fortran_add(
            inputs.ground_snow_flux_column,
            -inputs.snow_to_surface_water_flux_column,
        ),
        fractional_rain_dew,
        inputs.surface_water_to_ice_flux_column,
    )
    snow_sources_candidate = jnp.where(
        lake_column,
        lake_snow_sources,
        default_snow_sources,
    )
    snow_sinks_candidate = jnp.where(
        lake_column,
        overridden_snow_sinks,
        default_snow_sinks,
    )
    snow_sources_candidate = jnp.where(
        natural_or_pervious,
        natural_snow_sources,
        snow_sources_candidate,
    )
    snow_sinks_candidate = jnp.where(
        natural_or_pervious,
        overridden_snow_sinks,
        snow_sinks_candidate,
    )
    has_snow_layer = inputs.snow_layer_count_column < jnp.int32(0)
    update_snow = active_column & has_snow_layer
    clear_snow = active_column & ~has_snow_layer
    snow_sources = jnp.where(
        update_snow,
        snow_sources_candidate,
        jnp.where(clear_snow, jnp.float64(0.0), state.snow_sources_column),
    )
    snow_sinks = jnp.where(
        update_snow,
        snow_sinks_candidate,
        jnp.where(clear_snow, jnp.float64(0.0), state.snow_sinks_column),
    )
    snow_storage_change = fortran_add(
        inputs.snow_water_equivalent_column,
        -inputs.old_snow_water_equivalent_column,
    )
    snow_flux_change = fortran_add(snow_sources_candidate, -snow_sinks_candidate)
    snow_error_candidate = fortran_add(
        snow_storage_change,
        -fortran_multiply(
            snow_flux_change,
            jnp.float64(parameters.time_step_seconds),
        ),
    )
    snow_error = jnp.where(
        update_snow,
        snow_error_candidate,
        jnp.where(clear_snow, jnp.float64(0.0), state.snow_balance_error_column),
    )

    urban_patch = (patch_landunit_type >= jnp.int32(LANDUNIT_URBAN_MIN)) & (
        patch_landunit_type <= jnp.int32(LANDUNIT_URBAN_MAX)
    )
    direct = inputs.direct_shortwave_gridcell_band[patch_gridcell]
    diffuse = inputs.diffuse_shortwave_gridcell_band[patch_gridcell]
    incident_shortwave = _ordered_sum(
        direct[:, 0],
        direct[:, 1],
        diffuse[:, 0],
        diffuse[:, 1],
    )
    solar_error_nonurban = fortran_add(
        fortran_add(
            inputs.absorbed_shortwave_patch,
            inputs.reflected_shortwave_patch,
        ),
        -incident_shortwave,
    )
    solar_candidate = jnp.where(
        urban_patch,
        jnp.float64(CLM_SPECIAL_VALUE),
        solar_error_nonurban,
    )
    longwave_nonurban = _ordered_sum(
        inputs.outgoing_longwave_patch,
        -inputs.net_longwave_patch,
        -inputs.downward_longwave_column[patch_column],
    )
    longwave_candidate = jnp.where(
        urban_patch,
        jnp.float64(CLM_SPECIAL_VALUE),
        longwave_nonurban,
    )
    nonurban_surface = _ordered_sum(
        inputs.absorbed_vegetation_shortwave_patch,
        inputs.checked_absorbed_ground_shortwave_patch,
        inputs.downward_longwave_column[patch_column],
        -inputs.outgoing_longwave_patch,
        -inputs.sensible_heat_flux_patch,
        -inputs.latent_heat_flux_patch,
        -inputs.ground_heat_flux_patch,
    )
    urban_surface = _ordered_sum(
        inputs.absorbed_vegetation_shortwave_patch,
        inputs.absorbed_ground_shortwave_patch,
        -inputs.net_longwave_patch,
        -inputs.sensible_heat_flux_patch,
        -inputs.latent_heat_flux_patch,
        -inputs.ground_heat_flux_patch,
        inputs.waste_heat_flux_patch,
        inputs.air_conditioning_heat_flux_patch,
        inputs.traffic_heat_flux_patch,
    )
    surface_candidate = jnp.where(urban_patch, urban_surface, nonurban_surface)
    net_radiation_candidate = fortran_add(
        inputs.absorbed_shortwave_patch,
        -inputs.net_longwave_patch,
    )
    solar_error = jnp.where(
        active_patch,
        solar_candidate,
        state.solar_balance_error_patch,
    )
    longwave_error = jnp.where(
        active_patch,
        longwave_candidate,
        state.longwave_balance_error_patch,
    )
    surface_error = jnp.where(
        active_patch,
        surface_candidate,
        state.surface_energy_balance_error_patch,
    )
    net_radiation = jnp.where(
        active_patch,
        net_radiation_candidate,
        state.net_radiation_patch,
    )

    water_warning = jnp.abs(water_error) > jnp.float64(1.0e-9)
    snow_warning = active_column & (jnp.abs(snow_error) > jnp.float64(1.0e-9))
    solar_warning = (
        active_patch
        & (solar_error != jnp.float64(CLM_SPECIAL_VALUE))
        & (jnp.abs(solar_error) > jnp.float64(1.0e-7))
    )
    longwave_warning = (
        active_patch
        & (longwave_error != jnp.float64(CLM_SPECIAL_VALUE))
        & (jnp.abs(longwave_error) > jnp.float64(1.0e-7))
    )
    surface_warning = active_patch & (jnp.abs(surface_error) > jnp.float64(1.0e-7))
    soil_warning = active_column & (
        jnp.abs(inputs.soil_energy_balance_error_column) > jnp.float64(1.0e-5)
    )
    last_water = _last_true_index(water_warning)
    last_snow = _last_true_index(snow_warning)
    last_solar = _last_true_index(solar_warning)
    last_longwave = _last_true_index(longwave_warning)
    last_surface = _last_true_index(surface_warning)
    last_soil = _last_true_index(soil_warning)
    after_skip = inputs.steps_since_restart > jnp.int32(parameters.skip_steps)

    diagnostics = BalanceCheckDiagnostics(
        after_skip_steps=after_skip,
        water_warning_column=water_warning,
        snow_warning_column=snow_warning,
        solar_warning_patch=solar_warning,
        longwave_warning_patch=longwave_warning,
        surface_energy_warning_patch=surface_warning,
        soil_energy_warning_column=soil_warning,
        last_water_warning_index=last_water,
        last_snow_warning_index=last_snow,
        last_solar_warning_index=last_solar,
        last_longwave_warning_index=last_longwave,
        last_surface_energy_warning_index=last_surface,
        last_soil_energy_warning_index=last_soil,
        water_fatal=_fatal_at_last(water_error, last_water, 1.0e-5, after_skip),
        snow_fatal=_fatal_at_last(snow_error, last_snow, 1.0e-5, after_skip),
        solar_fatal=_fatal_at_last(solar_error, last_solar, 1.0e-5, after_skip),
        longwave_fatal=_fatal_at_last(longwave_error, last_longwave, 1.0e-5, after_skip),
        surface_energy_fatal=_fatal_at_last(
            surface_error,
            last_surface,
            1.0e-5,
            after_skip,
        ),
        soil_energy_fatal=_fatal_at_last(
            inputs.soil_energy_balance_error_column,
            last_soil,
            1.0e-4,
            after_skip,
        ),
    )
    return BalanceCheckResult(
        state=BalanceCheckState(
            water_balance_error_column=water_error,
            snow_balance_error_column=snow_error,
            snow_sources_column=snow_sources,
            snow_sinks_column=snow_sinks,
            solar_balance_error_patch=solar_error,
            longwave_balance_error_patch=longwave_error,
            surface_energy_balance_error_patch=surface_error,
            net_radiation_patch=net_radiation,
        ),
        diagnostics=diagnostics,
    )
