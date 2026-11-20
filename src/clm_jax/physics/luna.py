
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.constants import (
    CLM_SPECIAL_VALUE,
    FREEZING_TEMPERATURE_FRESHWATER,
    UNIVERSAL_GAS_CONSTANT,
)
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.converged_iteration import checkpointed_scan, converged_iteration
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_linear_combination,
    fortran_masked_materialized_divide,
    fortran_materialize,
    fortran_multiply,
    strict_divide,
    strict_exp,
)
from clm_jax.physics.pointwise import scalar_map
from clm_jax.physics.solver_scalarization import _native_tile_trace

_USE_FUSED_LUNA_ALLOCATION = True


def _allocation_scan(step, state, count):
    if _native_tile_trace.get():
        return converged_iteration(step, state, range(count), lambda value: value.active)
    return checkpointed_scan(step, state, jnp.arange(count, dtype=jnp.int32))[0]

CARBON_CONVERSION = 1.2e-5 * 3600.0
MICHAELIS_MENTEN_CO2_25C = 40.49
MICHAELIS_MENTEN_O2_25C = 27840.0
CO2_COMPENSATION_POINT_25C = 4.275
CARBOXYLATION_CAPACITY_PER_NITROGEN_25C = 294.2
ELECTRON_TRANSPORT_CAPACITY_PER_NITROGEN_25C = 1257.0
RESPIRATION_NITROGEN_USE_EFFICIENCY_25C = 33.69
LIGHT_CAPTURE_NITROGEN_USE_EFFICIENCY = 1.78
REFERENCE_O2_PPM = 209460.0
REFERENCE_CO2_PPM = 380.0
REFERENCE_PRESSURE = 101325.0
RESPIRATION_MAINTENANCE_FRACTION = 0.715
MINIMUM_OPTIMIZATION_TEMPERATURE = 5.0
MAXIMUM_OPTIMIZATION_TEMPERATURE = 42.0
BALL_BERRY_SLOPE = 9.0
MINIMUM_OPTIMIZATION_PAR = 200.0
MINIMUM_RELATIVE_HUMIDITY = 0.25
LEAF_RESPIRATION_VCMAX_FRACTION = 0.015
COLIMITATION_CURVATURE = 0.95
MAXIMUM_ALLOCATION_ITERATIONS = 99
MAXIMUM_CI_ITERATIONS = 99
ALLOCATION_STEP_FRACTION = 0.019999999552965164
ENZYME_TURNOVER_Q10 = 2.0
ENZYME_TURNOVER_DAILY = 0.1
STRUCTURAL_NITROGEN_CONCENTRATION = 0.004
MINIMUM_RELATIVE_CANOPY_NITROGEN = 0.2
MAXIMUM_FUNCTIONAL_NITROGEN = 15.0
DEFAULT_REAL_PAR_CONVERSION = 4.599999904632568
PAR_CONVERSION = 4.6


class LunaNitrogenAllocationInputs(NamedTuple):

    functional_nitrogen: jnp.ndarray
    atmospheric_pressure: jnp.ndarray
    relative_humidity: jnp.ndarray
    atmospheric_co2: jnp.ndarray
    atmospheric_o2: jnp.ndarray
    mean_absorbed_par: jnp.ndarray
    maximum_absorbed_par: jnp.ndarray
    boundary_layer_resistance: jnp.ndarray
    daylight_hours: jnp.ndarray
    air_temperature_celsius: jnp.ndarray
    daytime_leaf_temperature_celsius: jnp.ndarray
    nighttime_leaf_temperature_celsius: jnp.ndarray
    jmax_baseline_fraction: jnp.ndarray
    jmax_light_response: jnp.ndarray
    wc_to_wj_baseline: jnp.ndarray
    relative_humidity_exponent: jnp.ndarray
    old_light_capture_fraction: jnp.ndarray
    old_electron_transport_fraction: jnp.ndarray
    old_respiration_fraction: jnp.ndarray
    old_carboxylation_fraction: jnp.ndarray


class LunaNitrogenAllocationResult(NamedTuple):
    storage_fraction: jnp.ndarray
    light_capture_fraction: jnp.ndarray
    electron_transport_fraction: jnp.ndarray
    respiration_fraction: jnp.ndarray
    carboxylation_fraction: jnp.ndarray
    iterations: jnp.ndarray
    storage_nitrogen: jnp.ndarray
    light_capture_nitrogen: jnp.ndarray
    electron_transport_nitrogen: jnp.ndarray
    respiration_nitrogen: jnp.ndarray
    carboxylation_nitrogen: jnp.ndarray
    carboxylation_conversion: jnp.ndarray
    electron_transport_conversion: jnp.ndarray
    intracellular_co2: jnp.ndarray
    maximum_carboxylation_rate: jnp.ndarray
    maximum_electron_transport_rate: jnp.ndarray
    mean_electron_transport_rate: jnp.ndarray
    maximum_light_electron_transport_rate: jnp.ndarray
    photosynthesis: jnp.ndarray
    respiration: jnp.ndarray


class LunaDailyClimateInputs(NamedTuple):

    absorbed_shortwave_vegetation_patch: jnp.ndarray
    vegetation_temperature_patch: jnp.ndarray
    absorbed_par_sunlit_by_layer_patch: jnp.ndarray
    absorbed_par_shaded_by_layer_patch: jnp.ndarray
    sunlit_leaf_area_by_layer_patch: jnp.ndarray
    shaded_leaf_area_by_layer_patch: jnp.ndarray
    radiative_layer_count_patch: jnp.ndarray
    photosynthesis_patch: jnp.ndarray


class LunaDailyClimateState(NamedTuple):

    absorbed_par_sum_by_layer_patch: jnp.ndarray
    absorbed_par_maximum_by_layer_patch: jnp.ndarray
    photosynthesis_sum_patch: jnp.ndarray
    daytime_temperature_sum_patch: jnp.ndarray
    nighttime_temperature_sum_patch: jnp.ndarray
    nighttime_step_count_patch: jnp.ndarray
    daytime_step_count_patch: jnp.ndarray


class LunaTenDayClimateInputs(NamedTuple):

    relative_humidity_patch: jnp.ndarray
    boundary_layer_resistance_patch: jnp.ndarray


class LunaTenDayClimateState(NamedTuple):

    absorbed_par_mean_by_layer_patch: jnp.ndarray
    absorbed_par_maximum_by_layer_patch: jnp.ndarray
    daytime_temperature_patch: jnp.ndarray
    nighttime_temperature_patch: jnp.ndarray
    relative_humidity_patch: jnp.ndarray
    boundary_layer_resistance_patch: jnp.ndarray


class LunaCapacityParameters(NamedTuple):

    c3_pathway_pft: jnp.ndarray
    top_specific_leaf_area_pft: jnp.ndarray
    visible_leaf_reflectance_pft: jnp.ndarray
    visible_leaf_transmittance_pft: jnp.ndarray
    jmax_light_response: jnp.ndarray


class LunaCapacityInputs(NamedTuple):

    atmospheric_pressure_ten_day_patch: jnp.ndarray
    atmospheric_co2_ten_day_patch: jnp.ndarray
    atmospheric_o2_ten_day_patch: jnp.ndarray
    total_leaf_area_patch: jnp.ndarray
    total_leaf_area_by_layer_patch: jnp.ndarray
    daylength_seconds_gridcell: jnp.ndarray
    absorbed_par_mean_by_layer_patch: jnp.ndarray
    absorbed_par_maximum_by_layer_patch: jnp.ndarray
    radiative_layer_count_patch: jnp.ndarray
    leaf_nitrogen_concentration_patch: jnp.ndarray
    air_temperature_ten_day_patch: jnp.ndarray
    daytime_temperature_accumulator_patch: jnp.ndarray
    daytime_leaf_temperature_ten_day_patch: jnp.ndarray
    nighttime_leaf_temperature_ten_day_patch: jnp.ndarray
    relative_humidity_ten_day_patch: jnp.ndarray
    boundary_layer_resistance_ten_day_patch: jnp.ndarray
    daily_photosynthesis_patch: jnp.ndarray


class LunaCapacityState(NamedTuple):

    maximum_carboxylation_rate_by_layer_patch: jnp.ndarray
    maximum_electron_transport_rate_by_layer_patch: jnp.ndarray
    light_capture_fraction_by_layer_patch: jnp.ndarray
    enzyme_activity_by_layer_patch: jnp.ndarray


def _luna_weighted_running_mean(
    previous: jnp.ndarray,
    current: jnp.ndarray,
) -> jnp.ndarray:
    coefficients = jnp.stack(
        (
            jnp.full_like(current, jnp.float64(0.9)),
            jnp.full_like(current, jnp.float64(0.1)),
        )
    )
    return fortran_linear_combination(jnp.stack((previous, current)), coefficients)


def luna_accumulate_daily(
    active_patch: jnp.ndarray,
    timestep_seconds: jnp.ndarray,
    inputs: LunaDailyClimateInputs,
    state: LunaDailyClimateState,
) -> LunaDailyClimateState:

    active = jnp.asarray(active_patch, dtype=jnp.bool_)
    timestep = jnp.asarray(timestep_seconds, dtype=jnp.float64)
    absorbed_shortwave = jnp.asarray(inputs.absorbed_shortwave_vegetation_patch, dtype=jnp.float64)
    vegetation_temperature = jnp.asarray(inputs.vegetation_temperature_patch, dtype=jnp.float64)
    sunlit_par = jnp.asarray(inputs.absorbed_par_sunlit_by_layer_patch, dtype=jnp.float64)
    shaded_par = jnp.asarray(inputs.absorbed_par_shaded_by_layer_patch, dtype=jnp.float64)
    sunlit_leaf_area = jnp.asarray(inputs.sunlit_leaf_area_by_layer_patch, dtype=jnp.float64)
    shaded_leaf_area = jnp.asarray(inputs.shaded_leaf_area_by_layer_patch, dtype=jnp.float64)
    radiative_layer_count = jnp.asarray(inputs.radiative_layer_count_patch, dtype=jnp.int32)
    photosynthesis = jnp.asarray(inputs.photosynthesis_patch, dtype=jnp.float64)

    par_sum = jnp.asarray(state.absorbed_par_sum_by_layer_patch, dtype=jnp.float64)
    par_maximum = jnp.asarray(state.absorbed_par_maximum_by_layer_patch, dtype=jnp.float64)
    photosynthesis_sum = jnp.asarray(state.photosynthesis_sum_patch, dtype=jnp.float64)
    daytime_temperature_sum = jnp.asarray(state.daytime_temperature_sum_patch, dtype=jnp.float64)
    nighttime_temperature_sum = jnp.asarray(
        state.nighttime_temperature_sum_patch, dtype=jnp.float64
    )
    nighttime_step_count = jnp.asarray(state.nighttime_step_count_patch, dtype=jnp.int32)
    daytime_step_count = jnp.asarray(state.daytime_step_count_patch, dtype=jnp.int32)

    if active.ndim != 1 or timestep.shape != ():
        raise ValueError("LUNA active filter must be one-dimensional and timestep scalar")
    patch_shape = active.shape
    scalar_fields = (
        absorbed_shortwave,
        vegetation_temperature,
        radiative_layer_count,
        photosynthesis,
        photosynthesis_sum,
        daytime_temperature_sum,
        nighttime_temperature_sum,
        nighttime_step_count,
        daytime_step_count,
    )
    if any(value.shape != patch_shape for value in scalar_fields):
        raise ValueError("LUNA daily scalar fields must contain one value per patch")
    layer_shape = sunlit_par.shape
    layer_fields = (shaded_par, sunlit_leaf_area, shaded_leaf_area, par_sum, par_maximum)
    if len(layer_shape) != 2 or layer_shape[0] != active.shape[0]:
        raise ValueError("LUNA daily layer fields must have shape (patch, canopy_layer)")
    if any(value.shape != layer_shape for value in layer_fields):
        raise ValueError("all LUNA daily canopy-layer fields must have the same shape")

    special_value = jnp.float64(CLM_SPECIAL_VALUE)
    update_patch = active & (daytime_temperature_sum != special_value)
    is_daytime = absorbed_shortwave > jnp.float64(0.0)
    update_daytime = update_patch & is_daytime
    update_nighttime = update_patch & ~is_daytime
    layer_index = jnp.arange(layer_shape[1], dtype=jnp.int32)
    layer_has_leaf_area = (sunlit_leaf_area + shaded_leaf_area) > jnp.float64(0.0)
    update_layer = (
        update_patch[:, None]
        & (layer_index[None, :] < radiative_layer_count[:, None])
        & layer_has_leaf_area
    )

    par_increment = fortran_multiply(timestep, sunlit_par)
    par_sum_candidate = fortran_add(par_sum, par_increment)
    par_maximum_candidate = jnp.maximum(par_maximum, sunlit_par)
    photosynthesis_sum_candidate = fortran_add(
        photosynthesis_sum,
        fortran_multiply(timestep, photosynthesis),
    )
    daytime_temperature_candidate = fortran_add(
        daytime_temperature_sum,
        vegetation_temperature,
    )
    nighttime_temperature_candidate = fortran_add(
        nighttime_temperature_sum,
        vegetation_temperature,
    )

    return LunaDailyClimateState(
        absorbed_par_sum_by_layer_patch=jnp.where(update_layer, par_sum_candidate, par_sum),
        absorbed_par_maximum_by_layer_patch=jnp.where(
            update_layer,
            par_maximum_candidate,
            par_maximum,
        ),
        photosynthesis_sum_patch=jnp.where(
            update_patch,
            photosynthesis_sum_candidate,
            photosynthesis_sum,
        ),
        daytime_temperature_sum_patch=jnp.where(
            update_daytime,
            daytime_temperature_candidate,
            daytime_temperature_sum,
        ),
        nighttime_temperature_sum_patch=jnp.where(
            update_nighttime,
            nighttime_temperature_candidate,
            nighttime_temperature_sum,
        ),
        nighttime_step_count_patch=jnp.where(
            update_nighttime,
            nighttime_step_count + jnp.int32(1),
            nighttime_step_count,
        ),
        daytime_step_count_patch=jnp.where(
            update_daytime,
            daytime_step_count + jnp.int32(1),
            daytime_step_count,
        ),
    )


def luna_accumulate_ten_day(
    active_patch: jnp.ndarray,
    timestep_seconds: jnp.ndarray,
    inputs: LunaTenDayClimateInputs,
    daily_state: LunaDailyClimateState,
    state: LunaTenDayClimateState,
) -> LunaTenDayClimateState:

    active = jnp.asarray(active_patch, dtype=jnp.bool_)
    timestep = jnp.asarray(timestep_seconds, dtype=jnp.float64)
    relative_humidity = jnp.asarray(inputs.relative_humidity_patch, dtype=jnp.float64)
    boundary_resistance = jnp.asarray(inputs.boundary_layer_resistance_patch, dtype=jnp.float64)
    daily_par_sum = jnp.asarray(daily_state.absorbed_par_sum_by_layer_patch, dtype=jnp.float64)
    daily_par_maximum = jnp.asarray(
        daily_state.absorbed_par_maximum_by_layer_patch, dtype=jnp.float64
    )
    daily_daytime_temperature = jnp.asarray(
        daily_state.daytime_temperature_sum_patch, dtype=jnp.float64
    )
    daily_nighttime_temperature = jnp.asarray(
        daily_state.nighttime_temperature_sum_patch, dtype=jnp.float64
    )
    daytime_count = jnp.asarray(daily_state.daytime_step_count_patch, dtype=jnp.int32)
    nighttime_count = jnp.asarray(daily_state.nighttime_step_count_patch, dtype=jnp.int32)

    running_par_mean = jnp.asarray(state.absorbed_par_mean_by_layer_patch, dtype=jnp.float64)
    running_par_maximum = jnp.asarray(state.absorbed_par_maximum_by_layer_patch, dtype=jnp.float64)
    running_daytime_temperature = jnp.asarray(state.daytime_temperature_patch, dtype=jnp.float64)
    running_nighttime_temperature = jnp.asarray(
        state.nighttime_temperature_patch, dtype=jnp.float64
    )
    running_relative_humidity = jnp.asarray(state.relative_humidity_patch, dtype=jnp.float64)
    running_boundary_resistance = jnp.asarray(
        state.boundary_layer_resistance_patch, dtype=jnp.float64
    )

    if active.ndim != 1 or timestep.shape != ():
        raise ValueError("LUNA active filter must be one-dimensional and timestep scalar")
    patch_shape = active.shape
    scalar_fields = (
        relative_humidity,
        boundary_resistance,
        daily_daytime_temperature,
        daily_nighttime_temperature,
        daytime_count,
        nighttime_count,
        running_daytime_temperature,
        running_nighttime_temperature,
        running_relative_humidity,
        running_boundary_resistance,
    )
    if any(value.shape != patch_shape for value in scalar_fields):
        raise ValueError("LUNA ten-day scalar fields must contain one value per patch")
    layer_shape = daily_par_sum.shape
    if len(layer_shape) != 2 or layer_shape[0] != active.shape[0] or layer_shape[1] < 1:
        raise ValueError("LUNA ten-day layer fields must have shape (patch, canopy_layer>=1)")
    if any(
        value.shape != layer_shape
        for value in (daily_par_maximum, running_par_mean, running_par_maximum)
    ):
        raise ValueError("all LUNA ten-day canopy-layer fields must have the same shape")

    special_value = jnp.float64(CLM_SPECIAL_VALUE)
    update_patch = active & (daily_daytime_temperature != special_value)
    has_daytime = daytime_count > jnp.int32(0)
    has_nighttime = nighttime_count > jnp.int32(0)
    daytime_count_float = daytime_count.astype(jnp.float64)
    nighttime_count_float = nighttime_count.astype(jnp.float64)

    par_denominator = fortran_multiply(timestep, daytime_count_float)
    daily_par_mean = fortran_masked_materialized_divide(
        daily_par_sum,
        par_denominator[:, None],
        update_patch[:, None] & has_daytime[:, None],
        jnp.zeros_like(daily_par_sum),
    )
    first_radiation_day = running_par_mean[:, 0] == special_value
    par_mean_candidate = jnp.where(
        first_radiation_day[:, None],
        daily_par_mean,
        _luna_weighted_running_mean(running_par_mean, daily_par_mean),
    )
    par_maximum_candidate = jnp.where(
        first_radiation_day[:, None],
        daily_par_maximum,
        _luna_weighted_running_mean(running_par_maximum, daily_par_maximum),
    )

    daytime_numerator = jnp.where(
        has_daytime,
        daily_daytime_temperature,
        daily_nighttime_temperature,
    )
    daytime_denominator = jnp.where(
        has_daytime,
        daytime_count_float,
        nighttime_count_float,
    )
    daytime_temperature = fortran_masked_materialized_divide(
        daytime_numerator,
        jnp.where(update_patch, daytime_denominator, jnp.float64(1.0)),
        update_patch,
        running_daytime_temperature,
    )
    nighttime_numerator = jnp.where(
        has_nighttime,
        daily_nighttime_temperature,
        daily_daytime_temperature,
    )
    nighttime_denominator = jnp.where(
        has_nighttime,
        nighttime_count_float,
        daytime_count_float,
    )
    nighttime_temperature = fortran_masked_materialized_divide(
        nighttime_numerator,
        jnp.where(update_patch, nighttime_denominator, jnp.float64(1.0)),
        update_patch,
        running_nighttime_temperature,
    )

    initialized_daytime_temperature = jnp.where(
        running_daytime_temperature == special_value,
        daytime_temperature,
        running_daytime_temperature,
    )
    initialized_nighttime_temperature = jnp.where(
        running_nighttime_temperature == special_value,
        nighttime_temperature,
        running_nighttime_temperature,
    )
    daytime_temperature_candidate = _luna_weighted_running_mean(
        initialized_daytime_temperature,
        daytime_temperature,
    )
    nighttime_temperature_candidate = _luna_weighted_running_mean(
        initialized_nighttime_temperature,
        nighttime_temperature,
    )

    initialized_relative_humidity = jnp.where(
        running_relative_humidity == special_value,
        relative_humidity,
        running_relative_humidity,
    )
    relative_humidity_candidate = _luna_weighted_running_mean(
        initialized_relative_humidity,
        jnp.minimum(jnp.float64(1.0), relative_humidity),
    )
    initialized_boundary_resistance = jnp.where(
        running_boundary_resistance == special_value,
        boundary_resistance,
        running_boundary_resistance,
    )
    boundary_resistance_candidate = _luna_weighted_running_mean(
        initialized_boundary_resistance,
        boundary_resistance,
    )

    return LunaTenDayClimateState(
        absorbed_par_mean_by_layer_patch=jnp.where(
            update_patch[:, None],
            par_mean_candidate,
            running_par_mean,
        ),
        absorbed_par_maximum_by_layer_patch=jnp.where(
            update_patch[:, None],
            par_maximum_candidate,
            running_par_maximum,
        ),
        daytime_temperature_patch=jnp.where(
            update_patch,
            daytime_temperature_candidate,
            running_daytime_temperature,
        ),
        nighttime_temperature_patch=jnp.where(
            update_patch,
            nighttime_temperature_candidate,
            running_nighttime_temperature,
        ),
        relative_humidity_patch=jnp.where(
            update_patch,
            relative_humidity_candidate,
            running_relative_humidity,
        ),
        boundary_layer_resistance_patch=jnp.where(
            update_patch,
            boundary_resistance_candidate,
            running_boundary_resistance,
        ),
    )


def luna_clear_daily(
    active_patch: jnp.ndarray,
    state: LunaDailyClimateState,
) -> LunaDailyClimateState:

    active = jnp.asarray(active_patch, dtype=jnp.bool_)
    arrays = tuple(jnp.asarray(value) for value in state)
    if active.ndim != 1:
        raise ValueError("active_patch must have shape (patch,)")
    if any(value.shape[0] != active.shape[0] for value in arrays):
        raise ValueError("LUNA daily state must use the active patch dimension")
    layer_shape = arrays[0].shape
    if len(layer_shape) != 2 or arrays[1].shape != layer_shape:
        raise ValueError("LUNA daily PAR state must have shape (patch, canopy_layer)")
    if any(value.shape != active.shape for value in arrays[2:]):
        raise ValueError("LUNA daily scalar state must contain one value per patch")

    layer_mask = active[:, None]
    return LunaDailyClimateState(
        absorbed_par_sum_by_layer_patch=jnp.where(
            layer_mask,
            jnp.zeros_like(arrays[0]),
            arrays[0],
        ),
        absorbed_par_maximum_by_layer_patch=jnp.where(
            layer_mask,
            jnp.zeros_like(arrays[1]),
            arrays[1],
        ),
        photosynthesis_sum_patch=jnp.where(active, jnp.zeros_like(arrays[2]), arrays[2]),
        daytime_temperature_sum_patch=jnp.where(
            active,
            jnp.zeros_like(arrays[3]),
            arrays[3],
        ),
        nighttime_temperature_sum_patch=jnp.where(
            active,
            jnp.zeros_like(arrays[4]),
            arrays[4],
        ),
        nighttime_step_count_patch=jnp.where(
            active,
            jnp.zeros_like(arrays[5]),
            arrays[5],
        ),
        daytime_step_count_patch=jnp.where(
            active,
            jnp.zeros_like(arrays[6]),
            arrays[6],
        ),
    )


class _LunaPhotosynthesisResult(NamedTuple):
    intracellular_co2: jnp.ndarray
    carboxylation_conversion: jnp.ndarray
    electron_transport_conversion: jnp.ndarray
    assimilation: jnp.ndarray


class _CiState(NamedTuple):
    intracellular_co2: jnp.ndarray
    previous_intracellular_co2: jnp.ndarray
    stomatal_conductance: jnp.ndarray
    active: jnp.ndarray
    iteration: jnp.ndarray


class _Investments(NamedTuple):
    carboxylation_conversion: jnp.ndarray
    electron_transport_conversion: jnp.ndarray
    intracellular_co2: jnp.ndarray
    maximum_carboxylation_rate: jnp.ndarray
    maximum_electron_transport_rate: jnp.ndarray
    mean_electron_transport_rate: jnp.ndarray
    maximum_light_electron_transport_rate: jnp.ndarray
    electron_transport_nitrogen: jnp.ndarray
    carboxylation_nitrogen: jnp.ndarray
    respiration_nitrogen: jnp.ndarray
    photosynthesis: jnp.ndarray
    respiration: jnp.ndarray


class _AllocationContext(NamedTuple):
    inputs: LunaNitrogenAllocationInputs
    mean_par: jnp.ndarray
    maximum_par: jnp.ndarray
    daytime_leaf_temperature: jnp.ndarray
    nighttime_leaf_temperature: jnp.ndarray
    jmax_coefficient: jnp.ndarray
    carboxylation_capacity_per_nitrogen: jnp.ndarray
    electron_transport_capacity_per_nitrogen: jnp.ndarray
    respiration_nitrogen_use_efficiency: jnp.ndarray
    reference_carboxylation_nue: jnp.ndarray
    reference_electron_transport_nue: jnp.ndarray
    respiration_temperature_sum: jnp.ndarray
    nue_compensation_point: jnp.ndarray
    nue_rubisco_offset: jnp.ndarray
    photosynthesis_compensation_point: jnp.ndarray
    photosynthesis_rubisco_offset: jnp.ndarray
    photosynthesis_boundary_conductance: jnp.ndarray
    nighttime_vcmax_ratio: jnp.ndarray


class _AllocationState(NamedTuple):
    light_capture_nitrogen: jnp.ndarray
    electron_transport_nitrogen: jnp.ndarray
    carboxylation_nitrogen: jnp.ndarray
    respiration_nitrogen: jnp.ndarray
    storage_nitrogen: jnp.ndarray
    carboxylation_conversion: jnp.ndarray
    electron_transport_conversion: jnp.ndarray
    intracellular_co2: jnp.ndarray
    maximum_carboxylation_rate: jnp.ndarray
    maximum_electron_transport_rate: jnp.ndarray
    mean_electron_transport_rate: jnp.ndarray
    maximum_light_electron_transport_rate: jnp.ndarray
    photosynthesis: jnp.ndarray
    respiration: jnp.ndarray
    increasing: jnp.ndarray
    active: jnp.ndarray
    iterations: jnp.ndarray


def _product(*factors: jnp.ndarray | float) -> jnp.ndarray:
    result = jnp.asarray(factors[0], dtype=jnp.float64)
    for factor in factors[1:]:
        result = fortran_multiply(result, jnp.asarray(factor, dtype=jnp.float64))
    return result


def _sum(*terms: jnp.ndarray | float) -> jnp.ndarray:
    values = jnp.broadcast_arrays(*(jnp.asarray(term, dtype=jnp.float64) for term in terms))
    result = values[0]
    for term in values[1:]:
        result = fortran_add(result, term)
    return result


def _temperature_exponent(
    activation_energy: float,
    leaf_temperature: jnp.ndarray,
    gas_constant: jnp.ndarray,
) -> jnp.ndarray:
    reference_temperature = _sum(25.0, FREEZING_TEMPERATURE_FRESHWATER)
    leaf_temperature_kelvin = _sum(FREEZING_TEMPERATURE_FRESHWATER, leaf_temperature)
    activation = fortran_divide(
        jnp.float64(activation_energy),
        _product(gas_constant, reference_temperature),
    )
    temperature_difference = _sum(
        1.0,
        -fortran_divide(reference_temperature, leaf_temperature_kelvin),
    )
    return _product(activation, temperature_difference)


def _kinetic_parameter(
    value_at_25c: float,
    activation_energy: float,
    leaf_temperature: jnp.ndarray,
    gas_constant: jnp.ndarray,
) -> jnp.ndarray:
    return _product(
        value_at_25c,
        strict_exp(_temperature_exponent(activation_energy, leaf_temperature, gas_constant)),
    )


def _vcmx_temperature_response(
    growth_temperature: jnp.ndarray,
    leaf_temperature: jnp.ndarray,
) -> jnp.ndarray:
    gas_constant = _product(UNIVERSAL_GAS_CONSTANT, 1.0e-3)
    reference_temperature = _sum(25.0, FREEZING_TEMPERATURE_FRESHWATER)
    leaf_temperature_kelvin = _sum(leaf_temperature, FREEZING_TEMPERATURE_FRESHWATER)
    constrained_growth = jnp.minimum(
        jnp.maximum(growth_temperature, jnp.float64(11.0)),
        jnp.float64(35.0),
    )
    temperature_limit = _sum(668.39, -_product(1.07, constrained_growth))
    first = _sum(
        1.0,
        strict_exp(
            fortran_divide(
                _sum(_product(temperature_limit, reference_temperature), -200000.0),
                _product(gas_constant, reference_temperature),
            )
        ),
    )
    second = strict_exp(
        _product(
            fortran_divide(72000.0, _product(gas_constant, reference_temperature)),
            _sum(1.0, -fortran_divide(reference_temperature, leaf_temperature_kelvin)),
        )
    )
    third = _sum(
        1.0,
        strict_exp(
            fortran_divide(
                _sum(_product(temperature_limit, leaf_temperature_kelvin), -200000.0),
                _product(gas_constant, leaf_temperature_kelvin),
            )
        ),
    )
    return fortran_divide(_product(first, second), third)


def _jmax_temperature_response(
    growth_temperature: jnp.ndarray,
    leaf_temperature: jnp.ndarray,
) -> jnp.ndarray:
    gas_constant = _product(UNIVERSAL_GAS_CONSTANT, 1.0e-3)
    reference_temperature = _sum(25.0, FREEZING_TEMPERATURE_FRESHWATER)
    leaf_temperature_kelvin = _sum(leaf_temperature, FREEZING_TEMPERATURE_FRESHWATER)
    constrained_growth = jnp.minimum(
        jnp.maximum(growth_temperature, jnp.float64(11.0)),
        jnp.float64(35.0),
    )
    temperature_limit = _sum(659.7, -_product(0.75, constrained_growth))
    first = _sum(
        1.0,
        strict_exp(
            fortran_divide(
                _sum(_product(temperature_limit, reference_temperature), -200000.0),
                _product(gas_constant, reference_temperature),
            )
        ),
    )
    second = strict_exp(
        _product(
            fortran_divide(50000.0, _product(gas_constant, reference_temperature)),
            _sum(1.0, -fortran_divide(reference_temperature, leaf_temperature_kelvin)),
        )
    )
    third = _sum(
        1.0,
        strict_exp(
            fortran_divide(
                _sum(_product(temperature_limit, leaf_temperature_kelvin), -200000.0),
                _product(gas_constant, leaf_temperature_kelvin),
            )
        ),
    )
    return fortran_divide(_product(first, second), third)


def _respiration_temperature_response(leaf_temperature: jnp.ndarray) -> jnp.ndarray:
    denominator = _product(
        UNIVERSAL_GAS_CONSTANT,
        1.0e-6,
        _sum(leaf_temperature, FREEZING_TEMPERATURE_FRESHWATER),
    )
    return strict_exp(_sum(18.72, -fortran_divide(46.39, denominator)))


def _quadratic_roots(
    coefficient_a: jnp.ndarray,
    coefficient_b: jnp.ndarray,
    coefficient_c: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    discriminant = _sum(
        _product(coefficient_b, coefficient_b),
        -_product(4.0, coefficient_a, coefficient_c),
    )
    square_root = jnp.sqrt(discriminant)
    signed_term = jnp.where(
        coefficient_b >= jnp.float64(0.0),
        _sum(coefficient_b, square_root),
        _sum(coefficient_b, -square_root),
    )
    temporary = -_product(0.5, signed_term)
    first = jnp.where(
        coefficient_a == jnp.float64(0.0),
        jnp.float64(1.0e36),
        fortran_divide(temporary, coefficient_a),
    )
    second = jnp.where(
        (coefficient_a != jnp.float64(0.0)) & (temporary != jnp.float64(0.0)),
        fortran_divide(coefficient_c, temporary),
        jnp.float64(1.0e36),
    )
    return first, second


def _conversion_factors(
    intracellular_co2: jnp.ndarray,
    compensation_point: jnp.ndarray,
    rubisco_denominator_offset: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    co2_excess = jnp.maximum(
        _sum(intracellular_co2, -compensation_point),
        jnp.float64(0.0),
    )
    electron = fortran_divide(
        co2_excess,
        _sum(_product(4.0, intracellular_co2), _product(8.0, compensation_point)),
    )
    carboxylation = fortran_divide(
        co2_excess,
        _sum(intracellular_co2, rubisco_denominator_offset),
    )
    return carboxylation, electron


def _photosynthesis_luna(
    context: _AllocationContext,
    maximum_carboxylation_rate: jnp.ndarray,
    mean_electron_transport_rate: jnp.ndarray,
) -> _LunaPhotosynthesisResult:
    inputs = context.inputs
    constrained_humidity = jnp.maximum(
        jnp.float64(MINIMUM_RELATIVE_HUMIDITY),
        inputs.relative_humidity,
    )
    minimum_conductance = fortran_divide(1.0, 2000.0)
    compensation_point = context.photosynthesis_compensation_point
    rubisco_offset = context.photosynthesis_rubisco_offset
    boundary_conductance = context.photosynthesis_boundary_conductance
    initial_ci = _product(0.7, inputs.atmospheric_co2)
    state = _CiState(
        intracellular_co2=initial_ci,
        previous_intracellular_co2=_sum(initial_ci, -0.02),
        stomatal_conductance=jnp.float64(0.0),
        active=jnp.bool_(True),
        iteration=jnp.int32(1),
    )

    def rubisco_step(current: _CiState, _: jnp.ndarray):
        def active_step(value: _CiState):
            old_ci = value.intracellular_co2
            carboxylation_conversion, _ = _conversion_factors(
                old_ci,
                compensation_point,
                rubisco_offset,
            )
            rubisco_rate = _product(
                carboxylation_conversion,
                maximum_carboxylation_rate,
            )
            stomatal_conductance = _sum(
                minimum_conductance,
                _product(
                    fortran_divide(
                        _product(BALL_BERRY_SLOPE, rubisco_rate),
                        inputs.atmospheric_co2,
                    ),
                    inputs.atmospheric_pressure,
                    constrained_humidity,
                ),
            )
            phi = fortran_divide(
                _product(
                    inputs.atmospheric_pressure,
                    _sum(
                        _product(1.37, stomatal_conductance),
                        _product(1.6, boundary_conductance),
                    ),
                ),
                _product(boundary_conductance, stomatal_conductance),
            )
            coefficient_b = _sum(
                rubisco_offset,
                -inputs.atmospheric_co2,
                _product(phi, maximum_carboxylation_rate),
            )
            coefficient_c = -_sum(
                _product(compensation_point, phi, maximum_carboxylation_rate),
                _product(rubisco_offset, inputs.atmospheric_co2),
            )
            first, second = _quadratic_roots(jnp.float64(1.0), coefficient_b, coefficient_c)
            new_ci = jnp.maximum(first, second)
            new_ci = jnp.where(
                new_ci < jnp.float64(0.0),
                _sum(compensation_point, _product(0.5, old_ci)),
                new_ci,
            )
            new_iteration = value.iteration + jnp.int32(1)
            active = (jnp.abs(_sum(new_ci, -old_ci)) > jnp.float64(0.01)) & (
                new_iteration < jnp.int32(100)
            )
            return _CiState(
                new_ci,
                old_ci,
                stomatal_conductance,
                active,
                new_iteration,
            )

        updated = lax.cond(current.active, active_step, lambda value: value, current)
        return updated, None

    state = _allocation_scan(rubisco_step, state, MAXIMUM_CI_ITERATIONS)
    carboxylation_conversion, electron_conversion = _conversion_factors(
        state.intracellular_co2,
        compensation_point,
        rubisco_offset,
    )
    rubisco_rate = _product(carboxylation_conversion, maximum_carboxylation_rate)
    electron_rate = _product(electron_conversion, mean_electron_transport_rate)
    light_limited = electron_rate < rubisco_rate
    light_state = _CiState(
        intracellular_co2=state.intracellular_co2,
        previous_intracellular_co2=_sum(state.intracellular_co2, -0.02),
        stomatal_conductance=state.stomatal_conductance,
        active=light_limited,
        iteration=jnp.int32(1),
    )

    def light_step(current: _CiState, _: jnp.ndarray):
        def active_step(value: _CiState):
            old_ci = value.intracellular_co2
            _, electron_conversion = _conversion_factors(
                old_ci,
                compensation_point,
                rubisco_offset,
            )
            electron_rate = _product(electron_conversion, mean_electron_transport_rate)
            stomatal_conductance = _sum(
                minimum_conductance,
                _product(
                    fortran_divide(
                        _product(BALL_BERRY_SLOPE, electron_rate),
                        inputs.atmospheric_co2,
                    ),
                    inputs.atmospheric_pressure,
                    constrained_humidity,
                ),
            )
            phi = fortran_divide(
                _product(
                    inputs.atmospheric_pressure,
                    _sum(
                        _product(1.37, stomatal_conductance),
                        _product(1.6, boundary_conductance),
                    ),
                ),
                _product(boundary_conductance, stomatal_conductance),
            )
            coefficient_b = _sum(
                _product(2.0, compensation_point),
                -inputs.atmospheric_co2,
                fortran_divide(_product(phi, mean_electron_transport_rate), 4.0),
            )
            coefficient_c = -_sum(
                fortran_divide(
                    _product(compensation_point, phi, mean_electron_transport_rate),
                    4.0,
                ),
                _product(2.0, compensation_point, inputs.atmospheric_co2),
            )
            first, second = _quadratic_roots(jnp.float64(1.0), coefficient_b, coefficient_c)
            new_ci = jnp.maximum(first, second)
            new_ci = jnp.where(
                new_ci < jnp.float64(0.0),
                _sum(compensation_point, _product(0.5, old_ci)),
                new_ci,
            )
            new_iteration = value.iteration + jnp.int32(1)
            active = (jnp.abs(_sum(new_ci, -old_ci)) > jnp.float64(0.01)) & (
                new_iteration < jnp.int32(100)
            )
            return _CiState(
                new_ci,
                old_ci,
                stomatal_conductance,
                active,
                new_iteration,
            )

        updated = lax.cond(current.active, active_step, lambda value: value, current)
        return updated, None

    light_state = _allocation_scan(light_step, light_state, MAXIMUM_CI_ITERATIONS)
    final_ci = light_state.intracellular_co2
    final_carboxylation, final_electron = _conversion_factors(
        final_ci,
        compensation_point,
        rubisco_offset,
    )
    final_rubisco_rate = _product(final_carboxylation, maximum_carboxylation_rate)
    final_electron_rate = _product(final_electron, mean_electron_transport_rate)
    assimilation = _sum(
        _product(
            1.0 - COLIMITATION_CURVATURE,
            jnp.maximum(final_rubisco_rate, final_electron_rate),
        ),
        _product(COLIMITATION_CURVATURE, jnp.minimum(final_rubisco_rate, final_electron_rate)),
    )
    return _LunaPhotosynthesisResult(
        final_ci,
        final_carboxylation,
        final_electron,
        assimilation,
    )


def _reference_nitrogen_use_efficiency() -> tuple[jnp.ndarray, jnp.ndarray]:
    growth_temperature = jnp.float64(25.0)
    leaf_temperature = jnp.float64(25.0)
    carboxylation_capacity = _product(
        _vcmx_temperature_response(growth_temperature, leaf_temperature),
        CARBOXYLATION_CAPACITY_PER_NITROGEN_25C,
    )
    electron_capacity = _product(
        _jmax_temperature_response(growth_temperature, leaf_temperature),
        ELECTRON_TRANSPORT_CAPACITY_PER_NITROGEN_25C,
    )
    co2 = _product(REFERENCE_CO2_PPM, REFERENCE_PRESSURE, 1.0e-6)
    o2 = _product(REFERENCE_O2_PPM, REFERENCE_PRESSURE, 1.0e-6)
    gas_constant = _product(UNIVERSAL_GAS_CONSTANT, 1.0e-3)
    michaelis_co2 = _kinetic_parameter(
        MICHAELIS_MENTEN_CO2_25C,
        79430.0,
        leaf_temperature,
        gas_constant,
    )
    michaelis_o2 = _kinetic_parameter(
        MICHAELIS_MENTEN_O2_25C,
        36380.0,
        leaf_temperature,
        gas_constant,
    )
    compensation_point = _kinetic_parameter(
        CO2_COMPENSATION_POINT_25C,
        37830.0,
        leaf_temperature,
        gas_constant,
    )
    rubisco_offset = _product(michaelis_co2, _sum(1.0, fortran_divide(o2, michaelis_o2)))
    ci = _product(0.7, co2)
    carboxylation, electron = _conversion_factors(ci, compensation_point, rubisco_offset)
    return _product(electron, electron_capacity), _product(carboxylation, carboxylation_capacity)


def _nitrogen_use_efficiency(
    context: _AllocationContext,
    intracellular_co2: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    carboxylation, electron = _conversion_factors(
        intracellular_co2,
        context.nue_compensation_point,
        context.nue_rubisco_offset,
    )
    carboxylation = fortran_materialize(carboxylation)
    electron = fortran_materialize(electron)
    electron_nue = _product(
        electron,
        context.electron_transport_capacity_per_nitrogen,
    )
    carboxylation_nue = _product(
        carboxylation,
        context.carboxylation_capacity_per_nitrogen,
    )
    return electron_nue, carboxylation_nue, strict_divide(electron, carboxylation)


def _nitrogen_investments(
    context: _AllocationContext,
    light_capture_nitrogen: jnp.ndarray,
    carboxylation_nue: jnp.ndarray,
    electron_transport_nue: jnp.ndarray,
    electron_to_carboxylation_conversion_ratio: jnp.ndarray,
    carboxylation_conversion: jnp.ndarray,
    electron_transport_conversion: jnp.ndarray,
    intracellular_co2: jnp.ndarray,
    *,
    update_conversion: bool,
) -> _Investments:
    inputs = context.inputs
    light_efficiency = fortran_materialize(
        fortran_divide(
            0.292,
            _sum(
                1.0,
                fortran_divide(
                    0.076,
                    _product(light_capture_nitrogen, LIGHT_CAPTURE_NITROGEN_USE_EFFICIENCY),
                ),
            ),
        )
    )
    absorbed_electrons = fortran_materialize(_product(light_efficiency, context.mean_par))
    baseline_jmax = fortran_materialize(
        _product(
            inputs.jmax_baseline_fraction,
            inputs.functional_nitrogen,
            context.electron_transport_capacity_per_nitrogen,
        )
    )
    maximum_electron_transport = fortran_materialize(
        _sum(
            baseline_jmax,
            _product(context.jmax_coefficient, absorbed_electrons),
        )
    )
    maximum_light_numerator = fortran_materialize(_product(light_efficiency, context.maximum_par))
    maximum_light_ratio = fortran_materialize(
        strict_divide(maximum_light_numerator, maximum_electron_transport)
    )
    maximum_light_electron_transport = fortran_materialize(
        fortran_divide(
            maximum_light_numerator,
            jnp.sqrt(
                _sum(
                    1.0,
                    _product(maximum_light_ratio, maximum_light_ratio),
                )
            ),
        )
    )
    nue_change = fortran_materialize(
        _product(
            fortran_divide(carboxylation_nue, context.reference_carboxylation_nue),
            fortran_divide(context.reference_electron_transport_nue, electron_transport_nue),
        )
    )
    wc_to_wj = fortran_materialize(
        jnp.minimum(
            jnp.float64(1.0),
            _product(inputs.wc_to_wj_baseline, jnp.sqrt(nue_change)),
        )
    )
    maximum_carboxylation = fortran_materialize(
        _product(
            wc_to_wj,
            maximum_light_electron_transport,
            electron_to_carboxylation_conversion_ratio,
        )
    )
    mean_light_numerator = _product(light_efficiency, context.mean_par)
    mean_light_ratio = fortran_materialize(
        strict_divide(absorbed_electrons, maximum_electron_transport)
    )
    mean_electron_transport = fortran_materialize(
        fortran_divide(
            mean_light_numerator,
            jnp.sqrt(
                _sum(
                    1.0,
                    _product(mean_light_ratio, mean_light_ratio),
                )
            ),
        )
    )
    if update_conversion:
        photosynthesis_result = _photosynthesis_luna(
            context,
            maximum_carboxylation,
            mean_electron_transport,
        )
        carboxylation_conversion = photosynthesis_result.carboxylation_conversion
        electron_transport_conversion = photosynthesis_result.electron_transport_conversion
        intracellular_co2 = photosynthesis_result.intracellular_co2
        assimilation = photosynthesis_result.assimilation
    else:
        rubisco_rate = _product(carboxylation_conversion, maximum_carboxylation)
        electron_rate = _product(electron_transport_conversion, mean_electron_transport)
        assimilation = _sum(
            _product(1.0 - COLIMITATION_CURVATURE, jnp.maximum(rubisco_rate, electron_rate)),
            _product(COLIMITATION_CURVATURE, jnp.minimum(rubisco_rate, electron_rate)),
        )
    photosynthesis = fortran_materialize(
        _product(CARBON_CONVERSION, assimilation, inputs.daylight_hours)
    )
    nighttime_carboxylation = fortran_materialize(
        _product(
            context.nighttime_vcmax_ratio,
            maximum_carboxylation,
        )
    )
    respiration = fortran_materialize(
        _product(
            CARBON_CONVERSION,
            LEAF_RESPIRATION_VCMAX_FRACTION,
            _sum(
                _product(maximum_carboxylation, inputs.daylight_hours),
                _product(nighttime_carboxylation, _sum(24.0, -inputs.daylight_hours)),
            ),
        )
    )
    return _Investments(
        carboxylation_conversion,
        electron_transport_conversion,
        intracellular_co2,
        maximum_carboxylation,
        maximum_electron_transport,
        mean_electron_transport,
        maximum_light_electron_transport,
        fortran_divide(
            maximum_electron_transport,
            context.electron_transport_capacity_per_nitrogen,
        ),
        fortran_divide(
            maximum_carboxylation,
            context.carboxylation_capacity_per_nitrogen,
        ),
        fortran_divide(respiration, context.respiration_nitrogen_use_efficiency),
        photosynthesis,
        respiration,
    )


def _validate_inputs(inputs: LunaNitrogenAllocationInputs) -> None:
    if any(value.shape != () for value in inputs):
        raise ValueError("LUNA nitrogen-allocation inputs must be scalar")


def luna_nitrogen_allocation(
    inputs: LunaNitrogenAllocationInputs,
) -> LunaNitrogenAllocationResult:

    _validate_inputs(inputs)
    return _allocation_iterations(_allocation_context(inputs))


def _allocation_context(inputs: LunaNitrogenAllocationInputs) -> _AllocationContext:
    daytime_temperature = jnp.minimum(
        jnp.maximum(
            inputs.daytime_leaf_temperature_celsius,
            jnp.float64(MINIMUM_OPTIMIZATION_TEMPERATURE),
        ),
        jnp.float64(MAXIMUM_OPTIMIZATION_TEMPERATURE),
    )
    nighttime_temperature = jnp.minimum(
        jnp.maximum(
            inputs.nighttime_leaf_temperature_celsius,
            jnp.float64(MINIMUM_OPTIMIZATION_TEMPERATURE),
        ),
        jnp.float64(MAXIMUM_OPTIMIZATION_TEMPERATURE),
    )
    mean_par = jnp.maximum(jnp.float64(MINIMUM_OPTIMIZATION_PAR), inputs.mean_absorbed_par)
    maximum_par = jnp.maximum(
        jnp.float64(MINIMUM_OPTIMIZATION_PAR),
        inputs.maximum_absorbed_par,
    )
    daylight_ratio = fortran_divide(inputs.daylight_hours, 12.0)
    humidity_response = _sum(
        1.0,
        -strict_exp(
            fortran_divide(
                -_product(
                    inputs.relative_humidity_exponent,
                    jnp.maximum(
                        _sum(inputs.relative_humidity, -MINIMUM_RELATIVE_HUMIDITY),
                        jnp.float64(0.0),
                    ),
                ),
                1.0 - MINIMUM_RELATIVE_HUMIDITY,
            )
        ),
    )
    jmax_coefficient = _product(
        inputs.jmax_light_response,
        _product(daylight_ratio, daylight_ratio),
        humidity_response,
    )
    carboxylation_capacity = _product(
        _vcmx_temperature_response(inputs.air_temperature_celsius, daytime_temperature),
        CARBOXYLATION_CAPACITY_PER_NITROGEN_25C,
    )
    electron_transport_capacity = _product(
        _jmax_temperature_response(inputs.air_temperature_celsius, daytime_temperature),
        ELECTRON_TRANSPORT_CAPACITY_PER_NITROGEN_25C,
    )
    daytime_respiration = _respiration_temperature_response(daytime_temperature)
    nighttime_respiration = _respiration_temperature_response(nighttime_temperature)
    respiration_temperature_sum = _sum(
        _product(daytime_respiration, inputs.daylight_hours),
        _product(nighttime_respiration, _sum(24.0, -inputs.daylight_hours)),
    )
    respiration_nue = _product(
        CARBON_CONVERSION,
        RESPIRATION_NITROGEN_USE_EFFICIENCY_25C,
        respiration_temperature_sum,
    )
    reference_electron_nue, reference_carboxylation_nue = _reference_nitrogen_use_efficiency()
    nue_gas_constant = _product(UNIVERSAL_GAS_CONSTANT, 1.0e-3)
    nue_michaelis_co2 = _kinetic_parameter(
        MICHAELIS_MENTEN_CO2_25C,
        79430.0,
        daytime_temperature,
        nue_gas_constant,
    )
    nue_michaelis_o2 = _kinetic_parameter(
        MICHAELIS_MENTEN_O2_25C,
        36380.0,
        daytime_temperature,
        nue_gas_constant,
    )
    nue_compensation_point = _kinetic_parameter(
        CO2_COMPENSATION_POINT_25C,
        37830.0,
        daytime_temperature,
        nue_gas_constant,
    )
    nue_rubisco_offset = _product(
        nue_michaelis_co2,
        _sum(1.0, fortran_divide(inputs.atmospheric_o2, nue_michaelis_o2)),
    )
    photosynthesis_gas_constant = jnp.float64(8.314)
    photosynthesis_michaelis_co2 = _kinetic_parameter(
        MICHAELIS_MENTEN_CO2_25C,
        79430.0,
        daytime_temperature,
        photosynthesis_gas_constant,
    )
    photosynthesis_michaelis_o2 = _kinetic_parameter(
        MICHAELIS_MENTEN_O2_25C,
        36380.0,
        daytime_temperature,
        photosynthesis_gas_constant,
    )
    photosynthesis_compensation_point = _kinetic_parameter(
        CO2_COMPENSATION_POINT_25C,
        37830.0,
        daytime_temperature,
        photosynthesis_gas_constant,
    )
    photosynthesis_rubisco_offset = _product(
        photosynthesis_michaelis_co2,
        _sum(
            1.0,
            fortran_divide(inputs.atmospheric_o2, photosynthesis_michaelis_o2),
        ),
    )
    leaf_temperature_kelvin = _sum(
        daytime_temperature,
        FREEZING_TEMPERATURE_FRESHWATER,
    )
    photosynthesis_conversion_factor = _product(
        fortran_divide(
            inputs.atmospheric_pressure,
            _product(8.314, leaf_temperature_kelvin),
        ),
        1.0e6,
    )
    photosynthesis_boundary_conductance = fortran_divide(
        photosynthesis_conversion_factor,
        inputs.boundary_layer_resistance,
    )
    nighttime_vcmax_ratio = fortran_divide(
        _vcmx_temperature_response(inputs.air_temperature_celsius, nighttime_temperature),
        _vcmx_temperature_response(inputs.air_temperature_celsius, daytime_temperature),
    )
    return _AllocationContext(
        inputs,
        mean_par,
        maximum_par,
        daytime_temperature,
        nighttime_temperature,
        jmax_coefficient,
        carboxylation_capacity,
        electron_transport_capacity,
        respiration_nue,
        reference_carboxylation_nue,
        reference_electron_nue,
        respiration_temperature_sum,
        nue_compensation_point,
        nue_rubisco_offset,
        photosynthesis_compensation_point,
        photosynthesis_rubisco_offset,
        photosynthesis_boundary_conductance,
        nighttime_vcmax_ratio,
    )


def _allocation_iterations(context: _AllocationContext) -> LunaNitrogenAllocationResult:
    inputs = context.inputs
    respiration_temperature_sum = context.respiration_temperature_sum
    light_capture = _product(inputs.old_light_capture_fraction, inputs.functional_nitrogen)
    light_capture = jnp.where(
        light_capture > _product(inputs.functional_nitrogen, 0.5),
        _product(0.5, inputs.functional_nitrogen),
        light_capture,
    )
    electron_transport = _product(
        inputs.old_electron_transport_fraction,
        inputs.functional_nitrogen,
    )
    respiration = _product(inputs.old_respiration_fraction, inputs.functional_nitrogen)
    carboxylation = _product(inputs.old_carboxylation_fraction, inputs.functional_nitrogen)
    storage = _sum(
        inputs.functional_nitrogen,
        -light_capture,
        -electron_transport,
        -carboxylation,
        -respiration,
    )
    state = _AllocationState(
        light_capture,
        electron_transport,
        carboxylation,
        respiration,
        storage,
        jnp.float64(0.0),
        jnp.float64(0.0),
        _product(0.7, inputs.atmospheric_co2),
        jnp.float64(0.0),
        jnp.float64(0.0),
        jnp.float64(0.0),
        jnp.float64(0.0),
        jnp.float64(0.0),
        jnp.float64(0.0),
        jnp.bool_(False),
        jnp.bool_(True),
        jnp.int32(0),
    )
    change_per_step = _product(
        ALLOCATION_STEP_FRACTION,
        inputs.functional_nitrogen,
    )

    def allocation_step(current: _AllocationState, iteration_index: jnp.ndarray):
        def active_step(value: _AllocationState):
            iteration = iteration_index + jnp.int32(1)
            electron_nue, carboxylation_nue, conversion_ratio = _nitrogen_use_efficiency(
                context,
                value.intracellular_co2,
            )

            def investments(
                light_nitrogen: jnp.ndarray,
                *,
                update_conversion: bool,
            ) -> _Investments:
                return _nitrogen_investments(
                    context,
                    light_nitrogen,
                    carboxylation_nue,
                    electron_nue,
                    conversion_ratio,
                    value.carboxylation_conversion,
                    value.electron_transport_conversion,
                    value.intracellular_co2,
                    update_conversion=update_conversion,
                )

            main = investments(value.light_capture_nitrogen, update_conversion=True)

            def trial_investments(light_nitrogen: jnp.ndarray) -> _Investments:
                return _nitrogen_investments(
                    context,
                    light_nitrogen,
                    carboxylation_nue,
                    electron_nue,
                    conversion_ratio,
                    main.carboxylation_conversion,
                    main.electron_transport_conversion,
                    main.intracellular_co2,
                    update_conversion=False,
                )

            target = _sum(
                value.light_capture_nitrogen,
                main.carboxylation_nitrogen,
                main.electron_transport_nitrogen,
            )
            storage = _sum(
                inputs.functional_nitrogen,
                -target,
                -main.respiration_nitrogen,
            )
            increase_allowed = (storage > jnp.float64(0.0)) & (
                value.increasing | (iteration == jnp.int32(1))
            )

            def try_increase(_: None):
                candidate_light = _sum(value.light_capture_nitrogen, change_per_step)
                candidate_light = jnp.where(
                    fortran_divide(candidate_light, inputs.functional_nitrogen) > jnp.float64(0.95),
                    _product(0.95, inputs.functional_nitrogen),
                    candidate_light,
                )
                candidate = trial_investments(candidate_light)
                candidate_target = _sum(
                    candidate_light,
                    candidate.carboxylation_nitrogen,
                    candidate.electron_transport_nitrogen,
                )
                carbon_cost = _product(
                    _sum(candidate_target, -target),
                    RESPIRATION_MAINTENANCE_FRACTION,
                    CARBON_CONVERSION,
                    respiration_temperature_sum,
                )
                carbon_gain = _sum(candidate.photosynthesis, -main.photosynthesis)
                accept = (carbon_gain > carbon_cost) & (
                    _sum(candidate_target, candidate.respiration_nitrogen)
                    < _product(0.95, inputs.functional_nitrogen)
                )
                return (
                    jnp.where(accept, candidate_light, value.light_capture_nitrogen),
                    jnp.where(
                        accept,
                        candidate.electron_transport_nitrogen,
                        main.electron_transport_nitrogen,
                    ),
                    jnp.where(
                        accept,
                        candidate.carboxylation_nitrogen,
                        main.carboxylation_nitrogen,
                    ),
                    jnp.where(
                        accept,
                        _sum(
                            inputs.functional_nitrogen,
                            -candidate_target,
                            -candidate.respiration_nitrogen,
                        ),
                        storage,
                    ),
                    value.increasing | (accept & (iteration == jnp.int32(1))),
                )

            increased = lax.cond(
                increase_allowed,
                try_increase,
                lambda _: (
                    value.light_capture_nitrogen,
                    main.electron_transport_nitrogen,
                    main.carboxylation_nitrogen,
                    storage,
                    value.increasing,
                ),
                operand=None,
            )

            def try_decrease(_: None):
                candidate_light = jnp.where(
                    increased[3] < jnp.float64(0.0),
                    _product(increased[0], 0.8),
                    _sum(increased[0], -change_per_step),
                )
                candidate_light = jnp.maximum(candidate_light, jnp.float64(0.05))
                candidate = trial_investments(candidate_light)
                candidate_target = _sum(
                    candidate_light,
                    candidate.carboxylation_nitrogen,
                    candidate.electron_transport_nitrogen,
                )
                carbon_cost = _product(
                    _sum(target, -candidate_target),
                    RESPIRATION_MAINTENANCE_FRACTION,
                    CARBON_CONVERSION,
                    respiration_temperature_sum,
                )
                carbon_gain = _sum(main.photosynthesis, -candidate.photosynthesis)
                accept = ((carbon_gain < carbon_cost) & (candidate_light > jnp.float64(0.05))) | (
                    _sum(target, main.respiration_nitrogen)
                    > _product(0.95, inputs.functional_nitrogen)
                )
                return (
                    jnp.where(accept, candidate_light, increased[0]),
                    jnp.where(accept, candidate.electron_transport_nitrogen, increased[1]),
                    jnp.where(accept, candidate.carboxylation_nitrogen, increased[2]),
                    jnp.where(
                        accept,
                        _sum(
                            inputs.functional_nitrogen,
                            -candidate_target,
                            -candidate.respiration_nitrogen,
                        ),
                        increased[3],
                    ),
                )

            decreased = lax.cond(
                ~increased[4],
                try_decrease,
                lambda _: increased[:4],
                operand=None,
            )
            old_fraction = fortran_divide(
                value.light_capture_nitrogen,
                inputs.functional_nitrogen,
            )
            new_fraction = fortran_divide(decreased[0], inputs.functional_nitrogen)
            return _AllocationState(
                decreased[0],
                decreased[1],
                decreased[2],
                main.respiration_nitrogen,
                decreased[3],
                main.carboxylation_conversion,
                main.electron_transport_conversion,
                main.intracellular_co2,
                main.maximum_carboxylation_rate,
                main.maximum_electron_transport_rate,
                main.mean_electron_transport_rate,
                main.maximum_light_electron_transport_rate,
                main.photosynthesis,
                main.respiration,
                increased[4],
                old_fraction != new_fraction,
                iteration,
            )

        updated = lax.cond(current.active, active_step, lambda value: value, current)
        return updated, None

    state = _allocation_scan(allocation_step, state, MAXIMUM_ALLOCATION_ITERATIONS)
    return LunaNitrogenAllocationResult(
        storage_fraction=fortran_divide(state.storage_nitrogen, inputs.functional_nitrogen),
        light_capture_fraction=fortran_divide(
            state.light_capture_nitrogen,
            inputs.functional_nitrogen,
        ),
        electron_transport_fraction=fortran_divide(
            state.electron_transport_nitrogen,
            inputs.functional_nitrogen,
        ),
        respiration_fraction=fortran_divide(
            state.respiration_nitrogen,
            inputs.functional_nitrogen,
        ),
        carboxylation_fraction=fortran_divide(
            state.carboxylation_nitrogen,
            inputs.functional_nitrogen,
        ),
        iterations=state.iterations,
        storage_nitrogen=state.storage_nitrogen,
        light_capture_nitrogen=state.light_capture_nitrogen,
        electron_transport_nitrogen=state.electron_transport_nitrogen,
        respiration_nitrogen=state.respiration_nitrogen,
        carboxylation_nitrogen=state.carboxylation_nitrogen,
        carboxylation_conversion=state.carboxylation_conversion,
        electron_transport_conversion=state.electron_transport_conversion,
        intracellular_co2=state.intracellular_co2,
        maximum_carboxylation_rate=state.maximum_carboxylation_rate,
        maximum_electron_transport_rate=state.maximum_electron_transport_rate,
        mean_electron_transport_rate=state.mean_electron_transport_rate,
        maximum_light_electron_transport_rate=state.maximum_light_electron_transport_rate,
        photosynthesis=state.photosynthesis,
        respiration=state.respiration,
    )


def _luna_elementwise_log(value: jnp.ndarray) -> jnp.ndarray:
    value = jnp.asarray(value, dtype=jnp.float64)
    shape = value.shape
    return jnp.reshape(scalar_map(jnp.log, jnp.reshape(value, (-1,))), shape)


def _luna_elementwise_power(
    base: jnp.ndarray,
    exponent: jnp.ndarray,
) -> jnp.ndarray:
    base, exponent = jnp.broadcast_arrays(
        jnp.asarray(base, dtype=jnp.float64),
        jnp.asarray(exponent, dtype=jnp.float64),
    )
    if jax.default_backend() == "gpu":
        from clm_jax.physics.native_power import native_power

        return native_power(base, exponent)
    shape = base.shape
    result = scalar_map(
        lambda operands: operands[0] ** operands[1],
        (jnp.reshape(base, (-1,)), jnp.reshape(exponent, (-1,))),
    )
    return jnp.reshape(result, shape)


def _validate_luna_capacity_shapes(
    topology: SubgridTopology,
    active_patch: jnp.ndarray,
    parameters: LunaCapacityParameters,
    inputs: LunaCapacityInputs,
    state: LunaCapacityState,
) -> tuple[int, int]:
    number_of_patches = topology.patch.gridcell.shape[0]
    number_of_gridcells = topology.grid.longitude_degrees.shape[0]
    patch_shape = (number_of_patches,)
    if active_patch.shape != patch_shape or active_patch.ndim != 1:
        raise ValueError("LUNA capacity filter must contain one value per patch")
    if topology.patch.vegetation_type.shape != patch_shape:
        raise ValueError("LUNA capacity topology must contain one vegetation type per patch")
    if inputs.daylength_seconds_gridcell.shape != (number_of_gridcells,):
        raise ValueError("LUNA capacity daylength must contain one value per gridcell")

    scalar_fields = (
        inputs.atmospheric_pressure_ten_day_patch,
        inputs.atmospheric_co2_ten_day_patch,
        inputs.atmospheric_o2_ten_day_patch,
        inputs.total_leaf_area_patch,
        inputs.radiative_layer_count_patch,
        inputs.leaf_nitrogen_concentration_patch,
        inputs.air_temperature_ten_day_patch,
        inputs.daytime_temperature_accumulator_patch,
        inputs.daytime_leaf_temperature_ten_day_patch,
        inputs.nighttime_leaf_temperature_ten_day_patch,
        inputs.relative_humidity_ten_day_patch,
        inputs.boundary_layer_resistance_ten_day_patch,
        inputs.daily_photosynthesis_patch,
    )
    if any(value.shape != patch_shape for value in scalar_fields):
        raise ValueError("LUNA capacity scalar inputs must contain one value per patch")

    layer_shape = state.maximum_carboxylation_rate_by_layer_patch.shape
    if len(layer_shape) != 2 or layer_shape[0] != number_of_patches or layer_shape[1] < 1:
        raise ValueError("LUNA capacity state must have shape (patch, canopy_layer>=1)")
    layer_fields = (
        inputs.total_leaf_area_by_layer_patch,
        inputs.absorbed_par_mean_by_layer_patch,
        inputs.absorbed_par_maximum_by_layer_patch,
        state.maximum_electron_transport_rate_by_layer_patch,
        state.light_capture_fraction_by_layer_patch,
        state.enzyme_activity_by_layer_patch,
    )
    if any(value.shape != layer_shape for value in layer_fields):
        raise ValueError("all LUNA capacity layer fields must have the same shape")

    pft_shape = parameters.c3_pathway_pft.shape
    if len(pft_shape) != 1 or pft_shape[0] < 1:
        raise ValueError("LUNA capacity parameters must contain at least one PFT")
    if any(
        value.shape != pft_shape
        for value in (
            parameters.top_specific_leaf_area_pft,
            parameters.visible_leaf_reflectance_pft,
            parameters.visible_leaf_transmittance_pft,
        )
    ):
        raise ValueError("all LUNA capacity PFT parameters must have the same shape")
    if parameters.jmax_light_response.shape != ():
        raise ValueError("LUNA jmax light response must be scalar")
    return number_of_patches, layer_shape[1]


def luna_update_photosynthesis_capacity(
    topology: SubgridTopology,
    active_patch: jnp.ndarray,
    parameters: LunaCapacityParameters,
    inputs: LunaCapacityInputs,
    state: LunaCapacityState,
) -> LunaCapacityState:

    active = jnp.asarray(active_patch, dtype=jnp.bool_)
    parameters = LunaCapacityParameters(
        c3_pathway_pft=jnp.asarray(parameters.c3_pathway_pft, dtype=jnp.bool_),
        top_specific_leaf_area_pft=jnp.asarray(
            parameters.top_specific_leaf_area_pft, dtype=jnp.float64
        ),
        visible_leaf_reflectance_pft=jnp.asarray(
            parameters.visible_leaf_reflectance_pft, dtype=jnp.float64
        ),
        visible_leaf_transmittance_pft=jnp.asarray(
            parameters.visible_leaf_transmittance_pft, dtype=jnp.float64
        ),
        jmax_light_response=jnp.asarray(parameters.jmax_light_response, dtype=jnp.float64),
    )
    inputs = LunaCapacityInputs(
        *(
            jnp.asarray(value, dtype=jnp.int32)
            if name == "radiative_layer_count_patch"
            else jnp.asarray(value, dtype=jnp.float64)
            for name, value in zip(LunaCapacityInputs._fields, inputs, strict=True)
        )
    )
    state = LunaCapacityState(*(jnp.asarray(value, dtype=jnp.float64) for value in state))
    _, number_of_layers = _validate_luna_capacity_shapes(
        topology,
        active,
        parameters,
        inputs,
        state,
    )

    patch_gridcell = jnp.asarray(topology.patch.gridcell, dtype=jnp.int32)
    patch_vegetation = jnp.asarray(topology.patch.vegetation_type, dtype=jnp.int32)
    daylength = inputs.daylength_seconds_gridcell[patch_gridcell]
    c3_pathway = parameters.c3_pathway_pft[patch_vegetation]
    specific_leaf_area = parameters.top_specific_leaf_area_pft[patch_vegetation]
    visible_reflectance = parameters.visible_leaf_reflectance_pft[patch_vegetation]
    visible_transmittance = parameters.visible_leaf_transmittance_pft[patch_vegetation]

    layer_number = jnp.arange(number_of_layers, dtype=jnp.int32)
    initialized = inputs.daytime_temperature_accumulator_patch != jnp.float64(CLM_SPECIAL_VALUE)
    update_patch = active & initialized
    update_layer = update_patch[:, None] & (
        layer_number[None, :] < inputs.radiative_layer_count_patch[:, None]
    )
    has_canopy_nitrogen = (inputs.total_leaf_area_patch > jnp.float64(0.0)) & (
        inputs.leaf_nitrogen_concentration_patch > jnp.float64(0.0)
    )
    growth_patch = (
        update_patch
        & has_canopy_nitrogen
        & c3_pathway
        & (inputs.daily_photosynthesis_patch > jnp.float64(0.0))
    )
    growth_layer = growth_patch[:, None] & update_layer
    decay_layer = (
        update_patch[:, None]
        & has_canopy_nitrogen[:, None]
        & c3_pathway[:, None]
        & (inputs.daily_photosynthesis_patch <= jnp.float64(0.0))[:, None]
        & update_layer
        & (state.enzyme_activity_by_layer_patch > jnp.float64(0.5))
    )
    reset_layer = update_layer & ~has_canopy_nitrogen[:, None]

    daytime_leaf_temperature = _sum(
        inputs.daytime_leaf_temperature_ten_day_patch,
        -FREEZING_TEMPERATURE_FRESHWATER,
    )
    nighttime_leaf_temperature = _sum(
        inputs.nighttime_leaf_temperature_ten_day_patch,
        -FREEZING_TEMPERATURE_FRESHWATER,
    )
    air_temperature = _sum(
        inputs.air_temperature_ten_day_patch,
        -FREEZING_TEMPERATURE_FRESHWATER,
    )
    daylight_temperature = _product(daylength, daytime_leaf_temperature)
    nighttime_seconds = _sum(86400.0, -daylength)
    nighttime_temperature = _product(nighttime_seconds, daytime_leaf_temperature)
    mean_leaf_temperature = fortran_divide(
        _sum(daylight_temperature, nighttime_temperature),
        86400.0,
    )
    turnover_exponent = _product(
        0.1,
        _sum(jnp.minimum(jnp.float64(40.0), mean_leaf_temperature), -25.0),
    )
    turnover_temperature_factor = _luna_elementwise_power(
        jnp.float64(ENZYME_TURNOVER_Q10),
        turnover_exponent,
    )
    maximum_daily_change = _product(
        turnover_temperature_factor,
        ENZYME_TURNOVER_DAILY,
    )

    absorbed_fraction = _sum(1.0, -visible_reflectance, -visible_transmittance)
    calculate_patch = growth_patch & (inputs.radiative_layer_count_patch > jnp.int32(0))
    safe_absorbed_fraction = jnp.where(
        calculate_patch,
        absorbed_fraction,
        jnp.float64(1.0),
    )
    top_radiation = fortran_divide(
        jnp.where(
            calculate_patch,
            inputs.absorbed_par_mean_by_layer_patch[:, 0],
            jnp.float64(1.0),
        ),
        safe_absorbed_fraction,
    )
    top_par = _product(top_radiation, DEFAULT_REAL_PAR_CONVERSION)
    absorbed_par = fortran_divide(
        jnp.where(
            growth_layer,
            inputs.absorbed_par_mean_by_layer_patch,
            jnp.float64(1.0),
        ),
        safe_absorbed_fraction[:, None],
    )
    mean_par = jnp.where(
        inputs.total_leaf_area_by_layer_patch > jnp.float64(0.0),
        _product(absorbed_par, PAR_CONVERSION),
        jnp.float64(0.01),
    )
    mean_par = jnp.where(growth_layer, mean_par, jnp.float64(1.0))
    relative_radiation = fortran_divide(
        mean_par,
        jnp.where(calculate_patch, top_par, jnp.float64(1.0))[:, None],
    )
    relative_canopy_nitrogen = _sum(
        _product(0.1802, _luna_elementwise_log(relative_radiation)),
        1.0,
    )
    relative_canopy_nitrogen = jnp.maximum(
        jnp.float64(MINIMUM_RELATIVE_CANOPY_NITROGEN),
        jnp.minimum(jnp.float64(1.0), relative_canopy_nitrogen),
    )

    safe_specific_leaf_area = jnp.where(
        calculate_patch,
        specific_leaf_area,
        jnp.float64(1.0),
    )
    structural_nitrogen = _product(
        fortran_divide(1.0, safe_specific_leaf_area),
        STRUCTURAL_NITROGEN_CONCENTRATION,
    )
    leaf_nitrogen = inputs.leaf_nitrogen_concentration_patch[:, None]
    functional_nitrogen = jnp.where(
        _product(0.9, leaf_nitrogen) > structural_nitrogen[:, None],
        _product(
            relative_canopy_nitrogen,
            _sum(leaf_nitrogen, -structural_nitrogen[:, None]),
        ),
        _product(relative_canopy_nitrogen, 0.1, leaf_nitrogen),
    )
    functional_nitrogen = jnp.where(
        growth_layer,
        jnp.minimum(jnp.float64(MAXIMUM_FUNCTIONAL_NITROGEN), functional_nitrogen),
        jnp.float64(1.0),
    )

    maximum_to_mean_ratio = fortran_divide(
        jnp.where(
            growth_layer,
            inputs.absorbed_par_maximum_by_layer_patch,
            jnp.float64(1.0),
        ),
        jnp.where(
            growth_layer,
            inputs.absorbed_par_mean_by_layer_patch,
            jnp.float64(1.0),
        ),
    )
    maximum_par = _product(mean_par, maximum_to_mean_ratio)

    layer_shape = growth_layer.shape
    limited_relative_humidity = jnp.minimum(
        jnp.float64(1.0),
        inputs.relative_humidity_ten_day_patch,
    )

    def climate_layer(value: jnp.ndarray, fallback: float) -> jnp.ndarray:
        return jnp.where(
            growth_layer,
            jnp.broadcast_to(value[:, None], layer_shape),
            jnp.float64(fallback),
        )

    allocation_inputs = LunaNitrogenAllocationInputs(
        functional_nitrogen=functional_nitrogen,
        atmospheric_pressure=climate_layer(
            inputs.atmospheric_pressure_ten_day_patch,
            REFERENCE_PRESSURE,
        ),
        relative_humidity=climate_layer(limited_relative_humidity, 0.5),
        atmospheric_co2=climate_layer(inputs.atmospheric_co2_ten_day_patch, 40.0),
        atmospheric_o2=climate_layer(inputs.atmospheric_o2_ten_day_patch, 21000.0),
        mean_absorbed_par=jnp.where(growth_layer, mean_par, jnp.float64(500.0)),
        maximum_absorbed_par=jnp.where(growth_layer, maximum_par, jnp.float64(750.0)),
        boundary_layer_resistance=climate_layer(
            inputs.boundary_layer_resistance_ten_day_patch,
            100.0,
        ),
        daylight_hours=climate_layer(fortran_divide(daylength, 3600.0), 12.0),
        air_temperature_celsius=climate_layer(air_temperature, 20.0),
        daytime_leaf_temperature_celsius=climate_layer(daytime_leaf_temperature, 25.0),
        nighttime_leaf_temperature_celsius=climate_layer(
            nighttime_leaf_temperature,
            15.0,
        ),
        jmax_baseline_fraction=jnp.full(layer_shape, jnp.float64(0.0311)),
        jmax_light_response=jnp.full(layer_shape, parameters.jmax_light_response),
        wc_to_wj_baseline=jnp.full(layer_shape, jnp.float64(0.8054)),
        relative_humidity_exponent=jnp.full(layer_shape, jnp.float64(6.0999)),
        old_light_capture_fraction=jnp.where(
            growth_layer,
            state.light_capture_fraction_by_layer_patch,
            jnp.float64(0.2),
        ),
        old_electron_transport_fraction=jnp.zeros(layer_shape, dtype=jnp.float64),
        old_respiration_fraction=jnp.zeros(layer_shape, dtype=jnp.float64),
        old_carboxylation_fraction=jnp.zeros(layer_shape, dtype=jnp.float64),
    )
    if _USE_FUSED_LUNA_ALLOCATION and jax.default_backend() == "gpu":
        from clm_jax.physics.luna_gpu import allocation_batch
        allocation = allocation_batch(allocation_inputs)
    else:
        allocation = jax.vmap(jax.vmap(luna_nitrogen_allocation))(allocation_inputs)

    optimal_carboxylation = _product(
        allocation.carboxylation_fraction,
        functional_nitrogen,
        CARBOXYLATION_CAPACITY_PER_NITROGEN_25C,
    )
    optimal_electron_transport = _product(
        allocation.electron_transport_fraction,
        functional_nitrogen,
        ELECTRON_TRANSPORT_CAPACITY_PER_NITROGEN_25C,
    )
    maximum_daily_change_layer = maximum_daily_change[:, None]

    carboxylation_change = _sum(
        optimal_carboxylation,
        -state.maximum_carboxylation_rate_by_layer_patch,
    )
    carboxylation_constrained_change = jnp.minimum(
        jnp.abs(carboxylation_change),
        _product(
            state.maximum_carboxylation_rate_by_layer_patch,
            maximum_daily_change_layer,
        ),
    )
    growth_carboxylation = _sum(
        state.maximum_carboxylation_rate_by_layer_patch,
        _product(jnp.copysign(1.0, carboxylation_change), carboxylation_constrained_change),
    )
    growth_carboxylation = jnp.where(
        (growth_carboxylation > jnp.float64(1000.0)) | (growth_carboxylation < jnp.float64(0.0)),
        jnp.float64(50.0),
        growth_carboxylation,
    )

    electron_transport_change = _sum(
        optimal_electron_transport,
        -state.maximum_electron_transport_rate_by_layer_patch,
    )
    electron_transport_constrained_change = jnp.minimum(
        jnp.abs(electron_transport_change),
        _product(
            state.maximum_electron_transport_rate_by_layer_patch,
            maximum_daily_change_layer,
        ),
    )
    growth_electron_transport = _sum(
        state.maximum_electron_transport_rate_by_layer_patch,
        _product(
            jnp.copysign(1.0, electron_transport_change),
            electron_transport_constrained_change,
        ),
    )
    growth_electron_transport = jnp.where(
        (growth_electron_transport > jnp.float64(2000.0))
        | (growth_electron_transport < jnp.float64(0.0)),
        jnp.float64(85.0),
        growth_electron_transport,
    )

    maximum_daily_decay = jnp.minimum(
        jnp.float64(0.5),
        _product(0.1, maximum_daily_change),
    )
    decay_factor = _sum(1.0, -maximum_daily_decay)[:, None]
    decayed_carboxylation = _product(
        state.maximum_carboxylation_rate_by_layer_patch,
        decay_factor,
    )
    decayed_electron_transport = _product(
        state.maximum_electron_transport_rate_by_layer_patch,
        decay_factor,
    )
    decayed_enzyme_activity = _product(
        state.enzyme_activity_by_layer_patch,
        decay_factor,
    )
    recovered_enzyme_activity = jnp.where(
        state.enzyme_activity_by_layer_patch < jnp.float64(1.0),
        _product(
            state.enzyme_activity_by_layer_patch,
            _sum(1.0, maximum_daily_change_layer),
        ),
        state.enzyme_activity_by_layer_patch,
    )

    carboxylation = jnp.where(
        reset_layer,
        jnp.float64(50.0),
        state.maximum_carboxylation_rate_by_layer_patch,
    )
    carboxylation = jnp.where(decay_layer, decayed_carboxylation, carboxylation)
    carboxylation = jnp.where(growth_layer, growth_carboxylation, carboxylation)
    electron_transport = jnp.where(
        reset_layer,
        jnp.float64(85.0),
        state.maximum_electron_transport_rate_by_layer_patch,
    )
    electron_transport = jnp.where(
        decay_layer,
        decayed_electron_transport,
        electron_transport,
    )
    electron_transport = jnp.where(
        growth_layer,
        growth_electron_transport,
        electron_transport,
    )
    light_capture = jnp.where(
        growth_layer,
        allocation.light_capture_fraction,
        state.light_capture_fraction_by_layer_patch,
    )
    enzyme_activity = jnp.where(
        decay_layer,
        decayed_enzyme_activity,
        state.enzyme_activity_by_layer_patch,
    )
    enzyme_activity = jnp.where(
        growth_layer,
        recovered_enzyme_activity,
        enzyme_activity,
    )
    return LunaCapacityState(
        maximum_carboxylation_rate_by_layer_patch=carboxylation,
        maximum_electron_transport_rate_by_layer_patch=electron_transport,
        light_capture_fraction_by_layer_patch=light_capture,
        enzyme_activity_by_layer_patch=enzyme_activity,
    )
