
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.constants import FREEZING_TEMPERATURE_FRESHWATER, SECONDS_PER_DAY
from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    get_cn_flux_field,
    set_cn_flux_field,
)
from clm_jax.physics.evergreen_phenology import (
    CARBON_TRANSFER_POOL_NAMES,
    NITROGEN_TRANSFER_POOL_NAMES,
    EvergreenPhenologyState,
)
from clm_jax.physics.numerics import (
    fortran_divide,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.seasonal_deciduous_phenology import (
    SeasonalDeciduousPhenologyState,
)

_CROP_SEED_C_TO_LEAF = "vegetation_carbon.crop_seedc_to_leaf_patch"
_CROP_SEED_N_TO_LEAF = "vegetation_nitrogen.crop_seedn_to_leaf_patch"
_LEAF_TRANSFER_INDEX = 0


class CropPhenologyInputs(NamedTuple):

    prognostic_crop_patch: jnp.ndarray
    patch_column_index: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    hemisphere_index_patch: jnp.ndarray
    temperature_2m_patch: jnp.ndarray
    minimum_temperature_2m_patch: jnp.ndarray
    maximum_temperature_2m_patch: jnp.ndarray
    ten_day_temperature_2m_patch: jnp.ndarray
    five_day_minimum_temperature_2m_patch: jnp.ndarray
    ten_day_minimum_temperature_2m_patch: jnp.ndarray
    twenty_year_gdd0_patch: jnp.ndarray
    twenty_year_gdd8_patch: jnp.ndarray
    twenty_year_gdd10_patch: jnp.ndarray
    growing_degree_days_topsoil_patch: jnp.ndarray
    fertilizer_nitrogen_patch: jnp.ndarray
    total_leaf_area_index_patch: jnp.ndarray
    peak_leaf_area_index_patch: jnp.ndarray
    snow_depth_column: jnp.ndarray
    total_vegetation_carbon_patch: jnp.ndarray
    c13_total_vegetation_carbon_patch: jnp.ndarray
    c14_total_vegetation_carbon_patch: jnp.ndarray
    current_julian_day: int | jnp.ndarray
    seconds_of_day: int | jnp.ndarray


class CropPhenologyState(NamedTuple):

    harvest_date_patch: jnp.ndarray
    planting_date_patch: jnp.ndarray
    crop_live_patch: jnp.ndarray
    crop_planted_patch: jnp.ndarray
    vernalization_factor_patch: jnp.ndarray
    crop_phase_patch: jnp.ndarray
    growing_degree_days_since_planting_patch: jnp.ndarray
    growing_degree_days_to_maturity_patch: jnp.ndarray
    heat_units_to_leaf_emergence_patch: jnp.ndarray
    heat_units_to_grain_fill_patch: jnp.ndarray
    cumulative_vernalization_days_patch: jnp.ndarray
    hardening_index_patch: jnp.ndarray
    fertilizer_counter_seconds_patch: jnp.ndarray
    fertilizer_flux_patch: jnp.ndarray
    c13_leaf_carbon_transfer_patch: jnp.ndarray
    c14_leaf_carbon_transfer_patch: jnp.ndarray


class CropPhenologyParameters(NamedTuple):

    leaf_longevity_years_pft: jnp.ndarray
    leaf_carbon_nitrogen_ratio_pft: jnp.ndarray
    manure_nitrogen_pft: jnp.ndarray
    maximum_days_to_maturity_pft: jnp.ndarray
    minimum_planting_temperature_pft: jnp.ndarray
    planting_temperature_pft: jnp.ndarray
    minimum_growing_degree_days_pft: jnp.ndarray
    hybrid_growing_degree_days_pft: jnp.ndarray
    leaf_emergence_fraction_pft: jnp.ndarray
    grain_fill_fraction_pft: jnp.ndarray
    minimum_planting_day_pft_hemisphere: jnp.ndarray
    maximum_planting_day_pft_hemisphere: jnp.ndarray
    winter_wheat_pft: jnp.ndarray
    soybean_pft: jnp.ndarray
    corn_or_sugarcane_pft: jnp.ndarray
    spring_wheat_cotton_or_rice_pft: jnp.ndarray
    year_start_day_hemisphere: jnp.ndarray
    time_step_seconds: float | jnp.ndarray = 1800.0
    radiation_step_seconds: float | jnp.ndarray = 1800.0
    days_per_year: float | jnp.ndarray = 365.0
    seconds_per_day: float | jnp.ndarray = SECONDS_PER_DAY
    freezing_temperature: float | jnp.ndarray = (
        FREEZING_TEMPERATURE_FRESHWATER
    )
    special_value: float | jnp.ndarray = 1.0e36
    initial_seed_carbon: float | jnp.ndarray = 3.0
    carbon_13_ratio: float | jnp.ndarray = 0.011046391077221919
    carbon_14_ratio: float | jnp.ndarray = 1.0e-12
    use_fertilizer: bool | jnp.ndarray = True
    use_carbon_13: bool | jnp.ndarray = False
    use_carbon_14: bool | jnp.ndarray = False
    vernalization_sensitivity: float | jnp.ndarray = 0.003
    hardening_threshold: float | jnp.ndarray = 1.0
    crop_base_temperature: float | jnp.ndarray = 0.0
    not_planted: int | jnp.ndarray = 999
    not_harvested: int | jnp.ndarray = 999


class CropPhenologyDiagnostics(NamedTuple):
    prognostic_crop_patch_count: jnp.ndarray
    planting_patch_count: jnp.ndarray
    normal_planting_patch_count: jnp.ndarray
    forced_planting_patch_count: jnp.ndarray
    phase_two_patch_count: jnp.ndarray
    phase_three_patch_count: jnp.ndarray
    harvest_patch_count: jnp.ndarray
    vernalization_patch_count: jnp.ndarray
    cold_damage_patch_count: jnp.ndarray
    cold_kill_patch_count: jnp.ndarray
    no_emergence_cleanup_patch_count: jnp.ndarray
    inactive_cleanup_patch_count: jnp.ndarray


class CropPhenologyResult(NamedTuple):
    state: CropPhenologyState
    common_phenology_state: SeasonalDeciduousPhenologyState
    phenology_rates: EvergreenPhenologyState
    cn_fluxes: CnTransientFluxState
    diagnostics: CropPhenologyDiagnostics


def _subtract(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return fortran_add(left, -jnp.asarray(right, dtype=jnp.float64))


def _validate_shapes(
    inputs: CropPhenologyInputs,
    state: CropPhenologyState,
    common_state: SeasonalDeciduousPhenologyState,
    phenology_rates: EvergreenPhenologyState,
    cn_fluxes: CnTransientFluxState,
    parameters: CropPhenologyParameters,
) -> int:
    if jnp.ndim(inputs.prognostic_crop_patch) != 1:
        raise ValueError("prognostic_crop_patch must be a vector")
    patch_count = inputs.prognostic_crop_patch.shape[0]
    for name in CropPhenologyInputs._fields[1:21]:
        if name == "snow_depth_column":
            if jnp.ndim(getattr(inputs, name)) != 1:
                raise ValueError("snow_depth_column must be a vector")
        elif jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    for name in CropPhenologyInputs._fields[21:]:
        if jnp.shape(getattr(inputs, name)) != ():
            raise ValueError(f"{name} must be scalar")
    for name in CropPhenologyState._fields:
        if jnp.shape(getattr(state, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    for name in SeasonalDeciduousPhenologyState._fields:
        expected = (patch_count,)
        if name == "carbon_transfer_patch_pool":
            expected = (patch_count, len(CARBON_TRANSFER_POOL_NAMES))
        elif name == "nitrogen_transfer_patch_pool":
            expected = (patch_count, len(NITROGEN_TRANSFER_POOL_NAMES))
        if jnp.shape(getattr(common_state, name)) != expected:
            raise ValueError(f"{name} must have shape {expected}")
    for name in EvergreenPhenologyState._fields:
        if jnp.shape(getattr(phenology_rates, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    pft_names = CropPhenologyParameters._fields[:16]
    pft_count = parameters.leaf_longevity_years_pft.shape[0]
    for name in pft_names[:10]:
        if jnp.shape(getattr(parameters, name)) != (pft_count,):
            raise ValueError(f"{name} must share one PFT axis")
    for name in pft_names[10:12]:
        if jnp.shape(getattr(parameters, name)) != (pft_count, 2):
            raise ValueError(f"{name} must have shape ({pft_count}, 2)")
    for name in pft_names[12:]:
        if jnp.shape(getattr(parameters, name)) != (pft_count,):
            raise ValueError(f"{name} must share one PFT axis")
    if jnp.shape(parameters.year_start_day_hemisphere) != (2,):
        raise ValueError("year_start_day_hemisphere must contain north and south")
    for name in CropPhenologyParameters._fields[17:]:
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    for field_name in (_CROP_SEED_C_TO_LEAF, _CROP_SEED_N_TO_LEAF):
        if get_cn_flux_field(cn_fluxes, field_name).shape != (patch_count,):
            raise ValueError("crop seed flux state must match patch count")
    return patch_count


def _apply_vernalization(
    mask: jnp.ndarray,
    inputs: CropPhenologyInputs,
    state: CropPhenologyState,
    parameters: CropPhenologyParameters,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    zero = jnp.float64(0.0)
    one = jnp.float64(1.0)
    freezing = jnp.asarray(parameters.freezing_temperature, dtype=jnp.float64)
    base = jnp.asarray(parameters.crop_base_temperature, dtype=jnp.float64)
    hardening_threshold = jnp.asarray(parameters.hardening_threshold, dtype=jnp.float64)
    sensitivity = jnp.asarray(parameters.vernalization_sensitivity, dtype=jnp.float64)

    temperature = jnp.asarray(inputs.temperature_2m_patch, dtype=jnp.float64)
    minimum_temperature = jnp.asarray(
        inputs.minimum_temperature_2m_patch,
        dtype=jnp.float64,
    )
    maximum_temperature = jnp.asarray(
        inputs.maximum_temperature_2m_patch,
        dtype=jnp.float64,
    )
    columns = jnp.asarray(inputs.patch_column_index, dtype=jnp.int32)
    snow_depth = jnp.asarray(inputs.snow_depth_column, dtype=jnp.float64)[columns]

    snow_centimeters = fortran_multiply(snow_depth, jnp.float64(100.0))
    snow_term = _subtract(jnp.minimum(snow_centimeters, jnp.float64(15.0)), 15.0)
    snow_square = fortran_multiply(snow_term, snow_term)
    crown_factor = fortran_add(
        jnp.float64(0.4),
        fortran_multiply(jnp.float64(0.0018), snow_square),
    )
    crown_below_freezing = fortran_add(
        jnp.float64(2.0),
        fortran_multiply(_subtract(temperature, freezing), crown_factor),
    )
    crown_temperature = jnp.where(
        temperature < freezing,
        crown_below_freezing,
        _subtract(temperature, freezing),
    )

    cumulative_days = jnp.asarray(
        state.cumulative_vernalization_days_patch,
        dtype=jnp.float64,
    )
    vernalization_factor = jnp.asarray(
        state.vernalization_factor_patch,
        dtype=jnp.float64,
    )
    maximum_above_freezing = mask & (maximum_temperature > freezing)
    accumulation = maximum_above_freezing & (
        minimum_temperature <= fortran_add(freezing, jnp.float64(15.0))
    )
    vd1 = _subtract(
        jnp.float64(1.4),
        fortran_multiply(jnp.float64(0.0778), crown_temperature),
    )
    temperature_range = fortran_add(
        _subtract(maximum_temperature, minimum_temperature),
        jnp.float64(3.0),
    )
    range_square = fortran_multiply(temperature_range, temperature_range)
    safe_range_square = jnp.where(accumulation, range_square, one)
    vd2 = fortran_add(
        jnp.float64(0.5),
        fortran_multiply(
            fortran_divide(jnp.float64(13.44), safe_range_square),
            crown_temperature,
        ),
    )
    vernalization_days = jnp.maximum(zero, jnp.minimum(one, jnp.minimum(vd1, vd2)))
    cumulative_days = jnp.where(
        accumulation,
        fortran_add(cumulative_days, vernalization_days),
        cumulative_days,
    )
    devernalization = maximum_above_freezing & (cumulative_days < 10.0) & (
        maximum_temperature > fortran_add(freezing, jnp.float64(30.0))
    )
    hot_excess = _subtract(
        _subtract(maximum_temperature, freezing),
        jnp.float64(30.0),
    )
    cumulative_days = jnp.where(
        devernalization,
        _subtract(
            cumulative_days,
            fortran_multiply(jnp.float64(0.5), hot_excess),
        ),
        cumulative_days,
    )
    cumulative_days = jnp.where(
        maximum_above_freezing,
        jnp.maximum(zero, cumulative_days),
        cumulative_days,
    )
    factor_candidate = _subtract(
        one,
        fortran_multiply(sensitivity, _subtract(jnp.float64(50.0), cumulative_days)),
    )
    vernalization_factor = jnp.where(
        maximum_above_freezing,
        jnp.maximum(zero, jnp.minimum(factor_candidate, one)),
        vernalization_factor,
    )

    hardening_index = jnp.asarray(state.hardening_index_patch, dtype=jnp.float64)
    hardening_branch = mask & (
        (minimum_temperature <= _subtract(freezing, jnp.float64(3.0)))
        | (hardening_index != zero)
    )
    phase_two_hardening = hardening_branch & (hardening_index >= hardening_threshold)
    hardening_index = jnp.where(
        phase_two_hardening,
        jnp.minimum(
            fortran_add(hardening_index, jnp.float64(0.083)),
            fortran_multiply(hardening_threshold, jnp.float64(2.0)),
        ),
        hardening_index,
    )
    dehardening_temperature = fortran_add(
        fortran_add(base, freezing),
        jnp.float64(10.0),
    )
    dehardening_branch = hardening_branch & (
        maximum_temperature >= dehardening_temperature
    )
    temperature_excess = _subtract(maximum_temperature, base)
    temperature_excess = _subtract(temperature_excess, freezing)
    temperature_excess = _subtract(temperature_excess, jnp.float64(10.0))
    dehardening_amount = fortran_multiply(jnp.float64(0.02), temperature_excess)
    hardening_index = jnp.where(
        dehardening_branch,
        _subtract(hardening_index, dehardening_amount),
        hardening_index,
    )
    second_dehardening = dehardening_branch & (hardening_index > hardening_threshold)
    hardening_index = jnp.where(
        second_dehardening,
        _subtract(hardening_index, dehardening_amount),
        hardening_index,
    )
    hardening_index = jnp.where(
        dehardening_branch,
        jnp.maximum(zero, hardening_index),
        hardening_index,
    )

    initial_hardening_branch = mask & (~hardening_branch) & (
        crown_temperature >= _subtract(base, one)
    )
    cool_crown = initial_hardening_branch & (
        crown_temperature <= fortran_add(base, jnp.float64(8.0))
    )
    crown_displacement = fortran_add(
        _subtract(crown_temperature, base),
        jnp.float64(3.5),
    )
    crown_square = fortran_multiply(crown_displacement, crown_displacement)
    initial_increment = _subtract(
        jnp.float64(0.1),
        fortran_divide(crown_square, jnp.float64(506.0)),
    )
    hardening_index = jnp.where(
        cool_crown,
        fortran_add(hardening_index, initial_increment),
        hardening_index,
    )
    second_initial_phase = cool_crown & (hardening_index >= hardening_threshold) & (
        crown_temperature <= base
    )
    hardening_index = jnp.where(
        second_initial_phase,
        jnp.minimum(
            fortran_add(hardening_index, jnp.float64(0.083)),
            fortran_multiply(hardening_threshold, jnp.float64(2.0)),
        ),
        hardening_index,
    )
    initial_dehardening = initial_hardening_branch & (
        maximum_temperature >= dehardening_temperature
    )
    hardening_index = jnp.where(
        initial_dehardening,
        _subtract(hardening_index, dehardening_amount),
        hardening_index,
    )
    second_initial_dehardening = initial_dehardening & (
        hardening_index > hardening_threshold
    )
    hardening_index = jnp.where(
        second_initial_dehardening,
        _subtract(hardening_index, dehardening_amount),
        hardening_index,
    )
    hardening_index = jnp.where(
        initial_dehardening,
        jnp.maximum(zero, hardening_index),
        hardening_index,
    )

    freeze_exposure = mask & (
        minimum_temperature <= _subtract(freezing, jnp.float64(6.0))
    )
    killing_temperature = _subtract(
        _subtract(base, jnp.float64(6.0)),
        fortran_multiply(jnp.float64(6.0), hardening_index),
    )
    threshold_crossed = freeze_exposure & (killing_temperature >= crown_temperature)
    kill_distance = _subtract(crown_temperature, killing_temperature)
    survival_fraction = _subtract(
        jnp.float64(0.95),
        fortran_multiply(
            jnp.float64(0.02),
            fortran_multiply(kill_distance, kill_distance),
        ),
    )
    cold_damage = threshold_crossed & (survival_fraction >= jnp.float64(0.02))
    cold_kill = threshold_crossed & (~cold_damage) & (
        jnp.asarray(inputs.total_leaf_area_index_patch, dtype=jnp.float64) > zero
    )
    maturity = jnp.where(
        cold_kill,
        zero,
        jnp.asarray(state.growing_degree_days_to_maturity_patch, dtype=jnp.float64),
    )
    grain_heat_units = jnp.where(
        cold_kill,
        zero,
        jnp.asarray(state.heat_units_to_grain_fill_patch, dtype=jnp.float64),
    )
    return (
        cumulative_days,
        vernalization_factor,
        hardening_index,
        maturity,
        grain_heat_units,
        jnp.stack((cold_damage, cold_kill), axis=0),
    )


def crop_phenology(
    inputs: CropPhenologyInputs,
    state: CropPhenologyState,
    common_state: SeasonalDeciduousPhenologyState,
    phenology_rates: EvergreenPhenologyState,
    cn_fluxes: CnTransientFluxState,
    parameters: CropPhenologyParameters,
) -> CropPhenologyResult:

    _validate_shapes(inputs, state, common_state, phenology_rates, cn_fluxes, parameters)
    zero = jnp.float64(0.0)
    one = jnp.float64(1.0)
    crop = jnp.asarray(inputs.prognostic_crop_patch, dtype=jnp.bool_)
    vegetation_type = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    hemisphere = jnp.asarray(inputs.hemisphere_index_patch, dtype=jnp.int32)
    current_day = jnp.asarray(inputs.current_julian_day, dtype=jnp.int32)
    seconds_of_day = jnp.asarray(inputs.seconds_of_day, dtype=jnp.int32)

    def pft_values(name: str, dtype=jnp.float64) -> jnp.ndarray:
        return jnp.asarray(getattr(parameters, name), dtype=dtype)[vegetation_type]

    winter = pft_values("winter_wheat_pft", jnp.bool_)
    soybean = pft_values("soybean_pft", jnp.bool_)
    corn_or_sugarcane = pft_values("corn_or_sugarcane_pft", jnp.bool_)
    spring_wheat_cotton_or_rice = pft_values(
        "spring_wheat_cotton_or_rice_pft",
        jnp.bool_,
    )
    min_planting_day = jnp.asarray(
        parameters.minimum_planting_day_pft_hemisphere,
        dtype=jnp.int32,
    )[vegetation_type, hemisphere]
    max_planting_day = jnp.asarray(
        parameters.maximum_planting_day_pft_hemisphere,
        dtype=jnp.int32,
    )[vegetation_type, hemisphere]
    year_start_day = jnp.asarray(
        parameters.year_start_day_hemisphere,
        dtype=jnp.int32,
    )[hemisphere]

    time_step = jnp.asarray(parameters.time_step_seconds, dtype=jnp.float64)
    radiation_step = jnp.asarray(parameters.radiation_step_seconds, dtype=jnp.float64)
    days_per_year = jnp.asarray(parameters.days_per_year, dtype=jnp.float64)
    seconds_per_day = jnp.asarray(parameters.seconds_per_day, dtype=jnp.float64)
    special_value = jnp.asarray(parameters.special_value, dtype=jnp.float64)
    not_planted = jnp.asarray(parameters.not_planted, dtype=jnp.int32)
    not_harvested = jnp.asarray(parameters.not_harvested, dtype=jnp.int32)

    harvest_date = jnp.asarray(state.harvest_date_patch, dtype=jnp.int32)
    crop_live = jnp.asarray(state.crop_live_patch, dtype=jnp.bool_)
    crop_planted = jnp.asarray(state.crop_planted_patch, dtype=jnp.bool_)
    vernalization_factor = jnp.asarray(
        state.vernalization_factor_patch,
        dtype=jnp.float64,
    )
    crop_phase = jnp.asarray(state.crop_phase_patch, dtype=jnp.float64)
    growing_degree_days = jnp.asarray(
        state.growing_degree_days_since_planting_patch,
        dtype=jnp.float64,
    )
    maturity = jnp.asarray(
        state.growing_degree_days_to_maturity_patch,
        dtype=jnp.float64,
    )
    leaf_heat_units = jnp.asarray(
        state.heat_units_to_leaf_emergence_patch,
        dtype=jnp.float64,
    )
    grain_heat_units = jnp.asarray(
        state.heat_units_to_grain_fill_patch,
        dtype=jnp.float64,
    )
    cumulative_days = jnp.asarray(
        state.cumulative_vernalization_days_patch,
        dtype=jnp.float64,
    )
    hardening_index = jnp.asarray(state.hardening_index_patch, dtype=jnp.float64)
    fertilizer_counter = jnp.asarray(
        state.fertilizer_counter_seconds_patch,
        dtype=jnp.float64,
    )
    fertilizer_flux = jnp.asarray(state.fertilizer_flux_patch, dtype=jnp.float64)
    c13_leaf_transfer = jnp.asarray(
        state.c13_leaf_carbon_transfer_patch,
        dtype=jnp.float64,
    )
    c14_leaf_transfer = jnp.asarray(
        state.c14_leaf_carbon_transfer_patch,
        dtype=jnp.float64,
    )

    onset_flag = jnp.asarray(common_state.onset_flag_patch, dtype=jnp.float64)
    onset_counter = jnp.asarray(
        common_state.onset_counter_seconds_patch,
        dtype=jnp.float64,
    )
    offset_flag = jnp.asarray(common_state.offset_flag_patch, dtype=jnp.float64)
    offset_counter = jnp.asarray(
        common_state.offset_counter_seconds_patch,
        dtype=jnp.float64,
    )
    planting_day_integer = jnp.asarray(state.planting_date_patch, dtype=jnp.int32)
    carbon_transfer = jnp.asarray(
        common_state.carbon_transfer_patch_pool,
        dtype=jnp.float64,
    )
    nitrogen_transfer = jnp.asarray(
        common_state.nitrogen_transfer_patch_pool,
        dtype=jnp.float64,
    )
    leaf_carbon_transfer = carbon_transfer[:, _LEAF_TRANSFER_INDEX]
    leaf_nitrogen_transfer = nitrogen_transfer[:, _LEAF_TRANSFER_INDEX]
    seed_carbon_flux = get_cn_flux_field(cn_fluxes, _CROP_SEED_C_TO_LEAF)
    seed_nitrogen_flux = get_cn_flux_field(cn_fluxes, _CROP_SEED_N_TO_LEAF)

    background_litterfall = jnp.where(
        crop,
        zero,
        phenology_rates.background_litterfall_rate_patch,
    )
    background_transfer = jnp.where(
        crop,
        zero,
        phenology_rates.background_transfer_growth_rate_patch,
    )
    long_growing_season = jnp.where(
        crop,
        zero,
        phenology_rates.long_growing_season_factor_patch,
    )

    annual_reset = crop & (current_day == year_start_day) & (seconds_of_day == 0)
    reset_inactive = annual_reset & (~crop_live)
    reset_winter_live = annual_reset & crop_live & winter
    crop_planted = jnp.where(reset_inactive | reset_winter_live, False, crop_planted)
    planting_day_integer = jnp.where(reset_inactive, not_planted, planting_day_integer)

    eligible = crop & (~crop_live) & (~crop_planted)
    five_day_minimum = jnp.asarray(
        inputs.five_day_minimum_temperature_2m_patch,
        dtype=jnp.float64,
    )
    ten_day_temperature = jnp.asarray(
        inputs.ten_day_temperature_2m_patch,
        dtype=jnp.float64,
    )
    ten_day_minimum = jnp.asarray(
        inputs.ten_day_minimum_temperature_2m_patch,
        dtype=jnp.float64,
    )
    gdd0 = jnp.asarray(inputs.twenty_year_gdd0_patch, dtype=jnp.float64)
    gdd8 = jnp.asarray(inputs.twenty_year_gdd8_patch, dtype=jnp.float64)
    gdd10 = jnp.asarray(inputs.twenty_year_gdd10_patch, dtype=jnp.float64)
    min_plant_temperature = pft_values("minimum_planting_temperature_pft")
    plant_temperature = pft_values("planting_temperature_pft")
    minimum_gdd = pft_values("minimum_growing_degree_days_pft")
    hybrid_gdd = pft_values("hybrid_growing_degree_days_pft")

    winter_normal = (
        eligible
        & winter
        & (five_day_minimum != special_value)
        & (five_day_minimum <= min_plant_temperature)
        & (current_day >= min_planting_day)
        & (gdd0 != special_value)
        & (gdd0 >= minimum_gdd)
    )
    winter_forced = (
        eligible
        & winter
        & (~winter_normal)
        & (current_day >= max_planting_day)
        & (gdd0 != special_value)
        & (gdd0 >= minimum_gdd)
    )
    nonwinter_normal = (
        eligible
        & (~winter)
        & (ten_day_temperature != special_value)
        & (ten_day_minimum != special_value)
        & (ten_day_temperature > plant_temperature)
        & (ten_day_minimum > min_plant_temperature)
        & (current_day >= min_planting_day)
        & (current_day <= max_planting_day)
        & (gdd8 != special_value)
        & (gdd8 >= minimum_gdd)
    )
    nonwinter_forced = (
        eligible
        & (~winter)
        & (~nonwinter_normal)
        & (current_day == max_planting_day)
        & (gdd8 > zero)
        & (gdd8 != special_value)
    )
    normal_planting = winter_normal | nonwinter_normal
    forced_planting = winter_forced | nonwinter_forced
    planting = normal_planting | forced_planting

    corn_first_maturity = jnp.maximum(
        jnp.float64(950.0),
        jnp.minimum(
            fortran_multiply(gdd8, jnp.float64(0.85)),
            hybrid_gdd,
        ),
    )
    corn_normal_maturity = jnp.maximum(
        jnp.float64(950.0),
        jnp.minimum(
            fortran_add(corn_first_maturity, jnp.float64(150.0)),
            jnp.float64(1850.0),
        ),
    )
    nonwinter_maturity = maturity
    nonwinter_maturity = jnp.where(soybean, jnp.minimum(gdd10, hybrid_gdd), nonwinter_maturity)
    nonwinter_maturity = jnp.where(
        corn_or_sugarcane,
        jnp.where(nonwinter_normal, corn_normal_maturity, corn_first_maturity),
        nonwinter_maturity,
    )
    nonwinter_maturity = jnp.where(
        spring_wheat_cotton_or_rice,
        jnp.minimum(gdd0, hybrid_gdd),
        nonwinter_maturity,
    )
    maturity_candidate = jnp.where(winter, hybrid_gdd, nonwinter_maturity)
    maturity = jnp.where(eligible & (~planting), zero, maturity)
    maturity = jnp.where(planting, maturity_candidate, maturity)
    crop_live = jnp.where(planting, True, crop_live)
    crop_planted = jnp.where(planting, True, crop_planted)
    planting_day_integer = jnp.where(planting, current_day, planting_day_integer)
    harvest_date = jnp.where(planting, not_harvested, harvest_date)
    winter_planting = planting & winter
    cumulative_days = jnp.where(winter_planting, zero, cumulative_days)
    hardening_index = jnp.where(winter_planting, zero, hardening_index)
    vernalization_factor = jnp.where(winter_planting, zero, vernalization_factor)

    initial_seed = jnp.asarray(parameters.initial_seed_carbon, dtype=jnp.float64)
    leaf_carbon_transfer = jnp.where(planting, initial_seed, leaf_carbon_transfer)
    leaf_cn = pft_values("leaf_carbon_nitrogen_ratio_pft")
    safe_leaf_cn = jnp.where(crop, leaf_cn, one)
    planted_leaf_nitrogen = fortran_divide(leaf_carbon_transfer, safe_leaf_cn)
    leaf_nitrogen_transfer = jnp.where(
        planting,
        planted_leaf_nitrogen,
        leaf_nitrogen_transfer,
    )
    seed_carbon_flux = jnp.where(
        planting,
        fortran_divide(leaf_carbon_transfer, time_step),
        seed_carbon_flux,
    )
    seed_nitrogen_flux = jnp.where(
        planting,
        fortran_divide(leaf_nitrogen_transfer, time_step),
        seed_nitrogen_flux,
    )
    total_carbon = jnp.asarray(inputs.total_vegetation_carbon_patch, dtype=jnp.float64)
    safe_total_carbon = jnp.where(total_carbon > zero, total_carbon, one)
    c13_seed = jnp.where(
        total_carbon > zero,
        fortran_divide(
            fortran_multiply(
                leaf_carbon_transfer,
                jnp.asarray(inputs.c13_total_vegetation_carbon_patch, dtype=jnp.float64),
            ),
            safe_total_carbon,
        ),
        fortran_multiply(
            leaf_carbon_transfer,
            jnp.asarray(parameters.carbon_13_ratio, dtype=jnp.float64),
        ),
    )
    c14_seed = jnp.where(
        total_carbon > zero,
        fortran_divide(
            fortran_multiply(
                leaf_carbon_transfer,
                jnp.asarray(inputs.c14_total_vegetation_carbon_patch, dtype=jnp.float64),
            ),
            safe_total_carbon,
        ),
        fortran_multiply(
            leaf_carbon_transfer,
            jnp.asarray(parameters.carbon_14_ratio, dtype=jnp.float64),
        ),
    )
    c13_leaf_transfer = jnp.where(
        planting & jnp.asarray(parameters.use_carbon_13, dtype=jnp.bool_),
        c13_seed,
        c13_leaf_transfer,
    )
    c14_leaf_transfer = jnp.where(
        planting & jnp.asarray(parameters.use_carbon_14, dtype=jnp.bool_),
        c14_seed,
        c14_leaf_transfer,
    )

    leaf_fraction = pft_values("leaf_emergence_fraction_pft")
    leaf_heat_units = jnp.where(
        eligible,
        fortran_multiply(leaf_fraction, maturity),
        leaf_heat_units,
    )
    grain_fraction = pft_values("grain_fill_fraction_pft")
    relative_corn_maturity = jnp.maximum(
        jnp.float64(73.0),
        jnp.minimum(
            jnp.float64(135.0),
            fortran_divide(
                fortran_add(maturity, jnp.float64(53.683)),
                jnp.float64(13.882),
            ),
        ),
    )
    corn_grain_fraction = fortran_add(
        fortran_multiply(
            jnp.float64(-0.002),
            _subtract(relative_corn_maturity, jnp.float64(73.0)),
        ),
        grain_fraction,
    )
    corn_grain_fraction = jnp.minimum(
        jnp.maximum(corn_grain_fraction, _subtract(grain_fraction, jnp.float64(0.1))),
        grain_fraction,
    )
    grain_candidate = jnp.where(
        corn_or_sugarcane,
        fortran_multiply(corn_grain_fraction, maturity),
        fortran_multiply(grain_fraction, maturity),
    )
    grain_heat_units = jnp.where(eligible, grain_candidate, grain_heat_units)

    onset_flag = jnp.where(crop, zero, onset_flag)
    offset_flag = jnp.where(crop, zero, offset_flag)
    vernalization_mask = (
        crop
        & crop_live
        & winter
        & (
            jnp.asarray(inputs.minimum_temperature_2m_patch, dtype=jnp.float64)
            < jnp.float64(1.0e30)
        )
        & (vernalization_factor != one)
    )
    vernalization_state = state._replace(
        vernalization_factor_patch=vernalization_factor,
        growing_degree_days_to_maturity_patch=maturity,
        heat_units_to_grain_fill_patch=grain_heat_units,
        cumulative_vernalization_days_patch=cumulative_days,
        hardening_index_patch=hardening_index,
    )
    (
        cumulative_candidate,
        factor_candidate,
        hardening_candidate,
        maturity_candidate,
        grain_candidate,
        cold_diagnostics,
    ) = _apply_vernalization(
        vernalization_mask,
        inputs,
        vernalization_state,
        parameters,
    )
    cumulative_days = jnp.where(vernalization_mask, cumulative_candidate, cumulative_days)
    vernalization_factor = jnp.where(
        vernalization_mask,
        factor_candidate,
        vernalization_factor,
    )
    hardening_index = jnp.where(vernalization_mask, hardening_candidate, hardening_index)
    maturity = jnp.where(vernalization_mask, maturity_candidate, maturity)
    grain_heat_units = jnp.where(vernalization_mask, grain_candidate, grain_heat_units)

    live_branch = crop & crop_live
    crop_phase = jnp.where(live_branch, one, crop_phase)
    integer_days_per_year = jnp.asarray(days_per_year, dtype=jnp.int32)
    days_past_planting = jnp.where(
        current_day >= planting_day_integer,
        current_day - planting_day_integer,
        integer_days_per_year + current_day - planting_day_integer,
    )
    onset_counter = jnp.where(
        live_branch,
        _subtract(onset_counter, time_step),
        onset_counter,
    )
    growing_degree_days = jnp.where(
        live_branch & (jnp.asarray(inputs.peak_leaf_area_index_patch, dtype=jnp.int32) >= 1),
        jnp.maximum(growing_degree_days, grain_heat_units),
        growing_degree_days,
    )
    maximum_days = pft_values("maximum_days_to_maturity_pft")
    leafout = jnp.asarray(inputs.growing_degree_days_topsoil_patch, dtype=jnp.float64)
    phase_two = (
        live_branch
        & (leafout >= leaf_heat_units)
        & (growing_degree_days < grain_heat_units)
        & (days_past_planting < maximum_days)
    )
    crop_phase = jnp.where(phase_two, jnp.float64(2.0), crop_phase)
    first_onset = phase_two & (jnp.abs(onset_counter) > jnp.float64(1.0e-6))
    onset_flag = jnp.where(first_onset, one, onset_flag)
    onset_counter = jnp.where(phase_two, time_step, onset_counter)
    fertilizer_days = jnp.where(
        jnp.asarray(parameters.use_fertilizer, dtype=jnp.bool_),
        jnp.float64(20.0),
        zero,
    )
    new_fertilizer_counter = fortran_multiply(fertilizer_days, seconds_per_day)
    fertilizer_counter = jnp.where(
        first_onset,
        new_fertilizer_counter,
        fertilizer_counter,
    )
    manure = pft_values("manure_nitrogen_pft")
    fertilizer_numerator = fortran_add(
        fortran_multiply(manure, jnp.float64(1000.0)),
        jnp.asarray(inputs.fertilizer_nitrogen_patch, dtype=jnp.float64),
    )
    safe_fertilizer_counter = jnp.where(
        new_fertilizer_counter > zero,
        new_fertilizer_counter,
        one,
    )
    onset_fertilizer_flux = jnp.where(
        fertilizer_days > zero,
        fortran_divide(fertilizer_numerator, safe_fertilizer_counter),
        zero,
    )
    fertilizer_flux = jnp.where(first_onset, onset_fertilizer_flux, fertilizer_flux)

    harvest = (
        live_branch
        & (~phase_two)
        & (
            (growing_degree_days >= maturity)
            | (days_past_planting >= maximum_days)
        )
    )
    harvest_date = jnp.where(
        harvest & (harvest_date >= not_harvested),
        current_day,
        harvest_date,
    )
    crop_live = jnp.where(harvest, False, crop_live)
    crop_phase = jnp.where(harvest, jnp.float64(4.0), crop_phase)
    emerged_harvest = harvest & (
        jnp.asarray(inputs.total_leaf_area_index_patch, dtype=jnp.float64) > zero
    )
    no_emergence_harvest = harvest & (~emerged_harvest)
    offset_flag = jnp.where(emerged_harvest, one, offset_flag)
    offset_counter = jnp.where(emerged_harvest, time_step, offset_counter)
    seed_carbon_flux = jnp.where(
        no_emergence_harvest,
        _subtract(
            seed_carbon_flux,
            fortran_divide(leaf_carbon_transfer, time_step),
        ),
        seed_carbon_flux,
    )
    seed_nitrogen_flux = jnp.where(
        no_emergence_harvest,
        _subtract(
            seed_nitrogen_flux,
            fortran_divide(leaf_nitrogen_transfer, time_step),
        ),
        seed_nitrogen_flux,
    )
    leaf_carbon_transfer = jnp.where(no_emergence_harvest, zero, leaf_carbon_transfer)
    leaf_nitrogen_transfer = jnp.where(
        no_emergence_harvest,
        fortran_divide(zero, safe_leaf_cn),
        leaf_nitrogen_transfer,
    )
    c13_leaf_transfer = jnp.where(
        no_emergence_harvest & jnp.asarray(parameters.use_carbon_13, dtype=jnp.bool_),
        zero,
        c13_leaf_transfer,
    )
    c14_leaf_transfer = jnp.where(
        no_emergence_harvest & jnp.asarray(parameters.use_carbon_14, dtype=jnp.bool_),
        zero,
        c14_leaf_transfer,
    )

    phase_three = (
        live_branch
        & (~phase_two)
        & (~harvest)
        & (growing_degree_days >= grain_heat_units)
    )
    crop_phase = jnp.where(phase_three, jnp.float64(3.0), crop_phase)
    litterfall_denominator = fortran_multiply(
        fortran_multiply(pft_values("leaf_longevity_years_pft"), days_per_year),
        seconds_per_day,
    )
    safe_litterfall_denominator = jnp.where(
        phase_three,
        litterfall_denominator,
        one,
    )
    background_litterfall = jnp.where(
        phase_three,
        fortran_divide(one, safe_litterfall_denominator),
        background_litterfall,
    )

    fertilizer_off = live_branch & (fertilizer_counter <= zero)
    fertilizer_flux = jnp.where(fertilizer_off, zero, fertilizer_flux)
    fertilizer_active = live_branch & (~fertilizer_off)
    fertilizer_counter = jnp.where(
        fertilizer_active,
        _subtract(fertilizer_counter, radiation_step),
        fertilizer_counter,
    )

    inactive_cleanup = crop & (~live_branch)
    seed_carbon_flux = jnp.where(
        inactive_cleanup,
        _subtract(
            seed_carbon_flux,
            fortran_divide(leaf_carbon_transfer, time_step),
        ),
        seed_carbon_flux,
    )
    seed_nitrogen_flux = jnp.where(
        inactive_cleanup,
        _subtract(
            seed_nitrogen_flux,
            fortran_divide(leaf_nitrogen_transfer, time_step),
        ),
        seed_nitrogen_flux,
    )
    onset_counter = jnp.where(inactive_cleanup, zero, onset_counter)
    leaf_carbon_transfer = jnp.where(inactive_cleanup, zero, leaf_carbon_transfer)
    leaf_nitrogen_transfer = jnp.where(
        inactive_cleanup,
        fortran_divide(zero, safe_leaf_cn),
        leaf_nitrogen_transfer,
    )
    c13_leaf_transfer = jnp.where(
        inactive_cleanup & jnp.asarray(parameters.use_carbon_13, dtype=jnp.bool_),
        zero,
        c13_leaf_transfer,
    )
    c14_leaf_transfer = jnp.where(
        inactive_cleanup & jnp.asarray(parameters.use_carbon_14, dtype=jnp.bool_),
        zero,
        c14_leaf_transfer,
    )

    carbon_transfer = carbon_transfer.at[:, _LEAF_TRANSFER_INDEX].set(
        leaf_carbon_transfer,
    )
    nitrogen_transfer = nitrogen_transfer.at[:, _LEAF_TRANSFER_INDEX].set(
        leaf_nitrogen_transfer,
    )
    cn_fluxes = set_cn_flux_field(cn_fluxes, _CROP_SEED_C_TO_LEAF, seed_carbon_flux)
    cn_fluxes = set_cn_flux_field(cn_fluxes, _CROP_SEED_N_TO_LEAF, seed_nitrogen_flux)

    updated_common = common_state._replace(
        onset_flag_patch=onset_flag,
        onset_counter_seconds_patch=onset_counter,
        offset_flag_patch=offset_flag,
        offset_counter_seconds_patch=offset_counter,
        carbon_transfer_patch_pool=carbon_transfer,
        nitrogen_transfer_patch_pool=nitrogen_transfer,
    )
    return CropPhenologyResult(
        state=CropPhenologyState(
            harvest_date_patch=harvest_date,
            planting_date_patch=planting_day_integer,
            crop_live_patch=crop_live,
            crop_planted_patch=crop_planted,
            vernalization_factor_patch=vernalization_factor,
            crop_phase_patch=crop_phase,
            growing_degree_days_since_planting_patch=growing_degree_days,
            growing_degree_days_to_maturity_patch=maturity,
            heat_units_to_leaf_emergence_patch=leaf_heat_units,
            heat_units_to_grain_fill_patch=grain_heat_units,
            cumulative_vernalization_days_patch=cumulative_days,
            hardening_index_patch=hardening_index,
            fertilizer_counter_seconds_patch=fertilizer_counter,
            fertilizer_flux_patch=fertilizer_flux,
            c13_leaf_carbon_transfer_patch=c13_leaf_transfer,
            c14_leaf_carbon_transfer_patch=c14_leaf_transfer,
        ),
        common_phenology_state=updated_common,
        phenology_rates=EvergreenPhenologyState(
            background_litterfall_rate_patch=background_litterfall,
            background_transfer_growth_rate_patch=background_transfer,
            long_growing_season_factor_patch=long_growing_season,
        ),
        cn_fluxes=cn_fluxes,
        diagnostics=CropPhenologyDiagnostics(
            prognostic_crop_patch_count=jnp.sum(crop, dtype=jnp.int32),
            planting_patch_count=jnp.sum(planting, dtype=jnp.int32),
            normal_planting_patch_count=jnp.sum(normal_planting, dtype=jnp.int32),
            forced_planting_patch_count=jnp.sum(forced_planting, dtype=jnp.int32),
            phase_two_patch_count=jnp.sum(phase_two, dtype=jnp.int32),
            phase_three_patch_count=jnp.sum(phase_three, dtype=jnp.int32),
            harvest_patch_count=jnp.sum(harvest, dtype=jnp.int32),
            vernalization_patch_count=jnp.sum(vernalization_mask, dtype=jnp.int32),
            cold_damage_patch_count=jnp.sum(cold_diagnostics[0], dtype=jnp.int32),
            cold_kill_patch_count=jnp.sum(cold_diagnostics[1], dtype=jnp.int32),
            no_emergence_cleanup_patch_count=jnp.sum(
                no_emergence_harvest,
                dtype=jnp.int32,
            ),
            inactive_cleanup_patch_count=jnp.sum(
                inactive_cleanup,
                dtype=jnp.int32,
            ),
        ),
    )


__all__ = [
    "CropPhenologyDiagnostics",
    "CropPhenologyInputs",
    "CropPhenologyParameters",
    "CropPhenologyResult",
    "CropPhenologyState",
    "crop_phenology",
]
