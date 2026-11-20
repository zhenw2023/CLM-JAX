
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import FREEZING_TEMPERATURE_FRESHWATER, SECONDS_PER_DAY
from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    get_cn_flux_field,
    set_cn_flux_field,
)
from clm_jax.physics.evergreen_phenology import (
    CARBON_STORAGE_POOL_NAMES,
    CARBON_STORAGE_TO_TRANSFER_FLUX_FIELDS,
    CARBON_TRANSFER_POOL_NAMES,
    CARBON_TRANSFER_TO_POOL_FLUX_FIELDS,
    NITROGEN_STORAGE_POOL_NAMES,
    NITROGEN_STORAGE_TO_TRANSFER_FLUX_FIELDS,
    NITROGEN_TRANSFER_POOL_NAMES,
    NITROGEN_TRANSFER_TO_POOL_FLUX_FIELDS,
    EvergreenPhenologyState,
)
from clm_jax.physics.numerics import (
    fortran_divide,
    strict_exp,
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


class StressDeciduousPhenologyInputs(NamedTuple):

    soil_patch: jnp.ndarray
    patch_column_index: jnp.ndarray
    patch_gridcell_index: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    daylength_seconds_gridcell: jnp.ndarray
    ten_day_mean_precipitation_patch: jnp.ndarray
    third_soil_layer_temperature_column: jnp.ndarray
    third_soil_layer_water_potential_column: jnp.ndarray
    annual_average_temperature_2m_patch: jnp.ndarray
    leaf_carbon_patch: jnp.ndarray
    fine_root_carbon_patch: jnp.ndarray
    carbon_storage_patch_pool: jnp.ndarray
    nitrogen_storage_patch_pool: jnp.ndarray


class StressDeciduousPhenologyState(NamedTuple):

    onset_freezing_degree_days_patch: jnp.ndarray
    onset_soil_water_index_patch: jnp.ndarray
    offset_freezing_degree_days_patch: jnp.ndarray
    offset_soil_water_index_patch: jnp.ndarray


class StressDeciduousPhenologyParameters(NamedTuple):

    stress_deciduous_pft: jnp.ndarray
    leaf_longevity_years_pft: jnp.ndarray
    woody_pft: jnp.ndarray
    time_step_seconds: float | jnp.ndarray = 1800.0
    days_per_year: float | jnp.ndarray = 365.0
    seconds_per_day: float | jnp.ndarray = SECONDS_PER_DAY
    freezing_temperature: float | jnp.ndarray = (
        FREEZING_TEMPERATURE_FRESHWATER
    )
    critical_onset_freezing_days: float | jnp.ndarray = 15.0
    critical_onset_soil_water_index_days: float | jnp.ndarray = 15.0
    onset_soil_water_potential_mpa: float | jnp.ndarray = -0.6
    critical_offset_freezing_days: float | jnp.ndarray = 15.0
    critical_offset_soil_water_index_days: float | jnp.ndarray = 15.0
    offset_soil_water_potential_mpa: float | jnp.ndarray = -0.8
    onset_period_days: float | jnp.ndarray = 30.0
    offset_period_days: float | jnp.ndarray = 15.0
    storage_transfer_fraction: float | jnp.ndarray = 0.5
    constrain_onset_by_precipitation: bool | jnp.ndarray = True
    onset_rain_threshold_millimeters: float | jnp.ndarray = 20.0
    onset_rain_accumulation_seconds: float | jnp.ndarray = 864_000.0


_REFERENCE_STRESS_DECIDUOUS_PFT = np.zeros(79, dtype=np.float64)
_REFERENCE_STRESS_DECIDUOUS_PFT[[6, 10, 13, 14, 15, 16]] = np.float64(1.0)
_REFERENCE_LEAF_LONGEVITY = np.ones(79, dtype=np.float64)
_REFERENCE_LEAF_LONGEVITY[6] = np.float64(0.4833333333333333)
_REFERENCE_LEAF_LONGEVITY[10] = np.float64(0.39)
_REFERENCE_LEAF_LONGEVITY[13] = np.float64(0.3208333333333333)
_REFERENCE_LEAF_LONGEVITY[14] = np.float64(0.14)
_REFERENCE_LEAF_LONGEVITY[15] = np.float64(0.3208333333333333)
_REFERENCE_WOODY_PFT = np.zeros(79, dtype=np.float64)
_REFERENCE_WOODY_PFT[1:12] = np.float64(1.0)
DEFAULT_STRESS_DECIDUOUS_PHENOLOGY_PARAMETERS = (
    StressDeciduousPhenologyParameters(
        stress_deciduous_pft=_REFERENCE_STRESS_DECIDUOUS_PFT,
        leaf_longevity_years_pft=_REFERENCE_LEAF_LONGEVITY,
        woody_pft=_REFERENCE_WOODY_PFT,
    )
)


class StressDeciduousPhenologyDiagnostics(NamedTuple):
    stress_deciduous_patch_count: jnp.ndarray
    woody_stress_deciduous_patch_count: jnp.ndarray
    offset_cleanup_patch_count: jnp.ndarray
    onset_cleanup_patch_count: jnp.ndarray
    freeze_trigger_patch_count: jnp.ndarray
    precipitation_blocked_patch_count: jnp.ndarray
    onset_trigger_patch_count: jnp.ndarray
    offset_trigger_patch_count: jnp.ndarray


class StressDeciduousPhenologyResult(NamedTuple):
    state: StressDeciduousPhenologyState
    common_phenology_state: SeasonalDeciduousPhenologyState
    phenology_rates: EvergreenPhenologyState
    cn_fluxes: CnTransientFluxState
    diagnostics: StressDeciduousPhenologyDiagnostics


def _validate_shapes(
    inputs: StressDeciduousPhenologyInputs,
    state: StressDeciduousPhenologyState,
    common_state: SeasonalDeciduousPhenologyState,
    phenology_rates: EvergreenPhenologyState,
    cn_fluxes: CnTransientFluxState,
    parameters: StressDeciduousPhenologyParameters,
) -> int:
    if jnp.ndim(inputs.soil_patch) != 1:
        raise ValueError("soil_patch must be a vector")
    patch_count = inputs.soil_patch.shape[0]
    for name in (
        "patch_column_index",
        "patch_gridcell_index",
        "vegetation_type_patch",
        "ten_day_mean_precipitation_patch",
        "annual_average_temperature_2m_patch",
        "leaf_carbon_patch",
        "fine_root_carbon_patch",
    ):
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    if jnp.ndim(inputs.daylength_seconds_gridcell) != 1:
        raise ValueError("daylength_seconds_gridcell must be a vector")
    for name in (
        "third_soil_layer_temperature_column",
        "third_soil_layer_water_potential_column",
    ):
        if jnp.ndim(getattr(inputs, name)) != 1:
            raise ValueError(f"{name} must be a vector")
    expected_pools = {
        "carbon_storage_patch_pool": (
            patch_count,
            len(CARBON_STORAGE_POOL_NAMES),
        ),
        "nitrogen_storage_patch_pool": (
            patch_count,
            len(NITROGEN_STORAGE_POOL_NAMES),
        ),
    }
    for name, expected in expected_pools.items():
        if jnp.shape(getattr(inputs, name)) != expected:
            raise ValueError(f"{name} must have shape {expected}")
    for name in StressDeciduousPhenologyState._fields:
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
    for name in (
        "stress_deciduous_pft",
        "leaf_longevity_years_pft",
        "woody_pft",
    ):
        values = getattr(parameters, name)
        if jnp.ndim(values) != 1 or values.shape[0] < 1:
            raise ValueError(f"{name} must be a nonempty PFT vector")
    pft_count = parameters.stress_deciduous_pft.shape[0]
    if parameters.leaf_longevity_years_pft.shape != (pft_count,):
        raise ValueError("leaf_longevity_years_pft must match stress_deciduous_pft")
    if parameters.woody_pft.shape != (pft_count,):
        raise ValueError("woody_pft must match stress_deciduous_pft")
    for name in StressDeciduousPhenologyParameters._fields[3:]:
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    for field_name in (
        *CARBON_STORAGE_TO_TRANSFER_FLUX_FIELDS,
        *CARBON_TRANSFER_TO_POOL_FLUX_FIELDS,
        *NITROGEN_STORAGE_TO_TRANSFER_FLUX_FIELDS,
        *NITROGEN_TRANSFER_TO_POOL_FLUX_FIELDS,
    ):
        if get_cn_flux_field(cn_fluxes, field_name).shape != (patch_count,):
            raise ValueError("stress phenology flux state must match patch count")
    return patch_count


def _replace_flux_family(
    cn_fluxes: CnTransientFluxState,
    field_names: tuple[str, ...],
    candidates: jnp.ndarray,
    update_mask: jnp.ndarray,
) -> CnTransientFluxState:
    for index, field_name in enumerate(field_names):
        previous = get_cn_flux_field(cn_fluxes, field_name)
        cn_fluxes = set_cn_flux_field(
            cn_fluxes,
            field_name,
            jnp.where(update_mask[:, index], candidates[:, index], previous),
        )
    return cn_fluxes


def stress_deciduous_phenology(
    inputs: StressDeciduousPhenologyInputs,
    state: StressDeciduousPhenologyState,
    common_state: SeasonalDeciduousPhenologyState,
    phenology_rates: EvergreenPhenologyState,
    cn_fluxes: CnTransientFluxState,
    parameters: StressDeciduousPhenologyParameters = (
        DEFAULT_STRESS_DECIDUOUS_PHENOLOGY_PARAMETERS
    ),
) -> StressDeciduousPhenologyResult:

    _validate_shapes(
        inputs,
        state,
        common_state,
        phenology_rates,
        cn_fluxes,
        parameters,
    )
    soil = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    vegetation_type = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    stress_by_pft = jnp.asarray(
        parameters.stress_deciduous_pft,
        dtype=jnp.float64,
    )
    woody_by_pft = jnp.asarray(parameters.woody_pft, dtype=jnp.float64)
    longevity_by_pft = jnp.asarray(
        parameters.leaf_longevity_years_pft,
        dtype=jnp.float64,
    )
    stress = soil & (stress_by_pft[vegetation_type] == jnp.float64(1.0))
    woody = woody_by_pft[vegetation_type] == jnp.float64(1.0)

    column_index = jnp.asarray(inputs.patch_column_index, dtype=jnp.int32)
    gridcell_index = jnp.asarray(inputs.patch_gridcell_index, dtype=jnp.int32)
    daylength = jnp.asarray(
        inputs.daylength_seconds_gridcell,
        dtype=jnp.float64,
    )[gridcell_index]
    soil_temperature = jnp.asarray(
        inputs.third_soil_layer_temperature_column,
        dtype=jnp.float64,
    )[column_index]
    soil_water_potential = jnp.asarray(
        inputs.third_soil_layer_water_potential_column,
        dtype=jnp.float64,
    )[column_index]
    precipitation = jnp.asarray(
        inputs.ten_day_mean_precipitation_patch,
        dtype=jnp.float64,
    )

    time_step = jnp.asarray(parameters.time_step_seconds, dtype=jnp.float64)
    days_per_year = jnp.asarray(parameters.days_per_year, dtype=jnp.float64)
    seconds_per_day = jnp.asarray(parameters.seconds_per_day, dtype=jnp.float64)
    freezing = jnp.asarray(parameters.freezing_temperature, dtype=jnp.float64)
    fraction_of_day = fortran_divide(time_step, seconds_per_day)
    quarter_day_seconds = fortran_divide(seconds_per_day, jnp.float64(4.0))

    dormant = jnp.asarray(common_state.dormant_flag_patch, dtype=jnp.float64)
    days_active = jnp.asarray(common_state.days_active_patch, dtype=jnp.float64)
    onset_flag = jnp.asarray(common_state.onset_flag_patch, dtype=jnp.float64)
    onset_counter = jnp.asarray(
        common_state.onset_counter_seconds_patch,
        dtype=jnp.float64,
    )
    onset_gdd_flag = jnp.asarray(
        common_state.onset_gdd_flag_patch,
        dtype=jnp.float64,
    )
    onset_gdd = jnp.asarray(
        common_state.onset_growing_degree_days_patch,
        dtype=jnp.float64,
    )
    offset_flag = jnp.asarray(common_state.offset_flag_patch, dtype=jnp.float64)
    offset_counter = jnp.asarray(
        common_state.offset_counter_seconds_patch,
        dtype=jnp.float64,
    )
    carbon_transfer = jnp.asarray(
        common_state.carbon_transfer_patch_pool,
        dtype=jnp.float64,
    )
    nitrogen_transfer = jnp.asarray(
        common_state.nitrogen_transfer_patch_pool,
        dtype=jnp.float64,
    )
    previous_leaf_litter = jnp.asarray(
        common_state.previous_leaf_carbon_to_litter_patch,
        dtype=jnp.float64,
    )
    previous_fine_root_litter = jnp.asarray(
        common_state.previous_fine_root_carbon_to_litter_patch,
        dtype=jnp.float64,
    )
    pft_may_exist = jnp.asarray(common_state.pft_may_exist_patch, dtype=jnp.bool_)
    onset_fdd = jnp.asarray(
        state.onset_freezing_degree_days_patch,
        dtype=jnp.float64,
    )
    onset_swi = jnp.asarray(
        state.onset_soil_water_index_patch,
        dtype=jnp.float64,
    )
    offset_fdd = jnp.asarray(
        state.offset_freezing_degree_days_patch,
        dtype=jnp.float64,
    )
    offset_swi = jnp.asarray(
        state.offset_soil_water_index_patch,
        dtype=jnp.float64,
    )

    offset_in_progress = stress & (offset_flag == jnp.float64(1.0))
    offset_counter = jnp.where(
        offset_in_progress,
        fortran_add(offset_counter, -time_step),
        offset_counter,
    )
    offset_cleanup = offset_in_progress & (offset_counter == jnp.float64(0.0))
    offset_flag = jnp.where(offset_cleanup, jnp.float64(0.0), offset_flag)
    offset_counter = jnp.where(offset_cleanup, jnp.float64(0.0), offset_counter)
    dormant = jnp.where(offset_cleanup, jnp.float64(1.0), dormant)
    days_active = jnp.where(offset_cleanup, jnp.float64(0.0), days_active)
    previous_leaf_litter = jnp.where(
        offset_cleanup,
        jnp.float64(0.0),
        previous_leaf_litter,
    )
    previous_fine_root_litter = jnp.where(
        offset_cleanup,
        jnp.float64(0.0),
        previous_fine_root_litter,
    )

    onset_in_progress = stress & (onset_flag == jnp.float64(1.0))
    onset_counter = jnp.where(
        onset_in_progress,
        fortran_add(onset_counter, -time_step),
        onset_counter,
    )
    onset_cleanup = onset_in_progress & (onset_counter == jnp.float64(0.0))
    onset_flag = jnp.where(onset_cleanup, jnp.float64(0.0), onset_flag)
    onset_counter = jnp.where(onset_cleanup, jnp.float64(0.0), onset_counter)
    transfer_cleanup_mask = jnp.stack(
        (
            onset_cleanup,
            onset_cleanup,
            *(onset_cleanup & woody for _ in range(4)),
        ),
        axis=1,
    )
    carbon_transfer = jnp.where(
        transfer_cleanup_mask,
        jnp.float64(0.0),
        carbon_transfer,
    )
    nitrogen_transfer = jnp.where(
        transfer_cleanup_mask,
        jnp.float64(0.0),
        nitrogen_transfer,
    )
    updated_fluxes = _replace_flux_family(
        cn_fluxes,
        CARBON_TRANSFER_TO_POOL_FLUX_FIELDS,
        jnp.zeros_like(carbon_transfer),
        transfer_cleanup_mask,
    )
    updated_fluxes = _replace_flux_family(
        updated_fluxes,
        NITROGEN_TRANSFER_TO_POOL_FLUX_FIELDS,
        jnp.zeros_like(nitrogen_transfer),
        transfer_cleanup_mask,
    )

    safe_annual_temperature = jnp.where(
        stress,
        jnp.asarray(inputs.annual_average_temperature_2m_patch, dtype=jnp.float64),
        freezing,
    )
    critical_gdd_argument = fortran_add(
        jnp.float64(4.8),
        fortran_multiply(
            jnp.float64(0.13),
            fortran_add(safe_annual_temperature, -freezing),
        ),
    )
    critical_onset_gdd = strict_exp(critical_gdd_argument)

    dormant_branch = stress & (dormant == jnp.float64(1.0))
    accumulate_onset_fdd = (
        dormant_branch
        & (onset_gdd_flag == jnp.float64(0.0))
        & (soil_temperature < freezing)
    )
    onset_fdd = jnp.where(
        accumulate_onset_fdd,
        fortran_add(onset_fdd, fraction_of_day),
        onset_fdd,
    )
    freeze_trigger = dormant_branch & (
        onset_fdd
        > jnp.asarray(parameters.critical_onset_freezing_days, dtype=jnp.float64)
    )
    onset_gdd_flag = jnp.where(
        freeze_trigger,
        jnp.float64(1.0),
        onset_gdd_flag,
    )
    onset_fdd = jnp.where(freeze_trigger, jnp.float64(0.0), onset_fdd)
    onset_swi = jnp.where(freeze_trigger, jnp.float64(0.0), onset_swi)

    accumulate_onset_gdd = (
        dormant_branch
        & (onset_gdd_flag == jnp.float64(1.0))
        & (soil_temperature > freezing)
    )
    onset_gdd_increment = fortran_multiply(
        fortran_add(soil_temperature, -freezing),
        fraction_of_day,
    )
    onset_gdd = jnp.where(
        accumulate_onset_gdd,
        fortran_add(onset_gdd, onset_gdd_increment),
        onset_gdd,
    )

    constrain_rain = jnp.asarray(
        parameters.constrain_onset_by_precipitation,
        dtype=jnp.bool_,
    )
    ten_day_accumulation = fortran_multiply(
        precipitation,
        jnp.asarray(
            parameters.onset_rain_accumulation_seconds,
            dtype=jnp.float64,
        ),
    )
    precipitation_blocked = (
        dormant_branch
        & constrain_rain
        & (
            ten_day_accumulation
            < jnp.asarray(
                parameters.onset_rain_threshold_millimeters,
                dtype=jnp.float64,
            )
        )
    )
    wet_for_onset = dormant_branch & (
        soil_water_potential
        >= jnp.asarray(parameters.onset_soil_water_potential_mpa, dtype=jnp.float64)
    )
    onset_swi = jnp.where(
        wet_for_onset,
        fortran_add(onset_swi, fraction_of_day),
        onset_swi,
    )
    water_trigger = (
        dormant_branch
        & (
            onset_swi
            > jnp.asarray(
                parameters.critical_onset_soil_water_index_days,
                dtype=jnp.float64,
            )
        )
        & ~precipitation_blocked
    )
    onset_flag = jnp.where(water_trigger, jnp.float64(1.0), onset_flag)
    temperature_override = (
        water_trigger
        & (onset_gdd_flag == jnp.float64(1.0))
        & (onset_gdd < critical_onset_gdd)
    )
    onset_flag = jnp.where(
        temperature_override,
        jnp.float64(0.0),
        onset_flag,
    )
    short_day_onset_block = (
        dormant_branch
        & (onset_flag == jnp.float64(1.0))
        & (daylength <= quarter_day_seconds)
    )
    onset_flag = jnp.where(
        short_day_onset_block,
        jnp.float64(0.0),
        onset_flag,
    )
    onset_trigger = dormant_branch & (onset_flag == jnp.float64(1.0))
    dormant = jnp.where(onset_trigger, jnp.float64(0.0), dormant)
    days_active = jnp.where(onset_trigger, jnp.float64(0.0), days_active)
    onset_gdd_flag = jnp.where(onset_trigger, jnp.float64(0.0), onset_gdd_flag)
    onset_fdd = jnp.where(onset_trigger, jnp.float64(0.0), onset_fdd)
    onset_gdd = jnp.where(onset_trigger, jnp.float64(0.0), onset_gdd)
    onset_swi = jnp.where(onset_trigger, jnp.float64(0.0), onset_swi)
    onset_period_seconds = fortran_multiply(
        jnp.asarray(parameters.onset_period_days, dtype=jnp.float64),
        seconds_per_day,
    )
    onset_counter = jnp.where(onset_trigger, onset_period_seconds, onset_counter)

    onset_carbon_mask = jnp.stack(
        (
            onset_trigger,
            onset_trigger,
            *(onset_trigger & woody for _ in range(5)),
        ),
        axis=1,
    )
    onset_nitrogen_mask = jnp.stack(
        (
            onset_trigger,
            onset_trigger,
            *(onset_trigger & woody for _ in range(4)),
        ),
        axis=1,
    )
    storage_fraction = jnp.asarray(
        parameters.storage_transfer_fraction,
        dtype=jnp.float64,
    )
    carbon_storage = jnp.asarray(
        inputs.carbon_storage_patch_pool,
        dtype=jnp.float64,
    )
    nitrogen_storage = jnp.asarray(
        inputs.nitrogen_storage_patch_pool,
        dtype=jnp.float64,
    )
    onset_carbon_rate = fortran_divide(
        fortran_multiply(
            storage_fraction,
            jnp.where(onset_carbon_mask, carbon_storage, jnp.float64(0.0)),
        ),
        time_step,
    )
    onset_nitrogen_rate = fortran_divide(
        fortran_multiply(
            storage_fraction,
            jnp.where(onset_nitrogen_mask, nitrogen_storage, jnp.float64(0.0)),
        ),
        time_step,
    )
    updated_fluxes = _replace_flux_family(
        updated_fluxes,
        CARBON_STORAGE_TO_TRANSFER_FLUX_FIELDS,
        onset_carbon_rate,
        onset_carbon_mask,
    )
    updated_fluxes = _replace_flux_family(
        updated_fluxes,
        NITROGEN_STORAGE_TO_TRANSFER_FLUX_FIELDS,
        onset_nitrogen_rate,
        onset_nitrogen_mask,
    )

    active_branch = stress & ~dormant_branch & (offset_flag == jnp.float64(0.0))
    dry_for_offset = active_branch & (
        soil_water_potential
        <= jnp.asarray(parameters.offset_soil_water_potential_mpa, dtype=jnp.float64)
    )
    offset_swi = jnp.where(
        dry_for_offset,
        fortran_add(offset_swi, fraction_of_day),
        offset_swi,
    )
    water_offset_trigger = (
        dry_for_offset
        & (
            offset_swi
            >= jnp.asarray(
                parameters.critical_offset_soil_water_index_days,
                dtype=jnp.float64,
            )
        )
        & (onset_flag == jnp.float64(0.0))
    )
    offset_flag = jnp.where(
        water_offset_trigger,
        jnp.float64(1.0),
        offset_flag,
    )
    wet_offset_recovery = (
        active_branch
        & ~dry_for_offset
        & (
            soil_water_potential
            >= jnp.asarray(
                parameters.onset_soil_water_potential_mpa,
                dtype=jnp.float64,
            )
        )
    )
    offset_swi_recovered = jnp.maximum(
        fortran_add(offset_swi, -fraction_of_day),
        jnp.float64(0.0),
    )
    offset_swi = jnp.where(
        wet_offset_recovery,
        offset_swi_recovered,
        offset_swi,
    )

    warm_offset_recovery = (
        active_branch
        & (offset_fdd > jnp.float64(0.0))
        & (soil_temperature > freezing)
    )
    offset_fdd_recovered = jnp.maximum(
        jnp.float64(0.0),
        fortran_add(offset_fdd, -fraction_of_day),
    )
    offset_fdd = jnp.where(
        warm_offset_recovery,
        offset_fdd_recovered,
        offset_fdd,
    )
    cold_for_offset = active_branch & (soil_temperature <= freezing)
    offset_fdd = jnp.where(
        cold_for_offset,
        fortran_add(offset_fdd, fraction_of_day),
        offset_fdd,
    )
    freeze_offset_trigger = (
        cold_for_offset
        & (
            offset_fdd
            > jnp.asarray(
                parameters.critical_offset_freezing_days,
                dtype=jnp.float64,
            )
        )
        & (onset_flag == jnp.float64(0.0))
    )
    offset_flag = jnp.where(
        freeze_offset_trigger,
        jnp.float64(1.0),
        offset_flag,
    )
    short_day_offset_trigger = active_branch & (daylength <= quarter_day_seconds)
    offset_flag = jnp.where(
        short_day_offset_trigger,
        jnp.float64(1.0),
        offset_flag,
    )
    offset_trigger = active_branch & (offset_flag == jnp.float64(1.0))
    offset_fdd = jnp.where(offset_trigger, jnp.float64(0.0), offset_fdd)
    offset_swi = jnp.where(offset_trigger, jnp.float64(0.0), offset_swi)
    offset_period_seconds = fortran_multiply(
        jnp.asarray(parameters.offset_period_days, dtype=jnp.float64),
        seconds_per_day,
    )
    offset_counter = jnp.where(
        offset_trigger,
        offset_period_seconds,
        offset_counter,
    )
    previous_leaf_litter = jnp.where(
        offset_trigger,
        jnp.float64(0.0),
        previous_leaf_litter,
    )
    previous_fine_root_litter = jnp.where(
        offset_trigger,
        jnp.float64(0.0),
        previous_fine_root_litter,
    )

    active_after_branches = stress & (dormant == jnp.float64(0.0))
    days_active = jnp.where(
        active_after_branches,
        fortran_add(days_active, fraction_of_day),
        days_active,
    )
    selected_longevity = jnp.where(
        stress,
        longevity_by_pft[vegetation_type],
        jnp.float64(1.0),
    )
    longevity_days = fortran_multiply(selected_longevity, days_per_year)
    active_beyond_longevity = fortran_add(
        jnp.where(stress, days_active, jnp.float64(0.0)), -longevity_days,
    )
    lgsf_candidate = fortran_divide(
        fortran_multiply(jnp.float64(3.0), active_beyond_longevity),
        days_per_year,
    )
    lgsf_candidate = jnp.maximum(
        jnp.minimum(lgsf_candidate, jnp.float64(1.0)),
        jnp.float64(0.0),
    )
    previous_lgsf = jnp.asarray(
        phenology_rates.long_growing_season_factor_patch,
        dtype=jnp.float64,
    )
    lgsf = jnp.where(stress, lgsf_candidate, previous_lgsf)

    longevity_seconds = fortran_multiply(longevity_days, seconds_per_day)
    background_litterfall_candidate = fortran_multiply(
        fortran_divide(jnp.float64(1.0), longevity_seconds),
        lgsf_candidate,
    )
    background_litterfall_candidate = jnp.where(
        offset_flag == jnp.float64(1.0),
        jnp.float64(0.0),
        background_litterfall_candidate,
    )
    previous_litterfall_rate = jnp.asarray(
        phenology_rates.background_litterfall_rate_patch,
        dtype=jnp.float64,
    )
    background_litterfall = jnp.where(
        stress,
        background_litterfall_candidate,
        previous_litterfall_rate,
    )

    year_seconds = fortran_multiply(days_per_year, seconds_per_day)
    background_transfer_candidate = fortran_multiply(
        fortran_divide(jnp.float64(1.0), year_seconds),
        lgsf_candidate,
    )
    background_transfer_candidate = jnp.where(
        onset_flag == jnp.float64(1.0),
        jnp.float64(0.0),
        background_transfer_candidate,
    )
    previous_transfer_rate = jnp.asarray(
        phenology_rates.background_transfer_growth_rate_patch,
        dtype=jnp.float64,
    )
    background_transfer = jnp.where(
        stress,
        background_transfer_candidate,
        previous_transfer_rate,
    )
    updated_rates = EvergreenPhenologyState(
        background_litterfall_rate_patch=background_litterfall,
        background_transfer_growth_rate_patch=background_transfer,
        long_growing_season_factor_patch=lgsf,
    )

    background_flux_branch = stress & (onset_flag != jnp.float64(1.0))
    background_woody = background_flux_branch & woody
    background_carbon_mask = jnp.stack(
        (
            background_flux_branch,
            background_flux_branch,
            *(background_woody for _ in range(5)),
        ),
        axis=1,
    )
    background_nitrogen_mask = jnp.stack(
        (
            background_flux_branch,
            background_flux_branch,
            *(background_woody for _ in range(4)),
        ),
        axis=1,
    )
    leaf_carbon = jnp.asarray(inputs.leaf_carbon_patch, dtype=jnp.float64)
    fine_root_carbon = jnp.asarray(
        inputs.fine_root_carbon_patch,
        dtype=jnp.float64,
    )
    carbon_background_source = carbon_storage
    carbon_background_source = carbon_background_source.at[:, 0].set(
        jnp.maximum(
            jnp.float64(0.0),
            fortran_add(carbon_storage[:, 0], -leaf_carbon),
        )
    )
    carbon_background_source = carbon_background_source.at[:, 1].set(
        jnp.maximum(
            jnp.float64(0.0),
            fortran_add(carbon_storage[:, 1], -fine_root_carbon),
        )
    )
    background_carbon_rate = fortran_multiply(
        jnp.where(
            background_carbon_mask,
            carbon_background_source,
            jnp.float64(0.0),
        ),
        background_transfer_candidate[:, None],
    )
    background_nitrogen_rate = fortran_multiply(
        jnp.where(
            background_nitrogen_mask,
            nitrogen_storage,
            jnp.float64(0.0),
        ),
        background_transfer_candidate[:, None],
    )
    updated_fluxes = _replace_flux_family(
        updated_fluxes,
        CARBON_STORAGE_TO_TRANSFER_FLUX_FIELDS,
        background_carbon_rate,
        background_carbon_mask,
    )
    updated_fluxes = _replace_flux_family(
        updated_fluxes,
        NITROGEN_STORAGE_TO_TRANSFER_FLUX_FIELDS,
        background_nitrogen_rate,
        background_nitrogen_mask,
    )

    return StressDeciduousPhenologyResult(
        state=StressDeciduousPhenologyState(
            onset_freezing_degree_days_patch=onset_fdd,
            onset_soil_water_index_patch=onset_swi,
            offset_freezing_degree_days_patch=offset_fdd,
            offset_soil_water_index_patch=offset_swi,
        ),
        common_phenology_state=SeasonalDeciduousPhenologyState(
            dormant_flag_patch=dormant,
            days_active_patch=days_active,
            onset_flag_patch=onset_flag,
            onset_counter_seconds_patch=onset_counter,
            onset_gdd_flag_patch=onset_gdd_flag,
            onset_growing_degree_days_patch=onset_gdd,
            offset_flag_patch=offset_flag,
            offset_counter_seconds_patch=offset_counter,
            carbon_transfer_patch_pool=carbon_transfer,
            nitrogen_transfer_patch_pool=nitrogen_transfer,
            previous_leaf_carbon_to_litter_patch=previous_leaf_litter,
            previous_fine_root_carbon_to_litter_patch=(
                previous_fine_root_litter
            ),
            pft_may_exist_patch=pft_may_exist,
        ),
        phenology_rates=updated_rates,
        cn_fluxes=updated_fluxes,
        diagnostics=StressDeciduousPhenologyDiagnostics(
            stress_deciduous_patch_count=jnp.sum(stress, dtype=jnp.int32),
            woody_stress_deciduous_patch_count=jnp.sum(
                stress & woody,
                dtype=jnp.int32,
            ),
            offset_cleanup_patch_count=jnp.sum(offset_cleanup, dtype=jnp.int32),
            onset_cleanup_patch_count=jnp.sum(onset_cleanup, dtype=jnp.int32),
            freeze_trigger_patch_count=jnp.sum(freeze_trigger, dtype=jnp.int32),
            precipitation_blocked_patch_count=jnp.sum(
                precipitation_blocked,
                dtype=jnp.int32,
            ),
            onset_trigger_patch_count=jnp.sum(onset_trigger, dtype=jnp.int32),
            offset_trigger_patch_count=jnp.sum(offset_trigger, dtype=jnp.int32),
        ),
    )


__all__ = [
    "DEFAULT_STRESS_DECIDUOUS_PHENOLOGY_PARAMETERS",
    "StressDeciduousPhenologyDiagnostics",
    "StressDeciduousPhenologyInputs",
    "StressDeciduousPhenologyParameters",
    "StressDeciduousPhenologyResult",
    "StressDeciduousPhenologyState",
    "stress_deciduous_phenology",
]
