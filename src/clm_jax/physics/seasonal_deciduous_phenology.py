
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


class SeasonalDeciduousPhenologyInputs(NamedTuple):

    soil_patch: jnp.ndarray
    patch_column_index: jnp.ndarray
    patch_gridcell_index: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    daylength_seconds_gridcell: jnp.ndarray
    previous_daylength_seconds_gridcell: jnp.ndarray
    third_soil_layer_temperature_column: jnp.ndarray
    annual_average_temperature_2m_patch: jnp.ndarray
    carbon_storage_patch_pool: jnp.ndarray
    nitrogen_storage_patch_pool: jnp.ndarray


class SeasonalDeciduousPhenologyState(NamedTuple):

    dormant_flag_patch: jnp.ndarray
    days_active_patch: jnp.ndarray
    onset_flag_patch: jnp.ndarray
    onset_counter_seconds_patch: jnp.ndarray
    onset_gdd_flag_patch: jnp.ndarray
    onset_growing_degree_days_patch: jnp.ndarray
    offset_flag_patch: jnp.ndarray
    offset_counter_seconds_patch: jnp.ndarray
    carbon_transfer_patch_pool: jnp.ndarray
    nitrogen_transfer_patch_pool: jnp.ndarray
    previous_leaf_carbon_to_litter_patch: jnp.ndarray
    previous_fine_root_carbon_to_litter_patch: jnp.ndarray
    pft_may_exist_patch: jnp.ndarray


class SeasonalDeciduousPhenologyParameters(NamedTuple):

    seasonal_deciduous_pft: jnp.ndarray
    woody_pft: jnp.ndarray
    time_step_seconds: float | jnp.ndarray = 1800.0
    seconds_per_day: float | jnp.ndarray = SECONDS_PER_DAY
    freezing_temperature: float | jnp.ndarray = (
        FREEZING_TEMPERATURE_FRESHWATER
    )
    critical_daylength_seconds: float | jnp.ndarray = 39_300.0
    onset_period_days: float | jnp.ndarray = 30.0
    offset_period_days: float | jnp.ndarray = 15.0
    storage_transfer_fraction: float | jnp.ndarray = 0.5
    use_dynamic_vegetation: bool | jnp.ndarray = False


_REFERENCE_SEASONAL_DECIDUOUS_PFT = np.zeros(79, dtype=np.float64)
_REFERENCE_SEASONAL_DECIDUOUS_PFT[[3, 7, 8, 11, 12]] = np.float64(1.0)
_REFERENCE_WOODY_PFT = np.zeros(79, dtype=np.float64)
_REFERENCE_WOODY_PFT[1:12] = np.float64(1.0)
DEFAULT_SEASONAL_DECIDUOUS_PHENOLOGY_PARAMETERS = (
    SeasonalDeciduousPhenologyParameters(
        seasonal_deciduous_pft=_REFERENCE_SEASONAL_DECIDUOUS_PFT,
        woody_pft=_REFERENCE_WOODY_PFT,
    )
)


class SeasonalDeciduousPhenologyDiagnostics(NamedTuple):
    seasonal_deciduous_patch_count: jnp.ndarray
    offset_cleanup_patch_count: jnp.ndarray
    onset_cleanup_patch_count: jnp.ndarray
    onset_trigger_patch_count: jnp.ndarray
    offset_trigger_patch_count: jnp.ndarray


class SeasonalDeciduousPhenologyResult(NamedTuple):
    state: SeasonalDeciduousPhenologyState
    phenology_rates: EvergreenPhenologyState
    cn_fluxes: CnTransientFluxState
    diagnostics: SeasonalDeciduousPhenologyDiagnostics


def _validate_shapes(
    inputs: SeasonalDeciduousPhenologyInputs,
    state: SeasonalDeciduousPhenologyState,
    phenology_rates: EvergreenPhenologyState,
    cn_fluxes: CnTransientFluxState,
    parameters: SeasonalDeciduousPhenologyParameters,
) -> int:
    if jnp.ndim(inputs.soil_patch) != 1:
        raise ValueError("soil_patch must be a vector")
    patch_count = inputs.soil_patch.shape[0]
    for name in (
        "patch_column_index",
        "patch_gridcell_index",
        "vegetation_type_patch",
        "annual_average_temperature_2m_patch",
    ):
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    if jnp.ndim(inputs.daylength_seconds_gridcell) != 1:
        raise ValueError("current daylength must be a gridcell vector")
    gridcell_count = inputs.daylength_seconds_gridcell.shape[0]
    if jnp.shape(inputs.previous_daylength_seconds_gridcell) != (gridcell_count,):
        raise ValueError("current and previous daylength must share one gridcell axis")
    if jnp.ndim(inputs.third_soil_layer_temperature_column) != 1:
        raise ValueError("third_soil_layer_temperature_column must be a vector")
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
    for name in SeasonalDeciduousPhenologyState._fields:
        expected = (patch_count,)
        if name == "carbon_transfer_patch_pool":
            expected = (patch_count, len(CARBON_TRANSFER_POOL_NAMES))
        elif name == "nitrogen_transfer_patch_pool":
            expected = (patch_count, len(NITROGEN_TRANSFER_POOL_NAMES))
        if jnp.shape(getattr(state, name)) != expected:
            raise ValueError(f"{name} must have shape {expected}")
    for name in EvergreenPhenologyState._fields:
        if jnp.shape(getattr(phenology_rates, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    for name in ("seasonal_deciduous_pft", "woody_pft"):
        values = getattr(parameters, name)
        if jnp.ndim(values) != 1 or values.shape[0] < 1:
            raise ValueError(f"{name} must be a nonempty PFT vector")
    if parameters.woody_pft.shape != parameters.seasonal_deciduous_pft.shape:
        raise ValueError("seasonal_deciduous_pft and woody_pft must have equal length")
    for name in SeasonalDeciduousPhenologyParameters._fields[2:]:
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    for field_name in (
        *CARBON_STORAGE_TO_TRANSFER_FLUX_FIELDS,
        *CARBON_TRANSFER_TO_POOL_FLUX_FIELDS,
        *NITROGEN_STORAGE_TO_TRANSFER_FLUX_FIELDS,
        *NITROGEN_TRANSFER_TO_POOL_FLUX_FIELDS,
    ):
        if get_cn_flux_field(cn_fluxes, field_name).shape != (patch_count,):
            raise ValueError("seasonal phenology flux state must match patch count")
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


def seasonal_deciduous_phenology(
    inputs: SeasonalDeciduousPhenologyInputs,
    state: SeasonalDeciduousPhenologyState,
    phenology_rates: EvergreenPhenologyState,
    cn_fluxes: CnTransientFluxState,
    parameters: SeasonalDeciduousPhenologyParameters = (
        DEFAULT_SEASONAL_DECIDUOUS_PHENOLOGY_PARAMETERS
    ),
) -> SeasonalDeciduousPhenologyResult:

    _validate_shapes(inputs, state, phenology_rates, cn_fluxes, parameters)
    soil = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    vegetation_type = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    seasonal_by_pft = jnp.asarray(
        parameters.seasonal_deciduous_pft,
        dtype=jnp.float64,
    )
    woody_by_pft = jnp.asarray(parameters.woody_pft, dtype=jnp.float64)
    seasonal = soil & (seasonal_by_pft[vegetation_type] == jnp.float64(1.0))
    woody = woody_by_pft[vegetation_type] == jnp.float64(1.0)

    column_index = jnp.asarray(inputs.patch_column_index, dtype=jnp.int32)
    gridcell_index = jnp.asarray(inputs.patch_gridcell_index, dtype=jnp.int32)
    daylength = jnp.asarray(
        inputs.daylength_seconds_gridcell,
        dtype=jnp.float64,
    )[gridcell_index]
    previous_daylength = jnp.asarray(
        inputs.previous_daylength_seconds_gridcell,
        dtype=jnp.float64,
    )[gridcell_index]
    winter_to_summer = daylength >= previous_daylength
    soil_temperature = jnp.asarray(
        inputs.third_soil_layer_temperature_column,
        dtype=jnp.float64,
    )[column_index]

    time_step = jnp.asarray(parameters.time_step_seconds, dtype=jnp.float64)
    seconds_per_day = jnp.asarray(parameters.seconds_per_day, dtype=jnp.float64)
    freezing = jnp.asarray(parameters.freezing_temperature, dtype=jnp.float64)
    fraction_of_day = fortran_divide(time_step, seconds_per_day)
    critical_daylength = jnp.asarray(
        parameters.critical_daylength_seconds,
        dtype=jnp.float64,
    )
    use_dynamic_vegetation = jnp.asarray(
        parameters.use_dynamic_vegetation,
        dtype=jnp.bool_,
    )

    updated_rates = EvergreenPhenologyState(
        background_litterfall_rate_patch=jnp.where(
            seasonal,
            jnp.float64(0.0),
            jnp.asarray(
                phenology_rates.background_litterfall_rate_patch,
                dtype=jnp.float64,
            ),
        ),
        background_transfer_growth_rate_patch=jnp.where(
            seasonal,
            jnp.float64(0.0),
            jnp.asarray(
                phenology_rates.background_transfer_growth_rate_patch,
                dtype=jnp.float64,
            ),
        ),
        long_growing_season_factor_patch=jnp.where(
            seasonal,
            jnp.float64(0.0),
            jnp.asarray(
                phenology_rates.long_growing_season_factor_patch,
                dtype=jnp.float64,
            ),
        ),
    )

    dormant = jnp.asarray(state.dormant_flag_patch, dtype=jnp.float64)
    days_active = jnp.asarray(state.days_active_patch, dtype=jnp.float64)
    onset_flag = jnp.asarray(state.onset_flag_patch, dtype=jnp.float64)
    onset_counter = jnp.asarray(
        state.onset_counter_seconds_patch,
        dtype=jnp.float64,
    )
    onset_gdd_flag = jnp.asarray(state.onset_gdd_flag_patch, dtype=jnp.float64)
    onset_gdd = jnp.asarray(
        state.onset_growing_degree_days_patch,
        dtype=jnp.float64,
    )
    offset_flag = jnp.asarray(state.offset_flag_patch, dtype=jnp.float64)
    offset_counter = jnp.asarray(
        state.offset_counter_seconds_patch,
        dtype=jnp.float64,
    )
    carbon_transfer = jnp.asarray(
        state.carbon_transfer_patch_pool,
        dtype=jnp.float64,
    )
    nitrogen_transfer = jnp.asarray(
        state.nitrogen_transfer_patch_pool,
        dtype=jnp.float64,
    )
    previous_leaf_litter = jnp.asarray(
        state.previous_leaf_carbon_to_litter_patch,
        dtype=jnp.float64,
    )
    previous_fine_root_litter = jnp.asarray(
        state.previous_fine_root_carbon_to_litter_patch,
        dtype=jnp.float64,
    )
    pft_may_exist = jnp.asarray(state.pft_may_exist_patch, dtype=jnp.bool_)

    offset_in_progress = seasonal & (offset_flag == jnp.float64(1.0))
    offset_counter = jnp.where(
        offset_in_progress,
        fortran_add(offset_counter, -time_step),
        offset_counter,
    )
    offset_cleanup = offset_in_progress & (offset_counter == jnp.float64(0.0))
    offset_flag = jnp.where(offset_cleanup, jnp.float64(0.0), offset_flag)
    offset_counter = jnp.where(
        offset_cleanup,
        jnp.float64(0.0),
        offset_counter,
    )
    dormant = jnp.where(offset_cleanup, jnp.float64(1.0), dormant)
    days_active = jnp.where(offset_cleanup, jnp.float64(0.0), days_active)
    pft_may_exist = jnp.where(
        offset_cleanup & use_dynamic_vegetation,
        jnp.bool_(True),
        pft_may_exist,
    )
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

    onset_in_progress = seasonal & (onset_flag == jnp.float64(1.0))
    onset_counter = jnp.where(
        onset_in_progress,
        fortran_add(onset_counter, -time_step),
        onset_counter,
    )
    onset_cleanup = onset_in_progress & (onset_counter == jnp.float64(0.0))
    onset_flag = jnp.where(onset_cleanup, jnp.float64(0.0), onset_flag)
    onset_counter = jnp.where(
        onset_cleanup,
        jnp.float64(0.0),
        onset_counter,
    )
    ordinary_cleanup_pool_mask = jnp.stack(
        (
            onset_cleanup,
            onset_cleanup,
            *(onset_cleanup & woody for _ in range(4)),
        ),
        axis=1,
    )
    carbon_transfer = jnp.where(
        ordinary_cleanup_pool_mask,
        jnp.float64(0.0),
        carbon_transfer,
    )
    nitrogen_transfer = jnp.where(
        ordinary_cleanup_pool_mask,
        jnp.float64(0.0),
        nitrogen_transfer,
    )
    zero_transfer_rates = jnp.zeros_like(carbon_transfer)
    updated_fluxes = _replace_flux_family(
        cn_fluxes,
        CARBON_TRANSFER_TO_POOL_FLUX_FIELDS,
        zero_transfer_rates,
        ordinary_cleanup_pool_mask,
    )
    updated_fluxes = _replace_flux_family(
        updated_fluxes,
        NITROGEN_TRANSFER_TO_POOL_FLUX_FIELDS,
        jnp.zeros_like(nitrogen_transfer),
        ordinary_cleanup_pool_mask,
    )

    dormant_branch = seasonal & (dormant == jnp.float64(1.0))
    begin_gdd = (
        dormant_branch
        & (onset_gdd_flag == jnp.float64(0.0))
        & winter_to_summer
    )
    onset_gdd_flag = jnp.where(begin_gdd, jnp.float64(1.0), onset_gdd_flag)
    onset_gdd = jnp.where(begin_gdd, jnp.float64(0.0), onset_gdd)
    reset_gdd = (
        dormant_branch
        & (onset_gdd_flag == jnp.float64(1.0))
        & ~winter_to_summer
    )
    onset_gdd_flag = jnp.where(reset_gdd, jnp.float64(0.0), onset_gdd_flag)
    onset_gdd = jnp.where(reset_gdd, jnp.float64(0.0), onset_gdd)
    accumulate_gdd = (
        dormant_branch
        & (onset_gdd_flag == jnp.float64(1.0))
        & (soil_temperature > freezing)
    )
    gdd_increment = fortran_multiply(
        fortran_add(soil_temperature, -freezing),
        fraction_of_day,
    )
    onset_gdd = jnp.where(
        accumulate_gdd,
        fortran_add(onset_gdd, gdd_increment),
        onset_gdd,
    )
    safe_annual_temperature = jnp.where(
        seasonal,
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
    onset_trigger = dormant_branch & (onset_gdd > critical_onset_gdd)
    onset_flag = jnp.where(onset_trigger, jnp.float64(1.0), onset_flag)
    dormant = jnp.where(onset_trigger, jnp.float64(0.0), dormant)
    onset_gdd_flag = jnp.where(onset_trigger, jnp.float64(0.0), onset_gdd_flag)
    onset_gdd = jnp.where(onset_trigger, jnp.float64(0.0), onset_gdd)
    onset_period_seconds = fortran_multiply(
        jnp.asarray(parameters.onset_period_days, dtype=jnp.float64),
        seconds_per_day,
    )
    onset_counter = jnp.where(onset_trigger, onset_period_seconds, onset_counter)

    carbon_storage_mask = jnp.stack(
        (
            onset_trigger,
            onset_trigger,
            *(onset_trigger & woody for _ in range(5)),
        ),
        axis=1,
    )
    nitrogen_storage_mask = jnp.stack(
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
    safe_carbon_storage = jnp.where(
        carbon_storage_mask,
        jnp.asarray(inputs.carbon_storage_patch_pool, dtype=jnp.float64),
        jnp.float64(0.0),
    )
    safe_nitrogen_storage = jnp.where(
        nitrogen_storage_mask,
        jnp.asarray(inputs.nitrogen_storage_patch_pool, dtype=jnp.float64),
        jnp.float64(0.0),
    )
    carbon_storage_rate = fortran_divide(
        fortran_multiply(storage_fraction, safe_carbon_storage),
        time_step,
    )
    nitrogen_storage_rate = fortran_divide(
        fortran_multiply(storage_fraction, safe_nitrogen_storage),
        time_step,
    )
    updated_fluxes = _replace_flux_family(
        updated_fluxes,
        CARBON_STORAGE_TO_TRANSFER_FLUX_FIELDS,
        carbon_storage_rate,
        carbon_storage_mask,
    )
    updated_fluxes = _replace_flux_family(
        updated_fluxes,
        NITROGEN_STORAGE_TO_TRANSFER_FLUX_FIELDS,
        nitrogen_storage_rate,
        nitrogen_storage_mask,
    )

    active_branch = (
        seasonal
        & ~dormant_branch
        & (offset_flag == jnp.float64(0.0))
    )
    days_active_candidate = fortran_add(days_active, fraction_of_day)
    update_days_active = active_branch & use_dynamic_vegetation
    days_active = jnp.where(update_days_active, days_active_candidate, days_active)
    pft_may_exist = jnp.where(
        update_days_active & (days_active > jnp.float64(355.0)),
        jnp.bool_(False),
        pft_may_exist,
    )
    offset_trigger = (
        active_branch
        & ~winter_to_summer
        & (daylength < critical_daylength)
    )
    offset_flag = jnp.where(offset_trigger, jnp.float64(1.0), offset_flag)
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

    return SeasonalDeciduousPhenologyResult(
        state=SeasonalDeciduousPhenologyState(
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
            previous_fine_root_carbon_to_litter_patch=previous_fine_root_litter,
            pft_may_exist_patch=pft_may_exist,
        ),
        phenology_rates=updated_rates,
        cn_fluxes=updated_fluxes,
        diagnostics=SeasonalDeciduousPhenologyDiagnostics(
            seasonal_deciduous_patch_count=jnp.sum(seasonal, dtype=jnp.int32),
            offset_cleanup_patch_count=jnp.sum(offset_cleanup, dtype=jnp.int32),
            onset_cleanup_patch_count=jnp.sum(onset_cleanup, dtype=jnp.int32),
            onset_trigger_patch_count=jnp.sum(onset_trigger, dtype=jnp.int32),
            offset_trigger_patch_count=jnp.sum(offset_trigger, dtype=jnp.int32),
        ),
    )


__all__ = [
    "DEFAULT_SEASONAL_DECIDUOUS_PHENOLOGY_PARAMETERS",
    "SeasonalDeciduousPhenologyDiagnostics",
    "SeasonalDeciduousPhenologyInputs",
    "SeasonalDeciduousPhenologyParameters",
    "SeasonalDeciduousPhenologyResult",
    "SeasonalDeciduousPhenologyState",
    "seasonal_deciduous_phenology",
]
