
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    get_cn_flux_field,
    set_cn_flux_field,
)
from clm_jax.physics.numerics import (
    fortran_divide,
    fortran_ordered_sum,
    strict_exp,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.numerics import (
    fortran_hardware_power as fortran_power,
)


class PlantNutrientDemandInputs(NamedTuple):

    soil_patch: jnp.ndarray
    patch_column_index: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    sunlit_photosynthesis_patch: jnp.ndarray
    shaded_photosynthesis_patch: jnp.ndarray
    c13_sunlit_photosynthesis_patch: jnp.ndarray
    c13_shaded_photosynthesis_patch: jnp.ndarray
    c14_sunlit_photosynthesis_patch: jnp.ndarray
    c14_shaded_photosynthesis_patch: jnp.ndarray
    sunlit_leaf_area_index_patch: jnp.ndarray
    shaded_leaf_area_index_patch: jnp.ndarray
    growing_degree_days_since_planting_patch: jnp.ndarray
    growing_degree_days_topsoil_patch: jnp.ndarray
    crop_live_patch: jnp.ndarray
    growing_degree_days_to_maturity_patch: jnp.ndarray
    heat_units_to_leaf_emergence_patch: jnp.ndarray
    heat_units_to_grain_fill_patch: jnp.ndarray
    peak_leaf_area_index_patch: jnp.ndarray
    temporary_photosynthate_carbon_pool_patch: jnp.ndarray
    leaf_carbon_patch: jnp.ndarray
    fine_root_carbon_patch: jnp.ndarray
    live_stem_carbon_patch: jnp.ndarray
    live_coarse_root_carbon_patch: jnp.ndarray
    retranslocated_nitrogen_pool_patch: jnp.ndarray
    annual_net_primary_productivity_patch: jnp.ndarray
    leaf_nitrogen_patch: jnp.ndarray
    fine_root_nitrogen_patch: jnp.ndarray
    live_stem_nitrogen_patch: jnp.ndarray
    annual_potential_gpp_patch: jnp.ndarray
    annual_maximum_retranslocated_nitrogen_patch: jnp.ndarray
    soil_mineral_nitrogen_column_level: jnp.ndarray
    decomposition_temperature_scalar_column_level: jnp.ndarray
    transpiration_wetness_factor_patch: jnp.ndarray


class PlantNutrientDemandParameters(NamedTuple):

    woody_pft: jnp.ndarray
    fine_root_to_leaf_carbon_pft: jnp.ndarray
    coarse_root_to_stem_carbon_pft: jnp.ndarray
    stem_to_leaf_carbon_pft: jnp.ndarray
    live_wood_fraction_pft: jnp.ndarray
    leaf_carbon_nitrogen_ratio_pft: jnp.ndarray
    fine_root_carbon_nitrogen_ratio_pft: jnp.ndarray
    live_wood_carbon_nitrogen_ratio_pft: jnp.ndarray
    dead_wood_carbon_nitrogen_ratio_pft: jnp.ndarray
    grain_carbon_nitrogen_ratio_pft: jnp.ndarray
    organ_fill_leaf_carbon_nitrogen_ratio_pft: jnp.ndarray
    organ_fill_fine_root_carbon_nitrogen_ratio_pft: jnp.ndarray
    organ_fill_stem_carbon_nitrogen_ratio_pft: jnp.ndarray
    crop_allocation_shape_pft: jnp.ndarray
    final_leaf_allocation_pft: jnp.ndarray
    final_root_allocation_pft: jnp.ndarray
    final_stem_allocation_pft: jnp.ndarray
    initial_root_allocation_pft: jnp.ndarray
    initial_leaf_fraction_pft: jnp.ndarray
    leaf_allocation_exponent_pft: jnp.ndarray
    stem_allocation_exponent_pft: jnp.ndarray
    growth_respiration_fraction_pft: jnp.ndarray
    current_growth_respiration_fraction_pft: jnp.ndarray
    crop_decline_fraction_pft: jnp.ndarray
    seasonal_deciduous_pft: jnp.ndarray
    stress_deciduous_pft: jnp.ndarray
    soybean_pft: jnp.ndarray
    soil_layer_thickness: jnp.ndarray
    first_crop_pft: int | jnp.ndarray = 17
    time_step_seconds: float | jnp.ndarray = 1800.0
    xsmrpool_recovery_days: float | jnp.ndarray = 30.0
    seconds_per_day: float | jnp.ndarray = 86_400.0
    minimum_nitrogen: float | jnp.ndarray = 1.0e-9
    special_value: float | jnp.ndarray = 1.0e36
    use_fun: bool | jnp.ndarray = True
    use_carbon_13: bool | jnp.ndarray = False
    use_carbon_14: bool | jnp.ndarray = False
    use_nitrogen_scalar: bool | jnp.ndarray = True
    plant_nitrogen_demand_option: int | jnp.ndarray = 3
    use_substrate_term: bool | jnp.ndarray = True
    use_temperature_scalar: bool | jnp.ndarray = True


_DEFAULT_PFT_COUNT = 79
_DEFAULT_REAL_PFT = np.ones(_DEFAULT_PFT_COUNT, dtype=np.float64)
_DEFAULT_CN_PFT = np.full(_DEFAULT_PFT_COUNT, 25.0, dtype=np.float64)
_DEFAULT_FALSE_PFT = np.zeros(_DEFAULT_PFT_COUNT, dtype=bool)
DEFAULT_PLANT_NUTRIENT_DEMAND_PARAMETERS = PlantNutrientDemandParameters(
    woody_pft=_DEFAULT_REAL_PFT,
    fine_root_to_leaf_carbon_pft=_DEFAULT_REAL_PFT,
    coarse_root_to_stem_carbon_pft=_DEFAULT_REAL_PFT,
    stem_to_leaf_carbon_pft=_DEFAULT_REAL_PFT,
    live_wood_fraction_pft=np.full(_DEFAULT_PFT_COUNT, 0.1, dtype=np.float64),
    leaf_carbon_nitrogen_ratio_pft=_DEFAULT_CN_PFT,
    fine_root_carbon_nitrogen_ratio_pft=_DEFAULT_CN_PFT,
    live_wood_carbon_nitrogen_ratio_pft=_DEFAULT_CN_PFT,
    dead_wood_carbon_nitrogen_ratio_pft=_DEFAULT_CN_PFT,
    grain_carbon_nitrogen_ratio_pft=_DEFAULT_CN_PFT,
    organ_fill_leaf_carbon_nitrogen_ratio_pft=_DEFAULT_CN_PFT,
    organ_fill_fine_root_carbon_nitrogen_ratio_pft=_DEFAULT_CN_PFT,
    organ_fill_stem_carbon_nitrogen_ratio_pft=_DEFAULT_CN_PFT,
    crop_allocation_shape_pft=_DEFAULT_REAL_PFT,
    final_leaf_allocation_pft=np.full(_DEFAULT_PFT_COUNT, 0.1, dtype=np.float64),
    final_root_allocation_pft=np.full(_DEFAULT_PFT_COUNT, 0.1, dtype=np.float64),
    final_stem_allocation_pft=np.full(_DEFAULT_PFT_COUNT, 0.1, dtype=np.float64),
    initial_root_allocation_pft=np.full(_DEFAULT_PFT_COUNT, 0.4, dtype=np.float64),
    initial_leaf_fraction_pft=np.full(_DEFAULT_PFT_COUNT, 0.6, dtype=np.float64),
    leaf_allocation_exponent_pft=_DEFAULT_REAL_PFT,
    stem_allocation_exponent_pft=_DEFAULT_REAL_PFT,
    growth_respiration_fraction_pft=np.full(
        _DEFAULT_PFT_COUNT,
        0.25,
        dtype=np.float64,
    ),
    current_growth_respiration_fraction_pft=np.full(
        _DEFAULT_PFT_COUNT,
        0.25,
        dtype=np.float64,
    ),
    crop_decline_fraction_pft=_DEFAULT_REAL_PFT,
    seasonal_deciduous_pft=_DEFAULT_FALSE_PFT,
    stress_deciduous_pft=_DEFAULT_FALSE_PFT,
    soybean_pft=_DEFAULT_FALSE_PFT,
    soil_layer_thickness=np.asarray(
        [
            0.02,
            0.04,
            0.06,
            0.08,
            0.12,
            0.16,
            0.20,
            0.24,
            0.28,
            0.32,
            0.36,
            0.40,
            0.44,
            0.54,
            0.64,
            0.74,
            0.84,
            0.94,
            1.04,
            1.14,
        ],
        dtype=np.float64,
    ),
)


class PlantNutrientDemandDirectFluxState(NamedTuple):

    gpp_before_downregulation_patch: jnp.ndarray
    available_carbon_patch: jnp.ndarray
    xsmrpool_recovery_patch: jnp.ndarray
    plant_nitrogen_demand_patch: jnp.ndarray
    available_retranslocated_nitrogen_patch: jnp.ndarray
    c13_sunlit_photosynthesis_to_cpool_patch: jnp.ndarray
    c13_shaded_photosynthesis_to_cpool_patch: jnp.ndarray
    c14_sunlit_photosynthesis_to_cpool_patch: jnp.ndarray
    c14_shaded_photosynthesis_to_cpool_patch: jnp.ndarray


class PlantNutrientDemandState(NamedTuple):

    saved_leaf_allocation_patch: jnp.ndarray
    saved_stem_allocation_patch: jnp.ndarray
    leaf_allocation_patch: jnp.ndarray
    stem_allocation_patch: jnp.ndarray
    grain_fill_flag_patch: jnp.ndarray
    carbon_allometry_patch: jnp.ndarray
    nitrogen_allometry_patch: jnp.ndarray
    temporary_sum_potential_gpp_patch: jnp.ndarray
    temporary_maximum_retranslocated_nitrogen_patch: jnp.ndarray
    actual_leaf_carbon_nitrogen_ratio_patch: jnp.ndarray
    root_allocation_patch: jnp.ndarray
    reproductive_allocation_patch: jnp.ndarray
    direct_fluxes: PlantNutrientDemandDirectFluxState
    cn_fluxes: CnTransientFluxState


class PlantNutrientDemandDiagnostics(NamedTuple):
    soil_patch_count: jnp.ndarray
    live_crop_patch_count: jnp.ndarray
    crop_phase_two_patch_count: jnp.ndarray
    crop_grain_fill_patch_count: jnp.ndarray
    crop_pre_emergence_patch_count: jnp.ndarray
    retranslocation_start_patch_count: jnp.ndarray
    xsmrpool_recovery_patch_count: jnp.ndarray


class PlantNutrientDemandResult(NamedTuple):
    state: PlantNutrientDemandState
    diagnostics: PlantNutrientDemandDiagnostics


_PATCH_PARAMETER_FIELDS = (
    "woody_pft",
    "fine_root_to_leaf_carbon_pft",
    "coarse_root_to_stem_carbon_pft",
    "stem_to_leaf_carbon_pft",
    "live_wood_fraction_pft",
    "leaf_carbon_nitrogen_ratio_pft",
    "fine_root_carbon_nitrogen_ratio_pft",
    "live_wood_carbon_nitrogen_ratio_pft",
    "dead_wood_carbon_nitrogen_ratio_pft",
    "grain_carbon_nitrogen_ratio_pft",
    "organ_fill_leaf_carbon_nitrogen_ratio_pft",
    "organ_fill_fine_root_carbon_nitrogen_ratio_pft",
    "organ_fill_stem_carbon_nitrogen_ratio_pft",
    "crop_allocation_shape_pft",
    "final_leaf_allocation_pft",
    "final_root_allocation_pft",
    "final_stem_allocation_pft",
    "initial_root_allocation_pft",
    "initial_leaf_fraction_pft",
    "leaf_allocation_exponent_pft",
    "stem_allocation_exponent_pft",
    "growth_respiration_fraction_pft",
    "current_growth_respiration_fraction_pft",
    "crop_decline_fraction_pft",
    "seasonal_deciduous_pft",
    "stress_deciduous_pft",
    "soybean_pft",
)


def _validate_shapes(
    inputs: PlantNutrientDemandInputs,
    state: PlantNutrientDemandState,
    parameters: PlantNutrientDemandParameters,
) -> tuple[int, int]:
    if jnp.ndim(inputs.soil_patch) != 1:
        raise ValueError("soil_patch must be a vector")
    patch_count = inputs.soil_patch.shape[0]
    patch_fields = set(PlantNutrientDemandInputs._fields).difference(
        {
            "soil_mineral_nitrogen_column_level",
            "decomposition_temperature_scalar_column_level",
        }
    )
    for name in patch_fields:
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    if jnp.ndim(inputs.soil_mineral_nitrogen_column_level) != 2:
        raise ValueError("soil_mineral_nitrogen_column_level must be a matrix")
    column_count, level_count = inputs.soil_mineral_nitrogen_column_level.shape
    if jnp.shape(inputs.decomposition_temperature_scalar_column_level) != (
        column_count,
        level_count,
    ):
        raise ValueError("decomposition temperature scalar shape is incompatible")
    for name in PlantNutrientDemandState._fields[:12]:
        if jnp.shape(getattr(state, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    for name in PlantNutrientDemandDirectFluxState._fields:
        if jnp.shape(getattr(state.direct_fluxes, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    if state.cn_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("cn_fluxes must contain every patch")
    if state.cn_fluxes.column_scalar.shape[0] != column_count:
        raise ValueError("cn_fluxes must contain every column")
    pft_count = jnp.shape(parameters.woody_pft)
    if len(pft_count) != 1 or pft_count[0] < 1:
        raise ValueError("woody_pft must be a nonempty PFT vector")
    for name in _PATCH_PARAMETER_FIELDS:
        if jnp.shape(getattr(parameters, name)) != pft_count:
            raise ValueError(f"{name} must share the PFT axis")
    if jnp.shape(parameters.soil_layer_thickness) != (level_count,):
        raise ValueError("soil_layer_thickness must share the decomposition level axis")
    for name in PlantNutrientDemandParameters._fields[28:]:
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    return patch_count, column_count


def _selected(parameters: PlantNutrientDemandParameters, name: str, pft):
    values = getattr(parameters, name)
    dtype = jnp.bool_ if name.endswith("_pft") and name in {
        "seasonal_deciduous_pft",
        "stress_deciduous_pft",
        "soybean_pft",
    } else jnp.float64
    return jnp.asarray(values, dtype=dtype)[pft]


def _masked_update(old, new, mask):
    return jnp.where(mask, jnp.asarray(new, dtype=jnp.float64), old)


def _set_masked_flux(cn_fluxes, name: str, values, mask):
    old = get_cn_flux_field(cn_fluxes, name)
    return set_cn_flux_field(cn_fluxes, name, _masked_update(old, values, mask))


def _clip_unit_interval(value):
    return jnp.minimum(jnp.maximum(jnp.float64(0.0), value), jnp.float64(1.0))


def plant_nutrient_demand(
    inputs: PlantNutrientDemandInputs,
    state: PlantNutrientDemandState,
    parameters: PlantNutrientDemandParameters = (
        DEFAULT_PLANT_NUTRIENT_DEMAND_PARAMETERS
    ),
) -> PlantNutrientDemandResult:

    _validate_shapes(inputs, state, parameters)
    soil = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    column = jnp.asarray(inputs.patch_column_index, dtype=jnp.int32)
    pft = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    first_crop = jnp.asarray(parameters.first_crop_pft, dtype=jnp.int32)
    crop = pft >= first_crop
    crop_live = jnp.asarray(inputs.crop_live_patch, dtype=jnp.bool_)
    live_crop = soil & crop & crop_live
    woody = _selected(parameters, "woody_pft", pft) == jnp.float64(1.0)
    time_step = jnp.asarray(parameters.time_step_seconds, dtype=jnp.float64)

    psnsun_to_cpool = fortran_multiply(
        fortran_multiply(
            inputs.sunlit_photosynthesis_patch,
            inputs.sunlit_leaf_area_index_patch,
        ),
        jnp.float64(12.011e-6),
    )
    psnshade_to_cpool = fortran_multiply(
        fortran_multiply(
            inputs.shaded_photosynthesis_patch,
            inputs.shaded_leaf_area_index_patch,
        ),
        jnp.float64(12.011e-6),
    )
    gpp = fortran_add(psnsun_to_cpool, psnshade_to_cpool)

    cn_fluxes = state.cn_fluxes
    leaf_mr = get_cn_flux_field(cn_fluxes, "vegetation_carbon.leaf_mr_patch")
    fine_root_mr = get_cn_flux_field(
        cn_fluxes,
        "vegetation_carbon.froot_mr_patch",
    )
    live_stem_mr = get_cn_flux_field(
        cn_fluxes,
        "vegetation_carbon.livestem_mr_patch",
    )
    live_coarse_root_mr = get_cn_flux_field(
        cn_fluxes,
        "vegetation_carbon.livecroot_mr_patch",
    )
    grain_mr = get_cn_flux_field(cn_fluxes, "vegetation_carbon.grain_mr_patch")
    maintenance_respiration = fortran_add(leaf_mr, fine_root_mr)
    woody_respiration = fortran_add(
        fortran_add(maintenance_respiration, live_stem_mr),
        live_coarse_root_mr,
    )
    live_crop_respiration = fortran_add(
        fortran_add(maintenance_respiration, live_stem_mr),
        grain_mr,
    )
    maintenance_respiration = jnp.where(
        woody,
        woody_respiration,
        jnp.where(crop & crop_live, live_crop_respiration, maintenance_respiration),
    )
    available_carbon_raw = fortran_add(gpp, -maintenance_respiration)
    respiration_shortfall = (maintenance_respiration > jnp.float64(0.0)) & (
        available_carbon_raw < jnp.float64(0.0)
    )
    current_mr_ratio = jnp.where(
        respiration_shortfall,
        fortran_divide(gpp, jnp.where(respiration_shortfall, maintenance_respiration, 1.0)),
        jnp.float64(1.0),
    )

    mr_pairs = (
        ("leaf", leaf_mr),
        ("froot", fine_root_mr),
        ("livestem", live_stem_mr),
        ("livecroot", live_coarse_root_mr),
        ("grain", grain_mr),
    )
    for prefix, respiration in mr_pairs:
        current = fortran_multiply(respiration, current_mr_ratio)
        xsmr = fortran_add(respiration, -current)
        cn_fluxes = _set_masked_flux(
            cn_fluxes,
            f"vegetation_carbon.{prefix}_curmr_patch",
            current,
            soil,
        )
        cn_fluxes = _set_masked_flux(
            cn_fluxes,
            f"vegetation_carbon.{prefix}_xsmr_patch",
            xsmr,
            soil,
        )

    available_carbon = jnp.maximum(available_carbon_raw, jnp.float64(0.0))
    negative_xsmrpool = (
        jnp.asarray(inputs.temporary_photosynthate_carbon_pool_patch)
        < jnp.float64(0.0)
    )
    recovery_denominator = fortran_multiply(
        parameters.xsmrpool_recovery_days,
        parameters.seconds_per_day,
    )
    candidate_recovery = fortran_divide(
        -jnp.asarray(inputs.temporary_photosynthate_carbon_pool_patch),
        recovery_denominator,
    )
    recovery_below_available = candidate_recovery < available_carbon
    recovery = jnp.where(
        recovery_below_available,
        candidate_recovery,
        available_carbon,
    )
    recovered_available_carbon = jnp.where(
        recovery_below_available,
        fortran_add(available_carbon, -candidate_recovery),
        jnp.float64(0.0),
    )
    available_carbon = jnp.where(
        negative_xsmrpool,
        recovered_available_carbon,
        available_carbon,
    )
    xsmrpool_recovery = jnp.where(
        negative_xsmrpool,
        recovery,
        state.direct_fluxes.xsmrpool_recovery_patch,
    )
    cpool_to_xsmrpool = get_cn_flux_field(
        cn_fluxes,
        "vegetation_carbon.cpool_to_xsmrpool_patch",
    )
    cpool_to_xsmrpool = jnp.where(
        soil & negative_xsmrpool,
        recovery,
        cpool_to_xsmrpool,
    )
    cn_fluxes = set_cn_flux_field(
        cn_fluxes,
        "vegetation_carbon.cpool_to_xsmrpool_patch",
        cpool_to_xsmrpool,
    )

    f1 = _selected(parameters, "fine_root_to_leaf_carbon_pft", pft)
    f2 = _selected(parameters, "coarse_root_to_stem_carbon_pft", pft)
    stem_leaf = _selected(parameters, "stem_to_leaf_carbon_pft", pft)
    default_real_negative_point_zero_zero_four = jnp.float64(
        np.float32(-0.004)
    )
    default_real_two_point_seven = jnp.float64(np.float32(2.7))
    default_real_point_four = jnp.float64(np.float32(0.4))
    dynamic_stem_argument = fortran_multiply(
        default_real_negative_point_zero_zero_four,
        fortran_add(inputs.annual_net_primary_productivity_patch, -300.0),
    )
    dynamic_stem = fortran_add(
        fortran_divide(
            default_real_two_point_seven,
            fortran_add(jnp.float64(1.0), strict_exp(dynamic_stem_argument)),
        ),
        -default_real_point_four,
    )
    f3 = jnp.where(stem_leaf == jnp.float64(-1.0), dynamic_stem, stem_leaf)
    f4 = _selected(parameters, "live_wood_fraction_pft", pft)
    g1 = _selected(parameters, "growth_respiration_fraction_pft", pft)
    _ = _selected(parameters, "current_growth_respiration_fraction_pft", pft)
    _ = _selected(parameters, "seasonal_deciduous_pft", pft)
    _ = _selected(parameters, "stress_deciduous_pft", pft)
    _ = inputs.transpiration_wetness_factor_patch
    leaf_cn = _selected(parameters, "leaf_carbon_nitrogen_ratio_pft", pft)
    fine_root_cn = _selected(
        parameters,
        "fine_root_carbon_nitrogen_ratio_pft",
        pft,
    )
    live_wood_cn = _selected(
        parameters,
        "live_wood_carbon_nitrogen_ratio_pft",
        pft,
    )
    dead_wood_cn = _selected(
        parameters,
        "dead_wood_carbon_nitrogen_ratio_pft",
        pft,
    )
    grain_cn = _selected(parameters, "grain_carbon_nitrogen_ratio_pft", pft)

    leaf_allocation = jnp.asarray(state.leaf_allocation_patch, dtype=jnp.float64)
    stem_allocation = jnp.asarray(state.stem_allocation_patch, dtype=jnp.float64)
    saved_leaf_allocation = jnp.asarray(
        state.saved_leaf_allocation_patch,
        dtype=jnp.float64,
    )
    saved_stem_allocation = jnp.asarray(
        state.saved_stem_allocation_patch,
        dtype=jnp.float64,
    )
    root_allocation = jnp.asarray(state.root_allocation_patch, dtype=jnp.float64)
    reproductive_allocation = jnp.asarray(
        state.reproductive_allocation_patch,
        dtype=jnp.float64,
    )
    grain_fill_flag = jnp.asarray(state.grain_fill_flag_patch, dtype=jnp.float64)

    hui = jnp.asarray(inputs.growing_degree_days_since_planting_patch)
    leafout = jnp.asarray(inputs.growing_degree_days_topsoil_patch)
    maturity = jnp.asarray(inputs.growing_degree_days_to_maturity_patch)
    leaf_emergence = jnp.asarray(inputs.heat_units_to_leaf_emergence_patch)
    grain_start = jnp.asarray(inputs.heat_units_to_grain_fill_patch)
    peak_lai = jnp.asarray(inputs.peak_leaf_area_index_patch) == jnp.int32(1)
    phase_two = live_crop & (leafout >= leaf_emergence) & (hui < grain_start)
    grain_fill = live_crop & ~phase_two & (hui >= grain_start)
    pre_emergence = live_crop & ~phase_two & ~grain_fill

    initial_root = _selected(parameters, "initial_root_allocation_pft", pft)
    final_root = _selected(parameters, "final_root_allocation_pft", pft)
    maturity_fraction = jnp.minimum(
        jnp.float64(1.0),
        fortran_divide(hui, jnp.where(maturity != 0.0, maturity, 1.0)),
    )
    root_decline = fortran_multiply(
        fortran_add(initial_root, -final_root),
        maturity_fraction,
    )
    phase_root = _clip_unit_interval(fortran_add(initial_root, -root_decline))

    shape = _selected(parameters, "crop_allocation_shape_pft", pft)
    exp_shape = strict_exp(-shape)
    heat_fraction_argument = fortran_divide(
        fortran_multiply(-shape, hui),
        jnp.where(grain_start != 0.0, grain_start, 1.0),
    )
    initial_leaf_fraction = _selected(
        parameters,
        "initial_leaf_fraction_pft",
        pft,
    )
    leaf_fraction = fortran_divide(
        fortran_multiply(
            initial_leaf_fraction,
            fortran_add(exp_shape, -strict_exp(heat_fraction_argument)),
        ),
        jnp.where(
            live_crop & (exp_shape != jnp.float64(1.0)),
            fortran_add(exp_shape, jnp.float64(-1.0)),
            jnp.float64(1.0),
        ),
    )
    nonpeak_leaf = jnp.maximum(
        jnp.float64(1.0e-5),
        fortran_multiply(
            fortran_add(jnp.float64(1.0), -phase_root),
            leaf_fraction,
        ),
    )
    phase_two_leaf = jnp.where(peak_lai, jnp.float64(1.0e-5), nonpeak_leaf)
    phase_two_stem = jnp.where(
        peak_lai,
        jnp.float64(0.0),
        fortran_add(
            fortran_add(
                fortran_add(jnp.float64(1.0), -jnp.float64(0.0)),
                -phase_two_leaf,
            ),
            -phase_root,
        ),
    )
    phase_two_root = jnp.where(
        peak_lai,
        fortran_add(jnp.float64(1.0), -phase_two_leaf),
        phase_root,
    )
    leaf_allocation = jnp.where(phase_two, phase_two_leaf, leaf_allocation)
    stem_allocation = jnp.where(phase_two, phase_two_stem, stem_allocation)
    root_allocation = jnp.where(phase_two, phase_two_root, root_allocation)
    reproductive_allocation = jnp.where(
        phase_two,
        jnp.float64(0.0),
        reproductive_allocation,
    )
    saved_leaf_allocation = jnp.where(
        phase_two,
        phase_two_leaf,
        saved_leaf_allocation,
    )
    saved_stem_allocation = jnp.where(
        phase_two,
        phase_two_stem,
        saved_stem_allocation,
    )
    grain_fill_flag = jnp.where(phase_two, jnp.float64(0.0), grain_fill_flag)

    decline_fraction = _selected(parameters, "crop_decline_fraction_pft", pft)
    decline_denominator = fortran_add(
        fortran_multiply(maturity, decline_fraction),
        -grain_start,
    )
    decline_ratio = jnp.minimum(
        fortran_divide(
            fortran_add(hui, -grain_start),
            jnp.where(decline_denominator != 0.0, decline_denominator, 1.0),
        ),
        jnp.float64(1.0),
    )
    decline_ratio_for_power = jnp.where(
        grain_fill,
        decline_ratio,
        jnp.float64(0.0),
    )
    final_stem = _selected(parameters, "final_stem_allocation_pft", pft)
    stem_decline_factor = fortran_add(
        jnp.float64(1.0),
        -fortran_power(
            decline_ratio_for_power,
            _selected(parameters, "stem_allocation_exponent_pft", pft),
        ),
    )
    declining_stem = jnp.maximum(
        jnp.float64(0.0),
        jnp.maximum(
            final_stem,
            fortran_multiply(stem_allocation, stem_decline_factor),
        ),
    )
    grain_stem = jnp.where(
        saved_stem_allocation > final_stem,
        declining_stem,
        stem_allocation,
    )

    final_leaf = _selected(parameters, "final_leaf_allocation_pft", pft)
    leaf_decline_factor = fortran_add(
        jnp.float64(1.0),
        -fortran_power(
            decline_ratio_for_power,
            _selected(parameters, "leaf_allocation_exponent_pft", pft),
        ),
    )
    declining_leaf = jnp.maximum(
        jnp.float64(1.0e-5),
        jnp.maximum(
            final_leaf,
            fortran_multiply(leaf_allocation, leaf_decline_factor),
        ),
    )
    grain_leaf = jnp.where(
        peak_lai,
        jnp.float64(1.0e-5),
        jnp.where(saved_leaf_allocation > final_leaf, declining_leaf, leaf_allocation),
    )
    grain_root = phase_root

    soybean = _selected(parameters, "soybean_pft", pft)
    start_retranslocation = grain_fill & (
        (grain_stem == final_stem) | ~soybean
    ) & (grain_fill_flag == jnp.float64(0.0))
    inverse_time_step = fortran_divide(jnp.float64(1.0), time_step)
    organ_fill_leaf_cn = _selected(
        parameters,
        "organ_fill_leaf_carbon_nitrogen_ratio_pft",
        pft,
    )
    organ_fill_stem_cn = _selected(
        parameters,
        "organ_fill_stem_carbon_nitrogen_ratio_pft",
        pft,
    )
    organ_fill_root_cn = _selected(
        parameters,
        "organ_fill_fine_root_carbon_nitrogen_ratio_pft",
        pft,
    )
    leaf_to_retrans = fortran_multiply(
        inverse_time_step,
        jnp.maximum(
            fortran_add(
                inputs.leaf_nitrogen_patch,
                -fortran_divide(inputs.leaf_carbon_patch, organ_fill_leaf_cn),
            ),
            jnp.float64(0.0),
        ),
    )
    stem_to_retrans = fortran_multiply(
        inverse_time_step,
        jnp.maximum(
            fortran_add(
                inputs.live_stem_nitrogen_patch,
                -fortran_divide(inputs.live_stem_carbon_patch, organ_fill_stem_cn),
            ),
            jnp.float64(0.0),
        ),
    )
    root_to_retrans = jnp.where(
        organ_fill_root_cn > jnp.float64(0.0),
        fortran_multiply(
            inverse_time_step,
            jnp.maximum(
                fortran_add(
                    inputs.fine_root_nitrogen_patch,
                    -fortran_divide(
                        inputs.fine_root_carbon_patch,
                        jnp.where(organ_fill_root_cn > 0.0, organ_fill_root_cn, 1.0),
                    ),
                ),
                jnp.float64(0.0),
            ),
        ),
        jnp.float64(0.0),
    )
    cn_fluxes = _set_masked_flux(
        cn_fluxes,
        "vegetation_nitrogen.leafn_to_retransn_patch",
        leaf_to_retrans,
        start_retranslocation,
    )
    cn_fluxes = _set_masked_flux(
        cn_fluxes,
        "vegetation_nitrogen.livestemn_to_retransn_patch",
        stem_to_retrans,
        start_retranslocation,
    )
    cn_fluxes = _set_masked_flux(
        cn_fluxes,
        "vegetation_nitrogen.frootn_to_retransn_patch",
        root_to_retrans,
        start_retranslocation,
    )
    grain_fill_flag = jnp.where(
        start_retranslocation,
        jnp.float64(1.0),
        grain_fill_flag,
    )
    grain_reproductive = fortran_add(
        fortran_add(
            fortran_add(jnp.float64(1.0), -grain_root),
            -grain_stem,
        ),
        -grain_leaf,
    )
    root_allocation = jnp.where(grain_fill, grain_root, root_allocation)
    stem_allocation = jnp.where(grain_fill, grain_stem, stem_allocation)
    leaf_allocation = jnp.where(grain_fill, grain_leaf, leaf_allocation)
    reproductive_allocation = jnp.where(
        grain_fill,
        grain_reproductive,
        reproductive_allocation,
    )

    leaf_allocation = jnp.where(pre_emergence, jnp.float64(1.0e-5), leaf_allocation)
    stem_allocation = jnp.where(pre_emergence, jnp.float64(0.0), stem_allocation)
    root_allocation = jnp.where(pre_emergence, jnp.float64(0.0), root_allocation)
    reproductive_allocation = jnp.where(
        pre_emergence,
        jnp.float64(0.0),
        reproductive_allocation,
    )

    safe_crop_leaf = jnp.where(leaf_allocation != 0.0, leaf_allocation, 1.0)
    crop_f1 = fortran_divide(root_allocation, safe_crop_leaf)
    crop_f3 = fortran_divide(stem_allocation, safe_crop_leaf)
    crop_f5 = fortran_divide(reproductive_allocation, safe_crop_leaf)
    f1 = jnp.where(live_crop, crop_f1, jnp.where(soil & crop, 0.0, f1))
    f3 = jnp.where(live_crop, crop_f3, jnp.where(soil & crop, 0.0, f3))
    f5 = jnp.where(live_crop, crop_f5, jnp.float64(0.0))
    g1 = jnp.where(soil & crop, jnp.float64(0.25), g1)

    one_plus_f2 = fortran_add(jnp.float64(1.0), f2)
    wood_amount = fortran_multiply(f3, one_plus_f2)
    woody_c_allometry = fortran_add(
        fortran_add(jnp.float64(1.0), f1),
        wood_amount,
    )
    crop_c_allometry = fortran_add(
        fortran_add(fortran_add(jnp.float64(1.0), f1), f5),
        wood_amount,
    )
    nonwoody_c_allometry = fortran_add(jnp.float64(1.0), f1)

    leaf_n_term = fortran_divide(jnp.float64(1.0), leaf_cn)
    fine_root_n_term = fortran_divide(f1, fine_root_cn)
    live_wood_n_term = fortran_divide(
        fortran_multiply(fortran_multiply(f3, f4), one_plus_f2),
        jnp.where(woody | crop, live_wood_cn, jnp.float64(1.0)),
    )
    dead_wood_n_term = fortran_divide(
        fortran_multiply(
            fortran_multiply(
                f3,
                fortran_add(jnp.float64(1.0), -f4),
            ),
            one_plus_f2,
        ),
        jnp.where(woody | crop, dead_wood_cn, jnp.float64(1.0)),
    )
    woody_n_allometry = fortran_add(
        fortran_add(fortran_add(leaf_n_term, fine_root_n_term), live_wood_n_term),
        dead_wood_n_term,
    )
    crop_n_allometry = fortran_add(
        fortran_add(
            fortran_add(
                fortran_add(
                    leaf_n_term,
                    fine_root_n_term,
                ),
                fortran_divide(
                    f5,
                    jnp.where(crop, grain_cn, jnp.float64(1.0)),
                ),
            ),
            live_wood_n_term,
        ),
        dead_wood_n_term,
    )
    nonwoody_n_allometry = fortran_add(leaf_n_term, fine_root_n_term)

    use_fun = jnp.asarray(parameters.use_fun, dtype=jnp.bool_)
    growth_factor = fortran_add(jnp.float64(1.0), g1)
    no_fun_woody_c = fortran_multiply(growth_factor, woody_c_allometry)
    no_fun_crop_c = fortran_multiply(growth_factor, crop_c_allometry)
    no_fun_nonwoody_c = fortran_add(
        fortran_add(fortran_add(jnp.float64(1.0), g1), f1),
        fortran_multiply(f1, g1),
    )
    carbon_allometry = jnp.where(
        woody,
        jnp.where(use_fun, woody_c_allometry, no_fun_woody_c),
        jnp.where(
            crop,
            jnp.where(use_fun, crop_c_allometry, no_fun_crop_c),
            jnp.where(use_fun, nonwoody_c_allometry, no_fun_nonwoody_c),
        ),
    )
    nitrogen_allometry = jnp.where(
        woody,
        woody_n_allometry,
        jnp.where(crop, crop_n_allometry, nonwoody_n_allometry),
    )

    valid_leaf_nitrogen = (
        jnp.asarray(inputs.leaf_nitrogen_patch) >= parameters.minimum_nitrogen
    )
    actual_leaf_cn = jnp.where(
        ~valid_leaf_nitrogen,
        jnp.asarray(parameters.special_value, dtype=jnp.float64),
        fortran_divide(
            inputs.leaf_carbon_patch,
            jnp.where(valid_leaf_nitrogen, inputs.leaf_nitrogen_patch, 1.0),
        ),
    )
    leaf_cn_minimum = fortran_add(leaf_cn, jnp.float64(-10.0))
    leaf_cn_maximum = fortran_add(leaf_cn, jnp.float64(10.0))
    bounded_leaf_cn = jnp.minimum(
        jnp.maximum(
            actual_leaf_cn,
            fortran_add(leaf_cn_minimum, jnp.float64(-0.0001)),
        ),
        leaf_cn_maximum,
    )
    nitrogen_scalar = _clip_unit_interval(
        fortran_divide(
            fortran_add(bounded_leaf_cn, -leaf_cn_minimum),
            fortran_add(leaf_cn_maximum, -leaf_cn_minimum),
        )
    )
    use_nitrogen_scalar = jnp.asarray(
        parameters.use_nitrogen_scalar,
        dtype=jnp.bool_,
    )
    actual_leaf_cn = jnp.where(use_nitrogen_scalar, bounded_leaf_cn, actual_leaf_cn)
    nitrogen_scalar = jnp.where(
        use_nitrogen_scalar,
        nitrogen_scalar,
        jnp.float64(1.0),
    )

    soil_nitrogen = jnp.asarray(
        inputs.soil_mineral_nitrogen_column_level,
        dtype=jnp.float64,
    )[column, :]
    layer_thickness = jnp.asarray(parameters.soil_layer_thickness, dtype=jnp.float64)
    mineral_nitrogen_total = fortran_ordered_sum(
        jnp.swapaxes(
            fortran_multiply(soil_nitrogen, layer_thickness[None, :]),
            0,
            1,
        ),
        initial=jnp.zeros(soil_nitrogen.shape[0], dtype=jnp.float64),
    )
    substrate_term = fortran_divide(
        mineral_nitrogen_total,
        fortran_add(mineral_nitrogen_total, jnp.float64(1.0)),
    )
    substrate_term = jnp.where(
        jnp.asarray(parameters.use_substrate_term, dtype=jnp.bool_),
        substrate_term,
        jnp.float64(1.0),
    )
    temperature_scalar = _clip_unit_interval(
        jnp.asarray(inputs.decomposition_temperature_scalar_column_level)[column, 0]
    )
    temperature_scalar = jnp.where(
        jnp.asarray(parameters.use_temperature_scalar, dtype=jnp.bool_),
        temperature_scalar,
        jnp.float64(1.0),
    )

    allometric_demand = fortran_multiply(
        available_carbon,
        fortran_divide(nitrogen_allometry, carbon_allometry),
    )
    option = jnp.asarray(parameters.plant_nitrogen_demand_option, dtype=jnp.int32)
    option_one_demand = fortran_multiply(allometric_demand, substrate_term)
    vmax_demand = fortran_multiply(
        fortran_multiply(
            fortran_multiply(
                fortran_multiply(jnp.float64(2.7e-8), inputs.fine_root_carbon_patch),
                substrate_term,
            ),
            temperature_scalar,
        ),
        nitrogen_scalar,
    )
    option_two_demand = jnp.where(gpp > 0.0, vmax_demand, jnp.float64(0.0))
    option_three_demand = jnp.where(
        fortran_add(
            inputs.sunlit_leaf_area_index_patch,
            inputs.shaded_leaf_area_index_patch,
        )
        > 0.0,
        vmax_demand,
        jnp.float64(0.0),
    )
    option_three_demand = jnp.where(
        actual_leaf_cn < leaf_cn_minimum,
        jnp.float64(0.0),
        option_three_demand,
    )
    no_fun_demand = jnp.where(
        option == 0,
        allometric_demand,
        jnp.where(
            option == 1,
            option_one_demand,
            jnp.where(option == 2, option_two_demand, option_three_demand),
        ),
    )
    plant_nitrogen_demand = jnp.where(use_fun, allometric_demand, no_fun_demand)

    temporary_sum_gpp = fortran_add(state.temporary_sum_potential_gpp_patch, gpp)
    temporary_max_retransn = jnp.maximum(
        state.temporary_maximum_retranslocated_nitrogen_patch,
        inputs.retranslocated_nitrogen_pool_patch,
    )
    crop_retranslocation = crop & (grain_fill_flag == jnp.float64(1.0))
    noncrop_retranslocation = (~crop) & (inputs.annual_potential_gpp_patch > 0.0)
    seasonal_retranslocation = fortran_divide(
        fortran_multiply(
            fortran_divide(
                inputs.annual_maximum_retranslocated_nitrogen_patch,
                jnp.float64(2.0),
            ),
            fortran_divide(
                gpp,
                jnp.where(
                    inputs.annual_potential_gpp_patch > 0.0,
                    inputs.annual_potential_gpp_patch,
                    1.0,
                ),
            ),
        ),
        time_step,
    )
    available_retransn = jnp.where(
        crop_retranslocation,
        plant_nitrogen_demand,
        jnp.where(noncrop_retranslocation, seasonal_retranslocation, 0.0),
    )
    available_retransn = jnp.minimum(
        available_retransn,
        fortran_divide(inputs.retranslocated_nitrogen_pool_patch, time_step),
    )
    retransn_to_npool = jnp.minimum(plant_nitrogen_demand, available_retransn)
    plant_nitrogen_demand = jnp.where(
        use_fun,
        plant_nitrogen_demand,
        fortran_add(plant_nitrogen_demand, -retransn_to_npool),
    )

    cn_fluxes = _set_masked_flux(
        cn_fluxes,
        "vegetation_carbon.psnsun_to_cpool_patch",
        psnsun_to_cpool,
        soil,
    )
    cn_fluxes = _set_masked_flux(
        cn_fluxes,
        "vegetation_carbon.psnshade_to_cpool_patch",
        psnshade_to_cpool,
        soil,
    )
    cn_fluxes = _set_masked_flux(
        cn_fluxes,
        "vegetation_nitrogen.retransn_to_npool_patch",
        retransn_to_npool,
        soil,
    )

    use_c13 = jnp.asarray(parameters.use_carbon_13, dtype=jnp.bool_)
    use_c14 = jnp.asarray(parameters.use_carbon_14, dtype=jnp.bool_)
    c13_sun = fortran_multiply(
        fortran_multiply(
            inputs.c13_sunlit_photosynthesis_patch,
            inputs.sunlit_leaf_area_index_patch,
        ),
        jnp.float64(12.011e-6),
    )
    c13_shade = fortran_multiply(
        fortran_multiply(
            inputs.c13_shaded_photosynthesis_patch,
            inputs.shaded_leaf_area_index_patch,
        ),
        jnp.float64(12.011e-6),
    )
    c14_sun = fortran_multiply(
        fortran_multiply(
            inputs.c14_sunlit_photosynthesis_patch,
            inputs.sunlit_leaf_area_index_patch,
        ),
        jnp.float64(12.011e-6),
    )
    c14_shade = fortran_multiply(
        fortran_multiply(
            inputs.c14_shaded_photosynthesis_patch,
            inputs.shaded_leaf_area_index_patch,
        ),
        jnp.float64(12.011e-6),
    )
    direct = state.direct_fluxes
    return PlantNutrientDemandResult(
        state=PlantNutrientDemandState(
            saved_leaf_allocation_patch=_masked_update(
                state.saved_leaf_allocation_patch,
                saved_leaf_allocation,
                soil,
            ),
            saved_stem_allocation_patch=_masked_update(
                state.saved_stem_allocation_patch,
                saved_stem_allocation,
                soil,
            ),
            leaf_allocation_patch=_masked_update(
                state.leaf_allocation_patch,
                leaf_allocation,
                soil,
            ),
            stem_allocation_patch=_masked_update(
                state.stem_allocation_patch,
                stem_allocation,
                soil,
            ),
            grain_fill_flag_patch=_masked_update(
                state.grain_fill_flag_patch,
                grain_fill_flag,
                soil,
            ),
            carbon_allometry_patch=_masked_update(
                state.carbon_allometry_patch,
                carbon_allometry,
                soil,
            ),
            nitrogen_allometry_patch=_masked_update(
                state.nitrogen_allometry_patch,
                nitrogen_allometry,
                soil,
            ),
            temporary_sum_potential_gpp_patch=_masked_update(
                state.temporary_sum_potential_gpp_patch,
                temporary_sum_gpp,
                soil,
            ),
            temporary_maximum_retranslocated_nitrogen_patch=_masked_update(
                state.temporary_maximum_retranslocated_nitrogen_patch,
                temporary_max_retransn,
                soil,
            ),
            actual_leaf_carbon_nitrogen_ratio_patch=_masked_update(
                state.actual_leaf_carbon_nitrogen_ratio_patch,
                actual_leaf_cn,
                soil,
            ),
            root_allocation_patch=_masked_update(
                state.root_allocation_patch,
                root_allocation,
                soil,
            ),
            reproductive_allocation_patch=_masked_update(
                state.reproductive_allocation_patch,
                reproductive_allocation,
                soil,
            ),
            direct_fluxes=PlantNutrientDemandDirectFluxState(
                gpp_before_downregulation_patch=_masked_update(
                    direct.gpp_before_downregulation_patch,
                    gpp,
                    soil,
                ),
                available_carbon_patch=_masked_update(
                    direct.available_carbon_patch,
                    available_carbon,
                    soil,
                ),
                xsmrpool_recovery_patch=_masked_update(
                    direct.xsmrpool_recovery_patch,
                    xsmrpool_recovery,
                    soil,
                ),
                plant_nitrogen_demand_patch=_masked_update(
                    direct.plant_nitrogen_demand_patch,
                    plant_nitrogen_demand,
                    soil,
                ),
                available_retranslocated_nitrogen_patch=_masked_update(
                    direct.available_retranslocated_nitrogen_patch,
                    available_retransn,
                    soil,
                ),
                c13_sunlit_photosynthesis_to_cpool_patch=_masked_update(
                    direct.c13_sunlit_photosynthesis_to_cpool_patch,
                    c13_sun,
                    soil & use_c13,
                ),
                c13_shaded_photosynthesis_to_cpool_patch=_masked_update(
                    direct.c13_shaded_photosynthesis_to_cpool_patch,
                    c13_shade,
                    soil & use_c13,
                ),
                c14_sunlit_photosynthesis_to_cpool_patch=_masked_update(
                    direct.c14_sunlit_photosynthesis_to_cpool_patch,
                    c14_sun,
                    soil & use_c14,
                ),
                c14_shaded_photosynthesis_to_cpool_patch=_masked_update(
                    direct.c14_shaded_photosynthesis_to_cpool_patch,
                    c14_shade,
                    soil & use_c14,
                ),
            ),
            cn_fluxes=cn_fluxes,
        ),
        diagnostics=PlantNutrientDemandDiagnostics(
            soil_patch_count=jnp.sum(soil, dtype=jnp.int32),
            live_crop_patch_count=jnp.sum(live_crop, dtype=jnp.int32),
            crop_phase_two_patch_count=jnp.sum(phase_two, dtype=jnp.int32),
            crop_grain_fill_patch_count=jnp.sum(grain_fill, dtype=jnp.int32),
            crop_pre_emergence_patch_count=jnp.sum(pre_emergence, dtype=jnp.int32),
            retranslocation_start_patch_count=jnp.sum(
                start_retranslocation,
                dtype=jnp.int32,
            ),
            xsmrpool_recovery_patch_count=jnp.sum(
                soil & negative_xsmrpool,
                dtype=jnp.int32,
            ),
        ),
    )


__all__ = [
    "DEFAULT_PLANT_NUTRIENT_DEMAND_PARAMETERS",
    "PlantNutrientDemandDiagnostics",
    "PlantNutrientDemandDirectFluxState",
    "PlantNutrientDemandInputs",
    "PlantNutrientDemandParameters",
    "PlantNutrientDemandResult",
    "PlantNutrientDemandState",
    "plant_nutrient_demand",
]
