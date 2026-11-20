
from __future__ import annotations

from contextvars import ContextVar
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.physics.cn_fun import (
    fun_active_uptake_cost,
    fun_fixation_cost,
    fun_nonmycorrhizal_uptake_cost,
    fun_retranslocation,
)
from clm_jax.physics.cn_zero_manifest import NUMBER_OF_DECOMPOSITION_LEVELS
from clm_jax.physics.numerics import (
    fortran_divide,
    fortran_ordered_sum,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)

NUMBER_OF_FUN_LEVELS = 20
NUMBER_OF_MYCORRHIZAL_STEPS = 2
ECM_STEP = 0
AM_STEP = 1
_PARALLEL_FUN_LEVELS: bool | None = None
_materialized_fun = ContextVar("clm_materialized_fun", default=False)


def _parallel_fun_levels() -> bool:
    return (
        jax.default_backend() == "gpu" and _materialized_fun.get()
        if _PARALLEL_FUN_LEVELS is None else _PARALLEL_FUN_LEVELS
    )


class FunAllocationInputs(NamedTuple):

    soil_patch: jnp.ndarray
    patch_column_index: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    available_carbon_patch: jnp.ndarray
    plant_nitrogen_demand_patch: jnp.ndarray
    leaf_carbon_patch: jnp.ndarray
    leaf_carbon_storage_patch: jnp.ndarray
    fine_root_carbon_patch: jnp.ndarray
    leaf_nitrogen_patch: jnp.ndarray
    leaf_nitrogen_storage_patch: jnp.ndarray
    leaf_nitrogen_storage_to_transfer_patch: jnp.ndarray
    fine_root_nitrogen_patch: jnp.ndarray
    live_stem_nitrogen_patch: jnp.ndarray
    live_coarse_root_nitrogen_patch: jnp.ndarray
    leaf_carbon_to_litter_fun_patch: jnp.ndarray
    leaf_carbon_storage_to_transfer_patch: jnp.ndarray
    carbon_allometry_patch: jnp.ndarray
    nitrogen_allometry_patch: jnp.ndarray
    onset_flag_patch: jnp.ndarray
    offset_flag_patch: jnp.ndarray
    root_carbon_fraction_patch_level: jnp.ndarray
    soil_temperature_column_level: jnp.ndarray
    soil_liquid_water_column_level: jnp.ndarray
    maximum_nitrate_uptake_column_level: jnp.ndarray
    maximum_ammonium_uptake_column_level: jnp.ndarray


class FunAllocationState(NamedTuple):

    leaf_carbon_nitrogen_offset_patch: jnp.ndarray
    plant_carbon_nitrogen_ratio_patch: jnp.ndarray
    leaf_carbon_storage_transfer_accumulator_patch: jnp.ndarray
    leaf_nitrogen_storage_transfer_accumulator_patch: jnp.ndarray
    storage_carbon_demand_patch: jnp.ndarray
    storage_nitrogen_demand_patch: jnp.ndarray


class FunAllocationParameters(NamedTuple):

    leaf_carbon_nitrogen_ratio_pft: jnp.ndarray
    seasonal_deciduous_pft: jnp.ndarray
    stress_deciduous_pft: jnp.ndarray
    c3_photosynthesis_pft: jnp.ndarray
    fixation_a_pft: jnp.ndarray
    fixation_b_pft: jnp.ndarray
    fixation_c_pft: jnp.ndarray
    fixation_scale_pft: jnp.ndarray
    am_carbon_cost_pft: jnp.ndarray
    am_nitrogen_cost_pft: jnp.ndarray
    ecm_carbon_cost_pft: jnp.ndarray
    ecm_nitrogen_cost_pft: jnp.ndarray
    nonmycorrhizal_carbon_cost_pft: jnp.ndarray
    nonmycorrhizal_nitrogen_cost_pft: jnp.ndarray
    ecm_fraction_pft: jnp.ndarray
    growth_respiration_fraction_pft: jnp.ndarray
    flexible_cn_cost_threshold_pft: jnp.ndarray
    flexible_cn_cost_shape_pft: jnp.ndarray
    flexible_cn_maximum_offset_pft: jnp.ndarray
    fixer_fraction_pft: jnp.ndarray
    soil_layer_thickness: jnp.ndarray
    time_step_seconds: float | jnp.ndarray = 1800.0
    leaf_offset_days: float | jnp.ndarray = 30.0
    first_crop_pft: int | jnp.ndarray = 17
    freezing_temperature: float | jnp.ndarray = 273.15
    small_value: float | jnp.ndarray = 1.0e-12
    special_value: float | jnp.ndarray = 1.0e36
    large_cost: float | jnp.ndarray = 1.0e9
    use_flexible_cn: bool | jnp.ndarray = True


class FunNitrogenFluxes(NamedTuple):

    active_nitrate_patch: jnp.ndarray
    active_ammonium_patch: jnp.ndarray
    active_total_patch: jnp.ndarray
    nonmycorrhizal_nitrate_patch: jnp.ndarray
    nonmycorrhizal_ammonium_patch: jnp.ndarray
    nonmycorrhizal_total_patch: jnp.ndarray
    ecm_nitrate_patch: jnp.ndarray
    ecm_ammonium_patch: jnp.ndarray
    ecm_total_patch: jnp.ndarray
    am_nitrate_patch: jnp.ndarray
    am_ammonium_patch: jnp.ndarray
    am_total_patch: jnp.ndarray
    passive_patch: jnp.ndarray
    fixation_patch: jnp.ndarray
    retranslocation_patch: jnp.ndarray
    total_uptake_patch: jnp.ndarray
    paid_retranslocation_to_pool_patch: jnp.ndarray
    free_retranslocation_to_pool_patch: jnp.ndarray
    soil_nitrogen_to_plant_fun_patch: jnp.ndarray
    soil_nitrate_to_plant_fun_patch_level: jnp.ndarray
    soil_ammonium_to_plant_fun_patch_level: jnp.ndarray
    plant_nitrogen_demand_retranslocation_patch: jnp.ndarray


class FunCarbonFluxes(NamedTuple):

    active_nitrate_patch: jnp.ndarray
    active_ammonium_patch: jnp.ndarray
    active_total_patch: jnp.ndarray
    nonmycorrhizal_nitrate_patch: jnp.ndarray
    nonmycorrhizal_ammonium_patch: jnp.ndarray
    nonmycorrhizal_total_patch: jnp.ndarray
    fixation_patch: jnp.ndarray
    retranslocation_patch: jnp.ndarray
    nitrogen_uptake_patch: jnp.ndarray
    growth_patch: jnp.ndarray
    burned_off_patch: jnp.ndarray
    soil_carbon_change_patch: jnp.ndarray


class FunAllocationDiagnostics(NamedTuple):

    fixation_nitrogen_per_carbon_patch: jnp.ndarray
    active_nitrogen_per_carbon_patch: jnp.ndarray
    retranslocation_nitrogen_per_carbon_patch: jnp.ndarray
    nitrogen_uptake_npp_fraction_patch: jnp.ndarray
    active_patch_count: jnp.ndarray


class FunAllocationResult(NamedTuple):
    state: FunAllocationState
    nitrogen_fluxes: FunNitrogenFluxes
    carbon_fluxes: FunCarbonFluxes
    diagnostics: FunAllocationDiagnostics


class _FixerAllocation(NamedTuple):
    nitrogen_active_nitrate_level: jnp.ndarray
    nitrogen_active_ammonium_level: jnp.ndarray
    nitrogen_nonmycorrhizal_nitrate_level: jnp.ndarray
    nitrogen_nonmycorrhizal_ammonium_level: jnp.ndarray
    carbon_active_nitrate_level: jnp.ndarray
    carbon_active_ammonium_level: jnp.ndarray
    carbon_nonmycorrhizal_nitrate_level: jnp.ndarray
    carbon_nonmycorrhizal_ammonium_level: jnp.ndarray
    nitrogen_fixation: jnp.ndarray
    nitrogen_paid_retranslocation: jnp.ndarray
    nitrogen_free_retranslocation: jnp.ndarray
    carbon_active_nitrate: jnp.ndarray
    carbon_active_ammonium: jnp.ndarray
    carbon_nonmycorrhizal_nitrate: jnp.ndarray
    carbon_nonmycorrhizal_ammonium: jnp.ndarray
    carbon_fixation: jnp.ndarray
    carbon_retranslocation: jnp.ndarray
    carbon_burned_off: jnp.ndarray
    carbon_excess_for_growth: jnp.ndarray


class _PatchAllocation(NamedTuple):
    state: FunAllocationState
    nitrogen: FunNitrogenFluxes
    carbon: FunCarbonFluxes
    diagnostics: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


def _zero_fix_allocation(dtype=jnp.float64) -> _FixerAllocation:
    levels = jnp.zeros((NUMBER_OF_FUN_LEVELS,), dtype=dtype)
    zero = jnp.asarray(0.0, dtype=dtype)
    return _FixerAllocation(
        levels,
        levels,
        levels,
        levels,
        levels,
        levels,
        levels,
        levels,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
    )


def _left_sum(*terms: jnp.ndarray) -> jnp.ndarray:
    result = terms[0]
    for term in terms[1:]:
        result = fortran_add(result, term)
    return result


def _accumulate_levels(initial: jnp.ndarray, terms: jnp.ndarray) -> jnp.ndarray:

    return fortran_ordered_sum(terms, initial=initial)


def _ordered_level_terms(initial, *terms):
    for level in range(NUMBER_OF_FUN_LEVELS):
        initial = _left_sum(initial, *(term[level] for term in terms))
    return initial


def _allocate_fixer_fraction(
    *,
    fixing: bool,
    fixer_fraction: jnp.ndarray,
    npp_remaining: jnp.ndarray,
    plant_nitrogen_demand: jnp.ndarray,
    plant_cn: jnp.ndarray,
    leaf_carbon: jnp.ndarray,
    leaf_carbon_storage: jnp.ndarray,
    leaf_nitrogen: jnp.ndarray,
    leaf_nitrogen_storage: jnp.ndarray,
    litterfall_carbon: jnp.ndarray,
    litterfall_nitrogen: jnp.ndarray,
    target_leaf_cn: jnp.ndarray,
    growth_respiration_fraction: jnp.ndarray,
    flexible_cn_a: jnp.ndarray,
    flexible_cn_b: jnp.ndarray,
    flexible_cn_c: jnp.ndarray,
    is_crop: jnp.ndarray,
    use_flexible_cn: jnp.ndarray,
    cost_fixation: jnp.ndarray,
    cost_active_nitrate: jnp.ndarray,
    cost_active_ammonium: jnp.ndarray,
    cost_nonmycorrhizal_nitrate: jnp.ndarray,
    cost_nonmycorrhizal_ammonium: jnp.ndarray,
    nitrate_available: jnp.ndarray,
    ammonium_available: jnp.ndarray,
) -> _FixerAllocation:

    zero = jnp.float64(0.0)
    one = jnp.float64(1.0)
    npp_to_spend = fortran_multiply(npp_remaining, fixer_fraction)

    def conductance_body(level, conductance):
        updated = _left_sum(
            conductance,
            fortran_divide(one, cost_active_nitrate[level]),
            fortran_divide(one, cost_active_ammonium[level]),
            fortran_divide(one, cost_nonmycorrhizal_nitrate[level]),
            fortran_divide(one, cost_nonmycorrhizal_ammonium[level]),
        )
        if fixing:
            updated = fortran_add(
                updated,
                fortran_multiply(one, fortran_divide(one, cost_fixation[level])),
            )
        return updated

    if _parallel_fun_levels():
        conductances = tuple(fortran_divide(one, cost) for cost in (
            cost_active_nitrate, cost_active_ammonium,
            cost_nonmycorrhizal_nitrate, cost_nonmycorrhizal_ammonium,
        ))
        if fixing:
            conductances += (fortran_multiply(one, fortran_divide(one, cost_fixation)),)
        conductance = _ordered_level_terms(zero, *conductances)
    else:
        conductance = lax.fori_loop(
            0, NUMBER_OF_FUN_LEVELS, conductance_body, zero
        )

    fraction_active_nh4 = jnp.zeros((NUMBER_OF_FUN_LEVELS,), dtype=jnp.float64)
    fraction_nonmyc_nh4 = jnp.zeros_like(fraction_active_nh4)
    fraction_active_no3 = jnp.zeros_like(fraction_active_nh4)
    fraction_nonmyc_no3 = jnp.zeros_like(fraction_active_nh4)
    fraction_fixation = jnp.zeros_like(fraction_active_nh4)
    def exchange_body(level, carry):
        (
            fraction_active_nh4,
            fraction_nonmyc_nh4,
            fraction_active_no3,
            fraction_nonmyc_no3,
            fraction_fixation,
            sum_nitrogen_acquired,
        ) = carry
        frac_active_nh4 = fortran_divide(
            fortran_divide(one, cost_active_ammonium[level]), conductance
        )
        frac_nonmyc_nh4 = fortran_divide(
            fortran_divide(one, cost_nonmycorrhizal_ammonium[level]), conductance
        )
        frac_active_no3 = fortran_divide(
            fortran_divide(one, cost_active_nitrate[level]), conductance
        )
        frac_nonmyc_no3 = fortran_divide(
            fortran_divide(one, cost_nonmycorrhizal_nitrate[level]), conductance
        )
        frac_fix = (
            fortran_divide(
                fortran_multiply(one, fortran_divide(one, cost_fixation[level])),
                conductance,
            )
            if fixing
            else zero
        )
        fraction_active_nh4 = fraction_active_nh4.at[level].set(frac_active_nh4)
        fraction_nonmyc_nh4 = fraction_nonmyc_nh4.at[level].set(frac_nonmyc_nh4)
        fraction_active_no3 = fraction_active_no3.at[level].set(frac_active_no3)
        fraction_nonmyc_no3 = fraction_nonmyc_no3.at[level].set(frac_nonmyc_no3)
        fraction_fixation = fraction_fixation.at[level].set(frac_fix)

        exchanged_active_nh4 = fortran_divide(
            frac_active_nh4, cost_active_ammonium[level]
        )
        exchanged_nonmyc_nh4 = fortran_divide(
            frac_nonmyc_nh4, cost_nonmycorrhizal_ammonium[level]
        )
        exchanged_active_no3 = fortran_divide(
            frac_active_no3, cost_active_nitrate[level]
        )
        exchanged_nonmyc_no3 = fortran_divide(
            frac_nonmyc_no3, cost_nonmycorrhizal_nitrate[level]
        )
        sum_nitrogen_acquired = _left_sum(
            sum_nitrogen_acquired,
            exchanged_active_nh4,
            exchanged_nonmyc_nh4,
            exchanged_active_no3,
            exchanged_nonmyc_no3,
        )
        if fixing:
            sum_nitrogen_acquired = fortran_add(
                sum_nitrogen_acquired,
                fortran_divide(frac_fix, cost_fixation[level]),
            )
        return (
            fraction_active_nh4,
            fraction_nonmyc_nh4,
            fraction_active_no3,
            fraction_nonmyc_no3,
            fraction_fixation,
            sum_nitrogen_acquired,
        )

    if _parallel_fun_levels():
        fraction_active_nh4 = fortran_divide(conductances[1], conductance)
        fraction_nonmyc_nh4 = fortran_divide(conductances[3], conductance)
        fraction_active_no3 = fortran_divide(conductances[0], conductance)
        fraction_nonmyc_no3 = fortran_divide(conductances[2], conductance)
        exchanges = (
            fortran_divide(fraction_active_nh4, cost_active_ammonium),
            fortran_divide(fraction_nonmyc_nh4, cost_nonmycorrhizal_ammonium),
            fortran_divide(fraction_active_no3, cost_active_nitrate),
            fortran_divide(fraction_nonmyc_no3, cost_nonmycorrhizal_nitrate),
        )
        if fixing:
            fraction_fixation = fortran_divide(conductances[4], conductance)
            exchanges += (fortran_divide(fraction_fixation, cost_fixation),)
        sum_nitrogen_acquired = _ordered_level_terms(zero, *exchanges)
    else:
        (
            fraction_active_nh4,
            fraction_nonmyc_nh4,
            fraction_active_no3,
            fraction_nonmyc_no3,
            fraction_fixation,
            sum_nitrogen_acquired,
        ) = lax.fori_loop(
            0,
            NUMBER_OF_FUN_LEVELS,
            exchange_body,
            (
                fraction_active_nh4,
                fraction_nonmyc_nh4,
                fraction_active_no3,
                fraction_nonmyc_no3,
                fraction_fixation,
                zero,
            ),
        )
    total_nitrogen_resistance = fortran_divide(one, sum_nitrogen_acquired)

    retranslocation_active = (
        (leaf_carbon > zero)
        & (fortran_multiply(litterfall_nitrogen, fixer_fraction) > zero)
        & (~is_crop)
    )

    def calculate_retranslocation(_):
        return fun_retranslocation(
            npp_to_spend,
            fortran_multiply(litterfall_carbon, fixer_fraction),
            fortran_multiply(litterfall_nitrogen, fixer_fraction),
            total_nitrogen_resistance,
            target_leaf_cn,
            growth_respiration_fraction,
            plant_cn,
        )

    def zero_retranslocation(_):
        from clm_jax.physics.cn_fun import FunRetranslocationResult

        return FunRetranslocationResult(zero, zero, zero, zero, jnp.int32(0))

    retranslocation = lax.cond(
        retranslocation_active,
        calculate_retranslocation,
        zero_retranslocation,
        operand=None,
    )
    npp_to_spend = fortran_add(npp_to_spend, -retranslocation.carbon_spent)
    npp_to_spend = fortran_add(
        npp_to_spend, -retranslocation.carbon_accounted_for_growth
    )

    def allocate_from_soil_and_fixation(_):
        delta_cn = lax.cond(
            leaf_nitrogen == zero,
            lambda unused: flexible_cn_c,
            lambda unused: fortran_add(
                fortran_divide(
                    fortran_add(leaf_carbon, leaf_carbon_storage),
                    fortran_add(leaf_nitrogen, leaf_nitrogen_storage),
                ),
                -target_leaf_cn,
            ),
            operand=None,
        )

        def flexible_fraction(_):
            fraction = jnp.maximum(
                zero,
                fortran_add(
                    one,
                    -fortran_divide(
                        fortran_add(total_nitrogen_resistance, -flexible_cn_a),
                        flexible_cn_b,
                    ),
                ),
            )
            positive_adjustment = (delta_cn > zero) & (fraction < one)
            positive_fraction = fortran_add(
                fraction,
                fortran_multiply(
                    fortran_add(one, -fraction),
                    jnp.minimum(one, fortran_divide(delta_cn, flexible_cn_c)),
                ),
            )
            fraction = jnp.where(positive_adjustment, positive_fraction, fraction)
            negative_fraction = fortran_add(
                fraction,
                fortran_multiply(
                    jnp.float64(0.5),
                    fortran_divide(delta_cn, flexible_cn_c),
                ),
            )
            fraction = jnp.where(delta_cn < zero, negative_fraction, fraction)
            return jnp.maximum(jnp.minimum(one, fraction), jnp.float64(0.5))

        ideal_carbon_fraction = lax.cond(
            use_flexible_cn,
            flexible_fraction,
            lambda unused: one,
            operand=None,
        )
        excess_carbon = fortran_multiply(
            npp_to_spend, fortran_add(one, -ideal_carbon_fraction)
        )
        one_plus_growth_respiration = fortran_add(
            one, growth_respiration_fraction
        )
        excessive = (
            fortran_multiply(excess_carbon, one_plus_growth_respiration)
            > npp_to_spend
        )
        excess_carbon = jnp.where(
            excessive,
            fortran_divide(npp_to_spend, one_plus_growth_respiration),
            excess_carbon,
        )
        carbon_remaining = fortran_add(
            npp_to_spend,
            -fortran_multiply(excess_carbon, one_plus_growth_respiration),
        )
        denominator = fortran_add(
            fortran_multiply(
                one_plus_growth_respiration,
                fortran_divide(plant_cn, total_nitrogen_resistance),
            ),
            one,
        )
        carbon_for_uptake = fortran_divide(carbon_remaining, denominator)
        carbon_for_uptake = fortran_multiply(
            carbon_for_uptake, ideal_carbon_fraction
        )

        carbon_active_nh4 = fortran_multiply(
            fraction_active_nh4, carbon_for_uptake
        )
        carbon_nonmyc_nh4 = fortran_multiply(
            fraction_nonmyc_nh4, carbon_for_uptake
        )
        carbon_active_no3 = fortran_multiply(
            fraction_active_no3, carbon_for_uptake
        )
        carbon_nonmyc_no3 = fortran_multiply(
            fraction_nonmyc_no3, carbon_for_uptake
        )
        carbon_fixation_level = (
            fortran_multiply(fraction_fixation, carbon_for_uptake)
            if fixing
            else jnp.zeros_like(fraction_fixation)
        )
        nitrogen_active_nh4 = fortran_divide(
            carbon_active_nh4, cost_active_ammonium
        )
        nitrogen_nonmyc_nh4 = fortran_divide(
            carbon_nonmyc_nh4, cost_nonmycorrhizal_ammonium
        )
        nitrogen_active_no3 = fortran_divide(
            carbon_active_no3, cost_active_nitrate
        )
        nitrogen_nonmyc_no3 = fortran_divide(
            carbon_nonmyc_no3, cost_nonmycorrhizal_nitrate
        )
        nitrogen_fixation_level = (
            fortran_divide(carbon_fixation_level, cost_fixation)
            if fixing
            else jnp.zeros_like(carbon_fixation_level)
        )

        def limit_layer(level, layer_arrays):
            (
                n_active_no3, n_nonmyc_no3, n_active_nh4, n_nonmyc_nh4,
                c_active_no3, c_nonmyc_no3, c_active_nh4, c_nonmyc_nh4,
            ) = layer_arrays

            nitrate_limit = fortran_multiply(
                nitrate_available[level], fixer_fraction
            )
            nitrate_sum = fortran_add(
                n_active_no3[level], n_nonmyc_no3[level]
            )
            old_active_no3 = n_active_no3[level]
            nitrate_limited = nitrate_sum > nitrate_limit

            def limit_nitrate(_):
                nitrate_excess = fortran_add(nitrate_sum, -nitrate_limit)
                adjusted_active = fortran_add(
                    old_active_no3,
                    -fortran_multiply(
                        nitrate_excess,
                        fortran_divide(old_active_no3, nitrate_sum),
                    ),
                )
                adjusted_nonmyc = fortran_add(
                    n_nonmyc_no3[level],
                    -fortran_multiply(
                        nitrate_excess,
                        fortran_divide(
                            n_nonmyc_no3[level],
                            fortran_add(
                                old_active_no3, n_nonmyc_no3[level]
                            ),
                        ),
                    ),
                )
                return (
                    adjusted_active,
                    adjusted_nonmyc,
                    fortran_multiply(
                        adjusted_active, cost_active_nitrate[level]
                    ),
                    fortran_multiply(
                        adjusted_nonmyc,
                        cost_nonmycorrhizal_nitrate[level],
                    ),
                )

            def retain_nitrate(_):
                return (
                    old_active_no3,
                    n_nonmyc_no3[level],
                    c_active_no3[level],
                    c_nonmyc_no3[level],
                )

            (
                level_active_no3,
                level_nonmyc_no3,
                level_carbon_active_no3,
                level_carbon_nonmyc_no3,
            ) = lax.cond(
                nitrate_limited, limit_nitrate, retain_nitrate, operand=None
            )

            ammonium_limit = fortran_multiply(
                ammonium_available[level], fixer_fraction
            )
            ammonium_sum = fortran_add(
                n_active_nh4[level], n_nonmyc_nh4[level]
            )
            old_active_nh4 = n_active_nh4[level]
            ammonium_limited = ammonium_sum > ammonium_limit

            def limit_ammonium(_):
                ammonium_excess = fortran_add(ammonium_sum, -ammonium_limit)
                adjusted_active = fortran_add(
                    old_active_nh4,
                    -fortran_divide(
                        fortran_multiply(ammonium_excess, old_active_nh4),
                        ammonium_sum,
                    ),
                )
                adjusted_nonmyc = fortran_add(
                    n_nonmyc_nh4[level],
                    -fortran_divide(
                        fortran_multiply(
                            ammonium_excess, n_nonmyc_nh4[level]
                        ),
                        fortran_add(
                            old_active_nh4, n_nonmyc_nh4[level]
                        ),
                    ),
                )
                return (
                    adjusted_active,
                    adjusted_nonmyc,
                    fortran_multiply(
                        adjusted_active, cost_active_ammonium[level]
                    ),
                    fortran_multiply(
                        adjusted_nonmyc,
                        cost_nonmycorrhizal_ammonium[level],
                    ),
                )

            def retain_ammonium(_):
                return (
                    old_active_nh4,
                    n_nonmyc_nh4[level],
                    c_active_nh4[level],
                    c_nonmyc_nh4[level],
                )

            (
                level_active_nh4,
                level_nonmyc_nh4,
                level_carbon_active_nh4,
                level_carbon_nonmyc_nh4,
            ) = lax.cond(
                ammonium_limited, limit_ammonium, retain_ammonium, operand=None
            )
            nitrogen_acquired = _left_sum(
                level_active_no3,
                level_nonmyc_no3,
                level_active_nh4,
                level_nonmyc_nh4,
            )
            carbon_spent = _left_sum(
                level_carbon_active_no3,
                level_carbon_nonmyc_no3,
                level_carbon_active_nh4,
                level_carbon_nonmyc_nh4,
            )
            if fixing:
                nitrogen_acquired = fortran_add(
                    nitrogen_acquired, nitrogen_fixation_level[level]
                )
                carbon_spent = fortran_add(
                    carbon_spent, carbon_fixation_level[level]
                )
            growth_carbon = fortran_multiply(nitrogen_acquired, plant_cn)
            growth_carbon = fortran_multiply(
                growth_carbon, one_plus_growth_respiration
            )
            return (
                level_active_no3, level_nonmyc_no3,
                level_active_nh4, level_nonmyc_nh4,
                level_carbon_active_no3, level_carbon_nonmyc_no3,
                level_carbon_active_nh4, level_carbon_nonmyc_nh4,
                carbon_spent, growth_carbon,
            )

        def layer_body(level, carry):
            (
                n_active_no3, n_nonmyc_no3, n_active_nh4, n_nonmyc_nh4,
                c_active_no3, c_nonmyc_no3, c_active_nh4, c_nonmyc_nh4,
                remaining,
                n_active_no3_total, n_nonmyc_no3_total,
                n_active_nh4_total, n_nonmyc_nh4_total, n_fix_total,
                c_active_no3_total, c_nonmyc_no3_total,
                c_active_nh4_total, c_nonmyc_nh4_total, c_fix_total,
            ) = carry
            (
                level_active_no3, level_nonmyc_no3,
                level_active_nh4, level_nonmyc_nh4,
                level_carbon_active_no3, level_carbon_nonmyc_no3,
                level_carbon_active_nh4, level_carbon_nonmyc_nh4,
                carbon_spent, growth_carbon,
            ) = limit_layer(level, carry[:8])
            remaining = fortran_add(remaining, -carbon_spent)
            remaining = fortran_add(remaining, -growth_carbon)

            return (
                n_active_no3.at[level].set(level_active_no3),
                n_nonmyc_no3.at[level].set(level_nonmyc_no3),
                n_active_nh4.at[level].set(level_active_nh4),
                n_nonmyc_nh4.at[level].set(level_nonmyc_nh4),
                c_active_no3.at[level].set(level_carbon_active_no3),
                c_nonmyc_no3.at[level].set(level_carbon_nonmyc_no3),
                c_active_nh4.at[level].set(level_carbon_active_nh4),
                c_nonmyc_nh4.at[level].set(level_carbon_nonmyc_nh4),
                remaining,
                fortran_add(n_active_no3_total, level_active_no3),
                fortran_add(n_nonmyc_no3_total, level_nonmyc_no3),
                fortran_add(n_active_nh4_total, level_active_nh4),
                fortran_add(n_nonmyc_nh4_total, level_nonmyc_nh4),
                fortran_add(n_fix_total, nitrogen_fixation_level[level])
                if fixing
                else n_fix_total,
                fortran_add(c_active_no3_total, level_carbon_active_no3),
                fortran_add(c_nonmyc_no3_total, level_carbon_nonmyc_no3),
                fortran_add(c_active_nh4_total, level_carbon_active_nh4),
                fortran_add(c_nonmyc_nh4_total, level_carbon_nonmyc_nh4),
                fortran_add(c_fix_total, carbon_fixation_level[level])
                if fixing
                else c_fix_total,
            )

        initial = (
            nitrogen_active_no3,
            nitrogen_nonmyc_no3,
            nitrogen_active_nh4,
            nitrogen_nonmyc_nh4,
            carbon_active_no3,
            carbon_nonmyc_no3,
            carbon_active_nh4,
            carbon_nonmyc_nh4,
            carbon_remaining,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
        )
        if _parallel_fun_levels():
            levels = jax.vmap(limit_layer, in_axes=(0, None))(
                jnp.arange(NUMBER_OF_FUN_LEVELS), initial[:8],
            )
            remaining = _ordered_level_terms(carbon_remaining, -levels[8], -levels[9])
            totals = tuple(_accumulate_levels(zero, values) for values in levels[:8])
            final = (
                *levels[:8], remaining, *totals[:4],
                _accumulate_levels(zero, nitrogen_fixation_level) if fixing else zero,
                *totals[4:],
                _accumulate_levels(zero, carbon_fixation_level) if fixing else zero,
            )
        else:
            final = lax.fori_loop(0, NUMBER_OF_FUN_LEVELS, layer_body, initial)
        burned = jnp.where(final[8] >= jnp.float64(1.0e-13), final[8], zero)
        return _FixerAllocation(
            final[0],
            final[2],
            final[1],
            final[3],
            final[4],
            final[6],
            final[5],
            final[7],
            final[13],
            retranslocation.paid_nitrogen_retranslocated,
            retranslocation.free_nitrogen_retranslocated,
            final[14],
            final[16],
            final[15],
            final[17],
            final[18],
            retranslocation.carbon_spent,
            burned,
            excess_carbon,
        )

    def no_soil_allocation(_):
        result = _zero_fix_allocation()
        return result._replace(
            nitrogen_paid_retranslocation=(
                retranslocation.paid_nitrogen_retranslocated
            ),
            nitrogen_free_retranslocation=(
                retranslocation.free_nitrogen_retranslocated
            ),
            carbon_retranslocation=retranslocation.carbon_spent,
        )

    return lax.cond(
        plant_nitrogen_demand > zero,
        allocate_from_soil_and_fixation,
        no_soil_allocation,
        operand=None,
    )


def _active_patch(
    patch_inputs: tuple[jnp.ndarray, ...],
    patch_state: tuple[jnp.ndarray, ...],
    patch_parameters: tuple[jnp.ndarray, ...],
    column_inputs: tuple[jnp.ndarray, ...],
    scalar_parameters: tuple[jnp.ndarray, ...],
) -> _PatchAllocation:
    (
        available_carbon,
        plant_nitrogen_demand,
        leaf_carbon,
        leaf_carbon_storage,
        fine_root_carbon,
        leaf_nitrogen,
        leaf_nitrogen_storage,
        leaf_nitrogen_storage_to_transfer,
        _fine_root_nitrogen,
        _live_stem_nitrogen,
        _live_coarse_root_nitrogen,
        leaf_carbon_to_litter_fun,
        leaf_carbon_storage_to_transfer,
        carbon_allometry,
        nitrogen_allometry,
        onset_flag,
        offset_flag,
        root_carbon_fraction,
    ) = patch_inputs
    (
        leaf_cn_offset,
        old_plant_cn,
        leaf_carbon_transfer_accumulator,
        leaf_nitrogen_transfer_accumulator,
        old_storage_carbon_demand,
        old_storage_nitrogen_demand,
    ) = patch_state
    (
        target_leaf_cn,
        seasonal_deciduous,
        stress_deciduous,
        c3_photosynthesis,
        fixation_a,
        fixation_b,
        fixation_c,
        fixation_scale,
        am_carbon_cost,
        am_nitrogen_cost,
        ecm_carbon_cost,
        ecm_nitrogen_cost,
        nonmyc_carbon_cost,
        nonmyc_nitrogen_cost,
        ecm_fraction,
        growth_respiration_fraction,
        flexible_cn_a,
        flexible_cn_b,
        flexible_cn_c,
        pft_fixer_fraction,
        is_crop,
    ) = patch_parameters
    (
        soil_temperature,
        soil_liquid_water,
        maximum_nitrate_uptake,
        maximum_ammonium_uptake,
    ) = column_inputs
    (
        soil_layer_thickness,
        dt,
        leaf_offset_days,
        freezing_temperature,
        small_value,
        special_value,
        large_cost,
        use_flexible_cn,
    ) = scalar_parameters

    zero = jnp.float64(0.0)
    one = jnp.float64(1.0)
    plant_cn = lax.cond(
        nitrogen_allometry > zero,
        lambda unused: fortran_divide(carbon_allometry, nitrogen_allometry),
        lambda unused: zero,
        operand=None,
    )
    deciduous = seasonal_deciduous | stress_deciduous
    onset = deciduous & (onset_flag == one)
    leaf_carbon_transfer_accumulator = jnp.where(
        onset,
        fortran_add(
            leaf_carbon_transfer_accumulator,
            fortran_multiply(leaf_carbon_storage_to_transfer, dt),
        ),
        leaf_carbon_transfer_accumulator,
    )
    leaf_nitrogen_transfer_accumulator = jnp.where(
        onset,
        fortran_add(
            leaf_nitrogen_transfer_accumulator,
            fortran_multiply(leaf_nitrogen_storage_to_transfer, dt),
        ),
        leaf_nitrogen_transfer_accumulator,
    )
    offset = deciduous & (offset_flag == one)
    offset_steps = fortran_multiply(leaf_offset_days, jnp.float64(48.0))
    storage_carbon_demand = jnp.where(
        offset,
        fortran_divide(leaf_carbon_storage, offset_steps),
        zero,
    )
    storage_nitrogen_demand = jnp.where(
        offset,
        jnp.maximum(
            fortran_divide(leaf_nitrogen_transfer_accumulator, offset_steps),
            zero,
        ),
        zero,
    )
    storage_carbon_demand = jnp.where(deciduous, storage_carbon_demand, zero)
    storage_nitrogen_demand = jnp.where(deciduous, storage_nitrogen_demand, zero)

    new_state = FunAllocationState(
        leaf_cn_offset,
        plant_cn,
        leaf_carbon_transfer_accumulator,
        leaf_nitrogen_transfer_accumulator,
        storage_carbon_demand,
        storage_nitrogen_demand,
    )

    available_carbon_pool = fortran_multiply(available_carbon, dt)
    plant_nitrogen_demand_pool = jnp.maximum(
        fortran_multiply(plant_nitrogen_demand, dt), zero
    )
    root_carbon_density = jnp.where(
        available_carbon_pool > zero,
        fortran_multiply(root_carbon_fraction[:NUMBER_OF_FUN_LEVELS], fine_root_carbon),
        jnp.zeros((NUMBER_OF_FUN_LEVELS,), dtype=jnp.float64),
    )
    nitrate_layer = fortran_multiply(
        maximum_nitrate_uptake[:NUMBER_OF_FUN_LEVELS],
        soil_layer_thickness[:NUMBER_OF_FUN_LEVELS],
    )
    nitrate_layer = fortran_multiply(nitrate_layer, dt)
    ammonium_layer = fortran_multiply(
        maximum_ammonium_uptake[:NUMBER_OF_FUN_LEVELS],
        soil_layer_thickness[:NUMBER_OF_FUN_LEVELS],
    )
    ammonium_layer = fortran_multiply(ammonium_layer, dt)
    wet = soil_liquid_water[:NUMBER_OF_FUN_LEVELS] >= small_value
    nitrate_layer = jnp.maximum(jnp.where(wet, nitrate_layer, zero), zero)
    ammonium_layer = jnp.maximum(jnp.where(wet, ammonium_layer, zero), zero)

    fixation_cost = fun_fixation_cost(
        jnp.where(c3_photosynthesis == one, jnp.int32(1), jnp.int32(0)),
        fixation_a,
        fixation_b,
        fixation_c,
        root_carbon_fraction[:NUMBER_OF_FUN_LEVELS],
        fixation_scale,
        fortran_add(
            soil_temperature[:NUMBER_OF_FUN_LEVELS], -freezing_temperature
        ),
        large_cost=large_cost,
    )

    n_active_no3_total = zero
    n_active_nh4_total = zero
    n_nonmyc_no3_total = zero
    n_nonmyc_nh4_total = zero
    n_fix_total = zero
    n_paid_retrans_total = zero
    n_free_retrans_total = zero
    c_active_no3_total = zero
    c_active_nh4_total = zero
    c_nonmyc_no3_total = zero
    c_nonmyc_nh4_total = zero
    c_fix_total = zero
    c_retrans_total = zero
    c_burned_total = zero
    c_excess_total = zero
    vertical_active_no3 = jnp.zeros((NUMBER_OF_FUN_LEVELS,), dtype=jnp.float64)
    vertical_active_nh4 = jnp.zeros_like(vertical_active_no3)
    vertical_nonmyc_no3 = jnp.zeros_like(vertical_active_no3)
    vertical_nonmyc_nh4 = jnp.zeros_like(vertical_active_no3)
    ecm_active_no3 = zero
    ecm_active_nh4 = zero
    am_active_no3 = zero
    am_active_nh4 = zero

    for step in range(NUMBER_OF_MYCORRHIZAL_STEPS):
        if step == ECM_STEP:
            mycorrhizal_fraction = ecm_fraction
            active_carbon_cost = ecm_carbon_cost
            active_nitrogen_cost = ecm_nitrogen_cost
        else:
            mycorrhizal_fraction = fortran_add(one, -ecm_fraction)
            active_carbon_cost = am_carbon_cost
            active_nitrogen_cost = am_nitrogen_cost
        nitrate_step = fortran_multiply(nitrate_layer, mycorrhizal_fraction)
        ammonium_step = fortran_multiply(ammonium_layer, mycorrhizal_fraction)
        root_carbon_step = fortran_multiply(
            root_carbon_density, mycorrhizal_fraction
        )
        active_no3_cost = fun_active_uptake_cost(
            nitrate_step,
            active_carbon_cost,
            active_nitrogen_cost,
            root_carbon_step,
            small_value=small_value,
            large_cost=large_cost,
        )
        active_nh4_cost = fun_active_uptake_cost(
            ammonium_step,
            active_carbon_cost,
            active_nitrogen_cost,
            root_carbon_step,
            small_value=small_value,
            large_cost=large_cost,
        )
        nonmyc_no3_cost = fun_nonmycorrhizal_uptake_cost(
            nitrate_step,
            nonmyc_carbon_cost,
            nonmyc_nitrogen_cost,
            root_carbon_step,
            small_value=small_value,
            large_cost=large_cost,
        )
        nonmyc_nh4_cost = fun_nonmycorrhizal_uptake_cost(
            ammonium_step,
            nonmyc_carbon_cost,
            nonmyc_nitrogen_cost,
            root_carbon_step,
            small_value=small_value,
            large_cost=large_cost,
        )
        litterfall_carbon = jnp.where(
            leaf_carbon > zero,
            fortran_multiply(
                fortran_multiply(dt, mycorrhizal_fraction),
                leaf_carbon_to_litter_fun,
            ),
            zero,
        )
        litterfall_nitrogen = lax.cond(
            leaf_carbon > zero,
            lambda unused, mycorrhizal_fraction=mycorrhizal_fraction: fortran_divide(
                fortran_multiply(
                    fortran_multiply(
                        fortran_multiply(dt, mycorrhizal_fraction),
                        leaf_nitrogen,
                    ),
                    leaf_carbon_to_litter_fun,
                ),
                leaf_carbon,
            ),
            lambda unused: zero,
            operand=None,
        )
        retain_litterfall = (~deciduous) | (offset_flag == one)
        litterfall_carbon = jnp.where(retain_litterfall, litterfall_carbon, zero)
        litterfall_nitrogen = jnp.where(
            retain_litterfall, litterfall_nitrogen, zero
        )
        step_demand = fortran_multiply(
            plant_nitrogen_demand_pool, mycorrhizal_fraction
        )
        step_available_carbon = fortran_multiply(
            available_carbon_pool, mycorrhizal_fraction
        )

        step_active_no3 = zero
        step_active_nh4 = zero
        step_nonmyc_no3 = zero
        step_nonmyc_nh4 = zero
        step_fixation = zero
        step_paid_retranslocation = zero
        step_free_retranslocation = zero
        step_carbon_active_no3 = zero
        step_carbon_active_nh4 = zero
        step_carbon_nonmyc_no3 = zero
        step_carbon_nonmyc_nh4 = zero
        step_carbon_fixation = zero
        step_carbon_retranslocation = zero
        for fixing in (True, False):
            fraction = (
                pft_fixer_fraction
                if fixing
                else fortran_add(one, -pft_fixer_fraction)
            )
            allocation = _allocate_fixer_fraction(
                fixing=fixing,
                fixer_fraction=fraction,
                npp_remaining=step_available_carbon,
                plant_nitrogen_demand=step_demand,
                plant_cn=plant_cn,
                leaf_carbon=leaf_carbon,
                leaf_carbon_storage=leaf_carbon_storage,
                leaf_nitrogen=leaf_nitrogen,
                leaf_nitrogen_storage=leaf_nitrogen_storage,
                litterfall_carbon=litterfall_carbon,
                litterfall_nitrogen=litterfall_nitrogen,
                target_leaf_cn=target_leaf_cn,
                growth_respiration_fraction=growth_respiration_fraction,
                flexible_cn_a=flexible_cn_a,
                flexible_cn_b=flexible_cn_b,
                flexible_cn_c=flexible_cn_c,
                is_crop=is_crop,
                use_flexible_cn=use_flexible_cn,
                cost_fixation=fixation_cost,
                cost_active_nitrate=active_no3_cost,
                cost_active_ammonium=active_nh4_cost,
                cost_nonmycorrhizal_nitrate=nonmyc_no3_cost,
                cost_nonmycorrhizal_ammonium=nonmyc_nh4_cost,
                nitrate_available=nitrate_step,
                ammonium_available=ammonium_step,
            )
            step_active_no3 = _accumulate_levels(
                step_active_no3,
                allocation.nitrogen_active_nitrate_level,
            )
            step_active_nh4 = _accumulate_levels(
                step_active_nh4,
                allocation.nitrogen_active_ammonium_level,
            )
            step_nonmyc_no3 = _accumulate_levels(
                step_nonmyc_no3,
                allocation.nitrogen_nonmycorrhizal_nitrate_level,
            )
            step_nonmyc_nh4 = _accumulate_levels(
                step_nonmyc_nh4,
                allocation.nitrogen_nonmycorrhizal_ammonium_level,
            )
            step_fixation = fortran_add(
                step_fixation, allocation.nitrogen_fixation
            )
            step_paid_retranslocation = fortran_add(
                step_paid_retranslocation,
                allocation.nitrogen_paid_retranslocation,
            )
            step_free_retranslocation = fortran_add(
                step_free_retranslocation,
                allocation.nitrogen_free_retranslocation,
            )
            step_carbon_active_no3 = _accumulate_levels(
                step_carbon_active_no3,
                allocation.carbon_active_nitrate_level,
            )
            step_carbon_active_nh4 = _accumulate_levels(
                step_carbon_active_nh4,
                allocation.carbon_active_ammonium_level,
            )
            step_carbon_nonmyc_no3 = _accumulate_levels(
                step_carbon_nonmyc_no3,
                allocation.carbon_nonmycorrhizal_nitrate_level,
            )
            step_carbon_nonmyc_nh4 = _accumulate_levels(
                step_carbon_nonmyc_nh4,
                allocation.carbon_nonmycorrhizal_ammonium_level,
            )
            step_carbon_fixation = fortran_add(
                step_carbon_fixation, allocation.carbon_fixation
            )
            step_carbon_retranslocation = fortran_add(
                step_carbon_retranslocation, allocation.carbon_retranslocation
            )
            c_burned_total = fortran_add(
                c_burned_total, allocation.carbon_burned_off
            )
            c_excess_total = fortran_add(
                c_excess_total, allocation.carbon_excess_for_growth
            )
            vertical_active_no3 = fortran_add(
                vertical_active_no3, allocation.nitrogen_active_nitrate_level
            )
            vertical_active_nh4 = fortran_add(
                vertical_active_nh4, allocation.nitrogen_active_ammonium_level
            )
            vertical_nonmyc_no3 = fortran_add(
                vertical_nonmyc_no3,
                allocation.nitrogen_nonmycorrhizal_nitrate_level,
            )
            vertical_nonmyc_nh4 = fortran_add(
                vertical_nonmyc_nh4,
                allocation.nitrogen_nonmycorrhizal_ammonium_level,
            )
        n_active_no3_total = fortran_add(n_active_no3_total, step_active_no3)
        n_active_nh4_total = fortran_add(n_active_nh4_total, step_active_nh4)
        n_nonmyc_no3_total = fortran_add(n_nonmyc_no3_total, step_nonmyc_no3)
        n_nonmyc_nh4_total = fortran_add(n_nonmyc_nh4_total, step_nonmyc_nh4)
        n_fix_total = fortran_add(n_fix_total, step_fixation)
        n_paid_retrans_total = fortran_add(
            n_paid_retrans_total, step_paid_retranslocation
        )
        n_free_retrans_total = fortran_add(
            n_free_retrans_total, step_free_retranslocation
        )
        c_active_no3_total = fortran_add(
            c_active_no3_total, step_carbon_active_no3
        )
        c_active_nh4_total = fortran_add(
            c_active_nh4_total, step_carbon_active_nh4
        )
        c_nonmyc_no3_total = fortran_add(
            c_nonmyc_no3_total, step_carbon_nonmyc_no3
        )
        c_nonmyc_nh4_total = fortran_add(
            c_nonmyc_nh4_total, step_carbon_nonmyc_nh4
        )
        c_fix_total = fortran_add(c_fix_total, step_carbon_fixation)
        c_retrans_total = fortran_add(
            c_retrans_total, step_carbon_retranslocation
        )
        if step == ECM_STEP:
            ecm_active_no3 = step_active_no3
            ecm_active_nh4 = step_active_nh4
        else:
            am_active_no3 = step_active_no3
            am_active_nh4 = step_active_nh4

    nitrogen_active_no3 = fortran_divide(n_active_no3_total, dt)
    nitrogen_active_nh4 = fortran_divide(n_active_nh4_total, dt)
    nitrogen_nonmyc_no3 = fortran_divide(n_nonmyc_no3_total, dt)
    nitrogen_nonmyc_nh4 = fortran_divide(n_nonmyc_nh4_total, dt)
    nitrogen_fixation = fortran_divide(n_fix_total, dt)
    nitrogen_paid_retrans = fortran_divide(n_paid_retrans_total, dt)
    nitrogen_free_retrans = fortran_divide(n_free_retrans_total, dt)
    nitrogen_retrans = fortran_add(nitrogen_paid_retrans, nitrogen_free_retrans)
    nitrogen_nonmyc = fortran_add(nitrogen_nonmyc_no3, nitrogen_nonmyc_nh4)
    nitrogen_ecm_no3 = fortran_divide(ecm_active_no3, dt)
    nitrogen_ecm_nh4 = fortran_divide(ecm_active_nh4, dt)
    nitrogen_am_no3 = fortran_divide(am_active_no3, dt)
    nitrogen_am_nh4 = fortran_divide(am_active_nh4, dt)
    nitrogen_total_uptake = _left_sum(
        nitrogen_active_no3,
        nitrogen_active_nh4,
        nitrogen_nonmyc_no3,
        nitrogen_nonmyc_nh4,
        nitrogen_fixation,
        zero,
        nitrogen_paid_retrans,
        nitrogen_free_retrans,
    )
    nitrogen_active = _left_sum(
        nitrogen_active_no3,
        nitrogen_active_nh4,
        nitrogen_nonmyc_no3,
        nitrogen_nonmyc_nh4,
    )
    soil_nitrogen_to_plant = _left_sum(
        nitrogen_active_no3,
        nitrogen_active_nh4,
        nitrogen_nonmyc_no3,
        nitrogen_nonmyc_nh4,
        nitrogen_fixation,
        zero,
    )
    layer_volume_time = fortran_multiply(
        soil_layer_thickness[:NUMBER_OF_FUN_LEVELS], dt
    )
    nitrate_flux_level = fortran_divide(
        fortran_add(vertical_active_no3, vertical_nonmyc_no3),
        layer_volume_time,
    )
    ammonium_flux_level = fortran_divide(
        fortran_add(vertical_active_nh4, vertical_nonmyc_nh4),
        layer_volume_time,
    )
    nitrate_flux_full = jnp.zeros(
        (NUMBER_OF_DECOMPOSITION_LEVELS,), dtype=jnp.float64
    ).at[:NUMBER_OF_FUN_LEVELS].set(nitrate_flux_level)
    ammonium_flux_full = jnp.zeros_like(nitrate_flux_full).at[
        :NUMBER_OF_FUN_LEVELS
    ].set(ammonium_flux_level)

    carbon_active_no3 = fortran_divide(c_active_no3_total, dt)
    carbon_active_nh4 = fortran_divide(c_active_nh4_total, dt)
    carbon_nonmyc_no3 = fortran_divide(c_nonmyc_no3_total, dt)
    carbon_nonmyc_nh4 = fortran_divide(c_nonmyc_nh4_total, dt)
    carbon_fixation = fortran_divide(c_fix_total, dt)
    carbon_retrans = fortran_divide(c_retrans_total, dt)
    soil_carbon_change = fortran_divide(
        _left_sum(
            c_active_no3_total,
            c_active_nh4_total,
            c_nonmyc_no3_total,
            c_nonmyc_nh4_total,
            c_fix_total,
        ),
        dt,
    )
    soil_carbon_change = fortran_add(
        soil_carbon_change, carbon_retrans
    )
    soil_carbon_change = fortran_add(
        soil_carbon_change, fortran_divide(c_burned_total, dt)
    )
    carbon_growth = fortran_add(
        fortran_multiply(
            fortran_add(nitrogen_total_uptake, -nitrogen_free_retrans),
            plant_cn,
        ),
        fortran_divide(c_excess_total, dt),
    )
    carbon_active = _left_sum(
        carbon_active_no3,
        carbon_active_nh4,
        carbon_nonmyc_no3,
        carbon_nonmyc_nh4,
    )
    carbon_nonmyc = fortran_add(carbon_nonmyc_no3, carbon_nonmyc_nh4)

    nitrogen_fluxes = FunNitrogenFluxes(
        nitrogen_active_no3,
        nitrogen_active_nh4,
        nitrogen_active,
        nitrogen_nonmyc_no3,
        nitrogen_nonmyc_nh4,
        nitrogen_nonmyc,
        nitrogen_ecm_no3,
        nitrogen_ecm_nh4,
        fortran_add(nitrogen_ecm_no3, nitrogen_ecm_nh4),
        nitrogen_am_no3,
        nitrogen_am_nh4,
        fortran_add(nitrogen_am_no3, nitrogen_am_nh4),
        zero,
        nitrogen_fixation,
        nitrogen_retrans,
        nitrogen_total_uptake,
        nitrogen_paid_retrans,
        nitrogen_free_retrans,
        soil_nitrogen_to_plant,
        nitrate_flux_full,
        ammonium_flux_full,
        fortran_divide(storage_nitrogen_demand, dt),
    )
    carbon_fluxes = FunCarbonFluxes(
        carbon_active_no3,
        carbon_active_nh4,
        carbon_active,
        carbon_nonmyc_no3,
        carbon_nonmyc_nh4,
        carbon_nonmyc,
        carbon_fixation,
        carbon_retrans,
        soil_carbon_change,
        carbon_growth,
        fortran_divide(c_burned_total, dt),
        soil_carbon_change,
    )
    diagnostic_values = (
        lax.cond(
            carbon_fixation > zero,
            lambda unused: fortran_divide(nitrogen_fixation, carbon_fixation),
            lambda unused: special_value,
            operand=None,
        ),
        lax.cond(
            carbon_active > zero,
            lambda unused: fortran_divide(nitrogen_active, carbon_active),
            lambda unused: special_value,
            operand=None,
        ),
        lax.cond(
            carbon_retrans > zero,
            lambda unused: fortran_divide(nitrogen_retrans, carbon_retrans),
            lambda unused: special_value,
            operand=None,
        ),
        lax.cond(
            available_carbon > zero,
            lambda unused: fortran_divide(soil_carbon_change, available_carbon),
            lambda unused: special_value,
            operand=None,
        ),
    )
    return _PatchAllocation(new_state, nitrogen_fluxes, carbon_fluxes, diagnostic_values)


def _validate_shapes(
    inputs: FunAllocationInputs,
    state: FunAllocationState,
    parameters: FunAllocationParameters,
) -> None:
    patch_count = inputs.soil_patch.shape[0]
    if inputs.soil_patch.ndim != 1:
        raise ValueError("soil_patch must be a vector")
    if inputs.patch_column_index.shape != (patch_count,):
        raise ValueError("patch_column_index must contain every patch")
    for name in FunAllocationInputs._fields[2:20]:
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    if inputs.root_carbon_fraction_patch_level.shape != (
        patch_count,
        NUMBER_OF_DECOMPOSITION_LEVELS,
    ):
        raise ValueError("root_carbon_fraction_patch_level must have shape (patch, 25)")
    column_count = inputs.soil_temperature_column_level.shape[0]
    for name in FunAllocationInputs._fields[21:]:
        if jnp.shape(getattr(inputs, name)) != (
            column_count,
            NUMBER_OF_DECOMPOSITION_LEVELS,
        ):
            raise ValueError(f"{name} must have shape (column, 25)")
    for name in FunAllocationState._fields:
        if jnp.shape(getattr(state, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    pft_count = parameters.leaf_carbon_nitrogen_ratio_pft.shape[0]
    for name in FunAllocationParameters._fields[:20]:
        if jnp.shape(getattr(parameters, name)) != (pft_count,):
            raise ValueError(f"{name} must contain every PFT")
    if parameters.soil_layer_thickness.shape != (
        NUMBER_OF_DECOMPOSITION_LEVELS,
    ):
        raise ValueError("soil_layer_thickness must contain 25 levels")


def _prepare_fun_patches(inputs, state, parameters):

    _validate_shapes(inputs, state, parameters)
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    pft_count = parameters.leaf_carbon_nitrogen_ratio_pft.shape[0]
    column_count = inputs.soil_temperature_column_level.shape[0]
    pft = jnp.clip(
        jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32),
        jnp.int32(0),
        jnp.int32(pft_count - 1),
    )
    column = jnp.clip(
        jnp.asarray(inputs.patch_column_index, dtype=jnp.int32),
        jnp.int32(0),
        jnp.int32(column_count - 1),
    )
    first_crop = jnp.asarray(parameters.first_crop_pft, dtype=jnp.int32)
    patch_inputs = (
        inputs.available_carbon_patch,
        inputs.plant_nitrogen_demand_patch,
        inputs.leaf_carbon_patch,
        inputs.leaf_carbon_storage_patch,
        inputs.fine_root_carbon_patch,
        inputs.leaf_nitrogen_patch,
        inputs.leaf_nitrogen_storage_patch,
        inputs.leaf_nitrogen_storage_to_transfer_patch,
        inputs.fine_root_nitrogen_patch,
        inputs.live_stem_nitrogen_patch,
        inputs.live_coarse_root_nitrogen_patch,
        inputs.leaf_carbon_to_litter_fun_patch,
        inputs.leaf_carbon_storage_to_transfer_patch,
        inputs.carbon_allometry_patch,
        inputs.nitrogen_allometry_patch,
        inputs.onset_flag_patch,
        inputs.offset_flag_patch,
        inputs.root_carbon_fraction_patch_level,
    )
    patch_state = tuple(getattr(state, name) for name in FunAllocationState._fields)
    patch_parameters = (
        parameters.leaf_carbon_nitrogen_ratio_pft[pft],
        jnp.asarray(parameters.seasonal_deciduous_pft[pft], dtype=jnp.bool_),
        jnp.asarray(parameters.stress_deciduous_pft[pft], dtype=jnp.bool_),
        parameters.c3_photosynthesis_pft[pft],
        parameters.fixation_a_pft[pft],
        parameters.fixation_b_pft[pft],
        parameters.fixation_c_pft[pft],
        parameters.fixation_scale_pft[pft],
        parameters.am_carbon_cost_pft[pft],
        parameters.am_nitrogen_cost_pft[pft],
        parameters.ecm_carbon_cost_pft[pft],
        parameters.ecm_nitrogen_cost_pft[pft],
        parameters.nonmycorrhizal_carbon_cost_pft[pft],
        parameters.nonmycorrhizal_nitrogen_cost_pft[pft],
        parameters.ecm_fraction_pft[pft],
        parameters.growth_respiration_fraction_pft[pft],
        parameters.flexible_cn_cost_threshold_pft[pft],
        parameters.flexible_cn_cost_shape_pft[pft],
        parameters.flexible_cn_maximum_offset_pft[pft],
        parameters.fixer_fraction_pft[pft],
        pft >= first_crop,
    )
    column_inputs = (
        inputs.soil_temperature_column_level[column],
        inputs.soil_liquid_water_column_level[column],
        inputs.maximum_nitrate_uptake_column_level[column],
        inputs.maximum_ammonium_uptake_column_level[column],
    )
    scalar_parameters = (
        parameters.soil_layer_thickness,
        jnp.asarray(parameters.time_step_seconds, dtype=jnp.float64),
        jnp.asarray(parameters.leaf_offset_days, dtype=jnp.float64),
        jnp.asarray(parameters.freezing_temperature, dtype=jnp.float64),
        jnp.asarray(parameters.small_value, dtype=jnp.float64),
        jnp.asarray(parameters.special_value, dtype=jnp.float64),
        jnp.asarray(parameters.large_cost, dtype=jnp.float64),
        jnp.asarray(parameters.use_flexible_cn, dtype=jnp.bool_),
    )

    return (
        (soil_patch, patch_inputs, patch_state, patch_parameters, column_inputs),
        scalar_parameters,
    )


def _allocate_fun_patch(
    active,
    local_inputs,
    local_state,
    local_parameters,
    local_column,
    scalar_parameters,
):
    def calculate(_):
        return _active_patch(
            local_inputs,
            local_state,
            local_parameters,
            local_column,
            scalar_parameters,
        )

    def inactive(_):
        zero = jnp.float64(0.0)
        zero_level = jnp.zeros((NUMBER_OF_DECOMPOSITION_LEVELS,), dtype=jnp.float64)
        inactive_state = FunAllocationState(*local_state)
        nitrogen = FunNitrogenFluxes(
            *(zero for _ in range(19)),
            zero_level,
            zero_level,
            zero,
        )
        carbon = FunCarbonFluxes(*(zero for _ in range(12)))
        diagnostics = (zero, zero, zero, zero)
        return _PatchAllocation(inactive_state, nitrogen, carbon, diagnostics)

    return lax.cond(active, calculate, inactive, operand=None)


def _assemble_fun_allocation(allocated, soil_patch_count):

    diagnostics = FunAllocationDiagnostics(
        allocated.diagnostics[0],
        allocated.diagnostics[1],
        allocated.diagnostics[2],
        allocated.diagnostics[3],
        soil_patch_count,
    )
    return FunAllocationResult(
        allocated.state,
        allocated.nitrogen,
        allocated.carbon,
        diagnostics,
    )


def _carbon_use_efficiency_nitrogen_uptake(
    inputs: FunAllocationInputs,
    state: FunAllocationState,
    parameters: FunAllocationParameters,
) -> FunAllocationResult:

    operands, scalar_parameters = _prepare_fun_patches(inputs, state, parameters)
    allocated = jax.vmap(_allocate_fun_patch, in_axes=(0, 0, 0, 0, 0, None))(
        *operands,
        scalar_parameters,
    )
    return _assemble_fun_allocation(allocated, jnp.sum(operands[0], dtype=jnp.int32))


@jax.custom_jvp
def _gpu_fun_allocation(inputs, state, parameters):
    if _materialized_fun.get() and inputs.soil_patch.size >= 4096:
        from clm_jax.physics.cn_fun_gpu import sorted_fun_allocation

        return sorted_fun_allocation(inputs, state, parameters)
    return _carbon_use_efficiency_nitrogen_uptake(inputs, state, parameters)


@_gpu_fun_allocation.defjvp
def _gpu_fun_allocation_jvp(primals, tangents):
    from clm_jax.physics.cn_fun import _retranslocation_fused_gpu

    fused = _retranslocation_fused_gpu.set(False)
    materialized = _materialized_fun.set(False)
    try:
        return jax.jvp(_carbon_use_efficiency_nitrogen_uptake, primals, tangents)
    finally:
        _materialized_fun.reset(materialized)
        _retranslocation_fused_gpu.reset(fused)


def carbon_use_efficiency_nitrogen_uptake(
    inputs: FunAllocationInputs,
    state: FunAllocationState,
    parameters: FunAllocationParameters,
) -> FunAllocationResult:
    if jax.default_backend() == "gpu":
        return _gpu_fun_allocation(inputs, state, parameters)
    return _carbon_use_efficiency_nitrogen_uptake(inputs, state, parameters)


__all__ = [
    "AM_STEP",
    "ECM_STEP",
    "NUMBER_OF_FUN_LEVELS",
    "FunAllocationDiagnostics",
    "FunAllocationInputs",
    "FunAllocationParameters",
    "FunAllocationResult",
    "FunAllocationState",
    "FunCarbonFluxes",
    "FunNitrogenFluxes",
    "carbon_use_efficiency_nitrogen_uptake",
]
