
from __future__ import annotations

from contextvars import ContextVar
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

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
from clm_jax.physics.numerics import (
    fortran_hardware_power as fortran_power,
)

_retranslocation_early_exit = ContextVar("clm_fun_retranslocation_early_exit", default=False)
_retranslocation_fused_gpu = ContextVar("clm_fun_retranslocation_fused_gpu", default=True)
_FUN_PATCH_AXIS = "clm_fun_retranslocation_patch"


class FunRetranslocationResult(NamedTuple):

    carbon_spent: jnp.ndarray
    carbon_accounted_for_growth: jnp.ndarray
    free_nitrogen_retranslocated: jnp.ndarray
    paid_nitrogen_retranslocated: jnp.ndarray
    iteration_count: jnp.ndarray


def fun_fixation_cost(
    fixer: jnp.ndarray,
    a_fix: jnp.ndarray,
    b_fix: jnp.ndarray,
    c_fix: jnp.ndarray,
    coarse_root_fraction: jnp.ndarray,
    s_fix: jnp.ndarray,
    soil_temperature_celsius: jnp.ndarray,
    *,
    large_cost: float | jnp.ndarray = 1.0e9,
) -> jnp.ndarray:

    (
        fixer,
        a_fix,
        b_fix,
        c_fix,
        coarse_root_fraction,
        s_fix,
        soil_temperature_celsius,
        large_cost,
    ) = jnp.broadcast_arrays(
        jnp.asarray(fixer, dtype=jnp.int32),
        jnp.asarray(a_fix, dtype=jnp.float64),
        jnp.asarray(b_fix, dtype=jnp.float64),
        jnp.asarray(c_fix, dtype=jnp.float64),
        jnp.asarray(coarse_root_fraction, dtype=jnp.float64),
        jnp.asarray(s_fix, dtype=jnp.float64),
        jnp.asarray(soil_temperature_celsius, dtype=jnp.float64),
        jnp.asarray(large_cost, dtype=jnp.float64),
    )
    active = (fixer == jnp.int32(1)) & (
        coarse_root_fraction > jnp.float64(1.0e-6)
    )
    safe_c_fix = jnp.where(active, c_fix, jnp.float64(1.0))
    half_temperature = fortran_multiply(
        jnp.float64(0.5), soil_temperature_celsius
    )
    scaled_temperature = fortran_divide(half_temperature, safe_c_fix)
    temperature_shape = fortran_add(jnp.float64(1.0), -scaled_temperature)
    temperature_product = fortran_multiply(b_fix, soil_temperature_celsius)
    exponent = fortran_add(
        a_fix,
        fortran_multiply(temperature_product, temperature_shape),
    )
    denominator = fortran_multiply(jnp.float64(1.25), strict_exp(exponent))
    numerator = fortran_multiply(jnp.float64(-1.0), s_fix)
    numerator = fortran_multiply(numerator, jnp.float64(1.0))
    cost = fortran_divide(numerator, denominator)
    return jnp.where(active, cost, large_cost)


def fun_active_uptake_cost(
    mineral_nitrogen_layer: jnp.ndarray,
    carbon_cost_coefficient: jnp.ndarray,
    nitrogen_cost_coefficient: jnp.ndarray,
    root_carbon_density: jnp.ndarray,
    *,
    small_value: float | jnp.ndarray,
    large_cost: float | jnp.ndarray = 1.0e9,
) -> jnp.ndarray:

    (
        mineral_nitrogen_layer,
        carbon_cost_coefficient,
        nitrogen_cost_coefficient,
        root_carbon_density,
        small_value,
        large_cost,
    ) = jnp.broadcast_arrays(
        jnp.asarray(mineral_nitrogen_layer, dtype=jnp.float64),
        jnp.asarray(carbon_cost_coefficient, dtype=jnp.float64),
        jnp.asarray(nitrogen_cost_coefficient, dtype=jnp.float64),
        jnp.asarray(root_carbon_density, dtype=jnp.float64),
        jnp.asarray(small_value, dtype=jnp.float64),
        jnp.asarray(large_cost, dtype=jnp.float64),
    )
    active = (root_carbon_density > jnp.float64(1.0e-6)) & (
        mineral_nitrogen_layer > small_value
    )
    safe_nitrogen = jnp.where(active, mineral_nitrogen_layer, jnp.float64(1.0))
    safe_root_carbon = jnp.where(active, root_carbon_density, jnp.float64(1.0))
    nitrogen_term = fortran_divide(nitrogen_cost_coefficient, safe_nitrogen)
    carbon_term = fortran_divide(carbon_cost_coefficient, safe_root_carbon)
    return jnp.where(active, fortran_add(nitrogen_term, carbon_term), large_cost)


def fun_nonmycorrhizal_uptake_cost(
    mineral_nitrogen_layer: jnp.ndarray,
    carbon_cost_coefficient: jnp.ndarray,
    nitrogen_cost_coefficient: jnp.ndarray,
    root_carbon_density: jnp.ndarray,
    *,
    small_value: float | jnp.ndarray,
    large_cost: float | jnp.ndarray = 1.0e9,
) -> jnp.ndarray:

    return fun_active_uptake_cost(
        mineral_nitrogen_layer,
        carbon_cost_coefficient,
        nitrogen_cost_coefficient,
        root_carbon_density,
        small_value=small_value,
        large_cost=large_cost,
    )


def fun_retranslocation(
    available_carbon: jnp.ndarray,
    falling_leaf_carbon: jnp.ndarray,
    falling_leaf_nitrogen: jnp.ndarray,
    total_nitrogen_resistance: jnp.ndarray,
    target_leaf_cn_ratio: jnp.ndarray,
    growth_respiration_fraction: jnp.ndarray,
    plant_cn_ratio: jnp.ndarray,
) -> FunRetranslocationResult:

    if _retranslocation_fused_gpu.get() and jax.default_backend() == "gpu":
        from clm_jax.physics.fun_retranslocation_gpu import scalar_retranslocation
        return scalar_retranslocation(
            available_carbon, falling_leaf_carbon, falling_leaf_nitrogen,
            total_nitrogen_resistance, target_leaf_cn_ratio,
            growth_respiration_fraction, plant_cn_ratio,
        )

    (
        available_carbon,
        falling_leaf_carbon,
        falling_leaf_nitrogen,
        total_nitrogen_resistance,
        target_leaf_cn_ratio,
        growth_respiration_fraction,
        plant_cn_ratio,
    ) = jnp.broadcast_arrays(
        jnp.asarray(available_carbon, dtype=jnp.float64),
        jnp.asarray(falling_leaf_carbon, dtype=jnp.float64),
        jnp.asarray(falling_leaf_nitrogen, dtype=jnp.float64),
        jnp.asarray(total_nitrogen_resistance, dtype=jnp.float64),
        jnp.asarray(target_leaf_cn_ratio, dtype=jnp.float64),
        jnp.asarray(growth_respiration_fraction, dtype=jnp.float64),
        jnp.asarray(plant_cn_ratio, dtype=jnp.float64),
    )
    zero = jnp.zeros_like(available_carbon)
    maximum_leaf_cn = fortran_multiply(target_leaf_cn_ratio, jnp.float64(3.0))
    minimum_leaf_cn = fortran_multiply(target_leaf_cn_ratio, jnp.float64(1.5))
    free_retranslocated = jnp.maximum(
        fortran_add(
            falling_leaf_nitrogen,
            -fortran_divide(falling_leaf_carbon, minimum_leaf_cn),
        ),
        zero,
    )
    leaf_nitrogen = fortran_add(
        falling_leaf_nitrogen, -free_retranslocated
    )
    leaf_cn = fortran_divide(falling_leaf_carbon, leaf_nitrogen)
    resorption_coefficient = fortran_divide(
        jnp.float64(1.0), target_leaf_cn_ratio
    )
    inverse_leaf_cn = fortran_divide(jnp.float64(1.0), leaf_cn)
    retranslocation_cost = fortran_divide(
        resorption_coefficient,
        fortran_power(inverse_leaf_cn, jnp.float64(1.3)),
    )

    initial = (
        leaf_nitrogen,
        leaf_cn,
        retranslocation_cost,
        available_carbon,
        zero,
        zero,
        zero,
        jnp.zeros_like(available_carbon, dtype=jnp.bool_),
        jnp.zeros_like(available_carbon, dtype=jnp.int32),
    )

    def active_patches(carry):
        return (
            (~carry[7])
            & (carry[2] < total_nitrogen_resistance)
            & (carry[0] >= jnp.float64(0.0))
            & (available_carbon > jnp.float64(0.0))
        )

    def body(_, carry):
        (
            current_leaf_n,
            current_leaf_cn,
            current_cost,
            carbon_remaining,
            total_carbon_spent,
            total_carbon_accounted,
            total_paid_n,
            exited,
            iteration_count,
        ) = carry
        active = active_patches(carry)

        next_cn_target = fortran_add(current_leaf_cn, jnp.float64(1.0))
        nitrogen_at_next_cn = fortran_divide(
            falling_leaf_carbon, next_cn_target
        )
        nitrogen_increment = fortran_add(current_leaf_n, -nitrogen_at_next_cn)
        carbon_spent = fortran_multiply(current_cost, nitrogen_increment)
        carbon_spent = jnp.minimum(carbon_remaining, carbon_spent)
        nitrogen_extracted = fortran_divide(carbon_spent, current_cost)
        nitrogen_extracted = jnp.minimum(current_leaf_n, nitrogen_extracted)
        growth_carbon = fortran_multiply(nitrogen_extracted, plant_cn_ratio)
        growth_carbon = fortran_multiply(
            growth_carbon,
            fortran_add(jnp.float64(1.0), growth_respiration_fraction),
        )

        candidate_leaf_n = fortran_add(current_leaf_n, -nitrogen_extracted)
        positive_leaf_n = candidate_leaf_n > jnp.float64(0.0)
        safe_leaf_n = jnp.where(positive_leaf_n, candidate_leaf_n, jnp.float64(1.0))
        candidate_leaf_cn = fortran_divide(falling_leaf_carbon, safe_leaf_n)
        candidate_inverse_cn = fortran_divide(
            jnp.float64(1.0), candidate_leaf_cn
        )
        candidate_cost = fortran_divide(
            resorption_coefficient,
            fortran_power(candidate_inverse_cn, jnp.float64(1.3)),
        )
        candidate_leaf_cn = jnp.where(
            positive_leaf_n, candidate_leaf_cn, current_leaf_cn
        )
        candidate_cost = jnp.where(positive_leaf_n, candidate_cost, current_cost)

        candidate_total_spent = fortran_add(total_carbon_spent, carbon_spent)
        candidate_total_accounted = fortran_add(
            total_carbon_accounted, growth_carbon
        )
        candidate_total_paid = fortran_add(total_paid_n, nitrogen_extracted)
        candidate_remaining = fortran_add(
            fortran_add(carbon_remaining, -carbon_spent), -growth_carbon
        )
        depleted_carbon = candidate_remaining <= jnp.float64(0.0)
        candidate_total_spent = jnp.where(
            depleted_carbon,
            fortran_add(candidate_total_spent, candidate_remaining),
            candidate_total_spent,
        )
        candidate_count = iteration_count + jnp.int32(1)
        candidate_exited = (
            (~positive_leaf_n)
            | depleted_carbon
            | (candidate_leaf_cn >= maximum_leaf_cn)
            | (candidate_count >= jnp.int32(150))
        )

        def choose(old, new):
            return jnp.where(active, new, old)

        return (
            choose(current_leaf_n, candidate_leaf_n),
            choose(current_leaf_cn, candidate_leaf_cn),
            choose(current_cost, candidate_cost),
            choose(carbon_remaining, candidate_remaining),
            choose(total_carbon_spent, candidate_total_spent),
            choose(total_carbon_accounted, candidate_total_accounted),
            choose(total_paid_n, candidate_total_paid),
            jnp.where(active, candidate_exited, exited),
            jnp.where(active, candidate_count, iteration_count),
        )

    def iteration(index, carry):
        return lax.cond(
            jnp.any(active_patches(carry)),
            lambda current: body(index, current),
            lambda current: current,
            carry,
        )

    if _retranslocation_early_exit.get():
        def condition(carry):
            index, values = carry
            enabled = jnp.any(active_patches(values))
            try:
                enabled = lax.pmax(enabled.astype(jnp.int32), _FUN_PATCH_AXIS) > 0
            except NameError:
                pass
            return (index < jnp.int32(150)) & enabled

        def advance(carry):
            index, values = carry
            return index + jnp.int32(1), body(index, values)

        final = lax.while_loop(condition, advance, (jnp.int32(0), initial))[1]
    else:
        from clm_jax.physics.converged_iteration import checkpointed_fori_loop

        final = checkpointed_fori_loop(0, 150, iteration, initial)
    return FunRetranslocationResult(
        carbon_spent=final[4],
        carbon_accounted_for_growth=final[5],
        free_nitrogen_retranslocated=free_retranslocated,
        paid_nitrogen_retranslocated=final[6],
        iteration_count=final[8],
    )


__all__ = [
    "FunRetranslocationResult",
    "fun_active_uptake_cost",
    "fun_fixation_cost",
    "fun_nonmycorrhizal_uptake_cost",
    "fun_retranslocation",
]
