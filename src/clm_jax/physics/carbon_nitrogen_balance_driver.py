
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.physics.carbon_nitrogen_balance_check import (
    DEFAULT_CARBON_NITROGEN_BALANCE_CHECK_PARAMETERS,
    CarbonNitrogenBalanceCheckInputs,
    CarbonNitrogenBalanceCheckParameters,
    CarbonNitrogenBalanceCheckResult,
    CarbonNitrogenBalanceCheckState,
    carbon_nitrogen_balance_check,
)
from clm_jax.physics.carbon_nitrogen_balance_initialization import (
    CarbonNitrogenBalanceInitializationInputs,
    CarbonNitrogenBalanceInitializationResult,
    CarbonNitrogenStateSummaries,
)
from clm_jax.physics.cn_zero import get_cn_flux_field
from clm_jax.physics.ecosystem_dynamics_post_drainage import (
    EcosystemDynamicsPostDrainageResult,
)


class CarbonNitrogenBalanceCarry(NamedTuple):

    soil_column: jnp.ndarray
    state: CarbonNitrogenBalanceCheckState


class CarbonNitrogenBalanceDriverResult(NamedTuple):

    inputs: CarbonNitrogenBalanceCheckInputs
    balance: CarbonNitrogenBalanceCheckResult


def carbon_nitrogen_balance_carry(
    inputs: CarbonNitrogenBalanceInitializationInputs,
    initialization: CarbonNitrogenBalanceInitializationResult,
) -> CarbonNitrogenBalanceCarry:

    beginning_carbon = initialization.state.beginning_carbon_balance_column
    beginning_nitrogen = initialization.state.beginning_nitrogen_balance_column
    return CarbonNitrogenBalanceCarry(
        soil_column=jnp.asarray(inputs.soil_column, dtype=jnp.bool_),
        state=CarbonNitrogenBalanceCheckState(
            beginning_carbon_balance_column=beginning_carbon,
            ending_carbon_balance_column=jnp.full_like(beginning_carbon, jnp.nan),
            beginning_nitrogen_balance_column=beginning_nitrogen,
            ending_nitrogen_balance_column=jnp.full_like(beginning_nitrogen, jnp.nan),
        ),
    )


def carbon_nitrogen_balance_carry_from_summaries(
    inputs: CarbonNitrogenBalanceInitializationInputs,
    summaries: CarbonNitrogenStateSummaries,
) -> CarbonNitrogenBalanceCarry:

    beginning_carbon = jnp.asarray(summaries.total_carbon_column, dtype=jnp.float64)
    beginning_nitrogen = jnp.asarray(
        summaries.total_nitrogen_column,
        dtype=jnp.float64,
    )
    column_count = jnp.asarray(inputs.soil_column).shape[0]
    if beginning_carbon.shape != (column_count,):
        raise ValueError("C/N summaries carbon column axis disagrees with topology")
    if beginning_nitrogen.shape != (column_count,):
        raise ValueError("C/N summaries nitrogen column axis disagrees with topology")
    return CarbonNitrogenBalanceCarry(
        soil_column=jnp.asarray(inputs.soil_column, dtype=jnp.bool_),
        state=CarbonNitrogenBalanceCheckState(
            beginning_carbon_balance_column=beginning_carbon,
            ending_carbon_balance_column=jnp.full_like(beginning_carbon, jnp.nan),
            beginning_nitrogen_balance_column=beginning_nitrogen,
            ending_nitrogen_balance_column=jnp.full_like(
                beginning_nitrogen,
                jnp.nan,
            ),
        ),
    )


def assemble_carbon_nitrogen_balance_inputs(
    carry: CarbonNitrogenBalanceCarry,
    post_drainage: EcosystemDynamicsPostDrainageResult,
    *,
    steps_since_restart: int | jnp.ndarray,
) -> CarbonNitrogenBalanceCheckInputs:

    summaries = post_drainage.state.state_summaries
    fluxes = post_drainage.state.biogeochemistry.cn_fluxes

    def flux(name: str) -> jnp.ndarray:
        return get_cn_flux_field(fluxes, name)

    nitrate_leaching = flux("soil_nitrogen.smin_no3_leached_col")
    return CarbonNitrogenBalanceCheckInputs(
        steps_since_restart=jnp.asarray(steps_since_restart, dtype=jnp.int32),
        soil_column=carry.soil_column,
        total_carbon_column=summaries.total_carbon_column,
        total_nitrogen_column=summaries.total_nitrogen_column,
        gross_primary_production_column=flux("vegetation_carbon.gpp_col"),
        ecosystem_respiration_column=flux("vegetation_carbon.er_col"),
        fire_carbon_loss_column=flux("vegetation_carbon.fire_closs_col"),
        harvest_excess_respiration_carbon_loss_column=flux(
            "vegetation_carbon.hrv_xsmrpool_to_atm_col"
        ),
        excess_respiration_carbon_loss_column=flux("vegetation_carbon.xsmrpool_to_atm_col"),
        wood_harvest_carbon_column=flux("vegetation_carbon.wood_harvestc_col"),
        grain_to_crop_product_carbon_column=flux("vegetation_carbon.grainc_to_cropprodc_col"),
        soil_organic_carbon_leaching_column=flux("soil_carbon.som_c_leached_col"),
        nitrogen_deposition_column=flux("soil_nitrogen.ndep_to_sminn_col"),
        nitrogen_fixation_column=flux("soil_nitrogen.nfix_to_sminn_col"),
        free_living_nitrogen_fixation_column=flux("soil_nitrogen.ffix_to_sminn_col"),
        fertilizer_nitrogen_column=flux("soil_nitrogen.fert_to_sminn_col"),
        soybean_nitrogen_fixation_column=flux("soil_nitrogen.soyfixn_to_sminn_col"),
        supplemental_nitrogen_column=flux("soil_nitrogen.supplement_to_sminn_col"),
        denitrification_column=flux("soil_nitrogen.denit_col"),
        mineral_nitrogen_leaching_column=jnp.zeros_like(nitrate_leaching),
        nitrate_leaching_column=nitrate_leaching,
        nitrate_runoff_column=flux("soil_nitrogen.smin_no3_runoff_col"),
        nitrification_n2o_loss_column=flux("soil_nitrogen.f_n2o_nit_col"),
        soil_organic_nitrogen_leaching_column=flux("soil_nitrogen.som_n_leached_col"),
        fire_nitrogen_loss_column=flux("vegetation_nitrogen.fire_nloss_col"),
        wood_harvest_nitrogen_column=flux("vegetation_nitrogen.wood_harvestn_col"),
        grain_to_crop_product_nitrogen_column=flux("vegetation_nitrogen.grainn_to_cropprodn_col"),
    )


def carbon_nitrogen_balance_driver_step(
    carry: CarbonNitrogenBalanceCarry,
    post_drainage: EcosystemDynamicsPostDrainageResult,
    *,
    steps_since_restart: int | jnp.ndarray,
    parameters: CarbonNitrogenBalanceCheckParameters = (
        DEFAULT_CARBON_NITROGEN_BALANCE_CHECK_PARAMETERS
    ),
) -> CarbonNitrogenBalanceDriverResult:

    inputs = assemble_carbon_nitrogen_balance_inputs(
        carry,
        post_drainage,
        steps_since_restart=steps_since_restart,
    )
    balance = carbon_nitrogen_balance_check(
        inputs,
        carry.state,
        parameters=parameters,
    )
    return CarbonNitrogenBalanceDriverResult(inputs=inputs, balance=balance)


__all__ = [
    "CarbonNitrogenBalanceCarry",
    "CarbonNitrogenBalanceDriverResult",
    "assemble_carbon_nitrogen_balance_inputs",
    "carbon_nitrogen_balance_carry",
    "carbon_nitrogen_balance_carry_from_summaries",
    "carbon_nitrogen_balance_driver_step",
]
