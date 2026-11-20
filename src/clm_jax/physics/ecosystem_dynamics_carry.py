
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.physics.cn_state_update2 import CnStateUpdate2State
from clm_jax.physics.cn_zero import CnTransientFluxState
from clm_jax.physics.soil_biogeochem_n_state_update1 import (
    SoilBiogeochemNStateUpdate1State,
)


class CropAccumulatorCarry(NamedTuple):

    crop_live_patch: jnp.ndarray
    growing_degree_days_since_planting_patch: jnp.ndarray
    vernalization_factor_patch: jnp.ndarray


class EcosystemDynamicsPostDrainageCarry(NamedTuple):

    carbon_nitrogen_state: CnStateUpdate2State
    mineral_nitrogen_state: SoilBiogeochemNStateUpdate1State
    cn_fluxes: CnTransientFluxState
    total_product_loss_gridcell: jnp.ndarray
    total_burned_area_fraction_column: jnp.ndarray
    harvest_date_patch: jnp.ndarray
    crop_accumulators: CropAccumulatorCarry
    excess_maintenance_respiration_carbon_loss_patch: jnp.ndarray
    oxygen_decomposition_scalar_column_level: jnp.ndarray
    annual_sum_npp_patch: jnp.ndarray
    temporary_sum_npp_patch: jnp.ndarray


__all__ = ["CropAccumulatorCarry", "EcosystemDynamicsPostDrainageCarry"]
