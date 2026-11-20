
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import core as jax_core

from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    get_cn_flux_field,
    set_cn_flux_field,
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


class PlantNutrientCompetitionInputs(NamedTuple):

    soil_patch: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    crop_live_patch: jnp.ndarray
    annual_net_primary_productivity_patch: jnp.ndarray
    leaf_allocation_patch: jnp.ndarray
    stem_allocation_patch: jnp.ndarray
    carbon_allometry_patch: jnp.ndarray
    root_allocation_patch: jnp.ndarray
    reproductive_allocation_patch: jnp.ndarray
    plant_nitrogen_pool_patch: jnp.ndarray
    fun_soil_nitrogen_uptake_patch: jnp.ndarray
    sunlit_leaf_area_index_patch: jnp.ndarray
    shaded_leaf_area_index_patch: jnp.ndarray
    leaf_carbon_storage_patch: jnp.ndarray
    leaf_nitrogen_storage_patch: jnp.ndarray


class PlantNutrientCompetitionState(NamedTuple):

    cn_fluxes: CnTransientFluxState
    plant_carbon_allocation_patch: jnp.ndarray
    plant_nitrogen_allocation_patch: jnp.ndarray
    actual_storage_leaf_carbon_nitrogen_ratio_patch: jnp.ndarray


class PlantNutrientCompetitionParameters(NamedTuple):

    woody_pft: jnp.ndarray
    fine_root_to_leaf_carbon_pft: jnp.ndarray
    coarse_root_to_stem_carbon_pft: jnp.ndarray
    stem_to_leaf_carbon_pft: jnp.ndarray
    live_wood_fraction_pft: jnp.ndarray
    leaf_carbon_nitrogen_ratio_pft: jnp.ndarray
    fine_root_carbon_nitrogen_ratio_pft: jnp.ndarray
    live_wood_carbon_nitrogen_ratio_pft: jnp.ndarray
    dead_wood_carbon_nitrogen_ratio_pft: jnp.ndarray
    current_growth_fraction_pft: jnp.ndarray
    grain_carbon_nitrogen_ratio_pft: jnp.ndarray
    growth_respiration_fraction_pft: jnp.ndarray
    current_growth_respiration_fraction_pft: jnp.ndarray
    evergreen_pft: jnp.ndarray
    first_crop_pft: int | jnp.ndarray = 17
    time_step_seconds: float | jnp.ndarray = 1800.0
    use_fun: bool = True
    downregulation: bool = False
    carbon_nitrogen_partition_option: int = 0
    carbon_nitrogen_residual_option: int = 0
    carbon_respiration_option: int = 0
    missing_value: float | jnp.ndarray = 1.0e36


_DEFAULT_PFT_COUNT = 79
_DEFAULT_ONE = np.ones(_DEFAULT_PFT_COUNT, dtype=np.float64)
_DEFAULT_ZERO = np.zeros(_DEFAULT_PFT_COUNT, dtype=np.float64)
DEFAULT_PLANT_NUTRIENT_COMPETITION_PARAMETERS = (
    PlantNutrientCompetitionParameters(
        woody_pft=_DEFAULT_ZERO,
        fine_root_to_leaf_carbon_pft=_DEFAULT_ONE,
        coarse_root_to_stem_carbon_pft=_DEFAULT_ONE,
        stem_to_leaf_carbon_pft=_DEFAULT_ONE,
        live_wood_fraction_pft=np.full(_DEFAULT_PFT_COUNT, 0.1),
        leaf_carbon_nitrogen_ratio_pft=np.full(_DEFAULT_PFT_COUNT, 25.0),
        fine_root_carbon_nitrogen_ratio_pft=np.full(_DEFAULT_PFT_COUNT, 42.0),
        live_wood_carbon_nitrogen_ratio_pft=np.full(_DEFAULT_PFT_COUNT, 50.0),
        dead_wood_carbon_nitrogen_ratio_pft=np.full(_DEFAULT_PFT_COUNT, 500.0),
        current_growth_fraction_pft=np.full(_DEFAULT_PFT_COUNT, 0.5),
        grain_carbon_nitrogen_ratio_pft=np.full(_DEFAULT_PFT_COUNT, 20.0),
        growth_respiration_fraction_pft=np.full(_DEFAULT_PFT_COUNT, 0.25),
        current_growth_respiration_fraction_pft=_DEFAULT_ZERO,
        evergreen_pft=_DEFAULT_ZERO,
    )
)


class PlantNutrientCompetitionDiagnostics(NamedTuple):
    updated_patch_count: jnp.ndarray
    woody_patch_count: jnp.ndarray
    live_crop_patch_count: jnp.ndarray
    dormant_crop_patch_count: jnp.ndarray


class PlantNutrientCompetitionResult(NamedTuple):
    state: PlantNutrientCompetitionState
    diagnostics: PlantNutrientCompetitionDiagnostics


_CARBON_GENERAL_FIELDS = {
    "vegetation_carbon.cpool_to_leafc_patch": "leaf_current",
    "vegetation_carbon.cpool_to_leafc_storage_patch": "leaf_storage",
    "vegetation_carbon.cpool_to_frootc_patch": "fine_root_current",
    "vegetation_carbon.cpool_to_frootc_storage_patch": "fine_root_storage",
}
_CARBON_WOOD_FIELDS = {
    "vegetation_carbon.cpool_to_livestemc_patch": "live_stem_current",
    "vegetation_carbon.cpool_to_livestemc_storage_patch": "live_stem_storage",
    "vegetation_carbon.cpool_to_deadstemc_patch": "dead_stem_current",
    "vegetation_carbon.cpool_to_deadstemc_storage_patch": "dead_stem_storage",
    "vegetation_carbon.cpool_to_livecrootc_patch": "live_coarse_root_current",
    "vegetation_carbon.cpool_to_livecrootc_storage_patch": (
        "live_coarse_root_storage"
    ),
    "vegetation_carbon.cpool_to_deadcrootc_patch": "dead_coarse_root_current",
    "vegetation_carbon.cpool_to_deadcrootc_storage_patch": (
        "dead_coarse_root_storage"
    ),
}
_CARBON_CROP_FIELDS = {
    "vegetation_carbon.cpool_to_grainc_patch": "grain_current",
    "vegetation_carbon.cpool_to_grainc_storage_patch": "grain_storage",
}
_NITROGEN_GENERAL_FIELDS = {
    "vegetation_nitrogen.npool_to_leafn_patch": "leaf_current",
    "vegetation_nitrogen.npool_to_leafn_storage_patch": "leaf_storage",
    "vegetation_nitrogen.npool_to_frootn_patch": "fine_root_current",
    "vegetation_nitrogen.npool_to_frootn_storage_patch": "fine_root_storage",
}
_NITROGEN_WOOD_FIELDS = {
    "vegetation_nitrogen.npool_to_livestemn_patch": "live_stem_current",
    "vegetation_nitrogen.npool_to_livestemn_storage_patch": (
        "live_stem_storage"
    ),
    "vegetation_nitrogen.npool_to_deadstemn_patch": "dead_stem_current",
    "vegetation_nitrogen.npool_to_deadstemn_storage_patch": (
        "dead_stem_storage"
    ),
    "vegetation_nitrogen.npool_to_livecrootn_patch": (
        "live_coarse_root_current"
    ),
    "vegetation_nitrogen.npool_to_livecrootn_storage_patch": (
        "live_coarse_root_storage"
    ),
    "vegetation_nitrogen.npool_to_deadcrootn_patch": (
        "dead_coarse_root_current"
    ),
    "vegetation_nitrogen.npool_to_deadcrootn_storage_patch": (
        "dead_coarse_root_storage"
    ),
}
_NITROGEN_CROP_FIELDS = {
    "vegetation_nitrogen.npool_to_grainn_patch": "grain_current",
    "vegetation_nitrogen.npool_to_grainn_storage_patch": "grain_storage",
}
_CARBON_RESPIRATION_FIELDS = (
    "vegetation_carbon.cpool_to_resp_patch",
    "vegetation_carbon.cpool_to_leafc_resp_patch",
    "vegetation_carbon.cpool_to_leafc_storage_resp_patch",
    "vegetation_carbon.cpool_to_frootc_resp_patch",
    "vegetation_carbon.cpool_to_frootc_storage_resp_patch",
    "vegetation_carbon.cpool_to_livecrootc_resp_patch",
    "vegetation_carbon.cpool_to_livecrootc_storage_resp_patch",
    "vegetation_carbon.cpool_to_livestemc_resp_patch",
    "vegetation_carbon.cpool_to_livestemc_storage_resp_patch",
)


class _OrganFluxes(NamedTuple):
    leaf_current: jnp.ndarray
    leaf_storage: jnp.ndarray
    fine_root_current: jnp.ndarray
    fine_root_storage: jnp.ndarray
    live_stem_current: jnp.ndarray
    live_stem_storage: jnp.ndarray
    dead_stem_current: jnp.ndarray
    dead_stem_storage: jnp.ndarray
    live_coarse_root_current: jnp.ndarray
    live_coarse_root_storage: jnp.ndarray
    dead_coarse_root_current: jnp.ndarray
    dead_coarse_root_storage: jnp.ndarray
    grain_current: jnp.ndarray
    grain_storage: jnp.ndarray


def _validate_active_configuration(
    parameters: PlantNutrientCompetitionParameters,
) -> None:
    values = (
        parameters.use_fun,
        parameters.downregulation,
        parameters.carbon_nitrogen_partition_option,
        parameters.carbon_nitrogen_residual_option,
        parameters.carbon_respiration_option,
    )
    if any(isinstance(value, jax_core.Tracer) for value in values):
        return
    if not bool(parameters.use_fun):
        raise NotImplementedError("non-FUN plant competition is not yet ported")
    if bool(parameters.downregulation):
        raise NotImplementedError("downregulated plant competition is not yet ported")
    if int(parameters.carbon_nitrogen_partition_option) != 0:
        raise NotImplementedError("CN_partition_opt=1 is not yet ported")
    if int(parameters.carbon_nitrogen_residual_option) != 0:
        raise NotImplementedError("CN_residual_opt=1 is not yet ported")
    if int(parameters.carbon_respiration_option) != 0:
        raise NotImplementedError("carbon_resp_opt=1 is not yet ported")


def _validate_shapes(
    inputs: PlantNutrientCompetitionInputs,
    state: PlantNutrientCompetitionState,
    parameters: PlantNutrientCompetitionParameters,
) -> int:
    patch_count = inputs.soil_patch.shape[0]
    patch_shape = (patch_count,)
    for name in PlantNutrientCompetitionInputs._fields:
        if jnp.shape(getattr(inputs, name)) != patch_shape:
            raise ValueError(f"{name} must contain every patch")
    for name in PlantNutrientCompetitionState._fields[1:]:
        if jnp.shape(getattr(state, name)) != patch_shape:
            raise ValueError(f"state {name} must contain every patch")
    if state.cn_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("packed CN flux state has an incompatible patch count")
    pft_shape = jnp.shape(parameters.woody_pft)
    if len(pft_shape) != 1 or pft_shape[0] == 0:
        raise ValueError("plant competition PFT parameters must be nonempty vectors")
    for name in PlantNutrientCompetitionParameters._fields[1:14]:
        if jnp.shape(getattr(parameters, name)) != pft_shape:
            raise ValueError(f"PFT parameter {name} must have shape {pft_shape!r}")
    return patch_count


def _selected(values: jnp.ndarray, pft: jnp.ndarray) -> jnp.ndarray:
    return jnp.take(jnp.asarray(values, dtype=jnp.float64), pft, axis=0)


def _multiply(*values: jnp.ndarray) -> jnp.ndarray:
    result = values[0]
    for value in values[1:]:
        result = fortran_multiply(result, value)
    return result


def _subtract(first: jnp.ndarray, *rest: jnp.ndarray) -> jnp.ndarray:
    result = first
    for value in rest:
        result = fortran_add(result, -value)
    return result


def _bounded_supply(supply: jnp.ndarray, demand: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(jnp.minimum(supply, demand), jnp.float64(0.0))


def _carbon_fluxes(
    new_leaf_carbon: jnp.ndarray,
    current_fraction: jnp.ndarray,
    storage_fraction: jnp.ndarray,
    fine_root_ratio: jnp.ndarray,
    coarse_root_ratio: jnp.ndarray,
    stem_ratio: jnp.ndarray,
    live_wood_fraction: jnp.ndarray,
    grain_ratio: jnp.ndarray,
) -> _OrganFluxes:
    dead_wood_fraction = fortran_add(jnp.float64(1.0), -live_wood_fraction)
    return _OrganFluxes(
        leaf_current=_multiply(new_leaf_carbon, current_fraction),
        leaf_storage=_multiply(new_leaf_carbon, storage_fraction),
        fine_root_current=_multiply(
            new_leaf_carbon, fine_root_ratio, current_fraction
        ),
        fine_root_storage=_multiply(
            new_leaf_carbon, fine_root_ratio, storage_fraction
        ),
        live_stem_current=_multiply(
            new_leaf_carbon,
            stem_ratio,
            live_wood_fraction,
            current_fraction,
        ),
        live_stem_storage=_multiply(
            new_leaf_carbon,
            stem_ratio,
            live_wood_fraction,
            storage_fraction,
        ),
        dead_stem_current=_multiply(
            new_leaf_carbon,
            stem_ratio,
            dead_wood_fraction,
            current_fraction,
        ),
        dead_stem_storage=_multiply(
            new_leaf_carbon,
            stem_ratio,
            dead_wood_fraction,
            storage_fraction,
        ),
        live_coarse_root_current=_multiply(
            new_leaf_carbon,
            coarse_root_ratio,
            stem_ratio,
            live_wood_fraction,
            current_fraction,
        ),
        live_coarse_root_storage=_multiply(
            new_leaf_carbon,
            coarse_root_ratio,
            stem_ratio,
            live_wood_fraction,
            storage_fraction,
        ),
        dead_coarse_root_current=_multiply(
            new_leaf_carbon,
            coarse_root_ratio,
            stem_ratio,
            dead_wood_fraction,
            current_fraction,
        ),
        dead_coarse_root_storage=_multiply(
            new_leaf_carbon,
            coarse_root_ratio,
            stem_ratio,
            dead_wood_fraction,
            storage_fraction,
        ),
        grain_current=_multiply(new_leaf_carbon, grain_ratio, current_fraction),
        grain_storage=_multiply(new_leaf_carbon, grain_ratio, storage_fraction),
    )


def _sequential_nitrogen_fluxes(
    inputs: PlantNutrientCompetitionInputs,
    new_leaf_carbon: jnp.ndarray,
    current_fraction: jnp.ndarray,
    storage_fraction: jnp.ndarray,
    woody: jnp.ndarray,
    crop: jnp.ndarray,
    fine_root_ratio: jnp.ndarray,
    coarse_root_ratio: jnp.ndarray,
    stem_ratio: jnp.ndarray,
    live_wood_fraction: jnp.ndarray,
    grain_ratio: jnp.ndarray,
    leaf_cn: jnp.ndarray,
    fine_root_cn: jnp.ndarray,
    live_wood_cn: jnp.ndarray,
    dead_wood_cn: jnp.ndarray,
    grain_cn: jnp.ndarray,
    time_step: jnp.ndarray,
) -> _OrganFluxes:
    pool_rate = fortran_divide(inputs.plant_nitrogen_pool_patch, time_step)
    current_supply = fortran_multiply(pool_rate, current_fraction)
    storage_supply = fortran_multiply(pool_rate, storage_fraction)
    dead_wood_fraction = fortran_add(jnp.float64(1.0), -live_wood_fraction)
    leaf_demand_base = fortran_divide(new_leaf_carbon, leaf_cn)
    fine_root_demand_base = fortran_divide(
        _multiply(new_leaf_carbon, fine_root_ratio), fine_root_cn
    )
    live_stem_demand_base = fortran_divide(
        _multiply(new_leaf_carbon, stem_ratio, live_wood_fraction),
        live_wood_cn,
    )
    live_coarse_root_demand_base = fortran_divide(
        _multiply(
            new_leaf_carbon,
            coarse_root_ratio,
            stem_ratio,
            live_wood_fraction,
        ),
        live_wood_cn,
    )
    dead_stem_demand_base = fortran_divide(
        _multiply(new_leaf_carbon, stem_ratio, dead_wood_fraction),
        dead_wood_cn,
    )
    dead_coarse_root_demand_base = fortran_divide(
        _multiply(
            new_leaf_carbon,
            coarse_root_ratio,
            stem_ratio,
            dead_wood_fraction,
        ),
        dead_wood_cn,
    )
    grain_demand_base = fortran_divide(
        _multiply(new_leaf_carbon, grain_ratio), grain_cn
    )
    fine_root_current_demand = fortran_multiply(
        fine_root_demand_base, current_fraction
    )
    fine_root_storage_demand = fortran_multiply(
        fine_root_demand_base, storage_fraction
    )
    fine_root_current = _bounded_supply(
        current_supply, fine_root_current_demand
    )
    fine_root_storage = _bounded_supply(
        storage_supply, fine_root_storage_demand
    )

    live_stem_current = _bounded_supply(
        _subtract(current_supply, fine_root_current),
        fortran_multiply(live_stem_demand_base, current_fraction),
    )
    live_stem_storage = _bounded_supply(
        _subtract(storage_supply, fine_root_storage),
        fortran_multiply(live_stem_demand_base, storage_fraction),
    )
    live_coarse_root_current = _bounded_supply(
        _subtract(current_supply, fine_root_current, live_stem_current),
        fortran_multiply(live_coarse_root_demand_base, current_fraction),
    )
    live_coarse_root_storage = _bounded_supply(
        _subtract(storage_supply, fine_root_storage, live_stem_storage),
        fortran_multiply(live_coarse_root_demand_base, storage_fraction),
    )
    dead_stem_current = _bounded_supply(
        _subtract(
            current_supply,
            fine_root_current,
            live_stem_current,
            live_coarse_root_current,
        ),
        fortran_multiply(dead_stem_demand_base, current_fraction),
    )
    dead_stem_storage = _bounded_supply(
        _subtract(
            storage_supply,
            fine_root_storage,
            live_stem_storage,
            live_coarse_root_storage,
        ),
        fortran_multiply(dead_stem_demand_base, storage_fraction),
    )
    dead_coarse_root_current = _bounded_supply(
        _subtract(
            current_supply,
            fine_root_current,
            live_stem_current,
            live_coarse_root_current,
            dead_stem_current,
        ),
        fortran_multiply(dead_coarse_root_demand_base, current_fraction),
    )
    dead_coarse_root_storage = _bounded_supply(
        _subtract(
            storage_supply,
            fine_root_storage,
            live_stem_storage,
            live_coarse_root_storage,
            dead_stem_storage,
        ),
        fortran_multiply(dead_coarse_root_demand_base, storage_fraction),
    )
    grain_current = _bounded_supply(
        _subtract(
            current_supply,
            fine_root_current,
            live_stem_current,
            live_coarse_root_current,
            dead_stem_current,
            dead_coarse_root_current,
        ),
        fortran_multiply(grain_demand_base, current_fraction),
    )
    grain_storage = _bounded_supply(
        _subtract(
            storage_supply,
            fine_root_storage,
            live_stem_storage,
            live_coarse_root_storage,
            dead_stem_storage,
            dead_coarse_root_storage,
        ),
        fortran_multiply(grain_demand_base, storage_fraction),
    )

    nonwoody_leaf_current_supply = _subtract(current_supply, fine_root_current)
    nonwoody_leaf_storage_supply = _subtract(storage_supply, fine_root_storage)
    woody_leaf_current_supply = _subtract(
        current_supply,
        fine_root_current,
        live_stem_current,
        live_coarse_root_current,
        dead_stem_current,
        dead_coarse_root_current,
    )
    woody_leaf_storage_supply = _subtract(
        storage_supply,
        fine_root_storage,
        live_stem_storage,
        live_coarse_root_storage,
        dead_stem_storage,
        dead_coarse_root_storage,
    )
    crop_leaf_current_supply = _subtract(woody_leaf_current_supply, grain_current)
    crop_leaf_storage_supply = _subtract(
        woody_leaf_storage_supply, grain_storage
    )
    leaf_current_supply = jnp.where(
        crop,
        crop_leaf_current_supply,
        jnp.where(woody, woody_leaf_current_supply, nonwoody_leaf_current_supply),
    )
    leaf_storage_supply = jnp.where(
        crop,
        crop_leaf_storage_supply,
        jnp.where(woody, woody_leaf_storage_supply, nonwoody_leaf_storage_supply),
    )
    leaf_current = _bounded_supply(
        leaf_current_supply,
        fortran_multiply(leaf_demand_base, current_fraction),
    )
    leaf_storage = _bounded_supply(
        leaf_storage_supply,
        fortran_multiply(leaf_demand_base, storage_fraction),
    )
    return _OrganFluxes(
        leaf_current=leaf_current,
        leaf_storage=leaf_storage,
        fine_root_current=fine_root_current,
        fine_root_storage=fine_root_storage,
        live_stem_current=live_stem_current,
        live_stem_storage=live_stem_storage,
        dead_stem_current=dead_stem_current,
        dead_stem_storage=dead_stem_storage,
        live_coarse_root_current=live_coarse_root_current,
        live_coarse_root_storage=live_coarse_root_storage,
        dead_coarse_root_current=dead_coarse_root_current,
        dead_coarse_root_storage=dead_coarse_root_storage,
        grain_current=grain_current,
        grain_storage=grain_storage,
    )


def _sum_in_source_order(*values: jnp.ndarray) -> jnp.ndarray:
    result = values[0]
    for value in values[1:]:
        result = fortran_add(result, value)
    return result


def _proportional_nitrogen_fluxes(
    plant_nitrogen_pool_patch: jnp.ndarray,
    new_leaf_carbon: jnp.ndarray,
    current_fraction: jnp.ndarray,
    storage_fraction: jnp.ndarray,
    woody: jnp.ndarray,
    crop: jnp.ndarray,
    fine_root_ratio: jnp.ndarray,
    coarse_root_ratio: jnp.ndarray,
    stem_ratio: jnp.ndarray,
    live_wood_fraction: jnp.ndarray,
    grain_ratio: jnp.ndarray,
    leaf_cn: jnp.ndarray,
    fine_root_cn: jnp.ndarray,
    live_wood_cn: jnp.ndarray,
    dead_wood_cn: jnp.ndarray,
    grain_cn: jnp.ndarray,
    time_step: jnp.ndarray,
) -> _OrganFluxes:

    live_wood_cn = jnp.where(woody | crop, live_wood_cn, 1.0)
    dead_wood_cn = jnp.where(woody | crop, dead_wood_cn, 1.0)
    grain_cn = jnp.where(crop, grain_cn, 1.0)
    dead_wood_fraction = fortran_add(jnp.float64(1.0), -live_wood_fraction)
    leaf_base = fortran_divide(new_leaf_carbon, leaf_cn)
    fine_root_base = fortran_divide(
        _multiply(new_leaf_carbon, fine_root_ratio), fine_root_cn
    )
    live_stem_base = fortran_divide(
        _multiply(new_leaf_carbon, stem_ratio, live_wood_fraction),
        live_wood_cn,
    )
    dead_stem_base = fortran_divide(
        _multiply(new_leaf_carbon, stem_ratio, dead_wood_fraction),
        dead_wood_cn,
    )
    live_coarse_root_base = fortran_divide(
        _multiply(
            new_leaf_carbon,
            coarse_root_ratio,
            stem_ratio,
            live_wood_fraction,
        ),
        live_wood_cn,
    )
    dead_coarse_root_base = fortran_divide(
        _multiply(
            new_leaf_carbon,
            coarse_root_ratio,
            stem_ratio,
            dead_wood_fraction,
        ),
        dead_wood_cn,
    )
    grain_base = fortran_divide(
        _multiply(new_leaf_carbon, grain_ratio), grain_cn
    )
    demands = _OrganFluxes(
        leaf_current=fortran_multiply(leaf_base, current_fraction),
        leaf_storage=fortran_multiply(leaf_base, storage_fraction),
        fine_root_current=fortran_multiply(
            fine_root_base, current_fraction
        ),
        fine_root_storage=fortran_multiply(
            fine_root_base, storage_fraction
        ),
        live_stem_current=fortran_multiply(
            live_stem_base, current_fraction
        ),
        live_stem_storage=fortran_multiply(
            live_stem_base, storage_fraction
        ),
        dead_stem_current=fortran_multiply(
            dead_stem_base, current_fraction
        ),
        dead_stem_storage=fortran_multiply(
            dead_stem_base, storage_fraction
        ),
        live_coarse_root_current=fortran_multiply(
            live_coarse_root_base, current_fraction
        ),
        live_coarse_root_storage=fortran_multiply(
            live_coarse_root_base, storage_fraction
        ),
        dead_coarse_root_current=fortran_multiply(
            dead_coarse_root_base, current_fraction
        ),
        dead_coarse_root_storage=fortran_multiply(
            dead_coarse_root_base, storage_fraction
        ),
        grain_current=fortran_multiply(grain_base, current_fraction),
        grain_storage=fortran_multiply(grain_base, storage_fraction),
    )
    natural_total = _sum_in_source_order(
        demands.leaf_current,
        demands.leaf_storage,
        demands.fine_root_current,
        demands.fine_root_storage,
    )
    woody_total = _sum_in_source_order(
        demands.leaf_current,
        demands.leaf_storage,
        demands.fine_root_current,
        demands.fine_root_storage,
        demands.live_stem_current,
        demands.live_stem_storage,
        demands.dead_stem_current,
        demands.dead_stem_storage,
        demands.live_coarse_root_current,
        demands.live_coarse_root_storage,
        demands.dead_coarse_root_current,
        demands.dead_coarse_root_storage,
    )
    crop_total = _sum_in_source_order(
        demands.leaf_current,
        demands.leaf_storage,
        demands.fine_root_current,
        demands.fine_root_storage,
        demands.live_stem_current,
        demands.live_stem_storage,
        demands.dead_stem_current,
        demands.dead_stem_storage,
        demands.live_coarse_root_current,
        demands.live_coarse_root_storage,
        demands.dead_coarse_root_current,
        demands.dead_coarse_root_storage,
        demands.grain_current,
        demands.grain_storage,
    )
    total = jnp.where(crop, crop_total, jnp.where(woody, woody_total, natural_total))
    zero_total = total == jnp.float64(0.0)
    safe_total = jnp.where(zero_total, jnp.float64(1.0), total)

    def allocate(demand: jnp.ndarray) -> jnp.ndarray:
        fraction = fortran_divide(demand, safe_total)
        fraction = jnp.where(zero_total, jnp.float64(0.0), fraction)
        return fortran_divide(
            fortran_multiply(fraction, plant_nitrogen_pool_patch),
            time_step,
        )

    return _OrganFluxes(*(allocate(value) for value in demands))


def plant_nutrient_competition(
    inputs: PlantNutrientCompetitionInputs,
    state: PlantNutrientCompetitionState,
    parameters: PlantNutrientCompetitionParameters = (
        DEFAULT_PLANT_NUTRIENT_COMPETITION_PARAMETERS
    ),
) -> PlantNutrientCompetitionResult:

    _validate_active_configuration(parameters)
    _validate_shapes(inputs, state, parameters)
    soil = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    pft_count = parameters.woody_pft.shape[0]
    pft = jnp.clip(
        jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32),
        0,
        pft_count - 1,
    )
    crop = pft >= jnp.asarray(parameters.first_crop_pft, dtype=jnp.int32)
    live_crop = crop & jnp.asarray(inputs.crop_live_patch, dtype=jnp.bool_)
    dormant_crop = crop & ~live_crop
    woody = _selected(parameters.woody_pft, pft) == jnp.float64(1.0)

    fine_root_ratio = _selected(parameters.fine_root_to_leaf_carbon_pft, pft)
    coarse_root_ratio = _selected(parameters.coarse_root_to_stem_carbon_pft, pft)
    stem_parameter = _selected(parameters.stem_to_leaf_carbon_pft, pft)
    default_minus_point_zero_zero_four = jnp.float64(np.float32(-0.004))
    default_two_point_seven = jnp.float64(np.float32(2.7))
    default_one = jnp.float64(np.float32(1.0))
    default_three_hundred = jnp.float64(np.float32(300.0))
    default_point_four = jnp.float64(np.float32(0.4))
    dynamic_argument = fortran_multiply(
        default_minus_point_zero_zero_four,
        fortran_add(
            jnp.where(
                soil,
                jnp.asarray(
                    inputs.annual_net_primary_productivity_patch,
                    dtype=jnp.float64,
                ),
                jnp.float64(0.0),
            ),
            -default_three_hundred,
        ),
    )
    dynamic_stem = fortran_add(
        fortran_divide(
            default_two_point_seven,
            fortran_add(default_one, strict_exp(dynamic_argument)),
        ),
        -default_point_four,
    )
    stem_ratio = jnp.where(
        stem_parameter == jnp.float64(-1.0), dynamic_stem, stem_parameter
    )
    live_wood_fraction = _selected(parameters.live_wood_fraction_pft, pft)
    growth_respiration_fraction = _selected(
        parameters.growth_respiration_fraction_pft, pft
    )
    current_growth_respiration_fraction = _selected(
        parameters.current_growth_respiration_fraction_pft, pft
    )
    current_fraction = _selected(parameters.current_growth_fraction_pft, pft)
    evergreen = _selected(parameters.evergreen_pft, pft) == jnp.float64(1.0)
    current_fraction = jnp.where(evergreen, jnp.float64(0.0), current_fraction)

    safe_leaf_allocation = jnp.where(
        live_crop,
        jnp.asarray(inputs.leaf_allocation_patch, dtype=jnp.float64),
        jnp.float64(1.0),
    )
    crop_fine_root_ratio = fortran_divide(
        inputs.root_allocation_patch, safe_leaf_allocation
    )
    crop_stem_ratio = fortran_divide(
        inputs.stem_allocation_patch, safe_leaf_allocation
    )
    crop_grain_ratio = fortran_divide(
        inputs.reproductive_allocation_patch, safe_leaf_allocation
    )
    fine_root_ratio = jnp.where(
        live_crop,
        crop_fine_root_ratio,
        jnp.where(dormant_crop, jnp.float64(0.0), fine_root_ratio),
    )
    stem_ratio = jnp.where(
        live_crop,
        crop_stem_ratio,
        jnp.where(dormant_crop, jnp.float64(0.0), stem_ratio),
    )
    grain_ratio = jnp.where(live_crop, crop_grain_ratio, jnp.float64(0.0))
    growth_respiration_fraction = jnp.where(
        crop, jnp.float64(0.25), growth_respiration_fraction
    )
    storage_fraction = fortran_add(jnp.float64(1.0), -current_fraction)

    retranslocation = get_cn_flux_field(
        state.cn_fluxes, "vegetation_nitrogen.retransn_to_npool_patch"
    )
    fun_growth = get_cn_flux_field(
        state.cn_fluxes, "vegetation_carbon.npp_growth_patch"
    )
    soil_nitrogen = jnp.asarray(
        inputs.fun_soil_nitrogen_uptake_patch, dtype=jnp.float64
    )
    plant_nitrogen_allocation = fortran_add(soil_nitrogen, retranslocation)
    safe_allometry = jnp.where(
        soil,
        jnp.asarray(inputs.carbon_allometry_patch, dtype=jnp.float64),
        jnp.float64(1.0),
    )
    new_leaf_carbon = fortran_divide(fun_growth, safe_allometry)
    carbon = _carbon_fluxes(
        new_leaf_carbon,
        current_fraction,
        storage_fraction,
        fine_root_ratio,
        coarse_root_ratio,
        stem_ratio,
        live_wood_fraction,
        grain_ratio,
    )
    nitrogen = _proportional_nitrogen_fluxes(
        jnp.asarray(inputs.plant_nitrogen_pool_patch, dtype=jnp.float64),
        new_leaf_carbon,
        current_fraction,
        storage_fraction,
        woody,
        crop,
        fine_root_ratio,
        coarse_root_ratio,
        stem_ratio,
        live_wood_fraction,
        grain_ratio,
        _selected(parameters.leaf_carbon_nitrogen_ratio_pft, pft),
        _selected(parameters.fine_root_carbon_nitrogen_ratio_pft, pft),
        _selected(parameters.live_wood_carbon_nitrogen_ratio_pft, pft),
        _selected(parameters.dead_wood_carbon_nitrogen_ratio_pft, pft),
        _selected(parameters.grain_carbon_nitrogen_ratio_pft, pft),
        jnp.asarray(parameters.time_step_seconds, dtype=jnp.float64),
    )

    updated = state.cn_fluxes

    def write_fields(mapping, fluxes, mask):
        result = updated
        for field, attribute in mapping.items():
            old = get_cn_flux_field(result, field)
            result = set_cn_flux_field(
                result,
                field,
                jnp.where(mask, getattr(fluxes, attribute), old),
            )
        return result

    updated = write_fields(_CARBON_GENERAL_FIELDS, carbon, soil)
    for mapping, fluxes, mask in (
        (_CARBON_WOOD_FIELDS, carbon, soil & (woody | crop)),
        (_CARBON_CROP_FIELDS, carbon, soil & crop),
        (_NITROGEN_GENERAL_FIELDS, nitrogen, soil),
        (_NITROGEN_WOOD_FIELDS, nitrogen, soil & (woody | crop)),
        (_NITROGEN_CROP_FIELDS, nitrogen, soil & crop),
    ):
        for field, attribute in mapping.items():
            old = get_cn_flux_field(updated, field)
            updated = set_cn_flux_field(
                updated,
                field,
                jnp.where(mask, getattr(fluxes, attribute), old),
            )

    growth_respiration_storage = fortran_add(
        carbon.leaf_storage, carbon.fine_root_storage
    )
    for value in (
        carbon.live_stem_storage,
        carbon.dead_stem_storage,
        carbon.live_coarse_root_storage,
        carbon.dead_coarse_root_storage,
    ):
        growth_respiration_storage = jnp.where(
            woody,
            fortran_add(growth_respiration_storage, value),
            growth_respiration_storage,
        )
    for value in (carbon.live_stem_storage, carbon.grain_storage):
        growth_respiration_storage = jnp.where(
            crop,
            fortran_add(growth_respiration_storage, value),
            growth_respiration_storage,
        )
    growth_respiration_storage = _multiply(
        growth_respiration_storage,
        growth_respiration_fraction,
        fortran_add(
            jnp.float64(1.0), -current_growth_respiration_fraction
        ),
    )
    for field, values in (
        (
            "vegetation_carbon.cpool_to_gresp_storage_patch",
            growth_respiration_storage,
        ),
        ("vegetation_nitrogen.sminn_to_npool_patch", soil_nitrogen),
    ):
        old = get_cn_flux_field(updated, field)
        updated = set_cn_flux_field(
            updated, field, jnp.where(soil, values, old)
        )

    for field in _CARBON_RESPIRATION_FIELDS:
        old = get_cn_flux_field(updated, field)
        updated = set_cn_flux_field(
            updated,
            field,
            jnp.where(soil, jnp.float64(0.0), old),
        )

    leaf_area_index = fortran_add(
        jnp.asarray(inputs.sunlit_leaf_area_index_patch, dtype=jnp.float64),
        jnp.asarray(inputs.shaded_leaf_area_index_patch, dtype=jnp.float64),
    )
    leaf_nitrogen_storage = jnp.asarray(
        inputs.leaf_nitrogen_storage_patch, dtype=jnp.float64
    )
    safe_leaf_nitrogen_storage = jnp.where(
        leaf_nitrogen_storage == jnp.float64(0.0),
        jnp.float64(1.0),
        leaf_nitrogen_storage,
    )
    storage_leaf_cn = fortran_divide(
        jnp.asarray(inputs.leaf_carbon_storage_patch, dtype=jnp.float64),
        safe_leaf_nitrogen_storage,
    )
    storage_leaf_cn = jnp.where(
        leaf_nitrogen_storage == jnp.float64(0.0),
        jnp.asarray(parameters.missing_value, dtype=jnp.float64),
        storage_leaf_cn,
    )
    update_storage_leaf_cn = soil & (leaf_area_index > jnp.float64(0.0))
    actual_storage_leaf_cn = jnp.where(
        update_storage_leaf_cn,
        storage_leaf_cn,
        state.actual_storage_leaf_carbon_nitrogen_ratio_patch,
    )

    return PlantNutrientCompetitionResult(
        state=PlantNutrientCompetitionState(
            cn_fluxes=updated,
            plant_carbon_allocation_patch=jnp.where(
                soil, fun_growth, state.plant_carbon_allocation_patch
            ),
            plant_nitrogen_allocation_patch=jnp.where(
                soil,
                plant_nitrogen_allocation,
                state.plant_nitrogen_allocation_patch,
            ),
            actual_storage_leaf_carbon_nitrogen_ratio_patch=(
                actual_storage_leaf_cn
            ),
        ),
        diagnostics=PlantNutrientCompetitionDiagnostics(
            updated_patch_count=jnp.sum(soil, dtype=jnp.int32),
            woody_patch_count=jnp.sum(soil & woody, dtype=jnp.int32),
            live_crop_patch_count=jnp.sum(soil & live_crop, dtype=jnp.int32),
            dormant_crop_patch_count=jnp.sum(
                soil & dormant_crop, dtype=jnp.int32
            ),
        ),
    )


__all__ = [
    "DEFAULT_PLANT_NUTRIENT_COMPETITION_PARAMETERS",
    "PlantNutrientCompetitionDiagnostics",
    "PlantNutrientCompetitionInputs",
    "PlantNutrientCompetitionParameters",
    "PlantNutrientCompetitionResult",
    "PlantNutrientCompetitionState",
    "plant_nutrient_competition",
]
