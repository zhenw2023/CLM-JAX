
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from clm_jax.constants import SECONDS_PER_DAY
from clm_jax.physics.cn_precision_control import (
    CARBON_ONLY_FIELDS,
    NITROGEN_ONLY_FIELDS,
    PAIRED_CARBON_FIELDS,
    PAIRED_NITROGEN_FIELDS,
    CnPrecisionControlState,
    get_cn_precision_control_state_field,
)
from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    get_cn_flux_field,
    set_cn_flux_field,
)
from clm_jax.physics.cn_zero_manifest import (
    NUMBER_OF_DECOMPOSITION_LEVELS,
    NUMBER_OF_DECOMPOSITION_POOLS,
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


class CnFireFluxInputs(NamedTuple):

    soil_patch: jnp.ndarray
    soil_column: jnp.ndarray
    column_patch_indices: jnp.ndarray
    column_gridcell_index: jnp.ndarray
    patch_column_index: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    patch_weight_on_column: jnp.ndarray
    patch_active: jnp.ndarray
    gridcell_latitude_degrees: jnp.ndarray
    vegetation_state: CnPrecisionControlState
    crop_fraction_column: jnp.ndarray
    total_burned_area_fraction_column: jnp.ndarray
    conversion_burned_area_fraction_column: jnp.ndarray
    conversion_fire_burned_area_fraction_column: jnp.ndarray
    crop_burned_area_fraction_column: jnp.ndarray
    peat_burned_area_fraction_column: jnp.ndarray
    tropical_evergreen_tree_fraction_column: jnp.ndarray
    tropical_deciduous_tree_fraction_column: jnp.ndarray
    decreased_tropical_tree_fraction_column: jnp.ndarray
    leaf_profile_patch_level: jnp.ndarray
    fine_root_profile_patch_level: jnp.ndarray
    coarse_root_profile_patch_level: jnp.ndarray
    stem_profile_patch_level: jnp.ndarray
    total_soil_organic_carbon_column: jnp.ndarray
    decomposition_carbon_pool_column_level_pool: jnp.ndarray
    decomposition_nitrogen_pool_column_level_pool: jnp.ndarray


class CnFireFluxState(NamedTuple):

    unburned_conversion_fraction_column: jnp.ndarray
    burned_conversion_fraction_column: jnp.ndarray
    individual_density_patch: jnp.ndarray
    maximum_leaf_carbon_patch: jnp.ndarray


class CnFireFluxParameters(NamedTuple):

    woody_pft: jnp.ndarray
    leaf_combustion_fraction_pft: jnp.ndarray
    live_stem_combustion_fraction_pft: jnp.ndarray
    dead_stem_combustion_fraction_pft: jnp.ndarray
    other_combustion_fraction_pft: jnp.ndarray
    leaf_mortality_fraction_pft: jnp.ndarray
    live_stem_mortality_fraction_pft: jnp.ndarray
    other_mortality_fraction_pft: jnp.ndarray
    fine_root_mortality_fraction_pft: jnp.ndarray
    live_root_mortality_fraction_pft: jnp.ndarray
    dead_root_mortality_fraction_pft: jnp.ndarray
    leaf_labile_fraction_pft: jnp.ndarray
    leaf_cellulose_fraction_pft: jnp.ndarray
    leaf_lignin_fraction_pft: jnp.ndarray
    fine_root_labile_fraction_pft: jnp.ndarray
    fine_root_cellulose_fraction_pft: jnp.ndarray
    fine_root_lignin_fraction_pft: jnp.ndarray
    decomposition_pool_is_litter: jnp.ndarray
    decomposition_pool_is_cwd: jnp.ndarray
    litter_combustion_completion_fraction: float | jnp.ndarray = 0.5
    cwd_combustion_completion_fraction: float | jnp.ndarray = 0.28
    boreal_latitude_degrees: float | jnp.ndarray = 40.0
    time_step: float | jnp.ndarray = 1800.0
    days_per_year: float | jnp.ndarray = 365.0
    decomposition_level_count: int | jnp.ndarray = 20
    spinup_state: int | jnp.ndarray = 0
    first_generic_crop_pft: int | jnp.ndarray = 15
    bare_ground_pft: int | jnp.ndarray = 0
    use_cndv: bool | jnp.ndarray = False
    transient_landcover: bool | jnp.ndarray = False
    month: int | jnp.ndarray = 1
    day_of_month: int | jnp.ndarray = 1
    seconds_of_day: int | jnp.ndarray = 1800


class CnFireFluxDiagnostics(NamedTuple):
    fire_rate_patch: jnp.ndarray
    updated_patch_count: jnp.ndarray
    updated_column_count: jnp.ndarray
    active_patch_visit_count: jnp.ndarray
    accelerated_deadwood_patch_count: jnp.ndarray
    dynamic_vegetation_patch_count: jnp.ndarray
    conversion_fire_column_count: jnp.ndarray


class CnFireFluxResult(NamedTuple):
    state: CnFireFluxState
    cn_fluxes: CnTransientFluxState
    diagnostics: CnFireFluxDiagnostics


def _mul(*values: jnp.ndarray) -> jnp.ndarray:
    result = jnp.asarray(values[0], dtype=jnp.float64)
    for value in values[1:]:
        result = fortran_multiply(result, value)
    return result


def _add(*values: jnp.ndarray) -> jnp.ndarray:
    result = jnp.asarray(values[0], dtype=jnp.float64)
    for value in values[1:]:
        result = fortran_add(result, value)
    return result


def _subtract(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return fortran_add(left, -jnp.asarray(right, dtype=jnp.float64))


_CARBON_COMBUSTION_ASSIGNMENTS = (
    ("vegetation_carbon.leafc_patch", "vegetation_carbon.m_leafc_to_fire_patch", "leaf", False),
    (
        "vegetation_carbon.leafc_storage_patch",
        "vegetation_carbon.m_leafc_storage_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_carbon.leafc_xfer_patch",
        "vegetation_carbon.m_leafc_xfer_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_carbon.livestemc_patch",
        "vegetation_carbon.m_livestemc_to_fire_patch",
        "live_stem",
        False,
    ),
    (
        "vegetation_carbon.livestemc_storage_patch",
        "vegetation_carbon.m_livestemc_storage_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_carbon.livestemc_xfer_patch",
        "vegetation_carbon.m_livestemc_xfer_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_carbon.deadstemc_patch",
        "vegetation_carbon.m_deadstemc_to_fire_patch",
        "dead_stem",
        True,
    ),
    (
        "vegetation_carbon.deadstemc_storage_patch",
        "vegetation_carbon.m_deadstemc_storage_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_carbon.deadstemc_xfer_patch",
        "vegetation_carbon.m_deadstemc_xfer_to_fire_patch",
        "other",
        False,
    ),
    ("vegetation_carbon.frootc_patch", "vegetation_carbon.m_frootc_to_fire_patch", "zero", False),
    (
        "vegetation_carbon.frootc_storage_patch",
        "vegetation_carbon.m_frootc_storage_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_carbon.frootc_xfer_patch",
        "vegetation_carbon.m_frootc_xfer_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_carbon.livecrootc_patch",
        "vegetation_carbon.m_livecrootc_to_fire_patch",
        "zero",
        False,
    ),
    (
        "vegetation_carbon.livecrootc_storage_patch",
        "vegetation_carbon.m_livecrootc_storage_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_carbon.livecrootc_xfer_patch",
        "vegetation_carbon.m_livecrootc_xfer_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_carbon.deadcrootc_patch",
        "vegetation_carbon.m_deadcrootc_to_fire_patch",
        "zero",
        False,
    ),
    (
        "vegetation_carbon.deadcrootc_storage_patch",
        "vegetation_carbon.m_deadcrootc_storage_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_carbon.deadcrootc_xfer_patch",
        "vegetation_carbon.m_deadcrootc_xfer_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_carbon.gresp_storage_patch",
        "vegetation_carbon.m_gresp_storage_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_carbon.gresp_xfer_patch",
        "vegetation_carbon.m_gresp_xfer_to_fire_patch",
        "other",
        False,
    ),
)

_NITROGEN_COMBUSTION_ASSIGNMENTS = (
    ("vegetation_nitrogen.leafn_patch", "vegetation_nitrogen.m_leafn_to_fire_patch", "leaf", False),
    (
        "vegetation_nitrogen.leafn_storage_patch",
        "vegetation_nitrogen.m_leafn_storage_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.leafn_xfer_patch",
        "vegetation_nitrogen.m_leafn_xfer_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.livestemn_patch",
        "vegetation_nitrogen.m_livestemn_to_fire_patch",
        "live_stem",
        False,
    ),
    (
        "vegetation_nitrogen.livestemn_storage_patch",
        "vegetation_nitrogen.m_livestemn_storage_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.livestemn_xfer_patch",
        "vegetation_nitrogen.m_livestemn_xfer_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.deadstemn_patch",
        "vegetation_nitrogen.m_deadstemn_to_fire_patch",
        "dead_stem",
        True,
    ),
    (
        "vegetation_nitrogen.deadstemn_storage_patch",
        "vegetation_nitrogen.m_deadstemn_storage_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.deadstemn_xfer_patch",
        "vegetation_nitrogen.m_deadstemn_xfer_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.frootn_patch",
        "vegetation_nitrogen.m_frootn_to_fire_patch",
        "zero",
        False,
    ),
    (
        "vegetation_nitrogen.frootn_storage_patch",
        "vegetation_nitrogen.m_frootn_storage_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.frootn_xfer_patch",
        "vegetation_nitrogen.m_frootn_xfer_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.livecrootn_patch",
        "vegetation_nitrogen.m_livecrootn_to_fire_patch",
        "zero",
        False,
    ),
    (
        "vegetation_nitrogen.livecrootn_storage_patch",
        "vegetation_nitrogen.m_livecrootn_storage_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.livecrootn_xfer_patch",
        "vegetation_nitrogen.m_livecrootn_xfer_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.deadcrootn_patch",
        "vegetation_nitrogen.m_deadcrootn_to_fire_patch",
        "zero",
        False,
    ),
    (
        "vegetation_nitrogen.deadcrootn_xfer_patch",
        "vegetation_nitrogen.m_deadcrootn_xfer_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.deadcrootn_storage_patch",
        "vegetation_nitrogen.m_deadcrootn_storage_to_fire_patch",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.retransn_patch",
        "vegetation_nitrogen.m_retransn_to_fire_patch",
        "other",
        False,
    ),
)


_CARBON_MORTALITY_ASSIGNMENTS = (
    (
        "vegetation_carbon.leafc_patch",
        "vegetation_carbon.m_leafc_to_litter_fire_patch",
        "leaf",
        "leaf",
        False,
    ),
    (
        "vegetation_carbon.leafc_storage_patch",
        "vegetation_carbon.m_leafc_storage_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_carbon.leafc_xfer_patch",
        "vegetation_carbon.m_leafc_xfer_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_carbon.livestemc_patch",
        "vegetation_carbon.m_livestemc_to_litter_fire_patch",
        "live_stem",
        "dead_root",
        False,
    ),
    (
        "vegetation_carbon.livestemc_storage_patch",
        "vegetation_carbon.m_livestemc_storage_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_carbon.livestemc_xfer_patch",
        "vegetation_carbon.m_livestemc_xfer_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_carbon.deadstemc_patch",
        "vegetation_carbon.m_deadstemc_to_litter_fire_patch",
        "dead_stem",
        "dead_root",
        True,
    ),
    (
        "vegetation_carbon.deadstemc_storage_patch",
        "vegetation_carbon.m_deadstemc_storage_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_carbon.deadstemc_xfer_patch",
        "vegetation_carbon.m_deadstemc_xfer_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_carbon.frootc_patch",
        "vegetation_carbon.m_frootc_to_litter_fire_patch",
        None,
        "fine_root",
        False,
    ),
    (
        "vegetation_carbon.frootc_storage_patch",
        "vegetation_carbon.m_frootc_storage_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_carbon.frootc_xfer_patch",
        "vegetation_carbon.m_frootc_xfer_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_carbon.livecrootc_patch",
        "vegetation_carbon.m_livecrootc_to_litter_fire_patch",
        None,
        "dead_root",
        False,
    ),
    (
        "vegetation_carbon.livecrootc_storage_patch",
        "vegetation_carbon.m_livecrootc_storage_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_carbon.livecrootc_xfer_patch",
        "vegetation_carbon.m_livecrootc_xfer_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_carbon.deadcrootc_patch",
        "vegetation_carbon.m_deadcrootc_to_litter_fire_patch",
        None,
        "dead_root",
        True,
    ),
    (
        "vegetation_carbon.deadcrootc_storage_patch",
        "vegetation_carbon.m_deadcrootc_storage_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_carbon.deadcrootc_xfer_patch",
        "vegetation_carbon.m_deadcrootc_xfer_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_carbon.gresp_storage_patch",
        "vegetation_carbon.m_gresp_storage_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_carbon.gresp_xfer_patch",
        "vegetation_carbon.m_gresp_xfer_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
)

_CARBON_TRANSFER_ASSIGNMENTS = (
    (
        "vegetation_carbon.livestemc_patch",
        "vegetation_carbon.m_livestemc_to_deadstemc_fire_patch",
        "live_stem",
        "live_stem_minus_dead_root",
    ),
    (
        "vegetation_carbon.livecrootc_patch",
        "vegetation_carbon.m_livecrootc_to_deadcrootc_fire_patch",
        None,
        "live_root_minus_dead_root",
    ),
)

_NITROGEN_MORTALITY_ASSIGNMENTS = (
    (
        "vegetation_nitrogen.leafn_patch",
        "vegetation_nitrogen.m_leafn_to_litter_fire_patch",
        "leaf",
        "leaf",
        False,
    ),
    (
        "vegetation_nitrogen.leafn_storage_patch",
        "vegetation_nitrogen.m_leafn_storage_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.leafn_xfer_patch",
        "vegetation_nitrogen.m_leafn_xfer_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.livestemn_patch",
        "vegetation_nitrogen.m_livestemn_to_litter_fire_patch",
        "live_stem",
        "dead_root",
        False,
    ),
    (
        "vegetation_nitrogen.livestemn_storage_patch",
        "vegetation_nitrogen.m_livestemn_storage_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.livestemn_xfer_patch",
        "vegetation_nitrogen.m_livestemn_xfer_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.deadstemn_patch",
        "vegetation_nitrogen.m_deadstemn_to_litter_fire_patch",
        "dead_stem",
        "dead_root",
        True,
    ),
    (
        "vegetation_nitrogen.deadstemn_storage_patch",
        "vegetation_nitrogen.m_deadstemn_storage_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.deadstemn_xfer_patch",
        "vegetation_nitrogen.m_deadstemn_xfer_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.frootn_patch",
        "vegetation_nitrogen.m_frootn_to_litter_fire_patch",
        None,
        "fine_root",
        False,
    ),
    (
        "vegetation_nitrogen.frootn_storage_patch",
        "vegetation_nitrogen.m_frootn_storage_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.frootn_xfer_patch",
        "vegetation_nitrogen.m_frootn_xfer_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.livecrootn_patch",
        "vegetation_nitrogen.m_livecrootn_to_litter_fire_patch",
        None,
        "dead_root",
        False,
    ),
    (
        "vegetation_nitrogen.livecrootn_storage_patch",
        "vegetation_nitrogen.m_livecrootn_storage_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.livecrootn_xfer_patch",
        "vegetation_nitrogen.m_livecrootn_xfer_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.deadcrootn_patch",
        "vegetation_nitrogen.m_deadcrootn_to_litter_fire_patch",
        None,
        "dead_root",
        True,
    ),
    (
        "vegetation_nitrogen.deadcrootn_storage_patch",
        "vegetation_nitrogen.m_deadcrootn_storage_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.deadcrootn_xfer_patch",
        "vegetation_nitrogen.m_deadcrootn_xfer_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
    (
        "vegetation_nitrogen.retransn_patch",
        "vegetation_nitrogen.m_retransn_to_litter_fire_patch",
        "other",
        "other",
        False,
    ),
)

_NITROGEN_TRANSFER_ASSIGNMENTS = (
    (
        "vegetation_nitrogen.livestemn_patch",
        "vegetation_nitrogen.m_livestemn_to_deadstemn_fire_patch",
        "live_stem",
        "live_stem_minus_dead_root",
    ),
    (
        "vegetation_nitrogen.livecrootn_patch",
        "vegetation_nitrogen.m_livecrootn_to_deadcrootn_fire_patch",
        None,
        "live_root_minus_dead_root",
    ),
)


_COLUMN_OUTPUT_FIELDS = (
    "vegetation_carbon.fire_mortality_c_to_cwdc_col",
    "vegetation_nitrogen.fire_mortality_n_to_cwdn_col",
    "vegetation_carbon.m_c_to_litr_met_fire_col",
    "vegetation_carbon.m_c_to_litr_cel_fire_col",
    "vegetation_carbon.m_c_to_litr_lig_fire_col",
    "vegetation_nitrogen.m_n_to_litr_met_fire_col",
    "vegetation_nitrogen.m_n_to_litr_cel_fire_col",
    "vegetation_nitrogen.m_n_to_litr_lig_fire_col",
)


def _validate_shapes(
    inputs: CnFireFluxInputs,
    state: CnFireFluxState,
    cn_fluxes: CnTransientFluxState,
    parameters: CnFireFluxParameters,
) -> tuple[int, int, int]:
    if jnp.ndim(inputs.soil_patch) != 1 or jnp.ndim(inputs.soil_column) != 1:
        raise ValueError("soil_patch and soil_column must be vectors")
    patch_count = inputs.soil_patch.shape[0]
    column_count = inputs.soil_column.shape[0]
    gridcell_count = inputs.gridcell_latitude_degrees.shape[0]
    if (
        jnp.shape(inputs.column_patch_indices)[0:1] != (column_count,)
        or jnp.ndim(inputs.column_patch_indices) != 2
    ):
        raise ValueError("column_patch_indices must have shape (column, patch slot)")
    if jnp.shape(inputs.column_gridcell_index) != (column_count,):
        raise ValueError("column_gridcell_index must contain every column")
    for name in (
        "patch_column_index",
        "vegetation_type_patch",
        "patch_weight_on_column",
        "patch_active",
    ):
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    if jnp.ndim(inputs.gridcell_latitude_degrees) != 1:
        raise ValueError("gridcell_latitude_degrees must be a vector")
    if column_count and gridcell_count == 0:
        raise ValueError("columns require at least one gridcell")

    for name in (
        "crop_fraction_column",
        "total_burned_area_fraction_column",
        "conversion_burned_area_fraction_column",
        "conversion_fire_burned_area_fraction_column",
        "crop_burned_area_fraction_column",
        "peat_burned_area_fraction_column",
        "tropical_evergreen_tree_fraction_column",
        "tropical_deciduous_tree_fraction_column",
        "decreased_tropical_tree_fraction_column",
        "total_soil_organic_carbon_column",
    ):
        if jnp.shape(getattr(inputs, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    for name in (
        "leaf_profile_patch_level",
        "fine_root_profile_patch_level",
        "coarse_root_profile_patch_level",
        "stem_profile_patch_level",
    ):
        if jnp.shape(getattr(inputs, name)) != (
            patch_count,
            NUMBER_OF_DECOMPOSITION_LEVELS,
        ):
            raise ValueError(f"{name} must have shape (patch, 25)")
    expected_decomposition_shape = (
        column_count,
        NUMBER_OF_DECOMPOSITION_LEVELS,
        NUMBER_OF_DECOMPOSITION_POOLS,
    )
    if jnp.shape(inputs.decomposition_carbon_pool_column_level_pool) != (
        expected_decomposition_shape
    ):
        raise ValueError("decomposition carbon pools have incompatible dimensions")
    if jnp.shape(inputs.decomposition_nitrogen_pool_column_level_pool) != (
        expected_decomposition_shape
    ):
        raise ValueError("decomposition nitrogen pools have incompatible dimensions")

    expected_vegetation_shapes = {
        "paired_carbon_patch": (patch_count, len(PAIRED_CARBON_FIELDS)),
        "paired_nitrogen_patch": (patch_count, len(PAIRED_NITROGEN_FIELDS)),
        "carbon_only_patch": (patch_count, len(CARBON_ONLY_FIELDS)),
        "nitrogen_only_patch": (patch_count, len(NITROGEN_ONLY_FIELDS)),
        "carbon_truncation_patch": (patch_count,),
        "nitrogen_truncation_patch": (patch_count,),
    }
    for name, expected in expected_vegetation_shapes.items():
        if jnp.shape(getattr(inputs.vegetation_state, name)) != expected:
            raise ValueError(f"vegetation state {name} has an incompatible shape")
    if jnp.shape(state.unburned_conversion_fraction_column) != (column_count,):
        raise ValueError("unburned conversion fraction must contain every column")
    if jnp.shape(state.burned_conversion_fraction_column) != (column_count,):
        raise ValueError("burned conversion fraction must contain every column")
    if jnp.shape(state.individual_density_patch) != (patch_count,):
        raise ValueError("individual density must contain every patch")
    if jnp.shape(state.maximum_leaf_carbon_patch) != (patch_count,):
        raise ValueError("maximum leaf carbon must contain every patch")
    if cn_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("CN patch fluxes must match patch count")
    if cn_fluxes.column_scalar.shape[0] != column_count:
        raise ValueError("CN column fluxes must match column count")
    if cn_fluxes.column_level_pool.shape[:3] != expected_decomposition_shape:
        raise ValueError("CN decomposition fluxes have incompatible dimensions")

    pft_count = None
    for name in parameters._fields[:17]:
        values = getattr(parameters, name)
        if jnp.ndim(values) != 1 or values.shape[0] < 1:
            raise ValueError(f"{name} must be a nonempty PFT vector")
        if pft_count is None:
            pft_count = values.shape[0]
        elif values.shape[0] != pft_count:
            raise ValueError("all PFT parameter vectors must have equal length")
    for name in ("decomposition_pool_is_litter", "decomposition_pool_is_cwd"):
        if jnp.shape(getattr(parameters, name)) != (NUMBER_OF_DECOMPOSITION_POOLS,):
            raise ValueError(f"{name} must contain every decomposition pool")
    for name in parameters._fields[19:]:
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    return patch_count, column_count, gridcell_count


def _fire_rate_patch(
    inputs: CnFireFluxInputs,
    parameters: CnFireFluxParameters,
) -> jnp.ndarray:
    patch_count = inputs.soil_patch.shape[0]
    if patch_count == 0:
        return jnp.zeros((0,), dtype=jnp.float64)
    patch_column = jnp.asarray(inputs.patch_column_index, dtype=jnp.int32)

    vegetation_type = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    crop_fraction = jnp.asarray(inputs.crop_fraction_column, dtype=jnp.float64)[patch_column]
    crop_burned = jnp.asarray(
        inputs.crop_burned_area_fraction_column,
        dtype=jnp.float64,
    )[patch_column]
    total_burned = jnp.asarray(
        inputs.total_burned_area_fraction_column,
        dtype=jnp.float64,
    )[patch_column]
    conversion_burned = jnp.asarray(
        inputs.conversion_burned_area_fraction_column,
        dtype=jnp.float64,
    )[patch_column]
    source_burned = jnp.where(
        jnp.asarray(parameters.transient_landcover, dtype=jnp.bool_),
        conversion_burned,
        total_burned,
    )
    noncrop_numerator = _subtract(source_burned, crop_burned)
    noncrop_denominator = _subtract(jnp.float64(1.0), crop_fraction)
    safe_noncrop_denominator = jnp.where(
        noncrop_denominator != jnp.float64(0.0),
        noncrop_denominator,
        jnp.float64(1.0),
    )
    noncrop_rate = fortran_divide(noncrop_numerator, safe_noncrop_denominator)
    safe_crop_fraction = jnp.where(
        crop_fraction > jnp.float64(0.0),
        crop_fraction,
        jnp.float64(1.0),
    )
    crop_rate = jnp.where(
        crop_fraction > jnp.float64(0.0),
        fortran_divide(crop_burned, safe_crop_fraction),
        jnp.float64(0.0),
    )
    noncrop_patch = (
        vegetation_type < jnp.asarray(parameters.first_generic_crop_pft, dtype=jnp.int32)
    ) & (crop_fraction < jnp.float64(1.0))
    return jnp.where(noncrop_patch, noncrop_rate, crop_rate)


def _selected_factors(
    parameters: CnFireFluxParameters,
    vegetation_type: jnp.ndarray,
) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    combustion = {
        "leaf": jnp.asarray(parameters.leaf_combustion_fraction_pft, dtype=jnp.float64)[
            vegetation_type
        ],
        "live_stem": jnp.asarray(
            parameters.live_stem_combustion_fraction_pft,
            dtype=jnp.float64,
        )[vegetation_type],
        "dead_stem": jnp.asarray(
            parameters.dead_stem_combustion_fraction_pft,
            dtype=jnp.float64,
        )[vegetation_type],
        "other": jnp.asarray(
            parameters.other_combustion_fraction_pft,
            dtype=jnp.float64,
        )[vegetation_type],
        "zero": jnp.zeros(vegetation_type.shape, dtype=jnp.float64),
    }
    dead_root = jnp.asarray(
        parameters.dead_root_mortality_fraction_pft,
        dtype=jnp.float64,
    )[vegetation_type]
    mortality = {
        "leaf": jnp.asarray(parameters.leaf_mortality_fraction_pft, dtype=jnp.float64)[
            vegetation_type
        ],
        "live_stem": jnp.asarray(
            parameters.live_stem_mortality_fraction_pft,
            dtype=jnp.float64,
        )[vegetation_type],
        "other": jnp.asarray(parameters.other_mortality_fraction_pft, dtype=jnp.float64)[
            vegetation_type
        ],
        "fine_root": jnp.asarray(
            parameters.fine_root_mortality_fraction_pft,
            dtype=jnp.float64,
        )[vegetation_type],
        "live_root": jnp.asarray(
            parameters.live_root_mortality_fraction_pft,
            dtype=jnp.float64,
        )[vegetation_type],
        "dead_root": dead_root,
    }
    mortality["live_stem_minus_dead_root"] = _subtract(
        mortality["live_stem"],
        mortality["dead_root"],
    )
    mortality["live_root_minus_dead_root"] = _subtract(
        mortality["live_root"],
        mortality["dead_root"],
    )
    return combustion, mortality


def _set_patch_flux(
    cn_fluxes: CnTransientFluxState,
    field_name: str,
    candidate: jnp.ndarray,
    soil_patch: jnp.ndarray,
) -> CnTransientFluxState:
    previous = get_cn_flux_field(cn_fluxes, field_name)
    return set_cn_flux_field(
        cn_fluxes,
        field_name,
        jnp.where(soil_patch, candidate, previous),
    )


def _assign_patch_fluxes(
    inputs: CnFireFluxInputs,
    cn_fluxes: CnTransientFluxState,
    parameters: CnFireFluxParameters,
    fire_rate: jnp.ndarray,
) -> tuple[CnTransientFluxState, dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    vegetation_type = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    combustion, mortality = _selected_factors(parameters, vegetation_type)
    acceleration = jnp.where(
        jnp.asarray(parameters.spinup_state, dtype=jnp.int32) == jnp.int32(2),
        jnp.float64(10.0),
        jnp.float64(1.0),
    )

    for state_name, flux_name, factor_name, accelerated in (
        *_CARBON_COMBUSTION_ASSIGNMENTS,
        *_NITROGEN_COMBUSTION_ASSIGNMENTS,
    ):
        source = get_cn_precision_control_state_field(inputs.vegetation_state, state_name)
        candidate = _mul(source, fire_rate, combustion[factor_name])
        if accelerated:
            candidate = fortran_multiply(candidate, acceleration)
        cn_fluxes = _set_patch_flux(cn_fluxes, flux_name, candidate, soil_patch)

    for state_name, flux_name, remaining_name, mortality_name, accelerated in (
        *_CARBON_MORTALITY_ASSIGNMENTS,
        *_NITROGEN_MORTALITY_ASSIGNMENTS,
    ):
        source = get_cn_precision_control_state_field(inputs.vegetation_state, state_name)
        candidate = _mul(source, fire_rate)
        if accelerated:
            candidate = fortran_multiply(candidate, acceleration)
        if remaining_name is not None:
            candidate = fortran_multiply(
                candidate,
                _subtract(jnp.float64(1.0), combustion[remaining_name]),
            )
        candidate = fortran_multiply(candidate, mortality[mortality_name])
        cn_fluxes = _set_patch_flux(cn_fluxes, flux_name, candidate, soil_patch)

    for state_name, flux_name, remaining_name, mortality_name in (
        *_CARBON_TRANSFER_ASSIGNMENTS,
        *_NITROGEN_TRANSFER_ASSIGNMENTS,
    ):
        source = get_cn_precision_control_state_field(inputs.vegetation_state, state_name)
        candidate = _mul(source, fire_rate)
        if remaining_name is not None:
            candidate = fortran_multiply(
                candidate,
                _subtract(jnp.float64(1.0), combustion[remaining_name]),
            )
        candidate = fortran_multiply(candidate, mortality[mortality_name])
        cn_fluxes = _set_patch_flux(cn_fluxes, flux_name, candidate, soil_patch)

    return cn_fluxes, combustion, mortality


_PATCH_COLUMN_TERM_TARGETS = (0, 0, 1, 1, 0, 0, 1, 1, 2, 3, 4, 5, 6, 7)


def _patch_column_terms(profiles, leaf_fractions, root_fractions, patch_fluxes, weight, pft):
    leaf_profile, fine_root_profile, coarse_root_profile, stem_profile = profiles
    leaf_labile, leaf_cellulose, leaf_lignin = (fraction[pft] for fraction in leaf_fractions)
    root_labile, root_cellulose, root_lignin = (fraction[pft] for fraction in root_fractions)

    def flux(name):
        return patch_fluxes[name]

    result = []

    def profile_term(value: jnp.ndarray, profile: jnp.ndarray) -> jnp.ndarray:
        return fortran_multiply(value[:, None], profile)

    def weighted_profile_term(value: jnp.ndarray, profile: jnp.ndarray) -> jnp.ndarray:
        return _mul(value[:, None], weight[:, None], profile)

    result.append(
        weighted_profile_term(
            flux("vegetation_carbon.m_deadstemc_to_litter_fire_patch"), stem_profile
        )
    )
    result.append(
        weighted_profile_term(
            flux("vegetation_carbon.m_deadcrootc_to_litter_fire_patch"), coarse_root_profile
        )
    )
    result.append(
        weighted_profile_term(
            flux("vegetation_nitrogen.m_deadstemn_to_litter_fire_patch"), stem_profile
        )
    )
    result.append(
        weighted_profile_term(
            flux("vegetation_nitrogen.m_deadcrootn_to_litter_fire_patch"), coarse_root_profile
        )
    )
    result.append(
        weighted_profile_term(
            flux("vegetation_carbon.m_livestemc_to_litter_fire_patch"), stem_profile
        )
    )
    result.append(
        weighted_profile_term(
            flux("vegetation_carbon.m_livecrootc_to_litter_fire_patch"), coarse_root_profile
        )
    )
    result.append(
        weighted_profile_term(
            flux("vegetation_nitrogen.m_livestemn_to_litter_fire_patch"), stem_profile
        )
    )
    result.append(
        weighted_profile_term(
            flux("vegetation_nitrogen.m_livecrootn_to_litter_fire_patch"), coarse_root_profile
        )
    )
    carbon_leaf_metabolic = _add(
        _mul(flux("vegetation_carbon.m_leafc_to_litter_fire_patch"), leaf_labile),
        flux("vegetation_carbon.m_leafc_storage_to_litter_fire_patch"),
        flux("vegetation_carbon.m_leafc_xfer_to_litter_fire_patch"),
        flux("vegetation_carbon.m_gresp_storage_to_litter_fire_patch"),
        flux("vegetation_carbon.m_gresp_xfer_to_litter_fire_patch"),
    )
    carbon_root_metabolic = _add(
        _mul(flux("vegetation_carbon.m_frootc_to_litter_fire_patch"), root_labile),
        flux("vegetation_carbon.m_frootc_storage_to_litter_fire_patch"),
        flux("vegetation_carbon.m_frootc_xfer_to_litter_fire_patch"),
    )
    carbon_stem_metabolic = _add(
        flux("vegetation_carbon.m_livestemc_storage_to_litter_fire_patch"),
        flux("vegetation_carbon.m_livestemc_xfer_to_litter_fire_patch"),
        flux("vegetation_carbon.m_deadstemc_storage_to_litter_fire_patch"),
        flux("vegetation_carbon.m_deadstemc_xfer_to_litter_fire_patch"),
    )
    carbon_croot_metabolic = _add(
        flux("vegetation_carbon.m_livecrootc_storage_to_litter_fire_patch"),
        flux("vegetation_carbon.m_livecrootc_xfer_to_litter_fire_patch"),
        flux("vegetation_carbon.m_deadcrootc_storage_to_litter_fire_patch"),
        flux("vegetation_carbon.m_deadcrootc_xfer_to_litter_fire_patch"),
    )
    carbon_metabolic = _mul(
        _add(
            profile_term(carbon_leaf_metabolic, leaf_profile),
            profile_term(carbon_root_metabolic, fine_root_profile),
            profile_term(carbon_stem_metabolic, stem_profile),
            profile_term(carbon_croot_metabolic, coarse_root_profile),
        ),
        weight[:, None],
    )
    result.append(carbon_metabolic)
    carbon_cellulose = _mul(
        _add(
            profile_term(
                _mul(flux("vegetation_carbon.m_leafc_to_litter_fire_patch"), leaf_cellulose),
                leaf_profile,
            ),
            profile_term(
                _mul(flux("vegetation_carbon.m_frootc_to_litter_fire_patch"), root_cellulose),
                fine_root_profile,
            ),
        ),
        weight[:, None],
    )
    result.append(carbon_cellulose)
    carbon_lignin = _mul(
        _add(
            profile_term(
                _mul(flux("vegetation_carbon.m_leafc_to_litter_fire_patch"), leaf_lignin),
                leaf_profile,
            ),
            profile_term(
                _mul(flux("vegetation_carbon.m_frootc_to_litter_fire_patch"), root_lignin),
                fine_root_profile,
            ),
        ),
        weight[:, None],
    )
    result.append(carbon_lignin)
    nitrogen_leaf_metabolic = _add(
        _mul(flux("vegetation_nitrogen.m_leafn_to_litter_fire_patch"), leaf_labile),
        flux("vegetation_nitrogen.m_leafn_storage_to_litter_fire_patch"),
        flux("vegetation_nitrogen.m_leafn_xfer_to_litter_fire_patch"),
        flux("vegetation_nitrogen.m_retransn_to_litter_fire_patch"),
    )
    nitrogen_root_metabolic = _add(
        _mul(flux("vegetation_nitrogen.m_frootn_to_litter_fire_patch"), root_labile),
        flux("vegetation_nitrogen.m_frootn_storage_to_litter_fire_patch"),
        flux("vegetation_nitrogen.m_frootn_xfer_to_litter_fire_patch"),
    )
    nitrogen_stem_metabolic = _add(
        flux("vegetation_nitrogen.m_livestemn_storage_to_litter_fire_patch"),
        flux("vegetation_nitrogen.m_livestemn_xfer_to_litter_fire_patch"),
        flux("vegetation_nitrogen.m_deadstemn_storage_to_litter_fire_patch"),
        flux("vegetation_nitrogen.m_deadstemn_xfer_to_litter_fire_patch"),
    )
    nitrogen_croot_metabolic = _add(
        flux("vegetation_nitrogen.m_livecrootn_storage_to_litter_fire_patch"),
        flux("vegetation_nitrogen.m_livecrootn_xfer_to_litter_fire_patch"),
        flux("vegetation_nitrogen.m_deadcrootn_storage_to_litter_fire_patch"),
        flux("vegetation_nitrogen.m_deadcrootn_xfer_to_litter_fire_patch"),
    )
    nitrogen_metabolic = _mul(
        _add(
            profile_term(nitrogen_leaf_metabolic, leaf_profile),
            profile_term(nitrogen_root_metabolic, fine_root_profile),
            profile_term(nitrogen_stem_metabolic, stem_profile),
            profile_term(nitrogen_croot_metabolic, coarse_root_profile),
        ),
        weight[:, None],
    )
    result.append(nitrogen_metabolic)
    nitrogen_cellulose = _mul(
        _add(
            profile_term(
                _mul(flux("vegetation_nitrogen.m_leafn_to_litter_fire_patch"), leaf_cellulose),
                leaf_profile,
            ),
            profile_term(
                _mul(flux("vegetation_nitrogen.m_frootn_to_litter_fire_patch"), root_cellulose),
                fine_root_profile,
            ),
        ),
        weight[:, None],
    )
    result.append(nitrogen_cellulose)
    nitrogen_lignin = _mul(
        _add(
            profile_term(
                _mul(flux("vegetation_nitrogen.m_leafn_to_litter_fire_patch"), leaf_lignin),
                leaf_profile,
            ),
            profile_term(
                _mul(flux("vegetation_nitrogen.m_frootn_to_litter_fire_patch"), root_lignin),
                fine_root_profile,
            ),
        ),
        weight[:, None],
    )
    result.append(nitrogen_lignin)
    return tuple(result)


def _accumulate_patch_fluxes_to_columns(inputs, cn_fluxes, parameters):
    from clm_jax.physics.cn_fire_fluxes_gpu import gather_fire_fluxes

    return gather_fire_fluxes(inputs, cn_fluxes, parameters)


def _accumulate_patch_fluxes_to_columns_reference(
    inputs: CnFireFluxInputs,
    cn_fluxes: CnTransientFluxState,
    parameters: CnFireFluxParameters,
) -> tuple[CnTransientFluxState, jnp.ndarray]:

    patch_count = inputs.soil_patch.shape[0]
    column_patch = jnp.asarray(inputs.column_patch_indices, dtype=jnp.int32)
    if patch_count == 0 or column_patch.shape[1] == 0:
        return cn_fluxes, jnp.int32(0)

    outputs = tuple(get_cn_flux_field(cn_fluxes, name) for name in _COLUMN_OUTPUT_FIELDS)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    patch_active = jnp.asarray(inputs.patch_active, dtype=jnp.bool_)
    patch_weight = jnp.asarray(inputs.patch_weight_on_column, dtype=jnp.float64)
    vegetation_type = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    profiles = tuple(
        jnp.asarray(getattr(inputs, name), dtype=jnp.float64)
        for name in (
            "leaf_profile_patch_level",
            "fine_root_profile_patch_level",
            "coarse_root_profile_patch_level",
            "stem_profile_patch_level",
        )
    )
    leaf_fractions = tuple(
        jnp.asarray(getattr(parameters, name), dtype=jnp.float64)
        for name in (
            "leaf_labile_fraction_pft",
            "leaf_cellulose_fraction_pft",
            "leaf_lignin_fraction_pft",
        )
    )
    root_fractions = tuple(
        jnp.asarray(getattr(parameters, name), dtype=jnp.float64)
        for name in (
            "fine_root_labile_fraction_pft",
            "fine_root_cellulose_fraction_pft",
            "fine_root_lignin_fraction_pft",
        )
    )
    patch_flux_names = {
        flux_name
        for assignments in (
            _CARBON_MORTALITY_ASSIGNMENTS,
            _CARBON_TRANSFER_ASSIGNMENTS,
            _NITROGEN_MORTALITY_ASSIGNMENTS,
            _NITROGEN_TRANSFER_ASSIGNMENTS,
        )
        for _, flux_name, *_ in assignments
    }
    patch_fluxes = {name: get_cn_flux_field(cn_fluxes, name) for name in patch_flux_names}
    active_level = jnp.arange(
        NUMBER_OF_DECOMPOSITION_LEVELS,
        dtype=jnp.int32,
    ) < jnp.asarray(parameters.decomposition_level_count, dtype=jnp.int32)

    def accumulate_patch_slot(slot: int, carry: tuple[jnp.ndarray, ...]) -> tuple[jnp.ndarray, ...]:
        unbounded_patch = column_patch[:, slot]
        valid = unbounded_patch >= jnp.int32(0)
        bounded_patch = jnp.clip(unbounded_patch, 0, patch_count - 1)
        update = (soil_column & valid & patch_active[bounded_patch])[:, None] & active_level[
            None, :
        ]
        weight = patch_weight[bounded_patch]
        pft = vegetation_type[bounded_patch]
        leaf_profile = profiles[0][bounded_patch]
        fine_root_profile = profiles[1][bounded_patch]
        coarse_root_profile = profiles[2][bounded_patch]
        stem_profile = profiles[3][bounded_patch]
        terms = _patch_column_terms(
            (leaf_profile, fine_root_profile, coarse_root_profile, stem_profile),
            leaf_fractions,
            root_fractions,
            {name: values[bounded_patch] for name, values in patch_fluxes.items()},
            weight,
            pft,
        )
        updated = list(carry)
        for target, term in zip(_PATCH_COLUMN_TERM_TARGETS, terms, strict=True):
            updated[target] = jnp.where(update, fortran_add(updated[target], term), updated[target])
        return tuple(updated)

    final = lax.fori_loop(
        0,
        column_patch.shape[1],
        accumulate_patch_slot,
        outputs,
    )
    for name, values in zip(_COLUMN_OUTPUT_FIELDS, final, strict=True):
        cn_fluxes = set_cn_flux_field(cn_fluxes, name, values)

    valid_patch = column_patch >= jnp.int32(0)
    bounded_patch = jnp.clip(column_patch, 0, patch_count - 1)
    active_visit = soil_column[:, None] & valid_patch & patch_active[bounded_patch]
    return cn_fluxes, jnp.sum(active_visit, dtype=jnp.int32)


def _assign_decomposition_fire_fluxes(
    inputs: CnFireFluxInputs,
    cn_fluxes: CnTransientFluxState,
    parameters: CnFireFluxParameters,
) -> CnTransientFluxState:
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    level_mask = jnp.arange(
        NUMBER_OF_DECOMPOSITION_LEVELS,
        dtype=jnp.int32,
    ) < jnp.asarray(parameters.decomposition_level_count, dtype=jnp.int32)
    update = soil_column[:, None, None] & level_mask[None, :, None]
    total_burned = jnp.asarray(
        inputs.total_burned_area_fraction_column,
        dtype=jnp.float64,
    )[:, None, None]
    crop_burned = jnp.asarray(
        inputs.crop_burned_area_fraction_column,
        dtype=jnp.float64,
    )[:, None, None]
    is_litter = jnp.asarray(
        parameters.decomposition_pool_is_litter,
        dtype=jnp.bool_,
    )[None, None, :]
    is_cwd = jnp.asarray(
        parameters.decomposition_pool_is_cwd,
        dtype=jnp.bool_,
    )[None, None, :]
    cwd_fire_rate = _subtract(total_burned, crop_burned)

    assignments = (
        (
            "vegetation_carbon.m_decomp_cpools_to_fire_vr_col",
            jnp.asarray(
                inputs.decomposition_carbon_pool_column_level_pool,
                dtype=jnp.float64,
            ),
        ),
        (
            "vegetation_nitrogen.m_decomp_npools_to_fire_vr_col",
            jnp.asarray(
                inputs.decomposition_nitrogen_pool_column_level_pool,
                dtype=jnp.float64,
            ),
        ),
    )
    for field_name, pools in assignments:
        previous = get_cn_flux_field(cn_fluxes, field_name)
        litter_candidate = _mul(
            pools,
            total_burned,
            jnp.asarray(
                parameters.litter_combustion_completion_fraction,
                dtype=jnp.float64,
            ),
        )
        cwd_candidate = _mul(
            pools,
            cwd_fire_rate,
            jnp.asarray(
                parameters.cwd_combustion_completion_fraction,
                dtype=jnp.float64,
            ),
        )
        candidate = jnp.where(is_litter, litter_candidate, previous)
        candidate = jnp.where(is_cwd, cwd_candidate, candidate)
        cn_fluxes = set_cn_flux_field(
            cn_fluxes,
            field_name,
            jnp.where(update, candidate, previous),
        )
    return cn_fluxes


def _assign_peat_fire_flux(
    inputs: CnFireFluxInputs,
    cn_fluxes: CnTransientFluxState,
    parameters: CnFireFluxParameters,
) -> CnTransientFluxState:
    field_name = "soil_carbon.somc_fire_col"
    previous = get_cn_flux_field(cn_fluxes, field_name)
    gridcell = jnp.asarray(inputs.column_gridcell_index, dtype=jnp.int32)
    latitude = jnp.asarray(inputs.gridcell_latitude_degrees, dtype=jnp.float64)[gridcell]
    peat_burned = jnp.asarray(
        inputs.peat_burned_area_fraction_column,
        dtype=jnp.float64,
    )
    non_boreal = fortran_divide(
        _mul(
            jnp.asarray(inputs.total_soil_organic_carbon_column, dtype=jnp.float64),
            peat_burned,
            jnp.float64(6.0),
        ),
        jnp.float64(33.9),
    )
    boreal = _mul(peat_burned, jnp.float64(2.2e3))
    candidate = jnp.where(
        latitude < jnp.asarray(parameters.boreal_latitude_degrees, dtype=jnp.float64),
        non_boreal,
        boreal,
    )
    return set_cn_flux_field(
        cn_fluxes,
        field_name,
        jnp.where(jnp.asarray(inputs.soil_column, dtype=jnp.bool_), candidate, previous),
    )


def _update_dynamic_vegetation_state(
    inputs: CnFireFluxInputs,
    state: CnFireFluxState,
    cn_fluxes: CnTransientFluxState,
    parameters: CnFireFluxParameters,
    fire_rate: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    use_cndv = jnp.asarray(parameters.use_cndv, dtype=jnp.bool_)
    update = soil_patch & use_cndv
    vegetation_type = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    woody = jnp.asarray(parameters.woody_pft, dtype=jnp.float64)[vegetation_type]
    dead_root_mortality = jnp.asarray(
        parameters.dead_root_mortality_fraction_pft,
        dtype=jnp.float64,
    )[vegetation_type]
    live_stem = get_cn_precision_control_state_field(
        inputs.vegetation_state,
        "vegetation_carbon.livestemc_patch",
    )
    dead_stem = get_cn_precision_control_state_field(
        inputs.vegetation_state,
        "vegetation_carbon.deadstemc_patch",
    )
    survival_loss = _mul(jnp.float64(1.0), dead_root_mortality, fire_rate)
    survival_fraction = _subtract(jnp.float64(1.0), survival_loss)
    surviving_density = fortran_multiply(
        jnp.asarray(state.individual_density_patch, dtype=jnp.float64),
        survival_fraction,
    )
    woody_density = jnp.where(
        _add(live_stem, dead_stem) > jnp.float64(0.0),
        surviving_density,
        jnp.float64(0.0),
    )
    density_candidate = jnp.where(
        woody == jnp.float64(1.0),
        woody_density,
        jnp.asarray(state.individual_density_patch, dtype=jnp.float64),
    )
    individual_density = jnp.where(
        update,
        density_candidate,
        state.individual_density_patch,
    )

    leaf_carbon = get_cn_precision_control_state_field(
        inputs.vegetation_state,
        "vegetation_carbon.leafc_patch",
    )
    leaf_fire = get_cn_flux_field(
        cn_fluxes,
        "vegetation_carbon.m_leafc_to_fire_patch",
    )
    leaf_after_fire = _subtract(
        leaf_carbon,
        _mul(
            leaf_fire,
            jnp.asarray(parameters.time_step, dtype=jnp.float64),
        ),
    )
    maximum_leaf = jnp.maximum(
        leaf_after_fire,
        jnp.asarray(state.maximum_leaf_carbon_patch, dtype=jnp.float64),
    )
    maximum_leaf = jnp.where(
        vegetation_type == jnp.asarray(parameters.bare_ground_pft, dtype=jnp.int32),
        jnp.float64(0.0),
        maximum_leaf,
    )
    maximum_leaf = jnp.where(
        update,
        maximum_leaf,
        state.maximum_leaf_carbon_patch,
    )
    return individual_density, maximum_leaf, update


def _update_transient_landcover_state(
    inputs: CnFireFluxInputs,
    state: CnFireFluxState,
    parameters: CnFireFluxParameters,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    transient = jnp.asarray(parameters.transient_landcover, dtype=jnp.bool_)
    unburned = jnp.asarray(
        state.unburned_conversion_fraction_column,
        dtype=jnp.float64,
    )
    burned_conversion = jnp.where(
        soil_column & transient,
        jnp.float64(0.0),
        jnp.asarray(state.burned_conversion_fraction_column, dtype=jnp.float64),
    )
    not_initial_instant = ~(
        (jnp.asarray(parameters.month, dtype=jnp.int32) == jnp.int32(1))
        & (jnp.asarray(parameters.day_of_month, dtype=jnp.int32) == jnp.int32(1))
        & (jnp.asarray(parameters.seconds_of_day, dtype=jnp.int32) == jnp.int32(0))
    )
    tropical_tree_fraction = _add(
        jnp.asarray(
            inputs.tropical_evergreen_tree_fraction_column,
            dtype=jnp.float64,
        ),
        jnp.asarray(
            inputs.tropical_deciduous_tree_fraction_column,
            dtype=jnp.float64,
        ),
    )
    decreased_tree_fraction = jnp.asarray(
        inputs.decreased_tropical_tree_fraction_column,
        dtype=jnp.float64,
    )
    eligible = (
        soil_column
        & transient
        & not_initial_instant
        & (tropical_tree_fraction > jnp.float64(0.6))
        & (decreased_tree_fraction > jnp.float64(0.0))
        & (unburned > jnp.float64(0.0))
        & (
            jnp.asarray(
                inputs.conversion_fire_burned_area_fraction_column,
                dtype=jnp.float64,
            )
            == jnp.float64(0.0)
        )
    )
    available_fire = _subtract(
        _subtract(
            jnp.asarray(
                inputs.total_burned_area_fraction_column,
                dtype=jnp.float64,
            ),
            jnp.asarray(
                inputs.crop_burned_area_fraction_column,
                dtype=jnp.float64,
            ),
        ),
        jnp.asarray(inputs.peat_burned_area_fraction_column, dtype=jnp.float64),
    )
    time_step = jnp.asarray(parameters.time_step, dtype=jnp.float64)
    burn_this_step_for_rate = _mul(
        fortran_divide(available_fire, jnp.float64(2.0)),
        time_step,
    )
    limited_for_rate = jnp.maximum(
        jnp.float64(0.0),
        jnp.minimum(unburned, burn_this_step_for_rate),
    )
    rate_denominator = _mul(
        decreased_tree_fraction,
        jnp.asarray(parameters.days_per_year, dtype=jnp.float64),
        jnp.asarray(SECONDS_PER_DAY, dtype=jnp.float64),
    )
    rate_denominator = fortran_divide(rate_denominator, time_step)
    safe_denominator = jnp.where(
        eligible,
        rate_denominator,
        jnp.float64(1.0),
    )
    conversion_rate = fortran_divide(limited_for_rate, safe_denominator)
    conversion_rate = fortran_divide(conversion_rate, time_step)
    burned_conversion = jnp.where(eligible, conversion_rate, burned_conversion)

    burn_this_step_for_state = fortran_divide(
        _mul(available_fire, time_step),
        jnp.float64(2.0),
    )
    limited_for_state = jnp.maximum(
        jnp.float64(0.0),
        jnp.minimum(unburned, burn_this_step_for_state),
    )
    unburned_candidate = _subtract(unburned, limited_for_state)
    unburned = jnp.where(eligible, unburned_candidate, unburned)
    return unburned, burned_conversion, eligible


def cn_fire_fluxes(
    inputs: CnFireFluxInputs,
    state: CnFireFluxState,
    cn_fluxes: CnTransientFluxState,
    parameters: CnFireFluxParameters,
) -> CnFireFluxResult:

    _validate_shapes(inputs, state, cn_fluxes, parameters)
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    fire_rate = _fire_rate_patch(inputs, parameters)
    cn_fluxes, _, _ = _assign_patch_fluxes(
        inputs,
        cn_fluxes,
        parameters,
        fire_rate,
    )
    individual_density, maximum_leaf_carbon, dynamic_update = _update_dynamic_vegetation_state(
        inputs,
        state,
        cn_fluxes,
        parameters,
        fire_rate,
    )
    cn_fluxes, active_patch_visits = _accumulate_patch_fluxes_to_columns(
        inputs,
        cn_fluxes,
        parameters,
    )
    cn_fluxes = _assign_decomposition_fire_fluxes(inputs, cn_fluxes, parameters)
    unburned_conversion, burned_conversion, conversion_update = _update_transient_landcover_state(
        inputs, state, parameters
    )
    cn_fluxes = _assign_peat_fire_flux(inputs, cn_fluxes, parameters)
    accelerated = soil_patch & (
        jnp.asarray(parameters.spinup_state, dtype=jnp.int32) == jnp.int32(2)
    )
    return CnFireFluxResult(
        state=CnFireFluxState(
            unburned_conversion_fraction_column=unburned_conversion,
            burned_conversion_fraction_column=burned_conversion,
            individual_density_patch=individual_density,
            maximum_leaf_carbon_patch=maximum_leaf_carbon,
        ),
        cn_fluxes=cn_fluxes,
        diagnostics=CnFireFluxDiagnostics(
            fire_rate_patch=fire_rate,
            updated_patch_count=jnp.sum(soil_patch, dtype=jnp.int32),
            updated_column_count=jnp.sum(soil_column, dtype=jnp.int32),
            active_patch_visit_count=active_patch_visits,
            accelerated_deadwood_patch_count=jnp.sum(accelerated, dtype=jnp.int32),
            dynamic_vegetation_patch_count=jnp.sum(dynamic_update, dtype=jnp.int32),
            conversion_fire_column_count=jnp.sum(conversion_update, dtype=jnp.int32),
        ),
    )


__all__ = [
    "CnFireFluxDiagnostics",
    "CnFireFluxInputs",
    "CnFireFluxParameters",
    "CnFireFluxResult",
    "CnFireFluxState",
    "cn_fire_fluxes",
]
