
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

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
from clm_jax.physics.cn_zero_manifest import NUMBER_OF_DECOMPOSITION_LEVELS
from clm_jax.physics.numerics import (
    fortran_divide,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)

_USE_FUSED_GAP_GATHER = True


class CnGapMortalityInputs(NamedTuple):

    soil_patch: jnp.ndarray
    soil_column: jnp.ndarray
    column_patch_indices: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    patch_weight_on_column: jnp.ndarray
    patch_active: jnp.ndarray
    vegetation_state: CnPrecisionControlState
    growth_efficiency_patch: jnp.ndarray
    heat_stress_patch: jnp.ndarray
    leaf_profile_patch_level: jnp.ndarray
    fine_root_profile_patch_level: jnp.ndarray
    coarse_root_profile_patch_level: jnp.ndarray
    stem_profile_patch_level: jnp.ndarray


class CnGapMortalityState(NamedTuple):

    individual_density_patch: jnp.ndarray


class CnGapMortalityParameters(NamedTuple):

    woody_pft: jnp.ndarray
    leaf_labile_fraction_pft: jnp.ndarray
    leaf_cellulose_fraction_pft: jnp.ndarray
    leaf_lignin_fraction_pft: jnp.ndarray
    fine_root_labile_fraction_pft: jnp.ndarray
    fine_root_cellulose_fraction_pft: jnp.ndarray
    fine_root_lignin_fraction_pft: jnp.ndarray
    annual_mortality_rate: float | jnp.ndarray
    growth_efficiency_coefficient: float | jnp.ndarray
    days_per_year: float | jnp.ndarray = 365.0
    seconds_per_day: float | jnp.ndarray = 86400.0
    first_prognostic_crop_pft: int | jnp.ndarray = 17
    decomposition_level_count: int | jnp.ndarray = 20
    use_cndv: bool | jnp.ndarray = False
    spinup_state: int | jnp.ndarray = 0


class CnGapMortalityDiagnostics(NamedTuple):
    updated_patch_count: jnp.ndarray
    updated_column_count: jnp.ndarray
    active_patch_visit_count: jnp.ndarray
    dynamic_woody_patch_count: jnp.ndarray
    accelerated_deadwood_patch_count: jnp.ndarray


class CnGapMortalityResult(NamedTuple):
    state: CnGapMortalityState
    cn_fluxes: CnTransientFluxState
    diagnostics: CnGapMortalityDiagnostics


_CARBON_PATCH_ASSIGNMENTS = (
    ("vegetation_carbon.leafc_patch", "vegetation_carbon.m_leafc_to_litter_patch", False),
    ("vegetation_carbon.frootc_patch", "vegetation_carbon.m_frootc_to_litter_patch", False),
    ("vegetation_carbon.livestemc_patch", "vegetation_carbon.m_livestemc_to_litter_patch", False),
    ("vegetation_carbon.livecrootc_patch", "vegetation_carbon.m_livecrootc_to_litter_patch", False),
    ("vegetation_carbon.deadstemc_patch", "vegetation_carbon.m_deadstemc_to_litter_patch", True),
    ("vegetation_carbon.deadcrootc_patch", "vegetation_carbon.m_deadcrootc_to_litter_patch", True),
    (
        "vegetation_carbon.leafc_storage_patch",
        "vegetation_carbon.m_leafc_storage_to_litter_patch",
        False,
    ),
    (
        "vegetation_carbon.frootc_storage_patch",
        "vegetation_carbon.m_frootc_storage_to_litter_patch",
        False,
    ),
    (
        "vegetation_carbon.livestemc_storage_patch",
        "vegetation_carbon.m_livestemc_storage_to_litter_patch",
        False,
    ),
    (
        "vegetation_carbon.deadstemc_storage_patch",
        "vegetation_carbon.m_deadstemc_storage_to_litter_patch",
        False,
    ),
    (
        "vegetation_carbon.livecrootc_storage_patch",
        "vegetation_carbon.m_livecrootc_storage_to_litter_patch",
        False,
    ),
    (
        "vegetation_carbon.deadcrootc_storage_patch",
        "vegetation_carbon.m_deadcrootc_storage_to_litter_patch",
        False,
    ),
    (
        "vegetation_carbon.gresp_storage_patch",
        "vegetation_carbon.m_gresp_storage_to_litter_patch",
        False,
    ),
    ("vegetation_carbon.leafc_xfer_patch", "vegetation_carbon.m_leafc_xfer_to_litter_patch", False),
    (
        "vegetation_carbon.frootc_xfer_patch",
        "vegetation_carbon.m_frootc_xfer_to_litter_patch",
        False,
    ),
    (
        "vegetation_carbon.livestemc_xfer_patch",
        "vegetation_carbon.m_livestemc_xfer_to_litter_patch",
        False,
    ),
    (
        "vegetation_carbon.deadstemc_xfer_patch",
        "vegetation_carbon.m_deadstemc_xfer_to_litter_patch",
        False,
    ),
    (
        "vegetation_carbon.livecrootc_xfer_patch",
        "vegetation_carbon.m_livecrootc_xfer_to_litter_patch",
        False,
    ),
    (
        "vegetation_carbon.deadcrootc_xfer_patch",
        "vegetation_carbon.m_deadcrootc_xfer_to_litter_patch",
        False,
    ),
    ("vegetation_carbon.gresp_xfer_patch", "vegetation_carbon.m_gresp_xfer_to_litter_patch", False),
)
_NITROGEN_PATCH_ASSIGNMENTS = (
    ("vegetation_nitrogen.leafn_patch", "vegetation_nitrogen.m_leafn_to_litter_patch", False),
    ("vegetation_nitrogen.frootn_patch", "vegetation_nitrogen.m_frootn_to_litter_patch", False),
    (
        "vegetation_nitrogen.livestemn_patch",
        "vegetation_nitrogen.m_livestemn_to_litter_patch",
        False,
    ),
    (
        "vegetation_nitrogen.livecrootn_patch",
        "vegetation_nitrogen.m_livecrootn_to_litter_patch",
        False,
    ),
    (
        "vegetation_nitrogen.deadstemn_patch",
        "vegetation_nitrogen.m_deadstemn_to_litter_patch",
        True,
    ),
    (
        "vegetation_nitrogen.deadcrootn_patch",
        "vegetation_nitrogen.m_deadcrootn_to_litter_patch",
        True,
    ),
    (
        "vegetation_nitrogen.leafn_storage_patch",
        "vegetation_nitrogen.m_leafn_storage_to_litter_patch",
        False,
    ),
    (
        "vegetation_nitrogen.frootn_storage_patch",
        "vegetation_nitrogen.m_frootn_storage_to_litter_patch",
        False,
    ),
    (
        "vegetation_nitrogen.livestemn_storage_patch",
        "vegetation_nitrogen.m_livestemn_storage_to_litter_patch",
        False,
    ),
    (
        "vegetation_nitrogen.deadstemn_storage_patch",
        "vegetation_nitrogen.m_deadstemn_storage_to_litter_patch",
        False,
    ),
    (
        "vegetation_nitrogen.livecrootn_storage_patch",
        "vegetation_nitrogen.m_livecrootn_storage_to_litter_patch",
        False,
    ),
    (
        "vegetation_nitrogen.deadcrootn_storage_patch",
        "vegetation_nitrogen.m_deadcrootn_storage_to_litter_patch",
        False,
    ),
    (
        "vegetation_nitrogen.leafn_xfer_patch",
        "vegetation_nitrogen.m_leafn_xfer_to_litter_patch",
        False,
    ),
    (
        "vegetation_nitrogen.frootn_xfer_patch",
        "vegetation_nitrogen.m_frootn_xfer_to_litter_patch",
        False,
    ),
    (
        "vegetation_nitrogen.livestemn_xfer_patch",
        "vegetation_nitrogen.m_livestemn_xfer_to_litter_patch",
        False,
    ),
    (
        "vegetation_nitrogen.deadstemn_xfer_patch",
        "vegetation_nitrogen.m_deadstemn_xfer_to_litter_patch",
        False,
    ),
    (
        "vegetation_nitrogen.livecrootn_xfer_patch",
        "vegetation_nitrogen.m_livecrootn_xfer_to_litter_patch",
        False,
    ),
    (
        "vegetation_nitrogen.deadcrootn_xfer_patch",
        "vegetation_nitrogen.m_deadcrootn_xfer_to_litter_patch",
        False,
    ),
)
_RETRANS_N_STATE = "vegetation_nitrogen.retransn_patch"
_RETRANS_N_FLUX = "vegetation_nitrogen.m_retransn_to_litter_patch"

_CARBON_COLUMN_OUTPUTS = (
    "vegetation_carbon.gap_mortality_c_to_litr_met_c_col",
    "vegetation_carbon.gap_mortality_c_to_litr_cel_c_col",
    "vegetation_carbon.gap_mortality_c_to_litr_lig_c_col",
    "vegetation_carbon.gap_mortality_c_to_cwdc_col",
)
_NITROGEN_COLUMN_OUTPUTS = (
    "vegetation_nitrogen.gap_mortality_n_to_litr_met_n_col",
    "vegetation_nitrogen.gap_mortality_n_to_litr_cel_n_col",
    "vegetation_nitrogen.gap_mortality_n_to_litr_lig_n_col",
    "vegetation_nitrogen.gap_mortality_n_to_cwdn_col",
)


def _validate_shapes(
    inputs: CnGapMortalityInputs,
    state: CnGapMortalityState,
    cn_fluxes: CnTransientFluxState,
    parameters: CnGapMortalityParameters,
) -> tuple[int, int]:
    if jnp.ndim(inputs.soil_patch) != 1 or jnp.ndim(inputs.soil_column) != 1:
        raise ValueError("soil_patch and soil_column must be vectors")
    patch_count = inputs.soil_patch.shape[0]
    column_count = inputs.soil_column.shape[0]
    if (
        jnp.shape(inputs.column_patch_indices)[0:1] != (column_count,)
        or jnp.ndim(inputs.column_patch_indices) != 2
    ):
        raise ValueError("column_patch_indices must have shape (column, patch slot)")
    for name in (
        "vegetation_type_patch",
        "patch_weight_on_column",
        "patch_active",
        "growth_efficiency_patch",
        "heat_stress_patch",
    ):
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
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
    if jnp.shape(state.individual_density_patch) != (patch_count,):
        raise ValueError("individual_density_patch must contain every patch")
    if cn_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("CN patch fluxes must match patch count")
    if cn_fluxes.column_level.shape[0:2] != (
        column_count,
        NUMBER_OF_DECOMPOSITION_LEVELS,
    ):
        raise ValueError("CN column-level fluxes have incompatible dimensions")

    pft_count = None
    for name in parameters._fields[:7]:
        values = getattr(parameters, name)
        if jnp.ndim(values) != 1 or values.shape[0] < 1:
            raise ValueError(f"{name} must be a nonempty PFT vector")
        if pft_count is None:
            pft_count = values.shape[0]
        elif values.shape[0] != pft_count:
            raise ValueError("all PFT parameter vectors must have equal length")
    for name in parameters._fields[7:]:
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    return patch_count, column_count


def _mortality_rate_per_second(
    inputs: CnGapMortalityInputs,
    parameters: CnGapMortalityParameters,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    vegetation_type = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    woody = jnp.asarray(parameters.woody_pft, dtype=jnp.float64)[vegetation_type]
    is_woody = woody == jnp.float64(1.0)
    mortality_maximum = jnp.where(
        vegetation_type == jnp.int32(8),
        jnp.float64(0.03),
        jnp.float64(0.01),
    )
    efficiency_term = fortran_multiply(
        jnp.asarray(parameters.growth_efficiency_coefficient, dtype=jnp.float64),
        jnp.asarray(inputs.growth_efficiency_patch, dtype=jnp.float64),
    )
    efficiency_denominator = fortran_add(jnp.float64(1.0), efficiency_term)
    dynamic_rate = fortran_divide(mortality_maximum, efficiency_denominator)
    dynamic_rate = fortran_add(
        dynamic_rate,
        jnp.asarray(inputs.heat_stress_patch, dtype=jnp.float64),
    )
    dynamic_rate = jnp.minimum(jnp.float64(1.0), dynamic_rate)
    annual_rate = jnp.asarray(parameters.annual_mortality_rate, dtype=jnp.float64)
    dynamic_or_default = jnp.where(is_woody, dynamic_rate, annual_rate)
    use_cndv = jnp.asarray(parameters.use_cndv, dtype=jnp.bool_)
    selected_rate = jnp.where(use_cndv, dynamic_or_default, annual_rate)
    seconds_per_year = fortran_multiply(
        jnp.asarray(parameters.days_per_year, dtype=jnp.float64),
        jnp.asarray(parameters.seconds_per_day, dtype=jnp.float64),
    )
    return fortran_divide(selected_rate, seconds_per_year), is_woody


def _assign_patch_fluxes(
    inputs: CnGapMortalityInputs,
    cn_fluxes: CnTransientFluxState,
    parameters: CnGapMortalityParameters,
    mortality_rate: jnp.ndarray,
) -> CnTransientFluxState:
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    accelerate_deadwood = (
        jnp.asarray(parameters.spinup_state, dtype=jnp.int32) == jnp.int32(2)
    ) & ~jnp.asarray(parameters.use_cndv, dtype=jnp.bool_)

    for state_name, flux_name, accelerated in _CARBON_PATCH_ASSIGNMENTS:
        source = get_cn_precision_control_state_field(
            inputs.vegetation_state,
            state_name,
        )
        candidate = fortran_multiply(source, mortality_rate)
        if accelerated:
            fast_candidate = fortran_multiply(candidate, jnp.float64(10.0))
            candidate = jnp.where(
                accelerate_deadwood,
                fast_candidate,
                candidate,
            )
        previous = get_cn_flux_field(cn_fluxes, flux_name)
        cn_fluxes = set_cn_flux_field(
            cn_fluxes,
            flux_name,
            jnp.where(soil_patch, candidate, previous),
        )

    for state_name, flux_name, accelerated in _NITROGEN_PATCH_ASSIGNMENTS:
        source = get_cn_precision_control_state_field(
            inputs.vegetation_state,
            state_name,
        )
        candidate = fortran_multiply(source, mortality_rate)
        if accelerated:
            fast_candidate = fortran_multiply(candidate, jnp.float64(10.0))
            candidate = jnp.where(
                accelerate_deadwood,
                fast_candidate,
                candidate,
            )
        previous = get_cn_flux_field(cn_fluxes, flux_name)
        cn_fluxes = set_cn_flux_field(
            cn_fluxes,
            flux_name,
            jnp.where(soil_patch, candidate, previous),
        )

    retransmitted = fortran_multiply(
        get_cn_precision_control_state_field(
            inputs.vegetation_state,
            _RETRANS_N_STATE,
        ),
        mortality_rate,
    )
    previous_retransmitted = get_cn_flux_field(cn_fluxes, _RETRANS_N_FLUX)
    noncrop_patch = jnp.asarray(
        inputs.vegetation_type_patch,
        dtype=jnp.int32,
    ) < jnp.asarray(parameters.first_prognostic_crop_pft, dtype=jnp.int32)
    return set_cn_flux_field(
        cn_fluxes,
        _RETRANS_N_FLUX,
        jnp.where(
            soil_patch & noncrop_patch,
            retransmitted,
            previous_retransmitted,
        ),
    )


def _multiply_term(
    flux: jnp.ndarray,
    weight: jnp.ndarray,
    profile: jnp.ndarray,
    fraction: jnp.ndarray | None = None,
) -> jnp.ndarray:
    term = flux
    if fraction is not None:
        term = fortran_multiply(term, fraction)
    term = fortran_multiply(term, weight)
    return fortran_multiply(term[:, None], profile)


def _accumulate_patch_fluxes_to_columns(
    inputs: CnGapMortalityInputs,
    cn_fluxes: CnTransientFluxState,
    parameters: CnGapMortalityParameters,
) -> tuple[CnTransientFluxState, jnp.ndarray]:
    patch_count = inputs.soil_patch.shape[0]
    outputs = tuple(
        get_cn_flux_field(cn_fluxes, name)
        for name in (*_CARBON_COLUMN_OUTPUTS, *_NITROGEN_COLUMN_OUTPUTS)
    )
    if patch_count == 0 or inputs.column_patch_indices.shape[1] == 0:
        return cn_fluxes, jnp.int32(0)

    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    column_patch = jnp.asarray(inputs.column_patch_indices, dtype=jnp.int32)
    vegetation_type_patch = jnp.asarray(
        inputs.vegetation_type_patch,
        dtype=jnp.int32,
    )
    patch_weight = jnp.asarray(inputs.patch_weight_on_column, dtype=jnp.float64)
    patch_active = jnp.asarray(inputs.patch_active, dtype=jnp.bool_)
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
    carbon_patch = {
        name: get_cn_flux_field(cn_fluxes, name) for _, name, _ in _CARBON_PATCH_ASSIGNMENTS
    }
    nitrogen_patch = {
        name: get_cn_flux_field(cn_fluxes, name) for _, name, _ in _NITROGEN_PATCH_ASSIGNMENTS
    }
    retransmitted_n = get_cn_flux_field(cn_fluxes, _RETRANS_N_FLUX)
    active_level = jnp.arange(
        NUMBER_OF_DECOMPOSITION_LEVELS,
        dtype=jnp.int32,
    ) < jnp.asarray(parameters.decomposition_level_count, dtype=jnp.int32)

    def accumulate_patch_slot(
        slot: int,
        carry: tuple[jnp.ndarray, ...],
    ) -> tuple[jnp.ndarray, ...]:
        unbounded_patch = column_patch[:, slot]
        valid = unbounded_patch >= jnp.int32(0)
        bounded_patch = jnp.clip(unbounded_patch, 0, patch_count - 1)
        update = (soil_column & valid & patch_active[bounded_patch])[:, None] & active_level[
            None, :
        ]
        vegetation_type = vegetation_type_patch[bounded_patch]
        weight = patch_weight[bounded_patch]
        leaf_profile = profiles[0][bounded_patch]
        fine_root_profile = profiles[1][bounded_patch]
        coarse_root_profile = profiles[2][bounded_patch]
        stem_profile = profiles[3][bounded_patch]
        selected_leaf_fractions = tuple(values[vegetation_type] for values in leaf_fractions)
        selected_root_fractions = tuple(values[vegetation_type] for values in root_fractions)

        def flux(values: dict[str, jnp.ndarray], name: str) -> jnp.ndarray:
            return values[name][bounded_patch]

        def add(
            output_index: int,
            mortality_flux: jnp.ndarray,
            profile: jnp.ndarray,
            fraction: jnp.ndarray | None = None,
        ) -> None:
            current = updated[output_index]
            candidate = fortran_add(
                current,
                _multiply_term(mortality_flux, weight, profile, fraction),
            )
            updated[output_index] = jnp.where(update, candidate, current)

        updated = list(carry)
        add(
            0,
            flux(carbon_patch, "vegetation_carbon.m_leafc_to_litter_patch"),
            leaf_profile,
            selected_leaf_fractions[0],
        )
        add(
            1,
            flux(carbon_patch, "vegetation_carbon.m_leafc_to_litter_patch"),
            leaf_profile,
            selected_leaf_fractions[1],
        )
        add(
            2,
            flux(carbon_patch, "vegetation_carbon.m_leafc_to_litter_patch"),
            leaf_profile,
            selected_leaf_fractions[2],
        )
        add(
            0,
            flux(carbon_patch, "vegetation_carbon.m_frootc_to_litter_patch"),
            fine_root_profile,
            selected_root_fractions[0],
        )
        add(
            1,
            flux(carbon_patch, "vegetation_carbon.m_frootc_to_litter_patch"),
            fine_root_profile,
            selected_root_fractions[1],
        )
        add(
            2,
            flux(carbon_patch, "vegetation_carbon.m_frootc_to_litter_patch"),
            fine_root_profile,
            selected_root_fractions[2],
        )
        add(
            3,
            fortran_add(
                flux(carbon_patch, "vegetation_carbon.m_livestemc_to_litter_patch"),
                flux(carbon_patch, "vegetation_carbon.m_deadstemc_to_litter_patch"),
            ),
            stem_profile,
        )
        add(
            3,
            fortran_add(
                flux(carbon_patch, "vegetation_carbon.m_livecrootc_to_litter_patch"),
                flux(carbon_patch, "vegetation_carbon.m_deadcrootc_to_litter_patch"),
            ),
            coarse_root_profile,
        )

        add(
            0,
            fortran_add(
                flux(carbon_patch, "vegetation_carbon.m_leafc_storage_to_litter_patch"),
                flux(carbon_patch, "vegetation_carbon.m_gresp_storage_to_litter_patch"),
            ),
            leaf_profile,
        )
        add(
            0,
            flux(carbon_patch, "vegetation_carbon.m_frootc_storage_to_litter_patch"),
            fine_root_profile,
        )
        add(
            0,
            fortran_add(
                flux(carbon_patch, "vegetation_carbon.m_livestemc_storage_to_litter_patch"),
                flux(carbon_patch, "vegetation_carbon.m_deadstemc_storage_to_litter_patch"),
            ),
            stem_profile,
        )
        add(
            0,
            fortran_add(
                flux(carbon_patch, "vegetation_carbon.m_livecrootc_storage_to_litter_patch"),
                flux(carbon_patch, "vegetation_carbon.m_deadcrootc_storage_to_litter_patch"),
            ),
            coarse_root_profile,
        )
        add(
            0,
            fortran_add(
                flux(carbon_patch, "vegetation_carbon.m_leafc_xfer_to_litter_patch"),
                flux(carbon_patch, "vegetation_carbon.m_gresp_xfer_to_litter_patch"),
            ),
            leaf_profile,
        )
        add(
            0,
            flux(carbon_patch, "vegetation_carbon.m_frootc_xfer_to_litter_patch"),
            fine_root_profile,
        )
        add(
            0,
            fortran_add(
                flux(carbon_patch, "vegetation_carbon.m_livestemc_xfer_to_litter_patch"),
                flux(carbon_patch, "vegetation_carbon.m_deadstemc_xfer_to_litter_patch"),
            ),
            stem_profile,
        )
        add(
            0,
            fortran_add(
                flux(carbon_patch, "vegetation_carbon.m_livecrootc_xfer_to_litter_patch"),
                flux(carbon_patch, "vegetation_carbon.m_deadcrootc_xfer_to_litter_patch"),
            ),
            coarse_root_profile,
        )

        add(
            4,
            flux(nitrogen_patch, "vegetation_nitrogen.m_leafn_to_litter_patch"),
            leaf_profile,
            selected_leaf_fractions[0],
        )
        add(
            5,
            flux(nitrogen_patch, "vegetation_nitrogen.m_leafn_to_litter_patch"),
            leaf_profile,
            selected_leaf_fractions[1],
        )
        add(
            6,
            flux(nitrogen_patch, "vegetation_nitrogen.m_leafn_to_litter_patch"),
            leaf_profile,
            selected_leaf_fractions[2],
        )
        add(
            4,
            flux(nitrogen_patch, "vegetation_nitrogen.m_frootn_to_litter_patch"),
            fine_root_profile,
            selected_root_fractions[0],
        )
        add(
            5,
            flux(nitrogen_patch, "vegetation_nitrogen.m_frootn_to_litter_patch"),
            fine_root_profile,
            selected_root_fractions[1],
        )
        add(
            6,
            flux(nitrogen_patch, "vegetation_nitrogen.m_frootn_to_litter_patch"),
            fine_root_profile,
            selected_root_fractions[2],
        )
        add(
            7,
            fortran_add(
                flux(nitrogen_patch, "vegetation_nitrogen.m_livestemn_to_litter_patch"),
                flux(nitrogen_patch, "vegetation_nitrogen.m_deadstemn_to_litter_patch"),
            ),
            stem_profile,
        )
        add(
            7,
            fortran_add(
                flux(nitrogen_patch, "vegetation_nitrogen.m_livecrootn_to_litter_patch"),
                flux(nitrogen_patch, "vegetation_nitrogen.m_deadcrootn_to_litter_patch"),
            ),
            coarse_root_profile,
        )
        add(4, retransmitted_n[bounded_patch], leaf_profile)

        add(
            4,
            flux(nitrogen_patch, "vegetation_nitrogen.m_leafn_storage_to_litter_patch"),
            leaf_profile,
        )
        add(
            4,
            flux(nitrogen_patch, "vegetation_nitrogen.m_frootn_storage_to_litter_patch"),
            fine_root_profile,
        )
        add(
            4,
            fortran_add(
                flux(nitrogen_patch, "vegetation_nitrogen.m_livestemn_storage_to_litter_patch"),
                flux(nitrogen_patch, "vegetation_nitrogen.m_deadstemn_storage_to_litter_patch"),
            ),
            stem_profile,
        )
        add(
            4,
            fortran_add(
                flux(nitrogen_patch, "vegetation_nitrogen.m_livecrootn_storage_to_litter_patch"),
                flux(nitrogen_patch, "vegetation_nitrogen.m_deadcrootn_storage_to_litter_patch"),
            ),
            coarse_root_profile,
        )
        add(
            4,
            flux(nitrogen_patch, "vegetation_nitrogen.m_leafn_xfer_to_litter_patch"),
            leaf_profile,
        )
        add(
            4,
            flux(nitrogen_patch, "vegetation_nitrogen.m_frootn_xfer_to_litter_patch"),
            fine_root_profile,
        )
        add(
            4,
            fortran_add(
                flux(nitrogen_patch, "vegetation_nitrogen.m_livestemn_xfer_to_litter_patch"),
                flux(nitrogen_patch, "vegetation_nitrogen.m_deadstemn_xfer_to_litter_patch"),
            ),
            stem_profile,
        )
        add(
            4,
            fortran_add(
                flux(nitrogen_patch, "vegetation_nitrogen.m_livecrootn_xfer_to_litter_patch"),
                flux(nitrogen_patch, "vegetation_nitrogen.m_deadcrootn_xfer_to_litter_patch"),
            ),
            coarse_root_profile,
        )
        return tuple(updated)

    final = lax.fori_loop(
        0,
        column_patch.shape[1],
        accumulate_patch_slot,
        outputs,
    )
    for name, values in zip(
        (*_CARBON_COLUMN_OUTPUTS, *_NITROGEN_COLUMN_OUTPUTS),
        final,
        strict=True,
    ):
        cn_fluxes = set_cn_flux_field(cn_fluxes, name, values)

    valid_patch = column_patch >= jnp.int32(0)
    bounded_patch = jnp.clip(column_patch, 0, patch_count - 1)
    active_visit = soil_column[:, None] & valid_patch & patch_active[bounded_patch]
    return cn_fluxes, jnp.sum(active_visit, dtype=jnp.int32)


def cn_gap_mortality(
    inputs: CnGapMortalityInputs,
    state: CnGapMortalityState,
    cn_fluxes: CnTransientFluxState,
    parameters: CnGapMortalityParameters,
) -> CnGapMortalityResult:

    _validate_shapes(inputs, state, cn_fluxes, parameters)
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    mortality_rate, is_woody = _mortality_rate_per_second(inputs, parameters)
    cn_fluxes = _assign_patch_fluxes(
        inputs,
        cn_fluxes,
        parameters,
        mortality_rate,
    )
    gather = _accumulate_patch_fluxes_to_columns
    if _USE_FUSED_GAP_GATHER and jax.default_backend() == "gpu":
        from clm_jax.physics.cn_gap_mortality_gpu import gather_gap_fluxes
        gather = gather_gap_fluxes
    cn_fluxes, active_patch_visits = gather(
        inputs,
        cn_fluxes,
        parameters,
    )

    dynamic_woody = soil_patch & jnp.asarray(parameters.use_cndv, dtype=jnp.bool_) & is_woody
    live_plus_dead_stem = fortran_add(
        get_cn_precision_control_state_field(
            inputs.vegetation_state,
            "vegetation_carbon.livestemc_patch",
        ),
        get_cn_precision_control_state_field(
            inputs.vegetation_state,
            "vegetation_carbon.deadstemc_patch",
        ),
    )
    survival_fraction = fortran_add(jnp.float64(1.0), -mortality_rate)
    surviving_density = fortran_multiply(
        jnp.asarray(state.individual_density_patch, dtype=jnp.float64),
        survival_fraction,
    )
    dynamic_density = jnp.where(
        live_plus_dead_stem > jnp.float64(0.0),
        surviving_density,
        jnp.float64(0.0),
    )
    individual_density = jnp.where(
        dynamic_woody,
        dynamic_density,
        state.individual_density_patch,
    )
    accelerate_deadwood = (
        soil_patch
        & (jnp.asarray(parameters.spinup_state, dtype=jnp.int32) == jnp.int32(2))
        & ~jnp.asarray(parameters.use_cndv, dtype=jnp.bool_)
    )
    return CnGapMortalityResult(
        state=CnGapMortalityState(
            individual_density_patch=individual_density,
        ),
        cn_fluxes=cn_fluxes,
        diagnostics=CnGapMortalityDiagnostics(
            updated_patch_count=jnp.sum(soil_patch, dtype=jnp.int32),
            updated_column_count=jnp.sum(soil_column, dtype=jnp.int32),
            active_patch_visit_count=active_patch_visits,
            dynamic_woody_patch_count=jnp.sum(dynamic_woody, dtype=jnp.int32),
            accelerated_deadwood_patch_count=jnp.sum(
                accelerate_deadwood,
                dtype=jnp.int32,
            ),
        ),
    )


__all__ = [
    "CnGapMortalityDiagnostics",
    "CnGapMortalityInputs",
    "CnGapMortalityParameters",
    "CnGapMortalityResult",
    "CnGapMortalityState",
    "cn_gap_mortality",
]
