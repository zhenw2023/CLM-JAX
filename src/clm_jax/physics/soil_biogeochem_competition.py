
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import lax

from clm_jax.floating_point import (
    correctly_rounded_divide,
    nonnegative_divide_preserving_subnormal,
)
from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    get_cn_flux_field,
    set_cn_flux_field,
)
from clm_jax.physics.cn_zero_manifest import NUMBER_OF_DECOMPOSITION_LEVELS
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.numerics import (
    fortran_ordered_sum,
)

NUMBER_OF_SOIL_LEVELS = 20

_ACTUAL_IMMOBILIZATION = "soil_nitrogen.actual_immob_col"
_ACTUAL_IMMOBILIZATION_LEVEL = "soil_nitrogen.actual_immob_vr_col"
_ACTUAL_IMMOBILIZATION_NH4 = "soil_nitrogen.actual_immob_nh4_vr_col"
_ACTUAL_IMMOBILIZATION_NO3 = "soil_nitrogen.actual_immob_no3_vr_col"
_DENITRIFICATION = "soil_nitrogen.f_denit_vr_col"
_DENITRIFICATION_N2O = "soil_nitrogen.f_n2o_denit_vr_col"
_NITRIFICATION = "soil_nitrogen.f_nit_vr_col"
_NITRIFICATION_N2O = "soil_nitrogen.f_n2o_nit_vr_col"
_POTENTIAL_IMMOBILIZATION = "soil_nitrogen.potential_immob_col"
_POTENTIAL_IMMOBILIZATION_LEVEL = "soil_nitrogen.potential_immob_vr_col"
_PLANT_NITROGEN = "soil_nitrogen.sminn_to_plant_col"
_PLANT_NITROGEN_LEVEL = "soil_nitrogen.sminn_to_plant_vr_col"
_PLANT_NH4_LEVEL = "soil_nitrogen.smin_nh4_to_plant_vr_col"
_PLANT_NO3_LEVEL = "soil_nitrogen.smin_no3_to_plant_vr_col"
_FUN_PLANT_NH4_LEVEL = "soil_nitrogen.sminn_to_plant_fun_nh4_vr_col"
_FUN_PLANT_NO3_LEVEL = "soil_nitrogen.sminn_to_plant_fun_no3_vr_col"
_SUPPLEMENT_LEVEL = "soil_nitrogen.supplement_to_sminn_vr_col"


class SoilBiogeochemCompetitionInputs(NamedTuple):

    soil_column: jnp.ndarray
    plant_nitrogen_demand_column: jnp.ndarray
    nitrogen_fixation_profile_column_level: jnp.ndarray
    mineral_nitrogen_column_level: jnp.ndarray
    mineral_ammonium_column_level: jnp.ndarray
    mineral_nitrate_column_level: jnp.ndarray
    potential_nitrification_column_level: jnp.ndarray
    potential_denitrification_column_level: jnp.ndarray
    potential_immobilization_column_level: jnp.ndarray
    n2_to_n2o_ratio_denitrification_column_level: jnp.ndarray


class SoilBiogeochemCompetitionState(NamedTuple):

    fraction_potential_growth_column: jnp.ndarray
    fraction_potential_immobilization_column: jnp.ndarray
    fraction_potential_immobilization_column_level: jnp.ndarray


class SoilBiogeochemCompetitionParameters(NamedTuple):

    soil_layer_thickness: jnp.ndarray
    plant_competitiveness_nitrate: float | jnp.ndarray = 1.0
    plant_competitiveness_ammonium: float | jnp.ndarray = 1.0
    decomposer_competitiveness_nitrate: float | jnp.ndarray = 1.0
    decomposer_competitiveness_ammonium: float | jnp.ndarray = 1.0
    denitrifier_competitiveness: float | jnp.ndarray = 1.0
    nitrifier_competitiveness: float | jnp.ndarray = 1.0
    nitrification_n2o_loss_fraction: float | jnp.ndarray = 6.0e-4
    time_step_seconds: float | jnp.ndarray = 1800.0
    decomposition_level_count: int | jnp.ndarray = NUMBER_OF_SOIL_LEVELS
    carbon_only: bool | jnp.ndarray = False


DEFAULT_SOIL_BIOGEOCHEM_COMPETITION_PARAMETERS = (
    SoilBiogeochemCompetitionParameters(
        soil_layer_thickness=np.ones(
            NUMBER_OF_DECOMPOSITION_LEVELS, dtype=np.float64
        )
    )
)


class SoilBiogeochemCompetitionDiagnostics(NamedTuple):
    updated_column_count: jnp.ndarray
    ammonium_limited_cell_count: jnp.ndarray
    nitrate_limited_cell_count: jnp.ndarray
    maximum_fun_nitrate_excess: jnp.ndarray
    maximum_fun_ammonium_excess: jnp.ndarray


class SoilBiogeochemCompetitionPreFunResult(NamedTuple):

    cn_fluxes: CnTransientFluxState
    state: SoilBiogeochemCompetitionState
    nitrogen_uptake_profile_column_level: jnp.ndarray
    diagnostics: SoilBiogeochemCompetitionDiagnostics


class SoilBiogeochemCompetitionResult(NamedTuple):

    cn_fluxes: CnTransientFluxState
    state: SoilBiogeochemCompetitionState
    diagnostics: SoilBiogeochemCompetitionDiagnostics


def _validate_shapes(
    inputs: SoilBiogeochemCompetitionInputs,
    state: SoilBiogeochemCompetitionState,
    cn_fluxes: CnTransientFluxState,
    parameters: SoilBiogeochemCompetitionParameters,
) -> int:
    if jnp.ndim(inputs.soil_column) != 1:
        raise ValueError("soil_column must be a vector")
    column_count = inputs.soil_column.shape[0]
    scalar_shape = (column_count,)
    level_shape = (column_count, NUMBER_OF_DECOMPOSITION_LEVELS)
    if jnp.shape(inputs.plant_nitrogen_demand_column) != scalar_shape:
        raise ValueError("plant_nitrogen_demand_column must contain every column")
    for name in SoilBiogeochemCompetitionInputs._fields[2:]:
        if jnp.shape(getattr(inputs, name)) != level_shape:
            raise ValueError(f"{name} must have shape (column, 25)")
    if jnp.shape(state.fraction_potential_growth_column) != scalar_shape:
        raise ValueError("fraction_potential_growth_column must contain every column")
    if jnp.shape(state.fraction_potential_immobilization_column) != scalar_shape:
        raise ValueError(
            "fraction_potential_immobilization_column must contain every column"
        )
    if jnp.shape(state.fraction_potential_immobilization_column_level) != level_shape:
        raise ValueError(
            "fraction_potential_immobilization_column_level must have shape (column, 25)"
        )
    if jnp.shape(parameters.soil_layer_thickness) != (
        NUMBER_OF_DECOMPOSITION_LEVELS,
    ):
        raise ValueError("soil_layer_thickness must contain 25 levels")
    for field in (
        _ACTUAL_IMMOBILIZATION_LEVEL,
        _ACTUAL_IMMOBILIZATION_NH4,
        _ACTUAL_IMMOBILIZATION_NO3,
        _DENITRIFICATION,
        _DENITRIFICATION_N2O,
        _NITRIFICATION,
        _NITRIFICATION_N2O,
        _POTENTIAL_IMMOBILIZATION_LEVEL,
        _PLANT_NITROGEN_LEVEL,
        _PLANT_NH4_LEVEL,
        _PLANT_NO3_LEVEL,
        _FUN_PLANT_NH4_LEVEL,
        _FUN_PLANT_NO3_LEVEL,
        _SUPPLEMENT_LEVEL,
    ):
        if get_cn_flux_field(cn_fluxes, field).shape != level_shape:
            raise ValueError(f"{field} must have shape (column, 25)")
    return column_count


def _validate_parameters(parameters: SoilBiogeochemCompetitionParameters) -> None:
    for name in SoilBiogeochemCompetitionParameters._fields[1:]:
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    count = parameters.decomposition_level_count
    if isinstance(count, (int, np.integer)) and not (
        1 <= count <= NUMBER_OF_DECOMPOSITION_LEVELS
    ):
        raise ValueError("decomposition_level_count must lie in [1, 25]")


def _divide(numerator: jnp.ndarray, denominator: jnp.ndarray) -> jnp.ndarray:
    return correctly_rounded_divide(numerator, denominator)


def _sum_three(first: jnp.ndarray, second: jnp.ndarray, third: jnp.ndarray):
    return fortran_add(fortran_add(first, second), third)


def _filtered_level_update(
    old: jnp.ndarray,
    new: jnp.ndarray,
    active_cell: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.where(active_cell, new, old)


def _binary64_is_zero(value: jnp.ndarray) -> jnp.ndarray:
    bits = lax.bitcast_convert_type(
        jnp.asarray(value, dtype=jnp.float64),
        jnp.uint64,
    )
    return (bits & jnp.uint64(0x7FFFFFFFFFFFFFFF)) == jnp.uint64(0)


def _binary64_less(
    left: jnp.ndarray,
    right: jnp.ndarray,
) -> jnp.ndarray:

    left, right = jnp.broadcast_arrays(
        jnp.asarray(left, dtype=jnp.float64),
        jnp.asarray(right, dtype=jnp.float64),
    )
    left_bits = lax.bitcast_convert_type(left, jnp.uint64)
    right_bits = lax.bitcast_convert_type(right, jnp.uint64)
    sign_mask = jnp.uint64(0x8000000000000000)
    magnitude_mask = jnp.uint64(0x7FFFFFFFFFFFFFFF)
    left_negative = (left_bits & sign_mask) != jnp.uint64(0)
    right_negative = (right_bits & sign_mask) != jnp.uint64(0)
    both_zero = (
        ((left_bits & magnitude_mask) == jnp.uint64(0))
        & ((right_bits & magnitude_mask) == jnp.uint64(0))
    )
    different_sign = left_negative != right_negative
    same_sign_order = jnp.where(
        left_negative,
        left_bits > right_bits,
        left_bits < right_bits,
    )
    return (~both_zero) & jnp.where(
        different_sign,
        left_negative,
        same_sign_order,
    )


def soil_biogeochem_competition_pre_fun(
    inputs: SoilBiogeochemCompetitionInputs,
    state: SoilBiogeochemCompetitionState,
    cn_fluxes: CnTransientFluxState,
    parameters: SoilBiogeochemCompetitionParameters = (
        DEFAULT_SOIL_BIOGEOCHEM_COMPETITION_PARAMETERS
    ),
) -> SoilBiogeochemCompetitionPreFunResult:

    _validate_shapes(inputs, state, cn_fluxes, parameters)
    _validate_parameters(parameters)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    level_index = jnp.arange(NUMBER_OF_DECOMPOSITION_LEVELS, dtype=jnp.int32)
    active_level = level_index < jnp.asarray(
        parameters.decomposition_level_count, dtype=jnp.int32
    )
    active_cell = soil_column[:, None] & active_level[None, :]
    thickness = jnp.asarray(parameters.soil_layer_thickness, dtype=jnp.float64)
    dt = jnp.asarray(parameters.time_step_seconds, dtype=jnp.float64)
    plant_demand = jnp.asarray(
        inputs.plant_nitrogen_demand_column, dtype=jnp.float64
    )[:, None]
    mineral_n = jnp.asarray(inputs.mineral_nitrogen_column_level, dtype=jnp.float64)
    ammonium = jnp.asarray(inputs.mineral_ammonium_column_level, dtype=jnp.float64)
    nitrate = jnp.asarray(inputs.mineral_nitrate_column_level, dtype=jnp.float64)
    potential_nitrification = jnp.asarray(
        inputs.potential_nitrification_column_level, dtype=jnp.float64
    )
    potential_denitrification = jnp.asarray(
        inputs.potential_denitrification_column_level, dtype=jnp.float64
    )
    potential_immobilization = jnp.asarray(
        inputs.potential_immobilization_column_level, dtype=jnp.float64
    )
    n2_to_n2o = jnp.asarray(
        inputs.n2_to_n2o_ratio_denitrification_column_level,
        dtype=jnp.float64,
    )

    mineral_sum = fortran_add(nitrate, ammonium)
    mineral_mass = fortran_multiply(mineral_sum, thickness[None, :])
    mineral_total = fortran_ordered_sum(
        jnp.where(active_cell, mineral_mass, jnp.float64(0.0)).T,
        initial=jnp.zeros(mineral_mass.shape[0], dtype=jnp.float64),
    )
    positive_total = mineral_total > jnp.float64(0.0)
    safe_total = jnp.where(positive_total, mineral_total, jnp.float64(1.0))
    uptake_profile = jnp.where(
        positive_total[:, None],
        _divide(mineral_n, safe_total[:, None]),
        jnp.asarray(
            inputs.nitrogen_fixation_profile_column_level, dtype=jnp.float64
        ),
    )
    uptake_profile = jnp.where(active_cell, uptake_profile, jnp.float64(0.0))
    profiled_plant_demand = fortran_multiply(plant_demand, uptake_profile)

    compet_plant_nh4 = jnp.asarray(
        parameters.plant_competitiveness_ammonium, dtype=jnp.float64
    )
    compet_decomp_nh4 = jnp.asarray(
        parameters.decomposer_competitiveness_ammonium, dtype=jnp.float64
    )
    compet_nit = jnp.asarray(
        parameters.nitrifier_competitiveness, dtype=jnp.float64
    )
    ammonium_demand = _sum_three(
        profiled_plant_demand,
        potential_immobilization,
        potential_nitrification,
    )
    ammonium_demand_scaled = _sum_three(
        fortran_multiply(profiled_plant_demand, compet_plant_nh4),
        fortran_multiply(potential_immobilization, compet_decomp_nh4),
        fortran_multiply(potential_nitrification, compet_nit),
    )
    ammonium_unlimited = _binary64_less(
        fortran_multiply(ammonium_demand, dt),
        ammonium,
    )
    ammonium_has_demand = ammonium_demand > jnp.float64(0.0)
    safe_ammonium_scaled = jnp.where(
        ammonium_has_demand, ammonium_demand_scaled, jnp.float64(1.0)
    )
    ammonium_per_second = nonnegative_divide_preserving_subnormal(ammonium, dt)
    limited_immob_nh4_fraction = _divide(
        fortran_multiply(potential_immobilization, compet_decomp_nh4),
        safe_ammonium_scaled,
    )
    limited_immob_nh4 = jnp.minimum(
        fortran_multiply(ammonium_per_second, limited_immob_nh4_fraction),
        potential_immobilization,
    )
    limited_nitrification_fraction = _divide(
        fortran_multiply(potential_nitrification, compet_nit),
        safe_ammonium_scaled,
    )
    limited_nitrification = jnp.minimum(
        fortran_multiply(ammonium_per_second, limited_nitrification_fraction),
        potential_nitrification,
    )
    actual_immob_nh4 = jnp.where(
        ammonium_unlimited,
        potential_immobilization,
        jnp.where(ammonium_has_demand, limited_immob_nh4, jnp.float64(0.0)),
    )
    nitrification = jnp.where(
        ammonium_unlimited,
        potential_nitrification,
        jnp.where(
            ammonium_has_demand, limited_nitrification, jnp.float64(0.0)
        ),
    )
    plant_ammonium = fortran_add(
        fortran_add(ammonium_per_second, -actual_immob_nh4),
        -nitrification,
    )
    plant_ammonium = jnp.where(
        _binary64_is_zero(actual_immob_nh4)
        & _binary64_is_zero(nitrification),
        ammonium_per_second,
        plant_ammonium,
    )
    plant_ammonium = jnp.where(
        ammonium_unlimited | ammonium_has_demand,
        plant_ammonium,
        jnp.float64(0.0),
    )
    fpi_nh4 = jnp.where(
        ammonium_unlimited,
        jnp.float64(1.0),
        jnp.where(
            potential_immobilization > jnp.float64(0.0),
            _divide(
                actual_immob_nh4,
                jnp.where(
                    potential_immobilization > jnp.float64(0.0),
                    potential_immobilization,
                    jnp.float64(1.0),
                ),
            ),
            jnp.float64(0.0),
        ),
    )

    remaining_immobilization = fortran_add(
        potential_immobilization, -actual_immob_nh4
    )
    compet_plant_no3 = jnp.asarray(
        parameters.plant_competitiveness_nitrate, dtype=jnp.float64
    )
    compet_decomp_no3 = jnp.asarray(
        parameters.decomposer_competitiveness_nitrate, dtype=jnp.float64
    )
    compet_denit = jnp.asarray(
        parameters.denitrifier_competitiveness, dtype=jnp.float64
    )
    nitrate_demand = _sum_three(
        profiled_plant_demand,
        remaining_immobilization,
        potential_denitrification,
    )
    nitrate_demand_scaled = _sum_three(
        fortran_multiply(profiled_plant_demand, compet_plant_no3),
        fortran_multiply(remaining_immobilization, compet_decomp_no3),
        fortran_multiply(potential_denitrification, compet_denit),
    )
    nitrate_unlimited = _binary64_less(
        fortran_multiply(nitrate_demand, dt),
        nitrate,
    )
    nitrate_has_demand = nitrate_demand > jnp.float64(0.0)
    safe_nitrate_scaled = jnp.where(
        nitrate_has_demand, nitrate_demand_scaled, jnp.float64(1.0)
    )
    nitrate_per_second = nonnegative_divide_preserving_subnormal(nitrate, dt)
    limited_immob_no3_fraction = _divide(
        fortran_multiply(remaining_immobilization, compet_decomp_no3),
        safe_nitrate_scaled,
    )
    limited_immob_no3 = jnp.minimum(
        fortran_multiply(nitrate_per_second, limited_immob_no3_fraction),
        remaining_immobilization,
    )
    limited_denitrification_fraction = _divide(
        fortran_multiply(potential_denitrification, compet_denit),
        safe_nitrate_scaled,
    )
    limited_denitrification = jnp.minimum(
        fortran_multiply(nitrate_per_second, limited_denitrification_fraction),
        potential_denitrification,
    )
    actual_immob_no3 = jnp.where(
        nitrate_unlimited,
        remaining_immobilization,
        jnp.where(nitrate_has_demand, limited_immob_no3, jnp.float64(0.0)),
    )
    denitrification = jnp.where(
        nitrate_unlimited,
        potential_denitrification,
        jnp.where(
            nitrate_has_demand, limited_denitrification, jnp.float64(0.0)
        ),
    )
    plant_nitrate = fortran_add(
        fortran_add(nitrate_per_second, -actual_immob_no3),
        -denitrification,
    )
    plant_nitrate = jnp.where(
        _binary64_is_zero(actual_immob_no3)
        & _binary64_is_zero(denitrification),
        nitrate_per_second,
        plant_nitrate,
    )
    plant_nitrate = jnp.where(
        nitrate_unlimited | nitrate_has_demand,
        plant_nitrate,
        jnp.float64(0.0),
    )
    fpi_no3 = jnp.where(
        nitrate_unlimited,
        fortran_add(jnp.float64(1.0), -fpi_nh4),
        jnp.where(
            potential_immobilization > jnp.float64(0.0),
            _divide(
                actual_immob_no3,
                jnp.where(
                    potential_immobilization > jnp.float64(0.0),
                    potential_immobilization,
                    jnp.float64(1.0),
                ),
            ),
            jnp.float64(0.0),
        ),
    )

    nitrification_n2o = fortran_multiply(
        nitrification,
        jnp.asarray(parameters.nitrification_n2o_loss_fraction, dtype=jnp.float64),
    )
    denitrification_n2o = _divide(
        denitrification,
        fortran_add(jnp.float64(1.0), n2_to_n2o),
    )

    supplement = get_cn_flux_field(cn_fluxes, _SUPPLEMENT_LEVEL)
    carbon_only = jnp.asarray(parameters.carbon_only, dtype=jnp.bool_)
    immobilization_limited = fortran_add(fpi_no3, fpi_nh4) < jnp.float64(1.0)
    revised_fpi_nh4 = fortran_add(jnp.float64(1.0), -fpi_no3)
    revised_supplement = fortran_add(
        fortran_add(potential_immobilization, -actual_immob_no3),
        -actual_immob_nh4,
    )
    revised_immob_nh4 = fortran_add(
        potential_immobilization, -actual_immob_no3
    )
    revise_immobilization = carbon_only & immobilization_limited
    fpi_nh4 = jnp.where(revise_immobilization, revised_fpi_nh4, fpi_nh4)
    supplement = jnp.where(revise_immobilization, revised_supplement, supplement)
    actual_immob_nh4 = jnp.where(
        revise_immobilization, revised_immob_nh4, actual_immob_nh4
    )
    plant_limited = fortran_add(plant_nitrate, plant_ammonium) < (
        profiled_plant_demand
    )
    plant_supplement = fortran_add(
        fortran_add(profiled_plant_demand, -plant_nitrate),
        -plant_ammonium,
    )
    revise_plant = carbon_only & plant_limited
    supplement = jnp.where(
        revise_plant, fortran_add(supplement, plant_supplement), supplement
    )
    plant_ammonium = jnp.where(
        revise_plant,
        fortran_add(profiled_plant_demand, -plant_nitrate),
        plant_ammonium,
    )

    fpi_level = fortran_add(fpi_no3, fpi_nh4)
    plant_total = fortran_add(plant_nitrate, plant_ammonium)
    actual_immob_total = fortran_add(actual_immob_no3, actual_immob_nh4)

    updates = {
        _ACTUAL_IMMOBILIZATION_NH4: actual_immob_nh4,
        _ACTUAL_IMMOBILIZATION_NO3: actual_immob_no3,
        _ACTUAL_IMMOBILIZATION_LEVEL: actual_immob_total,
        _DENITRIFICATION: denitrification,
        _DENITRIFICATION_N2O: denitrification_n2o,
        _NITRIFICATION: nitrification,
        _NITRIFICATION_N2O: nitrification_n2o,
        _PLANT_NH4_LEVEL: plant_ammonium,
        _PLANT_NO3_LEVEL: plant_nitrate,
        _PLANT_NITROGEN_LEVEL: plant_total,
        _SUPPLEMENT_LEVEL: supplement,
    }
    updated_fluxes = cn_fluxes
    for field, values in updates.items():
        old = get_cn_flux_field(updated_fluxes, field)
        updated_fluxes = set_cn_flux_field(
            updated_fluxes,
            field,
            _filtered_level_update(old, values, active_cell),
        )
    updated_state = SoilBiogeochemCompetitionState(
        fraction_potential_growth_column=jnp.asarray(
            state.fraction_potential_growth_column, dtype=jnp.float64
        ),
        fraction_potential_immobilization_column=jnp.asarray(
            state.fraction_potential_immobilization_column, dtype=jnp.float64
        ),
        fraction_potential_immobilization_column_level=_filtered_level_update(
            jnp.asarray(
                state.fraction_potential_immobilization_column_level,
                dtype=jnp.float64,
            ),
            fpi_level,
            active_cell,
        ),
    )
    diagnostics = SoilBiogeochemCompetitionDiagnostics(
        updated_column_count=jnp.sum(soil_column, dtype=jnp.int32),
        ammonium_limited_cell_count=jnp.sum(
            active_cell & (~ammonium_unlimited), dtype=jnp.int32
        ),
        nitrate_limited_cell_count=jnp.sum(
            active_cell & (~nitrate_unlimited), dtype=jnp.int32
        ),
        maximum_fun_nitrate_excess=jnp.float64(0.0),
        maximum_fun_ammonium_excess=jnp.float64(0.0),
    )
    return SoilBiogeochemCompetitionPreFunResult(
        cn_fluxes=updated_fluxes,
        state=updated_state,
        nitrogen_uptake_profile_column_level=uptake_profile,
        diagnostics=diagnostics,
    )


def finalize_soil_biogeochem_competition_fun(
    inputs: SoilBiogeochemCompetitionInputs,
    state: SoilBiogeochemCompetitionState,
    cn_fluxes: CnTransientFluxState,
    fun_nitrate_uptake_column_level: jnp.ndarray,
    fun_ammonium_uptake_column_level: jnp.ndarray,
    parameters: SoilBiogeochemCompetitionParameters = (
        DEFAULT_SOIL_BIOGEOCHEM_COMPETITION_PARAMETERS
    ),
) -> SoilBiogeochemCompetitionResult:

    _validate_shapes(inputs, state, cn_fluxes, parameters)
    _validate_parameters(parameters)
    column_count = inputs.soil_column.shape[0]
    level_shape = (column_count, NUMBER_OF_DECOMPOSITION_LEVELS)
    if jnp.shape(fun_nitrate_uptake_column_level) != level_shape:
        raise ValueError("fun_nitrate_uptake_column_level must have shape (column, 25)")
    if jnp.shape(fun_ammonium_uptake_column_level) != level_shape:
        raise ValueError("fun_ammonium_uptake_column_level must have shape (column, 25)")

    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    active_level = jnp.arange(
        NUMBER_OF_DECOMPOSITION_LEVELS, dtype=jnp.int32
    ) < jnp.asarray(parameters.decomposition_level_count, dtype=jnp.int32)
    active_cell = soil_column[:, None] & active_level[None, :]
    patch_to_column_cell = jnp.broadcast_to(
        active_level[None, :],
        (column_count, NUMBER_OF_DECOMPOSITION_LEVELS),
    )
    thickness = jnp.asarray(parameters.soil_layer_thickness, dtype=jnp.float64)
    fun_nitrate = jnp.asarray(fun_nitrate_uptake_column_level, dtype=jnp.float64)
    fun_ammonium = jnp.asarray(fun_ammonium_uptake_column_level, dtype=jnp.float64)
    maximum_nitrate = get_cn_flux_field(cn_fluxes, _PLANT_NO3_LEVEL)
    maximum_ammonium = get_cn_flux_field(cn_fluxes, _PLANT_NH4_LEVEL)
    maximum_total = get_cn_flux_field(cn_fluxes, _PLANT_NITROGEN_LEVEL)
    actual_immob_level = get_cn_flux_field(
        cn_fluxes, _ACTUAL_IMMOBILIZATION_LEVEL
    )
    potential_immob_level = jnp.asarray(
        inputs.potential_immobilization_column_level, dtype=jnp.float64
    )

    maximum_plant_mass = fortran_multiply(maximum_total, thickness[None, :])
    actual_immob_mass = fortran_multiply(
        actual_immob_level, thickness[None, :]
    )
    potential_immob_mass = fortran_multiply(
        potential_immob_level, thickness[None, :]
    )

    def ordered_active_sum(values: jnp.ndarray) -> jnp.ndarray:
        return fortran_ordered_sum(
            jnp.where(active_cell, values, jnp.float64(0.0)).T,
            initial=jnp.zeros(column_count, dtype=jnp.float64),
        )

    maximum_plant_column = ordered_active_sum(maximum_plant_mass)
    actual_immob_column = ordered_active_sum(actual_immob_mass)
    potential_immob_column = ordered_active_sum(potential_immob_mass)
    positive_potential = potential_immob_column > jnp.float64(0.0)
    fpi_column = jnp.where(
        positive_potential,
        _divide(
            actual_immob_column,
            jnp.where(
                positive_potential,
                potential_immob_column,
                jnp.float64(1.0),
            ),
        ),
        jnp.float64(1.0),
    )

    updated_fluxes = cn_fluxes
    for field, values in {
        _FUN_PLANT_NO3_LEVEL: fun_nitrate,
        _FUN_PLANT_NH4_LEVEL: fun_ammonium,
    }.items():
        old = get_cn_flux_field(updated_fluxes, field)
        updated_fluxes = set_cn_flux_field(
            updated_fluxes,
            field,
            _filtered_level_update(old, values, patch_to_column_cell),
        )
    for field, values in {
        _PLANT_NITROGEN: maximum_plant_column,
        _ACTUAL_IMMOBILIZATION: actual_immob_column,
        _POTENTIAL_IMMOBILIZATION: potential_immob_column,
    }.items():
        old = get_cn_flux_field(updated_fluxes, field)
        updated_fluxes = set_cn_flux_field(
            updated_fluxes, field, jnp.where(soil_column, values, old)
        )

    nitrate_excess = fortran_add(fun_nitrate, -maximum_nitrate)
    ammonium_excess = fortran_add(fun_ammonium, -maximum_ammonium)
    masked_nitrate_excess = jnp.where(
        active_cell, nitrate_excess, jnp.float64(-jnp.inf)
    )
    masked_ammonium_excess = jnp.where(
        active_cell, ammonium_excess, jnp.float64(-jnp.inf)
    )
    return SoilBiogeochemCompetitionResult(
        cn_fluxes=updated_fluxes,
        state=SoilBiogeochemCompetitionState(
            fraction_potential_growth_column=jnp.asarray(
                state.fraction_potential_growth_column, dtype=jnp.float64
            ),
            fraction_potential_immobilization_column=jnp.where(
                soil_column,
                fpi_column,
                jnp.asarray(
                    state.fraction_potential_immobilization_column,
                    dtype=jnp.float64,
                ),
            ),
            fraction_potential_immobilization_column_level=jnp.asarray(
                state.fraction_potential_immobilization_column_level,
                dtype=jnp.float64,
            ),
        ),
        diagnostics=SoilBiogeochemCompetitionDiagnostics(
            updated_column_count=jnp.sum(soil_column, dtype=jnp.int32),
            ammonium_limited_cell_count=jnp.int32(0),
            nitrate_limited_cell_count=jnp.int32(0),
            maximum_fun_nitrate_excess=jnp.maximum(
                jnp.max(masked_nitrate_excess), jnp.float64(0.0)
            ),
            maximum_fun_ammonium_excess=jnp.maximum(
                jnp.max(masked_ammonium_excess), jnp.float64(0.0)
            ),
        ),
    )


__all__ = [
    "DEFAULT_SOIL_BIOGEOCHEM_COMPETITION_PARAMETERS",
    "NUMBER_OF_SOIL_LEVELS",
    "SoilBiogeochemCompetitionDiagnostics",
    "SoilBiogeochemCompetitionInputs",
    "SoilBiogeochemCompetitionParameters",
    "SoilBiogeochemCompetitionPreFunResult",
    "SoilBiogeochemCompetitionResult",
    "SoilBiogeochemCompetitionState",
    "finalize_soil_biogeochem_competition_fun",
    "soil_biogeochem_competition_pre_fun",
]
