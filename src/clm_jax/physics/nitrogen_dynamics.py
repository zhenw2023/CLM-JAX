
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp

from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    get_cn_flux_field,
    set_cn_flux_field,
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
from clm_jax.physics.subgrid import patch_to_column_average

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase

_NITROGEN_DEPOSITION_FIELD = "soil_nitrogen.ndep_to_sminn_col"
_FREE_LIVING_FIXATION_FIELD = "soil_nitrogen.ffix_to_sminn_col"
_FERTILIZER_FIELD = "soil_nitrogen.fert_to_sminn_col"


class NitrogenDepositionInputs(NamedTuple):

    forcing_ndep_gridcell: jnp.ndarray
    column_gridcell_index: jnp.ndarray


class NitrogenDepositionDiagnostics(NamedTuple):
    updated_column_count: jnp.ndarray


class NitrogenDepositionResult(NamedTuple):
    state: CnTransientFluxState
    diagnostics: NitrogenDepositionDiagnostics


class FreeLivingFixationParameters(NamedTuple):

    slope_with_annual_evapotranspiration: float = 0.0006
    intercept: float = 0.0117

    @classmethod
    def from_case(cls, case: CimeCase) -> FreeLivingFixationParameters:

        from clm_jax.io.namelist import read_case_namelist_group  

        group = read_case_namelist_group(
            case,
            "lnd_in",
            "mineral_nitrogen_dynamics",
        )
        return cls(
            slope_with_annual_evapotranspiration=float(group.get("freelivfix_slope_wet", 0.0006)),
            intercept=float(group.get("freelivfix_intercept", 0.0117)),
        )


DEFAULT_FREE_LIVING_FIXATION_PARAMETERS = FreeLivingFixationParameters()


class FreeLivingFixationInputs(NamedTuple):

    annual_average_evapotranspiration_column: jnp.ndarray
    soil_column: jnp.ndarray
    days_per_year: float | jnp.ndarray


class FreeLivingFixationDiagnostics(NamedTuple):
    seconds_per_year: jnp.ndarray
    updated_column_count: jnp.ndarray


class FreeLivingFixationResult(NamedTuple):
    state: CnTransientFluxState
    diagnostics: FreeLivingFixationDiagnostics


class NitrogenFertilizerParameters(NamedTuple):

    maximum_patches_per_column: int = 79

    @classmethod
    def from_case(cls, case: CimeCase) -> NitrogenFertilizerParameters:

        from clm_jax.io.namelist import read_case_namelist_group  

        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        maximum = int(general["maxpatch_pft"])
        if maximum < 0:
            raise ValueError("maxpatch_pft must be nonnegative")
        return cls(maximum_patches_per_column=maximum)


DEFAULT_NITROGEN_FERTILIZER_PARAMETERS = NitrogenFertilizerParameters()


class NitrogenFertilizerInputs(NamedTuple):

    fertilizer_patch: jnp.ndarray
    patch_column_index: jnp.ndarray
    patch_weight_on_column: jnp.ndarray
    patch_active: jnp.ndarray
    soil_column: jnp.ndarray


class NitrogenFertilizerDiagnostics(NamedTuple):
    updated_column_count: jnp.ndarray


class NitrogenFertilizerResult(NamedTuple):
    state: CnTransientFluxState
    diagnostics: NitrogenFertilizerDiagnostics


def nitrogen_deposition(
    inputs: NitrogenDepositionInputs,
    state: CnTransientFluxState,
) -> NitrogenDepositionResult:

    forcing = jnp.asarray(inputs.forcing_ndep_gridcell, dtype=jnp.float64)
    column_gridcell = jnp.asarray(inputs.column_gridcell_index, dtype=jnp.int32)
    if forcing.ndim != 1:
        raise ValueError("forcing_ndep_gridcell must be a vector")
    if column_gridcell.ndim != 1:
        raise ValueError("column_gridcell_index must be a vector")
    previous = get_cn_flux_field(state, _NITROGEN_DEPOSITION_FIELD)
    if column_gridcell.shape != previous.shape:
        raise ValueError("column_gridcell_index must contain one value per column")

    deposition = forcing[column_gridcell]
    return NitrogenDepositionResult(
        state=set_cn_flux_field(state, _NITROGEN_DEPOSITION_FIELD, deposition),
        diagnostics=NitrogenDepositionDiagnostics(
            updated_column_count=jnp.asarray(column_gridcell.shape[0], dtype=jnp.int32)
        ),
    )


def free_living_fixation(
    inputs: FreeLivingFixationInputs,
    state: CnTransientFluxState,
    parameters: FreeLivingFixationParameters = DEFAULT_FREE_LIVING_FIXATION_PARAMETERS,
) -> FreeLivingFixationResult:

    annual_et = jnp.asarray(
        inputs.annual_average_evapotranspiration_column,
        dtype=jnp.float64,
    )
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    days_per_year = jnp.asarray(inputs.days_per_year, dtype=jnp.float64)
    if annual_et.ndim != 1:
        raise ValueError("annual_average_evapotranspiration_column must be a vector")
    if soil_column.shape != annual_et.shape:
        raise ValueError("soil_column must match annual_average_evapotranspiration_column")
    if days_per_year.shape != ():
        raise ValueError("days_per_year must be a scalar")

    previous = get_cn_flux_field(state, _FREE_LIVING_FIXATION_FIELD)
    if previous.shape != annual_et.shape:
        raise ValueError("annual_average_evapotranspiration_column must match column state")

    seconds_per_year = fortran_multiply(
        fortran_multiply(days_per_year, jnp.float64(24.0)),
        jnp.float64(3600.0),
    )
    nonnegative_annual_et = jnp.maximum(jnp.float64(0.0), annual_et)
    annual_et_total = fortran_multiply(nonnegative_annual_et, seconds_per_year)
    slope_term = fortran_multiply(
        jnp.asarray(
            parameters.slope_with_annual_evapotranspiration,
            dtype=jnp.float64,
        ),
        annual_et_total,
    )
    annual_fixation = fortran_add(
        slope_term,
        jnp.asarray(parameters.intercept, dtype=jnp.float64),
    )
    fixation = fortran_divide(annual_fixation, seconds_per_year)
    updated = jnp.where(soil_column, fixation, previous)
    return FreeLivingFixationResult(
        state=set_cn_flux_field(state, _FREE_LIVING_FIXATION_FIELD, updated),
        diagnostics=FreeLivingFixationDiagnostics(
            seconds_per_year=seconds_per_year,
            updated_column_count=jnp.sum(soil_column, dtype=jnp.int32),
        ),
    )


def nitrogen_fertilizer(
    inputs: NitrogenFertilizerInputs,
    state: CnTransientFluxState,
    parameters: NitrogenFertilizerParameters = DEFAULT_NITROGEN_FERTILIZER_PARAMETERS,
) -> NitrogenFertilizerResult:

    fertilizer = jnp.asarray(inputs.fertilizer_patch, dtype=jnp.float64)
    patch_column = jnp.asarray(inputs.patch_column_index, dtype=jnp.int32)
    patch_weight = jnp.asarray(inputs.patch_weight_on_column, dtype=jnp.float64)
    patch_active = jnp.asarray(inputs.patch_active, dtype=jnp.bool_)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    if not (
        fertilizer.ndim == 1
        and fertilizer.shape == patch_column.shape == patch_weight.shape == patch_active.shape
    ):
        raise ValueError("fertilizer and patch topology must be identically sized vectors")
    if soil_column.ndim != 1:
        raise ValueError("soil_column must be a vector")
    previous = get_cn_flux_field(state, _FERTILIZER_FIELD)
    if previous.shape != soil_column.shape:
        raise ValueError("soil_column must match column state")
    averaged = patch_to_column_average(
        fertilizer,
        patch_column,
        patch_weight,
        patch_active,
        soil_column,
        maximum_patches_per_column=parameters.maximum_patches_per_column,
    )
    updated = jnp.where(soil_column, averaged, previous)
    return NitrogenFertilizerResult(
        state=set_cn_flux_field(state, _FERTILIZER_FIELD, updated),
        diagnostics=NitrogenFertilizerDiagnostics(
            updated_column_count=jnp.sum(soil_column, dtype=jnp.int32)
        ),
    )
