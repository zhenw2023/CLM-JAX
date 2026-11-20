
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
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)

CARBON_STORAGE_POOL_NAMES = (
    "leafc_storage",
    "frootc_storage",
    "livestemc_storage",
    "deadstemc_storage",
    "livecrootc_storage",
    "deadcrootc_storage",
    "gresp_storage",
)
CARBON_TRANSFER_POOL_NAMES = (
    "leafc_xfer",
    "frootc_xfer",
    "livestemc_xfer",
    "deadstemc_xfer",
    "livecrootc_xfer",
    "deadcrootc_xfer",
)
NITROGEN_STORAGE_POOL_NAMES = (
    "leafn_storage",
    "frootn_storage",
    "livestemn_storage",
    "deadstemn_storage",
    "livecrootn_storage",
    "deadcrootn_storage",
)
NITROGEN_TRANSFER_POOL_NAMES = (
    "leafn_xfer",
    "frootn_xfer",
    "livestemn_xfer",
    "deadstemn_xfer",
    "livecrootn_xfer",
    "deadcrootn_xfer",
)

CARBON_STORAGE_TO_TRANSFER_FLUX_FIELDS = tuple(
    f"vegetation_carbon.{name}_to_xfer_patch"
    for name in CARBON_STORAGE_POOL_NAMES
)
CARBON_TRANSFER_TO_POOL_FLUX_FIELDS = tuple(
    f"vegetation_carbon.{name}_to_{name.removesuffix('_xfer')}_patch"
    for name in CARBON_TRANSFER_POOL_NAMES
)
NITROGEN_STORAGE_TO_TRANSFER_FLUX_FIELDS = tuple(
    f"vegetation_nitrogen.{name}_to_xfer_patch"
    for name in NITROGEN_STORAGE_POOL_NAMES
)
NITROGEN_TRANSFER_TO_POOL_FLUX_FIELDS = tuple(
    f"vegetation_nitrogen.{name}_to_{name.removesuffix('_xfer')}_patch"
    for name in NITROGEN_TRANSFER_POOL_NAMES
)


class EvergreenPhenologyInputs(NamedTuple):

    soil_patch: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    carbon_storage_patch_pool: jnp.ndarray
    carbon_transfer_patch_pool: jnp.ndarray
    nitrogen_storage_patch_pool: jnp.ndarray
    nitrogen_transfer_patch_pool: jnp.ndarray


class EvergreenPhenologyState(NamedTuple):

    background_litterfall_rate_patch: jnp.ndarray
    background_transfer_growth_rate_patch: jnp.ndarray
    long_growing_season_factor_patch: jnp.ndarray


class EvergreenPhenologyParameters(NamedTuple):

    evergreen_pft: jnp.ndarray
    leaf_longevity_years_pft: jnp.ndarray
    woody_pft: jnp.ndarray
    time_step_seconds: float | jnp.ndarray = 1800.0
    days_per_year: float | jnp.ndarray = 365.0
    seconds_per_day: float | jnp.ndarray = 86_400.0
    evergreen_phenology_option: int | jnp.ndarray = 1
    storage_transfer_fraction: float | jnp.ndarray = 0.0002


_REFERENCE_EVERGREEN_PFT = np.zeros(79, dtype=np.float64)
_REFERENCE_EVERGREEN_PFT[[1, 2, 4, 5, 9]] = np.float64(1.0)
_REFERENCE_LEAF_LONGEVITY = np.ones(79, dtype=np.float64)
_REFERENCE_LEAF_LONGEVITY[1:3] = np.float64(3.3091666666666666)
_REFERENCE_LEAF_LONGEVITY[4:6] = np.float64(1.4024999999999999)
_REFERENCE_LEAF_LONGEVITY[9] = np.float64(1.3233333333333335)
_REFERENCE_WOODY_PFT = np.zeros(79, dtype=np.float64)
_REFERENCE_WOODY_PFT[1:12] = np.float64(1.0)
DEFAULT_EVERGREEN_PHENOLOGY_PARAMETERS = EvergreenPhenologyParameters(
    evergreen_pft=_REFERENCE_EVERGREEN_PFT,
    leaf_longevity_years_pft=_REFERENCE_LEAF_LONGEVITY,
    woody_pft=_REFERENCE_WOODY_PFT,
)


class EvergreenPhenologyDiagnostics(NamedTuple):
    evergreen_patch_count: jnp.ndarray
    woody_evergreen_patch_count: jnp.ndarray
    flux_update_patch_count: jnp.ndarray


class EvergreenPhenologyResult(NamedTuple):
    state: EvergreenPhenologyState
    cn_fluxes: CnTransientFluxState
    diagnostics: EvergreenPhenologyDiagnostics


def _validate_shapes(
    inputs: EvergreenPhenologyInputs,
    state: EvergreenPhenologyState,
    cn_fluxes: CnTransientFluxState,
    parameters: EvergreenPhenologyParameters,
) -> int:
    if jnp.ndim(inputs.soil_patch) != 1:
        raise ValueError("soil_patch must be a vector")
    patch_count = inputs.soil_patch.shape[0]
    if jnp.shape(inputs.vegetation_type_patch) != (patch_count,):
        raise ValueError("vegetation_type_patch must contain every patch")
    expected_pool_shapes = {
        "carbon_storage_patch_pool": (
            patch_count,
            len(CARBON_STORAGE_POOL_NAMES),
        ),
        "carbon_transfer_patch_pool": (
            patch_count,
            len(CARBON_TRANSFER_POOL_NAMES),
        ),
        "nitrogen_storage_patch_pool": (
            patch_count,
            len(NITROGEN_STORAGE_POOL_NAMES),
        ),
        "nitrogen_transfer_patch_pool": (
            patch_count,
            len(NITROGEN_TRANSFER_POOL_NAMES),
        ),
    }
    for name, expected in expected_pool_shapes.items():
        if jnp.shape(getattr(inputs, name)) != expected:
            raise ValueError(f"{name} must have shape {expected}")
    for name in EvergreenPhenologyState._fields:
        if jnp.shape(getattr(state, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    for name in (
        "evergreen_pft",
        "leaf_longevity_years_pft",
        "woody_pft",
    ):
        values = getattr(parameters, name)
        if jnp.ndim(values) != 1 or values.shape[0] < 1:
            raise ValueError(f"{name} must be a nonempty PFT vector")
    pft_count = parameters.evergreen_pft.shape[0]
    if parameters.leaf_longevity_years_pft.shape != (pft_count,):
        raise ValueError("leaf_longevity_years_pft must match evergreen_pft")
    if parameters.woody_pft.shape != (pft_count,):
        raise ValueError("woody_pft must match evergreen_pft")
    for field_name in (
        *CARBON_STORAGE_TO_TRANSFER_FLUX_FIELDS,
        *CARBON_TRANSFER_TO_POOL_FLUX_FIELDS,
        *NITROGEN_STORAGE_TO_TRANSFER_FLUX_FIELDS,
        *NITROGEN_TRANSFER_TO_POOL_FLUX_FIELDS,
    ):
        if get_cn_flux_field(cn_fluxes, field_name).shape != (patch_count,):
            raise ValueError("evergreen phenology flux state must match patch count")
    for name in (
        "time_step_seconds",
        "days_per_year",
        "seconds_per_day",
        "evergreen_phenology_option",
        "storage_transfer_fraction",
    ):
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    return patch_count


def _replace_flux_family(
    cn_fluxes: CnTransientFluxState,
    field_names: tuple[str, ...],
    candidates: jnp.ndarray,
    update_mask: jnp.ndarray,
) -> CnTransientFluxState:
    for index, field_name in enumerate(field_names):
        previous = get_cn_flux_field(cn_fluxes, field_name)
        cn_fluxes = set_cn_flux_field(
            cn_fluxes,
            field_name,
            jnp.where(update_mask[:, index], candidates[:, index], previous),
        )
    return cn_fluxes


def evergreen_phenology(
    inputs: EvergreenPhenologyInputs,
    state: EvergreenPhenologyState,
    cn_fluxes: CnTransientFluxState,
    parameters: EvergreenPhenologyParameters = DEFAULT_EVERGREEN_PHENOLOGY_PARAMETERS,
) -> EvergreenPhenologyResult:

    _validate_shapes(inputs, state, cn_fluxes, parameters)
    soil = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    vegetation_type = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    evergreen_by_pft = jnp.asarray(parameters.evergreen_pft, dtype=jnp.float64)
    leaf_longevity_by_pft = jnp.asarray(
        parameters.leaf_longevity_years_pft,
        dtype=jnp.float64,
    )
    woody_by_pft = jnp.asarray(parameters.woody_pft, dtype=jnp.float64)
    evergreen = soil & (evergreen_by_pft[vegetation_type] == jnp.float64(1.0))
    woody_evergreen = evergreen & (
        woody_by_pft[vegetation_type] == jnp.float64(1.0)
    )

    days_per_year = jnp.asarray(parameters.days_per_year, dtype=jnp.float64)
    seconds_per_day = jnp.asarray(parameters.seconds_per_day, dtype=jnp.float64)
    time_step = jnp.asarray(parameters.time_step_seconds, dtype=jnp.float64)
    transfer_fraction = jnp.asarray(
        parameters.storage_transfer_fraction,
        dtype=jnp.float64,
    )
    option = jnp.asarray(parameters.evergreen_phenology_option, dtype=jnp.int32)

    selected_longevity = jnp.where(
        evergreen,
        leaf_longevity_by_pft[vegetation_type],
        jnp.float64(1.0),
    )
    longevity_days = fortran_multiply(
        selected_longevity,
        days_per_year,
    )
    longevity_seconds = fortran_multiply(longevity_days, seconds_per_day)
    litterfall_rate = fortran_divide(jnp.float64(1.0), longevity_seconds)
    updated_state = EvergreenPhenologyState(
        background_litterfall_rate_patch=jnp.where(
            evergreen,
            litterfall_rate,
            jnp.asarray(state.background_litterfall_rate_patch, dtype=jnp.float64),
        ),
        background_transfer_growth_rate_patch=jnp.where(
            evergreen,
            jnp.float64(0.0),
            jnp.asarray(
                state.background_transfer_growth_rate_patch,
                dtype=jnp.float64,
            ),
        ),
        long_growing_season_factor_patch=jnp.where(
            evergreen,
            jnp.float64(0.0),
            jnp.asarray(state.long_growing_season_factor_patch, dtype=jnp.float64),
        ),
    )

    flux_enabled = option == jnp.int32(1)
    ordinary_mask = evergreen & flux_enabled
    woody_mask = woody_evergreen & flux_enabled
    carbon_storage_mask = jnp.stack(
        (ordinary_mask, ordinary_mask, *(woody_mask for _ in range(5))),
        axis=1,
    )
    transfer_mask = jnp.stack(
        (ordinary_mask, ordinary_mask, *(woody_mask for _ in range(4))),
        axis=1,
    )
    nitrogen_storage_mask = transfer_mask

    carbon_storage_source = jnp.where(
        carbon_storage_mask,
        jnp.asarray(inputs.carbon_storage_patch_pool, dtype=jnp.float64),
        jnp.float64(0.0),
    )
    nitrogen_storage_source = jnp.where(
        nitrogen_storage_mask,
        jnp.asarray(inputs.nitrogen_storage_patch_pool, dtype=jnp.float64),
        jnp.float64(0.0),
    )
    carbon_transfer_source = jnp.where(
        transfer_mask,
        jnp.asarray(inputs.carbon_transfer_patch_pool, dtype=jnp.float64),
        jnp.float64(0.0),
    )
    nitrogen_transfer_source = jnp.where(
        transfer_mask,
        jnp.asarray(inputs.nitrogen_transfer_patch_pool, dtype=jnp.float64),
        jnp.float64(0.0),
    )
    carbon_storage_rate = fortran_divide(
        fortran_multiply(
            transfer_fraction,
            carbon_storage_source,
        ),
        time_step,
    )
    nitrogen_storage_rate = fortran_divide(
        fortran_multiply(
            transfer_fraction,
            nitrogen_storage_source,
        ),
        time_step,
    )
    inverse_time_step = fortran_divide(jnp.float64(1.0), time_step)
    carbon_transfer_rate = fortran_multiply(
        inverse_time_step,
        carbon_transfer_source,
    )
    nitrogen_transfer_rate = fortran_multiply(
        inverse_time_step,
        nitrogen_transfer_source,
    )

    updated_fluxes = _replace_flux_family(
        cn_fluxes,
        CARBON_STORAGE_TO_TRANSFER_FLUX_FIELDS,
        carbon_storage_rate,
        carbon_storage_mask,
    )
    updated_fluxes = _replace_flux_family(
        updated_fluxes,
        CARBON_TRANSFER_TO_POOL_FLUX_FIELDS,
        carbon_transfer_rate,
        transfer_mask,
    )
    updated_fluxes = _replace_flux_family(
        updated_fluxes,
        NITROGEN_STORAGE_TO_TRANSFER_FLUX_FIELDS,
        nitrogen_storage_rate,
        nitrogen_storage_mask,
    )
    updated_fluxes = _replace_flux_family(
        updated_fluxes,
        NITROGEN_TRANSFER_TO_POOL_FLUX_FIELDS,
        nitrogen_transfer_rate,
        transfer_mask,
    )
    return EvergreenPhenologyResult(
        state=updated_state,
        cn_fluxes=updated_fluxes,
        diagnostics=EvergreenPhenologyDiagnostics(
            evergreen_patch_count=jnp.sum(evergreen, dtype=jnp.int32),
            woody_evergreen_patch_count=jnp.sum(
                woody_evergreen,
                dtype=jnp.int32,
            ),
            flux_update_patch_count=jnp.sum(ordinary_mask, dtype=jnp.int32),
        ),
    )


__all__ = [
    "CARBON_STORAGE_POOL_NAMES",
    "CARBON_STORAGE_TO_TRANSFER_FLUX_FIELDS",
    "CARBON_TRANSFER_POOL_NAMES",
    "CARBON_TRANSFER_TO_POOL_FLUX_FIELDS",
    "DEFAULT_EVERGREEN_PHENOLOGY_PARAMETERS",
    "EvergreenPhenologyDiagnostics",
    "EvergreenPhenologyInputs",
    "EvergreenPhenologyParameters",
    "EvergreenPhenologyResult",
    "EvergreenPhenologyState",
    "NITROGEN_STORAGE_POOL_NAMES",
    "NITROGEN_STORAGE_TO_TRANSFER_FLUX_FIELDS",
    "NITROGEN_TRANSFER_POOL_NAMES",
    "NITROGEN_TRANSFER_TO_POOL_FLUX_FIELDS",
    "evergreen_phenology",
]
