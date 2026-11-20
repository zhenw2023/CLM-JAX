
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    get_cn_flux_field,
    set_cn_flux_field,
)
from clm_jax.physics.evergreen_phenology import (
    CARBON_TRANSFER_POOL_NAMES,
    CARBON_TRANSFER_TO_POOL_FLUX_FIELDS,
    NITROGEN_TRANSFER_POOL_NAMES,
    NITROGEN_TRANSFER_TO_POOL_FLUX_FIELDS,
)
from clm_jax.physics.numerics import (
    fortran_divide,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)


class OnsetGrowthInputs(NamedTuple):

    soil_patch: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    onset_flag_patch: jnp.ndarray
    onset_counter_seconds_patch: jnp.ndarray
    background_transfer_growth_rate_patch: jnp.ndarray
    carbon_transfer_patch_pool: jnp.ndarray
    nitrogen_transfer_patch_pool: jnp.ndarray


class OnsetGrowthParameters(NamedTuple):

    woody_pft: jnp.ndarray
    time_step_seconds: float | jnp.ndarray = 1800.0


_REFERENCE_WOODY_PFT = np.zeros(79, dtype=np.float64)
_REFERENCE_WOODY_PFT[1:12] = np.float64(1.0)
DEFAULT_ONSET_GROWTH_PARAMETERS = OnsetGrowthParameters(
    woody_pft=_REFERENCE_WOODY_PFT,
)


class OnsetGrowthDiagnostics(NamedTuple):
    onset_patch_count: jnp.ndarray
    background_transfer_patch_count: jnp.ndarray
    flux_update_patch_count: jnp.ndarray
    woody_flux_update_patch_count: jnp.ndarray


class OnsetGrowthResult(NamedTuple):
    cn_fluxes: CnTransientFluxState
    diagnostics: OnsetGrowthDiagnostics


def _validate_shapes(
    inputs: OnsetGrowthInputs,
    cn_fluxes: CnTransientFluxState,
    parameters: OnsetGrowthParameters,
) -> int:
    if jnp.ndim(inputs.soil_patch) != 1:
        raise ValueError("soil_patch must be a vector")
    patch_count = inputs.soil_patch.shape[0]
    for name in (
        "vegetation_type_patch",
        "onset_flag_patch",
        "onset_counter_seconds_patch",
        "background_transfer_growth_rate_patch",
    ):
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    expected_pool_shapes = {
        "carbon_transfer_patch_pool": (
            patch_count,
            len(CARBON_TRANSFER_POOL_NAMES),
        ),
        "nitrogen_transfer_patch_pool": (
            patch_count,
            len(NITROGEN_TRANSFER_POOL_NAMES),
        ),
    }
    for name, expected in expected_pool_shapes.items():
        if jnp.shape(getattr(inputs, name)) != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if jnp.ndim(parameters.woody_pft) != 1 or parameters.woody_pft.shape[0] < 1:
        raise ValueError("woody_pft must be a nonempty PFT vector")
    if jnp.shape(parameters.time_step_seconds) != ():
        raise ValueError("time_step_seconds must be scalar")
    for field_name in (
        *CARBON_TRANSFER_TO_POOL_FLUX_FIELDS,
        *NITROGEN_TRANSFER_TO_POOL_FLUX_FIELDS,
    ):
        if get_cn_flux_field(cn_fluxes, field_name).shape != (patch_count,):
            raise ValueError("onset-growth flux state must match patch count")
    return patch_count


def _replace_flux_family(
    cn_fluxes: CnTransientFluxState,
    field_names: tuple[str, ...],
    onset_candidates: jnp.ndarray,
    background_candidates: jnp.ndarray,
    onset_mask: jnp.ndarray,
    background_mask: jnp.ndarray,
    woody_patch: jnp.ndarray,
) -> CnTransientFluxState:
    for pool_index, field_name in enumerate(field_names):
        permitted = (
            jnp.ones_like(woody_patch)
            if pool_index < 2
            else woody_patch
        )
        previous = get_cn_flux_field(cn_fluxes, field_name)
        after_onset = jnp.where(
            onset_mask & permitted,
            onset_candidates[:, pool_index],
            previous,
        )
        cn_fluxes = set_cn_flux_field(
            cn_fluxes,
            field_name,
            jnp.where(
                background_mask & permitted,
                background_candidates[:, pool_index],
                after_onset,
            ),
        )
    return cn_fluxes


def onset_growth(
    inputs: OnsetGrowthInputs,
    cn_fluxes: CnTransientFluxState,
    parameters: OnsetGrowthParameters = DEFAULT_ONSET_GROWTH_PARAMETERS,
) -> OnsetGrowthResult:

    _validate_shapes(inputs, cn_fluxes, parameters)
    soil = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    vegetation_type = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    woody_by_pft = jnp.asarray(parameters.woody_pft, dtype=jnp.float64)
    woody_patch = woody_by_pft[vegetation_type] == jnp.float64(1.0)
    onset_flag = jnp.asarray(inputs.onset_flag_patch, dtype=jnp.float64)
    onset_counter = jnp.asarray(
        inputs.onset_counter_seconds_patch,
        dtype=jnp.float64,
    )
    background_rate = jnp.asarray(
        inputs.background_transfer_growth_rate_patch,
        dtype=jnp.float64,
    )
    time_step = jnp.asarray(parameters.time_step_seconds, dtype=jnp.float64)

    onset_mask = soil & (onset_flag == jnp.float64(1.0))
    background_mask = soil & (background_rate > jnp.float64(0.0))
    any_update = onset_mask | background_mask

    safe_counter = jnp.where(onset_mask, onset_counter, time_step)
    onset_rate = jnp.where(
        onset_counter == time_step,
        fortran_divide(jnp.float64(1.0), time_step),
        fortran_divide(jnp.float64(2.0), safe_counter),
    )

    carbon_transfer = jnp.asarray(
        inputs.carbon_transfer_patch_pool,
        dtype=jnp.float64,
    )
    nitrogen_transfer = jnp.asarray(
        inputs.nitrogen_transfer_patch_pool,
        dtype=jnp.float64,
    )
    carbon_onset = fortran_multiply(onset_rate[:, None], carbon_transfer)
    nitrogen_onset = fortran_multiply(onset_rate[:, None], nitrogen_transfer)
    carbon_background = fortran_divide(carbon_transfer, time_step)
    nitrogen_background = fortran_divide(nitrogen_transfer, time_step)

    updated_fluxes = _replace_flux_family(
        cn_fluxes,
        CARBON_TRANSFER_TO_POOL_FLUX_FIELDS,
        carbon_onset,
        carbon_background,
        onset_mask,
        background_mask,
        woody_patch,
    )
    updated_fluxes = _replace_flux_family(
        updated_fluxes,
        NITROGEN_TRANSFER_TO_POOL_FLUX_FIELDS,
        nitrogen_onset,
        nitrogen_background,
        onset_mask,
        background_mask,
        woody_patch,
    )

    return OnsetGrowthResult(
        cn_fluxes=updated_fluxes,
        diagnostics=OnsetGrowthDiagnostics(
            onset_patch_count=jnp.sum(onset_mask, dtype=jnp.int32),
            background_transfer_patch_count=jnp.sum(
                background_mask,
                dtype=jnp.int32,
            ),
            flux_update_patch_count=jnp.sum(any_update, dtype=jnp.int32),
            woody_flux_update_patch_count=jnp.sum(
                any_update & woody_patch,
                dtype=jnp.int32,
            ),
        ),
    )


__all__ = [
    "DEFAULT_ONSET_GROWTH_PARAMETERS",
    "OnsetGrowthDiagnostics",
    "OnsetGrowthInputs",
    "OnsetGrowthParameters",
    "OnsetGrowthResult",
    "onset_growth",
]
