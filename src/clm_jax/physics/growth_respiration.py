
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    get_cn_flux_field,
    set_cn_flux_field,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)


class GrowthRespirationInputs(NamedTuple):

    soil_patch: jnp.ndarray
    vegetation_type_patch: jnp.ndarray


class GrowthRespirationParameters(NamedTuple):

    woody_pft: jnp.ndarray
    growth_respiration_fraction_pft: jnp.ndarray
    immediate_growth_respiration_fraction_pft: jnp.ndarray
    first_prognostic_crop_pft: int | jnp.ndarray = 17


class GrowthRespirationDiagnostics(NamedTuple):
    updated_patch_count: jnp.ndarray
    crop_patch_count: jnp.ndarray
    woody_patch_count: jnp.ndarray


class GrowthRespirationResult(NamedTuple):
    cn_fluxes: CnTransientFluxState
    diagnostics: GrowthRespirationDiagnostics


_ORGAN_FIELDS = {
    "leaf": (
        "vegetation_carbon.cpool_to_leafc_patch",
        "vegetation_carbon.cpool_to_leafc_storage_patch",
        "vegetation_carbon.leafc_xfer_to_leafc_patch",
        "vegetation_carbon.cpool_leaf_gr_patch",
        "vegetation_carbon.cpool_leaf_storage_gr_patch",
        "vegetation_carbon.transfer_leaf_gr_patch",
    ),
    "froot": (
        "vegetation_carbon.cpool_to_frootc_patch",
        "vegetation_carbon.cpool_to_frootc_storage_patch",
        "vegetation_carbon.frootc_xfer_to_frootc_patch",
        "vegetation_carbon.cpool_froot_gr_patch",
        "vegetation_carbon.cpool_froot_storage_gr_patch",
        "vegetation_carbon.transfer_froot_gr_patch",
    ),
    "livestem": (
        "vegetation_carbon.cpool_to_livestemc_patch",
        "vegetation_carbon.cpool_to_livestemc_storage_patch",
        "vegetation_carbon.livestemc_xfer_to_livestemc_patch",
        "vegetation_carbon.cpool_livestem_gr_patch",
        "vegetation_carbon.cpool_livestem_storage_gr_patch",
        "vegetation_carbon.transfer_livestem_gr_patch",
    ),
    "deadstem": (
        "vegetation_carbon.cpool_to_deadstemc_patch",
        "vegetation_carbon.cpool_to_deadstemc_storage_patch",
        "vegetation_carbon.deadstemc_xfer_to_deadstemc_patch",
        "vegetation_carbon.cpool_deadstem_gr_patch",
        "vegetation_carbon.cpool_deadstem_storage_gr_patch",
        "vegetation_carbon.transfer_deadstem_gr_patch",
    ),
    "livecroot": (
        "vegetation_carbon.cpool_to_livecrootc_patch",
        "vegetation_carbon.cpool_to_livecrootc_storage_patch",
        "vegetation_carbon.livecrootc_xfer_to_livecrootc_patch",
        "vegetation_carbon.cpool_livecroot_gr_patch",
        "vegetation_carbon.cpool_livecroot_storage_gr_patch",
        "vegetation_carbon.transfer_livecroot_gr_patch",
    ),
    "deadcroot": (
        "vegetation_carbon.cpool_to_deadcrootc_patch",
        "vegetation_carbon.cpool_to_deadcrootc_storage_patch",
        "vegetation_carbon.deadcrootc_xfer_to_deadcrootc_patch",
        "vegetation_carbon.cpool_deadcroot_gr_patch",
        "vegetation_carbon.cpool_deadcroot_storage_gr_patch",
        "vegetation_carbon.transfer_deadcroot_gr_patch",
    ),
    "grain": (
        "vegetation_carbon.cpool_to_grainc_patch",
        "vegetation_carbon.cpool_to_grainc_storage_patch",
        "vegetation_carbon.grainc_xfer_to_grainc_patch",
        "vegetation_carbon.cpool_grain_gr_patch",
        "vegetation_carbon.cpool_grain_storage_gr_patch",
        "vegetation_carbon.transfer_grain_gr_patch",
    ),
}


def _validate_shapes(
    inputs: GrowthRespirationInputs,
    cn_fluxes: CnTransientFluxState,
    parameters: GrowthRespirationParameters,
) -> int:
    if jnp.ndim(inputs.soil_patch) != 1:
        raise ValueError("soil_patch must be a vector")
    patch_count = inputs.soil_patch.shape[0]
    if jnp.shape(inputs.vegetation_type_patch) != (patch_count,):
        raise ValueError("vegetation_type_patch must contain every patch")
    pft_count = None
    for name in parameters._fields[:3]:
        values = getattr(parameters, name)
        if jnp.ndim(values) != 1 or values.shape[0] < 1:
            raise ValueError(f"{name} must be a nonempty PFT vector")
        if pft_count is None:
            pft_count = values.shape[0]
        elif values.shape[0] != pft_count:
            raise ValueError("all growth-respiration PFT vectors must have equal length")
    if jnp.shape(parameters.first_prognostic_crop_pft) != ():
        raise ValueError("first_prognostic_crop_pft must be scalar")
    for fields in _ORGAN_FIELDS.values():
        for field_name in fields:
            if get_cn_flux_field(cn_fluxes, field_name).shape != (patch_count,):
                raise ValueError("growth-respiration fluxes must match patch count")
    return patch_count


def _organ_growth_respiration(
    cn_fluxes: CnTransientFluxState,
    fields: tuple[str, str, str, str, str, str],
    growth_fraction: jnp.ndarray,
    immediate_fraction: jnp.ndarray,
    update: jnp.ndarray,
) -> CnTransientFluxState:
    direct_source = get_cn_flux_field(cn_fluxes, fields[0])
    storage_source = get_cn_flux_field(cn_fluxes, fields[1])
    transfer_source = get_cn_flux_field(cn_fluxes, fields[2])
    direct = fortran_multiply(direct_source, growth_fraction)
    storage = fortran_multiply(storage_source, growth_fraction)
    storage = fortran_multiply(storage, immediate_fraction)
    delayed_fraction = fortran_add(jnp.float64(1.0), -immediate_fraction)
    transfer = fortran_multiply(transfer_source, growth_fraction)
    transfer = fortran_multiply(transfer, delayed_fraction)
    for field_name, candidate in zip(
        fields[3:],
        (direct, storage, transfer),
        strict=True,
    ):
        previous = get_cn_flux_field(cn_fluxes, field_name)
        cn_fluxes = set_cn_flux_field(
            cn_fluxes,
            field_name,
            jnp.where(update, candidate, previous),
        )
    return cn_fluxes


def growth_respiration(
    inputs: GrowthRespirationInputs,
    cn_fluxes: CnTransientFluxState,
    parameters: GrowthRespirationParameters,
) -> GrowthRespirationResult:

    _validate_shapes(inputs, cn_fluxes, parameters)
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    vegetation_type = jnp.asarray(
        inputs.vegetation_type_patch,
        dtype=jnp.int32,
    )
    woody_pft = jnp.asarray(parameters.woody_pft, dtype=jnp.bool_)
    growth_fraction_pft = jnp.asarray(
        parameters.growth_respiration_fraction_pft,
        dtype=jnp.float64,
    )
    immediate_fraction_pft = jnp.asarray(
        parameters.immediate_growth_respiration_fraction_pft,
        dtype=jnp.float64,
    )
    woody_patch = soil_patch & woody_pft[vegetation_type]
    crop_patch = soil_patch & (
        vegetation_type
        >= jnp.asarray(parameters.first_prognostic_crop_pft, dtype=jnp.int32)
    )
    growth_fraction = growth_fraction_pft[vegetation_type]
    immediate_fraction = immediate_fraction_pft[vegetation_type]

    cn_fluxes = _organ_growth_respiration(
        cn_fluxes,
        _ORGAN_FIELDS["livestem"],
        growth_fraction,
        immediate_fraction,
        crop_patch,
    )
    cn_fluxes = _organ_growth_respiration(
        cn_fluxes,
        _ORGAN_FIELDS["grain"],
        growth_fraction,
        immediate_fraction,
        crop_patch,
    )
    for organ in ("leaf", "froot"):
        cn_fluxes = _organ_growth_respiration(
            cn_fluxes,
            _ORGAN_FIELDS[organ],
            growth_fraction,
            immediate_fraction,
            soil_patch,
        )
    for organ in ("livestem", "deadstem", "livecroot", "deadcroot"):
        cn_fluxes = _organ_growth_respiration(
            cn_fluxes,
            _ORGAN_FIELDS[organ],
            growth_fraction,
            immediate_fraction,
            woody_patch,
        )

    return GrowthRespirationResult(
        cn_fluxes=cn_fluxes,
        diagnostics=GrowthRespirationDiagnostics(
            updated_patch_count=jnp.sum(soil_patch, dtype=jnp.int32),
            crop_patch_count=jnp.sum(crop_patch, dtype=jnp.int32),
            woody_patch_count=jnp.sum(woody_patch, dtype=jnp.int32),
        ),
    )


__all__ = [
    "GrowthRespirationDiagnostics",
    "GrowthRespirationInputs",
    "GrowthRespirationParameters",
    "GrowthRespirationResult",
    "growth_respiration",
]
