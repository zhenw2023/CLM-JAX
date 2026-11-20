
from __future__ import annotations

from typing import NamedTuple

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


class LivewoodTurnoverInputs(NamedTuple):

    soil_patch: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    live_stem_carbon_patch: jnp.ndarray
    live_coarse_root_carbon_patch: jnp.ndarray
    live_stem_nitrogen_patch: jnp.ndarray
    live_coarse_root_nitrogen_patch: jnp.ndarray


class LivewoodTurnoverParameters(NamedTuple):

    woody_pft: jnp.ndarray
    live_wood_carbon_nitrogen_ratio_pft: jnp.ndarray
    dead_wood_carbon_nitrogen_ratio_pft: jnp.ndarray
    live_wood_turnover_rate: float | jnp.ndarray
    use_fun: bool | jnp.ndarray = True
    carbon_nitrogen_ratio_floating: bool | jnp.ndarray = True


class LivewoodTurnoverDiagnostics(NamedTuple):
    woody_patch_count: jnp.ndarray
    zero_live_stem_carbon_patch_count: jnp.ndarray
    zero_live_coarse_root_carbon_patch_count: jnp.ndarray


class LivewoodTurnoverResult(NamedTuple):
    cn_fluxes: CnTransientFluxState
    diagnostics: LivewoodTurnoverDiagnostics


_FLUX_FIELDS = {
    "live_stem_carbon_to_dead_stem": (
        "vegetation_carbon.livestemc_to_deadstemc_patch"
    ),
    "live_coarse_root_carbon_to_dead_coarse_root": (
        "vegetation_carbon.livecrootc_to_deadcrootc_patch"
    ),
    "live_stem_nitrogen_to_dead_stem": (
        "vegetation_nitrogen.livestemn_to_deadstemn_patch"
    ),
    "live_stem_nitrogen_to_retranslocation": (
        "vegetation_nitrogen.livestemn_to_retransn_patch"
    ),
    "live_coarse_root_nitrogen_to_dead_coarse_root": (
        "vegetation_nitrogen.livecrootn_to_deadcrootn_patch"
    ),
    "live_coarse_root_nitrogen_to_retranslocation": (
        "vegetation_nitrogen.livecrootn_to_retransn_patch"
    ),
}


def _validate_shapes(
    inputs: LivewoodTurnoverInputs,
    cn_fluxes: CnTransientFluxState,
    parameters: LivewoodTurnoverParameters,
) -> int:
    if jnp.ndim(inputs.soil_patch) != 1:
        raise ValueError("soil_patch must be a vector")
    patch_count = inputs.soil_patch.shape[0]
    for name in inputs._fields[1:]:
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    for field_name in _FLUX_FIELDS.values():
        if get_cn_flux_field(cn_fluxes, field_name).shape != (patch_count,):
            raise ValueError("livewood-turnover flux state must match patch count")
    for name in (
        "woody_pft",
        "live_wood_carbon_nitrogen_ratio_pft",
        "dead_wood_carbon_nitrogen_ratio_pft",
    ):
        values = getattr(parameters, name)
        if jnp.ndim(values) != 1 or values.shape[0] < 1:
            raise ValueError(f"{name} must be a nonempty PFT vector")
    for name in (
        "live_wood_turnover_rate",
        "use_fun",
        "carbon_nitrogen_ratio_floating",
    ):
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    return patch_count


def _update_flux(
    cn_fluxes: CnTransientFluxState,
    name: str,
    candidate: jnp.ndarray,
    mask: jnp.ndarray,
) -> CnTransientFluxState:
    field = _FLUX_FIELDS[name]
    return set_cn_flux_field(
        cn_fluxes,
        field,
        jnp.where(mask, candidate, get_cn_flux_field(cn_fluxes, field)),
    )


def _organ_turnover(
    carbon: jnp.ndarray,
    nitrogen: jnp.ndarray,
    live_ratio: jnp.ndarray,
    dead_ratio: jnp.ndarray,
    turnover_rate: jnp.ndarray,
    floating: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    carbon_turnover = fortran_multiply(carbon, turnover_rate)
    fixed_nitrogen_turnover = fortran_divide(carbon_turnover, live_ratio)
    fixed_dead_nitrogen = fortran_divide(carbon_turnover, dead_ratio)
    safe_carbon = jnp.where(
        carbon == jnp.float64(0.0),
        jnp.float64(1.0),
        carbon,
    )
    floating_nitrogen_turnover = jnp.where(
        carbon == jnp.float64(0.0),
        jnp.float64(0.0),
        fortran_multiply(
            carbon_turnover,
            fortran_divide(nitrogen, safe_carbon),
        ),
    )
    nitrogen_turnover = jnp.where(
        floating,
        floating_nitrogen_turnover,
        fixed_nitrogen_turnover,
    )
    dead_nitrogen = jnp.where(
        floating,
        fortran_multiply(jnp.float64(0.5), nitrogen_turnover),
        fixed_dead_nitrogen,
    )
    retranslocated_nitrogen = fortran_add(
        nitrogen_turnover,
        -dead_nitrogen,
    )
    return carbon_turnover, dead_nitrogen, retranslocated_nitrogen


def livewood_turnover(
    inputs: LivewoodTurnoverInputs,
    cn_fluxes: CnTransientFluxState,
    parameters: LivewoodTurnoverParameters,
) -> LivewoodTurnoverResult:

    _validate_shapes(inputs, cn_fluxes, parameters)
    soil = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    vegetation_type = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    woody = soil & (
        jnp.asarray(parameters.woody_pft, dtype=jnp.float64)[vegetation_type]
        > jnp.float64(0.0)
    )
    live_ratio = jnp.asarray(
        parameters.live_wood_carbon_nitrogen_ratio_pft,
        dtype=jnp.float64,
    )[vegetation_type]
    dead_ratio = jnp.asarray(
        parameters.dead_wood_carbon_nitrogen_ratio_pft,
        dtype=jnp.float64,
    )[vegetation_type]
    turnover_rate = jnp.asarray(
        parameters.live_wood_turnover_rate,
        dtype=jnp.float64,
    )
    floating = jnp.asarray(
        parameters.carbon_nitrogen_ratio_floating,
        dtype=jnp.bool_,
    )
    use_fun = jnp.asarray(parameters.use_fun, dtype=jnp.bool_)
    fixed_woody = woody & ~floating
    live_ratio = jnp.where(fixed_woody, live_ratio, jnp.float64(1.0))
    dead_ratio = jnp.where(fixed_woody, dead_ratio, jnp.float64(1.0))

    live_stem_carbon = jnp.asarray(
        inputs.live_stem_carbon_patch,
        dtype=jnp.float64,
    )
    live_coarse_root_carbon = jnp.asarray(
        inputs.live_coarse_root_carbon_patch,
        dtype=jnp.float64,
    )
    live_stem_nitrogen = jnp.asarray(
        inputs.live_stem_nitrogen_patch,
        dtype=jnp.float64,
    )
    live_coarse_root_nitrogen = jnp.asarray(
        inputs.live_coarse_root_nitrogen_patch,
        dtype=jnp.float64,
    )
    (
        stem_carbon_turnover,
        stem_dead_nitrogen,
        stem_retranslocated_nitrogen,
    ) = _organ_turnover(
        live_stem_carbon,
        live_stem_nitrogen,
        live_ratio,
        dead_ratio,
        turnover_rate,
        floating,
    )
    (
        coarse_root_carbon_turnover,
        coarse_root_dead_nitrogen,
        coarse_root_retranslocated_nitrogen,
    ) = _organ_turnover(
        live_coarse_root_carbon,
        live_coarse_root_nitrogen,
        live_ratio,
        dead_ratio,
        turnover_rate,
        floating,
    )
    stem_retranslocated_nitrogen = jnp.where(
        use_fun,
        jnp.float64(0.0),
        stem_retranslocated_nitrogen,
    )
    coarse_root_retranslocated_nitrogen = jnp.where(
        use_fun,
        jnp.float64(0.0),
        coarse_root_retranslocated_nitrogen,
    )

    for name, candidate in (
        ("live_stem_carbon_to_dead_stem", stem_carbon_turnover),
        (
            "live_coarse_root_carbon_to_dead_coarse_root",
            coarse_root_carbon_turnover,
        ),
        ("live_stem_nitrogen_to_dead_stem", stem_dead_nitrogen),
        (
            "live_stem_nitrogen_to_retranslocation",
            stem_retranslocated_nitrogen,
        ),
        (
            "live_coarse_root_nitrogen_to_dead_coarse_root",
            coarse_root_dead_nitrogen,
        ),
        (
            "live_coarse_root_nitrogen_to_retranslocation",
            coarse_root_retranslocated_nitrogen,
        ),
    ):
        cn_fluxes = _update_flux(cn_fluxes, name, candidate, woody)

    return LivewoodTurnoverResult(
        cn_fluxes=cn_fluxes,
        diagnostics=LivewoodTurnoverDiagnostics(
            woody_patch_count=jnp.sum(woody, dtype=jnp.int32),
            zero_live_stem_carbon_patch_count=jnp.sum(
                woody & (live_stem_carbon == jnp.float64(0.0)),
                dtype=jnp.int32,
            ),
            zero_live_coarse_root_carbon_patch_count=jnp.sum(
                woody & (live_coarse_root_carbon == jnp.float64(0.0)),
                dtype=jnp.int32,
            ),
        ),
    )


__all__ = [
    "LivewoodTurnoverDiagnostics",
    "LivewoodTurnoverInputs",
    "LivewoodTurnoverParameters",
    "LivewoodTurnoverResult",
    "livewood_turnover",
]
