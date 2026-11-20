
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


class BackgroundLitterfallInputs(NamedTuple):

    soil_patch: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    background_litterfall_rate_patch: jnp.ndarray
    leaf_carbon_patch: jnp.ndarray
    fine_root_carbon_patch: jnp.ndarray
    leaf_nitrogen_patch: jnp.ndarray
    fine_root_nitrogen_patch: jnp.ndarray


class BackgroundLitterfallState(NamedTuple):

    leaf_carbon_to_litter_fun_patch: jnp.ndarray
    leaf_carbon_nitrogen_offset_patch: jnp.ndarray


class BackgroundLitterfallParameters(NamedTuple):

    leaf_carbon_nitrogen_ratio_pft: jnp.ndarray
    leaf_litter_carbon_nitrogen_ratio_pft: jnp.ndarray
    fine_root_carbon_nitrogen_ratio_pft: jnp.ndarray
    use_fun: bool | jnp.ndarray = True
    carbon_nitrogen_ratio_floating: bool | jnp.ndarray = True
    time_step_seconds: float | jnp.ndarray = 1800.0


class BackgroundLitterfallDiagnostics(NamedTuple):
    background_patch_count: jnp.ndarray
    zero_leaf_carbon_patch_count: jnp.ndarray
    zero_fine_root_carbon_patch_count: jnp.ndarray
    fine_root_nitrogen_cap_patch_count: jnp.ndarray


class BackgroundLitterfallResult(NamedTuple):
    state: BackgroundLitterfallState
    cn_fluxes: CnTransientFluxState
    diagnostics: BackgroundLitterfallDiagnostics


_FLUX_FIELDS = {
    "leaf_carbon_to_litter": "vegetation_carbon.leafc_to_litter_patch",
    "fine_root_carbon_to_litter": (
        "vegetation_carbon.frootc_to_litter_patch"
    ),
    "leaf_nitrogen_to_litter": "vegetation_nitrogen.leafn_to_litter_patch",
    "leaf_nitrogen_to_retranslocation": (
        "vegetation_nitrogen.leafn_to_retransn_patch"
    ),
    "fine_root_nitrogen_to_litter": (
        "vegetation_nitrogen.frootn_to_litter_patch"
    ),
    "free_retranslocation_to_nitrogen_pool": (
        "vegetation_nitrogen.free_retransn_to_npool_patch"
    ),
    "paid_retranslocation_to_nitrogen_pool": (
        "vegetation_nitrogen.retransn_to_npool_patch"
    ),
}


def _validate_shapes(
    inputs: BackgroundLitterfallInputs,
    state: BackgroundLitterfallState,
    cn_fluxes: CnTransientFluxState,
    parameters: BackgroundLitterfallParameters,
) -> int:
    if jnp.ndim(inputs.soil_patch) != 1:
        raise ValueError("soil_patch must be a vector")
    patch_count = inputs.soil_patch.shape[0]
    for name in inputs._fields[1:]:
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    for name in state._fields:
        if jnp.shape(getattr(state, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    for field_name in _FLUX_FIELDS.values():
        if get_cn_flux_field(cn_fluxes, field_name).shape != (patch_count,):
            raise ValueError("background-litterfall flux state must match patch count")
    for name in (
        "leaf_carbon_nitrogen_ratio_pft",
        "leaf_litter_carbon_nitrogen_ratio_pft",
        "fine_root_carbon_nitrogen_ratio_pft",
    ):
        values = getattr(parameters, name)
        if jnp.ndim(values) != 1 or values.shape[0] < 1:
            raise ValueError(f"{name} must be a nonempty PFT vector")
    for name in (
        "use_fun",
        "carbon_nitrogen_ratio_floating",
        "time_step_seconds",
    ):
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    return patch_count


def _flux(cn_fluxes: CnTransientFluxState, name: str) -> jnp.ndarray:
    return get_cn_flux_field(cn_fluxes, _FLUX_FIELDS[name])


def _update_flux(
    cn_fluxes: CnTransientFluxState,
    name: str,
    candidate: jnp.ndarray,
    mask: jnp.ndarray,
) -> CnTransientFluxState:
    return set_cn_flux_field(
        cn_fluxes,
        _FLUX_FIELDS[name],
        jnp.where(mask, candidate, _flux(cn_fluxes, name)),
    )


def background_litterfall(
    inputs: BackgroundLitterfallInputs,
    state: BackgroundLitterfallState,
    cn_fluxes: CnTransientFluxState,
    parameters: BackgroundLitterfallParameters,
) -> BackgroundLitterfallResult:

    _validate_shapes(inputs, state, cn_fluxes, parameters)
    soil = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    vegetation_type = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    background_rate = jnp.asarray(
        inputs.background_litterfall_rate_patch,
        dtype=jnp.float64,
    )
    background = soil & (background_rate > jnp.float64(0.0))
    background_rate = jnp.where(background, background_rate, 0.0)
    use_fun = jnp.asarray(parameters.use_fun, dtype=jnp.bool_)
    floating = jnp.asarray(
        parameters.carbon_nitrogen_ratio_floating,
        dtype=jnp.bool_,
    )
    time_step = jnp.asarray(parameters.time_step_seconds, dtype=jnp.float64)

    leaf_carbon = jnp.asarray(inputs.leaf_carbon_patch, dtype=jnp.float64)
    fine_root_carbon = jnp.asarray(
        inputs.fine_root_carbon_patch,
        dtype=jnp.float64,
    )
    leaf_nitrogen = jnp.asarray(inputs.leaf_nitrogen_patch, dtype=jnp.float64)
    fine_root_nitrogen = jnp.asarray(
        inputs.fine_root_nitrogen_patch,
        dtype=jnp.float64,
    )
    leaf_ratio_pft = jnp.asarray(
        parameters.leaf_carbon_nitrogen_ratio_pft,
        dtype=jnp.float64,
    )[vegetation_type]
    leaf_litter_ratio_pft = jnp.asarray(
        parameters.leaf_litter_carbon_nitrogen_ratio_pft,
        dtype=jnp.float64,
    )[vegetation_type]
    fine_root_ratio_pft = jnp.asarray(
        parameters.fine_root_carbon_nitrogen_ratio_pft,
        dtype=jnp.float64,
    )[vegetation_type]

    leaf_carbon_to_litter = fortran_multiply(background_rate, leaf_carbon)
    fine_root_carbon_to_litter = fortran_multiply(
        background_rate,
        fine_root_carbon,
    )

    paid_retranslocation = _flux(
        cn_fluxes,
        "paid_retranslocation_to_nitrogen_pool",
    )
    free_retranslocation = _flux(
        cn_fluxes,
        "free_retranslocation_to_nitrogen_pool",
    )
    fun_retranslocation = fortran_add(
        paid_retranslocation,
        free_retranslocation,
    )
    nitrogen_after_retranslocation = fortran_add(
        leaf_nitrogen,
        -fortran_multiply(fun_retranslocation, time_step),
    )
    safe_leaf_nitrogen = jnp.where(
        leaf_nitrogen > jnp.float64(0.0),
        leaf_nitrogen,
        jnp.float64(1.0),
    )
    safe_nitrogen_after = jnp.where(
        nitrogen_after_retranslocation > jnp.float64(0.0),
        nitrogen_after_retranslocation,
        safe_leaf_nitrogen,
    )
    fun_leaf_ratio = jnp.where(
        leaf_nitrogen > jnp.float64(0.0),
        fortran_divide(leaf_carbon, safe_nitrogen_after),
        leaf_ratio_pft,
    )
    previous_leaf_ratio = jnp.asarray(
        state.leaf_carbon_nitrogen_offset_patch,
        dtype=jnp.float64,
    )
    leaf_ratio = jnp.where(
        background & use_fun,
        fun_leaf_ratio,
        previous_leaf_ratio,
    )
    safe_fun_leaf_ratio = jnp.where(
        background & use_fun,
        leaf_ratio,
        jnp.float64(1.0),
    )
    fun_leaf_nitrogen_to_litter = jnp.maximum(
        fortran_add(
            fortran_divide(leaf_carbon_to_litter, safe_fun_leaf_ratio),
            -fun_retranslocation,
        ),
        jnp.float64(0.0),
    )
    denominator = fortran_add(
        fun_retranslocation,
        fun_leaf_nitrogen_to_litter,
    )
    safe_denominator = jnp.where(
        denominator != jnp.float64(0.0),
        denominator,
        jnp.float64(1.0),
    )
    fun_litter_fraction = jnp.where(
        denominator != jnp.float64(0.0),
        fortran_divide(fun_leaf_nitrogen_to_litter, safe_denominator),
        jnp.where(
            fun_leaf_nitrogen_to_litter == jnp.float64(0.0),
            jnp.float64(0.0),
            jnp.float64(1.0),
        ),
    )

    fixed_leaf_nitrogen_to_litter = fortran_divide(
        leaf_carbon_to_litter,
        jnp.where(background & ~use_fun & ~floating, leaf_litter_ratio_pft, 1.0),
    )
    fixed_leaf_retranslocation = fortran_add(
        fortran_divide(leaf_carbon_to_litter,
                       jnp.where(background & ~use_fun & ~floating, leaf_ratio_pft, 1.0)),
        -fixed_leaf_nitrogen_to_litter,
    )
    leaf_nitrogen_to_litter = jnp.where(
        use_fun,
        fun_leaf_nitrogen_to_litter,
        fixed_leaf_nitrogen_to_litter,
    )
    leaf_nitrogen_to_retranslocation = jnp.where(
        use_fun,
        fun_retranslocation,
        fixed_leaf_retranslocation,
    )
    litter_fraction = jnp.where(
        use_fun,
        fun_litter_fraction,
        jnp.float64(0.5),
    )
    fine_root_nitrogen_to_litter = fortran_divide(
        fine_root_carbon_to_litter,
        jnp.where(background & ~floating, fine_root_ratio_pft, 1.0),
    )

    safe_leaf_carbon = jnp.where(
        leaf_carbon == jnp.float64(0.0),
        jnp.float64(1.0),
        leaf_carbon,
    )
    leaf_nitrogen_turnover = jnp.where(
        leaf_carbon == jnp.float64(0.0),
        jnp.float64(0.0),
        fortran_multiply(
            leaf_carbon_to_litter,
            fortran_divide(leaf_nitrogen, safe_leaf_carbon),
        ),
    )
    floating_leaf_nitrogen_to_litter = fortran_multiply(
        litter_fraction,
        leaf_nitrogen_turnover,
    )
    floating_leaf_retranslocation = fortran_add(
        leaf_nitrogen_turnover,
        -floating_leaf_nitrogen_to_litter,
    )
    safe_fine_root_carbon = jnp.where(
        fine_root_carbon == jnp.float64(0.0),
        jnp.float64(1.0),
        fine_root_carbon,
    )
    floating_fine_root_nitrogen_to_litter = jnp.where(
        fine_root_carbon == jnp.float64(0.0),
        jnp.float64(0.0),
        fortran_multiply(
            fine_root_carbon_to_litter,
            fortran_divide(fine_root_nitrogen, safe_fine_root_carbon),
        ),
    )
    leaf_nitrogen_to_litter = jnp.where(
        floating,
        floating_leaf_nitrogen_to_litter,
        leaf_nitrogen_to_litter,
    )
    leaf_nitrogen_to_retranslocation = jnp.where(
        floating,
        floating_leaf_retranslocation,
        leaf_nitrogen_to_retranslocation,
    )
    fine_root_nitrogen_to_litter = jnp.where(
        floating,
        floating_fine_root_nitrogen_to_litter,
        fine_root_nitrogen_to_litter,
    )

    fine_root_nitrogen_cap = background & use_fun & (
        fortran_multiply(fine_root_nitrogen_to_litter, time_step)
        > fine_root_nitrogen
    )
    fine_root_nitrogen_to_litter = jnp.where(
        fine_root_nitrogen_cap,
        fortran_divide(fine_root_nitrogen, time_step),
        fine_root_nitrogen_to_litter,
    )

    for name, candidate in (
        ("leaf_carbon_to_litter", leaf_carbon_to_litter),
        ("fine_root_carbon_to_litter", fine_root_carbon_to_litter),
        ("leaf_nitrogen_to_litter", leaf_nitrogen_to_litter),
        (
            "leaf_nitrogen_to_retranslocation",
            leaf_nitrogen_to_retranslocation,
        ),
        ("fine_root_nitrogen_to_litter", fine_root_nitrogen_to_litter),
    ):
        cn_fluxes = _update_flux(cn_fluxes, name, candidate, background)

    return BackgroundLitterfallResult(
        state=BackgroundLitterfallState(
            leaf_carbon_to_litter_fun_patch=jnp.where(
                background & use_fun,
                leaf_carbon_to_litter,
                jnp.asarray(
                    state.leaf_carbon_to_litter_fun_patch,
                    dtype=jnp.float64,
                ),
            ),
            leaf_carbon_nitrogen_offset_patch=leaf_ratio,
        ),
        cn_fluxes=cn_fluxes,
        diagnostics=BackgroundLitterfallDiagnostics(
            background_patch_count=jnp.sum(background, dtype=jnp.int32),
            zero_leaf_carbon_patch_count=jnp.sum(
                background & (leaf_carbon == jnp.float64(0.0)),
                dtype=jnp.int32,
            ),
            zero_fine_root_carbon_patch_count=jnp.sum(
                background & (fine_root_carbon == jnp.float64(0.0)),
                dtype=jnp.int32,
            ),
            fine_root_nitrogen_cap_patch_count=jnp.sum(
                fine_root_nitrogen_cap,
                dtype=jnp.int32,
            ),
        ),
    )


__all__ = [
    "BackgroundLitterfallDiagnostics",
    "BackgroundLitterfallInputs",
    "BackgroundLitterfallParameters",
    "BackgroundLitterfallResult",
    "BackgroundLitterfallState",
    "background_litterfall",
]
