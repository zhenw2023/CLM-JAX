
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


class OffsetLitterfallInputs(NamedTuple):

    soil_patch: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    offset_flag_patch: jnp.ndarray
    offset_counter_seconds_patch: jnp.ndarray
    leaf_carbon_patch: jnp.ndarray
    fine_root_carbon_patch: jnp.ndarray
    grain_carbon_patch: jnp.ndarray
    crop_seed_carbon_deficit_patch: jnp.ndarray
    live_stem_carbon_patch: jnp.ndarray
    leaf_nitrogen_patch: jnp.ndarray
    fine_root_nitrogen_patch: jnp.ndarray
    grain_nitrogen_patch: jnp.ndarray
    crop_seed_nitrogen_deficit_patch: jnp.ndarray
    live_stem_nitrogen_patch: jnp.ndarray


class OffsetLitterfallState(NamedTuple):

    previous_leaf_carbon_to_litter_patch: jnp.ndarray
    previous_fine_root_carbon_to_litter_patch: jnp.ndarray
    leaf_carbon_to_litter_fun_patch: jnp.ndarray
    leaf_carbon_nitrogen_offset_patch: jnp.ndarray


class OffsetLitterfallParameters(NamedTuple):

    leaf_carbon_nitrogen_ratio_pft: jnp.ndarray
    leaf_litter_carbon_nitrogen_ratio_pft: jnp.ndarray
    fine_root_carbon_nitrogen_ratio_pft: jnp.ndarray
    first_crop_pft: int | jnp.ndarray = 17
    use_fun: bool | jnp.ndarray = True
    carbon_nitrogen_ratio_floating: bool | jnp.ndarray = True
    time_step_seconds: float | jnp.ndarray = 1800.0


class OffsetLitterfallDiagnostics(NamedTuple):
    offset_patch_count: jnp.ndarray
    final_offset_step_patch_count: jnp.ndarray
    crop_offset_patch_count: jnp.ndarray
    leaf_carbon_cap_patch_count: jnp.ndarray
    fine_root_carbon_cap_patch_count: jnp.ndarray
    fine_root_nitrogen_cap_patch_count: jnp.ndarray


class OffsetLitterfallResult(NamedTuple):
    state: OffsetLitterfallState
    cn_fluxes: CnTransientFluxState
    diagnostics: OffsetLitterfallDiagnostics


_FLUX_FIELDS = {
    "carbon_to_grain": "vegetation_carbon.cpool_to_grainc_patch",
    "nitrogen_to_grain": "vegetation_nitrogen.npool_to_grainn_patch",
    "carbon_to_live_stem": "vegetation_carbon.cpool_to_livestemc_patch",
    "carbon_to_leaf": "vegetation_carbon.cpool_to_leafc_patch",
    "carbon_to_fine_root": "vegetation_carbon.cpool_to_frootc_patch",
    "leaf_carbon_to_litter": "vegetation_carbon.leafc_to_litter_patch",
    "fine_root_carbon_to_litter": "vegetation_carbon.frootc_to_litter_patch",
    "live_stem_carbon_to_litter": "vegetation_carbon.livestemc_to_litter_patch",
    "grain_carbon_to_food": "vegetation_carbon.grainc_to_food_patch",
    "grain_carbon_to_seed": "vegetation_carbon.grainc_to_seed_patch",
    "live_stem_nitrogen_to_litter": (
        "vegetation_nitrogen.livestemn_to_litter_patch"
    ),
    "grain_nitrogen_to_food": "vegetation_nitrogen.grainn_to_food_patch",
    "grain_nitrogen_to_seed": "vegetation_nitrogen.grainn_to_seed_patch",
    "leaf_nitrogen_to_litter": "vegetation_nitrogen.leafn_to_litter_patch",
    "leaf_nitrogen_to_retranslocation": (
        "vegetation_nitrogen.leafn_to_retransn_patch"
    ),
    "free_retranslocation_to_nitrogen_pool": (
        "vegetation_nitrogen.free_retransn_to_npool_patch"
    ),
    "paid_retranslocation_to_nitrogen_pool": (
        "vegetation_nitrogen.retransn_to_npool_patch"
    ),
    "fine_root_nitrogen_to_litter": (
        "vegetation_nitrogen.frootn_to_litter_patch"
    ),
}


def _validate_shapes(
    inputs: OffsetLitterfallInputs,
    state: OffsetLitterfallState,
    cn_fluxes: CnTransientFluxState,
    parameters: OffsetLitterfallParameters,
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
            raise ValueError("offset-litterfall flux state must match patch count")
    for name in (
        "leaf_carbon_nitrogen_ratio_pft",
        "leaf_litter_carbon_nitrogen_ratio_pft",
        "fine_root_carbon_nitrogen_ratio_pft",
    ):
        values = getattr(parameters, name)
        if jnp.ndim(values) != 1 or values.shape[0] < 1:
            raise ValueError(f"{name} must be a nonempty PFT vector")
    for name in (
        "first_crop_pft",
        "use_fun",
        "carbon_nitrogen_ratio_floating",
        "time_step_seconds",
    ):
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    return patch_count


def _flux(cn_fluxes: CnTransientFluxState, name: str) -> jnp.ndarray:
    return get_cn_flux_field(cn_fluxes, _FLUX_FIELDS[name])


def _set_flux(
    cn_fluxes: CnTransientFluxState,
    name: str,
    values: jnp.ndarray,
) -> CnTransientFluxState:
    return set_cn_flux_field(cn_fluxes, _FLUX_FIELDS[name], values)


def _update_flux(
    cn_fluxes: CnTransientFluxState,
    name: str,
    candidate: jnp.ndarray,
    mask: jnp.ndarray,
) -> CnTransientFluxState:
    return _set_flux(
        cn_fluxes,
        name,
        jnp.where(mask, candidate, _flux(cn_fluxes, name)),
    )


def offset_litterfall(
    inputs: OffsetLitterfallInputs,
    state: OffsetLitterfallState,
    cn_fluxes: CnTransientFluxState,
    parameters: OffsetLitterfallParameters,
) -> OffsetLitterfallResult:

    _validate_shapes(inputs, state, cn_fluxes, parameters)
    soil = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    vegetation_type = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    offset_flag = jnp.asarray(inputs.offset_flag_patch, dtype=jnp.float64)
    offset_counter = jnp.asarray(
        inputs.offset_counter_seconds_patch,
        dtype=jnp.float64,
    )
    time_step = jnp.asarray(parameters.time_step_seconds, dtype=jnp.float64)
    use_fun = jnp.asarray(parameters.use_fun, dtype=jnp.bool_)
    floating = jnp.asarray(
        parameters.carbon_nitrogen_ratio_floating,
        dtype=jnp.bool_,
    )
    crop = vegetation_type >= jnp.asarray(parameters.first_crop_pft, dtype=jnp.int32)
    offset = soil & (offset_flag == jnp.float64(1.0))
    final_step = offset & (offset_counter == time_step)
    ordinary_step = offset & ~final_step
    crop_offset = offset & crop
    final_crop = final_step & crop

    leaf_carbon = jnp.asarray(inputs.leaf_carbon_patch, dtype=jnp.float64)
    fine_root_carbon = jnp.asarray(
        inputs.fine_root_carbon_patch,
        dtype=jnp.float64,
    )
    grain_carbon = jnp.asarray(inputs.grain_carbon_patch, dtype=jnp.float64)
    live_stem_carbon = jnp.asarray(
        inputs.live_stem_carbon_patch,
        dtype=jnp.float64,
    )
    leaf_nitrogen = jnp.asarray(inputs.leaf_nitrogen_patch, dtype=jnp.float64)
    fine_root_nitrogen = jnp.asarray(
        inputs.fine_root_nitrogen_patch,
        dtype=jnp.float64,
    )
    grain_nitrogen = jnp.asarray(inputs.grain_nitrogen_patch, dtype=jnp.float64)
    live_stem_nitrogen = jnp.asarray(
        inputs.live_stem_nitrogen_patch,
        dtype=jnp.float64,
    )

    final_rate = fortran_divide(jnp.float64(1.0), time_step)
    safe_counter = jnp.where(ordinary_step, offset_counter, time_step)
    ordinary_rate = fortran_divide(
        fortran_multiply(time_step, jnp.float64(2.0)),
        fortran_multiply(safe_counter, safe_counter),
    )

    carbon_to_leaf = _flux(cn_fluxes, "carbon_to_leaf")
    carbon_to_fine_root = _flux(cn_fluxes, "carbon_to_fine_root")
    final_leaf_carbon = fortran_add(
        fortran_multiply(final_rate, leaf_carbon),
        carbon_to_leaf,
    )
    final_fine_root_carbon = fortran_add(
        fortran_multiply(final_rate, fine_root_carbon),
        carbon_to_fine_root,
    )
    previous_leaf_carbon = jnp.asarray(
        state.previous_leaf_carbon_to_litter_patch,
        dtype=jnp.float64,
    )
    previous_fine_root_carbon = jnp.asarray(
        state.previous_fine_root_carbon_to_litter_patch,
        dtype=jnp.float64,
    )
    ordinary_previous_leaf = jnp.where(ordinary_step, previous_leaf_carbon, 0.0)
    ordinary_previous_root = jnp.where(ordinary_step, previous_fine_root_carbon, 0.0)
    ordinary_leaf_carbon = fortran_add(
        ordinary_previous_leaf,
        fortran_multiply(
            ordinary_rate,
            fortran_add(
                leaf_carbon,
                -fortran_multiply(ordinary_previous_leaf, safe_counter),
            ),
        ),
    )
    ordinary_fine_root_carbon = fortran_add(
        ordinary_previous_root,
        fortran_multiply(
            ordinary_rate,
            fortran_add(
                fine_root_carbon,
                -fortran_multiply(ordinary_previous_root, safe_counter),
            ),
        ),
    )
    leaf_carbon_to_litter = jnp.where(
        final_step,
        final_leaf_carbon,
        jnp.where(
            ordinary_step,
            ordinary_leaf_carbon,
            _flux(cn_fluxes, "leaf_carbon_to_litter"),
        ),
    )
    fine_root_carbon_to_litter = jnp.where(
        final_step,
        final_fine_root_carbon,
        jnp.where(
            ordinary_step,
            ordinary_fine_root_carbon,
            _flux(cn_fluxes, "fine_root_carbon_to_litter"),
        ),
    )

    leaf_carbon_cap = offset & use_fun & (
        fortran_multiply(leaf_carbon_to_litter, time_step) > leaf_carbon
    )
    fine_root_carbon_cap = offset & use_fun & (
        fortran_multiply(fine_root_carbon_to_litter, time_step)
        > fine_root_carbon
    )
    leaf_carbon_to_litter = jnp.where(
        leaf_carbon_cap,
        fortran_add(
            fortran_divide(leaf_carbon, time_step),
            carbon_to_leaf,
        ),
        leaf_carbon_to_litter,
    )
    fine_root_carbon_to_litter = jnp.where(
        fine_root_carbon_cap,
        fortran_add(
            fortran_divide(fine_root_carbon, time_step),
            carbon_to_fine_root,
        ),
        fine_root_carbon_to_litter,
    )
    cn_fluxes = _update_flux(
        cn_fluxes,
        "leaf_carbon_to_litter",
        leaf_carbon_to_litter,
        offset,
    )
    cn_fluxes = _update_flux(
        cn_fluxes,
        "fine_root_carbon_to_litter",
        fine_root_carbon_to_litter,
        offset,
    )

    grain_carbon_to_seed = fortran_multiply(
        final_rate,
        jnp.minimum(
            -jnp.asarray(inputs.crop_seed_carbon_deficit_patch, dtype=jnp.float64),
            grain_carbon,
        ),
    )
    grain_nitrogen_to_seed = fortran_multiply(
        final_rate,
        jnp.minimum(
            -jnp.asarray(
                inputs.crop_seed_nitrogen_deficit_patch,
                dtype=jnp.float64,
            ),
            grain_nitrogen,
        ),
    )
    grain_carbon_to_food = fortran_add(
        fortran_add(
            fortran_multiply(final_rate, grain_carbon),
            _flux(cn_fluxes, "carbon_to_grain"),
        ),
        -grain_carbon_to_seed,
    )
    grain_nitrogen_to_food = fortran_add(
        fortran_add(
            fortran_multiply(final_rate, grain_nitrogen),
            _flux(cn_fluxes, "nitrogen_to_grain"),
        ),
        -grain_nitrogen_to_seed,
    )
    live_stem_carbon_to_litter = fortran_add(
        fortran_multiply(final_rate, live_stem_carbon),
        _flux(cn_fluxes, "carbon_to_live_stem"),
    )
    for name, candidate in (
        ("grain_carbon_to_seed", grain_carbon_to_seed),
        ("grain_nitrogen_to_seed", grain_nitrogen_to_seed),
        ("grain_carbon_to_food", grain_carbon_to_food),
        ("grain_nitrogen_to_food", grain_nitrogen_to_food),
        ("live_stem_carbon_to_litter", live_stem_carbon_to_litter),
    ):
        cn_fluxes = _update_flux(cn_fluxes, name, candidate, final_crop)

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
        offset & use_fun,
        fun_leaf_ratio,
        previous_leaf_ratio,
    )
    safe_fun_leaf_ratio = jnp.where(
        offset & use_fun,
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
        jnp.where(offset & ~use_fun & ~floating, leaf_litter_ratio_pft, 1.0),
    )
    fixed_leaf_retranslocation = fortran_add(
        fortran_divide(leaf_carbon_to_litter,
                       jnp.where(offset & ~use_fun & ~floating, leaf_ratio_pft, 1.0)),
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
        jnp.where(offset & ~floating, fine_root_ratio_pft, 1.0),
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

    fine_root_nitrogen_cap = offset & use_fun & (
        fortran_multiply(fine_root_nitrogen_to_litter, time_step)
        > fine_root_nitrogen
    )
    fine_root_nitrogen_to_litter = jnp.where(
        fine_root_nitrogen_cap,
        fortran_divide(fine_root_nitrogen, time_step),
        fine_root_nitrogen_to_litter,
    )
    cn_fluxes = _update_flux(
        cn_fluxes,
        "leaf_nitrogen_to_litter",
        leaf_nitrogen_to_litter,
        offset,
    )
    cn_fluxes = _update_flux(
        cn_fluxes,
        "leaf_nitrogen_to_retranslocation",
        leaf_nitrogen_to_retranslocation,
        offset,
    )
    cn_fluxes = _update_flux(
        cn_fluxes,
        "fine_root_nitrogen_to_litter",
        fine_root_nitrogen_to_litter,
        offset,
    )
    cn_fluxes = _update_flux(
        cn_fluxes,
        "live_stem_nitrogen_to_litter",
        fortran_divide(live_stem_nitrogen, time_step),
        crop_offset,
    )

    return OffsetLitterfallResult(
        state=OffsetLitterfallState(
            previous_leaf_carbon_to_litter_patch=jnp.where(
                offset,
                leaf_carbon_to_litter,
                previous_leaf_carbon,
            ),
            previous_fine_root_carbon_to_litter_patch=jnp.where(
                offset,
                fine_root_carbon_to_litter,
                previous_fine_root_carbon,
            ),
            leaf_carbon_to_litter_fun_patch=jnp.where(
                offset & use_fun,
                leaf_carbon_to_litter,
                jnp.asarray(
                    state.leaf_carbon_to_litter_fun_patch,
                    dtype=jnp.float64,
                ),
            ),
            leaf_carbon_nitrogen_offset_patch=leaf_ratio,
        ),
        cn_fluxes=cn_fluxes,
        diagnostics=OffsetLitterfallDiagnostics(
            offset_patch_count=jnp.sum(offset, dtype=jnp.int32),
            final_offset_step_patch_count=jnp.sum(final_step, dtype=jnp.int32),
            crop_offset_patch_count=jnp.sum(crop_offset, dtype=jnp.int32),
            leaf_carbon_cap_patch_count=jnp.sum(leaf_carbon_cap, dtype=jnp.int32),
            fine_root_carbon_cap_patch_count=jnp.sum(
                fine_root_carbon_cap,
                dtype=jnp.int32,
            ),
            fine_root_nitrogen_cap_patch_count=jnp.sum(
                fine_root_nitrogen_cap,
                dtype=jnp.int32,
            ),
        ),
    )


__all__ = [
    "OffsetLitterfallDiagnostics",
    "OffsetLitterfallInputs",
    "OffsetLitterfallParameters",
    "OffsetLitterfallResult",
    "OffsetLitterfallState",
    "offset_litterfall",
]
