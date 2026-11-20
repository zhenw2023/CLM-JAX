
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
from clm_jax.physics.ordered_gpu_reduction import _instruction


def ordered_profile_gather(terms, profiles, initial, column_patch, valid, last_slot,
                           active_level, masks=()):
    coefficients = tuple(item[2] for item in terms)
    patch_count, levels = profiles[0].shape
    columns = column_patch.shape[0]
    nc, np, ni, nm = len(coefficients), len(profiles), len(initial), len(masks)
    block = 128

    def kernel(*refs):
        cref, pref = refs[:nc], refs[nc:nc + np]
        iref, mref = refs[nc + np:nc + np + ni], refs[nc + np + ni:nc + np + ni + nm]
        offset = nc + np + ni + nm
        patch_ref, valid_ref, last_ref, level_ref = refs[offset:offset + 4]
        oref = refs[offset + 4:]
        index = pl.program_id(0) * block + jnp.arange(block)
        column, level = index // levels, index % levels
        in_bounds = column < columns
        level_count = plt.load(level_ref.at[()])
        limit = jnp.max(plt.load(last_ref.at[column], mask=in_bounds, other=0))
        totals = tuple(plt.load(ref.at[column, level], mask=in_bounds, other=0.) for ref in iref)

        def accumulate(slot, carry):
            enabled = in_bounds & (level < level_count)
            enabled &= plt.load(valid_ref.at[column, slot], mask=in_bounds, other=False)
            patch = plt.load(patch_ref.at[column, slot], mask=in_bounds, other=0)
            patch = jnp.clip(patch, 0, patch_count - 1)
            profile = tuple(plt.load(ref.at[patch, level], mask=enabled, other=0.) for ref in pref)
            term_masks = tuple(plt.load(ref.at[patch], mask=enabled, other=False) for ref in mref)
            updated = list(carry)
            for term_index, (target, profile_index, _, mask_index) in enumerate(terms):
                update = enabled if mask_index < 0 else enabled & term_masks[mask_index]
                coefficient = plt.load(cref[term_index].at[patch], mask=update, other=0.)
                product = _instruction("mul", coefficient, profile[profile_index])
                product = jnp.copysign(jnp.abs(product), product)
                value = _instruction("add", updated[target], product)
                value = jnp.copysign(jnp.abs(value), value)
                updated[target] = jnp.where(update, value, updated[target])
            return tuple(updated)

        totals = lax.fori_loop(0, limit, accumulate, totals)
        for ref, total in zip(oref, totals, strict=True):
            plt.store(ref.at[column, level], total, mask=in_bounds)

    return pl.pallas_call(
        kernel, out_shape=tuple(jax.ShapeDtypeStruct(value.shape, value.dtype) for value in initial),
        grid=(pl.cdiv(columns * levels, block),),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
    )(*coefficients, *profiles, *initial, *masks, column_patch, valid, last_slot, active_level)
