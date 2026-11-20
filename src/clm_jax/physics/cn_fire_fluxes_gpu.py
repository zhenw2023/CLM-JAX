
import jax
import jax.numpy as jnp

from clm_jax.physics.cn_zero import get_cn_flux_field, set_cn_flux_field
from clm_jax.physics.ordered_profile_gather_gpu import ordered_profile_gather


def _gather(inputs, fluxes, parameters):
    from clm_jax.physics.cn_fire_fluxes import (
        _CARBON_MORTALITY_ASSIGNMENTS,
        _CARBON_TRANSFER_ASSIGNMENTS,
        _COLUMN_OUTPUT_FIELDS,
        _NITROGEN_MORTALITY_ASSIGNMENTS,
        _NITROGEN_TRANSFER_ASSIGNMENTS,
        _PATCH_COLUMN_TERM_TARGETS,
        _accumulate_patch_fluxes_to_columns_reference,
        _patch_column_terms,
    )

    patches, columns = inputs.soil_patch.size, inputs.soil_column.size
    column_patch = inputs.column_patch_indices
    if not patches or not columns or not column_patch.shape[1]:
        return _accumulate_patch_fluxes_to_columns_reference(inputs, fluxes, parameters)
    profiles = tuple(
        getattr(inputs, name + "_profile_patch_level")
        for name in (
            "leaf",
            "fine_root",
            "coarse_root",
            "stem",
        )
    )
    leaf = tuple(
        getattr(parameters, "leaf_" + name + "_fraction_pft")
        for name in (
            "labile",
            "cellulose",
            "lignin",
        )
    )
    root = tuple(
        getattr(parameters, "fine_root_" + name + "_fraction_pft")
        for name in (
            "labile",
            "cellulose",
            "lignin",
        )
    )
    names = {
        name
        for assignments in (
            _CARBON_MORTALITY_ASSIGNMENTS,
            _CARBON_TRANSFER_ASSIGNMENTS,
            _NITROGEN_MORTALITY_ASSIGNMENTS,
            _NITROGEN_TRANSFER_ASSIGNMENTS,
        )
        for _, name, *_ in assignments
    }
    patch_fluxes = {name: get_cn_flux_field(fluxes, name) for name in names}
    contributions = _patch_column_terms(
        profiles,
        leaf,
        root,
        patch_fluxes,
        inputs.patch_weight_on_column,
        inputs.vegetation_type_patch,
    )
    bounded = jnp.clip(column_patch, 0, patches - 1)
    valid = inputs.soil_column[:, None] & (column_patch >= 0) & inputs.patch_active[bounded]
    last = jnp.max(jnp.where(valid, jnp.arange(column_patch.shape[1])[None, :] + 1, 0), axis=1)
    initial = tuple(get_cn_flux_field(fluxes, name) for name in _COLUMN_OUTPUT_FIELDS)
    unity = jnp.ones(patches, jnp.float64)
    terms = tuple(
        (target, index, unity, -1) for index, target in enumerate(_PATCH_COLUMN_TERM_TARGETS)
    )
    result = ordered_profile_gather(
        terms,
        contributions,
        initial,
        column_patch,
        valid,
        last,
        jnp.asarray(parameters.decomposition_level_count, jnp.int32),
    )
    for name, value in zip(_COLUMN_OUTPUT_FIELDS, result, strict=True):
        fluxes = set_cn_flux_field(fluxes, name, value)
    return fluxes, jnp.sum(valid, dtype=jnp.int32)


@jax.custom_jvp
def gather_fire_fluxes(inputs, fluxes, parameters):
    if jax.default_backend() != "gpu":
        from clm_jax.physics.cn_fire_fluxes import _accumulate_patch_fluxes_to_columns_reference

        return _accumulate_patch_fluxes_to_columns_reference(inputs, fluxes, parameters)
    return _gather(inputs, fluxes, parameters)


@gather_fire_fluxes.defjvp
def _derivative(primals, tangents):
    from clm_jax.physics.cn_fire_fluxes import _accumulate_patch_fluxes_to_columns_reference

    return jax.jvp(_accumulate_patch_fluxes_to_columns_reference, primals, tangents)
