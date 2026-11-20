
import jax
import jax.numpy as jnp
from clm_jax.physics.cn_zero import get_cn_flux_field, set_cn_flux_field
from clm_jax.physics.numerics import fortran_hardware_multiply as multiply
from clm_jax.physics.ordered_profile_gather_gpu import ordered_profile_gather


def _gather(inputs, fluxes, parameters):
    from clm_jax.physics.litter_to_column import (
        _validate_shapes, _INPUT_CARBON_FIELDS, _INPUT_NITROGEN_FIELDS,
        _OUTPUT_CARBON_FIELDS, _OUTPUT_NITROGEN_FIELDS,
        LitterToColumnResult, LitterToColumnDiagnostics,
    )
    patches, columns = _validate_shapes(inputs, fluxes, parameters)
    column_patch = inputs.column_patch_indices
    if not patches or not columns or not column_patch.shape[1]:
        from clm_jax.physics.litter_to_column import _litter_to_column_reference
        return _litter_to_column_reference(inputs, fluxes, parameters)
    names = (*_OUTPUT_CARBON_FIELDS, *_OUTPUT_NITROGEN_FIELDS)
    initial = tuple(get_cn_flux_field(fluxes, name) for name in names)
    bounded = jnp.clip(column_patch, 0, patches - 1)
    valid = inputs.soil_column[:, None] & (column_patch >= 0) & inputs.patch_active[bounded]
    last_slot = jnp.max(jnp.where(valid, jnp.arange(column_patch.shape[1])[None, :] + 1, 0), axis=1)
    pft, weight = inputs.vegetation_type_patch, inputs.patch_weight_on_column
    crop = pft >= parameters.first_prognostic_crop_pft
    grain = crop & ~jnp.asarray(parameters.use_grain_product, jnp.bool_)
    terms = []
    for organ in range(4):
        profile = 1 if organ == 1 else 0
        fraction_names = parameters._fields[3:6] if organ == 1 else parameters._fields[:3]
        mask = -1 if organ < 2 else organ - 2
        for element, fields in enumerate((_INPUT_CARBON_FIELDS, _INPUT_NITROGEN_FIELDS)):
            value = get_cn_flux_field(fluxes, fields[organ])
            for fraction_index, name in enumerate(fraction_names):
                coefficient = multiply(multiply(value, getattr(parameters, name)[pft]), weight)
                terms.append((3 * element + fraction_index, profile, coefficient, mask))
    outputs = ordered_profile_gather(
        tuple(terms), (inputs.leaf_profile_patch_level, inputs.fine_root_profile_patch_level),
        initial, column_patch, valid, last_slot,
        jnp.asarray(parameters.decomposition_level_count, jnp.int32), (crop, grain),
    )
    for name, value in zip(names, outputs, strict=True):
        fluxes = set_cn_flux_field(fluxes, name, value)
    return LitterToColumnResult(fluxes, LitterToColumnDiagnostics(
        jnp.sum(inputs.soil_column, dtype=jnp.int32), jnp.sum(valid, dtype=jnp.int32),
        jnp.sum(valid & crop[bounded], dtype=jnp.int32),
    ))


@jax.custom_jvp
def gather_litter_fluxes(inputs, fluxes, parameters):
    from clm_jax.physics.litter_to_column import _litter_to_column_reference
    if jax.default_backend() != "gpu":
        return _litter_to_column_reference(inputs, fluxes, parameters)
    return _gather(inputs, fluxes, parameters)


@gather_litter_fluxes.defjvp
def _derivative(primals, tangents):
    from clm_jax.physics.litter_to_column import _litter_to_column_reference
    return jax.jvp(_litter_to_column_reference, primals, tangents)
