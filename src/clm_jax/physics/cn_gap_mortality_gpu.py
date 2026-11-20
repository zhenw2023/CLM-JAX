
import jax
import jax.numpy as jnp

from clm_jax.physics.cn_zero import get_cn_flux_field, set_cn_flux_field
from clm_jax.physics.numerics import (
    fortran_hardware_add as add, fortran_hardware_multiply as multiply,
)


def _terms(inputs, fluxes, parameters):
    terms = []
    pft = inputs.vegetation_type_patch
    weight = inputs.patch_weight_on_column

    def term(target, profile, element, names, fraction=None):
        prefix = "vegetation_carbon" if element == "c" else "vegetation_nitrogen"
        values = tuple(get_cn_flux_field(fluxes, f"{prefix}.m_{name}_to_litter_patch")
                       for name in names)
        coefficient = values[0] if len(values) == 1 else add(*values)
        if fraction is not None:
            coefficient = multiply(coefficient, fraction[pft])
        terms.append((target, profile, multiply(coefficient, weight)))

    leaf_fractions = tuple(getattr(parameters, f"leaf_{name}_fraction_pft")
                           for name in ("labile", "cellulose", "lignin"))
    root_fractions = tuple(getattr(parameters, f"fine_root_{name}_fraction_pft")
                           for name in ("labile", "cellulose", "lignin"))
    for element, base in (("c", 0), ("n", 4)):
        for offset, fraction in enumerate(leaf_fractions):
            term(base + offset, 0, element, (f"leaf{element}",), fraction)
        for offset, fraction in enumerate(root_fractions):
            term(base + offset, 1, element, (f"froot{element}",), fraction)
        term(base + 3, 3, element, (f"livestem{element}", f"deadstem{element}"))
        term(base + 3, 2, element, (f"livecroot{element}", f"deadcroot{element}"))
        if element == "n":
            term(base, 0, element, ("retransn",))
        for pool in ("storage", "xfer"):
            leaf = (f"leaf{element}_{pool}",)
            if element == "c":
                leaf += (f"gresp_{pool}",)
            term(base, 0, element, leaf)
            term(base, 1, element, (f"froot{element}_{pool}",))
            term(base, 3, element, (f"livestem{element}_{pool}", f"deadstem{element}_{pool}"))
            term(base, 2, element, (f"livecroot{element}_{pool}", f"deadcroot{element}_{pool}"))
    return tuple(terms)


def _gather(inputs, fluxes, parameters):
    from clm_jax.physics.cn_gap_mortality import _CARBON_COLUMN_OUTPUTS, _NITROGEN_COLUMN_OUTPUTS

    names = (*_CARBON_COLUMN_OUTPUTS, *_NITROGEN_COLUMN_OUTPUTS)
    initial = tuple(get_cn_flux_field(fluxes, name) for name in names)
    patch_count = inputs.soil_patch.size
    column_patch = inputs.column_patch_indices
    columns, slots = column_patch.shape
    if not patch_count or not slots or not columns:
        return fluxes, jnp.int32(0)
    bounded = jnp.clip(column_patch, 0, patch_count - 1)
    valid = inputs.soil_column[:, None] & (column_patch >= 0) & inputs.patch_active[bounded]
    last_slot = jnp.max(jnp.where(valid, jnp.arange(slots)[None, :] + 1, 0), axis=1)
    terms = _terms(inputs, fluxes, parameters)
    profiles = tuple(getattr(inputs, name + "_profile_patch_level")
                     for name in ("leaf", "fine_root", "coarse_root", "stem"))
    from clm_jax.physics.ordered_profile_gather_gpu import ordered_profile_gather
    outputs = ordered_profile_gather(
        tuple((target, profile, coefficient, -1) for target, profile, coefficient in terms),
        profiles, initial, column_patch, valid, last_slot,
        jnp.asarray(parameters.decomposition_level_count, jnp.int32),
    )
    for name, value in zip(names, outputs, strict=True):
        fluxes = set_cn_flux_field(fluxes, name, value)
    return fluxes, jnp.sum(valid, dtype=jnp.int32)


@jax.custom_jvp
def gather_gap_fluxes(inputs, fluxes, parameters):
    from clm_jax.physics.cn_gap_mortality import _accumulate_patch_fluxes_to_columns
    if jax.default_backend() != "gpu":
        return _accumulate_patch_fluxes_to_columns(inputs, fluxes, parameters)
    return _gather(inputs, fluxes, parameters)


@gather_gap_fluxes.defjvp
def _derivative(primals, tangents):
    from clm_jax.physics.cn_gap_mortality import _accumulate_patch_fluxes_to_columns
    return jax.jvp(_accumulate_patch_fluxes_to_columns, primals, tangents)
