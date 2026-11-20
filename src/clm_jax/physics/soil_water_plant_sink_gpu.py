from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from clm_jax.physics.plant_hydraulics import ROOT
from clm_jax.physics.surface_hydrology import NUMBER_OF_SOIL_LEVELS


def _rn(operation, left, right):
    result, = plt.elementwise_inline_asm(
        f"{operation}.rn.f64 $0, $1, $2;", args=[left, right],
        constraints="=d,d,d", pack=1,
        result_shape_dtypes=[jax.ShapeDtypeStruct(left.shape, jnp.float64)],
    )
    return jnp.where(result == 0., 0., result)


@partial(jax.custom_jvp, nondiff_argnums=(7,))
def accumulate(inputs, first, owners, active, weight, hydraulic, road, limit):
    columns, patches, levels, block = first.size, owners.size, NUMBER_OF_SOIL_LEVELS, 128
    if columns == 0 or patches == 0:
        zeros = jnp.zeros((columns, levels), jnp.float64)
        return zeros, zeros, jnp.zeros(columns, jnp.float64)

    def kernel(first_ref, owner_ref, active_ref, weight_ref, hydraulic_ref, road_ref,
               conductance_ref, fraction_ref, transpiration_ref, potential_ref,
               exposed_ref, vegetation_ref, depth_ref, flux_ref, numerator_ref, total_ref):
        index = pl.program_id(0) * block + jnp.arange(block)
        column, level = index // levels, index % levels
        valid = column < columns
        start = plt.load(first_ref.at[column], mask=valid, other=patches)
        rural = plt.load(hydraulic_ref.at[column], mask=valid, other=False)
        urban = plt.load(road_ref.at[column], mask=valid, other=False)
        potential = plt.load(potential_ref.at[column, level], mask=valid, other=0.)
        depth = plt.load(depth_ref.at[column, level], mask=valid, other=0.)
        gravity = _rn("mul", depth, jnp.full((block,), 1000., jnp.float64))

        def step(offset, carry):
            patch = start + offset
            in_bounds = valid & (patch < patches)
            owner = plt.load(owner_ref.at[patch], mask=in_bounds, other=-1)
            enabled = in_bounds & (owner == column)
            enabled &= plt.load(active_ref.at[patch], mask=enabled, other=False)
            w = plt.load(weight_ref.at[patch], mask=enabled, other=0.)
            exposed = plt.load(exposed_ref.at[patch], mask=enabled, other=0)
            use_rural = enabled & rural & (exposed > 0) & (w > 0.)
            vegetation = plt.load(vegetation_ref.at[patch, ROOT], mask=use_rural, other=0.)
            conductance = plt.load(conductance_ref.at[patch, level], mask=use_rural, other=0.)
            difference = _rn("add", _rn("add", potential, -vegetation), -gravity)
            contribution = _rn("mul", _rn("mul", conductance, difference), w)
            flux = jnp.where(use_rural, _rn("add", carry[0], contribution), carry[0])
            use_road = enabled & urban
            transpiration = plt.load(transpiration_ref.at[patch], mask=use_road, other=0.)
            fraction = plt.load(fraction_ref.at[patch, level], mask=use_road, other=0.)
            contribution = _rn("mul", _rn("mul", fraction, transpiration), w)
            numerator = jnp.where(use_road, _rn("add", carry[1], contribution), carry[1])
            total = jnp.where(
                use_road, _rn("add", carry[2], _rn("mul", transpiration, w)), carry[2],
            )
            return flux, numerator, total

        zero = jnp.zeros((block,), jnp.float64)
        flux, numerator, total = lax.fori_loop(0, limit, step, (zero, zero, zero))
        plt.store(flux_ref.at[column, level], flux, mask=valid)
        plt.store(numerator_ref.at[column, level], numerator, mask=valid)
        plt.store(total_ref.at[column], total, mask=valid & (level == 0))

    shape = jax.ShapeDtypeStruct((columns, levels), jnp.float64)
    return pl.pallas_call(
        kernel, out_shape=(shape, shape, jax.ShapeDtypeStruct((columns,), jnp.float64)),
        grid=(pl.cdiv(columns * levels, block),),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
    )(first, owners, active, weight, hydraulic, road,
      inputs.soil_root_conductance_patch_level, inputs.root_fraction_patch_level,
      inputs.vegetation_transpiration_patch, inputs.soil_water_potential_column_level,
      inputs.exposed_vegetation_flag_patch, inputs.vegetation_water_potential_patch_segment,
      inputs.soil_node_depth_column_level)


@accumulate.defjvp
def _accumulate_jvp(limit, primals, tangents):
    from clm_jax.physics.soil_water_plant_sink import _accumulate_patch_sinks_reference

    return jax.jvp(
        lambda *args: _accumulate_patch_sinks_reference(*args, limit),
        primals, tangents,
    )
