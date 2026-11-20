
from functools import lru_cache

import jax
import jax.numpy as jnp

from clm_jax.physics.numerics import fortran_hardware_add as add
from clm_jax.physics.numerics import fortran_hardware_multiply as multiply
from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch


@lru_cache(maxsize=16)
def _cascade_kernel(element, donors, receivers):
    def carbon(source, phenology, transfer, respiration, active, time_step):
        pools = [source[i] for i in range(7)]
        for pool in range(3):
            pools[pool] = jnp.where(
                active, multiply(phenology[pool], time_step), pools[pool]
            )
        pools[3] = jnp.where(active, jnp.float64(0), pools[3])
        for transition, donor in enumerate(donors):
            loss = add(respiration[transition], transfer[transition])
            candidate = add(pools[donor], -multiply(loss, time_step))
            pools[donor] = jnp.where(active, candidate, pools[donor])
        for transition, receiver in enumerate(receivers):
            if receiver >= 0:
                candidate = add(pools[receiver], multiply(transfer[transition], time_step))
                pools[receiver] = jnp.where(active, candidate, pools[receiver])
        return jnp.stack(pools)

    def nitrogen(source, transfer, mineral_flux, active, time_step):
        pools = [source[i] for i in range(7)]
        for transition, donor in enumerate(donors):
            candidate = add(pools[donor], -multiply(transfer[transition], time_step))
            pools[donor] = jnp.where(active, candidate, pools[donor])
        for transition, (donor, receiver) in enumerate(zip(donors, receivers, strict=True)):
            if receiver >= 0:
                flux = add(transfer[transition], mineral_flux[transition])
                candidate = add(pools[receiver], multiply(flux, time_step))
                pools[receiver] = jnp.where(active, candidate, pools[receiver])
            else:
                candidate = add(pools[donor], -multiply(mineral_flux[transition], time_step))
                pools[donor] = jnp.where(active, candidate, pools[donor])
        return jnp.stack(pools)

    point = carbon if element == 'carbon' else nitrogen
    shared = 5 if element == 'carbon' else 4
    reference = jax.vmap(point, in_axes=(*([0] * shared), None))

    @jax.custom_jvp
    def execute(*arguments):
        return fused_native_batch(
            point, *arguments, shared_argnums=(shared,), block_size=128,
            fold_literals=True, direct_io=True,
        )

    @execute.defjvp
    def derivative(primals, tangents):
        return execute(*primals), jax.jvp(reference, primals, tangents)[1]

    return execute


def carbon_sources(source, phenology, transfer, respiration, active, time_step,
                   donors, receivers):
    kernel = _cascade_kernel('carbon', tuple(donors), tuple(receivers))
    return kernel(
        source.reshape(-1, 7), tuple(field.reshape(-1) for field in phenology),
        transfer.reshape(-1, 10), respiration.reshape(-1, 10),
        active.reshape(-1), time_step,
    ).reshape(source.shape)


def nitrogen_sources(source, transfer, mineral_flux, active, time_step, donors, receivers):
    kernel = _cascade_kernel('nitrogen', tuple(donors), tuple(receivers))
    return kernel(
        source.reshape(-1, 7), transfer.reshape(-1, 10), mineral_flux.reshape(-1, 10),
        active.reshape(-1), time_step,
    ).reshape(source.shape)
