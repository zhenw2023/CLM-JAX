
import jax

from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch


def _point(liquid, aquifer, minimum):
    from clm_jax.physics.soil_water_movement import _flexible_cn_liquid_cleanup

    return _flexible_cn_liquid_cleanup(liquid, aquifer, minimum)


def _reference(liquid, aquifer, minimum):
    return jax.vmap(_point, in_axes=(0, 0, None))(liquid, aquifer, minimum)


@jax.custom_jvp
def liquid_cleanup_columns(liquid, aquifer, minimum):
    if jax.default_backend() != "gpu" or not liquid.shape[0]:
        return _reference(liquid, aquifer, minimum)
    return fused_native_batch(
        _point, liquid, aquifer, minimum,
        shared_argnums=(2,), block_size=64,
        hardware_rounding=True, direct_io=True,
    )


@liquid_cleanup_columns.defjvp
def _derivative(primals, tangents):
    return liquid_cleanup_columns(*primals), jax.jvp(_reference, primals, tangents)[1]
