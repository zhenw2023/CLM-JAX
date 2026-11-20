
import jax

from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch


def _tiled_recurrence(reference):
    def landunit(*arguments):
        expanded = jax.tree.map(lambda value: value[None], arguments)
        result = reference(*expanded)
        return jax.tree.map(lambda value: value[0], result)

    @jax.custom_jvp
    def execute(*arguments):
        return fused_native_batch(
            landunit, *arguments, block_size=32, fold_literals=True, direct_io=True,
        )

    @execute.defjvp
    def derivative(primals, tangents):
        return jax.jvp(reference, primals, tangents)

    return execute


@_tiled_recurrence
def shortwave_reflections(*arguments):
    from clm_jax.physics.urban_albedo import _iterate_reflections_reference

    return _iterate_reflections_reference(*arguments)


@_tiled_recurrence
def longwave_reflections(*arguments):
    from clm_jax.physics.urban_radiation import _iterate_canyon_longwave_reference

    return _iterate_canyon_longwave_reference(*arguments)
