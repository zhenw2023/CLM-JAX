
from functools import lru_cache, partial

import jax

from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch
from clm_jax.physics.soil_temperature_step import (
    SoilTemperatureRuntimeColumnResult,
    _soil_temperature_column_boundary,
    _soil_temperature_column_transition,
    _soil_temperature_runtime_column_reference,
)
from clm_jax.physics.solver_fma import native_exp_residual


@lru_cache(maxsize=8)
def _column_step(time_step_seconds):
    def column(inputs, ground, building, landunit_type):
        arguments = jax.tree.map(lambda value: value[None],
                                 (inputs, ground, building, landunit_type))
        token = native_exp_residual.set(True)
        try:
            complete = _soil_temperature_column_transition(
                *arguments, time_step_seconds=time_step_seconds,
            )
        finally:
            native_exp_residual.reset(token)
        result = SoilTemperatureRuntimeColumnResult(**{
            name: getattr(complete, name) for name in SoilTemperatureRuntimeColumnResult._fields
        })
        return jax.tree.map(lambda value: value[0], result)

    return column


@partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def soil_temperature_columns(inputs, time_step_seconds=1800., maximum_patches_per_column=79):

    ground, building, landunit_type = _soil_temperature_column_boundary(
        inputs, maximum_patches_per_column,
    )
    selected = inputs._replace(
        topology=None, urban_columns=None, ground_heat_flux=None,
        ground_heat_flux_patch_state=None, building_temperature_template=None,
    )
    columns = fused_native_batch(
        _column_step(time_step_seconds), selected, ground._replace(patch_state=None),
        building, landunit_type, block_size=32, hardware_rounding=True,
        fold_literals=True, direct_io=True, guard_branches=True,
        preserve_literal_barriers=True,
    )
    return columns._replace(ground_heat_flux=ground)


@partial(soil_temperature_columns.defjvp, symbolic_zeros=True)
def _derivative(time_step_seconds, maximum_patches_per_column, primals, tangents):
    def reference(inputs):
        return _soil_temperature_runtime_column_reference(
            inputs, time_step_seconds=time_step_seconds,
            maximum_patches_per_column=maximum_patches_per_column,
        )

    from clm_jax.compilation import _symbolic_zero_jvp
    return _symbolic_zero_jvp(reference, primals, tangents)
