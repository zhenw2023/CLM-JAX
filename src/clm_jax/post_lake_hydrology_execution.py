
from __future__ import annotations

from collections.abc import Callable
from functools import cache

import jax
import jax.numpy as jnp

from clm_jax.compilation import jit_unique, process_lifetime_jit
from clm_jax.execution_observer import stage_barrier
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.aerosol import snow_aerosol_masses
from clm_jax.physics.lake_hydrology_driver import LakeHydrologyDriverResult
from clm_jax.physics.physical_driver import (
    PhysicalDriverResult,
    PhysicalDriverRuntimeResult,
)
from clm_jax.physics.post_hydrology_no_drainage_driver import (
    PostHydrologyNoDrainageResult,
)
from clm_jax.physics.post_lake_hydrology_driver import (
    PostLakeHydrologyResult,
    assemble_lake_aerosol_mass_boundary,
    assemble_lake_snow_age_boundary,
    urban_snow_fraction,
    post_lake_hydrology_step,
)
from clm_jax.physics.snow_age_grain import (
    SnowAgeGrainParameters,
    SnowAgingLookupTable,
    snow_age_grain,
)
from clm_jax.physics.snow_water import build_snow_column_masks

_USE_FUSED_GPU_POST_LAKE = True


@cache
def _post_lake_program(parameters):
    return jit_unique(
        lambda *arguments: post_lake_hydrology_step(*arguments, parameters=parameters),
        forward_inputs=True,
    )


def _compiled_post_lake(topology, physical, post_hydrology, lake_hydrology,
                        lake_column_index, lookup, parameters):
    return _post_lake_program(parameters)(
        topology, physical, post_hydrology, lake_hydrology, lake_column_index, lookup,
    )


class PostLakeHydrologyExecutor:

    def __init__(
        self,
        *,
        jit: bool = True,
        stage_completed: Callable[[str], None] | None = None,
    ) -> None:
        self.stage_completed = stage_completed
        self.fused_gpu = jit and _USE_FUSED_GPU_POST_LAKE and jax.default_backend() == "gpu"
        if jit:
            self.aerosol_step = process_lifetime_jit(snow_aerosol_masses)
            self.age_step = process_lifetime_jit(
                snow_age_grain,
                static_argnames=("parameters",),
            )
        else:
            self.aerosol_step = snow_aerosol_masses
            self.age_step = snow_age_grain

    def _completed(self, name: str, result):
        if self.stage_completed is not None:
            stage_barrier(result)
            self.stage_completed(name)
        return result

    def __call__(
        self,
        topology: SubgridTopology,
        physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
        post_hydrology: PostHydrologyNoDrainageResult,
        lake_hydrology: LakeHydrologyDriverResult,
        lake_column_index: jnp.ndarray,
        lookup: SnowAgingLookupTable,
        parameters: SnowAgeGrainParameters,
    ) -> PostLakeHydrologyResult:
        if self.fused_gpu:
            result = _compiled_post_lake(
                topology, physical, post_hydrology, lake_hydrology,
                lake_column_index, lookup, parameters,
            )
            for name, fields in (
                ("stage385_lake_aerosol_masses", result.lake_aerosol_mass),
                ("stage390_lake_snow_age", result.lake_snow_age),
                ("stage395_urban_snow_fraction", result.snow_fraction_column),
                ("stage400_nonlake_snow_age", result.nonlake_snow_age),
            ):
                self._completed(name, fields)
            return result
        aerosol_inputs, aerosol_state = assemble_lake_aerosol_mass_boundary(
            topology,
            post_hydrology,
            lake_hydrology,
        )
        lake_aerosol = self._completed(
            "stage385_lake_aerosol_masses",
            self.aerosol_step(aerosol_inputs, aerosol_state),
        )
        age_inputs, age_state = assemble_lake_snow_age_boundary(
            topology,
            physical,
            lake_hydrology,
            lake_column_index,
            lake_aerosol,
        )
        lake_age = self._completed(
            "stage390_lake_snow_age",
            self.age_step(
                lookup,
                age_inputs,
                age_state,
                parameters=parameters,
            ),
        )
        snow_fraction = self._completed(
            "stage395_urban_snow_fraction",
            urban_snow_fraction(topology, lake_hydrology),
        )
        nonlake_masks = build_snow_column_masks(
            topology,
            age_inputs.snow_layer_count_column,
        )
        nonlake_age = self._completed(
            "stage400_nonlake_snow_age",
            self.age_step(
                lookup,
                age_inputs._replace(
                    snow_column=nonlake_masks.snow_column,
                    no_snow_column=nonlake_masks.no_snow_column,
                ),
                lake_age.state,
                parameters=parameters,
            ),
        )
        return PostLakeHydrologyResult(
            lake_aerosol_mass=lake_aerosol,
            lake_snow_age=lake_age,
            snow_fraction_column=snow_fraction,
            nonlake_snow_age=nonlake_age,
        )


__all__ = ["PostLakeHydrologyExecutor"]
