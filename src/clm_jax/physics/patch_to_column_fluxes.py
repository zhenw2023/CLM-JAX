
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp

from clm_jax.io.restart import LANDUNIT_LAKE, SubgridTopology
from clm_jax.physics.subgrid import patch_to_column_average

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase


class PatchToColumnFluxParameters(NamedTuple):

    maximum_patches_per_column: int = 79

    @classmethod
    def from_case(cls, case: CimeCase) -> PatchToColumnFluxParameters:

        from clm_jax.io.namelist import read_case_namelist_group  

        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        return cls(maximum_patches_per_column=int(general["maxpatch_pft"]))


DEFAULT_PATCH_TO_COLUMN_FLUX_PARAMETERS = PatchToColumnFluxParameters()


class PatchToColumnFluxInputs(NamedTuple):

    snow_evaporation_patch: jnp.ndarray
    bare_soil_evaporation_patch: jnp.ndarray
    surface_water_evaporation_patch: jnp.ndarray
    soil_evaporation_patch: jnp.ndarray
    total_evaporation_patch: jnp.ndarray
    ground_rainfall_patch: jnp.ndarray
    ground_snowfall_patch: jnp.ndarray
    vegetation_transpiration_patch: jnp.ndarray
    ground_liquid_evaporation_patch: jnp.ndarray
    ground_precipitation_patch: jnp.ndarray
    ground_dew_patch: jnp.ndarray
    snow_sublimation_patch: jnp.ndarray
    snow_dew_patch: jnp.ndarray


class PatchToColumnFluxState(NamedTuple):

    snow_evaporation_column: jnp.ndarray
    bare_soil_evaporation_column: jnp.ndarray
    surface_water_evaporation_column: jnp.ndarray
    soil_evaporation_column: jnp.ndarray
    total_evaporation_column: jnp.ndarray
    ground_rainfall_column: jnp.ndarray
    ground_snowfall_column: jnp.ndarray
    vegetation_transpiration_column: jnp.ndarray
    ground_liquid_evaporation_column: jnp.ndarray
    ground_precipitation_column: jnp.ndarray
    ground_dew_column: jnp.ndarray
    snow_sublimation_column: jnp.ndarray
    snow_dew_column: jnp.ndarray


class PatchToColumnFluxDiagnostics(NamedTuple):

    soil_evaporation_nonlake_column: jnp.ndarray


class PatchToColumnFluxResult(NamedTuple):
    state: PatchToColumnFluxState
    diagnostics: PatchToColumnFluxDiagnostics


def _validate_shapes(
    topology: SubgridTopology,
    inputs: PatchToColumnFluxInputs,
    state: PatchToColumnFluxState,
    parameters: PatchToColumnFluxParameters,
) -> None:
    patch_shape = topology.patch.column.shape
    column_shape = topology.column.gridcell.shape
    if len(patch_shape) != 1 or len(column_shape) != 1:
        raise ValueError("patch and column topology must be vectors")
    if parameters.maximum_patches_per_column < 0:
        raise ValueError("maximum_patches_per_column must be nonnegative")
    for name in PatchToColumnFluxInputs._fields:
        if getattr(inputs, name).shape != patch_shape:
            raise ValueError(f"{name} must have shape {patch_shape}")
    for name in PatchToColumnFluxState._fields:
        if getattr(state, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")


def aggregate_patch_fluxes_to_columns(
    topology: SubgridTopology,
    inputs: PatchToColumnFluxInputs,
    state: PatchToColumnFluxState,
    *,
    parameters: PatchToColumnFluxParameters = DEFAULT_PATCH_TO_COLUMN_FLUX_PARAMETERS,
) -> PatchToColumnFluxResult:

    _validate_shapes(topology, inputs, state, parameters)
    column_active = jnp.asarray(topology.column.active, dtype=jnp.bool_)
    nonlake_column = column_active & (
        jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
        != jnp.int32(LANDUNIT_LAKE)
    )
    patch_column = jnp.asarray(topology.patch.column, dtype=jnp.int32)
    patch_weight = jnp.asarray(topology.patch.weight_on_column, dtype=jnp.float64)
    patch_active = jnp.asarray(topology.patch.active, dtype=jnp.bool_)

    def aggregate(values: jnp.ndarray, update: jnp.ndarray) -> jnp.ndarray:
        return patch_to_column_average(
            values,
            patch_column,
            patch_weight,
            patch_active,
            update,
            maximum_patches_per_column=parameters.maximum_patches_per_column,
        )

    def update_nonlake(values: jnp.ndarray, previous: jnp.ndarray) -> jnp.ndarray:
        averaged = aggregate(values, nonlake_column)
        return jnp.where(nonlake_column, averaged, previous)

    snow_evaporation = update_nonlake(
        inputs.snow_evaporation_patch,
        state.snow_evaporation_column,
    )
    bare_soil_evaporation = update_nonlake(
        inputs.bare_soil_evaporation_patch,
        state.bare_soil_evaporation_column,
    )
    surface_water_evaporation = update_nonlake(
        inputs.surface_water_evaporation_patch,
        state.surface_water_evaporation_column,
    )

    soil_evaporation_nonlake = update_nonlake(
        inputs.soil_evaporation_patch,
        state.soil_evaporation_column,
    )
    soil_evaporation_all = aggregate(inputs.soil_evaporation_patch, column_active)
    soil_evaporation = jnp.where(
        column_active,
        soil_evaporation_all,
        soil_evaporation_nonlake,
    )

    total_evaporation = update_nonlake(
        inputs.total_evaporation_patch,
        state.total_evaporation_column,
    )
    ground_rainfall = update_nonlake(
        inputs.ground_rainfall_patch,
        state.ground_rainfall_column,
    )
    ground_snowfall = update_nonlake(
        inputs.ground_snowfall_patch,
        state.ground_snowfall_column,
    )
    vegetation_transpiration = update_nonlake(
        inputs.vegetation_transpiration_patch,
        state.vegetation_transpiration_column,
    )
    ground_liquid_evaporation = update_nonlake(
        inputs.ground_liquid_evaporation_patch,
        state.ground_liquid_evaporation_column,
    )
    ground_precipitation = update_nonlake(
        inputs.ground_precipitation_patch,
        state.ground_precipitation_column,
    )
    ground_dew = update_nonlake(
        inputs.ground_dew_patch,
        state.ground_dew_column,
    )
    snow_sublimation = update_nonlake(
        inputs.snow_sublimation_patch,
        state.snow_sublimation_column,
    )
    snow_dew = update_nonlake(
        inputs.snow_dew_patch,
        state.snow_dew_column,
    )

    return PatchToColumnFluxResult(
        state=PatchToColumnFluxState(
            snow_evaporation_column=snow_evaporation,
            bare_soil_evaporation_column=bare_soil_evaporation,
            surface_water_evaporation_column=surface_water_evaporation,
            soil_evaporation_column=soil_evaporation,
            total_evaporation_column=total_evaporation,
            ground_rainfall_column=ground_rainfall,
            ground_snowfall_column=ground_snowfall,
            vegetation_transpiration_column=vegetation_transpiration,
            ground_liquid_evaporation_column=ground_liquid_evaporation,
            ground_precipitation_column=ground_precipitation,
            ground_dew_column=ground_dew,
            snow_sublimation_column=snow_sublimation,
            snow_dew_column=snow_dew,
        ),
        diagnostics=PatchToColumnFluxDiagnostics(
            soil_evaporation_nonlake_column=soil_evaporation_nonlake,
        ),
    )
