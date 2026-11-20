
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from clm_jax.io.restart import LANDUNIT_URBAN_MAX, LANDUNIT_URBAN_MIN, SubgridTopology
from clm_jax.physics.aerosol import (
    SnowAerosolMassInputs,
    SnowAerosolMassResult,
    SnowAerosolMassState,
    snow_aerosol_masses,
)
from clm_jax.physics.lake_hydrology_driver import LakeHydrologyDriverResult
from clm_jax.physics.physical_driver import (
    PhysicalDriverResult,
    PhysicalDriverRuntimeResult,
)
from clm_jax.physics.post_hydrology_no_drainage_driver import (
    PostHydrologyNoDrainageResult,
)
from clm_jax.physics.snow_age_grain import (
    DEFAULT_SNOW_AGE_GRAIN_PARAMETERS,
    SnowAgeGrainInputs,
    SnowAgeGrainParameters,
    SnowAgeGrainResult,
    SnowAgeGrainState,
    SnowAgingLookupTable,
    snow_age_grain,
)
from clm_jax.physics.snow_water import NUMBER_OF_SNOW_LEVELS, build_snow_column_masks


class PostLakeHydrologyResult(NamedTuple):

    lake_aerosol_mass: SnowAerosolMassResult
    lake_snow_age: SnowAgeGrainResult
    snow_fraction_column: jnp.ndarray
    nonlake_snow_age: SnowAgeGrainResult


def assemble_lake_aerosol_mass_boundary(
    topology: SubgridTopology,
    post_hydrology: PostHydrologyNoDrainageResult,
    lake_hydrology: LakeHydrologyDriverResult,
) -> tuple[SnowAerosolMassInputs, SnowAerosolMassState]:

    lake = lake_hydrology.hydrology.state
    layer_count = lake.geometry.snow_layer_count_column
    masks = build_snow_column_masks(topology, layer_count, lake_only=True)
    retained_mass = post_hydrology.aerosol_mass.state
    return (
        SnowAerosolMassInputs(
            snow_column=masks.snow_column,
            no_snow_column=masks.no_snow_column,
            snow_layer_count_column=layer_count,
            snow_ice_column_level=(
                lake.water.soil_snow_ice_column_level[:, :NUMBER_OF_SNOW_LEVELS]
            ),
            snow_liquid_column_level=(
                lake.water.soil_snow_liquid_column_level[:, :NUMBER_OF_SNOW_LEVELS]
            ),
        ),
        retained_mass._replace(
            top_snow_water_column=lake.water.top_snow_water_column,
            snow_grain_radius_column_level=(
                lake.water.snow_grain_radius_column_level
            ),
            aerosol=lake.aerosol.mass,
        ),
    )


def assemble_lake_snow_age_boundary(
    topology: SubgridTopology,
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    lake_hydrology: LakeHydrologyDriverResult,
    lake_column_index: jnp.ndarray,
    lake_aerosol: SnowAerosolMassResult,
) -> tuple[SnowAgeGrainInputs, SnowAgeGrainState]:

    lake = lake_hydrology.hydrology.state
    layer_count = lake.geometry.snow_layer_count_column
    masks = build_snow_column_masks(topology, layer_count, lake_only=True)
    freezing_flux = jnp.asarray(
        physical.soil_temperature.temperature.soil_snow_phase_change.snow_layer_freeze_flux,
        dtype=jnp.float64,
    )
    mapped_column = jnp.asarray(lake_column_index, dtype=jnp.int32)
    compact_freezing = jnp.asarray(
        physical.lake_temperature.temperature.water.snow_layer_freeze,
        dtype=jnp.float64,
    )
    if compact_freezing.shape != (
        mapped_column.shape[0],
        NUMBER_OF_SNOW_LEVELS,
    ):
        raise ValueError("lake freezing flux and native column map disagree")
    freezing_flux = freezing_flux.at[mapped_column].set(compact_freezing)

    return (
        SnowAgeGrainInputs(
            snow_column=masks.snow_column,
            no_snow_column=masks.no_snow_column,
            snow_layer_count_column=layer_count,
            soil_snow_thickness_column_level=(
                lake.geometry.soil_snow_thickness_column_level
            ),
            effective_snow_fraction_column=(
                lake.water.effective_snow_fraction_column
            ),
            snow_water_equivalent_column=lake.water.snow_water_equivalent_column,
            soil_snow_liquid_water_column_level=(
                lake.water.soil_snow_liquid_column_level
            ),
            soil_snow_ice_water_column_level=(
                lake.water.soil_snow_ice_column_level
            ),
            ground_snowfall_flux_column=(
                lake.water_flux.ground_snowfall_flux_column
            ),
            snow_layer_freezing_flux_column_level=freezing_flux,
            soil_snow_temperature_column_level=(
                lake.temperature.soil_snow_temperature_column_level
            ),
            atmospheric_temperature_column=(
                physical.atmospheric.column_forcing.temperature
            ),
        ),
        SnowAgeGrainState(
            snow_grain_radius_column_level=(
                lake_aerosol.state.snow_grain_radius_column_level
            ),
            top_snow_grain_radius_column=(
                lake.water.top_snow_grain_radius_column
            ),
            top_snow_liquid_fraction_column=(
                lake.water.top_snow_liquid_fraction_column
            ),
            top_snow_temperature_column=(
                lake.temperature.top_snow_temperature_column
            ),
            top_snow_temperature_gradient_column=(
                lake.temperature.top_temperature_gradient_column
            ),
        ),
    )


def urban_snow_fraction(
    topology: SubgridTopology,
    lake_hydrology: LakeHydrologyDriverResult,
) -> jnp.ndarray:

    lake = lake_hydrology.hydrology.state
    landunit_type = jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
    urban = jnp.asarray(topology.column.active, dtype=jnp.bool_) & (
        (landunit_type >= jnp.int32(LANDUNIT_URBAN_MIN))
        & (landunit_type <= jnp.int32(LANDUNIT_URBAN_MAX))
    )
    depth, scale = lax.optimization_barrier(
        (lake.water.snow_depth_column, jnp.float64(0.05)),
    )
    urban_fraction = jnp.minimum(
        depth / scale,
        jnp.float64(1.0),
    )
    return jnp.where(
        urban,
        urban_fraction,
        lake.water.snow_fraction_column,
    )


def post_lake_hydrology_step(
    topology: SubgridTopology,
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    post_hydrology: PostHydrologyNoDrainageResult,
    lake_hydrology: LakeHydrologyDriverResult,
    lake_column_index: jnp.ndarray,
    lookup: SnowAgingLookupTable,
    parameters: SnowAgeGrainParameters = DEFAULT_SNOW_AGE_GRAIN_PARAMETERS,
) -> PostLakeHydrologyResult:

    aerosol_inputs, aerosol_state = assemble_lake_aerosol_mass_boundary(
        topology,
        post_hydrology,
        lake_hydrology,
    )
    lake_aerosol = snow_aerosol_masses(aerosol_inputs, aerosol_state)
    age_inputs, age_state = assemble_lake_snow_age_boundary(
        topology,
        physical,
        lake_hydrology,
        lake_column_index,
        lake_aerosol,
    )
    lake_age = snow_age_grain(
        lookup,
        age_inputs,
        age_state,
        parameters=parameters,
    )

    snow_fraction = urban_snow_fraction(topology, lake_hydrology)

    nonlake_masks = build_snow_column_masks(
        topology,
        age_inputs.snow_layer_count_column,
    )
    nonlake_age = snow_age_grain(
        lookup,
        age_inputs._replace(
            snow_column=nonlake_masks.snow_column,
            no_snow_column=nonlake_masks.no_snow_column,
        ),
        lake_age.state,
        parameters=parameters,
    )
    return PostLakeHydrologyResult(
        lake_aerosol_mass=lake_aerosol,
        lake_snow_age=lake_age,
        snow_fraction_column=snow_fraction,
        nonlake_snow_age=nonlake_age,
    )


__all__ = [
    "PostLakeHydrologyResult",
    "assemble_lake_aerosol_mass_boundary",
    "assemble_lake_snow_age_boundary",
    "post_lake_hydrology_step",
    "urban_snow_fraction",
]
