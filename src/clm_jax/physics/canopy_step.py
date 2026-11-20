
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp

from clm_jax.io.restart import LANDUNIT_LAKE, SubgridTopology
from clm_jax.physics.aerosol import SnowAerosolState, reset_new_snow_aerosols
from clm_jax.physics.canopy_hydrology import (
    URBAN_SHADE_WALL_COLUMN,
    URBAN_SUN_WALL_COLUMN,
    CanopyHydrologyParameters,
    CanopyPrecipitationFluxes,
    CanopyWaterState,
    CanopyWetFractions,
    canopy_interception,
    canopy_wet_fractions,
    surface_water_fraction,
)
from clm_jax.physics.snow import (
    NewSnowDensityParameters,
    SnowAccumulationParameters,
    SnowColumnState,
    accumulate_snow,
    new_snow_bulk_density,
)
from clm_jax.physics.subgrid import patch_to_column_average

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase


class CanopyHydrologyForcing(NamedTuple):
    rain_column: jnp.ndarray
    snow_column: jnp.ndarray
    temperature_column: jnp.ndarray
    wind_gridcell: jnp.ndarray
    flood_gridcell: jnp.ndarray
    irrigation_patch: jnp.ndarray


class CanopyStructure(NamedTuple):
    vegetation_exposed_patch: jnp.ndarray
    maximum_dew_patch: jnp.ndarray
    exposed_leaf_area_index_patch: jnp.ndarray
    exposed_stem_area_index_patch: jnp.ndarray
    microtopography_sigma_column: jnp.ndarray
    ground_temperature_column: jnp.ndarray
    snow_cover_shape_column: jnp.ndarray


class CanopyHydrologyState(NamedTuple):
    canopy_water: CanopyWaterState
    canopy_wet_fractions: CanopyWetFractions
    snow: SnowColumnState
    surface_water_column: jnp.ndarray
    top_soil_liquid_column: jnp.ndarray
    surface_water_fraction_column: jnp.ndarray
    surface_water_fraction_without_snow_column: jnp.ndarray
    snow_aerosol: SnowAerosolState


class CanopyHydrologyStepParameters(NamedTuple):
    canopy: CanopyHydrologyParameters
    new_snow_density: NewSnowDensityParameters
    snow_accumulation: SnowAccumulationParameters
    maximum_patches_per_column: int = 79

    @classmethod
    def from_case(cls, case: CimeCase) -> CanopyHydrologyStepParameters:

        from clm_jax.io.namelist import read_case_namelist_group  

        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        return cls(
            canopy=CanopyHydrologyParameters.from_case(case),
            new_snow_density=NewSnowDensityParameters.from_case(case),
            snow_accumulation=SnowAccumulationParameters.from_case(case),
            maximum_patches_per_column=int(general["maxpatch_pft"]),
        )


class CanopyHydrologyStepFluxes(NamedTuple):
    patch_precipitation: CanopyPrecipitationFluxes
    ground_snow_column: jnp.ndarray
    flood_column: jnp.ndarray
    snow_on_surface_water_column: jnp.ndarray
    new_snow_density_column: jnp.ndarray
    new_snow_column: jnp.ndarray
    previous_step_snowmelt_column: jnp.ndarray
    snow_depth_rate_column: jnp.ndarray


class CanopyHydrologyStepResult(NamedTuple):
    state: CanopyHydrologyState
    fluxes: CanopyHydrologyStepFluxes
    previous_snow_layer_water: jnp.ndarray
    new_snow_layer_created: jnp.ndarray


def _select_columns(
    update_mask: jnp.ndarray,
    candidate: jnp.ndarray,
    previous: jnp.ndarray,
) -> jnp.ndarray:
    candidate = jnp.asarray(candidate)
    previous = jnp.asarray(previous)
    broadcast_shape = (update_mask.shape[0],) + (1,) * (candidate.ndim - 1)
    return jnp.where(update_mask.reshape(broadcast_shape), candidate, previous)


def _safe_snow_state(
    state: SnowColumnState,
    update_mask: jnp.ndarray,
) -> SnowColumnState:
    def safe(value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.asarray(value)
        broadcast_shape = (update_mask.shape[0],) + (1,) * (value.ndim - 1)
        return jnp.where(update_mask.reshape(broadcast_shape), value, jnp.zeros_like(value))

    return SnowColumnState(*(safe(value) for value in state))


def canopy_hydrology_step(
    topology: SubgridTopology,
    forcing: CanopyHydrologyForcing,
    structure: CanopyStructure,
    state: CanopyHydrologyState,
    previous_snow_drainage_flux_column: jnp.ndarray,
    *,
    step_seconds: int | float,
    parameters: CanopyHydrologyStepParameters,
) -> CanopyHydrologyStepResult:

    patch = topology.patch
    column = topology.column
    column_update = jnp.asarray(column.active, dtype=jnp.bool_) & (
        jnp.asarray(column.landunit_type, dtype=jnp.int32) != LANDUNIT_LAKE
    )
    patch_update = jnp.asarray(patch.active, dtype=jnp.bool_) & column_update[
        patch.column
    ]

    patch_rain = jnp.asarray(forcing.rain_column, dtype=jnp.float64)[patch.column]
    patch_snow = jnp.asarray(forcing.snow_column, dtype=jnp.float64)[patch.column]
    patch_temperature = jnp.asarray(
        forcing.temperature_column,
        dtype=jnp.float64,
    )[patch.column]
    patch_wind = jnp.asarray(forcing.wind_gridcell, dtype=jnp.float64)[patch.gridcell]
    patch_zeros = jnp.zeros(patch_update.shape, dtype=jnp.float64)
    safe_canopy_state = CanopyWaterState(
        *(
            jnp.where(patch_update, jnp.asarray(value, dtype=jnp.float64), patch_zeros)
            for value in state.canopy_water
        )
    )
    interception_candidate = canopy_interception(
        landunit_type=patch.landunit_type,
        column_type=patch.column_type,
        vegetation_exposed=jnp.where(
            patch_update,
            structure.vegetation_exposed_patch,
            jnp.int32(0),
        ),
        rain=jnp.where(patch_update, patch_rain, patch_zeros),
        snow=jnp.where(patch_update, patch_snow, patch_zeros),
        temperature=jnp.where(patch_update, patch_temperature, jnp.float64(273.15)),
        wind_speed=jnp.where(patch_update, patch_wind, patch_zeros),
        maximum_dew=jnp.where(
            patch_update,
            structure.maximum_dew_patch,
            jnp.float64(1.0),
        ),
        exposed_leaf_area_index=jnp.where(
            patch_update,
            structure.exposed_leaf_area_index_patch,
            patch_zeros,
        ),
        exposed_stem_area_index=jnp.where(
            patch_update,
            structure.exposed_stem_area_index_patch,
            patch_zeros,
        ),
        irrigation=jnp.where(
            patch_update,
            forcing.irrigation_patch,
            patch_zeros,
        ),
        state=safe_canopy_state,
        step_seconds=step_seconds,
        parameters=parameters.canopy,
    )
    canopy_water = CanopyWaterState(
        *(
            jnp.where(patch_update, candidate, previous)
            for candidate, previous in zip(
                interception_candidate.state,
                state.canopy_water,
                strict=True,
            )
        )
    )
    patch_fluxes = CanopyPrecipitationFluxes(
        *(
            jnp.where(patch_update, value, patch_zeros)
            for value in interception_candidate.fluxes
        )
    )

    wet_candidate = canopy_wet_fractions(
        vegetation_exposed=jnp.where(
            patch_update,
            structure.vegetation_exposed_patch,
            jnp.int32(0),
        ),
        maximum_dew=jnp.where(
            patch_update,
            structure.maximum_dew_patch,
            jnp.float64(1.0),
        ),
        exposed_leaf_area_index=jnp.where(
            patch_update,
            structure.exposed_leaf_area_index_patch,
            patch_zeros,
        ),
        exposed_stem_area_index=jnp.where(
            patch_update,
            structure.exposed_stem_area_index_patch,
            patch_zeros,
        ),
        total_canopy_water=canopy_water.total_water,
        canopy_snow=canopy_water.snow,
        previous_snow_covered_fraction=state.canopy_wet_fractions.snow_covered,
        parameters=parameters.canopy,
    )
    wet_fractions = CanopyWetFractions(
        *(
            jnp.where(patch_update, candidate, previous)
            for candidate, previous in zip(
                wet_candidate,
                state.canopy_wet_fractions,
                strict=True,
            )
        )
    )

    ground_snow_column = patch_to_column_average(
        patch_values=patch_fluxes.ground_snow,
        patch_column=patch.column,
        patch_weight_on_column=patch.weight_on_column,
        patch_active=patch_update,
        column_update_mask=column_update,
        maximum_patches_per_column=parameters.maximum_patches_per_column,
    )
    wall = (column.column_type == URBAN_SUN_WALL_COLUMN) | (
        column.column_type == URBAN_SHADE_WALL_COLUMN
    )
    flood_column = jnp.where(
        column_update & ~wall,
        jnp.asarray(forcing.flood_gridcell, dtype=jnp.float64)[column.gridcell],
        jnp.float64(0.0),
    )
    column_wind = jnp.asarray(forcing.wind_gridcell, dtype=jnp.float64)[column.gridcell]
    density = new_snow_bulk_density(
        jnp.where(column_update, forcing.temperature_column, jnp.float64(273.15)),
        jnp.where(column_update, column_wind, jnp.float64(0.0)),
        parameters.new_snow_density,
    )
    safe_snow_state = _safe_snow_state(state.snow, column_update)
    snow_candidate = accumulate_snow(
        landunit_type=column.landunit_type,
        ground_snow_flux=jnp.where(
            column_update,
            ground_snow_column,
            jnp.float64(0.0),
        ),
        previous_snow_drainage_flux=jnp.where(
            column_update,
            previous_snow_drainage_flux_column,
            jnp.float64(0.0),
        ),
        new_snow_density=density,
        atmospheric_temperature=jnp.where(
            column_update,
            forcing.temperature_column,
            jnp.float64(273.15),
        ),
        ground_temperature=jnp.where(
            column_update,
            structure.ground_temperature_column,
            jnp.float64(273.15),
        ),
        snow_cover_shape=jnp.where(
            column_update,
            structure.snow_cover_shape_column,
            jnp.float64(1.0),
        ),
        state=safe_snow_state,
        step_seconds=step_seconds,
        parameters=parameters.snow_accumulation,
    )
    snow_state = SnowColumnState(
        *(
            _select_columns(column_update, candidate, previous)
            for candidate, previous in zip(
                snow_candidate.state,
                state.snow,
                strict=True,
            )
        )
    )
    new_layer = column_update & snow_candidate.new_layer_created
    aerosol = reset_new_snow_aerosols(state.snow_aerosol, new_layer)

    surface_candidate = surface_water_fraction(
        landunit_type=column.landunit_type,
        microtopography_sigma=jnp.where(
            column_update,
            structure.microtopography_sigma_column,
            jnp.float64(0.01),
        ),
        snow_water_equivalent=snow_state.water_equivalent,
        surface_water=jnp.where(
            column_update,
            state.surface_water_column,
            jnp.float64(0.0),
        ),
        top_soil_liquid=jnp.where(
            column_update,
            state.top_soil_liquid_column,
            jnp.float64(0.0),
        ),
        snow_fraction=snow_state.snow_fraction,
        effective_snow_fraction=snow_state.effective_snow_fraction,
        previous_surface_water_fraction_without_snow=jnp.where(
            column_update,
            state.surface_water_fraction_without_snow_column,
            jnp.float64(0.0),
        ),
    )
    snow_state = snow_state._replace(
        snow_fraction=_select_columns(
            column_update,
            surface_candidate.snow_fraction,
            state.snow.snow_fraction,
        ),
        effective_snow_fraction=_select_columns(
            column_update,
            surface_candidate.effective_snow_fraction,
            state.snow.effective_snow_fraction,
        ),
    )
    updated_state = CanopyHydrologyState(
        canopy_water=canopy_water,
        canopy_wet_fractions=wet_fractions,
        snow=snow_state,
        surface_water_column=_select_columns(
            column_update,
            surface_candidate.surface_water,
            state.surface_water_column,
        ),
        top_soil_liquid_column=_select_columns(
            column_update,
            surface_candidate.top_soil_liquid,
            state.top_soil_liquid_column,
        ),
        surface_water_fraction_column=_select_columns(
            column_update,
            surface_candidate.surface_water_fraction,
            state.surface_water_fraction_column,
        ),
        surface_water_fraction_without_snow_column=_select_columns(
            column_update,
            surface_candidate.surface_water_fraction_without_snow,
            state.surface_water_fraction_without_snow_column,
        ),
        snow_aerosol=aerosol,
    )

    return CanopyHydrologyStepResult(
        state=updated_state,
        fluxes=CanopyHydrologyStepFluxes(
            patch_precipitation=patch_fluxes,
            ground_snow_column=ground_snow_column,
            flood_column=flood_column,
            snow_on_surface_water_column=jnp.where(
                column_update,
                snow_candidate.snow_on_surface_water,
                jnp.float64(0.0),
            ),
            new_snow_density_column=jnp.where(
                column_update,
                density,
                jnp.float64(0.0),
            ),
            new_snow_column=jnp.where(
                column_update,
                snow_candidate.new_snow,
                jnp.float64(0.0),
            ),
            previous_step_snowmelt_column=jnp.where(
                column_update,
                snow_candidate.previous_step_snowmelt,
                jnp.float64(0.0),
            ),
            snow_depth_rate_column=jnp.where(
                column_update,
                snow_candidate.snow_depth_rate,
                jnp.float64(0.0),
            ),
        ),
        previous_snow_layer_water=_select_columns(
            column_update,
            snow_candidate.previous_layer_water,
            jnp.zeros_like(snow_candidate.previous_layer_water),
        ),
        new_snow_layer_created=new_layer,
    )
