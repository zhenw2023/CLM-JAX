
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from clm_jax.constants import CLM_SPECIAL_VALUE, FREEZING_TEMPERATURE_FRESHWATER
from clm_jax.io.restart import (
    COLUMN_URBAN_IMPERVIOUS_ROAD,
    COLUMN_URBAN_PERVIOUS_ROAD,
    COLUMN_URBAN_ROOF,
    COLUMN_URBAN_SHADE_WALL,
    COLUMN_URBAN_SUN_WALL,
    LANDUNIT_CROP,
    LANDUNIT_GLACIER,
    LANDUNIT_SOIL,
    SubgridTopology,
)
from clm_jax.physics.annual_flux_dribbler import (
    AnnualFluxDribblerResult,
    AnnualFluxDribblerState,
    annual_flux_dribbler_step,
)
from clm_jax.physics.numerics import fortran_add, fortran_divide, fortran_multiply
from clm_jax.physics.subgrid import column_to_gridcell_mean
from clm_jax.physics.total_water_and_heat import (
    TotalWaterAndHeatInputs,
    adjust_delta_heat_for_delta_liquid,
    compute_total_water_and_heat,
)


class DynamicHeatWaterContent(NamedTuple):
    liquid_water_mass_gridcell: jnp.ndarray
    ice_water_mass_gridcell: jnp.ndarray
    heat_content_gridcell: jnp.ndarray
    liquid_sensible_heat_gridcell: jnp.ndarray
    liquid_heat_capacity_gridcell: jnp.ndarray
    liquid_water_temperature_gridcell: jnp.ndarray


class DynamicBiogeophysicalDribblerState(NamedTuple):
    liquid_water: AnnualFluxDribblerState
    ice_water: AnnualFluxDribblerState
    energy: AnnualFluxDribblerState


class DynamicBiogeophysicalBalanceResult(NamedTuple):
    content_after: DynamicHeatWaterContent
    delta_liquid_water_gridcell: jnp.ndarray
    delta_ice_water_gridcell: jnp.ndarray
    delta_heat_gridcell: jnp.ndarray
    liquid_water_flux: AnnualFluxDribblerResult
    ice_water_flux: AnnualFluxDribblerResult
    energy_flux: AnnualFluxDribblerResult


class DynamicSubgridWeightDiagnostics(NamedTuple):

    percent_landunit_gridcell_type: jnp.ndarray
    percent_natural_pft_gridcell_type: jnp.ndarray
    percent_crop_pft_gridcell_type: jnp.ndarray
    percent_glacier_elevation_gridcell_class: jnp.ndarray


def dynamic_subgrid_weight_diagnostics(
    topology: SubgridTopology,
    *,
    number_of_landunit_types: int = 9,
    natural_pft_lower_bound: int = 0,
    number_of_natural_pfts: int = 15,
    crop_pft_lower_bound: int = 15,
    number_of_crop_pfts: int = 64,
    number_of_glacier_elevation_classes: int = 10,
) -> DynamicSubgridWeightDiagnostics:

    number_of_gridcells = topology.grid.longitude_degrees.shape[0]
    percent = jnp.float64(100.0)
    landunit = jnp.zeros((number_of_gridcells, number_of_landunit_types), dtype=jnp.float64)
    natural_pft = jnp.zeros((number_of_gridcells, number_of_natural_pfts), dtype=jnp.float64)
    crop_pft = jnp.zeros((number_of_gridcells, number_of_crop_pfts), dtype=jnp.float64)
    glacier = jnp.zeros(
        (number_of_gridcells, number_of_glacier_elevation_classes),
        dtype=jnp.float64,
    )

    def set_landunit(index: int, values: jnp.ndarray) -> jnp.ndarray:
        gridcell = topology.landunit.gridcell[index]
        landunit_type = topology.landunit.landunit_type[index] - jnp.int32(1)
        value = fortran_multiply(topology.landunit.weight_on_gridcell[index], percent)
        return values.at[gridcell, landunit_type].set(value)

    landunit = lax.fori_loop(
        0,
        topology.landunit.gridcell.shape[0],
        set_landunit,
        landunit,
    )

    def set_patch(
        index: int,
        values: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        natural_values, crop_values = values
        gridcell = topology.patch.gridcell[index]
        landunit_index = topology.patch.landunit[index]
        landunit_type = topology.landunit.landunit_type[landunit_index]
        vegetation_type = topology.patch.vegetation_type[index]
        value = fortran_multiply(topology.patch.weight_on_landunit[index], percent)
        natural_index = jnp.clip(
            vegetation_type - jnp.int32(natural_pft_lower_bound),
            jnp.int32(0),
            jnp.int32(number_of_natural_pfts - 1),
        )
        crop_index = jnp.clip(
            vegetation_type - jnp.int32(crop_pft_lower_bound),
            jnp.int32(0),
            jnp.int32(number_of_crop_pfts - 1),
        )
        natural_values = lax.cond(
            landunit_type == jnp.int32(LANDUNIT_SOIL),
            lambda array: array.at[gridcell, natural_index].set(value),
            lambda array: array,
            natural_values,
        )
        crop_values = lax.cond(
            landunit_type == jnp.int32(LANDUNIT_CROP),
            lambda array: array.at[gridcell, crop_index].set(value),
            lambda array: array,
            crop_values,
        )
        return natural_values, crop_values

    natural_pft, crop_pft = lax.fori_loop(
        0,
        topology.patch.gridcell.shape[0],
        set_patch,
        (natural_pft, crop_pft),
    )

    def set_glacier(index: int, values: jnp.ndarray) -> jnp.ndarray:
        gridcell = topology.column.gridcell[index]
        landunit_index = topology.column.landunit[index]
        landunit_type = topology.landunit.landunit_type[landunit_index]
        glacier_class = jnp.clip(
            topology.column.column_type[index] - jnp.int32(LANDUNIT_GLACIER * 100) - jnp.int32(1),
            jnp.int32(0),
            jnp.int32(number_of_glacier_elevation_classes - 1),
        )
        value = fortran_multiply(topology.column.weight_on_landunit[index], percent)
        return lax.cond(
            landunit_type == jnp.int32(LANDUNIT_GLACIER),
            lambda array: array.at[gridcell, glacier_class].set(value),
            lambda array: array,
            values,
        )

    glacier = lax.fori_loop(
        0,
        topology.column.gridcell.shape[0],
        set_glacier,
        glacier,
    )
    return DynamicSubgridWeightDiagnostics(
        percent_landunit_gridcell_type=landunit,
        percent_natural_pft_gridcell_type=natural_pft,
        percent_crop_pft_gridcell_type=crop_pft,
        percent_glacier_elevation_gridcell_class=glacier,
    )


def column_to_gridcell_urbanf(
    topology: SubgridTopology,
    column_values: jnp.ndarray,
    canyon_height_to_width_ratio_landunit: jnp.ndarray,
    *,
    maximum_columns_per_gridcell: int = 64,
    special_value: float = CLM_SPECIAL_VALUE,
) -> jnp.ndarray:

    values = jnp.asarray(column_values, dtype=jnp.float64)
    canyon_ratio = jnp.asarray(canyon_height_to_width_ratio_landunit, dtype=jnp.float64)
    column_shape = topology.column.gridcell.shape
    landunit_shape = topology.landunit.gridcell.shape
    if values.shape != column_shape:
        raise ValueError(f"column_values must have shape {column_shape}")
    if canyon_ratio.shape != landunit_shape:
        raise ValueError(
            "canyon_height_to_width_ratio_landunit must match landunit topology"
        )
    column_type = jnp.asarray(topology.column.column_type, dtype=jnp.int32)
    landunit = jnp.asarray(topology.column.landunit, dtype=jnp.int32)
    walls = (column_type == jnp.int32(COLUMN_URBAN_SUN_WALL)) | (
        column_type == jnp.int32(COLUMN_URBAN_SHADE_WALL)
    )
    roads = (column_type == jnp.int32(COLUMN_URBAN_PERVIOUS_ROAD)) | (
        column_type == jnp.int32(COLUMN_URBAN_IMPERVIOUS_ROAD)
    )
    roof = column_type == jnp.int32(COLUMN_URBAN_ROOF)
    wall_scale = fortran_multiply(jnp.float64(3.0), canyon_ratio[landunit])
    scale = jnp.where(
        walls,
        wall_scale,
        jnp.where(roads, jnp.float64(3.0), jnp.float64(1.0)),
    )
    scale = jnp.where(roof, jnp.float64(1.0), scale)
    scaled = fortran_multiply(values, scale)
    scaled = jnp.where(values == jnp.float64(special_value), values, scaled)
    return column_to_gridcell_mean(
        scaled,
        topology.column.gridcell,
        topology.column.weight_on_gridcell,
        topology.column.active,
        number_of_gridcells=topology.grid.longitude_degrees.shape[0],
        maximum_columns_per_gridcell=maximum_columns_per_gridcell,
        special_value=special_value,
    )


def dynamic_heat_water_content(
    topology: SubgridTopology,
    inputs: TotalWaterAndHeatInputs,
    canyon_height_to_width_ratio_landunit: jnp.ndarray,
    *,
    maximum_patches_per_column: int = 79,
    maximum_columns_per_gridcell: int = 64,
) -> DynamicHeatWaterContent:

    column = compute_total_water_and_heat(
        topology,
        inputs,
        maximum_patches_per_column=maximum_patches_per_column,
    )

    def aggregate(values: jnp.ndarray) -> jnp.ndarray:
        return column_to_gridcell_urbanf(
            topology,
            values,
            canyon_height_to_width_ratio_landunit,
            maximum_columns_per_gridcell=maximum_columns_per_gridcell,
        )

    liquid = aggregate(column.liquid_water_mass_column)
    ice = aggregate(column.ice_water_mass_column)
    heat = aggregate(column.heat_content_column)
    liquid_heat = aggregate(column.liquid_sensible_heat_column)
    liquid_capacity = aggregate(column.liquid_heat_capacity_column)
    positive_capacity = liquid_capacity > jnp.float64(0.0)
    safe_capacity = jnp.where(positive_capacity, liquid_capacity, jnp.float64(1.0))
    liquid_temperature = fortran_add(
        fortran_divide(liquid_heat, safe_capacity),
        jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
    )
    liquid_temperature = jnp.where(
        positive_capacity,
        liquid_temperature,
        jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
    )
    return DynamicHeatWaterContent(
        liquid_water_mass_gridcell=liquid,
        ice_water_mass_gridcell=ice,
        heat_content_gridcell=heat,
        liquid_sensible_heat_gridcell=liquid_heat,
        liquid_heat_capacity_gridcell=liquid_capacity,
        liquid_water_temperature_gridcell=liquid_temperature,
    )


def dynamic_biogeophysical_balance_step(
    content_before: DynamicHeatWaterContent,
    topology_after: SubgridTopology,
    inputs_after: TotalWaterAndHeatInputs,
    canyon_height_to_width_ratio_landunit_after: jnp.ndarray,
    dribbler_state: DynamicBiogeophysicalDribblerState,
    *,
    is_beginning_current_year: jnp.ndarray | bool,
    is_first_step: jnp.ndarray | bool,
    previous_year_fraction: jnp.ndarray | float,
    current_year_fraction: jnp.ndarray | float,
    time_step_seconds: jnp.ndarray | float,
    days_per_year: jnp.ndarray | float,
    for_testing_zero_dynamic_balance_fluxes: jnp.ndarray | bool = False,
    maximum_patches_per_column: int = 79,
    maximum_columns_per_gridcell: int = 64,
) -> DynamicBiogeophysicalBalanceResult:

    after = dynamic_heat_water_content(
        topology_after,
        inputs_after,
        canyon_height_to_width_ratio_landunit_after,
        maximum_patches_per_column=maximum_patches_per_column,
        maximum_columns_per_gridcell=maximum_columns_per_gridcell,
    )
    zero_fluxes = jnp.asarray(for_testing_zero_dynamic_balance_fluxes, dtype=jnp.bool_)
    delta_liquid_candidate = fortran_add(
        after.liquid_water_mass_gridcell,
        -content_before.liquid_water_mass_gridcell,
    )
    delta_ice_candidate = fortran_add(
        after.ice_water_mass_gridcell,
        -content_before.ice_water_mass_gridcell,
    )
    delta_heat_candidate = fortran_add(
        after.heat_content_gridcell,
        -content_before.heat_content_gridcell,
    )
    delta_liquid = jnp.where(
        zero_fluxes,
        jnp.zeros_like(delta_liquid_candidate),
        delta_liquid_candidate,
    )
    delta_ice = jnp.where(zero_fluxes, jnp.zeros_like(delta_ice_candidate), delta_ice_candidate)
    delta_heat = jnp.where(zero_fluxes, jnp.zeros_like(delta_heat_candidate), delta_heat_candidate)
    delta_heat = adjust_delta_heat_for_delta_liquid(
        delta_liquid,
        content_before.liquid_water_temperature_gridcell,
        after.liquid_water_temperature_gridcell,
        delta_heat,
    )

    step_arguments = {
        "is_beginning_current_year": is_beginning_current_year,
        "is_first_step": is_first_step,
        "previous_year_fraction": previous_year_fraction,
        "current_year_fraction": current_year_fraction,
        "time_step_seconds": time_step_seconds,
        "days_per_year": days_per_year,
    }
    liquid_flux = annual_flux_dribbler_step(
        dribbler_state.liquid_water,
        delta_liquid,
        **step_arguments,
    )
    ice_flux = annual_flux_dribbler_step(
        dribbler_state.ice_water,
        delta_ice,
        **step_arguments,
    )
    energy_flux = annual_flux_dribbler_step(
        dribbler_state.energy,
        delta_heat,
        **step_arguments,
    )
    return DynamicBiogeophysicalBalanceResult(
        content_after=after,
        delta_liquid_water_gridcell=delta_liquid,
        delta_ice_water_gridcell=delta_ice,
        delta_heat_gridcell=delta_heat,
        liquid_water_flux=liquid_flux,
        ice_water_flux=ice_flux,
        energy_flux=energy_flux,
    )
