
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.constants import DENSITY_FRESH_WATER, DENSITY_ICE
from clm_jax.io.restart import LANDUNIT_LAKE, SubgridTopology
from clm_jax.physics.numerics import (
    fortran_divide,
    fortran_ordered_sum,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)

NUMBER_OF_SNOW_LEVELS = 12
NUMBER_OF_GROUND_LEVELS = 25
NUMBER_OF_SOIL_SNOW_LEVELS = NUMBER_OF_SNOW_LEVELS + NUMBER_OF_GROUND_LEVELS
GROUND_SURFACE_INDEX = NUMBER_OF_SNOW_LEVELS
IRREDUCIBLE_SNOW_WATER_SATURATION = 0.033
IMPERMEABLE_SNOW_POROSITY = 0.05
SCAVENGING_FACTORS = (
    0.20,
    0.03,
    0.20,
    0.03,
    0.02,
    0.02,
    0.01,
    0.01,
)


class SnowColumnMasks(NamedTuple):
    snow_column: jnp.ndarray
    no_snow_column: jnp.ndarray


class SnowAerosolDepositionState(NamedTuple):
    black_carbon_total_column: jnp.ndarray
    black_carbon_wet_column: jnp.ndarray
    black_carbon_dry_column: jnp.ndarray
    black_carbon_hydrophilic_column: jnp.ndarray
    black_carbon_hydrophobic_column: jnp.ndarray
    organic_carbon_total_column: jnp.ndarray
    organic_carbon_wet_column: jnp.ndarray
    organic_carbon_dry_column: jnp.ndarray
    organic_carbon_hydrophilic_column: jnp.ndarray
    organic_carbon_hydrophobic_column: jnp.ndarray
    dust_total_column: jnp.ndarray
    dust_species_1_wet_column: jnp.ndarray
    dust_species_1_dry_column: jnp.ndarray
    dust_species_2_wet_column: jnp.ndarray
    dust_species_2_dry_column: jnp.ndarray
    dust_species_3_wet_column: jnp.ndarray
    dust_species_3_dry_column: jnp.ndarray
    dust_species_4_wet_column: jnp.ndarray
    dust_species_4_dry_column: jnp.ndarray


class SnowWaterAerosolState(NamedTuple):
    black_carbon_hydrophilic_layer: jnp.ndarray
    black_carbon_hydrophobic_layer: jnp.ndarray
    organic_carbon_hydrophilic_layer: jnp.ndarray
    organic_carbon_hydrophobic_layer: jnp.ndarray
    dust_species_1_layer: jnp.ndarray
    dust_species_2_layer: jnp.ndarray
    dust_species_3_layer: jnp.ndarray
    dust_species_4_layer: jnp.ndarray
    deposition: SnowAerosolDepositionState


class SnowWaterInputs(NamedTuple):
    time_step_seconds: jnp.ndarray
    snow_layer_count_column: jnp.ndarray
    effective_snow_fraction_column: jnp.ndarray
    snow_water_equivalent_column: jnp.ndarray
    snowmelt_flux_column: jnp.ndarray
    ground_rainfall_flux_column: jnp.ndarray
    snow_sublimation_flux_column: jnp.ndarray
    snow_dew_flux_column: jnp.ndarray
    ground_evaporation_flux_column: jnp.ndarray
    ground_dew_flux_column: jnp.ndarray
    atmospheric_aerosol_flux_gridcell_species: jnp.ndarray


class SnowWaterState(NamedTuple):
    integrated_snow_column: jnp.ndarray
    snow_fraction_column: jnp.ndarray
    snow_depth_column: jnp.ndarray
    soil_snow_liquid_column_level: jnp.ndarray
    soil_snow_ice_column_level: jnp.ndarray
    soil_snow_thickness_column_level: jnp.ndarray
    snow_drainage_flux_column: jnp.ndarray
    top_soil_water_flux_column: jnp.ndarray
    aerosol: SnowWaterAerosolState


class SnowWaterDiagnostics(NamedTuple):
    masks: SnowColumnMasks
    bottom_snow_outflow_mass_column: jnp.ndarray
    snow_ice_volume_fraction_column_level: jnp.ndarray
    snow_liquid_volume_fraction_column_level: jnp.ndarray
    snow_effective_porosity_column_level: jnp.ndarray


class SnowWaterResult(NamedTuple):
    state: SnowWaterState
    diagnostics: SnowWaterDiagnostics


def build_snow_column_masks(
    topology: SubgridTopology,
    snow_layer_count_column: jnp.ndarray,
    *,
    lake_only: bool = False,
) -> SnowColumnMasks:

    layer_count = jnp.asarray(snow_layer_count_column, dtype=jnp.int32)
    active = jnp.asarray(topology.column.active, dtype=jnp.bool_)
    is_lake = jnp.asarray(
        topology.column.landunit_type,
        dtype=jnp.int32,
    ) == jnp.int32(LANDUNIT_LAKE)
    selected = active & (is_lake if lake_only else ~is_lake)
    return SnowColumnMasks(
        snow_column=selected & (layer_count < jnp.int32(0)),
        no_snow_column=selected & (layer_count >= jnp.int32(0)),
    )


def _validate_shapes(
    topology: SubgridTopology,
    inputs: SnowWaterInputs,
    state: SnowWaterState,
) -> None:
    column_shape = topology.column.gridcell.shape
    level_shape = (column_shape[0], NUMBER_OF_SOIL_SNOW_LEVELS)
    snow_shape = (column_shape[0], NUMBER_OF_SNOW_LEVELS)
    if len(column_shape) != 1:
        raise ValueError("column topology must be a vector")
    scalar_inputs = {
        "snow_layer_count_column",
        "effective_snow_fraction_column",
        "snow_water_equivalent_column",
        "snowmelt_flux_column",
        "ground_rainfall_flux_column",
        "snow_sublimation_flux_column",
        "snow_dew_flux_column",
        "ground_evaporation_flux_column",
        "ground_dew_flux_column",
    }
    for name in scalar_inputs:
        if getattr(inputs, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")
    if jnp.shape(inputs.time_step_seconds) != ():
        raise ValueError("time_step_seconds must be scalar")
    aerosol_forcing = inputs.atmospheric_aerosol_flux_gridcell_species
    if aerosol_forcing.ndim != 2 or aerosol_forcing.shape[1] != 14:
        raise ValueError(
            "atmospheric_aerosol_flux_gridcell_species must have shape (gridcell, 14)"
        )
    for name in (
        "integrated_snow_column",
        "snow_fraction_column",
        "snow_depth_column",
        "snow_drainage_flux_column",
        "top_soil_water_flux_column",
    ):
        if getattr(state, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")
    for name in (
        "soil_snow_liquid_column_level",
        "soil_snow_ice_column_level",
        "soil_snow_thickness_column_level",
    ):
        if getattr(state, name).shape != level_shape:
            raise ValueError(f"{name} must have shape {level_shape}")
    for name in SnowWaterAerosolState._fields[:-1]:
        if getattr(state.aerosol, name).shape != snow_shape:
            raise ValueError(f"{name} must have shape {snow_shape}")
    for name in SnowAerosolDepositionState._fields:
        if getattr(state.aerosol.deposition, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")


def _aerosol_deposition(
    forcing: jnp.ndarray,
) -> SnowAerosolDepositionState:
    black_carbon_dry = fortran_add(forcing[:, 0], forcing[:, 1])
    black_carbon_wet = forcing[:, 2]
    black_carbon_hydrophilic = fortran_add(forcing[:, 0], forcing[:, 2])
    black_carbon_hydrophobic = forcing[:, 1]
    black_carbon_total = fortran_ordered_sum(forcing[:, 0:3].T)

    organic_carbon_dry = fortran_add(forcing[:, 3], forcing[:, 4])
    organic_carbon_wet = forcing[:, 5]
    organic_carbon_hydrophilic = fortran_add(forcing[:, 3], forcing[:, 5])
    organic_carbon_hydrophobic = forcing[:, 4]
    organic_carbon_total = fortran_ordered_sum(forcing[:, 3:6].T)

    dust_total = fortran_ordered_sum(forcing[:, 6:14].T)
    return SnowAerosolDepositionState(
        black_carbon_total_column=black_carbon_total,
        black_carbon_wet_column=black_carbon_wet,
        black_carbon_dry_column=black_carbon_dry,
        black_carbon_hydrophilic_column=black_carbon_hydrophilic,
        black_carbon_hydrophobic_column=black_carbon_hydrophobic,
        organic_carbon_total_column=organic_carbon_total,
        organic_carbon_wet_column=organic_carbon_wet,
        organic_carbon_dry_column=organic_carbon_dry,
        organic_carbon_hydrophilic_column=organic_carbon_hydrophilic,
        organic_carbon_hydrophobic_column=organic_carbon_hydrophobic,
        dust_total_column=dust_total,
        dust_species_1_wet_column=forcing[:, 6],
        dust_species_1_dry_column=forcing[:, 7],
        dust_species_2_wet_column=forcing[:, 8],
        dust_species_2_dry_column=forcing[:, 9],
        dust_species_3_wet_column=forcing[:, 10],
        dust_species_3_dry_column=forcing[:, 11],
        dust_species_4_wet_column=forcing[:, 12],
        dust_species_4_dry_column=forcing[:, 13],
    )


def snow_water(
    topology: SubgridTopology,
    inputs: SnowWaterInputs,
    state: SnowWaterState,
    *,
    lake_only: bool = False,
) -> SnowWaterResult:

    _validate_shapes(topology, inputs, state)
    if jax.default_backend() == "gpu" and topology.column.gridcell.size:
        from clm_jax.physics.snow_water_gpu import snow_water_gpu
        return snow_water_gpu(topology, inputs, state, lake_only=lake_only)
    return _snow_water_reference(topology, inputs, state, lake_only=lake_only)


def _snow_water_reference(
    topology: SubgridTopology,
    inputs: SnowWaterInputs,
    state: SnowWaterState,
    *,
    lake_only: bool = False,
) -> SnowWaterResult:
    column = topology.column
    local = _SnowWaterTopology(_SnowWaterColumns(
        column.gridcell, column.landunit_type, column.active,
    ))
    return _snow_water_storage_adjoint(local, inputs, state, lake_only)


class _SnowWaterColumns(NamedTuple):
    gridcell: jax.Array
    landunit_type: jax.Array
    active: jax.Array


class _SnowWaterTopology(NamedTuple):
    column: _SnowWaterColumns


@partial(jax.custom_jvp, nondiff_argnums=(3,))
def _snow_water_storage_adjoint(topology, inputs, state, lake_only):
    return _snow_water_primal(topology, inputs, state, lake_only=lake_only)


@_snow_water_storage_adjoint.defjvp
def _snow_water_storage_jvp(lake_only, primals, tangents):
    def guarded(topology, inputs, state):
        active = build_snow_column_masks(
            topology, inputs.snow_layer_count_column, lake_only=lake_only,
        ).snow_column
        top = inputs.snow_layer_count_column + jnp.int32(NUMBER_OF_SNOW_LEVELS)

        def used_slots(value):
            if value.ndim != 2:
                return jnp.ones_like(value, dtype=jnp.bool_)
            levels = jnp.arange(value.shape[1])[None, :]
            return (levels >= NUMBER_OF_SNOW_LEVELS) | (
                active[:, None] & (levels >= top[:, None]))

        safe = jax.tree.map(lambda value: jnp.where(
            used_slots(value), value, jnp.zeros_like(value),
        ), state)
        result = _snow_water_primal(topology, inputs, safe, lake_only=lake_only)
        restored = jax.tree.map(lambda value, original: jnp.where(
            used_slots(original), value, original,
        ), result.state, state)
        return result._replace(state=restored)

    return (_snow_water_storage_adjoint(*primals, lake_only),
            jax.jvp(guarded, primals, tangents)[1])


def _snow_water_primal(
    topology: SubgridTopology,
    inputs: SnowWaterInputs,
    state: SnowWaterState,
    *,
    lake_only: bool = False,
) -> SnowWaterResult:
    number_of_columns = topology.column.gridcell.shape[0]
    column_index = jnp.arange(number_of_columns, dtype=jnp.int32)
    masks = build_snow_column_masks(
        topology,
        inputs.snow_layer_count_column,
        lake_only=lake_only,
    )
    snow_column = masks.snow_column
    no_snow_column = masks.no_snow_column
    layer_count = jnp.asarray(inputs.snow_layer_count_column, dtype=jnp.int32)
    top_index = layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS)
    time_step = jnp.asarray(inputs.time_step_seconds, dtype=jnp.float64)
    snow_fraction = jnp.asarray(
        inputs.effective_snow_fraction_column,
        dtype=jnp.float64,
    )
    liquid = jnp.asarray(state.soil_snow_liquid_column_level, dtype=jnp.float64)
    ice = jnp.asarray(state.soil_snow_ice_column_level, dtype=jnp.float64)
    thickness = jnp.asarray(state.soil_snow_thickness_column_level, dtype=jnp.float64)

    top_ice = ice[column_index, top_index]
    snow_dew_minus_sublimation = fortran_add(
        jnp.asarray(inputs.snow_dew_flux_column, dtype=jnp.float64),
        -jnp.asarray(inputs.snow_sublimation_flux_column, dtype=jnp.float64),
    )
    ice_surface_change = fortran_multiply(
        fortran_multiply(snow_fraction, snow_dew_minus_sublimation),
        time_step,
    )
    updated_top_ice = fortran_add(top_ice, ice_surface_change)
    negative_top_ice = snow_column & (updated_top_ice < jnp.float64(0.0))
    ice = ice.at[column_index, top_index].set(
        jnp.where(
            snow_column,
            jnp.where(negative_top_ice, jnp.float64(0.0), updated_top_ice),
            top_ice,
        )
    )

    top_liquid = liquid[column_index, top_index]
    top_liquid = jnp.where(
        negative_top_ice,
        fortran_add(top_liquid, updated_top_ice),
        top_liquid,
    )
    liquid_surface_flux = fortran_add(
        fortran_add(
            jnp.asarray(inputs.ground_rainfall_flux_column, dtype=jnp.float64),
            jnp.asarray(inputs.ground_dew_flux_column, dtype=jnp.float64),
        ),
        -jnp.asarray(inputs.ground_evaporation_flux_column, dtype=jnp.float64),
    )
    liquid_surface_change = fortran_multiply(
        fortran_multiply(snow_fraction, liquid_surface_flux),
        time_step,
    )
    updated_top_liquid = fortran_add(top_liquid, liquid_surface_change)
    liquid = liquid.at[column_index, top_index].set(
        jnp.where(snow_column, updated_top_liquid, liquid[column_index, top_index])
    )

    def propagate_negative_liquid(
        level_index: int,
        carry: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        layer_liquid, still_propagating = carry
        reached_top = jnp.int32(level_index) >= top_index
        process = snow_column & reached_top & still_propagating
        current = layer_liquid[:, level_index]
        negative = process & (current < jnp.float64(0.0))
        current_updated = jnp.where(negative, jnp.float64(0.0), current)
        next_value = layer_liquid[:, level_index + 1]
        next_updated = jnp.where(
            negative,
            fortran_add(next_value, current),
            next_value,
        )
        layer_liquid = layer_liquid.at[:, level_index].set(current_updated)
        layer_liquid = layer_liquid.at[:, level_index + 1].set(next_updated)
        still_propagating = jnp.where(process, negative, still_propagating)
        return layer_liquid, still_propagating

    liquid, _ = lax.fori_loop(
        0,
        GROUND_SURFACE_INDEX + 1,
        propagate_negative_liquid,
        (liquid, snow_column),
    )

    snow_level = jnp.arange(NUMBER_OF_SNOW_LEVELS, dtype=jnp.int32)
    active_snow_level = snow_column[:, None] & (snow_level[None, :] >= top_index[:, None])
    snow_thickness = thickness[:, :NUMBER_OF_SNOW_LEVELS]
    snow_liquid = liquid[:, :NUMBER_OF_SNOW_LEVELS]
    snow_ice = ice[:, :NUMBER_OF_SNOW_LEVELS]
    ice_denominator = fortran_multiply(
        fortran_multiply(snow_thickness, snow_fraction[:, None]),
        jnp.float64(DENSITY_ICE),
    )
    safe_ice_denominator = jnp.where(
        active_snow_level,
        ice_denominator,
        jnp.float64(1.0),
    )
    ice_volume = jnp.where(
        active_snow_level,
        jnp.minimum(
            jnp.float64(1.0),
            fortran_divide(snow_ice, safe_ice_denominator),
        ),
        jnp.float64(0.0),
    )
    effective_porosity = jnp.where(
        active_snow_level,
        fortran_add(jnp.float64(1.0), -ice_volume),
        jnp.float64(0.0),
    )
    liquid_denominator = fortran_multiply(
        fortran_multiply(snow_thickness, snow_fraction[:, None]),
        jnp.float64(DENSITY_FRESH_WATER),
    )
    safe_liquid_denominator = jnp.where(
        active_snow_level,
        liquid_denominator,
        jnp.float64(1.0),
    )
    liquid_volume = jnp.where(
        active_snow_level,
        jnp.minimum(
            effective_porosity,
            fortran_divide(snow_liquid, safe_liquid_denominator),
        ),
        jnp.float64(0.0),
    )

    aerosol_mass = jnp.stack(state.aerosol[:-1], axis=1)
    incoming_water = jnp.zeros(number_of_columns, dtype=jnp.float64)
    incoming_aerosol = jnp.zeros((number_of_columns, 8), dtype=jnp.float64)
    bottom_outflow = jnp.zeros(number_of_columns, dtype=jnp.float64)
    scavenging = jnp.asarray(SCAVENGING_FACTORS, dtype=jnp.float64)

    def percolate_level(
        carry: tuple[jnp.ndarray, jnp.ndarray],
        level_index: int,
    ):
        water_in, aerosol_in = carry
        active = active_snow_level[:, level_index]
        current_liquid = snow_liquid[:, level_index]
        current_liquid = jnp.where(
            active,
            fortran_add(current_liquid, water_in),
            current_liquid,
        )
        current_aerosol = aerosol_mass[:, :, level_index]
        current_aerosol = jnp.where(
            active[:, None],
            fortran_add(current_aerosol, aerosol_in),
            current_aerosol,
        )

        retained_fraction = fortran_multiply(
            jnp.float64(IRREDUCIBLE_SNOW_WATER_SATURATION),
            effective_porosity[:, level_index],
        )
        excess_volume = fortran_add(liquid_volume[:, level_index], -retained_fraction)
        outflow_volume = fortran_multiply(
            fortran_multiply(excess_volume, snow_thickness[:, level_index]),
            snow_fraction,
        )
        outflow = jnp.maximum(jnp.float64(0.0), outflow_volume)
        next_index = jnp.minimum(
            jnp.int32(level_index + 1),
            jnp.int32(NUMBER_OF_SNOW_LEVELS - 1),
        )
        impermeable = (
            effective_porosity[:, level_index]
            < jnp.float64(IMPERMEABLE_SNOW_POROSITY)
        ) | (
            effective_porosity[:, next_index]
            < jnp.float64(IMPERMEABLE_SNOW_POROSITY)
        )
        next_available_fraction = fortran_add(
            fortran_add(jnp.float64(1.0), -ice_volume[:, next_index]),
            -liquid_volume[:, next_index],
        )
        next_capacity = fortran_multiply(
            fortran_multiply(
                next_available_fraction,
                snow_thickness[:, next_index],
            ),
            snow_fraction,
        )
        upper_layer_outflow = jnp.where(
            impermeable,
            jnp.float64(0.0),
            jnp.minimum(outflow, next_capacity),
        )
        outflow = jnp.where(
            jnp.int32(level_index) < jnp.int32(NUMBER_OF_SNOW_LEVELS - 1),
            upper_layer_outflow,
            outflow,
        )
        outflow = fortran_multiply(outflow, jnp.float64(1000.0))
        outflow = jnp.where(active, outflow, water_in)
        updated_liquid = jnp.where(
            active,
            fortran_add(current_liquid, -outflow),
            current_liquid,
        )

        liquid_ice_mass = fortran_add(updated_liquid, snow_ice[:, level_index])
        liquid_ice_mass = jnp.where(
            liquid_ice_mass < jnp.float64(1.0e-30),
            jnp.float64(1.0e-30),
            liquid_ice_mass,
        )
        concentration = fortran_divide(current_aerosol, liquid_ice_mass[:, None])
        aerosol_out = fortran_multiply(
            fortran_multiply(outflow[:, None], scavenging[None, :]),
            concentration,
        )
        aerosol_out = jnp.minimum(aerosol_out, current_aerosol)
        aerosol_out = jnp.where(active[:, None], aerosol_out, aerosol_in)
        updated_aerosol = jnp.where(
            active[:, None],
            fortran_add(current_aerosol, -aerosol_out),
            current_aerosol,
        )
        return (outflow, aerosol_out), (updated_liquid, updated_aerosol)

    (bottom_outflow, _), (layer_liquid, layer_aerosol) = lax.scan(
        percolate_level,
        (incoming_water, incoming_aerosol),
        jnp.arange(NUMBER_OF_SNOW_LEVELS, dtype=jnp.int32),
    )
    snow_liquid = jnp.moveaxis(layer_liquid, 0, -1)
    aerosol_mass = jnp.moveaxis(layer_aerosol, 0, -1)
    liquid = liquid.at[:, :NUMBER_OF_SNOW_LEVELS].set(snow_liquid)

    gridcell_index = jnp.asarray(topology.column.gridcell, dtype=jnp.int32)
    atmospheric_aerosol = jnp.asarray(
        inputs.atmospheric_aerosol_flux_gridcell_species,
        dtype=jnp.float64,
    )[gridcell_index]
    deposition = _aerosol_deposition(atmospheric_aerosol)
    aerosol_top_index = jnp.clip(
        top_index,
        jnp.int32(0),
        jnp.int32(NUMBER_OF_SNOW_LEVELS - 1),
    )
    top_mass = aerosol_mass[column_index, :, aerosol_top_index]
    deposition_to_top = jnp.stack(
        (
            deposition.black_carbon_hydrophilic_column,
            deposition.black_carbon_hydrophobic_column,
            deposition.organic_carbon_hydrophilic_column,
            deposition.organic_carbon_hydrophobic_column,
            fortran_add(
                deposition.dust_species_1_dry_column,
                deposition.dust_species_1_wet_column,
            ),
            fortran_add(
                deposition.dust_species_2_dry_column,
                deposition.dust_species_2_wet_column,
            ),
            fortran_add(
                deposition.dust_species_3_dry_column,
                deposition.dust_species_3_wet_column,
            ),
            fortran_add(
                deposition.dust_species_4_dry_column,
                deposition.dust_species_4_wet_column,
            ),
        ),
        axis=1,
    )
    top_mass = jnp.where(
        snow_column[:, None],
        fortran_add(top_mass, fortran_multiply(deposition_to_top, time_step)),
        top_mass,
    )
    aerosol_mass = aerosol_mass.at[column_index, :, aerosol_top_index].set(top_mass)

    minimum_thickness = fortran_add(
        fortran_divide(snow_liquid, jnp.float64(DENSITY_FRESH_WATER)),
        fortran_divide(snow_ice, jnp.float64(DENSITY_ICE)),
    )
    updated_snow_thickness = jnp.where(
        active_snow_level,
        jnp.maximum(snow_thickness, minimum_thickness),
        snow_thickness,
    )
    thickness = thickness.at[:, :NUMBER_OF_SNOW_LEVELS].set(updated_snow_thickness)

    snow_drainage = jnp.asarray(state.snow_drainage_flux_column, dtype=jnp.float64)
    top_soil_flux = jnp.asarray(state.top_soil_water_flux_column, dtype=jnp.float64)
    integrated_snow = jnp.asarray(state.integrated_snow_column, dtype=jnp.float64)
    snow_cover_fraction = jnp.asarray(state.snow_fraction_column, dtype=jnp.float64)
    snow_depth = jnp.asarray(state.snow_depth_column, dtype=jnp.float64)
    outflow_rate = fortran_divide(bottom_outflow, time_step)
    snow_drainage = jnp.where(
        snow_column,
        fortran_add(snow_drainage, outflow_rate),
        snow_drainage,
    )
    exposed_rain = fortran_multiply(
        fortran_add(jnp.float64(1.0), -snow_fraction),
        jnp.asarray(inputs.ground_rainfall_flux_column, dtype=jnp.float64),
    )
    top_soil_flux = jnp.where(
        snow_column,
        fortran_add(outflow_rate, exposed_rain),
        top_soil_flux,
    )
    accumulated_surface_water = fortran_ordered_sum(
        jnp.stack(
            (
                jnp.asarray(inputs.snow_dew_flux_column, dtype=jnp.float64),
                jnp.asarray(inputs.ground_dew_flux_column, dtype=jnp.float64),
                jnp.asarray(inputs.ground_rainfall_flux_column, dtype=jnp.float64),
            ),
            axis=0,
        )
    )
    integrated_snow = jnp.where(
        snow_column,
        fortran_add(
            integrated_snow,
            fortran_multiply(
                fortran_multiply(snow_fraction, accumulated_surface_water),
                time_step,
            ),
        ),
        integrated_snow,
    )

    snow_drainage = jnp.where(
        no_snow_column,
        jnp.asarray(inputs.snowmelt_flux_column, dtype=jnp.float64),
        snow_drainage,
    )
    top_soil_flux = jnp.where(
        no_snow_column,
        fortran_add(
            jnp.asarray(inputs.ground_rainfall_flux_column, dtype=jnp.float64),
            jnp.asarray(inputs.snowmelt_flux_column, dtype=jnp.float64),
        ),
        top_soil_flux,
    )
    reset_snow = no_snow_column & (
        jnp.asarray(inputs.snow_water_equivalent_column, dtype=jnp.float64)
        <= jnp.float64(0.0)
    )
    integrated_snow = jnp.where(reset_snow, jnp.float64(0.0), integrated_snow)
    snow_cover_fraction = jnp.where(
        reset_snow,
        jnp.float64(0.0),
        snow_cover_fraction,
    )
    snow_depth = jnp.where(reset_snow, jnp.float64(0.0), snow_depth)

    aerosol_state = SnowWaterAerosolState(
        black_carbon_hydrophilic_layer=aerosol_mass[:, 0, :],
        black_carbon_hydrophobic_layer=aerosol_mass[:, 1, :],
        organic_carbon_hydrophilic_layer=aerosol_mass[:, 2, :],
        organic_carbon_hydrophobic_layer=aerosol_mass[:, 3, :],
        dust_species_1_layer=aerosol_mass[:, 4, :],
        dust_species_2_layer=aerosol_mass[:, 5, :],
        dust_species_3_layer=aerosol_mass[:, 6, :],
        dust_species_4_layer=aerosol_mass[:, 7, :],
        deposition=deposition,
    )
    return SnowWaterResult(
        state=SnowWaterState(
            integrated_snow_column=integrated_snow,
            snow_fraction_column=snow_cover_fraction,
            snow_depth_column=snow_depth,
            soil_snow_liquid_column_level=liquid,
            soil_snow_ice_column_level=ice,
            soil_snow_thickness_column_level=thickness,
            snow_drainage_flux_column=snow_drainage,
            top_soil_water_flux_column=top_soil_flux,
            aerosol=aerosol_state,
        ),
        diagnostics=SnowWaterDiagnostics(
            masks=masks,
            bottom_snow_outflow_mass_column=bottom_outflow,
            snow_ice_volume_fraction_column_level=ice_volume,
            snow_liquid_volume_fraction_column_level=liquid_volume,
            snow_effective_porosity_column_level=effective_porosity,
        ),
    )
