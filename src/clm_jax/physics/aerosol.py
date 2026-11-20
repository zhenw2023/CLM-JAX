
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from clm_jax.physics.numerics import fortran_add, fortran_divide

NUMBER_OF_SNOW_LEVELS = 12


class SnowAerosolState(NamedTuple):

    hydrophobic_black_carbon_layer: jnp.ndarray
    hydrophilic_black_carbon_layer: jnp.ndarray
    total_black_carbon_layer: jnp.ndarray
    black_carbon_column: jnp.ndarray
    black_carbon_top: jnp.ndarray
    hydrophobic_organic_carbon_layer: jnp.ndarray
    hydrophilic_organic_carbon_layer: jnp.ndarray
    total_organic_carbon_layer: jnp.ndarray
    organic_carbon_column: jnp.ndarray
    organic_carbon_top: jnp.ndarray
    dust_species_1_layer: jnp.ndarray
    dust_species_2_layer: jnp.ndarray
    dust_species_3_layer: jnp.ndarray
    dust_species_4_layer: jnp.ndarray
    total_dust_layer: jnp.ndarray
    dust_column: jnp.ndarray
    dust_top: jnp.ndarray


class SnowAerosolConcentrationState(NamedTuple):

    hydrophobic_black_carbon_layer: jnp.ndarray
    hydrophilic_black_carbon_layer: jnp.ndarray
    hydrophobic_organic_carbon_layer: jnp.ndarray
    hydrophilic_organic_carbon_layer: jnp.ndarray
    dust_species_1_layer: jnp.ndarray
    dust_species_2_layer: jnp.ndarray
    dust_species_3_layer: jnp.ndarray
    dust_species_4_layer: jnp.ndarray


class SnowAerosolMassInputs(NamedTuple):

    snow_column: jnp.ndarray
    no_snow_column: jnp.ndarray
    snow_layer_count_column: jnp.ndarray
    snow_ice_column_level: jnp.ndarray
    snow_liquid_column_level: jnp.ndarray


class SnowAerosolMassState(NamedTuple):

    top_snow_water_column: jnp.ndarray
    snow_grain_radius_column_level: jnp.ndarray
    aerosol: SnowAerosolState
    concentration: SnowAerosolConcentrationState


class SnowAerosolMassDiagnostics(NamedTuple):

    snow_column: jnp.ndarray
    no_snow_column: jnp.ndarray
    active_snow_layer: jnp.ndarray


class SnowAerosolMassResult(NamedTuple):

    state: SnowAerosolMassState
    diagnostics: SnowAerosolMassDiagnostics


def reset_new_snow_aerosols(
    state: SnowAerosolState,
    new_snow_column: jnp.ndarray,
) -> SnowAerosolState:

    reset = jnp.asarray(new_snow_column, dtype=jnp.bool_)

    def reset_field(value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.asarray(value, dtype=jnp.float64)
        broadcast_shape = (reset.shape[0],) + (1,) * (value.ndim - 1)
        return jnp.where(reset.reshape(broadcast_shape), jnp.float64(0.0), value)

    return SnowAerosolState(*(reset_field(value) for value in state))


def _validate_aerosol_mass_shapes(
    inputs: SnowAerosolMassInputs,
    state: SnowAerosolMassState,
) -> None:
    if inputs.snow_layer_count_column.ndim != 1:
        raise ValueError("snow_layer_count_column must be a vector")
    column_shape = inputs.snow_layer_count_column.shape
    layer_shape = (column_shape[0], NUMBER_OF_SNOW_LEVELS)
    for name in ("snow_column", "no_snow_column"):
        if getattr(inputs, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")
    for name in ("snow_ice_column_level", "snow_liquid_column_level"):
        if getattr(inputs, name).shape != layer_shape:
            raise ValueError(f"{name} must have shape {layer_shape}")
    if state.top_snow_water_column.shape != column_shape:
        raise ValueError(f"top_snow_water_column must have shape {column_shape}")
    if state.snow_grain_radius_column_level.shape != layer_shape:
        raise ValueError(
            "snow_grain_radius_column_level must have shape "
            f"{layer_shape}"
        )
    layer_aerosol_names = {
        "hydrophobic_black_carbon_layer",
        "hydrophilic_black_carbon_layer",
        "total_black_carbon_layer",
        "hydrophobic_organic_carbon_layer",
        "hydrophilic_organic_carbon_layer",
        "total_organic_carbon_layer",
        "dust_species_1_layer",
        "dust_species_2_layer",
        "dust_species_3_layer",
        "dust_species_4_layer",
        "total_dust_layer",
    }
    for name in SnowAerosolState._fields:
        expected = layer_shape if name in layer_aerosol_names else column_shape
        if getattr(state.aerosol, name).shape != expected:
            raise ValueError(f"aerosol.{name} must have shape {expected}")
    for name in SnowAerosolConcentrationState._fields:
        if getattr(state.concentration, name).shape != layer_shape:
            raise ValueError(f"concentration.{name} must have shape {layer_shape}")


def snow_aerosol_masses(
    inputs: SnowAerosolMassInputs,
    state: SnowAerosolMassState,
) -> SnowAerosolMassResult:

    _validate_aerosol_mass_shapes(inputs, state)
    snow_column = jnp.asarray(inputs.snow_column, dtype=jnp.bool_)
    no_snow_column = jnp.asarray(inputs.no_snow_column, dtype=jnp.bool_)
    snow_layer_count = jnp.asarray(
        inputs.snow_layer_count_column,
        dtype=jnp.int32,
    )
    snow_ice = jnp.asarray(inputs.snow_ice_column_level, dtype=jnp.float64)
    snow_liquid = jnp.asarray(
        inputs.snow_liquid_column_level,
        dtype=jnp.float64,
    )
    top_snow_water = jnp.asarray(
        state.top_snow_water_column,
        dtype=jnp.float64,
    )
    snow_grain_radius = jnp.asarray(
        state.snow_grain_radius_column_level,
        dtype=jnp.float64,
    )

    primary_mass = jnp.stack(
        (
            state.aerosol.hydrophobic_black_carbon_layer,
            state.aerosol.hydrophilic_black_carbon_layer,
            state.aerosol.hydrophobic_organic_carbon_layer,
            state.aerosol.hydrophilic_organic_carbon_layer,
            state.aerosol.dust_species_1_layer,
            state.aerosol.dust_species_2_layer,
            state.aerosol.dust_species_3_layer,
            state.aerosol.dust_species_4_layer,
        ),
        axis=1,
    ).astype(jnp.float64)
    total_mass = jnp.stack(
        (
            state.aerosol.total_black_carbon_layer,
            state.aerosol.total_organic_carbon_layer,
            state.aerosol.total_dust_layer,
        ),
        axis=1,
    ).astype(jnp.float64)
    column_mass = jnp.stack(
        (
            state.aerosol.black_carbon_column,
            state.aerosol.organic_carbon_column,
            state.aerosol.dust_column,
        ),
        axis=1,
    ).astype(jnp.float64)
    top_mass = jnp.stack(
        (
            state.aerosol.black_carbon_top,
            state.aerosol.organic_carbon_top,
            state.aerosol.dust_top,
        ),
        axis=1,
    ).astype(jnp.float64)
    concentration = jnp.stack(tuple(state.concentration), axis=1).astype(
        jnp.float64
    )

    level_number = jnp.arange(
        -NUMBER_OF_SNOW_LEVELS + 1,
        1,
        dtype=jnp.int32,
    )
    active_snow_layer = (
        level_number[None, :] >= snow_layer_count[:, None] + jnp.int32(1)
    )
    column_mass = jnp.where(
        snow_column[:, None],
        jnp.float64(0.0),
        column_mass,
    )

    active = snow_column[:, None] & active_snow_layer
    empty = snow_column[:, None] & ~active_snow_layer
    black_carbon_total = fortran_add(primary_mass[:, 0, :], primary_mass[:, 1, :])
    organic_carbon_total = fortran_add(primary_mass[:, 2, :], primary_mass[:, 3, :])
    dust_total = fortran_add(
        fortran_add(
            fortran_add(primary_mass[:, 4, :], primary_mass[:, 5, :]),
            primary_mass[:, 6, :],
        ),
        primary_mass[:, 7, :],
    )
    layer_total = jnp.stack((black_carbon_total, organic_carbon_total, dust_total), axis=1)

    def accumulate_level(level, current):
        return jnp.where(
            active[:, level, None],
            fortran_add(current, layer_total[:, :, level]),
            current,
        )

    column_mass = lax.fori_loop(0, NUMBER_OF_SNOW_LEVELS, accumulate_level, column_mass)
    total_mass = jnp.where(
        active[:, None, :], layer_total,
        jnp.where(empty[:, None, :], jnp.float64(0.0), total_mass),
    )
    snow_mass = fortran_add(snow_ice, snow_liquid)
    safe_snow_mass = jnp.where(active, snow_mass, jnp.float64(1.0))
    candidate_concentration = fortran_divide(primary_mass, safe_snow_mass[:, None, :])
    concentration = jnp.where(
        active[:, None, :], candidate_concentration,
        jnp.where(empty[:, None, :], jnp.float64(0.0), concentration),
    )
    primary_mass = jnp.where(empty[:, None, :], jnp.float64(0.0), primary_mass)
    snow_grain_radius = jnp.where(empty, jnp.float64(0.0), snow_grain_radius)

    column_index = jnp.arange(snow_layer_count.shape[0], dtype=jnp.int32)
    top_index = jnp.clip(
        snow_layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS),
        jnp.int32(0),
        jnp.int32(NUMBER_OF_SNOW_LEVELS - 1),
    )
    top_snow_water = jnp.where(
        snow_column,
        fortran_add(
            snow_ice[column_index, top_index],
            snow_liquid[column_index, top_index],
        ),
        top_snow_water,
    )
    diagnosed_top_mass = total_mass[column_index, :, top_index]
    top_mass = jnp.where(snow_column[:, None], diagnosed_top_mass, top_mass)

    primary_mass = jnp.where(
        no_snow_column[:, None, None],
        jnp.float64(0.0),
        primary_mass,
    )
    total_mass = jnp.where(
        no_snow_column[:, None, None],
        jnp.float64(0.0),
        total_mass,
    )
    concentration = jnp.where(
        no_snow_column[:, None, None],
        jnp.float64(0.0),
        concentration,
    )
    column_mass = jnp.where(
        no_snow_column[:, None],
        jnp.float64(0.0),
        column_mass,
    )
    top_mass = jnp.where(
        no_snow_column[:, None],
        jnp.float64(0.0),
        top_mass,
    )

    aerosol = SnowAerosolState(
        hydrophobic_black_carbon_layer=primary_mass[:, 0, :],
        hydrophilic_black_carbon_layer=primary_mass[:, 1, :],
        total_black_carbon_layer=total_mass[:, 0, :],
        black_carbon_column=column_mass[:, 0],
        black_carbon_top=top_mass[:, 0],
        hydrophobic_organic_carbon_layer=primary_mass[:, 2, :],
        hydrophilic_organic_carbon_layer=primary_mass[:, 3, :],
        total_organic_carbon_layer=total_mass[:, 1, :],
        organic_carbon_column=column_mass[:, 1],
        organic_carbon_top=top_mass[:, 1],
        dust_species_1_layer=primary_mass[:, 4, :],
        dust_species_2_layer=primary_mass[:, 5, :],
        dust_species_3_layer=primary_mass[:, 6, :],
        dust_species_4_layer=primary_mass[:, 7, :],
        total_dust_layer=total_mass[:, 2, :],
        dust_column=column_mass[:, 2],
        dust_top=top_mass[:, 2],
    )
    diagnosed_concentration = SnowAerosolConcentrationState(
        *(concentration[:, species, :] for species in range(8))
    )
    return SnowAerosolMassResult(
        state=SnowAerosolMassState(
            top_snow_water_column=top_snow_water,
            snow_grain_radius_column_level=snow_grain_radius,
            aerosol=aerosol,
            concentration=diagnosed_concentration,
        ),
        diagnostics=SnowAerosolMassDiagnostics(
            snow_column=snow_column,
            no_snow_column=no_snow_column,
            active_snow_layer=active_snow_layer,
        ),
    )
