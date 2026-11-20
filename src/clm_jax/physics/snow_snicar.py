
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import lax, tree_util

from clm_jax.physics.converged_iteration import (
    checkpointed_scan,
    iteration_ad_checkpointing,
)
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_materialize,
    fortran_multiply,
    strict_negative_exp,
)

NUMBER_OF_SNOW_LAYERS = 12
NUMBER_OF_SNOW_SPECTRAL_BANDS = 5
NUMBER_OF_SNOW_AEROSOLS = 8
NUMBER_OF_SNOW_GRAIN_RADII = 1471
MINIMUM_SNOW_GRAIN_RADIUS_MICRONS = 30
MAXIMUM_SNOW_GRAIN_RADIUS_MICRONS = 1500
FRESH_SNOW_GRAIN_RADIUS_MICRONS = 55
MINIMUM_SNOW_WATER_EQUIVALENT = 1.0e-30

DIRECT_SPECTRAL_WEIGHTS = np.asarray(
    (
        1.0,
        0.49352158521175,
        0.18099494230665,
        0.12094898498813,
        0.20453448749347,
    ),
    dtype=np.float64,
)
DIFFUSE_SPECTRAL_WEIGHTS = np.asarray(
    (
        1.0,
        0.58581507618433,
        0.20156903770812,
        0.10917889346386,
        0.10343699264369,
    ),
    dtype=np.float64,
)


class SnowOpticalTables(NamedTuple):

    snow_single_scattering_albedo_direct_radius_band: jnp.ndarray
    snow_asymmetry_parameter_direct_radius_band: jnp.ndarray
    snow_mass_extinction_coefficient_direct_radius_band: jnp.ndarray
    snow_single_scattering_albedo_diffuse_radius_band: jnp.ndarray
    snow_asymmetry_parameter_diffuse_radius_band: jnp.ndarray
    snow_mass_extinction_coefficient_diffuse_radius_band: jnp.ndarray
    aerosol_single_scattering_albedo_species_band: jnp.ndarray
    aerosol_asymmetry_parameter_species_band: jnp.ndarray
    aerosol_mass_extinction_coefficient_species_band: jnp.ndarray


class SnicarOpticalInputs(NamedTuple):

    nonurban_column: jnp.ndarray
    cosine_solar_zenith_column: jnp.ndarray
    snow_layer_count_column: jnp.ndarray
    snow_water_equivalent_column: jnp.ndarray
    snow_liquid_column_layer: jnp.ndarray
    snow_ice_column_layer: jnp.ndarray
    snow_grain_radius_column_layer: jnp.ndarray
    aerosol_mass_concentration_column_layer_species: jnp.ndarray


class SnicarOpticalLayerState(NamedTuple):

    snow_mass_column_layer: jnp.ndarray
    snow_optical_depth_column_layer_band: jnp.ndarray
    aerosol_mass_column_layer_species_band: jnp.ndarray
    aerosol_optical_depth_column_layer_species_band: jnp.ndarray
    optical_depth_column_layer_band: jnp.ndarray
    single_scattering_albedo_column_layer_band: jnp.ndarray
    asymmetry_parameter_column_layer_band: jnp.ndarray
    transformed_optical_depth_column_layer_band: jnp.ndarray
    transformed_single_scattering_albedo_column_layer_band: jnp.ndarray
    transformed_asymmetry_parameter_column_layer_band: jnp.ndarray
    cumulative_transformed_optical_depth_column_layer_band: jnp.ndarray


class SnicarOpticalDiagnostics(NamedTuple):

    qualified_column: jnp.ndarray
    synthetic_snow_layer_column: jnp.ndarray
    active_snow_layer_column_layer: jnp.ndarray
    effective_snow_grain_radius_column_layer: jnp.ndarray
    invalid_active_snow_grain_radius_column_layer: jnp.ndarray
    spectral_weight_band: jnp.ndarray


class SnicarOpticalResult(NamedTuple):

    state: SnicarOpticalLayerState
    diagnostics: SnicarOpticalDiagnostics


class SnicarRadiativeTransferInputs(NamedTuple):

    optical: SnicarOpticalInputs
    underlying_surface_albedo_column_band: jnp.ndarray


class SnicarRadiativeTransferState(NamedTuple):

    snow_albedo_column_band: jnp.ndarray
    absorbed_flux_column_level_band: jnp.ndarray


class SnicarRadiativeTransferDiagnostics(NamedTuple):

    spectral_snow_albedo_column_band: jnp.ndarray
    spectral_absorbed_flux_column_level_band: jnp.ndarray
    final_cosine_solar_zenith_column_band: jnp.ndarray
    final_approximation_type_column_band: jnp.ndarray
    attempt_count_column_band: jnp.ndarray
    final_trip_column_band: jnp.ndarray
    fatal_retry_failure_column_band: jnp.ndarray
    energy_residual_column_band: jnp.ndarray
    energy_conservation_failure_column_band: jnp.ndarray


class SnicarRadiativeTransferResult(NamedTuple):

    state: SnicarRadiativeTransferState
    optical: SnicarOpticalResult
    diagnostics: SnicarRadiativeTransferDiagnostics


class SnicarRuntimeDiagnostics(NamedTuple):
    fatal_retry_failure_count: jnp.ndarray
    energy_conservation_failure_count: jnp.ndarray
    invalid_active_snow_grain_radius_count: jnp.ndarray
    maximum_attempt_count: jnp.ndarray
    maximum_absolute_energy_residual: jnp.ndarray


class SnicarRuntimeResult(NamedTuple):

    state: SnicarRadiativeTransferState
    diagnostics: SnicarRuntimeDiagnostics
    all_finite: jnp.ndarray


def _summarize_snicar(result):
    finite = jnp.asarray(True)
    for value in tree_util.tree_leaves(result):
        if jnp.issubdtype(value.dtype, jnp.inexact):
            finite = finite & jnp.all(jnp.isfinite(value))
    diagnostics = result.diagnostics
    return SnicarRuntimeResult(result.state, SnicarRuntimeDiagnostics(
        jnp.sum(diagnostics.fatal_retry_failure_column_band, dtype=jnp.int32),
        jnp.sum(diagnostics.energy_conservation_failure_column_band, dtype=jnp.int32),
        jnp.sum(result.optical.diagnostics.invalid_active_snow_grain_radius_column_layer,
                dtype=jnp.int32),
        jnp.max(diagnostics.attempt_count_column_band, initial=0),
        jnp.max(jnp.abs(diagnostics.energy_residual_column_band), initial=0.),
    ), finite)


class _SnicarSpectralAttempt(NamedTuple):
    albedo: jnp.ndarray
    absorbed_flux: jnp.ndarray
    trip: jnp.ndarray
    energy_residual: jnp.ndarray


def _validate_shapes(inputs: SnicarOpticalInputs, tables: SnowOpticalTables) -> None:
    if inputs.snow_layer_count_column.ndim != 1:
        raise ValueError("snow_layer_count_column must be a vector")
    column_count = inputs.snow_layer_count_column.shape[0]
    column_shape = (column_count,)
    layer_shape = (column_count, NUMBER_OF_SNOW_LAYERS)
    for name in (
        "nonurban_column",
        "cosine_solar_zenith_column",
        "snow_water_equivalent_column",
    ):
        if getattr(inputs, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")
    for name in (
        "snow_liquid_column_layer",
        "snow_ice_column_layer",
        "snow_grain_radius_column_layer",
    ):
        if getattr(inputs, name).shape != layer_shape:
            raise ValueError(f"{name} must have shape {layer_shape}")
    aerosol_shape = (
        column_count,
        NUMBER_OF_SNOW_LAYERS,
        NUMBER_OF_SNOW_AEROSOLS,
    )
    if inputs.aerosol_mass_concentration_column_layer_species.shape != aerosol_shape:
        raise ValueError(
            f"aerosol_mass_concentration_column_layer_species must have shape {aerosol_shape}"
        )

    snow_table_shape = (
        NUMBER_OF_SNOW_GRAIN_RADII,
        NUMBER_OF_SNOW_SPECTRAL_BANDS,
    )
    for name in SnowOpticalTables._fields[:6]:
        if getattr(tables, name).shape != snow_table_shape:
            raise ValueError(f"tables.{name} must have shape {snow_table_shape}")
    aerosol_table_shape = (
        NUMBER_OF_SNOW_AEROSOLS,
        NUMBER_OF_SNOW_SPECTRAL_BANDS,
    )
    for name in SnowOpticalTables._fields[6:]:
        if getattr(tables, name).shape != aerosol_table_shape:
            raise ValueError(f"tables.{name} must have shape {aerosol_table_shape}")


def snicar_optical_properties(
    inputs: SnicarOpticalInputs,
    tables: SnowOpticalTables,
    direct_beam: bool | jnp.ndarray,
) -> SnicarOpticalResult:

    _validate_shapes(inputs, tables)
    direct = jnp.asarray(direct_beam, dtype=jnp.bool_)
    nonurban = jnp.asarray(inputs.nonurban_column, dtype=jnp.bool_)
    cosine_solar_zenith = jnp.asarray(
        inputs.cosine_solar_zenith_column,
        dtype=jnp.float64,
    )
    snow_layer_count = jnp.asarray(
        inputs.snow_layer_count_column,
        dtype=jnp.int32,
    )
    snow_water_equivalent = jnp.asarray(
        inputs.snow_water_equivalent_column,
        dtype=jnp.float64,
    )
    snow_liquid_input = jnp.asarray(
        inputs.snow_liquid_column_layer,
        dtype=jnp.float64,
    )
    snow_ice_input = jnp.asarray(
        inputs.snow_ice_column_layer,
        dtype=jnp.float64,
    )
    grain_radius_input = jnp.asarray(
        inputs.snow_grain_radius_column_layer,
        dtype=jnp.int32,
    )
    aerosol_concentration_input = jnp.asarray(
        inputs.aerosol_mass_concentration_column_layer_species,
        dtype=jnp.float64,
    )

    qualified = (
        nonurban
        & (cosine_solar_zenith > jnp.float64(0.0))
        & (snow_water_equivalent > jnp.float64(MINIMUM_SNOW_WATER_EQUIVALENT))
    )
    synthetic = qualified & (snow_layer_count > jnp.int32(-1))
    local_snow_layer_count = jnp.where(
        synthetic,
        jnp.int32(-1),
        snow_layer_count,
    )
    top_layer_index = jnp.clip(
        jnp.int32(NUMBER_OF_SNOW_LAYERS) + local_snow_layer_count,
        jnp.int32(0),
        jnp.int32(NUMBER_OF_SNOW_LAYERS - 1),
    )
    layer_index = jnp.arange(NUMBER_OF_SNOW_LAYERS, dtype=jnp.int32)
    active_layer = qualified[:, None] & (layer_index[None, :] >= top_layer_index[:, None])
    bottom_layer = layer_index[None, :] == jnp.int32(NUMBER_OF_SNOW_LAYERS - 1)
    synthetic_bottom = synthetic[:, None] & bottom_layer

    local_ice = jnp.where(
        synthetic_bottom,
        snow_water_equivalent[:, None],
        snow_ice_input,
    )
    local_liquid = jnp.where(
        synthetic_bottom,
        jnp.float64(0.0),
        snow_liquid_input,
    )
    effective_radius = jnp.where(
        synthetic_bottom,
        jnp.int32(FRESH_SNOW_GRAIN_RADIUS_MICRONS),
        grain_radius_input,
    )
    invalid_radius = active_layer & (
        (effective_radius < jnp.int32(MINIMUM_SNOW_GRAIN_RADIUS_MICRONS))
        | (effective_radius > jnp.int32(MAXIMUM_SNOW_GRAIN_RADIUS_MICRONS))
    )

    safe_ice = jnp.where(active_layer, local_ice, jnp.float64(1.0))
    safe_liquid = jnp.where(active_layer, local_liquid, jnp.float64(0.0))
    safe_radius = jnp.clip(
        effective_radius,
        jnp.int32(MINIMUM_SNOW_GRAIN_RADIUS_MICRONS),
        jnp.int32(MAXIMUM_SNOW_GRAIN_RADIUS_MICRONS),
    )
    radius_index = safe_radius - jnp.int32(MINIMUM_SNOW_GRAIN_RADIUS_MICRONS)

    snow_ssa_table = jnp.where(
        direct,
        jnp.asarray(
            tables.snow_single_scattering_albedo_direct_radius_band,
            dtype=jnp.float64,
        ),
        jnp.asarray(
            tables.snow_single_scattering_albedo_diffuse_radius_band,
            dtype=jnp.float64,
        ),
    )
    snow_asymmetry_table = jnp.where(
        direct,
        jnp.asarray(
            tables.snow_asymmetry_parameter_direct_radius_band,
            dtype=jnp.float64,
        ),
        jnp.asarray(
            tables.snow_asymmetry_parameter_diffuse_radius_band,
            dtype=jnp.float64,
        ),
    )
    snow_extinction_table = jnp.where(
        direct,
        jnp.asarray(
            tables.snow_mass_extinction_coefficient_direct_radius_band,
            dtype=jnp.float64,
        ),
        jnp.asarray(
            tables.snow_mass_extinction_coefficient_diffuse_radius_band,
            dtype=jnp.float64,
        ),
    )
    snow_ssa = snow_ssa_table[radius_index, :]
    snow_asymmetry = snow_asymmetry_table[radius_index, :]
    snow_extinction = snow_extinction_table[radius_index, :]
    aerosol_ssa = jnp.asarray(
        tables.aerosol_single_scattering_albedo_species_band,
        dtype=jnp.float64,
    )
    aerosol_asymmetry = jnp.asarray(
        tables.aerosol_asymmetry_parameter_species_band,
        dtype=jnp.float64,
    )
    aerosol_extinction = jnp.asarray(
        tables.aerosol_mass_extinction_coefficient_species_band,
        dtype=jnp.float64,
    )

    snow_mass_safe = fortran_add(safe_ice, safe_liquid)
    snow_optical_depth_safe = fortran_multiply(
        snow_mass_safe[:, :, None],
        snow_extinction,
    )
    aerosol_enabled_band = jnp.arange(NUMBER_OF_SNOW_SPECTRAL_BANDS) < 3
    aerosol_concentration = jnp.where(
        aerosol_enabled_band[None, None, None, :],
        aerosol_concentration_input[:, :, :, None],
        jnp.float64(0.0),
    )
    aerosol_concentration = jnp.where(
        active_layer[:, :, None, None],
        aerosol_concentration,
        jnp.float64(0.0),
    )
    aerosol_mass_safe = fortran_multiply(
        snow_mass_safe[:, :, None, None],
        aerosol_concentration,
    )
    aerosol_optical_depth_safe = fortran_multiply(
        aerosol_mass_safe,
        aerosol_extinction[None, None, :, :],
    )

    mixed_shape = (
        snow_layer_count.shape[0],
        NUMBER_OF_SNOW_LAYERS,
        NUMBER_OF_SNOW_SPECTRAL_BANDS,
    )
    tau_sum = jnp.zeros(mixed_shape, dtype=jnp.float64)
    omega_sum = jnp.zeros(mixed_shape, dtype=jnp.float64)
    asymmetry_sum = jnp.zeros(mixed_shape, dtype=jnp.float64)
    for species in range(NUMBER_OF_SNOW_AEROSOLS):
        species_tau = aerosol_optical_depth_safe[:, :, species, :]
        tau_sum = fortran_add(tau_sum, species_tau)
        species_scattering = fortran_multiply(species_tau, aerosol_ssa[species, :])
        omega_sum = fortran_add(omega_sum, species_scattering)
        species_asymmetry = fortran_multiply(
            species_scattering,
            aerosol_asymmetry[species, :],
        )
        asymmetry_sum = fortran_add(asymmetry_sum, species_asymmetry)

    optical_depth_safe = fortran_add(tau_sum, snow_optical_depth_safe)
    snow_scattering = fortran_multiply(snow_ssa, snow_optical_depth_safe)
    scattering_numerator = fortran_add(omega_sum, snow_scattering)
    inverse_optical_depth = fortran_divide(
        jnp.float64(1.0),
        optical_depth_safe,
    )
    single_scattering_albedo_safe = fortran_multiply(
        inverse_optical_depth,
        scattering_numerator,
    )
    snow_asymmetry_scattering = fortran_multiply(snow_asymmetry, snow_ssa)
    snow_asymmetry_scattering = fortran_multiply(
        snow_asymmetry_scattering,
        snow_optical_depth_safe,
    )
    asymmetry_numerator = fortran_add(
        asymmetry_sum,
        snow_asymmetry_scattering,
    )
    scattering_optical_depth = fortran_multiply(
        optical_depth_safe,
        single_scattering_albedo_safe,
    )
    inverse_scattering_optical_depth = fortran_divide(
        jnp.float64(1.0),
        scattering_optical_depth,
    )
    asymmetry_parameter_safe = fortran_multiply(
        inverse_scattering_optical_depth,
        asymmetry_numerator,
    )

    one = jnp.float64(1.0)
    asymmetry_squared = fortran_multiply(
        asymmetry_parameter_safe,
        asymmetry_parameter_safe,
    )
    transformed_asymmetry_safe = fortran_divide(
        asymmetry_parameter_safe,
        fortran_add(one, asymmetry_parameter_safe),
    )
    one_minus_asymmetry_squared = fortran_add(one, -asymmetry_squared)
    transformed_ssa_numerator = fortran_multiply(
        one_minus_asymmetry_squared,
        single_scattering_albedo_safe,
    )
    removed_forward_scattering = fortran_multiply(
        single_scattering_albedo_safe,
        asymmetry_squared,
    )
    delta_denominator = fortran_add(one, -removed_forward_scattering)
    transformed_ssa_safe = fortran_divide(
        transformed_ssa_numerator,
        delta_denominator,
    )
    transformed_optical_depth_safe = fortran_multiply(
        delta_denominator,
        optical_depth_safe,
    )

    cumulative_values = [jnp.zeros_like(transformed_optical_depth_safe[:, 0, :])]
    for layer in range(1, NUMBER_OF_SNOW_LAYERS):
        has_layer_above = active_layer[:, layer] & active_layer[:, layer - 1]
        next_value = fortran_add(
            cumulative_values[-1],
            transformed_optical_depth_safe[:, layer - 1, :],
        )
        cumulative_values.append(jnp.where(has_layer_above[:, None], next_value, jnp.float64(0.0)))
    cumulative_safe = jnp.stack(cumulative_values, axis=1)

    layer_mask = active_layer[:, :, None]
    aerosol_layer_mask = active_layer[:, :, None, None]

    def mask_layer(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(layer_mask, value, jnp.float64(0.0))

    state = SnicarOpticalLayerState(
        snow_mass_column_layer=jnp.where(
            active_layer,
            snow_mass_safe,
            jnp.float64(0.0),
        ),
        snow_optical_depth_column_layer_band=mask_layer(snow_optical_depth_safe),
        aerosol_mass_column_layer_species_band=jnp.where(
            aerosol_layer_mask,
            aerosol_mass_safe,
            jnp.float64(0.0),
        ),
        aerosol_optical_depth_column_layer_species_band=jnp.where(
            aerosol_layer_mask,
            aerosol_optical_depth_safe,
            jnp.float64(0.0),
        ),
        optical_depth_column_layer_band=mask_layer(optical_depth_safe),
        single_scattering_albedo_column_layer_band=mask_layer(single_scattering_albedo_safe),
        asymmetry_parameter_column_layer_band=mask_layer(asymmetry_parameter_safe),
        transformed_optical_depth_column_layer_band=mask_layer(transformed_optical_depth_safe),
        transformed_single_scattering_albedo_column_layer_band=mask_layer(transformed_ssa_safe),
        transformed_asymmetry_parameter_column_layer_band=mask_layer(transformed_asymmetry_safe),
        cumulative_transformed_optical_depth_column_layer_band=mask_layer(cumulative_safe),
    )
    diagnostics = SnicarOpticalDiagnostics(
        qualified_column=qualified,
        synthetic_snow_layer_column=synthetic,
        active_snow_layer_column_layer=active_layer,
        effective_snow_grain_radius_column_layer=jnp.where(
            active_layer,
            effective_radius,
            jnp.int32(0),
        ),
        invalid_active_snow_grain_radius_column_layer=invalid_radius,
        spectral_weight_band=jnp.where(
            direct,
            DIRECT_SPECTRAL_WEIGHTS,
            DIFFUSE_SPECTRAL_WEIGHTS,
        ),
    )
    return SnicarOpticalResult(state=state, diagnostics=diagnostics)


def _layer_at(values: jnp.ndarray, layer_index: jnp.ndarray) -> jnp.ndarray:

    indices = jnp.broadcast_to(
        layer_index[:, None, None],
        (values.shape[0], 1, values.shape[2]),
    )
    return jnp.take_along_axis(values, indices, axis=1)[:, 0, :]


def _snicar_spectral_attempt(
    optical: SnicarOpticalResult,
    underlying_albedo_spectral: jnp.ndarray,
    cosine_solar_zenith: jnp.ndarray,
    approximation_type: jnp.ndarray,
    direct_beam: jnp.ndarray,
) -> _SnicarSpectralAttempt:

    add = fortran_add
    multiply = fortran_multiply
    divide = fortran_divide
    one = jnp.float64(1.0)
    two = jnp.float64(2.0)
    three = jnp.float64(3.0)
    four = jnp.float64(4.0)
    seven = jnp.float64(7.0)
    pi = jnp.float64(3.14159265358979323846)
    square_root_three = jnp.float64(1.73205077648162841796875)

    state = optical.state
    active_layer = optical.diagnostics.active_snow_layer_column_layer
    qualified = optical.diagnostics.qualified_column
    tau_star = state.transformed_optical_depth_column_layer_band
    omega_star = state.transformed_single_scattering_albedo_column_layer_band
    asymmetry_star = state.transformed_asymmetry_parameter_column_layer_band
    tau_column = state.cumulative_transformed_optical_depth_column_layer_band
    mu = jnp.asarray(cosine_solar_zenith, dtype=jnp.float64)

    three_g = multiply(three, asymmetry_star)
    eddington_gamma1 = divide(
        add(seven, -multiply(omega_star, add(four, three_g))),
        four,
    )
    eddington_gamma2 = -divide(
        add(one, -multiply(omega_star, add(four, -three_g))),
        four,
    )
    eddington_gamma3 = divide(
        add(two, -multiply(three_g, mu[:, None, :])),
        four,
    )
    eddington_gamma4 = add(one, -eddington_gamma3)

    hemispheric_gamma1 = add(
        two,
        -multiply(omega_star, add(one, asymmetry_star)),
    )
    hemispheric_gamma2 = multiply(
        omega_star,
        add(one, -asymmetry_star),
    )
    hemispheric_gamma3 = divide(
        add(
            one,
            -multiply(
                multiply(square_root_three, asymmetry_star),
                mu[:, None, :],
            ),
        ),
        two,
    )
    hemispheric_gamma4 = add(one, -hemispheric_gamma3)
    eddington = approximation_type[:, None, :] == jnp.int32(1)
    gamma1 = jnp.where(eddington, eddington_gamma1, hemispheric_gamma1)
    gamma2 = jnp.where(eddington, eddington_gamma2, hemispheric_gamma2)
    gamma3 = jnp.where(eddington, eddington_gamma3, hemispheric_gamma3)
    gamma4 = jnp.where(eddington, eddington_gamma4, hemispheric_gamma4)

    lambda_squared = add(
        multiply(gamma1, gamma1),
        -multiply(gamma2, gamma2),
    )
    attenuation_coefficient = fortran_materialize(jnp.sqrt(jnp.abs(lambda_squared)))
    capital_gamma = divide(
        gamma2,
        add(gamma1, attenuation_coefficient),
    )
    layer_exponential = strict_negative_exp(-multiply(attenuation_coefficient, tau_star))
    gamma_exponential = multiply(capital_gamma, layer_exponential)
    e1 = add(one, gamma_exponential)
    e2 = add(one, -gamma_exponential)
    e3 = add(capital_gamma, layer_exponential)
    e4 = add(capital_gamma, -layer_exponential)

    mu_pi = multiply(mu, pi)
    direct_irradiance = divide(one, mu_pi)
    flux_direct = jnp.where(direct_beam, direct_irradiance, jnp.float64(0.0))
    flux_diffuse = jnp.where(direct_beam, jnp.float64(0.0), one)
    total_optical_depth = add(tau_column, tau_star)
    direct_exponential_bottom = strict_negative_exp(-divide(total_optical_depth, mu[:, None, :]))
    direct_exponential_top = strict_negative_exp(-divide(tau_column, mu[:, None, :]))

    bottom_layer = NUMBER_OF_SNOW_LAYERS - 1
    bottom_direct_reflection = multiply(
        multiply(
            multiply(
                multiply(
                    underlying_albedo_spectral,
                    mu,
                ),
                direct_exponential_bottom[:, bottom_layer, :],
            ),
            pi,
        ),
        flux_direct,
    )

    inverse_mu = divide(one, mu)
    inverse_mu_squared = divide(one, multiply(mu, mu))
    c_denominator = add(
        multiply(attenuation_coefficient, attenuation_coefficient),
        -inverse_mu_squared[:, None, :],
    )
    plus_bracket = add(
        multiply(add(gamma1, -inverse_mu[:, None, :]), gamma3),
        multiply(gamma4, gamma2),
    )
    minus_bracket = add(
        multiply(add(gamma1, inverse_mu[:, None, :]), gamma4),
        multiply(gamma2, gamma3),
    )
    c_common = multiply(multiply(omega_star, pi), flux_direct[:, None, :])
    c_plus_bottom = divide(
        multiply(
            multiply(c_common, direct_exponential_bottom),
            plus_bracket,
        ),
        c_denominator,
    )
    c_minus_bottom = divide(
        multiply(
            multiply(c_common, direct_exponential_bottom),
            minus_bracket,
        ),
        c_denominator,
    )
    c_plus_top = divide(
        multiply(multiply(c_common, direct_exponential_top), plus_bracket),
        c_denominator,
    )
    c_minus_top = divide(
        multiply(multiply(c_common, direct_exponential_top), minus_bracket),
        c_denominator,
    )
    c_plus_bottom = jnp.where(direct_beam, c_plus_bottom, jnp.float64(0.0))
    c_minus_bottom = jnp.where(direct_beam, c_minus_bottom, jnp.float64(0.0))
    c_plus_top = jnp.where(direct_beam, c_plus_top, jnp.float64(0.0))
    c_minus_top = jnp.where(direct_beam, c_minus_top, jnp.float64(0.0))

    layer_count = jnp.sum(active_layer.astype(jnp.int32), axis=1)
    layer_count = jnp.where(qualified, layer_count, jnp.int32(1))
    top_layer = jnp.int32(NUMBER_OF_SNOW_LAYERS) - layer_count
    matrix_start = jnp.int32(2 * NUMBER_OF_SNOW_LAYERS) - jnp.int32(2) * layer_count
    top_e1 = _layer_at(e1, top_layer)
    top_e2 = _layer_at(e2, top_layer)
    top_c_minus = _layer_at(c_minus_top, top_layer)
    bottom_e1 = e1[:, bottom_layer, :]
    bottom_e2 = e2[:, bottom_layer, :]
    bottom_e3 = e3[:, bottom_layer, :]
    bottom_e4 = e4[:, bottom_layer, :]
    bottom_c_plus = c_plus_bottom[:, bottom_layer, :]
    bottom_c_minus = c_minus_bottom[:, bottom_layer, :]

    matrix_a: list[jnp.ndarray] = []
    matrix_b: list[jnp.ndarray] = []
    matrix_d: list[jnp.ndarray] = []
    matrix_e: list[jnp.ndarray] = []
    for slot in range(2 * NUMBER_OF_SNOW_LAYERS):
        native_index = slot - (2 * NUMBER_OF_SNOW_LAYERS - 1)
        active_matrix = slot >= matrix_start[:, None]
        top_boundary = slot == matrix_start[:, None]
        bottom_boundary = native_index == 0
        native_layer = native_index // 2
        layer = min(max(native_layer + NUMBER_OF_SNOW_LAYERS - 1, 0), 10)
        next_layer = layer + 1

        if native_index < 0 and native_index % 2 != 0:
            internal_a = add(
                multiply(e2[:, layer, :], e3[:, layer, :]),
                -multiply(e4[:, layer, :], e1[:, layer, :]),
            )
            internal_b = add(
                multiply(e1[:, layer, :], e1[:, next_layer, :]),
                -multiply(e3[:, layer, :], e3[:, next_layer, :]),
            )
            internal_d = add(
                multiply(e3[:, layer, :], e4[:, next_layer, :]),
                -multiply(e1[:, layer, :], e2[:, next_layer, :]),
            )
            internal_e = add(
                multiply(
                    e3[:, layer, :],
                    add(c_plus_top[:, next_layer, :], -c_plus_bottom[:, layer, :]),
                ),
                multiply(
                    e1[:, layer, :],
                    add(c_minus_bottom[:, layer, :], -c_minus_top[:, next_layer, :]),
                ),
            )
        else:
            internal_a = add(
                multiply(e2[:, next_layer, :], e1[:, layer, :]),
                -multiply(e3[:, layer, :], e4[:, next_layer, :]),
            )
            internal_b = add(
                multiply(e2[:, layer, :], e2[:, next_layer, :]),
                -multiply(e4[:, layer, :], e4[:, next_layer, :]),
            )
            internal_d = add(
                multiply(e1[:, next_layer, :], e4[:, next_layer, :]),
                -multiply(e2[:, next_layer, :], e3[:, next_layer, :]),
            )
            internal_e = add(
                multiply(
                    e2[:, next_layer, :],
                    add(c_plus_top[:, next_layer, :], -c_plus_bottom[:, layer, :]),
                ),
                multiply(
                    e4[:, next_layer, :],
                    add(c_minus_top[:, next_layer, :], -c_minus_bottom[:, layer, :]),
                ),
            )

        top_a = jnp.zeros_like(top_e1)
        top_b = top_e1
        top_d = -top_e2
        top_rhs = add(flux_diffuse, -top_c_minus)
        bottom_a = add(
            bottom_e1,
            -multiply(underlying_albedo_spectral, bottom_e3),
        )
        bottom_b = add(
            bottom_e2,
            -multiply(underlying_albedo_spectral, bottom_e4),
        )
        bottom_d = jnp.zeros_like(bottom_a)
        bottom_rhs = add(
            add(bottom_direct_reflection, -bottom_c_plus),
            multiply(underlying_albedo_spectral, bottom_c_minus),
        )
        selected_a = jnp.where(top_boundary, top_a, internal_a)
        selected_b = jnp.where(top_boundary, top_b, internal_b)
        selected_d = jnp.where(top_boundary, top_d, internal_d)
        selected_e = jnp.where(top_boundary, top_rhs, internal_e)
        if bottom_boundary:
            selected_a = bottom_a
            selected_b = bottom_b
            selected_d = bottom_d
            selected_e = bottom_rhs
        matrix_a.append(jnp.where(active_matrix, selected_a, jnp.float64(0.0)))
        matrix_b.append(jnp.where(active_matrix, selected_b, jnp.float64(1.0)))
        matrix_d.append(jnp.where(active_matrix, selected_d, jnp.float64(0.0)))
        matrix_e.append(jnp.where(active_matrix, selected_e, jnp.float64(0.0)))

    matrix_a_array = jnp.stack(matrix_a, axis=2)
    matrix_b_array = jnp.stack(matrix_b, axis=2)
    matrix_d_array = jnp.stack(matrix_d, axis=2)
    matrix_e_array = jnp.stack(matrix_e, axis=2)
    matrix_size = 2 * NUMBER_OF_SNOW_LAYERS
    as_values: list[jnp.ndarray | None] = [None] * matrix_size
    ds_values: list[jnp.ndarray | None] = [None] * matrix_size
    as_values[-1] = divide(matrix_a_array[:, :, -1], matrix_b_array[:, :, -1])
    ds_values[-1] = divide(matrix_e_array[:, :, -1], matrix_b_array[:, :, -1])
    for slot in range(matrix_size - 2, -1, -1):
        denominator = add(
            matrix_b_array[:, :, slot],
            -multiply(matrix_d_array[:, :, slot], as_values[slot + 1]),
        )
        inverse = divide(one, denominator)
        as_values[slot] = multiply(matrix_a_array[:, :, slot], inverse)
        ds_values[slot] = multiply(
            add(
                matrix_e_array[:, :, slot],
                -multiply(matrix_d_array[:, :, slot], ds_values[slot + 1]),
            ),
            inverse,
        )
    as_array = jnp.stack(as_values, axis=2)
    ds_array = jnp.stack(ds_values, axis=2)
    y_values = [jnp.where(matrix_start[:, None] == 0, ds_array[:, :, 0], jnp.float64(0.0))]
    for slot in range(1, matrix_size):
        solved = add(
            ds_array[:, :, slot],
            -multiply(as_array[:, :, slot], y_values[-1]),
        )
        y_values.append(
            jnp.where(
                slot == matrix_start[:, None],
                ds_array[:, :, slot],
                jnp.where(slot > matrix_start[:, None], solved, jnp.float64(0.0)),
            )
        )
    y = jnp.stack(y_values, axis=2)

    y_first = jnp.stack(
        tuple(y[:, :, 2 * layer] for layer in range(NUMBER_OF_SNOW_LAYERS)),
        axis=1,
    )
    y_second = jnp.stack(
        tuple(y[:, :, 2 * layer + 1] for layer in range(NUMBER_OF_SNOW_LAYERS)),
        axis=1,
    )
    direct_flux = multiply(
        multiply(multiply(mu[:, None, :], pi), flux_direct[:, None, :]),
        direct_exponential_bottom,
    )
    net_flux = add(
        add(
            add(
                multiply(y_first, add(e1, -e3)),
                multiply(y_second, add(e2, -e4)),
            ),
            c_plus_bottom,
        ),
        -c_minus_bottom,
    )
    net_flux = add(net_flux, -direct_flux)
    surface_y_first = jnp.take_along_axis(
        y,
        jnp.broadcast_to(matrix_start[:, None, None], (y.shape[0], y.shape[1], 1)),
        axis=2,
    )[:, :, 0]
    surface_y_second = jnp.take_along_axis(
        y,
        jnp.broadcast_to(
            (matrix_start + jnp.int32(1))[:, None, None],
            (y.shape[0], y.shape[1], 1),
        ),
        axis=2,
    )[:, :, 0]
    top_exponential = _layer_at(layer_exponential, top_layer)
    top_capital_gamma = _layer_at(capital_gamma, top_layer)
    top_c_plus = _layer_at(c_plus_top, top_layer)
    surface_upward_flux = add(
        add(
            multiply(surface_y_first, add(top_exponential, top_capital_gamma)),
            multiply(surface_y_second, add(top_exponential, -top_capital_gamma)),
        ),
        top_c_plus,
    )
    bottom_net_flux = -net_flux[:, bottom_layer, :]
    incident_flux = add(multiply(multiply(mu, pi), flux_direct), flux_diffuse)
    albedo = divide(surface_upward_flux, incident_flux)
    surface_net_flux = add(surface_upward_flux, -incident_flux)

    raw_absorbed_values: list[jnp.ndarray] = []
    for layer in range(NUMBER_OF_SNOW_LAYERS):
        top = layer == top_layer[:, None]
        if layer == 0:
            layer_absorption = add(net_flux[:, layer, :], -surface_net_flux)
        else:
            layer_absorption = jnp.where(
                top,
                add(net_flux[:, layer, :], -surface_net_flux),
                add(net_flux[:, layer, :], -net_flux[:, layer - 1, :]),
            )
        raw_absorbed_values.append(
            jnp.where(active_layer[:, layer, None], layer_absorption, jnp.float64(0.0))
        )
    raw_absorbed = jnp.stack(raw_absorbed_values, axis=1)
    negative_absorption = jnp.any(
        active_layer[:, :, None] & (raw_absorbed < jnp.float64(-0.00001)),
        axis=1,
    )
    absorbed_sum = jnp.zeros_like(albedo)
    for layer in range(NUMBER_OF_SNOW_LAYERS):
        absorbed_sum = add(absorbed_sum, raw_absorbed[:, layer, :])
    trip = negative_absorption | (absorbed_sum > one)
    trip = trip | ((albedo < jnp.float64(0.0)) & ~trip)
    absorbed_flux = jnp.concatenate(
        (
            jnp.where(
                raw_absorbed < jnp.float64(0.0),
                jnp.float64(0.0),
                raw_absorbed,
            ),
            jnp.where(
                bottom_net_flux[:, None, :] < jnp.float64(0.0),
                jnp.float64(0.0),
                bottom_net_flux[:, None, :],
            ),
        ),
        axis=1,
    )
    energy_residual = add(
        incident_flux,
        -add(add(absorbed_sum, bottom_net_flux), surface_upward_flux),
    )
    qualified_band = qualified[:, None]
    return _SnicarSpectralAttempt(
        albedo=jnp.where(qualified_band, albedo, jnp.float64(0.0)),
        absorbed_flux=jnp.where(
            qualified[:, None, None],
            absorbed_flux,
            jnp.float64(0.0),
        ),
        trip=qualified_band & trip,
        energy_residual=jnp.where(
            qualified_band,
            energy_residual,
            jnp.float64(0.0),
        ),
    )


def _sparse_spectral_attempt(
    optical, underlying, cosine, approximation, direct, running,
    *, capacities=(2048, 32768),
):
    active = jnp.any(running, axis=1)
    count = active.shape[0]
    capacities = tuple(size for size in capacities if size < count)
    active_count = jnp.count_nonzero(active)

    def complete():
        return _snicar_spectral_attempt(optical, underlying, cosine, approximation, direct)

    def compact(capacity):
        indices = jnp.nonzero(active, size=capacity, fill_value=count)[0]
        valid = indices < count
        safe = jnp.where(valid, indices, indices[0])

        packed = SnicarOpticalResult(
            tree_util.tree_map(lambda value: value[safe], optical.state),
            SnicarOpticalDiagnostics(*(
                value if name == "spectral_weight_band" else value[safe]
                for name, value in zip(
                    optical.diagnostics._fields, optical.diagnostics, strict=True,
                )
            )),
        )
        result = _snicar_spectral_attempt(
            packed, underlying[safe], cosine[safe], approximation[safe], direct,
        )

        def scatter(value):
            shape = (count, *value.shape[1:])
            return jnp.zeros(shape, value.dtype).at[indices].set(value, mode="drop")

        return tree_util.tree_map(scatter, result)

    def dispatch(position):
        if position == len(capacities):
            return complete()
        capacity = capacities[position]
        return lax.cond(
            (active_count > 0) & (active_count <= capacity),
            lambda: compact(capacity), lambda: dispatch(position + 1),
        )

    return dispatch(0)


def _snicar_radiative_transfer_dense(
    inputs: SnicarRadiativeTransferInputs,
    tables: SnowOpticalTables,
    direct_beam: bool | jnp.ndarray,
    *,
    early_exit_retry_loop: bool = False,
    compact_active_columns: bool = False,
) -> SnicarRadiativeTransferResult:

    optical_inputs = inputs.optical
    optical = snicar_optical_properties(optical_inputs, tables, direct_beam)
    column_count = optical_inputs.snow_layer_count_column.shape[0]
    if inputs.underlying_surface_albedo_column_band.shape != (column_count, 2):
        raise ValueError(
            f"underlying_surface_albedo_column_band must have shape {(column_count, 2)}"
        )
    direct = jnp.asarray(direct_beam, dtype=jnp.bool_)
    underlying_broadband = jnp.asarray(
        inputs.underlying_surface_albedo_column_band,
        dtype=jnp.float64,
    )
    underlying_spectral = jnp.concatenate(
        (
            underlying_broadband[:, :1],
            jnp.broadcast_to(
                underlying_broadband[:, 1:2],
                (column_count, NUMBER_OF_SNOW_SPECTRAL_BANDS - 1),
            ),
        ),
        axis=1,
    )
    qualified = optical.diagnostics.qualified_column
    original_cosine = jnp.asarray(
        optical_inputs.cosine_solar_zenith_column,
        dtype=jnp.float64,
    )
    safe_cosine = jnp.where(qualified, original_cosine, jnp.float64(0.7))
    cosine = jnp.broadcast_to(
        safe_cosine[:, None],
        (column_count, NUMBER_OF_SNOW_SPECTRAL_BANDS),
    )
    running = jnp.broadcast_to(
        qualified[:, None],
        (column_count, NUMBER_OF_SNOW_SPECTRAL_BANDS),
    )
    do_over = jnp.ones_like(running, dtype=jnp.int32)
    error_index = jnp.zeros_like(do_over)
    attempt_count = jnp.zeros_like(do_over)
    approximation = jnp.ones_like(do_over)
    spectral_albedo = jnp.zeros_like(cosine)
    spectral_absorption = jnp.zeros(
        (column_count, NUMBER_OF_SNOW_LAYERS + 1, NUMBER_OF_SNOW_SPECTRAL_BANDS),
        dtype=jnp.float64,
    )
    final_trip = jnp.zeros_like(running)
    fatal_failure = jnp.zeros_like(running)
    energy_residual = jnp.zeros_like(cosine)
    visible_band = jnp.arange(NUMBER_OF_SNOW_SPECTRAL_BANDS)[None, :] == 0

    def retry_step(carry, _):
        (
            current_running,
            current_do_over,
            current_error_index,
            current_cosine,
            current_attempt_count,
            current_approximation,
            current_albedo,
            current_absorption,
            current_trip,
            current_fatal,
            current_energy,
        ) = carry
        change_cosine = current_running & (current_do_over == jnp.int32(3))
        cosine_delta = jnp.where(
            original_cosine[:, None] > jnp.float64(0.5),
            jnp.float64(-0.02),
            jnp.float64(0.02),
        )
        trial_cosine = jnp.where(
            change_cosine,
            fortran_add(current_cosine, cosine_delta),
            current_cosine,
        )
        alternate = (current_do_over == jnp.int32(2)) | (current_do_over == jnp.int32(4))
        trial_approximation = jnp.where(
            visible_band,
            jnp.where(alternate, jnp.int32(3), jnp.int32(1)),
            jnp.where(alternate, jnp.int32(1), jnp.int32(3)),
        )
        trial = lax.cond(
            jnp.any(current_running),
            lambda: (
                _sparse_spectral_attempt(
                    optical, underlying_spectral, trial_cosine,
                    trial_approximation, direct, current_running,
                ) if compact_active_columns else _snicar_spectral_attempt(
                    optical, underlying_spectral, trial_cosine, trial_approximation, direct,
                )
            ),
            lambda: _SnicarSpectralAttempt(
                albedo=jnp.zeros_like(current_albedo),
                absorbed_flux=jnp.zeros_like(current_absorption),
                trip=jnp.zeros_like(current_trip),
                energy_residual=jnp.zeros_like(current_energy),
            ),
        )
        active_trip = current_running & trial.trip
        at_four = current_do_over == jnp.int32(4)
        fatal_now = active_trip & at_four & (current_error_index >= jnp.int32(20))
        next_do_over = jnp.where(
            active_trip & (current_do_over == jnp.int32(1)),
            jnp.int32(2),
            jnp.where(
                active_trip & (current_do_over == jnp.int32(2)),
                jnp.int32(3),
                jnp.where(
                    active_trip & (current_do_over == jnp.int32(3)),
                    jnp.int32(4),
                    jnp.where(
                        active_trip & at_four & (current_error_index < jnp.int32(20)),
                        jnp.int32(3),
                        jnp.int32(0),
                    ),
                ),
            ),
        )
        next_error_index = jnp.where(
            active_trip & at_four & (current_error_index < jnp.int32(20)),
            current_error_index + jnp.int32(1),
            current_error_index,
        )
        next_running = current_running & (next_do_over > jnp.int32(0))
        select_absorption = current_running[:, None, :]
        return (
            next_running,
            next_do_over,
            next_error_index,
            jnp.where(current_running, trial_cosine, current_cosine),
            current_attempt_count + current_running.astype(jnp.int32),
            jnp.where(current_running, trial_approximation, current_approximation),
            jnp.where(current_running, trial.albedo, current_albedo),
            jnp.where(select_absorption, trial.absorbed_flux, current_absorption),
            jnp.where(current_running, trial.trip, current_trip),
            current_fatal | fatal_now,
            jnp.where(current_running, trial.energy_residual, current_energy),
        ), None

    initial = (
        running,
        do_over,
        error_index,
        cosine,
        attempt_count,
        approximation,
        spectral_albedo,
        spectral_absorption,
        final_trip,
        fatal_failure,
        energy_residual,
    )
    if early_exit_retry_loop:
        def retry_needed(loop):
            iteration, current = loop
            return (iteration < jnp.int32(44)) & jnp.any(current[0])

        def retry_once(loop):
            iteration, current = loop
            updated, _ = retry_step(current, None)
            return iteration + jnp.int32(1), updated

        _, final = lax.while_loop(
            retry_needed,
            retry_once,
            (jnp.int32(0), initial),
        )
    elif iteration_ad_checkpointing():
        final, _ = checkpointed_scan(
            retry_step, initial, jnp.arange(44, dtype=jnp.int32),
        )
    else:
        final, _ = lax.scan(retry_step, initial, xs=None, length=44)
    (
        running,
        _do_over,
        _error_index,
        cosine,
        attempt_count,
        approximation,
        spectral_albedo,
        spectral_absorption,
        final_trip,
        fatal_failure,
        energy_residual,
    ) = final
    fatal_failure = fatal_failure | running

    weights = optical.diagnostics.spectral_weight_band
    nir_weight_sum = jnp.float64(0.0)
    nir_albedo_sum = jnp.zeros(column_count, dtype=jnp.float64)
    nir_absorption_sum = jnp.zeros(
        (column_count, NUMBER_OF_SNOW_LAYERS + 1),
        dtype=jnp.float64,
    )
    for band in range(1, NUMBER_OF_SNOW_SPECTRAL_BANDS):
        nir_weight_sum = fortran_add(nir_weight_sum, weights[band])
        nir_albedo_sum = fortran_add(
            nir_albedo_sum,
            fortran_multiply(weights[band], spectral_albedo[:, band]),
        )
        nir_absorption_sum = fortran_add(
            nir_absorption_sum,
            fortran_multiply(weights[band], spectral_absorption[:, :, band]),
        )
    nir_albedo = fortran_divide(nir_albedo_sum, nir_weight_sum)
    nir_absorption = fortran_divide(nir_absorption_sum, nir_weight_sum)
    broadband_albedo_computed = jnp.stack(
        (spectral_albedo[:, 0], nir_albedo),
        axis=1,
    )
    broadband_absorption = jnp.stack(
        (spectral_absorption[:, :, 0], nir_absorption),
        axis=2,
    )
    snow_water = jnp.asarray(
        optical_inputs.snow_water_equivalent_column,
        dtype=jnp.float64,
    )
    tiny_snow = (
        jnp.asarray(optical_inputs.nonurban_column, dtype=jnp.bool_)
        & (original_cosine > jnp.float64(0.0))
        & (snow_water < jnp.float64(MINIMUM_SNOW_WATER_EQUIVALENT))
        & (snow_water > jnp.float64(0.0))
    )
    broadband_albedo = jnp.where(
        qualified[:, None],
        broadband_albedo_computed,
        jnp.where(tiny_snow[:, None], underlying_broadband, jnp.float64(0.0)),
    )
    broadband_absorption = jnp.where(
        qualified[:, None, None],
        broadband_absorption,
        jnp.float64(0.0),
    )
    state = SnicarRadiativeTransferState(
        snow_albedo_column_band=broadband_albedo,
        absorbed_flux_column_level_band=broadband_absorption,
    )
    diagnostics = SnicarRadiativeTransferDiagnostics(
        spectral_snow_albedo_column_band=spectral_albedo,
        spectral_absorbed_flux_column_level_band=spectral_absorption,
        final_cosine_solar_zenith_column_band=cosine,
        final_approximation_type_column_band=approximation,
        attempt_count_column_band=attempt_count,
        final_trip_column_band=final_trip,
        fatal_retry_failure_column_band=fatal_failure,
        energy_residual_column_band=energy_residual,
        energy_conservation_failure_column_band=(jnp.abs(energy_residual) > jnp.float64(0.00001)),
    )
    return SnicarRadiativeTransferResult(
        state=state,
        optical=optical,
        diagnostics=diagnostics,
    )


def _compact_radiative_transfer(
    inputs, tables, direct_beam, *, early_exit_retry_loop, capacities=(32768, 65536),
    runtime_diagnostics=False,
):
    optical = inputs.optical
    count = optical.snow_layer_count_column.shape[0]
    capacities = tuple(capacity for capacity in capacities if capacity < count)
    options = dict(early_exit_retry_loop=early_exit_retry_loop, compact_active_columns=True)

    def dense():
        result = _snicar_radiative_transfer_dense(inputs, tables, direct_beam, **options)
        return _summarize_snicar(result) if runtime_diagnostics else result

    if not capacities:
        return dense()
    _validate_shapes(optical, tables)
    if inputs.underlying_surface_albedo_column_band.shape != (count, 2):
        raise ValueError(f"underlying_surface_albedo_column_band must have shape {(count, 2)}")
    nonurban_day = (
        jnp.asarray(optical.nonurban_column, jnp.bool_)
        & (optical.cosine_solar_zenith_column > 0.)
    )
    snow_water = optical.snow_water_equivalent_column
    qualified = nonurban_day & (snow_water > MINIMUM_SNOW_WATER_EQUIVALENT)
    active_count = jnp.count_nonzero(qualified)
    tiny = nonurban_day & (snow_water > 0.) & (snow_water < MINIMUM_SNOW_WATER_EQUIVALENT)

    def packed(capacity):
        indices = jnp.nonzero(qualified, size=capacity, fill_value=count)[0]
        safe = jnp.minimum(indices, count - 1)
        selected = tree_util.tree_map(lambda value: jnp.asarray(value)[safe], inputs)
        selected = selected._replace(optical=selected.optical._replace(
            nonurban_column=(jnp.asarray(selected.optical.nonurban_column, jnp.bool_)
                             & (indices < count))))
        result = _snicar_radiative_transfer_dense(selected, tables, direct_beam, **options)

        def restore(path, value):
            name = path[-1].name
            if name == "spectral_weight_band":
                return value
            shape = (count, *value.shape[1:])
            if name == "snow_albedo_column_band":
                base = jnp.where(tiny[:, None], inputs.underlying_surface_albedo_column_band, 0.)
            else:
                default = {
                    "final_cosine_solar_zenith_column_band": .7,
                    "final_approximation_type_column_band": 1,
                }.get(name, 0)
                base = jnp.full(shape, default, value.dtype)
            return base.at[indices].set(value, mode="drop")

        if runtime_diagnostics:
            state = tree_util.tree_map_with_path(restore, result.state)
            return _summarize_snicar(result._replace(state=state))
        return tree_util.tree_map_with_path(restore, result)

    def dispatch(index):
        if index == len(capacities):
            return dense()
        return lax.cond(active_count <= capacities[index],
                        lambda: packed(capacities[index]), lambda: dispatch(index + 1))

    return dispatch(0)


def snicar_radiative_transfer(
    inputs: SnicarRadiativeTransferInputs,
    tables: SnowOpticalTables,
    direct_beam: bool | jnp.ndarray,
    *,
    early_exit_retry_loop: bool = False,
    compact_active_columns: bool = False,
    runtime_diagnostics: bool = False,
) -> SnicarRadiativeTransferResult | SnicarRuntimeResult:
    if compact_active_columns:
        return _compact_radiative_transfer(
            inputs, tables, direct_beam, early_exit_retry_loop=early_exit_retry_loop,
            runtime_diagnostics=runtime_diagnostics)
    result = _snicar_radiative_transfer_dense(
        inputs, tables, direct_beam, early_exit_retry_loop=early_exit_retry_loop)
    return _summarize_snicar(result) if runtime_diagnostics else result


__all__ = [
    "DIFFUSE_SPECTRAL_WEIGHTS",
    "DIRECT_SPECTRAL_WEIGHTS",
    "FRESH_SNOW_GRAIN_RADIUS_MICRONS",
    "MAXIMUM_SNOW_GRAIN_RADIUS_MICRONS",
    "MINIMUM_SNOW_GRAIN_RADIUS_MICRONS",
    "MINIMUM_SNOW_WATER_EQUIVALENT",
    "NUMBER_OF_SNOW_AEROSOLS",
    "NUMBER_OF_SNOW_GRAIN_RADII",
    "NUMBER_OF_SNOW_LAYERS",
    "NUMBER_OF_SNOW_SPECTRAL_BANDS",
    "SnicarOpticalDiagnostics",
    "SnicarOpticalInputs",
    "SnicarOpticalLayerState",
    "SnicarOpticalResult",
    "SnicarRadiativeTransferDiagnostics",
    "SnicarRadiativeTransferInputs",
    "SnicarRadiativeTransferResult",
    "SnicarRadiativeTransferState",
    "SnicarRuntimeDiagnostics",
    "SnicarRuntimeResult",
    "SnowOpticalTables",
    "snicar_optical_properties",
    "snicar_radiative_transfer",
]
