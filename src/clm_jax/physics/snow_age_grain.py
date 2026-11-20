
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.constants import FREEZING_TEMPERATURE_FRESHWATER, PI
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
    fortran_power,
)
from clm_jax.physics.snow_water import (
    NUMBER_OF_SNOW_LEVELS,
    NUMBER_OF_SOIL_SNOW_LEVELS,
)

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase


NUMBER_OF_SNOW_AGING_DENSITIES = 8
NUMBER_OF_SNOW_AGING_GRADIENTS = 31
NUMBER_OF_SNOW_AGING_TEMPERATURES = 11


class SnowAgingLookupTable(NamedTuple):

    time_scale_hours: jnp.ndarray
    shape_exponent: jnp.ndarray
    initial_growth_rate_microns_per_hour: jnp.ndarray


class SnowAgeGrainParameters(NamedTuple):

    time_step_seconds: float = 1800.0
    minimum_grain_radius_microns: float = 54.526
    maximum_fresh_grain_radius_microns: float = 204.526
    maximum_grain_radius_microns: float = 1500.0
    refrozen_grain_radius_microns: float = 1000.0
    wet_snow_growth_coefficient: float = 4.22e-13
    scale_aging_rate: bool = False
    aging_rate_scale: float = 1.0

    @classmethod
    def from_case(cls, case: CimeCase) -> SnowAgeGrainParameters:

        from clm_jax.io.namelist import read_case_namelist_group  

        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        aerosol = read_case_namelist_group(case, "lnd_in", "aerosol")
        return cls(
            time_step_seconds=float(general["dtime"]),
            maximum_fresh_grain_radius_microns=float(aerosol["fresh_snw_rds_max"]),
        )


DEFAULT_SNOW_AGE_GRAIN_PARAMETERS = SnowAgeGrainParameters()


class SnowAgeGrainInputs(NamedTuple):

    snow_column: jnp.ndarray
    no_snow_column: jnp.ndarray
    snow_layer_count_column: jnp.ndarray
    soil_snow_thickness_column_level: jnp.ndarray
    effective_snow_fraction_column: jnp.ndarray
    snow_water_equivalent_column: jnp.ndarray
    soil_snow_liquid_water_column_level: jnp.ndarray
    soil_snow_ice_water_column_level: jnp.ndarray
    ground_snowfall_flux_column: jnp.ndarray
    snow_layer_freezing_flux_column_level: jnp.ndarray
    soil_snow_temperature_column_level: jnp.ndarray
    atmospheric_temperature_column: jnp.ndarray


_USE_PARALLEL_GRAIN_LEVELS = True


class SnowAgeGrainState(NamedTuple):

    snow_grain_radius_column_level: jnp.ndarray
    top_snow_grain_radius_column: jnp.ndarray
    top_snow_liquid_fraction_column: jnp.ndarray
    top_snow_temperature_column: jnp.ndarray
    top_snow_temperature_gradient_column: jnp.ndarray


class SnowAgeGrainDiagnostics(NamedTuple):

    active_snow_layer: jnp.ndarray
    temperature_gradient_column_level: jnp.ndarray
    snow_density_column_level: jnp.ndarray
    temperature_lookup_index_column_level: jnp.ndarray
    gradient_lookup_index_column_level: jnp.ndarray
    density_lookup_index_column_level: jnp.ndarray
    fresh_snow_grain_radius_column: jnp.ndarray


class SnowAgeGrainResult(NamedTuple):
    state: SnowAgeGrainState
    diagnostics: SnowAgeGrainDiagnostics


def _validate_shapes(
    lookup: SnowAgingLookupTable,
    inputs: SnowAgeGrainInputs,
    state: SnowAgeGrainState,
    parameters: SnowAgeGrainParameters,
) -> None:
    if inputs.snow_layer_count_column.ndim != 1:
        raise ValueError("snow_layer_count_column must be a vector")
    number_of_columns = inputs.snow_layer_count_column.shape[0]
    column_shape = (number_of_columns,)
    snow_shape = (number_of_columns, NUMBER_OF_SNOW_LEVELS)
    soil_snow_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS)
    expected_inputs = {
        "snow_column": column_shape,
        "no_snow_column": column_shape,
        "soil_snow_thickness_column_level": soil_snow_shape,
        "effective_snow_fraction_column": column_shape,
        "snow_water_equivalent_column": column_shape,
        "soil_snow_liquid_water_column_level": soil_snow_shape,
        "soil_snow_ice_water_column_level": soil_snow_shape,
        "ground_snowfall_flux_column": column_shape,
        "snow_layer_freezing_flux_column_level": snow_shape,
        "soil_snow_temperature_column_level": soil_snow_shape,
        "atmospheric_temperature_column": column_shape,
    }
    for name, shape in expected_inputs.items():
        if getattr(inputs, name).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    expected_state = {
        "snow_grain_radius_column_level": snow_shape,
        "top_snow_grain_radius_column": column_shape,
        "top_snow_liquid_fraction_column": column_shape,
        "top_snow_temperature_column": column_shape,
        "top_snow_temperature_gradient_column": column_shape,
    }
    for name, shape in expected_state.items():
        if getattr(state, name).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    lookup_shape = (
        NUMBER_OF_SNOW_AGING_DENSITIES,
        NUMBER_OF_SNOW_AGING_GRADIENTS,
        NUMBER_OF_SNOW_AGING_TEMPERATURES,
    )
    for name in SnowAgingLookupTable._fields:
        if getattr(lookup, name).shape != lookup_shape:
            raise ValueError(f"lookup.{name} must have shape {lookup_shape}")
    if parameters.time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")
    if parameters.minimum_grain_radius_microns <= 0.0:
        raise ValueError("minimum_grain_radius_microns must be positive")
    if parameters.maximum_fresh_grain_radius_microns < parameters.minimum_grain_radius_microns:
        raise ValueError("maximum fresh grain radius cannot be below the minimum")
    if parameters.maximum_grain_radius_microns < parameters.maximum_fresh_grain_radius_microns:
        raise ValueError("maximum grain radius cannot be below the fresh maximum")
    if parameters.aging_rate_scale <= 0.0:
        raise ValueError("aging_rate_scale must be positive")


def _fortran_nint(value: jnp.ndarray) -> jnp.ndarray:

    value = jnp.asarray(value, dtype=jnp.float64)
    rounded = jnp.where(
        value >= jnp.float64(0.0),
        jnp.floor(fortran_add(value, jnp.float64(0.5))),
        jnp.ceil(fortran_add(value, jnp.float64(-0.5))),
    )
    return rounded.astype(jnp.int32)


def _fresh_snow_radius(
    atmospheric_temperature: jnp.ndarray,
    parameters: SnowAgeGrainParameters,
) -> jnp.ndarray:
    minimum = jnp.float64(parameters.minimum_grain_radius_microns)
    maximum = jnp.float64(parameters.maximum_fresh_grain_radius_microns)
    if parameters.maximum_fresh_grain_radius_microns <= (parameters.minimum_grain_radius_microns):
        return jnp.full_like(atmospheric_temperature, minimum)

    freezing = jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)
    ramp_minimum_temperature = fortran_add(freezing, jnp.float64(-30.0))
    ramp_width = fortran_add(freezing, -ramp_minimum_temperature)
    cold_weight = fortran_divide(
        fortran_add(freezing, -atmospheric_temperature),
        ramp_width,
    )
    warm_weight = fortran_divide(
        fortran_add(atmospheric_temperature, -ramp_minimum_temperature),
        ramp_width,
    )
    interpolated = fortran_add(
        fortran_multiply(cold_weight, minimum),
        fortran_multiply(warm_weight, maximum),
    )
    return jnp.where(
        atmospheric_temperature < ramp_minimum_temperature,
        minimum,
        jnp.where(atmospheric_temperature > freezing, maximum, interpolated),
    )


def _snow_age_grain_reference(
    lookup: SnowAgingLookupTable,
    inputs: SnowAgeGrainInputs,
    state: SnowAgeGrainState,
    *,
    parameters: SnowAgeGrainParameters = DEFAULT_SNOW_AGE_GRAIN_PARAMETERS,
) -> SnowAgeGrainResult:

    _validate_shapes(lookup, inputs, state, parameters)
    number_of_columns = inputs.snow_layer_count_column.shape[0]
    snow_column = jnp.asarray(inputs.snow_column, dtype=jnp.bool_)
    no_snow_column = jnp.asarray(inputs.no_snow_column, dtype=jnp.bool_)
    layer_count = jnp.asarray(inputs.snow_layer_count_column, dtype=jnp.int32)
    top_snow_index = layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS)
    level_number = jnp.arange(NUMBER_OF_SNOW_LEVELS, dtype=jnp.int32)
    active_snow_layer = snow_column[:, None] & (level_number[None, :] >= top_snow_index[:, None])

    thickness = jnp.asarray(
        inputs.soil_snow_thickness_column_level,
        dtype=jnp.float64,
    )
    snow_fraction = jnp.asarray(
        inputs.effective_snow_fraction_column,
        dtype=jnp.float64,
    )
    liquid = jnp.asarray(
        inputs.soil_snow_liquid_water_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_SNOW_LEVELS]
    ice = jnp.asarray(
        inputs.soil_snow_ice_water_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_SNOW_LEVELS]
    temperature = jnp.asarray(
        inputs.soil_snow_temperature_column_level,
        dtype=jnp.float64,
    )
    snowfall_flux = jnp.asarray(
        inputs.ground_snowfall_flux_column,
        dtype=jnp.float64,
    )
    freezing_flux = jnp.asarray(
        inputs.snow_layer_freezing_flux_column_level,
        dtype=jnp.float64,
    )
    snow_water_equivalent = jnp.asarray(
        inputs.snow_water_equivalent_column,
        dtype=jnp.float64,
    )
    atmospheric_temperature = jnp.asarray(
        inputs.atmospheric_temperature_column,
        dtype=jnp.float64,
    )

    radius = jnp.asarray(
        state.snow_grain_radius_column_level,
        dtype=jnp.float64,
    )
    top_radius = jnp.asarray(
        state.top_snow_grain_radius_column,
        dtype=jnp.float64,
    )
    top_liquid_fraction = jnp.asarray(
        state.top_snow_liquid_fraction_column,
        dtype=jnp.float64,
    )
    top_temperature = jnp.asarray(
        state.top_snow_temperature_column,
        dtype=jnp.float64,
    )
    top_gradient = jnp.asarray(
        state.top_snow_temperature_gradient_column,
        dtype=jnp.float64,
    )

    tau_table = jnp.asarray(lookup.time_scale_hours, dtype=jnp.float64)
    kappa_table = jnp.asarray(lookup.shape_exponent, dtype=jnp.float64)
    initial_rate_table = jnp.asarray(
        lookup.initial_growth_rate_microns_per_hour,
        dtype=jnp.float64,
    )
    fresh_radius = _fresh_snow_radius(atmospheric_temperature, parameters)
    minimum_radius = jnp.float64(parameters.minimum_grain_radius_microns)
    maximum_radius = jnp.float64(parameters.maximum_grain_radius_microns)
    refrozen_radius = jnp.float64(parameters.refrozen_grain_radius_microns)
    time_step = jnp.float64(parameters.time_step_seconds)
    dry_time_scale = fortran_divide(time_step, jnp.float64(3600.0))

    zeros = jnp.zeros(
        (number_of_columns, NUMBER_OF_SNOW_LEVELS),
        dtype=jnp.float64,
    )
    integer_zeros = jnp.zeros_like(zeros, dtype=jnp.int32)

    def update_level(level: int, carry):
        (
            current_radius,
            current_top_radius,
            current_top_liquid,
            current_top_temperature,
            current_top_gradient,
            gradient_diagnostic,
            density_diagnostic,
            temperature_index_diagnostic,
            gradient_index_diagnostic,
            density_index_diagnostic,
        ) = carry
        active = active_snow_layer[:, level]
        top = active & (jnp.int32(level) == top_snow_index)
        next_level = level + 1
        previous_level = jnp.maximum(level - 1, 0)

        layer_thickness = thickness[:, level]
        next_thickness = thickness[:, next_level]
        previous_thickness = thickness[:, previous_level]
        layer_temperature = temperature[:, level]
        next_temperature = temperature[:, next_level]
        previous_temperature = temperature[:, previous_level]

        lower_boundary_denominator = fortran_add(
            layer_thickness,
            next_thickness,
        )
        lower_boundary_temperature = fortran_divide(
            fortran_add(
                fortran_multiply(next_temperature, layer_thickness),
                fortran_multiply(layer_temperature, next_thickness),
            ),
            jnp.where(
                active,
                lower_boundary_denominator,
                jnp.float64(1.0),
            ),
        )
        upper_internal_denominator = fortran_add(
            layer_thickness,
            previous_thickness,
        )
        upper_internal_temperature = fortran_divide(
            fortran_add(
                fortran_multiply(previous_temperature, layer_thickness),
                fortran_multiply(layer_temperature, previous_thickness),
            ),
            jnp.where(
                active & ~top,
                upper_internal_denominator,
                jnp.float64(1.0),
            ),
        )
        upper_boundary_temperature = jnp.where(
            top,
            layer_temperature,
            upper_internal_temperature,
        )
        column_average_thickness = fortran_multiply(
            snow_fraction,
            layer_thickness,
        )
        safe_average_thickness = jnp.where(
            active,
            column_average_thickness,
            jnp.float64(1.0),
        )
        temperature_gradient = jnp.abs(
            fortran_divide(
                fortran_add(
                    upper_boundary_temperature,
                    -lower_boundary_temperature,
                ),
                safe_average_thickness,
            )
        )
        layer_mass = fortran_add(liquid[:, level], ice[:, level])
        safe_layer_mass = jnp.where(active, layer_mass, jnp.float64(1.0))
        density = jnp.maximum(
            jnp.float64(50.0),
            fortran_divide(safe_layer_mass, safe_average_thickness),
        )

        temperature_index = jnp.clip(
            _fortran_nint(
                fortran_divide(
                    fortran_add(layer_temperature, jnp.float64(-223.0)),
                    jnp.float64(5.0),
                )
            ),
            jnp.int32(0),
            jnp.int32(NUMBER_OF_SNOW_AGING_TEMPERATURES - 1),
        )
        gradient_index = jnp.clip(
            _fortran_nint(fortran_divide(temperature_gradient, jnp.float64(10.0))),
            jnp.int32(0),
            jnp.int32(NUMBER_OF_SNOW_AGING_GRADIENTS - 1),
        )
        density_index = jnp.clip(
            _fortran_nint(
                fortran_divide(
                    fortran_add(density, jnp.float64(-50.0)),
                    jnp.float64(50.0),
                )
            ),
            jnp.int32(0),
            jnp.int32(NUMBER_OF_SNOW_AGING_DENSITIES - 1),
        )
        tau = tau_table[density_index, gradient_index, temperature_index]
        kappa = kappa_table[density_index, gradient_index, temperature_index]
        initial_rate = initial_rate_table[
            density_index,
            gradient_index,
            temperature_index,
        ]

        old_radius = jnp.maximum(current_radius[:, level], minimum_radius)
        radius_above_fresh = fortran_add(old_radius, -minimum_radius)
        dry_power = fortran_power(
            fortran_divide(tau, fortran_add(radius_above_fresh, tau)),
            fortran_divide(jnp.float64(1.0), kappa),
        )
        dry_growth = fortran_multiply(
            fortran_multiply(initial_rate, dry_power),
            dry_time_scale,
        )

        liquid_fraction = jnp.minimum(
            jnp.float64(0.1),
            fortran_divide(liquid[:, level], safe_layer_mass),
        )
        liquid_fraction_cubed = fortran_multiply(
            fortran_multiply(liquid_fraction, liquid_fraction),
            liquid_fraction,
        )
        radius_squared = fortran_multiply(old_radius, old_radius)
        wet_growth = fortran_multiply(
            jnp.float64(1.0e18),
            fortran_divide(
                fortran_multiply(
                    time_step,
                    fortran_multiply(
                        jnp.float64(parameters.wet_snow_growth_coefficient),
                        liquid_fraction_cubed,
                    ),
                ),
                fortran_multiply(jnp.float64(4.0 * PI), radius_squared),
            ),
        )
        growth = fortran_add(dry_growth, wet_growth)
        if parameters.scale_aging_rate:
            growth = fortran_multiply(
                growth,
                jnp.float64(parameters.aging_rate_scale),
            )

        new_snow = jnp.maximum(
            jnp.float64(0.0),
            fortran_multiply(snowfall_flux, time_step),
        )
        refrozen_snow = jnp.maximum(
            jnp.float64(0.0),
            fortran_multiply(freezing_flux[:, level], time_step),
        )
        refrozen_fraction = fortran_divide(refrozen_snow, safe_layer_mass)
        new_snow_fraction = jnp.where(
            top,
            fortran_divide(new_snow, safe_layer_mass),
            jnp.float64(0.0),
        )
        fresh_and_refrozen = fortran_add(
            refrozen_fraction,
            new_snow_fraction,
        )
        normalize = fresh_and_refrozen > jnp.float64(1.0)
        normalized_refrozen = fortran_divide(
            refrozen_fraction,
            jnp.where(normalize, fresh_and_refrozen, jnp.float64(1.0)),
        )
        final_refrozen_fraction = jnp.where(
            normalize,
            normalized_refrozen,
            refrozen_fraction,
        )
        final_new_snow_fraction = jnp.where(
            normalize,
            fortran_add(jnp.float64(1.0), -normalized_refrozen),
            new_snow_fraction,
        )
        old_snow_fraction = jnp.where(
            normalize,
            jnp.float64(0.0),
            fortran_add(
                fortran_add(jnp.float64(1.0), -refrozen_fraction),
                -new_snow_fraction,
            ),
        )
        mixed_radius = fortran_add(
            fortran_add(
                fortran_multiply(
                    fortran_add(old_radius, growth),
                    old_snow_fraction,
                ),
                fortran_multiply(fresh_radius, final_new_snow_fraction),
            ),
            fortran_multiply(refrozen_radius, final_refrozen_fraction),
        )
        updated_radius = jnp.minimum(
            maximum_radius,
            jnp.maximum(minimum_radius, mixed_radius),
        )
        current_radius = current_radius.at[:, level].set(
            jnp.where(active, updated_radius, current_radius[:, level])
        )
        current_top_temperature = jnp.where(
            top,
            layer_temperature,
            current_top_temperature,
        )
        current_top_gradient = jnp.where(
            top,
            temperature_gradient,
            current_top_gradient,
        )
        current_top_radius = jnp.where(
            top,
            updated_radius,
            current_top_radius,
        )
        current_top_liquid = jnp.where(
            top,
            fortran_divide(liquid[:, level], safe_layer_mass),
            current_top_liquid,
        )
        gradient_diagnostic = gradient_diagnostic.at[:, level].set(
            jnp.where(active, temperature_gradient, jnp.float64(0.0))
        )
        density_diagnostic = density_diagnostic.at[:, level].set(
            jnp.where(active, density, jnp.float64(0.0))
        )
        temperature_index_diagnostic = temperature_index_diagnostic.at[:, level].set(
            jnp.where(active, temperature_index, jnp.int32(0))
        )
        gradient_index_diagnostic = gradient_index_diagnostic.at[:, level].set(
            jnp.where(active, gradient_index, jnp.int32(0))
        )
        density_index_diagnostic = density_index_diagnostic.at[:, level].set(
            jnp.where(active, density_index, jnp.int32(0))
        )
        return (
            current_radius,
            current_top_radius,
            current_top_liquid,
            current_top_temperature,
            current_top_gradient,
            gradient_diagnostic,
            density_diagnostic,
            temperature_index_diagnostic,
            gradient_index_diagnostic,
            density_index_diagnostic,
        )

    initial = (
        radius, top_radius, top_liquid_fraction, top_temperature, top_gradient,
        zeros, zeros, integer_zeros, integer_zeros, integer_zeros,
    )
    if _USE_PARALLEL_GRAIN_LEVELS:
        def independent_level(level):
            result = update_level(level, initial)
            return (result[0][:, level], *result[1:5],
                    *(value[:, level] for value in result[5:]))

        parallel = jax.vmap(independent_level, out_axes=1)(level_number)
        top_index = jnp.clip(top_snow_index, 0, NUMBER_OF_SNOW_LEVELS - 1)
        rows = jnp.arange(number_of_columns)
        final = (parallel[0], *(value[rows, top_index] for value in parallel[1:5]),
                 *parallel[5:])
    else:
        final = lax.fori_loop(0, NUMBER_OF_SNOW_LEVELS, update_level, initial)

    (
        radius,
        top_radius,
        top_liquid_fraction,
        top_temperature,
        top_gradient,
        gradient_diagnostic,
        density_diagnostic,
        temperature_index_diagnostic,
        gradient_index_diagnostic,
        density_index_diagnostic,
    ) = final

    unresolved_snow = no_snow_column & (snow_water_equivalent > jnp.float64(0.0))
    radius = radius.at[:, NUMBER_OF_SNOW_LEVELS - 1].set(
        jnp.where(
            unresolved_snow,
            minimum_radius,
            radius[:, NUMBER_OF_SNOW_LEVELS - 1],
        )
    )
    return SnowAgeGrainResult(
        state=SnowAgeGrainState(
            snow_grain_radius_column_level=radius,
            top_snow_grain_radius_column=top_radius,
            top_snow_liquid_fraction_column=top_liquid_fraction,
            top_snow_temperature_column=top_temperature,
            top_snow_temperature_gradient_column=top_gradient,
        ),
        diagnostics=SnowAgeGrainDiagnostics(
            active_snow_layer=active_snow_layer,
            temperature_gradient_column_level=gradient_diagnostic,
            snow_density_column_level=density_diagnostic,
            temperature_lookup_index_column_level=(temperature_index_diagnostic),
            gradient_lookup_index_column_level=gradient_index_diagnostic,
            density_lookup_index_column_level=density_index_diagnostic,
            fresh_snow_grain_radius_column=fresh_radius,
        ),
    )


@partial(jax.custom_jvp, nondiff_argnums=(3,))
def _snow_age_with_storage_adjoint(lookup, inputs, state, parameters):
    return _snow_age_grain_reference(lookup, inputs, state, parameters=parameters)


@_snow_age_with_storage_adjoint.defjvp
def _snow_age_storage_jvp(parameters, primals, tangents):
    def guarded(lookup, inputs, state):
        snow = inputs.snow_column
        first = inputs.snow_layer_count_column + NUMBER_OF_SNOW_LEVELS
        levels = jnp.arange(NUMBER_OF_SOIL_SNOW_LEVELS)[None, :]
        geometry = snow[:, None] & (levels >= first[:, None]) & (levels <= NUMBER_OF_SNOW_LEVELS)
        active = geometry[:, :NUMBER_OF_SNOW_LEVELS]
        water = jnp.pad(active, ((0, 0), (0, NUMBER_OF_SOIL_SNOW_LEVELS - NUMBER_OF_SNOW_LEVELS)))
        safe_inputs = inputs._replace(
            soil_snow_thickness_column_level=jnp.where(geometry, inputs.soil_snow_thickness_column_level, 1.),
            soil_snow_temperature_column_level=jnp.where(geometry, inputs.soil_snow_temperature_column_level,
                                                         FREEZING_TEMPERATURE_FRESHWATER),
            soil_snow_liquid_water_column_level=jnp.where(water, inputs.soil_snow_liquid_water_column_level, 0.),
            soil_snow_ice_water_column_level=jnp.where(water, inputs.soil_snow_ice_water_column_level, 0.),
            effective_snow_fraction_column=jnp.where(snow, inputs.effective_snow_fraction_column, 1.),
            ground_snowfall_flux_column=jnp.where(snow, inputs.ground_snowfall_flux_column, 0.),
            snow_layer_freezing_flux_column_level=jnp.where(active, inputs.snow_layer_freezing_flux_column_level, 0.),
        )
        safe_state = SnowAgeGrainState(
            jnp.where(active, state.snow_grain_radius_column_level, parameters.minimum_grain_radius_microns),
            *(jnp.where(snow, value, 0.) for value in state[1:]),
        )
        result = _snow_age_grain_reference(lookup, safe_inputs, safe_state, parameters=parameters)
        unresolved = inputs.no_snow_column & (inputs.snow_water_equivalent_column > 0.)
        updated_radius = active | (unresolved[:, None] & (levels[:, :NUMBER_OF_SNOW_LEVELS] == NUMBER_OF_SNOW_LEVELS - 1))
        restored = SnowAgeGrainState(
            jnp.where(updated_radius, result.state.snow_grain_radius_column_level, state.snow_grain_radius_column_level),
            *(jnp.where(snow, value, original) for value, original in zip(result.state[1:], state[1:], strict=True)),
        )
        return result._replace(state=restored)

    return (_snow_age_with_storage_adjoint(*primals, parameters),
            jax.jvp(guarded, primals, tangents)[1])


def snow_age_grain(lookup, inputs, state, *, parameters=DEFAULT_SNOW_AGE_GRAIN_PARAMETERS):
    return _snow_age_with_storage_adjoint(lookup, inputs, state, parameters)
