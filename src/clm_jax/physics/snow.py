
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import custom_jvp, default_backend, jvp

from clm_jax.constants import FREEZING_TEMPERATURE_FRESHWATER, PI
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.restart import (
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    LANDUNIT_WETLAND,
)
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
    fortran_power,
    strict_acos,
    strict_tanh,
)
from clm_jax.physics.native_power import native_power

LOW_TEMPERATURE_DENSITY_TRUNCATED_ANDERSON_1976 = 1
LOW_TEMPERATURE_DENSITY_SLATER_2017 = 2


class NewSnowDensityParameters(NamedTuple):

    wind_dependent: bool
    low_temperature_method: int

    @classmethod
    def from_case(cls, case: CimeCase) -> NewSnowDensityParameters:
        values = read_case_namelist_group(case, "lnd_in", "clm_snowhydrology_inparm")
        method_name = str(values["lotmp_snowdensity_method"]).strip()
        methods = {
            "TruncatedAnderson1976": LOW_TEMPERATURE_DENSITY_TRUNCATED_ANDERSON_1976,
            "Slater2017": LOW_TEMPERATURE_DENSITY_SLATER_2017,
        }
        if method_name not in methods:
            raise ValueError(f"unsupported low-temperature snow-density method: {method_name}")
        return cls(
            wind_dependent=bool(values["wind_dependent_snow_density"]),
            low_temperature_method=methods[method_name],
        )


class SnowAccumulationParameters(NamedTuple):
    integrated_snow_maximum: float
    use_subgrid_snow: bool
    old_snow_cover_fraction: bool
    soil_roughness_length: float = 0.01
    fresh_snow_grain_radius: float = 54.526

    @classmethod
    def from_case(cls, case: CimeCase) -> SnowAccumulationParameters:
        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        canopy = read_case_namelist_group(
            case,
            "lnd_in",
            "clm_canopyhydrology_inparm",
        )
        return cls(
            integrated_snow_maximum=float(general["int_snow_max"]),
            use_subgrid_snow=True,
            old_snow_cover_fraction=bool(canopy.get("oldfflag", 0)),
        )


class SnowColumnState(NamedTuple):
    layer_count: jnp.ndarray
    water_equivalent: jnp.ndarray
    depth: jnp.ndarray
    integrated_snow: jnp.ndarray
    snow_fraction: jnp.ndarray
    effective_snow_fraction: jnp.ndarray
    layer_thickness: jnp.ndarray
    layer_depth: jnp.ndarray
    layer_interface_depth: jnp.ndarray
    layer_temperature: jnp.ndarray
    layer_ice: jnp.ndarray
    layer_liquid: jnp.ndarray
    old_ice_fraction: jnp.ndarray
    snow_grain_radius: jnp.ndarray


class SnowAccumulationResult(NamedTuple):
    state: SnowColumnState
    previous_layer_water: jnp.ndarray
    new_snow: jnp.ndarray
    previous_step_snowmelt: jnp.ndarray
    snow_depth_rate: jnp.ndarray
    snow_on_surface_water: jnp.ndarray
    new_layer_created: jnp.ndarray


def _snow_power(base, exponent):
    if default_backend() == "gpu":
        from clm_jax.physics.positive_power import positive_power
        return positive_power(base, exponent)
    return fortran_power(base, exponent)


@custom_jvp
def _snow_depth_quotient(snowfall, denominator):
    return snowfall / denominator


@_snow_depth_quotient.defjvp
def _snow_depth_quotient_jvp(primals, tangents):
    snowfall, denominator = primals
    snowfall_tangent, denominator_tangent = tangents
    quotient = _snow_depth_quotient(snowfall, denominator)
    return quotient, (snowfall_tangent - quotient * denominator_tangent) / denominator


def new_snow_bulk_density(
    temperature: jnp.ndarray,
    wind_speed: jnp.ndarray,
    parameters: NewSnowDensityParameters,
) -> jnp.ndarray:

    temperature = jnp.asarray(temperature, dtype=jnp.float64)
    wind_speed = jnp.asarray(wind_speed, dtype=jnp.float64)
    freezing = jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)

    warm_density = fortran_add(
        jnp.float64(50.0),
        fortran_multiply(
            jnp.float64(1.7),
            fortran_power(jnp.float64(17.0), jnp.float64(1.5)),
        ),
    )
    temperature_delta = fortran_add(
        fortran_add(temperature, -freezing),
        jnp.float64(15.0),
    )
    temperature_power = _snow_power(
        jnp.maximum(temperature_delta, jnp.float64(0.0)),
        jnp.float64(1.5),
    )
    intermediate_density = fortran_add(
        jnp.float64(50.0),
        fortran_multiply(jnp.float64(1.7), temperature_power),
    )

    temperature_celsius = jnp.where(
        temperature > fortran_add(freezing, jnp.float64(-57.55)),
        fortran_add(temperature, -freezing),
        jnp.float64(-57.55),
    )
    slater_linear_coefficient = fortran_add(
        fortran_divide(jnp.float64(50.0), jnp.float64(15.0)),
        fortran_multiply(jnp.float64(0.0333), jnp.float64(15.0)),
    )
    slater_density = fortran_add(
        fortran_multiply(-slater_linear_coefficient, temperature_celsius),
        -fortran_multiply(
            jnp.float64(0.0333),
            fortran_power(temperature_celsius, jnp.float64(2.0)),
        ),
    )
    low_temperature_density = jnp.where(
        jnp.asarray(parameters.low_temperature_method, dtype=jnp.int32)
        == LOW_TEMPERATURE_DENSITY_SLATER_2017,
        slater_density,
        jnp.float64(50.0),
    )
    density = jnp.where(
        temperature > fortran_add(freezing, jnp.float64(2.0)),
        warm_density,
        jnp.where(
            temperature > fortran_add(freezing, jnp.float64(-15.0)),
            intermediate_density,
            low_temperature_density,
        ),
    )

    scaled_wind = fortran_divide(wind_speed, jnp.float64(5.0))
    wind_shape = fortran_divide(
        fortran_add(jnp.float64(1.0), strict_tanh(scaled_wind)),
        jnp.float64(2.0),
    )
    wind_offset = fortran_multiply(
        jnp.float64(266.861),
        _snow_power(wind_shape, jnp.float64(8.8)),
    )
    apply_wind = jnp.asarray(parameters.wind_dependent, dtype=jnp.bool_) & (
        wind_speed > jnp.float64(0.1)
    )
    return fortran_add(
        density,
        jnp.where(apply_wind, wind_offset, jnp.float64(0.0)),
    )


def _integrated_snow_denominator(snow_fraction, snow_cover_shape):
    complement = fortran_add(
        jnp.float64(1.0), -jnp.maximum(snow_fraction, jnp.float64(1.0e-6)),
    )
    exponent = fortran_divide(jnp.float64(1.0), snow_cover_shape)
    angle = fortran_multiply(jnp.float64(PI), _snow_power(complement, exponent))
    if default_backend() == "gpu":
        from clm_jax.physics.snow_cosine import snow_cover_cosine
        cosine = snow_cover_cosine(angle)
    else:
        cosine = jnp.cos(angle)
    return fortran_multiply(jnp.float64(0.5), fortran_add(cosine, jnp.float64(1.0)))


def _updated_snow_denominator(snow_fraction, snow_cover_shape, update):
    full_cover = (update & (snow_fraction == jnp.float64(1.0))
                  & (snow_cover_shape > 0.0) & jnp.isfinite(snow_cover_shape))
    fraction = jnp.where(update & ~full_cover, snow_fraction, jnp.float64(0.5))
    value = _integrated_snow_denominator(fraction, snow_cover_shape)
    return jnp.where(full_cover, jnp.float64(1.0), value)


def _melt_snow_cover_curve(ratio, shape):
    argument = jnp.minimum(
        jnp.float64(1.0),
        jnp.maximum(
            jnp.float64(-1.0),
            fortran_add(
                fortran_multiply(jnp.float64(2.0), jnp.minimum(jnp.float64(1.0), ratio)),
                jnp.float64(-1.0),
            ),
        ),
    )
    return fortran_add(
        jnp.float64(1.0),
        -native_power(fortran_divide(strict_acos(argument), jnp.float64(PI)), shape),
    )


@custom_jvp
def _melt_snow_cover_fraction(ratio, shape):
    return _melt_snow_cover_curve(ratio, shape)


@_melt_snow_cover_fraction.defjvp
def _melt_snow_cover_fraction_jvp(primals, tangents):
    ratio, shape = primals
    rounded_lower = ((ratio > 0.) & (ratio < 1.)
                     & (fortran_add(fortran_multiply(2., ratio), -1.) == -1.))

    def differentiable(ratio, shape):
        saturated = ((ratio > jnp.float64(1.0)) & (shape > jnp.float64(0.0))) | (
            (ratio == jnp.float64(1.0)) & (shape > jnp.float64(2.0)))
        safe_ratio = jnp.where(saturated | rounded_lower, jnp.float64(0.5), ratio)
        return jnp.where(saturated, jnp.float64(1.0), _melt_snow_cover_curve(safe_ratio, shape))

    regular_tangent = jvp(differentiable, primals, tangents)[1]
    interior_ratio = jnp.where(rounded_lower, ratio, jnp.float64(0.5))
    interior_shape = jnp.where(rounded_lower, shape, jnp.float64(1.))
    angle_offset = (jnp.float64(2. / PI) * jnp.arcsin(jnp.sqrt(interior_ratio)))
    log_angle = jnp.log1p(-angle_offset)
    ratio_slope = (interior_shape * jnp.exp((interior_shape - 1.) * log_angle)
                   / (jnp.float64(PI) * jnp.sqrt(interior_ratio)
                      * jnp.sqrt(1. - interior_ratio)))
    shape_slope = -jnp.exp(interior_shape * log_angle) * log_angle
    interior_tangent = ratio_slope * tangents[0] + shape_slope * tangents[1]
    return (_melt_snow_cover_fraction(*primals),
            jnp.where(rounded_lower, interior_tangent, regular_tangent))


def accumulate_snow(
    landunit_type: jnp.ndarray,
    ground_snow_flux: jnp.ndarray,
    previous_snow_drainage_flux: jnp.ndarray,
    new_snow_density: jnp.ndarray,
    atmospheric_temperature: jnp.ndarray,
    ground_temperature: jnp.ndarray,
    snow_cover_shape: jnp.ndarray,
    state: SnowColumnState,
    *,
    step_seconds: int | float,
    parameters: SnowAccumulationParameters,
) -> SnowAccumulationResult:

    landunit_type = jnp.asarray(landunit_type, dtype=jnp.int32)
    ground_snow_flux = jnp.asarray(ground_snow_flux, dtype=jnp.float64)
    snow_drainage = jnp.asarray(previous_snow_drainage_flux, dtype=jnp.float64)
    snowfall_density = jnp.asarray(new_snow_density, dtype=jnp.float64)
    atmospheric_temperature = jnp.asarray(atmospheric_temperature, dtype=jnp.float64)
    ground_temperature = jnp.asarray(ground_temperature, dtype=jnp.float64)
    snow_cover_shape = jnp.asarray(snow_cover_shape, dtype=jnp.float64)
    step_seconds = jnp.asarray(step_seconds, dtype=jnp.float64)

    layer_count = jnp.asarray(state.layer_count, dtype=jnp.int32)
    initial_water = jnp.asarray(state.water_equivalent, dtype=jnp.float64)
    initial_depth = jnp.asarray(state.depth, dtype=jnp.float64)
    integrated_snow = jnp.asarray(state.integrated_snow, dtype=jnp.float64)
    snow_fraction = jnp.asarray(state.snow_fraction, dtype=jnp.float64)
    effective_snow_fraction = jnp.asarray(
        state.effective_snow_fraction,
        dtype=jnp.float64,
    )
    layer_thickness = jnp.asarray(state.layer_thickness, dtype=jnp.float64)
    layer_depth = jnp.asarray(state.layer_depth, dtype=jnp.float64)
    interface_depth = jnp.asarray(state.layer_interface_depth, dtype=jnp.float64)
    layer_temperature = jnp.asarray(state.layer_temperature, dtype=jnp.float64)
    layer_ice = jnp.asarray(state.layer_ice, dtype=jnp.float64)
    layer_liquid = jnp.asarray(state.layer_liquid, dtype=jnp.float64)
    old_ice_fraction = jnp.asarray(state.old_ice_fraction, dtype=jnp.float64)
    snow_grain_radius = jnp.asarray(state.snow_grain_radius, dtype=jnp.float64)

    number_of_levels = layer_ice.shape[-1]
    level_number = jnp.arange(-number_of_levels + 1, 1, dtype=jnp.int32)
    initially_active_layer = level_number[None, :] >= layer_count[:, None] + jnp.int32(1)
    previous_layer_water = jnp.where(
        initially_active_layer,
        layer_liquid + layer_ice,
        jnp.float64(0.0),
    )
    new_snow = fortran_multiply(ground_snow_flux, step_seconds)
    previous_step_snowmelt = fortran_multiply(snow_drainage, step_seconds)
    integrated_snow = jnp.maximum(integrated_snow, initial_water)
    initially_has_snow = initial_water > jnp.float64(0.0)
    has_new_snow = new_snow > jnp.float64(0.0)
    urban = (landunit_type >= LANDUNIT_URBAN_MIN) & (landunit_type <= LANDUNIT_URBAN_MAX)
    subgrid_nonurban = jnp.asarray(parameters.use_subgrid_snow, dtype=jnp.bool_) & ~urban

    melt_update = initially_has_snow & (previous_step_snowmelt > jnp.float64(0.0))
    limited_integrated_snow = jnp.minimum(
        integrated_snow,
        jnp.asarray(parameters.integrated_snow_maximum, dtype=jnp.float64),
    )
    safe_integrated_snow = jnp.where(
        melt_update & (limited_integrated_snow != jnp.float64(0.0)),
        limited_integrated_snow,
        jnp.float64(1.0),
    )
    snow_mass_ratio = jnp.where(
        melt_update,
        fortran_divide(initial_water, safe_integrated_snow),
        jnp.float64(0.5),
    )
    melt_fraction = _melt_snow_cover_fraction(snow_mass_ratio, snow_cover_shape)
    snow_fraction = jnp.where(melt_update, melt_fraction, snow_fraction)

    accumulation_update = initially_has_snow & has_new_snow
    accumulation_factor = jnp.float64(0.100000001490116119384765625)
    accumulated_fraction = fortran_add(
        jnp.float64(1.0),
        -fortran_multiply(
            fortran_add(
                jnp.float64(1.0),
                -strict_tanh(fortran_multiply(accumulation_factor, new_snow)),
            ),
            fortran_add(jnp.float64(1.0), -snow_fraction),
        ),
    )
    snow_fraction = jnp.where(accumulation_update, accumulated_fraction, snow_fraction)

    safe_shape = jnp.where(snow_cover_shape != 0.0, snow_cover_shape, jnp.float64(1.0))
    integrated_denominator = _updated_snow_denominator(
        snow_fraction, safe_shape, accumulation_update,
    )
    safe_integrated_denominator = jnp.where(
        has_new_snow & (integrated_denominator != 0.0),
        integrated_denominator,
        jnp.float64(1.0),
    )
    diagnosed_integrated_snow = jnp.minimum(
        jnp.float64(1.0e8),
        fortran_divide(fortran_add(initial_water, new_snow), safe_integrated_denominator),
    )
    integrated_snow = jnp.where(
        accumulation_update,
        diagnosed_integrated_snow,
        integrated_snow,
    )

    safe_density = jnp.where(snowfall_density > 0.0, snowfall_density, jnp.float64(1.0))
    safe_fraction = jnp.where(snow_fraction > 0.0, snow_fraction, jnp.float64(1.0))
    existing_depth = jnp.where(
        subgrid_nonurban,
        jnp.where(
            snow_fraction > 0.0,
            initial_depth + _snow_depth_quotient(new_snow, safe_density * safe_fraction),
            jnp.float64(0.0),
        ),
        initial_depth + _snow_depth_quotient(new_snow, safe_density),
    )
    depth = jnp.where(initially_has_snow, existing_depth, initial_depth)

    new_snow_on_bare_ground = ~initially_has_snow & has_new_snow
    bare_ground_fraction = strict_tanh(fortran_multiply(accumulation_factor, new_snow))
    snow_fraction = jnp.where(new_snow_on_bare_ground, bare_ground_fraction, snow_fraction)
    bare_integrated_denominator = _updated_snow_denominator(
        snow_fraction, safe_shape, new_snow_on_bare_ground,
    )
    safe_bare_denominator = jnp.where(
        new_snow_on_bare_ground & (bare_integrated_denominator != 0.0),
        bare_integrated_denominator,
        jnp.float64(1.0),
    )
    bare_integrated_snow = jnp.minimum(
        jnp.float64(1.0e8),
        fortran_divide(new_snow, safe_bare_denominator),
    )
    integrated_snow = jnp.where(
        new_snow_on_bare_ground,
        bare_integrated_snow,
        integrated_snow,
    )
    average_new_snow_depth = fortran_divide(new_snow, safe_density)
    safe_bare_fraction = jnp.where(snow_fraction > 0.0, snow_fraction, jnp.float64(1.0))
    bare_depth = jnp.where(
        subgrid_nonurban,
        fortran_divide(average_new_snow_depth, safe_bare_fraction),
        average_new_snow_depth,
    )
    depth = jnp.where(new_snow_on_bare_ground, _coupled_bare_snow_depth(bare_depth, new_snow, safe_density, accumulation_factor, subgrid_nonurban, new_snow_on_bare_ground), depth)
    no_snow = ~initially_has_snow & ~has_new_snow
    depth = jnp.where(no_snow, jnp.float64(0.0), depth)
    snow_fraction = jnp.where(no_snow, jnp.float64(0.0), snow_fraction)

    old_fraction_enabled = jnp.asarray(parameters.old_snow_cover_fraction, dtype=jnp.bool_)
    apply_old_fraction = (
        old_fraction_enabled & (initially_has_snow | new_snow_on_bare_ground) & (depth > 0.0)
    )
    roughness = jnp.asarray(parameters.soil_roughness_length, dtype=jnp.float64)
    safe_depth = jnp.where(apply_old_fraction, depth, jnp.float64(1.0))
    bulk_density_candidate = jnp.minimum(
        jnp.float64(800.0),
        (initial_water + new_snow) / safe_depth,
    )
    bulk_density = jnp.where(
        apply_old_fraction,
        bulk_density_candidate,
        jnp.float64(100.0),
    )
    old_fraction_depth = jnp.where(
        apply_old_fraction,
        depth,
        jnp.float64(0.0),
    )
    old_fraction = jnp.tanh(
        old_fraction_depth
        / (jnp.float64(2.5) * roughness * (bulk_density / jnp.float64(100.0)) ** jnp.float64(1.0))
    )
    snow_fraction = jnp.where(apply_old_fraction, old_fraction, snow_fraction)
    cap_existing_old_fraction = (
        old_fraction_enabled & initially_has_snow & (initial_water < jnp.float64(1.0))
    )
    snow_fraction = jnp.where(
        cap_existing_old_fraction,
        jnp.minimum(snow_fraction, initial_water),
        snow_fraction,
    )

    water_equivalent = initial_water + new_snow
    integrated_snow = fortran_add(integrated_snow, new_snow)
    snow_depth_rate = (depth - initial_depth) / step_seconds
    effective_snow_fraction = jnp.where(
        ~urban & jnp.asarray(parameters.use_subgrid_snow, dtype=jnp.bool_),
        snow_fraction,
        jnp.float64(1.0),
    )

    warm_wetland = (landunit_type == LANDUNIT_WETLAND) & (
        ground_temperature > jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)
    )
    water_equivalent = jnp.where(warm_wetland, jnp.float64(0.0), water_equivalent)
    depth = jnp.where(warm_wetland, jnp.float64(0.0), depth)

    new_layer = (layer_count == jnp.int32(0)) & (snow_fraction * depth >= jnp.float64(0.01))
    layer_count = jnp.where(new_layer, jnp.int32(-1), layer_count)
    surface_layer = number_of_levels - 1
    surface_interface = number_of_levels - 1
    layer_thickness = layer_thickness.at[:, surface_layer].set(
        jnp.where(new_layer, depth, layer_thickness[:, surface_layer])
    )
    layer_depth = layer_depth.at[:, surface_layer].set(
        jnp.where(
            new_layer,
            -jnp.float64(0.5) * depth,
            layer_depth[:, surface_layer],
        )
    )
    interface_depth = interface_depth.at[:, surface_interface].set(
        jnp.where(
            new_layer,
            -depth,
            interface_depth[:, surface_interface],
        )
    )
    layer_temperature = layer_temperature.at[:, surface_layer].set(
        jnp.where(
            new_layer,
            jnp.minimum(
                jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
                atmospheric_temperature,
            ),
            layer_temperature[:, surface_layer],
        )
    )
    layer_ice = layer_ice.at[:, surface_layer].set(
        jnp.where(new_layer, water_equivalent, layer_ice[:, surface_layer])
    )
    layer_liquid = layer_liquid.at[:, surface_layer].set(
        jnp.where(new_layer, jnp.float64(0.0), layer_liquid[:, surface_layer])
    )
    old_ice_fraction = old_ice_fraction.at[:, surface_layer].set(
        jnp.where(new_layer, jnp.float64(1.0), old_ice_fraction[:, surface_layer])
    )
    snow_grain_radius = snow_grain_radius.at[:, surface_layer].set(
        jnp.where(
            new_layer,
            jnp.asarray(parameters.fresh_snow_grain_radius, dtype=jnp.float64),
            snow_grain_radius[:, surface_layer],
        )
    )

    add_to_existing_layer = (layer_count < jnp.int32(0)) & ~new_layer
    existing_layer_index = jnp.clip(layer_count + number_of_levels, 0, number_of_levels - 1)
    row = jnp.arange(layer_count.shape[0], dtype=jnp.int32)
    existing_ice = layer_ice[row, existing_layer_index]
    existing_thickness = layer_thickness[row, existing_layer_index]
    layer_ice = layer_ice.at[row, existing_layer_index].set(
        jnp.where(
            add_to_existing_layer,
            existing_ice + new_snow,
            existing_ice,
        )
    )
    layer_thickness = layer_thickness.at[row, existing_layer_index].set(
        jnp.where(
            add_to_existing_layer,
            existing_thickness + snow_depth_rate * step_seconds,
            existing_thickness,
        )
    )

    return SnowAccumulationResult(
        state=SnowColumnState(
            layer_count=layer_count,
            water_equivalent=water_equivalent,
            depth=depth,
            integrated_snow=integrated_snow,
            snow_fraction=snow_fraction,
            effective_snow_fraction=effective_snow_fraction,
            layer_thickness=layer_thickness,
            layer_depth=layer_depth,
            layer_interface_depth=interface_depth,
            layer_temperature=layer_temperature,
            layer_ice=layer_ice,
            layer_liquid=layer_liquid,
            old_ice_fraction=old_ice_fraction,
            snow_grain_radius=snow_grain_radius,
        ),
        previous_layer_water=previous_layer_water,
        new_snow=new_snow,
        previous_step_snowmelt=previous_step_snowmelt,
        snow_depth_rate=snow_depth_rate,
        snow_on_surface_water=jnp.zeros_like(new_snow),
        new_layer_created=new_layer,
    )


@custom_jvp
def _coupled_bare_snow_depth(native_depth, snowfall, density, factor, subgrid, active):
    return native_depth


@_coupled_bare_snow_depth.defjvp
def _coupled_bare_snow_depth_jvp(primals, tangents):
    native_depth, snowfall, density, factor, subgrid, active = primals
    native_dot, dsnow, ddensity, dfactor, _, _ = tangents
    from clm_jax.physics.adjoint_scaling import sensitivity_product, sensitivity_sum
    native_depth = _coupled_bare_snow_depth(*primals)

    z = jnp.where(active & subgrid, factor * snowfall, jnp.float64(1.0))
    squared = z * z
    series = 1.0 + squared * (1.0/3.0 + squared * (-1.0/45.0 + squared * (
        2.0/945.0 + squared * (-1.0/4725.0 + squared * (2.0/93555.0)))))
    series_derivative = z * (2.0/3.0 + squared * (-4.0/45.0 + squared * (
        4.0/315.0 + squared * (-8.0/4725.0 + squared * (4.0/18711.0)))))
    safe_z = jnp.where(jnp.abs(z) < 0.05, jnp.float64(1.0), z)
    tanh_z = jnp.tanh(safe_z)
    exponential = jnp.exp(-2.0 * jnp.abs(z))
    sech_squared = 4.0 * exponential / ((1.0 + exponential) * (1.0 + exponential))
    regular_derivative = (tanh_z - safe_z * (
        4.0 * jnp.exp(-2.0*jnp.abs(safe_z)) /
        ((1.0+jnp.exp(-2.0*jnp.abs(safe_z)))**2))) / (tanh_z*tanh_z)
    small = jnp.abs(z) < 0.05
    f = jnp.where(small, series, safe_z / tanh_z)
    df = jnp.where(small, series_derivative, regular_derivative)
    factor_series = -1.0 + squared * (1.0/3.0 + squared * (-1.0/15.0 + squared * (
        2.0/189.0 + squared * (-1.0/675.0 + squared * (2.0/10395.0)))))
    snow_partial = jnp.where(subgrid, df / density, 1.0 / density)
    factor_partial = jnp.where(subgrid,
        jnp.where(small, factor_series, -(f*f)*sech_squared) /
        (density * factor * factor), 0.0)
    stable = sensitivity_sum(
        sensitivity_product(snow_partial, dsnow),
        sensitivity_product(-native_depth / density, ddensity),
        sensitivity_product(factor_partial, dfactor),
    )
    return native_depth, jnp.where(active, stable, native_dot)
