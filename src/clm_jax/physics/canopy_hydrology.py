
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import default_backend, lax

from clm_jax.constants import FREEZING_TEMPERATURE_FRESHWATER, PI
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_GLACIER,
    LANDUNIT_SOIL,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    LANDUNIT_WETLAND,
)
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_hardware_add,
    fortran_hardware_multiply,
    fortran_multiply,
    fortran_power,
    strict_negative_exp,
    strict_tanh,
)
from clm_jax.physics.native_error_function import native_erf as erf

SNOW_VEGETATION_OFF = 0
SNOW_VEGETATION_ON = 1
SNOW_VEGETATION_ON_RADIATION = 2
URBAN_SUN_WALL_COLUMN = LANDUNIT_URBAN_MIN * 10 + 2
URBAN_SHADE_WALL_COLUMN = LANDUNIT_URBAN_MIN * 10 + 3


class CanopyHydrologyParameters(NamedTuple):
    old_snow_cover_fraction: bool
    interception_fraction: float
    maximum_leaf_wetted_fraction: float
    use_clm5_interception: bool
    snow_vegetation_mode: int

    @classmethod
    def from_case(cls, case: CimeCase) -> CanopyHydrologyParameters:
        values = read_case_namelist_group(
            case,
            "lnd_in",
            "clm_canopyhydrology_inparm",
        )
        flag = str(values["snowveg_flag"]).strip()
        modes = {
            "OFF": SNOW_VEGETATION_OFF,
            "ON": SNOW_VEGETATION_ON,
            "ON_RAD": SNOW_VEGETATION_ON_RADIATION,
        }
        if flag not in modes:
            raise ValueError(f"unsupported snow vegetation flag: {flag}")
        return cls(
            old_snow_cover_fraction=bool(values.get("oldfflag", 0)),
            interception_fraction=float(values["interception_fraction"]),
            maximum_leaf_wetted_fraction=float(values["maximum_leaf_wetted_fraction"]),
            use_clm5_interception=bool(values["use_clm5_fpi"]),
            snow_vegetation_mode=modes[flag],
        )


class CanopyWaterState(NamedTuple):
    total_water: jnp.ndarray
    snow: jnp.ndarray
    liquid: jnp.ndarray
    snow_unloaded: jnp.ndarray


class CanopyPrecipitationFluxes(NamedTuple):
    drainage: jnp.ndarray
    snow_fall: jnp.ndarray
    liquid_fall: jnp.ndarray
    snow_wind_unloading: jnp.ndarray
    snow_temperature_unloading: jnp.ndarray
    snow_throughfall: jnp.ndarray
    rain_throughfall: jnp.ndarray
    intercepted_precipitation: jnp.ndarray
    ground_snow: jnp.ndarray
    ground_rain: jnp.ndarray
    ground_precipitation: jnp.ndarray


class CanopyInterceptionResult(NamedTuple):
    state: CanopyWaterState
    fluxes: CanopyPrecipitationFluxes


class CanopyWetFractions(NamedTuple):
    wet: jnp.ndarray
    snow_covered: jnp.ndarray
    dry_leaf: jnp.ndarray


class SurfaceWaterState(NamedTuple):
    surface_water: jnp.ndarray
    top_soil_liquid: jnp.ndarray
    snow_fraction: jnp.ndarray
    effective_snow_fraction: jnp.ndarray
    surface_water_fraction: jnp.ndarray
    surface_water_fraction_without_snow: jnp.ndarray


def canopy_interception(
    landunit_type: jnp.ndarray,
    column_type: jnp.ndarray,
    vegetation_exposed: jnp.ndarray,
    rain: jnp.ndarray,
    snow: jnp.ndarray,
    temperature: jnp.ndarray,
    wind_speed: jnp.ndarray,
    maximum_dew: jnp.ndarray,
    exposed_leaf_area_index: jnp.ndarray,
    exposed_stem_area_index: jnp.ndarray,
    irrigation: jnp.ndarray,
    state: CanopyWaterState,
    *,
    step_seconds: int | float,
    parameters: CanopyHydrologyParameters,
) -> CanopyInterceptionResult:

    landunit_type = jnp.asarray(landunit_type, dtype=jnp.int32)
    column_type = jnp.asarray(column_type, dtype=jnp.int32)
    exposed = jnp.asarray(vegetation_exposed, dtype=jnp.int32) == jnp.int32(1)
    rain = jnp.asarray(rain, dtype=jnp.float64)
    snow = jnp.asarray(snow, dtype=jnp.float64)
    temperature = jnp.asarray(temperature, dtype=jnp.float64)
    wind_speed = jnp.asarray(wind_speed, dtype=jnp.float64)
    maximum_dew = jnp.asarray(maximum_dew, dtype=jnp.float64)
    leaf_area = jnp.asarray(exposed_leaf_area_index, dtype=jnp.float64)
    stem_area = jnp.asarray(exposed_stem_area_index, dtype=jnp.float64)
    irrigation = jnp.asarray(irrigation, dtype=jnp.float64)
    step_seconds = jnp.asarray(step_seconds, dtype=jnp.float64)
    total_water = jnp.asarray(state.total_water, dtype=jnp.float64)
    canopy_snow = jnp.asarray(state.snow, dtype=jnp.float64)
    canopy_liquid = jnp.asarray(state.liquid, dtype=jnp.float64)

    is_urban = (landunit_type >= LANDUNIT_URBAN_MIN) & (
        landunit_type <= LANDUNIT_URBAN_MAX
    )
    supports_canopy = (
        (landunit_type == LANDUNIT_SOIL)
        | (landunit_type == LANDUNIT_CROP)
        | (landunit_type == LANDUNIT_WETLAND)
        | is_urban
    )
    is_glacier = landunit_type == LANDUNIT_GLACIER
    is_wall = (column_type == URBAN_SUN_WALL_COLUMN) | (
        column_type == URBAN_SHADE_WALL_COLUMN
    )
    positive_precipitation = (rain + snow) > jnp.float64(0.0)
    intercept = supports_canopy & ~is_wall & exposed & positive_precipitation
    snow_vegetation_enabled = (
        jnp.asarray(parameters.snow_vegetation_mode, dtype=jnp.int32)
        != SNOW_VEGETATION_OFF
    )

    zeros = jnp.zeros_like(total_water)
    total_water = jnp.where(is_glacier, zeros, total_water)
    canopy_snow = jnp.where(is_glacier, zeros, canopy_snow)
    canopy_liquid = jnp.where(is_glacier, zeros, canopy_liquid)
    vegetation_area = leaf_area + stem_area
    liquid_capacity = maximum_dew * vegetation_area
    clm5_interception = jnp.asarray(parameters.interception_fraction, dtype=jnp.float64) * (
        strict_tanh(vegetation_area)
    )
    legacy_interception = jnp.float64(0.25) * (
        jnp.float64(1.0) - strict_negative_exp(-jnp.float64(0.5) * vegetation_area)
    )
    rain_interception_fraction = jnp.where(
        jnp.asarray(parameters.use_clm5_interception, dtype=jnp.bool_),
        clm5_interception,
        legacy_interception,
    )
    snow_interception_fraction = jnp.where(
        snow_vegetation_enabled,
        jnp.float64(1.0) - strict_negative_exp(-jnp.float64(0.5) * vegetation_area),
        rain_interception_fraction,
    )

    snow_throughfall = jnp.where(
        intercept,
        snow * (jnp.float64(1.0) - snow_interception_fraction),
        zeros,
    )
    rain_throughfall = jnp.where(
        intercept,
        rain * (jnp.float64(1.0) - rain_interception_fraction),
        zeros,
    )
    intercepted_precipitation = jnp.where(
        intercept,
        fortran_hardware_add(
            fortran_hardware_multiply(snow, snow_interception_fraction),
            fortran_hardware_multiply(rain, rain_interception_fraction),
        ),
        zeros,
    )
    canopy_snow_candidate = jnp.maximum(
        jnp.float64(0.0),
        canopy_snow + step_seconds * snow * snow_interception_fraction,
    )
    canopy_liquid_candidate = jnp.maximum(
        jnp.float64(0.0),
        canopy_liquid + step_seconds * rain * rain_interception_fraction,
    )
    canopy_snow = jnp.where(
        intercept & snow_vegetation_enabled,
        canopy_snow_candidate,
        canopy_snow,
    )
    canopy_liquid = jnp.where(
        intercept & snow_vegetation_enabled,
        canopy_liquid_candidate,
        canopy_liquid,
    )
    total_water = jnp.where(
        intercept,
        jnp.maximum(
            jnp.float64(0.0),
            total_water + step_seconds * intercepted_precipitation,
        ),
        total_water,
    )

    above_freezing = temperature > jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)
    snow_capacity = jnp.float64(60.0) * maximum_dew * vegetation_area
    liquid_excess_rate = (canopy_liquid - liquid_capacity) / step_seconds
    snow_excess_rate = (canopy_snow - snow_capacity) / step_seconds
    liquid_fall = jnp.where(
        intercept & snow_vegetation_enabled & above_freezing & (liquid_excess_rate > 0.0),
        liquid_excess_rate,
        zeros,
    )
    snow_fall = jnp.where(
        intercept & snow_vegetation_enabled & ~above_freezing & (snow_excess_rate > 0.0),
        snow_excess_rate,
        zeros,
    )
    canopy_liquid = jnp.where(liquid_fall > 0.0, liquid_capacity, canopy_liquid)
    canopy_snow = jnp.where(snow_fall > 0.0, snow_capacity, canopy_snow)
    drainage_snow_mode = snow_fall + liquid_fall

    total_excess_rate = (total_water - liquid_capacity) / step_seconds
    drainage_legacy = jnp.where(
        intercept & ~snow_vegetation_enabled & (total_excess_rate > 0.0),
        total_excess_rate,
        zeros,
    )
    total_water = jnp.where(
        drainage_legacy > 0.0,
        liquid_capacity,
        total_water,
    )
    drainage = jnp.where(snow_vegetation_enabled, drainage_snow_mode, drainage_legacy)
    total_water = jnp.where(
        intercept & snow_vegetation_enabled,
        canopy_snow + canopy_liquid,
        total_water,
    )

    unload_branch = ~is_wall & exposed & snow_vegetation_enabled
    snow_temperature_unloading = jnp.where(
        unload_branch & (canopy_snow > 0.0),
        jnp.maximum(
            jnp.float64(0.0),
            fortran_divide(
                fortran_hardware_multiply(canopy_snow, temperature - jnp.float64(270.15)),
                jnp.float64(1.87e5),
            ),
        ),
        zeros,
    )
    snow_wind_unloading = jnp.where(
        unload_branch & (canopy_snow > 0.0),
        fortran_divide(
            fortran_hardware_multiply(jnp.float64(0.5) * canopy_snow, wind_speed),
            jnp.float64(1.56e5),
        ),
        zeros,
    )
    snow_unloaded = jnp.minimum(
        fortran_hardware_multiply(
            fortran_hardware_add(snow_wind_unloading, snow_temperature_unloading),
            step_seconds,
        ),
        jnp.maximum(canopy_snow, jnp.float64(0.0)),
    )
    snow_unloaded = jnp.where(unload_branch, snow_unloaded, zeros)
    canopy_snow = fortran_hardware_add(canopy_snow, -snow_unloaded)
    total_water = fortran_hardware_add(total_water, -snow_unloaded)

    safe_total_precipitation = jnp.where(
        positive_precipitation,
        rain + snow,
        jnp.float64(1.0),
    )
    snow_fraction = jnp.where(
        intercept,
        snow / safe_total_precipitation,
        zeros,
    )
    rain_fraction = jnp.where(
        intercept,
        rain / safe_total_precipitation,
        zeros,
    )
    ground_snow_exposed = jnp.where(
        snow_vegetation_enabled,
        snow_throughfall + snow_fall + snow_unloaded / step_seconds,
        snow_throughfall + drainage * snow_fraction,
    )
    ground_rain_exposed = jnp.where(
        snow_vegetation_enabled,
        rain_throughfall + liquid_fall,
        rain_throughfall + drainage * rain_fraction,
    )
    ground_snow = jnp.where(exposed, ground_snow_exposed, snow)
    ground_rain = jnp.where(exposed, ground_rain_exposed, rain)
    ground_snow = jnp.where(is_wall, zeros, ground_snow)
    ground_rain = jnp.where(is_wall, zeros, ground_rain) + irrigation

    return CanopyInterceptionResult(
        state=CanopyWaterState(total_water, canopy_snow, canopy_liquid, snow_unloaded),
        fluxes=CanopyPrecipitationFluxes(
            drainage=drainage,
            snow_fall=snow_fall,
            liquid_fall=liquid_fall,
            snow_wind_unloading=snow_wind_unloading,
            snow_temperature_unloading=snow_temperature_unloading,
            snow_throughfall=snow_throughfall,
            rain_throughfall=rain_throughfall,
            intercepted_precipitation=intercepted_precipitation,
            ground_snow=ground_snow,
            ground_rain=ground_rain,
            ground_precipitation=ground_snow + ground_rain,
        ),
    )


def canopy_wet_fractions(
    vegetation_exposed: jnp.ndarray,
    maximum_dew: jnp.ndarray,
    exposed_leaf_area_index: jnp.ndarray,
    exposed_stem_area_index: jnp.ndarray,
    total_canopy_water: jnp.ndarray,
    canopy_snow: jnp.ndarray,
    previous_snow_covered_fraction: jnp.ndarray,
    parameters: CanopyHydrologyParameters,
) -> CanopyWetFractions:

    exposed = jnp.asarray(vegetation_exposed, dtype=jnp.int32) == jnp.int32(1)
    maximum_dew = jnp.asarray(maximum_dew, dtype=jnp.float64)
    leaf_area = jnp.asarray(exposed_leaf_area_index, dtype=jnp.float64)
    stem_area = jnp.asarray(exposed_stem_area_index, dtype=jnp.float64)
    total_water = jnp.asarray(total_canopy_water, dtype=jnp.float64)
    canopy_snow = jnp.asarray(canopy_snow, dtype=jnp.float64)
    previous_snow_fraction = jnp.asarray(previous_snow_covered_fraction, dtype=jnp.float64)
    vegetation_area = leaf_area + stem_area
    has_water = exposed & (total_water > jnp.float64(0.0))
    safe_dew = jnp.where(has_water, maximum_dew, jnp.float64(1.0))
    safe_area = jnp.where(has_water, vegetation_area, jnp.float64(1.0))
    inverse_dew = fortran_divide(jnp.ones_like(safe_dew), safe_dew)
    wet_power_base = jnp.where(
        has_water,
        fortran_multiply(fortran_divide(inverse_dew, safe_area), total_water),
        jnp.float64(1.0),
    )
    power = fortran_power
    if default_backend() == "gpu":
        from clm_jax.physics.positive_power import positive_power
        power = positive_power
    wet = power(wet_power_base, jnp.float64(0.666666666666))
    wet = jnp.where(
        has_water,
        jnp.minimum(wet, jnp.asarray(parameters.maximum_leaf_wetted_fraction)),
        jnp.float64(0.0),
    )
    snow_radiation_enabled = (
        jnp.asarray(parameters.snow_vegetation_mode, dtype=jnp.int32)
        == SNOW_VEGETATION_ON_RADIATION
    )
    has_canopy_snow = has_water & (canopy_snow > jnp.float64(0.0))
    snow_power_base = jnp.where(
        snow_radiation_enabled & has_canopy_snow,
        fortran_multiply(fortran_divide(
            inverse_dew,
            fortran_multiply(fortran_multiply(safe_area, jnp.float64(6.0)),
                             jnp.float64(10.0))), canopy_snow),
        jnp.float64(1.0),
    )
    snow_covered_candidate = power(snow_power_base, jnp.float64(0.15))
    snow_covered_exposed = jnp.where(
        snow_radiation_enabled & has_canopy_snow,
        jnp.minimum(snow_covered_candidate, jnp.float64(1.0)),
        jnp.float64(0.0),
    )
    snow_covered = jnp.where(exposed, snow_covered_exposed, previous_snow_fraction)
    safe_leaf_and_stem = jnp.where(
        exposed & (vegetation_area > jnp.float64(0.0)),
        vegetation_area,
        jnp.float64(1.0),
    )
    dry_leaf = jnp.where(
        exposed,
        (jnp.float64(1.0) - wet) * leaf_area / safe_leaf_and_stem,
        jnp.float64(0.0),
    )
    return CanopyWetFractions(wet=wet, snow_covered=snow_covered, dry_leaf=dry_leaf)


def surface_water_fraction(
    landunit_type: jnp.ndarray,
    microtopography_sigma: jnp.ndarray,
    snow_water_equivalent: jnp.ndarray,
    surface_water: jnp.ndarray,
    top_soil_liquid: jnp.ndarray,
    snow_fraction: jnp.ndarray,
    effective_snow_fraction: jnp.ndarray,
    previous_surface_water_fraction_without_snow: jnp.ndarray,
    *,
    update_snow_fraction: bool = True,
) -> SurfaceWaterState:

    landunit_type = jnp.asarray(landunit_type, dtype=jnp.int32)
    microtopography_sigma = jnp.asarray(microtopography_sigma, dtype=jnp.float64)
    snow_water = jnp.asarray(snow_water_equivalent, dtype=jnp.float64)
    surface_water = jnp.asarray(surface_water, dtype=jnp.float64)
    top_soil_liquid = jnp.asarray(top_soil_liquid, dtype=jnp.float64)
    snow_fraction = jnp.asarray(snow_fraction, dtype=jnp.float64)
    effective_snow_fraction = jnp.asarray(effective_snow_fraction, dtype=jnp.float64)
    fraction_without_snow = jnp.asarray(
        previous_surface_water_fraction_without_snow,
        dtype=jnp.float64,
    )

    supported = (landunit_type == LANDUNIT_SOIL) | (landunit_type == LANDUNIT_CROP)
    has_surface_water = supported & (surface_water > jnp.float64(1.0e-8))
    sigma_millimeters = fortran_multiply(jnp.float64(1.0e3), microtopography_sigma)
    safe_sigma = jnp.where(has_surface_water, sigma_millimeters, jnp.float64(1.0))
    square_root_two = jnp.float64(1.41421353816986083984375)
    square_root_two_pi = jnp.sqrt(jnp.float64(2.0) * jnp.float64(PI))
    safe_surface_water = jnp.where(
        has_surface_water,
        surface_water,
        fortran_divide(safe_sigma, square_root_two_pi),
    )

    def newton_step(_iteration: int, depth: jnp.ndarray) -> jnp.ndarray:
        normalized_depth = fortran_divide(
            depth, fortran_multiply(safe_sigma, square_root_two)
        )
        error_function = erf(normalized_depth)
        gaussian_mass = fortran_add(jnp.float64(1.0), error_function)
        exponent = -fortran_divide(
            fortran_multiply(depth, depth),
            fortran_multiply(jnp.float64(2.0), fortran_multiply(safe_sigma, safe_sigma)),
        )
        residual = fortran_add(
            fortran_add(
                fortran_multiply(fortran_multiply(jnp.float64(0.5), depth), gaussian_mass),
                fortran_multiply(
                    fortran_divide(safe_sigma, square_root_two_pi),
                    strict_negative_exp(exponent),
                ),
            ),
            -safe_surface_water,
        )
        derivative = fortran_multiply(jnp.float64(0.5), gaussian_mass)
        return fortran_add(depth, -fortran_divide(residual, derivative))

    depth = lax.fori_loop(0, 10, newton_step, jnp.zeros_like(surface_water))
    submerged_candidate = fortran_multiply(
        jnp.float64(0.5),
        fortran_add(
            jnp.float64(1.0),
            erf(fortran_divide(depth, fortran_multiply(safe_sigma, square_root_two))),
        ),
    )
    submerged = jnp.where(has_surface_water, submerged_candidate, jnp.float64(0.0))

    transfer_to_soil = supported & ~has_surface_water
    top_soil_liquid = jnp.where(
        transfer_to_soil,
        top_soil_liquid + surface_water,
        top_soil_liquid,
    )
    surface_water = jnp.where(transfer_to_soil, jnp.float64(0.0), surface_water)
    fraction_without_snow = jnp.where(supported, submerged, fraction_without_snow)

    adjust_snow = (
        supported
        & jnp.asarray(update_snow_fraction, dtype=jnp.bool_)
        & (snow_fraction > (jnp.float64(1.0) - submerged))
        & (snow_water > jnp.float64(0.0))
    )
    adjusted_submerged = jnp.where(
        submerged > jnp.float64(0.01),
        jnp.maximum(jnp.float64(1.0) - snow_fraction, jnp.float64(0.01)),
        submerged,
    )
    adjusted_snow = jnp.float64(1.0) - adjusted_submerged
    submerged = jnp.where(adjust_snow, adjusted_submerged, submerged)
    snow_fraction = jnp.where(adjust_snow, adjusted_snow, snow_fraction)
    effective_snow_fraction = jnp.where(
        adjust_snow,
        adjusted_snow,
        effective_snow_fraction,
    )

    return SurfaceWaterState(
        surface_water=surface_water,
        top_soil_liquid=top_soil_liquid,
        snow_fraction=snow_fraction,
        effective_snow_fraction=effective_snow_fraction,
        surface_water_fraction=submerged,
        surface_water_fraction_without_snow=fraction_without_snow,
    )
