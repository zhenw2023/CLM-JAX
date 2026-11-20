
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import default_backend, lax, tree_util

from clm_jax.constants import GRAVITY, PI, SPECIFIC_HEAT_DRY_AIR, VON_KARMAN
from clm_jax.physics.friction_velocity import (
    FrictionVelocityState,
    friction_velocity,
    monin_obukhov_initialize,
)
from clm_jax.physics.human_index import FastHumanStressIndices, fast_human_stress_indices
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_materialize,
    fortran_multiply,
    fortran_rounding_barrier,
    strict_exp,
)
from clm_jax.physics.qsat import qsat

NUMBER_OF_STABILITY_ITERATIONS = 3
BUILDING_TEMPERATURE_METHOD_SIMPLE = 0
BUILDING_TEMPERATURE_METHOD_PROGNOSTIC = 1
DRY_ADIABATIC_LAPSE_RATE = 0.0098
URBAN_PONDING_DEPTH = 1.0
HEATING_WASTE_HEAT_FACTOR = 0.2
AIR_CONDITIONING_WASTE_HEAT_FACTOR = 0.6
WASTE_HEAT_LIMIT = 100.0

MACDONALD_ALPHA = 4.43
MACDONALD_BETA = 1.0
URBAN_DRAG_COEFFICIENT = 1.2

URBAN_SURFACE_ROOF = 0
URBAN_SURFACE_SUN_WALL = 1
URBAN_SURFACE_SHADE_WALL = 2
URBAN_SURFACE_IMPERVIOUS_ROAD = 3
URBAN_SURFACE_PERVIOUS_ROAD = 4
NUMBER_OF_URBAN_SURFACES = 5


def _urban_power(base, exponent):
    if default_backend() == "gpu":
        from clm_jax.physics.native_power import native_power
        return native_power(base, exponent)
    return jnp.power(base, exponent)


class UrbanAerodynamicGeometry(NamedTuple):

    plan_area_index: jnp.ndarray
    frontal_area_index: jnp.ndarray
    displacement_height: jnp.ndarray
    roughness_length: jnp.ndarray


class SimpleWasteHeatFluxes(NamedTuple):

    waste_heat: jnp.ndarray
    heat_from_air_conditioning: jnp.ndarray


class UrbanTurbulenceInputs(NamedTuple):

    active_landunit: jnp.ndarray
    eastward_wind_landunit: jnp.ndarray
    northward_wind_landunit: jnp.ndarray
    atmospheric_temperature_landunit: jnp.ndarray
    atmospheric_potential_temperature_landunit: jnp.ndarray
    atmospheric_density_landunit: jnp.ndarray
    atmospheric_specific_humidity_landunit: jnp.ndarray
    forcing_height_wind_landunit: jnp.ndarray
    forcing_height_temperature_landunit: jnp.ndarray
    forcing_height_humidity_landunit: jnp.ndarray
    momentum_roughness_length_landunit: jnp.ndarray
    displacement_height_landunit: jnp.ndarray
    roof_height_landunit: jnp.ndarray
    roof_fraction_landunit: jnp.ndarray
    canyon_height_width_ratio_landunit: jnp.ndarray
    pervious_road_fraction_landunit: jnp.ndarray
    canyon_wind_height_landunit: jnp.ndarray
    traffic_heat_factor_landunit: jnp.ndarray
    ground_temperature_surface: jnp.ndarray
    ground_specific_humidity_surface: jnp.ndarray
    snow_depth_surface: jnp.ndarray
    surface_liquid_water_surface: jnp.ndarray
    surface_ice_water_surface: jnp.ndarray
    air_conditioning_flux_surface: jnp.ndarray
    heating_flux_surface: jnp.ndarray
    air_conditioning_flux_landunit: jnp.ndarray
    heating_flux_landunit: jnp.ndarray


class UrbanTurbulenceState(NamedTuple):

    canopy_air_temperature_landunit: jnp.ndarray
    canopy_air_specific_humidity_landunit: jnp.ndarray
    friction_velocity: FrictionVelocityState


class UrbanSurfaceConductances(NamedTuple):

    sensible_scaled_surface: jnp.ndarray
    latent_scaled_surface: jnp.ndarray
    sensible_unscaled_surface: jnp.ndarray
    latent_unscaled_surface: jnp.ndarray


class UrbanTurbulenceDiagnostics(NamedTuple):
    reference_wind_speed_landunit: jnp.ndarray
    stability_wind_speed_landunit: jnp.ndarray
    monin_obukhov_length_landunit: jnp.ndarray
    stability_wind_iteration_landunit: jnp.ndarray
    monin_obukhov_length_iteration_landunit: jnp.ndarray
    friction_velocity_iteration_landunit: jnp.ndarray
    thermal_resistance_iteration_landunit: jnp.ndarray
    canopy_air_temperature_iteration_landunit: jnp.ndarray
    canopy_air_specific_humidity_iteration_landunit: jnp.ndarray
    surface_latent_conductance_iteration: jnp.ndarray
    surface_wet_fraction_iteration: jnp.ndarray
    canyon_top_wind_landunit: jnp.ndarray
    canyon_wind_landunit: jnp.ndarray
    canyon_resistance_landunit: jnp.ndarray
    friction_velocity_landunit: jnp.ndarray
    momentum_resistance_landunit: jnp.ndarray
    thermal_resistance_landunit: jnp.ndarray
    moisture_resistance_landunit: jnp.ndarray
    atmospheric_sensible_conductance_landunit: jnp.ndarray
    atmospheric_latent_conductance_landunit: jnp.ndarray
    sensible_conductance_sum_landunit: jnp.ndarray
    latent_conductance_sum_landunit: jnp.ndarray
    surface_conductances: UrbanSurfaceConductances
    surface_wet_fraction: jnp.ndarray
    waste_heat_landunit: jnp.ndarray
    heat_from_air_conditioning_landunit: jnp.ndarray
    traffic_heat_landunit: jnp.ndarray
    sensible_conservation_error_landunit: jnp.ndarray
    moisture_conservation_error_landunit: jnp.ndarray


class UrbanTurbulenceResult(NamedTuple):
    state: UrbanTurbulenceState
    diagnostics: UrbanTurbulenceDiagnostics


class UrbanSurfaceFluxInputs(NamedTuple):

    atmospheric_pressure_landunit: jnp.ndarray
    ground_specific_humidity_temperature_derivative_surface: jnp.ndarray
    latent_heat_surface: jnp.ndarray
    snow_fraction_surface: jnp.ndarray
    soil_evaporation_alpha_surface: jnp.ndarray
    inner_layer_temperature_surface: jnp.ndarray
    pervious_road_root_fraction_landunit_level: jnp.ndarray
    internal_building_temperature_landunit: jnp.ndarray


class UrbanSurfaceEnergyFluxes(NamedTuple):
    aerodynamic_resistance_surface: jnp.ndarray
    downward_longwave_below_canopy_surface: jnp.ndarray
    upward_longwave_above_canopy_surface: jnp.ndarray
    ground_sensible_derivative_surface: jnp.ndarray
    ground_latent_derivative_surface: jnp.ndarray
    ground_energy_derivative_surface: jnp.ndarray
    eastward_wind_stress_surface: jnp.ndarray
    northward_wind_stress_surface: jnp.ndarray
    sensible_heat_ground_surface: jnp.ndarray
    sensible_heat_total_surface: jnp.ndarray
    urban_sensible_heat_total_surface: jnp.ndarray
    sensible_heat_snow_surface: jnp.ndarray
    sensible_heat_soil_surface: jnp.ndarray
    sensible_heat_surface_water_surface: jnp.ndarray


class UrbanSurfaceWaterFluxes(NamedTuple):
    soil_evaporation_surface: jnp.ndarray
    vegetation_transpiration_surface: jnp.ndarray
    vegetation_evaporation_surface: jnp.ndarray


class UrbanSurfaceAtmosphericState(NamedTuple):
    reference_temperature_surface: jnp.ndarray
    urban_reference_temperature_surface: jnp.ndarray
    vegetation_temperature_surface: jnp.ndarray
    reference_specific_humidity_surface: jnp.ndarray
    reference_relative_humidity_surface: jnp.ndarray
    urban_reference_relative_humidity_surface: jnp.ndarray


class UrbanSurfaceFluxResult(NamedTuple):
    energy: UrbanSurfaceEnergyFluxes
    water: UrbanSurfaceWaterFluxes
    atmosphere: UrbanSurfaceAtmosphericState
    human_stress: FastHumanStressIndices
    urban_human_stress: FastHumanStressIndices
    internal_building_temperature_landunit: jnp.ndarray
    effective_root_fraction_surface_level: jnp.ndarray
    scaled_sensible_heat_surface: jnp.ndarray
    scaled_moisture_flux_surface: jnp.ndarray


def _add(*terms: jnp.ndarray) -> jnp.ndarray:
    values = jnp.broadcast_arrays(*(jnp.asarray(term, dtype=jnp.float64) for term in terms))
    result = values[0]
    for value in values[1:]:
        result = fortran_add(result, value)
    return lax.optimization_barrier(result)


def _multiply(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    left, right = jnp.broadcast_arrays(
        jnp.asarray(left, dtype=jnp.float64),
        jnp.asarray(right, dtype=jnp.float64),
    )
    return lax.optimization_barrier(fortran_multiply(left, right))


def urban_aerodynamic_parameters(
    canyon_height_width_ratio: jnp.ndarray,
    roof_height: jnp.ndarray,
) -> UrbanAerodynamicGeometry:

    ratio, height = jnp.broadcast_arrays(
        jnp.asarray(canyon_height_width_ratio, dtype=jnp.float64),
        jnp.asarray(roof_height, dtype=jnp.float64),
    )
    one = jnp.float64(1.0)
    plan_area = fortran_divide(ratio, _add(ratio, one))

    frontal_area = _multiply(_add(one, -plan_area), ratio)
    frontal_area = _multiply(
        _multiply(frontal_area, jnp.sqrt(fortran_divide(one, plan_area))),
        jnp.sqrt(plan_area),
    )

    alpha_power = _urban_power(jnp.float64(MACDONALD_ALPHA), -plan_area)
    displacement_fraction = _add(
        one,
        _multiply(alpha_power, _add(plan_area, -one)),
    )
    displacement = _multiply(displacement_fraction, height)
    remaining_height_fraction = _add(
        one,
        -fortran_divide(displacement, height),
    )

    von_karman_squared = _multiply(jnp.float64(VON_KARMAN), jnp.float64(VON_KARMAN))
    exponent_base = _multiply(
        _multiply(
            fortran_divide(
                _multiply(
                    _multiply(jnp.float64(0.5), jnp.float64(MACDONALD_BETA)),
                    jnp.float64(URBAN_DRAG_COEFFICIENT),
                ),
                von_karman_squared,
            ),
            remaining_height_fraction,
        ),
        frontal_area,
    )
    roughness = _multiply(
        _multiply(height, remaining_height_fraction),
        strict_exp(-_urban_power(exponent_base, jnp.float64(-0.5))),
    )
    return UrbanAerodynamicGeometry(
        plan_area_index=plan_area,
        frontal_area_index=frontal_area,
        displacement_height=displacement,
        roughness_length=roughness,
    )


def simple_waste_heat_from_air_conditioning(
    air_conditioning_flux: jnp.ndarray,
    heating_flux: jnp.ndarray,
) -> SimpleWasteHeatFluxes:

    air_conditioning, heating = jnp.broadcast_arrays(
        jnp.asarray(air_conditioning_flux, dtype=jnp.float64),
        jnp.asarray(heating_flux, dtype=jnp.float64),
    )
    waste_heat = _add(
        _multiply(jnp.float64(AIR_CONDITIONING_WASTE_HEAT_FACTOR), air_conditioning),
        _multiply(jnp.float64(HEATING_WASTE_HEAT_FACTOR), heating),
    )
    return SimpleWasteHeatFluxes(
        waste_heat=waste_heat,
        heat_from_air_conditioning=jnp.abs(air_conditioning),
    )


def aggregate_simple_waste_heat(
    roof_fraction: jnp.ndarray,
    canyon_height_width_ratio: jnp.ndarray,
    roof: SimpleWasteHeatFluxes,
    sun_wall: SimpleWasteHeatFluxes,
    shade_wall: SimpleWasteHeatFluxes,
) -> SimpleWasteHeatFluxes:

    roof_weight, ratio = jnp.broadcast_arrays(
        jnp.asarray(roof_fraction, dtype=jnp.float64),
        jnp.asarray(canyon_height_width_ratio, dtype=jnp.float64),
    )
    canyon_weight = _add(jnp.float64(1.0), -roof_weight)

    def aggregate(
        roof_flux: jnp.ndarray,
        sun_wall_flux: jnp.ndarray,
        shade_wall_flux: jnp.ndarray,
    ) -> jnp.ndarray:
        wall_flux = _multiply(ratio, _add(sun_wall_flux, shade_wall_flux))
        return _add(
            _multiply(roof_weight, roof_flux),
            _multiply(canyon_weight, wall_flux),
        )

    waste_heat = jnp.minimum(
        aggregate(roof.waste_heat, sun_wall.waste_heat, shade_wall.waste_heat),
        jnp.float64(WASTE_HEAT_LIMIT),
    )
    heat_from_air_conditioning = aggregate(
        roof.heat_from_air_conditioning,
        sun_wall.heat_from_air_conditioning,
        shade_wall.heat_from_air_conditioning,
    )
    return SimpleWasteHeatFluxes(
        waste_heat=waste_heat,
        heat_from_air_conditioning=heat_from_air_conditioning,
    )


def simple_internal_building_temperature(
    roof_height: jnp.ndarray,
    canyon_height_width_ratio: jnp.ndarray,
    roof_fraction: jnp.ndarray,
    inner_roof_temperature: jnp.ndarray,
    inner_sun_wall_temperature: jnp.ndarray,
    inner_shade_wall_temperature: jnp.ndarray,
) -> jnp.ndarray:

    (
        height,
        ratio,
        roof_weight,
        roof_temperature,
        sun_wall_temperature,
        shade_wall_temperature,
    ) = jnp.broadcast_arrays(
        jnp.asarray(roof_height, dtype=jnp.float64),
        jnp.asarray(canyon_height_width_ratio, dtype=jnp.float64),
        jnp.asarray(roof_fraction, dtype=jnp.float64),
        jnp.asarray(inner_roof_temperature, dtype=jnp.float64),
        jnp.asarray(inner_sun_wall_temperature, dtype=jnp.float64),
        jnp.asarray(inner_shade_wall_temperature, dtype=jnp.float64),
    )
    roof_length = _multiply(fortran_divide(height, ratio), roof_weight)
    roof_length = fortran_divide(
        roof_length,
        _add(jnp.float64(1.0), -roof_weight),
    )
    wall_contribution = _multiply(
        height,
        _add(shade_wall_temperature, sun_wall_temperature),
    )
    roof_contribution = _multiply(roof_length, roof_temperature)
    numerator = _add(wall_contribution, roof_contribution)
    denominator = _add(_multiply(jnp.float64(2.0), height), roof_length)
    return fortran_divide(numerator, denominator)


def _validate_turbulence_shapes(
    inputs: UrbanTurbulenceInputs,
    state: UrbanTurbulenceState,
) -> None:
    number_of_landunits = inputs.active_landunit.shape[0]
    if inputs.active_landunit.shape != (number_of_landunits,):
        raise ValueError("active_landunit must be one-dimensional")
    surface_fields = {
        "ground_temperature_surface",
        "ground_specific_humidity_surface",
        "snow_depth_surface",
        "surface_liquid_water_surface",
        "surface_ice_water_surface",
        "air_conditioning_flux_surface",
        "heating_flux_surface",
    }
    for name in UrbanTurbulenceInputs._fields:
        if name == "active_landunit":
            continue
        expected = (
            (number_of_landunits, NUMBER_OF_URBAN_SURFACES)
            if name in surface_fields
            else (number_of_landunits,)
        )
        if getattr(inputs, name).shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    for name in (
        "canopy_air_temperature_landunit",
        "canopy_air_specific_humidity_landunit",
    ):
        if getattr(state, name).shape != (number_of_landunits,):
            raise ValueError(f"{name} must have shape ({number_of_landunits},)")
    for name, value in zip(
        FrictionVelocityState._fields,
        state.friction_velocity,
        strict=True,
    ):
        if value.shape != (number_of_landunits,):
            raise ValueError(f"friction field {name} must have shape ({number_of_landunits},)")


def _surface_wet_fraction(
    snow_depth: jnp.ndarray,
    liquid_water: jnp.ndarray,
    ice_water: jnp.ndarray,
    canopy_air_humidity: jnp.ndarray,
    ground_humidity: jnp.ndarray,
) -> jnp.ndarray:
    water_depth = jnp.maximum(
        jnp.float64(0.0),
        _add(liquid_water, ice_water),
    )
    water_wetness = _urban_power(
        fortran_divide(water_depth, jnp.float64(URBAN_PONDING_DEPTH)),
        jnp.float64(0.666666666666),
    )
    snow_wetness = jnp.minimum(
        fortran_divide(snow_depth, jnp.float64(0.05)),
        jnp.float64(1.0),
    )
    wetness = jnp.where(snow_depth > jnp.float64(0.0), snow_wetness, water_wetness)
    wetness = jnp.minimum(wetness, jnp.float64(1.0))
    return jnp.where(canopy_air_humidity[:, None] > ground_humidity, 1.0, wetness)


def _urban_canyon_wind(
    reference_wind: jnp.ndarray,
    forcing_height_wind: jnp.ndarray,
    roughness: jnp.ndarray,
    displacement: jnp.ndarray,
    roof_height: jnp.ndarray,
    canyon_ratio: jnp.ndarray,
    canyon_wind_height: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    roof_log = jnp.log(fortran_divide(_add(roof_height, -displacement), roughness))
    forcing_log = jnp.log(fortran_divide(_add(forcing_height_wind, -displacement), roughness))
    canyon_top = fortran_divide(_multiply(reference_wind, roof_log), forcing_log)
    attenuation_exponent = _multiply(
        _multiply(jnp.float64(-0.5), canyon_ratio),
        _add(
            jnp.float64(1.0),
            -fortran_divide(canyon_wind_height, roof_height),
        ),
    )
    attenuation = strict_exp(attenuation_exponent)
    isolated = _multiply(canyon_top, attenuation)
    wake_geometry = _add(
        jnp.float64(1.0),
        _multiply(
            _multiply(
                jnp.float64(2.0),
                _add(
                    fortran_divide(jnp.float64(2.0), jnp.float64(PI)),
                    jnp.float64(-1.0),
                ),
            ),
            _add(
                fortran_divide(
                    roof_height,
                    fortran_divide(roof_height, canyon_ratio),
                ),
                jnp.float64(-0.5),
            ),
        ),
    )
    wake = _multiply(_multiply(canyon_top, wake_geometry), attenuation)
    skimming = _multiply(
        _multiply(canyon_top, fortran_divide(jnp.float64(2.0), jnp.float64(PI))),
        attenuation,
    )
    canyon_horizontal = jnp.where(
        canyon_ratio < jnp.float64(0.5),
        isolated,
        jnp.where(canyon_ratio < jnp.float64(1.0), wake, skimming),
    )
    return canyon_top, canyon_horizontal


def _surface_conductances(
    roof_fraction: jnp.ndarray,
    pervious_road_fraction: jnp.ndarray,
    canyon_ratio: jnp.ndarray,
    canyon_resistance: jnp.ndarray,
    wet_fraction: jnp.ndarray,
) -> UrbanSurfaceConductances:
    one_minus_roof = _add(jnp.float64(1.0), -roof_fraction)
    one_minus_pervious = _add(jnp.float64(1.0), -pervious_road_fraction)
    inverse_canyon_resistance = fortran_divide(jnp.float64(1.0), canyon_resistance)

    sensible_roof = fortran_divide(roof_fraction, canyon_resistance)
    sensible_wall = fortran_divide(_multiply(canyon_ratio, one_minus_roof), canyon_resistance)
    sensible_impervious = fortran_divide(
        _multiply(one_minus_pervious, one_minus_roof), canyon_resistance
    )
    sensible_pervious = fortran_divide(
        _multiply(pervious_road_fraction, one_minus_roof), canyon_resistance
    )
    sensible_scaled = jnp.stack(
        (
            sensible_roof,
            sensible_wall,
            sensible_wall,
            sensible_impervious,
            sensible_pervious,
        ),
        axis=1,
    )
    sensible_unscaled = jnp.broadcast_to(
        inverse_canyon_resistance[:, None],
        sensible_scaled.shape,
    )

    latent_roof = _multiply(wet_fraction[:, URBAN_SURFACE_ROOF], sensible_roof)
    latent_impervious = fortran_divide(
        _multiply(
            _multiply(
                wet_fraction[:, URBAN_SURFACE_IMPERVIOUS_ROAD],
                one_minus_pervious,
            ),
            one_minus_roof,
        ),
        canyon_resistance,
    )
    latent_scaled = jnp.stack(
        (
            latent_roof,
            jnp.zeros_like(roof_fraction),
            jnp.zeros_like(roof_fraction),
            latent_impervious,
            sensible_pervious,
        ),
        axis=1,
    )
    latent_unscaled = jnp.stack(
        (
            _multiply(wet_fraction[:, URBAN_SURFACE_ROOF], inverse_canyon_resistance),
            jnp.zeros_like(roof_fraction),
            jnp.zeros_like(roof_fraction),
            _multiply(
                wet_fraction[:, URBAN_SURFACE_IMPERVIOUS_ROAD],
                inverse_canyon_resistance,
            ),
            inverse_canyon_resistance,
        ),
        axis=1,
    )
    return UrbanSurfaceConductances(
        sensible_scaled_surface=sensible_scaled,
        latent_scaled_surface=latent_scaled,
        sensible_unscaled_surface=sensible_unscaled,
        latent_unscaled_surface=latent_unscaled,
    )


def _native_surface_sum(initial: jnp.ndarray, terms: jnp.ndarray) -> jnp.ndarray:

    result = initial
    for surface in range(NUMBER_OF_URBAN_SURFACES):
        result = _add(result, terms[:, surface])
    return result


def urban_turbulent_fluxes(
    inputs: UrbanTurbulenceInputs,
    state: UrbanTurbulenceState,
    maximum_stable_zeta: float | jnp.ndarray = 0.5,
    building_temperature_method: int | jnp.ndarray = BUILDING_TEMPERATURE_METHOD_PROGNOSTIC,
) -> UrbanTurbulenceResult:

    _validate_turbulence_shapes(inputs, state)
    inputs = UrbanTurbulenceInputs(
        *(
            jnp.asarray(value, dtype=jnp.bool_)
            if name == "active_landunit"
            else jnp.asarray(value, dtype=jnp.float64)
            for name, value in zip(UrbanTurbulenceInputs._fields, inputs, strict=True)
        )
    )
    active = inputs.active_landunit

    def safe_landunit(value: jnp.ndarray, fallback: float) -> jnp.ndarray:
        return jnp.where(active, value, jnp.float64(fallback))

    def safe_surface(value: jnp.ndarray, fallback: float) -> jnp.ndarray:
        return jnp.where(active[:, None], value, jnp.float64(fallback))

    eastward_wind = safe_landunit(inputs.eastward_wind_landunit, 1.0)
    northward_wind = safe_landunit(inputs.northward_wind_landunit, 0.0)
    atmospheric_temperature = safe_landunit(inputs.atmospheric_temperature_landunit, 280.0)
    potential_temperature = safe_landunit(inputs.atmospheric_potential_temperature_landunit, 280.0)
    density = safe_landunit(inputs.atmospheric_density_landunit, 1.2)
    atmospheric_humidity = safe_landunit(inputs.atmospheric_specific_humidity_landunit, 0.005)
    forcing_height_wind = safe_landunit(inputs.forcing_height_wind_landunit, 30.0)
    forcing_height_temperature = safe_landunit(inputs.forcing_height_temperature_landunit, 30.0)
    forcing_height_humidity = safe_landunit(inputs.forcing_height_humidity_landunit, 30.0)
    roughness = safe_landunit(inputs.momentum_roughness_length_landunit, 0.1)
    displacement = safe_landunit(inputs.displacement_height_landunit, 0.0)
    roof_height = safe_landunit(inputs.roof_height_landunit, 10.0)
    roof_fraction = safe_landunit(inputs.roof_fraction_landunit, 0.3)
    canyon_ratio = safe_landunit(inputs.canyon_height_width_ratio_landunit, 1.0)
    pervious_fraction = safe_landunit(inputs.pervious_road_fraction_landunit, 0.5)
    canyon_wind_height = safe_landunit(inputs.canyon_wind_height_landunit, 2.0)
    traffic_factor = safe_landunit(inputs.traffic_heat_factor_landunit, 0.0)
    ground_temperature = safe_surface(inputs.ground_temperature_surface, 280.0)
    ground_humidity = safe_surface(inputs.ground_specific_humidity_surface, 0.005)
    snow_depth = safe_surface(inputs.snow_depth_surface, 0.0)
    liquid_water = safe_surface(inputs.surface_liquid_water_surface, 0.0)
    ice_water = safe_surface(inputs.surface_ice_water_surface, 0.0)

    reference_wind = jnp.maximum(
        jnp.float64(1.0),
        jnp.sqrt(
            _add(
                _multiply(eastward_wind, eastward_wind),
                _multiply(northward_wind, northward_wind),
            )
        ),
    )
    canyon_top_wind, canyon_horizontal_wind = _urban_canyon_wind(
        reference_wind,
        forcing_height_wind,
        roughness,
        displacement,
        roof_height,
        canyon_ratio,
        canyon_wind_height,
    )

    atmospheric_reference_temperature = _add(
        atmospheric_temperature,
        _multiply(jnp.float64(DRY_ADIABATIC_LAPSE_RATE), forcing_height_temperature),
    )
    atmospheric_virtual_temperature = _multiply(
        potential_temperature,
        _add(
            jnp.float64(1.0),
            _multiply(jnp.float64(0.61), atmospheric_humidity),
        ),
    )
    canopy_temperature = safe_landunit(state.canopy_air_temperature_landunit, 280.0)
    canopy_humidity = safe_landunit(state.canopy_air_specific_humidity_landunit, 0.005)
    temperature_difference = _add(atmospheric_reference_temperature, -canopy_temperature)
    humidity_difference = _add(atmospheric_humidity, -canopy_humidity)
    virtual_temperature_difference = _add(
        _multiply(
            temperature_difference,
            _add(
                jnp.float64(1.0),
                _multiply(jnp.float64(0.61), atmospheric_humidity),
            ),
        ),
        _multiply(
            _multiply(jnp.float64(0.61), potential_temperature),
            humidity_difference,
        ),
    )
    reference_height = _add(forcing_height_wind, -displacement)
    initialized = monin_obukhov_initialize(
        reference_wind,
        atmospheric_virtual_temperature,
        virtual_temperature_difference,
        reference_height,
        roughness,
        maximum_stable_zeta,
    )

    zero_landunit = jnp.zeros_like(reference_wind)
    zero_surface = jnp.zeros_like(ground_temperature)
    initial_iteration_state = (
        initialized.stability_wind_speed,
        initialized.monin_obukhov_length,
        canopy_temperature,
        canopy_humidity,
        state.friction_velocity,
        zero_landunit,
        zero_landunit,
        zero_landunit,
        zero_landunit,
        zero_landunit,
        zero_landunit,
        zero_landunit,
        zero_landunit,
        zero_landunit,
        UrbanSurfaceConductances(zero_surface, zero_surface, zero_surface, zero_surface),
        zero_surface,
        zero_landunit,
        zero_landunit,
        zero_landunit,
        zero_landunit,
    )

    simple_surface_waste = simple_waste_heat_from_air_conditioning(
        inputs.air_conditioning_flux_surface,
        inputs.heating_flux_surface,
    )
    simple_landunit_waste = aggregate_simple_waste_heat(
        roof_fraction,
        canyon_ratio,
        SimpleWasteHeatFluxes(
            simple_surface_waste.waste_heat[:, URBAN_SURFACE_ROOF],
            simple_surface_waste.heat_from_air_conditioning[:, URBAN_SURFACE_ROOF],
        ),
        SimpleWasteHeatFluxes(
            simple_surface_waste.waste_heat[:, URBAN_SURFACE_SUN_WALL],
            simple_surface_waste.heat_from_air_conditioning[:, URBAN_SURFACE_SUN_WALL],
        ),
        SimpleWasteHeatFluxes(
            simple_surface_waste.waste_heat[:, URBAN_SURFACE_SHADE_WALL],
            simple_surface_waste.heat_from_air_conditioning[:, URBAN_SURFACE_SHADE_WALL],
        ),
    )
    prognostic_landunit_waste = simple_waste_heat_from_air_conditioning(
        inputs.air_conditioning_flux_landunit,
        inputs.heating_flux_landunit,
    )
    prognostic_landunit_waste = SimpleWasteHeatFluxes(
        waste_heat=jnp.minimum(
            prognostic_landunit_waste.waste_heat,
            jnp.float64(WASTE_HEAT_LIMIT),
        ),
        heat_from_air_conditioning=prognostic_landunit_waste.heat_from_air_conditioning,
    )
    simple_building = jnp.asarray(building_temperature_method, dtype=jnp.int32) == jnp.int32(
        BUILDING_TEMPERATURE_METHOD_SIMPLE
    )
    landunit_waste = tree_util.tree_map(
        lambda simple, prognostic: jnp.where(simple_building, simple, prognostic),
        simple_landunit_waste,
        prognostic_landunit_waste,
    )
    traffic_heat = _multiply(
        _multiply(
            _add(jnp.float64(1.0), -roof_fraction),
            _add(jnp.float64(1.0), -pervious_fraction),
        ),
        traffic_factor,
    )

    def stability_iteration(iteration_state, zero_based_iteration):
        (
            stability_wind,
            monin_length,
            previous_canopy_temperature,
            previous_canopy_humidity,
            friction_state,
            _friction,
            _momentum_resistance,
            _thermal_resistance,
            _moisture_resistance,
            _canyon_wind,
            _canyon_resistance,
            _atmospheric_sensible_conductance,
            _atmospheric_latent_conductance,
            _sensible_sum,
            _conductances,
            _wet_fraction,
            _latent_sum,
            _temperature_profile,
            _humidity_profile,
            _dummy,
        ) = iteration_state
        friction_result = friction_velocity(
            active,
            displacement,
            roughness,
            roughness,
            roughness,
            monin_length,
            zero_based_iteration + jnp.int32(1),
            reference_wind,
            stability_wind,
            forcing_height_wind,
            forcing_height_temperature,
            forcing_height_humidity,
            friction_state,
        )
        friction = fortran_materialize(
            fortran_rounding_barrier(
                jnp.where(active, friction_result.friction_velocity_patch, 0.1)
            )
        )
        temperature_profile = fortran_materialize(
            fortran_rounding_barrier(
                jnp.where(active, friction_result.temperature_profile_patch, 0.1)
            )
        )
        humidity_profile = fortran_materialize(
            fortran_rounding_barrier(jnp.where(active, friction_result.humidity_profile_patch, 0.1))
        )
        momentum_resistance = fortran_divide(
            jnp.float64(1.0),
            fortran_divide(_multiply(friction, friction), stability_wind),
        )
        thermal_resistance = fortran_divide(
            jnp.float64(1.0),
            _multiply(temperature_profile, friction),
        )
        moisture_resistance = fortran_divide(
            jnp.float64(1.0),
            _multiply(humidity_profile, friction),
        )
        canyon_wind = jnp.sqrt(
            _add(
                _multiply(canyon_horizontal_wind, canyon_horizontal_wind),
                _multiply(friction, friction),
            )
        )
        canyon_resistance = fortran_divide(
            _multiply(jnp.float64(SPECIFIC_HEAT_DRY_AIR), density),
            _add(jnp.float64(11.8), _multiply(jnp.float64(4.2), canyon_wind)),
        )
        wet_fraction = _surface_wet_fraction(
            snow_depth,
            liquid_water,
            ice_water,
            previous_canopy_humidity,
            ground_humidity,
        )
        conductances = _surface_conductances(
            roof_fraction,
            pervious_fraction,
            canyon_ratio,
            canyon_resistance,
            wet_fraction,
        )
        atmospheric_sensible_conductance = fortran_divide(jnp.float64(1.0), thermal_resistance)
        atmospheric_latent_conductance = fortran_divide(jnp.float64(1.0), moisture_resistance)
        temperature_numerator = fortran_divide(
            atmospheric_reference_temperature,
            thermal_resistance,
        )
        temperature_denominator = atmospheric_sensible_conductance
        humidity_numerator = fortran_divide(atmospheric_humidity, moisture_resistance)
        humidity_denominator = atmospheric_latent_conductance
        temperature_numerator = _native_surface_sum(
            temperature_numerator,
            _multiply(
                ground_temperature,
                conductances.sensible_scaled_surface,
            ),
        )
        temperature_denominator = _native_surface_sum(
            temperature_denominator,
            conductances.sensible_scaled_surface,
        )
        humidity_numerator = _native_surface_sum(
            humidity_numerator,
            _multiply(
                ground_humidity,
                conductances.latent_scaled_surface,
            ),
        )
        humidity_denominator = _native_surface_sum(
            humidity_denominator,
            conductances.latent_scaled_surface,
        )
        new_canopy_temperature = fortran_divide(temperature_numerator, temperature_denominator)
        new_canopy_humidity = fortran_divide(humidity_numerator, humidity_denominator)

        sensible_sum = _add(
            atmospheric_sensible_conductance,
            conductances.sensible_scaled_surface[:, URBAN_SURFACE_ROOF],
            conductances.sensible_scaled_surface[:, URBAN_SURFACE_PERVIOUS_ROAD],
            conductances.sensible_scaled_surface[:, URBAN_SURFACE_IMPERVIOUS_ROAD],
            conductances.sensible_scaled_surface[:, URBAN_SURFACE_SUN_WALL],
            conductances.sensible_scaled_surface[:, URBAN_SURFACE_SHADE_WALL],
        )
        latent_sum = _add(
            atmospheric_latent_conductance,
            conductances.latent_scaled_surface[:, URBAN_SURFACE_ROOF],
            conductances.latent_scaled_surface[:, URBAN_SURFACE_PERVIOUS_ROAD],
            conductances.latent_scaled_surface[:, URBAN_SURFACE_IMPERVIOUS_ROAD],
            conductances.latent_scaled_surface[:, URBAN_SURFACE_SUN_WALL],
            conductances.latent_scaled_surface[:, URBAN_SURFACE_SHADE_WALL],
        )

        temperature_difference = _add(atmospheric_reference_temperature, -new_canopy_temperature)
        humidity_difference = _add(atmospheric_humidity, -new_canopy_humidity)
        temperature_scale = _multiply(temperature_profile, temperature_difference)
        humidity_scale = _multiply(humidity_profile, humidity_difference)
        virtual_temperature_scale = _add(
            _multiply(
                temperature_scale,
                _add(
                    jnp.float64(1.0),
                    _multiply(jnp.float64(0.61), atmospheric_humidity),
                ),
            ),
            _multiply(
                _multiply(jnp.float64(0.61), potential_temperature),
                humidity_scale,
            ),
        )
        zeta = fortran_divide(
            _multiply(
                _multiply(
                    _multiply(reference_height, jnp.float64(VON_KARMAN)),
                    jnp.float64(GRAVITY),
                ),
                virtual_temperature_scale,
            ),
            _multiply(_multiply(friction, friction), atmospheric_virtual_temperature),
        )
        stable = zeta >= jnp.float64(0.0)
        bounded_zeta = jnp.where(
            stable,
            jnp.minimum(
                jnp.asarray(maximum_stable_zeta, dtype=jnp.float64),
                jnp.maximum(zeta, jnp.float64(0.01)),
            ),
            jnp.maximum(jnp.float64(-100.0), jnp.minimum(zeta, jnp.float64(-0.01))),
        )
        convective_argument = fortran_divide(
            -_multiply(
                _multiply(
                    _multiply(jnp.float64(GRAVITY), friction),
                    virtual_temperature_scale,
                ),
                jnp.float64(1000.0),
            ),
            atmospheric_virtual_temperature,
        )
        safe_convective_argument = jnp.where(
            active & ~stable,
            jnp.maximum(convective_argument, jnp.float64(0.0)),
            jnp.float64(1.0),
        )
        convective_velocity = _urban_power(
            safe_convective_argument,
            jnp.float64(0.333),
        )
        unstable_wind = jnp.sqrt(
            _add(
                _multiply(reference_wind, reference_wind),
                _multiply(convective_velocity, convective_velocity),
            )
        )
        new_stability_wind = jnp.where(
            stable,
            jnp.maximum(reference_wind, jnp.float64(0.1)),
            unstable_wind,
        )
        new_monin_length = fortran_divide(reference_height, bounded_zeta)
        return (
            jnp.where(active, new_stability_wind, stability_wind),
            jnp.where(active, new_monin_length, monin_length),
            jnp.where(active, new_canopy_temperature, previous_canopy_temperature),
            jnp.where(active, new_canopy_humidity, previous_canopy_humidity),
            friction_result.state,
            jnp.where(active, friction, 0.0),
            jnp.where(active, momentum_resistance, 0.0),
            jnp.where(active, thermal_resistance, 0.0),
            jnp.where(active, moisture_resistance, 0.0),
            jnp.where(active, canyon_wind, 0.0),
            jnp.where(active, canyon_resistance, 0.0),
            jnp.where(active, atmospheric_sensible_conductance, 0.0),
            jnp.where(active, atmospheric_latent_conductance, 0.0),
            jnp.where(active, sensible_sum, 0.0),
            tree_util.tree_map(
                lambda value: jnp.where(active[:, None], value, 0.0),
                conductances,
            ),
            jnp.where(active[:, None], wet_fraction, 0.0),
            jnp.where(active, latent_sum, 0.0),
            jnp.where(active, temperature_profile, 0.0),
            jnp.where(active, humidity_profile, 0.0),
            zero_landunit,
        )

    final_iteration_state = initial_iteration_state
    iteration_states = []
    for zero_based_iteration in range(NUMBER_OF_STABILITY_ITERATIONS):
        final_iteration_state = stability_iteration(
            final_iteration_state,
            jnp.int32(zero_based_iteration),
        )
        final_iteration_state = tree_util.tree_map(
            lax.optimization_barrier,
            final_iteration_state,
        )
        iteration_states.append(final_iteration_state)
    (
        stability_wind,
        monin_length,
        canopy_temperature,
        canopy_humidity,
        friction_state,
        friction,
        momentum_resistance,
        thermal_resistance,
        moisture_resistance,
        canyon_wind,
        canyon_resistance,
        atmospheric_sensible_conductance,
        atmospheric_latent_conductance,
        sensible_sum,
        conductances,
        wet_fraction,
        latent_sum,
        _temperature_profile,
        _humidity_profile,
        _dummy,
    ) = final_iteration_state

    temperature_difference_surface = _add(canopy_temperature[:, None], -ground_temperature)
    humidity_difference_surface = _add(canopy_humidity[:, None], -ground_humidity)
    scaled_sensible_flux = -_multiply(
        _multiply(
            density[:, None],
            jnp.float64(SPECIFIC_HEAT_DRY_AIR),
        ),
        _multiply(conductances.sensible_scaled_surface, temperature_difference_surface),
    )
    scaled_moisture_flux = -_multiply(
        density[:, None],
        _multiply(conductances.latent_scaled_surface, humidity_difference_surface),
    )
    surface_sensible_total = _native_surface_sum(
        jnp.zeros_like(reference_wind), scaled_sensible_flux
    )
    surface_moisture_total = _native_surface_sum(
        jnp.zeros_like(reference_wind), scaled_moisture_flux
    )
    atmospheric_sensible_flux = -_multiply(
        fortran_divide(
            _multiply(density, jnp.float64(SPECIFIC_HEAT_DRY_AIR)),
            thermal_resistance,
        ),
        _add(atmospheric_reference_temperature, -canopy_temperature),
    )
    atmospheric_moisture_flux = -_multiply(
        fortran_divide(density, moisture_resistance),
        _add(atmospheric_humidity, -canopy_humidity),
    )
    sensible_error = _add(surface_sensible_total, -atmospheric_sensible_flux)
    moisture_error = _add(surface_moisture_total, -atmospheric_moisture_flux)

    result_state = UrbanTurbulenceState(
        canopy_air_temperature_landunit=jnp.where(
            active,
            canopy_temperature,
            state.canopy_air_temperature_landunit,
        ),
        canopy_air_specific_humidity_landunit=jnp.where(
            active,
            canopy_humidity,
            state.canopy_air_specific_humidity_landunit,
        ),
        friction_velocity=friction_state,
    )
    diagnostics = UrbanTurbulenceDiagnostics(
        reference_wind_speed_landunit=jnp.where(active, reference_wind, 0.0),
        stability_wind_speed_landunit=jnp.where(active, stability_wind, 0.0),
        monin_obukhov_length_landunit=jnp.where(active, monin_length, 0.0),
        stability_wind_iteration_landunit=jnp.stack(
            [jnp.where(active, value[0], 0.0) for value in iteration_states]
        ),
        monin_obukhov_length_iteration_landunit=jnp.stack(
            [jnp.where(active, value[1], 0.0) for value in iteration_states]
        ),
        friction_velocity_iteration_landunit=jnp.stack(
            [jnp.where(active, value[5], 0.0) for value in iteration_states]
        ),
        thermal_resistance_iteration_landunit=jnp.stack(
            [jnp.where(active, value[7], 0.0) for value in iteration_states]
        ),
        canopy_air_temperature_iteration_landunit=jnp.stack(
            [jnp.where(active, value[2], 0.0) for value in iteration_states]
        ),
        canopy_air_specific_humidity_iteration_landunit=jnp.stack(
            [jnp.where(active, value[3], 0.0) for value in iteration_states]
        ),
        surface_latent_conductance_iteration=jnp.stack(
            [
                jnp.where(active[:, None], value[14].latent_scaled_surface, 0.0)
                for value in iteration_states
            ]
        ),
        surface_wet_fraction_iteration=jnp.stack(
            [jnp.where(active[:, None], value[15], 0.0) for value in iteration_states]
        ),
        canyon_top_wind_landunit=jnp.where(active, canyon_top_wind, 0.0),
        canyon_wind_landunit=canyon_wind,
        canyon_resistance_landunit=canyon_resistance,
        friction_velocity_landunit=friction,
        momentum_resistance_landunit=momentum_resistance,
        thermal_resistance_landunit=thermal_resistance,
        moisture_resistance_landunit=moisture_resistance,
        atmospheric_sensible_conductance_landunit=atmospheric_sensible_conductance,
        atmospheric_latent_conductance_landunit=atmospheric_latent_conductance,
        sensible_conductance_sum_landunit=sensible_sum,
        latent_conductance_sum_landunit=latent_sum,
        surface_conductances=conductances,
        surface_wet_fraction=wet_fraction,
        waste_heat_landunit=jnp.where(active, landunit_waste.waste_heat, 0.0),
        heat_from_air_conditioning_landunit=jnp.where(
            active, landunit_waste.heat_from_air_conditioning, 0.0
        ),
        traffic_heat_landunit=jnp.where(active, traffic_heat, 0.0),
        sensible_conservation_error_landunit=jnp.where(active, sensible_error, 0.0),
        moisture_conservation_error_landunit=jnp.where(active, moisture_error, 0.0),
    )
    return UrbanTurbulenceResult(result_state, diagnostics)


def _validate_surface_flux_shapes(
    turbulence_inputs: UrbanTurbulenceInputs,
    inputs: UrbanSurfaceFluxInputs,
) -> None:
    number_of_landunits = turbulence_inputs.active_landunit.shape[0]
    surface_shape = (number_of_landunits, NUMBER_OF_URBAN_SURFACES)
    for name in UrbanSurfaceFluxInputs._fields:
        value = getattr(inputs, name)
        if name in (
            "atmospheric_pressure_landunit",
            "internal_building_temperature_landunit",
        ):
            expected = (number_of_landunits,)
        elif name == "pervious_road_root_fraction_landunit_level":
            if value.ndim != 2 or value.shape[0] != number_of_landunits or value.shape[1] < 1:
                raise ValueError(
                    "pervious_road_root_fraction_landunit_level must have shape "
                    "(landunit, level>=1)"
                )
            continue
        else:
            expected = surface_shape
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")


def _other_surface_conductance_sums(
    atmospheric_conductance: jnp.ndarray,
    surface_conductance: jnp.ndarray,
) -> jnp.ndarray:
    source_sum_order = (
        URBAN_SURFACE_ROOF,
        URBAN_SURFACE_PERVIOUS_ROAD,
        URBAN_SURFACE_IMPERVIOUS_ROAD,
        URBAN_SURFACE_SUN_WALL,
        URBAN_SURFACE_SHADE_WALL,
    )
    other_sums = []
    for own_surface in range(NUMBER_OF_URBAN_SURFACES):
        terms = [atmospheric_conductance]
        terms.extend(
            surface_conductance[:, surface]
            for surface in source_sum_order
            if surface != own_surface
        )
        other_sums.append(_add(*terms))
    return jnp.stack(other_sums, axis=1)


def urban_surface_fluxes(
    turbulence_inputs: UrbanTurbulenceInputs,
    inputs: UrbanSurfaceFluxInputs,
    turbulence: UrbanTurbulenceResult,
    building_temperature_method: int | jnp.ndarray = BUILDING_TEMPERATURE_METHOD_PROGNOSTIC,
) -> UrbanSurfaceFluxResult:

    _validate_surface_flux_shapes(turbulence_inputs, inputs)
    inputs = UrbanSurfaceFluxInputs(*(jnp.asarray(value, dtype=jnp.float64) for value in inputs))
    active = jnp.asarray(turbulence_inputs.active_landunit, dtype=jnp.bool_)
    active_surface = active[:, None]

    def safe_landunit(value: jnp.ndarray, fallback: float) -> jnp.ndarray:
        return jnp.where(active, jnp.asarray(value, dtype=jnp.float64), jnp.float64(fallback))

    def safe_surface(value: jnp.ndarray, fallback: float) -> jnp.ndarray:
        return jnp.where(active_surface, value, jnp.float64(fallback))

    density = safe_landunit(turbulence_inputs.atmospheric_density_landunit, 1.2)
    eastward_wind = safe_landunit(turbulence_inputs.eastward_wind_landunit, 1.0)
    northward_wind = safe_landunit(turbulence_inputs.northward_wind_landunit, 0.0)
    atmospheric_temperature = safe_landunit(
        turbulence_inputs.atmospheric_temperature_landunit, 280.0
    )
    atmospheric_pressure = safe_landunit(inputs.atmospheric_pressure_landunit, 100000.0)
    ground_temperature = safe_surface(turbulence_inputs.ground_temperature_surface, 280.0)
    ground_humidity = safe_surface(turbulence_inputs.ground_specific_humidity_surface, 0.005)
    humidity_derivative = safe_surface(
        inputs.ground_specific_humidity_temperature_derivative_surface, 0.0
    )
    latent_heat = safe_surface(inputs.latent_heat_surface, 2.5e6)
    snow_fraction = safe_surface(inputs.snow_fraction_surface, 0.0)
    soil_alpha = safe_surface(inputs.soil_evaporation_alpha_surface, 1.0)
    inner_temperature = safe_surface(inputs.inner_layer_temperature_surface, 280.0)

    canopy_temperature = safe_landunit(turbulence.state.canopy_air_temperature_landunit, 280.0)
    canopy_humidity = safe_landunit(turbulence.state.canopy_air_specific_humidity_landunit, 0.005)
    diagnostics = turbulence.diagnostics
    conductances = diagnostics.surface_conductances
    sensible_scaled = conductances.sensible_scaled_surface
    latent_scaled = conductances.latent_scaled_surface
    sensible_unscaled = conductances.sensible_unscaled_surface
    latent_unscaled = conductances.latent_unscaled_surface
    safe_sensible_sum = jnp.where(
        active,
        diagnostics.sensible_conductance_sum_landunit,
        jnp.float64(1.0),
    )
    safe_latent_sum = jnp.where(
        active,
        diagnostics.latent_conductance_sum_landunit,
        jnp.float64(1.0),
    )

    other_sensible = _other_surface_conductance_sums(
        diagnostics.atmospheric_sensible_conductance_landunit,
        sensible_scaled,
    )
    other_latent = _other_surface_conductance_sums(
        diagnostics.atmospheric_latent_conductance_landunit,
        latent_scaled,
    )
    ground_sensible_derivative = _multiply(
        _multiply(
            _multiply(density[:, None], jnp.float64(SPECIFIC_HEAT_DRY_AIR)),
            other_sensible,
        ),
        fortran_divide(
            sensible_unscaled,
            safe_sensible_sum[:, None],
        ),
    )
    ground_latent_derivative = _multiply(
        _multiply(
            _multiply(density[:, None], other_latent),
            fortran_divide(
                latent_unscaled,
                safe_latent_sum[:, None],
            ),
        ),
        humidity_derivative,
    )
    ground_latent_derivative = ground_latent_derivative.at[:, URBAN_SURFACE_SUN_WALL].set(0.0)
    ground_latent_derivative = ground_latent_derivative.at[:, URBAN_SURFACE_SHADE_WALL].set(0.0)
    ground_energy_derivative = _add(
        ground_sensible_derivative,
        _multiply(ground_latent_derivative, latent_heat),
    )

    momentum_resistance = jnp.where(
        active,
        diagnostics.momentum_resistance_landunit,
        jnp.float64(1.0),
    )
    eastward_stress = -fortran_divide(_multiply(density, eastward_wind), momentum_resistance)
    northward_stress = -fortran_divide(_multiply(density, northward_wind), momentum_resistance)
    eastward_stress = jnp.broadcast_to(eastward_stress[:, None], ground_temperature.shape)
    northward_stress = jnp.broadcast_to(northward_stress[:, None], ground_temperature.shape)

    surface_temperature_difference = _add(canopy_temperature[:, None], -ground_temperature)
    sensible_ground = -_multiply(
        _multiply(
            _multiply(density[:, None], jnp.float64(SPECIFIC_HEAT_DRY_AIR)),
            sensible_unscaled,
        ),
        surface_temperature_difference,
    )
    surface_humidity_difference = _add(canopy_humidity[:, None], -ground_humidity)
    evaporation = -_multiply(
        _multiply(density[:, None], latent_unscaled),
        surface_humidity_difference,
    )
    zero_surface = jnp.zeros_like(evaporation)
    soil_evaporation = zero_surface
    soil_evaporation = soil_evaporation.at[:, URBAN_SURFACE_ROOF].set(
        evaporation[:, URBAN_SURFACE_ROOF]
    )
    soil_evaporation = soil_evaporation.at[:, URBAN_SURFACE_IMPERVIOUS_ROAD].set(
        evaporation[:, URBAN_SURFACE_IMPERVIOUS_ROAD]
    )
    pervious_soil_assignment = (
        (surface_humidity_difference[:, URBAN_SURFACE_PERVIOUS_ROAD] > jnp.float64(0.0))
        | (snow_fraction[:, URBAN_SURFACE_PERVIOUS_ROAD] > jnp.float64(0.0))
        | (soil_alpha[:, URBAN_SURFACE_PERVIOUS_ROAD] <= jnp.float64(0.0))
    )
    pervious_evaporation = evaporation[:, URBAN_SURFACE_PERVIOUS_ROAD]
    soil_evaporation = soil_evaporation.at[:, URBAN_SURFACE_PERVIOUS_ROAD].set(
        jnp.where(pervious_soil_assignment, pervious_evaporation, 0.0)
    )
    transpiration = zero_surface.at[:, URBAN_SURFACE_PERVIOUS_ROAD].set(
        jnp.where(pervious_soil_assignment, 0.0, pervious_evaporation)
    )
    vegetation_evaporation = transpiration

    scaled_sensible_heat = -_multiply(
        _multiply(
            _multiply(density[:, None], jnp.float64(SPECIFIC_HEAT_DRY_AIR)),
            sensible_scaled,
        ),
        surface_temperature_difference,
    )
    scaled_moisture_flux = -_multiply(
        _multiply(density[:, None], latent_scaled),
        surface_humidity_difference,
    )

    simple_building_temperature = simple_internal_building_temperature(
        safe_landunit(turbulence_inputs.roof_height_landunit, 10.0),
        safe_landunit(turbulence_inputs.canyon_height_width_ratio_landunit, 1.0),
        safe_landunit(turbulence_inputs.roof_fraction_landunit, 0.3),
        inner_temperature[:, URBAN_SURFACE_ROOF],
        inner_temperature[:, URBAN_SURFACE_SUN_WALL],
        inner_temperature[:, URBAN_SURFACE_SHADE_WALL],
    )
    simple_building = jnp.asarray(building_temperature_method, dtype=jnp.int32) == jnp.int32(
        BUILDING_TEMPERATURE_METHOD_SIMPLE
    )
    internal_building_temperature = jnp.where(
        simple_building,
        simple_building_temperature,
        safe_landunit(inputs.internal_building_temperature_landunit, 280.0),
    )

    root_levels = inputs.pervious_road_root_fraction_landunit_level.shape[1]
    effective_root_fraction = jnp.zeros(
        (
            inputs.pervious_road_root_fraction_landunit_level.shape[0],
            NUMBER_OF_URBAN_SURFACES,
            root_levels,
        ),
        dtype=jnp.float64,
    )
    effective_root_fraction = effective_root_fraction.at[:, URBAN_SURFACE_PERVIOUS_ROAD, :].set(
        inputs.pervious_road_root_fraction_landunit_level
    )

    reference_temperature = jnp.broadcast_to(canopy_temperature[:, None], ground_temperature.shape)
    reference_humidity = jnp.broadcast_to(canopy_humidity[:, None], ground_temperature.shape)
    saturation = qsat(reference_temperature, atmospheric_pressure[:, None])
    relative_humidity = jnp.minimum(
        jnp.float64(100.0),
        _multiply(
            fortran_divide(reference_humidity, saturation.specific_humidity),
            jnp.float64(100.0),
        ),
    )
    wind_10m = jnp.broadcast_to(
        turbulence.state.friction_velocity.wind_10m_patch[:, None],
        ground_temperature.shape,
    )
    human_stress = fast_human_stress_indices(
        reference_temperature,
        relative_humidity,
        saturation.vapor_pressure,
        wind_10m,
    )

    def mask_surface(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(active_surface, value, 0.0)

    zero = jnp.zeros_like(ground_temperature)
    energy = UrbanSurfaceEnergyFluxes(
        aerodynamic_resistance_surface=mask_surface(
            jnp.broadcast_to(momentum_resistance[:, None], ground_temperature.shape)
        ),
        downward_longwave_below_canopy_surface=zero,
        upward_longwave_above_canopy_surface=zero,
        ground_sensible_derivative_surface=mask_surface(ground_sensible_derivative),
        ground_latent_derivative_surface=mask_surface(ground_latent_derivative),
        ground_energy_derivative_surface=mask_surface(ground_energy_derivative),
        eastward_wind_stress_surface=mask_surface(eastward_stress),
        northward_wind_stress_surface=mask_surface(northward_stress),
        sensible_heat_ground_surface=mask_surface(sensible_ground),
        sensible_heat_total_surface=mask_surface(sensible_ground),
        urban_sensible_heat_total_surface=mask_surface(sensible_ground),
        sensible_heat_snow_surface=zero,
        sensible_heat_soil_surface=zero,
        sensible_heat_surface_water_surface=zero,
    )
    water = UrbanSurfaceWaterFluxes(
        soil_evaporation_surface=mask_surface(soil_evaporation),
        vegetation_transpiration_surface=mask_surface(transpiration),
        vegetation_evaporation_surface=mask_surface(vegetation_evaporation),
    )
    atmosphere = UrbanSurfaceAtmosphericState(
        reference_temperature_surface=mask_surface(reference_temperature),
        urban_reference_temperature_surface=mask_surface(reference_temperature),
        vegetation_temperature_surface=mask_surface(
            jnp.broadcast_to(atmospheric_temperature[:, None], ground_temperature.shape)
        ),
        reference_specific_humidity_surface=mask_surface(reference_humidity),
        reference_relative_humidity_surface=mask_surface(relative_humidity),
        urban_reference_relative_humidity_surface=mask_surface(relative_humidity),
    )
    human_stress = tree_util.tree_map(mask_surface, human_stress)
    return UrbanSurfaceFluxResult(
        energy=energy,
        water=water,
        atmosphere=atmosphere,
        human_stress=human_stress,
        urban_human_stress=human_stress,
        internal_building_temperature_landunit=jnp.where(
            active, internal_building_temperature, 0.0
        ),
        effective_root_fraction_surface_level=jnp.where(
            active[:, None, None], effective_root_fraction, 0.0
        ),
        scaled_sensible_heat_surface=mask_surface(scaled_sensible_heat),
        scaled_moisture_flux_surface=mask_surface(scaled_moisture_flux),
    )
