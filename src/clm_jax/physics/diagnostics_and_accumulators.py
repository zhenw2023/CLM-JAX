
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.constants import CLM_SPECIAL_VALUE, FREEZING_TEMPERATURE_FRESHWATER
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    SubgridTopology,
)
from clm_jax.physics.accumulators import (
    ACCUMULATOR_RESET_VALUE,
    AccumulatorState,
    update_running_accumulation,
    update_running_mean,
    update_time_average,
)
from clm_jax.physics.numerics import fortran_add, fortran_divide, fortran_multiply

SECONDS_PER_DAY = 86400.0


class AtmosphereAccumulatorState(NamedTuple):
    direct_solar_24_hour: AccumulatorState
    direct_solar_240_hour: AccumulatorState
    diffuse_solar_24_hour: AccumulatorState
    diffuse_solar_240_hour: AccumulatorState
    precipitation_10_day: AccumulatorState
    precipitation_60_day: AccumulatorState
    relative_humidity_30_day: AccumulatorState
    co2_partial_pressure_240_hour: AccumulatorState
    oxygen_partial_pressure_240_hour: AccumulatorState
    air_pressure_240_hour: AccumulatorState


class TemperatureAccumulatorState(NamedTuple):
    vegetation_temperature_24_hour: AccumulatorState
    vegetation_temperature_240_hour: AccumulatorState
    reference_temperature_hourly: AccumulatorState
    urban_reference_temperature_hourly: AccumulatorState
    rural_reference_temperature_hourly: AccumulatorState
    reference_temperature_10_day: AccumulatorState
    minimum_reference_temperature_10_day: AccumulatorState
    minimum_reference_temperature_5_day: AccumulatorState
    growing_degree_days_base_0: AccumulatorState
    growing_degree_days_base_8: AccumulatorState
    growing_degree_days_base_10: AccumulatorState


class CanopyAccumulatorState(NamedTuple):
    sunlit_fraction_24_hour: AccumulatorState
    sunlit_fraction_240_hour: AccumulatorState
    exposed_leaf_area_index_240_hour: AccumulatorState


class WaterEnergyAccumulatorState(NamedTuple):
    annual_evapotranspiration: AccumulatorState
    hourly_transpiration_stress: AccumulatorState


class CropAccumulatorState(NamedTuple):
    growing_degree_days_plant: AccumulatorState
    growing_degree_days_soil: AccumulatorState


class DailyExtremaState(NamedTuple):
    reference_temperature_maximum_patch: jnp.ndarray
    reference_temperature_minimum_patch: jnp.ndarray
    reference_temperature_maximum_instantaneous_patch: jnp.ndarray
    reference_temperature_minimum_instantaneous_patch: jnp.ndarray
    urban_reference_temperature_maximum_patch: jnp.ndarray
    urban_reference_temperature_minimum_patch: jnp.ndarray
    urban_reference_temperature_maximum_instantaneous_patch: jnp.ndarray
    urban_reference_temperature_minimum_instantaneous_patch: jnp.ndarray
    rural_reference_temperature_maximum_patch: jnp.ndarray
    rural_reference_temperature_minimum_patch: jnp.ndarray
    rural_reference_temperature_maximum_instantaneous_patch: jnp.ndarray
    rural_reference_temperature_minimum_instantaneous_patch: jnp.ndarray
    transpiration_stress_minimum_patch: jnp.ndarray
    transpiration_stress_minimum_instantaneous_patch: jnp.ndarray


class ConfiguredAccumulatorState(NamedTuple):
    atmosphere: AtmosphereAccumulatorState
    temperature: TemperatureAccumulatorState
    canopy: CanopyAccumulatorState
    water_energy: WaterEnergyAccumulatorState
    crop: CropAccumulatorState
    daily_extrema: DailyExtremaState


class ConfiguredAccumulatorInputs(NamedTuple):
    direct_solar_gridcell_band: jnp.ndarray
    diffuse_solar_gridcell_band: jnp.ndarray
    rain_downscaled_column: jnp.ndarray
    snow_downscaled_column: jnp.ndarray
    relative_humidity_gridcell: jnp.ndarray
    co2_partial_pressure_gridcell: jnp.ndarray
    oxygen_partial_pressure_gridcell: jnp.ndarray
    air_pressure_downscaled_column: jnp.ndarray
    vegetation_temperature_patch: jnp.ndarray
    reference_temperature_patch: jnp.ndarray
    urban_reference_temperature_patch: jnp.ndarray
    rural_reference_temperature_patch: jnp.ndarray
    sunlit_fraction_patch: jnp.ndarray
    exposed_leaf_area_index_patch: jnp.ndarray
    total_evapotranspiration_flux_column: jnp.ndarray
    transpiration_stress_patch: jnp.ndarray
    crop_live_patch: jnp.ndarray
    crop_growing_degree_days_plant_patch: jnp.ndarray
    crop_vernalization_factor_patch: jnp.ndarray
    crop_latitude_base_temperature_patch: jnp.ndarray
    soil_temperature_column_level: jnp.ndarray
    soil_layer_thickness_column_level: jnp.ndarray


class AccumulatorClock(NamedTuple):
    nstep: int | jnp.ndarray
    month: int | jnp.ndarray
    day: int | jnp.ndarray
    seconds_into_day: int | jnp.ndarray
    end_current_day: bool | jnp.ndarray


class ConfiguredAccumulatorParameters(NamedTuple):
    step_seconds: int | jnp.ndarray
    crop_base_temperature_by_pft: jnp.ndarray
    crop_maximum_temperature_by_pft: jnp.ndarray
    spring_wheat_pft: int | jnp.ndarray
    irrigated_spring_wheat_pft: int | jnp.ndarray
    winter_wheat_pft: int | jnp.ndarray
    irrigated_winter_wheat_pft: int | jnp.ndarray
    sugarcane_pft: int | jnp.ndarray
    irrigated_sugarcane_pft: int | jnp.ndarray
    latitude_varying_base_temperature: bool | jnp.ndarray


class ConfiguredAccumulatorResult(NamedTuple):
    state: ConfiguredAccumulatorState


def _runmean(state: AccumulatorState, sample: jnp.ndarray, active: jnp.ndarray) -> AccumulatorState:
    return update_running_mean(state, sample, active).state


def _runaccum(
    state: AccumulatorState, sample: jnp.ndarray, active: jnp.ndarray
) -> AccumulatorState:
    return update_running_accumulation(state, sample, active).state


def _timeavg(
    state: AccumulatorState,
    sample: jnp.ndarray,
    active: jnp.ndarray,
    nstep: jnp.ndarray,
) -> tuple[AccumulatorState, jnp.ndarray]:
    result = update_time_average(state, sample, active, nstep)
    return result.state, result.extracted


def _is_urban(landunit_type: jnp.ndarray) -> jnp.ndarray:
    return (landunit_type >= jnp.int32(LANDUNIT_URBAN_MIN)) & (
        landunit_type <= jnp.int32(LANDUNIT_URBAN_MAX)
    )


def _update_daily_temperature_extrema(
    daily: DailyExtremaState,
    hourly_reference: jnp.ndarray,
    hourly_urban: jnp.ndarray,
    hourly_rural: jnp.ndarray,
    urban_patch: jnp.ndarray,
    rural_patch: jnp.ndarray,
    clock: AccumulatorClock,
    step_seconds: jnp.ndarray,
) -> DailyExtremaState:
    special = jnp.float64(CLM_SPECIAL_VALUE)
    end_day = jnp.asarray(clock.end_current_day, dtype=jnp.bool_)
    first_step = jnp.asarray(clock.seconds_into_day, dtype=jnp.int32) == step_seconds

    def extrema(
        value: jnp.ndarray,
        maximum: jnp.ndarray,
        minimum: jnp.ndarray,
        maximum_instantaneous: jnp.ndarray,
        minimum_instantaneous: jnp.ndarray,
        end_mask: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        valid = value != special
        maximum_instantaneous = jnp.where(
            valid, jnp.maximum(value, maximum_instantaneous), maximum_instantaneous
        )
        minimum_instantaneous = jnp.where(
            valid, jnp.minimum(value, minimum_instantaneous), minimum_instantaneous
        )
        write = end_day & end_mask
        maximum = jnp.where(write, maximum_instantaneous, maximum)
        minimum = jnp.where(write, minimum_instantaneous, minimum)
        maximum_instantaneous = jnp.where(write, -special, maximum_instantaneous)
        minimum_instantaneous = jnp.where(write, special, minimum_instantaneous)
        clear = (~end_day) & first_step
        maximum = jnp.where(clear, special, maximum)
        minimum = jnp.where(clear, special, minimum)
        return maximum, minimum, maximum_instantaneous, minimum_instantaneous

    all_patch = jnp.ones_like(urban_patch, dtype=jnp.bool_)
    regular = extrema(
        hourly_reference,
        daily.reference_temperature_maximum_patch,
        daily.reference_temperature_minimum_patch,
        daily.reference_temperature_maximum_instantaneous_patch,
        daily.reference_temperature_minimum_instantaneous_patch,
        all_patch,
    )
    urban = extrema(
        hourly_urban,
        daily.urban_reference_temperature_maximum_patch,
        daily.urban_reference_temperature_minimum_patch,
        daily.urban_reference_temperature_maximum_instantaneous_patch,
        daily.urban_reference_temperature_minimum_instantaneous_patch,
        urban_patch,
    )
    rural = extrema(
        hourly_rural,
        daily.rural_reference_temperature_maximum_patch,
        daily.rural_reference_temperature_minimum_patch,
        daily.rural_reference_temperature_maximum_instantaneous_patch,
        daily.rural_reference_temperature_minimum_instantaneous_patch,
        rural_patch,
    )
    return daily._replace(
        reference_temperature_maximum_patch=regular[0],
        reference_temperature_minimum_patch=regular[1],
        reference_temperature_maximum_instantaneous_patch=regular[2],
        reference_temperature_minimum_instantaneous_patch=regular[3],
        urban_reference_temperature_maximum_patch=urban[0],
        urban_reference_temperature_minimum_patch=urban[1],
        urban_reference_temperature_maximum_instantaneous_patch=urban[2],
        urban_reference_temperature_minimum_instantaneous_patch=urban[3],
        rural_reference_temperature_maximum_patch=rural[0],
        rural_reference_temperature_minimum_patch=rural[1],
        rural_reference_temperature_maximum_instantaneous_patch=rural[2],
        rural_reference_temperature_minimum_instantaneous_patch=rural[3],
    )


def _growing_degree_sample(
    reference_temperature: jnp.ndarray,
    latitude: jnp.ndarray,
    clock: AccumulatorClock,
    step_seconds: jnp.ndarray,
    base_temperature_celsius: float,
    maximum_increment: float,
) -> jnp.ndarray:
    month = jnp.asarray(clock.month, dtype=jnp.int32)
    day = jnp.asarray(clock.day, dtype=jnp.int32)
    seconds = jnp.asarray(clock.seconds_into_day, dtype=jnp.int32)
    reset = (month == 1) & (day == 1) & (seconds == step_seconds)
    northern_season = (month > 3) & (month < 10) & (latitude >= jnp.float64(0.0))
    southern_season = ((month > 9) | (month < 4)) & (latitude < jnp.float64(0.0))
    temperature_above_base = fortran_add(
        reference_temperature,
        -jnp.float64(FREEZING_TEMPERATURE_FRESHWATER + base_temperature_celsius),
    )
    bounded = jnp.maximum(
        jnp.float64(0.0),
        jnp.minimum(jnp.float64(maximum_increment), temperature_above_base),
    )
    degree_days = fortran_divide(
        fortran_multiply(bounded, step_seconds.astype(jnp.float64)),
        jnp.float64(SECONDS_PER_DAY),
    )
    seasonal = jnp.where(northern_season | southern_season, degree_days, jnp.float64(0.0))
    return jnp.where(reset, jnp.float64(ACCUMULATOR_RESET_VALUE), seasonal)


def update_configured_accumulators(
    topology: SubgridTopology,
    inputs: ConfiguredAccumulatorInputs,
    state: ConfiguredAccumulatorState,
    clock: AccumulatorClock,
    parameters: ConfiguredAccumulatorParameters,
) -> ConfiguredAccumulatorResult:

    patch_active = jnp.asarray(topology.patch.active, dtype=jnp.bool_)
    column_active = jnp.asarray(topology.column.active, dtype=jnp.bool_)
    patch_gridcell = jnp.asarray(topology.patch.gridcell, dtype=jnp.int32)
    patch_column = jnp.asarray(topology.patch.column, dtype=jnp.int32)
    patch_landunit_type = jnp.asarray(topology.patch.landunit_type, dtype=jnp.int32)
    patch_pft = jnp.asarray(topology.patch.vegetation_type, dtype=jnp.int32)
    latitude_patch = jnp.asarray(topology.grid.latitude_degrees, dtype=jnp.float64)[patch_gridcell]
    urban_patch = _is_urban(patch_landunit_type)
    rural_patch = (patch_landunit_type == jnp.int32(LANDUNIT_SOIL)) | (
        patch_landunit_type == jnp.int32(LANDUNIT_CROP)
    )
    nstep = jnp.asarray(clock.nstep, dtype=jnp.int32)
    step_seconds = jnp.asarray(parameters.step_seconds, dtype=jnp.int32)

    atmosphere = state.atmosphere
    direct = jnp.asarray(inputs.direct_solar_gridcell_band, dtype=jnp.float64)[patch_gridcell, 0]
    diffuse = jnp.asarray(inputs.diffuse_solar_gridcell_band, dtype=jnp.float64)[patch_gridcell, 0]
    precipitation_column = fortran_add(inputs.rain_downscaled_column, inputs.snow_downscaled_column)
    precipitation_patch = precipitation_column[patch_column]
    relative_humidity_patch = jnp.asarray(inputs.relative_humidity_gridcell)[patch_gridcell]
    co2_patch = jnp.asarray(inputs.co2_partial_pressure_gridcell)[patch_gridcell]
    oxygen_patch = jnp.asarray(inputs.oxygen_partial_pressure_gridcell)[patch_gridcell]
    air_pressure_patch = jnp.asarray(inputs.air_pressure_downscaled_column)[patch_column]
    atmosphere = AtmosphereAccumulatorState(
        direct_solar_24_hour=_runmean(atmosphere.direct_solar_24_hour, direct, patch_active),
        direct_solar_240_hour=_runmean(atmosphere.direct_solar_240_hour, direct, patch_active),
        diffuse_solar_24_hour=_runmean(atmosphere.diffuse_solar_24_hour, diffuse, patch_active),
        diffuse_solar_240_hour=_runmean(atmosphere.diffuse_solar_240_hour, diffuse, patch_active),
        precipitation_10_day=_runmean(
            atmosphere.precipitation_10_day, precipitation_patch, patch_active
        ),
        precipitation_60_day=_runmean(
            atmosphere.precipitation_60_day, precipitation_patch, patch_active
        ),
        relative_humidity_30_day=_runmean(
            atmosphere.relative_humidity_30_day, relative_humidity_patch, patch_active
        ),
        co2_partial_pressure_240_hour=_runmean(
            atmosphere.co2_partial_pressure_240_hour, co2_patch, patch_active
        ),
        oxygen_partial_pressure_240_hour=_runmean(
            atmosphere.oxygen_partial_pressure_240_hour, oxygen_patch, patch_active
        ),
        air_pressure_240_hour=_runmean(
            atmosphere.air_pressure_240_hour, air_pressure_patch, patch_active
        ),
    )

    temperature = state.temperature
    vegetation_temperature = jnp.asarray(inputs.vegetation_temperature_patch)
    reference_temperature = jnp.asarray(inputs.reference_temperature_patch)
    urban_reference_temperature = jnp.asarray(inputs.urban_reference_temperature_patch)
    rural_reference_temperature = jnp.asarray(inputs.rural_reference_temperature_patch)
    vegetation_24 = _runmean(
        temperature.vegetation_temperature_24_hour, vegetation_temperature, patch_active
    )
    vegetation_240 = _runmean(
        temperature.vegetation_temperature_240_hour, vegetation_temperature, patch_active
    )
    reference_hourly, hourly_reference = _timeavg(
        temperature.reference_temperature_hourly, reference_temperature, patch_active, nstep
    )
    urban_hourly, hourly_urban = _timeavg(
        temperature.urban_reference_temperature_hourly,
        urban_reference_temperature,
        patch_active,
        nstep,
    )
    rural_hourly, hourly_rural = _timeavg(
        temperature.rural_reference_temperature_hourly,
        rural_reference_temperature,
        patch_active,
        nstep,
    )
    daily = _update_daily_temperature_extrema(
        state.daily_extrema,
        hourly_reference,
        hourly_urban,
        hourly_rural,
        urban_patch,
        rural_patch,
        clock,
        step_seconds,
    )
    reference_10 = _runmean(
        temperature.reference_temperature_10_day, reference_temperature, patch_active
    )
    daily_minimum = jnp.minimum(
        daily.reference_temperature_minimum_patch,
        daily.reference_temperature_minimum_instantaneous_patch,
    )
    daily_minimum = jnp.where(
        daily_minimum > jnp.float64(1.0e30),
        jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
        daily_minimum,
    )
    minimum_10 = _runmean(
        temperature.minimum_reference_temperature_10_day, daily_minimum, patch_active
    )
    minimum_5 = _runmean(
        temperature.minimum_reference_temperature_5_day, daily_minimum, patch_active
    )
    gdd0_sample = _growing_degree_sample(
        reference_temperature, latitude_patch, clock, step_seconds, 0.0, 26.0
    )
    gdd8_sample = _growing_degree_sample(
        reference_temperature, latitude_patch, clock, step_seconds, 8.0, 30.0
    )
    gdd10_sample = _growing_degree_sample(
        reference_temperature, latitude_patch, clock, step_seconds, 10.0, 30.0
    )
    temperature = TemperatureAccumulatorState(
        vegetation_temperature_24_hour=vegetation_24,
        vegetation_temperature_240_hour=vegetation_240,
        reference_temperature_hourly=reference_hourly,
        urban_reference_temperature_hourly=urban_hourly,
        rural_reference_temperature_hourly=rural_hourly,
        reference_temperature_10_day=reference_10,
        minimum_reference_temperature_10_day=minimum_10,
        minimum_reference_temperature_5_day=minimum_5,
        growing_degree_days_base_0=_runaccum(
            temperature.growing_degree_days_base_0, gdd0_sample, patch_active
        ),
        growing_degree_days_base_8=_runaccum(
            temperature.growing_degree_days_base_8, gdd8_sample, patch_active
        ),
        growing_degree_days_base_10=_runaccum(
            temperature.growing_degree_days_base_10, gdd10_sample, patch_active
        ),
    )

    canopy = state.canopy
    sunlit_fraction = jnp.asarray(inputs.sunlit_fraction_patch)
    exposed_lai = jnp.asarray(inputs.exposed_leaf_area_index_patch)
    canopy = CanopyAccumulatorState(
        sunlit_fraction_24_hour=_runmean(
            canopy.sunlit_fraction_24_hour, sunlit_fraction, patch_active
        ),
        sunlit_fraction_240_hour=_runmean(
            canopy.sunlit_fraction_240_hour, sunlit_fraction, patch_active
        ),
        exposed_leaf_area_index_240_hour=_runmean(
            canopy.exposed_leaf_area_index_240_hour, exposed_lai, patch_active
        ),
    )

    water_energy = state.water_energy
    annual_et = _runmean(
        water_energy.annual_evapotranspiration,
        jnp.asarray(inputs.total_evapotranspiration_flux_column),
        column_active,
    )
    btran_accumulator, hourly_btran = _timeavg(
        water_energy.hourly_transpiration_stress,
        jnp.asarray(inputs.transpiration_stress_patch),
        patch_active,
        nstep,
    )
    valid_btran = hourly_btran != jnp.float64(CLM_SPECIAL_VALUE)
    btran_minimum_instantaneous = jnp.where(
        valid_btran,
        jnp.minimum(hourly_btran, daily.transpiration_stress_minimum_instantaneous_patch),
        daily.transpiration_stress_minimum_instantaneous_patch,
    )
    end_day = jnp.asarray(clock.end_current_day, dtype=jnp.bool_)
    first_step = jnp.asarray(clock.seconds_into_day, dtype=jnp.int32) == step_seconds
    btran_minimum = jnp.where(
        end_day, btran_minimum_instantaneous, daily.transpiration_stress_minimum_patch
    )
    btran_minimum_instantaneous = jnp.where(
        end_day, jnp.float64(CLM_SPECIAL_VALUE), btran_minimum_instantaneous
    )
    btran_minimum = jnp.where(
        (~end_day) & first_step, jnp.float64(CLM_SPECIAL_VALUE), btran_minimum
    )
    daily = daily._replace(
        transpiration_stress_minimum_patch=btran_minimum,
        transpiration_stress_minimum_instantaneous_patch=btran_minimum_instantaneous,
    )
    water_energy = WaterEnergyAccumulatorState(
        annual_evapotranspiration=annual_et,
        hourly_transpiration_stress=btran_accumulator,
    )

    crop = state.crop
    external_crop_gdd = jnp.asarray(inputs.crop_growing_degree_days_plant_patch)
    synchronization_increment = jnp.maximum(
        jnp.float64(0.0),
        fortran_add(external_crop_gdd, -crop.growing_degree_days_plant.value),
    )
    synchronized_gddplant = _runaccum(
        crop.growing_degree_days_plant, synchronization_increment, patch_active
    )
    pft_base = jnp.asarray(parameters.crop_base_temperature_by_pft)[patch_pft]
    pft_maximum = jnp.asarray(parameters.crop_maximum_temperature_by_pft)[patch_pft]
    latitude_varying_pft = (
        (patch_pft == jnp.asarray(parameters.spring_wheat_pft, dtype=jnp.int32))
        | (patch_pft == jnp.asarray(parameters.irrigated_spring_wheat_pft, dtype=jnp.int32))
        | (patch_pft == jnp.asarray(parameters.sugarcane_pft, dtype=jnp.int32))
        | (patch_pft == jnp.asarray(parameters.irrigated_sugarcane_pft, dtype=jnp.int32))
    )
    use_latitude_base = (
        jnp.asarray(parameters.latitude_varying_base_temperature, dtype=jnp.bool_)
        & latitude_varying_pft
    )
    base_temperature = jnp.where(
        use_latitude_base,
        jnp.asarray(inputs.crop_latitude_base_temperature_patch),
        pft_base,
    )
    crop_temperature_above_base = fortran_add(
        reference_temperature,
        -(jnp.float64(FREEZING_TEMPERATURE_FRESHWATER) + base_temperature),
    )
    crop_increment = jnp.maximum(
        jnp.float64(0.0), jnp.minimum(pft_maximum, crop_temperature_above_base)
    )
    crop_increment = fortran_divide(
        fortran_multiply(crop_increment, step_seconds.astype(jnp.float64)),
        jnp.float64(SECONDS_PER_DAY),
    )
    winter_wheat = (patch_pft == jnp.asarray(parameters.winter_wheat_pft, dtype=jnp.int32)) | (
        patch_pft == jnp.asarray(parameters.irrigated_winter_wheat_pft, dtype=jnp.int32)
    )
    crop_increment = jnp.where(
        winter_wheat,
        fortran_multiply(crop_increment, inputs.crop_vernalization_factor_patch),
        crop_increment,
    )
    crop_increment = jnp.where(
        jnp.asarray(inputs.crop_live_patch, dtype=jnp.bool_),
        crop_increment,
        jnp.float64(ACCUMULATOR_RESET_VALUE),
    )
    gddplant = _runaccum(synchronized_gddplant, crop_increment, patch_active)

    soil_temperature = jnp.asarray(inputs.soil_temperature_column_level)
    layer_thickness = jnp.asarray(inputs.soil_layer_thickness_column_level)
    first_temperature = soil_temperature[patch_column, 0]
    second_temperature = soil_temperature[patch_column, 1]
    first_thickness = layer_thickness[patch_column, 0]
    second_thickness = layer_thickness[patch_column, 1]
    weighted_temperature = fortran_divide(
        fortran_add(
            fortran_multiply(first_temperature, first_thickness),
            fortran_multiply(second_temperature, second_thickness),
        ),
        fortran_add(first_thickness, second_thickness),
    )
    soil_temperature_above_base = fortran_add(
        weighted_temperature,
        -(jnp.float64(FREEZING_TEMPERATURE_FRESHWATER) + pft_base),
    )
    soil_increment = jnp.maximum(
        jnp.float64(0.0), jnp.minimum(pft_maximum, soil_temperature_above_base)
    )
    soil_increment = fortran_divide(
        fortran_multiply(soil_increment, step_seconds.astype(jnp.float64)),
        jnp.float64(SECONDS_PER_DAY),
    )
    soil_increment = jnp.where(
        patch_pft == jnp.asarray(parameters.winter_wheat_pft, dtype=jnp.int32),
        fortran_multiply(soil_increment, inputs.crop_vernalization_factor_patch),
        soil_increment,
    )
    soil_increment = jnp.where(
        jnp.asarray(inputs.crop_live_patch, dtype=jnp.bool_),
        soil_increment,
        jnp.float64(ACCUMULATOR_RESET_VALUE),
    )
    gddsoil = _runaccum(crop.growing_degree_days_soil, soil_increment, patch_active)
    crop = CropAccumulatorState(gddplant, gddsoil)

    return ConfiguredAccumulatorResult(
        ConfiguredAccumulatorState(atmosphere, temperature, canopy, water_energy, crop, daily)
    )


def global_mean_radiative_temperature(radiative_temperature_gridcell: jnp.ndarray) -> jnp.ndarray:

    values = jnp.asarray(radiative_temperature_gridcell, dtype=jnp.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("radiative_temperature_gridcell must be a nonempty vector")
    return fortran_divide(jnp.sum(values), jnp.float64(values.size))
