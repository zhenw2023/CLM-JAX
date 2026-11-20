
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from clm_jax.constants import (
    DENSITY_FRESH_WATER,
    FREEZING_TEMPERATURE_FRESHWATER,
    GAS_CONSTANT_DRY_AIR,
    GRAVITY,
    LATENT_HEAT_FUSION,
    SPECIFIC_HEAT_DRY_AIR,
    WATER_VAPOR_TO_DRY_AIR_MOLECULAR_WEIGHT_RATIO,
)
from clm_jax.forcing.datm import DatmForcing

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase
    from clm_jax.io.restart import ColumnTopology


class Atm2LndParameters(NamedTuple):

    repartition_rain_snow: bool
    glcmec_downscale_longwave: bool
    lapse_rate: float
    lapse_rate_longwave: float
    longwave_downscaling_limit: float
    precip_repartition_glc_all_snow_t: float
    precip_repartition_glc_all_rain_t: float
    precip_repartition_nonglc_all_snow_t: float
    precip_repartition_nonglc_all_rain_t: float

    @classmethod
    def from_case(cls, case: CimeCase) -> Atm2LndParameters:

        from clm_jax.io.namelist import read_case_namelist_group  

        values = read_case_namelist_group(case, "lnd_in", "atm2lnd_inparm")
        parameters = cls(
            repartition_rain_snow=bool(values["repartition_rain_snow"]),
            glcmec_downscale_longwave=bool(values["glcmec_downscale_longwave"]),
            lapse_rate=float(values["lapse_rate"]),
            lapse_rate_longwave=float(values["lapse_rate_longwave"]),
            longwave_downscaling_limit=float(values["longwave_downscaling_limit"]),
            precip_repartition_glc_all_snow_t=float(values["precip_repartition_glc_all_snow_t"]),
            precip_repartition_glc_all_rain_t=float(values["precip_repartition_glc_all_rain_t"]),
            precip_repartition_nonglc_all_snow_t=float(
                values["precip_repartition_nonglc_all_snow_t"]
            ),
            precip_repartition_nonglc_all_rain_t=float(
                values["precip_repartition_nonglc_all_rain_t"]
            ),
        )
        parameters.validate()
        return parameters

    def validate(self) -> None:
        if not 0.0 <= self.longwave_downscaling_limit <= 1.0:
            raise ValueError("longwave_downscaling_limit must be between 0 and 1")
        if self.precip_repartition_glc_all_rain_t <= self.precip_repartition_glc_all_snow_t:
            raise ValueError("glacier all-snow temperature must be below all-rain temperature")
        if self.precip_repartition_nonglc_all_rain_t <= self.precip_repartition_nonglc_all_snow_t:
            raise ValueError("non-glacier all-snow temperature must be below all-rain temperature")


class ThermodynamicForcing(NamedTuple):
    temperature: jnp.ndarray
    potential_temperature: jnp.ndarray
    specific_humidity: jnp.ndarray
    pressure: jnp.ndarray
    density: jnp.ndarray


class PrecipitationPartition(NamedTuple):
    rain: jnp.ndarray
    snow: jnp.ndarray
    rain_to_snow_runoff: jnp.ndarray
    sensible_heat_flux: jnp.ndarray


class ColumnAtmosphericForcing(NamedTuple):

    temperature: jnp.ndarray
    potential_temperature: jnp.ndarray
    specific_humidity: jnp.ndarray
    pressure: jnp.ndarray
    density: jnp.ndarray
    longwave_down: jnp.ndarray
    rain: jnp.ndarray
    snow: jnp.ndarray
    rain_to_snow_runoff: jnp.ndarray
    sensible_heat_flux: jnp.ndarray


def compress_datm_forcing_to_land_grid(
    forcing: DatmForcing,
    latitude_index: jnp.ndarray,
    longitude_index: jnp.ndarray,
    *,
    longitude_count: int,
) -> DatmForcing:

    flat_index = jnp.asarray(latitude_index, dtype=jnp.int32) * jnp.int32(
        longitude_count
    ) + jnp.asarray(longitude_index, dtype=jnp.int32)
    return jax.tree.map(lambda field: jnp.asarray(field).reshape(-1)[flat_index], forcing)


def air_density(
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    temperature: jnp.ndarray,
) -> jnp.ndarray:

    from clm_jax.physics.numerics import (  
        fortran_add,
        fortran_divide,
        fortran_multiply,
    )

    qbot = jnp.asarray(specific_humidity, dtype=jnp.float64)
    pbot = jnp.asarray(pressure, dtype=jnp.float64)
    tbot = jnp.asarray(temperature, dtype=jnp.float64)
    weight_ratio = jnp.float64(WATER_VAPOR_TO_DRY_AIR_MOLECULAR_WEIGHT_RATIO)
    dry_fraction = jnp.float64(1.0) - weight_ratio
    vapor_pressure = fortran_divide(
        fortran_multiply(qbot, pbot),
        fortran_add(weight_ratio, fortran_multiply(dry_fraction, qbot)),
    )
    return fortran_divide(
        fortran_add(pbot, -fortran_multiply(dry_fraction, vapor_pressure)),
        fortran_multiply(jnp.float64(GAS_CONSTANT_DRY_AIR), tbot),
    )


def downscale_thermodynamics(
    temperature: jnp.ndarray,
    potential_temperature: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    pressure: jnp.ndarray,
    density: jnp.ndarray,
    reference_height: jnp.ndarray,
    elevation_difference: jnp.ndarray,
    *,
    lapse_rate: jnp.ndarray | float,
) -> ThermodynamicForcing:

    from clm_jax.physics.qsat import qsat  
    from clm_jax.physics.numerics import fortran_divide, fortran_multiply  

    t_grid = jnp.asarray(temperature, dtype=jnp.float64)
    theta_grid = jnp.asarray(potential_temperature, dtype=jnp.float64)
    q_grid = jnp.asarray(specific_humidity, dtype=jnp.float64)
    p_grid = jnp.asarray(pressure, dtype=jnp.float64)
    rho_grid = jnp.asarray(density, dtype=jnp.float64)
    zbot = jnp.asarray(reference_height, dtype=jnp.float64)
    delta_height = jnp.asarray(elevation_difference, dtype=jnp.float64)
    lapse_rate = jnp.asarray(lapse_rate, dtype=jnp.float64)

    t_column = t_grid - lapse_rate * delta_height
    scale_height = GAS_CONSTANT_DRY_AIR * jnp.float64(0.5) * (t_grid + t_column) / GRAVITY
    p_column = p_grid * jnp.exp(-delta_height / scale_height)
    theta_column = theta_grid + (t_column - t_grid) * jnp.exp(
        (zbot / scale_height) * (GAS_CONSTANT_DRY_AIR / SPECIFIC_HEAT_DRY_AIR)
    )

    qsat_grid = qsat(t_grid, p_grid).specific_humidity
    qsat_column = qsat(t_column, p_column).specific_humidity
    q_column = q_grid * (qsat_column / qsat_grid)

    estimated_rho_column = air_density(q_column, p_column, t_column)
    estimated_rho_grid = air_density(q_grid, p_grid, t_grid)
    rho_column = fortran_multiply(
        rho_grid, fortran_divide(estimated_rho_column, estimated_rho_grid),
    )

    return ThermodynamicForcing(
        temperature=t_column,
        potential_temperature=theta_column,
        specific_humidity=q_column,
        pressure=p_column,
        density=rho_column,
    )


def precipitation_ramp(
    all_snow_temperature_celsius: jnp.ndarray | float,
    all_rain_temperature_celsius: jnp.ndarray | float,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    all_snow_celsius = jnp.asarray(all_snow_temperature_celsius, dtype=jnp.float64)
    all_rain_celsius = jnp.asarray(all_rain_temperature_celsius, dtype=jnp.float64)
    fraction_rain_slope = jnp.float64(1.0) / (all_rain_celsius - all_snow_celsius)
    all_snow_kelvin = all_snow_celsius + FREEZING_TEMPERATURE_FRESHWATER
    return all_snow_kelvin, fraction_rain_slope


def repartition_precipitation(
    temperature: jnp.ndarray,
    rain: jnp.ndarray,
    snow: jnp.ndarray,
    all_snow_temperature: jnp.ndarray,
    fraction_rain_slope: jnp.ndarray,
    *,
    rain_to_snow_runs_off: bool | jnp.ndarray = False,
) -> PrecipitationPartition:

    temperature = jnp.asarray(temperature, dtype=jnp.float64)
    rain_old = jnp.asarray(rain, dtype=jnp.float64)
    snow_old = jnp.asarray(snow, dtype=jnp.float64)
    all_snow_temperature = jnp.asarray(all_snow_temperature, dtype=jnp.float64)
    fraction_rain_slope = jnp.asarray(fraction_rain_slope, dtype=jnp.float64)
    runoff_enabled = jnp.asarray(rain_to_snow_runs_off, dtype=jnp.bool_)

    fraction_rain = (temperature - all_snow_temperature) * fraction_rain_slope
    fraction_rain = jnp.clip(fraction_rain, min=0.0, max=1.0)
    total_precipitation = rain_old + snow_old
    rain_new = total_precipitation * fraction_rain
    snow_candidate = total_precipitation - rain_new

    convert_to_runoff = runoff_enabled & (snow_candidate > snow_old)
    rain_to_snow_runoff = jnp.where(convert_to_runoff, snow_candidate - snow_old, 0.0)
    snow_new = jnp.where(convert_to_runoff, snow_old, snow_candidate)

    sensible_heat_flux = (
        (snow_new - snow_old) * jnp.float64(1.0e-3) * DENSITY_FRESH_WATER * LATENT_HEAT_FUSION
    )
    return PrecipitationPartition(
        rain=rain_new,
        snow=snow_new,
        rain_to_snow_runoff=rain_to_snow_runoff,
        sensible_heat_flux=sensible_heat_flux,
    )


def partition_precipitation_to_columns(
    grid_rain: jnp.ndarray,
    grid_snow: jnp.ndarray,
    column_temperature: jnp.ndarray,
    column_gridcell: jnp.ndarray,
    column_is_glacier: jnp.ndarray,
    rain_to_snow_runs_off_gridcell: jnp.ndarray,
    *,
    glacier_all_snow_temperature: jnp.ndarray,
    glacier_fraction_rain_slope: jnp.ndarray,
    nonglacier_all_snow_temperature: jnp.ndarray,
    nonglacier_fraction_rain_slope: jnp.ndarray,
    repartition_rain_snow: bool | jnp.ndarray = True,
) -> PrecipitationPartition:

    grid_index = jnp.asarray(column_gridcell, dtype=jnp.int32)
    rain_old = jnp.asarray(grid_rain, dtype=jnp.float64)[grid_index]
    snow_old = jnp.asarray(grid_snow, dtype=jnp.float64)[grid_index]
    is_glacier = jnp.asarray(column_is_glacier, dtype=jnp.bool_)
    all_snow_temperature = jnp.where(
        is_glacier, glacier_all_snow_temperature, nonglacier_all_snow_temperature
    )
    fraction_rain_slope = jnp.where(
        is_glacier, glacier_fraction_rain_slope, nonglacier_fraction_rain_slope
    )
    partitioned = repartition_precipitation(
        column_temperature,
        rain_old,
        snow_old,
        all_snow_temperature,
        fraction_rain_slope,
        rain_to_snow_runs_off=jnp.asarray(rain_to_snow_runs_off_gridcell, dtype=jnp.bool_)[
            grid_index
        ],
    )
    enabled = jnp.asarray(repartition_rain_snow, dtype=jnp.bool_)
    zeros = jnp.zeros_like(rain_old)
    return PrecipitationPartition(
        rain=jnp.where(enabled, partitioned.rain, rain_old),
        snow=jnp.where(enabled, partitioned.snow, snow_old),
        rain_to_snow_runoff=jnp.where(enabled, partitioned.rain_to_snow_runoff, zeros),
        sensible_heat_flux=jnp.where(enabled, partitioned.sensible_heat_flux, zeros),
    )


def build_normalization(
    original_field: jnp.ndarray,
    summed_field: jnp.ndarray,
    summed_weights: jnp.ndarray,
) -> jnp.ndarray:

    from clm_jax.physics.numerics import fortran_divide  

    original = jnp.asarray(original_field, dtype=jnp.float64)
    field_sum = jnp.asarray(summed_field, dtype=jnp.float64)
    weight_sum = jnp.asarray(summed_weights, dtype=jnp.float64)
    usable = (weight_sum != 0.0) & (field_sum != 0.0)
    safe_weight_sum = jnp.where(usable, weight_sum, 1.0)
    safe_field_sum = jnp.where(usable, field_sum, 1.0)
    calculated = fortran_divide(
        original, fortran_divide(safe_field_sum, safe_weight_sum),
    )
    return jnp.where(usable, calculated, 1.0)


def _longwave_sums_reference(values, weights, selected, groups, count):
    return (
        jax.ops.segment_sum(jnp.where(selected, weights * values, 0.0), groups, count),
        jax.ops.segment_sum(jnp.where(selected, weights, 0.0), groups, count),
    )


@partial(jax.custom_jvp, nondiff_argnums=(4,))
def _longwave_sums_gpu(values, weights, selected, groups, count):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    from clm_jax.physics.ordered_gpu_reduction import _instruction

    if values.size == 0 or count == 0:
        zero = jnp.zeros((count,), jnp.float64)
        return zero, zero
    index = jnp.arange(values.size, dtype=jnp.int32)
    first = jnp.full((count,), values.size, jnp.int32).at[groups].min(
        jnp.where(selected, index, values.size)
    )
    end = jnp.zeros((count,), jnp.int32).at[groups].max(
        jnp.where(selected, index + 1, 0)
    )
    block = 128

    def kernel(vref, wref, mref, gref, first_ref, end_ref, total_ref, weight_ref):
        group = pl.program_id(0) * block + jnp.arange(block)
        start = plt.load(first_ref.at[group], mask=group < count, other=values.size)
        stop = plt.load(end_ref.at[group], mask=group < count, other=0)

        def accumulate(offset, carry):
            position = start + offset
            in_bounds = (group < count) & (position < stop)
            owns = plt.load(gref.at[position], mask=in_bounds, other=-1) == group
            active = plt.load(mref.at[position], mask=in_bounds, other=False)
            enabled = in_bounds & owns & active
            value = plt.load(vref.at[position], mask=enabled, other=0.0)
            weight = plt.load(wref.at[position], mask=enabled, other=0.0)
            product = _instruction("mul", weight, value)
            total = _instruction("add", carry[0], product)
            sum_weight = _instruction("add", carry[1], weight)
            return jnp.where(enabled, total, carry[0]), jnp.where(
                enabled, sum_weight, carry[1]
            )

        zero = jnp.zeros((block,), jnp.float64)
        limit = jnp.max(jnp.maximum(stop - start, 0))
        total, sum_weight = jax.lax.fori_loop(0, limit, accumulate, (zero, zero))
        plt.store(total_ref.at[group], total, mask=group < count)
        plt.store(weight_ref.at[group], sum_weight, mask=group < count)

    shape = jax.ShapeDtypeStruct((count,), jnp.float64)
    return pl.pallas_call(
        kernel,
        out_shape=(shape, shape),
        grid=(pl.cdiv(count, block),),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=1),
    )(values, weights, selected, groups, first, end)


@_longwave_sums_gpu.defjvp
def _longwave_sums_gpu_jvp(count, primals, tangents):
    values, weights, selected, groups = primals
    primal = _longwave_sums_gpu(values, weights, selected, groups, count)
    _, tangent = jax.jvp(
        lambda v, w: _longwave_sums_reference(v, w, selected, groups, count),
        (values, weights),
        (tangents[0], tangents[1]),
    )
    return primal, tangent


def downscale_longwave(
    grid_longwave: jnp.ndarray,
    atmospheric_elevation: jnp.ndarray,
    column_elevation: jnp.ndarray,
    column_gridcell: jnp.ndarray,
    column_weight_on_gridcell: jnp.ndarray,
    downscale_column: jnp.ndarray,
    *,
    lapse_rate_longwave: jnp.ndarray | float,
    relative_limit: jnp.ndarray | float,
    number_of_gridcells: int,
) -> jnp.ndarray:

    from clm_jax.physics.numerics import fortran_add, fortran_multiply  

    longwave_grid = jnp.asarray(grid_longwave, dtype=jnp.float64)
    grid_index = jnp.asarray(column_gridcell, dtype=jnp.int32)
    weights = jnp.asarray(column_weight_on_gridcell, dtype=jnp.float64)
    selected = jnp.asarray(downscale_column, dtype=jnp.bool_)
    grid_values = longwave_grid[grid_index]
    height_difference = fortran_add(
        jnp.asarray(column_elevation, dtype=jnp.float64),
        -jnp.asarray(atmospheric_elevation, dtype=jnp.float64)[grid_index],
    )

    downscaled = fortran_add(
        grid_values,
        -fortran_multiply(jnp.asarray(lapse_rate_longwave, dtype=jnp.float64),
                          height_difference),
    )
    relative_limit = jnp.asarray(relative_limit, dtype=jnp.float64)
    downscaled = jnp.minimum(downscaled, fortran_multiply(
        grid_values, fortran_add(jnp.float64(1.0), relative_limit)))
    downscaled = jnp.maximum(downscaled, fortran_multiply(
        grid_values, fortran_add(jnp.float64(1.0), -relative_limit)))

    sum_columns = (
        _longwave_sums_gpu if jax.default_backend() == "gpu" else _longwave_sums_reference
    )
    summed_field, summed_weights = sum_columns(
        downscaled, weights, selected, grid_index, number_of_gridcells
    )
    normalization = build_normalization(longwave_grid, summed_field, summed_weights)
    return jnp.where(selected, fortran_multiply(downscaled, normalization[grid_index]),
                     grid_values)


def downscale_atmospheric_forcing_to_columns(
    grid_forcing: DatmForcing,
    atmospheric_elevation: jnp.ndarray,
    column_elevation: jnp.ndarray,
    column_topology: ColumnTopology,
    downscale_column: jnp.ndarray,
    rain_to_snow_runs_off_gridcell: jnp.ndarray,
    parameters: Atm2LndParameters,
) -> ColumnAtmosphericForcing:

    from clm_jax.io.restart import LANDUNIT_GLACIER  

    grid_index = jnp.asarray(column_topology.gridcell, dtype=jnp.int32)
    active = jnp.asarray(column_topology.active, dtype=jnp.bool_)
    selected = jnp.asarray(downscale_column, dtype=jnp.bool_) & active
    atmospheric_elevation = jnp.asarray(atmospheric_elevation, dtype=jnp.float64)
    column_elevation = jnp.asarray(column_elevation, dtype=jnp.float64)

    def gather(field: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(field, dtype=jnp.float64)[grid_index]

    temperature_grid = gather(grid_forcing.temperature)
    potential_temperature_grid = gather(grid_forcing.potential_temperature)
    specific_humidity_grid = gather(grid_forcing.specific_humidity)
    pressure_grid = gather(grid_forcing.pressure)
    density_grid = gather(grid_forcing.density)
    reference_height_grid = gather(grid_forcing.reference_height)

    thermodynamics = downscale_thermodynamics(
        temperature_grid,
        potential_temperature_grid,
        specific_humidity_grid,
        pressure_grid,
        density_grid,
        reference_height_grid,
        column_elevation - atmospheric_elevation[grid_index],
        lapse_rate=parameters.lapse_rate,
    )
    temperature = jnp.where(selected, thermodynamics.temperature, temperature_grid)
    potential_temperature = jnp.where(
        selected,
        thermodynamics.potential_temperature,
        potential_temperature_grid,
    )
    specific_humidity = jnp.where(
        selected,
        thermodynamics.specific_humidity,
        specific_humidity_grid,
    )
    pressure = jnp.where(selected, thermodynamics.pressure, pressure_grid)
    density = jnp.where(selected, thermodynamics.density, density_grid)

    glacier_all_snow, glacier_slope = precipitation_ramp(
        parameters.precip_repartition_glc_all_snow_t,
        parameters.precip_repartition_glc_all_rain_t,
    )
    nonglacier_all_snow, nonglacier_slope = precipitation_ramp(
        parameters.precip_repartition_nonglc_all_snow_t,
        parameters.precip_repartition_nonglc_all_rain_t,
    )
    precipitation = partition_precipitation_to_columns(
        grid_forcing.rain,
        grid_forcing.snow,
        temperature,
        grid_index,
        jnp.asarray(column_topology.landunit_type) == LANDUNIT_GLACIER,
        rain_to_snow_runs_off_gridcell,
        glacier_all_snow_temperature=glacier_all_snow,
        glacier_fraction_rain_slope=glacier_slope,
        nonglacier_all_snow_temperature=nonglacier_all_snow,
        nonglacier_fraction_rain_slope=nonglacier_slope,
        repartition_rain_snow=parameters.repartition_rain_snow,
    )

    def scaled_longwave(_: None) -> jnp.ndarray:
        return downscale_longwave(
            grid_forcing.longwave_down,
            atmospheric_elevation,
            column_elevation,
            grid_index,
            column_topology.weight_on_gridcell,
            selected,
            lapse_rate_longwave=parameters.lapse_rate_longwave,
            relative_limit=parameters.longwave_downscaling_limit,
            number_of_gridcells=atmospheric_elevation.shape[0],
        )

    longwave_down = jax.lax.cond(
        jnp.asarray(parameters.glcmec_downscale_longwave, dtype=jnp.bool_),
        scaled_longwave,
        lambda _: gather(grid_forcing.longwave_down),
        operand=None,
    )
    return ColumnAtmosphericForcing(
        temperature=temperature,
        potential_temperature=potential_temperature,
        specific_humidity=specific_humidity,
        pressure=pressure,
        density=density,
        longwave_down=longwave_down,
        rain=precipitation.rain,
        snow=precipitation.snow,
        rain_to_snow_runoff=precipitation.rain_to_snow_runoff,
        sensible_heat_flux=precipitation.sensible_heat_flux,
    )
