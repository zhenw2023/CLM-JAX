
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.constants import (
    FREEZING_TEMPERATURE_FRESHWATER,
    PI,
    SECONDS_PER_DAY,
)
from clm_jax.physics.cn_precision_control import (
    CnPrecisionControlState,
    get_cn_precision_control_state_field,
)
from clm_jax.physics.adjoint_scaling import sensitivity_product
from clm_jax.physics.converged_iteration import _dynamic_primal, gpu_convergence_scope
from clm_jax.physics.numerics import (
    fortran_divide,
    strict_exp,
    strict_negative_exp,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.numerics import (
    fortran_hardware_power as fortran_power,
)
from clm_jax.physics.pointwise import scalar_map
from clm_jax.physics.subgrid import patch_to_column_average

MAXIMUM_PATCHES_PER_FIRE_COLUMN = 79
NUMBER_OF_FIRE_DECOMPOSITION_LEVELS = 20
_FIRE_TOPOLOGY_BOUND = True


class CnFireAreaTopology(NamedTuple):

    soil_column: jnp.ndarray
    soil_patch: jnp.ndarray
    column_gridcell_index: jnp.ndarray
    patch_column_index: jnp.ndarray
    patch_weight_on_column: jnp.ndarray
    patch_active: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    gridcell_latitude_degrees: jnp.ndarray


class CnFireAreaForcing(NamedTuple):

    relative_humidity_gridcell: jnp.ndarray
    wind_speed_gridcell: jnp.ndarray
    temperature_downscaled_column: jnp.ndarray
    rain_downscaled_column: jnp.ndarray
    snow_downscaled_column: jnp.ndarray
    precipitation_60day_patch: jnp.ndarray
    precipitation_10day_patch: jnp.ndarray
    relative_humidity_30day_patch: jnp.ndarray
    root_zone_wetness_patch: jnp.ndarray
    saturated_area_fraction_column: jnp.ndarray
    topsoil_water_fraction_column: jnp.ndarray
    lightning_frequency_gridcell: jnp.ndarray
    human_population_density_gridcell: jnp.ndarray
    total_litter_carbon_column: jnp.ndarray
    total_vegetation_carbon_column: jnp.ndarray
    decomposition_carbon_pool_column_level_pool: jnp.ndarray
    soil_temperature_17cm_column: jnp.ndarray
    gdp_column: jnp.ndarray
    peatland_fraction_column: jnp.ndarray
    crop_fire_peak_month_column: jnp.ndarray


class CnFireAreaState(NamedTuple):

    burn_date_patch: jnp.ndarray
    unburned_conversion_fraction_column: jnp.ndarray
    crop_burned_area_fraction_column: jnp.ndarray
    peat_burned_area_fraction_column: jnp.ndarray
    conversion_burned_area_fraction_column: jnp.ndarray
    conversion_fire_burned_area_fraction_column: jnp.ndarray
    total_burned_area_fraction_column: jnp.ndarray
    fire_count_column: jnp.ndarray
    fire_spread_rate_column: jnp.ndarray
    fire_duration_column: jnp.ndarray
    gdp_occurrence_limit_column: jnp.ndarray
    gdp_spread_limit_column: jnp.ndarray
    population_spread_limit_column: jnp.ndarray
    natural_vegetation_fraction_column: jnp.ndarray
    tropical_evergreen_tree_fraction_column: jnp.ndarray
    tropical_deciduous_tree_fraction_column: jnp.ndarray
    decreased_tropical_tree_fraction_column: jnp.ndarray
    crop_fraction_column: jnp.ndarray
    noncrop_fraction_column: jnp.ndarray
    root_carbon_column: jnp.ndarray
    leaf_carbon_column: jnp.ndarray
    deadstem_carbon_column: jnp.ndarray
    fuel_carbon_column: jnp.ndarray
    crop_fuel_carbon_column: jnp.ndarray


class CnFireAreaParameters(NamedTuple):

    fire_spread_rate_pft: jnp.ndarray
    fire_duration_pft: jnp.ndarray
    decomposition_pool_is_cwd: jnp.ndarray
    decomposition_pool_spinup_factor: jnp.ndarray
    decomposition_layer_thickness: jnp.ndarray
    lower_fuel_threshold: float | jnp.ndarray = 105.0
    upper_fuel_threshold: float | jnp.ndarray = 1050.0
    relative_humidity_low: float | jnp.ndarray = 30.0
    relative_humidity_high: float | jnp.ndarray = 80.0
    root_wetness_minimum: float | jnp.ndarray = 0.85
    root_wetness_maximum: float | jnp.ndarray = 0.98
    deforestation_fire_scale: float | jnp.ndarray = 0.033
    crop_fire_coefficient: float | jnp.ndarray = 1.6e-4
    non_boreal_peat_fire_coefficient: float | jnp.ndarray = 0.17e-3
    human_ignition_coefficient: float | jnp.ndarray = 0.010
    boreal_peat_fire_coefficient: float | jnp.ndarray = 0.09e-4
    high_gdp_tree_occurrence: float | jnp.ndarray = 0.33
    base_spread_factor: float | jnp.ndarray = 0.05
    boreal_latitude_degrees: float | jnp.ndarray = 40.0
    time_step: float | jnp.ndarray = 1800.0
    days_per_year: float | jnp.ndarray = 365.0
    model_step: int | jnp.ndarray = 1
    month: int | jnp.ndarray = 1
    day_of_month: int | jnp.ndarray = 1
    seconds_of_day: int | jnp.ndarray = 1800
    spinup_state: int | jnp.ndarray = 0
    bare_ground_pft: int = 0
    first_natural_vegetation_pft: int = 1
    tropical_evergreen_tree_pft: int = 4
    tropical_deciduous_tree_pft: int = 6
    first_shrub_pft: int = 9
    c4_grass_pft: int = 14
    first_crop_pft: int = 15


class CnFireAreaDiagnostics(NamedTuple):
    precipitation_60day_column: jnp.ndarray
    precipitation_10day_column: jnp.ndarray
    relative_humidity_30day_column: jnp.ndarray
    root_zone_wetness_column: jnp.ndarray
    fuel_availability_column: jnp.ndarray
    fire_combustibility_column: jnp.ndarray
    first_step: jnp.ndarray
    updated_column_count: jnp.ndarray


class CnFireAreaResult(NamedTuple):
    state: CnFireAreaState
    diagnostics: CnFireAreaDiagnostics


def _mul(*values: jnp.ndarray) -> jnp.ndarray:
    result = jnp.asarray(values[0], dtype=jnp.float64)
    for value in values[1:]:
        result = fortran_multiply(result, value)
    return result


def _add(*values: jnp.ndarray) -> jnp.ndarray:
    result = jnp.asarray(values[0], dtype=jnp.float64)
    for value in values[1:]:
        result = fortran_add(result, value)
    return result


def _clamp01(value: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(jnp.float64(0.0), jnp.minimum(jnp.float64(1.0), value))


@jax.custom_jvp
def _fire_combustibility(humidity, wetness, active):
    return jnp.where(active, _mul(fortran_power(humidity, jnp.float64(1.5)),
                                  fortran_power(wetness, jnp.float64(0.5))), 0.0)


@_fire_combustibility.defjvp
def _fire_combustibility_jvp(primals, tangents):
    humidity, wetness, active = primals
    h = jnp.where(active, humidity, 1.0)
    w = jnp.where(active, wetness, 1.0)
    h_slope = jnp.where(active, 1.5 * jnp.sqrt(h) * jnp.sqrt(w), 0.0)
    safe_w = jnp.where((h == 0) & (w == 0), 1.0, w)
    w_slope = jnp.where(active, 0.5 * h**1.5 / jnp.sqrt(safe_w), 0.0)
    tangent = (sensitivity_product(h_slope, tangents[0])
               + sensitivity_product(w_slope, tangents[1]))
    return _fire_combustibility(*primals), tangent


@jax.custom_jvp
def _natural_fire_area(humidity, wetness, active, base_spread, spread, duration,
                       gdp_spread, population_spread, ignitions, fuel, occurrence, length):
    combustibility = _fire_combustibility(humidity, wetness, active)
    count = _mul(fortran_divide(ignitions, jnp.float64(3600.0)), fuel,
                 combustibility, occurrence)
    base = fortran_divide(_mul(base_spread, fortran_power(combustibility, jnp.float64(0.5)),
                              spread, duration), jnp.float64(1000.0))
    return _mul(_mul(base, base), gdp_spread, population_spread, count, jnp.float64(PI), length)


@_natural_fire_area.defjvp
def _natural_fire_area_jvp(primals, tangents):
    active = primals[2]
    continuous = primals[:2] + primals[3:]
    derivatives = tangents[:2] + tangents[3:]

    def differentiable(h, w, base, spread, duration, gdp, population, ignitions,
                       fuel, occurrence, length):
        h, w, base, spread, duration, gdp, population, ignitions, fuel, occurrence, length = (
            jnp.where(active, value, 1.0)
            for value in (h, w, base, spread, duration, gdp, population, ignitions,
                          fuel, occurrence, length))
        scaled = base * spread * duration / 1000.0
        value = (scaled * scaled * gdp * population * (ignitions / 3600.0)
                 * fuel * occurrence * PI * length * h * h * h * w)
        return jnp.where(active, value, 0.0)

    tangent = jax.jvp(differentiable, continuous, derivatives)[1]
    return _natural_fire_area(*primals), tangent


def _scalar_cos(value: jnp.ndarray) -> jnp.ndarray:
    value = jnp.asarray(value, dtype=jnp.float64)
    shape = value.shape
    result = scalar_map(jnp.cos, jnp.reshape(value, (-1,)))
    return jnp.reshape(result, shape)


def _validate_shapes(
    topology: CnFireAreaTopology,
    forcing: CnFireAreaForcing,
    vegetation_state: CnPrecisionControlState,
    state: CnFireAreaState,
    parameters: CnFireAreaParameters,
) -> tuple[int, int, int]:
    column_count = topology.soil_column.shape[0]
    patch_count = topology.soil_patch.shape[0]
    gridcell_count = topology.gridcell_latitude_degrees.shape[0]
    if topology.soil_column.shape != (column_count,):
        raise ValueError("soil_column must be a vector")
    for name in ("column_gridcell_index",):
        if jnp.shape(getattr(topology, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    for name in (
        "soil_patch",
        "patch_column_index",
        "patch_weight_on_column",
        "patch_active",
        "vegetation_type_patch",
    ):
        if jnp.shape(getattr(topology, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")

    grid_fields = (
        forcing.relative_humidity_gridcell,
        forcing.wind_speed_gridcell,
        forcing.lightning_frequency_gridcell,
        forcing.human_population_density_gridcell,
    )
    if any(jnp.shape(field) != (gridcell_count,) for field in grid_fields):
        raise ValueError("all gridcell forcing fields must share the grid extent")
    column_fields = (
        forcing.temperature_downscaled_column,
        forcing.rain_downscaled_column,
        forcing.snow_downscaled_column,
        forcing.saturated_area_fraction_column,
        forcing.topsoil_water_fraction_column,
        forcing.total_litter_carbon_column,
        forcing.total_vegetation_carbon_column,
        forcing.soil_temperature_17cm_column,
        forcing.gdp_column,
        forcing.peatland_fraction_column,
        forcing.crop_fire_peak_month_column,
    )
    if any(jnp.shape(field) != (column_count,) for field in column_fields):
        raise ValueError("all column forcing fields must share the column extent")
    patch_fields = (
        forcing.precipitation_60day_patch,
        forcing.precipitation_10day_patch,
        forcing.relative_humidity_30day_patch,
        forcing.root_zone_wetness_patch,
    )
    if any(jnp.shape(field) != (patch_count,) for field in patch_fields):
        raise ValueError("all patch forcing fields must share the patch extent")
    decomposition = forcing.decomposition_carbon_pool_column_level_pool
    if decomposition.ndim != 3 or decomposition.shape[0] != column_count:
        raise ValueError("decomposition carbon must be column x level x pool")
    if decomposition.shape[1] < NUMBER_OF_FIRE_DECOMPOSITION_LEVELS:
        raise ValueError("decomposition carbon has too few active levels")
    if decomposition.shape[2] != parameters.decomposition_pool_is_cwd.shape[0]:
        raise ValueError("decomposition carbon and cascade pools disagree")
    if parameters.decomposition_pool_spinup_factor.shape != (decomposition.shape[2],):
        raise ValueError("spinup factors must contain every decomposition pool")
    if parameters.decomposition_layer_thickness.shape[0] < NUMBER_OF_FIRE_DECOMPOSITION_LEVELS:
        raise ValueError("layer thickness has too few active levels")

    if vegetation_state.paired_carbon_patch.shape[0] != patch_count:
        raise ValueError("vegetation state has the wrong patch extent")
    if state.burn_date_patch.shape != (patch_count,):
        raise ValueError("burn date must contain every patch")
    for name in state._fields[1:]:
        if jnp.shape(getattr(state, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    return patch_count, column_count, gridcell_count


def _spinup_latitude_term(latitude: jnp.ndarray) -> jnp.ndarray:

    exponent = -_mul(
        jnp.float64(0.15),
        fortran_add(jnp.abs(latitude), -jnp.float64(60.0)),
    )
    return fortran_add(
        jnp.float64(1.0),
        fortran_divide(
            jnp.float64(50.0),
            fortran_add(jnp.float64(1.0), strict_exp(exponent)),
        ),
    )


def _first_patch_per_column(
    patch_column: jnp.ndarray,
    column_count: int,
) -> jnp.ndarray:
    patch_count = patch_column.shape[0]
    patch_index = jnp.arange(patch_count, dtype=jnp.int32)
    first = jnp.full(column_count, patch_count, dtype=jnp.int32)
    return first.at[patch_column].min(patch_index)


def _carbon(
    vegetation_state: CnPrecisionControlState,
    name: str,
) -> jnp.ndarray:
    return get_cn_precision_control_state_field(
        vegetation_state,
        f"vegetation_carbon.{name}_patch",
    )


def _fire_patch_limit(patch_column, first_patch, soil_column):
    last = jnp.full(first_patch.shape, -1, jnp.int32).at[patch_column].max(
        jnp.arange(patch_column.size, dtype=jnp.int32)
    )
    span = jnp.where(soil_column, jnp.maximum(last - first_patch + 1, 0), 0)
    return jnp.minimum(jnp.max(span, initial=0), MAXIMUM_PATCHES_PER_FIRE_COLUMN)


def _cn_fire_area_li2016_fixed_landcover(
    topology: CnFireAreaTopology,
    forcing: CnFireAreaForcing,
    vegetation_state: CnPrecisionControlState,
    state: CnFireAreaState,
    parameters: CnFireAreaParameters,
) -> CnFireAreaResult:

    patch_count, column_count, _ = _validate_shapes(
        topology,
        forcing,
        vegetation_state,
        state,
        parameters,
    )
    soil_column = jnp.asarray(topology.soil_column, dtype=jnp.bool_)
    soil_patch = jnp.asarray(topology.soil_patch, dtype=jnp.bool_)
    patch_column = jnp.asarray(topology.patch_column_index, dtype=jnp.int32)
    patch_weight = jnp.asarray(
        topology.patch_weight_on_column,
        dtype=jnp.float64,
    )
    patch_active = jnp.asarray(topology.patch_active, dtype=jnp.bool_)
    vegetation_type = jnp.asarray(
        topology.vegetation_type_patch,
        dtype=jnp.int32,
    )
    column_gridcell = jnp.asarray(
        topology.column_gridcell_index,
        dtype=jnp.int32,
    )
    first_patch = _first_patch_per_column(patch_column, column_count)
    column_index = jnp.arange(column_count, dtype=jnp.int32)
    patch_limit = (
        _fire_patch_limit(patch_column, first_patch, soil_column)
        if _FIRE_TOPOLOGY_BOUND and _dynamic_primal.get()
        else MAXIMUM_PATCHES_PER_FIRE_COLUMN
    )

    def p2c(values: jnp.ndarray) -> jnp.ndarray:
        return patch_to_column_average(
            values,
            patch_column,
            patch_weight,
            patch_active,
            soil_column,
            maximum_patches_per_column=MAXIMUM_PATCHES_PER_FIRE_COLUMN,
        )

    precipitation_10day = p2c(forcing.precipitation_10day_patch)
    precipitation_60day = p2c(forcing.precipitation_60day_patch)
    relative_humidity_30day = p2c(forcing.relative_humidity_30day_patch)
    leaf_carbon = p2c(_carbon(vegetation_state, "leafc"))
    deadstem_carbon = p2c(_carbon(vegetation_state, "deadstemc"))

    def first_step_result(_: None) -> CnFireAreaResult:
        zero = jnp.zeros(column_count, dtype=jnp.float64)
        updated_state = state._replace(
            crop_burned_area_fraction_column=jnp.where(
                soil_column,
                zero,
                state.crop_burned_area_fraction_column,
            ),
            peat_burned_area_fraction_column=jnp.where(
                soil_column,
                zero,
                state.peat_burned_area_fraction_column,
            ),
            conversion_burned_area_fraction_column=jnp.where(
                soil_column,
                zero,
                state.conversion_burned_area_fraction_column,
            ),
            conversion_fire_burned_area_fraction_column=jnp.where(
                soil_column,
                zero,
                state.conversion_fire_burned_area_fraction_column,
            ),
            total_burned_area_fraction_column=jnp.where(
                soil_column,
                zero,
                state.total_burned_area_fraction_column,
            ),
            crop_fraction_column=jnp.where(
                soil_column,
                zero,
                state.crop_fraction_column,
            ),
            leaf_carbon_column=jnp.where(
                soil_column,
                leaf_carbon,
                state.leaf_carbon_column,
            ),
            deadstem_carbon_column=jnp.where(
                soil_column,
                deadstem_carbon,
                state.deadstem_carbon_column,
            ),
        )
        return CnFireAreaResult(
            state=updated_state,
            diagnostics=CnFireAreaDiagnostics(
                precipitation_60day_column=precipitation_60day,
                precipitation_10day_column=precipitation_10day,
                relative_humidity_30day_column=relative_humidity_30day,
                root_zone_wetness_column=jnp.zeros_like(zero),
                fuel_availability_column=jnp.zeros_like(zero),
                fire_combustibility_column=jnp.zeros_like(zero),
                first_step=jnp.bool_(True),
                updated_column_count=jnp.sum(soil_column, dtype=jnp.int32),
            ),
        )

    def regular_result(_: None) -> CnFireAreaResult:
        zero = jnp.zeros(column_count, dtype=jnp.float64)
        crop_fraction = zero
        natural_fraction = zero

        def fraction_step(
            offset: int,
            carry: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            crop, natural = carry
            unbounded = first_patch + jnp.int32(offset)
            bounded = jnp.clip(unbounded, 0, patch_count - 1)
            belongs = (
                soil_column & (unbounded < patch_count) & (patch_column[bounded] == column_index)
            )
            pft = vegetation_type[bounded]
            weight = patch_weight[bounded]
            crop_condition = belongs & (pft > parameters.c4_grass_pft)
            natural_condition = (
                belongs
                & (pft >= parameters.first_natural_vegetation_pft)
                & (pft <= parameters.c4_grass_pft)
            )
            crop_candidate = fortran_add(crop, weight)
            natural_candidate = fortran_add(natural, weight)
            return (
                jnp.where(crop_condition, crop_candidate, crop),
                jnp.where(natural_condition, natural_candidate, natural),
            )

        crop_fraction, natural_fraction = lax.fori_loop(
            0,
            patch_limit,
            fraction_step,
            (crop_fraction, natural_fraction),
        )

        leaf = _carbon(vegetation_state, "leafc")
        leaf_storage = _carbon(vegetation_state, "leafc_storage")
        leaf_xfer = _carbon(vegetation_state, "leafc_xfer")
        crop_fuel = zero
        safe_crop_fraction = jnp.where(
            crop_fraction > jnp.float64(0.0),
            crop_fraction,
            jnp.float64(1.0),
        )
        safe_leaf_carbon = jnp.where(
            leaf_carbon > jnp.float64(0.0),
            leaf_carbon,
            jnp.float64(1.0),
        )

        def crop_fuel_step(offset: int, current: jnp.ndarray) -> jnp.ndarray:
            unbounded = first_patch + jnp.int32(offset)
            bounded = jnp.clip(unbounded, 0, patch_count - 1)
            belongs = (
                soil_column & (unbounded < patch_count) & (patch_column[bounded] == column_index)
            )
            weight = patch_weight[bounded]
            condition = (
                belongs
                & (vegetation_type[bounded] > parameters.c4_grass_pft)
                & (weight > jnp.float64(0.0))
                & (leaf_carbon > jnp.float64(0.0))
            )
            live_leaf = _add(
                leaf[bounded],
                leaf_storage[bounded],
                leaf_xfer[bounded],
            )
            first_term = fortran_divide(
                _mul(live_leaf, weight),
                safe_crop_fraction,
            )
            litter_leaf = _mul(
                forcing.total_litter_carbon_column,
                leaf[bounded],
            )
            litter_leaf = fortran_divide(litter_leaf, safe_leaf_carbon)
            litter_leaf = _mul(litter_leaf, weight)
            second_term = fortran_divide(
                litter_leaf,
                safe_crop_fraction,
            )
            candidate = _add(current, first_term, second_term)
            return jnp.where(condition, candidate, current)

        crop_fuel = lax.fori_loop(
            0,
            patch_limit,
            crop_fuel_step,
            crop_fuel,
        )

        fire_spread = zero
        fire_duration = zero
        root_carbon = zero
        gdp_occurrence = zero
        gdp_spread = zero
        population_spread = zero
        root_wetness = zero
        noncrop_fraction = zero
        tropical_evergreen = zero
        tropical_deciduous = zero
        safe_noncrop_denominator = jnp.where(
            crop_fraction < jnp.float64(1.0),
            fortran_add(jnp.float64(1.0), -crop_fraction),
            jnp.float64(1.0),
        )

        froot = _carbon(vegetation_state, "frootc")
        froot_storage = _carbon(vegetation_state, "frootc_storage")
        froot_xfer = _carbon(vegetation_state, "frootc_xfer")
        deadcroot = _carbon(vegetation_state, "deadcrootc")
        deadcroot_storage = _carbon(vegetation_state, "deadcrootc_storage")
        deadcroot_xfer = _carbon(vegetation_state, "deadcrootc_xfer")
        livecroot = _carbon(vegetation_state, "livecrootc")
        livecroot_storage = _carbon(vegetation_state, "livecrootc_storage")
        livecroot_xfer = _carbon(vegetation_state, "livecrootc_xfer")

        def noncrop_step(
            offset: int,
            carry: tuple[jnp.ndarray, ...],
        ) -> tuple[jnp.ndarray, ...]:
            (
                spread,
                duration,
                root,
                occurrence_limit,
                spread_limit,
                population_limit,
                wetness,
                noncrop_weight,
                tropical_evr,
                tropical_dcd,
            ) = carry
            unbounded = first_patch + jnp.int32(offset)
            bounded = jnp.clip(unbounded, 0, patch_count - 1)
            belongs = (
                soil_column & (unbounded < patch_count) & (patch_column[bounded] == column_index)
            )
            pft = vegetation_type[bounded]
            weight = patch_weight[bounded]
            noncrop = (
                belongs & (pft < parameters.first_crop_pft) & (crop_fraction < jnp.float64(1.0))
            )
            wetness_valid = (
                noncrop
                & ~jnp.isnan(forcing.root_zone_wetness_patch[bounded])
                & (forcing.root_zone_wetness_patch[bounded] <= jnp.float64(1.0))
            )
            wetness_candidate = fortran_add(
                wetness,
                _mul(forcing.root_zone_wetness_patch[bounded], weight),
            )
            noncrop_weight_candidate = fortran_add(noncrop_weight, weight)
            wetness = jnp.where(wetness_valid, wetness_candidate, wetness)
            noncrop_weight = jnp.where(
                wetness_valid,
                noncrop_weight_candidate,
                noncrop_weight,
            )

            tropical_evr = jnp.where(
                noncrop
                & (pft == parameters.tropical_evergreen_tree_pft)
                & (weight > jnp.float64(0.0)),
                fortran_add(tropical_evr, weight),
                tropical_evr,
            )
            tropical_dcd = jnp.where(
                noncrop
                & (pft == parameters.tropical_deciduous_tree_pft)
                & (weight > jnp.float64(0.0)),
                fortran_add(tropical_dcd, weight),
                tropical_dcd,
            )

            accelerated_deadcroot = jnp.where(
                jnp.asarray(parameters.spinup_state) == jnp.int32(2),
                _mul(deadcroot[bounded], jnp.float64(10.0)),
                deadcroot[bounded],
            )
            root_sum = _add(
                froot[bounded],
                froot_storage[bounded],
                froot_xfer[bounded],
                accelerated_deadcroot,
                deadcroot_storage[bounded],
                deadcroot_xfer[bounded],
                livecroot[bounded],
                livecroot_storage[bounded],
                livecroot_xfer[bounded],
            )
            root = jnp.where(
                noncrop,
                fortran_add(root, _mul(root_sum, weight)),
                root,
            )
            spread = jnp.where(
                noncrop,
                fortran_add(
                    spread,
                    fortran_divide(
                        _mul(parameters.fire_spread_rate_pft[pft], weight),
                        safe_noncrop_denominator,
                    ),
                ),
                spread,
            )

            gridcell = column_gridcell
            population = forcing.human_population_density_gridcell[gridcell]
            gdp = forcing.gdp_column
            populated = population > jnp.float64(0.1)
            vegetated = pft != parameters.bare_ground_pft
            shrub_or_grass = pft >= parameters.first_shrub_pft
            shrub_occurrence = _add(
                jnp.float64(0.1),
                _mul(
                    jnp.float64(0.9),
                    strict_negative_exp(
                        -_mul(
                            jnp.float64(PI),
                            fortran_power(
                                fortran_divide(gdp, jnp.float64(8.0)),
                                jnp.float64(0.5),
                            ),
                        )
                    ),
                ),
            )
            shrub_gdp_spread = _add(
                jnp.float64(0.2),
                _mul(
                    jnp.float64(0.8),
                    strict_negative_exp(
                        -_mul(
                            jnp.float64(PI),
                            fortran_divide(gdp, jnp.float64(7.0)),
                        )
                    ),
                ),
            )
            shrub_population = _add(
                jnp.float64(0.2),
                _mul(
                    jnp.float64(0.8),
                    strict_negative_exp(
                        -_mul(
                            jnp.float64(PI),
                            fortran_power(
                                fortran_divide(population, jnp.float64(450.0)),
                                jnp.float64(0.5),
                            ),
                        )
                    ),
                ),
            )
            tree_occurrence = jnp.where(
                gdp > jnp.float64(20.0),
                jnp.asarray(parameters.high_gdp_tree_occurrence, dtype=jnp.float64),
                jnp.where(
                    gdp > jnp.float64(8.0),
                    jnp.float64(0.79),
                    jnp.float64(1.0),
                ),
            )
            tree_gdp_spread = jnp.where(
                gdp > jnp.float64(20.0),
                jnp.float64(0.62),
                jnp.where(
                    gdp > jnp.float64(8.0),
                    jnp.float64(0.83),
                    jnp.float64(1.0),
                ),
            )
            tree_population = _add(
                jnp.float64(0.4),
                _mul(
                    jnp.float64(0.6),
                    strict_negative_exp(
                        -_mul(
                            jnp.float64(PI),
                            fortran_divide(population, jnp.float64(125.0)),
                        )
                    ),
                ),
            )
            occurrence_factor = jnp.where(
                populated,
                jnp.where(
                    vegetated,
                    jnp.where(shrub_or_grass, shrub_occurrence, tree_occurrence),
                    jnp.float64(0.0),
                ),
                jnp.float64(1.0),
            )
            spread_factor = jnp.where(
                populated,
                jnp.where(
                    vegetated,
                    jnp.where(shrub_or_grass, shrub_gdp_spread, tree_gdp_spread),
                    jnp.float64(0.0),
                ),
                jnp.float64(1.0),
            )
            population_factor = jnp.where(
                populated,
                jnp.where(
                    vegetated,
                    jnp.where(shrub_or_grass, shrub_population, tree_population),
                    jnp.float64(0.0),
                ),
                jnp.float64(1.0),
            )
            occurrence_contribution = fortran_divide(
                _mul(occurrence_factor, weight),
                safe_noncrop_denominator,
            )
            spread_contribution = fortran_divide(
                _mul(spread_factor, weight),
                safe_noncrop_denominator,
            )
            population_contribution = fortran_divide(
                _mul(population_factor, weight),
                safe_noncrop_denominator,
            )
            occurrence_limit = jnp.where(
                noncrop & (~populated | vegetated),
                fortran_add(occurrence_limit, occurrence_contribution),
                occurrence_limit,
            )
            spread_limit = jnp.where(
                noncrop & (~populated | vegetated),
                fortran_add(spread_limit, spread_contribution),
                spread_limit,
            )
            population_limit = jnp.where(
                noncrop & (~populated | vegetated),
                fortran_add(population_limit, population_contribution),
                population_limit,
            )
            duration = jnp.where(
                noncrop,
                fortran_add(
                    duration,
                    fortran_divide(
                        _mul(
                            parameters.fire_duration_pft[pft],
                            weight,
                            jnp.float64(3600.0),
                        ),
                        safe_noncrop_denominator,
                    ),
                ),
                duration,
            )
            return (
                spread,
                duration,
                root,
                occurrence_limit,
                spread_limit,
                population_limit,
                wetness,
                noncrop_weight,
                tropical_evr,
                tropical_dcd,
            )

        (
            fire_spread,
            fire_duration,
            root_carbon,
            gdp_occurrence,
            gdp_spread,
            population_spread,
            root_wetness,
            noncrop_fraction,
            tropical_evergreen,
            tropical_deciduous,
        ) = lax.fori_loop(
            0,
            patch_limit,
            noncrop_step,
            (
                fire_spread,
                fire_duration,
                root_carbon,
                gdp_occurrence,
                gdp_spread,
                population_spread,
                root_wetness,
                noncrop_fraction,
                tropical_evergreen,
                tropical_deciduous,
            ),
        )

        january_start = (
            (jnp.asarray(parameters.month) == jnp.int32(1))
            & (jnp.asarray(parameters.day_of_month) == jnp.int32(1))
            & (jnp.asarray(parameters.seconds_of_day) == jnp.int32(0))
        )
        burn_date = jnp.where(
            soil_patch & january_start,
            jnp.int32(10000),
            state.burn_date_patch,
        )
        crop_burned = zero
        grid_population = forcing.human_population_density_gridcell[column_gridcell]
        crop_human_effect = _add(
            jnp.float64(0.04),
            _mul(
                jnp.float64(0.96),
                strict_negative_exp(
                    -_mul(
                        jnp.float64(PI),
                        fortran_power(
                            fortran_divide(
                                grid_population,
                                jnp.float64(350.0),
                            ),
                            jnp.float64(0.5),
                        ),
                    )
                ),
            ),
        )
        crop_gdp_effect = _add(
            jnp.float64(0.01),
            _mul(
                jnp.float64(0.99),
                strict_negative_exp(
                    -_mul(
                        jnp.float64(PI),
                        fortran_divide(
                            forcing.gdp_column,
                            jnp.float64(10.0),
                        ),
                    )
                ),
            ),
        )
        crop_fuel_availability = _clamp01(
            fortran_divide(
                fortran_add(
                    crop_fuel,
                    -jnp.asarray(parameters.lower_fuel_threshold),
                ),
                fortran_add(
                    jnp.asarray(parameters.upper_fuel_threshold),
                    -jnp.asarray(parameters.lower_fuel_threshold),
                ),
            )
        )

        def crop_fire_step(
            offset: int,
            burned: jnp.ndarray,
        ) -> jnp.ndarray:
            unbounded = first_patch + jnp.int32(offset)
            bounded = jnp.clip(unbounded, 0, patch_count - 1)
            belongs = (
                soil_column & (unbounded < patch_count) & (patch_column[bounded] == column_index)
            )
            pft = vegetation_type[bounded]
            weight = patch_weight[bounded]
            crop_condition = (
                belongs
                & (
                    forcing.temperature_downscaled_column
                    >= jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)
                )
                & (pft > parameters.c4_grass_pft)
                & (jnp.asarray(parameters.month) == forcing.crop_fire_peak_month_column)
                & (burn_date[bounded] >= jnp.int32(999))
                & (weight > jnp.float64(0.0))
            )
            contribution = _mul(
                fortran_divide(
                    jnp.asarray(parameters.crop_fire_coefficient),
                    jnp.float64(3600.0),
                ),
                crop_human_effect,
                crop_gdp_effect,
                weight,
            )
            return jnp.where(
                crop_condition,
                fortran_add(burned, contribution),
                burned,
            )

        crop_burned = lax.fori_loop(
            0,
            patch_limit,
            crop_fire_step,
            crop_burned,
        )
        patch_human_effect = crop_human_effect[patch_column]
        patch_gdp_effect = crop_gdp_effect[patch_column]
        patch_fuel_availability = crop_fuel_availability[patch_column]
        patch_crop_condition = (
            soil_column[patch_column]
            & (
                forcing.temperature_downscaled_column[patch_column]
                >= jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)
            )
            & (vegetation_type > parameters.c4_grass_pft)
            & (jnp.asarray(parameters.month) == forcing.crop_fire_peak_month_column[patch_column])
            & (burn_date >= jnp.int32(999))
            & (patch_weight > jnp.float64(0.0))
        )
        patch_catches_fire = patch_crop_condition & (
            _mul(
                patch_fuel_availability,
                patch_human_effect,
                patch_gdp_effect,
                patch_weight,
            )
            > jnp.float64(0.0)
        )
        burn_date = jnp.where(
            patch_catches_fire,
            jnp.asarray(parameters.day_of_month, dtype=jnp.int32),
            burn_date,
        )

        latitude = topology.gridcell_latitude_degrees[column_gridcell]
        non_boreal = latitude < jnp.asarray(parameters.boreal_latitude_degrees)
        non_boreal_dryness = _clamp01(
            fortran_divide(
                fortran_add(
                    jnp.float64(4.0),
                    -_mul(
                        precipitation_60day,
                        jnp.float64(SECONDS_PER_DAY),
                    ),
                ),
                jnp.float64(4.0),
            )
        )
        peat_non_boreal = _mul(
            fortran_divide(
                jnp.asarray(parameters.non_boreal_peat_fire_coefficient),
                jnp.float64(3600.0),
            ),
            _mul(non_boreal_dryness, non_boreal_dryness),
            forcing.peatland_fraction_column,
            fortran_add(
                jnp.float64(1.0),
                -forcing.saturated_area_fraction_column,
            ),
        )
        peat_boreal = _mul(
            fortran_divide(
                jnp.asarray(parameters.boreal_peat_fire_coefficient),
                jnp.float64(3600.0),
            ),
            strict_negative_exp(
                -_mul(
                    jnp.float64(PI),
                    fortran_divide(
                        jnp.maximum(
                            forcing.topsoil_water_fraction_column,
                            jnp.float64(0.0),
                        ),
                        jnp.float64(0.3),
                    ),
                )
            ),
            _clamp01(
                fortran_divide(
                    fortran_add(
                        forcing.soil_temperature_17cm_column,
                        -jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
                    ),
                    jnp.float64(10.0),
                )
            ),
            forcing.peatland_fraction_column,
            fortran_add(
                jnp.float64(1.0),
                -forcing.saturated_area_fraction_column,
            ),
        )
        peat_burned = jnp.where(non_boreal, peat_non_boreal, peat_boreal)
        peat_burned = jnp.where(soil_column, peat_burned, zero)

        cwd_indices = jnp.nonzero(
            parameters.decomposition_pool_is_cwd,
            size=parameters.decomposition_pool_is_cwd.shape[0],
            fill_value=0,
        )[0]
        cwd_index = jnp.max(cwd_indices)
        fuel = _add(
            forcing.total_litter_carbon_column,
            forcing.total_vegetation_carbon_column,
            -root_carbon,
            -_mul(crop_fuel, crop_fraction),
        )
        accelerated = jnp.asarray(parameters.spinup_state) == jnp.int32(2)
        fuel = jnp.where(
            accelerated,
            fortran_add(
                fuel,
                _mul(
                    jnp.float64(9.0),
                    deadstem_carbon,
                ),
            ),
            fuel,
        )
        spinup_latitude = _spinup_latitude_term(latitude)

        def cwd_step(level: int, current: jnp.ndarray) -> jnp.ndarray:
            contribution = _mul(
                forcing.decomposition_carbon_pool_column_level_pool[:, level, cwd_index],
                parameters.decomposition_layer_thickness[level],
            )
            spinup_contribution = _mul(
                contribution,
                parameters.decomposition_pool_spinup_factor[cwd_index],
                spinup_latitude,
            )
            selected = jnp.where(accelerated, spinup_contribution, contribution)
            return jnp.where(
                soil_column,
                fortran_add(current, selected),
                current,
            )

        fuel = lax.fori_loop(
            0,
            NUMBER_OF_FIRE_DECOMPOSITION_LEVELS,
            cwd_step,
            fuel,
        )
        fuel_eligible = soil_column & (crop_fraction < jnp.float64(1.0))
        fuel = jnp.where(
            fuel_eligible,
            fortran_divide(fuel, safe_noncrop_denominator),
            jnp.float64(0.0),
        )
        fuel_availability = _clamp01(
            fortran_divide(
                fortran_add(
                    fuel,
                    -jnp.asarray(parameters.lower_fuel_threshold),
                ),
                fortran_add(
                    jnp.asarray(parameters.upper_fuel_threshold),
                    -jnp.asarray(parameters.lower_fuel_threshold),
                ),
            )
        )
        fuel_availability = jnp.where(
            fuel_eligible,
            fuel_availability,
            jnp.float64(0.0),
        )

        grid_humidity = forcing.relative_humidity_gridcell[column_gridcell]
        grid_population = forcing.human_population_density_gridcell[column_gridcell]
        grid_latitude = topology.gridcell_latitude_degrees[column_gridcell]
        open_tropical_forest = _add(tropical_evergreen, tropical_deciduous) <= 0.6
        fuel_weight = _clamp01(
            fortran_divide(
                fortran_add(fuel, -jnp.float64(2500.0)),
                jnp.float64(2500.0),
            )
        )
        current_humidity_factor = fortran_add(
            jnp.float64(1.0),
            -_clamp01(
                fortran_divide(
                    fortran_add(
                        grid_humidity,
                        -jnp.asarray(parameters.relative_humidity_low),
                    ),
                    fortran_add(
                        jnp.asarray(parameters.relative_humidity_high),
                        -jnp.asarray(parameters.relative_humidity_low),
                    ),
                )
            ),
        )
        running_humidity_factor = fortran_add(
            jnp.float64(1.0),
            -jnp.maximum(
                jnp.float64(0.7),
                jnp.minimum(
                    jnp.float64(1.0),
                    fortran_divide(relative_humidity_30day, jnp.float64(90.0)),
                ),
            ),
        )
        humidity_factor = _add(
            _mul(fuel_weight, running_humidity_factor),
            _mul(
                fortran_add(jnp.float64(1.0), -fuel_weight),
                current_humidity_factor,
            ),
        )
        safe_noncrop_fraction = jnp.where(
            noncrop_fraction > jnp.float64(0.0),
            noncrop_fraction,
            jnp.float64(1.0),
        )
        wetness_factor = fortran_add(
            jnp.float64(1.0),
            -_clamp01(
                fortran_divide(
                    fortran_add(
                        fortran_divide(root_wetness, safe_noncrop_fraction),
                        -jnp.asarray(parameters.root_wetness_minimum),
                    ),
                    fortran_add(
                        jnp.asarray(parameters.root_wetness_maximum),
                        -jnp.asarray(parameters.root_wetness_minimum),
                    ),
                )
            ),
        )
        fire_possible = (
            (grid_humidity < jnp.asarray(parameters.relative_humidity_high))
            & (noncrop_fraction > jnp.float64(0.0))
            & (forcing.soil_temperature_17cm_column > jnp.float64(FREEZING_TEMPERATURE_FRESHWATER))
        )
        combustibility = _fire_combustibility(humidity_factor, wetness_factor, fire_possible)
        human_ignitions = fortran_divide(
            fortran_divide(
                _mul(
                    parameters.human_ignition_coefficient,
                    jnp.float64(6.8),
                    fortran_power(grid_population, jnp.float64(0.43)),
                ),
                jnp.float64(30.0),
            ),
            jnp.float64(24.0),
        )
        suppression = fortran_add(
            jnp.float64(1.0),
            -_add(
                jnp.float64(0.01),
                _mul(
                    jnp.float64(0.98),
                    strict_negative_exp(-_mul(jnp.float64(0.025), grid_population)),
                ),
            ),
        )
        latitude_angle = _mul(
            jnp.float64(3.0),
            jnp.minimum(jnp.float64(60.0), jnp.abs(grid_latitude)),
        )
        lightning_denominator = _add(
            jnp.float64(5.16),
            _mul(jnp.float64(2.16), _scalar_cos(latitude_angle)),
        )
        total_ignitions = _mul(
            _add(
                human_ignitions,
                _mul(
                    fortran_divide(
                        forcing.lightning_frequency_gridcell[column_gridcell],
                        lightning_denominator,
                    ),
                    jnp.float64(0.22),
                ),
            ),
            fortran_add(jnp.float64(1.0), -suppression),
            fortran_add(jnp.float64(1.0), -crop_fraction),
        )
        fire_count_candidate = _mul(
            fortran_divide(total_ignitions, jnp.float64(3600.0)),
            fuel_availability,
            combustibility,
            gdp_occurrence,
        )
        fire_count = jnp.where(
            soil_column & (crop_fraction < 1.0) & open_tropical_forest,
            fire_count_candidate,
            state.fire_count_column,
        )
        length_breadth = _add(
            jnp.float64(1.0),
            _mul(
                jnp.float64(10.0),
                fortran_add(
                    jnp.float64(1.0),
                    -strict_negative_exp(
                        -_mul(
                            jnp.float64(0.06),
                            forcing.wind_speed_gridcell[column_gridcell],
                        )
                    ),
                ),
            ),
        )
        natural_area = _natural_fire_area(
            humidity_factor, wetness_factor, fire_possible,
            jnp.asarray(parameters.base_spread_factor), fire_spread, fire_duration,
            gdp_spread, population_spread, total_ignitions, fuel_availability,
            gdp_occurrence, length_breadth,
        )
        open_area = jnp.minimum(
            jnp.float64(1.0),
            _add(natural_area, crop_burned, peat_burned),
        )
        closed_area = jnp.minimum(
            jnp.float64(1.0),
            fortran_add(crop_burned, peat_burned),
        )
        total_burned = jnp.where(open_tropical_forest, open_area, closed_area)
        total_burned = jnp.where(
            crop_fraction < jnp.float64(1.0),
            total_burned,
            closed_area,
        )

        updated_state = CnFireAreaState(
            burn_date_patch=burn_date,
            unburned_conversion_fraction_column=(state.unburned_conversion_fraction_column),
            crop_burned_area_fraction_column=jnp.where(
                soil_column,
                crop_burned,
                state.crop_burned_area_fraction_column,
            ),
            peat_burned_area_fraction_column=jnp.where(
                soil_column,
                peat_burned,
                state.peat_burned_area_fraction_column,
            ),
            conversion_burned_area_fraction_column=(state.conversion_burned_area_fraction_column),
            conversion_fire_burned_area_fraction_column=(
                state.conversion_fire_burned_area_fraction_column
            ),
            total_burned_area_fraction_column=jnp.where(
                soil_column,
                total_burned,
                state.total_burned_area_fraction_column,
            ),
            fire_count_column=fire_count,
            fire_spread_rate_column=jnp.where(
                soil_column,
                fire_spread,
                state.fire_spread_rate_column,
            ),
            fire_duration_column=jnp.where(
                soil_column,
                fire_duration,
                state.fire_duration_column,
            ),
            gdp_occurrence_limit_column=jnp.where(
                soil_column,
                gdp_occurrence,
                state.gdp_occurrence_limit_column,
            ),
            gdp_spread_limit_column=jnp.where(
                soil_column,
                gdp_spread,
                state.gdp_spread_limit_column,
            ),
            population_spread_limit_column=jnp.where(
                soil_column,
                population_spread,
                state.population_spread_limit_column,
            ),
            natural_vegetation_fraction_column=jnp.where(
                soil_column,
                natural_fraction,
                state.natural_vegetation_fraction_column,
            ),
            tropical_evergreen_tree_fraction_column=jnp.where(
                soil_column,
                tropical_evergreen,
                state.tropical_evergreen_tree_fraction_column,
            ),
            tropical_deciduous_tree_fraction_column=jnp.where(
                soil_column,
                tropical_deciduous,
                state.tropical_deciduous_tree_fraction_column,
            ),
            decreased_tropical_tree_fraction_column=(state.decreased_tropical_tree_fraction_column),
            crop_fraction_column=jnp.where(
                soil_column,
                crop_fraction,
                state.crop_fraction_column,
            ),
            noncrop_fraction_column=jnp.where(
                soil_column,
                noncrop_fraction,
                state.noncrop_fraction_column,
            ),
            root_carbon_column=jnp.where(
                soil_column,
                root_carbon,
                state.root_carbon_column,
            ),
            leaf_carbon_column=jnp.where(
                soil_column,
                leaf_carbon,
                state.leaf_carbon_column,
            ),
            deadstem_carbon_column=jnp.where(
                soil_column,
                deadstem_carbon,
                state.deadstem_carbon_column,
            ),
            fuel_carbon_column=jnp.where(
                soil_column & (crop_fraction < 1.0),
                fuel,
                state.fuel_carbon_column,
            ),
            crop_fuel_carbon_column=jnp.where(
                soil_column,
                crop_fuel,
                state.crop_fuel_carbon_column,
            ),
        )
        return CnFireAreaResult(
            state=updated_state,
            diagnostics=CnFireAreaDiagnostics(
                precipitation_60day_column=precipitation_60day,
                precipitation_10day_column=precipitation_10day,
                relative_humidity_30day_column=relative_humidity_30day,
                root_zone_wetness_column=root_wetness,
                fuel_availability_column=fuel_availability,
                fire_combustibility_column=combustibility,
                first_step=jnp.bool_(False),
                updated_column_count=jnp.sum(soil_column, dtype=jnp.int32),
            ),
        )

    return lax.cond(
        jnp.asarray(parameters.model_step) == jnp.int32(0),
        first_step_result,
        regular_result,
        operand=None,
    )


_scoped_fire_area = gpu_convergence_scope(_cn_fire_area_li2016_fixed_landcover)


def cn_fire_area_li2016_fixed_landcover(
    topology: CnFireAreaTopology,
    forcing: CnFireAreaForcing,
    vegetation_state: CnPrecisionControlState,
    state: CnFireAreaState,
    parameters: CnFireAreaParameters,
) -> CnFireAreaResult:
    return _scoped_fire_area(topology, forcing, vegetation_state, state, parameters)


__all__ = [
    "MAXIMUM_PATCHES_PER_FIRE_COLUMN",
    "NUMBER_OF_FIRE_DECOMPOSITION_LEVELS",
    "CnFireAreaDiagnostics",
    "CnFireAreaForcing",
    "CnFireAreaParameters",
    "CnFireAreaResult",
    "CnFireAreaState",
    "CnFireAreaTopology",
    "cn_fire_area_li2016_fixed_landcover",
]
