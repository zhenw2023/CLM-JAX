
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.restart import (
    COLUMN_URBAN_IMPERVIOUS_ROAD,
    COLUMN_URBAN_PERVIOUS_ROAD,
    COLUMN_URBAN_ROOF,
    COLUMN_URBAN_SHADE_WALL,
    COLUMN_URBAN_SUN_WALL,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    SubgridTopology,
)
from clm_jax.physics.numerics import (
    fortran_linear_combination,
    fortran_ordered_sum,
    fortran_product,
)

STEFAN_BOLTZMANN = 5.67e-8
SNOW_EMISSIVITY = 0.97
MISSING_URBAN_SURFACE_TEMPERATURE = 292.15
LONGWAVE_CONVERGENCE = 0.001
MAXIMUM_LONGWAVE_ITERATIONS = 50


class UrbanRadiationInputs(NamedTuple):

    direct_shortwave_gridcell: jnp.ndarray
    diffuse_shortwave_gridcell: jnp.ndarray
    downward_longwave_gridcell: jnp.ndarray
    snow_fraction_column: jnp.ndarray
    ground_temperature_column: jnp.ndarray
    canyon_height_width_ratio_landunit: jnp.ndarray
    pervious_road_fraction_landunit: jnp.ndarray
    roof_emissivity_landunit: jnp.ndarray
    impervious_road_emissivity_landunit: jnp.ndarray
    pervious_road_emissivity_landunit: jnp.ndarray
    wall_emissivity_landunit: jnp.ndarray


class UrbanShortwaveAbsorptionState(NamedTuple):

    roof_direct_landunit: jnp.ndarray
    roof_diffuse_landunit: jnp.ndarray
    sun_wall_direct_landunit: jnp.ndarray
    sun_wall_diffuse_landunit: jnp.ndarray
    shade_wall_direct_landunit: jnp.ndarray
    shade_wall_diffuse_landunit: jnp.ndarray
    impervious_road_direct_landunit: jnp.ndarray
    impervious_road_diffuse_landunit: jnp.ndarray
    pervious_road_direct_landunit: jnp.ndarray
    pervious_road_diffuse_landunit: jnp.ndarray


class UrbanRadiationFluxState(NamedTuple):

    absorbed_ground_patch: jnp.ndarray
    absorbed_vegetation_patch: jnp.ndarray
    absorbed_total_patch: jnp.ndarray
    urban_absorbed_total_patch: jnp.ndarray
    emitted_longwave_patch: jnp.ndarray
    net_longwave_patch: jnp.ndarray
    urban_net_longwave_patch: jnp.ndarray


class UrbanRadiationState(NamedTuple):
    shortwave_absorption: UrbanShortwaveAbsorptionState
    fluxes: UrbanRadiationFluxState


class UrbanLongwaveDiagnostics(NamedTuple):

    net_landunit: jnp.ndarray
    upward_landunit: jnp.ndarray
    converged_landunit: jnp.ndarray
    iterations_landunit: jnp.ndarray


class UrbanRadiationResult(NamedTuple):
    state: UrbanRadiationState
    longwave: UrbanLongwaveDiagnostics


class _CanyonIterationState(NamedTuple):
    road_reflected_sun_wall: jnp.ndarray
    road_reflected_shade_wall: jnp.ndarray
    road_emitted_sun_wall: jnp.ndarray
    road_emitted_shade_wall: jnp.ndarray
    sun_wall_reflected_road: jnp.ndarray
    sun_wall_reflected_shade_wall: jnp.ndarray
    sun_wall_emitted_road: jnp.ndarray
    sun_wall_emitted_shade_wall: jnp.ndarray
    shade_wall_reflected_road: jnp.ndarray
    shade_wall_reflected_sun_wall: jnp.ndarray
    shade_wall_emitted_road: jnp.ndarray
    shade_wall_emitted_sun_wall: jnp.ndarray
    net_impervious_road: jnp.ndarray
    net_pervious_road: jnp.ndarray
    net_sun_wall: jnp.ndarray
    net_shade_wall: jnp.ndarray
    upward_impervious_road: jnp.ndarray
    upward_pervious_road: jnp.ndarray
    upward_sun_wall: jnp.ndarray
    upward_shade_wall: jnp.ndarray
    done: jnp.ndarray
    iterations: jnp.ndarray


def _add(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    left, right = jnp.broadcast_arrays(left, right)
    terms = jnp.stack((left, right), axis=0)
    return fortran_linear_combination(
        terms,
        jnp.ones_like(terms),
    )


def _subtract(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return _add(left, -right)


def _multiply(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return lax.optimization_barrier(fortran_product(left, right))


def _fourth_power(value: jnp.ndarray) -> jnp.ndarray:
    squared = _multiply(value, value)
    return _multiply(squared, squared)


def _urban_mask(landunit_type: jnp.ndarray) -> jnp.ndarray:
    return (landunit_type >= jnp.int32(LANDUNIT_URBAN_MIN)) & (
        landunit_type <= jnp.int32(LANDUNIT_URBAN_MAX)
    )


def urban_view_factors(
    canyon_height_width_ratio: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    ratio = jnp.asarray(canyon_height_width_ratio, dtype=jnp.float64)
    root = jnp.sqrt(_add(_multiply(ratio, ratio), jnp.float64(1.0)))
    sky_from_road = _subtract(root, ratio)
    wall_from_road = _multiply(
        jnp.float64(0.5), _subtract(jnp.float64(1.0), sky_from_road)
    )
    wall_numerator = _subtract(_add(ratio, jnp.float64(1.0)), root)
    sky_from_wall = _multiply(jnp.float64(0.5), wall_numerator) / ratio
    road_from_wall = sky_from_wall
    opposing_wall = _subtract(
        _subtract(jnp.float64(1.0), sky_from_wall), road_from_wall
    )
    return (
        sky_from_road,
        wall_from_road,
        sky_from_wall,
        road_from_wall,
        opposing_wall,
    )


def _cast_and_validate(
    topology: SubgridTopology,
    inputs: UrbanRadiationInputs,
    state: UrbanRadiationState,
) -> tuple[UrbanRadiationInputs, UrbanRadiationState]:
    number_of_gridcells = topology.grid.longitude_degrees.shape[0]
    number_of_landunits = topology.landunit.gridcell.shape[0]
    number_of_columns = topology.column.gridcell.shape[0]
    number_of_patches = topology.patch.gridcell.shape[0]

    arrays = {
        field: jnp.asarray(getattr(inputs, field), dtype=jnp.float64)
        for field in UrbanRadiationInputs._fields
    }
    for name in ("direct_shortwave_gridcell", "diffuse_shortwave_gridcell"):
        if arrays[name].shape != (number_of_gridcells, 2):
            raise ValueError(f"{name} must have shape (gridcell, 2)")
    if arrays["downward_longwave_gridcell"].shape != (number_of_gridcells,):
        raise ValueError("downward longwave forcing must contain one value per gridcell")
    for name in ("snow_fraction_column", "ground_temperature_column"):
        if arrays[name].shape != (number_of_columns,):
            raise ValueError(f"{name} must contain one value per column")
    for name in UrbanRadiationInputs._fields[5:]:
        if arrays[name].shape != (number_of_landunits,):
            raise ValueError(f"{name} must contain one value per landunit")

    coefficient_arrays = {
        field: jnp.asarray(getattr(state.shortwave_absorption, field), dtype=jnp.float64)
        for field in UrbanShortwaveAbsorptionState._fields
    }
    for name, value in coefficient_arrays.items():
        if value.shape != (number_of_landunits, 2):
            raise ValueError(f"urban absorption coefficient {name} has shape {value.shape}")

    flux_arrays = {
        field: jnp.asarray(getattr(state.fluxes, field), dtype=jnp.float64)
        for field in UrbanRadiationFluxState._fields
    }
    for name, value in flux_arrays.items():
        if value.shape != (number_of_patches,):
            raise ValueError(f"urban radiation flux {name} has shape {value.shape}")

    return (
        UrbanRadiationInputs(**arrays),
        UrbanRadiationState(
            shortwave_absorption=UrbanShortwaveAbsorptionState(**coefficient_arrays),
            fluxes=UrbanRadiationFluxState(**flux_arrays),
        ),
    )


def _column_surface_value(
    topology: SubgridTopology,
    values: jnp.ndarray,
    column_type: int,
    default: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    number_of_landunits = topology.landunit.gridcell.shape[0]
    landunit = jnp.asarray(topology.column.landunit, dtype=jnp.int32)
    safe_landunit = jnp.clip(landunit, 0, max(number_of_landunits - 1, 0))
    landunit_type = jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
    selected = (
        jnp.asarray(topology.column.active, dtype=jnp.bool_)
        & _urban_mask(landunit_type)
        & (jnp.asarray(topology.column.column_type, dtype=jnp.int32) == column_type)
    )
    selected_value = jnp.where(selected, values, jnp.float64(0.0))
    mapped = jnp.zeros((number_of_landunits,), dtype=jnp.float64).at[safe_landunit].add(
        selected_value
    )
    count = jnp.zeros((number_of_landunits,), dtype=jnp.int32).at[safe_landunit].add(
        selected.astype(jnp.int32)
    )
    exists = count > jnp.int32(0)
    return jnp.where(exists, mapped, jnp.float64(default)), exists


def _snow_adjusted_emissivity(
    emissivity: jnp.ndarray,
    snow_fraction: jnp.ndarray,
    surface_exists: jnp.ndarray,
) -> jnp.ndarray:
    snow_candidate = _add(
        _multiply(emissivity, _subtract(jnp.float64(1.0), snow_fraction)),
        _multiply(jnp.float64(SNOW_EMISSIVITY), snow_fraction),
    )
    return jnp.where(surface_exists, snow_candidate, emissivity)


def _initial_longwave_state(
    ratio: jnp.ndarray,
    pervious_fraction: jnp.ndarray,
    longwave_down: jnp.ndarray,
    roof_emissivity: jnp.ndarray,
    impervious_emissivity: jnp.ndarray,
    pervious_emissivity: jnp.ndarray,
    wall_emissivity: jnp.ndarray,
    roof_temperature: jnp.ndarray,
    impervious_temperature: jnp.ndarray,
    pervious_temperature: jnp.ndarray,
    sun_wall_temperature: jnp.ndarray,
    shade_wall_temperature: jnp.ndarray,
) -> tuple[_CanyonIterationState, tuple[jnp.ndarray, ...]]:
    vf_sr, vf_wr, vf_sw, vf_rw, vf_ww = urban_view_factors(ratio)
    impervious_fraction = _subtract(jnp.float64(1.0), pervious_fraction)

    longwave_down_road = _multiply(longwave_down, vf_sr)
    longwave_down_sun_wall = _multiply(longwave_down, vf_sw)
    longwave_down_shade_wall = _multiply(longwave_down, vf_sw)

    impervious_absorbed = _multiply(impervious_emissivity, longwave_down_road)
    impervious_reflected = _multiply(
        _subtract(jnp.float64(1.0), impervious_emissivity), longwave_down_road
    )
    impervious_reflected_sky = _multiply(impervious_reflected, vf_sr)
    impervious_emitted = _multiply(
        _multiply(impervious_emissivity, jnp.float64(STEFAN_BOLTZMANN)),
        _fourth_power(impervious_temperature),
    )
    impervious_emitted_sky = _multiply(impervious_emitted, vf_sr)

    pervious_absorbed = _multiply(pervious_emissivity, longwave_down_road)
    pervious_reflected = _multiply(
        _subtract(jnp.float64(1.0), pervious_emissivity), longwave_down_road
    )
    pervious_reflected_sky = _multiply(pervious_reflected, vf_sr)
    pervious_emitted = _multiply(
        _multiply(pervious_emissivity, jnp.float64(STEFAN_BOLTZMANN)),
        _fourth_power(pervious_temperature),
    )
    pervious_emitted_sky = _multiply(pervious_emitted, vf_sr)

    road_reflected = _add(
        _multiply(impervious_reflected, impervious_fraction),
        _multiply(pervious_reflected, pervious_fraction),
    )
    road_emitted = _add(
        _multiply(impervious_emitted, impervious_fraction),
        _multiply(pervious_emitted, pervious_fraction),
    )

    sun_wall_absorbed = _multiply(wall_emissivity, longwave_down_sun_wall)
    sun_wall_reflected = _multiply(
        _subtract(jnp.float64(1.0), wall_emissivity), longwave_down_sun_wall
    )
    sun_wall_emitted = _multiply(
        _multiply(wall_emissivity, jnp.float64(STEFAN_BOLTZMANN)),
        _fourth_power(sun_wall_temperature),
    )

    shade_wall_absorbed = _multiply(wall_emissivity, longwave_down_shade_wall)
    shade_wall_reflected = _multiply(
        _subtract(jnp.float64(1.0), wall_emissivity), longwave_down_shade_wall
    )
    shade_wall_emitted = _multiply(
        _multiply(wall_emissivity, jnp.float64(STEFAN_BOLTZMANN)),
        _fourth_power(shade_wall_temperature),
    )

    state = _CanyonIterationState(
        road_reflected_sun_wall=_multiply(road_reflected, vf_wr),
        road_reflected_shade_wall=_multiply(road_reflected, vf_wr),
        road_emitted_sun_wall=_multiply(road_emitted, vf_wr),
        road_emitted_shade_wall=_multiply(road_emitted, vf_wr),
        sun_wall_reflected_road=_multiply(sun_wall_reflected, vf_rw),
        sun_wall_reflected_shade_wall=_multiply(sun_wall_reflected, vf_ww),
        sun_wall_emitted_road=_multiply(sun_wall_emitted, vf_rw),
        sun_wall_emitted_shade_wall=_multiply(sun_wall_emitted, vf_ww),
        shade_wall_reflected_road=_multiply(shade_wall_reflected, vf_rw),
        shade_wall_reflected_sun_wall=_multiply(shade_wall_reflected, vf_ww),
        shade_wall_emitted_road=_multiply(shade_wall_emitted, vf_rw),
        shade_wall_emitted_sun_wall=_multiply(shade_wall_emitted, vf_ww),
        net_impervious_road=_subtract(impervious_emitted, impervious_absorbed),
        net_pervious_road=_subtract(pervious_emitted, pervious_absorbed),
        net_sun_wall=_subtract(sun_wall_emitted, sun_wall_absorbed),
        net_shade_wall=_subtract(shade_wall_emitted, shade_wall_absorbed),
        upward_impervious_road=_add(
            impervious_reflected_sky, impervious_emitted_sky
        ),
        upward_pervious_road=_add(pervious_reflected_sky, pervious_emitted_sky),
        upward_sun_wall=_add(
            _multiply(sun_wall_reflected, vf_sw),
            _multiply(sun_wall_emitted, vf_sw),
        ),
        upward_shade_wall=_add(
            _multiply(shade_wall_reflected, vf_sw),
            _multiply(shade_wall_emitted, vf_sw),
        ),
        done=jnp.zeros(ratio.shape, dtype=jnp.bool_),
        iterations=jnp.zeros(ratio.shape, dtype=jnp.int32),
    )
    constants = (
        vf_sr,
        vf_wr,
        vf_sw,
        vf_rw,
        vf_ww,
        impervious_fraction,
        impervious_emissivity,
        pervious_emissivity,
        wall_emissivity,
    )
    return state, constants


def _iterate_canyon_longwave(*arguments):
    if jax.default_backend() == "gpu":
        from clm_jax.physics.urban_reflections_gpu import longwave_reflections

        return longwave_reflections(*arguments)
    return _iterate_canyon_longwave_reference(*arguments)


def _iterate_canyon_longwave_reference(
    initial: _CanyonIterationState,
    constants: tuple[jnp.ndarray, ...],
    ratio: jnp.ndarray,
    pervious_fraction: jnp.ndarray,
) -> _CanyonIterationState:
    (
        vf_sr,
        vf_wr,
        vf_sw,
        vf_rw,
        vf_ww,
        impervious_fraction,
        impervious_emissivity,
        pervious_emissivity,
        wall_emissivity,
    ) = constants

    def iteration(previous: _CanyonIterationState, _: None):
        wall_to_road = fortran_ordered_sum(
            jnp.stack(
                (
                    previous.sun_wall_reflected_road,
                    previous.sun_wall_emitted_road,
                    previous.shade_wall_reflected_road,
                    previous.shade_wall_emitted_road,
                ),
                axis=0,
            )
        )
        road_incoming = _multiply(wall_to_road, ratio)
        impervious_reflected = _multiply(
            _subtract(jnp.float64(1.0), impervious_emissivity), road_incoming
        )
        impervious_absorbed = _multiply(impervious_emissivity, road_incoming)
        pervious_reflected = _multiply(
            _subtract(jnp.float64(1.0), pervious_emissivity), road_incoming
        )
        pervious_absorbed = _multiply(pervious_emissivity, road_incoming)
        road_absorbed = _add(
            _multiply(impervious_absorbed, impervious_fraction),
            _multiply(pervious_absorbed, pervious_fraction),
        )
        road_reflected = _add(
            _multiply(impervious_reflected, impervious_fraction),
            _multiply(pervious_reflected, pervious_fraction),
        )

        sun_wall_incoming = _add(
            _add(
                previous.road_reflected_sun_wall,
                previous.road_emitted_sun_wall,
            )
            / ratio,
            _add(
                previous.shade_wall_reflected_sun_wall,
                previous.shade_wall_emitted_sun_wall,
            ),
        )
        sun_wall_absorbed = _multiply(wall_emissivity, sun_wall_incoming)
        sun_wall_reflected = _multiply(
            _subtract(jnp.float64(1.0), wall_emissivity), sun_wall_incoming
        )

        shade_wall_incoming = _add(
            _add(
                previous.road_reflected_shade_wall,
                previous.road_emitted_shade_wall,
            )
            / ratio,
            _add(
                previous.sun_wall_reflected_shade_wall,
                previous.sun_wall_emitted_shade_wall,
            ),
        )
        shade_wall_absorbed = _multiply(wall_emissivity, shade_wall_incoming)
        shade_wall_reflected = _multiply(
            _subtract(jnp.float64(1.0), wall_emissivity), shade_wall_incoming
        )

        net_impervious = _subtract(
            previous.net_impervious_road, impervious_absorbed
        )
        net_pervious = _subtract(previous.net_pervious_road, pervious_absorbed)
        net_sun = _subtract(previous.net_sun_wall, sun_wall_absorbed)
        net_shade = _subtract(previous.net_shade_wall, shade_wall_absorbed)

        impervious_reflected_sky = _multiply(impervious_reflected, vf_sr)
        pervious_reflected_sky = _multiply(pervious_reflected, vf_sr)
        road_reflected_sun = _multiply(road_reflected, vf_wr)
        road_reflected_shade = _multiply(road_reflected, vf_wr)
        sun_reflected_sky = _multiply(sun_wall_reflected, vf_sw)
        shade_reflected_sky = _multiply(shade_wall_reflected, vf_sw)

        upward_impervious = _add(
            previous.upward_impervious_road, impervious_reflected_sky
        )
        upward_pervious = _add(
            previous.upward_pervious_road, pervious_reflected_sky
        )
        upward_sun = _add(previous.upward_sun_wall, sun_reflected_sky)
        upward_shade = _add(previous.upward_shade_wall, shade_reflected_sky)

        criterion = jnp.maximum(
            jnp.maximum(road_absorbed, sun_wall_absorbed), shade_wall_absorbed
        )
        newly_done = criterion < jnp.float64(LONGWAVE_CONVERGENCE)
        active = ~previous.done
        zero = jnp.zeros_like(road_reflected)
        candidate = _CanyonIterationState(
            road_reflected_sun_wall=road_reflected_sun,
            road_reflected_shade_wall=road_reflected_shade,
            road_emitted_sun_wall=zero,
            road_emitted_shade_wall=zero,
            sun_wall_reflected_road=_multiply(sun_wall_reflected, vf_rw),
            sun_wall_reflected_shade_wall=_multiply(sun_wall_reflected, vf_ww),
            sun_wall_emitted_road=zero,
            sun_wall_emitted_shade_wall=zero,
            shade_wall_reflected_road=_multiply(shade_wall_reflected, vf_rw),
            shade_wall_reflected_sun_wall=_multiply(shade_wall_reflected, vf_ww),
            shade_wall_emitted_road=zero,
            shade_wall_emitted_sun_wall=zero,
            net_impervious_road=net_impervious,
            net_pervious_road=net_pervious,
            net_sun_wall=net_sun,
            net_shade_wall=net_shade,
            upward_impervious_road=upward_impervious,
            upward_pervious_road=upward_pervious,
            upward_sun_wall=upward_sun,
            upward_shade_wall=upward_shade,
            done=previous.done | newly_done,
            iterations=previous.iterations + active.astype(jnp.int32),
        )
        updated = _CanyonIterationState(
            *(jnp.where(active, new, old) for new, old in zip(candidate, previous, strict=True))
        )
        return updated, None

    result, _ = lax.scan(
        iteration,
        initial,
        xs=None,
        length=MAXIMUM_LONGWAVE_ITERATIONS,
    )
    return result


def urban_radiation(
    topology: SubgridTopology,
    inputs: UrbanRadiationInputs,
    state: UrbanRadiationState,
) -> UrbanRadiationResult:

    inputs, state = _cast_and_validate(topology, inputs, state)
    landunit_type = jnp.asarray(topology.landunit.landunit_type, dtype=jnp.int32)
    active_landunit = jnp.asarray(topology.landunit.active, dtype=jnp.bool_)
    urban_landunit = active_landunit & _urban_mask(landunit_type)
    nonurban_landunit = active_landunit & ~_urban_mask(landunit_type)

    ratio = jnp.where(
        urban_landunit,
        inputs.canyon_height_width_ratio_landunit,
        jnp.float64(1.0),
    )
    pervious_fraction = jnp.where(
        urban_landunit,
        inputs.pervious_road_fraction_landunit,
        jnp.float64(0.5),
    )
    roof_emissivity = jnp.where(
        urban_landunit, inputs.roof_emissivity_landunit, jnp.float64(0.9)
    )
    impervious_emissivity = jnp.where(
        urban_landunit,
        inputs.impervious_road_emissivity_landunit,
        jnp.float64(0.9),
    )
    pervious_emissivity = jnp.where(
        urban_landunit,
        inputs.pervious_road_emissivity_landunit,
        jnp.float64(0.9),
    )
    wall_emissivity = jnp.where(
        urban_landunit, inputs.wall_emissivity_landunit, jnp.float64(0.9)
    )

    default_temperature = MISSING_URBAN_SURFACE_TEMPERATURE
    roof_temperature, roof_exists = _column_surface_value(
        topology,
        inputs.ground_temperature_column,
        COLUMN_URBAN_ROOF,
        default_temperature,
    )
    impervious_temperature, impervious_exists = _column_surface_value(
        topology,
        inputs.ground_temperature_column,
        COLUMN_URBAN_IMPERVIOUS_ROAD,
        default_temperature,
    )
    pervious_temperature, pervious_exists = _column_surface_value(
        topology,
        inputs.ground_temperature_column,
        COLUMN_URBAN_PERVIOUS_ROAD,
        default_temperature,
    )
    sun_wall_temperature, _ = _column_surface_value(
        topology,
        inputs.ground_temperature_column,
        COLUMN_URBAN_SUN_WALL,
        default_temperature,
    )
    shade_wall_temperature, _ = _column_surface_value(
        topology,
        inputs.ground_temperature_column,
        COLUMN_URBAN_SHADE_WALL,
        default_temperature,
    )

    roof_snow, _ = _column_surface_value(
        topology, inputs.snow_fraction_column, COLUMN_URBAN_ROOF, 0.0
    )
    impervious_snow, _ = _column_surface_value(
        topology,
        inputs.snow_fraction_column,
        COLUMN_URBAN_IMPERVIOUS_ROAD,
        0.0,
    )
    pervious_snow, _ = _column_surface_value(
        topology,
        inputs.snow_fraction_column,
        COLUMN_URBAN_PERVIOUS_ROAD,
        0.0,
    )
    roof_emissivity = _snow_adjusted_emissivity(
        roof_emissivity, roof_snow, roof_exists
    )
    impervious_emissivity = _snow_adjusted_emissivity(
        impervious_emissivity, impervious_snow, impervious_exists
    )
    pervious_emissivity = _snow_adjusted_emissivity(
        pervious_emissivity, pervious_snow, pervious_exists
    )

    landunit_gridcell = jnp.asarray(topology.landunit.gridcell, dtype=jnp.int32)
    safe_gridcell = jnp.clip(
        landunit_gridcell, 0, max(inputs.downward_longwave_gridcell.shape[0] - 1, 0)
    )
    longwave_down = jnp.where(
        urban_landunit,
        inputs.downward_longwave_gridcell[safe_gridcell],
        jnp.float64(300.0),
    )

    initial_longwave, constants = _initial_longwave_state(
        ratio,
        pervious_fraction,
        longwave_down,
        roof_emissivity,
        impervious_emissivity,
        pervious_emissivity,
        wall_emissivity,
        roof_temperature,
        impervious_temperature,
        pervious_temperature,
        sun_wall_temperature,
        shade_wall_temperature,
    )
    canyon = _iterate_canyon_longwave(
        initial_longwave, constants, ratio, pervious_fraction
    )
    impervious_fraction = _subtract(jnp.float64(1.0), pervious_fraction)
    net_canyon = fortran_ordered_sum(
        jnp.stack(
            (
                _multiply(canyon.net_impervious_road, impervious_fraction),
                _multiply(canyon.net_pervious_road, pervious_fraction),
                _multiply(_add(canyon.net_sun_wall, canyon.net_shade_wall), ratio),
            ),
            axis=0,
        ),
        initial=jnp.zeros_like(ratio),
    )
    upward_canyon = fortran_ordered_sum(
        jnp.stack(
            (
                _multiply(canyon.upward_impervious_road, impervious_fraction),
                _multiply(canyon.upward_pervious_road, pervious_fraction),
                _multiply(
                    _add(canyon.upward_sun_wall, canyon.upward_shade_wall), ratio
                ),
            ),
            axis=0,
        ),
        initial=jnp.zeros_like(ratio),
    )
    upward_roof = _add(
        _multiply(
            _multiply(roof_emissivity, jnp.float64(STEFAN_BOLTZMANN)),
            _fourth_power(roof_temperature),
        ),
        _multiply(_subtract(jnp.float64(1.0), roof_emissivity), longwave_down),
    )
    net_roof = _subtract(upward_roof, longwave_down)

    special_value = jnp.float64(CLM_SPECIAL_VALUE)
    longwave_net = jnp.stack(
        (
            net_roof,
            canyon.net_impervious_road,
            canyon.net_pervious_road,
            canyon.net_sun_wall,
            canyon.net_shade_wall,
            net_canyon,
        ),
        axis=1,
    )
    longwave_upward = jnp.stack(
        (
            upward_roof,
            canyon.upward_impervious_road,
            canyon.upward_pervious_road,
            canyon.upward_sun_wall,
            canyon.upward_shade_wall,
            upward_canyon,
        ),
        axis=1,
    )
    longwave_net = jnp.where(urban_landunit[:, None], longwave_net, special_value)
    longwave_upward = jnp.where(
        urban_landunit[:, None], longwave_upward, special_value
    )

    coefficients = UrbanShortwaveAbsorptionState(
        *(
            jnp.where(nonurban_landunit[:, None], special_value, field)
            for field in state.shortwave_absorption
        )
    )

    patch_landunit = jnp.asarray(topology.patch.landunit, dtype=jnp.int32)
    patch_gridcell = jnp.asarray(topology.patch.gridcell, dtype=jnp.int32)
    patch_column_type = jnp.asarray(topology.patch.column_type, dtype=jnp.int32)
    active_urban_patch = (
        jnp.asarray(topology.patch.active, dtype=jnp.bool_)
        & _urban_mask(jnp.asarray(topology.patch.landunit_type, dtype=jnp.int32))
    )
    safe_landunit = jnp.clip(patch_landunit, 0, max(landunit_type.shape[0] - 1, 0))
    safe_patch_gridcell = jnp.clip(
        patch_gridcell, 0, max(inputs.direct_shortwave_gridcell.shape[0] - 1, 0)
    )

    zero_coefficient = jnp.zeros((patch_landunit.shape[0], 2), dtype=jnp.float64)
    direct_coefficient = zero_coefficient
    diffuse_coefficient = zero_coefficient
    surface_fields = (
        (
            COLUMN_URBAN_ROOF,
            coefficients.roof_direct_landunit,
            coefficients.roof_diffuse_landunit,
        ),
        (
            COLUMN_URBAN_SUN_WALL,
            coefficients.sun_wall_direct_landunit,
            coefficients.sun_wall_diffuse_landunit,
        ),
        (
            COLUMN_URBAN_SHADE_WALL,
            coefficients.shade_wall_direct_landunit,
            coefficients.shade_wall_diffuse_landunit,
        ),
        (
            COLUMN_URBAN_PERVIOUS_ROAD,
            coefficients.pervious_road_direct_landunit,
            coefficients.pervious_road_diffuse_landunit,
        ),
        (
            COLUMN_URBAN_IMPERVIOUS_ROAD,
            coefficients.impervious_road_direct_landunit,
            coefficients.impervious_road_diffuse_landunit,
        ),
    )
    for column_type, direct_field, diffuse_field in surface_fields:
        selected = patch_column_type == jnp.int32(column_type)
        direct_coefficient = jnp.where(
            selected[:, None], direct_field[safe_landunit], direct_coefficient
        )
        diffuse_coefficient = jnp.where(
            selected[:, None], diffuse_field[safe_landunit], diffuse_coefficient
        )

    direct_forcing = inputs.direct_shortwave_gridcell[safe_patch_gridcell]
    diffuse_forcing = inputs.diffuse_shortwave_gridcell[safe_patch_gridcell]
    absorbed_ground = fortran_linear_combination(
        jnp.stack(
            (
                direct_coefficient[:, 0],
                diffuse_coefficient[:, 0],
                direct_coefficient[:, 1],
                diffuse_coefficient[:, 1],
            ),
            axis=0,
        ),
        jnp.stack(
            (
                direct_forcing[:, 0],
                diffuse_forcing[:, 0],
                direct_forcing[:, 1],
                diffuse_forcing[:, 1],
            ),
            axis=0,
        ),
    )

    net_by_surface = longwave_net[safe_landunit]
    upward_by_surface = longwave_upward[safe_landunit]
    selected_net = jnp.zeros_like(absorbed_ground)
    selected_upward = jnp.zeros_like(absorbed_ground)
    surface_slot = {
        COLUMN_URBAN_ROOF: 0,
        COLUMN_URBAN_IMPERVIOUS_ROAD: 1,
        COLUMN_URBAN_PERVIOUS_ROAD: 2,
        COLUMN_URBAN_SUN_WALL: 3,
        COLUMN_URBAN_SHADE_WALL: 4,
    }
    for column_type, slot in surface_slot.items():
        selected = patch_column_type == jnp.int32(column_type)
        selected_net = jnp.where(selected, net_by_surface[:, slot], selected_net)
        selected_upward = jnp.where(
            selected, upward_by_surface[:, slot], selected_upward
        )

    zero_patch = jnp.zeros_like(absorbed_ground)
    absorbed_total = _add(zero_patch, absorbed_ground)
    previous = state.fluxes
    fluxes = UrbanRadiationFluxState(
        absorbed_ground_patch=jnp.where(
            active_urban_patch, absorbed_ground, previous.absorbed_ground_patch
        ),
        absorbed_vegetation_patch=jnp.where(
            active_urban_patch, zero_patch, previous.absorbed_vegetation_patch
        ),
        absorbed_total_patch=jnp.where(
            active_urban_patch, absorbed_total, previous.absorbed_total_patch
        ),
        urban_absorbed_total_patch=jnp.where(
            active_urban_patch, absorbed_total, previous.urban_absorbed_total_patch
        ),
        emitted_longwave_patch=jnp.where(
            active_urban_patch, selected_upward, previous.emitted_longwave_patch
        ),
        net_longwave_patch=jnp.where(
            active_urban_patch, selected_net, previous.net_longwave_patch
        ),
        urban_net_longwave_patch=jnp.where(
            active_urban_patch, selected_net, previous.urban_net_longwave_patch
        ),
    )
    return UrbanRadiationResult(
        state=UrbanRadiationState(
            shortwave_absorption=coefficients,
            fluxes=fluxes,
        ),
        longwave=UrbanLongwaveDiagnostics(
            net_landunit=longwave_net,
            upward_landunit=longwave_upward,
            converged_landunit=jnp.where(
                urban_landunit,
                canyon.done
                & (canyon.iterations < jnp.int32(MAXIMUM_LONGWAVE_ITERATIONS)),
                False,
            ),
            iterations_landunit=jnp.where(
                urban_landunit, canyon.iterations, jnp.int32(0)
            ),
        ),
    )
