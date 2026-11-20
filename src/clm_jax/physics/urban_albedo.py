
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.constants import PI
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
from clm_jax.physics.native_solar_math import native_cos
from clm_jax.physics.native_tangent import native_tan
from clm_jax.physics.numerics import (
    fortran_divide,
    strict_acos,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.urban_radiation import (
    UrbanShortwaveAbsorptionState,
    urban_view_factors,
)

URBAN_SNOW_VISIBLE_ALBEDO = 0.66
URBAN_SNOW_NEAR_INFRARED_ALBEDO = 0.56
SOLAR_REFLECTION_CONVERGENCE = 1.0e-5
MAXIMUM_SOLAR_REFLECTIONS = 50
_MINIMUM_CANYON_TANGENT = float.fromhex("0x1.0c6f7a0b5f3b3p-20")


class UrbanAlbedoInputs(NamedTuple):

    cosine_solar_zenith_column: jnp.ndarray
    snow_fraction_column: jnp.ndarray
    snow_water_column: jnp.ndarray
    canyon_height_width_ratio_landunit: jnp.ndarray
    pervious_road_fraction_landunit: jnp.ndarray
    roof_direct_albedo_landunit: jnp.ndarray
    roof_diffuse_albedo_landunit: jnp.ndarray
    impervious_road_direct_albedo_landunit: jnp.ndarray
    impervious_road_diffuse_albedo_landunit: jnp.ndarray
    pervious_road_direct_albedo_landunit: jnp.ndarray
    pervious_road_diffuse_albedo_landunit: jnp.ndarray
    wall_direct_albedo_landunit: jnp.ndarray
    wall_diffuse_albedo_landunit: jnp.ndarray


class UrbanCanopyAlbedoState(NamedTuple):

    absorbed_direct_patch: jnp.ndarray
    absorbed_sunlit_direct_patch: jnp.ndarray
    absorbed_shaded_direct_patch: jnp.ndarray
    absorbed_diffuse_patch: jnp.ndarray
    absorbed_sunlit_diffuse_patch: jnp.ndarray
    absorbed_shaded_diffuse_patch: jnp.ndarray
    transmitted_direct_patch: jnp.ndarray
    transmitted_diffuse_from_direct_patch: jnp.ndarray
    transmitted_diffuse_patch: jnp.ndarray


class UrbanSurfaceAlbedoState(NamedTuple):

    ground_direct_column: jnp.ndarray
    ground_diffuse_column: jnp.ndarray
    surface_direct_patch: jnp.ndarray
    surface_diffuse_patch: jnp.ndarray
    canopy: UrbanCanopyAlbedoState


class UrbanAlbedoState(NamedTuple):
    shortwave_absorption: UrbanShortwaveAbsorptionState
    surface: UrbanSurfaceAlbedoState


class UrbanAlbedoDiagnostics(NamedTuple):
    cosine_solar_zenith_landunit: jnp.ndarray
    direct_converged_landunit_band: jnp.ndarray
    diffuse_converged_landunit_band: jnp.ndarray
    direct_iterations_landunit_band: jnp.ndarray
    diffuse_iterations_landunit_band: jnp.ndarray


class UrbanAlbedoResult(NamedTuple):
    state: UrbanAlbedoState
    diagnostics: UrbanAlbedoDiagnostics


class _ReflectionState(NamedTuple):
    road_reflected_sun_wall: jnp.ndarray
    road_reflected_shade_wall: jnp.ndarray
    sun_wall_reflected_road: jnp.ndarray
    sun_wall_reflected_shade_wall: jnp.ndarray
    shade_wall_reflected_road: jnp.ndarray
    shade_wall_reflected_sun_wall: jnp.ndarray
    absorbed_impervious_road: jnp.ndarray
    absorbed_pervious_road: jnp.ndarray
    absorbed_sun_wall: jnp.ndarray
    absorbed_shade_wall: jnp.ndarray
    reflected_impervious_road: jnp.ndarray
    reflected_pervious_road: jnp.ndarray
    reflected_sun_wall: jnp.ndarray
    reflected_shade_wall: jnp.ndarray
    done: jnp.ndarray
    iterations: jnp.ndarray


def _add(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return lax.optimization_barrier(fortran_add(left, right))


def _multiply(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return lax.optimization_barrier(fortran_multiply(left, right))


def _divide(numerator: jnp.ndarray, denominator: jnp.ndarray) -> jnp.ndarray:
    return lax.optimization_barrier(fortran_divide(numerator, denominator))


def _one_minus(value: jnp.ndarray) -> jnp.ndarray:
    return _add(jnp.float64(1.0), -value)


def _urban_mask(landunit_type: jnp.ndarray) -> jnp.ndarray:
    return (landunit_type >= jnp.int32(LANDUNIT_URBAN_MIN)) & (
        landunit_type <= jnp.int32(LANDUNIT_URBAN_MAX)
    )


def _cast_and_validate(
    topology: SubgridTopology,
    inputs: UrbanAlbedoInputs,
    state: UrbanAlbedoState,
) -> tuple[UrbanAlbedoInputs, UrbanAlbedoState]:
    number_of_landunits = topology.landunit.gridcell.shape[0]
    number_of_columns = topology.column.gridcell.shape[0]
    number_of_patches = topology.patch.gridcell.shape[0]
    values = {
        name: jnp.asarray(getattr(inputs, name), dtype=jnp.float64)
        for name in UrbanAlbedoInputs._fields
    }
    for name in UrbanAlbedoInputs._fields[:3]:
        if values[name].shape != (number_of_columns,):
            raise ValueError(f"{name} must contain one value per column")
    for name in UrbanAlbedoInputs._fields[3:5]:
        if values[name].shape != (number_of_landunits,):
            raise ValueError(f"{name} must contain one value per landunit")
    for name in UrbanAlbedoInputs._fields[5:]:
        if values[name].shape != (number_of_landunits, 2):
            raise ValueError(f"{name} must have shape (landunit, 2)")

    absorption = {
        name: jnp.asarray(getattr(state.shortwave_absorption, name), dtype=jnp.float64)
        for name in UrbanShortwaveAbsorptionState._fields
    }
    for name, value in absorption.items():
        if value.shape != (number_of_landunits, 2):
            raise ValueError(f"{name} must have shape (landunit, 2)")

    surface_values = {
        name: jnp.asarray(getattr(state.surface, name), dtype=jnp.float64)
        for name in UrbanSurfaceAlbedoState._fields[:-1]
    }
    for name in ("ground_direct_column", "ground_diffuse_column"):
        if surface_values[name].shape != (number_of_columns, 2):
            raise ValueError(f"{name} must have shape (column, 2)")
    for name in ("surface_direct_patch", "surface_diffuse_patch"):
        if surface_values[name].shape != (number_of_patches, 2):
            raise ValueError(f"{name} must have shape (patch, 2)")
    canopy = {
        name: jnp.asarray(getattr(state.surface.canopy, name), dtype=jnp.float64)
        for name in UrbanCanopyAlbedoState._fields
    }
    for name, value in canopy.items():
        if value.shape != (number_of_patches, 2):
            raise ValueError(f"{name} must have shape (patch, 2)")
    return (
        UrbanAlbedoInputs(**values),
        UrbanAlbedoState(
            UrbanShortwaveAbsorptionState(**absorption),
            UrbanSurfaceAlbedoState(
                **surface_values,
                canopy=UrbanCanopyAlbedoState(**canopy),
            ),
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
    selected = _urban_mask(jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)) & (
        jnp.asarray(topology.column.column_type, dtype=jnp.int32) == jnp.int32(column_type)
    )
    mapped = (
        jnp.zeros((number_of_landunits,), dtype=jnp.float64)
        .at[safe_landunit]
        .add(jnp.where(selected, values, jnp.float64(0.0)))
    )
    count = (
        jnp.zeros((number_of_landunits,), dtype=jnp.int32)
        .at[safe_landunit]
        .add(selected.astype(jnp.int32))
    )
    exists = count > jnp.int32(0)
    return jnp.where(exists, mapped, jnp.float64(default)), exists


@jax.custom_jvp
def _canyon_shadow_angle(value: jnp.ndarray) -> jnp.ndarray:
    from clm_jax.physics.native_inverse_sine import native_asin

    return native_asin(jnp.minimum(value, 1.0))


@_canyon_shadow_angle.defjvp
def _canyon_shadow_angle_jvp(primals, tangents):
    value, = primals
    tangent, = tangents
    varying = value < 1.0
    safe = jnp.where(varying, value, 0.0)
    slope_denominator = jnp.sqrt((1.0 - safe) * (1.0 + safe))
    return _canyon_shadow_angle(value), jnp.where(varying, tangent / slope_denominator, 0.0)


def _incident_direct(
    cosine_zenith: jnp.ndarray,
    ratio: jnp.ndarray,
    day: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    safe_cosine = jnp.where(day, cosine_zenith, jnp.float64(0.5))
    zenith = strict_acos(safe_cosine)
    tangent = native_tan(zenith)
    clamped_tangent = jnp.where(zenith >= jnp.float64(1.0e-6), tangent,
                               jnp.float64(_MINIMUM_CANYON_TANGENT))
    denominator = _multiply(ratio, clamped_tangent)
    theta = _canyon_shadow_angle(_divide(jnp.float64(1.0), denominator))
    one_minus_cosine = _one_minus(native_cos(theta))
    first_road = _divide(_multiply(jnp.float64(2.0), theta), jnp.float64(PI))
    second_road = _multiply(
        _multiply(
            _multiply(_divide(jnp.float64(2.0), jnp.float64(PI)), ratio),
            tangent,
        ),
        one_minus_cosine,
    )
    road = _add(first_road, -second_road)
    wall_first = _multiply(
        _divide(jnp.float64(1.0), ratio),
        _add(jnp.float64(0.5), -_divide(theta, jnp.float64(PI))),
    )
    wall_second = _multiply(
        _multiply(_divide(jnp.float64(1.0), jnp.float64(PI)), tangent),
        one_minus_cosine,
    )
    sun_wall = _multiply(jnp.float64(2.0), _add(wall_first, wall_second))
    zeros = jnp.zeros_like(road)
    return (
        jnp.where(day, road, zeros),
        jnp.where(day, sun_wall, zeros),
        zeros,
    )


def _initial_reflection_state(
    road_incident: jnp.ndarray,
    sun_wall_incident: jnp.ndarray,
    shade_wall_incident: jnp.ndarray,
    impervious_albedo: jnp.ndarray,
    pervious_albedo: jnp.ndarray,
    wall_albedo: jnp.ndarray,
    impervious_fraction: jnp.ndarray,
    pervious_fraction: jnp.ndarray,
    view_factors: tuple[jnp.ndarray, ...],
) -> _ReflectionState:
    vf_sr, vf_wr, vf_sw, vf_rw, vf_ww = (value[:, None] for value in view_factors)
    impervious_absorbed = _multiply(_one_minus(impervious_albedo), road_incident)
    impervious_reflected = _multiply(impervious_albedo, road_incident)
    pervious_absorbed = _multiply(_one_minus(pervious_albedo), road_incident)
    pervious_reflected = _multiply(pervious_albedo, road_incident)
    road_reflected = _add(
        _multiply(impervious_reflected, impervious_fraction),
        _multiply(pervious_reflected, pervious_fraction),
    )
    sun_absorbed = _multiply(_one_minus(wall_albedo), sun_wall_incident)
    sun_reflected = _multiply(wall_albedo, sun_wall_incident)
    shade_absorbed = _multiply(_one_minus(wall_albedo), shade_wall_incident)
    shade_reflected = _multiply(wall_albedo, shade_wall_incident)
    return _ReflectionState(
        road_reflected_sun_wall=_multiply(road_reflected, vf_wr),
        road_reflected_shade_wall=_multiply(road_reflected, vf_wr),
        sun_wall_reflected_road=_multiply(sun_reflected, vf_rw),
        sun_wall_reflected_shade_wall=_multiply(sun_reflected, vf_ww),
        shade_wall_reflected_road=_multiply(shade_reflected, vf_rw),
        shade_wall_reflected_sun_wall=_multiply(shade_reflected, vf_ww),
        absorbed_impervious_road=impervious_absorbed,
        absorbed_pervious_road=pervious_absorbed,
        absorbed_sun_wall=sun_absorbed,
        absorbed_shade_wall=shade_absorbed,
        reflected_impervious_road=_multiply(impervious_reflected, vf_sr),
        reflected_pervious_road=_multiply(pervious_reflected, vf_sr),
        reflected_sun_wall=_multiply(sun_reflected, vf_sw),
        reflected_shade_wall=_multiply(shade_reflected, vf_sw),
        done=jnp.zeros_like(impervious_albedo, dtype=jnp.bool_),
        iterations=jnp.zeros_like(impervious_albedo, dtype=jnp.int32),
    )


def _iterate_reflections(*arguments):
    if jax.default_backend() == "gpu":
        from clm_jax.physics.urban_reflections_gpu import shortwave_reflections

        return shortwave_reflections(*arguments)
    return _iterate_reflections_reference(*arguments)


def _iterate_reflections_reference(
    initial: _ReflectionState,
    ratio: jnp.ndarray,
    impervious_albedo: jnp.ndarray,
    pervious_albedo: jnp.ndarray,
    wall_albedo: jnp.ndarray,
    impervious_fraction: jnp.ndarray,
    pervious_fraction: jnp.ndarray,
    view_factors: tuple[jnp.ndarray, ...],
    enabled: jnp.ndarray,
) -> _ReflectionState:
    vf_sr, vf_wr, vf_sw, vf_rw, vf_ww = (value[:, None] for value in view_factors)
    ratio = ratio[:, None]

    def iteration(previous: _ReflectionState, _: None):
        road_incoming = _multiply(
            _add(
                previous.sun_wall_reflected_road,
                previous.shade_wall_reflected_road,
            ),
            ratio,
        )
        impervious_absorbed = _multiply(_one_minus(impervious_albedo), road_incoming)
        impervious_reflected = _multiply(impervious_albedo, road_incoming)
        pervious_absorbed = _multiply(_one_minus(pervious_albedo), road_incoming)
        pervious_reflected = _multiply(pervious_albedo, road_incoming)
        road_absorbed = _add(
            _multiply(impervious_absorbed, impervious_fraction),
            _multiply(pervious_absorbed, pervious_fraction),
        )
        road_reflected = _add(
            _multiply(impervious_reflected, impervious_fraction),
            _multiply(pervious_reflected, pervious_fraction),
        )

        sun_incoming = _add(
            _divide(previous.road_reflected_sun_wall, ratio),
            previous.shade_wall_reflected_sun_wall,
        )
        sun_absorbed = _multiply(_one_minus(wall_albedo), sun_incoming)
        sun_reflected = _multiply(wall_albedo, sun_incoming)
        shade_incoming = _add(
            _divide(previous.road_reflected_shade_wall, ratio),
            previous.sun_wall_reflected_shade_wall,
        )
        shade_absorbed = _multiply(_one_minus(wall_albedo), shade_incoming)
        shade_reflected = _multiply(wall_albedo, shade_incoming)

        candidate = _ReflectionState(
            road_reflected_sun_wall=_multiply(road_reflected, vf_wr),
            road_reflected_shade_wall=_multiply(road_reflected, vf_wr),
            sun_wall_reflected_road=_multiply(sun_reflected, vf_rw),
            sun_wall_reflected_shade_wall=_multiply(sun_reflected, vf_ww),
            shade_wall_reflected_road=_multiply(shade_reflected, vf_rw),
            shade_wall_reflected_sun_wall=_multiply(shade_reflected, vf_ww),
            absorbed_impervious_road=_add(previous.absorbed_impervious_road, impervious_absorbed),
            absorbed_pervious_road=_add(previous.absorbed_pervious_road, pervious_absorbed),
            absorbed_sun_wall=_add(previous.absorbed_sun_wall, sun_absorbed),
            absorbed_shade_wall=_add(previous.absorbed_shade_wall, shade_absorbed),
            reflected_impervious_road=_add(
                previous.reflected_impervious_road,
                _multiply(impervious_reflected, vf_sr),
            ),
            reflected_pervious_road=_add(
                previous.reflected_pervious_road,
                _multiply(pervious_reflected, vf_sr),
            ),
            reflected_sun_wall=_add(previous.reflected_sun_wall, _multiply(sun_reflected, vf_sw)),
            reflected_shade_wall=_add(
                previous.reflected_shade_wall, _multiply(shade_reflected, vf_sw)
            ),
            done=previous.done
            | (
                jnp.maximum(jnp.maximum(road_absorbed, sun_absorbed), shade_absorbed)
                < jnp.float64(SOLAR_REFLECTION_CONVERGENCE)
            ),
            iterations=previous.iterations + (enabled & ~previous.done).astype(jnp.int32),
        )
        active = enabled & ~previous.done
        updated = _ReflectionState(
            *(jnp.where(active, new, old) for new, old in zip(candidate, previous, strict=True))
        )
        return updated, None

    result, _ = lax.scan(
        iteration,
        initial,
        xs=None,
        length=MAXIMUM_SOLAR_REFLECTIONS,
    )
    return result


def _solve_canyon(
    ratio: jnp.ndarray,
    pervious_fraction: jnp.ndarray,
    cosine_zenith: jnp.ndarray,
    impervious_direct_albedo: jnp.ndarray,
    pervious_direct_albedo: jnp.ndarray,
    wall_direct_albedo: jnp.ndarray,
    impervious_diffuse_albedo: jnp.ndarray,
    pervious_diffuse_albedo: jnp.ndarray,
    wall_diffuse_albedo: jnp.ndarray,
    view_factors: tuple[jnp.ndarray, ...],
    urban: jnp.ndarray,
) -> tuple[_ReflectionState, _ReflectionState]:
    day = urban & (cosine_zenith > jnp.float64(0.0))
    road_direct, sun_direct, shade_direct = _incident_direct(cosine_zenith, ratio, day)
    vf_sr, _, vf_sw, _, _ = view_factors
    road_diffuse = vf_sr
    sun_diffuse = vf_sw
    shade_diffuse = vf_sw
    direct_incident = tuple(value[:, None] for value in (road_direct, sun_direct, shade_direct))
    diffuse_incident = tuple(value[:, None] for value in (road_diffuse, sun_diffuse, shade_diffuse))
    impervious_fraction = _one_minus(pervious_fraction)[:, None]
    pervious_fraction_2d = pervious_fraction[:, None]
    enabled = day[:, None]
    direct = _initial_reflection_state(
        *direct_incident,
        impervious_direct_albedo,
        pervious_direct_albedo,
        wall_direct_albedo,
        impervious_fraction,
        pervious_fraction_2d,
        view_factors,
    )
    diffuse = _initial_reflection_state(
        *diffuse_incident,
        impervious_diffuse_albedo,
        pervious_diffuse_albedo,
        wall_diffuse_albedo,
        impervious_fraction,
        pervious_fraction_2d,
        view_factors,
    )
    return (
        _iterate_reflections(
            direct,
            ratio,
            impervious_direct_albedo,
            pervious_direct_albedo,
            wall_direct_albedo,
            impervious_fraction,
            pervious_fraction_2d,
            view_factors,
            enabled,
        ),
        _iterate_reflections(
            diffuse,
            ratio,
            impervious_diffuse_albedo,
            pervious_diffuse_albedo,
            wall_diffuse_albedo,
            impervious_fraction,
            pervious_fraction_2d,
            view_factors,
            enabled,
        ),
    )


def urban_albedo(
    topology: SubgridTopology,
    inputs: UrbanAlbedoInputs,
    state: UrbanAlbedoState,
) -> UrbanAlbedoResult:

    inputs, state = _cast_and_validate(topology, inputs, state)
    landunit_type = jnp.asarray(topology.landunit.landunit_type, dtype=jnp.int32)
    urban_landunit = _urban_mask(landunit_type)
    safe_ratio = jnp.where(
        urban_landunit,
        inputs.canyon_height_width_ratio_landunit,
        jnp.float64(1.0),
    )
    safe_pervious = jnp.where(
        urban_landunit,
        inputs.pervious_road_fraction_landunit,
        jnp.float64(0.5),
    )
    view_factors = urban_view_factors(safe_ratio)

    cosine_zenith, _ = _column_surface_value(
        topology,
        inputs.cosine_solar_zenith_column,
        COLUMN_URBAN_ROOF,
        0.0,
    )
    day = urban_landunit & (cosine_zenith > jnp.float64(0.0))

    def snow_adjusted(
        base: jnp.ndarray,
        column_type: int,
    ) -> jnp.ndarray:
        snow_fraction, exists = _column_surface_value(
            topology, inputs.snow_fraction_column, column_type, 0.0
        )
        snow_water, _ = _column_surface_value(topology, inputs.snow_water_column, column_type, 0.0)
        snow_albedo = jnp.asarray(
            [URBAN_SNOW_VISIBLE_ALBEDO, URBAN_SNOW_NEAR_INFRARED_ALBEDO],
            dtype=jnp.float64,
        )
        snow_albedo = jnp.where(
            (day & (snow_water > jnp.float64(0.0)))[:, None],
            snow_albedo[None, :],
            jnp.float64(0.0),
        )
        blended = _add(
            _multiply(base, _one_minus(snow_fraction)[:, None]),
            _multiply(snow_albedo, snow_fraction[:, None]),
        )
        return jnp.where(exists[:, None], blended, base)

    def safe(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(urban_landunit[:, None], value, jnp.float64(0.2))

    roof_direct = snow_adjusted(safe(inputs.roof_direct_albedo_landunit), COLUMN_URBAN_ROOF)
    roof_diffuse = snow_adjusted(safe(inputs.roof_diffuse_albedo_landunit), COLUMN_URBAN_ROOF)
    impervious_direct = snow_adjusted(
        safe(inputs.impervious_road_direct_albedo_landunit),
        COLUMN_URBAN_IMPERVIOUS_ROAD,
    )
    impervious_diffuse = snow_adjusted(
        safe(inputs.impervious_road_diffuse_albedo_landunit),
        COLUMN_URBAN_IMPERVIOUS_ROAD,
    )
    pervious_direct = snow_adjusted(
        safe(inputs.pervious_road_direct_albedo_landunit),
        COLUMN_URBAN_PERVIOUS_ROAD,
    )
    pervious_diffuse = snow_adjusted(
        safe(inputs.pervious_road_diffuse_albedo_landunit),
        COLUMN_URBAN_PERVIOUS_ROAD,
    )
    wall_direct = safe(inputs.wall_direct_albedo_landunit)
    wall_diffuse = safe(inputs.wall_diffuse_albedo_landunit)

    direct, diffuse = _solve_canyon(
        safe_ratio,
        safe_pervious,
        cosine_zenith,
        impervious_direct,
        pervious_direct,
        wall_direct,
        impervious_diffuse,
        pervious_diffuse,
        wall_diffuse,
        view_factors,
        urban_landunit,
    )
    any_day = jnp.any(day)
    day_2d = day[:, None]
    vf_sr, _, vf_sw, _, _ = view_factors

    def reflected(
        candidate: jnp.ndarray,
        night_default: jnp.ndarray,
    ) -> jnp.ndarray:
        computed = jnp.where(day_2d, candidate, night_default[:, None])
        return jnp.where(any_day, computed, night_default[:, None])

    roof_reflected_direct = reflected(roof_direct, jnp.ones_like(vf_sr))
    roof_reflected_diffuse = reflected(roof_diffuse, jnp.ones_like(vf_sr))
    sun_reflected_direct = reflected(direct.reflected_sun_wall, vf_sw)
    sun_reflected_diffuse = reflected(diffuse.reflected_sun_wall, vf_sw)
    shade_reflected_direct = reflected(direct.reflected_shade_wall, vf_sw)
    shade_reflected_diffuse = reflected(diffuse.reflected_shade_wall, vf_sw)
    impervious_reflected_direct = reflected(direct.reflected_impervious_road, vf_sr)
    impervious_reflected_diffuse = reflected(diffuse.reflected_impervious_road, vf_sr)
    pervious_reflected_direct = reflected(direct.reflected_pervious_road, vf_sr)
    pervious_reflected_diffuse = reflected(diffuse.reflected_pervious_road, vf_sr)

    zero_landunit_band = jnp.zeros_like(roof_direct)
    old_absorption = state.shortwave_absorption

    def absorption(candidate: jnp.ndarray, old: jnp.ndarray) -> jnp.ndarray:
        urban_value = jnp.where(day_2d & any_day, candidate, zero_landunit_band)
        return jnp.where(urban_landunit[:, None], urban_value, old)

    absorption_state = UrbanShortwaveAbsorptionState(
        roof_direct_landunit=absorption(
            _one_minus(roof_direct), old_absorption.roof_direct_landunit
        ),
        roof_diffuse_landunit=absorption(
            _one_minus(roof_diffuse), old_absorption.roof_diffuse_landunit
        ),
        sun_wall_direct_landunit=absorption(
            direct.absorbed_sun_wall, old_absorption.sun_wall_direct_landunit
        ),
        sun_wall_diffuse_landunit=absorption(
            diffuse.absorbed_sun_wall, old_absorption.sun_wall_diffuse_landunit
        ),
        shade_wall_direct_landunit=absorption(
            direct.absorbed_shade_wall, old_absorption.shade_wall_direct_landunit
        ),
        shade_wall_diffuse_landunit=absorption(
            diffuse.absorbed_shade_wall, old_absorption.shade_wall_diffuse_landunit
        ),
        impervious_road_direct_landunit=absorption(
            direct.absorbed_impervious_road, old_absorption.impervious_road_direct_landunit
        ),
        impervious_road_diffuse_landunit=absorption(
            diffuse.absorbed_impervious_road, old_absorption.impervious_road_diffuse_landunit
        ),
        pervious_road_direct_landunit=absorption(
            direct.absorbed_pervious_road, old_absorption.pervious_road_direct_landunit
        ),
        pervious_road_diffuse_landunit=absorption(
            diffuse.absorbed_pervious_road, old_absorption.pervious_road_diffuse_landunit
        ),
    )

    column_landunit = jnp.asarray(topology.column.landunit, dtype=jnp.int32)
    patch_landunit = jnp.asarray(topology.patch.landunit, dtype=jnp.int32)
    number_of_landunits = landunit_type.shape[0]
    safe_column_landunit = jnp.clip(column_landunit, 0, max(number_of_landunits - 1, 0))
    safe_patch_landunit = jnp.clip(patch_landunit, 0, max(number_of_landunits - 1, 0))
    column_type = jnp.asarray(topology.column.column_type, dtype=jnp.int32)
    patch_type = jnp.asarray(topology.patch.column_type, dtype=jnp.int32)
    urban_column = _urban_mask(jnp.asarray(topology.column.landunit_type, dtype=jnp.int32))
    urban_patch = _urban_mask(jnp.asarray(topology.patch.landunit_type, dtype=jnp.int32))

    def select_surface(
        types: jnp.ndarray,
        parent: jnp.ndarray,
        roof: jnp.ndarray,
        sun: jnp.ndarray,
        shade: jnp.ndarray,
        impervious: jnp.ndarray,
        pervious: jnp.ndarray,
    ) -> jnp.ndarray:
        output = jnp.zeros((types.shape[0], 2), dtype=jnp.float64)
        for kind, field in (
            (COLUMN_URBAN_ROOF, roof),
            (COLUMN_URBAN_SUN_WALL, sun),
            (COLUMN_URBAN_SHADE_WALL, shade),
            (COLUMN_URBAN_IMPERVIOUS_ROAD, impervious),
            (COLUMN_URBAN_PERVIOUS_ROAD, pervious),
        ):
            output = jnp.where((types == jnp.int32(kind))[:, None], field[parent], output)
        return output

    selected_column_direct = select_surface(
        column_type,
        safe_column_landunit,
        roof_reflected_direct,
        sun_reflected_direct,
        shade_reflected_direct,
        impervious_reflected_direct,
        pervious_reflected_direct,
    )
    selected_column_diffuse = select_surface(
        column_type,
        safe_column_landunit,
        roof_reflected_diffuse,
        sun_reflected_diffuse,
        shade_reflected_diffuse,
        impervious_reflected_diffuse,
        pervious_reflected_diffuse,
    )
    selected_patch_direct = select_surface(
        patch_type,
        safe_patch_landunit,
        roof_reflected_direct,
        sun_reflected_direct,
        shade_reflected_direct,
        impervious_reflected_direct,
        pervious_reflected_direct,
    )
    selected_patch_diffuse = select_surface(
        patch_type,
        safe_patch_landunit,
        roof_reflected_diffuse,
        sun_reflected_diffuse,
        shade_reflected_diffuse,
        impervious_reflected_diffuse,
        pervious_reflected_diffuse,
    )

    previous_surface = state.surface
    ground_direct = jnp.where(
        urban_column[:, None], selected_column_direct, previous_surface.ground_direct_column
    )
    ground_diffuse = jnp.where(
        urban_column[:, None], selected_column_diffuse, previous_surface.ground_diffuse_column
    )
    surface_direct = jnp.where(
        urban_patch[:, None], selected_patch_direct, previous_surface.surface_direct_patch
    )
    surface_diffuse = jnp.where(
        urban_patch[:, None], selected_patch_diffuse, previous_surface.surface_diffuse_patch
    )
    zero_patch_band = jnp.zeros_like(surface_direct)
    transmitted = jnp.broadcast_to(
        day[safe_patch_landunit, None].astype(jnp.float64), surface_direct.shape
    )
    old_canopy = previous_surface.canopy

    def patch_zero(old: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(urban_patch[:, None], zero_patch_band, old)

    def patch_transmitted(old: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(urban_patch[:, None], transmitted, old)

    canopy = UrbanCanopyAlbedoState(
        absorbed_direct_patch=patch_zero(old_canopy.absorbed_direct_patch),
        absorbed_sunlit_direct_patch=patch_zero(old_canopy.absorbed_sunlit_direct_patch),
        absorbed_shaded_direct_patch=patch_zero(old_canopy.absorbed_shaded_direct_patch),
        absorbed_diffuse_patch=patch_zero(old_canopy.absorbed_diffuse_patch),
        absorbed_sunlit_diffuse_patch=patch_zero(old_canopy.absorbed_sunlit_diffuse_patch),
        absorbed_shaded_diffuse_patch=patch_zero(old_canopy.absorbed_shaded_diffuse_patch),
        transmitted_direct_patch=patch_transmitted(old_canopy.transmitted_direct_patch),
        transmitted_diffuse_from_direct_patch=patch_zero(
            old_canopy.transmitted_diffuse_from_direct_patch
        ),
        transmitted_diffuse_patch=patch_transmitted(old_canopy.transmitted_diffuse_patch),
    )
    return UrbanAlbedoResult(
        state=UrbanAlbedoState(
            shortwave_absorption=absorption_state,
            surface=UrbanSurfaceAlbedoState(
                ground_direct_column=ground_direct,
                ground_diffuse_column=ground_diffuse,
                surface_direct_patch=surface_direct,
                surface_diffuse_patch=surface_diffuse,
                canopy=canopy,
            ),
        ),
        diagnostics=UrbanAlbedoDiagnostics(
            cosine_solar_zenith_landunit=cosine_zenith,
            direct_converged_landunit_band=jnp.where(
                urban_landunit[:, None],
                direct.done & (direct.iterations < jnp.int32(MAXIMUM_SOLAR_REFLECTIONS)),
                False,
            ),
            diffuse_converged_landunit_band=jnp.where(
                urban_landunit[:, None],
                diffuse.done & (diffuse.iterations < jnp.int32(MAXIMUM_SOLAR_REFLECTIONS)),
                False,
            ),
            direct_iterations_landunit_band=jnp.where(
                urban_landunit[:, None], direct.iterations, jnp.int32(0)
            ),
            diffuse_iterations_landunit_band=jnp.where(
                urban_landunit[:, None], diffuse.iterations, jnp.int32(0)
            ),
        ),
    )
