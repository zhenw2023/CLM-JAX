
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import custom_jvp, jvp, lax

from clm_jax.constants import (
    DENSITY_FRESH_WATER,
    DENSITY_ICE,
    FREEZING_TEMPERATURE_FRESHWATER,
)
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.numerics import (
    fortran_divide,
    fortran_ordered_sum,
    fortran_product,
)
from clm_jax.physics.pointwise import scalar_map


class SoilMoistureStressInputs(NamedTuple):
    layer_thickness_column_level: jnp.ndarray
    saturated_water_content_column_level: jnp.ndarray
    saturated_suction_column_level: jnp.ndarray
    clapp_hornberger_exponent_column_level: jnp.ndarray
    soil_ice_water_column_level: jnp.ndarray
    soil_liquid_water_column_level: jnp.ndarray
    total_volumetric_water_column_level: jnp.ndarray
    soil_temperature_column_level: jnp.ndarray
    root_fraction_patch_level: jnp.ndarray
    full_open_water_potential_pft: jnp.ndarray
    full_close_water_potential_pft: jnp.ndarray
    previous_active_layer_index_column: jnp.ndarray
    current_active_layer_index_column: jnp.ndarray


class SoilMoistureStressState(NamedTuple):
    effective_porosity_column_level: jnp.ndarray
    liquid_water_fraction_column_level: jnp.ndarray
    root_resistance_patch_level: jnp.ndarray
    effective_root_fraction_patch_level: jnp.ndarray
    transpiration_wetness_patch: jnp.ndarray
    secondary_transpiration_wetness_patch: jnp.ndarray


def _add(*terms: jnp.ndarray) -> jnp.ndarray:
    broadcast = jnp.broadcast_arrays(
        *(jnp.asarray(term, dtype=jnp.float64) for term in terms)
    )
    return lax.optimization_barrier(
        fortran_ordered_sum(jnp.stack(broadcast, axis=0))
    )


def _multiply(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    left, right = jnp.broadcast_arrays(
        jnp.asarray(left, dtype=jnp.float64),
        jnp.asarray(right, dtype=jnp.float64),
    )
    return lax.optimization_barrier(fortran_product(left, right))


def _scalar_power(base: jnp.ndarray, exponent: jnp.ndarray) -> jnp.ndarray:
    base, exponent = jnp.broadcast_arrays(
        jnp.asarray(base, dtype=jnp.float64),
        jnp.asarray(exponent, dtype=jnp.float64),
    )
    shape = base.shape
    result = scalar_map(
        lambda operands: operands[0] ** operands[1],
        (jnp.reshape(base, (-1,)), jnp.reshape(exponent, (-1,))),
    )
    return jnp.reshape(result, shape)


def _active_column_mask(
    patch_column: jnp.ndarray,
    active_patch: jnp.ndarray,
    number_of_columns: int,
) -> jnp.ndarray:
    counts = jnp.zeros(number_of_columns, dtype=jnp.int32).at[patch_column].max(
        active_patch.astype(jnp.int32)
    )
    return counts > jnp.int32(0)


def effective_soil_porosity(
    layer_thickness_column_level: jnp.ndarray,
    saturated_water_content_column_level: jnp.ndarray,
    soil_ice_water_column_level: jnp.ndarray,
    active_column: jnp.ndarray,
    previous: jnp.ndarray,
) -> jnp.ndarray:

    denominator = _multiply(
        jnp.float64(DENSITY_ICE),
        layer_thickness_column_level,
    )
    volumetric_ice = jnp.minimum(
        saturated_water_content_column_level,
        soil_ice_water_column_level / denominator,
    )
    calculated = _add(saturated_water_content_column_level, -volumetric_ice)
    return jnp.where(active_column[:, None], calculated, previous)


def volumetric_liquid_water(
    layer_thickness_column_level: jnp.ndarray,
    effective_porosity_column_level: jnp.ndarray,
    soil_liquid_water_column_level: jnp.ndarray,
    active_column: jnp.ndarray,
    previous: jnp.ndarray,
) -> jnp.ndarray:

    denominator = _multiply(
        layer_thickness_column_level,
        jnp.float64(DENSITY_FRESH_WATER),
    )
    calculated = jnp.minimum(
        effective_porosity_column_level,
        soil_liquid_water_column_level / denominator,
    )
    return jnp.where(active_column[:, None], calculated, previous)


def clapp_hornberger_soil_suction(
    relative_saturation: jnp.ndarray,
    saturated_suction: jnp.ndarray,
    exponent: jnp.ndarray,
) -> jnp.ndarray:

    return -_multiply(
        saturated_suction,
        _scalar_power(relative_saturation, -exponent),
    )


@custom_jvp
def _bounded_clapp_hornberger_suction(relative_saturation, saturated_suction, exponent, closure):
    return jnp.maximum(
        closure,
        clapp_hornberger_soil_suction(relative_saturation, saturated_suction, exponent),
    )


@_bounded_clapp_hornberger_suction.defjvp
def _bounded_clapp_hornberger_suction_jvp(primals, tangents):
    relative_saturation, saturated_suction, exponent, closure = primals
    unbounded = clapp_hornberger_soil_suction(
        relative_saturation, saturated_suction, exponent,
    )
    value = jnp.maximum(closure, unbounded)
    selected = ~(unbounded < closure)

    def guarded(relative_saturation, saturated_suction, exponent, closure):
        suction = clapp_hornberger_soil_suction(
            jnp.where(selected, relative_saturation, jnp.float64(1.0)),
            jnp.where(selected, saturated_suction, jnp.float64(1.0)),
            jnp.where(selected, exponent, jnp.float64(1.0)),
        )
        return jnp.where(selected, jnp.maximum(closure, suction), closure)

    _, tangent = jvp(guarded, primals, tangents)
    return value, tangent


def root_zone_soil_wetness(
    root_fraction_patch_level: jnp.ndarray,
    total_volumetric_water_patch_level: jnp.ndarray,
    saturated_water_content_patch_level: jnp.ndarray,
    saturated_suction_patch_level: jnp.ndarray,
    clapp_hornberger_exponent_patch_level: jnp.ndarray,
    full_opening_potential_patch: jnp.ndarray,
    full_closure_potential_patch: jnp.ndarray,
    *,
    initial_patch: jnp.ndarray | None = None,
) -> jnp.ndarray:

    root_fraction = jnp.asarray(root_fraction_patch_level, dtype=jnp.float64)
    level_fields = tuple(
        jnp.asarray(value, dtype=jnp.float64)
        for value in (
            total_volumetric_water_patch_level,
            saturated_water_content_patch_level,
            saturated_suction_patch_level,
            clapp_hornberger_exponent_patch_level,
        )
    )
    if root_fraction.ndim != 2 or any(
        value.shape != root_fraction.shape for value in level_fields
    ):
        raise ValueError("root-zone wetness fields must share (patch, level)")
    opening = jnp.asarray(full_opening_potential_patch, dtype=jnp.float64)
    closure = jnp.asarray(full_closure_potential_patch, dtype=jnp.float64)
    if opening.shape != root_fraction.shape[:1] or closure.shape != opening.shape:
        raise ValueError("stomatal soil-potential thresholds must use the patch axis")
    if initial_patch is None:
        initial = jnp.zeros_like(opening)
    else:
        initial = jnp.asarray(initial_patch, dtype=jnp.float64)
        if initial.shape != opening.shape:
            raise ValueError("initial root-zone wetness must use the patch axis")

    total_water, water_at_saturation, suction_at_saturation, exponent = level_fields
    relative_saturation = jnp.maximum(
        fortran_divide(total_water, water_at_saturation),
        jnp.float64(0.01),
    )
    soil_potential = _bounded_clapp_hornberger_suction(
        relative_saturation,
        suction_at_saturation,
        exponent,
        closure[:, None],
    )
    potential_range = _add(opening, -closure)
    safe_range = jnp.where(
        potential_range != jnp.float64(0.0),
        potential_range,
        jnp.float64(1.0),
    )
    stress = jnp.clip(
        fortran_divide(
            _add(soil_potential, -closure[:, None]),
            safe_range[:, None],
        ),
        min=jnp.float64(0.0),
        max=jnp.float64(1.0),
    )
    return fortran_ordered_sum(
        _multiply(root_fraction, stress).T,
        initial=initial,
    )


def _normalized_unfrozen_root_fraction(
    root_fraction: jnp.ndarray,
    soil_temperature: jnp.ndarray,
    previous_active_layer_index: jnp.ndarray,
    current_active_layer_index: jnp.ndarray,
    *,
    perchroot: bool,
    perchroot_alt: bool,
) -> jnp.ndarray:
    if not (perchroot or perchroot_alt):
        return jnp.zeros_like(root_fraction)

    if perchroot_alt:
        maximum_index = jnp.maximum(
            jnp.maximum(previous_active_layer_index, current_active_layer_index),
            jnp.int32(1),
        )
        level_number = jnp.arange(1, root_fraction.shape[1] + 1, dtype=jnp.int32)
        included = level_number[None, :] <= maximum_index[:, None]
    else:
        included = soil_temperature >= jnp.float64(
            FREEZING_TEMPERATURE_FRESHWATER
        )

    unfrozen = jnp.where(included, root_fraction, jnp.float64(0.0))
    total = fortran_ordered_sum(unfrozen.T)
    return jnp.where(
        total[:, None] != jnp.float64(0.0),
        unfrozen / jnp.where(total[:, None] != 0.0, total[:, None], 1.0),
        unfrozen,
    )


def _validate_shapes(
    topology: SubgridTopology,
    active_patch: jnp.ndarray,
    inputs: SoilMoistureStressInputs,
    state: SoilMoistureStressState,
) -> tuple[int, int, int]:
    number_of_columns, number_of_levels = inputs.layer_thickness_column_level.shape
    number_of_patches = topology.patch.column.shape[0]
    column_level_fields = (
        inputs.saturated_water_content_column_level,
        inputs.saturated_suction_column_level,
        inputs.clapp_hornberger_exponent_column_level,
        inputs.soil_ice_water_column_level,
        inputs.soil_liquid_water_column_level,
        inputs.total_volumetric_water_column_level,
        inputs.soil_temperature_column_level,
        state.effective_porosity_column_level,
        state.liquid_water_fraction_column_level,
    )
    if any(field.shape != (number_of_columns, number_of_levels) for field in column_level_fields):
        raise ValueError("soil moisture column-level arrays must have matching shapes")
    patch_level_fields = (
        inputs.root_fraction_patch_level,
        state.root_resistance_patch_level,
        state.effective_root_fraction_patch_level,
    )
    if any(field.shape != (number_of_patches, number_of_levels) for field in patch_level_fields):
        raise ValueError("soil moisture patch-level arrays must have matching shapes")
    if active_patch.shape != (number_of_patches,):
        raise ValueError("active patch mask must match the patch dimension")
    if any(
        field.shape != (number_of_patches,)
        for field in (
            state.transpiration_wetness_patch,
            state.secondary_transpiration_wetness_patch,
        )
    ):
        raise ValueError("soil moisture patch state must match the patch dimension")
    if any(
        field.shape != (number_of_columns,)
        for field in (
            inputs.previous_active_layer_index_column,
            inputs.current_active_layer_index_column,
        )
    ):
        raise ValueError("active-layer indices must match the column dimension")
    return number_of_columns, number_of_patches, number_of_levels


def soil_moisture_stress(
    topology: SubgridTopology,
    active_patch: jnp.ndarray,
    inputs: SoilMoistureStressInputs,
    state: SoilMoistureStressState,
    *,
    hydraulic_stress: bool = True,
    perchroot: bool = False,
    perchroot_alt: bool = False,
) -> SoilMoistureStressState:

    number_of_columns, _, _ = _validate_shapes(topology, active_patch, inputs, state)
    active_patch = jnp.asarray(active_patch, dtype=jnp.bool_)
    inputs = SoilMoistureStressInputs(
        *(
            jnp.asarray(value, dtype=jnp.int32)
            if name.endswith("active_layer_index_column")
            else jnp.asarray(value, dtype=jnp.float64)
            for name, value in zip(SoilMoistureStressInputs._fields, inputs, strict=True)
        )
    )
    patch_column = topology.patch.column
    patch_vegetation = topology.patch.vegetation_type
    active_column = _active_column_mask(
        patch_column,
        active_patch,
        number_of_columns,
    )
    effective_porosity = effective_soil_porosity(
        inputs.layer_thickness_column_level,
        inputs.saturated_water_content_column_level,
        inputs.soil_ice_water_column_level,
        active_column,
        state.effective_porosity_column_level,
    )
    liquid_water_fraction = volumetric_liquid_water(
        inputs.layer_thickness_column_level,
        effective_porosity,
        inputs.soil_liquid_water_column_level,
        active_column,
        state.liquid_water_fraction_column_level,
    )

    patch_effective_porosity = effective_porosity[patch_column]
    patch_liquid_water = liquid_water_fraction[patch_column]
    patch_total_water = inputs.total_volumetric_water_column_level[patch_column]
    patch_saturated_water = inputs.saturated_water_content_column_level[patch_column]
    patch_saturated_suction = inputs.saturated_suction_column_level[patch_column]
    patch_exponent = inputs.clapp_hornberger_exponent_column_level[patch_column]
    patch_temperature = inputs.soil_temperature_column_level[patch_column]
    full_open = inputs.full_open_water_potential_pft[patch_vegetation]
    full_close = inputs.full_close_water_potential_pft[patch_vegetation]

    unfrozen_root_fraction = _normalized_unfrozen_root_fraction(
        inputs.root_fraction_patch_level,
        patch_temperature,
        inputs.previous_active_layer_index_column[patch_column],
        inputs.current_active_layer_index_column[patch_column],
        perchroot=perchroot,
        perchroot_alt=perchroot_alt,
    )
    root_fraction = (
        unfrozen_root_fraction
        if perchroot or perchroot_alt
        else inputs.root_fraction_patch_level
    )

    valid_root = (patch_liquid_water > jnp.float64(0.0)) & (
        patch_temperature
        > jnp.float64(FREEZING_TEMPERATURE_FRESHWATER - 2.0)
    )
    safe_effective_porosity = jnp.where(
        valid_root,
        patch_effective_porosity,
        jnp.float64(1.0),
    )
    relative_liquid_saturation = jnp.maximum(
        patch_liquid_water / safe_effective_porosity,
        jnp.float64(0.01),
    )
    soil_potential = clapp_hornberger_soil_suction(
        relative_liquid_saturation,
        patch_saturated_suction,
        patch_exponent,
    )
    soil_potential = jnp.maximum(full_close[:, None], soil_potential)
    stress = jnp.minimum(
        fortran_divide(
            _multiply(
                patch_effective_porosity / patch_saturated_water,
                _add(soil_potential, -full_close[:, None]),
            ),
            _add(full_open[:, None], -full_close[:, None]),
        ),
        jnp.float64(1.0),
    )
    root_resistance = jnp.where(
        active_patch[:, None] & valid_root,
        stress,
        state.root_resistance_patch_level,
    )
    effective_root = jnp.where(
        active_patch[:, None] & valid_root,
        _multiply(root_fraction, stress),
        jnp.where(
            active_patch[:, None],
            jnp.float64(0.0),
            state.effective_root_fraction_patch_level,
        ),
    )

    if hydraulic_stress:
        transpiration_wetness = state.transpiration_wetness_patch
    else:
        transpiration_wetness = fortran_ordered_sum(
            jnp.maximum(effective_root, jnp.float64(0.0)).T,
            initial=state.transpiration_wetness_patch,
        )
        transpiration_wetness = jnp.where(
            active_patch,
            transpiration_wetness,
            state.transpiration_wetness_patch,
        )

    effective_root = jnp.where(
        active_patch[:, None],
        jnp.where(
            transpiration_wetness[:, None] > jnp.float64(0.0),
            effective_root
            / jnp.where(
                transpiration_wetness[:, None] > 0.0,
                transpiration_wetness[:, None],
                1.0,
            ),
            jnp.float64(0.0),
        ),
        state.effective_root_fraction_patch_level,
    )

    secondary_transpiration_wetness = root_zone_soil_wetness(
        inputs.root_fraction_patch_level,
        patch_total_water,
        patch_saturated_water,
        patch_saturated_suction,
        patch_exponent,
        full_open,
        full_close,
        initial_patch=state.secondary_transpiration_wetness_patch,
    )
    secondary_transpiration_wetness = jnp.where(
        active_patch,
        secondary_transpiration_wetness,
        state.secondary_transpiration_wetness_patch,
    )

    return SoilMoistureStressState(
        effective_porosity_column_level=effective_porosity,
        liquid_water_fraction_column_level=liquid_water_fraction,
        root_resistance_patch_level=root_resistance,
        effective_root_fraction_patch_level=effective_root,
        transpiration_wetness_patch=transpiration_wetness,
        secondary_transpiration_wetness_patch=secondary_transpiration_wetness,
    )
