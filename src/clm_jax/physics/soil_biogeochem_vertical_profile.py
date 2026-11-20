
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import lax

from clm_jax.physics.cn_zero_manifest import NUMBER_OF_DECOMPOSITION_LEVELS
from clm_jax.physics.numerics import (
    fortran_divide,
    fortran_ordered_sum,
    strict_negative_exp,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)

NUMBER_OF_GROUND_LEVELS = NUMBER_OF_DECOMPOSITION_LEVELS
DEFAULT_SURFACE_PROFILE_EXPONENT = 10.0
DEFAULT_MINIMUM_BEDROCK_DEPTH = 0.4
DEFAULT_NOT_VEGETATED_PFT = 0
PROFILE_INTEGRAL_TOLERANCE = 1.0e-10


def _soil_layer_geometry_20sl_8p5m() -> tuple[np.ndarray, np.ndarray]:

    thickness = np.empty(NUMBER_OF_GROUND_LEVELS, dtype=np.float64)
    for level in range(1, 5):
        thickness[level - 1] = np.float64(level) * np.float64(0.02)
    for level in range(5, 14):
        thickness[level - 1] = thickness[3] + np.float64(level - 4) * np.float64(
            0.04
        )
    for level in range(14, 21):
        thickness[level - 1] = thickness[12] + np.float64(level - 13) * np.float64(
            0.10
        )
    for level in range(21, NUMBER_OF_GROUND_LEVELS + 1):
        increment = (
            (np.float64(level - 20) * np.float64(25.0)) ** np.float64(1.5)
        ) / np.float64(100.0)
        thickness[level - 1] = thickness[19] + increment

    interface = np.empty(NUMBER_OF_GROUND_LEVELS + 1, dtype=np.float64)
    interface[0] = np.float64(0.0)
    running = np.float64(0.0)
    for level in range(NUMBER_OF_GROUND_LEVELS):
        running = running + thickness[level]
        interface[level + 1] = running
    node = np.float64(0.5) * (interface[:-1] + interface[1:])
    return node, thickness


_DEFAULT_NODE_DEPTH, _DEFAULT_LAYER_THICKNESS = _soil_layer_geometry_20sl_8p5m()


class SoilBiogeochemVerticalProfileInputs(NamedTuple):

    soil_column: jnp.ndarray
    soil_patch: jnp.ndarray
    patch_column_index: jnp.ndarray
    column_patch_indices: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    patch_weight_on_column: jnp.ndarray
    carbon_root_fraction_patch_level: jnp.ndarray
    active_layer_index_column: jnp.ndarray


class SoilBiogeochemVerticalProfileParameters(NamedTuple):

    decomposition_level_count: int | jnp.ndarray = 20
    decomposition_full_level_count: int | jnp.ndarray = 25
    use_vertical_soil_carbon: bool | jnp.ndarray = True
    use_bedrock: bool | jnp.ndarray = True
    not_vegetated_pft: int | jnp.ndarray = DEFAULT_NOT_VEGETATED_PFT
    surface_profile_exponent: float | jnp.ndarray = (
        DEFAULT_SURFACE_PROFILE_EXPONENT
    )
    minimum_bedrock_depth: float | jnp.ndarray = DEFAULT_MINIMUM_BEDROCK_DEPTH
    soil_node_depth_level: jnp.ndarray = _DEFAULT_NODE_DEPTH
    decomposition_layer_thickness_level: jnp.ndarray = _DEFAULT_LAYER_THICKNESS


DEFAULT_SOIL_BIOGEOCHEM_VERTICAL_PROFILE_PARAMETERS = (
    SoilBiogeochemVerticalProfileParameters()
)


class SoilBiogeochemVerticalProfileState(NamedTuple):

    leaf_profile_patch_level: jnp.ndarray
    fine_root_profile_patch_level: jnp.ndarray
    coarse_root_profile_patch_level: jnp.ndarray
    stem_profile_patch_level: jnp.ndarray
    nitrogen_fixation_profile_column_level: jnp.ndarray
    nitrogen_deposition_profile_column_level: jnp.ndarray


class SoilBiogeochemVerticalProfileDiagnostics(NamedTuple):
    updated_patch_count: jnp.ndarray
    updated_column_count: jnp.ndarray
    fallback_patch_count: jnp.ndarray
    fallback_column_count: jnp.ndarray
    profile_integral_violation_count: jnp.ndarray
    maximum_profile_integral_error: jnp.ndarray


class SoilBiogeochemVerticalProfileResult(NamedTuple):
    state: SoilBiogeochemVerticalProfileState
    diagnostics: SoilBiogeochemVerticalProfileDiagnostics


def initialize_soil_biogeochem_vertical_profile_state(
    number_of_patches: int,
    number_of_columns: int,
    *,
    fill_value: float = 1.0e36,
) -> SoilBiogeochemVerticalProfileState:

    if number_of_patches < 0 or number_of_columns < 0:
        raise ValueError("patch and column counts cannot be negative")
    patch = jnp.full(
        (number_of_patches, NUMBER_OF_GROUND_LEVELS),
        fill_value,
        dtype=jnp.float64,
    )
    column = jnp.full(
        (number_of_columns, NUMBER_OF_GROUND_LEVELS),
        fill_value,
        dtype=jnp.float64,
    )
    return SoilBiogeochemVerticalProfileState(
        leaf_profile_patch_level=patch,
        fine_root_profile_patch_level=patch,
        coarse_root_profile_patch_level=patch,
        stem_profile_patch_level=patch,
        nitrogen_fixation_profile_column_level=column,
        nitrogen_deposition_profile_column_level=column,
    )


def _validate_shapes(
    inputs: SoilBiogeochemVerticalProfileInputs,
    state: SoilBiogeochemVerticalProfileState,
    parameters: SoilBiogeochemVerticalProfileParameters,
) -> tuple[int, int]:
    if jnp.ndim(inputs.soil_column) != 1 or jnp.ndim(inputs.soil_patch) != 1:
        raise ValueError("soil_column and soil_patch must be vectors")
    number_of_columns = inputs.soil_column.shape[0]
    number_of_patches = inputs.soil_patch.shape[0]
    if number_of_patches > 0 and number_of_columns == 0:
        raise ValueError("patch topology requires at least one column")
    patch_shape = (number_of_patches,)
    for name in (
        "patch_column_index",
        "vegetation_type_patch",
        "patch_weight_on_column",
    ):
        if jnp.shape(getattr(inputs, name)) != patch_shape:
            raise ValueError(f"{name} must contain every patch")
    if (
        jnp.ndim(inputs.column_patch_indices) != 2
        or inputs.column_patch_indices.shape[0] != number_of_columns
    ):
        raise ValueError("column_patch_indices must have shape (column, patch slot)")
    if jnp.shape(inputs.carbon_root_fraction_patch_level) != (
        number_of_patches,
        NUMBER_OF_GROUND_LEVELS,
    ):
        raise ValueError("carbon_root_fraction_patch_level must have shape (patch, 25)")
    if jnp.shape(inputs.active_layer_index_column) != (number_of_columns,):
        raise ValueError("active_layer_index_column must contain every column")

    for name in SoilBiogeochemVerticalProfileState._fields[:4]:
        if jnp.shape(getattr(state, name)) != (
            number_of_patches,
            NUMBER_OF_GROUND_LEVELS,
        ):
            raise ValueError(f"{name} must have shape (patch, 25)")
    for name in SoilBiogeochemVerticalProfileState._fields[4:]:
        if jnp.shape(getattr(state, name)) != (
            number_of_columns,
            NUMBER_OF_GROUND_LEVELS,
        ):
            raise ValueError(f"{name} must have shape (column, 25)")

    for name in (
        "decomposition_level_count",
        "decomposition_full_level_count",
        "use_vertical_soil_carbon",
        "use_bedrock",
        "not_vegetated_pft",
        "surface_profile_exponent",
        "minimum_bedrock_depth",
    ):
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    for name in (
        "soil_node_depth_level",
        "decomposition_layer_thickness_level",
    ):
        if jnp.shape(getattr(parameters, name)) != (NUMBER_OF_GROUND_LEVELS,):
            raise ValueError(f"{name} must have shape (25,)")
    for value, label in (
        (parameters.decomposition_level_count, "decomposition_level_count"),
        (
            parameters.decomposition_full_level_count,
            "decomposition_full_level_count",
        ),
    ):
        if isinstance(value, (int, np.integer)) and not (
            1 <= value <= NUMBER_OF_GROUND_LEVELS
        ):
            raise ValueError(f"{label} must lie in [1, 25]")
    if (
        isinstance(parameters.decomposition_level_count, (int, np.integer))
        and isinstance(
            parameters.decomposition_full_level_count,
            (int, np.integer),
        )
        and parameters.decomposition_full_level_count
        < parameters.decomposition_level_count
    ):
        raise ValueError(
            "decomposition_full_level_count cannot be smaller than the active count"
        )
    return number_of_patches, number_of_columns


def _active_level_masks(
    parameters: SoilBiogeochemVerticalProfileParameters,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    level = jnp.arange(NUMBER_OF_GROUND_LEVELS, dtype=jnp.int32)
    return (
        level
        < jnp.asarray(parameters.decomposition_level_count, dtype=jnp.int32),
        level
        < jnp.asarray(
            parameters.decomposition_full_level_count,
            dtype=jnp.int32,
        ),
    )


def _ordered_level_sum(terms: jnp.ndarray) -> jnp.ndarray:

    return fortran_ordered_sum(
        jnp.swapaxes(jnp.asarray(terms, dtype=jnp.float64), 0, 1),
        initial=jnp.zeros(terms.shape[0], dtype=jnp.float64),
    )


def _surface_profile(
    parameters: SoilBiogeochemVerticalProfileParameters,
    active_level: jnp.ndarray,
) -> jnp.ndarray:
    node = jnp.asarray(parameters.soil_node_depth_level, dtype=jnp.float64)
    thickness = jnp.asarray(
        parameters.decomposition_layer_thickness_level,
        dtype=jnp.float64,
    )
    exponent = fortran_multiply(
        -jnp.asarray(parameters.surface_profile_exponent, dtype=jnp.float64),
        node,
    )
    profile = fortran_divide(strict_negative_exp(exponent), thickness)
    profile = jnp.where(active_level, profile, jnp.float64(0.0))
    bedrock_allowed = node <= jnp.asarray(
        parameters.minimum_bedrock_depth,
        dtype=jnp.float64,
    )
    retain = ~jnp.asarray(parameters.use_bedrock, dtype=jnp.bool_) | bedrock_allowed
    return jnp.where(retain, profile, jnp.float64(0.0))


def _active_layer_mask(
    active_layer_index: jnp.ndarray,
    parameters: SoilBiogeochemVerticalProfileParameters,
) -> jnp.ndarray:
    active_count = jnp.minimum(
        jnp.maximum(jnp.asarray(active_layer_index, dtype=jnp.int32), jnp.int32(1)),
        jnp.asarray(parameters.decomposition_level_count, dtype=jnp.int32),
    )
    level = jnp.arange(NUMBER_OF_GROUND_LEVELS, dtype=jnp.int32)
    return level[None, :] < active_count[:, None]


def _masked_divide(
    numerator: jnp.ndarray,
    denominator: jnp.ndarray,
    active: jnp.ndarray,
) -> jnp.ndarray:
    safe_denominator = jnp.where(active, denominator, jnp.float64(1.0))
    return jnp.where(
        active,
        fortran_divide(numerator, safe_denominator),
        jnp.float64(0.0),
    )


def _vertical_profiles(
    operands: tuple[
        SoilBiogeochemVerticalProfileInputs,
        SoilBiogeochemVerticalProfileState,
        SoilBiogeochemVerticalProfileParameters,
        jnp.ndarray,
        jnp.ndarray,
    ],
) -> tuple[SoilBiogeochemVerticalProfileState, jnp.ndarray, jnp.ndarray]:
    inputs, state, parameters, active_level, full_level = operands
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    patch_column = jnp.asarray(inputs.patch_column_index, dtype=jnp.int32)
    patch_column = jnp.clip(patch_column, 0, soil_column.shape[0] - 1)
    vegetation_type = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    root_fraction = jnp.asarray(
        inputs.carbon_root_fraction_patch_level,
        dtype=jnp.float64,
    )
    thickness = jnp.asarray(
        parameters.decomposition_layer_thickness_level,
        dtype=jnp.float64,
    )
    surface_profile = _surface_profile(parameters, active_level)

    patch_zero = jnp.zeros_like(root_fraction)
    column_zero = jnp.zeros(
        (soil_column.shape[0], NUMBER_OF_GROUND_LEVELS),
        dtype=jnp.float64,
    )
    patch_previous = [
        jnp.asarray(getattr(state, name), dtype=jnp.float64)
        for name in state._fields[:4]
    ]
    column_previous = [
        jnp.asarray(getattr(state, name), dtype=jnp.float64) for name in state._fields[4:]
    ]
    patch_base = [
        jnp.where(full_level[None, :], patch_zero, value) for value in patch_previous
    ]
    column_base = [
        jnp.where(full_level[None, :], column_zero, value) for value in column_previous
    ]

    vegetated = vegetation_type != jnp.asarray(
        parameters.not_vegetated_pft,
        dtype=jnp.int32,
    )
    root_input_profile = fortran_divide(root_fraction, thickness[None, :])
    root_input_profile = jnp.where(
        soil_patch[:, None] & vegetated[:, None] & active_level[None, :],
        root_input_profile,
        jnp.float64(0.0),
    )

    patch_active_layer = _active_layer_mask(
        jnp.asarray(inputs.active_layer_index_column, dtype=jnp.int32)[patch_column],
        parameters,
    )
    patch_root_total = _ordered_level_sum(
        jnp.where(
            patch_active_layer,
            fortran_multiply(root_input_profile, thickness[None, :]),
            jnp.float64(0.0),
        )
    )
    patch_surface_total = _ordered_level_sum(
        jnp.where(
            patch_active_layer,
            fortran_multiply(surface_profile[None, :], thickness[None, :]),
            jnp.float64(0.0),
        )
    )
    patch_alt = jnp.asarray(inputs.active_layer_index_column, dtype=jnp.int32)[
        patch_column
    ]
    normal_patch = (
        soil_patch
        & (patch_alt > jnp.int32(0))
        & (patch_root_total > jnp.float64(0.0))
        & (patch_surface_total > jnp.float64(0.0))
    )
    patch_write = normal_patch[:, None] & patch_active_layer
    root_profile = _masked_divide(
        root_input_profile,
        patch_root_total[:, None],
        patch_write,
    )
    shallow_profile = _masked_divide(
        jnp.broadcast_to(surface_profile, root_input_profile.shape),
        patch_surface_total[:, None],
        patch_write,
    )
    reciprocal_top_thickness = fortran_divide(jnp.float64(1.0), thickness[0])
    fallback_patch = soil_patch & ~normal_patch
    root_profile = root_profile.at[:, 0].set(
        jnp.where(fallback_patch, reciprocal_top_thickness, root_profile[:, 0])
    )
    shallow_profile = shallow_profile.at[:, 0].set(
        jnp.where(fallback_patch, reciprocal_top_thickness, shallow_profile[:, 0])
    )
    root_profile = jnp.where(full_level[None, :], root_profile, patch_base[1])
    shallow_profile = jnp.where(full_level[None, :], shallow_profile, patch_base[0])
    coarse_root_profile = jnp.where(
        full_level[None, :],
        root_profile,
        patch_base[2],
    )
    stem_profile = jnp.where(
        full_level[None, :],
        shallow_profile,
        patch_base[3],
    )

    patch_weight = jnp.asarray(inputs.patch_weight_on_column, dtype=jnp.float64)
    column_patch_indices = jnp.asarray(inputs.column_patch_indices, dtype=jnp.int32)

    def aggregate_patch_slot(
        column_values: jnp.ndarray,
        unbounded_patch: jnp.ndarray,
    ) -> tuple[jnp.ndarray, None]:
        bounded_patch = jnp.clip(unbounded_patch, 0, root_fraction.shape[0] - 1)
        contributes = (
            soil_column
            & (unbounded_patch >= jnp.int32(0))
            & (unbounded_patch < jnp.int32(root_fraction.shape[0]))
        )
        contribution = fortran_multiply(
            root_input_profile[bounded_patch],
            patch_weight[bounded_patch, None],
        )
        contribution = jnp.where(
            contributes[:, None],
            contribution,
            jnp.float64(0.0),
        )
        return fortran_add(column_values, contribution), None

    if root_fraction.shape[0] == 0:
        column_root_input = column_zero
    else:
        column_root_input, _ = lax.scan(
            aggregate_patch_slot,
            column_zero,
            jnp.swapaxes(column_patch_indices, 0, 1),
        )
    column_active_layer = _active_layer_mask(
        inputs.active_layer_index_column,
        parameters,
    )
    column_root_total = _ordered_level_sum(
        jnp.where(
            column_active_layer,
            fortran_multiply(column_root_input, thickness[None, :]),
            jnp.float64(0.0),
        )
    )
    column_surface_total = _ordered_level_sum(
        jnp.where(
            column_active_layer,
            fortran_multiply(surface_profile[None, :], thickness[None, :]),
            jnp.float64(0.0),
        )
    )
    column_alt = jnp.asarray(inputs.active_layer_index_column, dtype=jnp.int32)
    normal_column = (
        soil_column
        & (column_alt > jnp.int32(0))
        & (column_root_total > jnp.float64(0.0))
        & (column_surface_total > jnp.float64(0.0))
    )
    column_write = normal_column[:, None] & column_active_layer
    nitrogen_fixation = _masked_divide(
        column_root_input,
        column_root_total[:, None],
        column_write,
    )
    nitrogen_deposition = _masked_divide(
        jnp.broadcast_to(surface_profile, column_root_input.shape),
        column_surface_total[:, None],
        column_write,
    )
    fallback_column = soil_column & ~normal_column
    nitrogen_fixation = nitrogen_fixation.at[:, 0].set(
        jnp.where(
            fallback_column,
            reciprocal_top_thickness,
            nitrogen_fixation[:, 0],
        )
    )
    nitrogen_deposition = nitrogen_deposition.at[:, 0].set(
        jnp.where(
            fallback_column,
            reciprocal_top_thickness,
            nitrogen_deposition[:, 0],
        )
    )
    nitrogen_fixation = jnp.where(
        full_level[None, :],
        nitrogen_fixation,
        column_base[0],
    )
    nitrogen_deposition = jnp.where(
        full_level[None, :],
        nitrogen_deposition,
        column_base[1],
    )
    return (
        SoilBiogeochemVerticalProfileState(
            leaf_profile_patch_level=shallow_profile,
            fine_root_profile_patch_level=root_profile,
            coarse_root_profile_patch_level=coarse_root_profile,
            stem_profile_patch_level=stem_profile,
            nitrogen_fixation_profile_column_level=nitrogen_fixation,
            nitrogen_deposition_profile_column_level=nitrogen_deposition,
        ),
        fallback_patch,
        fallback_column,
    )


def _single_layer_profiles(
    operands: tuple[
        SoilBiogeochemVerticalProfileInputs,
        SoilBiogeochemVerticalProfileState,
        SoilBiogeochemVerticalProfileParameters,
        jnp.ndarray,
        jnp.ndarray,
    ],
) -> tuple[SoilBiogeochemVerticalProfileState, jnp.ndarray, jnp.ndarray]:
    inputs, state, _, _, full_level = operands

    def fill(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(
            full_level[None, :],
            jnp.float64(1.0),
            jnp.asarray(value, dtype=jnp.float64),
        )

    return (
        SoilBiogeochemVerticalProfileState(*(fill(value) for value in state)),
        jnp.zeros_like(inputs.soil_patch, dtype=jnp.bool_),
        jnp.zeros_like(inputs.soil_column, dtype=jnp.bool_),
    )


def _profile_integral(
    profile: jnp.ndarray,
    thickness: jnp.ndarray,
    active_level: jnp.ndarray,
) -> jnp.ndarray:
    terms = jnp.where(
        active_level[None, :],
        fortran_multiply(profile, thickness[None, :]),
        jnp.float64(0.0),
    )
    return _ordered_level_sum(terms)


def soil_biogeochem_vertical_profile(
    inputs: SoilBiogeochemVerticalProfileInputs,
    state: SoilBiogeochemVerticalProfileState,
    parameters: SoilBiogeochemVerticalProfileParameters = (
        DEFAULT_SOIL_BIOGEOCHEM_VERTICAL_PROFILE_PARAMETERS
    ),
) -> SoilBiogeochemVerticalProfileResult:

    _validate_shapes(inputs, state, parameters)
    active_level, full_level = _active_level_masks(parameters)
    operands = (inputs, state, parameters, active_level, full_level)
    updated, fallback_patch, fallback_column = lax.cond(
        jnp.asarray(parameters.use_vertical_soil_carbon, dtype=jnp.bool_),
        _vertical_profiles,
        _single_layer_profiles,
        operands,
    )

    thickness = jnp.asarray(
        parameters.decomposition_layer_thickness_level,
        dtype=jnp.float64,
    )
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    patch_integrals = jnp.stack(
        tuple(
            _profile_integral(getattr(updated, name), thickness, active_level)
            for name in SoilBiogeochemVerticalProfileState._fields[:4]
        ),
        axis=1,
    )
    column_integrals = jnp.stack(
        tuple(
            _profile_integral(getattr(updated, name), thickness, active_level)
            for name in SoilBiogeochemVerticalProfileState._fields[4:]
        ),
        axis=1,
    )
    patch_error = jnp.where(
        soil_patch[:, None],
        jnp.abs(patch_integrals - jnp.float64(1.0)),
        jnp.float64(0.0),
    )
    column_error = jnp.where(
        soil_column[:, None],
        jnp.abs(column_integrals - jnp.float64(1.0)),
        jnp.float64(0.0),
    )
    all_error = jnp.concatenate((patch_error.reshape(-1), column_error.reshape(-1)))
    maximum_error = (
        jnp.max(all_error) if all_error.size else jnp.float64(0.0)
    )
    return SoilBiogeochemVerticalProfileResult(
        state=updated,
        diagnostics=SoilBiogeochemVerticalProfileDiagnostics(
            updated_patch_count=jnp.sum(soil_patch, dtype=jnp.int32),
            updated_column_count=jnp.sum(soil_column, dtype=jnp.int32),
            fallback_patch_count=jnp.sum(fallback_patch, dtype=jnp.int32),
            fallback_column_count=jnp.sum(fallback_column, dtype=jnp.int32),
            profile_integral_violation_count=jnp.sum(
                all_error > jnp.float64(PROFILE_INTEGRAL_TOLERANCE),
                dtype=jnp.int32,
            ),
            maximum_profile_integral_error=maximum_error,
        ),
    )


__all__ = [
    "DEFAULT_MINIMUM_BEDROCK_DEPTH",
    "DEFAULT_NOT_VEGETATED_PFT",
    "DEFAULT_SOIL_BIOGEOCHEM_VERTICAL_PROFILE_PARAMETERS",
    "DEFAULT_SURFACE_PROFILE_EXPONENT",
    "NUMBER_OF_GROUND_LEVELS",
    "PROFILE_INTEGRAL_TOLERANCE",
    "SoilBiogeochemVerticalProfileDiagnostics",
    "SoilBiogeochemVerticalProfileInputs",
    "SoilBiogeochemVerticalProfileParameters",
    "SoilBiogeochemVerticalProfileResult",
    "SoilBiogeochemVerticalProfileState",
    "initialize_soil_biogeochem_vertical_profile_state",
    "soil_biogeochem_vertical_profile",
]
