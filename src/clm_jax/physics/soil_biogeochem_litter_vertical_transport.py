
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import lax, vmap

from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    get_cn_flux_field,
    set_cn_flux_field,
)
from clm_jax.physics.cn_zero_manifest import (
    NUMBER_OF_DECOMPOSITION_LEVELS,
    NUMBER_OF_DECOMPOSITION_POOLS,
)
from clm_jax.physics.numerics import (
    fortran_divide,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.tridiagonal import transport_thomas_solve as _tridiagonal_solve

NUMBER_OF_TRANSPORT_LEVELS = 20
NUMBER_OF_TRANSPORT_INTERFACES = NUMBER_OF_TRANSPORT_LEVELS + 1
NUMBER_OF_TRIDIAGONAL_LEVELS = NUMBER_OF_TRANSPORT_LEVELS + 2

_CARBON_SOURCE_SINK = "soil_carbon.decomp_cpools_sourcesink_col"
_CARBON_TRANSPORT_TENDENCY = (
    "soil_carbon.decomp_cpools_transport_tendency_col"
)
_NITROGEN_SOURCE_SINK = "soil_nitrogen.decomp_npools_sourcesink_col"
_NITROGEN_TRANSPORT_TENDENCY = (
    "soil_nitrogen.decomp_npools_transport_tendency_col"
)


def _soil_layer_geometry_20sl_8p5m() -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    thickness = np.empty(NUMBER_OF_DECOMPOSITION_LEVELS, dtype=np.float64)
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
    for level in range(21, NUMBER_OF_DECOMPOSITION_LEVELS + 1):
        increment = (
            (np.float64(level - 20) * np.float64(25.0)) ** np.float64(1.5)
        ) / np.float64(100.0)
        thickness[level - 1] = thickness[19] + increment

    interface = np.empty(NUMBER_OF_DECOMPOSITION_LEVELS + 1, dtype=np.float64)
    interface[0] = np.float64(0.0)
    running = np.float64(0.0)
    for level in range(NUMBER_OF_DECOMPOSITION_LEVELS):
        running = running + thickness[level]
        interface[level + 1] = running
    node = np.empty(NUMBER_OF_DECOMPOSITION_LEVELS, dtype=np.float64)
    for level in range(NUMBER_OF_DECOMPOSITION_LEVELS):
        node[level] = np.float64(0.5) * (
            interface[level] + interface[level + 1]
        )
    return node, interface, thickness


_DEFAULT_NODE_DEPTH, _DEFAULT_INTERFACE_DEPTH, _DEFAULT_LAYER_THICKNESS = (
    _soil_layer_geometry_20sl_8p5m()
)
_DEFAULT_CWD_POOL = np.asarray(
    [False, False, False, True, False, False, False],
    dtype=np.bool_,
)
_CWD_POOL_INDEX = 3
_NON_CWD_POOL_INDICES = np.asarray([0, 1, 2, 4, 5, 6], dtype=np.int32)
_NUMBER_OF_NON_CWD_POOLS = len(_NON_CWD_POOL_INDICES)


class SoilBiogeochemLitterVerticalTransportInputs(NamedTuple):

    soil_column: jnp.ndarray
    bedrock_level_count_column: jnp.ndarray
    maximum_annual_thaw_depth_column: jnp.ndarray
    previous_year_maximum_annual_thaw_depth_column: jnp.ndarray


class SoilBiogeochemLitterVerticalTransportState(NamedTuple):

    carbon_pool_column_level_pool: jnp.ndarray
    nitrogen_pool_column_level_pool: jnp.ndarray
    som_advective_coefficient_column_level: jnp.ndarray
    som_diffusivity_coefficient_column_level: jnp.ndarray


class SoilBiogeochemLitterVerticalTransportParameters(NamedTuple):

    time_step: float | jnp.ndarray = 1800.0
    som_advective_flux: float | jnp.ndarray = 0.0
    som_diffusivity: float | jnp.ndarray = np.float64(1.0e-4) / np.float64(
        86400.0 * 365.0
    )
    cryoturbation_diffusivity: float | jnp.ndarray = np.float64(
        5.0e-4
    ) / np.float64(86400.0 * 365.0)
    maximum_active_layer_depth_for_cryoturbation: float | jnp.ndarray = 2.0
    maximum_cryoturbation_depth: float | jnp.ndarray = 3.0
    epsilon: float | jnp.ndarray = np.float64(np.float32(1.0e-30))
    soil_node_depth_level: jnp.ndarray = _DEFAULT_NODE_DEPTH
    soil_interface_depth_level: jnp.ndarray = _DEFAULT_INTERFACE_DEPTH
    decomposition_layer_thickness_level: jnp.ndarray = _DEFAULT_LAYER_THICKNESS
    is_coarse_woody_debris_pool: jnp.ndarray = _DEFAULT_CWD_POOL


DEFAULT_SOIL_BIOGEOCHEM_LITTER_VERTICAL_TRANSPORT_PARAMETERS = (
    SoilBiogeochemLitterVerticalTransportParameters()
)


class SoilBiogeochemLitterVerticalTransportDiagnostics(NamedTuple):
    updated_column_count: jnp.ndarray
    cryoturbated_column_count: jnp.ndarray
    bioturbated_column_count: jnp.ndarray
    frozen_column_count: jnp.ndarray
    transported_pool_profile_count: jnp.ndarray


class SoilBiogeochemLitterVerticalTransportResult(NamedTuple):
    state: SoilBiogeochemLitterVerticalTransportState
    cn_fluxes: CnTransientFluxState
    diagnostics: SoilBiogeochemLitterVerticalTransportDiagnostics


def _validate_shapes(
    inputs: SoilBiogeochemLitterVerticalTransportInputs,
    state: SoilBiogeochemLitterVerticalTransportState,
    cn_fluxes: CnTransientFluxState,
    parameters: SoilBiogeochemLitterVerticalTransportParameters,
) -> int:
    if jnp.ndim(inputs.soil_column) != 1:
        raise ValueError("soil_column must be a vector")
    column_count = inputs.soil_column.shape[0]
    for name in (
        "bedrock_level_count_column",
        "maximum_annual_thaw_depth_column",
        "previous_year_maximum_annual_thaw_depth_column",
    ):
        if jnp.shape(getattr(inputs, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    pool_shape = (
        column_count,
        NUMBER_OF_DECOMPOSITION_LEVELS,
        NUMBER_OF_DECOMPOSITION_POOLS,
    )
    for name in (
        "carbon_pool_column_level_pool",
        "nitrogen_pool_column_level_pool",
    ):
        if jnp.shape(getattr(state, name)) != pool_shape:
            raise ValueError(f"{name} must have shape (column, 25, 7)")
    level_shape = (column_count, NUMBER_OF_DECOMPOSITION_LEVELS)
    for name in (
        "som_advective_coefficient_column_level",
        "som_diffusivity_coefficient_column_level",
    ):
        if jnp.shape(getattr(state, name)) != level_shape:
            raise ValueError(f"{name} must have shape (column, 25)")
    if cn_fluxes.column_level_pool.shape[:3] != pool_shape:
        raise ValueError("CN column-level pool fluxes have incompatible dimensions")
    for name in (
        "time_step",
        "som_advective_flux",
        "som_diffusivity",
        "cryoturbation_diffusivity",
        "maximum_active_layer_depth_for_cryoturbation",
        "maximum_cryoturbation_depth",
        "epsilon",
    ):
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    if jnp.shape(parameters.soil_node_depth_level) != (
        NUMBER_OF_DECOMPOSITION_LEVELS,
    ):
        raise ValueError("soil_node_depth_level must contain 25 levels")
    if jnp.shape(parameters.soil_interface_depth_level) != (
        NUMBER_OF_DECOMPOSITION_LEVELS + 1,
    ):
        raise ValueError("soil_interface_depth_level must contain 26 interfaces")
    if jnp.shape(parameters.decomposition_layer_thickness_level) != (
        NUMBER_OF_DECOMPOSITION_LEVELS,
    ):
        raise ValueError("decomposition_layer_thickness_level must contain 25 levels")
    if jnp.shape(parameters.is_coarse_woody_debris_pool) != (
        NUMBER_OF_DECOMPOSITION_POOLS,
    ):
        raise ValueError("is_coarse_woody_debris_pool must contain seven flags")
    return column_count


def _mixing_coefficients(
    inputs: SoilBiogeochemLitterVerticalTransportInputs,
    state: SoilBiogeochemLitterVerticalTransportState,
    parameters: SoilBiogeochemLitterVerticalTransportParameters,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    soil = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    bedrock = jnp.asarray(inputs.bedrock_level_count_column, dtype=jnp.int32)
    thaw_depth = jnp.maximum(
        jnp.asarray(inputs.maximum_annual_thaw_depth_column, dtype=jnp.float64),
        jnp.asarray(
            inputs.previous_year_maximum_annual_thaw_depth_column,
            dtype=jnp.float64,
        ),
    )
    maximum_active_layer = jnp.asarray(
        parameters.maximum_active_layer_depth_for_cryoturbation,
        dtype=jnp.float64,
    )
    cryoturbated = soil & (thaw_depth <= maximum_active_layer) & (
        thaw_depth > jnp.float64(0.0)
    )
    bioturbated = soil & (thaw_depth > maximum_active_layer)
    frozen = soil & (thaw_depth <= jnp.float64(0.0))

    interface = jnp.asarray(
        parameters.soil_interface_depth_level,
        dtype=jnp.float64,
    )
    transport_interface = interface[1 : NUMBER_OF_TRANSPORT_INTERFACES + 1]
    interface_number = jnp.arange(
        1,
        NUMBER_OF_TRANSPORT_INTERFACES + 1,
        dtype=jnp.int32,
    )
    above_bedrock = interface_number[None, :] <= bedrock[:, None] + jnp.int32(1)

    bottom_interface = interface[bedrock + jnp.int32(1)]
    mixing_bottom = jnp.minimum(
        jnp.asarray(parameters.maximum_cryoturbation_depth, dtype=jnp.float64),
        bottom_interface,
    )
    depth_below_thaw = fortran_add(
        transport_interface[None, :],
        -thaw_depth[:, None],
    )
    taper_depth = fortran_add(mixing_bottom, -thaw_depth)
    taper_fraction = fortran_divide(depth_below_thaw, taper_depth[:, None])
    taper = fortran_add(jnp.float64(1.0), -taper_fraction)
    tapered_diffusivity = jnp.maximum(
        fortran_multiply(
            jnp.asarray(parameters.cryoturbation_diffusivity, dtype=jnp.float64),
            taper,
        ),
        jnp.float64(0.0),
    )
    cryoturbation_profile = jnp.where(
        transport_interface[None, :] < thaw_depth[:, None],
        jnp.asarray(parameters.cryoturbation_diffusivity, dtype=jnp.float64),
        tapered_diffusivity,
    )
    new_diffusivity = jnp.where(
        cryoturbated[:, None],
        cryoturbation_profile,
        jnp.where(
            bioturbated[:, None],
            jnp.asarray(parameters.som_diffusivity, dtype=jnp.float64),
            jnp.float64(0.0),
        ),
    )
    new_diffusivity = jnp.where(above_bedrock, new_diffusivity, jnp.float64(0.0))
    new_advection = jnp.where(
        bioturbated[:, None] & above_bedrock,
        jnp.asarray(parameters.som_advective_flux, dtype=jnp.float64),
        jnp.float64(0.0),
    )

    coefficient_mask = soil[:, None]
    advection = state.som_advective_coefficient_column_level.at[
        :, :NUMBER_OF_TRANSPORT_INTERFACES
    ].set(
        jnp.where(
            coefficient_mask,
            new_advection,
            state.som_advective_coefficient_column_level[
                :, :NUMBER_OF_TRANSPORT_INTERFACES
            ],
        )
    )
    diffusivity = state.som_diffusivity_coefficient_column_level.at[
        :, :NUMBER_OF_TRANSPORT_INTERFACES
    ].set(
        jnp.where(
            coefficient_mask,
            new_diffusivity,
            state.som_diffusivity_coefficient_column_level[
                :, :NUMBER_OF_TRANSPORT_INTERFACES
            ],
        )
    )
    return advection, diffusivity, cryoturbated, bioturbated, frozen


def _harmonic_diffusivity(
    first: jnp.ndarray,
    second: jnp.ndarray,
    second_weight: jnp.ndarray,
) -> jnp.ndarray:
    first_term = fortran_divide(
        fortran_add(jnp.float64(1.0), -second_weight),
        first,
    )
    second_term = fortran_divide(second_weight, second)
    return fortran_divide(
        jnp.float64(1.0),
        fortran_add(first_term, second_term),
    )


def _patankar_a(peclet: jnp.ndarray) -> jnp.ndarray:
    base = fortran_add(
        jnp.float64(1.0),
        -fortran_multiply(jnp.float64(0.1), jnp.abs(peclet)),
    )
    square = fortran_multiply(base, base)
    fourth = fortran_multiply(square, square)
    fifth = fortran_multiply(fourth, base)
    return jnp.maximum(jnp.float64(0.0), fifth)


def _transport_matrix(
    bedrock_level_count_column: jnp.ndarray,
    advection_coefficient: jnp.ndarray,
    diffusivity_coefficient: jnp.ndarray,
    parameters: SoilBiogeochemLitterVerticalTransportParameters,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    column_count = bedrock_level_count_column.shape[0]
    epsilon = jnp.asarray(parameters.epsilon, dtype=jnp.float64)
    advection = jnp.where(
        jnp.abs(advection_coefficient[:, :NUMBER_OF_TRANSPORT_INTERFACES])
        < epsilon,
        epsilon,
        advection_coefficient[:, :NUMBER_OF_TRANSPORT_INTERFACES],
    )
    diffusivity = jnp.where(
        jnp.abs(diffusivity_coefficient[:, :NUMBER_OF_TRANSPORT_INTERFACES])
        < epsilon,
        epsilon,
        diffusivity_coefficient[:, :NUMBER_OF_TRANSPORT_INTERFACES],
    )
    node = jnp.asarray(parameters.soil_node_depth_level, dtype=jnp.float64)
    interface = jnp.asarray(
        parameters.soil_interface_depth_level,
        dtype=jnp.float64,
    )
    thickness = jnp.asarray(
        parameters.decomposition_layer_thickness_level,
        dtype=jnp.float64,
    )
    node_spacing = jnp.empty((NUMBER_OF_TRANSPORT_INTERFACES,), dtype=jnp.float64)
    node_spacing = node_spacing.at[0].set(node[0])
    node_spacing = node_spacing.at[1:].set(
        fortran_add(
            node[1:NUMBER_OF_TRANSPORT_INTERFACES],
            -node[: NUMBER_OF_TRANSPORT_INTERFACES - 1],
        )
    )

    d_previous = jnp.zeros(
        (column_count, NUMBER_OF_TRANSPORT_INTERFACES),
        dtype=jnp.float64,
    )
    d_next = jnp.zeros_like(d_previous)
    flux_previous = jnp.zeros_like(d_previous)
    flux_next = jnp.zeros_like(d_previous)
    peclet_previous = jnp.zeros_like(d_previous)
    peclet_next = jnp.zeros_like(d_previous)

    next_weight = fortran_divide(
        fortran_add(node[1], -interface[1]),
        node_spacing[1],
    )
    next_diffusivity = _harmonic_diffusivity(
        diffusivity[:, 0], diffusivity[:, 1], next_weight
    )
    next_over_depth = fortran_divide(next_diffusivity, node_spacing[1])
    d_next = d_next.at[:, 0].set(next_over_depth)
    flux_previous = flux_previous.at[:, 0].set(advection[:, 0])
    flux_next = flux_next.at[:, 0].set(advection[:, 1])
    peclet_next = peclet_next.at[:, 0].set(
        fortran_divide(advection[:, 1], next_over_depth)
    )

    def interface_step(level):
        previous_weight = fortran_divide(
            fortran_add(interface[level], -node[level - 1]),
            node_spacing[level],
        )
        previous_diffusivity = _harmonic_diffusivity(
            diffusivity[:, level],
            diffusivity[:, level - 1],
            previous_weight,
        )
        previous_over_depth = fortran_divide(
            previous_diffusivity, node_spacing[level]
        )
        previous_flux = advection[:, level]
        previous_peclet = fortran_divide(previous_flux, previous_over_depth)

        next_level = jnp.minimum(
            level + jnp.int32(1),
            jnp.int32(NUMBER_OF_TRANSPORT_INTERFACES - 1),
        )
        next_weight = fortran_divide(
            fortran_add(node[next_level], -interface[next_level]),
            node_spacing[next_level],
        )
        harmonic_next = _harmonic_diffusivity(
            diffusivity[:, level],
            diffusivity[:, next_level],
            next_weight,
        )
        next_over_depth_interior = fortran_divide(
            harmonic_next, node_spacing[next_level]
        )
        next_flux_interior = advection[:, next_level]
        next_peclet_interior = fortran_divide(
            next_flux_interior, next_over_depth_interior
        )
        bottom_or_below = level >= bedrock_level_count_column
        next_over_depth = jnp.where(
            bottom_or_below,
            previous_over_depth,
            next_over_depth_interior,
        )
        next_flux = jnp.where(
            bottom_or_below,
            jnp.float64(0.0),
            next_flux_interior,
        )
        next_peclet = jnp.where(
            bottom_or_below,
            fortran_divide(jnp.float64(0.0), previous_over_depth),
            next_peclet_interior,
        )
        return (
            previous_over_depth, next_over_depth, previous_flux,
            next_flux, previous_peclet, next_peclet,
        )

    interior = vmap(interface_step, out_axes=1)(
        jnp.arange(1, NUMBER_OF_TRANSPORT_INTERFACES, dtype=jnp.int32)
    )
    d_previous, d_next, flux_previous, flux_next, peclet_previous, peclet_next = (
        previous.at[:, 1:].set(updated)
        for previous, updated in zip(
            (d_previous, d_next, flux_previous, flux_next, peclet_previous, peclet_next),
            interior,
            strict=True,
        )
    )

    lower = jnp.zeros(
        (column_count, NUMBER_OF_TRIDIAGONAL_LEVELS),
        dtype=jnp.float64,
    )
    diagonal = jnp.zeros_like(lower)
    upper = jnp.zeros_like(lower)
    diagonal = diagonal.at[:, 0].set(jnp.float64(1.0))
    upper = upper.at[:, 0].set(jnp.float64(-1.0))
    time_step = jnp.asarray(parameters.time_step, dtype=jnp.float64)

    def coefficient_step(level):
        storage_value = fortran_divide(thickness[level], time_step)
        previous_term = fortran_add(
            fortran_multiply(
                d_previous[:, level],
                _patankar_a(peclet_previous[:, level]),
            ),
            jnp.maximum(flux_previous[:, level], jnp.float64(0.0)),
        )
        next_term = fortran_add(
            fortran_multiply(
                d_next[:, level],
                _patankar_a(peclet_next[:, level]),
            ),
            jnp.maximum(-flux_next[:, level], jnp.float64(0.0)),
        )
        lower_value = -previous_term
        upper_value = -next_term
        diagonal_value = fortran_add(
            fortran_add(-lower_value, -upper_value),
            storage_value,
        )
        return lower_value, diagonal_value, upper_value, storage_value

    lower_active, diagonal_active, upper_active, storage = vmap(
        coefficient_step, out_axes=(1, 1, 1, 0)
    )(jnp.arange(NUMBER_OF_TRANSPORT_LEVELS, dtype=jnp.int32))
    lower = lower.at[:, 1:-1].set(lower_active)
    diagonal = diagonal.at[:, 1:-1].set(diagonal_active)
    upper = upper.at[:, 1:-1].set(upper_active)
    lower = lower.at[:, -1].set(jnp.float64(-1.0))
    diagonal = diagonal.at[:, -1].set(jnp.float64(1.0))
    return lower, diagonal, upper, storage




def _right_hand_side(
    concentration: jnp.ndarray,
    source_sink: jnp.ndarray,
    storage: jnp.ndarray,
    advection: jnp.ndarray,
    parameters: SoilBiogeochemLitterVerticalTransportParameters,
) -> jnp.ndarray:
    thickness = jnp.asarray(
        parameters.decomposition_layer_thickness_level,
        dtype=jnp.float64,
    )
    time_step = jnp.asarray(parameters.time_step, dtype=jnp.float64)
    levels = NUMBER_OF_TRANSPORT_LEVELS
    source_per_depth_time = fortran_divide(
        fortran_multiply(source_sink[..., :levels], thickness[:levels]),
        time_step,
    )
    concentration_coefficient = jnp.where(
        jnp.arange(levels) == 0,
        fortran_add(storage[:levels], -advection[..., :levels]),
        storage[:levels],
    )
    interior = fortran_add(
        source_per_depth_time,
        fortran_multiply(concentration_coefficient, concentration[..., 1 : levels + 1]),
    )
    return jnp.pad(interior, ((0, 0),) * (interior.ndim - 1) + ((1, 1),))


def _truncate_at_bedrock(
    pool_state: jnp.ndarray,
    concentration: jnp.ndarray,
    soil_column: jnp.ndarray,
    bedrock_level_count_column: jnp.ndarray,
    layer_thickness: jnp.ndarray,
) -> jnp.ndarray:

    row = jnp.arange(pool_state.shape[1], dtype=jnp.int32)
    bedrock_index = bedrock_level_count_column - jnp.int32(1)

    bedrock_value = pool_state[:, row, bedrock_index, :]
    copied_bedrock = concentration[
        :, row, jnp.minimum(bedrock_index, NUMBER_OF_TRANSPORT_LEVELS - 1), :
    ]
    bedrock_value = jnp.where(
        (soil_column & (bedrock_level_count_column <= NUMBER_OF_TRANSPORT_LEVELS))[
            None, :, None
        ],
        copied_bedrock,
        bedrock_value,
    )

    def level_step(current, row_values):
        level, layer_concentration = row_values
        below_bedrock = soil_column & (level >= bedrock_level_count_column)
        thickness_ratio = fortran_divide(
            layer_thickness[level], layer_thickness[bedrock_index]
        )
        returned_mass = fortran_multiply(
            layer_concentration, thickness_ratio[None, :, None]
        )
        updated = jnp.where(
            below_bedrock[None, :, None], fortran_add(current, returned_mass), current
        )
        return updated, None

    level = jnp.arange(NUMBER_OF_TRANSPORT_LEVELS, dtype=jnp.int32)
    bedrock_value, _ = lax.scan(
        level_step, bedrock_value, (level, jnp.moveaxis(concentration, 2, 0))
    )
    copied = jnp.where(
        soil_column[None, :, None, None],
        jnp.where(
            (level[None, :] >= bedrock_level_count_column[:, None])[None, :, :, None],
            jnp.float64(0.0),
            concentration,
        ),
        pool_state[:, :, :NUMBER_OF_TRANSPORT_LEVELS, :],
    )
    copied = jnp.concatenate(
        (copied, pool_state[:, :, NUMBER_OF_TRANSPORT_LEVELS:, :]), axis=2
    )
    destination = jnp.where(bedrock_index < 0, bedrock_index + pool_state.shape[2], bedrock_index)
    bedrock_layer = jnp.arange(pool_state.shape[2])[None, :] == destination[:, None]
    return jnp.where(bedrock_layer[None, :, :, None], bedrock_value[:, :, None, :], copied)


def _transport_both_element_types(
    pool_state: jnp.ndarray,
    source_sink: jnp.ndarray,
    transport_tendency: jnp.ndarray,
    soil_column: jnp.ndarray,
    bedrock_level_count_column: jnp.ndarray,
    advection: jnp.ndarray,
    lower: jnp.ndarray,
    diagonal: jnp.ndarray,
    upper: jnp.ndarray,
    storage: jnp.ndarray,
    parameters: SoilBiogeochemLitterVerticalTransportParameters,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    layer_thickness = jnp.asarray(
        parameters.decomposition_layer_thickness_level,
        dtype=jnp.float64,
    )
    time_step = jnp.asarray(parameters.time_step, dtype=jnp.float64)
    element_type_count, column_count = pool_state.shape[:2]

    selected_state = jnp.take(
        pool_state[:, :, :NUMBER_OF_TRANSPORT_INTERFACES, :],
        _NON_CWD_POOL_INDICES,
        axis=3,
    ).transpose(0, 1, 3, 2)
    selected_source = jnp.take(
        source_sink,
        _NON_CWD_POOL_INDICES,
        axis=3,
    ).transpose(0, 1, 3, 2)
    profile_shape = (
        element_type_count,
        column_count,
        _NUMBER_OF_NON_CWD_POOLS,
    )
    concentration = jnp.zeros(
        profile_shape + (NUMBER_OF_TRIDIAGONAL_LEVELS,),
        dtype=jnp.float64,
    )
    concentration = concentration.at[..., 1:].set(selected_state)
    right_hand_side = _right_hand_side(
        concentration,
        selected_source,
        storage,
        advection[None, :, None, :],
        parameters,
    )
    initial_tendency = -fortran_add(
        concentration[..., 1 : NUMBER_OF_TRANSPORT_LEVELS + 1],
        selected_source[..., :NUMBER_OF_TRANSPORT_LEVELS],
    )
    solved = _tridiagonal_solve(
        lower[None, :, None, :],
        diagonal[None, :, None, :],
        upper[None, :, None, :],
        right_hand_side,
        concentration,
    )
    final_tendency = fortran_divide(
        fortran_add(
            initial_tendency,
            solved[..., 1 : NUMBER_OF_TRANSPORT_LEVELS + 1],
        ),
        time_step,
    )
    solved_level_pool = solved[
        ..., 1 : NUMBER_OF_TRANSPORT_LEVELS + 1
    ].transpose(0, 1, 3, 2)
    final_tendency = final_tendency.transpose(0, 1, 3, 2)

    cwd_concentration = fortran_add(
        pool_state[:, :, :NUMBER_OF_TRANSPORT_LEVELS, _CWD_POOL_INDEX],
        source_sink[:, :, :NUMBER_OF_TRANSPORT_LEVELS, _CWD_POOL_INDEX],
    )
    tendency_fields, concentration_fields = [], []
    for pool in range(NUMBER_OF_DECOMPOSITION_POOLS):
        old = transport_tendency[:, :, :NUMBER_OF_TRANSPORT_LEVELS, pool]
        if pool == _CWD_POOL_INDEX:
            tendency_fields.append(old)
            concentration_fields.append(cwd_concentration)
        else:
            source_pool = pool if pool < _CWD_POOL_INDEX else pool - 1
            tendency_fields.append(jnp.where(
                soil_column[None, :, None], final_tendency[..., source_pool], old
            ))
            concentration_fields.append(solved_level_pool[..., source_pool])
    transport_tendency = jnp.concatenate(
        (jnp.stack(tendency_fields, axis=-1),
         transport_tendency[:, :, NUMBER_OF_TRANSPORT_LEVELS:, :]), axis=2,
    )
    concentration_level_pool = jnp.stack(concentration_fields, axis=-1)
    pool_state = _truncate_at_bedrock(
        pool_state,
        concentration_level_pool,
        soil_column,
        bedrock_level_count_column,
        layer_thickness,
    )
    return pool_state, transport_tendency


def soil_biogeochem_litter_vertical_transport(
    inputs: SoilBiogeochemLitterVerticalTransportInputs,
    state: SoilBiogeochemLitterVerticalTransportState,
    cn_fluxes: CnTransientFluxState,
    parameters: SoilBiogeochemLitterVerticalTransportParameters = (
        DEFAULT_SOIL_BIOGEOCHEM_LITTER_VERTICAL_TRANSPORT_PARAMETERS
    ),
) -> SoilBiogeochemLitterVerticalTransportResult:

    _validate_shapes(inputs, state, cn_fluxes, parameters)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    bedrock = jnp.asarray(inputs.bedrock_level_count_column, dtype=jnp.int32)
    advection_coefficient, diffusivity_coefficient, cryoturbated, bioturbated, frozen = (
        _mixing_coefficients(inputs, state, parameters)
    )
    epsilon = jnp.asarray(parameters.epsilon, dtype=jnp.float64)
    local_advection = jnp.where(
        jnp.abs(advection_coefficient[:, :NUMBER_OF_TRANSPORT_INTERFACES])
        < epsilon,
        epsilon,
        advection_coefficient[:, :NUMBER_OF_TRANSPORT_INTERFACES],
    )
    lower, diagonal, upper, storage = _transport_matrix(
        bedrock,
        advection_coefficient,
        diffusivity_coefficient,
        parameters,
    )

    carbon_source = get_cn_flux_field(cn_fluxes, _CARBON_SOURCE_SINK)
    nitrogen_source = get_cn_flux_field(cn_fluxes, _NITROGEN_SOURCE_SINK)
    carbon_tendency = get_cn_flux_field(cn_fluxes, _CARBON_TRANSPORT_TENDENCY)
    nitrogen_tendency = get_cn_flux_field(cn_fluxes, _NITROGEN_TRANSPORT_TENDENCY)
    pools, tendencies = _transport_both_element_types(
        jnp.stack(
            (
                jnp.asarray(
                    state.carbon_pool_column_level_pool,
                    dtype=jnp.float64,
                ),
                jnp.asarray(
                    state.nitrogen_pool_column_level_pool,
                    dtype=jnp.float64,
                ),
            ),
            axis=0,
        ),
        jnp.stack((carbon_source, nitrogen_source), axis=0),
        jnp.stack((carbon_tendency, nitrogen_tendency), axis=0),
        soil_column,
        bedrock,
        local_advection,
        lower,
        diagonal,
        upper,
        storage,
        parameters,
    )
    carbon_pool, nitrogen_pool = pools
    carbon_tendency, nitrogen_tendency = tendencies
    cn_fluxes = set_cn_flux_field(
        cn_fluxes, _CARBON_TRANSPORT_TENDENCY, carbon_tendency
    )
    cn_fluxes = set_cn_flux_field(
        cn_fluxes, _NITROGEN_TRANSPORT_TENDENCY, nitrogen_tendency
    )
    non_cwd_count = jnp.sum(
        ~jnp.asarray(parameters.is_coarse_woody_debris_pool, dtype=jnp.bool_),
        dtype=jnp.int32,
    )
    updated_column_count = jnp.sum(soil_column, dtype=jnp.int32)
    return SoilBiogeochemLitterVerticalTransportResult(
        state=SoilBiogeochemLitterVerticalTransportState(
            carbon_pool_column_level_pool=carbon_pool,
            nitrogen_pool_column_level_pool=nitrogen_pool,
            som_advective_coefficient_column_level=advection_coefficient,
            som_diffusivity_coefficient_column_level=diffusivity_coefficient,
        ),
        cn_fluxes=cn_fluxes,
        diagnostics=SoilBiogeochemLitterVerticalTransportDiagnostics(
            updated_column_count=updated_column_count,
            cryoturbated_column_count=jnp.sum(cryoturbated, dtype=jnp.int32),
            bioturbated_column_count=jnp.sum(bioturbated, dtype=jnp.int32),
            frozen_column_count=jnp.sum(frozen, dtype=jnp.int32),
            transported_pool_profile_count=(
                updated_column_count * jnp.int32(2) * non_cwd_count
            ),
        ),
    )
