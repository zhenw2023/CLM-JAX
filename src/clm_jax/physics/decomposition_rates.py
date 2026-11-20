
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import default_backend, lax, tree

from clm_jax.constants import FREEZING_TEMPERATURE_FRESHWATER
from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    get_cn_flux_field,
    set_cn_flux_field,
)
from clm_jax.physics.numerics import (
    fortran_divide,
    strict_exp,
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
from clm_jax.physics.native_power import native_power
from clm_jax.physics.native_arctangent import native_atan
from clm_jax.physics.native_logarithm import native_log

NUMBER_OF_GROUND_LEVELS = 25
NUMBER_OF_SOIL_LEVELS = 20
NUMBER_OF_STANDARD_SOIL_DECOMPOSITION_LEVELS = 5
SECONDS_PER_DAY = 86_400.0
PI = np.float64(np.pi)

_DECOMPOSITION_RATE_FIELD = "soil_carbon.decomp_k_col"


class DecompositionRateInputs(NamedTuple):

    soil_column: jnp.ndarray
    soil_water_potential_column_level: jnp.ndarray
    soil_temperature_column_level: jnp.ndarray
    oxygen_stress_saturated_column_level: jnp.ndarray
    oxygen_stress_unsaturated_column_level: jnp.ndarray
    inundated_fraction_column: jnp.ndarray
    thaw_depth_index_column: jnp.ndarray
    soil_layer_thickness_column_level: jnp.ndarray
    soil_node_depth_level: jnp.ndarray
    latitude_degrees_column: jnp.ndarray


class DecompositionRateParameters(NamedTuple):

    minimum_water_potential_mpa: float | jnp.ndarray = -2.0
    maximum_water_potential_mpa: float | jnp.ndarray = -0.002
    temperature_q10: float | jnp.ndarray = 1.5
    frozen_temperature_q10: float | jnp.ndarray = 1.5
    decomposition_depth_efolding_m: float | jnp.ndarray = 10.0
    minimum_oxygen_scalar: float | jnp.ndarray = 0.2
    days_per_year: float | jnp.ndarray = 365.0
    normalization_reference_celsius: float | jnp.ndarray = 15.0
    decomposition_level_count: int | jnp.ndarray = NUMBER_OF_SOIL_LEVELS
    use_century_temperature_function: bool | jnp.ndarray = False
    normalize_q10_to_century: bool | jnp.ndarray = True
    use_vertical_soil_carbon: bool | jnp.ndarray = True
    use_methane: bool | jnp.ndarray = True
    use_anoxia: bool | jnp.ndarray = True
    anoxia_uses_saturated_fraction: bool | jnp.ndarray = False
    use_fates: bool | jnp.ndarray = False
    spinup_state: int | jnp.ndarray = 0
    spinup_factor: jnp.ndarray = np.asarray(
        [1.0, 1.0, 1.0, 1.6666666269302368, 1.0, 5.0, 222.22222900390625],
        dtype=np.float64,
    )


DEFAULT_DECOMPOSITION_RATE_PARAMETERS = DecompositionRateParameters()


class DecompositionScalarState(NamedTuple):

    temperature_scalar_column_level: jnp.ndarray
    water_scalar_column_level: jnp.ndarray
    oxygen_scalar_column_level: jnp.ndarray


class DecompositionRateDiagnostics(NamedTuple):
    depth_scalar_column_level: jnp.ndarray
    latitude_spinup_term_column: jnp.ndarray
    base_rate_pool: jnp.ndarray
    updated_column_count: jnp.ndarray


class DecompositionRateResult(NamedTuple):
    cn_fluxes: CnTransientFluxState
    scalars: DecompositionScalarState
    diagnostics: DecompositionRateDiagnostics


def initialize_decomposition_scalar_state(
    number_of_columns: int,
    *,
    fill_value: float = 1.0e36,
) -> DecompositionScalarState:

    if number_of_columns < 0:
        raise ValueError("column count cannot be negative")
    shape = (number_of_columns, NUMBER_OF_GROUND_LEVELS)
    return DecompositionScalarState(
        temperature_scalar_column_level=jnp.full(shape, fill_value, dtype=jnp.float64),
        water_scalar_column_level=jnp.full(shape, fill_value, dtype=jnp.float64),
        oxygen_scalar_column_level=jnp.full(shape, fill_value, dtype=jnp.float64),
    )


def _elementwise_log(value: jnp.ndarray) -> jnp.ndarray:
    value = jnp.asarray(value, dtype=jnp.float64)
    if default_backend() == "gpu":
        return native_log(value)
    shape = value.shape
    return jnp.reshape(scalar_map(jnp.log, jnp.reshape(value, (-1,))), shape)


def _elementwise_atan(value: jnp.ndarray) -> jnp.ndarray:
    value = jnp.asarray(value, dtype=jnp.float64)
    if default_backend() == "gpu":
        return native_atan(value)
    shape = value.shape
    return jnp.reshape(scalar_map(jnp.arctan, jnp.reshape(value, (-1,))), shape)


def _century_temperature_function(temperature_celsius: jnp.ndarray) -> jnp.ndarray:
    argument = fortran_multiply(
        fortran_multiply(jnp.float64(PI), jnp.float64(0.031)),
        fortran_add(temperature_celsius, -jnp.float64(15.4)),
    )
    return fortran_add(
        jnp.float64(11.75),
        fortran_multiply(
            fortran_divide(jnp.float64(29.7), jnp.float64(PI)),
            _elementwise_atan(argument),
        ),
    )


def _multiply_chain(first: jnp.ndarray, *rest: jnp.ndarray) -> jnp.ndarray:
    result = jnp.asarray(first, dtype=jnp.float64)
    for factor in rest:
        result = fortran_multiply(result, factor)
    return result


def _validate_shapes(
    inputs: DecompositionRateInputs,
    cn_fluxes: CnTransientFluxState,
    scalars: DecompositionScalarState,
) -> int:
    if jnp.ndim(inputs.soil_column) != 1:
        raise ValueError("soil_column must be a vector")
    number_of_columns = inputs.soil_column.shape[0]
    level_shape = (number_of_columns, NUMBER_OF_GROUND_LEVELS)
    for name in (
        "soil_water_potential_column_level",
        "soil_temperature_column_level",
        "oxygen_stress_saturated_column_level",
        "oxygen_stress_unsaturated_column_level",
        "soil_layer_thickness_column_level",
    ):
        if jnp.shape(getattr(inputs, name)) != level_shape:
            raise ValueError(f"{name} must have shape (column, 25)")
    for name in (
        "inundated_fraction_column",
        "thaw_depth_index_column",
        "latitude_degrees_column",
    ):
        if jnp.shape(getattr(inputs, name)) != (number_of_columns,):
            raise ValueError(f"{name} must contain every column")
    if jnp.shape(inputs.soil_node_depth_level) != (NUMBER_OF_GROUND_LEVELS,):
        raise ValueError("soil_node_depth_level must contain 25 levels")
    for name in DecompositionScalarState._fields:
        if jnp.shape(getattr(scalars, name)) != level_shape:
            raise ValueError(f"{name} must have shape (column, 25)")
    if get_cn_flux_field(cn_fluxes, _DECOMPOSITION_RATE_FIELD).shape != (
        number_of_columns,
        NUMBER_OF_GROUND_LEVELS,
        10,
    ):
        raise ValueError("decomposition-rate flux state must have shape (column, 25, 10)")
    return number_of_columns


def _validate_scalar_parameters(parameters: DecompositionRateParameters) -> None:
    scalar_names = tuple(DecompositionRateParameters._fields[:-1])
    for name in scalar_names:
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    if jnp.ndim(parameters.spinup_factor) != 1 or parameters.spinup_factor.shape[0] < 7:
        raise ValueError("spinup_factor must contain at least seven pool values")
    century = parameters.use_century_temperature_function
    normalize = parameters.normalize_q10_to_century
    if isinstance(century, (bool, np.bool_)) and isinstance(normalize, (bool, np.bool_)):
        if century and normalize:
            raise ValueError(
                "CENTURY temperature function and Q10-to-CENTURY normalization are incompatible"
            )
    level_count = parameters.decomposition_level_count
    if isinstance(level_count, (int, np.integer)) and level_count not in (1, NUMBER_OF_SOIL_LEVELS):
        raise ValueError("decomposition_level_count must be 1 or 20")


def _temperature_q10(
    temperature: jnp.ndarray,
    q10: jnp.ndarray,
    frozen_q10: jnp.ndarray,
) -> jnp.ndarray:
    freezing = jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)
    warm_exponent = fortran_divide(
        fortran_add(
            temperature,
            -fortran_add(freezing, jnp.float64(25.0)),
        ),
        jnp.float64(10.0),
    )
    frozen_exponent = fortran_divide(
        fortran_add(temperature, -freezing),
        jnp.float64(10.0),
    )
    power = native_power if default_backend() == "gpu" else fortran_power
    warm = temperature >= freezing
    selected_power = power(jnp.where(warm, q10, frozen_q10),
                           jnp.where(warm, warm_exponent, frozen_exponent))
    frozen = fortran_multiply(
        power(q10, jnp.float64(-2.5)),
        selected_power,
    )
    return jnp.where(warm, selected_power, frozen)


def _water_scalar(
    water_potential: jnp.ndarray,
    minimum: jnp.ndarray,
    maximum: jnp.ndarray,
) -> jnp.ndarray:
    psi = jnp.minimum(water_potential, maximum)
    active = psi > minimum
    safe_psi = jnp.where(active, psi, maximum)
    numerator = _elementwise_log(fortran_divide(minimum, safe_psi))
    denominator = _elementwise_log(fortran_divide(minimum, maximum))
    return jnp.where(active, fortran_divide(numerator, denominator), jnp.float64(0.0))


def _weighted_one_level_scalars(
    inputs: DecompositionRateInputs,
    scalars: DecompositionScalarState,
    parameters: DecompositionRateParameters,
    soil_column: jnp.ndarray,
) -> DecompositionScalarState:
    temperature = jnp.where(soil_column[:, None], inputs.soil_temperature_column_level,
                            jnp.float64(FREEZING_TEMPERATURE_FRESHWATER))
    water_potential = jnp.asarray(
        inputs.soil_water_potential_column_level,
        dtype=jnp.float64,
    )
    saturated_oxygen = jnp.asarray(
        inputs.oxygen_stress_saturated_column_level,
        dtype=jnp.float64,
    )
    unsaturated_oxygen = jnp.asarray(
        inputs.oxygen_stress_unsaturated_column_level,
        dtype=jnp.float64,
    )
    inundated = jnp.asarray(inputs.inundated_fraction_column, dtype=jnp.float64)
    thickness = jnp.asarray(inputs.soil_layer_thickness_column_level, dtype=jnp.float64)

    def add_thickness(total: jnp.ndarray, level_thickness: jnp.ndarray):
        candidate = fortran_add(total, level_thickness)
        return jnp.where(soil_column, candidate, total), None

    total_thickness, _ = lax.scan(
        add_thickness,
        jnp.zeros((soil_column.shape[0],), dtype=jnp.float64),
        jnp.swapaxes(thickness[:, :NUMBER_OF_STANDARD_SOIL_DECOMPOSITION_LEVELS], 0, 1),
    )
    safe_total = jnp.where(total_thickness != 0.0, total_thickness, jnp.float64(1.0))
    weights = fortran_divide(
        thickness[:, :NUMBER_OF_STANDARD_SOIL_DECOMPOSITION_LEVELS],
        safe_total[:, None],
    )
    weights = jnp.where(total_thickness[:, None] != 0.0, weights, jnp.float64(0.0))

    q10 = jnp.asarray(parameters.temperature_q10, dtype=jnp.float64)
    frozen_q10 = jnp.asarray(parameters.frozen_temperature_q10, dtype=jnp.float64)
    q10_temperature = _temperature_q10(temperature[:, :5], q10, frozen_q10)
    celsius = fortran_add(
        temperature[:, :5],
        -jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
    )
    century_30 = _century_temperature_function(jnp.float64(30.0))
    century_ratio = fortran_divide(_century_temperature_function(celsius), century_30)
    century_contribution = jnp.maximum(
        fortran_multiply(century_ratio, weights),
        jnp.float64(0.01),
    )
    q10_contribution = fortran_multiply(q10_temperature, weights)
    temperature_contribution = lax.cond(
        jnp.asarray(parameters.use_century_temperature_function, dtype=jnp.bool_),
        lambda _: century_contribution,
        lambda _: q10_contribution,
        operand=None,
    )
    water_contribution = fortran_multiply(
        _water_scalar(
            water_potential[:, :5],
            jnp.asarray(parameters.minimum_water_potential_mpa, dtype=jnp.float64),
            jnp.asarray(parameters.maximum_water_potential_mpa, dtype=jnp.float64),
        ),
        weights,
    )

    minimum_oxygen = jnp.asarray(parameters.minimum_oxygen_scalar, dtype=jnp.float64)
    oxygen_unsaturated = jnp.maximum(unsaturated_oxygen[:, :5], minimum_oxygen)
    oxygen_saturated = jnp.maximum(saturated_oxygen[:, :5], minimum_oxygen)
    oxygen_mixed = fortran_add(
        fortran_multiply(oxygen_unsaturated, fortran_add(1.0, -inundated[:, None])),
        fortran_multiply(oxygen_saturated, inundated[:, None]),
    )
    oxygen_by_level = lax.cond(
        jnp.asarray(parameters.anoxia_uses_saturated_fraction, dtype=jnp.bool_),
        lambda _: oxygen_mixed,
        lambda _: oxygen_unsaturated,
        operand=None,
    )
    oxygen_contribution = fortran_multiply(weights, oxygen_by_level)

    initial_temperature = jnp.where(
        soil_column[:, None],
        jnp.float64(0.0),
        jnp.asarray(scalars.temperature_scalar_column_level, dtype=jnp.float64),
    )
    initial_water = jnp.where(
        soil_column[:, None],
        jnp.float64(0.0),
        jnp.asarray(scalars.water_scalar_column_level, dtype=jnp.float64),
    )
    initial_oxygen_filtered = jnp.where(
        soil_column[:, None],
        jnp.float64(0.0),
        jnp.asarray(scalars.oxygen_scalar_column_level, dtype=jnp.float64),
    )

    def accumulate(
        state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        contribution: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ):
        current_temperature, current_water, current_oxygen = state
        temperature_part, water_part, oxygen_part = contribution
        next_temperature_zero = fortran_add(current_temperature[:, 0], temperature_part)
        next_water_zero = fortran_add(current_water[:, 0], water_part)
        next_oxygen_zero = fortran_add(current_oxygen[:, 0], oxygen_part)
        current_temperature = current_temperature.at[:, 0].set(
            jnp.where(soil_column, next_temperature_zero, current_temperature[:, 0])
        )
        current_water = current_water.at[:, 0].set(
            jnp.where(soil_column, next_water_zero, current_water[:, 0])
        )
        current_oxygen = current_oxygen.at[:, 0].set(
            jnp.where(soil_column, next_oxygen_zero, current_oxygen[:, 0])
        )
        return (current_temperature, current_water, current_oxygen), None

    (temperature_scalar, water_scalar, oxygen_filtered), _ = lax.scan(
        accumulate,
        (initial_temperature, initial_water, initial_oxygen_filtered),
        (
            jnp.swapaxes(temperature_contribution, 0, 1),
            jnp.swapaxes(water_contribution, 0, 1),
            jnp.swapaxes(oxygen_contribution, 0, 1),
        ),
    )

    saturated_water = fortran_add(
        fortran_multiply(water_scalar[:, 0], fortran_add(1.0, -inundated)),
        inundated,
    )
    adjust_saturated = (
        jnp.asarray(parameters.use_methane, dtype=jnp.bool_)
        & jnp.asarray(parameters.anoxia_uses_saturated_fraction, dtype=jnp.bool_)
        & (jnp.asarray(inputs.thaw_depth_index_column, dtype=jnp.int32) >= 5)
        & (temperature[:, 0] > jnp.float64(FREEZING_TEMPERATURE_FRESHWATER))
        & soil_column
    )
    water_scalar = water_scalar.at[:, 0].set(
        jnp.where(adjust_saturated, saturated_water, water_scalar[:, 0])
    )

    oxygen_limited = jnp.asarray(parameters.use_methane, dtype=jnp.bool_) & jnp.asarray(
        parameters.use_anoxia,
        dtype=jnp.bool_,
    )
    oxygen_unlimited = jnp.asarray(scalars.oxygen_scalar_column_level, dtype=jnp.float64).at[
        :, 0
    ].set(jnp.float64(1.0))
    oxygen_scalar = lax.cond(
        oxygen_limited,
        lambda _: oxygen_filtered,
        lambda _: oxygen_unlimited,
        operand=None,
    )
    return DecompositionScalarState(temperature_scalar, water_scalar, oxygen_scalar)


def _vertical_level_scalars(
    inputs: DecompositionRateInputs,
    scalars: DecompositionScalarState,
    parameters: DecompositionRateParameters,
    soil_column: jnp.ndarray,
) -> DecompositionScalarState:
    temperature = jnp.where(soil_column[:, None], inputs.soil_temperature_column_level,
                            jnp.float64(FREEZING_TEMPERATURE_FRESHWATER))
    q10_temperature = _temperature_q10(
        temperature,
        jnp.asarray(parameters.temperature_q10, dtype=jnp.float64),
        jnp.asarray(parameters.frozen_temperature_q10, dtype=jnp.float64),
    )
    celsius = fortran_add(
        temperature,
        -jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
    )
    century_temperature = jnp.maximum(
        fortran_divide(
            _century_temperature_function(celsius),
            _century_temperature_function(jnp.float64(30.0)),
        ),
        jnp.float64(0.01),
    )
    temperature_candidate = lax.cond(
        jnp.asarray(parameters.use_century_temperature_function, dtype=jnp.bool_),
        lambda _: century_temperature,
        lambda _: q10_temperature,
        operand=None,
    )

    water_candidate = _water_scalar(
        jnp.asarray(inputs.soil_water_potential_column_level, dtype=jnp.float64),
        jnp.asarray(parameters.minimum_water_potential_mpa, dtype=jnp.float64),
        jnp.asarray(parameters.maximum_water_potential_mpa, dtype=jnp.float64),
    )
    inundated = jnp.asarray(inputs.inundated_fraction_column, dtype=jnp.float64)
    saturated_water = fortran_add(
        fortran_multiply(water_candidate, fortran_add(1.0, -inundated[:, None])),
        inundated[:, None],
    )
    adjust_saturated = (
        jnp.asarray(parameters.use_methane, dtype=jnp.bool_)
        & jnp.asarray(parameters.anoxia_uses_saturated_fraction, dtype=jnp.bool_)
        & (temperature > jnp.float64(FREEZING_TEMPERATURE_FRESHWATER))
    )
    water_candidate = jnp.where(adjust_saturated, saturated_water, water_candidate)

    minimum_oxygen = jnp.asarray(parameters.minimum_oxygen_scalar, dtype=jnp.float64)
    oxygen_unsaturated = jnp.maximum(
        jnp.asarray(inputs.oxygen_stress_unsaturated_column_level, dtype=jnp.float64),
        minimum_oxygen,
    )
    oxygen_saturated = jnp.maximum(
        jnp.asarray(inputs.oxygen_stress_saturated_column_level, dtype=jnp.float64),
        minimum_oxygen,
    )
    oxygen_mixed = fortran_add(
        fortran_multiply(oxygen_unsaturated, fortran_add(1.0, -inundated[:, None])),
        fortran_multiply(oxygen_saturated, inundated[:, None]),
    )
    oxygen_limited_candidate = lax.cond(
        jnp.asarray(parameters.anoxia_uses_saturated_fraction, dtype=jnp.bool_),
        lambda _: oxygen_mixed,
        lambda _: oxygen_unsaturated,
        operand=None,
    )
    oxygen_limited = jnp.asarray(parameters.use_methane, dtype=jnp.bool_) & jnp.asarray(
        parameters.use_anoxia,
        dtype=jnp.bool_,
    )
    oxygen_candidate = lax.cond(
        oxygen_limited,
        lambda _: oxygen_limited_candidate,
        lambda _: jnp.ones_like(oxygen_limited_candidate),
        operand=None,
    )

    active_level = jnp.arange(NUMBER_OF_GROUND_LEVELS) < NUMBER_OF_SOIL_LEVELS
    filtered_level = soil_column[:, None] & active_level[None, :]
    oxygen_write = jnp.where(oxygen_limited, filtered_level, active_level[None, :])
    return DecompositionScalarState(
        temperature_scalar_column_level=jnp.where(
            filtered_level,
            temperature_candidate,
            scalars.temperature_scalar_column_level,
        ),
        water_scalar_column_level=jnp.where(
            filtered_level,
            water_candidate,
            scalars.water_scalar_column_level,
        ),
        oxygen_scalar_column_level=jnp.where(
            oxygen_write,
            oxygen_candidate,
            scalars.oxygen_scalar_column_level,
        ),
    )


def _normalized_temperature(
    temperature_scalar: jnp.ndarray,
    parameters: DecompositionRateParameters,
    soil_column: jnp.ndarray,
    one_level: jnp.ndarray,
) -> jnp.ndarray:
    reference = jnp.asarray(parameters.normalization_reference_celsius, dtype=jnp.float64)
    century_ratio = fortran_divide(
        _century_temperature_function(reference),
        _century_temperature_function(jnp.float64(30.0)),
    )
    q10_exponent = fortran_divide(
        fortran_add(reference, -jnp.float64(25.0)),
        jnp.float64(10.0),
    )
    normalization = fortran_divide(
        century_ratio,
        (native_power if default_backend() == "gpu" else fortran_power)(
            jnp.asarray(parameters.temperature_q10, dtype=jnp.float64),
            q10_exponent,
        ),
    )
    candidate = fortran_multiply(temperature_scalar, normalization)
    level_count = jnp.where(one_level, jnp.int32(1), jnp.int32(NUMBER_OF_SOIL_LEVELS))
    active = soil_column[:, None] & (
        jnp.arange(NUMBER_OF_GROUND_LEVELS)[None, :] < level_count
    )
    normalized = jnp.where(active, candidate, temperature_scalar)
    return lax.cond(
        jnp.asarray(parameters.normalize_q10_to_century, dtype=jnp.bool_),
        lambda _: normalized,
        lambda _: temperature_scalar,
        operand=None,
    )


def _base_decomposition_rates(days_per_year: jnp.ndarray) -> jnp.ndarray:
    def default_real_reciprocal(value: float) -> jnp.ndarray:
        folded = np.float32(1.0) / np.float32(value)
        return jnp.float64(np.float64(folded))

    turnover = jnp.stack(
        (
            default_real_reciprocal(18.5),
            default_real_reciprocal(4.9),
            default_real_reciprocal(4.9),
            default_real_reciprocal(0.3),
            default_real_reciprocal(7.3),
            default_real_reciprocal(0.2),
            default_real_reciprocal(0.0045),
        )
    )
    seconds = fortran_multiply(jnp.float64(SECONDS_PER_DAY), days_per_year)
    return fortran_divide(jnp.float64(1.0), fortran_multiply(seconds, turnover))


def _latitude_spinup_term(latitude_degrees: jnp.ndarray) -> jnp.ndarray:
    exponent = fortran_multiply(
        jnp.float64(-0.15),
        fortran_add(jnp.abs(latitude_degrees), -jnp.float64(60.0)),
    )
    denominator = fortran_add(jnp.float64(1.0), strict_exp(exponent))
    return fortran_add(
        jnp.float64(1.0),
        fortran_divide(jnp.float64(50.0), denominator),
    )


def _geographic_factor(
    spinup_factor: jnp.ndarray,
    latitude_term: jnp.ndarray,
    spinup_state: jnp.ndarray,
) -> jnp.ndarray:
    accelerated = (
        (spinup_state >= jnp.int32(1))
        & (jnp.abs(fortran_add(spinup_factor, -jnp.float64(1.0))) > jnp.float64(1.0e-6))
    )
    return jnp.where(
        accelerated,
        fortran_multiply(spinup_factor, latitude_term),
        jnp.float64(1.0),
    )


def _write_rate_constants(
    previous: jnp.ndarray,
    scalars: DecompositionScalarState,
    depth_scalar: jnp.ndarray,
    inputs: DecompositionRateInputs,
    parameters: DecompositionRateParameters,
    base_rate: jnp.ndarray,
    one_level: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    latitude_term = _latitude_spinup_term(
        jnp.asarray(inputs.latitude_degrees_column, dtype=jnp.float64)
    )
    spinup_factor = jnp.asarray(parameters.spinup_factor, dtype=jnp.float64)
    spinup_state = jnp.asarray(parameters.spinup_state, dtype=jnp.int32)
    geographic = jnp.stack(
        tuple(
            _geographic_factor(spinup_factor[index], latitude_term, spinup_state)
            for index in range(7)
        ),
        axis=1,
    )

    def pool_rate(rate: jnp.ndarray, geographic_factor: jnp.ndarray) -> jnp.ndarray:
        without_depth = _multiply_chain(
            rate,
            scalars.temperature_scalar_column_level,
            scalars.water_scalar_column_level,
            scalars.oxygen_scalar_column_level,
            geographic_factor[:, None],
        )
        with_depth = _multiply_chain(
            rate,
            scalars.temperature_scalar_column_level,
            scalars.water_scalar_column_level,
            depth_scalar,
            scalars.oxygen_scalar_column_level,
            geographic_factor[:, None],
        )
        return lax.cond(
            jnp.asarray(parameters.use_vertical_soil_carbon, dtype=jnp.bool_),
            lambda _: with_depth,
            lambda _: without_depth,
            operand=None,
        )

    geographic_index = (0, 1, 1, 3, 4, 5, 6)
    all_rates = tuple(
        pool_rate(base_rate[index], geographic[:, geographic_index[index]])
        for index in range(7)
    )
    level_count = jnp.where(one_level, jnp.int32(1), jnp.int32(NUMBER_OF_SOIL_LEVELS))
    write_mask = soil_column[:, None] & (
        jnp.arange(NUMBER_OF_GROUND_LEVELS)[None, :] < level_count
    )

    def write_pool(values: jnp.ndarray, pool_index: int, rate: jnp.ndarray) -> jnp.ndarray:
        return values.at[:, :, pool_index].set(
            jnp.where(write_mask, rate, values[:, :, pool_index])
        )

    nonfates = previous
    for pool_index in range(7):
        nonfates = write_pool(nonfates, pool_index, all_rates[pool_index])

    fates = previous
    for output_index, source_index in ((0, 0), (1, 1), (2, 2), (3, 4), (4, 5), (5, 6)):
        fates = write_pool(fates, output_index, all_rates[source_index])
    final = lax.cond(
        jnp.asarray(parameters.use_fates, dtype=jnp.bool_),
        lambda _: fates,
        lambda _: nonfates,
        operand=None,
    )
    return final, latitude_term


def decomposition_rate_constants_bgc(
    inputs: DecompositionRateInputs,
    cn_fluxes: CnTransientFluxState,
    scalars: DecompositionScalarState,
    parameters: DecompositionRateParameters = DEFAULT_DECOMPOSITION_RATE_PARAMETERS,
) -> DecompositionRateResult:

    _validate_shapes(inputs, cn_fluxes, scalars)
    _validate_scalar_parameters(parameters)
    previous_rate = get_cn_flux_field(cn_fluxes, _DECOMPOSITION_RATE_FIELD)
    calculate = (
        _filtered_rate_arrays
        if default_backend() == "gpu" and inputs.soil_column.size >= 4096
        else _rate_arrays
    )
    rate, updated_scalars, diagnostics = calculate(inputs, previous_rate, scalars, parameters)
    return DecompositionRateResult(
        cn_fluxes=set_cn_flux_field(cn_fluxes, _DECOMPOSITION_RATE_FIELD, rate),
        scalars=updated_scalars,
        diagnostics=diagnostics,
    )


def _rate_arrays(inputs, previous_rate, scalars, parameters):
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    one_level = (
        jnp.asarray(parameters.decomposition_level_count, dtype=jnp.int32) == jnp.int32(1)
    )
    weighted_scalars = _weighted_one_level_scalars(inputs, scalars, parameters, soil_column)
    vertical_scalars = _vertical_level_scalars(inputs, scalars, parameters, soil_column)
    updated_scalars = lax.cond(
        one_level,
        lambda _: weighted_scalars,
        lambda _: vertical_scalars,
        operand=None,
    )
    updated_scalars = updated_scalars._replace(
        temperature_scalar_column_level=_normalized_temperature(
            updated_scalars.temperature_scalar_column_level,
            parameters,
            soil_column,
            one_level,
        )
    )

    depth_candidate = strict_exp(
        fortran_divide(
            -jnp.asarray(inputs.soil_node_depth_level, dtype=jnp.float64),
            jnp.asarray(parameters.decomposition_depth_efolding_m, dtype=jnp.float64),
        )
    )
    depth_scalar = jnp.broadcast_to(
        depth_candidate[None, :],
        updated_scalars.temperature_scalar_column_level.shape,
    )
    depth_scalar = lax.cond(
        jnp.asarray(parameters.use_vertical_soil_carbon, dtype=jnp.bool_),
        lambda _: depth_scalar,
        lambda _: jnp.ones_like(depth_scalar),
        operand=None,
    )
    base_rate = _base_decomposition_rates(
        jnp.asarray(parameters.days_per_year, dtype=jnp.float64)
    )
    rate, latitude_term = _write_rate_constants(
        previous_rate,
        updated_scalars,
        depth_scalar,
        inputs,
        parameters,
        base_rate,
        one_level,
    )
    return (
        rate,
        updated_scalars,
        DecompositionRateDiagnostics(
            depth_scalar_column_level=depth_scalar,
            latitude_spinup_term_column=latitude_term,
            base_rate_pool=base_rate,
            updated_column_count=jnp.sum(soil_column, dtype=jnp.int32),
        ),
    )


def _filtered_rate_arrays(inputs, previous_rate, scalars, parameters):
    inputs, previous_rate, scalars = tree.map(jnp.asarray, (inputs, previous_rate, scalars))
    columns = inputs.soil_column.size
    capacity = (columns * 5 + 7) // 8

    def compact():
        indices = jnp.nonzero(inputs.soil_column, size=capacity, fill_value=columns)[0]
        safe = jnp.minimum(indices, columns - 1)
        selected_inputs = type(inputs)(*(
            value if name == "soil_node_depth_level" else value[safe]
            for name, value in zip(inputs._fields, inputs, strict=True)
        ))._replace(soil_column=indices < columns)
        selected_scalars = tree.map(lambda value: value[safe], scalars)
        rate, updated, _ = _rate_arrays(
            selected_inputs, previous_rate[safe], selected_scalars, parameters,
        )
        one_level = jnp.asarray(parameters.decomposition_level_count) == 1
        oxygen_limited = (
            jnp.asarray(parameters.use_methane) & jnp.asarray(parameters.use_anoxia)
        )
        levels = jnp.arange(NUMBER_OF_GROUND_LEVELS) < jnp.where(
            one_level, 1, NUMBER_OF_SOIL_LEVELS,
        )
        base_scalars = scalars._replace(oxygen_scalar_column_level=jnp.where(
            ~oxygen_limited & levels[None, :], jnp.float64(1.0),
            scalars.oxygen_scalar_column_level,
        ))
        restored = tree.map(
            lambda base, value: base.at[indices].set(value, mode="drop"),
            base_scalars, updated,
        )
        return previous_rate.at[indices].set(rate, mode="drop"), restored

    def dense():
        rate, updated, _ = _rate_arrays(inputs, previous_rate, scalars, parameters)
        return rate, updated

    rate, updated = lax.cond(
        jnp.count_nonzero(inputs.soil_column) <= capacity, compact, dense,
    )
    depth = strict_exp(fortran_divide(
        -jnp.asarray(inputs.soil_node_depth_level, dtype=jnp.float64),
        jnp.asarray(parameters.decomposition_depth_efolding_m, dtype=jnp.float64),
    ))
    depth = jnp.where(parameters.use_vertical_soil_carbon, depth, jnp.ones_like(depth))
    diagnostics = DecompositionRateDiagnostics(
        jnp.broadcast_to(depth, (columns, NUMBER_OF_GROUND_LEVELS)),
        _latitude_spinup_term(jnp.asarray(inputs.latitude_degrees_column, dtype=jnp.float64)),
        _base_decomposition_rates(jnp.asarray(parameters.days_per_year, dtype=jnp.float64)),
        jnp.sum(inputs.soil_column, dtype=jnp.int32),
    )
    return rate, updated, diagnostics


__all__ = [
    "DEFAULT_DECOMPOSITION_RATE_PARAMETERS",
    "NUMBER_OF_GROUND_LEVELS",
    "NUMBER_OF_SOIL_LEVELS",
    "DecompositionRateDiagnostics",
    "DecompositionRateInputs",
    "DecompositionRateParameters",
    "DecompositionRateResult",
    "DecompositionScalarState",
    "decomposition_rate_constants_bgc",
    "initialize_decomposition_scalar_state",
]
