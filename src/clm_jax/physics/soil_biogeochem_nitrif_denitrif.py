
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from clm_jax.constants import (
    FREEZING_TEMPERATURE_FRESHWATER,
    GRAVITY,
    PI,
    SECONDS_PER_DAY,
)
from clm_jax.floating_point import correctly_rounded_divide
from clm_jax.physics.adjoint_scaling import sensitivity_product
from clm_jax.physics.cn_zero import (
    CnFluxFields,
    CnTransientFluxState,
    get_cn_flux_field,
    set_cn_flux_field,
)
from clm_jax.physics.cn_zero_manifest import (
    COLUMN_LEVEL_FIELDS,
    NUMBER_OF_DECOMPOSITION_LEVELS,
)
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_materialize,
    fortran_multiply,
    strict_negative_exp,
)
from clm_jax.physics.pointwise import scalar_map
from clm_jax.physics.native_arctangent import native_atan
from clm_jax.physics.native_power import native_diffusion_power, native_power

NUMBER_OF_GROUND_LEVELS = NUMBER_OF_DECOMPOSITION_LEVELS
NUMBER_OF_SOIL_LEVELS = 20

_FIELD_PREFIX = "soil_nitrogen."
_OUTPUT_FIELDS = {
    "anaerobic_fraction": _FIELD_PREFIX + "anaerobic_frac_col",
    "diffusivity": _FIELD_PREFIX + "diffus_col",
    "base_denitrification": _FIELD_PREFIX + "f_denit_base_vr_col",
    "carbon_limited_denitrification": (
        _FIELD_PREFIX + "fmax_denit_carbonsubstrate_vr_col"
    ),
    "nitrate_limited_denitrification": (
        _FIELD_PREFIX + "fmax_denit_nitrate_vr_col"
    ),
    "water_filled_pore_scalar": _FIELD_PREFIX + "fr_WFPS_col",
    "nitrification_water_scalar": _FIELD_PREFIX + "k_nitr_h2o_vr_col",
    "nitrification_ph_scalar": _FIELD_PREFIX + "k_nitr_ph_vr_col",
    "nitrification_temperature_scalar": _FIELD_PREFIX + "k_nitr_t_vr_col",
    "nitrification_rate_constant": _FIELD_PREFIX + "k_nitr_vr_col",
    "n2_to_n2o_ratio": _FIELD_PREFIX + "n2_n2o_ratio_denit_vr_col",
    "potential_denitrification": _FIELD_PREFIX + "pot_f_denit_vr_col",
    "potential_nitrification": _FIELD_PREFIX + "pot_f_nit_vr_col",
    "effective_pore_radius": _FIELD_PREFIX + "r_psi_col",
    "diffusivity_ratio_constant": _FIELD_PREFIX + "ratio_k1_col",
    "nitrate_to_co2_ratio": _FIELD_PREFIX + "ratio_no3_co2_col",
    "nitrate_mass_density": _FIELD_PREFIX + "smin_no3_massdens_vr_col",
    "wet_soil_bulk_density": _FIELD_PREFIX + "soil_bulkdensity_col",
    "soil_co2_production": _FIELD_PREFIX + "soil_co2_prod_col",
    "water_filled_pore_space_percent": _FIELD_PREFIX + "wfps_vr_col",
}
_OUTPUT_FIELD_INDICES = tuple(
    COLUMN_LEVEL_FIELDS.index(field) for field in _OUTPUT_FIELDS.values()
)


class SoilBiogeochemNitrifDenitrifInputs(NamedTuple):

    soil_column: jnp.ndarray
    saturated_water_content_column_level: jnp.ndarray
    field_capacity_water_content_column_level: jnp.ndarray
    dry_soil_bulk_density_column_level: jnp.ndarray
    clapp_hornberger_b_column_level: jnp.ndarray
    organic_matter_density_column_level: jnp.ndarray
    saturated_suction_column_level: jnp.ndarray
    soil_water_potential_column_level: jnp.ndarray
    volumetric_water_column_level: jnp.ndarray
    liquid_water_mass_column_level: jnp.ndarray
    soil_temperature_column_level: jnp.ndarray
    oxygen_decomposition_unsaturated_column_level: jnp.ndarray
    oxygen_concentration_unsaturated_column_level: jnp.ndarray
    oxygen_decomposition_saturated_column_level: jnp.ndarray
    oxygen_concentration_saturated_column_level: jnp.ndarray
    inundated_fraction_column: jnp.ndarray
    mineral_ammonium_column_level: jnp.ndarray
    mineral_nitrate_column_level: jnp.ndarray
    potential_heterotrophic_respiration_column_level: jnp.ndarray
    decomposition_water_scalar_column_level: jnp.ndarray
    decomposition_temperature_scalar_column_level: jnp.ndarray
    soil_layer_thickness_column_level: jnp.ndarray


class SoilBiogeochemNitrifDenitrifParameters(NamedTuple):

    decomposition_level_count: int | jnp.ndarray = NUMBER_OF_SOIL_LEVELS
    use_methane: bool | jnp.ndarray = True
    anoxia_uses_saturated_fraction: bool | jnp.ndarray = False
    suppress_frozen_nitrification_denitrification: bool | jnp.ndarray = False
    maximum_nitrification_rate_per_second: float | jnp.ndarray = (
        np.float64(0.1) / np.float64(SECONDS_PER_DAY)
    )
    surface_tension_water: float | jnp.ndarray = 0.072
    rijtema_kroes_a: float | jnp.ndarray = 1.0
    rijtema_kroes_alpha: float | jnp.ndarray = 1.0
    rijtema_kroes_beta: float | jnp.ndarray = 1.0
    rijtema_kroes_gamma: float | jnp.ndarray = 1.0
    rijtema_kroes_delta: float | jnp.ndarray = 1.0
    denitrification_respiration_coefficient: float | jnp.ndarray = 0.1
    denitrification_respiration_exponent: float | jnp.ndarray = 1.3
    denitrification_nitrate_coefficient: float | jnp.ndarray = 1.15
    denitrification_nitrate_exponent: float | jnp.ndarray = 0.57
    organic_matter_maximum_density: float | jnp.ndarray = 130.0
    soil_ph: float | jnp.ndarray = 6.5
    oxygen_gas_diffusivity_intercept: float | jnp.ndarray = 0.1759
    oxygen_gas_diffusivity_temperature_slope: float | jnp.ndarray = 0.00117
    oxygen_water_diffusivity_intercept: float | jnp.ndarray = 1.172
    oxygen_water_diffusivity_temperature_slope: float | jnp.ndarray = 0.03443
    oxygen_water_diffusivity_temperature_quadratic: float | jnp.ndarray = 0.0005048
    gravity: float | jnp.ndarray = GRAVITY
    water_density: float | jnp.ndarray = 1.0e3
    seconds_per_day: float | jnp.ndarray = SECONDS_PER_DAY
    freezing_temperature: float | jnp.ndarray = FREEZING_TEMPERATURE_FRESHWATER


DEFAULT_SOIL_BIOGEOCHEM_NITRIF_DENITRIF_PARAMETERS = (
    SoilBiogeochemNitrifDenitrifParameters()
)


class SoilBiogeochemNitrifDenitrifDiagnostics(NamedTuple):
    updated_column_count: jnp.ndarray
    anoxic_cell_count: jnp.ndarray
    frozen_suppressed_cell_count: jnp.ndarray


class SoilBiogeochemNitrifDenitrifResult(NamedTuple):
    cn_fluxes: CnTransientFluxState
    diagnostics: SoilBiogeochemNitrifDenitrifDiagnostics


def _validate_shapes(
    inputs: SoilBiogeochemNitrifDenitrifInputs,
    cn_fluxes: CnTransientFluxState,
) -> int:
    if jnp.ndim(inputs.soil_column) != 1:
        raise ValueError("soil_column must be a vector")
    number_of_columns = inputs.soil_column.shape[0]
    level_shape = (number_of_columns, NUMBER_OF_GROUND_LEVELS)
    for name in SoilBiogeochemNitrifDenitrifInputs._fields:
        if name == "soil_column" or name == "inundated_fraction_column":
            continue
        if jnp.shape(getattr(inputs, name)) != level_shape:
            raise ValueError(f"{name} must have shape (column, 25)")
    if jnp.shape(inputs.inundated_fraction_column) != (number_of_columns,):
        raise ValueError("inundated_fraction_column must contain every column")
    for field in _OUTPUT_FIELDS.values():
        if get_cn_flux_field(cn_fluxes, field).shape != level_shape:
            raise ValueError(f"{field} must have shape (column, 25)")
    return number_of_columns


def _validate_parameters(parameters: SoilBiogeochemNitrifDenitrifParameters) -> None:
    for name in SoilBiogeochemNitrifDenitrifParameters._fields:
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    count = parameters.decomposition_level_count
    if isinstance(count, (int, np.integer)) and not (
        1 <= count <= NUMBER_OF_GROUND_LEVELS
    ):
        raise ValueError("decomposition_level_count must lie in [1, 25]")


def _multiply_chain(first: jnp.ndarray, *rest: jnp.ndarray) -> jnp.ndarray:
    result = jnp.asarray(first, dtype=jnp.float64)
    for factor in rest:
        result = fortran_multiply(result, factor)
    return result


def _split_normal(value: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:

    scaled = fortran_multiply(jnp.float64(134217729.0), value)
    large = fortran_add(scaled, -value)
    high = fortran_add(scaled, -large)
    return high, fortran_add(value, -high)


def _two_product_normal(
    left: jnp.ndarray,
    right: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    product = fortran_multiply(left, right)
    left_high, left_low = _split_normal(left)
    right_high, right_low = _split_normal(right)
    error = fortran_add(fortran_multiply(left_high, right_high), -product)
    error = fortran_add(error, fortran_multiply(left_high, right_low))
    error = fortran_add(error, fortran_multiply(left_low, right_high))
    error = fortran_add(error, fortran_multiply(left_low, right_low))
    return product, error


@jax.custom_jvp
def _nonnegative_multiply_preserving_subnormal(
    left: jnp.ndarray,
    right: jnp.ndarray,
) -> jnp.ndarray:

    if jax.default_backend() == "gpu":
        return fortran_multiply(left, right)
    return _multiply_with_software_underflow(left, right)


def _multiply_with_software_underflow(left, right):

    left, right = jnp.broadcast_arrays(
        jnp.asarray(left, dtype=jnp.float64),
        jnp.asarray(right, dtype=jnp.float64),
    )
    ordinary = fortran_multiply(left, right)
    left_bits = lax.bitcast_convert_type(left, jnp.uint64)
    right_bits = lax.bitcast_convert_type(right, jnp.uint64)
    sign_mask = jnp.uint64(0x8000000000000000)
    magnitude_mask = jnp.uint64(0x7FFFFFFFFFFFFFFF)
    exponent_mask = jnp.uint64(0x7FF0000000000000)
    fraction_mask = jnp.uint64(0x000FFFFFFFFFFFFF)
    left_fraction = left_bits & fraction_mask
    right_fraction = right_bits & fraction_mask
    left_exponent_bits = (left_bits & exponent_mask) >> jnp.uint64(52)
    right_exponent_bits = (right_bits & exponent_mask) >> jnp.uint64(52)
    left_subnormal = ((left_bits & exponent_mask) == 0) & (left_fraction != 0)
    right_subnormal = ((right_bits & exponent_mask) == 0) & (right_fraction != 0)
    exactly_one_subnormal = left_subnormal ^ right_subnormal

    subnormal_count = jnp.where(left_subnormal, left_fraction, right_fraction)
    normal_factor = jnp.where(left_subnormal, right, left)
    count_high, count_low = _two_product_normal(
        normal_factor,
        subnormal_count.astype(jnp.float64),
    )
    floor_count = jnp.floor(count_high)
    fraction = fortran_add(
        fortran_add(count_high, -floor_count),
        count_low,
    )
    below_zero = fraction < jnp.float64(0.0)
    at_least_one = fraction >= jnp.float64(1.0)
    floor_count = floor_count + jnp.where(
        below_zero,
        jnp.float64(-1.0),
        jnp.where(at_least_one, jnp.float64(1.0), jnp.float64(0.0)),
    )
    fraction = fraction + jnp.where(
        below_zero,
        jnp.float64(1.0),
        jnp.where(at_least_one, jnp.float64(-1.0), jnp.float64(0.0)),
    )
    floor_bits = floor_count.astype(jnp.uint64)
    round_up = (fraction > jnp.float64(0.5)) | (
        (fraction == jnp.float64(0.5))
        & ((floor_bits & jnp.uint64(1)) == jnp.uint64(1))
    )
    result_bits = floor_bits + round_up.astype(jnp.uint64)
    subnormal_operand_preserved = lax.bitcast_convert_type(
        result_bits, jnp.float64
    )
    use_subnormal_operand_preserved = (
        exactly_one_subnormal
        & (ordinary == jnp.float64(0.0))
        & ((left_bits & sign_mask) == jnp.uint64(0))
        & ((right_bits & sign_mask) == jnp.uint64(0))
        & ((left_bits & magnitude_mask) != jnp.uint64(0))
        & ((right_bits & magnitude_mask) != jnp.uint64(0))
    )

    normal_exponent_bits = jnp.uint64(1022) << jnp.uint64(52)
    left_mantissa = lax.bitcast_convert_type(
        left_fraction | normal_exponent_bits,
        jnp.float64,
    )
    right_mantissa = lax.bitcast_convert_type(
        right_fraction | normal_exponent_bits,
        jnp.float64,
    )
    mantissa_high, mantissa_low = _two_product_normal(
        left_mantissa,
        right_mantissa,
    )
    unit_shift = (
        left_exponent_bits.astype(jnp.int32)
        + right_exponent_bits.astype(jnp.int32)
        - jnp.int32(970)
    )
    possible_normal_underflow = unit_shift <= jnp.int32(53)
    safe_shift = jnp.where(possible_normal_underflow, unit_shift, jnp.int32(0))
    scaled_high = jnp.ldexp(mantissa_high, safe_shift)
    scaled_low = jnp.ldexp(mantissa_low, safe_shift)
    normal_floor = jnp.floor(scaled_high)
    normal_fraction = fortran_add(
        fortran_add(scaled_high, -normal_floor),
        scaled_low,
    )
    normal_below_zero = normal_fraction < jnp.float64(0.0)
    normal_at_least_one = normal_fraction >= jnp.float64(1.0)
    normal_floor = normal_floor + jnp.where(
        normal_below_zero,
        jnp.float64(-1.0),
        jnp.where(
            normal_at_least_one,
            jnp.float64(1.0),
            jnp.float64(0.0),
        ),
    )
    normal_fraction = normal_fraction + jnp.where(
        normal_below_zero,
        jnp.float64(1.0),
        jnp.where(
            normal_at_least_one,
            jnp.float64(-1.0),
            jnp.float64(0.0),
        ),
    )
    normal_floor_bits = normal_floor.astype(jnp.uint64)
    normal_round_up = (normal_fraction > jnp.float64(0.5)) | (
        (normal_fraction == jnp.float64(0.5))
        & ((normal_floor_bits & jnp.uint64(1)) == jnp.uint64(1))
    )
    normal_result_bits = normal_floor_bits + normal_round_up.astype(jnp.uint64)
    normal_preserved = lax.bitcast_convert_type(normal_result_bits, jnp.float64)
    scaled_exact = fortran_add(scaled_high, scaled_low)
    both_normal = (
        (left_exponent_bits != jnp.uint64(0))
        & (right_exponent_bits != jnp.uint64(0))
        & (left_exponent_bits != jnp.uint64(0x7FF))
        & (right_exponent_bits != jnp.uint64(0x7FF))
    )
    use_normal_preserved = (
        both_normal
        & possible_normal_underflow
        & (scaled_exact < jnp.float64(4503599627370496.0))
        & ((left_bits & sign_mask) == jnp.uint64(0))
        & ((right_bits & sign_mask) == jnp.uint64(0))
    )
    result = jnp.where(use_normal_preserved, normal_preserved, ordinary)
    return jnp.where(
        use_subnormal_operand_preserved,
        subnormal_operand_preserved,
        result,
    )


@_nonnegative_multiply_preserving_subnormal.defjvp
def _nonnegative_multiply_preserving_subnormal_jvp(primals, tangents):
    left, right = primals
    left_tangent, right_tangent = tangents
    result = _nonnegative_multiply_preserving_subnormal(left, right)
    tangent = sensitivity_product(right, left_tangent) + sensitivity_product(left, right_tangent)
    return result, tangent


def _elementwise_atan(value: jnp.ndarray) -> jnp.ndarray:
    value = jnp.asarray(value, dtype=jnp.float64)
    if jax.default_backend() == "gpu":
        return native_atan(value)
    shape = value.shape
    return jnp.reshape(scalar_map(jnp.arctan, jnp.reshape(value, (-1,))), shape)


def _elementwise_sqrt(value: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.asarray(value, dtype=jnp.float64))


def _packed_divide(numerator: jnp.ndarray, denominator: jnp.ndarray) -> jnp.ndarray:

    return correctly_rounded_divide(numerator, denominator)


def _packed_power(base: jnp.ndarray, exponent: jnp.ndarray) -> jnp.ndarray:

    base, exponent = jnp.broadcast_arrays(
        jnp.asarray(base, dtype=jnp.float64),
        jnp.asarray(exponent, dtype=jnp.float64),
    )
    if jax.default_backend() == "gpu":
        return native_power(base, exponent)
    return lax.optimization_barrier(jnp.power(base, exponent))


def _oxygen_response_core(radius, use, oxygen, water, a, alpha, beta, gamma, delta):
    positive = use > 0.0
    safe_use = jnp.where(positive, use, jnp.float64(1.0))
    exponent = _multiply_chain(
        -a, _packed_power(radius, -alpha), _packed_power(safe_use, -beta),
        _packed_power(oxygen, gamma), _packed_power(water, delta),
    )
    return jnp.where(positive, strict_negative_exp(exponent), jnp.float64(0.0))


@jax.custom_jvp
def _oxygen_response(*arguments):
    return _oxygen_response_core(*arguments)


@_oxygen_response.defjvp
def _oxygen_response_jvp(primals, tangents):
    value = _oxygen_response(*primals)
    enabled = (primals[1] > 0.0) & (value != 0.0)

    def differentiable(radius, use, oxygen, water, *coefficients):
        safe = tuple(jnp.where(enabled, item, jnp.float64(1.0))
                     for item in (radius, use, oxygen, water))
        return jnp.where(enabled, _oxygen_response_core(*safe, *coefficients), 0.0)

    return value, jax.jvp(differentiable, primals, tangents)[1]


@jax.custom_jvp
def _nitrate_power(nitrate, exponent):
    return _packed_power(nitrate, exponent)


@_nitrate_power.defjvp
def _nitrate_power_jvp(primals, tangents):
    nitrate, exponent = primals
    nitrate_slope = jax.jvp(lambda x: jnp.power(x, exponent),
                           (nitrate,), (jnp.ones_like(nitrate),))[1]
    exponent_slope = jax.jvp(lambda x: jnp.power(nitrate, x),
                            (exponent,), (jnp.ones_like(exponent),))[1]
    tangent = (sensitivity_product(nitrate_slope, tangents[0])
               + sensitivity_product(exponent_slope, tangents[1]))
    return _nitrate_power(*primals), tangent


def _denitrification_limits_core(carbon, nitrate, conversion, ccoef, cexp, ncoef, nexp, active):
    carbon_limit = _packed_divide(
        fortran_multiply(ccoef, _packed_power(carbon, cexp)), conversion,
    )
    nitrate_limit = _packed_divide(
        fortran_multiply(ncoef, _nitrate_power(nitrate, nexp)), conversion,
    )
    base = jnp.maximum(jnp.minimum(carbon_limit, nitrate_limit), jnp.float64(0.0))
    return carbon_limit, nitrate_limit, base


@jax.custom_jvp
def _denitrification_limits(*arguments):
    return _denitrification_limits_core(*arguments)


@_denitrification_limits.defjvp
def _denitrification_limits_jvp(primals, tangents):
    value = _denitrification_limits(*primals)
    limit_tangents = jax.jvp(_denitrification_limits_core, primals, tangents)[1][:2]

    def base(carbon, nitrate, conversion, ccoef, cexp, ncoef, nexp, active):
        empty_carbon = (carbon == 0.0) & (cexp > 1.0)
        unused = ~active | empty_carbon
        safe = tuple(jnp.where(unused, jnp.float64(1.0), item)
                     for item in (carbon, nitrate, conversion))
        candidate = _denitrification_limits_core(
            *safe, ccoef, cexp, ncoef, nexp, active,
        )[2]
        return jnp.where(unused, jnp.float64(0.0), candidate)

    base_tangent = jax.jvp(base, primals, tangents)[1]
    return value, (*limit_tangents, base_tangent)


def _nitrification_candidates_core(
    inputs: SoilBiogeochemNitrifDenitrifInputs,
    parameters: SoilBiogeochemNitrifDenitrifParameters,
):
    soil = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    level = jnp.arange(NUMBER_OF_GROUND_LEVELS, dtype=jnp.int32)
    active_level = level < jnp.asarray(
        parameters.decomposition_level_count,
        dtype=jnp.int32,
    )
    active = soil[:, None] & active_level[None, :]

    def array(name: str) -> jnp.ndarray:
        return jnp.asarray(getattr(inputs, name), dtype=jnp.float64)

    watsat = array("saturated_water_content_column_level")
    watfc = array("field_capacity_water_content_column_level")
    dry_bulk_density = array("dry_soil_bulk_density_column_level")
    bsw = array("clapp_hornberger_b_column_level")
    organic_density = array("organic_matter_density_column_level")
    suction = array("saturated_suction_column_level")
    soil_water_potential = array("soil_water_potential_column_level")
    volumetric_water = array("volumetric_water_column_level")
    liquid_water = array("liquid_water_mass_column_level")
    temperature = array("soil_temperature_column_level")
    oxygen_use_unsaturated = array(
        "oxygen_decomposition_unsaturated_column_level"
    )
    oxygen_unsaturated = array("oxygen_concentration_unsaturated_column_level")
    oxygen_use_saturated = array("oxygen_decomposition_saturated_column_level")
    oxygen_saturated = array("oxygen_concentration_saturated_column_level")
    inundated = array("inundated_fraction_column")
    ammonium = array("mineral_ammonium_column_level")
    nitrate = array("mineral_nitrate_column_level")
    respiration = array("potential_heterotrophic_respiration_column_level")
    water_scalar = array("decomposition_water_scalar_column_level")
    temperature_scalar = array("decomposition_temperature_scalar_column_level")
    thickness = array("soil_layer_thickness_column_level")

    one = jnp.float64(1.0)
    zero = jnp.float64(0.0)
    gas_intercept = jnp.asarray(
        parameters.oxygen_gas_diffusivity_intercept,
        dtype=jnp.float64,
    )
    gas_slope = jnp.asarray(
        parameters.oxygen_gas_diffusivity_temperature_slope,
        dtype=jnp.float64,
    )
    gas_coefficient = fortran_materialize(
        fortran_add(gas_intercept, fortran_multiply(gas_slope, temperature))
    )
    air_fraction = fortran_materialize(
        fortran_add(one, -_packed_divide(watfc, watsat))
    )
    air_filled_porosity = fortran_materialize(fortran_add(watsat, -watfc))
    organic_maximum = jnp.asarray(
        parameters.organic_matter_maximum_density,
        dtype=jnp.float64,
    )
    safe_organic_maximum = jnp.where(organic_maximum > zero, organic_maximum, one)
    organic_fraction = jnp.where(
        organic_maximum > zero,
        jnp.minimum(
            _packed_divide(organic_density, safe_organic_maximum),
            one,
        ),
        one,
    )
    organic_fraction = fortran_materialize(organic_fraction)

    use_methane = jnp.asarray(parameters.use_methane, dtype=jnp.bool_)
    diffusion_power = native_diffusion_power if jax.default_backend() == "gpu" else _packed_power
    air_power_organic = diffusion_power(
        air_fraction,
        _packed_divide(jnp.float64(10.0), jnp.float64(3.0)),
    )
    watsat_squared = fortran_multiply(watsat, watsat)
    organic_diffusion = _packed_divide(
        fortran_multiply(organic_fraction, air_power_organic),
        watsat_squared,
    )
    air_power_mineral = diffusion_power(
        air_fraction,
        _packed_divide(jnp.float64(3.0), bsw),
    )
    mineral_diffusion = _multiply_chain(
        fortran_add(one, -organic_fraction),
        fortran_multiply(air_filled_porosity, air_filled_porosity),
        air_power_mineral,
    )
    diffusion_porosity = fortran_add(organic_diffusion, mineral_diffusion)
    diffusivity_methane = _multiply_chain(
        gas_coefficient,
        jnp.float64(1.0e-4),
        diffusion_porosity,
    )
    diffusivity = jnp.where(use_methane, diffusivity_methane, zero)

    surface_tension = jnp.asarray(parameters.surface_tension_water, dtype=jnp.float64)
    water_density = jnp.asarray(parameters.water_density, dtype=jnp.float64)
    gravity = jnp.asarray(parameters.gravity, dtype=jnp.float64)
    pore_radius_numerator = fortran_multiply(jnp.float64(2.0), surface_tension)
    pore_radius_denominator = _multiply_chain(
        water_density,
        gravity,
        jnp.abs(soil_water_potential),
    )
    minimum_pore_radius = _packed_divide(
        pore_radius_numerator,
        pore_radius_denominator,
    )
    maximum_pore_radius = _packed_divide(
        pore_radius_numerator,
        _multiply_chain(water_density, gravity, jnp.float64(0.1)),
    )
    effective_pore_radius_methane = _elementwise_sqrt(
        fortran_multiply(minimum_pore_radius, maximum_pore_radius)
    )

    water_diffusivity = fortran_add(
        fortran_add(
            jnp.asarray(
                parameters.oxygen_water_diffusivity_intercept,
                dtype=jnp.float64,
            ),
            fortran_multiply(
                jnp.asarray(
                    parameters.oxygen_water_diffusivity_temperature_slope,
                    dtype=jnp.float64,
                ),
                temperature,
            ),
        ),
        fortran_multiply(
            jnp.asarray(
                parameters.oxygen_water_diffusivity_temperature_quadratic,
                dtype=jnp.float64,
            ),
            fortran_multiply(temperature, temperature),
        ),
    )
    gas_water_diffusivity_ratio = _packed_divide(
        fortran_multiply(gas_coefficient, jnp.float64(1.0e-4)),
        fortran_multiply(water_diffusivity, jnp.float64(1.0e-9)),
    )

    rij_a = jnp.asarray(parameters.rijtema_kroes_a, dtype=jnp.float64)
    rij_alpha = jnp.asarray(parameters.rijtema_kroes_alpha, dtype=jnp.float64)
    rij_beta = jnp.asarray(parameters.rijtema_kroes_beta, dtype=jnp.float64)
    rij_gamma = jnp.asarray(parameters.rijtema_kroes_gamma, dtype=jnp.float64)
    rij_delta = jnp.asarray(parameters.rijtema_kroes_delta, dtype=jnp.float64)

    unsaturated_water_diffusion = fortran_add(
        volumetric_water,
        fortran_multiply(gas_water_diffusivity_ratio, watsat),
    )
    anaerobic_unsaturated = _oxygen_response(
        effective_pore_radius_methane, oxygen_use_unsaturated,
        oxygen_unsaturated, unsaturated_water_diffusion,
        rij_a, rij_alpha, rij_beta, rij_gamma, rij_delta,
    )

    saturated_suction_pressure = _multiply_chain(
        gravity,
        jnp.float64(1.0e-6),
        suction,
    )
    saturated_minimum_pore_radius = _packed_divide(
        pore_radius_numerator,
        _multiply_chain(
            water_density,
            gravity,
            jnp.abs(saturated_suction_pressure),
        ),
    )
    saturated_effective_pore_radius = _elementwise_sqrt(
        fortran_multiply(saturated_minimum_pore_radius, maximum_pore_radius)
    )
    saturated_water_diffusion = fortran_add(
        watsat,
        fortran_multiply(gas_water_diffusivity_ratio, watsat),
    )
    anaerobic_saturated = _oxygen_response(
        saturated_effective_pore_radius, oxygen_use_saturated,
        oxygen_saturated, saturated_water_diffusion,
        rij_a, rij_alpha, rij_beta, rij_gamma, rij_delta,
    )
    anaerobic_mixed = fortran_add(
        fortran_multiply(fortran_add(one, -inundated[:, None]), anaerobic_unsaturated),
        fortran_multiply(inundated[:, None], anaerobic_saturated),
    )
    anaerobic_methane = lax.cond(
        jnp.asarray(parameters.anoxia_uses_saturated_fraction, dtype=jnp.bool_),
        lambda _: anaerobic_mixed,
        lambda _: anaerobic_unsaturated,
        operand=None,
    )
    anaerobic_fraction = jnp.where(use_methane, anaerobic_methane, zero)

    nitrification_temperature = jnp.minimum(temperature_scalar, one)
    default_056 = jnp.float64(np.float32(0.56))
    default_045 = jnp.float64(np.float32(0.45))
    default_minus_5 = jnp.float64(np.float32(-5.0))
    ph_argument = _multiply_chain(
        jnp.float64(PI),
        default_045,
        fortran_add(default_minus_5, jnp.asarray(parameters.soil_ph, dtype=jnp.float64)),
    )
    nitrification_ph = fortran_add(
        default_056,
        _packed_divide(_elementwise_atan(ph_argument), jnp.float64(PI)),
    )
    nitrification_ph = jnp.broadcast_to(nitrification_ph, temperature.shape)
    nitrification_water = water_scalar
    nitrification_rate = _multiply_chain(
        jnp.asarray(
            parameters.maximum_nitrification_rate_per_second,
            dtype=jnp.float64,
        ),
        nitrification_temperature,
        nitrification_water,
        nitrification_ph,
    )
    ammonium_decay = fortran_multiply(ammonium, nitrification_rate)
    potential_nitrification = jnp.where(ammonium_decay >= zero, ammonium_decay, zero)
    potential_nitrification = fortran_multiply(
        potential_nitrification,
        fortran_add(one, -anaerobic_fraction),
    )

    suppress_frozen = jnp.asarray(
        parameters.suppress_frozen_nitrification_denitrification,
        dtype=jnp.bool_,
    )
    frozen = temperature <= jnp.asarray(parameters.freezing_temperature, dtype=jnp.float64)
    frozen_suppression = suppress_frozen & frozen
    potential_nitrification = jnp.where(
        frozen_suppression,
        zero,
        potential_nitrification,
    )

    wet_bulk_density = fortran_add(
        dry_bulk_density,
        _packed_divide(liquid_water, thickness),
    )
    mass_conversion = _packed_divide(jnp.float64(1.0e3), wet_bulk_density)
    daily_mass_conversion = fortran_multiply(
        mass_conversion,
        jnp.asarray(parameters.seconds_per_day, dtype=jnp.float64),
    )
    nitrate_mass_density = fortran_multiply(jnp.maximum(nitrate, zero), mass_conversion)
    soil_co2_production = fortran_multiply(respiration, daily_mass_conversion)
    carbon_limited_denitrification, nitrate_limited_denitrification, base_denitrification = (
        _denitrification_limits(
            soil_co2_production, nitrate_mass_density, daily_mass_conversion,
            jnp.asarray(parameters.denitrification_respiration_coefficient, jnp.float64),
            jnp.asarray(parameters.denitrification_respiration_exponent, jnp.float64),
            jnp.asarray(parameters.denitrification_nitrate_coefficient, jnp.float64),
            jnp.asarray(parameters.denitrification_nitrate_exponent, jnp.float64), active,
        )
    )
    base_denitrification = jnp.where(
        frozen_suppression,
        zero,
        base_denitrification,
    )
    potential_denitrification = _nonnegative_multiply_preserving_subnormal(
        base_denitrification,
        anaerobic_fraction,
    )

    diffusivity_ratio_constant = jnp.maximum(
        jnp.float64(1.7),
        fortran_add(
            jnp.float64(38.4),
            -fortran_multiply(jnp.float64(350.0), diffusivity),
        ),
    )
    nitrate_to_co2_ratio = jnp.where(
        soil_co2_production > jnp.float64(1.0e-9),
        _packed_divide(nitrate_mass_density, jnp.where(
            soil_co2_production > jnp.float64(1.0e-9), soil_co2_production, one,
        )),
        jnp.float64(100.0),
    )
    water_filled_pore_space = fortran_multiply(
        jnp.maximum(
            jnp.minimum(_packed_divide(volumetric_water, watsat), one),
            zero,
        ),
        jnp.float64(100.0),
    )
    water_filled_pore_scalar = jnp.maximum(
        jnp.float64(0.1),
        fortran_add(
            fortran_multiply(jnp.float64(0.015), water_filled_pore_space),
            -jnp.float64(0.32),
        ),
    )
    saturated_water_filled_pore_scalar = fortran_add(
        fortran_multiply(
            water_filled_pore_scalar,
            fortran_add(one, -inundated[:, None]),
        ),
        fortran_multiply(inundated[:, None], jnp.float64(1.18)),
    )
    mix_wfps = use_methane & jnp.asarray(
        parameters.anoxia_uses_saturated_fraction,
        dtype=jnp.bool_,
    )
    water_filled_pore_scalar = jnp.where(
        mix_wfps,
        saturated_water_filled_pore_scalar,
        water_filled_pore_scalar,
    )
    default_016 = jnp.float64(np.float32(0.16))
    default_minus_08 = jnp.float64(np.float32(-0.8))
    n2_to_n2o_ratio = fortran_multiply(
        jnp.maximum(
            fortran_multiply(default_016, diffusivity_ratio_constant),
            fortran_multiply(
                diffusivity_ratio_constant,
                strict_negative_exp(
                    fortran_multiply(default_minus_08, nitrate_to_co2_ratio)
                ),
            ),
        ),
        water_filled_pore_scalar,
    )

    candidates = {
        "anaerobic_fraction": anaerobic_fraction,
        "diffusivity": diffusivity,
        "base_denitrification": base_denitrification,
        "carbon_limited_denitrification": carbon_limited_denitrification,
        "nitrate_limited_denitrification": nitrate_limited_denitrification,
        "water_filled_pore_scalar": water_filled_pore_scalar,
        "nitrification_water_scalar": nitrification_water,
        "nitrification_ph_scalar": nitrification_ph,
        "nitrification_temperature_scalar": nitrification_temperature,
        "nitrification_rate_constant": nitrification_rate,
        "n2_to_n2o_ratio": n2_to_n2o_ratio,
        "potential_denitrification": potential_denitrification,
        "potential_nitrification": potential_nitrification,
        "effective_pore_radius": effective_pore_radius_methane,
        "diffusivity_ratio_constant": diffusivity_ratio_constant,
        "nitrate_to_co2_ratio": nitrate_to_co2_ratio,
        "nitrate_mass_density": nitrate_mass_density,
        "wet_soil_bulk_density": wet_bulk_density,
        "soil_co2_production": soil_co2_production,
        "water_filled_pore_space_percent": water_filled_pore_space,
    }
    return candidates, active, use_methane, frozen_suppression


@jax.custom_jvp
def _nitrification_candidates(inputs, parameters):
    return _nitrification_candidates_core(inputs, parameters)


@_nitrification_candidates.defjvp
def _nitrification_candidates_jvp(primals, tangents):
    defaults = {
        "saturated_water_content_column_level": .5,
        "field_capacity_water_content_column_level": .3,
        "dry_soil_bulk_density_column_level": 1500.,
        "clapp_hornberger_b_column_level": 4.,
        "organic_matter_density_column_level": 0.,
        "saturated_suction_column_level": 100.,
        "soil_water_potential_column_level": -1000.,
        "volumetric_water_column_level": .25,
        "liquid_water_mass_column_level": 2.,
        "soil_temperature_column_level": 273.15,
        "oxygen_decomposition_unsaturated_column_level": .1,
        "oxygen_concentration_unsaturated_column_level": 1.,
        "oxygen_decomposition_saturated_column_level": .1,
        "oxygen_concentration_saturated_column_level": 1.,
        "mineral_ammonium_column_level": .01,
        "mineral_nitrate_column_level": .01,
        "potential_heterotrophic_respiration_column_level": .001,
        "decomposition_water_scalar_column_level": .5,
        "decomposition_temperature_scalar_column_level": .5,
        "soil_layer_thickness_column_level": .1,
    }

    def differentiable(inputs, parameters):
        soil = jnp.asarray(inputs.soil_column, jnp.bool_)
        active = soil[:, None] & (jnp.arange(NUMBER_OF_GROUND_LEVELS)[None, :]
                                  < parameters.decomposition_level_count)
        replacements = {name: jnp.where(active, getattr(inputs, name), jnp.float64(value))
                        for name, value in defaults.items()}
        replacements["inundated_fraction_column"] = jnp.where(
            soil, inputs.inundated_fraction_column, jnp.float64(0.),
        )
        return _nitrification_candidates_core(inputs._replace(**replacements), parameters)

    return _nitrification_candidates(*primals), jax.jvp(differentiable, primals, tangents)[1]


def soil_biogeochem_nitrif_denitrif(
    inputs: SoilBiogeochemNitrifDenitrifInputs,
    cn_fluxes: CnTransientFluxState,
    parameters: SoilBiogeochemNitrifDenitrifParameters = (
        DEFAULT_SOIL_BIOGEOCHEM_NITRIF_DENITRIF_PARAMETERS
    ),
) -> SoilBiogeochemNitrifDenitrifResult:
    _validate_shapes(inputs, cn_fluxes)
    _validate_parameters(parameters)
    soil = jnp.asarray(inputs.soil_column, jnp.bool_)
    candidates, active, use_methane, frozen_suppression = _nitrification_candidates(inputs, parameters)
    if isinstance(cn_fluxes.column_level, CnFluxFields):
        updated = cn_fluxes
        for name in _OUTPUT_FIELDS:
            candidate = candidates[name]
            field = _OUTPUT_FIELDS[name]
            mask = active & use_methane if name == "effective_pore_radius" else active
            updated = set_cn_flux_field(
                updated, field, jnp.where(mask, candidate, get_cn_flux_field(updated, field))
            )
    else:
        candidate_values = jnp.stack(tuple(candidates[name] for name in _OUTPUT_FIELDS), axis=-1)
        current_values = cn_fluxes.column_level[..., jnp.asarray(_OUTPUT_FIELD_INDICES)]
        field_active = jnp.broadcast_to(active[..., None], candidate_values.shape)
        effective_pore_radius_index = tuple(_OUTPUT_FIELDS).index("effective_pore_radius")
        field_active = field_active.at[..., effective_pore_radius_index].set(
            active & use_methane
        )
        filtered_values = jnp.where(field_active, candidate_values, current_values)
        updated = cn_fluxes._replace(
            column_level=cn_fluxes.column_level.at[
                ..., jnp.asarray(_OUTPUT_FIELD_INDICES)
            ].set(filtered_values)
        )
    anaerobic_bits = lax.bitcast_convert_type(candidates["anaerobic_fraction"], jnp.uint64)
    anaerobic_positive = (
        (anaerobic_bits & jnp.uint64(0x8000000000000000)) == jnp.uint64(0)
    ) & ((anaerobic_bits & jnp.uint64(0x7FFFFFFFFFFFFFFF)) != jnp.uint64(0))

    return SoilBiogeochemNitrifDenitrifResult(
        cn_fluxes=updated,
        diagnostics=SoilBiogeochemNitrifDenitrifDiagnostics(
            updated_column_count=jnp.sum(soil, dtype=jnp.int32),
            anoxic_cell_count=jnp.sum(
                active & anaerobic_positive,
                dtype=jnp.int32,
            ),
            frozen_suppressed_cell_count=jnp.sum(
                active & frozen_suppression,
                dtype=jnp.int32,
            ),
        ),
    )


__all__ = [
    "DEFAULT_SOIL_BIOGEOCHEM_NITRIF_DENITRIF_PARAMETERS",
    "NUMBER_OF_GROUND_LEVELS",
    "NUMBER_OF_SOIL_LEVELS",
    "SoilBiogeochemNitrifDenitrifDiagnostics",
    "SoilBiogeochemNitrifDenitrifInputs",
    "SoilBiogeochemNitrifDenitrifParameters",
    "SoilBiogeochemNitrifDenitrifResult",
    "soil_biogeochem_nitrif_denitrif",
]
