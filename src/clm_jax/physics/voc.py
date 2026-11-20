
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
    fortran_power,
    strict_exp,
)
from clm_jax.physics.pointwise import scalar_map


class VOCParameters(NamedTuple):

    emission_factor_by_compound_pft: jnp.ndarray
    compound_class: jnp.ndarray
    compound_molecular_weight: jnp.ndarray
    compound_coefficient: jnp.ndarray
    is_isoprene: jnp.ndarray
    mechanism_compound_index: jnp.ndarray
    mechanism_compound_mask: jnp.ndarray
    light_dependent_fraction: jnp.ndarray
    growing_leaf_age_factor: jnp.ndarray
    mature_leaf_age_factor: jnp.ndarray
    new_leaf_age_factor: jnp.ndarray
    old_leaf_age_factor: jnp.ndarray
    temperature_beta: jnp.ndarray
    temperature_ct1: jnp.ndarray
    temperature_ct2: jnp.ndarray
    temperature_ceo: jnp.ndarray
    non_evergreen_pft: jnp.ndarray
    mapped_factor_category_by_pft: jnp.ndarray
    use_mapped_isoprene_factors: jnp.ndarray


class VOCInputs(NamedTuple):

    soil_patch: jnp.ndarray
    direct_visible_radiation_grid: jnp.ndarray
    diffuse_visible_radiation_grid: jnp.ndarray
    air_pressure_column: jnp.ndarray
    co2_partial_pressure_grid: jnp.ndarray
    direct_visible_radiation_24h_patch: jnp.ndarray
    direct_visible_radiation_240h_patch: jnp.ndarray
    diffuse_visible_radiation_24h_patch: jnp.ndarray
    diffuse_visible_radiation_240h_patch: jnp.ndarray
    sunlit_fraction_patch: jnp.ndarray
    sunlit_fraction_24h_patch: jnp.ndarray
    sunlit_fraction_240h_patch: jnp.ndarray
    exposed_leaf_area_index_patch: jnp.ndarray
    exposed_leaf_area_index_240h_patch: jnp.ndarray
    sunlit_intracellular_co2_patch: jnp.ndarray
    shaded_intracellular_co2_patch: jnp.ndarray
    vegetation_temperature_patch: jnp.ndarray
    vegetation_temperature_24h_patch: jnp.ndarray
    vegetation_temperature_240h_patch: jnp.ndarray
    mapped_isoprene_emission_factor_grid: jnp.ndarray


class VOCDiagnostics(NamedTuple):

    optimum_emission_factor_patch: jnp.ndarray
    optimum_temperature_patch: jnp.ndarray
    light_alpha_patch: jnp.ndarray
    light_cp_patch: jnp.ndarray
    sunlit_par_patch: jnp.ndarray
    sunlit_par_24h_patch: jnp.ndarray
    sunlit_par_240h_patch: jnp.ndarray
    shaded_par_patch: jnp.ndarray
    shaded_par_24h_patch: jnp.ndarray
    shaded_par_240h_patch: jnp.ndarray
    total_activity_factor_patch: jnp.ndarray
    leaf_area_activity_factor_patch: jnp.ndarray
    temperature_activity_factor_patch: jnp.ndarray
    light_activity_factor_patch: jnp.ndarray
    leaf_age_activity_factor_patch: jnp.ndarray
    soil_moisture_activity_factor_patch: jnp.ndarray
    co2_activity_factor_patch: jnp.ndarray


class VOCState(NamedTuple):

    mechanism_flux_patch: jnp.ndarray
    total_flux_patch: jnp.ndarray
    compound_mass_flux_patch: jnp.ndarray
    diagnostics: VOCDiagnostics


class VOCEmissionResult(NamedTuple):
    state: VOCState


def initialize_voc_state(
    topology: SubgridTopology,
    parameters: VOCParameters,
) -> VOCState:

    number_of_patches = int(topology.patch.gridcell.shape[0])
    number_of_compounds = int(parameters.compound_class.shape[0])
    number_of_mechanisms = int(parameters.mechanism_compound_index.shape[0])
    special = jnp.full(number_of_patches, jnp.float64(CLM_SPECIAL_VALUE))
    diagnostics = VOCDiagnostics(*(special for _ in range(17)))
    return VOCState(
        mechanism_flux_patch=jnp.full(
            (number_of_patches, number_of_mechanisms),
            jnp.float64(CLM_SPECIAL_VALUE),
        ),
        total_flux_patch=special,
        compound_mass_flux_patch=jnp.zeros(
            (number_of_patches, number_of_compounds), dtype=jnp.float64
        ),
        diagnostics=diagnostics,
    )


def _multiply(*values: jnp.ndarray) -> jnp.ndarray:
    result = jnp.asarray(values[0], dtype=jnp.float64)
    for value in values[1:]:
        result = fortran_multiply(result, jnp.asarray(value, dtype=jnp.float64))
    return result


def _add(*values: jnp.ndarray) -> jnp.ndarray:
    result = jnp.asarray(values[0], dtype=jnp.float64)
    for value in values[1:]:
        result = fortran_add(result, jnp.asarray(value, dtype=jnp.float64))
    return result


def _subtract(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return fortran_add(left, -jnp.asarray(right, dtype=jnp.float64))


def _positive_binary64(value: jnp.ndarray) -> jnp.ndarray:

    value = jnp.asarray(value, dtype=jnp.float64)
    bits = lax.bitcast_convert_type(value, jnp.uint64)
    magnitude = bits & jnp.uint64(0x7FFFFFFFFFFFFFFF)
    positive_subnormal = (
        ((bits & jnp.uint64(0x8000000000000000)) == jnp.uint64(0))
        & (magnitude > jnp.uint64(0))
        & (magnitude < jnp.uint64(0x0010000000000000))
    )
    return (value > jnp.float64(0.0)) | positive_subnormal


def _promote_positive_subnormal(value: jnp.ndarray) -> jnp.ndarray:

    value = jnp.asarray(value, dtype=jnp.float64)
    bits = lax.bitcast_convert_type(value, jnp.uint64)
    magnitude = bits & jnp.uint64(0x7FFFFFFFFFFFFFFF)
    positive_subnormal = (
        ((bits & jnp.uint64(0x8000000000000000)) == jnp.uint64(0))
        & (magnitude > jnp.uint64(0))
        & (magnitude < jnp.uint64(0x0010000000000000))
    )
    return jnp.where(
        positive_subnormal,
        jnp.float64(jnp.finfo(jnp.float64).tiny),
        value,
    )


def _elementwise_log(value: jnp.ndarray) -> jnp.ndarray:
    value = jnp.asarray(value, dtype=jnp.float64)
    shape = value.shape
    result = scalar_map(jnp.log, jnp.reshape(value, (-1,)))
    return jnp.reshape(result, shape)


def _light_response_term(
    fraction: jnp.ndarray,
    cp: jnp.ndarray,
    alpha: jnp.ndarray,
    par: jnp.ndarray,
) -> jnp.ndarray:
    denominator_base = _add(
        jnp.float64(1.0),
        _multiply(alpha, alpha, par, par),
    )
    response = _multiply(
        cp,
        alpha,
        par,
        fortran_power(denominator_base, jnp.float64(-0.5)),
    )
    return _multiply(fraction, response)


def _gamma_p(
    par_sun: jnp.ndarray,
    par24_sun: jnp.ndarray,
    par240_sun: jnp.ndarray,
    par_shade: jnp.ndarray,
    par240_shade: jnp.ndarray,
    sunlit_fraction: jnp.ndarray,
    sunlit_fraction_240h: jnp.ndarray,
    direct_240h: jnp.ndarray,
    diffuse_240h: jnp.ndarray,
    light_dependent_fraction: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    adaptive = (
        _positive_binary64(sunlit_fraction_240h)
        & (sunlit_fraction_240h < jnp.float64(1.0))
        & _positive_binary64(direct_240h)
        & _positive_binary64(diffuse_240h)
    )
    safe_par240_sun = jnp.where(adaptive, par240_sun, jnp.float64(1.0))
    safe_par240_shade = jnp.where(adaptive, par240_shade, jnp.float64(1.0))

    alpha_sun = _subtract(
        jnp.float64(0.004),
        _multiply(jnp.float64(0.0005), _elementwise_log(safe_par240_sun)),
    )
    cp_sun = _multiply(
        jnp.float64(0.0468),
        strict_exp(
            _multiply(
                jnp.float64(0.0005),
                _subtract(par24_sun, jnp.float64(200.0)),
            )
        ),
        fortran_power(safe_par240_sun, jnp.float64(0.6)),
    )
    adaptive_response = _light_response_term(
        sunlit_fraction,
        cp_sun,
        alpha_sun,
        par_sun,
    )

    alpha_shade = _subtract(
        jnp.float64(0.004),
        _multiply(jnp.float64(0.0005), _elementwise_log(safe_par240_shade)),
    )
    cp_shade = _multiply(
        jnp.float64(0.0468),
        strict_exp(
            _multiply(
                jnp.float64(0.0005),
                _subtract(par_shade, jnp.float64(50.0)),
            )
        ),
        fortran_power(safe_par240_shade, jnp.float64(0.6)),
    )
    adaptive_response = _add(
        adaptive_response,
        _light_response_term(
            _subtract(jnp.float64(1.0), sunlit_fraction),
            cp_shade,
            alpha_shade,
            par_shade,
        ),
    )

    fixed_alpha = jnp.full_like(par_sun, jnp.float64(0.001))
    fixed_cp = jnp.full_like(par_sun, jnp.float64(1.21))
    fixed_response = _add(
        _light_response_term(
            sunlit_fraction,
            fixed_cp,
            fixed_alpha,
            par_sun,
        ),
        _light_response_term(
            _subtract(jnp.float64(1.0), sunlit_fraction),
            fixed_cp,
            fixed_alpha,
            par_shade,
        ),
    )
    light_response = jnp.where(adaptive, adaptive_response, fixed_response)
    alpha = jnp.where(adaptive, alpha_shade, fixed_alpha)
    cp = jnp.where(adaptive, cp_shade, fixed_cp)

    gamma = _add(
        _subtract(jnp.float64(1.0), light_dependent_fraction[None, :]),
        _multiply(light_dependent_fraction[None, :], light_response[:, None]),
    )
    return gamma, cp, alpha


def _gamma_t(
    temperature_240h: jnp.ndarray,
    temperature_24h: jnp.ndarray,
    temperature: jnp.ndarray,
    ct1: jnp.ndarray,
    ct2: jnp.ndarray,
    beta: jnp.ndarray,
    light_dependent_fraction: jnp.ndarray,
    ceo: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    accumulated = (temperature_240h > jnp.float64(0.0)) & (temperature_240h < jnp.float64(1.0e30))
    optimum_temperature_dynamic = _add(
        jnp.float64(313.0),
        _multiply(
            jnp.float64(0.6),
            _subtract(temperature_240h, jnp.float64(297.0)),
        ),
    )
    optimum_temperature = jnp.where(
        accumulated,
        optimum_temperature_dynamic,
        jnp.float64(317.0),
    )
    optimum_emission_dynamic = _multiply(
        ceo[None, :],
        strict_exp(
            _multiply(
                jnp.float64(0.05),
                _subtract(temperature_24h[:, None], jnp.float64(297.0)),
            )
        ),
        strict_exp(
            _multiply(
                jnp.float64(0.05),
                _subtract(temperature_240h[:, None], jnp.float64(297.0)),
            )
        ),
    )
    optimum_emission = jnp.where(
        accumulated[:, None],
        optimum_emission_dynamic,
        jnp.float64(2.26),
    )
    inverse_difference = _subtract(
        fortran_divide(jnp.float64(1.0), optimum_temperature),
        fortran_divide(jnp.float64(1.0), temperature),
    )
    x = fortran_divide(inverse_difference, jnp.float64(0.00831))[:, None]
    ct1_x = _multiply(ct1[None, :], x)
    ct2_x = _multiply(ct2[None, :], x)
    numerator = _multiply(ct2[None, :], strict_exp(ct1_x))
    denominator = _subtract(
        ct2[None, :],
        _multiply(
            ct1[None, :],
            _subtract(jnp.float64(1.0), strict_exp(ct2_x)),
        ),
    )
    light_dependent = _multiply(
        optimum_emission,
        fortran_divide(numerator, denominator),
    )
    light_independent = strict_exp(
        _multiply(
            beta[None, :],
            _subtract(temperature[:, None], jnp.float64(303.15)),
        )
    )
    gamma = _add(
        _multiply(
            _subtract(jnp.float64(1.0), light_dependent_fraction[None, :]),
            light_independent,
        ),
        _multiply(light_dependent_fraction[None, :], light_dependent),
    )
    return gamma, optimum_emission, optimum_temperature


def _gamma_a(
    vegetation_type: jnp.ndarray,
    leaf_area_240h: jnp.ndarray,
    leaf_area: jnp.ndarray,
    compound_class: jnp.ndarray,
    parameters: VOCParameters,
) -> jnp.ndarray:
    leaf_area_240h_arithmetic = _promote_positive_subnormal(leaf_area_240h)
    previous = _subtract(
        _multiply(jnp.float64(2.0), leaf_area_240h_arithmetic),
        leaf_area,
    )
    valid_average = _positive_binary64(leaf_area_240h) & (leaf_area_240h < jnp.float64(1.0e30))
    equal = previous == leaf_area
    declining = previous > leaf_area
    growing = previous < leaf_area
    safe_previous = jnp.where(declining, previous, jnp.float64(1.0))
    safe_leaf_area = jnp.where(growing, leaf_area, jnp.float64(1.0))
    decline_fraction = fortran_divide(
        _subtract(previous, leaf_area),
        safe_previous,
    )
    growth_mature_fraction = fortran_divide(previous, safe_leaf_area)
    fnew = jnp.where(
        growing,
        _subtract(jnp.float64(1.0), growth_mature_fraction),
        jnp.float64(0.0),
    )
    fgro = jnp.zeros_like(fnew)
    fmat = jnp.where(
        equal,
        jnp.float64(1.0),
        jnp.where(
            declining,
            _subtract(jnp.float64(1.0), decline_fraction),
            growth_mature_fraction,
        ),
    )
    fold = jnp.where(declining, decline_fraction, jnp.float64(0.0))

    new_factor = parameters.new_leaf_age_factor[compound_class]
    growing_factor = parameters.growing_leaf_age_factor[compound_class]
    mature_factor = parameters.mature_leaf_age_factor[compound_class]
    old_factor = parameters.old_leaf_age_factor[compound_class]
    activity = _add(
        _multiply(fnew[:, None], new_factor[None, :]),
        _multiply(fgro[:, None], growing_factor[None, :]),
        _multiply(fmat[:, None], mature_factor[None, :]),
        _multiply(fold[:, None], old_factor[None, :]),
    )
    non_evergreen = parameters.non_evergreen_pft[vegetation_type]
    return jnp.where(
        (non_evergreen & valid_average)[:, None],
        activity,
        jnp.float64(1.0),
    )


def _co2_inhibition_parameters(
    co2_ppmv: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    below_400 = co2_ppmv < jnp.float64(400.0)
    between_400_600 = (co2_ppmv > jnp.float64(400.0)) & (co2_ppmv < jnp.float64(600.0))
    between_600_800 = (co2_ppmv > jnp.float64(600.0)) & (co2_ppmv < jnp.float64(800.0))
    above_800 = co2_ppmv > jnp.float64(800.0)

    fraction_400_600 = fortran_divide(_subtract(co2_ppmv, jnp.float64(400.0)), jnp.float64(200.0))
    fraction_600_800 = fortran_divide(_subtract(co2_ppmv, jnp.float64(600.0)), jnp.float64(200.0))

    def interpolate(
        fraction: jnp.ndarray,
        upper: float,
        lower: float,
    ) -> jnp.ndarray:
        return _add(
            _multiply(fraction, jnp.float64(upper)),
            _multiply(
                _subtract(jnp.float64(1.0), fraction),
                jnp.float64(lower),
            ),
        )

    undefined = jnp.full_like(co2_ppmv, jnp.nan)
    ismax = jnp.where(
        below_400,
        jnp.float64(1.072),
        jnp.where(
            between_400_600,
            interpolate(fraction_400_600, 1.036, 1.072),
            jnp.where(
                between_600_800,
                interpolate(fraction_600_800, 1.046, 1.036),
                jnp.where(above_800, jnp.float64(1.014), undefined),
            ),
        ),
    )
    exponent = jnp.where(
        below_400,
        jnp.float64(1.70),
        jnp.where(
            between_400_600,
            interpolate(fraction_400_600, 2.0125, 1.70),
            jnp.where(
                between_600_800,
                interpolate(fraction_600_800, 1.5380, 2.0125),
                jnp.where(above_800, jnp.float64(2.861), undefined),
            ),
        ),
    )
    cstar = jnp.where(
        below_400,
        jnp.float64(1218.0),
        jnp.where(
            between_400_600,
            interpolate(fraction_400_600, 1150.0, 1218.0),
            jnp.where(
                between_600_800,
                interpolate(fraction_600_800, 2025.0, 1150.0),
                jnp.where(above_800, jnp.float64(1525.0), undefined),
            ),
        ),
    )
    return ismax, exponent, cstar


def _gamma_c(
    sunlit_ci: jnp.ndarray,
    shaded_ci: jnp.ndarray,
    pressure: jnp.ndarray,
    sunlit_fraction: jnp.ndarray,
    co2_ppmv: jnp.ndarray,
) -> jnp.ndarray:
    ambient_internal = _multiply(jnp.float64(0.7), co2_ppmv)
    ambient_power = fortran_power(ambient_internal, jnp.float64(1.4614))
    gamma_ca = _subtract(
        jnp.float64(1.344),
        fortran_divide(
            _multiply(jnp.float64(1.344), ambient_power),
            _add(
                fortran_power(jnp.float64(585.0), jnp.float64(1.4614)),
                ambient_power,
            ),
        ),
    )
    ismax, exponent, cstar = _co2_inhibition_parameters(co2_ppmv)

    weighted_valid = (
        (sunlit_ci == sunlit_ci)
        & (shaded_ci == shaded_ci)
        & (pressure > jnp.float64(0.0))
        & (sunlit_fraction > jnp.float64(0.0))
    )
    sun_only = (
        (sunlit_ci > jnp.float64(0.0))
        & (sunlit_ci < jnp.float64(1.0e30))
        & (pressure > jnp.float64(0.0))
        & (sunlit_fraction == jnp.float64(1.0))
    )
    shade_only = (
        (shaded_ci > jnp.float64(0.0))
        & (shaded_ci < jnp.float64(1.0e30))
        & (pressure > jnp.float64(0.0))
        & (sunlit_fraction == jnp.float64(0.0))
    )
    safe_pressure = jnp.where(
        weighted_valid | sun_only | shade_only,
        pressure,
        jnp.float64(1.0),
    )
    weighted_ci = _multiply(
        fortran_divide(
            _add(
                _multiply(sunlit_fraction, sunlit_ci),
                _multiply(
                    _subtract(jnp.float64(1.0), sunlit_fraction),
                    shaded_ci,
                ),
            ),
            safe_pressure,
        ),
        jnp.float64(1.0e6),
    )
    sun_ci = _multiply(
        fortran_divide(sunlit_ci, safe_pressure),
        jnp.float64(1.0e6),
    )
    shade_ci = _multiply(
        fortran_divide(shaded_ci, safe_pressure),
        jnp.float64(1.0e6),
    )
    ci = jnp.where(weighted_valid, weighted_ci, jnp.where(sun_only, sun_ci, shade_ci))
    ci_power = fortran_power(ci, exponent)
    gamma_ci_diagnosed = _subtract(
        ismax,
        fortran_divide(
            _multiply(ismax, ci_power),
            _add(fortran_power(cstar, exponent), ci_power),
        ),
    )
    gamma_ci = jnp.where(
        weighted_valid | sun_only | shade_only,
        gamma_ci_diagnosed,
        jnp.float64(1.0),
    )
    return _multiply(gamma_ci, gamma_ca)


def _validate_shapes(
    topology: SubgridTopology,
    state: VOCState,
    inputs: VOCInputs,
    parameters: VOCParameters,
) -> None:
    patch_shape = topology.patch.gridcell.shape
    grid_shape = topology.grid.longitude_degrees.shape
    column_shape = topology.column.gridcell.shape
    patch_fields = (
        inputs.soil_patch,
        inputs.direct_visible_radiation_24h_patch,
        inputs.direct_visible_radiation_240h_patch,
        inputs.diffuse_visible_radiation_24h_patch,
        inputs.diffuse_visible_radiation_240h_patch,
        inputs.sunlit_fraction_patch,
        inputs.sunlit_fraction_24h_patch,
        inputs.sunlit_fraction_240h_patch,
        inputs.exposed_leaf_area_index_patch,
        inputs.exposed_leaf_area_index_240h_patch,
        inputs.sunlit_intracellular_co2_patch,
        inputs.shaded_intracellular_co2_patch,
        inputs.vegetation_temperature_patch,
        inputs.vegetation_temperature_24h_patch,
        inputs.vegetation_temperature_240h_patch,
    )
    if any(jnp.shape(field) != patch_shape for field in patch_fields):
        raise ValueError("VOC patch inputs must contain one value per patch")
    if jnp.shape(inputs.air_pressure_column) != column_shape:
        raise ValueError("VOC pressure must contain one value per column")
    grid_fields = (
        inputs.direct_visible_radiation_grid,
        inputs.diffuse_visible_radiation_grid,
        inputs.co2_partial_pressure_grid,
    )
    if any(jnp.shape(field) != grid_shape for field in grid_fields):
        raise ValueError("VOC grid inputs must contain one value per gridcell")
    if jnp.shape(inputs.mapped_isoprene_emission_factor_grid) != (grid_shape[0], 6):
        raise ValueError("mapped isoprene factors must have six values per gridcell")

    number_of_compounds = parameters.compound_class.shape[0]
    number_of_mechanisms = parameters.mechanism_compound_index.shape[0]
    if jnp.shape(state.compound_mass_flux_patch) != (patch_shape[0], number_of_compounds):
        raise ValueError("VOC state compound dimension is incompatible with parameters")
    if jnp.shape(state.mechanism_flux_patch) != (patch_shape[0], number_of_mechanisms):
        raise ValueError("VOC state mechanism dimension is incompatible with parameters")
    if jnp.shape(parameters.emission_factor_by_compound_pft)[0] != number_of_compounds:
        raise ValueError("VOC emission-factor table must contain every compound")
    if number_of_compounds < 1 or number_of_mechanisms < 1:
        raise ValueError("VOC parameters require at least one compound and mechanism")


def voc_emission(
    topology: SubgridTopology,
    state: VOCState,
    inputs: VOCInputs,
    parameters: VOCParameters,
) -> VOCEmissionResult:

    _validate_shapes(topology, state, inputs, parameters)
    patch = topology.patch
    grid_index = jnp.asarray(patch.gridcell, dtype=jnp.int32)
    column_index = jnp.asarray(patch.column, dtype=jnp.int32)
    vegetation_type_raw = jnp.asarray(patch.vegetation_type, dtype=jnp.int32)
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    compute = soil_patch & (vegetation_type_raw > jnp.int32(0))
    vegetation_type = jnp.where(compute, vegetation_type_raw, jnp.int32(1))

    def patch_value(value: jnp.ndarray, fallback: float) -> jnp.ndarray:
        return jnp.where(
            compute,
            jnp.asarray(value, dtype=jnp.float64),
            jnp.float64(fallback),
        )

    sunlit_fraction = patch_value(inputs.sunlit_fraction_patch, 0.5)
    sunlit_fraction_24h = patch_value(inputs.sunlit_fraction_24h_patch, 0.5)
    sunlit_fraction_240h = patch_value(inputs.sunlit_fraction_240h_patch, 0.5)
    direct = jnp.asarray(inputs.direct_visible_radiation_grid, dtype=jnp.float64)[grid_index]
    diffuse = jnp.asarray(inputs.diffuse_visible_radiation_grid, dtype=jnp.float64)[grid_index]
    direct_24h = patch_value(inputs.direct_visible_radiation_24h_patch, 100.0)
    direct_240h = patch_value(inputs.direct_visible_radiation_240h_patch, 100.0)
    diffuse_24h = patch_value(inputs.diffuse_visible_radiation_24h_patch, 50.0)
    diffuse_240h = patch_value(inputs.diffuse_visible_radiation_240h_patch, 50.0)
    direct = jnp.where(compute, direct, jnp.float64(100.0))
    diffuse = jnp.where(compute, diffuse, jnp.float64(50.0))

    par_sun = _multiply(
        _add(direct, _multiply(sunlit_fraction, diffuse)),
        jnp.float64(4.6),
    )
    par24_sun = _multiply(
        _add(direct_24h, _multiply(sunlit_fraction_24h, diffuse_24h)),
        jnp.float64(4.6),
    )
    par240_sun = _multiply(
        _add(direct_240h, _multiply(sunlit_fraction_240h, diffuse_240h)),
        jnp.float64(4.6),
    )
    par_shade = _multiply(
        _multiply(_subtract(jnp.float64(1.0), sunlit_fraction), diffuse),
        jnp.float64(4.6),
    )
    par24_shade = _multiply(
        _multiply(_subtract(jnp.float64(1.0), sunlit_fraction_24h), diffuse_24h),
        jnp.float64(4.6),
    )
    par240_shade = _multiply(
        _multiply(_subtract(jnp.float64(1.0), sunlit_fraction_240h), diffuse_240h),
        jnp.float64(4.6),
    )

    leaf_area = patch_value(inputs.exposed_leaf_area_index_patch, 1.0)
    leaf_area_240h = patch_value(inputs.exposed_leaf_area_index_240h_patch, 1.0)
    gamma_l = jnp.where(
        _positive_binary64(sunlit_fraction_240h) & (sunlit_fraction_240h < jnp.float64(1.0e30)),
        _multiply(jnp.float64(0.30), leaf_area),
        _multiply(jnp.float64(0.24), leaf_area),
    )
    gamma_sm = jnp.ones_like(gamma_l)

    compound_class = jnp.asarray(parameters.compound_class, dtype=jnp.int32)
    ldf = jnp.asarray(parameters.light_dependent_fraction, dtype=jnp.float64)[compound_class]
    gamma_p, cp, alpha = _gamma_p(
        par_sun,
        par24_sun,
        par240_sun,
        par_shade,
        par240_shade,
        sunlit_fraction,
        sunlit_fraction_240h,
        direct_240h,
        diffuse_240h,
        ldf,
    )
    temperature = patch_value(inputs.vegetation_temperature_patch, 303.15)
    temperature_24h = patch_value(inputs.vegetation_temperature_24h_patch, 297.0)
    temperature_240h = patch_value(inputs.vegetation_temperature_240h_patch, 297.0)
    gamma_t, optimum_emission, optimum_temperature = _gamma_t(
        temperature_240h,
        temperature_24h,
        temperature,
        jnp.asarray(parameters.temperature_ct1, dtype=jnp.float64)[compound_class],
        jnp.asarray(parameters.temperature_ct2, dtype=jnp.float64)[compound_class],
        jnp.asarray(parameters.temperature_beta, dtype=jnp.float64)[compound_class],
        ldf,
        jnp.asarray(parameters.temperature_ceo, dtype=jnp.float64)[compound_class],
    )
    gamma_a = _gamma_a(
        vegetation_type,
        leaf_area_240h,
        leaf_area,
        compound_class,
        parameters,
    )

    pressure = jnp.asarray(inputs.air_pressure_column, dtype=jnp.float64)[column_index]
    pressure = jnp.where(compute, pressure, jnp.float64(100000.0))
    co2_partial_pressure = jnp.asarray(inputs.co2_partial_pressure_grid, dtype=jnp.float64)[
        grid_index
    ]
    co2_partial_pressure = jnp.where(compute, co2_partial_pressure, jnp.float64(40.0))
    co2_ppmv = fortran_divide(
        _multiply(jnp.float64(1.0e6), co2_partial_pressure),
        pressure,
    )
    gamma_c_isoprene = _gamma_c(
        patch_value(inputs.sunlit_intracellular_co2_patch, 28.0),
        patch_value(inputs.shaded_intracellular_co2_patch, 28.0),
        pressure,
        sunlit_fraction,
        co2_ppmv,
    )
    is_isoprene = jnp.asarray(parameters.is_isoprene, dtype=jnp.bool_)
    gamma_c = jnp.where(is_isoprene[None, :], gamma_c_isoprene[:, None], jnp.float64(1.0))

    gamma = _multiply(
        gamma_l[:, None],
        gamma_sm[:, None],
        gamma_a,
        gamma_p,
        gamma_t,
        gamma_c,
    )
    valid_gamma = (gamma >= jnp.float64(0.0)) & (gamma < jnp.float64(100.0))

    scalar_factor = jnp.asarray(parameters.emission_factor_by_compound_pft, dtype=jnp.float64)[
        :, vegetation_type
    ].T
    mapped_category = jnp.asarray(parameters.mapped_factor_category_by_pft, dtype=jnp.int32)[
        vegetation_type
    ]
    safe_category = jnp.clip(mapped_category, 0, 5)
    mapped_factor = jnp.asarray(inputs.mapped_isoprene_emission_factor_grid, dtype=jnp.float64)[
        grid_index, safe_category
    ]
    mapped_enabled = jnp.asarray(parameters.use_mapped_isoprene_factors, dtype=jnp.bool_)
    epsilon = jnp.where(
        mapped_enabled & is_isoprene[None, :],
        mapped_factor[:, None],
        scalar_factor,
    )

    compound_flux = _multiply(
        jnp.asarray(parameters.compound_coefficient, dtype=jnp.float64)[None, :],
        epsilon,
        gamma,
        jnp.float64(1.0 / 3600.0 / 1.0e6),
    )
    compound_flux = fortran_divide(
        compound_flux,
        jnp.asarray(parameters.compound_molecular_weight, dtype=jnp.float64)[None, :],
    )
    compound_flux = jnp.where(compute[:, None] & valid_gamma, compound_flux, jnp.float64(0.0))
    compound_mass_flux = _multiply(
        epsilon,
        gamma,
        jnp.float64(1.0 / 3600.0 / 1.0e6),
        jnp.float64(1.0e-3),
    )
    compound_mass_flux = jnp.where(
        compute[:, None] & valid_gamma,
        compound_mass_flux,
        jnp.float64(0.0),
    )

    mechanism_index = jnp.asarray(parameters.mechanism_compound_index, dtype=jnp.int32)
    mechanism_mask = jnp.asarray(parameters.mechanism_compound_mask, dtype=jnp.bool_)
    number_of_mechanisms, maximum_terms = mechanism_index.shape

    def add_mechanism_term(term_index: int, running: jnp.ndarray) -> jnp.ndarray:
        indices = jnp.clip(mechanism_index[:, term_index], 0, compound_flux.shape[1] - 1)
        term = compound_flux[:, indices]
        return jnp.where(
            mechanism_mask[None, :, term_index],
            fortran_add(running, term),
            running,
        )

    mechanism_flux = lax.fori_loop(
        0,
        maximum_terms,
        add_mechanism_term,
        jnp.zeros((grid_index.shape[0], number_of_mechanisms), dtype=jnp.float64),
    )

    def add_mechanism(mechanism: int, running: jnp.ndarray) -> jnp.ndarray:
        return fortran_add(running, mechanism_flux[:, mechanism])

    total_flux = lax.fori_loop(
        0,
        number_of_mechanisms,
        add_mechanism,
        jnp.zeros(grid_index.shape[0], dtype=jnp.float64),
    )

    first_valid = compute & valid_gamma[:, 0]
    old = state.diagnostics

    def update(previous: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(first_valid, value, previous)

    diagnostics = VOCDiagnostics(
        optimum_emission_factor_patch=update(
            old.optimum_emission_factor_patch, optimum_emission[:, 0]
        ),
        optimum_temperature_patch=update(old.optimum_temperature_patch, optimum_temperature),
        light_alpha_patch=update(old.light_alpha_patch, alpha),
        light_cp_patch=update(old.light_cp_patch, cp),
        sunlit_par_patch=update(old.sunlit_par_patch, par_sun),
        sunlit_par_24h_patch=update(old.sunlit_par_24h_patch, par24_sun),
        sunlit_par_240h_patch=update(old.sunlit_par_240h_patch, par240_sun),
        shaded_par_patch=update(old.shaded_par_patch, par_shade),
        shaded_par_24h_patch=update(old.shaded_par_24h_patch, par24_shade),
        shaded_par_240h_patch=update(old.shaded_par_240h_patch, par240_shade),
        total_activity_factor_patch=update(old.total_activity_factor_patch, gamma[:, 0]),
        leaf_area_activity_factor_patch=update(old.leaf_area_activity_factor_patch, gamma_l),
        temperature_activity_factor_patch=update(
            old.temperature_activity_factor_patch, gamma_t[:, 0]
        ),
        light_activity_factor_patch=update(old.light_activity_factor_patch, gamma_p[:, 0]),
        leaf_age_activity_factor_patch=update(old.leaf_age_activity_factor_patch, gamma_a[:, 0]),
        soil_moisture_activity_factor_patch=update(
            old.soil_moisture_activity_factor_patch, gamma_sm
        ),
        co2_activity_factor_patch=update(old.co2_activity_factor_patch, gamma_c[:, 0]),
    )
    return VOCEmissionResult(
        state=VOCState(
            mechanism_flux_patch=mechanism_flux,
            total_flux_patch=total_flux,
            compound_mass_flux_patch=compound_mass_flux,
            diagnostics=diagnostics,
        )
    )
