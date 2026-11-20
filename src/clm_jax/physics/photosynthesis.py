
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import default_backend, lax, tree_util

from clm_jax.constants import (
    FREEZING_TEMPERATURE_FRESHWATER,
    UNIVERSAL_GAS_CONSTANT,
)
from clm_jax.physics.converged_iteration import converged_iteration
from clm_jax.physics.native_power import native_power
from clm_jax.physics.pointwise import scalar_map
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_divide as fortran_divide,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.numerics import (
    fortran_divide as rounded_divide,
    fortran_linear_combination,
    fortran_materialize,
    strict_divide,
    strict_exp,
)
from clm_jax.physics.plant_hydraulics import (
    PlantHydraulicPatchInputs,
    PlantHydraulicPatchParameters,
    PlantHydraulicPatchState,
    plant_hydraulic_stress,
    vegetation_water_potential,
)

SUNLIT = 0
SHADED = 1
NUMBER_OF_LEAF_CLASSES = 2
THETA_IP = 0.95
MINIMUM_LEAF_SURFACE_CO2 = 10.0e-6
HYBRID_RELATIVE_CI_TOLERANCE = 1.0e-2
HYBRID_ABSOLUTE_RESIDUAL_TOLERANCE = 1.0e-4
HYDRAULIC_STRESS_TOLERANCE = 1.0e-2
MAXIMUM_HYBRID_OUTER_ITERATIONS = 4
MAXIMUM_SECANT_ITERATIONS = 4
MAXIMUM_BRENT_ITERATIONS = 20
BRENT_RELATIVE_TOLERANCE = 1.0e-4
FRACTION_ABSORBED_BY_NONPHOTOSYNTHETIC_PIGMENTS = 0.15
ELECTRON_TRANSPORT_CURVATURE = 0.7
_USE_FUSED_HYDRAULIC_PHOTOSYNTHESIS = False

MICHAELIS_MENTEN_CO2_ACTIVATION_ENERGY = 79_430.0
MICHAELIS_MENTEN_O2_ACTIVATION_ENERGY = 36_380.0
CO2_COMPENSATION_POINT_ACTIVATION_ENERGY = 37_830.0
MAXIMUM_CARBOXYLATION_ACTIVATION_ENERGY = 72_000.0
MAXIMUM_ELECTRON_TRANSPORT_ACTIVATION_ENERGY = 50_000.0
TRIOSE_PHOSPHATE_UTILIZATION_ACTIVATION_ENERGY = 72_000.0
LEAF_RESPIRATION_ACTIVATION_ENERGY = 46_390.0

PHOTOSYNTHESIS_DEACTIVATION_ENERGY = 200_000.0
LEAF_RESPIRATION_DEACTIVATION_ENERGY = 150_650.0
LEAF_RESPIRATION_ENTROPY = 490.0


class QuadraticRoots(NamedTuple):
    first: jnp.ndarray
    second: jnp.ndarray


class HydraulicPhotosynthesisInputs(NamedTuple):

    c3_pathway: jnp.ndarray
    intracellular_co2: jnp.ndarray
    hydraulic_stress: jnp.ndarray
    boundary_layer_conductance: jnp.ndarray
    electron_transport_rate: jnp.ndarray
    atmospheric_co2: jnp.ndarray
    atmospheric_o2: jnp.ndarray
    atmospheric_pressure: jnp.ndarray
    leaf_maintenance_respiration: jnp.ndarray
    absorbed_par: jnp.ndarray
    canopy_vapor_pressure_deficit: jnp.ndarray
    maximum_carboxylation_rate: jnp.ndarray
    co2_compensation_point: jnp.ndarray
    michaelis_menten_co2: jnp.ndarray
    michaelis_menten_o2: jnp.ndarray
    quantum_efficiency_c4: jnp.ndarray
    triose_phosphate_utilization_rate: jnp.ndarray
    pep_carboxylase_co2_slope: jnp.ndarray
    ac_aj_colimitation_curvature: jnp.ndarray
    medlyn_slope: jnp.ndarray
    medlyn_intercept: jnp.ndarray


class HydraulicPhotosynthesisBiochemicalParameters(NamedTuple):

    c3_pathway: jnp.ndarray
    maximum_carboxylation_rate_25: jnp.ndarray
    maximum_electron_transport_rate_25: jnp.ndarray
    triose_phosphate_utilization_rate_25: jnp.ndarray
    pep_carboxylase_co2_slope_25: jnp.ndarray
    leaf_maintenance_respiration_25: jnp.ndarray
    absorbed_par: jnp.ndarray
    layer_leaf_area: jnp.ndarray
    ten_day_temperature: jnp.ndarray
    atmospheric_co2: jnp.ndarray
    atmospheric_o2: jnp.ndarray
    atmospheric_pressure: jnp.ndarray
    quantum_efficiency_c4: jnp.ndarray
    ac_aj_colimitation_curvature: jnp.ndarray
    medlyn_slope: jnp.ndarray
    medlyn_intercept: jnp.ndarray
    light_inhibition: jnp.ndarray


class HydraulicPhotosynthesisEnvironment(NamedTuple):

    leaf_temperature: jnp.ndarray
    leaf_saturation_vapor_pressure: jnp.ndarray
    canopy_vapor_pressure: jnp.ndarray
    boundary_layer_resistance: jnp.ndarray
    reference_temperature: jnp.ndarray


class HydraulicPhotosynthesisPreparation(NamedTuple):

    inputs: HydraulicPhotosynthesisInputs
    daylight: jnp.ndarray
    molar_conversion_factor: jnp.ndarray
    maximum_electron_transport_rate: jnp.ndarray


class HydraulicPhotosynthesisPatchControls(NamedTuple):

    crop: jnp.ndarray
    modify_photosynthesis_and_respiration_for_crop: jnp.ndarray
    sunlit_stomatal_ozone_coefficient: jnp.ndarray
    shaded_stomatal_ozone_coefficient: jnp.ndarray


class HydraulicPhotosynthesisResult(NamedTuple):
    rubisco_limited_rate: jnp.ndarray
    electron_transport_limited_rate: jnp.ndarray
    product_limited_rate: jnp.ndarray
    gross_photosynthesis: jnp.ndarray
    net_photosynthesis: jnp.ndarray
    leaf_surface_co2: jnp.ndarray
    stomatal_conductance: jnp.ndarray
    ci_residual: jnp.ndarray


class CoupledHydraulicPhotosynthesisResult(NamedTuple):
    intracellular_co2: jnp.ndarray
    hydraulic_stress: jnp.ndarray
    stomatal_conductance: jnp.ndarray
    photosynthesis: HydraulicPhotosynthesisResult
    plant_state: PlantHydraulicPatchState
    outer_iterations: jnp.ndarray
    inner_iterations: jnp.ndarray
    hydraulic_iterations: jnp.ndarray


class HydraulicPhotosynthesisPatchResult(NamedTuple):

    preparation: HydraulicPhotosynthesisPreparation
    coupled: CoupledHydraulicPhotosynthesisResult
    solver_intracellular_co2: jnp.ndarray
    stomatal_resistance: jnp.ndarray
    transpiration_wetness: jnp.ndarray
    hydraulic_iterations: jnp.ndarray


class PhotosynthesisTotalInputs(NamedTuple):

    sunlit_leaf_area_patch: jnp.ndarray
    shaded_leaf_area_patch: jnp.ndarray
    sunlit_photosynthesis_patch: jnp.ndarray
    shaded_photosynthesis_patch: jnp.ndarray
    sunlit_rubisco_limited_patch: jnp.ndarray
    shaded_rubisco_limited_patch: jnp.ndarray
    sunlit_electron_transport_limited_patch: jnp.ndarray
    shaded_electron_transport_limited_patch: jnp.ndarray
    sunlit_product_limited_patch: jnp.ndarray
    shaded_product_limited_patch: jnp.ndarray


class PhotosynthesisTotalState(NamedTuple):

    photosynthesis_patch: jnp.ndarray
    rubisco_limited_photosynthesis_patch: jnp.ndarray
    electron_transport_limited_photosynthesis_patch: jnp.ndarray
    product_limited_photosynthesis_patch: jnp.ndarray


class _BrentState(NamedTuple):
    a: jnp.ndarray
    b: jnp.ndarray
    c: jnp.ndarray
    d: jnp.ndarray
    e: jnp.ndarray
    fa: jnp.ndarray
    fb: jnp.ndarray
    fc: jnp.ndarray
    result: HydraulicPhotosynthesisResult
    active: jnp.ndarray


class _SecantState(NamedTuple):
    x0: jnp.ndarray
    x1: jnp.ndarray
    f0: jnp.ndarray
    f1: jnp.ndarray
    minimum_residual: jnp.ndarray
    minimum_ci: jnp.ndarray
    result: HydraulicPhotosynthesisResult
    active: jnp.ndarray
    iterations: jnp.ndarray


class _OuterState(NamedTuple):
    ci: jnp.ndarray
    hydraulic_stress: jnp.ndarray
    previous_hydraulic_stress: jnp.ndarray
    unstressed_conductance: jnp.ndarray
    result: HydraulicPhotosynthesisResult
    transpiration: jnp.ndarray
    active: jnp.ndarray
    outer_iterations: jnp.ndarray
    inner_iterations: jnp.ndarray
    hydraulic_iterations: jnp.ndarray


def _product(*factors: jnp.ndarray) -> jnp.ndarray:
    result = jnp.asarray(factors[0], dtype=jnp.float64)
    for factor in factors[1:]:
        result = fortran_multiply(result, factor)
    return result


def _sum(*terms: jnp.ndarray) -> jnp.ndarray:
    broadcast = jnp.broadcast_arrays(*(jnp.asarray(term, dtype=jnp.float64) for term in terms))
    result = broadcast[0]
    for term in broadcast[1:]:
        result = fortran_add(result, term)
    return result


def _temperature_activation(
    temperature: jnp.ndarray,
    activation_energy: float,
) -> jnp.ndarray:
    reference_temperature = _sum(
        jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
        jnp.float64(25.0),
    )
    denominator = _product(
        jnp.float64(UNIVERSAL_GAS_CONSTANT),
        jnp.float64(1.0e-3),
        reference_temperature,
    )
    exponent = _product(
        fortran_divide(jnp.float64(activation_energy), denominator),
        _sum(
            jnp.float64(1.0),
            -fortran_divide(reference_temperature, temperature),
        ),
    )
    return strict_exp(exponent)


def _temperature_inhibition_at_25(
    deactivation_energy: float,
    entropy: jnp.ndarray,
) -> jnp.ndarray:
    reference_temperature = _sum(
        jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
        jnp.float64(25.0),
    )
    numerator = _sum(
        -jnp.float64(deactivation_energy),
        _product(entropy, reference_temperature),
    )
    denominator = _product(
        jnp.float64(UNIVERSAL_GAS_CONSTANT),
        jnp.float64(1.0e-3),
        reference_temperature,
    )
    return _sum(
        jnp.float64(1.0),
        strict_exp(rounded_divide(numerator, denominator)),
    )


def _temperature_inhibition(
    temperature: jnp.ndarray,
    deactivation_energy: float,
    entropy: jnp.ndarray,
    scale_factor: jnp.ndarray,
) -> jnp.ndarray:
    numerator = _sum(
        -jnp.float64(deactivation_energy),
        _product(entropy, temperature),
    )
    denominator = _product(
        jnp.float64(UNIVERSAL_GAS_CONSTANT),
        jnp.float64(1.0e-3),
        temperature,
    )
    return fortran_divide(
        scale_factor,
        _sum(
            jnp.float64(1.0),
            strict_exp(fortran_divide(numerator, denominator)),
        ),
    )


def _base_two_temperature_response(
    temperature: jnp.ndarray,
) -> jnp.ndarray:
    reference_temperature = _sum(
        jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
        jnp.float64(25.0),
    )
    exponent = rounded_divide(
        _sum(temperature, -reference_temperature),
        jnp.float64(10.0),
    )
    return native_power(jnp.float64(2.0), exponent)


def _validate_preparation_inputs(
    parameters: HydraulicPhotosynthesisBiochemicalParameters,
    environment: HydraulicPhotosynthesisEnvironment,
) -> None:
    leaf_fields = (
        parameters.maximum_carboxylation_rate_25,
        parameters.maximum_electron_transport_rate_25,
        parameters.triose_phosphate_utilization_rate_25,
        parameters.pep_carboxylase_co2_slope_25,
        parameters.leaf_maintenance_respiration_25,
        parameters.absorbed_par,
    )
    if any(field.shape != (NUMBER_OF_LEAF_CLASSES,) for field in leaf_fields):
        raise ValueError("sunlit/shaded biochemical parameters must have shape (2,)")
    scalar_fields = (
        parameters.c3_pathway,
        parameters.layer_leaf_area,
        parameters.ten_day_temperature,
        parameters.atmospheric_co2,
        parameters.atmospheric_o2,
        parameters.atmospheric_pressure,
        parameters.quantum_efficiency_c4,
        parameters.ac_aj_colimitation_curvature,
        parameters.medlyn_slope,
        parameters.medlyn_intercept,
        parameters.light_inhibition,
        *environment,
    )
    if any(field.shape != () for field in scalar_fields):
        raise ValueError("patch biochemical environment fields must be scalar")


def quadratic_roots(
    coefficient_a: jnp.ndarray,
    coefficient_b: jnp.ndarray,
    coefficient_c: jnp.ndarray,
) -> QuadraticRoots:

    a, b, c = jnp.broadcast_arrays(
        jnp.asarray(coefficient_a, dtype=jnp.float64),
        jnp.asarray(coefficient_b, dtype=jnp.float64),
        jnp.asarray(coefficient_c, dtype=jnp.float64),
    )
    discriminant = _sum(
        _product(b, b),
        -_product(jnp.float64(4.0), a, c),
    )
    roundoff_negative = (discriminant < jnp.float64(0.0)) & (
        -discriminant < _product(jnp.float64(3.0), jnp.finfo(jnp.float64).eps)
    )
    discriminant = jnp.where(roundoff_negative, jnp.float64(0.0), discriminant)
    square_root = jnp.sqrt(discriminant)
    signed_sum = jnp.where(
        b >= jnp.float64(0.0),
        _sum(b, square_root),
        _sum(b, -square_root),
    )
    temporary = -_product(jnp.float64(0.5), signed_sum)
    first = fortran_divide(temporary, a)
    second = jnp.where(
        temporary != jnp.float64(0.0),
        fortran_divide(c, jnp.where(temporary != 0., temporary, jnp.float64(1.0))),
        jnp.float64(1.0e36),
    )
    return QuadraticRoots(first, second)


def _colimitation_reference(first, second, curvature):
    roots = quadratic_roots(curvature, -_sum(first, second), _product(first, second))
    return jnp.minimum(roots.first, roots.second)


@jax.custom_jvp
def _photosynthetic_colimitation(first, second, curvature):
    return _colimitation_reference(first, second, curvature)


@_photosynthetic_colimitation.defjvp
def _colimitation_jvp(primals, tangents):
    def physical_rates(first, second, curvature):
        absent = (first == 0.) & (second == 0.)
        value = _colimitation_reference(jnp.where(absent, 1., first),
                                        jnp.where(absent, 2., second), curvature)
        return jnp.where(absent, 0., value)

    return _photosynthetic_colimitation(*primals), jax.jvp(physical_rates, primals, tangents)[1]


def prepare_hydraulic_photosynthesis_inputs(
    parameters: HydraulicPhotosynthesisBiochemicalParameters,
    environment: HydraulicPhotosynthesisEnvironment,
) -> HydraulicPhotosynthesisPreparation:
    _validate_preparation_inputs(parameters, environment)
    temperature = jnp.asarray(environment.leaf_temperature, dtype=jnp.float64)
    pressure = jnp.asarray(parameters.atmospheric_pressure, dtype=jnp.float64)
    freezing = jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)
    kc25 = _product(fortran_divide(jnp.float64(404.9), jnp.float64(1000000.0)), pressure)
    ko25 = _product(fortran_divide(jnp.float64(278.4), jnp.float64(1000.0)), pressure)
    rubisco_specificity = fortran_divide(
        _product(jnp.float64(0.5), jnp.float64(0.209)),
        fortran_divide(jnp.float64(42.75), jnp.float64(1000000.0)),
    )
    cp25 = rounded_divide(
        _product(jnp.float64(0.5), parameters.atmospheric_o2), rubisco_specificity
    )
    michaelis_co2 = _product(
        kc25, _temperature_activation(temperature, MICHAELIS_MENTEN_CO2_ACTIVATION_ENERGY)
    )
    michaelis_o2 = _product(
        ko25, _temperature_activation(temperature, MICHAELIS_MENTEN_O2_ACTIVATION_ENERGY)
    )
    compensation_point = _product(
        cp25, _temperature_activation(temperature, CO2_COMPENSATION_POINT_ACTIVATION_ENERGY)
    )
    base_two_response = _base_two_temperature_response(temperature)

    def c3_respiration_rate(_):
        respiration_scale = _temperature_inhibition_at_25(
            LEAF_RESPIRATION_DEACTIVATION_ENERGY, jnp.float64(LEAF_RESPIRATION_ENTROPY)
        )
        c3_respiration = _product(
            parameters.leaf_maintenance_respiration_25,
            _temperature_activation(temperature, LEAF_RESPIRATION_ACTIVATION_ENERGY),
            _temperature_inhibition(
                temperature,
                LEAF_RESPIRATION_DEACTIVATION_ENERGY,
                jnp.float64(LEAF_RESPIRATION_ENTROPY),
                respiration_scale,
            ),
        )
        return c3_respiration

    def c4_respiration_rate(_):
        c4_respiration = fortran_divide(
            _product(parameters.leaf_maintenance_respiration_25, base_two_response),
            _sum(
                jnp.float64(1.0),
                strict_exp(
                    _product(
                        jnp.float64(1.3), _sum(temperature, -_sum(freezing, jnp.float64(55.0)))
                    )
                ),
            ),
        )
        return c4_respiration

    respiration = lax.cond(parameters.c3_pathway, c3_respiration_rate, c4_respiration_rate, None)
    low_leaf_area_factor = jnp.minimum(
        _product(
            jnp.float64(0.2), strict_exp(_product(jnp.float64(3.218), parameters.layer_leaf_area))
        ),
        jnp.float64(1.0),
    )
    respiration = _product(respiration, low_leaf_area_factor)
    respiration = jnp.where(
        parameters.light_inhibition & (parameters.absorbed_par > jnp.float64(0.0)),
        _product(respiration, jnp.float64(0.67)),
        respiration,
    )
    daylight = parameters.absorbed_par[SUNLIT] > jnp.float64(0.0)

    def day_capacities(_):
        acclimation_temperature = jnp.minimum(
            jnp.maximum(_sum(parameters.ten_day_temperature, -freezing), jnp.float64(11.0)),
            jnp.float64(35.0),
        )
        vcmax_entropy = _sum(
            jnp.float64(668.39), -_product(jnp.float64(1.07), acclimation_temperature)
        )
        jmax_entropy = _sum(
            jnp.float64(659.7), -_product(jnp.float64(0.75), acclimation_temperature)
        )
        vcmax_scale = _temperature_inhibition_at_25(
            PHOTOSYNTHESIS_DEACTIVATION_ENERGY, vcmax_entropy
        )
        jmax_scale = _temperature_inhibition_at_25(PHOTOSYNTHESIS_DEACTIVATION_ENERGY, jmax_entropy)
        tpu_scale = _temperature_inhibition_at_25(PHOTOSYNTHESIS_DEACTIVATION_ENERGY, vcmax_entropy)

        def c3_capacity(_):
            vcmax = _product(
                parameters.maximum_carboxylation_rate_25,
                _temperature_activation(temperature, MAXIMUM_CARBOXYLATION_ACTIVATION_ENERGY),
                _temperature_inhibition(
                    temperature, PHOTOSYNTHESIS_DEACTIVATION_ENERGY, vcmax_entropy, vcmax_scale
                ),
            )
            return vcmax

        def c4_capacity(_):
            c4_vcmax = fortran_divide(
                strict_divide(
                    _product(parameters.maximum_carboxylation_rate_25, base_two_response),
                    _sum(
                        jnp.float64(1.0),
                        strict_exp(
                            _product(
                                jnp.float64(0.2),
                                _sum(_sum(freezing, jnp.float64(15.0)), -temperature),
                            )
                        ),
                    ),
                ),
                _sum(
                    jnp.float64(1.0),
                    strict_exp(
                        _product(
                            jnp.float64(0.3), _sum(temperature, -_sum(freezing, jnp.float64(40.0)))
                        )
                    ),
                ),
            )
            return c4_vcmax

        maximum_carboxylation = lax.cond(parameters.c3_pathway, c3_capacity, c4_capacity, None)
        jmax = _product(
            parameters.maximum_electron_transport_rate_25,
            _temperature_activation(temperature, MAXIMUM_ELECTRON_TRANSPORT_ACTIVATION_ENERGY),
            _temperature_inhibition(
                temperature, PHOTOSYNTHESIS_DEACTIVATION_ENERGY, jmax_entropy, jmax_scale
            ),
        )
        tpu = _product(
            parameters.triose_phosphate_utilization_rate_25,
            _temperature_activation(temperature, TRIOSE_PHOSPHATE_UTILIZATION_ACTIVATION_ENERGY),
            _temperature_inhibition(
                temperature, PHOTOSYNTHESIS_DEACTIVATION_ENERGY, vcmax_entropy, tpu_scale
            ),
        )
        pep_slope = _product(parameters.pep_carboxylase_co2_slope_25, base_two_response)
        qabs = _product(
            jnp.float64(0.5),
            _sum(jnp.float64(1.0), -jnp.float64(FRACTION_ABSORBED_BY_NONPHOTOSYNTHETIC_PIGMENTS)),
            parameters.absorbed_par,
            jnp.float64(4.6),
        )
        electron_transport = _photosynthetic_colimitation(
            qabs, jmax, jnp.float64(ELECTRON_TRANSPORT_CURVATURE)
        )
        return (maximum_carboxylation, jmax, tpu, pep_slope, electron_transport)

    def night_capacities(_):
        zeros = jnp.zeros((NUMBER_OF_LEAF_CLASSES,), dtype=jnp.float64)
        return (zeros, zeros, zeros, zeros, zeros)

    maximum_carboxylation, jmax, tpu, pep_slope, electron_transport = lax.cond(
        daylight, day_capacities, night_capacities, None
    )
    constrained_canopy_vapor_pressure = fortran_materialize(
        jnp.minimum(environment.canopy_vapor_pressure, environment.leaf_saturation_vapor_pressure)
    )
    vapor_pressure_difference = fortran_materialize(
        _sum(environment.leaf_saturation_vapor_pressure, -constrained_canopy_vapor_pressure)
    )
    vapor_pressure_deficit = fortran_materialize(
        _product(jnp.maximum(vapor_pressure_difference, jnp.float64(50.0)), jnp.float64(0.001))
    )
    conversion_denominator = _product(
        jnp.float64(UNIVERSAL_GAS_CONSTANT), jnp.float64(0.001), environment.reference_temperature
    )
    molar_conversion = _product(
        fortran_divide(pressure, conversion_denominator), jnp.float64(1000000.0)
    )
    boundary_layer_conductance = _product(
        fortran_divide(jnp.float64(1.0), environment.boundary_layer_resistance), molar_conversion
    )
    initial_ci_fraction = jnp.where(parameters.c3_pathway, jnp.float64(0.7), jnp.float64(0.4))
    initial_ci = jnp.where(
        daylight,
        _product(initial_ci_fraction, parameters.atmospheric_co2),
        jnp.zeros((NUMBER_OF_LEAF_CLASSES,), dtype=jnp.float64),
    )
    inputs = HydraulicPhotosynthesisInputs(
        c3_pathway=jnp.asarray(parameters.c3_pathway, dtype=jnp.bool_),
        intracellular_co2=jnp.broadcast_to(initial_ci, (NUMBER_OF_LEAF_CLASSES,)),
        hydraulic_stress=jnp.ones((NUMBER_OF_LEAF_CLASSES,), dtype=jnp.float64),
        boundary_layer_conductance=boundary_layer_conductance,
        electron_transport_rate=electron_transport,
        atmospheric_co2=parameters.atmospheric_co2,
        atmospheric_o2=parameters.atmospheric_o2,
        atmospheric_pressure=pressure,
        leaf_maintenance_respiration=respiration,
        absorbed_par=parameters.absorbed_par,
        canopy_vapor_pressure_deficit=vapor_pressure_deficit,
        maximum_carboxylation_rate=maximum_carboxylation,
        co2_compensation_point=compensation_point,
        michaelis_menten_co2=michaelis_co2,
        michaelis_menten_o2=michaelis_o2,
        quantum_efficiency_c4=parameters.quantum_efficiency_c4,
        triose_phosphate_utilization_rate=tpu,
        pep_carboxylase_co2_slope=pep_slope,
        ac_aj_colimitation_curvature=parameters.ac_aj_colimitation_curvature,
        medlyn_slope=parameters.medlyn_slope,
        medlyn_intercept=parameters.medlyn_intercept,
    )
    preparation = HydraulicPhotosynthesisPreparation(
        inputs=inputs,
        daylight=daylight,
        molar_conversion_factor=molar_conversion,
        maximum_electron_transport_rate=jmax,
    )

    def materialize_prepared_value(value: jnp.ndarray) -> jnp.ndarray:
        if jnp.issubdtype(value.dtype, jnp.floating):
            return fortran_materialize(value)
        return lax.optimization_barrier(value)

    return tree_util.tree_map(materialize_prepared_value, preparation)


def photosynthesis_total(
    active_patch: jnp.ndarray,
    inputs: PhotosynthesisTotalInputs,
    state: PhotosynthesisTotalState,
) -> PhotosynthesisTotalState:

    active = jnp.asarray(active_patch, dtype=jnp.bool_)
    input_arrays = tuple(jnp.asarray(value, dtype=jnp.float64) for value in inputs)
    state_arrays = tuple(jnp.asarray(value, dtype=jnp.float64) for value in state)
    if active.ndim != 1:
        raise ValueError("active_patch must have shape (patch,)")
    if any(value.shape != active.shape for value in input_arrays + state_arrays):
        raise ValueError("photosynthesis total fields must contain one value per patch")

    leaf_area = jnp.stack((input_arrays[0], input_arrays[1]))

    def canopy_total(sunlit: jnp.ndarray, shaded: jnp.ndarray) -> jnp.ndarray:
        return fortran_linear_combination(jnp.stack((sunlit, shaded)), leaf_area)

    candidates = (
        canopy_total(input_arrays[2], input_arrays[3]),
        canopy_total(input_arrays[4], input_arrays[5]),
        canopy_total(input_arrays[6], input_arrays[7]),
        canopy_total(input_arrays[8], input_arrays[9]),
    )
    return PhotosynthesisTotalState(
        *(
            jnp.where(active, candidate, previous)
            for candidate, previous in zip(candidates, state_arrays, strict=True)
        )
    )


def _validate_inputs(inputs: HydraulicPhotosynthesisInputs) -> None:
    leaf_fields = (
        inputs.intracellular_co2,
        inputs.hydraulic_stress,
        inputs.electron_transport_rate,
        inputs.leaf_maintenance_respiration,
        inputs.absorbed_par,
        inputs.maximum_carboxylation_rate,
        inputs.triose_phosphate_utilization_rate,
        inputs.pep_carboxylase_co2_slope,
    )
    if any(field.shape != (NUMBER_OF_LEAF_CLASSES,) for field in leaf_fields):
        raise ValueError("sunlit/shaded photosynthesis inputs must have shape (2,)")
    scalar_fields = (
        inputs.c3_pathway,
        inputs.boundary_layer_conductance,
        inputs.atmospheric_co2,
        inputs.atmospheric_o2,
        inputs.atmospheric_pressure,
        inputs.canopy_vapor_pressure_deficit,
        inputs.co2_compensation_point,
        inputs.michaelis_menten_co2,
        inputs.michaelis_menten_o2,
        inputs.quantum_efficiency_c4,
        inputs.ac_aj_colimitation_curvature,
        inputs.medlyn_slope,
        inputs.medlyn_intercept,
    )
    if any(field.shape != () for field in scalar_fields):
        raise ValueError("patch photosynthesis parameters must be scalar")


def _limitation_rates(
    inputs: HydraulicPhotosynthesisInputs,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    ci = inputs.intracellular_co2
    stress = inputs.hydraulic_stress

    def c3_rates(_: None):
        co2_excess = jnp.maximum(_sum(ci, -inputs.co2_compensation_point), jnp.float64(0.0))
        rubisco_denominator = _sum(
            ci,
            _product(
                inputs.michaelis_menten_co2,
                _sum(
                    jnp.float64(1.0),
                    fortran_divide(
                        inputs.atmospheric_o2,
                        inputs.michaelis_menten_o2,
                    ),
                ),
            ),
        )
        rubisco = fortran_divide(
            _product(
                stress,
                inputs.maximum_carboxylation_rate,
                co2_excess,
            ),
            rubisco_denominator,
        )
        electron = fortran_divide(
            _product(inputs.electron_transport_rate, co2_excess),
            _sum(
                _product(jnp.float64(4.0), ci),
                _product(jnp.float64(8.0), inputs.co2_compensation_point),
            ),
        )
        product = _product(jnp.float64(3.0), inputs.triose_phosphate_utilization_rate)
        return rubisco, electron, product

    def c4_rates(_: None):
        rubisco = _product(stress, inputs.maximum_carboxylation_rate)
        electron = _product(
            inputs.quantum_efficiency_c4,
            inputs.absorbed_par,
            jnp.float64(4.6),
        )
        product = fortran_divide(
            _product(
                inputs.pep_carboxylase_co2_slope,
                jnp.maximum(ci, jnp.float64(0.0)),
            ),
            inputs.atmospheric_pressure,
        )
        return rubisco, electron, product

    return lax.cond(inputs.c3_pathway, c3_rates, c4_rates, operand=None)


def _gross_photosynthesis(
    rubisco: jnp.ndarray,
    electron: jnp.ndarray,
    product: jnp.ndarray,
    curvature: jnp.ndarray,
) -> jnp.ndarray:
    intermediate = _photosynthetic_colimitation(rubisco, electron, curvature)
    combined = _photosynthetic_colimitation(intermediate, product, jnp.float64(THETA_IP))
    return jnp.maximum(jnp.float64(0.0), combined)


def _daylight_stomatal_resistance(conductance, conversion, ozone):
    velocity = rounded_divide(conductance, conversion)
    resistance = jnp.minimum(
        rounded_divide(jnp.float64(1.0), velocity), jnp.float64(20_000.0)
    )
    return rounded_divide(resistance, ozone)


def _medlyn_conductance_reference(
    net_photosynthesis: jnp.ndarray,
    leaf_surface_co2: jnp.ndarray,
    inputs: HydraulicPhotosynthesisInputs,
) -> jnp.ndarray:
    co2_mole_fraction = _product(
        fortran_divide(leaf_surface_co2, inputs.atmospheric_pressure),
        jnp.float64(1.0e6),
    )
    term = fortran_divide(
        _product(jnp.float64(1.6), net_photosynthesis),
        co2_mole_fraction,
    )
    intercept_moles = _product(inputs.medlyn_intercept, jnp.float64(1.0e-6))
    boundary_moles = _product(inputs.boundary_layer_conductance, jnp.float64(1.0e-6))
    slope_term = _product(inputs.medlyn_slope, term)
    coefficient_b = -_sum(
        _product(
            jnp.float64(2.0),
            _sum(intercept_moles, term),
        ),
        strict_divide(
            _product(slope_term, slope_term),
            _product(boundary_moles, inputs.canopy_vapor_pressure_deficit),
        ),
    )
    coefficient_c = _sum(
        _product(
            inputs.medlyn_intercept,
            inputs.medlyn_intercept,
            jnp.float64(1.0e-12),
        ),
        _product(
            _sum(
                _product(jnp.float64(2.0), intercept_moles),
                _product(
                    term,
                    _sum(
                        jnp.float64(1.0),
                        -fortran_divide(
                            _product(inputs.medlyn_slope, inputs.medlyn_slope),
                            inputs.canopy_vapor_pressure_deficit,
                        ),
                    ),
                ),
            ),
            term,
        ),
    )
    roots = quadratic_roots(jnp.float64(1.0), coefficient_b, coefficient_c)
    return _product(jnp.maximum(roots.first, roots.second), jnp.float64(1.0e6))


def _medlyn_conductance_factored(net, surface_co2, pressure, intercept, boundary, slope, vpd):
    term = 1.6 * net / (surface_co2 / pressure * 1.e6)
    intercept_moles = intercept * 1.e-6
    boundary_moles = boundary * 1.e-6
    square_root = jnp.abs(term) * jnp.abs(slope) * jnp.sqrt(
        4. / vpd * (1. + (intercept_moles + term) / boundary_moles)
        + (slope * term / (boundary_moles * vpd)) ** 2
    )
    return 1.e6 * (
        intercept_moles + term
        + .5 * (slope * term) ** 2 / (boundary_moles * vpd)
        + .5 * square_root
    )


@jax.custom_jvp
def _medlyn_conductance(net_photosynthesis, leaf_surface_co2, inputs):
    return _medlyn_conductance_reference(net_photosynthesis, leaf_surface_co2, inputs)


@_medlyn_conductance.defjvp
def _medlyn_conductance_jvp(primals, tangents):
    net, co2, inputs = primals
    dnet, dco2, dinputs = tangents
    fields = ("atmospheric_pressure", "medlyn_intercept", "boundary_layer_conductance",
              "medlyn_slope", "canopy_vapor_pressure_deficit")
    point = (net, co2, *(getattr(inputs, name) for name in fields))
    direction = (dnet, dco2, *(getattr(dinputs, name) for name in fields))
    tangent = jax.jvp(_medlyn_conductance_factored, point, direction)[1]
    return _medlyn_conductance(*primals), tangent


def hydraulic_photosynthesis_medlyn_residual(
    inputs: HydraulicPhotosynthesisInputs,
) -> HydraulicPhotosynthesisResult:

    _validate_inputs(inputs)
    rubisco, electron, product = _limitation_rates(inputs)
    rubisco = lax.optimization_barrier(rubisco)
    electron = lax.optimization_barrier(electron)
    product = lax.optimization_barrier(product)
    gross = _gross_photosynthesis(
        rubisco,
        electron,
        product,
        inputs.ac_aj_colimitation_curvature,
    )
    gross = lax.optimization_barrier(gross)
    net = _sum(
        gross,
        -_product(inputs.hydraulic_stress, inputs.leaf_maintenance_respiration),
    )
    minimum_conductance = jnp.maximum(
        _product(inputs.hydraulic_stress, inputs.medlyn_intercept),
        jnp.float64(1.0),
    )
    initial_residual = jnp.zeros((NUMBER_OF_LEAF_CLASSES,), dtype=jnp.float64)

    def surface_co2() -> jnp.ndarray:
        return jnp.maximum(
            _sum(
                inputs.atmospheric_co2,
                -_product(
                    fortran_divide(jnp.float64(1.4), inputs.boundary_layer_conductance),
                    net,
                    inputs.atmospheric_pressure,
                ),
            ),
            jnp.float64(MINIMUM_LEAF_SURFACE_CO2),
        )

    def both_negative(_: None):
        return surface_co2(), minimum_conductance, initial_residual

    def at_least_one_nonnegative(_: None):
        surface = surface_co2()
        conductance = _medlyn_conductance(net, surface, inputs)
        resistance_numerator = _product(
            net,
            inputs.atmospheric_pressure,
            _sum(
                _product(jnp.float64(1.4), conductance),
                _product(jnp.float64(1.6), inputs.boundary_layer_conductance),
            ),
        )
        resistance_denominator = _product(inputs.boundary_layer_conductance, conductance)
        solved_residual = _sum(
            inputs.intracellular_co2,
            -inputs.atmospheric_co2,
            fortran_divide(resistance_numerator, resistance_denominator),
        )
        residual = jnp.where(net >= jnp.float64(0.0), solved_residual, initial_residual)
        return surface, conductance, residual

    surface, conductance, residual = lax.cond(
        jnp.all(net < jnp.float64(0.0)),
        both_negative,
        at_least_one_nonnegative,
        operand=None,
    )
    return HydraulicPhotosynthesisResult(
        rubisco_limited_rate=rubisco,
        electron_transport_limited_rate=electron,
        product_limited_rate=product,
        gross_photosynthesis=gross,
        net_photosynthesis=net,
        leaf_surface_co2=surface,
        stomatal_conductance=conductance,
        ci_residual=residual,
    )


def _evaluate_at_ci(
    template: HydraulicPhotosynthesisInputs,
    intracellular_co2: jnp.ndarray,
    hydraulic_stress: jnp.ndarray,
) -> HydraulicPhotosynthesisResult:
    return hydraulic_photosynthesis_medlyn_residual(
        template._replace(
            intracellular_co2=intracellular_co2,
            hydraulic_stress=hydraulic_stress,
        )
    )


def _brent_phase_step(values):
    a, b, c, d, e, fa, fb, fc, tolerance, midpoint = values
    interpolate_used = ((jnp.abs(e) >= tolerance) & (jnp.abs(fa) > jnp.abs(fb))
                        & ((a == c) | (fc != 0.)))

    def interpolate(_: None):
        ratio_s = fortran_divide(fb, jnp.where(interpolate_used, fa, 1.))

        def linear(_: None):
            numerator = _product(jnp.float64(2.0), midpoint, ratio_s)
            denominator = _sum(jnp.float64(1.0), -ratio_s)
            return numerator, denominator

        def inverse_quadratic(_: None):
            denominator = jnp.where(interpolate_used & (a != c), fc, 1.)
            ratio_q = fortran_divide(fa, denominator)
            ratio_r = fortran_divide(fb, denominator)
            numerator = _product(
                ratio_s,
                _sum(
                    _product(
                        jnp.float64(2.0),
                        midpoint,
                        ratio_q,
                        _sum(ratio_q, -ratio_r),
                    ),
                    -_product(
                        _sum(b, -a),
                        _sum(ratio_r, -jnp.float64(1.0)),
                    ),
                ),
            )
            denominator = _product(
                _sum(ratio_q, -jnp.float64(1.0)),
                _sum(ratio_r, -jnp.float64(1.0)),
                _sum(ratio_s, -jnp.float64(1.0)),
            )
            return numerator, denominator

        numerator, denominator = lax.cond(a == c, linear, inverse_quadratic, operand=None)
        denominator = jnp.where(numerator > jnp.float64(0.0), -denominator, denominator)
        numerator = jnp.abs(numerator)
        interpolation_limit = jnp.minimum(
            _sum(
                _product(jnp.float64(3.0), midpoint, denominator),
                -jnp.abs(_product(tolerance, denominator)),
            ),
            jnp.abs(_product(e, denominator)),
        )

        accepted = _product(jnp.float64(2.0), numerator) < interpolation_limit

        def accept(_: None):
            return fortran_divide(numerator, jnp.where(accepted, denominator, 1.)), d

        return lax.cond(
            accepted,
            accept,
            lambda _: (midpoint, midpoint),
            operand=None,
        )

    return lax.cond(
        interpolate_used,
        interpolate,
        lambda _: (midpoint, midpoint),
        operand=None,
    )


def _brent_coupled(
    template: HydraulicPhotosynthesisInputs,
    hydraulic_stress: jnp.ndarray,
    first_ci: jnp.ndarray,
    second_ci: jnp.ndarray,
    first_residual: jnp.ndarray,
    second_result: HydraulicPhotosynthesisResult,
    absolute_tolerance: jnp.ndarray,
    *,
    enabled: jnp.ndarray = True,
) -> tuple[jnp.ndarray, HydraulicPhotosynthesisResult]:
    state = _BrentState(
        a=first_ci,
        b=second_ci,
        c=second_ci,
        d=jnp.zeros((NUMBER_OF_LEAF_CLASSES,), dtype=jnp.float64),
        e=jnp.zeros((NUMBER_OF_LEAF_CLASSES,), dtype=jnp.float64),
        fa=first_residual,
        fb=second_result.ci_residual,
        fc=second_result.ci_residual,
        result=second_result,
        active=jnp.asarray(enabled, dtype=jnp.bool_),
    )

    def iteration_step(current: _BrentState, _: jnp.ndarray):
        def active_step(value: _BrentState):
            same_sign = ((value.fb > jnp.float64(0.0)) & (value.fc > jnp.float64(0.0))) | (
                (value.fb < jnp.float64(0.0)) & (value.fc < jnp.float64(0.0))
            )
            c = jnp.where(same_sign, value.a, value.c)
            fc = jnp.where(same_sign, value.fa, value.fc)
            interval = _sum(value.b, -value.a)
            d = jnp.where(same_sign, interval, value.d)
            e = jnp.where(same_sign, interval, value.e)

            rotate = jnp.abs(fc) < jnp.abs(value.fb)
            a = jnp.where(rotate, value.b, value.a)
            b = jnp.where(rotate, c, value.b)
            c = jnp.where(rotate, value.b, c)
            fa = jnp.where(rotate, value.fb, value.fa)
            fb = jnp.where(rotate, fc, value.fb)
            fc = jnp.where(rotate, value.fb, fc)
            tolerance = _sum(
                _product(
                    jnp.float64(2.0),
                    jnp.float64(BRENT_RELATIVE_TOLERANCE),
                    jnp.abs(b),
                ),
                _product(jnp.float64(0.5), absolute_tolerance),
            )
            midpoint = _product(jnp.float64(0.5), _sum(c, -b))
            converged = jnp.all((jnp.abs(midpoint) <= tolerance) | (fb == jnp.float64(0.0)))

            def finish(_: None):
                return _BrentState(a, b, c, d, e, fa, fb, fc, value.result, jnp.bool_(False))

            def advance(_: None):
                d_new, e_new = scalar_map(
                    _brent_phase_step,
                    (a, b, c, d, e, fa, fb, fc, tolerance, midpoint),
                )
                a_new = b
                fa_new = fb
                step = jnp.where(
                    jnp.abs(d_new) > tolerance,
                    d_new,
                    jnp.copysign(tolerance, midpoint),
                )
                b_new = _sum(b, step)
                result = _evaluate_at_ci(template, b_new, hydraulic_stress)
                active = ~jnp.all(result.ci_residual == jnp.float64(0.0))
                return _BrentState(
                    a_new,
                    b_new,
                    c,
                    d_new,
                    e_new,
                    fa_new,
                    result.ci_residual,
                    fc,
                    result,
                    active,
                )

            return lax.cond(converged, finish, advance, operand=None)

        updated = lax.cond(current.active, active_step, lambda value: value, current)
        return updated, None

    state = converged_iteration(
        iteration_step,
        state,
        range(MAXIMUM_BRENT_ITERATIONS),
        lambda state: state.active,
    )
    return state.b, state.result


def _solve_ci_secant_brent(
    template: HydraulicPhotosynthesisInputs,
    hydraulic_stress: jnp.ndarray,
    first_ci: jnp.ndarray,
    second_ci: jnp.ndarray,
    first_result: HydraulicPhotosynthesisResult,
    second_result: HydraulicPhotosynthesisResult,
    tolerance: jnp.ndarray,
    *,
    enabled: jnp.ndarray = True,
) -> tuple[jnp.ndarray, HydraulicPhotosynthesisResult, jnp.ndarray]:
    first_converged = jnp.all(
        jnp.abs(first_result.ci_residual) < jnp.float64(HYBRID_ABSOLUTE_RESIDUAL_TOLERANCE)
    )
    second_converged = jnp.all(
        jnp.abs(second_result.ci_residual) < jnp.float64(HYBRID_ABSOLUTE_RESIDUAL_TOLERANCE)
    )
    second_ci = jnp.where(first_converged, first_ci, second_ci)
    initially_active = enabled & ~(first_converged | second_converged)
    state = _SecantState(
        x0=first_ci,
        x1=second_ci,
        f0=first_result.ci_residual,
        f1=second_result.ci_residual,
        minimum_residual=jnp.float64(0.0),
        minimum_ci=second_ci,
        result=second_result,
        active=initially_active,
        iterations=jnp.int32(0),
    )

    def iteration_step(current: _SecantState, iteration: jnp.ndarray):
        def active_step(value: _SecantState):
            denominator = _sum(value.f1, -value.f0)
            midpoint_step = _sum(
                _product(jnp.float64(0.5), _sum(value.x1, value.x0)),
                -value.x1,
            )
            secant_step = fortran_divide(
                -_product(value.f1, _sum(value.x1, -value.x0)),
                jnp.where(denominator != 0., denominator, jnp.float64(1.0)),
            )
            delta = jnp.where(denominator == jnp.float64(0.0), midpoint_step, secant_step)
            x0 = value.x1
            x1 = _sum(value.x1, delta)
            result = _evaluate_at_ci(template, x1, hydraulic_stress)
            f1 = result.ci_residual
            delta_converged = jnp.all(jnp.abs(delta) < tolerance)
            candidate_residual = jnp.abs(_sum(f1[SUNLIT], f1[SHADED]))
            first_minimum = iteration == jnp.int32(1)
            better = first_minimum | (candidate_residual < value.minimum_residual)
            minimum_residual = jnp.where(better, candidate_residual, value.minimum_residual)
            minimum_ci = jnp.where(better, x1, value.minimum_ci)
            residual_converged = jnp.all(
                jnp.abs(f1) < jnp.float64(HYBRID_ABSOLUTE_RESIDUAL_TOLERANCE)
            )
            bracketed = jnp.all(_product(f1, value.f0) < jnp.float64(0.0))

            def finish_delta(_: None):
                return _SecantState(
                    x1,
                    x1,
                    value.f0,
                    f1,
                    minimum_residual,
                    minimum_ci,
                    result,
                    jnp.bool_(False),
                    iteration,
                )

            def after_delta(_: None):
                def finish_residual(_: None):
                    return _SecantState(
                        x0,
                        x1,
                        value.f0,
                        f1,
                        minimum_residual,
                        minimum_ci,
                        result,
                        jnp.bool_(False),
                        iteration,
                    )

                def after_residual(_: None):
                    def finish_brent(_: None):
                        roots, brent_result = _brent_coupled(
                            template,
                            hydraulic_stress,
                            x0,
                            x1,
                            value.f0,
                            result,
                            tolerance[SUNLIT],
                            enabled=(
                                value.active & ~delta_converged
                                & ~residual_converged & bracketed
                            ),
                        )
                        return _SecantState(
                            roots,
                            x1,
                            value.f0,
                            brent_result.ci_residual,
                            minimum_residual,
                            minimum_ci,
                            brent_result,
                            jnp.bool_(False),
                            iteration,
                        )

                    def after_brent(_: None):
                        def finish_maximum(_: None):
                            minimum_result = _evaluate_at_ci(template, minimum_ci, hydraulic_stress)
                            return _SecantState(
                                x0,
                                minimum_ci,
                                value.f0,
                                minimum_result.ci_residual,
                                minimum_residual,
                                minimum_ci,
                                minimum_result,
                                jnp.bool_(False),
                                iteration,
                            )

                        return lax.cond(
                            iteration > jnp.int32(3),
                            finish_maximum,
                            lambda _: _SecantState(
                                x0,
                                x1,
                                value.f0,
                                f1,
                                minimum_residual,
                                minimum_ci,
                                result,
                                jnp.bool_(True),
                                iteration,
                            ),
                            operand=None,
                        )

                    return lax.cond(bracketed, finish_brent, after_brent, operand=None)

                return lax.cond(
                    residual_converged,
                    finish_residual,
                    after_residual,
                    operand=None,
                )

            return lax.cond(delta_converged, finish_delta, after_delta, operand=None)

        updated = lax.cond(current.active, active_step, lambda value: value, current)
        return updated, None

    state = converged_iteration(
        iteration_step,
        state,
        range(1, MAXIMUM_SECANT_ITERATIONS + 1),
        lambda state: state.active,
    )
    return state.x1, state.result, state.iterations


def _updated_unstressed_conductance(conductance, stress, previous):
    refresh = stress > jnp.float64(0.01)
    divisor = jnp.where(refresh, stress, jnp.float64(1.0))
    return jnp.where(refresh, fortran_divide(conductance, divisor), previous)


def solve_hydraulic_photosynthesis_medlyn(
    photosynthesis_inputs: HydraulicPhotosynthesisInputs,
    plant_inputs: PlantHydraulicPatchInputs,
    plant_parameters: PlantHydraulicPatchParameters,
    plant_state: PlantHydraulicPatchState,
    *,
    enabled: jnp.ndarray = True,
) -> CoupledHydraulicPhotosynthesisResult:

    _validate_inputs(photosynthesis_inputs)
    initial_stress = jnp.ones((NUMBER_OF_LEAF_CLASSES,), dtype=jnp.float64)
    initial_result = _evaluate_at_ci(
        photosynthesis_inputs,
        photosynthesis_inputs.intracellular_co2,
        initial_stress,
    )
    state = _OuterState(
        ci=photosynthesis_inputs.intracellular_co2,
        hydraulic_stress=initial_stress,
        previous_hydraulic_stress=-initial_stress,
        unstressed_conductance=jnp.zeros((NUMBER_OF_LEAF_CLASSES,), dtype=jnp.float64),
        result=initial_result,
        transpiration=plant_state.vegetation_transpiration,
        active=jnp.asarray(enabled, dtype=jnp.bool_),
        outer_iterations=jnp.int32(0),
        inner_iterations=jnp.int32(0),
        hydraulic_iterations=jnp.int32(0),
    )
    def outer_step(current: _OuterState, iteration: jnp.ndarray):
        def active_step(value: _OuterState):
            first_ci = jnp.maximum(jnp.float64(0.1), value.ci)
            second_ci = _product(jnp.float64(0.99), value.ci)
            tolerance = _product(
                jnp.abs(second_ci),
                jnp.float64(HYBRID_RELATIVE_CI_TOLERANCE),
            )

            def update_stress(_: None):
                stress_result = plant_hydraulic_stress(
                    plant_inputs,
                    plant_parameters,
                    PlantHydraulicPatchState(
                        plant_state.water_potential_segment,
                        value.transpiration,
                    ),
                    photosynthesis_inputs.boundary_layer_conductance,
                    value.unstressed_conductance[SUNLIT],
                    value.unstressed_conductance[SHADED],
                    enabled=value.active & (iteration > jnp.int32(1)),
                )
                return (
                    jnp.stack(
                        (
                            stress_result.sunlit_stress,
                            stress_result.shaded_stress,
                        )
                    ),
                    stress_result.state.vegetation_transpiration,
                    stress_result.iterations,
                )

            stress, transpiration, hydraulic_iterations = lax.cond(
                iteration > jnp.int32(1),
                update_stress,
                lambda _: (
                    value.hydraulic_stress,
                    value.transpiration,
                    value.hydraulic_iterations,
                ),
                operand=None,
            )
            first_result = _evaluate_at_ci(photosynthesis_inputs, first_ci, stress)
            stress_delta = _sum(value.previous_hydraulic_stress, -stress)
            second_result = _evaluate_at_ci(photosynthesis_inputs, second_ci, stress)
            ci_arguments = (photosynthesis_inputs, stress, first_ci, second_ci,
                            first_result, second_result, tolerance)
            if default_backend() == "gpu":
                from clm_jax.physics.photosynthesis_gpu import solve_ci_gpu
                solved_ci, result, inner_iterations = solve_ci_gpu(
                    *ci_arguments, value.active,
                )
            else:
                solved_ci, result, inner_iterations = _solve_ci_secant_brent(
                    *ci_arguments, enabled=value.active,
                )
            unstressed = _updated_unstressed_conductance(
                result.stomatal_conductance, stress, value.unstressed_conductance,
            )
            stress_converged = jnp.all(
                jnp.abs(stress_delta) < jnp.float64(HYDRAULIC_STRESS_TOLERANCE)
            )
            active = ~(stress_converged | (iteration > jnp.int32(3)))
            return _OuterState(
                ci=solved_ci,
                hydraulic_stress=stress,
                previous_hydraulic_stress=stress,
                unstressed_conductance=unstressed,
                result=result,
                transpiration=transpiration,
                active=active,
                outer_iterations=iteration,
                inner_iterations=inner_iterations,
                hydraulic_iterations=hydraulic_iterations,
            )

        updated = lax.cond(current.active, active_step, lambda value: value, current)
        return updated, None

    state = converged_iteration(
        outer_step,
        state,
        range(1, MAXIMUM_HYBRID_OUTER_ITERATIONS + 1),
        lambda state: state.active,
    )
    final_water = vegetation_water_potential(
        plant_inputs,
        plant_parameters,
        photosynthesis_inputs.boundary_layer_conductance,
        state.result.stomatal_conductance[SUNLIT],
        state.result.stomatal_conductance[SHADED],
    )
    final_plant_state = PlantHydraulicPatchState(
        final_water.water_potential_segment,
        jnp.maximum(final_water.soil_flux, jnp.float64(0.0)),
    )
    return CoupledHydraulicPhotosynthesisResult(
        intracellular_co2=state.ci,
        hydraulic_stress=state.hydraulic_stress,
        stomatal_conductance=state.result.stomatal_conductance,
        photosynthesis=state.result,
        plant_state=final_plant_state,
        outer_iterations=state.outer_iterations,
        inner_iterations=state.inner_iterations,
        hydraulic_iterations=state.hydraulic_iterations,
    )


def solve_hydraulic_photosynthesis_patch(
    biochemical_parameters: HydraulicPhotosynthesisBiochemicalParameters,
    environment: HydraulicPhotosynthesisEnvironment,
    controls: HydraulicPhotosynthesisPatchControls,
    plant_inputs: PlantHydraulicPatchInputs,
    plant_parameters: PlantHydraulicPatchParameters,
    plant_state: PlantHydraulicPatchState,
) -> HydraulicPhotosynthesisPatchResult:

    control_fields = tuple(jnp.asarray(value) for value in controls)
    if any(value.shape != () for value in control_fields):
        raise ValueError("hydraulic photosynthesis patch controls must be scalar")
    preparation = prepare_hydraulic_photosynthesis_inputs(
        biochemical_parameters,
        environment,
    )
    prepared = preparation.inputs

    def day_branch(_: None):
        if _USE_FUSED_HYDRAULIC_PHOTOSYNTHESIS and default_backend() == "gpu":
            from clm_jax.physics.hydraulic_photosynthesis_gpu import solve_hybrid_gpu
            solved = solve_hybrid_gpu(
                prepared, plant_inputs, plant_parameters, plant_state,
                preparation.daylight,
            )
        else:
            solved = solve_hydraulic_photosynthesis_medlyn(
                prepared,
                plant_inputs,
                plant_parameters,
                plant_state,
                enabled=preparation.daylight,
            )
        minimum_conductance = jnp.maximum(
            _product(solved.hydraulic_stress, prepared.medlyn_intercept),
            jnp.float64(1.0),
        )
        conductance = jnp.where(
            solved.photosynthesis.net_photosynthesis < jnp.float64(0.0),
            minimum_conductance,
            solved.stomatal_conductance,
        )
        surface_co2 = jnp.maximum(
            _sum(
                prepared.atmospheric_co2,
                -_product(
                    fortran_divide(
                        jnp.float64(1.4),
                        prepared.boundary_layer_conductance,
                    ),
                    solved.photosynthesis.net_photosynthesis,
                    prepared.atmospheric_pressure,
                ),
            ),
            jnp.float64(1.0e-6),
        )
        final_ci = jnp.maximum(
            _sum(
                prepared.atmospheric_co2,
                -fortran_divide(
                    _product(
                        solved.photosynthesis.net_photosynthesis,
                        prepared.atmospheric_pressure,
                        _sum(
                            _product(jnp.float64(1.4), conductance),
                            _product(
                                jnp.float64(1.6),
                                prepared.boundary_layer_conductance,
                            ),
                        ),
                    ),
                    _product(prepared.boundary_layer_conductance, conductance),
                ),
            ),
            jnp.float64(1.0e-6),
        )
        photosynthesis = solved.photosynthesis._replace(
            leaf_surface_co2=surface_co2,
            stomatal_conductance=conductance,
        )
        corrected = solved._replace(
            intracellular_co2=final_ci,
            stomatal_conductance=conductance,
            photosynthesis=photosynthesis,
        )
        ozone = jnp.stack(
            (
                controls.sunlit_stomatal_ozone_coefficient,
                controls.shaded_stomatal_ozone_coefficient,
            )
        )
        return (
            corrected,
            solved.intracellular_co2,
            _daylight_stomatal_resistance(
                conductance, preparation.molar_conversion_factor, ozone
            ),
            solved.hydraulic_iterations,
        )

    def night_branch(_: None):
        night_water_potential = plant_state.water_potential_segment.at[SUNLIT].set(jnp.float64(1.0))
        stress_result = plant_hydraulic_stress(
            plant_inputs,
            plant_parameters,
            PlantHydraulicPatchState(
                night_water_potential,
                plant_state.vegetation_transpiration,
            ),
            prepared.boundary_layer_conductance,
            prepared.medlyn_intercept,
            prepared.medlyn_intercept,
            enabled=~preparation.daylight,
        )
        stress = jnp.stack((stress_result.sunlit_stress, stress_result.shaded_stress))
        conductance = jnp.maximum(
            _product(stress, prepared.medlyn_intercept),
            jnp.float64(1.0),
        )
        stress_scales_respiration = (~jnp.asarray(controls.crop, dtype=jnp.bool_)) | (
            ~jnp.asarray(
                controls.modify_photosynthesis_and_respiration_for_crop,
                dtype=jnp.bool_,
            )
        )
        respiration_scale = jnp.where(
            stress_scales_respiration,
            stress,
            jnp.ones((NUMBER_OF_LEAF_CLASSES,), dtype=jnp.float64),
        )
        zeros = jnp.zeros((NUMBER_OF_LEAF_CLASSES,), dtype=jnp.float64)
        photosynthesis = HydraulicPhotosynthesisResult(
            rubisco_limited_rate=zeros,
            electron_transport_limited_rate=zeros,
            product_limited_rate=zeros,
            gross_photosynthesis=zeros,
            net_photosynthesis=-_product(
                respiration_scale,
                prepared.leaf_maintenance_respiration,
            ),
            leaf_surface_co2=zeros,
            stomatal_conductance=conductance,
            ci_residual=zeros,
        )
        coupled = CoupledHydraulicPhotosynthesisResult(
            intracellular_co2=zeros,
            hydraulic_stress=stress,
            stomatal_conductance=conductance,
            photosynthesis=photosynthesis,
            plant_state=stress_result.state,
            outer_iterations=jnp.int32(0),
            inner_iterations=jnp.int32(0),
            hydraulic_iterations=stress_result.iterations,
        )
        resistance = jnp.minimum(
            _product(
                fortran_divide(jnp.float64(1.0), conductance),
                preparation.molar_conversion_factor,
            ),
            jnp.float64(20_000.0),
        )
        return coupled, zeros, resistance, stress_result.iterations

    coupled, solver_intracellular_co2, resistance, hydraulic_iterations = lax.cond(
        preparation.daylight,
        day_branch,
        night_branch,
        operand=None,
    )
    total_leaf_area = _sum(
        plant_inputs.sunlit_leaf_area,
        plant_inputs.shaded_leaf_area,
    )
    safe_leaf_area = jnp.where(total_leaf_area > 0., total_leaf_area, jnp.float64(1.0))
    transpiration_wetness = lax.cond(
        total_leaf_area > jnp.float64(0.0),
        lambda _: _sum(
            _product(
                coupled.hydraulic_stress[SUNLIT],
                fortran_divide(plant_inputs.sunlit_leaf_area, safe_leaf_area),
            ),
            _product(
                coupled.hydraulic_stress[SHADED],
                fortran_divide(plant_inputs.shaded_leaf_area, safe_leaf_area),
            ),
        ),
        lambda _: coupled.hydraulic_stress[SUNLIT],
        operand=None,
    )
    return HydraulicPhotosynthesisPatchResult(
        preparation=preparation,
        coupled=coupled,
        solver_intracellular_co2=solver_intracellular_co2,
        stomatal_resistance=resistance,
        transpiration_wetness=transpiration_wetness,
        hydraulic_iterations=hydraulic_iterations,
    )
