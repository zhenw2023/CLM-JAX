
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import default_backend, lax

from clm_jax.constants import PI, UNIVERSAL_GAS_CONSTANT
from clm_jax.physics.converged_iteration import converged_iteration
from clm_jax.physics.native_power import native_power, native_power_of_two
from clm_jax.physics.numerics import (
    fortran_divide,
    fortran_masked_materialized_divide,
    fortran_materialize,
    fortran_ordered_sum,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_product,
)
from clm_jax.physics.solver_scalarization import _native_tile_trace

SUNLIT_LEAF = 0
SHADED_LEAF = 1
XYLEM = 2
ROOT = 3
NUMBER_OF_VEGETATION_SEGMENTS = 4
MINIMUM_TRANSPIRING_LEAF_AREA = 0.001
MAXIMUM_NEWTON_ITERATIONS = 50
MAXIMUM_WATER_POTENTIAL_STEP = 50_000.0
FLUX_BALANCE_TOLERANCE = 1.0e-6
WATER_POTENTIAL_STEP_TOLERANCE = 1.0e-9
SINGULAR_JACOBIAN_TOLERANCE = 1.0e-50
_USE_FUSED_NATIVE_NEWTON = True


class PlantHydraulicParameters(NamedTuple):

    maximum_root_conductance_pft: jnp.ndarray
    maximum_conductance_pft_segment: jnp.ndarray
    water_potential_at_half_conductance_pft_segment: jnp.ndarray
    vulnerability_shape_pft_segment: jnp.ndarray
    reference_soil_water_potential_pft: jnp.ndarray
    soil_water_potential_full_opening_pft: jnp.ndarray
    soil_water_potential_full_closure_pft: jnp.ndarray
    medlyn_slope_pft: jnp.ndarray
    medlyn_intercept_pft: jnp.ndarray
    fine_root_to_leaf_ratio_pft: jnp.ndarray
    root_profile_beta_pft_variant: jnp.ndarray
    root_radius_pft: jnp.ndarray
    root_density_pft: jnp.ndarray


class PlantHydraulicPatchParameters(NamedTuple):

    maximum_conductance_segment: jnp.ndarray
    water_potential_at_half_conductance_segment: jnp.ndarray
    vulnerability_shape_segment: jnp.ndarray


class PlantHydraulicPatchInputs(NamedTuple):

    sunlit_leaf_area: jnp.ndarray
    shaded_leaf_area: jnp.ndarray
    exposed_leaf_area: jnp.ndarray
    exposed_stem_area: jnp.ndarray
    total_stem_area: jnp.ndarray
    canopy_top_height: jnp.ndarray
    dry_foliage_fraction: jnp.ndarray
    air_density: jnp.ndarray
    atmospheric_pressure: jnp.ndarray
    reference_temperature: jnp.ndarray
    leaf_saturation_specific_humidity: jnp.ndarray
    canopy_air_specific_humidity: jnp.ndarray
    soil_root_conductance_level: jnp.ndarray
    soil_water_potential_level: jnp.ndarray
    soil_node_depth_level: jnp.ndarray


class PlantHydraulicPatchState(NamedTuple):
    water_potential_segment: jnp.ndarray
    vegetation_transpiration: jnp.ndarray


class SoilRootConductanceInputs(NamedTuple):

    fine_root_carbon: jnp.ndarray
    root_fraction_level: jnp.ndarray
    soil_layer_thickness_level: jnp.ndarray
    soil_node_depth_level: jnp.ndarray
    saturated_hydraulic_conductivity_level: jnp.ndarray
    hydraulic_conductivity_level: jnp.ndarray
    soil_water_potential_level: jnp.ndarray
    total_leaf_area: jnp.ndarray
    total_stem_area: jnp.ndarray


class SoilRootConductanceParameters(NamedTuple):

    fine_root_to_leaf_ratio: jnp.ndarray
    maximum_root_conductance: jnp.ndarray
    root_half_conductance_potential: jnp.ndarray
    root_vulnerability_shape: jnp.ndarray
    root_radius: jnp.ndarray
    root_density: jnp.ndarray


class SoilRootConductanceResult(NamedTuple):

    soil_root_conductance_level: jnp.ndarray
    root_conductance_level: jnp.ndarray
    soil_conductance_level: jnp.ndarray
    root_area_index_level: jnp.ndarray


class TranspirationFluxes(NamedTuple):
    sunlit: jnp.ndarray
    shaded: jnp.ndarray


class VegetationWaterPotentialResult(NamedTuple):
    water_potential_segment: jnp.ndarray
    soil_flux: jnp.ndarray
    transpiration: TranspirationFluxes


class SpacLinearization(NamedTuple):
    negative_jacobian: jnp.ndarray
    inverse_negative_jacobian: jnp.ndarray
    singular: jnp.ndarray


class PlantHydraulicStressResult(NamedTuple):
    state: PlantHydraulicPatchState
    sunlit_stress: jnp.ndarray
    shaded_stress: jnp.ndarray
    iterations: jnp.ndarray
    used_algebraic_fallback: jnp.ndarray


def _product(*factors: jnp.ndarray) -> jnp.ndarray:
    result = jnp.asarray(factors[0], dtype=jnp.float64)
    for factor in factors[1:]:
        result = lax.optimization_barrier(fortran_product(result, factor))
    return result


def _sum(*terms: jnp.ndarray) -> jnp.ndarray:
    broadcast = jnp.broadcast_arrays(
        *(jnp.asarray(term, dtype=jnp.float64) for term in terms)
    )
    result = broadcast[0]
    for term in broadcast[1:]:
        result = lax.optimization_barrier(fortran_add(result, term))
    return result


def _assigned_product(*factors: jnp.ndarray) -> jnp.ndarray:
    result = jnp.asarray(factors[0], dtype=jnp.float64)
    for factor in factors[1:]:
        result = fortran_multiply(result, factor)
    return result


def _assigned_sum(*terms: jnp.ndarray) -> jnp.ndarray:
    broadcast = jnp.broadcast_arrays(
        *(jnp.asarray(term, dtype=jnp.float64) for term in terms)
    )
    result = broadcast[0]
    for term in broadcast[1:]:
        result = fortran_add(result, term)
    return result


def _matrix_vector_product(
    matrix: jnp.ndarray,
    vector: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.stack(
        tuple(
            _sum(
                *(
                    _product(matrix[row, column], vector[column])
                    for column in range(matrix.shape[1])
                )
            )
            for row in range(matrix.shape[0])
        )
    )


def select_patch_parameters(
    parameters: PlantHydraulicParameters,
    vegetation_type: jnp.ndarray,
) -> PlantHydraulicPatchParameters:

    return PlantHydraulicPatchParameters(
        maximum_conductance_segment=parameters.maximum_conductance_pft_segment[
            vegetation_type
        ],
        water_potential_at_half_conductance_segment=(
            parameters.water_potential_at_half_conductance_pft_segment[vegetation_type]
        ),
        vulnerability_shape_segment=parameters.vulnerability_shape_pft_segment[
            vegetation_type
        ],
    )


def jackson_root_fraction(
    root_profile_beta: jnp.ndarray,
    soil_interface_depth: jnp.ndarray,
    bedrock_level_count: jnp.ndarray,
) -> jnp.ndarray:

    beta = jnp.asarray(root_profile_beta, dtype=jnp.float64)
    interface = jnp.asarray(soil_interface_depth, dtype=jnp.float64)
    bedrock = jnp.asarray(bedrock_level_count, dtype=jnp.int32)
    if interface.ndim < 1 or interface.shape[-1] < 2:
        raise ValueError("soil interfaces must contain a nonempty boundary axis")
    if interface.ndim > 1 and interface.shape[:-1] != beta.shape:
        raise ValueError("per-patch soil interfaces must match root beta dimensions")
    if bedrock.shape != beta.shape:
        raise ValueError("root beta and bedrock level count must have matching patch shapes")

    number_of_levels = interface.shape[-1] - 1
    if number_of_levels < 1:
        raise ValueError("Jackson root profile requires at least one soil layer")

    exponent = fortran_multiply(interface, jnp.float64(1.0e2))
    upper_cumulative = native_power(beta[..., None], exponent[..., :-1])
    lower_cumulative = native_power(beta[..., None], exponent[..., 1:])
    root_fraction = fortran_add(upper_cumulative, -lower_cumulative)

    level = jnp.arange(number_of_levels, dtype=jnp.int32)
    below_bedrock = jnp.where(
        level >= bedrock[..., None],
        root_fraction,
        jnp.float64(0.0),
    )
    below_sum = fortran_ordered_sum(
        jnp.moveaxis(below_bedrock, -1, 0),
        initial=jnp.zeros_like(beta),
    )
    safe_bedrock = jnp.clip(bedrock, jnp.int32(1), jnp.int32(number_of_levels))
    redistribution = fortran_divide(
        below_sum,
        safe_bedrock.astype(jnp.float64),
    )
    shifted = fortran_add(root_fraction, redistribution[..., None])
    return jnp.where(level < safe_bedrock[..., None], shifted, jnp.float64(0.0))


def _weibull_power_pair(water_potential, half_conductance_potential, shape,
                       power_base=2.0):
    ratio_power = native_power(
        fortran_divide(water_potential, half_conductance_potential), shape
    )
    if isinstance(power_base, (int, float)) and power_base == 2.:
        fraction = native_power_of_two(-ratio_power)
    else:
        fraction = native_power(jnp.asarray(power_base, dtype=jnp.float64), -ratio_power)
    return ratio_power, fraction


def _weibull_derivative(water_potential, shape, power_pair):
    ratio_power, unconstrained_fraction = power_pair
    return fortran_divide(
        _product(
            -shape, jnp.float64(0.6931471805599453),
            unconstrained_fraction, ratio_power,
        ),
        water_potential,
    )


def vulnerability_fraction(
    water_potential: jnp.ndarray,
    half_conductance_potential: jnp.ndarray,
    shape: jnp.ndarray,
    *,
    power_base: jnp.ndarray | float = 2.0,
) -> jnp.ndarray:

    water_potential = jnp.asarray(water_potential, dtype=jnp.float64)
    half_conductance_potential = jnp.asarray(
        half_conductance_potential, dtype=jnp.float64
    )
    shape = jnp.asarray(shape, dtype=jnp.float64)
    _, fraction = _weibull_power_pair(
        water_potential, half_conductance_potential, shape, power_base
    )
    return jnp.where(fraction < jnp.float64(0.005), jnp.float64(0.0), fraction)


def vulnerability_derivative(
    water_potential: jnp.ndarray,
    half_conductance_potential: jnp.ndarray,
    shape: jnp.ndarray,
) -> jnp.ndarray:

    water_potential = jnp.asarray(water_potential, dtype=jnp.float64)
    half_conductance_potential = jnp.asarray(
        half_conductance_potential, dtype=jnp.float64
    )
    shape = jnp.asarray(shape, dtype=jnp.float64)
    return _weibull_derivative(
        water_potential, shape,
        _weibull_power_pair(water_potential, half_conductance_potential, shape),
    )


def soil_root_interface_conductance(
    inputs: SoilRootConductanceInputs,
    parameters: SoilRootConductanceParameters,
    *,
    vulnerability_power_base: jnp.ndarray | float = 2.0,
) -> SoilRootConductanceResult:

    layer_fields = (
        inputs.root_fraction_level,
        inputs.soil_layer_thickness_level,
        inputs.soil_node_depth_level,
        inputs.saturated_hydraulic_conductivity_level,
        inputs.hydraulic_conductivity_level,
        inputs.soil_water_potential_level,
    )
    layer_shape = jnp.asarray(layer_fields[0]).shape
    if len(layer_shape) < 1 or layer_shape[-1] < 1:
        raise ValueError("soil-root conductance inputs require a nonempty level axis")
    if any(jnp.asarray(value).shape != layer_shape for value in layer_fields[1:]):
        raise ValueError("soil-root conductance layer fields must have matching shapes")
    patch_shape = layer_shape[:-1]
    for name in (
        "fine_root_carbon",
        "total_leaf_area",
        "total_stem_area",
    ):
        if jnp.asarray(getattr(inputs, name)).shape != patch_shape:
            raise ValueError(f"{name} must have shape {patch_shape}")
    for name, value in zip(
        SoilRootConductanceParameters._fields,
        parameters,
        strict=True,
    ):
        if jnp.asarray(value).shape != patch_shape:
            raise ValueError(f"{name} must have shape {patch_shape}")

    root_fraction = jnp.asarray(inputs.root_fraction_level, dtype=jnp.float64)
    layer_thickness = jnp.asarray(
        inputs.soil_layer_thickness_level, dtype=jnp.float64
    )
    node_depth = jnp.asarray(inputs.soil_node_depth_level, dtype=jnp.float64)
    saturated_conductivity = jnp.asarray(
        inputs.saturated_hydraulic_conductivity_level, dtype=jnp.float64
    )
    hydraulic_conductivity = jnp.asarray(
        inputs.hydraulic_conductivity_level, dtype=jnp.float64
    )
    soil_water_potential = jnp.asarray(
        inputs.soil_water_potential_level, dtype=jnp.float64
    )

    def patch_field(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(value, dtype=jnp.float64)[..., None]

    root_biomass_density = fortran_divide(
        _product(
            jnp.float64(2.0),
            patch_field(inputs.fine_root_carbon),
            root_fraction,
        ),
        layer_thickness,
    )
    root_biomass_density = jnp.maximum(
        _product(jnp.float64(2.0), jnp.float64(1.0)),
        root_biomass_density,
    )
    root_cross_sectional_area = _product(
        jnp.float64(PI),
        _product(
            patch_field(parameters.root_radius),
            patch_field(parameters.root_radius),
        ),
    )
    root_length_density = fortran_divide(
        root_biomass_density,
        _product(
            patch_field(parameters.root_density),
            root_cross_sectional_area,
        ),
    )
    root_area_index = _product(
        _sum(
            patch_field(inputs.total_stem_area),
            patch_field(inputs.total_leaf_area),
        ),
        patch_field(parameters.fine_root_to_leaf_ratio),
        root_fraction,
    )
    soil_distance = jnp.sqrt(
        fortran_divide(
            jnp.float64(1.0),
            _product(jnp.float64(PI), root_length_density),
        )
    )
    soil_conductance = fortran_divide(
        jnp.minimum(saturated_conductivity, hydraulic_conductivity),
        _product(jnp.float64(1.0e3), soil_distance),
    )
    root_vulnerability = native_power(
        jnp.asarray(vulnerability_power_base, dtype=jnp.float64),
        -native_power(
            fortran_divide(
                soil_water_potential,
                patch_field(parameters.root_half_conductance_potential),
            ),
            patch_field(parameters.root_vulnerability_shape),
        ),
    )
    root_vulnerability = jnp.where(
        root_vulnerability < jnp.float64(0.005),
        jnp.float64(0.0),
        root_vulnerability,
    )
    root_conductance = fortran_divide(
        _product(
            root_vulnerability,
            root_area_index,
            patch_field(parameters.maximum_root_conductance),
        ),
        _sum(jnp.float64(0.25), node_depth),
    )
    soil_conductance = jnp.maximum(soil_conductance, jnp.float64(1.0e-16))
    root_conductance = jnp.maximum(root_conductance, jnp.float64(1.0e-16))
    combined_resistance = _sum(
        fortran_divide(jnp.float64(1.0), soil_conductance),
        fortran_divide(jnp.float64(1.0), root_conductance),
    )
    conductance = fortran_divide(jnp.float64(1.0), combined_resistance)
    level_index = jnp.arange(layer_shape[-1], dtype=jnp.int32)
    active = (_product(root_area_index, root_fraction) > jnp.float64(0.0)) & (
        level_index > 0
    )
    conductance = jnp.where(active, conductance, jnp.float64(0.0))
    return SoilRootConductanceResult(
        soil_root_conductance_level=conductance,
        root_conductance_level=root_conductance,
        soil_conductance_level=soil_conductance,
        root_area_index_level=root_area_index,
    )


def _validate_patch_shapes(
    inputs: PlantHydraulicPatchInputs,
    parameters: PlantHydraulicPatchParameters,
) -> None:
    segment_fields = (
        parameters.maximum_conductance_segment,
        parameters.water_potential_at_half_conductance_segment,
        parameters.vulnerability_shape_segment,
    )
    if any(field.shape != (NUMBER_OF_VEGETATION_SEGMENTS,) for field in segment_fields):
        raise ValueError("plant hydraulic segment parameters must have shape (4,)")
    soil_shape = inputs.soil_root_conductance_level.shape
    if len(soil_shape) != 1 or soil_shape[0] < 1:
        raise ValueError("plant hydraulic soil inputs must be nonempty one-dimensional arrays")
    if inputs.soil_water_potential_level.shape != soil_shape:
        raise ValueError("soil water potential and root conductance shapes disagree")
    if inputs.soil_node_depth_level.shape != soil_shape:
        raise ValueError("soil depth and root conductance shapes disagree")


def _molar_conversion_factor(inputs: PlantHydraulicPatchInputs) -> jnp.ndarray:
    denominator = _product(
        jnp.float64(UNIVERSAL_GAS_CONSTANT),
        jnp.float64(1.0e-3),
        inputs.reference_temperature,
    )
    return _product(
        fortran_divide(inputs.atmospheric_pressure, denominator),
        jnp.float64(1.0e6),
    )


def _potential_transpiration(
    inputs: PlantHydraulicPatchInputs,
    boundary_layer_conductance: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    conversion = _molar_conversion_factor(inputs)
    heat_conductance = _product(
        _sum(inputs.exposed_leaf_area, inputs.exposed_stem_area),
        boundary_layer_conductance,
    )
    potential = _product(
        inputs.air_density,
        heat_conductance,
        _sum(
            inputs.leaf_saturation_specific_humidity,
            -inputs.canopy_air_specific_humidity,
        ),
    )
    return conversion, potential


def transpiration_from_stomatal_conductance(
    inputs: PlantHydraulicPatchInputs,
    boundary_layer_conductance: jnp.ndarray,
    sunlit_stomatal_conductance: jnp.ndarray,
    shaded_stomatal_conductance: jnp.ndarray,
) -> TranspirationFluxes:

    conversion, potential = _potential_transpiration(
        inputs, boundary_layer_conductance
    )

    def leaf_flux(leaf_area: jnp.ndarray, stomatal: jnp.ndarray) -> jnp.ndarray:
        def positive_conductance(_: None) -> jnp.ndarray:
            series_resistance = _sum(
                fortran_divide(
                    jnp.float64(1.0), boundary_layer_conductance
                ),
                fortran_divide(jnp.float64(1.0), stomatal),
            )
            dry_fraction = fortran_divide(
                _product(
                    fortran_divide(
                        inputs.dry_foliage_fraction,
                        boundary_layer_conductance,
                    ),
                    fortran_divide(leaf_area, series_resistance),
                ),
                inputs.exposed_leaf_area,
            )
            return fortran_divide(
                _product(potential, dry_fraction), conversion
            )

        return lax.cond(
            stomatal > jnp.float64(0.0),
            positive_conductance,
            lambda _: jnp.float64(0.0),
            operand=None,
        )

    def positive_potential(_: None) -> TranspirationFluxes:
        return TranspirationFluxes(
            sunlit=leaf_flux(inputs.sunlit_leaf_area, sunlit_stomatal_conductance),
            shaded=leaf_flux(inputs.shaded_leaf_area, shaded_stomatal_conductance),
        )

    return lax.cond(
        (potential > jnp.float64(0.0))
        & (inputs.exposed_leaf_area > jnp.float64(0.0)),
        positive_potential,
        lambda _: TranspirationFluxes(jnp.float64(0.0), jnp.float64(0.0)),
        operand=None,
    )


def stomatal_conductance_from_transpiration(
    inputs: PlantHydraulicPatchInputs,
    boundary_layer_conductance: jnp.ndarray,
    transpiration: TranspirationFluxes,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    conversion, potential = _potential_transpiration(
        inputs, boundary_layer_conductance
    )

    leaf_area = jnp.stack((inputs.sunlit_leaf_area, inputs.shaded_leaf_area))
    flux = jnp.stack((transpiration.sunlit, transpiration.shaded))
    numerator = _assigned_product(
        boundary_layer_conductance,
        flux,
        conversion,
        inputs.exposed_leaf_area,
    )
    denominator = _assigned_sum(
        _assigned_product(
            potential,
            inputs.dry_foliage_fraction,
            leaf_area,
        ),
        -_assigned_product(flux, conversion, inputs.exposed_leaf_area),
    )
    conductance = fortran_masked_materialized_divide(
        fortran_materialize(numerator),
        fortran_materialize(denominator),
        flux > jnp.float64(0.0),
        jnp.zeros((2,), dtype=jnp.float64),
    )
    return conductance[SUNLIT_LEAF], conductance[SHADED_LEAF]


def _algebraic_root_water_balance_reference(conductance, potential, gravity, sunlit, shaded):
    total = fortran_ordered_sum(conductance)

    def without_conductance(_):
        return fortran_divide(
            fortran_ordered_sum(_assigned_sum(potential, -gravity)),
            jnp.float64(conductance.shape[0]),
        )

    def with_conductance(_):
        supply = fortran_ordered_sum(
            _assigned_product(conductance, _assigned_sum(potential, -gravity)),
        )
        return fortran_divide(_assigned_sum(supply, -sunlit, -shaded), total)

    root = lax.cond(jnp.abs(total) == jnp.float64(0.0),
                    without_conductance, with_conductance, operand=None)
    flux = fortran_ordered_sum(
        _assigned_product(conductance, _assigned_sum(potential, -root, -gravity)),
        initial=jnp.float64(0.0),
    )
    return root, flux


@jax.custom_jvp
def _algebraic_root_water_balance(conductance, potential, gravity, sunlit, shaded):
    return _algebraic_root_water_balance_reference(
        conductance, potential, gravity, sunlit, shaded,
    )


@_algebraic_root_water_balance.defjvp
def _algebraic_root_water_balance_jvp(primals, tangents):
    root, flux = _algebraic_root_water_balance_reference(*primals)
    conductance, potential, gravity, _, _ = primals
    dots = tuple(jnp.zeros_like(value, dtype=jnp.float64)
                 if getattr(dot, "dtype", None) == jax.dtypes.float0 else dot
                 for value, dot in zip(primals, tangents, strict=True))
    dg, dp, dz, dsun, dshade = dots
    total = fortran_ordered_sum(conductance)
    conducting = jnp.abs(total) != jnp.float64(0.0)
    drop = _assigned_sum(potential, -root, -gravity)
    droot_conducting = (
        jnp.sum(dg * drop + conductance * (dp - dz)) - dsun - dshade
    ) / jnp.where(conducting, total, jnp.float64(1.0))
    droot_empty = jnp.sum(dp - dz) / jnp.float64(conductance.shape[0])
    droot = jnp.where(conducting, droot_conducting, droot_empty)
    dflux_empty = jnp.sum(dg * drop + conductance * (dp - droot - dz))
    return (root, flux), (droot, jnp.where(conducting, dsun + dshade, dflux_empty))


def _vegetation_water_potential_reference(
    inputs: PlantHydraulicPatchInputs,
    parameters: PlantHydraulicPatchParameters,
    boundary_layer_conductance: jnp.ndarray,
    sunlit_stomatal_conductance: jnp.ndarray,
    shaded_stomatal_conductance: jnp.ndarray,
    *,
    _root_balance=_algebraic_root_water_balance_reference,
) -> VegetationWaterPotentialResult:

    _validate_patch_shapes(inputs, parameters)
    transpiration = transpiration_from_stomatal_conductance(
        inputs,
        boundary_layer_conductance,
        sunlit_stomatal_conductance,
        shaded_stomatal_conductance,
    )
    gravity_canopy = _assigned_product(jnp.float64(1000.0), inputs.canopy_top_height)
    gravity_soil = _assigned_product(jnp.float64(1000.0), inputs.soil_node_depth_level)
    root_potential, soil_flux = _root_balance(
        inputs.soil_root_conductance_level, inputs.soil_water_potential_level,
        gravity_soil, transpiration.sunlit, transpiration.shaded,
    )
    root_fraction = vulnerability_fraction(
        root_potential,
        parameters.water_potential_at_half_conductance_segment[ROOT],
        parameters.vulnerability_shape_segment[ROOT],
    )

    def conducting_stem(_: None) -> jnp.ndarray:
        denominator = _assigned_product(
            fortran_divide(
                _assigned_product(root_fraction, parameters.maximum_conductance_segment[ROOT]),
                inputs.canopy_top_height,
            ),
            inputs.total_stem_area,
        )
        return _assigned_sum(
            root_potential,
            -gravity_canopy,
            -fortran_divide(_assigned_sum(transpiration.sunlit, transpiration.shaded), denominator),
        )

    xylem_potential = lax.cond(
        (inputs.total_stem_area > jnp.float64(0.0))
        & (root_fraction > jnp.float64(0.0)),
        conducting_stem,
        lambda _: _assigned_sum(root_potential, -gravity_canopy),
        operand=None,
    )
    xylem_fraction = vulnerability_fraction(
        xylem_potential,
        parameters.water_potential_at_half_conductance_segment[XYLEM],
        parameters.vulnerability_shape_segment[XYLEM],
    )

    def leaf_potential(leaf_area: jnp.ndarray, flux: jnp.ndarray) -> jnp.ndarray:
        return lax.cond(
            (leaf_area > jnp.float64(0.0))
            & (xylem_fraction > jnp.float64(0.0)),
            lambda _: _assigned_sum(
                xylem_potential,
                -fortran_divide(
                    flux,
                    _assigned_product(
                        xylem_fraction, parameters.maximum_conductance_segment[XYLEM], leaf_area,
                    ),
                ),
            ),
            lambda _: xylem_potential,
            operand=None,
        )

    shaded_potential = leaf_potential(
        inputs.shaded_leaf_area, transpiration.shaded
    )
    sunlit_potential = leaf_potential(
        inputs.sunlit_leaf_area, transpiration.sunlit
    )
    return VegetationWaterPotentialResult(
        water_potential_segment=jnp.stack(
            (sunlit_potential, shaded_potential, xylem_potential, root_potential)
        ),
        soil_flux=soil_flux,
        transpiration=transpiration,
    )


def vegetation_water_potential(
    inputs: PlantHydraulicPatchInputs,
    parameters: PlantHydraulicPatchParameters,
    boundary_layer_conductance: jnp.ndarray,
    sunlit_stomatal_conductance: jnp.ndarray,
    shaded_stomatal_conductance: jnp.ndarray,
) -> VegetationWaterPotentialResult:
    return _vegetation_water_potential_reference(
        inputs, parameters, boundary_layer_conductance,
        sunlit_stomatal_conductance, shaded_stomatal_conductance,
        _root_balance=_algebraic_root_water_balance,
    )


def spac_flux_divergence(
    water_potential_segment: jnp.ndarray,
    inputs: PlantHydraulicPatchInputs,
    parameters: PlantHydraulicPatchParameters,
    transpiration: TranspirationFluxes,
    *,
    _power_pair=None,
) -> jnp.ndarray:

    _validate_patch_shapes(inputs, parameters)
    if water_potential_segment.shape != (NUMBER_OF_VEGETATION_SEGMENTS,):
        raise ValueError("vegetation water potential must have shape (4,)")
    x = jnp.asarray(water_potential_segment, dtype=jnp.float64)
    if _power_pair is None:
        _power_pair = _weibull_power_pair(
            x, parameters.water_potential_at_half_conductance_segment,
            parameters.vulnerability_shape_segment,
        )
    fraction = jnp.where(_power_pair[1] < jnp.float64(0.005),
                         jnp.float64(0.0), _power_pair[1])
    gravity_canopy = _product(
        inputs.canopy_top_height, jnp.float64(1000.0)
    )
    gravity_soil = _product(
        inputs.soil_node_depth_level, jnp.float64(1000.0)
    )
    sunlit_flow = _product(
        inputs.sunlit_leaf_area,
        parameters.maximum_conductance_segment[SUNLIT_LEAF],
        fraction[XYLEM],
        _sum(x[XYLEM], -x[SUNLIT_LEAF]),
    )
    shaded_flow = _product(
        inputs.shaded_leaf_area,
        parameters.maximum_conductance_segment[SHADED_LEAF],
        fraction[XYLEM],
        _sum(x[XYLEM], -x[SHADED_LEAF]),
    )
    stem_flow = _product(
        fortran_divide(
            _product(
                inputs.total_stem_area,
                parameters.maximum_conductance_segment[XYLEM],
            ),
            inputs.canopy_top_height,
        ),
        fraction[ROOT],
        _sum(x[ROOT], -x[XYLEM], -gravity_canopy),
    )
    root_soil_potential = fortran_ordered_sum(
        inputs.soil_root_conductance_level * (x[ROOT] + gravity_soil)
    )
    soil_supply = fortran_ordered_sum(
        inputs.soil_root_conductance_level * inputs.soil_water_potential_level
    )
    residual = jnp.stack(
        (
            _sum(
                _product(transpiration.sunlit, fraction[SUNLIT_LEAF]),
                -sunlit_flow,
            ),
            _sum(
                _product(transpiration.shaded, fraction[SHADED_LEAF]),
                -shaded_flow,
            ),
            _sum(sunlit_flow, shaded_flow, -stem_flow),
            _sum(stem_flow, root_soil_potential, -soil_supply),
        )
    )
    return lax.cond(
        inputs.shaded_leaf_area < jnp.float64(MINIMUM_TRANSPIRING_LEAF_AREA),
        lambda value: (
            jnp.stack((value[1], value[0], value[2], value[3]))
            if default_backend() == "gpu" and (
                _USE_FUSED_NATIVE_NEWTON or _native_tile_trace.get()
            )
            else value[jnp.array([1, 0, 2, 3])]
        ),
        lambda value: value,
        residual,
    )


def spac_negative_jacobian(
    water_potential_segment: jnp.ndarray,
    inputs: PlantHydraulicPatchInputs,
    parameters: PlantHydraulicPatchParameters,
    transpiration: TranspirationFluxes,
    *,
    _power_pair=None,
) -> jnp.ndarray:

    _validate_patch_shapes(inputs, parameters)
    if water_potential_segment.shape != (NUMBER_OF_VEGETATION_SEGMENTS,):
        raise ValueError("vegetation water potential must have shape (4,)")
    x = jnp.asarray(water_potential_segment, dtype=jnp.float64)
    if _power_pair is None:
        _power_pair = _weibull_power_pair(
            x, parameters.water_potential_at_half_conductance_segment,
            parameters.vulnerability_shape_segment,
        )
    fraction = jnp.where(_power_pair[1] < jnp.float64(0.005),
                         jnp.float64(0.0), _power_pair[1])
    derivative = _weibull_derivative(
        x, parameters.vulnerability_shape_segment, _power_pair,
    )
    gravity_canopy = _product(
        inputs.canopy_top_height, jnp.float64(1000.0)
    )
    sunlit_conductance = _product(
        inputs.sunlit_leaf_area,
        parameters.maximum_conductance_segment[SUNLIT_LEAF],
    )
    shaded_conductance = _product(
        inputs.shaded_leaf_area,
        parameters.maximum_conductance_segment[SHADED_LEAF],
    )
    stem_conductance = fortran_divide(
        _product(
            inputs.total_stem_area,
            parameters.maximum_conductance_segment[XYLEM],
        ),
        inputs.canopy_top_height,
    )
    matrix = jnp.zeros(
        (NUMBER_OF_VEGETATION_SEGMENTS, NUMBER_OF_VEGETATION_SEGMENTS),
        dtype=jnp.float64,
    )
    matrix = matrix.at[0, 0].set(
        _sum(
            -_product(sunlit_conductance, fraction[XYLEM]),
            -_product(transpiration.sunlit, derivative[SUNLIT_LEAF]),
        )
    )
    matrix = matrix.at[0, 2].set(
        _sum(
            _product(
                sunlit_conductance,
                derivative[XYLEM],
                _sum(x[XYLEM], -x[SUNLIT_LEAF]),
            ),
            _product(sunlit_conductance, fraction[XYLEM]),
        )
    )
    matrix = matrix.at[1, 1].set(
        _sum(
            -_product(shaded_conductance, fraction[XYLEM]),
            -_product(transpiration.shaded, derivative[SHADED_LEAF]),
        )
    )
    matrix = matrix.at[1, 2].set(
        _sum(
            _product(
                shaded_conductance,
                derivative[XYLEM],
                _sum(x[XYLEM], -x[SHADED_LEAF]),
            ),
            _product(shaded_conductance, fraction[XYLEM]),
        )
    )
    matrix = matrix.at[2, 0].set(
        _product(sunlit_conductance, fraction[XYLEM])
    )
    matrix = matrix.at[2, 1].set(
        _product(shaded_conductance, fraction[XYLEM])
    )
    matrix = matrix.at[2, 2].set(
        _sum(
            -_product(
                sunlit_conductance,
                derivative[XYLEM],
                _sum(x[XYLEM], -x[SUNLIT_LEAF]),
            ),
            -_product(sunlit_conductance, fraction[XYLEM]),
            -_product(
                shaded_conductance,
                derivative[XYLEM],
                _sum(x[XYLEM], -x[SHADED_LEAF]),
            ),
            -_product(shaded_conductance, fraction[XYLEM]),
            -_product(stem_conductance, fraction[ROOT]),
        )
    )
    matrix = matrix.at[2, 3].set(
        _sum(
            _product(
                stem_conductance,
                derivative[ROOT],
                _sum(x[ROOT], -x[XYLEM], -gravity_canopy),
            ),
            _product(stem_conductance, fraction[ROOT]),
        )
    )
    matrix = matrix.at[3, 2].set(
        _product(stem_conductance, fraction[ROOT])
    )
    matrix = matrix.at[3, 3].set(
        _sum(
            -_product(stem_conductance, fraction[ROOT]),
            -_product(
                stem_conductance,
                derivative[ROOT],
                _sum(x[ROOT], -x[XYLEM], -gravity_canopy),
            ),
            -fortran_ordered_sum(inputs.soil_root_conductance_level),
        )
    )
    return matrix


def _inverse_scale(determinant):
    divisor = jnp.where(
        jnp.abs(determinant) <= jnp.float64(SINGULAR_JACOBIAN_TOLERANCE),
        jnp.float64(1.0), determinant,
    )
    return fortran_divide(jnp.float64(1.0), divisor)


def _general_inverse(matrix: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    determinant = _sum(
        _product(matrix[3, 3], matrix[1, 1], matrix[2, 2], matrix[0, 0]),
        -_product(matrix[3, 3], matrix[1, 1], matrix[2, 0], matrix[0, 2]),
        -_product(matrix[3, 3], matrix[2, 1], matrix[1, 2], matrix[0, 0]),
        -_product(matrix[3, 2], matrix[0, 0], matrix[1, 1], matrix[2, 3]),
    )
    leading = _inverse_scale(determinant)
    inverse = jnp.zeros_like(matrix)
    inverse = inverse.at[0, 0].set(
        _sum(
            _product(leading, matrix[3, 3], matrix[1, 1], matrix[2, 2]),
            -_product(leading, matrix[3, 3], matrix[2, 1], matrix[1, 2]),
            -_product(leading, matrix[3, 2], matrix[1, 1], matrix[2, 3]),
        )
    )
    inverse = inverse.at[1, 0].set(
        _product(leading, matrix[1, 2], matrix[3, 3], matrix[2, 0])
    )
    inverse = inverse.at[2, 0].set(
        -_product(leading, matrix[3, 3], matrix[1, 1], matrix[2, 0])
    )
    inverse = inverse.at[3, 0].set(
        _product(leading, matrix[3, 2], matrix[1, 1], matrix[2, 0])
    )
    inverse = inverse.at[0, 1].set(
        _product(leading, matrix[0, 2], matrix[3, 3], matrix[2, 1])
    )
    inverse = inverse.at[1, 1].set(
        _sum(
            _product(leading, matrix[3, 3], matrix[2, 2], matrix[0, 0]),
            -_product(leading, matrix[3, 3], matrix[2, 0], matrix[0, 2]),
            -_product(leading, matrix[3, 2], matrix[0, 0], matrix[2, 3]),
        )
    )
    inverse = inverse.at[2, 1].set(
        -_product(leading, matrix[0, 0], matrix[3, 3], matrix[2, 1])
    )
    inverse = inverse.at[3, 1].set(
        _product(leading, matrix[3, 2], matrix[0, 0], matrix[2, 1])
    )
    inverse = inverse.at[0, 2].set(
        -_product(leading, matrix[0, 2], matrix[1, 1], matrix[3, 3])
    )
    inverse = inverse.at[1, 2].set(
        -_product(leading, matrix[1, 2], matrix[0, 0], matrix[3, 3])
    )
    inverse = inverse.at[2, 2].set(
        _product(leading, matrix[1, 1], matrix[0, 0], matrix[3, 3])
    )
    inverse = inverse.at[3, 2].set(
        -_product(leading, matrix[3, 2], matrix[0, 0], matrix[1, 1])
    )
    inverse = inverse.at[0, 3].set(
        _product(leading, matrix[0, 2], matrix[2, 3], matrix[1, 1])
    )
    inverse = inverse.at[1, 3].set(
        _product(leading, matrix[1, 2], matrix[2, 3], matrix[0, 0])
    )
    inverse = inverse.at[2, 3].set(
        -_product(leading, matrix[2, 3], matrix[0, 0], matrix[1, 1])
    )
    inverse = inverse.at[3, 3].set(
        _sum(
            _product(leading, matrix[1, 1], matrix[2, 2], matrix[0, 0]),
            -_product(leading, matrix[1, 1], matrix[2, 0], matrix[0, 2]),
            -_product(leading, matrix[2, 1], matrix[1, 2], matrix[0, 0]),
        )
    )
    return determinant, inverse


def _reduced_inverse(
    matrix: jnp.ndarray,
    shaded_leaf_area: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    matrix = lax.cond(
        shaded_leaf_area
        <= jnp.float64(MINIMUM_TRANSPIRING_LEAF_AREA),
        lambda value: value.at[1, 1]
        .set(value[0, 0])
        .at[2, 1]
        .set(value[2, 0])
        .at[1, 2]
        .set(value[0, 2]),
        lambda value: value,
        matrix,
    )
    determinant = _sum(
        _product(matrix[1, 1], matrix[2, 2], matrix[3, 3]),
        -_product(matrix[2, 3], matrix[1, 1], matrix[3, 2]),
        -_product(matrix[1, 2], matrix[2, 1], matrix[3, 3]),
    )
    inverse = jnp.zeros_like(matrix)
    inverse = inverse.at[1, 1].set(
        _sum(
            _product(matrix[2, 2], matrix[3, 3]),
            -_product(matrix[2, 3], matrix[3, 2]),
        )
    )
    inverse = inverse.at[1, 2].set(-_product(matrix[1, 2], matrix[3, 3]))
    inverse = inverse.at[1, 3].set(_product(matrix[2, 3], matrix[1, 2]))
    inverse = inverse.at[2, 1].set(-_product(matrix[2, 1], matrix[3, 3]))
    inverse = inverse.at[2, 2].set(_product(matrix[1, 1], matrix[3, 3]))
    inverse = inverse.at[2, 3].set(-_product(matrix[2, 3], matrix[1, 1]))
    inverse = inverse.at[3, 1].set(_product(matrix[2, 1], matrix[3, 2]))
    inverse = inverse.at[3, 2].set(-_product(matrix[1, 1], matrix[3, 2]))
    inverse = inverse.at[3, 3].set(
        _sum(
            _product(matrix[1, 1], matrix[2, 2]),
            -_product(matrix[1, 2], matrix[2, 1]),
        )
    )
    inverse = _product(_inverse_scale(determinant), inverse)
    return determinant, inverse


def spac_inverse_jacobian(
    water_potential_segment: jnp.ndarray,
    inputs: PlantHydraulicPatchInputs,
    parameters: PlantHydraulicPatchParameters,
    transpiration: TranspirationFluxes,
    *,
    _power_pair=None,
) -> SpacLinearization:

    matrix = spac_negative_jacobian(
        water_potential_segment, inputs, parameters, transpiration,
        _power_pair=_power_pair,
    )
    full_canopy = (
        inputs.sunlit_leaf_area
        > jnp.float64(MINIMUM_TRANSPIRING_LEAF_AREA)
    ) & (
        inputs.shaded_leaf_area
        > jnp.float64(MINIMUM_TRANSPIRING_LEAF_AREA)
    )
    determinant, inverse = lax.cond(
        full_canopy,
        _general_inverse,
        lambda value: _reduced_inverse(value, inputs.shaded_leaf_area),
        matrix,
    )
    singular = jnp.abs(determinant) <= jnp.float64(SINGULAR_JACOBIAN_TOLERANCE)
    inverse = jnp.where(singular, jnp.zeros_like(inverse), inverse)
    return SpacLinearization(matrix, inverse, singular)


def _newton_spac_solution_reference(
    initial_water_potential: jnp.ndarray,
    inputs: PlantHydraulicPatchInputs,
    parameters: PlantHydraulicPatchParameters,
    transpiration: TranspirationFluxes,
    *,
    enabled: jnp.ndarray = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    initial = (
        initial_water_potential,
        jnp.asarray(enabled, dtype=jnp.bool_),
        jnp.bool_(False),
        jnp.int32(0),
    )

    def scan_step(carry, iteration):
        def active_step(current):
            x, _, _, _ = current
            power_pair = _weibull_power_pair(
                x, parameters.water_potential_at_half_conductance_segment,
                parameters.vulnerability_shape_segment,
            )
            residual = spac_flux_divergence(
                x, inputs, parameters, transpiration, _power_pair=power_pair
            )
            flux_norm = jnp.sqrt(fortran_ordered_sum(residual * residual))
            flux_converged = flux_norm < jnp.float64(
                FLUX_BALANCE_TOLERANCE
            ) * (transpiration.sunlit + transpiration.shaded)
            iteration_exceeded = iteration > jnp.int32(MAXIMUM_NEWTON_ITERATIONS)

            def stopped_before_linearization(_: None):
                return x, jnp.bool_(False), jnp.bool_(False), iteration

            def linearized(_: None):
                linearization = spac_inverse_jacobian(
                    x, inputs, parameters, transpiration, _power_pair=power_pair
                )

                def singular_result(_: None):
                    return x, jnp.bool_(False), jnp.bool_(True), iteration

                def regular_result(_: None):
                    full_canopy = (
                        inputs.sunlit_leaf_area
                        > jnp.float64(MINIMUM_TRANSPIRING_LEAF_AREA)
                    ) & (
                        inputs.shaded_leaf_area
                        > jnp.float64(MINIMUM_TRANSPIRING_LEAF_AREA)
                    )
                    full_step = _matrix_vector_product(
                        linearization.inverse_negative_jacobian, residual
                    )
                    reduced_step = jnp.concatenate(
                        (
                            jnp.zeros((1,), dtype=jnp.float64),
                            _matrix_vector_product(
                                linearization.inverse_negative_jacobian[1:, 1:],
                                residual[1:],
                            ),
                        )
                    )
                    delta = jnp.where(full_canopy, full_step, reduced_step)
                    maximum_delta = jnp.max(jnp.abs(delta))
                    delta = lax.cond(
                        maximum_delta
                        > jnp.float64(MAXIMUM_WATER_POTENTIAL_STEP),
                        lambda value: jnp.float64(
                            MAXIMUM_WATER_POTENTIAL_STEP
                        )
                        * value
                        / maximum_delta,
                        lambda value: value,
                        delta,
                    )

                    def update_full(_: None):
                        return fortran_add(x, delta)

                    def update_reduced(_: None):
                        def shaded_only(_: None):
                            updated = fortran_add(x, delta)
                            return updated.at[SUNLIT_LEAF].set(updated[XYLEM])

                        def sunlit_only(_: None):
                            updated = x.at[XYLEM:ROOT + 1].set(
                                fortran_add(
                                    x[XYLEM:ROOT + 1],
                                    delta[XYLEM:ROOT + 1],
                                )
                            )
                            updated = updated.at[SUNLIT_LEAF].set(
                                fortran_add(
                                    x[SUNLIT_LEAF],
                                    delta[SHADED_LEAF],
                                )
                            )
                            return updated.at[SHADED_LEAF].set(updated[XYLEM])

                        return lax.cond(
                            inputs.shaded_leaf_area
                            > jnp.float64(MINIMUM_TRANSPIRING_LEAF_AREA),
                            shaded_only,
                            sunlit_only,
                            operand=None,
                        )

                    updated = lax.cond(
                        full_canopy, update_full, update_reduced, operand=None
                    )
                    step_norm = jnp.sqrt(fortran_ordered_sum(delta * delta))
                    step_converged = step_norm < jnp.float64(
                        WATER_POTENTIAL_STEP_TOLERANCE
                    )

                    def constrain_gradient(value):
                        value = value.at[XYLEM].set(
                            jnp.minimum(value[XYLEM], value[ROOT])
                        )
                        value = value.at[SUNLIT_LEAF].set(
                            jnp.minimum(value[SUNLIT_LEAF], value[XYLEM])
                        )
                        return value.at[SHADED_LEAF].set(
                            jnp.minimum(value[SHADED_LEAF], value[XYLEM])
                        )

                    updated = lax.cond(
                        step_converged,
                        lambda value: value,
                        constrain_gradient,
                        updated,
                    )
                    updated = fortran_materialize(updated)
                    return (
                        updated,
                        ~step_converged,
                        jnp.bool_(False),
                        iteration,
                    )

                return lax.cond(
                    linearization.singular,
                    singular_result,
                    regular_result,
                    operand=None,
                )

            return lax.cond(
                flux_converged | iteration_exceeded,
                stopped_before_linearization,
                linearized,
                operand=None,
            )

        next_carry = lax.cond(
            carry[1], active_step, lambda current: current, carry
        )
        return next_carry, None

    final = converged_iteration(
        scan_step,
        initial,
        jnp.arange(1, MAXIMUM_NEWTON_ITERATIONS + 2, dtype=jnp.int32),
        lambda state: state[1],
    )
    water_potential, _, singular, iterations = final
    return water_potential, singular, iterations


def _newton_spac_solution(
    initial_water_potential: jnp.ndarray,
    inputs: PlantHydraulicPatchInputs,
    parameters: PlantHydraulicPatchParameters,
    transpiration: TranspirationFluxes,
    *,
    enabled: jnp.ndarray = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if _USE_FUSED_NATIVE_NEWTON and default_backend() == "gpu":
        from clm_jax.physics.plant_hydraulics_gpu import solve_newton_gpu
        return solve_newton_gpu(
            initial_water_potential, inputs, parameters, transpiration, enabled,
        )
    return _newton_spac_solution_reference(
        initial_water_potential, inputs, parameters, transpiration, enabled=enabled,
    )


def plant_hydraulic_stress(
    inputs: PlantHydraulicPatchInputs,
    parameters: PlantHydraulicPatchParameters,
    state: PlantHydraulicPatchState,
    boundary_layer_conductance: jnp.ndarray,
    sunlit_stomatal_conductance: jnp.ndarray,
    shaded_stomatal_conductance: jnp.ndarray,
    *,
    enabled: jnp.ndarray = True,
) -> PlantHydraulicStressResult:

    _validate_patch_shapes(inputs, parameters)
    if state.water_potential_segment.shape != (
        NUMBER_OF_VEGETATION_SEGMENTS,
    ):
        raise ValueError("vegetation water potential state must have shape (4,)")
    incoming = jnp.asarray(state.water_potential_segment, dtype=jnp.float64)
    night = incoming[SUNLIT_LEAF] > jnp.float64(0.0)
    initial = lax.cond(
        night,
        lambda value: value.at[SUNLIT_LEAF].set(value[SHADED_LEAF]),
        lambda value: value,
        incoming,
    )
    demand = transpiration_from_stomatal_conductance(
        inputs,
        boundary_layer_conductance,
        sunlit_stomatal_conductance,
        shaded_stomatal_conductance,
    )
    has_leaf_area = (
        inputs.sunlit_leaf_area
        > jnp.float64(MINIMUM_TRANSPIRING_LEAF_AREA)
    ) | (
        inputs.shaded_leaf_area
        > jnp.float64(MINIMUM_TRANSPIRING_LEAF_AREA)
    )
    has_demand = has_leaf_area & (
        (demand.sunlit > jnp.float64(0.0))
        | (demand.shaded > jnp.float64(0.0))
    )

    def solve(_: None):
        return _newton_spac_solution(
            initial, inputs, parameters, demand, enabled=enabled & has_demand
        )

    solved, singular, iterations = lax.cond(
        has_demand,
        solve,
        lambda _: (initial, jnp.bool_(True), jnp.int32(0)),
        operand=None,
    )

    def algebraic(_: None):
        result = vegetation_water_potential(
            inputs,
            parameters,
            boundary_layer_conductance,
            sunlit_stomatal_conductance,
            shaded_stomatal_conductance,
        )
        fraction = vulnerability_fraction(
            result.water_potential_segment,
            parameters.water_potential_at_half_conductance_segment,
            parameters.vulnerability_shape_segment,
        )
        return (
            result.water_potential_segment,
            fraction[SUNLIT_LEAF],
            fraction[SHADED_LEAF],
        )

    fraction = vulnerability_fraction(
        solved,
        parameters.water_potential_at_half_conductance_segment,
        parameters.vulnerability_shape_segment,
    )
    attenuated = TranspirationFluxes(
        _product(demand.sunlit, fraction[SUNLIT_LEAF]),
        _product(demand.shaded, fraction[SHADED_LEAF]),
    )
    stressed_sunlit, stressed_shaded = stomatal_conductance_from_transpiration(
        inputs, boundary_layer_conductance, attenuated
    )
    stressed_conductance = jnp.stack((stressed_sunlit, stressed_shaded))
    unstressed_conductance = jnp.stack(
        (sunlit_stomatal_conductance, shaded_stomatal_conductance)
    )
    positive_demand = jnp.stack((demand.sunlit, demand.shaded)) > jnp.float64(0.0)
    stress = fortran_masked_materialized_divide(
        fortran_materialize(stressed_conductance),
        fortran_materialize(unstressed_conductance),
        positive_demand,
        fraction[:2],
    )
    sunlit_stress = stress[SUNLIT_LEAF]
    shaded_stress = stress[SHADED_LEAF]

    algebraic_water_potential, algebraic_sunlit, algebraic_shaded = lax.cond(
        singular,
        algebraic,
        lambda _: (
            solved,
            jnp.float64(0.0),
            jnp.float64(0.0),
        ),
        operand=None,
    )
    water_potential = jnp.where(singular, algebraic_water_potential, solved)
    sunlit_stress = jnp.where(singular, algebraic_sunlit, sunlit_stress)
    shaded_stress = jnp.where(singular, algebraic_shaded, shaded_stress)
    sunlit_stress = jnp.where(
        sunlit_stress < jnp.float64(0.01), jnp.float64(0.0), sunlit_stress
    )
    shaded_stress = jnp.where(
        shaded_stress < jnp.float64(0.01), jnp.float64(0.0), shaded_stress
    )

    def nighttime_state(_: None):
        final = vegetation_water_potential(
            inputs,
            parameters,
            boundary_layer_conductance,
            _product(sunlit_stress, sunlit_stomatal_conductance),
            _product(shaded_stress, shaded_stomatal_conductance),
        )
        return PlantHydraulicPatchState(
            final.water_potential_segment,
            jnp.maximum(final.soil_flux, jnp.float64(0.0)),
        )

    final_state = lax.cond(
        night,
        nighttime_state,
        lambda _: PlantHydraulicPatchState(
            water_potential, state.vegetation_transpiration
        ),
        operand=None,
    )
    return PlantHydraulicStressResult(
        state=final_state,
        sunlit_stress=sunlit_stress,
        shaded_stress=shaded_stress,
        iterations=iterations,
        used_algebraic_fallback=singular,
    )
