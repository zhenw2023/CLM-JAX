
from __future__ import annotations

from contextvars import ContextVar
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct, default_backend, eval_shape, lax, tree_util

from clm_jax.constants import (
    FREEZING_TEMPERATURE_FRESHWATER,
    GRAVITY,
    LATENT_HEAT_VAPORIZATION,
    SPECIFIC_HEAT_DRY_AIR,
    STEFAN_BOLTZMANN,
    VON_KARMAN,
)
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.converged_iteration import (
    _dynamic_primal,
    checkpointed_scan, checkpoint_iteration,
    converged_iteration,
)
from clm_jax.physics.friction_velocity import (
    FrictionVelocityState,
    friction_velocity,
    monin_obukhov_initialize,
)
from clm_jax.physics.human_index import fast_human_stress_indices
from clm_jax.physics.luna import (
    LunaCapacityInputs,
    LunaCapacityParameters,
    LunaCapacityState,
    LunaDailyClimateInputs,
    LunaDailyClimateState,
    LunaTenDayClimateInputs,
    LunaTenDayClimateState,
    luna_accumulate_daily,
    luna_accumulate_ten_day,
    luna_clear_daily,
    luna_update_photosynthesis_capacity,
)
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_hardware_add,
    fortran_hardware_multiply,
    fortran_materialize,
    fortran_ordered_sum,
    fortran_product,
    fortran_rounding_barrier,
    strict_exp,
    strict_negative_exp,
)
from clm_jax.physics.ozone import OzoneStressState, ozone_stress_off
from clm_jax.physics.photosynthesis import (
    SHADED,
    SUNLIT,
    HydraulicPhotosynthesisBiochemicalParameters,
    HydraulicPhotosynthesisEnvironment,
    HydraulicPhotosynthesisPatchControls,
    PhotosynthesisTotalInputs,
    PhotosynthesisTotalState,
    photosynthesis_total,
    solve_hydraulic_photosynthesis_patch,
)
from clm_jax.physics.plant_hydraulics import (
    PlantHydraulicPatchInputs,
    PlantHydraulicPatchParameters,
    PlantHydraulicPatchState,
)
from clm_jax.physics.pointwise import active_scalar_map, scalar_map
from clm_jax.physics.qsat import qsat

MAXIMUM_CANOPY_ENERGY_ITERATIONS = 41
MINIMUM_CANOPY_ENERGY_ITERATIONS = 3
MAXIMUM_LEAF_TEMPERATURE_STEP = 1.0
LEAF_TEMPERATURE_TOLERANCE = 0.01
LEAF_LATENT_HEAT_TOLERANCE = 0.1
MOLECULAR_VISCOSITY_AIR = 1.5e-5
SOIL_DRAG_COEFFICIENT = 0.004
CRITICAL_TOTAL_LEAF_STEM_AREA = 2.0
DRY_LITTER_AREA_INDEX = 0.5
DRY_LITTER_THICKNESS = 0.05
UNDERCANOPY_STABILITY_COEFFICIENT = 0.5
_coupled_water_balance_derivatives = ContextVar("clm_coupled_water_balance_ad", default=False)


class PrescribedCanopyPhotosynthesis(NamedTuple):

    sunlit_stomatal_resistance_patch: jnp.ndarray
    shaded_stomatal_resistance_patch: jnp.ndarray
    transpiration_wetness_patch: jnp.ndarray
    vegetation_transpiration_patch: jnp.ndarray


class CanopyHydraulicCouplingInputs(NamedTuple):

    biochemistry: HydraulicPhotosynthesisBiochemicalParameters
    controls: HydraulicPhotosynthesisPatchControls
    plant_inputs: PlantHydraulicPatchInputs
    plant_parameters: PlantHydraulicPatchParameters
    initial_plant_state: PlantHydraulicPatchState
    root_zone_wetness_patch: jnp.ndarray


class CanopyHydraulicIterationState(NamedTuple):

    plant_state: PlantHydraulicPatchState
    photosynthesis: PrescribedCanopyPhotosynthesis
    intracellular_co2_patch_leaf: jnp.ndarray
    hydraulic_stress_patch_leaf: jnp.ndarray
    stomatal_conductance_patch_leaf: jnp.ndarray
    rubisco_limited_photosynthesis_patch_leaf: jnp.ndarray
    electron_transport_limited_photosynthesis_patch_leaf: jnp.ndarray
    product_limited_photosynthesis_patch_leaf: jnp.ndarray
    gross_photosynthesis_patch_leaf: jnp.ndarray
    net_photosynthesis_patch_leaf: jnp.ndarray
    leaf_surface_co2_patch_leaf: jnp.ndarray
    solver_intracellular_co2_patch_leaf: jnp.ndarray
    maximum_carboxylation_rate_patch_leaf: jnp.ndarray
    maximum_electron_transport_rate_patch_leaf: jnp.ndarray
    triose_phosphate_utilization_rate_patch_leaf: jnp.ndarray
    pep_carboxylase_co2_slope_patch_leaf: jnp.ndarray
    leaf_maintenance_respiration_patch_leaf: jnp.ndarray
    electron_transport_rate_patch_leaf: jnp.ndarray
    co2_compensation_point_patch: jnp.ndarray
    michaelis_menten_co2_patch: jnp.ndarray
    michaelis_menten_o2_patch: jnp.ndarray
    canopy_vapor_pressure_deficit_patch: jnp.ndarray
    canopy_relative_humidity_patch: jnp.ndarray
    molar_conversion_factor_patch: jnp.ndarray
    boundary_layer_conductance_patch: jnp.ndarray
    outer_iterations_patch: jnp.ndarray
    inner_iterations_patch: jnp.ndarray
    hydraulic_iterations_patch: jnp.ndarray


class _CanopyHydraulicScanState(NamedTuple):
    energy: CanopyEnergyIterationState
    hydraulic: CanopyHydraulicIterationState


class CanopyEnergyInputs(NamedTuple):

    active_patch: jnp.ndarray
    eastward_wind_patch: jnp.ndarray
    northward_wind_patch: jnp.ndarray
    reference_temperature_patch: jnp.ndarray
    atmospheric_potential_temperature_patch: jnp.ndarray
    atmospheric_specific_humidity_patch: jnp.ndarray
    atmospheric_pressure_patch: jnp.ndarray
    air_density_patch: jnp.ndarray
    downward_longwave_patch: jnp.ndarray
    ground_temperature_patch: jnp.ndarray
    ground_specific_humidity_patch: jnp.ndarray
    snow_surface_temperature_patch: jnp.ndarray
    soil_surface_temperature_patch: jnp.ndarray
    surface_water_temperature_patch: jnp.ndarray
    snow_fraction_patch: jnp.ndarray
    surface_water_fraction_patch: jnp.ndarray
    ground_emissivity_patch: jnp.ndarray
    absorbed_shortwave_vegetation_patch: jnp.ndarray
    vegetation_emissivity_patch: jnp.ndarray
    exposed_leaf_area_index_patch: jnp.ndarray
    exposed_stem_area_index_patch: jnp.ndarray
    sunlit_leaf_area_index_patch: jnp.ndarray
    shaded_leaf_area_index_patch: jnp.ndarray
    vegetation_fraction_without_snow_patch: jnp.ndarray
    dry_foliage_fraction_patch: jnp.ndarray
    wet_foliage_fraction_patch: jnp.ndarray
    canopy_water_patch: jnp.ndarray
    snow_depth_patch: jnp.ndarray
    soil_evaporation_resistance_patch: jnp.ndarray
    ground_momentum_roughness_patch: jnp.ndarray
    canopy_top_height_patch: jnp.ndarray
    leaf_dimension_patch: jnp.ndarray
    forcing_height_wind_patch: jnp.ndarray
    forcing_height_temperature_patch: jnp.ndarray
    forcing_height_humidity_patch: jnp.ndarray


class CanopyEnergyParameters(NamedTuple):

    step_seconds: float | jnp.ndarray
    maximum_stable_zeta: float | jnp.ndarray = 0.5
    use_undercanopy_stability: bool | jnp.ndarray = False
    use_hydraulic_stress: bool | jnp.ndarray = True


class CanopyEnergyInitialState(NamedTuple):
    vegetation_temperature_patch: jnp.ndarray
    displacement_height_patch: jnp.ndarray
    vegetation_momentum_roughness_patch: jnp.ndarray
    friction_velocity: FrictionVelocityState
    iteration_work_patch: jnp.ndarray | None = None


class CanopyEnergyIterationState(NamedTuple):
    active_patch: jnp.ndarray
    vegetation_temperature_patch: jnp.ndarray
    canopy_air_temperature_patch: jnp.ndarray
    canopy_air_specific_humidity_patch: jnp.ndarray
    ground_humidity_difference_patch: jnp.ndarray
    monin_obukhov_length_patch: jnp.ndarray
    previous_monin_obukhov_length_patch: jnp.ndarray
    stability_wind_speed_patch: jnp.ndarray
    stability_sign_changes_patch: jnp.ndarray
    leaf_temperature_change_patch: jnp.ndarray
    previous_leaf_temperature_change_patch: jnp.ndarray
    leaf_latent_heat_flux_patch: jnp.ndarray
    previous_leaf_latent_heat_flux_patch: jnp.ndarray
    leaf_latent_heat_flux_change_patch: jnp.ndarray
    previous_vegetation_temperature_patch: jnp.ndarray
    vegetation_temperature_step_patch: jnp.ndarray
    leaf_saturation_vapor_pressure_patch: jnp.ndarray
    leaf_saturation_specific_humidity_patch: jnp.ndarray
    leaf_saturation_specific_humidity_derivative_patch: jnp.ndarray
    vegetation_sensible_heat_flux_patch: jnp.ndarray
    vegetation_evaporation_patch: jnp.ndarray
    vegetation_transpiration_patch: jnp.ndarray
    friction_velocity_value_patch: jnp.ndarray
    friction_stability_wind_input_patch: jnp.ndarray
    friction_momentum_numerator_patch: jnp.ndarray
    friction_momentum_denominator_patch: jnp.ndarray
    friction_momentum_zeta_patch: jnp.ndarray
    friction_temperature_profile_patch: jnp.ndarray
    friction_humidity_profile_patch: jnp.ndarray
    friction_temperature_profile_2m_patch: jnp.ndarray
    friction_humidity_profile_2m_patch: jnp.ndarray
    within_canopy_wind_patch: jnp.ndarray
    bare_soil_drag_patch: jnp.ndarray
    soil_drag_patch: jnp.ndarray
    leaf_boundary_resistance_patch: jnp.ndarray
    momentum_conductance_patch: jnp.ndarray
    momentum_resistance_patch: jnp.ndarray
    atmospheric_thermal_resistance_patch: jnp.ndarray
    ground_thermal_resistance_patch: jnp.ndarray
    atmospheric_moisture_resistance_patch: jnp.ndarray
    ground_moisture_resistance_patch: jnp.ndarray
    ground_sensible_conductance_patch: jnp.ndarray
    ground_latent_conductance_patch: jnp.ndarray
    normalized_leaf_sensible_conductance_patch: jnp.ndarray
    normalized_ground_sensible_conductance_patch: jnp.ndarray
    normalized_air_sensible_conductance_patch: jnp.ndarray
    normalized_ground_air_sensible_conductance_patch: jnp.ndarray
    normalized_air_leaf_sensible_conductance_patch: jnp.ndarray
    normalized_leaf_latent_conductance_patch: jnp.ndarray
    normalized_ground_latent_conductance_patch: jnp.ndarray
    normalized_air_latent_conductance_patch: jnp.ndarray
    normalized_ground_air_latent_conductance_patch: jnp.ndarray
    normalized_air_leaf_latent_conductance_patch: jnp.ndarray
    canopy_methane_conductance_patch: jnp.ndarray
    ground_methane_conductance_patch: jnp.ndarray
    leaf_sensible_linearization_patch: jnp.ndarray
    leaf_latent_humidity_gradient_patch: jnp.ndarray
    sensible_temperature_derivative_patch: jnp.ndarray
    latent_temperature_derivative_patch: jnp.ndarray
    temperature_scale_patch: jnp.ndarray
    humidity_scale_patch: jnp.ndarray
    virtual_temperature_scale_patch: jnp.ndarray
    stability_zeta_patch: jnp.ndarray
    convective_argument_patch: jnp.ndarray
    convective_velocity_patch: jnp.ndarray
    stability_wind_squared_sum_patch: jnp.ndarray
    energy_balance_correction_patch: jnp.ndarray
    iteration_count_patch: jnp.ndarray
    friction_velocity: FrictionVelocityState
    canopy_water_change_patch: jnp.ndarray | None = None
    canopy_water_remaining_patch: jnp.ndarray | None = None


class CanopyEnergyResult(NamedTuple):

    iteration: CanopyEnergyIterationState
    converged_patch: jnp.ndarray
    valid_forcing_height_patch: jnp.ndarray
    limited_vegetation_area_patch: jnp.ndarray
    aerodynamic_fraction_patch: jnp.ndarray
    displacement_height_patch: jnp.ndarray
    vegetation_momentum_roughness_patch: jnp.ndarray
    vegetation_sensible_roughness_patch: jnp.ndarray
    vegetation_latent_roughness_patch: jnp.ndarray
    atmospheric_longwave_coefficient_patch: jnp.ndarray
    vegetation_longwave_coefficient_patch: jnp.ndarray
    ground_longwave_coefficient_patch: jnp.ndarray


class CanopyHydraulicEnergyResult(NamedTuple):

    energy: CanopyEnergyResult
    hydraulic: CanopyHydraulicIterationState


def canopy_iteration_work(result: CanopyHydraulicEnergyResult) -> jnp.ndarray:
    return (
        (result.energy.iteration.iteration_count_patch << 12)
        + (result.hydraulic.outer_iterations_patch << 6)
        + result.hydraulic.inner_iterations_patch
    )


class CanopyEnergyTrace(NamedTuple):

    result: CanopyEnergyResult
    iteration_history: CanopyEnergyIterationState


class CanopyHydraulicEnergyTrace(NamedTuple):

    result: CanopyHydraulicEnergyResult
    energy_iteration_history: CanopyEnergyIterationState
    hydraulic_iteration_history: CanopyHydraulicIterationState


class CanopyFluxesPostprocessInputs(NamedTuple):

    snow_surface_specific_humidity_patch: jnp.ndarray
    soil_surface_specific_humidity_patch: jnp.ndarray
    surface_water_specific_humidity_patch: jnp.ndarray
    ground_specific_humidity_temperature_derivative_patch: jnp.ndarray
    ground_latent_heat_patch: jnp.ndarray


class CanopyFastHumanStressState(NamedTuple):

    temperature_celsius_patch: jnp.ndarray
    vapor_pressure_patch: jnp.ndarray
    wet_bulb_stull_patch: jnp.ndarray
    rural_wet_bulb_stull_patch: jnp.ndarray
    heat_index_patch: jnp.ndarray
    rural_heat_index_patch: jnp.ndarray
    apparent_temperature_patch: jnp.ndarray
    rural_apparent_temperature_patch: jnp.ndarray
    simplified_wet_bulb_globe_temperature_patch: jnp.ndarray
    rural_simplified_wet_bulb_globe_temperature_patch: jnp.ndarray
    humidex_patch: jnp.ndarray
    rural_humidex_patch: jnp.ndarray
    discomfort_index_stull_patch: jnp.ndarray
    rural_discomfort_index_stull_patch: jnp.ndarray


class CanopyFluxesPostprocessState(NamedTuple):

    canopy_energy_balance_error_patch: jnp.ndarray
    eastward_wind_stress_patch: jnp.ndarray
    northward_wind_stress_patch: jnp.ndarray
    sensible_heat_ground_patch: jnp.ndarray
    sensible_heat_snow_patch: jnp.ndarray
    sensible_heat_soil_patch: jnp.ndarray
    sensible_heat_surface_water_patch: jnp.ndarray
    ground_evaporation_patch: jnp.ndarray
    snow_evaporation_patch: jnp.ndarray
    soil_evaporation_patch: jnp.ndarray
    surface_water_evaporation_patch: jnp.ndarray
    reference_temperature_2m_patch: jnp.ndarray
    rural_reference_temperature_2m_patch: jnp.ndarray
    reference_specific_humidity_2m_patch: jnp.ndarray
    reference_relative_humidity_2m_patch: jnp.ndarray
    rural_reference_relative_humidity_2m_patch: jnp.ndarray
    human_stress: CanopyFastHumanStressState
    downward_longwave_below_canopy_patch: jnp.ndarray
    upward_longwave_above_canopy_patch: jnp.ndarray
    skin_temperature_patch: jnp.ndarray
    ground_sensible_derivative_patch: jnp.ndarray
    ground_latent_derivative_patch: jnp.ndarray
    ground_energy_derivative_patch: jnp.ndarray
    canopy_water_patch: jnp.ndarray
    liquid_canopy_water_patch: jnp.ndarray
    snow_canopy_water_patch: jnp.ndarray


class CanopyFluxesPostprocessParameters(NamedTuple):

    step_seconds: float | jnp.ndarray
    update_snow_canopy_water: bool | jnp.ndarray = True


class CanopyFluxesLunaInputs(NamedTuple):

    absorbed_par_sunlit_by_layer_patch: jnp.ndarray
    absorbed_par_shaded_by_layer_patch: jnp.ndarray
    sunlit_leaf_area_by_layer_patch: jnp.ndarray
    shaded_leaf_area_by_layer_patch: jnp.ndarray
    radiative_layer_count_patch: jnp.ndarray
    capacity_inputs: LunaCapacityInputs


class CanopyFluxesActiveState(NamedTuple):

    postprocess: CanopyFluxesPostprocessState
    photosynthesis_total: PhotosynthesisTotalState
    ozone_stress: OzoneStressState
    luna_daily: LunaDailyClimateState
    luna_ten_day: LunaTenDayClimateState
    luna_capacity: LunaCapacityState


class CanopyFluxesActiveParameters(NamedTuple):

    energy: CanopyEnergyParameters
    luna_capacity: LunaCapacityParameters
    is_end_day: bool | jnp.ndarray = False
    update_snow_canopy_water: bool | jnp.ndarray = True


class CanopyFluxesActiveResult(NamedTuple):

    energy_hydraulics: CanopyHydraulicEnergyResult
    state: CanopyFluxesActiveState


def _add(*terms: jnp.ndarray) -> jnp.ndarray:
    values = jnp.broadcast_arrays(*(jnp.asarray(term, dtype=jnp.float64) for term in terms))
    total = values[0]
    for term in values[1:]:
        total = fortran_hardware_add(total, term)
    return lax.optimization_barrier(total)


def _multiply(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    left, right = jnp.broadcast_arrays(
        jnp.asarray(left, dtype=jnp.float64),
        jnp.asarray(right, dtype=jnp.float64),
    )
    return lax.optimization_barrier(fortran_hardware_multiply(left, right))


def _divide(numerator: jnp.ndarray, denominator: jnp.ndarray) -> jnp.ndarray:
    return fortran_divide(numerator, denominator)


def _canopy_evaporation_balance_reference(water, transpiration, potential, seconds):
    evaporation = jnp.minimum(potential, _add(transpiration, _divide(water, seconds)))
    change = _multiply(_add(transpiration, -evaporation), seconds)
    remaining = jnp.maximum(jnp.float64(0.0), _add(water, change))
    return evaporation, change, remaining


@jax.custom_jvp
def _canopy_evaporation_balance(water, transpiration, potential, seconds):
    return _canopy_evaporation_balance_reference(water, transpiration, potential, seconds)


@_canopy_evaporation_balance.defjvp
def _canopy_evaporation_balance_jvp(primals, tangents):
    water, transpiration, potential, seconds = primals
    dw, dq, dp, dt = (
        jnp.zeros_like(value, dtype=jnp.float64)
        if getattr(dot, "dtype", None) == jax.dtypes.float0 else dot
        for value, dot in zip(primals, tangents, strict=True)
    )
    supply = _divide(water, seconds)
    limit = _add(transpiration, supply)
    capped = potential > limit
    at_limit = potential == limit
    supply_weight = jnp.where(capped, 1.0, jnp.where(at_limit, 0.5, 0.0))
    potential_weight = 1.0 - supply_weight
    limit_dot = dq + (dw - supply * dt) / seconds
    uncapped_change_dot = (dq - dp) * seconds + (transpiration - potential) * dt
    return _canopy_evaporation_balance_reference(*primals), (
        supply_weight * limit_dot + potential_weight * dp,
        -supply_weight * dw + potential_weight * uncapped_change_dot,
        potential_weight * (dw + uncapped_change_dot),
    )


def _coupled_canopy_evaporation_balance_reference(
    water, hydraulic_flow, nonhydraulic_flow, available, old_potential,
    new_potential, seconds, hydraulic,
):
    positive = old_potential > 0.0
    supply = _add(hydraulic_flow, _divide(water, seconds))
    maximum = _divide(supply, jnp.where(positive, old_potential, 1.0))
    fraction = jnp.where(positive, jnp.minimum(available, maximum), 1.0)
    potential = _multiply(fraction, new_potential)
    flow = jnp.where(hydraulic, hydraulic_flow, nonhydraulic_flow)
    excess = jnp.maximum(0.0, _add(potential, -flow, -_divide(water, seconds)))
    return (*_canopy_evaporation_balance_reference(water, flow, potential, seconds), excess)


@jax.custom_jvp
def _coupled_canopy_evaporation_balance(*arguments):
    return _coupled_canopy_evaporation_balance_reference(*arguments)


@_coupled_canopy_evaporation_balance.defjvp
def _coupled_canopy_evaporation_balance_jvp(primals, tangents):
    from clm_jax.physics.adjoint_scaling import sensitivity_product, sensitivity_sum

    water, qh, qn, available, old, new, seconds, hydraulic = primals
    dw, dqh, dqn, da, dold, dnew, dt = (
        jnp.zeros_like(value, dtype=jnp.float64)
        if getattr(dot, "dtype", None) == jax.dtypes.float0 else dot
        for value, dot in zip(primals[:7], tangents[:7], strict=True)
    )
    positive = old > 0.0
    safe_old = jnp.where(positive, old, 1.0)
    supply = _add(qh, _divide(water, seconds))
    maximum = _divide(supply, safe_old)
    fraction = jnp.where(positive, jnp.minimum(available, maximum), 1.0)
    raw = _multiply(fraction, new)
    q, dq = jnp.where(hydraulic, qh, qn), jnp.where(hydraulic, dqh, dqn)

    supply_active = positive & (maximum <= available)
    safe_supply_old = jnp.where(supply_active, old, 1.0)
    difference_ratio = _divide(old - new, safe_supply_old)
    amount = _add(water, _multiply(qh, seconds))
    amount_over_old = _divide(jnp.where(supply_active, amount, 0.0), safe_supply_old)
    damount = sensitivity_sum(dw, dqh * seconds, qh * dt)
    difference = jnp.where(hydraulic, 0.0, qn - qh)
    ddifference = jnp.where(hydraulic, 0.0, dqn - dqh)
    supply_store = seconds * difference + amount * difference_ratio
    potential_change = sensitivity_sum(
        dold, -dnew, -sensitivity_product(difference_ratio, dold))
    dsupply_store = sensitivity_sum(dt * difference, seconds * ddifference,
        sensitivity_product(difference_ratio, damount),
        sensitivity_product(amount_over_old, potential_change))
    available_rate = available * new
    davailable_rate = da * new + available * dnew
    available_store = water + (q - available_rate) * seconds
    davailable_store = sensitivity_sum(
        dw, (dq - davailable_rate) * seconds, (q - available_rate) * dt)
    nonpositive_store = water + (q - new) * seconds
    dnonpositive_store = sensitivity_sum(dw, (dq - dnew) * seconds, (q - new) * dt)
    fraction_supply = jnp.where(positive, jnp.where(
        maximum < available, 1.0, jnp.where(maximum > available, 0.0, 0.5)), 0.0)
    fraction_available = jnp.where(positive, 1.0 - fraction_supply, 0.0)
    fraction_nonpositive = jnp.where(positive, 0.0, 1.0)
    store = (fraction_supply * supply_store + fraction_available * available_store
             + fraction_nonpositive * nonpositive_store)
    dstore = sensitivity_sum(fraction_supply * dsupply_store,
        fraction_available * davailable_store, fraction_nonpositive * dnonpositive_store)
    limit = _add(q, _divide(water, seconds))
    tied_cap = jnp.where(store < 0.0, 1.0, jnp.where(store > 0.0, 0.0, 0.5))
    capped = jnp.where(raw > limit, 1.0, jnp.where(raw < limit, 0.0, tied_cap))
    retained = 1.0 - capped
    native = _coupled_canopy_evaporation_balance_reference(*primals)
    remaining_dot = retained * dstore
    change_dot = remaining_dot - dw
    evaporation_dot = dq - change_dot / seconds + native[1] * dt / (seconds * seconds)
    return native, (
        evaporation_dot,
        change_dot,
        remaining_dot,
        capped * (-dstore / seconds + store * dt / (seconds * seconds)),
    )


def _bounded_leaf_temperature_step(step, absolute_step):
    limited = absolute_step > jnp.float64(MAXIMUM_LEAF_TEMPERATURE_STEP)
    return jnp.where(
        limited,
        _multiply(
            jnp.float64(MAXIMUM_LEAF_TEMPERATURE_STEP),
            _divide(step, jnp.where(limited, absolute_step, jnp.float64(1.0))),
        ),
        step,
    )


def _canopy_phase_water_balance_reference(liquid, snow, change, warm):
    evaporated = -change
    liquid_exhausted = evaporated > liquid
    snow_exhausted = evaporated > snow
    warm_snow = _add(snow, liquid, change)
    warm_liquid = jnp.maximum(jnp.float64(0.0), _add(liquid, change))
    cold_liquid = _add(liquid, snow, change)
    cold_snow = jnp.maximum(jnp.float64(0.0), _add(snow, change))
    return (
        jnp.where(warm, warm_liquid, jnp.where(snow_exhausted, cold_liquid, liquid)),
        jnp.where(warm, jnp.where(liquid_exhausted, warm_snow, snow), cold_snow),
    )


@jax.custom_jvp
def _canopy_phase_water_balance(liquid, snow, change, warm):
    return _canopy_phase_water_balance_reference(liquid, snow, change, warm)


@_canopy_phase_water_balance.defjvp
def _canopy_phase_water_balance_jvp(primals, tangents):
    liquid, snow, change, warm = primals
    dl, ds, dc = (
        jnp.zeros_like(value, dtype=jnp.float64)
        if getattr(dot, "dtype", None) == jax.dtypes.float0 else dot
        for value, dot in zip(primals[:3], tangents[:3], strict=True)
    )

    def retained_fraction(trial):
        return jnp.where(trial > 0.0, 1.0, jnp.where(trial < 0.0, 0.0, 0.5))

    liquid_fraction = retained_fraction(_add(liquid, change))
    snow_fraction = retained_fraction(_add(snow, change))
    warm_change, cold_change = dl + dc, ds + dc
    total_change = lax.optimization_barrier(lax.optimization_barrier(dl + ds) + dc)
    return _canopy_phase_water_balance_reference(*primals), (
        jnp.where(warm, liquid_fraction * warm_change,
                  jnp.where(snow_fraction == 0.0, total_change,
                            dl + (1.0 - snow_fraction) * cold_change)),
        jnp.where(warm, jnp.where(liquid_fraction == 0.0, total_change,
                                 ds + (1.0 - liquid_fraction) * warm_change),
                  snow_fraction * cold_change),
    )


def _scalar_log(value: jnp.ndarray) -> jnp.ndarray:
    if default_backend() == "gpu":
        from clm_jax.physics.native_logarithm import native_log
        return native_log(value)
    shape = value.shape
    return jnp.reshape(scalar_map(jnp.log, jnp.reshape(value, (-1,))), shape)


def _scalar_exp(value: jnp.ndarray) -> jnp.ndarray:
    return strict_negative_exp(value)


def _scalar_power(value: jnp.ndarray, exponent: float) -> jnp.ndarray:
    if default_backend() == "gpu" and exponent in (0.333, 0.45):
        from clm_jax.physics.native_power import native_power
        return native_power(value, jnp.float64(exponent))
    shape = value.shape
    flattened = jnp.reshape(value, (-1,))
    exponents = jnp.full(flattened.shape, exponent, dtype=jnp.float64)
    return jnp.reshape(scalar_map(lambda pair: pair[0] ** pair[1], (flattened, exponents)), shape)


def _square(value: jnp.ndarray) -> jnp.ndarray:
    return _multiply(value, value)


def _cube(value: jnp.ndarray) -> jnp.ndarray:
    return _multiply(_square(value), value)


def _fourth_power(value: jnp.ndarray) -> jnp.ndarray:
    squared = _square(value)
    return _multiply(squared, squared)


def _update(mask: jnp.ndarray, candidate: jnp.ndarray, previous: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(mask, candidate, previous)


def _validate_shapes(
    inputs: CanopyEnergyInputs,
    photosynthesis: PrescribedCanopyPhotosynthesis,
    initial_state: CanopyEnergyInitialState,
) -> None:
    size = inputs.active_patch.shape[0]
    for group_name, group in (
        ("inputs", inputs),
        ("photosynthesis", photosynthesis),
        ("initial state", initial_state[:3]),
    ):
        names = group._fields if hasattr(group, "_fields") else CanopyEnergyInitialState._fields[:3]
        for name, value in zip(names, group, strict=True):
            if jnp.shape(value) != (size,):
                raise ValueError(f"{group_name} field {name} must have shape ({size},)")
    for name, value in zip(
        FrictionVelocityState._fields,
        initial_state.friction_velocity,
        strict=True,
    ):
        if jnp.shape(value) != (size,):
            raise ValueError(f"friction state field {name} must have shape ({size},)")


def _safe_inputs(inputs: CanopyEnergyInputs) -> CanopyEnergyInputs:
    active = jnp.asarray(inputs.active_patch, dtype=jnp.bool_)
    defaults = {
        "eastward_wind_patch": 1.0,
        "northward_wind_patch": 0.0,
        "reference_temperature_patch": 280.0,
        "atmospheric_potential_temperature_patch": 280.0,
        "atmospheric_specific_humidity_patch": 0.005,
        "atmospheric_pressure_patch": 100000.0,
        "air_density_patch": 1.2,
        "downward_longwave_patch": 300.0,
        "ground_temperature_patch": 280.0,
        "ground_specific_humidity_patch": 0.005,
        "snow_surface_temperature_patch": 280.0,
        "soil_surface_temperature_patch": 280.0,
        "surface_water_temperature_patch": 280.0,
        "snow_fraction_patch": 0.0,
        "surface_water_fraction_patch": 0.0,
        "ground_emissivity_patch": 0.96,
        "absorbed_shortwave_vegetation_patch": 0.0,
        "vegetation_emissivity_patch": 0.5,
        "exposed_leaf_area_index_patch": 1.0,
        "exposed_stem_area_index_patch": 0.1,
        "sunlit_leaf_area_index_patch": 0.6,
        "shaded_leaf_area_index_patch": 0.4,
        "vegetation_fraction_without_snow_patch": 1.0,
        "dry_foliage_fraction_patch": 0.8,
        "wet_foliage_fraction_patch": 0.2,
        "canopy_water_patch": 0.1,
        "snow_depth_patch": 0.0,
        "soil_evaporation_resistance_patch": 100.0,
        "ground_momentum_roughness_patch": 0.01,
        "canopy_top_height_patch": 1.0,
        "leaf_dimension_patch": 0.04,
        "forcing_height_wind_patch": 30.0,
        "forcing_height_temperature_patch": 30.0,
        "forcing_height_humidity_patch": 30.0,
    }
    values = [active]
    for name in CanopyEnergyInputs._fields[1:]:
        value = jnp.asarray(getattr(inputs, name), dtype=jnp.float64)
        values.append(jnp.where(active, value, jnp.float64(defaults[name])))
    return CanopyEnergyInputs(*values)


def _safe_photosynthesis(
    active: jnp.ndarray,
    photosynthesis: PrescribedCanopyPhotosynthesis,
) -> PrescribedCanopyPhotosynthesis:
    defaults = (100.0, 100.0, 1.0, 0.0)
    return PrescribedCanopyPhotosynthesis(
        *(
            jnp.where(
                active,
                jnp.asarray(value, dtype=jnp.float64),
                jnp.float64(default),
            )
            for value, default in zip(photosynthesis, defaults, strict=True)
        )
    )


def _validate_hydraulic_coupling(
    patch_count: int,
    coupling: CanopyHydraulicCouplingInputs,
) -> None:
    leaf_fields = {
        "maximum_carboxylation_rate_25",
        "maximum_electron_transport_rate_25",
        "triose_phosphate_utilization_rate_25",
        "pep_carboxylase_co2_slope_25",
        "leaf_maintenance_respiration_25",
        "absorbed_par",
    }
    for name, value in zip(
        coupling.biochemistry._fields,
        coupling.biochemistry,
        strict=True,
    ):
        expected = (patch_count, 2) if name in leaf_fields else (patch_count,)
        if jnp.shape(value) != expected:
            raise ValueError(f"batched biochemistry field {name} must have shape {expected}")
    for name, value in zip(coupling.controls._fields, coupling.controls, strict=True):
        if jnp.shape(value) != (patch_count,):
            raise ValueError(f"batched control field {name} must have shape ({patch_count},)")
    scalar_plant_fields = coupling.plant_inputs._fields[:-3]
    for name in scalar_plant_fields:
        if jnp.shape(getattr(coupling.plant_inputs, name)) != (patch_count,):
            raise ValueError(f"batched plant input field {name} must have shape ({patch_count},)")
    soil_shapes = tuple(jnp.shape(value) for value in coupling.plant_inputs[-3:])
    if any(len(shape) != 2 or shape[0] != patch_count for shape in soil_shapes):
        raise ValueError("batched plant soil inputs must have shape (patch, soil_level)")
    if len(set(soil_shapes)) != 1:
        raise ValueError("batched plant soil input shapes disagree")
    for name, value in zip(
        coupling.plant_parameters._fields,
        coupling.plant_parameters,
        strict=True,
    ):
        if jnp.shape(value) != (patch_count, 4):
            raise ValueError(f"batched plant parameter {name} must have shape ({patch_count}, 4)")
    if jnp.shape(coupling.initial_plant_state.water_potential_segment) != (patch_count, 4):
        raise ValueError("initial plant water potential must have shape (patch, 4)")
    if jnp.shape(coupling.initial_plant_state.vegetation_transpiration) != (patch_count,):
        raise ValueError("initial vegetation transpiration must have shape (patch,)")


def _initial_hydraulic_iteration_state(
    patch_count: int,
    coupling: CanopyHydraulicCouplingInputs,
    photosynthesis: PrescribedCanopyPhotosynthesis,
) -> CanopyHydraulicIterationState:
    zeros_leaf = jnp.zeros((patch_count, 2), dtype=jnp.float64)
    zeros_iteration = jnp.zeros((patch_count,), dtype=jnp.int32)
    return CanopyHydraulicIterationState(
        plant_state=coupling.initial_plant_state,
        photosynthesis=photosynthesis,
        intracellular_co2_patch_leaf=zeros_leaf,
        hydraulic_stress_patch_leaf=jnp.ones((patch_count, 2), dtype=jnp.float64),
        stomatal_conductance_patch_leaf=zeros_leaf,
        rubisco_limited_photosynthesis_patch_leaf=zeros_leaf,
        electron_transport_limited_photosynthesis_patch_leaf=zeros_leaf,
        product_limited_photosynthesis_patch_leaf=zeros_leaf,
        gross_photosynthesis_patch_leaf=zeros_leaf,
        net_photosynthesis_patch_leaf=zeros_leaf,
        leaf_surface_co2_patch_leaf=zeros_leaf,
        solver_intracellular_co2_patch_leaf=zeros_leaf,
        maximum_carboxylation_rate_patch_leaf=zeros_leaf,
        maximum_electron_transport_rate_patch_leaf=zeros_leaf,
        triose_phosphate_utilization_rate_patch_leaf=zeros_leaf,
        pep_carboxylase_co2_slope_patch_leaf=zeros_leaf,
        leaf_maintenance_respiration_patch_leaf=zeros_leaf,
        electron_transport_rate_patch_leaf=zeros_leaf,
        co2_compensation_point_patch=jnp.zeros((patch_count,), dtype=jnp.float64),
        michaelis_menten_co2_patch=jnp.zeros((patch_count,), dtype=jnp.float64),
        michaelis_menten_o2_patch=jnp.zeros((patch_count,), dtype=jnp.float64),
        canopy_vapor_pressure_deficit_patch=jnp.zeros((patch_count,), dtype=jnp.float64),
        canopy_relative_humidity_patch=jnp.zeros((patch_count,), dtype=jnp.float64),
        molar_conversion_factor_patch=jnp.zeros((patch_count,), dtype=jnp.float64),
        boundary_layer_conductance_patch=jnp.zeros((patch_count,), dtype=jnp.float64),
        outer_iterations_patch=zeros_iteration,
        inner_iterations_patch=zeros_iteration,
        hydraulic_iterations_patch=zeros_iteration,
    )


def _native_leaf_class_stomatal_resistance(
    leaf_area_patch_leaf: jnp.ndarray,
    solved_resistance_patch_leaf: jnp.ndarray,
    boundary_layer_resistance_patch: jnp.ndarray,
) -> jnp.ndarray:

    leaf_area = jnp.asarray(leaf_area_patch_leaf, dtype=jnp.float64)
    resistance = jnp.asarray(solved_resistance_patch_leaf, dtype=jnp.float64)
    if leaf_area.shape != resistance.shape:
        raise ValueError("leaf area and solved resistance arrays must share shape")
    boundary = jnp.asarray(boundary_layer_resistance_patch, dtype=jnp.float64)[..., None]
    has_leaf = leaf_area > jnp.float64(0.0)
    safe_area = jnp.where(has_leaf, leaf_area, jnp.float64(1.0))
    conductance = _divide(safe_area, _add(boundary, resistance))
    effective = _add(_divide(safe_area, conductance), -boundary)
    return jnp.where(
        has_leaf,
        effective,
        jnp.float64(0.0),
    )


def _native_one_layer_canopy_mean(
    layer_value_patch_leaf: jnp.ndarray,
    leaf_area_patch_leaf: jnp.ndarray,
    layer_scale_patch_leaf: jnp.ndarray,
) -> jnp.ndarray:

    layer_value, leaf_area, layer_scale = jnp.broadcast_arrays(
        jnp.asarray(layer_value_patch_leaf, dtype=jnp.float64),
        jnp.asarray(leaf_area_patch_leaf, dtype=jnp.float64),
        jnp.asarray(layer_scale_patch_leaf, dtype=jnp.float64),
    )
    has_leaf = leaf_area > jnp.float64(0.0)
    safe_leaf_area = jnp.where(has_leaf, leaf_area, jnp.float64(1.0))
    canopy_sum = _multiply(
        _multiply(layer_value, safe_leaf_area),
        layer_scale,
    )
    canopy_mean = _divide(canopy_sum, safe_leaf_area)
    return jnp.where(has_leaf, canopy_mean, jnp.float64(0.0))


def _solve_hydraulic_patch(arguments):
    (
        biochemistry,
        controls,
        plant_inputs,
        plant_parameters,
        plant_state,
        leaf_temperature,
        leaf_saturation_vapor_pressure,
        leaf_saturation_specific_humidity,
        canopy_air_specific_humidity,
        boundary_layer_resistance,
    ) = arguments
    dynamic_plant_inputs = plant_inputs._replace(
        leaf_saturation_specific_humidity=leaf_saturation_specific_humidity,
        canopy_air_specific_humidity=canopy_air_specific_humidity,
    )
    canopy_vapor_pressure = fortran_materialize(
        _divide(
            _multiply(
                biochemistry.atmospheric_pressure,
                canopy_air_specific_humidity,
            ),
            jnp.float64(0.622),
        )
    )
    environment = HydraulicPhotosynthesisEnvironment(
        leaf_temperature=leaf_temperature,
        leaf_saturation_vapor_pressure=leaf_saturation_vapor_pressure,
        canopy_vapor_pressure=canopy_vapor_pressure,
        boundary_layer_resistance=boundary_layer_resistance,
        reference_temperature=dynamic_plant_inputs.reference_temperature,
    )
    return solve_hydraulic_photosynthesis_patch(
        biochemistry,
        environment,
        controls,
        dynamic_plant_inputs,
        plant_parameters,
        plant_state,
    )


def _solve_hydraulic_coupling_iteration(
    active_patch: jnp.ndarray,
    coupling: CanopyHydraulicCouplingInputs,
    previous: CanopyHydraulicIterationState,
    vegetation_temperature_patch: jnp.ndarray,
    leaf_saturation_vapor_pressure_patch: jnp.ndarray,
    leaf_saturation_specific_humidity_patch: jnp.ndarray,
    canopy_air_specific_humidity_patch: jnp.ndarray,
    leaf_boundary_resistance_patch: jnp.ndarray,
) -> CanopyHydraulicIterationState:

    patch_arguments = (
        coupling.biochemistry,
        coupling.controls,
        coupling.plant_inputs,
        coupling.plant_parameters,
        previous.plant_state,
        vegetation_temperature_patch,
        leaf_saturation_vapor_pressure_patch,
        leaf_saturation_specific_humidity_patch,
        canopy_air_specific_humidity_patch,
        leaf_boundary_resistance_patch,
    )
    result_shape = eval_shape(
        _solve_hydraulic_patch,
        tree_util.tree_map(
            lambda value: ShapeDtypeStruct(value.shape[1:], value.dtype),
            patch_arguments,
        ),
    )
    empty = tree_util.tree_map(lambda value: jnp.zeros(value.shape, value.dtype), result_shape)

    candidate = active_scalar_map(
        _solve_hydraulic_patch, active_patch, patch_arguments, empty
    )
    canopy_vapor_pressure = fortran_materialize(
        _divide(
            _multiply(
                coupling.biochemistry.atmospheric_pressure,
                canopy_air_specific_humidity_patch,
            ),
            jnp.float64(0.622),
        )
    )
    canopy_relative_humidity = fortran_materialize(
        _divide(
            canopy_vapor_pressure,
            leaf_saturation_vapor_pressure_patch,
        )
    )
    leaf_area_patch_leaf = jnp.stack(
        (
            coupling.plant_inputs.sunlit_leaf_area,
            coupling.plant_inputs.shaded_leaf_area,
        ),
        axis=1,
    )
    stomatal_resistance = _native_leaf_class_stomatal_resistance(
        leaf_area_patch_leaf,
        candidate.stomatal_resistance,
        leaf_boundary_resistance_patch,
    )
    unit_scale = jnp.ones_like(leaf_area_patch_leaf)
    rubisco_limited_photosynthesis = _native_one_layer_canopy_mean(
        candidate.coupled.photosynthesis.rubisco_limited_rate,
        leaf_area_patch_leaf,
        unit_scale,
    )
    electron_transport_limited_photosynthesis = _native_one_layer_canopy_mean(
        candidate.coupled.photosynthesis.electron_transport_limited_rate,
        leaf_area_patch_leaf,
        unit_scale,
    )
    product_limited_photosynthesis = _native_one_layer_canopy_mean(
        candidate.coupled.photosynthesis.product_limited_rate,
        leaf_area_patch_leaf,
        unit_scale,
    )
    gross_photosynthesis = jnp.where(
        leaf_area_patch_leaf > jnp.float64(0.0),
        candidate.coupled.photosynthesis.gross_photosynthesis,
        jnp.float64(0.0),
    )
    stress_scales_respiration = (
        ~jnp.asarray(coupling.controls.crop, dtype=jnp.bool_)
        & jnp.asarray(
            coupling.controls.modify_photosynthesis_and_respiration_for_crop,
            dtype=jnp.bool_,
        )
    )[:, None]
    respiration_scale = jnp.where(
        stress_scales_respiration,
        candidate.coupled.hydraulic_stress,
        unit_scale,
    )
    leaf_maintenance_respiration = _native_one_layer_canopy_mean(
        candidate.preparation.inputs.leaf_maintenance_respiration,
        leaf_area_patch_leaf,
        respiration_scale,
    )
    candidate_state = CanopyHydraulicIterationState(
        plant_state=candidate.coupled.plant_state,
        photosynthesis=PrescribedCanopyPhotosynthesis(
            sunlit_stomatal_resistance_patch=stomatal_resistance[:, 0],
            shaded_stomatal_resistance_patch=stomatal_resistance[:, 1],
            transpiration_wetness_patch=candidate.transpiration_wetness,
            vegetation_transpiration_patch=(candidate.coupled.plant_state.vegetation_transpiration),
        ),
        intracellular_co2_patch_leaf=candidate.coupled.intracellular_co2,
        hydraulic_stress_patch_leaf=candidate.coupled.hydraulic_stress,
        stomatal_conductance_patch_leaf=candidate.coupled.stomatal_conductance,
        rubisco_limited_photosynthesis_patch_leaf=rubisco_limited_photosynthesis,
        electron_transport_limited_photosynthesis_patch_leaf=(
            electron_transport_limited_photosynthesis
        ),
        product_limited_photosynthesis_patch_leaf=product_limited_photosynthesis,
        gross_photosynthesis_patch_leaf=gross_photosynthesis,
        net_photosynthesis_patch_leaf=candidate.coupled.photosynthesis.net_photosynthesis,
        leaf_surface_co2_patch_leaf=candidate.coupled.photosynthesis.leaf_surface_co2,
        solver_intracellular_co2_patch_leaf=candidate.solver_intracellular_co2,
        maximum_carboxylation_rate_patch_leaf=(
            candidate.preparation.inputs.maximum_carboxylation_rate
        ),
        maximum_electron_transport_rate_patch_leaf=(
            candidate.preparation.maximum_electron_transport_rate
        ),
        triose_phosphate_utilization_rate_patch_leaf=(
            candidate.preparation.inputs.triose_phosphate_utilization_rate
        ),
        pep_carboxylase_co2_slope_patch_leaf=(
            candidate.preparation.inputs.pep_carboxylase_co2_slope
        ),
        leaf_maintenance_respiration_patch_leaf=leaf_maintenance_respiration,
        electron_transport_rate_patch_leaf=(candidate.preparation.inputs.electron_transport_rate),
        co2_compensation_point_patch=(candidate.preparation.inputs.co2_compensation_point),
        michaelis_menten_co2_patch=(candidate.preparation.inputs.michaelis_menten_co2),
        michaelis_menten_o2_patch=(candidate.preparation.inputs.michaelis_menten_o2),
        canopy_vapor_pressure_deficit_patch=(
            candidate.preparation.inputs.canopy_vapor_pressure_deficit
        ),
        canopy_relative_humidity_patch=canopy_relative_humidity,
        molar_conversion_factor_patch=candidate.preparation.molar_conversion_factor,
        boundary_layer_conductance_patch=(candidate.preparation.inputs.boundary_layer_conductance),
        outer_iterations_patch=candidate.coupled.outer_iterations,
        inner_iterations_patch=candidate.coupled.inner_iterations,
        hydraulic_iterations_patch=candidate.hydraulic_iterations,
    )

    def materialize_subroutine_output(value: jnp.ndarray) -> jnp.ndarray:

        if jnp.issubdtype(value.dtype, jnp.floating):
            return fortran_materialize(value)
        return lax.optimization_barrier(value)

    candidate_state = tree_util.tree_map(
        materialize_subroutine_output,
        candidate_state,
    )

    def update_patch(candidate_value: jnp.ndarray, previous_value: jnp.ndarray) -> jnp.ndarray:
        mask = jnp.reshape(
            active_patch,
            active_patch.shape + (1,) * (candidate_value.ndim - 1),
        )
        return jnp.where(mask, candidate_value, previous_value)

    return tree_util.tree_map(update_patch, candidate_state, previous)


def _ground_emitted_longwave_temperature_fourth(inputs: CanopyEnergyInputs) -> jnp.ndarray:
    exposed_soil = _add(
        jnp.float64(1.0),
        -inputs.snow_fraction_patch,
        -inputs.surface_water_fraction_patch,
    )
    return _add(
        _multiply(inputs.snow_fraction_patch, _fourth_power(inputs.snow_surface_temperature_patch)),
        _multiply(exposed_soil, _fourth_power(inputs.soil_surface_temperature_patch)),
        _multiply(
            inputs.surface_water_fraction_patch,
            _fourth_power(inputs.surface_water_temperature_patch),
        ),
    )


def _canopy_energy_balance_with_prescribed_photosynthesis(
    inputs: CanopyEnergyInputs,
    photosynthesis: PrescribedCanopyPhotosynthesis,
    initial_state: CanopyEnergyInitialState,
    parameters: CanopyEnergyParameters,
    *,
    capture_history: bool,
    hydraulic_coupling: CanopyHydraulicCouplingInputs | None = None,
) -> tuple[
    CanopyEnergyResult,
    CanopyEnergyIterationState | _CanopyHydraulicScanState | None,
    CanopyHydraulicIterationState | None,
]:

    _validate_shapes(inputs, photosynthesis, initial_state)
    patch_count = inputs.active_patch.shape[0]
    if hydraulic_coupling is not None:
        _validate_hydraulic_coupling(patch_count, hydraulic_coupling)
    original_active = jnp.asarray(inputs.active_patch, dtype=jnp.bool_)
    safe = _safe_inputs(inputs)
    photo = _safe_photosynthesis(original_active, photosynthesis)
    parameters = CanopyEnergyParameters(
        step_seconds=jnp.asarray(parameters.step_seconds, dtype=jnp.float64),
        maximum_stable_zeta=jnp.asarray(parameters.maximum_stable_zeta, dtype=jnp.float64),
        use_undercanopy_stability=jnp.asarray(
            parameters.use_undercanopy_stability, dtype=jnp.bool_
        ),
        use_hydraulic_stress=jnp.asarray(parameters.use_hydraulic_stress, dtype=jnp.bool_),
    )

    initial_temperature = jnp.asarray(
        initial_state.vegetation_temperature_patch,
        dtype=jnp.float64,
    )
    initial_displacement = jnp.asarray(initial_state.displacement_height_patch, dtype=jnp.float64)
    initial_roughness = jnp.asarray(
        initial_state.vegetation_momentum_roughness_patch,
        dtype=jnp.float64,
    )
    total_area = _add(safe.exposed_leaf_area_index_patch, safe.exposed_stem_area_index_patch)
    canopy_attenuation = _scalar_exp(-total_area)
    limited_area = jnp.minimum(total_area, jnp.float64(CRITICAL_TOTAL_LEAF_STEM_AREA))
    aerodynamic_fraction = _divide(
        _add(jnp.float64(1.0), -_scalar_exp(-limited_area)),
        _add(
            jnp.float64(1.0),
            -_scalar_exp(jnp.full_like(limited_area, -CRITICAL_TOTAL_LEAF_STEM_AREA)),
        ),
    )
    displacement_candidate = _multiply(aerodynamic_fraction, initial_displacement)
    roughness_candidate = strict_exp(
        _add(
            _multiply(aerodynamic_fraction, _scalar_log(initial_roughness)),
            _multiply(
                _add(jnp.float64(1.0), -aerodynamic_fraction),
                _scalar_log(safe.ground_momentum_roughness_patch),
            ),
        )
    )
    displacement = _update(original_active, displacement_candidate, initial_displacement)
    momentum_roughness = _update(original_active, roughness_candidate, initial_roughness)
    sensible_roughness = momentum_roughness
    latent_roughness = momentum_roughness

    one_minus_emissivity = _add(jnp.float64(1.0), -safe.vegetation_emissivity_patch)
    one_minus_ground_emissivity = _add(jnp.float64(1.0), -safe.ground_emissivity_patch)
    atmospheric_longwave = _multiply(
        _multiply(
            safe.vegetation_emissivity_patch,
            _add(
                jnp.float64(1.0),
                _multiply(one_minus_emissivity, one_minus_ground_emissivity),
            ),
        ),
        safe.downward_longwave_patch,
    )
    vegetation_longwave = -_multiply(
        _multiply(
            _add(
                jnp.float64(2.0),
                -_multiply(safe.vegetation_emissivity_patch, one_minus_ground_emissivity),
            ),
            safe.vegetation_emissivity_patch,
        ),
        jnp.float64(STEFAN_BOLTZMANN),
    )
    ground_longwave = _multiply(
        _multiply(safe.vegetation_emissivity_patch, safe.ground_emissivity_patch),
        jnp.float64(STEFAN_BOLTZMANN),
    )

    safe_initial_temperature = jnp.where(original_active, initial_temperature, jnp.float64(280.0))
    saturation = qsat(safe_initial_temperature, safe.atmospheric_pressure_patch)
    canopy_air_temperature = _multiply(
        _add(safe.ground_temperature_patch, safe.reference_temperature_patch),
        jnp.float64(0.5),
    )
    canopy_air_humidity = _multiply(
        _add(
            safe.atmospheric_specific_humidity_patch,
            safe.ground_specific_humidity_patch,
        ),
        jnp.float64(0.5),
    )
    reference_wind = jnp.maximum(
        jnp.float64(1.0),
        jnp.sqrt(_add(_square(safe.eastward_wind_patch), _square(safe.northward_wind_patch))),
    )
    temperature_difference = _add(safe.reference_temperature_patch, -canopy_air_temperature)
    humidity_difference = _add(safe.atmospheric_specific_humidity_patch, -canopy_air_humidity)
    ground_humidity_difference = _add(
        safe.ground_specific_humidity_patch,
        -canopy_air_humidity,
    )
    virtual_temperature_difference = _add(
        _multiply(
            temperature_difference,
            _add(
                jnp.float64(1.0),
                _multiply(jnp.float64(0.61), safe.atmospheric_specific_humidity_patch),
            ),
        ),
        _multiply(
            _multiply(
                jnp.float64(0.61),
                safe.atmospheric_potential_temperature_patch,
            ),
            humidity_difference,
        ),
    )
    height = _add(safe.forcing_height_wind_patch, -displacement)
    virtual_potential_temperature = _multiply(
        safe.atmospheric_potential_temperature_patch,
        _add(
            jnp.float64(1.0),
            _multiply(jnp.float64(0.61), safe.atmospheric_specific_humidity_patch),
        ),
    )
    initialized = monin_obukhov_initialize(
        reference_wind,
        virtual_potential_temperature,
        virtual_temperature_difference,
        height,
        momentum_roughness,
        parameters.maximum_stable_zeta,
    )
    zeros = jnp.zeros_like(safe_initial_temperature)
    zero_int = jnp.zeros_like(original_active, dtype=jnp.int32)
    initial_transpiration = (
        photosynthesis.vegetation_transpiration_patch
        if hydraulic_coupling is None
        else hydraulic_coupling.initial_plant_state.vegetation_transpiration
    )
    initial_iteration = CanopyEnergyIterationState(
        active_patch=original_active,
        vegetation_temperature_patch=initial_temperature,
        canopy_air_temperature_patch=canopy_air_temperature,
        canopy_air_specific_humidity_patch=canopy_air_humidity,
        ground_humidity_difference_patch=ground_humidity_difference,
        monin_obukhov_length_patch=initialized.monin_obukhov_length,
        previous_monin_obukhov_length_patch=zeros,
        stability_wind_speed_patch=initialized.stability_wind_speed,
        stability_sign_changes_patch=zero_int,
        leaf_temperature_change_patch=zeros,
        previous_leaf_temperature_change_patch=zeros,
        leaf_latent_heat_flux_patch=zeros,
        previous_leaf_latent_heat_flux_patch=zeros,
        leaf_latent_heat_flux_change_patch=zeros,
        previous_vegetation_temperature_patch=initial_temperature,
        vegetation_temperature_step_patch=zeros,
        leaf_saturation_vapor_pressure_patch=saturation.vapor_pressure,
        leaf_saturation_specific_humidity_patch=saturation.specific_humidity,
        leaf_saturation_specific_humidity_derivative_patch=(
            saturation.specific_humidity_temperature_derivative
        ),
        vegetation_sensible_heat_flux_patch=zeros,
        vegetation_evaporation_patch=zeros,
        vegetation_transpiration_patch=jnp.asarray(
            initial_transpiration,
            dtype=jnp.float64,
        ),
        friction_velocity_value_patch=zeros,
        friction_stability_wind_input_patch=zeros,
        friction_momentum_numerator_patch=zeros,
        friction_momentum_denominator_patch=zeros,
        friction_momentum_zeta_patch=zeros,
        friction_temperature_profile_patch=zeros,
        friction_humidity_profile_patch=zeros,
        friction_temperature_profile_2m_patch=zeros,
        friction_humidity_profile_2m_patch=zeros,
        within_canopy_wind_patch=zeros,
        bare_soil_drag_patch=zeros,
        soil_drag_patch=zeros,
        leaf_boundary_resistance_patch=zeros,
        momentum_conductance_patch=zeros,
        momentum_resistance_patch=zeros,
        atmospheric_thermal_resistance_patch=zeros,
        ground_thermal_resistance_patch=zeros,
        atmospheric_moisture_resistance_patch=zeros,
        ground_moisture_resistance_patch=zeros,
        ground_sensible_conductance_patch=zeros,
        ground_latent_conductance_patch=zeros,
        normalized_leaf_sensible_conductance_patch=zeros,
        normalized_ground_sensible_conductance_patch=zeros,
        normalized_air_sensible_conductance_patch=zeros,
        normalized_ground_air_sensible_conductance_patch=zeros,
        normalized_air_leaf_sensible_conductance_patch=zeros,
        normalized_leaf_latent_conductance_patch=zeros,
        normalized_ground_latent_conductance_patch=zeros,
        normalized_air_latent_conductance_patch=zeros,
        normalized_ground_air_latent_conductance_patch=zeros,
        normalized_air_leaf_latent_conductance_patch=zeros,
        canopy_methane_conductance_patch=zeros,
        ground_methane_conductance_patch=zeros,
        leaf_sensible_linearization_patch=zeros,
        leaf_latent_humidity_gradient_patch=zeros,
        sensible_temperature_derivative_patch=zeros,
        latent_temperature_derivative_patch=zeros,
        temperature_scale_patch=zeros,
        humidity_scale_patch=zeros,
        virtual_temperature_scale_patch=zeros,
        stability_zeta_patch=zeros,
        convective_argument_patch=zeros,
        convective_velocity_patch=zeros,
        stability_wind_squared_sum_patch=zeros,
        energy_balance_correction_patch=zeros,
        iteration_count_patch=zero_int,
        friction_velocity=initial_state.friction_velocity,
        canopy_water_change_patch=(zeros if _coupled_water_balance_derivatives.get() else None),
        canopy_water_remaining_patch=(
            safe.canopy_water_patch if _coupled_water_balance_derivatives.get() else None
        ),
    )
    ground_longwave_temperature_fourth = _ground_emitted_longwave_temperature_fourth(safe)

    initial_hydraulic = (
        None
        if hydraulic_coupling is None
        else _initial_hydraulic_iteration_state(
            patch_count,
            hydraulic_coupling,
            photo,
        )
    )
    initial_scan_state = (
        initial_iteration
        if initial_hydraulic is None
        else _CanopyHydraulicScanState(initial_iteration, initial_hydraulic)
    )

    def iteration_step(scan_state, zero_based_iteration: jnp.ndarray):
        if hydraulic_coupling is None:
            state = scan_state
            hydraulic_state = None
        else:
            state = scan_state.energy
            hydraulic_state = scan_state.hydraulic
        active = state.active_patch
        friction = friction_velocity(
            active,
            displacement,
            momentum_roughness,
            sensible_roughness,
            latent_roughness,
            state.monin_obukhov_length_patch,
            zero_based_iteration + jnp.int32(1),
            reference_wind,
            state.stability_wind_speed_patch,
            safe.forcing_height_wind_patch,
            safe.forcing_height_temperature_patch,
            safe.forcing_height_humidity_patch,
            state.friction_velocity,
        )
        friction_velocity_value = fortran_rounding_barrier(
            jnp.where(
                active,
                friction.friction_velocity_patch,
                jnp.float64(0.1),
            )
        )
        temperature_profile = fortran_rounding_barrier(
            jnp.where(
                active,
                friction.temperature_profile_patch,
                jnp.float64(0.1),
            )
        )
        humidity_profile = fortran_rounding_barrier(
            jnp.where(
                active,
                friction.humidity_profile_patch,
                jnp.float64(0.1),
            )
        )
        stability_wind = fortran_rounding_barrier(
            jnp.where(
                active,
                state.stability_wind_speed_patch,
                jnp.float64(1.0),
            )
        )
        previous_temperature = jnp.where(
            active,
            state.vegetation_temperature_patch,
            jnp.float64(280.0),
        )
        previous_change = state.leaf_temperature_change_patch

        friction_velocity_squared = fortran_rounding_barrier(_square(friction_velocity_value))
        momentum_conductance = _divide(
            friction_velocity_squared,
            stability_wind,
        )
        momentum_resistance = _divide(
            jnp.float64(1.0),
            fortran_rounding_barrier(momentum_conductance),
        )
        atmospheric_thermal_resistance = _divide(
            jnp.float64(1.0),
            _multiply(temperature_profile, friction_velocity_value),
        )
        atmospheric_moisture_resistance = _divide(
            jnp.float64(1.0),
            _multiply(humidity_profile, friction_velocity_value),
        )
        within_canopy_wind = _multiply(
            stability_wind,
            jnp.sqrt(
                _divide(
                    jnp.float64(1.0),
                    _multiply(momentum_resistance, stability_wind),
                )
            ),
        )
        transfer_coefficient = _divide(
            jnp.float64(0.01),
            _multiply(
                jnp.sqrt(within_canopy_wind),
                jnp.sqrt(safe.leaf_dimension_patch),
            ),
        )
        leaf_boundary_resistance = _divide(
            jnp.float64(1.0),
            _multiply(transfer_coefficient, within_canopy_wind),
        )
        if hydraulic_coupling is None:
            iteration_hydraulic = None
            iteration_photo = photo
        else:
            iteration_hydraulic = _solve_hydraulic_coupling_iteration(
                active,
                hydraulic_coupling,
                hydraulic_state,
                state.vegetation_temperature_patch,
                state.leaf_saturation_vapor_pressure_patch,
                state.leaf_saturation_specific_humidity_patch,
                state.canopy_air_specific_humidity_patch,
                leaf_boundary_resistance,
            )
            iteration_photo = iteration_hydraulic.photosynthesis
        bare_soil_drag = _divide(
            jnp.float64(VON_KARMAN),
            _multiply(
                jnp.float64(0.13),
                _scalar_power(
                    _divide(
                        _multiply(safe.ground_momentum_roughness_patch, within_canopy_wind),
                        jnp.float64(MOLECULAR_VISCOSITY_AIR),
                    ),
                    0.45,
                ),
            ),
        )
        undercanopy_richardson = _divide(
            _multiply(
                _multiply(jnp.float64(GRAVITY), safe.canopy_top_height_patch),
                _add(state.canopy_air_temperature_patch, -safe.ground_temperature_patch),
            ),
            _multiply(state.canopy_air_temperature_patch, _square(within_canopy_wind)),
        )
        stable_drag = _divide(
            jnp.float64(SOIL_DRAG_COEFFICIENT),
            _add(
                jnp.float64(1.0),
                _multiply(
                    jnp.float64(UNDERCANOPY_STABILITY_COEFFICIENT),
                    jnp.minimum(undercanopy_richardson, jnp.float64(10.0)),
                ),
            ),
        )
        use_stable_drag = parameters.use_undercanopy_stability & (
            state.canopy_air_temperature_patch > safe.ground_temperature_patch
        )
        dense_canopy_drag = jnp.where(
            use_stable_drag,
            stable_drag,
            jnp.float64(SOIL_DRAG_COEFFICIENT),
        )
        soil_drag = _add(
            fortran_rounding_barrier(_multiply(bare_soil_drag, canopy_attenuation)),
            fortran_rounding_barrier(
                _multiply(
                    dense_canopy_drag,
                    _add(jnp.float64(1.0), -canopy_attenuation),
                )
            ),
        )
        ground_thermal_resistance = _divide(
            jnp.float64(1.0),
            _multiply(soil_drag, within_canopy_wind),
        )
        ground_moisture_resistance = ground_thermal_resistance
        ground_methane_conductance = _divide(
            jnp.float64(1.0),
            _add(atmospheric_moisture_resistance, ground_moisture_resistance),
        )

        air_sensible_conductance = _divide(
            jnp.float64(1.0),
            atmospheric_thermal_resistance,
        )
        leaf_sensible_conductance = _divide(total_area, leaf_boundary_resistance)
        ground_sensible_conductance = _divide(
            jnp.float64(1.0),
            ground_thermal_resistance,
        )
        inverse_sensible_total = _divide(
            jnp.float64(1.0),
            _add(
                air_sensible_conductance,
                leaf_sensible_conductance,
                ground_sensible_conductance,
            ),
        )
        normalized_leaf_sensible = fortran_materialize(
            _multiply(leaf_sensible_conductance, inverse_sensible_total)
        )
        normalized_ground_sensible = fortran_materialize(
            _multiply(ground_sensible_conductance, inverse_sensible_total)
        )
        normalized_air_sensible = fortran_materialize(
            _multiply(air_sensible_conductance, inverse_sensible_total)
        )
        normalized_ground_air_sensible = _add(
            fortran_rounding_barrier(normalized_air_sensible),
            fortran_rounding_barrier(normalized_ground_sensible),
        )
        normalized_air_leaf_sensible = _add(
            fortran_rounding_barrier(normalized_air_sensible),
            fortran_rounding_barrier(normalized_leaf_sensible),
        )

        dry_evaporation_fraction = jnp.where(
            safe.dry_foliage_fraction_patch > jnp.float64(0.0),
            _divide(
                _multiply(
                    _multiply(
                        safe.dry_foliage_fraction_patch,
                        leaf_boundary_resistance,
                    ),
                    _add(
                        _divide(
                            safe.sunlit_leaf_area_index_patch,
                            _add(
                                leaf_boundary_resistance,
                                iteration_photo.sunlit_stomatal_resistance_patch,
                            ),
                        ),
                        _divide(
                            safe.shaded_leaf_area_index_patch,
                            _add(
                                leaf_boundary_resistance,
                                iteration_photo.shaded_stomatal_resistance_patch,
                            ),
                        ),
                    ),
                ),
                jnp.where(safe.dry_foliage_fraction_patch > 0.,
                          safe.exposed_leaf_area_index_patch, jnp.float64(1.0)),
            ),
            jnp.float64(0.0),
        )
        canopy_methane_conductance = _divide(
            _add(
                _divide(
                    safe.sunlit_leaf_area_index_patch,
                    _add(
                        leaf_boundary_resistance,
                        iteration_photo.sunlit_stomatal_resistance_patch,
                    ),
                ),
                _divide(
                    safe.shaded_leaf_area_index_patch,
                    _add(
                        leaf_boundary_resistance,
                        iteration_photo.shaded_stomatal_resistance_patch,
                    ),
                ),
            ),
            jnp.maximum(safe.exposed_leaf_area_index_patch, jnp.float64(0.01)),
        )
        potential_leaf_evaporation = _multiply(
            _multiply(safe.air_density_patch, leaf_sensible_conductance),
            _add(
                state.leaf_saturation_specific_humidity_patch,
                -state.canopy_air_specific_humidity_patch,
            ),
        )
        positive_potential = potential_leaf_evaporation > jnp.float64(0.0)
        unstressed = iteration_photo.transpiration_wetness_patch > jnp.float64(0.0)
        available_fraction = jnp.where(
            unstressed,
            _add(dry_evaporation_fraction, safe.wet_foliage_fraction_patch),
            safe.wet_foliage_fraction_patch,
        )
        maximum_fraction = _divide(
            _add(
                iteration_photo.vegetation_transpiration_patch,
                _divide(safe.canopy_water_patch, parameters.step_seconds),
            ),
            jnp.where(positive_potential, potential_leaf_evaporation, jnp.float64(1.0)),
        )
        evaporation_fraction = jnp.where(
            positive_potential,
            jnp.minimum(available_fraction, maximum_fraction),
            jnp.float64(1.0),
        )
        nonhydraulic_transpiration = jnp.where(
            positive_potential & unstressed,
            _multiply(potential_leaf_evaporation, dry_evaporation_fraction),
            jnp.float64(0.0),
        )
        transpiration = jnp.where(
            parameters.use_hydraulic_stress,
            iteration_photo.vegetation_transpiration_patch,
            nonhydraulic_transpiration,
        )

        air_latent_conductance = _divide(
            safe.vegetation_fraction_without_snow_patch,
            atmospheric_moisture_resistance,
        )
        leaf_latent_conductance = _multiply(
            _multiply(
                _divide(
                    _multiply(safe.vegetation_fraction_without_snow_patch, total_area),
                    leaf_boundary_resistance,
                ),
                evaporation_fraction,
            ),
            jnp.float64(1.0),
        )
        exposed_litter_area = _multiply(
            jnp.float64(DRY_LITTER_AREA_INDEX),
            _add(
                jnp.float64(1.0),
                -jnp.minimum(
                    _divide(safe.snow_depth_patch, jnp.float64(DRY_LITTER_THICKNESS)),
                    jnp.float64(1.0),
                ),
            ),
        )
        dry_litter_resistance = _divide(
            _add(jnp.float64(1.0), -strict_exp(-exposed_litter_area)),
            _multiply(jnp.float64(SOIL_DRAG_COEFFICIENT), within_canopy_wind),
        )
        dew = state.ground_humidity_difference_patch < jnp.float64(0.0)
        ground_latent_conductance = jnp.where(
            dew,
            _divide(
                safe.vegetation_fraction_without_snow_patch,
                _add(ground_moisture_resistance, dry_litter_resistance),
            ),
            _divide(
                safe.vegetation_fraction_without_snow_patch,
                _add(
                    ground_moisture_resistance,
                    safe.soil_evaporation_resistance_patch,
                ),
            ),
        )
        inverse_latent_total = _divide(
            jnp.float64(1.0),
            _add(
                air_latent_conductance,
                leaf_latent_conductance,
                ground_latent_conductance,
            ),
        )
        normalized_ground_latent = fortran_materialize(
            _multiply(ground_latent_conductance, inverse_latent_total)
        )
        normalized_leaf_latent = fortran_materialize(
            _multiply(leaf_latent_conductance, inverse_latent_total)
        )
        normalized_air_latent = fortran_materialize(
            _multiply(air_latent_conductance, inverse_latent_total)
        )
        normalized_ground_air_latent = _add(
            fortran_rounding_barrier(normalized_air_latent),
            fortran_rounding_barrier(normalized_ground_latent),
        )
        normalized_air_leaf_latent = _add(
            fortran_rounding_barrier(normalized_air_latent),
            fortran_rounding_barrier(normalized_leaf_latent),
        )
        sensible_derivative = _multiply(
            _multiply(safe.air_density_patch, jnp.float64(SPECIFIC_HEAT_DRY_AIR)),
            leaf_sensible_conductance,
        )
        latent_derivative = _multiply(
            _multiply(jnp.float64(LATENT_HEAT_VAPORIZATION), safe.air_density_patch),
            leaf_latent_conductance,
        )
        leaf_sensible = _multiply(
            sensible_derivative,
            _add(
                _multiply(normalized_ground_air_sensible, previous_temperature),
                -_multiply(normalized_ground_sensible, safe.ground_temperature_patch),
                -_multiply(normalized_air_sensible, safe.reference_temperature_patch),
            ),
        )
        leaf_latent_humidity_gradient = _add(
            _multiply(
                normalized_ground_air_latent,
                state.leaf_saturation_specific_humidity_patch,
            ),
            -_multiply(
                normalized_ground_latent,
                safe.ground_specific_humidity_patch,
            ),
            -_multiply(
                normalized_air_latent,
                safe.atmospheric_specific_humidity_patch,
            ),
        )
        leaf_latent = _multiply(
            latent_derivative,
            leaf_latent_humidity_gradient,
        )
        sign_changed = _multiply(
            leaf_latent,
            state.previous_leaf_latent_heat_flux_patch,
        ) < jnp.float64(0.0)
        unbounded_leaf_latent = leaf_latent
        leaf_latent = jnp.where(sign_changed, _multiply(jnp.float64(0.1), leaf_latent), leaf_latent)
        latent_sign_correction = jnp.where(
            sign_changed,
            _add(leaf_latent, -unbounded_leaf_latent),
            jnp.float64(0.0),
        )
        numerator = _add(
            safe.absorbed_shortwave_vegetation_patch,
            atmospheric_longwave,
            _multiply(vegetation_longwave, _fourth_power(previous_temperature)),
            _multiply(ground_longwave, ground_longwave_temperature_fourth),
            -leaf_sensible,
            -leaf_latent,
        )
        denominator = _add(
            -_multiply(
                _multiply(jnp.float64(4.0), vegetation_longwave),
                _cube(previous_temperature),
            ),
            _multiply(sensible_derivative, normalized_ground_air_sensible),
            _multiply(
                _multiply(latent_derivative, normalized_ground_air_latent),
                state.leaf_saturation_specific_humidity_derivative_patch,
            ),
        )
        unbounded_temperature_step = _divide(numerator, denominator)
        leaf_temperature_change = jnp.abs(unbounded_temperature_step)
        temperature_step = _bounded_leaf_temperature_step(
            unbounded_temperature_step, leaf_temperature_change,
        )
        new_temperature = _add(previous_temperature, temperature_step)
        limited = leaf_temperature_change > jnp.float64(MAXIMUM_LEAF_TEMPERATURE_STEP)
        truncation_correction = jnp.where(
            limited,
            _add(
                safe.absorbed_shortwave_vegetation_patch,
                atmospheric_longwave,
                _multiply(
                    _multiply(vegetation_longwave, _cube(previous_temperature)),
                    _add(previous_temperature, _multiply(jnp.float64(4.0), temperature_step)),
                ),
                _multiply(ground_longwave, ground_longwave_temperature_fourth),
                -_add(
                    leaf_sensible,
                    _multiply(
                        _multiply(sensible_derivative, normalized_ground_air_sensible),
                        temperature_step,
                    ),
                ),
                -_add(
                    leaf_latent,
                    _multiply(
                        _multiply(
                            _multiply(latent_derivative, normalized_ground_air_latent),
                            state.leaf_saturation_specific_humidity_derivative_patch,
                        ),
                        temperature_step,
                    ),
                ),
            ),
            jnp.float64(0.0),
        )
        new_potential_evaporation = _multiply(
            _multiply(safe.air_density_patch, leaf_sensible_conductance),
            _add(
                _multiply(
                    normalized_ground_air_latent,
                    _add(
                        state.leaf_saturation_specific_humidity_patch,
                        _multiply(
                            state.leaf_saturation_specific_humidity_derivative_patch,
                            temperature_step,
                        ),
                    ),
                ),
                -_multiply(
                    normalized_ground_latent,
                    safe.ground_specific_humidity_patch,
                ),
                -_multiply(
                    normalized_air_latent,
                    safe.atmospheric_specific_humidity_patch,
                ),
            ),
        )
        vegetation_evaporation = _multiply(evaporation_fraction, new_potential_evaporation)
        canopy_evaporation_supply = _divide(safe.canopy_water_patch, parameters.step_seconds)
        excess_evaporation = jnp.maximum(
            jnp.float64(0.0),
            _add(vegetation_evaporation, -transpiration, -canopy_evaporation_supply),
        )
        if _coupled_water_balance_derivatives.get():
            (vegetation_evaporation, canopy_water_change, canopy_water_remaining,
             excess_evaporation) = (
                _coupled_canopy_evaporation_balance(
                    safe.canopy_water_patch, iteration_photo.vegetation_transpiration_patch,
                    nonhydraulic_transpiration, available_fraction,
                    potential_leaf_evaporation, new_potential_evaporation,
                    jnp.asarray(parameters.step_seconds, dtype=jnp.float64),
                    parameters.use_hydraulic_stress,
                )
            )
        else:
            evaporation_limit = _add(transpiration, canopy_evaporation_supply)
            vegetation_evaporation = jnp.minimum(vegetation_evaporation, evaporation_limit)
        vegetation_sensible = _add(
            leaf_sensible,
            _multiply(
                _multiply(sensible_derivative, normalized_ground_air_sensible),
                temperature_step,
            ),
            truncation_correction,
            latent_sign_correction,
            _multiply(jnp.float64(LATENT_HEAT_VAPORIZATION), excess_evaporation),
        )
        new_saturation = qsat(new_temperature, safe.atmospheric_pressure_patch)
        new_canopy_air_temperature = _add(
            _multiply(normalized_ground_sensible, safe.ground_temperature_patch),
            _multiply(normalized_air_sensible, safe.reference_temperature_patch),
            _multiply(normalized_leaf_sensible, new_temperature),
        )
        new_canopy_air_humidity = _add(
            _multiply(normalized_leaf_latent, new_saturation.specific_humidity),
            _multiply(normalized_ground_latent, safe.ground_specific_humidity_patch),
            _multiply(normalized_air_latent, safe.atmospheric_specific_humidity_patch),
        )
        new_temperature_difference = _add(
            safe.reference_temperature_patch,
            -new_canopy_air_temperature,
        )
        new_humidity_difference = _add(
            safe.atmospheric_specific_humidity_patch,
            -new_canopy_air_humidity,
        )
        new_ground_humidity_difference = _add(
            _multiply(
                normalized_air_leaf_latent,
                safe.ground_specific_humidity_patch,
            ),
            -_multiply(normalized_leaf_latent, new_saturation.specific_humidity),
            -_multiply(
                normalized_air_latent,
                safe.atmospheric_specific_humidity_patch,
            ),
        )
        temperature_scale = _multiply(temperature_profile, new_temperature_difference)
        humidity_scale = _multiply(humidity_profile, new_humidity_difference)
        virtual_temperature_term1 = fortran_rounding_barrier(
            _multiply(
                fortran_rounding_barrier(temperature_scale),
                _add(
                    jnp.float64(1.0),
                    _multiply(jnp.float64(0.61), safe.atmospheric_specific_humidity_patch),
                ),
            )
        )
        virtual_temperature_term2 = fortran_rounding_barrier(
            _multiply(
                _multiply(
                    jnp.float64(0.61),
                    safe.atmospheric_potential_temperature_patch,
                ),
                fortran_rounding_barrier(humidity_scale),
            )
        )
        virtual_temperature_scale = _add(
            virtual_temperature_term1,
            virtual_temperature_term2,
        )
        zeta = _divide(
            _multiply(
                _multiply(_multiply(height, jnp.float64(VON_KARMAN)), jnp.float64(GRAVITY)),
                virtual_temperature_scale,
            ),
            _multiply(_square(friction_velocity_value), virtual_potential_temperature),
        )
        stable = zeta >= jnp.float64(0.0)
        bounded_zeta = jnp.where(
            stable,
            jnp.minimum(
                parameters.maximum_stable_zeta,
                jnp.maximum(zeta, jnp.float64(0.01)),
            ),
            jnp.maximum(jnp.float64(-100.0), jnp.minimum(zeta, jnp.float64(-0.01))),
        )
        convective_argument = -_divide(
            _multiply(
                _multiply(
                    _multiply(jnp.float64(GRAVITY), friction_velocity_value),
                    virtual_temperature_scale,
                ),
                jnp.float64(1000.0),
            ),
            virtual_potential_temperature,
        )
        convective_velocity = _scalar_power(
            jnp.where(stable, jnp.float64(1.0), convective_argument),
            0.333,
        )
        stability_wind_squared_sum = _add(
            fortran_rounding_barrier(_square(reference_wind)),
            fortran_rounding_barrier(_square(convective_velocity)),
        )
        new_stability_wind = jnp.where(
            stable,
            jnp.maximum(reference_wind, jnp.float64(0.1)),
            jnp.sqrt(stability_wind_squared_sum),
        )
        new_monin_length = _divide(height, bounded_zeta)
        sign_change_count = state.stability_sign_changes_patch + (
            _multiply(state.previous_monin_obukhov_length_patch, new_monin_length)
            < jnp.float64(0.0)
        ).astype(jnp.int32)
        new_monin_length = jnp.where(
            sign_change_count >= jnp.int32(4),
            _divide(height, jnp.float64(-0.01)),
            new_monin_length,
        )

        iteration_number = zero_based_iteration + jnp.int32(1)
        latent_change = jnp.abs(_add(leaf_latent, -state.previous_leaf_latent_heat_flux_patch))
        maximum_two_step_change = jnp.maximum(leaf_temperature_change, previous_change)
        may_converge = iteration_number > jnp.int32(2)
        converged = (
            may_converge
            & (maximum_two_step_change < jnp.float64(LEAF_TEMPERATURE_TOLERANCE))
            & (latent_change < jnp.float64(LEAF_LATENT_HEAT_TOLERANCE))
        )
        next_active = active & ~converged
        previous_leaf_latent = jnp.where(
            active & may_converge,
            leaf_latent,
            state.previous_leaf_latent_heat_flux_patch,
        )

        def update(candidate: jnp.ndarray, previous: jnp.ndarray) -> jnp.ndarray:
            return _update(active, candidate, previous)

        next_state = CanopyEnergyIterationState(
            active_patch=next_active,
            vegetation_temperature_patch=update(
                new_temperature,
                state.vegetation_temperature_patch,
            ),
            canopy_air_temperature_patch=update(
                new_canopy_air_temperature,
                state.canopy_air_temperature_patch,
            ),
            canopy_air_specific_humidity_patch=update(
                new_canopy_air_humidity,
                state.canopy_air_specific_humidity_patch,
            ),
            ground_humidity_difference_patch=update(
                new_ground_humidity_difference,
                state.ground_humidity_difference_patch,
            ),
            monin_obukhov_length_patch=update(
                new_monin_length,
                state.monin_obukhov_length_patch,
            ),
            previous_monin_obukhov_length_patch=update(
                new_monin_length,
                state.previous_monin_obukhov_length_patch,
            ),
            stability_wind_speed_patch=update(
                new_stability_wind,
                state.stability_wind_speed_patch,
            ),
            stability_sign_changes_patch=update(
                sign_change_count,
                state.stability_sign_changes_patch,
            ),
            leaf_temperature_change_patch=update(
                leaf_temperature_change,
                state.leaf_temperature_change_patch,
            ),
            previous_leaf_temperature_change_patch=update(
                previous_change,
                state.previous_leaf_temperature_change_patch,
            ),
            leaf_latent_heat_flux_patch=update(
                leaf_latent,
                state.leaf_latent_heat_flux_patch,
            ),
            previous_leaf_latent_heat_flux_patch=previous_leaf_latent,
            leaf_latent_heat_flux_change_patch=update(
                latent_change,
                state.leaf_latent_heat_flux_change_patch,
            ),
            previous_vegetation_temperature_patch=update(
                previous_temperature,
                state.previous_vegetation_temperature_patch,
            ),
            vegetation_temperature_step_patch=update(
                temperature_step,
                state.vegetation_temperature_step_patch,
            ),
            leaf_saturation_vapor_pressure_patch=update(
                new_saturation.vapor_pressure,
                state.leaf_saturation_vapor_pressure_patch,
            ),
            leaf_saturation_specific_humidity_patch=update(
                new_saturation.specific_humidity,
                state.leaf_saturation_specific_humidity_patch,
            ),
            leaf_saturation_specific_humidity_derivative_patch=update(
                new_saturation.specific_humidity_temperature_derivative,
                state.leaf_saturation_specific_humidity_derivative_patch,
            ),
            vegetation_sensible_heat_flux_patch=update(
                vegetation_sensible,
                state.vegetation_sensible_heat_flux_patch,
            ),
            vegetation_evaporation_patch=update(
                vegetation_evaporation,
                state.vegetation_evaporation_patch,
            ),
            vegetation_transpiration_patch=update(
                transpiration,
                state.vegetation_transpiration_patch,
            ),
            friction_velocity_value_patch=update(
                friction_velocity_value,
                state.friction_velocity_value_patch,
            ),
            friction_stability_wind_input_patch=update(
                stability_wind,
                state.friction_stability_wind_input_patch,
            ),
            friction_momentum_numerator_patch=update(
                friction.momentum_numerator_patch,
                state.friction_momentum_numerator_patch,
            ),
            friction_momentum_denominator_patch=update(
                friction.momentum_denominator_patch,
                state.friction_momentum_denominator_patch,
            ),
            friction_momentum_zeta_patch=update(
                friction.momentum_zeta_patch,
                state.friction_momentum_zeta_patch,
            ),
            friction_temperature_profile_patch=update(
                temperature_profile,
                state.friction_temperature_profile_patch,
            ),
            friction_humidity_profile_patch=update(
                humidity_profile,
                state.friction_humidity_profile_patch,
            ),
            friction_temperature_profile_2m_patch=update(
                friction.temperature_profile_2m_patch,
                state.friction_temperature_profile_2m_patch,
            ),
            friction_humidity_profile_2m_patch=update(
                friction.humidity_profile_2m_patch,
                state.friction_humidity_profile_2m_patch,
            ),
            within_canopy_wind_patch=update(
                within_canopy_wind,
                state.within_canopy_wind_patch,
            ),
            bare_soil_drag_patch=update(
                bare_soil_drag,
                state.bare_soil_drag_patch,
            ),
            soil_drag_patch=update(
                soil_drag,
                state.soil_drag_patch,
            ),
            leaf_boundary_resistance_patch=update(
                leaf_boundary_resistance,
                state.leaf_boundary_resistance_patch,
            ),
            momentum_conductance_patch=update(
                momentum_conductance,
                state.momentum_conductance_patch,
            ),
            momentum_resistance_patch=update(
                momentum_resistance,
                state.momentum_resistance_patch,
            ),
            atmospheric_thermal_resistance_patch=update(
                atmospheric_thermal_resistance,
                state.atmospheric_thermal_resistance_patch,
            ),
            ground_thermal_resistance_patch=update(
                ground_thermal_resistance,
                state.ground_thermal_resistance_patch,
            ),
            atmospheric_moisture_resistance_patch=update(
                atmospheric_moisture_resistance,
                state.atmospheric_moisture_resistance_patch,
            ),
            ground_moisture_resistance_patch=update(
                ground_moisture_resistance,
                state.ground_moisture_resistance_patch,
            ),
            ground_sensible_conductance_patch=update(
                ground_sensible_conductance,
                state.ground_sensible_conductance_patch,
            ),
            ground_latent_conductance_patch=update(
                ground_latent_conductance,
                state.ground_latent_conductance_patch,
            ),
            normalized_leaf_sensible_conductance_patch=update(
                normalized_leaf_sensible,
                state.normalized_leaf_sensible_conductance_patch,
            ),
            normalized_ground_sensible_conductance_patch=update(
                normalized_ground_sensible,
                state.normalized_ground_sensible_conductance_patch,
            ),
            normalized_air_sensible_conductance_patch=update(
                normalized_air_sensible,
                state.normalized_air_sensible_conductance_patch,
            ),
            normalized_ground_air_sensible_conductance_patch=update(
                normalized_ground_air_sensible,
                state.normalized_ground_air_sensible_conductance_patch,
            ),
            normalized_air_leaf_sensible_conductance_patch=update(
                normalized_air_leaf_sensible,
                state.normalized_air_leaf_sensible_conductance_patch,
            ),
            normalized_leaf_latent_conductance_patch=update(
                normalized_leaf_latent,
                state.normalized_leaf_latent_conductance_patch,
            ),
            normalized_ground_latent_conductance_patch=update(
                normalized_ground_latent,
                state.normalized_ground_latent_conductance_patch,
            ),
            normalized_air_latent_conductance_patch=update(
                normalized_air_latent,
                state.normalized_air_latent_conductance_patch,
            ),
            normalized_ground_air_latent_conductance_patch=update(
                normalized_ground_air_latent,
                state.normalized_ground_air_latent_conductance_patch,
            ),
            normalized_air_leaf_latent_conductance_patch=update(
                normalized_air_leaf_latent,
                state.normalized_air_leaf_latent_conductance_patch,
            ),
            canopy_methane_conductance_patch=update(
                canopy_methane_conductance,
                state.canopy_methane_conductance_patch,
            ),
            ground_methane_conductance_patch=update(
                ground_methane_conductance,
                state.ground_methane_conductance_patch,
            ),
            leaf_sensible_linearization_patch=update(
                leaf_sensible,
                state.leaf_sensible_linearization_patch,
            ),
            leaf_latent_humidity_gradient_patch=update(
                leaf_latent_humidity_gradient,
                state.leaf_latent_humidity_gradient_patch,
            ),
            sensible_temperature_derivative_patch=update(
                sensible_derivative,
                state.sensible_temperature_derivative_patch,
            ),
            latent_temperature_derivative_patch=update(
                latent_derivative,
                state.latent_temperature_derivative_patch,
            ),
            temperature_scale_patch=update(
                temperature_scale,
                state.temperature_scale_patch,
            ),
            humidity_scale_patch=update(
                humidity_scale,
                state.humidity_scale_patch,
            ),
            virtual_temperature_scale_patch=update(
                virtual_temperature_scale,
                state.virtual_temperature_scale_patch,
            ),
            stability_zeta_patch=update(
                bounded_zeta,
                state.stability_zeta_patch,
            ),
            convective_argument_patch=update(
                convective_argument,
                state.convective_argument_patch,
            ),
            convective_velocity_patch=update(
                convective_velocity,
                state.convective_velocity_patch,
            ),
            stability_wind_squared_sum_patch=update(
                stability_wind_squared_sum,
                state.stability_wind_squared_sum_patch,
            ),
            energy_balance_correction_patch=update(
                truncation_correction,
                state.energy_balance_correction_patch,
            ),
            iteration_count_patch=jnp.where(
                active,
                state.iteration_count_patch + jnp.int32(1),
                state.iteration_count_patch,
            ),
            friction_velocity=friction.state,
            canopy_water_change_patch=(
                update(canopy_water_change, state.canopy_water_change_patch)
                if _coupled_water_balance_derivatives.get() else None
            ),
            canopy_water_remaining_patch=(
                update(canopy_water_remaining, state.canopy_water_remaining_patch)
                if _coupled_water_balance_derivatives.get() else None
            ),
        )

        def assign_fortran_state(value: jnp.ndarray) -> jnp.ndarray:

            if jnp.issubdtype(value.dtype, jnp.floating):
                return fortran_rounding_barrier(value)
            return lax.optimization_barrier(value)

        bounded_state = tree_util.tree_map(assign_fortran_state, next_state)
        if hydraulic_coupling is None:
            next_scan_state = bounded_state
        else:
            bounded_hydraulic = tree_util.tree_map(
                assign_fortran_state,
                iteration_hydraulic,
            )
            next_scan_state = _CanopyHydraulicScanState(
                bounded_state,
                bounded_hydraulic,
            )
        history_state = bounded_state if hydraulic_coupling is None else next_scan_state
        return next_scan_state, history_state if capture_history else None

    iterations = jnp.arange(MAXIMUM_CANOPY_ENERGY_ITERATIONS, dtype=jnp.int32)
    if _dynamic_primal.get() and not capture_history:
        final_scan_state = converged_iteration(
            iteration_step, initial_scan_state, iterations,
            lambda current: (
                current.active_patch if hydraulic_coupling is None
                else current.energy.active_patch
            ),
        )
        iteration_history = None
    else:
        final_scan_state, iteration_history = checkpointed_scan(
            iteration_step, initial_scan_state, iterations,
        )
    if hydraulic_coupling is None:
        final_iteration = final_scan_state
        final_hydraulic = None
    else:
        final_iteration = final_scan_state.energy
        final_hydraulic = final_scan_state.hydraulic
    result = CanopyEnergyResult(
        iteration=final_iteration,
        converged_patch=original_active & ~final_iteration.active_patch,
        valid_forcing_height_patch=(safe.forcing_height_wind_patch - displacement) >= 0.0,
        limited_vegetation_area_patch=limited_area,
        aerodynamic_fraction_patch=aerodynamic_fraction,
        displacement_height_patch=displacement,
        vegetation_momentum_roughness_patch=momentum_roughness,
        vegetation_sensible_roughness_patch=sensible_roughness,
        vegetation_latent_roughness_patch=latent_roughness,
        atmospheric_longwave_coefficient_patch=atmospheric_longwave,
        vegetation_longwave_coefficient_patch=vegetation_longwave,
        ground_longwave_coefficient_patch=ground_longwave,
    )
    return result, iteration_history, final_hydraulic


def canopy_energy_balance_with_prescribed_photosynthesis(
    inputs: CanopyEnergyInputs,
    photosynthesis: PrescribedCanopyPhotosynthesis,
    initial_state: CanopyEnergyInitialState,
    parameters: CanopyEnergyParameters,
) -> CanopyEnergyResult:

    result, _, _ = _canopy_energy_balance_with_prescribed_photosynthesis(
        inputs,
        photosynthesis,
        initial_state,
        parameters,
        capture_history=False,
    )
    return result


def _validate_canopy_fluxes_postprocess_shapes(
    energy_inputs: CanopyEnergyInputs,
    inputs: CanopyFluxesPostprocessInputs,
    previous: CanopyFluxesPostprocessState,
) -> None:
    patch_count = energy_inputs.active_patch.shape[0]
    expected = (patch_count,)
    for group_name, group in (("postprocess inputs", inputs), ("postprocess state", previous)):
        for name, value in zip(group._fields, group, strict=True):
            if name == "human_stress":
                continue
            if jnp.shape(value) != expected:
                raise ValueError(f"{group_name} field {name} must have shape {expected}")
    for name, value in zip(
        previous.human_stress._fields,
        previous.human_stress,
        strict=True,
    ):
        if jnp.shape(value) != expected:
            raise ValueError(f"human-stress field {name} must have shape {expected}")


def _finalize_canopy_fluxes_reference(
    energy_inputs: CanopyEnergyInputs,
    inputs: CanopyFluxesPostprocessInputs,
    energy: CanopyEnergyResult,
    previous: CanopyFluxesPostprocessState,
    parameters: CanopyFluxesPostprocessParameters,
    *,
    water_balance: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> CanopyFluxesPostprocessState:

    _validate_canopy_fluxes_postprocess_shapes(energy_inputs, inputs, previous)
    active = jnp.asarray(energy_inputs.active_patch, dtype=jnp.bool_)
    safe = _safe_inputs(energy_inputs)
    post_defaults = (0.005, 0.005, 0.005, 0.0, LATENT_HEAT_VAPORIZATION)
    post = CanopyFluxesPostprocessInputs(
        *(
            jnp.where(active, jnp.asarray(value, dtype=jnp.float64), jnp.float64(default))
            for value, default in zip(inputs, post_defaults, strict=True)
        )
    )
    iteration = energy.iteration
    step_seconds = jnp.asarray(parameters.step_seconds, dtype=jnp.float64)
    update_snow_canopy_water = jnp.asarray(
        parameters.update_snow_canopy_water,
        dtype=jnp.bool_,
    )

    def update(candidate: jnp.ndarray, old: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(active, fortran_rounding_barrier(candidate), old)

    ground_longwave_temperature_fourth = _ground_emitted_longwave_temperature_fourth(safe)
    previous_temperature = iteration.previous_vegetation_temperature_patch
    temperature_step = iteration.vegetation_temperature_step_patch
    linearized_temperature_factor = _add(
        previous_temperature,
        _multiply(jnp.float64(4.0), temperature_step),
    )
    previous_temperature_cubed = _cube(previous_temperature)
    canopy_energy_balance_error = _add(
        safe.absorbed_shortwave_vegetation_patch,
        energy.atmospheric_longwave_coefficient_patch,
        _multiply(
            _multiply(
                energy.vegetation_longwave_coefficient_patch,
                previous_temperature_cubed,
            ),
            linearized_temperature_factor,
        ),
        _multiply(
            energy.ground_longwave_coefficient_patch,
            ground_longwave_temperature_fourth,
        ),
        -iteration.vegetation_sensible_heat_flux_patch,
        -_multiply(
            jnp.float64(LATENT_HEAT_VAPORIZATION),
            iteration.vegetation_evaporation_patch,
        ),
    )

    ground_temperature_difference = _add(
        _multiply(
            iteration.normalized_air_leaf_sensible_conductance_patch,
            safe.ground_temperature_patch,
        ),
        -_multiply(
            iteration.normalized_leaf_sensible_conductance_patch,
            iteration.vegetation_temperature_patch,
        ),
        -_multiply(
            iteration.normalized_air_sensible_conductance_patch,
            safe.reference_temperature_patch,
        ),
    )
    eastward_wind_stress = _divide(
        _multiply(-safe.air_density_patch, safe.eastward_wind_patch),
        iteration.momentum_resistance_patch,
    )
    northward_wind_stress = _divide(
        _multiply(-safe.air_density_patch, safe.northward_wind_patch),
        iteration.momentum_resistance_patch,
    )

    def sensible_heat(surface_temperature: jnp.ndarray) -> jnp.ndarray:
        temperature_difference = _add(
            _multiply(
                iteration.normalized_air_leaf_sensible_conductance_patch,
                surface_temperature,
            ),
            -_multiply(
                iteration.normalized_leaf_sensible_conductance_patch,
                iteration.vegetation_temperature_patch,
            ),
            -_multiply(
                iteration.normalized_air_sensible_conductance_patch,
                safe.reference_temperature_patch,
            ),
        )
        return _multiply(
            _multiply(
                _multiply(jnp.float64(SPECIFIC_HEAT_DRY_AIR), safe.air_density_patch),
                iteration.ground_sensible_conductance_patch,
            ),
            temperature_difference,
        )

    sensible_ground = _multiply(
        _multiply(
            _multiply(jnp.float64(SPECIFIC_HEAT_DRY_AIR), safe.air_density_patch),
            iteration.ground_sensible_conductance_patch,
        ),
        ground_temperature_difference,
    )
    sensible_snow = sensible_heat(safe.snow_surface_temperature_patch)
    sensible_soil = sensible_heat(safe.soil_surface_temperature_patch)
    sensible_surface_water = sensible_heat(safe.surface_water_temperature_patch)
    ground_evaporation = _multiply(
        _multiply(safe.air_density_patch, iteration.ground_latent_conductance_patch),
        iteration.ground_humidity_difference_patch,
    )

    def surface_evaporation(surface_humidity: jnp.ndarray) -> jnp.ndarray:
        humidity_difference = _add(
            _multiply(
                iteration.normalized_air_leaf_latent_conductance_patch,
                surface_humidity,
            ),
            -_multiply(
                iteration.normalized_leaf_latent_conductance_patch,
                iteration.leaf_saturation_specific_humidity_patch,
            ),
            -_multiply(
                iteration.normalized_air_latent_conductance_patch,
                safe.atmospheric_specific_humidity_patch,
            ),
        )
        return _multiply(
            _multiply(safe.air_density_patch, iteration.ground_latent_conductance_patch),
            humidity_difference,
        )

    snow_evaporation = surface_evaporation(post.snow_surface_specific_humidity_patch)
    soil_evaporation = surface_evaporation(post.soil_surface_specific_humidity_patch)
    surface_water_evaporation = surface_evaporation(post.surface_water_specific_humidity_patch)

    reference_temperature_2m = _add(
        safe.reference_temperature_patch,
        _multiply(
            iteration.temperature_scale_patch,
            _add(
                _divide(
                    jnp.float64(1.0),
                    iteration.friction_temperature_profile_2m_patch,
                ),
                -_divide(
                    jnp.float64(1.0),
                    iteration.friction_temperature_profile_patch,
                ),
            ),
        ),
    )
    reference_specific_humidity_2m = _add(
        safe.atmospheric_specific_humidity_patch,
        _multiply(
            iteration.humidity_scale_patch,
            _add(
                _divide(
                    jnp.float64(1.0),
                    iteration.friction_humidity_profile_2m_patch,
                ),
                -_divide(
                    jnp.float64(1.0),
                    iteration.friction_humidity_profile_patch,
                ),
            ),
        ),
    )
    reference_saturation = qsat(reference_temperature_2m, safe.atmospheric_pressure_patch)
    reference_relative_humidity_2m = jnp.minimum(
        jnp.float64(100.0),
        _multiply(
            _divide(
                reference_specific_humidity_2m,
                reference_saturation.specific_humidity,
            ),
            jnp.float64(100.0),
        ),
    )
    calculated_human_stress = fast_human_stress_indices(
        reference_temperature_2m,
        reference_relative_humidity_2m,
        reference_saturation.vapor_pressure,
        iteration.friction_velocity.wind_10m_patch,
    )
    human_stress = CanopyFastHumanStressState(
        temperature_celsius_patch=update(
            calculated_human_stress.temperature_celsius,
            previous.human_stress.temperature_celsius_patch,
        ),
        vapor_pressure_patch=update(
            calculated_human_stress.vapor_pressure,
            previous.human_stress.vapor_pressure_patch,
        ),
        wet_bulb_stull_patch=update(
            calculated_human_stress.wet_bulb_stull,
            previous.human_stress.wet_bulb_stull_patch,
        ),
        rural_wet_bulb_stull_patch=update(
            calculated_human_stress.wet_bulb_stull,
            previous.human_stress.rural_wet_bulb_stull_patch,
        ),
        heat_index_patch=update(
            calculated_human_stress.heat_index,
            previous.human_stress.heat_index_patch,
        ),
        rural_heat_index_patch=update(
            calculated_human_stress.heat_index,
            previous.human_stress.rural_heat_index_patch,
        ),
        apparent_temperature_patch=update(
            calculated_human_stress.apparent_temperature,
            previous.human_stress.apparent_temperature_patch,
        ),
        rural_apparent_temperature_patch=update(
            calculated_human_stress.apparent_temperature,
            previous.human_stress.rural_apparent_temperature_patch,
        ),
        simplified_wet_bulb_globe_temperature_patch=update(
            calculated_human_stress.simplified_wet_bulb_globe_temperature,
            previous.human_stress.simplified_wet_bulb_globe_temperature_patch,
        ),
        rural_simplified_wet_bulb_globe_temperature_patch=update(
            calculated_human_stress.simplified_wet_bulb_globe_temperature,
            previous.human_stress.rural_simplified_wet_bulb_globe_temperature_patch,
        ),
        humidex_patch=update(
            calculated_human_stress.humidex,
            previous.human_stress.humidex_patch,
        ),
        rural_humidex_patch=update(
            calculated_human_stress.humidex,
            previous.human_stress.rural_humidex_patch,
        ),
        discomfort_index_stull_patch=update(
            calculated_human_stress.discomfort_index_stull,
            previous.human_stress.discomfort_index_stull_patch,
        ),
        rural_discomfort_index_stull_patch=update(
            calculated_human_stress.discomfort_index_stull,
            previous.human_stress.rural_discomfort_index_stull_patch,
        ),
    )

    one_minus_vegetation_emissivity = _add(
        jnp.float64(1.0),
        -safe.vegetation_emissivity_patch,
    )
    one_minus_ground_emissivity = _add(jnp.float64(1.0), -safe.ground_emissivity_patch)
    downward_longwave_below_canopy = _add(
        _multiply(
            _multiply(one_minus_vegetation_emissivity, safe.ground_emissivity_patch),
            safe.downward_longwave_patch,
        ),
        _multiply(
            _multiply(
                _multiply(
                    _multiply(
                        safe.vegetation_emissivity_patch,
                        safe.ground_emissivity_patch,
                    ),
                    jnp.float64(STEFAN_BOLTZMANN),
                ),
                previous_temperature_cubed,
            ),
            linearized_temperature_factor,
        ),
    )
    upward_longwave_above_canopy = _add(
        _multiply(
            _multiply(
                _multiply(one_minus_ground_emissivity, one_minus_vegetation_emissivity),
                one_minus_vegetation_emissivity,
            ),
            safe.downward_longwave_patch,
        ),
        _multiply(
            _multiply(
                _multiply(
                    _multiply(
                        safe.vegetation_emissivity_patch,
                        _add(
                            jnp.float64(1.0),
                            _multiply(
                                one_minus_ground_emissivity,
                                one_minus_vegetation_emissivity,
                            ),
                        ),
                    ),
                    jnp.float64(STEFAN_BOLTZMANN),
                ),
                previous_temperature_cubed,
            ),
            linearized_temperature_factor,
        ),
        _multiply(
            _multiply(
                _multiply(safe.ground_emissivity_patch, one_minus_vegetation_emissivity),
                jnp.float64(STEFAN_BOLTZMANN),
            ),
            ground_longwave_temperature_fourth,
        ),
    )
    skin_temperature = fortran_materialize(
        fortran_add(
            _multiply(
                safe.vegetation_emissivity_patch,
                iteration.vegetation_temperature_patch,
            ),
            _multiply(
                one_minus_vegetation_emissivity,
                jnp.sqrt(jnp.sqrt(ground_longwave_temperature_fourth)),
            ),
        )
    )

    ground_sensible_derivative = _add(
        previous.ground_sensible_derivative_patch,
        _multiply(
            _multiply(
                _multiply(jnp.float64(SPECIFIC_HEAT_DRY_AIR), safe.air_density_patch),
                iteration.ground_sensible_conductance_patch,
            ),
            iteration.normalized_air_leaf_sensible_conductance_patch,
        ),
    )
    ground_latent_derivative = _add(
        previous.ground_latent_derivative_patch,
        _multiply(
            _multiply(
                _multiply(safe.air_density_patch, iteration.ground_latent_conductance_patch),
                iteration.normalized_air_leaf_latent_conductance_patch,
            ),
            post.ground_specific_humidity_temperature_derivative_patch,
        ),
    )
    ground_energy_derivative = _add(
        ground_sensible_derivative,
        _multiply(ground_latent_derivative, post.ground_latent_heat_patch),
    )

    if water_balance is None:
        canopy_water_change = _multiply(
            _add(
                iteration.vegetation_transpiration_patch,
                -iteration.vegetation_evaporation_patch,
            ),
            step_seconds,
        )
        canopy_water = jnp.maximum(
            jnp.float64(0.0),
            _add(safe.canopy_water_patch, canopy_water_change),
        )
    else:
        canopy_water_change, canopy_water = water_balance
    warm_canopy = iteration.vegetation_temperature_patch > jnp.float64(
        FREEZING_TEMPERATURE_FRESHWATER
    )
    phase_balance = (
        _canopy_phase_water_balance if _coupled_water_balance_derivatives.get()
        else _canopy_phase_water_balance_reference
    )
    liquid_canopy_water, snow_canopy_water = phase_balance(
        previous.liquid_canopy_water_patch, previous.snow_canopy_water_patch,
        canopy_water_change, warm_canopy,
    )
    liquid_canopy_water = jnp.where(
        update_snow_canopy_water,
        liquid_canopy_water,
        previous.liquid_canopy_water_patch,
    )
    snow_canopy_water = jnp.where(
        update_snow_canopy_water,
        snow_canopy_water,
        previous.snow_canopy_water_patch,
    )

    return CanopyFluxesPostprocessState(
        canopy_energy_balance_error_patch=update(
            canopy_energy_balance_error,
            previous.canopy_energy_balance_error_patch,
        ),
        eastward_wind_stress_patch=update(
            eastward_wind_stress,
            previous.eastward_wind_stress_patch,
        ),
        northward_wind_stress_patch=update(
            northward_wind_stress,
            previous.northward_wind_stress_patch,
        ),
        sensible_heat_ground_patch=update(
            sensible_ground,
            previous.sensible_heat_ground_patch,
        ),
        sensible_heat_snow_patch=update(sensible_snow, previous.sensible_heat_snow_patch),
        sensible_heat_soil_patch=update(sensible_soil, previous.sensible_heat_soil_patch),
        sensible_heat_surface_water_patch=update(
            sensible_surface_water,
            previous.sensible_heat_surface_water_patch,
        ),
        ground_evaporation_patch=update(
            ground_evaporation,
            previous.ground_evaporation_patch,
        ),
        snow_evaporation_patch=update(
            snow_evaporation,
            previous.snow_evaporation_patch,
        ),
        soil_evaporation_patch=update(
            soil_evaporation,
            previous.soil_evaporation_patch,
        ),
        surface_water_evaporation_patch=update(
            surface_water_evaporation,
            previous.surface_water_evaporation_patch,
        ),
        reference_temperature_2m_patch=update(
            reference_temperature_2m,
            previous.reference_temperature_2m_patch,
        ),
        rural_reference_temperature_2m_patch=update(
            reference_temperature_2m,
            previous.rural_reference_temperature_2m_patch,
        ),
        reference_specific_humidity_2m_patch=update(
            reference_specific_humidity_2m,
            previous.reference_specific_humidity_2m_patch,
        ),
        reference_relative_humidity_2m_patch=update(
            reference_relative_humidity_2m,
            previous.reference_relative_humidity_2m_patch,
        ),
        rural_reference_relative_humidity_2m_patch=update(
            reference_relative_humidity_2m,
            previous.rural_reference_relative_humidity_2m_patch,
        ),
        human_stress=human_stress,
        downward_longwave_below_canopy_patch=update(
            downward_longwave_below_canopy,
            previous.downward_longwave_below_canopy_patch,
        ),
        upward_longwave_above_canopy_patch=update(
            upward_longwave_above_canopy,
            previous.upward_longwave_above_canopy_patch,
        ),
        skin_temperature_patch=update(skin_temperature, previous.skin_temperature_patch),
        ground_sensible_derivative_patch=update(
            ground_sensible_derivative,
            previous.ground_sensible_derivative_patch,
        ),
        ground_latent_derivative_patch=update(
            ground_latent_derivative,
            previous.ground_latent_derivative_patch,
        ),
        ground_energy_derivative_patch=update(
            ground_energy_derivative,
            previous.ground_energy_derivative_patch,
        ),
        canopy_water_patch=update(canopy_water, previous.canopy_water_patch),
        liquid_canopy_water_patch=update(
            liquid_canopy_water,
            previous.liquid_canopy_water_patch,
        ),
        snow_canopy_water_patch=update(
            snow_canopy_water,
            previous.snow_canopy_water_patch,
        ),
    )


@jax.custom_jvp
def finalize_canopy_fluxes(
    energy_inputs: CanopyEnergyInputs,
    inputs: CanopyFluxesPostprocessInputs,
    energy: CanopyEnergyResult,
    previous: CanopyFluxesPostprocessState,
    parameters: CanopyFluxesPostprocessParameters,
    water_balance: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> CanopyFluxesPostprocessState:
    return _finalize_canopy_fluxes_reference(
        energy_inputs, inputs, energy, previous, parameters,
        water_balance=water_balance,
    )


def _finalize_canopy_fluxes_jvp(primals, tangents):
    from clm_jax.compilation import _symbolic_zero_jvp

    def differentiated(energy_inputs, inputs, energy, previous, parameters, water_balance):
        active = jnp.asarray(energy_inputs.active_patch, dtype=jnp.bool_)
        denominators = (
            "momentum_resistance_patch",
            "friction_temperature_profile_2m_patch",
            "friction_temperature_profile_patch",
            "friction_humidity_profile_2m_patch",
            "friction_humidity_profile_patch",
        )
        iteration = energy.iteration._replace(**{
            name: jnp.where(active, getattr(energy.iteration, name), 1.0)
            for name in denominators
        })
        return _finalize_canopy_fluxes_reference(
            energy_inputs, inputs, energy._replace(iteration=iteration), previous,
            parameters, water_balance=water_balance,
        )

    return _symbolic_zero_jvp(differentiated, primals, tangents)


finalize_canopy_fluxes.defjvp(_finalize_canopy_fluxes_jvp, symbolic_zeros=True)


def _canopy_hydraulic_energy_reference(
    inputs: CanopyEnergyInputs,
    coupling: CanopyHydraulicCouplingInputs,
    initial_state: CanopyEnergyInitialState,
    parameters: CanopyEnergyParameters,
) -> CanopyHydraulicEnergyResult:

    patch_count = inputs.active_patch.shape[0]
    initial_photo = PrescribedCanopyPhotosynthesis(
        sunlit_stomatal_resistance_patch=jnp.full(
            (patch_count,),
            jnp.float64(100.0),
        ),
        shaded_stomatal_resistance_patch=jnp.full(
            (patch_count,),
            jnp.float64(100.0),
        ),
        transpiration_wetness_patch=jnp.ones((patch_count,), dtype=jnp.float64),
        vegetation_transpiration_patch=(coupling.initial_plant_state.vegetation_transpiration),
    )
    energy, _, hydraulic = _canopy_energy_balance_with_prescribed_photosynthesis(
        inputs,
        initial_photo,
        initial_state,
        parameters,
        capture_history=False,
        hydraulic_coupling=coupling,
    )
    if hydraulic is None:
        raise AssertionError("hydraulic coupling must return a hydraulic state")
    return CanopyHydraulicEnergyResult(energy=energy, hydraulic=hydraulic)


def canopy_energy_balance_with_hydraulic_photosynthesis(
    inputs: CanopyEnergyInputs,
    coupling: CanopyHydraulicCouplingInputs,
    initial_state: CanopyEnergyInitialState,
    parameters: CanopyEnergyParameters,
) -> CanopyHydraulicEnergyResult:
    from clm_jax.physics.solver_scalarization import _native_tile_trace

    if (_dynamic_primal.get() and default_backend() == "gpu"
            and not _native_tile_trace.get() and inputs.active_patch.size):
        from clm_jax.physics.canopy_fluxes_gpu import canopy_hydraulic_energy_gpu
        return canopy_hydraulic_energy_gpu(inputs, coupling, initial_state, parameters)
    return _canopy_hydraulic_energy_reference(inputs, coupling, initial_state, parameters)


def _complete_active_canopy_fluxes_state(
    topology: SubgridTopology,
    energy_inputs: CanopyEnergyInputs,
    postprocess_inputs: CanopyFluxesPostprocessInputs,
    luna_inputs: CanopyFluxesLunaInputs,
    solved: CanopyHydraulicEnergyResult,
    previous: CanopyFluxesActiveState,
    parameters: CanopyFluxesActiveParameters,
) -> CanopyFluxesActiveState:

    active = jnp.asarray(energy_inputs.active_patch, dtype=jnp.bool_)
    step_seconds = jnp.asarray(parameters.energy.step_seconds, dtype=jnp.float64)
    postprocess = finalize_canopy_fluxes(
        energy_inputs,
        postprocess_inputs,
        solved.energy,
        previous.postprocess,
        CanopyFluxesPostprocessParameters(
            step_seconds=step_seconds,
            update_snow_canopy_water=parameters.update_snow_canopy_water,
        ),
        water_balance=(
            None if solved.energy.iteration.canopy_water_change_patch is None
            else (solved.energy.iteration.canopy_water_change_patch,
                  solved.energy.iteration.canopy_water_remaining_patch)
        ),
    )

    hydraulic = solved.hydraulic
    photosynthesis = photosynthesis_total(
        active,
        PhotosynthesisTotalInputs(
            sunlit_leaf_area_patch=energy_inputs.sunlit_leaf_area_index_patch,
            shaded_leaf_area_patch=energy_inputs.shaded_leaf_area_index_patch,
            sunlit_photosynthesis_patch=(hydraulic.gross_photosynthesis_patch_leaf[:, SUNLIT]),
            shaded_photosynthesis_patch=(hydraulic.gross_photosynthesis_patch_leaf[:, SHADED]),
            sunlit_rubisco_limited_patch=(
                hydraulic.rubisco_limited_photosynthesis_patch_leaf[:, SUNLIT]
            ),
            shaded_rubisco_limited_patch=(
                hydraulic.rubisco_limited_photosynthesis_patch_leaf[:, SHADED]
            ),
            sunlit_electron_transport_limited_patch=(
                hydraulic.electron_transport_limited_photosynthesis_patch_leaf[:, SUNLIT]
            ),
            shaded_electron_transport_limited_patch=(
                hydraulic.electron_transport_limited_photosynthesis_patch_leaf[:, SHADED]
            ),
            sunlit_product_limited_patch=(
                hydraulic.product_limited_photosynthesis_patch_leaf[:, SUNLIT]
            ),
            shaded_product_limited_patch=(
                hydraulic.product_limited_photosynthesis_patch_leaf[:, SHADED]
            ),
        ),
        previous.photosynthesis_total,
    )

    ozone_stress = ozone_stress_off(previous.ozone_stress)

    daily = luna_accumulate_daily(
        active,
        step_seconds,
        LunaDailyClimateInputs(
            absorbed_shortwave_vegetation_patch=(energy_inputs.absorbed_shortwave_vegetation_patch),
            vegetation_temperature_patch=(solved.energy.iteration.vegetation_temperature_patch),
            absorbed_par_sunlit_by_layer_patch=(luna_inputs.absorbed_par_sunlit_by_layer_patch),
            absorbed_par_shaded_by_layer_patch=(luna_inputs.absorbed_par_shaded_by_layer_patch),
            sunlit_leaf_area_by_layer_patch=(luna_inputs.sunlit_leaf_area_by_layer_patch),
            shaded_leaf_area_by_layer_patch=(luna_inputs.shaded_leaf_area_by_layer_patch),
            radiative_layer_count_patch=luna_inputs.radiative_layer_count_patch,
            photosynthesis_patch=photosynthesis.photosynthesis_patch,
        ),
        previous.luna_daily,
    )

    ten_day_inputs = LunaTenDayClimateInputs(
        relative_humidity_patch=hydraulic.canopy_relative_humidity_patch,
        boundary_layer_resistance_patch=(solved.energy.iteration.leaf_boundary_resistance_patch),
    )

    def finish_day(
        states: tuple[LunaDailyClimateState, LunaTenDayClimateState, LunaCapacityState],
    ) -> tuple[LunaDailyClimateState, LunaTenDayClimateState, LunaCapacityState]:
        daily_state, ten_day_state, capacity_state = states
        updated_ten_day = luna_accumulate_ten_day(
            active,
            step_seconds,
            ten_day_inputs,
            daily_state,
            ten_day_state,
        )
        capacity_inputs = luna_inputs.capacity_inputs._replace(
            radiative_layer_count_patch=luna_inputs.radiative_layer_count_patch,
            absorbed_par_mean_by_layer_patch=(updated_ten_day.absorbed_par_mean_by_layer_patch),
            absorbed_par_maximum_by_layer_patch=(
                updated_ten_day.absorbed_par_maximum_by_layer_patch
            ),
            daytime_temperature_accumulator_patch=(daily_state.daytime_temperature_sum_patch),
            daytime_leaf_temperature_ten_day_patch=(updated_ten_day.daytime_temperature_patch),
            nighttime_leaf_temperature_ten_day_patch=(updated_ten_day.nighttime_temperature_patch),
            relative_humidity_ten_day_patch=(updated_ten_day.relative_humidity_patch),
            boundary_layer_resistance_ten_day_patch=(
                updated_ten_day.boundary_layer_resistance_patch
            ),
            daily_photosynthesis_patch=daily_state.photosynthesis_sum_patch,
        )
        updated_capacity = luna_update_photosynthesis_capacity(
            topology,
            active,
            parameters.luna_capacity,
            capacity_inputs,
            capacity_state,
        )
        return (
            luna_clear_daily(active, daily_state),
            updated_ten_day,
            updated_capacity,
        )

    def retain_day(
        states: tuple[LunaDailyClimateState, LunaTenDayClimateState, LunaCapacityState],
    ) -> tuple[LunaDailyClimateState, LunaTenDayClimateState, LunaCapacityState]:
        return states

    is_end_day = jnp.asarray(parameters.is_end_day, dtype=jnp.bool_)
    if is_end_day.shape != ():
        raise ValueError("is_end_day must be scalar")
    daily, ten_day, capacity = lax.cond(
        is_end_day,
        checkpoint_iteration(finish_day),
        retain_day,
        (daily, previous.luna_ten_day, previous.luna_capacity),
    )
    return CanopyFluxesActiveState(
        postprocess=postprocess,
        photosynthesis_total=photosynthesis,
        ozone_stress=ozone_stress,
        luna_daily=daily,
        luna_ten_day=ten_day,
        luna_capacity=capacity,
    )


def _prepare_active_canopy_fluxes_inputs(
    energy_inputs: CanopyEnergyInputs,
    hydraulic_coupling: CanopyHydraulicCouplingInputs,
    state: CanopyFluxesActiveState,
) -> tuple[CanopyEnergyInputs, CanopyHydraulicCouplingInputs]:

    energy_inputs = energy_inputs._replace(
        canopy_water_patch=state.postprocess.canopy_water_patch,
    )
    hydraulic_coupling = hydraulic_coupling._replace(
        controls=hydraulic_coupling.controls._replace(
            sunlit_stomatal_ozone_coefficient=(
                state.ozone_stress.sunlit_conductance_coefficient_patch
            ),
            shaded_stomatal_ozone_coefficient=(
                state.ozone_stress.shaded_conductance_coefficient_patch
            ),
        )
    )
    return energy_inputs, hydraulic_coupling


def _canopy_fluxes_active_case_reference(
    topology: SubgridTopology,
    energy_inputs: CanopyEnergyInputs,
    hydraulic_coupling: CanopyHydraulicCouplingInputs,
    initial_energy_state: CanopyEnergyInitialState,
    postprocess_inputs: CanopyFluxesPostprocessInputs,
    luna_inputs: CanopyFluxesLunaInputs,
    state: CanopyFluxesActiveState,
    parameters: CanopyFluxesActiveParameters,
) -> CanopyFluxesActiveResult:

    energy_inputs, coupling = _prepare_active_canopy_fluxes_inputs(
        energy_inputs,
        hydraulic_coupling,
        state,
    )
    solved = canopy_energy_balance_with_hydraulic_photosynthesis(
        energy_inputs,
        coupling,
        initial_energy_state,
        parameters.energy,
    )
    updated_state = _complete_active_canopy_fluxes_state(
        topology,
        energy_inputs,
        postprocess_inputs,
        luna_inputs,
        solved,
        state,
        parameters,
    )
    if _coupled_water_balance_derivatives.get():
        solved = solved._replace(energy=solved.energy._replace(
            iteration=solved.energy.iteration._replace(
                canopy_water_change_patch=None, canopy_water_remaining_patch=None,
            ),
        ))
    return CanopyFluxesActiveResult(
        energy_hydraulics=solved,
        state=updated_state,
    )


@jax.custom_jvp
def canopy_fluxes_active_case(
    topology: SubgridTopology,
    energy_inputs: CanopyEnergyInputs,
    hydraulic_coupling: CanopyHydraulicCouplingInputs,
    initial_energy_state: CanopyEnergyInitialState,
    postprocess_inputs: CanopyFluxesPostprocessInputs,
    luna_inputs: CanopyFluxesLunaInputs,
    state: CanopyFluxesActiveState,
    parameters: CanopyFluxesActiveParameters,
) -> CanopyFluxesActiveResult:
    return _canopy_fluxes_active_case_reference(
        topology, energy_inputs, hydraulic_coupling, initial_energy_state,
        postprocess_inputs, luna_inputs, state, parameters,
    )


@partial(canopy_fluxes_active_case.defjvp, symbolic_zeros=True)
def _canopy_fluxes_active_case_jvp(primals, tangents):
    token = _coupled_water_balance_derivatives.set(True)
    try:
        return _complete_canopy_phase_budget_jvp(primals, tangents)
    finally:
        _coupled_water_balance_derivatives.reset(token)


def trace_canopy_energy_balance_with_prescribed_photosynthesis(
    inputs: CanopyEnergyInputs,
    photosynthesis: PrescribedCanopyPhotosynthesis,
    initial_state: CanopyEnergyInitialState,
    parameters: CanopyEnergyParameters,
) -> CanopyEnergyTrace:

    result, iteration_history, _ = _canopy_energy_balance_with_prescribed_photosynthesis(
        inputs,
        photosynthesis,
        initial_state,
        parameters,
        capture_history=True,
    )
    if iteration_history is None:
        raise AssertionError("capture_history=True must return an iteration history")
    return CanopyEnergyTrace(result=result, iteration_history=iteration_history)


def trace_canopy_energy_balance_with_hydraulic_photosynthesis(
    inputs: CanopyEnergyInputs,
    coupling: CanopyHydraulicCouplingInputs,
    initial_state: CanopyEnergyInitialState,
    parameters: CanopyEnergyParameters,
) -> CanopyHydraulicEnergyTrace:

    patch_count = inputs.active_patch.shape[0]
    initial_photo = PrescribedCanopyPhotosynthesis(
        sunlit_stomatal_resistance_patch=jnp.full(
            (patch_count,),
            jnp.float64(100.0),
        ),
        shaded_stomatal_resistance_patch=jnp.full(
            (patch_count,),
            jnp.float64(100.0),
        ),
        transpiration_wetness_patch=jnp.ones((patch_count,), dtype=jnp.float64),
        vegetation_transpiration_patch=(coupling.initial_plant_state.vegetation_transpiration),
    )
    energy, iteration_history, hydraulic = _canopy_energy_balance_with_prescribed_photosynthesis(
        inputs,
        initial_photo,
        initial_state,
        parameters,
        capture_history=True,
        hydraulic_coupling=coupling,
    )
    if hydraulic is None or not isinstance(iteration_history, _CanopyHydraulicScanState):
        raise AssertionError("coupled trace must return hydraulic histories")
    result = CanopyHydraulicEnergyResult(energy=energy, hydraulic=hydraulic)
    return CanopyHydraulicEnergyTrace(
        result=result,
        energy_iteration_history=iteration_history.energy,
        hydraulic_iteration_history=iteration_history.hydraulic,
    )


def _phase_tangents_from_coupled_budget(
    liquid, snow, change, remaining, warm, dl, ds, dw, dc, dr, *, water,
):
    from clm_jax.physics.adjoint_scaling import sensitivity_product, sensitivity_sum

    liquid_change = sensitivity_sum(dr, dl, -dw)
    snow_change = sensitivity_sum(dr, ds, -dw)
    total_change = sensitivity_sum(dr, dl, ds, -dw)

    def retained_fraction(trial):
        return jnp.where(trial > 0.0, 1.0, jnp.where(trial < 0.0, 0.0, 0.5))

    liquid_fraction = retained_fraction(sensitivity_sum(liquid, -water, remaining))
    snow_fraction = retained_fraction(sensitivity_sum(snow, -water, remaining))
    dry_corner = (remaining == 0.0) & (water >= 0.0)
    liquid_fraction = jnp.where(
        dry_corner & (liquid == water), 1.0, liquid_fraction,
    )
    snow_fraction = jnp.where(
        dry_corner & (snow == water), 1.0, snow_fraction,
    )
    liquid_dot = jnp.where(
        warm, sensitivity_product(liquid_fraction, liquid_change),
        jnp.where(snow_fraction == 0.0, total_change,
                  sensitivity_sum(dl,
                      sensitivity_product(1.0 - snow_fraction, dr),
                      sensitivity_product(1.0 - snow_fraction, ds),
                      -sensitivity_product(1.0 - snow_fraction, dw))),
    )
    snow_dot = jnp.where(
        warm, jnp.where(liquid_fraction == 0.0, total_change,
                        sensitivity_sum(ds,
                            sensitivity_product(1.0 - liquid_fraction, dr),
                            sensitivity_product(1.0 - liquid_fraction, dl),
                            -sensitivity_product(1.0 - liquid_fraction, dw))),
        sensitivity_product(snow_fraction, snow_change),
    )
    return liquid_dot, snow_dot


def _complete_canopy_phase_budget_jvp(primals, tangents):
    from clm_jax.compilation import _symbolic_zero_jvp
    from jax.custom_derivatives import SymbolicZero
    def extended(topology, energy_inputs, hydraulic_coupling, initial_energy_state,
                 postprocess_inputs, luna_inputs, state, parameters):
        energy_inputs, coupling = _prepare_active_canopy_fluxes_inputs(
            energy_inputs, hydraulic_coupling, state,
        )
        solved = canopy_energy_balance_with_hydraulic_photosynthesis(
            energy_inputs, coupling, initial_energy_state, parameters.energy,
        )
        updated_state = _complete_active_canopy_fluxes_state(
            topology, energy_inputs, postprocess_inputs, luna_inputs,
            solved, state, parameters,
        )
        iteration = solved.energy.iteration
        budget = iteration.canopy_water_change_patch, iteration.canopy_water_remaining_patch
        solved = solved._replace(energy=solved.energy._replace(
            iteration=iteration._replace(
                canopy_water_change_patch=None, canopy_water_remaining_patch=None,
            ),
        ))
        return CanopyFluxesActiveResult(solved, updated_state), budget

    (result, (change, remaining)), (derivative, (dc, dr)) = _symbolic_zero_jvp(
        extended, primals, tangents,
    )
    previous, previous_dot = primals[6].postprocess, tangents[6].postprocess
    directions=(previous_dot.liquid_canopy_water_patch,previous_dot.snow_canopy_water_patch,
                previous_dot.canopy_water_patch,dc,dr)
    if all(isinstance(value,SymbolicZero) for value in directions):
        return result,derivative._replace(state=derivative.state._replace(
            postprocess=derivative.state.postprocess._replace(
                liquid_canopy_water_patch=SymbolicZero(
                    jax.typeof(previous.liquid_canopy_water_patch).to_tangent_aval()),
                snow_canopy_water_patch=SymbolicZero(
                    jax.typeof(previous.snow_canopy_water_patch).to_tangent_aval()))))
    dl0,ds0,dw,dc,dr=(jnp.zeros(value.shape,value.dtype)
        if isinstance(value,SymbolicZero) else value for value in directions)
    warm = (result.energy_hydraulics.energy.iteration.vegetation_temperature_patch
            > jnp.float64(FREEZING_TEMPERATURE_FRESHWATER))
    dl, ds = _phase_tangents_from_coupled_budget(
        previous.liquid_canopy_water_patch, previous.snow_canopy_water_patch,
        change, remaining, warm,
        dl0, ds0, dw, dc, dr,
        water=previous.canopy_water_patch,
    )
    enabled = (jnp.asarray(primals[1].active_patch, dtype=jnp.bool_)
               & jnp.asarray(primals[7].update_snow_canopy_water, dtype=jnp.bool_))
    postprocess_dot = derivative.state.postprocess._replace(
        liquid_canopy_water_patch=jnp.where(
            enabled, dl, dl0,
        ),
        snow_canopy_water_patch=jnp.where(
            enabled, ds, ds0,
        ),
    )
    return result, derivative._replace(state=derivative.state._replace(
        postprocess=postprocess_dot,
    ))
