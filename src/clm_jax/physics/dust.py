
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import lax
from jax.scipy.special import erf

from clm_jax.constants import (
    BOLTZMANN,
    CLM_SPECIAL_VALUE,
    DENSITY_FRESH_WATER,
    GAS_CONSTANT_DRY_AIR,
    GRAVITY,
    PI,
)
from clm_jax.io.restart import LANDUNIT_CROP, LANDUNIT_LAKE, LANDUNIT_SOIL, SubgridTopology
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
    fortran_ordered_sum,
    fortran_power,
    strict_exp,
)
from clm_jax.physics.pointwise import scalar_map

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase


NUMBER_OF_DUST_BINS = 4
NUMBER_OF_DUST_SOURCE_MODES = 3
NUMBER_OF_PARTICLE_SUBBINS = 200


class DustExecutionParameters(NamedTuple):

    maximum_patches_per_landunit: int = 79

    @classmethod
    def from_case(cls, case: CimeCase) -> DustExecutionParameters:
        from clm_jax.io.namelist import read_case_namelist_group  

        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        return cls(maximum_patches_per_landunit=int(general["maxpatch_pft"]))


DEFAULT_DUST_EXECUTION_PARAMETERS = DustExecutionParameters()


class DustParameters(NamedTuple):

    source_to_sink_mass_overlap: jnp.ndarray
    mass_weighted_diameter: jnp.ndarray
    stokes_settling_correction: jnp.ndarray
    saltation_threshold_numerator: jnp.ndarray
    aerosol_density: jnp.ndarray


class DustState(NamedTuple):

    emission_flux_patch: jnp.ndarray
    total_emission_flux_patch: jnp.ndarray
    turbulent_deposition_velocity_patch: jnp.ndarray
    turbulent_deposition_velocity_1_patch: jnp.ndarray
    turbulent_deposition_velocity_2_patch: jnp.ndarray
    turbulent_deposition_velocity_3_patch: jnp.ndarray
    turbulent_deposition_velocity_4_patch: jnp.ndarray
    mobile_basin_factor_column: jnp.ndarray


class DustEmissionInputs(NamedTuple):

    nonlake_patch: jnp.ndarray
    air_density_column: jnp.ndarray
    gravimetric_water_threshold_column: jnp.ndarray
    clay_mass_fraction_limited_column: jnp.ndarray
    saturated_water_content_top_column: jnp.ndarray
    snow_cover_fraction_column: jnp.ndarray
    volumetric_soil_water_top_column: jnp.ndarray
    soil_liquid_water_top_column: jnp.ndarray
    soil_ice_top_column: jnp.ndarray
    leaf_area_index_patch: jnp.ndarray
    stem_area_index_patch: jnp.ndarray
    dust_friction_velocity_patch: jnp.ndarray
    dust_wind_10m_patch: jnp.ndarray


class DustDepositionInputs(NamedTuple):

    air_pressure_column: jnp.ndarray
    air_density_column: jnp.ndarray
    air_temperature_column: jnp.ndarray
    aerodynamic_resistance_patch: jnp.ndarray
    dust_friction_velocity_patch: jnp.ndarray


class DustEmissionDiagnostics(NamedTuple):
    total_vegetation_area_patch: jnp.ndarray
    vegetation_area_landunit: jnp.ndarray
    vegetation_weight_landunit: jnp.ndarray
    mobile_land_fraction_patch: jnp.ndarray
    dry_bulk_density_patch: jnp.ndarray
    surface_gravimetric_water_patch: jnp.ndarray
    wet_threshold_factor_patch: jnp.ndarray
    liquid_water_fraction_patch: jnp.ndarray
    threshold_friction_velocity_patch: jnp.ndarray
    threshold_reference_wind_patch: jnp.ndarray
    saltating_friction_velocity_patch: jnp.ndarray
    horizontal_saltation_flux_patch: jnp.ndarray
    vertical_to_horizontal_flux_ratio_patch: jnp.ndarray
    total_vertical_dust_flux_patch: jnp.ndarray


class DustDepositionDiagnostics(NamedTuple):
    dynamic_air_viscosity_patch: jnp.ndarray
    mean_free_path_patch: jnp.ndarray
    kinematic_air_viscosity_patch: jnp.ndarray
    slip_correction_patch: jnp.ndarray
    gravitational_settling_velocity_patch: jnp.ndarray
    stokes_number_patch: jnp.ndarray
    aerosol_diffusivity_patch: jnp.ndarray
    schmidt_number_patch: jnp.ndarray
    collection_efficiency_patch: jnp.ndarray
    quasi_laminar_resistance_patch: jnp.ndarray
    turbulent_resistance_patch: jnp.ndarray


class DustEmissionResult(NamedTuple):
    state: DustState
    diagnostics: DustEmissionDiagnostics


class DustDepositionResult(NamedTuple):
    state: DustState
    diagnostics: DustDepositionDiagnostics


class DustStepResult(NamedTuple):
    state: DustState
    emission: DustEmissionDiagnostics
    deposition: DustDepositionDiagnostics


def _elementwise_log(value: jnp.ndarray) -> jnp.ndarray:
    value = jnp.asarray(value, dtype=jnp.float64)
    shape = value.shape
    return jnp.reshape(scalar_map(jnp.log, jnp.reshape(value, (-1,))), shape)


def _particle_grid_statistics(
    minimum_diameter: jnp.ndarray,
    maximum_diameter: jnp.ndarray,
    number_median_diameter: jnp.ndarray,
    logarithmic_standard_deviation: jnp.ndarray,
) -> jnp.ndarray:

    ratio = fortran_power(
        fortran_divide(maximum_diameter, minimum_diameter),
        fortran_divide(jnp.float64(1.0), jnp.float64(NUMBER_OF_PARTICLE_SUBBINS)),
    )

    def next_minimum(previous: jnp.ndarray, _: None):
        current = fortran_multiply(previous, ratio)
        return current, current

    _, remaining_minima = lax.scan(
        next_minimum,
        minimum_diameter,
        xs=None,
        length=NUMBER_OF_PARTICLE_SUBBINS - 1,
    )
    subbin_minimum = jnp.concatenate((minimum_diameter[None, :], remaining_minima), axis=0)
    subbin_maximum = jnp.concatenate(
        (subbin_minimum[1:, :], maximum_diameter[None, :]),
        axis=0,
    )
    subbin_center = fortran_multiply(
        jnp.float64(0.5),
        fortran_add(subbin_minimum, subbin_maximum),
    )
    subbin_width = fortran_add(subbin_maximum, -subbin_minimum)
    normalization = fortran_divide(
        jnp.float64(1.0),
        fortran_multiply(
            logarithmic_standard_deviation,
            jnp.sqrt(fortran_multiply(jnp.float64(2.0), jnp.float64(PI))),
        ),
    )
    log_argument = fortran_divide(subbin_center, number_median_diameter)
    standardized_log = fortran_divide(
        _elementwise_log(log_argument),
        logarithmic_standard_deviation,
    )
    lognormal = fortran_divide(
        fortran_multiply(
            normalization,
            strict_exp(
                fortran_multiply(
                    jnp.float64(-0.5),
                    fortran_multiply(standardized_log, standardized_log),
                )
            ),
        ),
        subbin_center,
    )

    def integrate(
        index: int,
        carry: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        diameter_moment, volume_moment = carry
        center = subbin_center[index]
        center_cubed = fortran_power(center, jnp.float64(3.0))
        particle_volume = fortran_multiply(
            fortran_divide(jnp.float64(PI), jnp.float64(6.0)),
            center_cubed,
        )
        volume_term = fortran_multiply(
            fortran_multiply(particle_volume, lognormal[index]),
            subbin_width[index],
        )
        diameter_term = fortran_multiply(center, jnp.float64(PI))
        diameter_term = fortran_divide(diameter_term, jnp.float64(6.0))
        diameter_term = fortran_multiply(diameter_term, center_cubed)
        diameter_term = fortran_multiply(diameter_term, lognormal[index])
        diameter_term = fortran_multiply(diameter_term, subbin_width[index])
        return (
            fortran_add(diameter_moment, diameter_term),
            fortran_add(volume_moment, volume_term),
        )

    diameter_moment, volume_moment = lax.fori_loop(
        0,
        NUMBER_OF_PARTICLE_SUBBINS,
        integrate,
        (
            jnp.zeros(NUMBER_OF_DUST_BINS, dtype=jnp.float64),
            jnp.zeros(NUMBER_OF_DUST_BINS, dtype=jnp.float64),
        ),
    )
    return fortran_divide(diameter_moment, volume_moment)


def _stokes_settling_correction(
    mass_weighted_diameter: jnp.ndarray,
    aerosol_density: jnp.ndarray,
) -> jnp.ndarray:

    pressure = jnp.float64(100000.0)
    temperature = jnp.float64(295.0)
    air_density = fortran_divide(
        pressure,
        fortran_multiply(temperature, jnp.float64(GAS_CONSTANT_DRY_AIR)),
    )
    temperature_ratio = fortran_divide(temperature, jnp.float64(273.0))
    dynamic_viscosity = fortran_divide(
        fortran_multiply(
            fortran_multiply(
                jnp.float64(1.72e-5),
                fortran_power(temperature_ratio, jnp.float64(1.5)),
            ),
            jnp.float64(393.0),
        ),
        fortran_add(temperature, jnp.float64(120.0)),
    )
    root_argument = fortran_divide(
        jnp.float64(8.0),
        fortran_multiply(
            fortran_multiply(jnp.float64(PI), jnp.float64(GAS_CONSTANT_DRY_AIR)),
            temperature,
        ),
    )
    mean_free_path = fortran_divide(
        fortran_multiply(jnp.float64(2.0), dynamic_viscosity),
        fortran_multiply(pressure, jnp.sqrt(root_argument)),
    )
    kinematic_viscosity = fortran_divide(dynamic_viscosity, air_density)
    exponential_argument = fortran_divide(
        fortran_multiply(jnp.float64(-1.1), mass_weighted_diameter),
        fortran_multiply(jnp.float64(2.0), mean_free_path),
    )
    slip = fortran_add(
        jnp.float64(1.0),
        fortran_divide(
            fortran_multiply(
                fortran_multiply(jnp.float64(2.0), mean_free_path),
                fortran_add(
                    jnp.float64(1.257),
                    fortran_multiply(jnp.float64(0.4), strict_exp(exponential_argument)),
                ),
            ),
            mass_weighted_diameter,
        ),
    )
    stokes_velocity = fortran_divide(
        fortran_multiply(
            fortran_multiply(
                fortran_multiply(
                    fortran_multiply(
                        fortran_divide(jnp.float64(1.0), jnp.float64(18.0)),
                        mass_weighted_diameter,
                    ),
                    mass_weighted_diameter,
                ),
                aerosol_density,
            ),
            fortran_multiply(jnp.float64(GRAVITY), slip),
        ),
        dynamic_viscosity,
    )

    initial = (
        stokes_velocity,
        jnp.full(NUMBER_OF_DUST_BINS, jnp.float64(1.0001), dtype=jnp.float64),
        jnp.zeros(NUMBER_OF_DUST_BINS, dtype=jnp.float64),
        jnp.ones(NUMBER_OF_DUST_BINS, dtype=jnp.bool_),
    )

    def continue_iteration(state) -> jnp.ndarray:
        return jnp.any(state[3])

    def iteration(state):
        velocity, error, iteration_index, active = state
        del error
        old_velocity = velocity
        reynolds = fortran_divide(
            fortran_multiply(old_velocity, mass_weighted_diameter),
            kinematic_viscosity,
        )
        drag_stokes = fortran_divide(jnp.float64(24.0), reynolds)
        drag_transition = fortran_multiply(
            drag_stokes,
            fortran_ordered_sum(
                jnp.stack(
                    (
                        jnp.ones_like(reynolds),
                        fortran_divide(
                            fortran_multiply(jnp.float64(3.0), reynolds),
                            jnp.float64(16.0),
                        ),
                        fortran_divide(
                            fortran_multiply(
                                fortran_multiply(
                                    fortran_multiply(jnp.float64(9.0), reynolds),
                                    reynolds,
                                ),
                                _elementwise_log(fortran_multiply(jnp.float64(2.0), reynolds)),
                            ),
                            jnp.float64(160.0),
                        ),
                    ),
                    axis=0,
                )
            ),
        )
        drag_intermediate = fortran_multiply(
            drag_stokes,
            fortran_add(
                jnp.float64(1.0),
                fortran_multiply(
                    jnp.float64(0.15),
                    fortran_power(reynolds, jnp.float64(0.687)),
                ),
            ),
        )
        drag = jnp.where(
            reynolds < jnp.float64(0.1),
            drag_stokes,
            jnp.where(
                reynolds < jnp.float64(2.0),
                drag_transition,
                jnp.where(reynolds < jnp.float64(500.0), drag_intermediate, jnp.float64(0.44)),
            ),
        )
        numerator = fortran_multiply(
            fortran_multiply(
                fortran_multiply(
                    fortran_multiply(jnp.float64(4.0), jnp.float64(GRAVITY)),
                    mass_weighted_diameter,
                ),
                slip,
            ),
            aerosol_density,
        )
        denominator = fortran_multiply(
            fortran_multiply(jnp.float64(3.0), drag),
            air_density,
        )
        candidate = jnp.sqrt(fortran_divide(numerator, denominator))
        candidate_error = jnp.abs(
            fortran_divide(fortran_add(candidate, -old_velocity), candidate)
        )
        averaged = fortran_multiply(
            jnp.float64(0.5),
            fortran_add(candidate, old_velocity),
        )
        candidate = jnp.where(iteration_index == jnp.float64(12.0), averaged, candidate)
        new_velocity = jnp.where(active, candidate, velocity)
        new_error = jnp.where(active, candidate_error, jnp.float64(0.0))
        forced_exit = iteration_index > jnp.float64(20.0)
        still_active = active & (candidate_error > jnp.float64(1.0e-4)) & ~forced_exit
        new_index = jnp.where(still_active, iteration_index + jnp.float64(1.0), iteration_index)
        return new_velocity, new_error, new_index, still_active

    terminal_velocity, _, _, _ = lax.while_loop(continue_iteration, iteration, initial)
    return fortran_divide(terminal_velocity, stokes_velocity)


def initialize_dust_parameters() -> DustParameters:

    source_mass_median = jnp.array([0.832e-6, 4.82e-6, 19.38e-6], dtype=jnp.float64)
    source_geometric_stddev = jnp.array([2.10, 1.90, 1.60], dtype=jnp.float64)
    source_mass_fraction = jnp.array([0.036, 0.957, 0.007], dtype=jnp.float64)
    diameter_edges = jnp.array([0.1e-6, 1.0e-6, 2.5e-6, 5.0e-6, 10.0e-6], dtype=jnp.float64)

    logarithmic_width = fortran_multiply(
        jnp.sqrt(jnp.float64(2.0)),
        _elementwise_log(source_geometric_stddev),
    )
    upper_argument = fortran_divide(
        _elementwise_log(
            fortran_divide(diameter_edges[1:][None, :], source_mass_median[:, None])
        ),
        logarithmic_width[:, None],
    )
    lower_argument = fortran_divide(
        _elementwise_log(
            fortran_divide(diameter_edges[:-1][None, :], source_mass_median[:, None])
        ),
        logarithmic_width[:, None],
    )
    overlap_fraction = fortran_multiply(
        jnp.float64(0.5),
        fortran_add(erf(upper_argument), -erf(lower_argument)),
    )
    source_to_sink = fortran_multiply(overlap_fraction, source_mass_fraction[:, None])

    optimal_diameter = jnp.float64(75.0e-6)
    saltation_density = jnp.float64(2650.0)
    reynolds_approximation = fortran_add(
        jnp.float64(0.38),
        fortran_multiply(
            jnp.float64(1331.0),
            fortran_power(
                fortran_multiply(jnp.float64(100.0), optimal_diameter),
                jnp.float64(1.56),
            ),
        ),
    )
    low_reynolds_factor = fortran_divide(
        fortran_multiply(jnp.float64(0.1291), jnp.float64(0.1291)),
        fortran_add(
            jnp.float64(-1.0),
            fortran_multiply(
                jnp.float64(1.928),
                fortran_power(reynolds_approximation, jnp.float64(0.0922)),
            ),
        ),
    )
    high_reynolds_base = fortran_add(
        jnp.float64(1.0),
        -fortran_multiply(
            jnp.float64(0.0858),
            strict_exp(
                fortran_multiply(
                    jnp.float64(-0.0617),
                    fortran_add(reynolds_approximation, jnp.float64(-10.0)),
                )
            ),
        ),
    )
    high_reynolds_factor = fortran_multiply(
        fortran_multiply(jnp.float64(0.120), jnp.float64(0.120)),
        fortran_multiply(high_reynolds_base, high_reynolds_base),
    )
    reynolds_factor = jnp.where(
        reynolds_approximation < jnp.float64(10.0),
        low_reynolds_factor,
        high_reynolds_factor,
    )
    cohesive_factor = fortran_add(
        jnp.float64(1.0),
        fortran_divide(
            jnp.float64(6.0e-7),
            fortran_multiply(
                fortran_multiply(saltation_density, jnp.float64(GRAVITY)),
                fortran_power(optimal_diameter, jnp.float64(2.5)),
            ),
        ),
    )
    density_factor = fortran_multiply(
        fortran_multiply(saltation_density, jnp.float64(GRAVITY)),
        optimal_diameter,
    )
    threshold_numerator = jnp.sqrt(
        fortran_multiply(
            fortran_multiply(cohesive_factor, density_factor),
            reynolds_factor,
        )
    )

    logarithmic_stddev = _elementwise_log(jnp.float64(2.0))
    mass_median = jnp.float64(3.5000e-6)
    number_median = fortran_multiply(
        mass_median,
        strict_exp(
            -fortran_multiply(
                jnp.float64(3.0),
                fortran_multiply(logarithmic_stddev, logarithmic_stddev),
            )
        ),
    )
    mass_weighted_diameter = _particle_grid_statistics(
        diameter_edges[:-1],
        diameter_edges[1:],
        number_median,
        logarithmic_stddev,
    )
    aerosol_density = jnp.float64(2.5e3)
    stokes_correction = _stokes_settling_correction(
        mass_weighted_diameter,
        aerosol_density,
    )
    return DustParameters(
        source_to_sink_mass_overlap=source_to_sink,
        mass_weighted_diameter=mass_weighted_diameter,
        stokes_settling_correction=stokes_correction,
        saltation_threshold_numerator=threshold_numerator,
        aerosol_density=aerosol_density,
    )


def initialize_dust_state(topology: SubgridTopology) -> DustState:

    number_of_patches = topology.patch.column.shape[0]
    number_of_columns = topology.column.gridcell.shape[0]
    patch_bins = (number_of_patches, NUMBER_OF_DUST_BINS)
    special_patch = jnp.full(number_of_patches, jnp.float64(CLM_SPECIAL_VALUE))
    basin = jnp.where(
        jnp.asarray(topology.column.landunit_type, dtype=jnp.int32) != LANDUNIT_LAKE,
        jnp.float64(1.0),
        jnp.float64(jnp.nan),
    )
    return DustState(
        emission_flux_patch=jnp.full(patch_bins, jnp.float64(jnp.nan)),
        total_emission_flux_patch=special_patch,
        turbulent_deposition_velocity_patch=jnp.full(patch_bins, jnp.float64(jnp.nan)),
        turbulent_deposition_velocity_1_patch=special_patch,
        turbulent_deposition_velocity_2_patch=special_patch,
        turbulent_deposition_velocity_3_patch=special_patch,
        turbulent_deposition_velocity_4_patch=special_patch,
        mobile_basin_factor_column=jnp.broadcast_to(basin, (number_of_columns,)),
    )


def _validate_emission_shapes(
    topology: SubgridTopology,
    state: DustState,
    inputs: DustEmissionInputs,
) -> None:
    patch_shape = topology.patch.column.shape
    column_shape = topology.column.gridcell.shape
    patch_fields = (
        inputs.nonlake_patch,
        inputs.leaf_area_index_patch,
        inputs.stem_area_index_patch,
        inputs.dust_friction_velocity_patch,
        inputs.dust_wind_10m_patch,
    )
    column_fields = (
        inputs.air_density_column,
        inputs.gravimetric_water_threshold_column,
        inputs.clay_mass_fraction_limited_column,
        inputs.saturated_water_content_top_column,
        inputs.snow_cover_fraction_column,
        inputs.volumetric_soil_water_top_column,
        inputs.soil_liquid_water_top_column,
        inputs.soil_ice_top_column,
        state.mobile_basin_factor_column,
    )
    if any(jnp.shape(field) != patch_shape for field in patch_fields):
        raise ValueError("dust-emission patch inputs must contain one value per patch")
    if any(jnp.shape(field) != column_shape for field in column_fields):
        raise ValueError("dust-emission column inputs must contain one value per column")
    if jnp.shape(state.emission_flux_patch) != (patch_shape[0], NUMBER_OF_DUST_BINS):
        raise ValueError("dust emission state must contain four bins per patch")


def _patch_to_landunit_mean(
    topology: SubgridTopology,
    values: jnp.ndarray,
    *,
    maximum_patches_per_landunit: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    values = jnp.asarray(values, dtype=jnp.float64)
    landunit = jnp.asarray(topology.patch.landunit, dtype=jnp.int32)
    weight = jnp.asarray(topology.patch.weight_on_landunit, dtype=jnp.float64)
    active = jnp.asarray(topology.patch.active, dtype=jnp.bool_)
    number_of_patches = values.shape[0]
    number_of_landunits = topology.landunit.gridcell.shape[0]
    if number_of_patches == 0:
        return (
            jnp.full(number_of_landunits, jnp.float64(CLM_SPECIAL_VALUE)),
            jnp.zeros(number_of_landunits, dtype=jnp.float64),
        )
    patch_index = jnp.arange(number_of_patches, dtype=jnp.int32)
    first_patch = jnp.full(number_of_landunits, number_of_patches, dtype=jnp.int32)
    first_patch = first_patch.at[landunit].min(patch_index)
    landunit_index = jnp.arange(number_of_landunits, dtype=jnp.int32)

    def accumulate(offset: int, carry):
        totals, sum_weights = carry
        unbounded = first_patch + jnp.int32(offset)
        bounded = jnp.clip(unbounded, 0, number_of_patches - 1)
        belongs = (unbounded < number_of_patches) & (landunit[bounded] == landunit_index)
        selected_value = values[bounded]
        selected_weight = weight[bounded]
        contributes = (
            belongs
            & active[bounded]
            & (selected_weight != jnp.float64(0.0))
            & (selected_value != jnp.float64(CLM_SPECIAL_VALUE))
        )
        contribution = jnp.where(
            contributes,
            fortran_multiply(selected_value, selected_weight),
            jnp.float64(0.0),
        )
        contribution_weight = jnp.where(contributes, selected_weight, jnp.float64(0.0))
        return (
            fortran_add(totals, contribution),
            fortran_add(sum_weights, contribution_weight),
        )

    totals, sum_weights = lax.fori_loop(
        0,
        maximum_patches_per_landunit,
        accumulate,
        (
            jnp.zeros(number_of_landunits, dtype=jnp.float64),
            jnp.zeros(number_of_landunits, dtype=jnp.float64),
        ),
    )
    has_weight = sum_weights != jnp.float64(0.0)
    means = jnp.where(
        has_weight,
        fortran_divide(totals, jnp.where(has_weight, sum_weights, jnp.float64(1.0))),
        jnp.float64(CLM_SPECIAL_VALUE),
    )
    return means, sum_weights


def dust_emission(
    topology: SubgridTopology,
    state: DustState,
    inputs: DustEmissionInputs,
    parameters: DustParameters,
    *,
    execution: DustExecutionParameters = DEFAULT_DUST_EXECUTION_PARAMETERS,
) -> DustEmissionResult:

    _validate_emission_shapes(topology, state, inputs)
    patch = topology.patch
    column_index = jnp.asarray(patch.column, dtype=jnp.int32)
    landunit_index = jnp.asarray(patch.landunit, dtype=jnp.int32)
    nonlake = jnp.asarray(inputs.nonlake_patch, dtype=jnp.bool_)
    total_vegetation_area = jnp.where(
        nonlake,
        fortran_add(inputs.leaf_area_index_patch, inputs.stem_area_index_patch),
        jnp.float64(0.0),
    )
    vegetation_area_landunit, vegetation_weight_landunit = _patch_to_landunit_mean(
        topology,
        total_vegetation_area,
        maximum_patches_per_landunit=execution.maximum_patches_per_landunit,
    )
    selected_vegetation_area = vegetation_area_landunit[landunit_index]
    soil_or_crop = (
        (jnp.asarray(patch.landunit_type, dtype=jnp.int32) == LANDUNIT_SOIL)
        | (jnp.asarray(patch.landunit_type, dtype=jnp.int32) == LANDUNIT_CROP)
    )
    below_threshold = selected_vegetation_area < jnp.float64(0.3)
    bare_fraction = jnp.where(
        below_threshold,
        fortran_add(
            jnp.float64(1.0),
            -fortran_divide(selected_vegetation_area, jnp.float64(0.3)),
        ),
        jnp.float64(0.0),
    )
    snow_free = fortran_add(
        jnp.float64(1.0),
        -jnp.asarray(inputs.snow_cover_fraction_column, dtype=jnp.float64)[column_index],
    )
    mobile_fraction = jnp.where(
        nonlake,
        jnp.where(
            soil_or_crop,
            fortran_multiply(bare_fraction, snow_free),
            jnp.float64(0.0),
        ),
        jnp.float64(CLM_SPECIAL_VALUE),
    )
    mobile = nonlake & (mobile_fraction > jnp.float64(0.0))

    def selected_column(field: jnp.ndarray, fallback: float = 1.0) -> jnp.ndarray:
        selected = jnp.asarray(field, dtype=jnp.float64)[column_index]
        return jnp.where(mobile, selected, jnp.float64(fallback))

    saturation = selected_column(inputs.saturated_water_content_top_column, 0.4)
    bulk_density = fortran_multiply(
        fortran_add(jnp.float64(1.0), -saturation),
        jnp.float64(2.7e3),
    )
    gravimetric_water = fortran_divide(
        fortran_multiply(
            selected_column(inputs.volumetric_soil_water_top_column, 0.0),
            jnp.float64(DENSITY_FRESH_WATER),
        ),
        bulk_density,
    )
    water_threshold = selected_column(inputs.gravimetric_water_threshold_column, 0.0)
    water_excess = fortran_add(gravimetric_water, -water_threshold)
    has_water_excess = gravimetric_water > water_threshold
    wet_power_base = fortran_multiply(
        jnp.float64(100.0),
        jnp.where(has_water_excess, water_excess, jnp.float64(1.0)),
    )
    wet_threshold = jnp.where(
        has_water_excess,
        jnp.sqrt(
            fortran_add(
                jnp.float64(1.0),
                fortran_multiply(
                    jnp.float64(1.21),
                    fortran_power(wet_power_base, jnp.float64(0.68)),
                ),
            )
        ),
        jnp.float64(1.0),
    )
    liquid = selected_column(inputs.soil_liquid_water_top_column, 0.0)
    ice = selected_column(inputs.soil_ice_top_column, 0.0)
    liquid_fraction = jnp.maximum(
        jnp.float64(0.0),
        jnp.minimum(
            jnp.float64(1.0),
            fortran_divide(
                liquid,
                fortran_ordered_sum(
                    jnp.stack((ice, liquid, jnp.full_like(liquid, 1.0e-6)), axis=0)
                ),
            ),
        ),
    )
    air_density = selected_column(inputs.air_density_column, 1.0)
    threshold_friction = fortran_multiply(
        fortran_divide(parameters.saltation_threshold_numerator, jnp.sqrt(air_density)),
        wet_threshold,
    )
    friction = jnp.where(
        mobile,
        jnp.asarray(inputs.dust_friction_velocity_patch, dtype=jnp.float64),
        jnp.float64(1.0),
    )
    wind_10m = jnp.where(
        mobile,
        jnp.asarray(inputs.dust_wind_10m_patch, dtype=jnp.float64),
        jnp.float64(0.0),
    )
    threshold_reference_wind = fortran_divide(
        fortran_multiply(wind_10m, threshold_friction),
        friction,
    )
    wind_excess = fortran_add(wind_10m, -threshold_reference_wind)
    saltation_increment = fortran_multiply(
        fortran_multiply(jnp.float64(0.003), wind_excess),
        wind_excess,
    )
    saltating_friction = jnp.where(
        wind_10m >= threshold_reference_wind,
        fortran_add(friction, saltation_increment),
        friction,
    )
    exceeds_friction_threshold = mobile & (saltating_friction > threshold_friction)
    friction_ratio = fortran_divide(threshold_friction, saltating_friction)
    horizontal_flux = fortran_divide(
        fortran_multiply(
            fortran_multiply(
                fortran_multiply(
                    fortran_multiply(
                        fortran_multiply(
                            jnp.float64(2.61),
                            air_density,
                        ),
                        fortran_power(saltating_friction, jnp.float64(3.0)),
                    ),
                    fortran_add(jnp.float64(1.0), -friction_ratio),
                ),
                fortran_add(jnp.float64(1.0), friction_ratio),
            ),
            fortran_add(jnp.float64(1.0), friction_ratio),
        ),
        jnp.float64(GRAVITY),
    )
    horizontal_flux = jnp.where(exceeds_friction_threshold, horizontal_flux, jnp.float64(0.0))
    basin = jnp.asarray(state.mobile_basin_factor_column, dtype=jnp.float64)[column_index]
    horizontal_flux = fortran_multiply(horizontal_flux, mobile_fraction)
    horizontal_flux = fortran_multiply(horizontal_flux, basin)
    horizontal_flux = fortran_multiply(horizontal_flux, jnp.float64(5.0e-4))
    horizontal_flux = fortran_multiply(horizontal_flux, liquid_fraction)

    clay = selected_column(inputs.clay_mass_fraction_limited_column, 0.0)
    clay_exponent = fortran_multiply(
        _elementwise_log(jnp.float64(10.0)),
        fortran_add(
            fortran_multiply(jnp.float64(13.4), clay),
            jnp.float64(-6.0),
        ),
    )
    vertical_ratio = fortran_multiply(jnp.float64(100.0), strict_exp(clay_exponent))
    total_vertical_flux = fortran_multiply(horizontal_flux, vertical_ratio)
    total_vertical_flux = jnp.where(mobile, total_vertical_flux, jnp.float64(0.0))

    def add_source(source_index: int, bins: jnp.ndarray) -> jnp.ndarray:
        contribution = fortran_multiply(
            parameters.source_to_sink_mass_overlap[source_index][None, :],
            total_vertical_flux[:, None],
        )
        return jnp.where(mobile[:, None], fortran_add(bins, contribution), bins)

    emission = lax.fori_loop(
        0,
        NUMBER_OF_DUST_SOURCE_MODES,
        add_source,
        jnp.zeros((column_index.shape[0], NUMBER_OF_DUST_BINS), dtype=jnp.float64),
    )

    def add_bin(bin_index: int, total: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(mobile, fortran_add(total, emission[:, bin_index]), total)

    total_emission = lax.fori_loop(
        0,
        NUMBER_OF_DUST_BINS,
        add_bin,
        jnp.zeros(column_index.shape[0], dtype=jnp.float64),
    )
    updated_total = jnp.where(nonlake, total_emission, state.total_emission_flux_patch)
    updated_state = state._replace(
        emission_flux_patch=emission,
        total_emission_flux_patch=updated_total,
    )

    def diagnostic(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(mobile, value, jnp.float64(CLM_SPECIAL_VALUE))

    return DustEmissionResult(
        state=updated_state,
        diagnostics=DustEmissionDiagnostics(
            total_vegetation_area_patch=total_vegetation_area,
            vegetation_area_landunit=vegetation_area_landunit,
            vegetation_weight_landunit=vegetation_weight_landunit,
            mobile_land_fraction_patch=mobile_fraction,
            dry_bulk_density_patch=diagnostic(bulk_density),
            surface_gravimetric_water_patch=diagnostic(gravimetric_water),
            wet_threshold_factor_patch=diagnostic(wet_threshold),
            liquid_water_fraction_patch=diagnostic(liquid_fraction),
            threshold_friction_velocity_patch=diagnostic(threshold_friction),
            threshold_reference_wind_patch=diagnostic(threshold_reference_wind),
            saltating_friction_velocity_patch=diagnostic(saltating_friction),
            horizontal_saltation_flux_patch=diagnostic(horizontal_flux),
            vertical_to_horizontal_flux_ratio_patch=diagnostic(vertical_ratio),
            total_vertical_dust_flux_patch=diagnostic(total_vertical_flux),
        ),
    )


def dust_dry_deposition(
    topology: SubgridTopology,
    state: DustState,
    inputs: DustDepositionInputs,
    parameters: DustParameters,
) -> DustDepositionResult:

    patch_shape = topology.patch.column.shape
    column_shape = topology.column.gridcell.shape
    patch_fields = (inputs.aerodynamic_resistance_patch, inputs.dust_friction_velocity_patch)
    column_fields = (
        inputs.air_pressure_column,
        inputs.air_density_column,
        inputs.air_temperature_column,
    )
    if any(jnp.shape(field) != patch_shape for field in patch_fields):
        raise ValueError("dust-deposition patch inputs must contain one value per patch")
    if any(jnp.shape(field) != column_shape for field in column_fields):
        raise ValueError("dust-deposition column inputs must contain one value per column")

    active = jnp.asarray(topology.patch.active, dtype=jnp.bool_)
    column = jnp.asarray(topology.patch.column, dtype=jnp.int32)
    temperature = jnp.asarray(inputs.air_temperature_column, dtype=jnp.float64)[column]
    pressure = jnp.asarray(inputs.air_pressure_column, dtype=jnp.float64)[column]
    density = jnp.asarray(inputs.air_density_column, dtype=jnp.float64)[column]
    temperature = jnp.where(active, temperature, jnp.float64(295.0))
    pressure = jnp.where(active, pressure, jnp.float64(100000.0))
    density = jnp.where(active, density, jnp.float64(1.0))
    friction = jnp.where(
        active,
        jnp.asarray(inputs.dust_friction_velocity_patch, dtype=jnp.float64),
        jnp.float64(1.0),
    )
    resistance = jnp.where(
        active,
        jnp.asarray(inputs.aerodynamic_resistance_patch, dtype=jnp.float64),
        jnp.float64(1.0),
    )

    dynamic_viscosity = fortran_divide(
        fortran_multiply(
            fortran_multiply(
                jnp.float64(1.72e-5),
                fortran_power(
                    fortran_divide(temperature, jnp.float64(273.0)),
                    jnp.float64(1.5),
                ),
            ),
            jnp.float64(393.0),
        ),
        fortran_add(temperature, jnp.float64(120.0)),
    )
    mean_free_path = fortran_divide(
        fortran_multiply(jnp.float64(2.0), dynamic_viscosity),
        fortran_multiply(
            pressure,
            jnp.sqrt(
                fortran_divide(
                    jnp.float64(8.0),
                    fortran_multiply(
                        fortran_multiply(jnp.float64(PI), jnp.float64(GAS_CONSTANT_DRY_AIR)),
                        temperature,
                    ),
                )
            ),
        ),
    )
    kinematic_viscosity = fortran_divide(dynamic_viscosity, density)
    diameter = jnp.asarray(parameters.mass_weighted_diameter, dtype=jnp.float64)[None, :]
    mean_free_path_bin = mean_free_path[:, None]
    slip = fortran_add(
        jnp.float64(1.0),
        fortran_divide(
            fortran_multiply(
                fortran_multiply(jnp.float64(2.0), mean_free_path_bin),
                fortran_add(
                    jnp.float64(1.257),
                    fortran_multiply(
                        jnp.float64(0.4),
                        strict_exp(
                            fortran_divide(
                                fortran_multiply(jnp.float64(-1.1), diameter),
                                fortran_multiply(jnp.float64(2.0), mean_free_path_bin),
                            )
                        ),
                    ),
                ),
            ),
            diameter,
        ),
    )
    settling = fortran_divide(jnp.float64(1.0), jnp.float64(18.0))
    settling = fortran_multiply(settling, diameter)
    settling = fortran_multiply(settling, diameter)
    settling = fortran_multiply(settling, parameters.aerosol_density)
    settling = fortran_multiply(settling, jnp.float64(GRAVITY))
    settling = fortran_multiply(settling, slip)
    settling = fortran_divide(settling, dynamic_viscosity[:, None])
    settling = fortran_multiply(settling, parameters.stokes_settling_correction[None, :])
    stokes_number = fortran_divide(
        fortran_multiply(
            fortran_multiply(settling, friction[:, None]),
            friction[:, None],
        ),
        fortran_multiply(jnp.float64(GRAVITY), kinematic_viscosity[:, None]),
    )
    aerosol_diffusivity = fortran_divide(
        fortran_multiply(
            fortran_multiply(jnp.float64(BOLTZMANN), temperature[:, None]),
            slip,
        ),
        fortran_multiply(
            fortran_multiply(
                fortran_multiply(jnp.float64(3.0), jnp.float64(PI)),
                dynamic_viscosity[:, None],
            ),
            diameter,
        ),
    )
    schmidt = fortran_divide(kinematic_viscosity[:, None], aerosol_diffusivity)
    collection_efficiency = fortran_add(
        fortran_power(
            schmidt,
            fortran_divide(jnp.float64(-2.0), jnp.float64(3.0)),
        ),
        fortran_power(
            jnp.float64(10.0),
            fortran_divide(jnp.float64(-3.0), stokes_number),
        ),
    )
    quasi_laminar_resistance = fortran_divide(
        jnp.float64(1.0),
        fortran_multiply(collection_efficiency, friction[:, None]),
    )
    turbulent_resistance = fortran_ordered_sum(
        jnp.stack(
            (
                jnp.broadcast_to(resistance[:, None], quasi_laminar_resistance.shape),
                quasi_laminar_resistance,
                fortran_multiply(
                    fortran_multiply(resistance[:, None], quasi_laminar_resistance),
                    settling,
                ),
            ),
            axis=0,
        )
    )
    velocity = fortran_divide(jnp.float64(1.0), turbulent_resistance)
    old_velocity = jnp.asarray(state.turbulent_deposition_velocity_patch, dtype=jnp.float64)
    updated_velocity = jnp.where(active[:, None], velocity, old_velocity)
    scalar = tuple(
        jnp.where(active, updated_velocity[:, index], getattr(state, field))
        for index, field in enumerate(
            (
                "turbulent_deposition_velocity_1_patch",
                "turbulent_deposition_velocity_2_patch",
                "turbulent_deposition_velocity_3_patch",
                "turbulent_deposition_velocity_4_patch",
            )
        )
    )
    updated_state = state._replace(
        turbulent_deposition_velocity_patch=updated_velocity,
        turbulent_deposition_velocity_1_patch=scalar[0],
        turbulent_deposition_velocity_2_patch=scalar[1],
        turbulent_deposition_velocity_3_patch=scalar[2],
        turbulent_deposition_velocity_4_patch=scalar[3],
    )

    def diagnostic(value: jnp.ndarray) -> jnp.ndarray:
        mask = active if value.ndim == 1 else active[:, None]
        return jnp.where(mask, value, jnp.float64(CLM_SPECIAL_VALUE))

    return DustDepositionResult(
        state=updated_state,
        diagnostics=DustDepositionDiagnostics(
            dynamic_air_viscosity_patch=diagnostic(dynamic_viscosity),
            mean_free_path_patch=diagnostic(mean_free_path),
            kinematic_air_viscosity_patch=diagnostic(kinematic_viscosity),
            slip_correction_patch=diagnostic(slip),
            gravitational_settling_velocity_patch=diagnostic(settling),
            stokes_number_patch=diagnostic(stokes_number),
            aerosol_diffusivity_patch=diagnostic(aerosol_diffusivity),
            schmidt_number_patch=diagnostic(schmidt),
            collection_efficiency_patch=diagnostic(collection_efficiency),
            quasi_laminar_resistance_patch=diagnostic(quasi_laminar_resistance),
            turbulent_resistance_patch=diagnostic(turbulent_resistance),
        ),
    )


def dust_emission_and_deposition(
    topology: SubgridTopology,
    state: DustState,
    emission_inputs: DustEmissionInputs,
    deposition_inputs: DustDepositionInputs,
    parameters: DustParameters,
    *,
    execution: DustExecutionParameters = DEFAULT_DUST_EXECUTION_PARAMETERS,
) -> DustStepResult:

    emission = dust_emission(
        topology,
        state,
        emission_inputs,
        parameters,
        execution=execution,
    )
    deposition = dust_dry_deposition(
        topology,
        emission.state,
        deposition_inputs,
        parameters,
    )
    return DustStepResult(
        state=deposition.state,
        emission=emission.diagnostics,
        deposition=deposition.diagnostics,
    )
