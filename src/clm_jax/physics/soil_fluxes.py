
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import lax

from clm_jax.constants import (
    FREEZING_TEMPERATURE_FRESHWATER,
    LATENT_HEAT_VAPORIZATION,
    STEFAN_BOLTZMANN,
)
from clm_jax.io.restart import (
    COLUMN_URBAN_ROOF,
    COLUMN_URBAN_SHADE_WALL,
    COLUMN_URBAN_SUN_WALL,
    LANDUNIT_CROP,
    LANDUNIT_LAKE,
    LANDUNIT_SOIL,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    SubgridTopology,
)
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
)

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase

NUMBER_OF_SNOW_LEVELS = 12
NUMBER_OF_GROUND_LEVELS = 25
NUMBER_OF_SOIL_SNOW_LEVELS = NUMBER_OF_SNOW_LEVELS + NUMBER_OF_GROUND_LEVELS
NUMBER_OF_URBAN_LEVELS = 5
GROUND_SURFACE_INDEX = NUMBER_OF_SNOW_LEVELS


class SoilFluxesParameters(NamedTuple):

    time_step_seconds: float = 1800.0
    maximum_patches_per_column: int = 79

    @classmethod
    def from_case(cls, case: CimeCase) -> SoilFluxesParameters:

        from clm_jax.io.namelist import read_case_namelist_group  
        from clm_jax.io.restart import (  
            configured_land_restart,
            load_restart_clock,
        )

        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        clock = load_restart_clock(configured_land_restart(case))
        return cls(
            time_step_seconds=float(clock.step_seconds),
            maximum_patches_per_column=int(general["maxpatch_pft"]),
        )


DEFAULT_SOIL_FLUXES_PARAMETERS = SoilFluxesParameters()


class SoilFluxesInputs(NamedTuple):

    snow_layer_count_column: jnp.ndarray
    downward_longwave_column: jnp.ndarray
    effective_snow_fraction_column: jnp.ndarray
    surface_water_fraction_column: jnp.ndarray
    liquid_water_column_level: jnp.ndarray
    ice_water_column_level: jnp.ndarray
    ground_emissivity_column: jnp.ndarray
    surface_water_temperature_column: jnp.ndarray
    previous_surface_water_temperature_column: jnp.ndarray
    previous_soil_snow_temperature_column_level: jnp.ndarray
    ground_temperature_column: jnp.ndarray
    soil_snow_temperature_column_level: jnp.ndarray
    phase_change_heat_flux_column: jnp.ndarray
    surface_water_phase_change_heat_flux_column: jnp.ndarray
    temperature_factor_column_level: jnp.ndarray
    surface_water_heat_capacity_column: jnp.ndarray
    latent_heat_column: jnp.ndarray
    surface_water_to_snow_heat_flux_column: jnp.ndarray
    building_heat_error_column: jnp.ndarray
    vegetation_exposed_patch: jnp.ndarray
    absorbed_soil_patch: jnp.ndarray
    absorbed_snow_patch: jnp.ndarray
    absorbed_ground_patch: jnp.ndarray
    downward_longwave_below_canopy_patch: jnp.ndarray
    upward_longwave_above_canopy_patch: jnp.ndarray
    ground_sensible_derivative_patch: jnp.ndarray
    ground_latent_derivative_patch: jnp.ndarray
    vegetation_transpiration_patch: jnp.ndarray
    vegetation_evaporation_patch: jnp.ndarray
    vegetation_sensible_heat_patch: jnp.ndarray
    waste_heat_patch: jnp.ndarray
    air_conditioner_heat_patch: jnp.ndarray
    traffic_heat_patch: jnp.ndarray


class SoilFluxesWaterState(NamedTuple):

    soil_evaporation_patch: jnp.ndarray
    snow_evaporation_patch: jnp.ndarray
    bare_soil_evaporation_patch: jnp.ndarray
    surface_water_evaporation_patch: jnp.ndarray
    canopy_evaporation_patch: jnp.ndarray
    total_evaporation_patch: jnp.ndarray
    ground_liquid_evaporation_patch: jnp.ndarray
    snow_sublimation_patch: jnp.ndarray
    snow_dew_patch: jnp.ndarray
    ground_dew_patch: jnp.ndarray


class SoilFluxesEnergyState(NamedTuple):

    ground_sensible_heat_patch: jnp.ndarray
    soil_ground_heat_patch: jnp.ndarray
    urban_soil_ground_heat_patch: jnp.ndarray
    rural_soil_ground_heat_patch: jnp.ndarray
    total_sensible_heat_patch: jnp.ndarray
    urban_total_sensible_heat_patch: jnp.ndarray
    rural_total_sensible_heat_patch: jnp.ndarray
    total_latent_heat_patch: jnp.ndarray
    urban_total_latent_heat_patch: jnp.ndarray
    rural_total_latent_heat_patch: jnp.ndarray
    outgoing_longwave_patch: jnp.ndarray
    net_longwave_patch: jnp.ndarray
    rural_net_longwave_patch: jnp.ndarray
    rural_outgoing_longwave_patch: jnp.ndarray
    urban_net_longwave_patch: jnp.ndarray
    urban_outgoing_longwave_patch: jnp.ndarray
    vegetation_evaporation_latent_heat_patch: jnp.ndarray
    vegetation_transpiration_latent_heat_patch: jnp.ndarray
    ground_latent_heat_patch: jnp.ndarray
    soil_energy_error_patch: jnp.ndarray
    soil_energy_error_column: jnp.ndarray


class SoilFluxesState(NamedTuple):

    skin_temperature_patch: jnp.ndarray
    water: SoilFluxesWaterState
    energy: SoilFluxesEnergyState


class SoilFluxesDiagnostics(NamedTuple):

    previous_ground_temperature_column: jnp.ndarray
    ground_temperature_increment_column: jnp.ndarray
    maximum_top_layer_evaporation_column: jnp.ndarray
    requested_top_layer_evaporation_column: jnp.ndarray
    evaporation_availability_ratio_column: jnp.ndarray
    ground_longwave_temperature_fourth_column: jnp.ndarray
    longwave_temperature_correction_patch: jnp.ndarray


class SoilFluxesResult(NamedTuple):
    state: SoilFluxesState
    diagnostics: SoilFluxesDiagnostics


def _validate_soil_fluxes_shapes(
    topology: SubgridTopology,
    inputs: SoilFluxesInputs,
    state: SoilFluxesState,
    parameters: SoilFluxesParameters,
) -> None:
    column_shape = topology.column.gridcell.shape
    patch_shape = topology.patch.column.shape
    layer_shape = (column_shape[0], NUMBER_OF_SOIL_SNOW_LEVELS)
    if len(column_shape) != 1 or len(patch_shape) != 1:
        raise ValueError("column and patch topology must be vectors")
    if parameters.maximum_patches_per_column < 0:
        raise ValueError("maximum_patches_per_column must be nonnegative")

    layer_fields = {
        "liquid_water_column_level",
        "ice_water_column_level",
        "previous_soil_snow_temperature_column_level",
        "soil_snow_temperature_column_level",
        "temperature_factor_column_level",
    }
    patch_fields = {
        "vegetation_exposed_patch",
        "absorbed_soil_patch",
        "absorbed_snow_patch",
        "absorbed_ground_patch",
        "downward_longwave_below_canopy_patch",
        "upward_longwave_above_canopy_patch",
        "ground_sensible_derivative_patch",
        "ground_latent_derivative_patch",
        "vegetation_transpiration_patch",
        "vegetation_evaporation_patch",
        "vegetation_sensible_heat_patch",
        "waste_heat_patch",
        "air_conditioner_heat_patch",
        "traffic_heat_patch",
    }
    for name in SoilFluxesInputs._fields:
        if name in layer_fields:
            expected = layer_shape
        elif name in patch_fields:
            expected = patch_shape
        else:
            expected = column_shape
        if getattr(inputs, name).shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if state.skin_temperature_patch.shape != patch_shape:
        raise ValueError(f"skin_temperature_patch must have shape {patch_shape}")
    for name in SoilFluxesWaterState._fields:
        if getattr(state.water, name).shape != patch_shape:
            raise ValueError(f"{name} must have shape {patch_shape}")
    for name in SoilFluxesEnergyState._fields:
        expected = column_shape if name == "soil_energy_error_column" else patch_shape
        if getattr(state.energy, name).shape != expected:
            raise ValueError(f"{name} must have shape {expected}")


def _square(value: jnp.ndarray) -> jnp.ndarray:
    return fortran_multiply(value, value)


def _cube(value: jnp.ndarray) -> jnp.ndarray:
    return fortran_multiply(_square(value), value)


def _fourth_power(value: jnp.ndarray) -> jnp.ndarray:
    squared = _square(value)
    return fortran_multiply(squared, squared)


def _ordered_patch_to_column_sum(
    topology: SubgridTopology,
    patch_values: jnp.ndarray,
    column_update_mask: jnp.ndarray,
    *,
    maximum_patches_per_column: int,
) -> jnp.ndarray:

    patch_values = jnp.asarray(patch_values, dtype=jnp.float64)
    patch_column = jnp.asarray(topology.patch.column, dtype=jnp.int32)
    patch_weight = jnp.asarray(topology.patch.weight_on_column, dtype=jnp.float64)
    patch_active = jnp.asarray(topology.patch.active, dtype=jnp.bool_)
    update = jnp.asarray(column_update_mask, dtype=jnp.bool_)
    number_of_patches = patch_values.shape[0]
    totals = jnp.zeros(update.shape, dtype=jnp.float64)
    if number_of_patches == 0:
        return totals

    patch_index = jnp.arange(number_of_patches, dtype=jnp.int32)
    first_patch = jnp.full(update.shape, number_of_patches, dtype=jnp.int32)
    first_patch = first_patch.at[patch_column].min(patch_index)
    column_index = jnp.arange(update.shape[0], dtype=jnp.int32)

    def accumulate(offset, column_totals):
        unbounded_patch = first_patch + jnp.int32(offset)
        bounded_patch = jnp.clip(unbounded_patch, 0, number_of_patches - 1)
        contributes = (
            update
            & (unbounded_patch < number_of_patches)
            & (patch_column[bounded_patch] == column_index)
            & patch_active[bounded_patch]
        )
        contribution = jnp.where(
            contributes,
            fortran_multiply(patch_values[bounded_patch], patch_weight[bounded_patch]),
            jnp.float64(0.0),
        )
        return fortran_add(column_totals, contribution)

    return lax.fori_loop(
        0,
        maximum_patches_per_column,
        accumulate,
        totals,
    )


def soil_fluxes(
    topology: SubgridTopology,
    inputs: SoilFluxesInputs,
    state: SoilFluxesState,
    *,
    parameters: SoilFluxesParameters = DEFAULT_SOIL_FLUXES_PARAMETERS,
) -> SoilFluxesResult:

    _validate_soil_fluxes_shapes(topology, inputs, state, parameters)

    column_active = jnp.asarray(topology.column.active, dtype=jnp.bool_)
    patch_active = jnp.asarray(topology.patch.active, dtype=jnp.bool_)
    column_landunit_type = jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
    patch_landunit_type = jnp.asarray(topology.patch.landunit_type, dtype=jnp.int32)
    column_type = jnp.asarray(topology.column.column_type, dtype=jnp.int32)
    patch_column = jnp.asarray(topology.patch.column, dtype=jnp.int32)

    patch_urban = (patch_landunit_type >= jnp.int32(LANDUNIT_URBAN_MIN)) & (
        patch_landunit_type <= jnp.int32(LANDUNIT_URBAN_MAX)
    )
    nonlake_column = column_active & (
        column_landunit_type != jnp.int32(LANDUNIT_LAKE)
    )
    nonlake_patch = patch_active & (
        patch_landunit_type != jnp.int32(LANDUNIT_LAKE)
    )
    urban_patch = nonlake_patch & patch_urban

    snow_layer_count = jnp.asarray(inputs.snow_layer_count_column, dtype=jnp.int32)
    first_active_index = snow_layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS)
    gather_index = first_active_index[:, None]
    previous_temperature = jnp.asarray(
        inputs.previous_soil_snow_temperature_column_level,
        dtype=jnp.float64,
    )
    previous_top_temperature = jnp.take_along_axis(
        previous_temperature,
        gather_index,
        axis=1,
    )[:, 0]
    previous_ground_layer_temperature = previous_temperature[:, GROUND_SURFACE_INDEX]
    snow_fraction = jnp.asarray(inputs.effective_snow_fraction_column, dtype=jnp.float64)
    water_fraction = jnp.asarray(inputs.surface_water_fraction_column, dtype=jnp.float64)
    previous_water_temperature = jnp.asarray(
        inputs.previous_surface_water_temperature_column,
        dtype=jnp.float64,
    )

    uncovered_fraction = fortran_add(
        fortran_add(jnp.float64(1.0), -snow_fraction),
        -water_fraction,
    )
    snow_previous_ground = fortran_add(
        fortran_add(
            fortran_multiply(snow_fraction, previous_top_temperature),
            fortran_multiply(uncovered_fraction, previous_ground_layer_temperature),
        ),
        fortran_multiply(water_fraction, previous_water_temperature),
    )
    snow_free_previous_ground = fortran_add(
        fortran_multiply(
            fortran_add(jnp.float64(1.0), -water_fraction),
            previous_ground_layer_temperature,
        ),
        fortran_multiply(water_fraction, previous_water_temperature),
    )
    previous_ground_temperature = jnp.where(
        snow_layer_count < jnp.int32(0),
        snow_previous_ground,
        snow_free_previous_ground,
    )
    ground_temperature = jnp.asarray(inputs.ground_temperature_column, dtype=jnp.float64)
    temperature_increment = fortran_add(ground_temperature, -previous_ground_temperature)

    liquid_water = jnp.asarray(inputs.liquid_water_column_level, dtype=jnp.float64)
    ice_water = jnp.asarray(inputs.ice_water_column_level, dtype=jnp.float64)
    top_liquid = jnp.take_along_axis(liquid_water, gather_index, axis=1)[:, 0]
    top_ice = jnp.take_along_axis(ice_water, gather_index, axis=1)[:, 0]
    top_water = fortran_add(top_ice, top_liquid)
    time_step_seconds = jnp.float64(parameters.time_step_seconds)
    maximum_evaporation = fortran_divide(top_water, time_step_seconds)
    maximum_evaporation = jnp.where(
        maximum_evaporation < jnp.float64(0.0),
        jnp.float64(0.0),
        maximum_evaporation,
    )

    patch_temperature_increment = temperature_increment[patch_column]
    sensible_correction = fortran_multiply(
        patch_temperature_increment,
        jnp.asarray(inputs.ground_sensible_derivative_patch, dtype=jnp.float64),
    )
    latent_correction = fortran_multiply(
        patch_temperature_increment,
        jnp.asarray(inputs.ground_latent_derivative_patch, dtype=jnp.float64),
    )
    ground_sensible_heat = jnp.where(
        nonlake_patch,
        fortran_add(state.energy.ground_sensible_heat_patch, sensible_correction),
        state.energy.ground_sensible_heat_patch,
    )
    soil_evaporation = jnp.where(
        nonlake_patch,
        fortran_add(state.water.soil_evaporation_patch, latent_correction),
        state.water.soil_evaporation_patch,
    )
    snow_evaporation = jnp.where(
        nonlake_patch,
        jnp.where(
            patch_urban,
            soil_evaporation,
            fortran_add(state.water.snow_evaporation_patch, latent_correction),
        ),
        state.water.snow_evaporation_patch,
    )
    bare_soil_evaporation = jnp.where(
        nonlake_patch,
        jnp.where(
            patch_urban,
            jnp.float64(0.0),
            fortran_add(state.water.bare_soil_evaporation_patch, latent_correction),
        ),
        state.water.bare_soil_evaporation_patch,
    )
    surface_water_evaporation = jnp.where(
        nonlake_patch,
        jnp.where(
            patch_urban,
            jnp.float64(0.0),
            fortran_add(state.water.surface_water_evaporation_patch, latent_correction),
        ),
        state.water.surface_water_evaporation_patch,
    )

    requested_evaporation = _ordered_patch_to_column_sum(
        topology,
        soil_evaporation,
        nonlake_column,
        maximum_patches_per_column=parameters.maximum_patches_per_column,
    )
    safe_requested = jnp.where(
        requested_evaporation != jnp.float64(0.0),
        requested_evaporation,
        jnp.float64(1.0),
    )
    availability_ratio = jnp.where(
        requested_evaporation > maximum_evaporation,
        fortran_divide(maximum_evaporation, safe_requested),
        jnp.float64(1.0),
    )
    rescale = nonlake_patch & (availability_ratio[patch_column] < jnp.float64(1.0))
    rescaled_soil_evaporation = fortran_multiply(
        soil_evaporation,
        availability_ratio[patch_column],
    )
    removed_evaporation = fortran_add(soil_evaporation, -rescaled_soil_evaporation)
    sensible_rescaling = fortran_multiply(
        removed_evaporation,
        jnp.asarray(inputs.latent_heat_column, dtype=jnp.float64)[patch_column],
    )
    soil_evaporation = jnp.where(rescale, rescaled_soil_evaporation, soil_evaporation)
    ground_sensible_heat = jnp.where(
        rescale,
        fortran_add(ground_sensible_heat, sensible_rescaling),
        ground_sensible_heat,
    )
    snow_evaporation = jnp.where(
        rescale,
        fortran_multiply(snow_evaporation, availability_ratio[patch_column]),
        snow_evaporation,
    )
    bare_soil_evaporation = jnp.where(
        rescale,
        fortran_multiply(bare_soil_evaporation, availability_ratio[patch_column]),
        bare_soil_evaporation,
    )
    surface_water_evaporation = jnp.where(
        rescale,
        fortran_multiply(surface_water_evaporation, availability_ratio[patch_column]),
        surface_water_evaporation,
    )

    snow_fraction_patch = snow_fraction[patch_column]
    emissivity_patch = jnp.asarray(inputs.ground_emissivity_column, dtype=jnp.float64)[
        patch_column
    ]
    previous_ground_patch = previous_ground_temperature[patch_column]
    longwave_ground_fourth_column = fortran_add(
        fortran_add(
            fortran_multiply(snow_fraction, _fourth_power(previous_top_temperature)),
            fortran_multiply(
                uncovered_fraction,
                _fourth_power(previous_ground_layer_temperature),
            ),
        ),
        fortran_multiply(water_fraction, _fourth_power(previous_water_temperature)),
    )
    longwave_ground_fourth = longwave_ground_fourth_column[patch_column]
    previous_ground_cubed = _cube(previous_ground_patch)
    four_temperature_increment = fortran_multiply(
        jnp.float64(4.0),
        patch_temperature_increment,
    )
    longwave_temperature_correction = fortran_multiply(
        fortran_multiply(
            fortran_multiply(
                fortran_multiply(jnp.float64(4.0), emissivity_patch),
                jnp.float64(STEFAN_BOLTZMANN),
            ),
            previous_ground_cubed,
        ),
        patch_temperature_increment,
    )

    vegetation_exposed = jnp.asarray(inputs.vegetation_exposed_patch, dtype=jnp.int32)
    one_minus_vegetation = fortran_add(
        jnp.float64(1.0),
        -vegetation_exposed.astype(jnp.float64),
    )
    absorbed_soil = jnp.asarray(inputs.absorbed_soil_patch, dtype=jnp.float64)
    absorbed_snow = jnp.asarray(inputs.absorbed_snow_patch, dtype=jnp.float64)
    downward_canopy = jnp.asarray(
        inputs.downward_longwave_below_canopy_patch,
        dtype=jnp.float64,
    )
    downward_atmosphere = jnp.asarray(inputs.downward_longwave_column, dtype=jnp.float64)[
        patch_column
    ]
    latent_heat_patch = jnp.asarray(inputs.latent_heat_column, dtype=jnp.float64)[
        patch_column
    ]
    transpiration = jnp.asarray(inputs.vegetation_transpiration_patch, dtype=jnp.float64)
    evaporation_vegetation = jnp.asarray(
        inputs.vegetation_evaporation_patch,
        dtype=jnp.float64,
    )
    sensible_vegetation = jnp.asarray(
        inputs.vegetation_sensible_heat_patch,
        dtype=jnp.float64,
    )

    solar_ground = fortran_add(
        fortran_multiply(
            fortran_add(jnp.float64(1.0), -snow_fraction_patch),
            absorbed_soil,
        ),
        fortran_multiply(snow_fraction_patch, absorbed_snow),
    )
    rural_downward_ground = fortran_multiply(
        fortran_multiply(one_minus_vegetation, emissivity_patch),
        downward_atmosphere,
    )
    rural_emitted_ground = fortran_multiply(
        fortran_multiply(emissivity_patch, jnp.float64(STEFAN_BOLTZMANN)),
        longwave_ground_fourth,
    )
    rural_linearized_longwave = fortran_multiply(
        fortran_multiply(
            fortran_multiply(
                emissivity_patch,
                jnp.float64(STEFAN_BOLTZMANN),
            ),
            previous_ground_cubed,
        ),
        four_temperature_increment,
    )
    rural_turbulent_loss = fortran_add(
        ground_sensible_heat,
        fortran_multiply(soil_evaporation, latent_heat_patch),
    )
    rural_soil_ground_heat = fortran_add(
        fortran_add(
            fortran_add(
                fortran_add(
                    fortran_add(solar_ground, downward_canopy),
                    rural_downward_ground,
                ),
                -rural_emitted_ground,
            ),
            -rural_linearized_longwave,
        ),
        -rural_turbulent_loss,
    )

    urban_turbulent_loss = fortran_add(
        fortran_add(
            ground_sensible_heat,
            fortran_multiply(soil_evaporation, latent_heat_patch),
        ),
        fortran_multiply(transpiration, jnp.float64(LATENT_HEAT_VAPORIZATION)),
    )
    urban_soil_ground_heat = fortran_add(
        fortran_add(
            fortran_add(
                fortran_add(
                    fortran_add(
                        fortran_add(
                            fortran_add(
                                jnp.asarray(inputs.absorbed_ground_patch, dtype=jnp.float64),
                                downward_canopy,
                            ),
                            -state.energy.net_longwave_patch,
                        ),
                        -longwave_temperature_correction,
                    ),
                    -urban_turbulent_loss,
                ),
                jnp.asarray(inputs.waste_heat_patch, dtype=jnp.float64),
            ),
            jnp.asarray(inputs.air_conditioner_heat_patch, dtype=jnp.float64),
        ),
        jnp.asarray(inputs.traffic_heat_patch, dtype=jnp.float64),
    )
    soil_ground_heat = jnp.where(
        nonlake_patch,
        jnp.where(patch_urban, urban_soil_ground_heat, rural_soil_ground_heat),
        state.energy.soil_ground_heat_patch,
    )
    soil_or_crop = (patch_landunit_type == jnp.int32(LANDUNIT_SOIL)) | (
        patch_landunit_type == jnp.int32(LANDUNIT_CROP)
    )
    rural_soil_ground_diagnostic = jnp.where(
        nonlake_patch & soil_or_crop,
        soil_ground_heat,
        state.energy.rural_soil_ground_heat_patch,
    )
    urban_soil_ground_diagnostic = jnp.where(
        urban_patch,
        soil_ground_heat,
        state.energy.urban_soil_ground_heat_patch,
    )

    total_sensible_heat_value = fortran_add(sensible_vegetation, ground_sensible_heat)
    total_evaporation_value = fortran_add(evaporation_vegetation, soil_evaporation)
    total_latent_heat_value = fortran_add(
        fortran_multiply(jnp.float64(LATENT_HEAT_VAPORIZATION), evaporation_vegetation),
        fortran_multiply(latent_heat_patch, soil_evaporation),
    )
    total_sensible_heat = jnp.where(
        nonlake_patch,
        total_sensible_heat_value,
        state.energy.total_sensible_heat_patch,
    )
    total_evaporation = jnp.where(
        nonlake_patch,
        total_evaporation_value,
        state.water.total_evaporation_patch,
    )
    total_latent_heat = jnp.where(
        nonlake_patch,
        total_latent_heat_value,
        state.energy.total_latent_heat_patch,
    )
    rural_total_sensible = jnp.where(
        nonlake_patch & soil_or_crop,
        total_sensible_heat,
        state.energy.rural_total_sensible_heat_patch,
    )
    rural_total_latent = jnp.where(
        nonlake_patch & soil_or_crop,
        total_latent_heat,
        state.energy.rural_total_latent_heat_patch,
    )
    urban_total_sensible = jnp.where(
        urban_patch,
        total_sensible_heat,
        state.energy.urban_total_sensible_heat_patch,
    )
    urban_total_latent = jnp.where(
        urban_patch,
        total_latent_heat,
        state.energy.urban_total_latent_heat_patch,
    )

    patch_top_liquid = top_liquid[patch_column]
    patch_top_ice = top_ice[patch_column]
    patch_top_water = fortran_add(patch_top_liquid, patch_top_ice)
    safe_top_water = jnp.where(
        patch_top_water != jnp.float64(0.0),
        patch_top_water,
        jnp.float64(1.0),
    )
    liquid_fraction = fortran_divide(patch_top_liquid, safe_top_water)
    positive_snow_evaporation = snow_evaporation >= jnp.float64(0.0)
    liquid_evaporation_candidate = jnp.maximum(
        fortran_multiply(snow_evaporation, liquid_fraction),
        jnp.float64(0.0),
    )
    ground_liquid_evaporation = jnp.where(
        nonlake_patch & positive_snow_evaporation,
        jnp.where(
            patch_top_water > jnp.float64(0.0),
            liquid_evaporation_candidate,
            jnp.float64(0.0),
        ),
        jnp.where(
            nonlake_patch,
            jnp.float64(0.0),
            state.water.ground_liquid_evaporation_patch,
        ),
    )
    snow_sublimation = jnp.where(
        nonlake_patch & positive_snow_evaporation,
        fortran_add(snow_evaporation, -ground_liquid_evaporation),
        jnp.where(
            nonlake_patch,
            jnp.float64(0.0),
            state.water.snow_sublimation_patch,
        ),
    )
    dew = jnp.abs(snow_evaporation)
    snow_dew = jnp.where(
        nonlake_patch & ~positive_snow_evaporation & (
            ground_temperature[patch_column]
            < jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)
        ),
        dew,
        jnp.where(nonlake_patch, jnp.float64(0.0), state.water.snow_dew_patch),
    )
    ground_dew = jnp.where(
        nonlake_patch & ~positive_snow_evaporation & (
            ground_temperature[patch_column]
            >= jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)
        ),
        dew,
        jnp.where(nonlake_patch, jnp.float64(0.0), state.water.ground_dew_patch),
    )

    canopy_evaporation = jnp.where(
        nonlake_patch,
        fortran_add(evaporation_vegetation, -transpiration),
        state.water.canopy_evaporation_patch,
    )
    vegetation_evaporation_latent = jnp.where(
        nonlake_patch,
        fortran_multiply(
            fortran_add(evaporation_vegetation, -transpiration),
            jnp.float64(LATENT_HEAT_VAPORIZATION),
        ),
        state.energy.vegetation_evaporation_latent_heat_patch,
    )
    vegetation_transpiration_latent = jnp.where(
        nonlake_patch,
        fortran_multiply(transpiration, jnp.float64(LATENT_HEAT_VAPORIZATION)),
        state.energy.vegetation_transpiration_latent_heat_patch,
    )
    ground_latent = jnp.where(
        nonlake_patch,
        fortran_multiply(soil_evaporation, latent_heat_patch),
        state.energy.ground_latent_heat_patch,
    )

    surface_water_temperature = jnp.asarray(
        inputs.surface_water_temperature_column,
        dtype=jnp.float64,
    )
    surface_water_capacity_rate = fortran_divide(
        jnp.asarray(inputs.surface_water_heat_capacity_column, dtype=jnp.float64),
        time_step_seconds,
    )
    surface_water_storage = fortran_multiply(
        fortran_multiply(
            water_fraction,
            fortran_add(surface_water_temperature, -previous_water_temperature),
        ),
        surface_water_capacity_rate,
    )
    base_energy_error = fortran_add(
        fortran_add(
            fortran_add(
                fortran_add(
                    soil_ground_heat,
                    -jnp.asarray(inputs.phase_change_heat_flux_column, dtype=jnp.float64)[
                        patch_column
                    ],
                ),
                -jnp.asarray(
                    inputs.surface_water_phase_change_heat_flux_column,
                    dtype=jnp.float64,
                )[patch_column],
            ),
            -surface_water_storage[patch_column],
        ),
        jnp.asarray(inputs.surface_water_to_snow_heat_flux_column, dtype=jnp.float64)[
            patch_column
        ],
    )
    enclosure_column = (
        (column_type == jnp.int32(COLUMN_URBAN_SUN_WALL))
        | (column_type == jnp.int32(COLUMN_URBAN_SHADE_WALL))
        | (column_type == jnp.int32(COLUMN_URBAN_ROOF))
    )
    enclosure_patch = enclosure_column[patch_column]
    base_energy_error = jnp.where(
        enclosure_patch,
        fortran_add(
            base_energy_error,
            jnp.asarray(inputs.building_heat_error_column, dtype=jnp.float64)[patch_column],
        ),
        base_energy_error,
    )
    soil_energy_error = jnp.where(
        nonlake_patch,
        base_energy_error,
        state.energy.soil_energy_error_patch,
    )
    current_temperature = jnp.asarray(
        inputs.soil_snow_temperature_column_level,
        dtype=jnp.float64,
    )
    temperature_factor = jnp.asarray(
        inputs.temperature_factor_column_level,
        dtype=jnp.float64,
    )
    fortran_levels = jnp.arange(
        -NUMBER_OF_SNOW_LEVELS + 1,
        NUMBER_OF_GROUND_LEVELS + 1,
        dtype=jnp.int32,
    )

    def subtract_level_energy(level_index, error):
        fortran_level = fortran_levels[level_index]
        eligible = nonlake_patch & (
            ~enclosure_patch | (fortran_level <= jnp.int32(NUMBER_OF_URBAN_LEVELS))
        )
        snow_contribution = eligible & (
            (fortran_level >= snow_layer_count[patch_column] + jnp.int32(1))
            & (fortran_level < jnp.int32(1))
        )
        ground_contribution = eligible & (fortran_level >= jnp.int32(1))
        contributes = snow_contribution | ground_contribution
        selected_factor = temperature_factor[patch_column, level_index]
        safe_factor = jnp.where(contributes, selected_factor, jnp.float64(1.0))
        temperature_change = fortran_add(
            current_temperature[patch_column, level_index],
            -previous_temperature[patch_column, level_index],
        )
        temperature_change = jnp.where(contributes, temperature_change, jnp.float64(0.0))
        ground_layer_energy = fortran_divide(temperature_change, safe_factor)
        snow_layer_energy = fortran_divide(
            fortran_multiply(snow_fraction_patch, temperature_change),
            safe_factor,
        )
        layer_energy = jnp.where(
            snow_contribution,
            snow_layer_energy,
            ground_layer_energy,
        )
        return jnp.where(contributes, fortran_add(error, -layer_energy), error)

    soil_energy_error = lax.fori_loop(
        0,
        NUMBER_OF_SOIL_SNOW_LEVELS,
        subtract_level_energy,
        soil_energy_error,
    )

    outgoing_longwave_rural = fortran_add(
        fortran_add(
            fortran_add(
                jnp.asarray(inputs.upward_longwave_above_canopy_patch, dtype=jnp.float64),
                fortran_multiply(
                    fortran_multiply(
                        one_minus_vegetation,
                        fortran_add(jnp.float64(1.0), -emissivity_patch),
                    ),
                    downward_atmosphere,
                ),
            ),
            fortran_multiply(
                fortran_multiply(
                    fortran_multiply(one_minus_vegetation, emissivity_patch),
                    jnp.float64(STEFAN_BOLTZMANN),
                ),
                longwave_ground_fourth,
            ),
        ),
        longwave_temperature_correction,
    )
    outgoing_longwave_urban = fortran_add(
        state.energy.outgoing_longwave_patch,
        longwave_temperature_correction,
    )
    outgoing_longwave = jnp.where(
        nonlake_patch,
        jnp.where(patch_urban, outgoing_longwave_urban, outgoing_longwave_rural),
        state.energy.outgoing_longwave_patch,
    )
    net_longwave = jnp.where(
        nonlake_patch,
        jnp.where(
            patch_urban,
            fortran_add(state.energy.net_longwave_patch, longwave_temperature_correction),
            fortran_add(outgoing_longwave, -downward_atmosphere),
        ),
        state.energy.net_longwave_patch,
    )
    rural_net_longwave = jnp.where(
        nonlake_patch & soil_or_crop,
        fortran_add(outgoing_longwave, -downward_atmosphere),
        state.energy.rural_net_longwave_patch,
    )
    rural_outgoing_longwave = jnp.where(
        nonlake_patch & soil_or_crop,
        outgoing_longwave,
        state.energy.rural_outgoing_longwave_patch,
    )
    urban_net_longwave = jnp.where(
        urban_patch,
        fortran_add(state.energy.urban_net_longwave_patch, longwave_temperature_correction),
        state.energy.urban_net_longwave_patch,
    )
    urban_outgoing_longwave = jnp.where(
        urban_patch,
        outgoing_longwave,
        state.energy.urban_outgoing_longwave_patch,
    )

    skin_temperature = jnp.where(
        nonlake_patch & ~patch_urban & (vegetation_exposed == jnp.int32(0)),
        jnp.sqrt(jnp.sqrt(longwave_ground_fourth)),
        state.skin_temperature_patch,
    )
    current_top_temperature = jnp.take_along_axis(
        current_temperature,
        gather_index,
        axis=1,
    )[:, 0]
    skin_temperature = jnp.where(
        urban_patch,
        current_top_temperature[patch_column],
        skin_temperature,
    )

    soil_energy_error_column_update = _ordered_patch_to_column_sum(
        topology,
        soil_energy_error,
        nonlake_column,
        maximum_patches_per_column=parameters.maximum_patches_per_column,
    )
    soil_energy_error_column = jnp.where(
        nonlake_column,
        soil_energy_error_column_update,
        state.energy.soil_energy_error_column,
    )

    water_state = SoilFluxesWaterState(
        soil_evaporation_patch=soil_evaporation,
        snow_evaporation_patch=snow_evaporation,
        bare_soil_evaporation_patch=bare_soil_evaporation,
        surface_water_evaporation_patch=surface_water_evaporation,
        canopy_evaporation_patch=canopy_evaporation,
        total_evaporation_patch=total_evaporation,
        ground_liquid_evaporation_patch=ground_liquid_evaporation,
        snow_sublimation_patch=snow_sublimation,
        snow_dew_patch=snow_dew,
        ground_dew_patch=ground_dew,
    )
    energy_state = SoilFluxesEnergyState(
        ground_sensible_heat_patch=ground_sensible_heat,
        soil_ground_heat_patch=soil_ground_heat,
        urban_soil_ground_heat_patch=urban_soil_ground_diagnostic,
        rural_soil_ground_heat_patch=rural_soil_ground_diagnostic,
        total_sensible_heat_patch=total_sensible_heat,
        urban_total_sensible_heat_patch=urban_total_sensible,
        rural_total_sensible_heat_patch=rural_total_sensible,
        total_latent_heat_patch=total_latent_heat,
        urban_total_latent_heat_patch=urban_total_latent,
        rural_total_latent_heat_patch=rural_total_latent,
        outgoing_longwave_patch=outgoing_longwave,
        net_longwave_patch=net_longwave,
        rural_net_longwave_patch=rural_net_longwave,
        rural_outgoing_longwave_patch=rural_outgoing_longwave,
        urban_net_longwave_patch=urban_net_longwave,
        urban_outgoing_longwave_patch=urban_outgoing_longwave,
        vegetation_evaporation_latent_heat_patch=vegetation_evaporation_latent,
        vegetation_transpiration_latent_heat_patch=vegetation_transpiration_latent,
        ground_latent_heat_patch=ground_latent,
        soil_energy_error_patch=soil_energy_error,
        soil_energy_error_column=soil_energy_error_column,
    )
    return SoilFluxesResult(
        state=SoilFluxesState(
            skin_temperature_patch=skin_temperature,
            water=water_state,
            energy=energy_state,
        ),
        diagnostics=SoilFluxesDiagnostics(
            previous_ground_temperature_column=previous_ground_temperature,
            ground_temperature_increment_column=temperature_increment,
            maximum_top_layer_evaporation_column=maximum_evaporation,
            requested_top_layer_evaporation_column=requested_evaporation,
            evaporation_availability_ratio_column=availability_ratio,
            ground_longwave_temperature_fourth_column=longwave_ground_fourth_column,
            longwave_temperature_correction_patch=longwave_temperature_correction,
        ),
    )
