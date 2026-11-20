
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from clm_jax.physics.methane_balance_initialization import (
    NUMBER_OF_METHANE_SOIL_LEVELS,
)
from clm_jax.physics.methane_biogeochemistry import (
    MethaneGasCompetitionInputs,
    MethaneGasCompetitionParameters,
    MethaneGasCompetitionResult,
    MethaneGasCompetitionState,
    methane_gas_competition_active_case,
)
from clm_jax.physics.methane_profile_recurrences import (
    clamp_negative_methane,
    methane_balance,
)
from clm_jax.physics.methane_profile_recurrences import (
    serial_column_integral as _serial_column_integral,
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
from clm_jax.physics.native_power import native_diffusion_power

NUMBER_OF_METHANE_SNOW_LEVELS = 12
NUMBER_OF_METHANE_GASES = 2
NUMBER_OF_METHANE_TRANSPORT_LEVELS = NUMBER_OF_METHANE_SOIL_LEVELS + 1


def _diffusion_power_primal(base, exponent, active):
    if jax.default_backend() == "gpu":
        value = native_diffusion_power(jnp.where(active, base, jnp.float64(1.)),
                                       jnp.where(active, exponent, jnp.float64(1.)))
        return jnp.where(active, value, fortran_power(base, exponent))
    return fortran_power(base, exponent)


@jax.custom_jvp
def _diffusion_power(base, exponent, active):
    return _diffusion_power_primal(base, exponent, active)


@_diffusion_power.defjvp
def _diffusion_power_jvp(primals, tangents):
    value = _diffusion_power_primal(*primals)
    _, tangent = jax.jvp(jnp.power, primals[:2], tangents[:2])
    return value, jnp.broadcast_to(tangent, value.shape)


def _mineral_diffusion_term_primal(organic_fraction, air_fraction, air_share, exponent, active):
    term = fortran_multiply(
        fortran_add(jnp.float64(1.0), -organic_fraction),
        fortran_power(air_fraction, jnp.float64(2.0)),
    )
    return fortran_multiply(term, _diffusion_power(air_share, exponent, active))


@jax.custom_jvp
def _mineral_diffusion_term(organic_fraction, air_fraction, air_share, exponent, active):
    return _mineral_diffusion_term_primal(organic_fraction, air_fraction, air_share, exponent, active)


@_mineral_diffusion_term.defjvp
def _mineral_diffusion_term_jvp(primals, tangents):
    organic_fraction, air_fraction, air_share, exponent, _ = primals
    empty_air = ((air_fraction == 0.) & (air_share == 0.) & (exponent > 0.)
                 & jnp.isfinite(exponent) & jnp.isfinite(organic_fraction))

    def guarded(organic_fraction, air_fraction, air_share, exponent, active):
        term = _mineral_diffusion_term_primal(
            organic_fraction,
            jnp.where(empty_air, 1., air_fraction),
            jnp.where(empty_air, 1., air_share),
            jnp.where(empty_air, 1., exponent),
            active,
        )
        return jnp.where(empty_air, jnp.float64(0.), term)

    return _mineral_diffusion_term_primal(*primals), jax.jvp(guarded, primals, tangents)[1]


class MethaneSoilGasTransportInputs(NamedTuple):
    methane_column: jnp.ndarray
    column_gridcell: jnp.ndarray
    unsaturated_water_table_layer_column: jnp.ndarray
    snow_layer_index_column: jnp.ndarray
    soil_layer_thickness_column_level: jnp.ndarray
    snow_layer_thickness_column_level: jnp.ndarray
    soil_b_parameter_column_level: jnp.ndarray
    soil_porosity_column_level: jnp.ndarray
    soil_organic_matter_column_level: jnp.ndarray
    soil_temperature_column_level: jnp.ndarray
    snow_temperature_column_level: jnp.ndarray
    ground_temperature_column: jnp.ndarray
    surface_water_temperature_column: jnp.ndarray
    surface_water_fraction_column: jnp.ndarray
    surface_water_depth_mm_column: jnp.ndarray
    volumetric_soil_water_column_level: jnp.ndarray
    soil_liquid_water_kg_m2_column_level: jnp.ndarray
    soil_ice_kg_m2_column_level: jnp.ndarray
    snow_liquid_water_kg_m2_column_level: jnp.ndarray
    snow_ice_kg_m2_column_level: jnp.ndarray
    atmospheric_methane_gridcell: jnp.ndarray
    atmospheric_oxygen_gridcell: jnp.ndarray
    unsaturated_methane_production_column_level: jnp.ndarray
    saturated_methane_production_column_level: jnp.ndarray
    unsaturated_oxygen_aerenchyma_column_level: jnp.ndarray
    saturated_oxygen_aerenchyma_column_level: jnp.ndarray


class MethaneSoilGasTransportState(NamedTuple):
    competition: MethaneGasCompetitionState
    unsaturated_methane_concentration_column_level: jnp.ndarray
    saturated_methane_concentration_column_level: jnp.ndarray
    unsaturated_oxygen_concentration_column_level: jnp.ndarray
    saturated_oxygen_concentration_column_level: jnp.ndarray
    unsaturated_surface_aerenchyma_column: jnp.ndarray
    saturated_surface_aerenchyma_column: jnp.ndarray
    unsaturated_ebullition_total_column: jnp.ndarray
    saturated_ebullition_total_column: jnp.ndarray
    unsaturated_surface_ebullition_column: jnp.ndarray
    saturated_surface_ebullition_column: jnp.ndarray
    unsaturated_surface_diffusion_column: jnp.ndarray
    saturated_surface_diffusion_column: jnp.ndarray
    ground_methane_conductance_column: jnp.ndarray


class MethaneSoilGasTransportParameters(NamedTuple):
    land_time_step_seconds: float | jnp.ndarray = 1800.0
    methane_time_step_seconds: float | jnp.ndarray = 1800.0
    saturated_diffusion_exponent: float | jnp.ndarray = 2.0
    gas_diffusion_scale: float | jnp.ndarray = 1.0
    liquid_diffusion_scale: float | jnp.ndarray = 1.0
    surface_ice_cap_mm: float | jnp.ndarray = 100.0
    organic_matter_maximum_kg_m3: float | jnp.ndarray = 130.0
    freezing_temperature_kelvin: float | jnp.ndarray = 273.15
    water_density_kg_m3: float | jnp.ndarray = 1000.0
    ice_density_kg_m3: float | jnp.ndarray = 917.0
    gas_diffusion_constants_species_coefficient: jnp.ndarray = np.asarray(
        ((0.1875, 0.0013), (0.1759, 0.00117)),
        dtype=np.float64,
    )
    water_diffusion_constants_species_coefficient: jnp.ndarray = np.asarray(
        (
            (0.9798, 0.02986, 0.0004381),
            (1.172, 0.03443, 0.0005048),
        ),
        dtype=np.float64,
    )
    henry_inverse_temperature_species_kelvin: jnp.ndarray = np.asarray(
        (1600.0, 1500.0),
        dtype=np.float64,
    )
    henry_constant_species_litre_atm_per_mol: jnp.ndarray = np.asarray(
        (714.29, 769.23),
        dtype=np.float64,
    )
    henry_reference_temperature_kelvin: float | jnp.ndarray = 298.0
    gas_constant_litre_atm_per_mol_kelvin: float | jnp.ndarray = 0.0821
    minimum_transport_value: float | jnp.ndarray = 1.0e-12
    conservation_tolerance_mol_m2: float | jnp.ndarray = 1.0e-8


DEFAULT_METHANE_SOIL_GAS_TRANSPORT_PARAMETERS = (
    MethaneSoilGasTransportParameters()
)


class MethaneSoilGasTransportDiagnostics(NamedTuple):
    competition: MethaneGasCompetitionResult
    unsaturated_henry_ratio_column_level_species: jnp.ndarray
    saturated_henry_ratio_column_level_species: jnp.ndarray
    unsaturated_effective_storage_column_level_species: jnp.ndarray
    saturated_effective_storage_column_level_species: jnp.ndarray
    unsaturated_diffusivity_column_level_species: jnp.ndarray
    saturated_diffusivity_column_level_species: jnp.ndarray
    unsaturated_species_ground_conductance_column_species: jnp.ndarray
    saturated_species_ground_conductance_column_species: jnp.ndarray
    unsaturated_raw_methane_conservation_error_column: jnp.ndarray
    saturated_raw_methane_conservation_error_column: jnp.ndarray
    methane_conservation_violation_column: jnp.ndarray
    methane_column_count: jnp.ndarray


class MethaneSoilGasTransportResult(NamedTuple):
    state: MethaneSoilGasTransportState
    diagnostics: MethaneSoilGasTransportDiagnostics


class _BranchState(NamedTuple):
    methane_concentration_column_level: jnp.ndarray
    oxygen_concentration_column_level: jnp.ndarray
    methane_oxidation_column_level: jnp.ndarray
    oxygen_oxidation_column_level: jnp.ndarray
    methane_aerenchyma_column_level: jnp.ndarray
    methane_ebullition_column_level: jnp.ndarray
    oxygen_decomposition_column_level: jnp.ndarray
    oxygen_stress_column_level: jnp.ndarray
    methane_stress_column_level: jnp.ndarray
    surface_aerenchyma_column: jnp.ndarray
    ebullition_total_column: jnp.ndarray
    surface_ebullition_column: jnp.ndarray
    surface_diffusion_column: jnp.ndarray


class _BranchDiagnostics(NamedTuple):
    henry_ratio_column_level_species: jnp.ndarray
    effective_storage_column_level_species: jnp.ndarray
    diffusivity_column_level_species: jnp.ndarray
    species_ground_conductance_column_species: jnp.ndarray
    raw_methane_conservation_error_column: jnp.ndarray
    conservation_violation_column: jnp.ndarray


class MethaneSoilGasTransportBranchResult(NamedTuple):

    state: _BranchState
    ground_methane_conductance_column: jnp.ndarray
    diagnostics: _BranchDiagnostics


def _validate_shapes(
    inputs: MethaneSoilGasTransportInputs,
    state: MethaneSoilGasTransportState,
    parameters: MethaneSoilGasTransportParameters,
) -> tuple[int, int]:
    if jnp.ndim(inputs.methane_column) != 1:
        raise ValueError("methane_column must be a vector")
    column_count = inputs.methane_column.shape[0]
    soil_shape = (column_count, 25)
    snow_shape = (column_count, NUMBER_OF_METHANE_SNOW_LEVELS)
    for name in (
        "soil_layer_thickness_column_level",
        "soil_b_parameter_column_level",
        "soil_porosity_column_level",
        "soil_organic_matter_column_level",
        "soil_temperature_column_level",
        "volumetric_soil_water_column_level",
        "soil_liquid_water_kg_m2_column_level",
        "soil_ice_kg_m2_column_level",
        "unsaturated_methane_production_column_level",
        "saturated_methane_production_column_level",
        "unsaturated_oxygen_aerenchyma_column_level",
        "saturated_oxygen_aerenchyma_column_level",
    ):
        if jnp.shape(getattr(inputs, name)) != soil_shape:
            raise ValueError(f"{name} must have shape (column, 25)")
    for name in (
        "snow_layer_thickness_column_level",
        "snow_temperature_column_level",
        "snow_liquid_water_kg_m2_column_level",
        "snow_ice_kg_m2_column_level",
    ):
        if jnp.shape(getattr(inputs, name)) != snow_shape:
            raise ValueError(f"{name} must have shape (column, 12)")
    for name in (
        "column_gridcell",
        "unsaturated_water_table_layer_column",
        "snow_layer_index_column",
        "ground_temperature_column",
        "surface_water_temperature_column",
        "surface_water_fraction_column",
        "surface_water_depth_mm_column",
    ):
        if jnp.shape(getattr(inputs, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    gridcell_count = inputs.atmospheric_methane_gridcell.shape[0]
    if jnp.shape(inputs.atmospheric_methane_gridcell) != (gridcell_count,):
        raise ValueError("atmospheric methane must be a gridcell vector")
    if jnp.shape(inputs.atmospheric_oxygen_gridcell) != (gridcell_count,):
        raise ValueError("atmospheric gases must use the same gridcell dimension")
    for name in (
        "unsaturated_methane_concentration_column_level",
        "saturated_methane_concentration_column_level",
        "unsaturated_oxygen_concentration_column_level",
        "saturated_oxygen_concentration_column_level",
    ):
        if jnp.shape(getattr(state, name)) != soil_shape:
            raise ValueError(f"{name} must have shape (column, 25)")
    for name in MethaneGasCompetitionState._fields:
        if jnp.shape(getattr(state.competition, name)) != soil_shape:
            raise ValueError(f"competition.{name} must have shape (column, 25)")
    for name in MethaneSoilGasTransportState._fields[5:]:
        if name == "competition":
            continue
        if jnp.shape(getattr(state, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    scalar_names = (
        "land_time_step_seconds",
        "methane_time_step_seconds",
        "saturated_diffusion_exponent",
        "gas_diffusion_scale",
        "liquid_diffusion_scale",
        "surface_ice_cap_mm",
        "organic_matter_maximum_kg_m3",
        "freezing_temperature_kelvin",
        "water_density_kg_m3",
        "ice_density_kg_m3",
        "henry_reference_temperature_kelvin",
        "gas_constant_litre_atm_per_mol_kelvin",
        "minimum_transport_value",
        "conservation_tolerance_mol_m2",
    )
    for name in scalar_names:
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    if jnp.shape(parameters.gas_diffusion_constants_species_coefficient) != (2, 2):
        raise ValueError("gas diffusion constants must have shape (2, 2)")
    if jnp.shape(parameters.water_diffusion_constants_species_coefficient) != (
        2,
        3,
    ):
        raise ValueError("water diffusion constants must have shape (2, 3)")
    for name in (
        "henry_inverse_temperature_species_kelvin",
        "henry_constant_species_litre_atm_per_mol",
    ):
        if jnp.shape(getattr(parameters, name)) != (2,):
            raise ValueError(f"{name} must contain CH4 and O2 values")
    return column_count, gridcell_count


def _active_soil(value: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(value, dtype=jnp.float64)[
        :, :NUMBER_OF_METHANE_SOIL_LEVELS
    ]


def _preserve_level_state(
    methane_column: jnp.ndarray,
    previous: jnp.ndarray,
    active_result: jnp.ndarray,
) -> jnp.ndarray:
    previous = jnp.asarray(previous, dtype=jnp.float64)
    first_twenty = jnp.where(
        methane_column[:, None],
        active_result,
        previous[:, :NUMBER_OF_METHANE_SOIL_LEVELS],
    )
    return jnp.concatenate(
        (first_twenty, previous[:, NUMBER_OF_METHANE_SOIL_LEVELS:]),
        axis=1,
    )


def _preserve_column_state(
    methane_column: jnp.ndarray,
    previous: jnp.ndarray,
    active_result: jnp.ndarray,
) -> jnp.ndarray:
    previous = jnp.asarray(previous, dtype=jnp.float64)
    return jnp.where(methane_column, active_result, previous)


def _henry_ratios(
    inputs: MethaneSoilGasTransportInputs,
    parameters: MethaneSoilGasTransportParameters,
) -> jnp.ndarray:
    temperature = jnp.concatenate(
        (
            jnp.asarray(inputs.ground_temperature_column, dtype=jnp.float64)[
                :, None
            ],
            _active_soil(inputs.soil_temperature_column_level),
        ),
        axis=1,
    )
    inverse_temperature_difference = fortran_add(
        fortran_divide(jnp.float64(1.0), temperature[:, :, None]),
        -fortran_divide(
            jnp.float64(1.0),
            jnp.asarray(
                parameters.henry_reference_temperature_kelvin,
                dtype=jnp.float64,
            ),
        ),
    )
    exponent = fortran_multiply(
        -jnp.asarray(
            parameters.henry_inverse_temperature_species_kelvin,
            dtype=jnp.float64,
        )[None, None, :],
        inverse_temperature_difference,
    )
    exponent = fortran_add(
        exponent,
        jnp.log(
            jnp.asarray(
                parameters.henry_constant_species_litre_atm_per_mol,
                dtype=jnp.float64,
            )
        )[None, None, :],
    )
    inverse_henry = strict_exp(exponent)
    ratio = fortran_divide(temperature[:, :, None], inverse_henry)
    return fortran_multiply(
        ratio,
        jnp.asarray(
            parameters.gas_constant_litre_atm_per_mol_kelvin,
            dtype=jnp.float64,
        ),
    )


def _reaction_sources(
    water_table_layer_column: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    methane_production: jnp.ndarray,
    oxygen_aerenchyma: jnp.ndarray,
    branch: _BranchState,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    methane_production = _active_soil(methane_production)
    oxygen_aerenchyma = _active_soil(oxygen_aerenchyma)
    methane_oxidation = _active_soil(branch.methane_oxidation_column_level)
    oxygen_oxidation = _active_soil(branch.oxygen_oxidation_column_level)
    methane_aerenchyma = _active_soil(branch.methane_aerenchyma_column_level)
    methane_ebullition = _active_soil(branch.methane_ebullition_column_level)
    oxygen_decomposition = _active_soil(
        branch.oxygen_decomposition_column_level
    )

    methane_source = fortran_add(methane_production, -methane_oxidation)
    methane_source = fortran_add(methane_source, -methane_aerenchyma)
    methane_source = fortran_add(methane_source, -methane_ebullition)
    oxygen_source = fortran_add(-oxygen_oxidation, -oxygen_decomposition)
    oxygen_source = fortran_add(oxygen_source, oxygen_aerenchyma)

    ebullition_total = _serial_column_integral(
        methane_ebullition,
        layer_thickness,
    )
    surface_aerenchyma = _serial_column_integral(
        methane_aerenchyma,
        layer_thickness,
    )
    water_table = jnp.asarray(water_table_layer_column, dtype=jnp.int32)
    water_table_index = jnp.clip(
        water_table - jnp.int32(1),
        jnp.int32(0),
        jnp.int32(NUMBER_OF_METHANE_SOIL_LEVELS - 1),
    )
    water_table_thickness = jnp.take_along_axis(
        layer_thickness,
        water_table_index[:, None],
        axis=1,
    )[:, 0]
    ebullition_source = fortran_divide(
        ebullition_total,
        water_table_thickness,
    )
    level_number = jnp.arange(
        1,
        NUMBER_OF_METHANE_SOIL_LEVELS + 1,
        dtype=jnp.int32,
    )[None, :]
    methane_source = fortran_add(
        methane_source,
        jnp.where(
            (water_table[:, None] != jnp.int32(0))
            & (level_number == water_table[:, None]),
            ebullition_source[:, None],
            jnp.float64(0.0),
        ),
    )
    return (
        jnp.stack((methane_source, oxygen_source), axis=2),
        ebullition_total,
        surface_aerenchyma,
    )


def _water_diffusion_polynomial(
    temperature_celsius: jnp.ndarray,
    constants: jnp.ndarray,
) -> jnp.ndarray:
    result = fortran_add(
        constants[0],
        fortran_multiply(constants[1], temperature_celsius),
    )
    return fortran_add(
        result,
        fortran_multiply(
            constants[2],
            fortran_power(temperature_celsius, jnp.float64(2.0)),
        ),
    )


def _species_ground_conductance(
    inputs: MethaneSoilGasTransportInputs,
    ground_methane_conductance: jnp.ndarray,
    saturated: bool,
    parameters: MethaneSoilGasTransportParameters,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    small = jnp.asarray(parameters.minimum_transport_value, dtype=jnp.float64)
    freezing = jnp.asarray(
        parameters.freezing_temperature_kelvin,
        dtype=jnp.float64,
    )
    water_density = jnp.asarray(
        parameters.water_density_kg_m3,
        dtype=jnp.float64,
    )
    ice_density = jnp.asarray(
        parameters.ice_density_kg_m3,
        dtype=jnp.float64,
    )
    satpow = jnp.asarray(
        parameters.saturated_diffusion_exponent,
        dtype=jnp.float64,
    )
    gas_scale = jnp.asarray(parameters.gas_diffusion_scale, dtype=jnp.float64)
    liquid_scale = jnp.asarray(
        parameters.liquid_diffusion_scale,
        dtype=jnp.float64,
    )
    base_ground = jnp.maximum(
        jnp.asarray(ground_methane_conductance, dtype=jnp.float64),
        small,
    )
    snow_index = jnp.asarray(inputs.snow_layer_index_column, dtype=jnp.int32)
    snow_thickness = jnp.asarray(
        inputs.snow_layer_thickness_column_level,
        dtype=jnp.float64,
    )
    snow_temperature = jnp.asarray(
        inputs.snow_temperature_column_level,
        dtype=jnp.float64,
    )
    snow_liquid = jnp.asarray(
        inputs.snow_liquid_water_kg_m2_column_level,
        dtype=jnp.float64,
    )
    snow_ice = jnp.asarray(
        inputs.snow_ice_kg_m2_column_level,
        dtype=jnp.float64,
    )
    gas_constants = jnp.asarray(
        parameters.gas_diffusion_constants_species_coefficient,
        dtype=jnp.float64,
    )
    water_constants = jnp.asarray(
        parameters.water_diffusion_constants_species_coefficient,
        dtype=jnp.float64,
    )
    ten_thirds = fortran_divide(jnp.float64(10.0), jnp.float64(3.0))

    def one_species(species: int) -> jnp.ndarray:
        def snow_step(level: int, resistance: jnp.ndarray) -> jnp.ndarray:
            native_level = jnp.int32(level - NUMBER_OF_METHANE_SNOW_LEVELS + 1)
            present = native_level >= snow_index + jnp.int32(1)
            thickness = jnp.where(
                present,
                snow_thickness[:, level],
                jnp.float64(1.0),
            )
            temperature_celsius = fortran_add(
                jnp.where(present, snow_temperature[:, level], freezing),
                -freezing,
            )
            ice_fraction = fortran_divide(
                fortran_divide(jnp.where(present, snow_ice[:, level], jnp.float64(1.0)), ice_density),
                thickness,
            )
            water_fraction = fortran_divide(
                fortran_divide(jnp.where(present, snow_liquid[:, level], jnp.float64(1.0)), water_density),
                thickness,
            )
            air_fraction = jnp.maximum(
                fortran_add(
                    fortran_add(jnp.float64(1.0), -ice_fraction),
                    -water_fraction,
                ),
                jnp.float64(0.0),
            )
            air_and_water = fortran_add(air_fraction, water_fraction)
            use_gas_diffusion = air_fraction > jnp.float64(0.05)
            air_share = fortran_divide(
                air_fraction,
                jnp.where(
                    air_and_water != jnp.float64(0.0),
                    air_and_water,
                    jnp.float64(1.0),
                ),
            )
            gas_diffusivity = fortran_add(
                gas_constants[species, 0],
                fortran_multiply(
                    gas_constants[species, 1],
                    temperature_celsius,
                ),
            )
            gas_diffusivity = fortran_multiply(
                gas_diffusivity,
                jnp.float64(1.0e-4),
            )
            gas_diffusivity = fortran_multiply(
                gas_diffusivity,
                _diffusion_power(air_share, ten_thirds,
                                 present & inputs.methane_column),
            )
            gas_diffusivity = fortran_divide(
                gas_diffusivity,
                jnp.where(
                    use_gas_diffusion,
                    fortran_power(air_and_water, jnp.float64(2.0)),
                    jnp.float64(1.0),
                ),
            )
            gas_diffusivity = fortran_multiply(gas_diffusivity, gas_scale)
            water_diffusivity = fortran_multiply(
                _diffusion_power(water_fraction, satpow,
                                 present & inputs.methane_column),
                _water_diffusion_polynomial(
                    temperature_celsius,
                    water_constants[species],
                ),
            )
            water_diffusivity = fortran_multiply(
                water_diffusivity,
                jnp.float64(1.0e-9),
            )
            water_diffusivity = fortran_multiply(
                water_diffusivity,
                liquid_scale,
            )
            diffusivity = jnp.where(
                use_gas_diffusion,
                gas_diffusivity,
                water_diffusivity,
            )
            diffusivity = jnp.maximum(diffusivity, small)
            increment = fortran_divide(thickness, diffusivity)
            return jnp.where(
                present,
                fortran_add(resistance, increment),
                resistance,
            )

        snow_resistance = lax.fori_loop(
            0,
            NUMBER_OF_METHANE_SNOW_LEVELS,
            snow_step,
            jnp.zeros_like(base_ground),
        )

        soil_temperature = _active_soil(inputs.soil_temperature_column_level)
        soil_thickness = _active_soil(
            inputs.soil_layer_thickness_column_level
        )
        soil_water = _active_soil(inputs.volumetric_soil_water_column_level)
        porosity = _active_soil(inputs.soil_porosity_column_level)
        soil_liquid = _active_soil(
            inputs.soil_liquid_water_kg_m2_column_level
        )
        soil_ice = _active_soil(inputs.soil_ice_kg_m2_column_level)
        top_temperature_celsius = fortran_add(
            soil_temperature[:, 0],
            -freezing,
        )
        top_water_polynomial = _water_diffusion_polynomial(
            top_temperature_celsius,
            water_constants[species],
        )
        top_diffusivity = fortran_multiply(
            top_water_polynomial,
            jnp.float64(1.0e-9),
        )
        liquid_volume = fortran_divide(soil_liquid[:, 0], water_density)
        ice_volume = fortran_divide(soil_ice[:, 0], ice_density)
        liquid_numerator = fortran_add(liquid_volume, small)
        phase_denominator = fortran_add(liquid_volume, ice_volume)
        phase_denominator = fortran_add(phase_denominator, small)
        frozen_top_diffusivity = fortran_multiply(
            top_diffusivity,
            liquid_numerator,
        )
        frozen_top_diffusivity = fortran_divide(
            frozen_top_diffusivity,
            phase_denominator,
        )
        top_diffusivity = jnp.where(
            soil_temperature[:, 0] <= freezing,
            frozen_top_diffusivity,
            top_diffusivity,
        )
        top_diffusivity = fortran_multiply(top_diffusivity, liquid_scale)
        old_pond = (
            (snow_index == jnp.int32(0))
            & (soil_water[:, 0] > porosity[:, 0])
        )
        old_pond_depth = fortran_multiply(
            soil_thickness[:, 0],
            fortran_add(soil_water[:, 0], -porosity[:, 0]),
        )
        old_pond_resistance = fortran_divide(
            old_pond_depth,
            top_diffusivity,
        )
        pond_resistance = jnp.where(
            old_pond,
            old_pond_resistance,
            jnp.float64(0.0),
        )

        surface_fraction = jnp.asarray(
            inputs.surface_water_fraction_column,
            dtype=jnp.float64,
        )
        surface_water = jnp.asarray(
            inputs.surface_water_depth_mm_column,
            dtype=jnp.float64,
        )
        surface_temperature = jnp.asarray(
            inputs.surface_water_temperature_column,
            dtype=jnp.float64,
        )
        positive_fraction = surface_fraction > jnp.float64(0.0)
        safe_fraction = jnp.where(positive_fraction, surface_fraction, jnp.float64(1.0))
        surface_temperature_celsius = fortran_add(
            surface_temperature,
            -freezing,
        )
        surface_diffusivity = _water_diffusion_polynomial(
            surface_temperature_celsius,
            water_constants[species],
        )
        surface_diffusivity = fortran_multiply(
            surface_diffusivity,
            jnp.float64(1.0e-9),
        )
        surface_diffusivity = fortran_multiply(
            surface_diffusivity,
            liquid_scale,
        )
        surface_depth = fortran_divide(surface_water, jnp.float64(1000.0))
        surface_depth = fortran_divide(surface_depth, safe_fraction)
        new_pond_increment = fortran_divide(
            surface_depth,
            surface_diffusivity,
        )
        unfrozen_surface_pond = (
            saturated
            & positive_fraction
            & (surface_temperature >= freezing)
        )
        pond_resistance = jnp.where(
            unfrozen_surface_pond,
            fortran_add(pond_resistance, new_pond_increment),
            pond_resistance,
        )
        water_per_covered_area = fortran_divide(surface_water, safe_fraction)
        frozen_cap = (
            saturated
            & positive_fraction
            & ~unfrozen_surface_pond
            & (
                water_per_covered_area
                > jnp.asarray(parameters.surface_ice_cap_mm, dtype=jnp.float64)
            )
        )
        pond_resistance = jnp.where(
            frozen_cap,
            fortran_divide(jnp.float64(1.0), small),
            pond_resistance,
        )
        total_resistance = fortran_add(
            fortran_divide(jnp.float64(1.0), base_ground),
            snow_resistance,
        )
        total_resistance = fortran_add(total_resistance, pond_resistance)
        return fortran_divide(jnp.float64(1.0), total_resistance)

    species_conductance = jnp.stack(
        tuple(one_species(species) for species in range(NUMBER_OF_METHANE_GASES)),
        axis=1,
    )
    return species_conductance, base_ground


def _soil_diffusivity(
    inputs: MethaneSoilGasTransportInputs,
    water_table_layer_column: jnp.ndarray,
    parameters: MethaneSoilGasTransportParameters,
) -> jnp.ndarray:
    temperature = _active_soil(inputs.soil_temperature_column_level)
    temperature_celsius = fortran_add(
        temperature,
        -jnp.asarray(
            parameters.freezing_temperature_kelvin,
            dtype=jnp.float64,
        ),
    )
    porosity = _active_soil(inputs.soil_porosity_column_level)
    water = jnp.minimum(
        porosity,
        _active_soil(inputs.volumetric_soil_water_column_level),
    )
    organic_matter = _active_soil(inputs.soil_organic_matter_column_level)
    soil_b = _active_soil(inputs.soil_b_parameter_column_level)
    liquid = _active_soil(inputs.soil_liquid_water_kg_m2_column_level)
    ice = _active_soil(inputs.soil_ice_kg_m2_column_level)
    water_density = jnp.asarray(
        parameters.water_density_kg_m3,
        dtype=jnp.float64,
    )
    ice_density = jnp.asarray(
        parameters.ice_density_kg_m3,
        dtype=jnp.float64,
    )
    small = jnp.asarray(parameters.minimum_transport_value, dtype=jnp.float64)
    satpow = jnp.asarray(
        parameters.saturated_diffusion_exponent,
        dtype=jnp.float64,
    )
    gas_scale = jnp.asarray(parameters.gas_diffusion_scale, dtype=jnp.float64)
    liquid_scale = jnp.asarray(
        parameters.liquid_diffusion_scale,
        dtype=jnp.float64,
    )
    gas_constants = jnp.asarray(
        parameters.gas_diffusion_constants_species_coefficient,
        dtype=jnp.float64,
    )
    water_constants = jnp.asarray(
        parameters.water_diffusion_constants_species_coefficient,
        dtype=jnp.float64,
    )
    level_number = jnp.arange(
        1,
        NUMBER_OF_METHANE_SOIL_LEVELS + 1,
        dtype=jnp.int32,
    )[None, :]
    above_water_table = level_number <= jnp.asarray(
        water_table_layer_column,
        dtype=jnp.int32,
    )[:, None]
    air_share = fortran_add(
        jnp.float64(1.0),
        -fortran_divide(water, porosity),
    )
    air_fraction = fortran_add(porosity, -water)
    organic_maximum = jnp.asarray(
        parameters.organic_matter_maximum_kg_m3,
        dtype=jnp.float64,
    )
    positive_organic_maximum = organic_maximum > jnp.float64(0.0)
    organic_fraction = jnp.where(
        positive_organic_maximum,
        jnp.minimum(
            fortran_divide(
                organic_matter,
                jnp.where(
                    positive_organic_maximum,
                    organic_maximum,
                    jnp.float64(1.0),
                ),
            ),
            jnp.float64(1.0),
        ),
        jnp.float64(1.0),
    )
    ten_thirds = fortran_divide(jnp.float64(10.0), jnp.float64(3.0))
    mineral_exponent = fortran_divide(jnp.float64(3.0), soil_b)
    peat_term = fortran_multiply(
        organic_fraction,
        _diffusion_power(air_share, ten_thirds, inputs.methane_column[:, None]),
    )
    peat_term = fortran_divide(
        peat_term,
        fortran_power(porosity, jnp.float64(2.0)),
    )
    mineral_term = _mineral_diffusion_term(
        organic_fraction, air_fraction, air_share, mineral_exponent,
        inputs.methane_column[:, None],
    )
    texture_factor = fortran_add(peat_term, mineral_term)

    liquid_volume = fortran_divide(liquid, water_density)
    ice_volume = fortran_divide(ice, ice_density)
    liquid_numerator = fortran_add(liquid_volume, small)
    phase_denominator = fortran_add(liquid_volume, ice_volume)
    phase_denominator = fortran_add(phase_denominator, small)
    species_results = []
    for species in range(NUMBER_OF_METHANE_GASES):
        gas_diffusivity = fortran_add(
            gas_constants[species, 0],
            fortran_multiply(
                gas_constants[species, 1],
                temperature_celsius,
            ),
        )
        gas_diffusivity = fortran_multiply(
            gas_diffusivity,
            jnp.float64(1.0e-4),
        )
        gas_diffusivity = fortran_multiply(
            gas_diffusivity,
            texture_factor,
        )
        gas_diffusivity = fortran_multiply(gas_diffusivity, gas_scale)

        water_diffusivity = fortran_multiply(
            _diffusion_power(porosity, satpow, inputs.methane_column[:, None]),
            _water_diffusion_polynomial(
                temperature_celsius,
                water_constants[species],
            ),
        )
        water_diffusivity = fortran_multiply(
            water_diffusivity,
            jnp.float64(1.0e-9),
        )
        water_diffusivity = fortran_multiply(
            water_diffusivity,
            liquid_scale,
        )
        water_diffusivity = jnp.where(
            temperature
            <= jnp.asarray(
                parameters.freezing_temperature_kelvin,
                dtype=jnp.float64,
            ),
            fortran_divide(
                fortran_multiply(water_diffusivity, liquid_numerator),
                phase_denominator,
            ),
            water_diffusivity,
        )
        diffusivity = jnp.where(
            above_water_table,
            gas_diffusivity,
            water_diffusivity,
        )
        species_results.append(jnp.maximum(diffusivity, small))
    return jnp.stack(species_results, axis=2)


def _effective_storage_and_relative_concentration(
    inputs: MethaneSoilGasTransportInputs,
    water_table_layer_column: jnp.ndarray,
    henry_ratio: jnp.ndarray,
    methane_concentration: jnp.ndarray,
    oxygen_concentration: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    porosity = _active_soil(inputs.soil_porosity_column_level)
    water = jnp.minimum(
        porosity,
        _active_soil(inputs.volumetric_soil_water_column_level),
    )
    level_number = jnp.arange(
        1,
        NUMBER_OF_METHANE_SOIL_LEVELS + 1,
        dtype=jnp.int32,
    )[None, :]
    above_water_table = level_number <= jnp.asarray(
        water_table_layer_column,
        dtype=jnp.int32,
    )[:, None]

    liquid_fraction = jnp.ones_like(porosity)
    gas_and_liquid_storage = fortran_add(
        jnp.float64(1.0),
        -henry_ratio[:, 1:, :],
    )
    gas_and_liquid_storage = fortran_multiply(
        gas_and_liquid_storage,
        water[:, :, None],
    )
    gas_and_liquid_storage = fortran_multiply(
        gas_and_liquid_storage,
        liquid_fraction[:, :, None],
    )
    gas_and_liquid_storage = fortran_add(
        porosity[:, :, None],
        -gas_and_liquid_storage,
    )
    liquid_storage = fortran_multiply(
        porosity[:, :, None],
        liquid_fraction[:, :, None],
    )
    effective_storage = jnp.where(
        above_water_table[:, :, None],
        gas_and_liquid_storage,
        liquid_storage,
    )

    column_gridcell = jnp.asarray(inputs.column_gridcell, dtype=jnp.int32)
    atmospheric_methane = jnp.take(
        jnp.asarray(inputs.atmospheric_methane_gridcell, dtype=jnp.float64),
        column_gridcell,
    )
    atmospheric_oxygen = jnp.take(
        jnp.asarray(inputs.atmospheric_oxygen_gridcell, dtype=jnp.float64),
        column_gridcell,
    )
    methane_relative = fortran_divide(
        _active_soil(methane_concentration),
        effective_storage[:, :, 0],
    )
    oxygen_relative = fortran_divide(
        _active_soil(oxygen_concentration),
        effective_storage[:, :, 1],
    )
    relative = jnp.stack(
        (
            jnp.concatenate(
                (atmospheric_methane[:, None], methane_relative),
                axis=1,
            ),
            jnp.concatenate(
                (atmospheric_oxygen[:, None], oxygen_relative),
                axis=1,
            ),
        ),
        axis=2,
    )
    return effective_storage, relative


def _interface_conductance(
    layer_thickness: jnp.ndarray,
    diffusivity: jnp.ndarray,
    species_ground_conductance: jnp.ndarray,
    henry_ratio_species: jnp.ndarray,
    water_table_layer_column: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    column_count = layer_thickness.shape[0]
    water_table = jnp.asarray(water_table_layer_column, dtype=jnp.int32)
    previous_diffusivity = jnp.concatenate(
        (diffusivity[:, :1], diffusivity[:, :-1]),
        axis=1,
    )
    next_diffusivity = jnp.concatenate(
        (diffusivity[:, 1:], diffusivity[:, -1:]),
        axis=1,
    )
    previous_thickness = jnp.concatenate(
        (layer_thickness[:, :1], layer_thickness[:, :-1]),
        axis=1,
    )
    next_thickness = jnp.concatenate(
        (layer_thickness[:, 1:], layer_thickness[:, -1:]),
        axis=1,
    )
    previous_henry = henry_ratio_species[:, :-1]

    resistance_here = fortran_divide(layer_thickness, diffusivity)
    resistance_previous = fortran_divide(
        previous_thickness,
        previous_diffusivity,
    )
    resistance_next = fortran_divide(next_thickness, next_diffusivity)
    normal_previous = fortran_divide(
        jnp.float64(2.0),
        fortran_add(resistance_here, resistance_previous),
    )
    normal_next = fortran_divide(
        jnp.float64(2.0),
        fortran_add(resistance_here, resistance_next),
    )
    cross_previous_resistance = fortran_divide(
        fortran_multiply(previous_thickness, previous_henry),
        previous_diffusivity,
    )
    cross_previous = fortran_divide(
        jnp.float64(2.0),
        fortran_add(resistance_here, cross_previous_resistance),
    )
    cross_next_resistance = fortran_divide(
        fortran_multiply(layer_thickness, henry_ratio_species[:, 1:]),
        diffusivity,
    )
    cross_next = fortran_divide(
        jnp.float64(2.0),
        fortran_add(cross_next_resistance, resistance_next),
    )

    top_layer_resistance = fortran_divide(
        layer_thickness[:, 0],
        fortran_multiply(diffusivity[:, 0], jnp.float64(2.0)),
    )
    top_normal = fortran_divide(
        jnp.float64(1.0),
        fortran_add(
            fortran_divide(jnp.float64(1.0), species_ground_conductance),
            top_layer_resistance,
        ),
    )
    top_surface_water = fortran_divide(
        jnp.float64(1.0),
        fortran_add(
            fortran_divide(
                henry_ratio_species[:, 0],
                species_ground_conductance,
            ),
            top_layer_resistance,
        ),
    )

    level_number = jnp.arange(
        1,
        NUMBER_OF_METHANE_SOIL_LEVELS + 1,
        dtype=jnp.int32,
    )[None, :]
    at_water_table = level_number == water_table[:, None]
    below_water_table_interface = level_number == water_table[:, None] + 1
    top = level_number == jnp.int32(1)
    bottom = level_number == jnp.int32(NUMBER_OF_METHANE_SOIL_LEVELS)

    previous = jnp.where(below_water_table_interface, cross_previous, normal_previous)
    previous = jnp.where(
        top,
        jnp.where(
            water_table[:, None] == jnp.int32(0),
            top_surface_water[:, None],
            top_normal[:, None],
        ),
        previous,
    )
    next_value = jnp.where(at_water_table, cross_next, normal_next)
    next_value = jnp.where(
        bottom,
        jnp.zeros((column_count, NUMBER_OF_METHANE_SOIL_LEVELS), dtype=jnp.float64),
        next_value,
    )
    return previous, next_value


def _tridiagonal_coefficients(
    layer_thickness: jnp.ndarray,
    effective_storage_species: jnp.ndarray,
    previous_conductance: jnp.ndarray,
    next_conductance: jnp.ndarray,
    henry_ratio_species: jnp.ndarray,
    water_table_layer_column: jnp.ndarray,
    methane_time_step: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    column_count = layer_thickness.shape[0]
    lower = jnp.zeros(
        (column_count, NUMBER_OF_METHANE_TRANSPORT_LEVELS),
        dtype=jnp.float64,
    )
    diagonal = jnp.ones_like(lower)
    upper = jnp.zeros_like(lower)
    half_over_thickness = fortran_divide(
        jnp.float64(0.5),
        layer_thickness,
    )
    lower_soil = fortran_multiply(
        -half_over_thickness,
        previous_conductance,
    )
    upper_soil = fortran_multiply(
        -half_over_thickness,
        next_conductance,
    )
    level_number = jnp.arange(
        1,
        NUMBER_OF_METHANE_SOIL_LEVELS + 1,
        dtype=jnp.int32,
    )[None, :]
    water_table = jnp.asarray(water_table_layer_column, dtype=jnp.int32)
    below_water_table_interface = level_number == water_table[:, None] + 1
    lower_soil = jnp.where(
        below_water_table_interface,
        fortran_multiply(lower_soil, henry_ratio_species[:, :-1]),
        lower_soil,
    )
    bottom = level_number == jnp.int32(NUMBER_OF_METHANE_SOIL_LEVELS)
    upper_soil = jnp.where(bottom, jnp.float64(0.0), upper_soil)

    at_water_table = level_number == water_table[:, None]
    diagonal_transport = jnp.where(
        at_water_table & ~bottom,
        fortran_add(
            fortran_multiply(
                next_conductance,
                henry_ratio_species[:, 1:],
            ),
            previous_conductance,
        ),
        fortran_add(next_conductance, previous_conductance),
    )
    diagonal_transport = jnp.where(
        bottom,
        previous_conductance,
        diagonal_transport,
    )
    diagonal_soil = fortran_add(
        fortran_divide(effective_storage_species, methane_time_step),
        fortran_multiply(half_over_thickness, diagonal_transport),
    )
    lower = lower.at[:, 1:].set(lower_soil)
    diagonal = diagonal.at[:, 1:].set(diagonal_soil)
    upper = upper.at[:, 1:].set(upper_soil)
    return lower, diagonal, upper


def _transport_right_hand_side(
    relative_concentration: jnp.ndarray,
    source: jnp.ndarray,
    effective_storage_species: jnp.ndarray,
    previous_conductance: jnp.ndarray,
    next_conductance: jnp.ndarray,
    henry_ratio_species: jnp.ndarray,
    water_table_layer_column: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    methane_time_step: jnp.ndarray,
) -> jnp.ndarray:
    current = relative_concentration[:, 1:]
    previous = relative_concentration[:, :-1]
    next_value = jnp.concatenate(
        (relative_concentration[:, 2:], relative_concentration[:, -1:]),
        axis=1,
    )
    level_number = jnp.arange(
        1,
        NUMBER_OF_METHANE_SOIL_LEVELS + 1,
        dtype=jnp.int32,
    )[None, :]
    water_table = jnp.asarray(water_table_layer_column, dtype=jnp.int32)
    at_water_table = level_number == water_table[:, None]
    below_water_table_interface = level_number == water_table[:, None] + 1
    bottom = level_number == jnp.int32(NUMBER_OF_METHANE_SOIL_LEVELS)

    next_difference = fortran_add(next_value, -current)
    next_difference_at_interface = fortran_add(
        next_value,
        -fortran_multiply(current, henry_ratio_species[:, 1:]),
    )
    next_difference = jnp.where(
        at_water_table,
        next_difference_at_interface,
        next_difference,
    )
    previous_difference = fortran_add(current, -previous)
    previous_difference_at_interface = fortran_add(
        current,
        -fortran_multiply(previous, henry_ratio_species[:, :-1]),
    )
    previous_difference = jnp.where(
        below_water_table_interface,
        previous_difference_at_interface,
        previous_difference,
    )
    next_flux = fortran_multiply(next_conductance, next_difference)
    previous_flux = fortran_multiply(
        previous_conductance,
        previous_difference,
    )
    diffusion = fortran_add(next_flux, -previous_flux)
    diffusion = jnp.where(bottom, -previous_flux, diffusion)
    storage = fortran_multiply(
        fortran_divide(effective_storage_species, methane_time_step),
        current,
    )
    transport = fortran_multiply(
        fortran_divide(jnp.float64(0.5), layer_thickness),
        diffusion,
    )
    source_average = fortran_multiply(
        jnp.float64(0.5),
        fortran_add(source, source),
    )
    soil_rhs = fortran_add(storage, transport)
    soil_rhs = fortran_add(soil_rhs, source_average)
    return jnp.concatenate(
        (relative_concentration[:, :1], soil_rhs),
        axis=1,
    )


def _tridiagonal_solve(lower, diagonal, upper, right_hand_side, initial_solution):
    from jax import default_backend

    from clm_jax.physics.tridiagonal import ordered_thomas_solve, tiled_thomas_solve
    if default_backend() == "gpu":
        return tiled_thomas_solve(lower, diagonal, upper, right_hand_side)
    return ordered_thomas_solve(lower, diagonal, upper, right_hand_side, initial_solution)


def _transport_branch_primal(
    inputs: MethaneSoilGasTransportInputs,
    branch: _BranchState,
    methane_production: jnp.ndarray,
    oxygen_aerenchyma: jnp.ndarray,
    water_table_layer_column: jnp.ndarray,
    ground_methane_conductance: jnp.ndarray,
    saturated: bool,
    parameters: MethaneSoilGasTransportParameters,
) -> tuple[_BranchState, jnp.ndarray, _BranchDiagnostics]:
    methane_column = jnp.asarray(inputs.methane_column, dtype=jnp.bool_)
    layer_thickness = _active_soil(
        inputs.soil_layer_thickness_column_level
    )
    methane_time_step = jnp.asarray(
        parameters.methane_time_step_seconds,
        dtype=jnp.float64,
    )
    land_time_step = jnp.asarray(
        parameters.land_time_step_seconds,
        dtype=jnp.float64,
    )
    water_table = jnp.asarray(water_table_layer_column, dtype=jnp.int32)

    henry_ratio = _henry_ratios(inputs, parameters)
    source, ebullition_total, surface_aerenchyma = _reaction_sources(
        water_table,
        layer_thickness,
        methane_production,
        oxygen_aerenchyma,
        branch,
    )
    effective_storage, relative = (
        _effective_storage_and_relative_concentration(
            inputs,
            water_table,
            henry_ratio,
            branch.methane_concentration_column_level,
            branch.oxygen_concentration_column_level,
        )
    )
    species_ground_conductance, positive_base_ground = (
        _species_ground_conductance(
            inputs,
            ground_methane_conductance,
            saturated,
            parameters,
        )
    )
    diffusivity = _soil_diffusivity(inputs, water_table, parameters)
    old_methane_relative = relative[:, :, 0]
    solved_species: list[jnp.ndarray] = []
    methane_surface_diffusion = jnp.asarray(
        branch.surface_diffusion_column,
        dtype=jnp.float64,
    )
    methane_surface_ebullition = jnp.asarray(
        branch.surface_ebullition_column,
        dtype=jnp.float64,
    )

    for species in range(NUMBER_OF_METHANE_GASES):
        previous_conductance, next_conductance = _interface_conductance(
            layer_thickness,
            diffusivity[:, :, species],
            species_ground_conductance[:, species],
            henry_ratio[:, :, species],
            water_table,
        )
        lower, diagonal, upper = _tridiagonal_coefficients(
            layer_thickness,
            effective_storage[:, :, species],
            previous_conductance,
            next_conductance,
            henry_ratio[:, :, species],
            water_table,
            methane_time_step,
        )
        right_hand_side = _transport_right_hand_side(
            relative[:, :, species],
            source[:, :, species],
            effective_storage[:, :, species],
            previous_conductance,
            next_conductance,
            henry_ratio[:, :, species],
            water_table,
            layer_thickness,
            methane_time_step,
        )
        solved = _tridiagonal_solve(
            lower,
            diagonal,
            upper,
            right_hand_side,
            relative[:, :, species],
        )

        if species == 0:
            atmospheric = relative[:, 0, 0]
            mean_surface_concentration = fortran_divide(
                fortran_add(solved[:, 1], old_methane_relative[:, 1]),
                jnp.float64(2.0),
            )
            atmospheric_boundary = jnp.where(
                water_table == jnp.int32(0),
                fortran_multiply(atmospheric, henry_ratio[:, 0, 0]),
                atmospheric,
            )
            methane_surface_diffusion = fortran_multiply(
                previous_conductance[:, 0],
                fortran_add(
                    mean_surface_concentration,
                    -atmospheric_boundary,
                ),
            )
            methane_surface_ebullition = jnp.where(
                water_table == jnp.int32(0),
                ebullition_total,
                jnp.float64(0.0),
            )

            solved, methane_surface_diffusion = clamp_negative_methane(
                solved,
                effective_storage[:, :, 0],
                layer_thickness,
                methane_surface_diffusion,
                methane_time_step,
            )
        else:
            atmospheric_oxygen = relative[:, 0, 1]
            oxygen_cap = fortran_divide(
                atmospheric_oxygen[:, None],
                effective_storage[:, :, 1],
            )
            oxygen_soil = jnp.maximum(
                solved[:, 1:],
                jnp.asarray(
                    parameters.minimum_transport_value,
                    dtype=jnp.float64,
                ),
            )
            oxygen_soil = jnp.minimum(oxygen_soil, oxygen_cap)
            solved = solved.at[:, 1:].set(oxygen_soil)
        solved_species.append(solved)

    solved_relative = jnp.stack(solved_species, axis=2)
    methane_concentration = fortran_multiply(
        solved_relative[:, 1:, 0],
        effective_storage[:, :, 0],
    )
    oxygen_concentration = fortran_multiply(
        solved_relative[:, 1:, 1],
        effective_storage[:, :, 1],
    )
    methane_before = _active_soil(branch.methane_concentration_column_level)
    methane_production_active = _active_soil(methane_production)
    methane_oxidation = _active_soil(branch.methane_oxidation_column_level)

    methane_error = methane_balance(
        methane_concentration,
        methane_before,
        methane_production_active,
        methane_oxidation,
        layer_thickness,
        land_time_step,
    )
    total_surface_flux = fortran_add(
        surface_aerenchyma,
        methane_surface_ebullition,
    )
    total_surface_flux = fortran_add(
        total_surface_flux,
        methane_surface_diffusion,
    )
    methane_error = fortran_add(
        methane_error,
        fortran_multiply(total_surface_flux, land_time_step),
    )
    conservation_ok = jnp.abs(methane_error) < jnp.asarray(
        parameters.conservation_tolerance_mol_m2,
        dtype=jnp.float64,
    )
    corrected_surface_diffusion = fortran_add(
        methane_surface_diffusion,
        -fortran_divide(methane_error, land_time_step),
    )
    methane_surface_diffusion = jnp.where(
        conservation_ok,
        corrected_surface_diffusion,
        methane_surface_diffusion,
    )

    result = _BranchState(
        methane_concentration_column_level=_preserve_level_state(
            methane_column,
            branch.methane_concentration_column_level,
            methane_concentration,
        ),
        oxygen_concentration_column_level=_preserve_level_state(
            methane_column,
            branch.oxygen_concentration_column_level,
            oxygen_concentration,
        ),
        methane_oxidation_column_level=branch.methane_oxidation_column_level,
        oxygen_oxidation_column_level=branch.oxygen_oxidation_column_level,
        methane_aerenchyma_column_level=branch.methane_aerenchyma_column_level,
        methane_ebullition_column_level=branch.methane_ebullition_column_level,
        oxygen_decomposition_column_level=(
            branch.oxygen_decomposition_column_level
        ),
        oxygen_stress_column_level=branch.oxygen_stress_column_level,
        methane_stress_column_level=branch.methane_stress_column_level,
        surface_aerenchyma_column=_preserve_column_state(
            methane_column,
            branch.surface_aerenchyma_column,
            surface_aerenchyma,
        ),
        ebullition_total_column=_preserve_column_state(
            methane_column,
            branch.ebullition_total_column,
            ebullition_total,
        ),
        surface_ebullition_column=_preserve_column_state(
            methane_column,
            branch.surface_ebullition_column,
            methane_surface_ebullition,
        ),
        surface_diffusion_column=_preserve_column_state(
            methane_column,
            branch.surface_diffusion_column,
            methane_surface_diffusion,
        ),
    )
    next_ground_conductance = _preserve_column_state(
        methane_column,
        ground_methane_conductance,
        species_ground_conductance[:, 0],
    )
    return (
        result,
        next_ground_conductance,
        _BranchDiagnostics(
            henry_ratio_column_level_species=henry_ratio,
            effective_storage_column_level_species=effective_storage,
            diffusivity_column_level_species=diffusivity,
            species_ground_conductance_column_species=(
                species_ground_conductance
            ),
            raw_methane_conservation_error_column=methane_error,
            conservation_violation_column=methane_column & ~conservation_ok,
        ),
    )


@partial(jax.custom_jvp, nondiff_argnums=(6,))
def _transport_branch(inputs, branch, methane_production, oxygen_aerenchyma,
                      water_table, ground_conductance, saturated, parameters):
    return _transport_branch_primal(
        inputs, branch, methane_production, oxygen_aerenchyma,
        water_table, ground_conductance, saturated, parameters,
    )


@_transport_branch.defjvp
def _transport_branch_jvp(saturated, primals, tangents):
    inputs = primals[0]
    active = inputs.methane_column

    def column_mask(value):
        return active.reshape((active.size, *((1,) * (value.ndim - 1))))

    def guarded(inputs, branch, methane_production, oxygen_aerenchyma,
                water_table, ground_conductance, parameters):
        defaults = dict(
            soil_layer_thickness_column_level=.1,
            snow_layer_thickness_column_level=1.,
            soil_b_parameter_column_level=4., soil_porosity_column_level=.5,
            soil_organic_matter_column_level=1.,
            soil_temperature_column_level=280., snow_temperature_column_level=273.15,
            ground_temperature_column=280., surface_water_temperature_column=280.,
            volumetric_soil_water_column_level=.25,
            soil_liquid_water_kg_m2_column_level=10., soil_ice_kg_m2_column_level=1.,
            snow_liquid_water_kg_m2_column_level=1., snow_ice_kg_m2_column_level=1.,
        )
        grid_active = jnp.zeros_like(inputs.atmospheric_methane_gridcell, dtype=jnp.bool_)
        grid_active = grid_active.at[inputs.column_gridcell].max(active)
        fields = {}
        for name, field in inputs._asdict().items():
            if jnp.issubdtype(field.dtype, jnp.floating):
                mask = grid_active if name.startswith("atmospheric_") else column_mask(field)
                default = 1. if name.startswith("atmospheric_") else defaults.get(name, 0.)
                fields[name] = jnp.where(mask, field, jnp.float64(default))
        safe_inputs = inputs._replace(**fields)
        safe_branch = branch._replace(**{
            name: jnp.where(column_mask(field), field,
                            jnp.float64(1. if "concentration" in name else 0.))
            for name, field in branch._asdict().items()
        })
        result, ground, diagnostics = _transport_branch_primal(
            safe_inputs, safe_branch,
            jnp.where(column_mask(methane_production), methane_production, 0.),
            jnp.where(column_mask(oxygen_aerenchyma), oxygen_aerenchyma, 0.),
            jnp.where(active, water_table, jnp.int32(20)),
            jnp.where(active, ground_conductance, jnp.float64(1.)),
            saturated, parameters,
        )
        result = jax.tree.map(
            lambda value, old: jnp.where(column_mask(value), value, old), result, branch,
        )
        ground = jnp.where(active, ground, ground_conductance)
        diagnostics = jax.tree.map(
            lambda field: jnp.where(column_mask(field), field, jnp.zeros_like(field)), diagnostics,
        )
        return result, ground, diagnostics

    value = _transport_branch_primal(*primals[:6], saturated, primals[6])
    return value, jax.jvp(guarded, primals, tangents)[1]


def methane_soil_gas_transport_competition_active_case(
    inputs: MethaneSoilGasTransportInputs,
    state: MethaneSoilGasTransportState,
    parameters: MethaneSoilGasTransportParameters,
) -> MethaneGasCompetitionResult:

    methane_column = jnp.asarray(inputs.methane_column, dtype=jnp.bool_)
    competition_inputs = MethaneGasCompetitionInputs(
        methane_column=methane_column,
        unsaturated_methane_concentration_column_level=(
            state.unsaturated_methane_concentration_column_level
        ),
        saturated_methane_concentration_column_level=(
            state.saturated_methane_concentration_column_level
        ),
        unsaturated_oxygen_concentration_column_level=(
            state.unsaturated_oxygen_concentration_column_level
        ),
        saturated_oxygen_concentration_column_level=(
            state.saturated_oxygen_concentration_column_level
        ),
        unsaturated_methane_production_column_level=(
            inputs.unsaturated_methane_production_column_level
        ),
        saturated_methane_production_column_level=(
            inputs.saturated_methane_production_column_level
        ),
        unsaturated_oxygen_aerenchyma_column_level=(
            inputs.unsaturated_oxygen_aerenchyma_column_level
        ),
        saturated_oxygen_aerenchyma_column_level=(
            inputs.saturated_oxygen_aerenchyma_column_level
        ),
    )
    return methane_gas_competition_active_case(
        competition_inputs,
        state.competition,
        MethaneGasCompetitionParameters(
            time_step_seconds=parameters.land_time_step_seconds
        ),
    )


def _initial_transport_branch_state(
    state: MethaneSoilGasTransportState,
    competition: MethaneGasCompetitionState,
    *,
    saturated: bool,
) -> _BranchState:
    prefix = "saturated" if saturated else "unsaturated"
    return _BranchState(
        methane_concentration_column_level=getattr(
            state,
            f"{prefix}_methane_concentration_column_level",
        ),
        oxygen_concentration_column_level=getattr(
            state,
            f"{prefix}_oxygen_concentration_column_level",
        ),
        methane_oxidation_column_level=getattr(
            competition,
            f"{prefix}_methane_oxidation_column_level",
        ),
        oxygen_oxidation_column_level=getattr(
            competition,
            f"{prefix}_oxygen_oxidation_column_level",
        ),
        methane_aerenchyma_column_level=getattr(
            competition,
            f"{prefix}_methane_aerenchyma_column_level",
        ),
        methane_ebullition_column_level=getattr(
            competition,
            f"{prefix}_methane_ebullition_column_level",
        ),
        oxygen_decomposition_column_level=getattr(
            competition,
            f"{prefix}_oxygen_decomposition_column_level",
        ),
        oxygen_stress_column_level=getattr(
            competition,
            f"{prefix}_oxygen_stress_column_level",
        ),
        methane_stress_column_level=getattr(
            competition,
            f"{prefix}_methane_stress_column_level",
        ),
        surface_aerenchyma_column=getattr(
            state,
            f"{prefix}_surface_aerenchyma_column",
        ),
        ebullition_total_column=getattr(
            state,
            f"{prefix}_ebullition_total_column",
        ),
        surface_ebullition_column=getattr(
            state,
            f"{prefix}_surface_ebullition_column",
        ),
        surface_diffusion_column=getattr(
            state,
            f"{prefix}_surface_diffusion_column",
        ),
    )


def methane_soil_gas_transport_unsaturated_branch_active_case(
    inputs: MethaneSoilGasTransportInputs,
    state: MethaneSoilGasTransportState,
    competition: MethaneGasCompetitionState,
    parameters: MethaneSoilGasTransportParameters,
) -> MethaneSoilGasTransportBranchResult:

    branch, ground, diagnostics = _transport_branch(
        inputs,
        _initial_transport_branch_state(
            state,
            competition,
            saturated=False,
        ),
        inputs.unsaturated_methane_production_column_level,
        inputs.unsaturated_oxygen_aerenchyma_column_level,
        inputs.unsaturated_water_table_layer_column,
        state.ground_methane_conductance_column,
        False,
        parameters,
    )
    return MethaneSoilGasTransportBranchResult(branch, ground, diagnostics)


def methane_soil_gas_transport_saturated_branch_active_case(
    inputs: MethaneSoilGasTransportInputs,
    state: MethaneSoilGasTransportState,
    competition: MethaneGasCompetitionState,
    ground_after_unsaturated: jnp.ndarray,
    parameters: MethaneSoilGasTransportParameters,
) -> MethaneSoilGasTransportBranchResult:

    branch, ground, diagnostics = _transport_branch(
        inputs,
        _initial_transport_branch_state(
            state,
            competition,
            saturated=True,
        ),
        inputs.saturated_methane_production_column_level,
        inputs.saturated_oxygen_aerenchyma_column_level,
        jnp.zeros_like(
            inputs.unsaturated_water_table_layer_column,
            dtype=jnp.int32,
        ),
        ground_after_unsaturated,
        True,
        parameters,
    )
    return MethaneSoilGasTransportBranchResult(branch, ground, diagnostics)


def compose_methane_soil_gas_transport_result(
    inputs: MethaneSoilGasTransportInputs,
    competition_result: MethaneGasCompetitionResult,
    unsaturated: MethaneSoilGasTransportBranchResult,
    saturated: MethaneSoilGasTransportBranchResult,
) -> MethaneSoilGasTransportResult:

    methane_column = jnp.asarray(inputs.methane_column, dtype=jnp.bool_)
    unsaturated_state = unsaturated.state
    saturated_state = saturated.state
    final_competition = MethaneGasCompetitionState(
        unsaturated_methane_oxidation_column_level=(
            unsaturated_state.methane_oxidation_column_level
        ),
        saturated_methane_oxidation_column_level=(
            saturated_state.methane_oxidation_column_level
        ),
        unsaturated_oxygen_oxidation_column_level=(
            unsaturated_state.oxygen_oxidation_column_level
        ),
        saturated_oxygen_oxidation_column_level=(
            saturated_state.oxygen_oxidation_column_level
        ),
        unsaturated_methane_aerenchyma_column_level=(
            unsaturated_state.methane_aerenchyma_column_level
        ),
        saturated_methane_aerenchyma_column_level=(
            saturated_state.methane_aerenchyma_column_level
        ),
        unsaturated_methane_ebullition_column_level=(
            unsaturated_state.methane_ebullition_column_level
        ),
        saturated_methane_ebullition_column_level=(
            saturated_state.methane_ebullition_column_level
        ),
        unsaturated_oxygen_decomposition_column_level=(
            unsaturated_state.oxygen_decomposition_column_level
        ),
        saturated_oxygen_decomposition_column_level=(
            saturated_state.oxygen_decomposition_column_level
        ),
        unsaturated_oxygen_stress_column_level=(
            unsaturated_state.oxygen_stress_column_level
        ),
        saturated_oxygen_stress_column_level=(
            saturated_state.oxygen_stress_column_level
        ),
        unsaturated_methane_stress_column_level=(
            unsaturated_state.methane_stress_column_level
        ),
        saturated_methane_stress_column_level=(
            saturated_state.methane_stress_column_level
        ),
    )
    result_state = MethaneSoilGasTransportState(
        competition=final_competition,
        unsaturated_methane_concentration_column_level=(
            unsaturated_state.methane_concentration_column_level
        ),
        saturated_methane_concentration_column_level=(
            saturated_state.methane_concentration_column_level
        ),
        unsaturated_oxygen_concentration_column_level=(
            unsaturated_state.oxygen_concentration_column_level
        ),
        saturated_oxygen_concentration_column_level=(
            saturated_state.oxygen_concentration_column_level
        ),
        unsaturated_surface_aerenchyma_column=(
            unsaturated_state.surface_aerenchyma_column
        ),
        saturated_surface_aerenchyma_column=(
            saturated_state.surface_aerenchyma_column
        ),
        unsaturated_ebullition_total_column=(
            unsaturated_state.ebullition_total_column
        ),
        saturated_ebullition_total_column=(
            saturated_state.ebullition_total_column
        ),
        unsaturated_surface_ebullition_column=(
            unsaturated_state.surface_ebullition_column
        ),
        saturated_surface_ebullition_column=(
            saturated_state.surface_ebullition_column
        ),
        unsaturated_surface_diffusion_column=(
            unsaturated_state.surface_diffusion_column
        ),
        saturated_surface_diffusion_column=(
            saturated_state.surface_diffusion_column
        ),
        ground_methane_conductance_column=(
            saturated.ground_methane_conductance_column
        ),
    )
    return MethaneSoilGasTransportResult(
        state=result_state,
        diagnostics=MethaneSoilGasTransportDiagnostics(
            competition=competition_result,
            unsaturated_henry_ratio_column_level_species=(
                unsaturated.diagnostics.henry_ratio_column_level_species
            ),
            saturated_henry_ratio_column_level_species=(
                saturated.diagnostics.henry_ratio_column_level_species
            ),
            unsaturated_effective_storage_column_level_species=(
                unsaturated.diagnostics.effective_storage_column_level_species
            ),
            saturated_effective_storage_column_level_species=(
                saturated.diagnostics.effective_storage_column_level_species
            ),
            unsaturated_diffusivity_column_level_species=(
                unsaturated.diagnostics.diffusivity_column_level_species
            ),
            saturated_diffusivity_column_level_species=(
                saturated.diagnostics.diffusivity_column_level_species
            ),
            unsaturated_species_ground_conductance_column_species=(
                unsaturated.diagnostics.species_ground_conductance_column_species
            ),
            saturated_species_ground_conductance_column_species=(
                saturated.diagnostics.species_ground_conductance_column_species
            ),
            unsaturated_raw_methane_conservation_error_column=(
                unsaturated.diagnostics.raw_methane_conservation_error_column
            ),
            saturated_raw_methane_conservation_error_column=(
                saturated.diagnostics.raw_methane_conservation_error_column
            ),
            methane_conservation_violation_column=(
                unsaturated.diagnostics.conservation_violation_column
                | saturated.diagnostics.conservation_violation_column
            ),
            methane_column_count=jnp.sum(methane_column, dtype=jnp.int32),
        ),
    )


def methane_soil_gas_transport_active_case(
    inputs: MethaneSoilGasTransportInputs,
    state: MethaneSoilGasTransportState,
    parameters: MethaneSoilGasTransportParameters = (
        DEFAULT_METHANE_SOIL_GAS_TRANSPORT_PARAMETERS
    ),
) -> MethaneSoilGasTransportResult:

    _validate_shapes(inputs, state, parameters)
    competition_result = methane_soil_gas_transport_competition_active_case(
        inputs,
        state,
        parameters,
    )
    competition = competition_result.state

    unsaturated = methane_soil_gas_transport_unsaturated_branch_active_case(
        inputs,
        state,
        competition,
        parameters,
    )

    saturated = methane_soil_gas_transport_saturated_branch_active_case(
        inputs,
        state,
        competition,
        unsaturated.ground_methane_conductance_column,
        parameters,
    )
    return compose_methane_soil_gas_transport_result(
        inputs,
        competition_result,
        unsaturated,
        saturated,
    )


__all__ = [
    "DEFAULT_METHANE_SOIL_GAS_TRANSPORT_PARAMETERS",
    "NUMBER_OF_METHANE_GASES",
    "NUMBER_OF_METHANE_SNOW_LEVELS",
    "NUMBER_OF_METHANE_TRANSPORT_LEVELS",
    "MethaneSoilGasTransportDiagnostics",
    "MethaneSoilGasTransportBranchResult",
    "MethaneSoilGasTransportInputs",
    "MethaneSoilGasTransportParameters",
    "MethaneSoilGasTransportResult",
    "MethaneSoilGasTransportState",
    "compose_methane_soil_gas_transport_result",
    "methane_soil_gas_transport_active_case",
    "methane_soil_gas_transport_competition_active_case",
    "methane_soil_gas_transport_saturated_branch_active_case",
    "methane_soil_gas_transport_unsaturated_branch_active_case",
]
