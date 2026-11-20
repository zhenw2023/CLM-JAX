
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.forcing.datm import DatmForcing
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_LAKE,
    LANDUNIT_SOIL,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    SubgridTopology,
)
from clm_jax.physics.diagnostics_and_accumulators import (
    AccumulatorClock,
    ConfiguredAccumulatorInputs,
    ConfiguredAccumulatorParameters,
    ConfiguredAccumulatorState,
    update_configured_accumulators,
)
from clm_jax.physics.ecosystem_dynamics_carry import (
    EcosystemDynamicsPostDrainageCarry,
)
from clm_jax.physics.ecosystem_dynamics_post_drainage import (
    EcosystemDynamicsPostDrainageResult,
)
from clm_jax.physics.end_of_step_vegetation import (
    end_of_step_vegetation_dynamics,
)
from clm_jax.physics.hydrology_drainage_driver import (
    HydrologyDrainageDriverResult,
)
from clm_jax.physics.lake_hydrology_driver import LakeHydrologyDriverResult
from clm_jax.physics.lnd2glc import (
    LandToGlacierInputs,
    LandToGlacierResult,
    land_to_glacier_coupling,
)
from clm_jax.physics.numerics import fortran_add, fortran_multiply
from clm_jax.physics.physical_driver import (
    PhysicalDriverResult,
    PhysicalDriverRuntimeResult,
)
from clm_jax.physics.radiation_driver import datm_shortwave_bands
from clm_jax.physics.snow_water import NUMBER_OF_SNOW_LEVELS
from clm_jax.trajectory import NoleapModelClock


class EndOfTimestepCompositionParameters(NamedTuple):

    atmospheric_co2_ppmv: float
    atmospheric_oxygen_mole_fraction: float
    crop_latitude_base_temperature_intercept: float
    crop_latitude_base_temperature_slope: float


class EndOfTimestepPhysicalCarry(NamedTuple):

    sunlit_fraction_patch: jnp.ndarray


class EndOfTimestepInputs(NamedTuple):

    land_to_glacier: LandToGlacierInputs
    accumulators: ConfiguredAccumulatorInputs


class EndOfTimestepState(NamedTuple):

    accumulators: ConfiguredAccumulatorState
    vegetation: Any


class EndOfTimestepResult(NamedTuple):

    state: EndOfTimestepState
    history_state: EndOfTimestepState
    land_to_glacier: LandToGlacierResult


def accumulator_clock(clock: NoleapModelClock) -> AccumulatorClock:

    return AccumulatorClock(
        nstep=clock.model_step,
        month=clock.month,
        day=clock.day,
        seconds_into_day=clock.seconds_into_day,
        end_current_day=(
            jnp.asarray(clock.seconds_into_day, dtype=jnp.int32) == jnp.int32(0)
        ),
    )


def _current_physical_fields(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
):
    fields = physical.land_to_atmosphere
    if fields is None:
        raise ValueError(
            "physical result lacks its current land-to-atmosphere projection"
        )
    return fields


def end_of_timestep_physical_carry(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
) -> EndOfTimestepPhysicalCarry:

    radiation = physical.radiation
    if radiation is None:
        raise ValueError(
            "physical result lacks current radiation state; project the "
            "end-of-timestep physical carry before releasing stage 230"
        )
    return EndOfTimestepPhysicalCarry(
        sunlit_fraction_patch=jnp.asarray(
            radiation.state.canopy_surface.canopy_sunshade.sunlit_fraction_patch,
            dtype=jnp.float64,
        )
    )


def compose_land_to_glacier_inputs(
    topology: SubgridTopology,
    template: LandToGlacierInputs,
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    lake_hydrology: LakeHydrologyDriverResult,
    drainage: HydrologyDrainageDriverResult,
) -> LandToGlacierInputs:

    column_count = topology.column.gridcell.shape[0]
    lake = lake_hydrology.hydrology.state
    temperature = jnp.asarray(
        lake.temperature.soil_snow_temperature_column_level,
        dtype=jnp.float64,
    )
    if temperature.ndim != 2 or temperature.shape[0] != column_count:
        raise ValueError("final soil/snow temperature and column topology disagree")
    surface_temperature = temperature[:, NUMBER_OF_SNOW_LEVELS]

    growth = jnp.asarray(
        drainage.inputs.glacier_ice_growth_flux_column,
        dtype=jnp.float64,
    )
    melt = jnp.asarray(
        drainage.inputs.glacier_ice_melt_flux_column,
        dtype=jnp.float64,
    )
    if growth.shape != (column_count,) or melt.shape != (column_count,):
        raise ValueError("current glacier-SMB flux and column topology disagree")
    net_ice = fortran_add(growth, -melt)

    topography = physical.atmospheric.state.column_topography
    elevation = jnp.asarray(topography.elevation, dtype=jnp.float64)
    column_type = jnp.asarray(topography.column_type, dtype=jnp.int32)
    if elevation.shape != (column_count,) or column_type.shape != (column_count,):
        raise ValueError("current atmospheric topography and column topology disagree")

    return template._replace(
        column_gridcell=topology.column.gridcell,
        column_landunit_type=topology.column.landunit_type,
        column_type=column_type,
        column_weight_on_gridcell=topology.column.weight_on_gridcell,
        landunit_gridcell=topology.landunit.gridcell,
        landunit_type=topology.landunit.landunit_type,
        landunit_weight_on_gridcell=topology.landunit.weight_on_gridcell,
        do_surface_mass_balance=(drainage.inputs.do_surface_mass_balance_column),
        surface_temperature=surface_temperature,
        elevation=elevation,
        ice_flux=net_ice,
        initialization=jnp.asarray(False, dtype=jnp.bool_),
    )


def _latitude_base_temperature(
    topology: SubgridTopology,
    parameters: ConfiguredAccumulatorParameters,
    composition: EndOfTimestepCompositionParameters,
) -> jnp.ndarray:

    patch_gridcell = jnp.asarray(topology.patch.gridcell, dtype=jnp.int32)
    patch_pft = jnp.asarray(topology.patch.vegetation_type, dtype=jnp.int32)
    latitude = jnp.asarray(topology.grid.latitude_degrees, dtype=jnp.float64)[
        patch_gridcell
    ]
    base = jnp.asarray(
        parameters.crop_base_temperature_by_pft,
        dtype=jnp.float64,
    )[patch_pft]
    intercept = jnp.float64(composition.crop_latitude_base_temperature_intercept)
    slope = jnp.float64(composition.crop_latitude_base_temperature_slope)
    northern = fortran_add(
        fortran_add(base, intercept),
        -fortran_multiply(slope, latitude),
    )
    southern = fortran_add(
        fortran_add(base, intercept),
        fortran_multiply(slope, latitude),
    )
    return jnp.where(
        (latitude >= jnp.float64(0.0)) & (latitude <= jnp.float64(30.0)),
        northern,
        jnp.where(
            (latitude < jnp.float64(0.0)) & (latitude >= jnp.float64(-30.0)),
            southern,
            base,
        ),
    )


def compose_accumulator_inputs(
    topology: SubgridTopology,
    forcing: DatmForcing,
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    physical_carry: EndOfTimestepPhysicalCarry,
    lake_hydrology: LakeHydrologyDriverResult,
    post_drainage: EcosystemDynamicsPostDrainageResult,
    ecosystem_carry: EcosystemDynamicsPostDrainageCarry,
    parameters: ConfiguredAccumulatorParameters,
    composition: EndOfTimestepCompositionParameters,
) -> ConfiguredAccumulatorInputs:

    patch_count = topology.patch.gridcell.shape[0]
    column_count = topology.column.gridcell.shape[0]
    gridcell_count = topology.grid.latitude_degrees.shape[0]
    direct, diffuse = datm_shortwave_bands(forcing)
    pressure_gridcell = jnp.asarray(forcing.pressure, dtype=jnp.float64)
    if pressure_gridcell.shape != (gridcell_count,):
        raise ValueError("DATM pressure and land-grid topology disagree")
    co2_mole_fraction = fortran_multiply(
        jnp.float64(composition.atmospheric_co2_ppmv),
        jnp.float64(1.0e-6),
    )
    co2_partial_pressure = fortran_multiply(
        co2_mole_fraction,
        pressure_gridcell,
    )
    oxygen_partial_pressure = fortran_multiply(
        jnp.float64(composition.atmospheric_oxygen_mole_fraction),
        pressure_gridcell,
    )

    atmospheric = physical.atmospheric.column_forcing
    final_physical = _current_physical_fields(physical)
    reference_temperature = jnp.asarray(
        final_physical.reference_temperature_2m_patch,
        dtype=jnp.float64,
    )
    if reference_temperature.shape != (patch_count,):
        raise ValueError("current reference temperature and patch topology disagree")
    landunit_type = jnp.asarray(topology.patch.landunit_type, dtype=jnp.int32)
    urban = (landunit_type >= jnp.int32(LANDUNIT_URBAN_MIN)) & (
        landunit_type <= jnp.int32(LANDUNIT_URBAN_MAX)
    )
    rural = (landunit_type == jnp.int32(LANDUNIT_SOIL)) | (
        landunit_type == jnp.int32(LANDUNIT_CROP)
    )
    special = jnp.float64(CLM_SPECIAL_VALUE)
    urban_reference = jnp.where(urban, reference_temperature, special)
    rural_reference = jnp.where(rural, reference_temperature, special)

    lake = lake_hydrology.hydrology.state
    soil_temperature = jnp.asarray(
        lake.temperature.soil_snow_temperature_column_level[
            :, NUMBER_OF_SNOW_LEVELS:
        ],
        dtype=jnp.float64,
    )
    soil_thickness = jnp.asarray(
        lake.geometry.soil_snow_thickness_column_level[:, NUMBER_OF_SNOW_LEVELS:],
        dtype=jnp.float64,
    )
    if (
        soil_temperature.ndim != 2
        or soil_temperature.shape[0] != column_count
        or soil_temperature.shape != soil_thickness.shape
        or soil_temperature.shape[1] < 2
    ):
        raise ValueError("final soil temperature/thickness axes are inconsistent")

    crop = ecosystem_carry.crop_accumulators
    if crop is None:
        raise ValueError("ecosystem carry lacks current crop accumulator state")
    vegetation_structure = post_drainage.state.vegetation_structure
    sunlit_fraction = jnp.asarray(
        physical_carry.sunlit_fraction_patch,
        dtype=jnp.float64,
    )
    exposed_leaf_area = jnp.asarray(
        vegetation_structure.exposed_leaf_area_index_patch,
        dtype=jnp.float64,
    )
    final_vegetation_temperature = final_physical.vegetation_temperature_patch
    if final_vegetation_temperature is None:
        final_vegetation_temperature = (
            physical.canopy_flux.state.vegetation_temperature_patch
        )
    vegetation_temperature = jnp.asarray(
        final_vegetation_temperature,
        dtype=jnp.float64,
    )
    canopy_flux = physical.canopy_flux.ecosystem
    btran_bare_complement = jnp.where(
        (landunit_type == jnp.int32(LANDUNIT_LAKE)) | urban,
        special,
        jnp.float64(0.0),
    )
    transpiration_stress = jnp.where(
        jnp.asarray(canopy_flux.vegetation_exposed_patch, dtype=jnp.bool_),
        jnp.asarray(canopy_flux.transpiration_wetness_patch, dtype=jnp.float64),
        btran_bare_complement,
    )
    for name, value in (
        ("sunlit fraction", sunlit_fraction),
        ("exposed leaf area", exposed_leaf_area),
        ("vegetation temperature", vegetation_temperature),
        ("transpiration stress", transpiration_stress),
        ("crop live", crop.crop_live_patch),
        ("crop GDD", crop.growing_degree_days_since_planting_patch),
        ("crop vernalization", crop.vernalization_factor_patch),
    ):
        if jnp.shape(value) != (patch_count,):
            raise ValueError(f"current {name} and patch topology disagree")

    return ConfiguredAccumulatorInputs(
        direct_solar_gridcell_band=direct,
        diffuse_solar_gridcell_band=diffuse,
        rain_downscaled_column=atmospheric.rain,
        snow_downscaled_column=atmospheric.snow,
        relative_humidity_gridcell=forcing.relative_humidity,
        co2_partial_pressure_gridcell=co2_partial_pressure,
        oxygen_partial_pressure_gridcell=oxygen_partial_pressure,
        air_pressure_downscaled_column=atmospheric.pressure,
        vegetation_temperature_patch=vegetation_temperature,
        reference_temperature_patch=reference_temperature,
        urban_reference_temperature_patch=urban_reference,
        rural_reference_temperature_patch=rural_reference,
        sunlit_fraction_patch=sunlit_fraction,
        exposed_leaf_area_index_patch=exposed_leaf_area,
        total_evapotranspiration_flux_column=(
            lake.water_flux.total_evaporation_flux_column
        ),
        transpiration_stress_patch=transpiration_stress,
        crop_live_patch=crop.crop_live_patch,
        crop_growing_degree_days_plant_patch=(
            crop.growing_degree_days_since_planting_patch
        ),
        crop_vernalization_factor_patch=crop.vernalization_factor_patch,
        crop_latitude_base_temperature_patch=_latitude_base_temperature(
            topology,
            parameters,
            composition,
        ),
        soil_temperature_column_level=soil_temperature,
        soil_layer_thickness_column_level=soil_thickness,
    )


def compose_end_of_timestep_inputs(
    topology: SubgridTopology,
    forcing: DatmForcing,
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    physical_carry: EndOfTimestepPhysicalCarry,
    lake_hydrology: LakeHydrologyDriverResult,
    drainage: HydrologyDrainageDriverResult,
    post_drainage: EcosystemDynamicsPostDrainageResult,
    ecosystem_carry: EcosystemDynamicsPostDrainageCarry,
    land_to_glacier_template: LandToGlacierInputs,
    accumulator_parameters: ConfiguredAccumulatorParameters,
    composition_parameters: EndOfTimestepCompositionParameters,
) -> EndOfTimestepInputs:

    return EndOfTimestepInputs(
        land_to_glacier=compose_land_to_glacier_inputs(
            topology,
            land_to_glacier_template,
            physical,
            lake_hydrology,
            drainage,
        ),
        accumulators=compose_accumulator_inputs(
            topology,
            forcing,
            physical,
            physical_carry,
            lake_hydrology,
            post_drainage,
            ecosystem_carry,
            accumulator_parameters,
            composition_parameters,
        ),
    )


def end_of_timestep_step(
    topology: SubgridTopology,
    inputs: EndOfTimestepInputs,
    state: EndOfTimestepState,
    clock: NoleapModelClock,
    accumulator_parameters: ConfiguredAccumulatorParameters,
    *,
    number_of_gridcells: int,
    maximum_glacier_classes: int,
    use_cndv: bool,
) -> EndOfTimestepResult:

    glacier = land_to_glacier_coupling(
        inputs.land_to_glacier,
        number_of_gridcells=number_of_gridcells,
        maximum_glacier_classes=maximum_glacier_classes,
    )
    current_clock = accumulator_clock(clock)
    accumulators = jax.lax.cond(
        jnp.asarray(clock.model_step, dtype=jnp.int64) > jnp.int64(0),
        lambda previous: update_configured_accumulators(
            topology,
            inputs.accumulators,
            previous,
            current_clock,
            accumulator_parameters,
        ).state,
        lambda previous: previous,
        state.accumulators,
    )
    history_state = state._replace(accumulators=accumulators)
    vegetation = end_of_step_vegetation_dynamics(
        history_state.vegetation,
        use_cndv=use_cndv,
    )
    return EndOfTimestepResult(
        state=history_state._replace(vegetation=vegetation),
        history_state=history_state,
        land_to_glacier=glacier,
    )


__all__ = [
    "EndOfTimestepCompositionParameters",
    "EndOfTimestepInputs",
    "EndOfTimestepPhysicalCarry",
    "EndOfTimestepResult",
    "EndOfTimestepState",
    "accumulator_clock",
    "compose_accumulator_inputs",
    "compose_end_of_timestep_inputs",
    "compose_land_to_glacier_inputs",
    "end_of_timestep_physical_carry",
    "end_of_timestep_step",
]
