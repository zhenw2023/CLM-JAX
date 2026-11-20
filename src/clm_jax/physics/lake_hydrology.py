
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.constants import (
    CLM_SPECIAL_VALUE,
    DENSITY_FRESH_WATER,
    DENSITY_ICE,
    FREEZING_TEMPERATURE_FRESHWATER,
    LATENT_HEAT_FUSION,
)
from clm_jax.io.restart import LANDUNIT_LAKE, SubgridTopology
from clm_jax.physics.aerosol import SnowAerosolState, reset_new_snow_aerosols
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
)
from clm_jax.physics.snow import (
    LOW_TEMPERATURE_DENSITY_SLATER_2017,
    NewSnowDensityParameters,
    new_snow_bulk_density,
)
from clm_jax.physics.snow_capping import (
    DEFAULT_SNOW_CAPPING_PARAMETERS,
    SnowCappingInputs,
    SnowCappingParameters,
    SnowCappingState,
    snow_capping_active_case,
)
from clm_jax.physics.snow_compaction import (
    DEFAULT_SNOW_COMPACTION_PARAMETERS,
    SnowCompactionInputs,
    SnowCompactionParameters,
    SnowCompactionState,
    snow_compaction_active_case,
)
from clm_jax.physics.snow_layer_combining import (
    DEFAULT_SNOW_LAYER_COMBINING_PARAMETERS,
    ICE_SPECIFIC_HEAT,
    LIQUID_WATER_SPECIFIC_HEAT,
    NUMBER_OF_SOIL_SNOW_INTERFACES,
    SnowLayerCombiningParameters,
    SnowLayerCombiningState,
    combine_snow_layers_active_case,
)
from clm_jax.physics.snow_layer_division import (
    DEFAULT_SNOW_LAYER_DIVISION_PARAMETERS,
    SnowLayerDivisionParameters,
    SnowLayerDivisionState,
    divide_snow_layers_active_case,
)
from clm_jax.physics.snow_water import (
    GROUND_SURFACE_INDEX,
    NUMBER_OF_GROUND_LEVELS,
    NUMBER_OF_SNOW_LEVELS,
    NUMBER_OF_SOIL_SNOW_LEVELS,
    SnowAerosolDepositionState,
    SnowWaterAerosolState,
    SnowWaterInputs,
    SnowWaterState,
    build_snow_column_masks,
    snow_water,
)

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase


NUMBER_OF_HYDROLOGICALLY_ACTIVE_SOIL_LEVELS = 20
LAKE_SNOW_LAYER_OFFSET = 0.03
UNRESOLVED_LAKE_SNOW_BULK_DENSITY = 250.0
FRESH_SNOW_GRAIN_RADIUS = 54.526


class LakeHydrologyParameters(NamedTuple):

    time_step_seconds: float = 1800.0
    number_of_hydrologically_active_soil_levels: int = NUMBER_OF_HYDROLOGICALLY_ACTIVE_SOIL_LEVELS
    lake_snow_layer_offset: float = LAKE_SNOW_LAYER_OFFSET
    unresolved_snow_bulk_density: float = UNRESOLVED_LAKE_SNOW_BULK_DENSITY
    fresh_snow_grain_radius: float = FRESH_SNOW_GRAIN_RADIUS
    new_snow_density: NewSnowDensityParameters = NewSnowDensityParameters(
        wind_dependent=True,
        low_temperature_method=LOW_TEMPERATURE_DENSITY_SLATER_2017,
    )
    snow_capping: SnowCappingParameters = DEFAULT_SNOW_CAPPING_PARAMETERS
    snow_compaction: SnowCompactionParameters = DEFAULT_SNOW_COMPACTION_PARAMETERS
    snow_layer_combining: SnowLayerCombiningParameters = DEFAULT_SNOW_LAYER_COMBINING_PARAMETERS
    snow_layer_division: SnowLayerDivisionParameters = DEFAULT_SNOW_LAYER_DIVISION_PARAMETERS

    @classmethod
    def from_case(cls, case: CimeCase) -> LakeHydrologyParameters:

        from clm_jax.io.namelist import read_case_namelist_group  

        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        step = float(general["dtime"])
        return cls(
            time_step_seconds=step,
            new_snow_density=NewSnowDensityParameters.from_case(case),
            snow_capping=SnowCappingParameters.from_case(case),
            snow_compaction=SnowCompactionParameters.from_case(case),
            snow_layer_combining=SnowLayerCombiningParameters.from_case(case),
        )


DEFAULT_LAKE_HYDROLOGY_PARAMETERS = LakeHydrologyParameters()


class LakeHydrologyInputs(NamedTuple):
    atmospheric_temperature_column: jnp.ndarray
    rainfall_flux_column: jnp.ndarray
    snowfall_flux_column: jnp.ndarray
    wind_speed_gridcell: jnp.ndarray
    flood_flux_gridcell: jnp.ndarray
    atmospheric_aerosol_flux_gridcell_species: jnp.ndarray
    soil_saturation_column_level: jnp.ndarray
    lake_layer_thickness_column_level: jnp.ndarray
    ground_temperature_column: jnp.ndarray
    snow_cover_shape_column: jnp.ndarray
    soil_snow_melt_flag_column_level: jnp.ndarray
    previous_snow_water_column_level: jnp.ndarray
    begin_water_balance_column: jnp.ndarray
    soil_evaporation_flux_patch: jnp.ndarray
    total_evaporation_flux_patch: jnp.ndarray


class LakeHydrologyGeometryState(NamedTuple):
    snow_layer_count_column: jnp.ndarray
    soil_snow_thickness_column_level: jnp.ndarray
    soil_snow_node_depth_column_level: jnp.ndarray
    soil_snow_interface_depth_column_interface: jnp.ndarray


class LakeHydrologyTemperatureState(NamedTuple):
    lake_temperature_column_level: jnp.ndarray
    soil_snow_temperature_column_level: jnp.ndarray
    top_temperature_gradient_column: jnp.ndarray
    top_snow_temperature_column: jnp.ndarray
    snow_temperature_mass_sum_column: jnp.ndarray


class LakeHydrologyWaterState(NamedTuple):
    effective_snow_fraction_column: jnp.ndarray
    snow_fraction_column: jnp.ndarray
    old_ice_fraction_column_level: jnp.ndarray
    snow_depth_column: jnp.ndarray
    snow_water_equivalent_column: jnp.ndarray
    integrated_snow_column: jnp.ndarray
    snow_ice_column: jnp.ndarray
    snow_liquid_column: jnp.ndarray
    soil_snow_ice_column_level: jnp.ndarray
    soil_snow_liquid_column_level: jnp.ndarray
    soil_water_volume_fraction_column_level: jnp.ndarray
    snow_grain_radius_column_level: jnp.ndarray
    top_snow_grain_radius_column: jnp.ndarray
    top_snow_water_column: jnp.ndarray
    top_snow_liquid_fraction_column: jnp.ndarray
    end_water_balance_column: jnp.ndarray


class LakeHydrologyWaterFluxState(NamedTuple):
    ground_precipitation_flux_patch: jnp.ndarray
    ground_snowfall_flux_patch: jnp.ndarray
    ground_rainfall_flux_patch: jnp.ndarray
    ground_evaporation_flux_patch: jnp.ndarray
    snow_sublimation_flux_patch: jnp.ndarray
    snow_dew_flux_patch: jnp.ndarray
    ground_dew_flux_patch: jnp.ndarray
    ground_rainfall_flux_column: jnp.ndarray
    ground_snowfall_flux_column: jnp.ndarray
    ground_precipitation_flux_column: jnp.ndarray
    ground_evaporation_flux_column: jnp.ndarray
    ground_dew_flux_column: jnp.ndarray
    snow_dew_flux_column: jnp.ndarray
    snow_sublimation_flux_column: jnp.ndarray
    total_evaporation_flux_column: jnp.ndarray
    snowmelt_flux_column: jnp.ndarray
    snowmelt_flux_column_level: jnp.ndarray
    snow_capping_ice_flux_column: jnp.ndarray
    snow_capping_liquid_flux_column: jnp.ndarray
    snow_capping_discarded_ice_flux_column: jnp.ndarray
    snow_capping_discarded_liquid_flux_column: jnp.ndarray
    perched_drainage_flux_column: jnp.ndarray
    surface_water_runoff_flux_column: jnp.ndarray
    snow_drainage_flux_column: jnp.ndarray
    saturation_excess_flux_column: jnp.ndarray
    surface_runoff_flux_column: jnp.ndarray
    drainage_flux_column: jnp.ndarray
    infiltration_flux_column: jnp.ndarray
    water_balance_runoff_flux_column: jnp.ndarray
    total_runoff_flux_column: jnp.ndarray
    ice_runoff_snow_capping_flux_column: jnp.ndarray
    top_soil_water_flux_column: jnp.ndarray
    snow_layer_to_top_soil_flux_column: jnp.ndarray
    flood_flux_column: jnp.ndarray


class LakeHydrologyEnergyFluxState(NamedTuple):
    snowmelt_energy_flux_column: jnp.ndarray
    sensible_heat_total_patch: jnp.ndarray
    sensible_heat_ground_patch: jnp.ndarray
    soil_heat_flux_ground_patch: jnp.ndarray
    net_ground_heat_flux_patch: jnp.ndarray
    ground_lake_heat_flux_patch: jnp.ndarray


class LakeHydrologyAerosolState(NamedTuple):
    mass: SnowAerosolState
    deposition: SnowAerosolDepositionState


class LakeHydrologyState(NamedTuple):
    geometry: LakeHydrologyGeometryState
    temperature: LakeHydrologyTemperatureState
    water: LakeHydrologyWaterState
    water_flux: LakeHydrologyWaterFluxState
    energy_flux: LakeHydrologyEnergyFluxState
    aerosol: LakeHydrologyAerosolState
    lake_ice_fraction_column_level: jnp.ndarray


class LakeHydrologyDiagnostics(NamedTuple):
    lake_column: jnp.ndarray
    lake_patch: jnp.ndarray
    initial_snow_column: jnp.ndarray
    initial_no_snow_column: jnp.ndarray
    final_snow_column: jnp.ndarray
    final_no_snow_column: jnp.ndarray
    new_snow_layer_column: jnp.ndarray
    unfrozen_lake_snow_column: jnp.ndarray
    snow_removed_into_lake_column: jnp.ndarray
    lake_snow_ice_mass_column: jnp.ndarray
    lake_snow_heat_requirement_column: jnp.ndarray


class LakeHydrologyResult(NamedTuple):
    state: LakeHydrologyState
    diagnostics: LakeHydrologyDiagnostics


def _to_snow_water_aerosol(
    aerosol: LakeHydrologyAerosolState,
) -> SnowWaterAerosolState:
    mass = aerosol.mass
    return SnowWaterAerosolState(
        black_carbon_hydrophilic_layer=mass.hydrophilic_black_carbon_layer,
        black_carbon_hydrophobic_layer=mass.hydrophobic_black_carbon_layer,
        organic_carbon_hydrophilic_layer=mass.hydrophilic_organic_carbon_layer,
        organic_carbon_hydrophobic_layer=mass.hydrophobic_organic_carbon_layer,
        dust_species_1_layer=mass.dust_species_1_layer,
        dust_species_2_layer=mass.dust_species_2_layer,
        dust_species_3_layer=mass.dust_species_3_layer,
        dust_species_4_layer=mass.dust_species_4_layer,
        deposition=aerosol.deposition,
    )


def _replace_primary_aerosol(
    aerosol: LakeHydrologyAerosolState,
    primary: SnowWaterAerosolState,
) -> LakeHydrologyAerosolState:
    mass = aerosol.mass
    return LakeHydrologyAerosolState(
        mass=SnowAerosolState(
            hydrophobic_black_carbon_layer=(primary.black_carbon_hydrophobic_layer),
            hydrophilic_black_carbon_layer=(primary.black_carbon_hydrophilic_layer),
            total_black_carbon_layer=mass.total_black_carbon_layer,
            black_carbon_column=mass.black_carbon_column,
            black_carbon_top=mass.black_carbon_top,
            hydrophobic_organic_carbon_layer=(primary.organic_carbon_hydrophobic_layer),
            hydrophilic_organic_carbon_layer=(primary.organic_carbon_hydrophilic_layer),
            total_organic_carbon_layer=mass.total_organic_carbon_layer,
            organic_carbon_column=mass.organic_carbon_column,
            organic_carbon_top=mass.organic_carbon_top,
            dust_species_1_layer=primary.dust_species_1_layer,
            dust_species_2_layer=primary.dust_species_2_layer,
            dust_species_3_layer=primary.dust_species_3_layer,
            dust_species_4_layer=primary.dust_species_4_layer,
            total_dust_layer=mass.total_dust_layer,
            dust_column=mass.dust_column,
            dust_top=mass.dust_top,
        ),
        deposition=primary.deposition,
    )


def _copy_patch_to_column(
    column_values: jnp.ndarray,
    patch_values: jnp.ndarray,
    patch_column: jnp.ndarray,
    patch_mask: jnp.ndarray,
) -> jnp.ndarray:

    if patch_values.shape[0] == 0:
        return column_values
    patch_index = jnp.arange(patch_values.shape[0], dtype=jnp.int32)
    last_patch = jnp.full(column_values.shape[0], -1, dtype=jnp.int32)
    last_patch = last_patch.at[patch_column].max(jnp.where(patch_mask, patch_index, -1))
    selected = patch_values[jnp.maximum(last_patch, 0)]
    mask = (last_patch >= 0).reshape(
        (column_values.shape[0],) + (1,) * (column_values.ndim - 1)
    )
    return jnp.where(mask, selected, column_values)


def _gpu_patch_overwrites() -> bool:

    return jax.default_backend() == "gpu"


def _validate_shapes(
    topology: SubgridTopology,
    inputs: LakeHydrologyInputs,
    state: LakeHydrologyState,
    parameters: LakeHydrologyParameters,
) -> None:
    column_shape = topology.column.gridcell.shape
    patch_shape = topology.patch.column.shape
    grid_shape = topology.grid.longitude_degrees.shape
    if len(column_shape) != 1 or len(patch_shape) != 1 or len(grid_shape) != 1:
        raise ValueError("grid, column and patch topology must be vectors")
    number_of_columns = column_shape[0]
    level_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS)
    ground_shape = (number_of_columns, NUMBER_OF_GROUND_LEVELS)
    snow_shape = (number_of_columns, NUMBER_OF_SNOW_LEVELS)
    interface_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_INTERFACES)

    column_inputs = (
        "atmospheric_temperature_column",
        "rainfall_flux_column",
        "snowfall_flux_column",
        "ground_temperature_column",
        "snow_cover_shape_column",
        "begin_water_balance_column",
    )
    for name in column_inputs:
        if getattr(inputs, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")
    for name in ("wind_speed_gridcell", "flood_flux_gridcell"):
        if getattr(inputs, name).shape != grid_shape:
            raise ValueError(f"{name} must have shape {grid_shape}")
    for name in ("soil_evaporation_flux_patch", "total_evaporation_flux_patch"):
        if getattr(inputs, name).shape != patch_shape:
            raise ValueError(f"{name} must have shape {patch_shape}")
    if inputs.atmospheric_aerosol_flux_gridcell_species.shape != (
        grid_shape[0],
        14,
    ):
        raise ValueError("atmospheric aerosol forcing must have shape (gridcell, 14)")
    if inputs.soil_saturation_column_level.shape != ground_shape:
        raise ValueError(f"soil_saturation_column_level must have shape {ground_shape}")
    if inputs.soil_snow_melt_flag_column_level.shape != level_shape:
        raise ValueError(f"soil_snow_melt_flag_column_level must have shape {level_shape}")
    if inputs.previous_snow_water_column_level.shape != snow_shape:
        raise ValueError(f"previous_snow_water_column_level must have shape {snow_shape}")
    if (
        inputs.lake_layer_thickness_column_level.ndim != 2
        or inputs.lake_layer_thickness_column_level.shape[0] != number_of_columns
        or inputs.lake_layer_thickness_column_level.shape[1] < 1
    ):
        raise ValueError("lake layer thickness must have shape (column, lake_level>=1)")

    if state.geometry.snow_layer_count_column.shape != column_shape:
        raise ValueError(f"snow_layer_count_column must have shape {column_shape}")
    for name in (
        "soil_snow_thickness_column_level",
        "soil_snow_node_depth_column_level",
    ):
        if getattr(state.geometry, name).shape != level_shape:
            raise ValueError(f"geometry.{name} must have shape {level_shape}")
    if state.geometry.soil_snow_interface_depth_column_interface.shape != interface_shape:
        raise ValueError("soil/snow interface depth has the wrong shape")
    if state.temperature.lake_temperature_column_level.shape != (
        inputs.lake_layer_thickness_column_level.shape
    ):
        raise ValueError("lake temperature and lake thickness shapes must match")
    if state.temperature.soil_snow_temperature_column_level.shape != level_shape:
        raise ValueError("soil/snow temperature has the wrong shape")
    for name in state.temperature._fields[2:]:
        if getattr(state.temperature, name).shape != column_shape:
            raise ValueError(f"temperature.{name} must have shape {column_shape}")

    water_column_fields = (
        "effective_snow_fraction_column",
        "snow_fraction_column",
        "snow_depth_column",
        "snow_water_equivalent_column",
        "integrated_snow_column",
        "snow_ice_column",
        "snow_liquid_column",
        "top_snow_grain_radius_column",
        "top_snow_water_column",
        "top_snow_liquid_fraction_column",
        "end_water_balance_column",
    )
    for name in water_column_fields:
        if getattr(state.water, name).shape != column_shape:
            raise ValueError(f"water.{name} must have shape {column_shape}")
    for name in (
        "old_ice_fraction_column_level",
        "soil_snow_ice_column_level",
        "soil_snow_liquid_column_level",
    ):
        if getattr(state.water, name).shape != level_shape:
            raise ValueError(f"water.{name} must have shape {level_shape}")
    if state.water.soil_water_volume_fraction_column_level.shape != ground_shape:
        raise ValueError(
            f"water.soil_water_volume_fraction_column_level must have shape {ground_shape}"
        )
    if state.water.snow_grain_radius_column_level.shape != snow_shape:
        raise ValueError("snow grain radius has the wrong shape")

    patch_flux_fields = state.water_flux._fields[:7]
    for name in patch_flux_fields:
        if getattr(state.water_flux, name).shape != patch_shape:
            raise ValueError(f"water_flux.{name} must have shape {patch_shape}")
    for name in state.water_flux._fields[7:]:
        expected = snow_shape if name == "snowmelt_flux_column_level" else column_shape
        if getattr(state.water_flux, name).shape != expected:
            raise ValueError(f"water_flux.{name} must have shape {expected}")
    if state.energy_flux.snowmelt_energy_flux_column.shape != column_shape:
        raise ValueError("column snowmelt energy flux has the wrong shape")
    for name in state.energy_flux._fields[1:]:
        if getattr(state.energy_flux, name).shape != patch_shape:
            raise ValueError(f"energy_flux.{name} must have shape {patch_shape}")
    if state.lake_ice_fraction_column_level.shape != (
        inputs.lake_layer_thickness_column_level.shape
    ):
        raise ValueError("lake ice fraction and lake thickness shapes must match")

    for name in SnowAerosolState._fields:
        value = getattr(state.aerosol.mass, name)
        expected = snow_shape if value.ndim == 2 else column_shape
        if value.shape != expected:
            raise ValueError(f"aerosol.mass.{name} must have shape {expected}")
    for name in SnowAerosolDepositionState._fields:
        if getattr(state.aerosol.deposition, name).shape != column_shape:
            raise ValueError(f"aerosol.deposition.{name} must have shape {column_shape}")

    if parameters.time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")
    if not 1 <= parameters.number_of_hydrologically_active_soil_levels <= NUMBER_OF_GROUND_LEVELS:
        raise ValueError("hydrologically active soil level count is out of range")
    if parameters.lake_snow_layer_offset < 0.0:
        raise ValueError("lake snow layer offset must be nonnegative")
    if parameters.unresolved_snow_bulk_density <= 0.0:
        raise ValueError("unresolved snow bulk density must be positive")


def lake_hydrology(
    topology: SubgridTopology,
    inputs: LakeHydrologyInputs,
    state: LakeHydrologyState,
    *,
    parameters: LakeHydrologyParameters = DEFAULT_LAKE_HYDROLOGY_PARAMETERS,
) -> LakeHydrologyResult:

    _validate_shapes(topology, inputs, state, parameters)
    number_of_columns = topology.column.gridcell.shape[0]
    column_index = jnp.arange(number_of_columns, dtype=jnp.int32)
    patch_column = jnp.asarray(topology.patch.column, dtype=jnp.int32)
    patch_gridcell = jnp.asarray(topology.patch.gridcell, dtype=jnp.int32)
    column_gridcell = jnp.asarray(topology.column.gridcell, dtype=jnp.int32)
    lake_column = jnp.asarray(topology.column.active, dtype=jnp.bool_) & (
        jnp.asarray(topology.column.landunit_type, dtype=jnp.int32) == jnp.int32(LANDUNIT_LAKE)
    )
    lake_patch = jnp.asarray(topology.patch.active, dtype=jnp.bool_) & (
        jnp.asarray(topology.patch.landunit_type, dtype=jnp.int32) == jnp.int32(LANDUNIT_LAKE)
    )
    time_step = jnp.float64(parameters.time_step_seconds)
    freezing = jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)

    geometry = state.geometry
    temperature_state = state.temperature
    water = state.water
    flux = state.water_flux
    energy = state.energy_flux
    aerosol = state.aerosol

    layer_count = jnp.asarray(
        geometry.snow_layer_count_column,
        dtype=jnp.int32,
    )
    thickness = jnp.asarray(
        geometry.soil_snow_thickness_column_level,
        dtype=jnp.float64,
    )
    node_depth = jnp.asarray(
        geometry.soil_snow_node_depth_column_level,
        dtype=jnp.float64,
    )
    interface_depth = jnp.asarray(
        geometry.soil_snow_interface_depth_column_interface,
        dtype=jnp.float64,
    )
    soil_snow_temperature = jnp.asarray(
        temperature_state.soil_snow_temperature_column_level,
        dtype=jnp.float64,
    )
    lake_temperature = jnp.asarray(
        temperature_state.lake_temperature_column_level,
        dtype=jnp.float64,
    )
    lake_ice_fraction = jnp.asarray(
        state.lake_ice_fraction_column_level,
        dtype=jnp.float64,
    )
    snow_depth = jnp.asarray(water.snow_depth_column, dtype=jnp.float64)
    snow_water_equivalent = jnp.asarray(
        water.snow_water_equivalent_column,
        dtype=jnp.float64,
    )
    old_ice_fraction = jnp.asarray(
        water.old_ice_fraction_column_level,
        dtype=jnp.float64,
    )
    soil_snow_ice = jnp.asarray(
        water.soil_snow_ice_column_level,
        dtype=jnp.float64,
    )
    soil_snow_liquid = jnp.asarray(
        water.soil_snow_liquid_column_level,
        dtype=jnp.float64,
    )
    snow_grain_radius = jnp.asarray(
        water.snow_grain_radius_column_level,
        dtype=jnp.float64,
    )

    patch_snowfall = jnp.where(
        lake_patch,
        jnp.asarray(inputs.snowfall_flux_column, dtype=jnp.float64)[patch_column],
        jnp.asarray(flux.ground_snowfall_flux_patch, dtype=jnp.float64),
    )
    patch_rainfall = jnp.where(
        lake_patch,
        jnp.asarray(inputs.rainfall_flux_column, dtype=jnp.float64)[patch_column],
        jnp.asarray(flux.ground_rainfall_flux_patch, dtype=jnp.float64),
    )
    patch_precipitation = jnp.where(
        lake_patch,
        fortran_add(patch_snowfall, patch_rainfall),
        jnp.asarray(flux.ground_precipitation_flux_patch, dtype=jnp.float64),
    )
    column_snowfall = _copy_patch_to_column(
        jnp.asarray(flux.ground_snowfall_flux_column, dtype=jnp.float64),
        patch_snowfall,
        patch_column,
        lake_patch,
    )
    column_rainfall = _copy_patch_to_column(
        jnp.asarray(flux.ground_rainfall_flux_column, dtype=jnp.float64),
        patch_rainfall,
        patch_column,
        lake_patch,
    )

    column_wind = jnp.asarray(inputs.wind_speed_gridcell, dtype=jnp.float64)[column_gridcell]
    snowfall_density = new_snow_bulk_density(
        inputs.atmospheric_temperature_column,
        column_wind,
        parameters.new_snow_density,
    )
    snow_depth_rate = fortran_divide(column_snowfall, snowfall_density)
    updated_snow_depth = fortran_add(
        snow_depth,
        fortran_multiply(snow_depth_rate, time_step),
    )
    updated_snow_water = fortran_add(
        snow_water_equivalent,
        fortran_multiply(column_snowfall, time_step),
    )
    snow_depth = jnp.where(lake_column, updated_snow_depth, snow_depth)
    snow_water_equivalent = jnp.where(
        lake_column,
        updated_snow_water,
        snow_water_equivalent,
    )

    new_layer_depth_threshold = fortran_add(
        jnp.float64(0.01),
        jnp.float64(parameters.lake_snow_layer_offset),
    )
    new_snow_layer = (
        lake_column
        & (layer_count == jnp.int32(0))
        & (column_snowfall > jnp.float64(0.0))
        & (snow_depth >= new_layer_depth_threshold)
    )
    bottom_snow_index = NUMBER_OF_SNOW_LEVELS - 1
    layer_count = jnp.where(new_snow_layer, jnp.int32(-1), layer_count)
    thickness = thickness.at[:, bottom_snow_index].set(
        jnp.where(
            new_snow_layer,
            snow_depth,
            thickness[:, bottom_snow_index],
        )
    )
    node_depth = node_depth.at[:, bottom_snow_index].set(
        jnp.where(
            new_snow_layer,
            fortran_multiply(jnp.float64(-0.5), snow_depth),
            node_depth[:, bottom_snow_index],
        )
    )
    interface_depth = interface_depth.at[:, bottom_snow_index].set(
        jnp.where(
            new_snow_layer,
            -snow_depth,
            interface_depth[:, bottom_snow_index],
        )
    )
    fresh_temperature = jnp.minimum(
        freezing,
        jnp.asarray(inputs.atmospheric_temperature_column, dtype=jnp.float64),
    )
    soil_snow_temperature = soil_snow_temperature.at[:, bottom_snow_index].set(
        jnp.where(
            new_snow_layer,
            fresh_temperature,
            soil_snow_temperature[:, bottom_snow_index],
        )
    )
    soil_snow_ice = soil_snow_ice.at[:, bottom_snow_index].set(
        jnp.where(
            new_snow_layer,
            snow_water_equivalent,
            soil_snow_ice[:, bottom_snow_index],
        )
    )
    soil_snow_liquid = soil_snow_liquid.at[:, bottom_snow_index].set(
        jnp.where(
            new_snow_layer,
            jnp.float64(0.0),
            soil_snow_liquid[:, bottom_snow_index],
        )
    )
    old_ice_fraction = old_ice_fraction.at[:, bottom_snow_index].set(
        jnp.where(
            new_snow_layer,
            jnp.float64(1.0),
            old_ice_fraction[:, bottom_snow_index],
        )
    )
    snow_grain_radius = snow_grain_radius.at[:, bottom_snow_index].set(
        jnp.where(
            new_snow_layer,
            jnp.float64(parameters.fresh_snow_grain_radius),
            snow_grain_radius[:, bottom_snow_index],
        )
    )
    aerosol = LakeHydrologyAerosolState(
        mass=reset_new_snow_aerosols(aerosol.mass, new_snow_layer),
        deposition=aerosol.deposition,
    )

    top_snow_index = jnp.clip(
        layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS),
        jnp.int32(0),
        jnp.int32(NUMBER_OF_SNOW_LEVELS - 1),
    )
    add_to_existing_layer = lake_column & (layer_count < jnp.int32(0)) & ~new_snow_layer
    existing_top_ice = soil_snow_ice[column_index, top_snow_index]
    soil_snow_ice = soil_snow_ice.at[column_index, top_snow_index].set(
        jnp.where(
            add_to_existing_layer,
            fortran_add(
                existing_top_ice,
                fortran_multiply(time_step, column_snowfall),
            ),
            existing_top_ice,
        )
    )
    existing_top_thickness = thickness[column_index, top_snow_index]
    thickness = thickness.at[column_index, top_snow_index].set(
        jnp.where(
            add_to_existing_layer,
            fortran_add(
                existing_top_thickness,
                fortran_multiply(snow_depth_rate, time_step),
            ),
            existing_top_thickness,
        )
    )

    patch_top_index = jnp.clip(
        layer_count[patch_column] + jnp.int32(NUMBER_OF_SNOW_LEVELS),
        jnp.int32(0),
        jnp.int32(NUMBER_OF_SOIL_SNOW_LEVELS - 1),
    )
    patch_top_liquid = soil_snow_liquid[patch_column, patch_top_index]
    patch_top_ice = soil_snow_ice[patch_column, patch_top_index]
    patch_top_mass = fortran_add(patch_top_liquid, patch_top_ice)
    safe_patch_top_mass = jnp.where(
        patch_top_mass > jnp.float64(0.0),
        patch_top_mass,
        jnp.float64(1.0),
    )
    patch_soil_evaporation = jnp.asarray(
        inputs.soil_evaporation_flux_patch,
        dtype=jnp.float64,
    )
    resolved_snow_patch = lake_patch & (layer_count[patch_column] < jnp.int32(0))
    no_resolved_snow_patch = lake_patch & ~resolved_snow_patch
    positive_evaporation = patch_soil_evaporation >= jnp.float64(0.0)
    limited_evaporation = jnp.minimum(
        patch_soil_evaporation,
        fortran_divide(patch_top_mass, time_step),
    )
    resolved_ground_evaporation = jnp.where(
        patch_top_mass > jnp.float64(0.0),
        jnp.maximum(
            fortran_multiply(
                limited_evaporation,
                fortran_divide(patch_top_liquid, safe_patch_top_mass),
            ),
            jnp.float64(0.0),
        ),
        jnp.float64(0.0),
    )
    patch_ground_evaporation = jnp.where(
        resolved_snow_patch & positive_evaporation,
        resolved_ground_evaporation,
        jnp.float64(0.0),
    )
    patch_sublimation = jnp.where(
        resolved_snow_patch & positive_evaporation,
        fortran_add(limited_evaporation, -resolved_ground_evaporation),
        jnp.float64(0.0),
    )

    patch_ground_temperature = jnp.asarray(
        inputs.ground_temperature_column,
        dtype=jnp.float64,
    )[patch_column]
    patch_top_temperature = soil_snow_temperature[patch_column, patch_top_index]
    resolved_frost = (
        resolved_snow_patch
        & ~positive_evaporation
        & (patch_ground_temperature < freezing)
        & (patch_top_temperature < freezing)
    )
    resolved_ground_dew = (
        resolved_snow_patch
        & ~positive_evaporation
        & ~resolved_frost
        & (
            (layer_count[patch_column] + jnp.int32(1) < jnp.int32(0))
            | ((patch_ground_temperature == freezing) & (patch_top_temperature == freezing))
        )
    )
    patch_snow_dew = jnp.where(
        resolved_frost,
        jnp.abs(patch_soil_evaporation),
        jnp.float64(0.0),
    )
    patch_ground_dew = jnp.where(
        resolved_ground_dew,
        jnp.abs(patch_soil_evaporation),
        jnp.float64(0.0),
    )

    unresolved_sublimation = jnp.minimum(
        patch_soil_evaporation,
        fortran_divide(snow_water_equivalent[patch_column], time_step),
    )
    patch_sublimation = jnp.where(
        no_resolved_snow_patch & positive_evaporation,
        unresolved_sublimation,
        patch_sublimation,
    )
    patch_ground_evaporation = jnp.where(
        no_resolved_snow_patch & positive_evaporation,
        fortran_add(patch_soil_evaporation, -unresolved_sublimation),
        patch_ground_evaporation,
    )
    unresolved_frost = (
        no_resolved_snow_patch
        & ~positive_evaporation
        & (patch_ground_temperature < fortran_add(freezing, jnp.float64(-0.1)))
    )
    patch_snow_dew = jnp.where(
        unresolved_frost,
        jnp.abs(patch_soil_evaporation),
        patch_snow_dew,
    )
    patch_ground_dew = jnp.where(
        no_resolved_snow_patch & ~positive_evaporation & ~unresolved_frost,
        jnp.abs(patch_soil_evaporation),
        patch_ground_dew,
    )

    unresolved_old_snow_water = snow_water_equivalent[patch_column]
    unresolved_snow_water = fortran_add(
        unresolved_old_snow_water,
        fortran_multiply(
            fortran_add(-patch_sublimation, patch_snow_dew),
            time_step,
        ),
    )
    safe_unresolved_old_snow_water = jnp.where(
        unresolved_old_snow_water > jnp.float64(0.0),
        unresolved_old_snow_water,
        jnp.float64(1.0),
    )
    unresolved_snow_depth = jnp.where(
        unresolved_old_snow_water > jnp.float64(0.0),
        fortran_divide(
            fortran_multiply(
                snow_depth[patch_column],
                unresolved_snow_water,
            ),
            safe_unresolved_old_snow_water,
        ),
        fortran_divide(
            unresolved_snow_water,
            jnp.float64(parameters.unresolved_snow_bulk_density),
        ),
    )

    def update_unresolved_patch(
        patch_index: int,
        carry: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        output_water, output_depth = carry
        column = patch_column[patch_index]
        update = no_resolved_snow_patch[patch_index]
        output_water = output_water.at[column].set(
            jnp.where(
                update,
                jnp.maximum(unresolved_snow_water[patch_index], jnp.float64(0.0)),
                output_water[column],
            )
        )
        output_depth = output_depth.at[column].set(
            jnp.where(update, unresolved_snow_depth[patch_index], output_depth[column])
        )
        return output_water, output_depth

    if _gpu_patch_overwrites():
        snow_water_equivalent = _copy_patch_to_column(
            snow_water_equivalent,
            jnp.maximum(unresolved_snow_water, jnp.float64(0.0)),
            patch_column,
            no_resolved_snow_patch,
        )
        snow_depth = _copy_patch_to_column(
            snow_depth, unresolved_snow_depth, patch_column, no_resolved_snow_patch,
        )
    else:
        snow_water_equivalent, snow_depth = lax.fori_loop(
            0,
            patch_column.shape[0],
            update_unresolved_patch,
            (snow_water_equivalent, snow_depth),
        )

    patch_ground_evaporation = jnp.where(
        lake_patch,
        patch_ground_evaporation,
        jnp.asarray(flux.ground_evaporation_flux_patch, dtype=jnp.float64),
    )
    patch_sublimation = jnp.where(
        lake_patch,
        patch_sublimation,
        jnp.asarray(flux.snow_sublimation_flux_patch, dtype=jnp.float64),
    )
    patch_snow_dew = jnp.where(
        lake_patch,
        patch_snow_dew,
        jnp.asarray(flux.snow_dew_flux_patch, dtype=jnp.float64),
    )
    patch_ground_dew = jnp.where(
        lake_patch,
        patch_ground_dew,
        jnp.asarray(flux.ground_dew_flux_patch, dtype=jnp.float64),
    )

    column_total_evaporation = _copy_patch_to_column(
        jnp.asarray(flux.total_evaporation_flux_column, dtype=jnp.float64),
        jnp.asarray(inputs.total_evaporation_flux_patch, dtype=jnp.float64),
        patch_column,
        lake_patch,
    )
    column_precipitation = _copy_patch_to_column(
        jnp.asarray(flux.ground_precipitation_flux_column, dtype=jnp.float64),
        patch_precipitation,
        patch_column,
        lake_patch,
    )
    column_ground_evaporation = _copy_patch_to_column(
        jnp.asarray(flux.ground_evaporation_flux_column, dtype=jnp.float64),
        patch_ground_evaporation,
        patch_column,
        lake_patch,
    )
    column_ground_dew = _copy_patch_to_column(
        jnp.asarray(flux.ground_dew_flux_column, dtype=jnp.float64),
        patch_ground_dew,
        patch_column,
        lake_patch,
    )
    column_snow_dew = _copy_patch_to_column(
        jnp.asarray(flux.snow_dew_flux_column, dtype=jnp.float64),
        patch_snow_dew,
        patch_column,
        lake_patch,
    )
    column_sublimation = _copy_patch_to_column(
        jnp.asarray(flux.snow_sublimation_flux_column, dtype=jnp.float64),
        patch_sublimation,
        patch_column,
        lake_patch,
    )

    initial_masks = build_snow_column_masks(
        topology,
        layer_count,
        lake_only=True,
    )
    initial_snow_column = initial_masks.snow_column
    initial_no_snow_column = initial_masks.no_snow_column
    effective_snow_fraction = jnp.where(
        lake_column,
        jnp.where(
            snow_water_equivalent > jnp.float64(0.0),
            jnp.float64(1.0),
            jnp.float64(0.0),
        ),
        jnp.asarray(water.effective_snow_fraction_column, dtype=jnp.float64),
    )

    snow_water_result = snow_water(
        topology,
        SnowWaterInputs(
            time_step_seconds=time_step,
            snow_layer_count_column=layer_count,
            effective_snow_fraction_column=effective_snow_fraction,
            snow_water_equivalent_column=snow_water_equivalent,
            snowmelt_flux_column=jnp.asarray(
                flux.snowmelt_flux_column,
                dtype=jnp.float64,
            ),
            ground_rainfall_flux_column=column_rainfall,
            snow_sublimation_flux_column=column_sublimation,
            snow_dew_flux_column=column_snow_dew,
            ground_evaporation_flux_column=column_ground_evaporation,
            ground_dew_flux_column=column_ground_dew,
            atmospheric_aerosol_flux_gridcell_species=jnp.asarray(
                inputs.atmospheric_aerosol_flux_gridcell_species,
                dtype=jnp.float64,
            ),
        ),
        SnowWaterState(
            integrated_snow_column=jnp.asarray(
                water.integrated_snow_column,
                dtype=jnp.float64,
            ),
            snow_fraction_column=jnp.asarray(
                water.snow_fraction_column,
                dtype=jnp.float64,
            ),
            snow_depth_column=snow_depth,
            soil_snow_liquid_column_level=soil_snow_liquid,
            soil_snow_ice_column_level=soil_snow_ice,
            soil_snow_thickness_column_level=thickness,
            snow_drainage_flux_column=jnp.asarray(
                flux.snow_drainage_flux_column,
                dtype=jnp.float64,
            ),
            top_soil_water_flux_column=jnp.asarray(
                flux.top_soil_water_flux_column,
                dtype=jnp.float64,
            ),
            aerosol=_to_snow_water_aerosol(aerosol),
        ),
        lake_only=True,
    )
    integrated_snow = snow_water_result.state.integrated_snow_column
    snow_fraction = snow_water_result.state.snow_fraction_column
    snow_depth = snow_water_result.state.snow_depth_column
    soil_snow_liquid = snow_water_result.state.soil_snow_liquid_column_level
    soil_snow_ice = snow_water_result.state.soil_snow_ice_column_level
    thickness = snow_water_result.state.soil_snow_thickness_column_level
    snow_drainage_flux = snow_water_result.state.snow_drainage_flux_column
    top_soil_water_flux = snow_water_result.state.top_soil_water_flux_column
    aerosol = _replace_primary_aerosol(
        aerosol,
        snow_water_result.state.aerosol,
    )

    capping_result = snow_capping_active_case(
        topology,
        SnowCappingInputs(
            snow_layer_count_column=layer_count,
            snow_water_equivalent_column=snow_water_equivalent,
        ),
        SnowCappingState(
            soil_snow_liquid_column_level=soil_snow_liquid,
            soil_snow_ice_column_level=soil_snow_ice,
            soil_snow_thickness_column_level=thickness,
            snow_capping_ice_flux_column=jnp.asarray(
                flux.snow_capping_ice_flux_column,
                dtype=jnp.float64,
            ),
            snow_capping_liquid_flux_column=jnp.asarray(
                flux.snow_capping_liquid_flux_column,
                dtype=jnp.float64,
            ),
            snow_capping_discarded_ice_flux_column=jnp.asarray(
                flux.snow_capping_discarded_ice_flux_column,
                dtype=jnp.float64,
            ),
            snow_capping_discarded_liquid_flux_column=jnp.asarray(
                flux.snow_capping_discarded_liquid_flux_column,
                dtype=jnp.float64,
            ),
            aerosol=_to_snow_water_aerosol(aerosol),
        ),
        parameters=parameters.snow_capping,
        lake_only=True,
    )
    soil_snow_liquid = capping_result.state.soil_snow_liquid_column_level
    soil_snow_ice = capping_result.state.soil_snow_ice_column_level
    thickness = capping_result.state.soil_snow_thickness_column_level
    snow_capping_ice_flux = capping_result.state.snow_capping_ice_flux_column
    snow_capping_liquid_flux = capping_result.state.snow_capping_liquid_flux_column
    snow_capping_discarded_ice_flux = capping_result.state.snow_capping_discarded_ice_flux_column
    snow_capping_discarded_liquid_flux = (
        capping_result.state.snow_capping_discarded_liquid_flux_column
    )
    aerosol = _replace_primary_aerosol(aerosol, capping_result.state.aerosol)

    active_soil_count = parameters.number_of_hydrologically_active_soil_levels
    soil_slice = slice(
        GROUND_SURFACE_INDEX,
        GROUND_SURFACE_INDEX + active_soil_count,
    )
    active_thickness = thickness[:, soil_slice]
    active_liquid = soil_snow_liquid[:, soil_slice]
    active_ice = soil_snow_ice[:, soil_slice]
    active_saturation = jnp.asarray(
        inputs.soil_saturation_column_level,
        dtype=jnp.float64,
    )[:, :active_soil_count]
    safe_active_thickness = jnp.where(
        lake_column[:, None],
        active_thickness,
        jnp.float64(1.0),
    )
    liquid_volume = fortran_divide(
        active_liquid,
        fortran_multiply(
            safe_active_thickness,
            jnp.float64(DENSITY_FRESH_WATER),
        ),
    )
    ice_volume = fortran_divide(
        active_ice,
        fortran_multiply(safe_active_thickness, jnp.float64(DENSITY_ICE)),
    )
    active_volume_fraction = fortran_add(liquid_volume, ice_volume)
    saturation_liquid = fortran_multiply(
        fortran_add(
            fortran_multiply(active_saturation, active_thickness),
            -fortran_divide(active_ice, jnp.float64(DENSITY_ICE)),
        ),
        jnp.float64(DENSITY_FRESH_WATER),
    )
    maximum_liquid = fortran_multiply(
        fortran_multiply(
            active_saturation,
            jnp.float64(DENSITY_FRESH_WATER),
        ),
        active_thickness,
    )
    adjusted_liquid = jnp.where(
        active_volume_fraction < active_saturation,
        saturation_liquid,
        jnp.where(active_liquid > maximum_liquid, maximum_liquid, active_liquid),
    )
    soil_snow_liquid = soil_snow_liquid.at[:, soil_slice].set(
        jnp.where(lake_column[:, None], adjusted_liquid, active_liquid)
    )

    compaction_result = snow_compaction_active_case(
        topology,
        SnowCompactionInputs(
            snow_layer_count_column=layer_count,
            snow_cover_shape_column=jnp.asarray(
                inputs.snow_cover_shape_column,
                dtype=jnp.float64,
            ),
            wind_speed_gridcell=jnp.asarray(
                inputs.wind_speed_gridcell,
                dtype=jnp.float64,
            ),
            soil_snow_temperature_column_level=soil_snow_temperature,
            soil_snow_melt_flag_column_level=jnp.asarray(
                inputs.soil_snow_melt_flag_column_level,
                dtype=jnp.int32,
            ),
            effective_snow_fraction_column=effective_snow_fraction,
            previous_snow_water_column_level=jnp.asarray(
                inputs.previous_snow_water_column_level,
                dtype=jnp.float64,
            ),
            integrated_snow_column=integrated_snow,
            old_ice_fraction_column_level=old_ice_fraction,
            soil_snow_ice_column_level=soil_snow_ice,
            soil_snow_liquid_column_level=soil_snow_liquid,
        ),
        SnowCompactionState(
            soil_snow_thickness_column_level=thickness,
        ),
        parameters=parameters.snow_compaction,
        lake_only=True,
    )
    thickness = compaction_result.state.soil_snow_thickness_column_level

    combining_result = combine_snow_layers_active_case(
        topology,
        SnowLayerCombiningState(
            snow_layer_count_column=layer_count,
            snow_fraction_column=snow_fraction,
            effective_snow_fraction_column=effective_snow_fraction,
            snow_depth_column=snow_depth,
            integrated_snow_column=integrated_snow,
            snow_water_equivalent_column=snow_water_equivalent,
            soil_snow_thickness_column_level=thickness,
            soil_snow_node_depth_column_level=node_depth,
            soil_snow_interface_depth_column_interface=interface_depth,
            soil_snow_temperature_column_level=soil_snow_temperature,
            soil_snow_ice_column_level=soil_snow_ice,
            soil_snow_liquid_column_level=soil_snow_liquid,
            snow_grain_radius_column_level=snow_grain_radius,
            top_soil_water_flux_column=jnp.asarray(
                flux.snow_layer_to_top_soil_flux_column,
                dtype=jnp.float64,
            ),
            aerosol=_to_snow_water_aerosol(aerosol),
        ),
        parameters=parameters.snow_layer_combining,
        lake_only=True,
    )
    combined = combining_result.state
    layer_count = combined.snow_layer_count_column
    snow_fraction = combined.snow_fraction_column
    effective_snow_fraction = combined.effective_snow_fraction_column
    snow_depth = combined.snow_depth_column
    integrated_snow = combined.integrated_snow_column
    snow_water_equivalent = combined.snow_water_equivalent_column
    thickness = combined.soil_snow_thickness_column_level
    node_depth = combined.soil_snow_node_depth_column_level
    interface_depth = combined.soil_snow_interface_depth_column_interface
    soil_snow_temperature = combined.soil_snow_temperature_column_level
    soil_snow_ice = combined.soil_snow_ice_column_level
    soil_snow_liquid = combined.soil_snow_liquid_column_level
    snow_grain_radius = combined.snow_grain_radius_column_level
    snow_layer_to_top_soil_flux = combined.top_soil_water_flux_column
    aerosol = _replace_primary_aerosol(aerosol, combined.aerosol)

    division_result = divide_snow_layers_active_case(
        topology,
        SnowLayerDivisionState(
            snow_layer_count_column=layer_count,
            effective_snow_fraction_column=effective_snow_fraction,
            soil_snow_thickness_column_level=thickness,
            soil_snow_node_depth_column_level=node_depth,
            soil_snow_interface_depth_column_interface=interface_depth,
            soil_snow_temperature_column_level=soil_snow_temperature,
            soil_snow_ice_column_level=soil_snow_ice,
            soil_snow_liquid_column_level=soil_snow_liquid,
            snow_grain_radius_column_level=snow_grain_radius,
            aerosol=_to_snow_water_aerosol(aerosol),
        ),
        parameters=parameters.snow_layer_division,
        is_lake=True,
    )
    divided = division_result.state
    layer_count = divided.snow_layer_count_column
    thickness = divided.soil_snow_thickness_column_level
    node_depth = divided.soil_snow_node_depth_column_level
    interface_depth = divided.soil_snow_interface_depth_column_interface
    soil_snow_temperature = divided.soil_snow_temperature_column_level
    soil_snow_ice = divided.soil_snow_ice_column_level
    soil_snow_liquid = divided.soil_snow_liquid_column_level
    snow_grain_radius = divided.snow_grain_radius_column_level
    aerosol = _replace_primary_aerosol(aerosol, divided.aerosol)

    bottom_ice_patch = soil_snow_ice[patch_column, bottom_snow_index]
    bottom_liquid_patch = soil_snow_liquid[patch_column, bottom_snow_index]
    bottom_temperature_patch = soil_snow_temperature[
        patch_column,
        bottom_snow_index,
    ]
    single_layer_patch = lake_patch & (layer_count[patch_column] == jnp.int32(-1))
    warm_ice_patch = (
        single_layer_patch
        & (bottom_ice_patch > jnp.float64(0.0))
        & (bottom_temperature_patch > freezing)
    )
    liquid_only_patch = single_layer_patch & (bottom_ice_patch == jnp.float64(0.0))
    single_layer_heat = fortran_multiply(
        fortran_multiply(
            jnp.float64(LIQUID_WATER_SPECIFIC_HEAT),
            bottom_liquid_patch,
        ),
        fortran_add(bottom_temperature_patch, -freezing),
    )
    single_layer_heat_flux = fortran_divide(single_layer_heat, time_step)
    single_layer_energy_adjustment = warm_ice_patch | liquid_only_patch

    single_layer_snow_water_patch = snow_water_equivalent[patch_column]

    def update_single_layer_patch(
        patch_index: int,
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        output_count, output_water, output_depth, output_temperature = carry
        column = patch_column[patch_index]
        warm = warm_ice_patch[patch_index]
        liquid_only = liquid_only_patch[patch_index]
        output_count = output_count.at[column].set(
            jnp.where(liquid_only, jnp.int32(0), output_count[column])
        )
        output_water = output_water.at[column].set(
            jnp.where(liquid_only, jnp.float64(0.0), output_water[column])
        )
        output_depth = output_depth.at[column].set(
            jnp.where(liquid_only, jnp.float64(0.0), output_depth[column])
        )
        output_temperature = output_temperature.at[column, bottom_snow_index].set(
            jnp.where(
                warm,
                freezing,
                output_temperature[column, bottom_snow_index],
            )
        )
        return output_count, output_water, output_depth, output_temperature

    if _gpu_patch_overwrites():
        zero_patch = jnp.zeros(patch_column.shape, dtype=jnp.float64)
        layer_count = _copy_patch_to_column(
            layer_count, jnp.zeros_like(patch_column), patch_column, liquid_only_patch,
        )
        snow_water_equivalent = _copy_patch_to_column(
            snow_water_equivalent, zero_patch, patch_column, liquid_only_patch,
        )
        snow_depth = _copy_patch_to_column(
            snow_depth, zero_patch, patch_column, liquid_only_patch,
        )
        bottom_temperature = _copy_patch_to_column(
            soil_snow_temperature[:, bottom_snow_index],
            jnp.full(patch_column.shape, freezing),
            patch_column,
            warm_ice_patch,
        )
        soil_snow_temperature = soil_snow_temperature.at[:, bottom_snow_index].set(
            bottom_temperature,
        )
    else:
        layer_count, snow_water_equivalent, snow_depth, soil_snow_temperature = lax.fori_loop(
            0,
            patch_column.shape[0],
            update_single_layer_patch,
            (
                layer_count,
                snow_water_equivalent,
                snow_depth,
                soil_snow_temperature,
            ),
        )
    single_layer_drain_patch = fortran_add(
        snow_drainage_flux[patch_column],
        fortran_divide(single_layer_snow_water_patch, time_step),
    )
    snow_drainage_flux = _copy_patch_to_column(
        snow_drainage_flux,
        single_layer_drain_patch,
        patch_column,
        liquid_only_patch,
    )

    sensible_heat_total_patch = jnp.where(
        single_layer_energy_adjustment,
        fortran_add(
            jnp.asarray(energy.sensible_heat_total_patch, dtype=jnp.float64),
            single_layer_heat_flux,
        ),
        jnp.asarray(energy.sensible_heat_total_patch, dtype=jnp.float64),
    )
    sensible_heat_ground_patch = jnp.where(
        single_layer_energy_adjustment,
        fortran_add(
            jnp.asarray(energy.sensible_heat_ground_patch, dtype=jnp.float64),
            single_layer_heat_flux,
        ),
        jnp.asarray(energy.sensible_heat_ground_patch, dtype=jnp.float64),
    )
    soil_heat_flux_ground_patch = jnp.where(
        single_layer_energy_adjustment,
        fortran_add(
            jnp.asarray(energy.soil_heat_flux_ground_patch, dtype=jnp.float64),
            -single_layer_heat_flux,
        ),
        jnp.asarray(energy.soil_heat_flux_ground_patch, dtype=jnp.float64),
    )
    net_ground_heat_flux_patch = jnp.where(
        single_layer_energy_adjustment,
        fortran_add(
            jnp.asarray(energy.net_ground_heat_flux_patch, dtype=jnp.float64),
            -single_layer_heat_flux,
        ),
        jnp.asarray(energy.net_ground_heat_flux_patch, dtype=jnp.float64),
    )
    ground_lake_heat_flux_patch = jnp.where(
        lake_patch,
        jnp.where(
            single_layer_energy_adjustment,
            fortran_add(
                jnp.asarray(energy.ground_lake_heat_flux_patch, dtype=jnp.float64),
                -single_layer_heat_flux,
            ),
            net_ground_heat_flux_patch,
        ),
        jnp.asarray(energy.ground_lake_heat_flux_patch, dtype=jnp.float64),
    )

    unfrozen_lake_snow = (
        lake_column
        & (lake_temperature[:, 0] > freezing)
        & (lake_ice_fraction[:, 0] == jnp.float64(0.0))
        & (layer_count < jnp.int32(0))
    )
    snow_level_number = jnp.arange(
        -NUMBER_OF_SNOW_LEVELS + 1,
        1,
        dtype=jnp.int32,
    )
    zero_column = jnp.zeros(number_of_columns, dtype=jnp.float64)

    def accumulate_lake_snow_heat(
        level: int,
        carry: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        ice_total, heat_total = carry
        active = unfrozen_lake_snow & (snow_level_number[level] >= layer_count + jnp.int32(1))
        layer_ice = soil_snow_ice[:, level]
        layer_liquid = soil_snow_liquid[:, level]
        temperature_deficit = fortran_add(
            freezing,
            -soil_snow_temperature[:, level],
        )
        ice_sensible_heat = fortran_multiply(
            fortran_multiply(layer_ice, jnp.float64(ICE_SPECIFIC_HEAT)),
            temperature_deficit,
        )
        liquid_sensible_heat = fortran_multiply(
            fortran_multiply(
                layer_liquid,
                jnp.float64(LIQUID_WATER_SPECIFIC_HEAT),
            ),
            temperature_deficit,
        )
        return (
            jnp.where(active, fortran_add(ice_total, layer_ice), ice_total),
            jnp.where(
                active,
                fortran_add(
                    fortran_add(heat_total, ice_sensible_heat),
                    liquid_sensible_heat,
                ),
                heat_total,
            ),
        )

    lake_snow_ice_mass, lake_snow_heat = lax.fori_loop(
        0,
        NUMBER_OF_SNOW_LEVELS,
        accumulate_lake_snow_heat,
        (zero_column, zero_column),
    )
    lake_snow_heat = jnp.where(
        unfrozen_lake_snow,
        fortran_add(
            lake_snow_heat,
            fortran_multiply(
                lake_snow_ice_mass,
                jnp.float64(LATENT_HEAT_FUSION),
            ),
        ),
        lake_snow_heat,
    )
    top_lake_thickness = jnp.asarray(
        inputs.lake_layer_thickness_column_level,
        dtype=jnp.float64,
    )[:, 0]
    safe_top_lake_thickness = jnp.where(
        lake_column,
        top_lake_thickness,
        jnp.float64(1.0),
    )
    top_lake_heat_capacity = fortran_multiply(
        fortran_multiply(
            jnp.float64(LIQUID_WATER_SPECIFIC_HEAT),
            jnp.float64(DENSITY_FRESH_WATER),
        ),
        safe_top_lake_thickness,
    )
    lake_heat_above_freezing = fortran_multiply(
        fortran_multiply(
            fortran_multiply(
                fortran_add(lake_temperature[:, 0], -freezing),
                jnp.float64(LIQUID_WATER_SPECIFIC_HEAT),
            ),
            jnp.float64(DENSITY_FRESH_WATER),
        ),
        safe_top_lake_thickness,
    )
    lake_heat_remaining = fortran_add(lake_heat_above_freezing, -lake_snow_heat)
    top_lake_freezing_capacity = fortran_multiply(
        fortran_multiply(
            jnp.float64(DENSITY_FRESH_WATER),
            safe_top_lake_thickness,
        ),
        jnp.float64(LATENT_HEAT_FUSION),
    )
    snow_removed_into_lake = unfrozen_lake_snow & (
        fortran_add(lake_heat_remaining, top_lake_freezing_capacity) > jnp.float64(0.0)
    )
    layer_count_before_lake_removal = layer_count
    snow_water_before_lake_removal = snow_water_equivalent
    snowmelt_flux = jnp.where(
        snow_removed_into_lake,
        fortran_add(
            jnp.asarray(flux.snowmelt_flux_column, dtype=jnp.float64),
            fortran_divide(lake_snow_ice_mass, time_step),
        ),
        jnp.asarray(flux.snowmelt_flux_column, dtype=jnp.float64),
    )
    snowmelt_energy_flux = jnp.where(
        snow_removed_into_lake,
        fortran_add(
            jnp.asarray(energy.snowmelt_energy_flux_column, dtype=jnp.float64),
            fortran_divide(
                fortran_multiply(
                    lake_snow_ice_mass,
                    jnp.float64(LATENT_HEAT_FUSION),
                ),
                time_step,
            ),
        ),
        jnp.asarray(energy.snowmelt_energy_flux_column, dtype=jnp.float64),
    )
    snowmelt_flux_layer = jnp.asarray(
        flux.snowmelt_flux_column_level,
        dtype=jnp.float64,
    )

    def add_lake_snowmelt_layer(level: int, values: jnp.ndarray) -> jnp.ndarray:
        active = snow_removed_into_lake & (
            snow_level_number[level] >= layer_count_before_lake_removal + jnp.int32(1)
        )
        return values.at[:, level].set(
            jnp.where(
                active,
                fortran_add(
                    values[:, level],
                    fortran_divide(soil_snow_ice[:, level], time_step),
                ),
                values[:, level],
            )
        )

    snowmelt_flux_layer = lax.fori_loop(
        0,
        NUMBER_OF_SNOW_LEVELS,
        add_lake_snowmelt_layer,
        snowmelt_flux_layer,
    )
    snow_drainage_flux = jnp.where(
        snow_removed_into_lake,
        fortran_add(
            snow_drainage_flux,
            fortran_divide(snow_water_before_lake_removal, time_step),
        ),
        snow_drainage_flux,
    )
    layer_count = jnp.where(
        snow_removed_into_lake,
        jnp.int32(0),
        layer_count,
    )
    snow_water_equivalent = jnp.where(
        snow_removed_into_lake,
        jnp.float64(0.0),
        snow_water_equivalent,
    )
    snow_depth = jnp.where(
        snow_removed_into_lake,
        jnp.float64(0.0),
        snow_depth,
    )
    positive_lake_heat = lake_heat_remaining > jnp.float64(0.0)
    warmed_lake_temperature = fortran_add(
        lake_temperature[:, 0],
        -fortran_divide(lake_heat_remaining, top_lake_heat_capacity),
    )
    frozen_lake_fraction = fortran_divide(
        -lake_heat_remaining,
        top_lake_freezing_capacity,
    )
    lake_temperature = lake_temperature.at[:, 0].set(
        jnp.where(
            snow_removed_into_lake,
            jnp.where(positive_lake_heat, warmed_lake_temperature, freezing),
            lake_temperature[:, 0],
        )
    )
    lake_ice_fraction = lake_ice_fraction.at[:, 0].set(
        jnp.where(
            snow_removed_into_lake & ~positive_lake_heat,
            frozen_lake_fraction,
            lake_ice_fraction[:, 0],
        )
    )

    clear_empty_snow_level = (
        initial_snow_column[:, None]
        & (snow_level_number[None, :] <= layer_count[:, None])
        & (layer_count[:, None] > jnp.int32(-NUMBER_OF_SNOW_LEVELS))
    )
    soil_snow_ice = soil_snow_ice.at[:, :NUMBER_OF_SNOW_LEVELS].set(
        jnp.where(
            clear_empty_snow_level,
            jnp.float64(0.0),
            soil_snow_ice[:, :NUMBER_OF_SNOW_LEVELS],
        )
    )
    soil_snow_liquid = soil_snow_liquid.at[:, :NUMBER_OF_SNOW_LEVELS].set(
        jnp.where(
            clear_empty_snow_level,
            jnp.float64(0.0),
            soil_snow_liquid[:, :NUMBER_OF_SNOW_LEVELS],
        )
    )
    soil_snow_temperature = soil_snow_temperature.at[
        :,
        :NUMBER_OF_SNOW_LEVELS,
    ].set(
        jnp.where(
            clear_empty_snow_level,
            jnp.float64(0.0),
            soil_snow_temperature[:, :NUMBER_OF_SNOW_LEVELS],
        )
    )
    thickness = thickness.at[:, :NUMBER_OF_SNOW_LEVELS].set(
        jnp.where(
            clear_empty_snow_level,
            jnp.float64(0.0),
            thickness[:, :NUMBER_OF_SNOW_LEVELS],
        )
    )
    node_depth = node_depth.at[:, :NUMBER_OF_SNOW_LEVELS].set(
        jnp.where(
            clear_empty_snow_level,
            jnp.float64(0.0),
            node_depth[:, :NUMBER_OF_SNOW_LEVELS],
        )
    )
    interface_depth = interface_depth.at[:, :NUMBER_OF_SNOW_LEVELS].set(
        jnp.where(
            clear_empty_snow_level,
            jnp.float64(0.0),
            interface_depth[:, :NUMBER_OF_SNOW_LEVELS],
        )
    )

    final_masks = build_snow_column_masks(
        topology,
        layer_count,
        lake_only=True,
    )
    final_snow_column = final_masks.snow_column
    final_no_snow_column = final_masks.no_snow_column

    snow_ice_total = jnp.where(
        lake_column,
        jnp.float64(0.0),
        jnp.asarray(water.snow_ice_column, dtype=jnp.float64),
    )
    snow_liquid_total = jnp.where(
        lake_column,
        jnp.float64(0.0),
        jnp.asarray(water.snow_liquid_column, dtype=jnp.float64),
    )
    snow_temperature_mass_sum = jnp.where(
        lake_column,
        jnp.float64(0.0),
        jnp.asarray(
            temperature_state.snow_temperature_mass_sum_column,
            dtype=jnp.float64,
        ),
    )

    def accumulate_final_snow(level: int, carry):
        ice_total, liquid_total, temperature_mass = carry
        active = final_snow_column & (snow_level_number[level] >= layer_count + jnp.int32(1))
        updated_ice = fortran_add(ice_total, soil_snow_ice[:, level])
        updated_liquid = fortran_add(liquid_total, soil_snow_liquid[:, level])
        updated_temperature_mass = fortran_add(
            fortran_add(
                temperature_mass,
                fortran_multiply(
                    soil_snow_ice[:, level],
                    soil_snow_temperature[:, level],
                ),
            ),
            fortran_multiply(soil_snow_liquid[:, level], freezing),
        )
        return (
            jnp.where(active, updated_ice, ice_total),
            jnp.where(active, updated_liquid, liquid_total),
            jnp.where(active, updated_temperature_mass, temperature_mass),
        )

    snow_ice_total, snow_liquid_total, snow_temperature_mass_sum = lax.fori_loop(
        0,
        NUMBER_OF_SNOW_LEVELS,
        accumulate_final_snow,
        (snow_ice_total, snow_liquid_total, snow_temperature_mass_sum),
    )

    unresolved_snow = (
        lake_column & (layer_count >= jnp.int32(0)) & (snow_water_equivalent != jnp.float64(0.0))
    )
    liquid_mass = jnp.zeros(number_of_columns, dtype=jnp.float64)
    ice_mass = jnp.where(
        unresolved_snow,
        snow_water_equivalent,
        jnp.float64(0.0),
    )

    def accumulate_water_mass_snow(level: int, carry):
        liquid_total, ice_total = carry
        active = final_snow_column & (snow_level_number[level] >= layer_count + jnp.int32(1))
        return (
            jnp.where(
                active,
                fortran_add(liquid_total, soil_snow_liquid[:, level]),
                liquid_total,
            ),
            jnp.where(
                active,
                fortran_add(ice_total, soil_snow_ice[:, level]),
                ice_total,
            ),
        )

    liquid_mass, ice_mass = lax.fori_loop(
        0,
        NUMBER_OF_SNOW_LEVELS,
        accumulate_water_mass_snow,
        (liquid_mass, ice_mass),
    )

    def accumulate_water_mass_ground(level: int, carry):
        liquid_total, ice_total = carry
        native_index = level + GROUND_SURFACE_INDEX
        return (
            jnp.where(
                lake_column,
                fortran_add(liquid_total, soil_snow_liquid[:, native_index]),
                liquid_total,
            ),
            jnp.where(
                lake_column,
                fortran_add(ice_total, soil_snow_ice[:, native_index]),
                ice_total,
            ),
        )

    liquid_mass, ice_mass = lax.fori_loop(
        0,
        NUMBER_OF_GROUND_LEVELS,
        accumulate_water_mass_ground,
        (liquid_mass, ice_mass),
    )
    end_water_balance = jnp.where(
        lake_column,
        fortran_add(liquid_mass, ice_mass),
        jnp.asarray(water.end_water_balance_column, dtype=jnp.float64),
    )

    ground_thickness = thickness[:, GROUND_SURFACE_INDEX:]
    ground_liquid = soil_snow_liquid[:, GROUND_SURFACE_INDEX:]
    ground_ice = soil_snow_ice[:, GROUND_SURFACE_INDEX:]
    safe_ground_thickness = jnp.where(
        lake_column[:, None],
        ground_thickness,
        jnp.float64(1.0),
    )
    ground_volume_fraction = fortran_add(
        fortran_divide(
            ground_liquid,
            fortran_multiply(
                safe_ground_thickness,
                jnp.float64(DENSITY_FRESH_WATER),
            ),
        ),
        fortran_divide(
            ground_ice,
            fortran_multiply(safe_ground_thickness, jnp.float64(DENSITY_ICE)),
        ),
    )
    soil_water_volume_fraction = jnp.asarray(
        water.soil_water_volume_fraction_column_level,
        dtype=jnp.float64,
    )
    soil_water_volume_fraction = jnp.where(
        lake_column[:, None],
        ground_volume_fraction,
        soil_water_volume_fraction,
    )

    perched_drainage = jnp.where(
        lake_column,
        jnp.float64(0.0),
        jnp.asarray(flux.perched_drainage_flux_column, dtype=jnp.float64),
    )
    surface_water_runoff = jnp.where(
        lake_column,
        jnp.float64(0.0),
        jnp.asarray(flux.surface_water_runoff_flux_column, dtype=jnp.float64),
    )
    saturation_excess = jnp.where(
        lake_column,
        jnp.float64(0.0),
        jnp.asarray(flux.saturation_excess_flux_column, dtype=jnp.float64),
    )
    infiltration = jnp.where(
        lake_column,
        jnp.float64(0.0),
        jnp.asarray(flux.infiltration_flux_column, dtype=jnp.float64),
    )
    surface_runoff = jnp.where(
        lake_column,
        jnp.float64(0.0),
        jnp.asarray(flux.surface_runoff_flux_column, dtype=jnp.float64),
    )
    drainage = jnp.where(
        lake_column,
        jnp.float64(0.0),
        jnp.asarray(flux.drainage_flux_column, dtype=jnp.float64),
    )

    patch_column_rain = jnp.asarray(
        inputs.rainfall_flux_column,
        dtype=jnp.float64,
    )[patch_column]
    patch_column_snow = jnp.asarray(
        inputs.snowfall_flux_column,
        dtype=jnp.float64,
    )[patch_column]
    patch_end_minus_begin = fortran_add(
        end_water_balance[patch_column],
        -jnp.asarray(inputs.begin_water_balance_column, dtype=jnp.float64)[patch_column],
    )
    patch_flood = jnp.asarray(inputs.flood_flux_gridcell, dtype=jnp.float64)[patch_gridcell]
    patch_water_balance_runoff = fortran_add(
        fortran_add(
            fortran_add(
                fortran_add(
                    fortran_add(
                        fortran_add(
                            fortran_add(
                                patch_column_rain,
                                patch_column_snow,
                            ),
                            -jnp.asarray(
                                inputs.total_evaporation_flux_patch,
                                dtype=jnp.float64,
                            ),
                        ),
                        -snow_capping_ice_flux[patch_column],
                    ),
                    -snow_capping_discarded_ice_flux[patch_column],
                ),
                -snow_capping_discarded_liquid_flux[patch_column],
            ),
            -fortran_divide(patch_end_minus_begin, time_step),
        ),
        patch_flood,
    )
    water_balance_runoff = _copy_patch_to_column(
        jnp.asarray(flux.water_balance_runoff_flux_column, dtype=jnp.float64),
        patch_water_balance_runoff,
        patch_column,
        lake_patch,
    )
    flood_flux_column = _copy_patch_to_column(
        jnp.asarray(flux.flood_flux_column, dtype=jnp.float64),
        patch_flood,
        patch_column,
        lake_patch,
    )
    patch_total_runoff = fortran_add(
        drainage[patch_column],
        patch_water_balance_runoff,
    )
    total_runoff = _copy_patch_to_column(
        jnp.asarray(flux.total_runoff_flux_column, dtype=jnp.float64),
        patch_total_runoff,
        patch_column,
        lake_patch,
    )
    patch_top_soil_water = fortran_add(
        patch_rainfall,
        snow_drainage_flux[patch_column],
    )
    top_soil_water_flux = _copy_patch_to_column(
        top_soil_water_flux,
        patch_top_soil_water,
        patch_column,
        lake_patch,
    )
    ice_runoff_snow_capping = _copy_patch_to_column(
        jnp.asarray(
            flux.ice_runoff_snow_capping_flux_column,
            dtype=jnp.float64,
        ),
        snow_capping_ice_flux[patch_column],
        patch_column,
        lake_patch,
    )

    final_top_index = jnp.clip(
        layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS),
        jnp.int32(0),
        jnp.int32(NUMBER_OF_SNOW_LEVELS - 1),
    )
    final_top_water = fortran_add(
        soil_snow_ice[column_index, final_top_index],
        soil_snow_liquid[column_index, final_top_index],
    )
    top_snow_water = jnp.where(
        final_snow_column,
        final_top_water,
        jnp.where(
            final_no_snow_column,
            jnp.float64(0.0),
            jnp.asarray(water.top_snow_water_column, dtype=jnp.float64),
        ),
    )
    snow_grain_radius = jnp.where(
        final_no_snow_column[:, None],
        jnp.float64(0.0),
        snow_grain_radius,
    )
    special = jnp.float64(CLM_SPECIAL_VALUE)
    top_snow_temperature = jnp.where(
        final_no_snow_column,
        special,
        jnp.asarray(temperature_state.top_snow_temperature_column, dtype=jnp.float64),
    )
    top_temperature_gradient = jnp.where(
        final_no_snow_column,
        special,
        jnp.asarray(
            temperature_state.top_temperature_gradient_column,
            dtype=jnp.float64,
        ),
    )
    top_snow_grain_radius = jnp.where(
        final_no_snow_column,
        special,
        jnp.asarray(water.top_snow_grain_radius_column, dtype=jnp.float64),
    )
    top_snow_liquid_fraction = jnp.where(
        final_no_snow_column,
        special,
        jnp.asarray(water.top_snow_liquid_fraction_column, dtype=jnp.float64),
    )

    output_state = LakeHydrologyState(
        geometry=LakeHydrologyGeometryState(
            snow_layer_count_column=layer_count,
            soil_snow_thickness_column_level=thickness,
            soil_snow_node_depth_column_level=node_depth,
            soil_snow_interface_depth_column_interface=interface_depth,
        ),
        temperature=LakeHydrologyTemperatureState(
            lake_temperature_column_level=lake_temperature,
            soil_snow_temperature_column_level=soil_snow_temperature,
            top_temperature_gradient_column=top_temperature_gradient,
            top_snow_temperature_column=top_snow_temperature,
            snow_temperature_mass_sum_column=snow_temperature_mass_sum,
        ),
        water=LakeHydrologyWaterState(
            effective_snow_fraction_column=effective_snow_fraction,
            snow_fraction_column=snow_fraction,
            old_ice_fraction_column_level=old_ice_fraction,
            snow_depth_column=snow_depth,
            snow_water_equivalent_column=snow_water_equivalent,
            integrated_snow_column=integrated_snow,
            snow_ice_column=snow_ice_total,
            snow_liquid_column=snow_liquid_total,
            soil_snow_ice_column_level=soil_snow_ice,
            soil_snow_liquid_column_level=soil_snow_liquid,
            soil_water_volume_fraction_column_level=soil_water_volume_fraction,
            snow_grain_radius_column_level=snow_grain_radius,
            top_snow_grain_radius_column=top_snow_grain_radius,
            top_snow_water_column=top_snow_water,
            top_snow_liquid_fraction_column=top_snow_liquid_fraction,
            end_water_balance_column=end_water_balance,
        ),
        water_flux=LakeHydrologyWaterFluxState(
            ground_precipitation_flux_patch=patch_precipitation,
            ground_snowfall_flux_patch=patch_snowfall,
            ground_rainfall_flux_patch=patch_rainfall,
            ground_evaporation_flux_patch=patch_ground_evaporation,
            snow_sublimation_flux_patch=patch_sublimation,
            snow_dew_flux_patch=patch_snow_dew,
            ground_dew_flux_patch=patch_ground_dew,
            ground_rainfall_flux_column=column_rainfall,
            ground_snowfall_flux_column=column_snowfall,
            ground_precipitation_flux_column=column_precipitation,
            ground_evaporation_flux_column=column_ground_evaporation,
            ground_dew_flux_column=column_ground_dew,
            snow_dew_flux_column=column_snow_dew,
            snow_sublimation_flux_column=column_sublimation,
            total_evaporation_flux_column=column_total_evaporation,
            snowmelt_flux_column=snowmelt_flux,
            snowmelt_flux_column_level=snowmelt_flux_layer,
            snow_capping_ice_flux_column=snow_capping_ice_flux,
            snow_capping_liquid_flux_column=snow_capping_liquid_flux,
            snow_capping_discarded_ice_flux_column=(snow_capping_discarded_ice_flux),
            snow_capping_discarded_liquid_flux_column=(snow_capping_discarded_liquid_flux),
            perched_drainage_flux_column=perched_drainage,
            surface_water_runoff_flux_column=surface_water_runoff,
            snow_drainage_flux_column=snow_drainage_flux,
            saturation_excess_flux_column=saturation_excess,
            surface_runoff_flux_column=surface_runoff,
            drainage_flux_column=drainage,
            infiltration_flux_column=infiltration,
            water_balance_runoff_flux_column=water_balance_runoff,
            total_runoff_flux_column=total_runoff,
            ice_runoff_snow_capping_flux_column=ice_runoff_snow_capping,
            top_soil_water_flux_column=top_soil_water_flux,
            snow_layer_to_top_soil_flux_column=snow_layer_to_top_soil_flux,
            flood_flux_column=flood_flux_column,
        ),
        energy_flux=LakeHydrologyEnergyFluxState(
            snowmelt_energy_flux_column=snowmelt_energy_flux,
            sensible_heat_total_patch=sensible_heat_total_patch,
            sensible_heat_ground_patch=sensible_heat_ground_patch,
            soil_heat_flux_ground_patch=soil_heat_flux_ground_patch,
            net_ground_heat_flux_patch=net_ground_heat_flux_patch,
            ground_lake_heat_flux_patch=ground_lake_heat_flux_patch,
        ),
        aerosol=aerosol,
        lake_ice_fraction_column_level=lake_ice_fraction,
    )
    return LakeHydrologyResult(
        state=output_state,
        diagnostics=LakeHydrologyDiagnostics(
            lake_column=lake_column,
            lake_patch=lake_patch,
            initial_snow_column=initial_snow_column,
            initial_no_snow_column=initial_no_snow_column,
            final_snow_column=final_snow_column,
            final_no_snow_column=final_no_snow_column,
            new_snow_layer_column=new_snow_layer,
            unfrozen_lake_snow_column=unfrozen_lake_snow,
            snow_removed_into_lake_column=snow_removed_into_lake,
            lake_snow_ice_mass_column=lake_snow_ice_mass,
            lake_snow_heat_requirement_column=lake_snow_heat,
        ),
    )
