
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.forcing.atmosphere import ColumnAtmosphericForcing
from clm_jax.forcing.datm import DatmForcing
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.luna import LunaRestartData
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_LAKE,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    SubgridTopology,
)
from clm_jax.physics.canopy_fluxes import (
    CanopyEnergyInputs,
    CanopyFluxesActiveState,
    CanopyFluxesLunaInputs,
    CanopyFluxesPostprocessInputs,
)
from clm_jax.physics.canopy_hydrology import CanopyWaterState
from clm_jax.physics.canopy_radiation import CanopySunShadeState
from clm_jax.physics.canopy_step import CanopyHydrologyState
from clm_jax.physics.canopy_temperature import (
    NUMBER_OF_SNOW_LEVELS,
    CanopyTemperatureInputs,
    CanopyTemperatureState,
)
from clm_jax.physics.daylength import daylength
from clm_jax.physics.luna import LunaCapacityInputs
from clm_jax.physics.solar import FIXED_YEAR_2000_ORBIT, solar_declination
from clm_jax.physics.solar_runtime import native_grid_radians
from clm_jax.physics.surface_radiation import SurfaceRadiationState


class CanopyFluxesPftParameters(NamedTuple):

    leaf_dimension_pft: jnp.ndarray


class CanopyFluxesFastInputs(NamedTuple):

    energy: CanopyEnergyInputs
    postprocess: CanopyFluxesPostprocessInputs


def _configured_parameter_path(case: CimeCase) -> Path:
    group = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    declared = group.get("paramfile")
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(f"clm_inparm in {case.case_docs / 'lnd_in'} lacks paramfile")
    path = Path(case.expand(declared)).expanduser()
    if not path.is_absolute():
        path = case.root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM parameter file does not exist: {path}")
    return path


def load_canopy_fluxes_pft_parameters(
    case: CimeCase,
    parameter_data_path: str | Path | None = None,
) -> CanopyFluxesPftParameters:

    path = (
        _configured_parameter_path(case)
        if parameter_data_path is None
        else Path(parameter_data_path).expanduser().resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(f"CLM parameter file does not exist: {path}")
    values = load_native_variables(path, ["dleaf"], dtype=np.float64)
    leaf_dimension = jnp.asarray(values["dleaf"], dtype=jnp.float64)
    if leaf_dimension.ndim != 1:
        raise ValueError("parameter dleaf must be one-dimensional over PFT")
    host = np.asarray(leaf_dimension)
    if not np.all(np.isfinite(host)) or np.any(host < 0.0):
        raise ValueError("parameter dleaf must contain finite nonnegative values")
    return CanopyFluxesPftParameters(leaf_dimension_pft=leaf_dimension)


def _validate_upstream_shapes(
    topology: SubgridTopology,
    grid_forcing: DatmForcing,
    column_forcing: ColumnAtmosphericForcing,
    temperature_inputs: CanopyTemperatureInputs,
    temperature: CanopyTemperatureState,
    hydrology: CanopyHydrologyState,
    surface_radiation: SurfaceRadiationState,
    canopy_radiation: CanopySunShadeState,
    pft_parameters: CanopyFluxesPftParameters,
) -> None:
    number_of_gridcells = topology.grid.longitude_degrees.shape[0]
    number_of_columns = topology.column.gridcell.shape[0]
    number_of_patches = topology.patch.gridcell.shape[0]
    grid_shape = (number_of_gridcells,)
    column_shape = (number_of_columns,)
    patch_shape = (number_of_patches,)

    for name in ("wind_u", "wind_v"):
        if jnp.shape(getattr(grid_forcing, name)) != grid_shape:
            raise ValueError(f"grid forcing {name} must have shape {grid_shape}")
    for name in (
        "potential_temperature",
        "specific_humidity",
        "pressure",
        "density",
        "longwave_down",
    ):
        if jnp.shape(getattr(column_forcing, name)) != column_shape:
            raise ValueError(f"column forcing {name} must have shape {column_shape}")
    for name in (
        "snow_layer_count_column",
        "surface_water_temperature_column",
        "surface_water_fraction_column",
    ):
        if jnp.shape(getattr(temperature_inputs, name)) != column_shape:
            raise ValueError(f"temperature input {name} must have shape {column_shape}")
    for name in (
        "vegetation_exposed_patch",
        "exposed_leaf_area_index_patch",
        "exposed_stem_area_index_patch",
        "canopy_top_height_patch",
    ):
        if jnp.shape(getattr(temperature_inputs, name)) != patch_shape:
            raise ValueError(f"temperature input {name} must have shape {patch_shape}")
    if temperature.column.temperature_before_column_level.shape[0] != number_of_columns:
        raise ValueError("temperature-before state must contain every column")
    if temperature.column.temperature_before_column_level.shape[1] <= NUMBER_OF_SNOW_LEVELS:
        raise ValueError("temperature-before state must include snow and top-soil levels")
    for name in (
        "ground_temperature_column",
        "ground_specific_humidity_column",
        "snow_surface_specific_humidity_column",
        "soil_surface_specific_humidity_column",
        "surface_water_specific_humidity_column",
        "ground_specific_humidity_temperature_derivative_column",
        "latent_heat_column",
        "ground_emissivity_column",
        "ground_momentum_roughness_column",
        "soil_evaporation_resistance_column",
        "surface_water_temperature_before_column",
    ):
        if jnp.shape(getattr(temperature.column, name)) != column_shape:
            raise ValueError(f"temperature state {name} must have shape {column_shape}")
    for name in (
        "vegetation_emissivity_patch",
        "forcing_height_wind_patch",
        "forcing_height_temperature_patch",
        "forcing_height_humidity_patch",
        "reference_temperature_patch",
    ):
        if jnp.shape(getattr(temperature.patch, name)) != patch_shape:
            raise ValueError(f"temperature state {name} must have shape {patch_shape}")
    for value in hydrology.canopy_water:
        if jnp.shape(value) != patch_shape:
            raise ValueError("canopy-water state must contain every patch")
    for value in hydrology.canopy_wet_fractions:
        if jnp.shape(value) != patch_shape:
            raise ValueError("canopy wet-fraction state must contain every patch")
    for name in (
        "surface_water_fraction_column",
        "surface_water_fraction_without_snow_column",
    ):
        if jnp.shape(getattr(hydrology, name)) != column_shape:
            raise ValueError(f"hydrology state {name} must have shape {column_shape}")
    for name in ("depth", "effective_snow_fraction"):
        if jnp.shape(getattr(hydrology.snow, name)) != column_shape:
            raise ValueError(f"snow state {name} must have shape {column_shape}")
    if jnp.shape(surface_radiation.fluxes.absorbed_vegetation_patch) != patch_shape:
        raise ValueError("surface radiation must contain absorbed vegetation per patch")
    for name in ("sunlit_leaf_area_patch", "shaded_leaf_area_patch"):
        if jnp.shape(getattr(canopy_radiation, name)) != patch_shape:
            raise ValueError(f"canopy radiation {name} must have shape {patch_shape}")
    if pft_parameters.leaf_dimension_pft.ndim != 1:
        raise ValueError("leaf dimension must be one-dimensional over PFT")


def assemble_canopy_fluxes_fast_inputs(
    topology: SubgridTopology,
    grid_forcing: DatmForcing,
    column_forcing: ColumnAtmosphericForcing,
    temperature_inputs: CanopyTemperatureInputs,
    temperature: CanopyTemperatureState,
    hydrology: CanopyHydrologyState,
    surface_radiation: SurfaceRadiationState,
    canopy_radiation: CanopySunShadeState,
    pft_parameters: CanopyFluxesPftParameters,
) -> CanopyFluxesFastInputs:

    _validate_upstream_shapes(
        topology,
        grid_forcing,
        column_forcing,
        temperature_inputs,
        temperature,
        hydrology,
        surface_radiation,
        canopy_radiation,
        pft_parameters,
    )
    patch = topology.patch
    column_index = jnp.asarray(patch.column, dtype=jnp.int32)
    gridcell_index = jnp.asarray(patch.gridcell, dtype=jnp.int32)
    vegetation_type = jnp.asarray(patch.vegetation_type, dtype=jnp.int32)

    landunit_type = jnp.asarray(patch.landunit_type, dtype=jnp.int32)
    urban = (landunit_type >= jnp.int32(LANDUNIT_URBAN_MIN)) & (
        landunit_type <= jnp.int32(LANDUNIT_URBAN_MAX)
    )
    vegetation_exposed = jnp.asarray(
        temperature_inputs.vegetation_exposed_patch,
        dtype=jnp.int32,
    )
    active = (
        jnp.asarray(patch.active, dtype=jnp.bool_)
        & (landunit_type != jnp.int32(LANDUNIT_LAKE))
        & ~urban
        & (vegetation_exposed > jnp.int32(0))
    )

    def column(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(value, dtype=jnp.float64)[column_index]

    def gridcell(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(value, dtype=jnp.float64)[gridcell_index]

    before = jnp.asarray(
        temperature.column.temperature_before_column_level,
        dtype=jnp.float64,
    )
    snow_layer_count = jnp.asarray(
        temperature_inputs.snow_layer_count_column,
        dtype=jnp.int32,
    )
    surface_level = jnp.clip(
        snow_layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS),
        jnp.int32(0),
        jnp.int32(before.shape[1] - 1),
    )
    row = jnp.arange(before.shape[0], dtype=jnp.int32)
    snow_surface_temperature_column = before[row, surface_level]
    soil_surface_temperature_column = before[:, NUMBER_OF_SNOW_LEVELS]

    energy = CanopyEnergyInputs(
        active_patch=active,
        eastward_wind_patch=gridcell(grid_forcing.wind_u),
        northward_wind_patch=gridcell(grid_forcing.wind_v),
        reference_temperature_patch=jnp.asarray(
            temperature.patch.reference_temperature_patch,
            dtype=jnp.float64,
        ),
        atmospheric_potential_temperature_patch=column(
            column_forcing.potential_temperature
        ),
        atmospheric_specific_humidity_patch=column(column_forcing.specific_humidity),
        atmospheric_pressure_patch=column(column_forcing.pressure),
        air_density_patch=column(column_forcing.density),
        downward_longwave_patch=column(column_forcing.longwave_down),
        ground_temperature_patch=column(temperature.column.ground_temperature_column),
        ground_specific_humidity_patch=column(
            temperature.column.ground_specific_humidity_column
        ),
        snow_surface_temperature_patch=column(snow_surface_temperature_column),
        soil_surface_temperature_patch=column(soil_surface_temperature_column),
        surface_water_temperature_patch=column(
            temperature.column.surface_water_temperature_before_column
        ),
        snow_fraction_patch=column(hydrology.snow.effective_snow_fraction),
        surface_water_fraction_patch=column(hydrology.surface_water_fraction_column),
        ground_emissivity_patch=column(temperature.column.ground_emissivity_column),
        absorbed_shortwave_vegetation_patch=jnp.asarray(
            surface_radiation.fluxes.absorbed_vegetation_patch,
            dtype=jnp.float64,
        ),
        vegetation_emissivity_patch=jnp.asarray(
            temperature.patch.vegetation_emissivity_patch,
            dtype=jnp.float64,
        ),
        exposed_leaf_area_index_patch=jnp.asarray(
            temperature_inputs.exposed_leaf_area_index_patch,
            dtype=jnp.float64,
        ),
        exposed_stem_area_index_patch=jnp.asarray(
            temperature_inputs.exposed_stem_area_index_patch,
            dtype=jnp.float64,
        ),
        sunlit_leaf_area_index_patch=jnp.asarray(
            canopy_radiation.sunlit_leaf_area_patch,
            dtype=jnp.float64,
        ),
        shaded_leaf_area_index_patch=jnp.asarray(
            canopy_radiation.shaded_leaf_area_patch,
            dtype=jnp.float64,
        ),
        vegetation_fraction_without_snow_patch=jnp.asarray(
            jnp.where(patch.active, vegetation_exposed, jnp.int32(0)),
            dtype=jnp.float64,
        ),
        dry_foliage_fraction_patch=jnp.asarray(
            hydrology.canopy_wet_fractions.dry_leaf,
            dtype=jnp.float64,
        ),
        wet_foliage_fraction_patch=jnp.asarray(
            hydrology.canopy_wet_fractions.wet,
            dtype=jnp.float64,
        ),
        canopy_water_patch=jnp.asarray(hydrology.canopy_water.total_water, dtype=jnp.float64),
        snow_depth_patch=column(hydrology.snow.depth),
        soil_evaporation_resistance_patch=column(
            temperature.column.soil_evaporation_resistance_column
        ),
        ground_momentum_roughness_patch=column(
            temperature.column.ground_momentum_roughness_column
        ),
        canopy_top_height_patch=jnp.asarray(
            temperature_inputs.canopy_top_height_patch,
            dtype=jnp.float64,
        ),
        leaf_dimension_patch=jnp.asarray(
            pft_parameters.leaf_dimension_pft,
            dtype=jnp.float64,
        )[vegetation_type],
        forcing_height_wind_patch=jnp.asarray(
            temperature.patch.forcing_height_wind_patch,
            dtype=jnp.float64,
        ),
        forcing_height_temperature_patch=jnp.asarray(
            temperature.patch.forcing_height_temperature_patch,
            dtype=jnp.float64,
        ),
        forcing_height_humidity_patch=jnp.asarray(
            temperature.patch.forcing_height_humidity_patch,
            dtype=jnp.float64,
        ),
    )
    postprocess = CanopyFluxesPostprocessInputs(
        snow_surface_specific_humidity_patch=column(
            temperature.column.snow_surface_specific_humidity_column
        ),
        soil_surface_specific_humidity_patch=column(
            temperature.column.soil_surface_specific_humidity_column
        ),
        surface_water_specific_humidity_patch=column(
            temperature.column.surface_water_specific_humidity_column
        ),
        ground_specific_humidity_temperature_derivative_patch=column(
            temperature.column.ground_specific_humidity_temperature_derivative_column
        ),
        ground_latent_heat_patch=column(temperature.column.latent_heat_column),
    )
    return CanopyFluxesFastInputs(energy=energy, postprocess=postprocess)


def assemble_canopy_fluxes_luna_inputs(
    topology: SubgridTopology,
    calendar_day: jnp.ndarray,
    canopy_radiation: CanopySunShadeState,
    restart: LunaRestartData,
    *,
    solar_declination_radians: jnp.ndarray | None = None,
) -> CanopyFluxesLunaInputs:

    patch_count = topology.patch.gridcell.shape[0]
    gridcell_count = topology.grid.latitude_degrees.shape[0]
    patch_shape = (patch_count,)
    if jnp.shape(calendar_day) != ():
        raise ValueError("LUNA calendar day must be scalar")
    if restart.topology.patch.gridcell.shape != patch_shape:
        raise ValueError("LUNA restart and active topology patch dimensions disagree")
    if restart.topology.grid.latitude_degrees.shape != (gridcell_count,):
        raise ValueError("LUNA restart and active topology grid dimensions disagree")

    layer_shape = restart.capacity.maximum_carboxylation_rate_by_layer_patch.shape
    if len(layer_shape) != 2 or layer_shape[0] != patch_count or layer_shape[1] < 1:
        raise ValueError("LUNA restart capacity must have shape (patch, canopy_layer>=1)")
    layer_fields = (
        canopy_radiation.sunlit_leaf_area_by_layer_patch,
        canopy_radiation.shaded_leaf_area_by_layer_patch,
        canopy_radiation.absorbed_par_sunlit_by_layer_patch,
        canopy_radiation.absorbed_par_shaded_by_layer_patch,
        restart.capacity_inputs.total_leaf_area_by_layer_patch,
        restart.ten_day.absorbed_par_mean_by_layer_patch,
        restart.ten_day.absorbed_par_maximum_by_layer_patch,
    )
    if any(jnp.shape(value) != layer_shape for value in layer_fields):
        raise ValueError("current radiation and LUNA restart layer dimensions disagree")

    scalar_fields = (
        *restart.capacity_inputs[:4],
        restart.capacity_inputs.radiative_layer_count_patch,
        restart.capacity_inputs.leaf_nitrogen_concentration_patch,
        restart.capacity_inputs.air_temperature_ten_day_patch,
        restart.daily.daytime_temperature_sum_patch,
        restart.daily.photosynthesis_sum_patch,
        restart.ten_day.daytime_temperature_patch,
        restart.ten_day.nighttime_temperature_patch,
        restart.ten_day.relative_humidity_patch,
        restart.ten_day.boundary_layer_resistance_patch,
    )
    if any(jnp.shape(value) != patch_shape for value in scalar_fields):
        raise ValueError("LUNA restart scalar fields must contain every patch")

    latitude_radians = native_grid_radians(topology.grid.latitude_degrees)
    if solar_declination_radians is None:
        solar_declination_radians, _ = solar_declination(
            jnp.asarray(calendar_day, dtype=jnp.float64),
            FIXED_YEAR_2000_ORBIT,
        )
    daylength_seconds = daylength(latitude_radians, solar_declination_radians)
    capacity_restart = restart.capacity_inputs
    capacity = LunaCapacityInputs(
        atmospheric_pressure_ten_day_patch=jnp.asarray(
            capacity_restart.atmospheric_pressure_ten_day_patch,
            dtype=jnp.float64,
        ),
        atmospheric_co2_ten_day_patch=jnp.asarray(
            capacity_restart.atmospheric_co2_ten_day_patch,
            dtype=jnp.float64,
        ),
        atmospheric_o2_ten_day_patch=jnp.asarray(
            capacity_restart.atmospheric_o2_ten_day_patch,
            dtype=jnp.float64,
        ),
        total_leaf_area_patch=jnp.asarray(
            capacity_restart.total_leaf_area_patch,
            dtype=jnp.float64,
        ),
        total_leaf_area_by_layer_patch=jnp.asarray(
            capacity_restart.total_leaf_area_by_layer_patch,
            dtype=jnp.float64,
        ),
        daylength_seconds_gridcell=daylength_seconds,
        absorbed_par_mean_by_layer_patch=jnp.asarray(
            restart.ten_day.absorbed_par_mean_by_layer_patch,
            dtype=jnp.float64,
        ),
        absorbed_par_maximum_by_layer_patch=jnp.asarray(
            restart.ten_day.absorbed_par_maximum_by_layer_patch,
            dtype=jnp.float64,
        ),
        radiative_layer_count_patch=jnp.asarray(
            capacity_restart.radiative_layer_count_patch,
            dtype=jnp.int32,
        ),
        leaf_nitrogen_concentration_patch=jnp.asarray(
            capacity_restart.leaf_nitrogen_concentration_patch,
            dtype=jnp.float64,
        ),
        air_temperature_ten_day_patch=jnp.asarray(
            capacity_restart.air_temperature_ten_day_patch,
            dtype=jnp.float64,
        ),
        daytime_temperature_accumulator_patch=jnp.asarray(
            restart.daily.daytime_temperature_sum_patch,
            dtype=jnp.float64,
        ),
        daytime_leaf_temperature_ten_day_patch=jnp.asarray(
            restart.ten_day.daytime_temperature_patch,
            dtype=jnp.float64,
        ),
        nighttime_leaf_temperature_ten_day_patch=jnp.asarray(
            restart.ten_day.nighttime_temperature_patch,
            dtype=jnp.float64,
        ),
        relative_humidity_ten_day_patch=jnp.asarray(
            restart.ten_day.relative_humidity_patch,
            dtype=jnp.float64,
        ),
        boundary_layer_resistance_ten_day_patch=jnp.asarray(
            restart.ten_day.boundary_layer_resistance_patch,
            dtype=jnp.float64,
        ),
        daily_photosynthesis_patch=jnp.asarray(
            restart.daily.photosynthesis_sum_patch,
            dtype=jnp.float64,
        ),
    )
    return CanopyFluxesLunaInputs(
        absorbed_par_sunlit_by_layer_patch=jnp.asarray(
            canopy_radiation.absorbed_par_sunlit_by_layer_patch,
            dtype=jnp.float64,
        ),
        absorbed_par_shaded_by_layer_patch=jnp.asarray(
            canopy_radiation.absorbed_par_shaded_by_layer_patch,
            dtype=jnp.float64,
        ),
        sunlit_leaf_area_by_layer_patch=jnp.asarray(
            canopy_radiation.sunlit_leaf_area_by_layer_patch,
            dtype=jnp.float64,
        ),
        shaded_leaf_area_by_layer_patch=jnp.asarray(
            canopy_radiation.shaded_leaf_area_by_layer_patch,
            dtype=jnp.float64,
        ),
        radiative_layer_count_patch=jnp.asarray(
            capacity_restart.radiative_layer_count_patch,
            dtype=jnp.int32,
        ),
        capacity_inputs=capacity,
    )


def synchronize_canopy_fluxes_water_state(
    state: CanopyFluxesActiveState,
    canopy_water: CanopyWaterState,
) -> CanopyFluxesActiveState:

    expected = state.postprocess.canopy_water_patch.shape
    if any(jnp.shape(value) != expected for value in canopy_water):
        raise ValueError("canopy hydrology and CanopyFluxes patch dimensions disagree")
    return state._replace(
        postprocess=state.postprocess._replace(
            canopy_water_patch=jnp.asarray(canopy_water.total_water, dtype=jnp.float64),
            liquid_canopy_water_patch=jnp.asarray(canopy_water.liquid, dtype=jnp.float64),
            snow_canopy_water_patch=jnp.asarray(canopy_water.snow, dtype=jnp.float64),
        )
    )
