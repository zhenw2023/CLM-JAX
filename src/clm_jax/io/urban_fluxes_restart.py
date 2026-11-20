
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.forcing.datm import DatmForcing
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    COLUMN_URBAN_IMPERVIOUS_ROAD,
    COLUMN_URBAN_PERVIOUS_ROAD,
    COLUMN_URBAN_ROOF,
    COLUMN_URBAN_SHADE_WALL,
    COLUMN_URBAN_SUN_WALL,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    RestartClock,
    SubgridTopology,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)
from clm_jax.physics.canopy_step import CanopyHydrologyState
from clm_jax.physics.canopy_temperature import (
    NUMBER_OF_GROUND_LEVELS,
    NUMBER_OF_SNOW_LEVELS,
    NUMBER_OF_SOIL_LEVELS,
    NUMBER_OF_TOTAL_LEVELS,
    NUMBER_OF_URBAN_LEVELS,
    CanopyTemperatureInputs,
    CanopyTemperatureParameters,
    CanopyTemperatureState,
)
from clm_jax.physics.friction_velocity import FrictionVelocityState
from clm_jax.physics.urban_fluxes import (
    BUILDING_TEMPERATURE_METHOD_PROGNOSTIC,
    BUILDING_TEMPERATURE_METHOD_SIMPLE,
    NUMBER_OF_URBAN_SURFACES,
    UrbanSurfaceFluxInputs,
    UrbanTurbulenceInputs,
    UrbanTurbulenceState,
)


class UrbanFluxesNativeMap(NamedTuple):

    landunit_index: jnp.ndarray
    gridcell_index: jnp.ndarray
    surface_column_index: jnp.ndarray
    surface_patch_index: jnp.ndarray


class UrbanFluxesStaticParameters(NamedTuple):

    roof_height_landunit: jnp.ndarray
    roof_fraction_landunit: jnp.ndarray
    canyon_height_width_ratio_landunit: jnp.ndarray
    pervious_road_fraction_landunit: jnp.ndarray
    canyon_wind_height_landunit: jnp.ndarray
    traffic_heat_factor_landunit: jnp.ndarray


class UrbanFluxesRestartState(NamedTuple):

    turbulence: UrbanTurbulenceState
    internal_building_temperature_landunit: jnp.ndarray
    air_conditioning_flux_landunit: jnp.ndarray
    heating_flux_landunit: jnp.ndarray
    air_conditioning_flux_surface: jnp.ndarray
    heating_flux_surface: jnp.ndarray


class UrbanFluxesRestartData(NamedTuple):

    topology: SubgridTopology
    clock: RestartClock
    native_map: UrbanFluxesNativeMap
    parameters: UrbanFluxesStaticParameters
    state: UrbanFluxesRestartState
    building_temperature_method: int
    maximum_stable_zeta: float
    restart_path: Path
    surface_data_path: Path


class UrbanFluxesStepData(NamedTuple):

    turbulence_inputs: UrbanTurbulenceInputs
    turbulence_state: UrbanTurbulenceState
    surface_inputs: UrbanSurfaceFluxInputs


_SURFACE_TYPES = np.array(
    (
        COLUMN_URBAN_ROOF,
        COLUMN_URBAN_SUN_WALL,
        COLUMN_URBAN_SHADE_WALL,
        COLUMN_URBAN_IMPERVIOUS_ROAD,
        COLUMN_URBAN_PERVIOUS_ROAD,
    ),
    dtype=np.int64,
)

_RESTART_VARIABLES = (
    "taf",
    "qaf",
    "t_building",
    "URBAN_AC_L",
    "URBAN_HEAT_L",
)

_SURFACE_VARIABLES = (
    "HT_ROOF",
    "WTLUNIT_ROOF",
    "CANYON_HWR",
    "WTROAD_PERV",
    "WIND_HGT_CANYON",
)


def _configured_surface_data(case: CimeCase) -> Path:
    values = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    declared = values.get("fsurdat")
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(f"clm_inparm in {case.case_docs / 'lnd_in'} lacks fsurdat")
    path = Path(case.expand(declared)).expanduser()
    if not path.is_absolute():
        path = case.root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM surface data does not exist: {path}")
    return path


def _urban_native_map(topology: SubgridTopology) -> UrbanFluxesNativeMap:
    landunit_type = np.asarray(topology.landunit.landunit_type, dtype=np.int64)
    landunit_active = np.asarray(topology.landunit.active, dtype=bool)
    urban_landunit = (
        landunit_active
        & (landunit_type >= LANDUNIT_URBAN_MIN)
        & (landunit_type <= LANDUNIT_URBAN_MAX)
    )
    landunit_index = np.flatnonzero(urban_landunit)
    if landunit_index.size == 0:
        return UrbanFluxesNativeMap(
            landunit_index=jnp.empty((0,), dtype=jnp.int32),
            gridcell_index=jnp.empty((0,), dtype=jnp.int32),
            surface_column_index=jnp.empty((0, NUMBER_OF_URBAN_SURFACES), dtype=jnp.int32),
            surface_patch_index=jnp.empty((0, NUMBER_OF_URBAN_SURFACES), dtype=jnp.int32),
        )

    compact_landunit = np.full(landunit_type.size, -1, dtype=np.int64)
    compact_landunit[landunit_index] = np.arange(landunit_index.size, dtype=np.int64)

    column_landunit = np.asarray(topology.column.landunit, dtype=np.int64)
    column_type = np.asarray(topology.column.column_type, dtype=np.int64)
    column_active = np.asarray(topology.column.active, dtype=bool)
    surface_column = np.full((landunit_index.size, NUMBER_OF_URBAN_SURFACES), -1, dtype=np.int64)
    for surface, surface_type in enumerate(_SURFACE_TYPES):
        selected = column_active & (column_type == surface_type)
        columns = np.flatnonzero(selected)
        parents = column_landunit[columns]
        valid_parent = (parents >= 0) & (parents < compact_landunit.size)
        columns = columns[valid_parent]
        compact = compact_landunit[parents[valid_parent]]
        urban_parent = compact >= 0
        columns = columns[urban_parent]
        compact = compact[urban_parent]
        if np.unique(compact).size != compact.size:
            raise ValueError(f"an urban landunit has duplicate type-{surface_type} columns")
        surface_column[compact, surface] = columns

    missing = np.argwhere(surface_column < 0)
    if missing.size:
        compact, surface = missing[0]
        raise ValueError(
            "active urban landunit "
            f"{landunit_index[compact]} lacks column type {_SURFACE_TYPES[surface]}"
        )

    patch_column = np.asarray(topology.patch.column, dtype=np.int64)
    patch_active = np.asarray(topology.patch.active, dtype=bool)
    surface_patch = np.full_like(surface_column, -1)
    column_to_slot = {
        int(column): (compact, surface)
        for compact in range(surface_column.shape[0])
        for surface, column in enumerate(surface_column[compact])
    }
    for patch in np.flatnonzero(patch_active):
        slot = column_to_slot.get(int(patch_column[patch]))
        if slot is None:
            continue
        if surface_patch[slot] >= 0:
            raise ValueError(f"urban column {patch_column[patch]} has duplicate active patches")
        surface_patch[slot] = patch
    missing = np.argwhere(surface_patch < 0)
    if missing.size:
        compact, surface = missing[0]
        raise ValueError(f"urban column {surface_column[compact, surface]} lacks one active patch")

    gridcell = np.asarray(topology.landunit.gridcell, dtype=np.int64)[landunit_index]
    number_of_gridcells = topology.grid.longitude_degrees.shape[0]
    if np.any((gridcell < 0) | (gridcell >= number_of_gridcells)):
        raise ValueError("an active urban landunit has an invalid parent gridcell")
    return UrbanFluxesNativeMap(
        landunit_index=jnp.asarray(landunit_index, dtype=jnp.int32),
        gridcell_index=jnp.asarray(gridcell, dtype=jnp.int32),
        surface_column_index=jnp.asarray(surface_column, dtype=jnp.int32),
        surface_patch_index=jnp.asarray(surface_patch, dtype=jnp.int32),
    )


def _select_urban_surface_fields(
    topology: SubgridTopology,
    native_map: UrbanFluxesNativeMap,
    values: dict[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    landunit = np.asarray(native_map.landunit_index, dtype=np.int64)
    gridcell = np.asarray(native_map.gridcell_index, dtype=np.int64)
    density = (
        np.asarray(topology.landunit.landunit_type, dtype=np.int64)[landunit] - LANDUNIT_URBAN_MIN
    )
    latitude = np.asarray(topology.grid.latitude_index, dtype=np.int64)[gridcell]
    longitude = np.asarray(topology.grid.longitude_index, dtype=np.int64)[gridcell]
    result: dict[str, jnp.ndarray] = {}
    for name in _SURFACE_VARIABLES:
        source = np.asarray(values[name], dtype=np.float64)
        if source.ndim != 3 or source.shape[0] != 3:
            raise ValueError(f"surface variable {name} must have dimensions (numurbl, lat, lon)")
        if np.any(
            (latitude < 0)
            | (latitude >= source.shape[1])
            | (longitude < 0)
            | (longitude >= source.shape[2])
        ):
            raise ValueError(f"native grid indices are outside surface variable {name}")
        selected = source[density, latitude, longitude]
        if not np.all(np.isfinite(selected)):
            raise ValueError(f"surface variable {name} is missing on an urban landunit")
        result[name] = jnp.asarray(selected, dtype=jnp.float64)
    return result


def _finite_selected(
    values: dict[str, jnp.ndarray],
    name: str,
    index: np.ndarray,
) -> jnp.ndarray:
    selected = np.asarray(values[name], dtype=np.float64)[index]
    if not np.all(np.isfinite(selected)):
        raise ValueError(f"restart variable {name} is missing on an active urban landunit")
    return jnp.asarray(selected, dtype=jnp.float64)


def load_urban_fluxes_restart(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
) -> UrbanFluxesRestartData:

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    surface_data = (
        _configured_surface_data(case)
        if surface_data_path is None
        else Path(surface_data_path).expanduser().resolve()
    )
    if not surface_data.is_file():
        raise FileNotFoundError(f"CLM surface data does not exist: {surface_data}")

    dimensions = load_dimension_sizes(restart)
    expected_dimensions = {
        "levsno": NUMBER_OF_SNOW_LEVELS,
        "levgrnd": NUMBER_OF_GROUND_LEVELS,
        "levtot": NUMBER_OF_TOTAL_LEVELS,
    }
    for name, expected in expected_dimensions.items():
        if dimensions.get(name) != expected:
            raise ValueError(
                f"restart dimension {name} must be {expected}, got {dimensions.get(name)}"
            )

    urban_controls = read_case_namelist_group(case, "lnd_in", "clmu_inparm")
    method = int(urban_controls.get("building_temp_method", -1))
    if method not in (
        BUILDING_TEMPERATURE_METHOD_SIMPLE,
        BUILDING_TEMPERATURE_METHOD_PROGNOSTIC,
    ):
        raise ValueError(f"unsupported CLM building_temp_method={method}")
    hvac = str(urban_controls.get("urban_hac", "")).strip().strip("'").upper()
    if hvac != "ON_WASTEHEAT":
        raise ValueError(
            f"UrbanFluxes adapter currently requires urban_hac='ON_WASTEHEAT', got {hvac!r}"
        )
    urban_traffic = bool(urban_controls.get("urban_traffic", False))
    friction = read_case_namelist_group(case, "lnd_in", "friction_velocity")
    maximum_stable_zeta = float(friction.get("zetamaxstable", 0.5))

    topology = load_subgrid_topology(restart)
    native_map = _urban_native_map(topology)
    restart_values = load_native_variables(restart, list(_RESTART_VARIABLES))
    surface_values = load_native_variables(surface_data, list(_SURFACE_VARIABLES))
    selected_surface = _select_urban_surface_fields(
        topology,
        native_map,
        surface_values,
    )

    landunit = np.asarray(native_map.landunit_index, dtype=np.int64)
    number_of_landunits = topology.landunit.gridcell.shape[0]
    for name in _RESTART_VARIABLES:
        if restart_values[name].shape != (number_of_landunits,):
            raise ValueError(f"restart variable {name} has unexpected shape")

    ratio = selected_surface["CANYON_HWR"]
    traffic = jnp.where(
        jnp.asarray(urban_traffic, dtype=jnp.bool_),
        jnp.float64(3.6) * (ratio - jnp.float64(0.5)) + jnp.float64(1.0),
        jnp.zeros_like(ratio),
    )
    parameters = UrbanFluxesStaticParameters(
        roof_height_landunit=selected_surface["HT_ROOF"],
        roof_fraction_landunit=selected_surface["WTLUNIT_ROOF"],
        canyon_height_width_ratio_landunit=ratio,
        pervious_road_fraction_landunit=selected_surface["WTROAD_PERV"],
        canyon_wind_height_landunit=selected_surface["WIND_HGT_CANYON"],
        traffic_heat_factor_landunit=traffic,
    )

    number_of_urban_landunits = landunit.size
    special = jnp.full(
        (number_of_urban_landunits,),
        jnp.float64(CLM_SPECIAL_VALUE),
        dtype=jnp.float64,
    )
    friction_state = FrictionVelocityState(*(special for _ in FrictionVelocityState._fields))
    zero_surface = jnp.zeros(
        (number_of_urban_landunits, NUMBER_OF_URBAN_SURFACES), dtype=jnp.float64
    )
    state = UrbanFluxesRestartState(
        turbulence=UrbanTurbulenceState(
            canopy_air_temperature_landunit=_finite_selected(restart_values, "taf", landunit),
            canopy_air_specific_humidity_landunit=_finite_selected(restart_values, "qaf", landunit),
            friction_velocity=friction_state,
        ),
        internal_building_temperature_landunit=_finite_selected(
            restart_values, "t_building", landunit
        ),
        air_conditioning_flux_landunit=_finite_selected(restart_values, "URBAN_AC_L", landunit),
        heating_flux_landunit=_finite_selected(restart_values, "URBAN_HEAT_L", landunit),
        air_conditioning_flux_surface=zero_surface,
        heating_flux_surface=zero_surface,
    )
    if method == BUILDING_TEMPERATURE_METHOD_SIMPLE:
        raise ValueError(
            "a SIMPLE-building restart needs per-column HVAC state, which this "
            "PROG-case restart does not contain"
        )

    return UrbanFluxesRestartData(
        topology=topology,
        clock=load_restart_clock(restart),
        native_map=native_map,
        parameters=parameters,
        state=state,
        building_temperature_method=method,
        maximum_stable_zeta=maximum_stable_zeta,
        restart_path=restart,
        surface_data_path=surface_data,
    )


def _validate_dynamic_shapes(
    topology: SubgridTopology,
    grid_forcing: DatmForcing,
    temperature_inputs: CanopyTemperatureInputs,
    temperature_parameters: CanopyTemperatureParameters,
    temperature_state: CanopyTemperatureState,
    hydrology_state: CanopyHydrologyState,
) -> None:
    number_of_gridcells = topology.grid.longitude_degrees.shape[0]
    number_of_landunits = topology.landunit.gridcell.shape[0]
    number_of_columns = topology.column.gridcell.shape[0]
    number_of_patches = topology.patch.gridcell.shape[0]
    if any(jnp.shape(value) != (number_of_gridcells,) for value in grid_forcing):
        raise ValueError("every compact DATM forcing field must use the native gridcell axis")
    if temperature_inputs.soil_snow_temperature_column_level.shape != (
        number_of_columns,
        NUMBER_OF_TOTAL_LEVELS,
    ):
        raise ValueError("temperature inputs have the wrong native column/level shape")
    if temperature_state.column.ground_temperature_column.shape != (number_of_columns,):
        raise ValueError("CanopyTemperature state has the wrong column axis")
    if temperature_state.patch.forcing_height_wind_patch.shape != (number_of_patches,):
        raise ValueError("CanopyTemperature state has the wrong patch axis")
    if temperature_parameters.urban_roughness_length_landunit.shape != (number_of_landunits,):
        raise ValueError("CanopyTemperature parameters have the wrong landunit axis")
    if hydrology_state.snow.depth.shape != (number_of_columns,):
        raise ValueError("CanopyHydrology state has the wrong column axis")


def assemble_urban_fluxes_step_data(
    topology: SubgridTopology,
    native_map: UrbanFluxesNativeMap,
    parameters: UrbanFluxesStaticParameters,
    restart_state: UrbanFluxesRestartState,
    grid_forcing: DatmForcing,
    temperature_inputs: CanopyTemperatureInputs,
    temperature_parameters: CanopyTemperatureParameters,
    temperature_state: CanopyTemperatureState,
    hydrology_state: CanopyHydrologyState,
) -> UrbanFluxesStepData:

    _validate_dynamic_shapes(
        topology,
        grid_forcing,
        temperature_inputs,
        temperature_parameters,
        temperature_state,
        hydrology_state,
    )
    landunit = jnp.asarray(native_map.landunit_index, dtype=jnp.int32)
    gridcell = jnp.asarray(native_map.gridcell_index, dtype=jnp.int32)
    column = jnp.asarray(native_map.surface_column_index, dtype=jnp.int32)
    patch = jnp.asarray(native_map.surface_patch_index, dtype=jnp.int32)
    number_of_urban_landunits = landunit.shape[0]

    def gather_grid(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(value, dtype=jnp.float64)[gridcell]

    def gather_column(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(value, dtype=jnp.float64)[column]

    first_patch = patch[:, 0]
    top_ground_level = NUMBER_OF_SNOW_LEVELS
    inner_urban_level = NUMBER_OF_SNOW_LEVELS + NUMBER_OF_URBAN_LEVELS - 1
    column_state = temperature_state.column
    patch_state = temperature_state.patch

    turbulence_inputs = UrbanTurbulenceInputs(
        active_landunit=jnp.ones((number_of_urban_landunits,), dtype=jnp.bool_),
        eastward_wind_landunit=gather_grid(grid_forcing.wind_u),
        northward_wind_landunit=gather_grid(grid_forcing.wind_v),
        atmospheric_temperature_landunit=gather_grid(grid_forcing.temperature),
        atmospheric_potential_temperature_landunit=gather_grid(grid_forcing.potential_temperature),
        atmospheric_density_landunit=gather_grid(grid_forcing.density),
        atmospheric_specific_humidity_landunit=gather_grid(grid_forcing.specific_humidity),
        forcing_height_wind_landunit=jnp.asarray(
            patch_state.forcing_height_wind_patch, dtype=jnp.float64
        )[first_patch],
        forcing_height_temperature_landunit=jnp.asarray(
            patch_state.forcing_height_temperature_patch, dtype=jnp.float64
        )[first_patch],
        forcing_height_humidity_landunit=jnp.asarray(
            patch_state.forcing_height_humidity_patch, dtype=jnp.float64
        )[first_patch],
        momentum_roughness_length_landunit=jnp.asarray(
            temperature_parameters.urban_roughness_length_landunit,
            dtype=jnp.float64,
        )[landunit],
        displacement_height_landunit=jnp.asarray(
            temperature_parameters.urban_displacement_height_landunit,
            dtype=jnp.float64,
        )[landunit],
        roof_height_landunit=parameters.roof_height_landunit,
        roof_fraction_landunit=parameters.roof_fraction_landunit,
        canyon_height_width_ratio_landunit=(parameters.canyon_height_width_ratio_landunit),
        pervious_road_fraction_landunit=parameters.pervious_road_fraction_landunit,
        canyon_wind_height_landunit=parameters.canyon_wind_height_landunit,
        traffic_heat_factor_landunit=parameters.traffic_heat_factor_landunit,
        ground_temperature_surface=gather_column(column_state.ground_temperature_column),
        ground_specific_humidity_surface=gather_column(
            column_state.ground_specific_humidity_column
        ),
        snow_depth_surface=gather_column(hydrology_state.snow.depth),
        surface_liquid_water_surface=gather_column(hydrology_state.top_soil_liquid_column),
        surface_ice_water_surface=gather_column(
            temperature_inputs.soil_ice_water_column_level[:, top_ground_level]
        ),
        air_conditioning_flux_surface=restart_state.air_conditioning_flux_surface,
        heating_flux_surface=restart_state.heating_flux_surface,
        air_conditioning_flux_landunit=restart_state.air_conditioning_flux_landunit,
        heating_flux_landunit=restart_state.heating_flux_landunit,
    )

    pervious_column = column[:, NUMBER_OF_URBAN_SURFACES - 1]
    road_root_soil = jnp.asarray(
        column_state.pervious_road_root_resistance_column,
        dtype=jnp.float64,
    )[pervious_column]
    road_root = jnp.pad(
        road_root_soil,
        ((0, 0), (0, NUMBER_OF_GROUND_LEVELS - NUMBER_OF_SOIL_LEVELS)),
    )
    surface_inputs = UrbanSurfaceFluxInputs(
        atmospheric_pressure_landunit=gather_grid(grid_forcing.pressure),
        ground_specific_humidity_temperature_derivative_surface=gather_column(
            column_state.ground_specific_humidity_temperature_derivative_column
        ),
        latent_heat_surface=gather_column(column_state.latent_heat_column),
        snow_fraction_surface=gather_column(hydrology_state.snow.snow_fraction),
        soil_evaporation_alpha_surface=gather_column(
            column_state.urban_soil_evaporation_alpha_column
        ),
        inner_layer_temperature_surface=gather_column(
            temperature_inputs.soil_snow_temperature_column_level[:, inner_urban_level]
        ),
        pervious_road_root_fraction_landunit_level=road_root,
        internal_building_temperature_landunit=(
            restart_state.internal_building_temperature_landunit
        ),
    )
    return UrbanFluxesStepData(
        turbulence_inputs=turbulence_inputs,
        turbulence_state=restart_state.turbulence,
        surface_inputs=surface_inputs,
    )
