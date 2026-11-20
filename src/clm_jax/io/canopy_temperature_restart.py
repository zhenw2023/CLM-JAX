
from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE, VON_KARMAN
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    COLUMN_URBAN_IMPERVIOUS_ROAD,
    COLUMN_URBAN_PERVIOUS_ROAD,
    COLUMN_URBAN_ROOF,
    COLUMN_URBAN_SHADE_WALL,
    COLUMN_URBAN_SUN_WALL,
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
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
    CanopyTemperatureColumnState,
    CanopyTemperatureInputs,
    CanopyTemperatureParameters,
    CanopyTemperaturePatchState,
    CanopyTemperatureState,
)
from clm_jax.physics.canopy_temperature_driver import (
    synchronize_canopy_temperature_water_state as _synchronize_water_state,
)


class CanopyTemperatureRestartData(NamedTuple):

    topology: SubgridTopology
    clock: RestartClock
    inputs: CanopyTemperatureInputs
    parameters: CanopyTemperatureParameters
    state: CanopyTemperatureState
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path


class ColumnSoilHydraulicProperties(NamedTuple):

    saturated_water_content_column_level: np.ndarray
    clapp_hornberger_b_column_level: np.ndarray
    saturated_suction_column_level: np.ndarray
    dry_water_content_column_level: np.ndarray
    optimal_water_content_column_level: np.ndarray
    saturated_hydraulic_conductivity_column_level: np.ndarray
    field_capacity_water_content_column_level: np.ndarray
    dry_soil_bulk_density_column_level: np.ndarray
    organic_matter_density_column_level: np.ndarray


class ColumnSoilThermalProperties(NamedTuple):

    mineral_thermal_conductivity_column_level: np.ndarray
    dry_thermal_conductivity_column_level: np.ndarray
    solid_volumetric_heat_capacity_column_level: np.ndarray


class LakeSoilThermalProperties(NamedTuple):

    saturated_water_content: np.ndarray
    saturated_thermal_conductivity: np.ndarray
    mineral_thermal_conductivity: np.ndarray
    dry_thermal_conductivity: np.ndarray
    solid_volumetric_heat_capacity: np.ndarray


def synchronize_canopy_temperature_water_state(
    inputs: CanopyTemperatureInputs,
    hydrology: CanopyHydrologyState,
) -> CanopyTemperatureInputs:

    return _synchronize_water_state(inputs, hydrology)


_RESTART_VARIABLES = (
    "SNLSNO",
    "T_SOISNO",
    "TH2OSFC",
    "H2OSOI_LIQ",
    "H2OSOI_ICE",
    "FH2OSFC",
    "frac_sno_eff",
    "frac_sno",
    "FRAC_VEG_NOSNO_ALB",
    "elai",
    "esai",
    "htop",
    "DSL",
    "SOILRESIS",
    "Z0MG",
    "T_GRND",
)

_SURFACE_VARIABLES = (
    "PCT_SAND",
    "PCT_CLAY",
    "ORGANIC",
    "zbedrock",
    "CANYON_HWR",
    "HT_ROOF",
    "EM_ROOF",
    "EM_WALL",
    "EM_IMPROAD",
    "EM_PERROAD",
)

_PARAMETER_VARIABLES = ("organic_max", "z0mr", "displar")


def _configured_input_path(case: CimeCase, key: str) -> Path:
    group = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    declared = group.get(key)
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(f"clm_inparm in {case.case_docs / 'lnd_in'} lacks {key}")
    path = Path(case.expand(declared)).expanduser()
    if not path.is_absolute():
        path = case.root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM {key} does not exist: {path}")
    return path


def _selected_input_path(
    case: CimeCase,
    key: str,
    override: str | Path | None,
) -> Path:
    path = (
        _configured_input_path(case, key)
        if override is None
        else Path(override).expanduser().resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(f"CLM {key} inputdata does not exist: {path}")
    return path


def _finite_restart_float(value: jnp.ndarray) -> jnp.ndarray:
    value = jnp.asarray(value, dtype=jnp.float64)
    return jnp.where(jnp.isnan(value), jnp.float64(CLM_SPECIAL_VALUE), value)


def soil_layer_geometry_20sl_8p5m() -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    thickness = np.empty(NUMBER_OF_GROUND_LEVELS, dtype=np.float64)
    for level in range(1, 5):
        thickness[level - 1] = np.float64(level) * np.float64(0.02)
    for level in range(5, 14):
        thickness[level - 1] = thickness[3] + np.float64(level - 4) * np.float64(0.04)
    for level in range(14, NUMBER_OF_SOIL_LEVELS + 1):
        thickness[level - 1] = thickness[12] + np.float64(level - 13) * np.float64(0.10)
    for level in range(NUMBER_OF_SOIL_LEVELS + 1, NUMBER_OF_GROUND_LEVELS + 1):
        bedrock_increment = (
            (np.float64(level - NUMBER_OF_SOIL_LEVELS) * np.float64(25.0)) ** np.float64(1.5)
        ) / np.float64(100.0)
        thickness[level - 1] = thickness[NUMBER_OF_SOIL_LEVELS - 1] + bedrock_increment

    interface = np.empty(NUMBER_OF_GROUND_LEVELS + 1, dtype=np.float64)
    interface[0] = np.float64(0.0)
    running = np.float64(0.0)
    for level in range(NUMBER_OF_GROUND_LEVELS):
        running = running + thickness[level]
        interface[level + 1] = running
    node = np.empty(NUMBER_OF_GROUND_LEVELS, dtype=np.float64)
    for level in range(NUMBER_OF_GROUND_LEVELS):
        node[level] = np.float64(0.5) * (interface[level] + interface[level + 1])
    return node, interface, thickness


def _original_soil_interfaces() -> np.ndarray:
    number_of_input_levels = 10
    node = np.empty(number_of_input_levels, dtype=np.float64)
    for level in range(1, number_of_input_levels + 1):
        node[level - 1] = np.float64(0.025) * (
            np.exp(np.float64(0.5) * (np.float64(level) - np.float64(0.5))) - np.float64(1.0)
        )
    thickness = np.empty(number_of_input_levels, dtype=np.float64)
    thickness[0] = np.float64(0.5) * (node[0] + node[1])
    for level in range(1, number_of_input_levels - 1):
        thickness[level] = np.float64(0.5) * (node[level + 1] - node[level - 1])
    thickness[-1] = node[-1] - node[-2]
    interface = np.empty(number_of_input_levels + 1, dtype=np.float64)
    interface[0] = np.float64(0.0)
    for level in range(1, number_of_input_levels):
        interface[level] = np.float64(0.5) * (node[level - 1] + node[level])
    interface[-1] = node[-1] + np.float64(0.5) * thickness[-1]
    return interface


def _texture_level_indices(interface: np.ndarray) -> np.ndarray:

    original_interface = _original_soil_interfaces()
    indices = np.empty(NUMBER_OF_GROUND_LEVELS, dtype=np.int64)
    indices[0] = 0
    for level in range(1, NUMBER_OF_SOIL_LEVELS):
        indices[level] = np.searchsorted(
            original_interface[1:-1],
            interface[level + 1],
            side="right",
        )
    indices[NUMBER_OF_SOIL_LEVELS:] = 9
    return indices


def _gridcell_surface_values(
    topology: SubgridTopology,
    source: jnp.ndarray,
) -> np.ndarray:
    values = np.asarray(source, dtype=np.float64)
    longitude = np.asarray(topology.grid.longitude_index, dtype=np.int64)
    latitude = np.asarray(topology.grid.latitude_index, dtype=np.int64)
    if values.ndim == 2:
        return values[latitude, longitude]
    if values.ndim == 3:
        return values[:, latitude, longitude]
    raise ValueError("surface field must have dimensions (lat, lon) or (level, lat, lon)")


_SCALAR_PARAMETER_POWER = np.frompyfunc(math.pow, 2, 1)


def _native_parameter_power(base, exponent):
    return np.asarray(_SCALAR_PARAMETER_POWER(base, exponent), dtype=np.float64)


def _column_soil_parameters(
    topology: SubgridTopology,
    surface_values: dict[str, jnp.ndarray],
    organic_maximum: float,
    node: np.ndarray,
    interface: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sand_grid = _gridcell_surface_values(topology, surface_values["PCT_SAND"])
    clay_grid = _gridcell_surface_values(topology, surface_values["PCT_CLAY"])
    organic_grid = _gridcell_surface_values(topology, surface_values["ORGANIC"])
    if sand_grid.shape[0] != 10 or clay_grid.shape != sand_grid.shape:
        raise ValueError("fsurdat soil texture must contain ten original CLM soil levels")
    if organic_grid.shape != sand_grid.shape:
        raise ValueError("fsurdat ORGANIC and soil texture shapes disagree")

    zero_surface = sand_grid[0] + clay_grid[0] == np.float64(0.0)
    inconsistent_zero = zero_surface & np.any(sand_grid + clay_grid != np.float64(0.0), axis=0)
    if np.any(inconsistent_zero):
        raise ValueError("soil texture has zero surface but nonzero deeper levels")
    sand_grid[:, zero_surface] = np.float64(1.0)
    clay_grid[:, zero_surface] = np.float64(1.0)
    if np.any(sand_grid + clay_grid == np.float64(0.0)):
        raise ValueError("soil texture contains a zero sand-plus-clay level")

    level_index = _texture_level_indices(interface)
    column_gridcell = np.asarray(topology.column.gridcell, dtype=np.int64)
    sand = sand_grid[level_index][:, column_gridcell].T
    clay = clay_grid[level_index][:, column_gridcell].T
    organic = organic_grid[level_index][:, column_gridcell].T
    organic_fraction = organic / np.float64(organic_maximum)
    organic_fraction[:, NUMBER_OF_SOIL_LEVELS:] = np.float64(0.0)

    landunit_type = np.asarray(topology.column.landunit_type, dtype=np.int64)
    column_type = np.asarray(topology.column.column_type, dtype=np.int64)
    urban = (landunit_type >= LANDUNIT_URBAN_MIN) & (landunit_type <= LANDUNIT_URBAN_MAX)
    organic_fraction[urban] = np.float64(0.0)

    mineral_watsat = np.float64(0.489) - np.float64(0.00126) * sand
    mineral_b = np.float64(2.91) + np.float64(0.159) * clay
    mineral_suction = np.float64(10.0) * _native_parameter_power(
        np.float64(10.0), np.float64(1.88) - np.float64(0.0131) * sand,
    )
    organic_watsat = np.maximum(
        np.float64(0.93) - np.float64(0.1) * (node / np.float64(0.5)),
        np.float64(0.83),
    )[None, :]
    organic_b = np.minimum(
        np.float64(2.7) + np.float64(9.3) * (node / np.float64(0.5)),
        np.float64(12.0),
    )[None, :]
    organic_suction = np.minimum(
        np.float64(10.3) - np.float64(0.2) * (node / np.float64(0.5)),
        np.float64(10.1),
    )[None, :]
    mineral_fraction = np.float64(1.0) - organic_fraction
    watsat = mineral_fraction * mineral_watsat + organic_watsat * organic_fraction
    bsw = mineral_fraction * mineral_b + organic_fraction * organic_b
    suction = mineral_fraction * mineral_suction + organic_suction * organic_fraction
    dry = watsat * _native_parameter_power(
        np.float64(316230.0) / suction, -np.float64(1.0) / bsw,
    )
    optimal = watsat * _native_parameter_power(
        np.float64(158490.0) / suction, -np.float64(1.0) / bsw,
    )

    default_real_xksat_scale = np.float64(np.float32(0.0070556))
    default_real_xksat_intercept = np.float64(np.float32(-0.884))
    default_real_xksat_sand_slope = np.float64(np.float32(0.0153))
    mineral_hksat = default_real_xksat_scale * _native_parameter_power(
        np.float64(10.0), default_real_xksat_intercept + default_real_xksat_sand_slope * sand,
    )
    organic_hksat = np.maximum(
        np.float64(0.28) - np.float64(0.2799) * (node[None, :] / np.float64(0.5)),
        mineral_hksat,
    )
    percolation_threshold = np.float64(0.5)
    percolation_exponent = np.float64(0.139)
    percolation_fraction = np.zeros_like(organic_fraction)
    percolating = organic_fraction > percolation_threshold
    percolation_normalization = _native_parameter_power(
        np.float64(1.0) - percolation_threshold, -percolation_exponent,
    )
    percolation_fraction[percolating] = (
        percolation_normalization
        * _native_parameter_power(
            organic_fraction[percolating] - percolation_threshold, percolation_exponent,
        )
    )
    unconsolidated_fraction = (np.float64(1.0) - organic_fraction) + (
        np.float64(1.0) - percolation_fraction
    ) * organic_fraction
    unconsolidated_hksat = np.zeros_like(organic_fraction)
    mixed = organic_fraction < np.float64(1.0)
    unconsolidated_hksat[mixed] = unconsolidated_fraction[mixed] / (
        (np.float64(1.0) - organic_fraction[mixed]) / mineral_hksat[mixed]
        + ((np.float64(1.0) - percolation_fraction[mixed]) * organic_fraction[mixed])
        / organic_hksat[mixed]
    )
    hksat = (
        unconsolidated_fraction * unconsolidated_hksat
        + percolation_fraction * organic_fraction * organic_hksat
    )

    used = (
        (landunit_type == LANDUNIT_SOIL)
        | (landunit_type == LANDUNIT_CROP)
        | (
            urban
            & (
                (column_type == COLUMN_URBAN_PERVIOUS_ROAD)
                | (column_type == COLUMN_URBAN_IMPERVIOUS_ROAD)
            )
        )
    )
    special = np.float64(CLM_SPECIAL_VALUE)
    for value in (watsat, bsw, suction, dry, optimal, hksat):
        value[~used] = special
    dry[urban & (column_type == COLUMN_URBAN_IMPERVIOUS_ROAD)] = special
    optimal[urban & (column_type == COLUMN_URBAN_IMPERVIOUS_ROAD)] = special
    return watsat, bsw, suction, dry, optimal, hksat


def column_soil_hydraulic_properties_20sl_8p5m(
    topology: SubgridTopology,
    surface_values: dict[str, jnp.ndarray],
    organic_maximum: float,
) -> ColumnSoilHydraulicProperties:

    if not np.isfinite(organic_maximum) or organic_maximum <= 0.0:
        raise ValueError("organic_maximum must be finite and positive")
    node, interface, _ = soil_layer_geometry_20sl_8p5m()
    watsat, bsw, suction, dry, optimal, hksat = _column_soil_parameters(
        topology,
        surface_values,
        organic_maximum,
        node,
        interface,
    )

    sand_grid = _gridcell_surface_values(topology, surface_values["PCT_SAND"])
    clay_grid = _gridcell_surface_values(topology, surface_values["PCT_CLAY"])
    organic_grid = _gridcell_surface_values(topology, surface_values["ORGANIC"])
    zero_surface = sand_grid[0] + clay_grid[0] == np.float64(0.0)
    sand_grid[:, zero_surface] = np.float64(1.0)
    clay_grid[:, zero_surface] = np.float64(1.0)
    level_index = _texture_level_indices(interface)
    column_gridcell = np.asarray(topology.column.gridcell, dtype=np.int64)
    sand = sand_grid[level_index][:, column_gridcell].T
    organic = organic_grid[level_index][:, column_gridcell].T
    organic = (organic / np.float64(organic_maximum)) * np.float64(organic_maximum)
    organic[:, NUMBER_OF_SOIL_LEVELS:] = np.float64(0.0)
    mineral_watsat = np.float64(0.489) - np.float64(0.00126) * sand
    dry_bulk_density = (np.float64(1.0) - mineral_watsat) * np.float64(2.7e3)
    field_capacity = watsat * (np.float64(0.1) / (hksat * np.float64(86_400.0))) ** (
        np.float64(1.0) / (np.float64(2.0) * bsw + np.float64(3.0))
    )

    landunit_type = np.asarray(topology.column.landunit_type, dtype=np.int64)
    soil_or_crop = (landunit_type == LANDUNIT_SOIL) | (landunit_type == LANDUNIT_CROP)
    special = np.float64(CLM_SPECIAL_VALUE)
    for value in (field_capacity, dry_bulk_density, organic):
        value[~soil_or_crop] = special
    return ColumnSoilHydraulicProperties(
        saturated_water_content_column_level=watsat,
        clapp_hornberger_b_column_level=bsw,
        saturated_suction_column_level=suction,
        dry_water_content_column_level=dry,
        optimal_water_content_column_level=optimal,
        saturated_hydraulic_conductivity_column_level=hksat,
        field_capacity_water_content_column_level=field_capacity,
        dry_soil_bulk_density_column_level=dry_bulk_density,
        organic_matter_density_column_level=organic,
    )


def column_soil_thermal_properties_20sl_8p5m(
    topology: SubgridTopology,
    surface_values: dict[str, jnp.ndarray],
    organic_maximum: float,
) -> ColumnSoilThermalProperties:

    if not np.isfinite(organic_maximum) or organic_maximum <= 0.0:
        raise ValueError("organic_maximum must be finite and positive")
    node, interface, _ = soil_layer_geometry_20sl_8p5m()
    watsat, _, _, _, _, _ = _column_soil_parameters(
        topology,
        surface_values,
        organic_maximum,
        node,
        interface,
    )

    sand_grid = _gridcell_surface_values(topology, surface_values["PCT_SAND"])
    clay_grid = _gridcell_surface_values(topology, surface_values["PCT_CLAY"])
    organic_grid = _gridcell_surface_values(topology, surface_values["ORGANIC"])
    if sand_grid.shape != clay_grid.shape or sand_grid.shape != organic_grid.shape:
        raise ValueError("fsurdat soil texture and organic fields must agree")
    if sand_grid.shape[0] != 10:
        raise ValueError("fsurdat soil texture must contain ten source levels")
    zero_surface = sand_grid[0] + clay_grid[0] == np.float64(0.0)
    inconsistent_zero = zero_surface & np.any(sand_grid + clay_grid != np.float64(0.0), axis=0)
    if np.any(inconsistent_zero):
        raise ValueError("soil texture has zero surface but nonzero deeper levels")
    sand_grid[:, zero_surface] = np.float64(1.0)
    clay_grid[:, zero_surface] = np.float64(1.0)
    if np.any(sand_grid + clay_grid == np.float64(0.0)):
        raise ValueError("soil texture contains a zero sand-plus-clay level")

    source_level = _texture_level_indices(interface)
    column_gridcell = np.asarray(topology.column.gridcell, dtype=np.int64)
    sand = sand_grid[source_level][:, column_gridcell].T
    clay = clay_grid[source_level][:, column_gridcell].T
    organic_fraction = organic_grid[source_level][:, column_gridcell].T / np.float64(
        organic_maximum
    )
    organic_fraction[:, NUMBER_OF_SOIL_LEVELS:] = np.float64(0.0)

    landunit_type = np.asarray(topology.column.landunit_type, dtype=np.int64)
    column_type = np.asarray(topology.column.column_type, dtype=np.int64)
    urban = (landunit_type >= LANDUNIT_URBAN_MIN) & (landunit_type <= LANDUNIT_URBAN_MAX)
    organic_fraction[urban] = np.float64(0.0)
    mineral_fraction = np.float64(1.0) - organic_fraction
    used = (
        (landunit_type == LANDUNIT_SOIL)
        | (landunit_type == LANDUNIT_CROP)
        | (
            urban
            & (
                (column_type == COLUMN_URBAN_PERVIOUS_ROAD)
                | (column_type == COLUMN_URBAN_IMPERVIOUS_ROAD)
            )
        )
    )

    mineral_porosity = np.float64(0.489) - np.float64(0.00126) * sand
    dry_bulk_density = (np.float64(1.0) - mineral_porosity) * np.float64(2.7e3)
    texture_sum = sand + clay
    material_conductivity = (
        mineral_fraction * (np.float64(8.80) * sand + np.float64(2.92) * clay)
    ) / texture_sum + np.float64(0.25) * organic_fraction
    safe_porosity = np.where(used[:, None], watsat, np.float64(0.5))
    mineral_conductivity = _native_parameter_power(
        material_conductivity, np.float64(1.0) - safe_porosity,
    )
    dry_conductivity = (
        (np.float64(0.135) * dry_bulk_density + np.float64(64.7))
        / (np.float64(2.7e3) - np.float64(0.947) * dry_bulk_density)
    ) * mineral_fraction + np.float64(0.05) * organic_fraction
    solid_heat_capacity = (
        (
            mineral_fraction
            * (np.float64(2.128) * sand + np.float64(2.385) * clay)
        )
        / texture_sum
        + np.float64(2.5) * organic_fraction
    ) * np.float64(1.0e6)

    special = np.float64(CLM_SPECIAL_VALUE)
    for value in (mineral_conductivity, dry_conductivity, solid_heat_capacity):
        value[~used] = special
    return ColumnSoilThermalProperties(
        mineral_thermal_conductivity_column_level=mineral_conductivity,
        dry_thermal_conductivity_column_level=dry_conductivity,
        solid_volumetric_heat_capacity_column_level=solid_heat_capacity,
    )


def column_bedrock_layer_count_20sl_8p5m(
    topology: SubgridTopology,
    zbedrock: jnp.ndarray,
) -> np.ndarray:

    _, interface, _ = soil_layer_geometry_20sl_8p5m()
    bedrock_grid = _gridcell_surface_values(topology, zbedrock)
    minimum_level = 3
    for level in range(3, NUMBER_OF_SOIL_LEVELS + 1):
        if interface[level - 1] < np.float64(0.4) <= interface[level]:
            minimum_level = level
    result_grid = np.full(
        topology.grid.longitude_degrees.shape[0],
        NUMBER_OF_SOIL_LEVELS,
        dtype=np.int32,
    )
    for level in range(minimum_level, NUMBER_OF_SOIL_LEVELS + 1):
        selected = (interface[level - 1] < bedrock_grid) & (interface[level] >= bedrock_grid)
        result_grid[selected] = np.int32(level)
    return result_grid[np.asarray(topology.column.gridcell, dtype=np.int64)]


def lake_soil_thermal_properties_20sl_8p5m(
    topology: SubgridTopology,
    surface_values: dict[str, jnp.ndarray],
    organic_maximum: float,
    column_index: np.ndarray,
) -> LakeSoilThermalProperties:

    if not np.isfinite(organic_maximum) or organic_maximum <= 0.0:
        raise ValueError("organic_maximum must be finite and positive")
    selected = np.asarray(column_index, dtype=np.int64)
    if selected.ndim != 1 or np.any(selected < 0):
        raise ValueError("lake column indices must be a nonnegative vector")
    column_count = topology.column.gridcell.shape[0]
    if np.any(selected >= column_count):
        raise ValueError("lake column index lies outside native topology")

    sand_grid = _gridcell_surface_values(topology, surface_values["PCT_SAND"])
    clay_grid = _gridcell_surface_values(topology, surface_values["PCT_CLAY"])
    organic_grid = _gridcell_surface_values(topology, surface_values["ORGANIC"])
    if sand_grid.shape != clay_grid.shape or sand_grid.shape != organic_grid.shape:
        raise ValueError("lake sediment texture fields must have matching shapes")
    if sand_grid.shape[0] != 10:
        raise ValueError("lake sediment texture must contain ten source levels")

    zero_surface = sand_grid[0] + clay_grid[0] == np.float64(0.0)
    inconsistent_zero = zero_surface & np.any(
        sand_grid + clay_grid != np.float64(0.0),
        axis=0,
    )
    if np.any(inconsistent_zero):
        raise ValueError("lake sediment has zero surface but nonzero deeper texture")
    sand_grid[:, zero_surface] = np.float64(1.0)
    clay_grid[:, zero_surface] = np.float64(1.0)

    _, interface, _ = soil_layer_geometry_20sl_8p5m()
    source_level = _texture_level_indices(interface)
    column_gridcell = np.asarray(topology.column.gridcell, dtype=np.int64)[selected]
    sand = sand_grid[source_level][:, column_gridcell].T
    clay = clay_grid[source_level][:, column_gridcell].T
    organic = organic_grid[source_level][:, column_gridcell].T
    organic_fraction = organic / np.float64(organic_maximum)
    organic_fraction[:, NUMBER_OF_SOIL_LEVELS:] = np.float64(0.0)
    mineral_fraction = np.float64(1.0) - organic_fraction

    mineral_porosity = np.float64(0.489) - np.float64(0.00126) * sand
    dry_bulk_density = (np.float64(1.0) - mineral_porosity) * np.float64(2.7e3)
    porosity = mineral_fraction * mineral_porosity + np.float64(0.9) * organic_fraction
    texture_sum = sand + clay
    mineral_conductivity = (
        mineral_fraction * (np.float64(8.80) * sand + np.float64(2.92) * clay)
    ) / texture_sum + np.float64(0.25) * organic_fraction
    mineral_conductivity = _native_parameter_power(
        mineral_conductivity, np.float64(1.0) - porosity,
    )
    saturated_conductivity = mineral_conductivity * _native_parameter_power(
        np.float64(0.57), porosity,
    )
    dry_conductivity = (
        (np.float64(0.135) * dry_bulk_density + np.float64(64.7))
        / (np.float64(2.7e3) - np.float64(0.947) * dry_bulk_density)
    ) * mineral_fraction + np.float64(0.05) * organic_fraction
    solid_heat_capacity = (
        (
            mineral_fraction
            * (np.float64(2.128) * sand + np.float64(2.385) * clay)
        )
        / texture_sum
        + np.float64(2.5) * organic_fraction
    ) * np.float64(1.0e6)
    outputs = (
        porosity,
        saturated_conductivity,
        mineral_conductivity,
        dry_conductivity,
        solid_heat_capacity,
    )
    if any(np.any(~np.isfinite(value)) for value in outputs):
        raise ValueError("active lake sediment properties contain non-finite values")
    return LakeSoilThermalProperties(*outputs)


def _bedrock_root_fraction(
    topology: SubgridTopology,
    surface_values: dict[str, jnp.ndarray],
    interface: np.ndarray,
) -> np.ndarray:
    bedrock_grid = _gridcell_surface_values(topology, surface_values["zbedrock"])
    minimum_bedrock_level = 3
    for level in range(3, NUMBER_OF_SOIL_LEVELS + 1):
        if interface[level - 1] < np.float64(0.4) and interface[level] >= np.float64(0.4):
            minimum_bedrock_level = level
    bedrock_level_grid = np.full(
        topology.grid.longitude_degrees.shape[0],
        NUMBER_OF_SOIL_LEVELS,
        dtype=np.int32,
    )
    for level in range(minimum_bedrock_level, NUMBER_OF_SOIL_LEVELS + 1):
        selected = (interface[level - 1] < bedrock_grid) & (interface[level] >= bedrock_grid)
        bedrock_level_grid[selected] = level

    number_of_columns = topology.column.gridcell.shape[0]
    root = np.zeros((number_of_columns, NUMBER_OF_SOIL_LEVELS), dtype=np.float64)
    landunit_type = np.asarray(topology.column.landunit_type, dtype=np.int64)
    column_type = np.asarray(topology.column.column_type, dtype=np.int64)
    pervious = (
        (landunit_type >= LANDUNIT_URBAN_MIN)
        & (landunit_type <= LANDUNIT_URBAN_MAX)
        & (column_type == COLUMN_URBAN_PERVIOUS_ROAD)
    )
    column_bedrock = bedrock_level_grid[np.asarray(topology.column.gridcell, dtype=np.int64)]
    initial_fraction = np.float64(1.0) / np.float64(NUMBER_OF_SOIL_LEVELS)
    for column in np.flatnonzero(pervious):
        number_of_root_layers = int(column_bedrock[column])
        root[column, :] = initial_fraction
        below_bedrock = np.float64(0.0)
        for level in range(number_of_root_layers, NUMBER_OF_SOIL_LEVELS):
            below_bedrock += root[column, level]
        redistribution = below_bedrock / np.float64(number_of_root_layers)
        for level in range(number_of_root_layers):
            root[column, level] += redistribution
        root[column, number_of_root_layers:] = np.float64(0.0)
    return root


def _urban_landunit_values(
    topology: SubgridTopology,
    source: jnp.ndarray,
) -> np.ndarray:
    source_values = np.asarray(source, dtype=np.float64)
    if source_values.ndim != 3 or source_values.shape[0] != 3:
        raise ValueError("urban fsurdat fields must have dimensions (numurbl, lat, lon)")
    number_of_landunits = topology.landunit.gridcell.shape[0]
    result = np.full(number_of_landunits, CLM_SPECIAL_VALUE, dtype=np.float64)
    landunit_type = np.asarray(topology.landunit.landunit_type, dtype=np.int64)
    active = np.asarray(topology.landunit.active, dtype=bool)
    urban = active & (landunit_type >= LANDUNIT_URBAN_MIN) & (landunit_type <= LANDUNIT_URBAN_MAX)
    landunit_gridcell = np.asarray(topology.landunit.gridcell, dtype=np.int64)
    gridcell = landunit_gridcell[urban]
    latitude = np.asarray(topology.grid.latitude_index, dtype=np.int64)[gridcell]
    longitude = np.asarray(topology.grid.longitude_index, dtype=np.int64)[gridcell]
    density = landunit_type[urban] - LANDUNIT_URBAN_MIN
    result[urban] = source_values[density, latitude, longitude]
    if not np.all(np.isfinite(result[urban])):
        raise ValueError("an active urban landunit has missing fsurdat values")
    return result


def _urban_aerodynamic_parameters(
    topology: SubgridTopology,
    surface_values: dict[str, jnp.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    canyon_ratio = _urban_landunit_values(topology, surface_values["CANYON_HWR"])
    roof_height = _urban_landunit_values(topology, surface_values["HT_ROOF"])
    number_of_landunits = topology.landunit.gridcell.shape[0]
    roughness = np.full(number_of_landunits, CLM_SPECIAL_VALUE, dtype=np.float64)
    displacement = np.full(number_of_landunits, CLM_SPECIAL_VALUE, dtype=np.float64)
    landunit_type = np.asarray(topology.landunit.landunit_type, dtype=np.int64)
    active = np.asarray(topology.landunit.active, dtype=bool)
    urban = active & (landunit_type >= LANDUNIT_URBAN_MIN) & (landunit_type <= LANDUNIT_URBAN_MAX)
    ratio = canyon_ratio[urban]
    height = roof_height[urban]
    plan_area = ratio / (ratio + np.float64(1.0))
    building_length_width_ratio = plan_area
    frontal_area = (np.float64(1.0) - plan_area) * ratio
    frontal_area = (
        frontal_area * np.sqrt(np.float64(1.0) / building_length_width_ratio) * np.sqrt(plan_area)
    )
    import math
    alpha_power = np.fromiter((math.pow(4.43, -float(v)) for v in plan_area),
                              dtype=np.float64, count=plan_area.size)
    displacement_urban = (
        np.float64(1.0) + alpha_power * (plan_area - np.float64(1.0))
    ) * height
    remaining_height_fraction = np.float64(1.0) - displacement_urban / height
    exponent_base = (
        np.float64(0.5) * np.float64(1.0) * np.float64(1.2)
        / np.float64(VON_KARMAN) ** np.float64(2.0)
        * remaining_height_fraction * frontal_area
    )
    attenuation = np.fromiter(
        (math.exp(-math.pow(float(v), -0.5)) for v in exponent_base),
        dtype=np.float64, count=exponent_base.size,
    )
    roughness_urban = height * remaining_height_fraction * attenuation
    displacement[urban] = displacement_urban
    roughness[urban] = roughness_urban
    return roughness, displacement


def _urban_ground_emissivity(
    topology: SubgridTopology,
    surface_values: dict[str, jnp.ndarray],
) -> np.ndarray:
    by_surface = {
        COLUMN_URBAN_ROOF: _urban_landunit_values(topology, surface_values["EM_ROOF"]),
        COLUMN_URBAN_SUN_WALL: _urban_landunit_values(topology, surface_values["EM_WALL"]),
        COLUMN_URBAN_SHADE_WALL: _urban_landunit_values(topology, surface_values["EM_WALL"]),
        COLUMN_URBAN_IMPERVIOUS_ROAD: _urban_landunit_values(
            topology, surface_values["EM_IMPROAD"]
        ),
        COLUMN_URBAN_PERVIOUS_ROAD: _urban_landunit_values(topology, surface_values["EM_PERROAD"]),
    }
    number_of_columns = topology.column.gridcell.shape[0]
    emissivity = np.full(number_of_columns, CLM_SPECIAL_VALUE, dtype=np.float64)
    landunit = np.asarray(topology.column.landunit, dtype=np.int64)
    landunit_type = np.asarray(topology.column.landunit_type, dtype=np.int64)
    column_type = np.asarray(topology.column.column_type, dtype=np.int64)
    active = np.asarray(topology.column.active, dtype=bool)
    urban = active & (landunit_type >= LANDUNIT_URBAN_MIN) & (landunit_type <= LANDUNIT_URBAN_MAX)
    for surface_type, landunit_value in by_surface.items():
        selected = urban & (column_type == surface_type)
        emissivity[selected] = landunit_value[landunit[selected]]
    if not np.all(np.isfinite(emissivity[urban])):
        raise ValueError("an active urban column has missing ground emissivity")
    return emissivity


def _initial_state(
    topology: SubgridTopology,
    restart_values: dict[str, jnp.ndarray],
    urban_emissivity: np.ndarray,
) -> CanopyTemperatureState:
    number_of_columns = topology.column.gridcell.shape[0]
    number_of_patches = topology.patch.gridcell.shape[0]
    special_column = jnp.full(
        (number_of_columns,), jnp.float64(CLM_SPECIAL_VALUE), dtype=jnp.float64
    )
    special_column_level = jnp.full(
        (number_of_columns, NUMBER_OF_TOTAL_LEVELS),
        jnp.float64(CLM_SPECIAL_VALUE),
        dtype=jnp.float64,
    )
    special_column_soil = jnp.full(
        (number_of_columns, NUMBER_OF_SOIL_LEVELS),
        jnp.float64(CLM_SPECIAL_VALUE),
        dtype=jnp.float64,
    )
    special_patch = jnp.full(
        (number_of_patches,), jnp.float64(CLM_SPECIAL_VALUE), dtype=jnp.float64
    )
    ground_roughness = _finite_restart_float(restart_values["Z0MG"])
    column = CanopyTemperatureColumnState(
        temperature_before_column_level=special_column_level,
        surface_water_temperature_before_column=special_column,
        dry_surface_layer_thickness_column=_finite_restart_float(restart_values["DSL"]),
        soil_evaporation_resistance_column=_finite_restart_float(restart_values["SOILRESIS"]),
        ground_temperature_column=_finite_restart_float(restart_values["T_GRND"]),
        snow_surface_specific_humidity_column=special_column,
        soil_surface_specific_humidity_column=special_column,
        surface_water_specific_humidity_column=special_column,
        ground_specific_humidity_column=special_column,
        ground_specific_humidity_temperature_derivative_column=special_column,
        soil_evaporation_alpha_column=special_column,
        urban_soil_evaporation_alpha_column=special_column,
        pervious_road_root_resistance_column=special_column_soil,
        latent_heat_column=special_column,
        ground_emissivity_column=jnp.asarray(urban_emissivity, dtype=jnp.float64),
        ground_momentum_roughness_column=ground_roughness,
        ground_sensible_roughness_column=ground_roughness,
        ground_latent_roughness_column=ground_roughness,
        convective_velocity_coefficient_column=special_column,
        convective_boundary_height_column=special_column,
        virtual_potential_temperature_column=special_column,
    )
    patch = CanopyTemperaturePatchState(
        *(special_patch for _ in CanopyTemperaturePatchState._fields)
    )
    return CanopyTemperatureState(column=column, patch=patch)


def _forcing_array(
    value: jnp.ndarray,
    expected_shape: tuple[int, ...],
    name: str,
) -> jnp.ndarray:
    result = jnp.asarray(value, dtype=jnp.float64)
    if result.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {result.shape}")
    return result


def load_canopy_temperature_restart(
    case: CimeCase,
    pressure_column: jnp.ndarray,
    specific_humidity_column: jnp.ndarray,
    temperature_column: jnp.ndarray,
    potential_temperature_column: jnp.ndarray,
    forcing_height_wind_gridcell: jnp.ndarray,
    forcing_height_temperature_gridcell: jnp.ndarray,
    forcing_height_humidity_gridcell: jnp.ndarray,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> CanopyTemperatureRestartData:

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    surface_data = _selected_input_path(case, "fsurdat", surface_data_path)
    parameter_data = _selected_input_path(case, "paramfile", parameter_data_path)

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    if general.get("soil_layerstruct") != "20SL_8.5m":
        raise ValueError("CanopyTemperature adapter currently requires soil_layerstruct=20SL_8.5m")
    if int(general.get("nlevsno", -1)) != NUMBER_OF_SNOW_LEVELS:
        raise ValueError(f"CanopyTemperature adapter requires nlevsno={NUMBER_OF_SNOW_LEVELS}")
    if not bool(general.get("use_bedrock", False)):
        raise ValueError("active CanopyTemperature adapter requires use_bedrock=.true.")
    soil_resistance = read_case_namelist_group(case, "lnd_in", "soil_resis_inparm")
    if int(soil_resistance.get("soil_resis_method", -1)) != 1:
        raise ValueError("active CanopyTemperature port requires soil_resis_method=1")

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
    topology = load_subgrid_topology(restart)
    number_of_gridcells = topology.grid.longitude_degrees.shape[0]
    number_of_columns = topology.column.gridcell.shape[0]
    number_of_patches = topology.patch.gridcell.shape[0]
    column_shape = (number_of_columns,)
    gridcell_shape = (number_of_gridcells,)

    pressure = _forcing_array(pressure_column, column_shape, "pressure_column")
    humidity = _forcing_array(specific_humidity_column, column_shape, "specific_humidity_column")
    temperature = _forcing_array(temperature_column, column_shape, "temperature_column")
    potential_temperature = _forcing_array(
        potential_temperature_column,
        column_shape,
        "potential_temperature_column",
    )
    wind_height = _forcing_array(
        forcing_height_wind_gridcell,
        gridcell_shape,
        "forcing_height_wind_gridcell",
    )
    temperature_height = _forcing_array(
        forcing_height_temperature_gridcell,
        gridcell_shape,
        "forcing_height_temperature_gridcell",
    )
    humidity_height = _forcing_array(
        forcing_height_humidity_gridcell,
        gridcell_shape,
        "forcing_height_humidity_gridcell",
    )

    restart_values = load_native_variables(restart, list(_RESTART_VARIABLES))
    surface_values = load_native_variables(surface_data, list(_SURFACE_VARIABLES))
    parameter_values = load_native_variables(parameter_data, list(_PARAMETER_VARIABLES))
    organic_maximum = float(np.asarray(parameter_values["organic_max"]).item())
    if not np.isfinite(organic_maximum) or organic_maximum <= 0.0:
        raise ValueError("parameter organic_max must be a finite positive scalar")

    node, interface, thickness = soil_layer_geometry_20sl_8p5m()
    watsat, bsw, suction, dry, optimal, hksat = _column_soil_parameters(
        topology,
        surface_values,
        organic_maximum,
        node,
        interface,
    )
    root_fraction = _bedrock_root_fraction(topology, surface_values, interface)
    urban_roughness, urban_displacement = _urban_aerodynamic_parameters(topology, surface_values)
    urban_emissivity = _urban_ground_emissivity(topology, surface_values)

    vegetation_type = np.asarray(topology.patch.vegetation_type, dtype=np.int64)
    z0mr = np.asarray(parameter_values["z0mr"], dtype=np.float64)
    displar = np.asarray(parameter_values["displar"], dtype=np.float64)
    if z0mr.shape != displar.shape or z0mr.ndim != 1:
        raise ValueError("parameter z0mr/displar must be matching one-dimensional arrays")
    if np.any((vegetation_type < 0) | (vegetation_type >= z0mr.size)):
        raise ValueError("restart PFT type lies outside parameter z0mr/displar arrays")
    layer_thickness = np.broadcast_to(
        thickness[None, :], (number_of_columns, NUMBER_OF_GROUND_LEVELS)
    )

    for name in (
        "SNLSNO",
        "TH2OSFC",
        "FH2OSFC",
        "frac_sno_eff",
        "frac_sno",
        "DSL",
        "SOILRESIS",
        "Z0MG",
        "T_GRND",
    ):
        if restart_values[name].shape != column_shape:
            raise ValueError(f"restart variable {name} has unexpected shape")
    for name in ("T_SOISNO", "H2OSOI_LIQ", "H2OSOI_ICE"):
        if restart_values[name].shape != (number_of_columns, NUMBER_OF_TOTAL_LEVELS):
            raise ValueError(f"restart variable {name} has unexpected shape")
    for name in ("FRAC_VEG_NOSNO_ALB", "elai", "esai", "htop"):
        if restart_values[name].shape != (number_of_patches,):
            raise ValueError(f"restart variable {name} has unexpected shape")

    inputs = CanopyTemperatureInputs(
        forcing_height_wind_gridcell=wind_height,
        forcing_height_temperature_gridcell=temperature_height,
        forcing_height_humidity_gridcell=humidity_height,
        pressure_column=pressure,
        specific_humidity_column=humidity,
        temperature_column=temperature,
        potential_temperature_column=potential_temperature,
        snow_layer_count_column=jnp.asarray(restart_values["SNLSNO"], dtype=jnp.int32),
        soil_snow_temperature_column_level=_finite_restart_float(restart_values["T_SOISNO"]),
        surface_water_temperature_column=_finite_restart_float(restart_values["TH2OSFC"]),
        surface_water_fraction_column=_finite_restart_float(restart_values["FH2OSFC"]),
        effective_snow_fraction_column=_finite_restart_float(restart_values["frac_sno_eff"]),
        snow_fraction_column=_finite_restart_float(restart_values["frac_sno"]),
        soil_liquid_water_column_level=_finite_restart_float(restart_values["H2OSOI_LIQ"]),
        soil_ice_water_column_level=_finite_restart_float(restart_values["H2OSOI_ICE"]),
        vegetation_exposed_patch=jnp.asarray(restart_values["FRAC_VEG_NOSNO_ALB"], dtype=jnp.int32),
        exposed_leaf_area_index_patch=_finite_restart_float(restart_values["elai"]),
        exposed_stem_area_index_patch=_finite_restart_float(restart_values["esai"]),
        canopy_top_height_patch=_finite_restart_float(restart_values["htop"]),
    )
    parameters = CanopyTemperatureParameters(
        ground_layer_thickness_column=jnp.asarray(layer_thickness),
        saturated_water_content_column=jnp.asarray(watsat),
        clapp_hornberger_b_column=jnp.asarray(bsw),
        saturated_suction_column=jnp.asarray(suction),
        dry_water_content_column=jnp.asarray(dry),
        optimal_water_content_column=jnp.asarray(optimal),
        saturated_hydraulic_conductivity_column=jnp.asarray(hksat),
        pervious_road_root_fraction_column=jnp.asarray(root_fraction),
        momentum_roughness_ratio_patch=jnp.asarray(z0mr[vegetation_type]),
        displacement_ratio_patch=jnp.asarray(displar[vegetation_type]),
        urban_roughness_length_landunit=jnp.asarray(urban_roughness),
        urban_displacement_height_landunit=jnp.asarray(urban_displacement),
    )
    state = _initial_state(topology, restart_values, urban_emissivity)
    return CanopyTemperatureRestartData(
        topology=topology,
        clock=load_restart_clock(restart),
        inputs=inputs,
        parameters=parameters,
        state=state,
        restart_path=restart,
        surface_data_path=surface_data,
        parameter_data_path=parameter_data,
    )
