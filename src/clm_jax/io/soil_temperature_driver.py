
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.forcing.urban import interpolate_urban_time_varying
from clm_jax.io.canopy_restart import load_canopy_hydrology_restart
from clm_jax.io.canopy_temperature_driver import (
    load_canopy_temperature_driver_native_data,
)
from clm_jax.io.canopy_temperature_restart import (
    column_bedrock_layer_count_20sl_8p5m,
    column_soil_hydraulic_properties_20sl_8p5m,
    column_soil_thermal_properties_20sl_8p5m,
    soil_layer_geometry_20sl_8p5m,
)
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.radiation_driver import assert_same_topology
from clm_jax.io.restart import (
    COLUMN_URBAN_ROOF,
    COLUMN_URBAN_SHADE_WALL,
    COLUMN_URBAN_SUN_WALL,
    LANDUNIT_LAKE,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    SubgridTopology,
)
from clm_jax.io.urban_flux_driver import load_urban_flux_driver_native_data
from clm_jax.io.urban_time_varying import load_urban_time_varying_native_data
from clm_jax.physics.canopy_temperature import (
    NUMBER_OF_GROUND_LEVELS,
    NUMBER_OF_SNOW_LEVELS,
    NUMBER_OF_TOTAL_LEVELS,
    NUMBER_OF_URBAN_LEVELS,
)
from clm_jax.physics.soil_temperature import (
    NUMBER_OF_TEMPERATURE_MATRIX_BANDS,
    NUMBER_OF_TEMPERATURE_SOLVE_LEVELS,
    GroundHeatFluxInputs,
    GroundHeatFluxPatchState,
    SoilHeatDiffusionInputs,
    SoilSnowPhaseChangeInputs,
    SoilTemperatureFinalDiagnosticsInputs,
    SoilTemperatureLinearSolveInputs,
    SoilTemperatureMatrixInputs,
    SoilTemperaturePostSolveFluxInputs,
    SoilTemperatureRHSInputs,
    SoilTemperatureSolutionUpdateInputs,
    SoilThermalPropertyInputs,
    StandingWaterPhaseChangeInputs,
)
from clm_jax.physics.soil_temperature_driver import SoilTemperatureDriverInputs
from clm_jax.physics.soil_temperature_step import (
    SoilTemperatureActiveInputs,
    make_urban_construction_column_map,
)
from clm_jax.physics.urban_building_temperature import UrbanBuildingTemperatureInputs


class SoilTemperatureDriverNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: SoilTemperatureDriverInputs
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path
    urban_stream_path: Path


_SURFACE_VARIABLES = (
    "PCT_SAND",
    "PCT_CLAY",
    "ORGANIC",
    "zbedrock",
    "TK_ROOF",
    "TK_WALL",
    "TK_IMPROAD",
    "CV_ROOF",
    "CV_WALL",
    "CV_IMPROAD",
    "NLEV_IMPROAD",
    "T_BUILDING_MIN",
    "THICK_ROOF",
    "THICK_WALL",
)

_RESTART_VARIABLES = (
    "t_roof_inner",
    "t_sunw_inner",
    "t_shdw_inner",
    "t_floor",
)


def _configured_clm_input(case: CimeCase, key: str) -> Path:
    group = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    declared = group.get(key)
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(f"configured clm_inparm does not define {key}")
    path = Path(case.expand(declared)).expanduser()
    if not path.is_absolute():
        path = case.root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM {key} input is absent: {path}")
    return path


def _finite_restart(value: jnp.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.where(np.isnan(array), np.float64(CLM_SPECIAL_VALUE), array)


def _scatter_compact(
    size: int,
    index: np.ndarray,
    values: jnp.ndarray,
    *,
    fill: float = CLM_SPECIAL_VALUE,
) -> np.ndarray:
    result = np.full(size, np.float64(fill), dtype=np.float64)
    result[index] = np.asarray(values, dtype=np.float64)
    return result


def _selected_urban_landunit_values(
    topology: SubgridTopology,
    source: jnp.ndarray,
) -> np.ndarray:

    values = np.asarray(source)
    if values.ndim not in (3, 4) or values.shape[-3] != 3:
        raise ValueError(
            "urban field must have (density,lat,lon) or (level,density,lat,lon) dimensions"
        )
    landunit_type = np.asarray(topology.landunit.landunit_type, dtype=np.int64)
    active = np.asarray(topology.landunit.active, dtype=bool)
    urban = active & (landunit_type >= LANDUNIT_URBAN_MIN) & (landunit_type <= LANDUNIT_URBAN_MAX)
    landunit = np.flatnonzero(urban)
    parent_gridcell = np.asarray(topology.landunit.gridcell, dtype=np.int64)[landunit]
    latitude = np.asarray(topology.grid.latitude_index, dtype=np.int64)[parent_gridcell]
    longitude = np.asarray(topology.grid.longitude_index, dtype=np.int64)[parent_gridcell]
    density = landunit_type[landunit] - LANDUNIT_URBAN_MIN
    if values.ndim == 3:
        result = np.full(landunit_type.shape, CLM_SPECIAL_VALUE, dtype=np.float64)
        result[landunit] = values[density, latitude, longitude]
    else:
        result = np.full(
            (landunit_type.size, values.shape[0]),
            CLM_SPECIAL_VALUE,
            dtype=np.float64,
        )
        result[landunit] = values[:, density, latitude, longitude].T
    if np.any(~np.isfinite(result[landunit])):
        raise ValueError("an active urban landunit has missing surface values")
    return result


def _urban_construction_geometry(
    total_thickness_landunit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    total = np.asarray(total_thickness_landunit, dtype=np.float64)
    count = total.shape[0]
    node = np.empty((count, NUMBER_OF_URBAN_LEVELS), dtype=np.float64)
    for level in range(1, NUMBER_OF_URBAN_LEVELS + 1):
        node[:, level - 1] = (np.float64(level) - np.float64(0.5)) * (
            total / np.float64(NUMBER_OF_URBAN_LEVELS)
        )
    thickness = np.empty_like(node)
    thickness[:, 0] = np.float64(0.5) * (node[:, 0] + node[:, 1])
    for level in range(1, NUMBER_OF_URBAN_LEVELS - 1):
        thickness[:, level] = np.float64(0.5) * (node[:, level + 1] - node[:, level - 1])
    thickness[:, -1] = node[:, -1] - node[:, -2]
    interface = np.empty((count, NUMBER_OF_URBAN_LEVELS + 1), dtype=np.float64)
    interface[:, 0] = np.float64(0.0)
    for level in range(1, NUMBER_OF_URBAN_LEVELS):
        interface[:, level] = np.float64(0.5) * (node[:, level - 1] + node[:, level])
    interface[:, -1] = node[:, -1] + np.float64(0.5) * thickness[:, -1]
    return node, interface, thickness


def _column_geometry(
    topology: SubgridTopology,
    snow_layer_thickness: jnp.ndarray,
    snow_node_depth: jnp.ndarray,
    snow_interface_depth: jnp.ndarray,
    thick_roof_landunit: np.ndarray,
    thick_wall_landunit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    number_of_columns = topology.column.gridcell.shape[0]
    ground_node, ground_interface, ground_thickness = soil_layer_geometry_20sl_8p5m()
    node = np.broadcast_to(ground_node, (number_of_columns, NUMBER_OF_GROUND_LEVELS)).copy()
    interface = np.broadcast_to(
        ground_interface,
        (number_of_columns, NUMBER_OF_GROUND_LEVELS + 1),
    ).copy()
    thickness = np.broadcast_to(
        ground_thickness,
        (number_of_columns, NUMBER_OF_GROUND_LEVELS),
    ).copy()

    roof_node, roof_interface, roof_thickness = _urban_construction_geometry(thick_roof_landunit)
    wall_node, wall_interface, wall_thickness = _urban_construction_geometry(thick_wall_landunit)
    column_type = np.asarray(topology.column.column_type, dtype=np.int64)
    column_landunit = np.asarray(topology.column.landunit, dtype=np.int64)
    special = np.float64(CLM_SPECIAL_VALUE)
    for selected_type, construction in (
        (COLUMN_URBAN_ROOF, (roof_node, roof_interface, roof_thickness)),
        (COLUMN_URBAN_SUN_WALL, (wall_node, wall_interface, wall_thickness)),
        (COLUMN_URBAN_SHADE_WALL, (wall_node, wall_interface, wall_thickness)),
    ):
        selected = np.flatnonzero(column_type == selected_type)
        parent = column_landunit[selected]
        selected_node, selected_interface, selected_thickness = construction
        node[selected] = special
        interface[selected] = special
        thickness[selected] = special
        node[selected, :NUMBER_OF_URBAN_LEVELS] = selected_node[parent]
        interface[selected, : NUMBER_OF_URBAN_LEVELS + 1] = selected_interface[parent]
        thickness[selected, :NUMBER_OF_URBAN_LEVELS] = selected_thickness[parent]

    return (
        np.concatenate((np.asarray(snow_layer_thickness), thickness), axis=1),
        np.concatenate((np.asarray(snow_node_depth), node), axis=1),
        np.concatenate(
            (np.asarray(snow_interface_depth), interface[:, 1:]),
            axis=1,
        ),
    )


def load_soil_temperature_driver_native_data(
    case: CimeCase,
    restart_path: str | Path,
) -> SoilTemperatureDriverNativeData:

    restart = Path(restart_path).expanduser().resolve()
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    canopy = load_canopy_hydrology_restart(case, restart)
    temperature = load_canopy_temperature_driver_native_data(case, restart)
    urban = load_urban_flux_driver_native_data(case, restart)
    urban_time = load_urban_time_varying_native_data(case, restart)
    topology = canopy.topology
    for candidate in (temperature.topology, urban.topology, urban_time.topology):
        assert_same_topology(topology, candidate)
    expected_column_shape = topology.column.gridcell.shape

    surface_data = _configured_clm_input(case, "fsurdat")
    parameter_data = _configured_clm_input(case, "paramfile")
    surface = load_native_variables(surface_data, list(_SURFACE_VARIABLES))
    parameter = load_native_variables(parameter_data, ["organic_max"])
    restart_values = load_native_variables(restart, list(_RESTART_VARIABLES))
    organic_maximum = float(np.asarray(parameter["organic_max"]).item())

    hydraulic = column_soil_hydraulic_properties_20sl_8p5m(
        topology,
        surface,
        organic_maximum,
    )
    soil_thermal = column_soil_thermal_properties_20sl_8p5m(
        topology,
        surface,
        organic_maximum,
    )
    bedrock = column_bedrock_layer_count_20sl_8p5m(
        topology,
        surface["zbedrock"],
    )
    wall_conductivity_landunit = _selected_urban_landunit_values(topology, surface["TK_WALL"])
    roof_conductivity_landunit = _selected_urban_landunit_values(topology, surface["TK_ROOF"])
    road_conductivity_landunit = _selected_urban_landunit_values(topology, surface["TK_IMPROAD"])
    wall_capacity_landunit = _selected_urban_landunit_values(topology, surface["CV_WALL"])
    roof_capacity_landunit = _selected_urban_landunit_values(topology, surface["CV_ROOF"])
    road_capacity_landunit = _selected_urban_landunit_values(topology, surface["CV_IMPROAD"])
    impervious_layers_landunit = _selected_urban_landunit_values(topology, surface["NLEV_IMPROAD"])
    impervious_layers_landunit = np.where(
        np.abs(impervious_layers_landunit) < np.float64(1.0e35),
        impervious_layers_landunit,
        np.float64(0.0),
    ).astype(np.int32)
    building_minimum = _selected_urban_landunit_values(topology, surface["T_BUILDING_MIN"])
    thick_roof = _selected_urban_landunit_values(topology, surface["THICK_ROOF"])
    thick_wall = _selected_urban_landunit_values(topology, surface["THICK_WALL"])
    column_landunit = np.asarray(topology.column.landunit, dtype=np.int64)

    snow = canopy.state.snow
    layer_thickness, node_depth, interface_depth = _column_geometry(
        topology,
        snow.layer_thickness,
        snow.layer_depth,
        snow.layer_interface_depth,
        thick_roof,
        thick_wall,
    )
    number_of_columns = expected_column_shape[0]
    number_of_patches = topology.patch.gridcell.shape[0]
    number_of_landunits = topology.landunit.gridcell.shape[0]
    layer_shape = (number_of_columns, NUMBER_OF_TOTAL_LEVELS)
    interface_shape = (number_of_columns, NUMBER_OF_TOTAL_LEVELS + 1)
    radiation_shape = (number_of_columns, NUMBER_OF_SNOW_LEVELS + 1)
    solve_shape = (number_of_columns, NUMBER_OF_TEMPERATURE_SOLVE_LEVELS)
    zero_column = jnp.zeros((number_of_columns,), dtype=jnp.float64)
    zero_patch = jnp.zeros((number_of_patches,), dtype=jnp.float64)
    zero_landunit = jnp.zeros((number_of_landunits,), dtype=jnp.float64)
    zero_layer = jnp.zeros(layer_shape, dtype=jnp.float64)
    zero_interface = jnp.zeros(interface_shape, dtype=jnp.float64)
    zero_radiation = jnp.zeros(radiation_shape, dtype=jnp.float64)
    zero_solve = jnp.zeros(solve_shape, dtype=jnp.float64)
    active_nonlake = jnp.asarray(topology.column.active) & (
        jnp.asarray(topology.column.landunit_type) != jnp.int32(LANDUNIT_LAKE)
    )
    snow_layer_count = jnp.asarray(snow.layer_count, dtype=jnp.int32)
    landunit_type = jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
    column_type = jnp.asarray(topology.column.column_type, dtype=jnp.int32)
    temperature_inputs = temperature.inputs
    water_temperature = temperature_inputs.surface_water_temperature_column
    water_fraction = canopy.state.surface_water_fraction_column

    thermal = SoilThermalPropertyInputs(
        active_nonlake=active_nonlake,
        snow_layer_count=snow_layer_count,
        landunit_type=landunit_type,
        column_type=column_type,
        bedrock_layer_count=jnp.asarray(bedrock, dtype=jnp.int32),
        impervious_road_layer_count=jnp.asarray(
            impervious_layers_landunit[column_landunit], dtype=jnp.int32
        ),
        layer_thickness=jnp.asarray(layer_thickness),
        node_depth=jnp.asarray(node_depth),
        interface_depth=jnp.asarray(interface_depth),
        soil_snow_temperature=temperature_inputs.soil_snow_temperature_column_level,
        liquid_water=temperature_inputs.soil_liquid_water_column_level,
        ice_water=temperature_inputs.soil_ice_water_column_level,
        effective_snow_fraction=snow.effective_snow_fraction,
        surface_water=canopy.state.surface_water_column,
        snow_water=snow.water_equivalent,
        soil_mineral_conductivity=jnp.asarray(
            soil_thermal.mineral_thermal_conductivity_column_level
        ),
        dry_soil_conductivity=jnp.asarray(soil_thermal.dry_thermal_conductivity_column_level),
        soil_solid_heat_capacity=jnp.asarray(
            soil_thermal.solid_volumetric_heat_capacity_column_level
        ),
        soil_porosity=jnp.asarray(hydraulic.saturated_water_content_column_level),
        urban_wall_conductivity=jnp.asarray(wall_conductivity_landunit[column_landunit]),
        urban_roof_conductivity=jnp.asarray(roof_conductivity_landunit[column_landunit]),
        urban_impervious_road_conductivity=jnp.asarray(road_conductivity_landunit[column_landunit]),
        urban_wall_volumetric_heat_capacity=jnp.asarray(wall_capacity_landunit[column_landunit]),
        urban_roof_volumetric_heat_capacity=jnp.asarray(roof_capacity_landunit[column_landunit]),
        urban_impervious_road_volumetric_heat_capacity=jnp.asarray(
            road_capacity_landunit[column_landunit]
        ),
    )
    ground = GroundHeatFluxInputs(
        snow_layer_count_column=snow_layer_count,
        downward_longwave_column=zero_column,
        effective_snow_fraction_column=snow.effective_snow_fraction,
        ground_emissivity_column=temperature.state.column.ground_emissivity_column,
        surface_water_temperature_column=water_temperature,
        ground_temperature_column=temperature.state.column.ground_temperature_column,
        soil_snow_temperature_column=thermal.soil_snow_temperature,
        latent_heat_column=temperature.state.column.latent_heat_column,
        vegetation_exposed_patch=jnp.zeros((number_of_patches,), dtype=jnp.int32),
        snow_evaporation_patch=zero_patch,
        soil_evaporation_patch=zero_patch,
        surface_water_evaporation_patch=zero_patch,
        total_soil_evaporation_patch=zero_patch,
        vegetation_transpiration_patch=zero_patch,
        ground_energy_derivative_patch=zero_patch,
        canopy_downward_longwave_patch=zero_patch,
        snow_sensible_heat_patch=zero_patch,
        soil_sensible_heat_patch=zero_patch,
        surface_water_sensible_heat_patch=zero_patch,
        ground_sensible_heat_patch=zero_patch,
        net_longwave_patch=zero_patch,
        absorbed_ground_patch=zero_patch,
        absorbed_soil_patch=zero_patch,
        absorbed_snow_patch=zero_patch,
        absorbed_ground_layers_patch=jnp.zeros(
            (number_of_patches, NUMBER_OF_SNOW_LEVELS + 1), dtype=jnp.float64
        ),
        waste_heat_landunit=zero_landunit,
        air_conditioner_heat_landunit=zero_landunit,
        traffic_heat_landunit=zero_landunit,
        urban_roof_fraction_landunit=zero_landunit,
    )
    patch_state = GroundHeatFluxPatchState(*(zero_patch for _ in GroundHeatFluxPatchState._fields))

    compact_landunit = np.asarray(urban.inputs.native_map.landunit_index, dtype=np.int64)
    roof_height = _scatter_compact(
        number_of_landunits,
        compact_landunit,
        urban.inputs.static_parameters.roof_height_landunit,
    )
    roof_fraction = _scatter_compact(
        number_of_landunits,
        compact_landunit,
        urban.inputs.static_parameters.roof_fraction_landunit,
    )
    canyon_ratio = _scatter_compact(
        number_of_landunits,
        compact_landunit,
        urban.inputs.static_parameters.canyon_height_width_ratio_landunit,
    )
    canyon_temperature = _scatter_compact(
        number_of_landunits,
        compact_landunit,
        urban.state.turbulence.canopy_air_temperature_landunit,
    )
    building_temperature = _scatter_compact(
        number_of_landunits,
        compact_landunit,
        urban.state.internal_building_temperature_landunit,
    )
    air_conditioning = _scatter_compact(
        number_of_landunits,
        compact_landunit,
        urban.state.air_conditioning_flux_landunit,
    )
    heating = _scatter_compact(
        number_of_landunits,
        compact_landunit,
        urban.state.heating_flux_landunit,
    )
    urban_maximum = interpolate_urban_time_varying(urban_time.inputs)
    if int(urban_maximum.diagnostics.invalid_urban_landunit_count) != 0:
        raise ValueError("urban time-varying stream contains invalid landunits")
    active_urban = (
        jnp.asarray(topology.landunit.active)
        & (jnp.asarray(topology.landunit.landunit_type) >= jnp.int32(LANDUNIT_URBAN_MIN))
        & (jnp.asarray(topology.landunit.landunit_type) <= jnp.int32(LANDUNIT_URBAN_MAX))
    )
    building = UrbanBuildingTemperatureInputs(
        active_urban=active_urban,
        roof_height=jnp.asarray(roof_height),
        canyon_height_width_ratio=jnp.asarray(canyon_ratio),
        roof_weight=jnp.asarray(roof_fraction),
        canyon_air_temperature=jnp.asarray(canyon_temperature),
        minimum_building_temperature=jnp.asarray(building_minimum),
        maximum_building_temperature=(urban_maximum.state.building_maximum_temperature_landunit),
        roof_inner_temperature=jnp.asarray(_finite_restart(restart_values["t_roof_inner"])),
        sunwall_inner_temperature=jnp.asarray(_finite_restart(restart_values["t_sunw_inner"])),
        shadewall_inner_temperature=jnp.asarray(_finite_restart(restart_values["t_shdw_inner"])),
        floor_temperature=jnp.asarray(_finite_restart(restart_values["t_floor"])),
        building_temperature=jnp.asarray(building_temperature),
        previous_roof_boundary_temperature=zero_landunit,
        previous_sunwall_boundary_temperature=zero_landunit,
        previous_shadewall_boundary_temperature=zero_landunit,
        roof_boundary_temperature=zero_landunit,
        sunwall_boundary_temperature=zero_landunit,
        shadewall_boundary_temperature=zero_landunit,
        roof_node_depth=zero_landunit,
        sunwall_node_depth=zero_landunit,
        shadewall_node_depth=zero_landunit,
        roof_interface_depth=zero_landunit,
        sunwall_interface_depth=zero_landunit,
        shadewall_interface_depth=zero_landunit,
        roof_thermal_conductivity=zero_landunit,
        sunwall_thermal_conductivity=zero_landunit,
        shadewall_thermal_conductivity=zero_landunit,
        previous_building_heat_flux=zero_landunit,
        previous_air_conditioning_flux=jnp.asarray(air_conditioning),
        previous_heating_flux=jnp.asarray(heating),
    )
    heat_diffusion = SoilHeatDiffusionInputs(
        active_nonlake=active_nonlake,
        snow_layer_count=snow_layer_count,
        column_type=column_type,
        layer_thickness=zero_layer,
        node_depth=zero_layer,
        interface_depth=zero_interface,
        interface_thermal_conductivity=zero_layer,
        layer_heat_capacity=zero_layer,
        soil_snow_temperature=zero_layer,
        bottom_heat_flux=zero_column,
        building_temperature=zero_column,
        roof_inner_temperature=zero_column,
        sunwall_inner_temperature=zero_column,
        shadewall_inner_temperature=zero_column,
    )
    rhs = SoilTemperatureRHSInputs(
        active_nonlake=active_nonlake,
        snow_layer_count=snow_layer_count,
        landunit_type=landunit_type,
        column_type=column_type,
        standing_water_heat_flux=zero_column,
        top_snow_heat_flux=zero_column,
        soil_heat_flux=zero_column,
        top_layer_heat_flux=zero_column,
        heat_flux_derivative=zero_column,
        absorbed_ground_layers=zero_radiation,
        layer_thermal_conductivity=zero_layer,
        standing_water_thermal_conductivity=zero_column,
        timestep_over_heat_capacity=zero_layer,
        interface_heat_flux=zero_layer,
        standing_water_heat_capacity=zero_column,
        standing_water_thickness=zero_column,
        node_depth=zero_layer,
        soil_snow_temperature=zero_layer,
        standing_water_temperature=water_temperature,
        standing_water_fraction=water_fraction,
        effective_snow_fraction=snow.effective_snow_fraction,
    )
    matrix = SoilTemperatureMatrixInputs(
        active_nonlake=active_nonlake,
        snow_layer_count=snow_layer_count,
        landunit_type=landunit_type,
        column_type=column_type,
        heat_flux_derivative=zero_column,
        layer_thermal_conductivity=zero_layer,
        standing_water_thermal_conductivity=zero_column,
        timestep_over_heat_capacity=zero_layer,
        standing_water_heat_capacity=zero_column,
        standing_water_thickness=zero_column,
        node_depth=zero_layer,
        interface_depth=zero_interface,
        standing_water_fraction=water_fraction,
        effective_snow_fraction=snow.effective_snow_fraction,
    )
    linear_solve = SoilTemperatureLinearSolveInputs(
        active_nonlake=active_nonlake,
        snow_layer_count=snow_layer_count,
        column_type=column_type,
        band_matrix=jnp.zeros(
            (
                number_of_columns,
                NUMBER_OF_TEMPERATURE_MATRIX_BANDS,
                NUMBER_OF_TEMPERATURE_SOLVE_LEVELS,
            ),
            dtype=jnp.float64,
        ),
        right_hand_side=zero_solve,
    )
    solution = SoilTemperatureSolutionUpdateInputs(
        active_nonlake=active_nonlake,
        snow_layer_count=snow_layer_count,
        column_type=column_type,
        standing_water_fraction=water_fraction,
        previous_soil_snow_temperature=zero_layer,
        previous_standing_water_temperature=water_temperature,
        solution_vector=zero_solve,
    )
    postsolve = SoilTemperaturePostSolveFluxInputs(
        active_nonlake=active_nonlake,
        snow_layer_count=snow_layer_count,
        urban_landunit=(landunit_type >= LANDUNIT_URBAN_MIN)
        & (landunit_type <= LANDUNIT_URBAN_MAX),
        column_type=column_type,
        node_depth=zero_layer,
        interface_depth=zero_interface,
        interface_thermal_conductivity=zero_layer,
        previous_interface_heat_flux=zero_layer,
        previous_updated_interface_heat_flux=zero_layer,
        previous_soil_snow_temperature=thermal.soil_snow_temperature,
        soil_snow_temperature=zero_layer,
        building_temperature=zero_column,
        roof_inner_temperature=zero_column,
        sunwall_inner_temperature=zero_column,
        shadewall_inner_temperature=zero_column,
        cooling_on=jnp.zeros((number_of_columns,), dtype=jnp.bool_),
        heating_on=jnp.zeros((number_of_columns,), dtype=jnp.bool_),
        previous_building_heat_flux=zero_column,
        previous_urban_air_conditioning_flux=zero_column,
        previous_urban_heating_flux=zero_column,
    )
    standing_phase = StandingWaterPhaseChangeInputs(
        active_nonlake=active_nonlake,
        snow_layer_count=snow_layer_count,
        surface_flux_temperature_derivative=zero_column,
        effective_snow_fraction=snow.effective_snow_fraction,
        standing_water_fraction=water_fraction,
        snow_water=snow.water_equivalent,
        standing_water=canopy.state.surface_water_column,
        integrated_snowfall=snow.integrated_snow,
        snow_depth=snow.depth,
        ice_water=thermal.ice_water,
        timestep_over_heat_capacity=zero_layer,
        standing_water_heat_capacity=zero_column,
        soil_snow_temperature=zero_layer,
        standing_water_temperature=water_temperature,
        previous_phase_change_heat_flux=zero_column,
        previous_water_to_ice_flux=zero_column,
        previous_water_to_snow_heat_flux=zero_column,
    )
    soil_phase = SoilSnowPhaseChangeInputs(
        active_nonlake=active_nonlake,
        snow_layer_count=snow_layer_count,
        landunit_type=landunit_type,
        column_type=column_type,
        layer_thickness=thermal.layer_thickness,
        soil_clapp_hornberger_exponent=jnp.asarray(hydraulic.clapp_hornberger_b_column_level),
        soil_saturated_suction=jnp.asarray(hydraulic.saturated_suction_column_level),
        soil_porosity=thermal.soil_porosity,
        surface_flux_temperature_derivative=zero_column,
        effective_snow_fraction=snow.effective_snow_fraction,
        standing_water_fraction=water_fraction,
        snow_depth=snow.depth,
        snow_water=snow.water_equivalent,
        liquid_water=thermal.liquid_water,
        ice_water=thermal.ice_water,
        timestep_over_heat_capacity=zero_layer,
        soil_snow_temperature=zero_layer,
        previous_phase_flag=jnp.zeros(layer_shape, dtype=jnp.int32),
        previous_phase_change_heat_flux=zero_column,
        previous_snow_melt_flux=zero_column,
        previous_snow_freeze_flux=zero_column,
        previous_snow_drain_flux=zero_column,
        previous_snow_layer_melt_flux=jnp.zeros(
            (number_of_columns, NUMBER_OF_SNOW_LEVELS), dtype=jnp.float64
        ),
        previous_snow_layer_freeze_flux=jnp.zeros(
            (number_of_columns, NUMBER_OF_SNOW_LEVELS), dtype=jnp.float64
        ),
        previous_snow_melt_heat_flux=zero_column,
        previous_rural_snow_melt_heat_flux=zero_column,
        previous_urban_snow_melt_heat_flux=zero_column,
    )
    final = SoilTemperatureFinalDiagnosticsInputs(
        active_nonlake=active_nonlake,
        snow_layer_count=snow_layer_count,
        landunit_type=landunit_type,
        effective_snow_fraction=snow.effective_snow_fraction,
        standing_water_fraction=water_fraction,
        soil_snow_temperature=zero_layer,
        standing_water_temperature=water_temperature,
        interface_heat_flux=zero_layer,
        updated_interface_heat_flux=zero_layer,
        previous_ground_temperature=temperature.state.column.ground_temperature_column,
        previous_layer_one_two_heat_flux=zero_column,
        previous_ground_layer_heat_flux=jnp.zeros(
            (number_of_columns, NUMBER_OF_GROUND_LEVELS), dtype=jnp.float64
        ),
    )
    active_template = SoilTemperatureActiveInputs(
        topology=topology,
        urban_columns=make_urban_construction_column_map(topology),
        thermal_properties=thermal,
        ground_heat_flux=ground,
        ground_heat_flux_patch_state=patch_state,
        heat_diffusion_template=heat_diffusion,
        rhs_template=rhs,
        matrix_template=matrix,
        linear_solve_template=linear_solve,
        solution_update_template=solution,
        postsolve_flux_template=postsolve,
        standing_water_phase_change_template=standing_phase,
        soil_snow_phase_change_template=soil_phase,
        building_temperature_template=building,
        final_diagnostics_template=final,
    )
    return SoilTemperatureDriverNativeData(
        topology=topology,
        inputs=SoilTemperatureDriverInputs(
            active_template=active_template,
            urban=urban.inputs,
        ),
        restart_path=restart,
        surface_data_path=surface_data,
        parameter_data_path=parameter_data,
        urban_stream_path=urban_time.stream_path,
    )


__all__ = [
    "SoilTemperatureDriverNativeData",
    "load_soil_temperature_driver_native_data",
]
