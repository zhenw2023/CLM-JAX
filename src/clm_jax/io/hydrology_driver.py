
from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE, DENSITY_FRESH_WATER, DENSITY_ICE
from clm_jax.forcing.area import load_land_flux_area_factor
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.coupler import load_atmospheric_aerosol_coupler_forcing
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.physical_driver import PhysicalDriverNativeData
from clm_jax.io.restart import (
    COLUMN_URBAN_IMPERVIOUS_ROAD,
    COLUMN_URBAN_PERVIOUS_ROAD,
    COLUMN_URBAN_ROOF,
    COLUMN_URBAN_SHADE_WALL,
    COLUMN_URBAN_SUN_WALL,
    LANDUNIT_LAKE,
    SubgridTopology,
    load_restart_clock,
)
from clm_jax.physics.hydrology_diagnostics import (
    HydrologyDiagnosticsInputs,
    HydrologyDiagnosticsState,
)
from clm_jax.physics.hydrology_driver import HydrologyDriverInputs
from clm_jax.physics.hydrology_no_drainage import (
    HydrologyNoDrainageInputs,
    HydrologyNoDrainageParameters,
    HydrologyNoDrainageState,
)
from clm_jax.physics.snow_capping import SnowCappingState
from clm_jax.physics.snow_compaction import SnowCompactionInputs
from clm_jax.physics.snow_layer_combining import SnowLayerCombiningState
from clm_jax.physics.snow_layer_postprocess import SnowLayerPostprocessState
from clm_jax.physics.snow_water import (
    NUMBER_OF_GROUND_LEVELS,
    NUMBER_OF_SNOW_LEVELS,
    NUMBER_OF_SOIL_SNOW_LEVELS,
    SnowAerosolDepositionState,
    SnowWaterAerosolState,
    SnowWaterInputs,
    SnowWaterState,
)
from clm_jax.physics.soil_water_movement import (
    NUMBER_OF_SOIL_LEVELS,
    SoilWaterMovementInputs,
    SoilWaterMovementState,
)
from clm_jax.physics.soil_water_plant_sink import (
    SoilWaterPlantSinkInputs,
    SoilWaterPlantSinkState,
)
from clm_jax.physics.surface_hydrology import (
    SurfaceHydrologyInputs,
    SurfaceHydrologyState,
)
from clm_jax.physics.water_table import (
    WaterTableCondensationInputs,
    WaterTableCondensationState,
)


class HydrologyDriverNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: HydrologyDriverInputs
    state: HydrologyNoDrainageState
    parameters: HydrologyNoDrainageParameters
    land_restart_path: Path
    coupler_restart_path: Path
    surface_data_path: Path


_RESTART_VARIABLES = (
    "WA",
    "ZWT",
    "ZWT_PERCH",
    "FROST_TABLE",
    "SMP",
    "HK",
    "qflx_snow_drain",
    "SNOW_PERS",
    "T_GRND",
    "T_GRND_R",
    "T_GRND_U",
    "wf",
)


def _configured_surface_data(case: CimeCase) -> Path:
    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    declared = general.get("fsurdat")
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError("configured clm_inparm does not define fsurdat")
    path = Path(case.expand(declared)).expanduser()
    if not path.is_absolute():
        path = case.root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM surface data does not exist: {path}")
    return path


def _select_surface_columns(
    source: jnp.ndarray,
    topology: SubgridTopology,
) -> np.ndarray:
    values = np.asarray(source, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("hydrology surface fields must use (lat, lon) axes")
    latitude = np.asarray(topology.grid.latitude_index, dtype=np.int64)
    longitude = np.asarray(topology.grid.longitude_index, dtype=np.int64)
    gridcell = values[latitude, longitude]
    return gridcell[np.asarray(topology.column.gridcell, dtype=np.int64)]


def _initial_top_17cm_soil_temperature(topology: SubgridTopology) -> jnp.ndarray:

    landunit = jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
    column_type = jnp.asarray(topology.column.column_type, dtype=jnp.int32)
    enclosure = (
        (column_type == jnp.int32(COLUMN_URBAN_ROOF))
        | (column_type == jnp.int32(COLUMN_URBAN_SUN_WALL))
        | (column_type == jnp.int32(COLUMN_URBAN_SHADE_WALL))
    )
    road = (
        (column_type == jnp.int32(COLUMN_URBAN_IMPERVIOUS_ROAD))
        | (column_type == jnp.int32(COLUMN_URBAN_PERVIOUS_ROAD))
    )
    return jnp.where(
        landunit == jnp.int32(LANDUNIT_LAKE),
        jnp.float64(277.0),
        jnp.where(
            enclosure,
            jnp.float64(292.0),
            jnp.where(road, jnp.float64(274.0), jnp.float64(0.0)),
        ),
    )


def _surface_water_threshold(microtopography_sigma: jnp.ndarray) -> np.ndarray:

    sigma = np.asarray(microtopography_sigma, dtype=np.float64)
    selected = sigma > np.float64(1.0e-6)
    safe_sigma = np.where(selected, sigma, np.float64(1.0))
    d = np.zeros_like(safe_sigma)
    percolation_probability = np.float64(np.float32(0.4))
    square_root_two = np.float64(np.sqrt(np.float32(2.0)))
    square_root_two_pi = np.sqrt(np.float64(2.0) * np.float64(math.pi))

    def error_function(value: np.ndarray) -> np.ndarray:
        return np.fromiter(
            (math.erf(float(item)) for item in value),
            dtype=np.float64,
            count=value.size,
        )

    for _ in range(4):
        ratio = d / (safe_sigma * square_root_two)
        fd = (
            np.float64(0.5) * (np.float64(1.0) + error_function(ratio))
            - percolation_probability
        )
        derivative = np.exp(-(d**2) / (np.float64(2.0) * safe_sigma**2)) / (
            safe_sigma * square_root_two_pi
        )
        d = d - fd / derivative
    ratio = d / (safe_sigma * square_root_two)
    threshold = np.float64(0.5) * d * (
        np.float64(1.0) + error_function(ratio)
    ) + safe_sigma / square_root_two_pi * np.exp(-(d**2) / (np.float64(2.0) * safe_sigma**2))
    return np.where(selected, np.float64(1000.0) * threshold, np.float64(0.0))


def _snow_aerosol(native: PhysicalDriverNativeData) -> SnowWaterAerosolState:
    aerosol = native.state.canopy.canopy_hydrology.snow_aerosol
    number_of_columns = native.static.topology.column.gridcell.shape[0]
    zero_column = jnp.zeros((number_of_columns,), dtype=jnp.float64)
    deposition = SnowAerosolDepositionState(
        **{name: zero_column for name in SnowAerosolDepositionState._fields}
    )
    return SnowWaterAerosolState(
        black_carbon_hydrophilic_layer=aerosol.hydrophilic_black_carbon_layer,
        black_carbon_hydrophobic_layer=aerosol.hydrophobic_black_carbon_layer,
        organic_carbon_hydrophilic_layer=aerosol.hydrophilic_organic_carbon_layer,
        organic_carbon_hydrophobic_layer=aerosol.hydrophobic_organic_carbon_layer,
        dust_species_1_layer=aerosol.dust_species_1_layer,
        dust_species_2_layer=aerosol.dust_species_2_layer,
        dust_species_3_layer=aerosol.dust_species_3_layer,
        dust_species_4_layer=aerosol.dust_species_4_layer,
        deposition=deposition,
    )


def _initial_water_diagnostics(
    liquid: np.ndarray,
    ice: np.ndarray,
    thickness: np.ndarray,
    porosity: np.ndarray,
    saturated_suction: np.ndarray,
    exponent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = (
        np.isfinite(liquid)
        & np.isfinite(ice)
        & np.isfinite(thickness)
        & (thickness > np.float64(0.0))
        & np.isfinite(porosity)
        & (porosity > np.float64(0.0))
    )
    safe_thickness = np.where(valid, thickness, np.float64(1.0))
    safe_porosity = np.where(valid, porosity, np.float64(1.0))
    liquid_volume = liquid / (safe_thickness * np.float64(DENSITY_FRESH_WATER))
    ice_volume = np.minimum(
        safe_porosity,
        ice / (safe_thickness * np.float64(DENSITY_ICE)),
    )
    volumetric = np.where(valid, liquid_volume + ice_volume, np.float64(0.0))
    ice_fraction = np.where(valid, np.minimum(np.float64(1.0), ice_volume / safe_porosity), 0.0)
    effective_porosity = np.where(
        valid,
        np.maximum(np.float64(0.01), safe_porosity - ice_volume),
        np.float64(0.01),
    )
    exp_minus_three = np.exp(np.float64(-3.0))
    fractional_impermeability = np.where(
        valid,
        np.maximum(
            np.float64(0.0),
            (np.exp(np.float64(-3.0) * (np.float64(1.0) - ice_fraction)) - exp_minus_three)
            / (np.float64(1.0) - exp_minus_three),
        ),
        np.float64(0.0),
    )

    safe_suction = np.where(valid & np.isfinite(saturated_suction), saturated_suction, 1.0)
    safe_exponent = np.where(valid & np.isfinite(exponent), exponent, 1.0)
    fractional_saturation = np.maximum(liquid_volume / safe_porosity, np.float64(0.001))
    with np.errstate(over="ignore", invalid="ignore"):
        soil_water_potential = (
            safe_suction
            * np.float64(-9.8e-6)
            * (fractional_saturation ** (-safe_exponent))
        )
    soil_water_potential = np.minimum(
        np.maximum(soil_water_potential, np.float64(-15.0)),
        np.float64(0.0),
    )
    soil_water_potential = np.where(
        valid & (liquid > np.float64(0.0)),
        soil_water_potential,
        np.float64(-15.0),
    )
    return (
        volumetric,
        ice_fraction,
        effective_porosity,
        fractional_impermeability,
        soil_water_potential,
    )


def load_hydrology_driver_native_data(
    case: CimeCase,
    physical: PhysicalDriverNativeData,
) -> HydrologyDriverNativeData:

    restart = physical.land_restart_path.resolve()
    coupler_restart = physical.coupler_restart_path.resolve()
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    if not coupler_restart.is_file():
        raise FileNotFoundError(f"CPL restart does not exist: {coupler_restart}")
    topology = physical.static.topology
    number_of_gridcells = topology.grid.longitude_degrees.shape[0]
    number_of_columns = topology.column.gridcell.shape[0]
    number_of_patches = topology.patch.gridcell.shape[0]
    clock = load_restart_clock(restart)
    values = load_native_variables(
        restart,
        list(_RESTART_VARIABLES),
        dtype=np.float64,
        masked_floating_fill_value=CLM_SPECIAL_VALUE,
    )
    surface_path = _configured_surface_data(case)
    surface = load_native_variables(surface_path, ["FMAX", "SLOPE"], dtype=np.float64)

    canopy = physical.state.canopy.canopy_hydrology
    snow = canopy.snow
    structure = physical.inputs.canopy.structure
    active = physical.inputs.soil_temperature.active_template
    thermal = active.thermal_properties
    phase = active.soil_snow_phase_change_template
    standing = active.standing_water_phase_change_template
    hydraulic = physical.inputs.canopy_flux.temperature_parameters
    urban_map = physical.inputs.urban_flux.native_map

    thickness = jnp.asarray(thermal.layer_thickness, dtype=jnp.float64)
    node_depth = jnp.asarray(thermal.node_depth, dtype=jnp.float64)
    interface_depth = jnp.asarray(thermal.interface_depth, dtype=jnp.float64)
    temperature = jnp.asarray(thermal.soil_snow_temperature, dtype=jnp.float64)
    liquid = jnp.asarray(phase.liquid_water, dtype=jnp.float64)
    ice = jnp.asarray(phase.ice_water, dtype=jnp.float64)
    ground_slice = slice(NUMBER_OF_SNOW_LEVELS, None)
    ground_liquid = np.asarray(liquid[:, ground_slice], dtype=np.float64)
    ground_ice = np.asarray(ice[:, ground_slice], dtype=np.float64)
    ground_thickness = np.asarray(thickness[:, ground_slice], dtype=np.float64)
    porosity = np.asarray(
        hydraulic.saturated_water_content_column,
        dtype=np.float64,
    )
    saturated_suction = np.asarray(hydraulic.saturated_suction_column, dtype=np.float64)
    exponent = np.asarray(hydraulic.clapp_hornberger_b_column, dtype=np.float64)
    (
        volumetric,
        ice_fraction,
        effective_porosity,
        fractional_impermeability,
        soil_water_potential,
    ) = _initial_water_diagnostics(
        ground_liquid,
        ground_ice,
        ground_thickness,
        porosity,
        saturated_suction,
        exponent,
    )

    zero_column = jnp.zeros((number_of_columns,), dtype=jnp.float64)
    zero_patch = jnp.zeros((number_of_patches,), dtype=jnp.float64)
    zero_ground = jnp.zeros(
        (number_of_columns, NUMBER_OF_GROUND_LEVELS),
        dtype=jnp.float64,
    )
    zero_soil = jnp.zeros(
        (number_of_columns, NUMBER_OF_SOIL_LEVELS),
        dtype=jnp.float64,
    )
    zero_soil_snow = jnp.zeros(
        (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS),
        dtype=jnp.float64,
    )
    aerosol = _snow_aerosol(physical)

    snow_structure = SnowLayerCombiningState(
        snow_layer_count_column=snow.layer_count,
        snow_fraction_column=snow.snow_fraction,
        effective_snow_fraction_column=snow.effective_snow_fraction,
        snow_depth_column=snow.depth,
        integrated_snow_column=snow.integrated_snow,
        snow_water_equivalent_column=snow.water_equivalent,
        soil_snow_thickness_column_level=thickness,
        soil_snow_node_depth_column_level=node_depth,
        soil_snow_interface_depth_column_interface=interface_depth,
        soil_snow_temperature_column_level=temperature,
        soil_snow_ice_column_level=ice,
        soil_snow_liquid_column_level=liquid,
        snow_grain_radius_column_level=snow.snow_grain_radius,
        top_soil_water_flux_column=zero_column,
        aerosol=aerosol,
    )
    snow_water_state = SnowWaterState(
        integrated_snow_column=snow.integrated_snow,
        snow_fraction_column=snow.snow_fraction,
        snow_depth_column=snow.depth,
        soil_snow_liquid_column_level=liquid,
        soil_snow_ice_column_level=ice,
        soil_snow_thickness_column_level=thickness,
        snow_drainage_flux_column=jnp.asarray(values["qflx_snow_drain"]),
        top_soil_water_flux_column=zero_column,
        aerosol=aerosol,
    )
    surface_state = SurfaceHydrologyState(
        soil_snow_liquid_column_level=liquid,
        surface_water_column=canopy.surface_water_column,
        top_soil_water_flux_column=zero_column,
        surface_runoff_flux_column=zero_column,
        surface_water_runoff_flux_column=zero_column,
        infiltration_flux_column=zero_column,
        saturated_fraction_column=zero_column,
        impermeable_fraction_column=zero_column,
        fractional_impermeability_column_level=jnp.asarray(fractional_impermeability),
        soil_ice_fraction_column_level=jnp.asarray(ice_fraction),
        effective_porosity_column_level=jnp.asarray(effective_porosity),
    )
    plant_state = SoilWaterPlantSinkState(
        effective_root_fraction_column_level=zero_ground,
        root_soil_water_flux_column_level=zero_soil,
        negative_hydraulic_redistribution_flux_column=zero_column,
    )
    movement_state = SoilWaterMovementState(
        liquid_water_mass_column_level=jnp.asarray(ground_liquid),
        volumetric_water_column_level=jnp.asarray(volumetric),
        soil_water_potential_column_level=jnp.asarray(values["SMP"]),
        hydraulic_conductivity_column_level=jnp.asarray(values["HK"]),
        aquifer_water_column=jnp.asarray(values["WA"]),
        recharge_flux_column=zero_column,
        adaptive_substep_count_column=jnp.zeros((number_of_columns,), dtype=jnp.int32),
    )
    water_table_state = WaterTableCondensationState(
        soil_snow_liquid_water_column_level=liquid,
        soil_snow_ice_water_column_level=ice,
        volumetric_water_column_level=jnp.asarray(volumetric),
        water_table_depth_column=jnp.asarray(values["ZWT"]),
        perched_water_table_depth_column=jnp.asarray(values["ZWT_PERCH"]),
        frost_table_depth_column=jnp.asarray(values["FROST_TABLE"]),
        snow_sublimation_flux_column=zero_column,
    )
    capping_state = SnowCappingState(
        soil_snow_liquid_column_level=liquid,
        soil_snow_ice_column_level=ice,
        soil_snow_thickness_column_level=thickness,
        snow_capping_ice_flux_column=zero_column,
        snow_capping_liquid_flux_column=zero_column,
        snow_capping_discarded_ice_flux_column=zero_column,
        snow_capping_discarded_liquid_flux_column=zero_column,
        aerosol=aerosol,
    )

    snow_mass = np.asarray(liquid[:, :NUMBER_OF_SNOW_LEVELS] + ice[:, :NUMBER_OF_SNOW_LEVELS])
    snow_ice = jnp.sum(ice[:, :NUMBER_OF_SNOW_LEVELS], axis=1)
    snow_liquid = jnp.sum(liquid[:, :NUMBER_OF_SNOW_LEVELS], axis=1)
    snow_temperature_mass = jnp.sum(
        temperature[:, :NUMBER_OF_SNOW_LEVELS]
        * (liquid[:, :NUMBER_OF_SNOW_LEVELS] + ice[:, :NUMBER_OF_SNOW_LEVELS]),
        axis=1,
    )
    layer_count = np.asarray(snow.layer_count, dtype=np.int32)
    top_index = np.clip(layer_count + NUMBER_OF_SNOW_LEVELS, 0, NUMBER_OF_SNOW_LEVELS - 1)
    row = np.arange(number_of_columns, dtype=np.int64)
    has_snow = layer_count < 0
    top_snow_water = np.where(has_snow, snow_mass[row, top_index], np.float64(0.0))
    postprocess_state = SnowLayerPostprocessState(
        snow_layer_count_column=snow.layer_count,
        snow_persistence_column=jnp.asarray(values["SNOW_PERS"]),
        soil_snow_thickness_column_level=thickness,
        soil_snow_node_depth_column_level=node_depth,
        soil_snow_interface_depth_column_interface=interface_depth,
        soil_snow_temperature_column_level=temperature,
        soil_snow_ice_column_level=ice,
        soil_snow_liquid_column_level=liquid,
        snow_ice_column=snow_ice,
        snow_liquid_column=snow_liquid,
        column_average_snow_depth_column=snow.depth * snow.snow_fraction,
        snow_temperature_mass_product_column=snow_temperature_mass,
        top_snow_water_column=jnp.asarray(top_snow_water),
    )
    final_state = HydrologyDiagnosticsState(
        near_surface_soil_temperature_column=zero_column,
        top_10cm_soil_temperature_column=zero_column,
        top_17cm_soil_temperature_column=(
            _initial_top_17cm_soil_temperature(topology)
        ),
        top_10cm_soil_water_column=zero_column,
        total_soil_liquid_water_column=jnp.sum(liquid[:, 12:32], axis=1),
        total_soil_ice_water_column=jnp.sum(ice[:, 12:32], axis=1),
        ground_temperature_column=jnp.asarray(values["T_GRND"]),
        urban_ground_temperature_column=jnp.asarray(values["T_GRND_U"]),
        rural_ground_temperature_column=jnp.asarray(values["T_GRND_R"]),
        volumetric_soil_water_column_level=jnp.asarray(volumetric),
        soil_water_potential_column_level=jnp.asarray(soil_water_potential),
        soil_matrix_potential_column_level=jnp.asarray(values["SMP"]),
        top_5cm_water_availability_column=jnp.asarray(values["wf"]),
        top_17cm_water_availability_column=zero_column,
        top_snow_water_column=jnp.asarray(top_snow_water),
        snow_grain_radius_column_level=snow.snow_grain_radius,
        top_snow_temperature_column=zero_column,
        top_snow_temperature_gradient_column=zero_column,
        top_snow_grain_radius_column=zero_column,
        top_snow_liquid_fraction_column=zero_column,
    )

    fmax = _select_surface_columns(surface["FMAX"], topology)
    slope = np.maximum(
        _select_surface_columns(surface["SLOPE"], topology),
        np.float64(0.2),
    )
    surface_threshold = _surface_water_threshold(structure.microtopography_sigma_column)
    soil_layer_count = jnp.asarray(thermal.bedrock_layer_count, dtype=jnp.int32)
    lower_interface = interface_depth[
        :, NUMBER_OF_SNOW_LEVELS + 1 : NUMBER_OF_SNOW_LEVELS + 1 + NUMBER_OF_SOIL_LEVELS
    ]
    timestep = jnp.asarray(float(clock.step_seconds), dtype=jnp.float64)

    snow_inputs = SnowWaterInputs(
        time_step_seconds=timestep,
        snow_layer_count_column=snow.layer_count,
        effective_snow_fraction_column=snow.effective_snow_fraction,
        snow_water_equivalent_column=snow.water_equivalent,
        snowmelt_flux_column=zero_column,
        ground_rainfall_flux_column=zero_column,
        snow_sublimation_flux_column=zero_column,
        snow_dew_flux_column=zero_column,
        ground_evaporation_flux_column=zero_column,
        ground_dew_flux_column=zero_column,
        atmospheric_aerosol_flux_gridcell_species=jnp.zeros(
            (number_of_gridcells, 14), dtype=jnp.float64
        ),
    )
    surface_inputs = SurfaceHydrologyInputs(
        time_step_seconds=timestep,
        snow_layer_count_column=snow.layer_count,
        soil_snow_thickness_column_level=thickness,
        soil_snow_ice_column_level=ice,
        snow_to_surface_water_flux_column=zero_column,
        flood_flux_column=zero_column,
        ground_evaporation_flux_column=zero_column,
        soil_evaporation_flux_column=zero_column,
        surface_water_evaporation_flux_column=zero_column,
        effective_snow_fraction_column=snow.effective_snow_fraction,
        surface_water_fraction_column=canopy.surface_water_fraction_column,
        snow_free_surface_water_fraction_column=(canopy.surface_water_fraction_without_snow_column),
        water_table_depth_column=jnp.asarray(values["ZWT"]),
        frost_table_depth_column=jnp.asarray(values["FROST_TABLE"]),
        perched_water_table_depth_column=jnp.asarray(values["ZWT_PERCH"]),
        maximum_saturated_fraction_column=jnp.asarray(fmax),
        surface_water_threshold_column=jnp.asarray(surface_threshold),
        topographic_slope_degrees_column=jnp.asarray(slope),
        soil_porosity_column_level=jnp.asarray(porosity),
        saturated_hydraulic_conductivity_column_level=jnp.asarray(
            hydraulic.saturated_hydraulic_conductivity_column
        ),
    )
    plant_inputs = SoilWaterPlantSinkInputs(
        soil_root_conductance_patch_level=jnp.zeros(
            (number_of_patches, NUMBER_OF_SOIL_LEVELS), dtype=jnp.float64
        ),
        root_fraction_patch_level=jnp.zeros(
            (number_of_patches, NUMBER_OF_GROUND_LEVELS), dtype=jnp.float64
        ),
        vegetation_transpiration_patch=zero_patch,
        vegetation_transpiration_column=zero_column,
        soil_water_potential_column_level=jnp.asarray(values["SMP"]),
        exposed_vegetation_flag_patch=jnp.zeros((number_of_patches,), dtype=jnp.int32),
        vegetation_water_potential_patch_segment=(
            physical.state.canopy_flux.plant.water_potential_segment
        ),
        soil_node_depth_column_level=node_depth[:, ground_slice],
    )
    movement_inputs = SoilWaterMovementInputs(
        soil_layer_count_column=soil_layer_count,
        layer_thickness_column_level=thickness[:, ground_slice],
        node_depth_column_level=node_depth[:, ground_slice],
        ice_fraction_column_level=jnp.asarray(ice_fraction),
        saturated_water_content_column_level=jnp.asarray(porosity),
        saturated_hydraulic_conductivity_column_level=jnp.asarray(
            hydraulic.saturated_hydraulic_conductivity_column
        ),
        clapp_hornberger_exponent_column_level=jnp.asarray(exponent),
        saturated_suction_column_level=jnp.asarray(saturated_suction),
        infiltration_flux_column=zero_column,
        root_water_sink_column_level=zero_soil,
        ice_water_mass_column_level=ice[:, ground_slice],
    )
    water_table_inputs = WaterTableCondensationInputs(
        soil_layer_count_column=soil_layer_count,
        snow_layer_count_column=snow.layer_count,
        layer_thickness_column_level=thickness[:, ground_slice],
        node_depth_column_level=node_depth[:, ground_slice],
        lower_interface_depth_column_level=lower_interface,
        soil_temperature_column_level=temperature[:, ground_slice],
        saturated_water_content_column_level=jnp.asarray(porosity),
        surface_water_fraction_column=canopy.surface_water_fraction_column,
        ground_dew_flux_column=zero_column,
        snow_dew_flux_column=zero_column,
    )
    old_ice_fraction = zero_soil_snow.at[:, :NUMBER_OF_SNOW_LEVELS].set(snow.old_ice_fraction)
    compaction_inputs = SnowCompactionInputs(
        snow_layer_count_column=snow.layer_count,
        snow_cover_shape_column=structure.snow_cover_shape_column,
        wind_speed_gridcell=jnp.zeros((number_of_gridcells,), dtype=jnp.float64),
        soil_snow_temperature_column_level=temperature,
        soil_snow_melt_flag_column_level=jnp.zeros_like(temperature, dtype=jnp.int32),
        effective_snow_fraction_column=snow.effective_snow_fraction,
        previous_snow_water_column_level=(snow.layer_ice + snow.layer_liquid),
        integrated_snow_column=snow.integrated_snow,
        old_ice_fraction_column_level=old_ice_fraction,
        soil_snow_ice_column_level=ice,
        soil_snow_liquid_column_level=liquid,
    )
    final_inputs = HydrologyDiagnosticsInputs(
        snow_layer_count_column=snow.layer_count,
        soil_snow_thickness_column_level=thickness,
        soil_snow_node_depth_column_level=node_depth,
        soil_snow_interface_depth_column_interface=interface_depth,
        soil_snow_temperature_column_level=temperature,
        soil_snow_ice_column_level=ice,
        soil_snow_liquid_column_level=liquid,
        effective_snow_fraction_column=snow.effective_snow_fraction,
        surface_water_fraction_column=canopy.surface_water_fraction_column,
        surface_water_temperature_column=standing.standing_water_temperature,
        soil_porosity_column_level=jnp.asarray(porosity),
        saturated_suction_column_level=jnp.asarray(saturated_suction),
        clapp_hornberger_exponent_column_level=jnp.asarray(exponent),
        minimum_soil_matrix_potential_column=jnp.full(
            (number_of_columns,), -1.0e8, dtype=jnp.float64
        ),
    )
    template = HydrologyNoDrainageInputs(
        snow_water=snow_inputs,
        surface_hydrology=surface_inputs,
        plant_sink=plant_inputs,
        soil_water_movement=movement_inputs,
        water_table_condensation=water_table_inputs,
        snow_compaction=compaction_inputs,
        final_diagnostics=final_inputs,
    )
    state = HydrologyNoDrainageState(
        snow_structure=snow_structure,
        snow_water=snow_water_state,
        surface_hydrology=surface_state,
        plant_sink=plant_state,
        soil_water_movement=movement_state,
        water_table_condensation=water_table_state,
        snow_capping=capping_state,
        snow_postprocess=postprocess_state,
        final_diagnostics=final_state,
    )
    return HydrologyDriverNativeData(
        topology=topology,
        inputs=HydrologyDriverInputs(
            template=template,
            atmospheric_aerosol_flux_gridcell_species=(
                load_atmospheric_aerosol_coupler_forcing(
                    coupler_restart, topology,
                    land_flux_area_factor=load_land_flux_area_factor(case, topology),
                )
            ),
            urban_surface_patch_index=urban_map.surface_patch_index,
        ),
        state=state,
        parameters=HydrologyNoDrainageParameters.from_case(case),
        land_restart_path=restart,
        coupler_restart_path=coupler_restart,
        surface_data_path=surface_path,
    )


__all__ = [
    "HydrologyDriverNativeData",
    "load_hydrology_driver_native_data",
]
