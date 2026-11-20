
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.canopy_restart import load_canopy_hydrology_restart
from clm_jax.io.canopy_temperature_restart import (
    lake_soil_thermal_properties_20sl_8p5m,
    soil_layer_geometry_20sl_8p5m,
)
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.lake_flux_driver import (
    lake_node_depth_from_thickness,
    load_lake_flux_driver_native_data,
)
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.lake_temperature import (
    NUMBER_OF_GROUND_LEVELS,
    LakeColumnTemperatureState,
    LakeTemperatureGeometry,
)
from clm_jax.physics.lake_temperature_driver import LakeTemperatureDriverInputs


class LakeTemperatureDriverNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: LakeTemperatureDriverInputs
    state: LakeColumnTemperatureState
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path


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


def _selected_grid_field(
    topology: SubgridTopology,
    gridcell: np.ndarray,
    source: np.ndarray,
) -> np.ndarray:
    values = np.asarray(source, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("lake surface field must have latitude/longitude dimensions")
    latitude = np.asarray(topology.grid.latitude_index, dtype=np.int64)[gridcell]
    longitude = np.asarray(topology.grid.longitude_index, dtype=np.int64)[gridcell]
    if (
        np.any(latitude < 0)
        or np.any(latitude >= values.shape[0])
        or np.any(longitude < 0)
        or np.any(longitude >= values.shape[1])
    ):
        raise ValueError("native lake grid lies outside configured surface data")
    return values[latitude, longitude]


def _restart_float(values: dict[str, jnp.ndarray], name: str) -> np.ndarray:
    value = np.asarray(values[name], dtype=np.float64)
    return np.where(np.isnan(value), np.float64(CLM_SPECIAL_VALUE), value)


def load_lake_temperature_driver_native_data(
    case: CimeCase,
    restart_path: str | Path,
) -> LakeTemperatureDriverNativeData:

    restart = Path(restart_path).expanduser().resolve()
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    lake_flux = load_lake_flux_driver_native_data(case, restart)
    canopy = load_canopy_hydrology_restart(case, restart)
    topology = lake_flux.topology
    if canopy.topology.column.gridcell.shape != topology.column.gridcell.shape:
        raise ValueError("canopy and lake native adapters disagree on topology")

    native_map = lake_flux.inputs.native_map
    column = np.asarray(native_map.column_index, dtype=np.int64)
    gridcell = np.asarray(native_map.gridcell_index, dtype=np.int64)
    lake_count = column.size
    surface_data = lake_flux.surface_data_path
    parameter_data = _configured_clm_input(case, "paramfile")
    surface_values = load_native_variables(
        surface_data,
        ["PCT_SAND", "PCT_CLAY", "ORGANIC"],
    )
    parameter_values = load_native_variables(parameter_data, ["organic_max"])
    organic_maximum = float(np.asarray(parameter_values["organic_max"]).item())
    thermal = lake_soil_thermal_properties_20sl_8p5m(
        topology,
        surface_values,
        organic_maximum,
        column,
    )

    try:
        extinction_source = load_native_variables(surface_data, ["ETALAKE"])["ETALAKE"]
    except (IndexError, KeyError):
        extinction = np.full((lake_count,), np.float64(-1.0), dtype=np.float64)
    else:
        extinction = _selected_grid_field(
            topology,
            gridcell,
            np.asarray(extinction_source),
        )
        extinction = np.where(np.isfinite(extinction), extinction, np.float64(-1.0))

    ground_node, ground_interface, ground_thickness = soil_layer_geometry_20sl_8p5m()
    ground_node = np.broadcast_to(
        ground_node[None, :],
        (lake_count, NUMBER_OF_GROUND_LEVELS),
    ).copy()
    ground_interface = np.broadcast_to(
        ground_interface[None, 1:],
        (lake_count, NUMBER_OF_GROUND_LEVELS),
    ).copy()
    ground_thickness = np.broadcast_to(
        ground_thickness[None, :],
        (lake_count, NUMBER_OF_GROUND_LEVELS),
    ).copy()
    snow = canopy.state.snow
    snow_thickness = np.asarray(snow.layer_thickness, dtype=np.float64)[column]
    snow_node = np.asarray(snow.layer_depth, dtype=np.float64)[column]
    snow_interface = np.asarray(snow.layer_interface_depth, dtype=np.float64)[column, 1:]
    lake_thickness = np.asarray(
        lake_flux.inputs.lake_layer_thickness_lake_level,
        dtype=np.float64,
    )
    lake_node = lake_node_depth_from_thickness(lake_thickness)
    geometry = LakeTemperatureGeometry(
        soil_snow_layer_thickness=jnp.asarray(
            np.concatenate((snow_thickness, ground_thickness), axis=1)
        ),
        soil_snow_node_depth=jnp.asarray(np.concatenate((snow_node, ground_node), axis=1)),
        soil_snow_interface_depth=jnp.asarray(
            np.concatenate((snow_interface, ground_interface), axis=1)
        ),
        lake_layer_thickness=jnp.asarray(lake_thickness),
        lake_node_depth=jnp.asarray(lake_node),
        lake_depth=jnp.asarray(lake_flux.inputs.lake_depth),
        light_extinction_coefficient=jnp.asarray(extinction),
        saturated_water_content=jnp.asarray(thermal.saturated_water_content),
        saturated_thermal_conductivity=jnp.asarray(thermal.saturated_thermal_conductivity),
        mineral_thermal_conductivity=jnp.asarray(thermal.mineral_thermal_conductivity),
        dry_thermal_conductivity=jnp.asarray(thermal.dry_thermal_conductivity),
        solid_volumetric_heat_capacity=jnp.asarray(thermal.solid_volumetric_heat_capacity),
    )

    restart_values = load_native_variables(
        restart,
        [
            "SNLSNO",
            "SNOW_DEPTH",
            "H2OSNO",
            "T_SOISNO",
            "H2OSOI_LIQ",
            "H2OSOI_ICE",
            "T_LAKE",
            "LAKE_ICEFRAC",
            "SAVEDTKE1",
        ],
    )
    soil_temperature = _restart_float(restart_values, "T_SOISNO")[column]
    soil_liquid = _restart_float(restart_values, "H2OSOI_LIQ")[column]
    soil_ice = _restart_float(restart_values, "H2OSOI_ICE")[column]
    lake_temperature = _restart_float(restart_values, "T_LAKE")[column]
    lake_ice = _restart_float(restart_values, "LAKE_ICEFRAC")[column]
    saved_tke = _restart_float(restart_values, "SAVEDTKE1")[column]
    for name, value in (
        ("T_LAKE", lake_temperature),
        ("LAKE_ICEFRAC", lake_ice),
        ("SAVEDTKE1", saved_tke),
    ):
        if np.any(~np.isfinite(value)) or np.any(np.abs(value) >= 1.0e35):
            raise ValueError(f"active lake restart field {name} contains missing values")

    snow_old_ice = np.asarray(snow.old_ice_fraction, dtype=np.float64)[column]
    old_ice = np.concatenate(
        (
            snow_old_ice,
            np.zeros((lake_count, NUMBER_OF_GROUND_LEVELS), dtype=np.float64),
        ),
        axis=1,
    )
    zeros = jnp.zeros((lake_count,), dtype=jnp.float64)
    state = LakeColumnTemperatureState(
        snow_layer_count=jnp.asarray(np.asarray(restart_values["SNLSNO"], dtype=np.int32)[column]),
        snow_depth=jnp.asarray(_restart_float(restart_values, "SNOW_DEPTH")[column]),
        snow_water=jnp.asarray(_restart_float(restart_values, "H2OSNO")[column]),
        soil_snow_temperature=jnp.asarray(soil_temperature),
        soil_snow_liquid_water=jnp.asarray(soil_liquid),
        soil_snow_ice_water=jnp.asarray(soil_ice),
        soil_snow_old_ice_fraction=jnp.asarray(old_ice),
        soil_snow_phase_flag=jnp.zeros(soil_temperature.shape, dtype=jnp.int32),
        lake_temperature=jnp.asarray(lake_temperature),
        lake_ice_fraction=jnp.asarray(lake_ice),
        saved_top_eddy_conductivity=jnp.asarray(saved_tke),
        surface_absorption_fraction=zeros,
        lake_surface_ice_fraction=zeros,
        lake_ice_thickness=zeros,
        lake_resistance=zeros,
        ground_methane_conductance=zeros,
    )
    return LakeTemperatureDriverNativeData(
        topology=topology,
        inputs=LakeTemperatureDriverInputs(
            native_map=native_map,
            geometry=geometry,
        ),
        state=state,
        restart_path=restart,
        surface_data_path=surface_data,
        parameter_data_path=parameter_data,
    )


__all__ = [
    "LakeTemperatureDriverNativeData",
    "load_lake_temperature_driver_native_data",
]
