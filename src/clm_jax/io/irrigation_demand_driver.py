
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.canopy_hydraulic_inputs import bedrock_level_count_column
from clm_jax.io.canopy_temperature_driver import (
    load_canopy_temperature_driver_native_data,
)
from clm_jax.io.canopy_temperature_restart import soil_layer_geometry_20sl_8p5m
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.canopy_temperature import NUMBER_OF_SOIL_LEVELS
from clm_jax.physics.irrigation import IrrigationDemandParameters
from clm_jax.physics.irrigation_demand_driver import (
    WILTING_POINT_SOIL_MATRIC_POTENTIAL,
    IrrigationDemandDriverInputs,
    clapp_hornberger_relative_saturation,
)


class IrrigationDemandDriverNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: IrrigationDemandDriverInputs
    parameters: IrrigationDemandParameters
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path


def _configured_clm_input_path(case: CimeCase, name: str) -> Path:
    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    declared = general.get(name)
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(f"CLM clm_inparm lacks a scalar {name} path")
    path = Path(case.expand(declared)).expanduser()
    if not path.is_absolute():
        path = case.root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM {name} is absent: {path}")
    return path


def _gridcell_area(
    topology: SubgridTopology,
    surface_data_path: Path,
) -> np.ndarray:
    source = np.asarray(
        load_native_variables(surface_data_path, ["AREA"], dtype=np.float64)["AREA"],
        dtype=np.float64,
    )
    if source.ndim != 2:
        raise ValueError("fsurdat AREA must have dimensions (lat, lon)")
    latitude = np.asarray(topology.grid.latitude_index, dtype=np.int64)
    longitude = np.asarray(topology.grid.longitude_index, dtype=np.int64)
    if (
        np.any(latitude < 0)
        or np.any(latitude >= source.shape[0])
        or np.any(longitude < 0)
        or np.any(longitude >= source.shape[1])
    ):
        raise ValueError("native grid indices lie outside fsurdat AREA")
    area = source[latitude, longitude]
    if np.any(~np.isfinite(area)) or np.any(area <= np.float64(0.0)):
        raise ValueError("active native grid contains invalid fsurdat AREA")
    return area


def _irrigated_patch(
    topology: SubgridTopology,
    parameter_data_path: Path,
) -> np.ndarray:
    irrigated_pft = np.asarray(
        load_native_variables(
            parameter_data_path,
            ["irrigated"],
            dtype=np.float64,
        )["irrigated"],
        dtype=np.float64,
    )
    if irrigated_pft.ndim != 1:
        raise ValueError("CLM parameter irrigated flag must be one-dimensional")
    if np.any((irrigated_pft != np.float64(0.0)) & (irrigated_pft != np.float64(1.0))):
        raise ValueError("CLM parameter irrigated flag contains values other than zero or one")
    vegetation_type = np.asarray(topology.patch.vegetation_type, dtype=np.int64)
    if np.any((vegetation_type < 0) | (vegetation_type >= irrigated_pft.size)):
        raise ValueError("native patch PFT type lies outside the irrigated parameter array")
    return irrigated_pft[vegetation_type] == np.float64(1.0)


def _case_runoff_prognostic(case: CimeCase) -> bool:
    component = case.value("COMP_ROF", "").strip().lower()
    if component in {"", "srof", "drof"}:
        return False
    if component == "mosart":
        return case.value("MOSART_MODE", "ACTIVE").strip().upper() == "ACTIVE"
    raise ValueError(f"stage-290 adapter does not recognize COMP_ROF={component!r}")


def load_irrigation_demand_driver_native_data(
    case: CimeCase,
    restart_path: str | Path,
    *,
    river_water_volume_gridcell: jnp.ndarray | None = None,
    runoff_prognostic: bool | None = None,
) -> IrrigationDemandDriverNativeData:

    restart = Path(restart_path).expanduser().resolve()
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    if general.get("irrigate") is not True:
        raise ValueError("stage-290 native adapter requires irrigate=.true.")
    soil_movement = read_case_namelist_group(
        case,
        "lnd_in",
        "soilwater_movement_inparm",
    )
    if int(soil_movement.get("soilwater_movement_method", -1)) != 1:
        raise ValueError("stage-290 adapter requires the Clapp-Hornberger retention curve")

    temperature = load_canopy_temperature_driver_native_data(case, restart)
    topology = temperature.topology
    surface_data = _configured_clm_input_path(case, "fsurdat")
    parameter_data = _configured_clm_input_path(case, "paramfile")
    parameters = IrrigationDemandParameters.from_case(case)

    column_count = topology.column.gridcell.shape[0]
    gridcell_count = topology.grid.longitude_degrees.shape[0]
    soil_shape = (column_count, NUMBER_OF_SOIL_LEVELS)
    saturated = jnp.asarray(
        temperature.parameters.saturated_water_content_column[
            :, :NUMBER_OF_SOIL_LEVELS
        ],
        dtype=jnp.float64,
    )
    clapp_b = jnp.asarray(
        temperature.parameters.clapp_hornberger_b_column[
            :, :NUMBER_OF_SOIL_LEVELS
        ],
        dtype=jnp.float64,
    )
    saturated_suction = jnp.asarray(
        temperature.parameters.saturated_suction_column[
            :, :NUMBER_OF_SOIL_LEVELS
        ],
        dtype=jnp.float64,
    )
    thickness = jnp.asarray(
        temperature.parameters.ground_layer_thickness_column[
            :, :NUMBER_OF_SOIL_LEVELS
        ],
        dtype=jnp.float64,
    )
    if not (
        saturated.shape
        == clapp_b.shape
        == saturated_suction.shape
        == thickness.shape
        == soil_shape
    ):
        raise ValueError("stage-240 soil parameters do not cover all native soil columns")

    node, _, _ = soil_layer_geometry_20sl_8p5m()
    depth = jnp.broadcast_to(
        jnp.asarray(node[:NUMBER_OF_SOIL_LEVELS], dtype=jnp.float64)[None, :],
        soil_shape,
    )
    wilting = clapp_hornberger_relative_saturation(
        WILTING_POINT_SOIL_MATRIC_POTENTIAL,
        saturated_suction,
        clapp_b,
    )
    target = clapp_hornberger_relative_saturation(
        parameters.target_soil_matric_potential,
        saturated_suction,
        clapp_b,
    )

    is_runoff_prognostic = (
        _case_runoff_prognostic(case)
        if runoff_prognostic is None
        else bool(runoff_prognostic)
    )
    if river_water_volume_gridcell is None:
        if parameters.limit_irrigation_if_runoff_prognostic and is_runoff_prognostic:
            raise ValueError(
                "river-volume-limited irrigation requires explicit remapped main-channel volume"
            )
        river_volume = jnp.zeros(gridcell_count, dtype=jnp.float64)
    else:
        river_volume = jnp.asarray(river_water_volume_gridcell, dtype=jnp.float64)
        if river_volume.shape != (gridcell_count,):
            raise ValueError("river water volume must contain every native land gridcell")

    return IrrigationDemandDriverNativeData(
        topology=topology,
        inputs=IrrigationDemandDriverInputs(
            irrigated_patch=jnp.asarray(
                _irrigated_patch(topology, parameter_data),
                dtype=jnp.bool_,
            ),
            saturated_water_content_column_level=saturated,
            relative_saturation_wilting_point_column_level=wilting,
            relative_saturation_target_column_level=target,
            soil_layer_depth_column_level=depth,
            soil_layer_thickness_column_level=thickness,
            bedrock_soil_layer_count_column=jnp.asarray(
                bedrock_level_count_column(topology, surface_data),
                dtype=jnp.int32,
            ),
            river_water_volume_gridcell=river_volume,
            gridcell_area_square_kilometers=jnp.asarray(
                _gridcell_area(topology, surface_data),
                dtype=jnp.float64,
            ),
            runoff_prognostic=jnp.asarray(is_runoff_prognostic, dtype=jnp.bool_),
        ),
        parameters=parameters,
        restart_path=restart,
        surface_data_path=surface_data,
        parameter_data_path=parameter_data,
    )


__all__ = [
    "IrrigationDemandDriverNativeData",
    "load_irrigation_demand_driver_native_data",
]
