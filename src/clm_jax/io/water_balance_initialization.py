
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.canopy_temperature_restart import soil_layer_geometry_20sl_8p5m
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    SubgridTopology,
    configured_land_restart,
    load_subgrid_topology,
)
from clm_jax.physics.water_balance_initialization import (
    WaterBalanceInitializationInputs,
    WaterBalanceInitializationParameters,
    WaterBalanceInitializationState,
)

_RESTART_VARIABLES = (
    "SNLSNO",
    "WA",
    "ZWT",
    "H2OSFC",
    "H2OSNO",
    "H2OCAN",
    "SNOCAN",
    "H2OSOI_LIQ",
    "H2OSOI_ICE",
)


class WaterBalanceInitializationNativeData(NamedTuple):
    topology: SubgridTopology
    inputs: WaterBalanceInitializationInputs
    state: WaterBalanceInitializationState
    parameters: WaterBalanceInitializationParameters
    restart_path: Path


def load_water_balance_initialization_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
) -> WaterBalanceInitializationNativeData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    if str(general.get("soil_layerstruct", "")).strip() != "20SL_8.5m":
        raise ValueError("water-balance adapter requires soil_layerstruct='20SL_8.5m'")
    if general.get("use_fates") is not False:
        raise ValueError("native zero plant-storage default currently requires use_fates=.false.")
    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    dimensions = load_dimension_sizes(restart)
    expected_dimensions = {
        "levsno": 12,
        "levgrnd": 25,
        "levtot": 37,
        "levurb": 5,
    }
    for name, expected in expected_dimensions.items():
        if dimensions.get(name) != expected:
            raise ValueError(f"restart {name} dimension must be {expected}")
    if int(general.get("nlevsno", -1)) != dimensions["levsno"]:
        raise ValueError("case and restart disagree on snow-level count")

    topology = load_subgrid_topology(restart)
    column_count = topology.column.gridcell.shape[0]
    patch_count = topology.patch.gridcell.shape[0]
    if dimensions.get("column") != column_count or dimensions.get("pft") != patch_count:
        raise ValueError("restart dimensions disagree with native topology")
    values = load_native_variables(
        restart,
        list(_RESTART_VARIABLES),
        dtype=np.float64,
        masked_integer_fill_value=-9999,
    )
    layer_count = np.asarray(values["SNLSNO"], dtype=np.int32)
    if layer_count.shape != (column_count,) or np.any(
        (layer_count < -dimensions["levsno"]) | (layer_count > 0)
    ):
        raise ValueError("restart SNLSNO lies outside the native snow-level bounds")
    for name in ("WA", "ZWT", "H2OSFC", "H2OSNO"):
        if values[name].shape != (column_count,):
            raise ValueError(f"restart {name} must contain every column")
    for name in ("H2OCAN", "SNOCAN"):
        if values[name].shape != (patch_count,):
            raise ValueError(f"restart {name} must contain every patch")
    level_shape = (column_count, dimensions["levtot"])
    for name in ("H2OSOI_LIQ", "H2OSOI_ICE"):
        if values[name].shape != level_shape:
            raise ValueError(f"restart {name} must use native (column, levtot) axes")

    _, interface_depth, _ = soil_layer_geometry_20sl_8p5m()
    soil_bottom = np.full(
        column_count,
        interface_depth[20],
        dtype=np.float64,
    )
    parameters = WaterBalanceInitializationParameters(
        maximum_patches_per_column=int(general["maxpatch_pft"]),
    )
    return WaterBalanceInitializationNativeData(
        topology=topology,
        inputs=WaterBalanceInitializationInputs(
            snow_layer_count_column=jnp.asarray(layer_count),
            water_table_depth_column=jnp.asarray(values["ZWT"]),
            soil_bottom_interface_depth_column=jnp.asarray(soil_bottom),
            surface_water_column=jnp.asarray(values["H2OSFC"]),
            snow_water_equivalent_column=jnp.asarray(values["H2OSNO"]),
            canopy_water_patch=jnp.asarray(values["H2OCAN"]),
            canopy_snow_patch=jnp.asarray(values["SNOCAN"]),
            total_plant_stored_water_column=jnp.zeros(
                column_count,
                dtype=jnp.float64,
            ),
            soil_snow_liquid_water_column_level=jnp.asarray(values["H2OSOI_LIQ"]),
            soil_snow_ice_water_column_level=jnp.asarray(values["H2OSOI_ICE"]),
        ),
        state=WaterBalanceInitializationState(
            aquifer_water_column=jnp.asarray(values["WA"]),
            beginning_water_mass_column=jnp.zeros(
                column_count,
                dtype=jnp.float64,
            ),
        ),
        parameters=parameters,
        restart_path=restart,
    )


__all__ = [
    "WaterBalanceInitializationNativeData",
    "load_water_balance_initialization_native_data",
]
