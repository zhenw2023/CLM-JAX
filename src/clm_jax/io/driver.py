
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_LAKE,
    SubgridTopology,
    configured_land_restart,
    load_subgrid_topology,
)

_RESTART_VARIABLES = (
    "SNLSNO",
    "H2OSNO",
    "H2OSOI_ICE",
    "H2OSOI_LIQ",
    "FRAC_VEG_NOSNO_ALB",
)


class DriverInitializationNativeData(NamedTuple):

    topology: SubgridTopology
    intracellular_sunlit_co2: jnp.ndarray
    intracellular_shaded_co2: jnp.ndarray
    snow_water_equivalent: jnp.ndarray
    snow_layer_count: jnp.ndarray
    snow_layer_ice: jnp.ndarray
    snow_layer_liquid: jnp.ndarray
    previous_snow_ice_fraction: jnp.ndarray
    patch_active: jnp.ndarray
    vegetation_exposed_for_albedo: jnp.ndarray
    non_lake_column: jnp.ndarray
    number_of_snow_levels: int
    restart_path: Path


def load_driver_initialization_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
) -> DriverInitializationNativeData:

    general = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    if general.get("use_cn") is not True:
        raise ValueError("reference driver adapter requires the active BGC configuration")
    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")

    dimensions = load_dimension_sizes(restart)
    required = ("column", "pft", "levsno", "levgrnd", "levtot", "levcan")
    missing = [name for name in required if name not in dimensions]
    if missing:
        raise ValueError(f"restart lacks dimensions {missing}: {restart}")
    snow_levels = dimensions["levsno"]
    if dimensions["levtot"] != snow_levels + dimensions["levgrnd"]:
        raise ValueError("restart levtot must concatenate snow and ground levels")
    if int(general.get("nlevsno", -1)) != snow_levels:
        raise ValueError("case and restart disagree on nlevsno")

    topology = load_subgrid_topology(restart)
    number_of_columns = topology.column.gridcell.shape[0]
    number_of_patches = topology.patch.gridcell.shape[0]
    if dimensions["column"] != number_of_columns or dimensions["pft"] != number_of_patches:
        raise ValueError("restart dimensions disagree with native subgrid topology")
    values = load_native_variables(
        restart,
        list(_RESTART_VARIABLES),
        masked_integer_fill_value=-9999,
        masked_floating_fill_value=float(CLM_SPECIAL_VALUE),
    )
    layer_count = np.asarray(values["SNLSNO"], dtype=np.int32)
    layer_ice = np.asarray(values["H2OSOI_ICE"], dtype=np.float64)
    layer_liquid = np.asarray(values["H2OSOI_LIQ"], dtype=np.float64)
    if layer_count.shape != (number_of_columns,) or np.any(
        (layer_count < -snow_levels) | (layer_count > 0)
    ):
        raise ValueError("restart SNLSNO lies outside the native snow-level bounds")
    level_shape = (number_of_columns, dimensions["levtot"])
    if layer_ice.shape != level_shape or layer_liquid.shape != level_shape:
        raise ValueError("restart water arrays must use native (column, levtot) axes")
    if values["H2OSNO"].shape != (number_of_columns,):
        raise ValueError("restart H2OSNO must use the native column axis")
    if values["FRAC_VEG_NOSNO_ALB"].shape != (number_of_patches,):
        raise ValueError("restart FRAC_VEG_NOSNO_ALB must use the native patch axis")

    column_active = np.asarray(topology.column.active, dtype=bool)
    column_landunit_type = np.asarray(topology.column.landunit_type, dtype=np.int32)
    non_lake = column_active & (column_landunit_type != LANDUNIT_LAKE)
    level_number = np.arange(-snow_levels + 1, 1, dtype=np.int32)
    active_snow = non_lake[:, None] & (
        level_number[None, :] >= layer_count[:, None] + 1
    )
    snow_water = layer_ice[:, :snow_levels] + layer_liquid[:, :snow_levels]
    if np.any(~np.isfinite(snow_water[active_snow])) or np.any(
        snow_water[active_snow] <= 0.0
    ):
        raise ValueError("an active nonlake snow layer has invalid water mass")

    transient_co2 = jnp.full(
        (number_of_patches, dimensions["levcan"]),
        jnp.float64(CLM_SPECIAL_VALUE),
    )
    return DriverInitializationNativeData(
        topology=topology,
        intracellular_sunlit_co2=transient_co2,
        intracellular_shaded_co2=transient_co2,
        snow_water_equivalent=jnp.asarray(values["H2OSNO"], dtype=jnp.float64),
        snow_layer_count=jnp.asarray(layer_count),
        snow_layer_ice=jnp.asarray(layer_ice),
        snow_layer_liquid=jnp.asarray(layer_liquid),
        previous_snow_ice_fraction=jnp.full(
            level_shape,
            jnp.float64(CLM_SPECIAL_VALUE),
        ),
        patch_active=topology.patch.active,
        vegetation_exposed_for_albedo=jnp.asarray(
            values["FRAC_VEG_NOSNO_ALB"],
            dtype=jnp.int32,
        ),
        non_lake_column=jnp.asarray(non_lake),
        number_of_snow_levels=snow_levels,
        restart_path=restart,
    )


__all__ = [
    "DriverInitializationNativeData",
    "load_driver_initialization_native_data",
]
