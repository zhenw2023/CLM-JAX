
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_GLACIER,
    LANDUNIT_SOIL,
    configured_land_restart,
    load_subgrid_topology,
)
from clm_jax.physics.lnd2glc import LandToGlacierInputs


class LandToGlacierNativeData(NamedTuple):
    inputs: LandToGlacierInputs
    number_of_gridcells: int
    maximum_glacier_classes: int
    restart_path: Path
    surface_data_path: Path


def _surface_data_path(case: CimeCase) -> Path:
    general = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    declared = general.get("fsurdat")
    if not isinstance(declared, str):
        raise ValueError("CLM namelist lacks a scalar fsurdat path")
    path = Path(case.expand(declared)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"surface dataset does not exist: {path}")
    return path


def load_land_to_glacier_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
) -> LandToGlacierNativeData:

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    dimensions = load_dimension_sizes(restart)
    for name in ("gridcell", "column", "glc_nec", "levsno", "levtot"):
        if name not in dimensions:
            raise ValueError(f"restart lacks required {name} dimension")
    topology = load_subgrid_topology(restart)
    number_of_gridcells = dimensions["gridcell"]
    maximum_classes = dimensions["glc_nec"]
    if topology.grid.longitude_degrees.shape != (number_of_gridcells,):
        raise ValueError("restart grid dimension disagrees with subgrid topology")

    values = load_native_variables(restart, ["T_SOISNO"])
    temperature = np.asarray(values["T_SOISNO"], dtype=np.float64)
    elevation = np.asarray(topology.column.elevation, dtype=np.float64)
    if temperature.shape != (dimensions["column"], dimensions["levtot"]):
        raise ValueError("restart T_SOISNO must use native (column, levtot) axes")
    if elevation.shape != (dimensions["column"],):
        raise ValueError("restart TOPO_COL must use the native column axis")
    top_ground_temperature = temperature[:, dimensions["levsno"]]

    surface_path = _surface_data_path(case)
    glacier_region = np.asarray(
        load_native_variables(surface_path, ["GLACIER_REGION"])["GLACIER_REGION"],
        dtype=np.int32,
    )
    longitude_index = np.asarray(topology.grid.longitude_index, dtype=np.int64)
    latitude_index = np.asarray(topology.grid.latitude_index, dtype=np.int64)
    if glacier_region.ndim != 2 or np.any(latitude_index >= glacier_region.shape[0]) or np.any(
        longitude_index >= glacier_region.shape[1]
    ):
        raise ValueError("surface GLACIER_REGION is incompatible with the restart grid")
    native_region = glacier_region[latitude_index, longitude_index]
    behavior = dict(
        read_case_namelist_group(case, "lnd_in", "clm_glacier_behavior")
    ).get("glacier_region_melt_behavior")
    if not isinstance(behavior, tuple) or np.any(native_region < 0) or np.any(
        native_region >= len(behavior)
    ):
        raise ValueError("glacier-region melt behavior does not cover the surface map")
    replace_melt = np.asarray(
        [str(behavior[index]).strip().lower() == "replaced_by_ice" for index in native_region],
        dtype=bool,
    )

    column_gridcell = np.asarray(topology.column.gridcell, dtype=np.int32)
    column_landunit_type = np.asarray(topology.column.landunit_type, dtype=np.int32)
    column_active = np.asarray(topology.column.active, dtype=bool)
    do_smb = column_active & replace_melt[column_gridcell] & (
        (column_landunit_type == LANDUNIT_GLACIER)
        | (column_landunit_type == LANDUNIT_SOIL)
    )
    elevation_class = np.where(
        column_landunit_type == LANDUNIT_GLACIER,
        np.asarray(topology.column.column_type, dtype=np.int32) - LANDUNIT_GLACIER * 100,
        0,
    )
    if np.any(
        do_smb
        & ((elevation_class < 0) | (elevation_class > maximum_classes))
    ):
        raise ValueError("an SMB column has an invalid glacier elevation class")
    flat = column_gridcell[do_smb] * (maximum_classes + 1) + elevation_class[do_smb]
    if np.unique(flat).size != flat.size:
        raise ValueError("native topology violates update_lnd2glc's unique-assignment invariant")

    inputs = LandToGlacierInputs(
        column_gridcell=topology.column.gridcell,
        column_landunit_type=topology.column.landunit_type,
        column_type=topology.column.column_type,
        column_weight_on_gridcell=topology.column.weight_on_gridcell,
        landunit_gridcell=topology.landunit.gridcell,
        landunit_type=topology.landunit.landunit_type,
        landunit_weight_on_gridcell=topology.landunit.weight_on_gridcell,
        do_surface_mass_balance=jnp.asarray(do_smb),
        surface_temperature=jnp.asarray(top_ground_temperature),
        elevation=jnp.asarray(elevation),
        ice_flux=jnp.zeros(dimensions["column"], dtype=jnp.float64),
        initialization=jnp.asarray(True),
    )
    return LandToGlacierNativeData(
        inputs=inputs,
        number_of_gridcells=number_of_gridcells,
        maximum_glacier_classes=maximum_classes,
        restart_path=restart,
        surface_data_path=surface_path,
    )


__all__ = ["LandToGlacierNativeData", "load_land_to_glacier_native_data"]
