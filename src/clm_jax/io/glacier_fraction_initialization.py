
from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.forcing.remap import load_sparse_mapping
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.coupler import load_glacier_coupler_state
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import (
    SubgridTopology,
    configured_land_restart,
    load_subgrid_topology,
)
from clm_jax.physics.glacier_fraction_initialization import (
    GlacierFractionInitializationInputs,
    GlacierFractionInitializationResult,
)
from clm_jax.physics.topography import map_glacier_state_to_land, select_land_gridcells


class GlacierFractionInitializationNativeData(NamedTuple):

    inputs: GlacierFractionInitializationInputs
    glc_do_dynglacier: bool
    run_zero_weight_urban: bool
    all_active: bool
    restart_path: Path
    coupler_restart_path: Path
    mapping_path: Path
    surface_data_path: Path


def apply_glacier_fraction_initialization(
    topology: SubgridTopology,
    result: GlacierFractionInitializationResult,
) -> SubgridTopology:

    return topology._replace(
        landunit=topology.landunit._replace(
            active=result.landunit_active,
            weight_on_gridcell=result.landunit_weight_on_gridcell,
        ),
        column=topology.column._replace(
            active=result.column_active,
            weight_on_gridcell=result.column_weight_on_gridcell,
            weight_on_landunit=result.column_weight_on_landunit,
        ),
        patch=topology.patch._replace(
            active=result.patch_active,
            weight_on_gridcell=result.patch_weight_on_gridcell,
            weight_on_landunit=result.patch_weight_on_landunit,
            weight_on_column=result.patch_weight_on_column,
        ),
    )


def _configured_input_path(case: CimeCase, component: str, key: str) -> Path:
    for entry in case.input_file_entries():
        if entry.component == component and entry.key == key and entry.path.is_file():
            return entry.path.resolve()
    raise FileNotFoundError(f"case has no readable {component}.{key} input")


def _surface_data_path(case: CimeCase) -> Path:
    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    declared = general.get("fsurdat")
    if not isinstance(declared, str):
        raise ValueError("CLM namelist lacks a scalar fsurdat path")
    path = Path(case.expand(declared)).expanduser()
    if not path.is_absolute():
        path = case.root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured surface dataset does not exist: {path}")
    return path


def _coupler_restart_path(case: CimeCase) -> Path:
    pointer = case.path_value("RUNDIR") / "rpointer.cpl"
    if pointer.is_file():
        for line in pointer.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if not name:
                continue
            candidate = Path(name)
            if not candidate.is_absolute():
                candidate = pointer.parent / candidate
            if candidate.is_file():
                return candidate.resolve()
            break
    raise FileNotFoundError(
        "no readable coupler restart is configured; pass coupler_restart_path explicitly"
    )


@cache
def load_glacier_fraction_initialization_native_data(
    case: CimeCase,
    *,
    restart_path: str | Path | None = None,
    coupler_restart_path: str | Path | None = None,
    mapping_path: str | Path | None = None,
) -> GlacierFractionInitializationNativeData:

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    coupler_restart = (
        _coupler_restart_path(case)
        if coupler_restart_path is None
        else Path(coupler_restart_path).expanduser().resolve()
    )
    mapping_file = (
        _configured_input_path(case, "cpl", "glc2lnd_smapname")
        if mapping_path is None
        else Path(mapping_path).expanduser().resolve()
    )
    for label, path in (
        ("land restart", restart),
        ("coupler restart", coupler_restart),
        ("GLC-to-land mapping", mapping_file),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    topology = load_subgrid_topology(restart)
    mapping = load_sparse_mapping(mapping_file)
    from clm_jax.io.glacier_mapping import prepare_glacier_mapping

    mapping = prepare_glacier_mapping(case, mapping, "GLC2LND_SMAPTYPE")
    glacier_full_grid = map_glacier_state_to_land(
        load_glacier_coupler_state(coupler_restart),
        mapping,
    )
    surface_path = _surface_data_path(case)
    surface_region = np.asarray(
        load_native_variables(surface_path, ["GLACIER_REGION"])["GLACIER_REGION"],
        dtype=np.int32,
    )
    if surface_region.ndim != 2 or surface_region.size != mapping.destination_size:
        raise ValueError("surface grid and GLC-to-land mapping destination disagree")
    longitude_index = np.asarray(topology.grid.longitude_index, dtype=np.int64)
    latitude_index = np.asarray(topology.grid.latitude_index, dtype=np.int64)
    if (
        np.any(longitude_index < 0)
        or np.any(longitude_index >= surface_region.shape[1])
        or np.any(latitude_index < 0)
        or np.any(latitude_index >= surface_region.shape[0])
    ):
        raise ValueError("restart grid indices lie outside the configured surface grid")
    destination_index = jnp.asarray(
        latitude_index * surface_region.shape[1] + longitude_index,
        dtype=jnp.int32,
    )
    glacier = select_land_gridcells(glacier_full_grid, destination_index)

    glacier_group = read_case_namelist_group(
        case,
        "lnd_in",
        "clm_glacier_behavior",
    )
    behavior_value = glacier_group.get("glacier_region_behavior")
    behaviors = behavior_value if isinstance(behavior_value, tuple) else (behavior_value,)
    if not behaviors or not all(isinstance(value, str) for value in behaviors):
        raise TypeError("clm_glacier_behavior.glacier_region_behavior must contain strings")
    region = surface_region[latitude_index, longitude_index]
    if np.any(region < 0) or np.any(region >= len(behaviors)):
        raise ValueError("surface GLACIER_REGION is not covered by the configured behavior")
    virtual_by_region = np.asarray(
        [str(value).strip().lower() == "virtual" for value in behaviors],
        dtype=bool,
    )
    virtual = virtual_by_region[region]

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    glc_do_dynglacier = general.get("glc_do_dynglacier")
    run_zero_weight_urban = general.get("run_zero_weight_urban", False)
    all_active = general.get("all_active", False)
    if not isinstance(glc_do_dynglacier, bool):
        raise TypeError("clm_inparm.glc_do_dynglacier must be logical")
    if not isinstance(run_zero_weight_urban, bool) or not isinstance(all_active, bool):
        raise TypeError("subgrid active controls must be logical")

    inputs = GlacierFractionInitializationInputs(
        landunit_gridcell=topology.landunit.gridcell,
        landunit_type=topology.landunit.landunit_type,
        landunit_weight_on_gridcell=topology.landunit.weight_on_gridcell,
        column_gridcell=topology.column.gridcell,
        column_landunit=topology.column.landunit,
        column_landunit_type=topology.column.landunit_type,
        column_type=topology.column.column_type,
        column_weight_on_landunit=topology.column.weight_on_landunit,
        patch_column=topology.patch.column,
        patch_weight_on_column=topology.patch.weight_on_column,
        glacier_ice_mask=glacier.ice_mask,
        glacier_fraction_by_class=glacier.fraction_by_class,
        has_virtual_columns_gridcell=jnp.asarray(virtual, dtype=jnp.bool_),
    )
    return GlacierFractionInitializationNativeData(
        inputs=inputs,
        glc_do_dynglacier=glc_do_dynglacier,
        run_zero_weight_urban=run_zero_weight_urban,
        all_active=all_active,
        restart_path=restart,
        coupler_restart_path=coupler_restart,
        mapping_path=mapping_file,
        surface_data_path=surface_path,
    )


__all__ = [
    "GlacierFractionInitializationNativeData",
    "apply_glacier_fraction_initialization",
    "load_glacier_fraction_initialization_native_data",
]
