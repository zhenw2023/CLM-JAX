
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.canopy_temperature_restart import soil_layer_geometry_20sl_8p5m
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.native_templates import native_template
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    SubgridTopology,
    configured_land_restart,
    load_subgrid_topology,
)
from clm_jax.physics.carbon_nitrogen_balance_initialization import (
    CARBON_DISPLAYED_COMPONENTS,
    CARBON_STORED_COMPONENTS,
    NITROGEN_DISPLAYED_COMPONENTS,
    NITROGEN_STORED_COMPONENTS,
    NUMBER_OF_DECOMPOSITION_LEVELS,
    NUMBER_OF_DECOMPOSITION_POOLS,
    CarbonNitrogenBalanceInitializationInputs,
    CarbonNitrogenBalanceInitializationParameters,
    CarbonNitrogenBalanceInitializationState,
    CarbonNitrogenStateSummaries,
)

_CARBON_POOL_VARIABLES = (
    "litr1c_vr",
    "litr2c_vr",
    "litr3c_vr",
    "cwdc_vr",
    "soil1c_vr",
    "soil2c_vr",
    "soil3c_vr",
)
_NITROGEN_POOL_VARIABLES = (
    "litr1n_vr",
    "litr2n_vr",
    "litr3n_vr",
    "cwdn_vr",
    "soil1n_vr",
    "soil2n_vr",
    "soil3n_vr",
)
_COLUMN_LEVEL_VARIABLES = (
    *_CARBON_POOL_VARIABLES,
    "col_ctrunc_vr",
    *_NITROGEN_POOL_VARIABLES,
    "sminn_vr",
    "smin_no3_vr",
    "smin_nh4_vr",
    "col_ntrunc_vr",
)
_PATCH_VARIABLES = tuple(
    dict.fromkeys(
        (
            *CARBON_DISPLAYED_COMPONENTS,
            *CARBON_STORED_COMPONENTS,
            "grainc",
            "grainc_storage",
            "grainc_xfer",
            "cropseedc_deficit",
            "xsmrpool",
            "xsmrpool_loss",
            "pft_ctrunc",
            *NITROGEN_DISPLAYED_COMPONENTS,
            *NITROGEN_STORED_COMPONENTS,
            "grainn",
            "grainn_storage",
            "grainn_xfer",
            "cropseedn_deficit",
            "pft_ntrunc",
        )
    )
)
_RESTART_VARIABLES = (*_COLUMN_LEVEL_VARIABLES, *_PATCH_VARIABLES)

_POOL_SUMMARIES = {
    "decomposition_carbon_column_pool",
    "decomposition_carbon_one_meter_column_pool",
    "decomposition_nitrogen_column_pool",
    "decomposition_nitrogen_one_meter_column_pool",
}
_LEVEL_SUMMARIES = {
    "decomposition_soil_carbon_column_level",
    "decomposition_soil_nitrogen_column_level",
}
_PATCH_SUMMARIES = {
    "displayed_vegetation_carbon_patch",
    "stored_vegetation_carbon_patch",
    "total_vegetation_carbon_patch",
    "total_carbon_patch",
    "wood_carbon_patch",
    "displayed_vegetation_nitrogen_patch",
    "stored_vegetation_nitrogen_patch",
    "total_vegetation_nitrogen_patch",
    "total_nitrogen_patch",
}


class CarbonNitrogenBalanceInitializationNativeData(NamedTuple):
    topology: SubgridTopology
    inputs: CarbonNitrogenBalanceInitializationInputs
    state: CarbonNitrogenBalanceInitializationState
    parameters: CarbonNitrogenBalanceInitializationParameters
    restart_path: Path
    parameter_data_path: Path


def _selected_parameter_path(
    case: CimeCase,
    general: Mapping[str, object],
    override: str | Path | None,
) -> Path:
    if override is None:
        declared = general.get("paramfile")
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError("clm_inparm lacks paramfile")
        path = Path(case.expand(declared)).expanduser()
        if not path.is_absolute():
            path = case.root / path
        path = path.resolve()
    else:
        path = Path(override).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM paramfile does not exist: {path}")
    return path


def _pft_names(parameter_data: Path) -> tuple[str, ...]:
    from netCDF4 import Dataset  

    with Dataset(parameter_data, mode="r") as dataset:
        if "pftname" not in dataset.variables:
            raise ValueError("CLM parameter file lacks pftname")
        characters = np.asarray(dataset.variables["pftname"][:])
    if characters.ndim != 2 or characters.dtype.kind not in {"S", "U"}:
        raise ValueError("CLM pftname must be a character matrix")
    if characters.dtype.kind == "S":
        return tuple(
            b"".join(np.asarray(row, dtype="S1").tolist()).decode("ascii").rstrip()
            for row in characters
        )
    return tuple("".join(np.asarray(row, dtype=str).tolist()).rstrip() for row in characters)


def _empty_state(
    column_count: int,
    patch_count: int,
) -> CarbonNitrogenBalanceInitializationState:
    summary_values: dict[str, jnp.ndarray] = {}
    for name in CarbonNitrogenStateSummaries._fields:
        if name in _POOL_SUMMARIES:
            shape = (column_count, NUMBER_OF_DECOMPOSITION_POOLS)
        elif name in _LEVEL_SUMMARIES:
            shape = (column_count, NUMBER_OF_DECOMPOSITION_LEVELS)
        elif name in _PATCH_SUMMARIES:
            shape = (patch_count,)
        else:
            shape = (column_count,)
        summary_values[name] = jnp.zeros(shape, dtype=jnp.float64)
    return CarbonNitrogenBalanceInitializationState(
        summaries=CarbonNitrogenStateSummaries(**summary_values),
        beginning_carbon_balance_column=jnp.zeros(column_count, dtype=jnp.float64),
        beginning_nitrogen_balance_column=jnp.zeros(column_count, dtype=jnp.float64),
    )


def _validated_state_array(
    values: dict[str, jnp.ndarray],
    name: str,
    expected_shape: tuple[int, ...],
    active: np.ndarray,
) -> jnp.ndarray:
    host = np.asarray(values[name], dtype=np.float64)
    if host.shape != expected_shape:
        raise ValueError(f"restart {name} must have shape {expected_shape}")
    if not np.all(np.isfinite(host[active])):
        raise ValueError(f"restart {name} is non-finite on the active CN filter")
    return jnp.asarray(np.where(np.isfinite(host), host, np.float64(0.0)))


@native_template
def load_carbon_nitrogen_balance_initialization_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> CarbonNitrogenBalanceInitializationNativeData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    required = {
        "use_cn": True,
        "use_crop": True,
        "use_century_decomp": True,
        "use_vertsoilc": True,
        "use_nitrif_denitrif": True,
        "use_fates": False,
        "soil_layerstruct": "20SL_8.5m",
    }
    for name, expected in required.items():
        if general.get(name) != expected:
            raise ValueError(f"native C/N balance adapter requires {name}={expected!r}")
    maximum_patches = general.get("maxpatch_pft")
    if not isinstance(maximum_patches, int) or maximum_patches < 1:
        raise ValueError("clm_inparm maxpatch_pft must be a positive integer")

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    parameter_data = _selected_parameter_path(case, general, parameter_data_path)
    dimensions = load_dimension_sizes(restart)
    if dimensions.get("levgrnd") != 25:
        raise ValueError("native C/N balance adapter requires 25 ground levels")

    topology = load_subgrid_topology(restart)
    column_count = topology.column.gridcell.shape[0]
    patch_count = topology.patch.gridcell.shape[0]
    if dimensions.get("column") != column_count or dimensions.get("pft") != patch_count:
        raise ValueError("restart dimensions disagree with native topology")
    column_active = np.asarray(topology.column.active, dtype=bool)
    patch_active = np.asarray(topology.patch.active, dtype=bool)
    soil_column = column_active & np.isin(
        np.asarray(topology.column.landunit_type, dtype=np.int32),
        (LANDUNIT_SOIL, LANDUNIT_CROP),
    )
    soil_patch = patch_active & np.isin(
        np.asarray(topology.patch.landunit_type, dtype=np.int32),
        (LANDUNIT_SOIL, LANDUNIT_CROP),
    )
    patch_column = np.asarray(topology.patch.column, dtype=np.int32)
    if np.any((patch_column < 0) | (patch_column >= column_count)):
        raise ValueError("restart patch-to-column indices lie outside native bounds")
    if np.any(~soil_column[patch_column[soil_patch]]):
        raise ValueError("an active soil patch maps to a non-soil column")

    names = _pft_names(parameter_data)
    try:
        first_crop_pft = names.index("temperate_corn")
    except ValueError as error:
        raise ValueError("CLM parameter file lacks temperate_corn") from error
    vegetation_type = np.asarray(topology.patch.vegetation_type, dtype=np.int32)
    if np.any((vegetation_type < 0) | (vegetation_type >= len(names))):
        raise ValueError("restart vegetation type lies outside CLM PFT parameters")

    raw = load_native_variables(
        restart,
        list(_RESTART_VARIABLES),
        dtype=np.float64,
    )
    column_level: dict[str, jnp.ndarray] = {
        name: _validated_state_array(
            raw,
            name,
            (column_count, dimensions["levgrnd"]),
            soil_column,
        )[:, :NUMBER_OF_DECOMPOSITION_LEVELS]
        for name in _COLUMN_LEVEL_VARIABLES
    }
    patch: dict[str, jnp.ndarray] = {
        name: _validated_state_array(raw, name, (patch_count,), soil_patch)
        for name in _PATCH_VARIABLES
    }
    _, interface, thickness = soil_layer_geometry_20sl_8p5m()

    inputs = CarbonNitrogenBalanceInitializationInputs(
        all_column=topology.column.active,
        soil_column=jnp.asarray(soil_column),
        soil_patch=jnp.asarray(soil_patch),
        patch_active=topology.patch.active,
        patch_column_index=topology.patch.column,
        patch_weight_on_column=topology.patch.weight_on_column,
        vegetation_type_patch=topology.patch.vegetation_type,
        decomposition_layer_thickness=jnp.asarray(thickness[:NUMBER_OF_DECOMPOSITION_LEVELS]),
        decomposition_interface_depth=jnp.asarray(
            interface[1 : NUMBER_OF_DECOMPOSITION_LEVELS + 1]
        ),
        decomposition_carbon_column_level_pool=jnp.stack(
            tuple(column_level[name] for name in _CARBON_POOL_VARIABLES), axis=2
        ),
        carbon_truncation_column_level=column_level["col_ctrunc_vr"],
        decomposition_nitrogen_column_level_pool=jnp.stack(
            tuple(column_level[name] for name in _NITROGEN_POOL_VARIABLES), axis=2
        ),
        mineral_nitrogen_column_level=column_level["sminn_vr"],
        nitrate_nitrogen_column_level=column_level["smin_no3_vr"],
        ammonium_nitrogen_column_level=column_level["smin_nh4_vr"],
        nitrogen_truncation_column_level=column_level["col_ntrunc_vr"],
        displayed_carbon_patch_component=jnp.stack(
            tuple(patch[name] for name in CARBON_DISPLAYED_COMPONENTS), axis=1
        ),
        stored_carbon_patch_component=jnp.stack(
            tuple(patch[name] for name in CARBON_STORED_COMPONENTS), axis=1
        ),
        grain_carbon_patch=patch["grainc"],
        grain_carbon_storage_patch=patch["grainc_storage"],
        grain_carbon_transfer_patch=patch["grainc_xfer"],
        crop_seed_carbon_deficit_patch=patch["cropseedc_deficit"],
        excess_maintenance_respiration_carbon_patch=patch["xsmrpool"],
        excess_maintenance_respiration_carbon_loss_patch=patch["xsmrpool_loss"],
        patch_carbon_truncation=patch["pft_ctrunc"],
        displayed_nitrogen_patch_component=jnp.stack(
            tuple(patch[name] for name in NITROGEN_DISPLAYED_COMPONENTS), axis=1
        ),
        stored_nitrogen_patch_component=jnp.stack(
            tuple(patch[name] for name in NITROGEN_STORED_COMPONENTS), axis=1
        ),
        grain_nitrogen_patch=patch["grainn"],
        grain_nitrogen_storage_patch=patch["grainn_storage"],
        grain_nitrogen_transfer_patch=patch["grainn_xfer"],
        crop_seed_nitrogen_deficit_patch=patch["cropseedn_deficit"],
        patch_nitrogen_truncation=patch["pft_ntrunc"],
    )
    return CarbonNitrogenBalanceInitializationNativeData(
        topology=topology,
        inputs=inputs,
        state=_empty_state(column_count, patch_count),
        parameters=CarbonNitrogenBalanceInitializationParameters(
            use_crop=True,
            use_nitrification_denitrification=True,
            first_prognostic_crop_pft=first_crop_pft,
            maximum_patches_per_column=maximum_patches,
        ),
        restart_path=restart,
        parameter_data_path=parameter_data,
    )


__all__ = [
    "CarbonNitrogenBalanceInitializationNativeData",
    "load_carbon_nitrogen_balance_initialization_native_data",
]
