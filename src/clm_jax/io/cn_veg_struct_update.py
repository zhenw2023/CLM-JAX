
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    SubgridTopology,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)
from clm_jax.physics.cn_fire_area import CnFireAreaResult
from clm_jax.physics.cn_precision_control import (
    get_cn_precision_control_state_field,
)
from clm_jax.physics.cn_veg_struct_update import (
    CnVegStructUpdateInputs,
    CnVegStructUpdateParameters,
    CnVegStructUpdateState,
)
from clm_jax.physics.crop_phenology import CropPhenologyResult
from clm_jax.physics.post_drainage_precision_control import (
    PostDrainagePrecisionControlResult,
)

_PARAMETER_VARIABLES = (
    "woody",
    "slatop",
    "dsladlai",
    "z0mr",
    "displar",
    "ztopmx",
    "laimx",
)
_RESTART_REAL_STATE = (
    "htmx",
    "tlai",
    "tsai",
    "htop",
    "hbot",
    "elai",
    "esai",
)
_RESTART_INTEGER_STATE = (
    "peaklai",
    "FRAC_VEG_NOSNO_ALB",
    "harvdate",
)
_PFT_NAMES = {
    "bare_ground_pft": "not_vegetated",
    "generic_c3_crop_pft": "c3_crop",
    "generic_irrigated_c3_crop_pft": "c3_irrigated",
    "first_shrub_pft": "broadleaf_evergreen_shrub",
    "last_shrub_pft": "broadleaf_deciduous_boreal_shrub",
    "first_prognostic_crop_pft": "temperate_corn",
    "temperate_corn_pft": "temperate_corn",
    "irrigated_temperate_corn_pft": "irrigated_temperate_corn",
    "sugarcane_pft": "sugarcane",
    "irrigated_sugarcane_pft": "irrigated_sugarcane",
    "tropical_corn_pft": "tropical_corn",
    "irrigated_tropical_corn_pft": "irrigated_tropical_corn",
}


class CnVegStructUpdateNativeData(NamedTuple):

    topology: SubgridTopology
    soil_patch: jnp.ndarray
    restart_state: CnVegStructUpdateState
    restart_harvest_date_patch: jnp.ndarray
    parameters: CnVegStructUpdateParameters
    restart_path: Path
    parameter_data_path: Path


class CnVegStructUpdateCall(NamedTuple):
    inputs: CnVegStructUpdateInputs
    state: CnVegStructUpdateState
    parameters: CnVegStructUpdateParameters


def _selected_parameter_path(
    case: CimeCase,
    override: str | Path | None,
) -> Path:
    if override is None:
        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        declared = general.get("paramfile")
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError("configured clm_inparm lacks paramfile")
        path = Path(case.expand(declared)).expanduser()
        if not path.is_absolute():
            path = case.root / path
        path = path.resolve()
    else:
        path = Path(override).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM parameter file does not exist: {path}")
    return path


def _pft_names(parameter_path: Path) -> tuple[str, ...]:
    from netCDF4 import Dataset  

    with Dataset(parameter_path, mode="r") as dataset:
        if "pftname" not in dataset.variables:
            raise ValueError("CLM parameter file lacks pftname")
        raw = np.asarray(dataset.variables["pftname"][:])
    if raw.ndim != 2 or raw.dtype.kind not in {"S", "U"}:
        raise ValueError("CLM pftname must be a two-dimensional character array")
    if raw.dtype.kind == "S":
        names = tuple(
            b"".join(np.asarray(row, dtype="S1").tolist()).decode("ascii").rstrip()
            for row in raw
        )
    else:
        names = tuple("".join(np.asarray(row, dtype=str).tolist()).rstrip() for row in raw)
    if not all(names) or len(names) != len(set(names)):
        raise ValueError("CLM pftname entries must be nonempty and unique")
    return names


def _pft_identifiers(names: tuple[str, ...]) -> dict[str, int]:
    identifiers: dict[str, int] = {}
    for field_name, pft_name in _PFT_NAMES.items():
        try:
            identifiers[field_name] = names.index(pft_name)
        except ValueError as error:
            raise ValueError(f"CLM parameter file lacks PFT {pft_name!r}") from error
    return identifiers


def _validate_configured_case(case: CimeCase) -> tuple[dict[str, object], int]:
    general = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    required = {
        "use_cn": True,
        "use_crop": True,
        "use_fates": False,
    }
    for name, expected in required.items():
        if general.get(name) is not expected:
            raise ValueError(f"native vegetation-structure adapter requires {name}={expected!r}")
    if general.get("use_cndv", False) is not False:
        raise NotImplementedError(
            "the configured vegetation-structure kernel requires use_cndv=.false."
        )
    compset = case.value("COMPSET", "").upper()
    if "BGC-CROP" not in compset or "CNDV" in compset:
        raise ValueError("native vegetation structure requires non-CNDV BGC-Crop")
    atmosphere = read_case_namelist_group(case, "datm_in", "datm_nml")
    if atmosphere.get("iradsw") != 1:
        raise NotImplementedError(
            "radiation intervals other than every land step require an explicit runtime interval"
        )
    spinup_state = general.get("spinup_state", 0)
    if isinstance(spinup_state, bool) or not isinstance(spinup_state, int):
        raise TypeError("clm_inparm spinup_state must be an integer")
    return general, spinup_state


def _restart_state(
    restart_path: Path,
    patch_count: int,
) -> tuple[CnVegStructUpdateState, jnp.ndarray]:
    floating = load_native_variables(
        restart_path,
        list(_RESTART_REAL_STATE),
        dtype=np.float64,
        masked_floating_fill_value=float(CLM_SPECIAL_VALUE),
    )
    integer = load_native_variables(
        restart_path,
        list(_RESTART_INTEGER_STATE),
        masked_integer_fill_value=999,
    )
    for name, value in {**floating, **integer}.items():
        if np.asarray(value).shape != (patch_count,):
            raise ValueError(f"restart {name!r} must contain every native patch")
    state = CnVegStructUpdateState(
        maximum_crop_height_patch=jnp.asarray(floating["htmx"], dtype=jnp.float64),
        peak_leaf_area_index_patch=jnp.asarray(integer["peaklai"], dtype=jnp.int32),
        total_leaf_area_index_patch=jnp.asarray(floating["tlai"], dtype=jnp.float64),
        total_stem_area_index_patch=jnp.asarray(floating["tsai"], dtype=jnp.float64),
        canopy_top_height_patch=jnp.asarray(floating["htop"], dtype=jnp.float64),
        canopy_bottom_height_patch=jnp.asarray(floating["hbot"], dtype=jnp.float64),
        exposed_leaf_area_index_patch=jnp.asarray(floating["elai"], dtype=jnp.float64),
        exposed_stem_area_index_patch=jnp.asarray(floating["esai"], dtype=jnp.float64),
        fraction_vegetation_nosnow_albedo_patch=jnp.asarray(
            integer["FRAC_VEG_NOSNO_ALB"],
            dtype=jnp.int32,
        ),
    )
    return state, jnp.asarray(integer["harvdate"], dtype=jnp.int32)


def load_cn_veg_struct_update_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> CnVegStructUpdateNativeData:

    general, spinup_state = _validate_configured_case(case)
    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"configured CLM restart does not exist: {restart}")
    parameter_path = _selected_parameter_path(case, parameter_data_path)
    topology = load_subgrid_topology(restart)
    patch_count = topology.patch.column.shape[0]
    column_count = topology.column.gridcell.shape[0]
    patch_column = np.asarray(topology.patch.column, dtype=np.int32)
    vegetation_type = np.asarray(topology.patch.vegetation_type, dtype=np.int32)
    if np.any((patch_column < 0) | (patch_column >= column_count)):
        raise ValueError("restart patch-to-column topology is out of bounds")

    names = _pft_names(parameter_path)
    pft_count = len(names)
    if np.any((vegetation_type < 0) | (vegetation_type >= pft_count)):
        raise ValueError("restart vegetation type lies outside the PFT parameter axis")
    raw = load_native_variables(
        parameter_path,
        list(_PARAMETER_VARIABLES),
        dtype=np.float64,
        masked_floating_fill_value=0.0,
    )
    for name, value in raw.items():
        host = np.asarray(value, dtype=np.float64)
        if host.shape != (pft_count,) or not np.all(np.isfinite(host)):
            raise ValueError(f"PFT parameter {name!r} must be one finite PFT vector")
    if np.any(np.asarray(raw["woody"]) < 0.0):
        raise ValueError("PFT woody flags must be nonnegative")
    active = np.asarray(topology.patch.active, dtype=bool)
    landunit = np.asarray(topology.patch.landunit_type, dtype=np.int32)
    soil_patch = active & np.isin(landunit, (LANDUNIT_SOIL, LANDUNIT_CROP))
    restart_state, harvest_date = _restart_state(restart, patch_count)
    for name in restart_state._fields:
        values = np.asarray(getattr(restart_state, name))
        if not np.all(np.isfinite(values[soil_patch])):
            raise ValueError(f"restart structure field {name!r} is invalid on soil patches")
    prognostic_crop = soil_patch & (
        vegetation_type >= _pft_identifiers(names)["first_prognostic_crop_pft"]
    )
    if np.any(np.asarray(harvest_date)[prognostic_crop] > 999):
        raise ValueError("restart prognostic-crop harvest date exceeds CLM sentinel 999")

    clock = load_restart_clock(restart)
    configured_step = general.get("dtime")
    if isinstance(configured_step, bool) or not isinstance(configured_step, int):
        raise TypeError("clm_inparm dtime must be an integer")
    if configured_step <= 0 or configured_step != clock.step_seconds:
        raise ValueError("case dtime does not match the CLM restart time step")

    identifiers = _pft_identifiers(names)
    parameters = CnVegStructUpdateParameters(
        woody_pft=jnp.asarray(raw["woody"], dtype=jnp.float64),
        specific_leaf_area_top_pft=jnp.asarray(raw["slatop"], dtype=jnp.float64),
        specific_leaf_area_slope_pft=jnp.asarray(
            raw["dsladlai"],
            dtype=jnp.float64,
        ),
        roughness_to_height_ratio_pft=jnp.asarray(raw["z0mr"], dtype=jnp.float64),
        displacement_to_height_ratio_pft=jnp.asarray(
            raw["displar"],
            dtype=jnp.float64,
        ),
        wood_density_pft=jnp.full(pft_count, np.float64(2.5e5)),
        maximum_crop_height_pft=jnp.asarray(raw["ztopmx"], dtype=jnp.float64),
        maximum_leaf_area_index_pft=jnp.asarray(raw["laimx"], dtype=jnp.float64),
        radiation_step_seconds=np.float64(configured_step),
        spinup_state=spinup_state,
        **identifiers,
    )
    return CnVegStructUpdateNativeData(
        topology=topology,
        soil_patch=jnp.asarray(soil_patch),
        restart_state=restart_state,
        restart_harvest_date_patch=harvest_date,
        parameters=parameters,
        restart_path=restart,
        parameter_data_path=parameter_path,
    )


def assemble_cn_veg_struct_update_call(
    native: CnVegStructUpdateNativeData,
    precision_result: PostDrainagePrecisionControlResult,
    fire_area_result: CnFireAreaResult,
    crop_phenology_result: CropPhenologyResult,
    snow_depth_column: jnp.ndarray,
    wind_forcing_height_patch: jnp.ndarray,
    previous_state: CnVegStructUpdateState | None = None,
) -> CnVegStructUpdateCall:

    patch_count = native.topology.patch.column.shape[0]
    column_count = native.topology.column.gridcell.shape[0]
    vegetation = precision_result.state.vegetation
    leaf_carbon = get_cn_precision_control_state_field(
        vegetation,
        "vegetation_carbon.leafc_patch",
    )
    dead_stem_carbon = get_cn_precision_control_state_field(
        vegetation,
        "vegetation_carbon.deadstemc_patch",
    )
    burned_area = fire_area_result.state.total_burned_area_fraction_column
    harvest_date = crop_phenology_result.state.harvest_date_patch
    shapes = {
        "same-step leaf carbon": (leaf_carbon, (patch_count,)),
        "same-step dead stem carbon": (dead_stem_carbon, (patch_count,)),
        "same-step burned area": (burned_area, (column_count,)),
        "same-step harvest date": (harvest_date, (patch_count,)),
        "same-step snow depth": (snow_depth_column, (column_count,)),
        "same-step wind forcing height": (
            wind_forcing_height_patch,
            (patch_count,),
        ),
    }
    for name, (values, expected) in shapes.items():
        if jnp.shape(values) != expected:
            raise ValueError(f"{name} has shape {jnp.shape(values)}, expected {expected}")
    state = native.restart_state if previous_state is None else previous_state
    for name in state._fields:
        if jnp.shape(getattr(state, name)) != (patch_count,):
            raise ValueError(f"current structure state {name!r} has the wrong extent")

    return CnVegStructUpdateCall(
        inputs=CnVegStructUpdateInputs(
            soil_patch=native.soil_patch,
            patch_column_index=native.topology.patch.column,
            vegetation_type_patch=native.topology.patch.vegetation_type,
            snow_depth_column=jnp.asarray(snow_depth_column, dtype=jnp.float64),
            wind_forcing_height_patch=jnp.asarray(
                wind_forcing_height_patch,
                dtype=jnp.float64,
            ),
            leaf_carbon_patch=jnp.asarray(leaf_carbon, dtype=jnp.float64),
            dead_stem_carbon_patch=jnp.asarray(
                dead_stem_carbon,
                dtype=jnp.float64,
            ),
            burned_area_fraction_column=jnp.asarray(burned_area, dtype=jnp.float64),
            harvest_date_patch=jnp.asarray(harvest_date, dtype=jnp.int32),
        ),
        state=state,
        parameters=native.parameters,
    )


__all__ = [
    "CnVegStructUpdateCall",
    "CnVegStructUpdateNativeData",
    "assemble_cn_veg_struct_update_call",
    "load_cn_veg_struct_update_native_data",
]
