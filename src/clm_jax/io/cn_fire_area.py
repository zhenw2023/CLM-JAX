
from __future__ import annotations

import calendar as calendar_module
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.canopy_temperature_restart import (
    soil_layer_geometry_20sl_8p5m,
)
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    RestartClock,
    advance_restart_clock,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
    model_step_from_restart_clock,
)
from clm_jax.physics.cn_fire_area import (
    CnFireAreaForcing,
    CnFireAreaParameters,
    CnFireAreaState,
    CnFireAreaTopology,
)
from clm_jax.physics.cn_precision_control import CnPrecisionControlState

_PARAMETER_VARIABLES = (
    "fsr_pft",
    "fd_pft",
    "tau_cwd",
    "tau_s2",
    "tau_s3",
)
_DECOMPOSITION_POOL_IS_CWD = (
    False,
    False,
    False,
    True,
    False,
    False,
    False,
)
_PFT_BOUNDARY_NAMES = {
    "bare_ground_pft": "not_vegetated",
    "first_natural_vegetation_pft": "needleleaf_evergreen_temperate_tree",
    "tropical_evergreen_tree_pft": "broadleaf_evergreen_tropical_tree",
    "tropical_deciduous_tree_pft": "broadleaf_deciduous_tropical_tree",
    "first_shrub_pft": "broadleaf_evergreen_shrub",
    "c4_grass_pft": "c4_grass",
    "first_crop_pft": "c3_crop",
}


class CnFireAreaPersistentInputs(NamedTuple):

    topology: CnFireAreaTopology
    gdp_column: jnp.ndarray
    peatland_fraction_column: jnp.ndarray
    crop_fire_peak_month_column: jnp.ndarray


class CnFireAreaSameStepForcing(NamedTuple):

    relative_humidity_gridcell: jnp.ndarray
    wind_speed_gridcell: jnp.ndarray
    temperature_downscaled_column: jnp.ndarray
    rain_downscaled_column: jnp.ndarray
    snow_downscaled_column: jnp.ndarray
    precipitation_60day_patch: jnp.ndarray
    precipitation_10day_patch: jnp.ndarray
    relative_humidity_30day_patch: jnp.ndarray
    root_zone_wetness_patch: jnp.ndarray
    saturated_area_fraction_column: jnp.ndarray
    topsoil_water_fraction_column: jnp.ndarray
    lightning_frequency_gridcell: jnp.ndarray
    human_population_density_gridcell: jnp.ndarray
    total_litter_carbon_column: jnp.ndarray
    total_vegetation_carbon_column: jnp.ndarray
    decomposition_carbon_pool_column_level_pool: jnp.ndarray
    soil_temperature_17cm_column: jnp.ndarray


class CnFireAreaStepClock(NamedTuple):

    model_step: int
    current_date: int
    current_time_of_day: int


class CnFireAreaNativeData(NamedTuple):

    persistent_inputs: CnFireAreaPersistentInputs
    restart_state: CnFireAreaState
    parameters: CnFireAreaParameters
    restart_clock: RestartClock
    calendar_name: str
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path
    population_data_path: Path
    lightning_data_path: Path


class CnFireAreaCall(NamedTuple):

    topology: CnFireAreaTopology
    forcing: CnFireAreaForcing
    vegetation_state: CnPrecisionControlState
    state: CnFireAreaState
    parameters: CnFireAreaParameters


def _selected_path(
    case: CimeCase,
    group_name: str,
    key: str,
    override: str | Path | None,
) -> Path:
    if override is None:
        group = read_case_namelist_group(case, "lnd_in", group_name)
        declared = group.get(key)
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError(f"{group_name} in {case.case_docs / 'lnd_in'} lacks {key}")
        path = Path(case.expand(declared)).expanduser()
        if not path.is_absolute():
            path = case.root / path
        path = path.resolve()
    else:
        path = Path(override).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM {key} file does not exist: {path}")
    return path


def _logical(group: dict[str, object], name: str, default: bool) -> bool:
    value = group.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"CLM namelist value {name!r} must be logical")
    return value


def _integer(group: dict[str, object], name: str, default: int) -> int:
    value = group.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"CLM namelist value {name!r} must be integer")
    return value


def _real(group: dict[str, object], name: str, default: float) -> np.float64:
    value = group.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"CLM namelist value {name!r} must be real")
    result = np.float64(value)
    if not np.isfinite(result):
        raise ValueError(f"CLM namelist value {name!r} must be finite")
    return result


def _finite_scalar(name: str, value: jnp.ndarray) -> np.float64:
    host = np.asarray(value, dtype=np.float64)
    if host.size != 1 or not np.isfinite(host.reshape(())):
        raise ValueError(f"CLM parameter {name!r} must contain one finite scalar")
    return np.float64(host.reshape(()))


def _pft_names(parameter_data_path: Path) -> tuple[str, ...]:
    from netCDF4 import Dataset  

    with Dataset(parameter_data_path, mode="r") as dataset:
        if "pftname" not in dataset.variables:
            raise ValueError("CLM parameter file lacks pftname")
        raw = np.asarray(dataset.variables["pftname"][:])
    if raw.ndim != 2 or raw.dtype.kind not in {"S", "U"}:
        raise ValueError("CLM pftname must be a character matrix")
    if raw.dtype.kind == "S":
        names = tuple(
            b"".join(np.asarray(row, dtype="S1").tolist()).decode("ascii").rstrip() for row in raw
        )
    else:
        names = tuple("".join(np.asarray(row, dtype=str).tolist()).rstrip() for row in raw)
    if not all(names) or len(names) != len(set(names)):
        raise ValueError("CLM pftname entries must be nonempty and unique")
    return names


def _pft_boundaries(names: tuple[str, ...]) -> dict[str, int]:
    result: dict[str, int] = {}
    for field_name, pft_name in _PFT_BOUNDARY_NAMES.items():
        try:
            result[field_name] = names.index(pft_name)
        except ValueError as error:
            raise ValueError(f"CLM parameter pftname lacks required entry {pft_name!r}") from error
    return result


def _days_per_year(calendar_name: str, current_date: int) -> np.float64:
    year = current_date // 10_000
    if year < 1:
        raise ValueError("CLM current date must contain a positive year")
    normalized = calendar_name.strip().upper()
    if normalized == "NO_LEAP":
        return np.float64(365.0)
    if normalized == "GREGORIAN":
        return np.float64(366.0 if calendar_module.isleap(year) else 365.0)
    if normalized == "360_DAY":
        return np.float64(360.0)
    raise ValueError(f"unsupported CLM calendar for fire area: {normalized!r}")


def _validate_current_case(
    general: dict[str, object],
    dynamic: dict[str, object],
    fire: dict[str, object],
) -> int:
    required = {
        "use_cn": True,
        "use_crop": True,
        "use_century_decomp": True,
        "use_vertsoilc": True,
        "use_fates": False,
        "soil_layerstruct": "20SL_8.5m",
    }
    for name, expected in required.items():
        if general.get(name) != expected:
            raise ValueError(f"native Li2016 fire-area adapter requires {name}={expected!r}")
    if fire.get("fire_method") != "li2016crufrc":
        raise ValueError("native fire-area adapter requires fire_method='li2016crufrc'")
    for name in ("do_transient_pfts", "do_transient_crops"):
        if _logical(dynamic, name, False):
            raise NotImplementedError(
                "this kernel covers the active fixed-landcover CNFireArea path"
            )
    max_patches = _integer(general, "maxpatch_pft", 79)
    if max_patches != 79:
        raise ValueError("Li2016 kernel requires the production maxpatch_pft=79")
    return _integer(general, "spinup_state", 0)


def _surface_fire_fields(
    surface_data_path: Path,
    longitude_index: np.ndarray,
    latitude_index: np.ndarray,
    column_gridcell: np.ndarray,
    soil_column: np.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    floating = load_native_variables(
        surface_data_path,
        ["gdp", "peatf"],
        dtype=np.float64,
    )
    integer = load_native_variables(
        surface_data_path,
        ["abm"],
        masked_integer_fill_value=np.iinfo(np.int32).max,
    )
    gdp_grid = np.asarray(floating["gdp"], dtype=np.float64)[
        latitude_index,
        longitude_index,
    ]
    peat_grid = np.asarray(floating["peatf"], dtype=np.float64)[
        latitude_index,
        longitude_index,
    ]
    crop_month_grid = np.asarray(integer["abm"], dtype=np.int32)[
        latitude_index,
        longitude_index,
    ]
    gdp = gdp_grid[column_gridcell]
    peat = peat_grid[column_gridcell]
    crop_month = crop_month_grid[column_gridcell]
    if not np.all(np.isfinite(gdp[soil_column])) or np.any(gdp[soil_column] < 0.0):
        raise ValueError("surface-data GDP must be finite and nonnegative on soil columns")
    if not np.all(np.isfinite(peat[soil_column])) or np.any(
        (peat[soil_column] < 0.0) | (peat[soil_column] > 1.0)
    ):
        raise ValueError("surface-data peat fraction must lie in [0, 1]")
    if np.any((crop_month[soil_column] < 1) | (crop_month[soil_column] > 13)):
        raise ValueError("surface-data crop fire month must lie in [1, 13]")
    return jnp.asarray(gdp), jnp.asarray(peat), jnp.asarray(crop_month)


def _restart_fire_state(
    restart_path: Path,
    soil_column: np.ndarray,
    soil_patch: np.ndarray,
) -> CnFireAreaState:
    patch_count = soil_patch.size
    column_count = soil_column.size
    burn_values = load_native_variables(
        restart_path,
        ["burndate"],
        masked_integer_fill_value=-9999,
    )
    lfc_values = load_native_variables(
        restart_path,
        ["lfc"],
        dtype=np.float64,
        masked_floating_fill_value=float(CLM_SPECIAL_VALUE),
    )
    burn_date = np.asarray(burn_values["burndate"], dtype=np.int32)
    lfc = np.asarray(lfc_values["lfc"], dtype=np.float64)
    if burn_date.shape != (patch_count,) or lfc.shape != (column_count,):
        raise ValueError("restart fire state does not match native subgrid extents")
    if np.any((burn_date[soil_patch] < 1) | (burn_date[soil_patch] > 10_000)):
        raise ValueError("restart burn dates are invalid on active soil patches")
    if not np.all(np.isfinite(lfc[soil_column])):
        raise ValueError("restart lfc is non-finite on active soil columns")

    initial_column = np.full(
        (column_count,),
        CLM_SPECIAL_VALUE,
        dtype=np.float64,
    )
    initial_column[soil_column] = np.float64(0.0)
    arrays = {name: jnp.asarray(initial_column.copy()) for name in CnFireAreaState._fields[1:]}
    arrays["unburned_conversion_fraction_column"] = jnp.asarray(lfc)
    arrays["fuel_carbon_column"] = jnp.full(
        (column_count,),
        CLM_SPECIAL_VALUE,
        dtype=jnp.float64,
    )
    return CnFireAreaState(
        burn_date_patch=jnp.asarray(burn_date),
        **arrays,
    )


def load_cn_fire_area_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    population_data_path: str | Path | None = None,
    lightning_data_path: str | Path | None = None,
) -> CnFireAreaNativeData:

    general = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    dynamic = dict(read_case_namelist_group(case, "lnd_in", "dynamic_subgrid"))
    fire = dict(read_case_namelist_group(case, "lnd_in", "cnfire_inparm"))
    li_parameters = dict(read_case_namelist_group(case, "lnd_in", "lifire_inparm"))
    spinup_state = _validate_current_case(general, dynamic, fire)

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    surface_data = _selected_path(
        case,
        "clm_inparm",
        "fsurdat",
        surface_data_path,
    )
    parameter_data = _selected_path(
        case,
        "clm_inparm",
        "paramfile",
        parameter_data_path,
    )
    population_data = _selected_path(
        case,
        "popd_streams",
        "stream_fldfilename_popdens",
        population_data_path,
    )
    lightning_data = _selected_path(
        case,
        "light_streams",
        "stream_fldfilename_lightng",
        lightning_data_path,
    )

    native_topology = load_subgrid_topology(restart)
    column_active = np.asarray(native_topology.column.active, dtype=bool)
    column_landunit = np.asarray(
        native_topology.column.landunit_type,
        dtype=np.int32,
    )
    patch_active = np.asarray(native_topology.patch.active, dtype=bool)
    patch_landunit = np.asarray(
        native_topology.patch.landunit_type,
        dtype=np.int32,
    )
    soil_column = column_active & np.isin(
        column_landunit,
        (LANDUNIT_SOIL, LANDUNIT_CROP),
    )
    soil_patch = patch_active & np.isin(
        patch_landunit,
        (LANDUNIT_SOIL, LANDUNIT_CROP),
    )
    column_gridcell = np.asarray(native_topology.column.gridcell, dtype=np.int32)
    grid_count = native_topology.grid.latitude_degrees.shape[0]
    if np.any((column_gridcell < 0) | (column_gridcell >= grid_count)):
        raise ValueError("restart column-to-gridcell topology is out of bounds")
    gdp, peat, crop_month = _surface_fire_fields(
        surface_data,
        np.asarray(native_topology.grid.longitude_index, dtype=np.int64),
        np.asarray(native_topology.grid.latitude_index, dtype=np.int64),
        column_gridcell,
        soil_column,
    )

    raw_parameters = load_native_variables(
        parameter_data,
        list(_PARAMETER_VARIABLES),
        dtype=np.float64,
    )
    names = _pft_names(parameter_data)
    fire_spread = np.asarray(raw_parameters["fsr_pft"], dtype=np.float64)
    fire_duration = np.asarray(raw_parameters["fd_pft"], dtype=np.float64)
    if fire_spread.shape != (len(names),) or fire_duration.shape != (len(names),):
        raise ValueError("CLM fire PFT parameters must match pftname")
    if not np.all(np.isfinite(fire_spread)) or not np.all(np.isfinite(fire_duration)):
        raise ValueError("CLM fire PFT parameters must be finite")
    if np.any(fire_spread < 0.0) or np.any(fire_duration < 0.0):
        raise ValueError("CLM fire PFT parameters must be nonnegative")
    scalar = {
        name: _finite_scalar(name, raw_parameters[name]) for name in ("tau_cwd", "tau_s2", "tau_s3")
    }
    spinup_factor = np.asarray(
        [
            1.0,
            1.0,
            1.0,
            max(1.0, scalar["tau_cwd"] / np.float64(2.0)),
            1.0,
            max(1.0, scalar["tau_s2"]),
            max(1.0, scalar["tau_s3"]),
        ],
        dtype=np.float64,
    )
    _, _, layer_thickness = soil_layer_geometry_20sl_8p5m()
    clock = load_restart_clock(restart)
    if clock.step_seconds <= 0:
        raise ValueError("restart CLM time step must be positive")
    boundaries = _pft_boundaries(names)
    calendar_name = case.value("CALENDAR").strip().upper()
    call_clock = advance_restart_clock(clock, calendar_name)
    month = (call_clock.current_date // 100) % 100
    day = call_clock.current_date % 100
    parameters = CnFireAreaParameters(
        fire_spread_rate_pft=jnp.asarray(fire_spread),
        fire_duration_pft=jnp.asarray(fire_duration),
        decomposition_pool_is_cwd=jnp.asarray(
            _DECOMPOSITION_POOL_IS_CWD,
            dtype=jnp.bool_,
        ),
        decomposition_pool_spinup_factor=jnp.asarray(spinup_factor),
        decomposition_layer_thickness=jnp.asarray(layer_thickness),
        lower_fuel_threshold=_real(li_parameters, "lfuel", 75.0),
        upper_fuel_threshold=_real(li_parameters, "ufuel", 650.0),
        relative_humidity_low=_real(li_parameters, "rh_low", 30.0),
        relative_humidity_high=_real(li_parameters, "rh_hgh", 80.0),
        root_wetness_minimum=_real(li_parameters, "bt_min", 0.3),
        root_wetness_maximum=_real(li_parameters, "bt_max", 0.7),
        deforestation_fire_scale=_real(li_parameters, "cli_scale", 0.035),
        crop_fire_coefficient=_real(li_parameters, "cropfire_a1", 0.3),
        non_boreal_peat_fire_coefficient=_real(
            li_parameters,
            "non_boreal_peatfire_c",
            0.001,
        ),
        human_ignition_coefficient=_real(
            li_parameters,
            "pot_hmn_ign_counts_alpha",
            0.0035,
        ),
        boreal_peat_fire_coefficient=_real(
            li_parameters,
            "boreal_peatfire_c",
            4.2e-5,
        ),
        high_gdp_tree_occurrence=_real(
            li_parameters,
            "occur_hi_gdp_tree",
            0.39,
        ),
        base_spread_factor=np.float64(0.05),
        boreal_latitude_degrees=np.float64(40.0),
        time_step=np.float64(clock.step_seconds),
        days_per_year=_days_per_year(calendar_name, call_clock.current_date),
        model_step=model_step_from_restart_clock(call_clock, calendar_name),
        month=month,
        day_of_month=day,
        seconds_of_day=call_clock.current_time_of_day,
        spinup_state=spinup_state,
        **boundaries,
    )
    topology = CnFireAreaTopology(
        soil_column=jnp.asarray(soil_column),
        soil_patch=jnp.asarray(soil_patch),
        column_gridcell_index=native_topology.column.gridcell,
        patch_column_index=native_topology.patch.column,
        patch_weight_on_column=native_topology.patch.weight_on_column,
        patch_active=native_topology.patch.active,
        vegetation_type_patch=native_topology.patch.vegetation_type,
        gridcell_latitude_degrees=native_topology.grid.latitude_degrees,
    )
    return CnFireAreaNativeData(
        persistent_inputs=CnFireAreaPersistentInputs(
            topology=topology,
            gdp_column=gdp,
            peatland_fraction_column=peat,
            crop_fire_peak_month_column=crop_month,
        ),
        restart_state=_restart_fire_state(restart, soil_column, soil_patch),
        parameters=parameters,
        restart_clock=clock,
        calendar_name=calendar_name,
        restart_path=restart,
        surface_data_path=surface_data,
        parameter_data_path=parameter_data,
        population_data_path=population_data,
        lightning_data_path=lightning_data,
    )


def initial_cn_fire_area_clock(native: CnFireAreaNativeData) -> CnFireAreaStepClock:

    call_clock = advance_restart_clock(
        native.restart_clock,
        native.calendar_name,
    )

    return CnFireAreaStepClock(
        model_step=int(native.parameters.model_step),
        current_date=call_clock.current_date,
        current_time_of_day=call_clock.current_time_of_day,
    )


def assemble_cn_fire_area_call(
    native: CnFireAreaNativeData,
    same_step_forcing: CnFireAreaSameStepForcing,
    vegetation_state: CnPrecisionControlState,
    clock: CnFireAreaStepClock,
    *,
    current_state: CnFireAreaState | None = None,
) -> CnFireAreaCall:

    persistent = native.persistent_inputs
    topology = persistent.topology
    column_count = topology.soil_column.shape[0]
    patch_count = topology.soil_patch.shape[0]
    gridcell_count = topology.gridcell_latitude_degrees.shape[0]
    grid_fields = (
        "relative_humidity_gridcell",
        "wind_speed_gridcell",
        "lightning_frequency_gridcell",
        "human_population_density_gridcell",
    )
    column_fields = (
        "temperature_downscaled_column",
        "rain_downscaled_column",
        "snow_downscaled_column",
        "saturated_area_fraction_column",
        "topsoil_water_fraction_column",
        "total_litter_carbon_column",
        "total_vegetation_carbon_column",
        "soil_temperature_17cm_column",
    )
    patch_fields = (
        "precipitation_60day_patch",
        "precipitation_10day_patch",
        "relative_humidity_30day_patch",
        "root_zone_wetness_patch",
    )
    for name in grid_fields:
        if jnp.shape(getattr(same_step_forcing, name)) != (gridcell_count,):
            raise ValueError(f"same-step {name} must contain every gridcell")
    for name in column_fields:
        if jnp.shape(getattr(same_step_forcing, name)) != (column_count,):
            raise ValueError(f"same-step {name} must contain every column")
    for name in patch_fields:
        if jnp.shape(getattr(same_step_forcing, name)) != (patch_count,):
            raise ValueError(f"same-step {name} must contain every patch")
    decomposition = same_step_forcing.decomposition_carbon_pool_column_level_pool
    expected_decomposition = (
        column_count,
        native.parameters.decomposition_layer_thickness.shape[0],
        native.parameters.decomposition_pool_is_cwd.shape[0],
    )
    if jnp.shape(decomposition) != expected_decomposition:
        raise ValueError(f"same-step decomposition carbon must have shape {expected_decomposition}")
    if vegetation_state.paired_carbon_patch.shape[0] != patch_count:
        raise ValueError("same-step vegetation state has the wrong patch count")
    state = native.restart_state if current_state is None else current_state
    if state.burn_date_patch.shape != (patch_count,):
        raise ValueError("current fire state has the wrong patch count")
    for name in state._fields[1:]:
        if jnp.shape(getattr(state, name)) != (column_count,):
            raise ValueError(f"current fire state {name} has the wrong column count")

    model_step = int(clock.model_step)
    current_date = int(clock.current_date)
    current_time = int(clock.current_time_of_day)
    if model_step < 0:
        raise ValueError("CLM model step must be nonnegative")
    month = (current_date // 100) % 100
    day = current_date % 100
    if month < 1 or month > 12 or day < 1 or day > 31:
        raise ValueError("CLM current date is not a valid YYYYMMDD value")
    if current_time < 0 or current_time >= 86_400:
        raise ValueError("CLM current time of day must lie in [0, 86400)")

    forcing = CnFireAreaForcing(
        **same_step_forcing._asdict(),
        gdp_column=persistent.gdp_column,
        peatland_fraction_column=persistent.peatland_fraction_column,
        crop_fire_peak_month_column=persistent.crop_fire_peak_month_column,
    )
    return CnFireAreaCall(
        topology=topology,
        forcing=forcing,
        vegetation_state=vegetation_state,
        state=state,
        parameters=native.parameters._replace(
            days_per_year=_days_per_year(native.calendar_name, current_date),
            model_step=model_step,
            month=month,
            day_of_month=day,
            seconds_of_day=current_time,
        ),
    )


__all__ = [
    "CnFireAreaCall",
    "CnFireAreaNativeData",
    "CnFireAreaPersistentInputs",
    "CnFireAreaSameStepForcing",
    "CnFireAreaStepClock",
    "assemble_cn_fire_area_call",
    "initial_cn_fire_area_clock",
    "load_cn_fire_area_native_data",
]
