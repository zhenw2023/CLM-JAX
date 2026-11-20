
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import SECONDS_PER_DAY
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import (
    load_dimension_sizes,
    load_native_variables,
)
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    RestartClock,
    SubgridTopology,
    advance_restart_clock,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)
from clm_jax.physics.daylength import daylength
from clm_jax.physics.evergreen_phenology import (
    CARBON_STORAGE_POOL_NAMES,
    CARBON_TRANSFER_POOL_NAMES,
    NITROGEN_STORAGE_POOL_NAMES,
    NITROGEN_TRANSFER_POOL_NAMES,
    EvergreenPhenologyState,
)
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
)
from clm_jax.physics.seasonal_deciduous_phenology import (
    SeasonalDeciduousPhenologyInputs,
    SeasonalDeciduousPhenologyParameters,
    SeasonalDeciduousPhenologyState,
)
from clm_jax.physics.solar import FIXED_YEAR_2000_ORBIT, solar_declination
from clm_jax.physics.solar_runtime import native_grid_radians

_NUMBER_OF_GROUND_LEVELS = 25
_STATE_VARIABLES = (
    "dormant_flag",
    "days_active",
    "onset_flag",
    "onset_counter",
    "onset_gddflag",
    "onset_gdd",
    "offset_flag",
    "offset_counter",
    "bglfr",
    "bgtr",
    "lgsf",
    "prev_leafc_to_litter",
    "prev_frootc_to_litter",
)
_PARAMETER_VARIABLES = (
    "season_decid",
    "woody",
    "crit_dayl",
    "ndays_on",
    "ndays_off",
    "fstor2tran",
)
_POOL_VARIABLES = (
    *CARBON_STORAGE_POOL_NAMES,
    *CARBON_TRANSFER_POOL_NAMES,
    *NITROGEN_STORAGE_POOL_NAMES,
    *NITROGEN_TRANSFER_POOL_NAMES,
)


class SeasonalDeciduousPhenologyNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: SeasonalDeciduousPhenologyInputs
    state: SeasonalDeciduousPhenologyState
    phenology_rates: EvergreenPhenologyState
    parameters: SeasonalDeciduousPhenologyParameters
    restart_path: Path
    parameter_data_path: Path


def _selected_parameter_path(
    case: CimeCase,
    override: str | Path | None,
) -> Path:
    if override is None:
        group = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        declared = group.get("paramfile")
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError(f"clm_inparm in {case.case_docs / 'lnd_in'} lacks paramfile")
        path = Path(case.expand(declared)).expanduser()
        if not path.is_absolute():
            path = case.root / path
        path = path.resolve()
    else:
        path = Path(override).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM paramfile does not exist: {path}")
    return path


def _stack_patch_pools(
    values: dict[str, jnp.ndarray],
    names: tuple[str, ...],
) -> np.ndarray:
    return np.stack([np.asarray(values[name], dtype=np.float64) for name in names], axis=1)


def _noleap_calendar_day(clock: RestartClock) -> np.float64:
    ymd = int(clock.current_date)
    month = (ymd // 100) % 100
    day = ymd % 100
    month_lengths = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    if month < 1 or month > 12 or day < 1 or day > month_lengths[month - 1]:
        raise ValueError(f"invalid NO_LEAP restart date {ymd}")
    if clock.current_time_of_day < 0 or clock.current_time_of_day >= int(
        SECONDS_PER_DAY
    ):
        raise ValueError("restart time of day must lie in [0, 86400)")
    integer_day = 1 + sum(month_lengths[: month - 1]) + day - 1
    return np.float64(integer_day) + np.float64(
        clock.current_time_of_day
    ) / SECONDS_PER_DAY


def load_seasonal_deciduous_phenology_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> SeasonalDeciduousPhenologyNativeData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    for key, expected in {
        "use_cn": True,
        "use_crop": True,
        "use_fates": False,
    }.items():
        if general.get(key) is not expected:
            raise ValueError(f"native seasonal adapter requires {key}={expected!r}")
    if case.value("CALENDAR").upper() != "NO_LEAP":
        raise ValueError("native seasonal adapter currently requires NO_LEAP")
    compset = case.value("COMPSET", "").upper()
    if "BGC-CROP" not in compset or "CNDV" in compset:
        raise ValueError("native seasonal adapter requires a non-CNDV BGC-Crop compset")

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    dimensions = load_dimension_sizes(restart)
    if dimensions.get("levgrnd") != _NUMBER_OF_GROUND_LEVELS:
        raise ValueError("restart must contain the native 25 ground levels")
    parameter_data = _selected_parameter_path(case, parameter_data_path)
    topology = load_subgrid_topology(restart)
    patch_count = topology.patch.gridcell.shape[0]
    column_count = topology.column.gridcell.shape[0]
    gridcell_count = topology.grid.latitude_degrees.shape[0]

    values = load_native_variables(
        restart,
        ["T_SOISNO", "annavg_t2m", *_POOL_VARIABLES, *_STATE_VARIABLES],
        dtype=np.float64,
    )
    for name in ("annavg_t2m", *_POOL_VARIABLES, *_STATE_VARIABLES):
        if values[name].shape != (patch_count,):
            raise ValueError(f"restart variable {name!r} must contain every patch")
    soil_temperature = np.asarray(values["T_SOISNO"], dtype=np.float64)
    if soil_temperature.ndim != 2 or soil_temperature.shape[0] != column_count:
        raise ValueError("restart T_SOISNO must have native (column, level) dimensions")
    if soil_temperature.shape[1] < _NUMBER_OF_GROUND_LEVELS:
        raise ValueError("restart T_SOISNO lacks the ground-level tail")
    third_soil_layer_temperature = soil_temperature[
        :,
        -_NUMBER_OF_GROUND_LEVELS + 2,
    ]

    pft_values = load_native_variables(
        parameter_data,
        list(_PARAMETER_VARIABLES),
        dtype=np.float64,
    )
    seasonal_pft = np.asarray(pft_values["season_decid"], dtype=np.float64)
    woody_pft = np.asarray(pft_values["woody"], dtype=np.float64)
    if seasonal_pft.ndim != 1 or woody_pft.shape != seasonal_pft.shape:
        raise ValueError("season_decid and woody must share one PFT axis")
    if not np.all((seasonal_pft == 0.0) | (seasonal_pft == 1.0)):
        raise ValueError("CLM season_decid flags must contain only zero or one")
    if not np.all((woody_pft == 0.0) | (woody_pft == 1.0)):
        raise ValueError("CLM woody flags must contain only zero or one")
    scalar_parameters: dict[str, np.float64] = {}
    for name in _PARAMETER_VARIABLES[2:]:
        array = np.asarray(pft_values[name], dtype=np.float64)
        if array.size != 1 or not np.isfinite(array.reshape(())):
            raise ValueError(f"CLM parameter {name!r} must be one finite scalar")
        scalar_parameters[name] = np.float64(array.reshape(()))

    active = np.asarray(topology.patch.active, dtype=bool)
    landunit = np.asarray(topology.patch.landunit_type, dtype=np.int32)
    soil_patch = active & (
        (landunit == LANDUNIT_SOIL) | (landunit == LANDUNIT_CROP)
    )
    vegetation_type = np.asarray(topology.patch.vegetation_type, dtype=np.int32)
    if np.any((vegetation_type < 0) | (vegetation_type >= seasonal_pft.shape[0])):
        raise ValueError("restart vegetation type lies outside PFT parameter vectors")
    patch_column = np.asarray(topology.patch.column, dtype=np.int32)
    patch_gridcell = np.asarray(topology.patch.gridcell, dtype=np.int32)
    if np.any((patch_column < 0) | (patch_column >= column_count)):
        raise ValueError("restart patch-to-column topology is out of bounds")
    if np.any((patch_gridcell < 0) | (patch_gridcell >= gridcell_count)):
        raise ValueError("restart patch-to-gridcell topology is out of bounds")
    seasonal_patch = soil_patch & (seasonal_pft[vegetation_type] == 1.0)
    for name in ("annavg_t2m", *_POOL_VARIABLES, *_STATE_VARIABLES):
        if not np.all(np.isfinite(np.asarray(values[name])[seasonal_patch])):
            raise ValueError(f"seasonal deciduous patches contain missing {name}")
    seasonal_column = np.zeros(column_count, dtype=bool)
    seasonal_column[patch_column[seasonal_patch]] = True
    if not np.all(np.isfinite(third_soil_layer_temperature[seasonal_column])):
        raise ValueError("seasonal deciduous columns contain missing soil temperature")

    clock = advance_restart_clock(
        load_restart_clock(restart),
        case.value("CALENDAR"),
    )
    calendar_day = jnp.asarray(_noleap_calendar_day(clock), dtype=jnp.float64)
    step_fraction = fortran_divide(
        jnp.asarray(clock.step_seconds, dtype=jnp.float64),
        jnp.float64(SECONDS_PER_DAY),
    )
    previous_calendar_day = fortran_add(calendar_day, -step_fraction)
    declination, _ = solar_declination(calendar_day, FIXED_YEAR_2000_ORBIT)
    previous_declination, _ = solar_declination(
        previous_calendar_day,
        FIXED_YEAR_2000_ORBIT,
    )
    latitude_radians = native_grid_radians(topology.grid.latitude_degrees)
    current_daylength = daylength(latitude_radians, declination)
    previous_daylength = daylength(latitude_radians, previous_declination)

    return SeasonalDeciduousPhenologyNativeData(
        topology=topology,
        inputs=SeasonalDeciduousPhenologyInputs(
            soil_patch=jnp.asarray(soil_patch),
            patch_column_index=jnp.asarray(patch_column),
            patch_gridcell_index=jnp.asarray(patch_gridcell),
            vegetation_type_patch=jnp.asarray(vegetation_type),
            daylength_seconds_gridcell=current_daylength,
            previous_daylength_seconds_gridcell=previous_daylength,
            third_soil_layer_temperature_column=jnp.asarray(
                third_soil_layer_temperature
            ),
            annual_average_temperature_2m_patch=jnp.asarray(values["annavg_t2m"]),
            carbon_storage_patch_pool=jnp.asarray(
                _stack_patch_pools(values, CARBON_STORAGE_POOL_NAMES)
            ),
            nitrogen_storage_patch_pool=jnp.asarray(
                _stack_patch_pools(values, NITROGEN_STORAGE_POOL_NAMES)
            ),
        ),
        state=SeasonalDeciduousPhenologyState(
            dormant_flag_patch=jnp.asarray(values["dormant_flag"]),
            days_active_patch=jnp.asarray(values["days_active"]),
            onset_flag_patch=jnp.asarray(values["onset_flag"]),
            onset_counter_seconds_patch=jnp.asarray(values["onset_counter"]),
            onset_gdd_flag_patch=jnp.asarray(values["onset_gddflag"]),
            onset_growing_degree_days_patch=jnp.asarray(values["onset_gdd"]),
            offset_flag_patch=jnp.asarray(values["offset_flag"]),
            offset_counter_seconds_patch=jnp.asarray(values["offset_counter"]),
            carbon_transfer_patch_pool=jnp.asarray(
                _stack_patch_pools(values, CARBON_TRANSFER_POOL_NAMES)
            ),
            nitrogen_transfer_patch_pool=jnp.asarray(
                _stack_patch_pools(values, NITROGEN_TRANSFER_POOL_NAMES)
            ),
            previous_leaf_carbon_to_litter_patch=jnp.asarray(
                values["prev_leafc_to_litter"]
            ),
            previous_fine_root_carbon_to_litter_patch=jnp.asarray(
                values["prev_frootc_to_litter"]
            ),
            pft_may_exist_patch=jnp.ones(patch_count, dtype=jnp.bool_),
        ),
        phenology_rates=EvergreenPhenologyState(
            background_litterfall_rate_patch=jnp.asarray(values["bglfr"]),
            background_transfer_growth_rate_patch=jnp.asarray(values["bgtr"]),
            long_growing_season_factor_patch=jnp.asarray(values["lgsf"]),
        ),
        parameters=SeasonalDeciduousPhenologyParameters(
            seasonal_deciduous_pft=jnp.asarray(seasonal_pft),
            woody_pft=jnp.asarray(woody_pft),
            time_step_seconds=np.float64(clock.step_seconds),
            seconds_per_day=np.float64(SECONDS_PER_DAY),
            critical_daylength_seconds=scalar_parameters["crit_dayl"],
            onset_period_days=scalar_parameters["ndays_on"],
            offset_period_days=scalar_parameters["ndays_off"],
            storage_transfer_fraction=scalar_parameters["fstor2tran"],
            use_dynamic_vegetation=False,
        ),
        restart_path=restart,
        parameter_data_path=parameter_data,
    )


__all__ = [
    "SeasonalDeciduousPhenologyNativeData",
    "load_seasonal_deciduous_phenology_native_data",
]
