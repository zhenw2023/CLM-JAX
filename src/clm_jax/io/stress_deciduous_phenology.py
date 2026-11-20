
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import SECONDS_PER_DAY
from clm_jax.io.canopy_temperature_restart import (
    column_soil_hydraulic_properties_20sl_8p5m,
    soil_layer_geometry_20sl_8p5m,
)
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
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
from clm_jax.physics.seasonal_deciduous_phenology import (
    SeasonalDeciduousPhenologyState,
)
from clm_jax.physics.solar import FIXED_YEAR_2000_ORBIT, solar_declination
from clm_jax.physics.solar_runtime import native_grid_radians
from clm_jax.physics.stress_deciduous_phenology import (
    StressDeciduousPhenologyInputs,
    StressDeciduousPhenologyParameters,
    StressDeciduousPhenologyState,
)

_NUMBER_OF_GROUND_LEVELS = 25
_NUMBER_OF_TOTAL_LEVELS = 37
_THIRD_GROUND_LEVEL = 2
_STATE_VARIABLES = (
    "dormant_flag",
    "days_active",
    "onset_flag",
    "onset_counter",
    "onset_gddflag",
    "onset_fdd",
    "onset_gdd",
    "onset_swi",
    "offset_flag",
    "offset_counter",
    "offset_fdd",
    "offset_swi",
    "bglfr",
    "bgtr",
    "lgsf",
    "prev_leafc_to_litter",
    "prev_frootc_to_litter",
)
_PARAMETER_VARIABLES = (
    "stress_decid",
    "leaf_long",
    "woody",
    "crit_onset_fdd",
    "crit_onset_swi",
    "soilpsi_on",
    "crit_offset_fdd",
    "crit_offset_swi",
    "soilpsi_off",
    "ndays_on",
    "ndays_off",
    "fstor2tran",
    "organic_max",
)
_POOL_VARIABLES = (
    *CARBON_STORAGE_POOL_NAMES,
    *CARBON_TRANSFER_POOL_NAMES,
    *NITROGEN_STORAGE_POOL_NAMES,
    *NITROGEN_TRANSFER_POOL_NAMES,
)
_RESTART_INPUT_VARIABLES = (
    "T_SOISNO",
    "H2OSOI_LIQ",
    "PREC10_VALUE",
    "annavg_t2m",
    "leafc",
    "frootc",
    *_POOL_VARIABLES,
    *_STATE_VARIABLES,
)
_SURFACE_VARIABLES = ("PCT_SAND", "PCT_CLAY", "ORGANIC")


class StressDeciduousPhenologyPersistentInputs(NamedTuple):

    soil_patch: jnp.ndarray
    patch_column_index: jnp.ndarray
    patch_gridcell_index: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    daylength_seconds_gridcell: jnp.ndarray
    ten_day_mean_precipitation_patch: jnp.ndarray
    third_soil_layer_temperature_column: jnp.ndarray
    annual_average_temperature_2m_patch: jnp.ndarray
    leaf_carbon_patch: jnp.ndarray
    fine_root_carbon_patch: jnp.ndarray
    carbon_storage_patch_pool: jnp.ndarray
    nitrogen_storage_patch_pool: jnp.ndarray


class StressDeciduousPhenologyNativeData(NamedTuple):

    topology: SubgridTopology
    persistent_inputs: StressDeciduousPhenologyPersistentInputs
    common_state: SeasonalDeciduousPhenologyState
    state: StressDeciduousPhenologyState
    phenology_rates: EvergreenPhenologyState
    parameters: StressDeciduousPhenologyParameters
    restart_soil_water_potential_snapshot: jnp.ndarray
    restart_path: Path
    parameter_data_path: Path
    surface_data_path: Path


def _selected_input_path(
    case: CimeCase,
    key: str,
    override: str | Path | None,
) -> Path:
    if override is None:
        group = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        declared = group.get(key)
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError(f"clm_inparm in {case.case_docs / 'lnd_in'} lacks {key}")
        path = Path(case.expand(declared)).expanduser()
        if not path.is_absolute():
            path = case.root / path
        path = path.resolve()
    else:
        path = Path(override).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM {key} does not exist: {path}")
    return path


def _stack_patch_pools(
    values: dict[str, jnp.ndarray],
    names: tuple[str, ...],
) -> np.ndarray:
    return np.stack([np.asarray(values[name], dtype=np.float64) for name in names], axis=1)


def _noleap_calendar_day(clock: RestartClock) -> np.float64:
    ymd = int(clock.current_date)
    month = (ymd // 100) % 100
    day_of_month = ymd % 100
    month_lengths = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    if (
        month < 1
        or month > 12
        or day_of_month < 1
        or day_of_month > month_lengths[month - 1]
    ):
        raise ValueError(f"invalid NO_LEAP restart date {ymd}")
    if clock.current_time_of_day < 0 or clock.current_time_of_day >= int(
        SECONDS_PER_DAY
    ):
        raise ValueError("restart time of day must lie in [0, 86400)")
    integer_day = 1 + sum(month_lengths[: month - 1]) + day_of_month - 1
    return np.float64(integer_day) + np.float64(
        clock.current_time_of_day
    ) / np.float64(SECONDS_PER_DAY)


def _finite_scalar(name: str, value: jnp.ndarray) -> np.float64:
    array = np.asarray(value, dtype=np.float64)
    if array.size != 1 or not np.isfinite(array.reshape(())):
        raise ValueError(f"CLM parameter {name!r} must contain one finite scalar")
    return np.float64(array.reshape(()))


def _restart_soil_water_potential_snapshot(
    topology: SubgridTopology,
    liquid_water_total_level: np.ndarray,
    surface_values: dict[str, jnp.ndarray],
    organic_maximum: float,
    hydrology_column: np.ndarray,
) -> np.ndarray:

    properties = column_soil_hydraulic_properties_20sl_8p5m(
        topology,
        surface_values,
        organic_maximum,
    )
    _, _, thickness = soil_layer_geometry_20sl_8p5m()
    liquid = np.asarray(liquid_water_total_level, dtype=np.float64)[
        :,
        -_NUMBER_OF_GROUND_LEVELS + _THIRD_GROUND_LEVEL,
    ]
    selected = np.asarray(hydrology_column, dtype=bool)
    if selected.shape != liquid.shape:
        raise ValueError("hydrology-column mask must contain every column")
    porosity = properties.saturated_water_content_column_level[
        selected,
        _THIRD_GROUND_LEVEL,
    ]
    suction = properties.saturated_suction_column_level[
        selected,
        _THIRD_GROUND_LEVEL,
    ]
    exponent = properties.clapp_hornberger_b_column_level[
        selected,
        _THIRD_GROUND_LEVEL,
    ]
    selected_liquid = liquid[selected]
    volumetric_liquid = selected_liquid / (
        thickness[_THIRD_GROUND_LEVEL] * np.float64(1000.0)
    )
    fractional_saturation = np.maximum(
        volumetric_liquid / porosity,
        np.float64(0.001),
    )
    potential = suction * np.float64(-9.8e-6) * (
        fractional_saturation ** (-exponent)
    )
    potential = np.minimum(np.maximum(potential, np.float64(-15.0)), 0.0)
    result = np.full(liquid.shape, np.float64(-15.0), dtype=np.float64)
    result[selected] = np.where(
        selected_liquid > np.float64(0.0),
        potential,
        np.float64(-15.0),
    )
    return result


def assemble_stress_deciduous_phenology_inputs(
    native: StressDeciduousPhenologyNativeData,
    third_soil_layer_water_potential_column: jnp.ndarray,
) -> StressDeciduousPhenologyInputs:

    potential = jnp.asarray(
        third_soil_layer_water_potential_column,
        dtype=jnp.float64,
    )
    column_count = native.topology.column.gridcell.shape[0]
    if potential.shape != (column_count,):
        raise ValueError("third-layer soil water potential must contain every column")
    persistent = native.persistent_inputs
    return StressDeciduousPhenologyInputs(
        soil_patch=persistent.soil_patch,
        patch_column_index=persistent.patch_column_index,
        patch_gridcell_index=persistent.patch_gridcell_index,
        vegetation_type_patch=persistent.vegetation_type_patch,
        daylength_seconds_gridcell=persistent.daylength_seconds_gridcell,
        ten_day_mean_precipitation_patch=(
            persistent.ten_day_mean_precipitation_patch
        ),
        third_soil_layer_temperature_column=(
            persistent.third_soil_layer_temperature_column
        ),
        third_soil_layer_water_potential_column=potential,
        annual_average_temperature_2m_patch=(
            persistent.annual_average_temperature_2m_patch
        ),
        leaf_carbon_patch=persistent.leaf_carbon_patch,
        fine_root_carbon_patch=persistent.fine_root_carbon_patch,
        carbon_storage_patch_pool=persistent.carbon_storage_patch_pool,
        nitrogen_storage_patch_pool=persistent.nitrogen_storage_patch_pool,
    )


def load_stress_deciduous_phenology_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
) -> StressDeciduousPhenologyNativeData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    bgc_shared = read_case_namelist_group(case, "lnd_in", "bgc_shared")
    for key, expected in {
        "use_cn": True,
        "use_crop": True,
        "use_fates": False,
    }.items():
        if general.get(key) is not expected:
            raise ValueError(f"native stress adapter requires {key}={expected!r}")
    if bgc_shared.get("constrain_stress_deciduous_onset") is not True:
        raise ValueError(
            "native stress adapter requires "
            "constrain_stress_deciduous_onset=True"
        )
    if general.get("soil_layerstruct") != "20SL_8.5m":
        raise ValueError("native stress adapter requires soil_layerstruct='20SL_8.5m'")
    if case.value("CALENDAR").upper() != "NO_LEAP":
        raise ValueError("native stress adapter currently requires NO_LEAP")
    compset = case.value("COMPSET", "").upper()
    if "BGC-CROP" not in compset or "CNDV" in compset:
        raise ValueError("native stress adapter requires a non-CNDV BGC-Crop compset")

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
    if dimensions.get("levtot") != _NUMBER_OF_TOTAL_LEVELS:
        raise ValueError("restart must contain the native 37 snow-plus-ground levels")
    parameter_data = _selected_input_path(case, "paramfile", parameter_data_path)
    surface_data = _selected_input_path(case, "fsurdat", surface_data_path)
    topology = load_subgrid_topology(restart)
    patch_count = topology.patch.gridcell.shape[0]
    column_count = topology.column.gridcell.shape[0]
    gridcell_count = topology.grid.latitude_degrees.shape[0]

    values = load_native_variables(
        restart,
        list(_RESTART_INPUT_VARIABLES),
        dtype=np.float64,
    )
    for name in (
        "PREC10_VALUE",
        "annavg_t2m",
        "leafc",
        "frootc",
        *_POOL_VARIABLES,
        *_STATE_VARIABLES,
    ):
        if values[name].shape != (patch_count,):
            raise ValueError(f"restart variable {name!r} must contain every patch")
    for name in ("T_SOISNO", "H2OSOI_LIQ"):
        if values[name].shape != (column_count, _NUMBER_OF_TOTAL_LEVELS):
            raise ValueError(f"restart variable {name!r} must have native levels")

    pft_values = load_native_variables(
        parameter_data,
        list(_PARAMETER_VARIABLES),
        dtype=np.float64,
    )
    stress_pft = np.asarray(pft_values["stress_decid"], dtype=np.float64)
    leaf_longevity = np.asarray(pft_values["leaf_long"], dtype=np.float64)
    woody_pft = np.asarray(pft_values["woody"], dtype=np.float64)
    if (
        stress_pft.ndim != 1
        or leaf_longevity.shape != stress_pft.shape
        or woody_pft.shape != stress_pft.shape
    ):
        raise ValueError("stress_decid, leaf_long and woody must share one PFT axis")
    if not np.all((stress_pft == 0.0) | (stress_pft == 1.0)):
        raise ValueError("CLM stress_decid flags must contain only zero or one")
    if not np.all((woody_pft == 0.0) | (woody_pft == 1.0)):
        raise ValueError("CLM woody flags must contain only zero or one")
    if np.any(leaf_longevity[stress_pft == 1.0] <= 0.0):
        raise ValueError("stress-deciduous PFT leaf longevity must be positive")
    scalar_parameters = {
        name: _finite_scalar(name, pft_values[name])
        for name in _PARAMETER_VARIABLES[3:]
    }

    active = np.asarray(topology.patch.active, dtype=bool)
    landunit = np.asarray(topology.patch.landunit_type, dtype=np.int32)
    soil_patch = active & (
        (landunit == LANDUNIT_SOIL) | (landunit == LANDUNIT_CROP)
    )
    vegetation_type = np.asarray(topology.patch.vegetation_type, dtype=np.int32)
    if np.any((vegetation_type < 0) | (vegetation_type >= stress_pft.shape[0])):
        raise ValueError("restart vegetation type lies outside PFT parameter vectors")
    patch_column = np.asarray(topology.patch.column, dtype=np.int32)
    patch_gridcell = np.asarray(topology.patch.gridcell, dtype=np.int32)
    if np.any((patch_column < 0) | (patch_column >= column_count)):
        raise ValueError("restart patch-to-column topology is out of bounds")
    if np.any((patch_gridcell < 0) | (patch_gridcell >= gridcell_count)):
        raise ValueError("restart patch-to-gridcell topology is out of bounds")
    stress_patch = soil_patch & (stress_pft[vegetation_type] == 1.0)
    for name in (
        "PREC10_VALUE",
        "annavg_t2m",
        "leafc",
        "frootc",
        *_POOL_VARIABLES,
        *_STATE_VARIABLES,
    ):
        if not np.all(np.isfinite(np.asarray(values[name])[stress_patch])):
            raise ValueError(f"stress-deciduous patches contain missing {name}")
    stress_column = np.zeros(column_count, dtype=bool)
    stress_column[patch_column[stress_patch]] = True
    column_landunit = np.asarray(
        topology.column.landunit_type,
        dtype=np.int32,
    )
    hydrology_column = np.asarray(topology.column.active, dtype=bool) & (
        (column_landunit == LANDUNIT_SOIL) | (column_landunit == LANDUNIT_CROP)
    )
    third_temperature = np.asarray(values["T_SOISNO"], dtype=np.float64)[
        :,
        -_NUMBER_OF_GROUND_LEVELS + _THIRD_GROUND_LEVEL,
    ]
    if not np.all(np.isfinite(third_temperature[stress_column])):
        raise ValueError("stress-deciduous columns contain missing soil temperature")

    surface_values = load_native_variables(
        surface_data,
        list(_SURFACE_VARIABLES),
        dtype=np.float64,
    )
    soil_water_potential_snapshot = _restart_soil_water_potential_snapshot(
        topology,
        np.asarray(values["H2OSOI_LIQ"], dtype=np.float64),
        surface_values,
        scalar_parameters["organic_max"],
        hydrology_column,
    )
    if not np.all(np.isfinite(soil_water_potential_snapshot[stress_column])):
        raise ValueError("stress-deciduous columns contain missing soil water potential")

    clock = advance_restart_clock(
        load_restart_clock(restart),
        case.value("CALENDAR"),
    )
    calendar_day = jnp.asarray(_noleap_calendar_day(clock), dtype=jnp.float64)
    declination, _ = solar_declination(calendar_day, FIXED_YEAR_2000_ORBIT)
    latitude_radians = native_grid_radians(topology.grid.latitude_degrees)
    current_daylength = daylength(latitude_radians, declination)

    return StressDeciduousPhenologyNativeData(
        topology=topology,
        persistent_inputs=StressDeciduousPhenologyPersistentInputs(
            soil_patch=jnp.asarray(soil_patch),
            patch_column_index=jnp.asarray(patch_column),
            patch_gridcell_index=jnp.asarray(patch_gridcell),
            vegetation_type_patch=jnp.asarray(vegetation_type),
            daylength_seconds_gridcell=current_daylength,
            ten_day_mean_precipitation_patch=jnp.asarray(
                values["PREC10_VALUE"]
            ),
            third_soil_layer_temperature_column=jnp.asarray(third_temperature),
            annual_average_temperature_2m_patch=jnp.asarray(values["annavg_t2m"]),
            leaf_carbon_patch=jnp.asarray(values["leafc"]),
            fine_root_carbon_patch=jnp.asarray(values["frootc"]),
            carbon_storage_patch_pool=jnp.asarray(
                _stack_patch_pools(values, CARBON_STORAGE_POOL_NAMES)
            ),
            nitrogen_storage_patch_pool=jnp.asarray(
                _stack_patch_pools(values, NITROGEN_STORAGE_POOL_NAMES)
            ),
        ),
        common_state=SeasonalDeciduousPhenologyState(
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
        state=StressDeciduousPhenologyState(
            onset_freezing_degree_days_patch=jnp.asarray(values["onset_fdd"]),
            onset_soil_water_index_patch=jnp.asarray(values["onset_swi"]),
            offset_freezing_degree_days_patch=jnp.asarray(values["offset_fdd"]),
            offset_soil_water_index_patch=jnp.asarray(values["offset_swi"]),
        ),
        phenology_rates=EvergreenPhenologyState(
            background_litterfall_rate_patch=jnp.asarray(values["bglfr"]),
            background_transfer_growth_rate_patch=jnp.asarray(values["bgtr"]),
            long_growing_season_factor_patch=jnp.asarray(values["lgsf"]),
        ),
        parameters=StressDeciduousPhenologyParameters(
            stress_deciduous_pft=jnp.asarray(stress_pft),
            leaf_longevity_years_pft=jnp.asarray(leaf_longevity),
            woody_pft=jnp.asarray(woody_pft),
            time_step_seconds=np.float64(clock.step_seconds),
            days_per_year=np.float64(365.0),
            critical_onset_freezing_days=scalar_parameters["crit_onset_fdd"],
            critical_onset_soil_water_index_days=(
                scalar_parameters["crit_onset_swi"]
            ),
            onset_soil_water_potential_mpa=scalar_parameters["soilpsi_on"],
            critical_offset_freezing_days=scalar_parameters["crit_offset_fdd"],
            critical_offset_soil_water_index_days=(
                scalar_parameters["crit_offset_swi"]
            ),
            offset_soil_water_potential_mpa=scalar_parameters["soilpsi_off"],
            onset_period_days=scalar_parameters["ndays_on"],
            offset_period_days=scalar_parameters["ndays_off"],
            storage_transfer_fraction=scalar_parameters["fstor2tran"],
            constrain_onset_by_precipitation=True,
        ),
        restart_soil_water_potential_snapshot=jnp.asarray(
            soil_water_potential_snapshot
        ),
        restart_path=restart,
        parameter_data_path=parameter_data,
        surface_data_path=surface_data,
    )


__all__ = [
    "StressDeciduousPhenologyNativeData",
    "StressDeciduousPhenologyPersistentInputs",
    "assemble_stress_deciduous_phenology_inputs",
    "load_stress_deciduous_phenology_native_data",
]
