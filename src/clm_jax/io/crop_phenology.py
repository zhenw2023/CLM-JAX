
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import SECONDS_PER_DAY
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import (
    SubgridTopology,
    advance_restart_clock,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)
from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    initialize_cn_transient_flux_state,
)
from clm_jax.physics.crop_phenology import (
    CropPhenologyInputs,
    CropPhenologyParameters,
    CropPhenologyState,
)
from clm_jax.physics.evergreen_phenology import (
    CARBON_TRANSFER_POOL_NAMES,
    NITROGEN_TRANSFER_POOL_NAMES,
    EvergreenPhenologyState,
)
from clm_jax.physics.seasonal_deciduous_phenology import (
    SeasonalDeciduousPhenologyState,
)

_SPECIAL_VALUE = np.float64(1.0e36)
_MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_PARAMETER_VARIABLES = (
    "leaf_long",
    "leafcn",
    "manunitro",
    "mxmat",
    "min_planting_temp",
    "planting_temp",
    "gddmin",
    "hybgdd",
    "lfemerg",
    "grnfill",
    "min_NH_planting_date",
    "max_NH_planting_date",
    "min_SH_planting_date",
    "max_SH_planting_date",
    "mergetoclmpft",
)
_COMMON_STATE_VARIABLES = (
    "dormant_flag",
    "days_active",
    "onset_flag",
    "onset_counter",
    "onset_gddflag",
    "onset_gdd",
    "offset_flag",
    "offset_counter",
    "prev_leafc_to_litter",
    "prev_frootc_to_litter",
)
_RESTART_VARIABLES = (
    "T_REF2M",
    "T_REF2M_MIN",
    "T_REF2M_MAX",
    "T10_VALUE",
    "TDM5_VALUE",
    "TDM10_VALUE",
    "gdd020",
    "gdd820",
    "gdd1020",
    "GDDPLANT_VALUE",
    "GDDTSOI_VALUE",
    "harvdate",
    "croplive",
    "cropplant",
    "vf",
    "cphase",
    "fert_counter",
    "fert",
    "peaklai",
    "idop",
    "gddmaturity",
    "huileaf",
    "huigrain",
    "cumvd",
    "hdidx",
    "bglfr",
    "bgtr",
    "lgsf",
    "leafc_xfer",
    "leafn_xfer",
    "tlai",
    "totvegc",
    "SNOW_DEPTH",
    *_COMMON_STATE_VARIABLES,
    *CARBON_TRANSFER_POOL_NAMES,
    *NITROGEN_TRANSFER_POOL_NAMES,
)
_WINTER_WHEAT_NAMES = {"winter_wheat", "irrigated_winter_wheat"}
_SOYBEAN_NAMES = {
    "temperate_soybean",
    "irrigated_temperate_soybean",
    "tropical_soybean",
    "irrigated_tropical_soybean",
}
_CORN_OR_SUGARCANE_NAMES = {
    "temperate_corn",
    "irrigated_temperate_corn",
    "tropical_corn",
    "irrigated_tropical_corn",
    "sugarcane",
    "irrigated_sugarcane",
}
_SPRING_WHEAT_COTTON_OR_RICE_NAMES = {
    "spring_wheat",
    "irrigated_spring_wheat",
    "cotton",
    "irrigated_cotton",
    "rice",
    "irrigated_rice",
}


class CropPhenologyNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: CropPhenologyInputs
    state: CropPhenologyState
    common_state: SeasonalDeciduousPhenologyState
    phenology_rates: EvergreenPhenologyState
    cn_fluxes: CnTransientFluxState
    parameters: CropPhenologyParameters
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


def _decode_pft_names(characters: jnp.ndarray) -> tuple[str, ...]:
    raw = np.asarray(characters)
    if raw.ndim != 2 or raw.dtype.kind not in {"S", "U"}:
        raise ValueError("parameter pftname must be a two-dimensional character array")
    names: list[str] = []
    for row in raw:
        if raw.dtype.kind == "S":
            name = b"".join(np.asarray(row, dtype="S1").tolist()).decode("ascii")
        else:
            name = "".join(np.asarray(row, dtype=str).tolist())
        names.append(name.rstrip())
    if len(set(names)) != len(names):
        raise ValueError("parameter pftname entries must be unique")
    return tuple(names)


def _load_pft_names(parameter_data: Path) -> tuple[str, ...]:
    from netCDF4 import Dataset  

    with Dataset(parameter_data, mode="r") as dataset:
        if "pftname" not in dataset.variables:
            raise ValueError("parameter file lacks pftname")
        characters = np.asarray(dataset.variables["pftname"][:])
    return _decode_pft_names(characters)


def _julian_day_from_month_day(value: int) -> np.int32:
    month = value // 100
    day = value % 100
    if month < 1 or month > 12 or day < 1 or day > 31:
        raise ValueError(f"invalid NO_LEAP planting date {value}")
    julian_day = sum(_MONTH_LENGTHS[: month - 1]) + day
    if julian_day > sum(_MONTH_LENGTHS):
        raise ValueError(f"planting date {value} lies beyond the NO_LEAP year")
    return np.int32(julian_day)


def _planting_days(
    parameter_values: dict[str, jnp.ndarray],
    pft_names: tuple[str, ...],
    first_crop_pft: int,
) -> tuple[np.ndarray, np.ndarray]:
    pft_count = len(pft_names)
    minimum = np.full((pft_count, 2), np.iinfo(np.int32).max, dtype=np.int32)
    maximum = np.full((pft_count, 2), np.iinfo(np.int32).max, dtype=np.int32)
    merge_target = np.asarray(parameter_values["mergetoclmpft"], dtype=np.int64)
    if merge_target.shape != (pft_count,):
        raise ValueError("mergetoclmpft must share the PFT axis")
    known_to_model = np.zeros(pft_count, dtype=bool)
    known_to_model[0] = True
    targets = merge_target[1:]
    if np.any((targets < 0) | (targets >= pft_count)):
        raise ValueError("mergetoclmpft contains an out-of-range PFT")
    known_to_model[targets] = True
    native_names = (
        ("min_NH_planting_date", minimum, 0),
        ("max_NH_planting_date", maximum, 0),
        ("min_SH_planting_date", minimum, 1),
        ("max_SH_planting_date", maximum, 1),
    )
    for variable_name, destination, hemisphere in native_names:
        values = np.asarray(parameter_values[variable_name], dtype=np.int64)
        if values.shape != (pft_count,):
            raise ValueError(f"{variable_name} must share the PFT axis")
        for pft_index in range(first_crop_pft, pft_count):
            if not known_to_model[pft_index]:
                continue
            destination[pft_index, hemisphere] = _julian_day_from_month_day(
                int(values[pft_index])
            )
    return minimum, maximum


def _pft_class_flags(
    pft_names: tuple[str, ...],
    selected_names: set[str],
) -> np.ndarray:
    missing = selected_names.difference(pft_names)
    if missing:
        raise ValueError(f"parameter file lacks required crop PFTs: {sorted(missing)}")
    return np.asarray([name in selected_names for name in pft_names], dtype=bool)


def _current_julian_day(current_date: int) -> np.int32:
    month = (current_date // 100) % 100
    day = current_date % 100
    return _julian_day_from_month_day(month * 100 + day)


def _stack_patch_pools(
    values: dict[str, jnp.ndarray],
    names: tuple[str, ...],
) -> np.ndarray:
    return np.stack(
        [np.asarray(values[name], dtype=np.float64) for name in names],
        axis=1,
    )


def _fertilizer_nitrogen_patch(
    topology: SubgridTopology,
    vegetation_type: np.ndarray,
    crop_patch: np.ndarray,
    surface_data: Path,
    merge_to_pft: np.ndarray,
    *,
    irrigate: bool,
    crop_cft_lower_bound: int,
) -> np.ndarray:
    surface = load_native_variables(
        surface_data,
        ["cft", "PCT_CFT", "CONST_FERTNITRO_CFT"],
        dtype=None,
    )
    cft = np.asarray(surface["cft"], dtype=np.int32)
    crop_weight = np.asarray(surface["PCT_CFT"], dtype=np.float64).copy()
    fertilizer = np.asarray(
        surface["CONST_FERTNITRO_CFT"], dtype=np.float64
    ).copy()
    if (
        fertilizer.ndim != 3
        or crop_weight.shape != fertilizer.shape
        or fertilizer.shape[0] != cft.shape[0]
    ):
        raise ValueError(
            "surface crop weight and fertilizer must share native "
            "(cft, lat, lon) axes"
        )
    merge_to_pft = np.asarray(merge_to_pft, dtype=np.int32)
    pft_count = merge_to_pft.shape[0]
    cft_axis = np.full(pft_count, -1, dtype=np.int32)
    if np.any((cft < 0) | (cft >= pft_count)) or len(np.unique(cft)) != len(cft):
        raise ValueError("surface cft coordinate is invalid")
    cft_axis[cft] = np.arange(cft.shape[0], dtype=np.int32)
    if not np.array_equal(cft, np.arange(cft[0], cft[-1] + 1)):
        raise ValueError("surface cft coordinate must be contiguous")
    if cft[0] != crop_cft_lower_bound:
        raise ValueError("surface cft lower bound disagrees with the parameter file")

    crop_weight /= np.float64(100.0)

    if not irrigate:
        rainfed = np.arange(crop_cft_lower_bound, cft[-1], 2, dtype=np.int32)
        irrigated = rainfed + 1
        rainfed_axis = cft_axis[rainfed]
        irrigated_axis = cft_axis[irrigated]
        crop_weight[rainfed_axis] = (
            crop_weight[rainfed_axis] + crop_weight[irrigated_axis]
        )
        crop_weight[irrigated_axis] = 0.0

    for source_pft in cft:
        destination_pft = int(merge_to_pft[source_pft])
        if destination_pft == source_pft:
            continue
        if destination_pft < 0 or destination_pft >= pft_count:
            raise ValueError("parameter mergetoclmpft contains an invalid PFT")
        destination_axis = int(cft_axis[destination_pft])
        if destination_axis < 0:
            raise ValueError("parameter mergetoclmpft targets a non-CFT PFT")
        source_axis = int(cft_axis[source_pft])
        weight_to = crop_weight[destination_axis].copy()
        weight_from = crop_weight[source_axis].copy()
        weight_merged = weight_to + weight_from
        positive = weight_merged > 0.0
        fertilizer_to = fertilizer[destination_axis].copy()
        fertilizer_from = fertilizer[source_axis]
        fertilizer[destination_axis, positive] = (
            weight_to[positive] * fertilizer_to[positive]
            + weight_from[positive] * fertilizer_from[positive]
        ) / weight_merged[positive]
        crop_weight[destination_axis] = weight_merged
        crop_weight[source_axis] = 0.0

    gridcell = np.asarray(topology.patch.gridcell, dtype=np.int32)
    latitude_index = np.asarray(topology.grid.latitude_index, dtype=np.int32)
    longitude_index = np.asarray(topology.grid.longitude_index, dtype=np.int32)
    if np.any(
        (latitude_index < 0)
        | (latitude_index >= fertilizer.shape[1])
        | (longitude_index < 0)
        | (longitude_index >= fertilizer.shape[2])
    ):
        raise ValueError("restart grid indices lie outside the surface grid")
    selected_axis = cft_axis[vegetation_type[crop_patch]]
    if np.any(selected_axis < 0):
        raise ValueError("a prognostic crop PFT is absent from surface cft")
    selected_gridcell = gridcell[crop_patch]
    result = np.full(vegetation_type.shape, _SPECIAL_VALUE, dtype=np.float64)
    result[crop_patch] = fertilizer[
        selected_axis,
        latitude_index[selected_gridcell],
        longitude_index[selected_gridcell],
    ]
    return result


def load_crop_phenology_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
) -> CropPhenologyNativeData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    phenology_namelist = read_case_namelist_group(case, "lnd_in", "cnphenology")
    for key, expected in {
        "use_cn": True,
        "use_crop": True,
        "use_fates": False,
    }.items():
        if general.get(key) is not expected:
            raise ValueError(f"native crop adapter requires {key}={expected!r}")
    if case.value("CALENDAR").upper() != "NO_LEAP":
        raise ValueError("native crop adapter currently requires NO_LEAP")
    compset = case.value("COMPSET", "").upper()
    if "BGC-CROP" not in compset or "CNDV" in compset:
        raise ValueError("native crop adapter requires a non-CNDV BGC-Crop compset")

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    parameter_data = _selected_input_path(case, "paramfile", parameter_data_path)
    surface_data = _selected_input_path(case, "fsurdat", surface_data_path)
    topology = load_subgrid_topology(restart)
    patch_count = topology.patch.gridcell.shape[0]
    column_count = topology.column.gridcell.shape[0]

    parameter_values = load_native_variables(
        parameter_data,
        list(_PARAMETER_VARIABLES),
        masked_integer_fill_value=0,
        masked_floating_fill_value=0.0,
    )
    pft_names = _load_pft_names(parameter_data)
    pft_count = len(pft_names)
    try:
        first_crop_pft = pft_names.index("temperate_corn")
    except ValueError as error:
        raise ValueError("parameter file lacks the prognostic-crop lower bound") from error
    minimum_planting_day, maximum_planting_day = _planting_days(
        parameter_values,
        pft_names,
        first_crop_pft,
    )
    for name in _PARAMETER_VARIABLES[:10]:
        if np.asarray(parameter_values[name]).shape != (pft_count,):
            raise ValueError(f"parameter {name!r} must share the PFT axis")

    vegetation_type = np.asarray(topology.patch.vegetation_type, dtype=np.int32)
    if np.any((vegetation_type < 0) | (vegetation_type >= pft_count)):
        raise ValueError("restart vegetation type lies outside PFT parameters")
    active = np.asarray(topology.patch.active, dtype=bool)
    crop_patch = active & (vegetation_type >= first_crop_pft)
    patch_column = np.asarray(topology.patch.column, dtype=np.int32)
    patch_gridcell = np.asarray(topology.patch.gridcell, dtype=np.int32)
    if np.any((patch_column < 0) | (patch_column >= column_count)):
        raise ValueError("restart patch-to-column topology is out of bounds")
    if np.any(
        (patch_gridcell < 0)
        | (patch_gridcell >= topology.grid.latitude_degrees.shape[0])
    ):
        raise ValueError("restart patch-to-gridcell topology is out of bounds")

    use_carbon_13 = bool(general.get("use_c13", False))
    use_carbon_14 = bool(general.get("use_c14", False))
    restart_names = list(_RESTART_VARIABLES)
    if use_carbon_13:
        restart_names.extend(("C13_LEAFC_XFER", "C13_TOTVEGC"))
    if use_carbon_14:
        restart_names.extend(("C14_LEAFC_XFER", "C14_TOTVEGC"))
    values = load_native_variables(
        restart,
        restart_names,
        masked_integer_fill_value=999,
        masked_floating_fill_value=float(_SPECIAL_VALUE),
    )
    patch_variables = set(_RESTART_VARIABLES).difference({"SNOW_DEPTH"})
    for name in patch_variables:
        if np.asarray(values[name]).shape != (patch_count,):
            raise ValueError(f"restart variable {name!r} must contain every patch")
    if np.asarray(values["SNOW_DEPTH"]).shape != (column_count,):
        raise ValueError("restart SNOW_DEPTH must contain every column")
    for name in restart_names:
        array = np.asarray(values[name])
        if array.shape == (patch_count,) and not np.all(np.isfinite(array[crop_patch])):
            raise ValueError(f"prognostic crop patches contain missing {name}")

    for name in ("croplive", "cropplant"):
        selected = np.asarray(values[name])[crop_patch]
        if not np.all((selected == 0) | (selected == 1)):
            raise ValueError(f"restart {name} must contain logical zero or one")
    grid_latitude = np.asarray(topology.grid.latitude_degrees, dtype=np.float64)
    hemisphere = np.where(grid_latitude[patch_gridcell] > 0.0, 0, 1).astype(np.int32)
    fertilizer_nitrogen = _fertilizer_nitrogen_patch(
        topology,
        vegetation_type,
        crop_patch,
        surface_data,
        np.asarray(parameter_values["mergetoclmpft"], dtype=np.int32),
        irrigate=bool(general.get("irrigate", False)),
        crop_cft_lower_bound=pft_names.index("c3_crop"),
    )
    if not np.all(np.isfinite(fertilizer_nitrogen[crop_patch])):
        raise ValueError("surface fertilizer contains missing crop values")

    clock = advance_restart_clock(
        load_restart_clock(restart),
        case.value("CALENDAR"),
    )
    initial_seed = phenology_namelist.get("initial_seed_at_planting", 3.0)
    if not isinstance(initial_seed, int | float) or not np.isfinite(initial_seed):
        raise ValueError("initial_seed_at_planting must be one finite number")
    c13_leaf_transfer = (
        np.asarray(values["C13_LEAFC_XFER"], dtype=np.float64)
        if use_carbon_13
        else np.zeros(patch_count, dtype=np.float64)
    )
    c13_total_carbon = (
        np.asarray(values["C13_TOTVEGC"], dtype=np.float64)
        if use_carbon_13
        else np.zeros(patch_count, dtype=np.float64)
    )
    c14_leaf_transfer = (
        np.asarray(values["C14_LEAFC_XFER"], dtype=np.float64)
        if use_carbon_14
        else np.zeros(patch_count, dtype=np.float64)
    )
    c14_total_carbon = (
        np.asarray(values["C14_TOTVEGC"], dtype=np.float64)
        if use_carbon_14
        else np.zeros(patch_count, dtype=np.float64)
    )

    pft_real = {
        name: jnp.asarray(parameter_values[name], dtype=jnp.float64)
        for name in _PARAMETER_VARIABLES[:10]
    }
    return CropPhenologyNativeData(
        topology=topology,
        inputs=CropPhenologyInputs(
            prognostic_crop_patch=jnp.asarray(crop_patch),
            patch_column_index=jnp.asarray(patch_column),
            vegetation_type_patch=jnp.asarray(vegetation_type),
            hemisphere_index_patch=jnp.asarray(hemisphere),
            temperature_2m_patch=jnp.asarray(values["T_REF2M"]),
            minimum_temperature_2m_patch=jnp.asarray(values["T_REF2M_MIN"]),
            maximum_temperature_2m_patch=jnp.asarray(values["T_REF2M_MAX"]),
            ten_day_temperature_2m_patch=jnp.asarray(values["T10_VALUE"]),
            five_day_minimum_temperature_2m_patch=jnp.asarray(values["TDM5_VALUE"]),
            ten_day_minimum_temperature_2m_patch=jnp.asarray(values["TDM10_VALUE"]),
            twenty_year_gdd0_patch=jnp.asarray(values["gdd020"]),
            twenty_year_gdd8_patch=jnp.asarray(values["gdd820"]),
            twenty_year_gdd10_patch=jnp.asarray(values["gdd1020"]),
            growing_degree_days_topsoil_patch=jnp.asarray(values["GDDTSOI_VALUE"]),
            fertilizer_nitrogen_patch=jnp.asarray(fertilizer_nitrogen),
            total_leaf_area_index_patch=jnp.asarray(values["tlai"]),
            peak_leaf_area_index_patch=jnp.asarray(values["peaklai"], dtype=jnp.int32),
            snow_depth_column=jnp.asarray(values["SNOW_DEPTH"]),
            total_vegetation_carbon_patch=jnp.asarray(values["totvegc"]),
            c13_total_vegetation_carbon_patch=jnp.asarray(c13_total_carbon),
            c14_total_vegetation_carbon_patch=jnp.asarray(c14_total_carbon),
            current_julian_day=jnp.asarray(
                _current_julian_day(clock.current_date),
                dtype=jnp.int32,
            ),
            seconds_of_day=jnp.asarray(clock.current_time_of_day, dtype=jnp.int32),
        ),
        state=CropPhenologyState(
            harvest_date_patch=jnp.asarray(values["harvdate"], dtype=jnp.int32),
            planting_date_patch=jnp.asarray(values["idop"], dtype=jnp.int32),
            crop_live_patch=jnp.asarray(values["croplive"] == 1),
            crop_planted_patch=jnp.asarray(values["cropplant"] == 1),
            vernalization_factor_patch=jnp.asarray(values["vf"]),
            crop_phase_patch=jnp.asarray(values["cphase"]),
            growing_degree_days_since_planting_patch=jnp.asarray(
                values["GDDPLANT_VALUE"]
            ),
            growing_degree_days_to_maturity_patch=jnp.asarray(values["gddmaturity"]),
            heat_units_to_leaf_emergence_patch=jnp.asarray(values["huileaf"]),
            heat_units_to_grain_fill_patch=jnp.asarray(values["huigrain"]),
            cumulative_vernalization_days_patch=jnp.asarray(values["cumvd"]),
            hardening_index_patch=jnp.asarray(values["hdidx"]),
            fertilizer_counter_seconds_patch=jnp.asarray(values["fert_counter"]),
            fertilizer_flux_patch=jnp.asarray(values["fert"]),
            c13_leaf_carbon_transfer_patch=jnp.asarray(c13_leaf_transfer),
            c14_leaf_carbon_transfer_patch=jnp.asarray(c14_leaf_transfer),
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
        phenology_rates=EvergreenPhenologyState(
            background_litterfall_rate_patch=jnp.asarray(values["bglfr"]),
            background_transfer_growth_rate_patch=jnp.asarray(values["bgtr"]),
            long_growing_season_factor_patch=jnp.asarray(values["lgsf"]),
        ),
        cn_fluxes=initialize_cn_transient_flux_state(
            patch_count, column_count, fieldwise=True
        ),
        parameters=CropPhenologyParameters(
            leaf_longevity_years_pft=pft_real["leaf_long"],
            leaf_carbon_nitrogen_ratio_pft=pft_real["leafcn"],
            manure_nitrogen_pft=pft_real["manunitro"],
            maximum_days_to_maturity_pft=pft_real["mxmat"],
            minimum_planting_temperature_pft=pft_real["min_planting_temp"],
            planting_temperature_pft=pft_real["planting_temp"],
            minimum_growing_degree_days_pft=pft_real["gddmin"],
            hybrid_growing_degree_days_pft=pft_real["hybgdd"],
            leaf_emergence_fraction_pft=pft_real["lfemerg"],
            grain_fill_fraction_pft=pft_real["grnfill"],
            minimum_planting_day_pft_hemisphere=jnp.asarray(minimum_planting_day),
            maximum_planting_day_pft_hemisphere=jnp.asarray(maximum_planting_day),
            winter_wheat_pft=jnp.asarray(
                _pft_class_flags(pft_names, _WINTER_WHEAT_NAMES)
            ),
            soybean_pft=jnp.asarray(_pft_class_flags(pft_names, _SOYBEAN_NAMES)),
            corn_or_sugarcane_pft=jnp.asarray(
                _pft_class_flags(pft_names, _CORN_OR_SUGARCANE_NAMES)
            ),
            spring_wheat_cotton_or_rice_pft=jnp.asarray(
                _pft_class_flags(pft_names, _SPRING_WHEAT_COTTON_OR_RICE_NAMES)
            ),
            year_start_day_hemisphere=jnp.asarray([1, 182], dtype=jnp.int32),
            time_step_seconds=np.float64(clock.step_seconds),
            radiation_step_seconds=np.float64(clock.step_seconds),
            days_per_year=np.float64(365.0),
            seconds_per_day=np.float64(SECONDS_PER_DAY),
            special_value=_SPECIAL_VALUE,
            initial_seed_carbon=np.float64(initial_seed),
            carbon_13_ratio=np.float64(0.011046391077221919),
            carbon_14_ratio=np.float64(1.0e-12),
            use_fertilizer=bool(general.get("use_fertilizer", False)),
            use_carbon_13=use_carbon_13,
            use_carbon_14=use_carbon_14,
        ),
        restart_path=restart,
        parameter_data_path=parameter_data,
        surface_data_path=surface_data,
    )


__all__ = ["CropPhenologyNativeData", "load_crop_phenology_native_data"]
