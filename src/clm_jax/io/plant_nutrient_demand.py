
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE, SECONDS_PER_DAY
from clm_jax.io.canopy_temperature_restart import soil_layer_geometry_20sl_8p5m
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    SubgridTopology,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)
from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    initialize_cn_transient_flux_state,
)
from clm_jax.physics.crop_phenology import CropPhenologyState
from clm_jax.physics.decomposition_rates import NUMBER_OF_SOIL_LEVELS
from clm_jax.physics.plant_nutrient_demand import (
    PlantNutrientDemandDirectFluxState,
    PlantNutrientDemandInputs,
    PlantNutrientDemandParameters,
    PlantNutrientDemandState,
)

_PARAMETER_VARIABLES = (
    "woody",
    "froot_leaf",
    "croot_stem",
    "stem_leaf",
    "flivewd",
    "leafcn",
    "frootcn",
    "livewdcn",
    "deadwdcn",
    "graincn",
    "fleafcn",
    "ffrootcn",
    "fstemcn",
    "bfact",
    "aleaff",
    "arootf",
    "astemf",
    "arooti",
    "fleafi",
    "allconsl",
    "allconss",
    "grperc",
    "grpnow",
    "declfact",
    "season_decid",
    "stress_decid",
    "dayscrecover",
)
_RESTART_INPUT_VARIABLES = (
    "GDDPLANT_VALUE",
    "GDDTSOI_VALUE",
    "croplive",
    "gddmaturity",
    "huileaf",
    "huigrain",
    "peaklai",
    "xsmrpool",
    "leafc",
    "frootc",
    "livestemc",
    "livecrootc",
    "retransn",
    "annsum_npp",
    "leafn",
    "frootn",
    "livestemn",
    "annsum_potential_gpp",
    "annmax_retransn",
    "sminn_vr",
)
_RESTART_STATE_VARIABLES = (
    "aleafi",
    "astemi",
    "aleaf",
    "astem",
    "grain_flag",
    "c_allometry",
    "n_allometry",
    "tempsum_potential_gpp",
    "tempmax_retransn",
)
_RESTART_DIRECT_FLUX_VARIABLES = (
    "availc",
    "xsmrpool_recover",
    "plant_ndemand",
    "avail_retransn",
)
_SOYBEAN_NAMES = {
    "temperate_soybean",
    "irrigated_temperate_soybean",
    "tropical_soybean",
    "irrigated_tropical_soybean",
}


class PlantNutrientDemandPersistentInputs(NamedTuple):

    soil_patch: jnp.ndarray
    patch_column_index: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    growing_degree_days_since_planting_patch: jnp.ndarray
    growing_degree_days_topsoil_patch: jnp.ndarray
    crop_live_patch: jnp.ndarray
    growing_degree_days_to_maturity_patch: jnp.ndarray
    heat_units_to_leaf_emergence_patch: jnp.ndarray
    heat_units_to_grain_fill_patch: jnp.ndarray
    peak_leaf_area_index_patch: jnp.ndarray
    temporary_photosynthate_carbon_pool_patch: jnp.ndarray
    leaf_carbon_patch: jnp.ndarray
    fine_root_carbon_patch: jnp.ndarray
    live_stem_carbon_patch: jnp.ndarray
    live_coarse_root_carbon_patch: jnp.ndarray
    retranslocated_nitrogen_pool_patch: jnp.ndarray
    annual_net_primary_productivity_patch: jnp.ndarray
    leaf_nitrogen_patch: jnp.ndarray
    fine_root_nitrogen_patch: jnp.ndarray
    live_stem_nitrogen_patch: jnp.ndarray
    annual_potential_gpp_patch: jnp.ndarray
    annual_maximum_retranslocated_nitrogen_patch: jnp.ndarray
    soil_mineral_nitrogen_column_level: jnp.ndarray


class PlantNutrientDemandNativeData(NamedTuple):

    topology: SubgridTopology
    persistent_inputs: PlantNutrientDemandPersistentInputs
    state: PlantNutrientDemandState
    parameters: PlantNutrientDemandParameters
    restart_path: Path
    parameter_data_path: Path


class PlantNutrientDemandCall(NamedTuple):

    inputs: PlantNutrientDemandInputs
    state: PlantNutrientDemandState
    parameters: PlantNutrientDemandParameters


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


def _pft_names(parameter_data: Path) -> tuple[str, ...]:
    from netCDF4 import Dataset  

    with Dataset(parameter_data, mode="r") as dataset:
        if "pftname" not in dataset.variables:
            raise ValueError("parameter file lacks pftname")
        raw = np.asarray(dataset.variables["pftname"][:])
    if raw.ndim != 2 or raw.dtype.kind not in {"S", "U"}:
        raise ValueError("parameter pftname must be a character matrix")
    if raw.dtype.kind == "S":
        names = tuple(
            b"".join(np.asarray(row, dtype="S1").tolist()).decode("ascii").rstrip()
            for row in raw
        )
    else:
        names = tuple("".join(np.asarray(row, dtype=str).tolist()).rstrip() for row in raw)
    if len(set(names)) != len(names):
        raise ValueError("parameter pftname entries must be unique")
    return names


def _real_pft_parameter(
    values: dict[str, jnp.ndarray],
    name: str,
    pft_count: int,
) -> jnp.ndarray:
    host = np.asarray(values[name], dtype=np.float64)
    if host.shape != (pft_count,) or not np.all(np.isfinite(host)):
        raise ValueError(f"CLM parameter {name!r} must be one finite PFT vector")
    return jnp.asarray(host)


def _logical_pft_parameter(
    values: dict[str, jnp.ndarray],
    name: str,
    pft_count: int,
) -> jnp.ndarray:
    host = np.asarray(values[name], dtype=np.float64)
    if host.shape != (pft_count,) or not np.all((host == 0.0) | (host == 1.0)):
        raise ValueError(f"CLM parameter {name!r} must contain only zero or one")
    return jnp.asarray(host == 1.0)


def _finite_scalar(name: str, value: jnp.ndarray) -> np.float64:
    host = np.asarray(value, dtype=np.float64)
    if host.size != 1 or not np.isfinite(host.reshape(())):
        raise ValueError(f"CLM parameter {name!r} must contain one finite scalar")
    return np.float64(host.reshape(()))


def load_plant_nutrient_demand_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> PlantNutrientDemandNativeData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    for key, expected in {
        "use_cn": True,
        "use_crop": True,
        "use_fates": False,
        "use_flexiblecn": True,
    }.items():
        if general.get(key) is not expected:
            raise ValueError(f"native plant-demand adapter requires {key}={expected!r}")
    if general.get("use_vertsoilc", True) is not True:
        raise ValueError("native plant-demand adapter requires vertical soil carbon")
    if general.get("soil_layerstruct") != "20SL_8.5m":
        raise ValueError("native plant-demand adapter requires soil_layerstruct='20SL_8.5m'")

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    parameter_data = _selected_parameter_path(case, parameter_data_path)
    dimensions = load_dimension_sizes(restart)
    if dimensions.get("levgrnd", 0) < NUMBER_OF_SOIL_LEVELS:
        raise ValueError("restart lacks the twenty active decomposition levels")

    topology = load_subgrid_topology(restart)
    patch_count = topology.patch.gridcell.shape[0]
    column_count = topology.column.gridcell.shape[0]
    patch_active = np.asarray(topology.patch.active, dtype=bool)
    landunit_type = np.asarray(topology.patch.landunit_type, dtype=np.int32)
    soil_patch = patch_active & (
        (landunit_type == LANDUNIT_SOIL) | (landunit_type == LANDUNIT_CROP)
    )
    patch_column = np.asarray(topology.patch.column, dtype=np.int32)
    if np.any((patch_column < 0) | (patch_column >= column_count)):
        raise ValueError("restart patch-to-column topology is out of bounds")

    parameter_values = load_native_variables(
        parameter_data,
        list(_PARAMETER_VARIABLES),
        dtype=np.float64,
        masked_floating_fill_value=0.0,
    )
    pft_names = _pft_names(parameter_data)
    pft_count = len(pft_names)
    try:
        first_crop_pft = pft_names.index("temperate_corn")
    except ValueError as error:
        raise ValueError("parameter file lacks the prognostic-crop lower bound") from error
    vegetation_type = np.asarray(topology.patch.vegetation_type, dtype=np.int32)
    if np.any((vegetation_type < 0) | (vegetation_type >= pft_count)):
        raise ValueError("restart vegetation type lies outside native PFT parameters")
    crop_patch = soil_patch & (vegetation_type >= first_crop_pft)

    restart_values = load_native_variables(
        restart,
        [
            *_RESTART_INPUT_VARIABLES,
            *_RESTART_STATE_VARIABLES,
            *_RESTART_DIRECT_FLUX_VARIABLES,
        ],
        masked_integer_fill_value=-9999,
        masked_floating_fill_value=float(CLM_SPECIAL_VALUE),
    )
    patch_variables = set(
        (
            *_RESTART_INPUT_VARIABLES,
            *_RESTART_STATE_VARIABLES,
            *_RESTART_DIRECT_FLUX_VARIABLES,
        )
    ).difference({"sminn_vr"})
    for name in patch_variables:
        if np.asarray(restart_values[name]).shape != (patch_count,):
            raise ValueError(f"restart variable {name!r} must contain every patch")
    sminn = np.asarray(restart_values["sminn_vr"], dtype=np.float64)
    if sminn.ndim != 2 or sminn.shape[0] != column_count:
        raise ValueError("restart sminn_vr must use native (column, level) dimensions")
    sminn = sminn[:, :NUMBER_OF_SOIL_LEVELS]
    if not np.all(np.isfinite(sminn)):
        raise ValueError("restart sminn_vr contains non-finite values")
    crop_live = np.asarray(restart_values["croplive"], dtype=np.int32)
    if not np.all((crop_live[crop_patch] == 0) | (crop_live[crop_patch] == 1)):
        raise ValueError("restart croplive must be logical on prognostic crops")
    for name in patch_variables.difference({"croplive", "peaklai"}):
        values = np.asarray(restart_values[name], dtype=np.float64)
        if not np.all(np.isfinite(values[soil_patch])):
            raise ValueError(f"active soil patches contain non-finite {name}")

    _, _, layer_thickness = soil_layer_geometry_20sl_8p5m()
    clock = load_restart_clock(restart)
    use_carbon_13 = bool(general.get("use_c13", False))
    use_carbon_14 = bool(general.get("use_c14", False))
    soybean = np.asarray([name in _SOYBEAN_NAMES for name in pft_names], dtype=bool)
    parameters = PlantNutrientDemandParameters(
        woody_pft=_real_pft_parameter(parameter_values, "woody", pft_count),
        fine_root_to_leaf_carbon_pft=_real_pft_parameter(
            parameter_values, "froot_leaf", pft_count
        ),
        coarse_root_to_stem_carbon_pft=_real_pft_parameter(
            parameter_values, "croot_stem", pft_count
        ),
        stem_to_leaf_carbon_pft=_real_pft_parameter(
            parameter_values, "stem_leaf", pft_count
        ),
        live_wood_fraction_pft=_real_pft_parameter(
            parameter_values, "flivewd", pft_count
        ),
        leaf_carbon_nitrogen_ratio_pft=_real_pft_parameter(
            parameter_values, "leafcn", pft_count
        ),
        fine_root_carbon_nitrogen_ratio_pft=_real_pft_parameter(
            parameter_values, "frootcn", pft_count
        ),
        live_wood_carbon_nitrogen_ratio_pft=_real_pft_parameter(
            parameter_values, "livewdcn", pft_count
        ),
        dead_wood_carbon_nitrogen_ratio_pft=_real_pft_parameter(
            parameter_values, "deadwdcn", pft_count
        ),
        grain_carbon_nitrogen_ratio_pft=_real_pft_parameter(
            parameter_values, "graincn", pft_count
        ),
        organ_fill_leaf_carbon_nitrogen_ratio_pft=_real_pft_parameter(
            parameter_values, "fleafcn", pft_count
        ),
        organ_fill_fine_root_carbon_nitrogen_ratio_pft=_real_pft_parameter(
            parameter_values, "ffrootcn", pft_count
        ),
        organ_fill_stem_carbon_nitrogen_ratio_pft=_real_pft_parameter(
            parameter_values, "fstemcn", pft_count
        ),
        crop_allocation_shape_pft=_real_pft_parameter(
            parameter_values, "bfact", pft_count
        ),
        final_leaf_allocation_pft=_real_pft_parameter(
            parameter_values, "aleaff", pft_count
        ),
        final_root_allocation_pft=_real_pft_parameter(
            parameter_values, "arootf", pft_count
        ),
        final_stem_allocation_pft=_real_pft_parameter(
            parameter_values, "astemf", pft_count
        ),
        initial_root_allocation_pft=_real_pft_parameter(
            parameter_values, "arooti", pft_count
        ),
        initial_leaf_fraction_pft=_real_pft_parameter(
            parameter_values, "fleafi", pft_count
        ),
        leaf_allocation_exponent_pft=_real_pft_parameter(
            parameter_values, "allconsl", pft_count
        ),
        stem_allocation_exponent_pft=_real_pft_parameter(
            parameter_values, "allconss", pft_count
        ),
        growth_respiration_fraction_pft=_real_pft_parameter(
            parameter_values, "grperc", pft_count
        ),
        current_growth_respiration_fraction_pft=_real_pft_parameter(
            parameter_values, "grpnow", pft_count
        ),
        crop_decline_fraction_pft=_real_pft_parameter(
            parameter_values, "declfact", pft_count
        ),
        seasonal_deciduous_pft=_logical_pft_parameter(
            parameter_values, "season_decid", pft_count
        ),
        stress_deciduous_pft=_logical_pft_parameter(
            parameter_values, "stress_decid", pft_count
        ),
        soybean_pft=jnp.asarray(soybean),
        soil_layer_thickness=jnp.asarray(
            layer_thickness[:NUMBER_OF_SOIL_LEVELS]
        ),
        first_crop_pft=np.int32(first_crop_pft),
        time_step_seconds=np.float64(clock.step_seconds),
        xsmrpool_recovery_days=_finite_scalar(
            "dayscrecover", parameter_values["dayscrecover"]
        ),
        seconds_per_day=np.float64(SECONDS_PER_DAY),
        minimum_nitrogen=np.float64(1.0e-9),
        special_value=np.float64(CLM_SPECIAL_VALUE),
        use_fun=bool(general.get("use_fun", False)),
        use_carbon_13=use_carbon_13,
        use_carbon_14=use_carbon_14,
        use_nitrogen_scalar=bool(general.get("nscalar_opt", True)),
        plant_nitrogen_demand_option=np.int32(
            int(general.get("plant_ndemand_opt", 3))
        ),
        use_substrate_term=bool(general.get("substrate_term_opt", True)),
        use_temperature_scalar=bool(general.get("temp_scalar_opt", True)),
    )

    zero_patch = jnp.zeros(patch_count, dtype=jnp.float64)
    persistent = PlantNutrientDemandPersistentInputs(
        soil_patch=jnp.asarray(soil_patch),
        patch_column_index=jnp.asarray(patch_column),
        vegetation_type_patch=jnp.asarray(vegetation_type),
        growing_degree_days_since_planting_patch=jnp.asarray(
            restart_values["GDDPLANT_VALUE"]
        ),
        growing_degree_days_topsoil_patch=jnp.asarray(
            restart_values["GDDTSOI_VALUE"]
        ),
        crop_live_patch=jnp.asarray(crop_live == 1),
        growing_degree_days_to_maturity_patch=jnp.asarray(
            restart_values["gddmaturity"]
        ),
        heat_units_to_leaf_emergence_patch=jnp.asarray(restart_values["huileaf"]),
        heat_units_to_grain_fill_patch=jnp.asarray(restart_values["huigrain"]),
        peak_leaf_area_index_patch=jnp.asarray(
            restart_values["peaklai"], dtype=jnp.int32
        ),
        temporary_photosynthate_carbon_pool_patch=jnp.asarray(
            restart_values["xsmrpool"]
        ),
        leaf_carbon_patch=jnp.asarray(restart_values["leafc"]),
        fine_root_carbon_patch=jnp.asarray(restart_values["frootc"]),
        live_stem_carbon_patch=jnp.asarray(restart_values["livestemc"]),
        live_coarse_root_carbon_patch=jnp.asarray(restart_values["livecrootc"]),
        retranslocated_nitrogen_pool_patch=jnp.asarray(restart_values["retransn"]),
        annual_net_primary_productivity_patch=jnp.asarray(
            restart_values["annsum_npp"]
        ),
        leaf_nitrogen_patch=jnp.asarray(restart_values["leafn"]),
        fine_root_nitrogen_patch=jnp.asarray(restart_values["frootn"]),
        live_stem_nitrogen_patch=jnp.asarray(restart_values["livestemn"]),
        annual_potential_gpp_patch=jnp.asarray(
            restart_values["annsum_potential_gpp"]
        ),
        annual_maximum_retranslocated_nitrogen_patch=jnp.asarray(
            restart_values["annmax_retransn"]
        ),
        soil_mineral_nitrogen_column_level=jnp.asarray(sminn),
    )
    state = PlantNutrientDemandState(
        saved_leaf_allocation_patch=jnp.asarray(restart_values["aleafi"]),
        saved_stem_allocation_patch=jnp.asarray(restart_values["astemi"]),
        leaf_allocation_patch=jnp.asarray(restart_values["aleaf"]),
        stem_allocation_patch=jnp.asarray(restart_values["astem"]),
        grain_fill_flag_patch=jnp.asarray(restart_values["grain_flag"]),
        carbon_allometry_patch=jnp.asarray(restart_values["c_allometry"]),
        nitrogen_allometry_patch=jnp.asarray(restart_values["n_allometry"]),
        temporary_sum_potential_gpp_patch=jnp.asarray(
            restart_values["tempsum_potential_gpp"]
        ),
        temporary_maximum_retranslocated_nitrogen_patch=jnp.asarray(
            restart_values["tempmax_retransn"]
        ),
        actual_leaf_carbon_nitrogen_ratio_patch=jnp.full(
            patch_count,
            jnp.float64(CLM_SPECIAL_VALUE),
        ),
        root_allocation_patch=zero_patch,
        reproductive_allocation_patch=zero_patch,
        direct_fluxes=PlantNutrientDemandDirectFluxState(
            gpp_before_downregulation_patch=zero_patch,
            available_carbon_patch=jnp.asarray(restart_values["availc"]),
            xsmrpool_recovery_patch=jnp.asarray(
                restart_values["xsmrpool_recover"]
            ),
            plant_nitrogen_demand_patch=jnp.asarray(
                restart_values["plant_ndemand"]
            ),
            available_retranslocated_nitrogen_patch=jnp.asarray(
                restart_values["avail_retransn"]
            ),
            c13_sunlit_photosynthesis_to_cpool_patch=zero_patch,
            c13_shaded_photosynthesis_to_cpool_patch=zero_patch,
            c14_sunlit_photosynthesis_to_cpool_patch=zero_patch,
            c14_shaded_photosynthesis_to_cpool_patch=zero_patch,
        ),
        cn_fluxes=initialize_cn_transient_flux_state(
            patch_count, column_count, fieldwise=True
        ),
    )
    return PlantNutrientDemandNativeData(
        topology=topology,
        persistent_inputs=persistent,
        state=state,
        parameters=parameters,
        restart_path=restart,
        parameter_data_path=parameter_data,
    )


def _patch_fast_value(name: str, value: jnp.ndarray, patch_count: int) -> jnp.ndarray:
    result = jnp.asarray(value, dtype=jnp.float64)
    if result.shape != (patch_count,):
        raise ValueError(f"{name} must contain every native patch")
    return result


def _isotope_fast_value(
    name: str,
    value: jnp.ndarray | None,
    enabled: bool,
    patch_count: int,
) -> jnp.ndarray:
    if value is None:
        if enabled:
            raise ValueError(f"{name} is required when its isotope is enabled")
        return jnp.zeros(patch_count, dtype=jnp.float64)
    return _patch_fast_value(name, value, patch_count)


def assemble_plant_nutrient_demand_call(
    native: PlantNutrientDemandNativeData,
    cn_fluxes: CnTransientFluxState,
    *,
    sunlit_photosynthesis_patch: jnp.ndarray,
    shaded_photosynthesis_patch: jnp.ndarray,
    sunlit_leaf_area_index_patch: jnp.ndarray,
    shaded_leaf_area_index_patch: jnp.ndarray,
    decomposition_temperature_scalar_column_level: jnp.ndarray,
    transpiration_wetness_factor_patch: jnp.ndarray | None = None,
    c13_sunlit_photosynthesis_patch: jnp.ndarray | None = None,
    c13_shaded_photosynthesis_patch: jnp.ndarray | None = None,
    c14_sunlit_photosynthesis_patch: jnp.ndarray | None = None,
    c14_shaded_photosynthesis_patch: jnp.ndarray | None = None,
    persistent_inputs: PlantNutrientDemandPersistentInputs | None = None,
    state: PlantNutrientDemandState | None = None,
    crop_state: CropPhenologyState | None = None,
) -> PlantNutrientDemandCall:

    patch_count = native.topology.patch.gridcell.shape[0]
    column_count = native.topology.column.gridcell.shape[0]
    persistent = native.persistent_inputs if persistent_inputs is None else persistent_inputs
    current_state = native.state if state is None else state
    if cn_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("cn_fluxes must contain every native patch")
    if cn_fluxes.column_scalar.shape[0] != column_count:
        raise ValueError("cn_fluxes must contain every native column")
    temperature_scalar = jnp.asarray(
        decomposition_temperature_scalar_column_level,
        dtype=jnp.float64,
    )
    if temperature_scalar.ndim != 2 or temperature_scalar.shape[0] != column_count:
        raise ValueError("decomposition temperature scalar must have a native column axis")
    if temperature_scalar.shape[1] < NUMBER_OF_SOIL_LEVELS:
        raise ValueError("decomposition temperature scalar lacks active soil levels")
    temperature_scalar = temperature_scalar[:, :NUMBER_OF_SOIL_LEVELS]
    btran = (
        jnp.zeros(patch_count, dtype=jnp.float64)
        if transpiration_wetness_factor_patch is None
        else _patch_fast_value(
            "transpiration_wetness_factor_patch",
            transpiration_wetness_factor_patch,
            patch_count,
        )
    )

    if crop_state is None:
        crop_live = persistent.crop_live_patch
        growing_degree_days = persistent.growing_degree_days_since_planting_patch
        maturity = persistent.growing_degree_days_to_maturity_patch
        leaf_emergence = persistent.heat_units_to_leaf_emergence_patch
        grain_fill = persistent.heat_units_to_grain_fill_patch
    else:
        for name in CropPhenologyState._fields:
            if jnp.shape(getattr(crop_state, name)) != (patch_count,):
                raise ValueError(f"crop_state {name} must contain every native patch")
        crop_live = crop_state.crop_live_patch
        growing_degree_days = crop_state.growing_degree_days_since_planting_patch
        maturity = crop_state.growing_degree_days_to_maturity_patch
        leaf_emergence = crop_state.heat_units_to_leaf_emergence_patch
        grain_fill = crop_state.heat_units_to_grain_fill_patch

    inputs = PlantNutrientDemandInputs(
        soil_patch=persistent.soil_patch,
        patch_column_index=persistent.patch_column_index,
        vegetation_type_patch=persistent.vegetation_type_patch,
        sunlit_photosynthesis_patch=_patch_fast_value(
            "sunlit_photosynthesis_patch", sunlit_photosynthesis_patch, patch_count
        ),
        shaded_photosynthesis_patch=_patch_fast_value(
            "shaded_photosynthesis_patch", shaded_photosynthesis_patch, patch_count
        ),
        c13_sunlit_photosynthesis_patch=_isotope_fast_value(
            "c13_sunlit_photosynthesis_patch",
            c13_sunlit_photosynthesis_patch,
            bool(native.parameters.use_carbon_13),
            patch_count,
        ),
        c13_shaded_photosynthesis_patch=_isotope_fast_value(
            "c13_shaded_photosynthesis_patch",
            c13_shaded_photosynthesis_patch,
            bool(native.parameters.use_carbon_13),
            patch_count,
        ),
        c14_sunlit_photosynthesis_patch=_isotope_fast_value(
            "c14_sunlit_photosynthesis_patch",
            c14_sunlit_photosynthesis_patch,
            bool(native.parameters.use_carbon_14),
            patch_count,
        ),
        c14_shaded_photosynthesis_patch=_isotope_fast_value(
            "c14_shaded_photosynthesis_patch",
            c14_shaded_photosynthesis_patch,
            bool(native.parameters.use_carbon_14),
            patch_count,
        ),
        sunlit_leaf_area_index_patch=_patch_fast_value(
            "sunlit_leaf_area_index_patch", sunlit_leaf_area_index_patch, patch_count
        ),
        shaded_leaf_area_index_patch=_patch_fast_value(
            "shaded_leaf_area_index_patch", shaded_leaf_area_index_patch, patch_count
        ),
        growing_degree_days_since_planting_patch=growing_degree_days,
        growing_degree_days_topsoil_patch=persistent.growing_degree_days_topsoil_patch,
        crop_live_patch=crop_live,
        growing_degree_days_to_maturity_patch=maturity,
        heat_units_to_leaf_emergence_patch=leaf_emergence,
        heat_units_to_grain_fill_patch=grain_fill,
        peak_leaf_area_index_patch=persistent.peak_leaf_area_index_patch,
        temporary_photosynthate_carbon_pool_patch=(
            persistent.temporary_photosynthate_carbon_pool_patch
        ),
        leaf_carbon_patch=persistent.leaf_carbon_patch,
        fine_root_carbon_patch=persistent.fine_root_carbon_patch,
        live_stem_carbon_patch=persistent.live_stem_carbon_patch,
        live_coarse_root_carbon_patch=persistent.live_coarse_root_carbon_patch,
        retranslocated_nitrogen_pool_patch=(
            persistent.retranslocated_nitrogen_pool_patch
        ),
        annual_net_primary_productivity_patch=(
            persistent.annual_net_primary_productivity_patch
        ),
        leaf_nitrogen_patch=persistent.leaf_nitrogen_patch,
        fine_root_nitrogen_patch=persistent.fine_root_nitrogen_patch,
        live_stem_nitrogen_patch=persistent.live_stem_nitrogen_patch,
        annual_potential_gpp_patch=persistent.annual_potential_gpp_patch,
        annual_maximum_retranslocated_nitrogen_patch=(
            persistent.annual_maximum_retranslocated_nitrogen_patch
        ),
        soil_mineral_nitrogen_column_level=(
            persistent.soil_mineral_nitrogen_column_level
        ),
        decomposition_temperature_scalar_column_level=temperature_scalar,
        transpiration_wetness_factor_patch=btran,
    )
    return PlantNutrientDemandCall(
        inputs=inputs,
        state=current_state._replace(cn_fluxes=cn_fluxes),
        parameters=native.parameters,
    )


__all__ = [
    "PlantNutrientDemandCall",
    "PlantNutrientDemandNativeData",
    "PlantNutrientDemandPersistentInputs",
    "assemble_plant_nutrient_demand_call",
    "load_plant_nutrient_demand_native_data",
]
