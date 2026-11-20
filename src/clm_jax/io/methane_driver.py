
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.canopy_hydraulic_inputs import bedrock_level_count_column
from clm_jax.io.canopy_temperature_restart import (
    column_soil_hydraulic_properties_20sl_8p5m,
    soil_layer_geometry_20sl_8p5m,
)
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.methane_balance_initialization import (
    MethaneBalanceInitializationNativeData,
    load_methane_balance_initialization_native_data,
)
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.native_templates import native_template
from clm_jax.io.netcdf import (
    load_dimension_sizes,
    load_native_variables,
)
from clm_jax.io.plant_hydraulics import load_plant_hydraulic_parameters
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_LAKE,
    LANDUNIT_SOIL,
    SubgridTopology,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)
from clm_jax.physics.methane_biogeochemistry import (
    MethaneAerenchymaParameters,
    MethaneAerenchymaState,
    MethaneAnnualUpdateParameters,
    MethaneAnnualUpdateState,
    MethaneEbullitionParameters,
    MethaneEbullitionState,
    MethaneGasCompetitionState,
    MethaneOxidationParameters,
    MethaneOxidationState,
    MethaneProductionParameters,
    MethaneProductionState,
    MethaneWaterTableParameters,
    MethaneWaterTableState,
)
from clm_jax.physics.methane_driver import (
    INUNDATION_METHOD_TOTAL_WATER_STORAGE,
    MethaneDriverFinalizeParameters,
    MethaneDriverFinalizeState,
    MethaneDriverPreprocessParameters,
    MethaneDriverPreprocessState,
)
from clm_jax.physics.methane_driver_facade import (
    MethaneDriverActiveParameters,
    MethaneDriverActiveState,
)
from clm_jax.physics.methane_soil_gas_transport import (
    DEFAULT_METHANE_SOIL_GAS_TRANSPORT_PARAMETERS,
    MethaneSoilGasTransportState,
)
from clm_jax.physics.plant_hydraulics import jackson_root_fraction

_RESTART_COLUMN_VARIABLES = (
    "FINUNDATED",
    "FINUNDATED_LAG",
    "FINUNDATED_PRESNOW",
    "QFLX_SURF_LAG",
    "annavg_somhr",
    "annavg_finrw",
    "annsum_counter_ch4",
    "tempavg_somhr",
    "tempavg_finrw",
)
_RESTART_COLUMN_LEVEL_VARIABLES = (
    "LAYER_SAT_LAG",
    "CONC_CH4_SAT",
    "CONC_CH4_UNSAT",
    "CONC_O2_SAT",
    "CONC_O2_UNSAT",
    "O2STRESS_SAT",
    "O2STRESS_UNSAT",
    "O2_DECOMP_DEPTH_SAT",
    "O2_DECOMP_DEPTH_UNSAT",
    "SMP",
)
_RESTART_PATCH_VARIABLES = (
    "tempavg_agnpp",
    "tempavg_bgnpp",
    "annavg_agnpp",
    "annavg_bgnpp",
)
_RESTART_GRIDCELL_VARIABLES = ("TWS",)
_RESTART_VARIABLES = (
    *_RESTART_COLUMN_VARIABLES,
    *_RESTART_COLUMN_LEVEL_VARIABLES,
    *_RESTART_PATCH_VARIABLES,
    *_RESTART_GRIDCELL_VARIABLES,
)
_SURFACE_VARIABLES = ("PCT_SAND", "PCT_CLAY", "ORGANIC", "zbedrock")
_PARAMETER_VARIABLES = (
    "crop",
    "organic_max",
    "q10ch4",
    "q10ch4base",
    "f_ch4",
    "cnscalefactor",
    "redoxlag",
    "redoxlag_vertical",
    "q10_ch4oxid",
    "smp_crit",
    "scale_factor_aere",
    "nongrassporosratio",
    "porosmin",
    "vgc_max",
    "satpow",
    "scale_factor_gasdiff",
    "scale_factor_liqdiff",
    "f_sat",
    "qflxlagd",
    "highlatfact",
    "atmch4",
    "rob",
    "capthick",
)
_INUNDATION_VARIABLES = ("FWS_TWS_A", "FWS_TWS_B")


class MethaneDriverNativeInputs(NamedTuple):

    methane_column: jnp.ndarray
    nonlake_column: jnp.ndarray
    lake_column: jnp.ndarray
    methane_patch: jnp.ndarray
    vegetated_patch: jnp.ndarray
    grass_or_crop_patch: jnp.ndarray
    column_gridcell: jnp.ndarray
    patch_column: jnp.ndarray
    column_active: jnp.ndarray
    column_weight_on_gridcell: jnp.ndarray
    patch_weight_on_column: jnp.ndarray
    latitude_degrees_gridcell: jnp.ndarray
    atmospheric_oxygen_mole_fraction: float
    atmospheric_carbon_dioxide_mole_fraction: float
    atmospheric_methane_mole_fraction: float
    soil_layer_depth_column_level: jnp.ndarray
    soil_interface_depth_column_interface: jnp.ndarray
    soil_layer_thickness_column_level: jnp.ndarray
    soil_porosity_column_level: jnp.ndarray
    soil_b_parameter_column_level: jnp.ndarray
    soil_organic_matter_column_level: jnp.ndarray
    root_fraction_patch_level: jnp.ndarray
    effective_root_fraction_patch_level: jnp.ndarray
    total_water_storage_gridcell: jnp.ndarray
    total_water_storage_slope_gridcell: jnp.ndarray
    total_water_storage_intercept_gridcell: jnp.ndarray
    soil_matrix_potential_column_level: jnp.ndarray


class MethaneDriverNativeData(NamedTuple):
    topology: SubgridTopology
    inputs: MethaneDriverNativeInputs
    state: MethaneDriverActiveState
    parameters: MethaneDriverActiveParameters
    balance_initialization: MethaneBalanceInitializationNativeData
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path
    inundation_data_path: Path


def _configured_path(
    case: CimeCase,
    group: dict[str, object],
    key: str,
    override: str | Path | None,
) -> Path:
    if override is None:
        declared = group.get(key)
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError(f"configured lnd_in group lacks {key}")
        path = Path(case.expand(declared)).expanduser()
        if not path.is_absolute():
            path = case.root / path
        path = path.resolve()
    else:
        path = Path(override).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM inputdata does not exist: {path}")
    return path


def _scalar(values: dict[str, jnp.ndarray], name: str) -> float:
    value = float(np.asarray(values[name], dtype=np.float64).item())
    if not np.isfinite(value):
        raise ValueError(f"CLM parameter {name} must be finite")
    return value


def _pft_names(parameter_data_path: Path) -> tuple[str, ...]:
    from netCDF4 import Dataset  

    with Dataset(parameter_data_path, mode="r") as dataset:
        characters = np.asarray(dataset.variables["pftname"][:])
    if characters.ndim != 2 or characters.dtype.kind not in {"S", "U"}:
        raise ValueError("CLM pftname must be a character matrix")
    if characters.dtype.kind == "S":
        return tuple(
            b"".join(np.asarray(row, dtype="S1").tolist()).decode("ascii").rstrip()
            for row in characters
        )
    return tuple("".join(np.asarray(row, dtype=str).tolist()).rstrip() for row in characters)


def _validated_restart_arrays(
    restart_path: Path,
    topology: SubgridTopology,
) -> dict[str, jnp.ndarray]:
    values = load_native_variables(
        restart_path,
        list(_RESTART_VARIABLES),
        dtype=np.float64,
    )
    expected = {
        **{name: (topology.column.gridcell.shape[0],) for name in _RESTART_COLUMN_VARIABLES},
        **{
            name: (topology.column.gridcell.shape[0], 25)
            for name in _RESTART_COLUMN_LEVEL_VARIABLES
        },
        **{name: (topology.patch.gridcell.shape[0],) for name in _RESTART_PATCH_VARIABLES},
        **{
            name: (topology.grid.longitude_degrees.shape[0],)
            for name in _RESTART_GRIDCELL_VARIABLES
        },
    }
    for name, shape in expected.items():
        if values[name].shape != shape:
            raise ValueError(f"restart {name} must have shape {shape}")
    methane_column = np.asarray(topology.column.active, dtype=bool) & np.isin(
        np.asarray(topology.column.landunit_type, dtype=np.int32),
        (LANDUNIT_SOIL, LANDUNIT_CROP),
    )
    methane_patch = np.asarray(topology.patch.active, dtype=bool) & np.isin(
        np.asarray(topology.patch.landunit_type, dtype=np.int32),
        (LANDUNIT_SOIL, LANDUNIT_CROP),
    )
    for name in _RESTART_COLUMN_VARIABLES:
        if not np.all(np.isfinite(np.asarray(values[name])[methane_column])):
            raise ValueError(f"restart {name} is non-finite on the methane column filter")
    for name in _RESTART_COLUMN_LEVEL_VARIABLES:
        if not np.all(np.isfinite(np.asarray(values[name])[methane_column, :20])):
            raise ValueError(f"restart {name} is non-finite on active methane levels")
    for name in _RESTART_PATCH_VARIABLES:
        if not np.all(np.isfinite(np.asarray(values[name])[methane_patch])):
            raise ValueError(f"restart {name} is non-finite on the methane patch filter")
    return values


def _empty_level_state(column_count: int) -> jnp.ndarray:
    return jnp.zeros((column_count, 25), dtype=jnp.float64)


def _active_state(
    topology: SubgridTopology,
    values: dict[str, jnp.ndarray],
) -> MethaneDriverActiveState:
    column_count = topology.column.gridcell.shape[0]
    gridcell_count = topology.grid.longitude_degrees.shape[0]
    zero_column = jnp.zeros((column_count,), dtype=jnp.float64)
    zero_gridcell = jnp.zeros((gridcell_count,), dtype=jnp.float64)
    zero_level = _empty_level_state(column_count)
    first_time = jnp.zeros((column_count,), dtype=jnp.bool_)

    production = MethaneProductionState(
        unsaturated_methane_production_column_level=zero_level,
        saturated_methane_production_column_level=zero_level,
        unsaturated_oxygen_decomposition_demand_column_level=values[
            "O2_DECOMP_DEPTH_UNSAT"
        ],
        saturated_oxygen_decomposition_demand_column_level=values[
            "O2_DECOMP_DEPTH_SAT"
        ],
        seasonal_inundation_factor_column=zero_column,
    )
    competition = MethaneGasCompetitionState(
        unsaturated_methane_oxidation_column_level=zero_level,
        saturated_methane_oxidation_column_level=zero_level,
        unsaturated_oxygen_oxidation_column_level=zero_level,
        saturated_oxygen_oxidation_column_level=zero_level,
        unsaturated_methane_aerenchyma_column_level=zero_level,
        saturated_methane_aerenchyma_column_level=zero_level,
        unsaturated_methane_ebullition_column_level=zero_level,
        saturated_methane_ebullition_column_level=zero_level,
        unsaturated_oxygen_decomposition_column_level=values[
            "O2_DECOMP_DEPTH_UNSAT"
        ],
        saturated_oxygen_decomposition_column_level=values[
            "O2_DECOMP_DEPTH_SAT"
        ],
        unsaturated_oxygen_stress_column_level=values["O2STRESS_UNSAT"],
        saturated_oxygen_stress_column_level=values["O2STRESS_SAT"],
        unsaturated_methane_stress_column_level=zero_level,
        saturated_methane_stress_column_level=zero_level,
    )
    return MethaneDriverActiveState(
        preprocess=MethaneDriverPreprocessState(
            lagged_surface_runoff_column=values["QFLX_SURF_LAG"],
            inundated_fraction_column=values["FINUNDATED"],
            presnow_inundated_fraction_column=values["FINUNDATED_PRESNOW"],
            lagged_inundated_fraction_column=values["FINUNDATED_LAG"],
            saturated_methane_concentration_column_level=values["CONC_CH4_SAT"],
            inundation_change_surface_flux_column=zero_column,
            methane_first_time_column=first_time,
        ),
        annual=MethaneAnnualUpdateState(
            annual_accumulator_seconds_column=values["annsum_counter_ch4"],
            temporary_soil_respiration_mean_column=values["tempavg_somhr"],
            annual_soil_respiration_mean_column=values["annavg_somhr"],
            temporary_respiration_weighted_inundation_column=values["tempavg_finrw"],
            annual_respiration_weighted_inundation_column=values["annavg_finrw"],
            temporary_aboveground_npp_mean_patch=values["tempavg_agnpp"],
            annual_aboveground_npp_mean_patch=values["annavg_agnpp"],
            temporary_belowground_npp_mean_patch=values["tempavg_bgnpp"],
            annual_belowground_npp_mean_patch=values["annavg_bgnpp"],
        ),
        water_table=MethaneWaterTableState(
            layer_above_water_table_column=jnp.zeros((column_count,), dtype=jnp.int32)
        ),
        production=production,
        oxidation=MethaneOxidationState(
            unsaturated_methane_oxidation_column_level=zero_level,
            saturated_methane_oxidation_column_level=zero_level,
            unsaturated_oxygen_oxidation_demand_column_level=zero_level,
            saturated_oxygen_oxidation_demand_column_level=zero_level,
        ),
        aerenchyma=MethaneAerenchymaState(
            unsaturated_methane_loss_column_level=zero_level,
            saturated_methane_loss_column_level=zero_level,
            unsaturated_transpiration_loss_column_level=zero_level,
            saturated_transpiration_loss_column_level=zero_level,
            unsaturated_oxygen_gain_column_level=zero_level,
            saturated_oxygen_gain_column_level=zero_level,
        ),
        ebullition=MethaneEbullitionState(
            unsaturated_ebullition_column_level=zero_level,
            saturated_ebullition_column_level=zero_level,
            unsaturated_ebullition_total_column=zero_column,
            saturated_ebullition_total_column=zero_column,
        ),
        transport=MethaneSoilGasTransportState(
            competition=competition,
            unsaturated_methane_concentration_column_level=values["CONC_CH4_UNSAT"],
            saturated_methane_concentration_column_level=values["CONC_CH4_SAT"],
            unsaturated_oxygen_concentration_column_level=values["CONC_O2_UNSAT"],
            saturated_oxygen_concentration_column_level=values["CONC_O2_SAT"],
            unsaturated_surface_aerenchyma_column=zero_column,
            saturated_surface_aerenchyma_column=zero_column,
            unsaturated_ebullition_total_column=zero_column,
            saturated_ebullition_total_column=zero_column,
            unsaturated_surface_ebullition_column=zero_column,
            saturated_surface_ebullition_column=zero_column,
            unsaturated_surface_diffusion_column=zero_column,
            saturated_surface_diffusion_column=zero_column,
            ground_methane_conductance_column=zero_column,
        ),
        finalize=MethaneDriverFinalizeState(
            total_methane_carbon_column=zero_column,
            surface_methane_flux_column=zero_column,
            methane_oxidation_gridcell=zero_gridcell,
            methane_production_gridcell=zero_gridcell,
            net_methane_carbon_correction_gridcell=zero_gridcell,
            methane_first_time_column=first_time,
        ),
        lagged_layer_saturation_column_level=values["LAYER_SAT_LAG"],
        unsaturated_water_table_depth_column=zero_column,
    )


def _parameters(
    topology: SubgridTopology,
    clock_step_seconds: int,
    values: dict[str, jnp.ndarray],
) -> MethaneDriverActiveParameters:
    dt = float(clock_step_seconds)
    maximum_patches = int(
        np.bincount(np.asarray(topology.patch.column, dtype=np.int64)).max(initial=0)
    )
    maximum_columns = int(
        np.bincount(np.asarray(topology.column.gridcell, dtype=np.int64)).max(initial=0)
    )
    transport = DEFAULT_METHANE_SOIL_GAS_TRANSPORT_PARAMETERS._replace(
        land_time_step_seconds=dt,
        methane_time_step_seconds=dt,
        saturated_diffusion_exponent=_scalar(values, "satpow"),
        gas_diffusion_scale=_scalar(values, "scale_factor_gasdiff"),
        liquid_diffusion_scale=_scalar(values, "scale_factor_liqdiff"),
        surface_ice_cap_mm=_scalar(values, "capthick"),
        organic_matter_maximum_kg_m3=_scalar(values, "organic_max"),
    )
    return MethaneDriverActiveParameters(
        preprocess=MethaneDriverPreprocessParameters(
            time_step_seconds=dt,
            runoff_lag_days=_scalar(values, "qflxlagd"),
            high_latitude_lag_factor=_scalar(values, "highlatfact"),
            redox_lag_days=_scalar(values, "redoxlag"),
            atmospheric_methane_mole_fraction=_scalar(values, "atmch4"),
            methane_offline=True,
            inundation_method=INUNDATION_METHOD_TOTAL_WATER_STORAGE,
        ),
        annual=MethaneAnnualUpdateParameters(time_step_seconds=dt, days_per_year=365),
        water_table=MethaneWaterTableParameters(
            saturation_fraction=_scalar(values, "f_sat")
        ),
        production=MethaneProductionParameters(
            methane_temperature_q10=_scalar(values, "q10ch4"),
            methane_temperature_reference_kelvin=_scalar(values, "q10ch4base"),
            methane_fraction_of_mineralization=_scalar(values, "f_ch4"),
            carbon_nitrogen_decomposition_scale=_scalar(values, "cnscalefactor"),
        ),
        oxidation=MethaneOxidationParameters(
            oxidation_temperature_q10=_scalar(values, "q10_ch4oxid"),
            critical_soil_matrix_potential_mm=_scalar(values, "smp_crit"),
        ),
        aerenchyma=MethaneAerenchymaParameters(
            time_step_seconds=dt,
            aerenchyma_area_scale=_scalar(values, "scale_factor_aere"),
            nongrass_porosity_ratio=_scalar(values, "nongrassporosratio"),
            minimum_aerenchyma_porosity=_scalar(values, "porosmin"),
            root_obliquity=_scalar(values, "rob"),
        ),
        ebullition=MethaneEbullitionParameters(
            time_step_seconds=dt,
            maximum_bubble_saturation_ratio=_scalar(values, "vgc_max"),
        ),
        transport=transport,
        finalize=MethaneDriverFinalizeParameters(
            time_step_seconds=dt,
            allow_lake_production=False,
            maximum_columns_per_gridcell=maximum_columns,
        ),
        vertical_redox_lag_days=_scalar(values, "redoxlag_vertical"),
        maximum_patches_per_column=maximum_patches,
    )


@native_template
def load_methane_driver_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    inundation_data_path: str | Path | None = None,
) -> MethaneDriverNativeData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    required = {
        "use_lch4": True,
        "use_cn": True,
        "use_crop": True,
        "use_vertsoilc": True,
        "use_nitrif_denitrif": True,
        "use_fates": False,
        "use_hydrstress": True,
        "soil_layerstruct": "20SL_8.5m",
    }
    for name, expected in required.items():
        if general.get(name) != expected:
            raise ValueError(f"native methane adapter requires {name}={expected!r}")
    if general.get("co2_type") != "constant":
        raise ValueError("native methane adapter requires co2_type='constant'")
    carbon_dioxide_ppmv = float(general.get("co2_ppmv", np.nan))
    if not np.isfinite(carbon_dioxide_ppmv) or carbon_dioxide_ppmv <= 0.0:
        raise ValueError("native methane adapter requires a positive finite co2_ppmv")
    methane_group = read_case_namelist_group(case, "lnd_in", "ch4par_in")
    if methane_group.get("finundation_method") != "TWS_inversion":
        raise ValueError("native methane adapter requires TWS_inversion inundation")
    if methane_group.get("use_aereoxid_prog") is not True:
        raise ValueError("native methane adapter requires use_aereoxid_prog=.true.")
    if methane_group.get("allowlakeprod", False) is not False:
        raise ValueError("active methane port requires allowlakeprod=.false.")

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    dimensions = load_dimension_sizes(restart)
    expected_dimensions = {"levgrnd": 25}
    for name, expected in expected_dimensions.items():
        if dimensions.get(name) != expected:
            raise ValueError(f"native methane adapter requires {name}={expected}")
    topology = load_subgrid_topology(restart)
    surface = _configured_path(case, general, "fsurdat", surface_data_path)
    parameter = _configured_path(case, general, "paramfile", parameter_data_path)
    inundation_group = read_case_namelist_group(case, "lnd_in", "ch4finundated")
    inundation = _configured_path(
        case,
        inundation_group,
        "stream_fldfilename_ch4finundated",
        inundation_data_path,
    )

    restart_values = _validated_restart_arrays(restart, topology)
    surface_values = load_native_variables(surface, list(_SURFACE_VARIABLES), dtype=np.float64)
    parameter_values = load_native_variables(
        parameter,
        list(_PARAMETER_VARIABLES),
        dtype=np.float64,
    )
    inundation_values = load_native_variables(
        inundation,
        list(_INUNDATION_VARIABLES),
        indexers={"time": 0},
        dtype=np.float64,
    )
    organic_maximum = _scalar(parameter_values, "organic_max")
    soil = column_soil_hydraulic_properties_20sl_8p5m(
        topology,
        surface_values,
        organic_maximum,
    )
    node, interface, thickness = soil_layer_geometry_20sl_8p5m()
    column_count = topology.column.gridcell.shape[0]
    patch_count = topology.patch.gridcell.shape[0]
    node_column = np.broadcast_to(node, (column_count, 25))
    interface_column = np.broadcast_to(interface, (column_count, 26))
    thickness_column = np.broadcast_to(thickness, (column_count, 25))

    column_active = np.asarray(topology.column.active, dtype=bool)
    column_landunit_type = np.asarray(topology.column.landunit_type, dtype=np.int32)
    methane_column = column_active & np.isin(
        column_landunit_type,
        (LANDUNIT_SOIL, LANDUNIT_CROP),
    )
    lake_column = column_active & (column_landunit_type == LANDUNIT_LAKE)
    nonlake_column = column_active & ~lake_column
    patch_active = np.asarray(topology.patch.active, dtype=bool)
    patch_landunit_type = np.asarray(topology.patch.landunit_type, dtype=np.int32)
    methane_patch = patch_active & np.isin(
        patch_landunit_type,
        (LANDUNIT_SOIL, LANDUNIT_CROP),
    )
    vegetation_type = np.asarray(topology.patch.vegetation_type, dtype=np.int32)
    vegetated_patch = vegetation_type != 0

    crop_pft = np.asarray(parameter_values["crop"], dtype=np.float64) != 0.0
    pft_names = _pft_names(parameter)
    if crop_pft.shape != (len(pft_names),):
        raise ValueError("CLM crop and pftname dimensions disagree")
    grass_pft = np.asarray(
        [name in {"c3_arctic_grass", "c3_non-arctic_grass", "c4_grass"} for name in pft_names],
        dtype=bool,
    )
    grass_or_crop_patch = (grass_pft | crop_pft)[vegetation_type]

    plant_parameters = load_plant_hydraulic_parameters(case, parameter)
    root_beta = np.asarray(
        plant_parameters.root_profile_beta_pft_variant,
        dtype=np.float64,
    )[vegetation_type, 0]
    bedrock_column = bedrock_level_count_column(topology, surface)
    hydrologic_root_fraction = jackson_root_fraction(
        jnp.asarray(root_beta),
        jnp.asarray(interface[:21]),
        jnp.asarray(bedrock_column[np.asarray(topology.patch.column, dtype=np.int64)]),
    )
    root_fraction = jnp.pad(hydrologic_root_fraction, ((0, 0), (0, 5)))
    if root_fraction.shape != (patch_count, 25):
        raise ValueError("reconstructed root fractions must have shape (patch, 25)")

    longitude = np.asarray(topology.grid.longitude_index, dtype=np.int64)
    latitude = np.asarray(topology.grid.latitude_index, dtype=np.int64)
    tws_slope_source = np.asarray(inundation_values["FWS_TWS_A"], dtype=np.float64)
    tws_intercept_source = np.asarray(inundation_values["FWS_TWS_B"], dtype=np.float64)
    if tws_slope_source.ndim != 2 or tws_intercept_source.shape != tws_slope_source.shape:
        raise ValueError("TWS inundation coefficients must have dimensions (lat, lon)")
    tws_slope = tws_slope_source[latitude, longitude]
    tws_intercept = tws_intercept_source[latitude, longitude]
    if not np.all(np.isfinite(tws_slope)) or not np.all(np.isfinite(tws_intercept)):
        raise ValueError("TWS inundation coefficients are non-finite on the land grid")

    clock = load_restart_clock(restart)
    state = _active_state(topology, restart_values)
    parameters = _parameters(topology, clock.step_seconds, parameter_values)
    balance = load_methane_balance_initialization_native_data(case, restart)
    return MethaneDriverNativeData(
        topology=topology,
        inputs=MethaneDriverNativeInputs(
            methane_column=jnp.asarray(methane_column),
            nonlake_column=jnp.asarray(nonlake_column),
            lake_column=jnp.asarray(lake_column),
            methane_patch=jnp.asarray(methane_patch),
            vegetated_patch=jnp.asarray(vegetated_patch),
            grass_or_crop_patch=jnp.asarray(grass_or_crop_patch),
            column_gridcell=jnp.asarray(topology.column.gridcell, dtype=jnp.int32),
            patch_column=jnp.asarray(topology.patch.column, dtype=jnp.int32),
            column_active=jnp.asarray(column_active),
            column_weight_on_gridcell=jnp.asarray(topology.column.weight_on_gridcell),
            patch_weight_on_column=jnp.asarray(topology.patch.weight_on_column),
            latitude_degrees_gridcell=jnp.asarray(topology.grid.latitude_degrees),
            atmospheric_oxygen_mole_fraction=0.209,
            atmospheric_carbon_dioxide_mole_fraction=(
                carbon_dioxide_ppmv * 1.0e-6
            ),
            atmospheric_methane_mole_fraction=_scalar(
                parameter_values,
                "atmch4",
            ),
            soil_layer_depth_column_level=jnp.asarray(node_column),
            soil_interface_depth_column_interface=jnp.asarray(interface_column),
            soil_layer_thickness_column_level=jnp.asarray(thickness_column),
            soil_porosity_column_level=jnp.asarray(
                soil.saturated_water_content_column_level
            ),
            soil_b_parameter_column_level=jnp.asarray(
                soil.clapp_hornberger_b_column_level
            ),
            soil_organic_matter_column_level=jnp.asarray(
                soil.organic_matter_density_column_level
            ),
            root_fraction_patch_level=jnp.asarray(root_fraction),
            effective_root_fraction_patch_level=jnp.zeros(
                (patch_count, 25), dtype=jnp.float64
            ),
            total_water_storage_gridcell=restart_values["TWS"],
            total_water_storage_slope_gridcell=jnp.asarray(tws_slope),
            total_water_storage_intercept_gridcell=jnp.asarray(tws_intercept),
            soil_matrix_potential_column_level=restart_values["SMP"],
        ),
        state=state,
        parameters=parameters,
        balance_initialization=balance,
        restart_path=restart,
        surface_data_path=surface,
        parameter_data_path=parameter,
        inundation_data_path=inundation,
    )


__all__ = [
    "MethaneDriverNativeData",
    "MethaneDriverNativeInputs",
    "load_methane_driver_native_data",
]
