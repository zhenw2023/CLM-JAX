
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import (
    CLM_SPECIAL_VALUE,
    DENSITY_FRESH_WATER,
    DENSITY_ICE,
    FREEZING_TEMPERATURE_FRESHWATER,
    SECONDS_PER_DAY,
)
from clm_jax.io.canopy_temperature_restart import (
    soil_layer_geometry_20sl_8p5m,
)
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.plant_hydraulics import (
    initialize_jackson_root_fraction,
    load_plant_hydraulic_parameters,
    load_plant_hydraulic_restart,
)
from clm_jax.io.restart import (
    LANDUNIT_LAKE,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    RestartClock,
    SubgridTopology,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)
from clm_jax.physics.canopy_fluxes import (
    CanopyEnergyInitialState,
    CanopyEnergyInputs,
    CanopyHydraulicCouplingInputs,
)
from clm_jax.physics.canopy_radiation import CanopySunShadeState
from clm_jax.physics.canopy_temperature import (
    NUMBER_OF_SNOW_LEVELS,
    NUMBER_OF_SOIL_LEVELS,
    NUMBER_OF_TOTAL_LEVELS,
    CanopyTemperatureParameters,
)
from clm_jax.physics.daylength import daylength, maximum_daylength
from clm_jax.physics.solar_runtime import native_grid_radians
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
)
from clm_jax.physics.photosynthesis import (
    HydraulicPhotosynthesisBiochemicalParameters,
    HydraulicPhotosynthesisPatchControls,
)
from clm_jax.physics.plant_hydraulics import (
    ROOT,
    PlantHydraulicParameters,
    PlantHydraulicPatchInputs,
    PlantHydraulicPatchState,
    SoilRootConductanceInputs,
    SoilRootConductanceParameters,
    jackson_root_fraction,
    select_patch_parameters,
    soil_root_interface_conductance,
)
from clm_jax.physics.qsat import qsat
from clm_jax.physics.soil_moisture_stress import root_zone_soil_wetness
from clm_jax.physics.solar import FIXED_YEAR_2000_ORBIT, solar_declination

_RESTART_SCALAR_FLOAT_VARIABLES = (
    "frootc",
    "leafn",
    "tlai",
    "tsai",
    "vcmaxcintsun",
    "vcmaxcintsha",
    "tair10",
    "FRAC_VEG_NOSNO_ALB",
)
_RESTART_LAYER_VARIABLES = (
    "vcmx25_z",
    "jmx25_z",
    "tlai_z",
)
_RESTART_COLUMN_LAYER_VARIABLES = ("HK", "SMP")
_PARAMETER_VARIABLES = (
    "c3psn",
    "crop",
    "i_vcad",
    "s_vcad",
    "lmr_intercept_atkin",
)


class CanopyBiochemicalPftParameters(NamedTuple):

    c3_pathway_pft: jnp.ndarray
    crop_pft: jnp.ndarray
    vcmax_intercept_pft: jnp.ndarray
    vcmax_leaf_nitrogen_slope_pft: jnp.ndarray
    leaf_respiration_intercept_pft: jnp.ndarray


class CanopyHydraulicNativeInputs(NamedTuple):

    fine_root_carbon_patch: jnp.ndarray
    leaf_nitrogen_patch: jnp.ndarray
    total_leaf_area_patch: jnp.ndarray
    total_stem_area_patch: jnp.ndarray
    hydraulic_conductivity_column_level: jnp.ndarray
    soil_water_potential_column_level: jnp.ndarray
    bedrock_level_count_column: jnp.ndarray
    vcmax_scaling_sunlit_patch: jnp.ndarray
    vcmax_scaling_shaded_patch: jnp.ndarray
    luna_vcmax25_by_layer_patch: jnp.ndarray
    luna_jmax25_by_layer_patch: jnp.ndarray
    total_leaf_area_by_layer_patch: jnp.ndarray
    radiative_layer_count_patch: jnp.ndarray
    air_temperature_ten_day_patch: jnp.ndarray
    atmospheric_co2_ppmv: jnp.ndarray
    calendar_day: jnp.ndarray
    root_fraction_patch_level: jnp.ndarray | None = None


class CanopyHydraulicNativeData(NamedTuple):

    topology: SubgridTopology
    clock: RestartClock
    inputs: CanopyHydraulicNativeInputs
    plant_parameters: PlantHydraulicParameters
    biochemical_parameters: CanopyBiochemicalPftParameters
    initial_plant_state: PlantHydraulicPatchState
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path


def _configured_input_path(case: CimeCase, key: str) -> Path:
    group = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    declared = group.get(key)
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(f"clm_inparm in {case.case_docs / 'lnd_in'} lacks {key}")
    path = Path(case.expand(declared)).expanduser()
    if not path.is_absolute():
        path = case.root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM {key} does not exist: {path}")
    return path


def _selected_input_path(
    case: CimeCase,
    key: str,
    override: str | Path | None,
) -> Path:
    path = (
        _configured_input_path(case, key)
        if override is None
        else Path(override).expanduser().resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(f"CLM {key} inputdata does not exist: {path}")
    return path


def _require_active_case_configuration(case: CimeCase) -> float:
    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    required_general = {
        "use_cn": True,
        "use_crop": True,
        "use_dynroot": False,
        "use_fates": False,
        "use_hydrstress": True,
        "use_luna": True,
    }
    for key, expected in required_general.items():
        if general.get(key) is not expected:
            raise ValueError(f"native canopy coupling requires {key}={expected!r}")
    if str(general.get("co2_type", "")).lower() != "constant":
        raise ValueError("native canopy coupling currently requires co2_type='constant'")
    co2_ppmv = general.get("co2_ppmv")
    if (
        isinstance(co2_ppmv, bool)
        or not isinstance(co2_ppmv, int | float)
        or not np.isfinite(co2_ppmv)
        or not 10.0 <= float(co2_ppmv) <= 15_000.0
    ):
        raise ValueError("active case co2_ppmv must be a finite physical scalar")

    nitrogen = read_case_namelist_group(case, "lnd_in", "clm_nitrogen")
    if nitrogen.get("lnc_opt") is not True:
        raise ValueError("native canopy coupling requires lnc_opt=.true.")
    if int(nitrogen.get("vcmax_opt", -1)) != 3:
        raise ValueError("native canopy coupling requires vcmax_opt=3")
    if nitrogen.get("reduce_dayl_factor") is not False:
        raise ValueError("native canopy coupling requires reduce_dayl_factor=.false.")

    photosynthesis = read_case_namelist_group(case, "lnd_in", "photosyns_inparm")
    if int(photosynthesis.get("leafresp_method", -1)) != 2:
        raise ValueError("native canopy coupling requires Atkin2015 leaf respiration")
    if photosynthesis.get("light_inhibit") is not True:
        raise ValueError("native canopy coupling requires light_inhibit=.true.")
    if photosynthesis.get("modifyphoto_and_lmr_forcrop") is not True:
        raise ValueError("native canopy coupling requires crop photosynthesis modification")
    if str(photosynthesis.get("stomatalcond_method", "")).lower() != "medlyn2011":
        raise ValueError("native canopy coupling requires stomatalcond_method='Medlyn2011'")

    roots = read_case_namelist_group(case, "lnd_in", "rooting_profile_inparm")
    if int(roots.get("rooting_profile_method_water", -1)) != 1:
        raise ValueError("native canopy coupling requires the Jackson1996 water-root profile")
    if int(roots.get("rooting_profile_varindex_water", 1)) != 1:
        raise ValueError("native canopy coupling requires water-root profile variant 1")

    if case.value("CALENDAR", "").upper() != "NO_LEAP":
        raise ValueError("native canopy clock adapter currently requires CALENDAR=NO_LEAP")
    return float(co2_ppmv)


def _noleap_calendar_day(clock: RestartClock) -> np.float64:
    ymd = int(clock.current_date)
    month = (ymd // 100) % 100
    day = ymd % 100
    month_lengths = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    if month < 1 or month > 12 or day < 1 or day > month_lengths[month - 1]:
        raise ValueError(f"invalid NO_LEAP restart date {ymd}")
    if clock.current_time_of_day < 0 or clock.current_time_of_day >= int(SECONDS_PER_DAY):
        raise ValueError("restart time of day must lie in [0, 86400)")
    integer_day = 1 + sum(month_lengths[: month - 1]) + day - 1
    return np.float64(integer_day) + np.float64(clock.current_time_of_day) / SECONDS_PER_DAY


def bedrock_level_count_column(
    topology: SubgridTopology,
    surface_data_path: Path,
) -> np.ndarray:

    surface = load_native_variables(surface_data_path, ["zbedrock"], dtype=np.float64)
    source = np.asarray(surface["zbedrock"], dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("fsurdat zbedrock must have dimensions (lat, lon)")
    longitude = np.asarray(topology.grid.longitude_index, dtype=np.int64)
    latitude = np.asarray(topology.grid.latitude_index, dtype=np.int64)
    bedrock_depth_gridcell = source[latitude, longitude]
    if not np.all(np.isfinite(bedrock_depth_gridcell)):
        raise ValueError("active land grid contains missing fsurdat zbedrock")

    _, interface, _ = soil_layer_geometry_20sl_8p5m()
    minimum_bedrock_level = 3
    for level in range(3, NUMBER_OF_SOIL_LEVELS + 1):
        if interface[level - 1] < np.float64(0.4) <= interface[level]:
            minimum_bedrock_level = level
    gridcell_level = np.full(
        bedrock_depth_gridcell.shape,
        NUMBER_OF_SOIL_LEVELS,
        dtype=np.int32,
    )
    for level in range(minimum_bedrock_level, NUMBER_OF_SOIL_LEVELS + 1):
        selected = (interface[level - 1] < bedrock_depth_gridcell) & (
            interface[level] >= bedrock_depth_gridcell
        )
        gridcell_level[selected] = level
    return gridcell_level[np.asarray(topology.column.gridcell, dtype=np.int64)]


def _load_biochemical_parameters(parameter_data_path: Path) -> CanopyBiochemicalPftParameters:
    values = load_native_variables(
        parameter_data_path,
        list(_PARAMETER_VARIABLES),
        dtype=np.float64,
    )
    arrays = tuple(np.asarray(values[name], dtype=np.float64) for name in _PARAMETER_VARIABLES)
    if any(value.ndim != 1 for value in arrays):
        raise ValueError("CLM biochemical PFT parameters must be one-dimensional")
    shape = arrays[0].shape
    if shape[0] < 1 or any(value.shape != shape for value in arrays):
        raise ValueError("CLM biochemical PFT parameter dimensions disagree")
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("CLM biochemical PFT parameters must be finite")
    c3, crop, intercept, slope, respiration_intercept = arrays
    if not np.all((c3 == 0.0) | (c3 == 1.0)):
        raise ValueError("CLM c3psn must contain only zero or one")
    if not np.all((crop == 0.0) | (crop == 1.0)):
        raise ValueError("CLM crop must contain only zero or one")
    return CanopyBiochemicalPftParameters(
        c3_pathway_pft=jnp.asarray(c3 == 1.0),
        crop_pft=jnp.asarray(crop == 1.0),
        vcmax_intercept_pft=jnp.asarray(intercept),
        vcmax_leaf_nitrogen_slope_pft=jnp.asarray(slope),
        leaf_respiration_intercept_pft=jnp.asarray(respiration_intercept),
    )


def load_canopy_hydraulic_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> CanopyHydraulicNativeData:

    co2_ppmv = _require_active_case_configuration(case)
    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    surface_data = _selected_input_path(case, "fsurdat", surface_data_path)
    parameter_data = _selected_input_path(case, "paramfile", parameter_data_path)

    dimensions = load_dimension_sizes(restart)
    if dimensions.get("levcan") != 1:
        raise ValueError("active native canopy coupling requires nlevcan=1")
    if dimensions.get("levgrnd", 0) < NUMBER_OF_SOIL_LEVELS:
        raise ValueError("restart does not contain all twenty hydrologic soil levels")
    topology = load_subgrid_topology(restart)
    clock = load_restart_clock(restart)
    patch_count = topology.patch.gridcell.shape[0]
    column_count = topology.column.gridcell.shape[0]

    restart_names = [
        *_RESTART_SCALAR_FLOAT_VARIABLES,
        *_RESTART_LAYER_VARIABLES,
        *_RESTART_COLUMN_LAYER_VARIABLES,
        "nrad",
    ]
    values = load_native_variables(
        restart,
        restart_names,
        masked_integer_fill_value=-9999,
    )
    for name in _RESTART_SCALAR_FLOAT_VARIABLES:
        if values[name].shape != (patch_count,):
            raise ValueError(f"restart variable {name!r} has unexpected shape {values[name].shape}")
    for name in _RESTART_LAYER_VARIABLES:
        if values[name].shape != (patch_count, 1):
            raise ValueError(f"restart variable {name!r} has unexpected shape {values[name].shape}")
    for name in _RESTART_COLUMN_LAYER_VARIABLES:
        if values[name].shape[0] != column_count or values[name].shape[1] < NUMBER_OF_SOIL_LEVELS:
            raise ValueError(f"restart variable {name!r} lacks native column/soil dimensions")
    if values["nrad"].shape != (patch_count,):
        raise ValueError("restart nrad has unexpected shape")

    vegetation_type = np.asarray(topology.patch.vegetation_type, dtype=np.int64)
    landunit_type = np.asarray(topology.patch.landunit_type, dtype=np.int64)
    exposed = np.asarray(values["FRAC_VEG_NOSNO_ALB"], dtype=np.float64) > 0.0
    active = (
        np.asarray(topology.patch.active, dtype=bool)
        & (landunit_type != LANDUNIT_LAKE)
        & ~((landunit_type >= LANDUNIT_URBAN_MIN) & (landunit_type <= LANDUNIT_URBAN_MAX))
        & exposed
    )
    if not np.any(active):
        raise ValueError("restart contains no active exposed vegetation patches")
    nrad = np.asarray(values["nrad"], dtype=np.int32)
    if not np.all(nrad[active] == 1):
        found = np.unique(nrad[active])
        raise ValueError(f"active nlevcan=1 patches must have nrad=1, found {found.tolist()}")

    required_patch = (
        "frootc",
        "leafn",
        "tlai",
        "tsai",
        "vcmaxcintsun",
        "vcmaxcintsha",
        "tair10",
        "tlai_z",
    )
    for name in required_patch:
        if not np.all(np.isfinite(np.asarray(values[name])[active])):
            raise ValueError(f"active restart field {name!r} contains missing values")
    patch_column = np.asarray(topology.patch.column, dtype=np.int64)
    active_columns = patch_column[active]
    for name in _RESTART_COLUMN_LAYER_VARIABLES:
        soil = np.asarray(values[name])[:, :NUMBER_OF_SOIL_LEVELS]
        if not np.all(np.isfinite(soil[active_columns])):
            raise ValueError(f"active restart soil field {name!r} contains missing values")

    biochemical = _load_biochemical_parameters(parameter_data)
    number_of_pfts = biochemical.c3_pathway_pft.shape[0]
    if np.any((vegetation_type < 0) | (vegetation_type >= number_of_pfts)):
        raise ValueError("restart patch PFT lies outside biochemical parameter arrays")
    c3_noncrop = (
        np.asarray(biochemical.c3_pathway_pft)[vegetation_type]
        & ~np.asarray(biochemical.crop_pft)[vegetation_type]
    )
    luna_active = active & c3_noncrop
    for name in ("vcmx25_z", "jmx25_z"):
        if not np.all(np.isfinite(np.asarray(values[name])[luna_active])):
            raise ValueError(f"active LUNA field {name!r} contains missing values")

    plant_parameters = load_plant_hydraulic_parameters(case, parameter_data)
    if plant_parameters.maximum_root_conductance_pft.shape[0] != number_of_pfts:
        raise ValueError("plant hydraulic and biochemical PFT dimensions disagree")
    plant_restart = load_plant_hydraulic_restart(case, restart)
    if plant_restart.topology.patch.gridcell.shape != topology.patch.gridcell.shape:
        raise ValueError("plant hydraulic restart topology disagrees with native topology")

    bedrock = bedrock_level_count_column(topology, surface_data)
    if np.any((bedrock < 1) | (bedrock > NUMBER_OF_SOIL_LEVELS)):
        raise ValueError("reconstructed column bedrock level lies outside soil levels")
    _, root_interfaces, _ = soil_layer_geometry_20sl_8p5m()
    root_fraction = initialize_jackson_root_fraction(
        np.asarray(plant_parameters.root_profile_beta_pft_variant)[vegetation_type, 0],
        root_interfaces[:NUMBER_OF_SOIL_LEVELS + 1],
        bedrock[patch_column],
    )
    root_fraction[vegetation_type == 0] = 0.0
    return CanopyHydraulicNativeData(
        topology=topology,
        clock=clock,
        inputs=CanopyHydraulicNativeInputs(
            fine_root_carbon_patch=jnp.asarray(values["frootc"], dtype=jnp.float64),
            leaf_nitrogen_patch=jnp.asarray(values["leafn"], dtype=jnp.float64),
            total_leaf_area_patch=jnp.asarray(values["tlai"], dtype=jnp.float64),
            total_stem_area_patch=jnp.asarray(values["tsai"], dtype=jnp.float64),
            hydraulic_conductivity_column_level=jnp.asarray(
                values["HK"][:, :NUMBER_OF_SOIL_LEVELS], dtype=jnp.float64
            ),
            soil_water_potential_column_level=jnp.asarray(
                values["SMP"][:, :NUMBER_OF_SOIL_LEVELS], dtype=jnp.float64
            ),
            bedrock_level_count_column=jnp.asarray(bedrock, dtype=jnp.int32),
            vcmax_scaling_sunlit_patch=jnp.asarray(
                values["vcmaxcintsun"], dtype=jnp.float64
            ),
            vcmax_scaling_shaded_patch=jnp.asarray(
                values["vcmaxcintsha"], dtype=jnp.float64
            ),
            luna_vcmax25_by_layer_patch=jnp.asarray(values["vcmx25_z"], dtype=jnp.float64),
            luna_jmax25_by_layer_patch=jnp.asarray(values["jmx25_z"], dtype=jnp.float64),
            total_leaf_area_by_layer_patch=jnp.asarray(values["tlai_z"], dtype=jnp.float64),
            radiative_layer_count_patch=jnp.asarray(values["nrad"], dtype=jnp.int32),
            air_temperature_ten_day_patch=jnp.asarray(values["tair10"], dtype=jnp.float64),
            atmospheric_co2_ppmv=jnp.float64(co2_ppmv),
            calendar_day=jnp.asarray(_noleap_calendar_day(clock), dtype=jnp.float64),
            root_fraction_patch_level=jnp.asarray(root_fraction),
        ),
        plant_parameters=plant_parameters,
        biochemical_parameters=biochemical,
        initial_plant_state=plant_restart.state,
        restart_path=restart,
        surface_data_path=surface_data,
        parameter_data_path=parameter_data,
    )


def _validate_assembly_shapes(
    topology: SubgridTopology,
    native: CanopyHydraulicNativeInputs,
    plant_parameters: PlantHydraulicParameters,
    biochemical_parameters: CanopyBiochemicalPftParameters,
    initial_plant_state: PlantHydraulicPatchState,
    energy: CanopyEnergyInputs,
    initial_energy: CanopyEnergyInitialState,
    temperature_parameters: CanopyTemperatureParameters,
    soil_liquid_water_column_level: jnp.ndarray,
    soil_ice_water_column_level: jnp.ndarray,
    canopy_radiation: CanopySunShadeState,
    atmospheric_pressure_gridcell: jnp.ndarray,
) -> None:
    patch_count = topology.patch.gridcell.shape[0]
    column_count = topology.column.gridcell.shape[0]
    gridcell_count = topology.grid.longitude_degrees.shape[0]
    patch_shape = (patch_count,)
    column_soil_shape = (column_count, NUMBER_OF_SOIL_LEVELS)
    for name in (
        "fine_root_carbon_patch",
        "leaf_nitrogen_patch",
        "total_leaf_area_patch",
        "total_stem_area_patch",
        "vcmax_scaling_sunlit_patch",
        "vcmax_scaling_shaded_patch",
        "radiative_layer_count_patch",
        "air_temperature_ten_day_patch",
    ):
        if jnp.shape(getattr(native, name)) != patch_shape:
            raise ValueError(f"native hydraulic field {name} must have shape {patch_shape}")
    for name in (
        "luna_vcmax25_by_layer_patch",
        "luna_jmax25_by_layer_patch",
        "total_leaf_area_by_layer_patch",
    ):
        if jnp.shape(getattr(native, name)) != (patch_count, 1):
            raise ValueError(f"native hydraulic field {name} must have shape ({patch_count}, 1)")
    for name in (
        "hydraulic_conductivity_column_level",
        "soil_water_potential_column_level",
    ):
        if jnp.shape(getattr(native, name)) != column_soil_shape:
            raise ValueError(f"native hydraulic field {name} must have shape {column_soil_shape}")
    if jnp.shape(native.bedrock_level_count_column) != (column_count,):
        raise ValueError("native bedrock levels must contain every column")
    if (native.root_fraction_patch_level is not None
            and jnp.shape(native.root_fraction_patch_level) != (patch_count, NUMBER_OF_SOIL_LEVELS)):
        raise ValueError("native root fractions must contain every patch and soil level")
    for name, value in (
        ("soil liquid water", soil_liquid_water_column_level),
        ("soil ice water", soil_ice_water_column_level),
    ):
        if jnp.shape(value) != (column_count, NUMBER_OF_TOTAL_LEVELS):
            raise ValueError(
                f"{name} must have shape ({column_count}, {NUMBER_OF_TOTAL_LEVELS})"
            )
    if jnp.shape(native.atmospheric_co2_ppmv) != () or jnp.shape(native.calendar_day) != ():
        raise ValueError("native CO2 and calendar day must be scalars")
    for name, value in zip(energy._fields, energy, strict=True):
        if jnp.shape(value) != patch_shape:
            raise ValueError(f"energy field {name} must have shape {patch_shape}")
    if jnp.shape(initial_energy.vegetation_temperature_patch) != patch_shape:
        raise ValueError("initial vegetation temperature must contain every patch")
    for name in (
        "ground_layer_thickness_column",
        "saturated_hydraulic_conductivity_column",
    ):
        value = getattr(temperature_parameters, name)
        if jnp.shape(value)[0] != column_count or jnp.shape(value)[1] < NUMBER_OF_SOIL_LEVELS:
            raise ValueError(f"temperature parameter {name} lacks column soil levels")
    for name in (
        "sunlit_leaf_area_patch",
        "shaded_leaf_area_patch",
    ):
        if jnp.shape(getattr(canopy_radiation, name)) != patch_shape:
            raise ValueError(f"canopy radiation field {name} must contain every patch")
    for name in (
        "absorbed_par_sunlit_by_layer_patch",
        "absorbed_par_shaded_by_layer_patch",
    ):
        if jnp.shape(getattr(canopy_radiation, name)) != (patch_count, 1):
            raise ValueError(f"canopy radiation field {name} must have one active layer")
    if jnp.shape(atmospheric_pressure_gridcell) != (gridcell_count,):
        raise ValueError("undownscaled atmospheric pressure must contain every land gridcell")
    pft_count = plant_parameters.maximum_root_conductance_pft.shape[0]
    if biochemical_parameters.c3_pathway_pft.shape != (pft_count,):
        raise ValueError("plant hydraulic and biochemical PFT dimensions disagree")
    if initial_plant_state.water_potential_segment.shape != (patch_count, 4):
        raise ValueError("initial plant water potential must have shape (patch, 4)")


def _leaf_nitrogen_per_area(leaf_nitrogen, total_leaf_area):
    has_leaf = total_leaf_area > jnp.float64(0.0)
    divisor = fortran_multiply(total_leaf_area, jnp.float64(1.0))
    return jnp.where(
        has_leaf,
        fortran_divide(leaf_nitrogen, jnp.where(has_leaf, divisor, jnp.float64(1.0))),
        jnp.float64(0.0),
    )


def _luna_shaded_capacity(sunlit, sunlit_scaling, shaded_scaling):
    has_sunlit_scaling = sunlit_scaling > jnp.float64(0.0)
    return jnp.where(
        has_sunlit_scaling,
        fortran_divide(
            fortran_multiply(sunlit, shaded_scaling),
            jnp.where(has_sunlit_scaling, sunlit_scaling, jnp.float64(1.0)),
        ),
        sunlit,
    )


def assemble_canopy_hydraulic_coupling(
    topology: SubgridTopology,
    native: CanopyHydraulicNativeInputs,
    plant_parameters: PlantHydraulicParameters,
    biochemical_parameters: CanopyBiochemicalPftParameters,
    initial_plant_state: PlantHydraulicPatchState,
    energy: CanopyEnergyInputs,
    initial_energy: CanopyEnergyInitialState,
    temperature_parameters: CanopyTemperatureParameters,
    soil_liquid_water_column_level: jnp.ndarray,
    soil_ice_water_column_level: jnp.ndarray,
    canopy_radiation: CanopySunShadeState,
    atmospheric_pressure_gridcell: jnp.ndarray,
    *,
    root_vulnerability_power_base: jnp.ndarray | float = 2.0,
    solar_declination_radians: jnp.ndarray | None = None,
) -> CanopyHydraulicCouplingInputs:

    _validate_assembly_shapes(
        topology,
        native,
        plant_parameters,
        biochemical_parameters,
        initial_plant_state,
        energy,
        initial_energy,
        temperature_parameters,
        soil_liquid_water_column_level,
        soil_ice_water_column_level,
        canopy_radiation,
        atmospheric_pressure_gridcell,
    )
    active = jnp.asarray(energy.active_patch, dtype=jnp.bool_)
    patch = topology.patch
    column_index = jnp.asarray(patch.column, dtype=jnp.int32)
    gridcell_index = jnp.asarray(patch.gridcell, dtype=jnp.int32)
    vegetation_type = jnp.asarray(patch.vegetation_type, dtype=jnp.int32)

    def safe_patch(value: jnp.ndarray, default: float) -> jnp.ndarray:
        return jnp.where(
            active,
            jnp.asarray(value, dtype=jnp.float64),
            jnp.float64(default),
        )

    def safe_patch_level(value: jnp.ndarray, default: float) -> jnp.ndarray:
        return jnp.where(
            active[:, None],
            jnp.asarray(value, dtype=jnp.float64),
            jnp.float64(default),
        )

    def selected_pft(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(value)[vegetation_type]

    node_host, interface_host, _ = soil_layer_geometry_20sl_8p5m()
    interface = jnp.asarray(interface_host[: NUMBER_OF_SOIL_LEVELS + 1], dtype=jnp.float64)
    node_depth = jnp.asarray(
        node_host[:NUMBER_OF_SOIL_LEVELS],
        dtype=jnp.float64,
    )
    bedrock = jnp.asarray(native.bedrock_level_count_column, dtype=jnp.int32)[column_index]
    root_beta = selected_pft(plant_parameters.root_profile_beta_pft_variant)[:, 0]
    root_fraction = (
        jackson_root_fraction(root_beta, interface, bedrock)
        if native.root_fraction_patch_level is None
        else jnp.asarray(native.root_fraction_patch_level, dtype=jnp.float64)
    )

    layer_thickness = jnp.asarray(
        temperature_parameters.ground_layer_thickness_column,
        dtype=jnp.float64,
    )[column_index, :NUMBER_OF_SOIL_LEVELS]
    saturated_conductivity = jnp.asarray(
        temperature_parameters.saturated_hydraulic_conductivity_column,
        dtype=jnp.float64,
    )[column_index, :NUMBER_OF_SOIL_LEVELS]
    hydraulic_conductivity = jnp.asarray(
        native.hydraulic_conductivity_column_level,
        dtype=jnp.float64,
    )[column_index]
    soil_water_potential = jnp.asarray(
        native.soil_water_potential_column_level,
        dtype=jnp.float64,
    )[column_index]
    layer_thickness = safe_patch_level(layer_thickness, 0.1)
    saturated_conductivity = safe_patch_level(saturated_conductivity, 1.0e-3)
    hydraulic_conductivity = safe_patch_level(hydraulic_conductivity, 1.0e-4)
    soil_water_potential = safe_patch_level(soil_water_potential, -100_000.0)
    node_depth_patch = jnp.broadcast_to(node_depth, root_fraction.shape)

    fine_root_carbon = safe_patch(native.fine_root_carbon_patch, 1.0)
    total_leaf_area = safe_patch(native.total_leaf_area_patch, 1.0)
    total_stem_area = safe_patch(native.total_stem_area_patch, 0.1)
    root_conductance_parameters = SoilRootConductanceParameters(
        fine_root_to_leaf_ratio=selected_pft(
            plant_parameters.fine_root_to_leaf_ratio_pft
        ),
        maximum_root_conductance=selected_pft(
            plant_parameters.maximum_root_conductance_pft
        ),
        root_half_conductance_potential=selected_pft(
            plant_parameters.water_potential_at_half_conductance_pft_segment
        )[:, ROOT],
        root_vulnerability_shape=selected_pft(
            plant_parameters.vulnerability_shape_pft_segment
        )[:, ROOT],
        root_radius=selected_pft(plant_parameters.root_radius_pft),
        root_density=selected_pft(plant_parameters.root_density_pft),
    )
    root_conductance = soil_root_interface_conductance(
        SoilRootConductanceInputs(
            fine_root_carbon=fine_root_carbon,
            root_fraction_level=root_fraction,
            soil_layer_thickness_level=layer_thickness,
            soil_node_depth_level=node_depth_patch,
            saturated_hydraulic_conductivity_level=saturated_conductivity,
            hydraulic_conductivity_level=hydraulic_conductivity,
            soil_water_potential_level=soil_water_potential,
            total_leaf_area=total_leaf_area,
            total_stem_area=total_stem_area,
        ),
        root_conductance_parameters,
        vulnerability_power_base=root_vulnerability_power_base,
    )
    soil_start = NUMBER_OF_SNOW_LEVELS
    soil_stop = soil_start + NUMBER_OF_SOIL_LEVELS
    layer_thickness_column = jnp.asarray(
        temperature_parameters.ground_layer_thickness_column,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_SOIL_LEVELS]
    liquid_water_column = jnp.asarray(
        soil_liquid_water_column_level,
        dtype=jnp.float64,
    )[:, soil_start:soil_stop]
    ice_water_column = jnp.asarray(
        soil_ice_water_column_level,
        dtype=jnp.float64,
    )[:, soil_start:soil_stop]
    total_volumetric_water_column = fortran_add(
        fortran_divide(
            liquid_water_column,
            fortran_multiply(
                layer_thickness_column,
                jnp.float64(DENSITY_FRESH_WATER),
            ),
        ),
        fortran_divide(
            ice_water_column,
            fortran_multiply(
                layer_thickness_column,
                jnp.float64(DENSITY_ICE),
            ),
        ),
    )
    root_zone_wetness = root_zone_soil_wetness(
        root_fraction,
        total_volumetric_water_column[column_index],
        jnp.asarray(
            temperature_parameters.saturated_water_content_column,
            dtype=jnp.float64,
        )[column_index, :NUMBER_OF_SOIL_LEVELS],
        jnp.asarray(
            temperature_parameters.saturated_suction_column,
            dtype=jnp.float64,
        )[column_index, :NUMBER_OF_SOIL_LEVELS],
        jnp.asarray(
            temperature_parameters.clapp_hornberger_b_column,
            dtype=jnp.float64,
        )[column_index, :NUMBER_OF_SOIL_LEVELS],
        selected_pft(
            plant_parameters.soil_water_potential_full_opening_pft
        ),
        selected_pft(
            plant_parameters.soil_water_potential_full_closure_pft
        ),
    )

    leaf_saturation = qsat(
        safe_patch(initial_energy.vegetation_temperature_patch, 280.0),
        safe_patch(energy.atmospheric_pressure_patch, 100_000.0),
    )
    canopy_air_humidity = fortran_divide(
        fortran_add(
            safe_patch(energy.atmospheric_specific_humidity_patch, 0.005),
            safe_patch(energy.ground_specific_humidity_patch, 0.005),
        ),
        jnp.float64(2.0),
    )
    plant_inputs = PlantHydraulicPatchInputs(
        sunlit_leaf_area=safe_patch(canopy_radiation.sunlit_leaf_area_patch, 0.6),
        shaded_leaf_area=safe_patch(canopy_radiation.shaded_leaf_area_patch, 0.4),
        exposed_leaf_area=safe_patch(energy.exposed_leaf_area_index_patch, 1.0),
        exposed_stem_area=safe_patch(energy.exposed_stem_area_index_patch, 0.1),
        total_stem_area=total_stem_area,
        canopy_top_height=safe_patch(energy.canopy_top_height_patch, 1.0),
        dry_foliage_fraction=safe_patch(energy.dry_foliage_fraction_patch, 0.8),
        air_density=safe_patch(energy.air_density_patch, 1.2),
        atmospheric_pressure=safe_patch(energy.atmospheric_pressure_patch, 100_000.0),
        reference_temperature=safe_patch(energy.reference_temperature_patch, 280.0),
        leaf_saturation_specific_humidity=leaf_saturation.specific_humidity,
        canopy_air_specific_humidity=canopy_air_humidity,
        soil_root_conductance_level=root_conductance.soil_root_conductance_level,
        soil_water_potential_level=soil_water_potential,
        soil_node_depth_level=node_depth_patch,
    )

    if solar_declination_radians is None:
        solar_declination_radians, _ = solar_declination(
            native.calendar_day, FIXED_YEAR_2000_ORBIT
        )
    latitude = native_grid_radians(topology.grid.latitude_degrees)
    current_daylength = daylength(latitude, solar_declination_radians)
    maximum = maximum_daylength(latitude, FIXED_YEAR_2000_ORBIT.obliquity_radians)
    daylength_ratio_squared = fortran_divide(
        fortran_multiply(current_daylength, current_daylength),
        fortran_multiply(maximum, maximum),
    )
    daylength_factor = jnp.minimum(
        jnp.float64(1.0),
        jnp.maximum(jnp.float64(0.01), daylength_ratio_squared),
    )[gridcell_index]

    c3_pathway = selected_pft(biochemical_parameters.c3_pathway_pft).astype(jnp.bool_)
    crop = selected_pft(biochemical_parameters.crop_pft).astype(jnp.bool_)
    leaf_nitrogen = safe_patch(native.leaf_nitrogen_patch, 0.0)
    leaf_nitrogen_concentration = _leaf_nitrogen_per_area(leaf_nitrogen, total_leaf_area)
    leaf_nitrogen_concentration = jnp.minimum(
        leaf_nitrogen_concentration,
        jnp.float64(10.0),
    )
    vcmax25_top = fortran_multiply(
        fortran_add(
            selected_pft(biochemical_parameters.vcmax_intercept_pft),
            fortran_multiply(
                selected_pft(biochemical_parameters.vcmax_leaf_nitrogen_slope_pft),
                leaf_nitrogen_concentration,
            ),
        ),
        daylength_factor,
    )
    ten_day_temperature = safe_patch(native.air_temperature_ten_day_patch, 288.15)
    acclimation_celsius = jnp.minimum(
        jnp.maximum(
            fortran_add(
                ten_day_temperature,
                -jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
            ),
            jnp.float64(11.0),
        ),
        jnp.float64(35.0),
    )
    jmax25_top = fortran_multiply(
        fortran_add(
            jnp.float64(2.59),
            -fortran_multiply(jnp.float64(0.035), acclimation_celsius),
        ),
        vcmax25_top,
    )
    tpu25_top = fortran_multiply(jnp.float64(0.167), vcmax25_top)
    kp25_top = fortran_multiply(jnp.float64(20_000.0), vcmax25_top)

    sunlit_scaling = safe_patch(native.vcmax_scaling_sunlit_patch, 1.0)
    shaded_scaling = safe_patch(native.vcmax_scaling_shaded_patch, 0.0)
    scaling = jnp.stack((sunlit_scaling, shaded_scaling), axis=1)
    default_vcmax = fortran_multiply(vcmax25_top[:, None], scaling)
    default_jmax = fortran_multiply(jmax25_top[:, None], scaling)
    default_tpu = fortran_multiply(tpu25_top[:, None], scaling)

    luna_vcmax_sunlit = safe_patch(native.luna_vcmax25_by_layer_patch[:, 0], 0.0)
    luna_jmax_sunlit = safe_patch(native.luna_jmax25_by_layer_patch[:, 0], 0.0)
    luna_tpu_sunlit = fortran_multiply(jnp.float64(0.167), luna_vcmax_sunlit)
    def luna_shaded(sunlit: jnp.ndarray) -> jnp.ndarray:
        return _luna_shaded_capacity(sunlit, sunlit_scaling, shaded_scaling)

    luna_vcmax = jnp.stack(
        (
            luna_vcmax_sunlit,
            luna_shaded(luna_vcmax_sunlit),
        ),
        axis=1,
    )
    luna_jmax = jnp.stack(
        (
            luna_jmax_sunlit,
            luna_shaded(luna_jmax_sunlit),
        ),
        axis=1,
    )
    luna_tpu = jnp.stack(
        (
            luna_tpu_sunlit,
            luna_shaded(luna_tpu_sunlit),
        ),
        axis=1,
    )
    use_luna_capacity = c3_pathway & ~crop
    vcmax25 = jnp.where(use_luna_capacity[:, None], luna_vcmax, default_vcmax)
    jmax25 = jnp.where(use_luna_capacity[:, None], luna_jmax, default_jmax)
    tpu25 = jnp.where(use_luna_capacity[:, None], luna_tpu, default_tpu)
    kp25 = fortran_multiply(kp25_top[:, None], scaling)

    temperature_celsius = fortran_add(
        ten_day_temperature,
        -jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
    )
    lmr25_top = jnp.where(
        leaf_nitrogen_concentration > jnp.float64(0.0),
        fortran_add(
            fortran_add(
                selected_pft(biochemical_parameters.leaf_respiration_intercept_pft),
                fortran_multiply(
                    leaf_nitrogen_concentration,
                    jnp.float64(0.2061),
                ),
            ),
            -fortran_multiply(jnp.float64(0.0402), temperature_celsius),
        ),
        jnp.float64(0.0),
    )
    lmr25 = fortran_multiply(lmr25_top[:, None], scaling)

    pressure_gridcell = jnp.asarray(atmospheric_pressure_gridcell, dtype=jnp.float64)
    co2_molar_fraction = fortran_multiply(
        native.atmospheric_co2_ppmv,
        jnp.float64(1.0e-6),
    )
    atmospheric_co2 = fortran_multiply(
        co2_molar_fraction,
        pressure_gridcell,
    )[gridcell_index]
    atmospheric_o2 = fortran_multiply(
        jnp.float64(0.209),
        pressure_gridcell,
    )[gridcell_index]
    absorbed_par = jnp.stack(
        (
            safe_patch(canopy_radiation.absorbed_par_sunlit_by_layer_patch[:, 0], 0.0),
            safe_patch(canopy_radiation.absorbed_par_shaded_by_layer_patch[:, 0], 0.0),
        ),
        axis=1,
    )
    layer_leaf_area = safe_patch(native.total_leaf_area_by_layer_patch[:, 0], 1.0)
    quantum_efficiency_c4 = jnp.where(
        c3_pathway,
        jnp.float64(0.0),
        jnp.float64(0.05),
    )
    colimitation_curvature = jnp.where(
        c3_pathway,
        jnp.float64(0.98),
        jnp.float64(0.80),
    )
    biochemistry = HydraulicPhotosynthesisBiochemicalParameters(
        c3_pathway=c3_pathway,
        maximum_carboxylation_rate_25=safe_patch_level(vcmax25, 1.0),
        maximum_electron_transport_rate_25=safe_patch_level(jmax25, 1.0),
        triose_phosphate_utilization_rate_25=safe_patch_level(tpu25, 0.167),
        pep_carboxylase_co2_slope_25=safe_patch_level(kp25, 20_000.0),
        leaf_maintenance_respiration_25=safe_patch_level(lmr25, 0.1),
        absorbed_par=absorbed_par,
        layer_leaf_area=layer_leaf_area,
        ten_day_temperature=ten_day_temperature,
        atmospheric_co2=safe_patch(atmospheric_co2, 36.7),
        atmospheric_o2=safe_patch(atmospheric_o2, 20_900.0),
        atmospheric_pressure=safe_patch(energy.atmospheric_pressure_patch, 100_000.0),
        quantum_efficiency_c4=quantum_efficiency_c4,
        ac_aj_colimitation_curvature=colimitation_curvature,
        medlyn_slope=selected_pft(plant_parameters.medlyn_slope_pft),
        medlyn_intercept=selected_pft(plant_parameters.medlyn_intercept_pft),
        light_inhibition=jnp.ones(active.shape, dtype=jnp.bool_),
    )
    controls = HydraulicPhotosynthesisPatchControls(
        crop=crop,
        modify_photosynthesis_and_respiration_for_crop=jnp.ones(
            active.shape, dtype=jnp.bool_
        ),
        sunlit_stomatal_ozone_coefficient=jnp.ones(active.shape, dtype=jnp.float64),
        shaded_stomatal_ozone_coefficient=jnp.ones(active.shape, dtype=jnp.float64),
    )
    safe_initial_state = PlantHydraulicPatchState(
        water_potential_segment=jnp.where(
            active[:, None],
            jnp.asarray(initial_plant_state.water_potential_segment, dtype=jnp.float64),
            jnp.float64(-100_000.0),
        ),
        vegetation_transpiration=jnp.where(
            active,
            jnp.asarray(initial_plant_state.vegetation_transpiration, dtype=jnp.float64),
            jnp.float64(0.0),
        ),
    )
    return CanopyHydraulicCouplingInputs(
        biochemistry=biochemistry,
        controls=controls,
        plant_inputs=plant_inputs,
        plant_parameters=select_patch_parameters(plant_parameters, vegetation_type),
        initial_plant_state=safe_initial_state,
        root_zone_wetness_patch=jnp.where(
            active,
            root_zone_wetness,
            jnp.float64(CLM_SPECIAL_VALUE),
        ),
    )
