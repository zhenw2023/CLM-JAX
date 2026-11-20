
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.canopy_hydraulic_inputs import bedrock_level_count_column
from clm_jax.io.canopy_temperature_restart import soil_layer_geometry_20sl_8p5m
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.plant_hydraulics import load_plant_hydraulic_parameters
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    SubgridTopology,
    configured_land_restart,
    load_subgrid_topology,
)
from clm_jax.physics.maintenance_respiration import (
    FIRST_PROGNOSTIC_CROP_PFT,
    NUMBER_OF_GROUND_LEVELS,
    MaintenanceRespirationInputs,
    MaintenanceRespirationParameters,
)
from clm_jax.physics.photosynthesis import SHADED, SUNLIT
from clm_jax.physics.plant_hydraulics import jackson_root_fraction

NUMBER_OF_HYDROLOGIC_SOIL_LEVELS = 20
_RESTART_PATCH_FLOAT_VARIABLES = (
    "T_REF2M",
    "T10_VALUE",
    "frootn",
    "livestemn",
    "livecrootn",
    "grainn",
)
_PARAMETER_VARIABLES = ("br_mr", "q10_mr", "woody")


class MaintenanceRespirationTemperatureState(NamedTuple):

    soil_temperature_column_level: jnp.ndarray
    reference_temperature_2m_patch: jnp.ndarray
    ten_day_temperature_patch: jnp.ndarray


class MaintenanceRespirationNitrogenState(NamedTuple):

    fine_root_nitrogen_patch: jnp.ndarray
    live_stem_nitrogen_patch: jnp.ndarray
    live_coarse_root_nitrogen_patch: jnp.ndarray
    grain_nitrogen_patch: jnp.ndarray


class MaintenanceRespirationNativeData(NamedTuple):

    topology: SubgridTopology
    soil_column: jnp.ndarray
    soil_patch: jnp.ndarray
    carbon_root_fraction_patch_level: jnp.ndarray
    temperature: MaintenanceRespirationTemperatureState
    nitrogen: MaintenanceRespirationNitrogenState
    parameters: MaintenanceRespirationParameters
    ten_day_accumulator_steps_patch: jnp.ndarray
    ten_day_accumulator_period_steps: int
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path


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


def _finite_scalar(name: str, value: jnp.ndarray) -> np.float64:
    host = np.asarray(value, dtype=np.float64)
    if host.size != 1 or not np.isfinite(host.reshape(())):
        raise ValueError(f"CLM parameter {name!r} must contain one finite scalar")
    return np.float64(host.reshape(()))


def load_maintenance_respiration_parameters(
    case: CimeCase,
    parameter_data_path: str | Path | None = None,
) -> MaintenanceRespirationParameters:

    parameter_data = _selected_input_path(case, "paramfile", parameter_data_path)
    loaded = load_native_variables(
        parameter_data,
        list(_PARAMETER_VARIABLES),
        dtype=np.float64,
    )
    base_rate = _finite_scalar("br_mr", loaded["br_mr"])
    temperature_q10 = _finite_scalar("q10_mr", loaded["q10_mr"])
    woody = np.asarray(loaded["woody"], dtype=np.float64)
    if woody.ndim != 1 or woody.size <= FIRST_PROGNOSTIC_CROP_PFT:
        raise ValueError("CLM woody must be a nonempty native PFT vector")
    if not np.all((woody == 0.0) | (woody == 1.0)):
        raise ValueError("CLM woody must contain only zero or one")

    cnmresp = read_case_namelist_group(case, "lnd_in", "cnmresp_inparm")
    configured_root_rate = cnmresp.get("br_root")
    if configured_root_rate is None:
        root_base_rate = base_rate
    elif (
        isinstance(configured_root_rate, bool)
        or not isinstance(configured_root_rate, int | float)
        or not np.isfinite(configured_root_rate)
    ):
        raise ValueError("cnmresp_inparm br_root must be a finite scalar")
    elif abs(float(configured_root_rate)) > 1.0e30:
        root_base_rate = base_rate
    else:
        root_base_rate = np.float64(configured_root_rate)

    photosynthesis = read_case_namelist_group(case, "lnd_in", "photosyns_inparm")
    root_stem_acclimation = photosynthesis.get("rootstem_acc", False)
    if not isinstance(root_stem_acclimation, bool):
        raise ValueError("photosyns_inparm rootstem_acc must be logical")
    if base_rate < 0.0 or root_base_rate < 0.0 or temperature_q10 <= 0.0:
        raise ValueError("CNMResp rates must be nonnegative and q10_mr must be positive")
    return MaintenanceRespirationParameters(
        base_rate=base_rate,
        root_base_rate=root_base_rate,
        temperature_q10=temperature_q10,
        woody_pft=jnp.asarray(woody),
        first_prognostic_crop_pft=FIRST_PROGNOSTIC_CROP_PFT,
        root_stem_acclimation=root_stem_acclimation,
    )


def _carbon_root_variant(case: CimeCase, number_of_variants: int) -> int:
    roots = read_case_namelist_group(case, "lnd_in", "rooting_profile_inparm")
    if int(roots.get("rooting_profile_method_carbon", -1)) != 1:
        raise ValueError("native CNMResp adapter requires the Jackson1996 carbon-root profile")
    variant = int(roots.get("rooting_profile_varindex_carbon", 2))
    if variant < 1 or variant > number_of_variants:
        raise ValueError("carbon-root profile variant lies outside rootprof_beta")
    return variant


def load_maintenance_respiration_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> MaintenanceRespirationNativeData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    for key, expected in {"use_cn": True, "use_crop": True, "use_fates": False}.items():
        if general.get(key) is not expected:
            raise ValueError(f"native CNMResp adapter requires {key}={expected!r}")

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
    if dimensions.get("levgrnd") != NUMBER_OF_GROUND_LEVELS:
        raise ValueError("restart must contain the native 25 ground levels")

    topology = load_subgrid_topology(restart)
    patch_count = topology.patch.gridcell.shape[0]
    column_count = topology.column.gridcell.shape[0]
    values = load_native_variables(
        restart,
        [
            "T_SOISNO",
            *_RESTART_PATCH_FLOAT_VARIABLES,
            "T10_NSTEPS",
            "T10_PERIOD",
        ],
        masked_integer_fill_value=-9999,
    )
    for name in _RESTART_PATCH_FLOAT_VARIABLES:
        if values[name].shape != (patch_count,):
            raise ValueError(f"restart variable {name!r} must contain every patch")
    if values["T10_NSTEPS"].shape != (patch_count,):
        raise ValueError("restart T10_NSTEPS must contain every patch")
    if values["T10_PERIOD"].size != 1:
        raise ValueError("restart T10_PERIOD must be scalar")
    ten_day_period = int(np.asarray(values["T10_PERIOD"]).reshape(()))
    if ten_day_period <= 0:
        raise ValueError("restart T10_PERIOD must be positive")

    soil_temperature = np.asarray(values["T_SOISNO"], dtype=np.float64)
    if soil_temperature.ndim != 2 or soil_temperature.shape[0] != column_count:
        raise ValueError("restart T_SOISNO must have native (column, level) dimensions")
    if soil_temperature.shape[1] < NUMBER_OF_GROUND_LEVELS:
        raise ValueError("restart T_SOISNO lacks the ground-level tail")
    soil_temperature = soil_temperature[:, -NUMBER_OF_GROUND_LEVELS:]

    column_active = np.asarray(topology.column.active, dtype=bool)
    column_landunit = np.asarray(topology.column.landunit_type, dtype=np.int32)
    soil_column = column_active & (
        (column_landunit == LANDUNIT_SOIL) | (column_landunit == LANDUNIT_CROP)
    )
    patch_active = np.asarray(topology.patch.active, dtype=bool)
    patch_landunit = np.asarray(topology.patch.landunit_type, dtype=np.int32)
    soil_patch = patch_active & (
        (patch_landunit == LANDUNIT_SOIL) | (patch_landunit == LANDUNIT_CROP)
    )
    patch_column = np.asarray(topology.patch.column, dtype=np.int32)
    if np.any((patch_column < 0) | (patch_column >= column_count)):
        raise ValueError("restart patch-to-column topology is out of bounds")
    if not np.all(soil_column[patch_column[soil_patch]]):
        raise ValueError("filtered soil patches must map to filtered soil columns")
    if not np.all(np.isfinite(soil_temperature[soil_column])):
        raise ValueError("filtered soil columns contain missing ground temperature")

    vegetation_type = np.asarray(topology.patch.vegetation_type, dtype=np.int32)
    plant_parameters = load_plant_hydraulic_parameters(case, parameter_data)
    root_profiles = np.asarray(
        plant_parameters.root_profile_beta_pft_variant,
        dtype=np.float64,
    )
    if np.any((vegetation_type < 0) | (vegetation_type >= root_profiles.shape[0])):
        raise ValueError("restart vegetation type lies outside rootprof_beta")
    root_variant = _carbon_root_variant(case, root_profiles.shape[1])
    bedrock_column = bedrock_level_count_column(topology, surface_data)
    bedrock_patch = bedrock_column[patch_column]
    _, soil_interface, _ = soil_layer_geometry_20sl_8p5m()
    hydrologic_root_fraction = jackson_root_fraction(
        jnp.asarray(root_profiles[vegetation_type, root_variant - 1]),
        jnp.asarray(
            soil_interface[: NUMBER_OF_HYDROLOGIC_SOIL_LEVELS + 1],
            dtype=jnp.float64,
        ),
        jnp.asarray(bedrock_patch, dtype=jnp.int32),
    )
    carbon_root_fraction = jnp.concatenate(
        (
            hydrologic_root_fraction,
            jnp.zeros(
                (patch_count, NUMBER_OF_GROUND_LEVELS - NUMBER_OF_HYDROLOGIC_SOIL_LEVELS),
                dtype=jnp.float64,
            ),
        ),
        axis=1,
    )

    for name in _RESTART_PATCH_FLOAT_VARIABLES:
        host = np.asarray(values[name], dtype=np.float64)
        if not np.all(np.isfinite(host[soil_patch])):
            raise ValueError(f"filtered soil patches contain missing restart field {name!r}")
    ten_day_steps = np.asarray(values["T10_NSTEPS"], dtype=np.int32)
    if np.any(ten_day_steps[soil_patch] < 0):
        raise ValueError("filtered soil patches contain invalid T10 sample counts")

    return MaintenanceRespirationNativeData(
        topology=topology,
        soil_column=jnp.asarray(soil_column),
        soil_patch=jnp.asarray(soil_patch),
        carbon_root_fraction_patch_level=carbon_root_fraction,
        temperature=MaintenanceRespirationTemperatureState(
            soil_temperature_column_level=jnp.asarray(soil_temperature),
            reference_temperature_2m_patch=jnp.asarray(values["T_REF2M"]),
            ten_day_temperature_patch=jnp.asarray(values["T10_VALUE"]),
        ),
        nitrogen=MaintenanceRespirationNitrogenState(
            fine_root_nitrogen_patch=jnp.asarray(values["frootn"]),
            live_stem_nitrogen_patch=jnp.asarray(values["livestemn"]),
            live_coarse_root_nitrogen_patch=jnp.asarray(values["livecrootn"]),
            grain_nitrogen_patch=jnp.asarray(values["grainn"]),
        ),
        parameters=load_maintenance_respiration_parameters(case, parameter_data),
        ten_day_accumulator_steps_patch=jnp.asarray(ten_day_steps),
        ten_day_accumulator_period_steps=ten_day_period,
        restart_path=restart,
        surface_data_path=surface_data,
        parameter_data_path=parameter_data,
    )


def assemble_maintenance_respiration_inputs(
    native: MaintenanceRespirationNativeData,
    vegetation_exposed_patch: jnp.ndarray,
    sunlit_leaf_area_patch: jnp.ndarray,
    shaded_leaf_area_patch: jnp.ndarray,
    leaf_maintenance_respiration_patch_leaf: jnp.ndarray,
    temperature: MaintenanceRespirationTemperatureState | None = None,
    nitrogen: MaintenanceRespirationNitrogenState | None = None,
) -> MaintenanceRespirationInputs:

    patch_count = native.topology.patch.gridcell.shape[0]
    patch_shape = (patch_count,)
    exposed = jnp.asarray(vegetation_exposed_patch, dtype=jnp.int32)
    sunlit_area = jnp.asarray(sunlit_leaf_area_patch, dtype=jnp.float64)
    shaded_area = jnp.asarray(shaded_leaf_area_patch, dtype=jnp.float64)
    leaf_respiration = jnp.asarray(
        leaf_maintenance_respiration_patch_leaf,
        dtype=jnp.float64,
    )
    for name, value in {
        "vegetation_exposed_patch": exposed,
        "sunlit_leaf_area_patch": sunlit_area,
        "shaded_leaf_area_patch": shaded_area,
    }.items():
        if value.shape != patch_shape:
            raise ValueError(f"{name} must contain every native patch")
    if leaf_respiration.shape != (patch_count, 2):
        raise ValueError("leaf maintenance respiration must have shape (patch, 2)")

    current_temperature = native.temperature if temperature is None else temperature
    current_nitrogen = native.nitrogen if nitrogen is None else nitrogen
    return MaintenanceRespirationInputs(
        soil_column=native.soil_column,
        soil_patch=native.soil_patch,
        patch_column_index=native.topology.patch.column,
        vegetation_type_patch=native.topology.patch.vegetation_type,
        vegetation_exposed_patch=exposed,
        sunlit_leaf_area_patch=sunlit_area,
        shaded_leaf_area_patch=shaded_area,
        carbon_root_fraction_patch_level=native.carbon_root_fraction_patch_level,
        soil_temperature_column_level=current_temperature.soil_temperature_column_level,
        reference_temperature_2m_patch=current_temperature.reference_temperature_2m_patch,
        ten_day_temperature_patch=current_temperature.ten_day_temperature_patch,
        sunlit_leaf_respiration_patch=leaf_respiration[:, SUNLIT],
        shaded_leaf_respiration_patch=leaf_respiration[:, SHADED],
        fine_root_nitrogen_patch=current_nitrogen.fine_root_nitrogen_patch,
        live_stem_nitrogen_patch=current_nitrogen.live_stem_nitrogen_patch,
        live_coarse_root_nitrogen_patch=current_nitrogen.live_coarse_root_nitrogen_patch,
        grain_nitrogen_patch=current_nitrogen.grain_nitrogen_patch,
    )


__all__ = [
    "NUMBER_OF_HYDROLOGIC_SOIL_LEVELS",
    "MaintenanceRespirationNativeData",
    "MaintenanceRespirationNitrogenState",
    "MaintenanceRespirationTemperatureState",
    "assemble_maintenance_respiration_inputs",
    "load_maintenance_respiration_native_data",
    "load_maintenance_respiration_parameters",
]
