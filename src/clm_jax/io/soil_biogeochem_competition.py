
from __future__ import annotations

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
    get_cn_flux_field,
)
from clm_jax.physics.cn_zero_manifest import NUMBER_OF_DECOMPOSITION_LEVELS
from clm_jax.physics.soil_biogeochem_competition import (
    NUMBER_OF_SOIL_LEVELS,
    SoilBiogeochemCompetitionInputs,
    SoilBiogeochemCompetitionParameters,
    SoilBiogeochemCompetitionState,
)

_PARAMETER_VARIABLES = (
    "compet_plant_no3",
    "compet_plant_nh4",
    "compet_decomp_no3",
    "compet_decomp_nh4",
    "compet_denit",
    "compet_nit",
)
_RESTART_VARIABLES = (
    "sminn_vr",
    "smin_nh4_vr",
    "smin_no3_vr",
    "fpi_vr",
    "fpg",
)


class SoilBiogeochemCompetitionPersistentState(NamedTuple):

    mineral_nitrogen_column_level: jnp.ndarray
    mineral_ammonium_column_level: jnp.ndarray
    mineral_nitrate_column_level: jnp.ndarray
    fraction_potential_growth_column: jnp.ndarray
    fraction_potential_immobilization_column_level: jnp.ndarray


class SoilBiogeochemCompetitionNativeData(NamedTuple):

    topology: SubgridTopology
    soil_column: jnp.ndarray
    persistent: SoilBiogeochemCompetitionPersistentState
    parameters: SoilBiogeochemCompetitionParameters
    restart_path: Path
    parameter_data_path: Path


class SoilBiogeochemCompetitionCall(NamedTuple):

    inputs: SoilBiogeochemCompetitionInputs
    state: SoilBiogeochemCompetitionState
    cn_fluxes: CnTransientFluxState
    parameters: SoilBiogeochemCompetitionParameters


def _selected_parameter_path(
    case: CimeCase,
    override: str | Path | None,
) -> Path:
    if override is None:
        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
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
        raise FileNotFoundError(f"configured CLM parameter file does not exist: {path}")
    return path


def _finite_scalar(name: str, value: jnp.ndarray) -> np.float64:
    host = np.asarray(value, dtype=np.float64)
    if host.size != 1 or not np.isfinite(host.reshape(())):
        raise ValueError(f"CLM parameter {name!r} must contain one finite scalar")
    return np.float64(host.reshape(()))


def load_soil_biogeochem_competition_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> SoilBiogeochemCompetitionNativeData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    for key, expected in {
        "use_cn": True,
        "use_fates": False,
        "use_flexiblecn": True,
        "use_fun": True,
        "use_nitrif_denitrif": True,
        "use_vertsoilc": True,
    }.items():
        if general.get(key) is not expected:
            raise ValueError(
                f"native soil-competition adapter requires {key}={expected!r}"
            )
    if general.get("soil_layerstruct") != "20SL_8.5m":
        raise ValueError(
            "native soil-competition adapter requires soil_layerstruct='20SL_8.5m'"
        )
    if str(general.get("suplnitro", "NONE")).upper() != "NONE":
        raise ValueError("native soil-competition adapter requires suplnitro='NONE'")

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    dimensions = load_dimension_sizes(restart)
    if dimensions.get("levgrnd") != NUMBER_OF_DECOMPOSITION_LEVELS:
        raise ValueError("restart must contain the native 25 ground levels")

    topology = load_subgrid_topology(restart)
    column_count = topology.column.gridcell.shape[0]
    active = np.asarray(topology.column.active, dtype=bool)
    landunit_type = np.asarray(topology.column.landunit_type, dtype=np.int32)
    soil_column = active & (
        (landunit_type == LANDUNIT_SOIL) | (landunit_type == LANDUNIT_CROP)
    )

    values = load_native_variables(
        restart,
        list(_RESTART_VARIABLES),
        dtype=np.float64,
        masked_floating_fill_value=float(CLM_SPECIAL_VALUE),
    )
    level_values: dict[str, jnp.ndarray] = {}
    for name in _RESTART_VARIABLES[:-1]:
        value = np.asarray(values[name], dtype=np.float64)
        if value.shape != (column_count, NUMBER_OF_DECOMPOSITION_LEVELS):
            raise ValueError(f"restart {name} must have native shape (column, 25)")
        if not np.all(np.isfinite(value[soil_column, :NUMBER_OF_SOIL_LEVELS])):
            raise ValueError(f"active soil columns contain non-finite {name}")
        level_values[name] = jnp.asarray(value)
    fpg = np.asarray(values["fpg"], dtype=np.float64)
    if fpg.shape != (column_count,):
        raise ValueError("restart fpg must contain every native column")
    if not np.all(np.isfinite(fpg[soil_column])):
        raise ValueError("active soil columns contain non-finite fpg")

    parameter_data = _selected_parameter_path(case, parameter_data_path)
    raw_parameters = load_native_variables(
        parameter_data,
        list(_PARAMETER_VARIABLES),
        dtype=np.float64,
    )
    parameter = {
        name: _finite_scalar(name, raw_parameters[name])
        for name in _PARAMETER_VARIABLES
    }
    _, _, layer_thickness = soil_layer_geometry_20sl_8p5m()
    clock = load_restart_clock(restart)
    parameters = SoilBiogeochemCompetitionParameters(
        soil_layer_thickness=jnp.asarray(layer_thickness, dtype=jnp.float64),
        plant_competitiveness_nitrate=parameter["compet_plant_no3"],
        plant_competitiveness_ammonium=parameter["compet_plant_nh4"],
        decomposer_competitiveness_nitrate=parameter["compet_decomp_no3"],
        decomposer_competitiveness_ammonium=parameter["compet_decomp_nh4"],
        denitrifier_competitiveness=parameter["compet_denit"],
        nitrifier_competitiveness=parameter["compet_nit"],
        nitrification_n2o_loss_fraction=np.float64(6.0e-4),
        time_step_seconds=np.float64(clock.step_seconds),
        decomposition_level_count=NUMBER_OF_SOIL_LEVELS,
        carbon_only=False,
    )
    return SoilBiogeochemCompetitionNativeData(
        topology=topology,
        soil_column=jnp.asarray(soil_column),
        persistent=SoilBiogeochemCompetitionPersistentState(
            mineral_nitrogen_column_level=level_values["sminn_vr"],
            mineral_ammonium_column_level=level_values["smin_nh4_vr"],
            mineral_nitrate_column_level=level_values["smin_no3_vr"],
            fraction_potential_growth_column=jnp.asarray(fpg),
            fraction_potential_immobilization_column_level=level_values["fpi_vr"],
        ),
        parameters=parameters,
        restart_path=restart,
        parameter_data_path=parameter_data,
    )


def assemble_soil_biogeochem_competition_call(
    native: SoilBiogeochemCompetitionNativeData,
    cn_fluxes: CnTransientFluxState,
    plant_nitrogen_demand_column: jnp.ndarray,
    nitrogen_fixation_profile_column_level: jnp.ndarray,
    *,
    persistent: SoilBiogeochemCompetitionPersistentState | None = None,
    state: SoilBiogeochemCompetitionState | None = None,
) -> SoilBiogeochemCompetitionCall:

    column_count = native.topology.column.gridcell.shape[0]
    scalar_shape = (column_count,)
    level_shape = (column_count, NUMBER_OF_DECOMPOSITION_LEVELS)
    if cn_fluxes.column_scalar.shape[0] != column_count:
        raise ValueError("cn_fluxes must contain every native column")
    plant_demand = jnp.asarray(plant_nitrogen_demand_column, dtype=jnp.float64)
    fixation_profile = jnp.asarray(
        nitrogen_fixation_profile_column_level, dtype=jnp.float64
    )
    if plant_demand.shape != scalar_shape:
        raise ValueError("plant_nitrogen_demand_column must contain every native column")
    if fixation_profile.shape != level_shape:
        raise ValueError(
            "nitrogen_fixation_profile_column_level must have shape (column, 25)"
        )

    current = native.persistent if persistent is None else persistent
    for name in SoilBiogeochemCompetitionPersistentState._fields[:3]:
        if jnp.shape(getattr(current, name)) != level_shape:
            raise ValueError(f"persistent {name} must have shape (column, 25)")
    if jnp.shape(current.fraction_potential_growth_column) != scalar_shape:
        raise ValueError("persistent fpg must contain every native column")
    if (
        jnp.shape(current.fraction_potential_immobilization_column_level)
        != level_shape
    ):
        raise ValueError("persistent fpi_vr must have shape (column, 25)")

    competition_state = state
    if competition_state is None:
        competition_state = SoilBiogeochemCompetitionState(
            fraction_potential_growth_column=(
                current.fraction_potential_growth_column
            ),
            fraction_potential_immobilization_column=jnp.full(
                scalar_shape,
                jnp.float64(CLM_SPECIAL_VALUE),
                dtype=jnp.float64,
            ),
            fraction_potential_immobilization_column_level=(
                current.fraction_potential_immobilization_column_level
            ),
        )
    return SoilBiogeochemCompetitionCall(
        inputs=SoilBiogeochemCompetitionInputs(
            soil_column=native.soil_column,
            plant_nitrogen_demand_column=plant_demand,
            nitrogen_fixation_profile_column_level=fixation_profile,
            mineral_nitrogen_column_level=current.mineral_nitrogen_column_level,
            mineral_ammonium_column_level=current.mineral_ammonium_column_level,
            mineral_nitrate_column_level=current.mineral_nitrate_column_level,
            potential_nitrification_column_level=get_cn_flux_field(
                cn_fluxes, "soil_nitrogen.pot_f_nit_vr_col"
            ),
            potential_denitrification_column_level=get_cn_flux_field(
                cn_fluxes, "soil_nitrogen.pot_f_denit_vr_col"
            ),
            potential_immobilization_column_level=get_cn_flux_field(
                cn_fluxes, "soil_nitrogen.potential_immob_vr_col"
            ),
            n2_to_n2o_ratio_denitrification_column_level=get_cn_flux_field(
                cn_fluxes, "soil_nitrogen.n2_n2o_ratio_denit_vr_col"
            ),
        ),
        state=competition_state,
        cn_fluxes=cn_fluxes,
        parameters=native.parameters,
    )


__all__ = [
    "SoilBiogeochemCompetitionCall",
    "SoilBiogeochemCompetitionNativeData",
    "SoilBiogeochemCompetitionPersistentState",
    "assemble_soil_biogeochem_competition_call",
    "load_soil_biogeochem_competition_native_data",
]
