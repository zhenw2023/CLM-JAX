
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import DENSITY_FRESH_WATER, DENSITY_ICE
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
    SubgridTopology,
    configured_land_restart,
    load_subgrid_topology,
)
from clm_jax.physics.soil_biogeochem_nitrif_denitrif import (
    NUMBER_OF_GROUND_LEVELS,
    NUMBER_OF_SOIL_LEVELS,
    SoilBiogeochemNitrifDenitrifInputs,
    SoilBiogeochemNitrifDenitrifParameters,
)

if TYPE_CHECKING:
    from clm_jax.physics.methane_driver_facade import MethaneDriverActiveState

_PARAMETER_VARIABLES = (
    "surface_tension_water",
    "rij_kro_a",
    "rij_kro_alpha",
    "rij_kro_beta",
    "rij_kro_gamma",
    "rij_kro_delta",
    "organic_max",
)
_RESTART_VARIABLES = (
    "H2OSOI_LIQ",
    "H2OSOI_ICE",
    "T_SOISNO",
    "O2_DECOMP_DEPTH_UNSAT",
    "CONC_O2_UNSAT",
    "O2_DECOMP_DEPTH_SAT",
    "CONC_O2_SAT",
    "FINUNDATED",
    "smin_nh4_vr",
    "smin_no3_vr",
)


class SoilBiogeochemNitrifDenitrifPersistentState(NamedTuple):

    volumetric_water_column_level: jnp.ndarray
    liquid_water_mass_column_level: jnp.ndarray
    soil_temperature_column_level: jnp.ndarray
    oxygen_decomposition_unsaturated_column_level: jnp.ndarray
    oxygen_concentration_unsaturated_column_level: jnp.ndarray
    oxygen_decomposition_saturated_column_level: jnp.ndarray
    oxygen_concentration_saturated_column_level: jnp.ndarray
    inundated_fraction_column: jnp.ndarray
    mineral_ammonium_column_level: jnp.ndarray
    mineral_nitrate_column_level: jnp.ndarray


def nitrification_feedback_from_methane(
    restart: SoilBiogeochemNitrifDenitrifPersistentState,
    methane: MethaneDriverActiveState | None,
) -> SoilBiogeochemNitrifDenitrifPersistentState:
    if methane is None:
        return restart
    transport = methane.transport
    competition = transport.competition
    return restart._replace(
        oxygen_decomposition_unsaturated_column_level=(
            competition.unsaturated_oxygen_decomposition_column_level
        ),
        oxygen_concentration_unsaturated_column_level=(
            transport.unsaturated_oxygen_concentration_column_level
        ),
        oxygen_decomposition_saturated_column_level=(
            competition.saturated_oxygen_decomposition_column_level
        ),
        oxygen_concentration_saturated_column_level=(
            transport.saturated_oxygen_concentration_column_level
        ),
        inundated_fraction_column=methane.preprocess.inundated_fraction_column,
    )


class SoilBiogeochemNitrifDenitrifNativeData(NamedTuple):

    topology: SubgridTopology
    soil_column: jnp.ndarray
    saturated_water_content_column_level: jnp.ndarray
    field_capacity_water_content_column_level: jnp.ndarray
    dry_soil_bulk_density_column_level: jnp.ndarray
    clapp_hornberger_b_column_level: jnp.ndarray
    organic_matter_density_column_level: jnp.ndarray
    saturated_suction_column_level: jnp.ndarray
    soil_layer_thickness_column_level: jnp.ndarray
    persistent: SoilBiogeochemNitrifDenitrifPersistentState
    parameters: SoilBiogeochemNitrifDenitrifParameters
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
        raise FileNotFoundError(f"configured CLM {key} file does not exist: {path}")
    return path


def _finite_scalar(name: str, value: jnp.ndarray) -> np.float64:
    host = np.asarray(value, dtype=np.float64)
    if host.size != 1 or not np.isfinite(host.reshape(())):
        raise ValueError(f"CLM parameter {name!r} must contain one finite scalar")
    return np.float64(host.reshape(()))


def _namelist_float(
    values: dict[str, object],
    name: str,
    default: float,
) -> np.float64:
    value = values.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"nitrif_inparm {name} must be numeric")
    result = np.float64(value)
    if not np.isfinite(result):
        raise ValueError(f"nitrif_inparm {name} must be finite")
    return result


def load_soil_biogeochem_nitrif_denitrif_parameters(
    case: CimeCase,
    parameter_data_path: str | Path | None = None,
) -> SoilBiogeochemNitrifDenitrifParameters:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    if general.get("use_nitrif_denitrif") is not True:
        raise ValueError("native adapter requires use_nitrif_denitrif=.true.")
    if general.get("soil_layerstruct") != "20SL_8.5m":
        raise ValueError("native adapter requires soil_layerstruct='20SL_8.5m'")
    parameter_data = _selected_input_path(case, "paramfile", parameter_data_path)
    raw = load_native_variables(
        parameter_data,
        list(_PARAMETER_VARIABLES),
        dtype=np.float64,
    )
    parameter = {
        name: _finite_scalar(name, raw[name]) for name in _PARAMETER_VARIABLES
    }
    nitrif = read_case_namelist_group(case, "lnd_in", "nitrif_inparm")
    maximum_per_day = _namelist_float(nitrif, "k_nitr_max_perday", 0.1)
    return SoilBiogeochemNitrifDenitrifParameters(
        decomposition_level_count=(
            NUMBER_OF_SOIL_LEVELS if general.get("use_vertsoilc") is True else 1
        ),
        use_methane=bool(general.get("use_lch4", False)),
        anoxia_uses_saturated_fraction=bool(general.get("anoxia_wtsat", False)),
        suppress_frozen_nitrification_denitrification=False,
        maximum_nitrification_rate_per_second=(
            maximum_per_day / np.float64(86_400.0)
        ),
        surface_tension_water=parameter["surface_tension_water"],
        rijtema_kroes_a=parameter["rij_kro_a"],
        rijtema_kroes_alpha=parameter["rij_kro_alpha"],
        rijtema_kroes_beta=parameter["rij_kro_beta"],
        rijtema_kroes_gamma=parameter["rij_kro_gamma"],
        rijtema_kroes_delta=parameter["rij_kro_delta"],
        denitrification_respiration_coefficient=_namelist_float(
            nitrif,
            "denitrif_respiration_coefficient",
            0.1,
        ),
        denitrification_respiration_exponent=_namelist_float(
            nitrif,
            "denitrif_respiration_exponent",
            1.3,
        ),
        denitrification_nitrate_coefficient=_namelist_float(
            nitrif,
            "denitrif_nitrateconc_coefficient",
            1.15,
        ),
        denitrification_nitrate_exponent=_namelist_float(
            nitrif,
            "denitrif_nitrateconc_exponent",
            0.57,
        ),
        organic_matter_maximum_density=parameter["organic_max"],
    )


def load_soil_biogeochem_nitrif_denitrif_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> SoilBiogeochemNitrifDenitrifNativeData:

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    dimensions = load_dimension_sizes(restart)
    if dimensions.get("levgrnd") != NUMBER_OF_GROUND_LEVELS:
        raise ValueError("restart must contain 25 native ground levels")
    topology = load_subgrid_topology(restart)
    number_of_columns = topology.column.gridcell.shape[0]
    active = np.asarray(topology.column.active, dtype=bool)
    landunit_type = np.asarray(topology.column.landunit_type, dtype=np.int32)
    soil_column = active & (
        (landunit_type == LANDUNIT_SOIL) | (landunit_type == LANDUNIT_CROP)
    )

    surface_data = _selected_input_path(case, "fsurdat", surface_data_path)
    parameter_data = _selected_input_path(case, "paramfile", parameter_data_path)
    surface = load_native_variables(
        surface_data,
        ["PCT_SAND", "PCT_CLAY", "ORGANIC"],
        dtype=np.float64,
    )
    raw_parameters = load_native_variables(
        parameter_data,
        ["organic_max"],
        dtype=np.float64,
    )
    organic_maximum = _finite_scalar("organic_max", raw_parameters["organic_max"])
    static = column_soil_hydraulic_properties_20sl_8p5m(
        topology,
        surface,
        float(organic_maximum),
    )
    _, _, layer_thickness = soil_layer_geometry_20sl_8p5m()
    thickness = np.broadcast_to(
        layer_thickness[None, :],
        (number_of_columns, NUMBER_OF_GROUND_LEVELS),
    ).copy()

    restart_values = load_native_variables(
        restart,
        list(_RESTART_VARIABLES),
        dtype=np.float64,
    )
    total_level_count = dimensions.get("levtot")
    if not isinstance(total_level_count, int) or total_level_count < NUMBER_OF_GROUND_LEVELS:
        raise ValueError("restart levtot cannot contain all ground levels")
    ground: dict[str, np.ndarray] = {}
    for name in ("H2OSOI_LIQ", "H2OSOI_ICE", "T_SOISNO"):
        value = np.asarray(restart_values[name], dtype=np.float64)
        if value.shape != (number_of_columns, total_level_count):
            raise ValueError(f"restart {name} has unexpected dimensions")
        ground[name] = value[:, -NUMBER_OF_GROUND_LEVELS:]
    for name in (
        "O2_DECOMP_DEPTH_UNSAT",
        "CONC_O2_UNSAT",
        "O2_DECOMP_DEPTH_SAT",
        "CONC_O2_SAT",
        "smin_nh4_vr",
        "smin_no3_vr",
    ):
        value = np.asarray(restart_values[name], dtype=np.float64)
        if value.shape != (number_of_columns, NUMBER_OF_GROUND_LEVELS):
            raise ValueError(f"restart {name} must have shape (column, 25)")
        ground[name] = value
    inundated = np.asarray(restart_values["FINUNDATED"], dtype=np.float64)
    if inundated.shape != (number_of_columns,):
        raise ValueError("restart FINUNDATED must contain every column")

    volumetric_water = (
        ground["H2OSOI_LIQ"]
        / (thickness * np.float64(DENSITY_FRESH_WATER))
        + ground["H2OSOI_ICE"] / (thickness * np.float64(DENSITY_ICE))
    )
    finite_fields = {
        "H2OSOI_LIQ": ground["H2OSOI_LIQ"],
        "T_SOISNO": ground["T_SOISNO"],
        "H2OSOI_VOL": volumetric_water,
        "O2_DECOMP_DEPTH_UNSAT": ground["O2_DECOMP_DEPTH_UNSAT"],
        "CONC_O2_UNSAT": ground["CONC_O2_UNSAT"],
        "O2_DECOMP_DEPTH_SAT": ground["O2_DECOMP_DEPTH_SAT"],
        "CONC_O2_SAT": ground["CONC_O2_SAT"],
        "smin_nh4_vr": ground["smin_nh4_vr"],
        "smin_no3_vr": ground["smin_no3_vr"],
    }
    for name, value in finite_fields.items():
        if not np.all(np.isfinite(value[soil_column])):
            raise ValueError(f"filtered soil columns contain missing field {name}")
    if not np.all(np.isfinite(inundated[soil_column])):
        raise ValueError("filtered soil columns contain missing FINUNDATED")

    return SoilBiogeochemNitrifDenitrifNativeData(
        topology=topology,
        soil_column=jnp.asarray(soil_column),
        saturated_water_content_column_level=jnp.asarray(
            static.saturated_water_content_column_level
        ),
        field_capacity_water_content_column_level=jnp.asarray(
            static.field_capacity_water_content_column_level
        ),
        dry_soil_bulk_density_column_level=jnp.asarray(
            static.dry_soil_bulk_density_column_level
        ),
        clapp_hornberger_b_column_level=jnp.asarray(
            static.clapp_hornberger_b_column_level
        ),
        organic_matter_density_column_level=jnp.asarray(
            static.organic_matter_density_column_level
        ),
        saturated_suction_column_level=jnp.asarray(
            static.saturated_suction_column_level
        ),
        soil_layer_thickness_column_level=jnp.asarray(thickness),
        persistent=SoilBiogeochemNitrifDenitrifPersistentState(
            volumetric_water_column_level=jnp.asarray(volumetric_water),
            liquid_water_mass_column_level=jnp.asarray(ground["H2OSOI_LIQ"]),
            soil_temperature_column_level=jnp.asarray(ground["T_SOISNO"]),
            oxygen_decomposition_unsaturated_column_level=jnp.asarray(
                ground["O2_DECOMP_DEPTH_UNSAT"]
            ),
            oxygen_concentration_unsaturated_column_level=jnp.asarray(
                ground["CONC_O2_UNSAT"]
            ),
            oxygen_decomposition_saturated_column_level=jnp.asarray(
                ground["O2_DECOMP_DEPTH_SAT"]
            ),
            oxygen_concentration_saturated_column_level=jnp.asarray(
                ground["CONC_O2_SAT"]
            ),
            inundated_fraction_column=jnp.asarray(inundated),
            mineral_ammonium_column_level=jnp.asarray(ground["smin_nh4_vr"]),
            mineral_nitrate_column_level=jnp.asarray(ground["smin_no3_vr"]),
        ),
        parameters=load_soil_biogeochem_nitrif_denitrif_parameters(
            case,
            parameter_data_path=parameter_data,
        ),
        restart_path=restart,
        surface_data_path=surface_data,
        parameter_data_path=parameter_data,
    )


def assemble_soil_biogeochem_nitrif_denitrif_inputs(
    native: SoilBiogeochemNitrifDenitrifNativeData,
    soil_water_potential_column_level: jnp.ndarray,
    potential_heterotrophic_respiration_column_level: jnp.ndarray,
    decomposition_water_scalar_column_level: jnp.ndarray,
    decomposition_temperature_scalar_column_level: jnp.ndarray,
    *,
    persistent: SoilBiogeochemNitrifDenitrifPersistentState | None = None,
) -> SoilBiogeochemNitrifDenitrifInputs:

    number_of_columns = native.soil_column.shape[0]
    expected = (number_of_columns, NUMBER_OF_GROUND_LEVELS)

    def current_array(name: str, value: jnp.ndarray) -> jnp.ndarray:
        result = jnp.asarray(value, dtype=jnp.float64)
        if result.shape != expected:
            raise ValueError(f"{name} must have shape (column, 25)")
        return result

    current = native.persistent if persistent is None else persistent
    for name in SoilBiogeochemNitrifDenitrifPersistentState._fields:
        expected_shape = (
            (number_of_columns,)
            if name == "inundated_fraction_column"
            else expected
        )
        if jnp.shape(getattr(current, name)) != expected_shape:
            raise ValueError(f"{name} has an incompatible shape")
    return SoilBiogeochemNitrifDenitrifInputs(
        soil_column=native.soil_column,
        saturated_water_content_column_level=(
            native.saturated_water_content_column_level
        ),
        field_capacity_water_content_column_level=(
            native.field_capacity_water_content_column_level
        ),
        dry_soil_bulk_density_column_level=(
            native.dry_soil_bulk_density_column_level
        ),
        clapp_hornberger_b_column_level=native.clapp_hornberger_b_column_level,
        organic_matter_density_column_level=(
            native.organic_matter_density_column_level
        ),
        saturated_suction_column_level=native.saturated_suction_column_level,
        soil_water_potential_column_level=current_array(
            "soil_water_potential_column_level",
            soil_water_potential_column_level,
        ),
        volumetric_water_column_level=(
            current.volumetric_water_column_level
        ),
        liquid_water_mass_column_level=current.liquid_water_mass_column_level,
        soil_temperature_column_level=current.soil_temperature_column_level,
        oxygen_decomposition_unsaturated_column_level=(
            current.oxygen_decomposition_unsaturated_column_level
        ),
        oxygen_concentration_unsaturated_column_level=(
            current.oxygen_concentration_unsaturated_column_level
        ),
        oxygen_decomposition_saturated_column_level=(
            current.oxygen_decomposition_saturated_column_level
        ),
        oxygen_concentration_saturated_column_level=(
            current.oxygen_concentration_saturated_column_level
        ),
        inundated_fraction_column=current.inundated_fraction_column,
        mineral_ammonium_column_level=current.mineral_ammonium_column_level,
        mineral_nitrate_column_level=current.mineral_nitrate_column_level,
        potential_heterotrophic_respiration_column_level=current_array(
            "potential_heterotrophic_respiration_column_level",
            potential_heterotrophic_respiration_column_level,
        ),
        decomposition_water_scalar_column_level=current_array(
            "decomposition_water_scalar_column_level",
            decomposition_water_scalar_column_level,
        ),
        decomposition_temperature_scalar_column_level=current_array(
            "decomposition_temperature_scalar_column_level",
            decomposition_temperature_scalar_column_level,
        ),
        soil_layer_thickness_column_level=native.soil_layer_thickness_column_level,
    )


__all__ = [
    "SoilBiogeochemNitrifDenitrifNativeData",
    "SoilBiogeochemNitrifDenitrifPersistentState",
    "assemble_soil_biogeochem_nitrif_denitrif_inputs",
    "load_soil_biogeochem_nitrif_denitrif_native_data",
    "load_soil_biogeochem_nitrif_denitrif_parameters",
    "nitrification_feedback_from_methane",
]
