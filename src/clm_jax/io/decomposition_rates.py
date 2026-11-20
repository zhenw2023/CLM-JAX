
from __future__ import annotations

import calendar as calendar_module
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.canopy_temperature_restart import soil_layer_geometry_20sl_8p5m
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    SubgridTopology,
    advance_restart_clock,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)
from clm_jax.physics.decomposition_rates import (
    NUMBER_OF_GROUND_LEVELS,
    NUMBER_OF_SOIL_LEVELS,
    DecompositionRateInputs,
    DecompositionRateParameters,
)

if TYPE_CHECKING:
    from clm_jax.physics.methane_driver_facade import MethaneDriverActiveState

_PARAMETER_VARIABLES = (
    "minpsi_hr",
    "maxpsi_hr",
    "q10_mr",
    "froz_q10",
    "mino2lim",
    "tau_cwd",
    "tau_s2",
    "tau_s3",
)


class DecompositionRatePersistentState(NamedTuple):

    soil_temperature_column_level: jnp.ndarray
    oxygen_stress_saturated_column_level: jnp.ndarray
    oxygen_stress_unsaturated_column_level: jnp.ndarray
    inundated_fraction_column: jnp.ndarray


def decomposition_feedback_from_methane(
    restart: DecompositionRatePersistentState,
    methane: MethaneDriverActiveState | None,
) -> DecompositionRatePersistentState:
    if methane is None:
        return restart
    competition = methane.transport.competition
    return restart._replace(
        oxygen_stress_saturated_column_level=(
            competition.saturated_oxygen_stress_column_level
        ),
        oxygen_stress_unsaturated_column_level=(
            competition.unsaturated_oxygen_stress_column_level
        ),
        inundated_fraction_column=methane.preprocess.inundated_fraction_column,
    )


class DecompositionRateNativeData(NamedTuple):

    topology: SubgridTopology
    soil_column: jnp.ndarray
    latitude_degrees_column: jnp.ndarray
    soil_node_depth_level: jnp.ndarray
    soil_layer_thickness_column_level: jnp.ndarray
    persistent: DecompositionRatePersistentState
    parameters: DecompositionRateParameters
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


def _finite_scalar(name: str, value: jnp.ndarray) -> np.float64:
    host = np.asarray(value, dtype=np.float64)
    if host.size != 1 or not np.isfinite(host.reshape(())):
        raise ValueError(f"CLM parameter {name!r} must contain one finite scalar")
    return np.float64(host.reshape(()))


def _days_per_current_year(case: CimeCase, current_date: int) -> np.float64:
    calendar_name = case.value("CALENDAR").strip().upper()
    if calendar_name == "NO_LEAP":
        return np.float64(365.0)
    if calendar_name == "GREGORIAN":
        year = current_date // 10_000
        return np.float64(366.0 if calendar_module.isleap(year) else 365.0)
    raise ValueError(f"unsupported CLM calendar: {calendar_name!r}")


def load_decomposition_rate_parameters(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> DecompositionRateParameters:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    if general.get("use_century_decomp") is not True:
        raise ValueError("BGC decomposition-rate adapter requires use_century_decomp=.true.")
    layer_structure = general.get("soil_layerstruct")
    if layer_structure != "20SL_8.5m":
        raise ValueError(
            "decomposition-rate adapter currently requires "
            "soil_layerstruct='20SL_8.5m'"
        )

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    parameter_data = _selected_parameter_path(case, parameter_data_path)
    values = load_native_variables(
        parameter_data,
        list(_PARAMETER_VARIABLES),
        dtype=np.float64,
    )
    scalar = {name: _finite_scalar(name, values[name]) for name in _PARAMETER_VARIABLES}

    use_vertical = bool(general.get("use_vertsoilc", False))
    use_methane = bool(general.get("use_lch4", False))
    use_fates = bool(general.get("use_fates", False))
    spinup_state = int(general.get("spinup_state", 0))
    anoxia = use_methane and use_vertical
    anoxia_uses_saturated_fraction = bool(general.get("anoxia_wtsat", False))
    if anoxia_uses_saturated_fraction and not anoxia:
        raise ValueError("anoxia_wtsat requires active methane and vertical soil carbon")

    shared = read_case_namelist_group(case, "lnd_in", "bgc_shared")
    depth_efolding = shared.get("decomp_depth_efolding")
    if (
        isinstance(depth_efolding, bool)
        or not isinstance(depth_efolding, int | float)
        or not np.isfinite(depth_efolding)
        or depth_efolding <= 0.0
    ):
        raise ValueError("bgc_shared decomp_depth_efolding must be positive and finite")

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
    clock = advance_restart_clock(
        load_restart_clock(restart),
        case.value("CALENDAR"),
    )
    return DecompositionRateParameters(
        minimum_water_potential_mpa=scalar["minpsi_hr"],
        maximum_water_potential_mpa=scalar["maxpsi_hr"],
        temperature_q10=scalar["q10_mr"],
        frozen_temperature_q10=scalar["froz_q10"],
        decomposition_depth_efolding_m=np.float64(depth_efolding),
        minimum_oxygen_scalar=scalar["mino2lim"],
        days_per_year=_days_per_current_year(case, clock.current_date),
        normalization_reference_celsius=np.float64(15.0),
        decomposition_level_count=NUMBER_OF_SOIL_LEVELS if use_vertical else 1,
        use_century_temperature_function=False,
        normalize_q10_to_century=True,
        use_vertical_soil_carbon=use_vertical,
        use_methane=use_methane,
        use_anoxia=anoxia,
        anoxia_uses_saturated_fraction=anoxia_uses_saturated_fraction,
        use_fates=use_fates,
        spinup_state=spinup_state,
        spinup_factor=jnp.asarray(spinup_factor),
    )


def load_decomposition_rate_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> DecompositionRateNativeData:

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    dimensions = load_dimension_sizes(restart)
    if dimensions.get("levgrnd") != NUMBER_OF_GROUND_LEVELS:
        raise ValueError("restart must contain the native 25 ground levels")

    topology = load_subgrid_topology(restart)
    column_count = topology.column.gridcell.shape[0]
    values = load_native_variables(
        restart,
        ["T_SOISNO", "O2STRESS_SAT", "O2STRESS_UNSAT", "FINUNDATED"],
        dtype=np.float64,
    )
    temperature = np.asarray(values["T_SOISNO"], dtype=np.float64)
    if temperature.ndim != 2 or temperature.shape[0] != column_count:
        raise ValueError("restart T_SOISNO must have native (column, level) dimensions")
    if temperature.shape[1] < NUMBER_OF_GROUND_LEVELS:
        raise ValueError("restart T_SOISNO lacks the ground-level tail")
    temperature = temperature[:, -NUMBER_OF_GROUND_LEVELS:]

    saturated = np.asarray(values["O2STRESS_SAT"], dtype=np.float64)
    unsaturated = np.asarray(values["O2STRESS_UNSAT"], dtype=np.float64)
    inundated = np.asarray(values["FINUNDATED"], dtype=np.float64)
    for name, value in {
        "O2STRESS_SAT": saturated,
        "O2STRESS_UNSAT": unsaturated,
    }.items():
        if value.shape != (column_count, NUMBER_OF_GROUND_LEVELS):
            raise ValueError(f"restart {name} must have shape (column, 25)")
    if inundated.shape != (column_count,):
        raise ValueError("restart FINUNDATED must contain every column")

    active = np.asarray(topology.column.active, dtype=bool)
    landunit_type = np.asarray(topology.column.landunit_type, dtype=np.int32)
    soil_column = active & (
        (landunit_type == LANDUNIT_SOIL) | (landunit_type == LANDUNIT_CROP)
    )
    gridcell = np.asarray(topology.column.gridcell, dtype=np.int32)
    grid_count = topology.grid.latitude_degrees.shape[0]
    if np.any((gridcell < 0) | (gridcell >= grid_count)):
        raise ValueError("restart column-to-gridcell topology is out of bounds")
    latitude = np.asarray(topology.grid.latitude_degrees, dtype=np.float64)[gridcell]
    for name, value in {
        "T_SOISNO": temperature,
        "O2STRESS_SAT": saturated,
        "O2STRESS_UNSAT": unsaturated,
    }.items():
        if not np.all(np.isfinite(value[soil_column])):
            raise ValueError(f"filtered soil columns contain missing restart field {name}")
    if not np.all(np.isfinite(inundated[soil_column])):
        raise ValueError("filtered soil columns contain missing restart field FINUNDATED")
    if not np.all(np.isfinite(latitude[soil_column])):
        raise ValueError("filtered soil columns contain missing latitude")

    node_depth, _, layer_thickness = soil_layer_geometry_20sl_8p5m()
    thickness = np.broadcast_to(
        np.asarray(layer_thickness, dtype=np.float64),
        (column_count, NUMBER_OF_GROUND_LEVELS),
    ).copy()
    parameter_data = _selected_parameter_path(case, parameter_data_path)
    return DecompositionRateNativeData(
        topology=topology,
        soil_column=jnp.asarray(soil_column),
        latitude_degrees_column=jnp.asarray(latitude),
        soil_node_depth_level=jnp.asarray(node_depth, dtype=jnp.float64),
        soil_layer_thickness_column_level=jnp.asarray(thickness),
        persistent=DecompositionRatePersistentState(
            soil_temperature_column_level=jnp.asarray(temperature),
            oxygen_stress_saturated_column_level=jnp.asarray(saturated),
            oxygen_stress_unsaturated_column_level=jnp.asarray(unsaturated),
            inundated_fraction_column=jnp.asarray(inundated),
        ),
        parameters=load_decomposition_rate_parameters(
            case,
            restart_path=restart,
            parameter_data_path=parameter_data,
        ),
        restart_path=restart,
        parameter_data_path=parameter_data,
    )


def assemble_decomposition_rate_inputs(
    native: DecompositionRateNativeData,
    soil_water_potential_column_level: jnp.ndarray,
    *,
    persistent: DecompositionRatePersistentState | None = None,
    thaw_depth_index_column: jnp.ndarray | None = None,
) -> DecompositionRateInputs:

    column_count = native.soil_column.shape[0]
    level_shape = (column_count, NUMBER_OF_GROUND_LEVELS)
    water_potential = jnp.asarray(soil_water_potential_column_level, dtype=jnp.float64)
    if water_potential.shape != level_shape:
        raise ValueError("soil water potential must have shape (column, 25)")
    current = native.persistent if persistent is None else persistent
    for name in (
        "soil_temperature_column_level",
        "oxygen_stress_saturated_column_level",
        "oxygen_stress_unsaturated_column_level",
    ):
        if jnp.shape(getattr(current, name)) != level_shape:
            raise ValueError(f"{name} must have shape (column, 25)")
    if jnp.shape(current.inundated_fraction_column) != (column_count,):
        raise ValueError("inundated_fraction_column must contain every column")

    if thaw_depth_index_column is None:
        if (
            native.parameters.decomposition_level_count == 1
            and native.parameters.use_methane
            and native.parameters.anoxia_uses_saturated_fraction
        ):
            raise ValueError("single-level saturated-anoxia mode requires current thaw index")
        thaw_index = jnp.zeros((column_count,), dtype=jnp.int32)
    else:
        thaw_index = jnp.asarray(thaw_depth_index_column, dtype=jnp.int32)
        if thaw_index.shape != (column_count,):
            raise ValueError("thaw_depth_index_column must contain every column")

    return DecompositionRateInputs(
        soil_column=native.soil_column,
        soil_water_potential_column_level=water_potential,
        soil_temperature_column_level=current.soil_temperature_column_level,
        oxygen_stress_saturated_column_level=(
            current.oxygen_stress_saturated_column_level
        ),
        oxygen_stress_unsaturated_column_level=(
            current.oxygen_stress_unsaturated_column_level
        ),
        inundated_fraction_column=current.inundated_fraction_column,
        thaw_depth_index_column=thaw_index,
        soil_layer_thickness_column_level=native.soil_layer_thickness_column_level,
        soil_node_depth_level=native.soil_node_depth_level,
        latitude_degrees_column=native.latitude_degrees_column,
    )


__all__ = [
    "DecompositionRateNativeData",
    "DecompositionRatePersistentState",
    "assemble_decomposition_rate_inputs",
    "decomposition_feedback_from_methane",
    "load_decomposition_rate_native_data",
    "load_decomposition_rate_parameters",
]
