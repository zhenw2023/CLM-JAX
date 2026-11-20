
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.canopy_temperature_restart import (
    soil_layer_geometry_20sl_8p5m,
)
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)
from clm_jax.physics.soil_biogeochem_n_leaching import (
    SoilBiogeochemNLeachingFluxState,
    SoilBiogeochemNLeachingInputs,
    SoilBiogeochemNLeachingParameters,
)

NUMBER_OF_GROUND_LEVELS = 25
NUMBER_OF_DECOMPOSITION_LEVELS = 20


class SoilBiogeochemNLeachingPersistentState(NamedTuple):

    liquid_water_column_level: jnp.ndarray
    nitrate_column_level: jnp.ndarray


class SoilBiogeochemNLeachingNativeData(NamedTuple):

    soil_column: jnp.ndarray
    layer_thickness_column_level: jnp.ndarray
    soil_interface_depth: jnp.ndarray
    persistent: SoilBiogeochemNLeachingPersistentState
    parameters: SoilBiogeochemNLeachingParameters
    restart_path: Path
    parameter_data_path: Path


class SoilBiogeochemNLeachingCall(NamedTuple):

    inputs: SoilBiogeochemNLeachingInputs
    state: SoilBiogeochemNLeachingFluxState
    parameters: SoilBiogeochemNLeachingParameters


def _configured_input_path(
    case: CimeCase,
    key: str,
    override: str | Path | None,
) -> Path:
    if override is None:
        group = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        declared = group.get(key)
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError(f"configured lnd_in lacks {key}")
        path = Path(case.expand(declared)).expanduser()
        if not path.is_absolute():
            path = case.root / path
        path = path.resolve()
    else:
        path = Path(override).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM {key} file does not exist: {path}")
    return path


def load_soil_biogeochem_n_leaching_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> SoilBiogeochemNLeachingNativeData:

    general = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    required = {
        "use_cn": True,
        "use_nitrif_denitrif": True,
        "use_vertsoilc": True,
        "soil_layerstruct": "20SL_8.5m",
    }
    for name, expected in required.items():
        if general.get(name) != expected:
            raise ValueError(
                f"native nitrate-leaching adapter requires {name}={expected!r}"
            )

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    dimensions = load_dimension_sizes(restart)
    if dimensions.get("levgrnd") != NUMBER_OF_GROUND_LEVELS:
        raise ValueError("nitrate-leaching restart must contain 25 ground levels")

    topology = load_subgrid_topology(restart)
    column_count = topology.column.active.shape[0]
    soil_column = np.asarray(topology.column.active, dtype=bool) & np.isin(
        np.asarray(topology.column.landunit_type, dtype=np.int32),
        (LANDUNIT_SOIL, LANDUNIT_CROP),
    )
    raw = load_native_variables(
        restart,
        ["H2OSOI_LIQ", "smin_no3_vr"],
        dtype=np.float64,
    )
    total_level_count = dimensions.get("levtot")
    liquid = np.asarray(raw["H2OSOI_LIQ"], dtype=np.float64)
    if not isinstance(total_level_count, int) or liquid.shape != (
        column_count,
        total_level_count,
    ):
        raise ValueError("restart H2OSOI_LIQ has unexpected dimensions")
    liquid = liquid[:, -NUMBER_OF_GROUND_LEVELS:]
    nitrate = np.asarray(raw["smin_no3_vr"], dtype=np.float64)
    if nitrate.shape != (column_count, NUMBER_OF_GROUND_LEVELS):
        raise ValueError("restart smin_no3_vr must have shape (column, 25)")
    if not np.all(np.isfinite(liquid[soil_column])):
        raise ValueError("active soil columns contain missing H2OSOI_LIQ")
    if not np.all(np.isfinite(nitrate[soil_column])):
        raise ValueError("active soil columns contain missing smin_no3_vr")

    parameter_data = _configured_input_path(case, "paramfile", parameter_data_path)
    parameter_values = load_native_variables(
        parameter_data,
        ["sf_no3"],
        dtype=np.float64,
    )
    soluble_fraction = np.asarray(parameter_values["sf_no3"], dtype=np.float64)
    if soluble_fraction.size != 1 or not np.isfinite(soluble_fraction.item()):
        raise ValueError("CLM parameter sf_no3 must be one finite scalar")

    _, interface, thickness = soil_layer_geometry_20sl_8p5m()
    layer_thickness = np.broadcast_to(
        thickness[None, :],
        (column_count, NUMBER_OF_GROUND_LEVELS),
    ).copy()
    clock = load_restart_clock(restart)
    if clock.step_seconds <= 0:
        raise ValueError("restart radiation time step must be positive")
    return SoilBiogeochemNLeachingNativeData(
        soil_column=jnp.asarray(soil_column),
        layer_thickness_column_level=jnp.asarray(layer_thickness),
        soil_interface_depth=jnp.asarray(interface),
        persistent=SoilBiogeochemNLeachingPersistentState(
            liquid_water_column_level=jnp.asarray(liquid),
            nitrate_column_level=jnp.asarray(nitrate),
        ),
        parameters=SoilBiogeochemNLeachingParameters(
            time_step=np.float64(clock.step_seconds),
            soluble_nitrate_fraction=np.float64(soluble_fraction.item()),
            soil_level_count=NUMBER_OF_DECOMPOSITION_LEVELS,
            decomposition_level_count=NUMBER_OF_DECOMPOSITION_LEVELS,
        ),
        restart_path=restart,
        parameter_data_path=parameter_data,
    )


def assemble_soil_biogeochem_n_leaching_call(
    native: SoilBiogeochemNLeachingNativeData,
    drainage_flux_column: jnp.ndarray,
    surface_runoff_flux_column: jnp.ndarray,
    *,
    persistent: SoilBiogeochemNLeachingPersistentState | None = None,
    flux_state: SoilBiogeochemNLeachingFluxState | None = None,
) -> SoilBiogeochemNLeachingCall:

    column_count = native.soil_column.shape[0]
    level_shape = (column_count, NUMBER_OF_GROUND_LEVELS)
    vector_shape = (column_count,)
    current = native.persistent if persistent is None else persistent
    if current.liquid_water_column_level.shape != level_shape:
        raise ValueError("current liquid water has an incompatible shape")
    if current.nitrate_column_level.shape != level_shape:
        raise ValueError("current nitrate has an incompatible shape")
    drainage = jnp.asarray(drainage_flux_column, dtype=jnp.float64)
    runoff = jnp.asarray(surface_runoff_flux_column, dtype=jnp.float64)
    if drainage.shape != vector_shape or runoff.shape != vector_shape:
        raise ValueError("drainage and surface runoff must contain every column")
    if flux_state is None:
        zeros = jnp.zeros(level_shape, dtype=jnp.float64)
        flux_state = SoilBiogeochemNLeachingFluxState(zeros, zeros)
    elif any(value.shape != level_shape for value in flux_state):
        raise ValueError("nitrate-loss flux state has an incompatible shape")
    return SoilBiogeochemNLeachingCall(
        inputs=SoilBiogeochemNLeachingInputs(
            soil_column=native.soil_column,
            liquid_water_column_level=current.liquid_water_column_level,
            drainage_flux_column=drainage,
            surface_runoff_flux_column=runoff,
            nitrate_column_level=current.nitrate_column_level,
            layer_thickness_column_level=native.layer_thickness_column_level,
            soil_interface_depth=native.soil_interface_depth,
        ),
        state=flux_state,
        parameters=native.parameters,
    )


__all__ = [
    "SoilBiogeochemNLeachingCall",
    "SoilBiogeochemNLeachingNativeData",
    "SoilBiogeochemNLeachingPersistentState",
    "assemble_soil_biogeochem_n_leaching_call",
    "load_soil_biogeochem_n_leaching_native_data",
]
