
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
import math

import jax.numpy as jnp
import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import (
    SubgridTopology,
    configured_land_restart,
    load_subgrid_topology,
)
from clm_jax.physics.plant_hydraulics import (
    NUMBER_OF_VEGETATION_SEGMENTS,
    PlantHydraulicParameters,
    PlantHydraulicPatchState,
)

_PARAMETER_VARIABLES = (
    "krmax",
    "kmax",
    "psi50",
    "ck",
    "psi_soil_ref",
    "smpso",
    "smpsc",
    "medlynslope",
    "medlynintercept",
    "froot_leaf",
    "rootprof_beta",
)

_ROOT_RADIUS = np.float64(0.29e-3)
_ROOT_DENSITY = np.float64(0.31e6)


def initialize_jackson_root_fraction(beta, interface_depth, bedrock_levels):
    beta = np.asarray(beta, dtype=np.float64)
    interface = np.asarray(interface_depth, dtype=np.float64)
    bedrock = np.asarray(bedrock_levels, dtype=np.int32)
    levels = interface.size - 1
    if beta.ndim != 1 or bedrock.shape != beta.shape or interface.ndim != 1:
        raise ValueError("root profiles require patch beta/bedrock and shared interfaces")
    if levels < 1 or np.any((bedrock < 1) | (bedrock > levels)):
        raise ValueError("root bedrock levels must lie within the hydrologic soil")
    unique_beta, inverse = np.unique(beta, return_inverse=True)
    exponent = interface * np.float64(100.0)
    cumulative = np.asarray([
        [math.pow(float(value), float(depth)) for depth in exponent]
        for value in unique_beta
    ], dtype=np.float64)
    raw = cumulative[:, :-1] - cumulative[:, 1:]
    profiles = np.zeros((unique_beta.size, levels, levels), dtype=np.float64)
    for depth in range(1, levels + 1):
        below = np.zeros(unique_beta.size, dtype=np.float64)
        for level in range(depth, levels):
            below += raw[:, level]
        profiles[:, depth - 1, :depth] = raw[:, :depth] + (below / depth)[:, None]
    return profiles[inverse, bedrock - 1]


class PlantHydraulicRestartData(NamedTuple):

    topology: SubgridTopology
    state: PlantHydraulicPatchState


def _configured_parameter_path(case: CimeCase) -> Path:
    group = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    declared = group.get("paramfile")
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(f"clm_inparm in {case.case_docs / 'lnd_in'} lacks paramfile")
    path = Path(case.expand(declared)).expanduser()
    if not path.is_absolute():
        path = case.root / path
    return path.resolve()


def load_plant_hydraulic_parameters(
    case: CimeCase,
    parameter_data_path: str | Path | None = None,
) -> PlantHydraulicParameters:

    path = (
        _configured_parameter_path(case)
        if parameter_data_path is None
        else Path(parameter_data_path).expanduser().resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM parameter file does not exist: {path}")
    loaded = load_native_variables(path, list(_PARAMETER_VARIABLES), dtype=np.float64)

    one_dimensional = (
        loaded["krmax"],
        loaded["psi_soil_ref"],
        loaded["smpso"],
        loaded["smpsc"],
        loaded["medlynslope"],
        loaded["medlynintercept"],
        loaded["froot_leaf"],
    )
    if any(value.ndim != 1 for value in one_dimensional):
        raise ValueError("CLM hydraulic and Medlyn PFT parameters must be one-dimensional")
    number_of_pfts = one_dimensional[0].shape[0]
    if any(value.shape != (number_of_pfts,) for value in one_dimensional):
        raise ValueError("CLM hydraulic and Medlyn PFT dimensions disagree")

    segmented = (loaded["kmax"], loaded["psi50"], loaded["ck"])
    expected_segment_shape = (NUMBER_OF_VEGETATION_SEGMENTS, number_of_pfts)
    if any(value.shape != expected_segment_shape for value in segmented):
        raise ValueError(
            "CLM kmax/psi50/ck must have NetCDF dimensions (segment=4, pft)"
        )
    if any(not np.all(np.isfinite(np.asarray(value))) for value in (*one_dimensional, *segmented)):
        raise ValueError("CLM hydraulic and Medlyn parameters must be finite")
    root_profile = np.asarray(loaded["rootprof_beta"], dtype=np.float64)
    if root_profile.ndim != 2 or root_profile.shape[1] != number_of_pfts:
        raise ValueError("CLM rootprof_beta must have dimensions (variant, pft)")
    if not np.all(np.isfinite(root_profile)):
        raise ValueError("CLM rootprof_beta must be finite")

    return PlantHydraulicParameters(
        maximum_root_conductance_pft=jnp.asarray(loaded["krmax"]),
        maximum_conductance_pft_segment=jnp.asarray(loaded["kmax"].T),
        water_potential_at_half_conductance_pft_segment=jnp.asarray(
            loaded["psi50"].T
        ),
        vulnerability_shape_pft_segment=jnp.asarray(loaded["ck"].T),
        reference_soil_water_potential_pft=jnp.asarray(loaded["psi_soil_ref"]),
        soil_water_potential_full_opening_pft=jnp.asarray(loaded["smpso"]),
        soil_water_potential_full_closure_pft=jnp.asarray(loaded["smpsc"]),
        medlyn_slope_pft=jnp.asarray(loaded["medlynslope"]),
        medlyn_intercept_pft=jnp.asarray(loaded["medlynintercept"]),
        fine_root_to_leaf_ratio_pft=jnp.asarray(loaded["froot_leaf"]),
        root_profile_beta_pft_variant=jnp.asarray(root_profile.T),
        root_radius_pft=jnp.full((number_of_pfts,), _ROOT_RADIUS),
        root_density_pft=jnp.full((number_of_pfts,), _ROOT_DENSITY),
    )


def load_plant_hydraulic_restart(
    case: CimeCase,
    restart_path: str | Path | None = None,
) -> PlantHydraulicRestartData:

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")

    topology = load_subgrid_topology(restart)
    loaded = load_native_variables(restart, ["vegwp"], dtype=np.float64)
    water_potential = jnp.asarray(loaded["vegwp"], dtype=jnp.float64)
    patch_count = topology.patch.gridcell.shape[0]
    expected_shape = (patch_count, NUMBER_OF_VEGETATION_SEGMENTS)
    if water_potential.shape != expected_shape:
        raise ValueError(
            f"restart vegwp must have native shape {expected_shape}, "
            f"got {water_potential.shape}"
        )
    if not np.all(np.isfinite(np.asarray(water_potential))):
        raise ValueError("restart vegwp contains missing or non-finite state")

    return PlantHydraulicRestartData(
        topology=topology,
        state=PlantHydraulicPatchState(
            water_potential_segment=water_potential,
            vegetation_transpiration=jnp.zeros((patch_count,), dtype=jnp.float64),
        ),
    )
