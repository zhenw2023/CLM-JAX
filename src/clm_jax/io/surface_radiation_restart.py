
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    RestartClock,
    SubgridTopology,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)
from clm_jax.physics.surface_radiation import (
    SurfaceRadiationDiagnosticsState,
    SurfaceRadiationFluxState,
    SurfaceRadiationInputs,
    SurfaceRadiationState,
)


class SurfaceRadiationRestartData(NamedTuple):
    topology: SubgridTopology
    clock: RestartClock
    inputs: SurfaceRadiationInputs
    state: SurfaceRadiationState


_RESTART_VARIABLES = (
    "SNLSNO",
    "SNOW_DEPTH",
    "fsun",
    "albd",
    "albi",
    "albgrd",
    "albgri",
    "albsod",
    "albsoi",
    "albsnd_hst",
    "albsni_hst",
    "fabd",
    "fabi",
    "ftdd",
    "ftid",
    "ftii",
    "flx_absdv",
    "flx_absdn",
    "flx_absiv",
    "flx_absin",
)


def _finite_restart_float(value: jnp.ndarray) -> jnp.ndarray:
    value = jnp.asarray(value, dtype=jnp.float64)
    return jnp.where(jnp.isnan(value), jnp.float64(CLM_SPECIAL_VALUE), value)


def _initial_state(
    sunlit_fraction_patch: jnp.ndarray,
    number_of_snow_and_soil_levels: int,
) -> SurfaceRadiationState:
    number_of_patches = sunlit_fraction_patch.shape[0]
    special_value = jnp.float64(CLM_SPECIAL_VALUE)
    scalar = jnp.full((number_of_patches,), special_value, dtype=jnp.float64)
    band = jnp.full((number_of_patches, 2), special_value, dtype=jnp.float64)
    layers = jnp.full(
        (number_of_patches, number_of_snow_and_soil_levels),
        special_value,
        dtype=jnp.float64,
    )
    return SurfaceRadiationState(
        fluxes=SurfaceRadiationFluxState(
            sunlit_fraction_patch=sunlit_fraction_patch,
            absorbed_total_patch=scalar,
            reflected_total_patch=scalar,
            rural_absorbed_total_patch=scalar,
            absorbed_vegetation_patch=scalar,
            absorbed_ground_patch=scalar,
            absorbed_ground_soil_patch=scalar,
            absorbed_ground_snow_patch=scalar,
            absorbed_ground_layers_patch=layers,
            penetrating_shortwave_patch=scalar,
            subsurface_snow_absorption_fraction_patch=scalar,
        ),
        diagnostics=SurfaceRadiationDiagnosticsState(
            incident_direct_patch=band,
            incident_diffuse_patch=band,
            reflected_direct_patch=band,
            reflected_diffuse_patch=band,
            local_noon_incident_direct_patch=band,
            local_noon_incident_visible_diffuse_patch=scalar,
            local_noon_reflected_direct_patch=band,
            local_noon_absorbed_par_patch=scalar,
            snow_incident_direct_patch=band,
            snow_incident_diffuse_patch=band,
            snow_reflected_direct_patch=band,
            snow_reflected_diffuse_patch=band,
        ),
    )


def load_surface_radiation_restart(
    case: CimeCase,
    direct_shortwave_gridcell: jnp.ndarray,
    diffuse_shortwave_gridcell: jnp.ndarray,
    restart_path: str | Path | None = None,
) -> SurfaceRadiationRestartData:

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")

    dimensions = load_dimension_sizes(restart)
    required_dimensions = ("gridcell", "column", "pft", "numrad", "levsno1")
    missing_dimensions = [name for name in required_dimensions if name not in dimensions]
    if missing_dimensions:
        raise ValueError(f"restart lacks dimensions {missing_dimensions}: {restart}")
    if dimensions["numrad"] != 2:
        raise ValueError("SurfaceRadiation requires the two native CLM solar bands")

    topology = load_subgrid_topology(restart)
    values = load_native_variables(restart, list(_RESTART_VARIABLES))
    direct = jnp.asarray(direct_shortwave_gridcell, dtype=jnp.float64)
    diffuse = jnp.asarray(diffuse_shortwave_gridcell, dtype=jnp.float64)
    grid_band_shape = (dimensions["gridcell"], 2)
    if direct.shape != grid_band_shape or diffuse.shape != grid_band_shape:
        raise ValueError(f"shortwave forcing must have shape {grid_band_shape}")

    inputs = SurfaceRadiationInputs(
        direct_shortwave_gridcell=direct,
        diffuse_shortwave_gridcell=diffuse,
        snow_layer_count_column=jnp.asarray(values["SNLSNO"], dtype=jnp.int32),
        snow_depth_column=_finite_restart_float(values["SNOW_DEPTH"]),
        ground_albedo_direct_column=_finite_restart_float(values["albgrd"]),
        ground_albedo_diffuse_column=_finite_restart_float(values["albgri"]),
        soil_albedo_direct_column=_finite_restart_float(values["albsod"]),
        soil_albedo_diffuse_column=_finite_restart_float(values["albsoi"]),
        snow_albedo_direct_column=_finite_restart_float(values["albsnd_hst"]),
        snow_albedo_diffuse_column=_finite_restart_float(values["albsni_hst"]),
        surface_albedo_direct_patch=_finite_restart_float(values["albd"]),
        surface_albedo_diffuse_patch=_finite_restart_float(values["albi"]),
        canopy_absorption_direct_patch=_finite_restart_float(values["fabd"]),
        canopy_absorption_diffuse_patch=_finite_restart_float(values["fabi"]),
        direct_transmittance_patch=_finite_restart_float(values["ftdd"]),
        direct_to_diffuse_transmittance_patch=_finite_restart_float(values["ftid"]),
        diffuse_transmittance_patch=_finite_restart_float(values["ftii"]),
        snow_absorption_direct_visible_column=_finite_restart_float(values["flx_absdv"]),
        snow_absorption_direct_nir_column=_finite_restart_float(values["flx_absdn"]),
        snow_absorption_diffuse_visible_column=_finite_restart_float(values["flx_absiv"]),
        snow_absorption_diffuse_nir_column=_finite_restart_float(values["flx_absin"]),
    )
    sunlit_fraction = _finite_restart_float(values["fsun"])
    return SurfaceRadiationRestartData(
        topology=topology,
        clock=load_restart_clock(restart),
        inputs=inputs,
        state=_initial_state(sunlit_fraction, dimensions["levsno1"]),
    )
