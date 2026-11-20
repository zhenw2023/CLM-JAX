
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.physics.snow_snicar import (
    NUMBER_OF_SNOW_AEROSOLS,
    NUMBER_OF_SNOW_GRAIN_RADII,
    NUMBER_OF_SNOW_SPECTRAL_BANDS,
    SnowOpticalTables,
)

_ICE_VARIABLES = (
    "ss_alb_ice_drc",
    "asm_prm_ice_drc",
    "ext_cff_mss_ice_drc",
    "ss_alb_ice_dfs",
    "asm_prm_ice_dfs",
    "ext_cff_mss_ice_dfs",
)
_AEROSOL_SUFFIXES = (
    "bcphil",
    "bcphob",
    "ocphil",
    "ocphob",
    "dust01",
    "dust02",
    "dust03",
    "dust04",
)
_AEROSOL_PREFIXES = ("ss_alb", "asm_prm", "ext_cff_mss")
_AEROSOL_VARIABLES = tuple(
    f"{prefix}_{suffix}" for prefix in _AEROSOL_PREFIXES for suffix in _AEROSOL_SUFFIXES
)


def configured_snow_optics_path(case: CimeCase) -> Path:

    group = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    declared = group.get("fsnowoptics")
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(f"clm_inparm in {case.case_docs / 'lnd_in'} lacks fsnowoptics")
    path = Path(case.expand(declared)).expanduser()
    if not path.is_absolute():
        path = case.root / path
    return path.resolve()


def load_snow_optical_tables(
    case: CimeCase,
    snow_optics_path: str | Path | None = None,
) -> SnowOpticalTables:

    path = (
        configured_snow_optics_path(case)
        if snow_optics_path is None
        else Path(snow_optics_path).expanduser().resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(f"configured snow-optics table does not exist: {path}")
    loaded = load_native_variables(
        path,
        list(_ICE_VARIABLES + _AEROSOL_VARIABLES),
        dtype=np.float64,
    )
    file_ice_shape = (
        NUMBER_OF_SNOW_SPECTRAL_BANDS,
        NUMBER_OF_SNOW_GRAIN_RADII,
    )
    for name in _ICE_VARIABLES:
        values = np.asarray(loaded[name])
        if values.shape != file_ice_shape:
            raise ValueError(
                f"snow-optics variable {name!r} must have file shape "
                f"{file_ice_shape}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"snow-optics variable {name!r} contains non-finite data")
    aerosol_shape = (NUMBER_OF_SNOW_SPECTRAL_BANDS,)
    for name in _AEROSOL_VARIABLES:
        values = np.asarray(loaded[name])
        if values.shape != aerosol_shape:
            raise ValueError(
                f"snow-optics variable {name!r} must have shape {aerosol_shape}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"snow-optics variable {name!r} contains non-finite data")

    def ice_native_order(name: str) -> jnp.ndarray:
        values = np.asarray(loaded[name], dtype=np.float64).transpose(1, 0)
        return jnp.asarray(np.ascontiguousarray(values))

    def aerosol_species_band(prefix: str) -> jnp.ndarray:
        values = np.stack(
            tuple(np.asarray(loaded[f"{prefix}_{suffix}"]) for suffix in _AEROSOL_SUFFIXES),
            axis=0,
        )
        expected = (NUMBER_OF_SNOW_AEROSOLS, NUMBER_OF_SNOW_SPECTRAL_BANDS)
        if values.shape != expected:
            raise AssertionError(f"assembled aerosol table has shape {values.shape}")
        return jnp.asarray(np.ascontiguousarray(values, dtype=np.float64))

    return SnowOpticalTables(
        snow_single_scattering_albedo_direct_radius_band=ice_native_order("ss_alb_ice_drc"),
        snow_asymmetry_parameter_direct_radius_band=ice_native_order("asm_prm_ice_drc"),
        snow_mass_extinction_coefficient_direct_radius_band=ice_native_order("ext_cff_mss_ice_drc"),
        snow_single_scattering_albedo_diffuse_radius_band=ice_native_order("ss_alb_ice_dfs"),
        snow_asymmetry_parameter_diffuse_radius_band=ice_native_order("asm_prm_ice_dfs"),
        snow_mass_extinction_coefficient_diffuse_radius_band=ice_native_order(
            "ext_cff_mss_ice_dfs"
        ),
        aerosol_single_scattering_albedo_species_band=aerosol_species_band("ss_alb"),
        aerosol_asymmetry_parameter_species_band=aerosol_species_band("asm_prm"),
        aerosol_mass_extinction_coefficient_species_band=aerosol_species_band("ext_cff_mss"),
    )


__all__ = [
    "configured_snow_optics_path",
    "load_snow_optical_tables",
]
