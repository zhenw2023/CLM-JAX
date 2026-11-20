
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from clm_jax.forcing.datm import DatmForcing


ATMOSPHERIC_FLUX_FIELDS = frozenset((
    "longwave_down", "shortwave_nir_direct", "shortwave_nir_diffuse",
    "shortwave_visible_direct", "shortwave_visible_diffuse",
    "rain_convective", "rain_large_scale", "snow_convective", "snow_large_scale",
))


def driver_to_land_area_factor(atmosphere_area, land_area, land_mask):
    atmosphere_area = np.asarray(atmosphere_area, dtype=np.float64)
    land_area = np.asarray(land_area, dtype=np.float64)
    mask = np.asarray(land_mask, dtype=np.float64)
    if atmosphere_area.shape != land_area.shape or mask.shape != land_area.shape:
        raise ValueError("atmosphere and land area correction grids must align")
    radius_km = np.float64(6371220.0) * np.float64(0.001)
    radius_squared = radius_km * radius_km
    model_area_km2 = land_area * radius_squared
    model_area = model_area_km2 / radius_squared
    active = np.abs(mask) >= np.float64(1e-6)
    if np.any(~np.isfinite(atmosphere_area[active])) or np.any(
        ~np.isfinite(model_area[active])
    ):
        raise ValueError("nonfinite active domain area")
    factor = np.ones_like(land_area)
    if np.any(active & (model_area * atmosphere_area == np.float64(0.0))):
        raise ValueError("zero active domain area")
    valid = active
    factor[valid] = np.float64(1.0) / (model_area[valid] / atmosphere_area[valid])
    return factor


def load_land_flux_area_factor(case, topology):
    from clm_jax.io.netcdf import load_native_variables

    atmosphere_path = case.path_value("ATM_DOMAIN_PATH") / case.value("ATM_DOMAIN_FILE")
    land_path = case.path_value("LND_DOMAIN_PATH") / case.value("LND_DOMAIN_FILE")
    atmosphere = load_native_variables(atmosphere_path, ["area", "xc", "yc"], dtype=np.float64)
    land = load_native_variables(land_path, ["area", "mask", "xc", "yc"], dtype=np.float64)
    if not all(np.array_equal(atmosphere[name], land[name]) for name in ("xc", "yc")):
        raise ValueError("driver-to-land flux correction requires the configured coincident grids")
    factor = driver_to_land_area_factor(atmosphere["area"], land["area"], land["mask"])
    latitude = np.asarray(topology.grid.latitude_index, dtype=np.int64)
    longitude = np.asarray(topology.grid.longitude_index, dtype=np.int64)
    if factor.ndim != 2 or np.any(latitude < 0) or np.any(latitude >= factor.shape[0]) \
            or np.any(longitude < 0) or np.any(longitude >= factor.shape[1]):
        raise ValueError("CLM indices lie outside the area correction domain")
    return factor[latitude, longitude]


@jax.jit
def correct_atmospheric_land_fluxes(forcing: DatmForcing, factor) -> DatmForcing:
    factor = jnp.asarray(factor, dtype=jnp.float64)
    return forcing._replace(**{
        name: getattr(forcing, name) * factor for name in ATMOSPHERIC_FLUX_FIELDS
    })
