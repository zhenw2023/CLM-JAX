
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from clm_jax.forcing.datm import DatmForcing
from clm_jax.forcing.area import ATMOSPHERIC_FLUX_FIELDS
from clm_jax.forcing.remap import (
    apply_normalized_sparse_mapping,
    load_sparse_mapping,
)
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import SubgridTopology


class GlacierCouplerState(NamedTuple):

    ice_mask: jnp.ndarray
    ice_fraction: jnp.ndarray
    elevation: jnp.ndarray


_ATMOSPHERIC_COUPLER_VARIABLES = {
    "reference_height": "a2x_ax_Sa_z",
    "temperature": "a2x_ax_Sa_tbot",
    "potential_temperature": "a2x_ax_Sa_ptem",
    "specific_humidity": "a2x_ax_Sa_shum",
    "density": "a2x_ax_Sa_dens",
    "pressure": "a2x_ax_Sa_pbot",
    "longwave_down": "a2x_ax_Faxa_lwdn",
    "wind_u": "a2x_ax_Sa_u",
    "wind_v": "a2x_ax_Sa_v",
    "shortwave_nir_direct": "a2x_ax_Faxa_swndr",
    "shortwave_nir_diffuse": "a2x_ax_Faxa_swndf",
    "shortwave_visible_direct": "a2x_ax_Faxa_swvdr",
    "shortwave_visible_diffuse": "a2x_ax_Faxa_swvdf",
    "rain_convective": "a2x_ax_Faxa_rainc",
    "rain_large_scale": "a2x_ax_Faxa_rainl",
    "snow_convective": "a2x_ax_Faxa_snowc",
    "snow_large_scale": "a2x_ax_Faxa_snowl",
}

_AEROSOL_COUPLER_VARIABLES = (
    "a2x_ax_Faxa_bcphidry",
    "a2x_ax_Faxa_bcphodry",
    "a2x_ax_Faxa_bcphiwet",
    "a2x_ax_Faxa_ocphidry",
    "a2x_ax_Faxa_ocphodry",
    "a2x_ax_Faxa_ocphiwet",
    "a2x_ax_Faxa_dstwet1",
    "a2x_ax_Faxa_dstdry1",
    "a2x_ax_Faxa_dstwet2",
    "a2x_ax_Faxa_dstdry2",
    "a2x_ax_Faxa_dstwet3",
    "a2x_ax_Faxa_dstdry3",
    "a2x_ax_Faxa_dstwet4",
    "a2x_ax_Faxa_dstdry4",
)


def _land_grid_indices(
    native: dict[str, np.ndarray],
    names: list[str] | tuple[str, ...],
    topology: SubgridTopology,
) -> np.ndarray:

    source_sizes = {np.asarray(native[name]).size for name in names}
    if len(source_sizes) != 1:
        raise ValueError("coupler atmospheric fields do not share one native grid")
    source_size = next(iter(source_sizes))

    longitude_index = np.asarray(topology.grid.longitude_index, dtype=np.int64)
    latitude_index = np.asarray(topology.grid.latitude_index, dtype=np.int64)
    if longitude_index.shape != latitude_index.shape or longitude_index.ndim != 1:
        raise ValueError("CLM grid indices must be one-dimensional and aligned")
    longitude_count = int(np.max(longitude_index)) + 1
    if longitude_count <= 0 or source_size % longitude_count:
        raise ValueError("coupler atmospheric grid is incompatible with CLM indices")
    latitude_count = source_size // longitude_count
    if (
        np.any(longitude_index < 0)
        or np.any(longitude_index >= longitude_count)
        or np.any(latitude_index < 0)
        or np.any(latitude_index >= latitude_count)
    ):
        raise ValueError("CLM land-grid indices lie outside the coupler atmosphere grid")
    return latitude_index * longitude_count + longitude_index


def load_glacier_coupler_state(path: str | Path) -> GlacierCouplerState:

    names = (
        "g2x_gx_Sg_icemask",
        "g2x_gx_Sg_ice_covered",
        "g2x_gx_Sg_topo",
    )
    values = load_native_variables(path, names, dtype=np.float64)
    return GlacierCouplerState(
        ice_mask=jnp.asarray(values[names[0]], dtype=jnp.float64).reshape(-1),
        ice_fraction=jnp.asarray(values[names[1]], dtype=jnp.float64).reshape(-1),
        elevation=jnp.asarray(values[names[2]], dtype=jnp.float64).reshape(-1),
    )


def load_atmospheric_coupler_forcing(
    path: str | Path,
    topology: SubgridTopology,
    *,
    land_flux_area_factor=None,
) -> DatmForcing:

    coupler_restart = Path(path).expanduser().resolve()
    if not coupler_restart.is_file():
        raise FileNotFoundError(f"CPL restart does not exist: {coupler_restart}")
    names = list(_ATMOSPHERIC_COUPLER_VARIABLES.values())
    native = load_native_variables(coupler_restart, names, dtype=np.float64)
    full_grid_index = _land_grid_indices(native, names, topology)

    selected: dict[str, jnp.ndarray] = {}
    for field, name in _ATMOSPHERIC_COUPLER_VARIABLES.items():
        value = np.asarray(native[name], dtype=np.float64).reshape(-1)[full_grid_index]
        if land_flux_area_factor is not None and field in ATMOSPHERIC_FLUX_FIELDS:
            value = value * np.asarray(land_flux_area_factor, dtype=np.float64)
        if not np.all(np.isfinite(value)):
            raise ValueError(f"coupler atmospheric field {name!r} is non-finite on land")
        selected[field] = jnp.asarray(value, dtype=jnp.float64)
    return DatmForcing(**selected)


def load_atmospheric_aerosol_coupler_forcing(
    path: str | Path,
    topology: SubgridTopology,
    *,
    land_flux_area_factor=None,
) -> jnp.ndarray:

    coupler_restart = Path(path).expanduser().resolve()
    if not coupler_restart.is_file():
        raise FileNotFoundError(f"CPL restart does not exist: {coupler_restart}")
    names = list(_AEROSOL_COUPLER_VARIABLES)
    native = load_native_variables(coupler_restart, names, dtype=np.float64)
    full_grid_index = _land_grid_indices(native, names, topology)
    selected = np.stack(
        [
            np.asarray(native[name], dtype=np.float64).reshape(-1)[full_grid_index]
            for name in names
        ],
        axis=1,
    )
    if not np.all(np.isfinite(selected)):
        raise ValueError("coupler aerosol deposition is non-finite on the CLM grid")
    if land_flux_area_factor is not None:
        selected = selected * np.asarray(land_flux_area_factor, dtype=np.float64)[:, None]
    return jnp.asarray(selected, dtype=jnp.float64)


def load_river_volume_coupler_state(
    path: str | Path,
    topology: SubgridTopology,
    mapping_path: str | Path,
    gridcell_area_square_kilometers: jnp.ndarray,
    *,
    land_flux_area_factor=None,
) -> jnp.ndarray:

    coupler_restart = Path(path).expanduser().resolve()
    if not coupler_restart.is_file():
        raise FileNotFoundError(f"CPL restart does not exist: {coupler_restart}")
    mapping_file = Path(mapping_path).expanduser().resolve()
    if not mapping_file.is_file():
        raise FileNotFoundError(f"ROF-to-LND mapping does not exist: {mapping_file}")

    field_name = "r2x_rx_Flrr_volr"
    source = jnp.asarray(
        load_native_variables(coupler_restart, [field_name], dtype=np.float64)[
            field_name
        ],
        dtype=jnp.float64,
    ).reshape(-1)
    mapping = load_sparse_mapping(mapping_file)
    return map_river_volume_to_land(
        source, topology, mapping, gridcell_area_square_kilometers,
        land_flux_area_factor=land_flux_area_factor,
    )


def map_river_volume_to_land(
    source, topology, mapping, gridcell_area_square_kilometers,
    *, land_flux_area_factor=None,
):
    source = jnp.asarray(source, dtype=jnp.float64).reshape(-1)
    if source.shape != (mapping.source_size,):
        raise ValueError(
            f"coupler river depth has size {source.size}, "
            f"but ROF-to-LND mapping expects {mapping.source_size}"
        )
    if not np.all(np.isfinite(np.asarray(source))):
        raise ValueError("coupler river depth contains non-finite values")

    if jax.default_backend() == "gpu":
        from clm_jax.forcing.ordered_mapping import prepare_ordered_mapping

        mapping = prepare_ordered_mapping(mapping)
    mapped_depth = apply_normalized_sparse_mapping(mapping, source)
    longitude_index = np.asarray(topology.grid.longitude_index, dtype=np.int64)
    latitude_index = np.asarray(topology.grid.latitude_index, dtype=np.int64)
    if longitude_index.shape != latitude_index.shape or longitude_index.ndim != 1:
        raise ValueError("CLM grid indices must be one-dimensional and aligned")
    longitude_count = int(np.max(longitude_index)) + 1
    if longitude_count <= 0 or mapping.destination_size % longitude_count:
        raise ValueError("ROF-to-LND destination grid is incompatible with CLM indices")
    latitude_count = mapping.destination_size // longitude_count
    if (
        np.any(longitude_index < 0)
        or np.any(longitude_index >= longitude_count)
        or np.any(latitude_index < 0)
        or np.any(latitude_index >= latitude_count)
    ):
        raise ValueError("CLM land-grid indices lie outside the ROF mapping destination")
    land_index = latitude_index * longitude_count + longitude_index
    land_depth = mapped_depth[jnp.asarray(land_index, dtype=jnp.int32)]
    if land_flux_area_factor is not None:
        land_depth = land_depth * jnp.asarray(land_flux_area_factor, dtype=jnp.float64)

    area = jnp.asarray(gridcell_area_square_kilometers, dtype=jnp.float64)
    if area.shape != land_depth.shape:
        raise ValueError(
            "gridcell_area_square_kilometers and compressed CLM grid disagree"
        )
    if not np.all(np.isfinite(np.asarray(area))) or np.any(np.asarray(area) <= 0.0):
        raise ValueError("CLM grid-cell area must be finite and positive")
    area_square_metres = area * jnp.float64(1.0e6)
    return land_depth * area_square_metres


__all__ = [
    "GlacierCouplerState",
    "load_atmospheric_aerosol_coupler_forcing",
    "load_atmospheric_coupler_forcing",
    "load_glacier_coupler_state",
    "load_river_volume_coupler_state",
    "map_river_volume_to_land",
]
