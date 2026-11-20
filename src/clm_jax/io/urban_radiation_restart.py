
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    RestartClock,
    SubgridTopology,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)
from clm_jax.physics.surface_radiation import SurfaceRadiationState
from clm_jax.physics.urban_radiation import (
    UrbanRadiationFluxState,
    UrbanRadiationInputs,
    UrbanRadiationState,
    UrbanShortwaveAbsorptionState,
)


class UrbanRadiationRestartData(NamedTuple):

    topology: SubgridTopology
    clock: RestartClock
    inputs: UrbanRadiationInputs
    state: UrbanRadiationState
    surface_data_path: Path


_RESTART_VARIABLES = (
    "T_GRND",
    "frac_sno",
    "sabs_roof_dir",
    "sabs_roof_dif",
    "sabs_sunwall_dir",
    "sabs_sunwall_dif",
    "sabs_shadewall_dir",
    "sabs_shadewall_dif",
    "sabs_improad_dir",
    "sabs_improad_dif",
    "sabs_perroad_dir",
    "sabs_perroad_dif",
    "EFLX_LWRAD_OUT",
)

_SURFACE_VARIABLES = (
    "CANYON_HWR",
    "WTROAD_PERV",
    "EM_ROOF",
    "EM_IMPROAD",
    "EM_PERROAD",
    "EM_WALL",
)


def _finite_restart_float(value: jnp.ndarray) -> jnp.ndarray:
    value = jnp.asarray(value, dtype=jnp.float64)
    return jnp.where(jnp.isnan(value), jnp.float64(CLM_SPECIAL_VALUE), value)


def _configured_surface_data(case: CimeCase) -> Path:
    group = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    declared = group.get("fsurdat")
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(f"clm_inparm in {case.case_docs / 'lnd_in'} lacks fsurdat")
    path = Path(case.expand(declared)).expanduser()
    if not path.is_absolute():
        path = case.root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM surface data does not exist: {path}")
    return path


def _validate_unique_urban_surfaces(topology: SubgridTopology) -> None:
    landunit = np.asarray(topology.column.landunit, dtype=np.int64)
    landunit_type = np.asarray(topology.column.landunit_type, dtype=np.int64)
    column_type = np.asarray(topology.column.column_type, dtype=np.int64)
    active = np.asarray(topology.column.active, dtype=bool)
    urban = (
        active
        & (landunit_type >= LANDUNIT_URBAN_MIN)
        & (landunit_type <= LANDUNIT_URBAN_MAX)
        & (column_type >= 71)
        & (column_type <= 75)
    )
    if not np.any(urban):
        return
    number_of_landunits = topology.landunit.gridcell.shape[0]
    if np.any((landunit[urban] < 0) | (landunit[urban] >= number_of_landunits)):
        raise ValueError("an active urban column has an invalid parent landunit")
    keys = np.stack((landunit[urban], column_type[urban]), axis=1)
    _, counts = np.unique(keys, axis=0, return_counts=True)
    if np.any(counts > 1):
        raise ValueError("an urban landunit has duplicate columns for one surface type")


def _urban_landunit_surface_fields(
    topology: SubgridTopology,
    surface_values: dict[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:

    landunit_type = np.asarray(topology.landunit.landunit_type, dtype=np.int64)
    active = np.asarray(topology.landunit.active, dtype=bool)
    urban = (
        active
        & (landunit_type >= LANDUNIT_URBAN_MIN)
        & (landunit_type <= LANDUNIT_URBAN_MAX)
    )
    number_of_landunits = landunit_type.size
    result: dict[str, jnp.ndarray] = {}
    if not np.any(urban):
        for name in _SURFACE_VARIABLES:
            result[name] = jnp.full(
                (number_of_landunits,), CLM_SPECIAL_VALUE, dtype=jnp.float64
            )
        return result

    landunit_gridcell = np.asarray(topology.landunit.gridcell, dtype=np.int64)
    number_of_gridcells = topology.grid.longitude_degrees.shape[0]
    if np.any(
        (landunit_gridcell[urban] < 0)
        | (landunit_gridcell[urban] >= number_of_gridcells)
    ):
        raise ValueError("an active urban landunit has an invalid parent gridcell")
    longitude_index = np.asarray(topology.grid.longitude_index, dtype=np.int64)
    latitude_index = np.asarray(topology.grid.latitude_index, dtype=np.int64)
    gridcell = landunit_gridcell[urban]
    longitude = longitude_index[gridcell]
    latitude = latitude_index[gridcell]
    density = landunit_type[urban] - LANDUNIT_URBAN_MIN

    for name in _SURFACE_VARIABLES:
        source = np.asarray(surface_values[name], dtype=np.float64)
        if source.ndim != 3 or source.shape[0] != 3:
            raise ValueError(
                f"surface variable {name} must have dimensions (numurbl, lat, lon)"
            )
        if np.any(
            (latitude < 0)
            | (latitude >= source.shape[1])
            | (longitude < 0)
            | (longitude >= source.shape[2])
        ):
            raise ValueError(f"native grid indices are outside surface variable {name}")
        selected = source[density, latitude, longitude]
        if not np.all(np.isfinite(selected)):
            raise ValueError(f"surface variable {name} is missing on an urban landunit")
        field = np.full(number_of_landunits, CLM_SPECIAL_VALUE, dtype=np.float64)
        field[urban] = selected
        result[name] = jnp.asarray(field)
    return result


def _initial_flux_state(
    emitted_longwave_patch: jnp.ndarray,
    number_of_patches: int,
    preceding_surface_state: SurfaceRadiationState | None,
) -> UrbanRadiationFluxState:
    special = jnp.full(
        (number_of_patches,), jnp.float64(CLM_SPECIAL_VALUE), dtype=jnp.float64
    )
    if preceding_surface_state is None:
        absorbed_ground = special
        absorbed_vegetation = special
        absorbed_total = special
    else:
        absorbed_ground = jnp.asarray(
            preceding_surface_state.fluxes.absorbed_ground_patch, dtype=jnp.float64
        )
        absorbed_vegetation = jnp.asarray(
            preceding_surface_state.fluxes.absorbed_vegetation_patch,
            dtype=jnp.float64,
        )
        absorbed_total = jnp.asarray(
            preceding_surface_state.fluxes.absorbed_total_patch, dtype=jnp.float64
        )
        for name, value in (
            ("absorbed_ground_patch", absorbed_ground),
            ("absorbed_vegetation_patch", absorbed_vegetation),
            ("absorbed_total_patch", absorbed_total),
        ):
            if value.shape != (number_of_patches,):
                raise ValueError(f"preceding SurfaceRadiation {name} has the wrong shape")

    return UrbanRadiationFluxState(
        absorbed_ground_patch=absorbed_ground,
        absorbed_vegetation_patch=absorbed_vegetation,
        absorbed_total_patch=absorbed_total,
        urban_absorbed_total_patch=special,
        emitted_longwave_patch=_finite_restart_float(emitted_longwave_patch),
        net_longwave_patch=special,
        urban_net_longwave_patch=special,
    )


def load_urban_radiation_restart(
    case: CimeCase,
    direct_shortwave_gridcell: jnp.ndarray,
    diffuse_shortwave_gridcell: jnp.ndarray,
    downward_longwave_gridcell: jnp.ndarray,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    preceding_surface_state: SurfaceRadiationState | None = None,
) -> UrbanRadiationRestartData:

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    surface_data = (
        _configured_surface_data(case)
        if surface_data_path is None
        else Path(surface_data_path).expanduser().resolve()
    )
    if not surface_data.is_file():
        raise FileNotFoundError(f"CLM surface data does not exist: {surface_data}")

    dimensions = load_dimension_sizes(restart)
    required_dimensions = ("gridcell", "landunit", "column", "pft", "numrad")
    missing = [name for name in required_dimensions if name not in dimensions]
    if missing:
        raise ValueError(f"restart lacks dimensions {missing}: {restart}")
    if dimensions["numrad"] != 2:
        raise ValueError("UrbanRadiation requires the two native CLM solar bands")

    topology = load_subgrid_topology(restart)
    _validate_unique_urban_surfaces(topology)
    restart_values = load_native_variables(restart, list(_RESTART_VARIABLES))
    surface_values = load_native_variables(surface_data, list(_SURFACE_VARIABLES))
    parameters = _urban_landunit_surface_fields(topology, surface_values)

    direct = jnp.asarray(direct_shortwave_gridcell, dtype=jnp.float64)
    diffuse = jnp.asarray(diffuse_shortwave_gridcell, dtype=jnp.float64)
    longwave = jnp.asarray(downward_longwave_gridcell, dtype=jnp.float64)
    grid_band_shape = (dimensions["gridcell"], 2)
    if direct.shape != grid_band_shape or diffuse.shape != grid_band_shape:
        raise ValueError(f"shortwave forcing must have shape {grid_band_shape}")
    if longwave.shape != (dimensions["gridcell"],):
        raise ValueError(
            f"downward longwave forcing must have shape ({dimensions['gridcell']},)"
        )

    inputs = UrbanRadiationInputs(
        direct_shortwave_gridcell=direct,
        diffuse_shortwave_gridcell=diffuse,
        downward_longwave_gridcell=longwave,
        snow_fraction_column=_finite_restart_float(restart_values["frac_sno"]),
        ground_temperature_column=_finite_restart_float(restart_values["T_GRND"]),
        canyon_height_width_ratio_landunit=parameters["CANYON_HWR"],
        pervious_road_fraction_landunit=parameters["WTROAD_PERV"],
        roof_emissivity_landunit=parameters["EM_ROOF"],
        impervious_road_emissivity_landunit=parameters["EM_IMPROAD"],
        pervious_road_emissivity_landunit=parameters["EM_PERROAD"],
        wall_emissivity_landunit=parameters["EM_WALL"],
    )
    coefficients = UrbanShortwaveAbsorptionState(
        roof_direct_landunit=_finite_restart_float(restart_values["sabs_roof_dir"]),
        roof_diffuse_landunit=_finite_restart_float(restart_values["sabs_roof_dif"]),
        sun_wall_direct_landunit=_finite_restart_float(
            restart_values["sabs_sunwall_dir"]
        ),
        sun_wall_diffuse_landunit=_finite_restart_float(
            restart_values["sabs_sunwall_dif"]
        ),
        shade_wall_direct_landunit=_finite_restart_float(
            restart_values["sabs_shadewall_dir"]
        ),
        shade_wall_diffuse_landunit=_finite_restart_float(
            restart_values["sabs_shadewall_dif"]
        ),
        impervious_road_direct_landunit=_finite_restart_float(
            restart_values["sabs_improad_dir"]
        ),
        impervious_road_diffuse_landunit=_finite_restart_float(
            restart_values["sabs_improad_dif"]
        ),
        pervious_road_direct_landunit=_finite_restart_float(
            restart_values["sabs_perroad_dir"]
        ),
        pervious_road_diffuse_landunit=_finite_restart_float(
            restart_values["sabs_perroad_dif"]
        ),
    )
    state = UrbanRadiationState(
        shortwave_absorption=coefficients,
        fluxes=_initial_flux_state(
            restart_values["EFLX_LWRAD_OUT"],
            dimensions["pft"],
            preceding_surface_state,
        ),
    )
    return UrbanRadiationRestartData(
        topology=topology,
        clock=load_restart_clock(restart),
        inputs=inputs,
        state=state,
        surface_data_path=surface_data,
    )
