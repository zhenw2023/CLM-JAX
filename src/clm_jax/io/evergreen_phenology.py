
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    SubgridTopology,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)
from clm_jax.physics.evergreen_phenology import (
    CARBON_STORAGE_POOL_NAMES,
    CARBON_TRANSFER_POOL_NAMES,
    NITROGEN_STORAGE_POOL_NAMES,
    NITROGEN_TRANSFER_POOL_NAMES,
    EvergreenPhenologyInputs,
    EvergreenPhenologyParameters,
    EvergreenPhenologyState,
)

_STATE_VARIABLES = ("bglfr", "bgtr", "lgsf")
_PARAMETER_VARIABLES = ("evergreen", "leaf_long", "woody")
_POOL_VARIABLES = (
    *CARBON_STORAGE_POOL_NAMES,
    *CARBON_TRANSFER_POOL_NAMES,
    *NITROGEN_STORAGE_POOL_NAMES,
    *NITROGEN_TRANSFER_POOL_NAMES,
)


class EvergreenPhenologyNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: EvergreenPhenologyInputs
    state: EvergreenPhenologyState
    parameters: EvergreenPhenologyParameters
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


def _stack_patch_pools(
    values: dict[str, jnp.ndarray],
    names: tuple[str, ...],
) -> np.ndarray:
    return np.stack([np.asarray(values[name], dtype=np.float64) for name in names], axis=1)


def load_evergreen_phenology_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> EvergreenPhenologyNativeData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    for key, expected in {"use_cn": True, "use_fates": False}.items():
        if general.get(key) is not expected:
            raise ValueError(f"native evergreen adapter requires {key}={expected!r}")
    if case.value("CALENDAR").upper() != "NO_LEAP":
        raise ValueError("native evergreen adapter currently requires NO_LEAP")

    nitrogen_options = read_case_namelist_group(case, "lnd_in", "clm_nitrogen")
    option = nitrogen_options.get("cn_evergreen_phenology_opt")
    if isinstance(option, bool) or not isinstance(option, int):
        raise ValueError("clm_nitrogen cn_evergreen_phenology_opt must be integer")

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    parameter_data = _selected_parameter_path(case, parameter_data_path)
    topology = load_subgrid_topology(restart)
    patch_count = topology.patch.gridcell.shape[0]

    values = load_native_variables(
        restart,
        [*_POOL_VARIABLES, *_STATE_VARIABLES],
        dtype=np.float64,
    )
    for name, array in values.items():
        if array.shape != (patch_count,):
            raise ValueError(f"restart variable {name!r} must contain every patch")
    pft_values = load_native_variables(
        parameter_data,
        list(_PARAMETER_VARIABLES),
        dtype=np.float64,
    )
    pft_count = pft_values["evergreen"].shape[0]
    for name in _PARAMETER_VARIABLES:
        array = np.asarray(pft_values[name], dtype=np.float64)
        if array.shape != (pft_count,) or not np.all(np.isfinite(array)):
            raise ValueError(f"CLM PFT parameter {name!r} must be one finite vector")
    evergreen_pft = np.asarray(pft_values["evergreen"], dtype=np.float64)
    woody_pft = np.asarray(pft_values["woody"], dtype=np.float64)
    leaf_longevity = np.asarray(pft_values["leaf_long"], dtype=np.float64)
    if not np.all((evergreen_pft == 0.0) | (evergreen_pft == 1.0)):
        raise ValueError("CLM evergreen flags must contain only zero or one")
    if not np.all((woody_pft == 0.0) | (woody_pft == 1.0)):
        raise ValueError("CLM woody flags must contain only zero or one")
    if np.any(leaf_longevity[evergreen_pft == 1.0] <= 0.0):
        raise ValueError("evergreen PFT leaf longevity must be positive")

    active = np.asarray(topology.patch.active, dtype=bool)
    landunit = np.asarray(topology.patch.landunit_type, dtype=np.int32)
    vegetation_type = np.asarray(topology.patch.vegetation_type, dtype=np.int32)
    if np.any((vegetation_type < 0) | (vegetation_type >= pft_count)):
        raise ValueError("restart vegetation type lies outside PFT parameter vectors")
    soil_patch = active & (
        (landunit == LANDUNIT_SOIL) | (landunit == LANDUNIT_CROP)
    )
    evergreen_patch = soil_patch & (evergreen_pft[vegetation_type] == 1.0)
    for name in _POOL_VARIABLES:
        if not np.all(np.isfinite(np.asarray(values[name])[evergreen_patch])):
            raise ValueError(f"evergreen patches contain missing {name}")
    for name in _STATE_VARIABLES:
        if not np.all(np.isfinite(np.asarray(values[name])[evergreen_patch])):
            raise ValueError(f"evergreen patches contain missing {name}")

    clock = load_restart_clock(restart)
    return EvergreenPhenologyNativeData(
        topology=topology,
        inputs=EvergreenPhenologyInputs(
            soil_patch=jnp.asarray(soil_patch),
            vegetation_type_patch=jnp.asarray(vegetation_type),
            carbon_storage_patch_pool=jnp.asarray(
                _stack_patch_pools(values, CARBON_STORAGE_POOL_NAMES)
            ),
            carbon_transfer_patch_pool=jnp.asarray(
                _stack_patch_pools(values, CARBON_TRANSFER_POOL_NAMES)
            ),
            nitrogen_storage_patch_pool=jnp.asarray(
                _stack_patch_pools(values, NITROGEN_STORAGE_POOL_NAMES)
            ),
            nitrogen_transfer_patch_pool=jnp.asarray(
                _stack_patch_pools(values, NITROGEN_TRANSFER_POOL_NAMES)
            ),
        ),
        state=EvergreenPhenologyState(
            background_litterfall_rate_patch=jnp.asarray(values["bglfr"]),
            background_transfer_growth_rate_patch=jnp.asarray(values["bgtr"]),
            long_growing_season_factor_patch=jnp.asarray(values["lgsf"]),
        ),
        parameters=EvergreenPhenologyParameters(
            evergreen_pft=jnp.asarray(evergreen_pft),
            leaf_longevity_years_pft=jnp.asarray(leaf_longevity),
            woody_pft=jnp.asarray(woody_pft),
            time_step_seconds=np.float64(clock.step_seconds),
            days_per_year=np.float64(365.0),
            seconds_per_day=np.float64(86_400.0),
            evergreen_phenology_option=option,
            storage_transfer_fraction=np.float64(0.0002),
        ),
        restart_path=restart,
        parameter_data_path=parameter_data,
    )


__all__ = [
    "EvergreenPhenologyNativeData",
    "load_evergreen_phenology_native_data",
]
