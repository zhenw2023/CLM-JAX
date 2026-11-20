
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
    CARBON_TRANSFER_POOL_NAMES,
    NITROGEN_TRANSFER_POOL_NAMES,
)
from clm_jax.physics.onset_growth import OnsetGrowthInputs, OnsetGrowthParameters

_STATE_VARIABLES = ("onset_flag", "onset_counter", "bgtr")
_PARAMETER_VARIABLES = ("woody",)
_POOL_VARIABLES = (
    *CARBON_TRANSFER_POOL_NAMES,
    *NITROGEN_TRANSFER_POOL_NAMES,
)


class OnsetGrowthNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: OnsetGrowthInputs
    parameters: OnsetGrowthParameters
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
    return np.stack(
        [np.asarray(values[name], dtype=np.float64) for name in names],
        axis=1,
    )


def load_onset_growth_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> OnsetGrowthNativeData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    for key, expected in {"use_cn": True, "use_fates": False}.items():
        if general.get(key) is not expected:
            raise ValueError(f"native onset-growth adapter requires {key}={expected!r}")

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
        [*_STATE_VARIABLES, *_POOL_VARIABLES],
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
    woody_pft = np.asarray(pft_values["woody"], dtype=np.float64)
    if woody_pft.ndim != 1 or woody_pft.size < 1:
        raise ValueError("CLM woody parameter must be a nonempty PFT vector")
    if not np.all(np.isfinite(woody_pft)):
        raise ValueError("CLM woody parameter must be finite")
    if not np.all((woody_pft == 0.0) | (woody_pft == 1.0)):
        raise ValueError("CLM woody flags must contain only zero or one")

    active = np.asarray(topology.patch.active, dtype=bool)
    landunit = np.asarray(topology.patch.landunit_type, dtype=np.int32)
    vegetation_type = np.asarray(topology.patch.vegetation_type, dtype=np.int32)
    if np.any((vegetation_type < 0) | (vegetation_type >= woody_pft.size)):
        raise ValueError("restart vegetation type lies outside woody PFT vector")
    soil_patch = active & (
        (landunit == LANDUNIT_SOIL) | (landunit == LANDUNIT_CROP)
    )
    for name in (*_STATE_VARIABLES, *_POOL_VARIABLES):
        if not np.all(np.isfinite(np.asarray(values[name])[soil_patch])):
            raise ValueError(f"active soil patches contain missing {name}")

    clock = load_restart_clock(restart)
    return OnsetGrowthNativeData(
        topology=topology,
        inputs=OnsetGrowthInputs(
            soil_patch=jnp.asarray(soil_patch),
            vegetation_type_patch=jnp.asarray(vegetation_type),
            onset_flag_patch=jnp.asarray(values["onset_flag"]),
            onset_counter_seconds_patch=jnp.asarray(values["onset_counter"]),
            background_transfer_growth_rate_patch=jnp.asarray(values["bgtr"]),
            carbon_transfer_patch_pool=jnp.asarray(
                _stack_patch_pools(values, CARBON_TRANSFER_POOL_NAMES)
            ),
            nitrogen_transfer_patch_pool=jnp.asarray(
                _stack_patch_pools(values, NITROGEN_TRANSFER_POOL_NAMES)
            ),
        ),
        parameters=OnsetGrowthParameters(
            woody_pft=jnp.asarray(woody_pft),
            time_step_seconds=np.float64(clock.step_seconds),
        ),
        restart_path=restart,
        parameter_data_path=parameter_data,
    )


__all__ = [
    "OnsetGrowthNativeData",
    "load_onset_growth_native_data",
]
