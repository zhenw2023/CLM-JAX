
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.cn_precision_control import (
    CnPrecisionControlNativeData,
    load_cn_precision_control_native_data,
)
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    load_restart_clock,
    load_subgrid_topology,
)
from clm_jax.physics.cn_precision_control import (
    CnPrecisionControlParameters,
    CnPrecisionControlResult,
)
from clm_jax.physics.cn_state_update1 import (
    CnStateUpdate1Inputs,
    CnStateUpdate1Parameters,
    CnStateUpdate1State,
)
from clm_jax.physics.cn_zero import CnTransientFluxState
from clm_jax.physics.growth_respiration import GrowthRespirationResult
from clm_jax.physics.soil_biogeochem_potential import (
    DEFAULT_SOIL_BIOGEOCHEM_POTENTIAL_PARAMETERS,
)


class CnStateUpdate1NativeData(NamedTuple):

    inputs: CnStateUpdate1Inputs
    restart_state: CnStateUpdate1State
    parameters: CnStateUpdate1Parameters
    precision_parameters: CnPrecisionControlParameters
    restart_path: Path
    parameter_data_path: Path


class CnStateUpdate1Call(NamedTuple):

    inputs: CnStateUpdate1Inputs
    state: CnStateUpdate1State
    cn_fluxes: CnTransientFluxState
    parameters: CnStateUpdate1Parameters
    precision_parameters: CnPrecisionControlParameters


def _logical(group: dict[str, object], name: str, default: bool) -> bool:
    value = group.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"CLM namelist value {name!r} must be logical")
    return value


def _integer(group: dict[str, object], name: str, default: int) -> int:
    value = group.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"CLM namelist value {name!r} must be integer")
    return value


def _require_same_selected_file(
    name: str,
    requested: str | Path | None,
    actual: Path,
) -> None:
    if requested is None:
        return
    if Path(requested).expanduser().resolve() != actual:
        raise ValueError(f"precision native data uses a different {name} file")


def _pft_names(parameter_data_path: Path) -> tuple[str, ...]:
    from netCDF4 import Dataset  

    with Dataset(parameter_data_path, mode="r") as dataset:
        if "pftname" not in dataset.variables:
            raise ValueError("CLM parameter file lacks pftname")
        raw = np.asarray(dataset.variables["pftname"][:])
    if raw.ndim != 2 or raw.dtype.kind not in {"S", "U"}:
        raise ValueError("CLM pftname must be a character matrix")
    if raw.dtype.kind == "S":
        return tuple(
            b"".join(np.asarray(row, dtype="S1").tolist())
            .decode("ascii")
            .rstrip()
            for row in raw
        )
    return tuple(
        "".join(np.asarray(row, dtype=str).tolist()).rstrip() for row in raw
    )


def _first_prognostic_crop_pft(names: tuple[str, ...]) -> int:
    matches = tuple(
        index for index, name in enumerate(names) if name == "temperate_corn"
    )
    if len(matches) != 1:
        raise ValueError(
            "CLM pftname must contain exactly one 'temperate_corn' entry"
        )
    return matches[0]


def _woody_pft(parameter_data_path: Path, pft_count: int) -> jnp.ndarray:
    values = load_native_variables(
        parameter_data_path,
        ["woody"],
        dtype=np.float64,
        masked_floating_fill_value=0.0,
    )
    woody = np.asarray(values["woody"], dtype=np.float64)
    if woody.shape != (pft_count,) or not np.all(np.isfinite(woody)):
        raise ValueError("CLM parameter 'woody' must be one finite PFT vector")
    if not np.all((woody == 0.0) | (woody == 1.0)):
        raise ValueError("CLM parameter 'woody' must contain only zero or one")
    return jnp.asarray(woody)


def load_cn_state_update1_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    precision_native_data: CnPrecisionControlNativeData | None = None,
) -> CnStateUpdate1NativeData:

    precision = precision_native_data
    if precision is None:
        precision = load_cn_precision_control_native_data(
            case,
            restart_path=restart_path,
            parameter_data_path=parameter_data_path,
        )
    else:
        _require_same_selected_file("restart", restart_path, precision.restart_path)
        _require_same_selected_file(
            "parameter-data",
            parameter_data_path,
            precision.parameter_data_path,
        )

    restart = precision.restart_path
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")

    clm_group = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    cn_group = dict(read_case_namelist_group(case, "lnd_in", "cn_general"))
    required = {
        "use_cn": True,
        "use_crop": True,
        "use_century_decomp": True,
        "use_vertsoilc": True,
        "use_fates": False,
        "soil_layerstruct": "20SL_8.5m",
    }
    for name, expected in required.items():
        if clm_group.get(name) != expected:
            raise ValueError(
                f"native Update1 adapter requires {name}={expected!r}"
            )

    dimensions = load_dimension_sizes(restart)
    if dimensions.get("levgrnd") != 25:
        raise ValueError("Update1 restart must contain 25 native ground levels")
    topology = load_subgrid_topology(restart)
    patch_count = topology.patch.active.shape[0]
    column_count = topology.column.active.shape[0]
    if precision.inputs.soil_patch.shape != (patch_count,):
        raise ValueError("precision state and restart patch extents differ")
    if precision.inputs.vegetation_type_patch.shape != (patch_count,):
        raise ValueError("precision PFT identity and restart patch extents differ")
    soil_column = np.asarray(topology.column.active, dtype=bool) & np.isin(
        np.asarray(topology.column.landunit_type, dtype=np.int32),
        (LANDUNIT_SOIL, LANDUNIT_CROP),
    )
    if soil_column.shape != (column_count,):
        raise ValueError("restart column topology is inconsistent")

    values = load_native_variables(
        restart,
        ["harvdate", "xsmrpool_loss"],
        masked_integer_fill_value=np.iinfo(np.int32).max,
        masked_floating_fill_value=float(CLM_SPECIAL_VALUE),
    )
    harvest_date = np.asarray(values["harvdate"], dtype=np.int32)
    xsmrpool_loss = np.asarray(values["xsmrpool_loss"], dtype=np.float64)
    if harvest_date.shape != (patch_count,):
        raise ValueError("restart harvdate must contain every patch")
    if xsmrpool_loss.shape != (patch_count,):
        raise ValueError("restart xsmrpool_loss must contain every patch")

    pft_names = _pft_names(precision.parameter_data_path)
    vegetation_type = np.asarray(
        precision.inputs.vegetation_type_patch,
        dtype=np.int32,
    )
    if np.any((vegetation_type < 0) | (vegetation_type >= len(pft_names))):
        raise ValueError("restart contains a PFT index outside the parameter file")
    clock = load_restart_clock(restart)
    if clock.step_seconds <= 0:
        raise ValueError("restart radiation time step must be positive")

    cascade = DEFAULT_SOIL_BIOGEOCHEM_POTENTIAL_PARAMETERS
    return CnStateUpdate1NativeData(
        inputs=CnStateUpdate1Inputs(
            soil_patch=precision.inputs.soil_patch,
            soil_column=jnp.asarray(soil_column),
            vegetation_type_patch=precision.inputs.vegetation_type_patch,
            harvest_date_patch=jnp.asarray(harvest_date),
            cascade_donor_pool=jnp.asarray(cascade.cascade_donor_pool),
            cascade_receiver_pool=jnp.asarray(cascade.cascade_receiver_pool),
        ),
        restart_state=CnStateUpdate1State(
            vegetation=precision.state,
            xsmrpool_loss_patch=jnp.asarray(xsmrpool_loss),
        ),
        parameters=CnStateUpdate1Parameters(
            woody_pft=_woody_pft(precision.parameter_data_path, len(pft_names)),
            time_step=np.float64(clock.step_seconds),
            first_prognostic_crop_pft=_first_prognostic_crop_pft(pft_names),
            decomposition_level_count=20,
            carbon_respiration_option=_integer(
                clm_group,
                "carbon_resp_opt",
                0,
            ),
            dribble_crop_harvest_xsmrpool_to_atmosphere=_logical(
                cn_group,
                "dribble_crophrv_xsmrpool_2atm",
                False,
            ),
        ),
        precision_parameters=precision.parameters,
        restart_path=restart,
        parameter_data_path=precision.parameter_data_path,
    )


def assemble_cn_state_update1_call(
    native: CnStateUpdate1NativeData,
    first_precision_control_result: CnPrecisionControlResult,
    growth_respiration_result: GrowthRespirationResult,
) -> CnStateUpdate1Call:

    vegetation = first_precision_control_result.state
    cn_fluxes = growth_respiration_result.cn_fluxes
    patch_count = native.inputs.soil_patch.shape[0]
    column_count = native.inputs.soil_column.shape[0]
    if vegetation.paired_carbon_patch.shape[0] != patch_count:
        raise ValueError("first precision-control result has the wrong patch count")
    if cn_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("growth-respiration result has the wrong patch count")
    if cn_fluxes.column_scalar.shape[0] != column_count:
        raise ValueError("growth-respiration result has the wrong column count")
    return CnStateUpdate1Call(
        inputs=native.inputs,
        state=native.restart_state._replace(vegetation=vegetation),
        cn_fluxes=cn_fluxes,
        parameters=native.parameters,
        precision_parameters=native.precision_parameters,
    )


__all__ = [
    "CnStateUpdate1Call",
    "CnStateUpdate1NativeData",
    "assemble_cn_state_update1_call",
    "load_cn_state_update1_native_data",
]
