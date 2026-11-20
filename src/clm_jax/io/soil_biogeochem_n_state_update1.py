
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.cn_state_update1 import (
    CnStateUpdate1NativeData,
    load_cn_state_update1_native_data,
)
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.physics.cn_state_update1 import CnStateUpdate1WithPrecisionResult
from clm_jax.physics.cn_zero import CnTransientFluxState
from clm_jax.physics.soil_biogeochem_n_state_update1 import (
    SoilBiogeochemNStateUpdate1Inputs,
    SoilBiogeochemNStateUpdate1Parameters,
    SoilBiogeochemNStateUpdate1State,
)
from clm_jax.physics.soil_biogeochem_vertical_profile import (
    SoilBiogeochemVerticalProfileResult,
)

NUMBER_OF_GROUND_LEVELS = 25
NUMBER_OF_DECOMPOSITION_LEVELS = 20
NITRIFICATION_N2O_LOSS_FRACTION = np.float64(6.0e-4)


class SoilBiogeochemNStateUpdate1NativeData(NamedTuple):

    soil_column: jnp.ndarray
    restart_state: SoilBiogeochemNStateUpdate1State
    parameters: SoilBiogeochemNStateUpdate1Parameters
    restart_path: Path


class SoilBiogeochemNStateUpdate1Call(NamedTuple):

    inputs: SoilBiogeochemNStateUpdate1Inputs
    state: SoilBiogeochemNStateUpdate1State
    cn_fluxes: CnTransientFluxState
    parameters: SoilBiogeochemNStateUpdate1Parameters


def _require_same_restart(
    requested: str | Path | None,
    selected: Path,
) -> None:
    if requested is None:
        return
    if Path(requested).expanduser().resolve() != selected:
        raise ValueError("CN Update1 native data uses a different restart file")


def load_soil_biogeochem_n_state_update1_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    *,
    cn_state_update1_native_data: CnStateUpdate1NativeData | None = None,
) -> SoilBiogeochemNStateUpdate1NativeData:

    cn_native = cn_state_update1_native_data
    if cn_native is None:
        cn_native = load_cn_state_update1_native_data(case, restart_path)
    else:
        _require_same_restart(restart_path, cn_native.restart_path)

    general = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    required = {
        "use_cn": True,
        "use_crop": True,
        "use_fun": True,
        "use_nitrif_denitrif": True,
        "use_century_decomp": True,
        "use_vertsoilc": True,
        "soil_layerstruct": "20SL_8.5m",
    }
    for name, expected in required.items():
        if general.get(name) != expected:
            raise ValueError(
                "native soil-N Update1 adapter requires "
                f"{name}={expected!r}"
            )

    restart = cn_native.restart_path
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    dimensions = load_dimension_sizes(restart)
    if dimensions.get("levgrnd") != NUMBER_OF_GROUND_LEVELS:
        raise ValueError("soil-N Update1 restart must contain 25 ground levels")

    soil_column = np.asarray(cn_native.inputs.soil_column, dtype=bool)
    column_count = soil_column.size
    raw = load_native_variables(
        restart,
        ["sminn_vr", "smin_nh4_vr", "smin_no3_vr"],
        dtype=np.float64,
    )
    states: dict[str, np.ndarray] = {}
    for name, values in raw.items():
        state = np.asarray(values, dtype=np.float64)
        if state.shape != (column_count, NUMBER_OF_GROUND_LEVELS):
            raise ValueError(f"restart {name} must have shape (column, 25)")
        if not np.all(np.isfinite(state[soil_column])):
            raise ValueError(f"filtered soil columns contain missing {name}")
        states[name] = state

    return SoilBiogeochemNStateUpdate1NativeData(
        soil_column=jnp.asarray(soil_column),
        restart_state=SoilBiogeochemNStateUpdate1State(
            mineral_nitrogen_column_level=jnp.asarray(states["sminn_vr"]),
            ammonium_column_level=jnp.asarray(states["smin_nh4_vr"]),
            nitrate_column_level=jnp.asarray(states["smin_no3_vr"]),
        ),
        parameters=SoilBiogeochemNStateUpdate1Parameters(
            time_step=cn_native.parameters.time_step,
            decomposition_level_count=NUMBER_OF_DECOMPOSITION_LEVELS,
            nitrification_n2o_loss_fraction=NITRIFICATION_N2O_LOSS_FRACTION,
            cascade_donor_pool=cn_native.inputs.cascade_donor_pool,
            cascade_receiver_pool=cn_native.inputs.cascade_receiver_pool,
        ),
        restart_path=restart,
    )


def assemble_soil_biogeochem_n_state_update1_call(
    native: SoilBiogeochemNStateUpdate1NativeData,
    vertical_profile_result: SoilBiogeochemVerticalProfileResult,
    cn_state_update1_result: CnStateUpdate1WithPrecisionResult,
) -> SoilBiogeochemNStateUpdate1Call:

    column_count = native.soil_column.shape[0]
    expected_level_shape = (column_count, NUMBER_OF_GROUND_LEVELS)
    profile_state = vertical_profile_result.state
    deposition = profile_state.nitrogen_deposition_profile_column_level
    fixation = profile_state.nitrogen_fixation_profile_column_level
    if deposition.shape != expected_level_shape:
        raise ValueError("vertical-profile result has the wrong column count")
    if fixation.shape != expected_level_shape:
        raise ValueError("vertical-profile result has the wrong column count")

    cn_fluxes = cn_state_update1_result.cn_fluxes
    if cn_fluxes.column_scalar.shape[0] != column_count:
        raise ValueError("CN Update1 result has the wrong column count")
    if cn_fluxes.column_level.shape[:2] != expected_level_shape:
        raise ValueError("CN Update1 column-level flux extent is incompatible")

    return SoilBiogeochemNStateUpdate1Call(
        inputs=SoilBiogeochemNStateUpdate1Inputs(
            soil_column=native.soil_column,
            nitrogen_deposition_profile_column_level=deposition,
            nitrogen_fixation_profile_column_level=fixation,
        ),
        state=native.restart_state,
        cn_fluxes=cn_fluxes,
        parameters=native.parameters,
    )


__all__ = [
    "NITRIFICATION_N2O_LOSS_FRACTION",
    "SoilBiogeochemNStateUpdate1Call",
    "SoilBiogeochemNStateUpdate1NativeData",
    "assemble_soil_biogeochem_n_state_update1_call",
    "load_soil_biogeochem_n_state_update1_native_data",
]
