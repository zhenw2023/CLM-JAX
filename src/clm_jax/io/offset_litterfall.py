
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
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
from clm_jax.physics.cn_zero import CnTransientFluxState
from clm_jax.physics.offset_litterfall import (
    OffsetLitterfallInputs,
    OffsetLitterfallParameters,
    OffsetLitterfallState,
)
from clm_jax.physics.soil_biogeochem_decomp import SoilBiogeochemDecompResult

_RESTART_VARIABLES = (
    "offset_flag",
    "offset_counter",
    "leafc",
    "frootc",
    "grainc",
    "cropseedc_deficit",
    "livestemc",
    "leafn",
    "frootn",
    "grainn",
    "cropseedn_deficit",
    "livestemn",
    "prev_leafc_to_litter",
    "prev_frootc_to_litter",
    "leafc_to_litter_fun",
    "leafcn_offset",
)
_PARAMETER_VARIABLES = ("leafcn", "lflitcn", "frootcn")


class OffsetLitterfallNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: OffsetLitterfallInputs
    state: OffsetLitterfallState
    parameters: OffsetLitterfallParameters
    restart_path: Path
    parameter_data_path: Path


class OffsetLitterfallCall(NamedTuple):

    inputs: OffsetLitterfallInputs
    state: OffsetLitterfallState
    cn_fluxes: CnTransientFluxState
    parameters: OffsetLitterfallParameters


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


def _pft_names(parameter_data: Path) -> tuple[str, ...]:
    from netCDF4 import Dataset  

    with Dataset(parameter_data, mode="r") as dataset:
        if "pftname" not in dataset.variables:
            raise ValueError("CLM parameter file lacks pftname")
        raw = np.asarray(dataset.variables["pftname"][:])
    if raw.ndim != 2 or raw.dtype.kind not in {"S", "U"}:
        raise ValueError("CLM pftname must be a character matrix")
    if raw.dtype.kind == "S":
        names = tuple(
            b"".join(np.asarray(row, dtype="S1").tolist()).decode("ascii").rstrip()
            for row in raw
        )
    else:
        names = tuple("".join(np.asarray(row, dtype=str).tolist()).rstrip() for row in raw)
    if not names or len(set(names)) != len(names):
        raise ValueError("CLM pftname entries must be nonempty and unique")
    return names


def _parameter_vector(
    values: dict[str, jnp.ndarray],
    name: str,
    pft_count: int,
) -> jnp.ndarray:
    host = np.asarray(values[name], dtype=np.float64)
    if host.shape != (pft_count,) or not np.all(np.isfinite(host)):
        raise ValueError(f"CLM parameter {name!r} must be one finite PFT vector")
    if np.any(host <= 0.0):
        raise ValueError(f"CLM C:N parameter {name!r} must be positive")
    return jnp.asarray(host)


def load_offset_litterfall_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> OffsetLitterfallNativeData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    required = {
        "use_cn": True,
        "use_crop": True,
        "use_fates": False,
        "use_fun": True,
    }
    for name, expected in required.items():
        if general.get(name) != expected:
            raise ValueError(
                f"native offset-litterfall adapter requires {name}={expected!r}"
            )
    nitrogen = read_case_namelist_group(case, "lnd_in", "clm_nitrogen")
    if nitrogen.get("cnratio_floating") is not True:
        raise ValueError(
            "native offset-litterfall adapter requires cnratio_floating=True"
        )

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
    active = np.asarray(topology.patch.active, dtype=bool)
    landunit = np.asarray(topology.patch.landunit_type, dtype=np.int32)
    soil_patch = active & (
        (landunit == LANDUNIT_SOIL) | (landunit == LANDUNIT_CROP)
    )
    vegetation_type = np.asarray(topology.patch.vegetation_type, dtype=np.int32)

    restart_values = load_native_variables(
        restart,
        list(_RESTART_VARIABLES),
        dtype=np.float64,
        masked_floating_fill_value=float(CLM_SPECIAL_VALUE),
    )
    for name in _RESTART_VARIABLES:
        host = np.asarray(restart_values[name], dtype=np.float64)
        if host.shape != (patch_count,):
            raise ValueError(f"restart variable {name!r} must contain every patch")
        if not np.all(np.isfinite(host[soil_patch])):
            raise ValueError(f"active soil patches contain non-finite {name!r}")

    parameter_values = load_native_variables(
        parameter_data,
        list(_PARAMETER_VARIABLES),
        dtype=np.float64,
    )
    pft_names = _pft_names(parameter_data)
    pft_count = len(pft_names)
    try:
        first_crop_pft = pft_names.index("temperate_corn")
    except ValueError as error:
        raise ValueError("CLM parameter file lacks temperate_corn") from error
    if np.any((vegetation_type < 0) | (vegetation_type >= pft_count)):
        raise ValueError("restart vegetation type lies outside native PFT parameters")

    clock = load_restart_clock(restart)
    return OffsetLitterfallNativeData(
        topology=topology,
        inputs=OffsetLitterfallInputs(
            soil_patch=jnp.asarray(soil_patch),
            vegetation_type_patch=jnp.asarray(vegetation_type),
            offset_flag_patch=jnp.asarray(restart_values["offset_flag"]),
            offset_counter_seconds_patch=jnp.asarray(
                restart_values["offset_counter"]
            ),
            leaf_carbon_patch=jnp.asarray(restart_values["leafc"]),
            fine_root_carbon_patch=jnp.asarray(restart_values["frootc"]),
            grain_carbon_patch=jnp.asarray(restart_values["grainc"]),
            crop_seed_carbon_deficit_patch=jnp.asarray(
                restart_values["cropseedc_deficit"]
            ),
            live_stem_carbon_patch=jnp.asarray(restart_values["livestemc"]),
            leaf_nitrogen_patch=jnp.asarray(restart_values["leafn"]),
            fine_root_nitrogen_patch=jnp.asarray(restart_values["frootn"]),
            grain_nitrogen_patch=jnp.asarray(restart_values["grainn"]),
            crop_seed_nitrogen_deficit_patch=jnp.asarray(
                restart_values["cropseedn_deficit"]
            ),
            live_stem_nitrogen_patch=jnp.asarray(restart_values["livestemn"]),
        ),
        state=OffsetLitterfallState(
            previous_leaf_carbon_to_litter_patch=jnp.asarray(
                restart_values["prev_leafc_to_litter"]
            ),
            previous_fine_root_carbon_to_litter_patch=jnp.asarray(
                restart_values["prev_frootc_to_litter"]
            ),
            leaf_carbon_to_litter_fun_patch=jnp.asarray(
                restart_values["leafc_to_litter_fun"]
            ),
            leaf_carbon_nitrogen_offset_patch=jnp.asarray(
                restart_values["leafcn_offset"]
            ),
        ),
        parameters=OffsetLitterfallParameters(
            leaf_carbon_nitrogen_ratio_pft=_parameter_vector(
                parameter_values,
                "leafcn",
                pft_count,
            ),
            leaf_litter_carbon_nitrogen_ratio_pft=_parameter_vector(
                parameter_values,
                "lflitcn",
                pft_count,
            ),
            fine_root_carbon_nitrogen_ratio_pft=_parameter_vector(
                parameter_values,
                "frootcn",
                pft_count,
            ),
            first_crop_pft=np.int32(first_crop_pft),
            use_fun=True,
            carbon_nitrogen_ratio_floating=True,
            time_step_seconds=np.float64(clock.step_seconds),
        ),
        restart_path=restart,
        parameter_data_path=parameter_data,
    )


def assemble_offset_litterfall_call(
    native: OffsetLitterfallNativeData,
    actual_decomposition_result: SoilBiogeochemDecompResult,
) -> OffsetLitterfallCall:

    patch_count = native.inputs.soil_patch.shape[0]
    final_fluxes = actual_decomposition_result.state.cn_fluxes
    if final_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("actual-decomposition result has the wrong patch count")
    return OffsetLitterfallCall(
        inputs=native.inputs,
        state=native.state,
        cn_fluxes=final_fluxes,
        parameters=native.parameters,
    )


__all__ = [
    "OffsetLitterfallCall",
    "OffsetLitterfallNativeData",
    "assemble_offset_litterfall_call",
    "load_offset_litterfall_native_data",
]
