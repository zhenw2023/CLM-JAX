
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.c_state_update0 import (
    CStateUpdate0NativeData,
    load_c_state_update0_native_data,
)
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.physics.c_state_update0 import CStateUpdate0Result
from clm_jax.physics.cn_precision_control import (
    CARBON_ONLY_FIELDS,
    CARBON_TRUNCATION_FIELD,
    NITROGEN_ONLY_FIELDS,
    NITROGEN_TRUNCATION_FIELD,
    PAIRED_CARBON_FIELDS,
    PAIRED_NITROGEN_FIELDS,
    CnPrecisionControlInputs,
    CnPrecisionControlParameters,
    CnPrecisionControlResult,
    CnPrecisionControlState,
    set_cn_precision_control_state_field,
)

_ALL_FIELDS = (
    *PAIRED_CARBON_FIELDS,
    *PAIRED_NITROGEN_FIELDS,
    *CARBON_ONLY_FIELDS,
    *NITROGEN_ONLY_FIELDS,
    CARBON_TRUNCATION_FIELD,
    NITROGEN_TRUNCATION_FIELD,
)
_RESTART_OVERRIDES = {
    CARBON_TRUNCATION_FIELD: "pft_ctrunc",
    NITROGEN_TRUNCATION_FIELD: "pft_ntrunc",
}


class CnPrecisionControlNativeData(NamedTuple):

    inputs: CnPrecisionControlInputs
    state: CnPrecisionControlState
    parameters: CnPrecisionControlParameters
    restart_path: Path
    parameter_data_path: Path


class CnPrecisionControlCall(NamedTuple):

    inputs: CnPrecisionControlInputs
    state: CnPrecisionControlState
    parameters: CnPrecisionControlParameters


def _restart_variable(field_name: str) -> str:
    if field_name in _RESTART_OVERRIDES:
        return _RESTART_OVERRIDES[field_name]
    return field_name.split(".", maxsplit=1)[1].removesuffix("_patch")


def _logical(group: dict[str, object], name: str, default: bool) -> bool:
    value = group.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"CLM namelist value {name!r} must be logical")
    return value


def _threshold(
    group: dict[str, object],
    name: str,
    default: float,
) -> np.float64:
    value = group.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"CLM namelist value {name!r} must be real")
    result = np.float64(value)
    if not np.isfinite(result):
        raise ValueError(f"CLM namelist value {name!r} must be finite")
    return result


def _first_generic_crop_pft(parameter_data_path: Path) -> int:
    from netCDF4 import Dataset  

    with Dataset(parameter_data_path, mode="r") as dataset:
        if "pftname" not in dataset.variables:
            raise ValueError("CLM parameter file lacks pftname")
        raw = np.asarray(dataset.variables["pftname"][:])
    if raw.ndim != 2 or raw.dtype.kind not in {"S", "U"}:
        raise ValueError("CLM pftname must be a character matrix")
    if raw.dtype.kind == "S":
        names = tuple(
            b"".join(np.asarray(row, dtype="S1").tolist())
            .decode("ascii")
            .rstrip()
            for row in raw
        )
    else:
        names = tuple(
            "".join(np.asarray(row, dtype=str).tolist()).rstrip()
            for row in raw
        )
    matches = tuple(index for index, name in enumerate(names) if name == "c3_crop")
    if len(matches) != 1:
        raise ValueError("CLM pftname must contain exactly one 'c3_crop' entry")
    return matches[0]


def _pack_state(
    values: dict[str, jnp.ndarray],
    patch_count: int,
    soil_patch: np.ndarray,
) -> CnPrecisionControlState:
    def stack(fields: tuple[str, ...]) -> jnp.ndarray:
        host_fields = []
        for field_name in fields:
            variable_name = _restart_variable(field_name)
            host = np.asarray(values[variable_name], dtype=np.float64)
            if host.shape != (patch_count,):
                raise ValueError(
                    f"restart variable {variable_name!r} must contain every patch"
                )
            if not np.all(np.isfinite(host[soil_patch])):
                raise ValueError(
                    f"active soil patches contain non-finite {variable_name!r}"
                )
            host_fields.append(host)
        return jnp.asarray(np.stack(host_fields, axis=1))

    carbon_truncation = np.asarray(
        values[_restart_variable(CARBON_TRUNCATION_FIELD)],
        dtype=np.float64,
    )
    nitrogen_truncation = np.asarray(
        values[_restart_variable(NITROGEN_TRUNCATION_FIELD)],
        dtype=np.float64,
    )
    for variable_name, array in (
        ("pft_ctrunc", carbon_truncation),
        ("pft_ntrunc", nitrogen_truncation),
    ):
        if array.shape != (patch_count,):
            raise ValueError(
                f"restart variable {variable_name!r} must contain every patch"
            )
        if not np.all(np.isfinite(array[soil_patch])):
            raise ValueError(
                f"active soil patches contain non-finite {variable_name!r}"
            )

    return CnPrecisionControlState(
        paired_carbon_patch=stack(PAIRED_CARBON_FIELDS),
        paired_nitrogen_patch=stack(PAIRED_NITROGEN_FIELDS),
        carbon_only_patch=stack(CARBON_ONLY_FIELDS),
        nitrogen_only_patch=stack(NITROGEN_ONLY_FIELDS),
        carbon_truncation_patch=jnp.asarray(carbon_truncation),
        nitrogen_truncation_patch=jnp.asarray(nitrogen_truncation),
    )


def load_cn_precision_control_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    c_state_update0_native_data: CStateUpdate0NativeData | None = None,
) -> CnPrecisionControlNativeData:

    carbon_update = c_state_update0_native_data
    if carbon_update is None:
        carbon_update = load_c_state_update0_native_data(
            case,
            restart_path=restart_path,
            parameter_data_path=parameter_data_path,
        )
    else:
        if (
            restart_path is not None
            and Path(restart_path).expanduser().resolve()
            != carbon_update.restart_path
        ):
            raise ValueError("carbon-state native data uses a different restart file")
        if (
            parameter_data_path is not None
            and Path(parameter_data_path).expanduser().resolve()
            != carbon_update.parameter_data_path
        ):
            raise ValueError(
                "carbon-state native data uses a different parameter-data file"
            )

    clm_group = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    precision_group = dict(
        read_case_namelist_group(case, "lnd_in", "cnprecision_inparm")
    )
    if not _logical(clm_group, "use_cn", False):
        raise ValueError("CNPrecisionControl requires use_cn=.true.")
    if _logical(clm_group, "use_c13", False) or _logical(
        clm_group,
        "use_c14",
        False,
    ):
        raise NotImplementedError(
            "this adapter currently covers the case's inactive C13/C14 paths"
        )

    carbon_threshold = _threshold(precision_group, "ccrit", 1.0e-8)
    critical_carbon = _threshold(precision_group, "cnegcrit", -60.0)
    nitrogen_threshold = _threshold(precision_group, "ncrit", 1.0e-8)
    critical_nitrogen = _threshold(precision_group, "nnegcrit", -6.0)
    if carbon_threshold <= 0.0 or nitrogen_threshold <= 0.0:
        raise ValueError("precision-control truncation thresholds must be positive")
    if critical_carbon > -carbon_threshold:
        raise ValueError("cnegcrit must be less than or equal to -ccrit")
    soil = np.asarray(carbon_update.inputs.soil_patch, dtype=bool)
    if soil.ndim != 1:
        raise ValueError("native soil patch filter must be a vector")
    variable_names = [
        *(_restart_variable(name) for name in _ALL_FIELDS),
        "pfts1d_itypveg",
    ]
    values = load_native_variables(
        carbon_update.restart_path,
        variable_names,
        dtype=np.float64,
        masked_floating_fill_value=float(CLM_SPECIAL_VALUE),
    )
    vegetation_type = np.asarray(values["pfts1d_itypveg"], dtype=np.int32)
    if vegetation_type.shape != soil.shape:
        raise ValueError("native patch filters have inconsistent shapes")
    state = _pack_state(values, soil.size, soil)

    return CnPrecisionControlNativeData(
        inputs=CnPrecisionControlInputs(
            soil_patch=carbon_update.inputs.soil_patch,
            vegetation_type_patch=jnp.asarray(vegetation_type),
        ),
        state=state,
        parameters=CnPrecisionControlParameters(
            carbon_truncation_threshold=carbon_threshold,
            critical_negative_carbon=critical_carbon,
            nitrogen_truncation_threshold=nitrogen_threshold,
            critical_negative_nitrogen=critical_nitrogen,
            use_nitrogen_guardrail=_logical(
                clm_group,
                "use_nguardrail",
                False,
            ),
            use_crop=_logical(clm_group, "use_crop", False),
            first_generic_crop_pft=_first_generic_crop_pft(
                carbon_update.parameter_data_path
            ),
        ),
        restart_path=carbon_update.restart_path,
        parameter_data_path=carbon_update.parameter_data_path,
    )


def assemble_cn_precision_control_call(
    native: CnPrecisionControlNativeData,
    c_state_update0_result: CStateUpdate0Result,
) -> CnPrecisionControlCall:

    carbon_pool = c_state_update0_result.state.carbon_pool_patch
    patch_count = native.inputs.soil_patch.shape[0]
    if carbon_pool.shape != (patch_count,):
        raise ValueError("carbon-state result has the wrong patch count")
    state = set_cn_precision_control_state_field(
        native.state,
        "vegetation_carbon.cpool_patch",
        carbon_pool,
    )
    return CnPrecisionControlCall(
        inputs=native.inputs,
        state=state,
        parameters=native.parameters,
    )


def require_valid_cn_precision_control_result(
    result: CnPrecisionControlResult,
) -> None:

    paired = int(np.asarray(result.diagnostics.critical_negative_pair_count))
    carbon = int(np.asarray(result.diagnostics.critical_negative_carbon_count))
    if paired or carbon:
        raise ValueError(
            "CNPrecisionControl found critically negative state: "
            f"{paired} paired C/N entries and {carbon} C-only entries"
        )


__all__ = [
    "CnPrecisionControlCall",
    "CnPrecisionControlNativeData",
    "assemble_cn_precision_control_call",
    "load_cn_precision_control_native_data",
    "require_valid_cn_precision_control_result",
]
