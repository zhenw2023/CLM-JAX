
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from clm_jax.physics.numerics import fortran_hardware_add as fortran_add

PAIRED_CARBON_FIELDS = (
    "vegetation_carbon.leafc_patch",
    "vegetation_carbon.leafc_storage_patch",
    "vegetation_carbon.leafc_xfer_patch",
    "vegetation_carbon.frootc_patch",
    "vegetation_carbon.frootc_storage_patch",
    "vegetation_carbon.frootc_xfer_patch",
    "vegetation_carbon.grainc_patch",
    "vegetation_carbon.grainc_storage_patch",
    "vegetation_carbon.grainc_xfer_patch",
    "vegetation_carbon.cropseedc_deficit_patch",
    "vegetation_carbon.livestemc_patch",
    "vegetation_carbon.livestemc_storage_patch",
    "vegetation_carbon.livestemc_xfer_patch",
    "vegetation_carbon.deadstemc_patch",
    "vegetation_carbon.deadstemc_storage_patch",
    "vegetation_carbon.deadstemc_xfer_patch",
    "vegetation_carbon.livecrootc_patch",
    "vegetation_carbon.livecrootc_storage_patch",
    "vegetation_carbon.livecrootc_xfer_patch",
    "vegetation_carbon.deadcrootc_patch",
    "vegetation_carbon.deadcrootc_storage_patch",
    "vegetation_carbon.deadcrootc_xfer_patch",
)
PAIRED_NITROGEN_FIELDS = (
    "vegetation_nitrogen.leafn_patch",
    "vegetation_nitrogen.leafn_storage_patch",
    "vegetation_nitrogen.leafn_xfer_patch",
    "vegetation_nitrogen.frootn_patch",
    "vegetation_nitrogen.frootn_storage_patch",
    "vegetation_nitrogen.frootn_xfer_patch",
    "vegetation_nitrogen.grainn_patch",
    "vegetation_nitrogen.grainn_storage_patch",
    "vegetation_nitrogen.grainn_xfer_patch",
    "vegetation_nitrogen.cropseedn_deficit_patch",
    "vegetation_nitrogen.livestemn_patch",
    "vegetation_nitrogen.livestemn_storage_patch",
    "vegetation_nitrogen.livestemn_xfer_patch",
    "vegetation_nitrogen.deadstemn_patch",
    "vegetation_nitrogen.deadstemn_storage_patch",
    "vegetation_nitrogen.deadstemn_xfer_patch",
    "vegetation_nitrogen.livecrootn_patch",
    "vegetation_nitrogen.livecrootn_storage_patch",
    "vegetation_nitrogen.livecrootn_xfer_patch",
    "vegetation_nitrogen.deadcrootn_patch",
    "vegetation_nitrogen.deadcrootn_storage_patch",
    "vegetation_nitrogen.deadcrootn_xfer_patch",
)
CARBON_ONLY_FIELDS = (
    "vegetation_carbon.gresp_storage_patch",
    "vegetation_carbon.gresp_xfer_patch",
    "vegetation_carbon.cpool_patch",
    "vegetation_carbon.xsmrpool_patch",
)
NITROGEN_ONLY_FIELDS = (
    "vegetation_nitrogen.retransn_patch",
    "vegetation_nitrogen.npool_patch",
)
CARBON_TRUNCATION_FIELD = "vegetation_carbon.ctrunc_patch"
NITROGEN_TRUNCATION_FIELD = "vegetation_nitrogen.ntrunc_patch"

_FINE_ROOT_PAIR_INDEX = 3
_CROP_PAIR_INDICES = frozenset((6, 7, 8, 9))
_ALLOW_NEGATIVE_PAIR_INDICES = frozenset((3, 9))
_CROP_CARBON_ONLY_INDICES = frozenset((3,))
_ALLOW_NEGATIVE_CARBON_ONLY_INDICES = frozenset((3,))


class CnPrecisionControlInputs(NamedTuple):

    soil_patch: jnp.ndarray
    vegetation_type_patch: jnp.ndarray


class CnPrecisionControlState(NamedTuple):

    paired_carbon_patch: jnp.ndarray
    paired_nitrogen_patch: jnp.ndarray
    carbon_only_patch: jnp.ndarray
    nitrogen_only_patch: jnp.ndarray
    carbon_truncation_patch: jnp.ndarray
    nitrogen_truncation_patch: jnp.ndarray


class CnPrecisionControlParameters(NamedTuple):
    carbon_truncation_threshold: float | jnp.ndarray = 1.0e-8
    critical_negative_carbon: float | jnp.ndarray = -60.0
    nitrogen_truncation_threshold: float | jnp.ndarray = 1.0e-9
    critical_negative_nitrogen: float | jnp.ndarray = -6.0
    use_nitrogen_guardrail: bool | jnp.ndarray = True
    use_crop: bool | jnp.ndarray = True
    first_generic_crop_pft: int | jnp.ndarray = 15


DEFAULT_CN_PRECISION_CONTROL_PARAMETERS = CnPrecisionControlParameters()


class CnPrecisionControlDiagnostics(NamedTuple):
    updated_patch_count: jnp.ndarray
    paired_truncation_count: jnp.ndarray
    carbon_only_truncation_count: jnp.ndarray
    nitrogen_only_truncation_count: jnp.ndarray
    critical_negative_pair_count: jnp.ndarray
    critical_negative_carbon_count: jnp.ndarray
    critical_negative_nitrogen_count: jnp.ndarray


class CnPrecisionControlResult(NamedTuple):
    state: CnPrecisionControlState
    diagnostics: CnPrecisionControlDiagnostics


_FIELD_LOCATIONS = {
    **{
        name: ("paired_carbon_patch", index)
        for index, name in enumerate(PAIRED_CARBON_FIELDS)
    },
    **{
        name: ("paired_nitrogen_patch", index)
        for index, name in enumerate(PAIRED_NITROGEN_FIELDS)
    },
    **{
        name: ("carbon_only_patch", index)
        for index, name in enumerate(CARBON_ONLY_FIELDS)
    },
    **{
        name: ("nitrogen_only_patch", index)
        for index, name in enumerate(NITROGEN_ONLY_FIELDS)
    },
    CARBON_TRUNCATION_FIELD: ("carbon_truncation_patch", None),
    NITROGEN_TRUNCATION_FIELD: ("nitrogen_truncation_patch", None),
}


def initialize_cn_precision_control_state(
    number_of_patches: int,
    *,
    fill_value: float = 0.0,
) -> CnPrecisionControlState:

    if number_of_patches < 0:
        raise ValueError("patch count cannot be negative")
    fill = jnp.float64(fill_value)
    return CnPrecisionControlState(
        paired_carbon_patch=jnp.full(
            (number_of_patches, len(PAIRED_CARBON_FIELDS)),
            fill,
            dtype=jnp.float64,
        ),
        paired_nitrogen_patch=jnp.full(
            (number_of_patches, len(PAIRED_NITROGEN_FIELDS)),
            fill,
            dtype=jnp.float64,
        ),
        carbon_only_patch=jnp.full(
            (number_of_patches, len(CARBON_ONLY_FIELDS)),
            fill,
            dtype=jnp.float64,
        ),
        nitrogen_only_patch=jnp.full(
            (number_of_patches, len(NITROGEN_ONLY_FIELDS)),
            fill,
            dtype=jnp.float64,
        ),
        carbon_truncation_patch=jnp.full(
            number_of_patches,
            fill,
            dtype=jnp.float64,
        ),
        nitrogen_truncation_patch=jnp.full(
            number_of_patches,
            fill,
            dtype=jnp.float64,
        ),
    )


def get_cn_precision_control_state_field(
    state: CnPrecisionControlState,
    field_name: str,
) -> jnp.ndarray:

    try:
        group, index = _FIELD_LOCATIONS[field_name]
    except KeyError as error:
        raise KeyError(f"unknown precision-control field: {field_name}") from error
    values = getattr(state, group)
    return values if index is None else values[:, index]


def set_cn_precision_control_state_field(
    state: CnPrecisionControlState,
    field_name: str,
    values: jnp.ndarray,
) -> CnPrecisionControlState:

    try:
        group, index = _FIELD_LOCATIONS[field_name]
    except KeyError as error:
        raise KeyError(f"unknown precision-control field: {field_name}") from error
    packed = getattr(state, group)
    values = jnp.asarray(values, dtype=jnp.float64)
    expected = packed.shape if index is None else packed.shape[:1]
    if values.shape != expected:
        raise ValueError(f"{field_name} has shape {values.shape}, expected {expected}")
    changed = values if index is None else packed.at[:, index].set(values)
    return state._replace(**{group: changed})


def replace_cn_precision_control_state_fields(
    state: CnPrecisionControlState,
    fields: dict[str, jnp.ndarray],
) -> CnPrecisionControlState:
    groups = {}
    for name, value in fields.items():
        group, index = _FIELD_LOCATIONS[name]
        packed = getattr(state, group)
        value = jnp.asarray(value, dtype=jnp.float64)
        expected = packed.shape if index is None else packed.shape[:1]
        if value.shape != expected:
            raise ValueError(f"{name} has shape {value.shape}, expected {expected}")
        if index is None:
            groups[group] = value
        else:
            if group not in groups:
                groups[group] = [packed[:, k] for k in range(packed.shape[1])]
            groups[group][index] = value
    return state._replace(**{
        name: jnp.stack(values, axis=1) if isinstance(values, list) else values
        for name, values in groups.items()
    })


def _validate_shapes(
    inputs: CnPrecisionControlInputs,
    state: CnPrecisionControlState,
    parameters: CnPrecisionControlParameters,
) -> int:
    if jnp.ndim(inputs.soil_patch) != 1:
        raise ValueError("soil_patch must be a vector")
    patch_count = inputs.soil_patch.shape[0]
    if jnp.shape(inputs.vegetation_type_patch) != (patch_count,):
        raise ValueError("vegetation_type_patch must contain every patch")
    expected = {
        "paired_carbon_patch": (patch_count, len(PAIRED_CARBON_FIELDS)),
        "paired_nitrogen_patch": (patch_count, len(PAIRED_NITROGEN_FIELDS)),
        "carbon_only_patch": (patch_count, len(CARBON_ONLY_FIELDS)),
        "nitrogen_only_patch": (patch_count, len(NITROGEN_ONLY_FIELDS)),
        "carbon_truncation_patch": (patch_count,),
        "nitrogen_truncation_patch": (patch_count,),
    }
    for name, shape in expected.items():
        if jnp.shape(getattr(state, name)) != shape:
            raise ValueError(f"{name} must have shape {shape}")
    for name in parameters._fields:
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    return patch_count


def _cn_precision_control_reference(
    inputs: CnPrecisionControlInputs,
    state: CnPrecisionControlState,
    parameters: CnPrecisionControlParameters = (
        DEFAULT_CN_PRECISION_CONTROL_PARAMETERS
    ),
) -> CnPrecisionControlResult:

    _validate_shapes(inputs, state, parameters)
    soil = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    vegetation_type = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    first_crop = jnp.asarray(
        parameters.first_generic_crop_pft,
        dtype=jnp.int32,
    )
    crop = soil & (vegetation_type >= first_crop)
    use_crop = jnp.asarray(parameters.use_crop, dtype=jnp.bool_)
    guardrail = jnp.asarray(parameters.use_nitrogen_guardrail, dtype=jnp.bool_)
    carbon_threshold = jnp.asarray(
        parameters.carbon_truncation_threshold,
        dtype=jnp.float64,
    )
    nitrogen_threshold = jnp.asarray(
        parameters.nitrogen_truncation_threshold,
        dtype=jnp.float64,
    )
    critical_carbon = jnp.asarray(
        parameters.critical_negative_carbon,
        dtype=jnp.float64,
    )
    critical_nitrogen = jnp.asarray(
        parameters.critical_negative_nitrogen,
        dtype=jnp.float64,
    )
    zero = jnp.float64(0.0)

    paired_carbon = jnp.asarray(state.paired_carbon_patch, dtype=jnp.float64)
    paired_nitrogen = jnp.asarray(
        state.paired_nitrogen_patch,
        dtype=jnp.float64,
    )
    carbon_only = jnp.asarray(state.carbon_only_patch, dtype=jnp.float64)
    nitrogen_only = jnp.asarray(state.nitrogen_only_patch, dtype=jnp.float64)
    carbon_correction = jnp.zeros_like(state.carbon_truncation_patch)
    nitrogen_correction = jnp.zeros_like(state.nitrogen_truncation_patch)
    paired_truncations = jnp.int32(0)
    carbon_truncations = jnp.int32(0)
    nitrogen_truncations = jnp.int32(0)
    critical_pairs = jnp.int32(0)
    critical_carbon_count = jnp.int32(0)
    critical_nitrogen_count = jnp.int32(0)

    for index in range(len(PAIRED_CARBON_FIELDS)):
        enabled = soil
        if index == _FINE_ROOT_PAIR_INDEX:
            enabled = enabled & ~guardrail
        if index in _CROP_PAIR_INDICES:
            enabled = enabled & use_crop & crop
        allow_negative = index in _ALLOW_NEGATIVE_PAIR_INDICES
        carbon = paired_carbon[:, index]
        nitrogen = paired_nitrogen[:, index]
        critical = enabled & ~jnp.asarray(allow_negative) & (
            (carbon < critical_carbon) | (nitrogen < critical_nitrogen)
        )
        truncate = enabled & ~critical & (
            (jnp.abs(carbon) < carbon_threshold)
            | (guardrail & (jnp.abs(nitrogen) < nitrogen_threshold))
        )
        carbon_correction = fortran_add(
            carbon_correction,
            jnp.where(truncate, carbon, zero),
        )
        nitrogen_correction = fortran_add(
            nitrogen_correction,
            jnp.where(truncate, nitrogen, zero),
        )
        paired_carbon = paired_carbon.at[:, index].set(
            jnp.where(truncate, zero, carbon)
        )
        paired_nitrogen = paired_nitrogen.at[:, index].set(
            jnp.where(truncate, zero, nitrogen)
        )
        paired_truncations = paired_truncations + jnp.sum(
            truncate,
            dtype=jnp.int32,
        )
        critical_pairs = critical_pairs + jnp.sum(critical, dtype=jnp.int32)

    for index in range(len(CARBON_ONLY_FIELDS)):
        enabled = soil
        if index in _CROP_CARBON_ONLY_INDICES:
            enabled = enabled & use_crop & crop
        allow_negative = index in _ALLOW_NEGATIVE_CARBON_ONLY_INDICES
        carbon = carbon_only[:, index]
        critical = (
            enabled
            & ~jnp.asarray(allow_negative)
            & (carbon < critical_carbon)
        )
        truncate = (
            enabled
            & ~critical
            & (jnp.abs(carbon) < carbon_threshold)
        )
        carbon_correction = fortran_add(
            carbon_correction,
            jnp.where(truncate, carbon, zero),
        )
        carbon_only = carbon_only.at[:, index].set(
            jnp.where(truncate, zero, carbon)
        )
        carbon_truncations = carbon_truncations + jnp.sum(
            truncate,
            dtype=jnp.int32,
        )
        critical_carbon_count = critical_carbon_count + jnp.sum(
            critical,
            dtype=jnp.int32,
        )

    for index in range(len(NITROGEN_ONLY_FIELDS)):
        nitrogen = nitrogen_only[:, index]
        critical = soil & (nitrogen < critical_nitrogen)
        truncate = soil & ~critical & (jnp.abs(nitrogen) < nitrogen_threshold)
        nitrogen_correction = fortran_add(
            nitrogen_correction,
            jnp.where(truncate, nitrogen, zero),
        )
        nitrogen_only = nitrogen_only.at[:, index].set(
            jnp.where(truncate, zero, nitrogen)
        )
        nitrogen_truncations = nitrogen_truncations + jnp.sum(
            truncate,
            dtype=jnp.int32,
        )
        critical_nitrogen_count = critical_nitrogen_count + jnp.sum(
            critical,
            dtype=jnp.int32,
        )

    carbon_truncation = fortran_add(
        state.carbon_truncation_patch,
        carbon_correction,
    )
    nitrogen_truncation = fortran_add(
        state.nitrogen_truncation_patch,
        nitrogen_correction,
    )
    carbon_truncation = jnp.where(
        soil,
        carbon_truncation,
        state.carbon_truncation_patch,
    )
    nitrogen_truncation = jnp.where(
        soil,
        nitrogen_truncation,
        state.nitrogen_truncation_patch,
    )

    return CnPrecisionControlResult(
        state=CnPrecisionControlState(
            paired_carbon_patch=paired_carbon,
            paired_nitrogen_patch=paired_nitrogen,
            carbon_only_patch=carbon_only,
            nitrogen_only_patch=nitrogen_only,
            carbon_truncation_patch=carbon_truncation,
            nitrogen_truncation_patch=nitrogen_truncation,
        ),
        diagnostics=CnPrecisionControlDiagnostics(
            updated_patch_count=jnp.sum(soil, dtype=jnp.int32),
            paired_truncation_count=paired_truncations,
            carbon_only_truncation_count=carbon_truncations,
            nitrogen_only_truncation_count=nitrogen_truncations,
            critical_negative_pair_count=critical_pairs,
            critical_negative_carbon_count=critical_carbon_count,
            critical_negative_nitrogen_count=critical_nitrogen_count,
        ),
    )


def cn_precision_control(
    inputs: CnPrecisionControlInputs,
    state: CnPrecisionControlState,
    parameters: CnPrecisionControlParameters = DEFAULT_CN_PRECISION_CONTROL_PARAMETERS,
) -> CnPrecisionControlResult:
    if jax.default_backend() == "gpu" and inputs.soil_patch.size:
        from clm_jax.physics.cn_patch_state_gpu import precision_control

        return precision_control(inputs, state, parameters)
    return _cn_precision_control_reference(inputs, state, parameters)


__all__ = [
    "CARBON_ONLY_FIELDS",
    "CARBON_TRUNCATION_FIELD",
    "DEFAULT_CN_PRECISION_CONTROL_PARAMETERS",
    "NITROGEN_ONLY_FIELDS",
    "NITROGEN_TRUNCATION_FIELD",
    "PAIRED_CARBON_FIELDS",
    "PAIRED_NITROGEN_FIELDS",
    "CnPrecisionControlDiagnostics",
    "CnPrecisionControlInputs",
    "CnPrecisionControlParameters",
    "CnPrecisionControlResult",
    "CnPrecisionControlState",
    "cn_precision_control",
    "get_cn_precision_control_state_field",
    "initialize_cn_precision_control_state",
    "set_cn_precision_control_state_field",
]
