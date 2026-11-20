
from __future__ import annotations

import calendar as calendar_module
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
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.restart import advance_restart_clock, load_restart_clock
from clm_jax.io.soil_biogeochem_vertical_profile import (
    SoilBiogeochemVerticalProfileNativeData,
    load_soil_biogeochem_vertical_profile_native_data,
)
from clm_jax.physics.cn_gap_mortality import (
    CnGapMortalityInputs,
    CnGapMortalityParameters,
    CnGapMortalityState,
)
from clm_jax.physics.cn_state_update1 import CnStateUpdate1WithPrecisionResult
from clm_jax.physics.cn_zero import CnTransientFluxState
from clm_jax.physics.soil_biogeochem_litter_vertical_transport import (
    SoilBiogeochemLitterVerticalTransportResult,
)
from clm_jax.physics.soil_biogeochem_vertical_profile import (
    NUMBER_OF_GROUND_LEVELS,
    SoilBiogeochemVerticalProfileResult,
)


class CnGapMortalityPersistentInputs(NamedTuple):

    soil_patch: jnp.ndarray
    soil_column: jnp.ndarray
    column_patch_indices: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    patch_weight_on_column: jnp.ndarray
    patch_active: jnp.ndarray
    growth_efficiency_patch: jnp.ndarray
    heat_stress_patch: jnp.ndarray


class CnGapMortalityNativeData(NamedTuple):

    persistent_inputs: CnGapMortalityPersistentInputs
    restart_state: CnGapMortalityState
    parameters: CnGapMortalityParameters
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path


class CnGapMortalityCall(NamedTuple):

    inputs: CnGapMortalityInputs
    state: CnGapMortalityState
    cn_fluxes: CnTransientFluxState
    parameters: CnGapMortalityParameters


_PFT_PARAMETER_NAMES = (
    "woody",
    "lf_flab",
    "lf_fcel",
    "lf_flig",
    "fr_flab",
    "fr_fcel",
    "fr_flig",
)


def _require_same_selected_file(
    name: str,
    requested: str | Path | None,
    actual: Path,
) -> None:
    if requested is None:
        return
    if Path(requested).expanduser().resolve() != actual:
        raise ValueError(f"native dependency uses a different {name} file")


def _validate_dependencies(
    update: CnStateUpdate1NativeData,
    vertical: SoilBiogeochemVerticalProfileNativeData,
) -> None:
    if update.restart_path != vertical.restart_path:
        raise ValueError("Update1 and vertical-profile dependencies use different restarts")
    if update.parameter_data_path != vertical.parameter_data_path:
        raise ValueError("Update1 and vertical-profile dependencies use different parameter files")
    if not np.array_equal(
        np.asarray(update.inputs.soil_patch, dtype=bool),
        np.asarray(vertical.soil_patch, dtype=bool),
    ):
        raise ValueError("native dependencies disagree on the soil patch filter")
    if not np.array_equal(
        np.asarray(update.inputs.soil_column, dtype=bool),
        np.asarray(vertical.soil_column, dtype=bool),
    ):
        raise ValueError("native dependencies disagree on the soil column filter")
    if not np.array_equal(
        np.asarray(update.inputs.vegetation_type_patch, dtype=np.int32),
        np.asarray(vertical.topology.patch.vegetation_type, dtype=np.int32),
    ):
        raise ValueError("native dependencies disagree on patch PFT identity")


def _days_per_year(case: CimeCase, year: int) -> float:
    calendar_name = case.value("CALENDAR").strip().upper()
    if calendar_name == "NO_LEAP":
        return 365.0
    if calendar_name == "GREGORIAN":
        return 366.0 if calendar_module.isleap(year) else 365.0
    if calendar_name == "360_DAY":
        return 360.0
    raise ValueError(f"unsupported CLM calendar for gap mortality: {calendar_name!r}")


def _parameter_values(
    parameter_data: Path,
    pft_count: int,
) -> tuple[dict[str, jnp.ndarray], float, float]:
    values = load_native_variables(
        parameter_data,
        [*_PFT_PARAMETER_NAMES, "r_mort", "k_mort"],
        dtype=np.float64,
        masked_floating_fill_value=np.nan,
    )
    pft_values: dict[str, jnp.ndarray] = {}
    for name in _PFT_PARAMETER_NAMES:
        host = np.asarray(values[name], dtype=np.float64)
        if host.shape != (pft_count,) or not np.all(np.isfinite(host)):
            raise ValueError(f"CLM parameter {name!r} must be one finite PFT vector")
        pft_values[name] = jnp.asarray(host)
    woody = np.asarray(pft_values["woody"])
    if not np.all((woody == 0.0) | (woody == 1.0)):
        raise ValueError("CLM woody parameter must contain only zero or one")
    for name in _PFT_PARAMETER_NAMES[1:]:
        host = np.asarray(pft_values[name])
        if np.any((host < 0.0) | (host > 1.0)):
            raise ValueError(f"CLM litter fraction {name!r} must lie in [0, 1]")
    vegetated = np.arange(pft_count) != 0
    leaf_sum = sum(np.asarray(pft_values[name]) for name in _PFT_PARAMETER_NAMES[1:4])
    root_sum = sum(np.asarray(pft_values[name]) for name in _PFT_PARAMETER_NAMES[4:7])
    if not np.all(leaf_sum[vegetated] == 1.0):
        raise ValueError("CLM leaf litter fractions must sum exactly to one")
    if not np.all(root_sum[vegetated] == 1.0):
        raise ValueError("CLM fine-root litter fractions must sum exactly to one")

    scalars = []
    for name in ("r_mort", "k_mort"):
        host = np.asarray(values[name], dtype=np.float64)
        if host.size != 1 or not np.isfinite(host.reshape(-1)[0]):
            raise ValueError(f"CLM parameter {name!r} must contain one finite scalar")
        scalars.append(float(host.reshape(-1)[0]))
    return pft_values, scalars[0], scalars[1]


def load_cn_gap_mortality_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    state_update_native_data: CnStateUpdate1NativeData | None = None,
    vertical_profile_native_data: (SoilBiogeochemVerticalProfileNativeData | None) = None,
) -> CnGapMortalityNativeData:

    update = state_update_native_data
    vertical = vertical_profile_native_data
    if vertical is None:
        dependency_restart = update.restart_path if update is not None else restart_path
        dependency_parameter = (
            update.parameter_data_path if update is not None else parameter_data_path
        )
        vertical = load_soil_biogeochem_vertical_profile_native_data(
            case,
            restart_path=dependency_restart,
            surface_data_path=surface_data_path,
            parameter_data_path=dependency_parameter,
        )
    else:
        _require_same_selected_file("restart", restart_path, vertical.restart_path)
        _require_same_selected_file(
            "surface-data",
            surface_data_path,
            vertical.surface_data_path,
        )
        _require_same_selected_file(
            "parameter-data",
            parameter_data_path,
            vertical.parameter_data_path,
        )

    if update is None:
        update = load_cn_state_update1_native_data(
            case,
            restart_path=vertical.restart_path,
            parameter_data_path=vertical.parameter_data_path,
        )
    else:
        _require_same_selected_file("restart", restart_path, update.restart_path)
        _require_same_selected_file(
            "parameter-data",
            parameter_data_path,
            update.parameter_data_path,
        )
    _validate_dependencies(update, vertical)

    general = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    required = {
        "use_cn": True,
        "use_crop": True,
        "use_fates": False,
        "soil_layerstruct": "20SL_8.5m",
    }
    for name, expected in required.items():
        if general.get(name) != expected:
            raise ValueError(f"native gap-mortality adapter requires {name}={expected!r}")
    use_cndv = general.get("use_cndv", False)
    if use_cndv is not False:
        raise ValueError("native gap-mortality adapter currently requires use_cndv=.false.")
    spinup_state = general.get("spinup_state", 0)
    if isinstance(spinup_state, bool) or not isinstance(spinup_state, int):
        raise TypeError("CLM spinup_state must be an integer")

    vegetation_type = np.asarray(
        vertical.topology.patch.vegetation_type,
        dtype=np.int32,
    )
    pft_count = int(update.parameters.woody_pft.shape[0])
    if np.any((vegetation_type < 0) | (vegetation_type >= pft_count)):
        raise ValueError("restart patch PFT identity lies outside the parameter file")
    pft, annual_mortality, mortality_coefficient = _parameter_values(
        vertical.parameter_data_path,
        pft_count,
    )
    if not np.array_equal(
        np.asarray(pft["woody"]),
        np.asarray(update.parameters.woody_pft),
    ):
        raise ValueError("native Update1 and gap-mortality woody traits disagree")
    patch_count = vegetation_type.size
    inactive_dynamic = jnp.zeros((patch_count,), dtype=jnp.float64)
    clock = advance_restart_clock(
        load_restart_clock(vertical.restart_path),
        case.value("CALENDAR"),
    )
    year = clock.current_date // 10000
    if year < 1:
        raise ValueError("could not determine a valid restart year")

    return CnGapMortalityNativeData(
        persistent_inputs=CnGapMortalityPersistentInputs(
            soil_patch=vertical.soil_patch,
            soil_column=vertical.soil_column,
            column_patch_indices=vertical.column_patch_indices,
            vegetation_type_patch=vertical.topology.patch.vegetation_type,
            patch_weight_on_column=vertical.topology.patch.weight_on_column,
            patch_active=vertical.topology.patch.active,
            growth_efficiency_patch=inactive_dynamic,
            heat_stress_patch=inactive_dynamic,
        ),
        restart_state=CnGapMortalityState(
            individual_density_patch=inactive_dynamic,
        ),
        parameters=CnGapMortalityParameters(
            woody_pft=pft["woody"],
            leaf_labile_fraction_pft=pft["lf_flab"],
            leaf_cellulose_fraction_pft=pft["lf_fcel"],
            leaf_lignin_fraction_pft=pft["lf_flig"],
            fine_root_labile_fraction_pft=pft["fr_flab"],
            fine_root_cellulose_fraction_pft=pft["fr_fcel"],
            fine_root_lignin_fraction_pft=pft["fr_flig"],
            annual_mortality_rate=annual_mortality,
            growth_efficiency_coefficient=mortality_coefficient,
            days_per_year=_days_per_year(case, year),
            seconds_per_day=86400.0,
            first_prognostic_crop_pft=update.parameters.first_prognostic_crop_pft,
            decomposition_level_count=vertical.parameters.decomposition_level_count,
            use_cndv=False,
            spinup_state=spinup_state,
        ),
        restart_path=vertical.restart_path,
        surface_data_path=vertical.surface_data_path,
        parameter_data_path=vertical.parameter_data_path,
    )


def assemble_cn_gap_mortality_call(
    native: CnGapMortalityNativeData,
    state_update_result: CnStateUpdate1WithPrecisionResult,
    litter_vertical_transport_result: SoilBiogeochemLitterVerticalTransportResult,
    vertical_profile_result: SoilBiogeochemVerticalProfileResult,
    *,
    current_state: CnGapMortalityState | None = None,
) -> CnGapMortalityCall:

    persistent = native.persistent_inputs
    patch_count = persistent.soil_patch.shape[0]
    column_count = persistent.soil_column.shape[0]
    vegetation_state = state_update_result.state.vegetation
    if vegetation_state.paired_carbon_patch.shape[0] != patch_count:
        raise ValueError("post-Update1 vegetation result has the wrong patch count")
    cn_fluxes = litter_vertical_transport_result.cn_fluxes
    if cn_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("vertical-transport flux result has the wrong patch count")
    if cn_fluxes.column_level.shape[0:2] != (
        column_count,
        NUMBER_OF_GROUND_LEVELS,
    ):
        raise ValueError("vertical-transport flux result has the wrong column extent")

    profiles = vertical_profile_result.state
    expected_profile_shape = (patch_count, NUMBER_OF_GROUND_LEVELS)
    for name in (
        "leaf_profile_patch_level",
        "fine_root_profile_patch_level",
        "coarse_root_profile_patch_level",
        "stem_profile_patch_level",
    ):
        if jnp.shape(getattr(profiles, name)) != expected_profile_shape:
            raise ValueError(f"same-step vertical profile {name} has the wrong shape")
    state = native.restart_state if current_state is None else current_state
    if jnp.shape(state.individual_density_patch) != (patch_count,):
        raise ValueError("current individual-density state has the wrong patch count")

    return CnGapMortalityCall(
        inputs=CnGapMortalityInputs(
            soil_patch=persistent.soil_patch,
            soil_column=persistent.soil_column,
            column_patch_indices=persistent.column_patch_indices,
            vegetation_type_patch=persistent.vegetation_type_patch,
            patch_weight_on_column=persistent.patch_weight_on_column,
            patch_active=persistent.patch_active,
            vegetation_state=vegetation_state,
            growth_efficiency_patch=persistent.growth_efficiency_patch,
            heat_stress_patch=persistent.heat_stress_patch,
            leaf_profile_patch_level=profiles.leaf_profile_patch_level,
            fine_root_profile_patch_level=profiles.fine_root_profile_patch_level,
            coarse_root_profile_patch_level=profiles.coarse_root_profile_patch_level,
            stem_profile_patch_level=profiles.stem_profile_patch_level,
        ),
        state=state,
        cn_fluxes=cn_fluxes,
        parameters=native.parameters,
    )


__all__ = [
    "CnGapMortalityCall",
    "CnGapMortalityNativeData",
    "CnGapMortalityPersistentInputs",
    "assemble_cn_gap_mortality_call",
    "load_cn_gap_mortality_native_data",
]
