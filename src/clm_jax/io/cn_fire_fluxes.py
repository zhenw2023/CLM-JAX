
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.cn_fire_area import (
    CnFireAreaCall,
    CnFireAreaNativeData,
    load_cn_fire_area_native_data,
)
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.soil_biogeochem_vertical_profile import (
    SoilBiogeochemVerticalProfileNativeData,
    load_soil_biogeochem_vertical_profile_native_data,
)
from clm_jax.physics.cn_fire_area import CnFireAreaResult
from clm_jax.physics.cn_fire_fluxes import (
    CnFireFluxInputs,
    CnFireFluxParameters,
    CnFireFluxState,
)
from clm_jax.physics.cn_state_update2 import CnStateUpdate2hWithPrecisionResult
from clm_jax.physics.cn_zero import CnTransientFluxState
from clm_jax.physics.soil_biogeochem_vertical_profile import (
    NUMBER_OF_GROUND_LEVELS,
    SoilBiogeochemVerticalProfileResult,
)


class CnFireFluxPersistentInputs(NamedTuple):

    soil_patch: jnp.ndarray
    soil_column: jnp.ndarray
    column_patch_indices: jnp.ndarray
    column_gridcell_index: jnp.ndarray
    patch_column_index: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    patch_weight_on_column: jnp.ndarray
    patch_active: jnp.ndarray
    gridcell_latitude_degrees: jnp.ndarray


class CnFireFluxNativeData(NamedTuple):

    persistent_inputs: CnFireFluxPersistentInputs
    restart_state: CnFireFluxState
    parameters: CnFireFluxParameters
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path


class CnFireFluxCall(NamedTuple):

    inputs: CnFireFluxInputs
    state: CnFireFluxState
    cn_fluxes: CnTransientFluxState
    parameters: CnFireFluxParameters


_PFT_PARAMETER_NAMES = (
    "woody",
    "cc_leaf",
    "cc_lstem",
    "cc_dstem",
    "cc_other",
    "fm_leaf",
    "fm_lstem",
    "fm_other",
    "fm_root",
    "fm_lroot",
    "fm_droot",
    "lf_flab",
    "lf_fcel",
    "lf_flig",
    "fr_flab",
    "fr_fcel",
    "fr_flig",
)
_LITTER_POOLS = np.asarray([True, True, True, False, False, False, False])


def _require_same_selected_file(
    name: str,
    requested: str | Path | None,
    actual: Path,
) -> None:
    if requested is None:
        return
    if Path(requested).expanduser().resolve() != actual:
        raise ValueError(f"native fire-flux dependency uses a different {name} file")


def _logical(group: dict[str, object], name: str, default: bool) -> bool:
    value = group.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"CLM namelist value {name!r} must be logical")
    return value


def _validate_case(case: CimeCase) -> tuple[bool, bool]:
    general = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    dynamic = dict(read_case_namelist_group(case, "lnd_in", "dynamic_subgrid"))
    required = {
        "use_cn": True,
        "use_crop": True,
        "use_century_decomp": True,
        "use_vertsoilc": True,
        "use_fates": False,
        "soil_layerstruct": "20SL_8.5m",
    }
    for name, expected in required.items():
        if general.get(name) != expected:
            raise ValueError(f"native fire-flux adapter requires {name}={expected!r}")
    use_cndv = _logical(general, "use_cndv", False)
    transient = any(
        _logical(dynamic, name, False) for name in ("do_transient_pfts", "do_transient_crops")
    )
    landuse = general.get("flanduse_timeseries")
    transient = transient or landuse not in (None, "", "null")
    if use_cndv:
        raise NotImplementedError(
            "the current native adapter requires inactive CNDV restart allocation"
        )
    if transient:
        raise NotImplementedError("the current native adapter requires fixed prescribed land cover")
    return use_cndv, transient


def _validate_dependencies(
    fire_area: CnFireAreaNativeData,
    vertical: SoilBiogeochemVerticalProfileNativeData,
) -> None:
    if fire_area.restart_path != vertical.restart_path:
        raise ValueError("fire-area and vertical-profile dependencies use different restarts")
    if fire_area.surface_data_path != vertical.surface_data_path:
        raise ValueError("fire-area and vertical-profile dependencies use different surface data")
    if fire_area.parameter_data_path != vertical.parameter_data_path:
        raise ValueError(
            "fire-area and vertical-profile dependencies use different parameter files"
        )
    area_topology = fire_area.persistent_inputs.topology
    pairs = (
        (area_topology.soil_patch, vertical.soil_patch, "soil patch filter"),
        (area_topology.soil_column, vertical.soil_column, "soil column filter"),
        (
            area_topology.patch_column_index,
            vertical.topology.patch.column,
            "patch-column topology",
        ),
        (
            area_topology.vegetation_type_patch,
            vertical.topology.patch.vegetation_type,
            "patch PFT identity",
        ),
    )
    for left, right, label in pairs:
        if not np.array_equal(np.asarray(left), np.asarray(right)):
            raise ValueError(f"native dependencies disagree on {label}")
    if int(vertical.parameters.decomposition_level_count) != 20:
        raise ValueError("native fire fluxes require 20 active decomposition levels")
    if int(vertical.parameters.decomposition_full_level_count) != 25:
        raise ValueError("native fire fluxes require all 25 ground levels")


def _pft_parameters(parameter_path: Path, pft_count: int) -> dict[str, jnp.ndarray]:
    raw = load_native_variables(
        parameter_path,
        list(_PFT_PARAMETER_NAMES),
        dtype=np.float64,
        masked_floating_fill_value=np.nan,
    )
    result: dict[str, jnp.ndarray] = {}
    for name in _PFT_PARAMETER_NAMES:
        values = np.asarray(raw[name], dtype=np.float64)
        if values.shape != (pft_count,) or not np.all(np.isfinite(values)):
            raise ValueError(f"CLM fire parameter {name!r} must be one finite PFT vector")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"CLM fire parameter {name!r} must lie in [0, 1]")
        result[name] = jnp.asarray(values)
    if not np.all(np.isin(np.asarray(result["woody"]), (0.0, 1.0))):
        raise ValueError("CLM woody parameter must contain only zero or one")
    return result


def _combustion_completion(case: CimeCase) -> tuple[np.float64, np.float64]:
    fire = dict(read_case_namelist_group(case, "lnd_in", "lifire_inparm"))
    raw = fire.get("cmb_cmplt_fact", (0.5, 0.25))
    if not isinstance(raw, tuple | list) or len(raw) != 2:
        raise ValueError("lifire_inparm cmb_cmplt_fact must contain litter and CWD values")
    values = tuple(np.float64(value) for value in raw)
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError("lifire_inparm cmb_cmplt_fact values must lie in [0, 1]")
    return values


def _restart_state(
    restart_path: Path,
    fire_area: CnFireAreaNativeData,
) -> CnFireFluxState:
    topology = fire_area.persistent_inputs.topology
    patch_count = topology.soil_patch.shape[0]
    column_count = topology.soil_column.shape[0]
    values = load_native_variables(
        restart_path,
        ["leafcmax"],
        dtype=np.float64,
        masked_floating_fill_value=float(CLM_SPECIAL_VALUE),
    )
    maximum_leaf_carbon = np.asarray(values["leafcmax"], dtype=np.float64)
    if maximum_leaf_carbon.shape != (patch_count,):
        raise ValueError("restart leafcmax must contain every patch")
    soil_patch = np.asarray(topology.soil_patch, dtype=bool)
    if not np.all(np.isfinite(maximum_leaf_carbon[soil_patch])):
        raise ValueError("restart leafcmax is non-finite on active soil patches")
    return CnFireFluxState(
        unburned_conversion_fraction_column=(
            fire_area.restart_state.unburned_conversion_fraction_column
        ),
        burned_conversion_fraction_column=jnp.zeros(
            (column_count,),
            dtype=jnp.float64,
        ),
        individual_density_patch=jnp.zeros((patch_count,), dtype=jnp.float64),
        maximum_leaf_carbon_patch=jnp.asarray(maximum_leaf_carbon),
    )


def load_cn_fire_flux_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    fire_area_native_data: CnFireAreaNativeData | None = None,
    vertical_profile_native_data: SoilBiogeochemVerticalProfileNativeData | None = None,
) -> CnFireFluxNativeData:

    use_cndv, transient = _validate_case(case)
    fire_area = fire_area_native_data
    if fire_area is None:
        fire_area = load_cn_fire_area_native_data(
            case,
            restart_path=restart_path,
            surface_data_path=surface_data_path,
            parameter_data_path=parameter_data_path,
        )
    else:
        _require_same_selected_file("restart", restart_path, fire_area.restart_path)
        _require_same_selected_file("surface-data", surface_data_path, fire_area.surface_data_path)
        _require_same_selected_file(
            "parameter-data",
            parameter_data_path,
            fire_area.parameter_data_path,
        )
    vertical = vertical_profile_native_data
    if vertical is None:
        vertical = load_soil_biogeochem_vertical_profile_native_data(
            case,
            restart_path=fire_area.restart_path,
            surface_data_path=fire_area.surface_data_path,
            parameter_data_path=fire_area.parameter_data_path,
        )
    else:
        _require_same_selected_file("restart", restart_path, vertical.restart_path)
        _require_same_selected_file("surface-data", surface_data_path, vertical.surface_data_path)
        _require_same_selected_file(
            "parameter-data",
            parameter_data_path,
            vertical.parameter_data_path,
        )
    _validate_dependencies(fire_area, vertical)

    area_topology = fire_area.persistent_inputs.topology
    pft_count = int(fire_area.parameters.fire_spread_rate_pft.shape[0])
    traits = _pft_parameters(fire_area.parameter_data_path, pft_count)
    vegetation_type = np.asarray(area_topology.vegetation_type_patch, dtype=np.int32)
    if np.any((vegetation_type < 0) | (vegetation_type >= pft_count)):
        raise ValueError("restart contains a PFT outside the fire parameter vectors")
    litter_completion, cwd_completion = _combustion_completion(case)
    parameters = CnFireFluxParameters(
        woody_pft=traits["woody"],
        leaf_combustion_fraction_pft=traits["cc_leaf"],
        live_stem_combustion_fraction_pft=traits["cc_lstem"],
        dead_stem_combustion_fraction_pft=traits["cc_dstem"],
        other_combustion_fraction_pft=traits["cc_other"],
        leaf_mortality_fraction_pft=traits["fm_leaf"],
        live_stem_mortality_fraction_pft=traits["fm_lstem"],
        other_mortality_fraction_pft=traits["fm_other"],
        fine_root_mortality_fraction_pft=traits["fm_root"],
        live_root_mortality_fraction_pft=traits["fm_lroot"],
        dead_root_mortality_fraction_pft=traits["fm_droot"],
        leaf_labile_fraction_pft=traits["lf_flab"],
        leaf_cellulose_fraction_pft=traits["lf_fcel"],
        leaf_lignin_fraction_pft=traits["lf_flig"],
        fine_root_labile_fraction_pft=traits["fr_flab"],
        fine_root_cellulose_fraction_pft=traits["fr_fcel"],
        fine_root_lignin_fraction_pft=traits["fr_flig"],
        decomposition_pool_is_litter=jnp.asarray(_LITTER_POOLS),
        decomposition_pool_is_cwd=fire_area.parameters.decomposition_pool_is_cwd,
        litter_combustion_completion_fraction=litter_completion,
        cwd_combustion_completion_fraction=cwd_completion,
        boreal_latitude_degrees=fire_area.parameters.boreal_latitude_degrees,
        time_step=fire_area.parameters.time_step,
        days_per_year=fire_area.parameters.days_per_year,
        decomposition_level_count=vertical.parameters.decomposition_level_count,
        spinup_state=fire_area.parameters.spinup_state,
        first_generic_crop_pft=fire_area.parameters.first_crop_pft,
        bare_ground_pft=fire_area.parameters.bare_ground_pft,
        use_cndv=use_cndv,
        transient_landcover=transient,
        month=fire_area.parameters.month,
        day_of_month=fire_area.parameters.day_of_month,
        seconds_of_day=fire_area.parameters.seconds_of_day,
    )
    return CnFireFluxNativeData(
        persistent_inputs=CnFireFluxPersistentInputs(
            soil_patch=area_topology.soil_patch,
            soil_column=area_topology.soil_column,
            column_patch_indices=vertical.column_patch_indices,
            column_gridcell_index=area_topology.column_gridcell_index,
            patch_column_index=area_topology.patch_column_index,
            vegetation_type_patch=area_topology.vegetation_type_patch,
            patch_weight_on_column=area_topology.patch_weight_on_column,
            patch_active=area_topology.patch_active,
            gridcell_latitude_degrees=area_topology.gridcell_latitude_degrees,
        ),
        restart_state=_restart_state(fire_area.restart_path, fire_area),
        parameters=parameters,
        restart_path=fire_area.restart_path,
        surface_data_path=fire_area.surface_data_path,
        parameter_data_path=fire_area.parameter_data_path,
    )


def assemble_cn_fire_flux_call(
    native: CnFireFluxNativeData,
    fire_area_call: CnFireAreaCall,
    fire_area_result: CnFireAreaResult,
    state_update2h_result: CnStateUpdate2hWithPrecisionResult,
    vertical_profile_result: SoilBiogeochemVerticalProfileResult,
    total_soil_organic_carbon_column: jnp.ndarray,
    *,
    current_state: CnFireFluxState | None = None,
) -> CnFireFluxCall:

    persistent = native.persistent_inputs
    patch_count = persistent.soil_patch.shape[0]
    column_count = persistent.soil_column.shape[0]
    pool_shape = (column_count, NUMBER_OF_GROUND_LEVELS, 7)
    profile_shape = (patch_count, NUMBER_OF_GROUND_LEVELS)
    if fire_area_call.topology.soil_patch.shape != (patch_count,):
        raise ValueError("same-step fire-area call has the wrong patch extent")
    if fire_area_result.state.total_burned_area_fraction_column.shape != (column_count,):
        raise ValueError("same-step fire-area result has the wrong column extent")
    update_state = state_update2h_result.state
    if update_state.vegetation.paired_carbon_patch.shape[0] != patch_count:
        raise ValueError("post-Update2h vegetation state has the wrong patch extent")
    if jnp.shape(update_state.soil_carbon_pool_column_level_pool) != pool_shape:
        raise ValueError("post-Update2h carbon pools have the wrong native shape")
    if jnp.shape(update_state.soil_nitrogen_pool_column_level_pool) != pool_shape:
        raise ValueError("post-Update2h nitrogen pools have the wrong native shape")
    profiles = vertical_profile_result.state
    for name in (
        "leaf_profile_patch_level",
        "fine_root_profile_patch_level",
        "coarse_root_profile_patch_level",
        "stem_profile_patch_level",
    ):
        if jnp.shape(getattr(profiles, name)) != profile_shape:
            raise ValueError(f"same-step vertical profile {name} has the wrong shape")
    total_soil_carbon = jnp.asarray(total_soil_organic_carbon_column, dtype=jnp.float64)
    if total_soil_carbon.shape != (column_count,):
        raise ValueError("start-of-step total soil organic carbon has the wrong column extent")
    fluxes = state_update2h_result.cn_fluxes
    if fluxes.patch_scalar.shape[0] != patch_count or fluxes.column_scalar.shape[0] != column_count:
        raise ValueError("post-Update2h packed fluxes have incompatible native extents")

    area = fire_area_result.state
    state = native.restart_state if current_state is None else current_state
    if state.individual_density_patch.shape != (patch_count,):
        raise ValueError("current fire-flux state has the wrong patch extent")
    if state.burned_conversion_fraction_column.shape != (column_count,):
        raise ValueError("current fire-flux state has the wrong column extent")
    state = state._replace(
        unburned_conversion_fraction_column=area.unburned_conversion_fraction_column
    )
    parameters = native.parameters._replace(
        time_step=fire_area_call.parameters.time_step,
        days_per_year=fire_area_call.parameters.days_per_year,
        spinup_state=fire_area_call.parameters.spinup_state,
        month=fire_area_call.parameters.month,
        day_of_month=fire_area_call.parameters.day_of_month,
        seconds_of_day=fire_area_call.parameters.seconds_of_day,
    )
    return CnFireFluxCall(
        inputs=CnFireFluxInputs(
            **persistent._asdict(),
            vegetation_state=update_state.vegetation,
            crop_fraction_column=area.crop_fraction_column,
            total_burned_area_fraction_column=area.total_burned_area_fraction_column,
            conversion_burned_area_fraction_column=(area.conversion_burned_area_fraction_column),
            conversion_fire_burned_area_fraction_column=(
                area.conversion_fire_burned_area_fraction_column
            ),
            crop_burned_area_fraction_column=area.crop_burned_area_fraction_column,
            peat_burned_area_fraction_column=area.peat_burned_area_fraction_column,
            tropical_evergreen_tree_fraction_column=(area.tropical_evergreen_tree_fraction_column),
            tropical_deciduous_tree_fraction_column=(area.tropical_deciduous_tree_fraction_column),
            decreased_tropical_tree_fraction_column=(area.decreased_tropical_tree_fraction_column),
            leaf_profile_patch_level=profiles.leaf_profile_patch_level,
            fine_root_profile_patch_level=profiles.fine_root_profile_patch_level,
            coarse_root_profile_patch_level=profiles.coarse_root_profile_patch_level,
            stem_profile_patch_level=profiles.stem_profile_patch_level,
            total_soil_organic_carbon_column=total_soil_carbon,
            decomposition_carbon_pool_column_level_pool=(
                update_state.soil_carbon_pool_column_level_pool
            ),
            decomposition_nitrogen_pool_column_level_pool=(
                update_state.soil_nitrogen_pool_column_level_pool
            ),
        ),
        state=state,
        cn_fluxes=fluxes,
        parameters=parameters,
    )


__all__ = [
    "CnFireFluxCall",
    "CnFireFluxNativeData",
    "CnFireFluxPersistentInputs",
    "assemble_cn_fire_flux_call",
    "load_cn_fire_flux_native_data",
]
