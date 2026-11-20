
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.grain_to_product_pools import (
    GrainToProductPoolsNativeData,
    load_grain_to_product_pools_native_data,
)
from clm_jax.io.netcdf import load_native_variables
from clm_jax.io.soil_biogeochem_vertical_profile import (
    SoilBiogeochemVerticalProfileNativeData,
    load_soil_biogeochem_vertical_profile_native_data,
)
from clm_jax.physics.cn_zero import CnTransientFluxState
from clm_jax.physics.grain_to_product_pools import GrainToProductPoolsResult
from clm_jax.physics.litter_to_column import (
    LitterToColumnInputs,
    LitterToColumnParameters,
)
from clm_jax.physics.soil_biogeochem_vertical_profile import (
    NUMBER_OF_GROUND_LEVELS,
    SoilBiogeochemVerticalProfileResult,
)


class LitterToColumnPersistentInputs(NamedTuple):

    soil_column: jnp.ndarray
    column_patch_indices: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    patch_weight_on_column: jnp.ndarray
    patch_active: jnp.ndarray


class LitterToColumnNativeData(NamedTuple):

    persistent_inputs: LitterToColumnPersistentInputs
    parameters: LitterToColumnParameters
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path


class LitterToColumnCall(NamedTuple):

    inputs: LitterToColumnInputs
    cn_fluxes: CnTransientFluxState
    parameters: LitterToColumnParameters


def _require_same_selected_file(
    name: str,
    requested: str | Path | None,
    actual: Path,
) -> None:
    if requested is None:
        return
    if Path(requested).expanduser().resolve() != actual:
        raise ValueError(f"native dependency uses a different {name} file")


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
            b"".join(np.asarray(row, dtype="S1").tolist())
            .decode("ascii")
            .rstrip()
            for row in raw
        )
    else:
        names = tuple(
            "".join(np.asarray(row, dtype=str).tolist()).rstrip() for row in raw
        )
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("CLM pftname entries must be nonempty and unique")
    return names


def _litter_fraction_parameters(
    parameter_data: Path,
    *,
    use_grain_product: bool,
    decomposition_level_count: int,
) -> LitterToColumnParameters:
    variable_names = (
        "lf_flab",
        "lf_fcel",
        "lf_flig",
        "fr_flab",
        "fr_fcel",
        "fr_flig",
    )
    loaded = load_native_variables(
        parameter_data,
        list(variable_names),
        dtype=np.float64,
        masked_floating_fill_value=0.0,
    )
    names = _pft_names(parameter_data)
    fractions: list[jnp.ndarray] = []
    for variable_name in variable_names:
        host = np.asarray(loaded[variable_name], dtype=np.float64)
        if host.shape != (len(names),) or not np.all(np.isfinite(host)):
            raise ValueError(
                f"CLM parameter {variable_name!r} must be one finite PFT vector"
            )
        if np.any((host < 0.0) | (host > 1.0)):
            raise ValueError(f"CLM parameter {variable_name!r} must lie in [0, 1]")
        fractions.append(jnp.asarray(host))
    leaf_sum = sum(np.asarray(value) for value in fractions[:3])
    root_sum = sum(np.asarray(value) for value in fractions[3:])
    vegetated = np.arange(len(names)) != 0
    if not np.all(leaf_sum[vegetated] == 1.0):
        raise ValueError("CLM leaf litter fractions must sum exactly to one")
    if not np.all(root_sum[vegetated] == 1.0):
        raise ValueError("CLM fine-root litter fractions must sum exactly to one")
    try:
        first_crop = names.index("temperate_corn")
    except ValueError as error:
        raise ValueError("CLM parameter file lacks temperate_corn") from error

    return LitterToColumnParameters(
        leaf_labile_fraction_pft=fractions[0],
        leaf_cellulose_fraction_pft=fractions[1],
        leaf_lignin_fraction_pft=fractions[2],
        fine_root_labile_fraction_pft=fractions[3],
        fine_root_cellulose_fraction_pft=fractions[4],
        fine_root_lignin_fraction_pft=fractions[5],
        first_prognostic_crop_pft=first_crop,
        use_grain_product=use_grain_product,
        decomposition_level_count=decomposition_level_count,
    )


def _validate_dependency_topology(
    grain: GrainToProductPoolsNativeData,
    vertical: SoilBiogeochemVerticalProfileNativeData,
) -> None:
    if grain.restart_path != vertical.restart_path:
        raise ValueError("grain and vertical-profile dependencies use different restarts")
    if grain.parameter_data_path != vertical.parameter_data_path:
        raise ValueError(
            "grain and vertical-profile dependencies use different parameter files"
        )
    for name in (
        "column",
        "vegetation_type",
        "active",
        "weight_on_column",
    ):
        grain_value = np.asarray(getattr(grain.topology.patch, name))
        vertical_value = np.asarray(getattr(vertical.topology.patch, name))
        if not np.array_equal(grain_value, vertical_value):
            raise ValueError(f"native dependencies disagree on patch {name}")


def load_litter_to_column_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    grain_native_data: GrainToProductPoolsNativeData | None = None,
    vertical_profile_native_data: (
        SoilBiogeochemVerticalProfileNativeData | None
    ) = None,
) -> LitterToColumnNativeData:

    grain = grain_native_data
    vertical = vertical_profile_native_data
    if vertical is None:
        dependency_restart = grain.restart_path if grain is not None else restart_path
        dependency_parameter = (
            grain.parameter_data_path if grain is not None else parameter_data_path
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

    if grain is None:
        grain = load_grain_to_product_pools_native_data(
            case,
            restart_path=vertical.restart_path,
            parameter_data_path=vertical.parameter_data_path,
        )
    else:
        _require_same_selected_file("restart", restart_path, grain.restart_path)
        _require_same_selected_file(
            "parameter-data",
            parameter_data_path,
            grain.parameter_data_path,
        )
    _validate_dependency_topology(grain, vertical)

    decomposition_level_count = int(
        vertical.parameters.decomposition_level_count
    )
    if int(vertical.parameters.decomposition_full_level_count) != 25:
        raise ValueError("native litter gather requires 25 decomposition levels")
    parameters = _litter_fraction_parameters(
        grain.parameter_data_path,
        use_grain_product=grain.parameters.use_grain_product,
        decomposition_level_count=decomposition_level_count,
    )
    return LitterToColumnNativeData(
        persistent_inputs=LitterToColumnPersistentInputs(
            soil_column=grain.inputs.soil_column,
            column_patch_indices=vertical.column_patch_indices,
            vegetation_type_patch=grain.topology.patch.vegetation_type,
            patch_weight_on_column=grain.inputs.patch_weight_on_column,
            patch_active=grain.inputs.patch_active,
        ),
        parameters=parameters,
        restart_path=grain.restart_path,
        surface_data_path=vertical.surface_data_path,
        parameter_data_path=grain.parameter_data_path,
    )


def assemble_litter_to_column_call(
    native: LitterToColumnNativeData,
    grain_to_product_pools_result: GrainToProductPoolsResult,
    vertical_profile_result: SoilBiogeochemVerticalProfileResult,
) -> LitterToColumnCall:

    persistent = native.persistent_inputs
    patch_count = persistent.vegetation_type_patch.shape[0]
    column_count = persistent.soil_column.shape[0]
    cn_fluxes = grain_to_product_pools_result.cn_fluxes
    if cn_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("grain-product result has the wrong patch count")
    if cn_fluxes.column_scalar.shape[0] != column_count:
        raise ValueError("grain-product result has the wrong column count")
    profile_state = vertical_profile_result.state
    leaf_profile = profile_state.leaf_profile_patch_level
    fine_root_profile = profile_state.fine_root_profile_patch_level
    expected_profile_shape = (patch_count, NUMBER_OF_GROUND_LEVELS)
    if leaf_profile.shape != expected_profile_shape:
        raise ValueError("vertical-profile leaf result has the wrong shape")
    if fine_root_profile.shape != expected_profile_shape:
        raise ValueError("vertical-profile fine-root result has the wrong shape")

    return LitterToColumnCall(
        inputs=LitterToColumnInputs(
            soil_column=persistent.soil_column,
            column_patch_indices=persistent.column_patch_indices,
            vegetation_type_patch=persistent.vegetation_type_patch,
            patch_weight_on_column=persistent.patch_weight_on_column,
            patch_active=persistent.patch_active,
            leaf_profile_patch_level=leaf_profile,
            fine_root_profile_patch_level=fine_root_profile,
        ),
        cn_fluxes=cn_fluxes,
        parameters=native.parameters,
    )


__all__ = [
    "LitterToColumnCall",
    "LitterToColumnNativeData",
    "LitterToColumnPersistentInputs",
    "assemble_litter_to_column_call",
    "load_litter_to_column_native_data",
]
