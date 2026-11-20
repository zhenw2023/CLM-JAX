
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.io.carbon_nitrogen_balance_initialization import (
    CarbonNitrogenBalanceInitializationNativeData,
    load_carbon_nitrogen_balance_initialization_native_data,
)
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.carbon_nitrogen_balance_initialization import (
    CARBON_DISPLAYED_COMPONENTS,
    CARBON_STORED_COMPONENTS,
    NITROGEN_DISPLAYED_COMPONENTS,
    NITROGEN_STORED_COMPONENTS,
    NUMBER_OF_DECOMPOSITION_LEVELS,
    NUMBER_OF_DECOMPOSITION_POOLS,
    CarbonNitrogenBalanceInitializationInputs,
    CarbonNitrogenStateSummaries,
)
from clm_jax.physics.cn_driver_summarize_states import (
    CnDriverSummarizeStatesInputs,
    CnDriverSummarizeStatesParameters,
    CnDriverSummarizeStatesState,
)
from clm_jax.physics.cn_precision_control import (
    CnPrecisionControlState,
    get_cn_precision_control_state_field,
)
from clm_jax.physics.post_drainage_precision_control import (
    PostDrainagePrecisionControlResult,
)


class CnDriverSummarizeStatesNativeData(NamedTuple):

    topology: SubgridTopology
    inputs: CnDriverSummarizeStatesInputs
    initial_state: CnDriverSummarizeStatesState
    parameters: CnDriverSummarizeStatesParameters
    restart_path: Path
    parameter_data_path: Path


class CnDriverSummarizeStatesCall(NamedTuple):
    inputs: CnDriverSummarizeStatesInputs
    state: CnDriverSummarizeStatesState
    parameters: CnDriverSummarizeStatesParameters


def _qualified_components(
    state: CnPrecisionControlState,
    species: str,
    names: tuple[str, ...],
) -> jnp.ndarray:
    return jnp.stack(
        tuple(
            get_cn_precision_control_state_field(
                state,
                f"vegetation_{species}.{name}_patch",
            )
            for name in names
        ),
        axis=1,
    )


def _field(
    state: CnPrecisionControlState,
    species: str,
    name: str,
) -> jnp.ndarray:
    return get_cn_precision_control_state_field(
        state,
        f"vegetation_{species}.{name}_patch",
    )


def load_cn_driver_summarize_states_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    balance_native_data: (
        CarbonNitrogenBalanceInitializationNativeData | None
    ) = None,
) -> CnDriverSummarizeStatesNativeData:

    base = balance_native_data
    if base is None:
        base = load_carbon_nitrogen_balance_initialization_native_data(
            case,
            restart_path=restart_path,
            parameter_data_path=parameter_data_path,
        )
    else:
        if (
            restart_path is not None
            and Path(restart_path).expanduser().resolve() != base.restart_path
        ):
            raise ValueError("state-summary dependency uses a different restart")
        if (
            parameter_data_path is not None
            and Path(parameter_data_path).expanduser().resolve()
            != base.parameter_data_path
        ):
            raise ValueError("state-summary dependency uses a different parameter file")

    return CnDriverSummarizeStatesNativeData(
        topology=base.topology,
        inputs=base.inputs,
        initial_state=base.state.summaries,
        parameters=base.parameters,
        restart_path=base.restart_path,
        parameter_data_path=base.parameter_data_path,
    )


def assemble_cn_driver_summarize_states_call(
    native: CnDriverSummarizeStatesNativeData,
    precision_result: PostDrainagePrecisionControlResult,
    same_step_xsmrpool_loss_patch: jnp.ndarray,
    previous_state: CarbonNitrogenStateSummaries,
) -> CnDriverSummarizeStatesCall:

    vegetation = precision_result.state.vegetation
    soil = precision_result.state.soil
    patch_count = native.inputs.soil_patch.shape[0]
    column_count = native.inputs.all_column.shape[0]
    if vegetation.paired_carbon_patch.shape[0] != patch_count:
        raise ValueError("post-precision vegetation has the wrong patch extent")
    if jnp.shape(same_step_xsmrpool_loss_patch) != (patch_count,):
        raise ValueError("same-step xsmrpool_loss must contain every patch")
    carbon_pool_shape = jnp.shape(soil.carbon_pool_column_level_pool)
    if len(carbon_pool_shape) != 3:
        raise ValueError("post-precision carbon pools must be a 3-D array")
    full_level_count = carbon_pool_shape[1]
    if carbon_pool_shape != (
        column_count,
        full_level_count,
        NUMBER_OF_DECOMPOSITION_POOLS,
    ):
        raise ValueError("post-precision carbon pools have the wrong native shape")
    if full_level_count < NUMBER_OF_DECOMPOSITION_LEVELS:
        raise ValueError("post-precision soil state lacks 20 decomposition levels")
    expected_level_shape = (column_count, full_level_count)
    for name in (
        "nitrogen_pool_column_level_pool",
    ):
        if getattr(soil, name).shape != (
            column_count,
            full_level_count,
            NUMBER_OF_DECOMPOSITION_POOLS,
        ):
            raise ValueError(f"post-precision {name} has the wrong native shape")
    for name in (
        "carbon_truncation_column_level",
        "nitrogen_truncation_column_level",
        "mineral_nitrogen_column_level",
        "ammonium_column_level",
        "nitrate_column_level",
    ):
        if getattr(soil, name).shape != expected_level_shape:
            raise ValueError(f"post-precision {name} has the wrong native shape")

    active = slice(0, NUMBER_OF_DECOMPOSITION_LEVELS)
    inputs: CarbonNitrogenBalanceInitializationInputs = native.inputs._replace(
        decomposition_carbon_column_level_pool=(
            soil.carbon_pool_column_level_pool[:, active, :]
        ),
        carbon_truncation_column_level=(
            soil.carbon_truncation_column_level[:, active]
        ),
        decomposition_nitrogen_column_level_pool=(
            soil.nitrogen_pool_column_level_pool[:, active, :]
        ),
        mineral_nitrogen_column_level=soil.mineral_nitrogen_column_level[:, active],
        nitrate_nitrogen_column_level=soil.nitrate_column_level[:, active],
        ammonium_nitrogen_column_level=soil.ammonium_column_level[:, active],
        nitrogen_truncation_column_level=(
            soil.nitrogen_truncation_column_level[:, active]
        ),
        displayed_carbon_patch_component=_qualified_components(
            vegetation,
            "carbon",
            CARBON_DISPLAYED_COMPONENTS,
        ),
        stored_carbon_patch_component=_qualified_components(
            vegetation,
            "carbon",
            CARBON_STORED_COMPONENTS,
        ),
        grain_carbon_patch=_field(vegetation, "carbon", "grainc"),
        grain_carbon_storage_patch=_field(
            vegetation,
            "carbon",
            "grainc_storage",
        ),
        grain_carbon_transfer_patch=_field(
            vegetation,
            "carbon",
            "grainc_xfer",
        ),
        crop_seed_carbon_deficit_patch=_field(
            vegetation,
            "carbon",
            "cropseedc_deficit",
        ),
        excess_maintenance_respiration_carbon_patch=_field(
            vegetation,
            "carbon",
            "xsmrpool",
        ),
        excess_maintenance_respiration_carbon_loss_patch=jnp.asarray(
            same_step_xsmrpool_loss_patch,
            dtype=jnp.float64,
        ),
        patch_carbon_truncation=vegetation.carbon_truncation_patch,
        displayed_nitrogen_patch_component=_qualified_components(
            vegetation,
            "nitrogen",
            NITROGEN_DISPLAYED_COMPONENTS,
        ),
        stored_nitrogen_patch_component=_qualified_components(
            vegetation,
            "nitrogen",
            NITROGEN_STORED_COMPONENTS,
        ),
        grain_nitrogen_patch=_field(vegetation, "nitrogen", "grainn"),
        grain_nitrogen_storage_patch=_field(
            vegetation,
            "nitrogen",
            "grainn_storage",
        ),
        grain_nitrogen_transfer_patch=_field(
            vegetation,
            "nitrogen",
            "grainn_xfer",
        ),
        crop_seed_nitrogen_deficit_patch=_field(
            vegetation,
            "nitrogen",
            "cropseedn_deficit",
        ),
        patch_nitrogen_truncation=vegetation.nitrogen_truncation_patch,
    )
    return CnDriverSummarizeStatesCall(
        inputs=inputs,
        state=previous_state,
        parameters=native.parameters,
    )


__all__ = [
    "CnDriverSummarizeStatesCall",
    "CnDriverSummarizeStatesNativeData",
    "assemble_cn_driver_summarize_states_call",
    "load_cn_driver_summarize_states_native_data",
]
