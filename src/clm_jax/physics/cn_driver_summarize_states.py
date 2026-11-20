
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.physics.carbon_nitrogen_balance_initialization import (
    CarbonNitrogenBalanceInitializationInputs,
    CarbonNitrogenBalanceInitializationParameters,
    CarbonNitrogenBalanceInitializationState,
    CarbonNitrogenStateSummaries,
    initialize_carbon_nitrogen_balance,
)

CnDriverSummarizeStatesInputs = CarbonNitrogenBalanceInitializationInputs
CnDriverSummarizeStatesParameters = CarbonNitrogenBalanceInitializationParameters
CnDriverSummarizeStatesState = CarbonNitrogenStateSummaries


class CnDriverSummarizeStatesDiagnostics(NamedTuple):
    summarized_column_count: jnp.ndarray
    summarized_soil_column_count: jnp.ndarray
    summarized_patch_count: jnp.ndarray


class CnDriverSummarizeStatesResult(NamedTuple):
    state: CnDriverSummarizeStatesState
    diagnostics: CnDriverSummarizeStatesDiagnostics


def cn_driver_summarize_states(
    inputs: CnDriverSummarizeStatesInputs,
    state: CnDriverSummarizeStatesState,
    parameters: CnDriverSummarizeStatesParameters,
) -> CnDriverSummarizeStatesResult:

    column_count = inputs.all_column.shape[0]
    combined = initialize_carbon_nitrogen_balance(
        inputs,
        CarbonNitrogenBalanceInitializationState(
            summaries=state,
            beginning_carbon_balance_column=jnp.zeros(
                column_count,
                dtype=jnp.float64,
            ),
            beginning_nitrogen_balance_column=jnp.zeros(
                column_count,
                dtype=jnp.float64,
            ),
        ),
        parameters,
    )
    return CnDriverSummarizeStatesResult(
        state=combined.state.summaries,
        diagnostics=CnDriverSummarizeStatesDiagnostics(
            summarized_column_count=combined.diagnostics.summarized_column_count,
            summarized_soil_column_count=combined.diagnostics.balanced_column_count,
            summarized_patch_count=combined.diagnostics.summarized_patch_count,
        ),
    )


__all__ = [
    "CnDriverSummarizeStatesDiagnostics",
    "CnDriverSummarizeStatesInputs",
    "CnDriverSummarizeStatesParameters",
    "CnDriverSummarizeStatesResult",
    "CnDriverSummarizeStatesState",
    "cn_driver_summarize_states",
]
