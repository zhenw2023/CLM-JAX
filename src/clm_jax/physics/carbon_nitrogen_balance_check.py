
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.physics.numerics import fortran_add, fortran_multiply

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase


class CarbonNitrogenBalanceCheckParameters(NamedTuple):
    time_step_seconds: float = 1800.0
    skip_steps: int = 2
    use_fun: bool = True
    use_crop: bool = True
    use_nitrification_denitrification: bool = True

    @classmethod
    def from_case(cls, case: CimeCase) -> CarbonNitrogenBalanceCheckParameters:

        from clm_jax.io.namelist import read_case_namelist_group  

        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        time_step = float(general["dtime"])
        if time_step <= 0.0:
            raise ValueError("dtime must be positive")
        return cls(
            time_step_seconds=time_step,
            skip_steps=max(2, int(np.floor(3600.0 / time_step + 0.5))),
            use_fun=bool(general.get("use_fun", False)),
            use_crop=bool(general.get("use_crop", False)),
            use_nitrification_denitrification=bool(general.get("use_nitrif_denitrif", False)),
        )


DEFAULT_CARBON_NITROGEN_BALANCE_CHECK_PARAMETERS = CarbonNitrogenBalanceCheckParameters()


class CarbonNitrogenBalanceCheckInputs(NamedTuple):
    steps_since_restart: jnp.ndarray
    soil_column: jnp.ndarray
    total_carbon_column: jnp.ndarray
    total_nitrogen_column: jnp.ndarray
    gross_primary_production_column: jnp.ndarray
    ecosystem_respiration_column: jnp.ndarray
    fire_carbon_loss_column: jnp.ndarray
    harvest_excess_respiration_carbon_loss_column: jnp.ndarray
    excess_respiration_carbon_loss_column: jnp.ndarray
    wood_harvest_carbon_column: jnp.ndarray
    grain_to_crop_product_carbon_column: jnp.ndarray
    soil_organic_carbon_leaching_column: jnp.ndarray
    nitrogen_deposition_column: jnp.ndarray
    nitrogen_fixation_column: jnp.ndarray
    free_living_nitrogen_fixation_column: jnp.ndarray
    fertilizer_nitrogen_column: jnp.ndarray
    soybean_nitrogen_fixation_column: jnp.ndarray
    supplemental_nitrogen_column: jnp.ndarray
    denitrification_column: jnp.ndarray
    mineral_nitrogen_leaching_column: jnp.ndarray
    nitrate_leaching_column: jnp.ndarray
    nitrate_runoff_column: jnp.ndarray
    nitrification_n2o_loss_column: jnp.ndarray
    soil_organic_nitrogen_leaching_column: jnp.ndarray
    fire_nitrogen_loss_column: jnp.ndarray
    wood_harvest_nitrogen_column: jnp.ndarray
    grain_to_crop_product_nitrogen_column: jnp.ndarray


class CarbonNitrogenBalanceCheckState(NamedTuple):
    beginning_carbon_balance_column: jnp.ndarray
    ending_carbon_balance_column: jnp.ndarray
    beginning_nitrogen_balance_column: jnp.ndarray
    ending_nitrogen_balance_column: jnp.ndarray


class CarbonNitrogenBalanceCheckDiagnostics(NamedTuple):
    skipped: jnp.ndarray
    carbon_inputs_column: jnp.ndarray
    carbon_outputs_column: jnp.ndarray
    carbon_balance_error_column: jnp.ndarray
    carbon_warning_column: jnp.ndarray
    last_carbon_fatal_index: jnp.ndarray
    carbon_fatal: jnp.ndarray
    nitrogen_inputs_column: jnp.ndarray
    nitrogen_outputs_column: jnp.ndarray
    nitrogen_balance_error_column: jnp.ndarray
    nitrogen_warning_column: jnp.ndarray
    last_nitrogen_fatal_index: jnp.ndarray
    nitrogen_fatal: jnp.ndarray


class CarbonNitrogenBalanceCheckResult(NamedTuple):
    state: CarbonNitrogenBalanceCheckState
    diagnostics: CarbonNitrogenBalanceCheckDiagnostics


def _ordered_sum(*terms: jnp.ndarray) -> jnp.ndarray:
    value = terms[0]
    for term in terms[1:]:
        value = fortran_add(value, term)
    return value


def _last_true_index(mask: jnp.ndarray) -> jnp.ndarray:
    indices = jnp.where(mask, jnp.arange(mask.size, dtype=jnp.int32), jnp.int32(-1))
    return jnp.max(jnp.concatenate((jnp.array([-1], dtype=jnp.int32), indices)))


def _cast_and_validate(
    inputs: CarbonNitrogenBalanceCheckInputs,
    state: CarbonNitrogenBalanceCheckState,
    parameters: CarbonNitrogenBalanceCheckParameters,
) -> tuple[CarbonNitrogenBalanceCheckInputs, CarbonNitrogenBalanceCheckState]:
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    if soil_column.ndim != 1:
        raise ValueError("soil_column must be a vector")
    shape = soil_column.shape
    steps = jnp.asarray(inputs.steps_since_restart, dtype=jnp.int32)
    if steps.shape != ():
        raise ValueError("steps_since_restart must be scalar")
    if parameters.time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")
    if parameters.skip_steps < 2:
        raise ValueError("skip_steps must be at least two")

    input_values: dict[str, jnp.ndarray] = {
        "steps_since_restart": steps,
        "soil_column": soil_column,
    }
    for field in CarbonNitrogenBalanceCheckInputs._fields[2:]:
        value = jnp.asarray(getattr(inputs, field), dtype=jnp.float64)
        if value.shape != shape:
            raise ValueError(f"{field} must have shape {shape}")
        input_values[field] = value
    state_values: dict[str, jnp.ndarray] = {}
    for field in CarbonNitrogenBalanceCheckState._fields:
        value = jnp.asarray(getattr(state, field), dtype=jnp.float64)
        if value.shape != shape:
            raise ValueError(f"{field} must have shape {shape}")
        state_values[field] = value
    return (
        CarbonNitrogenBalanceCheckInputs(**input_values),
        CarbonNitrogenBalanceCheckState(**state_values),
    )


def carbon_nitrogen_balance_check(
    inputs: CarbonNitrogenBalanceCheckInputs,
    state: CarbonNitrogenBalanceCheckState,
    *,
    parameters: CarbonNitrogenBalanceCheckParameters = (
        DEFAULT_CARBON_NITROGEN_BALANCE_CHECK_PARAMETERS
    ),
) -> CarbonNitrogenBalanceCheckResult:

    inputs, state = _cast_and_validate(inputs, state, parameters)
    execute = inputs.steps_since_restart > jnp.int32(parameters.skip_steps)
    selected = inputs.soil_column & execute
    time_step = jnp.float64(parameters.time_step_seconds)

    carbon_inputs = inputs.gross_primary_production_column
    carbon_outputs = _ordered_sum(
        inputs.ecosystem_respiration_column,
        inputs.fire_carbon_loss_column,
        inputs.harvest_excess_respiration_carbon_loss_column,
        inputs.excess_respiration_carbon_loss_column,
    )
    carbon_outputs = _ordered_sum(
        carbon_outputs,
        inputs.wood_harvest_carbon_column,
        inputs.grain_to_crop_product_carbon_column,
    )
    carbon_outputs = fortran_add(
        carbon_outputs,
        -inputs.soil_organic_carbon_leaching_column,
    )
    carbon_storage_change = fortran_add(
        inputs.total_carbon_column,
        -state.beginning_carbon_balance_column,
    )
    carbon_error = fortran_add(
        fortran_multiply(
            fortran_add(carbon_inputs, -carbon_outputs),
            time_step,
        ),
        -carbon_storage_change,
    )

    nitrogen_inputs = _ordered_sum(
        inputs.nitrogen_deposition_column,
        inputs.nitrogen_fixation_column,
        inputs.supplemental_nitrogen_column,
    )
    if parameters.use_fun:
        nitrogen_inputs = fortran_add(
            nitrogen_inputs,
            inputs.free_living_nitrogen_fixation_column,
        )
    if parameters.use_crop:
        nitrogen_inputs = _ordered_sum(
            nitrogen_inputs,
            inputs.fertilizer_nitrogen_column,
            inputs.soybean_nitrogen_fixation_column,
        )

    nitrogen_outputs = fortran_add(
        inputs.denitrification_column,
        inputs.fire_nitrogen_loss_column,
    )
    nitrogen_outputs = _ordered_sum(
        nitrogen_outputs,
        inputs.wood_harvest_nitrogen_column,
        inputs.grain_to_crop_product_nitrogen_column,
    )
    if parameters.use_nitrification_denitrification:
        nitrogen_outputs = fortran_add(
            nitrogen_outputs,
            inputs.nitrification_n2o_loss_column,
        )
        nitrogen_outputs = _ordered_sum(
            nitrogen_outputs,
            inputs.nitrate_leaching_column,
            inputs.nitrate_runoff_column,
        )
    else:
        nitrogen_outputs = fortran_add(
            nitrogen_outputs,
            inputs.mineral_nitrogen_leaching_column,
        )
    nitrogen_outputs = fortran_add(
        nitrogen_outputs,
        -inputs.soil_organic_nitrogen_leaching_column,
    )
    nitrogen_storage_change = fortran_add(
        inputs.total_nitrogen_column,
        -state.beginning_nitrogen_balance_column,
    )
    nitrogen_error = fortran_add(
        fortran_multiply(
            fortran_add(nitrogen_inputs, -nitrogen_outputs),
            time_step,
        ),
        -nitrogen_storage_change,
    )

    zero = jnp.zeros_like(carbon_error)
    reported_carbon_inputs = jnp.where(selected, carbon_inputs, zero)
    reported_carbon_outputs = jnp.where(selected, carbon_outputs, zero)
    reported_carbon_error = jnp.where(selected, carbon_error, zero)
    reported_nitrogen_inputs = jnp.where(selected, nitrogen_inputs, zero)
    reported_nitrogen_outputs = jnp.where(selected, nitrogen_outputs, zero)
    reported_nitrogen_error = jnp.where(selected, nitrogen_error, zero)
    carbon_warning = selected & (jnp.abs(carbon_error) > jnp.float64(1.0e-8))
    carbon_fatal_column = selected & (jnp.abs(carbon_error) > jnp.float64(1.0e-7))
    nitrogen_warning = selected & (jnp.abs(nitrogen_error) > jnp.float64(1.0e-7))
    nitrogen_fatal_column = selected & (jnp.abs(nitrogen_error) > jnp.float64(1.0e-3))
    last_carbon_fatal = _last_true_index(carbon_fatal_column)
    last_nitrogen_fatal = _last_true_index(nitrogen_fatal_column)

    return CarbonNitrogenBalanceCheckResult(
        state=CarbonNitrogenBalanceCheckState(
            beginning_carbon_balance_column=state.beginning_carbon_balance_column,
            ending_carbon_balance_column=jnp.where(
                selected,
                inputs.total_carbon_column,
                state.ending_carbon_balance_column,
            ),
            beginning_nitrogen_balance_column=state.beginning_nitrogen_balance_column,
            ending_nitrogen_balance_column=jnp.where(
                selected,
                inputs.total_nitrogen_column,
                state.ending_nitrogen_balance_column,
            ),
        ),
        diagnostics=CarbonNitrogenBalanceCheckDiagnostics(
            skipped=~execute,
            carbon_inputs_column=reported_carbon_inputs,
            carbon_outputs_column=reported_carbon_outputs,
            carbon_balance_error_column=reported_carbon_error,
            carbon_warning_column=carbon_warning,
            last_carbon_fatal_index=last_carbon_fatal,
            carbon_fatal=last_carbon_fatal >= jnp.int32(0),
            nitrogen_inputs_column=reported_nitrogen_inputs,
            nitrogen_outputs_column=reported_nitrogen_outputs,
            nitrogen_balance_error_column=reported_nitrogen_error,
            nitrogen_warning_column=nitrogen_warning,
            last_nitrogen_fatal_index=last_nitrogen_fatal,
            nitrogen_fatal=last_nitrogen_fatal >= jnp.int32(0),
        ),
    )
