
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from clm_jax.compilation import materialize_outputs
from clm_jax.execution_observer import stage_barrier
from clm_jax.physics.methane_balance_initialization import (
    NUMBER_OF_METHANE_SOIL_LEVELS,
)
from clm_jax.physics.methane_biogeochemistry import (
    DEFAULT_METHANE_AERENCHYMA_PARAMETERS,
    DEFAULT_METHANE_ANNUAL_UPDATE_PARAMETERS,
    DEFAULT_METHANE_EBULLITION_PARAMETERS,
    DEFAULT_METHANE_OXIDATION_PARAMETERS,
    DEFAULT_METHANE_PRODUCTION_PARAMETERS,
    DEFAULT_METHANE_WATER_TABLE_PARAMETERS,
    MethaneAerenchymaDiagnostics,
    MethaneAerenchymaInputs,
    MethaneAerenchymaParameters,
    MethaneAerenchymaResult,
    MethaneAerenchymaState,
    MethaneAnnualUpdateDiagnostics,
    MethaneAnnualUpdateInputs,
    MethaneAnnualUpdateParameters,
    MethaneAnnualUpdateState,
    MethaneEbullitionDiagnostics,
    MethaneEbullitionInputs,
    MethaneEbullitionParameters,
    MethaneEbullitionState,
    MethaneGasCompetitionState,
    MethaneOxidationDiagnostics,
    MethaneOxidationInputs,
    MethaneOxidationParameters,
    MethaneOxidationState,
    MethaneProductionDiagnostics,
    MethaneProductionInputs,
    MethaneProductionParameters,
    MethaneProductionState,
    MethaneWaterTableDiagnostics,
    MethaneWaterTableInputs,
    MethaneWaterTableParameters,
    MethaneWaterTableState,
    diagnose_methane_water_table_layer,
    methane_aerenchyma_active_case,
    methane_ebullition_active_case,
    methane_oxidation_active_case,
    methane_production_active_case,
    update_methane_annual_means,
)
from clm_jax.physics.methane_driver import (
    DEFAULT_METHANE_DRIVER_FINALIZE_PARAMETERS,
    DEFAULT_METHANE_DRIVER_PREPROCESS_PARAMETERS,
    MethaneDriverFinalizeDiagnostics,
    MethaneDriverFinalizeInputs,
    MethaneDriverFinalizeParameters,
    MethaneDriverFinalizeState,
    MethaneDriverPreprocessDiagnostics,
    MethaneDriverPreprocessInputs,
    MethaneDriverPreprocessParameters,
    MethaneDriverPreprocessState,
    methane_driver_finalize_active_case,
    methane_driver_preprocess_active_case,
)
from clm_jax.physics.methane_soil_gas_transport import (
    DEFAULT_METHANE_SOIL_GAS_TRANSPORT_PARAMETERS,
    MethaneSoilGasTransportDiagnostics,
    MethaneSoilGasTransportInputs,
    MethaneSoilGasTransportParameters,
    MethaneSoilGasTransportResult,
    MethaneSoilGasTransportState,
    methane_soil_gas_transport_active_case,
)
from clm_jax.physics.numerics import (
    fortran_divide,
    strict_exp,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.subgrid import patch_to_column_average


class MethaneDriverActiveInputs(NamedTuple):

    preprocess: MethaneDriverPreprocessInputs
    annual: MethaneAnnualUpdateInputs
    water_table: MethaneWaterTableInputs
    production: MethaneProductionInputs
    oxidation: MethaneOxidationInputs
    aerenchyma: MethaneAerenchymaInputs
    ebullition: MethaneEbullitionInputs
    transport: MethaneSoilGasTransportInputs
    finalize: MethaneDriverFinalizeInputs


class MethaneDriverActiveState(NamedTuple):

    preprocess: MethaneDriverPreprocessState
    annual: MethaneAnnualUpdateState
    water_table: MethaneWaterTableState
    production: MethaneProductionState
    oxidation: MethaneOxidationState
    aerenchyma: MethaneAerenchymaState
    ebullition: MethaneEbullitionState
    transport: MethaneSoilGasTransportState
    finalize: MethaneDriverFinalizeState
    lagged_layer_saturation_column_level: jnp.ndarray
    unsaturated_water_table_depth_column: jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MethaneDriverActiveParameters:

    preprocess: MethaneDriverPreprocessParameters = (
        DEFAULT_METHANE_DRIVER_PREPROCESS_PARAMETERS
    )
    annual: MethaneAnnualUpdateParameters = (
        DEFAULT_METHANE_ANNUAL_UPDATE_PARAMETERS
    )
    water_table: MethaneWaterTableParameters = (
        DEFAULT_METHANE_WATER_TABLE_PARAMETERS
    )
    production: MethaneProductionParameters = (
        DEFAULT_METHANE_PRODUCTION_PARAMETERS
    )
    oxidation: MethaneOxidationParameters = (
        DEFAULT_METHANE_OXIDATION_PARAMETERS
    )
    aerenchyma: MethaneAerenchymaParameters = (
        DEFAULT_METHANE_AERENCHYMA_PARAMETERS
    )
    ebullition: MethaneEbullitionParameters = (
        DEFAULT_METHANE_EBULLITION_PARAMETERS
    )
    transport: MethaneSoilGasTransportParameters = (
        DEFAULT_METHANE_SOIL_GAS_TRANSPORT_PARAMETERS
    )
    finalize: MethaneDriverFinalizeParameters = (
        DEFAULT_METHANE_DRIVER_FINALIZE_PARAMETERS
    )
    vertical_redox_lag_days: float = 0.0
    seconds_per_day: float = 86400.0
    maximum_patches_per_column: int | None = None

    def tree_flatten(self):
        children = (
            self.preprocess,
            self.annual,
            self.water_table,
            self.production,
            self.oxidation,
            self.aerenchyma,
            self.ebullition,
            self.transport,
            self.finalize,
        )
        auxiliary = (
            self.vertical_redox_lag_days,
            self.seconds_per_day,
            self.maximum_patches_per_column,
        )
        return children, auxiliary

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        return cls(
            *children,
            vertical_redox_lag_days=auxiliary[0],
            seconds_per_day=auxiliary[1],
            maximum_patches_per_column=auxiliary[2],
        )


DEFAULT_METHANE_DRIVER_ACTIVE_PARAMETERS = MethaneDriverActiveParameters()


class MethaneDriverActiveDiagnostics(NamedTuple):
    preprocess: MethaneDriverPreprocessDiagnostics
    annual: MethaneAnnualUpdateDiagnostics
    water_table: MethaneWaterTableDiagnostics
    production: MethaneProductionDiagnostics
    oxidation: MethaneOxidationDiagnostics
    aerenchyma: MethaneAerenchymaDiagnostics
    ebullition: MethaneEbullitionDiagnostics
    transport: MethaneSoilGasTransportDiagnostics
    finalize: MethaneDriverFinalizeDiagnostics
    initial_ground_methane_conductance_column: jnp.ndarray
    unsaturated_water_table_depth_column: jnp.ndarray
    lagged_layer_saturation_column_level: jnp.ndarray


class MethaneDriverActiveResult(NamedTuple):
    state: MethaneDriverActiveState
    diagnostics: MethaneDriverActiveDiagnostics


_COMPILED_DRIVER_PREPROCESS = jax.jit(
    methane_driver_preprocess_active_case,
    inline=False,
)
_COMPILED_ANNUAL_UPDATE = jax.jit(
    update_methane_annual_means,
    inline=False,
)
_COMPILED_WATER_TABLE = jax.jit(
    diagnose_methane_water_table_layer,
    inline=False,
)
_COMPILED_PRODUCTION = jax.jit(
    methane_production_active_case,
    inline=False,
)
_COMPILED_OXIDATION = jax.jit(
    methane_oxidation_active_case,
    inline=False,
)
_COMPILED_AERENCHYMA = jax.jit(
    methane_aerenchyma_active_case,
    inline=False,
)
_COMPILED_EBULLITION = jax.jit(
    methane_ebullition_active_case,
    inline=False,
)
_COMPILED_GROUND_CONDUCTANCE = jax.jit(
    patch_to_column_average,
    static_argnames=("maximum_patches_per_column",),
    inline=False,
)
_COMPILED_TRANSPORT = jax.jit(
    methane_soil_gas_transport_active_case,
    inline=False,
)
_COMPILED_DRIVER_FINALIZE = jax.jit(
    methane_driver_finalize_active_case,
    inline=False,
)


def _lagged_layer_saturation(
    methane_column: jnp.ndarray,
    water_table_layer_column: jnp.ndarray,
    previous: jnp.ndarray,
    parameters: MethaneDriverActiveParameters,
) -> jnp.ndarray:

    active = jnp.asarray(methane_column, dtype=jnp.bool_)
    water_table = jnp.asarray(water_table_layer_column, dtype=jnp.int32)
    old = jnp.asarray(previous, dtype=jnp.float64)
    if old.ndim != 2 or old.shape[1] < NUMBER_OF_METHANE_SOIL_LEVELS:
        raise ValueError("lagged layer saturation must cover 20 soil levels")
    level_number = jnp.arange(
        1,
        NUMBER_OF_METHANE_SOIL_LEVELS + 1,
        dtype=jnp.int32,
    )[None, :]
    currently_saturated = level_number > water_table[:, None]
    old_active_levels = old[:, :NUMBER_OF_METHANE_SOIL_LEVELS]
    if parameters.vertical_redox_lag_days > 0.0:
        lag_seconds = fortran_multiply(
            jnp.float64(parameters.vertical_redox_lag_days),
            jnp.float64(parameters.seconds_per_day),
        )
        decay = strict_exp(
            -fortran_divide(
                jnp.float64(parameters.preprocess.time_step_seconds),
                lag_seconds,
            )
        )
        updated = fortran_multiply(old_active_levels, decay)
        updated = jnp.where(
            currently_saturated,
            fortran_add(
                updated,
                fortran_add(jnp.float64(1.0), -decay),
            ),
            updated,
        )
    else:
        updated = currently_saturated.astype(jnp.float64)
    result = old.at[:, :NUMBER_OF_METHANE_SOIL_LEVELS].set(
        jnp.where(active[:, None], updated, old_active_levels)
    )
    return result


def _competition_state(
    previous: MethaneGasCompetitionState,
    production: MethaneProductionState,
    oxidation: MethaneOxidationState,
    aerenchyma: MethaneAerenchymaState,
    ebullition: MethaneEbullitionState,
) -> MethaneGasCompetitionState:

    return MethaneGasCompetitionState(
        unsaturated_methane_oxidation_column_level=(
            oxidation.unsaturated_methane_oxidation_column_level
        ),
        saturated_methane_oxidation_column_level=(
            oxidation.saturated_methane_oxidation_column_level
        ),
        unsaturated_oxygen_oxidation_column_level=(
            oxidation.unsaturated_oxygen_oxidation_demand_column_level
        ),
        saturated_oxygen_oxidation_column_level=(
            oxidation.saturated_oxygen_oxidation_demand_column_level
        ),
        unsaturated_methane_aerenchyma_column_level=(
            aerenchyma.unsaturated_methane_loss_column_level
        ),
        saturated_methane_aerenchyma_column_level=(
            aerenchyma.saturated_methane_loss_column_level
        ),
        unsaturated_methane_ebullition_column_level=(
            ebullition.unsaturated_ebullition_column_level
        ),
        saturated_methane_ebullition_column_level=(
            ebullition.saturated_ebullition_column_level
        ),
        unsaturated_oxygen_decomposition_column_level=(
            production.unsaturated_oxygen_decomposition_demand_column_level
        ),
        saturated_oxygen_decomposition_column_level=(
            production.saturated_oxygen_decomposition_demand_column_level
        ),
        unsaturated_oxygen_stress_column_level=(
            previous.unsaturated_oxygen_stress_column_level
        ),
        saturated_oxygen_stress_column_level=(
            previous.saturated_oxygen_stress_column_level
        ),
        unsaturated_methane_stress_column_level=(
            previous.unsaturated_methane_stress_column_level
        ),
        saturated_methane_stress_column_level=(
            previous.saturated_methane_stress_column_level
        ),
    )


def _methane_driver_active_case(
    inputs: MethaneDriverActiveInputs,
    state: MethaneDriverActiveState,
    parameters: MethaneDriverActiveParameters,
    *,
    compiled_children: bool,
    materialized_children: bool = False,
    aerenchyma_kernel_override: Callable[
        [
            MethaneAerenchymaInputs,
            MethaneAerenchymaState,
            MethaneAerenchymaParameters,
        ],
        MethaneAerenchymaResult,
    ]
    | None = None,
    transport_kernel_override: Callable[
        [
            MethaneSoilGasTransportInputs,
            MethaneSoilGasTransportState,
            MethaneSoilGasTransportParameters,
        ],
        MethaneSoilGasTransportResult,
    ]
    | None = None,
    stage_completed: Callable[[str], None] | None = None,
    memory_barrier: Callable[[], object] | None = None,
) -> MethaneDriverActiveResult:

    def completed(name: str, result: object) -> None:
        if stage_completed is not None:
            stage_barrier(result)
            stage_completed(name)

    def child(compiled):
        if materialized_children:
            return materialize_outputs(compiled.__wrapped__)
        return compiled if compiled_children else compiled.__wrapped__

    preprocess_kernel = child(_COMPILED_DRIVER_PREPROCESS)
    annual_kernel = child(_COMPILED_ANNUAL_UPDATE)
    water_table_kernel = child(_COMPILED_WATER_TABLE)
    production_kernel = child(_COMPILED_PRODUCTION)
    oxidation_kernel = child(_COMPILED_OXIDATION)
    aerenchyma_kernel = aerenchyma_kernel_override
    if aerenchyma_kernel is None:
        aerenchyma_kernel = child(_COMPILED_AERENCHYMA)
    ebullition_kernel = child(_COMPILED_EBULLITION)
    transport_kernel = transport_kernel_override
    if transport_kernel is None:
        transport_kernel = child(_COMPILED_TRANSPORT)
    finalize_kernel = child(_COMPILED_DRIVER_FINALIZE)

    methane_column = jnp.asarray(
        inputs.preprocess.methane_column,
        dtype=jnp.bool_,
    )
    column_gridcell = jnp.asarray(
        inputs.preprocess.column_gridcell,
        dtype=jnp.int32,
    )
    transport_state_on_entry = state.transport

    preprocess_inputs = inputs.preprocess._replace(
        unsaturated_methane_concentration_column_level=(
            transport_state_on_entry.unsaturated_methane_concentration_column_level
        )
    )
    preprocess_state = state.preprocess._replace(
        saturated_methane_concentration_column_level=(
            transport_state_on_entry.saturated_methane_concentration_column_level
        )
    )
    preprocess_result = preprocess_kernel(
        preprocess_inputs,
        preprocess_state,
        parameters.preprocess,
    )
    completed("stage500_preprocess", preprocess_result)
    preprocessed = preprocess_result.state
    atmospheric = (
        preprocess_result.diagnostics.atmospheric_concentration_gridcell_species
    )

    production_inputs_on_entry = inputs.production
    annual_inputs = inputs.annual._replace(
        methane_column=methane_column,
        methane_patch=production_inputs_on_entry.methane_patch,
        patch_column=production_inputs_on_entry.patch_column,
        inundated_fraction_column=preprocessed.inundated_fraction_column,
    )
    annual_result = annual_kernel(
        annual_inputs,
        state.annual,
        parameters.annual,
    )
    completed("stage500_annual_update", annual_result)

    water_table_inputs = inputs.water_table._replace(
        methane_column=methane_column,
    )
    water_table_result = water_table_kernel(
        water_table_inputs,
        state.water_table,
        parameters.water_table,
    )
    completed("stage500_water_table", water_table_result)
    water_table_layer = (
        water_table_result.state.layer_above_water_table_column
    )

    interface_depth = jnp.asarray(
        inputs.ebullition.interface_depth_column_interface,
        dtype=jnp.float64,
    )
    bounded_water_table = jnp.clip(
        water_table_layer,
        jnp.int32(0),
        jnp.int32(NUMBER_OF_METHANE_SOIL_LEVELS),
    )
    diagnosed_water_table_depth = jnp.take_along_axis(
        interface_depth,
        bounded_water_table[:, None],
        axis=1,
    )[:, 0]
    water_table_depth = jnp.where(
        methane_column,
        diagnosed_water_table_depth,
        jnp.asarray(
            state.unsaturated_water_table_depth_column,
            dtype=jnp.float64,
        ),
    )
    layer_saturation = _lagged_layer_saturation(
        methane_column,
        water_table_layer,
        state.lagged_layer_saturation_column_level,
        parameters,
    )

    production_inputs = production_inputs_on_entry._replace(
        methane_column=methane_column,
        water_table_layer_column=water_table_layer,
        inundated_fraction_column=preprocessed.inundated_fraction_column,
        lagged_inundated_fraction_column=(
            preprocessed.lagged_inundated_fraction_column
        ),
        lagged_layer_saturation_column_level=layer_saturation,
    )
    production_result = production_kernel(
        production_inputs,
        state.production,
        parameters.production,
    )
    completed("stage500_production", production_result)

    oxidation_inputs = inputs.oxidation._replace(
        methane_column=methane_column,
        water_table_layer_column=water_table_layer,
        unsaturated_methane_concentration_column_level=(
            transport_state_on_entry.unsaturated_methane_concentration_column_level
        ),
        saturated_methane_concentration_column_level=(
            preprocessed.saturated_methane_concentration_column_level
        ),
        unsaturated_oxygen_concentration_column_level=(
            transport_state_on_entry.unsaturated_oxygen_concentration_column_level
        ),
        saturated_oxygen_concentration_column_level=(
            transport_state_on_entry.saturated_oxygen_concentration_column_level
        ),
    )
    oxidation_result = oxidation_kernel(
        oxidation_inputs,
        state.oxidation,
        parameters.oxidation,
    )
    completed("stage500_oxidation", oxidation_result)

    aerenchyma_inputs = inputs.aerenchyma._replace(
        methane_column=methane_column,
        methane_patch=production_inputs.methane_patch,
        patch_column=production_inputs.patch_column,
        column_gridcell=column_gridcell,
        patch_weight_on_column=production_inputs.patch_weight_on_column,
        annual_average_aboveground_npp_patch=(
            annual_result.state.annual_aboveground_npp_mean_patch
        ),
        annual_average_belowground_npp_patch=(
            annual_result.state.annual_belowground_npp_mean_patch
        ),
        water_table_layer_column=water_table_layer,
        atmospheric_methane_concentration_gridcell=atmospheric[:, 0],
        atmospheric_oxygen_concentration_gridcell=atmospheric[:, 1],
        unsaturated_methane_concentration_column_level=(
            transport_state_on_entry.unsaturated_methane_concentration_column_level
        ),
        saturated_methane_concentration_column_level=(
            preprocessed.saturated_methane_concentration_column_level
        ),
        unsaturated_oxygen_concentration_column_level=(
            transport_state_on_entry.unsaturated_oxygen_concentration_column_level
        ),
        saturated_oxygen_concentration_column_level=(
            transport_state_on_entry.saturated_oxygen_concentration_column_level
        ),
        unsaturated_methane_production_column_level=(
            production_result.state.unsaturated_methane_production_column_level
        ),
        saturated_methane_production_column_level=(
            production_result.state.saturated_methane_production_column_level
        ),
    )
    aerenchyma_result = aerenchyma_kernel(
        aerenchyma_inputs,
        state.aerenchyma,
        parameters.aerenchyma,
    )
    completed("stage500_aerenchyma", aerenchyma_result)

    ebullition_inputs = inputs.ebullition._replace(
        methane_column=methane_column,
        water_table_layer_column=water_table_layer,
        unsaturated_methane_concentration_column_level=(
            transport_state_on_entry.unsaturated_methane_concentration_column_level
        ),
        saturated_methane_concentration_column_level=(
            preprocessed.saturated_methane_concentration_column_level
        ),
    )
    ebullition_result = ebullition_kernel(
        ebullition_inputs,
        state.ebullition,
        parameters.ebullition,
    )
    completed("stage500_ebullition", ebullition_result)

    ground_conductance_kernel = child(_COMPILED_GROUND_CONDUCTANCE)
    initial_ground_conductance = ground_conductance_kernel(
        aerenchyma_inputs.ground_methane_conductance_patch,
        aerenchyma_inputs.patch_column,
        aerenchyma_inputs.patch_weight_on_column,
        aerenchyma_inputs.methane_patch,
        methane_column,
        maximum_patches_per_column=parameters.maximum_patches_per_column,
    )
    competition_on_entry = _competition_state(
        transport_state_on_entry.competition,
        production_result.state,
        oxidation_result.state,
        aerenchyma_result.state,
        ebullition_result.state,
    )
    transport_state = transport_state_on_entry._replace(
        competition=competition_on_entry,
        saturated_methane_concentration_column_level=(
            preprocessed.saturated_methane_concentration_column_level
        ),
        unsaturated_ebullition_total_column=(
            ebullition_result.state.unsaturated_ebullition_total_column
        ),
        saturated_ebullition_total_column=(
            ebullition_result.state.saturated_ebullition_total_column
        ),
        ground_methane_conductance_column=initial_ground_conductance,
    )
    transport_inputs = inputs.transport._replace(
        methane_column=methane_column,
        column_gridcell=column_gridcell,
        unsaturated_water_table_layer_column=water_table_layer,
        atmospheric_methane_gridcell=atmospheric[:, 0],
        atmospheric_oxygen_gridcell=atmospheric[:, 1],
        unsaturated_methane_production_column_level=(
            production_result.state.unsaturated_methane_production_column_level
        ),
        saturated_methane_production_column_level=(
            production_result.state.saturated_methane_production_column_level
        ),
        unsaturated_oxygen_aerenchyma_column_level=(
            aerenchyma_result.state.unsaturated_oxygen_gain_column_level
        ),
        saturated_oxygen_aerenchyma_column_level=(
            aerenchyma_result.state.saturated_oxygen_gain_column_level
        ),
    )
    if memory_barrier is not None:
        memory_barrier()
    transport_result = transport_kernel(
        transport_inputs,
        transport_state,
        parameters.transport,
    )
    completed("stage500_transport", transport_result)
    if memory_barrier is not None:
        memory_barrier()
    transported = transport_result.state
    competition = transported.competition

    production_state = production_result.state._replace(
        unsaturated_oxygen_decomposition_demand_column_level=(
            competition.unsaturated_oxygen_decomposition_column_level
        ),
        saturated_oxygen_decomposition_demand_column_level=(
            competition.saturated_oxygen_decomposition_column_level
        ),
    )
    oxidation_state = oxidation_result.state._replace(
        unsaturated_methane_oxidation_column_level=(
            competition.unsaturated_methane_oxidation_column_level
        ),
        saturated_methane_oxidation_column_level=(
            competition.saturated_methane_oxidation_column_level
        ),
        unsaturated_oxygen_oxidation_demand_column_level=(
            competition.unsaturated_oxygen_oxidation_column_level
        ),
        saturated_oxygen_oxidation_demand_column_level=(
            competition.saturated_oxygen_oxidation_column_level
        ),
    )
    aerenchyma_state = aerenchyma_result.state._replace(
        unsaturated_methane_loss_column_level=(
            competition.unsaturated_methane_aerenchyma_column_level
        ),
        saturated_methane_loss_column_level=(
            competition.saturated_methane_aerenchyma_column_level
        ),
    )
    ebullition_state = ebullition_result.state._replace(
        unsaturated_ebullition_column_level=(
            competition.unsaturated_methane_ebullition_column_level
        ),
        saturated_ebullition_column_level=(
            competition.saturated_methane_ebullition_column_level
        ),
        unsaturated_ebullition_total_column=(
            transported.unsaturated_ebullition_total_column
        ),
        saturated_ebullition_total_column=(
            transported.saturated_ebullition_total_column
        ),
    )

    finalize_inputs = inputs.finalize._replace(
        methane_column=methane_column,
        column_gridcell=column_gridcell,
        inundated_fraction_column=preprocessed.inundated_fraction_column,
        soil_layer_thickness_column_level=(
            production_inputs.soil_layer_thickness_column_level
        ),
        saturated_methane_concentration_column_level=(
            transported.saturated_methane_concentration_column_level
        ),
        unsaturated_methane_concentration_column_level=(
            transported.unsaturated_methane_concentration_column_level
        ),
        saturated_surface_diffusion_column=(
            transported.saturated_surface_diffusion_column
        ),
        unsaturated_surface_diffusion_column=(
            transported.unsaturated_surface_diffusion_column
        ),
        saturated_surface_aerenchyma_column=(
            transported.saturated_surface_aerenchyma_column
        ),
        unsaturated_surface_aerenchyma_column=(
            transported.unsaturated_surface_aerenchyma_column
        ),
        saturated_surface_ebullition_column=(
            transported.saturated_surface_ebullition_column
        ),
        unsaturated_surface_ebullition_column=(
            transported.unsaturated_surface_ebullition_column
        ),
        saturated_methane_oxidation_column_level=(
            oxidation_state.saturated_methane_oxidation_column_level
        ),
        unsaturated_methane_oxidation_column_level=(
            oxidation_state.unsaturated_methane_oxidation_column_level
        ),
        saturated_methane_production_column_level=(
            production_state.saturated_methane_production_column_level
        ),
        unsaturated_methane_production_column_level=(
            production_state.unsaturated_methane_production_column_level
        ),
        inundation_change_surface_flux_column=(
            preprocessed.inundation_change_surface_flux_column
        ),
    )
    finalize_state = state.finalize._replace(
        methane_first_time_column=preprocessed.methane_first_time_column,
    )
    finalize_result = finalize_kernel(
        finalize_inputs,
        finalize_state,
        parameters.finalize,
    )
    completed("stage500_finalize", finalize_result)
    final_preprocess_state = preprocessed._replace(
        saturated_methane_concentration_column_level=(
            transported.saturated_methane_concentration_column_level
        ),
        methane_first_time_column=(
            finalize_result.state.methane_first_time_column
        ),
    )

    result_state = MethaneDriverActiveState(
        preprocess=final_preprocess_state,
        annual=annual_result.state,
        water_table=water_table_result.state,
        production=production_state,
        oxidation=oxidation_state,
        aerenchyma=aerenchyma_state,
        ebullition=ebullition_state,
        transport=transported,
        finalize=finalize_result.state,
        lagged_layer_saturation_column_level=layer_saturation,
        unsaturated_water_table_depth_column=water_table_depth,
    )
    return MethaneDriverActiveResult(
        state=result_state,
        diagnostics=MethaneDriverActiveDiagnostics(
            preprocess=preprocess_result.diagnostics,
            annual=annual_result.diagnostics,
            water_table=water_table_result.diagnostics,
            production=production_result.diagnostics,
            oxidation=oxidation_result.diagnostics,
            aerenchyma=aerenchyma_result.diagnostics,
            ebullition=ebullition_result.diagnostics,
            transport=transport_result.diagnostics,
            finalize=finalize_result.diagnostics,
            initial_ground_methane_conductance_column=(
                initial_ground_conductance
            ),
            unsaturated_water_table_depth_column=water_table_depth,
            lagged_layer_saturation_column_level=layer_saturation,
        ),
    )


def methane_driver_active_case(
    inputs: MethaneDriverActiveInputs,
    state: MethaneDriverActiveState,
    parameters: MethaneDriverActiveParameters = (
        DEFAULT_METHANE_DRIVER_ACTIVE_PARAMETERS
    ),
) -> MethaneDriverActiveResult:

    return _methane_driver_active_case(
        inputs,
        state,
        parameters,
        compiled_children=False,
    )


def methane_driver_active_case_staged(
    inputs: MethaneDriverActiveInputs,
    state: MethaneDriverActiveState,
    parameters: MethaneDriverActiveParameters = (
        DEFAULT_METHANE_DRIVER_ACTIVE_PARAMETERS
    ),
    *,
    aerenchyma_kernel: Callable[
        [
            MethaneAerenchymaInputs,
            MethaneAerenchymaState,
            MethaneAerenchymaParameters,
        ],
        MethaneAerenchymaResult,
    ]
    | None = None,
    transport_kernel: Callable[
        [
            MethaneSoilGasTransportInputs,
            MethaneSoilGasTransportState,
            MethaneSoilGasTransportParameters,
        ],
        MethaneSoilGasTransportResult,
    ]
    | None = None,
    stage_completed: Callable[[str], None] | None = None,
    memory_barrier: Callable[[], object] | None = None,
) -> MethaneDriverActiveResult:

    return _methane_driver_active_case(
        inputs,
        state,
        parameters,
        compiled_children=True,
        aerenchyma_kernel_override=aerenchyma_kernel,
        transport_kernel_override=transport_kernel,
        stage_completed=stage_completed,
        memory_barrier=memory_barrier,
    )


__all__ = [
    "DEFAULT_METHANE_DRIVER_ACTIVE_PARAMETERS",
    "MethaneDriverActiveDiagnostics",
    "MethaneDriverActiveInputs",
    "MethaneDriverActiveParameters",
    "MethaneDriverActiveResult",
    "MethaneDriverActiveState",
    "methane_driver_active_case",
    "methane_driver_active_case_staged",
]
