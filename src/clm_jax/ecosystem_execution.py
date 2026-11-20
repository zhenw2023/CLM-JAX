
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, partial
from typing import NamedTuple

import jax

from clm_jax.compilation import jit_discrete_parameters
from clm_jax.execution_observer import stage_barrier, stage_timing_synchronized
from clm_jax.fun_execution import parallel_fun_competition, staged_fun_competition
from clm_jax.physics import ecosystem_dynamics as ed
from clm_jax.physics.cn_flux_layout import with_fieldwise_cn_fluxes
from clm_jax.physics.cn_state_update2 import CnStateUpdate2State
from clm_jax.physics.ecosystem_dynamics_carry import (
    CropAccumulatorCarry,
    EcosystemDynamicsPostDrainageCarry,
)


@dataclass(frozen=True)
class _StageFunctions:
    prefix: Callable
    competition_decomposition: Callable
    phase_two_litter: Callable
    growth_respiration: Callable
    c_state_update0_precision: Callable
    cn_state_update1: Callable
    soil_n_state_update1: Callable
    litter_vertical_transport: Callable
    gap_mortality: Callable
    cn_state_update2: Callable
    cn_state_update2h_precision: Callable
    wood_products: Callable
    fire_area: Callable
    fire_flux: Callable
    cn_state_update3_precision: Callable
    annual_update: Callable
    growth_through_transport: Callable | None = None


@dataclass
class EcosystemCallOwner:

    inputs: ed.EcosystemDynamicsCompleteInputs | None
    state: ed.EcosystemDynamicsAnnualUpdateState | None

    def take(
        self,
    ) -> tuple[
        ed.EcosystemDynamicsCompleteInputs,
        ed.EcosystemDynamicsAnnualUpdateState,
    ]:
        if self.inputs is None or self.state is None:
            raise RuntimeError("ecosystem call was already consumed")
        inputs, state = self.inputs, self.state
        self.inputs = None
        self.state = None
        return inputs, state


class _RuntimeWoodProductsResult(NamedTuple):

    wood_products: object
    total_product_loss_gridcell: object


class _RuntimeGrowthRespirationResult(NamedTuple):
    cn_fluxes: object


class _RuntimeCStateUpdate0Result(NamedTuple):
    vegetation_carbon_nitrogen: object


class _RuntimeCnStateUpdate1Result(NamedTuple):
    cn_fluxes: object
    vegetation_carbon_nitrogen: object
    xsmrpool_loss_patch: object


class _RuntimeSoilNStateUpdate1Result(NamedTuple):
    cn_fluxes: object
    soil_nitrogen: object


class _RuntimeLitterVerticalTransportResult(NamedTuple):
    cn_fluxes: object
    vertical_transport: object


class _RuntimeGapMortalityResult(NamedTuple):
    cn_fluxes: object
    gap_mortality: object


class _RuntimeCnStateUpdateResult(NamedTuple):
    cn_fluxes: object
    carbon_nitrogen_state: CnStateUpdate2State


class _RuntimeFireAreaResult(NamedTuple):
    fire_area: object


class _RuntimeFireFluxResult(NamedTuple):
    cn_fluxes: object
    fire_flux: object
    fire_rate_patch: object


class _RuntimeAnnualUpdateResult(NamedTuple):
    annual: object


def _without_diagnostics(function: Callable) -> Callable:

    def projected(*args):
        return function(*args)._replace(diagnostics=None)

    projected.__name__ = function.__name__
    return projected


def _phase_two_runtime(*args):
    result = ed.ecosystem_dynamics_phase_two_litterfall(*args)
    return result._replace(cn_fluxes=None, diagnostics=None)


def _growth_respiration_runtime(*args):
    result = ed.ecosystem_dynamics_growth_respiration(*args)
    return _RuntimeGrowthRespirationResult(cn_fluxes=result.cn_fluxes)


def _c_state_update0_runtime(*args):
    result = ed.ecosystem_dynamics_c_state_update0_precision(*args)
    return _RuntimeCStateUpdate0Result(
        vegetation_carbon_nitrogen=result.state.vegetation_carbon_nitrogen,
    )


def _cn_state_update1_runtime(*args):
    result = ed.ecosystem_dynamics_cn_state_update1(*args)
    state = result.state
    return _RuntimeCnStateUpdate1Result(
        cn_fluxes=(state.first_guardrail.ecosystem.pre_decomposition.prefix.cn_fluxes),
        vegetation_carbon_nitrogen=(state.first_guardrail.vegetation_carbon_nitrogen),
        xsmrpool_loss_patch=state.xsmrpool_loss_patch,
    )


def _soil_n_state_update1_runtime(*args):
    result = ed.ecosystem_dynamics_soil_n_state_update1(*args)
    state = result.state
    return _RuntimeSoilNStateUpdate1Result(
        cn_fluxes=(
            state.second_guardrail.first_guardrail.ecosystem.pre_decomposition.prefix.cn_fluxes
        ),
        soil_nitrogen=state.soil_nitrogen,
    )


def _litter_vertical_transport_runtime(*args):
    result = ed.ecosystem_dynamics_litter_vertical_transport(*args)
    state = result.state
    return _RuntimeLitterVerticalTransportResult(
        cn_fluxes=(
            state.soil_n_state_update.second_guardrail.first_guardrail.ecosystem.pre_decomposition.prefix.cn_fluxes
        ),
        vertical_transport=state.vertical_transport,
    )


def _gap_mortality_runtime(*args):
    result = ed.ecosystem_dynamics_gap_mortality(*args)
    state = result.state
    return _RuntimeGapMortalityResult(
        cn_fluxes=(
            state.litter_vertical_transport.soil_n_state_update.second_guardrail.first_guardrail.ecosystem.pre_decomposition.prefix.cn_fluxes
        ),
        gap_mortality=state.gap_mortality,
    )


def _compact_cn_state(transport, cn_fluxes):
    guarded = transport.soil_n_state_update.second_guardrail.first_guardrail
    return _RuntimeCnStateUpdateResult(
        cn_fluxes=cn_fluxes,
        carbon_nitrogen_state=CnStateUpdate2State(
            vegetation=guarded.vegetation_carbon_nitrogen,
            soil_carbon_pool_column_level_pool=(
                transport.vertical_transport.carbon_pool_column_level_pool
            ),
            soil_nitrogen_pool_column_level_pool=(
                transport.vertical_transport.nitrogen_pool_column_level_pool
            ),
        ),
    )


def _cn_state_update2_runtime(*args):
    result = ed.ecosystem_dynamics_cn_state_update2(*args)
    transport = result.state.gap_mortality.litter_vertical_transport
    guarded = transport.soil_n_state_update.second_guardrail.first_guardrail
    cn_fluxes = guarded.ecosystem.pre_decomposition.prefix.cn_fluxes
    return _compact_cn_state(transport, cn_fluxes)


def _cn_state_update2h_precision_runtime(*args):
    result = ed.ecosystem_dynamics_cn_state_update2h_precision(*args)
    transport = result.state.state_update2.gap_mortality.litter_vertical_transport
    guarded = transport.soil_n_state_update.second_guardrail.first_guardrail
    cn_fluxes = guarded.ecosystem.pre_decomposition.prefix.cn_fluxes
    return _compact_cn_state(transport, cn_fluxes)


def _wood_products_runtime(*args) -> _RuntimeWoodProductsResult:
    result = ed.ecosystem_dynamics_wood_products(*args)
    return _RuntimeWoodProductsResult(
        wood_products=result.state.wood_products,
        total_product_loss_gridcell=(
            result.diagnostics.wood_products.carbon.total_product_loss_gridcell
        ),
    )


def _fire_area_runtime(*args):
    result = ed.ecosystem_dynamics_fire_area(*args)
    return _RuntimeFireAreaResult(fire_area=result.state.fire_area)


def _fire_flux_runtime(*args):
    result = ed.ecosystem_dynamics_fire_flux(*args)
    state = result.state
    update2h = state.fire_area.wood_products.state_update2h_precision
    transport = update2h.state_update2.gap_mortality.litter_vertical_transport
    guarded = transport.soil_n_state_update.second_guardrail.first_guardrail
    return _RuntimeFireFluxResult(
        cn_fluxes=guarded.ecosystem.pre_decomposition.prefix.cn_fluxes,
        fire_flux=state.fire_flux,
        fire_rate_patch=result.diagnostics.fire_flux.fire_rate_patch,
    )


def _cn_state_update3_precision_runtime(*args):
    result = ed.ecosystem_dynamics_cn_state_update3_precision(*args)
    products = result.state.fire_flux.fire_area.wood_products
    transport = (
        products.state_update2h_precision.state_update2.gap_mortality.litter_vertical_transport
    )
    guarded = transport.soil_n_state_update.second_guardrail.first_guardrail
    cn_fluxes = guarded.ecosystem.pre_decomposition.prefix.cn_fluxes
    return _compact_cn_state(transport, cn_fluxes)


def _annual_update_runtime(*args):
    result = ed.ecosystem_dynamics_annual_update(*args)
    return _RuntimeAnnualUpdateResult(annual=result.state.annual)


@cache
def _stage_functions(
    *, jit: bool, retain_diagnostics: bool, persistent_flux_fields: bool = False,
    fuse_state_updates: bool = False,
    share_input_aliases: bool | None = None,
) -> _StageFunctions:

    if jit:
        if share_input_aliases is None:
            share_input_aliases = False
        compiler_options = (
            {"xla_cpu_parallel_codegen_split_count": 1} if jax.default_backend() == "cpu" else None
        )

        def transform(function, *, reverse_replay=True):
            compile_stage = (
                partial(jit_discrete_parameters, forward_inputs=True,
                        share_inputs=share_input_aliases, reverse_replay=reverse_replay)
                if persistent_flux_fields else jax.jit
            )
            return compile_stage(
                with_fieldwise_cn_fluxes(function, pack_output=not persistent_flux_fields),
                inline=False,
                compiler_options=compiler_options,
            )

    else:

        def transform(function, *, reverse_replay=True):
            return function

    def runtime(function: Callable) -> Callable:
        return function if retain_diagnostics else _without_diagnostics(function)

    phase_two = (
        ed.ecosystem_dynamics_phase_two_litterfall if retain_diagnostics else _phase_two_runtime
    )
    growth_respiration = (
        ed.ecosystem_dynamics_growth_respiration
        if retain_diagnostics
        else _growth_respiration_runtime
    )
    cn_state_update1 = (
        ed.ecosystem_dynamics_cn_state_update1 if retain_diagnostics else _cn_state_update1_runtime
    )
    soil_n_state_update1 = (
        ed.ecosystem_dynamics_soil_n_state_update1
        if retain_diagnostics
        else _soil_n_state_update1_runtime
    )
    litter_vertical_transport = (
        ed.ecosystem_dynamics_litter_vertical_transport
        if retain_diagnostics
        else _litter_vertical_transport_runtime
    )
    gap_mortality = (
        ed.ecosystem_dynamics_gap_mortality if retain_diagnostics else _gap_mortality_runtime
    )
    cn_state_update2 = (
        ed.ecosystem_dynamics_cn_state_update2 if retain_diagnostics else _cn_state_update2_runtime
    )
    cn_state_update2h_precision = (
        ed.ecosystem_dynamics_cn_state_update2h_precision
        if retain_diagnostics
        else _cn_state_update2h_precision_runtime
    )
    wood_products = (
        ed.ecosystem_dynamics_wood_products if retain_diagnostics else _wood_products_runtime
    )
    fire_area = ed.ecosystem_dynamics_fire_area if retain_diagnostics else _fire_area_runtime
    fire_flux = ed.ecosystem_dynamics_fire_flux if retain_diagnostics else _fire_flux_runtime
    cn_state_update3_precision = (
        ed.ecosystem_dynamics_cn_state_update3_precision
        if retain_diagnostics
        else _cn_state_update3_precision_runtime
    )
    annual_update = (
        ed.ecosystem_dynamics_annual_update if retain_diagnostics else _annual_update_runtime
    )

    if jit and jax.default_backend() == "gpu":
        competition = staged_fun_competition(
            retain_diagnostics, persistent_flux_fields, share_input_aliases=share_input_aliases,
        )
    elif jit and jax.default_backend() == "cpu" and jax.local_device_count() > 1:
        competition = parallel_fun_competition(
            min(64, jax.local_device_count()), retain_diagnostics, persistent_flux_fields,
        )
    else:
        competition = transform(runtime(ed.ecosystem_dynamics_competition_through_decomposition))
    return _StageFunctions(
        prefix=transform(runtime(ed.ecosystem_dynamics_pre_drainage_prefix)),
        competition_decomposition=competition,
        phase_two_litter=transform(phase_two),
        growth_respiration=transform(growth_respiration),
        c_state_update0_precision=transform(
            ed.ecosystem_dynamics_c_state_update0_precision
            if retain_diagnostics
            else _c_state_update0_runtime
        ),
        cn_state_update1=transform(cn_state_update1),
        soil_n_state_update1=transform(soil_n_state_update1),
        litter_vertical_transport=transform(litter_vertical_transport),
        gap_mortality=transform(gap_mortality),
        cn_state_update2=transform(cn_state_update2),
        cn_state_update2h_precision=transform(cn_state_update2h_precision),
        wood_products=transform(wood_products),
        fire_area=transform(fire_area),
        fire_flux=transform(fire_flux),
        cn_state_update3_precision=transform(cn_state_update3_precision),
        annual_update=transform(annual_update),
        growth_through_transport=(
            transform(_growth_through_transport, reverse_replay=False)
            if fuse_state_updates and not retain_diagnostics else None
        ),
    )


_DIRECT_STAGES = _stage_functions(jit=False, retain_diagnostics=True)


def _ecosystem_with_cn_fluxes(ecosystem, cn_fluxes):

    pre_decomposition = ecosystem.pre_decomposition
    prefix = pre_decomposition.prefix
    middle = pre_decomposition.competition_decomposition
    synchronized_prefix = prefix._replace(
        cn_fluxes=cn_fluxes,
        plant_nutrient_demand=prefix.plant_nutrient_demand._replace(
            cn_fluxes=cn_fluxes,
        ),
    )
    synchronized_middle = middle._replace(
        plant_nutrient_competition=middle.plant_nutrient_competition._replace(
            cn_fluxes=cn_fluxes,
        ),
        soil_biogeochem_decomp=middle.soil_biogeochem_decomp._replace(
            cn_fluxes=cn_fluxes,
        ),
    )
    return ecosystem._replace(
        pre_decomposition=pre_decomposition._replace(
            prefix=synchronized_prefix,
            competition_decomposition=synchronized_middle,
        ),
    )


def _state_after_growth_respiration(state, cn_fluxes):

    return _ecosystem_with_cn_fluxes(state, cn_fluxes)


def _state_after_cn_state_update1(first_guardrail, result):

    synchronized_ecosystem = _ecosystem_with_cn_fluxes(
        first_guardrail.ecosystem,
        result.cn_fluxes,
    )
    return ed.EcosystemDynamicsCnStateUpdate1State(
        first_guardrail=first_guardrail._replace(
            ecosystem=synchronized_ecosystem,
            vegetation_carbon_nitrogen=result.vegetation_carbon_nitrogen,
        ),
        xsmrpool_loss_patch=result.xsmrpool_loss_patch,
    )


def _state_after_soil_n_state_update1(second_guardrail, result):

    first_guardrail = second_guardrail.first_guardrail
    synchronized_first_guardrail = first_guardrail._replace(
        ecosystem=_ecosystem_with_cn_fluxes(
            first_guardrail.ecosystem,
            result.cn_fluxes,
        ),
    )
    return ed.EcosystemDynamicsSoilNStateUpdate1State(
        second_guardrail=second_guardrail._replace(
            first_guardrail=synchronized_first_guardrail,
        ),
        soil_nitrogen=result.soil_nitrogen,
    )


def _soil_n_state_with_cn_fluxes(state, cn_fluxes):

    second_guardrail = state.second_guardrail
    first_guardrail = second_guardrail.first_guardrail
    return state._replace(
        second_guardrail=second_guardrail._replace(
            first_guardrail=first_guardrail._replace(
                ecosystem=_ecosystem_with_cn_fluxes(
                    first_guardrail.ecosystem,
                    cn_fluxes,
                ),
            ),
        ),
    )


def _state_after_litter_vertical_transport(soil_n_state_update, result):

    return ed.EcosystemDynamicsLitterVerticalTransportState(
        soil_n_state_update=_soil_n_state_with_cn_fluxes(
            soil_n_state_update,
            result.cn_fluxes,
        ),
        vertical_transport=result.vertical_transport,
    )


def _state_after_gap_mortality(litter_vertical_transport, result):

    synchronized_transport = litter_vertical_transport._replace(
        soil_n_state_update=_soil_n_state_with_cn_fluxes(
            litter_vertical_transport.soil_n_state_update,
            result.cn_fluxes,
        ),
    )
    return ed.EcosystemDynamicsGapMortalityState(
        litter_vertical_transport=synchronized_transport,
        gap_mortality=result.gap_mortality,
    )


def _gap_state_after_cn_update(gap_mortality, result):

    transport = gap_mortality.litter_vertical_transport
    soil_n = transport.soil_n_state_update
    second_guardrail = soil_n.second_guardrail
    first_guardrail = second_guardrail.first_guardrail
    cn_state = result.carbon_nitrogen_state
    synchronized_first_guardrail = first_guardrail._replace(
        ecosystem=_ecosystem_with_cn_fluxes(
            first_guardrail.ecosystem,
            result.cn_fluxes,
        ),
        vegetation_carbon_nitrogen=cn_state.vegetation,
    )
    synchronized_soil_n = soil_n._replace(
        second_guardrail=second_guardrail._replace(
            first_guardrail=synchronized_first_guardrail,
        ),
    )
    synchronized_transport = transport._replace(
        soil_n_state_update=synchronized_soil_n,
        vertical_transport=transport.vertical_transport._replace(
            carbon_pool_column_level_pool=(cn_state.soil_carbon_pool_column_level_pool),
            nitrogen_pool_column_level_pool=(cn_state.soil_nitrogen_pool_column_level_pool),
        ),
    )
    return gap_mortality._replace(
        litter_vertical_transport=synchronized_transport,
    )


def _state_update2h_with_cn_fluxes(state_update2h, cn_fluxes):

    state_update2 = state_update2h.state_update2
    gap_mortality = state_update2.gap_mortality
    transport = gap_mortality.litter_vertical_transport
    synchronized_transport = transport._replace(
        soil_n_state_update=_soil_n_state_with_cn_fluxes(
            transport.soil_n_state_update,
            cn_fluxes,
        ),
    )
    return state_update2h._replace(
        state_update2=state_update2._replace(
            gap_mortality=gap_mortality._replace(
                litter_vertical_transport=synchronized_transport,
            ),
        ),
    )


def _state_after_fire_flux(fire_area, result):

    wood_products = fire_area.wood_products
    synchronized_update2h = _state_update2h_with_cn_fluxes(
        wood_products.state_update2h_precision,
        result.cn_fluxes,
    )
    synchronized_fire_area = fire_area._replace(
        wood_products=wood_products._replace(
            state_update2h_precision=synchronized_update2h,
        ),
        fire_area=fire_area.fire_area._replace(
            unburned_conversion_fraction_column=(
                result.fire_flux.unburned_conversion_fraction_column
            ),
        ),
    )
    return ed.EcosystemDynamicsFireFluxState(
        fire_area=synchronized_fire_area,
        fire_flux=result.fire_flux,
    )


def _state_after_cn_state_update3(fire_flux, result):

    fire_area = fire_flux.fire_area
    wood_products = fire_area.wood_products
    state_update2h = wood_products.state_update2h_precision
    state_update2 = state_update2h.state_update2
    synchronized_update2 = state_update2._replace(
        gap_mortality=_gap_state_after_cn_update(
            state_update2.gap_mortality,
            result,
        ),
    )
    synchronized_fire_flux = fire_flux._replace(
        fire_area=fire_area._replace(
            wood_products=wood_products._replace(
                state_update2h_precision=state_update2h._replace(
                    state_update2=synchronized_update2,
                ),
            ),
        ),
    )
    return ed.EcosystemDynamicsCnStateUpdate3PrecisionState(
        fire_flux=synchronized_fire_flux,
    )


def _state_after_annual_update(state_update3, annual):

    fire_flux = state_update3.fire_flux
    fire_area = fire_flux.fire_area
    wood_products = fire_area.wood_products
    state_update2h = wood_products.state_update2h_precision
    state_update2 = state_update2h.state_update2
    gap_mortality = state_update2.gap_mortality
    transport = gap_mortality.litter_vertical_transport
    soil_n = transport.soil_n_state_update
    second_guardrail = soil_n.second_guardrail
    first_guardrail = second_guardrail.first_guardrail
    ecosystem = first_guardrail.ecosystem
    pre_decomposition = ecosystem.pre_decomposition
    prefix = pre_decomposition.prefix
    nutrient_state = prefix.plant_nutrient_demand._replace(
        temporary_sum_potential_gpp_patch=(annual.temporary_sum_potential_gpp_patch),
        temporary_maximum_retranslocated_nitrogen_patch=(
            annual.temporary_maximum_retranslocated_nitrogen_patch
        ),
    )
    synchronized_first_guardrail = first_guardrail._replace(
        ecosystem=ecosystem._replace(
            pre_decomposition=pre_decomposition._replace(
                prefix=prefix._replace(
                    plant_nutrient_demand=nutrient_state,
                    phenology_climate=prefix.phenology_climate._replace(
                        temperature_average_2m_patch=(
                            annual.temporary_average_air_temperature_patch
                        ),
                    ),
                ),
            ),
        ),
    )
    synchronized_transport = transport._replace(
        soil_n_state_update=soil_n._replace(
            second_guardrail=second_guardrail._replace(
                first_guardrail=synchronized_first_guardrail,
            ),
        ),
    )
    synchronized_update3 = state_update3._replace(
        fire_flux=fire_flux._replace(
            fire_area=fire_area._replace(
                wood_products=wood_products._replace(
                    state_update2h_precision=state_update2h._replace(
                        state_update2=state_update2._replace(
                            gap_mortality=gap_mortality._replace(
                                litter_vertical_transport=synchronized_transport,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    return ed.EcosystemDynamicsAnnualUpdateState(
        state_update3_precision=synchronized_update3,
        annual=annual,
    )


def _growth_through_transport(litter_state, growth_args, first_args,
                              second_args, soil_n_args, transport_args):

    growth = _growth_respiration_runtime(litter_state, *growth_args)
    growth_state = _state_after_growth_respiration(litter_state, growth.cn_fluxes)
    first_inputs, first_template = first_args
    first = _c_state_update0_runtime(
        first_inputs, first_template._replace(ecosystem=growth_state),
    )
    first_state = ed.EcosystemDynamicsCStateUpdate0PrecisionState(
        ecosystem=growth_state,
        vegetation_carbon_nitrogen=first.vegetation_carbon_nitrogen,
    )
    second_inputs, second_template = second_args
    second = _cn_state_update1_runtime(
        second_inputs, second_template._replace(first_guardrail=first_state),
    )
    second_state = _state_after_cn_state_update1(first_state, second)
    soil_n_inputs, soil_n_template = soil_n_args
    soil_n = _soil_n_state_update1_runtime(
        soil_n_inputs, soil_n_template._replace(second_guardrail=second_state),
    )
    soil_n_state = _state_after_soil_n_state_update1(second_state, soil_n)
    transport_inputs, transport_template = transport_args
    transport = _litter_vertical_transport_runtime(
        transport_inputs, transport_template._replace(soil_n_state_update=soil_n_state),
    )
    state = _state_after_litter_vertical_transport(soil_n_state, transport)
    return (
        state,
        second._replace(cn_fluxes=None, vegetation_carbon_nitrogen=None),
        soil_n._replace(cn_fluxes=None),
    )


class EcosystemDynamicsCarryObserver:

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:

        self._mineral_nitrogen_state = None
        self._total_product_loss_gridcell = None
        self._total_burned_area_fraction_column = None
        self._harvest_date_patch = None
        self._crop_accumulators = None
        self._carbon_nitrogen_state = None
        self._cn_fluxes = None
        self._xsmrpool_loss_patch = None
        self._oxygen_decomposition_scalar_column_level = None
        self._annual_sum_npp_patch = None
        self._temporary_sum_npp_patch = None

    @staticmethod
    def _prefix_from_soil_n_state(state):
        return state.second_guardrail.first_guardrail.ecosystem.pre_decomposition.prefix

    def __call__(self, name: str, result: object) -> None:
        if name == "prefix":
            self._harvest_date_patch = result.state.crop_phenology.harvest_date_patch
            crop = result.state.crop_phenology
            self._crop_accumulators = CropAccumulatorCarry(
                crop_live_patch=crop.crop_live_patch,
                growing_degree_days_since_planting_patch=(
                    crop.growing_degree_days_since_planting_patch
                ),
                vernalization_factor_patch=crop.vernalization_factor_patch,
            )
            self._oxygen_decomposition_scalar_column_level = (
                result.state.decomposition_scalars.oxygen_scalar_column_level
            )
        elif name == "soil_n_state_update1":
            self._mineral_nitrogen_state = getattr(
                result,
                "soil_nitrogen",
                None,
            )
            if self._mineral_nitrogen_state is None:
                self._mineral_nitrogen_state = result.state.soil_nitrogen
        elif name == "cn_state_update1":
            self._xsmrpool_loss_patch = getattr(
                result,
                "xsmrpool_loss_patch",
                None,
            )
            if self._xsmrpool_loss_patch is None:
                self._xsmrpool_loss_patch = result.state.xsmrpool_loss_patch
        elif name == "wood_products":
            self._total_product_loss_gridcell = getattr(
                result,
                "total_product_loss_gridcell",
                None,
            )
            if self._total_product_loss_gridcell is None:
                self._total_product_loss_gridcell = (
                    result.diagnostics.wood_products.carbon.total_product_loss_gridcell
                )
        elif name == "fire_area":
            area = getattr(result, "fire_area", None)
            if area is None:
                area = result.state.fire_area
            self._total_burned_area_fraction_column = area.total_burned_area_fraction_column
        elif name == "cn_state_update3_precision":
            compact_state = getattr(result, "carbon_nitrogen_state", None)
            if compact_state is not None:
                self._carbon_nitrogen_state = compact_state
                self._cn_fluxes = result.cn_fluxes
            else:
                products = result.state.fire_flux.fire_area.wood_products
                update2 = products.state_update2h_precision.state_update2
                transport = update2.gap_mortality.litter_vertical_transport
                soil_n = transport.soil_n_state_update
                guarded = soil_n.second_guardrail.first_guardrail
                prefix = self._prefix_from_soil_n_state(soil_n)
                self._carbon_nitrogen_state = CnStateUpdate2State(
                    vegetation=guarded.vegetation_carbon_nitrogen,
                    soil_carbon_pool_column_level_pool=(
                        transport.vertical_transport.carbon_pool_column_level_pool
                    ),
                    soil_nitrogen_pool_column_level_pool=(
                        transport.vertical_transport.nitrogen_pool_column_level_pool
                    ),
                )
                self._cn_fluxes = prefix.cn_fluxes
                self._xsmrpool_loss_patch = soil_n.second_guardrail.xsmrpool_loss_patch
        elif name == "annual_update":
            annual = getattr(result, "annual", None)
            if annual is None:
                annual = result.state.annual
            self._annual_sum_npp_patch = annual.annual_sum_npp_patch
            self._temporary_sum_npp_patch = annual.temporary_sum_npp_patch

    def carry(self) -> EcosystemDynamicsPostDrainageCarry:

        fields = {
            "carbon_nitrogen_state": self._carbon_nitrogen_state,
            "mineral_nitrogen_state": self._mineral_nitrogen_state,
            "cn_fluxes": self._cn_fluxes,
            "total_product_loss_gridcell": self._total_product_loss_gridcell,
            "total_burned_area_fraction_column": (self._total_burned_area_fraction_column),
            "harvest_date_patch": self._harvest_date_patch,
            "crop_accumulators": self._crop_accumulators,
            "excess_maintenance_respiration_carbon_loss_patch": (self._xsmrpool_loss_patch),
            "oxygen_decomposition_scalar_column_level": (
                self._oxygen_decomposition_scalar_column_level
            ),
            "annual_sum_npp_patch": self._annual_sum_npp_patch,
            "temporary_sum_npp_patch": self._temporary_sum_npp_patch,
        }
        missing = tuple(name for name, value in fields.items() if value is None)
        if missing:
            raise RuntimeError("ecosystem stage observer did not receive: " + ", ".join(missing))
        return EcosystemDynamicsPostDrainageCarry(**fields)


def _execute(
    owner_or_inputs: EcosystemCallOwner | ed.EcosystemDynamicsCompleteInputs,
    state_or_stages: ed.EcosystemDynamicsAnnualUpdateState | _StageFunctions,
    stages: _StageFunctions | None = None,
    stage_completed: Callable[[str], None] | None = None,
    stage_observer: Callable[[str, object], None] | None = None,
    memory_barrier: Callable[[], object] | None = None,
    retain_diagnostics: bool = True,
) -> ed.EcosystemDynamicsCompleteResult:
    if stages is None:
        if not isinstance(owner_or_inputs, EcosystemCallOwner):
            raise TypeError("owned ecosystem execution requires an EcosystemCallOwner")
        owner = owner_or_inputs
        stages = state_or_stages
    else:
        owner = EcosystemCallOwner(owner_or_inputs, state_or_stages)
    if not isinstance(stages, _StageFunctions):
        raise TypeError("ecosystem execution requires a complete stage plan")
    inputs, state = owner.take()

    def completed(name: str, result: object) -> None:
        if memory_barrier is not None and stage_timing_synchronized():
            unique = {id(value): value for value in jax.tree.leaves(result)}
            stage_barrier(tuple(unique.values()))
        if stage_observer is not None:
            stage_observer(name, result)
        if stage_completed is not None:
            stage_completed(name)

    def released() -> None:
        if memory_barrier is not None and not retain_diagnostics:
            memory_barrier()

    annual_inputs = inputs.annual_update
    update3_chain = inputs.through_state_update3_precision
    update3_stage_inputs = update3_chain.state_update3_precision
    fire_flux_chain = update3_chain.through_fire_flux
    fire_flux_stage_inputs = fire_flux_chain.fire_flux
    fire_area_chain = fire_flux_chain.through_fire_area
    fire_area_stage_inputs = fire_area_chain.fire_area
    products_chain = fire_area_chain.through_wood_products
    products_stage_inputs = products_chain.wood_products
    update2h_chain = products_chain.through_state_update2h_precision
    update2h_stage_inputs = update2h_chain.state_update2h_precision
    update2_chain = update2h_chain.through_state_update2
    update2_stage_inputs = update2_chain.state_update
    gap_chain = update2_chain.through_gap_mortality
    gap_stage_inputs = gap_chain.gap_mortality
    transport_chain = gap_chain.through_litter_vertical_transport
    transport_stage_inputs = transport_chain.litter_vertical_transport
    soil_n_chain = transport_chain.through_soil_n_state_update
    soil_n_stage_inputs = soil_n_chain.soil_n_state_update
    second_chain = soil_n_chain.through_second_guardrail
    second_stage_inputs = second_chain.state_update_and_precision
    first_chain = second_chain.through_first_guardrail
    first_stage_inputs = first_chain.state_update_and_precision
    growth_chain = first_chain.through_growth_respiration
    growth_stage_inputs = growth_chain.growth_respiration
    growth_stage_parameters = growth_chain.growth_respiration_parameters
    litter_chain = growth_chain.through_litter
    phase_two_inputs = litter_chain.phase_two_litter
    pre_chain = litter_chain.pre_decomposition
    prefix_inputs = pre_chain.prefix
    competition_inputs = pre_chain.competition_decomposition
    del (
        inputs,
        update3_chain,
        fire_flux_chain,
        fire_area_chain,
        products_chain,
        update2h_chain,
        update2_chain,
        gap_chain,
        transport_chain,
        soil_n_chain,
        second_chain,
        first_chain,
        growth_chain,
        litter_chain,
        pre_chain,
    )

    annual_state_template = state._replace(state_update3_precision=None)
    update3_state = state.state_update3_precision
    update3_state_template = update3_state._replace(fire_flux=None)
    fire_flux_state = update3_state.fire_flux
    fire_flux_state_template = fire_flux_state._replace(fire_area=None)
    fire_area_state = fire_flux_state.fire_area
    fire_area_state_template = fire_area_state._replace(wood_products=None)
    products_state = fire_area_state.wood_products
    products_state_template = products_state._replace(state_update2h_precision=None)
    update2h_state = products_state.state_update2h_precision
    update2h_state_template = update2h_state._replace(state_update2=None)
    update2_state = update2h_state.state_update2
    update2_state_template = update2_state._replace(gap_mortality=None)
    gap_state = update2_state.gap_mortality
    gap_state_template = gap_state._replace(litter_vertical_transport=None)
    transport_state = gap_state.litter_vertical_transport
    transport_state_template = transport_state._replace(soil_n_state_update=None)
    soil_n_state = transport_state.soil_n_state_update
    soil_n_state_template = soil_n_state._replace(second_guardrail=None)
    second_state = soil_n_state.second_guardrail
    second_state_template = second_state._replace(first_guardrail=None)
    first_state = second_state.first_guardrail
    first_state_template = first_state._replace(ecosystem=None)
    litter_state = first_state.ecosystem
    pre_state = litter_state.pre_decomposition
    prefix_state = pre_state.prefix
    competition_state = pre_state.competition_decomposition
    phase_two_state = litter_state.phase_two_litter
    del (
        state,
        update3_state,
        fire_flux_state,
        fire_area_state,
        products_state,
        update2h_state,
        update2_state,
        gap_state,
        transport_state,
        soil_n_state,
        second_state,
        first_state,
        litter_state,
        pre_state,
    )

    prefix = stages.prefix(prefix_inputs, prefix_state)
    del prefix_inputs, prefix_state
    completed("prefix", prefix)
    persistent_leaf_litter_fun = phase_two_state.leaf_carbon_to_litter_fun_patch
    synchronized_competition_inputs = competition_inputs._replace(
        fun_allocation=competition_inputs.fun_allocation._replace(
            leaf_carbon_to_litter_fun_patch=persistent_leaf_litter_fun,
        ),
    )
    middle = stages.competition_decomposition(
        prefix,
        synchronized_competition_inputs,
        competition_state,
    )
    del competition_inputs, competition_state, synchronized_competition_inputs
    completed("competition_decomposition", middle)
    allocation_state = middle.state.fun_allocation
    synchronized_fun = prefix.state.fun_initialization._replace(
        leaf_cn_ratio_offset_patch=(allocation_state.leaf_carbon_nitrogen_offset_patch),
        leaf_carbon_storage_transfer_accumulator_patch=(
            allocation_state.leaf_carbon_storage_transfer_accumulator_patch
        ),
        leaf_nitrogen_storage_transfer_accumulator_patch=(
            allocation_state.leaf_nitrogen_storage_transfer_accumulator_patch
        ),
        storage_carbon_demand_patch=allocation_state.storage_carbon_demand_patch,
        storage_nitrogen_demand_patch=(allocation_state.storage_nitrogen_demand_patch),
    )
    synchronized_prefix = prefix.state._replace(
        cn_fluxes=middle.cn_fluxes,
        soil_biogeochem_potential_work=middle.soil_biogeochem_potential_work,
        fun_initialization=synchronized_fun,
        plant_nutrient_demand=prefix.state.plant_nutrient_demand._replace(
            cn_fluxes=middle.cn_fluxes,
        ),
    )
    pre_state = ed.EcosystemDynamicsPreDrainageThroughDecompositionState(
        prefix=synchronized_prefix,
        competition_decomposition=middle.state,
    )
    if retain_diagnostics:
        pre_diagnostics = ed.EcosystemDynamicsPreDrainageThroughDecompositionDiagnostics(
            prefix=prefix.diagnostics,
            competition_decomposition=middle.diagnostics,
        )
    del prefix, middle
    released()

    phase_two = stages.phase_two_litter(
        pre_state,
        phase_two_inputs,
        phase_two_state,
    )
    del phase_two_inputs, phase_two_state
    completed("phase_two_litter", phase_two)
    litter_state = ed.EcosystemDynamicsThroughLitterToColumnState(
        pre_decomposition=phase_two.pre_decomposition_state,
        phase_two_litter=phase_two.phase_two_state,
    )
    if retain_diagnostics:
        litter_diagnostics = ed.EcosystemDynamicsThroughLitterToColumnDiagnostics(
            pre_decomposition=pre_diagnostics,
            phase_two_litter=phase_two.diagnostics,
        )
    del phase_two, pre_state
    released()

    if stages.growth_through_transport is not None:
        transport_output_state, second, soil_n = stages.growth_through_transport(
            litter_state,
            (growth_stage_inputs, growth_stage_parameters),
            (first_stage_inputs, first_state_template),
            (second_stage_inputs, second_state_template),
            (soil_n_stage_inputs, soil_n_state_template),
            (transport_stage_inputs, transport_state_template),
        )
        completed("cn_state_update1", second)
        completed("soil_n_state_update1", soil_n)
        completed("growth_through_transport", transport_output_state)
        del (
            litter_state, growth_stage_inputs, growth_stage_parameters,
            first_stage_inputs, first_state_template,
            second_stage_inputs, second_state_template,
            soil_n_stage_inputs, soil_n_state_template,
            transport_stage_inputs, transport_state_template, second, soil_n,
        )
        released()
    else:
        growth = stages.growth_respiration(
            litter_state,
            growth_stage_inputs,
            growth_stage_parameters,
        )
        del growth_stage_inputs, growth_stage_parameters
        completed("growth_respiration", growth)
        growth_state = (
            growth.state
            if retain_diagnostics
            else _state_after_growth_respiration(litter_state, growth.cn_fluxes)
        )
        if retain_diagnostics:
            growth_diagnostics = ed.EcosystemDynamicsThroughGrowthRespirationDiagnostics(
                through_litter=litter_diagnostics,
                growth_respiration=growth.diagnostics,
            )
        del growth, litter_state
        released()

        first = stages.c_state_update0_precision(
            first_stage_inputs,
            first_state_template._replace(ecosystem=growth_state),
        )
        del first_stage_inputs, first_state_template
        completed("c_state_update0_precision", first)
        first_output_state = (
            first.state
            if retain_diagnostics
            else ed.EcosystemDynamicsCStateUpdate0PrecisionState(
                ecosystem=growth_state,
                vegetation_carbon_nitrogen=first.vegetation_carbon_nitrogen,
            )
        )
        if retain_diagnostics:
            first_diagnostics = ed.EcosystemDynamicsThroughFirstGuardrailDiagnostics(
                through_growth_respiration=growth_diagnostics,
                state_update_and_precision=first.diagnostics,
            )
        del first, growth_state
        released()

        second = stages.cn_state_update1(
            second_stage_inputs,
            second_state_template._replace(first_guardrail=first_output_state),
        )
        del second_stage_inputs, second_state_template
        completed("cn_state_update1", second)
        second_output_state = (
            second.state
            if retain_diagnostics
            else _state_after_cn_state_update1(first_output_state, second)
        )
        if retain_diagnostics:
            second_diagnostics = ed.EcosystemDynamicsThroughSecondGuardrailDiagnostics(
                through_first_guardrail=first_diagnostics,
                state_update_and_precision=second.diagnostics,
            )
        del second, first_output_state
        released()

        soil_n = stages.soil_n_state_update1(
            soil_n_stage_inputs,
            soil_n_state_template._replace(second_guardrail=second_output_state),
        )
        del soil_n_stage_inputs, soil_n_state_template
        completed("soil_n_state_update1", soil_n)
        soil_n_output_state = (
            soil_n.state
            if retain_diagnostics
            else _state_after_soil_n_state_update1(second_output_state, soil_n)
        )
        if retain_diagnostics:
            soil_n_diagnostics = ed.EcosystemDynamicsThroughSoilNStateUpdate1Diagnostics(
                through_second_guardrail=second_diagnostics,
                soil_n_state_update=soil_n.diagnostics,
            )
        del soil_n, second_output_state
        released()

        transport = stages.litter_vertical_transport(
            transport_stage_inputs,
            transport_state_template._replace(soil_n_state_update=soil_n_output_state),
        )
        del transport_stage_inputs, transport_state_template
        completed("litter_vertical_transport", transport)
        transport_output_state = (
            transport.state
            if retain_diagnostics
            else _state_after_litter_vertical_transport(
                soil_n_output_state,
                transport,
            )
        )
        if retain_diagnostics:
            transport_diagnostics = ed.EcosystemDynamicsThroughLitterVerticalTransportDiagnostics(
                through_soil_n_state_update=soil_n_diagnostics,
                litter_vertical_transport=transport.diagnostics,
            )
        del transport, soil_n_output_state
        released()

    gap = stages.gap_mortality(
        gap_stage_inputs,
        gap_state_template._replace(litter_vertical_transport=transport_output_state),
    )
    del gap_stage_inputs, gap_state_template
    completed("gap_mortality", gap)
    gap_output_state = (
        gap.state if retain_diagnostics else _state_after_gap_mortality(transport_output_state, gap)
    )
    if retain_diagnostics:
        gap_diagnostics = ed.EcosystemDynamicsThroughGapMortalityDiagnostics(
            through_litter_vertical_transport=transport_diagnostics,
            gap_mortality=gap.diagnostics,
        )
    del gap, transport_output_state
    released()

    update2 = stages.cn_state_update2(
        update2_stage_inputs,
        update2_state_template._replace(gap_mortality=gap_output_state),
    )
    del update2_stage_inputs, update2_state_template
    completed("cn_state_update2", update2)
    update2_output_state = (
        update2.state
        if retain_diagnostics
        else ed.EcosystemDynamicsCnStateUpdate2State(
            gap_mortality=_gap_state_after_cn_update(
                gap_output_state,
                update2,
            ),
        )
    )
    if retain_diagnostics:
        update2_diagnostics = ed.EcosystemDynamicsThroughCnStateUpdate2Diagnostics(
            through_gap_mortality=gap_diagnostics,
            state_update=update2.diagnostics,
        )
    del update2, gap_output_state
    released()

    update2h = stages.cn_state_update2h_precision(
        update2h_stage_inputs,
        update2h_state_template._replace(state_update2=update2_output_state),
    )
    del update2h_stage_inputs, update2h_state_template
    completed("cn_state_update2h_precision", update2h)
    update2h_output_state = (
        update2h.state
        if retain_diagnostics
        else ed.EcosystemDynamicsCnStateUpdate2hPrecisionState(
            state_update2=update2_output_state._replace(
                gap_mortality=_gap_state_after_cn_update(
                    update2_output_state.gap_mortality,
                    update2h,
                ),
            ),
        )
    )
    if retain_diagnostics:
        update2h_diagnostics = ed.EcosystemDynamicsThroughCnStateUpdate2hPrecisionDiagnostics(
            through_state_update2=update2_diagnostics,
            state_update2h_precision=update2h.diagnostics,
        )
    del update2h, update2_output_state
    released()

    products = stages.wood_products(
        products_stage_inputs,
        products_state_template._replace(state_update2h_precision=update2h_output_state),
    )
    del products_stage_inputs, products_state_template
    completed("wood_products", products)
    products_output_state = (
        products.state
        if retain_diagnostics
        else ed.EcosystemDynamicsWoodProductsState(
            state_update2h_precision=update2h_output_state,
            wood_products=products.wood_products,
        )
    )
    if retain_diagnostics:
        products_diagnostics = ed.EcosystemDynamicsThroughWoodProductsDiagnostics(
            through_state_update2h_precision=update2h_diagnostics,
            wood_products=products.diagnostics,
        )
    del products, update2h_output_state
    released()

    fire_area = stages.fire_area(
        fire_area_stage_inputs,
        fire_area_state_template._replace(wood_products=products_output_state),
    )
    del fire_area_stage_inputs, fire_area_state_template
    completed("fire_area", fire_area)
    fire_area_output_state = (
        fire_area.state
        if retain_diagnostics
        else ed.EcosystemDynamicsFireAreaState(
            wood_products=products_output_state,
            fire_area=fire_area.fire_area,
        )
    )
    if retain_diagnostics:
        fire_area_diagnostics = ed.EcosystemDynamicsThroughFireAreaDiagnostics(
            through_wood_products=products_diagnostics,
            fire_area=fire_area.diagnostics,
        )
    del fire_area, products_output_state
    released()

    fire_flux = stages.fire_flux(
        fire_flux_stage_inputs,
        fire_flux_state_template._replace(fire_area=fire_area_output_state),
    )
    del fire_flux_stage_inputs, fire_flux_state_template
    completed("fire_flux", fire_flux)
    fire_flux_output_state = (
        fire_flux.state
        if retain_diagnostics
        else _state_after_fire_flux(fire_area_output_state, fire_flux)
    )
    if retain_diagnostics:
        fire_flux_diagnostics = ed.EcosystemDynamicsThroughFireFluxDiagnostics(
            through_fire_area=fire_area_diagnostics,
            fire_flux=fire_flux.diagnostics,
        )
    del fire_flux, fire_area_output_state
    released()

    update3 = stages.cn_state_update3_precision(
        update3_stage_inputs,
        update3_state_template._replace(fire_flux=fire_flux_output_state),
    )
    del update3_stage_inputs, update3_state_template
    completed("cn_state_update3_precision", update3)
    update3_output_state = (
        update3.state
        if retain_diagnostics
        else _state_after_cn_state_update3(fire_flux_output_state, update3)
    )
    if retain_diagnostics:
        update3_diagnostics = ed.EcosystemDynamicsThroughCnStateUpdate3PrecisionDiagnostics(
            through_fire_flux=fire_flux_diagnostics,
            state_update3_precision=update3.diagnostics,
        )
    del update3, fire_flux_output_state
    released()

    annual = stages.annual_update(
        annual_inputs,
        annual_state_template._replace(state_update3_precision=update3_output_state),
    )
    del annual_inputs, annual_state_template
    completed("annual_update", annual)
    annual_output_state = (
        annual.state
        if retain_diagnostics
        else _state_after_annual_update(update3_output_state, annual.annual)
    )
    diagnostics = None
    if retain_diagnostics:
        diagnostics = ed.EcosystemDynamicsCompleteDiagnostics(
            through_state_update3_precision=update3_diagnostics,
            annual_update=annual.diagnostics,
        )
    del annual, update3_output_state
    released()
    return ed.EcosystemDynamicsCompleteResult(
        state=annual_output_state,
        diagnostics=diagnostics,
    )


def ecosystem_dynamics_complete_staged(
    inputs: ed.EcosystemDynamicsCompleteInputs,
    state: ed.EcosystemDynamicsAnnualUpdateState,
) -> ed.EcosystemDynamicsCompleteResult:

    return _execute(EcosystemCallOwner(inputs, state), _DIRECT_STAGES)


class EcosystemDynamicsStageExecutor:

    def __init__(
        self,
        *,
        jit: bool = True,
        stage_completed: Callable[[str], None] | None = None,
        stage_observer: Callable[[str, object], None] | None = None,
        memory_barrier: Callable[[], object] | None = None,
        retain_diagnostics: bool = True,
        persistent_flux_fields: bool = False,
    ) -> None:
        self._stages = _stage_functions(
            jit=jit,
            retain_diagnostics=retain_diagnostics,
            persistent_flux_fields=persistent_flux_fields,
            fuse_state_updates=jit and persistent_flux_fields and not retain_diagnostics,
        )
        self._stage_completed = stage_completed
        self._stage_observer = stage_observer
        self._memory_barrier = memory_barrier
        self._retain_diagnostics = retain_diagnostics

    def __call__(
        self,
        inputs: ed.EcosystemDynamicsCompleteInputs,
        state: ed.EcosystemDynamicsAnnualUpdateState,
    ) -> ed.EcosystemDynamicsCompleteResult:
        return _execute(
            EcosystemCallOwner(inputs, state),
            self._stages,
            stage_completed=self._stage_completed,
            stage_observer=self._stage_observer,
            memory_barrier=self._memory_barrier,
            retain_diagnostics=self._retain_diagnostics,
        )

    def execute_prepared(
        self,
        owner: EcosystemCallOwner,
    ) -> ed.EcosystemDynamicsCompleteResult:

        return _execute(
            owner,
            self._stages,
            stage_completed=self._stage_completed,
            stage_observer=self._stage_observer,
            memory_barrier=self._memory_barrier,
            retain_diagnostics=self._retain_diagnostics,
        )


__all__ = [
    "EcosystemCallOwner",
    "EcosystemDynamicsCarryObserver",
    "EcosystemDynamicsStageExecutor",
    "ecosystem_dynamics_complete_staged",
]
