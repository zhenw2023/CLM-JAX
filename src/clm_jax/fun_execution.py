
from __future__ import annotations

from functools import cache, partial

import jax
import jax.numpy as jnp
from jax.extend.core import get_opaque_trace_state

from clm_jax.compilation import (
    _EAGER_TRACE,
    _inside_ad_stage,
    _stage_reverse_ad,
    jit_discrete_parameters,
)
from clm_jax.execution import _reshard_tree_for_pmap
from clm_jax.physics import ecosystem_dynamics as ed
from clm_jax.physics.cn_flux_layout import with_fieldwise_cn_fluxes
from clm_jax.physics.cn_fun_allocation import (
    _allocate_fun_patch,
    _assemble_fun_allocation,
    _prepare_fun_patches,
)
from clm_jax.physics.cn_fun_competition import (
    FunSoilCompetitionResult,
    integrate_fun_allocation_fluxes,
)
from clm_jax.physics.cn_zero import get_cn_flux_field
from clm_jax.physics.soil_biogeochem_competition import (
    finalize_soil_biogeochem_competition_fun,
    soil_biogeochem_competition_pre_fun,
)
from clm_jax.runtime import gather_mapped_result


def _prepare(prefix, inputs, state):
    competition_inputs, allocation_inputs, allocation_state = ed.assemble_ecosystem_fun_inputs(
        prefix,
        inputs,
        state,
    )
    pre_fun = soil_biogeochem_competition_pre_fun(
        competition_inputs,
        state.soil_biogeochem_competition,
        prefix.state.cn_fluxes,
        inputs.soil_biogeochem_competition_parameters,
    )
    allocation_inputs = allocation_inputs._replace(
        maximum_nitrate_uptake_column_level=get_cn_flux_field(
            pre_fun.cn_fluxes,
            "soil_nitrogen.smin_no3_to_plant_vr_col",
        ),
        maximum_ammonium_uptake_column_level=get_cn_flux_field(
            pre_fun.cn_fluxes,
            "soil_nitrogen.smin_nh4_to_plant_vr_col",
        ),
    )
    return competition_inputs, pre_fun, allocation_inputs, allocation_state


def _finish(prefix, inputs, state, competition_inputs, pre_fun, allocation):
    integrated = integrate_fun_allocation_fluxes(
        inputs.fun_competition_coupling,
        allocation,
        pre_fun.cn_fluxes,
        inputs.fun_competition_coupling_parameters,
    )
    competition = finalize_soil_biogeochem_competition_fun(
        competition_inputs,
        pre_fun.state,
        integrated.cn_fluxes,
        integrated.nitrate_uptake_column_level,
        integrated.ammonium_uptake_column_level,
        inputs.soil_biogeochem_competition_parameters,
    )
    return ed.complete_ecosystem_fun_decomposition(
        prefix,
        inputs,
        state,
        FunSoilCompetitionResult(pre_fun.diagnostics, allocation, competition),
    )


@cache
def parallel_fun_allocation(device_count: int):

    devices = tuple(jax.local_devices()[:device_count])
    if len(devices) != device_count or device_count < 1:
        raise ValueError("FUN worker count exceeds the available JAX devices")

    def prepare(inputs, state, parameters):
        operands, scalar_parameters = _prepare_fun_patches(inputs, state, parameters)
        patch_count = operands[0].shape[0]
        width = (patch_count + device_count - 1) // device_count

        def pack(value):
            padding = ((0, width * device_count - patch_count),) + ((0, 0),) * (value.ndim - 1)
            return jnp.pad(value, padding).reshape((device_count, width, *value.shape[1:]))

        return (
            jax.tree.map(pack, operands),
            scalar_parameters,
            jnp.sum(operands[0], dtype=jnp.int32),
        )

    def mapped(operands, scalar_parameters):
        return jax.vmap(_allocate_fun_patch, in_axes=(0, 0, 0, 0, 0, None))(
            *operands,
            scalar_parameters,
        )

    def finish(allocated, patch_count, soil_patch_count):
        allocated = jax.tree.map(
            lambda value: value.reshape((-1, *value.shape[2:]))[:patch_count],
            allocated,
        )
        return _assemble_fun_allocation(allocated, soil_patch_count)

    prepare = jax.jit(prepare)
    axis_name = "fun_patch"
    mapped = jax.pmap(mapped, axis_name=axis_name, in_axes=(0, None), devices=devices)
    finish = jax.jit(finish, static_argnums=(1,))

    def allocate(inputs, state, parameters):
        packed, scalars, count = prepare(inputs, state, parameters)
        packed, scalars = _reshard_tree_for_pmap(
            (packed, scalars), (0, None), devices, axis_name,
        )
        allocated = mapped(packed, scalars)
        allocated = gather_mapped_result(allocated, devices[0])
        return finish(allocated, inputs.soil_patch.shape[0], count)

    return allocate


@cache
def staged_fun_competition(
    retain_diagnostics: bool, persistent_flux_fields: bool = False,
    share_input_aliases: bool = False,
):
    from clm_jax.physics.cn_fun_allocation import (
        _materialized_fun,
        carbon_use_efficiency_nitrogen_uptake,
    )

    compile_stage = (
        partial(jit_discrete_parameters, forward_inputs=True, share_inputs=share_input_aliases,
                reverse_replay=False)
        if persistent_flux_fields else jax.jit
    )
    prepare = compile_stage(with_fieldwise_cn_fluxes(
        _prepare, pack_output=not persistent_flux_fields,
    ))
    def independent_allocation(*arguments):
        token = _materialized_fun.set(True)
        try:
            return carbon_use_efficiency_nitrogen_uptake(*arguments)
        finally:
            _materialized_fun.reset(token)

    allocate = compile_stage(independent_allocation)

    def finish(*arguments):
        result = _finish(*arguments)
        return result if retain_diagnostics else result._replace(diagnostics=None)

    complete = compile_stage(with_fieldwise_cn_fluxes(
        finish, pack_output=not persistent_flux_fields,
    ))

    def execute(prefix, inputs, state):
        if (get_opaque_trace_state() != _EAGER_TRACE
                and not (_stage_reverse_ad.get() and not _inside_ad_stage.get())):
            result = ed.ecosystem_dynamics_competition_through_decomposition(prefix, inputs, state)
            return result if retain_diagnostics else result._replace(diagnostics=None)
        competition_inputs, pre_fun, allocation_inputs, allocation_state = prepare(
            prefix, inputs, state,
        )
        allocation = allocate(allocation_inputs, allocation_state, inputs.fun_allocation_parameters)
        return complete(prefix, inputs, state, competition_inputs, pre_fun, allocation)

    return execute


@cache
def parallel_fun_competition(
    device_count: int, retain_diagnostics: bool, persistent_flux_fields: bool = False,
):

    options = {"xla_cpu_parallel_codegen_split_count": 1}
    compile_stage = (
        partial(jit_discrete_parameters, forward_inputs=True)
        if persistent_flux_fields else jax.jit
    )

    def compile_flux_stage(function):
        return compile_stage(
            with_fieldwise_cn_fluxes(function, pack_output=not persistent_flux_fields),
            compiler_options=options,
        )

    prepare = compile_flux_stage(_prepare)

    def finish(*arguments):
        result = _finish(*arguments)
        return result if retain_diagnostics else result._replace(diagnostics=None)

    complete = compile_flux_stage(finish)
    allocate = parallel_fun_allocation(device_count)

    def execute(prefix, inputs, state):
        competition_inputs, pre_fun, allocation_inputs, allocation_state = prepare(
            prefix,
            inputs,
            state,
        )
        allocation = allocate(allocation_inputs, allocation_state, inputs.fun_allocation_parameters)
        allocation = gather_mapped_result(allocation, jax.local_devices()[0])
        return complete(prefix, inputs, state, competition_inputs, pre_fun, allocation)

    return execute
