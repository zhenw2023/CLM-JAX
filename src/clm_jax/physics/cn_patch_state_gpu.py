
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from clm_jax.physics.cn_zero import (
    CnFluxFields,
    CnTransientFluxState,
    get_cn_flux_field,
    set_cn_flux_field,
)
from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch

_CARBON_CHANGED_FLUXES = tuple(
    "vegetation_carbon.cpool_to_" + organ + "c" + suffix + "_patch"
    for organ in ("leaf", "froot", "livecroot", "livestem")
    for suffix in ("", "_storage")
) + ("vegetation_carbon.xsmrpool_to_atm_patch",)
_CARBON_PARAMETER_NAMES = (
    "time_step",
    "carbon_respiration_option",
    "dribble_crop_harvest_xsmrpool_to_atmosphere",
    "crop_product_decay_rate",
)


def _fluxes(fields):
    return CnTransientFluxState(fields, None, None, None, None, None, None)


def _carbon_reference(harvest, state, fields, parameters, soil, woody, crop):
    from clm_jax.physics.cn_state_update1 import _update_carbon_state_reference

    state, result = _update_carbon_state_reference(
        SimpleNamespace(harvest_date_patch=harvest),
        state,
        _fluxes(fields),
        SimpleNamespace(**dict(zip(_CARBON_PARAMETER_NAMES, parameters, strict=True))),
        soil,
        woody,
        crop,
    )
    return state, tuple(get_cn_flux_field(result, name) for name in _CARBON_CHANGED_FLUXES)


def _carbon_point(harvest, state, fields, parameters, soil, woody, crop):
    harvest, state, fields, soil, woody, crop = jax.tree.map(
        lambda x: x[None], (harvest, state, fields, soil, woody, crop)
    )
    return jax.tree.map(
        lambda x: x[0], _carbon_reference(harvest, state, fields, parameters, soil, woody, crop)
    )


@jax.custom_jvp
def _carbon_tile(*arguments):
    return fused_native_batch(
        _carbon_point,
        *arguments,
        shared_argnums=(3,),
        block_size=128,
        hardware_rounding=True,
        direct_io=True,
    )


@_carbon_tile.defjvp
def _carbon_derivative(primals, tangents):
    return jax.jvp(_carbon_reference, primals, tangents)


def carbon(inputs, state, cn_fluxes, parameters, soil, woody, crop):
    fields = cn_fluxes.patch_scalar
    if not isinstance(fields, CnFluxFields):
        fields = CnFluxFields(tuple(fields[:, k] for k in range(fields.shape[-1])))
    state, values = _carbon_tile(
        inputs.harvest_date_patch,
        state,
        fields,
        tuple(getattr(parameters, k) for k in _CARBON_PARAMETER_NAMES),
        soil,
        woody,
        crop,
    )
    for name, value in zip(_CARBON_CHANGED_FLUXES, values, strict=True):
        cn_fluxes = set_cn_flux_field(cn_fluxes, name, value)
    return state, cn_fluxes


def _nitrogen_reference(state, fields, time, soil, woody, crop):
    from clm_jax.physics.cn_state_update1 import _update_nitrogen_state_reference

    return _update_nitrogen_state_reference(state, _fluxes(fields), time, soil, woody, crop)


def _nitrogen_point(state, fields, time, soil, woody, crop):
    state, fields, soil, woody, crop = jax.tree.map(
        lambda x: x[None], (state, fields, soil, woody, crop)
    )
    return jax.tree.map(lambda x: x[0], _nitrogen_reference(state, fields, time, soil, woody, crop))


@jax.custom_jvp
def _nitrogen_tile(*arguments):
    return fused_native_batch(
        _nitrogen_point,
        *arguments,
        shared_argnums=(2,),
        block_size=128,
        hardware_rounding=True,
        direct_io=True,
    )


@_nitrogen_tile.defjvp
def _nitrogen_derivative(primals, tangents):
    return jax.jvp(_nitrogen_reference, primals, tangents)


def nitrogen(state, cn_fluxes, time, soil, woody, crop):
    fields = cn_fluxes.patch_scalar
    if not isinstance(fields, CnFluxFields):
        fields = CnFluxFields(tuple(fields[:, k] for k in range(fields.shape[-1])))
    return _nitrogen_tile(state, fields, time, soil, woody, crop)


def _precision_point(inputs, state, parameters):
    inputs, state = jax.tree.map(lambda x: x[None], (inputs, state))
    from clm_jax.physics.cn_precision_control import _cn_precision_control_reference

    result = _cn_precision_control_reference(inputs, state, parameters)
    return result._replace(state=jax.tree.map(lambda x: x[0], result.state))


@jax.custom_jvp
def precision_control(inputs, state, parameters):
    result = fused_native_batch(
        _precision_point,
        inputs,
        state,
        parameters,
        shared_argnums=(2,),
        block_size=128,
        hardware_rounding=True,
    )
    return result._replace(
        diagnostics=jax.tree.map(lambda x: jnp.sum(x, dtype=jnp.int32), result.diagnostics)
    )


@precision_control.defjvp
def _precision_derivative(primals, tangents):
    from clm_jax.physics.cn_precision_control import _cn_precision_control_reference

    return jax.jvp(_cn_precision_control_reference, primals, tangents)
