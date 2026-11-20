
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    get_cn_flux_field,
    set_cn_flux_field,
)
from clm_jax.physics.cn_zero_manifest import (
    NUMBER_OF_DECOMPOSITION_LEVELS,
    NUMBER_OF_DECOMPOSITION_POOLS,
    NUMBER_OF_DECOMPOSITION_TRANSITIONS,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.soil_biogeochem_potential import (
    DEFAULT_SOIL_BIOGEOCHEM_POTENTIAL_PARAMETERS,
)


class SoilBiogeochemNStateUpdate1Inputs(NamedTuple):

    soil_column: jnp.ndarray
    nitrogen_deposition_profile_column_level: jnp.ndarray
    nitrogen_fixation_profile_column_level: jnp.ndarray


class SoilBiogeochemNStateUpdate1State(NamedTuple):

    mineral_nitrogen_column_level: jnp.ndarray
    ammonium_column_level: jnp.ndarray
    nitrate_column_level: jnp.ndarray


class SoilBiogeochemNStateUpdate1Parameters(NamedTuple):

    time_step: float | jnp.ndarray = 1800.0
    decomposition_level_count: int | jnp.ndarray = 20
    nitrification_n2o_loss_fraction: float | jnp.ndarray = 6.0e-4
    cascade_donor_pool: jnp.ndarray = (
        DEFAULT_SOIL_BIOGEOCHEM_POTENTIAL_PARAMETERS.cascade_donor_pool
    )
    cascade_receiver_pool: jnp.ndarray = (
        DEFAULT_SOIL_BIOGEOCHEM_POTENTIAL_PARAMETERS.cascade_receiver_pool
    )


DEFAULT_SOIL_BIOGEOCHEM_N_STATE_UPDATE1_PARAMETERS = (
    SoilBiogeochemNStateUpdate1Parameters()
)


class SoilBiogeochemNStateUpdate1Diagnostics(NamedTuple):
    updated_column_count: jnp.ndarray
    updated_column_level_count: jnp.ndarray
    terminal_transition_count: jnp.ndarray


class SoilBiogeochemNStateUpdate1Result(NamedTuple):
    state: SoilBiogeochemNStateUpdate1State
    cn_fluxes: CnTransientFluxState
    diagnostics: SoilBiogeochemNStateUpdate1Diagnostics


_DECOMPOSITION_NITROGEN_SOURCE_SINK = (
    "soil_nitrogen.decomp_npools_sourcesink_col"
)
_DECOMPOSITION_NITROGEN_TRANSFER = (
    "soil_nitrogen.decomp_cascade_ntransfer_vr_col"
)
_DECOMPOSITION_MINERAL_NITROGEN = (
    "soil_nitrogen.decomp_cascade_sminn_flux_vr_col"
)


def _validate_shapes(
    inputs: SoilBiogeochemNStateUpdate1Inputs,
    state: SoilBiogeochemNStateUpdate1State,
    cn_fluxes: CnTransientFluxState,
    parameters: SoilBiogeochemNStateUpdate1Parameters,
) -> int:
    if jnp.ndim(inputs.soil_column) != 1:
        raise ValueError("soil_column must be a vector")
    column_count = inputs.soil_column.shape[0]
    level_shape = (column_count, NUMBER_OF_DECOMPOSITION_LEVELS)
    for name in (
        "nitrogen_deposition_profile_column_level",
        "nitrogen_fixation_profile_column_level",
    ):
        if jnp.shape(getattr(inputs, name)) != level_shape:
            raise ValueError(f"{name} must have shape (column, 25)")
    for name in state._fields:
        if jnp.shape(getattr(state, name)) != level_shape:
            raise ValueError(f"{name} must have shape (column, 25)")
    if cn_fluxes.column_scalar.shape[0] != column_count:
        raise ValueError("CN column fluxes must match column count")
    if cn_fluxes.column_level.shape[:2] != level_shape:
        raise ValueError("CN column-level fluxes have incompatible dimensions")
    if cn_fluxes.column_level_pool.shape[:3] != (
        column_count,
        NUMBER_OF_DECOMPOSITION_LEVELS,
        NUMBER_OF_DECOMPOSITION_POOLS,
    ):
        raise ValueError("CN column-level pool fluxes have incompatible dimensions")
    if cn_fluxes.column_level_transition.shape[:3] != (
        column_count,
        NUMBER_OF_DECOMPOSITION_LEVELS,
        NUMBER_OF_DECOMPOSITION_TRANSITIONS,
    ):
        raise ValueError(
            "CN column-level transition fluxes have incompatible dimensions"
        )
    for name in (
        "time_step",
        "decomposition_level_count",
        "nitrification_n2o_loss_fraction",
    ):
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    for name in ("cascade_donor_pool", "cascade_receiver_pool"):
        if jnp.shape(getattr(parameters, name)) != (
            NUMBER_OF_DECOMPOSITION_TRANSITIONS,
        ):
            raise ValueError(f"{name} must contain every transition")
    return column_count


def _amount(flux: jnp.ndarray, time_step: jnp.ndarray) -> jnp.ndarray:
    return fortran_multiply(flux, time_step)


def _profile_amount(
    flux: jnp.ndarray,
    time_step: jnp.ndarray,
    profile: jnp.ndarray,
) -> jnp.ndarray:
    return fortran_multiply(_amount(flux[:, None], time_step), profile)


def _masked_add(
    state: jnp.ndarray,
    amount: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    sign: int = 1,
) -> jnp.ndarray:
    if sign < 0:
        amount = -amount
    candidate = fortran_add(state, amount)
    return jnp.where(mask, candidate, state)


def _cascade_source_update(source_sink, nitrogen_transfer, mineral_flux, active,
                           time_step, parameters):
    static_cascade = isinstance(parameters.cascade_donor_pool, np.ndarray) and isinstance(
        parameters.cascade_receiver_pool, np.ndarray,
    )
    if static_cascade and jax.default_backend() == "gpu" and source_sink.size:
        from clm_jax.physics.cn_soil_sources_gpu import nitrogen_sources

        return nitrogen_sources(
            source_sink, nitrogen_transfer, mineral_flux, active, time_step,
            parameters.cascade_donor_pool, parameters.cascade_receiver_pool,
        )
    donor_pool = (parameters.cascade_donor_pool if static_cascade else
                  jnp.asarray(parameters.cascade_donor_pool, dtype=jnp.int32))
    receiver_pool = (parameters.cascade_receiver_pool if static_cascade else
                     jnp.asarray(parameters.cascade_receiver_pool, dtype=jnp.int32))
    source_fields = (
        [source_sink[:, :, pool] for pool in range(NUMBER_OF_DECOMPOSITION_POOLS)]
        if static_cascade else None
    )
    for transition in range(NUMBER_OF_DECOMPOSITION_TRANSITIONS):
        donor = donor_pool[transition]
        previous = source_fields[donor] if static_cascade else source_sink[:, :, donor]
        candidate = fortran_add(
            previous, -_amount(nitrogen_transfer[:, :, transition], time_step),
        )
        updated = jnp.where(active, candidate, previous)
        if static_cascade:
            source_fields[donor] = updated
        else:
            source_sink = source_sink.at[:, :, donor].set(updated)
    for transition in range(NUMBER_OF_DECOMPOSITION_TRANSITIONS):
        donor = donor_pool[transition]
        receiver = receiver_pool[transition]
        bounded_receiver = (
            max(receiver, 0) if static_cascade else jnp.maximum(receiver, jnp.int32(0))
        )
        previous = (source_fields[bounded_receiver] if static_cascade else
                    source_sink[:, :, bounded_receiver])
        flux = fortran_add(nitrogen_transfer[:, :, transition], mineral_flux[:, :, transition])
        candidate = fortran_add(previous, _amount(flux, time_step))
        updated = jnp.where(active & (receiver >= jnp.int32(0)), candidate, previous)
        if static_cascade:
            source_fields[bounded_receiver] = updated
        else:
            source_sink = source_sink.at[:, :, bounded_receiver].set(updated)
        previous = source_fields[donor] if static_cascade else source_sink[:, :, donor]
        candidate = fortran_add(previous, -_amount(mineral_flux[:, :, transition], time_step))
        updated = jnp.where(active & (receiver < jnp.int32(0)), candidate, previous)
        if static_cascade:
            source_fields[donor] = updated
        else:
            source_sink = source_sink.at[:, :, donor].set(updated)
    return jnp.stack(source_fields, axis=-1) if static_cascade else source_sink


def soil_biogeochem_n_state_update1(
    inputs: SoilBiogeochemNStateUpdate1Inputs,
    state: SoilBiogeochemNStateUpdate1State,
    cn_fluxes: CnTransientFluxState,
    parameters: SoilBiogeochemNStateUpdate1Parameters = (
        DEFAULT_SOIL_BIOGEOCHEM_N_STATE_UPDATE1_PARAMETERS
    ),
) -> SoilBiogeochemNStateUpdate1Result:

    _validate_shapes(inputs, state, cn_fluxes, parameters)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    level = jnp.arange(NUMBER_OF_DECOMPOSITION_LEVELS, dtype=jnp.int32)
    active = soil_column[:, None] & (
        level[None, :]
        < jnp.asarray(parameters.decomposition_level_count, dtype=jnp.int32)
    )
    time_step = jnp.asarray(parameters.time_step, dtype=jnp.float64)
    deposition_profile = jnp.asarray(
        inputs.nitrogen_deposition_profile_column_level,
        dtype=jnp.float64,
    )
    fixation_profile = jnp.asarray(
        inputs.nitrogen_fixation_profile_column_level,
        dtype=jnp.float64,
    )
    mineral_nitrogen = jnp.asarray(
        state.mineral_nitrogen_column_level,
        dtype=jnp.float64,
    )
    ammonium = jnp.asarray(state.ammonium_column_level, dtype=jnp.float64)
    nitrate = jnp.asarray(state.nitrate_column_level, dtype=jnp.float64)

    def flux(name: str) -> jnp.ndarray:
        return get_cn_flux_field(cn_fluxes, "soil_nitrogen." + name)

    ammonium = _masked_add(
        ammonium,
        _profile_amount(
            flux("ndep_to_sminn_col"),
            time_step,
            deposition_profile,
        ),
        active,
    )
    ammonium = _masked_add(
        ammonium,
        _profile_amount(
            flux("ffix_to_sminn_col"),
            time_step,
            fixation_profile,
        ),
        active,
    )

    ammonium = _masked_add(
        ammonium,
        _profile_amount(
            flux("fert_to_sminn_col"),
            time_step,
            deposition_profile,
        ),
        active,
    )
    ammonium = _masked_add(
        ammonium,
        _profile_amount(
            flux("soyfixn_to_sminn_col"),
            time_step,
            fixation_profile,
        ),
        active,
    )

    source_sink = get_cn_flux_field(
        cn_fluxes,
        _DECOMPOSITION_NITROGEN_SOURCE_SINK,
    )
    nitrogen_transfer = get_cn_flux_field(
        cn_fluxes,
        _DECOMPOSITION_NITROGEN_TRANSFER,
    )
    mineral_flux = get_cn_flux_field(
        cn_fluxes,
        _DECOMPOSITION_MINERAL_NITROGEN,
    )
    source_sink = _cascade_source_update(
        source_sink, nitrogen_transfer, mineral_flux, active, time_step, parameters,
    )
    cn_fluxes = set_cn_flux_field(
        cn_fluxes,
        _DECOMPOSITION_NITROGEN_SOURCE_SINK,
        source_sink,
    )

    ammonium = _masked_add(
        ammonium,
        _amount(flux("gross_nmin_vr_col"), time_step),
        active,
    )
    ammonium = _masked_add(
        ammonium,
        _amount(flux("actual_immob_nh4_vr_col"), time_step),
        active,
        sign=-1,
    )
    nitrate = _masked_add(
        nitrate,
        _amount(flux("actual_immob_no3_vr_col"), time_step),
        active,
        sign=-1,
    )
    ammonium = _masked_add(
        ammonium,
        _amount(flux("sminn_to_plant_fun_nh4_vr_col"), time_step),
        active,
        sign=-1,
    )
    nitrate = _masked_add(
        nitrate,
        _amount(flux("sminn_to_plant_fun_no3_vr_col"), time_step),
        active,
        sign=-1,
    )
    nitrification = flux("f_nit_vr_col")
    ammonium = _masked_add(
        ammonium,
        _amount(nitrification, time_step),
        active,
        sign=-1,
    )
    retained_nitrification = fortran_multiply(
        _amount(nitrification, time_step),
        fortran_add(
            jnp.float64(1.0),
            -jnp.asarray(
                parameters.nitrification_n2o_loss_fraction,
                dtype=jnp.float64,
            ),
        ),
    )
    nitrate = _masked_add(nitrate, retained_nitrification, active)
    nitrate = _masked_add(
        nitrate,
        _amount(flux("f_denit_vr_col"), time_step),
        active,
        sign=-1,
    )
    ammonium = _masked_add(
        ammonium,
        _amount(flux("supplement_to_sminn_vr_col"), time_step),
        active,
    )
    mineral_candidate = fortran_add(ammonium, nitrate)
    mineral_nitrogen = jnp.where(active, mineral_candidate, mineral_nitrogen)

    receiver = jnp.asarray(parameters.cascade_receiver_pool, dtype=jnp.int32)
    return SoilBiogeochemNStateUpdate1Result(
        state=SoilBiogeochemNStateUpdate1State(
            mineral_nitrogen_column_level=mineral_nitrogen,
            ammonium_column_level=ammonium,
            nitrate_column_level=nitrate,
        ),
        cn_fluxes=cn_fluxes,
        diagnostics=SoilBiogeochemNStateUpdate1Diagnostics(
            updated_column_count=jnp.sum(soil_column, dtype=jnp.int32),
            updated_column_level_count=jnp.sum(active, dtype=jnp.int32),
            terminal_transition_count=jnp.sum(
                receiver < jnp.int32(0),
                dtype=jnp.int32,
            ),
        ),
    )


__all__ = [
    "DEFAULT_SOIL_BIOGEOCHEM_N_STATE_UPDATE1_PARAMETERS",
    "SoilBiogeochemNStateUpdate1Diagnostics",
    "SoilBiogeochemNStateUpdate1Inputs",
    "SoilBiogeochemNStateUpdate1Parameters",
    "SoilBiogeochemNStateUpdate1Result",
    "SoilBiogeochemNStateUpdate1State",
    "soil_biogeochem_n_state_update1",
]
