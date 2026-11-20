
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import lax

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
    fortran_divide,
    ordered_indexed_fold,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)

_DECOMPOSITION_RATE_FIELD = "soil_carbon.decomp_k_col"
_GROSS_N_MINERALIZATION_FIELD = "soil_nitrogen.gross_nmin_vr_col"
_POTENTIAL_IMMOBILIZATION_FIELD = "soil_nitrogen.potential_immob_vr_col"


class SoilBiogeochemPotentialInputs(NamedTuple):

    soil_column: jnp.ndarray
    decomposition_carbon_pool_column_level_pool: jnp.ndarray
    decomposition_nitrogen_pool_column_level_pool: jnp.ndarray
    respiration_fraction_column_level_transition: jnp.ndarray
    path_fraction_column_level_transition: jnp.ndarray


class SoilBiogeochemPotentialParameters(NamedTuple):

    decomposition_level_count: int | jnp.ndarray = 20
    decomposition_pool_count: int | jnp.ndarray = 7
    decomposition_transition_count: int | jnp.ndarray = 10
    use_fates: bool | jnp.ndarray = False
    cascade_donor_pool: jnp.ndarray = np.asarray(
        [0, 1, 2, 4, 4, 5, 5, 6, 3, 3],
        dtype=np.int32,
    )
    cascade_receiver_pool: jnp.ndarray = np.asarray(
        [4, 4, 5, 5, 6, 4, 6, 4, 1, 2],
        dtype=np.int32,
    )
    floating_cn_ratio_pool: jnp.ndarray = np.asarray(
        [True, True, True, True, False, False, False],
        dtype=np.bool_,
    )
    initial_cn_ratio_pool: jnp.ndarray = np.asarray(
        [90.0, 90.0, 90.0, 90.0, 8.0, 11.0, 11.0],
        dtype=np.float64,
    )
    denitrification_proportion: float | jnp.ndarray = 0.01


DEFAULT_SOIL_BIOGEOCHEM_POTENTIAL_PARAMETERS = SoilBiogeochemPotentialParameters()


class SoilBiogeochemPotentialWorkState(NamedTuple):

    carbon_nitrogen_ratio_column_level_pool: jnp.ndarray
    potential_carbon_loss_column_level_transition: jnp.ndarray
    potential_mineral_nitrogen_flux_column_level_transition: jnp.ndarray
    potential_heterotrophic_respiration_column_level: jnp.ndarray


class SoilBiogeochemPotentialDiagnostics(NamedTuple):
    updated_column_count: jnp.ndarray
    positive_immobilization_transition_count: jnp.ndarray
    mineralization_transition_count: jnp.ndarray


class SoilBiogeochemPotentialResult(NamedTuple):
    cn_fluxes: CnTransientFluxState
    work: SoilBiogeochemPotentialWorkState
    diagnostics: SoilBiogeochemPotentialDiagnostics


def initialize_soil_biogeochem_potential_work_state(
    number_of_columns: int,
    *,
    fill_value: float = 1.0e36,
) -> SoilBiogeochemPotentialWorkState:

    if number_of_columns < 0:
        raise ValueError("column count cannot be negative")
    level_pool_shape = (
        number_of_columns,
        NUMBER_OF_DECOMPOSITION_LEVELS,
        NUMBER_OF_DECOMPOSITION_POOLS,
    )
    level_transition_shape = (
        number_of_columns,
        NUMBER_OF_DECOMPOSITION_LEVELS,
        NUMBER_OF_DECOMPOSITION_TRANSITIONS,
    )
    return SoilBiogeochemPotentialWorkState(
        carbon_nitrogen_ratio_column_level_pool=jnp.full(
            level_pool_shape,
            fill_value,
            dtype=jnp.float64,
        ),
        potential_carbon_loss_column_level_transition=jnp.full(
            level_transition_shape,
            fill_value,
            dtype=jnp.float64,
        ),
        potential_mineral_nitrogen_flux_column_level_transition=jnp.full(
            level_transition_shape,
            fill_value,
            dtype=jnp.float64,
        ),
        potential_heterotrophic_respiration_column_level=jnp.full(
            (number_of_columns, NUMBER_OF_DECOMPOSITION_LEVELS),
            fill_value,
            dtype=jnp.float64,
        ),
    )


def _validate_shapes(
    inputs: SoilBiogeochemPotentialInputs,
    cn_fluxes: CnTransientFluxState,
    work: SoilBiogeochemPotentialWorkState,
) -> int:
    if jnp.ndim(inputs.soil_column) != 1:
        raise ValueError("soil_column must be a vector")
    number_of_columns = inputs.soil_column.shape[0]
    level_pool_shape = (
        number_of_columns,
        NUMBER_OF_DECOMPOSITION_LEVELS,
        NUMBER_OF_DECOMPOSITION_POOLS,
    )
    level_transition_shape = (
        number_of_columns,
        NUMBER_OF_DECOMPOSITION_LEVELS,
        NUMBER_OF_DECOMPOSITION_TRANSITIONS,
    )
    for name in (
        "decomposition_carbon_pool_column_level_pool",
        "decomposition_nitrogen_pool_column_level_pool",
    ):
        if jnp.shape(getattr(inputs, name)) != level_pool_shape:
            raise ValueError(f"{name} must have shape (column, 25, 7)")
    for name in (
        "respiration_fraction_column_level_transition",
        "path_fraction_column_level_transition",
    ):
        if jnp.shape(getattr(inputs, name)) != level_transition_shape:
            raise ValueError(f"{name} must have shape (column, 25, 10)")
    expected_work_shapes = {
        "carbon_nitrogen_ratio_column_level_pool": level_pool_shape,
        "potential_carbon_loss_column_level_transition": level_transition_shape,
        "potential_mineral_nitrogen_flux_column_level_transition": (
            level_transition_shape
        ),
        "potential_heterotrophic_respiration_column_level": (
            number_of_columns,
            NUMBER_OF_DECOMPOSITION_LEVELS,
        ),
    }
    for name, shape in expected_work_shapes.items():
        if jnp.shape(getattr(work, name)) != shape:
            raise ValueError(f"{name} has an incompatible shape")
    if get_cn_flux_field(cn_fluxes, _DECOMPOSITION_RATE_FIELD).shape != (
        level_transition_shape
    ):
        raise ValueError("decomp_k_col must have shape (column, 25, 10)")
    for name in (
        _GROSS_N_MINERALIZATION_FIELD,
        _POTENTIAL_IMMOBILIZATION_FIELD,
    ):
        if get_cn_flux_field(cn_fluxes, name).shape != (
            number_of_columns,
            NUMBER_OF_DECOMPOSITION_LEVELS,
        ):
            raise ValueError(f"{name} must have shape (column, 25)")
    return number_of_columns


def _validate_parameters(parameters: SoilBiogeochemPotentialParameters) -> None:
    for name in (
        "decomposition_level_count",
        "decomposition_pool_count",
        "decomposition_transition_count",
        "use_fates",
        "denitrification_proportion",
    ):
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    expected_shapes = {
        "cascade_donor_pool": (NUMBER_OF_DECOMPOSITION_TRANSITIONS,),
        "cascade_receiver_pool": (NUMBER_OF_DECOMPOSITION_TRANSITIONS,),
        "floating_cn_ratio_pool": (NUMBER_OF_DECOMPOSITION_POOLS,),
        "initial_cn_ratio_pool": (NUMBER_OF_DECOMPOSITION_POOLS,),
    }
    for name, shape in expected_shapes.items():
        if jnp.shape(getattr(parameters, name)) != shape:
            raise ValueError(f"{name} must have shape {shape}")

    level_count = parameters.decomposition_level_count
    pool_count = parameters.decomposition_pool_count
    transition_count = parameters.decomposition_transition_count
    if isinstance(level_count, (int, np.integer)) and not (
        1 <= level_count <= NUMBER_OF_DECOMPOSITION_LEVELS
    ):
        raise ValueError("decomposition_level_count must lie in [1, 25]")
    if isinstance(pool_count, (int, np.integer)) and not (
        1 <= pool_count <= NUMBER_OF_DECOMPOSITION_POOLS
    ):
        raise ValueError("decomposition_pool_count must lie in [1, 7]")
    if isinstance(transition_count, (int, np.integer)) and not (
        1 <= transition_count <= NUMBER_OF_DECOMPOSITION_TRANSITIONS
    ):
        raise ValueError("decomposition_transition_count must lie in [1, 10]")


def _active_masks(
    parameters: SoilBiogeochemPotentialParameters,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    level = jnp.arange(NUMBER_OF_DECOMPOSITION_LEVELS, dtype=jnp.int32)
    pool = jnp.arange(NUMBER_OF_DECOMPOSITION_POOLS, dtype=jnp.int32)
    transition = jnp.arange(NUMBER_OF_DECOMPOSITION_TRANSITIONS, dtype=jnp.int32)
    return (
        level < jnp.asarray(parameters.decomposition_level_count, dtype=jnp.int32),
        pool < jnp.asarray(parameters.decomposition_pool_count, dtype=jnp.int32),
        transition
        < jnp.asarray(parameters.decomposition_transition_count, dtype=jnp.int32),
    )


def _non_fates_potential(
    inputs, cn_fluxes, work, parameters, soil, active_level, active_pool, active_transition,
):
    arguments = (
        inputs, work, parameters, soil, active_level, active_pool, active_transition,
        get_cn_flux_field(cn_fluxes, _DECOMPOSITION_RATE_FIELD),
        get_cn_flux_field(cn_fluxes, _GROSS_N_MINERALIZATION_FIELD),
        get_cn_flux_field(cn_fluxes, _POTENTIAL_IMMOBILIZATION_FIELD),
    )
    from jax import default_backend
    if default_backend() == "gpu":
        from clm_jax.physics.soil_biogeochem_potential_gpu import non_fates_arrays
        values = non_fates_arrays(*arguments)
    else:
        values = _non_fates_arrays(*arguments)
    ratio, loss, mineral, gross, immobilization = values
    cn_fluxes = set_cn_flux_field(cn_fluxes, _GROSS_N_MINERALIZATION_FIELD, gross)
    cn_fluxes = set_cn_flux_field(cn_fluxes, _POTENTIAL_IMMOBILIZATION_FIELD, immobilization)
    return cn_fluxes, ratio, loss, mineral


def _non_fates_arrays(
    inputs: SoilBiogeochemPotentialInputs,
    work: SoilBiogeochemPotentialWorkState,
    parameters: SoilBiogeochemPotentialParameters,
    soil: jnp.ndarray,
    active_level: jnp.ndarray,
    active_pool: jnp.ndarray,
    active_transition: jnp.ndarray,
    decomposition_rate: jnp.ndarray,
    gross: jnp.ndarray,
    potential_immobilization: jnp.ndarray,
):
    carbon = jnp.asarray(
        inputs.decomposition_carbon_pool_column_level_pool,
        dtype=jnp.float64,
    )
    nitrogen = jnp.asarray(
        inputs.decomposition_nitrogen_pool_column_level_pool,
        dtype=jnp.float64,
    )
    respiration_fraction = jnp.asarray(
        inputs.respiration_fraction_column_level_transition,
        dtype=jnp.float64,
    )
    path_fraction = jnp.asarray(
        inputs.path_fraction_column_level_transition,
        dtype=jnp.float64,
    )

    production_pool = active_level[None, :, None] & active_pool[None, None, :]
    filtered_pool = soil[:, None, None] & production_pool
    floating = jnp.asarray(parameters.floating_cn_ratio_pool, dtype=jnp.bool_)
    initial_ratio = jnp.asarray(parameters.initial_cn_ratio_pool, dtype=jnp.float64)
    positive_nitrogen = nitrogen > jnp.float64(0.0)
    safe_nitrogen = jnp.where(positive_nitrogen, nitrogen, jnp.float64(1.0))
    floating_ratio = fortran_divide(carbon, safe_nitrogen)
    candidate_ratio = jnp.where(
        floating[None, None, :],
        floating_ratio,
        initial_ratio[None, None, :],
    )
    ratio_write = filtered_pool & (
        ~floating[None, None, :] | positive_nitrogen
    )
    carbon_nitrogen_ratio = jnp.where(
        ratio_write,
        candidate_ratio,
        jnp.asarray(work.carbon_nitrogen_ratio_column_level_pool, dtype=jnp.float64),
    )

    donor = jnp.asarray(parameters.cascade_donor_pool, dtype=jnp.int32)
    receiver = jnp.asarray(parameters.cascade_receiver_pool, dtype=jnp.int32)
    safe_donor = jnp.clip(donor, 0, NUMBER_OF_DECOMPOSITION_POOLS - 1)
    safe_receiver = jnp.clip(receiver, 0, NUMBER_OF_DECOMPOSITION_POOLS - 1)
    donor_carbon = jnp.take(carbon, safe_donor, axis=2)
    donor_nitrogen = jnp.take(nitrogen, safe_donor, axis=2)
    donor_rate = jnp.take(decomposition_rate, safe_donor, axis=2)
    donor_ratio = jnp.take(carbon_nitrogen_ratio, safe_donor, axis=2)
    receiver_ratio = jnp.take(carbon_nitrogen_ratio, safe_receiver, axis=2)
    receiver_floating = jnp.take(floating, safe_receiver, axis=0)

    production_transition = (
        active_level[None, :, None] & active_transition[None, None, :]
    )
    potential_carbon_loss = jnp.where(
        production_transition,
        jnp.float64(0.0),
        jnp.asarray(
            work.potential_carbon_loss_column_level_transition,
            dtype=jnp.float64,
        ),
    )
    potential_mineral_flux = jnp.where(
        production_transition,
        jnp.float64(0.0),
        jnp.asarray(
            work.potential_mineral_nitrogen_flux_column_level_transition,
            dtype=jnp.float64,
        ),
    )
    decomposes = (
        soil[:, None, None]
        & production_transition
        & (donor_carbon > jnp.float64(0.0))
        & (donor_rate > jnp.float64(0.0))
    )
    raw_loss = fortran_multiply(
        fortran_multiply(donor_carbon, donor_rate),
        path_fraction,
    )
    potential_carbon_loss = jnp.where(
        decomposes,
        raw_loss,
        potential_carbon_loss,
    )

    positive_donor_nitrogen = donor_nitrogen > jnp.float64(0.0)
    safe_donor_ratio = jnp.where(
        positive_donor_nitrogen,
        donor_ratio,
        jnp.float64(1.0),
    )
    transfer_ratio = jnp.where(
        positive_donor_nitrogen,
        fortran_divide(receiver_ratio, safe_donor_ratio),
        jnp.float64(0.0),
    )
    retained_fraction = fortran_add(
        fortran_add(jnp.float64(1.0), -respiration_fraction),
        -transfer_ratio,
    )
    receiver_is_atmosphere = receiver < jnp.int32(0)
    fixed_receiver = receiver_is_atmosphere | ~receiver_floating
    atmosphere_write = (
        decomposes & receiver_is_atmosphere[None, None, :]
    )
    non_atmosphere_write = (
        decomposes
        & fixed_receiver[None, None, :]
        & ~receiver_is_atmosphere[None, None, :]
    )
    receiver_denominator = jnp.where(
        non_atmosphere_write,
        receiver_ratio,
        jnp.float64(1.0),
    )
    non_atmosphere_flux = fortran_divide(
        fortran_multiply(potential_carbon_loss, retained_fraction),
        receiver_denominator,
    )
    donor_denominator = jnp.where(
        atmosphere_write,
        donor_ratio,
        jnp.float64(1.0),
    )
    atmosphere_flux = fortran_divide(
        -potential_carbon_loss,
        donor_denominator,
    )
    candidate_mineral_flux = jnp.where(
        receiver_is_atmosphere[None, None, :],
        atmosphere_flux,
        non_atmosphere_flux,
    )
    mineral_write = decomposes & fixed_receiver[None, None, :]
    potential_mineral_flux = jnp.where(
        mineral_write,
        candidate_mineral_flux,
        potential_mineral_flux,
    )

    immobilization = jnp.zeros_like(gross)

    def accumulate_nitrogen(
        state: tuple[jnp.ndarray, jnp.ndarray],
        transition_index: jnp.ndarray,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
        current_immobilization, current_gross = state
        flux = potential_mineral_flux[:, :, transition_index]
        active = (
            soil[:, None]
            & active_level[None, :]
            & active_transition[transition_index]
        )
        positive = flux > jnp.float64(0.0)
        current_immobilization = jnp.where(
            active & positive,
            fortran_add(current_immobilization, flux),
            current_immobilization,
        )
        current_gross = jnp.where(
            active & ~positive,
            fortran_add(current_gross, -flux),
            current_gross,
        )
        return (current_immobilization, current_gross), None

    immobilization, gross = ordered_indexed_fold(
        accumulate_nitrogen,
        (immobilization, gross),
        NUMBER_OF_DECOMPOSITION_TRANSITIONS,
    )
    potential_immobilization = jnp.where(
        soil[:, None] & active_level[None, :],
        immobilization,
        potential_immobilization,
    )
    return (
        carbon_nitrogen_ratio,
        potential_carbon_loss,
        potential_mineral_flux,
        gross,
        potential_immobilization,
    )


def _fates_potential(
    inputs: SoilBiogeochemPotentialInputs,
    cn_fluxes: CnTransientFluxState,
    work: SoilBiogeochemPotentialWorkState,
    parameters: SoilBiogeochemPotentialParameters,
    soil: jnp.ndarray,
    active_level: jnp.ndarray,
    active_pool: jnp.ndarray,
    active_transition: jnp.ndarray,
) -> tuple[CnTransientFluxState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    del active_pool
    carbon = jnp.asarray(
        inputs.decomposition_carbon_pool_column_level_pool,
        dtype=jnp.float64,
    )
    path_fraction = jnp.asarray(
        inputs.path_fraction_column_level_transition,
        dtype=jnp.float64,
    )
    decomposition_rate = get_cn_flux_field(cn_fluxes, _DECOMPOSITION_RATE_FIELD)
    donor = jnp.clip(
        jnp.asarray(parameters.cascade_donor_pool, dtype=jnp.int32),
        0,
        NUMBER_OF_DECOMPOSITION_POOLS - 1,
    )
    raw_loss = fortran_multiply(
        fortran_multiply(
            jnp.take(carbon, donor, axis=2),
            jnp.take(decomposition_rate, donor, axis=2),
        ),
        path_fraction,
    )
    active = (
        soil[:, None, None]
        & active_level[None, :, None]
        & active_transition[None, None, :]
    )
    potential_carbon_loss = jnp.where(
        active,
        raw_loss,
        jnp.asarray(
            work.potential_carbon_loss_column_level_transition,
            dtype=jnp.float64,
        ),
    )
    return (
        cn_fluxes,
        jnp.asarray(work.carbon_nitrogen_ratio_column_level_pool, dtype=jnp.float64),
        potential_carbon_loss,
        jnp.asarray(
            work.potential_mineral_nitrogen_flux_column_level_transition,
            dtype=jnp.float64,
        ),
    )


def soil_biogeochem_potential(
    inputs: SoilBiogeochemPotentialInputs,
    cn_fluxes: CnTransientFluxState,
    work: SoilBiogeochemPotentialWorkState,
    parameters: SoilBiogeochemPotentialParameters = (
        DEFAULT_SOIL_BIOGEOCHEM_POTENTIAL_PARAMETERS
    ),
) -> SoilBiogeochemPotentialResult:

    _validate_shapes(inputs, cn_fluxes, work)
    _validate_parameters(parameters)
    soil = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    active_level, active_pool, active_transition = _active_masks(parameters)
    branch_arguments = (
        inputs,
        cn_fluxes,
        work,
        parameters,
        soil,
        active_level,
        active_pool,
        active_transition,
    )
    cn_fluxes, carbon_nitrogen_ratio, potential_carbon_loss, potential_mineral_flux = (
        lax.cond(
            jnp.asarray(parameters.use_fates, dtype=jnp.bool_),
            lambda values: _fates_potential(*values),
            lambda values: _non_fates_potential(*values),
            branch_arguments,
        )
    )

    respiration_fraction = jnp.asarray(
        inputs.respiration_fraction_column_level_transition,
        dtype=jnp.float64,
    )
    potential_respiration = jnp.where(
        soil[:, None] & active_level[None, :],
        jnp.float64(0.0),
        jnp.asarray(
            work.potential_heterotrophic_respiration_column_level,
            dtype=jnp.float64,
        ),
    )

    def accumulate_respiration(
        current: jnp.ndarray,
        transition_index: jnp.ndarray,
    ) -> tuple[jnp.ndarray, None]:
        contribution = fortran_multiply(
            respiration_fraction[:, :, transition_index],
            potential_carbon_loss[:, :, transition_index],
        )
        active = (
            soil[:, None]
            & active_level[None, :]
            & active_transition[transition_index]
        )
        current = jnp.where(
            active,
            fortran_add(current, contribution),
            current,
        )
        return current, None

    potential_respiration = ordered_indexed_fold(
        accumulate_respiration,
        potential_respiration,
        NUMBER_OF_DECOMPOSITION_TRANSITIONS,
    )
    positive = potential_mineral_flux > jnp.float64(0.0)
    active_flux = (
        soil[:, None, None]
        & active_level[None, :, None]
        & active_transition[None, None, :]
    )
    return SoilBiogeochemPotentialResult(
        cn_fluxes=cn_fluxes,
        work=SoilBiogeochemPotentialWorkState(
            carbon_nitrogen_ratio_column_level_pool=carbon_nitrogen_ratio,
            potential_carbon_loss_column_level_transition=potential_carbon_loss,
            potential_mineral_nitrogen_flux_column_level_transition=(
                potential_mineral_flux
            ),
            potential_heterotrophic_respiration_column_level=(
                potential_respiration
            ),
        ),
        diagnostics=SoilBiogeochemPotentialDiagnostics(
            updated_column_count=jnp.sum(soil, dtype=jnp.int32),
            positive_immobilization_transition_count=jnp.sum(
                active_flux & positive,
                dtype=jnp.int32,
            ),
            mineralization_transition_count=jnp.sum(
                active_flux & ~positive,
                dtype=jnp.int32,
            ),
        ),
    )


__all__ = [
    "DEFAULT_SOIL_BIOGEOCHEM_POTENTIAL_PARAMETERS",
    "SoilBiogeochemPotentialDiagnostics",
    "SoilBiogeochemPotentialInputs",
    "SoilBiogeochemPotentialParameters",
    "SoilBiogeochemPotentialResult",
    "SoilBiogeochemPotentialWorkState",
    "initialize_soil_biogeochem_potential_work_state",
    "soil_biogeochem_potential",
]
