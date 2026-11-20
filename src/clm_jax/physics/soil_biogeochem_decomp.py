
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
from clm_jax.physics.soil_biogeochem_potential import (
    SoilBiogeochemPotentialWorkState,
)

_HETEROTROPHIC_RESPIRATION = "soil_carbon.decomp_cascade_hr_vr_col"
_CARBON_TRANSFER = "soil_carbon.decomp_cascade_ctransfer_vr_col"
_NITROGEN_TRANSFER = "soil_nitrogen.decomp_cascade_ntransfer_vr_col"
_MINERAL_NITROGEN_FLUX = (
    "soil_nitrogen.decomp_cascade_sminn_flux_vr_col"
)
_GROSS_MINERALIZATION_LEVEL = "soil_nitrogen.gross_nmin_vr_col"
_NET_MINERALIZATION_LEVEL = "soil_nitrogen.net_nmin_vr_col"
_GROSS_MINERALIZATION = "soil_nitrogen.gross_nmin_col"
_NET_MINERALIZATION = "soil_nitrogen.net_nmin_col"


class SoilBiogeochemDecompInputs(NamedTuple):

    soil_column: jnp.ndarray
    decomposition_carbon_pool_column_level_pool: jnp.ndarray
    decomposition_nitrogen_pool_column_level_pool: jnp.ndarray
    fraction_potential_immobilization_column_level: jnp.ndarray
    respiration_fraction_column_level_transition: jnp.ndarray
    water_scalar_column_level: jnp.ndarray


class SoilBiogeochemDecompState(NamedTuple):

    cn_fluxes: CnTransientFluxState
    fraction_potential_heterotrophic_respiration_column_level: jnp.ndarray
    denitrification_column_level_transition: jnp.ndarray


class SoilBiogeochemDecompParameters(NamedTuple):

    decomposition_level_count: int | jnp.ndarray = 20
    decomposition_pool_count: int | jnp.ndarray = 7
    decomposition_transition_count: int | jnp.ndarray = 10
    use_fates: bool | jnp.ndarray = False
    use_methane: bool | jnp.ndarray = True
    use_nitrification_denitrification: bool | jnp.ndarray = True
    denitrification_proportion: float | jnp.ndarray = 0.01
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
    soil_layer_thickness: jnp.ndarray = np.ones(
        NUMBER_OF_DECOMPOSITION_LEVELS,
        dtype=np.float64,
    )


DEFAULT_SOIL_BIOGEOCHEM_DECOMP_PARAMETERS = SoilBiogeochemDecompParameters()


class SoilBiogeochemDecompDiagnostics(NamedTuple):
    updated_column_count: jnp.ndarray
    positive_immobilization_transition_count: jnp.ndarray
    decomposing_transition_count: jnp.ndarray
    methane_diagnostic_cell_count: jnp.ndarray


class SoilBiogeochemDecompResult(NamedTuple):
    state: SoilBiogeochemDecompState
    work: SoilBiogeochemPotentialWorkState
    diagnostics: SoilBiogeochemDecompDiagnostics


class SoilBiogeochemDecompCoreResult(NamedTuple):

    heterotrophic_respiration_column_level_transition: jnp.ndarray
    carbon_transfer_column_level_transition: jnp.ndarray
    nitrogen_transfer_column_level_transition: jnp.ndarray
    mineral_nitrogen_flux_column_level_transition: jnp.ndarray
    net_mineralization_column_level: jnp.ndarray
    gross_mineralization_column: jnp.ndarray
    net_mineralization_column: jnp.ndarray
    fraction_potential_heterotrophic_respiration_column_level: jnp.ndarray
    denitrification_column_level_transition: jnp.ndarray
    work: SoilBiogeochemPotentialWorkState
    diagnostics: SoilBiogeochemDecompDiagnostics


def initialize_soil_biogeochem_decomp_state(
    cn_fluxes: CnTransientFluxState,
    *,
    fill_value: float = 1.0e36,
) -> SoilBiogeochemDecompState:

    column_count = cn_fluxes.column_scalar.shape[0]
    return SoilBiogeochemDecompState(
        cn_fluxes=cn_fluxes,
        fraction_potential_heterotrophic_respiration_column_level=jnp.full(
            (column_count, NUMBER_OF_DECOMPOSITION_LEVELS),
            fill_value,
            dtype=jnp.float64,
        ),
        denitrification_column_level_transition=jnp.full(
            (
                column_count,
                NUMBER_OF_DECOMPOSITION_LEVELS,
                NUMBER_OF_DECOMPOSITION_TRANSITIONS,
            ),
            fill_value,
            dtype=jnp.float64,
        ),
    )


def _validate_shapes(
    inputs: SoilBiogeochemDecompInputs,
    state: SoilBiogeochemDecompState,
    work: SoilBiogeochemPotentialWorkState,
) -> int:
    if jnp.ndim(inputs.soil_column) != 1:
        raise ValueError("soil_column must be a vector")
    column_count = inputs.soil_column.shape[0]
    level_shape = (column_count, NUMBER_OF_DECOMPOSITION_LEVELS)
    pool_shape = (*level_shape, NUMBER_OF_DECOMPOSITION_POOLS)
    transition_shape = (
        *level_shape,
        NUMBER_OF_DECOMPOSITION_TRANSITIONS,
    )
    for name in (
        "decomposition_carbon_pool_column_level_pool",
        "decomposition_nitrogen_pool_column_level_pool",
    ):
        if jnp.shape(getattr(inputs, name)) != pool_shape:
            raise ValueError(f"{name} must have shape (column, 25, 7)")
    for name in (
        "fraction_potential_immobilization_column_level",
        "water_scalar_column_level",
    ):
        if jnp.shape(getattr(inputs, name)) != level_shape:
            raise ValueError(f"{name} must have shape (column, 25)")
    if jnp.shape(inputs.respiration_fraction_column_level_transition) != (
        transition_shape
    ):
        raise ValueError(
            "respiration_fraction_column_level_transition must have shape "
            "(column, 25, 10)"
        )
    if state.cn_fluxes.column_scalar.shape[0] != column_count:
        raise ValueError("packed CN state has an incompatible column count")
    if jnp.shape(
        state.fraction_potential_heterotrophic_respiration_column_level
    ) != level_shape:
        raise ValueError("fraction potential HR must have shape (column, 25)")
    if jnp.shape(state.denitrification_column_level_transition) != (
        transition_shape
    ):
        raise ValueError("denitrification must have shape (column, 25, 10)")
    expected_work = {
        "carbon_nitrogen_ratio_column_level_pool": pool_shape,
        "potential_carbon_loss_column_level_transition": transition_shape,
        "potential_mineral_nitrogen_flux_column_level_transition": (
            transition_shape
        ),
        "potential_heterotrophic_respiration_column_level": level_shape,
    }
    for name, shape in expected_work.items():
        if jnp.shape(getattr(work, name)) != shape:
            raise ValueError(f"work field {name} has an incompatible shape")
    for field in (
        _HETEROTROPHIC_RESPIRATION,
        _CARBON_TRANSFER,
        _NITROGEN_TRANSFER,
        _MINERAL_NITROGEN_FLUX,
    ):
        if get_cn_flux_field(state.cn_fluxes, field).shape != transition_shape:
            raise ValueError(f"{field} must have shape (column, 25, 10)")
    for field in (
        _GROSS_MINERALIZATION_LEVEL,
        _NET_MINERALIZATION_LEVEL,
    ):
        if get_cn_flux_field(state.cn_fluxes, field).shape != level_shape:
            raise ValueError(f"{field} must have shape (column, 25)")
    for field in (_GROSS_MINERALIZATION, _NET_MINERALIZATION):
        if get_cn_flux_field(state.cn_fluxes, field).shape != (column_count,):
            raise ValueError(f"{field} must contain every column")
    return column_count


def _validate_parameters(parameters: SoilBiogeochemDecompParameters) -> None:
    for name in SoilBiogeochemDecompParameters._fields[:7]:
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    expected = {
        "cascade_donor_pool": (NUMBER_OF_DECOMPOSITION_TRANSITIONS,),
        "cascade_receiver_pool": (NUMBER_OF_DECOMPOSITION_TRANSITIONS,),
        "floating_cn_ratio_pool": (NUMBER_OF_DECOMPOSITION_POOLS,),
        "initial_cn_ratio_pool": (NUMBER_OF_DECOMPOSITION_POOLS,),
        "soil_layer_thickness": (NUMBER_OF_DECOMPOSITION_LEVELS,),
    }
    for name, shape in expected.items():
        if jnp.shape(getattr(parameters, name)) != shape:
            raise ValueError(f"{name} must have shape {shape!r}")
    counts = (
        (parameters.decomposition_level_count, 1, NUMBER_OF_DECOMPOSITION_LEVELS),
        (parameters.decomposition_pool_count, 1, NUMBER_OF_DECOMPOSITION_POOLS),
        (
            parameters.decomposition_transition_count,
            1,
            NUMBER_OF_DECOMPOSITION_TRANSITIONS,
        ),
    )
    for value, minimum, maximum in counts:
        if isinstance(value, (int, np.integer)) and not (
            minimum <= value <= maximum
        ):
            raise ValueError("active decomposition count is out of bounds")


def _active_masks(
    parameters: SoilBiogeochemDecompParameters,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return (
        jnp.arange(NUMBER_OF_DECOMPOSITION_LEVELS, dtype=jnp.int32)
        < jnp.asarray(parameters.decomposition_level_count, dtype=jnp.int32),
        jnp.arange(NUMBER_OF_DECOMPOSITION_POOLS, dtype=jnp.int32)
        < jnp.asarray(parameters.decomposition_pool_count, dtype=jnp.int32),
        jnp.arange(NUMBER_OF_DECOMPOSITION_TRANSITIONS, dtype=jnp.int32)
        < jnp.asarray(parameters.decomposition_transition_count, dtype=jnp.int32),
    )


def _carbon_nitrogen_ratios(
    inputs: SoilBiogeochemDecompInputs,
    previous: jnp.ndarray,
    parameters: SoilBiogeochemDecompParameters,
    soil: jnp.ndarray,
    active_level: jnp.ndarray,
    active_pool: jnp.ndarray,
) -> jnp.ndarray:
    carbon = jnp.asarray(
        inputs.decomposition_carbon_pool_column_level_pool,
        dtype=jnp.float64,
    )
    nitrogen = jnp.asarray(
        inputs.decomposition_nitrogen_pool_column_level_pool,
        dtype=jnp.float64,
    )
    floating = jnp.asarray(parameters.floating_cn_ratio_pool, dtype=jnp.bool_)
    positive_nitrogen = nitrogen > jnp.float64(0.0)
    safe_nitrogen = jnp.where(positive_nitrogen, nitrogen, jnp.float64(1.0))
    candidate = jnp.where(
        floating[None, None, :],
        fortran_divide(carbon, safe_nitrogen),
        jnp.asarray(parameters.initial_cn_ratio_pool, dtype=jnp.float64)[
            None, None, :
        ],
    )
    active = (
        soil[:, None, None]
        & active_level[None, :, None]
        & active_pool[None, None, :]
    )
    write = active & (~floating[None, None, :] | positive_nitrogen)
    updated = jnp.where(write, candidate, previous)
    return lax.cond(
        jnp.asarray(parameters.use_fates, dtype=jnp.bool_),
        lambda _: previous,
        lambda _: updated,
        operand=None,
    )


def soil_biogeochem_decomp_core(
    inputs: SoilBiogeochemDecompInputs,
    state: SoilBiogeochemDecompState,
    work: SoilBiogeochemPotentialWorkState,
    parameters: SoilBiogeochemDecompParameters = (
        DEFAULT_SOIL_BIOGEOCHEM_DECOMP_PARAMETERS
    ),
) -> SoilBiogeochemDecompCoreResult:

    _validate_shapes(inputs, state, work)
    _validate_parameters(parameters)
    soil = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    active_level, active_pool, active_transition = _active_masks(parameters)
    active = (
        soil[:, None, None]
        & active_level[None, :, None]
        & active_transition[None, None, :]
    )
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
    donor = jnp.clip(
        jnp.asarray(parameters.cascade_donor_pool, dtype=jnp.int32),
        0,
        NUMBER_OF_DECOMPOSITION_POOLS - 1,
    )
    receiver = jnp.asarray(parameters.cascade_receiver_pool, dtype=jnp.int32)
    donor_carbon = jnp.take(carbon, donor, axis=2)
    donor_nitrogen = jnp.take(nitrogen, donor, axis=2)
    donor_positive = donor_carbon > jnp.float64(0.0)
    positive_donor_nitrogen = donor_nitrogen > jnp.float64(0.0)
    receiver_is_atmosphere = receiver < jnp.int32(0)

    carbon_nitrogen_ratio = _carbon_nitrogen_ratios(
        inputs,
        jnp.asarray(
            work.carbon_nitrogen_ratio_column_level_pool,
            dtype=jnp.float64,
        ),
        parameters,
        soil,
        active_level,
        active_pool,
    )
    donor_ratio = jnp.take(carbon_nitrogen_ratio, donor, axis=2)
    potential_carbon_loss = jnp.asarray(
        work.potential_carbon_loss_column_level_transition,
        dtype=jnp.float64,
    )
    potential_mineral_flux = jnp.asarray(
        work.potential_mineral_nitrogen_flux_column_level_transition,
        dtype=jnp.float64,
    )
    positive_immobilization = potential_mineral_flux > jnp.float64(0.0)
    scale = active & donor_positive & positive_immobilization
    fpi = jnp.asarray(
        inputs.fraction_potential_immobilization_column_level,
        dtype=jnp.float64,
    )
    scaled_carbon_loss = jnp.where(
        scale,
        fortran_multiply(potential_carbon_loss, fpi[:, :, None]),
        potential_carbon_loss,
    )
    scaled_mineral_flux = jnp.where(
        scale,
        fortran_multiply(potential_mineral_flux, fpi[:, :, None]),
        potential_mineral_flux,
    )
    use_fates = jnp.asarray(parameters.use_fates, dtype=jnp.bool_)
    potential_carbon_loss = jnp.where(
        use_fates,
        potential_carbon_loss,
        scaled_carbon_loss,
    )
    potential_mineral_flux = jnp.where(
        use_fates,
        potential_mineral_flux,
        scaled_mineral_flux,
    )

    old_hr = get_cn_flux_field(state.cn_fluxes, _HETEROTROPHIC_RESPIRATION)
    old_carbon_transfer = get_cn_flux_field(state.cn_fluxes, _CARBON_TRANSFER)
    hr_candidate = fortran_multiply(
        respiration_fraction,
        potential_carbon_loss,
    )
    carbon_transfer_candidate = fortran_multiply(
        fortran_add(jnp.float64(1.0), -respiration_fraction),
        potential_carbon_loss,
    )
    carbon_write = active & (use_fates | donor_positive)
    heterotrophic_respiration = jnp.where(
        carbon_write,
        hr_candidate,
        old_hr,
    )
    carbon_transfer = jnp.where(
        carbon_write,
        carbon_transfer_candidate,
        old_carbon_transfer,
    )

    safe_donor_ratio = jnp.where(
        positive_donor_nitrogen,
        donor_ratio,
        jnp.float64(1.0),
    )
    nitrogen_transfer_candidate = fortran_divide(
        potential_carbon_loss,
        safe_donor_ratio,
    )
    nitrogen_transfer_candidate = jnp.where(
        positive_donor_nitrogen
        & ~receiver_is_atmosphere[None, None, :],
        nitrogen_transfer_candidate,
        jnp.float64(0.0),
    )
    mineral_flux_candidate = jnp.where(
        receiver_is_atmosphere[None, None, :],
        -potential_mineral_flux,
        potential_mineral_flux,
    )
    mineral_flux_candidate = jnp.where(
        donor_positive,
        mineral_flux_candidate,
        jnp.float64(0.0),
    )
    old_nitrogen_transfer = get_cn_flux_field(
        state.cn_fluxes,
        _NITROGEN_TRANSFER,
    )
    old_mineral_flux = get_cn_flux_field(
        state.cn_fluxes,
        _MINERAL_NITROGEN_FLUX,
    )
    non_fates_write = active & ~use_fates
    nitrogen_transfer = jnp.where(
        non_fates_write,
        jnp.where(
            donor_positive,
            nitrogen_transfer_candidate,
            jnp.float64(0.0),
        ),
        old_nitrogen_transfer,
    )
    mineral_nitrogen_flux = jnp.where(
        non_fates_write,
        mineral_flux_candidate,
        old_mineral_flux,
    )

    denitrification = jnp.asarray(
        state.denitrification_column_level_transition,
        dtype=jnp.float64,
    )
    denitrification_candidate = jnp.where(
        donor_positive & ~positive_immobilization,
        fortran_multiply(
            -jnp.asarray(
                parameters.denitrification_proportion,
                dtype=jnp.float64,
            ),
            potential_mineral_flux,
        ),
        jnp.float64(0.0),
    )
    write_denitrification = (
        non_fates_write
        & ~jnp.asarray(
            parameters.use_nitrification_denitrification,
            dtype=jnp.bool_,
        )
    )
    denitrification = jnp.where(
        write_denitrification,
        denitrification_candidate,
        denitrification,
    )

    net_mineralization_level = get_cn_flux_field(
        state.cn_fluxes,
        _NET_MINERALIZATION_LEVEL,
    )

    def subtract_transition(current, transition_index):
        write = (
            soil[:, None]
            & active_level[None, :]
            & active_transition[transition_index]
            & donor_positive[:, :, transition_index]
            & ~use_fates
        )
        candidate = fortran_add(
            current,
            -potential_mineral_flux[:, :, transition_index],
        )
        return jnp.where(write, candidate, current), None

    net_mineralization_level = ordered_indexed_fold(
        subtract_transition,
        net_mineralization_level,
        NUMBER_OF_DECOMPOSITION_TRANSITIONS,
    )

    fraction_potential_hr = jnp.asarray(
        state.fraction_potential_heterotrophic_respiration_column_level,
        dtype=jnp.float64,
    )

    def sum_respiration(current, transition_index):
        contribution = fortran_multiply(
            respiration_fraction[:, :, transition_index],
            potential_carbon_loss[:, :, transition_index],
        )
        write = (
            soil[:, None]
            & active_level[None, :]
            & active_transition[transition_index]
        )
        candidate = fortran_add(current, contribution)
        return jnp.where(write, candidate, current), None

    respiration_sum = ordered_indexed_fold(
        sum_respiration,
        jnp.zeros_like(fraction_potential_hr),
        NUMBER_OF_DECOMPOSITION_TRANSITIONS,
    )
    potential_hr = jnp.asarray(
        work.potential_heterotrophic_respiration_column_level,
        dtype=jnp.float64,
    )
    positive_potential_hr = potential_hr > jnp.float64(0.0)
    safe_potential_hr = jnp.where(
        positive_potential_hr,
        potential_hr,
        jnp.float64(1.0),
    )
    fraction_candidate = fortran_multiply(
        fortran_divide(respiration_sum, safe_potential_hr),
        jnp.asarray(inputs.water_scalar_column_level, dtype=jnp.float64),
    )
    fraction_candidate = jnp.where(
        positive_potential_hr,
        jnp.maximum(fraction_candidate, jnp.float64(0.01)),
        jnp.float64(1.0),
    )
    methane_write = (
        soil[:, None]
        & active_level[None, :]
        & jnp.asarray(parameters.use_methane, dtype=jnp.bool_)
    )
    fraction_potential_hr = jnp.where(
        methane_write,
        fraction_candidate,
        fraction_potential_hr,
    )

    gross_mineralization_level = get_cn_flux_field(
        state.cn_fluxes,
        _GROSS_MINERALIZATION_LEVEL,
    )
    gross_mineralization = get_cn_flux_field(
        state.cn_fluxes,
        _GROSS_MINERALIZATION,
    )
    net_mineralization = get_cn_flux_field(
        state.cn_fluxes,
        _NET_MINERALIZATION,
    )
    layer_thickness = jnp.asarray(
        parameters.soil_layer_thickness,
        dtype=jnp.float64,
    )

    def integrate_level(current, level_index):
        current_net, current_gross = current
        net_contribution = fortran_multiply(
            net_mineralization_level[:, level_index],
            layer_thickness[level_index],
        )
        gross_contribution = fortran_multiply(
            gross_mineralization_level[:, level_index],
            layer_thickness[level_index],
        )
        write = (
            soil
            & active_level[level_index]
            & ~use_fates
        )
        next_net = fortran_add(current_net, net_contribution)
        next_gross = fortran_add(current_gross, gross_contribution)
        return (
            jnp.where(write, next_net, current_net),
            jnp.where(write, next_gross, current_gross),
        ), None

    net_mineralization, gross_mineralization = ordered_indexed_fold(
        integrate_level,
        (net_mineralization, gross_mineralization),
        NUMBER_OF_DECOMPOSITION_LEVELS,
    )

    return SoilBiogeochemDecompCoreResult(
        heterotrophic_respiration_column_level_transition=(
            heterotrophic_respiration
        ),
        carbon_transfer_column_level_transition=carbon_transfer,
        nitrogen_transfer_column_level_transition=nitrogen_transfer,
        mineral_nitrogen_flux_column_level_transition=mineral_nitrogen_flux,
        net_mineralization_column_level=net_mineralization_level,
        gross_mineralization_column=gross_mineralization,
        net_mineralization_column=net_mineralization,
        fraction_potential_heterotrophic_respiration_column_level=(
            fraction_potential_hr
        ),
        denitrification_column_level_transition=denitrification,
        work=SoilBiogeochemPotentialWorkState(
            carbon_nitrogen_ratio_column_level_pool=carbon_nitrogen_ratio,
            potential_carbon_loss_column_level_transition=(
                potential_carbon_loss
            ),
            potential_mineral_nitrogen_flux_column_level_transition=(
                potential_mineral_flux
            ),
            potential_heterotrophic_respiration_column_level=potential_hr,
        ),
        diagnostics=SoilBiogeochemDecompDiagnostics(
            updated_column_count=jnp.sum(soil, dtype=jnp.int32),
            positive_immobilization_transition_count=jnp.sum(
                active & donor_positive & positive_immobilization & ~use_fates,
                dtype=jnp.int32,
            ),
            decomposing_transition_count=jnp.sum(
                active & (use_fates | donor_positive),
                dtype=jnp.int32,
            ),
            methane_diagnostic_cell_count=jnp.sum(
                methane_write,
                dtype=jnp.int32,
            ),
        ),
    )


def soil_biogeochem_decomp(
    inputs: SoilBiogeochemDecompInputs,
    state: SoilBiogeochemDecompState,
    work: SoilBiogeochemPotentialWorkState,
    parameters: SoilBiogeochemDecompParameters = (
        DEFAULT_SOIL_BIOGEOCHEM_DECOMP_PARAMETERS
    ),
) -> SoilBiogeochemDecompResult:

    core = soil_biogeochem_decomp_core(inputs, state, work, parameters)
    cn_fluxes = state.cn_fluxes
    for field, values in (
        (
            _HETEROTROPHIC_RESPIRATION,
            core.heterotrophic_respiration_column_level_transition,
        ),
        (_CARBON_TRANSFER, core.carbon_transfer_column_level_transition),
        (_NITROGEN_TRANSFER, core.nitrogen_transfer_column_level_transition),
        (
            _MINERAL_NITROGEN_FLUX,
            core.mineral_nitrogen_flux_column_level_transition,
        ),
        (_NET_MINERALIZATION_LEVEL, core.net_mineralization_column_level),
        (_GROSS_MINERALIZATION, core.gross_mineralization_column),
        (_NET_MINERALIZATION, core.net_mineralization_column),
    ):
        cn_fluxes = set_cn_flux_field(cn_fluxes, field, values)
    return SoilBiogeochemDecompResult(
        state=SoilBiogeochemDecompState(
            cn_fluxes=cn_fluxes,
            fraction_potential_heterotrophic_respiration_column_level=(
                core.fraction_potential_heterotrophic_respiration_column_level
            ),
            denitrification_column_level_transition=(
                core.denitrification_column_level_transition
            ),
        ),
        work=core.work,
        diagnostics=core.diagnostics,
    )


__all__ = [
    "DEFAULT_SOIL_BIOGEOCHEM_DECOMP_PARAMETERS",
    "SoilBiogeochemDecompCoreResult",
    "SoilBiogeochemDecompDiagnostics",
    "SoilBiogeochemDecompInputs",
    "SoilBiogeochemDecompParameters",
    "SoilBiogeochemDecompResult",
    "SoilBiogeochemDecompState",
    "initialize_soil_biogeochem_decomp_state",
    "soil_biogeochem_decomp",
    "soil_biogeochem_decomp_core",
]
