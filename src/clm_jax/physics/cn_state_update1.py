
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from clm_jax.physics.cn_precision_control import (
    CARBON_ONLY_FIELDS,
    DEFAULT_CN_PRECISION_CONTROL_PARAMETERS,
    NITROGEN_ONLY_FIELDS,
    PAIRED_CARBON_FIELDS,
    PAIRED_NITROGEN_FIELDS,
    CnPrecisionControlDiagnostics,
    CnPrecisionControlInputs,
    CnPrecisionControlParameters,
    CnPrecisionControlState,
    cn_precision_control,
    get_cn_precision_control_state_field,
    set_cn_precision_control_state_field,
)
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
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)


class CnStateUpdate1Inputs(NamedTuple):

    soil_patch: jnp.ndarray
    soil_column: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    harvest_date_patch: jnp.ndarray
    cascade_donor_pool: jnp.ndarray
    cascade_receiver_pool: jnp.ndarray


class CnStateUpdate1State(NamedTuple):

    vegetation: CnPrecisionControlState
    xsmrpool_loss_patch: jnp.ndarray


class CnStateUpdate1Parameters(NamedTuple):

    woody_pft: jnp.ndarray
    time_step: float | jnp.ndarray = 1800.0
    first_prognostic_crop_pft: int | jnp.ndarray = 17
    decomposition_level_count: int | jnp.ndarray = 20
    carbon_respiration_option: int | jnp.ndarray = 0
    dribble_crop_harvest_xsmrpool_to_atmosphere: bool | jnp.ndarray = False
    crop_product_decay_rate: float | jnp.ndarray = 1.44e-7


class CnStateUpdate1Diagnostics(NamedTuple):
    updated_patch_count: jnp.ndarray
    updated_column_count: jnp.ndarray
    woody_patch_count: jnp.ndarray
    prognostic_crop_patch_count: jnp.ndarray
    harvested_crop_patch_count: jnp.ndarray


class CnStateUpdate1Result(NamedTuple):
    state: CnStateUpdate1State
    cn_fluxes: CnTransientFluxState
    diagnostics: CnStateUpdate1Diagnostics


class CnStateUpdate1WithPrecisionResult(NamedTuple):

    state: CnStateUpdate1State
    cn_fluxes: CnTransientFluxState
    update_diagnostics: CnStateUpdate1Diagnostics
    precision_diagnostics: CnPrecisionControlDiagnostics


_CARBON_PREFIX = "vegetation_carbon."
_NITROGEN_PREFIX = "vegetation_nitrogen."
_SOIL_CARBON_SOURCE_SINK = "soil_carbon.decomp_cpools_sourcesink_col"
_SOIL_NITROGEN_SOURCE_SINK = "soil_nitrogen.decomp_npools_sourcesink_col"
_METABOLIC_LITTER_POOL = 0
_CELLULOSE_LITTER_POOL = 1
_LIGNIN_LITTER_POOL = 2
_COARSE_WOODY_DEBRIS_POOL = 3


def _c_state(name: str) -> str:
    return _CARBON_PREFIX + name + "_patch"


def _n_state(name: str) -> str:
    return _NITROGEN_PREFIX + name + "_patch"


def _c_flux(name: str) -> str:
    return _CARBON_PREFIX + name + "_patch"


def _n_flux(name: str) -> str:
    return _NITROGEN_PREFIX + name + "_patch"


def _amount(flux: jnp.ndarray, time_step: jnp.ndarray) -> jnp.ndarray:
    return fortran_multiply(flux, time_step)


def _add_state_amount(
    state: CnPrecisionControlState,
    field_name: str,
    amount: jnp.ndarray,
    mask: jnp.ndarray,
) -> CnPrecisionControlState:
    previous = get_cn_precision_control_state_field(state, field_name)
    candidate = fortran_add(previous, amount)
    return set_cn_precision_control_state_field(
        state,
        field_name,
        jnp.where(mask, candidate, previous),
    )


def _add_state_flux(
    state: CnPrecisionControlState,
    field_name: str,
    flux: jnp.ndarray,
    time_step: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    sign: int = 1,
) -> CnPrecisionControlState:
    amount = _amount(flux, time_step)
    if sign < 0:
        amount = -amount
    return _add_state_amount(state, field_name, amount, mask)


def _move_state_flux(
    state: CnPrecisionControlState,
    source_field: str,
    destination_field: str,
    flux: jnp.ndarray,
    time_step: jnp.ndarray,
    mask: jnp.ndarray,
) -> CnPrecisionControlState:
    amount = _amount(flux, time_step)
    state = _add_state_amount(state, source_field, -amount, mask)
    return _add_state_amount(state, destination_field, amount, mask)


def _validate_shapes(
    inputs: CnStateUpdate1Inputs,
    state: CnStateUpdate1State,
    cn_fluxes: CnTransientFluxState,
    parameters: CnStateUpdate1Parameters,
) -> tuple[int, int]:
    if jnp.ndim(inputs.soil_patch) != 1 or jnp.ndim(inputs.soil_column) != 1:
        raise ValueError("soil_patch and soil_column must be vectors")
    patch_count = inputs.soil_patch.shape[0]
    column_count = inputs.soil_column.shape[0]
    for name in ("vegetation_type_patch", "harvest_date_patch"):
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    for name in ("cascade_donor_pool", "cascade_receiver_pool"):
        if jnp.shape(getattr(inputs, name)) != (
            NUMBER_OF_DECOMPOSITION_TRANSITIONS,
        ):
            raise ValueError(f"{name} must contain every cascade transition")
    if jnp.shape(state.xsmrpool_loss_patch) != (patch_count,):
        raise ValueError("xsmrpool_loss_patch must contain every patch")
    expected_vegetation_shapes = {
        "paired_carbon_patch": (patch_count, len(PAIRED_CARBON_FIELDS)),
        "paired_nitrogen_patch": (patch_count, len(PAIRED_NITROGEN_FIELDS)),
        "carbon_only_patch": (patch_count, len(CARBON_ONLY_FIELDS)),
        "nitrogen_only_patch": (patch_count, len(NITROGEN_ONLY_FIELDS)),
        "carbon_truncation_patch": (patch_count,),
        "nitrogen_truncation_patch": (patch_count,),
    }
    for name, expected in expected_vegetation_shapes.items():
        if jnp.shape(getattr(state.vegetation, name)) != expected:
            raise ValueError(f"vegetation {name} has an incompatible shape")
    if cn_fluxes.patch_scalar.shape[0] != patch_count:
        raise ValueError("CN patch fluxes must match patch count")
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
    if jnp.ndim(parameters.woody_pft) != 1 or parameters.woody_pft.shape[0] < 1:
        raise ValueError("woody_pft must be a nonempty PFT vector")
    for name in parameters._fields[1:]:
        if jnp.shape(getattr(parameters, name)) != ():
            raise ValueError(f"{name} must be scalar")
    return patch_count, column_count


def _update_soil_carbon_sources(
    inputs: CnStateUpdate1Inputs,
    cn_fluxes: CnTransientFluxState,
    time_step: jnp.ndarray,
    decomposition_level_count: jnp.ndarray,
) -> CnTransientFluxState:
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    level_mask = jnp.arange(
        NUMBER_OF_DECOMPOSITION_LEVELS,
        dtype=jnp.int32,
    ) < jnp.asarray(decomposition_level_count, dtype=jnp.int32)
    update = soil_column[:, None] & level_mask[None, :]
    source = get_cn_flux_field(cn_fluxes, _SOIL_CARBON_SOURCE_SINK)
    if (jax.default_backend() == "gpu" and source.size
            and isinstance(inputs.cascade_donor_pool, np.ndarray)
            and isinstance(inputs.cascade_receiver_pool, np.ndarray)):
        from clm_jax.physics.cn_soil_sources_gpu import carbon_sources

        updated = carbon_sources(
            source,
            tuple(get_cn_flux_field(cn_fluxes, 'vegetation_carbon.' + name) for name in (
                'phenology_c_to_litr_met_c_col', 'phenology_c_to_litr_cel_c_col',
                'phenology_c_to_litr_lig_c_col',
            )),
            get_cn_flux_field(cn_fluxes, 'soil_carbon.decomp_cascade_ctransfer_vr_col'),
            get_cn_flux_field(cn_fluxes, 'soil_carbon.decomp_cascade_hr_vr_col'),
            update, time_step, inputs.cascade_donor_pool, inputs.cascade_receiver_pool,
        )
        return set_cn_flux_field(cn_fluxes, _SOIL_CARBON_SOURCE_SINK, updated)
    source_fields = [source[:, :, index] for index in range(source.shape[-1])]

    for pool, field_name in (
        (
            _METABOLIC_LITTER_POOL,
            "vegetation_carbon.phenology_c_to_litr_met_c_col",
        ),
        (
            _CELLULOSE_LITTER_POOL,
            "vegetation_carbon.phenology_c_to_litr_cel_c_col",
        ),
        (
            _LIGNIN_LITTER_POOL,
            "vegetation_carbon.phenology_c_to_litr_lig_c_col",
        ),
    ):
        previous = source_fields[pool]
        candidate = _amount(get_cn_flux_field(cn_fluxes, field_name), time_step)
        source_fields[pool] = jnp.where(update, candidate, previous)
    previous_cwd = source_fields[_COARSE_WOODY_DEBRIS_POOL]
    source_fields[_COARSE_WOODY_DEBRIS_POOL] = jnp.where(
        update, jnp.float64(0.0), previous_cwd
    )

    transfer = get_cn_flux_field(
        cn_fluxes,
        "soil_carbon.decomp_cascade_ctransfer_vr_col",
    )
    respiration = get_cn_flux_field(
        cn_fluxes,
        "soil_carbon.decomp_cascade_hr_vr_col",
    )

    if isinstance(inputs.cascade_donor_pool, np.ndarray) and isinstance(
        inputs.cascade_receiver_pool, np.ndarray
    ):
        for transition, donor in enumerate(inputs.cascade_donor_pool):
            previous = source_fields[donor]
            loss = fortran_add(respiration[:, :, transition], transfer[:, :, transition])
            candidate = fortran_add(previous, -_amount(loss, time_step))
            source_fields[donor] = jnp.where(update, candidate, previous)
        for transition, receiver in enumerate(inputs.cascade_receiver_pool):
            if receiver < 0:
                continue
            previous = source_fields[receiver]
            candidate = fortran_add(previous, _amount(transfer[:, :, transition], time_step))
            source_fields[receiver] = jnp.where(update, candidate, previous)
        return set_cn_flux_field(
            cn_fluxes, _SOIL_CARBON_SOURCE_SINK, jnp.stack(source_fields, axis=-1)
        )

    source = jnp.stack(source_fields, axis=-1)
    donor_pool = jnp.asarray(inputs.cascade_donor_pool, dtype=jnp.int32)
    receiver_pool = jnp.asarray(inputs.cascade_receiver_pool, dtype=jnp.int32)

    for transition in range(NUMBER_OF_DECOMPOSITION_TRANSITIONS):
        donor = donor_pool[transition]
        previous = source[:, :, donor]
        loss = fortran_add(
            respiration[:, :, transition],
            transfer[:, :, transition],
        )
        candidate = fortran_add(previous, -_amount(loss, time_step))
        source = source.at[:, :, donor].set(
            jnp.where(update, candidate, previous)
        )
    for transition in range(NUMBER_OF_DECOMPOSITION_TRANSITIONS):
        receiver = receiver_pool[transition]
        bounded_receiver = jnp.maximum(receiver, jnp.int32(0))
        previous = source[:, :, bounded_receiver]
        candidate = fortran_add(
            previous,
            _amount(transfer[:, :, transition], time_step),
        )
        receiver_update = update & (receiver >= jnp.int32(0))
        source = source.at[:, :, bounded_receiver].set(
            jnp.where(receiver_update, candidate, previous)
        )

    return set_cn_flux_field(cn_fluxes, _SOIL_CARBON_SOURCE_SINK, source)


def _update_soil_nitrogen_sources(
    inputs: CnStateUpdate1Inputs,
    cn_fluxes: CnTransientFluxState,
    time_step: jnp.ndarray,
    decomposition_level_count: jnp.ndarray,
) -> CnTransientFluxState:
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    level_mask = jnp.arange(
        NUMBER_OF_DECOMPOSITION_LEVELS,
        dtype=jnp.int32,
    ) < jnp.asarray(decomposition_level_count, dtype=jnp.int32)
    update = soil_column[:, None] & level_mask[None, :]
    source = get_cn_flux_field(cn_fluxes, _SOIL_NITROGEN_SOURCE_SINK)
    source_fields = [source[:, :, index] for index in range(source.shape[-1])]
    for pool, field_name in (
        (
            _METABOLIC_LITTER_POOL,
            "vegetation_nitrogen.phenology_n_to_litr_met_n_col",
        ),
        (
            _CELLULOSE_LITTER_POOL,
            "vegetation_nitrogen.phenology_n_to_litr_cel_n_col",
        ),
        (
            _LIGNIN_LITTER_POOL,
            "vegetation_nitrogen.phenology_n_to_litr_lig_n_col",
        ),
    ):
        previous = source_fields[pool]
        candidate = _amount(get_cn_flux_field(cn_fluxes, field_name), time_step)
        source_fields[pool] = jnp.where(update, candidate, previous)
    previous_cwd = source_fields[_COARSE_WOODY_DEBRIS_POOL]
    source_fields[_COARSE_WOODY_DEBRIS_POOL] = jnp.where(
        update, jnp.float64(0.0), previous_cwd
    )
    source = jnp.stack(source_fields, axis=-1)
    return set_cn_flux_field(cn_fluxes, _SOIL_NITROGEN_SOURCE_SINK, source)


def _update_carbon_state_reference(
    inputs: CnStateUpdate1Inputs,
    state: CnStateUpdate1State,
    cn_fluxes: CnTransientFluxState,
    parameters: CnStateUpdate1Parameters,
    soil_patch: jnp.ndarray,
    woody_patch: jnp.ndarray,
    crop_patch: jnp.ndarray,
) -> tuple[CnStateUpdate1State, CnTransientFluxState]:
    vegetation = state.vegetation
    time_step = jnp.asarray(parameters.time_step, dtype=jnp.float64)

    def flux(name: str) -> jnp.ndarray:
        return get_cn_flux_field(cn_fluxes, _c_flux(name))

    for organ in ("leaf", "froot"):
        vegetation = _move_state_flux(
            vegetation,
            _c_state(organ + "c_xfer"),
            _c_state(organ + "c"),
            flux(organ + "c_xfer_to_" + organ + "c"),
            time_step,
            soil_patch,
        )
    for organ in ("livestem", "deadstem", "livecroot", "deadcroot"):
        vegetation = _move_state_flux(
            vegetation,
            _c_state(organ + "c_xfer"),
            _c_state(organ + "c"),
            flux(organ + "c_xfer_to_" + organ + "c"),
            time_step,
            woody_patch,
        )
    for organ in ("livestem", "grain"):
        vegetation = _move_state_flux(
            vegetation,
            _c_state(organ + "c_xfer"),
            _c_state(organ + "c"),
            flux(organ + "c_xfer_to_" + organ + "c"),
            time_step,
            crop_patch,
        )

    vegetation = _add_state_flux(
        vegetation,
        _c_state("leafc"),
        flux("leafc_to_litter"),
        time_step,
        soil_patch,
        sign=-1,
    )
    vegetation = _add_state_flux(
        vegetation,
        _c_state("frootc"),
        flux("frootc_to_litter"),
        time_step,
        soil_patch,
        sign=-1,
    )
    for live, dead in (
        ("livestemc", "deadstemc"),
        ("livecrootc", "deadcrootc"),
    ):
        vegetation = _move_state_flux(
            vegetation,
            _c_state(live),
            _c_state(dead),
            flux(live + "_to_" + dead),
            time_step,
            woody_patch,
        )
    vegetation = _add_state_flux(
        vegetation,
        _c_state("livestemc"),
        flux("livestemc_to_litter"),
        time_step,
        crop_patch,
        sign=-1,
    )
    grain_removal = fortran_add(
        flux("grainc_to_food"),
        flux("grainc_to_seed"),
    )
    vegetation = _add_state_flux(
        vegetation,
        _c_state("grainc"),
        grain_removal,
        time_step,
        crop_patch,
        sign=-1,
    )
    vegetation = _add_state_flux(
        vegetation,
        _c_state("cropseedc_deficit"),
        flux("crop_seedc_to_leaf"),
        time_step,
        crop_patch,
        sign=-1,
    )
    vegetation = _add_state_flux(
        vegetation,
        _c_state("cropseedc_deficit"),
        flux("grainc_to_seed"),
        time_step,
        crop_patch,
    )

    cpool = _c_state("cpool")
    xsmrpool = _c_state("xsmrpool")
    for name, mask in (
        ("cpool_to_xsmrpool", soil_patch),
        ("leaf_curmr", soil_patch),
        ("froot_curmr", soil_patch),
        ("livestem_curmr", woody_patch),
        ("livecroot_curmr", woody_patch),
        ("livestem_curmr", crop_patch),
        ("grain_curmr", crop_patch),
        ("cpool_to_resp", soil_patch),
        ("soilc_change", soil_patch),
    ):
        vegetation = _add_state_flux(
            vegetation,
            cpool,
            flux(name),
            time_step,
            mask,
            sign=-1,
        )
    vegetation = _add_state_flux(
        vegetation,
        xsmrpool,
        flux("cpool_to_xsmrpool"),
        time_step,
        soil_patch,
    )
    for name, mask in (
        ("leaf_xsmr", soil_patch),
        ("froot_xsmr", soil_patch),
        ("livestem_xsmr", woody_patch),
        ("livecroot_xsmr", woody_patch),
    ):
        vegetation = _add_state_flux(
            vegetation,
            xsmrpool,
            flux(name),
            time_step,
            mask,
            sign=-1,
        )

    respiration_option_one = (
        jnp.asarray(parameters.carbon_respiration_option, dtype=jnp.int32)
        == jnp.int32(1)
    )
    allocation_respiration_pairs = (
        ("cpool_to_leafc", "cpool_to_leafc_resp", soil_patch),
        (
            "cpool_to_leafc_storage",
            "cpool_to_leafc_storage_resp",
            soil_patch,
        ),
        ("cpool_to_frootc", "cpool_to_frootc_resp", soil_patch),
        (
            "cpool_to_frootc_storage",
            "cpool_to_frootc_storage_resp",
            soil_patch,
        ),
        (
            "cpool_to_livecrootc",
            "cpool_to_livecrootc_resp",
            woody_patch,
        ),
        (
            "cpool_to_livecrootc_storage",
            "cpool_to_livecrootc_storage_resp",
            woody_patch,
        ),
        ("cpool_to_livestemc", "cpool_to_livestemc_resp", woody_patch),
        (
            "cpool_to_livestemc_storage",
            "cpool_to_livestemc_storage_resp",
            woody_patch,
        ),
        ("cpool_to_livestemc", "cpool_to_livestemc_resp", crop_patch),
        (
            "cpool_to_livestemc_storage",
            "cpool_to_livestemc_storage_resp",
            crop_patch,
        ),
    )
    for allocation_name, respiration_name, branch_mask in allocation_respiration_pairs:
        allocation_field = _c_flux(allocation_name)
        previous = get_cn_flux_field(cn_fluxes, allocation_field)
        candidate = fortran_add(previous, -flux(respiration_name))
        cn_fluxes = set_cn_flux_field(
            cn_fluxes,
            allocation_field,
            jnp.where(respiration_option_one & branch_mask, candidate, previous),
        )

    def current_flux(name: str) -> jnp.ndarray:
        return get_cn_flux_field(cn_fluxes, _c_flux(name))

    for organ in ("leaf", "froot"):
        for suffix in ("", "_storage"):
            name = "cpool_to_" + organ + "c" + suffix
            vegetation = _move_state_flux(
                vegetation,
                cpool,
                _c_state(organ + "c" + suffix),
                current_flux(name),
                time_step,
                soil_patch,
            )
    for organ in ("livestem", "deadstem", "livecroot", "deadcroot"):
        for suffix in ("", "_storage"):
            name = "cpool_to_" + organ + "c" + suffix
            vegetation = _move_state_flux(
                vegetation,
                cpool,
                _c_state(organ + "c" + suffix),
                current_flux(name),
                time_step,
                woody_patch,
            )
    for organ in ("livestem", "grain"):
        for suffix in ("", "_storage"):
            name = "cpool_to_" + organ + "c" + suffix
            vegetation = _move_state_flux(
                vegetation,
                cpool,
                _c_state(organ + "c" + suffix),
                current_flux(name),
                time_step,
                crop_patch,
            )

    for name, mask in (
        ("cpool_leaf_gr", soil_patch),
        ("cpool_froot_gr", soil_patch),
        ("cpool_livestem_gr", woody_patch),
        ("cpool_deadstem_gr", woody_patch),
        ("cpool_livecroot_gr", woody_patch),
        ("cpool_deadcroot_gr", woody_patch),
        ("cpool_livestem_gr", crop_patch),
        ("cpool_grain_gr", crop_patch),
    ):
        vegetation = _add_state_flux(
            vegetation,
            cpool,
            current_flux(name),
            time_step,
            mask,
            sign=-1,
        )
    gresp_xfer = _c_state("gresp_xfer")
    for name, mask in (
        ("transfer_leaf_gr", soil_patch),
        ("transfer_froot_gr", soil_patch),
        ("transfer_livestem_gr", woody_patch),
        ("transfer_deadstem_gr", woody_patch),
        ("transfer_livecroot_gr", woody_patch),
        ("transfer_deadcroot_gr", woody_patch),
        ("transfer_livestem_gr", crop_patch),
        ("transfer_grain_gr", crop_patch),
    ):
        vegetation = _add_state_flux(
            vegetation,
            gresp_xfer,
            current_flux(name),
            time_step,
            mask,
            sign=-1,
        )
    for name, mask in (
        ("cpool_leaf_storage_gr", soil_patch),
        ("cpool_froot_storage_gr", soil_patch),
        ("cpool_livestem_storage_gr", woody_patch),
        ("cpool_deadstem_storage_gr", woody_patch),
        ("cpool_livecroot_storage_gr", woody_patch),
        ("cpool_deadcroot_storage_gr", woody_patch),
        ("cpool_livestem_storage_gr", crop_patch),
        ("cpool_grain_storage_gr", crop_patch),
    ):
        vegetation = _add_state_flux(
            vegetation,
            cpool,
            current_flux(name),
            time_step,
            mask,
            sign=-1,
        )
    vegetation = _move_state_flux(
        vegetation,
        cpool,
        _c_state("gresp_storage"),
        current_flux("cpool_to_gresp_storage"),
        time_step,
        soil_patch,
    )

    for organ in ("leaf", "froot"):
        vegetation = _move_state_flux(
            vegetation,
            _c_state(organ + "c_storage"),
            _c_state(organ + "c_xfer"),
            current_flux(organ + "c_storage_to_xfer"),
            time_step,
            soil_patch,
        )
    for organ in ("livestem", "deadstem", "livecroot", "deadcroot"):
        vegetation = _move_state_flux(
            vegetation,
            _c_state(organ + "c_storage"),
            _c_state(organ + "c_xfer"),
            current_flux(organ + "c_storage_to_xfer"),
            time_step,
            woody_patch,
        )
    vegetation = _move_state_flux(
        vegetation,
        _c_state("gresp_storage"),
        gresp_xfer,
        current_flux("gresp_storage_to_xfer"),
        time_step,
        woody_patch,
    )
    for organ in ("livestem", "grain"):
        vegetation = _move_state_flux(
            vegetation,
            _c_state(organ + "c_storage"),
            _c_state(organ + "c_xfer"),
            current_flux(organ + "c_storage_to_xfer"),
            time_step,
            crop_patch,
        )

    vegetation = _add_state_flux(
        vegetation,
        xsmrpool,
        current_flux("livestem_xsmr"),
        time_step,
        crop_patch,
        sign=-1,
    )
    vegetation = _add_state_flux(
        vegetation,
        xsmrpool,
        current_flux("grain_xsmr"),
        time_step,
        crop_patch,
        sign=-1,
    )
    harvested = crop_patch & (
        jnp.asarray(inputs.harvest_date_patch, dtype=jnp.int32) < jnp.int32(999)
    )
    dribble = jnp.asarray(
        parameters.dribble_crop_harvest_xsmrpool_to_atmosphere,
        dtype=jnp.bool_,
    )
    instant_harvest = harvested & ~dribble
    dribble_harvest = harvested & dribble
    atmosphere_field = _c_flux("xsmrpool_to_atm")
    atmosphere_flux = get_cn_flux_field(cn_fluxes, atmosphere_field)
    for pool_field in (xsmrpool, cpool, _c_state("frootc")):
        pool = get_cn_precision_control_state_field(vegetation, pool_field)
        candidate = fortran_add(
            atmosphere_flux,
            fortran_divide(pool, time_step),
        )
        atmosphere_flux = jnp.where(
            instant_harvest,
            candidate,
            atmosphere_flux,
        )

    xsmrpool_loss = jnp.asarray(state.xsmrpool_loss_patch, dtype=jnp.float64)
    for pool_field in (xsmrpool, cpool, _c_state("frootc")):
        pool = get_cn_precision_control_state_field(vegetation, pool_field)
        candidate = fortran_add(xsmrpool_loss, pool)
        xsmrpool_loss = jnp.where(
            dribble_harvest,
            candidate,
            xsmrpool_loss,
        )
    for pool_field in (xsmrpool, cpool, _c_state("frootc")):
        previous = get_cn_precision_control_state_field(vegetation, pool_field)
        vegetation = set_cn_precision_control_state_field(
            vegetation,
            pool_field,
            jnp.where(harvested, jnp.float64(0.0), previous),
        )

    dribble_crop = crop_patch & dribble
    dribble_candidate = fortran_multiply(
        xsmrpool_loss,
        jnp.asarray(parameters.crop_product_decay_rate, dtype=jnp.float64),
    )
    atmosphere_flux = jnp.where(
        dribble_crop,
        dribble_candidate,
        atmosphere_flux,
    )
    loss_candidate = fortran_add(
        xsmrpool_loss,
        -_amount(atmosphere_flux, time_step),
    )
    xsmrpool_loss = jnp.where(
        dribble_crop,
        loss_candidate,
        xsmrpool_loss,
    )
    cn_fluxes = set_cn_flux_field(
        cn_fluxes,
        atmosphere_field,
        atmosphere_flux,
    )
    return (
        CnStateUpdate1State(
            vegetation=vegetation,
            xsmrpool_loss_patch=xsmrpool_loss,
        ),
        cn_fluxes,
    )


def _update_nitrogen_state_reference(
    state: CnStateUpdate1State,
    cn_fluxes: CnTransientFluxState,
    time_step: jnp.ndarray,
    soil_patch: jnp.ndarray,
    woody_patch: jnp.ndarray,
    crop_patch: jnp.ndarray,
) -> CnStateUpdate1State:
    vegetation = state.vegetation

    def flux(name: str) -> jnp.ndarray:
        return get_cn_flux_field(cn_fluxes, _n_flux(name))

    for organ in ("leaf", "froot"):
        vegetation = _move_state_flux(
            vegetation,
            _n_state(organ + "n_xfer"),
            _n_state(organ + "n"),
            flux(organ + "n_xfer_to_" + organ + "n"),
            time_step,
            soil_patch,
        )
    for organ in ("livestem", "deadstem", "livecroot", "deadcroot"):
        vegetation = _move_state_flux(
            vegetation,
            _n_state(organ + "n_xfer"),
            _n_state(organ + "n"),
            flux(organ + "n_xfer_to_" + organ + "n"),
            time_step,
            woody_patch,
        )
    for organ in ("livestem", "grain"):
        vegetation = _move_state_flux(
            vegetation,
            _n_state(organ + "n_xfer"),
            _n_state(organ + "n"),
            flux(organ + "n_xfer_to_" + organ + "n"),
            time_step,
            crop_patch,
        )

    for organ in ("leaf", "froot"):
        vegetation = _add_state_flux(
            vegetation,
            _n_state(organ + "n"),
            flux(organ + "n_to_litter"),
            time_step,
            soil_patch,
            sign=-1,
        )
    vegetation = _move_state_flux(
        vegetation,
        _n_state("leafn"),
        _n_state("retransn"),
        flux("leafn_to_retransn"),
        time_step,
        soil_patch,
    )
    for live, dead in (
        ("livestemn", "deadstemn"),
        ("livecrootn", "deadcrootn"),
    ):
        vegetation = _move_state_flux(
            vegetation,
            _n_state(live),
            _n_state(dead),
            flux(live + "_to_" + dead),
            time_step,
            woody_patch,
        )
        vegetation = _move_state_flux(
            vegetation,
            _n_state(live),
            _n_state("retransn"),
            flux(live + "_to_retransn"),
            time_step,
            woody_patch,
        )
    vegetation = _move_state_flux(
        vegetation,
        _n_state("frootn"),
        _n_state("retransn"),
        flux("frootn_to_retransn"),
        time_step,
        crop_patch,
    )
    vegetation = _add_state_flux(
        vegetation,
        _n_state("livestemn"),
        flux("livestemn_to_litter"),
        time_step,
        crop_patch,
        sign=-1,
    )
    vegetation = _move_state_flux(
        vegetation,
        _n_state("livestemn"),
        _n_state("retransn"),
        flux("livestemn_to_retransn"),
        time_step,
        crop_patch,
    )
    grain_removal = fortran_add(
        flux("grainn_to_food"),
        flux("grainn_to_seed"),
    )
    vegetation = _add_state_flux(
        vegetation,
        _n_state("grainn"),
        grain_removal,
        time_step,
        crop_patch,
        sign=-1,
    )
    vegetation = _add_state_flux(
        vegetation,
        _n_state("cropseedn_deficit"),
        flux("crop_seedn_to_leaf"),
        time_step,
        crop_patch,
        sign=-1,
    )
    vegetation = _add_state_flux(
        vegetation,
        _n_state("cropseedn_deficit"),
        flux("grainn_to_seed"),
        time_step,
        crop_patch,
    )

    npool = _n_state("npool")
    retransn = _n_state("retransn")
    vegetation = _add_state_flux(
        vegetation,
        npool,
        flux("sminn_to_npool"),
        time_step,
        soil_patch,
    )
    for name in ("retransn_to_npool", "free_retransn_to_npool"):
        vegetation = _move_state_flux(
            vegetation,
            retransn,
            npool,
            flux(name),
            time_step,
            soil_patch,
        )

    for organ in ("leaf", "froot"):
        for suffix in ("", "_storage"):
            vegetation = _move_state_flux(
                vegetation,
                npool,
                _n_state(organ + "n" + suffix),
                flux("npool_to_" + organ + "n" + suffix),
                time_step,
                soil_patch,
            )
    for organ in ("livestem", "deadstem", "livecroot", "deadcroot"):
        for suffix in ("", "_storage"):
            vegetation = _move_state_flux(
                vegetation,
                npool,
                _n_state(organ + "n" + suffix),
                flux("npool_to_" + organ + "n" + suffix),
                time_step,
                woody_patch,
            )
    for organ in ("livestem", "grain"):
        for suffix in ("", "_storage"):
            vegetation = _move_state_flux(
                vegetation,
                npool,
                _n_state(organ + "n" + suffix),
                flux("npool_to_" + organ + "n" + suffix),
                time_step,
                crop_patch,
            )

    for organ in ("leaf", "froot"):
        vegetation = _move_state_flux(
            vegetation,
            _n_state(organ + "n_storage"),
            _n_state(organ + "n_xfer"),
            flux(organ + "n_storage_to_xfer"),
            time_step,
            soil_patch,
        )
    for organ in ("livestem", "deadstem", "livecroot", "deadcroot"):
        vegetation = _move_state_flux(
            vegetation,
            _n_state(organ + "n_storage"),
            _n_state(organ + "n_xfer"),
            flux(organ + "n_storage_to_xfer"),
            time_step,
            woody_patch,
        )
    for organ in ("livestem", "grain"):
        vegetation = _move_state_flux(
            vegetation,
            _n_state(organ + "n_storage"),
            _n_state(organ + "n_xfer"),
            flux(organ + "n_storage_to_xfer"),
            time_step,
            crop_patch,
        )
    return CnStateUpdate1State(
        vegetation=vegetation,
        xsmrpool_loss_patch=state.xsmrpool_loss_patch,
    )


def _update_carbon_state(inputs, state, cn_fluxes, parameters, soil, woody, crop):
    if jax.default_backend() == "gpu" and soil.size:
        from clm_jax.physics.cn_patch_state_gpu import carbon

        return carbon(inputs, state, cn_fluxes, parameters, soil, woody, crop)
    return _update_carbon_state_reference(
        inputs, state, cn_fluxes, parameters, soil, woody, crop,
    )


def _update_nitrogen_state(state, cn_fluxes, time_step, soil, woody, crop):
    if jax.default_backend() == "gpu" and soil.size:
        from clm_jax.physics.cn_patch_state_gpu import nitrogen

        return nitrogen(state, cn_fluxes, time_step, soil, woody, crop)
    return _update_nitrogen_state_reference(state, cn_fluxes, time_step, soil, woody, crop)


def cn_state_update1(
    inputs: CnStateUpdate1Inputs,
    state: CnStateUpdate1State,
    cn_fluxes: CnTransientFluxState,
    parameters: CnStateUpdate1Parameters,
) -> CnStateUpdate1Result:

    _validate_shapes(inputs, state, cn_fluxes, parameters)
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    vegetation_type = jnp.asarray(
        inputs.vegetation_type_patch,
        dtype=jnp.int32,
    )
    woody_pft = jnp.asarray(parameters.woody_pft, dtype=jnp.float64)
    woody_patch = soil_patch & (
        woody_pft[vegetation_type] == jnp.float64(1.0)
    )
    crop_patch = soil_patch & (
        vegetation_type
        >= jnp.asarray(parameters.first_prognostic_crop_pft, dtype=jnp.int32)
    )
    time_step = jnp.asarray(parameters.time_step, dtype=jnp.float64)

    cn_fluxes = _update_soil_carbon_sources(
        inputs,
        cn_fluxes,
        time_step,
        jnp.asarray(parameters.decomposition_level_count, dtype=jnp.int32),
    )
    state, cn_fluxes = _update_carbon_state(
        inputs,
        state,
        cn_fluxes,
        parameters,
        soil_patch,
        woody_patch,
        crop_patch,
    )
    cn_fluxes = _update_soil_nitrogen_sources(
        inputs,
        cn_fluxes,
        time_step,
        jnp.asarray(parameters.decomposition_level_count, dtype=jnp.int32),
    )
    state = _update_nitrogen_state(
        state,
        cn_fluxes,
        time_step,
        soil_patch,
        woody_patch,
        crop_patch,
    )
    harvested = crop_patch & (
        jnp.asarray(inputs.harvest_date_patch, dtype=jnp.int32) < jnp.int32(999)
    )
    return CnStateUpdate1Result(
        state=state,
        cn_fluxes=cn_fluxes,
        diagnostics=CnStateUpdate1Diagnostics(
            updated_patch_count=jnp.sum(soil_patch, dtype=jnp.int32),
            updated_column_count=jnp.sum(soil_column, dtype=jnp.int32),
            woody_patch_count=jnp.sum(woody_patch, dtype=jnp.int32),
            prognostic_crop_patch_count=jnp.sum(crop_patch, dtype=jnp.int32),
            harvested_crop_patch_count=jnp.sum(harvested, dtype=jnp.int32),
        ),
    )


def cn_state_update1_with_precision_control(
    inputs: CnStateUpdate1Inputs,
    state: CnStateUpdate1State,
    cn_fluxes: CnTransientFluxState,
    parameters: CnStateUpdate1Parameters,
    precision_parameters: CnPrecisionControlParameters = (
        DEFAULT_CN_PRECISION_CONTROL_PARAMETERS
    ),
) -> CnStateUpdate1WithPrecisionResult:

    update = cn_state_update1(inputs, state, cn_fluxes, parameters)
    precision = cn_precision_control(
        CnPrecisionControlInputs(
            soil_patch=inputs.soil_patch,
            vegetation_type_patch=inputs.vegetation_type_patch,
        ),
        update.state.vegetation,
        precision_parameters,
    )
    return CnStateUpdate1WithPrecisionResult(
        state=update.state._replace(vegetation=precision.state),
        cn_fluxes=update.cn_fluxes,
        update_diagnostics=update.diagnostics,
        precision_diagnostics=precision.diagnostics,
    )


__all__ = [
    "CnStateUpdate1Diagnostics",
    "CnStateUpdate1Inputs",
    "CnStateUpdate1Parameters",
    "CnStateUpdate1Result",
    "CnStateUpdate1State",
    "CnStateUpdate1WithPrecisionResult",
    "cn_state_update1",
    "cn_state_update1_with_precision_control",
]
