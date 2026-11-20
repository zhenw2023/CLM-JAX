
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import default_backend, lax

from clm_jax.constants import FREEZING_TEMPERATURE_FRESHWATER
from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    get_cn_flux_field,
    set_cn_flux_field,
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
from clm_jax.physics.numerics import (
    fortran_hardware_power as fortran_power,
)
from clm_jax.physics.native_power import native_power

NUMBER_OF_GROUND_LEVELS = 25
FIRST_PROGNOSTIC_CROP_PFT = 17
LEAF_CO2_TO_CARBON = 12.011e-6

_LEAF_FIELD = "vegetation_carbon.leaf_mr_patch"
_FINE_ROOT_FIELD = "vegetation_carbon.froot_mr_patch"
_LIVE_STEM_FIELD = "vegetation_carbon.livestem_mr_patch"
_LIVE_COARSE_ROOT_FIELD = "vegetation_carbon.livecroot_mr_patch"
_GRAIN_FIELD = "vegetation_carbon.grain_mr_patch"


class MaintenanceRespirationInputs(NamedTuple):

    soil_column: jnp.ndarray
    soil_patch: jnp.ndarray
    patch_column_index: jnp.ndarray
    vegetation_type_patch: jnp.ndarray
    vegetation_exposed_patch: jnp.ndarray
    sunlit_leaf_area_patch: jnp.ndarray
    shaded_leaf_area_patch: jnp.ndarray
    carbon_root_fraction_patch_level: jnp.ndarray
    soil_temperature_column_level: jnp.ndarray
    reference_temperature_2m_patch: jnp.ndarray
    ten_day_temperature_patch: jnp.ndarray
    sunlit_leaf_respiration_patch: jnp.ndarray
    shaded_leaf_respiration_patch: jnp.ndarray
    fine_root_nitrogen_patch: jnp.ndarray
    live_stem_nitrogen_patch: jnp.ndarray
    live_coarse_root_nitrogen_patch: jnp.ndarray
    grain_nitrogen_patch: jnp.ndarray


class MaintenanceRespirationParameters(NamedTuple):

    base_rate: float | jnp.ndarray
    root_base_rate: float | jnp.ndarray
    temperature_q10: float | jnp.ndarray
    woody_pft: jnp.ndarray
    first_prognostic_crop_pft: int | jnp.ndarray = FIRST_PROGNOSTIC_CROP_PFT
    root_stem_acclimation: bool | jnp.ndarray = False


_REFERENCE_WOODY_PFT = np.zeros(79, dtype=np.float64)
_REFERENCE_WOODY_PFT[1:12] = np.float64(1.0)
DEFAULT_MAINTENANCE_RESPIRATION_PARAMETERS = MaintenanceRespirationParameters(
    base_rate=np.float64(2.525e-6),
    root_base_rate=np.float64(0.83e-6),
    temperature_q10=np.float64(1.5),
    woody_pft=_REFERENCE_WOODY_PFT,
)


class MaintenanceRespirationDiagnostics(NamedTuple):
    soil_temperature_factor_column_level: jnp.ndarray
    air_temperature_factor_patch: jnp.ndarray
    final_base_rate: jnp.ndarray
    final_root_base_rate: jnp.ndarray
    updated_patch_count: jnp.ndarray


class MaintenanceRespirationResult(NamedTuple):
    state: CnTransientFluxState
    diagnostics: MaintenanceRespirationDiagnostics


def _validate_shapes(
    inputs: MaintenanceRespirationInputs,
    state: CnTransientFluxState,
    woody_pft: jnp.ndarray,
) -> tuple[int, int]:
    patch_fields = {
        "soil_patch": inputs.soil_patch,
        "patch_column_index": inputs.patch_column_index,
        "vegetation_type_patch": inputs.vegetation_type_patch,
        "vegetation_exposed_patch": inputs.vegetation_exposed_patch,
        "sunlit_leaf_area_patch": inputs.sunlit_leaf_area_patch,
        "shaded_leaf_area_patch": inputs.shaded_leaf_area_patch,
        "reference_temperature_2m_patch": inputs.reference_temperature_2m_patch,
        "ten_day_temperature_patch": inputs.ten_day_temperature_patch,
        "sunlit_leaf_respiration_patch": inputs.sunlit_leaf_respiration_patch,
        "shaded_leaf_respiration_patch": inputs.shaded_leaf_respiration_patch,
        "fine_root_nitrogen_patch": inputs.fine_root_nitrogen_patch,
        "live_stem_nitrogen_patch": inputs.live_stem_nitrogen_patch,
        "live_coarse_root_nitrogen_patch": inputs.live_coarse_root_nitrogen_patch,
        "grain_nitrogen_patch": inputs.grain_nitrogen_patch,
    }
    if jnp.ndim(inputs.soil_column) != 1:
        raise ValueError("soil_column must be a vector")
    number_of_columns = inputs.soil_column.shape[0]
    number_of_patches = inputs.soil_patch.shape[0] if jnp.ndim(inputs.soil_patch) == 1 else -1
    for name, values in patch_fields.items():
        if jnp.shape(values) != (number_of_patches,):
            raise ValueError(f"{name} must contain one value per patch")
    if jnp.shape(inputs.carbon_root_fraction_patch_level) != (
        number_of_patches,
        NUMBER_OF_GROUND_LEVELS,
    ):
        raise ValueError("carbon_root_fraction_patch_level must have shape (patch, 25)")
    if jnp.shape(inputs.soil_temperature_column_level) != (
        number_of_columns,
        NUMBER_OF_GROUND_LEVELS,
    ):
        raise ValueError("soil_temperature_column_level must have shape (column, 25)")
    if woody_pft.ndim != 1 or woody_pft.shape[0] < 1:
        raise ValueError("woody_pft must be a nonempty PFT vector")
    for field in (
        _LEAF_FIELD,
        _FINE_ROOT_FIELD,
        _LIVE_STEM_FIELD,
        _LIVE_COARSE_ROOT_FIELD,
        _GRAIN_FIELD,
    ):
        if get_cn_flux_field(state, field).shape != (number_of_patches,):
            raise ValueError("maintenance-respiration flux state must match patch count")
    return number_of_columns, number_of_patches


def _temperature_factor(
    temperature: jnp.ndarray,
    q10: jnp.ndarray,
) -> jnp.ndarray:
    temperature_celsius = fortran_add(
        temperature,
        -jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
    )
    offset = fortran_add(temperature_celsius, -jnp.float64(20.0))
    exponent = fortran_divide(offset, jnp.float64(10.0))
    power = native_power if default_backend() == "gpu" else fortran_power
    return power(q10, exponent)


def _acclimation_factor(ten_day_temperature: jnp.ndarray) -> jnp.ndarray:
    temperature_celsius = fortran_add(
        ten_day_temperature,
        -jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
    )
    offset = fortran_add(temperature_celsius, -jnp.float64(25.0))
    exponent = fortran_multiply(jnp.float64(-0.00794), offset)
    power = native_power if default_backend() == "gpu" else fortran_power
    return power(jnp.float64(10.0), exponent)


def _running_rates(
    initial_rate: jnp.ndarray,
    factors: jnp.ndarray,
    selected: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    def update(rate: jnp.ndarray, values: tuple[jnp.ndarray, jnp.ndarray]):
        factor, active = values
        candidate = fortran_multiply(rate, factor)
        updated = jnp.where(active, candidate, rate)
        return updated, updated

    return lax.scan(update, initial_rate, (factors, selected))


def _fine_root_respiration(
    previous: jnp.ndarray,
    inputs: MaintenanceRespirationInputs,
    soil_temperature_factor: jnp.ndarray,
    root_rate_by_level_patch: jnp.ndarray,
) -> jnp.ndarray:
    patch_column = jnp.asarray(inputs.patch_column_index, dtype=jnp.int32)
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    fine_root_nitrogen = jnp.asarray(inputs.fine_root_nitrogen_patch, dtype=jnp.float64)
    root_fraction = jnp.asarray(inputs.carbon_root_fraction_patch_level, dtype=jnp.float64)
    selected_temperature_factor = soil_temperature_factor[patch_column]

    first = fortran_multiply(
        fine_root_nitrogen[None, :],
        root_rate_by_level_patch,
    )
    second = fortran_multiply(
        first,
        jnp.swapaxes(selected_temperature_factor, 0, 1),
    )
    contribution = fortran_multiply(second, jnp.swapaxes(root_fraction, 0, 1))

    def add_level(running: jnp.ndarray, value: jnp.ndarray):
        candidate = fortran_add(running, value)
        updated = jnp.where(soil_patch, candidate, running)
        return updated, None

    final, _ = lax.scan(add_level, previous, contribution)
    return final


def maintenance_respiration(
    inputs: MaintenanceRespirationInputs,
    state: CnTransientFluxState,
    parameters: MaintenanceRespirationParameters = (DEFAULT_MAINTENANCE_RESPIRATION_PARAMETERS),
) -> MaintenanceRespirationResult:

    woody_pft = jnp.asarray(parameters.woody_pft, dtype=jnp.float64)
    _, number_of_patches = _validate_shapes(inputs, state, woody_pft)
    soil_column = jnp.asarray(inputs.soil_column, dtype=jnp.bool_)
    soil_patch = jnp.asarray(inputs.soil_patch, dtype=jnp.bool_)
    vegetation_type = jnp.asarray(inputs.vegetation_type_patch, dtype=jnp.int32)
    exposed = jnp.asarray(inputs.vegetation_exposed_patch, dtype=jnp.int32)
    base_rate = jnp.asarray(parameters.base_rate, dtype=jnp.float64)
    root_base_rate = jnp.asarray(parameters.root_base_rate, dtype=jnp.float64)
    q10 = jnp.asarray(parameters.temperature_q10, dtype=jnp.float64)
    first_crop = jnp.asarray(parameters.first_prognostic_crop_pft, dtype=jnp.int32)
    acclimation_enabled = jnp.asarray(parameters.root_stem_acclimation, dtype=jnp.bool_)
    for name, value in {
        "base_rate": base_rate,
        "root_base_rate": root_base_rate,
        "temperature_q10": q10,
        "first_prognostic_crop_pft": first_crop,
        "root_stem_acclimation": acclimation_enabled,
    }.items():
        if value.shape != ():
            raise ValueError(f"{name} must be a scalar")

    soil_temperature_factor = _temperature_factor(
        jnp.where(soil_column[:, None], inputs.soil_temperature_column_level,
                  jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)),
        q10,
    )
    air_temperature_factor = _temperature_factor(
        jnp.asarray(inputs.reference_temperature_2m_patch, dtype=jnp.float64),
        q10,
    )
    acclimation_factor = _acclimation_factor(
        jnp.asarray(inputs.ten_day_temperature_patch, dtype=jnp.float64)
    )

    def acclimated_rates(_: None):
        final_base, base_by_patch = _running_rates(
            base_rate,
            acclimation_factor,
            soil_patch,
        )
        final_root, root_by_patch = _running_rates(
            root_base_rate,
            acclimation_factor,
            soil_patch,
        )
        flattened_factor = jnp.tile(acclimation_factor, NUMBER_OF_GROUND_LEVELS)
        flattened_filter = jnp.tile(soil_patch, NUMBER_OF_GROUND_LEVELS)
        final_root_after_levels, root_by_flat_position = _running_rates(
            final_root,
            flattened_factor,
            flattened_filter,
        )
        root_by_level_patch = jnp.reshape(
            root_by_flat_position,
            (NUMBER_OF_GROUND_LEVELS, number_of_patches),
        )
        return (
            final_base,
            final_root_after_levels,
            base_by_patch,
            root_by_patch,
            root_by_level_patch,
        )

    def fixed_rates(_: None):
        return (
            base_rate,
            root_base_rate,
            jnp.full((number_of_patches,), base_rate, dtype=jnp.float64),
            jnp.full((number_of_patches,), root_base_rate, dtype=jnp.float64),
            jnp.full(
                (NUMBER_OF_GROUND_LEVELS, number_of_patches),
                root_base_rate,
                dtype=jnp.float64,
            ),
        )

    (
        final_base_rate,
        final_root_base_rate,
        base_rate_patch,
        root_base_rate_patch,
        root_rate_by_level_patch,
    ) = lax.cond(acclimation_enabled, acclimated_rates, fixed_rates, operand=None)

    leaf_active = soil_patch & (exposed == jnp.int32(1))
    sunlit_leaf_flux = fortran_multiply(
        fortran_multiply(
            jnp.where(leaf_active, inputs.sunlit_leaf_respiration_patch, jnp.float64(0.0)),
            jnp.where(leaf_active, inputs.sunlit_leaf_area_patch, jnp.float64(0.0)),
        ),
        jnp.float64(LEAF_CO2_TO_CARBON),
    )
    shaded_leaf_flux = fortran_multiply(
        fortran_multiply(
            jnp.where(leaf_active, inputs.shaded_leaf_respiration_patch, jnp.float64(0.0)),
            jnp.where(leaf_active, inputs.shaded_leaf_area_patch, jnp.float64(0.0)),
        ),
        jnp.float64(LEAF_CO2_TO_CARBON),
    )
    leaf_candidate = jnp.where(
        exposed == jnp.int32(1),
        fortran_add(sunlit_leaf_flux, shaded_leaf_flux),
        jnp.float64(0.0),
    )

    live_stem_candidate = fortran_multiply(
        fortran_multiply(
            jnp.asarray(inputs.live_stem_nitrogen_patch, dtype=jnp.float64),
            base_rate_patch,
        ),
        air_temperature_factor,
    )
    live_coarse_root_candidate = fortran_multiply(
        fortran_multiply(
            jnp.asarray(inputs.live_coarse_root_nitrogen_patch, dtype=jnp.float64),
            root_base_rate_patch,
        ),
        air_temperature_factor,
    )
    grain_candidate = fortran_multiply(
        fortran_multiply(
            jnp.asarray(inputs.grain_nitrogen_patch, dtype=jnp.float64),
            base_rate_patch,
        ),
        air_temperature_factor,
    )
    woody_patch = woody_pft[vegetation_type] == jnp.float64(1.0)
    crop_patch = (~woody_patch) & (vegetation_type >= first_crop)

    leaf = jnp.where(
        soil_patch,
        leaf_candidate,
        get_cn_flux_field(state, _LEAF_FIELD),
    )
    live_stem = jnp.where(
        soil_patch & (woody_patch | crop_patch),
        live_stem_candidate,
        get_cn_flux_field(state, _LIVE_STEM_FIELD),
    )
    live_coarse_root = jnp.where(
        soil_patch & woody_patch,
        live_coarse_root_candidate,
        get_cn_flux_field(state, _LIVE_COARSE_ROOT_FIELD),
    )
    grain = jnp.where(
        soil_patch & crop_patch,
        grain_candidate,
        get_cn_flux_field(state, _GRAIN_FIELD),
    )
    fine_root = _fine_root_respiration(
        get_cn_flux_field(state, _FINE_ROOT_FIELD),
        inputs,
        soil_temperature_factor,
        root_rate_by_level_patch,
    )

    updated_state = set_cn_flux_field(state, _LEAF_FIELD, leaf)
    updated_state = set_cn_flux_field(updated_state, _FINE_ROOT_FIELD, fine_root)
    updated_state = set_cn_flux_field(updated_state, _LIVE_STEM_FIELD, live_stem)
    updated_state = set_cn_flux_field(updated_state, _LIVE_COARSE_ROOT_FIELD, live_coarse_root)
    updated_state = set_cn_flux_field(updated_state, _GRAIN_FIELD, grain)
    return MaintenanceRespirationResult(
        state=updated_state,
        diagnostics=MaintenanceRespirationDiagnostics(
            soil_temperature_factor_column_level=jnp.where(
                soil_column[:, None],
                soil_temperature_factor,
                jnp.float64(0.0),
            ),
            air_temperature_factor_patch=jnp.where(
                soil_patch,
                air_temperature_factor,
                jnp.float64(0.0),
            ),
            final_base_rate=final_base_rate,
            final_root_base_rate=final_root_base_rate,
            updated_patch_count=jnp.sum(soil_patch, dtype=jnp.int32),
        ),
    )


__all__ = [
    "DEFAULT_MAINTENANCE_RESPIRATION_PARAMETERS",
    "FIRST_PROGNOSTIC_CROP_PFT",
    "LEAF_CO2_TO_CARBON",
    "NUMBER_OF_GROUND_LEVELS",
    "MaintenanceRespirationDiagnostics",
    "MaintenanceRespirationInputs",
    "MaintenanceRespirationParameters",
    "MaintenanceRespirationResult",
    "maintenance_respiration",
]
