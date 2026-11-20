
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.canopy_temperature_restart import (
    soil_layer_geometry_20sl_8p5m,
)
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.restart import SubgridTopology
from clm_jax.io.soil_biogeochem_potential import (
    SoilBiogeochemPotentialNativeData,
    SoilBiogeochemPotentialPoolState,
    load_soil_biogeochem_potential_native_data,
)
from clm_jax.physics.cn_fun_competition import FunSoilCompetitionResult
from clm_jax.physics.cn_zero import CnTransientFluxState
from clm_jax.physics.cn_zero_manifest import (
    NUMBER_OF_DECOMPOSITION_LEVELS,
    NUMBER_OF_DECOMPOSITION_TRANSITIONS,
)
from clm_jax.physics.decomposition_rates import DecompositionRateResult
from clm_jax.physics.plant_nutrient_competition import (
    PlantNutrientCompetitionResult,
)
from clm_jax.physics.soil_biogeochem_decomp import (
    SoilBiogeochemDecompInputs,
    SoilBiogeochemDecompParameters,
    SoilBiogeochemDecompState,
)
from clm_jax.physics.soil_biogeochem_potential import (
    SoilBiogeochemPotentialResult,
    SoilBiogeochemPotentialWorkState,
)


class SoilBiogeochemDecompNativeData(NamedTuple):

    topology: SubgridTopology
    soil_column: jnp.ndarray
    pools: SoilBiogeochemPotentialPoolState
    respiration_fraction_column_level_transition: jnp.ndarray
    parameters: SoilBiogeochemDecompParameters
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path


class SoilBiogeochemDecompCall(NamedTuple):

    inputs: SoilBiogeochemDecompInputs
    state: SoilBiogeochemDecompState
    work: SoilBiogeochemPotentialWorkState
    parameters: SoilBiogeochemDecompParameters


def _resolved_optional_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def _require_same_selected_file(
    name: str,
    requested: str | Path | None,
    actual: Path,
) -> None:
    resolved = _resolved_optional_path(requested)
    if resolved is not None and resolved != actual:
        raise ValueError(f"potential native data uses a different {name} file")


def load_soil_biogeochem_decomp_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    potential_native_data: SoilBiogeochemPotentialNativeData | None = None,
) -> SoilBiogeochemDecompNativeData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    required = {
        "use_cn": True,
        "use_century_decomp": True,
        "use_vertsoilc": True,
        "use_fates": False,
        "use_lch4": True,
        "use_nitrif_denitrif": True,
        "soil_layerstruct": "20SL_8.5m",
    }
    for name, expected in required.items():
        if general.get(name) != expected:
            raise ValueError(
                f"native actual-decomposition adapter requires "
                f"{name}={expected!r}"
            )

    potential = potential_native_data
    if potential is None:
        potential = load_soil_biogeochem_potential_native_data(
            case,
            restart_path=restart_path,
            surface_data_path=surface_data_path,
            parameter_data_path=parameter_data_path,
        )
    else:
        _require_same_selected_file("restart", restart_path, potential.restart_path)
        _require_same_selected_file(
            "surface-data",
            surface_data_path,
            potential.surface_data_path,
        )
        _require_same_selected_file(
            "parameter-data",
            parameter_data_path,
            potential.parameter_data_path,
        )

    _, _, thickness = soil_layer_geometry_20sl_8p5m()
    source = potential.parameters
    parameters = SoilBiogeochemDecompParameters(
        decomposition_level_count=source.decomposition_level_count,
        decomposition_pool_count=source.decomposition_pool_count,
        decomposition_transition_count=source.decomposition_transition_count,
        use_fates=False,
        use_methane=True,
        use_nitrification_denitrification=True,
        denitrification_proportion=source.denitrification_proportion,
        cascade_donor_pool=source.cascade_donor_pool,
        cascade_receiver_pool=source.cascade_receiver_pool,
        floating_cn_ratio_pool=source.floating_cn_ratio_pool,
        initial_cn_ratio_pool=source.initial_cn_ratio_pool,
        soil_layer_thickness=jnp.asarray(thickness, dtype=jnp.float64),
    )
    return SoilBiogeochemDecompNativeData(
        topology=potential.topology,
        soil_column=potential.soil_column,
        pools=potential.pools,
        respiration_fraction_column_level_transition=(
            potential.respiration_fraction_column_level_transition
        ),
        parameters=parameters,
        restart_path=potential.restart_path,
        surface_data_path=potential.surface_data_path,
        parameter_data_path=potential.parameter_data_path,
    )


def _validate_flux_extent(
    name: str,
    fluxes: CnTransientFluxState,
    column_count: int,
) -> None:
    if fluxes.column_scalar.shape[0] != column_count:
        raise ValueError(f"same-step {name} result has the wrong column count")


def assemble_soil_biogeochem_decomp_call(
    native: SoilBiogeochemDecompNativeData,
    decomposition_rate_result: DecompositionRateResult,
    potential_result: SoilBiogeochemPotentialResult,
    fun_soil_competition_result: FunSoilCompetitionResult,
    plant_nutrient_competition_result: PlantNutrientCompetitionResult,
) -> SoilBiogeochemDecompCall:

    column_count = native.soil_column.shape[0]
    level_shape = (column_count, NUMBER_OF_DECOMPOSITION_LEVELS)
    transition_shape = (
        column_count,
        NUMBER_OF_DECOMPOSITION_LEVELS,
        NUMBER_OF_DECOMPOSITION_TRANSITIONS,
    )
    _validate_flux_extent(
        "decomposition-rate",
        decomposition_rate_result.cn_fluxes,
        column_count,
    )
    _validate_flux_extent("potential", potential_result.cn_fluxes, column_count)
    _validate_flux_extent(
        "FUN competition",
        fun_soil_competition_result.competition.cn_fluxes,
        column_count,
    )
    final_fluxes = plant_nutrient_competition_result.state.cn_fluxes
    _validate_flux_extent("plant allocation", final_fluxes, column_count)

    water_scalar = jnp.asarray(
        decomposition_rate_result.scalars.water_scalar_column_level,
        dtype=jnp.float64,
    )
    if water_scalar.shape != level_shape:
        raise ValueError("same-step water scalar must have shape (column, 25)")
    fpi = jnp.asarray(
        fun_soil_competition_result.competition.state
        .fraction_potential_immobilization_column_level,
        dtype=jnp.float64,
    )
    if fpi.shape != level_shape:
        raise ValueError("same-step fpi_vr must have shape (column, 25)")
    if jnp.shape(native.respiration_fraction_column_level_transition) != (
        transition_shape
    ):
        raise ValueError("native respiration fraction has the wrong shape")

    level = jnp.arange(NUMBER_OF_DECOMPOSITION_LEVELS, dtype=jnp.int32)
    active_tail = jnp.asarray(native.soil_column, dtype=jnp.bool_)[:, None] & (
        level[None, :]
        >= jnp.asarray(
            native.parameters.decomposition_level_count,
            dtype=jnp.int32,
        )
    )
    fraction_potential_hr = jnp.where(
        active_tail,
        jnp.float64(0.0),
        jnp.float64(CLM_SPECIAL_VALUE),
    )
    denitrification = jnp.full(
        transition_shape,
        jnp.float64(CLM_SPECIAL_VALUE),
        dtype=jnp.float64,
    )
    return SoilBiogeochemDecompCall(
        inputs=SoilBiogeochemDecompInputs(
            soil_column=native.soil_column,
            decomposition_carbon_pool_column_level_pool=(
                native.pools.decomposition_carbon_pool_column_level_pool
            ),
            decomposition_nitrogen_pool_column_level_pool=(
                native.pools.decomposition_nitrogen_pool_column_level_pool
            ),
            fraction_potential_immobilization_column_level=fpi,
            respiration_fraction_column_level_transition=(
                native.respiration_fraction_column_level_transition
            ),
            water_scalar_column_level=water_scalar,
        ),
        state=SoilBiogeochemDecompState(
            cn_fluxes=final_fluxes,
            fraction_potential_heterotrophic_respiration_column_level=(
                fraction_potential_hr
            ),
            denitrification_column_level_transition=denitrification,
        ),
        work=potential_result.work,
        parameters=native.parameters,
    )


__all__ = [
    "SoilBiogeochemDecompCall",
    "SoilBiogeochemDecompNativeData",
    "assemble_soil_biogeochem_decomp_call",
    "load_soil_biogeochem_decomp_native_data",
]
