
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE, SECONDS_PER_DAY
from clm_jax.io.background_litterfall import (
    load_background_litterfall_native_data,
)
from clm_jax.io.bgc_external_forcing import load_bgc_external_forcing_native_data
from clm_jax.io.c_state_update0 import load_c_state_update0_native_data
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.cn_annual_update import load_cn_annual_native_data
from clm_jax.io.cn_fire_area import load_cn_fire_area_native_data
from clm_jax.io.cn_fire_fluxes import load_cn_fire_flux_native_data
from clm_jax.io.cn_fun_allocation import load_fun_allocation_native_data
from clm_jax.io.cn_gap_mortality import load_cn_gap_mortality_native_data
from clm_jax.io.cn_precision_control import load_cn_precision_control_native_data
from clm_jax.io.cn_state_update1 import load_cn_state_update1_native_data
from clm_jax.io.cn_state_update2 import load_cn_state_update2_native_data
from clm_jax.io.cn_state_update3 import load_cn_state_update3_native_data
from clm_jax.io.cn_wood_products import load_cn_wood_products_native_data
from clm_jax.io.crop import load_crop_year_restart
from clm_jax.io.crop_phenology import (
    _current_julian_day,
    load_crop_phenology_native_data,
)
from clm_jax.io.decomposition_rates import load_decomposition_rate_native_data
from clm_jax.io.evergreen_phenology import load_evergreen_phenology_native_data
from clm_jax.io.fun_initialization import load_fun_initialization_native_data
from clm_jax.io.grain_to_product_pools import (
    load_grain_to_product_pools_native_data,
)
from clm_jax.io.growth_respiration import load_growth_respiration_native_data
from clm_jax.io.litter_to_column import load_litter_to_column_native_data
from clm_jax.io.livewood_turnover import load_livewood_turnover_native_data
from clm_jax.io.maintenance_respiration import (
    load_maintenance_respiration_native_data,
)
from clm_jax.io.nitrogen import (
    load_free_living_fixation_restart,
    load_nitrogen_fertilizer_restart,
)
from clm_jax.io.offset_litterfall import load_offset_litterfall_native_data
from clm_jax.io.onset_growth import load_onset_growth_native_data
from clm_jax.io.phenology_climate import load_phenology_climate_native_data
from clm_jax.io.plant_nutrient_competition import (
    load_plant_nutrient_competition_native_data,
)
from clm_jax.io.plant_nutrient_demand import (
    load_plant_nutrient_demand_native_data,
)
from clm_jax.io.restart import (
    SubgridTopology,
    configured_land_restart,
    model_step_from_restart_clock,
)
from clm_jax.io.seasonal_deciduous_phenology import (
    load_seasonal_deciduous_phenology_native_data,
)
from clm_jax.io.soil_biogeochem_competition import (
    load_soil_biogeochem_competition_native_data,
)
from clm_jax.io.soil_biogeochem_decomp import (
    load_soil_biogeochem_decomp_native_data,
)
from clm_jax.io.soil_biogeochem_litter_vertical_transport import (
    load_soil_biogeochem_litter_vertical_transport_native_data,
)
from clm_jax.io.soil_biogeochem_n_state_update1 import (
    load_soil_biogeochem_n_state_update1_native_data,
)
from clm_jax.io.soil_biogeochem_nitrif_denitrif import (
    load_soil_biogeochem_nitrif_denitrif_native_data,
)
from clm_jax.io.soil_biogeochem_potential import (
    load_soil_biogeochem_potential_native_data,
)
from clm_jax.io.soil_biogeochem_vertical_profile import (
    load_soil_biogeochem_vertical_profile_native_data,
)
from clm_jax.io.stress_deciduous_phenology import (
    load_stress_deciduous_phenology_native_data,
)
from clm_jax.physics.cn_zero import initialize_cn_transient_flux_state
from clm_jax.physics.crop import CropYearState
from clm_jax.physics.daylength import daylength
from clm_jax.physics.decomposition_rates import initialize_decomposition_scalar_state
from clm_jax.physics.ecosystem_dynamics import (
    EcosystemDynamicsAnnualUpdateState,
    EcosystemDynamicsCnStateUpdate1State,
    EcosystemDynamicsCnStateUpdate2hPrecisionState,
    EcosystemDynamicsCnStateUpdate2State,
    EcosystemDynamicsCnStateUpdate3PrecisionState,
    EcosystemDynamicsCompetitionDecompositionState,
    EcosystemDynamicsCStateUpdate0PrecisionState,
    EcosystemDynamicsFireAreaState,
    EcosystemDynamicsFireFluxState,
    EcosystemDynamicsGapMortalityState,
    EcosystemDynamicsLitterVerticalTransportState,
    EcosystemDynamicsPhaseTwoLitterState,
    EcosystemDynamicsPreDrainagePrefixState,
    EcosystemDynamicsPreDrainageThroughDecompositionState,
    EcosystemDynamicsSoilNStateUpdate1State,
    EcosystemDynamicsThroughLitterToColumnState,
    EcosystemDynamicsWoodProductsState,
)
from clm_jax.physics.fun_initialization import FunInitializationRuntimeState
from clm_jax.physics.numerics import fortran_add, fortran_divide
from clm_jax.physics.solar import FIXED_YEAR_2000_ORBIT, solar_declination
from clm_jax.physics.solar_runtime import native_grid_radians
from clm_jax.physics.soil_biogeochem_competition import (
    SoilBiogeochemCompetitionState,
)
from clm_jax.physics.soil_biogeochem_decomp import SoilBiogeochemDecompState
from clm_jax.physics.soil_biogeochem_potential import (
    initialize_soil_biogeochem_potential_work_state,
)
from clm_jax.physics.soil_biogeochem_vertical_profile import (
    initialize_soil_biogeochem_vertical_profile_state,
)


class EcosystemDynamicsInitialStateNativeData(NamedTuple):

    state: EcosystemDynamicsAnnualUpdateState
    topology: SubgridTopology
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path


class EcosystemDynamicsNativeChildren(NamedTuple):

    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path
    decomposition: object
    external_forcing: object
    free_living_fixation: object
    nitrogen_fertilizer: object
    maintenance_respiration: object
    potential: object
    nitrif_denitrif: object
    vertical_profile: object
    phenology_climate: object
    evergreen: object
    seasonal: object
    stress: object
    crop: object
    fun_initialization: object
    demand: object
    allocation: object
    soil_competition: object
    plant_competition: object
    soil_decomposition: object
    onset: object
    offset: object
    background: object
    livewood: object
    grain: object
    litter: object
    growth: object
    carbon_update0: object
    precision: object
    update1: object
    soil_nitrogen: object
    transport: object
    mortality: object
    update2: object
    update3: object
    products: object
    fire_area: object
    fire_flux: object
    annual: object


_ECOSYSTEM_NATIVE_CHILDREN_CACHE: dict[
    tuple[CimeCase, Path, Path | None, Path | None],
    EcosystemDynamicsNativeChildren,
] = {}


@jax.jit
def _phenology_daylength(latitude_degrees, calendar_day, step_seconds):
    latitude = native_grid_radians(latitude_degrees)
    previous_day = fortran_add(
        calendar_day, -fortran_divide(step_seconds, jnp.float64(SECONDS_PER_DAY)),
    )
    declination, _ = solar_declination(calendar_day, FIXED_YEAR_2000_ORBIT)
    previous_declination, _ = solar_declination(previous_day, FIXED_YEAR_2000_ORBIT)
    return daylength(latitude, declination), daylength(latitude, previous_declination)


def _refresh_ecosystem_clock(children, external):
    date, seconds = external.call_date, external.call_time_of_day
    clock = external.clock._replace(current_date=date, current_time_of_day=seconds)
    julian_day = _current_julian_day(date)
    model_step = np.int32(model_step_from_restart_clock(clock, "NO_LEAP"))
    calendar_day = np.float64(julian_day) + np.float64(seconds) / SECONDS_PER_DAY
    current_daylength, previous_daylength = _phenology_daylength(
        jnp.asarray(external.topology.grid.latitude_degrees, dtype=jnp.float64),
        jnp.float64(calendar_day), jnp.float64(clock.step_seconds),
    )
    date_fields = dict(month=np.int32((date // 100) % 100),
                       day_of_month=np.int32(date % 100), seconds_of_day=np.int32(seconds))
    return children._replace(
        external_forcing=external,
        phenology_climate=children.phenology_climate._replace(
            inputs=children.phenology_climate.inputs._replace(
                current_date_yyyymmdd=np.int32(date),
                current_time_of_day_seconds=np.int32(seconds),
            ),
        ),
        crop=children.crop._replace(inputs=children.crop.inputs._replace(
            current_julian_day=jnp.int32(julian_day), seconds_of_day=jnp.int32(seconds),
        )),
        seasonal=children.seasonal._replace(inputs=children.seasonal.inputs._replace(
            daylength_seconds_gridcell=current_daylength,
            previous_daylength_seconds_gridcell=previous_daylength,
        )),
        stress=children.stress._replace(persistent_inputs=children.stress.persistent_inputs._replace(
            daylength_seconds_gridcell=current_daylength,
        )),
        fun_initialization=children.fun_initialization._replace(
            inputs=children.fun_initialization.inputs._replace(model_step=model_step),
            clock=clock,
        ),
        fire_area=children.fire_area._replace(parameters=children.fire_area.parameters._replace(
            model_step=model_step, **date_fields,
        )),
        fire_flux=children.fire_flux._replace(parameters=children.fire_flux.parameters._replace(
            **date_fields,
        )),
    )


def _resolved(path: str | Path | None) -> Path | None:
    return None if path is None else Path(path).expanduser().resolve()


def _require_path(label: str, expected: Path, actual: Path) -> None:
    if expected != actual:
        raise ValueError(f"ecosystem initial-state children use different {label} files")


def _initial_soil_competition_state(native) -> SoilBiogeochemCompetitionState:
    persistent = native.persistent
    return SoilBiogeochemCompetitionState(
        fraction_potential_growth_column=persistent.fraction_potential_growth_column,
        fraction_potential_immobilization_column=jnp.full(
            native.soil_column.shape,
            jnp.float64(CLM_SPECIAL_VALUE),
            dtype=jnp.float64,
        ),
        fraction_potential_immobilization_column_level=(
            persistent.fraction_potential_immobilization_column_level
        ),
    )


def _initial_soil_decomposition_state(
    soil_column: jnp.ndarray,
    patch_count: int,
    column_count: int,
    decomposition_level_count: int,
) -> SoilBiogeochemDecompState:
    level_count = 25
    transition_count = 10
    level = jnp.arange(level_count, dtype=jnp.int32)
    active_tail = jnp.asarray(soil_column, dtype=jnp.bool_)[:, None] & (
        level[None, :] >= jnp.int32(decomposition_level_count)
    )
    fraction = jnp.where(
        active_tail,
        jnp.float64(0.0),
        jnp.float64(CLM_SPECIAL_VALUE),
    )
    return SoilBiogeochemDecompState(
        cn_fluxes=initialize_cn_transient_flux_state(
            patch_count, column_count, fieldwise=True
        ),
        fraction_potential_heterotrophic_respiration_column_level=fraction,
        denitrification_column_level_transition=jnp.full(
            (column_count, level_count, transition_count),
            jnp.float64(CLM_SPECIAL_VALUE),
            dtype=jnp.float64,
        ),
    )


def _runtime_state(parameters) -> FunInitializationRuntimeState:
    seconds_per_year = np.float64(parameters.seconds_per_day) * np.float64(parameters.days_per_year)
    steps_per_year = np.int32(seconds_per_year / np.float64(parameters.time_step_seconds))
    return FunInitializationRuntimeState(
        time_step_seconds=jnp.asarray(parameters.time_step_seconds, dtype=jnp.float64),
        onset_period_days=jnp.asarray(parameters.onset_period_days, dtype=jnp.float64),
        offset_period_days=jnp.asarray(parameters.offset_period_days, dtype=jnp.float64),
        steps_per_year=jnp.asarray(steps_per_year, dtype=jnp.int32),
    )


def load_ecosystem_dynamics_native_children(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    call_date: int | None = None,
    call_time_of_day: int | None = None,
) -> EcosystemDynamicsNativeChildren:

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")

    requested_surface = _resolved(surface_data_path)
    requested_parameter = _resolved(parameter_data_path)
    cache_key = (case, restart, requested_surface, requested_parameter)
    cached = _ECOSYSTEM_NATIVE_CHILDREN_CACHE.get(cache_key)
    if cached is not None:
        external_forcing = load_bgc_external_forcing_native_data(
            case,
            restart,
            call_date=call_date,
            call_time_of_day=call_time_of_day,
        )
        return _refresh_ecosystem_clock(cached, external_forcing)

    external_forcing = load_bgc_external_forcing_native_data(
        case,
        restart,
        call_date=call_date,
        call_time_of_day=call_time_of_day,
    )
    free_living_fixation = load_free_living_fixation_restart(case, restart)
    nitrogen_fertilizer = load_nitrogen_fertilizer_restart(case, restart)
    decomposition = load_decomposition_rate_native_data(
        case,
        restart_path=restart,
        parameter_data_path=parameter_data_path,
    )
    potential = load_soil_biogeochem_potential_native_data(
        case,
        restart_path=restart,
        surface_data_path=surface_data_path,
        parameter_data_path=parameter_data_path,
    )
    maintenance_respiration = load_maintenance_respiration_native_data(
        case,
        restart,
        potential.surface_data_path,
        potential.parameter_data_path,
    )
    nitrif_denitrif = load_soil_biogeochem_nitrif_denitrif_native_data(
        case,
        restart,
        potential.surface_data_path,
        potential.parameter_data_path,
    )
    soil_decomposition = load_soil_biogeochem_decomp_native_data(
        case,
        restart,
        potential.surface_data_path,
        potential.parameter_data_path,
        potential_native_data=potential,
    )
    vertical_profile = load_soil_biogeochem_vertical_profile_native_data(
        case,
        restart_path=restart,
        surface_data_path=potential.surface_data_path,
        parameter_data_path=potential.parameter_data_path,
        include_inactive=True,
    )
    phenology_climate = load_phenology_climate_native_data(case, restart)
    evergreen = load_evergreen_phenology_native_data(case, restart, potential.parameter_data_path)
    seasonal = load_seasonal_deciduous_phenology_native_data(
        case, restart, potential.parameter_data_path
    )
    stress = load_stress_deciduous_phenology_native_data(
        case,
        restart,
        potential.parameter_data_path,
        potential.surface_data_path,
    )
    crop = load_crop_phenology_native_data(
        case,
        restart,
        potential.parameter_data_path,
        potential.surface_data_path,
    )
    fun_initialization = load_fun_initialization_native_data(
        case, restart, potential.parameter_data_path
    )
    demand = load_plant_nutrient_demand_native_data(case, restart, potential.parameter_data_path)
    allocation = load_fun_allocation_native_data(
        case,
        restart,
        potential.surface_data_path,
        potential.parameter_data_path,
    )
    soil_competition = load_soil_biogeochem_competition_native_data(
        case, restart, potential.parameter_data_path
    )
    plant_competition = load_plant_nutrient_competition_native_data(
        case,
        restart,
        potential.parameter_data_path,
        demand_native_data=demand,
    )

    offset = load_offset_litterfall_native_data(case, restart, potential.parameter_data_path)
    onset = load_onset_growth_native_data(case, restart, potential.parameter_data_path)
    background = load_background_litterfall_native_data(
        case,
        restart,
        potential.parameter_data_path,
        offset_native_data=offset,
    )
    livewood = load_livewood_turnover_native_data(
        case,
        restart,
        potential.parameter_data_path,
        background_native_data=background,
    )
    grain = load_grain_to_product_pools_native_data(
        case,
        restart,
        potential.parameter_data_path,
        livewood_native_data=livewood,
    )
    litter = load_litter_to_column_native_data(
        case,
        restart,
        potential.surface_data_path,
        potential.parameter_data_path,
        grain_native_data=grain,
        vertical_profile_native_data=vertical_profile,
    )
    growth = load_growth_respiration_native_data(
        case,
        restart,
        potential.parameter_data_path,
        litter_native_data=litter,
    )
    carbon_update0 = load_c_state_update0_native_data(
        case,
        restart,
        potential.parameter_data_path,
        growth_respiration_native_data=growth,
    )
    precision = load_cn_precision_control_native_data(
        case,
        restart,
        potential.parameter_data_path,
        c_state_update0_native_data=carbon_update0,
    )
    update1 = load_cn_state_update1_native_data(
        case,
        restart,
        potential.parameter_data_path,
        precision_native_data=precision,
    )
    soil_nitrogen = load_soil_biogeochem_n_state_update1_native_data(
        case,
        restart,
        cn_state_update1_native_data=update1,
    )
    transport = load_soil_biogeochem_litter_vertical_transport_native_data(
        case,
        restart,
        potential.surface_data_path,
        potential.parameter_data_path,
        potential_native_data=potential,
    )
    mortality = load_cn_gap_mortality_native_data(
        case,
        restart,
        potential.surface_data_path,
        potential.parameter_data_path,
        state_update_native_data=update1,
        vertical_profile_native_data=vertical_profile,
    )
    update2 = load_cn_state_update2_native_data(
        case,
        restart,
        potential.surface_data_path,
        potential.parameter_data_path,
        gap_mortality_native_data=mortality,
        vertical_transport_native_data=transport,
        state_update1_native_data=update1,
    )
    update3 = load_cn_state_update3_native_data(
        case,
        restart,
        potential.surface_data_path,
        potential.parameter_data_path,
    )
    products = load_cn_wood_products_native_data(
        case,
        restart,
        potential.parameter_data_path,
        state_update2_native_data=update2,
    )
    fire_area = load_cn_fire_area_native_data(
        case,
        restart,
        potential.surface_data_path,
        potential.parameter_data_path,
    )
    fire_flux = load_cn_fire_flux_native_data(
        case,
        restart,
        potential.surface_data_path,
        potential.parameter_data_path,
        fire_area_native_data=fire_area,
        vertical_profile_native_data=vertical_profile,
    )
    annual = load_cn_annual_native_data(case, restart)

    path_children = (
        external_forcing,
        free_living_fixation,
        nitrogen_fertilizer,
        decomposition,
        maintenance_respiration,
        potential,
        nitrif_denitrif,
        soil_decomposition,
        vertical_profile,
        phenology_climate,
        evergreen,
        seasonal,
        stress,
        crop,
        fun_initialization,
        demand,
        allocation,
        soil_competition,
        plant_competition,
        onset,
        offset,
        update1,
        soil_nitrogen,
        transport,
        mortality,
        update3,
        products,
        fire_area,
        fire_flux,
        annual,
    )
    for child in path_children:
        _require_path("restart", restart, child.restart_path)
    for child in (
        decomposition,
        maintenance_respiration,
        potential,
        nitrif_denitrif,
        soil_decomposition,
        vertical_profile,
        evergreen,
        seasonal,
        stress,
        crop,
        fun_initialization,
        demand,
        allocation,
        soil_competition,
        plant_competition,
        onset,
        offset,
        update1,
        transport,
        mortality,
        update3,
        products,
        fire_area,
        fire_flux,
    ):
        _require_path("parameter-data", potential.parameter_data_path, child.parameter_data_path)
    surface_children = (
        vertical_profile,
        maintenance_respiration,
        nitrif_denitrif,
        soil_decomposition,
        stress,
        crop,
        allocation,
        transport,
        mortality,
        update3,
        fire_area,
        fire_flux,
    )
    for child in surface_children:
        _require_path("surface-data", potential.surface_data_path, child.surface_data_path)
    if requested_surface is not None:
        _require_path("surface-data", requested_surface, potential.surface_data_path)
    if requested_parameter is not None:
        _require_path("parameter-data", requested_parameter, potential.parameter_data_path)

    result = EcosystemDynamicsNativeChildren(
        restart_path=restart,
        surface_data_path=potential.surface_data_path,
        parameter_data_path=potential.parameter_data_path,
        decomposition=decomposition,
        external_forcing=external_forcing,
        free_living_fixation=free_living_fixation,
        nitrogen_fertilizer=nitrogen_fertilizer,
        maintenance_respiration=maintenance_respiration,
        potential=potential,
        nitrif_denitrif=nitrif_denitrif,
        vertical_profile=vertical_profile,
        phenology_climate=phenology_climate,
        evergreen=evergreen,
        seasonal=seasonal,
        stress=stress,
        crop=crop,
        fun_initialization=fun_initialization,
        demand=demand,
        allocation=allocation,
        soil_competition=soil_competition,
        plant_competition=plant_competition,
        soil_decomposition=soil_decomposition,
        onset=onset,
        offset=offset,
        background=background,
        livewood=livewood,
        grain=grain,
        litter=litter,
        growth=growth,
        carbon_update0=carbon_update0,
        precision=precision,
        update1=update1,
        soil_nitrogen=soil_nitrogen,
        transport=transport,
        mortality=mortality,
        update2=update2,
        update3=update3,
        products=products,
        fire_area=fire_area,
        fire_flux=fire_flux,
        annual=annual,
    )
    if jax.default_backend() == "gpu":
        from clm_jax.runtime import co_locate_array_leaves

        result = co_locate_array_leaves(result, jax.local_devices()[0])
    _ECOSYSTEM_NATIVE_CHILDREN_CACHE[cache_key] = result
    return _refresh_ecosystem_clock(result, external_forcing)


def load_ecosystem_dynamics_initial_state_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    native_children: EcosystemDynamicsNativeChildren | None = None,
) -> EcosystemDynamicsInitialStateNativeData:

    children = (
        load_ecosystem_dynamics_native_children(
            case,
            restart_path=restart_path,
            surface_data_path=surface_data_path,
            parameter_data_path=parameter_data_path,
        )
        if native_children is None
        else native_children
    )
    requested_restart = _resolved(restart_path)
    requested_surface = _resolved(surface_data_path)
    requested_parameter = _resolved(parameter_data_path)
    if requested_restart is not None:
        _require_path("restart", requested_restart, children.restart_path)
    if requested_surface is not None:
        _require_path("surface-data", requested_surface, children.surface_data_path)
    if requested_parameter is not None:
        _require_path("parameter-data", requested_parameter, children.parameter_data_path)

    restart = children.restart_path
    potential = children.potential
    vertical_profile = children.vertical_profile
    phenology_climate = children.phenology_climate
    evergreen = children.evergreen
    seasonal = children.seasonal
    stress = children.stress
    crop = children.crop
    fun_initialization = children.fun_initialization
    demand = children.demand
    allocation = children.allocation
    soil_competition = children.soil_competition
    plant_competition = children.plant_competition
    offset = children.offset
    update1 = children.update1
    soil_nitrogen = children.soil_nitrogen
    transport = children.transport
    mortality = children.mortality
    products = children.products
    fire_area = children.fire_area
    fire_flux = children.fire_flux
    annual = children.annual

    topology = vertical_profile.topology
    patch_count = topology.patch.gridcell.shape[0]
    column_count = topology.column.gridcell.shape[0]
    initial_fluxes = initialize_cn_transient_flux_state(
        patch_count, column_count, fieldwise=True
    )
    prefix = EcosystemDynamicsPreDrainagePrefixState(
        crop_year=CropYearState(
            years_active_patch=load_crop_year_restart(restart).years_active_patch
        ),
        cn_fluxes=initial_fluxes,
        decomposition_scalars=initialize_decomposition_scalar_state(column_count),
        soil_biogeochem_potential_work=(
            initialize_soil_biogeochem_potential_work_state(column_count)
        ),
        soil_biogeochem_vertical_profile=(
            initialize_soil_biogeochem_vertical_profile_state(
                patch_count,
                column_count,
            )
        ),
        phenology_climate=phenology_climate.state,
        evergreen_phenology=evergreen.state,
        seasonal_deciduous_phenology=seasonal.state,
        stress_deciduous_phenology=stress.state,
        crop_phenology=crop.state,
        fun_initialization=fun_initialization.state,
        fun_initialization_runtime=_runtime_state(fun_initialization.parameters),
        plant_nutrient_demand=demand.state,
        plant_nitrogen_demand_column=jnp.full(
            (column_count,),
            jnp.float64(CLM_SPECIAL_VALUE),
            dtype=jnp.float64,
        ),
    )
    competition = EcosystemDynamicsCompetitionDecompositionState(
        fun_allocation=allocation.state,
        soil_biogeochem_competition=_initial_soil_competition_state(soil_competition),
        plant_nutrient_competition=plant_competition.state,
        soil_biogeochem_decomp=_initial_soil_decomposition_state(
            potential.soil_column,
            patch_count,
            column_count,
            int(potential.parameters.decomposition_level_count),
        ),
    )
    pre_decomposition = EcosystemDynamicsPreDrainageThroughDecompositionState(
        prefix=prefix,
        competition_decomposition=competition,
    )
    litter_state = EcosystemDynamicsThroughLitterToColumnState(
        pre_decomposition=pre_decomposition,
        phase_two_litter=EcosystemDynamicsPhaseTwoLitterState(
            leaf_carbon_to_litter_fun_patch=(offset.state.leaf_carbon_to_litter_fun_patch)
        ),
    )
    first_guardrail = EcosystemDynamicsCStateUpdate0PrecisionState(
        ecosystem=litter_state,
        vegetation_carbon_nitrogen=update1.restart_state.vegetation,
    )
    second_guardrail = EcosystemDynamicsCnStateUpdate1State(
        first_guardrail=first_guardrail,
        xsmrpool_loss_patch=update1.restart_state.xsmrpool_loss_patch,
    )
    soil_n_state = EcosystemDynamicsSoilNStateUpdate1State(
        second_guardrail=second_guardrail,
        soil_nitrogen=soil_nitrogen.restart_state,
    )
    vertical_transport_state = EcosystemDynamicsLitterVerticalTransportState(
        soil_n_state_update=soil_n_state,
        vertical_transport=transport.restart_state,
    )
    gap_state = EcosystemDynamicsGapMortalityState(
        litter_vertical_transport=vertical_transport_state,
        gap_mortality=mortality.restart_state,
    )
    update2_state = EcosystemDynamicsCnStateUpdate2State(gap_mortality=gap_state)
    update2h_state = EcosystemDynamicsCnStateUpdate2hPrecisionState(state_update2=update2_state)
    product_state = EcosystemDynamicsWoodProductsState(
        state_update2h_precision=update2h_state,
        wood_products=products.restart_state,
    )
    fire_area_state = EcosystemDynamicsFireAreaState(
        wood_products=product_state,
        fire_area=fire_area.restart_state,
    )
    fire_flux_state = EcosystemDynamicsFireFluxState(
        fire_area=fire_area_state,
        fire_flux=fire_flux.restart_state,
    )
    state = EcosystemDynamicsAnnualUpdateState(
        state_update3_precision=EcosystemDynamicsCnStateUpdate3PrecisionState(
            fire_flux=fire_flux_state
        ),
        annual=annual.restart_state,
    )
    return EcosystemDynamicsInitialStateNativeData(
        state=state,
        topology=topology,
        restart_path=restart,
        surface_data_path=potential.surface_data_path,
        parameter_data_path=potential.parameter_data_path,
    )


__all__ = [
    "EcosystemDynamicsInitialStateNativeData",
    "EcosystemDynamicsNativeChildren",
    "load_ecosystem_dynamics_initial_state_native_data",
    "load_ecosystem_dynamics_native_children",
]
