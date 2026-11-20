
from __future__ import annotations

from typing import Any, Protocol

import jax
import jax.numpy as jnp

from clm_jax.io.ecosystem_dynamics_fast_trace import (
    EcosystemDynamicsFastTraceNativeData,
)
from clm_jax.io.ecosystem_fast_boundary import (
    EcosystemFastColumn,
    EcosystemFastGridcell,
    EcosystemFastPatch,
)
from clm_jax.physics.cn_precision_control import (
    CnPrecisionControlState,
    get_cn_precision_control_state_field,
)
from clm_jax.physics.ecosystem_dynamics import EcosystemDynamicsCompleteInputs
from clm_jax.physics.evergreen_phenology import (
    CARBON_STORAGE_POOL_NAMES,
    CARBON_TRANSFER_POOL_NAMES,
    NITROGEN_STORAGE_POOL_NAMES,
    NITROGEN_TRANSFER_POOL_NAMES,
)


def _real(value) -> jnp.ndarray:
    return jnp.asarray(value, dtype=jnp.float64)


def _integer(value) -> jnp.ndarray:
    return jnp.asarray(value, dtype=jnp.int32)


def _boolean(value) -> jnp.ndarray:
    return jnp.asarray(value, dtype=jnp.bool_)


class EcosystemDynamicsFastInputFields(Protocol):

    gridcell: EcosystemFastGridcell
    column: EcosystemFastColumn
    patch: EcosystemFastPatch


def inject_ecosystem_dynamics_fast_inputs(
    inputs: EcosystemDynamicsCompleteInputs,
    snapshot: EcosystemDynamicsFastInputFields,
) -> EcosystemDynamicsCompleteInputs:

    grid = snapshot.gridcell
    column = snapshot.column
    patch = snapshot.patch

    state_update3 = inputs.through_state_update3_precision
    through_fire_flux = state_update3.through_fire_flux
    through_fire_area = through_fire_flux.through_fire_area
    through_wood_products = through_fire_area.through_wood_products
    through_update2h = through_wood_products.through_state_update2h_precision
    through_update2 = through_update2h.through_state_update2
    through_gap = through_update2.through_gap_mortality
    through_transport = through_gap.through_litter_vertical_transport
    through_soil_n = through_transport.through_soil_n_state_update
    through_second_guardrail = through_soil_n.through_second_guardrail
    through_first_guardrail = through_second_guardrail.through_first_guardrail
    through_growth = through_first_guardrail.through_growth_respiration
    through_litter = through_growth.through_litter
    pre_decomposition = through_litter.pre_decomposition
    prefix = pre_decomposition.prefix
    competition = pre_decomposition.competition_decomposition

    patch_weight_on_column = _real(patch.weight_on_column)
    patch_weight_on_gridcell = _real(patch.weight_on_gridcell)
    patch_active = _boolean(patch.active)
    soil_temperature = _real(column.soil_temperature)
    soil_water_potential = _real(column.soil_water_potential)
    soil_liquid_water = _real(column.soil_liquid_water)
    soil_volumetric_water = _real(column.soil_volumetric_water)
    root_fraction = _real(patch.carbon_root_fraction)

    maintenance = prefix.maintenance_respiration._replace(
        vegetation_exposed_patch=_integer(patch.vegetation_exposed),
        sunlit_leaf_area_patch=_real(patch.sunlit_leaf_area),
        shaded_leaf_area_patch=_real(patch.shaded_leaf_area),
        carbon_root_fraction_patch_level=root_fraction,
        soil_temperature_column_level=soil_temperature,
        reference_temperature_2m_patch=_real(patch.reference_temperature_2m),
        ten_day_temperature_patch=_real(patch.temperature_10day),
        sunlit_leaf_respiration_patch=_real(patch.sunlit_leaf_respiration),
        shaded_leaf_respiration_patch=_real(patch.shaded_leaf_respiration),
    )
    decomposition = prefix.decomposition_rate._replace(
        soil_water_potential_column_level=soil_water_potential,
        soil_temperature_column_level=soil_temperature,
        oxygen_stress_saturated_column_level=_real(column.oxygen_stress_saturated),
        oxygen_stress_unsaturated_column_level=_real(column.oxygen_stress_unsaturated),
        inundated_fraction_column=_real(column.inundated_fraction),
    )
    vertical_profile = prefix.soil_biogeochem_vertical_profile._replace(
        patch_weight_on_column=patch_weight_on_column,
        carbon_root_fraction_patch_level=root_fraction,
        active_layer_index_column=_integer(column.active_layer_index),
    )
    nitrif_denitrif = prefix.soil_biogeochem_nitrif_denitrif._replace(
        soil_water_potential_column_level=soil_water_potential,
        volumetric_water_column_level=soil_volumetric_water,
        liquid_water_mass_column_level=soil_liquid_water,
        soil_temperature_column_level=soil_temperature,
        inundated_fraction_column=_real(column.inundated_fraction),
    )
    phenology_climate = prefix.phenology_climate._replace(
        reference_temperature_2m_patch=_real(patch.reference_temperature_2m),
        growing_degree_days_base_0_patch=_real(patch.growing_degree_days_base_0),
        growing_degree_days_base_8_patch=_real(patch.growing_degree_days_base_8),
        growing_degree_days_base_10_patch=_real(patch.growing_degree_days_base_10),
        years_crop_active_patch=_integer(patch.years_crop_active),
    )
    seasonal = prefix.seasonal_deciduous_phenology._replace(
        third_soil_layer_temperature_column=soil_temperature[:, 2],
    )
    stress = prefix.stress_deciduous_phenology._replace(
        ten_day_mean_precipitation_patch=_real(patch.precipitation_10day),
        third_soil_layer_temperature_column=soil_temperature[:, 2],
        third_soil_layer_water_potential_column=soil_water_potential[:, 2],
    )
    crop = prefix.crop_phenology._replace(
        temperature_2m_patch=_real(patch.reference_temperature_2m),
        minimum_temperature_2m_patch=_real(patch.minimum_temperature_2m),
        maximum_temperature_2m_patch=_real(patch.maximum_temperature_2m),
        ten_day_temperature_2m_patch=_real(patch.temperature_10day),
        five_day_minimum_temperature_2m_patch=_real(patch.minimum_temperature_5day),
        ten_day_minimum_temperature_2m_patch=_real(patch.minimum_temperature_10day),
        twenty_year_gdd0_patch=_real(patch.growing_degree_days_base_0_twenty_year),
        twenty_year_gdd8_patch=_real(patch.growing_degree_days_base_8_twenty_year),
        twenty_year_gdd10_patch=_real(patch.growing_degree_days_base_10_twenty_year),
        growing_degree_days_topsoil_patch=_real(patch.growing_degree_days_topsoil),
        fertilizer_nitrogen_patch=_real(patch.fertilizer_nitrogen),
        total_leaf_area_index_patch=_real(patch.total_leaf_area),
        snow_depth_column=_real(column.snow_depth),
    )
    nutrient_demand = prefix.plant_nutrient_demand._replace(
        sunlit_photosynthesis_patch=_real(patch.sunlit_photosynthesis),
        shaded_photosynthesis_patch=_real(patch.shaded_photosynthesis),
        sunlit_leaf_area_index_patch=_real(patch.sunlit_leaf_area),
        shaded_leaf_area_index_patch=_real(patch.shaded_leaf_area),
        growing_degree_days_since_planting_patch=(_real(patch.growing_degree_days_since_planting)),
        growing_degree_days_topsoil_patch=_real(patch.growing_degree_days_topsoil),
        crop_live_patch=_boolean(patch.crop_live),
        transpiration_wetness_factor_patch=_real(patch.transpiration_wetness),
    )
    prefix = prefix._replace(
        nitrogen_deposition=prefix.nitrogen_deposition._replace(
            forcing_ndep_gridcell=_real(grid.nitrogen_deposition),
        ),
        free_living_fixation=prefix.free_living_fixation._replace(
            annual_average_evapotranspiration_column=(_real(column.annual_evapotranspiration)),
        ),
        nitrogen_fertilizer=prefix.nitrogen_fertilizer._replace(
            patch_weight_on_column=patch_weight_on_column,
            patch_active=patch_active,
        ),
        maintenance_respiration=maintenance,
        decomposition_rate=decomposition,
        soil_biogeochem_vertical_profile=vertical_profile,
        soil_biogeochem_nitrif_denitrif=nitrif_denitrif,
        phenology_climate=phenology_climate,
        seasonal_deciduous_phenology=seasonal,
        stress_deciduous_phenology=stress,
        crop_phenology=crop,
        plant_nutrient_demand=nutrient_demand,
        plant_nutrient_demand_aggregation=(
            prefix.plant_nutrient_demand_aggregation._replace(
                patch_weight_on_column=patch_weight_on_column,
                patch_active=patch_active,
            )
        ),
    )

    competition = competition._replace(
        fun_competition_coupling=competition.fun_competition_coupling._replace(
            patch_weight_on_column=patch_weight_on_column,
            patch_active=patch_active,
        ),
        fun_allocation=competition.fun_allocation._replace(
            root_carbon_fraction_patch_level=root_fraction,
            soil_temperature_column_level=soil_temperature,
            soil_liquid_water_column_level=soil_liquid_water,
        ),
        plant_nutrient_competition=(
            competition.plant_nutrient_competition._replace(
                crop_live_patch=_boolean(patch.crop_live),
                sunlit_leaf_area_index_patch=_real(patch.sunlit_leaf_area),
                shaded_leaf_area_index_patch=_real(patch.shaded_leaf_area),
            )
        ),
    )
    pre_decomposition = pre_decomposition._replace(
        prefix=prefix,
        competition_decomposition=competition,
    )

    phase_two = through_litter.phase_two_litter._replace(
        grain_to_product_pools=(
            through_litter.phase_two_litter.grain_to_product_pools._replace(
                patch_weight_on_column=patch_weight_on_column,
                patch_active=patch_active,
            )
        ),
        litter_to_column=through_litter.phase_two_litter.litter_to_column._replace(
            patch_weight_on_column=patch_weight_on_column,
            patch_active=patch_active,
        ),
    )
    through_litter = through_litter._replace(
        pre_decomposition=pre_decomposition,
        phase_two_litter=phase_two,
    )
    through_growth = through_growth._replace(through_litter=through_litter)
    through_first_guardrail = through_first_guardrail._replace(
        through_growth_respiration=through_growth,
    )
    through_second_guardrail = through_second_guardrail._replace(
        through_first_guardrail=through_first_guardrail,
    )
    through_soil_n = through_soil_n._replace(
        through_second_guardrail=through_second_guardrail,
    )

    transport = through_transport.litter_vertical_transport._replace(
        vertical_transport=(
            through_transport.litter_vertical_transport.vertical_transport._replace(
                maximum_annual_thaw_depth_column=_real(column.maximum_thaw_depth),
                previous_year_maximum_annual_thaw_depth_column=(
                    _real(column.previous_maximum_thaw_depth)
                ),
            )
        ),
    )
    through_transport = through_transport._replace(
        through_soil_n_state_update=through_soil_n,
        litter_vertical_transport=transport,
    )
    through_gap = through_gap._replace(
        through_litter_vertical_transport=through_transport,
        gap_mortality=through_gap.gap_mortality._replace(
            gap_mortality=through_gap.gap_mortality.gap_mortality._replace(
                patch_weight_on_column=patch_weight_on_column,
                patch_active=patch_active,
            ),
        ),
    )
    through_update2 = through_update2._replace(through_gap_mortality=through_gap)
    through_update2h = through_update2h._replace(
        through_state_update2=through_update2,
    )

    wood_products = through_wood_products.wood_products._replace(
        wood_products=through_wood_products.wood_products.wood_products._replace(
            patch_active=patch_active,
            patch_weight_on_gridcell=patch_weight_on_gridcell,
        ),
    )
    through_wood_products = through_wood_products._replace(
        through_state_update2h_precision=through_update2h,
        wood_products=wood_products,
    )

    fire_area = through_fire_area.fire_area._replace(
        topology=through_fire_area.fire_area.topology._replace(
            patch_weight_on_column=patch_weight_on_column,
            patch_active=patch_active,
        ),
        forcing=through_fire_area.fire_area.forcing._replace(
            relative_humidity_gridcell=_real(grid.relative_humidity),
            wind_speed_gridcell=_real(grid.wind_speed),
            temperature_downscaled_column=_real(column.downscaled_temperature),
            rain_downscaled_column=_real(column.downscaled_rain),
            snow_downscaled_column=_real(column.downscaled_snow),
            precipitation_60day_patch=_real(patch.precipitation_60day),
            precipitation_10day_patch=_real(patch.precipitation_10day),
            relative_humidity_30day_patch=_real(patch.relative_humidity_30day),
            root_zone_wetness_patch=_real(patch.root_zone_wetness),
            saturated_area_fraction_column=_real(column.saturated_area_fraction),
            topsoil_water_fraction_column=_real(column.topsoil_water_fraction),
            total_litter_carbon_column=_real(column.total_litter_carbon),
            total_vegetation_carbon_column=_real(column.total_vegetation_carbon),
            soil_temperature_17cm_column=_real(column.soil_temperature_17cm),
        ),
    )
    through_fire_area = through_fire_area._replace(
        through_wood_products=through_wood_products,
        fire_area=fire_area,
    )

    fire_flux = through_fire_flux.fire_flux._replace(
        fire_flux=through_fire_flux.fire_flux.fire_flux._replace(
            patch_weight_on_column=patch_weight_on_column,
            patch_active=patch_active,
            total_soil_organic_carbon_column=(_real(column.total_soil_organic_carbon)),
        ),
    )
    through_fire_flux = through_fire_flux._replace(
        through_fire_area=through_fire_area,
        fire_flux=fire_flux,
    )
    state_update3 = state_update3._replace(through_fire_flux=through_fire_flux)

    annual = inputs.annual_update._replace(
        annual_update=inputs.annual_update.annual_update._replace(
            patch_weight_on_column=patch_weight_on_column,
            patch_active=patch_active,
        ),
    )
    return inputs._replace(
        through_state_update3_precision=state_update3,
        annual_update=annual,
    )


def _vegetation_field(
    state: CnPrecisionControlState,
    species: str,
    native_name: str,
) -> jnp.ndarray:
    return get_cn_precision_control_state_field(
        state,
        f"vegetation_{species}.{native_name}_patch",
    )


def _vegetation_fields(
    state: CnPrecisionControlState,
    species: str,
    native_names: tuple[str, ...],
) -> jnp.ndarray:
    return jnp.stack(
        tuple(_vegetation_field(state, species, name) for name in native_names),
        axis=1,
    )


@jax.jit
def _persistent_vegetation_fields(state: CnPrecisionControlState):

    return (
        _vegetation_fields(state, "carbon", CARBON_STORAGE_POOL_NAMES),
        _vegetation_fields(state, "carbon", CARBON_TRANSFER_POOL_NAMES),
        _vegetation_fields(state, "nitrogen", NITROGEN_STORAGE_POOL_NAMES),
        _vegetation_fields(state, "nitrogen", NITROGEN_TRANSFER_POOL_NAMES),
        {
            name: _vegetation_field(state, "carbon", name)
            for name in (
                "leafc", "frootc", "xsmrpool", "livestemc", "livecrootc",
                "leafc_storage", "grainc", "cropseedc_deficit",
            )
        },
        {
            name: _vegetation_field(state, "nitrogen", name)
            for name in (
                "frootn", "livestemn", "livecrootn", "grainn", "retransn",
                "leafn", "leafn_storage", "npool", "cropseedn_deficit",
            )
        },
    )


def inject_ecosystem_dynamics_persistent_state(
    inputs: EcosystemDynamicsCompleteInputs,
    state: Any,
    state_summaries: Any,
    peak_leaf_area_index_patch: jnp.ndarray,
) -> EcosystemDynamicsCompleteInputs:

    update3_state = state.state_update3_precision
    fire_flux_state = update3_state.fire_flux
    fire_area_state = fire_flux_state.fire_area
    wood_state = fire_area_state.wood_products
    update2h_state = wood_state.state_update2h_precision
    update2_state = update2h_state.state_update2
    gap_state = update2_state.gap_mortality
    transport_state = gap_state.litter_vertical_transport
    soil_n_state = transport_state.soil_n_state_update
    guarded_state = soil_n_state.second_guardrail.first_guardrail
    vegetation = guarded_state.vegetation_carbon_nitrogen
    soil = transport_state.vertical_transport
    mineral = soil_n_state.soil_nitrogen
    annual = state.annual

    (
        carbon_storage, carbon_transfer, nitrogen_storage, nitrogen_transfer,
        carbon_fields, nitrogen_fields,
    ) = _persistent_vegetation_fields(vegetation)

    def carbon(name: str) -> jnp.ndarray:
        return carbon_fields[name]

    def nitrogen(name: str) -> jnp.ndarray:
        return nitrogen_fields[name]

    update3 = inputs.through_state_update3_precision
    through_fire_flux = update3.through_fire_flux
    through_fire_area = through_fire_flux.through_fire_area
    through_wood_products = through_fire_area.through_wood_products
    through_update2h = through_wood_products.through_state_update2h_precision
    through_update2 = through_update2h.through_state_update2
    through_gap = through_update2.through_gap_mortality
    through_transport = through_gap.through_litter_vertical_transport
    through_soil_n = through_transport.through_soil_n_state_update
    through_second_guardrail = through_soil_n.through_second_guardrail
    through_first_guardrail = through_second_guardrail.through_first_guardrail
    through_growth = through_first_guardrail.through_growth_respiration
    through_litter = through_growth.through_litter
    pre_decomposition = through_litter.pre_decomposition
    prefix = pre_decomposition.prefix
    competition = pre_decomposition.competition_decomposition

    maintenance = prefix.maintenance_respiration._replace(
        fine_root_nitrogen_patch=nitrogen("frootn"),
        live_stem_nitrogen_patch=nitrogen("livestemn"),
        live_coarse_root_nitrogen_patch=nitrogen("livecrootn"),
        grain_nitrogen_patch=nitrogen("grainn"),
    )
    potential = prefix.soil_biogeochem_potential._replace(
        decomposition_carbon_pool_column_level_pool=(soil.carbon_pool_column_level_pool),
        decomposition_nitrogen_pool_column_level_pool=(soil.nitrogen_pool_column_level_pool),
    )
    nitrif_denitrif = prefix.soil_biogeochem_nitrif_denitrif._replace(
        mineral_ammonium_column_level=mineral.ammonium_column_level,
        mineral_nitrate_column_level=mineral.nitrate_column_level,
    )
    evergreen = prefix.evergreen_phenology._replace(
        carbon_storage_patch_pool=carbon_storage,
        carbon_transfer_patch_pool=carbon_transfer,
        nitrogen_storage_patch_pool=nitrogen_storage,
        nitrogen_transfer_patch_pool=nitrogen_transfer,
    )
    seasonal = prefix.seasonal_deciduous_phenology._replace(
        carbon_storage_patch_pool=carbon_storage,
        nitrogen_storage_patch_pool=nitrogen_storage,
        annual_average_temperature_2m_patch=annual.annual_average_air_temperature_patch,
    )
    stress = prefix.stress_deciduous_phenology._replace(
        leaf_carbon_patch=carbon("leafc"),
        fine_root_carbon_patch=carbon("frootc"),
        carbon_storage_patch_pool=carbon_storage,
        nitrogen_storage_patch_pool=nitrogen_storage,
        annual_average_temperature_2m_patch=annual.annual_average_air_temperature_patch,
    )
    crop = prefix.crop_phenology._replace(
        peak_leaf_area_index_patch=jnp.asarray(
            peak_leaf_area_index_patch,
            dtype=jnp.int32,
        ),
        total_vegetation_carbon_patch=jnp.asarray(
            state_summaries.total_vegetation_carbon_patch,
            dtype=jnp.float64,
        ),
    )
    plant_level_count = (
        prefix.plant_nutrient_demand.soil_mineral_nitrogen_column_level.shape[1]
    )
    nutrient_demand = prefix.plant_nutrient_demand._replace(
        peak_leaf_area_index_patch=jnp.asarray(
            peak_leaf_area_index_patch,
            dtype=jnp.int32,
        ),
        temporary_photosynthate_carbon_pool_patch=carbon("xsmrpool"),
        leaf_carbon_patch=carbon("leafc"),
        fine_root_carbon_patch=carbon("frootc"),
        live_stem_carbon_patch=carbon("livestemc"),
        live_coarse_root_carbon_patch=carbon("livecrootc"),
        retranslocated_nitrogen_pool_patch=nitrogen("retransn"),
        annual_net_primary_productivity_patch=annual.annual_sum_npp_patch,
        leaf_nitrogen_patch=nitrogen("leafn"),
        fine_root_nitrogen_patch=nitrogen("frootn"),
        live_stem_nitrogen_patch=nitrogen("livestemn"),
        annual_potential_gpp_patch=annual.annual_sum_potential_gpp_patch,
        annual_maximum_retranslocated_nitrogen_patch=(
            annual.annual_maximum_retranslocated_nitrogen_patch
        ),
        soil_mineral_nitrogen_column_level=(
            mineral.mineral_nitrogen_column_level[:, :plant_level_count]
        ),
    )
    prefix = prefix._replace(
        maintenance_respiration=maintenance,
        soil_biogeochem_potential=potential,
        soil_biogeochem_nitrif_denitrif=nitrif_denitrif,
        evergreen_phenology=evergreen,
        seasonal_deciduous_phenology=seasonal,
        stress_deciduous_phenology=stress,
        crop_phenology=crop,
        plant_nutrient_demand=nutrient_demand,
    )

    fun = competition.fun_allocation._replace(
        leaf_carbon_patch=carbon("leafc"),
        leaf_carbon_storage_patch=carbon("leafc_storage"),
        fine_root_carbon_patch=carbon("frootc"),
        leaf_nitrogen_patch=nitrogen("leafn"),
        leaf_nitrogen_storage_patch=nitrogen("leafn_storage"),
        fine_root_nitrogen_patch=nitrogen("frootn"),
        live_stem_nitrogen_patch=nitrogen("livestemn"),
        live_coarse_root_nitrogen_patch=nitrogen("livecrootn"),
    )
    soil_competition = competition.soil_biogeochem_competition._replace(
        mineral_nitrogen_column_level=mineral.mineral_nitrogen_column_level,
        mineral_ammonium_column_level=mineral.ammonium_column_level,
        mineral_nitrate_column_level=mineral.nitrate_column_level,
    )
    plant_competition = competition.plant_nutrient_competition._replace(
        annual_net_primary_productivity_patch=annual.annual_sum_npp_patch,
        plant_nitrogen_pool_patch=nitrogen("npool"),
        leaf_carbon_storage_patch=carbon("leafc_storage"),
        leaf_nitrogen_storage_patch=nitrogen("leafn_storage"),
    )
    decomposition = competition.soil_biogeochem_decomp._replace(
        decomposition_carbon_pool_column_level_pool=(soil.carbon_pool_column_level_pool),
        decomposition_nitrogen_pool_column_level_pool=(soil.nitrogen_pool_column_level_pool),
    )
    competition = competition._replace(
        fun_allocation=fun,
        soil_biogeochem_competition=soil_competition,
        plant_nutrient_competition=plant_competition,
        soil_biogeochem_decomp=decomposition,
    )
    pre_decomposition = pre_decomposition._replace(
        prefix=prefix,
        competition_decomposition=competition,
    )

    phase_two = through_litter.phase_two_litter
    phase_two = phase_two._replace(
        onset_growth=phase_two.onset_growth._replace(
            carbon_transfer_patch_pool=carbon_transfer,
            nitrogen_transfer_patch_pool=nitrogen_transfer,
        ),
        offset_litterfall=phase_two.offset_litterfall._replace(
            leaf_carbon_patch=carbon("leafc"),
            fine_root_carbon_patch=carbon("frootc"),
            grain_carbon_patch=carbon("grainc"),
            crop_seed_carbon_deficit_patch=carbon("cropseedc_deficit"),
            live_stem_carbon_patch=carbon("livestemc"),
            leaf_nitrogen_patch=nitrogen("leafn"),
            fine_root_nitrogen_patch=nitrogen("frootn"),
            grain_nitrogen_patch=nitrogen("grainn"),
            crop_seed_nitrogen_deficit_patch=nitrogen("cropseedn_deficit"),
            live_stem_nitrogen_patch=nitrogen("livestemn"),
        ),
        background_litterfall=phase_two.background_litterfall._replace(
            leaf_carbon_patch=carbon("leafc"),
            fine_root_carbon_patch=carbon("frootc"),
            leaf_nitrogen_patch=nitrogen("leafn"),
            fine_root_nitrogen_patch=nitrogen("frootn"),
        ),
        livewood_turnover=phase_two.livewood_turnover._replace(
            live_stem_carbon_patch=carbon("livestemc"),
            live_coarse_root_carbon_patch=carbon("livecrootc"),
            live_stem_nitrogen_patch=nitrogen("livestemn"),
            live_coarse_root_nitrogen_patch=nitrogen("livecrootn"),
        ),
    )
    through_litter = through_litter._replace(
        pre_decomposition=pre_decomposition,
        phase_two_litter=phase_two,
    )
    through_growth = through_growth._replace(through_litter=through_litter)
    through_first_guardrail = through_first_guardrail._replace(
        through_growth_respiration=through_growth,
    )
    through_second_guardrail = through_second_guardrail._replace(
        through_first_guardrail=through_first_guardrail,
    )
    through_soil_n = through_soil_n._replace(
        through_second_guardrail=through_second_guardrail,
    )
    through_transport = through_transport._replace(
        through_soil_n_state_update=through_soil_n,
    )
    through_gap = through_gap._replace(
        through_litter_vertical_transport=through_transport,
        gap_mortality=through_gap.gap_mortality._replace(
            gap_mortality=through_gap.gap_mortality.gap_mortality._replace(
                vegetation_state=vegetation,
            ),
        ),
    )
    through_update2 = through_update2._replace(through_gap_mortality=through_gap)
    through_update2h = through_update2h._replace(
        through_state_update2=through_update2,
    )
    through_wood_products = through_wood_products._replace(
        through_state_update2h_precision=through_update2h,
    )

    fire_area = through_fire_area.fire_area._replace(
        forcing=through_fire_area.fire_area.forcing._replace(
            decomposition_carbon_pool_column_level_pool=(soil.carbon_pool_column_level_pool),
        ),
    )
    through_fire_area = through_fire_area._replace(
        through_wood_products=through_wood_products,
        fire_area=fire_area,
    )
    fire_flux = through_fire_flux.fire_flux._replace(
        fire_flux=through_fire_flux.fire_flux.fire_flux._replace(
            vegetation_state=vegetation,
            decomposition_carbon_pool_column_level_pool=(soil.carbon_pool_column_level_pool),
            decomposition_nitrogen_pool_column_level_pool=(soil.nitrogen_pool_column_level_pool),
        ),
    )
    through_fire_flux = through_fire_flux._replace(
        through_fire_area=through_fire_area,
        fire_flux=fire_flux,
    )
    update3 = update3._replace(through_fire_flux=through_fire_flux)
    return inputs._replace(through_state_update3_precision=update3)


def inject_ecosystem_dynamics_fast_trace_inputs(
    inputs: EcosystemDynamicsCompleteInputs,
    native: EcosystemDynamicsFastTraceNativeData,
) -> EcosystemDynamicsCompleteInputs:

    return inject_ecosystem_dynamics_fast_inputs(inputs, native.snapshot)


__all__ = [
    "EcosystemDynamicsFastInputFields",
    "inject_ecosystem_dynamics_fast_inputs",
    "inject_ecosystem_dynamics_fast_trace_inputs",
    "inject_ecosystem_dynamics_persistent_state",
]
