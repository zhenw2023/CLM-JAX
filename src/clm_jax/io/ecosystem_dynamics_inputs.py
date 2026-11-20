
from __future__ import annotations

import jax
import jax.numpy as jnp

from clm_jax.forcing.bgc import interpolate_bgc_external_forcing
from clm_jax.io.cn_fun_allocation import assemble_fun_competition_coupling_inputs
from clm_jax.io.decomposition_rates import assemble_decomposition_rate_inputs
from clm_jax.io.ecosystem_dynamics_fast_inputs import (
    EcosystemDynamicsFastInputFields,
    inject_ecosystem_dynamics_fast_inputs,
    inject_ecosystem_dynamics_persistent_state,
)
from clm_jax.io.ecosystem_dynamics_state import EcosystemDynamicsNativeChildren
from clm_jax.io.maintenance_respiration import (
    MaintenanceRespirationTemperatureState,
    assemble_maintenance_respiration_inputs,
)
from clm_jax.io.native_templates import native_zeros
from clm_jax.io.soil_biogeochem_nitrif_denitrif import (
    assemble_soil_biogeochem_nitrif_denitrif_inputs,
)
from clm_jax.io.soil_biogeochem_potential import (
    assemble_soil_biogeochem_potential_inputs,
)
from clm_jax.io.soil_biogeochem_vertical_profile import (
    assemble_soil_biogeochem_vertical_profile_inputs,
)
from clm_jax.io.stress_deciduous_phenology import (
    assemble_stress_deciduous_phenology_inputs,
)
from clm_jax.physics.cn_fire_area import CnFireAreaForcing
from clm_jax.physics.cn_fire_fluxes import CnFireFluxInputs
from clm_jax.physics.cn_fun_allocation import FunAllocationInputs
from clm_jax.physics.cn_gap_mortality import CnGapMortalityInputs
from clm_jax.physics.cn_zero import CnZeroInputs
from clm_jax.physics.crop import CropYearInputs
from clm_jax.physics.ecosystem_dynamics import (
    EcosystemDynamicsAnnualUpdateInputs,
    EcosystemDynamicsCnStateUpdate1Inputs,
    EcosystemDynamicsCnStateUpdate2hPrecisionInputs,
    EcosystemDynamicsCnStateUpdate2Inputs,
    EcosystemDynamicsCnStateUpdate3PrecisionInputs,
    EcosystemDynamicsCompetitionDecompositionInputs,
    EcosystemDynamicsCompleteInputs,
    EcosystemDynamicsCStateUpdate0PrecisionInputs,
    EcosystemDynamicsFireAreaInputs,
    EcosystemDynamicsFireFluxInputs,
    EcosystemDynamicsGapMortalityInputs,
    EcosystemDynamicsLitterVerticalTransportInputs,
    EcosystemDynamicsPhaseTwoLitterInputs,
    EcosystemDynamicsPreDrainagePrefixInputs,
    EcosystemDynamicsPreDrainageThroughDecompositionInputs,
    EcosystemDynamicsSoilNStateUpdate1Inputs,
    EcosystemDynamicsThroughCnStateUpdate2hPrecisionInputs,
    EcosystemDynamicsThroughCnStateUpdate2Inputs,
    EcosystemDynamicsThroughCnStateUpdate3PrecisionInputs,
    EcosystemDynamicsThroughFireAreaInputs,
    EcosystemDynamicsThroughFireFluxInputs,
    EcosystemDynamicsThroughFirstGuardrailInputs,
    EcosystemDynamicsThroughGapMortalityInputs,
    EcosystemDynamicsThroughGrowthRespirationInputs,
    EcosystemDynamicsThroughLitterToColumnInputs,
    EcosystemDynamicsThroughLitterVerticalTransportInputs,
    EcosystemDynamicsThroughSecondGuardrailInputs,
    EcosystemDynamicsThroughSoilNStateUpdate1Inputs,
    EcosystemDynamicsThroughWoodProductsInputs,
    EcosystemDynamicsWoodProductsInputs,
)
from clm_jax.physics.litter_to_column import LitterToColumnInputs
from clm_jax.physics.nitrogen_dynamics import NitrogenDepositionInputs
from clm_jax.physics.plant_nutrient_competition import (
    PlantNutrientCompetitionInputs,
)
from clm_jax.physics.plant_nutrient_demand import PlantNutrientDemandInputs
from clm_jax.physics.plant_nutrient_demand_aggregation import (
    PlantNutrientDemandAggregationInputs,
)
from clm_jax.physics.soil_biogeochem_competition import (
    SoilBiogeochemCompetitionInputs,
)
from clm_jax.physics.soil_biogeochem_decomp import SoilBiogeochemDecompInputs
from clm_jax.physics.soil_biogeochem_litter_vertical_transport import (
    SoilBiogeochemLitterVerticalTransportInputs,
)
from clm_jax.physics.soil_biogeochem_n_state_update1 import (
    SoilBiogeochemNStateUpdate1Inputs,
)


def _zeros_patch(patch_count: int) -> jnp.ndarray:
    return native_zeros((patch_count,), dtype=jnp.float64)


def _zeros_column_level(column_count: int, level_count: int = 25) -> jnp.ndarray:
    return native_zeros((column_count, level_count), dtype=jnp.float64)


def _plant_nutrient_demand_inputs(
    native,
    snapshot,
    zero_column_level: jnp.ndarray,
) -> PlantNutrientDemandInputs:
    persistent = native.persistent_inputs
    if bool(native.parameters.use_carbon_13) or bool(native.parameters.use_carbon_14):
        raise NotImplementedError(
            "the selected case enables isotope photosynthesis not present in the fast trace"
        )
    zero_patch = _zeros_patch(snapshot.patch.active.shape[0])
    soil_level_count = persistent.soil_mineral_nitrogen_column_level.shape[1]
    return PlantNutrientDemandInputs(
        soil_patch=persistent.soil_patch,
        patch_column_index=persistent.patch_column_index,
        vegetation_type_patch=persistent.vegetation_type_patch,
        sunlit_photosynthesis_patch=jnp.asarray(
            snapshot.patch.sunlit_photosynthesis,
            dtype=jnp.float64,
        ),
        shaded_photosynthesis_patch=jnp.asarray(
            snapshot.patch.shaded_photosynthesis,
            dtype=jnp.float64,
        ),
        c13_sunlit_photosynthesis_patch=zero_patch,
        c13_shaded_photosynthesis_patch=zero_patch,
        c14_sunlit_photosynthesis_patch=zero_patch,
        c14_shaded_photosynthesis_patch=zero_patch,
        sunlit_leaf_area_index_patch=jnp.asarray(
            snapshot.patch.sunlit_leaf_area,
            dtype=jnp.float64,
        ),
        shaded_leaf_area_index_patch=jnp.asarray(
            snapshot.patch.shaded_leaf_area,
            dtype=jnp.float64,
        ),
        growing_degree_days_since_planting_patch=(
            persistent.growing_degree_days_since_planting_patch
        ),
        growing_degree_days_topsoil_patch=(persistent.growing_degree_days_topsoil_patch),
        crop_live_patch=persistent.crop_live_patch,
        growing_degree_days_to_maturity_patch=(persistent.growing_degree_days_to_maturity_patch),
        heat_units_to_leaf_emergence_patch=(persistent.heat_units_to_leaf_emergence_patch),
        heat_units_to_grain_fill_patch=(persistent.heat_units_to_grain_fill_patch),
        peak_leaf_area_index_patch=persistent.peak_leaf_area_index_patch,
        temporary_photosynthate_carbon_pool_patch=(
            persistent.temporary_photosynthate_carbon_pool_patch
        ),
        leaf_carbon_patch=persistent.leaf_carbon_patch,
        fine_root_carbon_patch=persistent.fine_root_carbon_patch,
        live_stem_carbon_patch=persistent.live_stem_carbon_patch,
        live_coarse_root_carbon_patch=persistent.live_coarse_root_carbon_patch,
        retranslocated_nitrogen_pool_patch=(persistent.retranslocated_nitrogen_pool_patch),
        annual_net_primary_productivity_patch=(persistent.annual_net_primary_productivity_patch),
        leaf_nitrogen_patch=persistent.leaf_nitrogen_patch,
        fine_root_nitrogen_patch=persistent.fine_root_nitrogen_patch,
        live_stem_nitrogen_patch=persistent.live_stem_nitrogen_patch,
        annual_potential_gpp_patch=persistent.annual_potential_gpp_patch,
        annual_maximum_retranslocated_nitrogen_patch=(
            persistent.annual_maximum_retranslocated_nitrogen_patch
        ),
        soil_mineral_nitrogen_column_level=(persistent.soil_mineral_nitrogen_column_level),
        decomposition_temperature_scalar_column_level=(zero_column_level[:, :soil_level_count]),
        transpiration_wetness_factor_patch=jnp.asarray(
            snapshot.patch.transpiration_wetness,
            dtype=jnp.float64,
        ),
    )


def _fun_allocation_inputs(native, snapshot) -> FunAllocationInputs:
    persistent = native.persistent_inputs
    patch_count = native.topology.patch.gridcell.shape[0]
    column_count = native.topology.column.gridcell.shape[0]
    zero_patch = _zeros_patch(patch_count)
    zero_level = _zeros_column_level(column_count)
    return FunAllocationInputs(
        soil_patch=native.soil_patch,
        patch_column_index=native.topology.patch.column,
        vegetation_type_patch=native.topology.patch.vegetation_type,
        available_carbon_patch=zero_patch,
        plant_nitrogen_demand_patch=zero_patch,
        leaf_carbon_patch=persistent.leaf_carbon_patch,
        leaf_carbon_storage_patch=persistent.leaf_carbon_storage_patch,
        fine_root_carbon_patch=persistent.fine_root_carbon_patch,
        leaf_nitrogen_patch=persistent.leaf_nitrogen_patch,
        leaf_nitrogen_storage_patch=persistent.leaf_nitrogen_storage_patch,
        leaf_nitrogen_storage_to_transfer_patch=zero_patch,
        fine_root_nitrogen_patch=persistent.fine_root_nitrogen_patch,
        live_stem_nitrogen_patch=persistent.live_stem_nitrogen_patch,
        live_coarse_root_nitrogen_patch=(persistent.live_coarse_root_nitrogen_patch),
        leaf_carbon_to_litter_fun_patch=zero_patch,
        leaf_carbon_storage_to_transfer_patch=zero_patch,
        carbon_allometry_patch=zero_patch,
        nitrogen_allometry_patch=zero_patch,
        onset_flag_patch=zero_patch,
        offset_flag_patch=zero_patch,
        root_carbon_fraction_patch_level=jnp.asarray(
            snapshot.patch.carbon_root_fraction,
            dtype=jnp.float64,
        ),
        soil_temperature_column_level=jnp.asarray(
            snapshot.column.soil_temperature,
            dtype=jnp.float64,
        ),
        soil_liquid_water_column_level=jnp.asarray(
            snapshot.column.soil_liquid_water,
            dtype=jnp.float64,
        ),
        maximum_nitrate_uptake_column_level=zero_level,
        maximum_ammonium_uptake_column_level=zero_level,
    )


def _competition_inputs(children, snapshot) -> EcosystemDynamicsCompetitionDecompositionInputs:
    column_count = snapshot.column.active.shape[0]
    patch_count = snapshot.patch.active.shape[0]
    zero_patch = _zeros_patch(patch_count)
    zero_level = _zeros_column_level(column_count)
    soil = children.soil_competition
    soil_persistent = soil.persistent
    plant = children.plant_competition
    plant_persistent = plant.persistent_inputs
    decomposition = children.soil_decomposition
    return EcosystemDynamicsCompetitionDecompositionInputs(
        fun_competition_coupling=assemble_fun_competition_coupling_inputs(children.allocation),
        fun_allocation=_fun_allocation_inputs(children.allocation, snapshot),
        soil_biogeochem_competition=SoilBiogeochemCompetitionInputs(
            soil_column=soil.soil_column,
            plant_nitrogen_demand_column=native_zeros((column_count,), dtype=jnp.float64),
            nitrogen_fixation_profile_column_level=zero_level,
            mineral_nitrogen_column_level=(soil_persistent.mineral_nitrogen_column_level),
            mineral_ammonium_column_level=(soil_persistent.mineral_ammonium_column_level),
            mineral_nitrate_column_level=soil_persistent.mineral_nitrate_column_level,
            potential_nitrification_column_level=zero_level,
            potential_denitrification_column_level=zero_level,
            potential_immobilization_column_level=zero_level,
            n2_to_n2o_ratio_denitrification_column_level=zero_level,
        ),
        plant_nutrient_competition=PlantNutrientCompetitionInputs(
            soil_patch=plant_persistent.soil_patch,
            vegetation_type_patch=plant_persistent.vegetation_type_patch,
            crop_live_patch=plant_persistent.crop_live_patch,
            annual_net_primary_productivity_patch=(
                plant_persistent.annual_net_primary_productivity_patch
            ),
            leaf_allocation_patch=zero_patch,
            stem_allocation_patch=zero_patch,
            carbon_allometry_patch=zero_patch,
            root_allocation_patch=zero_patch,
            reproductive_allocation_patch=zero_patch,
            plant_nitrogen_pool_patch=plant_persistent.plant_nitrogen_pool_patch,
            fun_soil_nitrogen_uptake_patch=zero_patch,
            sunlit_leaf_area_index_patch=jnp.asarray(
                snapshot.patch.sunlit_leaf_area,
                dtype=jnp.float64,
            ),
            shaded_leaf_area_index_patch=jnp.asarray(
                snapshot.patch.shaded_leaf_area,
                dtype=jnp.float64,
            ),
            leaf_carbon_storage_patch=plant_persistent.leaf_carbon_storage_patch,
            leaf_nitrogen_storage_patch=(plant_persistent.leaf_nitrogen_storage_patch),
        ),
        soil_biogeochem_decomp=SoilBiogeochemDecompInputs(
            soil_column=decomposition.soil_column,
            decomposition_carbon_pool_column_level_pool=(
                decomposition.pools.decomposition_carbon_pool_column_level_pool
            ),
            decomposition_nitrogen_pool_column_level_pool=(
                decomposition.pools.decomposition_nitrogen_pool_column_level_pool
            ),
            fraction_potential_immobilization_column_level=(
                soil_persistent.fraction_potential_immobilization_column_level
            ),
            respiration_fraction_column_level_transition=(
                decomposition.respiration_fraction_column_level_transition
            ),
            water_scalar_column_level=zero_level,
        ),
        fun_allocation_parameters=children.allocation.parameters,
        fun_competition_coupling_parameters=(children.allocation.coupling_parameters),
        soil_biogeochem_competition_parameters=soil.parameters,
        plant_nutrient_competition_parameters=plant.parameters,
        soil_biogeochem_decomp_parameters=decomposition.parameters,
    )


def _prefix_inputs(children, snapshot) -> EcosystemDynamicsPreDrainagePrefixInputs:
    column_count = snapshot.column.active.shape[0]
    zero_level = _zeros_column_level(column_count)
    leaf_respiration = jnp.stack(
        (
            jnp.asarray(snapshot.patch.sunlit_leaf_respiration, dtype=jnp.float64),
            jnp.asarray(snapshot.patch.shaded_leaf_respiration, dtype=jnp.float64),
        ),
        axis=1,
    )
    maintenance_temperature = MaintenanceRespirationTemperatureState(
        soil_temperature_column_level=jnp.asarray(
            snapshot.column.soil_temperature,
            dtype=jnp.float64,
        ),
        reference_temperature_2m_patch=jnp.asarray(
            snapshot.patch.reference_temperature_2m,
            dtype=jnp.float64,
        ),
        ten_day_temperature_patch=jnp.asarray(
            snapshot.patch.temperature_10day,
            dtype=jnp.float64,
        ),
    )
    maintenance = assemble_maintenance_respiration_inputs(
        children.maintenance_respiration,
        snapshot.patch.vegetation_exposed,
        snapshot.patch.sunlit_leaf_area,
        snapshot.patch.shaded_leaf_area,
        leaf_respiration,
        temperature=maintenance_temperature,
    )._replace(
        carbon_root_fraction_patch_level=jnp.asarray(
            snapshot.patch.carbon_root_fraction,
            dtype=jnp.float64,
        )
    )
    decomposition_persistent = children.decomposition.persistent._replace(
        soil_temperature_column_level=jnp.asarray(
            snapshot.column.soil_temperature,
            dtype=jnp.float64,
        ),
        oxygen_stress_saturated_column_level=jnp.asarray(
            snapshot.column.oxygen_stress_saturated,
            dtype=jnp.float64,
        ),
        oxygen_stress_unsaturated_column_level=jnp.asarray(
            snapshot.column.oxygen_stress_unsaturated,
            dtype=jnp.float64,
        ),
        inundated_fraction_column=jnp.asarray(
            snapshot.column.inundated_fraction,
            dtype=jnp.float64,
        ),
    )
    decomposition = assemble_decomposition_rate_inputs(
        children.decomposition,
        jnp.asarray(snapshot.column.soil_water_potential, dtype=jnp.float64),
        persistent=decomposition_persistent,
        thaw_depth_index_column=jnp.asarray(
            snapshot.column.active_layer_index,
            dtype=jnp.int32,
        ),
    )
    vertical_profile = assemble_soil_biogeochem_vertical_profile_inputs(
        children.vertical_profile,
        active_layer_index_column=jnp.asarray(
            snapshot.column.active_layer_index,
            dtype=jnp.int32,
        ),
        carbon_root_fraction_patch_level=jnp.asarray(
            snapshot.patch.carbon_root_fraction,
            dtype=jnp.float64,
        ),
    )
    nitrif_native = children.nitrif_denitrif
    nitrif_persistent = nitrif_native.persistent._replace(
        soil_temperature_column_level=jnp.asarray(
            snapshot.column.soil_temperature,
            dtype=jnp.float64,
        ),
        inundated_fraction_column=jnp.asarray(
            snapshot.column.inundated_fraction,
            dtype=jnp.float64,
        ),
    )
    nitrif_denitrif = assemble_soil_biogeochem_nitrif_denitrif_inputs(
        nitrif_native,
        jnp.asarray(snapshot.column.soil_water_potential, dtype=jnp.float64),
        zero_level,
        zero_level,
        zero_level,
        persistent=nitrif_persistent,
    )
    demand = _plant_nutrient_demand_inputs(
        children.demand,
        snapshot,
        zero_level,
    )
    climate = children.phenology_climate
    model_step = children.fun_initialization.inputs.model_step
    if not isinstance(model_step,jax.core.Tracer):
        model_step = int(model_step)
    return EcosystemDynamicsPreDrainagePrefixInputs(
        crop_year=CropYearInputs(
            prognostic_crop_patch=climate.inputs.prognostic_crop_patch,
            current_date_yyyymmdd=climate.inputs.current_date_yyyymmdd,
            current_time_of_day_seconds=(climate.inputs.current_time_of_day_seconds),
            first_model_step=jnp.asarray(model_step == 0, dtype=jnp.bool_),
        ),
        cn_zero=CnZeroInputs(
            soil_patch=children.vertical_profile.soil_patch,
            soil_column=children.potential.soil_column,
        ),
        nitrogen_deposition=NitrogenDepositionInputs(
            forcing_ndep_gridcell=jnp.asarray(
                snapshot.gridcell.nitrogen_deposition,
                dtype=jnp.float64,
            ),
            column_gridcell_index=children.potential.topology.column.gridcell,
        ),
        free_living_fixation=children.free_living_fixation.inputs._replace(
            annual_average_evapotranspiration_column=jnp.asarray(
                snapshot.column.annual_evapotranspiration,
                dtype=jnp.float64,
            )
        ),
        nitrogen_fertilizer=children.nitrogen_fertilizer.inputs,
        maintenance_respiration=maintenance,
        decomposition_rate=decomposition,
        soil_biogeochem_potential=assemble_soil_biogeochem_potential_inputs(children.potential),
        soil_biogeochem_vertical_profile=vertical_profile,
        soil_biogeochem_nitrif_denitrif=nitrif_denitrif,
        phenology_climate=climate.inputs,
        evergreen_phenology=children.evergreen.inputs,
        seasonal_deciduous_phenology=children.seasonal.inputs,
        stress_deciduous_phenology=assemble_stress_deciduous_phenology_inputs(
            children.stress,
            jnp.asarray(
                snapshot.column.soil_water_potential[:, 2],
                dtype=jnp.float64,
            ),
        ),
        crop_phenology=children.crop.inputs,
        crop_phenology_parameters=children.crop.parameters,
        fun_initialization=children.fun_initialization.inputs,
        plant_nutrient_demand=demand,
        plant_nutrient_demand_aggregation=PlantNutrientDemandAggregationInputs(
            patch_column_index=children.potential.topology.patch.column,
            patch_weight_on_column=jnp.asarray(
                snapshot.patch.weight_on_column,
                dtype=jnp.float64,
            ),
            patch_active=jnp.asarray(snapshot.patch.active, dtype=jnp.bool_),
            soil_column=children.potential.soil_column,
        ),
        fun_initialization_parameters=children.fun_initialization.parameters,
        plant_nutrient_demand_parameters=children.demand.parameters,
        free_living_fixation_parameters=children.free_living_fixation.parameters,
        nitrogen_fertilizer_parameters=children.nitrogen_fertilizer.parameters,
        maintenance_respiration_parameters=(children.maintenance_respiration.parameters),
        decomposition_rate_parameters=children.decomposition.parameters,
        soil_biogeochem_potential_parameters=children.potential.parameters,
        soil_biogeochem_vertical_profile_parameters=(children.vertical_profile.parameters),
        soil_biogeochem_nitrif_denitrif_parameters=nitrif_native.parameters,
        phenology_climate_parameters=climate.parameters,
        evergreen_phenology_parameters=children.evergreen.parameters,
        seasonal_deciduous_phenology_parameters=children.seasonal.parameters,
        stress_deciduous_phenology_parameters=children.stress.parameters,
    )


def assemble_ecosystem_dynamics_complete_inputs(
    children: EcosystemDynamicsNativeChildren,
    snapshot: EcosystemDynamicsFastInputFields,
    *,
    external_forcing_state=None,
    state=None,
    state_summaries=None,
    peak_leaf_area_index_patch=None,
) -> EcosystemDynamicsCompleteInputs:

    persistent_arguments = (
        state,
        state_summaries,
        peak_leaf_area_index_patch,
    )
    if any(value is not None for value in persistent_arguments) and any(
        value is None for value in persistent_arguments
    ):
        raise ValueError(
            "state, state_summaries and peak_leaf_area_index_patch must be supplied together"
        )

    patch_count = snapshot.patch.active.shape[0]
    column_count = snapshot.column.active.shape[0]
    profile = native_zeros((patch_count, 25), dtype=jnp.float64)
    zero_level = _zeros_column_level(column_count)

    prefix = _prefix_inputs(children, snapshot)
    pre_decomposition = EcosystemDynamicsPreDrainageThroughDecompositionInputs(
        prefix=prefix,
        competition_decomposition=_competition_inputs(children, snapshot),
    )
    litter_persistent = children.litter.persistent_inputs
    phase_two = EcosystemDynamicsPhaseTwoLitterInputs(
        onset_growth=children.onset.inputs,
        offset_litterfall=children.offset.inputs,
        background_litterfall=children.background.inputs,
        livewood_turnover=children.livewood.inputs,
        grain_to_product_pools=children.grain.inputs,
        litter_to_column=LitterToColumnInputs(
            soil_column=litter_persistent.soil_column,
            column_patch_indices=litter_persistent.column_patch_indices,
            vegetation_type_patch=litter_persistent.vegetation_type_patch,
            patch_weight_on_column=litter_persistent.patch_weight_on_column,
            patch_active=litter_persistent.patch_active,
            leaf_profile_patch_level=profile,
            fine_root_profile_patch_level=profile,
        ),
        offset_litterfall_parameters=children.offset.parameters,
        background_litterfall_parameters=children.background.parameters,
        livewood_turnover_parameters=children.livewood.parameters,
        litter_to_column_parameters=children.litter.parameters,
        onset_growth_parameters=children.onset.parameters,
        grain_to_product_pools_parameters=children.grain.parameters,
    )
    through_litter = EcosystemDynamicsThroughLitterToColumnInputs(
        pre_decomposition=pre_decomposition,
        phase_two_litter=phase_two,
    )
    through_growth = EcosystemDynamicsThroughGrowthRespirationInputs(
        through_litter=through_litter,
        growth_respiration=children.growth.inputs,
        growth_respiration_parameters=children.growth.parameters,
    )
    first_guardrail = EcosystemDynamicsThroughFirstGuardrailInputs(
        through_growth_respiration=through_growth,
        state_update_and_precision=EcosystemDynamicsCStateUpdate0PrecisionInputs(
            c_state_update0=children.carbon_update0.inputs,
            cn_precision_control=children.precision.inputs,
            c_state_update0_parameters=children.carbon_update0.parameters,
            cn_precision_control_parameters=children.precision.parameters,
        ),
    )
    second_guardrail = EcosystemDynamicsThroughSecondGuardrailInputs(
        through_first_guardrail=first_guardrail,
        state_update_and_precision=EcosystemDynamicsCnStateUpdate1Inputs(
            state_update=children.update1.inputs,
            state_update_parameters=children.update1.parameters,
            precision_parameters=children.update1.precision_parameters,
        ),
    )
    through_soil_n = EcosystemDynamicsThroughSoilNStateUpdate1Inputs(
        through_second_guardrail=second_guardrail,
        soil_n_state_update=EcosystemDynamicsSoilNStateUpdate1Inputs(
            state_update=SoilBiogeochemNStateUpdate1Inputs(
                soil_column=children.soil_nitrogen.soil_column,
                nitrogen_deposition_profile_column_level=zero_level,
                nitrogen_fixation_profile_column_level=zero_level,
            ),
            parameters=children.soil_nitrogen.parameters,
        ),
    )
    through_transport = EcosystemDynamicsThroughLitterVerticalTransportInputs(
        through_soil_n_state_update=through_soil_n,
        litter_vertical_transport=EcosystemDynamicsLitterVerticalTransportInputs(
            vertical_transport=SoilBiogeochemLitterVerticalTransportInputs(
                soil_column=children.transport.soil_column,
                bedrock_level_count_column=(children.transport.bedrock_level_count_column),
                maximum_annual_thaw_depth_column=jnp.asarray(
                    snapshot.column.maximum_thaw_depth,
                    dtype=jnp.float64,
                ),
                previous_year_maximum_annual_thaw_depth_column=jnp.asarray(
                    snapshot.column.previous_maximum_thaw_depth,
                    dtype=jnp.float64,
                ),
            ),
            parameters=children.transport.parameters,
        ),
    )
    mortality = children.mortality
    mortality_persistent = mortality.persistent_inputs
    gap = EcosystemDynamicsThroughGapMortalityInputs(
        through_litter_vertical_transport=through_transport,
        gap_mortality=EcosystemDynamicsGapMortalityInputs(
            gap_mortality=CnGapMortalityInputs(
                soil_patch=mortality_persistent.soil_patch,
                soil_column=mortality_persistent.soil_column,
                column_patch_indices=mortality_persistent.column_patch_indices,
                vegetation_type_patch=mortality_persistent.vegetation_type_patch,
                patch_weight_on_column=(mortality_persistent.patch_weight_on_column),
                patch_active=mortality_persistent.patch_active,
                vegetation_state=children.update1.restart_state.vegetation,
                growth_efficiency_patch=(mortality_persistent.growth_efficiency_patch),
                heat_stress_patch=mortality_persistent.heat_stress_patch,
                leaf_profile_patch_level=profile,
                fine_root_profile_patch_level=profile,
                coarse_root_profile_patch_level=profile,
                stem_profile_patch_level=profile,
            ),
            parameters=mortality.parameters,
        ),
    )
    through_update2 = EcosystemDynamicsThroughCnStateUpdate2Inputs(
        through_gap_mortality=gap,
        state_update=EcosystemDynamicsCnStateUpdate2Inputs(
            state_update=children.update2.inputs,
            parameters=children.update2.parameters,
        ),
    )
    through_update2h = EcosystemDynamicsThroughCnStateUpdate2hPrecisionInputs(
        through_state_update2=through_update2,
        state_update2h_precision=EcosystemDynamicsCnStateUpdate2hPrecisionInputs(
            state_update=children.update2.inputs,
            precision=children.update2.precision_inputs,
            state_update_parameters=children.update2.parameters,
            precision_parameters=children.update2.precision_parameters,
        ),
    )
    through_products = EcosystemDynamicsThroughWoodProductsInputs(
        through_state_update2h_precision=through_update2h,
        wood_products=EcosystemDynamicsWoodProductsInputs(
            wood_products=children.products.inputs,
            parameters=children.products.parameters,
        ),
    )

    external = external_forcing_state
    if external is None:
        external = interpolate_bgc_external_forcing(
            children.external_forcing.inputs,
            children.external_forcing.parameters,
        ).state
    fire_persistent = children.fire_area.persistent_inputs
    fire_forcing = CnFireAreaForcing(
        relative_humidity_gridcell=jnp.asarray(
            snapshot.gridcell.relative_humidity,
            dtype=jnp.float64,
        ),
        wind_speed_gridcell=jnp.asarray(snapshot.gridcell.wind_speed, dtype=jnp.float64),
        temperature_downscaled_column=jnp.asarray(
            snapshot.column.downscaled_temperature,
            dtype=jnp.float64,
        ),
        rain_downscaled_column=jnp.asarray(
            snapshot.column.downscaled_rain,
            dtype=jnp.float64,
        ),
        snow_downscaled_column=jnp.asarray(
            snapshot.column.downscaled_snow,
            dtype=jnp.float64,
        ),
        precipitation_60day_patch=jnp.asarray(
            snapshot.patch.precipitation_60day,
            dtype=jnp.float64,
        ),
        precipitation_10day_patch=jnp.asarray(
            snapshot.patch.precipitation_10day,
            dtype=jnp.float64,
        ),
        relative_humidity_30day_patch=jnp.asarray(
            snapshot.patch.relative_humidity_30day,
            dtype=jnp.float64,
        ),
        root_zone_wetness_patch=jnp.asarray(
            snapshot.patch.root_zone_wetness,
            dtype=jnp.float64,
        ),
        saturated_area_fraction_column=jnp.asarray(
            snapshot.column.saturated_area_fraction,
            dtype=jnp.float64,
        ),
        topsoil_water_fraction_column=jnp.asarray(
            snapshot.column.topsoil_water_fraction,
            dtype=jnp.float64,
        ),
        lightning_frequency_gridcell=external.lightning_frequency_gridcell,
        human_population_density_gridcell=(external.human_population_density_gridcell),
        total_litter_carbon_column=jnp.asarray(
            snapshot.column.total_litter_carbon,
            dtype=jnp.float64,
        ),
        total_vegetation_carbon_column=jnp.asarray(snapshot.column.total_vegetation_carbon),
        decomposition_carbon_pool_column_level_pool=(
            children.potential.pools.decomposition_carbon_pool_column_level_pool
        ),
        soil_temperature_17cm_column=jnp.asarray(
            snapshot.column.soil_temperature_17cm,
            dtype=jnp.float64,
        ),
        gdp_column=fire_persistent.gdp_column,
        peatland_fraction_column=fire_persistent.peatland_fraction_column,
        crop_fire_peak_month_column=fire_persistent.crop_fire_peak_month_column,
    )
    through_fire_area = EcosystemDynamicsThroughFireAreaInputs(
        through_wood_products=through_products,
        fire_area=EcosystemDynamicsFireAreaInputs(
            topology=fire_persistent.topology,
            forcing=fire_forcing,
            parameters=children.fire_area.parameters,
        ),
    )

    fire_flux_persistent = children.fire_flux.persistent_inputs
    fire_flux = CnFireFluxInputs(
        **fire_flux_persistent._asdict(),
        vegetation_state=children.update1.restart_state.vegetation,
        crop_fraction_column=native_zeros((column_count,), dtype=jnp.float64),
        total_burned_area_fraction_column=native_zeros((column_count,), dtype=jnp.float64),
        conversion_burned_area_fraction_column=native_zeros((column_count,), dtype=jnp.float64),
        conversion_fire_burned_area_fraction_column=native_zeros((column_count,), dtype=jnp.float64),
        crop_burned_area_fraction_column=native_zeros((column_count,), dtype=jnp.float64),
        peat_burned_area_fraction_column=native_zeros((column_count,), dtype=jnp.float64),
        tropical_evergreen_tree_fraction_column=native_zeros((column_count,), dtype=jnp.float64),
        tropical_deciduous_tree_fraction_column=native_zeros((column_count,), dtype=jnp.float64),
        decreased_tropical_tree_fraction_column=native_zeros((column_count,), dtype=jnp.float64),
        leaf_profile_patch_level=profile,
        fine_root_profile_patch_level=profile,
        coarse_root_profile_patch_level=profile,
        stem_profile_patch_level=profile,
        total_soil_organic_carbon_column=jnp.asarray(
            snapshot.column.total_soil_organic_carbon,
            dtype=jnp.float64,
        ),
        decomposition_carbon_pool_column_level_pool=(
            children.potential.pools.decomposition_carbon_pool_column_level_pool
        ),
        decomposition_nitrogen_pool_column_level_pool=(
            children.potential.pools.decomposition_nitrogen_pool_column_level_pool
        ),
    )
    through_fire_flux = EcosystemDynamicsThroughFireFluxInputs(
        through_fire_area=through_fire_area,
        fire_flux=EcosystemDynamicsFireFluxInputs(
            fire_flux=fire_flux,
            parameters=children.fire_flux.parameters,
        ),
    )
    through_update3 = EcosystemDynamicsThroughCnStateUpdate3PrecisionInputs(
        through_fire_flux=through_fire_flux,
        state_update3_precision=EcosystemDynamicsCnStateUpdate3PrecisionInputs(
            state_update=children.update3.inputs,
            precision=children.update3.precision_inputs,
            state_update_parameters=children.update3.parameters,
            precision_parameters=children.update3.precision_parameters,
        ),
    )
    inputs = EcosystemDynamicsCompleteInputs(
        through_state_update3_precision=through_update3,
        annual_update=EcosystemDynamicsAnnualUpdateInputs(
            annual_update=children.annual.inputs,
            parameters=children.annual.parameters,
        ),
    )
    inputs = inject_ecosystem_dynamics_fast_inputs(inputs, snapshot)
    if state is not None:
        inputs = inject_ecosystem_dynamics_persistent_state(
            inputs,
            state,
            state_summaries,
            peak_leaf_area_index_patch,
        )
    return inputs


__all__ = ["assemble_ecosystem_dynamics_complete_inputs"]
