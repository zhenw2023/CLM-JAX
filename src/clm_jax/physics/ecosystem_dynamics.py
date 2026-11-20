
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.physics.background_litterfall import (
    BackgroundLitterfallDiagnostics,
    BackgroundLitterfallInputs,
    BackgroundLitterfallParameters,
    BackgroundLitterfallState,
    background_litterfall,
)
from clm_jax.physics.c_state_update0 import (
    DEFAULT_C_STATE_UPDATE0_PARAMETERS,
    CStateUpdate0Diagnostics,
    CStateUpdate0Inputs,
    CStateUpdate0Parameters,
    CStateUpdate0State,
    c_state_update0,
)
from clm_jax.physics.cn_annual_update import (
    DEFAULT_CN_ANNUAL_PARAMETERS,
    CnAnnualDiagnostics,
    CnAnnualInputs,
    CnAnnualParameters,
    CnAnnualState,
    cn_annual_update,
)
from clm_jax.physics.cn_fire_area import (
    CnFireAreaDiagnostics,
    CnFireAreaForcing,
    CnFireAreaParameters,
    CnFireAreaState,
    CnFireAreaTopology,
    cn_fire_area_li2016_fixed_landcover,
)
from clm_jax.physics.cn_fire_fluxes import (
    CnFireFluxDiagnostics,
    CnFireFluxInputs,
    CnFireFluxParameters,
    CnFireFluxState,
    cn_fire_fluxes,
)
from clm_jax.physics.cn_fun_allocation import (
    FunAllocationDiagnostics,
    FunAllocationInputs,
    FunAllocationParameters,
    FunAllocationState,
)
from clm_jax.physics.cn_fun_competition import (
    DEFAULT_FUN_COMPETITION_COUPLING_PARAMETERS,
    FunCompetitionCouplingInputs,
    FunCompetitionCouplingParameters,
    FunSoilCompetitionResult,
    fun_soil_biogeochem_competition,
)
from clm_jax.physics.cn_gap_mortality import (
    CnGapMortalityDiagnostics,
    CnGapMortalityInputs,
    CnGapMortalityParameters,
    CnGapMortalityState,
    cn_gap_mortality,
)
from clm_jax.physics.cn_precision_control import (
    DEFAULT_CN_PRECISION_CONTROL_PARAMETERS,
    CnPrecisionControlDiagnostics,
    CnPrecisionControlInputs,
    CnPrecisionControlParameters,
    CnPrecisionControlState,
    cn_precision_control,
    get_cn_precision_control_state_field,
    set_cn_precision_control_state_field,
)
from clm_jax.physics.cn_state_update1 import (
    CnStateUpdate1Diagnostics,
    CnStateUpdate1Inputs,
    CnStateUpdate1Parameters,
    CnStateUpdate1State,
    cn_state_update1_with_precision_control,
)
from clm_jax.physics.cn_state_update2 import (
    DEFAULT_CN_STATE_UPDATE2_PARAMETERS,
    CnStateUpdate2Diagnostics,
    CnStateUpdate2Inputs,
    CnStateUpdate2Parameters,
    CnStateUpdate2State,
    cn_state_update2,
    cn_state_update2h_with_precision_control,
)
from clm_jax.physics.cn_state_update3 import (
    DEFAULT_CN_STATE_UPDATE3_PARAMETERS,
    CnStateUpdate3Diagnostics,
    CnStateUpdate3Parameters,
    cn_state_update3_with_precision_control,
)
from clm_jax.physics.cn_wood_products import (
    CnWoodProductsDiagnostics,
    CnWoodProductsInputs,
    CnWoodProductsParameters,
    CnWoodProductsState,
    cn_wood_products,
)
from clm_jax.physics.cn_zero import (
    CnTransientFluxState,
    CnZeroDiagnostics,
    CnZeroInputs,
    get_cn_flux_field,
    zero_cn_transient_fluxes,
)
from clm_jax.physics.crop import (
    CropYearDiagnostics,
    CropYearInputs,
    CropYearState,
    increment_crop_year,
)
from clm_jax.physics.crop_phenology import (
    CropPhenologyDiagnostics,
    CropPhenologyInputs,
    CropPhenologyParameters,
    CropPhenologyState,
    crop_phenology,
)
from clm_jax.physics.decomposition_rates import (
    DEFAULT_DECOMPOSITION_RATE_PARAMETERS,
    DecompositionRateDiagnostics,
    DecompositionRateInputs,
    DecompositionRateParameters,
    DecompositionScalarState,
    decomposition_rate_constants_bgc,
)
from clm_jax.physics.evergreen_phenology import (
    CARBON_TRANSFER_POOL_NAMES,
    DEFAULT_EVERGREEN_PHENOLOGY_PARAMETERS,
    NITROGEN_TRANSFER_POOL_NAMES,
    EvergreenPhenologyDiagnostics,
    EvergreenPhenologyInputs,
    EvergreenPhenologyParameters,
    EvergreenPhenologyState,
    evergreen_phenology,
)
from clm_jax.physics.fun_initialization import (
    DEFAULT_FUN_INITIALIZATION_PARAMETERS,
    FunInitializationDiagnostics,
    FunInitializationInputs,
    FunInitializationParameters,
    FunInitializationRuntimeState,
    FunInitializationState,
    fun_initialization,
)
from clm_jax.physics.grain_to_product_pools import (
    DEFAULT_GRAIN_TO_PRODUCT_POOLS_PARAMETERS,
    GrainToProductPoolsDiagnostics,
    GrainToProductPoolsInputs,
    GrainToProductPoolsParameters,
    grain_to_product_pools,
)
from clm_jax.physics.growth_respiration import (
    GrowthRespirationDiagnostics,
    GrowthRespirationInputs,
    GrowthRespirationParameters,
    growth_respiration,
)
from clm_jax.physics.litter_to_column import (
    LitterToColumnDiagnostics,
    LitterToColumnInputs,
    LitterToColumnParameters,
    litter_to_column,
)
from clm_jax.physics.livewood_turnover import (
    LivewoodTurnoverDiagnostics,
    LivewoodTurnoverInputs,
    LivewoodTurnoverParameters,
    livewood_turnover,
)
from clm_jax.physics.maintenance_respiration import (
    DEFAULT_MAINTENANCE_RESPIRATION_PARAMETERS,
    MaintenanceRespirationDiagnostics,
    MaintenanceRespirationInputs,
    MaintenanceRespirationParameters,
    maintenance_respiration,
)
from clm_jax.physics.nitrogen_dynamics import (
    DEFAULT_FREE_LIVING_FIXATION_PARAMETERS,
    DEFAULT_NITROGEN_FERTILIZER_PARAMETERS,
    FreeLivingFixationDiagnostics,
    FreeLivingFixationInputs,
    FreeLivingFixationParameters,
    NitrogenDepositionDiagnostics,
    NitrogenDepositionInputs,
    NitrogenFertilizerDiagnostics,
    NitrogenFertilizerInputs,
    NitrogenFertilizerParameters,
    free_living_fixation,
    nitrogen_deposition,
    nitrogen_fertilizer,
)
from clm_jax.physics.offset_litterfall import (
    OffsetLitterfallDiagnostics,
    OffsetLitterfallInputs,
    OffsetLitterfallParameters,
    OffsetLitterfallState,
    offset_litterfall,
)
from clm_jax.physics.onset_growth import (
    DEFAULT_ONSET_GROWTH_PARAMETERS,
    OnsetGrowthDiagnostics,
    OnsetGrowthInputs,
    OnsetGrowthParameters,
    onset_growth,
)
from clm_jax.physics.phenology_climate import (
    DEFAULT_PHENOLOGY_CLIMATE_PARAMETERS,
    PhenologyClimateDiagnostics,
    PhenologyClimateInputs,
    PhenologyClimateParameters,
    PhenologyClimateState,
    phenology_climate,
)
from clm_jax.physics.plant_nutrient_competition import (
    DEFAULT_PLANT_NUTRIENT_COMPETITION_PARAMETERS,
    PlantNutrientCompetitionDiagnostics,
    PlantNutrientCompetitionInputs,
    PlantNutrientCompetitionParameters,
    PlantNutrientCompetitionState,
    plant_nutrient_competition,
)
from clm_jax.physics.plant_nutrient_demand import (
    DEFAULT_PLANT_NUTRIENT_DEMAND_PARAMETERS,
    PlantNutrientDemandDiagnostics,
    PlantNutrientDemandInputs,
    PlantNutrientDemandParameters,
    PlantNutrientDemandState,
    plant_nutrient_demand,
)
from clm_jax.physics.plant_nutrient_demand_aggregation import (
    DEFAULT_PLANT_NUTRIENT_DEMAND_AGGREGATION_PARAMETERS,
    PlantNutrientDemandAggregationDiagnostics,
    PlantNutrientDemandAggregationInputs,
    PlantNutrientDemandAggregationParameters,
    aggregate_plant_nutrient_demand,
)
from clm_jax.physics.seasonal_deciduous_phenology import (
    DEFAULT_SEASONAL_DECIDUOUS_PHENOLOGY_PARAMETERS,
    SeasonalDeciduousPhenologyDiagnostics,
    SeasonalDeciduousPhenologyInputs,
    SeasonalDeciduousPhenologyParameters,
    SeasonalDeciduousPhenologyState,
    seasonal_deciduous_phenology,
)
from clm_jax.physics.soil_biogeochem_competition import (
    DEFAULT_SOIL_BIOGEOCHEM_COMPETITION_PARAMETERS,
    SoilBiogeochemCompetitionDiagnostics,
    SoilBiogeochemCompetitionInputs,
    SoilBiogeochemCompetitionParameters,
    SoilBiogeochemCompetitionState,
)
from clm_jax.physics.soil_biogeochem_decomp import (
    DEFAULT_SOIL_BIOGEOCHEM_DECOMP_PARAMETERS,
    SoilBiogeochemDecompDiagnostics,
    SoilBiogeochemDecompInputs,
    SoilBiogeochemDecompParameters,
    SoilBiogeochemDecompState,
    soil_biogeochem_decomp,
)
from clm_jax.physics.soil_biogeochem_litter_vertical_transport import (
    DEFAULT_SOIL_BIOGEOCHEM_LITTER_VERTICAL_TRANSPORT_PARAMETERS,
    SoilBiogeochemLitterVerticalTransportDiagnostics,
    SoilBiogeochemLitterVerticalTransportInputs,
    SoilBiogeochemLitterVerticalTransportParameters,
    SoilBiogeochemLitterVerticalTransportState,
    soil_biogeochem_litter_vertical_transport,
)
from clm_jax.physics.soil_biogeochem_n_state_update1 import (
    DEFAULT_SOIL_BIOGEOCHEM_N_STATE_UPDATE1_PARAMETERS,
    SoilBiogeochemNStateUpdate1Diagnostics,
    SoilBiogeochemNStateUpdate1Inputs,
    SoilBiogeochemNStateUpdate1Parameters,
    SoilBiogeochemNStateUpdate1State,
    soil_biogeochem_n_state_update1,
)
from clm_jax.physics.soil_biogeochem_nitrif_denitrif import (
    DEFAULT_SOIL_BIOGEOCHEM_NITRIF_DENITRIF_PARAMETERS,
    SoilBiogeochemNitrifDenitrifDiagnostics,
    SoilBiogeochemNitrifDenitrifInputs,
    SoilBiogeochemNitrifDenitrifParameters,
    soil_biogeochem_nitrif_denitrif,
)
from clm_jax.physics.soil_biogeochem_potential import (
    DEFAULT_SOIL_BIOGEOCHEM_POTENTIAL_PARAMETERS,
    SoilBiogeochemPotentialDiagnostics,
    SoilBiogeochemPotentialInputs,
    SoilBiogeochemPotentialParameters,
    SoilBiogeochemPotentialWorkState,
    soil_biogeochem_potential,
)
from clm_jax.physics.soil_biogeochem_vertical_profile import (
    DEFAULT_SOIL_BIOGEOCHEM_VERTICAL_PROFILE_PARAMETERS,
    SoilBiogeochemVerticalProfileDiagnostics,
    SoilBiogeochemVerticalProfileInputs,
    SoilBiogeochemVerticalProfileParameters,
    SoilBiogeochemVerticalProfileState,
    soil_biogeochem_vertical_profile,
)
from clm_jax.physics.stress_deciduous_phenology import (
    DEFAULT_STRESS_DECIDUOUS_PHENOLOGY_PARAMETERS,
    StressDeciduousPhenologyDiagnostics,
    StressDeciduousPhenologyInputs,
    StressDeciduousPhenologyParameters,
    StressDeciduousPhenologyState,
    stress_deciduous_phenology,
)


class EcosystemDynamicsPreDrainagePrefixInputs(NamedTuple):

    crop_year: CropYearInputs
    cn_zero: CnZeroInputs
    nitrogen_deposition: NitrogenDepositionInputs
    free_living_fixation: FreeLivingFixationInputs
    nitrogen_fertilizer: NitrogenFertilizerInputs
    maintenance_respiration: MaintenanceRespirationInputs
    decomposition_rate: DecompositionRateInputs
    soil_biogeochem_potential: SoilBiogeochemPotentialInputs
    soil_biogeochem_vertical_profile: SoilBiogeochemVerticalProfileInputs
    soil_biogeochem_nitrif_denitrif: SoilBiogeochemNitrifDenitrifInputs
    phenology_climate: PhenologyClimateInputs
    evergreen_phenology: EvergreenPhenologyInputs
    seasonal_deciduous_phenology: SeasonalDeciduousPhenologyInputs
    stress_deciduous_phenology: StressDeciduousPhenologyInputs
    crop_phenology: CropPhenologyInputs
    crop_phenology_parameters: CropPhenologyParameters
    fun_initialization: FunInitializationInputs
    plant_nutrient_demand: PlantNutrientDemandInputs
    plant_nutrient_demand_aggregation: PlantNutrientDemandAggregationInputs
    fun_initialization_parameters: FunInitializationParameters = (
        DEFAULT_FUN_INITIALIZATION_PARAMETERS
    )
    plant_nutrient_demand_parameters: PlantNutrientDemandParameters = (
        DEFAULT_PLANT_NUTRIENT_DEMAND_PARAMETERS
    )
    plant_nutrient_demand_aggregation_parameters: PlantNutrientDemandAggregationParameters = (
        DEFAULT_PLANT_NUTRIENT_DEMAND_AGGREGATION_PARAMETERS
    )
    free_living_fixation_parameters: FreeLivingFixationParameters = (
        DEFAULT_FREE_LIVING_FIXATION_PARAMETERS
    )
    nitrogen_fertilizer_parameters: NitrogenFertilizerParameters = (
        DEFAULT_NITROGEN_FERTILIZER_PARAMETERS
    )
    maintenance_respiration_parameters: MaintenanceRespirationParameters = (
        DEFAULT_MAINTENANCE_RESPIRATION_PARAMETERS
    )
    decomposition_rate_parameters: DecompositionRateParameters = (
        DEFAULT_DECOMPOSITION_RATE_PARAMETERS
    )
    soil_biogeochem_potential_parameters: SoilBiogeochemPotentialParameters = (
        DEFAULT_SOIL_BIOGEOCHEM_POTENTIAL_PARAMETERS
    )
    soil_biogeochem_vertical_profile_parameters: SoilBiogeochemVerticalProfileParameters = (
        DEFAULT_SOIL_BIOGEOCHEM_VERTICAL_PROFILE_PARAMETERS
    )
    soil_biogeochem_nitrif_denitrif_parameters: SoilBiogeochemNitrifDenitrifParameters = (
        DEFAULT_SOIL_BIOGEOCHEM_NITRIF_DENITRIF_PARAMETERS
    )
    phenology_climate_parameters: PhenologyClimateParameters = DEFAULT_PHENOLOGY_CLIMATE_PARAMETERS
    evergreen_phenology_parameters: EvergreenPhenologyParameters = (
        DEFAULT_EVERGREEN_PHENOLOGY_PARAMETERS
    )
    seasonal_deciduous_phenology_parameters: SeasonalDeciduousPhenologyParameters = (
        DEFAULT_SEASONAL_DECIDUOUS_PHENOLOGY_PARAMETERS
    )
    stress_deciduous_phenology_parameters: StressDeciduousPhenologyParameters = (
        DEFAULT_STRESS_DECIDUOUS_PHENOLOGY_PARAMETERS
    )


class EcosystemDynamicsPreDrainagePrefixState(NamedTuple):

    crop_year: CropYearState
    cn_fluxes: CnTransientFluxState
    decomposition_scalars: DecompositionScalarState
    soil_biogeochem_potential_work: SoilBiogeochemPotentialWorkState
    soil_biogeochem_vertical_profile: SoilBiogeochemVerticalProfileState
    phenology_climate: PhenologyClimateState
    evergreen_phenology: EvergreenPhenologyState
    seasonal_deciduous_phenology: SeasonalDeciduousPhenologyState
    stress_deciduous_phenology: StressDeciduousPhenologyState
    crop_phenology: CropPhenologyState
    fun_initialization: FunInitializationState
    fun_initialization_runtime: FunInitializationRuntimeState
    plant_nutrient_demand: PlantNutrientDemandState
    plant_nitrogen_demand_column: jnp.ndarray


class EcosystemDynamicsPreDrainagePrefixDiagnostics(NamedTuple):
    crop_year: CropYearDiagnostics
    cn_zero: CnZeroDiagnostics
    nitrogen_deposition: NitrogenDepositionDiagnostics
    free_living_fixation: FreeLivingFixationDiagnostics
    nitrogen_fertilizer: NitrogenFertilizerDiagnostics
    maintenance_respiration: MaintenanceRespirationDiagnostics
    decomposition_rate: DecompositionRateDiagnostics
    soil_biogeochem_potential: SoilBiogeochemPotentialDiagnostics
    soil_biogeochem_vertical_profile: SoilBiogeochemVerticalProfileDiagnostics
    soil_biogeochem_nitrif_denitrif: SoilBiogeochemNitrifDenitrifDiagnostics
    phenology_climate: PhenologyClimateDiagnostics
    evergreen_phenology: EvergreenPhenologyDiagnostics
    seasonal_deciduous_phenology: SeasonalDeciduousPhenologyDiagnostics
    stress_deciduous_phenology: StressDeciduousPhenologyDiagnostics
    crop_phenology: CropPhenologyDiagnostics
    fun_initialization: FunInitializationDiagnostics
    plant_nutrient_demand: PlantNutrientDemandDiagnostics
    plant_nutrient_demand_aggregation: PlantNutrientDemandAggregationDiagnostics


class EcosystemDynamicsPreDrainagePrefixResult(NamedTuple):
    state: EcosystemDynamicsPreDrainagePrefixState
    diagnostics: EcosystemDynamicsPreDrainagePrefixDiagnostics


def ecosystem_dynamics_pre_drainage_prefix(
    inputs: EcosystemDynamicsPreDrainagePrefixInputs,
    state: EcosystemDynamicsPreDrainagePrefixState,
) -> EcosystemDynamicsPreDrainagePrefixResult:

    crop_result = increment_crop_year(inputs.crop_year, state.crop_year)
    cn_zero_result = zero_cn_transient_fluxes(inputs.cn_zero, state.cn_fluxes)
    deposition_result = nitrogen_deposition(
        inputs.nitrogen_deposition,
        cn_zero_result.state,
    )
    fixation_result = free_living_fixation(
        inputs.free_living_fixation,
        deposition_result.state,
        inputs.free_living_fixation_parameters,
    )
    fertilizer_inputs = inputs.nitrogen_fertilizer._replace(
        fertilizer_patch=state.crop_phenology.fertilizer_flux_patch,
    )
    fertilizer_result = nitrogen_fertilizer(
        fertilizer_inputs,
        fixation_result.state,
        inputs.nitrogen_fertilizer_parameters,
    )
    respiration_result = maintenance_respiration(
        inputs.maintenance_respiration,
        fertilizer_result.state,
        inputs.maintenance_respiration_parameters,
    )
    decomposition_result = decomposition_rate_constants_bgc(
        inputs.decomposition_rate,
        respiration_result.state,
        state.decomposition_scalars,
        inputs.decomposition_rate_parameters,
    )
    potential_result = soil_biogeochem_potential(
        inputs.soil_biogeochem_potential,
        decomposition_result.cn_fluxes,
        state.soil_biogeochem_potential_work,
        inputs.soil_biogeochem_potential_parameters,
    )
    vertical_profile_result = soil_biogeochem_vertical_profile(
        inputs.soil_biogeochem_vertical_profile,
        state.soil_biogeochem_vertical_profile,
        inputs.soil_biogeochem_vertical_profile_parameters,
    )
    nitrif_denitrif_inputs = inputs.soil_biogeochem_nitrif_denitrif._replace(
        soil_water_potential_column_level=(
            inputs.decomposition_rate.soil_water_potential_column_level
        ),
        soil_temperature_column_level=(inputs.decomposition_rate.soil_temperature_column_level),
        inundated_fraction_column=(inputs.decomposition_rate.inundated_fraction_column),
        potential_heterotrophic_respiration_column_level=(
            potential_result.work.potential_heterotrophic_respiration_column_level
        ),
        decomposition_water_scalar_column_level=(
            decomposition_result.scalars.water_scalar_column_level
        ),
        decomposition_temperature_scalar_column_level=(
            decomposition_result.scalars.temperature_scalar_column_level
        ),
    )
    nitrif_denitrif_result = soil_biogeochem_nitrif_denitrif(
        nitrif_denitrif_inputs,
        potential_result.cn_fluxes,
        inputs.soil_biogeochem_nitrif_denitrif_parameters,
    )
    phenology_climate_inputs = inputs.phenology_climate._replace(
        years_crop_active_patch=crop_result.state.years_active_patch,
    )
    phenology_climate_result = phenology_climate(
        phenology_climate_inputs,
        state.phenology_climate,
        inputs.phenology_climate_parameters,
    )
    evergreen_inputs = inputs.evergreen_phenology._replace(
        carbon_transfer_patch_pool=(state.seasonal_deciduous_phenology.carbon_transfer_patch_pool),
        nitrogen_transfer_patch_pool=(
            state.seasonal_deciduous_phenology.nitrogen_transfer_patch_pool
        ),
    )
    evergreen_phenology_result = evergreen_phenology(
        evergreen_inputs,
        state.evergreen_phenology,
        nitrif_denitrif_result.cn_fluxes,
        inputs.evergreen_phenology_parameters,
    )
    seasonal_inputs = inputs.seasonal_deciduous_phenology._replace(
        carbon_storage_patch_pool=(inputs.evergreen_phenology.carbon_storage_patch_pool),
        nitrogen_storage_patch_pool=(inputs.evergreen_phenology.nitrogen_storage_patch_pool),
    )
    seasonal_deciduous_result = seasonal_deciduous_phenology(
        seasonal_inputs,
        state.seasonal_deciduous_phenology,
        evergreen_phenology_result.state,
        evergreen_phenology_result.cn_fluxes,
        inputs.seasonal_deciduous_phenology_parameters,
    )
    stress_inputs = inputs.stress_deciduous_phenology._replace(
        carbon_storage_patch_pool=(inputs.evergreen_phenology.carbon_storage_patch_pool),
        nitrogen_storage_patch_pool=(inputs.evergreen_phenology.nitrogen_storage_patch_pool),
    )
    stress_deciduous_result = stress_deciduous_phenology(
        stress_inputs,
        state.stress_deciduous_phenology,
        seasonal_deciduous_result.state,
        seasonal_deciduous_result.phenology_rates,
        seasonal_deciduous_result.cn_fluxes,
        inputs.stress_deciduous_phenology_parameters,
    )
    crop_phenology_inputs = inputs.crop_phenology._replace(
        twenty_year_gdd0_patch=(
            phenology_climate_result.state.growing_degree_days_base_0_twenty_year_patch
        ),
        twenty_year_gdd8_patch=(
            phenology_climate_result.state.growing_degree_days_base_8_twenty_year_patch
        ),
        twenty_year_gdd10_patch=(
            phenology_climate_result.state.growing_degree_days_base_10_twenty_year_patch
        ),
    )
    crop_phenology_result = crop_phenology(
        crop_phenology_inputs,
        state.crop_phenology,
        stress_deciduous_result.common_phenology_state,
        stress_deciduous_result.phenology_rates,
        stress_deciduous_result.cn_fluxes,
        inputs.crop_phenology_parameters,
    )
    fun_initialization_result = fun_initialization(
        inputs.fun_initialization,
        state.fun_initialization,
        inputs.fun_initialization_parameters,
    )
    plant_level_count = inputs.plant_nutrient_demand.soil_mineral_nitrogen_column_level.shape[1]
    plant_nutrient_demand_inputs = inputs.plant_nutrient_demand._replace(
        growing_degree_days_since_planting_patch=(
            crop_phenology_result.state.growing_degree_days_since_planting_patch
        ),
        growing_degree_days_topsoil_patch=(inputs.crop_phenology.growing_degree_days_topsoil_patch),
        crop_live_patch=crop_phenology_result.state.crop_live_patch,
        growing_degree_days_to_maturity_patch=(
            crop_phenology_result.state.growing_degree_days_to_maturity_patch
        ),
        heat_units_to_leaf_emergence_patch=(
            crop_phenology_result.state.heat_units_to_leaf_emergence_patch
        ),
        heat_units_to_grain_fill_patch=(crop_phenology_result.state.heat_units_to_grain_fill_patch),
        peak_leaf_area_index_patch=(inputs.crop_phenology.peak_leaf_area_index_patch),
        decomposition_temperature_scalar_column_level=(
            decomposition_result.scalars.temperature_scalar_column_level[:, :plant_level_count]
        ),
    )
    plant_nutrient_demand_result = plant_nutrient_demand(
        plant_nutrient_demand_inputs,
        state.plant_nutrient_demand._replace(
            cn_fluxes=crop_phenology_result.cn_fluxes,
        ),
        inputs.plant_nutrient_demand_parameters,
    )
    plant_nutrient_demand_aggregation_result = aggregate_plant_nutrient_demand(
        plant_nutrient_demand_result.state.direct_fluxes.plant_nitrogen_demand_patch,
        inputs.plant_nutrient_demand_aggregation,
        inputs.plant_nutrient_demand_aggregation_parameters,
    )
    return EcosystemDynamicsPreDrainagePrefixResult(
        state=EcosystemDynamicsPreDrainagePrefixState(
            crop_year=crop_result.state,
            cn_fluxes=plant_nutrient_demand_result.state.cn_fluxes,
            decomposition_scalars=decomposition_result.scalars,
            soil_biogeochem_potential_work=potential_result.work,
            soil_biogeochem_vertical_profile=vertical_profile_result.state,
            phenology_climate=phenology_climate_result.state,
            evergreen_phenology=crop_phenology_result.phenology_rates,
            seasonal_deciduous_phenology=(crop_phenology_result.common_phenology_state),
            stress_deciduous_phenology=stress_deciduous_result.state,
            crop_phenology=crop_phenology_result.state,
            fun_initialization=fun_initialization_result.state,
            fun_initialization_runtime=fun_initialization_result.runtime,
            plant_nutrient_demand=plant_nutrient_demand_result.state,
            plant_nitrogen_demand_column=(
                plant_nutrient_demand_aggregation_result.plant_nitrogen_demand_column
            ),
        ),
        diagnostics=EcosystemDynamicsPreDrainagePrefixDiagnostics(
            crop_year=crop_result.diagnostics,
            cn_zero=cn_zero_result.diagnostics,
            nitrogen_deposition=deposition_result.diagnostics,
            free_living_fixation=fixation_result.diagnostics,
            nitrogen_fertilizer=fertilizer_result.diagnostics,
            maintenance_respiration=respiration_result.diagnostics,
            decomposition_rate=decomposition_result.diagnostics,
            soil_biogeochem_potential=potential_result.diagnostics,
            soil_biogeochem_vertical_profile=vertical_profile_result.diagnostics,
            soil_biogeochem_nitrif_denitrif=(nitrif_denitrif_result.diagnostics),
            phenology_climate=phenology_climate_result.diagnostics,
            evergreen_phenology=evergreen_phenology_result.diagnostics,
            seasonal_deciduous_phenology=(seasonal_deciduous_result.diagnostics),
            stress_deciduous_phenology=(stress_deciduous_result.diagnostics),
            crop_phenology=crop_phenology_result.diagnostics,
            fun_initialization=fun_initialization_result.diagnostics,
            plant_nutrient_demand=(plant_nutrient_demand_result.diagnostics),
            plant_nutrient_demand_aggregation=(
                plant_nutrient_demand_aggregation_result.diagnostics
            ),
        ),
    )


class EcosystemDynamicsCompetitionDecompositionInputs(NamedTuple):

    fun_competition_coupling: FunCompetitionCouplingInputs
    fun_allocation: FunAllocationInputs
    soil_biogeochem_competition: SoilBiogeochemCompetitionInputs
    plant_nutrient_competition: PlantNutrientCompetitionInputs
    soil_biogeochem_decomp: SoilBiogeochemDecompInputs
    fun_allocation_parameters: FunAllocationParameters
    fun_competition_coupling_parameters: FunCompetitionCouplingParameters = (
        DEFAULT_FUN_COMPETITION_COUPLING_PARAMETERS
    )
    soil_biogeochem_competition_parameters: SoilBiogeochemCompetitionParameters = (
        DEFAULT_SOIL_BIOGEOCHEM_COMPETITION_PARAMETERS
    )
    plant_nutrient_competition_parameters: PlantNutrientCompetitionParameters = (
        DEFAULT_PLANT_NUTRIENT_COMPETITION_PARAMETERS
    )
    soil_biogeochem_decomp_parameters: SoilBiogeochemDecompParameters = (
        DEFAULT_SOIL_BIOGEOCHEM_DECOMP_PARAMETERS
    )


class EcosystemDynamicsCompetitionDecompositionState(NamedTuple):

    fun_allocation: FunAllocationState
    soil_biogeochem_competition: SoilBiogeochemCompetitionState
    plant_nutrient_competition: PlantNutrientCompetitionState
    soil_biogeochem_decomp: SoilBiogeochemDecompState


class EcosystemDynamicsCompetitionDecompositionDiagnostics(NamedTuple):
    soil_biogeochem_competition_pre_fun: SoilBiogeochemCompetitionDiagnostics
    fun_allocation: FunAllocationDiagnostics
    soil_biogeochem_competition: SoilBiogeochemCompetitionDiagnostics
    plant_nutrient_competition: PlantNutrientCompetitionDiagnostics
    soil_biogeochem_decomp: SoilBiogeochemDecompDiagnostics


class EcosystemDynamicsCompetitionDecompositionResult(NamedTuple):
    state: EcosystemDynamicsCompetitionDecompositionState
    cn_fluxes: CnTransientFluxState
    soil_biogeochem_potential_work: SoilBiogeochemPotentialWorkState
    diagnostics: EcosystemDynamicsCompetitionDecompositionDiagnostics


def assemble_ecosystem_fun_inputs(
    prefix: EcosystemDynamicsPreDrainagePrefixResult,
    inputs: EcosystemDynamicsCompetitionDecompositionInputs,
    state: EcosystemDynamicsCompetitionDecompositionState,
) -> tuple[SoilBiogeochemCompetitionInputs, FunAllocationInputs, FunAllocationState]:

    prefix_state = prefix.state
    cn_fluxes = prefix_state.cn_fluxes
    competition_inputs = inputs.soil_biogeochem_competition._replace(
        plant_nitrogen_demand_column=(prefix_state.plant_nitrogen_demand_column),
        nitrogen_fixation_profile_column_level=(
            prefix_state.soil_biogeochem_vertical_profile.nitrogen_fixation_profile_column_level
        ),
        potential_nitrification_column_level=get_cn_flux_field(
            cn_fluxes,
            "soil_nitrogen.pot_f_nit_vr_col",
        ),
        potential_denitrification_column_level=get_cn_flux_field(
            cn_fluxes,
            "soil_nitrogen.pot_f_denit_vr_col",
        ),
        potential_immobilization_column_level=get_cn_flux_field(
            cn_fluxes,
            "soil_nitrogen.potential_immob_vr_col",
        ),
        n2_to_n2o_ratio_denitrification_column_level=get_cn_flux_field(
            cn_fluxes,
            "soil_nitrogen.n2_n2o_ratio_denit_vr_col",
        ),
    )
    demand_state = prefix_state.plant_nutrient_demand
    common_phenology_state = prefix_state.seasonal_deciduous_phenology
    fun_inputs = inputs.fun_allocation._replace(
        available_carbon_patch=(demand_state.direct_fluxes.available_carbon_patch),
        plant_nitrogen_demand_patch=(demand_state.direct_fluxes.plant_nitrogen_demand_patch),
        leaf_nitrogen_storage_to_transfer_patch=get_cn_flux_field(
            cn_fluxes,
            "vegetation_nitrogen.leafn_storage_to_xfer_patch",
        ),
        leaf_carbon_storage_to_transfer_patch=get_cn_flux_field(
            cn_fluxes,
            "vegetation_carbon.leafc_storage_to_xfer_patch",
        ),
        carbon_allometry_patch=demand_state.carbon_allometry_patch,
        nitrogen_allometry_patch=demand_state.nitrogen_allometry_patch,
        onset_flag_patch=common_phenology_state.onset_flag_patch,
        offset_flag_patch=common_phenology_state.offset_flag_patch,
    )
    initialized_fun = prefix_state.fun_initialization
    fun_state = state.fun_allocation._replace(
        leaf_carbon_nitrogen_offset_patch=(initialized_fun.leaf_cn_ratio_offset_patch),
        leaf_carbon_storage_transfer_accumulator_patch=(
            initialized_fun.leaf_carbon_storage_transfer_accumulator_patch
        ),
        leaf_nitrogen_storage_transfer_accumulator_patch=(
            initialized_fun.leaf_nitrogen_storage_transfer_accumulator_patch
        ),
        storage_carbon_demand_patch=(initialized_fun.storage_carbon_demand_patch),
        storage_nitrogen_demand_patch=(initialized_fun.storage_nitrogen_demand_patch),
    )
    return competition_inputs, fun_inputs, fun_state


def complete_ecosystem_fun_decomposition(
    prefix: EcosystemDynamicsPreDrainagePrefixResult,
    inputs: EcosystemDynamicsCompetitionDecompositionInputs,
    state: EcosystemDynamicsCompetitionDecompositionState,
    fun_soil_result: FunSoilCompetitionResult,
) -> EcosystemDynamicsCompetitionDecompositionResult:

    prefix_state = prefix.state
    demand_state = prefix_state.plant_nutrient_demand
    plant_inputs = inputs.plant_nutrient_competition._replace(
        crop_live_patch=prefix_state.crop_phenology.crop_live_patch,
        leaf_allocation_patch=demand_state.leaf_allocation_patch,
        stem_allocation_patch=demand_state.stem_allocation_patch,
        carbon_allometry_patch=demand_state.carbon_allometry_patch,
        root_allocation_patch=demand_state.root_allocation_patch,
        reproductive_allocation_patch=demand_state.reproductive_allocation_patch,
        fun_soil_nitrogen_uptake_patch=(
            fun_soil_result.allocation.nitrogen_fluxes.soil_nitrogen_to_plant_fun_patch
        ),
    )
    plant_result = plant_nutrient_competition(
        plant_inputs,
        state.plant_nutrient_competition._replace(
            cn_fluxes=fun_soil_result.competition.cn_fluxes,
        ),
        inputs.plant_nutrient_competition_parameters,
    )

    decomp_inputs = inputs.soil_biogeochem_decomp._replace(
        fraction_potential_immobilization_column_level=(
            fun_soil_result.competition.state.fraction_potential_immobilization_column_level
        ),
        water_scalar_column_level=(prefix_state.decomposition_scalars.water_scalar_column_level),
    )
    decomp_result = soil_biogeochem_decomp(
        decomp_inputs,
        state.soil_biogeochem_decomp._replace(
            cn_fluxes=plant_result.state.cn_fluxes,
        ),
        prefix_state.soil_biogeochem_potential_work,
        inputs.soil_biogeochem_decomp_parameters,
    )
    final_cn_fluxes = decomp_result.state.cn_fluxes
    return EcosystemDynamicsCompetitionDecompositionResult(
        state=EcosystemDynamicsCompetitionDecompositionState(
            fun_allocation=fun_soil_result.allocation.state,
            soil_biogeochem_competition=fun_soil_result.competition.state,
            plant_nutrient_competition=plant_result.state._replace(
                cn_fluxes=final_cn_fluxes,
            ),
            soil_biogeochem_decomp=decomp_result.state,
        ),
        cn_fluxes=final_cn_fluxes,
        soil_biogeochem_potential_work=decomp_result.work,
        diagnostics=EcosystemDynamicsCompetitionDecompositionDiagnostics(
            soil_biogeochem_competition_pre_fun=(fun_soil_result.pre_fun_diagnostics),
            fun_allocation=fun_soil_result.allocation.diagnostics,
            soil_biogeochem_competition=(fun_soil_result.competition.diagnostics),
            plant_nutrient_competition=plant_result.diagnostics,
            soil_biogeochem_decomp=decomp_result.diagnostics,
        ),
    )


def ecosystem_dynamics_competition_through_decomposition(
    prefix: EcosystemDynamicsPreDrainagePrefixResult,
    inputs: EcosystemDynamicsCompetitionDecompositionInputs,
    state: EcosystemDynamicsCompetitionDecompositionState,
) -> EcosystemDynamicsCompetitionDecompositionResult:

    competition_inputs, fun_inputs, fun_state = assemble_ecosystem_fun_inputs(
        prefix,
        inputs,
        state,
    )
    cn_fluxes = prefix.state.cn_fluxes
    fun_soil_result = fun_soil_biogeochem_competition(
        inputs.fun_competition_coupling,
        fun_inputs,
        fun_state,
        inputs.fun_allocation_parameters,
        competition_inputs,
        state.soil_biogeochem_competition,
        cn_fluxes,
        inputs.soil_biogeochem_competition_parameters,
        inputs.fun_competition_coupling_parameters,
    )

    return complete_ecosystem_fun_decomposition(prefix, inputs, state, fun_soil_result)


class EcosystemDynamicsPreDrainageThroughDecompositionInputs(NamedTuple):
    prefix: EcosystemDynamicsPreDrainagePrefixInputs
    competition_decomposition: EcosystemDynamicsCompetitionDecompositionInputs


class EcosystemDynamicsPreDrainageThroughDecompositionState(NamedTuple):
    prefix: EcosystemDynamicsPreDrainagePrefixState
    competition_decomposition: EcosystemDynamicsCompetitionDecompositionState


class EcosystemDynamicsPreDrainageThroughDecompositionDiagnostics(NamedTuple):
    prefix: EcosystemDynamicsPreDrainagePrefixDiagnostics
    competition_decomposition: EcosystemDynamicsCompetitionDecompositionDiagnostics


class EcosystemDynamicsPreDrainageThroughDecompositionResult(NamedTuple):
    state: EcosystemDynamicsPreDrainageThroughDecompositionState
    diagnostics: EcosystemDynamicsPreDrainageThroughDecompositionDiagnostics


def ecosystem_dynamics_pre_drainage_through_decomposition(
    inputs: EcosystemDynamicsPreDrainageThroughDecompositionInputs,
    state: EcosystemDynamicsPreDrainageThroughDecompositionState,
) -> EcosystemDynamicsPreDrainageThroughDecompositionResult:

    prefix_result = ecosystem_dynamics_pre_drainage_prefix(
        inputs.prefix,
        state.prefix,
    )
    middle_result = ecosystem_dynamics_competition_through_decomposition(
        prefix_result,
        inputs.competition_decomposition,
        state.competition_decomposition,
    )
    allocation_state = middle_result.state.fun_allocation
    synchronized_fun_initialization = prefix_result.state.fun_initialization._replace(
        leaf_cn_ratio_offset_patch=(allocation_state.leaf_carbon_nitrogen_offset_patch),
        leaf_carbon_storage_transfer_accumulator_patch=(
            allocation_state.leaf_carbon_storage_transfer_accumulator_patch
        ),
        leaf_nitrogen_storage_transfer_accumulator_patch=(
            allocation_state.leaf_nitrogen_storage_transfer_accumulator_patch
        ),
        storage_carbon_demand_patch=allocation_state.storage_carbon_demand_patch,
        storage_nitrogen_demand_patch=(allocation_state.storage_nitrogen_demand_patch),
    )
    final_cn_fluxes = middle_result.cn_fluxes
    synchronized_prefix = prefix_result.state._replace(
        cn_fluxes=final_cn_fluxes,
        soil_biogeochem_potential_work=(middle_result.soil_biogeochem_potential_work),
        fun_initialization=synchronized_fun_initialization,
        plant_nutrient_demand=(
            prefix_result.state.plant_nutrient_demand._replace(
                cn_fluxes=final_cn_fluxes,
            )
        ),
    )
    return EcosystemDynamicsPreDrainageThroughDecompositionResult(
        state=EcosystemDynamicsPreDrainageThroughDecompositionState(
            prefix=synchronized_prefix,
            competition_decomposition=middle_result.state,
        ),
        diagnostics=EcosystemDynamicsPreDrainageThroughDecompositionDiagnostics(
            prefix=prefix_result.diagnostics,
            competition_decomposition=middle_result.diagnostics,
        ),
    )


class EcosystemDynamicsPhaseTwoLitterInputs(NamedTuple):

    onset_growth: OnsetGrowthInputs
    offset_litterfall: OffsetLitterfallInputs
    background_litterfall: BackgroundLitterfallInputs
    livewood_turnover: LivewoodTurnoverInputs
    grain_to_product_pools: GrainToProductPoolsInputs
    litter_to_column: LitterToColumnInputs
    offset_litterfall_parameters: OffsetLitterfallParameters
    background_litterfall_parameters: BackgroundLitterfallParameters
    livewood_turnover_parameters: LivewoodTurnoverParameters
    litter_to_column_parameters: LitterToColumnParameters
    onset_growth_parameters: OnsetGrowthParameters = DEFAULT_ONSET_GROWTH_PARAMETERS
    grain_to_product_pools_parameters: GrainToProductPoolsParameters = (
        DEFAULT_GRAIN_TO_PRODUCT_POOLS_PARAMETERS
    )


class EcosystemDynamicsPhaseTwoLitterState(NamedTuple):

    leaf_carbon_to_litter_fun_patch: jnp.ndarray


class EcosystemDynamicsPhaseTwoLitterDiagnostics(NamedTuple):
    onset_growth: OnsetGrowthDiagnostics
    offset_litterfall: OffsetLitterfallDiagnostics
    background_litterfall: BackgroundLitterfallDiagnostics
    livewood_turnover: LivewoodTurnoverDiagnostics
    grain_to_product_pools: GrainToProductPoolsDiagnostics
    litter_to_column: LitterToColumnDiagnostics


class EcosystemDynamicsPhaseTwoLitterResult(NamedTuple):
    pre_decomposition_state: EcosystemDynamicsPreDrainageThroughDecompositionState
    phase_two_state: EcosystemDynamicsPhaseTwoLitterState
    cn_fluxes: CnTransientFluxState
    diagnostics: EcosystemDynamicsPhaseTwoLitterDiagnostics


def ecosystem_dynamics_phase_two_litterfall(
    pre_decomposition_state: EcosystemDynamicsPreDrainageThroughDecompositionState,
    inputs: EcosystemDynamicsPhaseTwoLitterInputs,
    state: EcosystemDynamicsPhaseTwoLitterState,
) -> EcosystemDynamicsPhaseTwoLitterResult:

    prefix_state = pre_decomposition_state.prefix
    middle_state = pre_decomposition_state.competition_decomposition
    phenology_state = prefix_state.seasonal_deciduous_phenology
    phenology_rates = prefix_state.evergreen_phenology
    cn_fluxes = prefix_state.cn_fluxes

    onset_inputs = inputs.onset_growth._replace(
        onset_flag_patch=phenology_state.onset_flag_patch,
        onset_counter_seconds_patch=phenology_state.onset_counter_seconds_patch,
        background_transfer_growth_rate_patch=(
            phenology_rates.background_transfer_growth_rate_patch
        ),
        carbon_transfer_patch_pool=phenology_state.carbon_transfer_patch_pool,
        nitrogen_transfer_patch_pool=phenology_state.nitrogen_transfer_patch_pool,
    )
    onset_result = onset_growth(
        onset_inputs,
        cn_fluxes,
        inputs.onset_growth_parameters,
    )

    offset_inputs = inputs.offset_litterfall._replace(
        offset_flag_patch=phenology_state.offset_flag_patch,
        offset_counter_seconds_patch=phenology_state.offset_counter_seconds_patch,
    )
    offset_result = offset_litterfall(
        offset_inputs,
        OffsetLitterfallState(
            previous_leaf_carbon_to_litter_patch=(
                phenology_state.previous_leaf_carbon_to_litter_patch
            ),
            previous_fine_root_carbon_to_litter_patch=(
                phenology_state.previous_fine_root_carbon_to_litter_patch
            ),
            leaf_carbon_to_litter_fun_patch=(state.leaf_carbon_to_litter_fun_patch),
            leaf_carbon_nitrogen_offset_patch=(
                middle_state.fun_allocation.leaf_carbon_nitrogen_offset_patch
            ),
        ),
        onset_result.cn_fluxes,
        inputs.offset_litterfall_parameters,
    )

    background_inputs = inputs.background_litterfall._replace(
        background_litterfall_rate_patch=(phenology_rates.background_litterfall_rate_patch),
    )
    background_result = background_litterfall(
        background_inputs,
        BackgroundLitterfallState(
            leaf_carbon_to_litter_fun_patch=(offset_result.state.leaf_carbon_to_litter_fun_patch),
            leaf_carbon_nitrogen_offset_patch=(
                offset_result.state.leaf_carbon_nitrogen_offset_patch
            ),
        ),
        offset_result.cn_fluxes,
        inputs.background_litterfall_parameters,
    )
    livewood_result = livewood_turnover(
        inputs.livewood_turnover,
        background_result.cn_fluxes,
        inputs.livewood_turnover_parameters,
    )
    grain_result = grain_to_product_pools(
        inputs.grain_to_product_pools,
        livewood_result.cn_fluxes,
        inputs.grain_to_product_pools_parameters,
    )
    vertical_profile = prefix_state.soil_biogeochem_vertical_profile
    litter_inputs = inputs.litter_to_column._replace(
        leaf_profile_patch_level=vertical_profile.leaf_profile_patch_level,
        fine_root_profile_patch_level=(vertical_profile.fine_root_profile_patch_level),
    )
    litter_result = litter_to_column(
        litter_inputs,
        grain_result.cn_fluxes,
        inputs.litter_to_column_parameters,
    )

    final_cn_fluxes = litter_result.cn_fluxes
    final_leaf_cn_offset = background_result.state.leaf_carbon_nitrogen_offset_patch
    synchronized_phenology = phenology_state._replace(
        previous_leaf_carbon_to_litter_patch=(
            offset_result.state.previous_leaf_carbon_to_litter_patch
        ),
        previous_fine_root_carbon_to_litter_patch=(
            offset_result.state.previous_fine_root_carbon_to_litter_patch
        ),
    )
    synchronized_prefix = prefix_state._replace(
        cn_fluxes=final_cn_fluxes,
        seasonal_deciduous_phenology=synchronized_phenology,
        fun_initialization=prefix_state.fun_initialization._replace(
            leaf_cn_ratio_offset_patch=final_leaf_cn_offset,
        ),
        plant_nutrient_demand=prefix_state.plant_nutrient_demand._replace(
            cn_fluxes=final_cn_fluxes,
        ),
    )
    synchronized_middle = middle_state._replace(
        fun_allocation=middle_state.fun_allocation._replace(
            leaf_carbon_nitrogen_offset_patch=final_leaf_cn_offset,
        ),
        plant_nutrient_competition=(
            middle_state.plant_nutrient_competition._replace(
                cn_fluxes=final_cn_fluxes,
            )
        ),
        soil_biogeochem_decomp=middle_state.soil_biogeochem_decomp._replace(
            cn_fluxes=final_cn_fluxes,
        ),
    )
    return EcosystemDynamicsPhaseTwoLitterResult(
        pre_decomposition_state=(
            EcosystemDynamicsPreDrainageThroughDecompositionState(
                prefix=synchronized_prefix,
                competition_decomposition=synchronized_middle,
            )
        ),
        phase_two_state=EcosystemDynamicsPhaseTwoLitterState(
            leaf_carbon_to_litter_fun_patch=(
                background_result.state.leaf_carbon_to_litter_fun_patch
            ),
        ),
        cn_fluxes=final_cn_fluxes,
        diagnostics=EcosystemDynamicsPhaseTwoLitterDiagnostics(
            onset_growth=onset_result.diagnostics,
            offset_litterfall=offset_result.diagnostics,
            background_litterfall=background_result.diagnostics,
            livewood_turnover=livewood_result.diagnostics,
            grain_to_product_pools=grain_result.diagnostics,
            litter_to_column=litter_result.diagnostics,
        ),
    )


class EcosystemDynamicsThroughLitterToColumnInputs(NamedTuple):
    pre_decomposition: EcosystemDynamicsPreDrainageThroughDecompositionInputs
    phase_two_litter: EcosystemDynamicsPhaseTwoLitterInputs


class EcosystemDynamicsThroughLitterToColumnState(NamedTuple):
    pre_decomposition: EcosystemDynamicsPreDrainageThroughDecompositionState
    phase_two_litter: EcosystemDynamicsPhaseTwoLitterState


class EcosystemDynamicsThroughLitterToColumnDiagnostics(NamedTuple):
    pre_decomposition: EcosystemDynamicsPreDrainageThroughDecompositionDiagnostics
    phase_two_litter: EcosystemDynamicsPhaseTwoLitterDiagnostics


class EcosystemDynamicsThroughLitterToColumnResult(NamedTuple):
    state: EcosystemDynamicsThroughLitterToColumnState
    diagnostics: EcosystemDynamicsThroughLitterToColumnDiagnostics


def ecosystem_dynamics_through_litter_to_column(
    inputs: EcosystemDynamicsThroughLitterToColumnInputs,
    state: EcosystemDynamicsThroughLitterToColumnState,
) -> EcosystemDynamicsThroughLitterToColumnResult:

    persistent_leaf_litter_fun = state.phase_two_litter.leaf_carbon_to_litter_fun_patch
    competition_inputs = inputs.pre_decomposition.competition_decomposition
    synchronized_competition_inputs = competition_inputs._replace(
        fun_allocation=competition_inputs.fun_allocation._replace(
            leaf_carbon_to_litter_fun_patch=persistent_leaf_litter_fun,
        ),
    )
    pre_decomposition_result = ecosystem_dynamics_pre_drainage_through_decomposition(
        inputs.pre_decomposition._replace(
            competition_decomposition=synchronized_competition_inputs,
        ),
        state.pre_decomposition,
    )
    phase_two_result = ecosystem_dynamics_phase_two_litterfall(
        pre_decomposition_result.state,
        inputs.phase_two_litter,
        state.phase_two_litter,
    )
    return EcosystemDynamicsThroughLitterToColumnResult(
        state=EcosystemDynamicsThroughLitterToColumnState(
            pre_decomposition=phase_two_result.pre_decomposition_state,
            phase_two_litter=phase_two_result.phase_two_state,
        ),
        diagnostics=EcosystemDynamicsThroughLitterToColumnDiagnostics(
            pre_decomposition=pre_decomposition_result.diagnostics,
            phase_two_litter=phase_two_result.diagnostics,
        ),
    )


class EcosystemDynamicsGrowthRespirationResult(NamedTuple):
    state: EcosystemDynamicsThroughLitterToColumnState
    cn_fluxes: CnTransientFluxState
    diagnostics: GrowthRespirationDiagnostics


def ecosystem_dynamics_growth_respiration(
    state: EcosystemDynamicsThroughLitterToColumnState,
    inputs: GrowthRespirationInputs,
    parameters: GrowthRespirationParameters,
) -> EcosystemDynamicsGrowthRespirationResult:

    pre_decomposition = state.pre_decomposition
    prefix = pre_decomposition.prefix
    middle = pre_decomposition.competition_decomposition
    result = growth_respiration(
        inputs,
        prefix.cn_fluxes,
        parameters,
    )
    final_cn_fluxes = result.cn_fluxes
    synchronized_prefix = prefix._replace(
        cn_fluxes=final_cn_fluxes,
        plant_nutrient_demand=prefix.plant_nutrient_demand._replace(
            cn_fluxes=final_cn_fluxes,
        ),
    )
    synchronized_middle = middle._replace(
        plant_nutrient_competition=middle.plant_nutrient_competition._replace(
            cn_fluxes=final_cn_fluxes,
        ),
        soil_biogeochem_decomp=middle.soil_biogeochem_decomp._replace(
            cn_fluxes=final_cn_fluxes,
        ),
    )
    return EcosystemDynamicsGrowthRespirationResult(
        state=state._replace(
            pre_decomposition=pre_decomposition._replace(
                prefix=synchronized_prefix,
                competition_decomposition=synchronized_middle,
            ),
        ),
        cn_fluxes=final_cn_fluxes,
        diagnostics=result.diagnostics,
    )


class EcosystemDynamicsThroughGrowthRespirationInputs(NamedTuple):
    through_litter: EcosystemDynamicsThroughLitterToColumnInputs
    growth_respiration: GrowthRespirationInputs
    growth_respiration_parameters: GrowthRespirationParameters


class EcosystemDynamicsThroughGrowthRespirationDiagnostics(NamedTuple):
    through_litter: EcosystemDynamicsThroughLitterToColumnDiagnostics
    growth_respiration: GrowthRespirationDiagnostics


class EcosystemDynamicsThroughGrowthRespirationResult(NamedTuple):
    state: EcosystemDynamicsThroughLitterToColumnState
    diagnostics: EcosystemDynamicsThroughGrowthRespirationDiagnostics


def ecosystem_dynamics_through_growth_respiration(
    inputs: EcosystemDynamicsThroughGrowthRespirationInputs,
    state: EcosystemDynamicsThroughLitterToColumnState,
) -> EcosystemDynamicsThroughGrowthRespirationResult:

    litter_result = ecosystem_dynamics_through_litter_to_column(
        inputs.through_litter,
        state,
    )
    growth_result = ecosystem_dynamics_growth_respiration(
        litter_result.state,
        inputs.growth_respiration,
        inputs.growth_respiration_parameters,
    )
    return EcosystemDynamicsThroughGrowthRespirationResult(
        state=growth_result.state,
        diagnostics=EcosystemDynamicsThroughGrowthRespirationDiagnostics(
            through_litter=litter_result.diagnostics,
            growth_respiration=growth_result.diagnostics,
        ),
    )


class EcosystemDynamicsCStateUpdate0PrecisionInputs(NamedTuple):
    c_state_update0: CStateUpdate0Inputs
    cn_precision_control: CnPrecisionControlInputs
    c_state_update0_parameters: CStateUpdate0Parameters = DEFAULT_C_STATE_UPDATE0_PARAMETERS
    cn_precision_control_parameters: CnPrecisionControlParameters = (
        DEFAULT_CN_PRECISION_CONTROL_PARAMETERS
    )


class EcosystemDynamicsCStateUpdate0PrecisionState(NamedTuple):
    ecosystem: EcosystemDynamicsThroughLitterToColumnState
    vegetation_carbon_nitrogen: CnPrecisionControlState


class EcosystemDynamicsCStateUpdate0PrecisionDiagnostics(NamedTuple):
    c_state_update0: CStateUpdate0Diagnostics
    cn_precision_control: CnPrecisionControlDiagnostics


class EcosystemDynamicsCStateUpdate0PrecisionResult(NamedTuple):
    state: EcosystemDynamicsCStateUpdate0PrecisionState
    diagnostics: EcosystemDynamicsCStateUpdate0PrecisionDiagnostics


def _synchronize_phenology_transfer_pools(
    vegetation: CnPrecisionControlState,
    phenology: SeasonalDeciduousPhenologyState,
) -> CnPrecisionControlState:

    synchronized = vegetation
    for index, name in enumerate(CARBON_TRANSFER_POOL_NAMES):
        synchronized = set_cn_precision_control_state_field(
            synchronized,
            f"vegetation_carbon.{name}_patch",
            phenology.carbon_transfer_patch_pool[:, index],
        )
    for index, name in enumerate(NITROGEN_TRANSFER_POOL_NAMES):
        synchronized = set_cn_precision_control_state_field(
            synchronized,
            f"vegetation_nitrogen.{name}_patch",
            phenology.nitrogen_transfer_patch_pool[:, index],
        )
    return synchronized


def ecosystem_dynamics_c_state_update0_precision(
    inputs: EcosystemDynamicsCStateUpdate0PrecisionInputs,
    state: EcosystemDynamicsCStateUpdate0PrecisionState,
) -> EcosystemDynamicsCStateUpdate0PrecisionResult:

    ecosystem = state.ecosystem
    pre_decomposition = ecosystem.pre_decomposition
    prefix = pre_decomposition.prefix
    middle = pre_decomposition.competition_decomposition
    canonical_cn_fluxes = prefix.cn_fluxes
    synchronized_vegetation = _synchronize_phenology_transfer_pools(
        state.vegetation_carbon_nitrogen,
        prefix.seasonal_deciduous_phenology,
    )
    carbon_pool = get_cn_precision_control_state_field(
        synchronized_vegetation,
        "vegetation_carbon.cpool_patch",
    )
    carbon_result = c_state_update0(
        inputs.c_state_update0,
        CStateUpdate0State(carbon_pool_patch=carbon_pool),
        canonical_cn_fluxes,
        inputs.c_state_update0_parameters,
    )
    state_with_updated_carbon = set_cn_precision_control_state_field(
        synchronized_vegetation,
        "vegetation_carbon.cpool_patch",
        carbon_result.state.carbon_pool_patch,
    )
    precision_result = cn_precision_control(
        inputs.cn_precision_control._replace(
            soil_patch=inputs.c_state_update0.soil_patch,
        ),
        state_with_updated_carbon,
        inputs.cn_precision_control_parameters,
    )

    synchronized_prefix = prefix._replace(
        plant_nutrient_demand=prefix.plant_nutrient_demand._replace(
            cn_fluxes=canonical_cn_fluxes,
        ),
    )
    synchronized_middle = middle._replace(
        plant_nutrient_competition=middle.plant_nutrient_competition._replace(
            cn_fluxes=canonical_cn_fluxes,
        ),
        soil_biogeochem_decomp=middle.soil_biogeochem_decomp._replace(
            cn_fluxes=canonical_cn_fluxes,
        ),
    )
    synchronized_ecosystem = ecosystem._replace(
        pre_decomposition=pre_decomposition._replace(
            prefix=synchronized_prefix,
            competition_decomposition=synchronized_middle,
        ),
    )
    return EcosystemDynamicsCStateUpdate0PrecisionResult(
        state=EcosystemDynamicsCStateUpdate0PrecisionState(
            ecosystem=synchronized_ecosystem,
            vegetation_carbon_nitrogen=precision_result.state,
        ),
        diagnostics=EcosystemDynamicsCStateUpdate0PrecisionDiagnostics(
            c_state_update0=carbon_result.diagnostics,
            cn_precision_control=precision_result.diagnostics,
        ),
    )


class EcosystemDynamicsThroughFirstGuardrailInputs(NamedTuple):
    through_growth_respiration: EcosystemDynamicsThroughGrowthRespirationInputs
    state_update_and_precision: EcosystemDynamicsCStateUpdate0PrecisionInputs


class EcosystemDynamicsThroughFirstGuardrailDiagnostics(NamedTuple):
    through_growth_respiration: EcosystemDynamicsThroughGrowthRespirationDiagnostics
    state_update_and_precision: EcosystemDynamicsCStateUpdate0PrecisionDiagnostics


class EcosystemDynamicsThroughFirstGuardrailResult(NamedTuple):
    state: EcosystemDynamicsCStateUpdate0PrecisionState
    diagnostics: EcosystemDynamicsThroughFirstGuardrailDiagnostics


def ecosystem_dynamics_through_first_guardrail(
    inputs: EcosystemDynamicsThroughFirstGuardrailInputs,
    state: EcosystemDynamicsCStateUpdate0PrecisionState,
) -> EcosystemDynamicsThroughFirstGuardrailResult:

    growth_result = ecosystem_dynamics_through_growth_respiration(
        inputs.through_growth_respiration,
        state.ecosystem,
    )
    guarded_result = ecosystem_dynamics_c_state_update0_precision(
        inputs.state_update_and_precision,
        state._replace(ecosystem=growth_result.state),
    )
    return EcosystemDynamicsThroughFirstGuardrailResult(
        state=guarded_result.state,
        diagnostics=EcosystemDynamicsThroughFirstGuardrailDiagnostics(
            through_growth_respiration=growth_result.diagnostics,
            state_update_and_precision=guarded_result.diagnostics,
        ),
    )


class EcosystemDynamicsCnStateUpdate1Inputs(NamedTuple):

    state_update: CnStateUpdate1Inputs
    state_update_parameters: CnStateUpdate1Parameters
    precision_parameters: CnPrecisionControlParameters = DEFAULT_CN_PRECISION_CONTROL_PARAMETERS


class EcosystemDynamicsCnStateUpdate1State(NamedTuple):

    first_guardrail: EcosystemDynamicsCStateUpdate0PrecisionState
    xsmrpool_loss_patch: jnp.ndarray


class EcosystemDynamicsCnStateUpdate1Diagnostics(NamedTuple):
    state_update: CnStateUpdate1Diagnostics
    precision_control: CnPrecisionControlDiagnostics


class EcosystemDynamicsCnStateUpdate1Result(NamedTuple):
    state: EcosystemDynamicsCnStateUpdate1State
    diagnostics: EcosystemDynamicsCnStateUpdate1Diagnostics


def ecosystem_dynamics_cn_state_update1(
    inputs: EcosystemDynamicsCnStateUpdate1Inputs,
    state: EcosystemDynamicsCnStateUpdate1State,
) -> EcosystemDynamicsCnStateUpdate1Result:

    guarded = state.first_guardrail
    ecosystem = guarded.ecosystem
    pre_decomposition = ecosystem.pre_decomposition
    prefix = pre_decomposition.prefix
    middle = pre_decomposition.competition_decomposition
    canonical_cn_fluxes = prefix.cn_fluxes
    exact_inputs = inputs.state_update._replace(
        harvest_date_patch=prefix.crop_phenology.harvest_date_patch,
    )
    update = cn_state_update1_with_precision_control(
        exact_inputs,
        CnStateUpdate1State(
            vegetation=guarded.vegetation_carbon_nitrogen,
            xsmrpool_loss_patch=state.xsmrpool_loss_patch,
        ),
        canonical_cn_fluxes,
        inputs.state_update_parameters,
        inputs.precision_parameters,
    )

    final_cn_fluxes = update.cn_fluxes
    synchronized_prefix = prefix._replace(
        cn_fluxes=final_cn_fluxes,
        plant_nutrient_demand=prefix.plant_nutrient_demand._replace(
            cn_fluxes=final_cn_fluxes,
        ),
    )
    synchronized_middle = middle._replace(
        plant_nutrient_competition=middle.plant_nutrient_competition._replace(
            cn_fluxes=final_cn_fluxes,
        ),
        soil_biogeochem_decomp=middle.soil_biogeochem_decomp._replace(
            cn_fluxes=final_cn_fluxes,
        ),
    )
    synchronized_guarded = guarded._replace(
        ecosystem=ecosystem._replace(
            pre_decomposition=pre_decomposition._replace(
                prefix=synchronized_prefix,
                competition_decomposition=synchronized_middle,
            ),
        ),
        vegetation_carbon_nitrogen=update.state.vegetation,
    )
    return EcosystemDynamicsCnStateUpdate1Result(
        state=EcosystemDynamicsCnStateUpdate1State(
            first_guardrail=synchronized_guarded,
            xsmrpool_loss_patch=update.state.xsmrpool_loss_patch,
        ),
        diagnostics=EcosystemDynamicsCnStateUpdate1Diagnostics(
            state_update=update.update_diagnostics,
            precision_control=update.precision_diagnostics,
        ),
    )


class EcosystemDynamicsThroughSecondGuardrailInputs(NamedTuple):
    through_first_guardrail: EcosystemDynamicsThroughFirstGuardrailInputs
    state_update_and_precision: EcosystemDynamicsCnStateUpdate1Inputs


class EcosystemDynamicsThroughSecondGuardrailDiagnostics(NamedTuple):
    through_first_guardrail: EcosystemDynamicsThroughFirstGuardrailDiagnostics
    state_update_and_precision: EcosystemDynamicsCnStateUpdate1Diagnostics


class EcosystemDynamicsThroughSecondGuardrailResult(NamedTuple):
    state: EcosystemDynamicsCnStateUpdate1State
    diagnostics: EcosystemDynamicsThroughSecondGuardrailDiagnostics


def ecosystem_dynamics_through_second_guardrail(
    inputs: EcosystemDynamicsThroughSecondGuardrailInputs,
    state: EcosystemDynamicsCnStateUpdate1State,
) -> EcosystemDynamicsThroughSecondGuardrailResult:

    first = ecosystem_dynamics_through_first_guardrail(
        inputs.through_first_guardrail,
        state.first_guardrail,
    )
    second = ecosystem_dynamics_cn_state_update1(
        inputs.state_update_and_precision,
        state._replace(first_guardrail=first.state),
    )
    return EcosystemDynamicsThroughSecondGuardrailResult(
        state=second.state,
        diagnostics=EcosystemDynamicsThroughSecondGuardrailDiagnostics(
            through_first_guardrail=first.diagnostics,
            state_update_and_precision=second.diagnostics,
        ),
    )


class EcosystemDynamicsSoilNStateUpdate1Inputs(NamedTuple):

    state_update: SoilBiogeochemNStateUpdate1Inputs
    parameters: SoilBiogeochemNStateUpdate1Parameters = (
        DEFAULT_SOIL_BIOGEOCHEM_N_STATE_UPDATE1_PARAMETERS
    )


class EcosystemDynamicsSoilNStateUpdate1State(NamedTuple):

    second_guardrail: EcosystemDynamicsCnStateUpdate1State
    soil_nitrogen: SoilBiogeochemNStateUpdate1State


class EcosystemDynamicsSoilNStateUpdate1Diagnostics(NamedTuple):
    state_update: SoilBiogeochemNStateUpdate1Diagnostics


class EcosystemDynamicsSoilNStateUpdate1Result(NamedTuple):
    state: EcosystemDynamicsSoilNStateUpdate1State
    diagnostics: EcosystemDynamicsSoilNStateUpdate1Diagnostics


def ecosystem_dynamics_soil_n_state_update1(
    inputs: EcosystemDynamicsSoilNStateUpdate1Inputs,
    state: EcosystemDynamicsSoilNStateUpdate1State,
) -> EcosystemDynamicsSoilNStateUpdate1Result:

    second_guardrail = state.second_guardrail
    guarded = second_guardrail.first_guardrail
    ecosystem = guarded.ecosystem
    pre_decomposition = ecosystem.pre_decomposition
    prefix = pre_decomposition.prefix
    middle = pre_decomposition.competition_decomposition
    vertical_profile = prefix.soil_biogeochem_vertical_profile
    exact_inputs = inputs.state_update._replace(
        nitrogen_deposition_profile_column_level=(
            vertical_profile.nitrogen_deposition_profile_column_level
        ),
        nitrogen_fixation_profile_column_level=(
            vertical_profile.nitrogen_fixation_profile_column_level
        ),
    )
    update = soil_biogeochem_n_state_update1(
        exact_inputs,
        state.soil_nitrogen,
        prefix.cn_fluxes,
        inputs.parameters,
    )

    final_cn_fluxes = update.cn_fluxes
    synchronized_prefix = prefix._replace(
        cn_fluxes=final_cn_fluxes,
        plant_nutrient_demand=prefix.plant_nutrient_demand._replace(
            cn_fluxes=final_cn_fluxes,
        ),
    )
    synchronized_middle = middle._replace(
        plant_nutrient_competition=middle.plant_nutrient_competition._replace(
            cn_fluxes=final_cn_fluxes,
        ),
        soil_biogeochem_decomp=middle.soil_biogeochem_decomp._replace(
            cn_fluxes=final_cn_fluxes,
        ),
    )
    synchronized_second_guardrail = second_guardrail._replace(
        first_guardrail=guarded._replace(
            ecosystem=ecosystem._replace(
                pre_decomposition=pre_decomposition._replace(
                    prefix=synchronized_prefix,
                    competition_decomposition=synchronized_middle,
                ),
            ),
        ),
    )
    return EcosystemDynamicsSoilNStateUpdate1Result(
        state=EcosystemDynamicsSoilNStateUpdate1State(
            second_guardrail=synchronized_second_guardrail,
            soil_nitrogen=update.state,
        ),
        diagnostics=EcosystemDynamicsSoilNStateUpdate1Diagnostics(
            state_update=update.diagnostics,
        ),
    )


class EcosystemDynamicsThroughSoilNStateUpdate1Inputs(NamedTuple):
    through_second_guardrail: EcosystemDynamicsThroughSecondGuardrailInputs
    soil_n_state_update: EcosystemDynamicsSoilNStateUpdate1Inputs


class EcosystemDynamicsThroughSoilNStateUpdate1Diagnostics(NamedTuple):
    through_second_guardrail: EcosystemDynamicsThroughSecondGuardrailDiagnostics
    soil_n_state_update: EcosystemDynamicsSoilNStateUpdate1Diagnostics


class EcosystemDynamicsThroughSoilNStateUpdate1Result(NamedTuple):
    state: EcosystemDynamicsSoilNStateUpdate1State
    diagnostics: EcosystemDynamicsThroughSoilNStateUpdate1Diagnostics


def ecosystem_dynamics_through_soil_n_state_update1(
    inputs: EcosystemDynamicsThroughSoilNStateUpdate1Inputs,
    state: EcosystemDynamicsSoilNStateUpdate1State,
) -> EcosystemDynamicsThroughSoilNStateUpdate1Result:

    second_guardrail = ecosystem_dynamics_through_second_guardrail(
        inputs.through_second_guardrail,
        state.second_guardrail,
    )
    soil_n = ecosystem_dynamics_soil_n_state_update1(
        inputs.soil_n_state_update,
        state._replace(second_guardrail=second_guardrail.state),
    )
    return EcosystemDynamicsThroughSoilNStateUpdate1Result(
        state=soil_n.state,
        diagnostics=EcosystemDynamicsThroughSoilNStateUpdate1Diagnostics(
            through_second_guardrail=second_guardrail.diagnostics,
            soil_n_state_update=soil_n.diagnostics,
        ),
    )


class EcosystemDynamicsLitterVerticalTransportInputs(NamedTuple):

    vertical_transport: SoilBiogeochemLitterVerticalTransportInputs
    parameters: SoilBiogeochemLitterVerticalTransportParameters = (
        DEFAULT_SOIL_BIOGEOCHEM_LITTER_VERTICAL_TRANSPORT_PARAMETERS
    )


class EcosystemDynamicsLitterVerticalTransportState(NamedTuple):

    soil_n_state_update: EcosystemDynamicsSoilNStateUpdate1State
    vertical_transport: SoilBiogeochemLitterVerticalTransportState


class EcosystemDynamicsLitterVerticalTransportDiagnostics(NamedTuple):
    vertical_transport: SoilBiogeochemLitterVerticalTransportDiagnostics


class EcosystemDynamicsLitterVerticalTransportResult(NamedTuple):
    state: EcosystemDynamicsLitterVerticalTransportState
    diagnostics: EcosystemDynamicsLitterVerticalTransportDiagnostics


def ecosystem_dynamics_litter_vertical_transport(
    inputs: EcosystemDynamicsLitterVerticalTransportInputs,
    state: EcosystemDynamicsLitterVerticalTransportState,
) -> EcosystemDynamicsLitterVerticalTransportResult:

    soil_n_state_update = state.soil_n_state_update
    second_guardrail = soil_n_state_update.second_guardrail
    guarded = second_guardrail.first_guardrail
    ecosystem = guarded.ecosystem
    pre_decomposition = ecosystem.pre_decomposition
    prefix = pre_decomposition.prefix
    middle = pre_decomposition.competition_decomposition
    transport = soil_biogeochem_litter_vertical_transport(
        inputs.vertical_transport,
        state.vertical_transport,
        prefix.cn_fluxes,
        inputs.parameters,
    )

    final_cn_fluxes = transport.cn_fluxes
    synchronized_prefix = prefix._replace(
        cn_fluxes=final_cn_fluxes,
        plant_nutrient_demand=prefix.plant_nutrient_demand._replace(
            cn_fluxes=final_cn_fluxes,
        ),
    )
    synchronized_middle = middle._replace(
        plant_nutrient_competition=middle.plant_nutrient_competition._replace(
            cn_fluxes=final_cn_fluxes,
        ),
        soil_biogeochem_decomp=middle.soil_biogeochem_decomp._replace(
            cn_fluxes=final_cn_fluxes,
        ),
    )
    synchronized_soil_n_state_update = soil_n_state_update._replace(
        second_guardrail=second_guardrail._replace(
            first_guardrail=guarded._replace(
                ecosystem=ecosystem._replace(
                    pre_decomposition=pre_decomposition._replace(
                        prefix=synchronized_prefix,
                        competition_decomposition=synchronized_middle,
                    ),
                ),
            ),
        ),
    )
    return EcosystemDynamicsLitterVerticalTransportResult(
        state=EcosystemDynamicsLitterVerticalTransportState(
            soil_n_state_update=synchronized_soil_n_state_update,
            vertical_transport=transport.state,
        ),
        diagnostics=EcosystemDynamicsLitterVerticalTransportDiagnostics(
            vertical_transport=transport.diagnostics,
        ),
    )


class EcosystemDynamicsThroughLitterVerticalTransportInputs(NamedTuple):
    through_soil_n_state_update: EcosystemDynamicsThroughSoilNStateUpdate1Inputs
    litter_vertical_transport: EcosystemDynamicsLitterVerticalTransportInputs


class EcosystemDynamicsThroughLitterVerticalTransportDiagnostics(NamedTuple):
    through_soil_n_state_update: EcosystemDynamicsThroughSoilNStateUpdate1Diagnostics
    litter_vertical_transport: EcosystemDynamicsLitterVerticalTransportDiagnostics


class EcosystemDynamicsThroughLitterVerticalTransportResult(NamedTuple):
    state: EcosystemDynamicsLitterVerticalTransportState
    diagnostics: EcosystemDynamicsThroughLitterVerticalTransportDiagnostics


def ecosystem_dynamics_through_litter_vertical_transport(
    inputs: EcosystemDynamicsThroughLitterVerticalTransportInputs,
    state: EcosystemDynamicsLitterVerticalTransportState,
) -> EcosystemDynamicsThroughLitterVerticalTransportResult:

    soil_n = ecosystem_dynamics_through_soil_n_state_update1(
        inputs.through_soil_n_state_update,
        state.soil_n_state_update,
    )
    transport = ecosystem_dynamics_litter_vertical_transport(
        inputs.litter_vertical_transport,
        state._replace(soil_n_state_update=soil_n.state),
    )
    return EcosystemDynamicsThroughLitterVerticalTransportResult(
        state=transport.state,
        diagnostics=EcosystemDynamicsThroughLitterVerticalTransportDiagnostics(
            through_soil_n_state_update=soil_n.diagnostics,
            litter_vertical_transport=transport.diagnostics,
        ),
    )


class EcosystemDynamicsGapMortalityInputs(NamedTuple):

    gap_mortality: CnGapMortalityInputs
    parameters: CnGapMortalityParameters


class EcosystemDynamicsGapMortalityState(NamedTuple):

    litter_vertical_transport: EcosystemDynamicsLitterVerticalTransportState
    gap_mortality: CnGapMortalityState


class EcosystemDynamicsGapMortalityDiagnostics(NamedTuple):
    gap_mortality: CnGapMortalityDiagnostics


class EcosystemDynamicsGapMortalityResult(NamedTuple):
    state: EcosystemDynamicsGapMortalityState
    diagnostics: EcosystemDynamicsGapMortalityDiagnostics


def ecosystem_dynamics_gap_mortality(
    inputs: EcosystemDynamicsGapMortalityInputs,
    state: EcosystemDynamicsGapMortalityState,
) -> EcosystemDynamicsGapMortalityResult:

    transport_state = state.litter_vertical_transport
    soil_n_state_update = transport_state.soil_n_state_update
    second_guardrail = soil_n_state_update.second_guardrail
    guarded = second_guardrail.first_guardrail
    ecosystem = guarded.ecosystem
    pre_decomposition = ecosystem.pre_decomposition
    prefix = pre_decomposition.prefix
    middle = pre_decomposition.competition_decomposition
    profiles = prefix.soil_biogeochem_vertical_profile
    exact_inputs = inputs.gap_mortality._replace(
        vegetation_state=guarded.vegetation_carbon_nitrogen,
        leaf_profile_patch_level=profiles.leaf_profile_patch_level,
        fine_root_profile_patch_level=profiles.fine_root_profile_patch_level,
        coarse_root_profile_patch_level=profiles.coarse_root_profile_patch_level,
        stem_profile_patch_level=profiles.stem_profile_patch_level,
    )
    mortality = cn_gap_mortality(
        exact_inputs,
        state.gap_mortality,
        prefix.cn_fluxes,
        inputs.parameters,
    )

    final_cn_fluxes = mortality.cn_fluxes
    synchronized_prefix = prefix._replace(
        cn_fluxes=final_cn_fluxes,
        plant_nutrient_demand=prefix.plant_nutrient_demand._replace(
            cn_fluxes=final_cn_fluxes,
        ),
    )
    synchronized_middle = middle._replace(
        plant_nutrient_competition=middle.plant_nutrient_competition._replace(
            cn_fluxes=final_cn_fluxes,
        ),
        soil_biogeochem_decomp=middle.soil_biogeochem_decomp._replace(
            cn_fluxes=final_cn_fluxes,
        ),
    )
    synchronized_transport_state = transport_state._replace(
        soil_n_state_update=soil_n_state_update._replace(
            second_guardrail=second_guardrail._replace(
                first_guardrail=guarded._replace(
                    ecosystem=ecosystem._replace(
                        pre_decomposition=pre_decomposition._replace(
                            prefix=synchronized_prefix,
                            competition_decomposition=synchronized_middle,
                        ),
                    ),
                ),
            ),
        ),
    )
    return EcosystemDynamicsGapMortalityResult(
        state=EcosystemDynamicsGapMortalityState(
            litter_vertical_transport=synchronized_transport_state,
            gap_mortality=mortality.state,
        ),
        diagnostics=EcosystemDynamicsGapMortalityDiagnostics(
            gap_mortality=mortality.diagnostics,
        ),
    )


class EcosystemDynamicsThroughGapMortalityInputs(NamedTuple):
    through_litter_vertical_transport: EcosystemDynamicsThroughLitterVerticalTransportInputs
    gap_mortality: EcosystemDynamicsGapMortalityInputs


class EcosystemDynamicsThroughGapMortalityDiagnostics(NamedTuple):
    through_litter_vertical_transport: (
        EcosystemDynamicsThroughLitterVerticalTransportDiagnostics
    )
    gap_mortality: EcosystemDynamicsGapMortalityDiagnostics


class EcosystemDynamicsThroughGapMortalityResult(NamedTuple):
    state: EcosystemDynamicsGapMortalityState
    diagnostics: EcosystemDynamicsThroughGapMortalityDiagnostics


def ecosystem_dynamics_through_gap_mortality(
    inputs: EcosystemDynamicsThroughGapMortalityInputs,
    state: EcosystemDynamicsGapMortalityState,
) -> EcosystemDynamicsThroughGapMortalityResult:

    transport = ecosystem_dynamics_through_litter_vertical_transport(
        inputs.through_litter_vertical_transport,
        state.litter_vertical_transport,
    )
    mortality = ecosystem_dynamics_gap_mortality(
        inputs.gap_mortality,
        state._replace(litter_vertical_transport=transport.state),
    )
    return EcosystemDynamicsThroughGapMortalityResult(
        state=mortality.state,
        diagnostics=EcosystemDynamicsThroughGapMortalityDiagnostics(
            through_litter_vertical_transport=transport.diagnostics,
            gap_mortality=mortality.diagnostics,
        ),
    )


class EcosystemDynamicsCnStateUpdate2Inputs(NamedTuple):

    state_update: CnStateUpdate2Inputs
    parameters: CnStateUpdate2Parameters = DEFAULT_CN_STATE_UPDATE2_PARAMETERS


class EcosystemDynamicsCnStateUpdate2State(NamedTuple):

    gap_mortality: EcosystemDynamicsGapMortalityState


class EcosystemDynamicsCnStateUpdate2Diagnostics(NamedTuple):
    state_update: CnStateUpdate2Diagnostics


class EcosystemDynamicsCnStateUpdate2Result(NamedTuple):
    state: EcosystemDynamicsCnStateUpdate2State
    diagnostics: EcosystemDynamicsCnStateUpdate2Diagnostics


def ecosystem_dynamics_cn_state_update2(
    inputs: EcosystemDynamicsCnStateUpdate2Inputs,
    state: EcosystemDynamicsCnStateUpdate2State,
) -> EcosystemDynamicsCnStateUpdate2Result:

    gap_state = state.gap_mortality
    transport = gap_state.litter_vertical_transport
    soil_n = transport.soil_n_state_update
    second_guardrail = soil_n.second_guardrail
    guarded = second_guardrail.first_guardrail
    ecosystem = guarded.ecosystem
    pre_decomposition = ecosystem.pre_decomposition
    prefix = pre_decomposition.prefix
    middle = pre_decomposition.competition_decomposition
    update = cn_state_update2(
        inputs.state_update,
        CnStateUpdate2State(
            vegetation=guarded.vegetation_carbon_nitrogen,
            soil_carbon_pool_column_level_pool=(
                transport.vertical_transport.carbon_pool_column_level_pool
            ),
            soil_nitrogen_pool_column_level_pool=(
                transport.vertical_transport.nitrogen_pool_column_level_pool
            ),
        ),
        prefix.cn_fluxes,
        inputs.parameters,
    )

    final_cn_fluxes = update.cn_fluxes
    synchronized_prefix = prefix._replace(
        cn_fluxes=final_cn_fluxes,
        plant_nutrient_demand=prefix.plant_nutrient_demand._replace(
            cn_fluxes=final_cn_fluxes,
        ),
    )
    synchronized_middle = middle._replace(
        plant_nutrient_competition=middle.plant_nutrient_competition._replace(
            cn_fluxes=final_cn_fluxes,
        ),
        soil_biogeochem_decomp=middle.soil_biogeochem_decomp._replace(
            cn_fluxes=final_cn_fluxes,
        ),
    )
    synchronized_transport = transport._replace(
        soil_n_state_update=soil_n._replace(
            second_guardrail=second_guardrail._replace(
                first_guardrail=guarded._replace(
                    ecosystem=ecosystem._replace(
                        pre_decomposition=pre_decomposition._replace(
                            prefix=synchronized_prefix,
                            competition_decomposition=synchronized_middle,
                        ),
                    ),
                    vegetation_carbon_nitrogen=update.state.vegetation,
                ),
            ),
        ),
        vertical_transport=transport.vertical_transport._replace(
            carbon_pool_column_level_pool=(
                update.state.soil_carbon_pool_column_level_pool
            ),
            nitrogen_pool_column_level_pool=(
                update.state.soil_nitrogen_pool_column_level_pool
            ),
        ),
    )
    return EcosystemDynamicsCnStateUpdate2Result(
        state=EcosystemDynamicsCnStateUpdate2State(
            gap_mortality=gap_state._replace(
                litter_vertical_transport=synchronized_transport,
            ),
        ),
        diagnostics=EcosystemDynamicsCnStateUpdate2Diagnostics(
            state_update=update.diagnostics,
        ),
    )


class EcosystemDynamicsThroughCnStateUpdate2Inputs(NamedTuple):
    through_gap_mortality: EcosystemDynamicsThroughGapMortalityInputs
    state_update: EcosystemDynamicsCnStateUpdate2Inputs


class EcosystemDynamicsThroughCnStateUpdate2Diagnostics(NamedTuple):
    through_gap_mortality: EcosystemDynamicsThroughGapMortalityDiagnostics
    state_update: EcosystemDynamicsCnStateUpdate2Diagnostics


class EcosystemDynamicsThroughCnStateUpdate2Result(NamedTuple):
    state: EcosystemDynamicsCnStateUpdate2State
    diagnostics: EcosystemDynamicsThroughCnStateUpdate2Diagnostics


def ecosystem_dynamics_through_cn_state_update2(
    inputs: EcosystemDynamicsThroughCnStateUpdate2Inputs,
    state: EcosystemDynamicsCnStateUpdate2State,
) -> EcosystemDynamicsThroughCnStateUpdate2Result:

    gap = ecosystem_dynamics_through_gap_mortality(
        inputs.through_gap_mortality,
        state.gap_mortality,
    )
    update = ecosystem_dynamics_cn_state_update2(
        inputs.state_update,
        state._replace(gap_mortality=gap.state),
    )
    return EcosystemDynamicsThroughCnStateUpdate2Result(
        state=update.state,
        diagnostics=EcosystemDynamicsThroughCnStateUpdate2Diagnostics(
            through_gap_mortality=gap.diagnostics,
            state_update=update.diagnostics,
        ),
    )


class EcosystemDynamicsCnStateUpdate2hPrecisionInputs(NamedTuple):

    state_update: CnStateUpdate2Inputs
    precision: CnPrecisionControlInputs
    state_update_parameters: CnStateUpdate2Parameters = (
        DEFAULT_CN_STATE_UPDATE2_PARAMETERS
    )
    precision_parameters: CnPrecisionControlParameters = (
        DEFAULT_CN_PRECISION_CONTROL_PARAMETERS
    )


class EcosystemDynamicsCnStateUpdate2hPrecisionState(NamedTuple):

    state_update2: EcosystemDynamicsCnStateUpdate2State


class EcosystemDynamicsCnStateUpdate2hPrecisionDiagnostics(NamedTuple):
    state_update: CnStateUpdate2Diagnostics
    precision: CnPrecisionControlDiagnostics


class EcosystemDynamicsCnStateUpdate2hPrecisionResult(NamedTuple):
    state: EcosystemDynamicsCnStateUpdate2hPrecisionState
    diagnostics: EcosystemDynamicsCnStateUpdate2hPrecisionDiagnostics


def ecosystem_dynamics_cn_state_update2h_precision(
    inputs: EcosystemDynamicsCnStateUpdate2hPrecisionInputs,
    state: EcosystemDynamicsCnStateUpdate2hPrecisionState,
) -> EcosystemDynamicsCnStateUpdate2hPrecisionResult:

    update2_state = state.state_update2
    gap_state = update2_state.gap_mortality
    transport = gap_state.litter_vertical_transport
    soil_n = transport.soil_n_state_update
    second_guardrail = soil_n.second_guardrail
    guarded = second_guardrail.first_guardrail
    ecosystem = guarded.ecosystem
    pre_decomposition = ecosystem.pre_decomposition
    prefix = pre_decomposition.prefix
    middle = pre_decomposition.competition_decomposition
    update = cn_state_update2h_with_precision_control(
        inputs.state_update,
        CnStateUpdate2State(
            vegetation=guarded.vegetation_carbon_nitrogen,
            soil_carbon_pool_column_level_pool=(
                transport.vertical_transport.carbon_pool_column_level_pool
            ),
            soil_nitrogen_pool_column_level_pool=(
                transport.vertical_transport.nitrogen_pool_column_level_pool
            ),
        ),
        prefix.cn_fluxes,
        inputs.state_update_parameters,
        inputs.precision,
        inputs.precision_parameters,
    )

    final_cn_fluxes = update.cn_fluxes
    synchronized_prefix = prefix._replace(
        cn_fluxes=final_cn_fluxes,
        plant_nutrient_demand=prefix.plant_nutrient_demand._replace(
            cn_fluxes=final_cn_fluxes,
        ),
    )
    synchronized_middle = middle._replace(
        plant_nutrient_competition=middle.plant_nutrient_competition._replace(
            cn_fluxes=final_cn_fluxes,
        ),
        soil_biogeochem_decomp=middle.soil_biogeochem_decomp._replace(
            cn_fluxes=final_cn_fluxes,
        ),
    )
    synchronized_transport = transport._replace(
        soil_n_state_update=soil_n._replace(
            second_guardrail=second_guardrail._replace(
                first_guardrail=guarded._replace(
                    ecosystem=ecosystem._replace(
                        pre_decomposition=pre_decomposition._replace(
                            prefix=synchronized_prefix,
                            competition_decomposition=synchronized_middle,
                        ),
                    ),
                    vegetation_carbon_nitrogen=update.state.vegetation,
                ),
            ),
        ),
        vertical_transport=transport.vertical_transport._replace(
            carbon_pool_column_level_pool=(
                update.state.soil_carbon_pool_column_level_pool
            ),
            nitrogen_pool_column_level_pool=(
                update.state.soil_nitrogen_pool_column_level_pool
            ),
        ),
    )
    return EcosystemDynamicsCnStateUpdate2hPrecisionResult(
        state=EcosystemDynamicsCnStateUpdate2hPrecisionState(
            state_update2=update2_state._replace(
                gap_mortality=gap_state._replace(
                    litter_vertical_transport=synchronized_transport,
                ),
            ),
        ),
        diagnostics=EcosystemDynamicsCnStateUpdate2hPrecisionDiagnostics(
            state_update=update.update_diagnostics,
            precision=update.precision_diagnostics,
        ),
    )


class EcosystemDynamicsThroughCnStateUpdate2hPrecisionInputs(NamedTuple):
    through_state_update2: EcosystemDynamicsThroughCnStateUpdate2Inputs
    state_update2h_precision: EcosystemDynamicsCnStateUpdate2hPrecisionInputs


class EcosystemDynamicsThroughCnStateUpdate2hPrecisionDiagnostics(NamedTuple):
    through_state_update2: EcosystemDynamicsThroughCnStateUpdate2Diagnostics
    state_update2h_precision: EcosystemDynamicsCnStateUpdate2hPrecisionDiagnostics


class EcosystemDynamicsThroughCnStateUpdate2hPrecisionResult(NamedTuple):
    state: EcosystemDynamicsCnStateUpdate2hPrecisionState
    diagnostics: EcosystemDynamicsThroughCnStateUpdate2hPrecisionDiagnostics


def ecosystem_dynamics_through_cn_state_update2h_precision(
    inputs: EcosystemDynamicsThroughCnStateUpdate2hPrecisionInputs,
    state: EcosystemDynamicsCnStateUpdate2hPrecisionState,
) -> EcosystemDynamicsThroughCnStateUpdate2hPrecisionResult:

    update2 = ecosystem_dynamics_through_cn_state_update2(
        inputs.through_state_update2,
        state.state_update2,
    )
    update2h = ecosystem_dynamics_cn_state_update2h_precision(
        inputs.state_update2h_precision,
        state._replace(state_update2=update2.state),
    )
    return EcosystemDynamicsThroughCnStateUpdate2hPrecisionResult(
        state=update2h.state,
        diagnostics=EcosystemDynamicsThroughCnStateUpdate2hPrecisionDiagnostics(
            through_state_update2=update2.diagnostics,
            state_update2h_precision=update2h.diagnostics,
        ),
    )


class EcosystemDynamicsWoodProductsInputs(NamedTuple):
    wood_products: CnWoodProductsInputs
    parameters: CnWoodProductsParameters


class EcosystemDynamicsWoodProductsState(NamedTuple):
    state_update2h_precision: EcosystemDynamicsCnStateUpdate2hPrecisionState
    wood_products: CnWoodProductsState


class EcosystemDynamicsWoodProductsDiagnostics(NamedTuple):
    wood_products: CnWoodProductsDiagnostics


class EcosystemDynamicsWoodProductsResult(NamedTuple):
    state: EcosystemDynamicsWoodProductsState
    diagnostics: EcosystemDynamicsWoodProductsDiagnostics


def ecosystem_dynamics_wood_products(
    inputs: EcosystemDynamicsWoodProductsInputs,
    state: EcosystemDynamicsWoodProductsState,
) -> EcosystemDynamicsWoodProductsResult:

    update2h_state = state.state_update2h_precision
    update2_state = update2h_state.state_update2
    gap_state = update2_state.gap_mortality
    transport = gap_state.litter_vertical_transport
    guarded = transport.soil_n_state_update.second_guardrail.first_guardrail
    prefix = guarded.ecosystem.pre_decomposition.prefix
    products = cn_wood_products(
        inputs.wood_products,
        state.wood_products,
        prefix.cn_fluxes,
        inputs.parameters,
    )
    return EcosystemDynamicsWoodProductsResult(
        state=EcosystemDynamicsWoodProductsState(
            state_update2h_precision=update2h_state,
            wood_products=products.state,
        ),
        diagnostics=EcosystemDynamicsWoodProductsDiagnostics(
            wood_products=products.diagnostics,
        ),
    )


class EcosystemDynamicsThroughWoodProductsInputs(NamedTuple):
    through_state_update2h_precision: (
        EcosystemDynamicsThroughCnStateUpdate2hPrecisionInputs
    )
    wood_products: EcosystemDynamicsWoodProductsInputs


class EcosystemDynamicsThroughWoodProductsDiagnostics(NamedTuple):
    through_state_update2h_precision: (
        EcosystemDynamicsThroughCnStateUpdate2hPrecisionDiagnostics
    )
    wood_products: EcosystemDynamicsWoodProductsDiagnostics


class EcosystemDynamicsThroughWoodProductsResult(NamedTuple):
    state: EcosystemDynamicsWoodProductsState
    diagnostics: EcosystemDynamicsThroughWoodProductsDiagnostics


def ecosystem_dynamics_through_wood_products(
    inputs: EcosystemDynamicsThroughWoodProductsInputs,
    state: EcosystemDynamicsWoodProductsState,
) -> EcosystemDynamicsThroughWoodProductsResult:

    update2h = ecosystem_dynamics_through_cn_state_update2h_precision(
        inputs.through_state_update2h_precision,
        state.state_update2h_precision,
    )
    products = ecosystem_dynamics_wood_products(
        inputs.wood_products,
        state._replace(state_update2h_precision=update2h.state),
    )
    return EcosystemDynamicsThroughWoodProductsResult(
        state=products.state,
        diagnostics=EcosystemDynamicsThroughWoodProductsDiagnostics(
            through_state_update2h_precision=update2h.diagnostics,
            wood_products=products.diagnostics,
        ),
    )


class EcosystemDynamicsFireAreaInputs(NamedTuple):
    topology: CnFireAreaTopology
    forcing: CnFireAreaForcing
    parameters: CnFireAreaParameters


class EcosystemDynamicsFireAreaState(NamedTuple):
    wood_products: EcosystemDynamicsWoodProductsState
    fire_area: CnFireAreaState


class EcosystemDynamicsFireAreaDiagnostics(NamedTuple):
    fire_area: CnFireAreaDiagnostics


class EcosystemDynamicsFireAreaResult(NamedTuple):
    state: EcosystemDynamicsFireAreaState
    diagnostics: EcosystemDynamicsFireAreaDiagnostics


def ecosystem_dynamics_fire_area(
    inputs: EcosystemDynamicsFireAreaInputs,
    state: EcosystemDynamicsFireAreaState,
) -> EcosystemDynamicsFireAreaResult:

    wood_state = state.wood_products
    update2h_state = wood_state.state_update2h_precision
    update2_state = update2h_state.state_update2
    transport = update2_state.gap_mortality.litter_vertical_transport
    guarded = transport.soil_n_state_update.second_guardrail.first_guardrail
    exact_forcing = inputs.forcing._replace(
        decomposition_carbon_pool_column_level_pool=(
            transport.vertical_transport.carbon_pool_column_level_pool
        ),
    )
    fire = cn_fire_area_li2016_fixed_landcover(
        inputs.topology,
        exact_forcing,
        guarded.vegetation_carbon_nitrogen,
        state.fire_area,
        inputs.parameters,
    )
    return EcosystemDynamicsFireAreaResult(
        state=EcosystemDynamicsFireAreaState(
            wood_products=wood_state,
            fire_area=fire.state,
        ),
        diagnostics=EcosystemDynamicsFireAreaDiagnostics(
            fire_area=fire.diagnostics,
        ),
    )


class EcosystemDynamicsThroughFireAreaInputs(NamedTuple):
    through_wood_products: EcosystemDynamicsThroughWoodProductsInputs
    fire_area: EcosystemDynamicsFireAreaInputs


class EcosystemDynamicsThroughFireAreaDiagnostics(NamedTuple):
    through_wood_products: EcosystemDynamicsThroughWoodProductsDiagnostics
    fire_area: EcosystemDynamicsFireAreaDiagnostics


class EcosystemDynamicsThroughFireAreaResult(NamedTuple):
    state: EcosystemDynamicsFireAreaState
    diagnostics: EcosystemDynamicsThroughFireAreaDiagnostics


def ecosystem_dynamics_through_fire_area(
    inputs: EcosystemDynamicsThroughFireAreaInputs,
    state: EcosystemDynamicsFireAreaState,
) -> EcosystemDynamicsThroughFireAreaResult:

    products = ecosystem_dynamics_through_wood_products(
        inputs.through_wood_products,
        state.wood_products,
    )
    fire = ecosystem_dynamics_fire_area(
        inputs.fire_area,
        state._replace(wood_products=products.state),
    )
    return EcosystemDynamicsThroughFireAreaResult(
        state=fire.state,
        diagnostics=EcosystemDynamicsThroughFireAreaDiagnostics(
            through_wood_products=products.diagnostics,
            fire_area=fire.diagnostics,
        ),
    )


class EcosystemDynamicsFireFluxInputs(NamedTuple):
    fire_flux: CnFireFluxInputs
    parameters: CnFireFluxParameters


class EcosystemDynamicsFireFluxState(NamedTuple):
    fire_area: EcosystemDynamicsFireAreaState
    fire_flux: CnFireFluxState


class EcosystemDynamicsFireFluxDiagnostics(NamedTuple):
    fire_flux: CnFireFluxDiagnostics


class EcosystemDynamicsFireFluxResult(NamedTuple):
    state: EcosystemDynamicsFireFluxState
    diagnostics: EcosystemDynamicsFireFluxDiagnostics


def ecosystem_dynamics_fire_flux(
    inputs: EcosystemDynamicsFireFluxInputs,
    state: EcosystemDynamicsFireFluxState,
) -> EcosystemDynamicsFireFluxResult:

    fire_area_state = state.fire_area
    area = fire_area_state.fire_area
    wood_state = fire_area_state.wood_products
    update2h_state = wood_state.state_update2h_precision
    transport = (
        update2h_state.state_update2.gap_mortality.litter_vertical_transport
    )
    soil_n = transport.soil_n_state_update
    guarded = soil_n.second_guardrail.first_guardrail
    pre_decomposition = guarded.ecosystem.pre_decomposition
    prefix = pre_decomposition.prefix
    middle = pre_decomposition.competition_decomposition
    profiles = prefix.soil_biogeochem_vertical_profile
    exact_inputs = inputs.fire_flux._replace(
        vegetation_state=guarded.vegetation_carbon_nitrogen,
        crop_fraction_column=area.crop_fraction_column,
        total_burned_area_fraction_column=area.total_burned_area_fraction_column,
        conversion_burned_area_fraction_column=(
            area.conversion_burned_area_fraction_column
        ),
        conversion_fire_burned_area_fraction_column=(
            area.conversion_fire_burned_area_fraction_column
        ),
        crop_burned_area_fraction_column=area.crop_burned_area_fraction_column,
        peat_burned_area_fraction_column=area.peat_burned_area_fraction_column,
        tropical_evergreen_tree_fraction_column=(
            area.tropical_evergreen_tree_fraction_column
        ),
        tropical_deciduous_tree_fraction_column=(
            area.tropical_deciduous_tree_fraction_column
        ),
        decreased_tropical_tree_fraction_column=(
            area.decreased_tropical_tree_fraction_column
        ),
        leaf_profile_patch_level=profiles.leaf_profile_patch_level,
        fine_root_profile_patch_level=profiles.fine_root_profile_patch_level,
        coarse_root_profile_patch_level=profiles.coarse_root_profile_patch_level,
        stem_profile_patch_level=profiles.stem_profile_patch_level,
        decomposition_carbon_pool_column_level_pool=(
            transport.vertical_transport.carbon_pool_column_level_pool
        ),
        decomposition_nitrogen_pool_column_level_pool=(
            transport.vertical_transport.nitrogen_pool_column_level_pool
        ),
    )
    fire = cn_fire_fluxes(
        exact_inputs,
        state.fire_flux._replace(
            unburned_conversion_fraction_column=(
                area.unburned_conversion_fraction_column
            ),
        ),
        prefix.cn_fluxes,
        inputs.parameters,
    )

    final_cn_fluxes = fire.cn_fluxes
    synchronized_prefix = prefix._replace(
        cn_fluxes=final_cn_fluxes,
        plant_nutrient_demand=prefix.plant_nutrient_demand._replace(
            cn_fluxes=final_cn_fluxes,
        ),
    )
    synchronized_middle = middle._replace(
        plant_nutrient_competition=middle.plant_nutrient_competition._replace(
            cn_fluxes=final_cn_fluxes,
        ),
        soil_biogeochem_decomp=middle.soil_biogeochem_decomp._replace(
            cn_fluxes=final_cn_fluxes,
        ),
    )
    synchronized_transport = transport._replace(
        soil_n_state_update=soil_n._replace(
            second_guardrail=soil_n.second_guardrail._replace(
                first_guardrail=guarded._replace(
                    ecosystem=guarded.ecosystem._replace(
                        pre_decomposition=pre_decomposition._replace(
                            prefix=synchronized_prefix,
                            competition_decomposition=synchronized_middle,
                        ),
                    ),
                ),
            ),
        ),
    )
    synchronized_update2h = update2h_state._replace(
        state_update2=update2h_state.state_update2._replace(
            gap_mortality=(
                update2h_state.state_update2.gap_mortality._replace(
                    litter_vertical_transport=synchronized_transport,
                )
            ),
        ),
    )
    synchronized_fire_area = fire_area_state._replace(
        wood_products=wood_state._replace(
            state_update2h_precision=synchronized_update2h,
        ),
        fire_area=area._replace(
            unburned_conversion_fraction_column=(
                fire.state.unburned_conversion_fraction_column
            ),
        ),
    )
    return EcosystemDynamicsFireFluxResult(
        state=EcosystemDynamicsFireFluxState(
            fire_area=synchronized_fire_area,
            fire_flux=fire.state,
        ),
        diagnostics=EcosystemDynamicsFireFluxDiagnostics(
            fire_flux=fire.diagnostics,
        ),
    )


class EcosystemDynamicsThroughFireFluxInputs(NamedTuple):
    through_fire_area: EcosystemDynamicsThroughFireAreaInputs
    fire_flux: EcosystemDynamicsFireFluxInputs


class EcosystemDynamicsThroughFireFluxDiagnostics(NamedTuple):
    through_fire_area: EcosystemDynamicsThroughFireAreaDiagnostics
    fire_flux: EcosystemDynamicsFireFluxDiagnostics


class EcosystemDynamicsThroughFireFluxResult(NamedTuple):
    state: EcosystemDynamicsFireFluxState
    diagnostics: EcosystemDynamicsThroughFireFluxDiagnostics


def ecosystem_dynamics_through_fire_flux(
    inputs: EcosystemDynamicsThroughFireFluxInputs,
    state: EcosystemDynamicsFireFluxState,
) -> EcosystemDynamicsThroughFireFluxResult:

    area = ecosystem_dynamics_through_fire_area(
        inputs.through_fire_area,
        state.fire_area,
    )
    fire = ecosystem_dynamics_fire_flux(
        inputs.fire_flux,
        state._replace(fire_area=area.state),
    )
    return EcosystemDynamicsThroughFireFluxResult(
        state=fire.state,
        diagnostics=EcosystemDynamicsThroughFireFluxDiagnostics(
            through_fire_area=area.diagnostics,
            fire_flux=fire.diagnostics,
        ),
    )


class EcosystemDynamicsCnStateUpdate3PrecisionInputs(NamedTuple):
    state_update: CnStateUpdate2Inputs
    precision: CnPrecisionControlInputs
    state_update_parameters: CnStateUpdate3Parameters = (
        DEFAULT_CN_STATE_UPDATE3_PARAMETERS
    )
    precision_parameters: CnPrecisionControlParameters = (
        DEFAULT_CN_PRECISION_CONTROL_PARAMETERS
    )


class EcosystemDynamicsCnStateUpdate3PrecisionState(NamedTuple):
    fire_flux: EcosystemDynamicsFireFluxState


class EcosystemDynamicsCnStateUpdate3PrecisionDiagnostics(NamedTuple):
    state_update: CnStateUpdate3Diagnostics
    precision: CnPrecisionControlDiagnostics


class EcosystemDynamicsCnStateUpdate3PrecisionResult(NamedTuple):
    state: EcosystemDynamicsCnStateUpdate3PrecisionState
    diagnostics: EcosystemDynamicsCnStateUpdate3PrecisionDiagnostics


def ecosystem_dynamics_cn_state_update3_precision(
    inputs: EcosystemDynamicsCnStateUpdate3PrecisionInputs,
    state: EcosystemDynamicsCnStateUpdate3PrecisionState,
) -> EcosystemDynamicsCnStateUpdate3PrecisionResult:

    fire_flux_state = state.fire_flux
    fire_area_state = fire_flux_state.fire_area
    wood_state = fire_area_state.wood_products
    update2h_state = wood_state.state_update2h_precision
    update2_state = update2h_state.state_update2
    gap_state = update2_state.gap_mortality
    transport = gap_state.litter_vertical_transport
    soil_n = transport.soil_n_state_update
    second_guardrail = soil_n.second_guardrail
    guarded = second_guardrail.first_guardrail
    ecosystem = guarded.ecosystem
    pre_decomposition = ecosystem.pre_decomposition
    prefix = pre_decomposition.prefix
    middle = pre_decomposition.competition_decomposition
    update = cn_state_update3_with_precision_control(
        inputs.state_update,
        CnStateUpdate2State(
            vegetation=guarded.vegetation_carbon_nitrogen,
            soil_carbon_pool_column_level_pool=(
                transport.vertical_transport.carbon_pool_column_level_pool
            ),
            soil_nitrogen_pool_column_level_pool=(
                transport.vertical_transport.nitrogen_pool_column_level_pool
            ),
        ),
        prefix.cn_fluxes,
        inputs.state_update_parameters,
        inputs.precision,
        inputs.precision_parameters,
    )

    final_cn_fluxes = update.cn_fluxes
    synchronized_prefix = prefix._replace(
        cn_fluxes=final_cn_fluxes,
        plant_nutrient_demand=prefix.plant_nutrient_demand._replace(
            cn_fluxes=final_cn_fluxes,
        ),
    )
    synchronized_middle = middle._replace(
        plant_nutrient_competition=middle.plant_nutrient_competition._replace(
            cn_fluxes=final_cn_fluxes,
        ),
        soil_biogeochem_decomp=middle.soil_biogeochem_decomp._replace(
            cn_fluxes=final_cn_fluxes,
        ),
    )
    synchronized_transport = transport._replace(
        soil_n_state_update=soil_n._replace(
            second_guardrail=second_guardrail._replace(
                first_guardrail=guarded._replace(
                    ecosystem=ecosystem._replace(
                        pre_decomposition=pre_decomposition._replace(
                            prefix=synchronized_prefix,
                            competition_decomposition=synchronized_middle,
                        ),
                    ),
                    vegetation_carbon_nitrogen=update.state.vegetation,
                ),
            ),
        ),
        vertical_transport=transport.vertical_transport._replace(
            carbon_pool_column_level_pool=(
                update.state.soil_carbon_pool_column_level_pool
            ),
            nitrogen_pool_column_level_pool=(
                update.state.soil_nitrogen_pool_column_level_pool
            ),
        ),
    )
    synchronized_update2 = update2_state._replace(
        gap_mortality=gap_state._replace(
            litter_vertical_transport=synchronized_transport,
        ),
    )
    synchronized_fire_flux = fire_flux_state._replace(
        fire_area=fire_area_state._replace(
            wood_products=wood_state._replace(
                state_update2h_precision=update2h_state._replace(
                    state_update2=synchronized_update2,
                ),
            ),
        ),
    )
    return EcosystemDynamicsCnStateUpdate3PrecisionResult(
        state=EcosystemDynamicsCnStateUpdate3PrecisionState(
            fire_flux=synchronized_fire_flux,
        ),
        diagnostics=EcosystemDynamicsCnStateUpdate3PrecisionDiagnostics(
            state_update=update.update_diagnostics,
            precision=update.precision_diagnostics,
        ),
    )


class EcosystemDynamicsThroughCnStateUpdate3PrecisionInputs(NamedTuple):
    through_fire_flux: EcosystemDynamicsThroughFireFluxInputs
    state_update3_precision: EcosystemDynamicsCnStateUpdate3PrecisionInputs


class EcosystemDynamicsThroughCnStateUpdate3PrecisionDiagnostics(NamedTuple):
    through_fire_flux: EcosystemDynamicsThroughFireFluxDiagnostics
    state_update3_precision: EcosystemDynamicsCnStateUpdate3PrecisionDiagnostics


class EcosystemDynamicsThroughCnStateUpdate3PrecisionResult(NamedTuple):
    state: EcosystemDynamicsCnStateUpdate3PrecisionState
    diagnostics: EcosystemDynamicsThroughCnStateUpdate3PrecisionDiagnostics


def ecosystem_dynamics_through_cn_state_update3_precision(
    inputs: EcosystemDynamicsThroughCnStateUpdate3PrecisionInputs,
    state: EcosystemDynamicsCnStateUpdate3PrecisionState,
) -> EcosystemDynamicsThroughCnStateUpdate3PrecisionResult:

    fire = ecosystem_dynamics_through_fire_flux(
        inputs.through_fire_flux,
        state.fire_flux,
    )
    update = ecosystem_dynamics_cn_state_update3_precision(
        inputs.state_update3_precision,
        state._replace(fire_flux=fire.state),
    )
    return EcosystemDynamicsThroughCnStateUpdate3PrecisionResult(
        state=update.state,
        diagnostics=EcosystemDynamicsThroughCnStateUpdate3PrecisionDiagnostics(
            through_fire_flux=fire.diagnostics,
            state_update3_precision=update.diagnostics,
        ),
    )


class EcosystemDynamicsAnnualUpdateInputs(NamedTuple):
    annual_update: CnAnnualInputs
    parameters: CnAnnualParameters = DEFAULT_CN_ANNUAL_PARAMETERS


class EcosystemDynamicsAnnualUpdateState(NamedTuple):
    state_update3_precision: EcosystemDynamicsCnStateUpdate3PrecisionState
    annual: CnAnnualState


class EcosystemDynamicsAnnualUpdateDiagnostics(NamedTuple):
    annual: CnAnnualDiagnostics


class EcosystemDynamicsAnnualUpdateResult(NamedTuple):
    state: EcosystemDynamicsAnnualUpdateState
    diagnostics: EcosystemDynamicsAnnualUpdateDiagnostics


def ecosystem_dynamics_annual_update(
    inputs: EcosystemDynamicsAnnualUpdateInputs,
    state: EcosystemDynamicsAnnualUpdateState,
) -> EcosystemDynamicsAnnualUpdateResult:

    update3_state = state.state_update3_precision
    fire_flux_state = update3_state.fire_flux
    fire_area_state = fire_flux_state.fire_area
    wood_state = fire_area_state.wood_products
    update2h_state = wood_state.state_update2h_precision
    update2_state = update2h_state.state_update2
    gap_state = update2_state.gap_mortality
    transport = gap_state.litter_vertical_transport
    soil_n = transport.soil_n_state_update
    second_guardrail = soil_n.second_guardrail
    guarded = second_guardrail.first_guardrail
    ecosystem = guarded.ecosystem
    pre_decomposition = ecosystem.pre_decomposition
    prefix = pre_decomposition.prefix
    nutrient_state = prefix.plant_nutrient_demand
    exact_state = state.annual._replace(
        temporary_sum_potential_gpp_patch=(
            nutrient_state.temporary_sum_potential_gpp_patch
        ),
        temporary_maximum_retranslocated_nitrogen_patch=(
            nutrient_state.temporary_maximum_retranslocated_nitrogen_patch
        ),
        temporary_average_air_temperature_patch=(
            prefix.phenology_climate.temperature_average_2m_patch
        ),
    )
    annual = cn_annual_update(
        inputs.annual_update,
        exact_state,
        inputs.parameters,
    )
    synchronized_nutrient_state = nutrient_state._replace(
        temporary_sum_potential_gpp_patch=(
            annual.state.temporary_sum_potential_gpp_patch
        ),
        temporary_maximum_retranslocated_nitrogen_patch=(
            annual.state.temporary_maximum_retranslocated_nitrogen_patch
        ),
    )
    synchronized_transport = transport._replace(
        soil_n_state_update=soil_n._replace(
            second_guardrail=second_guardrail._replace(
                first_guardrail=guarded._replace(
                    ecosystem=ecosystem._replace(
                        pre_decomposition=pre_decomposition._replace(
                            prefix=prefix._replace(
                                plant_nutrient_demand=synchronized_nutrient_state,
                                phenology_climate=prefix.phenology_climate._replace(
                                    temperature_average_2m_patch=(
                                        annual.state.temporary_average_air_temperature_patch
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    synchronized_update2 = update2_state._replace(
        gap_mortality=gap_state._replace(
            litter_vertical_transport=synchronized_transport,
        ),
    )
    synchronized_update3 = update3_state._replace(
        fire_flux=fire_flux_state._replace(
            fire_area=fire_area_state._replace(
                wood_products=wood_state._replace(
                    state_update2h_precision=update2h_state._replace(
                        state_update2=synchronized_update2,
                    ),
                ),
            ),
        ),
    )
    return EcosystemDynamicsAnnualUpdateResult(
        state=EcosystemDynamicsAnnualUpdateState(
            state_update3_precision=synchronized_update3,
            annual=annual.state,
        ),
        diagnostics=EcosystemDynamicsAnnualUpdateDiagnostics(
            annual=annual.diagnostics,
        ),
    )


class EcosystemDynamicsCompleteInputs(NamedTuple):
    through_state_update3_precision: (
        EcosystemDynamicsThroughCnStateUpdate3PrecisionInputs
    )
    annual_update: EcosystemDynamicsAnnualUpdateInputs


class EcosystemDynamicsCompleteDiagnostics(NamedTuple):
    through_state_update3_precision: (
        EcosystemDynamicsThroughCnStateUpdate3PrecisionDiagnostics
    )
    annual_update: EcosystemDynamicsAnnualUpdateDiagnostics


class EcosystemDynamicsCompleteResult(NamedTuple):
    state: EcosystemDynamicsAnnualUpdateState
    diagnostics: EcosystemDynamicsCompleteDiagnostics


def ecosystem_dynamics_complete(
    inputs: EcosystemDynamicsCompleteInputs,
    state: EcosystemDynamicsAnnualUpdateState,
) -> EcosystemDynamicsCompleteResult:

    update3 = ecosystem_dynamics_through_cn_state_update3_precision(
        inputs.through_state_update3_precision,
        state.state_update3_precision,
    )
    annual = ecosystem_dynamics_annual_update(
        inputs.annual_update,
        state._replace(state_update3_precision=update3.state),
    )
    return EcosystemDynamicsCompleteResult(
        state=annual.state,
        diagnostics=EcosystemDynamicsCompleteDiagnostics(
            through_state_update3_precision=update3.diagnostics,
            annual_update=annual.diagnostics,
        ),
    )
