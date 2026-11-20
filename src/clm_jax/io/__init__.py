
from clm_jax.io.active_layer import (
    ActiveLayerNativeData,
    assemble_active_layer_inputs,
    load_active_layer_native_data,
)
from clm_jax.io.background_litterfall import (
    BackgroundLitterfallCall,
    BackgroundLitterfallNativeData,
    assemble_background_litterfall_call,
    load_background_litterfall_native_data,
)
from clm_jax.io.bare_ground_driver import (
    BareGroundDriverNativeData,
    load_bare_ground_driver_native_data,
)
from clm_jax.io.bgc_external_forcing import (
    BgcExternalForcingNativeData,
    StreamBracket,
    load_bgc_external_forcing_native_data,
)
from clm_jax.io.bgc_timestep_initialization import (
    BgcTimestepInitializationNativeData,
    assemble_bgc_timestep_initialization_state,
    load_bgc_timestep_initialization_native_data,
)
from clm_jax.io.c_state_update0 import (
    CStateUpdate0Call,
    CStateUpdate0NativeData,
    assemble_c_state_update0_call,
    load_c_state_update0_native_data,
)
from clm_jax.io.canopy_flux_driver import (
    CanopyFluxDriverNativeData,
    load_canopy_flux_driver_native_data,
)
from clm_jax.io.canopy_hydraulic_inputs import (
    CanopyBiochemicalPftParameters,
    CanopyHydraulicNativeData,
    CanopyHydraulicNativeInputs,
    assemble_canopy_hydraulic_coupling,
    bedrock_level_count_column,
    load_canopy_hydraulic_native_data,
)
from clm_jax.io.canopy_temperature_driver import (
    CanopyTemperatureDriverNativeData,
    load_canopy_temperature_driver_native_data,
)
from clm_jax.io.canopy_temperature_restart import (
    CanopyTemperatureRestartData,
    ColumnSoilHydraulicProperties,
    column_soil_hydraulic_properties_20sl_8p5m,
    load_canopy_temperature_restart,
    synchronize_canopy_temperature_water_state,
)
from clm_jax.io.carbon_nitrogen_balance_initialization import (
    CarbonNitrogenBalanceInitializationNativeData,
    load_carbon_nitrogen_balance_initialization_native_data,
)
from clm_jax.io.cime_case import CimeCase, InputFileEntry
from clm_jax.io.cn_annual_update import (
    CnAnnualCall,
    CnAnnualNativeData,
    assemble_cn_annual_call,
    initial_cn_annual_call,
    load_cn_annual_native_data,
)
from clm_jax.io.cn_driver_summarize_fluxes import (
    CnDriverSummarizeFluxesCall,
    CnDriverSummarizeFluxesNativeData,
    CnDriverSummarizeFluxesPersistentState,
    assemble_cn_driver_summarize_fluxes_call,
    load_cn_driver_summarize_fluxes_native_data,
)
from clm_jax.io.cn_driver_summarize_states import (
    CnDriverSummarizeStatesCall,
    CnDriverSummarizeStatesNativeData,
    assemble_cn_driver_summarize_states_call,
    load_cn_driver_summarize_states_native_data,
)
from clm_jax.io.cn_fire_area import (
    CnFireAreaCall,
    CnFireAreaNativeData,
    CnFireAreaPersistentInputs,
    CnFireAreaSameStepForcing,
    CnFireAreaStepClock,
    assemble_cn_fire_area_call,
    initial_cn_fire_area_clock,
    load_cn_fire_area_native_data,
)
from clm_jax.io.cn_fire_emissions import (
    CnFireEmissionsConfiguration,
    load_cn_fire_emissions_configuration,
    require_disabled_cn_fire_emissions,
)
from clm_jax.io.cn_fire_fluxes import (
    CnFireFluxCall,
    CnFireFluxNativeData,
    CnFireFluxPersistentInputs,
    assemble_cn_fire_flux_call,
    load_cn_fire_flux_native_data,
)
from clm_jax.io.cn_fun_allocation import (
    FunAllocationCall,
    FunAllocationNativeData,
    FunAllocationPersistentInputs,
    assemble_fun_allocation_call,
    assemble_fun_competition_coupling_inputs,
    load_fun_allocation_native_data,
)
from clm_jax.io.cn_gap_mortality import (
    CnGapMortalityCall,
    CnGapMortalityNativeData,
    CnGapMortalityPersistentInputs,
    assemble_cn_gap_mortality_call,
    load_cn_gap_mortality_native_data,
)
from clm_jax.io.cn_n_state_update3 import (
    CnNStateUpdate3Call,
    CnNStateUpdate3NativeData,
    assemble_cn_n_state_update3_call,
    load_cn_n_state_update3_native_data,
)
from clm_jax.io.cn_precision_control import (
    CnPrecisionControlCall,
    CnPrecisionControlNativeData,
    assemble_cn_precision_control_call,
    load_cn_precision_control_native_data,
    require_valid_cn_precision_control_result,
)
from clm_jax.io.cn_state_update1 import (
    CnStateUpdate1Call,
    CnStateUpdate1NativeData,
    assemble_cn_state_update1_call,
    load_cn_state_update1_native_data,
)
from clm_jax.io.cn_state_update2 import (
    CnStateUpdate2Call,
    CnStateUpdate2hWithPrecisionCall,
    CnStateUpdate2NativeData,
    assemble_cn_state_update2_call,
    assemble_cn_state_update2h_with_precision_call,
    load_cn_state_update2_native_data,
)
from clm_jax.io.cn_state_update3 import (
    CnStateUpdate3Call,
    CnStateUpdate3NativeData,
    CnStateUpdate3WithPrecisionCall,
    assemble_cn_state_update3_call,
    assemble_cn_state_update3_with_precision_call,
    load_cn_state_update3_native_data,
)
from clm_jax.io.cn_veg_struct_update import (
    CnVegStructUpdateCall,
    CnVegStructUpdateNativeData,
    assemble_cn_veg_struct_update_call,
    load_cn_veg_struct_update_native_data,
)
from clm_jax.io.cn_wood_products import (
    CnWoodProductsCall,
    CnWoodProductsNativeData,
    assemble_cn_wood_products_call,
    load_cn_wood_products_native_data,
)
from clm_jax.io.coupler import (
    GlacierCouplerState,
    load_atmospheric_aerosol_coupler_forcing,
    load_atmospheric_coupler_forcing,
    load_glacier_coupler_state,
    load_river_volume_coupler_state,
)
from clm_jax.io.crop import (
    FIRST_PROGNOSTIC_CROP_PFT,
    CropYearRestartData,
    load_crop_year_restart,
)
from clm_jax.io.crop_phenology import (
    CropPhenologyNativeData,
    load_crop_phenology_native_data,
)
from clm_jax.io.datm_stream import DatmStream, VariableMapping
from clm_jax.io.decomposition_rates import (
    DecompositionRateNativeData,
    DecompositionRatePersistentState,
    assemble_decomposition_rate_inputs,
    load_decomposition_rate_native_data,
    load_decomposition_rate_parameters,
)
from clm_jax.io.diagnostics_and_accumulators import (
    DiagnosticsAndAccumulatorsRestartData,
    load_diagnostics_and_accumulators_restart,
)
from clm_jax.io.driver import (
    DriverInitializationNativeData,
    load_driver_initialization_native_data,
)
from clm_jax.io.dry_deposition import (
    DryDepositionConfiguration,
    load_dry_deposition_configuration,
)
from clm_jax.io.ecosystem_dynamics_post_drainage import (
    EcosystemDynamicsPostDrainageCall,
    EcosystemDynamicsPostDrainageNativeData,
    assemble_ecosystem_dynamics_post_drainage_call,
    assemble_ecosystem_dynamics_post_drainage_call_from_carry,
    load_ecosystem_dynamics_post_drainage_native_data,
)
from clm_jax.io.end_of_step_vegetation import (
    EndOfStepVegetationConfiguration,
    load_end_of_step_vegetation_configuration,
)
from clm_jax.io.evergreen_phenology import (
    EvergreenPhenologyNativeData,
    load_evergreen_phenology_native_data,
)
from clm_jax.io.fun_initialization import (
    FunInitializationNativeData,
    load_fun_initialization_native_data,
)
from clm_jax.io.glacier_fraction_initialization import (
    GlacierFractionInitializationNativeData,
    load_glacier_fraction_initialization_native_data,
)
from clm_jax.io.grain_to_product_pools import (
    GrainToProductPoolsCall,
    GrainToProductPoolsNativeData,
    assemble_grain_to_product_pools_call,
    load_grain_to_product_pools_native_data,
)
from clm_jax.io.growth_respiration import (
    GrowthRespirationCall,
    GrowthRespirationNativeData,
    assemble_growth_respiration_call,
    load_growth_respiration_native_data,
)
from clm_jax.io.hydrology_drainage_driver import (
    HydrologyDrainageDriverNativeData,
    load_hydrology_drainage_driver_native_data,
)
from clm_jax.io.hydrology_driver import (
    HydrologyDriverNativeData,
    load_hydrology_driver_native_data,
)
from clm_jax.io.lake_hydrology_driver import (
    LakeHydrologyDriverNativeData,
    load_lake_hydrology_driver_native_data,
)
from clm_jax.io.litter_to_column import (
    LitterToColumnCall,
    LitterToColumnNativeData,
    LitterToColumnPersistentInputs,
    assemble_litter_to_column_call,
    load_litter_to_column_native_data,
)
from clm_jax.io.livewood_turnover import (
    LivewoodTurnoverCall,
    LivewoodTurnoverNativeData,
    assemble_livewood_turnover_call,
    load_livewood_turnover_native_data,
)
from clm_jax.io.lnd2glc import (
    LandToGlacierNativeData,
    load_land_to_glacier_native_data,
)
from clm_jax.io.luna import (
    LunaCapacityRestartInputs,
    LunaRestartData,
    load_luna_capacity_parameters,
    load_luna_restart,
)
from clm_jax.io.maintenance_respiration import (
    MaintenanceRespirationNativeData,
    MaintenanceRespirationNitrogenState,
    MaintenanceRespirationTemperatureState,
    assemble_maintenance_respiration_inputs,
    load_maintenance_respiration_native_data,
    load_maintenance_respiration_parameters,
)
from clm_jax.io.methane_balance_initialization import (
    MethaneBalanceInitializationNativeData,
    load_methane_balance_initialization_native_data,
)
from clm_jax.io.methane_driver import (
    MethaneDriverNativeData,
    MethaneDriverNativeInputs,
    load_methane_driver_native_data,
)
from clm_jax.io.namelist import read_case_namelist_group, read_namelist_group
from clm_jax.io.nitrogen import (
    FreeLivingFixationRestartData,
    NitrogenFertilizerRestartData,
    load_free_living_fixation_restart,
    load_nitrogen_fertilizer_restart,
)
from clm_jax.io.offset_litterfall import (
    OffsetLitterfallCall,
    OffsetLitterfallNativeData,
    assemble_offset_litterfall_call,
    load_offset_litterfall_native_data,
)
from clm_jax.io.onset_growth import (
    OnsetGrowthNativeData,
    load_onset_growth_native_data,
)
from clm_jax.io.phenology_climate import (
    PhenologyClimateNativeData,
    load_phenology_climate_native_data,
)
from clm_jax.io.plant_hydraulics import (
    PlantHydraulicRestartData,
    load_plant_hydraulic_parameters,
    load_plant_hydraulic_restart,
)
from clm_jax.io.plant_nutrient_competition import (
    PlantNutrientCompetitionCall,
    PlantNutrientCompetitionNativeData,
    PlantNutrientCompetitionPersistentInputs,
    assemble_plant_nutrient_competition_call,
    load_plant_nutrient_competition_native_data,
)
from clm_jax.io.plant_nutrient_demand import (
    PlantNutrientDemandCall,
    PlantNutrientDemandNativeData,
    PlantNutrientDemandPersistentInputs,
    assemble_plant_nutrient_demand_call,
    load_plant_nutrient_demand_native_data,
)
from clm_jax.io.post_drainage_precision_control import (
    PostDrainagePrecisionControlCall,
    PostDrainagePrecisionControlNativeData,
    assemble_post_drainage_precision_control_call,
    load_post_drainage_precision_control_native_data,
)
from clm_jax.io.post_hydrology_no_drainage_driver import (
    PostHydrologyNoDrainageNativeData,
    load_post_hydrology_no_drainage_native_data,
)
from clm_jax.io.restart import (
    ColumnTopology,
    GridTopology,
    LandunitTopology,
    PatchTopology,
    RestartClock,
    SubgridTopology,
    advance_restart_clock,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
    model_step_from_restart_clock,
)
from clm_jax.io.seasonal_deciduous_phenology import (
    SeasonalDeciduousPhenologyNativeData,
    load_seasonal_deciduous_phenology_native_data,
)
from clm_jax.io.snow_aging import (
    configured_snow_aging_path,
    load_snow_aging_lookup_table,
)
from clm_jax.io.snow_optics import (
    configured_snow_optics_path,
    load_snow_optical_tables,
)
from clm_jax.io.soil_biogeochem_competition import (
    SoilBiogeochemCompetitionCall,
    SoilBiogeochemCompetitionNativeData,
    SoilBiogeochemCompetitionPersistentState,
    assemble_soil_biogeochem_competition_call,
    load_soil_biogeochem_competition_native_data,
)
from clm_jax.io.soil_biogeochem_decomp import (
    SoilBiogeochemDecompCall,
    SoilBiogeochemDecompNativeData,
    assemble_soil_biogeochem_decomp_call,
    load_soil_biogeochem_decomp_native_data,
)
from clm_jax.io.soil_biogeochem_litter_vertical_transport import (
    SoilBiogeochemLitterVerticalTransportCall,
    SoilBiogeochemLitterVerticalTransportNativeData,
    assemble_soil_biogeochem_litter_vertical_transport_call,
    load_soil_biogeochem_litter_vertical_transport_native_data,
)
from clm_jax.io.soil_biogeochem_n_leaching import (
    SoilBiogeochemNLeachingCall,
    SoilBiogeochemNLeachingNativeData,
    SoilBiogeochemNLeachingPersistentState,
    assemble_soil_biogeochem_n_leaching_call,
    load_soil_biogeochem_n_leaching_native_data,
)
from clm_jax.io.soil_biogeochem_n_state_update1 import (
    SoilBiogeochemNStateUpdate1Call,
    SoilBiogeochemNStateUpdate1NativeData,
    assemble_soil_biogeochem_n_state_update1_call,
    load_soil_biogeochem_n_state_update1_native_data,
)
from clm_jax.io.soil_biogeochem_nitrif_denitrif import (
    SoilBiogeochemNitrifDenitrifNativeData,
    SoilBiogeochemNitrifDenitrifPersistentState,
    assemble_soil_biogeochem_nitrif_denitrif_inputs,
    load_soil_biogeochem_nitrif_denitrif_native_data,
    load_soil_biogeochem_nitrif_denitrif_parameters,
)
from clm_jax.io.soil_biogeochem_potential import (
    SoilBiogeochemPotentialNativeData,
    SoilBiogeochemPotentialPoolState,
    assemble_soil_biogeochem_potential_inputs,
    load_soil_biogeochem_potential_native_data,
)
from clm_jax.io.soil_biogeochem_precision_control import (
    SoilBiogeochemPrecisionControlCall,
    SoilBiogeochemPrecisionControlNativeData,
    SoilBiogeochemPrecisionControlPersistentState,
    assemble_soil_biogeochem_precision_control_call,
    load_soil_biogeochem_precision_control_native_data,
)
from clm_jax.io.soil_biogeochem_vertical_profile import (
    SoilBiogeochemVerticalProfileNativeData,
    assemble_soil_biogeochem_vertical_profile_inputs,
    load_soil_biogeochem_vertical_profile_native_data,
)
from clm_jax.io.stress_deciduous_phenology import (
    StressDeciduousPhenologyNativeData,
    StressDeciduousPhenologyPersistentInputs,
    assemble_stress_deciduous_phenology_inputs,
    load_stress_deciduous_phenology_native_data,
)
from clm_jax.io.subgrid_filter import (
    ExposedVegetationFilterNativeData,
    load_exposed_vegetation_filter_native_data,
)
from clm_jax.io.surface_radiation_restart import (
    SurfaceRadiationRestartData,
    load_surface_radiation_restart,
)
from clm_jax.io.urban_albedo_restart import (
    UrbanAlbedoRestartData,
    load_urban_albedo_restart,
)
from clm_jax.io.urban_fluxes_restart import (
    UrbanFluxesNativeMap,
    UrbanFluxesRestartData,
    UrbanFluxesRestartState,
    UrbanFluxesStaticParameters,
    UrbanFluxesStepData,
    assemble_urban_fluxes_step_data,
    load_urban_fluxes_restart,
)
from clm_jax.io.urban_radiation_restart import (
    UrbanRadiationRestartData,
    load_urban_radiation_restart,
)
from clm_jax.io.urban_time_varying import (
    UrbanTimeVaryingNativeData,
    load_urban_time_varying_native_data,
)
from clm_jax.io.water_balance_initialization import (
    WaterBalanceInitializationNativeData,
    load_water_balance_initialization_native_data,
)

__all__ = [
    "DryDepositionConfiguration",
    "EndOfStepVegetationConfiguration",
    "ActiveLayerNativeData",
    "BackgroundLitterfallCall",
    "BackgroundLitterfallNativeData",
    "BgcExternalForcingNativeData",
    "BgcTimestepInitializationNativeData",
    "BareGroundDriverNativeData",
    "CarbonNitrogenBalanceInitializationNativeData",
    "CStateUpdate0Call",
    "CStateUpdate0NativeData",
    "CimeCase",
    "CnAnnualCall",
    "CnAnnualNativeData",
    "CnFireAreaCall",
    "CnFireAreaNativeData",
    "CnFireAreaPersistentInputs",
    "CnFireAreaSameStepForcing",
    "CnFireAreaStepClock",
    "CnFireFluxCall",
    "CnFireFluxNativeData",
    "CnFireFluxPersistentInputs",
    "CnFireEmissionsConfiguration",
    "CnGapMortalityCall",
    "CnGapMortalityNativeData",
    "CnGapMortalityPersistentInputs",
    "CnNStateUpdate3Call",
    "CnNStateUpdate3NativeData",
    "CnPrecisionControlCall",
    "CnPrecisionControlNativeData",
    "CnStateUpdate1Call",
    "CnStateUpdate1NativeData",
    "CnStateUpdate2Call",
    "CnStateUpdate2NativeData",
    "CnStateUpdate2hWithPrecisionCall",
    "CnStateUpdate3Call",
    "CnStateUpdate3NativeData",
    "CnStateUpdate3WithPrecisionCall",
    "CnWoodProductsCall",
    "CnWoodProductsNativeData",
    "CanopyBiochemicalPftParameters",
    "CanopyFluxDriverNativeData",
    "CanopyHydraulicNativeData",
    "CanopyHydraulicNativeInputs",
    "CanopyTemperatureRestartData",
    "CanopyTemperatureDriverNativeData",
    "ColumnTopology",
    "ColumnSoilHydraulicProperties",
    "CropYearRestartData",
    "CropPhenologyNativeData",
    "DatmStream",
    "DecompositionRateNativeData",
    "DecompositionRatePersistentState",
    "DriverInitializationNativeData",
    "DiagnosticsAndAccumulatorsRestartData",
    "ExposedVegetationFilterNativeData",
    "EvergreenPhenologyNativeData",
    "FIRST_PROGNOSTIC_CROP_PFT",
    "FreeLivingFixationRestartData",
    "FunAllocationCall",
    "GlacierFractionInitializationNativeData",
    "FunAllocationNativeData",
    "FunAllocationPersistentInputs",
    "FunInitializationNativeData",
    "GridTopology",
    "GrainToProductPoolsCall",
    "GrainToProductPoolsNativeData",
    "GrowthRespirationCall",
    "GrowthRespirationNativeData",
    "GlacierCouplerState",
    "InputFileEntry",
    "LandToGlacierNativeData",
    "LandunitTopology",
    "LunaCapacityRestartInputs",
    "LunaRestartData",
    "MaintenanceRespirationNativeData",
    "MaintenanceRespirationNitrogenState",
    "MaintenanceRespirationTemperatureState",
    "MethaneBalanceInitializationNativeData",
    "MethaneDriverNativeData",
    "MethaneDriverNativeInputs",
    "LivewoodTurnoverCall",
    "LivewoodTurnoverNativeData",
    "LitterToColumnCall",
    "LitterToColumnNativeData",
    "LitterToColumnPersistentInputs",
    "NitrogenFertilizerRestartData",
    "OnsetGrowthNativeData",
    "OffsetLitterfallCall",
    "OffsetLitterfallNativeData",
    "PatchTopology",
    "PhenologyClimateNativeData",
    "SeasonalDeciduousPhenologyNativeData",
    "PlantHydraulicRestartData",
    "PlantNutrientCompetitionCall",
    "PlantNutrientCompetitionNativeData",
    "PlantNutrientCompetitionPersistentInputs",
    "PlantNutrientDemandCall",
    "PlantNutrientDemandNativeData",
    "PlantNutrientDemandPersistentInputs",
    "read_case_namelist_group",
    "read_namelist_group",
    "RestartClock",
    "SoilBiogeochemPotentialNativeData",
    "SoilBiogeochemPotentialPoolState",
    "SoilBiogeochemCompetitionCall",
    "SoilBiogeochemCompetitionNativeData",
    "SoilBiogeochemCompetitionPersistentState",
    "SoilBiogeochemDecompCall",
    "SoilBiogeochemDecompNativeData",
    "SoilBiogeochemLitterVerticalTransportCall",
    "SoilBiogeochemLitterVerticalTransportNativeData",
    "SoilBiogeochemNitrifDenitrifNativeData",
    "SoilBiogeochemNitrifDenitrifPersistentState",
    "SoilBiogeochemNStateUpdate1Call",
    "SoilBiogeochemNStateUpdate1NativeData",
    "SoilBiogeochemVerticalProfileNativeData",
    "StressDeciduousPhenologyNativeData",
    "StressDeciduousPhenologyPersistentInputs",
    "StreamBracket",
    "SubgridTopology",
    "SurfaceRadiationRestartData",
    "UrbanAlbedoRestartData",
    "UrbanRadiationRestartData",
    "UrbanTimeVaryingNativeData",
    "UrbanFluxesNativeMap",
    "UrbanFluxesRestartData",
    "UrbanFluxesRestartState",
    "UrbanFluxesStaticParameters",
    "UrbanFluxesStepData",
    "VariableMapping",
    "WaterBalanceInitializationNativeData",
    "assemble_background_litterfall_call",
    "assemble_bgc_timestep_initialization_state",
    "assemble_active_layer_inputs",
    "assemble_c_state_update0_call",
    "assemble_cn_fire_area_call",
    "assemble_cn_annual_call",
    "assemble_cn_fire_flux_call",
    "assemble_cn_state_update3_call",
    "assemble_cn_state_update3_with_precision_call",
    "assemble_cn_gap_mortality_call",
    "assemble_cn_n_state_update3_call",
    "assemble_cn_driver_summarize_fluxes_call",
    "assemble_cn_driver_summarize_states_call",
    "assemble_cn_precision_control_call",
    "assemble_cn_state_update1_call",
    "assemble_cn_state_update2_call",
    "assemble_cn_state_update2h_with_precision_call",
    "assemble_cn_wood_products_call",
    "assemble_canopy_hydraulic_coupling",
    "assemble_decomposition_rate_inputs",
    "assemble_fun_allocation_call",
    "assemble_fun_competition_coupling_inputs",
    "assemble_grain_to_product_pools_call",
    "assemble_growth_respiration_call",
    "assemble_maintenance_respiration_inputs",
    "assemble_livewood_turnover_call",
    "assemble_litter_to_column_call",
    "assemble_offset_litterfall_call",
    "assemble_plant_nutrient_competition_call",
    "assemble_plant_nutrient_demand_call",
    "assemble_soil_biogeochem_potential_inputs",
    "assemble_soil_biogeochem_competition_call",
    "assemble_soil_biogeochem_decomp_call",
    "assemble_soil_biogeochem_litter_vertical_transport_call",
    "assemble_soil_biogeochem_nitrif_denitrif_inputs",
    "assemble_soil_biogeochem_n_state_update1_call",
    "assemble_soil_biogeochem_vertical_profile_inputs",
    "assemble_stress_deciduous_phenology_inputs",
    "assemble_urban_fluxes_step_data",
    "advance_restart_clock",
    "bedrock_level_count_column",
    "column_soil_hydraulic_properties_20sl_8p5m",
    "configured_land_restart",
    "configured_snow_aging_path",
    "load_restart_clock",
    "model_step_from_restart_clock",
    "load_background_litterfall_native_data",
    "load_bare_ground_driver_native_data",
    "load_bgc_external_forcing_native_data",
    "load_bgc_timestep_initialization_native_data",
    "load_carbon_nitrogen_balance_initialization_native_data",
    "load_c_state_update0_native_data",
    "initial_cn_fire_area_clock",
    "initial_cn_annual_call",
    "load_cn_annual_native_data",
    "load_cn_fire_area_native_data",
    "load_cn_fire_flux_native_data",
    "load_cn_fire_emissions_configuration",
    "load_cn_state_update3_native_data",
    "load_cn_gap_mortality_native_data",
    "load_cn_n_state_update3_native_data",
    "load_cn_driver_summarize_fluxes_native_data",
    "load_cn_driver_summarize_states_native_data",
    "load_cn_precision_control_native_data",
    "load_cn_state_update1_native_data",
    "load_cn_state_update2_native_data",
    "load_cn_wood_products_native_data",
    "load_glacier_coupler_state",
    "load_river_volume_coupler_state",
    "load_atmospheric_aerosol_coupler_forcing",
    "load_atmospheric_coupler_forcing",
    "HydrologyDriverNativeData",
    "load_hydrology_driver_native_data",
    "HydrologyDrainageDriverNativeData",
    "load_hydrology_drainage_driver_native_data",
    "LakeHydrologyDriverNativeData",
    "load_lake_hydrology_driver_native_data",
    "PostHydrologyNoDrainageNativeData",
    "load_post_hydrology_no_drainage_native_data",
    "load_glacier_fraction_initialization_native_data",
    "load_grain_to_product_pools_native_data",
    "load_growth_respiration_native_data",
    "load_canopy_flux_driver_native_data",
    "load_canopy_hydraulic_native_data",
    "load_luna_capacity_parameters",
    "load_luna_restart",
    "load_livewood_turnover_native_data",
    "load_active_layer_native_data",
    "load_land_to_glacier_native_data",
    "load_end_of_step_vegetation_configuration",
    "load_litter_to_column_native_data",
    "load_maintenance_respiration_native_data",
    "load_maintenance_respiration_parameters",
    "load_methane_balance_initialization_native_data",
    "load_methane_driver_native_data",
    "load_phenology_climate_native_data",
    "load_seasonal_deciduous_phenology_native_data",
    "load_canopy_temperature_restart",
    "load_canopy_temperature_driver_native_data",
    "load_crop_year_restart",
    "load_crop_phenology_native_data",
    "load_decomposition_rate_native_data",
    "load_decomposition_rate_parameters",
    "load_driver_initialization_native_data",
    "load_diagnostics_and_accumulators_restart",
    "load_dry_deposition_configuration",
    "load_exposed_vegetation_filter_native_data",
    "load_evergreen_phenology_native_data",
    "load_free_living_fixation_restart",
    "load_fun_initialization_native_data",
    "load_fun_allocation_native_data",
    "load_nitrogen_fertilizer_restart",
    "load_onset_growth_native_data",
    "load_offset_litterfall_native_data",
    "load_urban_fluxes_restart",
    "load_plant_hydraulic_parameters",
    "load_plant_hydraulic_restart",
    "load_plant_nutrient_competition_native_data",
    "load_plant_nutrient_demand_native_data",
    "load_subgrid_topology",
    "load_surface_radiation_restart",
    "load_snow_aging_lookup_table",
    "load_soil_biogeochem_potential_native_data",
    "load_soil_biogeochem_competition_native_data",
    "load_soil_biogeochem_decomp_native_data",
    "load_soil_biogeochem_litter_vertical_transport_native_data",
    "load_soil_biogeochem_nitrif_denitrif_native_data",
    "load_soil_biogeochem_nitrif_denitrif_parameters",
    "load_soil_biogeochem_n_state_update1_native_data",
    "load_soil_biogeochem_vertical_profile_native_data",
    "load_stress_deciduous_phenology_native_data",
    "load_urban_albedo_restart",
    "load_urban_radiation_restart",
    "load_urban_time_varying_native_data",
    "load_water_balance_initialization_native_data",
    "require_valid_cn_precision_control_result",
    "require_disabled_cn_fire_emissions",
    "synchronize_canopy_temperature_water_state",
    "SoilBiogeochemNLeachingCall",
    "SoilBiogeochemNLeachingNativeData",
    "SoilBiogeochemNLeachingPersistentState",
    "assemble_soil_biogeochem_n_leaching_call",
    "load_soil_biogeochem_n_leaching_native_data",
    "PostDrainagePrecisionControlCall",
    "PostDrainagePrecisionControlNativeData",
    "assemble_post_drainage_precision_control_call",
    "load_post_drainage_precision_control_native_data",
    "EcosystemDynamicsPostDrainageCall",
    "EcosystemDynamicsPostDrainageNativeData",
    "assemble_ecosystem_dynamics_post_drainage_call",
    "assemble_ecosystem_dynamics_post_drainage_call_from_carry",
    "load_ecosystem_dynamics_post_drainage_native_data",
    "CnDriverSummarizeStatesCall",
    "CnDriverSummarizeStatesNativeData",
    "CnDriverSummarizeFluxesCall",
    "CnDriverSummarizeFluxesNativeData",
    "CnDriverSummarizeFluxesPersistentState",
    "CnVegStructUpdateCall",
    "CnVegStructUpdateNativeData",
    "assemble_cn_veg_struct_update_call",
    "load_cn_veg_struct_update_native_data",
    "SoilBiogeochemPrecisionControlCall",
    "SoilBiogeochemPrecisionControlNativeData",
    "SoilBiogeochemPrecisionControlPersistentState",
    "assemble_soil_biogeochem_precision_control_call",
    "load_soil_biogeochem_precision_control_native_data",
    "configured_snow_optics_path",
    "load_snow_optical_tables",
]
