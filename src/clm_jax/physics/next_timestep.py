
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from clm_jax.physics.atmospheric_driver import AtmosphericDriverInputs
from clm_jax.physics.canopy_driver import CanopyDriverInputs, CanopyDriverState
from clm_jax.physics.canopy_flux_driver import (
    CanopyFluxDriverInputs,
    CanopyFluxDriverState,
)
from clm_jax.physics.canopy_hydrology import CanopyWaterState
from clm_jax.physics.canopy_radiation import CanopySunShadeInputs
from clm_jax.physics.canopy_step import CanopyHydrologyState, CanopyStructure
from clm_jax.physics.canopy_temperature import (
    NUMBER_OF_SOIL_LEVELS,
    CanopyTemperatureInputs,
    CanopyTemperatureState,
)
from clm_jax.physics.cn_precision_control import (
    get_cn_precision_control_state_field,
)
from clm_jax.physics.dust_driver import DustDriverInputs
from clm_jax.physics.ecosystem_dynamics import EcosystemDynamicsAnnualUpdateState
from clm_jax.physics.evergreen_phenology import (
    CARBON_TRANSFER_POOL_NAMES,
    NITROGEN_TRANSFER_POOL_NAMES,
)
from clm_jax.physics.hydrology_no_drainage import HydrologyNoDrainageState
from clm_jax.physics.lake_flux_driver import (
    LakeFluxDriverInputs,
    LakeFluxDriverState,
)
from clm_jax.physics.lake_temperature import LakeColumnTemperatureState
from clm_jax.physics.lake_temperature_driver import LakeTemperatureDriverInputs
from clm_jax.physics.numerics import fortran_divide, fortran_multiply
from clm_jax.physics.physical_driver import PhysicalDriverInputs, PhysicalDriverState
from clm_jax.physics.radiation_driver import (
    RadiationDriverInputs,
    RadiationDriverState,
)
from clm_jax.physics.radiation_sequence import (
    RadiationSequenceInputs,
    RadiationSequenceState,
)
from clm_jax.physics.snow import SnowColumnState
from clm_jax.physics.snow_water import NUMBER_OF_SNOW_LEVELS
from clm_jax.physics.soil_biogeochem_n_state_update1 import (
    SoilBiogeochemNStateUpdate1State,
)
from clm_jax.physics.soil_temperature_driver import SoilTemperatureDriverInputs
from clm_jax.physics.surface_radiation import SurfaceRadiationInputs
from clm_jax.physics.urban_flux_driver import (
    UrbanFluxDriverInputs,
    UrbanFluxDriverState,
)
from clm_jax.physics.urban_radiation import UrbanRadiationInputs, UrbanRadiationState


def _snow_levels(value: Any, *, field: str) -> jnp.ndarray:

    array = jnp.asarray(value)
    if array.ndim != 2 or array.shape[1] < NUMBER_OF_SNOW_LEVELS:
        raise ValueError(f"{field} must have shape (column, >= {NUMBER_OF_SNOW_LEVELS})")
    return array[:, :NUMBER_OF_SNOW_LEVELS]


def _native_vector_index(value: Any, *, field: str) -> jnp.ndarray:

    index = jnp.asarray(value, dtype=jnp.int32)
    if index.ndim != 1:
        raise ValueError(f"{field} must be a vector")
    return index


def next_atmospheric_inputs(
    template: AtmosphericDriverInputs,
    carry: Any,
) -> AtmosphericDriverInputs:

    final_snow = carry.lake_hydrology
    drainage = carry.drainage
    vegetation = carry.ecosystem_post_drainage.vegetation_structure
    return AtmosphericDriverInputs(
        intracellular_sunlit_co2=template.intracellular_sunlit_co2,
        intracellular_shaded_co2=template.intracellular_shaded_co2,
        snow_water_equivalent=(final_snow.water.snow_water_equivalent_column),
        snow_layer_count=final_snow.geometry.snow_layer_count_column,
        snow_layer_ice=drainage.soil_snow_ice_water_column_level,
        snow_layer_liquid=drainage.soil_snow_liquid_water_column_level,
        vegetation_exposed_for_albedo=(vegetation.fraction_vegetation_nosnow_albedo_patch),
    )


def next_canopy_inputs(
    template: CanopyDriverInputs,
    carry: Any,
) -> CanopyDriverInputs:

    vegetation = carry.ecosystem_post_drainage.vegetation_structure
    structure = template.structure
    return CanopyDriverInputs(
        structure=CanopyStructure(
            vegetation_exposed_patch=(vegetation.fraction_vegetation_nosnow_albedo_patch),
            maximum_dew_patch=structure.maximum_dew_patch,
            exposed_leaf_area_index_patch=(vegetation.exposed_leaf_area_index_patch),
            exposed_stem_area_index_patch=(vegetation.exposed_stem_area_index_patch),
            microtopography_sigma_column=(structure.microtopography_sigma_column),
            ground_temperature_column=(carry.hydrology.final_diagnostics.ground_temperature_column),
            snow_cover_shape_column=structure.snow_cover_shape_column,
        ),
        previous_snow_drainage_flux_column=(
            carry.lake_hydrology.water_flux.snow_drainage_flux_column
        ),
        flood_gridcell=template.flood_gridcell,
    )


def next_canopy_state(carry: Any) -> CanopyDriverState:

    early = carry.physical.canopy.canopy_hydrology
    canopy_flux = carry.physical.canopy_flux.active.postprocess
    lake = carry.lake_hydrology
    drainage = carry.drainage
    post_lake = carry.post_lake_hydrology

    snow = SnowColumnState(
        layer_count=lake.geometry.snow_layer_count_column,
        water_equivalent=lake.water.snow_water_equivalent_column,
        depth=lake.water.snow_depth_column,
        integrated_snow=lake.water.integrated_snow_column,
        snow_fraction=post_lake.snow_fraction_column,
        effective_snow_fraction=lake.water.effective_snow_fraction_column,
        layer_thickness=_snow_levels(
            lake.geometry.soil_snow_thickness_column_level,
            field="lake geometry thickness",
        ),
        layer_depth=_snow_levels(
            lake.geometry.soil_snow_node_depth_column_level,
            field="lake geometry node depth",
        ),
        layer_interface_depth=jnp.asarray(lake.geometry.soil_snow_interface_depth_column_interface)[
            :, : NUMBER_OF_SNOW_LEVELS + 1
        ],
        layer_temperature=_snow_levels(
            lake.temperature.soil_snow_temperature_column_level,
            field="lake temperature",
        ),
        layer_ice=_snow_levels(
            drainage.soil_snow_ice_water_column_level,
            field="post-drainage ice water",
        ),
        layer_liquid=_snow_levels(
            drainage.soil_snow_liquid_water_column_level,
            field="post-drainage liquid water",
        ),
        old_ice_fraction=_snow_levels(
            lake.water.old_ice_fraction_column_level,
            field="lake old ice fraction",
        ),
        snow_grain_radius=(post_lake.snow_age.snow_grain_radius_column_level),
    )
    if snow.layer_interface_depth.ndim != 2 or (
        snow.layer_interface_depth.shape[1] != NUMBER_OF_SNOW_LEVELS + 1
    ):
        raise ValueError(
            "lake geometry interfaces must contain all snow interfaces and the ground surface"
        )

    canopy_hydrology = CanopyHydrologyState(
        canopy_water=CanopyWaterState(
            total_water=canopy_flux.canopy_water_patch,
            snow=canopy_flux.snow_canopy_water_patch,
            liquid=canopy_flux.liquid_canopy_water_patch,
            snow_unloaded=early.canopy_water.snow_unloaded,
        ),
        canopy_wet_fractions=early.canopy_wet_fractions,
        snow=snow,
        surface_water_column=drainage.surface_water_column,
        top_soil_liquid_column=(
            drainage.soil_snow_liquid_water_column_level[:, NUMBER_OF_SNOW_LEVELS]
        ),
        surface_water_fraction_column=(early.surface_water_fraction_column),
        surface_water_fraction_without_snow_column=(
            early.surface_water_fraction_without_snow_column
        ),
        snow_aerosol=post_lake.aerosol_mass.aerosol,
    )
    return CanopyDriverState(
        irrigation=carry.physical.canopy.irrigation,
        canopy_hydrology=canopy_hydrology,
    )


def next_canopy_flux_inputs(
    template: CanopyFluxDriverInputs,
    carry: Any,
) -> CanopyFluxDriverInputs:

    post = carry.ecosystem_post_drainage
    vegetation = post.vegetation_structure
    pools = post.biogeochemistry.vegetation
    hydrology = carry.hydrology.soil_water_movement
    surface = carry.surface_albedo.canopy
    two_stream = surface.two_stream
    accumulators = carry.end_of_timestep.accumulators
    capacity = carry.physical.canopy_flux.active.luna_capacity

    fine_root_carbon = get_cn_precision_control_state_field(
        pools,
        "vegetation_carbon.frootc_patch",
    )
    leaf_nitrogen = get_cn_precision_control_state_field(
        pools,
        "vegetation_nitrogen.leafn_patch",
    )
    total_leaf_area = vegetation.total_leaf_area_index_patch
    has_leaf = jnp.asarray(total_leaf_area, dtype=jnp.float64) > jnp.float64(0.0)
    leaf_nitrogen_concentration = jnp.where(
        has_leaf,
        fortran_divide(
            leaf_nitrogen,
            jnp.where(has_leaf, total_leaf_area, jnp.float64(1.0)),
        ),
        jnp.float64(0.0),
    )
    leaf_nitrogen_concentration = jnp.minimum(
        leaf_nitrogen_concentration,
        jnp.float64(10.0),
    )
    ten_day_air_temperature = (
        accumulators.temperature.reference_temperature_10_day.value
    )

    hydraulic = template.hydraulic._replace(
        fine_root_carbon_patch=fine_root_carbon,
        leaf_nitrogen_patch=leaf_nitrogen,
        total_leaf_area_patch=total_leaf_area,
        total_stem_area_patch=vegetation.total_stem_area_index_patch,
        hydraulic_conductivity_column_level=(
            hydrology.hydraulic_conductivity_column_level[:, :NUMBER_OF_SOIL_LEVELS]
        ),
        soil_water_potential_column_level=(
            hydrology.soil_water_potential_column_level[:, :NUMBER_OF_SOIL_LEVELS]
        ),
        vcmax_scaling_sunlit_patch=two_stream.sunlit_vcmax_scale_patch,
        vcmax_scaling_shaded_patch=two_stream.shaded_vcmax_scale_patch,
        luna_vcmax25_by_layer_patch=(
            capacity.maximum_carboxylation_rate_by_layer_patch
        ),
        luna_jmax25_by_layer_patch=(
            capacity.maximum_electron_transport_rate_by_layer_patch
        ),
        total_leaf_area_by_layer_patch=surface.leaf_area_patch_layer,
        radiative_layer_count_patch=surface.radiative_layer_count_patch,
        air_temperature_ten_day_patch=ten_day_air_temperature,
    )

    atmosphere = accumulators.atmosphere
    capacity_inputs = template.luna_template.capacity_inputs._replace(
        atmospheric_pressure_ten_day_patch=(
            atmosphere.air_pressure_240_hour.value
        ),
        atmospheric_co2_ten_day_patch=(
            atmosphere.co2_partial_pressure_240_hour.value
        ),
        atmospheric_o2_ten_day_patch=(
            atmosphere.oxygen_partial_pressure_240_hour.value
        ),
        total_leaf_area_patch=total_leaf_area,
        total_leaf_area_by_layer_patch=surface.leaf_area_patch_layer,
        radiative_layer_count_patch=surface.radiative_layer_count_patch,
        leaf_nitrogen_concentration_patch=leaf_nitrogen_concentration,
        air_temperature_ten_day_patch=ten_day_air_temperature,
    )
    return template._replace(
        hydraulic=hydraulic,
        luna_template=template.luna_template._replace(
            capacity_inputs=capacity_inputs,
        ),
    )


def next_canopy_flux_state(carry: Any) -> CanopyFluxDriverState:

    preceding = carry.physical.canopy_flux
    return preceding._replace(
        plant=preceding.plant._replace(
            vegetation_transpiration=jnp.zeros_like(
                preceding.plant.vegetation_transpiration
            )
        )
    )


def next_dust_inputs(
    template: DustDriverInputs,
    carry: Any,
) -> DustDriverInputs:

    vegetation = carry.ecosystem_post_drainage.vegetation_structure
    return template._replace(
        total_leaf_area_index_patch=vegetation.total_leaf_area_index_patch,
        total_stem_area_index_patch=vegetation.total_stem_area_index_patch,
    )


def next_radiation_inputs(
    template: RadiationSequenceInputs,
    carry: Any,
) -> RadiationSequenceInputs:

    surface = carry.surface_albedo
    urban = carry.urban_albedo
    vegetation = carry.ecosystem_post_drainage.vegetation_structure
    snow = carry.lake_hydrology
    two_stream = surface.canopy.two_stream
    urban_surface = urban.surface
    urban_canopy = urban_surface.canopy

    canopy_sunshade = CanopySunShadeInputs(
        exposed_leaf_area_index_patch=(vegetation.exposed_leaf_area_index_patch),
        total_leaf_area_by_layer_patch=surface.canopy.leaf_area_patch_layer,
        sunlit_fraction_by_layer_patch=two_stream.sunlit_fraction_patch_layer,
        radiative_layer_count_patch=surface.canopy.radiative_layer_count_patch,
        direct_absorption_sunlit_by_layer_patch=(two_stream.direct_sunlit_absorption_patch_layer),
        direct_absorption_shaded_by_layer_patch=(two_stream.direct_shaded_absorption_patch_layer),
        diffuse_absorption_sunlit_by_layer_patch=(two_stream.diffuse_sunlit_absorption_patch_layer),
        diffuse_absorption_shaded_by_layer_patch=(two_stream.diffuse_shaded_absorption_patch_layer),
    )
    preceding_surface = template.canopy_surface.surface_radiation
    surface_radiation = SurfaceRadiationInputs(
        direct_shortwave_gridcell=preceding_surface.direct_shortwave_gridcell,
        diffuse_shortwave_gridcell=preceding_surface.diffuse_shortwave_gridcell,
        snow_layer_count_column=snow.geometry.snow_layer_count_column,
        snow_depth_column=snow.water.snow_depth_column,
        ground_albedo_direct_column=urban_surface.ground_direct_column,
        ground_albedo_diffuse_column=urban_surface.ground_diffuse_column,
        soil_albedo_direct_column=surface.soil.direct_column_band,
        soil_albedo_diffuse_column=surface.soil.diffuse_column_band,
        snow_albedo_direct_column=(surface.ground.snow_direct_albedo_history_column_band),
        snow_albedo_diffuse_column=(surface.ground.snow_diffuse_albedo_history_column_band),
        surface_albedo_direct_patch=urban_surface.surface_direct_patch,
        surface_albedo_diffuse_patch=urban_surface.surface_diffuse_patch,
        canopy_absorption_direct_patch=urban_canopy.absorbed_direct_patch,
        canopy_absorption_diffuse_patch=urban_canopy.absorbed_diffuse_patch,
        direct_transmittance_patch=urban_canopy.transmitted_direct_patch,
        direct_to_diffuse_transmittance_patch=(urban_canopy.transmitted_diffuse_from_direct_patch),
        diffuse_transmittance_patch=urban_canopy.transmitted_diffuse_patch,
        snow_absorption_direct_visible_column=(
            surface.ground.direct_visible_absorption_column_level
        ),
        snow_absorption_direct_nir_column=(surface.ground.direct_nir_absorption_column_level),
        snow_absorption_diffuse_visible_column=(
            surface.ground.diffuse_visible_absorption_column_level
        ),
        snow_absorption_diffuse_nir_column=(surface.ground.diffuse_nir_absorption_column_level),
    )
    preceding_urban = template.urban
    urban_radiation = UrbanRadiationInputs(
        direct_shortwave_gridcell=preceding_urban.direct_shortwave_gridcell,
        diffuse_shortwave_gridcell=preceding_urban.diffuse_shortwave_gridcell,
        downward_longwave_gridcell=preceding_urban.downward_longwave_gridcell,
        snow_fraction_column=carry.post_lake_hydrology.snow_fraction_column,
        ground_temperature_column=(carry.hydrology.final_diagnostics.ground_temperature_column),
        canyon_height_width_ratio_landunit=(preceding_urban.canyon_height_width_ratio_landunit),
        pervious_road_fraction_landunit=(preceding_urban.pervious_road_fraction_landunit),
        roof_emissivity_landunit=preceding_urban.roof_emissivity_landunit,
        impervious_road_emissivity_landunit=(preceding_urban.impervious_road_emissivity_landunit),
        pervious_road_emissivity_landunit=(preceding_urban.pervious_road_emissivity_landunit),
        wall_emissivity_landunit=preceding_urban.wall_emissivity_landunit,
    )
    return RadiationSequenceInputs(
        canopy_surface=RadiationDriverInputs(
            canopy_sunshade=canopy_sunshade,
            surface_radiation=surface_radiation,
        ),
        urban=urban_radiation,
    )


def next_radiation_state(carry: Any) -> RadiationSequenceState:

    preceding = carry.physical.radiation
    return RadiationSequenceState(
        canopy_surface=RadiationDriverState(
            canopy_sunshade=preceding.canopy_surface.canopy_sunshade,
            surface_radiation=preceding.canopy_surface.surface_radiation,
        ),
        urban=UrbanRadiationState(
            shortwave_absorption=carry.urban_albedo.shortwave_absorption,
            fluxes=preceding.urban.fluxes,
        ),
    )


def next_canopy_temperature_inputs(
    template: CanopyTemperatureInputs,
    carry: Any,
) -> CanopyTemperatureInputs:

    lake = carry.lake_hydrology
    drainage = carry.drainage
    vegetation = carry.ecosystem_post_drainage.vegetation_structure
    canopy = carry.physical.canopy.canopy_hydrology
    return CanopyTemperatureInputs(
        forcing_height_wind_gridcell=template.forcing_height_wind_gridcell,
        forcing_height_temperature_gridcell=(template.forcing_height_temperature_gridcell),
        forcing_height_humidity_gridcell=(template.forcing_height_humidity_gridcell),
        pressure_column=template.pressure_column,
        specific_humidity_column=template.specific_humidity_column,
        temperature_column=template.temperature_column,
        potential_temperature_column=template.potential_temperature_column,
        snow_layer_count_column=lake.geometry.snow_layer_count_column,
        soil_snow_temperature_column_level=(lake.temperature.soil_snow_temperature_column_level),
        surface_water_temperature_column=(carry.soil_temperature.surface_water_temperature_column),
        surface_water_fraction_column=canopy.surface_water_fraction_column,
        effective_snow_fraction_column=(lake.water.effective_snow_fraction_column),
        snow_fraction_column=carry.post_lake_hydrology.snow_fraction_column,
        soil_liquid_water_column_level=(drainage.soil_snow_liquid_water_column_level),
        soil_ice_water_column_level=(drainage.soil_snow_ice_water_column_level),
        vegetation_exposed_patch=(vegetation.fraction_vegetation_nosnow_albedo_patch),
        exposed_leaf_area_index_patch=vegetation.exposed_leaf_area_index_patch,
        exposed_stem_area_index_patch=vegetation.exposed_stem_area_index_patch,
        canopy_top_height_patch=vegetation.canopy_top_height_patch,
    )


def next_canopy_temperature_state(
    lake_inputs: LakeFluxDriverInputs,
    carry: Any,
) -> CanopyTemperatureState:

    column = _native_vector_index(
        lake_inputs.native_map.column_index,
        field="lake column map",
    )
    preceding = carry.physical.canopy_temperature
    lake_surface = carry.physical.lake_flux.surface
    column_state = preceding.column._replace(
        ground_temperature_column=(
            jnp.asarray(
                carry.hydrology.final_diagnostics.ground_temperature_column,
                dtype=jnp.float64,
            ).at[column].set(
                lake_surface.ground_temperature
            )
        ),
        ground_momentum_roughness_column=(
            preceding.column.ground_momentum_roughness_column.at[column].set(
                lake_surface.momentum_roughness
            )
        ),
        ground_sensible_roughness_column=(
            preceding.column.ground_sensible_roughness_column.at[column].set(
                lake_surface.sensible_roughness
            )
        ),
        ground_latent_roughness_column=(
            preceding.column.ground_latent_roughness_column.at[column].set(
                lake_surface.latent_roughness
            )
        ),
    )
    return CanopyTemperatureState(column=column_state, patch=preceding.patch)


def next_urban_flux_state(
    inputs: UrbanFluxDriverInputs,
    carry: Any,
) -> UrbanFluxDriverState:

    landunit = _native_vector_index(
        inputs.native_map.landunit_index,
        field="urban landunit map",
    )
    preceding = carry.physical.urban_flux
    building = carry.soil_temperature.building
    return UrbanFluxDriverState(
        turbulence=preceding.turbulence,
        internal_building_temperature_landunit=(building.building_temperature[landunit]),
        air_conditioning_flux_landunit=building.air_conditioning_flux[landunit],
        heating_flux_landunit=building.heating_flux[landunit],
        air_conditioning_flux_surface=preceding.air_conditioning_flux_surface,
        heating_flux_surface=preceding.heating_flux_surface,
    )


def next_lake_flux_state(
    inputs: LakeFluxDriverInputs,
    carry: Any,
) -> LakeFluxDriverState:

    column = _native_vector_index(
        inputs.native_map.column_index,
        field="lake column map",
    )
    preceding = carry.physical.lake_flux
    lake_temperature = carry.physical.lake_temperature
    final_lake = carry.lake_hydrology.temperature.lake_temperature_column_level
    return LakeFluxDriverState(
        surface=preceding.surface,
        lake_temperature_lake_level=final_lake[column],
        saved_top_eddy_conductivity_lake=(lake_temperature.saved_top_eddy_conductivity),
    )


def next_lake_temperature_state(
    inputs: LakeTemperatureDriverInputs,
    carry: Any,
) -> LakeColumnTemperatureState:

    column = _native_vector_index(
        inputs.native_map.column_index,
        field="lake-temperature column map",
    )
    preceding = carry.physical.lake_temperature
    lake = carry.lake_hydrology
    drainage = carry.drainage
    return LakeColumnTemperatureState(
        snow_layer_count=lake.geometry.snow_layer_count_column[column],
        snow_depth=lake.water.snow_depth_column[column],
        snow_water=lake.water.snow_water_equivalent_column[column],
        soil_snow_temperature=(lake.temperature.soil_snow_temperature_column_level[column]),
        soil_snow_liquid_water=(drainage.soil_snow_liquid_water_column_level[column]),
        soil_snow_ice_water=(drainage.soil_snow_ice_water_column_level[column]),
        soil_snow_old_ice_fraction=(lake.water.old_ice_fraction_column_level[column]),
        soil_snow_phase_flag=preceding.soil_snow_phase_flag,
        lake_temperature=(lake.temperature.lake_temperature_column_level[column]),
        lake_ice_fraction=lake.lake_ice_fraction_column_level[column],
        saved_top_eddy_conductivity=preceding.saved_top_eddy_conductivity,
        surface_absorption_fraction=preceding.surface_absorption_fraction,
        lake_surface_ice_fraction=preceding.lake_surface_ice_fraction,
        lake_ice_thickness=preceding.lake_ice_thickness,
        lake_resistance=preceding.lake_resistance,
        ground_methane_conductance=preceding.ground_methane_conductance,
    )


def _next_hydrology_snow_aerosol(carry: Any) -> Any:

    preceding = carry.hydrology.snow_structure.aerosol
    mass = carry.post_lake_hydrology.aerosol_mass.aerosol
    return preceding._replace(
        black_carbon_hydrophilic_layer=(mass.hydrophilic_black_carbon_layer),
        black_carbon_hydrophobic_layer=(mass.hydrophobic_black_carbon_layer),
        organic_carbon_hydrophilic_layer=(mass.hydrophilic_organic_carbon_layer),
        organic_carbon_hydrophobic_layer=(mass.hydrophobic_organic_carbon_layer),
        dust_species_1_layer=mass.dust_species_1_layer,
        dust_species_2_layer=mass.dust_species_2_layer,
        dust_species_3_layer=mass.dust_species_3_layer,
        dust_species_4_layer=mass.dust_species_4_layer,
        deposition=carry.lake_hydrology.aerosol.deposition,
    )


def next_hydrology_state(carry: Any) -> HydrologyNoDrainageState:

    preceding = carry.hydrology
    lake = carry.lake_hydrology
    post_lake = carry.post_lake_hydrology
    drainage = carry.drainage
    geometry = lake.geometry
    temperature = lake.temperature
    water = lake.water
    flux = lake.water_flux
    ground_slice = slice(NUMBER_OF_SNOW_LEVELS, None)

    final_liquid = drainage.soil_snow_liquid_water_column_level
    final_ice = drainage.soil_snow_ice_water_column_level
    final_volume = drainage.volumetric_water_column_level
    final_fraction = post_lake.snow_fraction_column
    final_grain = post_lake.snow_age.snow_grain_radius_column_level
    final_top_water = post_lake.aerosol_mass.top_snow_water_column
    final_aerosol = _next_hydrology_snow_aerosol(carry)

    snow_structure = preceding.snow_structure._replace(
        snow_layer_count_column=geometry.snow_layer_count_column,
        snow_fraction_column=final_fraction,
        effective_snow_fraction_column=water.effective_snow_fraction_column,
        snow_depth_column=water.snow_depth_column,
        integrated_snow_column=water.integrated_snow_column,
        snow_water_equivalent_column=water.snow_water_equivalent_column,
        soil_snow_thickness_column_level=(geometry.soil_snow_thickness_column_level),
        soil_snow_node_depth_column_level=(geometry.soil_snow_node_depth_column_level),
        soil_snow_interface_depth_column_interface=(
            geometry.soil_snow_interface_depth_column_interface
        ),
        soil_snow_temperature_column_level=(temperature.soil_snow_temperature_column_level),
        soil_snow_ice_column_level=final_ice,
        soil_snow_liquid_column_level=final_liquid,
        snow_grain_radius_column_level=final_grain,
        top_soil_water_flux_column=(flux.snow_layer_to_top_soil_flux_column),
        aerosol=final_aerosol,
    )
    snow_water = preceding.snow_water._replace(
        integrated_snow_column=water.integrated_snow_column,
        snow_fraction_column=final_fraction,
        snow_depth_column=water.snow_depth_column,
        soil_snow_liquid_column_level=final_liquid,
        soil_snow_ice_column_level=final_ice,
        soil_snow_thickness_column_level=(geometry.soil_snow_thickness_column_level),
        snow_drainage_flux_column=flux.snow_drainage_flux_column,
        top_soil_water_flux_column=flux.top_soil_water_flux_column,
        aerosol=final_aerosol,
    )
    surface = preceding.surface_hydrology._replace(
        soil_snow_liquid_column_level=final_liquid,
        surface_water_column=drainage.surface_water_column,
        top_soil_water_flux_column=flux.top_soil_water_flux_column,
        surface_runoff_flux_column=drainage.surface_runoff_flux_column,
        surface_water_runoff_flux_column=(drainage.surface_water_runoff_flux_column),
        infiltration_flux_column=drainage.infiltration_flux_column,
        soil_ice_fraction_column_level=drainage.soil_ice_fraction_column_level,
    )
    movement = preceding.soil_water_movement._replace(
        liquid_water_mass_column_level=final_liquid[:, ground_slice],
        volumetric_water_column_level=final_volume,
    )
    water_table = preceding.water_table_condensation._replace(
        soil_snow_liquid_water_column_level=final_liquid,
        soil_snow_ice_water_column_level=final_ice,
        volumetric_water_column_level=final_volume,
        water_table_depth_column=drainage.water_table_depth_column,
        perched_water_table_depth_column=(drainage.perched_water_table_depth_column),
        frost_table_depth_column=drainage.frost_table_depth_column,
        snow_sublimation_flux_column=flux.snow_sublimation_flux_column,
    )
    capping = preceding.snow_capping._replace(
        soil_snow_liquid_column_level=final_liquid,
        soil_snow_ice_column_level=final_ice,
        soil_snow_thickness_column_level=(geometry.soil_snow_thickness_column_level),
        snow_capping_ice_flux_column=flux.snow_capping_ice_flux_column,
        snow_capping_liquid_flux_column=flux.snow_capping_liquid_flux_column,
        snow_capping_discarded_ice_flux_column=(flux.snow_capping_discarded_ice_flux_column),
        snow_capping_discarded_liquid_flux_column=(flux.snow_capping_discarded_liquid_flux_column),
        aerosol=final_aerosol,
    )
    postprocess = preceding.snow_postprocess._replace(
        snow_layer_count_column=geometry.snow_layer_count_column,
        soil_snow_thickness_column_level=(geometry.soil_snow_thickness_column_level),
        soil_snow_node_depth_column_level=geometry.soil_snow_node_depth_column_level,
        soil_snow_interface_depth_column_interface=(
            geometry.soil_snow_interface_depth_column_interface
        ),
        soil_snow_temperature_column_level=(temperature.soil_snow_temperature_column_level),
        soil_snow_ice_column_level=final_ice,
        soil_snow_liquid_column_level=final_liquid,
        snow_ice_column=water.snow_ice_column,
        snow_liquid_column=water.snow_liquid_column,
        column_average_snow_depth_column=fortran_multiply(
            water.snow_depth_column,
            final_fraction,
        ),
        snow_temperature_mass_product_column=(temperature.snow_temperature_mass_sum_column),
        top_snow_water_column=final_top_water,
    )
    snow_age = post_lake.snow_age
    final = preceding.final_diagnostics._replace(
        total_soil_liquid_water_column=jnp.sum(
            final_liquid[:, ground_slice][:, :NUMBER_OF_SOIL_LEVELS],
            axis=1,
        ),
        total_soil_ice_water_column=jnp.sum(
            final_ice[:, ground_slice][:, :NUMBER_OF_SOIL_LEVELS],
            axis=1,
        ),
        volumetric_soil_water_column_level=final_volume,
        top_snow_water_column=final_top_water,
        snow_grain_radius_column_level=final_grain,
        top_snow_temperature_column=snow_age.top_snow_temperature_column,
        top_snow_temperature_gradient_column=(snow_age.top_snow_temperature_gradient_column),
        top_snow_grain_radius_column=snow_age.top_snow_grain_radius_column,
        top_snow_liquid_fraction_column=(snow_age.top_snow_liquid_fraction_column),
    )
    return preceding._replace(
        snow_structure=snow_structure,
        snow_water=snow_water,
        surface_hydrology=surface,
        soil_water_movement=movement,
        water_table_condensation=water_table,
        snow_capping=capping,
        snow_postprocess=postprocess,
        final_diagnostics=final,
    )


def next_soil_temperature_inputs(
    template: SoilTemperatureDriverInputs,
    carry: Any,
) -> SoilTemperatureDriverInputs:

    active = template.active_template
    state = carry.soil_temperature
    postsolve = active.postsolve_flux_template._replace(
        previous_updated_interface_heat_flux=(state.postsolve.updated_interface_heat_flux),
        previous_building_heat_flux=state.postsolve.building_heat_flux,
        previous_urban_air_conditioning_flux=(state.postsolve.urban_air_conditioning_flux),
        previous_urban_heating_flux=state.postsolve.urban_heating_flux,
    )
    standing = active.standing_water_phase_change_template._replace(
        previous_phase_change_heat_flux=(state.standing_water_phase.phase_change_heat_flux),
        previous_water_to_ice_flux=(state.standing_water_phase.water_to_ice_flux),
        previous_water_to_snow_heat_flux=(state.standing_water_phase.water_to_snow_heat_flux),
    )
    phase = active.soil_snow_phase_change_template._replace(
        previous_phase_flag=state.soil_snow_phase.phase_flag,
        previous_phase_change_heat_flux=(state.soil_snow_phase.phase_change_heat_flux),
        previous_snow_melt_flux=state.soil_snow_phase.snow_melt_flux,
        previous_snow_freeze_flux=state.soil_snow_phase.snow_freeze_flux,
        previous_snow_drain_flux=state.soil_snow_phase.snow_drain_flux,
        previous_snow_layer_melt_flux=(state.soil_snow_phase.snow_layer_melt_flux),
        previous_snow_layer_freeze_flux=(state.soil_snow_phase.snow_layer_freeze_flux),
        previous_snow_melt_heat_flux=(state.soil_snow_phase.snow_melt_heat_flux),
        previous_rural_snow_melt_heat_flux=(state.soil_snow_phase.rural_snow_melt_heat_flux),
        previous_urban_snow_melt_heat_flux=(state.soil_snow_phase.urban_snow_melt_heat_flux),
    )
    building = active.building_temperature_template._replace(
        roof_inner_temperature=state.building.roof_inner_temperature,
        sunwall_inner_temperature=state.building.sunwall_inner_temperature,
        shadewall_inner_temperature=state.building.shadewall_inner_temperature,
        floor_temperature=state.building.floor_temperature,
        building_temperature=state.building.building_temperature,
        previous_building_heat_flux=state.building.building_heat_flux,
        previous_air_conditioning_flux=state.building.air_conditioning_flux,
        previous_heating_flux=state.building.heating_flux,
    )
    final = active.final_diagnostics_template._replace(
        previous_layer_one_two_heat_flux=state.final.layer_one_two_heat_flux,
        previous_ground_layer_heat_flux=state.final.ground_layer_heat_flux,
    )
    return SoilTemperatureDriverInputs(
        active_template=active._replace(
            ground_heat_flux_patch_state=state.ground_heat_flux_patch_state,
            postsolve_flux_template=postsolve,
            standing_water_phase_change_template=standing,
            soil_snow_phase_change_template=phase,
            building_temperature_template=building,
            final_diagnostics_template=final,
        ),
        urban=template.urban,
    )


def next_physical_state(
    inputs: PhysicalDriverInputs,
    carry: Any,
) -> PhysicalDriverState:

    preceding = carry.physical
    return PhysicalDriverState(
        atmospheric=preceding.atmospheric,
        canopy=next_canopy_state(carry),
        radiation=next_radiation_state(carry),
        canopy_temperature=next_canopy_temperature_state(
            inputs.lake_flux,
            carry,
        ),
        bare_ground=preceding.bare_ground,
        canopy_flux=next_canopy_flux_state(carry),
        urban_flux=next_urban_flux_state(inputs.urban_flux, carry),
        lake_flux=next_lake_flux_state(inputs.lake_flux, carry),
        dust=preceding.dust,
        voc=preceding.voc,
        lake_temperature=next_lake_temperature_state(
            inputs.lake_temperature,
            carry,
        ),
    )


@jax.jit
def _phenology_transfer_pools(vegetation):
    return tuple(jnp.stack(tuple(
        get_cn_precision_control_state_field(vegetation, f"vegetation_{element}.{name}_patch")
        for name in names
    ), axis=1) for element, names in (
        ("carbon", CARBON_TRANSFER_POOL_NAMES), ("nitrogen", NITROGEN_TRANSFER_POOL_NAMES),
    ))


def next_ecosystem_state_from_parts(
    state: EcosystemDynamicsAnnualUpdateState,
    post_drainage_state: Any,
) -> EcosystemDynamicsAnnualUpdateState:

    post = post_drainage_state.biogeochemistry
    final_soil = post.soil
    final_fluxes = post.cn_fluxes

    update3 = state.state_update3_precision
    fire_flux = update3.fire_flux
    fire_area = fire_flux.fire_area
    products = fire_area.wood_products
    update2h = products.state_update2h_precision
    update2 = update2h.state_update2
    gap = update2.gap_mortality
    transport = gap.litter_vertical_transport
    soil_n = transport.soil_n_state_update
    second = soil_n.second_guardrail
    first = second.first_guardrail
    ecosystem = first.ecosystem
    decomposition = ecosystem.pre_decomposition
    prefix = decomposition.prefix
    middle = decomposition.competition_decomposition

    final_carbon_transfer, final_nitrogen_transfer = _phenology_transfer_pools(post.vegetation)
    synchronized_phenology = prefix.seasonal_deciduous_phenology._replace(
        carbon_transfer_patch_pool=final_carbon_transfer,
        nitrogen_transfer_patch_pool=final_nitrogen_transfer,
    )

    synchronized_prefix = prefix._replace(
        cn_fluxes=final_fluxes,
        seasonal_deciduous_phenology=synchronized_phenology,
        plant_nutrient_demand=prefix.plant_nutrient_demand._replace(
            cn_fluxes=final_fluxes,
        ),
    )
    synchronized_middle = middle._replace(
        plant_nutrient_competition=middle.plant_nutrient_competition._replace(
            cn_fluxes=final_fluxes,
        ),
        soil_biogeochem_decomp=middle.soil_biogeochem_decomp._replace(
            cn_fluxes=final_fluxes,
        ),
    )
    synchronized_first = first._replace(
        ecosystem=ecosystem._replace(
            pre_decomposition=decomposition._replace(
                prefix=synchronized_prefix,
                competition_decomposition=synchronized_middle,
            ),
        ),
        vegetation_carbon_nitrogen=post.vegetation,
    )
    synchronized_soil_n = soil_n._replace(
        second_guardrail=second._replace(first_guardrail=synchronized_first),
        soil_nitrogen=SoilBiogeochemNStateUpdate1State(
            mineral_nitrogen_column_level=(final_soil.mineral_nitrogen_column_level),
            ammonium_column_level=final_soil.ammonium_column_level,
            nitrate_column_level=final_soil.nitrate_column_level,
        ),
    )
    synchronized_transport = transport._replace(
        soil_n_state_update=synchronized_soil_n,
        vertical_transport=transport.vertical_transport._replace(
            carbon_pool_column_level_pool=(final_soil.carbon_pool_column_level_pool),
            nitrogen_pool_column_level_pool=(final_soil.nitrogen_pool_column_level_pool),
        ),
    )
    synchronized_gap = gap._replace(
        litter_vertical_transport=synchronized_transport,
    )
    synchronized_update2 = update2._replace(gap_mortality=synchronized_gap)
    synchronized_update2h = update2h._replace(state_update2=synchronized_update2)
    synchronized_products = products._replace(
        state_update2h_precision=synchronized_update2h,
    )
    synchronized_fire_area = fire_area._replace(
        wood_products=synchronized_products,
    )
    synchronized_fire_flux = fire_flux._replace(
        fire_area=synchronized_fire_area,
    )
    return state._replace(
        state_update3_precision=update3._replace(
            fire_flux=synchronized_fire_flux,
        ),
        annual=state.annual._replace(
            temporary_average_air_temperature_patch=(
                prefix.phenology_climate.temperature_average_2m_patch
            ),
            temporary_sum_npp_patch=(
                post_drainage_state.flux_summary.temporary_npp_sum_patch
            ),
        ),
    )


def next_ecosystem_state(carry: Any) -> EcosystemDynamicsAnnualUpdateState:

    return next_ecosystem_state_from_parts(
        carry.ecosystem_pre_drainage,
        carry.ecosystem_post_drainage,
    )


__all__ = [
    "next_atmospheric_inputs",
    "next_canopy_flux_inputs",
    "next_canopy_flux_state",
    "next_canopy_inputs",
    "next_canopy_state",
    "next_canopy_temperature_inputs",
    "next_canopy_temperature_state",
    "next_ecosystem_state",
    "next_ecosystem_state_from_parts",
    "next_hydrology_state",
    "next_dust_inputs",
    "next_lake_flux_state",
    "next_lake_temperature_state",
    "next_physical_state",
    "next_radiation_inputs",
    "next_radiation_state",
    "next_soil_temperature_inputs",
    "next_urban_flux_state",
]
