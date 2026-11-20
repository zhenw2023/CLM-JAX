
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.forcing.datm import DatmForcing
from clm_jax.io.methane_driver import MethaneDriverNativeInputs
from clm_jax.physics.canopy_flux_driver import (
    CanopyFluxDriverResult,
    CanopyFluxRuntimeResult,
    CanopyFluxSoilTemperatureFields,
    canopy_flux_soil_temperature_fields,
)
from clm_jax.physics.cn_zero import get_cn_flux_field
from clm_jax.physics.ecosystem_dynamics_carry import (
    EcosystemDynamicsPostDrainageCarry,
)
from clm_jax.physics.ecosystem_dynamics_post_drainage import (
    EcosystemDynamicsPostDrainageResult,
)
from clm_jax.physics.hydrology_drainage_driver import (
    HydrologyDrainageDriverResult,
)
from clm_jax.physics.hydrology_driver import HydrologyDriverResult
from clm_jax.physics.lake_hydrology_driver import LakeHydrologyDriverResult
from clm_jax.physics.methane_balance_initialization import (
    MethaneBalanceInitializationResult,
)
from clm_jax.physics.methane_biogeochemistry import (
    MethaneAerenchymaInputs,
    MethaneAnnualUpdateInputs,
    MethaneEbullitionInputs,
    MethaneOxidationInputs,
    MethaneProductionInputs,
    MethaneWaterTableInputs,
)
from clm_jax.physics.methane_driver import (
    MethaneDriverFinalizeInputs,
    MethaneDriverPreprocessInputs,
)
from clm_jax.physics.methane_driver_facade import (
    MethaneDriverActiveInputs,
    MethaneDriverActiveState,
)
from clm_jax.physics.methane_soil_gas_transport import (
    MethaneSoilGasTransportInputs,
)
from clm_jax.physics.physical_driver import (
    PhysicalDriverResult,
    PhysicalDriverRuntimeResult,
)
from clm_jax.physics.snow_water import NUMBER_OF_SNOW_LEVELS


class MethaneDriverCarry(NamedTuple):

    beginning_methane_carbon_column: jnp.ndarray
    carbon_root_fraction_patch_level: jnp.ndarray


def methane_driver_carry(
    balance: MethaneBalanceInitializationResult,
    carbon_root_fraction_patch_level: jnp.ndarray,
) -> MethaneDriverCarry:

    beginning = jnp.asarray(
        balance.state.beginning_methane_carbon_column,
        dtype=jnp.float64,
    )
    root_fraction = jnp.asarray(
        carbon_root_fraction_patch_level,
        dtype=jnp.float64,
    )
    if beginning.ndim != 1:
        raise ValueError("beginning methane carbon must be a column vector")
    if root_fraction.ndim != 2:
        raise ValueError("carbon root fraction must be a patch-by-level matrix")
    return MethaneDriverCarry(beginning, root_fraction)


def methane_driver_carry_from_state(
    state: MethaneDriverActiveState,
    carbon_root_fraction_patch_level: jnp.ndarray,
) -> MethaneDriverCarry:

    beginning = jnp.asarray(
        state.finalize.total_methane_carbon_column,
        dtype=jnp.float64,
    )
    root_fraction = jnp.asarray(
        carbon_root_fraction_patch_level,
        dtype=jnp.float64,
    )
    if beginning.ndim != 1:
        raise ValueError("final methane carbon must be a column vector")
    if root_fraction.ndim != 2:
        raise ValueError("carbon root fraction must be a patch-by-level matrix")
    return MethaneDriverCarry(beginning, root_fraction)


def _canopy_flux_projection(
    result: CanopyFluxDriverResult | CanopyFluxRuntimeResult,
) -> CanopyFluxSoilTemperatureFields:
    if isinstance(result, CanopyFluxRuntimeResult):
        return result.soil_temperature
    if isinstance(result, CanopyFluxDriverResult):
        return canopy_flux_soil_temperature_fields(result)
    raise TypeError("unsupported canopy flux result at methane boundary")


def assemble_methane_driver_inputs(
    native: MethaneDriverNativeInputs,
    state: MethaneDriverActiveState,
    forcing: DatmForcing,
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    hydrology: HydrologyDriverResult,
    lake_hydrology: LakeHydrologyDriverResult,
    drainage: HydrologyDrainageDriverResult,
    post_drainage: EcosystemDynamicsPostDrainageResult,
    ecosystem_carry: EcosystemDynamicsPostDrainageCarry,
    methane_carry: MethaneDriverCarry,
    *,
    total_water_storage_gridcell: jnp.ndarray | None = None,
) -> MethaneDriverActiveInputs:

    column_count = native.methane_column.shape[0]
    patch_count = native.methane_patch.shape[0]
    gridcell_count = native.latitude_degrees_gridcell.shape[0]
    if methane_carry.beginning_methane_carbon_column.shape != (column_count,):
        raise ValueError("methane balance carry and native column axes disagree")
    if methane_carry.carbon_root_fraction_patch_level.shape != (patch_count, 25):
        raise ValueError("methane carbon-root carry must have shape (patch, 25)")
    total_water_storage = jnp.asarray(
        native.total_water_storage_gridcell
        if total_water_storage_gridcell is None
        else total_water_storage_gridcell,
        dtype=jnp.float64,
    )
    if total_water_storage.shape != (gridcell_count,):
        raise ValueError("methane total-water storage must contain every gridcell")

    water = drainage.drainage.state
    lake = lake_hydrology.hydrology.state
    canopy = _canopy_flux_projection(physical.canopy_flux)
    ground_conductance = canopy.ground_methane_conductance_patch
    if physical.bare_ground is not None:
        bare_patch = native.methane_patch & ~jnp.asarray(
            physical.bare_ground.inputs.vegetation_exposed_patch, dtype=jnp.bool_
        )
        ground_conductance = jnp.where(
            bare_patch,
            physical.bare_ground.state.ground_methane_conductance_patch,
            ground_conductance,
        )
    cn_fluxes = post_drainage.state.biogeochemistry.cn_fluxes
    soil_slice = slice(NUMBER_OF_SNOW_LEVELS, None)
    snow_slice = slice(0, NUMBER_OF_SNOW_LEVELS)

    soil_temperature = lake.temperature.soil_snow_temperature_column_level[:, soil_slice]
    snow_temperature = lake.temperature.soil_snow_temperature_column_level[:, snow_slice]
    soil_thickness = lake.geometry.soil_snow_thickness_column_level[:, soil_slice]
    snow_thickness = lake.geometry.soil_snow_thickness_column_level[:, snow_slice]
    soil_liquid = water.soil_snow_liquid_water_column_level[:, soil_slice]
    soil_ice = water.soil_snow_ice_water_column_level[:, soil_slice]
    snow_liquid = water.soil_snow_liquid_water_column_level[:, snow_slice]
    snow_ice = water.soil_snow_ice_water_column_level[:, snow_slice]

    atmospheric_temperature = jnp.asarray(forcing.temperature, dtype=jnp.float64)
    atmospheric_pressure = jnp.asarray(forcing.pressure, dtype=jnp.float64)
    if atmospheric_pressure.shape != (gridcell_count,):
        raise ValueError("DATM forcing and methane land-grid axes disagree")
    oxygen_partial_pressure = (
        jnp.float64(native.atmospheric_oxygen_mole_fraction) * atmospheric_pressure
    )
    carbon_dioxide_partial_pressure = (
        jnp.float64(native.atmospheric_carbon_dioxide_mole_fraction) * atmospheric_pressure
    )
    methane_partial_pressure = (
        jnp.float64(native.atmospheric_methane_mole_fraction) * atmospheric_pressure
    )
    zero_gridcell = jnp.zeros_like(atmospheric_pressure)

    preprocess = MethaneDriverPreprocessInputs(
        methane_column=native.methane_column,
        column_gridcell=native.column_gridcell,
        latitude_degrees_gridcell=native.latitude_degrees_gridcell,
        atmospheric_temperature_gridcell=atmospheric_temperature,
        atmospheric_bottom_pressure_gridcell=atmospheric_pressure,
        atmospheric_oxygen_partial_pressure_gridcell=oxygen_partial_pressure,
        atmospheric_carbon_dioxide_partial_pressure_gridcell=(carbon_dioxide_partial_pressure),
        atmospheric_methane_partial_pressure_gridcell=methane_partial_pressure,
        surface_runoff_column=water.surface_runoff_flux_column,
        surface_water_fraction_column=(
            hydrology.inputs.surface_hydrology.surface_water_fraction_column
        ),
        water_table_depth_column=water.water_table_depth_column,
        perched_water_table_depth_column=water.perched_water_table_depth_column,
        soil_bottom_depth_column=native.soil_interface_depth_column_interface[:, 20],
        total_water_storage_gridcell=total_water_storage,
        water_table_scale_gridcell=zero_gridcell,
        water_table_inundation_scale_gridcell=zero_gridcell,
        runoff_inundation_scale_gridcell=zero_gridcell,
        total_water_storage_slope_gridcell=(native.total_water_storage_slope_gridcell),
        total_water_storage_intercept_gridcell=(native.total_water_storage_intercept_gridcell),
        snow_depth_column=lake.water.snow_depth_column,
        soil_layer_thickness_column_level=soil_thickness,
        unsaturated_methane_concentration_column_level=(
            state.transport.unsaturated_methane_concentration_column_level
        ),
    )

    annual = MethaneAnnualUpdateInputs(
        methane_column=native.methane_column,
        methane_patch=native.methane_patch,
        patch_column=native.patch_column,
        soil_heterotrophic_respiration_column=get_cn_flux_field(
            cn_fluxes,
            "soil_carbon.somhr_col",
        ),
        inundated_fraction_column=state.preprocess.inundated_fraction_column,
        aboveground_npp_patch=get_cn_flux_field(
            cn_fluxes,
            "vegetation_carbon.agnpp_patch",
        ),
        belowground_npp_patch=get_cn_flux_field(
            cn_fluxes,
            "vegetation_carbon.bgnpp_patch",
        ),
    )

    water_table = MethaneWaterTableInputs(
        methane_column=native.methane_column,
        soil_porosity_column_level=native.soil_porosity_column_level,
        volumetric_soil_water_column_level=water.volumetric_water_column_level,
        soil_temperature_column_level=soil_temperature,
    )

    production = MethaneProductionInputs(
        methane_column=native.methane_column,
        methane_patch=native.methane_patch,
        patch_column=native.patch_column,
        patch_weight_on_column=native.patch_weight_on_column,
        vegetated_patch=native.vegetated_patch,
        root_respiration_patch=get_cn_flux_field(
            cn_fluxes,
            "vegetation_carbon.rr_patch",
        ),
        carbon_root_fraction_patch_level=(methane_carry.carbon_root_fraction_patch_level),
        water_table_layer_column=state.water_table.layer_above_water_table_column,
        soil_layer_thickness_column_level=soil_thickness,
        soil_temperature_column_level=soil_temperature,
        soil_heterotrophic_respiration_column=get_cn_flux_field(
            cn_fluxes,
            "soil_carbon.somhr_col",
        ),
        litter_heterotrophic_respiration_column=get_cn_flux_field(
            cn_fluxes,
            "soil_carbon.lithr_col",
        ),
        vertically_resolved_heterotrophic_respiration_column_level=(
            get_cn_flux_field(cn_fluxes, "soil_carbon.hr_vr_col")
        ),
        oxygen_decomposition_scalar_column_level=(
            ecosystem_carry.oxygen_decomposition_scalar_column_level
        ),
        potential_nitrification_column_level=get_cn_flux_field(
            cn_fluxes,
            "soil_nitrogen.pot_f_nit_vr_col",
        ),
        inundated_fraction_column=state.preprocess.inundated_fraction_column,
        lagged_inundated_fraction_column=(state.preprocess.lagged_inundated_fraction_column),
        lagged_layer_saturation_column_level=(state.lagged_layer_saturation_column_level),
    )

    oxidation = MethaneOxidationInputs(
        methane_column=native.methane_column,
        water_table_layer_column=state.water_table.layer_above_water_table_column,
        volumetric_soil_water_column_level=water.volumetric_water_column_level,
        soil_porosity_column_level=native.soil_porosity_column_level,
        soil_matrix_potential_column_level=(
            hydrology.hydrology.state.final_diagnostics.soil_matrix_potential_column_level
        ),
        soil_temperature_column_level=soil_temperature,
        unsaturated_methane_concentration_column_level=(
            state.transport.unsaturated_methane_concentration_column_level
        ),
        saturated_methane_concentration_column_level=(
            state.transport.saturated_methane_concentration_column_level
        ),
        unsaturated_oxygen_concentration_column_level=(
            state.transport.unsaturated_oxygen_concentration_column_level
        ),
        saturated_oxygen_concentration_column_level=(
            state.transport.saturated_oxygen_concentration_column_level
        ),
    )

    aerenchyma = MethaneAerenchymaInputs(
        methane_column=native.methane_column,
        methane_patch=native.methane_patch,
        patch_column=native.patch_column,
        column_gridcell=native.column_gridcell,
        patch_weight_on_column=native.patch_weight_on_column,
        vegetated_patch=native.vegetated_patch,
        grass_or_crop_patch=native.grass_or_crop_patch,
        annual_npp_patch=ecosystem_carry.annual_sum_npp_patch,
        annual_average_aboveground_npp_patch=(state.annual.annual_aboveground_npp_mean_patch),
        annual_average_belowground_npp_patch=(state.annual.annual_belowground_npp_mean_patch),
        effective_root_fraction_patch_level=(native.effective_root_fraction_patch_level),
        root_fraction_patch_level=native.root_fraction_patch_level,
        transpiration_patch=canopy.vegetation_transpiration_patch,
        ground_methane_conductance_patch=ground_conductance,
        water_table_layer_column=state.water_table.layer_above_water_table_column,
        layer_depth_column_level=native.soil_layer_depth_column_level,
        layer_thickness_column_level=soil_thickness,
        soil_temperature_column_level=soil_temperature,
        soil_porosity_column_level=native.soil_porosity_column_level,
        volumetric_soil_water_column_level=water.volumetric_water_column_level,
        atmospheric_methane_concentration_gridcell=zero_gridcell,
        atmospheric_oxygen_concentration_gridcell=zero_gridcell,
        unsaturated_methane_concentration_column_level=(
            state.transport.unsaturated_methane_concentration_column_level
        ),
        saturated_methane_concentration_column_level=(
            state.transport.saturated_methane_concentration_column_level
        ),
        unsaturated_oxygen_concentration_column_level=(
            state.transport.unsaturated_oxygen_concentration_column_level
        ),
        saturated_oxygen_concentration_column_level=(
            state.transport.saturated_oxygen_concentration_column_level
        ),
        unsaturated_methane_production_column_level=(
            state.production.unsaturated_methane_production_column_level
        ),
        saturated_methane_production_column_level=(
            state.production.saturated_methane_production_column_level
        ),
    )

    ebullition = MethaneEbullitionInputs(
        methane_column=native.methane_column,
        water_table_layer_column=state.water_table.layer_above_water_table_column,
        layer_depth_column_level=native.soil_layer_depth_column_level,
        interface_depth_column_interface=(native.soil_interface_depth_column_interface),
        soil_temperature_column_level=soil_temperature,
        soil_porosity_column_level=native.soil_porosity_column_level,
        atmospheric_pressure_column=physical.atmospheric.column_forcing.pressure,
        surface_water_depth_mm_column=water.surface_water_column,
        surface_water_fraction_column=(
            hydrology.inputs.surface_hydrology.surface_water_fraction_column
        ),
        unsaturated_methane_concentration_column_level=(
            state.transport.unsaturated_methane_concentration_column_level
        ),
        saturated_methane_concentration_column_level=(
            state.transport.saturated_methane_concentration_column_level
        ),
    )

    transport = MethaneSoilGasTransportInputs(
        methane_column=native.methane_column,
        column_gridcell=native.column_gridcell,
        unsaturated_water_table_layer_column=(state.water_table.layer_above_water_table_column),
        snow_layer_index_column=lake.geometry.snow_layer_count_column,
        soil_layer_thickness_column_level=soil_thickness,
        snow_layer_thickness_column_level=snow_thickness,
        soil_b_parameter_column_level=native.soil_b_parameter_column_level,
        soil_porosity_column_level=native.soil_porosity_column_level,
        soil_organic_matter_column_level=(native.soil_organic_matter_column_level),
        soil_temperature_column_level=soil_temperature,
        snow_temperature_column_level=snow_temperature,
        ground_temperature_column=lake_hydrology.inputs.ground_temperature_column,
        surface_water_temperature_column=(
            hydrology.inputs.final_diagnostics.surface_water_temperature_column
        ),
        surface_water_fraction_column=(
            hydrology.inputs.surface_hydrology.surface_water_fraction_column
        ),
        surface_water_depth_mm_column=water.surface_water_column,
        volumetric_soil_water_column_level=water.volumetric_water_column_level,
        soil_liquid_water_kg_m2_column_level=soil_liquid,
        soil_ice_kg_m2_column_level=soil_ice,
        snow_liquid_water_kg_m2_column_level=snow_liquid,
        snow_ice_kg_m2_column_level=snow_ice,
        atmospheric_methane_gridcell=zero_gridcell,
        atmospheric_oxygen_gridcell=zero_gridcell,
        unsaturated_methane_production_column_level=(
            state.production.unsaturated_methane_production_column_level
        ),
        saturated_methane_production_column_level=(
            state.production.saturated_methane_production_column_level
        ),
        unsaturated_oxygen_aerenchyma_column_level=(
            state.aerenchyma.unsaturated_oxygen_gain_column_level
        ),
        saturated_oxygen_aerenchyma_column_level=(
            state.aerenchyma.saturated_oxygen_gain_column_level
        ),
    )

    finalize = MethaneDriverFinalizeInputs(
        methane_column=native.methane_column,
        nonlake_column=native.nonlake_column,
        lake_column=native.lake_column,
        column_active=native.column_active,
        column_gridcell=native.column_gridcell,
        column_weight_on_gridcell=native.column_weight_on_gridcell,
        inundated_fraction_column=state.preprocess.inundated_fraction_column,
        soil_layer_thickness_column_level=soil_thickness,
        saturated_methane_concentration_column_level=(
            state.transport.saturated_methane_concentration_column_level
        ),
        unsaturated_methane_concentration_column_level=(
            state.transport.unsaturated_methane_concentration_column_level
        ),
        saturated_surface_diffusion_column=(state.transport.saturated_surface_diffusion_column),
        unsaturated_surface_diffusion_column=(state.transport.unsaturated_surface_diffusion_column),
        saturated_surface_aerenchyma_column=(state.transport.saturated_surface_aerenchyma_column),
        unsaturated_surface_aerenchyma_column=(
            state.transport.unsaturated_surface_aerenchyma_column
        ),
        saturated_surface_ebullition_column=(state.transport.saturated_surface_ebullition_column),
        unsaturated_surface_ebullition_column=(
            state.transport.unsaturated_surface_ebullition_column
        ),
        saturated_methane_oxidation_column_level=(
            state.oxidation.saturated_methane_oxidation_column_level
        ),
        unsaturated_methane_oxidation_column_level=(
            state.oxidation.unsaturated_methane_oxidation_column_level
        ),
        saturated_methane_production_column_level=(
            state.production.saturated_methane_production_column_level
        ),
        unsaturated_methane_production_column_level=(
            state.production.unsaturated_methane_production_column_level
        ),
        inundation_change_surface_flux_column=(
            state.preprocess.inundation_change_surface_flux_column
        ),
        beginning_methane_carbon_column=(methane_carry.beginning_methane_carbon_column),
    )
    return MethaneDriverActiveInputs(
        preprocess=preprocess,
        annual=annual,
        water_table=water_table,
        production=production,
        oxidation=oxidation,
        aerenchyma=aerenchyma,
        ebullition=ebullition,
        transport=transport,
        finalize=finalize,
    )


__all__ = [
    "MethaneDriverCarry",
    "assemble_methane_driver_inputs",
    "methane_driver_carry",
    "methane_driver_carry_from_state",
]
