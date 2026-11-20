
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import get_opaque_trace_state

from clm_jax.compilation import _EAGER_TRACE, process_lifetime_jit
from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.forcing.datm import DatmForcing
from clm_jax.io.crop import FIRST_PROGNOSTIC_CROP_PFT
from clm_jax.io.decomposition_rates import DecompositionRatePersistentState
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    SubgridTopology,
)
from clm_jax.io.soil_biogeochem_vertical_profile import (
    SoilBiogeochemVerticalProfileNativeData,
    assemble_soil_biogeochem_vertical_profile_inputs,
)
from clm_jax.physics.active_layer import ActiveLayerResult
from clm_jax.physics.carbon_nitrogen_balance_initialization import (
    CarbonNitrogenStateSummaries,
)
from clm_jax.physics.diagnostics_and_accumulators import ConfiguredAccumulatorState
from clm_jax.physics.ecosystem_dynamics import EcosystemDynamicsAnnualUpdateState
from clm_jax.physics.filters import exposed_vegetation_masks
from clm_jax.physics.hydrology_diagnostics import (
    NUMBER_OF_SOIL_LEVELS,
    NUMBER_OF_URBAN_LEVELS,
)
from clm_jax.physics.hydrology_driver import HydrologyDriverResult
from clm_jax.physics.lake_hydrology_driver import LakeHydrologyDriverResult
from clm_jax.physics.physical_driver import (
    PhysicalDriverResult,
    PhysicalDriverRuntimeResult,
)
from clm_jax.physics.plant_hydraulics import jackson_root_fraction
from clm_jax.physics.post_temperature_driver import PostTemperatureNativeMap
from clm_jax.physics.snow_water import NUMBER_OF_SNOW_LEVELS
from clm_jax.physics.soil_biogeochem_vertical_profile import (
    NUMBER_OF_GROUND_LEVELS,
    soil_biogeochem_vertical_profile,
)
from clm_jax.physics.soil_biogeochem_vertical_profile import (
    _validate_shapes as _validate_profile_shapes,
)


class EcosystemFastGridcell(NamedTuple):
    domain_index: jnp.ndarray
    relative_humidity: jnp.ndarray
    wind_speed: jnp.ndarray
    nitrogen_deposition: jnp.ndarray


class EcosystemFastColumn(NamedTuple):
    domain_gridcell: jnp.ndarray
    column_type: jnp.ndarray
    landunit_type: jnp.ndarray
    active: jnp.ndarray
    weight_on_gridcell: jnp.ndarray
    downscaled_temperature: jnp.ndarray
    downscaled_rain: jnp.ndarray
    downscaled_snow: jnp.ndarray
    annual_evapotranspiration: jnp.ndarray
    topsoil_water_fraction: jnp.ndarray
    snow_depth: jnp.ndarray
    saturated_area_fraction: jnp.ndarray
    inundated_fraction: jnp.ndarray
    soil_temperature_17cm: jnp.ndarray
    maximum_thaw_depth: jnp.ndarray
    previous_maximum_thaw_depth: jnp.ndarray
    active_layer_index: jnp.ndarray
    soil_water_potential: jnp.ndarray
    soil_liquid_water: jnp.ndarray
    soil_volumetric_water: jnp.ndarray
    soil_temperature: jnp.ndarray
    oxygen_stress_saturated: jnp.ndarray
    oxygen_stress_unsaturated: jnp.ndarray
    total_litter_carbon: jnp.ndarray
    total_soil_organic_carbon: jnp.ndarray
    total_vegetation_carbon: jnp.ndarray


class EcosystemFastPatch(NamedTuple):
    domain_gridcell: jnp.ndarray
    vegetation_type: jnp.ndarray
    column_domain_gridcell: jnp.ndarray
    column_type: jnp.ndarray
    landunit_type: jnp.ndarray
    active: jnp.ndarray
    weight_on_gridcell: jnp.ndarray
    weight_on_column: jnp.ndarray
    vegetation_exposed: jnp.ndarray
    sunlit_leaf_area: jnp.ndarray
    shaded_leaf_area: jnp.ndarray
    total_leaf_area: jnp.ndarray
    sunlit_leaf_respiration: jnp.ndarray
    shaded_leaf_respiration: jnp.ndarray
    sunlit_photosynthesis: jnp.ndarray
    shaded_photosynthesis: jnp.ndarray
    transpiration_wetness: jnp.ndarray
    root_zone_wetness: jnp.ndarray
    precipitation_60day: jnp.ndarray
    precipitation_10day: jnp.ndarray
    relative_humidity_30day: jnp.ndarray
    reference_temperature_2m: jnp.ndarray
    minimum_temperature_2m: jnp.ndarray
    maximum_temperature_2m: jnp.ndarray
    temperature_10day: jnp.ndarray
    minimum_temperature_5day: jnp.ndarray
    minimum_temperature_10day: jnp.ndarray
    growing_degree_days_base_0: jnp.ndarray
    growing_degree_days_base_8: jnp.ndarray
    growing_degree_days_base_10: jnp.ndarray
    growing_degree_days_base_0_twenty_year: jnp.ndarray
    growing_degree_days_base_8_twenty_year: jnp.ndarray
    growing_degree_days_base_10_twenty_year: jnp.ndarray
    years_crop_active: jnp.ndarray
    crop_live: jnp.ndarray
    crop_planted: jnp.ndarray
    harvest_date: jnp.ndarray
    fertilizer_nitrogen: jnp.ndarray
    growing_degree_days_since_planting: jnp.ndarray
    growing_degree_days_topsoil: jnp.ndarray
    carbon_root_fraction: jnp.ndarray


class EcosystemFastBoundary(NamedTuple):

    gridcell: EcosystemFastGridcell
    column: EcosystemFastColumn
    patch: EcosystemFastPatch


def ecosystem_pre_drainage_prefix_state(
    state: EcosystemDynamicsAnnualUpdateState,
):

    return (
        state.state_update3_precision.fire_flux.fire_area.wood_products
        .state_update2h_precision.state_update2.gap_mortality
        .litter_vertical_transport.soil_n_state_update.second_guardrail
        .first_guardrail.ecosystem.pre_decomposition.prefix
    )


def _replace_ecosystem_pre_drainage_prefix_state(
    state: EcosystemDynamicsAnnualUpdateState,
    prefix,
) -> EcosystemDynamicsAnnualUpdateState:
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
    decomposition = ecosystem.pre_decomposition._replace(prefix=prefix)
    ecosystem = ecosystem._replace(pre_decomposition=decomposition)
    first = first._replace(ecosystem=ecosystem)
    second = second._replace(first_guardrail=first)
    soil_n = soil_n._replace(second_guardrail=second)
    transport = transport._replace(soil_n_state_update=soil_n)
    gap = gap._replace(litter_vertical_transport=transport)
    update2 = update2._replace(gap_mortality=gap)
    update2h = update2h._replace(state_update2=update2)
    products = products._replace(state_update2h_precision=update2h)
    fire_area = fire_area._replace(wood_products=products)
    fire_flux = fire_flux._replace(fire_area=fire_area)
    update3 = update3._replace(fire_flux=fire_flux)
    return state._replace(state_update3_precision=update3)


def _early_vertical_profile_state(inputs, state, parameters):
    return soil_biogeochem_vertical_profile(inputs, state, parameters).state


_early_profile_entry = None


def _advance_early_vertical_profile(inputs, state, parameters, *, jit):
    global _early_profile_entry
    advance = process_lifetime_jit(_early_vertical_profile_state, enabled=jit)
    if (not jit or get_opaque_trace_state() != _EAGER_TRACE
            or not isinstance(parameters.decomposition_full_level_count, int)
            or parameters.decomposition_full_level_count != NUMBER_OF_GROUND_LEVELS):
        return advance(inputs, state, parameters)
    sources, structure = jax.tree.flatten((inputs, parameters))
    signature = []
    for value in sources:
        if isinstance(value, jax.Array):
            if value.is_deleted():
                return advance(inputs, state, parameters)
            signature.append(id(value))
        elif isinstance(value, (bool, int, float, np.generic)):
            scalar = np.asarray(value)
            signature.append((scalar.dtype.str, scalar.tobytes()))
        else:
            return advance(inputs, state, parameters)
    _validate_profile_shapes(inputs, state, parameters)
    key = (structure, tuple(signature), jax.config.x64_enabled)
    entry = _early_profile_entry
    if entry is not None and entry[0] == key and all(
        not value.is_deleted() for value in entry[2]
    ):
        return entry[2]
    result = advance(inputs, state, parameters)
    _early_profile_entry = (key, sources, result)
    return result


def apply_early_ecosystem_vertical_profile(
    state: EcosystemDynamicsAnnualUpdateState,
    native: SoilBiogeochemVerticalProfileNativeData,
    active_layer_index_column: jnp.ndarray,
    *,
    jit: bool = False,
) -> EcosystemDynamicsAnnualUpdateState:

    prefix = ecosystem_pre_drainage_prefix_state(state)
    inputs = assemble_soil_biogeochem_vertical_profile_inputs(
        native,
        active_layer_index_column=active_layer_index_column,
    )
    profile = _advance_early_vertical_profile(
        inputs,
        prefix.soil_biogeochem_vertical_profile,
        native.parameters,
        jit=jit,
    )
    return _replace_ecosystem_pre_drainage_prefix_state(
        state,
        prefix._replace(soil_biogeochem_vertical_profile=profile),
    )


def synchronize_crop_growing_degree_days(
    state: EcosystemDynamicsAnnualUpdateState,
    accumulators: ConfiguredAccumulatorState,
) -> EcosystemDynamicsAnnualUpdateState:

    prefix = ecosystem_pre_drainage_prefix_state(state)
    crop = prefix.crop_phenology
    growing_degree_days = jnp.asarray(
        accumulators.crop.growing_degree_days_plant.value,
        dtype=jnp.float64,
    )
    if growing_degree_days.shape != crop.growing_degree_days_since_planting_patch.shape:
        raise ValueError("crop GDD accumulator and ecosystem patch axes disagree")
    return _replace_ecosystem_pre_drainage_prefix_state(
        state,
        prefix._replace(
            crop_phenology=crop._replace(
                growing_degree_days_since_planting_patch=growing_degree_days,
            ),
        ),
    )


def _grid_domain_index(topology: SubgridTopology) -> jnp.ndarray:
    longitude_index = jnp.asarray(topology.grid.longitude_index, dtype=jnp.int32)
    latitude_index = jnp.asarray(topology.grid.latitude_index, dtype=jnp.int32)
    longitude_count = jnp.max(longitude_index) + jnp.int32(1)
    return latitude_index * longitude_count + longitude_index + jnp.int32(1)


def _replace_compact(
    base: jnp.ndarray,
    index: jnp.ndarray,
    compact: jnp.ndarray,
    *,
    label: str,
) -> jnp.ndarray:
    result = jnp.asarray(base, dtype=jnp.float64)
    index = jnp.asarray(index, dtype=jnp.int32)
    compact = jnp.asarray(compact, dtype=jnp.float64)
    if index.shape != compact.shape:
        raise ValueError(f"{label} patch map and compact field disagree")
    return result.at[index.reshape(-1)].set(compact.reshape(-1))


def _mask_urban_unused_water_levels(
    water_volume_fraction: jnp.ndarray,
    column_type: jnp.ndarray,
) -> jnp.ndarray:

    water = jnp.asarray(water_volume_fraction, dtype=jnp.float64)
    types = jnp.asarray(column_type, dtype=jnp.int32)
    if water.ndim != 2 or types.shape != (water.shape[0],):
        raise ValueError("urban water field and column axis disagree")
    enclosure = (types >= jnp.int32(71)) & (types <= jnp.int32(73))
    unused = jnp.arange(water.shape[1], dtype=jnp.int32) >= jnp.int32(
        NUMBER_OF_URBAN_LEVELS
    )
    return jnp.where(
        enclosure[:, None] & unused[None, :],
        jnp.float64(CLM_SPECIAL_VALUE),
        water,
    )


def reference_temperature_2m_patch(
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    native_map: PostTemperatureNativeMap,
) -> jnp.ndarray:

    result = physical.canopy_flux.state.active.postprocess.reference_temperature_2m_patch
    result = _replace_compact(
        result,
        native_map.urban_surface_patch_index,
        physical.urban_flux.surface.atmosphere.reference_temperature_surface,
        label="urban reference temperature",
    )
    return _replace_compact(
        result,
        native_map.lake_patch_index,
        physical.lake_flux.fluxes.temperature.reference_temperature,
        label="lake reference temperature",
    )


def assemble_ecosystem_fast_boundary(
    topology: SubgridTopology,
    forcing: DatmForcing,
    nitrogen_deposition_gridcell: jnp.ndarray,
    physical: PhysicalDriverResult | PhysicalDriverRuntimeResult,
    post_temperature_map: PostTemperatureNativeMap,
    hydrology: HydrologyDriverResult,
    lake: LakeHydrologyDriverResult,
    accumulators: ConfiguredAccumulatorState,
    active_layer: ActiveLayerResult,
    methane: DecompositionRatePersistentState,
    state_summaries: CarbonNitrogenStateSummaries,
    ecosystem_state: EcosystemDynamicsAnnualUpdateState,
    fertilizer_nitrogen_patch: jnp.ndarray,
    carbon_root_fraction_patch_level: jnp.ndarray,
    carbon_root_profile_beta_patch: jnp.ndarray,
) -> EcosystemFastBoundary:

    grid_domain = _grid_domain_index(topology)
    column_gridcell = jnp.asarray(topology.column.gridcell, dtype=jnp.int32)
    patch_gridcell = jnp.asarray(topology.patch.gridcell, dtype=jnp.int32)
    patch_column = jnp.asarray(topology.patch.column, dtype=jnp.int32)
    column_landunit = jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
    column_type = jnp.asarray(topology.column.column_type, dtype=jnp.int32)
    column_active = jnp.asarray(topology.column.active, dtype=jnp.bool_)
    patch_landunit = jnp.asarray(topology.patch.landunit_type, dtype=jnp.int32)
    patch_active = jnp.asarray(topology.patch.active, dtype=jnp.bool_)
    vegetation_type = jnp.asarray(topology.patch.vegetation_type, dtype=jnp.int32)
    column_domain = grid_domain[column_gridcell]
    patch_domain = grid_domain[patch_gridcell]
    prefix = ecosystem_pre_drainage_prefix_state(ecosystem_state)
    canopy = physical.canopy_flux.ecosystem
    atmosphere = accumulators.atmosphere
    temperature = accumulators.temperature
    crop = accumulators.crop
    daily = accumulators.daily_extrema
    water = lake.hydrology.state.water
    diagnostics = hydrology.hydrology.state.final_diagnostics
    vegetation_masks = exposed_vegetation_masks(
        patch_landunit,
        patch_active,
        canopy.vegetation_exposed_patch,
    )
    soil_column = column_active & (
        (column_landunit == jnp.int32(LANDUNIT_SOIL))
        | (column_landunit == jnp.int32(LANDUNIT_CROP))
    )
    hydrology_column = soil_column | (
        column_active & (column_type == jnp.int32(75))
    )
    prognostic_crop_patch = patch_active & (
        vegetation_type >= jnp.int32(FIRST_PROGNOSTIC_CROP_PFT)
    )
    generic_crop_patch = (
        patch_active
        & (patch_landunit == jnp.int32(LANDUNIT_CROP))
        & (vegetation_type >= jnp.int32(15))
        & (vegetation_type < jnp.int32(FIRST_PROGNOSTIC_CROP_PFT))
    )
    urban_patch = patch_active & (
        (patch_landunit >= jnp.int32(LANDUNIT_URBAN_MIN))
        & (patch_landunit <= jnp.int32(LANDUNIT_URBAN_MAX))
    )
    special = jnp.float64(CLM_SPECIAL_VALUE)
    transpiration_wetness = jnp.where(
        vegetation_masks.exposed,
        canopy.transpiration_wetness_patch,
        jnp.where(
            vegetation_masks.not_exposed,
            jnp.float64(0.0),
            jnp.float64(CLM_SPECIAL_VALUE),
        ),
    )

    column_count = topology.column.gridcell.shape[0]
    patch_count = topology.patch.gridcell.shape[0]
    root_fraction = jnp.asarray(carbon_root_fraction_patch_level, dtype=jnp.float64)
    if root_fraction.shape != (patch_count, 25):
        raise ValueError("carbon root fraction must have shape (patch, 25)")
    root_beta = jnp.asarray(carbon_root_profile_beta_patch, dtype=jnp.float64)
    if root_beta.shape != (patch_count,):
        raise ValueError("carbon root profile beta must contain every patch")
    if water.soil_snow_liquid_column_level.shape[0] != column_count:
        raise ValueError("lake hydrology result must retain the full column axis")

    enclosure_patch = (
        (topology.patch.column_type >= jnp.int32(71))
        & (topology.patch.column_type <= jnp.int32(73))
    )
    ground_interface = physical.soil_temperature.inputs.thermal_properties.interface_depth[
        :, NUMBER_OF_SNOW_LEVELS:
    ]
    enclosure_root_fraction = jackson_root_fraction(
        root_beta,
        ground_interface[patch_column, : NUMBER_OF_SOIL_LEVELS + 1],
        physical.soil_temperature.inputs.thermal_properties.bedrock_layer_count[
            patch_column
        ],
    )
    enclosure_root_fraction = jnp.concatenate(
        (
            enclosure_root_fraction,
            jnp.zeros(
                (patch_count, root_fraction.shape[1] - NUMBER_OF_SOIL_LEVELS),
                dtype=jnp.float64,
            ),
        ),
        axis=1,
    )
    root_fraction = jnp.where(
        enclosure_patch[:, None],
        enclosure_root_fraction,
        root_fraction,
    )

    return EcosystemFastBoundary(
        gridcell=EcosystemFastGridcell(
            domain_index=grid_domain,
            relative_humidity=forcing.relative_humidity,
            wind_speed=forcing.wind_speed,
            nitrogen_deposition=jnp.asarray(
                nitrogen_deposition_gridcell,
                dtype=jnp.float64,
            ),
        ),
        column=EcosystemFastColumn(
            domain_gridcell=column_domain,
            column_type=topology.column.column_type,
            landunit_type=topology.column.landunit_type,
            active=topology.column.active,
            weight_on_gridcell=topology.column.weight_on_gridcell,
            downscaled_temperature=physical.atmospheric.column_forcing.temperature,
            downscaled_rain=physical.atmospheric.column_forcing.rain,
            downscaled_snow=physical.atmospheric.column_forcing.snow,
            annual_evapotranspiration=(
                accumulators.water_energy.annual_evapotranspiration.value
            ),
            topsoil_water_fraction=jnp.where(
                hydrology_column,
                diagnostics.top_17cm_water_availability_column,
                special,
            ),
            snow_depth=water.snow_depth_column,
            saturated_area_fraction=jnp.where(
                hydrology_column,
                hydrology.hydrology.state.surface_hydrology
                .saturated_fraction_column,
                special,
            ),
            inundated_fraction=methane.inundated_fraction_column,
            soil_temperature_17cm=diagnostics.top_17cm_soil_temperature_column,
            maximum_thaw_depth=jnp.where(
                soil_column,
                active_layer.state.maximum_thaw_depth_column,
                special,
            ),
            previous_maximum_thaw_depth=jnp.where(
                soil_column,
                active_layer.state.prior_year_maximum_thaw_depth_column,
                special,
            ),
            active_layer_index=(
                active_layer.state.prior_year_maximum_thaw_depth_index_column
            ),
            soil_water_potential=jnp.where(
                hydrology_column[:, None],
                diagnostics.soil_water_potential_column_level,
                special,
            ),
            soil_liquid_water=(
                water.soil_snow_liquid_column_level[:, NUMBER_OF_SNOW_LEVELS:]
            ),
            soil_volumetric_water=_mask_urban_unused_water_levels(
                water.soil_water_volume_fraction_column_level,
                column_type,
            ),
            soil_temperature=(
                lake.hydrology.state.temperature
                .soil_snow_temperature_column_level[:, NUMBER_OF_SNOW_LEVELS:]
            ),
            oxygen_stress_saturated=jnp.where(
                soil_column[:, None],
                methane.oxygen_stress_saturated_column_level,
                special,
            ),
            oxygen_stress_unsaturated=jnp.where(
                soil_column[:, None],
                methane.oxygen_stress_unsaturated_column_level,
                special,
            ),
            total_litter_carbon=state_summaries.total_litter_carbon_column,
            total_soil_organic_carbon=state_summaries.total_soil_carbon_column,
            total_vegetation_carbon=state_summaries.total_vegetation_carbon_column,
        ),
        patch=EcosystemFastPatch(
            domain_gridcell=patch_domain,
            vegetation_type=topology.patch.vegetation_type,
            column_domain_gridcell=column_domain[patch_column],
            column_type=topology.patch.column_type,
            landunit_type=topology.patch.landunit_type,
            active=topology.patch.active,
            weight_on_gridcell=topology.patch.weight_on_gridcell,
            weight_on_column=topology.patch.weight_on_column,
            vegetation_exposed=canopy.vegetation_exposed_patch,
            sunlit_leaf_area=jnp.where(
                urban_patch,
                jnp.float64(0.0),
                canopy.sunlit_leaf_area_patch,
            ),
            shaded_leaf_area=jnp.where(
                urban_patch,
                jnp.float64(0.0),
                canopy.shaded_leaf_area_patch,
            ),
            total_leaf_area=canopy.total_leaf_area_patch,
            sunlit_leaf_respiration=jnp.where(
                vegetation_masks.exposed,
                canopy.sunlit_leaf_respiration_patch,
                jnp.float64(jnp.nan),
            ),
            shaded_leaf_respiration=jnp.where(
                vegetation_masks.exposed,
                canopy.shaded_leaf_respiration_patch,
                jnp.float64(jnp.nan),
            ),
            sunlit_photosynthesis=canopy.sunlit_photosynthesis_patch,
            shaded_photosynthesis=canopy.shaded_photosynthesis_patch,
            transpiration_wetness=transpiration_wetness,
            root_zone_wetness=canopy.root_zone_wetness_patch,
            precipitation_60day=atmosphere.precipitation_60_day.value,
            precipitation_10day=atmosphere.precipitation_10_day.value,
            relative_humidity_30day=atmosphere.relative_humidity_30_day.value,
            reference_temperature_2m=reference_temperature_2m_patch(
                physical,
                post_temperature_map,
            ),
            minimum_temperature_2m=daily.reference_temperature_minimum_patch,
            maximum_temperature_2m=daily.reference_temperature_maximum_patch,
            temperature_10day=temperature.reference_temperature_10_day.value,
            minimum_temperature_5day=(
                temperature.minimum_reference_temperature_5_day.value
            ),
            minimum_temperature_10day=(
                temperature.minimum_reference_temperature_10_day.value
            ),
            growing_degree_days_base_0=(
                temperature.growing_degree_days_base_0.value
            ),
            growing_degree_days_base_8=(
                temperature.growing_degree_days_base_8.value
            ),
            growing_degree_days_base_10=(
                temperature.growing_degree_days_base_10.value
            ),
            growing_degree_days_base_0_twenty_year=jnp.where(
                prognostic_crop_patch,
                prefix.phenology_climate.growing_degree_days_base_0_twenty_year_patch,
                special,
            ),
            growing_degree_days_base_8_twenty_year=jnp.where(
                prognostic_crop_patch,
                prefix.phenology_climate.growing_degree_days_base_8_twenty_year_patch,
                special,
            ),
            growing_degree_days_base_10_twenty_year=jnp.where(
                prognostic_crop_patch,
                prefix.phenology_climate.growing_degree_days_base_10_twenty_year_patch,
                special,
            ),
            years_crop_active=prefix.crop_year.years_active_patch,
            crop_live=prefix.crop_phenology.crop_live_patch,
            crop_planted=prefix.crop_phenology.crop_planted_patch,
            harvest_date=jnp.where(
                prognostic_crop_patch,
                prefix.crop_phenology.harvest_date_patch,
                jnp.iinfo(jnp.int32).max,
            ),
            fertilizer_nitrogen=jnp.where(
                generic_crop_patch,
                jnp.float64(0.0),
                jnp.asarray(fertilizer_nitrogen_patch, dtype=jnp.float64),
            ),
            growing_degree_days_since_planting=(
                prefix.crop_phenology.growing_degree_days_since_planting_patch
            ),
            growing_degree_days_topsoil=crop.growing_degree_days_soil.value,
            carbon_root_fraction=root_fraction,
        ),
    )


__all__ = [
    "EcosystemFastBoundary",
    "EcosystemFastColumn",
    "EcosystemFastGridcell",
    "EcosystemFastPatch",
    "apply_early_ecosystem_vertical_profile",
    "assemble_ecosystem_fast_boundary",
    "ecosystem_pre_drainage_prefix_state",
    "reference_temperature_2m_patch",
    "synchronize_crop_growing_degree_days",
]
