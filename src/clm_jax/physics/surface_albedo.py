
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import default_backend

from clm_jax.constants import FREEZING_TEMPERATURE_FRESHWATER
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_GLACIER,
    LANDUNIT_LAKE,
    LANDUNIT_SOIL,
)
from clm_jax.physics.numerics import (
    fortran_divide,
    fortran_materialize,
    strict_negative_exp,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.snow_snicar import (
    NUMBER_OF_SNOW_AEROSOLS,
    NUMBER_OF_SNOW_LAYERS,
    SnicarOpticalInputs,
    SnicarRadiativeTransferDiagnostics,
    SnicarRadiativeTransferInputs,
    SnicarRadiativeTransferResult,
    SnicarRuntimeDiagnostics,
    SnicarRuntimeResult,
    SnowOpticalTables,
    snicar_radiative_transfer,
)

NUMBER_OF_SHORTWAVE_BANDS = 2
NUMBER_OF_SNOW_ABSORPTION_LEVELS = 13


class SoilAlbedoInputs(NamedTuple):

    nonurban_column: jnp.ndarray
    landunit_type_column: jnp.ndarray
    cosine_solar_zenith_column: jnp.ndarray
    snow_layer_count_column: jnp.ndarray
    ground_temperature_column: jnp.ndarray
    top_soil_water_volume_fraction_column: jnp.ndarray
    lake_ice_fraction_top_two_column: jnp.ndarray
    soil_color_column: jnp.ndarray


class SoilAlbedoState(NamedTuple):

    direct_column_band: jnp.ndarray
    diffuse_column_band: jnp.ndarray


class SoilAlbedoParameters(NamedTuple):
    saturated_soil_albedo_color_band: jnp.ndarray
    dry_soil_albedo_color_band: jnp.ndarray
    land_ice_albedo_band: jnp.ndarray
    frozen_lake_albedo_band: jnp.ndarray
    melting_lake_albedo_band: jnp.ndarray
    lake_puddling: bool | jnp.ndarray = False
    freezing_temperature: float | jnp.ndarray = FREEZING_TEMPERATURE_FRESHWATER
    lake_ice_fraction_coefficient: float | jnp.ndarray = 95.6


class SoilAlbedoDiagnostics(NamedTuple):
    updated_column_count: jnp.ndarray
    soil_or_crop_column_count: jnp.ndarray
    land_ice_column_count: jnp.ndarray
    open_water_column_count: jnp.ndarray
    melting_lake_ice_column_count: jnp.ndarray
    cold_frozen_surface_column_count: jnp.ndarray


class SoilAlbedoResult(NamedTuple):
    state: SoilAlbedoState
    diagnostics: SoilAlbedoDiagnostics


class CanopyTwoStreamInputs(NamedTuple):

    active_patch: jnp.ndarray
    patch_column: jnp.ndarray
    leaf_orientation_patch: jnp.ndarray
    cosine_solar_zenith_patch: jnp.ndarray
    reflectance_patch_band: jnp.ndarray
    transmittance_patch_band: jnp.ndarray
    vegetation_temperature_patch: jnp.ndarray
    wet_fraction_patch: jnp.ndarray
    snow_fraction_patch: jnp.ndarray
    exposed_leaf_area_patch: jnp.ndarray
    exposed_stem_area_patch: jnp.ndarray
    ground_direct_albedo_column_band: jnp.ndarray
    ground_diffuse_albedo_column_band: jnp.ndarray
    snow_free_soil_direct_albedo_column_band: jnp.ndarray
    snow_free_soil_diffuse_albedo_column_band: jnp.ndarray


class CanopyTwoStreamState(NamedTuple):

    direct_surface_albedo_patch_band: jnp.ndarray
    diffuse_surface_albedo_patch_band: jnp.ndarray
    snow_free_direct_surface_albedo_patch_band: jnp.ndarray
    snow_free_diffuse_surface_albedo_patch_band: jnp.ndarray
    direct_absorption_patch_band: jnp.ndarray
    direct_sunlit_absorption_patch_band: jnp.ndarray
    direct_shaded_absorption_patch_band: jnp.ndarray
    diffuse_absorption_patch_band: jnp.ndarray
    diffuse_sunlit_absorption_patch_band: jnp.ndarray
    diffuse_shaded_absorption_patch_band: jnp.ndarray
    direct_beam_transmission_patch_band: jnp.ndarray
    direct_to_diffuse_transmission_patch_band: jnp.ndarray
    diffuse_transmission_patch_band: jnp.ndarray
    sunlit_fraction_patch_layer: jnp.ndarray
    direct_sunlit_absorption_patch_layer: jnp.ndarray
    direct_shaded_absorption_patch_layer: jnp.ndarray
    diffuse_sunlit_absorption_patch_layer: jnp.ndarray
    diffuse_shaded_absorption_patch_layer: jnp.ndarray
    sunlit_vcmax_scale_patch: jnp.ndarray
    shaded_vcmax_scale_patch: jnp.ndarray


class CanopyTwoStreamParameters(NamedTuple):

    snow_scattering_albedo_band: jnp.ndarray
    snow_direct_upscatter: float | jnp.ndarray = 0.5
    snow_diffuse_upscatter: float | jnp.ndarray = 0.5
    nitrogen_extinction_coefficient: float | jnp.ndarray = 0.30
    snow_free_only: bool | jnp.ndarray = False


class CanopyTwoStreamDiagnostics(NamedTuple):
    updated_patch_count: jnp.ndarray
    snow_adjusted_patch_count: jnp.ndarray
    snow_free_only: jnp.ndarray


class CanopyTwoStreamResult(NamedTuple):
    state: CanopyTwoStreamState
    diagnostics: CanopyTwoStreamDiagnostics


class CanopySurfaceAlbedoInputs(NamedTuple):

    nonurban_patch: jnp.ndarray
    patch_column: jnp.ndarray
    landunit_type_patch: jnp.ndarray
    pft_type_patch: jnp.ndarray
    cosine_solar_zenith_patch: jnp.ndarray
    leaf_orientation_pft: jnp.ndarray
    leaf_reflectance_pft_band: jnp.ndarray
    stem_reflectance_pft_band: jnp.ndarray
    leaf_transmittance_pft_band: jnp.ndarray
    stem_transmittance_pft_band: jnp.ndarray
    vegetation_temperature_patch: jnp.ndarray
    wet_fraction_patch: jnp.ndarray
    snow_fraction_patch: jnp.ndarray
    exposed_leaf_area_patch: jnp.ndarray
    exposed_stem_area_patch: jnp.ndarray
    ground_direct_albedo_column_band: jnp.ndarray
    ground_diffuse_albedo_column_band: jnp.ndarray
    snow_free_soil_direct_albedo_column_band: jnp.ndarray
    snow_free_soil_diffuse_albedo_column_band: jnp.ndarray


class CanopySurfaceAlbedoState(NamedTuple):

    two_stream: CanopyTwoStreamState
    leaf_area_patch_layer: jnp.ndarray
    stem_area_patch_layer: jnp.ndarray
    radiative_layer_count_patch: jnp.ndarray
    canopy_layer_count_patch: jnp.ndarray


class CanopySurfaceAlbedoParameters(NamedTuple):

    snow_scattering_albedo_band: jnp.ndarray
    snow_direct_upscatter: float | jnp.ndarray = 0.5
    snow_diffuse_upscatter: float | jnp.ndarray = 0.5
    nitrogen_extinction_coefficient: float | jnp.ndarray = 0.30
    snow_free_diagnostic: bool | jnp.ndarray = False


class CanopySurfaceAlbedoDiagnostics(NamedTuple):
    nonurban_patch_count: jnp.ndarray
    vegetated_daylight_patch_count: jnp.ndarray
    nonvegetated_daylight_patch_count: jnp.ndarray
    night_patch_count: jnp.ndarray
    snow_adjusted_patch_count: jnp.ndarray
    snow_free_diagnostic: jnp.ndarray


class CanopySurfaceAlbedoResult(NamedTuple):
    state: CanopySurfaceAlbedoState
    diagnostics: CanopySurfaceAlbedoDiagnostics


class GroundSnowAlbedoInputs(NamedTuple):

    nonurban_column: jnp.ndarray
    landunit_type_column: jnp.ndarray
    cosine_solar_zenith_column: jnp.ndarray
    snow_fraction_column: jnp.ndarray
    snow_water_equivalent_column: jnp.ndarray
    soil_direct_albedo_column_band: jnp.ndarray
    soil_diffuse_albedo_column_band: jnp.ndarray
    snow_direct_albedo_column_band: jnp.ndarray
    snow_diffuse_albedo_column_band: jnp.ndarray
    snow_direct_absorption_column_level_band: jnp.ndarray
    snow_diffuse_absorption_column_level_band: jnp.ndarray


class GroundSnowAlbedoState(NamedTuple):
    ground_direct_albedo_column_band: jnp.ndarray
    ground_diffuse_albedo_column_band: jnp.ndarray
    snow_direct_albedo_history_column_band: jnp.ndarray
    snow_diffuse_albedo_history_column_band: jnp.ndarray
    direct_visible_absorption_column_level: jnp.ndarray
    direct_nir_absorption_column_level: jnp.ndarray
    diffuse_visible_absorption_column_level: jnp.ndarray
    diffuse_nir_absorption_column_level: jnp.ndarray


class GroundSnowAlbedoDiagnostics(NamedTuple):
    daylight_column_count: jnp.ndarray
    energy_weighted_column_count: jnp.ndarray
    snow_history_column_count: jnp.ndarray


class GroundSnowAlbedoResult(NamedTuple):
    state: GroundSnowAlbedoState
    diagnostics: GroundSnowAlbedoDiagnostics


class SurfaceAlbedoInputs(NamedTuple):

    nonurban_column: jnp.ndarray
    nonurban_patch: jnp.ndarray
    column_gridcell: jnp.ndarray
    patch_gridcell: jnp.ndarray
    patch_column: jnp.ndarray
    cosine_solar_zenith_gridcell: jnp.ndarray
    landunit_type_column: jnp.ndarray
    pft_type_patch: jnp.ndarray
    snow_layer_count_column: jnp.ndarray
    ground_temperature_column: jnp.ndarray
    top_soil_water_volume_fraction_column: jnp.ndarray
    lake_ice_fraction_top_two_column: jnp.ndarray
    soil_color_column: jnp.ndarray
    snow_fraction_column: jnp.ndarray
    snow_water_equivalent_column: jnp.ndarray
    snow_liquid_column_layer: jnp.ndarray
    snow_ice_column_layer: jnp.ndarray
    snow_grain_radius_column_layer: jnp.ndarray
    aerosol_mass_concentration_column_layer_species: jnp.ndarray
    leaf_orientation_pft: jnp.ndarray
    leaf_reflectance_pft_band: jnp.ndarray
    stem_reflectance_pft_band: jnp.ndarray
    leaf_transmittance_pft_band: jnp.ndarray
    stem_transmittance_pft_band: jnp.ndarray
    vegetation_temperature_patch: jnp.ndarray
    wet_fraction_patch: jnp.ndarray
    canopy_snow_fraction_patch: jnp.ndarray
    exposed_leaf_area_patch: jnp.ndarray
    exposed_stem_area_patch: jnp.ndarray


class SurfaceAlbedoState(NamedTuple):

    cosine_solar_zenith_column: jnp.ndarray
    soil: SoilAlbedoState
    ground: GroundSnowAlbedoState
    canopy: CanopySurfaceAlbedoState
    pure_snow_ground_direct_albedo_column_band: jnp.ndarray
    pure_snow_ground_diffuse_albedo_column_band: jnp.ndarray
    no_black_carbon_ground_direct_albedo_column_band: jnp.ndarray
    no_black_carbon_ground_diffuse_albedo_column_band: jnp.ndarray
    no_organic_carbon_ground_direct_albedo_column_band: jnp.ndarray
    no_organic_carbon_ground_diffuse_albedo_column_band: jnp.ndarray
    no_dust_ground_direct_albedo_column_band: jnp.ndarray
    no_dust_ground_diffuse_albedo_column_band: jnp.ndarray


class SurfaceAlbedoParameters(NamedTuple):

    soil: SoilAlbedoParameters
    snow_optical_tables: SnowOpticalTables
    canopy: CanopySurfaceAlbedoParameters
    subgrid_fluxes_enabled: bool | jnp.ndarray = True


class SurfaceAlbedoDiagnostics(NamedTuple):
    soil: SoilAlbedoDiagnostics
    direct_snow: SnicarRadiativeTransferDiagnostics | SnicarRuntimeDiagnostics
    diffuse_snow: SnicarRadiativeTransferDiagnostics | SnicarRuntimeDiagnostics
    ground: GroundSnowAlbedoDiagnostics
    canopy: CanopySurfaceAlbedoDiagnostics


class SurfaceAlbedoResult(NamedTuple):
    state: SurfaceAlbedoState
    direct_snow: SnicarRadiativeTransferResult | SnicarRuntimeResult
    diffuse_snow: SnicarRadiativeTransferResult | SnicarRuntimeResult
    diagnostics: SurfaceAlbedoDiagnostics


class SurfaceAlbedoColumnInputs(NamedTuple):

    nonurban_column: jnp.ndarray
    cosine_solar_zenith_column: jnp.ndarray
    landunit_type_column: jnp.ndarray
    snow_layer_count_column: jnp.ndarray
    ground_temperature_column: jnp.ndarray
    top_soil_water_volume_fraction_column: jnp.ndarray
    lake_ice_fraction_top_two_column: jnp.ndarray
    soil_color_column: jnp.ndarray
    snow_fraction_column: jnp.ndarray
    snow_water_equivalent_column: jnp.ndarray
    snow_liquid_column_layer: jnp.ndarray
    snow_ice_column_layer: jnp.ndarray
    snow_grain_radius_column_layer: jnp.ndarray
    aerosol_mass_concentration_column_layer_species: jnp.ndarray


class SurfaceAlbedoColumnState(NamedTuple):

    cosine_solar_zenith_column: jnp.ndarray
    soil: SoilAlbedoState
    ground: GroundSnowAlbedoState
    pure_snow_ground_direct_albedo_column_band: jnp.ndarray
    pure_snow_ground_diffuse_albedo_column_band: jnp.ndarray
    no_black_carbon_ground_direct_albedo_column_band: jnp.ndarray
    no_black_carbon_ground_diffuse_albedo_column_band: jnp.ndarray
    no_organic_carbon_ground_direct_albedo_column_band: jnp.ndarray
    no_organic_carbon_ground_diffuse_albedo_column_band: jnp.ndarray
    no_dust_ground_direct_albedo_column_band: jnp.ndarray
    no_dust_ground_diffuse_albedo_column_band: jnp.ndarray


class SurfaceAlbedoColumnParameters(NamedTuple):
    soil: SoilAlbedoParameters
    snow_optical_tables: SnowOpticalTables
    subgrid_fluxes_enabled: bool | jnp.ndarray = True


class SurfaceAlbedoColumnDiagnostics(NamedTuple):
    soil: SoilAlbedoDiagnostics
    direct_snow: SnicarRadiativeTransferDiagnostics | SnicarRuntimeDiagnostics
    diffuse_snow: SnicarRadiativeTransferDiagnostics | SnicarRuntimeDiagnostics
    ground: GroundSnowAlbedoDiagnostics


class SurfaceAlbedoColumnResult(NamedTuple):
    state: SurfaceAlbedoColumnState
    direct_snow: SnicarRadiativeTransferResult | SnicarRuntimeResult
    diffuse_snow: SnicarRadiativeTransferResult | SnicarRuntimeResult
    diagnostics: SurfaceAlbedoColumnDiagnostics


def _validate_soil_albedo_shapes(
    inputs: SoilAlbedoInputs,
    state: SoilAlbedoState,
    parameters: SoilAlbedoParameters,
) -> int:
    if jnp.ndim(inputs.nonurban_column) != 1:
        raise ValueError("nonurban_column must be a vector")
    column_count = inputs.nonurban_column.shape[0]
    for name in (
        "landunit_type_column",
        "cosine_solar_zenith_column",
        "snow_layer_count_column",
        "ground_temperature_column",
        "top_soil_water_volume_fraction_column",
        "soil_color_column",
    ):
        if jnp.shape(getattr(inputs, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    if jnp.shape(inputs.lake_ice_fraction_top_two_column) != (column_count, 2):
        raise ValueError(f"lake_ice_fraction_top_two_column must have shape ({column_count}, 2)")
    output_shape = (column_count, NUMBER_OF_SHORTWAVE_BANDS)
    for name in SoilAlbedoState._fields:
        if jnp.shape(getattr(state, name)) != output_shape:
            raise ValueError(f"{name} must have shape {output_shape}")
    saturated_shape = jnp.shape(parameters.saturated_soil_albedo_color_band)
    if len(saturated_shape) != 2 or saturated_shape[1] != NUMBER_OF_SHORTWAVE_BANDS:
        raise ValueError("saturated soil albedo must have shape (color, 2)")
    if jnp.shape(parameters.dry_soil_albedo_color_band) != saturated_shape:
        raise ValueError("dry and saturated soil albedo tables must have equal shapes")
    for name in (
        "land_ice_albedo_band",
        "frozen_lake_albedo_band",
        "melting_lake_albedo_band",
    ):
        if jnp.shape(getattr(parameters, name)) != (NUMBER_OF_SHORTWAVE_BANDS,):
            raise ValueError(f"{name} must contain visible and near-infrared values")
    return column_count


def soil_albedo(
    inputs: SoilAlbedoInputs,
    state: SoilAlbedoState,
    parameters: SoilAlbedoParameters,
) -> SoilAlbedoResult:

    _validate_soil_albedo_shapes(inputs, state, parameters)
    active = jnp.asarray(inputs.nonurban_column, dtype=jnp.bool_) & (
        jnp.asarray(inputs.cosine_solar_zenith_column, dtype=jnp.float64) > 0.0
    )
    landunit_type = jnp.asarray(inputs.landunit_type_column, dtype=jnp.int32)
    soil_or_crop = (landunit_type == jnp.int32(LANDUNIT_SOIL)) | (
        landunit_type == jnp.int32(LANDUNIT_CROP)
    )
    land_ice = landunit_type == jnp.int32(LANDUNIT_GLACIER)
    lake = landunit_type == jnp.int32(LANDUNIT_LAKE)
    temperature = jnp.asarray(inputs.ground_temperature_column, dtype=jnp.float64)
    freezing_temperature = jnp.asarray(parameters.freezing_temperature, dtype=jnp.float64)
    lake_ice_fraction = jnp.asarray(
        inputs.lake_ice_fraction_top_two_column,
        dtype=jnp.float64,
    )
    lake_puddling = jnp.asarray(parameters.lake_puddling, dtype=jnp.bool_)
    puddled_open_water = (
        lake_puddling
        & lake
        & (temperature == freezing_temperature)
        & (lake_ice_fraction[:, 0] < 1.0)
        & (lake_ice_fraction[:, 1] > 0.0)
    )
    open_water = (temperature > freezing_temperature) | puddled_open_water
    melting_lake_ice = (
        lake
        & ~lake_puddling
        & (jnp.asarray(inputs.snow_layer_count_column, dtype=jnp.int32) == 0)
        & ~open_water
    )

    saturated = jnp.asarray(
        parameters.saturated_soil_albedo_color_band,
        dtype=jnp.float64,
    )
    dry = jnp.asarray(parameters.dry_soil_albedo_color_band, dtype=jnp.float64)
    soil_color = jnp.clip(
        jnp.asarray(inputs.soil_color_column, dtype=jnp.int32) - jnp.int32(1),
        0,
        saturated.shape[0] - 1,
    )
    moisture_decrement = fortran_multiply(
        jnp.float64(0.40),
        jnp.asarray(inputs.top_soil_water_volume_fraction_column, dtype=jnp.float64),
    )
    soil_increment = jnp.maximum(
        fortran_add(jnp.float64(0.11), -moisture_decrement),
        jnp.float64(0.0),
    )
    soil_direct = jnp.minimum(
        fortran_add(saturated[soil_color], soil_increment[:, None]),
        dry[soil_color],
    )

    cosine = jnp.asarray(inputs.cosine_solar_zenith_column, dtype=jnp.float64)
    open_water_denominator = fortran_add(
        jnp.maximum(jnp.float64(0.001), cosine),
        jnp.float64(0.15),
    )
    open_water_direct_scalar = fortran_divide(
        jnp.float64(0.05),
        open_water_denominator,
    )
    open_water_direct = jnp.broadcast_to(
        open_water_direct_scalar[:, None],
        soil_direct.shape,
    )
    open_water_diffuse = jnp.where(
        lake[:, None],
        jnp.float64(0.10),
        open_water_direct,
    )

    frozen_temperature = jnp.where(
        melting_lake_ice,
        temperature,
        freezing_temperature,
    )
    temperature_deficit = fortran_add(freezing_temperature, -frozen_temperature)
    exponent = -fortran_divide(
        fortran_multiply(
            jnp.asarray(parameters.lake_ice_fraction_coefficient, dtype=jnp.float64),
            temperature_deficit,
        ),
        freezing_temperature,
    )
    ice_fraction = fortran_add(jnp.float64(1.0), -strict_negative_exp(exponent))
    liquid_fraction = fortran_add(jnp.float64(1.0), -ice_fraction)
    frozen_lake = jnp.asarray(parameters.frozen_lake_albedo_band, dtype=jnp.float64)
    melting_lake = jnp.asarray(parameters.melting_lake_albedo_band, dtype=jnp.float64)
    melting_direct_floor = jnp.maximum(melting_lake, open_water_direct)
    melting_diffuse_floor = jnp.maximum(melting_lake, jnp.float64(0.10))
    partially_frozen_direct = fortran_add(
        fortran_multiply(ice_fraction[:, None], frozen_lake),
        fortran_multiply(liquid_fraction[:, None], melting_direct_floor),
    )
    partially_frozen_diffuse = fortran_add(
        fortran_multiply(ice_fraction[:, None], frozen_lake),
        fortran_multiply(liquid_fraction[:, None], melting_diffuse_floor),
    )

    land_ice_albedo = jnp.asarray(parameters.land_ice_albedo_band, dtype=jnp.float64)
    frozen_direct = jnp.where(
        melting_lake_ice[:, None],
        partially_frozen_direct,
        frozen_lake,
    )
    frozen_diffuse = jnp.where(
        melting_lake_ice[:, None],
        partially_frozen_diffuse,
        frozen_lake,
    )
    candidate_direct = jnp.where(
        soil_or_crop[:, None],
        soil_direct,
        jnp.where(
            land_ice[:, None],
            land_ice_albedo,
            jnp.where(open_water[:, None], open_water_direct, frozen_direct),
        ),
    )
    candidate_diffuse = jnp.where(
        soil_or_crop[:, None],
        soil_direct,
        jnp.where(
            land_ice[:, None],
            land_ice_albedo,
            jnp.where(open_water[:, None], open_water_diffuse, frozen_diffuse),
        ),
    )
    update = active[:, None]
    final_state = SoilAlbedoState(
        direct_column_band=jnp.where(
            update,
            candidate_direct,
            jnp.asarray(state.direct_column_band, dtype=jnp.float64),
        ),
        diffuse_column_band=jnp.where(
            update,
            candidate_diffuse,
            jnp.asarray(state.diffuse_column_band, dtype=jnp.float64),
        ),
    )
    cold_frozen = ~(soil_or_crop | land_ice | open_water | melting_lake_ice)
    return SoilAlbedoResult(
        state=final_state,
        diagnostics=SoilAlbedoDiagnostics(
            updated_column_count=jnp.sum(active, dtype=jnp.int32),
            soil_or_crop_column_count=jnp.sum(active & soil_or_crop, dtype=jnp.int32),
            land_ice_column_count=jnp.sum(active & land_ice, dtype=jnp.int32),
            open_water_column_count=jnp.sum(
                active & ~soil_or_crop & ~land_ice & open_water,
                dtype=jnp.int32,
            ),
            melting_lake_ice_column_count=jnp.sum(
                active & melting_lake_ice,
                dtype=jnp.int32,
            ),
            cold_frozen_surface_column_count=jnp.sum(
                active & cold_frozen,
                dtype=jnp.int32,
            ),
        ),
    )


def _validate_ground_snow_albedo_shapes(
    inputs: GroundSnowAlbedoInputs,
    state: GroundSnowAlbedoState,
) -> int:
    if jnp.ndim(inputs.nonurban_column) != 1:
        raise ValueError("nonurban_column must be a vector")
    column_count = inputs.nonurban_column.shape[0]
    for name in (
        "landunit_type_column",
        "cosine_solar_zenith_column",
        "snow_fraction_column",
        "snow_water_equivalent_column",
    ):
        if jnp.shape(getattr(inputs, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    band_shape = (column_count, NUMBER_OF_SHORTWAVE_BANDS)
    for name in (
        "soil_direct_albedo_column_band",
        "soil_diffuse_albedo_column_band",
        "snow_direct_albedo_column_band",
        "snow_diffuse_albedo_column_band",
    ):
        if jnp.shape(getattr(inputs, name)) != band_shape:
            raise ValueError(f"{name} must have shape {band_shape}")
    layer_band_shape = (
        column_count,
        NUMBER_OF_SNOW_ABSORPTION_LEVELS,
        NUMBER_OF_SHORTWAVE_BANDS,
    )
    for name in (
        "snow_direct_absorption_column_level_band",
        "snow_diffuse_absorption_column_level_band",
    ):
        if jnp.shape(getattr(inputs, name)) != layer_band_shape:
            raise ValueError(f"{name} must have shape {layer_band_shape}")
    for name in GroundSnowAlbedoState._fields[:4]:
        if jnp.shape(getattr(state, name)) != band_shape:
            raise ValueError(f"{name} must have shape {band_shape}")
    layer_shape = (column_count, NUMBER_OF_SNOW_ABSORPTION_LEVELS)
    for name in GroundSnowAlbedoState._fields[4:]:
        if jnp.shape(getattr(state, name)) != layer_shape:
            raise ValueError(f"{name} must have shape {layer_shape}")
    return column_count


def weight_ground_snow_albedo(
    inputs: GroundSnowAlbedoInputs,
    state: GroundSnowAlbedoState,
    *,
    subgrid_fluxes_enabled: bool | jnp.ndarray = True,
) -> GroundSnowAlbedoResult:

    _validate_ground_snow_albedo_shapes(inputs, state)
    active = jnp.asarray(inputs.nonurban_column, dtype=jnp.bool_)
    daylight = active & (jnp.asarray(inputs.cosine_solar_zenith_column, dtype=jnp.float64) > 0.0)
    snow_fraction = jnp.asarray(inputs.snow_fraction_column, dtype=jnp.float64)
    snow_free_fraction = fortran_add(jnp.float64(1.0), -snow_fraction)
    soil_direct = jnp.asarray(
        inputs.soil_direct_albedo_column_band,
        dtype=jnp.float64,
    )
    soil_diffuse = jnp.asarray(
        inputs.soil_diffuse_albedo_column_band,
        dtype=jnp.float64,
    )
    snow_direct = jnp.asarray(
        inputs.snow_direct_albedo_column_band,
        dtype=jnp.float64,
    )
    snow_diffuse = jnp.asarray(
        inputs.snow_diffuse_albedo_column_band,
        dtype=jnp.float64,
    )
    ground_direct = fortran_add(
        fortran_multiply(soil_direct, snow_free_fraction[:, None]),
        fortran_multiply(snow_direct, snow_fraction[:, None]),
    )
    ground_diffuse = fortran_add(
        fortran_multiply(soil_diffuse, snow_free_fraction[:, None]),
        fortran_multiply(snow_diffuse, snow_fraction[:, None]),
    )

    snow_direct_absorption = jnp.asarray(
        inputs.snow_direct_absorption_column_level_band,
        dtype=jnp.float64,
    )
    snow_diffuse_absorption = jnp.asarray(
        inputs.snow_diffuse_absorption_column_level_band,
        dtype=jnp.float64,
    )
    direct_ratio = fortran_divide(
        snow_direct_absorption,
        fortran_add(jnp.float64(1.0), -snow_direct)[:, None, :],
    )
    diffuse_ratio = fortran_divide(
        snow_diffuse_absorption,
        fortran_add(jnp.float64(1.0), -snow_diffuse)[:, None, :],
    )
    direct_snow_term = fortran_multiply(
        snow_direct_absorption,
        snow_fraction[:, None, None],
    )
    diffuse_snow_term = fortran_multiply(
        snow_diffuse_absorption,
        snow_fraction[:, None, None],
    )
    direct_soil_term = fortran_multiply(
        fortran_multiply(
            snow_free_fraction[:, None, None],
            fortran_add(jnp.float64(1.0), -soil_direct)[:, None, :],
        ),
        direct_ratio,
    )
    diffuse_soil_term = fortran_multiply(
        fortran_multiply(
            snow_free_fraction[:, None, None],
            fortran_add(jnp.float64(1.0), -soil_diffuse)[:, None, :],
        ),
        diffuse_ratio,
    )
    weighted_direct_absorption = fortran_add(direct_snow_term, direct_soil_term)
    weighted_diffuse_absorption = fortran_add(diffuse_snow_term, diffuse_soil_term)
    lake = jnp.asarray(inputs.landunit_type_column, dtype=jnp.int32) == jnp.int32(LANDUNIT_LAKE)
    energy_weighted = (~jnp.asarray(subgrid_fluxes_enabled, dtype=jnp.bool_)) | lake
    direct_absorption = jnp.where(
        energy_weighted[:, None, None],
        weighted_direct_absorption,
        snow_direct_absorption,
    )
    diffuse_absorption = jnp.where(
        energy_weighted[:, None, None],
        weighted_diffuse_absorption,
        snow_diffuse_absorption,
    )

    update_band = daylight[:, None]
    update_layer = daylight[:, None]
    snow_history = daylight & (
        jnp.asarray(inputs.snow_water_equivalent_column, dtype=jnp.float64) > 0.0
    )
    history_mask = active[:, None]
    history_direct = jnp.where(snow_history[:, None], snow_direct, jnp.float64(0.0))
    history_diffuse = jnp.where(snow_history[:, None], snow_diffuse, jnp.float64(0.0))
    final_state = GroundSnowAlbedoState(
        ground_direct_albedo_column_band=jnp.where(
            update_band,
            ground_direct,
            jnp.asarray(state.ground_direct_albedo_column_band, dtype=jnp.float64),
        ),
        ground_diffuse_albedo_column_band=jnp.where(
            update_band,
            ground_diffuse,
            jnp.asarray(state.ground_diffuse_albedo_column_band, dtype=jnp.float64),
        ),
        snow_direct_albedo_history_column_band=jnp.where(
            history_mask,
            history_direct,
            jnp.asarray(
                state.snow_direct_albedo_history_column_band,
                dtype=jnp.float64,
            ),
        ),
        snow_diffuse_albedo_history_column_band=jnp.where(
            history_mask,
            history_diffuse,
            jnp.asarray(
                state.snow_diffuse_albedo_history_column_band,
                dtype=jnp.float64,
            ),
        ),
        direct_visible_absorption_column_level=jnp.where(
            update_layer,
            direct_absorption[:, :, 0],
            jnp.asarray(
                state.direct_visible_absorption_column_level,
                dtype=jnp.float64,
            ),
        ),
        direct_nir_absorption_column_level=jnp.where(
            update_layer,
            direct_absorption[:, :, 1],
            jnp.asarray(state.direct_nir_absorption_column_level, dtype=jnp.float64),
        ),
        diffuse_visible_absorption_column_level=jnp.where(
            update_layer,
            diffuse_absorption[:, :, 0],
            jnp.asarray(
                state.diffuse_visible_absorption_column_level,
                dtype=jnp.float64,
            ),
        ),
        diffuse_nir_absorption_column_level=jnp.where(
            update_layer,
            diffuse_absorption[:, :, 1],
            jnp.asarray(state.diffuse_nir_absorption_column_level, dtype=jnp.float64),
        ),
    )
    return GroundSnowAlbedoResult(
        state=final_state,
        diagnostics=GroundSnowAlbedoDiagnostics(
            daylight_column_count=jnp.sum(daylight, dtype=jnp.int32),
            energy_weighted_column_count=jnp.sum(
                daylight & energy_weighted,
                dtype=jnp.int32,
            ),
            snow_history_column_count=jnp.sum(snow_history, dtype=jnp.int32),
        ),
    )


def _validate_canopy_two_stream_shapes(
    inputs: CanopyTwoStreamInputs,
    state: CanopyTwoStreamState,
    parameters: CanopyTwoStreamParameters,
) -> int:
    if jnp.ndim(inputs.active_patch) != 1:
        raise ValueError("active_patch must be a vector")
    patch_count = inputs.active_patch.shape[0]
    for name in (
        "patch_column",
        "leaf_orientation_patch",
        "cosine_solar_zenith_patch",
        "vegetation_temperature_patch",
        "wet_fraction_patch",
        "snow_fraction_patch",
        "exposed_leaf_area_patch",
        "exposed_stem_area_patch",
    ):
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    patch_band_shape = (patch_count, NUMBER_OF_SHORTWAVE_BANDS)
    for name in ("reflectance_patch_band", "transmittance_patch_band"):
        if jnp.shape(getattr(inputs, name)) != patch_band_shape:
            raise ValueError(f"{name} must have shape {patch_band_shape}")
    column_band_shape = jnp.shape(inputs.ground_direct_albedo_column_band)
    if len(column_band_shape) != 2 or column_band_shape[1] != NUMBER_OF_SHORTWAVE_BANDS:
        raise ValueError("ground direct albedo must have shape (column, 2)")
    for name in (
        "ground_diffuse_albedo_column_band",
        "snow_free_soil_direct_albedo_column_band",
        "snow_free_soil_diffuse_albedo_column_band",
    ):
        if jnp.shape(getattr(inputs, name)) != column_band_shape:
            raise ValueError(f"{name} must use the common column grid")
    for name in CanopyTwoStreamState._fields[:13]:
        if jnp.shape(getattr(state, name)) != patch_band_shape:
            raise ValueError(f"{name} must have shape {patch_band_shape}")
    patch_layer_shape = (patch_count, 1)
    for name in CanopyTwoStreamState._fields[13:18]:
        if jnp.shape(getattr(state, name)) != patch_layer_shape:
            raise ValueError(
                f"{name} must have shape {patch_layer_shape}; the reference case uses nlevcan=1"
            )
    for name in CanopyTwoStreamState._fields[18:]:
        if jnp.shape(getattr(state, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    if jnp.shape(parameters.snow_scattering_albedo_band) != (NUMBER_OF_SHORTWAVE_BANDS,):
        raise ValueError("snow scattering albedo must contain visible and NIR values")
    return patch_count


def _canopy_log(value: jnp.ndarray) -> jnp.ndarray:
    if default_backend() == "gpu":
        from clm_jax.physics.native_logarithm import native_log

        return native_log(value)
    return jnp.log(value)


def canopy_two_stream_big_leaf(
    inputs: CanopyTwoStreamInputs,
    state: CanopyTwoStreamState,
    parameters: CanopyTwoStreamParameters,
) -> CanopyTwoStreamResult:

    _validate_canopy_two_stream_shapes(inputs, state, parameters)
    add = fortran_add
    multiply = fortran_multiply
    divide = fortran_divide
    one = jnp.float64(1.0)
    half = jnp.float64(0.5)

    active = jnp.asarray(inputs.active_patch, dtype=jnp.bool_)
    column = jnp.asarray(inputs.patch_column, dtype=jnp.int32)
    coszen = jnp.asarray(inputs.cosine_solar_zenith_patch, dtype=jnp.float64)
    cosz = jnp.maximum(jnp.float64(0.001), coszen)
    chil = jnp.clip(
        jnp.asarray(inputs.leaf_orientation_patch, dtype=jnp.float64),
        jnp.float64(-0.4),
        jnp.float64(0.6),
    )
    chil = jnp.where(jnp.abs(chil) <= jnp.float64(0.01), jnp.float64(0.01), chil)
    phi1 = add(
        add(jnp.float64(0.5), -multiply(jnp.float64(0.633), chil)),
        -multiply(multiply(jnp.float64(0.330), chil), chil),
    )
    phi2 = multiply(
        jnp.float64(0.877),
        add(one, -multiply(jnp.float64(2.0), phi1)),
    )
    gdir = add(phi1, multiply(phi2, cosz))
    twostext = divide(gdir, cosz)
    orientation_log = _canopy_log(divide(add(phi1, phi2), phi1))
    avmu = divide(
        add(one, -multiply(divide(phi1, phi2), orientation_log)),
        phi2,
    )
    temp0 = jnp.maximum(add(gdir, multiply(phi2, cosz)), jnp.float64(1.0e-6))
    temp1 = multiply(phi1, cosz)
    temp2 = add(
        one,
        -multiply(
            divide(temp1, temp0),
            _canopy_log(divide(add(temp1, temp0), temp1)),
        ),
    )

    reflectance = jnp.asarray(inputs.reflectance_patch_band, dtype=jnp.float64)
    transmittance = jnp.asarray(inputs.transmittance_patch_band, dtype=jnp.float64)
    omegal = add(reflectance, transmittance)
    asu = multiply(
        divide(
            multiply(multiply(half, omegal), gdir[:, None]),
            temp0[:, None],
        ),
        temp2[:, None],
    )
    avmu_extinction = multiply(avmu, twostext)
    betadl = multiply(
        divide(
            add(one, avmu_extinction)[:, None],
            multiply(
                multiply(omegal, avmu[:, None]),
                twostext[:, None],
            ),
        ),
        asu,
    )
    orientation_factor = divide(add(one, chil), jnp.float64(2.0))
    orientation_factor_squared = multiply(orientation_factor, orientation_factor)
    betail_numerator = add(
        add(reflectance, transmittance),
        multiply(add(reflectance, -transmittance), orientation_factor_squared[:, None]),
    )
    betail = divide(multiply(half, betail_numerator), omegal)

    snow_fraction = jnp.where(
        active, jnp.asarray(inputs.snow_fraction_patch, dtype=jnp.float64),
        jnp.float64(0.0))[:, None]
    leaf_fraction = add(one, -snow_fraction)
    snow_scattering = jnp.asarray(
        parameters.snow_scattering_albedo_band,
        dtype=jnp.float64,
    )[None, :]
    adjusted_omega = add(
        multiply(leaf_fraction, omegal),
        multiply(snow_fraction, snow_scattering),
    )
    adjusted_betad = divide(
        add(
            multiply(multiply(leaf_fraction, omegal), betadl),
            multiply(
                multiply(snow_fraction, snow_scattering),
                jnp.asarray(parameters.snow_direct_upscatter, dtype=jnp.float64),
            ),
        ),
        adjusted_omega,
    )
    adjusted_betai = divide(
        add(
            multiply(multiply(leaf_fraction, omegal), betail),
            multiply(
                multiply(snow_fraction, snow_scattering),
                jnp.asarray(parameters.snow_diffuse_upscatter, dtype=jnp.float64),
            ),
        ),
        adjusted_omega,
    )
    snow_free_only = jnp.asarray(parameters.snow_free_only, dtype=jnp.bool_)
    omega = jnp.where(snow_free_only, omegal, adjusted_omega)
    betad = jnp.where(snow_free_only, betadl, adjusted_betad)
    betai = jnp.where(snow_free_only, betail, adjusted_betai)

    b = add(add(one, -omega), multiply(omega, betai))
    c1 = multiply(omega, betai)
    common_optical_depth = avmu_extinction[:, None]
    d = multiply(multiply(common_optical_depth, omega), betad)
    f = multiply(
        multiply(common_optical_depth, omega),
        add(one, -betad),
    )
    discriminant = add(multiply(b, b), -multiply(c1, c1))
    h = divide(jnp.sqrt(discriminant), avmu[:, None])
    sigma = add(multiply(common_optical_depth, common_optical_depth), -discriminant)
    avmu_h = multiply(avmu[:, None], h)
    p1 = add(b, avmu_h)
    p2 = add(b, -avmu_h)
    p3 = add(b, common_optical_depth)
    p4 = add(b, -common_optical_depth)

    area = add(
        jnp.asarray(inputs.exposed_leaf_area_patch, dtype=jnp.float64),
        jnp.asarray(inputs.exposed_stem_area_patch, dtype=jnp.float64),
    )
    diffuse_attenuation = jnp.minimum(multiply(h, area[:, None]), jnp.float64(40.0))
    s1 = strict_negative_exp(-diffuse_attenuation)
    direct_attenuation = jnp.minimum(
        multiply(twostext, area),
        jnp.float64(40.0),
    )
    s2_patch = strict_negative_exp(-direct_attenuation)
    s2 = s2_patch[:, None]

    ground_direct = jnp.asarray(
        inputs.ground_direct_albedo_column_band,
        dtype=jnp.float64,
    )[column]
    ground_diffuse = jnp.asarray(
        inputs.ground_diffuse_albedo_column_band,
        dtype=jnp.float64,
    )[column]
    soil_direct = jnp.asarray(
        inputs.snow_free_soil_direct_albedo_column_band,
        dtype=jnp.float64,
    )[column]
    soil_diffuse = jnp.asarray(
        inputs.snow_free_soil_diffuse_albedo_column_band,
        dtype=jnp.float64,
    )[column]
    ground_direct = jnp.where(active[:, None], ground_direct, jnp.float64(0.2))
    ground_diffuse = jnp.where(active[:, None], ground_diffuse, jnp.float64(0.2))
    soil_direct = jnp.where(active[:, None], soil_direct, jnp.float64(0.2))
    soil_diffuse = jnp.where(active[:, None], soil_diffuse, jnp.float64(0.2))
    selected_direct_ground = jnp.where(snow_free_only, soil_direct, ground_direct)
    selected_diffuse_ground = jnp.where(snow_free_only, soil_diffuse, ground_diffuse)

    u1 = add(b, -divide(c1, selected_direct_ground))
    u2 = add(b, -multiply(c1, selected_direct_ground))
    u3 = add(f, multiply(c1, selected_direct_ground))
    tmp2 = add(u1, -avmu_h)
    tmp3 = add(u1, avmu_h)
    d1 = add(
        divide(multiply(p1, tmp2), s1),
        -multiply(multiply(p2, tmp3), s1),
    )
    tmp4 = add(u2, avmu_h)
    tmp5 = add(u2, -avmu_h)
    d2 = add(divide(tmp4, s1), -multiply(tmp5, s1))
    h1 = add(multiply(-d, p4), -multiply(c1, f))
    tmp6 = add(d, -divide(multiply(h1, p3), sigma))
    tmp7 = multiply(
        add(
            add(d, -c1),
            -multiply(divide(h1, sigma), add(u1, common_optical_depth)),
        ),
        s2,
    )
    h2 = divide(
        add(divide(multiply(tmp6, tmp2), s1), -multiply(p2, tmp7)),
        d1,
    )
    h3 = divide(
        -add(multiply(multiply(tmp6, tmp3), s1), -multiply(p1, tmp7)),
        d1,
    )
    h4 = add(multiply(-f, p3), -multiply(c1, d))
    tmp8 = divide(h4, sigma)
    tmp9 = multiply(add(u3, -multiply(tmp8, add(u2, -common_optical_depth))), s2)
    h5 = divide(-add(divide(multiply(tmp8, tmp4), s1), tmp9), d2)
    h6 = divide(add(multiply(multiply(tmp8, tmp5), s1), tmp9), d2)

    direct_surface_albedo = add(add(divide(h1, sigma), h2), h3)
    direct_to_diffuse_transmission = add(
        add(divide(multiply(h4, s2), sigma), multiply(h5, s1)),
        divide(h6, s1),
    )
    direct_beam_transmission = s2
    direct_absorption = add(
        add(
            add(one, -direct_surface_albedo),
            -multiply(add(one, -ground_direct), direct_beam_transmission),
        ),
        -multiply(add(one, -ground_diffuse), direct_to_diffuse_transmission),
    )

    one_minus_s2_squared = add(one, -multiply(s2, s2))
    one_minus_s2_s1 = add(one, -multiply(s2, s1))
    one_minus_s2_over_s1 = add(one, -divide(s2, s1))
    a1 = add(
        add(
            divide(
                multiply(divide(h1, sigma), one_minus_s2_squared),
                multiply(jnp.float64(2.0), twostext[:, None]),
            ),
            divide(multiply(h2, one_minus_s2_s1), add(twostext[:, None], h)),
        ),
        divide(multiply(h3, one_minus_s2_over_s1), add(twostext[:, None], -h)),
    )
    a2 = add(
        add(
            divide(
                multiply(divide(h4, sigma), one_minus_s2_squared),
                multiply(jnp.float64(2.0), twostext[:, None]),
            ),
            divide(multiply(h5, one_minus_s2_s1), add(twostext[:, None], h)),
        ),
        divide(multiply(h6, one_minus_s2_over_s1), add(twostext[:, None], -h)),
    )
    direct_sunlit_absorption = multiply(
        add(one, -omega),
        add(
            add(one, -s2),
            multiply(divide(one, avmu[:, None]), add(a1, a2)),
        ),
    )
    direct_shaded_absorption = add(direct_absorption, -direct_sunlit_absorption)

    u1 = add(b, -divide(c1, selected_diffuse_ground))
    u2 = add(b, -multiply(c1, selected_diffuse_ground))
    tmp2 = add(u1, -avmu_h)
    tmp3 = add(u1, avmu_h)
    d1 = add(
        divide(multiply(p1, tmp2), s1),
        -multiply(multiply(p2, tmp3), s1),
    )
    tmp4 = add(u2, avmu_h)
    tmp5 = add(u2, -avmu_h)
    d2 = add(divide(tmp4, s1), -multiply(tmp5, s1))
    h7 = divide(multiply(c1, tmp2), multiply(d1, s1))
    h8 = divide(multiply(multiply(-c1, tmp3), s1), d1)
    h9 = divide(tmp4, multiply(d2, s1))
    h10 = divide(multiply(-tmp5, s1), d2)

    diffuse_surface_albedo = add(h7, h8)
    diffuse_transmission = add(multiply(h9, s1), divide(h10, s1))
    diffuse_absorption = add(
        add(one, -diffuse_surface_albedo),
        -multiply(add(one, -ground_diffuse), diffuse_transmission),
    )
    a1 = add(
        divide(multiply(h7, one_minus_s2_s1), add(twostext[:, None], h)),
        divide(multiply(h8, one_minus_s2_over_s1), add(twostext[:, None], -h)),
    )
    a2 = add(
        divide(multiply(h9, one_minus_s2_s1), add(twostext[:, None], h)),
        divide(multiply(h10, one_minus_s2_over_s1), add(twostext[:, None], -h)),
    )
    diffuse_sunlit_absorption = multiply(
        divide(add(one, -omega), avmu[:, None]),
        add(a1, a2),
    )
    diffuse_shaded_absorption = add(diffuse_absorption, -diffuse_sunlit_absorption)

    visible_sunlit_fraction = divide(add(one, -s2_patch), direct_attenuation)
    layer_denominator_sun = multiply(visible_sunlit_fraction, area)
    layer_denominator_shade = multiply(add(one, -visible_sunlit_fraction), area)
    direct_sunlit_layer = divide(direct_sunlit_absorption[:, 0], layer_denominator_sun)
    diffuse_sunlit_layer = divide(diffuse_sunlit_absorption[:, 0], layer_denominator_sun)
    direct_shaded_layer = divide(direct_shaded_absorption[:, 0], layer_denominator_shade)
    diffuse_shaded_layer = divide(diffuse_shaded_absorption[:, 0], layer_denominator_shade)

    exposed_leaf_area = jnp.asarray(inputs.exposed_leaf_area_patch, dtype=jnp.float64)
    extkn = jnp.asarray(parameters.nitrogen_extinction_coefficient, dtype=jnp.float64)
    extkb = twostext
    combined_extinction = add(extkn, extkb)
    vcmax_sun = divide(
        add(
            one,
            -strict_negative_exp(-multiply(combined_extinction, exposed_leaf_area)),
        ),
        combined_extinction,
    )
    vcmax_shade = add(
        divide(
            add(one, -strict_negative_exp(-multiply(extkn, exposed_leaf_area))),
            extkn,
        ),
        -vcmax_sun,
    )
    positive_leaf_area = exposed_leaf_area > jnp.float64(0.0)
    vcmax_sun = jnp.where(
        positive_leaf_area,
        divide(vcmax_sun, jnp.where(positive_leaf_area,
            multiply(visible_sunlit_fraction, exposed_leaf_area), one)),
        jnp.float64(0.0),
    )
    vcmax_shade = jnp.where(
        positive_leaf_area,
        divide(
            vcmax_shade,
            jnp.where(positive_leaf_area,
                multiply(add(one, -visible_sunlit_fraction), exposed_leaf_area), one),
        ),
        jnp.float64(0.0),
    )

    update_band = active[:, None]
    update_layer = active[:, None] & ~snow_free_only
    update_normal_band = update_band & ~snow_free_only
    update_snow_free_band = update_band & snow_free_only

    def retain_band(old: jnp.ndarray, candidate: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(mask, candidate, jnp.asarray(old, dtype=jnp.float64))

    final_state = CanopyTwoStreamState(
        direct_surface_albedo_patch_band=retain_band(
            state.direct_surface_albedo_patch_band,
            direct_surface_albedo,
            update_normal_band,
        ),
        diffuse_surface_albedo_patch_band=retain_band(
            state.diffuse_surface_albedo_patch_band,
            diffuse_surface_albedo,
            update_normal_band,
        ),
        snow_free_direct_surface_albedo_patch_band=retain_band(
            state.snow_free_direct_surface_albedo_patch_band,
            direct_surface_albedo,
            update_snow_free_band,
        ),
        snow_free_diffuse_surface_albedo_patch_band=retain_band(
            state.snow_free_diffuse_surface_albedo_patch_band,
            diffuse_surface_albedo,
            update_snow_free_band,
        ),
        direct_absorption_patch_band=retain_band(
            state.direct_absorption_patch_band,
            direct_absorption,
            update_normal_band,
        ),
        direct_sunlit_absorption_patch_band=retain_band(
            state.direct_sunlit_absorption_patch_band,
            direct_sunlit_absorption,
            update_normal_band,
        ),
        direct_shaded_absorption_patch_band=retain_band(
            state.direct_shaded_absorption_patch_band,
            direct_shaded_absorption,
            update_normal_band,
        ),
        diffuse_absorption_patch_band=retain_band(
            state.diffuse_absorption_patch_band,
            diffuse_absorption,
            update_normal_band,
        ),
        diffuse_sunlit_absorption_patch_band=retain_band(
            state.diffuse_sunlit_absorption_patch_band,
            diffuse_sunlit_absorption,
            update_normal_band,
        ),
        diffuse_shaded_absorption_patch_band=retain_band(
            state.diffuse_shaded_absorption_patch_band,
            diffuse_shaded_absorption,
            update_normal_band,
        ),
        direct_beam_transmission_patch_band=retain_band(
            state.direct_beam_transmission_patch_band,
            direct_beam_transmission,
            update_normal_band,
        ),
        direct_to_diffuse_transmission_patch_band=retain_band(
            state.direct_to_diffuse_transmission_patch_band,
            direct_to_diffuse_transmission,
            update_normal_band,
        ),
        diffuse_transmission_patch_band=retain_band(
            state.diffuse_transmission_patch_band,
            diffuse_transmission,
            update_normal_band,
        ),
        sunlit_fraction_patch_layer=retain_band(
            state.sunlit_fraction_patch_layer,
            visible_sunlit_fraction[:, None],
            update_layer,
        ),
        direct_sunlit_absorption_patch_layer=retain_band(
            state.direct_sunlit_absorption_patch_layer,
            direct_sunlit_layer[:, None],
            update_layer,
        ),
        direct_shaded_absorption_patch_layer=retain_band(
            state.direct_shaded_absorption_patch_layer,
            direct_shaded_layer[:, None],
            update_layer,
        ),
        diffuse_sunlit_absorption_patch_layer=retain_band(
            state.diffuse_sunlit_absorption_patch_layer,
            diffuse_sunlit_layer[:, None],
            update_layer,
        ),
        diffuse_shaded_absorption_patch_layer=retain_band(
            state.diffuse_shaded_absorption_patch_layer,
            diffuse_shaded_layer[:, None],
            update_layer,
        ),
        sunlit_vcmax_scale_patch=jnp.where(
            active & ~snow_free_only,
            vcmax_sun,
            jnp.asarray(state.sunlit_vcmax_scale_patch, dtype=jnp.float64),
        ),
        shaded_vcmax_scale_patch=jnp.where(
            active & ~snow_free_only,
            vcmax_shade,
            jnp.asarray(state.shaded_vcmax_scale_patch, dtype=jnp.float64),
        ),
    )
    return CanopyTwoStreamResult(
        state=final_state,
        diagnostics=CanopyTwoStreamDiagnostics(
            updated_patch_count=jnp.sum(active, dtype=jnp.int32),
            snow_adjusted_patch_count=jnp.sum(
                active
                & ~snow_free_only
                & (jnp.asarray(inputs.snow_fraction_patch, dtype=jnp.float64) > 0.0),
                dtype=jnp.int32,
            ),
            snow_free_only=snow_free_only,
        ),
    )


def _validate_canopy_surface_albedo_shapes(
    inputs: CanopySurfaceAlbedoInputs,
    state: CanopySurfaceAlbedoState,
    parameters: CanopySurfaceAlbedoParameters,
) -> int:
    if jnp.ndim(inputs.nonurban_patch) != 1:
        raise ValueError("nonurban_patch must be a vector")
    patch_count = inputs.nonurban_patch.shape[0]
    for name in (
        "patch_column",
        "landunit_type_patch",
        "pft_type_patch",
        "cosine_solar_zenith_patch",
        "vegetation_temperature_patch",
        "wet_fraction_patch",
        "snow_fraction_patch",
        "exposed_leaf_area_patch",
        "exposed_stem_area_patch",
    ):
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    table_shape = jnp.shape(inputs.leaf_reflectance_pft_band)
    if len(table_shape) != 2 or table_shape[1] != NUMBER_OF_SHORTWAVE_BANDS:
        raise ValueError("leaf reflectance must have shape (pft, 2)")
    pft_count = table_shape[0]
    for name in (
        "stem_reflectance_pft_band",
        "leaf_transmittance_pft_band",
        "stem_transmittance_pft_band",
    ):
        if jnp.shape(getattr(inputs, name)) != table_shape:
            raise ValueError(f"{name} must use the common PFT table")
    if jnp.shape(inputs.leaf_orientation_pft) != (pft_count,):
        raise ValueError("leaf_orientation_pft must contain every PFT")
    column_shape = jnp.shape(inputs.ground_direct_albedo_column_band)
    if len(column_shape) != 2 or column_shape[1] != NUMBER_OF_SHORTWAVE_BANDS:
        raise ValueError("ground direct albedo must have shape (column, 2)")
    for name in (
        "ground_diffuse_albedo_column_band",
        "snow_free_soil_direct_albedo_column_band",
        "snow_free_soil_diffuse_albedo_column_band",
    ):
        if jnp.shape(getattr(inputs, name)) != column_shape:
            raise ValueError(f"{name} must use the common column grid")
    two_stream = state.two_stream
    patch_band_shape = (patch_count, NUMBER_OF_SHORTWAVE_BANDS)
    for name in CanopyTwoStreamState._fields[:13]:
        if jnp.shape(getattr(two_stream, name)) != patch_band_shape:
            raise ValueError(f"two_stream.{name} must have shape {patch_band_shape}")
    patch_layer_shape = (patch_count, 1)
    for name in CanopyTwoStreamState._fields[13:18]:
        if jnp.shape(getattr(two_stream, name)) != patch_layer_shape:
            raise ValueError(f"two_stream.{name} must have shape {patch_layer_shape}")
    for name in CanopyTwoStreamState._fields[18:]:
        if jnp.shape(getattr(two_stream, name)) != (patch_count,):
            raise ValueError(f"two_stream.{name} must contain every patch")
    for name in ("leaf_area_patch_layer", "stem_area_patch_layer"):
        if jnp.shape(getattr(state, name)) != patch_layer_shape:
            raise ValueError(f"{name} must have shape {patch_layer_shape}")
    for name in ("radiative_layer_count_patch", "canopy_layer_count_patch"):
        if jnp.shape(getattr(state, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    if jnp.shape(parameters.snow_scattering_albedo_band) != (NUMBER_OF_SHORTWAVE_BANDS,):
        raise ValueError("snow scattering albedo must contain visible and NIR values")
    return patch_count


def canopy_surface_albedo_big_leaf(
    inputs: CanopySurfaceAlbedoInputs,
    state: CanopySurfaceAlbedoState,
    parameters: CanopySurfaceAlbedoParameters,
) -> CanopySurfaceAlbedoResult:

    _validate_canopy_surface_albedo_shapes(inputs, state, parameters)
    one = jnp.float64(1.0)
    zero = jnp.float64(0.0)
    nonurban = jnp.asarray(inputs.nonurban_patch, dtype=jnp.bool_)
    update_band = nonurban[:, None]
    update_layer = nonurban[:, None]
    snow_free_diagnostic = jnp.asarray(parameters.snow_free_diagnostic, dtype=jnp.bool_)

    old = state.two_stream

    def initialize_band(old_value: jnp.ndarray, value: float) -> jnp.ndarray:
        return jnp.where(
            update_band,
            jnp.float64(value),
            jnp.asarray(old_value, dtype=jnp.float64),
        )

    initialized_snow_free_direct = jnp.where(
        update_band & snow_free_diagnostic,
        one,
        jnp.asarray(old.snow_free_direct_surface_albedo_patch_band, dtype=jnp.float64),
    )
    initialized_snow_free_diffuse = jnp.where(
        update_band & snow_free_diagnostic,
        one,
        jnp.asarray(old.snow_free_diffuse_surface_albedo_patch_band, dtype=jnp.float64),
    )

    exposed_leaf_area = jnp.asarray(inputs.exposed_leaf_area_patch, dtype=jnp.float64)
    exposed_stem_area = jnp.asarray(inputs.exposed_stem_area_patch, dtype=jnp.float64)
    extkn = jnp.asarray(parameters.nitrogen_extinction_coefficient, dtype=jnp.float64)
    integrated_nitrogen = fortran_divide(
        fortran_add(
            one,
            -strict_negative_exp(-fortran_multiply(extkn, exposed_leaf_area)),
        ),
        extkn,
    )
    positive_leaf_area = exposed_leaf_area > zero
    safe_default_leaf_area = jnp.where(positive_leaf_area, exposed_leaf_area, one)
    shaded_vcmax_default = jnp.where(
        positive_leaf_area,
        fortran_divide(integrated_nitrogen, safe_default_leaf_area),
        zero,
    )
    initialized = CanopyTwoStreamState(
        direct_surface_albedo_patch_band=initialize_band(
            old.direct_surface_albedo_patch_band,
            1.0,
        ),
        diffuse_surface_albedo_patch_band=initialize_band(
            old.diffuse_surface_albedo_patch_band,
            1.0,
        ),
        snow_free_direct_surface_albedo_patch_band=initialized_snow_free_direct,
        snow_free_diffuse_surface_albedo_patch_band=initialized_snow_free_diffuse,
        direct_absorption_patch_band=initialize_band(old.direct_absorption_patch_band, 0.0),
        direct_sunlit_absorption_patch_band=initialize_band(
            old.direct_sunlit_absorption_patch_band,
            0.0,
        ),
        direct_shaded_absorption_patch_band=initialize_band(
            old.direct_shaded_absorption_patch_band,
            0.0,
        ),
        diffuse_absorption_patch_band=initialize_band(
            old.diffuse_absorption_patch_band,
            0.0,
        ),
        diffuse_sunlit_absorption_patch_band=initialize_band(
            old.diffuse_sunlit_absorption_patch_band,
            0.0,
        ),
        diffuse_shaded_absorption_patch_band=initialize_band(
            old.diffuse_shaded_absorption_patch_band,
            0.0,
        ),
        direct_beam_transmission_patch_band=initialize_band(
            old.direct_beam_transmission_patch_band,
            0.0,
        ),
        direct_to_diffuse_transmission_patch_band=initialize_band(
            old.direct_to_diffuse_transmission_patch_band,
            0.0,
        ),
        diffuse_transmission_patch_band=initialize_band(
            old.diffuse_transmission_patch_band,
            0.0,
        ),
        sunlit_fraction_patch_layer=jnp.where(
            update_layer,
            zero,
            jnp.asarray(old.sunlit_fraction_patch_layer, dtype=jnp.float64),
        ),
        direct_sunlit_absorption_patch_layer=jnp.where(
            update_layer,
            zero,
            jnp.asarray(old.direct_sunlit_absorption_patch_layer, dtype=jnp.float64),
        ),
        direct_shaded_absorption_patch_layer=jnp.where(
            update_layer,
            zero,
            jnp.asarray(old.direct_shaded_absorption_patch_layer, dtype=jnp.float64),
        ),
        diffuse_sunlit_absorption_patch_layer=jnp.where(
            update_layer,
            zero,
            jnp.asarray(old.diffuse_sunlit_absorption_patch_layer, dtype=jnp.float64),
        ),
        diffuse_shaded_absorption_patch_layer=jnp.where(
            update_layer,
            zero,
            jnp.asarray(old.diffuse_shaded_absorption_patch_layer, dtype=jnp.float64),
        ),
        sunlit_vcmax_scale_patch=jnp.where(
            nonurban,
            zero,
            jnp.asarray(old.sunlit_vcmax_scale_patch, dtype=jnp.float64),
        ),
        shaded_vcmax_scale_patch=jnp.where(
            nonurban,
            shaded_vcmax_default,
            jnp.asarray(old.shaded_vcmax_scale_patch, dtype=jnp.float64),
        ),
    )

    leaf_area_layer = jnp.where(
        update_layer,
        exposed_leaf_area[:, None],
        jnp.asarray(state.leaf_area_patch_layer, dtype=jnp.float64),
    )
    stem_area_layer = jnp.where(
        update_layer,
        exposed_stem_area[:, None],
        jnp.asarray(state.stem_area_patch_layer, dtype=jnp.float64),
    )
    radiative_layer_count = jnp.where(
        nonurban,
        jnp.int32(1),
        jnp.asarray(state.radiative_layer_count_patch, dtype=jnp.int32),
    )
    canopy_layer_count = jnp.where(
        nonurban,
        jnp.int32(1),
        jnp.asarray(state.canopy_layer_count_patch, dtype=jnp.int32),
    )

    cosine = jnp.asarray(inputs.cosine_solar_zenith_patch, dtype=jnp.float64)
    landunit_type = jnp.asarray(inputs.landunit_type_patch, dtype=jnp.int32)
    daylight = cosine > zero
    vegetated_landunit = (landunit_type == jnp.int32(LANDUNIT_SOIL)) | (
        landunit_type == jnp.int32(LANDUNIT_CROP)
    )
    exposed_area = fortran_add(exposed_leaf_area, exposed_stem_area)
    vegetated_daylight = nonurban & daylight & vegetated_landunit & (exposed_area > zero)
    nonvegetated_daylight = nonurban & daylight & ~vegetated_daylight

    leaf_weight = fortran_divide(
        exposed_leaf_area,
        jnp.maximum(exposed_area, jnp.float64(1.0e-6)),
    )
    stem_weight = fortran_divide(
        exposed_stem_area,
        jnp.maximum(exposed_area, jnp.float64(1.0e-6)),
    )
    pft_type = jnp.asarray(inputs.pft_type_patch, dtype=jnp.int32)
    leaf_reflectance = jnp.asarray(
        inputs.leaf_reflectance_pft_band,
        dtype=jnp.float64,
    )[pft_type]
    stem_reflectance = jnp.asarray(
        inputs.stem_reflectance_pft_band,
        dtype=jnp.float64,
    )[pft_type]
    leaf_transmittance = jnp.asarray(
        inputs.leaf_transmittance_pft_band,
        dtype=jnp.float64,
    )[pft_type]
    stem_transmittance = jnp.asarray(
        inputs.stem_transmittance_pft_band,
        dtype=jnp.float64,
    )[pft_type]
    reflectance = jnp.maximum(
        fortran_add(
            fortran_multiply(leaf_reflectance, leaf_weight[:, None]),
            fortran_multiply(stem_reflectance, stem_weight[:, None]),
        ),
        jnp.float64(1.0e-6),
    )
    transmittance = jnp.maximum(
        fortran_add(
            fortran_multiply(leaf_transmittance, leaf_weight[:, None]),
            fortran_multiply(stem_transmittance, stem_weight[:, None]),
        ),
        jnp.float64(1.0e-6),
    )

    safe_cosine = fortran_materialize(jnp.where(vegetated_daylight, cosine, jnp.float64(0.5)))
    safe_leaf_area = fortran_materialize(jnp.where(vegetated_daylight, exposed_leaf_area, one))
    safe_stem_area = fortran_materialize(
        jnp.where(vegetated_daylight, exposed_stem_area, jnp.float64(0.1))
    )
    leaf_orientation = jnp.asarray(inputs.leaf_orientation_pft, dtype=jnp.float64)[pft_type]
    two_stream_inputs = CanopyTwoStreamInputs(
        active_patch=vegetated_daylight,
        patch_column=inputs.patch_column,
        leaf_orientation_patch=fortran_materialize(
            jnp.where(vegetated_daylight, leaf_orientation, zero)
        ),
        cosine_solar_zenith_patch=safe_cosine,
        reflectance_patch_band=fortran_materialize(
            jnp.where(
                vegetated_daylight[:, None],
                reflectance,
                jnp.float64(0.1),
            )
        ),
        transmittance_patch_band=fortran_materialize(
            jnp.where(
                vegetated_daylight[:, None],
                transmittance,
                jnp.float64(0.05),
            )
        ),
        vegetation_temperature_patch=inputs.vegetation_temperature_patch,
        wet_fraction_patch=inputs.wet_fraction_patch,
        snow_fraction_patch=inputs.snow_fraction_patch,
        exposed_leaf_area_patch=safe_leaf_area,
        exposed_stem_area_patch=safe_stem_area,
        ground_direct_albedo_column_band=inputs.ground_direct_albedo_column_band,
        ground_diffuse_albedo_column_band=inputs.ground_diffuse_albedo_column_band,
        snow_free_soil_direct_albedo_column_band=(inputs.snow_free_soil_direct_albedo_column_band),
        snow_free_soil_diffuse_albedo_column_band=(
            inputs.snow_free_soil_diffuse_albedo_column_band
        ),
    )
    child_parameters = CanopyTwoStreamParameters(
        snow_scattering_albedo_band=parameters.snow_scattering_albedo_band,
        snow_direct_upscatter=parameters.snow_direct_upscatter,
        snow_diffuse_upscatter=parameters.snow_diffuse_upscatter,
        nitrogen_extinction_coefficient=parameters.nitrogen_extinction_coefficient,
    )
    normal = canopy_two_stream_big_leaf(two_stream_inputs, initialized, child_parameters)
    snow_free = canopy_two_stream_big_leaf(
        two_stream_inputs,
        normal.state,
        child_parameters._replace(snow_free_only=True),
    )
    normal_state = normal.state
    after_canopy = normal_state._replace(
        snow_free_direct_surface_albedo_patch_band=jnp.where(
            snow_free_diagnostic,
            snow_free.state.snow_free_direct_surface_albedo_patch_band,
            normal_state.snow_free_direct_surface_albedo_patch_band,
        ),
        snow_free_diffuse_surface_albedo_patch_band=jnp.where(
            snow_free_diagnostic,
            snow_free.state.snow_free_diffuse_surface_albedo_patch_band,
            normal_state.snow_free_diffuse_surface_albedo_patch_band,
        ),
    )

    column = jnp.asarray(inputs.patch_column, dtype=jnp.int32)
    ground_direct = jnp.asarray(
        inputs.ground_direct_albedo_column_band,
        dtype=jnp.float64,
    )[column]
    ground_diffuse = jnp.asarray(
        inputs.ground_diffuse_albedo_column_band,
        dtype=jnp.float64,
    )[column]
    soil_direct = jnp.asarray(
        inputs.snow_free_soil_direct_albedo_column_band,
        dtype=jnp.float64,
    )[column]
    soil_diffuse = jnp.asarray(
        inputs.snow_free_soil_diffuse_albedo_column_band,
        dtype=jnp.float64,
    )[column]
    nonvegetated_band = nonvegetated_daylight[:, None]

    def set_nonvegetated(old_value: jnp.ndarray, value: jnp.ndarray | float) -> jnp.ndarray:
        return jnp.where(
            nonvegetated_band,
            jnp.asarray(value, dtype=jnp.float64),
            old_value,
        )

    final_two_stream = after_canopy._replace(
        direct_surface_albedo_patch_band=set_nonvegetated(
            after_canopy.direct_surface_albedo_patch_band,
            ground_direct,
        ),
        diffuse_surface_albedo_patch_band=set_nonvegetated(
            after_canopy.diffuse_surface_albedo_patch_band,
            ground_diffuse,
        ),
        snow_free_direct_surface_albedo_patch_band=jnp.where(
            nonvegetated_band & snow_free_diagnostic,
            soil_direct,
            after_canopy.snow_free_direct_surface_albedo_patch_band,
        ),
        snow_free_diffuse_surface_albedo_patch_band=jnp.where(
            nonvegetated_band & snow_free_diagnostic,
            soil_diffuse,
            after_canopy.snow_free_diffuse_surface_albedo_patch_band,
        ),
        direct_absorption_patch_band=set_nonvegetated(
            after_canopy.direct_absorption_patch_band,
            zero,
        ),
        direct_sunlit_absorption_patch_band=set_nonvegetated(
            after_canopy.direct_sunlit_absorption_patch_band,
            zero,
        ),
        direct_shaded_absorption_patch_band=set_nonvegetated(
            after_canopy.direct_shaded_absorption_patch_band,
            zero,
        ),
        diffuse_absorption_patch_band=set_nonvegetated(
            after_canopy.diffuse_absorption_patch_band,
            zero,
        ),
        diffuse_sunlit_absorption_patch_band=set_nonvegetated(
            after_canopy.diffuse_sunlit_absorption_patch_band,
            zero,
        ),
        diffuse_shaded_absorption_patch_band=set_nonvegetated(
            after_canopy.diffuse_shaded_absorption_patch_band,
            zero,
        ),
        direct_beam_transmission_patch_band=set_nonvegetated(
            after_canopy.direct_beam_transmission_patch_band,
            one,
        ),
        direct_to_diffuse_transmission_patch_band=set_nonvegetated(
            after_canopy.direct_to_diffuse_transmission_patch_band,
            zero,
        ),
        diffuse_transmission_patch_band=set_nonvegetated(
            after_canopy.diffuse_transmission_patch_band,
            one,
        ),
    )
    return CanopySurfaceAlbedoResult(
        state=CanopySurfaceAlbedoState(
            two_stream=final_two_stream,
            leaf_area_patch_layer=leaf_area_layer,
            stem_area_patch_layer=stem_area_layer,
            radiative_layer_count_patch=radiative_layer_count,
            canopy_layer_count_patch=canopy_layer_count,
        ),
        diagnostics=CanopySurfaceAlbedoDiagnostics(
            nonurban_patch_count=jnp.sum(nonurban, dtype=jnp.int32),
            vegetated_daylight_patch_count=jnp.sum(
                vegetated_daylight,
                dtype=jnp.int32,
            ),
            nonvegetated_daylight_patch_count=jnp.sum(
                nonvegetated_daylight,
                dtype=jnp.int32,
            ),
            night_patch_count=jnp.sum(nonurban & ~daylight, dtype=jnp.int32),
            snow_adjusted_patch_count=normal.diagnostics.snow_adjusted_patch_count,
            snow_free_diagnostic=snow_free_diagnostic,
        ),
    )


def _validate_surface_albedo_shapes(
    inputs: SurfaceAlbedoInputs,
    state: SurfaceAlbedoState,
) -> tuple[int, int, int]:
    if jnp.ndim(inputs.nonurban_column) != 1:
        raise ValueError("nonurban_column must be a vector")
    if jnp.ndim(inputs.nonurban_patch) != 1:
        raise ValueError("nonurban_patch must be a vector")
    if jnp.ndim(inputs.cosine_solar_zenith_gridcell) != 1:
        raise ValueError("cosine_solar_zenith_gridcell must be a vector")
    column_count = inputs.nonurban_column.shape[0]
    patch_count = inputs.nonurban_patch.shape[0]
    gridcell_count = inputs.cosine_solar_zenith_gridcell.shape[0]
    for name in (
        "column_gridcell",
        "landunit_type_column",
        "snow_layer_count_column",
        "ground_temperature_column",
        "top_soil_water_volume_fraction_column",
        "soil_color_column",
        "snow_fraction_column",
        "snow_water_equivalent_column",
    ):
        if jnp.shape(getattr(inputs, name)) != (column_count,):
            raise ValueError(f"{name} must contain every column")
    if jnp.shape(inputs.lake_ice_fraction_top_two_column) != (column_count, 2):
        raise ValueError(f"lake_ice_fraction_top_two_column must have shape {(column_count, 2)}")
    snow_shape = (column_count, NUMBER_OF_SNOW_LAYERS)
    for name in (
        "snow_liquid_column_layer",
        "snow_ice_column_layer",
        "snow_grain_radius_column_layer",
    ):
        if jnp.shape(getattr(inputs, name)) != snow_shape:
            raise ValueError(f"{name} must have shape {snow_shape}")
    aerosol_shape = snow_shape + (NUMBER_OF_SNOW_AEROSOLS,)
    if jnp.shape(inputs.aerosol_mass_concentration_column_layer_species) != aerosol_shape:
        raise ValueError(
            f"aerosol_mass_concentration_column_layer_species must have shape {aerosol_shape}"
        )
    for name in (
        "patch_gridcell",
        "patch_column",
        "pft_type_patch",
        "vegetation_temperature_patch",
        "wet_fraction_patch",
        "canopy_snow_fraction_patch",
        "exposed_leaf_area_patch",
        "exposed_stem_area_patch",
    ):
        if jnp.shape(getattr(inputs, name)) != (patch_count,):
            raise ValueError(f"{name} must contain every patch")
    pft_band_shape = jnp.shape(inputs.leaf_reflectance_pft_band)
    if len(pft_band_shape) != 2 or pft_band_shape[1] != NUMBER_OF_SHORTWAVE_BANDS:
        raise ValueError("leaf_reflectance_pft_band must have shape (pft, 2)")
    for name in (
        "stem_reflectance_pft_band",
        "leaf_transmittance_pft_band",
        "stem_transmittance_pft_band",
    ):
        if jnp.shape(getattr(inputs, name)) != pft_band_shape:
            raise ValueError(f"{name} must use the common PFT table")
    if jnp.shape(inputs.leaf_orientation_pft) != (pft_band_shape[0],):
        raise ValueError("leaf_orientation_pft must contain every PFT")
    if jnp.shape(state.cosine_solar_zenith_column) != (column_count,):
        raise ValueError("state cosine_solar_zenith_column must contain every column")
    band_shape = (column_count, NUMBER_OF_SHORTWAVE_BANDS)
    for name in SurfaceAlbedoState._fields[4:]:
        if jnp.shape(getattr(state, name)) != band_shape:
            raise ValueError(f"state.{name} must have shape {band_shape}")
    return gridcell_count, column_count, patch_count


def _fortran_nint_surface(value: jnp.ndarray) -> jnp.ndarray:

    value = jnp.asarray(value, dtype=jnp.float64)
    rounded = jnp.where(
        value >= jnp.float64(0.0),
        jnp.floor(fortran_add(value, jnp.float64(0.5))),
        jnp.ceil(fortran_add(value, jnp.float64(-0.5))),
    )
    return rounded.astype(jnp.int32)


def surface_albedo_columns_configured(
    inputs: SurfaceAlbedoColumnInputs,
    state: SurfaceAlbedoColumnState,
    parameters: SurfaceAlbedoColumnParameters,
    *,
    early_exit_snicar_retry_loop: bool = False,
    compact_active_snow_columns: bool = False,
    compact_snow_diagnostics: bool = False,
) -> SurfaceAlbedoColumnResult:

    zero = jnp.float64(0.0)
    nonurban_column = jnp.asarray(inputs.nonurban_column, dtype=jnp.bool_)
    cosine_column = jnp.asarray(
        inputs.cosine_solar_zenith_column,
        dtype=jnp.float64,
    )

    def initialize_column_band(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(
            nonurban_column[:, None],
            zero,
            jnp.asarray(value, dtype=jnp.float64),
        )

    initialized_soil = SoilAlbedoState(
        direct_column_band=initialize_column_band(state.soil.direct_column_band),
        diffuse_column_band=initialize_column_band(state.soil.diffuse_column_band),
    )
    soil_inputs = SoilAlbedoInputs(
        nonurban_column=inputs.nonurban_column,
        landunit_type_column=inputs.landunit_type_column,
        cosine_solar_zenith_column=cosine_column,
        snow_layer_count_column=inputs.snow_layer_count_column,
        ground_temperature_column=inputs.ground_temperature_column,
        top_soil_water_volume_fraction_column=(inputs.top_soil_water_volume_fraction_column),
        lake_ice_fraction_top_two_column=inputs.lake_ice_fraction_top_two_column,
        soil_color_column=inputs.soil_color_column,
    )
    soil = soil_albedo(soil_inputs, initialized_soil, parameters.soil)

    aerosol = jnp.asarray(
        inputs.aerosol_mass_concentration_column_layer_species,
        dtype=jnp.float64,
    )
    species = jnp.arange(NUMBER_OF_SNOW_AEROSOLS, dtype=jnp.int32)
    configured_species = (species < jnp.int32(2)) | (species >= jnp.int32(4))
    feedback_aerosol = jnp.where(configured_species[None, None, :], aerosol, zero)
    optical_inputs = SnicarOpticalInputs(
        nonurban_column=inputs.nonurban_column,
        cosine_solar_zenith_column=cosine_column,
        snow_layer_count_column=inputs.snow_layer_count_column,
        snow_water_equivalent_column=inputs.snow_water_equivalent_column,
        snow_liquid_column_layer=inputs.snow_liquid_column_layer,
        snow_ice_column_layer=inputs.snow_ice_column_layer,
        snow_grain_radius_column_layer=_fortran_nint_surface(inputs.snow_grain_radius_column_layer),
        aerosol_mass_concentration_column_layer_species=feedback_aerosol,
    )
    snow_inputs = SnicarRadiativeTransferInputs(
        optical=optical_inputs,
        underlying_surface_albedo_column_band=soil.state.diffuse_column_band,
    )
    direct_snow = snicar_radiative_transfer(
        snow_inputs,
        parameters.snow_optical_tables,
        True,
        early_exit_retry_loop=early_exit_snicar_retry_loop,
        compact_active_columns=compact_active_snow_columns,
        runtime_diagnostics=compact_snow_diagnostics,
    )
    diffuse_snow = snicar_radiative_transfer(
        snow_inputs,
        parameters.snow_optical_tables,
        False,
        early_exit_retry_loop=early_exit_snicar_retry_loop,
        compact_active_columns=compact_active_snow_columns,
        runtime_diagnostics=compact_snow_diagnostics,
    )

    initialized_ground = GroundSnowAlbedoState(
        ground_direct_albedo_column_band=initialize_column_band(
            state.ground.ground_direct_albedo_column_band
        ),
        ground_diffuse_albedo_column_band=initialize_column_band(
            state.ground.ground_diffuse_albedo_column_band
        ),
        snow_direct_albedo_history_column_band=(
            state.ground.snow_direct_albedo_history_column_band
        ),
        snow_diffuse_albedo_history_column_band=(
            state.ground.snow_diffuse_albedo_history_column_band
        ),
        direct_visible_absorption_column_level=jnp.where(
            nonurban_column[:, None],
            zero,
            jnp.asarray(
                state.ground.direct_visible_absorption_column_level,
                dtype=jnp.float64,
            ),
        ),
        direct_nir_absorption_column_level=jnp.where(
            nonurban_column[:, None],
            zero,
            jnp.asarray(
                state.ground.direct_nir_absorption_column_level,
                dtype=jnp.float64,
            ),
        ),
        diffuse_visible_absorption_column_level=jnp.where(
            nonurban_column[:, None],
            zero,
            jnp.asarray(
                state.ground.diffuse_visible_absorption_column_level,
                dtype=jnp.float64,
            ),
        ),
        diffuse_nir_absorption_column_level=jnp.where(
            nonurban_column[:, None],
            zero,
            jnp.asarray(
                state.ground.diffuse_nir_absorption_column_level,
                dtype=jnp.float64,
            ),
        ),
    )
    ground = weight_ground_snow_albedo(
        GroundSnowAlbedoInputs(
            nonurban_column=inputs.nonurban_column,
            landunit_type_column=inputs.landunit_type_column,
            cosine_solar_zenith_column=cosine_column,
            snow_fraction_column=inputs.snow_fraction_column,
            snow_water_equivalent_column=inputs.snow_water_equivalent_column,
            soil_direct_albedo_column_band=soil.state.direct_column_band,
            soil_diffuse_albedo_column_band=soil.state.diffuse_column_band,
            snow_direct_albedo_column_band=(direct_snow.state.snow_albedo_column_band),
            snow_diffuse_albedo_column_band=(diffuse_snow.state.snow_albedo_column_band),
            snow_direct_absorption_column_level_band=(
                direct_snow.state.absorbed_flux_column_level_band
            ),
            snow_diffuse_absorption_column_level_band=(
                diffuse_snow.state.absorbed_flux_column_level_band
            ),
        ),
        initialized_ground,
        subgrid_fluxes_enabled=parameters.subgrid_fluxes_enabled,
    )

    final_state = SurfaceAlbedoColumnState(
        cosine_solar_zenith_column=cosine_column,
        soil=soil.state,
        ground=ground.state,
        pure_snow_ground_direct_albedo_column_band=initialize_column_band(
            state.pure_snow_ground_direct_albedo_column_band
        ),
        pure_snow_ground_diffuse_albedo_column_band=initialize_column_band(
            state.pure_snow_ground_diffuse_albedo_column_band
        ),
        no_black_carbon_ground_direct_albedo_column_band=initialize_column_band(
            state.no_black_carbon_ground_direct_albedo_column_band
        ),
        no_black_carbon_ground_diffuse_albedo_column_band=initialize_column_band(
            state.no_black_carbon_ground_diffuse_albedo_column_band
        ),
        no_organic_carbon_ground_direct_albedo_column_band=initialize_column_band(
            state.no_organic_carbon_ground_direct_albedo_column_band
        ),
        no_organic_carbon_ground_diffuse_albedo_column_band=initialize_column_band(
            state.no_organic_carbon_ground_diffuse_albedo_column_band
        ),
        no_dust_ground_direct_albedo_column_band=initialize_column_band(
            state.no_dust_ground_direct_albedo_column_band
        ),
        no_dust_ground_diffuse_albedo_column_band=initialize_column_band(
            state.no_dust_ground_diffuse_albedo_column_band
        ),
    )
    return SurfaceAlbedoColumnResult(
        state=final_state,
        direct_snow=direct_snow,
        diffuse_snow=diffuse_snow,
        diagnostics=SurfaceAlbedoColumnDiagnostics(
            soil=soil.diagnostics,
            direct_snow=direct_snow.diagnostics,
            diffuse_snow=diffuse_snow.diagnostics,
            ground=ground.diagnostics,
        ),
    )


def surface_albedo_column_inputs(
    inputs: SurfaceAlbedoInputs,
) -> SurfaceAlbedoColumnInputs:

    gridcell_cosine = jnp.asarray(
        inputs.cosine_solar_zenith_gridcell,
        dtype=jnp.float64,
    )
    column_gridcell = jnp.asarray(inputs.column_gridcell, dtype=jnp.int32)
    return SurfaceAlbedoColumnInputs(
        nonurban_column=inputs.nonurban_column,
        cosine_solar_zenith_column=gridcell_cosine[column_gridcell],
        landunit_type_column=inputs.landunit_type_column,
        snow_layer_count_column=inputs.snow_layer_count_column,
        ground_temperature_column=inputs.ground_temperature_column,
        top_soil_water_volume_fraction_column=(
            inputs.top_soil_water_volume_fraction_column
        ),
        lake_ice_fraction_top_two_column=inputs.lake_ice_fraction_top_two_column,
        soil_color_column=inputs.soil_color_column,
        snow_fraction_column=inputs.snow_fraction_column,
        snow_water_equivalent_column=inputs.snow_water_equivalent_column,
        snow_liquid_column_layer=inputs.snow_liquid_column_layer,
        snow_ice_column_layer=inputs.snow_ice_column_layer,
        snow_grain_radius_column_layer=inputs.snow_grain_radius_column_layer,
        aerosol_mass_concentration_column_layer_species=(
            inputs.aerosol_mass_concentration_column_layer_species
        ),
    )


def surface_albedo_column_state(
    state: SurfaceAlbedoState,
) -> SurfaceAlbedoColumnState:

    return SurfaceAlbedoColumnState(
        cosine_solar_zenith_column=state.cosine_solar_zenith_column,
        soil=state.soil,
        ground=state.ground,
        pure_snow_ground_direct_albedo_column_band=(
            state.pure_snow_ground_direct_albedo_column_band
        ),
        pure_snow_ground_diffuse_albedo_column_band=(
            state.pure_snow_ground_diffuse_albedo_column_band
        ),
        no_black_carbon_ground_direct_albedo_column_band=(
            state.no_black_carbon_ground_direct_albedo_column_band
        ),
        no_black_carbon_ground_diffuse_albedo_column_band=(
            state.no_black_carbon_ground_diffuse_albedo_column_band
        ),
        no_organic_carbon_ground_direct_albedo_column_band=(
            state.no_organic_carbon_ground_direct_albedo_column_band
        ),
        no_organic_carbon_ground_diffuse_albedo_column_band=(
            state.no_organic_carbon_ground_diffuse_albedo_column_band
        ),
        no_dust_ground_direct_albedo_column_band=(
            state.no_dust_ground_direct_albedo_column_band
        ),
        no_dust_ground_diffuse_albedo_column_band=(
            state.no_dust_ground_diffuse_albedo_column_band
        ),
    )


def surface_albedo_column_parameters(
    parameters: SurfaceAlbedoParameters,
) -> SurfaceAlbedoColumnParameters:
    return SurfaceAlbedoColumnParameters(
        soil=parameters.soil,
        snow_optical_tables=parameters.snow_optical_tables,
        subgrid_fluxes_enabled=parameters.subgrid_fluxes_enabled,
    )


def surface_albedo_canopy_inputs(
    inputs: SurfaceAlbedoInputs,
    columns: SurfaceAlbedoColumnState,
) -> CanopySurfaceAlbedoInputs:

    patch_column = jnp.asarray(inputs.patch_column, dtype=jnp.int32)
    patch_gridcell = jnp.asarray(inputs.patch_gridcell, dtype=jnp.int32)
    gridcell_cosine = jnp.asarray(
        inputs.cosine_solar_zenith_gridcell,
        dtype=jnp.float64,
    )
    landunit_type_column = jnp.asarray(
        inputs.landunit_type_column,
        dtype=jnp.int32,
    )
    return CanopySurfaceAlbedoInputs(
        nonurban_patch=inputs.nonurban_patch,
        patch_column=patch_column,
        landunit_type_patch=landunit_type_column[patch_column],
        pft_type_patch=inputs.pft_type_patch,
        cosine_solar_zenith_patch=gridcell_cosine[patch_gridcell],
        leaf_orientation_pft=inputs.leaf_orientation_pft,
        leaf_reflectance_pft_band=inputs.leaf_reflectance_pft_band,
        stem_reflectance_pft_band=inputs.stem_reflectance_pft_band,
        leaf_transmittance_pft_band=inputs.leaf_transmittance_pft_band,
        stem_transmittance_pft_band=inputs.stem_transmittance_pft_band,
        vegetation_temperature_patch=inputs.vegetation_temperature_patch,
        wet_fraction_patch=inputs.wet_fraction_patch,
        snow_fraction_patch=inputs.canopy_snow_fraction_patch,
        exposed_leaf_area_patch=inputs.exposed_leaf_area_patch,
        exposed_stem_area_patch=inputs.exposed_stem_area_patch,
        ground_direct_albedo_column_band=(
            columns.ground.ground_direct_albedo_column_band
        ),
        ground_diffuse_albedo_column_band=(
            columns.ground.ground_diffuse_albedo_column_band
        ),
        snow_free_soil_direct_albedo_column_band=columns.soil.direct_column_band,
        snow_free_soil_diffuse_albedo_column_band=columns.soil.diffuse_column_band,
    )


def surface_albedo_configured(
    inputs: SurfaceAlbedoInputs,
    state: SurfaceAlbedoState,
    parameters: SurfaceAlbedoParameters,
    *,
    early_exit_snicar_retry_loop: bool = False,
    compact_active_snow_columns: bool = False,
    compact_snow_diagnostics: bool = False,
) -> SurfaceAlbedoResult:

    _validate_surface_albedo_shapes(inputs, state)
    columns = surface_albedo_columns_configured(
        surface_albedo_column_inputs(inputs),
        surface_albedo_column_state(state),
        surface_albedo_column_parameters(parameters),
        early_exit_snicar_retry_loop=early_exit_snicar_retry_loop,
        compact_active_snow_columns=compact_active_snow_columns,
        compact_snow_diagnostics=compact_snow_diagnostics,
    )
    canopy = canopy_surface_albedo_big_leaf(
        surface_albedo_canopy_inputs(inputs, columns.state),
        state.canopy,
        parameters.canopy,
    )
    column_state = columns.state
    return SurfaceAlbedoResult(
        state=SurfaceAlbedoState(
            cosine_solar_zenith_column=column_state.cosine_solar_zenith_column,
            soil=column_state.soil,
            ground=column_state.ground,
            canopy=canopy.state,
            pure_snow_ground_direct_albedo_column_band=(
                column_state.pure_snow_ground_direct_albedo_column_band
            ),
            pure_snow_ground_diffuse_albedo_column_band=(
                column_state.pure_snow_ground_diffuse_albedo_column_band
            ),
            no_black_carbon_ground_direct_albedo_column_band=(
                column_state.no_black_carbon_ground_direct_albedo_column_band
            ),
            no_black_carbon_ground_diffuse_albedo_column_band=(
                column_state.no_black_carbon_ground_diffuse_albedo_column_band
            ),
            no_organic_carbon_ground_direct_albedo_column_band=(
                column_state.no_organic_carbon_ground_direct_albedo_column_band
            ),
            no_organic_carbon_ground_diffuse_albedo_column_band=(
                column_state.no_organic_carbon_ground_diffuse_albedo_column_band
            ),
            no_dust_ground_direct_albedo_column_band=(
                column_state.no_dust_ground_direct_albedo_column_band
            ),
            no_dust_ground_diffuse_albedo_column_band=(
                column_state.no_dust_ground_diffuse_albedo_column_band
            ),
        ),
        direct_snow=columns.direct_snow,
        diffuse_snow=columns.diffuse_snow,
        diagnostics=SurfaceAlbedoDiagnostics(
            soil=columns.diagnostics.soil,
            direct_snow=columns.diagnostics.direct_snow,
            diffuse_snow=columns.diagnostics.diffuse_snow,
            ground=columns.diagnostics.ground,
            canopy=canopy.diagnostics,
        ),
    )


__all__ = [
    "CanopySurfaceAlbedoDiagnostics",
    "CanopySurfaceAlbedoInputs",
    "CanopySurfaceAlbedoParameters",
    "CanopySurfaceAlbedoResult",
    "CanopySurfaceAlbedoState",
    "CanopyTwoStreamDiagnostics",
    "CanopyTwoStreamInputs",
    "CanopyTwoStreamParameters",
    "CanopyTwoStreamResult",
    "CanopyTwoStreamState",
    "GroundSnowAlbedoDiagnostics",
    "GroundSnowAlbedoInputs",
    "GroundSnowAlbedoResult",
    "GroundSnowAlbedoState",
    "NUMBER_OF_SNOW_ABSORPTION_LEVELS",
    "NUMBER_OF_SHORTWAVE_BANDS",
    "SoilAlbedoDiagnostics",
    "SoilAlbedoInputs",
    "SoilAlbedoParameters",
    "SoilAlbedoResult",
    "SoilAlbedoState",
    "SurfaceAlbedoDiagnostics",
    "SurfaceAlbedoColumnDiagnostics",
    "SurfaceAlbedoColumnInputs",
    "SurfaceAlbedoColumnParameters",
    "SurfaceAlbedoColumnResult",
    "SurfaceAlbedoColumnState",
    "SurfaceAlbedoInputs",
    "SurfaceAlbedoParameters",
    "SurfaceAlbedoResult",
    "SurfaceAlbedoState",
    "canopy_surface_albedo_big_leaf",
    "canopy_two_stream_big_leaf",
    "soil_albedo",
    "surface_albedo_canopy_inputs",
    "surface_albedo_column_inputs",
    "surface_albedo_column_parameters",
    "surface_albedo_column_state",
    "surface_albedo_columns_configured",
    "surface_albedo_configured",
    "weight_ground_snow_albedo",
]
