
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.canopy_temperature_restart import soil_layer_geometry_20sl_8p5m
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.native_templates import native_template
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    RestartClock,
    SubgridTopology,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)
from clm_jax.io.snow_optics import configured_snow_optics_path, load_snow_optical_tables
from clm_jax.physics.solar import raw_cosine_solar_zenith
from clm_jax.physics.surface_albedo import (
    CanopySurfaceAlbedoParameters,
    CanopySurfaceAlbedoState,
    CanopyTwoStreamState,
    GroundSnowAlbedoState,
    SoilAlbedoParameters,
    SoilAlbedoState,
    SurfaceAlbedoInputs,
    SurfaceAlbedoParameters,
    SurfaceAlbedoState,
)
from clm_jax.trajectory import (
    advance_noleap_clock,
    noleap_calendar_day_from_clock,
    noleap_clock_from_restart,
)

_NUMBER_OF_SNOW_LAYERS = 12
_NUMBER_OF_SNOW_ABSORPTION_LEVELS = 13
_NUMBER_OF_SHORTWAVE_BANDS = 2

_SATURATED_SOIL_ALBEDO = np.asarray(
    (
        (0.25, 0.50),
        (0.23, 0.46),
        (0.21, 0.42),
        (0.20, 0.40),
        (0.19, 0.38),
        (0.18, 0.36),
        (0.17, 0.34),
        (0.16, 0.32),
        (0.15, 0.30),
        (0.14, 0.28),
        (0.13, 0.26),
        (0.12, 0.24),
        (0.11, 0.22),
        (0.10, 0.20),
        (0.09, 0.18),
        (0.08, 0.16),
        (0.07, 0.14),
        (0.06, 0.12),
        (0.05, 0.10),
        (0.04, 0.08),
    ),
    dtype=np.float64,
)
_DRY_SOIL_ALBEDO = np.asarray(
    (
        (0.36, 0.61),
        (0.34, 0.57),
        (0.32, 0.53),
        (0.31, 0.51),
        (0.30, 0.49),
        (0.29, 0.48),
        (0.28, 0.45),
        (0.27, 0.43),
        (0.26, 0.41),
        (0.25, 0.39),
        (0.24, 0.37),
        (0.23, 0.35),
        (0.22, 0.33),
        (0.20, 0.31),
        (0.18, 0.29),
        (0.16, 0.27),
        (0.14, 0.25),
        (0.12, 0.23),
        (0.10, 0.21),
        (0.08, 0.16),
    ),
    dtype=np.float64,
)

_RESTART_VARIABLES = (
    "SNLSNO",
    "T_GRND",
    "H2OSNO",
    "H2OSOI_LIQ",
    "H2OSOI_ICE",
    "frac_sno",
    "LAKE_ICEFRAC",
    "T_VEG",
    "FWET",
    "FCANSNO",
    "snw_rds",
    "elai",
    "esai",
    "mss_bcpho",
    "mss_bcphi",
    "mss_ocpho",
    "mss_ocphi",
    "mss_dst1",
    "mss_dst2",
    "mss_dst3",
    "mss_dst4",
    "coszen",
    "albd",
    "albi",
    "albgrd",
    "albgri",
    "albsod",
    "albsoi",
    "albsnd_hst",
    "albsni_hst",
    "tlai_z",
    "tsai_z",
    "ncan",
    "nrad",
    "fsun_z",
    "vcmaxcintsun",
    "vcmaxcintsha",
    "fabd",
    "fabi",
    "fabd_sun",
    "fabd_sha",
    "fabi_sun",
    "fabi_sha",
    "fabd_sun_z",
    "fabd_sha_z",
    "fabi_sun_z",
    "fabi_sha_z",
    "ftdd",
    "ftid",
    "ftii",
    "flx_absdv",
    "flx_absdn",
    "flx_absiv",
    "flx_absin",
)

_PARAMETER_VARIABLES = (
    "xl",
    "rholvis",
    "rholnir",
    "rhosvis",
    "rhosnir",
    "taulvis",
    "taulnir",
    "tausvis",
    "tausnir",
)


class SurfaceAlbedoNativeData(NamedTuple):

    topology: SubgridTopology
    clock: RestartClock
    inputs: SurfaceAlbedoInputs
    state: SurfaceAlbedoState
    parameters: SurfaceAlbedoParameters
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path
    snow_optics_path: Path


def _configured_input_path(case: CimeCase, key: str, override: str | Path | None) -> Path:
    if override is None:
        group = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        declared = group.get(key)
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError(f"clm_inparm lacks configured {key}")
        path = Path(case.expand(declared)).expanduser()
        if not path.is_absolute():
            path = case.root / path
    else:
        path = Path(override).expanduser()
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured {key} file does not exist: {path}")
    return path


def _finite_or_default(value: jnp.ndarray, default: float) -> jnp.ndarray:
    value = jnp.asarray(value, dtype=jnp.float64)
    return jnp.where(jnp.isfinite(value), value, jnp.float64(default))


def _first_call_next_shortwave_cosine(
    topology: SubgridTopology,
    clock: RestartClock,
) -> jnp.ndarray:
    call_clock = advance_noleap_clock(noleap_clock_from_restart(clock))
    next_shortwave_day = noleap_calendar_day_from_clock(call_clock)
    return raw_cosine_solar_zenith(
        next_shortwave_day,
        jnp.deg2rad(topology.grid.latitude_degrees),
        jnp.deg2rad(topology.grid.longitude_degrees),
    )


def _aerosol_concentration(
    values: dict[str, jnp.ndarray],
    snow_layer_count: jnp.ndarray,
    snow_liquid: jnp.ndarray,
    snow_ice: jnp.ndarray,
) -> jnp.ndarray:

    mass = jnp.stack(
        (
            values["mss_bcphi"],
            values["mss_bcpho"],
            values["mss_ocphi"],
            values["mss_ocpho"],
            values["mss_dst1"],
            values["mss_dst2"],
            values["mss_dst3"],
            values["mss_dst4"],
        ),
        axis=2,
    ).astype(jnp.float64)
    level_number = jnp.arange(
        -_NUMBER_OF_SNOW_LAYERS + 1,
        1,
        dtype=jnp.int32,
    )
    active = level_number[None, :] >= snow_layer_count[:, None] + jnp.int32(1)
    water = jnp.asarray(snow_liquid, dtype=jnp.float64) + jnp.asarray(
        snow_ice,
        dtype=jnp.float64,
    )
    safe_water = jnp.where(active, water, jnp.float64(1.0))
    concentration = mass / safe_water[:, :, None]
    return jnp.where(active[:, :, None], concentration, jnp.float64(0.0))


def _state(
    values: dict[str, jnp.ndarray],
    column_count: int,
    patch_count: int,
) -> SurfaceAlbedoState:
    column_band_cold = jnp.full(
        (column_count, _NUMBER_OF_SHORTWAVE_BANDS),
        jnp.float64(0.2),
    )
    patch_band_cold = jnp.full(
        (patch_count, _NUMBER_OF_SHORTWAVE_BANDS),
        jnp.float64(0.2),
    )

    def restart_real(name: str, default: float = 0.0) -> jnp.ndarray:
        return _finite_or_default(values[name], default)

    two_stream = CanopyTwoStreamState(
        direct_surface_albedo_patch_band=restart_real("albd", 0.2),
        diffuse_surface_albedo_patch_band=restart_real("albi", 0.2),
        snow_free_direct_surface_albedo_patch_band=patch_band_cold,
        snow_free_diffuse_surface_albedo_patch_band=patch_band_cold,
        direct_absorption_patch_band=restart_real("fabd"),
        direct_sunlit_absorption_patch_band=restart_real("fabd_sun"),
        direct_shaded_absorption_patch_band=restart_real("fabd_sha"),
        diffuse_absorption_patch_band=restart_real("fabi"),
        diffuse_sunlit_absorption_patch_band=restart_real("fabi_sun"),
        diffuse_shaded_absorption_patch_band=restart_real("fabi_sha"),
        direct_beam_transmission_patch_band=restart_real("ftdd", 1.0),
        direct_to_diffuse_transmission_patch_band=restart_real("ftid"),
        diffuse_transmission_patch_band=restart_real("ftii", 1.0),
        sunlit_fraction_patch_layer=restart_real("fsun_z"),
        direct_sunlit_absorption_patch_layer=restart_real("fabd_sun_z"),
        direct_shaded_absorption_patch_layer=restart_real("fabd_sha_z"),
        diffuse_sunlit_absorption_patch_layer=restart_real("fabi_sun_z"),
        diffuse_shaded_absorption_patch_layer=restart_real("fabi_sha_z"),
        sunlit_vcmax_scale_patch=restart_real("vcmaxcintsun", 1.0),
        shaded_vcmax_scale_patch=restart_real("vcmaxcintsha", 1.0),
    )
    return SurfaceAlbedoState(
        cosine_solar_zenith_column=restart_real("coszen"),
        soil=SoilAlbedoState(
            direct_column_band=restart_real("albsod", 0.2),
            diffuse_column_band=restart_real("albsoi", 0.2),
        ),
        ground=GroundSnowAlbedoState(
            ground_direct_albedo_column_band=restart_real("albgrd", 0.2),
            ground_diffuse_albedo_column_band=restart_real("albgri", 0.2),
            snow_direct_albedo_history_column_band=restart_real("albsnd_hst", 0.6),
            snow_diffuse_albedo_history_column_band=restart_real("albsni_hst", 0.6),
            direct_visible_absorption_column_level=restart_real("flx_absdv"),
            direct_nir_absorption_column_level=restart_real("flx_absdn"),
            diffuse_visible_absorption_column_level=restart_real("flx_absiv"),
            diffuse_nir_absorption_column_level=restart_real("flx_absin"),
        ),
        canopy=CanopySurfaceAlbedoState(
            two_stream=two_stream,
            leaf_area_patch_layer=restart_real("tlai_z"),
            stem_area_patch_layer=restart_real("tsai_z"),
            radiative_layer_count_patch=jnp.asarray(values["nrad"], dtype=jnp.int32),
            canopy_layer_count_patch=jnp.asarray(values["ncan"], dtype=jnp.int32),
        ),
        pure_snow_ground_direct_albedo_column_band=column_band_cold,
        pure_snow_ground_diffuse_albedo_column_band=column_band_cold,
        no_black_carbon_ground_direct_albedo_column_band=column_band_cold,
        no_black_carbon_ground_diffuse_albedo_column_band=column_band_cold,
        no_organic_carbon_ground_direct_albedo_column_band=column_band_cold,
        no_organic_carbon_ground_diffuse_albedo_column_band=column_band_cold,
        no_dust_ground_direct_albedo_column_band=column_band_cold,
        no_dust_ground_diffuse_albedo_column_band=column_band_cold,
    )


@native_template
def load_surface_albedo_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    snow_optics_path: str | Path | None = None,
) -> SurfaceAlbedoNativeData:

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"configured CLM restart does not exist: {restart}")
    surface_path = _configured_input_path(case, "fsurdat", surface_data_path)
    parameter_path = _configured_input_path(case, "paramfile", parameter_data_path)
    optics_path = (
        configured_snow_optics_path(case)
        if snow_optics_path is None
        else Path(snow_optics_path).expanduser().resolve()
    )
    if not optics_path.is_file():
        raise FileNotFoundError(f"configured snow-optics file does not exist: {optics_path}")

    dimensions = load_dimension_sizes(restart)
    expected_dimensions = {
        "levsno": _NUMBER_OF_SNOW_LAYERS,
        "levsno1": _NUMBER_OF_SNOW_ABSORPTION_LEVELS,
        "numrad": _NUMBER_OF_SHORTWAVE_BANDS,
        "levcan": 1,
    }
    for name, expected in expected_dimensions.items():
        if dimensions.get(name) != expected:
            raise ValueError(
                f"restart dimension {name} must be {expected}, got {dimensions.get(name)}"
            )

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    canopy_hydrology = read_case_namelist_group(
        case,
        "lnd_in",
        "clm_canopyhydrology_inparm",
    )
    if int(general.get("nlevsno", -1)) != _NUMBER_OF_SNOW_LAYERS:
        raise ValueError("SurfaceAlbedo requires the configured 12-layer snow column")
    if str(canopy_hydrology.get("snowveg_flag", "")).upper() != "ON_RAD":
        raise ValueError("SurfaceAlbedo adapter currently requires snowveg_flag='ON_RAD'")
    if bool(general.get("use_snicar_frc", False)):
        raise NotImplementedError(
            "configured adapter currently excludes SNICAR forcing diagnostics"
        )
    if bool(general.get("use_SSRE", False)):
        raise NotImplementedError("configured adapter currently excludes SSRE diagnostics")

    topology = load_subgrid_topology(restart)
    clock = load_restart_clock(restart)
    values = load_native_variables(restart, list(_RESTART_VARIABLES))
    surface = load_native_variables(surface_path, ["SOIL_COLOR"])
    parameter = load_native_variables(
        parameter_path,
        list(_PARAMETER_VARIABLES),
        dtype=np.float64,
    )
    column_count = int(topology.column.gridcell.shape[0])
    patch_count = int(topology.patch.gridcell.shape[0])

    snow_layer_count = jnp.asarray(values["SNLSNO"], dtype=jnp.int32)
    snow_liquid = _finite_or_default(values["H2OSOI_LIQ"], 0.0)[
        :, :_NUMBER_OF_SNOW_LAYERS
    ]
    snow_ice = _finite_or_default(values["H2OSOI_ICE"], 0.0)[
        :, :_NUMBER_OF_SNOW_LAYERS
    ]
    _, _, ground_thickness = soil_layer_geometry_20sl_8p5m()
    top_soil_thickness = np.float64(ground_thickness[0])
    top_soil_index = _NUMBER_OF_SNOW_LAYERS
    top_soil_liquid = _finite_or_default(values["H2OSOI_LIQ"], 0.0)[:, top_soil_index]
    top_soil_ice = _finite_or_default(values["H2OSOI_ICE"], 0.0)[:, top_soil_index]
    top_soil_volume = top_soil_liquid / (
        jnp.float64(top_soil_thickness) * jnp.float64(1000.0)
    ) + top_soil_ice / (jnp.float64(top_soil_thickness) * jnp.float64(917.0))

    grid_soil_color = jnp.asarray(surface["SOIL_COLOR"], dtype=jnp.int32)[
        topology.grid.latitude_index,
        topology.grid.longitude_index,
    ]
    landunit_column = jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
    landunit_patch = jnp.asarray(topology.patch.landunit_type, dtype=jnp.int32)
    nonurban_column = (landunit_column < jnp.int32(LANDUNIT_URBAN_MIN)) | (
        landunit_column > jnp.int32(LANDUNIT_URBAN_MAX)
    )
    nonurban_patch = (landunit_patch < jnp.int32(LANDUNIT_URBAN_MIN)) | (
        landunit_patch > jnp.int32(LANDUNIT_URBAN_MAX)
    )

    leaf_reflectance = jnp.stack(
        (parameter["rholvis"], parameter["rholnir"]),
        axis=1,
    )
    stem_reflectance = jnp.stack(
        (parameter["rhosvis"], parameter["rhosnir"]),
        axis=1,
    )
    leaf_transmittance = jnp.stack(
        (parameter["taulvis"], parameter["taulnir"]),
        axis=1,
    )
    stem_transmittance = jnp.stack(
        (parameter["tausvis"], parameter["tausnir"]),
        axis=1,
    )
    albice = general.get("albice", (0.80, 0.55))
    if not isinstance(albice, tuple) or len(albice) != 2:
        raise ValueError("clm_inparm.albice must contain visible and NIR values")
    lake_melt = general.get("lake_melt_icealb", (0.10, 0.10))
    if not isinstance(lake_melt, tuple) or len(lake_melt) != 2:
        raise ValueError("clm_inparm.lake_melt_icealb must contain visible and NIR values")

    inputs = SurfaceAlbedoInputs(
        nonurban_column=nonurban_column,
        nonurban_patch=nonurban_patch,
        column_gridcell=topology.column.gridcell,
        patch_gridcell=topology.patch.gridcell,
        patch_column=topology.patch.column,
        cosine_solar_zenith_gridcell=_first_call_next_shortwave_cosine(topology, clock),
        landunit_type_column=landunit_column,
        pft_type_patch=topology.patch.vegetation_type,
        snow_layer_count_column=snow_layer_count,
        ground_temperature_column=_finite_or_default(values["T_GRND"], 273.15),
        top_soil_water_volume_fraction_column=top_soil_volume,
        lake_ice_fraction_top_two_column=_finite_or_default(values["LAKE_ICEFRAC"], 0.0)[
            :, :2
        ],
        soil_color_column=grid_soil_color[topology.column.gridcell],
        snow_fraction_column=_finite_or_default(values["frac_sno"], 0.0),
        snow_water_equivalent_column=_finite_or_default(values["H2OSNO"], 0.0),
        snow_liquid_column_layer=snow_liquid,
        snow_ice_column_layer=snow_ice,
        snow_grain_radius_column_layer=_finite_or_default(values["snw_rds"], 0.0),
        aerosol_mass_concentration_column_layer_species=_aerosol_concentration(
            values,
            snow_layer_count,
            snow_liquid,
            snow_ice,
        ),
        leaf_orientation_pft=jnp.asarray(parameter["xl"], dtype=jnp.float64),
        leaf_reflectance_pft_band=leaf_reflectance,
        stem_reflectance_pft_band=stem_reflectance,
        leaf_transmittance_pft_band=leaf_transmittance,
        stem_transmittance_pft_band=stem_transmittance,
        vegetation_temperature_patch=_finite_or_default(values["T_VEG"], 273.15),
        wet_fraction_patch=_finite_or_default(values["FWET"], 0.0),
        canopy_snow_fraction_patch=_finite_or_default(values["FCANSNO"], 0.0),
        exposed_leaf_area_patch=_finite_or_default(values["elai"], 0.0),
        exposed_stem_area_patch=_finite_or_default(values["esai"], 0.0),
    )
    parameters = SurfaceAlbedoParameters(
        soil=SoilAlbedoParameters(
            saturated_soil_albedo_color_band=jnp.asarray(_SATURATED_SOIL_ALBEDO),
            dry_soil_albedo_color_band=jnp.asarray(_DRY_SOIL_ALBEDO),
            land_ice_albedo_band=jnp.asarray(albice, dtype=jnp.float64),
            frozen_lake_albedo_band=jnp.asarray((0.60, 0.40), dtype=jnp.float64),
            melting_lake_albedo_band=jnp.asarray(lake_melt, dtype=jnp.float64),
            lake_puddling=False,
        ),
        snow_optical_tables=load_snow_optical_tables(case, optics_path),
        canopy=CanopySurfaceAlbedoParameters(
            snow_scattering_albedo_band=jnp.asarray((0.8, 0.4), dtype=jnp.float64),
            snow_direct_upscatter=0.5,
            snow_diffuse_upscatter=0.5,
            nitrogen_extinction_coefficient=0.30,
            snow_free_diagnostic=False,
        ),
        subgrid_fluxes_enabled=bool(int(general.get("subgridflag", 1))),
    )
    return SurfaceAlbedoNativeData(
        topology=topology,
        clock=clock,
        inputs=inputs,
        state=_state(values, column_count, patch_count),
        parameters=parameters,
        restart_path=restart,
        surface_data_path=surface_path,
        parameter_data_path=parameter_path,
        snow_optics_path=optics_path,
    )


__all__ = [
    "SurfaceAlbedoNativeData",
    "load_surface_albedo_native_data",
]
