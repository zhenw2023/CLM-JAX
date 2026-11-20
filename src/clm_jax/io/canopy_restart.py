
from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_GLACIER,
    SubgridTopology,
    configured_land_restart,
    load_subgrid_topology,
)
from clm_jax.physics.aerosol import SnowAerosolState
from clm_jax.physics.canopy_hydrology import CanopyWaterState, CanopyWetFractions
from clm_jax.physics.canopy_step import (
    CanopyHydrologyState,
    CanopyStructure,
)
from clm_jax.physics.snow import SnowColumnState


class CanopyHydrologyRestartData(NamedTuple):

    topology: SubgridTopology
    structure: CanopyStructure
    state: CanopyHydrologyState
    previous_snow_drainage_flux_column: jnp.ndarray


_RESTART_VARIABLES = (
    "SNLSNO",
    "DZSNO",
    "ZSNO",
    "ZISNO",
    "T_SOISNO",
    "T_GRND",
    "INT_SNOW",
    "H2OSFC",
    "H2OSNO",
    "H2OSOI_LIQ",
    "H2OSOI_ICE",
    "H2OCAN",
    "SNOCAN",
    "LIQCAN",
    "SNOUNLOAD",
    "FH2OSFC",
    "SNOW_DEPTH",
    "frac_sno_eff",
    "frac_sno",
    "FWET",
    "FCANSNO",
    "snw_rds",
    "FRAC_VEG_NOSNO_ALB",
    "elai",
    "esai",
    "qflx_snow_drain",
    "mss_bcpho",
    "mss_bcphi",
    "mss_ocpho",
    "mss_ocphi",
    "mss_dst1",
    "mss_dst2",
    "mss_dst3",
    "mss_dst4",
)


def _case_input_path(case: CimeCase, group: str, key: str) -> Path:
    values = read_case_namelist_group(case, "lnd_in", group)
    value = values[key]
    if not isinstance(value, str):
        raise TypeError(f"{group}.{key} must be a path string")
    path = Path(case.expand(value)).expanduser()
    if not path.is_absolute():
        path = case.root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured {group}.{key} does not exist: {path}")
    return path


def _microtopography_sigma(slope: jnp.ndarray) -> jnp.ndarray:
    values = np.maximum(np.asarray(slope, dtype=np.float64), np.float64(0.2))
    bases = values + math.pow(0.4, -1.0 / 3.0)
    sigma = np.fromiter(
        (math.pow(float(value), -3.0) for value in bases.flat),
        dtype=np.float64,
        count=bases.size,
    ).reshape(bases.shape)
    return jnp.asarray(sigma, dtype=jnp.float64)


def _column_structure(
    case: CimeCase,
    topology: SubgridTopology,
    restart_values: dict[str, jnp.ndarray],
) -> CanopyStructure:
    surface_path = _case_input_path(case, "clm_inparm", "fsurdat")
    surface = load_native_variables(
        surface_path,
        ["SLOPE", "STD_ELEV", "GLACIER_REGION"],
    )
    longitude_index = topology.grid.longitude_index
    latitude_index = topology.grid.latitude_index
    grid_slope = jnp.asarray(surface["SLOPE"], dtype=jnp.float64)[
        latitude_index,
        longitude_index,
    ]
    grid_elevation_standard_deviation = jnp.asarray(
        surface["STD_ELEV"],
        dtype=jnp.float64,
    )[latitude_index, longitude_index]
    grid_glacier_region = jnp.asarray(surface["GLACIER_REGION"], dtype=jnp.int32)[
        latitude_index,
        longitude_index,
    ]

    microtopography_sigma = _microtopography_sigma(grid_slope[topology.column.gridcell])

    standard_deviation = grid_elevation_standard_deviation[topology.column.gridcell]
    snow_cover_shape = jnp.float64(200.0) / jnp.maximum(
        jnp.float64(10.0),
        standard_deviation,
    )

    glacier_values = read_case_namelist_group(
        case,
        "lnd_in",
        "clm_glacier_behavior",
    )
    behavior_value = glacier_values["glacier_region_behavior"]
    behaviors = (
        behavior_value if isinstance(behavior_value, tuple) else (behavior_value,)
    )
    if not all(isinstance(value, str) for value in behaviors):
        raise TypeError("clm_glacier_behavior.glacier_region_behavior must contain strings")
    maximum_region = int(jnp.max(grid_glacier_region))
    minimum_region = int(jnp.min(grid_glacier_region))
    if minimum_region < 0 or maximum_region >= len(behaviors):
        raise ValueError(
            "surface GLACIER_REGION is outside the configured glacier behavior array: "
            f"[{minimum_region}, {maximum_region}] versus {len(behaviors)} entries"
        )
    allow_multiple_by_region = jnp.asarray(
        [str(value).strip().lower() != "single_at_atm_topo" for value in behaviors],
        dtype=jnp.bool_,
    )
    column_allows_multiple_glacier_classes = allow_multiple_by_region[
        grid_glacier_region[topology.column.gridcell]
    ]
    general_values = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    glacier_snow_cover_shape = jnp.float64(general_values["n_melt_glcmec"])
    use_glacier_shape = (
        topology.column.landunit_type == LANDUNIT_GLACIER
    ) & column_allows_multiple_glacier_classes
    snow_cover_shape = jnp.where(
        use_glacier_shape,
        glacier_snow_cover_shape,
        snow_cover_shape,
    )

    exposed_leaf_area = jnp.asarray(restart_values["elai"], dtype=jnp.float64)
    albedo_exposed = jnp.asarray(
        restart_values["FRAC_VEG_NOSNO_ALB"],
        dtype=jnp.int32,
    )
    return CanopyStructure(
        vegetation_exposed_patch=jnp.where(
            topology.patch.active,
            albedo_exposed,
            jnp.int32(0),
        ),
        maximum_dew_patch=jnp.full_like(exposed_leaf_area, jnp.float64(0.1)),
        exposed_leaf_area_index_patch=exposed_leaf_area,
        exposed_stem_area_index_patch=jnp.asarray(
            restart_values["esai"],
            dtype=jnp.float64,
        ),
        microtopography_sigma_column=microtopography_sigma,
        ground_temperature_column=jnp.asarray(
            restart_values["T_GRND"],
            dtype=jnp.float64,
        ),
        snow_cover_shape_column=snow_cover_shape,
    )


def _aerosol_state(
    restart_values: dict[str, jnp.ndarray],
    layer_count: jnp.ndarray,
    number_of_snow_levels: int,
) -> SnowAerosolState:
    level_number = jnp.arange(
        -number_of_snow_levels + 1,
        1,
        dtype=jnp.int32,
    )
    active_layer = level_number[None, :] >= layer_count[:, None] + jnp.int32(1)

    def primary(name: str) -> jnp.ndarray:
        value = jnp.asarray(restart_values[name], dtype=jnp.float64)
        return jnp.where(active_layer, value, jnp.float64(0.0))

    black_carbon_phobic = primary("mss_bcpho")
    black_carbon_philic = primary("mss_bcphi")
    organic_carbon_phobic = primary("mss_ocpho")
    organic_carbon_philic = primary("mss_ocphi")
    dust_1 = primary("mss_dst1")
    dust_2 = primary("mss_dst2")
    dust_3 = primary("mss_dst3")
    dust_4 = primary("mss_dst4")
    black_carbon_total = black_carbon_phobic + black_carbon_philic
    organic_carbon_total = organic_carbon_phobic + organic_carbon_philic
    dust_total = dust_1 + dust_2 + dust_3 + dust_4

    def column_sum(value: jnp.ndarray) -> jnp.ndarray:
        result = jnp.zeros(value.shape[0], dtype=jnp.float64)
        for level in range(number_of_snow_levels):
            result = result + value[:, level]
        return result

    row = jnp.arange(layer_count.shape[0], dtype=jnp.int32)
    top_index = jnp.clip(
        layer_count + jnp.int32(number_of_snow_levels),
        jnp.int32(0),
        jnp.int32(number_of_snow_levels - 1),
    )
    has_snow = layer_count < jnp.int32(0)

    def top(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(has_snow, value[row, top_index], jnp.float64(0.0))

    return SnowAerosolState(
        hydrophobic_black_carbon_layer=black_carbon_phobic,
        hydrophilic_black_carbon_layer=black_carbon_philic,
        total_black_carbon_layer=black_carbon_total,
        black_carbon_column=column_sum(black_carbon_total),
        black_carbon_top=top(black_carbon_total),
        hydrophobic_organic_carbon_layer=organic_carbon_phobic,
        hydrophilic_organic_carbon_layer=organic_carbon_philic,
        total_organic_carbon_layer=organic_carbon_total,
        organic_carbon_column=column_sum(organic_carbon_total),
        organic_carbon_top=top(organic_carbon_total),
        dust_species_1_layer=dust_1,
        dust_species_2_layer=dust_2,
        dust_species_3_layer=dust_3,
        dust_species_4_layer=dust_4,
        total_dust_layer=dust_total,
        dust_column=column_sum(dust_total),
        dust_top=top(dust_total),
    )


def load_canopy_hydrology_restart(
    case: CimeCase,
    restart_path: str | Path | None = None,
) -> CanopyHydrologyRestartData:

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")

    dimensions = load_dimension_sizes(restart)
    number_of_snow_levels = dimensions["levsno"]
    general_values = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    configured_snow_levels = int(general_values["nlevsno"])
    if configured_snow_levels != number_of_snow_levels:
        raise ValueError(
            "restart and case disagree on nlevsno: "
            f"{number_of_snow_levels} versus {configured_snow_levels}"
        )
    if dimensions["levtot"] <= number_of_snow_levels:
        raise ValueError("restart levtot does not contain the first soil layer")

    topology = load_subgrid_topology(restart)
    values = load_native_variables(
        restart,
        list(_RESTART_VARIABLES),
        indexers={"levtot": slice(0, number_of_snow_levels + 1)},
    )
    layer_count = jnp.asarray(values["SNLSNO"], dtype=jnp.int32)
    layer_ice = jnp.asarray(values["H2OSOI_ICE"], dtype=jnp.float64)[
        :, :number_of_snow_levels
    ]
    layer_liquid = jnp.asarray(values["H2OSOI_LIQ"], dtype=jnp.float64)[
        :, :number_of_snow_levels
    ]
    level_number = jnp.arange(
        -number_of_snow_levels + 1,
        1,
        dtype=jnp.int32,
    )
    active_layer = level_number[None, :] >= layer_count[:, None] + jnp.int32(1)
    layer_water = layer_ice + layer_liquid
    safe_layer_water = jnp.where(active_layer, layer_water, jnp.float64(1.0))
    old_ice_fraction = jnp.where(
        active_layer,
        layer_ice / safe_layer_water,
        jnp.float64(0.0),
    )

    interface_depth = jnp.concatenate(
        (
            jnp.asarray(values["ZISNO"], dtype=jnp.float64),
            jnp.zeros((layer_count.shape[0], 1), dtype=jnp.float64),
        ),
        axis=1,
    )
    exposed = jnp.asarray(values["FRAC_VEG_NOSNO_ALB"], dtype=jnp.int32)
    leaf_area = jnp.asarray(values["elai"], dtype=jnp.float64)
    stem_area = jnp.asarray(values["esai"], dtype=jnp.float64)
    wet_fraction = jnp.asarray(values["FWET"], dtype=jnp.float64)
    vegetation_area = leaf_area + stem_area
    dry_leaf_fraction = jnp.where(
        (exposed == jnp.int32(1)) & (vegetation_area > jnp.float64(0.0)),
        (jnp.float64(1.0) - wet_fraction)
        * leaf_area
        / jnp.where(vegetation_area > 0.0, vegetation_area, jnp.float64(1.0)),
        jnp.float64(0.0),
    )

    state = CanopyHydrologyState(
        canopy_water=CanopyWaterState(
            total_water=jnp.asarray(values["H2OCAN"], dtype=jnp.float64),
            snow=jnp.asarray(values["SNOCAN"], dtype=jnp.float64),
            liquid=jnp.asarray(values["LIQCAN"], dtype=jnp.float64),
            snow_unloaded=jnp.asarray(values["SNOUNLOAD"], dtype=jnp.float64),
        ),
        canopy_wet_fractions=CanopyWetFractions(
            wet=wet_fraction,
            snow_covered=jnp.asarray(values["FCANSNO"], dtype=jnp.float64),
            dry_leaf=dry_leaf_fraction,
        ),
        snow=SnowColumnState(
            layer_count=layer_count,
            water_equivalent=jnp.asarray(values["H2OSNO"], dtype=jnp.float64),
            depth=jnp.asarray(values["SNOW_DEPTH"], dtype=jnp.float64),
            integrated_snow=jnp.asarray(values["INT_SNOW"], dtype=jnp.float64),
            snow_fraction=jnp.asarray(values["frac_sno"], dtype=jnp.float64),
            effective_snow_fraction=jnp.asarray(
                values["frac_sno_eff"],
                dtype=jnp.float64,
            ),
            layer_thickness=jnp.asarray(values["DZSNO"], dtype=jnp.float64),
            layer_depth=jnp.asarray(values["ZSNO"], dtype=jnp.float64),
            layer_interface_depth=interface_depth,
            layer_temperature=jnp.asarray(
                values["T_SOISNO"],
                dtype=jnp.float64,
            )[:, :number_of_snow_levels],
            layer_ice=layer_ice,
            layer_liquid=layer_liquid,
            old_ice_fraction=old_ice_fraction,
            snow_grain_radius=jnp.asarray(values["snw_rds"], dtype=jnp.float64),
        ),
        surface_water_column=jnp.asarray(values["H2OSFC"], dtype=jnp.float64),
        top_soil_liquid_column=jnp.asarray(
            values["H2OSOI_LIQ"],
            dtype=jnp.float64,
        )[:, number_of_snow_levels],
        surface_water_fraction_column=jnp.asarray(
            values["FH2OSFC"],
            dtype=jnp.float64,
        ),
        surface_water_fraction_without_snow_column=jnp.asarray(
            values["FH2OSFC"],
            dtype=jnp.float64,
        ),
        snow_aerosol=_aerosol_state(
            values,
            layer_count,
            number_of_snow_levels,
        ),
    )
    return CanopyHydrologyRestartData(
        topology=topology,
        structure=_column_structure(case, topology, values),
        state=state,
        previous_snow_drainage_flux_column=jnp.asarray(
            values["qflx_snow_drain"],
            dtype=jnp.float64,
        ),
    )
