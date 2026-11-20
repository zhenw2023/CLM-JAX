
from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from clm_jax.compilation import jit_discrete_parameters
from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.restart import (
    COLUMN_URBAN_IMPERVIOUS_ROAD,
    COLUMN_URBAN_PERVIOUS_ROAD,
    COLUMN_URBAN_ROOF,
    COLUMN_URBAN_SHADE_WALL,
    COLUMN_URBAN_SUN_WALL,
    LANDUNIT_CROP,
    LANDUNIT_LAKE,
    LANDUNIT_SOIL,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
)
from clm_jax.physics.cn_precision_control import (
    get_cn_precision_control_state_field,
)
from clm_jax.physics.cn_zero import get_cn_flux_field
from clm_jax.physics.land_to_atmosphere import _aggregate_to_gridcell
from clm_jax.physics.numerics import fortran_add, fortran_divide, fortran_multiply
from clm_jax.physics.snow_water import NUMBER_OF_SNOW_LEVELS

_COMPILED_HISTORY_AGGREGATION = jax.jit(
    _aggregate_to_gridcell,
    static_argnames=("number_of_gridcells", "maximum_sources_per_gridcell"),
)

H1_HISTORY_VARIABLES = (
    "AR",
    "BTRAN2",
    "CPOOL",
    "CROPPROD1C",
    "EFLX_LH_TOT",
    "ERRH2O",
    "ERRH2OSNO",
    "ERRSEB",
    "ERRSOI",
    "FGR",
    "FIRA",
    "FLDS",
    "FPSN",
    "FSA",
    "FSDS",
    "FSH",
    "GPP",
    "GRAINC",
    "GRAINC_TO_FOOD",
    "GRAINC_TO_SEED",
    "GRAINN",
    "H2OCAN",
    "H2OSFC",
    "H2OSNO",
    "H2OSOI",
    "HR",
    "NBP",
    "NEE",
    "NEP",
    "NPOOL",
    "NPP",
    "PBOT",
    "PLANT_NDEMAND",
    "QBOT",
    "QDRAI",
    "QFLX_EVAP_TOT",
    "QIRRIG",
    "QOVER",
    "QRUNOFF",
    "QSOIL",
    "QVEGE",
    "QVEGT",
    "RAIN",
    "RAIN_FROM_ATM",
    "SMINN",
    "SNOW",
    "SNOW_FROM_ATM",
    "SOILICE",
    "SOILLIQ",
    "TBOT",
    "TG",
    "TLAI",
    "TOTECOSYSC",
    "TOTLITC",
    "TOTSOMC",
    "TOTVEGC",
    "TSA",
    "TSOI",
    "TWS",
    "WIND",
    "ZWT",
)

H2_HISTORY_VARIABLES = (
    "BTRAN2",
    "CPOOL",
    "FPSN",
    "GPP",
    "GRAINC",
    "GRAINC_TO_FOOD",
    "GRAINC_TO_SEED",
    "GRAINN",
    "NPOOL",
    "NPP",
    "PLANT_NDEMAND",
    "QIRRIG",
    "SMINN",
    "TLAI",
)


class EarlyHistoryFields(NamedTuple):

    plant_nitrogen_demand_patch: jnp.ndarray
    crop_product_1year_gridcell: jnp.ndarray


class InstantaneousHistorySnapshot(NamedTuple):

    gridcell: dict[str, jnp.ndarray]
    pft: dict[str, jnp.ndarray]


def project_early_history_fields(ecosystem_state: Any) -> EarlyHistoryFields:

    update3 = ecosystem_state.state_update3_precision
    products = update3.fire_flux.fire_area.wood_products
    update2 = products.state_update2h_precision.state_update2
    transport = update2.gap_mortality.litter_vertical_transport
    guarded = transport.soil_n_state_update.second_guardrail.first_guardrail
    demand = guarded.ecosystem.pre_decomposition.prefix.plant_nutrient_demand
    return EarlyHistoryFields(
        plant_nitrogen_demand_patch=(demand.direct_fluxes.plant_nitrogen_demand_patch),
        crop_product_1year_gridcell=(products.wood_products.carbon.crop_product_1year_gridcell),
    )


def _urban_scale(topology: Any, *, state_scale: bool) -> jnp.ndarray:

    subgrid = topology.subgrid
    column = subgrid.column
    landunit = jnp.asarray(column.landunit, dtype=jnp.int32)
    landunit_type = jnp.asarray(column.landunit_type, dtype=jnp.int32)
    column_type = jnp.asarray(column.column_type, dtype=jnp.int32)
    ratio = jnp.asarray(
        topology.canyon_height_width_ratio_landunit,
        dtype=jnp.float64,
    )[landunit]
    urban = (landunit_type >= jnp.int32(LANDUNIT_URBAN_MIN)) & (
        landunit_type <= jnp.int32(LANDUNIT_URBAN_MAX)
    )
    wall = (column_type == jnp.int32(COLUMN_URBAN_SUN_WALL)) | (
        column_type == jnp.int32(COLUMN_URBAN_SHADE_WALL)
    )
    road = (column_type == jnp.int32(COLUMN_URBAN_PERVIOUS_ROAD)) | (
        column_type == jnp.int32(COLUMN_URBAN_IMPERVIOUS_ROAD)
    )
    roof = column_type == jnp.int32(COLUMN_URBAN_ROOF)
    denominator = fortran_add(
        fortran_multiply(jnp.float64(2.0), ratio),
        jnp.float64(1.0),
    )
    wall_scale = fortran_multiply(jnp.float64(3.0), ratio)
    road_scale = jnp.full(ratio.shape, jnp.float64(3.0))
    if state_scale:
        wall_scale = fortran_divide(wall_scale, denominator)
        road_scale = fortran_divide(road_scale, denominator)
    selected = jnp.where(
        wall,
        wall_scale,
        jnp.where(road, road_scale, jnp.where(roof, jnp.float64(1.0), jnp.nan)),
    )
    return jnp.where(urban, selected, jnp.float64(1.0))


def _source_scale(
    topology: Any,
    *,
    source: str,
    c2l: str,
    l2g: str,
) -> jnp.ndarray:
    subgrid = topology.subgrid
    if source == "patch":
        column_index = jnp.asarray(subgrid.patch.column, dtype=jnp.int32)
        landunit_type = jnp.asarray(subgrid.patch.landunit_type, dtype=jnp.int32)
        source_count = subgrid.patch.gridcell.shape[0]
    elif source == "column":
        column_index = None
        landunit_type = jnp.asarray(subgrid.column.landunit_type, dtype=jnp.int32)
        source_count = subgrid.column.gridcell.shape[0]
    else:
        raise ValueError(f"unsupported history source: {source}")

    if c2l == "unity":
        scale = jnp.ones(source_count, dtype=jnp.float64)
    elif c2l in {"urbanf", "urbans"}:
        column_scale = _urban_scale(topology, state_scale=c2l == "urbans")
        scale = column_scale if column_index is None else column_scale[column_index]
    else:
        raise ValueError(f"unsupported history c2l scale: {c2l}")

    if l2g == "veg":
        vegetation = (landunit_type == jnp.int32(LANDUNIT_SOIL)) | (
            landunit_type == jnp.int32(LANDUNIT_CROP)
        )
        scale = jnp.where(
            vegetation,
            scale,
            jnp.float64(CLM_SPECIAL_VALUE),
        )
    elif l2g != "unity":
        raise ValueError(f"unsupported history l2g scale: {l2g}")
    return scale


@cache
def _compiled_named_aggregation(source, c2l, l2g, maximum):
    from clm_jax.physics.land_to_atmosphere import LandToAtmosphereParameters

    parameters = LandToAtmosphereParameters(
        maximum_patches_per_gridcell=maximum,
        maximum_columns_per_gridcell=maximum,
    )
    return jax.jit(lambda fields, topology: _aggregate_named_fields_reference(
        fields, topology, parameters, source=source, c2l=c2l, l2g=l2g,
    ))


def _aggregate_named_fields(
    fields, topology, parameters, *, source, c2l="unity", l2g="unity",
):
    if jax.default_backend() == "gpu" and fields:
        if source not in {"patch", "column"}:
            raise ValueError(f"unsupported history source: {source}")
        maximum = int(parameters.maximum_patches_per_gridcell if source == "patch"
                      else parameters.maximum_columns_per_gridcell)
        return _compiled_named_aggregation(source, c2l, l2g, maximum)(fields, topology)
    return _aggregate_named_fields_reference(
        fields, topology, parameters, source=source, c2l=c2l, l2g=l2g,
    )


def _aggregate_named_fields_reference(
    fields: Mapping[str, jnp.ndarray],
    topology: Any,
    parameters: Any,
    *,
    source: str,
    c2l: str = "unity",
    l2g: str = "unity",
) -> dict[str, jnp.ndarray]:

    if not fields:
        return {}
    subgrid = topology.subgrid
    if source == "patch":
        source_topology = subgrid.patch
        maximum = int(parameters.maximum_patches_per_gridcell)
    elif source == "column":
        source_topology = subgrid.column
        maximum = int(parameters.maximum_columns_per_gridcell)
    else:
        raise ValueError(f"unsupported history source: {source}")
    source_count = source_topology.gridcell.shape[0]
    names = tuple(fields)
    trailing_shapes: dict[str, tuple[int, ...]] = {}
    widths: dict[str, int] = {}
    flattened = []
    for name in names:
        value = jnp.asarray(fields[name], dtype=jnp.float64)
        if value.ndim not in (1, 2) or value.shape[0] != source_count:
            raise ValueError(
                f"history field {name} must have leading {source} size "
                f"{source_count}; got {value.shape}"
            )
        trailing_shapes[name] = value.shape[1:]
        widths[name] = 1 if value.ndim == 1 else value.shape[1]
        flattened.append(jnp.reshape(value, (source_count, widths[name])))
    matrix = jnp.concatenate(tuple(flattened), axis=1)
    aggregated = _COMPILED_HISTORY_AGGREGATION(
        matrix,
        source_topology.gridcell,
        source_topology.weight_on_gridcell,
        source_topology.active,
        _source_scale(topology, source=source, c2l=c2l, l2g=l2g),
        number_of_gridcells=subgrid.grid.longitude_degrees.shape[0],
        maximum_sources_per_gridcell=maximum,
        special_value=float(CLM_SPECIAL_VALUE),
    )
    result: dict[str, jnp.ndarray] = {}
    offset = 0
    for name in names:
        width = widths[name]
        selected = aggregated[:, offset : offset + width]
        result[name] = jnp.reshape(
            selected,
            (selected.shape[0], *trailing_shapes[name]),
        )
        offset += width
    return result


def _set_patch_landunits(
    values: jnp.ndarray,
    subgrid: Any,
    *,
    lake: float | None = None,
    urban: float | None = None,
) -> jnp.ndarray:
    result = jnp.asarray(values, dtype=jnp.float64)
    landunit_type = jnp.asarray(subgrid.patch.landunit_type, dtype=jnp.int32)
    if lake is not None:
        result = jnp.where(
            landunit_type == jnp.int32(LANDUNIT_LAKE),
            jnp.float64(lake),
            result,
        )
    if urban is not None:
        is_urban = (landunit_type >= jnp.int32(LANDUNIT_URBAN_MIN)) & (
            landunit_type <= jnp.int32(LANDUNIT_URBAN_MAX)
        )
        result = jnp.where(is_urban, jnp.float64(urban), result)
    return result


def _compose_instantaneous_history_reference(
    *,
    topology: Any,
    forcing: Any,
    physical: Any,
    post_temperature: Any,
    lake: Any,
    drainage: Any,
    post_drainage: Any,
    balance: Any,
    balance_radiation: Any,
    land_to_atmosphere_call: Any,
    land_to_atmosphere: Any,
    early: EarlyHistoryFields,
) -> InstantaneousHistorySnapshot:

    subgrid = topology
    atmospheric = physical.atmospheric.column_forcing
    canopy_flux = physical.canopy_flux
    canopy_ecosystem = canopy_flux.ecosystem
    post_flux = post_temperature.soil_fluxes.state
    lake_state = lake.hydrology.state
    water = drainage.drainage.state
    balance_state = balance.balance.state
    biogeochemistry = post_drainage.state.biogeochemistry
    vegetation = biogeochemistry.vegetation
    cn_fluxes = biogeochemistry.cn_fluxes
    summaries = post_drainage.state.state_summaries
    vegetation_structure = post_drainage.state.vegetation_structure
    grid_fluxes = post_drainage.gridcell_fluxes
    aggregation_topology = land_to_atmosphere_call.topology
    aggregation_parameters = land_to_atmosphere_call.parameters

    def aggregate(
        fields: Mapping[str, jnp.ndarray],
        *,
        source: str,
        c2l: str = "unity",
        l2g: str = "unity",
    ) -> dict[str, jnp.ndarray]:
        return _aggregate_named_fields(
            fields,
            aggregation_topology,
            aggregation_parameters,
            source=source,
            c2l=c2l,
            l2g=l2g,
        )

    photosynthesis = jnp.where(
        canopy_flux.dust.active_patch,
        canopy_flux.state.active.photosynthesis_total.photosynthesis_patch,
        jnp.float64(0.0),
    )
    photosynthesis = _set_patch_landunits(
        photosynthesis,
        subgrid,
        lake=0.0,
        urban=0.0,
    )
    root_wetness = _set_patch_landunits(
        canopy_ecosystem.root_zone_wetness_patch,
        subgrid,
        lake=float(CLM_SPECIAL_VALUE),
        urban=float(CLM_SPECIAL_VALUE),
    )
    canopy_water = _set_patch_landunits(
        canopy_flux.state.active.postprocess.canopy_water_patch,
        subgrid,
        lake=0.0,
    )
    vegetation_transpiration = _set_patch_landunits(
        post_temperature.soil_fluxes_inputs.vegetation_transpiration_patch,
        subgrid,
        lake=0.0,
    )
    canopy_evaporation = _set_patch_landunits(
        post_flux.water.canopy_evaporation_patch,
        subgrid,
        lake=0.0,
    )

    patch_pools = {
        "AR": get_cn_flux_field(cn_fluxes, "vegetation_carbon.ar_patch"),
        "CPOOL": get_cn_precision_control_state_field(
            vegetation,
            "vegetation_carbon.cpool_patch",
        ),
        "GPP": get_cn_flux_field(cn_fluxes, "vegetation_carbon.gpp_patch"),
        "GRAINC": get_cn_precision_control_state_field(
            vegetation,
            "vegetation_carbon.grainc_patch",
        ),
        "GRAINC_TO_FOOD": get_cn_flux_field(
            cn_fluxes,
            "vegetation_carbon.grainc_to_food_patch",
        ),
        "GRAINC_TO_SEED": get_cn_flux_field(
            cn_fluxes,
            "vegetation_carbon.grainc_to_seed_patch",
        ),
        "GRAINN": get_cn_precision_control_state_field(
            vegetation,
            "vegetation_nitrogen.grainn_patch",
        ),
        "NPOOL": get_cn_precision_control_state_field(
            vegetation,
            "vegetation_nitrogen.npool_patch",
        ),
        "NPP": get_cn_flux_field(cn_fluxes, "vegetation_carbon.npp_patch"),
        "PLANT_NDEMAND": early.plant_nitrogen_demand_patch,
        "QIRRIG": physical.canopy.irrigation.fluxes.applied_patch,
        "TLAI": vegetation_structure.total_leaf_area_index_patch,
        "TOTVEGC": summaries.total_vegetation_carbon_patch,
    }

    gridcell: dict[str, jnp.ndarray] = {
        "CROPPROD1C": early.crop_product_1year_gridcell,
        "FSDS": forcing.shortwave_down,
        "NBP": grid_fluxes.net_biome_production,
        "NEE": grid_fluxes.net_ecosystem_exchange,
        "RAIN_FROM_ATM": forcing.rain,
        "SNOW_FROM_ATM": forcing.snow,
        "TWS": land_to_atmosphere.state.total_water_storage_gridcell,
        "WIND": forcing.wind_speed,
    }
    gridcell.update(
        aggregate(
            {
                **patch_pools,
                "BTRAN2": root_wetness,
                "ERRSEB": balance_state.surface_energy_balance_error_patch,
                "FPSN": photosynthesis,
                "H2OCAN": canopy_water,
                "TSA": land_to_atmosphere_call.inputs.reference_temperature_2m_patch,
            },
            source="patch",
        )
    )
    gridcell.update(
        aggregate(
            {
                "EFLX_LH_TOT": post_flux.energy.total_latent_heat_patch,
                "FGR": lake_state.energy_flux.soil_heat_flux_ground_patch,
                "FIRA": post_flux.energy.net_longwave_patch,
                "FSA": balance_radiation.absorbed_shortwave_patch,
                "FSH": lake_state.energy_flux.sensible_heat_total_patch,
                "QFLX_EVAP_TOT": post_flux.water.total_evaporation_patch,
                "QSOIL": post_flux.water.soil_evaporation_patch,
                "QVEGE": canopy_evaporation,
                "QVEGT": vegetation_transpiration,
            },
            source="patch",
            c2l="urbanf",
        )
    )
    gridcell.update(
        aggregate(
            {
                "ERRH2O": balance_state.water_balance_error_column,
                "ERRSOI": post_flux.energy.soil_energy_error_column,
                "FLDS": atmospheric.longwave_down,
                "H2OSFC": water.surface_water_column,
                "HR": get_cn_flux_field(cn_fluxes, "soil_carbon.hr_col"),
                "NEP": get_cn_flux_field(cn_fluxes, "vegetation_carbon.nep_col"),
                "PBOT": atmospheric.pressure,
                "QBOT": atmospheric.specific_humidity,
                "RAIN": atmospheric.rain,
                "SMINN": summaries.mineral_nitrogen_column,
                "SNOW": atmospheric.snow,
                "TBOT": atmospheric.temperature,
                "TOTECOSYSC": summaries.total_ecosystem_carbon_column,
                "TOTLITC": summaries.total_litter_carbon_column,
                "TOTSOMC": summaries.total_soil_carbon_column,
            },
            source="column",
        )
    )
    gridcell.update(
        aggregate(
            {
                "ERRH2OSNO": balance_state.snow_balance_error_column,
                "H2OSNO": lake_state.water.snow_water_equivalent_column,
                "QDRAI": water.subsurface_drainage_flux_column,
                "QOVER": water.surface_runoff_flux_column,
                "QRUNOFF": land_to_atmosphere.state.total_runoff_column,
            },
            source="column",
            c2l="urbanf",
        )
    )
    gridcell.update(
        aggregate(
            {"TG": lake.inputs.ground_temperature_column},
            source="column",
            c2l="urbans",
        )
    )
    gridcell.update(
        aggregate(
            {
                "H2OSOI": water.volumetric_water_column_level[:, :20],
                "SOILICE": water.soil_snow_ice_water_column_level[
                    :, NUMBER_OF_SNOW_LEVELS : NUMBER_OF_SNOW_LEVELS + 20
                ],
                "SOILLIQ": water.soil_snow_liquid_water_column_level[
                    :, NUMBER_OF_SNOW_LEVELS : NUMBER_OF_SNOW_LEVELS + 20
                ],
                "TSOI": lake_state.temperature.soil_snow_temperature_column_level[
                    :, NUMBER_OF_SNOW_LEVELS:
                ],
                "ZWT": water.water_table_depth_column,
            },
            source="column",
            l2g="veg",
        )
    )

    missing = set(H1_HISTORY_VARIABLES).difference(gridcell)
    extra = set(gridcell).difference(H1_HISTORY_VARIABLES)
    if missing or extra:
        raise RuntimeError(
            f"invalid h1 history schema: missing={sorted(missing)}, extra={sorted(extra)}"
        )

    pft = {
        "BTRAN2": root_wetness,
        "CPOOL": patch_pools["CPOOL"],
        "FPSN": photosynthesis,
        "GPP": patch_pools["GPP"],
        "GRAINC": patch_pools["GRAINC"],
        "GRAINC_TO_FOOD": patch_pools["GRAINC_TO_FOOD"],
        "GRAINC_TO_SEED": patch_pools["GRAINC_TO_SEED"],
        "GRAINN": patch_pools["GRAINN"],
        "NPOOL": patch_pools["NPOOL"],
        "NPP": patch_pools["NPP"],
        "PLANT_NDEMAND": patch_pools["PLANT_NDEMAND"],
        "QIRRIG": patch_pools["QIRRIG"],
        "SMINN": summaries.mineral_nitrogen_column[subgrid.patch.column],
        "TLAI": patch_pools["TLAI"],
    }
    return InstantaneousHistorySnapshot(
        gridcell={name: gridcell[name] for name in H1_HISTORY_VARIABLES},
        pft={name: pft[name] for name in H2_HISTORY_VARIABLES},
    )


def _history_from_owners(owners):
    return _compose_instantaneous_history_reference(**owners)


_compiled_history_from_owners = jit_discrete_parameters(_history_from_owners, forward_inputs=True)
_USE_COMPILED_HISTORY_OWNERS = True


def compose_instantaneous_history(**owners):
    if _USE_COMPILED_HISTORY_OWNERS and jax.default_backend() == "gpu":
        return _compiled_history_from_owners(owners)
    return _compose_instantaneous_history_reference(**owners)


__all__ = [
    "EarlyHistoryFields",
    "H1_HISTORY_VARIABLES",
    "H2_HISTORY_VARIABLES",
    "InstantaneousHistorySnapshot",
    "compose_instantaneous_history",
    "project_early_history_fields",
]
