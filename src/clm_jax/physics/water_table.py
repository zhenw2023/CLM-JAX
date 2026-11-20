
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from clm_jax.constants import (
    DENSITY_FRESH_WATER,
    DENSITY_ICE,
    FREEZING_TEMPERATURE_FRESHWATER,
)
from clm_jax.io.restart import (
    COLUMN_URBAN_IMPERVIOUS_ROAD,
    COLUMN_URBAN_ROOF,
    SubgridTopology,
)
from clm_jax.physics.numerics import (
    fortran_divide,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.snow_water import (
    GROUND_SURFACE_INDEX,
    NUMBER_OF_SOIL_SNOW_LEVELS,
)
from clm_jax.physics.surface_hydrology import (
    NUMBER_OF_GROUND_LEVELS,
    NUMBER_OF_SOIL_LEVELS,
    build_surface_hydrology_masks,
)

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase

SATURATION_THRESHOLD = np.float64(np.float32(0.9))


class WaterTableCondensationParameters(NamedTuple):

    time_step_seconds: float = 1800.0
    original_hydrology_flag: int = 0
    saturation_threshold: float = float(SATURATION_THRESHOLD)

    @classmethod
    def from_case(cls, case: CimeCase) -> WaterTableCondensationParameters:

        from clm_jax.io.namelist import read_case_namelist_group  

        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        hydrology = read_case_namelist_group(
            case,
            "lnd_in",
            "clm_soilhydrology_inparm",
        )
        original_flag = int(hydrology.get("origflag", 0))
        if original_flag != 0:
            raise ValueError("CLM-JAX currently requires CLM5 origflag=0 hydrology")
        return cls(
            time_step_seconds=float(general["dtime"]),
            original_hydrology_flag=original_flag,
        )


DEFAULT_WATER_TABLE_CONDENSATION_PARAMETERS = WaterTableCondensationParameters()


class WaterTableCondensationInputs(NamedTuple):

    soil_layer_count_column: jnp.ndarray
    snow_layer_count_column: jnp.ndarray
    layer_thickness_column_level: jnp.ndarray
    node_depth_column_level: jnp.ndarray
    lower_interface_depth_column_level: jnp.ndarray
    soil_temperature_column_level: jnp.ndarray
    saturated_water_content_column_level: jnp.ndarray
    surface_water_fraction_column: jnp.ndarray
    ground_dew_flux_column: jnp.ndarray
    snow_dew_flux_column: jnp.ndarray


class WaterTableCondensationState(NamedTuple):

    soil_snow_liquid_water_column_level: jnp.ndarray
    soil_snow_ice_water_column_level: jnp.ndarray
    volumetric_water_column_level: jnp.ndarray
    water_table_depth_column: jnp.ndarray
    perched_water_table_depth_column: jnp.ndarray
    frost_table_depth_column: jnp.ndarray
    snow_sublimation_flux_column: jnp.ndarray


class WaterTableCondensationDiagnostics(NamedTuple):

    hydrology_column: jnp.ndarray
    urban_roof_or_impervious_column: jnp.ndarray
    condensation_renewed_column: jnp.ndarray
    perched_search_skipped_column: jnp.ndarray
    frost_level_column: jnp.ndarray
    perched_level_column: jnp.ndarray
    water_table_level_column: jnp.ndarray


class WaterTableCondensationResult(NamedTuple):
    state: WaterTableCondensationState
    diagnostics: WaterTableCondensationDiagnostics


def _validate_shapes(
    topology: SubgridTopology,
    inputs: WaterTableCondensationInputs,
    state: WaterTableCondensationState,
    parameters: WaterTableCondensationParameters,
) -> None:
    column_shape = topology.column.gridcell.shape
    if len(column_shape) != 1:
        raise ValueError("column topology must be a vector")
    number_of_columns = column_shape[0]
    ground_shape = (number_of_columns, NUMBER_OF_GROUND_LEVELS)
    soil_interface_shape = (number_of_columns, NUMBER_OF_SOIL_LEVELS)
    soil_snow_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS)

    expected_inputs = {
        "soil_layer_count_column": column_shape,
        "snow_layer_count_column": column_shape,
        "layer_thickness_column_level": ground_shape,
        "node_depth_column_level": ground_shape,
        "lower_interface_depth_column_level": soil_interface_shape,
        "soil_temperature_column_level": ground_shape,
        "saturated_water_content_column_level": ground_shape,
        "surface_water_fraction_column": column_shape,
        "ground_dew_flux_column": column_shape,
        "snow_dew_flux_column": column_shape,
    }
    for name, expected in expected_inputs.items():
        if getattr(inputs, name).shape != expected:
            raise ValueError(f"{name} must have shape {expected}")

    expected_state = {
        "soil_snow_liquid_water_column_level": soil_snow_shape,
        "soil_snow_ice_water_column_level": soil_snow_shape,
        "volumetric_water_column_level": ground_shape,
        "water_table_depth_column": column_shape,
        "perched_water_table_depth_column": column_shape,
        "frost_table_depth_column": column_shape,
        "snow_sublimation_flux_column": column_shape,
    }
    for name, expected in expected_state.items():
        if getattr(state, name).shape != expected:
            raise ValueError(f"{name} must have shape {expected}")

    if parameters.time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")
    if parameters.original_hydrology_flag not in (0, 1):
        raise ValueError("original_hydrology_flag must be zero or one")
    if not 0.0 < parameters.saturation_threshold <= 1.0:
        raise ValueError("saturation_threshold must be in (0, 1]")


def _water_content(
    liquid_water: jnp.ndarray,
    ice_water: jnp.ndarray,
    layer_thickness: jnp.ndarray,
) -> jnp.ndarray:
    liquid_volume = fortran_divide(
        liquid_water,
        fortran_multiply(layer_thickness, jnp.float64(DENSITY_FRESH_WATER)),
    )
    ice_volume = fortran_divide(
        ice_water,
        fortran_multiply(layer_thickness, jnp.float64(DENSITY_ICE)),
    )
    return fortran_add(liquid_volume, ice_volume)


def _perched_water_table_column(
    layer_thickness: jnp.ndarray,
    node_depth: jnp.ndarray,
    soil_temperature: jnp.ndarray,
    saturated_water_content: jnp.ndarray,
    liquid_water: jnp.ndarray,
    ice_water: jnp.ndarray,
    initial_volumetric_water: jnp.ndarray,
    initial_water_table_depth: jnp.ndarray,
    saturation_threshold: jnp.ndarray,
    original_hydrology_flag: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    freezing = jnp.float64(FREEZING_TEMPERATURE_FRESHWATER)
    last_soil_level = jnp.int32(NUMBER_OF_SOIL_LEVELS - 1)
    initial_frost_level = jnp.where(
        soil_temperature[0] > freezing,
        last_soil_level,
        jnp.int32(0),
    )

    def find_frost_transition(
        level: int,
        carry: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        frost_level, found = carry
        transition = (soil_temperature[level - 1] > freezing) & (
            soil_temperature[level] <= freezing
        )
        use_level = transition & ~found
        return jnp.where(use_level, jnp.int32(level), frost_level), found | transition

    frost_level, _ = lax.fori_loop(
        1,
        NUMBER_OF_SOIL_LEVELS,
        find_frost_transition,
        (initial_frost_level, jnp.bool_(False)),
    )
    frost_table_depth = node_depth[frost_level]
    skip_search = (
        (initial_water_table_depth < frost_table_depth)
        & (soil_temperature[frost_level] <= freezing)
        & (original_hydrology_flag == jnp.int32(0))
    )

    def search_from_frost(
        offset: int,
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        volumetric_water, perched_level, found = carry
        level = last_soil_level - jnp.int32(offset)
        visits_level = (level <= frost_level) & ~found & ~skip_search
        candidate_water = _water_content(
            liquid_water[level],
            ice_water[level],
            layer_thickness[level],
        )
        volumetric_water = volumetric_water.at[level].set(
            jnp.where(visits_level, candidate_water, volumetric_water[level])
        )
        unsaturated = visits_level & (
            fortran_divide(candidate_water, saturated_water_content[level])
            <= saturation_threshold
        )
        perched_level = jnp.where(unsaturated, level, perched_level)
        return volumetric_water, perched_level, found | unsaturated

    volumetric_water, perched_level, _ = lax.fori_loop(
        0,
        NUMBER_OF_SOIL_LEVELS,
        search_from_frost,
        (initial_volumetric_water, jnp.int32(0), jnp.bool_(False)),
    )
    perched_level = jnp.where(
        soil_temperature[frost_level] > freezing,
        frost_level,
        perched_level,
    )

    interpolate = ~skip_search & (frost_level > perched_level)
    lower_level = jnp.minimum(perched_level + jnp.int32(1), last_soil_level)
    saturation_above = fortran_divide(
        _water_content(
            liquid_water[perched_level],
            ice_water[perched_level],
            layer_thickness[perched_level],
        ),
        saturated_water_content[perched_level],
    )
    saturation_below = fortran_divide(
        _water_content(
            liquid_water[lower_level],
            ice_water[lower_level],
            layer_thickness[lower_level],
        ),
        saturated_water_content[lower_level],
    )
    depth_difference = fortran_add(
        node_depth[lower_level],
        -node_depth[perched_level],
    )
    saturation_difference = fortran_add(saturation_below, -saturation_above)
    safe_saturation_difference = jnp.where(
        interpolate,
        saturation_difference,
        jnp.float64(1.0),
    )
    slope = fortran_divide(depth_difference, safe_saturation_difference)
    intercept = fortran_add(
        node_depth[lower_level],
        -fortran_multiply(slope, saturation_below),
    )
    interpolated_depth = jnp.maximum(
        jnp.float64(0.0),
        fortran_add(
            fortran_multiply(slope, saturation_threshold),
            intercept,
        ),
    )
    perched_water_table_depth = jnp.where(
        interpolate,
        interpolated_depth,
        frost_table_depth,
    )
    return (
        volumetric_water,
        perched_water_table_depth,
        frost_table_depth,
        frost_level,
        jnp.where(skip_search, jnp.int32(-1), perched_level),
    )


def _theta_water_table_column(
    soil_layer_count: jnp.ndarray,
    layer_thickness: jnp.ndarray,
    node_depth: jnp.ndarray,
    lower_interface_depth: jnp.ndarray,
    saturated_water_content: jnp.ndarray,
    liquid_water: jnp.ndarray,
    ice_water: jnp.ndarray,
    initial_volumetric_water: jnp.ndarray,
    saturation_threshold: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    last_soil_level = jnp.int32(NUMBER_OF_SOIL_LEVELS - 1)
    bottom_level = soil_layer_count - jnp.int32(1)

    def search_from_bedrock(
        offset: int,
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        volumetric_water, water_table_level, found = carry
        level = last_soil_level - jnp.int32(offset)
        visits_level = (level <= bottom_level) & ~found
        candidate_water = _water_content(
            liquid_water[level],
            ice_water[level],
            layer_thickness[level],
        )
        volumetric_water = volumetric_water.at[level].set(
            jnp.where(visits_level, candidate_water, volumetric_water[level])
        )
        unsaturated = visits_level & (
            fortran_divide(candidate_water, saturated_water_content[level])
            <= saturation_threshold
        )
        water_table_level = jnp.where(unsaturated, level, water_table_level)
        return volumetric_water, water_table_level, found | unsaturated

    volumetric_water, water_table_level, found = lax.fori_loop(
        0,
        NUMBER_OF_SOIL_LEVELS,
        search_from_bedrock,
        (initial_volumetric_water, bottom_level, jnp.bool_(False)),
    )
    water_table_level = jnp.where(found, water_table_level, jnp.int32(0))
    interpolate = (water_table_level > jnp.int32(0)) & (
        water_table_level < bottom_level
    )
    lower_level = jnp.minimum(water_table_level + jnp.int32(1), last_soil_level)
    saturation_above = fortran_divide(
        _water_content(
            liquid_water[water_table_level],
            ice_water[water_table_level],
            layer_thickness[water_table_level],
        ),
        saturated_water_content[water_table_level],
    )
    saturation_below = fortran_divide(
        _water_content(
            liquid_water[lower_level],
            ice_water[lower_level],
            layer_thickness[lower_level],
        ),
        saturated_water_content[lower_level],
    )
    depth_difference = fortran_add(
        node_depth[lower_level],
        -node_depth[water_table_level],
    )
    saturation_difference = fortran_add(saturation_below, -saturation_above)
    safe_saturation_difference = jnp.where(
        interpolate,
        saturation_difference,
        jnp.float64(1.0),
    )
    slope = fortran_divide(depth_difference, safe_saturation_difference)
    intercept = fortran_add(
        node_depth[lower_level],
        -fortran_multiply(slope, saturation_below),
    )
    interpolated_depth = jnp.maximum(
        jnp.float64(0.0),
        fortran_add(
            fortran_multiply(slope, saturation_threshold),
            intercept,
        ),
    )
    water_table_depth = jnp.where(
        water_table_level == jnp.int32(0),
        lower_interface_depth[0],
        jnp.where(
            interpolate,
            interpolated_depth,
            lower_interface_depth[bottom_level],
        ),
    )
    return volumetric_water, water_table_depth, water_table_level


def water_tables_and_condensation_active_case(
    topology: SubgridTopology,
    inputs: WaterTableCondensationInputs,
    state: WaterTableCondensationState,
    *,
    parameters: WaterTableCondensationParameters = (
        DEFAULT_WATER_TABLE_CONDENSATION_PARAMETERS
    ),
) -> WaterTableCondensationResult:

    _validate_shapes(topology, inputs, state, parameters)
    masks = build_surface_hydrology_masks(topology)
    hydrology = masks.hydrology_column
    column_type = jnp.asarray(topology.column.column_type, dtype=jnp.int32)
    roof_or_impervious = masks.urban_column & (
        (column_type == jnp.int32(COLUMN_URBAN_ROOF))
        | (column_type == jnp.int32(COLUMN_URBAN_IMPERVIOUS_ROAD))
    )

    soil_slice = slice(
        GROUND_SURFACE_INDEX,
        GROUND_SURFACE_INDEX + NUMBER_OF_SOIL_LEVELS,
    )
    safe_layer_count = jnp.where(
        hydrology,
        jnp.asarray(inputs.soil_layer_count_column, dtype=jnp.int32),
        jnp.int32(1),
    )
    safe_thickness = jnp.where(
        hydrology[:, None],
        inputs.layer_thickness_column_level,
        jnp.float64(1.0),
    )
    safe_node_depth = jnp.where(
        hydrology[:, None],
        inputs.node_depth_column_level,
        jnp.float64(0.0),
    )
    safe_interface_depth = jnp.where(
        hydrology[:, None],
        inputs.lower_interface_depth_column_level,
        jnp.float64(0.0),
    )
    safe_temperature = jnp.where(
        hydrology[:, None],
        inputs.soil_temperature_column_level,
        jnp.float64(FREEZING_TEMPERATURE_FRESHWATER + 1.0),
    )
    safe_porosity = jnp.where(
        hydrology[:, None],
        inputs.saturated_water_content_column_level,
        jnp.float64(1.0),
    )
    safe_liquid = jnp.where(
        hydrology[:, None],
        state.soil_snow_liquid_water_column_level[:, soil_slice],
        jnp.float64(0.0),
    )
    safe_ice = jnp.where(
        hydrology[:, None],
        state.soil_snow_ice_water_column_level[:, soil_slice],
        jnp.float64(0.0),
    )
    safe_volumetric = jnp.where(
        hydrology[:, None],
        state.volumetric_water_column_level,
        jnp.float64(0.0),
    )
    safe_water_table = jnp.where(
        hydrology,
        state.water_table_depth_column,
        jnp.float64(0.0),
    )
    threshold = jnp.float64(parameters.saturation_threshold)
    original_flag = jnp.int32(parameters.original_hydrology_flag)

    (
        perched_volumetric,
        candidate_perched_depth,
        candidate_frost_depth,
        frost_level,
        perched_level,
    ) = jax.vmap(_perched_water_table_column)(
        safe_thickness[:, :NUMBER_OF_SOIL_LEVELS],
        safe_node_depth[:, :NUMBER_OF_SOIL_LEVELS],
        safe_temperature[:, :NUMBER_OF_SOIL_LEVELS],
        safe_porosity[:, :NUMBER_OF_SOIL_LEVELS],
        safe_liquid,
        safe_ice,
        safe_volumetric,
        safe_water_table,
        jnp.broadcast_to(threshold, hydrology.shape),
        jnp.broadcast_to(original_flag, hydrology.shape),
    )
    (
        candidate_volumetric,
        candidate_water_table_depth,
        water_table_level,
    ) = jax.vmap(_theta_water_table_column)(
        safe_layer_count,
        safe_thickness[:, :NUMBER_OF_SOIL_LEVELS],
        safe_node_depth[:, :NUMBER_OF_SOIL_LEVELS],
        safe_interface_depth,
        safe_porosity[:, :NUMBER_OF_SOIL_LEVELS],
        safe_liquid,
        safe_ice,
        perched_volumetric,
        jnp.broadcast_to(threshold, hydrology.shape),
    )

    volumetric_water = jnp.where(
        hydrology[:, None],
        candidate_volumetric,
        state.volumetric_water_column_level,
    )
    water_table_depth = jnp.where(
        hydrology,
        candidate_water_table_depth,
        state.water_table_depth_column,
    )
    perched_water_table_depth = jnp.where(
        hydrology,
        candidate_perched_depth,
        state.perched_water_table_depth_column,
    )
    frost_table_depth = jnp.where(
        hydrology,
        candidate_frost_depth,
        state.frost_table_depth_column,
    )

    no_snow = jnp.asarray(inputs.snow_layer_count_column, dtype=jnp.int32) >= 0
    condensation_renewed = no_snow & (hydrology | roof_or_impervious)
    rural_weight = fortran_add(
        jnp.float64(1.0),
        -jnp.asarray(inputs.surface_water_fraction_column, dtype=jnp.float64),
    )
    condensation_weight = jnp.where(hydrology, rural_weight, jnp.float64(1.0))
    time_step = jnp.float64(parameters.time_step_seconds)
    liquid_increment = fortran_multiply(
        fortran_multiply(condensation_weight, inputs.ground_dew_flux_column),
        time_step,
    )
    ice_increment = fortran_multiply(
        fortran_multiply(condensation_weight, inputs.snow_dew_flux_column),
        time_step,
    )
    top_liquid_before = state.soil_snow_liquid_water_column_level[
        :, GROUND_SURFACE_INDEX
    ]
    top_ice_before = state.soil_snow_ice_water_column_level[:, GROUND_SURFACE_INDEX]
    top_liquid_after_dew = fortran_add(top_liquid_before, liquid_increment)
    top_ice_after_dew = fortran_add(top_ice_before, ice_increment)
    sublimation_mass = fortran_multiply(
        state.snow_sublimation_flux_column,
        time_step,
    )
    cap_sublimation = condensation_renewed & (
        sublimation_mass > top_ice_after_dew
    )
    weighted_sublimation_mass = fortran_multiply(
        fortran_multiply(
            condensation_weight,
            state.snow_sublimation_flux_column,
        ),
        time_step,
    )
    top_ice_after_sublimation = fortran_add(
        top_ice_after_dew,
        -weighted_sublimation_mass,
    )
    top_liquid = jnp.where(
        condensation_renewed,
        top_liquid_after_dew,
        top_liquid_before,
    )
    top_ice = jnp.where(
        cap_sublimation,
        jnp.float64(0.0),
        jnp.where(
            condensation_renewed,
            top_ice_after_sublimation,
            top_ice_before,
        ),
    )
    snow_sublimation_flux = jnp.where(
        cap_sublimation,
        fortran_divide(top_ice_after_dew, time_step),
        state.snow_sublimation_flux_column,
    )
    liquid_water = state.soil_snow_liquid_water_column_level.at[
        :, GROUND_SURFACE_INDEX
    ].set(top_liquid)
    ice_water = state.soil_snow_ice_water_column_level.at[
        :, GROUND_SURFACE_INDEX
    ].set(top_ice)

    invalid_level = jnp.int32(-1)
    return WaterTableCondensationResult(
        state=WaterTableCondensationState(
            soil_snow_liquid_water_column_level=liquid_water,
            soil_snow_ice_water_column_level=ice_water,
            volumetric_water_column_level=volumetric_water,
            water_table_depth_column=water_table_depth,
            perched_water_table_depth_column=perched_water_table_depth,
            frost_table_depth_column=frost_table_depth,
            snow_sublimation_flux_column=snow_sublimation_flux,
        ),
        diagnostics=WaterTableCondensationDiagnostics(
            hydrology_column=hydrology,
            urban_roof_or_impervious_column=roof_or_impervious,
            condensation_renewed_column=condensation_renewed,
            perched_search_skipped_column=hydrology & (perched_level < 0),
            frost_level_column=jnp.where(hydrology, frost_level, invalid_level),
            perched_level_column=jnp.where(hydrology, perched_level, invalid_level),
            water_table_level_column=jnp.where(
                hydrology,
                water_table_level,
                invalid_level,
            ),
        ),
    )
