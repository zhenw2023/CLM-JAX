
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.constants import (
    DENSITY_FRESH_WATER,
    DENSITY_ICE,
    FREEZING_TEMPERATURE_FRESHWATER,
    PI,
)
from clm_jax.io.restart import (
    LANDUNIT_LAKE,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    SubgridTopology,
)
from clm_jax.physics.numerics import (
    fortran_divide,
    fortran_ordered_sum,
    strict_acos,
    strict_exp,
    strict_negative_exp,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.native_power import native_power
from clm_jax.physics.snow_water import (
    NUMBER_OF_SNOW_LEVELS,
    NUMBER_OF_SOIL_SNOW_LEVELS,
    build_snow_column_masks,
)

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase


OVERBURDEN_COMPACTION_VIONNET_2012 = 2


class SnowCompactionParameters(NamedTuple):

    time_step_seconds: float = 1800.0
    integrated_snow_maximum: float = 2000.0
    destructive_metamorphism_density_limit: float = 175.0
    overburden_compaction_method: int = OVERBURDEN_COMPACTION_VIONNET_2012
    overburden_temperature_factor: float = 0.08
    wind_dependent_snow_density: bool = True
    use_subgrid_snow: bool = True

    @classmethod
    def from_case(cls, case: CimeCase) -> SnowCompactionParameters:

        from clm_jax.io.namelist import read_case_namelist_group  

        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        snow = read_case_namelist_group(
            case,
            "lnd_in",
            "clm_snowhydrology_inparm",
        )
        method_name = str(snow["snow_overburden_compaction_method"]).strip()
        if method_name != "Vionnet2012":
            raise ValueError(
                "CLM-JAX reference snow compaction requires Vionnet2012, got "
                f"{method_name!r}"
            )
        return cls(
            time_step_seconds=float(general["dtime"]),
            integrated_snow_maximum=float(general["int_snow_max"]),
            destructive_metamorphism_density_limit=float(
                snow.get("upplim_destruct_metamorph", 100.0)
            ),
            overburden_temperature_factor=float(
                snow.get("overburden_compress_tfactor", 0.08)
            ),
            wind_dependent_snow_density=bool(
                snow.get("wind_dependent_snow_density", False)
            ),
            use_subgrid_snow=True,
        )


DEFAULT_SNOW_COMPACTION_PARAMETERS = SnowCompactionParameters()


class SnowCompactionInputs(NamedTuple):
    snow_layer_count_column: jnp.ndarray
    snow_cover_shape_column: jnp.ndarray
    wind_speed_gridcell: jnp.ndarray
    soil_snow_temperature_column_level: jnp.ndarray
    soil_snow_melt_flag_column_level: jnp.ndarray
    effective_snow_fraction_column: jnp.ndarray
    previous_snow_water_column_level: jnp.ndarray
    integrated_snow_column: jnp.ndarray
    old_ice_fraction_column_level: jnp.ndarray
    soil_snow_ice_column_level: jnp.ndarray
    soil_snow_liquid_column_level: jnp.ndarray


class SnowCompactionState(NamedTuple):
    soil_snow_thickness_column_level: jnp.ndarray


class SnowCompactionDiagnostics(NamedTuple):
    snow_column: jnp.ndarray
    compacted_column_level: jnp.ndarray
    destructive_rate_column_level: jnp.ndarray
    overburden_rate_column_level: jnp.ndarray
    melt_rate_column_level: jnp.ndarray
    wind_drift_rate_column_level: jnp.ndarray
    final_burden_column: jnp.ndarray
    final_wind_pseudo_depth_column: jnp.ndarray
    final_wind_mobile_column: jnp.ndarray


class SnowCompactionResult(NamedTuple):
    state: SnowCompactionState
    diagnostics: SnowCompactionDiagnostics


def _validate_shapes(
    topology: SubgridTopology,
    inputs: SnowCompactionInputs,
    state: SnowCompactionState,
    parameters: SnowCompactionParameters,
) -> None:
    column_shape = topology.column.gridcell.shape
    if len(column_shape) != 1:
        raise ValueError("column topology must be a vector")
    number_of_columns = column_shape[0]
    soil_snow_shape = (number_of_columns, NUMBER_OF_SOIL_SNOW_LEVELS)
    snow_shape = (number_of_columns, NUMBER_OF_SNOW_LEVELS)
    grid_shape = topology.grid.longitude_degrees.shape

    expected = {
        "snow_layer_count_column": column_shape,
        "snow_cover_shape_column": column_shape,
        "wind_speed_gridcell": grid_shape,
        "soil_snow_temperature_column_level": soil_snow_shape,
        "soil_snow_melt_flag_column_level": soil_snow_shape,
        "effective_snow_fraction_column": column_shape,
        "previous_snow_water_column_level": snow_shape,
        "integrated_snow_column": column_shape,
        "old_ice_fraction_column_level": soil_snow_shape,
        "soil_snow_ice_column_level": soil_snow_shape,
        "soil_snow_liquid_column_level": soil_snow_shape,
    }
    for name, shape in expected.items():
        if getattr(inputs, name).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if state.soil_snow_thickness_column_level.shape != soil_snow_shape:
        raise ValueError(
            "soil_snow_thickness_column_level must have shape "
            f"{soil_snow_shape}"
        )
    if parameters.time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")
    if parameters.integrated_snow_maximum <= 0.0:
        raise ValueError("integrated_snow_maximum must be positive")
    if parameters.overburden_compaction_method != OVERBURDEN_COMPACTION_VIONNET_2012:
        raise ValueError("only the active Vionnet2012 compaction method is supported")


def _snow_cover_curve(ratio, shape):
    argument = fortran_add(
        fortran_multiply(jnp.float64(2.0), jnp.minimum(jnp.float64(1.0), ratio)),
        jnp.float64(-1.0),
    )
    return fortran_add(jnp.float64(1.0), -native_power(
        fortran_divide(strict_acos(argument), jnp.float64(PI)), shape,
    ))


@jax.custom_jvp
def _snow_cover_after_melt(ratio, shape):
    return _snow_cover_curve(ratio, shape)


@_snow_cover_after_melt.defjvp
def _snow_cover_after_melt_jvp(primals, tangents):
    def guarded(ratio, shape):
        plateau = ((ratio > jnp.float64(1.0)) & (shape > jnp.float64(0.0))) | (
            (ratio == jnp.float64(1.0)) & (shape > jnp.float64(2.0)))
        safe_ratio = jnp.where(plateau, jnp.float64(0.5), ratio)
        return jnp.where(plateau, jnp.float64(1.0), _snow_cover_curve(safe_ratio, shape))

    return _snow_cover_after_melt(*primals), jax.jvp(guarded, primals, tangents)[1]


def snow_compaction_active_case(
    topology: SubgridTopology,
    inputs: SnowCompactionInputs,
    state: SnowCompactionState,
    *,
    parameters: SnowCompactionParameters = DEFAULT_SNOW_COMPACTION_PARAMETERS,
    lake_only: bool = False,
) -> SnowCompactionResult:
    if jax.default_backend() == "gpu" and inputs.snow_layer_count_column.size > 16384:
        from clm_jax.physics.snow_column_filter import filtered_snow_transition

        def transition(local_topology, local_inputs, local_state):
            return _snow_compaction_active_case(
                local_topology, local_inputs, local_state,
                parameters=parameters, lake_only=lake_only,
            )

        return filtered_snow_transition(
            transition, topology, state, lake_only=lake_only, inputs=inputs,
            shared_input_fields=("wind_speed_gridcell",),
            diagnostic_defaults={"final_wind_mobile_column": True},
        )
    return _snow_compaction_active_case(
        topology, inputs, state, parameters=parameters, lake_only=lake_only,
    )


def _snow_compaction_active_case(
    topology: SubgridTopology,
    inputs: SnowCompactionInputs,
    state: SnowCompactionState,
    *,
    parameters: SnowCompactionParameters = DEFAULT_SNOW_COMPACTION_PARAMETERS,
    lake_only: bool = False,
) -> SnowCompactionResult:

    _validate_shapes(topology, inputs, state, parameters)
    number_of_columns = topology.column.gridcell.shape[0]
    masks = build_snow_column_masks(
        topology,
        inputs.snow_layer_count_column,
        lake_only=lake_only,
    )
    snow_column = masks.snow_column
    layer_count = jnp.asarray(inputs.snow_layer_count_column, dtype=jnp.int32)
    top_snow_index = layer_count + jnp.int32(NUMBER_OF_SNOW_LEVELS)

    temperature = jnp.asarray(
        inputs.soil_snow_temperature_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_SNOW_LEVELS]
    melt_flag = jnp.asarray(
        inputs.soil_snow_melt_flag_column_level,
        dtype=jnp.int32,
    )[:, :NUMBER_OF_SNOW_LEVELS]
    ice = jnp.asarray(inputs.soil_snow_ice_column_level, dtype=jnp.float64)[
        :, :NUMBER_OF_SNOW_LEVELS
    ]
    liquid = jnp.asarray(inputs.soil_snow_liquid_column_level, dtype=jnp.float64)[
        :, :NUMBER_OF_SNOW_LEVELS
    ]
    old_ice_fraction = jnp.asarray(
        inputs.old_ice_fraction_column_level,
        dtype=jnp.float64,
    )[:, :NUMBER_OF_SNOW_LEVELS]
    previous_snow_water = jnp.asarray(
        inputs.previous_snow_water_column_level,
        dtype=jnp.float64,
    )
    snow_fraction = jnp.asarray(
        inputs.effective_snow_fraction_column,
        dtype=jnp.float64,
    )
    integrated_snow = jnp.asarray(inputs.integrated_snow_column, dtype=jnp.float64)
    snow_cover_shape = jnp.asarray(inputs.snow_cover_shape_column, dtype=jnp.float64)
    wind_speed = jnp.asarray(inputs.wind_speed_gridcell, dtype=jnp.float64)[
        jnp.asarray(topology.column.gridcell, dtype=jnp.int32)
    ]
    landunit_type = jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
    urban = (landunit_type >= jnp.int32(LANDUNIT_URBAN_MIN)) & (
        landunit_type <= jnp.int32(LANDUNIT_URBAN_MAX)
    )
    lake = landunit_type == jnp.int32(LANDUNIT_LAKE)
    rural_subgrid = jnp.bool_(parameters.use_subgrid_snow) & ~lake & ~urban

    snow_level = jnp.arange(NUMBER_OF_SNOW_LEVELS, dtype=jnp.int32)
    active_snow_level = snow_column[:, None] & (
        snow_level[None, :] >= top_snow_index[:, None]
    )
    layer_mass = fortran_add(ice, liquid)
    snowpack_mass = fortran_ordered_sum(
        jnp.where(active_snow_level, layer_mass, jnp.float64(0.0)).T
    )
    limited_integrated_snow = jnp.minimum(
        integrated_snow,
        jnp.float64(parameters.integrated_snow_maximum),
    )

    thickness = jnp.asarray(
        state.soil_snow_thickness_column_level,
        dtype=jnp.float64,
    )
    burden = jnp.zeros(number_of_columns, dtype=jnp.float64)
    pseudo_depth = jnp.zeros(number_of_columns, dtype=jnp.float64)
    mobile = jnp.ones(number_of_columns, dtype=jnp.bool_)

    active = active_snow_level
    layer_ice = jnp.where(active, ice, jnp.float64(1.0))
    layer_liquid = jnp.where(active, liquid, jnp.float64(0.0))
    water_mass = jnp.where(
        active,
        layer_mass,
        jnp.float64(1.0),
    )
    layer_thickness = thickness[:, :NUMBER_OF_SNOW_LEVELS]

    safe_fraction = jnp.where(active, snow_fraction[:, None], jnp.float64(1.0))
    safe_thickness = jnp.where(active, layer_thickness, jnp.float64(1.0))
    ice_volume = fortran_divide(layer_ice, jnp.float64(DENSITY_ICE))
    liquid_volume = fortran_divide(
        layer_liquid,
        jnp.float64(DENSITY_FRESH_WATER),
    )
    pore_volume = fortran_add(ice_volume, liquid_volume)
    fractional_layer_volume = fortran_multiply(safe_fraction, safe_thickness)
    void_fraction = fortran_add(
        jnp.float64(1.0),
        -fortran_divide(pore_volume, fractional_layer_volume),
    )
    eligible = (
        active
        & (void_fraction > jnp.float64(0.001))
        & (layer_ice > jnp.float64(0.1))
    )

    safe_layer_mass = jnp.where(eligible, water_mass, jnp.float64(1.0))
    partial_ice_density = fortran_divide(
        layer_ice,
        fractional_layer_volume,
    )
    ice_fraction = fortran_divide(layer_ice, safe_layer_mass)
    temperature_deficit = fortran_add(
        jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
        -jnp.where(
            active,
            temperature,
            jnp.float64(FREEZING_TEMPERATURE_FRESHWATER),
        ),
    )

    destructive_rate = fortran_multiply(
        jnp.float64(-2.777e-6),
        strict_negative_exp(
            -fortran_multiply(jnp.float64(0.04), temperature_deficit)
        ),
    )
    above_destructive_limit = (
        partial_ice_density
        > jnp.float64(parameters.destructive_metamorphism_density_limit)
    )
    density_attenuation_argument = -fortran_multiply(
        jnp.float64(46.0e-3),
        fortran_add(
            partial_ice_density,
            -jnp.float64(parameters.destructive_metamorphism_density_limit),
        ),
    )
    destructive_rate = jnp.where(
        above_destructive_limit,
        fortran_multiply(
            destructive_rate,
            strict_negative_exp(density_attenuation_argument),
        ),
        destructive_rate,
    )
    liquid_threshold = fortran_multiply(
        fortran_multiply(jnp.float64(0.01), safe_thickness),
        safe_fraction,
    )
    destructive_rate = jnp.where(
        layer_liquid > liquid_threshold,
        fortran_multiply(destructive_rate, jnp.float64(2.0)),
        destructive_rate,
    )
    destructive_rate = jnp.where(eligible, destructive_rate, jnp.float64(0.0))

    liquid_viscosity_ratio = fortran_divide(
        fortran_multiply(jnp.float64(60.0), layer_liquid),
        fortran_multiply(jnp.float64(DENSITY_FRESH_WATER), safe_thickness),
    )
    viscosity_liquid_factor = fortran_divide(
        jnp.float64(1.0),
        fortran_add(jnp.float64(1.0), liquid_viscosity_ratio),
    )
    viscosity_exponent = fortran_add(
        fortran_multiply(jnp.float64(0.1), temperature_deficit),
        fortran_multiply(jnp.float64(0.023), partial_ice_density),
    )
    viscosity = fortran_multiply(
        fortran_multiply(
            fortran_multiply(
                fortran_multiply(
                    viscosity_liquid_factor,
                    jnp.float64(4.0),
                ),
                fortran_divide(partial_ice_density, jnp.float64(450.0)),
            ),
            strict_exp(viscosity_exponent),
        ),
        jnp.float64(7.62237e6),
    )
    melting = eligible & (melt_flag == jnp.int32(1))
    rural_melting = melting & rural_subgrid[:, None]
    non_rural_melting = melting & ~rural_subgrid[:, None]
    melt_mass_loss = fortran_add(
        previous_snow_water,
        -water_mass,
    )
    initial_melt_fraction = jnp.maximum(
        jnp.float64(0.0),
        jnp.minimum(
            jnp.float64(1.0),
            fortran_divide(melt_mass_loss, safe_layer_mass),
        ),
    )
    positive_melt = rural_melting & (melt_mass_loss > jnp.float64(0.0))
    safe_integrated_snow = jnp.where(
        jnp.any(positive_melt, axis=1),
        limited_integrated_snow,
        jnp.float64(1.0),
    )
    snowpack_ratio = fortran_divide(snowpack_mass, safe_integrated_snow)
    snowpack_ratio = jnp.where(jnp.any(positive_melt, axis=1), snowpack_ratio, jnp.float64(0.5))
    snow_cover_after_melt = _snow_cover_after_melt(snowpack_ratio, snow_cover_shape)
    snow_cover_compaction = jnp.maximum(
        jnp.float64(0.0),
        fortran_divide(
            fortran_add(snow_cover_after_melt, -snow_fraction),
            jnp.where(snow_column, snow_fraction, jnp.float64(1.0)),
        ),
    )
    rural_compaction_fraction = jnp.where(
        positive_melt,
        fortran_add(initial_melt_fraction, -snow_cover_compaction[:, None]),
        initial_melt_fraction,
    )
    inverse_negative_time_step = fortran_divide(
        jnp.float64(-1.0),
        jnp.float64(parameters.time_step_seconds),
    )
    rural_melt_rate = fortran_multiply(
        inverse_negative_time_step,
        rural_compaction_fraction,
    )

    safe_old_ice_fraction = jnp.where(
        non_rural_melting,
        old_ice_fraction,
        jnp.float64(1.0),
    )
    urban_melt_fraction = jnp.maximum(
        jnp.float64(0.0),
        fortran_divide(
            fortran_add(safe_old_ice_fraction, -ice_fraction),
            safe_old_ice_fraction,
        ),
    )
    urban_melt_rate = fortran_multiply(
        inverse_negative_time_step,
        urban_melt_fraction,
    )
    melt_rate = jnp.where(
        rural_melting,
        rural_melt_rate,
        jnp.where(non_rural_melting, urban_melt_rate, jnp.float64(0.0)),
    )

    density_factor = fortran_add(
        jnp.float64(1.25),
        -fortran_multiply(
            jnp.float64(0.0042),
            fortran_add(
                jnp.maximum(jnp.float64(50.0), partial_ice_density),
                jnp.float64(-50.0),
            ),
        ),
    )
    morphology_inner = fortran_add(
        fortran_add(
            fortran_multiply(jnp.float64(-0.583), jnp.float64(0.35e-3)),
            -fortran_multiply(jnp.float64(0.833), jnp.float64(1.0)),
        ),
        jnp.float64(0.833),
    )
    mobility_index = fortran_add(
        fortran_multiply(jnp.float64(0.34), morphology_inner),
        fortran_multiply(jnp.float64(0.66), density_factor),
    )
    driftability_index = fortran_add(
        fortran_add(
            fortran_multiply(
                jnp.float64(-2.868),
                strict_negative_exp(
                    -fortran_multiply(jnp.float64(0.085), wind_speed[:, None])
                ),
            ),
            jnp.float64(1.0),
        ),
        mobility_index,
    )
    limited_driftability = jnp.minimum(driftability_index, jnp.float64(3.25))
    pseudo_increment = fortran_multiply(
        fortran_multiply(jnp.float64(0.5), safe_thickness),
        fortran_add(jnp.float64(3.25), -limited_driftability),
    )
    safe_viscosity = jnp.where(eligible, viscosity, jnp.float64(1.0))
    wind_enabled = jnp.bool_(parameters.wind_dependent_snow_density)

    def overlying_snow(carry, layer):
        current_burden, current_pseudo_depth, current_mobile = carry
        active, eligible, water_mass, pseudo_increment, driftability_index = layer
        wind_candidate = eligible & wind_enabled & current_mobile
        driftable = wind_candidate & (driftability_index > jnp.float64(0.0))
        middle_pseudo_depth = fortran_add(current_pseudo_depth, pseudo_increment)
        next_pseudo_depth = jnp.where(
            driftable, fortran_add(middle_pseudo_depth, pseudo_increment),
            current_pseudo_depth,
        )
        next_mobile = jnp.where(
            active & ~eligible, jnp.bool_(False),
            jnp.where(wind_candidate & ~driftable, jnp.bool_(False), current_mobile),
        )
        next_burden = jnp.where(
            active, fortran_add(current_burden, water_mass), current_burden,
        )
        return (
            (next_burden, next_pseudo_depth, next_mobile),
            (current_burden, middle_pseudo_depth, driftable),
        )

    (burden, pseudo_depth, mobile), recurrence = lax.scan(
        overlying_snow, (burden, pseudo_depth, mobile),
        tuple(value.T for value in (
            active, eligible, water_mass, pseudo_increment, driftability_index,
        )),
    )
    layer_burden, middle_pseudo_depth, driftable = (value.T for value in recurrence)
    overburden_numerator = fortran_add(
        layer_burden,
        fortran_divide(water_mass, jnp.float64(2.0)),
    )
    overburden_rate = -fortran_divide(overburden_numerator, safe_viscosity)
    overburden_rate = jnp.where(eligible, overburden_rate, jnp.float64(0.0))

    drift_scaling = fortran_multiply(
        limited_driftability,
        strict_negative_exp(
            -fortran_divide(middle_pseudo_depth, jnp.float64(0.1))
        ),
    )
    inverse_drift_time = fortran_divide(
        drift_scaling,
        jnp.float64(48.0 * 3600.0),
    )
    wind_rate = -fortran_multiply(
        jnp.maximum(
            jnp.float64(0.0),
            fortran_add(jnp.float64(350.0), -partial_ice_density),
        ),
        inverse_drift_time,
    )
    wind_rate = jnp.where(driftable, wind_rate, jnp.float64(0.0))
    total_rate = fortran_add(
        fortran_add(fortran_add(destructive_rate, overburden_rate), melt_rate),
        wind_rate,
    )
    thickness_factor = fortran_add(
        jnp.float64(1.0),
        fortran_multiply(total_rate, jnp.float64(parameters.time_step_seconds)),
    )
    compacted_thickness = fortran_multiply(safe_thickness, thickness_factor)
    saturated_thickness = fortran_divide(pore_volume, safe_fraction)
    updated_thickness = jnp.maximum(compacted_thickness, saturated_thickness)
    thickness = jnp.concatenate((
        jnp.where(eligible, updated_thickness, layer_thickness),
        thickness[:, NUMBER_OF_SNOW_LEVELS:],
    ), axis=1)

    return SnowCompactionResult(
        state=SnowCompactionState(
            soil_snow_thickness_column_level=thickness,
        ),
        diagnostics=SnowCompactionDiagnostics(
            snow_column=snow_column,
            compacted_column_level=eligible,
            destructive_rate_column_level=destructive_rate,
            overburden_rate_column_level=overburden_rate,
            melt_rate_column_level=melt_rate,
            wind_drift_rate_column_level=wind_rate,
            final_burden_column=burden,
            final_wind_pseudo_depth_column=pseudo_depth,
            final_wind_mobile_column=mobile,
        ),
    )
