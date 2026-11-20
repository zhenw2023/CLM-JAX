
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import default_backend, lax

from clm_jax.constants import (
    CLM_SPECIAL_VALUE,
    LATENT_HEAT_FUSION,
    STEFAN_BOLTZMANN,
)
from clm_jax.io.restart import (
    COLUMN_URBAN_IMPERVIOUS_ROAD,
    COLUMN_URBAN_PERVIOUS_ROAD,
    COLUMN_URBAN_ROOF,
    COLUMN_URBAN_SHADE_WALL,
    COLUMN_URBAN_SUN_WALL,
    LANDUNIT_GLACIER,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    SubgridTopology,
)
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
)

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase


class LandToAtmosphereParameters(NamedTuple):
    melt_non_icesheet_ice_runoff: bool = True
    use_methane: bool = True
    methane_offline: bool = True
    maximum_columns_per_gridcell: int = 64
    maximum_patches_per_gridcell: int = 64
    special_value: float = float(CLM_SPECIAL_VALUE)
    latent_heat_fusion: float = float(LATENT_HEAT_FUSION)
    stefan_boltzmann: float = float(STEFAN_BOLTZMANN)

    @classmethod
    def from_case(cls, case: CimeCase) -> LandToAtmosphereParameters:

        from clm_jax.io.namelist import read_case_namelist_group  

        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        coupling = read_case_namelist_group(case, "lnd_in", "lnd2atm_inparm")
        methane = read_case_namelist_group(case, "lnd_in", "ch4par_in")
        return cls(
            melt_non_icesheet_ice_runoff=bool(coupling["melt_non_icesheet_ice_runoff"]),
            use_methane=bool(general.get("use_lch4", False)),
            methane_offline=bool(methane.get("ch4offline", True)),
        )


DEFAULT_LAND_TO_ATMOSPHERE_PARAMETERS = LandToAtmosphereParameters()


class LandToAtmosphereTopology(NamedTuple):
    subgrid: SubgridTopology
    canyon_height_width_ratio_landunit: jnp.ndarray
    ice_runoff_melted_gridcell: jnp.ndarray
    gridcell_area_square_kilometers: jnp.ndarray


class LandToAtmosphereInputs(NamedTuple):
    snow_water_column: jnp.ndarray
    volumetric_soil_water_column: jnp.ndarray
    direct_albedo_patch: jnp.ndarray
    diffuse_albedo_patch: jnp.ndarray
    outgoing_longwave_patch: jnp.ndarray
    reference_temperature_2m_patch: jnp.ndarray
    reference_specific_humidity_2m_patch: jnp.ndarray
    reference_wind_speed_10m_patch: jnp.ndarray
    eastward_stress_patch: jnp.ndarray
    northward_stress_patch: jnp.ndarray
    total_evaporation_patch: jnp.ndarray
    absorbed_solar_patch: jnp.ndarray
    friction_velocity_patch: jnp.ndarray
    aerodynamic_resistance_patch: jnp.ndarray
    sensible_heat_patch: jnp.ndarray
    latent_heat_patch: jnp.ndarray
    dry_deposition_velocity_patch: jnp.ndarray
    voc_flux_patch: jnp.ndarray
    fire_flux_patch: jnp.ndarray
    fire_injection_top_patch: jnp.ndarray
    dust_flux_patch: jnp.ndarray
    ice_runoff_snow_capping_column: jnp.ndarray
    excess_ice_runoff_column: jnp.ndarray
    precipitation_conversion_sensible_heat_column: jnp.ndarray
    surface_runoff_column: jnp.ndarray
    drainage_column: jnp.ndarray
    rain_to_snow_runoff_conversion_column: jnp.ndarray
    surface_water_runoff_column: jnp.ndarray
    perched_drainage_column: jnp.ndarray
    irrigation_column: jnp.ndarray
    methane_surface_flux_column: jnp.ndarray
    ending_water_balance_column: jnp.ndarray
    dynamic_sensible_heat_gridcell: jnp.ndarray
    net_carbon_exchange_gridcell: jnp.ndarray
    net_methane_carbon_correction_gridcell: jnp.ndarray
    dynamic_liquid_water_gridcell: jnp.ndarray
    dynamic_ice_water_gridcell: jnp.ndarray
    river_volume_gridcell: jnp.ndarray


class LandToAtmosphereState(NamedTuple):
    groundwater_runoff_column: jnp.ndarray
    total_runoff_column: jnp.ndarray
    liquid_from_ice_column: jnp.ndarray
    ice_to_liquid_sensible_heat_column: jnp.ndarray
    total_water_storage_gridcell: jnp.ndarray


class LandToAtmosphereFluxes(NamedTuple):
    radiative_temperature_gridcell: jnp.ndarray
    reference_temperature_2m_gridcell: jnp.ndarray
    reference_specific_humidity_2m_gridcell: jnp.ndarray
    reference_wind_speed_10m_gridcell: jnp.ndarray
    snow_water_gridcell: jnp.ndarray
    volumetric_soil_water_gridcell: jnp.ndarray
    direct_albedo_gridcell: jnp.ndarray
    diffuse_albedo_gridcell: jnp.ndarray
    eastward_stress_gridcell: jnp.ndarray
    northward_stress_gridcell: jnp.ndarray
    latent_heat_gridcell: jnp.ndarray
    sensible_heat_gridcell: jnp.ndarray
    precipitation_conversion_sensible_heat_gridcell: jnp.ndarray
    outgoing_longwave_gridcell: jnp.ndarray
    total_evaporation_gridcell: jnp.ndarray
    absorbed_solar_gridcell: jnp.ndarray
    net_carbon_exchange_gridcell: jnp.ndarray
    aerodynamic_resistance_gridcell: jnp.ndarray
    friction_velocity_gridcell: jnp.ndarray
    dry_deposition_velocity_gridcell: jnp.ndarray
    voc_flux_gridcell: jnp.ndarray
    fire_flux_gridcell: jnp.ndarray
    fire_injection_top_gridcell: jnp.ndarray
    dust_flux_gridcell: jnp.ndarray
    methane_surface_flux_gridcell: jnp.ndarray
    liquid_runoff_gridcell: jnp.ndarray
    surface_runoff_gridcell: jnp.ndarray
    drainage_gridcell: jnp.ndarray
    groundwater_runoff_gridcell: jnp.ndarray
    surface_water_runoff_gridcell: jnp.ndarray
    perched_drainage_gridcell: jnp.ndarray
    ice_runoff_gridcell: jnp.ndarray
    irrigation_gridcell: jnp.ndarray


class LandToAtmosphereDiagnostics(NamedTuple):
    ice_to_liquid_conversion_column: jnp.ndarray
    ice_runoff_column: jnp.ndarray
    ice_to_liquid_sensible_heat_gridcell: jnp.ndarray


class LandToAtmosphereResult(NamedTuple):
    state: LandToAtmosphereState
    fluxes: LandToAtmosphereFluxes
    diagnostics: LandToAtmosphereDiagnostics


def _urban_flux_scale(topology: LandToAtmosphereTopology) -> jnp.ndarray:
    column = topology.subgrid.column
    landunit_type = jnp.asarray(column.landunit_type, dtype=jnp.int32)
    column_type = jnp.asarray(column.column_type, dtype=jnp.int32)
    landunit = jnp.asarray(column.landunit, dtype=jnp.int32)
    height_width_ratio = jnp.asarray(
        topology.canyon_height_width_ratio_landunit,
        dtype=jnp.float64,
    )[landunit]
    urban = (landunit_type >= LANDUNIT_URBAN_MIN) & (landunit_type <= LANDUNIT_URBAN_MAX)
    wall = (column_type == COLUMN_URBAN_SUN_WALL) | (column_type == COLUMN_URBAN_SHADE_WALL)
    road = (column_type == COLUMN_URBAN_PERVIOUS_ROAD) | (
        column_type == COLUMN_URBAN_IMPERVIOUS_ROAD
    )
    roof = column_type == COLUMN_URBAN_ROOF
    urban_scale = jnp.where(
        wall,
        fortran_multiply(jnp.float64(3.0), height_width_ratio),
        jnp.where(road, jnp.float64(3.0), jnp.where(roof, jnp.float64(1.0), jnp.nan)),
    )
    return jnp.where(urban, urban_scale, jnp.float64(1.0))


def _aggregate_to_gridcell(
    values: jnp.ndarray,
    source_gridcell: jnp.ndarray,
    source_weight: jnp.ndarray,
    source_active: jnp.ndarray,
    source_scale: jnp.ndarray,
    *,
    number_of_gridcells: int,
    maximum_sources_per_gridcell: int,
    special_value: float,
) -> jnp.ndarray:
    if default_backend() != "gpu":
        return _aggregate_to_gridcell_reference(
            values, source_gridcell, source_weight, source_active, source_scale,
            number_of_gridcells=number_of_gridcells,
            maximum_sources_per_gridcell=maximum_sources_per_gridcell,
            special_value=special_value,
        )
    from clm_jax.physics.grid_aggregation_gpu import ordered_scaled_mean

    array = jnp.asarray(values, jnp.float64)
    vector = array.ndim == 1
    if vector:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("aggregated source fields must be vectors or matrices")
    result = ordered_scaled_mean(
        array, jnp.asarray(source_weight, jnp.float64),
        jnp.asarray(source_active, jnp.bool_), jnp.asarray(source_scale, jnp.float64),
        jnp.asarray(source_gridcell, jnp.int32),
        number_of_gridcells, maximum_sources_per_gridcell, special_value,
    )
    return result[:, 0] if vector else result


def _aggregate_to_gridcell_reference(
    values: jnp.ndarray,
    source_gridcell: jnp.ndarray,
    source_weight: jnp.ndarray,
    source_active: jnp.ndarray,
    source_scale: jnp.ndarray,
    *,
    number_of_gridcells: int,
    maximum_sources_per_gridcell: int,
    special_value: float,
) -> jnp.ndarray:

    array = jnp.asarray(values, dtype=jnp.float64)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("aggregated source fields must be vectors or matrices")
    gridcell = jnp.asarray(source_gridcell, dtype=jnp.int32)
    weight = jnp.asarray(source_weight, dtype=jnp.float64)
    active = jnp.asarray(source_active, dtype=jnp.bool_)
    scale = jnp.asarray(source_scale, dtype=jnp.float64)
    source_count = array.shape[0]
    field_count = array.shape[1]
    output_shape = (number_of_gridcells, field_count)
    sentinel = jnp.float64(special_value)
    if source_count == 0:
        result = jnp.full(output_shape, sentinel)
        return result[:, 0] if was_vector else result

    source_index = jnp.arange(source_count, dtype=jnp.int32)
    first_source = jnp.full(number_of_gridcells, source_count, dtype=jnp.int32)
    first_source = first_source.at[gridcell].min(source_index)
    gridcell_index = jnp.arange(number_of_gridcells, dtype=jnp.int32)
    initial = (
        jnp.zeros(output_shape, dtype=jnp.float64),
        jnp.zeros(output_shape, dtype=jnp.float64),
    )

    def accumulate(
        offset: int,
        carry: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        totals, sum_weights = carry
        unbounded_source = first_source + jnp.int32(offset)
        bounded_source = jnp.clip(unbounded_source, 0, source_count - 1)
        selected_values = array[bounded_source, :]
        selected_weight = weight[bounded_source, None]
        selected_scale = scale[bounded_source, None]
        belongs = (unbounded_source < source_count) & (gridcell[bounded_source] == gridcell_index)
        contributes = (
            belongs[:, None]
            & active[bounded_source, None]
            & (selected_weight != jnp.float64(0.0))
            & (selected_values != sentinel)
            & (selected_scale != sentinel)
        )
        contribution = fortran_multiply(
            fortran_multiply(selected_values, selected_scale),
            selected_weight,
        )
        return (
            jnp.where(contributes, fortran_add(totals, contribution), totals),
            jnp.where(
                contributes,
                fortran_add(sum_weights, selected_weight),
                sum_weights,
            ),
        )

    totals, sum_weights = lax.fori_loop(
        0,
        maximum_sources_per_gridcell,
        accumulate,
        initial,
    )
    contributes = sum_weights != jnp.float64(0.0)
    safe_weight = jnp.where(contributes, sum_weights, jnp.float64(1.0))
    result = jnp.where(
        contributes,
        fortran_divide(totals, safe_weight),
        sentinel,
    )
    return result[:, 0] if was_vector else result


def _cast_and_validate(
    topology: LandToAtmosphereTopology,
    inputs: LandToAtmosphereInputs,
    state: LandToAtmosphereState,
    parameters: LandToAtmosphereParameters,
) -> tuple[LandToAtmosphereInputs, LandToAtmosphereState]:
    column_count = topology.subgrid.column.gridcell.shape[0]
    patch_count = topology.subgrid.patch.gridcell.shape[0]
    gridcell_count = topology.subgrid.grid.longitude_degrees.shape[0]
    landunit_count = topology.subgrid.landunit.gridcell.shape[0]
    if parameters.maximum_columns_per_gridcell <= 0:
        raise ValueError("maximum_columns_per_gridcell must be positive")
    if parameters.maximum_patches_per_gridcell <= 0:
        raise ValueError("maximum_patches_per_gridcell must be positive")
    if parameters.stefan_boltzmann <= 0.0:
        raise ValueError("stefan_boltzmann must be positive")
    topology_shapes = {
        "canyon_height_width_ratio_landunit": (
            topology.canyon_height_width_ratio_landunit,
            (landunit_count,),
        ),
        "ice_runoff_melted_gridcell": (
            topology.ice_runoff_melted_gridcell,
            (gridcell_count,),
        ),
        "gridcell_area_square_kilometers": (
            topology.gridcell_area_square_kilometers,
            (gridcell_count,),
        ),
    }
    for name, (value, shape) in topology_shapes.items():
        if jnp.shape(value) != shape:
            raise ValueError(f"{name} must have shape {shape}")

    column_fields = LandToAtmosphereInputs._fields[:2] + LandToAtmosphereInputs._fields[21:32]
    patch_fields = LandToAtmosphereInputs._fields[2:21]
    gridcell_fields = LandToAtmosphereInputs._fields[32:]
    cast_inputs: dict[str, jnp.ndarray] = {}
    for field in column_fields:
        value = jnp.asarray(getattr(inputs, field), dtype=jnp.float64)
        if value.shape[0] != column_count or value.ndim not in (1, 2):
            raise ValueError(f"{field} must have leading size {column_count}")
        cast_inputs[field] = value
    for field in patch_fields:
        value = jnp.asarray(getattr(inputs, field), dtype=jnp.float64)
        if value.shape[0] != patch_count or value.ndim not in (1, 2):
            raise ValueError(f"{field} must have leading size {patch_count}")
        cast_inputs[field] = value
    for field in gridcell_fields:
        value = jnp.asarray(getattr(inputs, field), dtype=jnp.float64)
        if value.shape != (gridcell_count,):
            raise ValueError(f"{field} must have shape {(gridcell_count,)}")
        cast_inputs[field] = value
    cast_state: dict[str, jnp.ndarray] = {}
    for field in LandToAtmosphereState._fields[:4]:
        value = jnp.asarray(getattr(state, field), dtype=jnp.float64)
        if value.shape != (column_count,):
            raise ValueError(f"{field} must have shape {(column_count,)}")
        cast_state[field] = value
    total_water = jnp.asarray(state.total_water_storage_gridcell, dtype=jnp.float64)
    if total_water.shape != (gridcell_count,):
        raise ValueError(f"total_water_storage_gridcell must have shape {(gridcell_count,)}")
    cast_state["total_water_storage_gridcell"] = total_water
    return LandToAtmosphereInputs(**cast_inputs), LandToAtmosphereState(**cast_state)


def land_to_atmosphere_coupling(
    topology: LandToAtmosphereTopology,
    inputs: LandToAtmosphereInputs,
    state: LandToAtmosphereState,
    *,
    parameters: LandToAtmosphereParameters = DEFAULT_LAND_TO_ATMOSPHERE_PARAMETERS,
) -> LandToAtmosphereResult:

    inputs, state = _cast_and_validate(topology, inputs, state, parameters)
    subgrid = topology.subgrid
    column_active = jnp.asarray(subgrid.column.active, dtype=jnp.bool_)
    column_gridcell = jnp.asarray(subgrid.column.gridcell, dtype=jnp.int32)
    patch_column = jnp.asarray(subgrid.patch.column, dtype=jnp.int32)
    gridcell_count = subgrid.grid.longitude_degrees.shape[0]
    urban_scale_column = _urban_flux_scale(topology)
    unity_column = jnp.ones_like(urban_scale_column)
    urban_scale_patch = urban_scale_column[patch_column]
    unity_patch = jnp.ones(subgrid.patch.gridcell.shape, dtype=jnp.float64)

    def c2g(values: jnp.ndarray, *, urban: bool = True) -> jnp.ndarray:
        return _aggregate_to_gridcell(
            values,
            subgrid.column.gridcell,
            subgrid.column.weight_on_gridcell,
            subgrid.column.active,
            urban_scale_column if urban else unity_column,
            number_of_gridcells=gridcell_count,
            maximum_sources_per_gridcell=parameters.maximum_columns_per_gridcell,
            special_value=parameters.special_value,
        )

    def p2g(values: jnp.ndarray, *, urban: bool = False) -> jnp.ndarray:
        return _aggregate_to_gridcell(
            values,
            subgrid.patch.gridcell,
            subgrid.patch.weight_on_gridcell,
            subgrid.patch.active,
            urban_scale_patch if urban else unity_patch,
            number_of_gridcells=gridcell_count,
            maximum_sources_per_gridcell=parameters.maximum_patches_per_gridcell,
            special_value=parameters.special_value,
        )

    ice_runoff = fortran_add(
        inputs.ice_runoff_snow_capping_column,
        inputs.excess_ice_runoff_column,
    )
    mechanical_ice = subgrid.column.landunit_type == jnp.int32(LANDUNIT_GLACIER)
    conversion = (
        column_active
        & parameters.melt_non_icesheet_ice_runoff
        & ((~mechanical_ice) | topology.ice_runoff_melted_gridcell[column_gridcell])
    )
    liquid_from_ice = jnp.where(
        column_active,
        jnp.where(conversion, ice_runoff, jnp.float64(0.0)),
        state.liquid_from_ice_column,
    )
    ice_to_liquid_heat = jnp.where(
        column_active,
        jnp.where(
            conversion,
            fortran_multiply(-ice_runoff, jnp.float64(parameters.latent_heat_fusion)),
            jnp.float64(0.0),
        ),
        state.ice_to_liquid_sensible_heat_column,
    )
    remaining_ice_runoff = jnp.where(conversion, jnp.float64(0.0), ice_runoff)

    snow_water = fortran_divide(c2g(inputs.snow_water_column), jnp.float64(1000.0))
    soil_water = c2g(inputs.volumetric_soil_water_column)
    direct_albedo = p2g(inputs.direct_albedo_patch, urban=True)
    diffuse_albedo = p2g(inputs.diffuse_albedo_patch, urban=True)
    outgoing_longwave = p2g(inputs.outgoing_longwave_patch, urban=True)
    radiative_temperature = jnp.sqrt(
        jnp.sqrt(
            fortran_divide(
                outgoing_longwave,
                jnp.float64(parameters.stefan_boltzmann),
            )
        )
    )
    reference_temperature = p2g(inputs.reference_temperature_2m_patch)
    reference_humidity = p2g(inputs.reference_specific_humidity_2m_patch)
    reference_wind = p2g(inputs.reference_wind_speed_10m_patch)
    eastward_stress = p2g(inputs.eastward_stress_patch)
    northward_stress = p2g(inputs.northward_stress_patch)
    evaporation = p2g(inputs.total_evaporation_patch, urban=True)
    absorbed_solar = p2g(inputs.absorbed_solar_patch, urban=True)
    friction_velocity = p2g(inputs.friction_velocity_patch)
    aerodynamic_resistance = p2g(inputs.aerodynamic_resistance_patch)
    precipitation_heat = c2g(inputs.precipitation_conversion_sensible_heat_column)
    ice_heat_gridcell = c2g(ice_to_liquid_heat)
    sensible_heat = p2g(inputs.sensible_heat_patch, urban=True)
    sensible_heat = fortran_add(sensible_heat, precipitation_heat)
    sensible_heat = fortran_add(sensible_heat, ice_heat_gridcell)
    sensible_heat = fortran_add(sensible_heat, -inputs.dynamic_sensible_heat_gridcell)
    latent_heat = p2g(inputs.latent_heat_patch, urban=True)

    net_carbon = inputs.net_carbon_exchange_gridcell
    if parameters.use_methane and not parameters.methane_offline:
        net_carbon = fortran_add(
            net_carbon,
            inputs.net_methane_carbon_correction_gridcell,
        )
    carbon_to_co2 = np.float64(1.0e-3) * (
        (np.float64(12.0) + np.float64(2.0) * np.float64(16.0)) / np.float64(12.0)
    )
    net_carbon = fortran_multiply(net_carbon, jnp.float64(carbon_to_co2))

    dry_deposition = p2g(inputs.dry_deposition_velocity_patch)
    voc_flux = p2g(inputs.voc_flux_patch)
    fire_flux = p2g(-inputs.fire_flux_patch)
    fire_top = p2g(inputs.fire_injection_top_patch)
    dust_flux = p2g(inputs.dust_flux_patch)
    methane_flux = (
        c2g(inputs.methane_surface_flux_column, urban=False)
        if parameters.use_methane
        else jnp.zeros(gridcell_count, dtype=jnp.float64)
    )

    groundwater_runoff = jnp.where(
        column_active,
        fortran_add(
            fortran_add(state.groundwater_runoff_column, liquid_from_ice),
            inputs.rain_to_snow_runoff_conversion_column,
        ),
        state.groundwater_runoff_column,
    )
    total_runoff = jnp.where(
        column_active,
        fortran_add(
            fortran_add(state.total_runoff_column, liquid_from_ice),
            inputs.rain_to_snow_runoff_conversion_column,
        ),
        state.total_runoff_column,
    )
    surface_runoff = c2g(inputs.surface_runoff_column)
    drainage = c2g(inputs.drainage_column)
    groundwater_runoff_gridcell = fortran_add(
        c2g(groundwater_runoff),
        -inputs.dynamic_liquid_water_gridcell,
    )
    total_runoff_gridcell = fortran_add(
        c2g(total_runoff),
        -inputs.dynamic_liquid_water_gridcell,
    )
    surface_water_runoff = c2g(inputs.surface_water_runoff_column)
    perched_drainage = c2g(inputs.perched_drainage_column)
    irrigation = c2g(inputs.irrigation_column)
    ice_runoff_gridcell = fortran_add(
        c2g(remaining_ice_runoff),
        -inputs.dynamic_ice_water_gridcell,
    )

    total_water = c2g(inputs.ending_water_balance_column)
    river_depth = fortran_multiply(
        fortran_divide(
            inputs.river_volume_gridcell,
            jnp.asarray(topology.gridcell_area_square_kilometers, dtype=jnp.float64),
        ),
        jnp.float64(1.0e-3),
    )
    total_water = fortran_add(total_water, river_depth)

    return LandToAtmosphereResult(
        state=LandToAtmosphereState(
            groundwater_runoff_column=groundwater_runoff,
            total_runoff_column=total_runoff,
            liquid_from_ice_column=liquid_from_ice,
            ice_to_liquid_sensible_heat_column=ice_to_liquid_heat,
            total_water_storage_gridcell=total_water,
        ),
        fluxes=LandToAtmosphereFluxes(
            radiative_temperature_gridcell=radiative_temperature,
            reference_temperature_2m_gridcell=reference_temperature,
            reference_specific_humidity_2m_gridcell=reference_humidity,
            reference_wind_speed_10m_gridcell=reference_wind,
            snow_water_gridcell=snow_water,
            volumetric_soil_water_gridcell=soil_water,
            direct_albedo_gridcell=direct_albedo,
            diffuse_albedo_gridcell=diffuse_albedo,
            eastward_stress_gridcell=eastward_stress,
            northward_stress_gridcell=northward_stress,
            latent_heat_gridcell=latent_heat,
            sensible_heat_gridcell=sensible_heat,
            precipitation_conversion_sensible_heat_gridcell=precipitation_heat,
            outgoing_longwave_gridcell=outgoing_longwave,
            total_evaporation_gridcell=evaporation,
            absorbed_solar_gridcell=absorbed_solar,
            net_carbon_exchange_gridcell=net_carbon,
            aerodynamic_resistance_gridcell=aerodynamic_resistance,
            friction_velocity_gridcell=friction_velocity,
            dry_deposition_velocity_gridcell=dry_deposition,
            voc_flux_gridcell=voc_flux,
            fire_flux_gridcell=fire_flux,
            fire_injection_top_gridcell=fire_top,
            dust_flux_gridcell=dust_flux,
            methane_surface_flux_gridcell=methane_flux,
            liquid_runoff_gridcell=total_runoff_gridcell,
            surface_runoff_gridcell=surface_runoff,
            drainage_gridcell=drainage,
            groundwater_runoff_gridcell=groundwater_runoff_gridcell,
            surface_water_runoff_gridcell=surface_water_runoff,
            perched_drainage_gridcell=perched_drainage,
            ice_runoff_gridcell=ice_runoff_gridcell,
            irrigation_gridcell=irrigation,
        ),
        diagnostics=LandToAtmosphereDiagnostics(
            ice_to_liquid_conversion_column=conversion,
            ice_runoff_column=remaining_ice_runoff,
            ice_to_liquid_sensible_heat_gridcell=ice_heat_gridcell,
        ),
    )
