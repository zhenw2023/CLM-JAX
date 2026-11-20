
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.forcing.atmosphere import (
    Atm2LndParameters,
    ColumnAtmosphericForcing,
    downscale_atmospheric_forcing_to_columns,
)
from clm_jax.forcing.datm import DatmForcing
from clm_jax.io.restart import LANDUNIT_LAKE, SubgridTopology
from clm_jax.physics.daylength import DaylengthState, update_daylength
from clm_jax.physics.driver import DriverInitializationResult, initialize_driver_state
from clm_jax.physics.filters import ExposedVegetationMasks, exposed_vegetation_masks
from clm_jax.physics.solar import FIXED_YEAR_2000_ORBIT, solar_declination
from clm_jax.physics.topography import (
    ColumnTopography,
    GlacierLandFields,
    update_column_topography,
)
from clm_jax.trajectory import NoleapModelClock, noleap_calendar_day_from_clock

NUMBER_OF_SNOW_LEVELS = 12


class AtmosphericDriverStatic(NamedTuple):

    topology: SubgridTopology
    latitude_radians: jnp.ndarray
    atmospheric_elevation: jnp.ndarray
    glacier: GlacierLandFields
    collapse_to_atmospheric_topography: jnp.ndarray
    rain_to_snow_runs_off: jnp.ndarray
    parameters: Atm2LndParameters


class AtmosphericDriverInputs(NamedTuple):

    intracellular_sunlit_co2: jnp.ndarray
    intracellular_shaded_co2: jnp.ndarray
    snow_water_equivalent: jnp.ndarray
    snow_layer_count: jnp.ndarray
    snow_layer_ice: jnp.ndarray
    snow_layer_liquid: jnp.ndarray
    vegetation_exposed_for_albedo: jnp.ndarray


class AtmosphericDriverState(NamedTuple):

    daylength: DaylengthState
    previous_snow_ice_fraction: jnp.ndarray
    column_topography: ColumnTopography
    first_step_of_run_segment: jnp.ndarray


class AtmosphericDriverResult(NamedTuple):

    state: AtmosphericDriverState
    driver_initialization: DriverInitializationResult
    column_forcing: ColumnAtmosphericForcing
    exposed_vegetation: ExposedVegetationMasks


def atmospheric_driver_step(
    static: AtmosphericDriverStatic,
    state: AtmosphericDriverState,
    inputs: AtmosphericDriverInputs,
    forcing: DatmForcing,
    clock: NoleapModelClock,
) -> AtmosphericDriverResult:

    calendar_day = noleap_calendar_day_from_clock(clock)
    declination, _ = solar_declination(calendar_day, FIXED_YEAR_2000_ORBIT)
    daylength = update_daylength(
        state.daylength,
        static.latitude_radians,
        declination,
        FIXED_YEAR_2000_ORBIT.obliquity_radians,
        first_step_of_run_segment=state.first_step_of_run_segment,
    )

    topology = static.topology
    driver = initialize_driver_state(
        inputs.intracellular_sunlit_co2,
        inputs.intracellular_shaded_co2,
        inputs.snow_water_equivalent,
        inputs.snow_layer_count,
        inputs.snow_layer_ice,
        inputs.snow_layer_liquid,
        state.previous_snow_ice_fraction,
        topology.patch.active,
        inputs.vegetation_exposed_for_albedo,
        topology.column.active
        & (topology.column.landunit_type != jnp.int32(LANDUNIT_LAKE)),
        number_of_snow_levels=NUMBER_OF_SNOW_LEVELS,
    )

    current_column = topology.column._replace(
        elevation=state.column_topography.elevation,
        column_type=state.column_topography.column_type,
    )
    column_topography = update_column_topography(
        state.column_topography.elevation,
        static.atmospheric_elevation,
        current_column,
        static.glacier,
        static.collapse_to_atmospheric_topography,
        glc_do_dynglacier=True,
    )
    updated_column = current_column._replace(
        elevation=column_topography.elevation,
        column_type=column_topography.column_type,
    )
    column_forcing = downscale_atmospheric_forcing_to_columns(
        forcing,
        static.atmospheric_elevation,
        column_topography.elevation,
        updated_column,
        column_topography.needs_downscaling,
        static.rain_to_snow_runs_off,
        static.parameters,
    )
    masks = exposed_vegetation_masks(
        topology.patch.landunit_type,
        topology.patch.active,
        driver.vegetation_exposed,
    )
    return AtmosphericDriverResult(
        state=AtmosphericDriverState(
            daylength=daylength,
            previous_snow_ice_fraction=driver.previous_snow_ice_fraction,
            column_topography=column_topography,
            first_step_of_run_segment=jnp.asarray(False, dtype=jnp.bool_),
        ),
        driver_initialization=driver,
        column_forcing=column_forcing,
        exposed_vegetation=masks,
    )


__all__ = [
    "AtmosphericDriverInputs",
    "AtmosphericDriverResult",
    "AtmosphericDriverState",
    "AtmosphericDriverStatic",
    "NUMBER_OF_SNOW_LEVELS",
    "atmospheric_driver_step",
]
