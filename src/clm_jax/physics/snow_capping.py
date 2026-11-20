
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.restart import SubgridTopology
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
    NUMBER_OF_SNOW_LEVELS,
    NUMBER_OF_SOIL_SNOW_LEVELS,
    SnowWaterAerosolState,
    build_snow_column_masks,
)

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase


MINIMUM_BOTTOM_SNOW_FRACTION = np.float64(np.float32(1.0e-9))


class SnowCappingParameters(NamedTuple):

    time_step_seconds: float = 1800.0
    maximum_snow_water_equivalent: float = 10000.0
    minimum_bottom_snow_fraction: float = float(MINIMUM_BOTTOM_SNOW_FRACTION)
    reset_snow: bool = False
    reset_glacier_snow: bool = False
    reset_glacier_equilibrium_line_altitude: float = 1.0e9

    @classmethod
    def from_case(cls, case: CimeCase) -> SnowCappingParameters:

        from clm_jax.io.namelist import read_case_namelist_group  

        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        snow_hydrology = read_case_namelist_group(
            case,
            "lnd_in",
            "clm_snowhydrology_inparm",
        )
        reset_snow = bool(snow_hydrology.get("reset_snow", False))
        reset_glacier_snow = bool(snow_hydrology.get("reset_snow_glc", False))
        if reset_snow or reset_glacier_snow:
            raise ValueError(
                "CLM-JAX snow capping currently requires reset_snow and "
                "reset_snow_glc to be false"
            )
        return cls(
            time_step_seconds=float(general["dtime"]),
            maximum_snow_water_equivalent=float(general["h2osno_max"]),
            reset_snow=reset_snow,
            reset_glacier_snow=reset_glacier_snow,
            reset_glacier_equilibrium_line_altitude=float(
                snow_hydrology.get("reset_snow_glc_ela", 1.0e9)
            ),
        )


DEFAULT_SNOW_CAPPING_PARAMETERS = SnowCappingParameters()


class SnowCappingInputs(NamedTuple):

    snow_layer_count_column: jnp.ndarray
    snow_water_equivalent_column: jnp.ndarray


class SnowCappingState(NamedTuple):

    soil_snow_liquid_column_level: jnp.ndarray
    soil_snow_ice_column_level: jnp.ndarray
    soil_snow_thickness_column_level: jnp.ndarray
    snow_capping_ice_flux_column: jnp.ndarray
    snow_capping_liquid_flux_column: jnp.ndarray
    snow_capping_discarded_ice_flux_column: jnp.ndarray
    snow_capping_discarded_liquid_flux_column: jnp.ndarray
    aerosol: SnowWaterAerosolState


class SnowCappingDiagnostics(NamedTuple):
    initialization_column: jnp.ndarray
    snow_column: jnp.ndarray
    capped_column: jnp.ndarray
    apply_runoff_column: jnp.ndarray
    excess_snow_water_column: jnp.ndarray
    removed_snow_mass_column: jnp.ndarray
    bottom_snow_ice_density_column: jnp.ndarray
    remaining_bottom_mass_fraction_column: jnp.ndarray


class SnowCappingResult(NamedTuple):
    state: SnowCappingState
    diagnostics: SnowCappingDiagnostics


def _validate_shapes(
    topology: SubgridTopology,
    inputs: SnowCappingInputs,
    state: SnowCappingState,
    parameters: SnowCappingParameters,
) -> None:
    column_shape = topology.column.gridcell.shape
    if len(column_shape) != 1:
        raise ValueError("column topology must be a vector")
    level_shape = (column_shape[0], NUMBER_OF_SOIL_SNOW_LEVELS)
    snow_shape = (column_shape[0], NUMBER_OF_SNOW_LEVELS)

    for name in SnowCappingInputs._fields:
        if getattr(inputs, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")
    for name in (
        "soil_snow_liquid_column_level",
        "soil_snow_ice_column_level",
        "soil_snow_thickness_column_level",
    ):
        if getattr(state, name).shape != level_shape:
            raise ValueError(f"{name} must have shape {level_shape}")
    for name in (
        "snow_capping_ice_flux_column",
        "snow_capping_liquid_flux_column",
        "snow_capping_discarded_ice_flux_column",
        "snow_capping_discarded_liquid_flux_column",
    ):
        if getattr(state, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")
    for name in SnowWaterAerosolState._fields[:-1]:
        if getattr(state.aerosol, name).shape != snow_shape:
            raise ValueError(f"{name} must have shape {snow_shape}")
    for name in state.aerosol.deposition._fields:
        if getattr(state.aerosol.deposition, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")

    if parameters.time_step_seconds <= 0.0:
        raise ValueError("time_step_seconds must be positive")
    if parameters.maximum_snow_water_equivalent <= 0.0:
        raise ValueError("maximum_snow_water_equivalent must be positive")
    if not 0.0 < parameters.minimum_bottom_snow_fraction < 1.0:
        raise ValueError("minimum_bottom_snow_fraction must lie strictly between 0 and 1")
    if parameters.reset_snow or parameters.reset_glacier_snow:
        raise ValueError("snow-reset branches are inactive in I2000Clm50BgcCrop")


def snow_capping_active_case(
    topology: SubgridTopology,
    inputs: SnowCappingInputs,
    state: SnowCappingState,
    *,
    parameters: SnowCappingParameters = DEFAULT_SNOW_CAPPING_PARAMETERS,
    lake_only: bool = False,
) -> SnowCappingResult:

    _validate_shapes(topology, inputs, state, parameters)
    masks = build_snow_column_masks(
        topology,
        inputs.snow_layer_count_column,
        lake_only=lake_only,
    )
    snow_column = masks.snow_column
    initialization_column = snow_column | masks.no_snow_column

    snow_water_equivalent = jnp.asarray(
        inputs.snow_water_equivalent_column,
        dtype=jnp.float64,
    )
    maximum_snow = jnp.float64(parameters.maximum_snow_water_equivalent)
    capped_column = snow_column & (snow_water_equivalent > maximum_snow)
    excess = jnp.where(
        capped_column,
        fortran_add(snow_water_equivalent, -maximum_snow),
        jnp.float64(0.0),
    )

    liquid = jnp.asarray(state.soil_snow_liquid_column_level, dtype=jnp.float64)
    ice = jnp.asarray(state.soil_snow_ice_column_level, dtype=jnp.float64)
    thickness = jnp.asarray(
        state.soil_snow_thickness_column_level,
        dtype=jnp.float64,
    )
    bottom_index = GROUND_SURFACE_INDEX - 1
    bottom_liquid = liquid[:, bottom_index]
    bottom_ice = ice[:, bottom_index]
    bottom_thickness = thickness[:, bottom_index]
    bottom_mass = fortran_add(bottom_ice, bottom_liquid)

    removable_fraction = fortran_add(
        jnp.float64(1.0),
        -jnp.float64(parameters.minimum_bottom_snow_fraction),
    )
    removable_mass = fortran_multiply(bottom_mass, removable_fraction)
    removed_mass = jnp.where(
        capped_column,
        jnp.minimum(excess, removable_mass),
        jnp.float64(0.0),
    )

    safe_bottom_mass = jnp.where(capped_column, bottom_mass, jnp.float64(1.0))
    ice_fraction = fortran_divide(bottom_ice, safe_bottom_mass)
    removal_rate = fortran_divide(
        removed_mass,
        jnp.float64(parameters.time_step_seconds),
    )
    capping_ice_flux = fortran_multiply(removal_rate, ice_fraction)
    capping_liquid_flux = fortran_multiply(
        removal_rate,
        fortran_add(jnp.float64(1.0), -ice_fraction),
    )

    def initialized_flux(values: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(
            initialization_column,
            jnp.float64(0.0),
            jnp.asarray(values, dtype=jnp.float64),
        )

    snow_capping_ice_flux = jnp.where(
        capped_column,
        capping_ice_flux,
        initialized_flux(state.snow_capping_ice_flux_column),
    )
    snow_capping_liquid_flux = jnp.where(
        capped_column,
        capping_liquid_flux,
        initialized_flux(state.snow_capping_liquid_flux_column),
    )
    discarded_ice_flux = initialized_flux(
        state.snow_capping_discarded_ice_flux_column
    )
    discarded_liquid_flux = initialized_flux(
        state.snow_capping_discarded_liquid_flux_column
    )

    time_step = jnp.float64(parameters.time_step_seconds)
    updated_bottom_ice = fortran_add(
        bottom_ice,
        -fortran_multiply(capping_ice_flux, time_step),
    )
    updated_bottom_liquid = fortran_add(
        bottom_liquid,
        -fortran_multiply(capping_liquid_flux, time_step),
    )
    ice = ice.at[:, bottom_index].set(
        jnp.where(capped_column, updated_bottom_ice, bottom_ice)
    )
    liquid = liquid.at[:, bottom_index].set(
        jnp.where(capped_column, updated_bottom_liquid, bottom_liquid)
    )

    safe_thickness = jnp.where(capped_column, bottom_thickness, jnp.float64(1.0))
    bottom_ice_density = fortran_divide(bottom_ice, safe_thickness)
    conserve_density = capped_column & (bottom_ice_density > jnp.float64(1.0))
    safe_density = jnp.where(conserve_density, bottom_ice_density, jnp.float64(1.0))
    updated_bottom_thickness = fortran_divide(updated_bottom_ice, safe_density)
    thickness = thickness.at[:, bottom_index].set(
        jnp.where(conserve_density, updated_bottom_thickness, bottom_thickness)
    )

    remaining_fraction = fortran_divide(
        jnp.where(capped_column, fortran_add(bottom_mass, -removed_mass), jnp.float64(1.0)),
        safe_bottom_mass,
    )
    aerosol_layers = []
    for aerosol_mass in state.aerosol[:-1]:
        aerosol_mass = jnp.asarray(aerosol_mass, dtype=jnp.float64)
        bottom_aerosol = aerosol_mass[:, bottom_index]
        updated_bottom_aerosol = fortran_multiply(
            bottom_aerosol,
            remaining_fraction,
        )
        aerosol_layers.append(
            aerosol_mass.at[:, bottom_index].set(
                jnp.where(capped_column, updated_bottom_aerosol, bottom_aerosol)
            )
        )
    aerosol = SnowWaterAerosolState(
        *aerosol_layers,
        deposition=state.aerosol.deposition,
    )

    return SnowCappingResult(
        state=SnowCappingState(
            soil_snow_liquid_column_level=liquid,
            soil_snow_ice_column_level=ice,
            soil_snow_thickness_column_level=thickness,
            snow_capping_ice_flux_column=snow_capping_ice_flux,
            snow_capping_liquid_flux_column=snow_capping_liquid_flux,
            snow_capping_discarded_ice_flux_column=discarded_ice_flux,
            snow_capping_discarded_liquid_flux_column=discarded_liquid_flux,
            aerosol=aerosol,
        ),
        diagnostics=SnowCappingDiagnostics(
            initialization_column=initialization_column,
            snow_column=snow_column,
            capped_column=capped_column,
            apply_runoff_column=capped_column,
            excess_snow_water_column=excess,
            removed_snow_mass_column=removed_mass,
            bottom_snow_ice_density_column=jnp.where(
                capped_column,
                bottom_ice_density,
                jnp.float64(0.0),
            ),
            remaining_bottom_mass_fraction_column=jnp.where(
                capped_column,
                remaining_fraction,
                jnp.float64(1.0),
            ),
        ),
    )
