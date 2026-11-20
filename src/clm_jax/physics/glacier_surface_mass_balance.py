
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import lax

from clm_jax.io.restart import LANDUNIT_GLACIER, SubgridTopology
from clm_jax.physics.numerics import (
    fortran_add,
    fortran_divide,
    fortran_multiply,
)

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase

NUMBER_OF_GROUND_LEVELS = 25
SECONDS_PER_DAY = 86400.0


class GlacierIceMeltInputs(NamedTuple):

    do_surface_mass_balance: jnp.ndarray
    landunit_type: jnp.ndarray
    time_step_seconds: jnp.ndarray


class GlacierIceMeltState(NamedTuple):

    liquid_water: jnp.ndarray
    ice_water: jnp.ndarray
    ice_melt_flux: jnp.ndarray


class GlacierIceMeltResult(NamedTuple):

    state: GlacierIceMeltState
    converted_layer: jnp.ndarray


class GlacierSurfaceMassBalanceParameters(NamedTuple):

    snow_persistence_max_days: int = 0

    @classmethod
    def from_case(cls, case: CimeCase) -> GlacierSurfaceMassBalanceParameters:

        from clm_jax.io.namelist import read_case_namelist_group  

        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        return cls(
            snow_persistence_max_days=int(
                general["glc_snow_persistence_max_days"]
            )
        )


DEFAULT_GLACIER_SURFACE_MASS_BALANCE_PARAMETERS = (
    GlacierSurfaceMassBalanceParameters()
)


class GlacierSurfaceMassBalanceInputs(NamedTuple):

    do_surface_mass_balance_column: jnp.ndarray
    dynamic_runoff_routing_gridcell: jnp.ndarray
    snow_persistence_column: jnp.ndarray
    snow_capping_ice_flux_column: jnp.ndarray


class GlacierSurfaceMassBalanceState(NamedTuple):

    ice_growth_flux_column: jnp.ndarray
    ice_melt_flux_column: jnp.ndarray
    net_ice_flux_column: jnp.ndarray
    dynamic_water_flux_column: jnp.ndarray


class GlacierSurfaceMassBalanceDiagnostics(NamedTuple):

    all_active_column: jnp.ndarray
    persistent_snow_or_glacier_column: jnp.ndarray
    dynamic_runoff_routing_column: jnp.ndarray


class GlacierSurfaceMassBalanceResult(NamedTuple):

    state: GlacierSurfaceMassBalanceState
    diagnostics: GlacierSurfaceMassBalanceDiagnostics


def _validate_glacier_ice_melt_shapes(
    inputs: GlacierIceMeltInputs,
    state: GlacierIceMeltState,
) -> None:
    if inputs.do_surface_mass_balance.ndim != 1:
        raise ValueError("do_surface_mass_balance must be a vector")
    column_shape = inputs.do_surface_mass_balance.shape
    layer_shape = (column_shape[0], NUMBER_OF_GROUND_LEVELS)
    if inputs.landunit_type.shape != column_shape:
        raise ValueError(f"landunit_type must have shape {column_shape}")
    if inputs.time_step_seconds.shape != ():
        raise ValueError("time_step_seconds must be a scalar")
    if state.liquid_water.shape != layer_shape:
        raise ValueError(f"liquid_water must have shape {layer_shape}")
    if state.ice_water.shape != layer_shape:
        raise ValueError(f"ice_water must have shape {layer_shape}")
    if state.ice_melt_flux.shape != column_shape:
        raise ValueError(f"ice_melt_flux must have shape {column_shape}")


def handle_glacier_ice_melt(
    inputs: GlacierIceMeltInputs,
    state: GlacierIceMeltState,
) -> GlacierIceMeltResult:

    _validate_glacier_ice_melt_shapes(inputs, state)

    do_surface_mass_balance = jnp.asarray(
        inputs.do_surface_mass_balance,
        dtype=jnp.bool_,
    )
    landunit_type = jnp.asarray(inputs.landunit_type, dtype=jnp.int32)
    time_step_seconds = jnp.asarray(inputs.time_step_seconds, dtype=jnp.float64)
    liquid_water = jnp.asarray(state.liquid_water, dtype=jnp.float64)
    ice_water = jnp.asarray(state.ice_water, dtype=jnp.float64)
    previous_melt_flux = jnp.asarray(state.ice_melt_flux, dtype=jnp.float64)

    glacier_column = landunit_type == jnp.int32(LANDUNIT_GLACIER)
    melt_flux = jnp.where(
        do_surface_mass_balance,
        jnp.float64(0.0),
        previous_melt_flux,
    )
    converted_layer = (
        (do_surface_mass_balance & glacier_column)[:, None]
        & (liquid_water > jnp.float64(0.0))
    )
    level_melt = fortran_divide(liquid_water, time_step_seconds)

    def accumulate_melt(level, melt):
        return jnp.where(
            converted_layer[:, level], fortran_add(melt, level_melt[:, level]), melt,
        )

    melt_flux = lax.fori_loop(
        0,
        NUMBER_OF_GROUND_LEVELS,
        accumulate_melt,
        melt_flux,
    )
    ice_water = jnp.where(converted_layer, fortran_add(ice_water, liquid_water), ice_water)
    liquid_water = jnp.where(converted_layer, jnp.float64(0.0), liquid_water)
    return GlacierIceMeltResult(
        state=GlacierIceMeltState(
            liquid_water=liquid_water,
            ice_water=ice_water,
            ice_melt_flux=melt_flux,
        ),
        converted_layer=converted_layer,
    )


def _validate_surface_mass_balance_shapes(
    topology: SubgridTopology,
    inputs: GlacierSurfaceMassBalanceInputs,
    state: GlacierSurfaceMassBalanceState,
) -> None:
    column_shape = topology.column.gridcell.shape
    if len(column_shape) != 1:
        raise ValueError("column topology must be a vector")
    if topology.column.active.shape != column_shape:
        raise ValueError(f"column active mask must have shape {column_shape}")
    if topology.column.landunit_type.shape != column_shape:
        raise ValueError(f"column landunit type must have shape {column_shape}")
    if inputs.do_surface_mass_balance_column.shape != column_shape:
        raise ValueError(
            "do_surface_mass_balance_column must have shape "
            f"{column_shape}"
        )
    if inputs.dynamic_runoff_routing_gridcell.ndim != 1:
        raise ValueError("dynamic_runoff_routing_gridcell must be a vector")
    for name in ("snow_persistence_column", "snow_capping_ice_flux_column"):
        if getattr(inputs, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")
    for name in GlacierSurfaceMassBalanceState._fields:
        if getattr(state, name).shape != column_shape:
            raise ValueError(f"{name} must have shape {column_shape}")


def compute_glacier_surface_mass_balance(
    topology: SubgridTopology,
    inputs: GlacierSurfaceMassBalanceInputs,
    state: GlacierSurfaceMassBalanceState,
    parameters: GlacierSurfaceMassBalanceParameters = (
        DEFAULT_GLACIER_SURFACE_MASS_BALANCE_PARAMETERS
    ),
) -> GlacierSurfaceMassBalanceResult:

    _validate_surface_mass_balance_shapes(topology, inputs, state)

    all_active = jnp.asarray(topology.column.active, dtype=jnp.bool_)
    do_surface_mass_balance = jnp.asarray(
        inputs.do_surface_mass_balance_column,
        dtype=jnp.bool_,
    )
    landunit_type = jnp.asarray(topology.column.landunit_type, dtype=jnp.int32)
    column_gridcell = jnp.asarray(topology.column.gridcell, dtype=jnp.int32)
    dynamic_runoff_routing_gridcell = jnp.asarray(
        inputs.dynamic_runoff_routing_gridcell,
        dtype=jnp.float64,
    )
    snow_persistence = jnp.asarray(
        inputs.snow_persistence_column,
        dtype=jnp.float64,
    )
    snow_capping_ice_flux = jnp.asarray(
        inputs.snow_capping_ice_flux_column,
        dtype=jnp.float64,
    )
    previous_ice_growth = jnp.asarray(
        state.ice_growth_flux_column,
        dtype=jnp.float64,
    )
    ice_melt = jnp.asarray(state.ice_melt_flux_column, dtype=jnp.float64)
    previous_net_ice = jnp.asarray(state.net_ice_flux_column, dtype=jnp.float64)
    previous_dynamic_water = jnp.asarray(
        state.dynamic_water_flux_column,
        dtype=jnp.float64,
    )

    persistence_seconds = fortran_multiply(
        jnp.asarray(parameters.snow_persistence_max_days, dtype=jnp.float64),
        jnp.float64(SECONDS_PER_DAY),
    )
    persistent_snow_or_glacier = (
        snow_persistence >= persistence_seconds
    ) | (landunit_type == jnp.int32(LANDUNIT_GLACIER))
    candidate_ice_growth = jnp.where(
        persistent_snow_or_glacier,
        snow_capping_ice_flux,
        jnp.float64(0.0),
    )
    ice_growth = jnp.where(
        do_surface_mass_balance,
        candidate_ice_growth,
        previous_ice_growth,
    )
    candidate_net_ice = fortran_add(candidate_ice_growth, -ice_melt)
    net_ice = jnp.where(
        do_surface_mass_balance,
        candidate_net_ice,
        previous_net_ice,
    )

    dynamic_runoff_routing_column = dynamic_runoff_routing_gridcell[
        column_gridcell
    ]
    candidate_dynamic_water = fortran_multiply(
        dynamic_runoff_routing_column,
        fortran_add(ice_melt, -candidate_ice_growth),
    )
    dynamic_water = jnp.where(
        all_active,
        jnp.float64(0.0),
        previous_dynamic_water,
    )
    dynamic_water = jnp.where(
        do_surface_mass_balance,
        candidate_dynamic_water,
        dynamic_water,
    )

    return GlacierSurfaceMassBalanceResult(
        state=GlacierSurfaceMassBalanceState(
            ice_growth_flux_column=ice_growth,
            ice_melt_flux_column=ice_melt,
            net_ice_flux_column=net_ice,
            dynamic_water_flux_column=dynamic_water,
        ),
        diagnostics=GlacierSurfaceMassBalanceDiagnostics(
            all_active_column=all_active,
            persistent_snow_or_glacier_column=persistent_snow_or_glacier,
            dynamic_runoff_routing_column=dynamic_runoff_routing_column,
        ),
    )
