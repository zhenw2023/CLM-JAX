
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    SubgridTopology,
    configured_land_restart,
    load_subgrid_topology,
)
from clm_jax.physics.luna import (
    LunaCapacityParameters,
    LunaCapacityState,
    LunaDailyClimateState,
    LunaTenDayClimateState,
)

_PARAMETER_VARIABLES = ("c3psn", "slatop", "rholvis", "taulvis")

_RESTART_SCALAR_FLOAT_VARIABLES = (
    "gpp24",
    "tvegd",
    "tvegn",
    "tvegd10",
    "tvegn10",
    "rh10",
    "rb10",
    "pbot240",
    "pco2_240",
    "po2_240",
    "tlai",
    "lnca",
    "tair10",
)
_RESTART_SCALAR_INTEGER_VARIABLES = ("ndaysteps", "nnightsteps", "nrad")
_RESTART_LAYER_VARIABLES = (
    "par24d",
    "par24x",
    "par240d",
    "par240x",
    "vcmx25_z",
    "jmx25_z",
    "pnlc_z",
    "enzs_z",
    "tlai_z",
)
_RESTART_VARIABLES = (
    *_RESTART_SCALAR_FLOAT_VARIABLES,
    *_RESTART_SCALAR_INTEGER_VARIABLES,
    *_RESTART_LAYER_VARIABLES,
)


class LunaCapacityRestartInputs(NamedTuple):

    atmospheric_pressure_ten_day_patch: jnp.ndarray
    atmospheric_co2_ten_day_patch: jnp.ndarray
    atmospheric_o2_ten_day_patch: jnp.ndarray
    total_leaf_area_patch: jnp.ndarray
    total_leaf_area_by_layer_patch: jnp.ndarray
    radiative_layer_count_patch: jnp.ndarray
    leaf_nitrogen_concentration_patch: jnp.ndarray
    air_temperature_ten_day_patch: jnp.ndarray


class LunaRestartData(NamedTuple):

    topology: SubgridTopology
    daily: LunaDailyClimateState
    ten_day: LunaTenDayClimateState
    capacity: LunaCapacityState
    capacity_inputs: LunaCapacityRestartInputs


def _finite_restart_float(value: jnp.ndarray) -> jnp.ndarray:
    value = jnp.asarray(value, dtype=jnp.float64)
    return jnp.where(jnp.isnan(value), jnp.float64(CLM_SPECIAL_VALUE), value)


def _configured_parameter_path(case: CimeCase) -> Path:
    group = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    declared = group.get("paramfile")
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(f"clm_inparm in {case.case_docs / 'lnd_in'} lacks paramfile")
    path = Path(case.expand(declared)).expanduser()
    if not path.is_absolute():
        path = case.root / path
    return path.resolve()


def load_luna_restart(
    case: CimeCase,
    restart_path: str | Path | None = None,
) -> LunaRestartData:

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")

    dimensions = load_dimension_sizes(restart)
    if "pft" not in dimensions or "levcan" not in dimensions:
        raise ValueError(f"restart lacks native pft/levcan dimensions: {restart}")
    number_of_patches = dimensions["pft"]
    number_of_layers = dimensions["levcan"]
    if number_of_patches < 1 or number_of_layers < 1:
        raise ValueError("restart must contain at least one patch and canopy layer")

    topology = load_subgrid_topology(restart)
    if topology.patch.gridcell.shape != (number_of_patches,):
        raise ValueError("restart topology and pft dimension disagree")
    values = load_native_variables(
        restart,
        list(_RESTART_VARIABLES),
        masked_integer_fill_value=-9999,
    )
    scalar_shape = (number_of_patches,)
    layer_shape = (number_of_patches, number_of_layers)
    for name in (*_RESTART_SCALAR_FLOAT_VARIABLES, *_RESTART_SCALAR_INTEGER_VARIABLES):
        if values[name].shape != scalar_shape:
            raise ValueError(f"restart variable {name!r} has unexpected shape {values[name].shape}")
    for name in _RESTART_LAYER_VARIABLES:
        if values[name].shape != layer_shape:
            raise ValueError(f"restart variable {name!r} has unexpected shape {values[name].shape}")

    real = {
        name: _finite_restart_float(values[name])
        for name in (*_RESTART_SCALAR_FLOAT_VARIABLES, *_RESTART_LAYER_VARIABLES)
    }
    integer = {
        name: jnp.asarray(values[name], dtype=jnp.int32)
        for name in _RESTART_SCALAR_INTEGER_VARIABLES
    }
    return LunaRestartData(
        topology=topology,
        daily=LunaDailyClimateState(
            absorbed_par_sum_by_layer_patch=real["par24d"],
            absorbed_par_maximum_by_layer_patch=real["par24x"],
            photosynthesis_sum_patch=real["gpp24"],
            daytime_temperature_sum_patch=real["tvegd"],
            nighttime_temperature_sum_patch=real["tvegn"],
            nighttime_step_count_patch=integer["nnightsteps"],
            daytime_step_count_patch=integer["ndaysteps"],
        ),
        ten_day=LunaTenDayClimateState(
            absorbed_par_mean_by_layer_patch=real["par240d"],
            absorbed_par_maximum_by_layer_patch=real["par240x"],
            daytime_temperature_patch=real["tvegd10"],
            nighttime_temperature_patch=real["tvegn10"],
            relative_humidity_patch=real["rh10"],
            boundary_layer_resistance_patch=real["rb10"],
        ),
        capacity=LunaCapacityState(
            maximum_carboxylation_rate_by_layer_patch=real["vcmx25_z"],
            maximum_electron_transport_rate_by_layer_patch=real["jmx25_z"],
            light_capture_fraction_by_layer_patch=real["pnlc_z"],
            enzyme_activity_by_layer_patch=real["enzs_z"],
        ),
        capacity_inputs=LunaCapacityRestartInputs(
            atmospheric_pressure_ten_day_patch=real["pbot240"],
            atmospheric_co2_ten_day_patch=real["pco2_240"],
            atmospheric_o2_ten_day_patch=real["po2_240"],
            total_leaf_area_patch=real["tlai"],
            total_leaf_area_by_layer_patch=real["tlai_z"],
            radiative_layer_count_patch=integer["nrad"],
            leaf_nitrogen_concentration_patch=real["lnca"],
            air_temperature_ten_day_patch=real["tair10"],
        ),
    )


def load_luna_capacity_parameters(
    case: CimeCase,
    parameter_data_path: str | Path | None = None,
) -> LunaCapacityParameters:

    path = (
        _configured_parameter_path(case)
        if parameter_data_path is None
        else Path(parameter_data_path).expanduser().resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM parameter file does not exist: {path}")
    loaded = load_native_variables(path, list(_PARAMETER_VARIABLES), dtype=np.float64)

    arrays = tuple(np.asarray(loaded[name]) for name in _PARAMETER_VARIABLES)
    if any(value.ndim != 1 for value in arrays):
        raise ValueError("CLM LUNA PFT parameters must be one-dimensional")
    pft_shape = arrays[0].shape
    if pft_shape[0] < 1 or any(value.shape != pft_shape for value in arrays):
        raise ValueError("CLM LUNA PFT parameter dimensions disagree")
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("CLM LUNA PFT parameters must be finite")

    c3_pathway, specific_leaf_area, reflectance, transmittance = arrays
    if not np.all((c3_pathway == 0.0) | (c3_pathway == 1.0)):
        raise ValueError("CLM c3psn must contain only zero or one")
    if np.any(specific_leaf_area < 0.0):
        raise ValueError("CLM slatop must be nonnegative")
    if np.any(1.0 - reflectance - transmittance <= 0.0):
        raise ValueError("CLM visible leaf absorptance must be positive")

    luna_group = read_case_namelist_group(case, "lnd_in", "luna")
    jmax_light_response = luna_group.get("jmaxb1")
    if (
        isinstance(jmax_light_response, bool)
        or not isinstance(jmax_light_response, int | float)
        or not np.isfinite(jmax_light_response)
    ):
        raise ValueError(f"luna in {case.case_docs / 'lnd_in'} lacks finite scalar jmaxb1")

    return LunaCapacityParameters(
        c3_pathway_pft=jnp.asarray(c3_pathway == 1.0),
        top_specific_leaf_area_pft=jnp.asarray(specific_leaf_area),
        visible_leaf_reflectance_pft=jnp.asarray(reflectance),
        visible_leaf_transmittance_pft=jnp.asarray(transmittance),
        jmax_light_response=jnp.float64(jmax_light_response),
    )
