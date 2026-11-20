
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.canopy_hydraulic_inputs import bedrock_level_count_column
from clm_jax.io.canopy_temperature_restart import (
    soil_layer_geometry_20sl_8p5m,
)
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    SubgridTopology,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)
from clm_jax.io.soil_biogeochem_potential import (
    SoilBiogeochemPotentialNativeData,
)
from clm_jax.physics.cn_zero import CnTransientFluxState
from clm_jax.physics.soil_biogeochem_litter_vertical_transport import (
    SoilBiogeochemLitterVerticalTransportInputs,
    SoilBiogeochemLitterVerticalTransportParameters,
    SoilBiogeochemLitterVerticalTransportState,
)
from clm_jax.physics.soil_biogeochem_n_state_update1 import (
    SoilBiogeochemNStateUpdate1Result,
)

NUMBER_OF_GROUND_LEVELS = 25
NUMBER_OF_DECOMPOSITION_POOLS = 7
_CARBON_RESTART_VARIABLES = (
    "litr1c_vr",
    "litr2c_vr",
    "litr3c_vr",
    "cwdc_vr",
    "soil1c_vr",
    "soil2c_vr",
    "soil3c_vr",
)
_NITROGEN_RESTART_VARIABLES = (
    "litr1n_vr",
    "litr2n_vr",
    "litr3n_vr",
    "cwdn_vr",
    "soil1n_vr",
    "soil2n_vr",
    "soil3n_vr",
)
_CWD_POOL = np.asarray(
    [False, False, False, True, False, False, False],
    dtype=np.bool_,
)


class SoilBiogeochemLitterVerticalTransportNativeData(NamedTuple):

    topology: SubgridTopology
    soil_column: jnp.ndarray
    bedrock_level_count_column: jnp.ndarray
    restart_state: SoilBiogeochemLitterVerticalTransportState
    restart_maximum_annual_thaw_depth_column: jnp.ndarray
    restart_previous_year_maximum_annual_thaw_depth_column: jnp.ndarray
    parameters: SoilBiogeochemLitterVerticalTransportParameters
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path


class SoilBiogeochemLitterVerticalTransportCall(NamedTuple):

    inputs: SoilBiogeochemLitterVerticalTransportInputs
    state: SoilBiogeochemLitterVerticalTransportState
    cn_fluxes: CnTransientFluxState
    parameters: SoilBiogeochemLitterVerticalTransportParameters


def _selected_input_path(
    case: CimeCase,
    key: str,
    override: str | Path | None,
) -> Path:
    if override is None:
        group = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        declared = group.get(key)
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError(f"clm_inparm lacks configured {key}")
        path = Path(case.expand(declared)).expanduser()
        if not path.is_absolute():
            path = case.root / path
        path = path.resolve()
    else:
        path = Path(override).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM {key} file does not exist: {path}")
    return path


def _require_same_file(
    name: str,
    requested: str | Path | None,
    actual: Path,
) -> None:
    if requested is None:
        return
    if Path(requested).expanduser().resolve() != actual:
        raise ValueError(f"potential native data uses a different {name} file")


def _finite_active(
    name: str,
    values: np.ndarray,
    soil_column: np.ndarray,
    *,
    levels: int | None = None,
) -> None:
    selected = values[soil_column]
    if levels is not None:
        selected = selected[:, :levels, ...]
    valid = np.isfinite(selected) & (np.abs(selected) < np.float64(CLM_SPECIAL_VALUE))
    if not np.all(valid):
        raise ValueError(f"active soil columns contain missing {name}")


def _load_topology_and_pools(
    restart: Path,
) -> tuple[SubgridTopology, np.ndarray, np.ndarray, np.ndarray]:
    topology = load_subgrid_topology(restart)
    active = np.asarray(topology.column.active, dtype=bool)
    landunit_type = np.asarray(topology.column.landunit_type, dtype=np.int32)
    soil_column = active & np.isin(landunit_type, (LANDUNIT_SOIL, LANDUNIT_CROP))
    column_count = soil_column.size
    variables = load_native_variables(
        restart,
        [*_CARBON_RESTART_VARIABLES, *_NITROGEN_RESTART_VARIABLES],
        dtype=np.float64,
        masked_floating_fill_value=float(CLM_SPECIAL_VALUE),
    )
    for name, values in variables.items():
        if values.shape != (column_count, NUMBER_OF_GROUND_LEVELS):
            raise ValueError(f"restart {name} must have shape (column, 25)")
    carbon = np.stack(
        [np.asarray(variables[name]) for name in _CARBON_RESTART_VARIABLES],
        axis=2,
    )
    nitrogen = np.stack(
        [np.asarray(variables[name]) for name in _NITROGEN_RESTART_VARIABLES],
        axis=2,
    )
    _finite_active("decomposition carbon pools", carbon, soil_column, levels=20)
    _finite_active("decomposition nitrogen pools", nitrogen, soil_column, levels=20)
    return topology, soil_column, carbon, nitrogen


def load_soil_biogeochem_litter_vertical_transport_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    potential_native_data: SoilBiogeochemPotentialNativeData | None = None,
) -> SoilBiogeochemLitterVerticalTransportNativeData:

    general = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    required = {
        "use_cn": True,
        "use_century_decomp": True,
        "use_vertsoilc": True,
        "use_bedrock": True,
        "use_fates": False,
        "spinup_state": 0,
        "soil_layerstruct": "20SL_8.5m",
    }
    for name, expected in required.items():
        if general.get(name) != expected:
            raise ValueError(
                "native litter vertical-transport adapter requires "
                f"{name}={expected!r}"
            )

    if potential_native_data is None:
        restart = (
            configured_land_restart(case)
            if restart_path is None
            else Path(restart_path).expanduser().resolve()
        )
        if not restart.is_file():
            raise FileNotFoundError(f"CLM restart does not exist: {restart}")
        surface_data = _selected_input_path(case, "fsurdat", surface_data_path)
        parameter_data = _selected_input_path(case, "paramfile", parameter_data_path)
        topology, soil_column, carbon, nitrogen = _load_topology_and_pools(restart)
    else:
        potential = potential_native_data
        _require_same_file("restart", restart_path, potential.restart_path)
        _require_same_file("surface-data", surface_data_path, potential.surface_data_path)
        _require_same_file(
            "parameter-data", parameter_data_path, potential.parameter_data_path
        )
        restart = potential.restart_path
        surface_data = potential.surface_data_path
        parameter_data = potential.parameter_data_path
        topology = potential.topology
        soil_column = np.asarray(potential.soil_column, dtype=bool)
        carbon = np.asarray(
            potential.pools.decomposition_carbon_pool_column_level_pool,
            dtype=np.float64,
        )
        nitrogen = np.asarray(
            potential.pools.decomposition_nitrogen_pool_column_level_pool,
            dtype=np.float64,
        )

    dimensions = load_dimension_sizes(restart)
    if dimensions.get("levgrnd") != NUMBER_OF_GROUND_LEVELS:
        raise ValueError("vertical-transport restart must contain 25 ground levels")
    column_count = soil_column.size
    persistent = load_native_variables(
        restart,
        [
            "altmax",
            "altmax_lastyear",
            "som_adv_coef_vr",
            "som_diffus_coef_vr",
        ],
        dtype=np.float64,
        masked_floating_fill_value=float(CLM_SPECIAL_VALUE),
    )
    altmax = np.asarray(persistent["altmax"], dtype=np.float64)
    altmax_lastyear = np.asarray(
        persistent["altmax_lastyear"], dtype=np.float64
    )
    advection = np.asarray(persistent["som_adv_coef_vr"], dtype=np.float64)
    diffusivity = np.asarray(persistent["som_diffus_coef_vr"], dtype=np.float64)
    if altmax.shape != (column_count,) or altmax_lastyear.shape != (column_count,):
        raise ValueError("restart active-layer fields must contain every column")
    if advection.shape != (column_count, 25) or diffusivity.shape != (column_count, 25):
        raise ValueError("restart SOM coefficient fields must have shape (column, 25)")
    _finite_active("altmax", altmax, soil_column)
    _finite_active("altmax_lastyear", altmax_lastyear, soil_column)
    _finite_active("som_adv_coef_vr", advection, soil_column)
    _finite_active("som_diffus_coef_vr", diffusivity, soil_column)

    raw_parameters = load_native_variables(
        parameter_data,
        ["som_diffus", "cryoturb_diffusion_k", "max_altdepth_cryoturbation"],
        dtype=np.float64,
    )
    for name, value in raw_parameters.items():
        scalar = np.asarray(value, dtype=np.float64)
        if scalar.size != 1 or not np.isfinite(scalar.reshape(())):
            raise ValueError(f"CLM parameter {name!r} must be one finite scalar")
    maximum_active_layer = np.float64(
        np.asarray(raw_parameters["max_altdepth_cryoturbation"]).reshape(())
    )
    clock = load_restart_clock(restart)
    if clock.step_seconds <= 0:
        raise ValueError("restart time step must be positive")
    node, interface, thickness = soil_layer_geometry_20sl_8p5m()
    bedrock = bedrock_level_count_column(topology, surface_data)
    if bedrock.shape != (column_count,) or np.any((bedrock < 1) | (bedrock > 20)):
        raise ValueError("native bedrock level must lie in [1, 20]")

    return SoilBiogeochemLitterVerticalTransportNativeData(
        topology=topology,
        soil_column=jnp.asarray(soil_column),
        bedrock_level_count_column=jnp.asarray(bedrock, dtype=jnp.int32),
        restart_state=SoilBiogeochemLitterVerticalTransportState(
            carbon_pool_column_level_pool=jnp.asarray(carbon),
            nitrogen_pool_column_level_pool=jnp.asarray(nitrogen),
            som_advective_coefficient_column_level=jnp.asarray(advection),
            som_diffusivity_coefficient_column_level=jnp.asarray(diffusivity),
        ),
        restart_maximum_annual_thaw_depth_column=jnp.asarray(altmax),
        restart_previous_year_maximum_annual_thaw_depth_column=jnp.asarray(
            altmax_lastyear
        ),
        parameters=SoilBiogeochemLitterVerticalTransportParameters(
            time_step=np.float64(clock.step_seconds),
            som_advective_flux=np.float64(0.0),
            som_diffusivity=np.float64(1.0e-4)
            / (np.float64(86400.0) * np.float64(365.0)),
            cryoturbation_diffusivity=np.float64(5.0e-4)
            / (np.float64(86400.0) * np.float64(365.0)),
            maximum_active_layer_depth_for_cryoturbation=maximum_active_layer,
            maximum_cryoturbation_depth=np.float64(3.0),
            epsilon=np.float64(np.float32(1.0e-30)),
            soil_node_depth_level=jnp.asarray(node),
            soil_interface_depth_level=jnp.asarray(interface),
            decomposition_layer_thickness_level=jnp.asarray(thickness),
            is_coarse_woody_debris_pool=jnp.asarray(_CWD_POOL),
        ),
        restart_path=restart,
        surface_data_path=surface_data,
        parameter_data_path=parameter_data,
    )


def assemble_soil_biogeochem_litter_vertical_transport_call(
    native: SoilBiogeochemLitterVerticalTransportNativeData,
    nitrogen_state_update_result: SoilBiogeochemNStateUpdate1Result,
    maximum_annual_thaw_depth_column: jnp.ndarray,
    previous_year_maximum_annual_thaw_depth_column: jnp.ndarray,
    *,
    current_state: SoilBiogeochemLitterVerticalTransportState | None = None,
) -> SoilBiogeochemLitterVerticalTransportCall:

    column_count = native.soil_column.shape[0]
    altmax = jnp.asarray(maximum_annual_thaw_depth_column, dtype=jnp.float64)
    altmax_lastyear = jnp.asarray(
        previous_year_maximum_annual_thaw_depth_column,
        dtype=jnp.float64,
    )
    if altmax.shape != (column_count,) or altmax_lastyear.shape != (column_count,):
        raise ValueError("same-step active-layer fields have the wrong column count")
    cn_fluxes = nitrogen_state_update_result.cn_fluxes
    if cn_fluxes.column_scalar.shape[0] != column_count:
        raise ValueError("soil-N Update1 result has the wrong column count")
    if cn_fluxes.column_level_pool.shape[:3] != (column_count, 25, 7):
        raise ValueError("soil-N Update1 pool flux extent is incompatible")

    state = native.restart_state if current_state is None else current_state
    expected_pool_shape = (column_count, 25, 7)
    expected_level_shape = (column_count, 25)
    for name in (
        "carbon_pool_column_level_pool",
        "nitrogen_pool_column_level_pool",
    ):
        if jnp.shape(getattr(state, name)) != expected_pool_shape:
            raise ValueError(f"current {name} has the wrong column count")
    for name in (
        "som_advective_coefficient_column_level",
        "som_diffusivity_coefficient_column_level",
    ):
        if jnp.shape(getattr(state, name)) != expected_level_shape:
            raise ValueError(f"current {name} has the wrong column count")

    return SoilBiogeochemLitterVerticalTransportCall(
        inputs=SoilBiogeochemLitterVerticalTransportInputs(
            soil_column=native.soil_column,
            bedrock_level_count_column=native.bedrock_level_count_column,
            maximum_annual_thaw_depth_column=altmax,
            previous_year_maximum_annual_thaw_depth_column=altmax_lastyear,
        ),
        state=state,
        cn_fluxes=cn_fluxes,
        parameters=native.parameters,
    )


__all__ = [
    "SoilBiogeochemLitterVerticalTransportCall",
    "SoilBiogeochemLitterVerticalTransportNativeData",
    "assemble_soil_biogeochem_litter_vertical_transport_call",
    "load_soil_biogeochem_litter_vertical_transport_native_data",
]
