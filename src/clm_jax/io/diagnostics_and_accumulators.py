
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
    RestartClock,
    SubgridTopology,
    configured_land_restart,
    load_restart_clock,
    load_subgrid_topology,
)
from clm_jax.physics.accumulators import AccumulatorState
from clm_jax.physics.diagnostics_and_accumulators import (
    AtmosphereAccumulatorState,
    CanopyAccumulatorState,
    ConfiguredAccumulatorParameters,
    ConfiguredAccumulatorState,
    CropAccumulatorState,
    DailyExtremaState,
    TemperatureAccumulatorState,
    WaterEnergyAccumulatorState,
)


class DiagnosticsAndAccumulatorsRestartData(NamedTuple):

    topology: SubgridTopology
    clock: RestartClock
    state: ConfiguredAccumulatorState
    parameters: ConfiguredAccumulatorParameters
    restart_path: Path
    parameter_data_path: Path


_ATMOSPHERE_FIELDS = (
    ("direct_solar_24_hour", "FSD24"),
    ("direct_solar_240_hour", "FSD240"),
    ("diffuse_solar_24_hour", "FSI24"),
    ("diffuse_solar_240_hour", "FSI240"),
    ("precipitation_10_day", "PREC10"),
    ("precipitation_60_day", "PREC60"),
    ("relative_humidity_30_day", "RH30"),
    ("co2_partial_pressure_240_hour", "pco2_240"),
    ("oxygen_partial_pressure_240_hour", "po2_240"),
    ("air_pressure_240_hour", "pbot240"),
)
_TEMPERATURE_FIELDS = (
    ("vegetation_temperature_24_hour", "T_VEG24"),
    ("vegetation_temperature_240_hour", "T_VEG240"),
    ("reference_temperature_hourly", "TREFAV"),
    ("urban_reference_temperature_hourly", "TREFAV_U"),
    ("rural_reference_temperature_hourly", "TREFAV_R"),
    ("reference_temperature_10_day", "T10"),
    ("minimum_reference_temperature_10_day", "TDM10"),
    ("minimum_reference_temperature_5_day", "TDM5"),
    ("growing_degree_days_base_0", "GDD0"),
    ("growing_degree_days_base_8", "GDD8"),
    ("growing_degree_days_base_10", "GDD10"),
)
_CANOPY_FIELDS = (
    ("sunlit_fraction_24_hour", "FSUN24"),
    ("sunlit_fraction_240_hour", "FSUN240"),
    ("exposed_leaf_area_index_240_hour", "LAI240"),
)
_WATER_ENERGY_FIELDS = (
    ("annual_evapotranspiration", "AnnET"),
    ("hourly_transpiration_stress", "BTRANAV"),
)
_CROP_FIELDS = (
    ("growing_degree_days_plant", "GDDPLANT"),
    ("growing_degree_days_soil", "GDDTSOI"),
)
_PATCH_ACCUMULATOR_NAMES = {
    native
    for fields in (
        _ATMOSPHERE_FIELDS,
        _TEMPERATURE_FIELDS,
        _CANOPY_FIELDS,
        _WATER_ENERGY_FIELDS,
        _CROP_FIELDS,
    )
    for python_name, native in fields
    if python_name != "annual_evapotranspiration"
}
_DAILY_FIELDS = (
    ("reference_temperature_maximum_patch", "T_REF2M_MAX"),
    ("reference_temperature_minimum_patch", "T_REF2M_MIN"),
    (
        "reference_temperature_maximum_instantaneous_patch",
        "T_REF2M_MAX_INST",
    ),
    (
        "reference_temperature_minimum_instantaneous_patch",
        "T_REF2M_MIN_INST",
    ),
    ("urban_reference_temperature_maximum_patch", "T_REF2M_MAX_U"),
    ("urban_reference_temperature_minimum_patch", "T_REF2M_MIN_U"),
    (
        "urban_reference_temperature_maximum_instantaneous_patch",
        "T_REF2M_MAX_INST_U",
    ),
    (
        "urban_reference_temperature_minimum_instantaneous_patch",
        "T_REF2M_MIN_INST_U",
    ),
    ("rural_reference_temperature_maximum_patch", "T_REF2M_MAX_R"),
    ("rural_reference_temperature_minimum_patch", "T_REF2M_MIN_R"),
    (
        "rural_reference_temperature_maximum_instantaneous_patch",
        "T_REF2M_MAX_INST_R",
    ),
    (
        "rural_reference_temperature_minimum_instantaneous_patch",
        "T_REF2M_MIN_INST_R",
    ),
    ("transpiration_stress_minimum_patch", "BTRAN_MIN"),
    (
        "transpiration_stress_minimum_instantaneous_patch",
        "BTRAN_MIN_INST",
    ),
)


def _selected_parameter_path(
    case: CimeCase,
    override: str | Path | None,
) -> Path:
    if override is None:
        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        declared = general.get("paramfile")
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError(f"clm_inparm in {case.case_docs / 'lnd_in'} lacks paramfile")
        path = Path(case.expand(declared)).expanduser()
        if not path.is_absolute():
            path = case.root / path
        path = path.resolve()
    else:
        path = Path(override).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM paramfile does not exist: {path}")
    return path


def _pft_names(parameter_data: Path) -> tuple[str, ...]:
    from netCDF4 import Dataset  

    with Dataset(parameter_data, mode="r") as dataset:
        if "pftname" not in dataset.variables:
            raise ValueError("CLM parameter file lacks pftname")
        characters = np.asarray(dataset.variables["pftname"][:])
    if characters.ndim != 2 or characters.dtype.kind not in {"S", "U"}:
        raise ValueError("CLM pftname must be a character matrix")
    if characters.dtype.kind == "S":
        names = tuple(
            b"".join(np.asarray(row, dtype="S1").tolist()).decode("ascii").rstrip()
            for row in characters
        )
    else:
        names = tuple("".join(np.asarray(row, dtype=str).tolist()).rstrip() for row in characters)
    if not all(names) or len(names) != len(set(names)):
        raise ValueError("CLM pftname entries must be nonempty and unique")
    return names


def _accumulator_variable_names() -> tuple[str, ...]:
    native_names = (
        native
        for fields in (
            _ATMOSPHERE_FIELDS,
            _TEMPERATURE_FIELDS,
            _CANOPY_FIELDS,
            _WATER_ENERGY_FIELDS,
            _CROP_FIELDS,
        )
        for _, native in fields
    )
    return tuple(
        f"{native}_{suffix}" for native in native_names for suffix in ("VALUE", "NSTEPS", "PERIOD")
    )


def _accumulator(
    values: dict[str, jnp.ndarray],
    native_name: str,
    expected_size: int,
) -> AccumulatorState:
    value = jnp.asarray(values[f"{native_name}_VALUE"], dtype=jnp.float64)
    nsteps = jnp.asarray(values[f"{native_name}_NSTEPS"], dtype=jnp.int32)
    period = jnp.asarray(values[f"{native_name}_PERIOD"], dtype=jnp.int32)
    if value.shape != (expected_size,) or nsteps.shape != (expected_size,):
        raise ValueError(f"restart accumulator {native_name} has an unexpected subgrid shape")
    if period.size != 1:
        raise ValueError(f"restart accumulator {native_name} has an invalid period")
    period = period.reshape(())
    if int(np.asarray(period)) <= 0:
        raise ValueError(f"restart accumulator {native_name} has an invalid period")
    return AccumulatorState(value, nsteps, period)


def _accumulator_group(
    group_type: type[NamedTuple],
    fields: tuple[tuple[str, str], ...],
    values: dict[str, jnp.ndarray],
    patch_count: int,
    column_count: int,
) -> NamedTuple:
    state = {
        python_name: _accumulator(
            values,
            native_name,
            patch_count if native_name in _PATCH_ACCUMULATOR_NAMES else column_count,
        )
        for python_name, native_name in fields
    }
    return group_type(**state)


def _daily_extrema(
    values: dict[str, jnp.ndarray],
    patch_count: int,
) -> DailyExtremaState:
    state: dict[str, jnp.ndarray] = {}
    for python_name, native_name in _DAILY_FIELDS:
        value = jnp.asarray(values[native_name], dtype=jnp.float64)
        if value.shape != (patch_count,):
            raise ValueError(f"restart daily state {native_name} has an unexpected shape")
        state[python_name] = value
    return DailyExtremaState(**state)


def _required_pft_index(names: tuple[str, ...], name: str) -> int:
    try:
        return names.index(name)
    except ValueError as error:
        raise ValueError(f"CLM parameter pftname lacks required entry {name!r}") from error


def load_diagnostics_and_accumulators_restart(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> DiagnosticsAndAccumulatorsRestartData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    compset = case.value("COMPSET", "").upper()
    if "BGC-CROP" not in compset or "CNDV" in compset or "FATES" in compset:
        raise ValueError("diagnostic adapter requires the non-CNDV BGC-Crop compset")
    if general.get("use_cn") is not True or general.get("use_crop") is not True:
        raise ValueError("diagnostic adapter requires use_cn=.true. and use_crop=.true.")
    if general.get("use_fates", False) is not False:
        raise ValueError("diagnostic adapter requires use_fates=.false.")

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    parameter_data = _selected_parameter_path(case, parameter_data_path)
    dimensions = load_dimension_sizes(restart)
    for dimension in ("pft", "column"):
        if dimension not in dimensions:
            raise ValueError(f"restart lacks dimension {dimension}: {restart}")
    patch_count = dimensions["pft"]
    column_count = dimensions["column"]

    accumulator_names = _accumulator_variable_names()
    value_names = [name for name in accumulator_names if name.endswith("_VALUE")]
    nsteps_names = [name for name in accumulator_names if name.endswith("_NSTEPS")]
    period_names = [name for name in accumulator_names if name.endswith("_PERIOD")]
    daily_names = [name for _, name in _DAILY_FIELDS]
    values = load_native_variables(
        restart,
        [*value_names, *daily_names],
        masked_floating_fill_value=CLM_SPECIAL_VALUE,
    )
    values.update(
        load_native_variables(
            restart,
            nsteps_names,
            masked_integer_fill_value=-9999,
        )
    )
    maximum_period = np.iinfo(np.int32).max
    values.update(
        load_native_variables(
            restart,
            period_names,
            masked_integer_fill_value=maximum_period,
            masked_floating_fill_value=float(maximum_period),
        )
    )
    state = ConfiguredAccumulatorState(
        atmosphere=_accumulator_group(
            AtmosphereAccumulatorState,
            _ATMOSPHERE_FIELDS,
            values,
            patch_count,
            column_count,
        ),
        temperature=_accumulator_group(
            TemperatureAccumulatorState,
            _TEMPERATURE_FIELDS,
            values,
            patch_count,
            column_count,
        ),
        canopy=_accumulator_group(
            CanopyAccumulatorState,
            _CANOPY_FIELDS,
            values,
            patch_count,
            column_count,
        ),
        water_energy=_accumulator_group(
            WaterEnergyAccumulatorState,
            _WATER_ENERGY_FIELDS,
            values,
            patch_count,
            column_count,
        ),
        crop=_accumulator_group(
            CropAccumulatorState,
            _CROP_FIELDS,
            values,
            patch_count,
            column_count,
        ),
        daily_extrema=_daily_extrema(values, patch_count),
    )

    parameter_values = load_native_variables(
        parameter_data,
        ["baset", "mxtmp"],
        masked_floating_fill_value=0.0,
    )
    names = _pft_names(parameter_data)
    base_temperature = jnp.asarray(parameter_values["baset"], dtype=jnp.float64)
    maximum_temperature = jnp.asarray(parameter_values["mxtmp"], dtype=jnp.float64)
    if base_temperature.shape != (len(names),) or maximum_temperature.shape != (len(names),):
        raise ValueError("CLM baset and mxtmp must share the pftname axis")
    if not np.all(np.isfinite(np.asarray(base_temperature))) or not np.all(
        np.isfinite(np.asarray(maximum_temperature))
    ):
        raise ValueError("CLM baset and mxtmp must be finite")

    step_seconds = int(general.get("dtime", 0))
    clock = load_restart_clock(restart)
    if step_seconds <= 0 or step_seconds != clock.step_seconds:
        raise ValueError("case dtime does not match the CLM restart time step")
    crop_group = read_case_namelist_group(case, "lnd_in", "crop")
    mapping = str(crop_group.get("baset_mapping", "")).strip().lower()
    if mapping not in {"constant", "varytropicsbylat"}:
        raise ValueError(f"unsupported CLM crop baset_mapping: {mapping!r}")
    parameters = ConfiguredAccumulatorParameters(
        step_seconds=step_seconds,
        crop_base_temperature_by_pft=base_temperature,
        crop_maximum_temperature_by_pft=maximum_temperature,
        spring_wheat_pft=_required_pft_index(names, "spring_wheat"),
        irrigated_spring_wheat_pft=_required_pft_index(names, "irrigated_spring_wheat"),
        winter_wheat_pft=_required_pft_index(names, "winter_wheat"),
        irrigated_winter_wheat_pft=_required_pft_index(names, "irrigated_winter_wheat"),
        sugarcane_pft=_required_pft_index(names, "sugarcane"),
        irrigated_sugarcane_pft=_required_pft_index(names, "irrigated_sugarcane"),
        latitude_varying_base_temperature=mapping == "varytropicsbylat",
    )
    return DiagnosticsAndAccumulatorsRestartData(
        topology=load_subgrid_topology(restart),
        clock=clock,
        state=state,
        parameters=parameters,
        restart_path=restart,
        parameter_data_path=parameter_data,
    )
