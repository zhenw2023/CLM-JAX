
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.diagnostics_and_accumulators import (
    DiagnosticsAndAccumulatorsRestartData,
    load_diagnostics_and_accumulators_restart,
)
from clm_jax.io.end_of_step_vegetation import (
    EndOfStepVegetationConfiguration,
    load_end_of_step_vegetation_configuration,
)
from clm_jax.io.lnd2glc import (
    LandToGlacierNativeData,
    load_land_to_glacier_native_data,
)
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.native_templates import native_template
from clm_jax.physics.end_of_timestep import EndOfTimestepCompositionParameters


class EndOfTimestepNativeData(NamedTuple):

    land_to_glacier: LandToGlacierNativeData
    accumulators: DiagnosticsAndAccumulatorsRestartData
    vegetation: EndOfStepVegetationConfiguration
    composition_parameters: EndOfTimestepCompositionParameters


@native_template
def load_end_of_timestep_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
) -> EndOfTimestepNativeData:

    glacier = load_land_to_glacier_native_data(case, restart_path)
    accumulators = load_diagnostics_and_accumulators_restart(
        case,
        restart_path=restart_path,
        parameter_data_path=parameter_data_path,
    )
    vegetation = load_end_of_step_vegetation_configuration(case)
    if glacier.restart_path != accumulators.restart_path:
        raise ValueError("tail adapters selected different CLM restart files")

    general = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    if str(general.get("co2_type", "")).strip().lower() != "constant":
        raise ValueError("configured end-of-timestep adapter requires co2_type='constant'")
    co2_ppmv = float(general.get("co2_ppmv", np.nan))
    if not np.isfinite(co2_ppmv) or not 10.0 <= co2_ppmv <= 15_000.0:
        raise ValueError("configured end-of-timestep adapter requires physical co2_ppmv")

    crop = dict(read_case_namelist_group(case, "lnd_in", "crop"))
    intercept = float(crop.get("baset_latvary_intercept", np.nan))
    slope = float(crop.get("baset_latvary_slope", np.nan))
    if intercept != 12.0 or slope != 0.4:
        raise ValueError(
            "reference CropType requires baset_latvary_intercept=12 and "
            "baset_latvary_slope=0.4"
        )

    return EndOfTimestepNativeData(
        land_to_glacier=glacier,
        accumulators=accumulators,
        vegetation=vegetation,
        composition_parameters=EndOfTimestepCompositionParameters(
            atmospheric_co2_ppmv=co2_ppmv,
            atmospheric_oxygen_mole_fraction=0.209,
            crop_latitude_base_temperature_intercept=intercept,
            crop_latitude_base_temperature_slope=slope,
        ),
    )


__all__ = ["EndOfTimestepNativeData", "load_end_of_timestep_native_data"]
