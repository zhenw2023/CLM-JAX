
from __future__ import annotations

from functools import cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_native_variables
from clm_jax.physics.snow_age_grain import (
    NUMBER_OF_SNOW_AGING_DENSITIES,
    NUMBER_OF_SNOW_AGING_GRADIENTS,
    NUMBER_OF_SNOW_AGING_TEMPERATURES,
    SnowAgingLookupTable,
)

_VARIABLES = ("tau", "kappa", "drdsdt0")


def configured_snow_aging_path(case: CimeCase) -> Path:

    group = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    declared = group.get("fsnowaging")
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError(f"clm_inparm in {case.case_docs / 'lnd_in'} lacks fsnowaging")
    path = Path(case.expand(declared)).expanduser()
    if not path.is_absolute():
        path = case.root / path
    return path.resolve()


@cache
def load_snow_aging_lookup_table(
    case: CimeCase,
    snow_aging_path: str | Path | None = None,
) -> SnowAgingLookupTable:

    path = (
        configured_snow_aging_path(case)
        if snow_aging_path is None
        else Path(snow_aging_path).expanduser().resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(f"configured snow-aging table does not exist: {path}")
    loaded = load_native_variables(path, list(_VARIABLES), dtype=np.float64)
    file_shape = (
        NUMBER_OF_SNOW_AGING_TEMPERATURES,
        NUMBER_OF_SNOW_AGING_GRADIENTS,
        NUMBER_OF_SNOW_AGING_DENSITIES,
    )
    for name, values in loaded.items():
        if values.shape != file_shape:
            raise ValueError(
                f"snow-aging variable {name!r} must have file shape {file_shape}, "
                f"got {values.shape}"
            )
        if not np.all(np.isfinite(np.asarray(values))):
            raise ValueError(f"snow-aging variable {name!r} contains non-finite data")

    def native_order(name: str) -> jnp.ndarray:
        with jax.core.eval_context():
            return jnp.asarray(np.ascontiguousarray(np.asarray(loaded[name]).transpose(2, 1, 0)))

    return SnowAgingLookupTable(
        time_scale_hours=native_order("tau"),
        shape_exponent=native_order("kappa"),
        initial_growth_rate_microns_per_hour=native_order("drdsdt0"),
    )
