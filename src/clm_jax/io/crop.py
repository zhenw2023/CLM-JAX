
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.io.netcdf import load_native_variables

FIRST_PROGNOSTIC_CROP_PFT = 17


class CropYearRestartData(NamedTuple):

    years_active_patch: jnp.ndarray
    prognostic_crop_patch: jnp.ndarray


def load_crop_year_restart(
    path: str | Path,
    *,
    first_prognostic_crop_pft: int = FIRST_PROGNOSTIC_CROP_PFT,
) -> CropYearRestartData:

    values = load_native_variables(
        path,
        ["nyrs_crop_active", "pfts1d_itypveg", "pfts1d_active"],
    )
    years = jnp.asarray(values["nyrs_crop_active"], dtype=jnp.int32)
    vegetation_type = jnp.asarray(values["pfts1d_itypveg"], dtype=jnp.int32)
    active = jnp.asarray(values["pfts1d_active"] == 1, dtype=jnp.bool_)
    if years.shape != vegetation_type.shape or years.shape != active.shape:
        raise ValueError("crop state and native patch topology have inconsistent shapes")
    return CropYearRestartData(
        years_active_patch=years,
        prognostic_crop_patch=active & (vegetation_type >= jnp.int32(first_prognostic_crop_pft)),
    )
