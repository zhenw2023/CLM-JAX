
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import jit, lax


class BgcExternalForcingInputs(NamedTuple):

    nitrogen_deposition_lower_gridcell: jnp.ndarray
    nitrogen_deposition_upper_gridcell: jnp.ndarray
    population_density_lower_gridcell: jnp.ndarray
    population_density_upper_gridcell: jnp.ndarray
    lightning_frequency_lower_gridcell: jnp.ndarray
    lightning_frequency_upper_gridcell: jnp.ndarray
    nitrogen_deposition_lower_weight: float | jnp.ndarray
    population_density_lower_weight: float | jnp.ndarray
    lightning_frequency_lower_weight: float | jnp.ndarray


class BgcExternalForcingParameters(NamedTuple):

    nitrogen_deposition_is_annual: bool | jnp.ndarray = False
    seconds_per_day: float | jnp.ndarray = 86_400.0
    days_per_year: float | jnp.ndarray = 365.0


class BgcExternalForcingState(NamedTuple):

    nitrogen_deposition_gridcell: jnp.ndarray
    human_population_density_gridcell: jnp.ndarray
    lightning_frequency_gridcell: jnp.ndarray


class BgcExternalForcingDiagnostics(NamedTuple):

    nitrogen_deposition_lower_weight: jnp.ndarray
    population_density_lower_weight: jnp.ndarray
    lightning_frequency_lower_weight: jnp.ndarray
    gridcell_count: jnp.ndarray


class BgcExternalForcingResult(NamedTuple):
    state: BgcExternalForcingState
    diagnostics: BgcExternalForcingDiagnostics


DEFAULT_BGC_EXTERNAL_FORCING_PARAMETERS = BgcExternalForcingParameters()


def _validate(inputs: BgcExternalForcingInputs) -> int:
    names = (
        "nitrogen_deposition_lower_gridcell",
        "nitrogen_deposition_upper_gridcell",
        "population_density_lower_gridcell",
        "population_density_upper_gridcell",
        "lightning_frequency_lower_gridcell",
        "lightning_frequency_upper_gridcell",
    )
    first = jnp.shape(getattr(inputs, names[0]))
    if len(first) != 1:
        raise ValueError("BGC external forcing records must use the gridcell axis")
    for name in names[1:]:
        if jnp.shape(getattr(inputs, name)) != first:
            raise ValueError(f"{name} does not match the native gridcell extent")
    return first[0]


def _interpolate(
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    lower_weight: float | jnp.ndarray,
) -> jnp.ndarray:

    from clm_jax.physics.numerics import fortran_add, fortran_multiply

    lower_weight = jnp.asarray(lower_weight, dtype=jnp.float64)
    upper_weight = jnp.float64(1.0) - lower_weight
    return fortran_add(
        fortran_multiply(jnp.asarray(lower, dtype=jnp.float64), lower_weight),
        fortran_multiply(jnp.asarray(upper, dtype=jnp.float64), upper_weight),
    )


@jit
def interpolate_bgc_external_forcing(
    inputs: BgcExternalForcingInputs,
    parameters: BgcExternalForcingParameters = DEFAULT_BGC_EXTERNAL_FORCING_PARAMETERS,
) -> BgcExternalForcingResult:

    from clm_jax.physics.numerics import fortran_divide

    gridcell_count = _validate(inputs)
    nitrogen_deposition = _interpolate(
        inputs.nitrogen_deposition_lower_gridcell,
        inputs.nitrogen_deposition_upper_gridcell,
        inputs.nitrogen_deposition_lower_weight,
    )
    denominator = jnp.asarray(parameters.seconds_per_day, dtype=jnp.float64) * jnp.asarray(
        parameters.days_per_year,
        dtype=jnp.float64,
    )
    nitrogen_deposition = lax.cond(
        jnp.asarray(parameters.nitrogen_deposition_is_annual, dtype=jnp.bool_),
        lambda value: fortran_divide(value, denominator),
        lambda value: value,
        nitrogen_deposition,
    )
    population_density = _interpolate(
        inputs.population_density_lower_gridcell,
        inputs.population_density_upper_gridcell,
        inputs.population_density_lower_weight,
    )
    lightning_frequency = _interpolate(
        inputs.lightning_frequency_lower_gridcell,
        inputs.lightning_frequency_upper_gridcell,
        inputs.lightning_frequency_lower_weight,
    )
    return BgcExternalForcingResult(
        state=BgcExternalForcingState(
            nitrogen_deposition_gridcell=nitrogen_deposition,
            human_population_density_gridcell=population_density,
            lightning_frequency_gridcell=lightning_frequency,
        ),
        diagnostics=BgcExternalForcingDiagnostics(
            nitrogen_deposition_lower_weight=jnp.asarray(
                inputs.nitrogen_deposition_lower_weight,
                dtype=jnp.float64,
            ),
            population_density_lower_weight=jnp.asarray(
                inputs.population_density_lower_weight,
                dtype=jnp.float64,
            ),
            lightning_frequency_lower_weight=jnp.asarray(
                inputs.lightning_frequency_lower_weight,
                dtype=jnp.float64,
            ),
            gridcell_count=jnp.asarray(gridcell_count, dtype=jnp.int32),
        ),
    )


__all__ = [
    "BgcExternalForcingDiagnostics",
    "BgcExternalForcingInputs",
    "BgcExternalForcingParameters",
    "BgcExternalForcingResult",
    "BgcExternalForcingState",
    "DEFAULT_BGC_EXTERNAL_FORCING_PARAMETERS",
    "interpolate_bgc_external_forcing",
]
