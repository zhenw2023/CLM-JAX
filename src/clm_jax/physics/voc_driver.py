
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clm_jax.forcing.datm import DatmForcing
from clm_jax.io.restart import LANDUNIT_CROP, LANDUNIT_SOIL, SubgridTopology
from clm_jax.physics.atmospheric_driver import AtmosphericDriverResult
from clm_jax.physics.canopy_flux_driver import (
    CanopyFluxDriverResult,
    CanopyFluxVOCFields,
)
from clm_jax.physics.canopy_temperature_driver import CanopyTemperatureDriverResult
from clm_jax.physics.diagnostics_and_accumulators import ConfiguredAccumulatorState
from clm_jax.physics.numerics import fortran_multiply
from clm_jax.physics.radiation_sequence import RadiationSequenceResult
from clm_jax.physics.voc import (
    VOCEmissionResult,
    VOCInputs,
    VOCParameters,
    VOCState,
    voc_emission,
)


class VOCDriverInputs(NamedTuple):

    co2_volume_mixing_ratio_ppmv: jnp.ndarray
    mapped_isoprene_emission_factor_grid: jnp.ndarray


class VOCDriverResult(NamedTuple):

    inputs: VOCInputs
    emission: VOCEmissionResult
    state: VOCState


def _current_intracellular_co2(
    atmospheric: AtmosphericDriverResult,
    canopy_flux: CanopyFluxDriverResult | CanopyFluxVOCFields,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    initial_sun = jnp.asarray(
        atmospheric.driver_initialization.intracellular_sunlit_co2,
        dtype=jnp.float64,
    )
    initial_shade = jnp.asarray(
        atmospheric.driver_initialization.intracellular_shaded_co2,
        dtype=jnp.float64,
    )
    if isinstance(canopy_flux, CanopyFluxVOCFields):
        solved_source = canopy_flux.intracellular_co2_patch_leaf
        active_source = canopy_flux.active_patch
    else:
        solved_source = canopy_flux.active.energy_hydraulics.hydraulic.intracellular_co2_patch_leaf
        active_source = canopy_flux.fast_inputs.energy.active_patch
    solved = jnp.asarray(solved_source, dtype=jnp.float64)
    active = jnp.asarray(active_source, dtype=jnp.bool_)
    if initial_sun.ndim != 2 or initial_sun.shape != initial_shade.shape:
        raise ValueError("driver intracellular CO2 must use native (patch, canopy-level) axes")
    if solved.ndim != 2 or solved.shape[0] != initial_sun.shape[0] or solved.shape[1] < 2:
        raise ValueError("stage-260 intracellular CO2 must contain sunlit and shaded leaves")
    if active.shape != (initial_sun.shape[0],):
        raise ValueError("stage-260 active mask must contain every native patch")
    return (
        jnp.where(active, solved[:, 0], initial_sun[:, 0]),
        jnp.where(active, solved[:, 1], initial_shade[:, 0]),
    )


def assemble_voc_driver_inputs(
    topology: SubgridTopology,
    grid_forcing: DatmForcing,
    atmospheric: AtmosphericDriverResult,
    radiation: RadiationSequenceResult,
    temperature: CanopyTemperatureDriverResult,
    canopy_flux: CanopyFluxDriverResult | CanopyFluxVOCFields,
    accumulators: ConfiguredAccumulatorState,
    inputs: VOCDriverInputs,
) -> VOCInputs:

    patch_count = topology.patch.gridcell.shape[0]
    grid_count = topology.grid.longitude_degrees.shape[0]
    co2_ppmv = jnp.asarray(inputs.co2_volume_mixing_ratio_ppmv, dtype=jnp.float64)
    if co2_ppmv.ndim != 0:
        raise ValueError("configured constant CO2 mixing ratio must be scalar")
    mapped = jnp.asarray(
        inputs.mapped_isoprene_emission_factor_grid,
        dtype=jnp.float64,
    )
    if mapped.shape != (grid_count, 6):
        raise ValueError("mapped isoprene factors must contain six values per land gridcell")

    landunit = jnp.asarray(topology.patch.landunit_type, dtype=jnp.int32)
    soil_patch = jnp.asarray(topology.patch.active, dtype=jnp.bool_) & (
        (landunit == jnp.int32(LANDUNIT_SOIL)) | (landunit == jnp.int32(LANDUNIT_CROP))
    )
    current_fsun = jnp.asarray(
        radiation.state.canopy_surface.canopy_sunshade.sunlit_fraction_patch,
        dtype=jnp.float64,
    )
    current_lai = jnp.asarray(
        temperature.inputs.exposed_leaf_area_index_patch,
        dtype=jnp.float64,
    )
    current_tveg = jnp.asarray(
        canopy_flux.vegetation_temperature_patch
        if isinstance(canopy_flux, CanopyFluxVOCFields)
        else canopy_flux.state.vegetation_temperature_patch,
        dtype=jnp.float64,
    )
    if any(value.shape != (patch_count,) for value in (current_fsun, current_lai, current_tveg)):
        raise ValueError("current VOC canopy fields must contain every native patch")
    ci_sun, ci_shade = _current_intracellular_co2(atmospheric, canopy_flux)

    atmosphere_accum = accumulators.atmosphere
    temperature_accum = accumulators.temperature
    canopy_accum = accumulators.canopy
    co2_mole_fraction = fortran_multiply(co2_ppmv, jnp.float64(1.0e-6))
    co2_partial_pressure = fortran_multiply(
        co2_mole_fraction,
        jnp.asarray(grid_forcing.pressure, dtype=jnp.float64),
    )
    return VOCInputs(
        soil_patch=soil_patch,
        direct_visible_radiation_grid=jnp.asarray(
            grid_forcing.shortwave_visible_direct,
            dtype=jnp.float64,
        ),
        diffuse_visible_radiation_grid=jnp.asarray(
            grid_forcing.shortwave_visible_diffuse,
            dtype=jnp.float64,
        ),
        air_pressure_column=jnp.asarray(
            atmospheric.column_forcing.pressure,
            dtype=jnp.float64,
        ),
        co2_partial_pressure_grid=co2_partial_pressure,
        direct_visible_radiation_24h_patch=atmosphere_accum.direct_solar_24_hour.value,
        direct_visible_radiation_240h_patch=atmosphere_accum.direct_solar_240_hour.value,
        diffuse_visible_radiation_24h_patch=atmosphere_accum.diffuse_solar_24_hour.value,
        diffuse_visible_radiation_240h_patch=atmosphere_accum.diffuse_solar_240_hour.value,
        sunlit_fraction_patch=current_fsun,
        sunlit_fraction_24h_patch=canopy_accum.sunlit_fraction_24_hour.value,
        sunlit_fraction_240h_patch=canopy_accum.sunlit_fraction_240_hour.value,
        exposed_leaf_area_index_patch=current_lai,
        exposed_leaf_area_index_240h_patch=(canopy_accum.exposed_leaf_area_index_240_hour.value),
        sunlit_intracellular_co2_patch=ci_sun,
        shaded_intracellular_co2_patch=ci_shade,
        vegetation_temperature_patch=current_tveg,
        vegetation_temperature_24h_patch=(temperature_accum.vegetation_temperature_24_hour.value),
        vegetation_temperature_240h_patch=(temperature_accum.vegetation_temperature_240_hour.value),
        mapped_isoprene_emission_factor_grid=mapped,
    )


def voc_driver_step(
    topology: SubgridTopology,
    grid_forcing: DatmForcing,
    atmospheric: AtmosphericDriverResult,
    radiation: RadiationSequenceResult,
    temperature: CanopyTemperatureDriverResult,
    canopy_flux: CanopyFluxDriverResult | CanopyFluxVOCFields,
    accumulators: ConfiguredAccumulatorState,
    inputs: VOCDriverInputs,
    state: VOCState,
    parameters: VOCParameters,
) -> VOCDriverResult:

    dynamic_inputs = assemble_voc_driver_inputs(
        topology,
        grid_forcing,
        atmospheric,
        radiation,
        temperature,
        canopy_flux,
        accumulators,
        inputs,
    )
    emission = voc_emission(topology, state, dynamic_inputs, parameters)
    return VOCDriverResult(
        inputs=dynamic_inputs,
        emission=emission,
        state=emission.state,
    )


__all__ = [
    "VOCDriverInputs",
    "VOCDriverResult",
    "assemble_voc_driver_inputs",
    "voc_driver_step",
]
