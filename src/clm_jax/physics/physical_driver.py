
from __future__ import annotations

from typing import NamedTuple

from clm_jax.forcing.datm import DatmForcing
from clm_jax.io.restart import SubgridTopology
from clm_jax.physics.atmospheric_driver import (
    AtmosphericDriverInputs,
    AtmosphericDriverResult,
    AtmosphericDriverState,
    AtmosphericDriverStatic,
    atmospheric_driver_step,
)
from clm_jax.physics.bare_ground_driver import (
    BareGroundDriverParameters,
    BareGroundDriverResult,
    bare_ground_driver_step,
)
from clm_jax.physics.bare_ground_fluxes import BareGroundFluxState
from clm_jax.physics.canopy_driver import (
    CanopyDriverInputs,
    CanopyDriverParameters,
    CanopyDriverResult,
    CanopyDriverState,
    canopy_driver_step,
)
from clm_jax.physics.canopy_flux_driver import (
    CanopyFluxDriverInputs,
    CanopyFluxDriverParameters,
    CanopyFluxDriverResult,
    CanopyFluxDriverState,
    CanopyFluxRuntimeResult,
    canopy_flux_driver_step,
)
from clm_jax.physics.canopy_temperature import (
    CanopyTemperatureInputs,
    CanopyTemperatureParameters,
    CanopyTemperatureState,
)
from clm_jax.physics.canopy_temperature_driver import (
    CanopyTemperatureDriverResult,
    canopy_temperature_driver_step,
)
from clm_jax.physics.diagnostics_and_accumulators import ConfiguredAccumulatorState
from clm_jax.physics.dust import (
    DustExecutionParameters,
    DustParameters,
    DustState,
)
from clm_jax.physics.dust_driver import (
    DustDriverInputs,
    DustDriverResult,
    dust_driver_step,
)
from clm_jax.physics.irrigation import IrrigationDemandParameters
from clm_jax.physics.irrigation_demand_driver import (
    IrrigationDemandDriverInputs,
    IrrigationDemandDriverResult,
    irrigation_demand_driver_step,
)
from clm_jax.physics.lake_flux_driver import (
    LakeFluxDriverInputs,
    LakeFluxDriverParameters,
    LakeFluxDriverResult,
    LakeFluxDriverState,
    lake_flux_driver_step,
)
from clm_jax.physics.lake_temperature import LakeColumnTemperatureState
from clm_jax.physics.lake_temperature_driver import (
    LakeTemperatureDriverInputs,
    LakeTemperatureDriverResult,
    lake_temperature_driver_step,
)
from clm_jax.physics.land_to_atmosphere_boundary import (
    LandToAtmospherePhysicalFields,
    compose_land_to_atmosphere_physical_fields,
)
from clm_jax.physics.radiation_sequence import (
    RadiationSequenceInputs,
    RadiationSequenceResult,
    RadiationSequenceState,
    radiation_sequence_step,
)
from clm_jax.physics.soil_temperature_driver import (
    SoilTemperatureDriverInputs,
    SoilTemperatureDriverResult,
    soil_temperature_driver_step,
)
from clm_jax.physics.urban_flux_driver import (
    UrbanFluxDriverInputs,
    UrbanFluxDriverParameters,
    UrbanFluxDriverResult,
    UrbanFluxDriverState,
    urban_flux_driver_step,
)
from clm_jax.physics.voc import VOCParameters, VOCState
from clm_jax.physics.voc_driver import (
    VOCDriverInputs,
    VOCDriverResult,
    voc_driver_step,
)
from clm_jax.trajectory import NoleapModelClock


class PhysicalDriverStatic(NamedTuple):

    topology: SubgridTopology
    atmospheric: AtmosphericDriverStatic


class PhysicalDriverInputs(NamedTuple):

    atmospheric: AtmosphericDriverInputs
    canopy: CanopyDriverInputs
    radiation: RadiationSequenceInputs
    canopy_temperature: CanopyTemperatureInputs
    canopy_flux: CanopyFluxDriverInputs
    urban_flux: UrbanFluxDriverInputs
    lake_flux: LakeFluxDriverInputs
    irrigation_demand: IrrigationDemandDriverInputs
    dust: DustDriverInputs
    voc: VOCDriverInputs
    voc_accumulators: ConfiguredAccumulatorState
    lake_temperature: LakeTemperatureDriverInputs
    soil_temperature: SoilTemperatureDriverInputs


class PhysicalDriverState(NamedTuple):

    atmospheric: AtmosphericDriverState
    canopy: CanopyDriverState
    radiation: RadiationSequenceState
    canopy_temperature: CanopyTemperatureState
    bare_ground: BareGroundFluxState
    canopy_flux: CanopyFluxDriverState
    urban_flux: UrbanFluxDriverState
    lake_flux: LakeFluxDriverState
    dust: DustState
    voc: VOCState
    lake_temperature: LakeColumnTemperatureState


class PhysicalDriverParameters(NamedTuple):

    canopy: CanopyDriverParameters
    canopy_temperature: CanopyTemperatureParameters
    bare_ground: BareGroundDriverParameters
    canopy_flux: CanopyFluxDriverParameters
    urban_flux: UrbanFluxDriverParameters
    lake_flux: LakeFluxDriverParameters
    irrigation_demand: IrrigationDemandParameters
    dust: DustParameters
    dust_execution: DustExecutionParameters
    voc: VOCParameters


class PhysicalDriverResult(NamedTuple):

    atmospheric: AtmosphericDriverResult
    canopy: CanopyDriverResult
    radiation: RadiationSequenceResult
    canopy_temperature: CanopyTemperatureDriverResult
    bare_ground: BareGroundDriverResult
    canopy_flux: CanopyFluxDriverResult
    urban_flux: UrbanFluxDriverResult
    lake_flux: LakeFluxDriverResult
    irrigation_demand: IrrigationDemandDriverResult
    dust: DustDriverResult
    voc: VOCDriverResult
    lake_temperature: LakeTemperatureDriverResult
    soil_temperature: SoilTemperatureDriverResult
    land_to_atmosphere: LandToAtmospherePhysicalFields | None = None


class PhysicalDriverRuntimeResult(NamedTuple):

    atmospheric: AtmosphericDriverResult
    canopy: CanopyDriverResult
    radiation: RadiationSequenceResult
    canopy_temperature: CanopyTemperatureDriverResult
    bare_ground: BareGroundDriverResult
    canopy_flux: CanopyFluxRuntimeResult
    urban_flux: UrbanFluxDriverResult
    lake_flux: LakeFluxDriverResult
    irrigation_demand: IrrigationDemandDriverResult
    dust: DustDriverResult
    voc: VOCDriverResult
    lake_temperature: LakeTemperatureDriverResult
    soil_temperature: SoilTemperatureDriverResult
    land_to_atmosphere: LandToAtmospherePhysicalFields | None = None


def physical_driver_step(
    static: PhysicalDriverStatic,
    state: PhysicalDriverState,
    inputs: PhysicalDriverInputs,
    parameters: PhysicalDriverParameters,
    forcing: DatmForcing,
    clock: NoleapModelClock,
) -> PhysicalDriverResult:

    topology = static.topology
    atmospheric = atmospheric_driver_step(
        static.atmospheric,
        state.atmospheric,
        inputs.atmospheric,
        forcing,
        clock,
    )
    canopy = canopy_driver_step(
        topology,
        atmospheric,
        forcing,
        inputs.canopy,
        state.canopy,
        step_seconds=clock.step_seconds,
        parameters=parameters.canopy,
    )
    radiation = radiation_sequence_step(
        topology,
        forcing,
        canopy,
        inputs.radiation,
        state.radiation,
        current_time_of_day_seconds=clock.seconds_into_day,
        step_seconds=clock.step_seconds,
    )
    canopy_temperature = canopy_temperature_driver_step(
        topology,
        atmospheric,
        forcing,
        canopy,
        inputs.canopy_temperature,
        parameters.canopy_temperature,
        state.canopy_temperature,
    )
    bare_ground = bare_ground_driver_step(
        topology,
        atmospheric,
        forcing,
        canopy,
        canopy_temperature,
        parameters.bare_ground,
        state.bare_ground,
    )
    canopy_flux = canopy_flux_driver_step(
        topology,
        atmospheric,
        forcing,
        canopy,
        radiation,
        canopy_temperature,
        bare_ground,
        clock,
        inputs.canopy_flux,
        parameters.canopy_flux,
        state.canopy_flux,
    )
    urban_flux = urban_flux_driver_step(
        topology,
        forcing,
        canopy,
        canopy_temperature,
        inputs.urban_flux,
        parameters.urban_flux,
        state.urban_flux,
    )
    lake_flux = lake_flux_driver_step(
        topology,
        atmospheric,
        forcing,
        canopy,
        radiation,
        canopy_temperature,
        inputs.lake_flux,
        parameters.lake_flux,
        state.lake_flux,
    )
    irrigation_demand = irrigation_demand_driver_step(
        topology,
        atmospheric,
        inputs.canopy,
        canopy,
        canopy_temperature,
        clock,
        inputs.irrigation_demand,
        parameters.irrigation_demand,
    )
    dust = dust_driver_step(
        topology,
        atmospheric,
        canopy,
        canopy_temperature,
        bare_ground,
        canopy_flux,
        inputs.urban_flux,
        urban_flux,
        inputs.lake_flux,
        lake_flux,
        inputs.dust,
        state.dust,
        parameters.dust,
        execution=parameters.dust_execution,
    )
    voc = voc_driver_step(
        topology,
        forcing,
        atmospheric,
        radiation,
        canopy_temperature,
        canopy_flux,
        inputs.voc_accumulators,
        inputs.voc,
        state.voc,
        parameters.voc,
    )
    lake_temperature = lake_temperature_driver_step(
        topology,
        canopy,
        radiation,
        canopy_temperature,
        lake_flux,
        inputs.lake_temperature,
        state.lake_temperature,
        time_step_seconds=clock.step_seconds,
    )
    soil_temperature = soil_temperature_driver_step(
        topology,
        atmospheric,
        canopy,
        radiation,
        canopy_temperature,
        canopy_flux,
        urban_flux,
        inputs.soil_temperature,
        time_step_seconds=clock.step_seconds,
    )
    return PhysicalDriverResult(
        atmospheric=atmospheric,
        canopy=canopy,
        radiation=radiation,
        canopy_temperature=canopy_temperature,
        bare_ground=bare_ground,
        canopy_flux=canopy_flux,
        urban_flux=urban_flux,
        lake_flux=lake_flux,
        irrigation_demand=irrigation_demand,
        dust=dust,
        voc=voc,
        lake_temperature=lake_temperature,
        soil_temperature=soil_temperature,
        land_to_atmosphere=compose_land_to_atmosphere_physical_fields(
            canopy_flux,
            inputs.urban_flux,
            urban_flux,
            inputs.lake_flux,
            lake_flux,
            dust,
            voc,
        ),
    )


__all__ = [
    "PhysicalDriverInputs",
    "PhysicalDriverParameters",
    "PhysicalDriverResult",
    "PhysicalDriverRuntimeResult",
    "PhysicalDriverState",
    "PhysicalDriverStatic",
    "physical_driver_step",
]
