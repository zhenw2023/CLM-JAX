
from __future__ import annotations

from typing import NamedTuple

from clm_jax.physics.cn_precision_control import (
    CnPrecisionControlDiagnostics,
    CnPrecisionControlInputs,
    CnPrecisionControlParameters,
    CnPrecisionControlState,
    cn_precision_control,
)
from clm_jax.physics.cn_zero import CnTransientFluxState
from clm_jax.physics.soil_biogeochem_precision_control import (
    SoilBiogeochemPrecisionControlDiagnostics,
    SoilBiogeochemPrecisionControlInputs,
    SoilBiogeochemPrecisionControlParameters,
    SoilBiogeochemPrecisionControlState,
    soil_biogeochem_precision_control,
)


class PostDrainagePrecisionControlInputs(NamedTuple):
    vegetation: CnPrecisionControlInputs
    soil: SoilBiogeochemPrecisionControlInputs


class PostDrainagePrecisionControlState(NamedTuple):
    vegetation: CnPrecisionControlState
    soil: SoilBiogeochemPrecisionControlState
    cn_fluxes: CnTransientFluxState


class PostDrainagePrecisionControlParameters(NamedTuple):
    vegetation: CnPrecisionControlParameters
    soil: SoilBiogeochemPrecisionControlParameters


class PostDrainagePrecisionControlDiagnostics(NamedTuple):
    vegetation: CnPrecisionControlDiagnostics
    soil: SoilBiogeochemPrecisionControlDiagnostics


class PostDrainagePrecisionControlResult(NamedTuple):
    state: PostDrainagePrecisionControlState
    diagnostics: PostDrainagePrecisionControlDiagnostics


def post_drainage_precision_control(
    inputs: PostDrainagePrecisionControlInputs,
    state: PostDrainagePrecisionControlState,
    parameters: PostDrainagePrecisionControlParameters,
) -> PostDrainagePrecisionControlResult:

    vegetation = cn_precision_control(
        inputs.vegetation,
        state.vegetation,
        parameters.vegetation,
    )
    soil = soil_biogeochem_precision_control(
        inputs.soil,
        state.soil,
        parameters.soil,
    )
    return PostDrainagePrecisionControlResult(
        state=PostDrainagePrecisionControlState(
            vegetation=vegetation.state,
            soil=soil.state,
            cn_fluxes=state.cn_fluxes,
        ),
        diagnostics=PostDrainagePrecisionControlDiagnostics(
            vegetation=vegetation.diagnostics,
            soil=soil.diagnostics,
        ),
    )


__all__ = [
    "PostDrainagePrecisionControlDiagnostics",
    "PostDrainagePrecisionControlInputs",
    "PostDrainagePrecisionControlParameters",
    "PostDrainagePrecisionControlResult",
    "PostDrainagePrecisionControlState",
    "post_drainage_precision_control",
]
