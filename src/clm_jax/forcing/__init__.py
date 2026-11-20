
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    **{
        name: "clm_jax.forcing.atmosphere"
        for name in (
            "Atm2LndParameters",
            "ColumnAtmosphericForcing",
            "PrecipitationPartition",
            "ThermodynamicForcing",
            "air_density",
            "build_normalization",
            "compress_datm_forcing_to_land_grid",
            "downscale_atmospheric_forcing_to_columns",
            "downscale_longwave",
            "downscale_thermodynamics",
            "partition_precipitation_to_columns",
            "precipitation_ramp",
            "repartition_precipitation",
        )
    },
    **{
        name: "clm_jax.forcing.bgc"
        for name in (
            "BgcExternalForcingDiagnostics",
            "BgcExternalForcingInputs",
            "BgcExternalForcingParameters",
            "BgcExternalForcingResult",
            "BgcExternalForcingState",
            "interpolate_bgc_external_forcing",
        )
    },
    **{
        name: "clm_jax.forcing.datm"
        for name in (
            "ClmncepShortwave",
            "DatmForcing",
            "clmncep_forcing",
            "clmncep_shortwave_partition",
            "datm_rain_fraction",
            "datm_saturation_vapor_pressure",
        )
    },
    **{
        name: "clm_jax.forcing.remap"
        for name in (
            "SparseMapping",
            "apply_normalized_sparse_mapping",
            "apply_sparse_mapping",
            "build_rectilinear_bilinear_mapping",
            "fill_and_remap",
            "load_sparse_mapping",
            "temporal_interpolate",
        )
    },
    **{
        name: "clm_jax.forcing.urban"
        for name in (
            "UrbanTimeVaryingDiagnostics",
            "UrbanTimeVaryingInputs",
            "UrbanTimeVaryingResult",
            "UrbanTimeVaryingState",
            "interpolate_urban_time_varying",
        )
    },
    **{
        name: "clm_jax.forcing.wfde5"
        for name in (
            "Wfde5Record",
            "Wfde5StreamSet",
            "map_wfde5_record",
            "prepare_wfde5_datm_step",
        )
    },
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):

    try:
        module_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
