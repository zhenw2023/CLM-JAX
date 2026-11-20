
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from clm_jax.constants import CLM_SPECIAL_VALUE
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_LAKE,
    LANDUNIT_SOIL,
    LANDUNIT_URBAN_MAX,
    LANDUNIT_URBAN_MIN,
    SubgridTopology,
)
from clm_jax.physics.numerics import (
    fortran_linear_combination,
    fortran_ordered_sum,
    fortran_product,
)


class SurfaceRadiationInputs(NamedTuple):

    direct_shortwave_gridcell: jnp.ndarray
    diffuse_shortwave_gridcell: jnp.ndarray
    snow_layer_count_column: jnp.ndarray
    snow_depth_column: jnp.ndarray
    ground_albedo_direct_column: jnp.ndarray
    ground_albedo_diffuse_column: jnp.ndarray
    soil_albedo_direct_column: jnp.ndarray
    soil_albedo_diffuse_column: jnp.ndarray
    snow_albedo_direct_column: jnp.ndarray
    snow_albedo_diffuse_column: jnp.ndarray
    surface_albedo_direct_patch: jnp.ndarray
    surface_albedo_diffuse_patch: jnp.ndarray
    canopy_absorption_direct_patch: jnp.ndarray
    canopy_absorption_diffuse_patch: jnp.ndarray
    direct_transmittance_patch: jnp.ndarray
    direct_to_diffuse_transmittance_patch: jnp.ndarray
    diffuse_transmittance_patch: jnp.ndarray
    snow_absorption_direct_visible_column: jnp.ndarray
    snow_absorption_direct_nir_column: jnp.ndarray
    snow_absorption_diffuse_visible_column: jnp.ndarray
    snow_absorption_diffuse_nir_column: jnp.ndarray


class SurfaceRadiationFluxState(NamedTuple):

    sunlit_fraction_patch: jnp.ndarray
    absorbed_total_patch: jnp.ndarray
    reflected_total_patch: jnp.ndarray
    rural_absorbed_total_patch: jnp.ndarray
    absorbed_vegetation_patch: jnp.ndarray
    absorbed_ground_patch: jnp.ndarray
    absorbed_ground_soil_patch: jnp.ndarray
    absorbed_ground_snow_patch: jnp.ndarray
    absorbed_ground_layers_patch: jnp.ndarray
    penetrating_shortwave_patch: jnp.ndarray
    subsurface_snow_absorption_fraction_patch: jnp.ndarray


class SurfaceRadiationDiagnosticsState(NamedTuple):

    incident_direct_patch: jnp.ndarray
    incident_diffuse_patch: jnp.ndarray
    reflected_direct_patch: jnp.ndarray
    reflected_diffuse_patch: jnp.ndarray
    local_noon_incident_direct_patch: jnp.ndarray
    local_noon_incident_visible_diffuse_patch: jnp.ndarray
    local_noon_reflected_direct_patch: jnp.ndarray
    local_noon_absorbed_par_patch: jnp.ndarray
    snow_incident_direct_patch: jnp.ndarray
    snow_incident_diffuse_patch: jnp.ndarray
    snow_reflected_direct_patch: jnp.ndarray
    snow_reflected_diffuse_patch: jnp.ndarray


class SurfaceRadiationState(NamedTuple):
    fluxes: SurfaceRadiationFluxState
    diagnostics: SurfaceRadiationDiagnosticsState


def local_noon_mask(
    longitude_degrees: jnp.ndarray,
    current_time_of_day_seconds: jnp.ndarray | int,
    time_step_seconds: jnp.ndarray | int,
) -> jnp.ndarray:

    longitude = jnp.asarray(longitude_degrees, dtype=jnp.float64)
    current_seconds = jnp.asarray(current_time_of_day_seconds, dtype=jnp.int64)
    step_seconds = jnp.asarray(time_step_seconds, dtype=jnp.int64)
    positive_longitude = jnp.where(
        longitude < jnp.float64(0.0),
        longitude + jnp.float64(360.0),
        longitude,
    )
    degrees_per_second = jnp.float64(15.0) / jnp.float64(3600.0)
    step_float = jnp.asarray(step_seconds, dtype=jnp.float64)
    offset_steps = jnp.floor(
        (positive_longitude / degrees_per_second) / step_float + jnp.float64(0.5)
    ).astype(jnp.int64)
    local_seconds = jnp.mod(
        current_seconds + offset_steps * step_seconds,
        jnp.int64(86400),
    )
    half_step = step_seconds // jnp.int64(2)
    return (local_seconds >= jnp.int64(43200) - half_step) & (
        local_seconds <= jnp.int64(43200) + half_step
    )


def _cast_and_validate(
    topology: SubgridTopology,
    inputs: SurfaceRadiationInputs,
    state: SurfaceRadiationState,
) -> tuple[SurfaceRadiationInputs, SurfaceRadiationState]:
    number_of_gridcells = topology.grid.longitude_degrees.shape[0]
    number_of_columns = topology.column.gridcell.shape[0]
    number_of_patches = topology.patch.gridcell.shape[0]
    band_shape_grid = (number_of_gridcells, 2)
    band_shape_column = (number_of_columns, 2)
    band_shape_patch = (number_of_patches, 2)
    scalar_shape_column = (number_of_columns,)
    scalar_shape_patch = (number_of_patches,)

    float_inputs = {
        field: jnp.asarray(getattr(inputs, field), dtype=jnp.float64)
        for field in SurfaceRadiationInputs._fields
        if field != "snow_layer_count_column"
    }
    snow_layer_count = jnp.asarray(inputs.snow_layer_count_column, dtype=jnp.int32)
    if float_inputs["direct_shortwave_gridcell"].shape != band_shape_grid or float_inputs[
        "diffuse_shortwave_gridcell"
    ].shape != band_shape_grid:
        raise ValueError("shortwave forcing must have shape (gridcell, 2)")
    if snow_layer_count.shape != scalar_shape_column or float_inputs[
        "snow_depth_column"
    ].shape != scalar_shape_column:
        raise ValueError("column snow scalars must contain one value per column")

    column_band_fields = (
        "ground_albedo_direct_column",
        "ground_albedo_diffuse_column",
        "soil_albedo_direct_column",
        "soil_albedo_diffuse_column",
        "snow_albedo_direct_column",
        "snow_albedo_diffuse_column",
    )
    if any(float_inputs[field].shape != band_shape_column for field in column_band_fields):
        raise ValueError("column albedos must have shape (column, 2)")
    patch_band_fields = (
        "surface_albedo_direct_patch",
        "surface_albedo_diffuse_patch",
        "canopy_absorption_direct_patch",
        "canopy_absorption_diffuse_patch",
        "direct_transmittance_patch",
        "direct_to_diffuse_transmittance_patch",
        "diffuse_transmittance_patch",
    )
    if any(float_inputs[field].shape != band_shape_patch for field in patch_band_fields):
        raise ValueError("patch radiation coefficients must have shape (patch, 2)")

    layer_fields = (
        "snow_absorption_direct_visible_column",
        "snow_absorption_direct_nir_column",
        "snow_absorption_diffuse_visible_column",
        "snow_absorption_diffuse_nir_column",
    )
    layer_shape = float_inputs[layer_fields[0]].shape
    if len(layer_shape) != 2 or layer_shape[0] != number_of_columns:
        raise ValueError("snow absorption factors must have shape (column, snow_layer+1)")
    if any(float_inputs[field].shape != layer_shape for field in layer_fields[1:]):
        raise ValueError("all snow absorption factor arrays must have the same shape")
    if layer_shape[1] < 2:
        raise ValueError("snow absorption arrays must include snow and top-soil levels")

    cast_inputs = SurfaceRadiationInputs(
        **float_inputs,
        snow_layer_count_column=snow_layer_count,
    )

    flux_arrays = {
        field: jnp.asarray(getattr(state.fluxes, field), dtype=jnp.float64)
        for field in SurfaceRadiationFluxState._fields
    }
    for field, value in flux_arrays.items():
        expected = (
            (number_of_patches, layer_shape[1])
            if field == "absorbed_ground_layers_patch"
            else scalar_shape_patch
        )
        if value.shape != expected:
            raise ValueError(f"surface-radiation flux state {field!r} has shape {value.shape}")

    diagnostic_arrays = {
        field: jnp.asarray(getattr(state.diagnostics, field), dtype=jnp.float64)
        for field in SurfaceRadiationDiagnosticsState._fields
    }
    scalar_diagnostics = (
        "local_noon_incident_visible_diffuse_patch",
        "local_noon_absorbed_par_patch",
    )
    for field, value in diagnostic_arrays.items():
        expected = scalar_shape_patch if field in scalar_diagnostics else band_shape_patch
        if value.shape != expected:
            raise ValueError(
                f"surface-radiation diagnostic state {field!r} has shape {value.shape}"
            )

    return (
        cast_inputs,
        SurfaceRadiationState(
            fluxes=SurfaceRadiationFluxState(**flux_arrays),
            diagnostics=SurfaceRadiationDiagnosticsState(**diagnostic_arrays),
        ),
    )


def _masked_ordered_layer_sum(values: jnp.ndarray, include: jnp.ndarray) -> jnp.ndarray:

    def add_selected(running_sum, operands):
        value, selected = operands
        updated = jnp.where(selected, running_sum + value, running_sum)
        return updated, None

    initial = jnp.zeros(values.shape[0], dtype=jnp.float64)
    total, _ = lax.scan(add_selected, initial, (values.T, include.T))
    return total


def _redistributed_snow_layers(
    absorbed_ground_snow: jnp.ndarray,
    snow_layer_count: jnp.ndarray,
    number_of_levels: int,
) -> jnp.ndarray:
    patch_count = absorbed_ground_snow.shape[0]
    rows = jnp.arange(patch_count, dtype=jnp.int32)
    result = jnp.zeros((patch_count, number_of_levels), dtype=jnp.float64)

    one_layer = snow_layer_count == jnp.int32(-1)
    one_layer_values = jnp.zeros_like(result)
    one_layer_values = one_layer_values.at[:, number_of_levels - 2].set(
        fortran_product(absorbed_ground_snow, jnp.full_like(absorbed_ground_snow, 0.6))
    )
    one_layer_values = one_layer_values.at[:, number_of_levels - 1].set(
        fortran_product(absorbed_ground_snow, jnp.full_like(absorbed_ground_snow, 0.4))
    )

    first_index = jnp.clip(
        snow_layer_count + jnp.int32(number_of_levels - 1),
        jnp.int32(0),
        jnp.int32(number_of_levels - 1),
    )
    second_index = jnp.clip(
        first_index + jnp.int32(1),
        jnp.int32(0),
        jnp.int32(number_of_levels - 1),
    )
    many_layer_values = result.at[rows, first_index].set(
        fortran_product(absorbed_ground_snow, jnp.full_like(absorbed_ground_snow, 0.75))
    )
    many_layer_values = many_layer_values.at[rows, second_index].set(
        fortran_product(absorbed_ground_snow, jnp.full_like(absorbed_ground_snow, 0.25))
    )
    return jnp.where(one_layer[:, None], one_layer_values, many_layer_values)


def _shallow_lake_snow_layers(
    absorbed_ground: jnp.ndarray,
    snow_layer_count: jnp.ndarray,
    number_of_levels: int,
) -> jnp.ndarray:
    patch_count = absorbed_ground.shape[0]
    rows = jnp.arange(patch_count, dtype=jnp.int32)
    result = jnp.zeros((patch_count, number_of_levels), dtype=jnp.float64)
    one_layer = snow_layer_count == jnp.int32(-1)
    one_layer_values = result.at[:, number_of_levels - 2].set(absorbed_ground)
    first_index = jnp.clip(
        snow_layer_count + jnp.int32(number_of_levels - 1),
        jnp.int32(0),
        jnp.int32(number_of_levels - 1),
    )
    second_index = jnp.clip(
        first_index + jnp.int32(1),
        jnp.int32(0),
        jnp.int32(number_of_levels - 1),
    )
    many_layer_values = result.at[rows, first_index].set(
        fortran_product(absorbed_ground, jnp.full_like(absorbed_ground, 0.75))
    )
    many_layer_values = many_layer_values.at[rows, second_index].set(
        fortran_product(absorbed_ground, jnp.full_like(absorbed_ground, 0.25))
    )
    return jnp.where(one_layer[:, None], one_layer_values, many_layer_values)


def surface_radiation(
    topology: SubgridTopology,
    inputs: SurfaceRadiationInputs,
    state: SurfaceRadiationState,
    current_time_of_day_seconds: jnp.ndarray | int,
    time_step_seconds: jnp.ndarray | int = 1800,
    subgrid_fluxes_enabled: jnp.ndarray | bool = True,
) -> SurfaceRadiationState:

    inputs, state = _cast_and_validate(topology, inputs, state)
    previous_flux = state.fluxes
    previous_diagnostic = state.diagnostics
    patch_gridcell = topology.patch.gridcell
    patch_column = topology.patch.column
    patch_landunit = topology.patch.landunit
    landunit_type = topology.landunit.landunit_type[patch_landunit]
    active = topology.patch.active
    urban = (landunit_type >= LANDUNIT_URBAN_MIN) & (landunit_type <= LANDUNIT_URBAN_MAX)
    nonurban = active & ~urban
    active_urban = active & urban
    soil_or_crop = (landunit_type == LANDUNIT_SOIL) | (landunit_type == LANDUNIT_CROP)
    lake = landunit_type == LANDUNIT_LAKE

    direct = inputs.direct_shortwave_gridcell[patch_gridcell]
    diffuse = inputs.diffuse_shortwave_gridcell[patch_gridcell]
    snow_layer_count = inputs.snow_layer_count_column[patch_column]
    snow_depth = inputs.snow_depth_column[patch_column]
    ground_albedo_direct = inputs.ground_albedo_direct_column[patch_column]
    ground_albedo_diffuse = inputs.ground_albedo_diffuse_column[patch_column]
    soil_albedo_direct = inputs.soil_albedo_direct_column[patch_column]
    soil_albedo_diffuse = inputs.soil_albedo_diffuse_column[patch_column]
    snow_albedo_direct = inputs.snow_albedo_direct_column[patch_column]
    snow_albedo_diffuse = inputs.snow_albedo_diffuse_column[patch_column]

    canopy_direct = fortran_product(direct, inputs.canopy_absorption_direct_patch)
    canopy_diffuse = fortran_product(diffuse, inputs.canopy_absorption_diffuse_patch)
    transmitted_direct = fortran_product(direct, inputs.direct_transmittance_patch)
    transmitted_diffuse = fortran_linear_combination(
        jnp.stack((direct, diffuse)),
        jnp.stack(
            (
                inputs.direct_to_diffuse_transmittance_patch,
                inputs.diffuse_transmittance_patch,
            )
        ),
    )
    absorbed_soil_by_band = fortran_linear_combination(
        jnp.stack((transmitted_direct, transmitted_diffuse)),
        jnp.stack(
            (
                jnp.float64(1.0) - soil_albedo_direct,
                jnp.float64(1.0) - soil_albedo_diffuse,
            )
        ),
    )
    absorbed_snow_by_band = fortran_linear_combination(
        jnp.stack((transmitted_direct, transmitted_diffuse)),
        jnp.stack(
            (
                jnp.float64(1.0) - snow_albedo_direct,
                jnp.float64(1.0) - snow_albedo_diffuse,
            )
        ),
    )
    absorbed_ground_by_band = fortran_linear_combination(
        jnp.stack((transmitted_direct, transmitted_diffuse)),
        jnp.stack(
            (
                jnp.float64(1.0) - ground_albedo_direct,
                jnp.float64(1.0) - ground_albedo_diffuse,
            )
        ),
    )

    def accumulate_band(carry, band_values):
        (
            absorbed_vegetation,
            absorbed_total,
            rural_absorbed,
            absorbed_ground,
            absorbed_soil,
            absorbed_snow,
        ) = carry
        canopy_direct_band, canopy_diffuse_band, ground_band, soil_band, snow_band = band_values
        absorbed_vegetation = fortran_ordered_sum(
            jnp.stack((canopy_direct_band, canopy_diffuse_band)),
            initial=absorbed_vegetation,
        )
        absorbed_total = fortran_ordered_sum(
            jnp.stack((canopy_direct_band, canopy_diffuse_band)),
            initial=absorbed_total,
        )
        rural_absorbed = fortran_ordered_sum(
            jnp.stack((canopy_direct_band, canopy_diffuse_band)),
            initial=rural_absorbed,
        )
        absorbed_ground = absorbed_ground + ground_band
        absorbed_soil = absorbed_soil + soil_band
        absorbed_snow = absorbed_snow + snow_band
        absorbed_total = absorbed_total + ground_band
        rural_absorbed = rural_absorbed + ground_band
        return (
            absorbed_vegetation,
            absorbed_total,
            rural_absorbed,
            absorbed_ground,
            absorbed_soil,
            absorbed_snow,
        ), None

    zero_patch = jnp.zeros(direct.shape[0], dtype=jnp.float64)
    accumulated, _ = lax.scan(
        accumulate_band,
        (zero_patch, zero_patch, zero_patch, zero_patch, zero_patch, zero_patch),
        (
            canopy_direct.T,
            canopy_diffuse.T,
            absorbed_ground_by_band.T,
            absorbed_soil_by_band.T,
            absorbed_snow_by_band.T,
        ),
    )
    (
        absorbed_vegetation_candidate,
        absorbed_total_candidate,
        rural_absorbed_candidate,
        absorbed_ground_candidate,
        absorbed_soil_candidate,
        absorbed_snow_candidate,
    ) = accumulated

    no_snow_or_averaged = (
        (snow_layer_count == jnp.int32(0))
        | lake
        | ~jnp.asarray(subgrid_fluxes_enabled, dtype=jnp.bool_)
    )
    absorbed_soil_candidate = jnp.where(
        no_snow_or_averaged,
        absorbed_ground_candidate,
        absorbed_soil_candidate,
    )
    absorbed_snow_candidate = jnp.where(
        no_snow_or_averaged,
        absorbed_ground_candidate,
        absorbed_snow_candidate,
    )

    number_of_levels = inputs.snow_absorption_direct_visible_column.shape[1]
    level_number = jnp.arange(-number_of_levels + 2, 2, dtype=jnp.int32)
    raw_layer_absorption = fortran_linear_combination(
        jnp.stack(
            (
                jnp.broadcast_to(
                    transmitted_direct[:, 0, None], (direct.shape[0], number_of_levels)
                ),
                jnp.broadcast_to(
                    transmitted_direct[:, 1, None], (direct.shape[0], number_of_levels)
                ),
                jnp.broadcast_to(
                    transmitted_diffuse[:, 0, None], (direct.shape[0], number_of_levels)
                ),
                jnp.broadcast_to(
                    transmitted_diffuse[:, 1, None], (direct.shape[0], number_of_levels)
                ),
            )
        ),
        jnp.stack(
            (
                inputs.snow_absorption_direct_visible_column[patch_column],
                inputs.snow_absorption_direct_nir_column[patch_column],
                inputs.snow_absorption_diffuse_visible_column[patch_column],
                inputs.snow_absorption_diffuse_nir_column[patch_column],
            )
        ),
    )
    active_snow_and_soil = level_number[None, :] >= snow_layer_count[:, None] + 1
    below_surface_snow = level_number[None, :] > snow_layer_count[:, None] + 1
    raw_active_sum = _masked_ordered_layer_sum(raw_layer_absorption, active_snow_and_soil)
    raw_subsurface_sum = _masked_ordered_layer_sum(raw_layer_absorption, below_surface_snow)
    safe_active_sum = jnp.where(raw_active_sum != jnp.float64(0.0), raw_active_sum, 1.0)
    subsurface_fraction_candidate = jnp.where(
        raw_active_sum != jnp.float64(0.0),
        raw_subsurface_sum / safe_active_sum,
        jnp.float64(0.0),
    )

    no_snow_layers = jnp.zeros_like(raw_layer_absorption).at[:, -1].set(
        absorbed_ground_candidate
    )
    redistributed_layers = _redistributed_snow_layers(
        absorbed_snow_candidate,
        snow_layer_count,
        number_of_levels,
    )
    redistribution_needed = (
        jnp.abs(raw_active_sum - absorbed_snow_candidate) > jnp.float64(0.00001)
    )
    snow_layers = jnp.where(
        redistribution_needed[:, None],
        redistributed_layers,
        raw_layer_absorption,
    )
    shallow_layers = _shallow_lake_snow_layers(
        absorbed_ground_candidate,
        snow_layer_count,
        number_of_levels,
    )
    shallow_averaged_snow = (
        (lake | ~jnp.asarray(subgrid_fluxes_enabled, dtype=jnp.bool_))
        & (snow_depth < jnp.float64(0.10))
    )
    snow_layers = jnp.where(shallow_averaged_snow[:, None], shallow_layers, snow_layers)
    layer_candidate = jnp.where(
        (snow_layer_count == jnp.int32(0))[:, None],
        no_snow_layers,
        snow_layers,
    )

    surface_level_index = snow_layer_count + jnp.int32(number_of_levels - 1)
    surface_layer_absorption = layer_candidate[
        jnp.arange(direct.shape[0], dtype=jnp.int32), surface_level_index
    ]
    penetrating_candidate = absorbed_ground_candidate - surface_layer_absorption

    reflected_direct_candidate = fortran_product(
        inputs.surface_albedo_direct_patch,
        direct,
    )
    reflected_diffuse_candidate = fortran_product(
        inputs.surface_albedo_diffuse_patch,
        diffuse,
    )
    reflected_by_band = jnp.stack(
        (
            fortran_ordered_sum(
                jnp.stack((reflected_direct_candidate[:, 0], reflected_diffuse_candidate[:, 0]))
            ),
            fortran_ordered_sum(
                jnp.stack((reflected_direct_candidate[:, 1], reflected_diffuse_candidate[:, 1]))
            ),
        ),
        axis=1,
    )
    reflected_nonurban = fortran_ordered_sum(reflected_by_band.T)
    reflected_urban = fortran_ordered_sum(
        jnp.stack(
            (
                reflected_direct_candidate[:, 0],
                reflected_direct_candidate[:, 1],
                reflected_diffuse_candidate[:, 0],
                reflected_diffuse_candidate[:, 1],
            )
        )
    )
    reflected_total_candidate = jnp.where(urban, reflected_urban, reflected_nonurban)

    visible_par = fortran_ordered_sum(
        jnp.stack((canopy_direct[:, 0], canopy_diffuse[:, 0]))
    )
    near_noon = local_noon_mask(
        topology.grid.longitude_degrees[patch_gridcell],
        current_time_of_day_seconds,
        time_step_seconds,
    )
    special_value = jnp.float64(CLM_SPECIAL_VALUE)
    local_noon_update = active & near_noon
    local_noon_direct_candidate = jnp.where(
        local_noon_update[:, None], direct, special_value
    )
    local_noon_visible_diffuse_candidate = jnp.where(
        local_noon_update,
        diffuse[:, 0],
        special_value,
    )
    local_noon_reflected_candidate = jnp.where(
        local_noon_update[:, None], reflected_direct_candidate, special_value
    )
    local_noon_par_candidate = jnp.where(
        local_noon_update,
        jnp.where(urban, jnp.float64(0.0), visible_par),
        special_value,
    )

    has_snow = snow_layer_count < jnp.int32(0)
    snow_diagnostic_update = nonurban & has_snow
    snow_incident_direct_candidate = jnp.where(
        snow_diagnostic_update[:, None], direct, special_value
    )
    snow_incident_diffuse_candidate = jnp.where(
        snow_diagnostic_update[:, None], diffuse, special_value
    )
    snow_reflected_direct_candidate = jnp.where(
        snow_diagnostic_update[:, None],
        fortran_product(direct, snow_albedo_direct),
        special_value,
    )
    snow_reflected_diffuse_candidate = jnp.where(
        snow_diagnostic_update[:, None],
        fortran_product(diffuse, snow_albedo_diffuse),
        special_value,
    )

    update_any = nonurban | active_urban
    update_rural = nonurban & soil_or_crop
    sunlit_fraction = jnp.where(
        active_urban,
        jnp.float64(0.0),
        previous_flux.sunlit_fraction_patch,
    )
    reflected_total = jnp.where(
        update_any,
        reflected_total_candidate,
        previous_flux.reflected_total_patch,
    )
    fluxes = SurfaceRadiationFluxState(
        sunlit_fraction_patch=sunlit_fraction,
        absorbed_total_patch=jnp.where(
            nonurban, absorbed_total_candidate, previous_flux.absorbed_total_patch
        ),
        reflected_total_patch=reflected_total,
        rural_absorbed_total_patch=jnp.where(
            update_rural,
            rural_absorbed_candidate,
            previous_flux.rural_absorbed_total_patch,
        ),
        absorbed_vegetation_patch=jnp.where(
            nonurban,
            absorbed_vegetation_candidate,
            previous_flux.absorbed_vegetation_patch,
        ),
        absorbed_ground_patch=jnp.where(
            nonurban, absorbed_ground_candidate, previous_flux.absorbed_ground_patch
        ),
        absorbed_ground_soil_patch=jnp.where(
            nonurban,
            absorbed_soil_candidate,
            previous_flux.absorbed_ground_soil_patch,
        ),
        absorbed_ground_snow_patch=jnp.where(
            nonurban,
            absorbed_snow_candidate,
            previous_flux.absorbed_ground_snow_patch,
        ),
        absorbed_ground_layers_patch=jnp.where(
            nonurban[:, None],
            layer_candidate,
            previous_flux.absorbed_ground_layers_patch,
        ),
        penetrating_shortwave_patch=jnp.where(
            update_rural,
            penetrating_candidate,
            previous_flux.penetrating_shortwave_patch,
        ),
        subsurface_snow_absorption_fraction_patch=jnp.where(
            nonurban,
            jnp.where(has_snow, subsurface_fraction_candidate, jnp.float64(0.0)),
            previous_flux.subsurface_snow_absorption_fraction_patch,
        ),
    )
    diagnostics = SurfaceRadiationDiagnosticsState(
        incident_direct_patch=jnp.where(
            update_any[:, None], direct, previous_diagnostic.incident_direct_patch
        ),
        incident_diffuse_patch=jnp.where(
            update_any[:, None], diffuse, previous_diagnostic.incident_diffuse_patch
        ),
        reflected_direct_patch=jnp.where(
            update_any[:, None],
            reflected_direct_candidate,
            previous_diagnostic.reflected_direct_patch,
        ),
        reflected_diffuse_patch=jnp.where(
            update_any[:, None],
            reflected_diffuse_candidate,
            previous_diagnostic.reflected_diffuse_patch,
        ),
        local_noon_incident_direct_patch=jnp.where(
            update_any[:, None],
            local_noon_direct_candidate,
            previous_diagnostic.local_noon_incident_direct_patch,
        ),
        local_noon_incident_visible_diffuse_patch=jnp.where(
            update_any,
            local_noon_visible_diffuse_candidate,
            previous_diagnostic.local_noon_incident_visible_diffuse_patch,
        ),
        local_noon_reflected_direct_patch=jnp.where(
            update_any[:, None],
            local_noon_reflected_candidate,
            previous_diagnostic.local_noon_reflected_direct_patch,
        ),
        local_noon_absorbed_par_patch=jnp.where(
            update_any,
            local_noon_par_candidate,
            previous_diagnostic.local_noon_absorbed_par_patch,
        ),
        snow_incident_direct_patch=jnp.where(
            nonurban[:, None],
            snow_incident_direct_candidate,
            previous_diagnostic.snow_incident_direct_patch,
        ),
        snow_incident_diffuse_patch=jnp.where(
            nonurban[:, None],
            snow_incident_diffuse_candidate,
            previous_diagnostic.snow_incident_diffuse_patch,
        ),
        snow_reflected_direct_patch=jnp.where(
            nonurban[:, None],
            snow_reflected_direct_candidate,
            previous_diagnostic.snow_reflected_direct_patch,
        ),
        snow_reflected_diffuse_patch=jnp.where(
            nonurban[:, None],
            snow_reflected_diffuse_candidate,
            previous_diagnostic.snow_reflected_diffuse_patch,
        ),
    )
    return SurfaceRadiationState(fluxes=fluxes, diagnostics=diagnostics)
