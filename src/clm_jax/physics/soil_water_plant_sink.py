
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.io.restart import COLUMN_URBAN_PERVIOUS_ROAD, SubgridTopology
from clm_jax.physics.numerics import (
    fortran_divide,
    fortran_ordered_sum,
)
from clm_jax.physics.numerics import (
    fortran_hardware_add as fortran_add,
)
from clm_jax.physics.numerics import (
    fortran_hardware_multiply as fortran_multiply,
)
from clm_jax.physics.plant_hydraulics import NUMBER_OF_VEGETATION_SEGMENTS, ROOT
from clm_jax.physics.surface_hydrology import (
    NUMBER_OF_GROUND_LEVELS,
    NUMBER_OF_SOIL_LEVELS,
    build_surface_hydrology_masks,
)

if TYPE_CHECKING:
    from clm_jax.io.cime_case import CimeCase


class SoilWaterPlantSinkParameters(NamedTuple):

    maximum_patches_per_column: int = 79

    @classmethod
    def from_case(cls, case: CimeCase) -> SoilWaterPlantSinkParameters:

        from clm_jax.io.namelist import read_case_namelist_group  

        general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        return cls(maximum_patches_per_column=int(general["maxpatch_pft"]))


DEFAULT_SOIL_WATER_PLANT_SINK_PARAMETERS = SoilWaterPlantSinkParameters()


class SoilWaterPlantSinkInputs(NamedTuple):

    soil_root_conductance_patch_level: jnp.ndarray
    root_fraction_patch_level: jnp.ndarray
    vegetation_transpiration_patch: jnp.ndarray
    vegetation_transpiration_column: jnp.ndarray
    soil_water_potential_column_level: jnp.ndarray
    exposed_vegetation_flag_patch: jnp.ndarray
    vegetation_water_potential_patch_segment: jnp.ndarray
    soil_node_depth_column_level: jnp.ndarray


class SoilWaterPlantSinkState(NamedTuple):

    effective_root_fraction_column_level: jnp.ndarray
    root_soil_water_flux_column_level: jnp.ndarray
    negative_hydraulic_redistribution_flux_column: jnp.ndarray


class SoilWaterPlantSinkDiagnostics(NamedTuple):

    hydrology_column: jnp.ndarray
    hydraulic_stress_column: jnp.ndarray
    pervious_road_column: jnp.ndarray
    hydraulic_stress_total_flux_column: jnp.ndarray


class SoilWaterPlantSinkResult(NamedTuple):
    state: SoilWaterPlantSinkState
    diagnostics: SoilWaterPlantSinkDiagnostics


def _validate_shapes(
    topology: SubgridTopology,
    inputs: SoilWaterPlantSinkInputs,
    state: SoilWaterPlantSinkState,
    parameters: SoilWaterPlantSinkParameters,
) -> None:
    patch_shape = topology.patch.column.shape
    column_shape = topology.column.gridcell.shape
    if len(patch_shape) != 1 or len(column_shape) != 1:
        raise ValueError("patch and column topology must be vectors")
    if parameters.maximum_patches_per_column < 0:
        raise ValueError("maximum_patches_per_column must be nonnegative")

    expected_inputs = {
        "soil_root_conductance_patch_level": (
            patch_shape[0],
            NUMBER_OF_SOIL_LEVELS,
        ),
        "root_fraction_patch_level": (
            patch_shape[0],
            NUMBER_OF_GROUND_LEVELS,
        ),
        "vegetation_transpiration_patch": patch_shape,
        "vegetation_transpiration_column": column_shape,
        "soil_water_potential_column_level": (
            column_shape[0],
            NUMBER_OF_GROUND_LEVELS,
        ),
        "exposed_vegetation_flag_patch": patch_shape,
        "vegetation_water_potential_patch_segment": (
            patch_shape[0],
            NUMBER_OF_VEGETATION_SEGMENTS,
        ),
        "soil_node_depth_column_level": (
            column_shape[0],
            NUMBER_OF_GROUND_LEVELS,
        ),
    }
    for name, expected in expected_inputs.items():
        if getattr(inputs, name).shape != expected:
            raise ValueError(f"{name} must have shape {expected}")

    expected_state = {
        "effective_root_fraction_column_level": (
            column_shape[0],
            NUMBER_OF_GROUND_LEVELS,
        ),
        "root_soil_water_flux_column_level": (
            column_shape[0],
            NUMBER_OF_SOIL_LEVELS,
        ),
        "negative_hydraulic_redistribution_flux_column": column_shape,
    }
    for name, expected in expected_state.items():
        if getattr(state, name).shape != expected:
            raise ValueError(f"{name} must have shape {expected}")


def _contiguous_patch_indices(
    topology: SubgridTopology,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    patch_column = jnp.asarray(topology.patch.column, dtype=jnp.int32)
    number_of_patches = patch_column.shape[0]
    number_of_columns = topology.column.gridcell.shape[0]
    if number_of_patches == 0:
        return (
            jnp.zeros(number_of_columns, dtype=jnp.int32),
            jnp.zeros(number_of_columns, dtype=jnp.int32),
            jnp.zeros(number_of_columns, dtype=jnp.bool_),
        )

    patch_index = jnp.arange(number_of_patches, dtype=jnp.int32)
    first_patch = jnp.full(number_of_columns, number_of_patches, dtype=jnp.int32)
    first_patch = first_patch.at[patch_column].min(patch_index)
    bounded_first = jnp.clip(first_patch, 0, number_of_patches - 1)
    has_patch = (first_patch < number_of_patches) & (
        patch_column[bounded_first] == jnp.arange(number_of_columns, dtype=jnp.int32)
    )
    return first_patch, bounded_first, has_patch


def _accumulate_patch_sinks_reference(
    inputs, first_patch, patch_column, patch_active, patch_weight,
    hydraulic_stress, pervious_road, maximum_patches,
):
    number_of_patches = patch_column.size
    number_of_columns = first_patch.size
    column_index = jnp.arange(number_of_columns, dtype=jnp.int32)
    candidate_root_flux = jnp.zeros((number_of_columns, NUMBER_OF_SOIL_LEVELS), jnp.float64)
    candidate_road_root_fraction = jnp.zeros_like(candidate_root_flux)
    road_transpiration_total = jnp.zeros(number_of_columns, jnp.float64)

    if number_of_patches == 0:
        return candidate_root_flux, candidate_road_root_fraction, road_transpiration_total

    def accumulate_patch_slot(
        offset: int,
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        root_flux, road_root_numerator, road_transpiration = carry
        unbounded_patch = first_patch + jnp.int32(offset)
        bounded_patch = jnp.clip(unbounded_patch, 0, number_of_patches - 1)
        belongs = (unbounded_patch < number_of_patches) & (
            patch_column[bounded_patch] == column_index
        )
        active = belongs & patch_active[bounded_patch]
        weight = patch_weight[bounded_patch]

        gravitational_potential = fortran_multiply(
            inputs.soil_node_depth_column_level[:, :NUMBER_OF_SOIL_LEVELS],
            jnp.float64(1000.0),
        )
        potential_difference = fortran_add(
            fortran_add(
                inputs.soil_water_potential_column_level[
                    :, :NUMBER_OF_SOIL_LEVELS
                ],
                -inputs.vegetation_water_potential_patch_segment[
                    bounded_patch,
                    ROOT,
                ][:, None],
            ),
            -gravitational_potential,
        )
        hydraulic_contribution = fortran_multiply(
            fortran_multiply(
                inputs.soil_root_conductance_patch_level[bounded_patch],
                potential_difference,
            ),
            weight[:, None],
        )
        hydraulic_contributes = (
            active
            & hydraulic_stress
            & (inputs.exposed_vegetation_flag_patch[bounded_patch] > 0)
            & (weight > jnp.float64(0.0))
        )
        updated_root_flux = fortran_add(root_flux, hydraulic_contribution)
        root_flux = jnp.where(
            hydraulic_contributes[:, None],
            updated_root_flux,
            root_flux,
        )

        transpiration = inputs.vegetation_transpiration_patch[bounded_patch]
        road_weighted_transpiration = fortran_multiply(transpiration, weight)
        road_contributes = active & pervious_road
        road_root_contribution = fortran_multiply(
            fortran_multiply(
                inputs.root_fraction_patch_level[
                    bounded_patch,
                    :NUMBER_OF_SOIL_LEVELS,
                ],
                transpiration[:, None],
            ),
            weight[:, None],
        )
        updated_road_root_numerator = fortran_add(
            road_root_numerator,
            road_root_contribution,
        )
        road_root_numerator = jnp.where(
            road_contributes[:, None],
            updated_road_root_numerator,
            road_root_numerator,
        )
        updated_road_transpiration = fortran_add(
            road_transpiration,
            road_weighted_transpiration,
        )
        road_transpiration = jnp.where(
            road_contributes,
            updated_road_transpiration,
            road_transpiration,
        )
        return root_flux, road_root_numerator, road_transpiration

    (
        candidate_root_flux,
        candidate_road_root_fraction,
        road_transpiration_total,
    ) = lax.fori_loop(
        0,
        maximum_patches,
        accumulate_patch_slot,
        (
            candidate_root_flux,
            candidate_road_root_fraction,
            road_transpiration_total,
        ),
    )

    return candidate_root_flux, candidate_road_root_fraction, road_transpiration_total


def compute_effective_root_fraction_and_vertical_sink(
    topology: SubgridTopology,
    inputs: SoilWaterPlantSinkInputs,
    state: SoilWaterPlantSinkState,
    *,
    parameters: SoilWaterPlantSinkParameters = (
        DEFAULT_SOIL_WATER_PLANT_SINK_PARAMETERS
    ),
) -> SoilWaterPlantSinkResult:

    _validate_shapes(topology, inputs, state, parameters)
    masks = build_surface_hydrology_masks(topology)
    column_type = jnp.asarray(topology.column.column_type, dtype=jnp.int32)
    hydrology = masks.hydrology_column
    pervious_road = hydrology & (
        column_type == jnp.int32(COLUMN_URBAN_PERVIOUS_ROAD)
    )
    hydraulic_stress = hydrology & ~pervious_road

    number_of_patches = topology.patch.column.shape[0]
    number_of_columns = topology.column.gridcell.shape[0]
    candidate_root_flux = jnp.zeros(
        (number_of_columns, NUMBER_OF_SOIL_LEVELS),
        dtype=jnp.float64,
    )
    candidate_road_root_fraction = jnp.zeros_like(candidate_root_flux)
    road_transpiration_total = jnp.zeros(number_of_columns, dtype=jnp.float64)

    if number_of_patches > 0:
        patch_column = jnp.asarray(topology.patch.column, dtype=jnp.int32)
        patch_active = jnp.asarray(topology.patch.active, dtype=jnp.bool_)
        patch_weight = jnp.asarray(
            topology.patch.weight_on_column,
            dtype=jnp.float64,
        )
        first_patch, _, _ = _contiguous_patch_indices(topology)

        if jax.default_backend() == "gpu":
            from clm_jax.physics.soil_water_plant_sink_gpu import accumulate
        else:
            accumulate = _accumulate_patch_sinks_reference
        (candidate_root_flux, candidate_road_root_fraction,
         road_transpiration_total) = accumulate(
            inputs, first_patch, patch_column, patch_active, patch_weight,
            hydraulic_stress, pervious_road, parameters.maximum_patches_per_column,
        )

    def accumulate_negative_flux(level: int, total: jnp.ndarray) -> jnp.ndarray:
        layer_flux = candidate_root_flux[:, level]
        negative = layer_flux < jnp.float64(0.0)
        return jnp.where(
            negative,
            fortran_add(total, layer_flux),
            total,
        )

    negative_hydraulic_flux = lax.fori_loop(
        0,
        NUMBER_OF_SOIL_LEVELS,
        accumulate_negative_flux,
        jnp.zeros(number_of_columns, dtype=jnp.float64),
    )
    hydraulic_total = fortran_ordered_sum(jnp.swapaxes(candidate_root_flux, 0, 1))
    positive_total = hydraulic_total > jnp.float64(0.0)
    safe_hydraulic_total = jnp.where(
        positive_total,
        hydraulic_total,
        jnp.float64(1.0),
    )
    hydraulic_root_fraction = fortran_divide(
        candidate_root_flux,
        safe_hydraulic_total[:, None],
    )

    road_has_transpiration = road_transpiration_total != jnp.float64(0.0)
    safe_road_transpiration = jnp.where(
        road_has_transpiration,
        road_transpiration_total,
        jnp.float64(1.0),
    )
    road_root_fraction = jnp.where(
        road_has_transpiration[:, None],
        fortran_divide(
            candidate_road_root_fraction,
            safe_road_transpiration[:, None],
        ),
        candidate_road_root_fraction,
    )
    road_root_flux = fortran_multiply(
        road_root_fraction,
        inputs.vegetation_transpiration_column[:, None],
    )

    root_fraction = state.effective_root_fraction_column_level
    hydraulic_first_twenty = jnp.where(
        positive_total[:, None],
        hydraulic_root_fraction,
        jnp.float64(0.0),
    )
    root_fraction = root_fraction.at[:, :NUMBER_OF_SOIL_LEVELS].set(
        jnp.where(
            hydraulic_stress[:, None],
            hydraulic_first_twenty,
            root_fraction[:, :NUMBER_OF_SOIL_LEVELS],
        )
    )
    root_fraction = jnp.where(
        (hydraulic_stress & ~positive_total)[:, None],
        jnp.float64(0.0),
        root_fraction,
    )
    root_fraction = root_fraction.at[:, :NUMBER_OF_SOIL_LEVELS].set(
        jnp.where(
            pervious_road[:, None],
            road_root_fraction,
            root_fraction[:, :NUMBER_OF_SOIL_LEVELS],
        )
    )

    root_flux = jnp.where(
        hydraulic_stress[:, None],
        candidate_root_flux,
        state.root_soil_water_flux_column_level,
    )
    root_flux = jnp.where(pervious_road[:, None], road_root_flux, root_flux)
    negative_flux = jnp.where(
        hydraulic_stress,
        negative_hydraulic_flux,
        state.negative_hydraulic_redistribution_flux_column,
    )

    return SoilWaterPlantSinkResult(
        state=SoilWaterPlantSinkState(
            effective_root_fraction_column_level=root_fraction,
            root_soil_water_flux_column_level=root_flux,
            negative_hydraulic_redistribution_flux_column=negative_flux,
        ),
        diagnostics=SoilWaterPlantSinkDiagnostics(
            hydrology_column=hydrology,
            hydraulic_stress_column=hydraulic_stress,
            pervious_road_column=pervious_road,
            hydraulic_stress_total_flux_column=jnp.where(
                hydraulic_stress,
                hydraulic_total,
                jnp.float64(0.0),
            ),
        ),
    )


__all__ = [
    "DEFAULT_SOIL_WATER_PLANT_SINK_PARAMETERS",
    "SoilWaterPlantSinkDiagnostics",
    "SoilWaterPlantSinkInputs",
    "SoilWaterPlantSinkParameters",
    "SoilWaterPlantSinkResult",
    "SoilWaterPlantSinkState",
    "compute_effective_root_fraction_and_vertical_sink",
]
