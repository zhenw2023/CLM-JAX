
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from clm_jax.io.canopy_hydraulic_inputs import bedrock_level_count_column
from clm_jax.io.canopy_temperature_restart import soil_layer_geometry_20sl_8p5m
from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group
from clm_jax.io.netcdf import load_dimension_sizes, load_native_variables
from clm_jax.io.plant_hydraulics import load_plant_hydraulic_parameters
from clm_jax.io.restart import (
    LANDUNIT_CROP,
    LANDUNIT_SOIL,
    SubgridTopology,
    configured_land_restart,
    load_subgrid_topology,
)
from clm_jax.physics.plant_hydraulics import jackson_root_fraction
from clm_jax.physics.soil_biogeochem_vertical_profile import (
    NUMBER_OF_GROUND_LEVELS,
    SoilBiogeochemVerticalProfileInputs,
    SoilBiogeochemVerticalProfileParameters,
)

NUMBER_OF_HYDROLOGIC_SOIL_LEVELS = 20


class SoilBiogeochemVerticalProfileNativeData(NamedTuple):

    topology: SubgridTopology
    soil_column: jnp.ndarray
    soil_patch: jnp.ndarray
    column_patch_indices: jnp.ndarray
    carbon_root_fraction_patch_level: jnp.ndarray
    carbon_root_profile_beta_patch: jnp.ndarray
    active_layer_index_column: jnp.ndarray
    parameters: SoilBiogeochemVerticalProfileParameters
    restart_path: Path
    surface_data_path: Path
    parameter_data_path: Path


def _selected_input_path(
    case: CimeCase,
    key: str,
    override: str | Path | None,
) -> Path:
    if override is None:
        group = read_case_namelist_group(case, "lnd_in", "clm_inparm")
        declared = group.get(key)
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError(f"clm_inparm in {case.case_docs / 'lnd_in'} lacks {key}")
        path = Path(case.expand(declared)).expanduser()
        if not path.is_absolute():
            path = case.root / path
        path = path.resolve()
    else:
        path = Path(override).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured CLM {key} does not exist: {path}")
    return path


def _carbon_root_variant(case: CimeCase, number_of_variants: int) -> int:
    roots = read_case_namelist_group(case, "lnd_in", "rooting_profile_inparm")
    if int(roots.get("rooting_profile_method_carbon", -1)) != 1:
        raise ValueError("native vertical-profile adapter requires Jackson1996 roots")
    variant = int(roots.get("rooting_profile_varindex_carbon", 2))
    if variant < 1 or variant > number_of_variants:
        raise ValueError("carbon-root profile variant lies outside rootprof_beta")
    return variant


def _column_patch_matrix(
    patch_column: np.ndarray,
    number_of_columns: int,
) -> np.ndarray:

    if patch_column.ndim != 1:
        raise ValueError("patch column topology must be a vector")
    if np.any((patch_column < 0) | (patch_column >= number_of_columns)):
        raise ValueError("patch-to-column topology is out of bounds")
    if patch_column.size and np.any(patch_column[1:] < patch_column[:-1]):
        raise ValueError("CLM patches must be contiguous and ordered by column")
    counts = np.bincount(patch_column, minlength=number_of_columns)
    maximum = int(counts.max(initial=0))
    matrix = np.full((number_of_columns, maximum), -1, dtype=np.int32)
    if patch_column.size:
        starts = np.cumsum(counts, dtype=np.int64) - counts
        slot = np.arange(patch_column.size, dtype=np.int64) - starts[patch_column]
        matrix[patch_column, slot] = np.arange(patch_column.size, dtype=np.int32)
    return matrix


def load_soil_biogeochem_vertical_profile_native_data(
    case: CimeCase,
    restart_path: str | Path | None = None,
    surface_data_path: str | Path | None = None,
    parameter_data_path: str | Path | None = None,
    *,
    include_inactive: bool = False,
) -> SoilBiogeochemVerticalProfileNativeData:

    general = read_case_namelist_group(case, "lnd_in", "clm_inparm")
    if general.get("use_cn") is not True:
        raise ValueError("native vertical-profile adapter requires use_cn=.true.")
    if general.get("use_fates") is not False:
        raise ValueError("native vertical-profile adapter requires use_fates=.false.")
    if general.get("soil_layerstruct") != "20SL_8.5m":
        raise ValueError("native vertical-profile adapter requires soil_layerstruct=20SL_8.5m")

    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    surface_data = _selected_input_path(case, "fsurdat", surface_data_path)
    parameter_data = _selected_input_path(case, "paramfile", parameter_data_path)
    dimensions = load_dimension_sizes(restart)
    if dimensions.get("levgrnd") != NUMBER_OF_GROUND_LEVELS:
        raise ValueError("restart must contain the native 25 ground levels")

    topology = load_subgrid_topology(restart)
    number_of_columns = topology.column.gridcell.shape[0]
    number_of_patches = topology.patch.gridcell.shape[0]
    column_landunit = np.asarray(topology.column.landunit_type, dtype=np.int32)
    patch_landunit = np.asarray(topology.patch.landunit_type, dtype=np.int32)
    soil_column = (column_landunit == LANDUNIT_SOIL) | (
        column_landunit == LANDUNIT_CROP
    )
    soil_patch = (patch_landunit == LANDUNIT_SOIL) | (patch_landunit == LANDUNIT_CROP)
    if not include_inactive:
        soil_column &= np.asarray(topology.column.active, dtype=bool)
        soil_patch &= np.asarray(topology.patch.active, dtype=bool)
    patch_column = np.asarray(topology.patch.column, dtype=np.int32)
    column_patch_indices = _column_patch_matrix(patch_column, number_of_columns)
    if not np.all(soil_column[patch_column[soil_patch]]):
        raise ValueError("filtered soil patches must map to filtered soil columns")

    values = load_native_variables(
        restart,
        ["altmax_lastyear_indx"],
        masked_integer_fill_value=-9999,
    )
    active_layer = np.asarray(values["altmax_lastyear_indx"], dtype=np.int32)
    if active_layer.shape != (number_of_columns,):
        raise ValueError("restart altmax_lastyear_indx must contain every column")
    if np.any((active_layer[soil_column] < 0) | (active_layer[soil_column] > 25)):
        raise ValueError("filtered soil columns contain invalid active-layer indices")

    vegetation_type = np.asarray(topology.patch.vegetation_type, dtype=np.int32)
    plant_parameters = load_plant_hydraulic_parameters(case, parameter_data)
    root_profiles = np.asarray(
        plant_parameters.root_profile_beta_pft_variant,
        dtype=np.float64,
    )
    if np.any((vegetation_type < 0) | (vegetation_type >= root_profiles.shape[0])):
        raise ValueError("restart vegetation type lies outside rootprof_beta")
    root_variant = _carbon_root_variant(case, root_profiles.shape[1])
    bedrock_column = bedrock_level_count_column(topology, surface_data)
    bedrock_patch = np.asarray(bedrock_column, dtype=np.int32)[patch_column]
    node, interface, thickness = soil_layer_geometry_20sl_8p5m()
    hydrologic_root_fraction = jackson_root_fraction(
        jnp.asarray(root_profiles[vegetation_type, root_variant - 1]),
        jnp.asarray(
            interface[: NUMBER_OF_HYDROLOGIC_SOIL_LEVELS + 1],
            dtype=jnp.float64,
        ),
        jnp.asarray(bedrock_patch, dtype=jnp.int32),
    )
    carbon_root_fraction = jnp.concatenate(
        (
            hydrologic_root_fraction,
            jnp.zeros(
                (
                    number_of_patches,
                    NUMBER_OF_GROUND_LEVELS - NUMBER_OF_HYDROLOGIC_SOIL_LEVELS,
                ),
                dtype=jnp.float64,
            ),
        ),
        axis=1,
    )

    use_vertical = general.get("use_vertsoilc") is True
    full_level_count = 25 if use_vertical else 1
    parameters = SoilBiogeochemVerticalProfileParameters(
        decomposition_level_count=20 if use_vertical else 1,
        decomposition_full_level_count=full_level_count,
        use_vertical_soil_carbon=use_vertical,
        use_bedrock=general.get("use_bedrock") is True,
        not_vegetated_pft=0,
        surface_profile_exponent=np.float64(general.get("surfprof_exp", 10.0)),
        minimum_bedrock_depth=np.float64(0.4),
        soil_node_depth_level=jnp.asarray(node),
        decomposition_layer_thickness_level=jnp.asarray(thickness),
    )
    return SoilBiogeochemVerticalProfileNativeData(
        topology=topology,
        soil_column=jnp.asarray(soil_column),
        soil_patch=jnp.asarray(soil_patch),
        column_patch_indices=jnp.asarray(column_patch_indices),
        carbon_root_fraction_patch_level=carbon_root_fraction,
        carbon_root_profile_beta_patch=jnp.asarray(
            root_profiles[vegetation_type, root_variant - 1]
        ),
        active_layer_index_column=jnp.asarray(active_layer),
        parameters=parameters,
        restart_path=restart,
        surface_data_path=surface_data,
        parameter_data_path=parameter_data,
    )


def assemble_soil_biogeochem_vertical_profile_inputs(
    native: SoilBiogeochemVerticalProfileNativeData,
    *,
    active_layer_index_column: jnp.ndarray | None = None,
    carbon_root_fraction_patch_level: jnp.ndarray | None = None,
) -> SoilBiogeochemVerticalProfileInputs:

    active_layer = (
        native.active_layer_index_column
        if active_layer_index_column is None
        else jnp.asarray(active_layer_index_column, dtype=jnp.int32)
    )
    root_fraction = (
        native.carbon_root_fraction_patch_level
        if carbon_root_fraction_patch_level is None
        else jnp.asarray(carbon_root_fraction_patch_level, dtype=jnp.float64)
    )
    number_of_columns = native.topology.column.gridcell.shape[0]
    number_of_patches = native.topology.patch.gridcell.shape[0]
    if active_layer.shape != (number_of_columns,):
        raise ValueError("active_layer_index_column must contain every native column")
    if root_fraction.shape != (number_of_patches, NUMBER_OF_GROUND_LEVELS):
        raise ValueError("carbon_root_fraction_patch_level must have shape (patch, 25)")
    return SoilBiogeochemVerticalProfileInputs(
        soil_column=native.soil_column,
        soil_patch=native.soil_patch,
        patch_column_index=native.topology.patch.column,
        column_patch_indices=native.column_patch_indices,
        vegetation_type_patch=native.topology.patch.vegetation_type,
        patch_weight_on_column=native.topology.patch.weight_on_column,
        carbon_root_fraction_patch_level=root_fraction,
        active_layer_index_column=active_layer,
    )


__all__ = [
    "NUMBER_OF_HYDROLOGIC_SOIL_LEVELS",
    "SoilBiogeochemVerticalProfileNativeData",
    "assemble_soil_biogeochem_vertical_profile_inputs",
    "load_soil_biogeochem_vertical_profile_native_data",
]
