
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, TypeVar

import numpy as np

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.ecosystem_fast_boundary import (
    EcosystemFastColumn,
    EcosystemFastGridcell,
    EcosystemFastPatch,
)
from clm_jax.io.restart import (
    SubgridTopology,
    configured_land_restart,
    load_subgrid_topology,
)

_MAGIC = b"CLMJAXFASTV0003 "
_VERSION = 3


class EcosystemFastTraceBounds(NamedTuple):
    gridcell_begin: int
    gridcell_end: int
    column_begin: int
    column_end: int
    patch_begin: int
    patch_end: int


EcosystemFastTraceGridcell = EcosystemFastGridcell
EcosystemFastTraceColumn = EcosystemFastColumn
EcosystemFastTracePatch = EcosystemFastPatch


class EcosystemDynamicsFastTrace(NamedTuple):
    path: Path
    rank: int
    bounds: EcosystemFastTraceBounds
    gridcell: EcosystemFastTraceGridcell
    column: EcosystemFastTraceColumn
    patch: EcosystemFastTracePatch


class EcosystemDynamicsFastTraceDirectory(NamedTuple):

    directory: Path
    trace_paths: tuple[Path, ...]
    rank_bounds: tuple[EcosystemFastTraceBounds, ...]
    bounds: EcosystemFastTraceBounds
    gridcell: EcosystemFastTraceGridcell
    column: EcosystemFastTraceColumn
    patch: EcosystemFastTracePatch

    @property
    def rank_count(self) -> int:
        return len(self.trace_paths)


class EcosystemDynamicsFastTraceNativeData(NamedTuple):

    snapshot: EcosystemDynamicsFastTraceDirectory
    alignment: EcosystemFastTraceAlignment
    topology: SubgridTopology
    restart_path: Path


class EcosystemFastTraceAlignment(NamedTuple):

    gridcell_source_index: np.ndarray
    column_source_index: np.ndarray
    patch_source_index: np.ndarray


class _Reader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = path.read_bytes()
        self.offset = 0
        self.byte_order: str | None = None

    def bytes(self, count: int) -> bytes:
        end = self.offset + count
        if end > len(self.data):
            raise ValueError(
                f"truncated ecosystem fast trace at byte {self.offset}: {self.path}"
            )
        result = self.data[self.offset : end]
        self.offset = end
        return result

    def header(self, count: int) -> np.ndarray:
        raw = self.bytes(np.dtype("i4").itemsize * count)
        for byte_order in (">", "<"):
            values = np.frombuffer(raw, dtype=f"{byte_order}i4")
            if int(values[0]) == _VERSION:
                self.byte_order = byte_order
                return values.astype(np.int64)
        little_version = int(np.frombuffer(raw[:4], dtype="<i4")[0])
        big_version = int(np.frombuffer(raw[:4], dtype=">i4")[0])
        raise ValueError(
            "unsupported ecosystem fast trace version "
            f"(little-endian={little_version}, big-endian={big_version})"
        )

    def array(self, typecode: str, count: int) -> np.ndarray:
        if self.byte_order is None:
            raise RuntimeError("ecosystem fast trace byte order is not initialized")
        native_dtype = np.dtype(typecode)
        disk_dtype = np.dtype(f"{self.byte_order}{typecode}")
        values = np.frombuffer(
            self.bytes(disk_dtype.itemsize * count), dtype=disk_dtype
        )
        return values.astype(native_dtype, copy=True)

    def integer(self, count: int) -> np.ndarray:
        return self.array("i4", count).astype(np.int64)

    def real(self, count: int) -> np.ndarray:
        return self.array("f8", count)

    def real_matrix(self, rows: int, columns: int) -> np.ndarray:
        return self.real(rows * columns).reshape((rows, columns), order="F")

    def finish(self) -> None:
        if self.offset != len(self.data):
            raise ValueError(
                f"ecosystem fast trace has {len(self.data) - self.offset} trailing bytes: "
                f"{self.path}"
            )


def _positive_count(name: str, begin: int, end: int) -> int:
    count = end - begin + 1
    if begin < 1 or count < 1:
        raise ValueError(f"invalid {name} bounds [{begin}, {end}]")
    return count


def load_ecosystem_dynamics_fast_trace(
    path: str | Path,
) -> EcosystemDynamicsFastTrace:

    trace_path = Path(path).expanduser().resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(f"ecosystem fast trace does not exist: {trace_path}")
    reader = _Reader(trace_path)
    if reader.bytes(len(_MAGIC)) != _MAGIC:
        raise ValueError(f"not a CLM-JAX ecosystem fast trace: {trace_path}")
    header = reader.header(10)
    version, rank = (int(header[0]), int(header[1]))
    if version != _VERSION:
        raise ValueError(f"unsupported ecosystem fast trace version {version}")
    if rank < 0:
        raise ValueError("ecosystem fast trace rank must be nonnegative")
    bounds = EcosystemFastTraceBounds(*(int(value) for value in header[2:8]))
    ground_level_count, soil_level_count = (int(header[8]), int(header[9]))
    if (ground_level_count, soil_level_count) != (25, 25):
        raise ValueError(
            "ecosystem fast trace must contain the native 25-level CLM geometry"
        )
    gridcell_count = _positive_count(
        "gridcell", bounds.gridcell_begin, bounds.gridcell_end
    )
    column_count = _positive_count("column", bounds.column_begin, bounds.column_end)
    patch_count = _positive_count("patch", bounds.patch_begin, bounds.patch_end)

    gridcell = EcosystemFastTraceGridcell(
        domain_index=reader.integer(gridcell_count),
        relative_humidity=reader.real(gridcell_count),
        wind_speed=reader.real(gridcell_count),
        nitrogen_deposition=reader.real(gridcell_count),
    )
    column = EcosystemFastTraceColumn(
        domain_gridcell=reader.integer(column_count),
        column_type=reader.integer(column_count),
        landunit_type=reader.integer(column_count),
        active=reader.integer(column_count).astype(bool),
        weight_on_gridcell=reader.real(column_count),
        downscaled_temperature=reader.real(column_count),
        downscaled_rain=reader.real(column_count),
        downscaled_snow=reader.real(column_count),
        annual_evapotranspiration=reader.real(column_count),
        topsoil_water_fraction=reader.real(column_count),
        snow_depth=reader.real(column_count),
        saturated_area_fraction=reader.real(column_count),
        inundated_fraction=reader.real(column_count),
        soil_temperature_17cm=reader.real(column_count),
        maximum_thaw_depth=reader.real(column_count),
        previous_maximum_thaw_depth=reader.real(column_count),
        active_layer_index=reader.integer(column_count),
        soil_water_potential=reader.real_matrix(column_count, ground_level_count),
        soil_liquid_water=reader.real_matrix(column_count, ground_level_count),
        soil_volumetric_water=reader.real_matrix(
            column_count, ground_level_count
        ),
        soil_temperature=reader.real_matrix(column_count, ground_level_count),
        oxygen_stress_saturated=reader.real_matrix(column_count, soil_level_count),
        oxygen_stress_unsaturated=reader.real_matrix(column_count, soil_level_count),
        total_litter_carbon=reader.real(column_count),
        total_soil_organic_carbon=reader.real(column_count),
        total_vegetation_carbon=reader.real(column_count),
    )
    patch = EcosystemFastTracePatch(
        domain_gridcell=reader.integer(patch_count),
        vegetation_type=reader.integer(patch_count),
        column_domain_gridcell=reader.integer(patch_count),
        column_type=reader.integer(patch_count),
        landunit_type=reader.integer(patch_count),
        active=reader.integer(patch_count).astype(bool),
        weight_on_gridcell=reader.real(patch_count),
        weight_on_column=reader.real(patch_count),
        vegetation_exposed=reader.integer(patch_count),
        sunlit_leaf_area=reader.real(patch_count),
        shaded_leaf_area=reader.real(patch_count),
        total_leaf_area=reader.real(patch_count),
        sunlit_leaf_respiration=reader.real(patch_count),
        shaded_leaf_respiration=reader.real(patch_count),
        sunlit_photosynthesis=reader.real(patch_count),
        shaded_photosynthesis=reader.real(patch_count),
        transpiration_wetness=reader.real(patch_count),
        root_zone_wetness=reader.real(patch_count),
        precipitation_60day=reader.real(patch_count),
        precipitation_10day=reader.real(patch_count),
        relative_humidity_30day=reader.real(patch_count),
        reference_temperature_2m=reader.real(patch_count),
        minimum_temperature_2m=reader.real(patch_count),
        maximum_temperature_2m=reader.real(patch_count),
        temperature_10day=reader.real(patch_count),
        minimum_temperature_5day=reader.real(patch_count),
        minimum_temperature_10day=reader.real(patch_count),
        growing_degree_days_base_0=reader.real(patch_count),
        growing_degree_days_base_8=reader.real(patch_count),
        growing_degree_days_base_10=reader.real(patch_count),
        growing_degree_days_base_0_twenty_year=reader.real(patch_count),
        growing_degree_days_base_8_twenty_year=reader.real(patch_count),
        growing_degree_days_base_10_twenty_year=reader.real(patch_count),
        years_crop_active=reader.integer(patch_count),
        crop_live=reader.integer(patch_count).astype(bool),
        crop_planted=reader.integer(patch_count).astype(bool),
        harvest_date=reader.integer(patch_count),
        fertilizer_nitrogen=reader.real(patch_count),
        growing_degree_days_since_planting=reader.real(patch_count),
        growing_degree_days_topsoil=reader.real(patch_count),
        carbon_root_fraction=reader.real_matrix(patch_count, ground_level_count),
    )
    reader.finish()
    if np.unique(gridcell.domain_index).size != gridcell_count:
        raise ValueError("ecosystem fast trace repeats a rank-local gridcell")
    return EcosystemDynamicsFastTrace(
        path=trace_path,
        rank=rank,
        bounds=bounds,
        gridcell=gridcell,
        column=column,
        patch=patch,
    )


_Group = TypeVar(
    "_Group",
    EcosystemFastTraceGridcell,
    EcosystemFastTraceColumn,
    EcosystemFastTracePatch,
)


def _concatenate_group(
    traces: tuple[EcosystemDynamicsFastTrace, ...],
    attribute: str,
    group_type: type[_Group],
) -> _Group:
    first = getattr(traces[0], attribute)
    return group_type(
        *(
            np.concatenate(
                [np.asarray(getattr(getattr(trace, attribute), name)) for trace in traces],
                axis=0,
            )
            for name in first._fields
        )
    )


def _global_end(
    traces: tuple[EcosystemDynamicsFastTrace, ...],
    entity: str,
) -> int:
    begin_attribute = f"{entity}_begin"
    end_attribute = f"{entity}_end"
    expected_begin = 1
    for trace in traces:
        begin = int(getattr(trace.bounds, begin_attribute))
        end = int(getattr(trace.bounds, end_attribute))
        if begin != expected_begin:
            raise ValueError(
                f"ecosystem fast trace {entity} bounds are not contiguous: "
                f"rank {trace.rank} begins at {begin}, expected {expected_begin}"
            )
        expected_begin = end + 1
    return expected_begin - 1


def _unique_rows(name: str, *arrays: np.ndarray) -> None:
    keys = np.column_stack(arrays)
    if np.unique(keys, axis=0).shape[0] != keys.shape[0]:
        raise ValueError(f"global ecosystem fast trace repeats a {name} topology key")


def load_ecosystem_dynamics_fast_trace_directory(
    trace_directory: str | Path,
    *,
    expected_rank_count: int | None = None,
) -> EcosystemDynamicsFastTraceDirectory:

    directory = Path(trace_directory).expanduser().resolve()
    paths = tuple(sorted(directory.glob("clm_jax_predrain_fast.rank*.bin")))
    if not paths:
        raise FileNotFoundError(f"no ecosystem fast traces found in {directory}")
    loaded = tuple(load_ecosystem_dynamics_fast_trace(path) for path in paths)
    ranks = tuple(trace.rank for trace in loaded)
    if len(set(ranks)) != len(ranks):
        raise ValueError("ecosystem fast trace directory contains duplicate ranks")
    expected = len(ranks) if expected_rank_count is None else expected_rank_count
    if set(ranks) != set(range(expected)):
        raise ValueError(
            f"ecosystem fast trace ranks differ: got={sorted(ranks)}, "
            f"expected=0..{expected - 1}"
        )
    traces = tuple(sorted(loaded, key=lambda item: item.rank))
    gridcell_end = _global_end(traces, "gridcell")
    column_end = _global_end(traces, "column")
    patch_end = _global_end(traces, "patch")
    gridcell = _concatenate_group(traces, "gridcell", EcosystemFastTraceGridcell)
    column = _concatenate_group(traces, "column", EcosystemFastTraceColumn)
    patch = _concatenate_group(traces, "patch", EcosystemFastTracePatch)
    if np.unique(gridcell.domain_index).size != gridcell_end:
        raise ValueError("global ecosystem fast trace repeats a gridcell domain index")
    _unique_rows(
        "column",
        column.domain_gridcell,
        column.column_type,
        column.landunit_type,
    )
    _unique_rows(
        "patch",
        patch.domain_gridcell,
        patch.vegetation_type,
        patch.column_domain_gridcell,
        patch.column_type,
        patch.landunit_type,
    )
    return EcosystemDynamicsFastTraceDirectory(
        directory=directory,
        trace_paths=tuple(trace.path for trace in traces),
        rank_bounds=tuple(trace.bounds for trace in traces),
        bounds=EcosystemFastTraceBounds(
            1,
            gridcell_end,
            1,
            column_end,
            1,
            patch_end,
        ),
        gridcell=gridcell,
        column=column,
        patch=patch,
    )


def _require_equal(name: str, expected: np.ndarray, actual: np.ndarray) -> None:
    if expected.shape != actual.shape or not np.array_equal(expected, actual):
        raise ValueError(
            f"ecosystem fast trace {name} disagrees with configured restart topology"
        )


def _require_roundoff_close(
    name: str,
    expected: np.ndarray,
    actual: np.ndarray,
) -> None:
    tolerance = 8.0 * np.finfo(np.float64).eps
    if (
        expected.shape != actual.shape
        or not np.all(np.isfinite(expected))
        or not np.all(np.isfinite(actual))
        or not np.allclose(expected, actual, rtol=0.0, atol=tolerance)
    ):
        raise ValueError(
            f"ecosystem fast trace {name} differs from the restart beyond "
            "same-step binary64 normalization roundoff"
        )


def _alignment_order(
    name: str,
    source_arrays: tuple[np.ndarray, ...],
    target_arrays: tuple[np.ndarray, ...],
) -> np.ndarray:
    source = np.column_stack(source_arrays).astype(np.int64, copy=False)
    target = np.column_stack(target_arrays).astype(np.int64, copy=False)
    if source.shape != target.shape:
        raise ValueError(
            f"ecosystem fast trace {name} count differs from configured restart topology"
        )
    source_sort = np.lexsort(tuple(source[:, index] for index in reversed(range(source.shape[1]))))
    target_sort = np.lexsort(tuple(target[:, index] for index in reversed(range(target.shape[1]))))
    if not np.array_equal(source[source_sort], target[target_sort]):
        raise ValueError(
            f"ecosystem fast trace {name} keys disagree with configured restart topology"
        )
    order = np.empty(source.shape[0], dtype=np.int64)
    order[target_sort] = source_sort
    return order


def _take_group(group: _Group, group_type: type[_Group], order: np.ndarray) -> _Group:
    return group_type(*(np.asarray(getattr(group, name))[order] for name in group._fields))


def _align_restart_topology(
    snapshot: EcosystemDynamicsFastTraceDirectory,
    topology: SubgridTopology,
) -> tuple[EcosystemDynamicsFastTraceDirectory, EcosystemFastTraceAlignment]:
    longitude_index = np.asarray(topology.grid.longitude_index, dtype=np.int64)
    latitude_index = np.asarray(topology.grid.latitude_index, dtype=np.int64)
    longitude_count = int(np.max(longitude_index)) + 1
    grid_domain = latitude_index * longitude_count + longitude_index + 1
    column_gridcell = np.asarray(topology.column.gridcell, dtype=np.int64)
    patch_gridcell = np.asarray(topology.patch.gridcell, dtype=np.int64)
    patch_column = np.asarray(topology.patch.column, dtype=np.int64)
    gridcell_order = _alignment_order(
        "gridcell",
        (snapshot.gridcell.domain_index,),
        (grid_domain,),
    )
    column_order = _alignment_order(
        "column",
        (
            snapshot.column.domain_gridcell,
            snapshot.column.column_type,
            snapshot.column.landunit_type,
        ),
        (
            grid_domain[column_gridcell],
            np.asarray(topology.column.column_type),
            np.asarray(topology.column.landunit_type),
        ),
    )
    patch_order = _alignment_order(
        "patch",
        (
            snapshot.patch.domain_gridcell,
            snapshot.patch.vegetation_type,
            snapshot.patch.column_domain_gridcell,
            snapshot.patch.column_type,
            snapshot.patch.landunit_type,
        ),
        (
            grid_domain[patch_gridcell],
            np.asarray(topology.patch.vegetation_type),
            grid_domain[column_gridcell[patch_column]],
            np.asarray(topology.patch.column_type),
            np.asarray(topology.patch.landunit_type),
        ),
    )
    aligned = snapshot._replace(
        gridcell=_take_group(
            snapshot.gridcell,
            EcosystemFastTraceGridcell,
            gridcell_order,
        ),
        column=_take_group(
            snapshot.column,
            EcosystemFastTraceColumn,
            column_order,
        ),
        patch=_take_group(
            snapshot.patch,
            EcosystemFastTracePatch,
            patch_order,
        ),
    )
    comparisons = (
        ("grid domain index", aligned.gridcell.domain_index, grid_domain),
        ("column domain gridcell", aligned.column.domain_gridcell, grid_domain[column_gridcell]),
        ("column type", aligned.column.column_type, topology.column.column_type),
        ("column landunit type", aligned.column.landunit_type, topology.column.landunit_type),
        ("column active mask", aligned.column.active, topology.column.active),
        ("patch domain gridcell", aligned.patch.domain_gridcell, grid_domain[patch_gridcell]),
        ("patch vegetation type", aligned.patch.vegetation_type, topology.patch.vegetation_type),
        (
            "patch column domain gridcell",
            aligned.patch.column_domain_gridcell,
            grid_domain[column_gridcell[patch_column]],
        ),
        ("patch column type", aligned.patch.column_type, topology.patch.column_type),
        ("patch landunit type", aligned.patch.landunit_type, topology.patch.landunit_type),
        ("patch active mask", aligned.patch.active, topology.patch.active),
    )
    for name, expected, actual in comparisons:
        _require_equal(name, np.asarray(expected), np.asarray(actual))
    weight_comparisons = (
        (
            "column gridcell weight",
            aligned.column.weight_on_gridcell,
            topology.column.weight_on_gridcell,
        ),
        (
            "patch gridcell weight",
            aligned.patch.weight_on_gridcell,
            topology.patch.weight_on_gridcell,
        ),
        (
            "patch column weight",
            aligned.patch.weight_on_column,
            topology.patch.weight_on_column,
        ),
    )
    for name, expected, actual in weight_comparisons:
        _require_roundoff_close(name, np.asarray(expected), np.asarray(actual))
    return aligned, EcosystemFastTraceAlignment(
        gridcell_source_index=gridcell_order,
        column_source_index=column_order,
        patch_source_index=patch_order,
    )


def load_ecosystem_dynamics_fast_trace_native_data(
    case: CimeCase,
    trace_directory: str | Path,
    restart_path: str | Path | None = None,
    *,
    expected_rank_count: int | None = None,
) -> EcosystemDynamicsFastTraceNativeData:

    snapshot = load_ecosystem_dynamics_fast_trace_directory(
        trace_directory,
        expected_rank_count=expected_rank_count,
    )
    restart = (
        configured_land_restart(case)
        if restart_path is None
        else Path(restart_path).expanduser().resolve()
    )
    if not restart.is_file():
        raise FileNotFoundError(f"CLM restart does not exist: {restart}")
    topology = load_subgrid_topology(restart)
    aligned, alignment = _align_restart_topology(snapshot, topology)
    return EcosystemDynamicsFastTraceNativeData(
        snapshot=aligned,
        alignment=alignment,
        topology=topology,
        restart_path=restart,
    )


__all__ = [
    "EcosystemDynamicsFastTrace",
    "EcosystemDynamicsFastTraceDirectory",
    "EcosystemDynamicsFastTraceNativeData",
    "EcosystemFastTraceBounds",
    "EcosystemFastTraceColumn",
    "EcosystemFastTraceGridcell",
    "EcosystemFastTraceAlignment",
    "EcosystemFastTracePatch",
    "load_ecosystem_dynamics_fast_trace",
    "load_ecosystem_dynamics_fast_trace_directory",
    "load_ecosystem_dynamics_fast_trace_native_data",
]
