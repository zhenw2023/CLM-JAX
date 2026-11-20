
from __future__ import annotations

import warnings
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from clm_jax.io.datm_stream import DatmStream

_NATIVE_VARIABLE_CACHE: dict[tuple[object, ...], jnp.ndarray] = {}
_native_cache_budget = ContextVar("clm_native_variable_cache_budget", default=None)


class _NativeVariableCacheBudget:
    def __init__(self, maximum_bytes, observer):
        self.maximum_bytes = maximum_bytes
        self.values = OrderedDict()
        self.evictions = self.evicted_bytes = 0
        self.observer, self.last_reported_evicted_bytes = observer, 0

    def trim(self, recently_read):
        for key in recently_read:
            self.values.move_to_end(key)
        retained = sum(value.nbytes for value in self.values.values())
        while retained > self.maximum_bytes:
            _, value = self.values.popitem(last=False)
            retained -= value.nbytes
            self.evictions += 1
            self.evicted_bytes += value.nbytes
        if (self.observer is not None
                and self.evicted_bytes - self.last_reported_evicted_bytes >= self.maximum_bytes):
            self.last_reported_evicted_bytes = self.evicted_bytes
            self.observer()


@contextmanager
def native_variable_cache_scope(*, maximum_bytes, observer=None):
    if (isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int)
            or maximum_bytes < 1):
        raise ValueError("native variable cache budget must be a positive integer")
    budget = _NativeVariableCacheBudget(maximum_bytes, observer)
    token = _native_cache_budget.set(budget)
    try:
        yield
    finally:
        _native_cache_budget.reset(token)
        budget.values.clear()


def native_variable_cache_statistics():
    budget = _native_cache_budget.get()
    if budget is None:
        return None
    return dict(
        native_cache_entries=len(budget.values),
        native_cache_bytes=sum(value.nbytes for value in budget.values.values()),
        native_cache_evictions=budget.evictions,
        native_cache_evicted_bytes=budget.evicted_bytes,
    )


def _native_variable_cache_key(
    path: Path,
    name: str,
    dtype: np.dtype | type[np.generic] | None,
    masked_integer_fill_value: int | None,
    masked_floating_fill_value: float | None,
) -> tuple[object, ...]:
    stat = path.stat()
    normalized_dtype = None if dtype is None else np.dtype(dtype).str
    normalized_float_fill = (
        None
        if masked_floating_fill_value is None
        else np.asarray(masked_floating_fill_value, dtype=np.float64).tobytes()
    )
    return (
        path,
        stat.st_mtime_ns,
        stat.st_size,
        name,
        normalized_dtype,
        masked_integer_fill_value,
        normalized_float_fill,
    )


def _decode_native_variable(
    variable,
    selection,
    *,
    path: Path,
    name: str,
    dtype: np.dtype | type[np.generic] | None,
    masked_integer_fill_value: int | None,
    masked_floating_fill_value: float | None,
) -> jnp.ndarray:
    value = variable[selection]
    if np.ma.isMaskedArray(value):
        mask = np.ma.getmaskarray(value)
        if np.any(mask):
            if np.issubdtype(value.dtype, np.floating):
                floating_fill = (
                    np.nan
                    if masked_floating_fill_value is None
                    else masked_floating_fill_value
                )
                value = value.filled(floating_fill)
            elif masked_integer_fill_value is None:
                raise ValueError(
                    f"integer variable {name!r} in {path} contains masked values"
                )
            else:
                value = value.filled(masked_integer_fill_value)
        else:
            value = value.data
    array = np.asarray(value, dtype=dtype)
    with jax.core.eval_context():
        return jnp.asarray(np.ascontiguousarray(array))


def _netcdf4_dataset_type():

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="numpy.ndarray size changed.*",
            category=RuntimeWarning,
        )
        from netCDF4 import Dataset  

    return Dataset


def load_variables(
    path: str | Path,
    variables: tuple[str, ...] | list[str],
    *,
    indexers: Mapping[str, int | slice] | None = None,
    dtype: np.dtype | type[np.floating] | None = None,
) -> dict[str, jnp.ndarray]:

    with xr.open_dataset(Path(path), decode_times=False, mask_and_scale=True) as dataset:
        selected = dataset if indexers is None else dataset.isel(indexers)
        with jax.core.eval_context():
            return {
                name: jnp.asarray(np.ascontiguousarray(selected[name].load().values, dtype=dtype))
                for name in variables
            }


def load_native_variables(
    path: str | Path,
    variables: tuple[str, ...] | list[str],
    *,
    indexers: Mapping[str, int | slice] | None = None,
    dtype: np.dtype | type[np.generic] | None = None,
    masked_integer_fill_value: int | None = None,
    masked_floating_fill_value: float | None = None,
) -> dict[str, jnp.ndarray]:

    Dataset = _netcdf4_dataset_type()
    source = Path(path).expanduser().resolve()
    result: dict[str, jnp.ndarray] = {}

    if indexers is not None:
        with Dataset(source, mode="r") as dataset:
            for name in variables:
                variable = dataset.variables[name]
                selection = tuple(
                    indexers.get(dimension, slice(None))
                    for dimension in variable.dimensions
                )
                result[name] = _decode_native_variable(
                    variable,
                    selection,
                    path=source,
                    name=name,
                    dtype=dtype,
                    masked_integer_fill_value=masked_integer_fill_value,
                    masked_floating_fill_value=masked_floating_fill_value,
                )
        return result

    budget = _native_cache_budget.get()
    cache = _NATIVE_VARIABLE_CACHE if budget is None else budget.values
    keys = {
        name: _native_variable_cache_key(
            source,
            name,
            dtype,
            masked_integer_fill_value,
            masked_floating_fill_value,
        )
        for name in variables
    }
    missing = [name for name in variables if keys[name] not in cache]
    if missing:
        with Dataset(source, mode="r") as dataset:
            for name in missing:
                cache[keys[name]] = _decode_native_variable(
                    dataset.variables[name],
                    slice(None),
                    path=source,
                    name=name,
                    dtype=dtype,
                    masked_integer_fill_value=masked_integer_fill_value,
                    masked_floating_fill_value=masked_floating_fill_value,
                )
    result = {name: cache[keys[name]] for name in variables}
    if budget is not None:
        budget.trim(keys.values())
    return result


def load_dimension_sizes(path: str | Path) -> dict[str, int]:

    Dataset = _netcdf4_dataset_type()
    with Dataset(Path(path), mode="r") as dataset:
        return {name: len(dimension) for name, dimension in dataset.dimensions.items()}


@lru_cache(maxsize=64)
def _datm_file_record_ends(paths: tuple[Path, ...]) -> tuple[int, ...]:

    Dataset = _netcdf4_dataset_type()
    ends = []
    count = 0
    for path in paths:
        with Dataset(path, mode="r") as dataset:
            count += len(dataset.dimensions["time"])
        ends.append(count)
    return tuple(ends)


def load_datm_time_slice(stream: DatmStream, time_index: int) -> dict[str, jnp.ndarray]:

    if not stream.field_files:
        raise ValueError("DATM stream has no field files")
    path = stream.field_files[0]
    if len(stream.field_files) > 1:
        ends = _datm_file_record_ends(stream.field_files)
        if time_index < 0 or time_index >= ends[-1]:
            raise IndexError(f"DATM record {time_index} outside 0..{ends[-1] - 1}")
        file_index = bisect_right(ends, time_index)
        path = stream.field_files[file_index]
        time_index -= ends[file_index - 1] if file_index else 0
    Dataset = _netcdf4_dataset_type()
    by_file_name = {}
    with Dataset(path, mode="r") as dataset:
        for mapping in stream.field_variables:
            variable = dataset.variables[mapping.file_variable]
            selection = tuple(
                time_index if dimension == "time" else slice(None)
                for dimension in variable.dimensions
            )
            value = variable[selection]
            if np.ma.isMaskedArray(value):
                value = value.astype(np.float64).filled(np.nan)
            by_file_name[mapping.file_variable] = jnp.asarray(
                np.ascontiguousarray(value, dtype=np.float64)
            )
    return {
        mapping.model_variable: by_file_name[mapping.file_variable]
        for mapping in stream.field_variables
    }
