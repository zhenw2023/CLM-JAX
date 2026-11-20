
from __future__ import annotations

import ctypes
import gc
import json
import os
import resource
import sys
import weakref
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache
from time import perf_counter
from typing import Any

import jax
import numpy as np


@dataclass(frozen=True)
class RuntimeInfo:
    backend: str
    devices: tuple[str, ...]
    x64_enabled: bool


@dataclass
class _MemoryReclamation:
    threshold_gib: float | None = None
    maximum_rss_gib: float | None = None
    requests: int = 0
    collections: int = 0
    trim_only_reclamations: int = 0
    host_pressure_requests: int = 0
    staging_cache_reclamations: int = 0
    last_staging_reclamation_seconds: float = float("-inf")
    wall_seconds: float = 0.0


_memory_reclamation = _MemoryReclamation()
_retained_host_bytes = ContextVar("clm_retained_host_bytes", default=0)
_validation_memory_floor_gib = ContextVar("clm_validation_memory_floor_gib", default=None)
_device_transfer_batch_size = ContextVar("clm_device_transfer_batch_size", default=None)


@contextmanager
def bounded_device_transfers(batch_size: int = 32):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("device transfer batch size must be a positive integer")
    previous = _device_transfer_batch_size.get()
    limit = batch_size if previous is None else min(previous, batch_size)
    token = _device_transfer_batch_size.set(limit)
    try:
        yield
    finally:
        _device_transfer_batch_size.reset(token)


def _transfer_arrays(transfer, values):
    size = _device_transfer_batch_size.get()
    if size is None:
        return transfer(values)
    result = []
    for start in range(0, len(values), size):
        transferred = transfer(tuple(values[start:start + size]))
        if not any(isinstance(v, jax.core.Tracer) for v in jax.tree.leaves(transferred)):
            jax.block_until_ready(transferred)
        result.extend(transferred)
    return tuple(result)


def array_values_to_host(values):
    return _transfer_arrays(jax.device_get, tuple(values))


@contextmanager
def retained_host_memory(nbytes: int):
    if nbytes < 0:
        raise ValueError("retained host bytes must be nonnegative")
    token = _retained_host_bytes.set(_retained_host_bytes.get() + nbytes)
    try:
        yield
    finally:
        _retained_host_bytes.reset(token)


def _collection_threshold(policy):
    threshold = policy.threshold_gib
    retained = _retained_host_bytes.get()
    if threshold is not None and retained:
        threshold += retained / float(1024**3)
        if policy.maximum_rss_gib is not None:
            threshold = min(threshold, 0.9 * policy.maximum_rss_gib)
    floor = _validation_memory_floor_gib.get()
    if threshold is not None and floor is not None:
        if policy.maximum_rss_gib is not None:
            floor = min(floor, 0.9 * policy.maximum_rss_gib)
        threshold = max(threshold, floor)
    return threshold


@contextmanager
def validation_memory_reclamation(maximum_rss_gib=None):
    if maximum_rss_gib is None:
        yield
        return
    if (isinstance(maximum_rss_gib, bool) or not np.isfinite(maximum_rss_gib)
            or maximum_rss_gib <= 0):
        raise ValueError("validation memory budget must be finite and positive")
    previous = _validation_memory_floor_gib.get()
    floor = 0.9 * float(maximum_rss_gib)
    if previous is not None:
        floor = max(previous, floor)
    token = _validation_memory_floor_gib.set(floor)
    try:
        yield
    finally:
        _validation_memory_floor_gib.reset(token)


_COMMITTED_ALIASES: dict[int, tuple[weakref.ReferenceType, jax.Device, jax.Array]] = {}


def _remember_committed_alias(source: jax.Array, target: jax.Array, device: jax.Device):
    if isinstance(source, jax.core.Tracer) or isinstance(target, jax.core.Tracer):
        return
    identity = id(source)

    def remove(reference):
        cached = _COMMITTED_ALIASES.get(identity)
        if cached is not None and cached[0] is reference:
            del _COMMITTED_ALIASES[identity]

    _COMMITTED_ALIASES[identity] = (weakref.ref(source, remove), device, target)


@cache
def _cpu_gather_workers() -> ThreadPoolExecutor:

    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1
    return ThreadPoolExecutor(max_workers=min(8, available), thread_name_prefix="clm-gather")


@contextmanager
def defer_full_collections_to_memory_policy():

    if jax.default_backend() != "cpu":
        yield
        return
    previous = gc.get_threshold()
    gc.set_threshold(previous[0], previous[1], 2**31 - 1)
    try:
        yield
    finally:
        gc.set_threshold(*previous)


def configure_memory_reclamation(maximum_rss_gib: float) -> None:

    if not np.isfinite(maximum_rss_gib) or maximum_rss_gib <= 0:
        raise ValueError("maximum resident memory must be finite and positive")
    global _memory_reclamation
    _memory_reclamation = _MemoryReclamation(
        threshold_gib=maximum_rss_gib * (2.0 / 3.0), maximum_rss_gib=maximum_rss_gib
    )


def memory_reclamation_statistics() -> dict[str, int | float | None]:

    policy = _memory_reclamation
    return {
        "threshold_gib": policy.threshold_gib,
        "effective_threshold_gib": _collection_threshold(policy),
        "retained_host_bytes": _retained_host_bytes.get(),
        "requests": policy.requests,
        "collections": policy.collections,
        "trim_only_reclamations": policy.trim_only_reclamations,
        "host_pressure_requests": policy.host_pressure_requests,
        "staging_cache_reclamations": policy.staging_cache_reclamations,
        "skipped": policy.requests - policy.collections - policy.trim_only_reclamations,
        "wall_seconds": policy.wall_seconds,
    }


def configure_runtime(*, enable_x64: bool = True) -> RuntimeInfo:

    jax.config.update("jax_enable_x64", enable_x64)
    return RuntimeInfo(
        backend=jax.default_backend(),
        devices=tuple(str(device) for device in jax.devices()),
        x64_enabled=bool(jax.config.read("jax_enable_x64")),
    )


@contextmanager
def gpu_dispatch_affinity(*, enabled: bool):

    supported = enabled and hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity")
    previous = os.sched_getaffinity(0) if supported else set()
    selected = max(previous) if previous else None
    pinned = False

    def pin():
        nonlocal pinned
        if selected is not None and len(previous) > 1 and not pinned:
            os.sched_setaffinity(0, {selected})
            pinned = True
        return selected

    try:
        yield pin
    finally:
        if pinned:
            os.sched_setaffinity(0, previous)


def _trim_host_heap() -> bool:

    try:
        trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return False
    trim.argtypes = (ctypes.c_size_t,)
    trim.restype = ctypes.c_int
    return bool(trim(0))


def available_host_memory_gib() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as stream:
            line = next(line for line in stream if line.startswith("MemAvailable:"))
        return int(line.split()[1]) / float(1024**2)
    except (OSError, StopIteration, ValueError):
        return None


def _host_memory_pressure(policy: _MemoryReclamation) -> bool:
    if policy.threshold_gib is None or jax.default_backend() != "cpu":
        return False
    available = available_host_memory_gib()
    return available is not None and available < max(8.0, policy.threshold_gib / 5.0)


def _reclaim_python_staging_caches() -> None:
    from jax._src import util

    util.clear_all_caches()


def _release_memory(*, force: bool = False) -> bool:
    policy = _memory_reclamation
    threshold = _collection_threshold(policy)
    policy.requests += 1
    host_pressure = not force and _host_memory_pressure(policy)
    if host_pressure:
        policy.host_pressure_requests += 1
    if (
        not force
        and threshold is not None
        and current_resident_set_size_gib() < threshold
        and not host_pressure
    ):
        return False
    started = perf_counter()
    try:
        if not force and threshold is not None and jax.default_backend() == "cpu":
            trimmed = _trim_host_heap()
            if current_resident_set_size_gib() < threshold and (
                not host_pressure or not _host_memory_pressure(policy)
            ):
                policy.trim_only_reclamations += 1
                return trimmed
        policy.collections += 1
        gc.collect()
        trimmed = _trim_host_heap() if jax.default_backend() == "cpu" else False
        if (
            host_pressure
            and perf_counter() - policy.last_staging_reclamation_seconds >= 30.0
            and _host_memory_pressure(policy)
        ):
            before = current_resident_set_size_gib()
            _reclaim_python_staging_caches()
            gc.collect()
            trimmed = _trim_host_heap() or trimmed
            policy.staging_cache_reclamations += 1
            policy.last_staging_reclamation_seconds = perf_counter()
            print(
                json.dumps(
                    {
                        "stage": "memory_staging_cache_reclamation",
                        "rss_before_gib": before,
                        "rss_after_gib": current_resident_set_size_gib(),
                        "available_gib": available_host_memory_gib(),
                        "wall_seconds": perf_counter() - started,
                    }
                ),
                flush=True,
            )
        return trimmed
    finally:
        policy.wall_seconds += perf_counter() - started


def release_completed_stage_memory() -> bool:

    return _release_memory()


def release_startup_executables() -> bool:

    return _release_memory(force=True)


def maximum_resident_set_size_gib() -> float:

    maximum = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    bytes_used = maximum if sys.platform == "darwin" else maximum * 1024.0
    return bytes_used / float(1024**3)


def current_resident_set_size_gib() -> float:

    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            line = next(value for value in status if value.startswith("VmRSS:"))
    except (OSError, StopIteration):
        return maximum_resident_set_size_gib()
    kibibytes = int(line.split()[1])
    return kibibytes / float(1024**2)


def current_memory_map_count() -> int | None:

    try:
        with open("/proc/self/maps", encoding="utf-8") as mappings:
            return sum(1 for _ in mappings)
    except OSError:
        return None


def gather_mapped_result(tree: Any, device: jax.Device) -> Any:

    if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(tree)):
        return jax.device_put(tree, device)
    return co_locate_array_leaves(tree, device)


def co_locate_array_leaves(tree: Any, device: jax.Device) -> Any:

    replacements: dict[int, Any] = {}
    direct_pending: list[jax.Array | np.ndarray] = []
    gathered_pending: list[jax.Array] = []
    target_devices = {device}

    leaves, structure = jax.tree.flatten(tree)
    for identity, leaf in dict(zip(map(id, leaves), leaves, strict=True)).items():
        if not isinstance(leaf, jax.Array):
            if isinstance(leaf, np.ndarray):
                direct_pending.append(leaf)
            continue
        if isinstance(leaf, jax.core.Tracer):
            continue
        if leaf.is_deleted():
            leaf.devices()  
        on_target = leaf.sharding.device_set == target_devices
        if leaf.committed and on_target:
            continue
        cached = _COMMITTED_ALIASES.get(identity)
        if (
            cached is not None and cached[0]() is leaf and cached[1] == device
            and not leaf.is_deleted() and not cached[2].is_deleted()
        ):
            replacements[identity] = cached[2]
            continue
        if on_target:
            direct_pending.append(leaf)
        else:
            gathered_pending.append(leaf)

    if not direct_pending and not gathered_pending and not replacements:
        return tree

    if direct_pending:
        device_values = _transfer_arrays(
            lambda values: jax.device_put(values, device), tuple(direct_pending)
        )
        replacements.update(
            (id(source), destination)
            for source, destination in zip(
                direct_pending,
                device_values,
                strict=True,
            )
        )
        for source, destination in zip(direct_pending, device_values, strict=True):
            if isinstance(source, jax.Array) and source is not destination:
                _remember_committed_alias(source, destination, device)

    if gathered_pending:
        if (
            device.platform == "cpu"
            and len(gathered_pending) > 1
            and all(
                source.platform == "cpu" for leaf in gathered_pending for source in leaf.devices()
            )
        ):
            host_values = tuple(_cpu_gather_workers().map(np.asarray, gathered_pending))
        else:
            host_values = array_values_to_host(gathered_pending)
        device_values = _transfer_arrays(
            lambda values: jax.device_put(values, device), tuple(host_values)
        )
        replacements.update(
            (id(source), destination)
            for source, destination in zip(
                gathered_pending,
                device_values,
                strict=True,
            )
        )

    return jax.tree.unflatten(
        structure, [replacements.get(id(leaf), leaf) for leaf in leaves],
    )


def all_scalar_checks(checks: Iterable[Any]) -> bool:

    return all(bool(np.asarray(jax.device_get(check))) for check in checks)


__all__ = [
    "RuntimeInfo",
    "all_scalar_checks",
    "array_values_to_host",
    "bounded_device_transfers",
    "co_locate_array_leaves",
    "configure_memory_reclamation",
    "configure_runtime",
    "current_memory_map_count",
    "current_resident_set_size_gib",
    "defer_full_collections_to_memory_policy",
    "maximum_resident_set_size_gib",
    "memory_reclamation_statistics",
    "release_completed_stage_memory",
    "release_startup_executables",
    "retained_host_memory",
]
