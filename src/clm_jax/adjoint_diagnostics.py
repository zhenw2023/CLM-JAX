
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, wraps

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import pjit
from jax._src.ad_util import Zero
from jax._src.interpreters import ad

from clm_jax.runtime import array_values_to_host, bounded_device_transfers

_observer = ContextVar("clm_adjoint_diagnostic_observer", default=None)


def _layout_preserving_check(function):
    default = jax.jit(function)

    @lru_cache(maxsize=128)
    def for_format(format):
        return jax.jit(function, in_shardings=format)

    @wraps(function)
    def execute(value):
        if (isinstance(value, jax.Array) and not isinstance(value, jax.core.Tracer)
                and isinstance(value.sharding, jax.sharding.SingleDeviceSharding)):
            return for_format(value.format)(value)
        return default(value)

    execute.compiled_for_format = for_format
    return execute


@_layout_preserving_check
def _floating_cotangent_leaf_flags(value):
    return jnp.stack((jnp.any(value != 0), jnp.all(jnp.isfinite(value))))


def _cotangent_leaf_flags(value):
    if jax.default_backend() == 'cpu' and np.dtype(value.dtype).kind in 'fc':
        values = np.asarray(value)
        size = values.dtype.itemsize//(2 if values.dtype.kind == 'c' else 1)
        unsigned = {2:np.uint16, 4:np.uint32, 8:np.uint64}[size]
        bits = values.reshape(-1).view(unsigned)
        magnitude = bits & unsigned((1 << (8*size-1))-1)
        exponent = unsigned({2:0x7c00, 4:0x7f800000, 8:0x7ff0000000000000}[size])
        return np.array([np.any(magnitude != 0), np.all((bits & exponent) != exponent)])
    return _floating_cotangent_leaf_flags(value)


def _host_flags(function, values):
    result = []
    with bounded_device_transfers():
        for batch in _flag_batches(values):
            flags = tuple(function(value) for value in batch)
            result.extend(array_values_to_host(flags))
    return tuple(result)


def _flag_batches(values, *, maximum_arrays=32, maximum_bytes=64 << 20):
    batch, size = [], 0
    for value in values:
        nbytes = int(value.nbytes)
        if batch and (len(batch) >= maximum_arrays or size + nbytes > maximum_bytes):
            yield tuple(batch)
            batch, size = [], 0
        batch.append(value)
        size += nbytes
    if batch:
        yield tuple(batch)


def exact_zero_cotangents(cotangents):
    leaves, structure = jax.tree.flatten(cotangents)
    unique, indices, slots = [], {}, []
    for value in leaves:
        identity = id(value)
        if identity not in indices:
            indices[identity] = len(unique)
            unique.append(value)
        slots.append(indices[identity])
    selected = [i for i, value in enumerate(unique) if value.dtype != jax.dtypes.float0]
    flags = (
        np.asarray(_host_flags(_cotangent_leaf_flags, [unique[i] for i in selected]))
        if selected else ()
    )
    invalid = {index for index, flag in zip(selected, flags, strict=True) if not flag[1]}
    if invalid:
        paths = jax.tree.flatten_with_path(cotangents)[0]
        affected = [jax.tree_util.keystr(path)
                    for (path, _), index in zip(paths, slots, strict=True)
                    if index in invalid]
        raise FloatingPointError("non-finite state adjoint in " + ", ".join(affected[:8]))
    nonzero = {index for index, flag in zip(selected, flags, strict=True) if flag[0]}
    converted = [value if i in nonzero else Zero(jax.typeof(value))
                 for i, value in enumerate(unique)]
    return structure.unflatten([converted[i] for i in slots])


@_layout_preserving_check
def _all_finite(value):
    return jnp.all(jnp.isfinite(value))


def _native_failure_event(phase, parameters, candidates, error):
    names = parameters["jaxpr"].jaxpr.debug_info.arg_names
    fields = []
    for index, value in sorted(candidates, key=lambda item: item[1].nbytes, reverse=True)[:3]:
        try:
            layout = repr(value.format)
        except Exception:
            layout = "unavailable after device failure"
        fields.append(dict(input=names[index] if names and index < len(names) else str(index),
                           shape=list(value.shape), dtype=str(value.dtype),
                           bytes=int(value.nbytes), format=layout))
    return dict(phase=phase, stage=parameters["name"], largest_adjoint_fields=fields,
                error=repr(error))


@contextmanager
def diagnose_stage_pullbacks(observer, *, report_stages=False):
    if _observer.get() is not None:
        yield
        return
    original = ad.fancy_transposes[pjit.jit_p]

    def checked(cotangents, *arguments, **parameters):
        notify = _observer.get()
        if notify is None:
            return original(cotangents, *arguments, **parameters)
        leaves = jax.tree.leaves(cotangents)
        if not any(isinstance(value, jax.core.Tracer) for value in leaves):
            cotangents = exact_zero_cotangents(cotangents)
            if leaves and not jax.tree.leaves(cotangents):
                notify(dict(phase="zero_stage_adjoint", stage=parameters["name"]))
            elif report_stages and leaves:
                notify(dict(phase="apply_stage_adjoint", stage=parameters["name"]))
        result = original(cotangents, *arguments, **parameters)
        candidates = [(i, argument.val) for i, argument in enumerate(arguments)
                      if isinstance(argument, ad.ValAccum)
                      and not isinstance(argument.val, (Zero, jax.core.Tracer))
                      and np.dtype(argument.val.dtype).kind in "fc"]
        values = tuple(value for _, value in candidates)
        try:
            jax.block_until_ready(values)
        except Exception as error:
            notify(_native_failure_event("native_stage_adjoint_execution_failed",
                                         parameters, candidates, error))
            raise
        try:
            flags = _host_flags(_all_finite, values)
        except Exception as error:
            notify(_native_failure_event("native_stage_adjoint_check_failed",
                                         parameters, candidates, error))
            raise
        invalid = [(i, value) for (i, value), finite in zip(candidates, flags, strict=True)
                   if not bool(finite)]
        if invalid:
            names = parameters["jaxpr"].jaxpr.debug_info.arg_names
            fields = []
            for i, value in invalid[:8]:
                array = np.asarray(value)
                indices = np.argwhere(~np.isfinite(array))
                fields.append(dict(input=names[i] if names and i < len(names) else str(i),
                                   shape=list(array.shape),
                                   nonfinite_count=int(np.count_nonzero(~np.isfinite(array))),
                                   sample_indices=indices[:4].tolist()))
            stage = parameters["name"]
            notify(dict(phase="invalid_stage_adjoint", stage=stage, fields=fields,
                        source=parameters["jaxpr"].jaxpr.debug_info.func_src_info))
            raise FloatingPointError(f"non-finite adjoint from {stage}: {fields}")
        return result

    token = _observer.set(observer)
    ad.fancy_transposes[pjit.jit_p] = checked
    try:
        yield
    finally:
        ad.fancy_transposes[pjit.jit_p] = original
        _observer.reset(token)
