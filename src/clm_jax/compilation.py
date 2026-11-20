
from __future__ import annotations

import weakref
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache, lru_cache, partial, wraps
from inspect import signature
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import get_opaque_trace_state

with jax.core.eval_context():
    _EAGER_TRACE = get_opaque_trace_state()

_stage_reverse_ad = ContextVar("clm_stage_reverse_ad", default=False)
_inside_ad_stage = ContextVar("clm_inside_ad_stage", default=False)
_reused_stage_options = ContextVar("clm_reused_stage_options", default=None)
_native_ad_stage = ContextVar("clm_native_ad_stage", default=None)
_unmaterialized_alias_inputs = ContextVar("clm_unmaterialized_alias_inputs", default=frozenset())


@contextmanager
def reused_stage_programs(*, cache_size=2, offload_inputs=True, compiler_options=None,
                          host_storage="pinned_host"):
    if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 1:
        raise ValueError("stage replay cache size must be a positive integer")
    if host_storage not in {"pinned_host", "numpy"}:
        raise ValueError("stage host storage must be 'pinned_host' or 'numpy'")
    from clm_jax.physics.plant_hydraulics_gpu import retain_native_tile_programs

    options = None if compiler_options is None else tuple(sorted(compiler_options.items()))
    token = _reused_stage_options.set((cache_size, bool(offload_inputs), options, host_storage))
    try:
        with retain_native_tile_programs():
            yield
    finally:
        _reused_stage_options.reset(token)


@contextmanager
def staged_reverse_ad(*, enabled=True):
    token = _stage_reverse_ad.set(enabled)
    try:
        yield
    finally:
        _stage_reverse_ad.reset(token)


@cache
def _compiled_jit(
    function: Callable[..., Any],
    static_argnames: tuple[str, ...],
    static_argnums: tuple[int, ...],
    inline: bool,
    cpu_parallel_codegen_split_count: int | None,
) -> Callable[..., Any]:
    compiler_options = (
        None
        if cpu_parallel_codegen_split_count is None
        else {"xla_cpu_parallel_codegen_split_count": (cpu_parallel_codegen_split_count)}
    )
    return jax.jit(
        function,
        static_argnames=static_argnames or None,
        static_argnums=static_argnums or None,
        inline=inline,
        compiler_options=compiler_options,
    )


def process_lifetime_jit(
    function: Callable[..., Any],
    *,
    enabled: bool = True,
    static_argnames: tuple[str, ...] = (),
    static_argnums: tuple[int, ...] = (),
    inline: bool = False,
    cpu_parallel_codegen_split_count: int | None = None,
) -> Callable[..., Any]:

    if not enabled:
        return function
    if cpu_parallel_codegen_split_count is not None:
        if (
            isinstance(cpu_parallel_codegen_split_count, bool)
            or cpu_parallel_codegen_split_count < 1
        ):
            raise ValueError("CPU parallel codegen split count must be positive")
    if _stage_reverse_ad.get() and get_opaque_trace_state() != _EAGER_TRACE:
        if _inside_ad_stage.get():
            return function
        return _checkpointed_stage(function, tuple(static_argnames), tuple(static_argnums), inline)
    return _compiled_jit(
        function,
        tuple(static_argnames),
        tuple(static_argnums),
        bool(inline),
        cpu_parallel_codegen_split_count,
    )


@cache
def _checkpointed_stage(function, static_argnames, static_argnums, inline):
    names = tuple(signature(function).parameters)

    @wraps(function)
    def execute(*args, **kwargs):
        static_positions = tuple(
            (index, value)
            for index, value in enumerate(args)
            if index in static_argnums
            or index - len(args) in static_argnums
            or (index < len(names) and names[index] in static_argnames)
        )
        fixed = dict(static_positions)
        dynamic_args = tuple(value for index, value in enumerate(args) if index not in fixed)
        static_keywords = tuple(
            sorted((name, value) for name, value in kwargs.items() if name in static_argnames)
        )
        dynamic_keywords = {
            name: value for name, value in kwargs.items() if name not in static_argnames
        }
        return _compiled_checkpointed_stage(
            function, len(args), static_positions, static_keywords, inline
        )(dynamic_args, dynamic_keywords)

    return execute


@cache
def _compiled_checkpointed_stage(function, count, static_positions, static_keywords, inline):
    def body(arguments, keywords):
        fixed = dict(static_positions)
        arguments = iter(arguments)
        positional = tuple(
            fixed[index] if index in fixed else next(arguments) for index in range(count)
        )
        token = _inside_ad_stage.set(True)
        try:
            return function(*positional, **keywords, **dict(static_keywords))
        finally:
            _inside_ad_stage.reset(token)

    def packed(arguments, keywords):
        leaves, tree = jax.tree.flatten(body(arguments, keywords))
        values, slots = _unique_leaves(leaves, (arguments, keywords))
        return _SharedResult(values, (tree, slots))

    packed.__name__ = f"{getattr(function, '__name__', 'stage')}_ad_checkpoint"
    compiled = jax.jit(jax.checkpoint(packed, prevent_cse=False), inline=inline)

    @cache
    def replayed(options):
        from clm_jax.adjoint_replay import recomputed_stage_replay

        return recomputed_stage_replay(
            lambda point: compiled(*point),
            cache_size=options[0],
            offload_inputs=options[1],
            host_storage=options[3],
            stage_name=packed.__name__,
            compiler_options=None if options[2] is None else dict(options[2]),
        )

    def execute(arguments, keywords):
        options = _reused_stage_options.get()
        result = (
            compiled(arguments, keywords)
            if options is None
            else replayed(options)((arguments, keywords))
        )
        return result.restore((arguments, keywords))

    return execute


def _unique_leaves(leaves, arguments):
    identities = {id(value): ~index for index, value in enumerate(jax.tree.leaves(arguments))}
    unique, slots = [], []
    for leaf in leaves:
        identity = id(leaf)
        if identity not in identities:
            identities[identity] = len(unique)
            unique.append(leaf)
        slots.append(identities[identity])
    return tuple(unique), tuple(slots)


@jax.custom_jvp
def _materialize_phase_values(values):
    return jax.lax.optimization_barrier(values)


@partial(_materialize_phase_values.defjvp,symbolic_zeros=True)
def _materialize_phase_values_jvp(primals, tangents):
    return _materialize_phase_values(primals[0]), tangents[0]


def materialize_outputs(function):

    @wraps(function)
    def call(*args, **kwargs):
        result = function(*args, **kwargs)
        leaves, tree = jax.tree.flatten(result)
        existing = {id(value) for value in jax.tree.leaves((args, kwargs))}
        computed = {id(value): value for value in leaves if id(value) not in existing}
        values = _materialize_phase_values(tuple(computed.values()))
        translated = dict(zip(computed, values, strict=True))
        return tree.unflatten([translated.get(id(value), value) for value in leaves])

    return call


@jax.tree_util.register_pytree_node_class
class _SharedResult:

    def __init__(self, values, metadata):
        self.values = values
        self.metadata = metadata

    def tree_flatten(self):
        return self.values, self.metadata

    @classmethod
    def tree_unflatten(cls, metadata, values):
        return cls(values, metadata)

    def restore(self, arguments):
        tree, slots = self.metadata
        input_slots = {index for index in slots if index < 0}
        inputs = jax.tree.leaves(arguments) if input_slots else ()
        forwarded = {}
        canonical_dtypes = {}
        for index in input_slots:
            value = inputs[~index]
            canonical_array = False
            if isinstance(value, jax.Array):
                dtype = value.dtype
                if dtype not in canonical_dtypes:
                    canonical_dtypes[dtype] = (
                        jax.dtypes.canonicalize_dtype(dtype, allow_extended_dtype=True) == dtype
                    )
                canonical_array = canonical_dtypes[dtype]
            if not canonical_array and not (
                isinstance(value, np.ndarray)
                and id(value) in _unmaterialized_alias_inputs.get()
            ):
                value = jnp.asarray(value)
            forwarded[index] = value
        return jax.tree.unflatten(
            tree,
            [forwarded[index] if index < 0 else self.values[index] for index in slots],
        )


@jax.tree_util.register_pytree_node_class
class _SharedArguments:

    def __init__(self, values, metadata):
        self.values, self.metadata = values, metadata

    def tree_flatten(self):
        return self.values, self.metadata

    @classmethod
    def tree_unflatten(cls, metadata, values):
        return cls(values, metadata)

    def restore(self):
        tree, slots = self.metadata
        return tree.unflatten([self.values[index] for index in slots])


@lru_cache(maxsize=128)
def _ordinary_unique_stage(compiled, share_inputs):
    def ordinary(*arguments):
        if share_inputs:
            leaves, tree = jax.tree.flatten(arguments)
            values, slots = _unique_leaves(leaves, ())
            result = compiled(_SharedArguments(values, (tree, slots)))
        else:
            result = compiled(*arguments)
        return result.restore(arguments)
    return ordinary


def jit_unique(
    function,
    *,
    compiler_options=None,
    inline=False,
    forward_inputs=False,
    share_inputs=False,
    reverse_replay=True,
):

    @wraps(function)
    def call(*arguments):
        leaves, tree = jax.tree.flatten(function(*arguments))
        values, slots = _unique_leaves(leaves, arguments if forward_inputs else ())
        return _SharedResult(values, (tree, slots))

    def shared_call(packed):
        return call(*packed.restore())

    compiled = jax.jit(
        shared_call if share_inputs else call,
        compiler_options=compiler_options,
        inline=inline,
    )

    @cache
    def differentiated_stage():
        return jax.jit(jax.checkpoint(call, prevent_cse=False), inline=inline)

    @cache
    def replayed_stage(options):
        from clm_jax.adjoint_replay import recomputed_stage_replay

        def body(arguments):
            token = _inside_ad_stage.set(True)
            try:
                return differentiated_stage()(*arguments)
            finally:
                _inside_ad_stage.reset(token)

        return recomputed_stage_replay(
            body,
            primal=lambda arguments: differentiated_stage()(*arguments),
            cache_size=options[0],
            offload_inputs=options[1],
            host_storage=options[3],
            stage_name=f"{getattr(function, '__name__', type(function).__name__)}_ad_checkpoint",
            compiler_options=None if options[2] is None else dict(options[2]),
        )

    @wraps(function)
    def execute(*arguments):
        if get_opaque_trace_state() != _EAGER_TRACE:
            if _stage_reverse_ad.get() and not _inside_ad_stage.get():
                native_policy = _native_ad_stage.get()
                if (native_policy is not None and reverse_replay
                        and _reused_stage_options.get() is not None
                        and native_policy[1]()):
                    return native_policy[0](function, _ordinary_unique_stage(compiled, share_inputs))(*arguments)
                token = _inside_ad_stage.set(True)
                try:
                    options = _reused_stage_options.get()
                    result = (
                        differentiated_stage()(*arguments)
                        if options is None or not reverse_replay
                        else replayed_stage(options)(arguments)
                    )
                finally:
                    _inside_ad_stage.reset(token)
                return result.restore(arguments)
            return function(*arguments)
        if share_inputs:
            leaves, tree = jax.tree.flatten(arguments)
            values, slots = _unique_leaves(leaves, ())
            result = compiled(_SharedArguments(values, (tree, slots)))
        else:
            result = compiled(*arguments)
        return result.restore(arguments)

    return execute


@jax.tree_util.register_pytree_node_class
class _DiscreteParameter:

    def __init__(self, value, dtype):
        self.value = value
        self.dtype = dtype

    def tree_flatten(self):
        return (), (self.value, self.dtype)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(*metadata)


_DYNAMIC_PARAMETER_FIELDS = frozenset(
    {
        "model_step",
        "month",
        "day_of_month",
        "seconds_of_day",
        "is_end_day",
    }
)

_CASCADE_TOPOLOGY_FIELDS = frozenset({"cascade_donor_pool", "cascade_receiver_pool"})
_PARAMETER_CASCADE_TOPOLOGY: bool | None = None

_DISCRETE_DEVICE_VALUES: dict[int, tuple[weakref.ReferenceType, _DiscreteParameter]] = {}


def _discrete_value(value):
    identity = id(value)
    if isinstance(value, jax.Array):
        cached = _DISCRETE_DEVICE_VALUES.get(identity)
        if cached is not None and cached[0]() is value and not value.is_deleted():
            return cached[1]
    array = np.asarray(value)
    result = _DiscreteParameter(
        array.item() if array.ndim == 0 else tuple(array.tolist()),
        array.dtype.str,
    )
    if isinstance(value, jax.Array):

        def remove(reference):
            cached = _DISCRETE_DEVICE_VALUES.get(identity)
            if cached is not None and cached[0] is reference:
                del _DISCRETE_DEVICE_VALUES[identity]

        _DISCRETE_DEVICE_VALUES[identity] = (weakref.ref(value, remove), result)
    return result


def _parameter_record(value):
    return isinstance(value, tuple) and (
        type(value).__name__.endswith("Parameters")
        or type(value).__name__ == "CnStateUpdate1Inputs"
    )


@lru_cache(maxsize=128)
def _discrete_parameter_slots(tree):

    scalar_slots, cascade_slots = [], []
    specialize_cascade = (
        jax.default_backend() == "gpu"
        if _PARAMETER_CASCADE_TOPOLOGY is None
        else _PARAMETER_CASCADE_TOPOLOGY
    )

    def collect(record):
        if not _parameter_record(record):
            return record

        def slot(path, index):
            if specialize_cascade and (getattr(path[-1], "name", None) in _CASCADE_TOPOLOGY_FIELDS):
                cascade_slots.append(index)
            elif type(record).__name__ == "CnStateUpdate1Inputs":
                if getattr(path[-1], "name", None) in _CASCADE_TOPOLOGY_FIELDS:
                    cascade_slots.append(index)
            elif not any(getattr(key, "name", None) in _DYNAMIC_PARAMETER_FIELDS for key in path):
                scalar_slots.append(index)
            return index

        return jax.tree.map_with_path(slot, record)

    placeholders = tree.unflatten(range(tree.num_leaves))
    jax.tree.map(collect, placeholders, is_leaf=_parameter_record)
    return tuple(scalar_slots), tuple(cascade_slots)


@lru_cache(maxsize=512)
def _resident_parameter(dtype, storage, weak_type, enable_x64, device):
    value = np.frombuffer(storage, dtype=np.dtype(dtype))[0]
    if weak_type:
        value = value.item()
    with jax.core.eval_context():
        return jax.device_put(value, device)


def _discrete_parameters(arguments):
    leaves, tree = jax.tree.flatten(arguments)
    scalar_slots, cascade_slots = _discrete_parameter_slots(tree)
    if not scalar_slots and not cascade_slots:
        return arguments
    parameter_device = jax.devices()[0] if jax.default_backend() == "gpu" else None
    for index in scalar_slots:
        value = leaves[index]
        if np.shape(value) != ():
            continue
        dtype = getattr(value, "dtype", None)
        if dtype is None:
            if not isinstance(value, (bool, int, float, complex)):
                continue
            dtype = np.asarray(value).dtype
        dtype = np.dtype(dtype)
        if dtype.kind in "biu":
            leaves[index] = _discrete_value(value)
        elif (
            dtype.kind in "fc" and parameter_device is not None and not isinstance(value, jax.Array)
        ):
            array = np.asarray(value)
            leaves[index] = _resident_parameter(
                array.dtype.str,
                array.tobytes(),
                type(value) in (float, complex),
                jax.config.x64_enabled,
                parameter_device,
            )
    for index in cascade_slots:
        value = leaves[index]
        if not hasattr(value, "dtype"):
            value = np.asarray(value)
        if np.ndim(value) == 1 and np.dtype(value.dtype).kind in "iu":
            leaves[index] = _discrete_value(value)
    return tree.unflatten(leaves)


def jit_discrete_parameters(function, **kwargs):

    @wraps(function)
    def call(*arguments):
        restored = jax.tree.map(
            lambda value: (
                np.asarray(value.value, dtype=value.dtype)
                if isinstance(value, _DiscreteParameter)
                else value
            ),
            arguments,
            is_leaf=lambda value: isinstance(value, _DiscreteParameter),
        )
        return function(*restored)

    compiled = jit_unique(call, **kwargs)

    @wraps(function)
    def execute(*arguments):
        if get_opaque_trace_state() != _EAGER_TRACE:
            if _stage_reverse_ad.get() and not _inside_ad_stage.get():
                return compiled(*_discrete_parameters(arguments))
            return function(*arguments)
        return compiled(*_discrete_parameters(arguments))

    return execute


def _symbolic_zero_jvp(function, primals, tangents):
    from jax._src.interpreters import ad
    from jax._src.tree_util import FlatTree
    from jax.custom_derivatives import SymbolicZero

    leaves, structure = jax.tree.flatten(primals)
    dots, tangent_structure = jax.tree.flatten(
        tangents, is_leaf=lambda value: isinstance(value, SymbolicZero))
    if structure != tangent_structure:
        raise ValueError('symbolic tangents must retain the complete input structure')
    values=FlatTree.flatten(primals)
    directions=values.update(ad.Zero(v.aval) if isinstance(v,SymbolicZero) else v for v in dots)
    outputs,directions=ad.jvp(function,values,directions,instantiate=False)
    return outputs.unflatten(),directions.map(
        lambda v:SymbolicZero(v.aval) if isinstance(v,ad.Zero) else v).unflatten()


__all__ = [
    "process_lifetime_jit",
    "jit_unique",
    "jit_discrete_parameters",
    "materialize_outputs",
    "staged_reverse_ad",
]
