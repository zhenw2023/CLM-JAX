
import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, wraps
from weakref import ref

import jax
import numpy as np
from jax._src import core
from jax._src.ad_util import Zero
from jax._src.interpreters import ad, partial_eval
from jax._src.lax.lax import _convert_element_type
from jax.custom_derivatives import CustomVJPPrimal, SymbolicZero

from clm_jax.adjoint_diagnostics import _observer, exact_zero_cotangents
from clm_jax.runtime import _transfer_arrays, array_values_to_host, bounded_device_transfers

_host_copies = ContextVar("clm_stage_input_host_copies", default=None)


@contextmanager
def stage_input_offload_scope():
    if _host_copies.get() is not None:
        yield
        return
    copies = {}
    token = _host_copies.set(copies)
    try:
        yield
    finally:
        _host_copies.reset(token)
        copies.clear()


@jax.tree_util.register_pytree_node_class
class _SavedStage:
    def __init__(self, point, mask, offloaded=False, host_weak_types=None):
        self.point, self.mask, self.offloaded = point, mask, offloaded
        self.host_weak_types = host_weak_types

    def tree_flatten(self):
        return (self.point,), (self.mask, self.offloaded, self.host_weak_types)

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(children[0], *metadata)


def _zero(value):
    return isinstance(value, (Zero, SymbolicZero))


def _place_arrays(point, memory_kind):
    if (memory_kind == "pinned_host"
            and os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "default").lower() == "vmm"):
        raise RuntimeError("VMM stage offload requires host_storage='numpy', not pinned_host")
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0], memory_kind=memory_kind)
    leaves, tree = jax.tree.flatten(point)
    unique, replacements = {}, {}
    copies = _host_copies.get() if memory_kind == "pinned_host" else None
    for value in leaves:
        if isinstance(value, jax.Array):
            if value.sharding.is_equivalent_to(sharding, value.ndim):
                replacements[id(value)] = value
                continue
            cached = None if copies is None else copies.get(id(value))
            if cached is not None and cached[0]() is value and not value.is_deleted():
                replacements[id(value)] = cached[1]
                continue
            unique.setdefault(id(value), value)
    with bounded_device_transfers():
        moved = _transfer_arrays(
            lambda values: jax.device_put(values, sharding),
            tuple(unique.values()),
        )
    replacements.update(zip(unique, moved, strict=True))
    if copies is not None:
        copies.update(
            (identity, (ref(value), replacements[identity])) for identity, value in unique.items()
        )
    return tree.unflatten([replacements.get(id(value), value) for value in leaves])


def _numpy_host_inputs(point):
    leaves, tree = jax.tree.flatten(point)
    weak_types = tuple(
        value.weak_type if isinstance(value, jax.Array) else None for value in leaves
    )
    copies, unique, replacements = _host_copies.get(), {}, {}
    for value in leaves:
        if isinstance(value, jax.Array):
            cached = None if copies is None else copies.get(("numpy", id(value)))
            if cached is not None and cached[0]() is value and not value.is_deleted():
                replacements[id(value)] = cached[1]
            else:
                unique.setdefault(id(value), value)
    with bounded_device_transfers():
        moved = array_values_to_host(tuple(unique.values()))
    replacements.update(zip(unique, moved, strict=True))
    if copies is not None:
        copies.update(
            (("numpy", identity), (ref(value), replacements[identity]))
            for identity, value in unique.items()
        )
    return tree.unflatten([replacements.get(id(value), value) for value in leaves]), weak_types


def _restore_numpy_inputs(point, weak_types):
    leaves, tree = jax.tree.flatten(point)
    if len(leaves) != len(weak_types):
        raise ValueError("host stage input structure differs from saved weak types")
    unique = {(id(value), weak): value for value, weak in zip(leaves, weak_types, strict=True)
              if weak is not None}
    with bounded_device_transfers():
        placed = _transfer_arrays(lambda values: jax.device_put(values, jax.devices()[0]),
                                 tuple(unique.values()))
    replacements = {
        key: (_convert_element_type(value, value.dtype, weak_type=key[1])
              if value.weak_type != key[1] else value)
        for key, value in zip(unique, placed, strict=True)
    }
    return tree.unflatten([
        value if weak is None else replacements[id(value), weak]
        for value, weak in zip(leaves, weak_types, strict=True)
    ])


def recomputed_stage_replay(
    function,
    *,
    primal=None,
    cache_size=4,
    offload_inputs=False,
    stage_name=None,
    compiler_options=None,
    host_storage="pinned_host",
    stage_under_jit=False,
    replay_call_factory=None,
    linearization_transform=None,
):
    if cache_size < 1:
        raise ValueError("stage replay cache size must be positive")
    if host_storage not in {"pinned_host", "numpy"}:
        raise ValueError("stage host storage must be 'pinned_host' or 'numpy'")
    if type(stage_under_jit) is not bool:
        raise ValueError("staged primal calls must be boolean")
    if replay_call_factory is not None and not callable(replay_call_factory):
        raise ValueError("replayed call factory must be callable")
    if linearization_transform is not None and not callable(linearization_transform):
        raise ValueError("linearization transform must be callable")
    primal = function if primal is None else primal
    label = (getattr(function, "__name__", type(function).__name__)
             if stage_name is None else stage_name)
    compiler_options = None if compiler_options is None else dict(compiler_options)

    @lru_cache(maxsize=cache_size)
    def program(input_tree, mask, output_tree, selected):
        variables = tuple(i for i, flag in enumerate(mask) if flag)
        constants = tuple(i for i, flag in enumerate(mask) if not flag)

        def projected(differentiated, fixed):
            values = [None] * len(mask)
            for i, value in zip(variables, differentiated, strict=True):
                values[i] = value
            for i, value in zip(constants, fixed, strict=True):
                values[i] = value
            outputs, actual = jax.tree.flatten(function(input_tree.unflatten(values)))
            if actual != output_tree:
                raise ValueError("stage replay output structure differs from its cotangent")
            return tuple(outputs[i] for i in selected)

        differentiated_program = (projected if linearization_transform is None
                                  else linearization_transform(projected))
        forward, backward = jax.fwd_and_bwd(differentiated_program, argnums=0, jitted=False)

        def residuals_only(*arguments):
            _, residuals = forward(*arguments)
            return (), residuals

        calls = (
            jax.jit(residuals_only, compiler_options=compiler_options),
            jax.jit(backward, compiler_options=compiler_options),
        )
        return calls if replay_call_factory is None else tuple(map(replay_call_factory,calls))

    @jax.custom_vjp
    def reverse_stage(point):
        return primal(point)

    def forward(wrapped):
        leaves, tree = jax.tree.flatten(wrapped, is_leaf=lambda x: isinstance(x, CustomVJPPrimal))
        values = [leaf.value for leaf in leaves]
        mask = tuple(
            leaf.perturbed and np.dtype(jax.typeof(leaf.value).dtype).kind in "fc"
            for leaf in leaves
        )
        point = tree.unflatten(values)
        result = primal(point)
        offloaded = (
            offload_inputs
            and jax.default_backend() == "gpu"
            and not any(isinstance(value, jax.core.Tracer) for value in values)
        )
        host_weak_types = None
        if offloaded:
            if host_storage == "numpy":
                point, host_weak_types = _numpy_host_inputs(point)
            else:
                point = _place_arrays(point, "pinned_host")
        return result, _SavedStage(point, mask, offloaded, host_weak_types)

    def backward(saved, cotangent):
        fixed_output_slots = _fixed_replay_output_slots(label, cotangent)
        cotangent = jax.tree.map(
            lambda value: Zero(value.aval) if isinstance(value, SymbolicZero) else value,
            cotangent,
            is_leaf=_zero,
        )
        if not any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(cotangent)):
            cotangent = exact_zero_cotangents(cotangent)
        leaves, output_tree = jax.tree.flatten(cotangent, is_leaf=_zero)
        leaves = _fixed_replay_output_leaves(label, leaves, fixed_output_slots)
        selected = tuple(i for i, value in enumerate(leaves) if not _zero(value))
        point = saved.point
        values, input_tree = jax.tree.flatten(point)
        if not selected or not any(saved.mask):
            return (input_tree.unflatten([None] * len(values)),)
        observer = _observer.get()
        if observer is not None:
            observer(
                dict(
                    phase="apply_replayed_stage_adjoint",
                    stage=label,
                    nonzero_stage_outputs=len(selected),
                )
            )
        if saved.offloaded:
            point = (_place_arrays(point, "device") if saved.host_weak_types is None
                     else _restore_numpy_inputs(point, saved.host_weak_types))
            values, input_tree = jax.tree.flatten(point)
        differentiated = tuple(
            value for value, active in zip(values, saved.mask, strict=True) if active
        )
        constants = tuple(
            value for value, active in zip(values, saved.mask, strict=True) if not active
        )
        fwd, bwd = program(input_tree, saved.mask, output_tree, selected)
        outputs, residuals = fwd(differentiated, constants)
        del outputs
        gradients = bwd(residuals, tuple(leaves[i] for i in selected))
        concrete = not any(isinstance(value, jax.core.Tracer)
                           for value in jax.tree.leaves(gradients))
        if concrete:
            jax.block_until_ready(gradients)
        del residuals, point, values, differentiated, constants
        gradients = iter(gradients)
        restored = input_tree.unflatten(
            [next(gradients) if active else None for active in saved.mask]
        )
        if concrete:
            try:
                restored = jax.tree.map(
                    lambda value: None if _zero(value) else value,
                    exact_zero_cotangents(restored), is_leaf=_zero,
                )
            except FloatingPointError as error:
                if observer is not None:
                    observer(dict(phase="invalid_stage_adjoint", stage=label, error=str(error)))
                raise
        return (restored,)

    reverse_stage.defvjp(forward, backward, symbolic_zeros=True)

    @wraps(function)
    def execute(point):
        trace = core.unsafe_get_current_trace()
        if (isinstance(trace, ad.LinearizeTrace) or stage_under_jit
                and isinstance(trace, partial_eval.DynamicJaxprTrace)):
            return reverse_stage(point)
        return primal(point)

    execute.cache_info = program.cache_info
    execute.clear_cache = program.cache_clear
    return execute


_fixed_replay_output_stages = ContextVar("clm_fixed_replay_output_stages", default=frozenset())


@contextmanager
def fixed_replay_output_stages(stage_names=()):
    if (not isinstance(stage_names, (tuple, list))
            or any(not isinstance(name, str) or not name for name in stage_names)
            or len(set(stage_names)) != len(stage_names)):
        raise ValueError("distinct nonempty stage names required")
    token = _fixed_replay_output_stages.set(frozenset(stage_names))
    try:
        yield
    finally:
        _fixed_replay_output_stages.reset(token)


def _fixed_replay_output_slots(label, cotangent):
    if label not in _fixed_replay_output_stages.get():
        return None
    leaves, _ = jax.tree.flatten(cotangent, is_leaf=_zero)
    return tuple(not _zero(value) for value in leaves)


def _fixed_replay_output_leaves(label, leaves, fixed_output_slots):
    if fixed_output_slots is None or all(_zero(value) for value in leaves):
        return leaves
    import jax.numpy as jnp

    return [jnp.zeros(value.aval.shape, dtype=value.aval.dtype)
            if live and _zero(value) and np.dtype(value.aval.dtype).kind in "fc" else value
            for live, value in zip(fixed_output_slots, leaves, strict=True)]
