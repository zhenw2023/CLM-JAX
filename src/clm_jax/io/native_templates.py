
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import jax
import jax.numpy as jnp
from jax.extend.core import get_opaque_trace_state

_templates = ContextVar("clm_native_templates", default=None)
with jax.core.eval_context():
    _EAGER_TRACE = get_opaque_trace_state()


@contextmanager
def native_template_scope(*, enabled=True):
    token = _templates.set({} if enabled else None)
    try:
        yield
    finally:
        _templates.reset(token)


def native_template(function):

    def construct(*args, **kwargs):
        with jax.core.eval_context():
            value = function(*args, **kwargs)
        if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value)):
            raise ValueError("native input templates cannot contain live traced state")
        return value

    @wraps(function)
    def load(*args, **kwargs):
        templates = _templates.get()
        if templates is None:
            return construct(*args, **kwargs)
        key = (function, args, tuple(sorted(kwargs.items())))
        if key not in templates:
            templates[key] = construct(*args, **kwargs)
        return templates[key]

    return load


@native_template
def _allocate_zeros(shape, dtype, device):
    return jnp.zeros(shape, dtype=dtype, device=device)


def native_zeros(shape, dtype):
    if get_opaque_trace_state() != _EAGER_TRACE or jax.default_backend() != "gpu":
        return jnp.zeros(shape, dtype=dtype)
    return _allocate_zeros(tuple(shape), jnp.dtype(dtype), jax.devices()[0])
