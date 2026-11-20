from contextlib import contextmanager
from contextvars import ContextVar

import jax

_synchronize = ContextVar("clm_stage_synchronization", default=True)


@contextmanager
def defer_stage_barriers(*, enabled):
    token = _synchronize.set(not enabled)
    try:
        yield
    finally:
        _synchronize.reset(token)


def stage_timing_synchronized():
    return _synchronize.get()


def stage_barrier(result):
    return jax.block_until_ready(result) if _synchronize.get() else result
