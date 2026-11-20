
from contextvars import ContextVar
from functools import wraps, partial

import jax
import jax.numpy as jnp
from jax import lax

from clm_jax.compilation import _stage_reverse_ad, _symbolic_zero_jvp

SPATIAL_SOLVE_AXIS = "clm_independent_spatial_solve"
_dynamic_primal = ContextVar("clm_dynamic_convergence_primal", default=False)
_checkpoint_iterations = ContextVar("clm_checkpoint_iteration_ad", default=False)


def iteration_ad_checkpointing():
    return _checkpoint_iterations.get() or _stage_reverse_ad.get()


def checkpoint_iteration(step):
    if iteration_ad_checkpointing():
        return jax.checkpoint(step, prevent_cse=False)
    return step


def checkpointed_scan(step, initial, iterations, *, block_size=8):
    if not iteration_ad_checkpointing():
        return lax.scan(step, initial, iterations)
    if block_size < 1:
        raise ValueError("checkpoint block size must be positive")
    count = iterations.shape[0]
    blocks, remainder = divmod(count, block_size)
    step = checkpoint_iteration(step)

    def block(state, indices):
        return lax.scan(step, state, indices)

    state, history = initial, None
    if blocks:
        state, history = lax.scan(
            checkpoint_iteration(block), state,
            iterations[:blocks * block_size].reshape(
                (blocks, block_size, *iterations.shape[1:])
            ),
        )
        history = jax.tree.map(
            lambda value: value.reshape((blocks * block_size, *value.shape[2:])), history
        )
    if remainder or not blocks:
        state, tail = lax.scan(step, state, iterations[blocks * block_size:])
        history = tail if not blocks else jax.tree.map(
            lambda a, b: jnp.concatenate((a, b), axis=0), history, tail
        )
    return state, history


def checkpointed_fori_loop(lower, upper, body, initial, *, block_size=16):
    if not iteration_ad_checkpointing():
        return lax.fori_loop(lower, upper, body, initial)
    if block_size < 1:
        raise ValueError("checkpoint block size must be positive")
    count = max(upper - lower, 0)
    blocks, remainder = divmod(count, block_size)
    body = checkpoint_iteration(body)

    def block(state, block_index):
        offset = lower + block_index * block_size
        state = lax.fori_loop(
            0, block_size, lambda index, carry: body(offset + index, carry), state
        )
        return state, None

    state = initial
    if blocks:
        state = lax.scan(
            checkpoint_iteration(block), state, jnp.arange(blocks, dtype=jnp.int32)
        )[0]
    if remainder:
        state = lax.fori_loop(lower + blocks * block_size, upper, body, state)
    return state


def gpu_convergence_scope(function):

    @jax.custom_jvp
    @wraps(function)
    def scoped(*arguments):
        token = _dynamic_primal.set(True)
        try:
            return function(*arguments)
        finally:
            _dynamic_primal.reset(token)

    @partial(scoped.defjvp, symbolic_zeros=True)
    def scoped_jvp(primals, tangents):
        token = _dynamic_primal.set(False)
        checkpoint_token = _checkpoint_iterations.set(True)
        try:
            return _symbolic_zero_jvp(function, primals, tangents)
        finally:
            _checkpoint_iterations.reset(checkpoint_token)
            _dynamic_primal.reset(token)

    @wraps(function)
    def call(*arguments):
        if jax.default_backend() == "cpu":
            return function(*arguments)
        return scoped(*arguments)

    return call


def converged_iteration(step, initial, iterations, is_active):
    count = len(iterations) if isinstance(iterations, range) else iterations.shape[0]
    if count == 0:
        return initial
    if jax.default_backend() == "cpu":
        if isinstance(iterations, range):
            iterations = jnp.asarray(iterations, dtype=jnp.int32)
        return checkpointed_scan(step, initial, iterations)[0]

    def enabled(state):
        enabled = jnp.any(is_active(state))
        try:
            enabled = lax.pmax(enabled.astype(jnp.int32), SPATIAL_SOLVE_AXIS) > 0
        except NameError:
            pass
        return enabled

    def guarded_step(state, iteration):
        return lax.cond(
            enabled(state),
            lambda current: step(current, iteration),
            lambda current: (current, None),
            state,
        )

    if _dynamic_primal.get():

        def condition(carry):
            index, state = carry
            return (index < count) & enabled(state)

        def body(carry):
            index, state = carry
            iteration = (iterations.start + index * iterations.step
                         if isinstance(iterations, range) else iterations[index])
            return index + 1, step(state, iteration)[0]

        return lax.while_loop(condition, body, (jnp.int32(0), initial))[1]

    if isinstance(iterations, range):
        iterations = jnp.asarray(iterations, dtype=jnp.int32)
    return checkpointed_scan(guarded_step, initial, iterations)[0]
