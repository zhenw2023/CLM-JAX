
import functools
from contextvars import ContextVar

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import core

from clm_jax.physics.native_tile_io import load
from clm_jax.physics.solver_constant_lookup import constant_lookup_p


_deferred_address_selection = ContextVar("clm_deferred_address_selection", default=None)


def _integer_coordinate(value):
    try:
        return jax.typeof(value).dtype.kind in "iu"
    except (TypeError, AttributeError):
        return False


def _select_deferred_input(selector, first, second):
    if not (isinstance(first, DeferredInput) and isinstance(second, DeferredInput)
            and first.ref is second.ref and first.block == second.block
            and first.batched == second.batched
            and len(first.index) == len(second.index)
            and jax.typeof(selector).dtype.kind == "b"):
        return None
    coordinates = []
    for left, right in zip(first.index, second.index, strict=True):
        if left is right or (type(left) is int and type(right) is int and left == right):
            coordinates.append(left)
        elif _integer_coordinate(left) and _integer_coordinate(right):
            coordinates.append(jnp.where(selector, right, left))
        else:
            return None
    return DeferredInput(first.ref, tuple(coordinates), first.block, first.batched)


def _deferred_select_outputs(program, values):
    constants = program.consts if isinstance(program, core.ClosedJaxpr) else ()
    program = program.jaxpr if isinstance(program, core.ClosedJaxpr) else program
    if len(program.eqns) != 1:
        return None
    equation, = program.eqns
    if (equation.primitive is not lax.select_n_p or len(equation.invars) != 3
            or len(program.outvars) != 1 or program.outvars[0] is not equation.outvars[0]):
        return None
    inputs = dict(zip((*program.constvars, *program.invars), (*constants, *values), strict=True))
    selector, first, second = [value.val if isinstance(value, core.Literal) else inputs[value]
                               for value in equation.invars]
    if isinstance(selector, DeferredInput):
        selector = selector.resolve()
    selected = _select_deferred_input(selector, first, second)
    return None if selected is None else [selected]



class DeferredInput:

    def __init__(self, ref, index, block, batched):
        self.ref, self.index, self.block, self.batched = ref, index, block, batched

    def resolve(self):
        value = load(self.ref.at[self.index])
        return value if self.batched else jnp.broadcast_to(value, (self.block,))


def tables(value, result, _seen=None):
    if _seen is None:
        _seen = set()
    if isinstance(value, (core.ClosedJaxpr, core.Jaxpr)):
        identity = id(value)
        if identity in _seen:
            return result
        _seen.add(identity)
    if isinstance(value, core.ClosedJaxpr):
        return tables(value.jaxpr, result, _seen)
    if isinstance(value, core.Jaxpr):
        for eq in value.eqns:
            if eq.primitive is constant_lookup_p:
                key = (eq.params["data"], eq.params["dtype"])
                result.setdefault(key, len(result))
            for v in eq.params.values():
                tables(v, result, _seen)
    elif isinstance(value, (list, tuple)):
        for v in value:
            tables(v, result, _seen)
    return result


@functools.lru_cache(maxsize=128)
def _nested_evaluator(program):
    return jax.jit(functools.partial(evaluate, program))


def evaluate(program, values, refs):
    from clm_jax.physics.native_tile_call import _while_p

    constants = program.consts if isinstance(program, core.ClosedJaxpr) else ()
    program = program.jaxpr if isinstance(program, core.ClosedJaxpr) else program
    env = dict(zip((*program.constvars, *program.invars), (*constants, *values), strict=True))

    def read(x):
        value = raw(x)
        if isinstance(value, DeferredInput):
            value = value.resolve()
            env[x] = value
        return value

    def raw(x):
        return x.val if isinstance(x, core.Literal) else env[x]

    for eq in program.eqns:
        params = eq.params
        if eq.primitive is constant_lookup_p:
            args = [read(x) for x in eq.invars]
            ref = refs[(params["data"], params["dtype"])]
            out = [load(ref.at[jnp.clip(args[0], 0, ref.shape[0] - 1)])]
        elif eq.primitive is lax.select_n_p and _deferred_address_selection.get() is not None:
            selected = (_select_deferred_input(read(eq.invars[0]), raw(eq.invars[1]), raw(eq.invars[2]))
                        if len(eq.invars) == 3 else None)
            if selected is None:
                out = [eq.primitive.bind(*(read(x) for x in eq.invars), **params)]
            else:
                _deferred_address_selection.get()["deferred_address_selections"] += 1
                out = [selected]
        elif eq.primitive in (_while_p, lax.while_p):
            nc, nb = params["cond_nconsts"], params["body_nconsts"]
            cs = [raw(x) for x in eq.invars[:nc]]
            bs = [raw(x) for x in eq.invars[nc : nc + nb]]
            carry = tuple(read(x) for x in eq.invars[nc + nb :])

            def condition(carry, p=params, c=cs):
                return evaluate(p["cond_jaxpr"], [*c, *carry], refs)[0]

            def body(carry, p=params, b=bs):
                return tuple(evaluate(p["body_jaxpr"], [*b, *carry], refs))

            out = lax.while_loop(condition, body, carry)
        elif eq.primitive is lax.cond_p:
            args = [raw(x) for x in eq.invars[1:]]
            if any(isinstance(value, DeferredInput) for value in args):
                branches = tuple(
                    lambda _, branch=branch: evaluate(branch, args, refs)
                    for branch in params["branches"]
                )
                out = lax.switch(read(eq.invars[0]), branches, ())
            else:
                branches = tuple(functools.partial(evaluate, b, refs=refs) for b in params["branches"])
                out = lax.switch(read(eq.invars[0]), branches, tuple(args))
        elif eq.primitive.name in ("jit", "pjit", "closed_call", "call"):
            nested = params.get("jaxpr", params.get("call_jaxpr"))
            args = [raw(x) for x in eq.invars]
            out = (_deferred_select_outputs(nested, args)
                   if _deferred_address_selection.get() is not None else None)
            if out is not None:
                _deferred_address_selection.get()["deferred_address_selections"] += 1
            elif any(isinstance(value, DeferredInput) for value in args):
                out = evaluate(nested, args, refs)
            else:
                from clm_jax.physics.native_tile_shared_call import share_tile_lowering, shared_call

                execute = _nested_evaluator(nested)
                out = (shared_call(execute, args, refs) if share_tile_lowering.get()
                       else execute(args, refs))
        else:
            args = [read(x) for x in eq.invars]
            out = eq.primitive.bind(*args, **params)
            if not eq.primitive.multiple_results:
                out = [out]
        for var, value in zip(eq.outvars, out, strict=True):
            env[var] = value
    return [read(x) for x in program.outvars]
