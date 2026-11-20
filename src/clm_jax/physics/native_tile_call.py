
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core, dispatch
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas.triton import lowering
from jax._src.state.types import TransformedRef
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

defer_native_inputs = ContextVar("clm_defer_native_inputs", default=False)
transpose_native_storage = ContextVar("clm_transpose_native_storage", default=False)
share_native_indices = ContextVar("clm_share_native_indices", default=False)


@contextmanager
def shared_native_tile_indices(*, enabled=True):
    if not isinstance(enabled, bool):
        raise ValueError("native tile index sharing must be Boolean")
    token = share_native_indices.set(enabled)
    try:
        yield
    finally:
        share_native_indices.reset(token)


def _indexed_tile_ref(ref, address, key, cache):
    if cache is None or isinstance(ref, TransformedRef):
        return ref.at[address]
    key = (ref.shape, key)
    template = cache.get(key)
    if template is None:
        view = ref.at[address]
        if view.ref is ref:
            cache[key] = (type(view), view.transforms)
        return view
    constructor, transforms = template
    return constructor(ref, transforms)

_while_p = core.Primitive("clm_output_sensitive_while")
_while_p.multiple_results = True
_while_p.def_impl(jax.lax.while_p.bind)
_while_p.def_effectful_abstract_eval(jax.lax.while_p.abstract_eval)
lowering.triton_lowering_rules[_while_p] = lowering.triton_lowering_rules[jax.lax.while_p]


def _eliminate_unused_carry(used, equation):
    if equation.effects:
        return [True] * len(equation.invars), equation
    if not any(used):
        return [False] * len(equation.invars), None
    parameters = equation.params
    nc, nb = parameters["cond_nconsts"], parameters["body_nconsts"]
    condition, body = parameters["cond_jaxpr"], parameters["body_jaxpr"]
    _, condition_inputs = pe.dce_jaxpr(condition.jaxpr, [True])
    needed = [a or b for a, b in zip(used, condition_inputs[nc:], strict=True)]
    while not all(needed):
        _, body_inputs = pe.dce_jaxpr(body.jaxpr, needed)
        expanded = [a or b for a, b in zip(needed, body_inputs[nb:], strict=True)]
        if expanded == needed:
            break
        needed = expanded
    condition_program, condition_inputs = pe.dce_jaxpr(
        condition.jaxpr, [True], instantiate=[False] * nc + needed,
    )
    body_program, body_inputs = pe.dce_jaxpr(
        body.jaxpr, needed, instantiate=[False] * nb + needed,
    )
    condition_constants, body_constants = condition_inputs[:nc], body_inputs[:nb]
    used_inputs = condition_constants + body_constants + needed
    return used_inputs, equation.replace(
        invars=[x for x, keep in zip(equation.invars, used_inputs, strict=True) if keep],
        outvars=[x for x, keep in zip(equation.outvars, needed, strict=True) if keep],
        params={**parameters, "cond_nconsts": sum(condition_constants),
                "body_nconsts": sum(body_constants),
                "cond_jaxpr": core.ClosedJaxpr(condition_program, condition.consts),
                "body_jaxpr": core.ClosedJaxpr(body_program, body.consts)},
    )


pe.dce_rules[_while_p] = _eliminate_unused_carry


def _own_loops(value, _memo=None):
    if _memo is None:
        _memo = {}
    if isinstance(value, (core.ClosedJaxpr, core.Jaxpr)):
        identity = id(value)
        if identity in _memo:
            return _memo[identity]
        if isinstance(value, core.ClosedJaxpr):
            result = core.ClosedJaxpr(_own_loops(value.jaxpr, _memo), value.consts)
        else:
            result = value.replace(eqns=[eqn.replace(
                primitive=_while_p if eqn.primitive is jax.lax.while_p else eqn.primitive,
                params={key: _own_loops(item, _memo) for key, item in eqn.params.items()},
            ) for eqn in value.eqns])
        _memo[identity] = result
        return result
    if isinstance(value, tuple) and not hasattr(value, "_fields"):
        return tuple(_own_loops(item, _memo) for item in value)
    return value


_program_with_owned_loops = lru_cache(maxsize=8)(_own_loops)


def _implementation(*leaves, program, signatures, batched, outputs, count, block,
                    defer_inputs=False, transpose_storage=False, share_indices=False):
    from clm_jax.physics.native_tile_io import load, store
    from clm_jax.physics.native_tile_tables import DeferredInput, evaluate, tables

    lookup = tables(program, {})
    table_values = tuple(
        jnp.asarray(np.frombuffer(data, dtype=np.dtype(dtype))) for data, dtype in lookup
    )
    input_count = len(leaves) + len(table_values)
    if transpose_storage:
        leaves = tuple(jnp.moveaxis(value, 0, -1) if batch else value
                       for value, batch in zip(leaves, batched, strict=True))

    def kernel(*refs):
        row = pl.program_id(0) * block + jnp.arange(block)
        safe = jnp.minimum(row, count - 1)
        index_cache = {} if share_indices else None
        values = []
        for ref, (shape, _), batch in zip(
            refs[:len(leaves)], signatures, batched, strict=True,
        ):
            for index in np.ndindex(shape):
                address = (
                    ((*index, safe) if transpose_storage else (safe, *index)) if batch else index
                )
                if defer_inputs:
                    values.append(DeferredInput(
                        ref, address, block, batch,
                    ))
                else:
                    value = load(_indexed_tile_ref(
                        ref, address, ("input", batch, index), index_cache,
                    ))
                    values.append(value if batch else jnp.broadcast_to(value, (block,)))
        table_refs = dict(zip(lookup, refs[len(leaves):input_count], strict=True))
        calculated = iter(evaluate(program, values, table_refs))
        for ref, output in zip(refs[input_count:], outputs, strict=True):
            for index in np.ndindex(output.shape):
                address = (*index, row) if transpose_storage else (row, *index)
                store(_indexed_tile_ref(ref, address, ("output", index), index_cache),
                      next(calculated), mask=row < count)

    result = pl.pallas_call(
        kernel,
        out_shape=tuple(jax.ShapeDtypeStruct(
            (*x.shape, count) if transpose_storage else (count, *x.shape), x.dtype,
        ) for x in outputs),
        grid=(pl.cdiv(count, block),),
        compiler_params=plt.CompilerParams(
            num_warps=min(4, max(1, block // 32)), num_stages=1,
        ),
    )(*leaves, *table_values)
    return tuple(jnp.moveaxis(value, -1, 0) for value in result) if transpose_storage else result


native_tile_p = core.Primitive("clm_pure_native_tile")
native_tile_p.multiple_results = True
dispatch.simple_impl(native_tile_p)
native_tile_p.def_abstract_eval(
    lambda *_, outputs, count, **__: tuple(
        core.ShapedArray((count, *x.shape), x.dtype) for x in outputs
    )
)
mlir.register_lowering(native_tile_p, mlir.lower_fun(_implementation, multiple_results=True))


def _eliminate_unused_outputs(used, equation):
    if not any(used):
        return [False] * len(equation.invars), None
    parameters = equation.params
    storage_used = [
        enabled for enabled, field in zip(used, parameters["outputs"], strict=True)
        for _ in np.ndindex(field.shape)
    ]
    original = parameters["program"].jaxpr
    _, input_elements = pe.dce_jaxpr(original, storage_used)
    cursor = 0
    input_fields = []
    for shape, _ in parameters["signatures"]:
        size = int(np.prod(shape))
        input_fields.append(any(input_elements[cursor:cursor + size]))
        cursor += size
    program, _ = pe.dce_jaxpr(original, storage_used, instantiate=[
        keep for keep, (shape, _) in zip(input_fields, parameters["signatures"], strict=True)
        for _ in np.ndindex(shape)
    ])
    result = equation.replace(
        invars=[value for value, keep in zip(equation.invars, input_fields, strict=True) if keep],
        outvars=[value for value, enabled in zip(equation.outvars, used, strict=True) if enabled],
        params={
            **parameters,
            "program": core.ClosedJaxpr(program, ()),
            "signatures": tuple(
                value for value, keep in zip(parameters["signatures"], input_fields, strict=True)
                if keep
            ),
            "batched": tuple(
                value for value, keep in zip(parameters["batched"], input_fields, strict=True)
                if keep
            ),
            "outputs": tuple(
                value for value, enabled in zip(parameters["outputs"], used, strict=True)
                if enabled
            ),
        },
    )
    return input_fields, result


pe.dce_rules[native_tile_p] = _eliminate_unused_outputs


def native_tile_call(leaves, program, signatures, batched, outputs, count, block):
    return native_tile_p.bind(
        *leaves, program=_program_with_owned_loops(program),
        signatures=signatures, batched=tuple(batched),
        outputs=tuple(outputs), count=count, block=block,
        defer_inputs=defer_native_inputs.get(),
        transpose_storage=transpose_native_storage.get(),
        share_indices=share_native_indices.get(),
    )
