
from contextvars import ContextVar
from dataclasses import replace
from functools import lru_cache

import jax
from jax._src import api_util, core, linear_util
from jax._src.interpreters import mlir, partial_eval
from jax._src.lib.mlir import ir
from jax._src.lib.triton import dialect as tt
from jax._src.pallas.triton import lowering

share_tile_lowering = ContextVar("clm_share_tile_lowering", default=True)

call_p = core.Primitive("clm_shared_tile_call")
call_p.multiple_results = True
call_p.def_impl(lambda *values, program: core.eval_jaxpr(program.jaxpr, program.consts, *values))
call_p.def_effectful_abstract_eval(
    lambda *_, program: ([value.aval for value in program.jaxpr.outvars],
                        core.positional_effects(program))
)


def _lower(ctx, *values, program):
    if program.consts or any(info is not None for info in ctx.block_infos):
        return lowering._closed_call_lowering_rule(ctx, *values, call_jaxpr=program)
    arguments = [lowering._ensure_ir_value(value, aval)
                 for value, aval in zip(values, ctx.avals_in, strict=True)]
    module = ctx.context.mlir_ctx.module
    cache = getattr(ctx.context, "_clm_shared_callees", None)
    if cache is None:
        cache = {}
        ctx.context._clm_shared_callees = cache
    program_ids = ctx.context.program_ids
    inputs = [*arguments, *program_ids]
    key = (program, tuple(value.type for value in inputs))
    if key not in cache:
        name = "clm_shared_" + str(len(cache))
        output_types = [
            ir.RankedTensorType.get(value.shape, lowering._dtype_to_ir_type(value.dtype))
            if value.shape else lowering._dtype_to_ir_type(value.dtype)
            for value in ctx.avals_out
        ]
        signature = ir.FunctionType.get([value.type for value in inputs], output_types)
        function = tt.FuncOp(
            name, ir.TypeAttr.get(signature), sym_visibility="private",
            res_attrs=ir.DictAttr.get(dict(noinline=ir.BoolAttr.get(False))),
            ip=ir.InsertionPoint.at_block_begin(module.body),
        )
        cache[key] = (name, output_types)
        function.body.blocks.append(*signature.inputs)
        entry = function.body.blocks[0]
        child = replace(ctx.context, program_ids=list(entry.arguments[len(arguments):]))
        child._clm_shared_callees = cache
        with ir.InsertionPoint(entry):
            results = lowering.lower_jaxpr_to_triton_ir(
                child, program.jaxpr, None, *entry.arguments[:len(arguments)],
            )
            tt.return_([lowering._ensure_ir_value(value, aval)
                        for value, aval in zip(results, ctx.avals_out, strict=True)])
    name, output_types = cache[key]
    return tt.CallOp(output_types, ir.FlatSymbolRefAttr.get(name), inputs).results


lowering.triton_lowering_rules[call_p] = _lower
mlir.register_lowering(
    call_p, lambda ctx, *values, program: mlir.core_call_lowering(
        ctx, *values, name="clm_shared_tile", call_jaxpr=program,
    ),
)


@lru_cache(maxsize=256)
def _program(evaluator, tree, avals):
    def calculate(*values):
        return evaluator(*tree.unflatten(values))

    wrapped = linear_util.wrap_init(
        calculate,
        debug_info=api_util.debug_info("CLM shared tile", calculate, (), {}),
    )
    program, _, constants = partial_eval.trace_to_jaxpr_dynamic(wrapped, avals)
    return core.ClosedJaxpr(program, constants)


def shared_call(evaluator, *arguments):
    leaves, tree = jax.tree.flatten(arguments)
    program = _program(evaluator, tree, tuple(jax.typeof(value) for value in leaves))
    return call_p.bind(*leaves, program=program)
