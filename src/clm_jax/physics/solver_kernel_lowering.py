from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src import api_util, linear_util
from jax._src.interpreters import partial_eval
from jax._src.pallas.triton import lowering
from jax._src.pallas.triton.primitives import elementwise_inline_asm_p
from jax.experimental.pallas import triton as plt
from jax.extend import core

from clm_jax.physics.solver_constant_lookup import constant_lookup_p
from clm_jax.physics.solver_fma import fma_p


def _constant_lookup(ctx, index, *, data, dtype):
    if not ctx.avals_in[0].shape:
        def scalar_lookup(value):
            return constant_lookup_p.bind(
                value.reshape((1,)), data=data, dtype=dtype,
            )[0]

        return lowering.lower_fun(scalar_lookup, multiple_results=False)(ctx, index)
    values = np.frombuffer(data, dtype=np.dtype(dtype))
    size = 1 << (values.size - 1).bit_length()
    values = np.pad(values, (0, size - values.size), mode="edge")
    table_type = lowering.ir.RankedTensorType.get(
        (size,), lowering._dtype_to_ir_type(values.dtype),
    )
    attribute = lowering.ir.DenseElementsAttr.get(values, type=table_type)
    table = lowering.arith_dialect.ConstantOp(table_type, attribute).result
    index = lowering._ensure_ir_value(index, ctx.avals_in[0])
    return lowering.tt_dialect.gather(table, index, 0)


lowering.triton_lowering_rules[constant_lookup_p] = _constant_lookup


def _vector_inline_asm(ctx, values, assembly, constraints):
    output = ctx.avals_out[0]
    values = [lowering._bcast_to(lowering._ensure_ir_value(value, aval), output.shape)
              for value, aval in zip(values, ctx.avals_in, strict=True)]
    context = ctx.replace(avals_in=(output,) * len(values))
    return lowering.triton_lowering_rules[elementwise_inline_asm_p](
        context, *values, asm=assembly, constraints=constraints, pack=1,
        result_shape_dtypes=(jax.ShapeDtypeStruct(output.shape, output.dtype),),
    )[0]


def _fused_multiply_add(ctx, left, right, addend):
    if ctx.avals_out[0].shape:
        return _vector_inline_asm(
            ctx, (left, right, addend), "fma.rn.f64 $0, $1, $2, $3;", "=d,d,d,d",
        )
    values = [lowering._ensure_ir_value(value, aval) for value, aval in
              zip((left, right, addend), ctx.avals_in, strict=True)]
    return lowering._closed_call_lowering_rule(
        ctx, *values, call_jaxpr=_fma_program(tuple(ctx.avals_in)),
    )[0]


@lru_cache(maxsize=128)
def _fma_program(avals):
    def calculate(a, b, c):
        scalar = a.ndim == 0
        if scalar:
            a, b, c = a.reshape((1,)), b.reshape((1,)), c.reshape((1,))
        result, = plt.elementwise_inline_asm(
            "fma.rn.f64 $0, $1, $2, $3;", args=[a, b, c], constraints="=d,d,d,d",
            pack=1, result_shape_dtypes=[jax.ShapeDtypeStruct(a.shape, jnp.float64)],
        )
        return jnp.max(result) if scalar else result

    return _trace_program(calculate, avals)


lowering.triton_lowering_rules[fma_p] = _fused_multiply_add


def _round(ctx, value, *, rounding_method):
    if rounding_method != lax.RoundingMethod.TO_NEAREST_EVEN:
        raise NotImplementedError("native solver only uses nearest-even table coordinates")
    dtype = ctx.avals_in[0].dtype
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise NotImplementedError("native solver rounding requires float32 or float64")
    value = lowering._ensure_ir_value(value, ctx.avals_in[0])
    if ctx.avals_out[0].shape:
        bits = dtype.itemsize * 8
        return _nan_preserving_vector_instruction(
            ctx, value, f"cvt.rni.f{bits}.f{bits} $0, $1;",
        )
    return lowering._closed_call_lowering_rule(
        ctx, value, call_jaxpr=_round_program(tuple(ctx.avals_in)),
    )[0]


@lru_cache(maxsize=128)
def _round_program(avals):
    dtype = avals[0].dtype
    bits = dtype.itemsize * 8
    register = "d" if bits == 64 else "f"
    def calculate(x):
        scalar = x.ndim == 0
        if scalar:
            x = x.reshape((1,))
        result = plt.elementwise_inline_asm(
            f"cvt.rni.f{bits}.f{bits} $0, $1;", args=[x],
            constraints=f"={register},{register}", pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct(x.shape, dtype)],
        )[0]
        result = jnp.where(jnp.isnan(x), x, result)
        return jnp.max(result) if scalar else result

    return _trace_program(calculate, avals)


@lru_cache(maxsize=128)
def _rounded_program(operation, avals):
    def calculate(a, b):
        a, b = jnp.broadcast_arrays(a, b)
        scalar = a.ndim == 0
        if scalar:
            a, b = a.reshape((1,)), b.reshape((1,))
        result = plt.elementwise_inline_asm(
            f"{operation}.rn.f64 $0, $1, $2;", args=[a, b],
            constraints="=d,d,d", pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct(a.shape, jnp.float64)],
        )[0]
        return jnp.max(result) if scalar else result

    return _trace_program(calculate, avals)


def _trace_program(function, avals):
    def results(*values):
        return (function(*values),)
    shapes = tuple(jax.ShapeDtypeStruct(aval.shape, aval.dtype) for aval in avals)
    wrapped = linear_util.wrap_init(
        results,
        debug_info=api_util.debug_info("CLM native rounded arithmetic", results, shapes, {}),
    )
    program, _, constants = partial_eval.trace_to_jaxpr_dynamic(wrapped, avals, lower=True)
    return core.ClosedJaxpr(program, constants)


def _rounded(operation, fallback, ctx, left, right, **params):
    if ctx.avals_out[0].dtype != np.dtype("float64"):
        return fallback(ctx, left, right, **params)
    if ctx.avals_out[0].shape:
        return _vector_inline_asm(
            ctx, (left, right), f"{operation}.rn.f64 $0, $1, $2;", "=d,d,d",
        )
    left = lowering._ensure_ir_value(left, ctx.avals_in[0])
    right = lowering._ensure_ir_value(right, ctx.avals_in[1])
    return lowering._closed_call_lowering_rule(
        ctx, left, right, call_jaxpr=_rounded_program(operation, tuple(ctx.avals_in)),
    )[0]


def _local_max(ctx, value, *, axes, axis_index_groups):
    if axis_index_groups is not None or not all(isinstance(axis, int) for axis in axes):
        raise ValueError("solver kernel only supports independent local spatial axes")
    return lowering._reduce_lowering(jnp.maximum, ctx, value, axes=axes)


def _slice(ctx, value, *, start_indices, limit_indices, strides):
    value = lowering._ensure_ir_value(value, ctx.avals_in[0])
    shape = list(ctx.avals_in[0].shape)
    for axis, (start, limit, stride) in enumerate(zip(
        start_indices, limit_indices, strides or (1,) * len(shape), strict=True,
    )):
        if start == 0 and limit == shape[axis] and stride == 1:
            continue
        if stride != 1:
            raise ValueError("solver kernel requires unit-stride static slices")
        shape[axis] = limit - start
        indices = lowering._make_range(start, limit)
        for dimension in range(len(shape)):
            if dimension != axis:
                indices = lowering._expand_dims(indices, dimension)
        value = lowering.tt_dialect.gather(value, lowering._bcast_to(indices, shape), axis)
    return value


def _select(ctx, which, *values):
    if ctx.avals_in[0].dtype != np.dtype("bool"):
        if len(values) != 2:
            raise ValueError("solver kernel requires binary indexed selection")
        which = lowering._ensure_ir_value(which, ctx.avals_in[0])
        which = lowering._not_equal(which, lowering._zeros_like(which), signed=True)
        ctx = ctx.replace(avals_in=(
            ctx.avals_in[0].update(dtype=np.dtype("bool")), *ctx.avals_in[1:],
        ))
    return lowering.triton_lowering_rules[lax.select_n_p](ctx, which, *values)


def _bitcast(ctx, value, *, new_dtype):
    value = lowering._ensure_ir_value(value, ctx.avals_in[0])
    return lowering.triton_lowering_rules[lax.bitcast_convert_type_p](
        ctx, value, new_dtype=new_dtype,
    )


def _negate(ctx, value):
    value = lowering._ensure_ir_value(value, ctx.avals_in[0])
    if ctx.avals_in[0].dtype in (np.dtype("float32"), np.dtype("float64")):
        if ctx.avals_out[0].shape:
            width = ctx.avals_out[0].dtype.itemsize * 8
            return _nan_preserving_vector_instruction(
                ctx, value, f"xor.b{width} $0, $1, 0x{1 << (width - 1):x};",
            )
        return lowering._closed_call_lowering_rule(
            ctx, value, call_jaxpr=_negate_program(tuple(ctx.avals_in)),
        )[0]
    return lowering.triton_lowering_rules[lax.neg_p](ctx, value)


def _nan_preserving_vector_instruction(ctx, value, instruction):
    output = ctx.avals_out[0]
    bits = output.dtype.itemsize * 8
    register = "d" if bits == 64 else "f"
    assembly = (
        f"{{ .reg .pred p; testp.notanumber.f{bits} p, $1; "
        f"mov.b{bits} $0, $1; @!p {instruction} }}"
    )
    return lowering.triton_lowering_rules[elementwise_inline_asm_p](
        ctx, value, asm=assembly, constraints=f"={register},{register}", pack=1,
        result_shape_dtypes=(jax.ShapeDtypeStruct(output.shape, output.dtype),),
    )[0]


@lru_cache(maxsize=128)
def _negate_program(avals):
    dtype = avals[0].dtype
    width = dtype.itemsize * 8
    integer = jnp.int64 if width == 64 else jnp.int32
    register = "l" if width == 64 else "r"

    def flip_sign(x):
        scalar = x.ndim == 0
        if scalar:
            x = x.reshape((1,))
        bits = lax.bitcast_convert_type(x, integer)
        flipped, = plt.elementwise_inline_asm(
            f"xor.b{width} $0, $1, 0x{1 << (width - 1):x};",
            args=[bits], constraints=f"={register},{register}", pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct(x.shape, integer)],
        )
        result = lax.bitcast_convert_type(flipped, dtype)
        result = jnp.where(jnp.isnan(x), x, result)
        return result[0] if scalar else result

    return _trace_program(flip_sign, avals)


def _atan(ctx, value):
    value = lowering._ensure_ir_value(value, ctx.avals_in[0])
    return lowering._closed_call_lowering_rule(
        ctx, value, call_jaxpr=_atan_program(tuple(ctx.avals_in)),
    )[0]


@lru_cache(maxsize=128)
def _atan_program(avals):
    return _trace_program(lambda x: jnp.arctan2(x, jnp.ones_like(x)), avals)


_rules = {
    lax.round_p: _round,
    lax.optimization_barrier_p: lambda ctx, *values: values,
    lax.reduce_and_p: partial(lowering._reduce_lowering, jnp.bitwise_and),
    lax.reduce_or_p: partial(lowering._reduce_lowering, jnp.bitwise_or),
    lax.pmax_p: _local_max,
    lax.slice_p: _slice,
    lax.select_n_p: _select,
    lax.bitcast_convert_type_p: _bitcast,
    lax.neg_p: _negate,
    lax.atan_p: _atan,
}
for _primitive, _name in ((lax.add_p, "add"), (lax.sub_p, "sub"),
                          (lax.mul_p, "mul"), (lax.div_p, "div")):
    _rules[_primitive] = partial(
        _rounded, _name, lowering.triton_lowering_rules[_primitive],
    )

_primitives = {}
for _original, _rule in _rules.items():
    _owned = core.Primitive("clm_solver_" + _original.name)
    _owned.multiple_results = _original.multiple_results
    _owned.def_impl(_original.bind)
    _owned.def_effectful_abstract_eval(_original.abstract_eval)
    lowering.triton_lowering_rules[_owned] = _rule
    _primitives[_original] = _owned


def native_solver_jaxpr(value, _memo=None):
    if _memo is None:
        _memo = {}
    if isinstance(value, (core.ClosedJaxpr, core.Jaxpr)):
        identity = id(value)
        if identity in _memo:
            return _memo[identity]
        if isinstance(value, core.ClosedJaxpr):
            result = core.ClosedJaxpr(native_solver_jaxpr(value.jaxpr, _memo), value.consts)
        else:
            result = value.replace(eqns=[eqn.replace(
                primitive=_primitives.get(eqn.primitive, eqn.primitive),
                params={key: native_solver_jaxpr(item, _memo) for key, item in eqn.params.items()},
            ) for eqn in value.eqns])
        _memo[identity] = result
        return result
    if isinstance(value, tuple):
        values = tuple(native_solver_jaxpr(item, _memo) for item in value)
        return type(value)(*values) if hasattr(value, "_fields") else values
    if isinstance(value, list):
        return [native_solver_jaxpr(item, _memo) for item in value]
    return value
