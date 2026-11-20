
import types

import jax.numpy as jnp
from jax._src.pallas import primitives as pp
from jax._src.pallas.triton import lowering as lo
from jax.extend import core


def clone(function, **replacements):
    result = types.FunctionType(
        function.__code__,
        {**function.__globals__, **replacements},
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def tensor_type(value, element):
    original = lo.ir.RankedTensorType(value.type)
    return lo.ir.RankedTensorType.get(original.shape, element, original.encoding)


def memory_types(ptr):
    element = lo.tt_dialect.PointerType(lo._element_type(ptr.type)).pointee_type
    bits = element.width
    width = 64 if bits == 64 else 32
    integer = lo.ir.IntegerType.get_signless(width)
    return element, bits, integer, tensor_type(ptr, integer)


def asm_load(ptr, mask=None, other=None, **options):
    if not isinstance(ptr.type, lo.ir.RankedTensorType):
        return lo._load(ptr, mask, other, **options)
    if any(value not in (None, False) for value in options.values()):
        raise NotImplementedError(options)
    element, bits, integer, result_type = memory_types(ptr)
    width = integer.width
    address = lo.tt_dialect.ptr_to_int(tensor_type(ptr, lo.ir.IntegerType.get_signless(64)), ptr)
    mask = (
        lo._full(tensor_type(ptr, lo.ir.IntegerType.get_signless(1)), 1) if mask is None else mask
    )
    pred = lo._ir_cast(mask, lo.ir.IntegerType.get_signless(32), signed=False)
    register = "l" if width == 64 else "r"
    suffix = f"b{bits}" if bits >= 32 else f"u{max(bits, 8)}"
    result = lo.tt_dialect.elementwise_inline_asm(
        [result_type],
        "{ .reg .pred p; setp.ne.u32 p, $2, 0; "
        f"mov.b{width} $0, 0; @p ld.global.{suffix} $0, [$1]; }}",
        constraints=f"={register},l,r",
        pure=True,
        packed_element=1,
        args=[address, pred],
    )
    target = tensor_type(ptr, element)
    if bits < width:
        result = lo._ir_cast(result, lo.ir.IntegerType.get_signless(bits), signed=False)
    if result.type != target:
        result = lo.tt_dialect.bitcast(target, result)
    if other is not None:
        result = lo.arith_dialect.select(mask, result, other)
    return result


def asm_store(ptr, value, mask=None, **options):
    if not isinstance(ptr.type, lo.ir.RankedTensorType):
        return lo._store(ptr, value, mask, **options)
    if any(value is not None for value in options.values()):
        raise NotImplementedError(options)
    element, bits, integer, result_type = memory_types(ptr)
    width = integer.width
    address = lo.tt_dialect.ptr_to_int(tensor_type(ptr, lo.ir.IntegerType.get_signless(64)), ptr)
    mask = (
        lo._full(tensor_type(ptr, lo.ir.IntegerType.get_signless(1)), 1) if mask is None else mask
    )
    pred = lo._ir_cast(mask, lo.ir.IntegerType.get_signless(32), signed=False)
    if isinstance(element, lo.ir.IntegerType):
        value = lo._ir_cast(value, integer, signed=False)
    else:
        value = lo.tt_dialect.bitcast(result_type, value)
    register = "l" if width == 64 else "r"
    suffix = f"b{bits}" if bits >= 32 else f"u{max(bits, 8)}"
    lo.tt_dialect.elementwise_inline_asm(
        [pred.type],
        f"{{ .reg .pred p; setp.ne.u32 p, $3, 0; mov.b32 $0, 0; @p st.global.{suffix} [$1], $2; }}",
        constraints=f"=r,l,{register},r",
        pure=False,
        packed_element=1,
        args=[address, value, pred],
    )


load_p = core.Primitive("clm_native_direct_load")
load_p.def_effectful_abstract_eval(pp.load_p.abstract_eval)
swap_p = core.Primitive("clm_native_direct_swap")
swap_p.def_effectful_abstract_eval(pp.swap_p.abstract_eval)
lo.triton_lowering_rules[load_p] = clone(lo._masked_load_lowering_rule, _load=asm_load)
lo.triton_lowering_rules[swap_p] = clone(
    lo._masked_swap_lowering_rule, _load=asm_load, _store=asm_store
)
load_impl = clone(pp.load, load_p=load_p)
swap_impl = clone(pp.swap, swap_p=swap_p)
store_impl = clone(pp.store, swap=swap_impl)


def load(ref, **options):
    return load_impl(ref, None, **options)


def store(ref, value, **options):
    if jnp.ndim(value) == 0 and options.get("mask") is not None:
        value = jnp.broadcast_to(value, options["mask"].shape)
    return store_impl(ref, None, value, **options)
