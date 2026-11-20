
import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax._src.interpreters import ad, batching, mlir


def _implementation(coefficient, sensitivity):
    absent = ((jnp.isinf(coefficient) & (sensitivity == 0))
              | ((coefficient == 0) & jnp.isinf(sensitivity)))
    return jnp.where(absent, jnp.zeros_like(coefficient), coefficient * sensitivity)


_scale_p = Primitive("clm_sensitivity_scale")
_scale_p.def_impl(_implementation)
_scale_p.def_abstract_eval(lambda coefficient, sensitivity: coefficient)


def _scale_lowering(ctx, coefficient, sensitivity):
    import numpy as np
    from jax._src.lib.mlir.dialects import hlo
    aval=ctx.avals_in[0]
    if aval.dtype not in (np.dtype('float32'),np.dtype('float64')):
        return mlir.lower_fun(_implementation,multiple_results=False)(ctx,coefficient,sensitivity)
    zero=mlir.full_like_aval(ctx,0.,aval)
    infinity=mlir.full_like_aval(ctx,np.inf,aval)
    direction=hlo.ComparisonDirectionAttr.get('EQ')
    comparison=hlo.ComparisonTypeAttr.get('FLOAT')
    equal=lambda left,right:hlo.compare(left,right,direction,compare_type=comparison)
    absent=hlo.or_(hlo.and_(equal(hlo.abs(coefficient),infinity),equal(sensitivity,zero)),
                   hlo.and_(equal(coefficient,zero),equal(hlo.abs(sensitivity),infinity)))
    return [hlo.select(absent,zero,hlo.multiply(coefficient,sensitivity))]


mlir.register_lowering(_scale_p, _scale_lowering)


def sensitivity_product(coefficient, sensitivity):
    dtype = jnp.result_type(coefficient, sensitivity)
    coefficient, sensitivity = jnp.asarray(coefficient, dtype), jnp.asarray(sensitivity, dtype)
    if coefficient.shape != sensitivity.shape:
        coefficient, sensitivity = jnp.broadcast_arrays(coefficient, sensitivity)
    return _scale_p.bind(coefficient, sensitivity)


def _jvp(primals, tangents):
    result = _scale_p.bind(*primals)
    terms = ([] if isinstance(tangents[0], ad.Zero)
             else [sensitivity_product(tangents[0], primals[1])])
    if not isinstance(tangents[1], ad.Zero):
        terms.append(sensitivity_product(primals[0], tangents[1]))
    tangent = (terms[0] if len(terms) == 1 else terms[0] + terms[1]) if terms else (
        ad.Zero.from_primal_value(result))
    return result, tangent


def _transpose(cotangent, coefficient, sensitivity):
    if ad.is_undefined_primal(coefficient):
        if ad.is_undefined_primal(sensitivity):
            raise TypeError("sensitivity scaling is linear in one argument at a time")
        value = (ad.Zero(coefficient.aval) if isinstance(cotangent, ad.Zero)
                 else sensitivity_product(cotangent, sensitivity))
        return value, None
    if ad.is_undefined_primal(sensitivity):
        value = (ad.Zero(sensitivity.aval) if isinstance(cotangent, ad.Zero)
                 else sensitivity_product(coefficient, cotangent))
        return None, value
    return None, None


def _batch(values, axes):
    size = next(value.shape[axis] for value, axis in zip(values, axes, strict=True)
                if axis is not None)
    values = tuple(batching.bdim_at_front(value, axis, size)
                   for value, axis in zip(values, axes, strict=True))
    return _scale_p.bind(*values), 0


ad.primitive_jvps[_scale_p] = _jvp
ad.primitive_transposes[_scale_p] = _transpose
batching.primitive_batchers[_scale_p] = _batch


def _sensitivity_sum_implementation(*terms):
    barrier = jax.lax.optimization_barrier
    partials = []
    valid = jnp.ones_like(terms[0], dtype=jnp.bool_)
    ordinary = terms[0]
    for index, term in enumerate(terms):
        valid = valid & jnp.isfinite(term)
        if index:
            ordinary = barrier(ordinary + term)
        carry, next_partials = term, []
        for previous in partials:
            total = barrier(carry + previous)
            virtual = barrier(total - carry)
            error = barrier(barrier(carry - barrier(total - virtual))
                            + barrier(previous - virtual))
            valid = valid & jnp.isfinite(total)
            next_partials.append(error)
            carry = total
        partials = [*next_partials, carry]
    result = partials[-1]
    for error in reversed(partials[:-1]):
        result = barrier(result + error)
    return jnp.where(valid, result, ordinary)


_sum_p = Primitive("clm_sensitivity_sum")
_sum_p.def_impl(_sensitivity_sum_implementation)
_sum_p.def_abstract_eval(lambda *terms: terms[0])
mlir.register_lowering(_sum_p, mlir.lower_fun(_sensitivity_sum_implementation,
                                            multiple_results=False))


def _sensitivity_sum_gpu(*terms):
    from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch
    if terms[0].size == 0:
        return jnp.empty_like(terms[0])
    shape = terms[0].shape
    flattened = tuple(term.reshape(-1) for term in terms)
    result = fused_native_batch(_sensitivity_sum_implementation, *flattened,
                                block_size=128, direct_io=True)
    return result.reshape(shape)


def _sensitivity_sum_gpu_lowering(ctx, *terms):
    implementation = (_sensitivity_sum_gpu if ctx.avals_in[0].dtype == jnp.float64
                      else _sensitivity_sum_implementation)
    return mlir.lower_fun(implementation, multiple_results=False)(ctx, *terms)


mlir.register_lowering(_sum_p, _sensitivity_sum_gpu_lowering, platform='cuda')


def sensitivity_sum(*terms):
    if not 2 <= len(terms) <= 4:
        raise ValueError("sensitivity addition requires two to four terms")
    dtype = jnp.result_type(*terms)
    values = tuple(jnp.asarray(term, dtype) for term in terms)
    if any(value.shape != values[0].shape for value in values[1:]):
        values = jnp.broadcast_arrays(*values)
    return _sum_p.bind(*values)


def _sum_transpose(cotangent, *terms):
    return tuple(cotangent if ad.is_undefined_primal(term) else None for term in terms)


def _sum_batch(values, axes):
    size = next(value.shape[axis] for value, axis in zip(values, axes, strict=True)
                if axis is not None)
    values = tuple(batching.bdim_at_front(value, axis, size)
                   for value, axis in zip(values, axes, strict=True))
    return _sum_p.bind(*values), 0


ad.deflinear2(_sum_p, _sum_transpose)
batching.primitive_batchers[_sum_p] = _sum_batch
