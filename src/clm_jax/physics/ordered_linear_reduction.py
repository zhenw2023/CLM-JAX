import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax._src import core
from jax._src.interpreters import ad, batching, mlir


def _cpu(values, groups, *, count):
    from clm_jax.physics.ordered_reduction import _ordered_segment_sum_reference
    return _ordered_segment_sum_reference(values, groups, count)


def _gpu(values, groups, *, count):
    from clm_jax.physics.ordered_segment_sum_gpu import ordered_segment_sum_gpu
    return ordered_segment_sum_gpu(values, groups, count)


_sum = Primitive('clm_ordered_linear_group_sum')
_sum.def_impl(lambda values, groups, *, count:
              (_gpu if jax.default_backend() == 'gpu' else _cpu)(values, groups, count=count))


def _shape(values, groups, *, count):
    if count < 0 or groups.shape != (values.shape[0],):
        raise ValueError('ordered linear sum needs a valid group count and topology shape')
    return core.ShapedArray((count, *values.shape[1:]), values.dtype)


_sum.def_abstract_eval(_shape)
mlir.register_lowering(_sum, mlir.lower_fun(_cpu, multiple_results=False))
mlir.register_lowering(_sum, mlir.lower_fun(_gpu, multiple_results=False), platform='cuda')


def linear_ordered_segment_sum(values, groups, count):
    return _sum.bind(values, groups, count=count)


def _jvp(primals, tangents, *, count):
    values, groups = primals
    result = _sum.bind(values, groups, count=count)
    tangent = (ad.Zero.from_primal_value(result) if type(tangents[0]) is ad.Zero
               else _sum.bind(tangents[0], groups, count=count))
    return result, tangent


def _transpose(cotangent, values, groups, *, count):
    if not ad.is_undefined_primal(values):
        return None, None
    if type(cotangent) is ad.Zero:
        return ad.Zero(values.aval), None
    if count == 0:
        return jnp.zeros(values.aval.shape, values.aval.dtype), None
    return cotangent[groups], None


def _batch(values, axes, *, count):
    source, groups = values
    source_axis, group_axis = axes
    if group_axis is None:
        source = jnp.moveaxis(source, source_axis, -1)
        result = _sum.bind(source, groups, count=count)
        return result, result.ndim - 1
    size = groups.shape[group_axis]
    source = batching.bdim_at_front(source, source_axis, size)
    groups = batching.bdim_at_front(groups, group_axis, size)
    result = jax.lax.map(lambda pair: _sum.bind(*pair, count=count), (source, groups))
    return result, 0


ad.primitive_jvps[_sum] = _jvp
ad.primitive_transposes[_sum] = _transpose
batching.primitive_batchers[_sum] = _batch
