
from jax.extend import core
from jax.interpreters import batching, mlir

from clm_jax.physics.solver_fma import native_exp_residual


def _point(values):
    from clm_jax.physics.positive_power import _positive_power_reference
    token = native_exp_residual.set(True)
    try:
        return _positive_power_reference(*values)
    finally:
        native_exp_residual.reset(token)


def _implementation(base, exponent):
    from clm_jax.physics.positive_power import _positive_power_reference
    from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch
    if base.size < 1024:
        return _positive_power_reference(base, exponent)
    return fused_native_batch(
        _point, (base.reshape(-1), exponent.reshape(-1)), block_size=128,
        hardware_rounding=True, preserve_literal_barriers=True,
    ).reshape(base.shape)


_power_p = core.Primitive("clm_fused_positive_power")
_power_p.def_impl(_implementation)
_power_p.def_abstract_eval(lambda base, exponent: base)
mlir.register_lowering(_power_p, mlir.lower_fun(_implementation, multiple_results=False))


def _batch(values, axes):
    if all(axis is None for axis in axes):
        return _power_p.bind(*values), None
    size = next(value.shape[axis] for value, axis in zip(values, axes, strict=True)
                if axis is not None)
    return _power_p.bind(*(batching.bdim_at_front(value, axis, size)
                           for value, axis in zip(values, axes, strict=True))), 0


batching.primitive_batchers[_power_p] = _batch


def positive_power_gpu(base, exponent):
    return _power_p.bind(base, exponent)
