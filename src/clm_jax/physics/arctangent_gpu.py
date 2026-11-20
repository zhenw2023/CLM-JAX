
from jax.extend import core
from jax.interpreters import batching, mlir

from clm_jax.physics.solver_fma import native_exp_residual


def _point(value):
    from clm_jax.physics.arctangent import _arctangent_reference
    token = native_exp_residual.set(True)
    try:
        return _arctangent_reference(value)
    finally:
        native_exp_residual.reset(token)


def _implementation(value):
    from clm_jax.physics.arctangent import _arctangent_reference
    from clm_jax.physics.plant_hydraulics_gpu import fused_native_batch
    if value.size < 1024:
        return _arctangent_reference(value)
    return fused_native_batch(
        _point, value.reshape(-1), block_size=128, hardware_rounding=True,
        preserve_literal_barriers=True,
    ).reshape(value.shape)


_atan_p = core.Primitive("clm_fused_arctangent")
_atan_p.def_impl(_implementation)
_atan_p.def_abstract_eval(lambda value: value)
mlir.register_lowering(_atan_p, mlir.lower_fun(_implementation, multiple_results=False))
batching.primitive_batchers[_atan_p] = (
    lambda values, axes: (_atan_p.bind(values[0]), axes[0])
)


def arctangent_gpu(value):
    return _atan_p.bind(value)
