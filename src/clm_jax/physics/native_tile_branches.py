
import jax.numpy as jnp
import numpy as np
from jax import lax

from clm_jax.physics import solver_scalarization as scalar


def guarded_condition(branches, index, arguments):
    predicate = index[()]
    if scalar._known(predicate):
        return scalar.evaluate(branches[int(np.clip(predicate, 0, len(branches) - 1))], arguments)
    shapes = tuple(value.shape for value in arguments)
    elements = scalar._flatten_storage(arguments)
    known = tuple(scalar._known(value) for value in elements)
    dynamic = tuple(value for value, keep in zip(elements, known, strict=True) if not keep)
    output_shapes = tuple(var.aval.shape for var in branches[0].jaxpr.outvars)

    def restore(values):
        values = iter(values)
        return scalar._restore_storage(
            shapes,
            tuple(
                value if keep else next(values) for value, keep in zip(elements, known, strict=True)
            ),
        )

    def evaluate(branch, values):
        return scalar._flatten_storage(scalar.evaluate(branch, restore(values)))

    if np.ndim(predicate) == 0:
        result = lax.switch(
            predicate, tuple(lambda *x, b=b: evaluate(b, x) for b in branches), *dynamic
        )
        return scalar._restore_storage(output_shapes, result)
    lane_shape = np.shape(predicate)
    zeros = tuple(
        jnp.zeros(lane_shape, var.aval.dtype)
        for var in branches[0].jaxpr.outvars
        for _ in np.ndindex(var.aval.shape)
    )
    values = []
    for position, branch in enumerate(branches):

        def enabled(*flat, branch=branch):
            return tuple(jnp.broadcast_to(x, lane_shape) for x in evaluate(branch, flat))

        result = lax.cond(jnp.any(predicate == position), enabled, lambda *flat: zeros, *dynamic)
        values.append(scalar._restore_storage(output_shapes, result))
    result = values[0]
    for position, branch in enumerate(values[1:], 1):
        result = [
            scalar._storage(
                a.shape,
                (
                    jnp.where(predicate == position, y, x)
                    for x, y in zip(a.flat, b.flat, strict=True)
                ),
            )
            for a, b in zip(result, branch, strict=True)
        ]
    return result
