
from contextvars import ContextVar
from functools import lru_cache, reduce

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend import core

from clm_jax.floating_point import _native_tile_rounding as _native_tile_rounding

_native_tile_trace = ContextVar("clm_native_solver_tile_trace", default=False)
_fold_tile_literals = ContextVar("clm_native_solver_fold_literals", default=False)
_preserve_tile_literal_barriers = ContextVar("clm_native_literal_barriers", default=False)
_guard_tile_branches = ContextVar("clm_native_solver_guard_branches", default=False)
_share_nested_storage = ContextVar("clm_share_nested_solver_storage", default=True)


def _storage(shape, values):
    result = np.empty(shape, dtype=object)
    for index, value in enumerate(values):
        result.flat[index] = value
    return result


def _constant(value):
    value = np.asarray(value)
    return _storage(value.shape, value.flat)


def split_lanes(value, shape):
    return _storage(shape, (value[(slice(None), *index)] for index in np.ndindex(shape)))


def merge_lanes(value, count):
    elements = [jnp.broadcast_to(item, (count,)) for item in value.flat]
    if value.shape == ():
        return elements[0]
    return jnp.stack(elements, axis=-1).reshape((count, *value.shape))


_INTEGER_OPERATIONS = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply,
    "neg": np.negative, "lt": np.less, "le": np.less_equal,
    "gt": np.greater, "ge": np.greater_equal, "eq": np.equal,
    "ne": np.not_equal, "and": np.bitwise_and, "or": np.bitwise_or,
    "xor": np.bitwise_xor, "not": np.bitwise_not,
    "min": np.minimum, "max": np.maximum,
}

_LITERAL_OPERATIONS = {
    **_INTEGER_OPERATIONS, "div": np.divide, "abs": np.abs,
    "sign": np.sign, "floor": np.floor, "ceil": np.ceil,
    "is_finite": np.isfinite, "shift_left": np.left_shift,
    "shift_right_arithmetic": np.right_shift, "shift_right_logical": np.right_shift,
}


def _known(value):
    return isinstance(value, (bool, int, float, complex, np.generic))


def _elementwise(primitive, arguments, parameters, output_shape):
    values = (arguments if all(value.shape == arguments[0].shape for value in arguments)
              else np.broadcast_arrays(*arguments))
    output = []
    for inputs in zip(*(value.flat for value in values), strict=True):
        if all(_known(value) for value in inputs):
            if _fold_tile_literals.get():
                if primitive.name in _LITERAL_OPERATIONS:
                    with np.errstate(all="ignore"):
                        output.append(_LITERAL_OPERATIONS[primitive.name](*inputs))
                    continue
                if primitive.name == "bitcast_convert_type":
                    output.append(np.asarray(inputs[0]).view(parameters["new_dtype"])[()])
                    continue
            if primitive.name == "convert_element_type":
                output.append(np.asarray(inputs[0], dtype=parameters["new_dtype"])[()])
                continue
            if primitive.name == "select_n":
                output.append(inputs[1 + int(inputs[0])])
                continue
            if (primitive.name in _INTEGER_OPERATIONS
                    and all(np.asarray(value).dtype.kind in "biu" for value in inputs)):
                output.append(_INTEGER_OPERATIONS[primitive.name](*inputs))
                continue
        if (_fold_tile_literals.get() and primitive is lax.div_p
                and _known(inputs[1]) and not _known(inputs[0])
                and np.asarray(inputs[1]).dtype == np.dtype("float64")
                and np.isfinite(inputs[1]) and inputs[1] != 0):
            output.append(lax.mul(inputs[0], jnp.float64(np.float64(1) / inputs[1])))
            continue
        if len({np.shape(value) for value in inputs}) > 1:
            inputs = jnp.broadcast_arrays(*inputs)
        output.append(primitive.bind(*inputs, **parameters))
    return _storage(output_shape, output)


def _static_indices(indices):
    if not all(_known(value) for value in indices.flat):
        raise NotImplementedError("native tile solver requires static storage indices")
    return np.asarray(list(indices.flat), dtype=np.int64).reshape(indices.shape)


def _dynamic_slice(arguments, parameters):
    operand, *starts = arguments
    if operand.shape == parameters["slice_sizes"]:
        return operand
    if all(_known(start[()]) for start in starts):
        bounds = [
            min(max(int(start[()]), 0), extent - size)
            for start, extent, size in zip(
                starts, operand.shape, parameters["slice_sizes"], strict=True,
            )
        ]
        return operand[tuple(slice(start, start + size) for start, size in
                             zip(bounds, parameters["slice_sizes"], strict=True))]
    if (operand.ndim == 2 and parameters["slice_sizes"] == (1, operand.shape[1])
            and all(_known(value) for value in operand.flat)):
        from clm_jax.physics.solver_constant_lookup import constant_lookup
        table = np.asarray(list(operand.flat)).reshape(operand.shape)
        return _storage((1, operand.shape[1]), (
            constant_lookup(table[:, column], starts[0][()])
            for column in range(operand.shape[1])
        ))
    integer_sequence = all(_known(value) and np.asarray(value).dtype.kind in "iu"
                           for value in operand.flat)
    if operand.ndim != 1 or parameters["slice_sizes"] != (1,):
        raise NotImplementedError("native tile requires a single bounded storage element")
    index = jnp.clip(starts[0][()], 0, operand.size - 1)
    if integer_sequence:
        values = np.asarray(list(operand.flat))
        step = np.diff(values)
        if not step.size or np.all(step == step[0]):
            value = values[0] + index * (step[0] if step.size else np.asarray(0, values.dtype))
            return _storage((1,), (value,))
    if all(_known(value) for value in operand.flat):
        from clm_jax.physics.solver_constant_lookup import constant_lookup
        return _storage((1,), (constant_lookup(np.asarray(list(operand.flat)), index),))
    if operand.size > 64:
        raise NotImplementedError("native tile dynamic storage selection exceeds 64 elements")
    value = operand[0]
    for position in range(1, operand.size):
        value = jnp.where(index == position, operand[position], value)
    return _storage((1,), (value,))


def _pad(operand, padding_value, configuration):
    shape = tuple(
        low + high + size + max(size - 1, 0) * interior
        for size, (low, high, interior) in zip(operand.shape, configuration, strict=True)
    )
    result = _storage(shape, (padding_value[()] for _ in range(int(np.prod(shape)))))
    for index in np.ndindex(operand.shape):
        target = tuple(low + position * (interior + 1) for position, (low, _, interior)
                       in zip(index, configuration, strict=True))
        if all(0 <= value < size for value, size in zip(target, shape, strict=True)):
            result[target] = operand[index]
    return result


def _gather(operand, indices, parameters):
    dims = parameters["dimension_numbers"]
    collapsed_row = (
        indices.shape[-1] == 1 and dims.start_index_map == (0,)
        and dims.collapsed_slice_dims == (0,)
        and dims.offset_dims == (indices.ndim - 1,)
    )
    full_row = (
        indices.shape[-1] == 2 and dims.start_index_map == (0, 1)
        and dims.collapsed_slice_dims == ()
        and dims.offset_dims == (indices.ndim - 1, indices.ndim)
        and all(_known(value) and value == 0 for value in indices[..., 1].flat)
    )
    if (operand.ndim == 2 and (collapsed_row or full_row)
            and not dims.operand_batching_dims and not dims.start_indices_batching_dims
            and parameters["slice_sizes"] == (1, operand.shape[1])
            and parameters["mode"] in (lax.GatherScatterMode.PROMISE_IN_BOUNDS,
                                       lax.GatherScatterMode.CLIP)
            and all(_known(value) for value in operand.flat)):
        from clm_jax.physics.solver_constant_lookup import constant_lookup
        window = (operand.shape[1],) if collapsed_row else (1, operand.shape[1])
        shape = (*indices.shape[:-1], *window)
        table = np.asarray(list(operand.flat)).reshape(operand.shape)
        return _storage(shape, (
            constant_lookup(table[:, column], indices[(*row, 0)])
            for row in np.ndindex(indices.shape[:-1]) for column in range(operand.shape[1])
        ))
    if parameters["mode"] not in (lax.GatherScatterMode.CLIP,
                                   lax.GatherScatterMode.PROMISE_IN_BOUNDS,
                                   lax.GatherScatterMode.FILL_OR_DROP):
        raise NotImplementedError("unsupported native solver gather mode")
    windows = [axis for axis in range(operand.ndim)
               if axis not in (*dims.collapsed_slice_dims, *dims.operand_batching_dims)]
    rank = len(dims.offset_dims) + indices.ndim - 1
    batch_axes = [axis for axis in range(rank) if axis not in dims.offset_dims]
    shape = [0] * rank
    for axis, extent in zip(batch_axes, indices.shape[:-1], strict=True):
        shape[axis] = extent
    for axis, window in zip(dims.offset_dims, windows, strict=True):
        shape[axis] = parameters["slice_sizes"][window]
    output = np.empty(shape, dtype=object)
    for position in np.ndindex(*shape):
        batch = tuple(position[axis] for axis in batch_axes)
        starts = dict(zip(dims.start_index_map, indices[batch], strict=True))
        starts.update((operand_axis, batch[index_axis]) for operand_axis, index_axis
                      in zip(dims.operand_batching_dims, dims.start_indices_batching_dims,
                             strict=True))
        valid = True
        point = []
        for axis, (extent, size) in enumerate(zip(
            operand.shape, parameters["slice_sizes"], strict=True,
        )):
            start = starts.get(axis, 0)
            valid = valid & (start >= 0) & (start <= extent - size)
            point.append(min(max(int(start), 0), extent - size) if _known(start)
                         else jnp.clip(start, 0, extent - size))
        for output_axis, operand_axis in zip(dims.offset_dims, windows, strict=True):
            point[operand_axis] = point[operand_axis] + position[output_axis]
        dynamic_axes = [axis for axis, value in enumerate(point) if not _known(value)]
        if not dynamic_axes:
            result = operand[tuple(point)]
        else:
            dynamic_shape = tuple(operand.shape[axis] for axis in dynamic_axes)
            if np.prod(dynamic_shape) > 64:
                raise NotImplementedError("dynamic solver storage gather exceeds 64 cells")
            cell = list(point)
            for axis in dynamic_axes:
                cell[axis] = 0
            result = operand[tuple(cell)]
            for coordinates in np.ndindex(dynamic_shape):
                for axis, index in zip(dynamic_axes, coordinates, strict=True):
                    cell[axis] = index
                mask = reduce(jnp.logical_and, (point[axis] == index for axis, index
                                               in zip(dynamic_axes, coordinates, strict=True)))
                result = jnp.where(mask, operand[tuple(cell)], result)
        if parameters["mode"] == lax.GatherScatterMode.FILL_OR_DROP:
            result = jnp.where(valid, result, parameters["fill_value"])
        output[position] = result
    return output


def _scatter(operand, indices, updates, parameters):
    dims = parameters["dimension_numbers"]
    if (parameters["update_jaxpr"] is not None or dims.operand_batching_dims
            or dims.scatter_indices_batching_dims
            or len(dims.scatter_dims_to_operand_dims) != indices.shape[-1]
            or dims.update_window_dims != tuple(range(updates.ndim))
            or indices.ndim != 1 or any(not _known(value) for value in indices.flat)):
        return _scatter_windows(operand, indices, updates, parameters)
    indices = _static_indices(indices)
    if indices.ndim != 1:
        raise NotImplementedError("native solver scatter requires one static window")
    starts = dict(zip(dims.scatter_dims_to_operand_dims, indices, strict=True))
    index = []
    window = iter(updates.shape)
    for axis in range(operand.ndim):
        start = int(starts.get(axis, 0))
        if axis in dims.inserted_window_dims:
            if not 0 <= start < operand.shape[axis]:
                raise NotImplementedError("native solver requires in-bounds static scatter")
            index.append(start)
        else:
            end = start + next(window)
            if start < 0 or end > operand.shape[axis]:
                raise NotImplementedError("native solver requires in-bounds static scatter")
            index.append(slice(start, end))
    result = operand.copy()
    result[tuple(index)] = updates[()] if updates.shape == () else updates
    return result


def _scatter_windows(operand, indices, updates, parameters):
    dims = parameters["dimension_numbers"]
    if (parameters["update_jaxpr"] is not None
            or parameters["mode"] not in (
                lax.GatherScatterMode.PROMISE_IN_BOUNDS,
                lax.GatherScatterMode.FILL_OR_DROP,
            )
            or (not parameters["unique_indices"] and np.prod(indices.shape[:-1]) > 1)):
        raise NotImplementedError("native solver requires disjoint replacement windows")
    batch_axes = [axis for axis in range(updates.ndim)
                  if axis not in dims.update_window_dims]
    windows = [axis for axis in range(operand.ndim)
               if axis not in (*dims.inserted_window_dims, *dims.operand_batching_dims)]
    result = operand.copy()
    for position in np.ndindex(updates.shape):
        batch = tuple(position[axis] for axis in batch_axes)
        starts = dict(zip(dims.scatter_dims_to_operand_dims, indices[batch], strict=True))
        starts.update((axis, batch[index_axis]) for axis, index_axis in zip(
            dims.operand_batching_dims, dims.scatter_indices_batching_dims, strict=True))
        point = [starts.get(axis, 0) for axis in range(operand.ndim)]
        for update_axis, operand_axis in zip(dims.update_window_dims, windows, strict=True):
            point[operand_axis] = point[operand_axis] + position[update_axis]
        dynamic = [axis for axis, value in enumerate(point) if not _known(value)]
        if any(not 0 <= value < operand.shape[axis] for axis, value in enumerate(point)
               if axis not in dynamic):
            continue
        if not dynamic:
            result[tuple(point)] = updates[position]
            continue
        extents = tuple(operand.shape[axis] for axis in dynamic)
        if np.prod(extents) > 64:
            raise NotImplementedError("dynamic solver storage scatter exceeds 64 cells")
        target = list(point)
        for coordinates in np.ndindex(extents):
            valid = True
            for axis, index in zip(dynamic, coordinates, strict=True):
                target[axis] = index
                valid = valid & (point[axis] == index)
            target_index = tuple(target)
            result[target_index] = jnp.where(valid, updates[position], result[target_index])
    return result


def _flatten_storage(arguments):
    return tuple(item for value in arguments for item in value.flat)


def _reduce_storage(operand, axes, operation):
    if not axes:
        return operand
    remaining = tuple(axis for axis in range(operand.ndim) if axis not in axes)
    view = operand.transpose((*remaining, *axes))
    output_shape = tuple(operand.shape[axis] for axis in remaining)
    if any(operand.shape[axis] == 0 for axis in axes):
        raise NotImplementedError("native solver requires nonempty reduced storage")
    return _storage(output_shape, (
        reduce(operation, view[index].flat) for index in np.ndindex(output_shape)
    ))


def _argreduce_storage(operand, axes, index_dtype, *, maximum):
    if len(axes) != 1 or not 0 < operand.shape[axes[0]] <= 64:
        raise NotImplementedError("native solver arg reduction requires one bounded storage axis")
    axis, = axes
    remaining = tuple(i for i in range(operand.ndim) if i != axis)
    view = operand.transpose((*remaining, axis))
    output_shape = tuple(operand.shape[i] for i in remaining)
    outputs = []
    for position in np.ndindex(output_shape):
        values = view[position]
        best = values[0]
        selected = jnp.zeros_like(best, dtype=index_dtype)
        for index, value in enumerate(values[1:], 1):
            better = value > best if maximum else value < best
            take = ~jnp.isnan(best) & (jnp.isnan(value) | better)
            selected = jnp.where(take, jnp.asarray(index, index_dtype), selected)
            best = jnp.where(take, value, best)
        outputs.append(selected)
    return _storage(output_shape, outputs)


def _restore_storage(shapes, values):
    position = 0
    arguments = []
    for shape in shapes:
        size = int(np.prod(shape, dtype=int))
        arguments.append(_storage(shape, values[position:position + size]))
        position += size
    if position != len(values):
        raise ValueError("native tile storage arity mismatch")
    return arguments


@lru_cache(maxsize=512)
def _shared_storage_program(closed, shapes, signatures, literals, options):
    fixed = {index: np.frombuffer(data, dtype=np.dtype(dtype))[0]
             for index, dtype, data in literals}
    count = sum(int(np.prod(shape)) for shape in shapes)
    constant_outputs = {}

    def transformed(*values):
        values = iter(values)
        arguments = _restore_storage(shapes, tuple(
            fixed[index] if index in fixed else next(values) for index in range(count)
        ))
        settings = (_fold_tile_literals, _preserve_tile_literal_barriers,
                    _guard_tile_branches, _share_nested_storage)
        tokens = [setting.set(value) for setting, value in zip(settings, options, strict=True)]
        try:
            output = _flatten_storage(evaluate(closed, arguments))
            constant_outputs.update({index: value for index, value in enumerate(output)
                                     if _known(value)})
            return output
        finally:
            for setting, token in reversed(tuple(zip(settings, tokens, strict=True))):
                setting.reset(token)

    program = jax.make_jaxpr(transformed)(*(
        jax.ShapeDtypeStruct(shape, dtype, weak_type=weak)
        for shape, dtype, weak in signatures
    ))
    execute = jax.jit(lambda *values: jax.core.eval_jaxpr(program.jaxpr, program.consts, *values))
    return execute, constant_outputs


def _shared_storage_call(closed, arguments):
    elements = _flatten_storage(arguments)
    literals, dynamic = [], []
    for index, value in enumerate(elements):
        if _known(value):
            number = np.asarray(value)
            literals.append((index, number.dtype.str, number.tobytes()))
        else:
            dynamic.append(value)
    if not dynamic:
        return evaluate(closed, arguments)
    signatures = tuple((np.shape(value), value.dtype, jax.typeof(value).weak_type)
                       for value in dynamic)
    options = (_fold_tile_literals.get(), _preserve_tile_literal_barriers.get(),
               _guard_tile_branches.get(), _share_nested_storage.get())
    execute, constants = _shared_storage_program(
        closed, tuple(value.shape for value in arguments), signatures, tuple(literals), options,
    )
    output = execute(*dynamic)
    output = tuple(constants.get(index, value) for index, value in enumerate(output))
    program = closed.jaxpr if isinstance(closed, core.ClosedJaxpr) else closed
    return _restore_storage(tuple(variable.aval.shape for variable in program.outvars), output)


def _condition(branches, index, arguments):
    if _guard_tile_branches.get():
        from clm_jax.physics.native_tile_branches import guarded_condition

        return guarded_condition(branches, index, arguments)
    predicate = index[()]
    if np.ndim(predicate) == 0:
        shapes = tuple(value.shape for value in arguments)
        outputs = lax.switch(
            predicate,
            tuple(lambda *items, branch=branch: _flatten_storage(
                evaluate(branch, _restore_storage(shapes, items)),
            ) for branch in branches),
            *_flatten_storage(arguments),
        )
        output_shapes = tuple(value.aval.shape for value in branches[0].jaxpr.outvars)
        return _restore_storage(output_shapes, outputs)
    if len(branches) != 2:
        raise NotImplementedError("native solver supports only binary batched branches")
    first, second = (evaluate(branch, arguments) for branch in branches)
    return [_storage(left.shape, (
        jnp.where(predicate != 0, b, a) for a, b in zip(left.flat, right.flat, strict=True)
    )) for left, right in zip(first, second, strict=True)]


def _while(arguments, parameters):
    ncond = parameters["cond_nconsts"]
    nbody = parameters["body_nconsts"]
    cond_constants = arguments[:ncond]
    body_constants = arguments[ncond:ncond + nbody]
    initial = arguments[ncond + nbody:]
    shapes = tuple(value.shape for value in initial)

    def active_lanes(flat):
        result, = evaluate(parameters["cond_jaxpr"], [
            *cond_constants, *_restore_storage(shapes, flat),
        ])
        return result[()]

    def condition(flat):
        return jnp.any(active_lanes(flat))

    def body(flat):
        active = active_lanes(flat)
        updated = _flatten_storage(evaluate(parameters["body_jaxpr"], [
            *body_constants, *_restore_storage(shapes, flat),
        ]))
        return tuple(jnp.where(active, after, before)
                     for before, after in zip(flat, updated, strict=True))

    flattened = _flatten_storage(initial)
    lane_shape = np.broadcast_shapes(*(
        np.shape(value) for value in _flatten_storage(arguments)
    ))
    flattened = tuple(jnp.broadcast_to(value, lane_shape) for value in flattened)
    result = lax.while_loop(condition, body, flattened)
    return _restore_storage(shapes, result)


def _scan(arguments, parameters):
    count = parameters["length"]
    if count > 64:
        raise NotImplementedError("native tile scan exceeds bounded storage extent")
    nconst, ncarry = parameters["num_consts"], parameters["num_carry"]
    constants = arguments[:nconst]
    carry = arguments[nconst:nconst + ncarry]
    sequences = arguments[nconst + ncarry:]
    program = parameters["jaxpr"]
    output_vars = program.jaxpr.outvars[ncarry:]
    histories = [[None] * count for _ in output_vars]
    order = reversed(range(count)) if parameters["reverse"] else range(count)
    for index in order:
        rows = (value[index:index+1].reshape(value.shape[1:]) for value in sequences)
        outputs = evaluate(program, [*constants, *carry, *rows])
        carry = outputs[:ncarry]
        for history, value in zip(histories, outputs[ncarry:], strict=True):
            history[index] = value
    return [*carry, *(
        np.stack(history) if count else np.empty((0, *variable.aval.shape), dtype=object)
        for history, variable in zip(histories, output_vars, strict=True)
    )]


def evaluate(closed, arguments):
    if isinstance(closed, core.ClosedJaxpr):
        program, constants = closed.jaxpr, closed.consts
    else:
        program, constants = closed, ()
    environment = dict(zip(program.invars, arguments, strict=True))
    environment.update(zip(program.constvars, map(_constant, constants), strict=True))

    def read(variable):
        if isinstance(variable, core.Literal):
            return _constant(variable.val)
        return environment[variable]

    for equation in program.eqns:
        primitive, parameters = equation.primitive, equation.params
        inputs = [read(variable) for variable in equation.invars]
        name = primitive.name
        if name in ("jit", "custom_jvp_call", "closed_call", "call"):
            nested = parameters.get("jaxpr", parameters.get("call_jaxpr"))
            outputs = (_shared_storage_call(nested, inputs) if _share_nested_storage.get()
                       else evaluate(nested, inputs))
        elif name in ("optimization_barrier", "stop_gradient"):
            operation = (lax.optimization_barrier if name == "optimization_barrier"
                         else lax.stop_gradient)
            outputs = [_storage(value.shape, (
                item if (_fold_tile_literals.get() and _known(item)
                         and not (name == "optimization_barrier"
                                  and _preserve_tile_literal_barriers.get()
                                  and np.asarray(item).dtype.kind in "fc"))
                else operation(item)
                for item in value.flat
            )) for value in inputs]
        elif name == "cond":
            outputs = _condition(parameters["branches"], inputs[0], inputs[1:])
        elif name == "while":
            outputs = _while(inputs, parameters)
        elif name == "scan":
            outputs = _scan(inputs, parameters)
        elif name == "pmax":
            from clm_jax.physics.converged_iteration import SPATIAL_SOLVE_AXIS
            axes = parameters["axes"]
            if parameters["axis_index_groups"] is not None:
                raise NotImplementedError(
                    f"native tile supports only its independent spatial collective: {parameters}"
                )
            if all(isinstance(axis, int) for axis in axes):
                outputs = [_reduce_storage(inputs[0], axes, jnp.maximum)]
            elif axes == (SPATIAL_SOLVE_AXIS,):
                outputs = [_storage(inputs[0].shape, (
                    jnp.max(value) for value in inputs[0].flat
                ))]
            else:
                raise NotImplementedError(f"unsupported native solver collective: {parameters}")
        elif name == "broadcast_in_dim":
            shape = [1] * len(parameters["shape"])
            for axis, size in zip(parameters["broadcast_dimensions"], inputs[0].shape, strict=True):
                shape[axis] = size
            outputs = [np.broadcast_to(inputs[0].reshape(shape), parameters["shape"])]
        elif name == "slice":
            strides = parameters["strides"] or (1,) * inputs[0].ndim
            outputs = [inputs[0][tuple(slice(a, b, c) for a, b, c in zip(
                parameters["start_indices"], parameters["limit_indices"], strides, strict=True,
            ))]]
        elif name == "dynamic_slice":
            outputs = [_dynamic_slice(inputs, parameters)]
        elif name == "pad":
            outputs = [_pad(*inputs, parameters["padding_config"])]
        elif name == "squeeze":
            outputs = [np.squeeze(inputs[0], parameters["dimensions"])]
        elif name == "reshape":
            value = inputs[0]
            if parameters["dimensions"] is not None:
                value = value.transpose(parameters["dimensions"])
            outputs = [value.reshape(parameters["new_sizes"])]
        elif name == "transpose":
            outputs = [inputs[0].transpose(parameters["permutation"])]
        elif name == "rev":
            outputs = [np.flip(inputs[0], axis=parameters["dimensions"])]
        elif name == "concatenate":
            outputs = [np.concatenate(inputs, axis=parameters["dimension"])]
        elif name == "stack":
            outputs = [np.stack(inputs, axis=parameters["axis"])]
        elif name == "iota":
            shape = parameters["shape"]
            dimension = parameters["dimension"]
            index_shape = [1] * len(shape)
            index_shape[dimension] = shape[dimension]
            outputs = [_constant(np.broadcast_to(np.arange(
                shape[dimension], dtype=parameters["dtype"],
            ).reshape(index_shape), shape))]
        elif name == "gather":
            outputs = [_gather(*inputs, parameters)]
        elif name == "scatter":
            outputs = [_scatter(*inputs, parameters)]
        elif name in ("argmax", "argmin"):
            outputs = [_argreduce_storage(
                inputs[0], parameters["axes"], parameters["index_dtype"],
                maximum=name == "argmax",
            )]
        elif name.startswith("reduce_"):
            operations = {"reduce_or": jnp.logical_or, "reduce_and": jnp.logical_and,
                          "reduce_max": jnp.maximum, "reduce_min": jnp.minimum}
            if name == "reduce_sum" and equation.outvars[0].aval.dtype.kind in "iu":
                operations[name] = jnp.add
            if name not in operations:
                raise NotImplementedError(f"unsupported native solver reduction {name}")
            outputs = [_reduce_storage(inputs[0], parameters["axes"], operations[name])]
        elif primitive.multiple_results:
            raise NotImplementedError(f"unsupported multi-output native primitive {name}")
        else:
            outputs = [_elementwise(primitive, inputs, parameters, equation.outvars[0].aval.shape)]
        if len(outputs) != len(equation.outvars):
            raise ValueError(f"native solver primitive {name} returned wrong arity")
        environment.update(zip(equation.outvars, outputs, strict=True))
    return [read(variable) for variable in program.outvars]
