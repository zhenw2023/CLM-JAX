
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core
from jax._src.ad_util import Zero
from jax._src.interpreters import ad
from jax._src.lax.lax import _convert_element_type

from clm_jax.adjoint_diagnostics import (
    _cotangent_leaf_flags as _cotangent_leaf_flags,
)
from clm_jax.adjoint_diagnostics import (
    diagnose_stage_pullbacks,
    exact_zero_cotangents,
)
from clm_jax.adjoint_replay import stage_input_offload_scope
from clm_jax.compilation import staged_reverse_ad
from clm_jax.reverse_checkpoint import ReverseCheckpoint, tree_fingerprint
from clm_jax.runtime import (
    array_values_to_host,
    bounded_device_transfers,
    co_locate_array_leaves,
    retained_host_memory,
)


def _cotangent_projection(cotangents):
    leaves, structure = jax.tree.flatten(
        exact_zero_cotangents(cotangents), is_leaf=lambda value: isinstance(value, Zero)
    )
    selected = tuple(i for i, value in enumerate(leaves) if not isinstance(value, Zero))
    return structure, selected, tuple(leaves[i] for i in selected)


def _projected_transition(advance, objective, structure, selected):
    def step(point):
        complete = advance(point)
        leaves, actual = jax.tree.flatten(complete)
        if actual != structure:
            raise ValueError("reverse state structure differs from its cotangent")
        return tuple(leaves[i] for i in selected), objective(complete)

    return step


def _objective_initial_adjoint(objective, state):
    value, pullback = jax.vjp(objective, state)
    aval = jax.typeof(value)
    if aval.shape or np.dtype(aval.dtype).kind != "f":
        raise TypeError("trajectory objective must be a real floating-point scalar")
    leaves, structure = jax.tree.flatten(state)
    variables = pullback.jaxpr.invars
    if len(variables) != len(leaves):
        raise RuntimeError("JAX objective VJP inputs differ from the complete state")
    read = {
        var for eqn in pullback.jaxpr.eqns for var in eqn.invars if isinstance(var, core.Var)
    }
    read.update(var for var in pullback.jaxpr.outvars if isinstance(var, core.Var))
    active = tuple(var in read for var in variables)
    accumulators = [
        (ad.ValAccum if used else ad.NullAccum)(var.aval.to_ct_aval())
        for var, used in zip(variables, active, strict=True)
    ]
    pullback.with_refs(structure.unflatten(accumulators))(jnp.ones_like(value))
    return value, structure.unflatten([
        accumulator.freeze() if used else Zero(var.aval.to_ct_aval())
        for accumulator, var, used in zip(accumulators, variables, active, strict=True)
    ])


def _checkpoint_bytes(checkpoints):
    owners = {
        id(value): getattr(value, "nbytes", 0)
        for saved in checkpoints.values()
        for value in saved.values
    }
    return sum(owners.values())


def _replay_checkpoint_splits(distance, slots):
    previous = [n * (n + 1) // 2 for n in range(distance + 1)]
    splits = [None]
    for _ in range(slots):
        current, selected = [0] * (distance + 1), [None] * (distance + 1)
        if distance:
            current[1] = 1
        for n in range(2, distance + 1):
            current[n], selected[n] = min(
                (k + previous[n - k] + current[k - 1], k) for k in range(1, n)
            )
        splits.append(selected)
        previous = current
    return splits


@dataclass
class HostCheckpoint:
    values: tuple[Any, ...]
    slots: tuple[int, ...]
    structure: Any
    weak_types: tuple[bool | None, ...] | None = None

    @classmethod
    def capture(cls, state):
        leaves, structure = jax.tree.flatten(state)
        lookup, values, slots = {}, [], []
        for leaf in leaves:
            identity = id(leaf)
            if identity not in lookup:
                lookup[identity] = len(values)
                values.append(leaf)
            slots.append(lookup[identity])
        weak_types = tuple(value.weak_type if isinstance(value, jax.Array) else None
                           for value in values)
        with bounded_device_transfers():
            values = array_values_to_host(values)
        return cls(tuple(values), tuple(slots), structure, weak_types)

    def restore_host(self):
        return self.structure.unflatten([self.values[slot] for slot in self.slots])

    def restore(self):
        with bounded_device_transfers():
            values = co_locate_array_leaves(self.values, jax.devices()[0])
        if self.weak_types is not None:
            values = tuple(
                _convert_element_type(value, value.dtype, weak_type=weak)
                if weak is not None and value.weak_type != weak else value
                for value, weak in zip(values, self.weak_types, strict=True)
            )
        return self.structure.unflatten([values[slot] for slot in self.slots])

    @property
    def nbytes(self):
        return sum(getattr(value, "nbytes", 0) for value in self.values)


def offload_pullback_residuals(pullback, *, minimum_bytes=1 << 20):
    if jax.default_backend() != "gpu":
        return pullback, 0
    placed, saved_bytes = {}, 0

    def offload(value):
        nonlocal saved_bytes
        if not isinstance(value, jax.Array) or value.nbytes < minimum_bytes:
            return value
        if len(value.devices()) != 1:
            raise ValueError("host residual offload requires a single-GPU array")
        identity = id(value)
        if identity not in placed:
            placed[identity] = (
                value if value.sharding.memory_kind == "pinned_host" else jax.device_get(value)
            )
            saved_bytes += value.nbytes
        return placed[identity]

    args_res, opaque = jax.tree.map(offload, (pullback.args_res, pullback.opaque_residuals))
    return replace(pullback, args_res=args_res, opaque_residuals=opaque), saved_bytes


def checkpointed_initial_gradient(
    first_step,
    transition,
    objective,
    control,
    *,
    number_of_steps: int,
    checkpoint_interval: int = 10,
    primal_absolute_tolerance: float = 1e-10,
    observer=None,
    before_reverse=None,
    on_pullback_failure=None,
    resume_checkpoint=None,
    save_reverse_checkpoint=None,
    reverse_checkpoint_interval: int = 10,
    checkpoint_key=None,
    replay_checkpoint_slots: int = 0,
    linearize_step=None,
):
    if number_of_steps < 1 or checkpoint_interval < 1:
        raise ValueError("step count and checkpoint interval must be positive")
    if linearize_step is not None and not callable(linearize_step):
        raise TypeError('linearize_step must be callable or None')
    if reverse_checkpoint_interval < 1:
        raise ValueError("reverse checkpoint interval must be positive")
    if (isinstance(replay_checkpoint_slots, bool)
            or not isinstance(replay_checkpoint_slots, int) or replay_checkpoint_slots < 0):
        raise ValueError("replay checkpoint slots must be a nonnegative integer")
    replay_checkpoint_slots = min(replay_checkpoint_slots, max(0, number_of_steps - 2))
    control_digest = None
    if resume_checkpoint is not None or save_reverse_checkpoint is not None:
        if not isinstance(checkpoint_key, str) or not checkpoint_key:
            raise ValueError("durable reverse checkpoints require a source/input key")
        control_digest = tree_fingerprint(control)
    if resume_checkpoint is not None:
        resume_checkpoint.validate(
            steps=number_of_steps,
            key=checkpoint_key,
            control_fingerprint=control_digest,
        )
    report = observer if observer is not None else lambda event: None
    checkpoints, forward_values, replay_checkpoints = {}, {}, set()
    recomputed_steps = 0
    splits = (_replay_checkpoint_splits(
        min(checkpoint_interval, number_of_steps - 1), replay_checkpoint_slots
    ) if replay_checkpoint_slots else None)

    def retained_bytes():
        return _checkpoint_bytes(checkpoints)

    state = first_step(control)
    for completed in range(1, number_of_steps + 1):
        if completed > 1:
            with retained_host_memory(retained_bytes()):
                state = transition(state, completed - 1)
        jax.block_until_ready(state)
        forward_values[completed] = float(objective(state))
        if completed < number_of_steps and (completed == 1 or completed % checkpoint_interval == 0):
            checkpoints[completed] = HostCheckpoint.capture(state)
        report(
            dict(
                phase="forward",
                completed_steps=completed,
                checkpoint_bytes=retained_bytes(),
            )
        )
    with staged_reverse_ad():
        value, adjoint = _objective_initial_adjoint(objective, state)
    jax.block_until_ready((value, adjoint))
    if not np.isfinite(float(value)):
        raise FloatingPointError("non-finite forward objective")
    del state
    if before_reverse is not None:
        before_reverse()

    start_reverse = number_of_steps
    if resume_checkpoint is not None:
        resume_checkpoint.validate(
            steps=number_of_steps,
            key=checkpoint_key,
            control_fingerprint=control_digest,
            forward_values=tuple(forward_values[i] for i in range(1, number_of_steps + 1)),
        )
        adjoint = resume_checkpoint.adjoint.restore()
        adjoint = exact_zero_cotangents(adjoint)
        start_reverse = resume_checkpoint.next_reverse_step
        for index in tuple(checkpoints):
            if index >= start_reverse:
                del checkpoints[index]
        report(
            dict(
                phase="resume_reverse",
                reverse_step=start_reverse,
                reversed_steps=number_of_steps - start_reverse,
                checkpoint_bytes=retained_bytes(),
            )
        )

    def preceding_state(target):
        nonlocal recomputed_steps
        saved = max(index for index in checkpoints if index <= target)
        positions, anchor = set(), saved
        for free in range(replay_checkpoint_slots - len(replay_checkpoints), 0, -1):
            offset = splits[free][target - anchor]
            if offset is None:
                break
            anchor += offset
            positions.add(anchor)
        with retained_host_memory(retained_bytes()):
            result = checkpoints[saved].restore()
        for index in range(saved, target):
            with retained_host_memory(retained_bytes()):
                result = transition(result, index)
                recomputed_steps += 1
                if index + 1 in positions:
                    checkpoints[index + 1] = HostCheckpoint.capture(result)
                    replay_checkpoints.add(index + 1)
                    report(dict(
                        phase="cache_replay_state", checkpoint_step=index + 1,
                        replay_checkpoints=len(replay_checkpoints),
                        recomputed_forward_steps=recomputed_steps,
                        checkpoint_bytes=retained_bytes(),
                    ))
        return result

    def preserve_failed_pullback(
        error, completed, restore, structure, selected, incoming, *, phase="apply_pullback",
    ):
        if on_pullback_failure is None:
            return
        try:
            report(dict(phase="capture_failed_reverse", reverse_step=completed))
            point = restore()
            if isinstance(point, HostCheckpoint):
                point = point.restore_host()
            snapshot = HostCheckpoint.capture((point, incoming))
            on_pullback_failure(
                dict(
                    step=completed,
                    checkpoint=snapshot,
                    output_structure=structure,
                    selected=selected,
                    expected_objective=forward_values[completed],
                    error=repr(error),
                    failure_phase=phase,
                )
            )
        except Exception as capture_error:
            report(dict(phase="reverse_failure_capture_error", capture_error=repr(capture_error)))

    for completed in range(start_reverse, 1, -1):
        structure, selected, incoming = _cotangent_projection(adjoint)
        del adjoint
        prior = preceding_state(completed - 1)
        report(
            dict(
                phase="linearize_step",
                reverse_step=completed,
                nonzero_cotangent_leaves=len(selected),
            )
        )
        step = _projected_transition(
            lambda s, index=completed - 1: transition(s, index),
            objective,
            structure,
            selected,
        )
        failure_phase = "linearize_step"
        try:
            with (
                staged_reverse_ad(),
                retained_host_memory(retained_bytes()),
                stage_input_offload_scope(),
            ):
                if linearize_step is None:
                    primal, pullback, primal_objective = jax.vjp(step, prior, has_aux=True)
                else:
                    primal, pullback, primal_objective = linearize_step(
                        transition,prior,completed,structure,selected)
            failure_phase = "check_ad_primal"
            difference = abs(float(primal_objective) - forward_values[completed])
            if not np.isfinite(difference) or difference > primal_absolute_tolerance:
                raise ArithmeticError(f"AD primal differs at step {completed}: {difference}")
            del primal
            failure_phase = "offload_pullback"
            pullback, offloaded_bytes = offload_pullback_residuals(pullback)
            failure_phase = "save_reverse_input"
            saved_prior = (
                HostCheckpoint.capture(prior) if on_pullback_failure is not None else None
            )
        except Exception as error:
            preserve_failed_pullback(
                error, completed, lambda saved=prior: saved, structure, selected, incoming,
                phase=failure_phase,
            )
            raise
        del prior
        report(
            dict(
                phase="apply_pullback",
                reverse_step=completed,
                offloaded_residual_bytes=offloaded_bytes,
                reverse_input_checkpoint_bytes=0 if saved_prior is None else saved_prior.nbytes,
            )
        )
        try:
            with (
                staged_reverse_ad(),
                retained_host_memory(
                    retained_bytes() + offloaded_bytes
                    + (0 if saved_prior is None else saved_prior.nbytes)
                ),
                diagnose_stage_pullbacks(report, report_stages=True),
            ):
                adjoint = pullback(incoming)[0]
            jax.block_until_ready(adjoint)
        except Exception as error:
            preserve_failed_pullback(
                error,
                completed,
                lambda saved=saved_prior: saved,
                structure,
                selected,
                incoming,
            )
            raise
        del pullback, incoming, step, saved_prior
        for index in tuple(checkpoints):
            if index >= completed - 1:
                del checkpoints[index]
                replay_checkpoints.discard(index)
        report(
            dict(
                phase="reverse",
                reversed_steps=number_of_steps - completed + 1,
                checkpoint_bytes=retained_bytes(),
                recomputed_forward_steps=recomputed_steps,
            )
        )
        reversed_steps = number_of_steps - completed + 1
        if (
            save_reverse_checkpoint is not None
            and reversed_steps % reverse_checkpoint_interval == 0
        ):
            seed = ReverseCheckpoint(
                steps=number_of_steps,
                next_reverse_step=completed - 1,
                key=checkpoint_key,
                control_fingerprint=control_digest,
                forward_values=tuple(forward_values[i] for i in range(1, number_of_steps + 1)),
                adjoint=HostCheckpoint.capture(exact_zero_cotangents(adjoint)),
            )
            seed.validate(
                steps=number_of_steps, key=checkpoint_key, control_fingerprint=control_digest
            )
            save_reverse_checkpoint(seed)
    structure, selected, incoming = _cotangent_projection(adjoint)
    del adjoint
    report(dict(phase="linearize_step", reverse_step=1, nonzero_cotangent_leaves=len(selected)))
    step = _projected_transition(first_step, objective, structure, selected)
    failure_phase = "linearize_step"
    try:
        with (
            staged_reverse_ad(),
            retained_host_memory(retained_bytes()),
            stage_input_offload_scope(),
        ):
            if linearize_step is None:
                primal, pullback, primal_objective = jax.vjp(step, control, has_aux=True)
            else:
                primal, pullback, primal_objective = linearize_step(
                    first_step,control,1,structure,selected)
        failure_phase = "check_ad_primal"
        difference = abs(float(primal_objective) - forward_values[1])
        if not np.isfinite(difference) or difference > primal_absolute_tolerance:
            raise ArithmeticError(f"AD primal differs at step 1: {difference}")
        del primal
        failure_phase = "offload_pullback"
        pullback, offloaded_bytes = offload_pullback_residuals(pullback)
    except Exception as error:
        preserve_failed_pullback(
            error, 1, lambda: control, structure, selected, incoming, phase=failure_phase,
        )
        raise
    report(dict(phase="apply_pullback", reverse_step=1, offloaded_residual_bytes=offloaded_bytes))
    try:
        with (
            staged_reverse_ad(),
            retained_host_memory(retained_bytes() + offloaded_bytes),
            diagnose_stage_pullbacks(report, report_stages=True),
        ):
            gradient = pullback(incoming)[0]
        jax.block_until_ready(gradient)
    except Exception as error:
        preserve_failed_pullback(error, 1, lambda: control, structure, selected, incoming)
        raise
    del pullback, incoming, step
    for leaf in jax.tree.leaves(gradient):
        array = np.asarray(leaf)
        finite = np.isfinite(array)
        if not finite.all():
            indices = np.flatnonzero(~finite)
            sample = (
                np.stack(np.unravel_index(indices[:8], array.shape), axis=1).tolist()
                if array.ndim
                else [[] for _ in indices[:8]]
            )
            report(
                dict(phase="invalid_gradient", nonfinite_count=len(indices), control_indices=sample)
            )
            raise FloatingPointError(
                f"non-finite initial-control gradient at {sample} ({len(indices)} entries)"
            )
    report(dict(phase="reverse", reversed_steps=number_of_steps, checkpoint_bytes=0,
                recomputed_forward_steps=recomputed_steps))
    return value, gradient
