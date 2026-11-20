
from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_MAGIC = b"CLM-JAX-REVERSE-1\n"


class _DigestWriter:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, data):
        self.digest.update(data)
        return self.stream.write(data)


def tree_fingerprint(tree):
    from clm_jax.trajectory_adjoint import HostCheckpoint

    saved = HostCheckpoint.capture(tree)
    digest = hashlib.sha256()
    digest.update(str(saved.structure).encode())
    digest.update(str(saved.slots).encode())
    digest.update(str(saved.weak_types).encode())
    for value in saved.values:
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("reverse checkpoint fingerprints require numerical inputs")
        digest.update(str((array.shape, array.dtype.descr)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ReverseCheckpoint:

    steps: int
    next_reverse_step: int
    key: str
    control_fingerprint: str
    forward_values: tuple[float, ...]
    adjoint: Any
    version: int = 1

    def validate(self, *, steps, key, control_fingerprint, forward_values=None):
        if (self.version != 1 or self.steps != steps or self.key != key
                or self.control_fingerprint != control_fingerprint):
            raise ValueError("reverse checkpoint source, inputs or horizon differ")
        if not 1 <= self.next_reverse_step < steps:
            raise ValueError("reverse checkpoint step is outside the complete horizon")
        saved = np.asarray(self.forward_values, dtype=np.float64)
        if saved.shape != (steps,) or not np.isfinite(saved).all():
            raise ValueError("reverse checkpoint requires every finite forward objective")
        if forward_values is not None:
            current = np.asarray(forward_values, dtype=np.float64)
            if current.shape != saved.shape or current.tobytes() != saved.tobytes():
                raise ValueError("recomputed forward objectives differ from reverse checkpoint")

    def save(self, path):
        path = Path(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(_MAGIC + bytes(32))
                writer = _DigestWriter(stream)
                pickle.dump(self, writer, protocol=5)
                stream.seek(len(_MAGIC))
                stream.write(writer.digest.digest())
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as stream:
            if stream.read(len(_MAGIC)) != _MAGIC:
                raise ValueError("invalid reverse checkpoint header")
            expected = stream.read(32)
            digest = hashlib.sha256()
            while data := stream.read(1 << 20):
                digest.update(data)
            if expected != digest.digest():
                raise ValueError("reverse checkpoint checksum differs")
            stream.seek(len(_MAGIC) + 32)
            result = pickle.load(stream)
        if not isinstance(result, cls):
            raise ValueError("file is not a reverse checkpoint")
        return result
