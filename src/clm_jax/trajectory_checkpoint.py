
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
import uuid
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from time import perf_counter


@lru_cache(maxsize=1)
def _code_digest():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def model_contract(configuration):
    values = dict(
        code_sha256=_code_digest(),
        case=asdict(configuration.case),
        forcing=asdict(configuration.forcing),
        start=configuration.time.start.isoformat(),
        time_step_seconds=configuration.time.time_step_seconds,
        calendar=configuration.time.calendar,
        backend=configuration.execution.backend,
        x64=configuration.execution.enable_x64,
        history_interval_steps=configuration.output.history_interval_steps,
        history_records_per_file=configuration.output.history_records_per_file,
    )
    return hashlib.sha256(json.dumps(values, default=str, sort_keys=True).encode()).hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.partial")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_checkpoint(directory, payload, contract):
    import jax  

    started = perf_counter()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    step = int(payload["completed_steps"])
    name = f"land.{step:010d}.pkl"
    path = directory / name
    if path.exists():
        raise FileExistsError(f"refusing to replace an existing checkpoint: {path}")
    leaves, structure = jax.tree.flatten(payload)
    unique = {id(value): value for value in leaves}
    copied = jax.device_get(tuple(unique.values()))
    by_identity = dict(zip(unique, copied, strict=True))
    for identity, value in unique.items():
        if getattr(value, "weak_type", False):
            by_identity[identity] = value
    host_payload = structure.unflatten([by_identity[id(value)] for value in leaves])
    temporary = directory / f".{name}.{uuid.uuid4().hex}.partial"
    with temporary.open("xb") as stream:
        pickle.dump(host_payload, stream, protocol=5)
        stream.flush()
        os.fsync(stream.fileno())
    with temporary.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    temporary.replace(path)
    manifest = dict(
        format_version=1,
        file=name,
        sha256=checksum,
        contract=contract,
        completed_steps=step,
        bytes=path.stat().st_size,
        weak_array_owners=sum(
            bool(getattr(value, "weak_type", False)) for value in unique.values()
        ),
        wall_seconds=perf_counter() - started,
    )
    clock = host_payload.get("clock")
    if clock is not None:
        seconds = int(clock.seconds_into_day)
        manifest["model_time"] = (
            f"{int(clock.year):04d}-{int(clock.month):02d}-{int(clock.day):02d} "
            f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
        )
    _atomic_json(path.with_suffix(".json"), manifest)
    _atomic_json(directory / "latest.json", manifest)
    return manifest


def load_checkpoint(directory, contract):
    directory = Path(directory)
    pointer = directory / "latest.json"
    if not pointer.is_file():
        return None
    manifest = json.loads(pointer.read_text())
    if manifest["format_version"] != 1 or manifest["contract"] != contract:
        raise ValueError("checkpoint source/input/clock contract differs from this run")
    if not re.fullmatch(r"land\.\d{10}\.pkl", manifest["file"]):
        raise ValueError("invalid checkpoint filename")
    path = directory / manifest["file"]
    if path.is_symlink():
        raise ValueError("checkpoint payload must not be a symlink")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["sha256"]:
            raise ValueError("checkpoint SHA-256 mismatch")
        stream.seek(0)
        payload = pickle.load(stream)  
    if int(payload["completed_steps"]) != manifest["completed_steps"]:
        raise ValueError("checkpoint step index differs from its manifest")
    return payload


def recover_uncommitted_history(directory, case_name, clock):
    directory = Path(directory)
    if not directory.exists():
        return []
    stamp = (
        f"{int(clock.year):04d}-{int(clock.month):02d}-{int(clock.day):02d}-"
        f"{int(clock.seconds_into_day):05d}"
    )
    pattern = re.compile(
        re.escape(case_name) + r"\.clm2\.h[12]\."
        r"(\d{4}-\d{2}-\d{2}-\d{5})\.(?:nc|complete\.json)$"
    )
    pending = []
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match and match[1] > stamp:
            if path.is_symlink():
                raise ValueError(f"refusing to move a symlinked history segment: {path}")
            pending.append(path)
    if not pending:
        return []
    archive = directory.parent / "recovery_orphans" / uuid.uuid4().hex
    archive.mkdir(parents=True)
    for path in pending:
        path.rename(archive / path.name)
    return [str(archive / path.name) for path in pending]


def prepare_observation_recovery(configuration):
    if not configuration.output.checkpointing:
        return
    pointer = configuration.output.directory.parent / "checkpoints/latest.json"
    if not pointer.is_file():
        return
    manifest = json.loads(pointer.read_text())
    if manifest["contract"] != model_contract(configuration):
        raise ValueError("checkpoint contract differs before observation recovery")
    cutoff = manifest["model_time"]
    root = configuration.source.parent
    archive = configuration.output.directory.parent / "recovery_orphans" / uuid.uuid4().hex
    for name, log_name in (("accuracy", "checkpoints.jsonl"), ("baseline", "comparisons.jsonl")):
        folder = root / name
        log = folder / log_name
        if log.is_file():
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            kept = [row for row in rows if row["time"] <= cutoff]
            if len(kept) != len(rows):
                destination = archive / name
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(log, destination / log.name)
                temporary = log.with_suffix(".jsonl.partial")
                temporary.write_text("".join(json.dumps(row) + "\n" for row in kept))
                temporary.replace(log)
                latest = folder / "latest.json"
                if latest.exists():
                    latest.rename(destination / latest.name)
                if kept:
                    _atomic_json(latest, kept[-1])
        for path in folder.glob("annual-*.npz"):
            if path.stem.removeprefix("annual-") > cutoff[:10]:
                destination = archive / name
                destination.mkdir(parents=True, exist_ok=True)
                path.rename(destination / path.name)
        for filename in ("observer_exit.json", "stop_report.json"):
            path = folder / filename
            if path.exists():
                destination = archive / name
                destination.mkdir(parents=True, exist_ok=True)
                path.rename(destination / path.name)
