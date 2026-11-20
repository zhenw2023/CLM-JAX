
from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from clm_jax.calendar import noleap_seconds_between, validate_noleap_datetime

_SCHEMA_VERSION = 1
_RUN_TYPES = frozenset({"startup", "branch", "hybrid"})
_BACKENDS = frozenset({"auto", "cpu", "gpu"})


@dataclass(frozen=True)
class CaseConfiguration:
    directory: Path
    inputdata_root: Path
    restart_file: Path
    coupler_restart_file: Path


@dataclass(frozen=True)
class ForcingConfiguration:
    nearest_fill_mapping: Path
    bilinear_remapping: Path
    chunk_steps: int
    river_volume_file: Path | None = None


@dataclass(frozen=True)
class TimeConfiguration:
    start: datetime
    stop: datetime
    time_step_seconds: int
    calendar: str
    run_type: str

    @property
    def step_count(self) -> int:
        return noleap_seconds_between(self.start, self.stop) // self.time_step_seconds


@dataclass(frozen=True)
class ExecutionConfiguration:
    backend: str
    device_count: int
    enable_x64: bool
    jit: bool
    compilation_cache_directory: Path
    maximum_rss_gib: float
    gpu_loop_command_buffers: bool = False
    profile_stages: bool = False
    gpu_pin_dispatch_thread: bool = True
    gpu_visible_devices: str | None = None


@dataclass(frozen=True)
class OutputConfiguration:
    directory: Path
    history_interval_steps: int
    restart_interval_steps: int
    overwrite: bool
    history_records_per_file: int | None = None
    checkpointing: bool = False

    def __post_init__(self):
        if self.checkpointing:
            if self.history_records_per_file is None:
                raise ValueError("checkpointing requires segmented history files")
            segment_steps = self.history_interval_steps * self.history_records_per_file
            if self.restart_interval_steps % segment_steps:
                raise ValueError("restart_interval_steps must end on a closed history segment")


@dataclass(frozen=True)
class DevelopmentConfiguration:
    preflight_only: bool


@dataclass(frozen=True)
class ValidationConfiguration:
    reference_h1: Path
    reference_h2: Path


@dataclass(frozen=True)
class ModelConfiguration:
    source: Path
    case: CaseConfiguration
    forcing: ForcingConfiguration
    time: TimeConfiguration
    execution: ExecutionConfiguration
    output: OutputConfiguration
    development: DevelopmentConfiguration
    validation: ValidationConfiguration | None


def _table(
    document: dict[str, Any],
    name: str,
    expected: set[str],
    *,
    optional: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"configuration requires a [{name}] table")
    unknown = set(value) - (expected | optional)
    if unknown:
        raise ValueError(f"unknown [{name}] keys: {sorted(unknown)}")
    missing = expected - set(value)
    if missing:
        raise ValueError(f"missing [{name}] keys: {sorted(missing)}")
    return value


def _path(value: Any, *, name: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _datetime(value: Any, *, name: str) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"{name} must be an ISO-8601 date-time") from error
    else:
        raise ValueError(f"{name} must be an ISO-8601 date-time")
    if result.tzinfo is not None:
        raise ValueError(f"{name} must use the model calendar without a timezone")
    return result


def _integer(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0.0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def load_model_configuration(path: str | Path) -> ModelConfiguration:

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"CLM-JAX configuration does not exist: {source}")
    with source.open("rb") as stream:
        document = tomllib.load(stream)
    allowed_tables = {
        "schema_version",
        "case",
        "forcing",
        "time",
        "execution",
        "output",
        "development",
        "validation",
    }
    unknown_tables = set(document) - allowed_tables
    if unknown_tables:
        raise ValueError(f"unknown top-level configuration keys: {sorted(unknown_tables)}")
    if document.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {_SCHEMA_VERSION}")
    base = source.parent
    case_values = _table(
        document,
        "case",
        {
            "directory",
            "inputdata_root",
            "restart_file",
            "coupler_restart_file",
        },
    )
    forcing_values = _table(
        document,
        "forcing",
        {"nearest_fill_mapping", "bilinear_remapping", "chunk_steps"},
        optional=frozenset({"river_volume_file"}),
    )
    time_values = _table(
        document,
        "time",
        {"start", "stop", "time_step_seconds", "calendar", "run_type"},
    )
    execution_values = _table(
        document,
        "execution",
        {
            "backend",
            "device_count",
            "enable_x64",
            "jit",
            "compilation_cache_directory",
            "maximum_rss_gib",
        },
        optional=frozenset(
            {
                "gpu_loop_command_buffers",
                "profile_stages",
                "gpu_pin_dispatch_thread",
                "gpu_visible_devices",
            }
        ),
    )
    output_values = _table(
        document,
        "output",
        {"directory", "history_interval_steps", "restart_interval_steps", "overwrite"},
        optional=frozenset({"history_records_per_file", "checkpointing"}),
    )
    development_values = _table(document, "development", {"preflight_only"})
    validation_values = (
        None
        if "validation" not in document
        else _table(document, "validation", {"reference_h1", "reference_h2"})
    )

    gpu_visible_devices = execution_values.get("gpu_visible_devices")
    if gpu_visible_devices is not None and (
        execution_values["backend"] != "gpu"
        or not isinstance(gpu_visible_devices, str)
        or not gpu_visible_devices.strip()
        or any(not part.strip() for part in gpu_visible_devices.split(","))
    ):
        raise ValueError("execution.gpu_visible_devices requires a nonempty GPU device list")

    start = _datetime(time_values["start"], name="time.start")
    stop = _datetime(time_values["stop"], name="time.stop")
    step_seconds = _integer(time_values["time_step_seconds"], name="time.time_step_seconds")
    validate_noleap_datetime(start, name="time.start")
    validate_noleap_datetime(stop, name="time.stop")
    duration_seconds = noleap_seconds_between(start, stop)
    if duration_seconds <= 0:
        raise ValueError("time.stop must be later than time.start")
    if duration_seconds % step_seconds:
        raise ValueError("configured duration must be exactly divisible by time_step_seconds")
    calendar = time_values["calendar"]
    if calendar != "noleap":
        raise ValueError("the I2000Clm50BgcCrop configuration requires calendar='noleap'")
    run_type = time_values["run_type"]
    if run_type not in _RUN_TYPES:
        raise ValueError(f"time.run_type must be one of {sorted(_RUN_TYPES)}")
    backend = execution_values["backend"]
    if backend not in _BACKENDS:
        raise ValueError(f"execution.backend must be one of {sorted(_BACKENDS)}")
    gpu_loop_command_buffers = _boolean(
        execution_values.get("gpu_loop_command_buffers", False),
        name="execution.gpu_loop_command_buffers",
    )
    if gpu_loop_command_buffers and backend != "gpu":
        raise ValueError("execution.gpu_loop_command_buffers requires backend='gpu'")

    return ModelConfiguration(
        source=source,
        case=CaseConfiguration(
            directory=_path(case_values["directory"], name="case.directory", base=base),
            inputdata_root=_path(
                case_values["inputdata_root"], name="case.inputdata_root", base=base
            ),
            restart_file=_path(case_values["restart_file"], name="case.restart_file", base=base),
            coupler_restart_file=_path(
                case_values["coupler_restart_file"],
                name="case.coupler_restart_file",
                base=base,
            ),
        ),
        forcing=ForcingConfiguration(
            nearest_fill_mapping=_path(
                forcing_values["nearest_fill_mapping"],
                name="forcing.nearest_fill_mapping",
                base=base,
            ),
            bilinear_remapping=_path(
                forcing_values["bilinear_remapping"],
                name="forcing.bilinear_remapping",
                base=base,
            ),
            chunk_steps=_integer(
                forcing_values["chunk_steps"],
                name="forcing.chunk_steps",
            ),
            river_volume_file=(
                _path(
                    forcing_values["river_volume_file"], name="forcing.river_volume_file", base=base
                )
                if "river_volume_file" in forcing_values
                else None
            ),
        ),
        time=TimeConfiguration(
            start=start,
            stop=stop,
            time_step_seconds=step_seconds,
            calendar=calendar,
            run_type=run_type,
        ),
        execution=ExecutionConfiguration(
            backend=backend,
            device_count=_integer(execution_values["device_count"], name="execution.device_count"),
            enable_x64=_boolean(execution_values["enable_x64"], name="execution.enable_x64"),
            jit=_boolean(execution_values["jit"], name="execution.jit"),
            compilation_cache_directory=_path(
                execution_values["compilation_cache_directory"],
                name="execution.compilation_cache_directory",
                base=base,
            ),
            maximum_rss_gib=_positive_number(
                execution_values["maximum_rss_gib"],
                name="execution.maximum_rss_gib",
            ),
            gpu_loop_command_buffers=gpu_loop_command_buffers,
            profile_stages=_boolean(
                execution_values.get("profile_stages", False),
                name="execution.profile_stages",
            ),
            gpu_pin_dispatch_thread=_boolean(
                execution_values.get("gpu_pin_dispatch_thread", True),
                name="execution.gpu_pin_dispatch_thread",
            ),
            gpu_visible_devices=gpu_visible_devices,
        ),
        output=OutputConfiguration(
            directory=_path(output_values["directory"], name="output.directory", base=base),
            history_interval_steps=_integer(
                output_values["history_interval_steps"],
                name="output.history_interval_steps",
            ),
            restart_interval_steps=_integer(
                output_values["restart_interval_steps"],
                name="output.restart_interval_steps",
            ),
            overwrite=_boolean(output_values["overwrite"], name="output.overwrite"),
            history_records_per_file=(
                _integer(
                    output_values["history_records_per_file"],
                    name="output.history_records_per_file",
                )
                if "history_records_per_file" in output_values
                else None
            ),
            checkpointing=_boolean(
                output_values.get("checkpointing", False), name="output.checkpointing"
            ),
        ),
        development=DevelopmentConfiguration(
            preflight_only=_boolean(
                development_values["preflight_only"],
                name="development.preflight_only",
            )
        ),
        validation=(
            None
            if validation_values is None
            else ValidationConfiguration(
                reference_h1=_path(
                    validation_values["reference_h1"],
                    name="validation.reference_h1",
                    base=base,
                ),
                reference_h2=_path(
                    validation_values["reference_h2"],
                    name="validation.reference_h2",
                    base=base,
                ),
            )
        ),
    )


def jax_environment(configuration: ExecutionConfiguration) -> dict[str, str]:

    environment: dict[str, str] = {}
    environment["JAX_COMPILATION_CACHE_DIR"] = str(configuration.compilation_cache_directory)
    if configuration.backend != "auto":
        environment["JAX_PLATFORMS"] = (
            "cuda" if configuration.backend == "gpu" else configuration.backend
        )
    if configuration.backend == "gpu":
        if configuration.gpu_visible_devices is not None:
            environment["CUDA_VISIBLE_DEVICES"] = configuration.gpu_visible_devices
        environment["JAX_HLO_SOURCE_FILE_CANONICALIZATION_REGEX"] = r"^.*(?=/clm_jax/)"
        environment["JAX_INCLUDE_FULL_TRACEBACKS_IN_LOCATIONS"] = "false"
    platform_count = execution_shard_count(configuration)
    if configuration.backend in {"auto", "cpu"}:
        existing = os.environ.get("XLA_FLAGS", "").strip()
        tokens = existing.split()
        tokens = [
            token
            for token in tokens
            if not token.startswith("--xla_force_host_platform_device_count=")
            and not token.startswith("--xla_cpu_parallel_codegen_split_count=")
        ]
        tokens.append("--xla_cpu_parallel_codegen_split_count=16")
        if platform_count > 1:
            tokens.append(f"--xla_force_host_platform_device_count={platform_count}")
        environment["XLA_FLAGS"] = " ".join(tokens)
    elif configuration.gpu_loop_command_buffers:
        tokens = [
            token
            for token in os.environ.get("XLA_FLAGS", "").split()
            if not token.startswith("--xla_gpu_enable_command_buffer=")
        ]
        tokens.append("--xla_gpu_enable_command_buffer=+WHILE")
        environment["XLA_FLAGS"] = " ".join(tokens)
    return environment


def execution_shard_count(configuration: ExecutionConfiguration) -> int:

    return configuration.device_count


def _bind_configured_cpu_affinity(configuration: ExecutionConfiguration) -> None:

    if configuration.backend in {"auto", "cpu"} and hasattr(os, "sched_getaffinity"):
        allowed = tuple(sorted(os.sched_getaffinity(0)))
        if len(allowed) < configuration.device_count:
            raise RuntimeError(
                f"configuration requests {configuration.device_count} CPU cores, "
                f"but the process may use only {len(allowed)}"
            )
        os.sched_setaffinity(0, allowed[: configuration.device_count])


def apply_jax_environment(configuration: ExecutionConfiguration) -> None:

    if "jax" in sys.modules:
        raise RuntimeError("JAX was imported before the model configuration was applied")
    _bind_configured_cpu_affinity(configuration)
    os.environ.update(jax_environment(configuration))


__all__ = [
    "CaseConfiguration",
    "DevelopmentConfiguration",
    "ExecutionConfiguration",
    "ForcingConfiguration",
    "ModelConfiguration",
    "OutputConfiguration",
    "TimeConfiguration",
    "ValidationConfiguration",
    "apply_jax_environment",
    "execution_shard_count",
    "jax_environment",
    "load_model_configuration",
]
