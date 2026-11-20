
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

import f90nml

from clm_jax.io.cime_case import CimeCase

NamelistScalar: TypeAlias = bool | int | float | complex | str | None
NamelistValue: TypeAlias = NamelistScalar | tuple["NamelistValue", ...]


def read_namelist_group(
    path: str | Path,
    group_name: str,
) -> Mapping[str, NamelistValue]:

    namelist_path = Path(path).expanduser().resolve()
    if not namelist_path.is_file():
        raise FileNotFoundError(f"Fortran namelist does not exist: {namelist_path}")

    return _read_unchanged_group(namelist_path, group_name, namelist_path.read_text())


@lru_cache(maxsize=128)
def _read_unchanged_group(
    namelist_path: Path,
    group_name: str,
    contents: str,
) -> Mapping[str, NamelistValue]:
    parsed = f90nml.reads(contents)
    if group_name not in parsed:
        raise KeyError(f"namelist group {group_name!r} is absent from {namelist_path}")
    group = parsed[group_name]
    return MappingProxyType({str(key): _freeze(value) for key, value in group.items()})


def read_case_namelist_group(
    case: CimeCase,
    file_name: str,
    group_name: str,
) -> Mapping[str, NamelistValue]:

    return read_namelist_group(case.case_docs / file_name, group_name)


def _freeze(value: object) -> NamelistValue:
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, bool | int | float | complex | str):
        return value
    raise TypeError(f"unsupported value returned by f90nml: {type(value).__name__}")
