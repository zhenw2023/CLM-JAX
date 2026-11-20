
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

_VARIABLE = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)
_MISSING = object()


@dataclass(frozen=True)
class InputFileEntry:
    component: str
    key: str
    declared_value: str
    path: Path

    @property
    def exists(self) -> bool:
        return self.path.exists()


class CimeCase:

    def __init__(self, root: str | Path, values: Mapping[str, str]) -> None:
        self.root = Path(root).expanduser().resolve()
        self._raw_values = dict(values)
        self._values = self._resolve_all(values)

    @classmethod
    def load(cls, root: str | Path) -> CimeCase:
        case_root = Path(root).expanduser().resolve()
        if not case_root.is_dir():
            raise FileNotFoundError(f"CIME case directory does not exist: {case_root}")

        values: dict[str, str] = {}
        xml_paths = sorted(case_root.glob("env_*.xml"))
        if not xml_paths:
            raise ValueError(f"no env_*.xml files found in CIME case: {case_root}")

        for xml_path in xml_paths:
            tree = ET.parse(xml_path)
            for entry in tree.getroot().iter("entry"):
                key = entry.attrib.get("id")
                value = entry.attrib.get("value")
                if key is not None and value is not None:
                    values[key] = value
                if key is not None:
                    for item in entry.findall("values/value"):
                        component = item.attrib.get("compclass")
                        if component is not None and item.text is not None:
                            values[f"{key}_{component}"] = item.text.strip()

        values.setdefault("CASEROOT", str(case_root))
        values.setdefault("CASE", case_root.name)
        return cls(case_root, values)

    @property
    def raw_values(self) -> Mapping[str, str]:
        return dict(self._raw_values)

    @property
    def values(self) -> Mapping[str, str]:
        return dict(self._values)

    def value(self, key: str, default: object = _MISSING, *, raw: bool = False) -> str:
        source = self._raw_values if raw else self._values
        if key in source:
            return source[key]
        if default is _MISSING:
            raise KeyError(key)
        return str(default)

    def path_value(self, key: str) -> Path:
        return Path(self.value(key)).expanduser()

    @property
    def case_docs(self) -> Path:
        return self.root / "CaseDocs"

    @property
    def buildconf(self) -> Path:
        return self.root / "Buildconf"

    def input_file_entries(self) -> Iterator[InputFileEntry]:

        if not self.buildconf.is_dir():
            return

        run_directory = self.path_value("RUNDIR")
        for list_path in sorted(self.buildconf.glob("*.input_data_list")):
            component = list_path.name.removesuffix(".input_data_list")
            for line in list_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, declared = (part.strip() for part in stripped.split("=", 1))
                declared = declared.strip("'\"")
                expanded = self.expand(declared)
                path = Path(expanded).expanduser()
                if not path.is_absolute():
                    path = run_directory / path
                yield InputFileEntry(component, key, declared, path)

    def datm_stream_paths(self) -> tuple[Path, ...]:
        return tuple(sorted(self.case_docs.glob("datm.streams.txt.*")))

    def expand(self, value: str) -> str:
        result = value
        lookup = {**os.environ, **self._values}
        for _ in range(100):
            expanded = _VARIABLE.sub(
                lambda match: lookup.get(
                    match.group("braced") or match.group("plain"), match.group(0)
                ),
                result,
            )
            if expanded == result:
                return expanded
            result = expanded
        raise ValueError(f"cyclic CIME variable expansion: {value!r}")

    def _resolve_all(self, values: Mapping[str, str]) -> dict[str, str]:
        resolved = {**os.environ, **values}
        for _ in range(100):
            changed = False
            for key, value in tuple(resolved.items()):
                expanded = _VARIABLE.sub(
                    lambda match: resolved.get(
                        match.group("braced") or match.group("plain"), match.group(0)
                    ),
                    value,
                )
                if expanded != value:
                    resolved[key] = expanded
                    changed = True
            if not changed:
                return {key: resolved[key] for key in values}
        raise ValueError("cyclic references in CIME XML values")
