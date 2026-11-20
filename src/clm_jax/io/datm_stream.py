
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VariableMapping:
    file_variable: str
    model_variable: str


@dataclass(frozen=True)
class DatmStream:
    source_path: Path
    data_source: str
    domain_files: tuple[Path, ...]
    field_files: tuple[Path, ...]
    domain_variables: tuple[VariableMapping, ...]
    field_variables: tuple[VariableMapping, ...]
    offset: int

    @classmethod
    def load(cls, source_path: str | Path) -> DatmStream:
        path = Path(source_path).expanduser().resolve()
        root = ET.parse(path).getroot()
        domain = root.find("domainInfo")
        field = root.find("fieldInfo")
        if domain is None or field is None:
            raise ValueError(f"DATM stream must contain domainInfo and fieldInfo: {path}")

        source = _text(root.find("dataSource"), default="GENERIC")
        offset_text = _text(field.find("offset"), default="0")
        return cls(
            source_path=path,
            data_source=source,
            domain_files=_files(domain),
            field_files=_files(field),
            domain_variables=_variable_mappings(domain),
            field_variables=_variable_mappings(field),
            offset=int(offset_text),
        )

    def missing_files(self) -> tuple[Path, ...]:
        return tuple(path for path in (*self.domain_files, *self.field_files) if not path.exists())


def _text(element: ET.Element | None, *, default: str | None = None) -> str:
    if element is None or element.text is None or not element.text.strip():
        if default is None:
            raise ValueError("required DATM stream element is empty")
        return default
    return element.text.strip()


def _files(section: ET.Element) -> tuple[Path, ...]:
    directory = Path(_text(section.find("filePath"))).expanduser()
    names = _text(section.find("fileNames")).split()
    return tuple(directory / name for name in names)


def _variable_mappings(section: ET.Element) -> tuple[VariableMapping, ...]:
    text = _text(section.find("variableNames"))
    mappings: list[VariableMapping] = []
    for line in text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) != 2:
            raise ValueError(f"invalid DATM variable mapping line: {line!r}")
        mappings.append(VariableMapping(tokens[0], tokens[1]))
    return tuple(mappings)
