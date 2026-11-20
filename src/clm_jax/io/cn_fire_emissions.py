
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import f90nml

from clm_jax.io.cime_case import CimeCase


class CnFireEmissionsConfiguration(NamedTuple):

    enabled: bool
    mechanism_component_count: int
    configuration_path: Path


def _specifier_count(value: object) -> int:
    if value is None:
        return 0
    values = value if isinstance(value, list | tuple) else (value,)
    return sum(isinstance(item, str) and bool(item.strip()) for item in values)


def load_cn_fire_emissions_configuration(
    case: CimeCase,
) -> CnFireEmissionsConfiguration:

    path = (case.case_docs / "drv_flds_in").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"generated driver-fields namelist does not exist: {path}")
    namelist = f90nml.read(path)
    if "fire_emis_nl" not in namelist:
        return CnFireEmissionsConfiguration(False, 0, path)
    count = _specifier_count(namelist["fire_emis_nl"].get("fire_emis_specifier"))
    return CnFireEmissionsConfiguration(count > 0, count, path)


def require_disabled_cn_fire_emissions(
    configuration: CnFireEmissionsConfiguration,
) -> None:

    if configuration.enabled or configuration.mechanism_component_count != 0:
        raise NotImplementedError(
            "active fire-chemistry mechanisms require the full CNFireEmisUpdate branch"
        )


__all__ = [
    "CnFireEmissionsConfiguration",
    "load_cn_fire_emissions_configuration",
    "require_disabled_cn_fire_emissions",
]
