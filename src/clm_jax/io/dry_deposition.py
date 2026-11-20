
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_namelist_group


class DryDepositionConfiguration(NamedTuple):
    species: tuple[str, ...]
    method: str
    driver_fields_path: Path

    @property
    def number_of_species(self) -> int:
        return len(self.species)

    @property
    def land_interactive(self) -> bool:
        return self.number_of_species > 0 and self.method == "xactive_lnd"


def load_dry_deposition_configuration(case: CimeCase) -> DryDepositionConfiguration:

    path = (case.buildconf / "clmconf" / "drv_flds_in").resolve()
    group = dict(read_namelist_group(path, "drydep_inparm"))
    raw_species = group.get("drydep_list", ())
    if isinstance(raw_species, str):
        species = (raw_species.strip(),) if raw_species.strip() else ()
    elif isinstance(raw_species, tuple):
        if not all(isinstance(value, str) for value in raw_species):
            raise ValueError("drydep_list must contain only species names")
        species = tuple(value.strip() for value in raw_species if value.strip())
    else:
        raise ValueError("drydep_list must be a string sequence")
    raw_method = group.get("drydep_method", "xactive_lnd")
    if not isinstance(raw_method, str):
        raise ValueError("drydep_method must be a scalar string")
    return DryDepositionConfiguration(
        species=species,
        method=raw_method.strip().lower(),
        driver_fields_path=path,
    )


__all__ = ["DryDepositionConfiguration", "load_dry_deposition_configuration"]
