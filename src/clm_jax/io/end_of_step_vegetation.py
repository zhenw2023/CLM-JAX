
from __future__ import annotations

import shlex
from typing import NamedTuple

from clm_jax.io.cime_case import CimeCase
from clm_jax.io.namelist import read_case_namelist_group


class EndOfStepVegetationConfiguration(NamedTuple):
    use_cn: bool
    use_crop: bool
    use_cndv: bool
    biogeochemistry_mode: str
    build_namelist_options: str
    compset: str


def _logical(value: object, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {".true.", "true", "t"}:
            return True
        if normalized in {".false.", "false", "f"}:
            return False
    raise ValueError(f"{name} is not a scalar Fortran logical: {value!r}")


def _option_value(tokens: list[str], option: str, default: str) -> str:
    matches = [index for index, token in enumerate(tokens) if token == option]
    if not matches:
        return default
    index = matches[-1]
    if index + 1 >= len(tokens):
        raise ValueError(f"{option} lacks a value in CLM_BLDNML_OPTS")
    return tokens[index + 1].lower()


def load_end_of_step_vegetation_configuration(
    case: CimeCase,
) -> EndOfStepVegetationConfiguration:

    options = case.value("CLM_BLDNML_OPTS", "")
    tokens = shlex.split(options)
    bgc_mode = _option_value(tokens, "-bgc", "none")
    general = dict(read_case_namelist_group(case, "lnd_in", "clm_inparm"))
    use_cn = _logical(general.get("use_cn", bgc_mode in {"bgc", "cn", "cndv"}), name="use_cn")
    use_crop = _logical(general.get("use_crop", "-crop" in tokens), name="use_crop")
    declared_cndv = general.get("use_cndv")
    use_cndv = (
        _logical(declared_cndv, name="use_cndv")
        if declared_cndv is not None
        else bgc_mode == "cndv" or "-dynamic_vegetation" in tokens
    )
    if not use_cn:
        raise ValueError("EndOfTimeStepVegDynamics is only called when use_cn=.true.")
    return EndOfStepVegetationConfiguration(
        use_cn=use_cn,
        use_crop=use_crop,
        use_cndv=use_cndv,
        biogeochemistry_mode=bgc_mode,
        build_namelist_options=options,
        compset=case.value("COMPSET"),
    )


__all__ = [
    "EndOfStepVegetationConfiguration",
    "load_end_of_step_vegetation_configuration",
]
