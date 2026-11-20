
from __future__ import annotations

from typing import TypeVar

State = TypeVar("State")


def end_of_step_vegetation_dynamics(state: State, *, use_cndv: bool) -> State:

    if use_cndv:
        raise NotImplementedError(
            "CNDV annual vegetation dynamics are outside the I2000Clm50BgcCrop "
            "reference configuration and have not been ported"
        )
    return state


__all__ = ["end_of_step_vegetation_dynamics"]
