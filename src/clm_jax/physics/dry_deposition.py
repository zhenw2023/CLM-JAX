
from __future__ import annotations

from typing import TypeVar

State = TypeVar("State")


def dry_deposition_velocity(
    state: State,
    *,
    species: tuple[str, ...],
    method: str,
) -> State:

    normalized_method = method.strip().lower()
    if not species or normalized_method != "xactive_lnd":
        return state
    raise NotImplementedError(
        "interactive land Wesely dry deposition is not active in the reference "
        "case and has not been ported"
    )


__all__ = ["dry_deposition_velocity"]
