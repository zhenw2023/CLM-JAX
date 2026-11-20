
from __future__ import annotations

from typing import Any


def configure_runtime(*args: Any, **kwargs: Any):

    from clm_jax.runtime import configure_runtime as _configure_runtime

    return _configure_runtime(*args, **kwargs)

__all__ = ["configure_runtime"]
__version__ = "3.2.46"
