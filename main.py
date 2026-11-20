
from __future__ import annotations

import sys
from pathlib import Path

from clm_jax.model_config import apply_jax_environment, load_model_configuration

CONFIGURATION_FILE = Path(__file__).with_name("config.toml")


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("main.py accepts no arguments; edit config.toml instead")
    configuration = load_model_configuration(CONFIGURATION_FILE)
    apply_jax_environment(configuration.execution)
    from clm_jax.application import run_configured_model

    return run_configured_model(configuration)


if __name__ == "__main__":
    raise SystemExit(main())
