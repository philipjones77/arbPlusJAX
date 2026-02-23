import os


def parity_enabled() -> bool:
    return os.getenv("arbplusjax_RUN_PARITY", "0") == "1"


__all__ = ["parity_enabled"]
