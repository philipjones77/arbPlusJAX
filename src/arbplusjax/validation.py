import os


def parity_enabled() -> bool:
    # Prefer canonical uppercase name, keep lowercase for backward compatibility.
    return os.getenv("ARBPLUSJAX_RUN_PARITY", os.getenv("arbplusjax_RUN_PARITY", "0")) == "1"


__all__ = ["parity_enabled"]
