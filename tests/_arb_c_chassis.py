from __future__ import annotations

# Auto-detected location for Arb C reference DLLs.
# If ARB_C_REF_DIR is set, that path is used. Otherwise we fall back to the
# archived migration build output stored in arbPlusJAX/stuff.

from tests._test_checks import _check

import os
from pathlib import Path


def get_c_ref_build_dir() -> Path:
    env = os.environ.get("ARB_C_REF_DIR")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parents[1]
    candidates = (
        repo_root / "stuff" / "migration" / "c_chassis" / "build_linux_wsl",
        repo_root / "stuff" / "migration" / "c_chassis" / "build",
        repo_root / "migration" / "c_chassis" / "build_linux_wsl",
        repo_root / "migration" / "c_chassis" / "build",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Fall back to the historical archived path even if it does not exist yet.
    return candidates[1]
