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
    # Default to archived migration build output in arbPlusJAX
    return Path(r"C:\Users\phili\OneDrive\Documents\GitHub\arbPlusJAX\stuff\migration\c_chassis\build")
