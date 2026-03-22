from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_on_path(anchor: str | Path) -> Path:
    path = Path(anchor).resolve()
    for candidate in (path.parent, *path.parents):
        if (candidate / "src" / "arbplusjax").exists():
            repo_root = candidate
            src_root = repo_root / "src"
            if str(src_root) not in sys.path:
                sys.path.insert(0, str(src_root))
            if str(repo_root) not in sys.path:
                sys.path.insert(1, str(repo_root))
            return repo_root
    raise RuntimeError(f"Could not locate repo root from {path}")
