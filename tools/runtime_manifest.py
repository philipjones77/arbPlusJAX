from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _detect_environment_kind() -> str:
    if os.getenv("COLAB_RELEASE_TAG") or os.getenv("COLAB_GPU"):
        return "colab"
    if "google.colab" in sys.modules:
        return "colab"
    if platform.system().lower() == "linux" and "microsoft" in platform.release().lower():
        return "wsl"
    return "local"


def _git_commit(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return ""
    return completed.stdout.strip()


def _safe_jax_info() -> dict[str, Any]:
    try:
        import jax
    except Exception:
        return {"available": False, "backend": "", "devices": [], "jax_enable_x64": None}
    return {
        "available": True,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
    }


def collect_runtime_manifest(repo_root: Path, *, jax_mode: str = "auto", python_path: str = "") -> dict[str, Any]:
    jax_info = _safe_jax_info()
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(repo_root),
        "environment_kind": _detect_environment_kind(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "python": {
            "executable": python_path or sys.executable,
            "version": sys.version.split()[0],
        },
        "git": {
            "commit": _git_commit(repo_root),
        },
        "jax": {
            "requested_mode": jax_mode,
            **jax_info,
        },
        "env": {
            "JAX_PLATFORMS": os.getenv("JAX_PLATFORMS", ""),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.getenv("XLA_PYTHON_CLIENT_PREALLOCATE", ""),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": os.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION", ""),
            "ARB_C_REF_DIR": os.getenv("ARB_C_REF_DIR", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
        },
    }


def write_runtime_manifest(outdir: Path, manifest: dict[str, Any], *, filename: str = "runtime_manifest.json") -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
