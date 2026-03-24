from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path


def _system_name() -> str:
    return platform.system().lower()


def _is_windows() -> bool:
    return _system_name().startswith("win")


def _is_linux() -> bool:
    return _system_name() == "linux"


def _candidates() -> list[Path]:
    out: list[Path] = []
    home = Path.home()
    if _is_windows():
        user = Path(os.environ.get("USERPROFILE", str(home)))
        out.extend(
            [
                user / "miniforge3" / "envs" / "jax" / "python.exe",
                user / "anaconda3" / "envs" / "jax" / "python.exe",
            ]
        )
    else:
        out.extend(
            [
                home / "miniforge3" / "envs" / "jax" / "bin" / "python",
                home / "anaconda3" / "envs" / "jax" / "bin" / "python",
            ]
        )
    return out


def resolve_python(explicit: str | None = None) -> str:
    # Highest priority: explicit CLI path.
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p)
        found = shutil.which(explicit)
        if found:
            return found

    # Next: environment override.
    env = os.getenv("ARBPLUSJAX_PYTHON", "").strip()
    if env:
        p = Path(env)
        if p.exists():
            return str(p)
        found = shutil.which(env)
        if found:
            return found

    # On Linux, the default interpreter policy is the shared JAX environment.
    # Prefer that well-known env before any ambient interpreter fallback so
    # harnesses/examples/benchmarks all route through the same runtime by
    # default unless the caller explicitly overrides it.
    for cand in _candidates():
        if cand.exists():
            return str(cand)

    # Next: activated conda env named jax.
    conda_prefix = os.getenv("CONDA_PREFIX", "").strip()
    conda_env = os.getenv("CONDA_DEFAULT_ENV", "").strip().lower()
    if conda_prefix and conda_env == "jax":
        if _is_windows():
            cand = Path(conda_prefix) / "python.exe"
        else:
            cand = Path(conda_prefix) / "bin" / "python"
        if cand.exists():
            return str(cand)

    # Final fallback: current interpreter.
    # Linux still falls back here when no JAX env can be resolved.
    return sys.executable


def jax_platform_env(mode: str) -> dict[str, str]:
    m = mode.strip().lower()
    if m in ("", "auto"):
        return {}
    if m == "cpu":
        return {
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_CUDA_VISIBLE_DEVICES": "",
        }
    if m == "gpu":
        return {"JAX_PLATFORMS": "cuda"}
    raise ValueError(f"Unsupported JAX mode: {mode}. Use auto|cpu|gpu.")
