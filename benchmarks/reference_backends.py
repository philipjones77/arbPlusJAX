from __future__ import annotations

import os
import shutil
from pathlib import Path


DEFAULT_REF_PREFIX = Path.home() / ".local" / "opt" / "arbplusjax_refs"


def reference_prefix() -> Path:
    raw = os.getenv("ARBPLUSJAX_REF_PREFIX", "")
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_REF_PREFIX.resolve()


def flint_root() -> Path | None:
    env = os.getenv("FLINT_ROOT", "")
    if env:
        path = Path(env).expanduser()
        return path if path.exists() else None
    path = reference_prefix() / "flint" / "current"
    return path if path.exists() else None


def boost_root() -> Path | None:
    env = os.getenv("BOOST_ROOT", "")
    if env:
        path = Path(env).expanduser()
        return path if path.exists() else None
    path = reference_prefix() / "boost" / "current"
    return path if path.exists() else None


def wolfram_linux_dir() -> Path | None:
    env = os.getenv("WOLFRAM_LINUX_DIR", "")
    if env:
        path = Path(env).expanduser()
        return path if path.exists() else None
    exe = shutil.which("wolframscript")
    if exe:
        path = Path(exe).resolve().parent.parent
        return path if path.exists() else None
    default = Path.home() / "Wolfram" / "14.3" / "Executables"
    if default.exists():
        return default
    return None


def default_boost_ref_cmd(repo_root: Path) -> str:
    wrapper = repo_root / "benchmarks" / "run_boost_ref_adapter.sh"
    if wrapper.exists():
        return str(wrapper)
    return ""


def _prepend_env_path(name: str, value: Path | None) -> None:
    if value is None or not value.exists():
        return
    current = os.getenv(name, "")
    parts = [str(value)]
    if current:
        parts.append(current)
    os.environ[name] = ":".join(parts)


def apply_reference_env(repo_root: Path | None = None) -> None:
    prefix = reference_prefix()
    os.environ.setdefault("ARBPLUSJAX_REF_PREFIX", str(prefix))

    flint = flint_root()
    if flint is not None:
        os.environ.setdefault("FLINT_ROOT", str(flint))
        _prepend_env_path("LD_LIBRARY_PATH", flint / "lib")
        _prepend_env_path("LIBRARY_PATH", flint / "lib")
        _prepend_env_path("CPATH", flint / "include")

    boost = boost_root()
    if boost is not None:
        os.environ.setdefault("BOOST_ROOT", str(boost))
        os.environ.setdefault("BOOST_INCLUDEDIR", str(boost / "include"))
        os.environ.setdefault("BOOST_LIBRARYDIR", str(boost / "lib"))
        _prepend_env_path("LD_LIBRARY_PATH", boost / "lib")
        _prepend_env_path("LIBRARY_PATH", boost / "lib")
        _prepend_env_path("CPATH", boost / "include")

    wolfram = wolfram_linux_dir()
    if wolfram is not None:
        os.environ.setdefault("WOLFRAM_LINUX_DIR", str(wolfram))

    if repo_root is not None:
        os.environ.setdefault("BOOST_REF_CMD", default_boost_ref_cmd(repo_root))
