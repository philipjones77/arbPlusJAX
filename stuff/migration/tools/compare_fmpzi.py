from __future__ import annotations

import argparse
import ctypes
import os
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess

import jax.numpy as jnp
import numpy as np

from arbjax import fmpzi


class FMPZI(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_int64), ("hi", ctypes.c_int64)]


def _find_lib(build_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        matches = list(build_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def _default_lib_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = repo_root / "migration" / "c_chassis" / "build"
    if not build_dir.exists():
        return None
    return _find_lib(build_dir, ["fmpzi_ref.dll", "libfmpzi_ref.dll", "libfmpzi_ref.so", "libfmpzi_ref.dylib"])


def _load_lib(lib_path: Path):
    lib = ctypes.CDLL(str(lib_path))
    lib.fmpzi_add_ref.argtypes = [FMPZI, FMPZI]
    lib.fmpzi_add_ref.restype = FMPZI
    lib.fmpzi_sub_ref.argtypes = [FMPZI, FMPZI]
    lib.fmpzi_sub_ref.restype = FMPZI
    return lib


def _git_commit(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        )
    except Exception:
        return "unknown"


def _log_run(tool: str, command: str, notes: str = "") -> None:
    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "migration" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "runs.csv"
    if not log_path.exists():
        log_path.write_text("run_id,timestamp_utc,tool,command,commit,platform,notes\n", encoding="ascii")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = f"{tool}-{timestamp}"
    commit = _git_commit(repo_root)
    plat = platform.platform()
    line = f"{run_id},{timestamp},{tool},\"{command}\",{commit},\"{plat}\",\"{notes}\"\n"
    with log_path.open("a", encoding="ascii") as f:
        f.write(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare fmpzi C and JAX kernels.")
    parser.add_argument("--samples", type=int, default=6000)
    args = parser.parse_args()

    lib_env = os.getenv("FMPZI_REF_LIB", "")
    lib_path = Path(lib_env) if lib_env else _default_lib_path()
    if lib_path is None or not lib_path.exists():
        print("Reference libraries not found. Build migration/c_chassis first.")
        return 1

    lib = _load_lib(lib_path)
    rng = np.random.default_rng(2312)
    lo = rng.integers(-50, 50, size=args.samples, dtype=np.int64)
    hi = lo + rng.integers(0, 50, size=args.samples, dtype=np.int64)
    lo2 = rng.integers(-50, 50, size=args.samples, dtype=np.int64)
    hi2 = lo2 + rng.integers(0, 50, size=args.samples, dtype=np.int64)
    a = np.stack([lo, hi], axis=-1)
    b = np.stack([lo2, hi2], axis=-1)

    out_add = np.empty_like(a)
    out_sub = np.empty_like(a)
    add_fn = lib.fmpzi_add_ref
    sub_fn = lib.fmpzi_sub_ref
    for i in range(args.samples):
        r_add = add_fn(FMPZI(int(a[i, 0]), int(a[i, 1])), FMPZI(int(b[i, 0]), int(b[i, 1])))
        r_sub = sub_fn(FMPZI(int(a[i, 0]), int(a[i, 1])), FMPZI(int(b[i, 0]), int(b[i, 1])))
        out_add[i, 0] = r_add.lo
        out_add[i, 1] = r_add.hi
        out_sub[i, 0] = r_sub.lo
        out_sub[i, 1] = r_sub.hi

    j_add = np.asarray(fmpzi.fmpzi_add_batch_jit(jnp.asarray(a), jnp.asarray(b)))
    j_sub = np.asarray(fmpzi.fmpzi_sub_batch_jit(jnp.asarray(a), jnp.asarray(b)))

    ok_add = np.array_equal(out_add, j_add)
    ok_sub = np.array_equal(out_sub, j_sub)
    ok = ok_add and ok_sub

    print(f"fmpzi_add | ok={ok_add}")
    print(f"fmpzi_sub | ok={ok_sub}")
    result = "PASS" if ok else "FAIL"
    print(f"\nresult: {result}")
    _log_run("compare_fmpzi", f"compare_fmpzi.py --samples {args.samples}", f"result={result}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
