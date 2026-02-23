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

from arbjax import arf


class ARF(ctypes.Structure):
    _fields_ = [("v", ctypes.c_double)]


def _find_lib(build_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        matches = list(build_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def default_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = repo_root / "migration" / "c_chassis" / "build"
    if not build_dir.exists():
        return None
    return _find_lib(build_dir, ["arf_ref.dll", "libarf_ref.dll", "libarf_ref.so", "libarf_ref.dylib"])


def load_lib(lib_path: Path):
    lib = ctypes.CDLL(str(lib_path))
    for fn_name in ("arf_add_ref", "arf_mul_ref"):
        fn = getattr(lib, fn_name)
        fn.argtypes = [ARF, ARF]
        fn.restype = ARF
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
    parser = argparse.ArgumentParser(description="Compare arf C and JAX kernels.")
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()

    lib_env = os.getenv("ARF_REF_LIB", "")
    lib_path = Path(lib_env) if lib_env else default_path()
    if lib_path is None or not lib_path.exists():
        print("Reference libraries not found. Build migration/c_chassis first.")
        return 1

    lib = load_lib(lib_path)
    rng = np.random.default_rng(2242)
    a = rng.normal(size=args.samples)
    b = rng.normal(size=args.samples)

    out_add = np.empty_like(a)
    out_mul = np.empty_like(a)
    add_fn = lib.arf_add_ref
    mul_fn = lib.arf_mul_ref
    for i in range(args.samples):
        r = add_fn(ARF(float(a[i])), ARF(float(b[i])))
        out_add[i] = r.v
        r = mul_fn(ARF(float(a[i])), ARF(float(b[i])))
        out_mul[i] = r.v

    j_add = np.asarray(arf.arf_add_batch_jit(jnp.asarray(a), jnp.asarray(b)))
    j_mul = np.asarray(arf.arf_mul_batch_jit(jnp.asarray(a), jnp.asarray(b)))

    ok = np.allclose(out_add, j_add) and np.allclose(out_mul, j_mul)
    print(f"arf_add | ok={np.allclose(out_add, j_add)}")
    print(f"arf_mul | ok={np.allclose(out_mul, j_mul)}")

    result = "PASS" if ok else "FAIL"
    print(f"\nresult: {result}")
    _log_run("compare_arf", f"compare_arf.py --samples {args.samples}", f"result={result}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
