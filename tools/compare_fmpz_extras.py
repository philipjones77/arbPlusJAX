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

from arbplusjax import fmpz_extras


class FMPZ(ctypes.Structure):
    _fields_ = [("v", ctypes.c_int64)]


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
    return _find_lib(
        build_dir,
        [
            "fmpz_extras_ref.dll",
            "libfmpz_extras_ref.dll",
            "libfmpz_extras_ref.so",
            "libfmpz_extras_ref.dylib",
        ],
    )


def _load_lib(lib_path: Path):
    lib = ctypes.CDLL(str(lib_path))
    for fn_name in ("fmpz_extras_add_ref", "fmpz_extras_mul_ref"):
        fn = getattr(lib, fn_name)
        fn.argtypes = [FMPZ, FMPZ]
        fn.restype = FMPZ
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
    parser = argparse.ArgumentParser(description="Compare fmpz_extras C and JAX kernels.")
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()

    lib_env = os.getenv("FMPZ_EXTRAS_REF_LIB", "")
    lib_path = Path(lib_env) if lib_env else _default_lib_path()
    if lib_path is None or not lib_path.exists():
        print("Reference libraries not found. Build C reference libraries in the Arb workspace first.")
        return 1

    lib = _load_lib(lib_path)
    rng = np.random.default_rng(2310)
    a = rng.integers(-1000, 1000, size=args.samples, dtype=np.int64)
    b = rng.integers(-1000, 1000, size=args.samples, dtype=np.int64)

    out_add = np.empty_like(a)
    out_mul = np.empty_like(a)
    add_fn = lib.fmpz_extras_add_ref
    mul_fn = lib.fmpz_extras_mul_ref
    for i in range(args.samples):
        out_add[i] = add_fn(FMPZ(int(a[i])), FMPZ(int(b[i]))).v
        out_mul[i] = mul_fn(FMPZ(int(a[i])), FMPZ(int(b[i]))).v

    j_add = np.asarray(fmpz_extras.fmpz_extras_add_batch_jit(jnp.asarray(a), jnp.asarray(b)))
    j_mul = np.asarray(fmpz_extras.fmpz_extras_mul_batch_jit(jnp.asarray(a), jnp.asarray(b)))

    ok_add = np.array_equal(out_add, j_add)
    ok_mul = np.array_equal(out_mul, j_mul)
    ok = ok_add and ok_mul

    print(f"fmpz_extras_add | ok={ok_add}")
    print(f"fmpz_extras_mul | ok={ok_mul}")
    result = "PASS" if ok else "FAIL"
    print(f"\nresult: {result}")
    _log_run("compare_fmpz_extras", f"compare_fmpz_extras.py --samples {args.samples}", f"result={result}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
