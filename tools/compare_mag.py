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

from arbplusjax import mag


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
    return _find_lib(build_dir, ["mag_ref.dll", "libmag_ref.dll", "libmag_ref.so", "libmag_ref.dylib"])


def _load_lib(lib_path: Path):
    lib = ctypes.CDLL(str(lib_path))
    lib.mag_add_ref.argtypes = [ctypes.c_double, ctypes.c_double]
    lib.mag_add_ref.restype = ctypes.c_double
    lib.mag_mul_ref.argtypes = [ctypes.c_double, ctypes.c_double]
    lib.mag_mul_ref.restype = ctypes.c_double
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
    parser = argparse.ArgumentParser(description="Compare mag C and JAX kernels.")
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()

    lib_env = os.getenv("MAG_REF_LIB", "")
    lib_path = Path(lib_env) if lib_env else _default_lib_path()
    if lib_path is None or not lib_path.exists():
        print("Reference libraries not found. Build C reference libraries in the Arb workspace first.")
        return 1

    lib = _load_lib(lib_path)
    rng = np.random.default_rng(2314)
    a = rng.normal(size=args.samples).astype(np.float64)
    b = rng.normal(size=args.samples).astype(np.float64)

    out_add = np.empty_like(a)
    out_mul = np.empty_like(a)
    add_fn = lib.mag_add_ref
    mul_fn = lib.mag_mul_ref
    for i in range(args.samples):
        out_add[i] = add_fn(float(a[i]), float(b[i]))
        out_mul[i] = mul_fn(float(a[i]), float(b[i]))

    j_add = np.asarray(mag.mag_add_batch_jit(jnp.asarray(a), jnp.asarray(b)))
    j_mul = np.asarray(mag.mag_mul_batch_jit(jnp.asarray(a), jnp.asarray(b)))

    ok_add = np.allclose(out_add, j_add, rtol=0.0, atol=0.0)
    ok_mul = np.allclose(out_mul, j_mul, rtol=0.0, atol=0.0)
    ok = ok_add and ok_mul

    print(f"mag_add | ok={ok_add}")
    print(f"mag_mul | ok={ok_mul}")
    result = "PASS" if ok else "FAIL"
    print(f"\nresult: {result}")
    _log_run("compare_mag", f"compare_mag.py --samples {args.samples}", f"result={result}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
