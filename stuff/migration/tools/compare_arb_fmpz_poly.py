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

from arbjax import arb_fmpz_poly


class DI(ctypes.Structure):
    _fields_ = [("a", ctypes.c_double), ("b", ctypes.c_double)]


def _find_lib(build_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        matches = list(build_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def default_paths() -> tuple[Path | None, Path | None]:
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = repo_root / "migration" / "c_chassis" / "build"
    if not build_dir.exists():
        return None, None

    di = _find_lib(build_dir, ["double_interval_ref.dll", "libdouble_interval_ref.dll", "libdouble_interval_ref.so", "libdouble_interval_ref.dylib"])
    lib = _find_lib(build_dir, ["arb_fmpz_poly_ref.dll", "libarb_fmpz_poly_ref.dll", "libarb_fmpz_poly_ref.so", "libarb_fmpz_poly_ref.dylib"])
    return di, lib


def load_lib(di_path: Path, lib_path: Path):
    ctypes.CDLL(str(di_path))
    lib = ctypes.CDLL(str(lib_path))
    fn = lib.arb_fmpz_poly_eval_cubic_ref
    fn.argtypes = [ctypes.POINTER(DI), DI]
    fn.restype = DI
    return lib


def random_intervals(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    a = rng.uniform(lo, hi, size=n)
    b = rng.uniform(lo, hi, size=n)
    low = np.minimum(a, b)
    high = np.maximum(a, b)
    return np.stack([low, high], axis=-1).astype(np.float64)


def report(name: str, c_out: np.ndarray, j_out: np.ndarray, rtol: float, atol: float = 0.0) -> bool:
    ok = np.allclose(c_out, j_out, rtol=rtol, atol=atol, equal_nan=True)
    with np.errstate(invalid="ignore"):
        diff = np.abs(c_out - j_out)
    finite = np.isfinite(diff)
    max_diff = float(np.max(diff[finite])) if np.any(finite) else 0.0
    print(f"{name:16s} | ok={ok} | max_abs_diff={max_diff:.3e}")
    return ok


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
    parser = argparse.ArgumentParser(description="Compare arb_fmpz_poly C and JAX kernels.")
    parser.add_argument("--samples", type=int, default=6000)
    args = parser.parse_args()

    di_env = os.getenv("DI_REF_LIB", "")
    lib_env = os.getenv("ARB_FMPZ_POLY_REF_LIB", "")
    d_di, d_lib = default_paths()
    di_path = Path(di_env) if di_env else d_di
    lib_path = Path(lib_env) if lib_env else d_lib
    if di_path is None or lib_path is None or not di_path.exists() or not lib_path.exists():
        print("Reference libraries not found. Build migration/c_chassis first.")
        return 1

    lib = load_lib(di_path, lib_path)
    rng = np.random.default_rng(2202)
    coeffs = random_intervals(rng, 4 * args.samples, -0.5, 0.5).reshape(args.samples, 4, 2)
    x = random_intervals(rng, args.samples, -0.3, 0.3)

    out_c = np.empty((args.samples, 2), dtype=np.float64)
    fn = lib.arb_fmpz_poly_eval_cubic_ref
    for i in range(args.samples):
        buf = (DI * 4)()
        for k in range(4):
            buf[k].a = float(coeffs[i, k, 0])
            buf[k].b = float(coeffs[i, k, 1])
        r = fn(buf, DI(float(x[i, 0]), float(x[i, 1])))
        out_c[i, 0] = r.a
        out_c[i, 1] = r.b

    out_j = np.asarray(arb_fmpz_poly.arb_fmpz_poly_eval_cubic_batch_jit(jnp.asarray(coeffs), jnp.asarray(x)))
    ok = report("arb_fmpz_poly_cubic", out_c, out_j, rtol=5e-12, atol=2e-12)

    result = "PASS" if ok else "FAIL"
    print(f"\nresult: {result}")
    _log_run("compare_arb_fmpz_poly", f"compare_arb_fmpz_poly.py --samples {args.samples}", f"result={result}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
