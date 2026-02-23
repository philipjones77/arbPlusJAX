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

from arbjax import arb_mat


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
    lib = _find_lib(build_dir, ["arb_mat_ref.dll", "libarb_mat_ref.dll", "libarb_mat_ref.so", "libarb_mat_ref.dylib"])
    return di, lib


def load_lib(di_path: Path, lib_path: Path):
    ctypes.CDLL(str(di_path))
    lib = ctypes.CDLL(str(lib_path))
    for fn_name in ("arb_mat_2x2_det_ref", "arb_mat_2x2_trace_ref"):
        fn = getattr(lib, fn_name)
        fn.argtypes = [ctypes.POINTER(DI)]
        fn.restype = DI
    return lib


def random_intervals(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    a = rng.uniform(lo, hi, size=n)
    b = rng.uniform(lo, hi, size=n)
    low = np.minimum(a, b)
    high = np.maximum(a, b)
    return np.stack([low, high], axis=-1).astype(np.float64)


def random_mats(rng: np.random.Generator, n: int) -> np.ndarray:
    return random_intervals(rng, n * 4, -0.5, 0.5).reshape(n, 2, 2, 2)


def call_unary(lib, fn_name: str, mats: np.ndarray) -> np.ndarray:
    fn = getattr(lib, fn_name)
    out = np.empty((mats.shape[0], 2), dtype=np.float64)
    for i in range(mats.shape[0]):
        buf = (DI * 4)()
        flat = mats[i].reshape(4, 2)
        for k in range(4):
            buf[k].a = float(flat[k, 0])
            buf[k].b = float(flat[k, 1])
        r = fn(buf)
        out[i, 0] = r.a
        out[i, 1] = r.b
    return out


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
    parser = argparse.ArgumentParser(description="Compare arb_mat C and JAX kernels.")
    parser.add_argument("--samples", type=int, default=6000)
    args = parser.parse_args()

    di_env = os.getenv("DI_REF_LIB", "")
    lib_env = os.getenv("ARB_MAT_REF_LIB", "")
    d_di, d_lib = default_paths()
    di_path = Path(di_env) if di_env else d_di
    lib_path = Path(lib_env) if lib_env else d_lib
    if di_path is None or lib_path is None or not di_path.exists() or not lib_path.exists():
        print("Reference libraries not found. Build migration/c_chassis first.")
        return 1

    lib = load_lib(di_path, lib_path)
    rng = np.random.default_rng(2222)
    mats = random_mats(rng, args.samples)

    ok = True
    ok &= report(
        "arb_mat_det",
        call_unary(lib, "arb_mat_2x2_det_ref", mats),
        np.asarray(arb_mat.arb_mat_2x2_det_batch_jit(jnp.asarray(mats))),
        rtol=5e-12,
        atol=2e-12,
    )
    ok &= report(
        "arb_mat_trace",
        call_unary(lib, "arb_mat_2x2_trace_ref", mats),
        np.asarray(arb_mat.arb_mat_2x2_trace_batch_jit(jnp.asarray(mats))),
        rtol=5e-12,
        atol=2e-12,
    )

    result = "PASS" if ok else "FAIL"
    print(f"\nresult: {result}")
    _log_run("compare_arb_mat", f"compare_arb_mat.py --samples {args.samples}", f"result={result}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
