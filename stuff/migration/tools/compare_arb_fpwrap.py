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

from arbjax import arb_fpwrap


class CDouble(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]


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
    return _find_lib(build_dir, ["arb_fpwrap_ref.dll", "libarb_fpwrap_ref.dll", "libarb_fpwrap_ref.so", "libarb_fpwrap_ref.dylib"])


def load_lib(lib_path: Path):
    lib = ctypes.CDLL(str(lib_path))
    lib.arb_fpwrap_double_exp_ref.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_double]
    lib.arb_fpwrap_double_exp_ref.restype = ctypes.c_int
    lib.arb_fpwrap_double_log_ref.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_double]
    lib.arb_fpwrap_double_log_ref.restype = ctypes.c_int
    lib.arb_fpwrap_cdouble_exp_ref.argtypes = [ctypes.POINTER(CDouble), CDouble]
    lib.arb_fpwrap_cdouble_exp_ref.restype = ctypes.c_int
    lib.arb_fpwrap_cdouble_log_ref.argtypes = [ctypes.POINTER(CDouble), CDouble]
    lib.arb_fpwrap_cdouble_log_ref.restype = ctypes.c_int
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
    parser = argparse.ArgumentParser(description="Compare arb_fpwrap C and JAX kernels.")
    parser.add_argument("--samples", type=int, default=6000)
    args = parser.parse_args()

    lib_env = os.getenv("ARB_FPWRAP_REF_LIB", "")
    lib_path = Path(lib_env) if lib_env else default_path()
    if lib_path is None or not lib_path.exists():
        print("Reference libraries not found. Build migration/c_chassis first.")
        return 1

    lib = load_lib(lib_path)
    rng = np.random.default_rng(2212)
    x = rng.uniform(0.1, 2.0, size=args.samples)
    z = rng.normal(size=args.samples) + 1j * rng.normal(size=args.samples)

    out_exp = np.empty_like(x)
    out_log = np.empty_like(x)
    for i in range(args.samples):
        v = ctypes.c_double()
        lib.arb_fpwrap_double_exp_ref(ctypes.byref(v), float(x[i]))
        out_exp[i] = v.value
        v2 = ctypes.c_double()
        lib.arb_fpwrap_double_log_ref(ctypes.byref(v2), float(x[i]))
        out_log[i] = v2.value

    j_exp = np.asarray(arb_fpwrap.arb_fpwrap_double_exp_jit(jnp.asarray(x)))
    j_log = np.asarray(arb_fpwrap.arb_fpwrap_double_log_jit(jnp.asarray(x)))

    ok = np.allclose(out_exp, j_exp) and np.allclose(out_log, j_log)
    print(f"arb_fpwrap_exp | ok={np.allclose(out_exp, j_exp)}")
    print(f"arb_fpwrap_log | ok={np.allclose(out_log, j_log)}")

    out_c_exp = np.empty_like(z)
    out_c_log = np.empty_like(z)
    for i in range(args.samples):
        c = CDouble(float(z[i].real), float(z[i].imag))
        r = CDouble()
        lib.arb_fpwrap_cdouble_exp_ref(ctypes.byref(r), c)
        out_c_exp[i] = r.real + 1j * r.imag
        r2 = CDouble()
        lib.arb_fpwrap_cdouble_log_ref(ctypes.byref(r2), c)
        out_c_log[i] = r2.real + 1j * r2.imag

    j_c_exp = np.asarray(arb_fpwrap.arb_fpwrap_cdouble_exp_jit(jnp.asarray(z)))
    j_c_log = np.asarray(arb_fpwrap.arb_fpwrap_cdouble_log_jit(jnp.asarray(z)))
    ok = ok and np.allclose(out_c_exp, j_c_exp) and np.allclose(out_c_log, j_c_log)
    print(f"arb_fpwrap_c_exp | ok={np.allclose(out_c_exp, j_c_exp)}")
    print(f"arb_fpwrap_c_log | ok={np.allclose(out_c_log, j_c_log)}")

    result = "PASS" if ok else "FAIL"
    print(f"\nresult: {result}")
    _log_run("compare_arb_fpwrap", f"compare_arb_fpwrap.py --samples {args.samples}", f"result={result}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
