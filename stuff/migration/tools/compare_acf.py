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

from arbjax import acf


class ACF(ctypes.Structure):
    _fields_ = [("re", ctypes.c_double), ("im", ctypes.c_double)]


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
    return _find_lib(build_dir, ["acf_ref.dll", "libacf_ref.dll", "libacf_ref.so", "libacf_ref.dylib"])


def load_lib(lib_path: Path):
    lib = ctypes.CDLL(str(lib_path))
    for fn_name in ("acf_add_ref", "acf_mul_ref"):
        fn = getattr(lib, fn_name)
        fn.argtypes = [ACF, ACF]
        fn.restype = ACF
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
    parser = argparse.ArgumentParser(description="Compare acf C and JAX kernels.")
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()

    lib_env = os.getenv("ACF_REF_LIB", "")
    lib_path = Path(lib_env) if lib_env else default_path()
    if lib_path is None or not lib_path.exists():
        print("Reference libraries not found. Build migration/c_chassis first.")
        return 1

    lib = load_lib(lib_path)
    rng = np.random.default_rng(2182)
    a = rng.normal(size=args.samples) + 1j * rng.normal(size=args.samples)
    b = rng.normal(size=args.samples) + 1j * rng.normal(size=args.samples)

    out_add = np.empty(args.samples, dtype=np.complex128)
    out_mul = np.empty(args.samples, dtype=np.complex128)
    add_fn = lib.acf_add_ref
    mul_fn = lib.acf_mul_ref
    for i in range(args.samples):
        r = add_fn(ACF(float(a[i].real), float(a[i].imag)), ACF(float(b[i].real), float(b[i].imag)))
        out_add[i] = r.re + 1j * r.im
        r = mul_fn(ACF(float(a[i].real), float(a[i].imag)), ACF(float(b[i].real), float(b[i].imag)))
        out_mul[i] = r.re + 1j * r.im

    j_add = np.asarray(acf.acf_add_batch_jit(jnp.asarray(a), jnp.asarray(b)))
    j_mul = np.asarray(acf.acf_mul_batch_jit(jnp.asarray(a), jnp.asarray(b)))

    ok = np.allclose(out_add, j_add) and np.allclose(out_mul, j_mul)
    print(f"acf_add | ok={np.allclose(out_add, j_add)}")
    print(f"acf_mul | ok={np.allclose(out_mul, j_mul)}")

    result = "PASS" if ok else "FAIL"
    print(f"\nresult: {result}")
    _log_run("compare_acf", f"compare_acf.py --samples {args.samples}", f"result={result}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
