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

from arbjax import dirichlet


class DI(ctypes.Structure):
    _fields_ = [("a", ctypes.c_double), ("b", ctypes.c_double)]


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
        ["dirichlet_ref.dll", "libdirichlet_ref.dll", "libdirichlet_ref.so", "libdirichlet_ref.dylib"],
    )


def _load_lib(lib_path: Path):
    lib = ctypes.CDLL(str(lib_path))
    lib.dirichlet_zeta_batch_ref.argtypes = [
        ctypes.POINTER(DI),
        ctypes.POINTER(DI),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.dirichlet_eta_batch_ref.argtypes = [
        ctypes.POINTER(DI),
        ctypes.POINTER(DI),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
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
    parser = argparse.ArgumentParser(description="Compare dirichlet C and JAX kernels.")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--terms", type=int, default=32)
    parser.add_argument("--which", type=str, default="zeta", choices=["zeta", "eta"])
    args = parser.parse_args()

    lib_env = os.getenv("DIRICHLET_REF_LIB", "")
    lib_path = Path(lib_env) if lib_env else _default_lib_path()
    if lib_path is None or not lib_path.exists():
        print("Reference libraries not found. Build migration/c_chassis first.")
        return 1

    lib = _load_lib(lib_path)
    rng = np.random.default_rng(1711)
    mid = rng.uniform(1.2, 4.0, size=args.samples)
    half = rng.uniform(0.0, 0.2, size=args.samples)
    lo = mid - half
    hi = mid + half
    s = np.stack([lo, hi], axis=-1).astype(np.float64)

    s_struct = (DI * args.samples)(*([DI(float(a), float(b)) for a, b in s]))
    out = (DI * args.samples)()

    if args.which == "zeta":
        lib.dirichlet_zeta_batch_ref(s_struct, out, args.samples, args.terms)
        j_out = np.asarray(dirichlet.dirichlet_zeta_batch_jit(jnp.asarray(s), n_terms=args.terms))
    else:
        lib.dirichlet_eta_batch_ref(s_struct, out, args.samples, args.terms)
        j_out = np.asarray(dirichlet.dirichlet_eta_batch_jit(jnp.asarray(s), n_terms=args.terms))

    c_out = np.array([[out[i].a, out[i].b] for i in range(args.samples)], dtype=np.float64)
    ok = np.allclose(c_out, j_out, rtol=1e-12, atol=0.0, equal_nan=True)

    print(f"dirichlet_{args.which} | ok={ok}")
    result = "PASS" if ok else "FAIL"
    print(f"\nresult: {result}")
    _log_run(
        "compare_dirichlet",
        f"compare_dirichlet.py --samples {args.samples} --terms {args.terms} --which {args.which}",
        f"result={result}",
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
