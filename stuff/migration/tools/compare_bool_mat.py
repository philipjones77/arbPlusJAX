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

from arbjax import bool_mat


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
        ["bool_mat_ref.dll", "libbool_mat_ref.dll", "libbool_mat_ref.so", "libbool_mat_ref.dylib"],
    )


def _load_lib(lib_path: Path):
    lib = ctypes.CDLL(str(lib_path))
    lib.bool_mat_2x2_det_batch_ref.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.bool_mat_2x2_trace_batch_ref.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
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
    parser = argparse.ArgumentParser(description="Compare bool_mat C and JAX kernels.")
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()

    lib_env = os.getenv("BOOL_MAT_REF_LIB", "")
    lib_path = Path(lib_env) if lib_env else _default_lib_path()
    if lib_path is None or not lib_path.exists():
        print("Reference libraries not found. Build migration/c_chassis first.")
        return 1

    lib = _load_lib(lib_path)
    rng = np.random.default_rng(1337)
    mats = rng.integers(0, 2, size=(args.samples, 4), dtype=np.uint8)
    flat = np.ascontiguousarray(mats.reshape(-1))

    det_out = np.empty(args.samples, dtype=np.uint8)
    tr_out = np.empty(args.samples, dtype=np.uint8)
    lib.bool_mat_2x2_det_batch_ref(
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        det_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        args.samples,
    )
    lib.bool_mat_2x2_trace_batch_ref(
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        tr_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        args.samples,
    )

    j_det = np.asarray(bool_mat.bool_mat_2x2_det_batch_jit(jnp.asarray(mats)))
    j_tr = np.asarray(bool_mat.bool_mat_2x2_trace_batch_jit(jnp.asarray(mats)))

    ok_det = np.array_equal(det_out, j_det)
    ok_tr = np.array_equal(tr_out, j_tr)
    ok = ok_det and ok_tr

    print(f"bool_mat det | ok={ok_det}")
    print(f"bool_mat trace | ok={ok_tr}")
    result = "PASS" if ok else "FAIL"
    print(f"\nresult: {result}")
    _log_run("compare_bool_mat", f"compare_bool_mat.py --samples {args.samples}", f"result={result}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
