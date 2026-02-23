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

from arbplusjax import bernoulli


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
    return _find_lib(build_dir, ["bernoulli_ref.dll", "libbernoulli_ref.dll", "libbernoulli_ref.so", "libbernoulli_ref.dylib"])


def load_lib(lib_path: Path):
    lib = ctypes.CDLL(str(lib_path))
    fn = lib.bernoulli_number_ref
    fn.argtypes = [ctypes.c_int]
    fn.restype = ctypes.c_double
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
    parser = argparse.ArgumentParser(description="Compare bernoulli C and JAX kernels.")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    lib_env = os.getenv("BERNOULLI_REF_LIB", "")
    lib_path = Path(lib_env) if lib_env else default_path()
    if lib_path is None or not lib_path.exists():
        print("Reference libraries not found. Build C reference libraries in the Arb workspace first.")
        return 1

    lib = load_lib(lib_path)
    n = np.array([0, 1, 2, 4, 6], dtype=np.int64)[: args.samples]
    out_c = np.array([lib.bernoulli_number_ref(int(k)) for k in n], dtype=np.float64)
    out_j = np.asarray(bernoulli.bernoulli_number_batch_jit(jnp.asarray(n)))

    ok = np.allclose(out_c, out_j)
    print(f"bernoulli | ok={ok}")
    result = "PASS" if ok else "FAIL"
    print(f"\nresult: {result}")
    _log_run("compare_bernoulli", f"compare_bernoulli.py --samples {args.samples}", f"result={result}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
