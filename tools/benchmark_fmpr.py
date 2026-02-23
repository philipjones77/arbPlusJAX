from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from arbplusjax import fmpr


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
    parser = argparse.ArgumentParser(description="Benchmark fmpr JAX kernels.")
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--which", type=str, default="mul", choices=["add", "mul"])
    args = parser.parse_args()

    rng = np.random.default_rng(2253)
    a = jnp.asarray(rng.normal(size=args.samples))
    b = jnp.asarray(rng.normal(size=args.samples))

    if args.which == "add":
        fn = jax.jit(fmpr.fmpr_add_batch)
    else:
        fn = jax.jit(fmpr.fmpr_mul_batch)

    fn(a, b).block_until_ready()
    t0 = time.perf_counter()
    out = fn(a, b)
    out.block_until_ready()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0

    print(f"fmpr ({args.which}) | samples={args.samples} | time_ms={ms:.2f}")
    _log_run(
        "benchmark_fmpr",
        f"benchmark_fmpr.py --samples {args.samples} --which {args.which}",
        f"time_ms={ms:.2f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
