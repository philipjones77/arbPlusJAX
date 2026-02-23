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

from arbjax import acb_modular


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


def _random_boxes(rng: np.random.Generator, n: int) -> np.ndarray:
    re = rng.uniform(0.0, 0.5, size=(n, 2))
    im = rng.uniform(0.5, 1.2, size=(n, 2))
    re_lo = np.minimum(re[:, 0], re[:, 1])
    re_hi = np.maximum(re[:, 0], re[:, 1])
    im_lo = np.minimum(im[:, 0], im[:, 1])
    im_hi = np.maximum(im[:, 0], im[:, 1])
    return np.stack([re_lo, re_hi, im_lo, im_hi], axis=-1).astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark acb_modular JAX kernels.")
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()

    rng = np.random.default_rng(2163)
    tau = jnp.asarray(_random_boxes(rng, args.samples))

    fn = jax.jit(acb_modular.acb_modular_j_batch)
    fn(tau).block_until_ready()
    t0 = time.perf_counter()
    out = fn(tau)
    out.block_until_ready()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0

    print(f"acb_modular (j) | samples={args.samples} | time_ms={ms:.2f}")
    _log_run(
        "benchmark_acb_modular",
        f"benchmark_acb_modular.py --samples {args.samples}",
        f"time_ms={ms:.2f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
