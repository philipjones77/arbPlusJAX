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

from arbplusjax import arb_fmpz_poly


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


def _random_intervals(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    a = rng.uniform(lo, hi, size=n)
    b = rng.uniform(lo, hi, size=n)
    low = np.minimum(a, b)
    high = np.maximum(a, b)
    return np.stack([low, high], axis=-1).astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark arb_fmpz_poly JAX kernels.")
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()

    rng = np.random.default_rng(2203)
    coeffs = jnp.asarray(_random_intervals(rng, 4 * args.samples, -0.5, 0.5).reshape(args.samples, 4, 2))
    x = jnp.asarray(_random_intervals(rng, args.samples, -0.3, 0.3))

    fn = jax.jit(arb_fmpz_poly.arb_fmpz_poly_eval_cubic_batch)
    fn(coeffs, x).block_until_ready()
    t0 = time.perf_counter()
    out = fn(coeffs, x)
    out.block_until_ready()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0

    print(f"arb_fmpz_poly (cubic) | samples={args.samples} | time_ms={ms:.2f}")
    _log_run(
        "benchmark_arb_fmpz_poly",
        f"benchmark_arb_fmpz_poly.py --samples {args.samples}",
        f"time_ms={ms:.2f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
