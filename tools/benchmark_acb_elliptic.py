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

from arbplusjax import acb_elliptic


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
    a = rng.uniform(0.0, 0.9, size=(n, 2))
    b = rng.uniform(-0.2, 0.2, size=(n, 2))
    re = np.stack([np.minimum(a[:, 0], a[:, 1]), np.maximum(a[:, 0], a[:, 1])], axis=-1)
    im = np.stack([np.minimum(b[:, 0], b[:, 1]), np.maximum(b[:, 0], b[:, 1])], axis=-1)
    return np.concatenate([re, im], axis=-1).astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark acb_elliptic JAX kernels.")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--which", type=str, default="k", choices=["k", "e"])
    args = parser.parse_args()

    rng = np.random.default_rng(2143)
    m = jnp.asarray(_random_boxes(rng, args.samples))

    if args.which == "k":
        fn = jax.jit(acb_elliptic.acb_elliptic_k_batch)
    else:
        fn = jax.jit(acb_elliptic.acb_elliptic_e_batch)

    fn(m).block_until_ready()
    t0 = time.perf_counter()
    out = fn(m)
    out.block_until_ready()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0

    print(f"acb_elliptic ({args.which}) | samples={args.samples} | time_ms={ms:.2f}")
    _log_run(
        \"benchmark_acb_elliptic\",
        f\"benchmark_acb_elliptic.py --samples {args.samples} --which {args.which}\",
        f\"time_ms={ms:.2f}\",
    )
    return 0


if __name__ == \"__main__\":
    raise SystemExit(main())
