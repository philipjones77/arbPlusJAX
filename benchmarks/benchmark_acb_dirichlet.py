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

from arbplusjax import acb_dirichlet


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
    a = rng.uniform(0.8, 2.5, size=(n, 2))
    b = rng.uniform(-0.5, 0.5, size=(n, 2))
    re = np.stack([np.minimum(a[:, 0], a[:, 1]), np.maximum(a[:, 0], a[:, 1])], axis=-1)
    im = np.stack([np.minimum(b[:, 0], b[:, 1]), np.maximum(b[:, 0], b[:, 1])], axis=-1)
    return np.concatenate([re, im], axis=-1).astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark acb_dirichlet JAX kernels.")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--terms", type=int, default=64)
    parser.add_argument("--which", type=str, default="zeta", choices=["zeta", "eta"])
    args = parser.parse_args()

    rng = np.random.default_rng(2133)
    s = jnp.asarray(_random_boxes(rng, args.samples))

    if args.which == "zeta":
        fn = jax.jit(acb_dirichlet.acb_dirichlet_zeta_batch, static_argnames=("n_terms",))
    else:
        fn = jax.jit(acb_dirichlet.acb_dirichlet_eta_batch, static_argnames=("n_terms",))

    fn(s, n_terms=args.terms).block_until_ready()
    t0 = time.perf_counter()
    out = fn(s, n_terms=args.terms)
    out.block_until_ready()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0

    print(f"acb_dirichlet ({args.which}) | samples={args.samples} terms={args.terms} | time_ms={ms:.2f}")
    _log_run(
        "benchmark_acb_dirichlet",
        f"benchmark_acb_dirichlet.py --samples {args.samples} --terms {args.terms} --which {args.which}",
        f"time_ms={ms:.2f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
