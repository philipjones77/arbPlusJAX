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

from arbplusjax import arb_fpwrap


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
    parser = argparse.ArgumentParser(description="Benchmark arb_fpwrap JAX kernels.")
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--which", type=str, default="exp", choices=["exp", "log"])
    parser.add_argument("--complex", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(2213)
    if args.complex:
        z = jnp.asarray(rng.normal(size=args.samples) + 1j * rng.normal(size=args.samples))
        fn = jax.jit(arb_fpwrap.arb_fpwrap_cdouble_exp if args.which == "exp" else arb_fpwrap.arb_fpwrap_cdouble_log)
        fn(z).block_until_ready()
        t0 = time.perf_counter()
        out = fn(z)
        out.block_until_ready()
    else:
        x = jnp.asarray(rng.uniform(0.1, 2.0, size=args.samples))
        fn = jax.jit(arb_fpwrap.arb_fpwrap_double_exp if args.which == "exp" else arb_fpwrap.arb_fpwrap_double_log)
        fn(x).block_until_ready()
        t0 = time.perf_counter()
        out = fn(x)
        out.block_until_ready()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0

    label = f"{'c' if args.complex else 'r'}{args.which}"
    print(f"arb_fpwrap ({label}) | samples={args.samples} | time_ms={ms:.2f}")
    _log_run(
        \"benchmark_arb_fpwrap\",
        f\"benchmark_arb_fpwrap.py --samples {args.samples} --which {args.which}{' --complex' if args.complex else ''}\",
        f\"time_ms={ms:.2f}\",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
