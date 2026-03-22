from __future__ import annotations

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


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
from benchmarks.schema import BenchmarkRecord
from benchmarks.schema import BenchmarkReport
from benchmarks.schema import write_benchmark_report
from tools.runtime_manifest import collect_runtime_manifest


def _git_commit(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        )
    except Exception:
        return "unknown"


def _log_run(tool: str, command: str, notes: str = "") -> None:
    repo_root = Path(__file__).resolve().parents[1]
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
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/benchmarks/outputs/scalar/benchmark_fmpr.json"),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(2253)
    a = jnp.asarray(rng.normal(size=args.samples))
    b = jnp.asarray(rng.normal(size=args.samples))

    if args.which == "add":
        fn = jax.jit(fmpr.fmpr_add_batch)
    else:
        fn = jax.jit(fmpr.fmpr_mul_batch)

    t0 = time.perf_counter()
    fn(a, b).block_until_ready()
    t1 = time.perf_counter()
    warm_times: list[float] = []
    for _ in range(args.runs):
        s0 = time.perf_counter()
        out = fn(a, b)
        out.block_until_ready()
        warm_times.append(time.perf_counter() - s0)
    alt_a = a[: max(8, args.samples // 2)]
    alt_b = b[: max(8, args.samples // 2)]
    t2 = time.perf_counter()
    fn(alt_a, alt_b).block_until_ready()
    t3 = time.perf_counter()
    ms = (sum(warm_times) / len(warm_times)) * 1000.0

    report = BenchmarkReport(
        benchmark_name="benchmark_fmpr.py",
        concern="scalar_speed",
        category="scalar",
        records=(
            BenchmarkRecord(
                benchmark_name="benchmark_fmpr.py",
                concern="scalar_speed",
                category="scalar",
                implementation="repo_native",
                operation=f"fmpr_{args.which}",
                device=jax.default_backend(),
                dtype=str(a.dtype),
                cold_time_s=t1 - t0,
                warm_time_s=sum(warm_times) / len(warm_times),
                recompile_time_s=t3 - t2,
            ),
        ),
        environment=collect_runtime_manifest(Path(__file__).resolve().parents[1], jax_mode="auto"),
    )
    write_benchmark_report(args.output, report)

    print(f"fmpr ({args.which}) | samples={args.samples} | warm_time_ms={ms:.2f}")
    _log_run(
        "benchmark_fmpr",
        f"benchmark_fmpr.py --samples {args.samples} --which {args.which} --runs {args.runs}",
        f"warm_time_ms={ms:.2f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
