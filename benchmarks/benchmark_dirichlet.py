from __future__ import annotations

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from arbplusjax import dirichlet
from benchmarks.schema import BenchmarkMeasurement
from benchmarks.schema import BenchmarkRecord
from benchmarks.schema import BenchmarkReport
from benchmarks.schema import write_benchmark_report
from tools.runtime_manifest import collect_runtime_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark dirichlet JAX kernels.")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--terms", type=int, default=32)
    parser.add_argument("--which", type=str, default="zeta", choices=["zeta", "eta"])
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--jax-mode", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--smoke", action="store_true", help="Run a reduced benchmark size for pytest-owned schema checks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/benchmarks/outputs/dirichlet/benchmark_dirichlet.json"),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(1712)
    sample_count = min(args.samples, 512) if args.smoke else args.samples
    terms = min(args.terms, 16) if args.smoke else args.terms
    real_dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    mid = rng.uniform(1.2, 4.0, size=sample_count)
    half = rng.uniform(0.0, 0.2, size=sample_count)
    lo = mid - half
    hi = mid + half
    s = jnp.asarray(np.stack([lo, hi], axis=-1).astype(np.float64), dtype=real_dtype)

    if args.which == "zeta":
        fn = jax.jit(dirichlet.dirichlet_zeta_batch, static_argnames=("n_terms",))
    else:
        fn = jax.jit(dirichlet.dirichlet_eta_batch, static_argnames=("n_terms",))
    fn(s, n_terms=terms).block_until_ready()

    start = time.perf_counter()
    out = fn(s, n_terms=terms)
    out.block_until_ready()
    warm_s = time.perf_counter() - start

    print(f"dirichlet ({args.which}) | samples={sample_count} | terms={terms} | time_ms={warm_s * 1000.0:.2f}")

    report = BenchmarkReport(
        benchmark_name="benchmark_dirichlet.py",
        concern="scalar_speed",
        category="special",
        records=(
            BenchmarkRecord(
                benchmark_name="benchmark_dirichlet.py",
                concern="scalar_speed",
                category="special",
                implementation="dirichlet",
                operation=f"{args.which}_batch",
                device=jax.default_backend(),
                dtype=args.dtype,
                warm_time_s=float(warm_s),
                measurements=(
                    BenchmarkMeasurement(name="samples", value=sample_count, unit="rows"),
                    BenchmarkMeasurement(name="terms", value=terms),
                    BenchmarkMeasurement(name="requested_dtype", value=args.dtype),
                    BenchmarkMeasurement(name="which", value=args.which),
                    BenchmarkMeasurement(name="smoke", value=args.smoke),
                ),
                notes="Legacy dirichlet benchmark normalized onto the shared benchmark schema; stdout remains summary-style for notebook compatibility.",
            ),
        ),
        environment=collect_runtime_manifest(Path(__file__).resolve().parents[1], jax_mode=args.jax_mode),
        notes="Real dirichlet benchmark with explicit dtype and requested runtime mode controls.",
    )
    write_benchmark_report(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
