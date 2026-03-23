from __future__ import annotations

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from arbplusjax import acb_calc
from benchmarks.schema import BenchmarkMeasurement
from benchmarks.schema import BenchmarkRecord
from benchmarks.schema import BenchmarkReport
from benchmarks.schema import write_benchmark_report
from tools.runtime_manifest import collect_runtime_manifest


def _random_boxes(rng: np.random.Generator, n: int, scale: float = 2.0) -> np.ndarray:
    a = rng.uniform(-scale, scale, size=(n, 2))
    b = rng.uniform(-scale, scale, size=(n, 2))
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    re = np.stack([lo[:, 0], hi[:, 0]], axis=-1)
    im = np.stack([lo[:, 1], hi[:, 1]], axis=-1)
    return np.concatenate([re, im], axis=-1).astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark acb_calc JAX kernels.")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--integrand", type=str, default="exp", choices=["exp", "sin", "cos"])
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--jax-mode", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--smoke", action="store_true", help="Run a reduced benchmark size for pytest-owned schema checks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/benchmarks/outputs/calc/benchmark_acb_calc.json"),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(2123)
    sample_count = min(args.samples, 512) if args.smoke else args.samples
    steps = min(args.steps, 16) if args.smoke else args.steps
    real_dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    complex_dtype = jnp.complex128 if args.dtype == "float64" else jnp.complex64
    a = jnp.asarray(_random_boxes(rng, sample_count, 1.5), dtype=real_dtype)
    b = jnp.asarray(_random_boxes(rng, sample_count, 1.5), dtype=real_dtype)

    fn = jax.jit(acb_calc.acb_calc_integrate_line_batch, static_argnames=("integrand", "n_steps"))
    fn(a, b, integrand=args.integrand, n_steps=steps).block_until_ready()

    start = time.perf_counter()
    out = fn(a, b, integrand=args.integrand, n_steps=steps)
    out.block_until_ready()
    warm_s = time.perf_counter() - start

    print(f"acb_calc ({args.integrand}) | samples={sample_count} steps={steps} | time_ms={warm_s * 1000.0:.2f}")

    report = BenchmarkReport(
        benchmark_name="benchmark_acb_calc.py",
        concern="scalar_speed",
        category="integration",
        records=(
            BenchmarkRecord(
                benchmark_name="benchmark_acb_calc.py",
                concern="scalar_speed",
                category="integration",
                implementation="acb_calc",
                operation=f"integrate_line_{args.integrand}",
                device=jax.default_backend(),
                dtype=jnp.dtype(complex_dtype).name,
                warm_time_s=float(warm_s),
                measurements=(
                    BenchmarkMeasurement(name="samples", value=sample_count, unit="rows"),
                    BenchmarkMeasurement(name="steps", value=steps, unit="panels"),
                    BenchmarkMeasurement(name="requested_dtype", value=args.dtype),
                    BenchmarkMeasurement(name="integrand", value=args.integrand),
                    BenchmarkMeasurement(name="smoke", value=args.smoke),
                ),
                notes="Legacy acb_calc benchmark normalized onto the shared benchmark schema; stdout remains summary-style for notebook compatibility.",
            ),
        ),
        environment=collect_runtime_manifest(Path(__file__).resolve().parents[1], jax_mode=args.jax_mode),
        notes="acb_calc benchmark with explicit dtype and requested runtime mode controls.",
    )
    write_benchmark_report(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
