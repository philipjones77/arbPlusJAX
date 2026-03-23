from __future__ import annotations

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from arbplusjax import arb_poly
from benchmarks.schema import BenchmarkMeasurement
from benchmarks.schema import BenchmarkRecord
from benchmarks.schema import BenchmarkReport
from benchmarks.schema import write_benchmark_report
from tools.runtime_manifest import collect_runtime_manifest


def _random_intervals(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    a = rng.uniform(lo, hi, size=n)
    b = rng.uniform(lo, hi, size=n)
    low = np.minimum(a, b)
    high = np.maximum(a, b)
    return np.stack([low, high], axis=-1).astype(np.float64)


def _block(value) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()


def _bench(fn, *args) -> tuple[float, float, float]:
    start = time.perf_counter()
    first = fn(*args)
    _block(first)
    cold = time.perf_counter() - start

    start = time.perf_counter()
    warm = fn(*args)
    _block(warm)
    warm_s = time.perf_counter() - start

    alt_coeffs = args[0][: max(1, args[0].shape[0] // 2)]
    alt_x = args[1][: max(1, args[1].shape[0] // 2)]
    start = time.perf_counter()
    recomp = fn(alt_coeffs, alt_x)
    _block(recomp)
    recompile_s = time.perf_counter() - start
    return cold, warm_s, recompile_s


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark arb_poly JAX kernels.")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--jax-mode", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--smoke", action="store_true", help="Run a reduced cubic-evaluation sample count for pytest-owned schema checks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/benchmarks/outputs/poly/benchmark_arb_poly.json"),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(2233)
    sample_count = min(args.samples, 512) if args.smoke else args.samples
    real_dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    coeffs = jnp.asarray(_random_intervals(rng, 4 * sample_count, -0.5, 0.5).reshape(sample_count, 4, 2), dtype=real_dtype)
    x = jnp.asarray(_random_intervals(rng, sample_count, -0.3, 0.3), dtype=real_dtype)

    fn = jax.jit(arb_poly.arb_poly_eval_cubic_batch)
    cold_s, warm_s, recompile_s = _bench(fn, coeffs, x)

    print(f"arb_poly (cubic) | samples={sample_count} | warm_time_ms={warm_s * 1000.0:.2f}")

    report = BenchmarkReport(
        benchmark_name="benchmark_arb_poly.py",
        concern="scalar_speed",
        category="special",
        records=(
            BenchmarkRecord(
                benchmark_name="benchmark_arb_poly.py",
                concern="scalar_speed",
                category="special",
                implementation="arb_poly",
                operation="eval_cubic_batch",
                device=jax.default_backend(),
                dtype=args.dtype,
                cold_time_s=float(cold_s),
                warm_time_s=float(warm_s),
                recompile_time_s=float(recompile_s),
                measurements=(
                    BenchmarkMeasurement(name="samples", value=sample_count, unit="rows"),
                    BenchmarkMeasurement(name="requested_dtype", value=args.dtype),
                    BenchmarkMeasurement(name="smoke", value=args.smoke),
                ),
                notes="Legacy polynomial benchmark normalized onto the shared benchmark schema; stdout remains summary-style for notebook compatibility.",
            ),
        ),
        environment=collect_runtime_manifest(
            Path(__file__).resolve().parents[1],
            jax_mode=args.jax_mode,
        ),
        notes="arb_poly cubic benchmark with explicit dtype and requested runtime mode controls.",
    )
    write_benchmark_report(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
