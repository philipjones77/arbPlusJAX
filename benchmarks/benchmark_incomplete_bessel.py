from __future__ import annotations

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from arbplusjax import api
from benchmarks.schema import BenchmarkMeasurement
from benchmarks.schema import BenchmarkRecord
from benchmarks.schema import BenchmarkReport
from benchmarks.schema import write_benchmark_report
from tools.runtime_manifest import collect_runtime_manifest


def _timer(fn, *, iters: int) -> tuple[float, float, jax.Array]:
    out = fn()
    jax.block_until_ready(out)
    start = time.perf_counter()
    cold = fn()
    jax.block_until_ready(cold)
    cold_s = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
        jax.block_until_ready(out)
    warm_s = (time.perf_counter() - start) / float(iters)
    return cold_s, warm_s, out


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark incomplete-Bessel production surfaces with normalized shared-schema output.")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--jax-mode", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--smoke", action="store_true", help="Run a reduced subset for pytest-owned schema checks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/benchmarks/outputs/special/benchmark_incomplete_bessel.json"),
    )
    args = parser.parse_args()

    real_dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    effective_iters = min(args.iters, 2) if args.smoke else args.iters

    nu = jnp.asarray(0.5, dtype=real_dtype)
    z_mid = jnp.asarray(1.4, dtype=real_dtype)
    z_large = jnp.asarray(20.0, dtype=real_dtype)
    z_fragile = jnp.asarray(0.5, dtype=real_dtype)
    lower_small = jnp.asarray(0.05, dtype=real_dtype)
    lower_mid = jnp.asarray(0.4, dtype=real_dtype)
    lower_large = jnp.asarray(1.2, dtype=real_dtype)
    upper = jnp.asarray(1.2, dtype=real_dtype)

    rows: list[tuple[str, str, callable]] = [
        (
            "incomplete_bessel_k",
            "quadrature_point",
            lambda: api.incomplete_bessel_k(nu, z_mid, lower_mid, mode="point", method="quadrature"),
        ),
        (
            "incomplete_bessel_k",
            "recurrence_point",
            lambda: api.incomplete_bessel_k(nu, z_large, lower_large, mode="point", method="recurrence"),
        ),
        (
            "incomplete_bessel_k",
            "asymptotic_point",
            lambda: api.incomplete_bessel_k(nu, z_large, lower_mid, mode="point", method="asymptotic"),
        ),
        (
            "incomplete_bessel_k",
            "high_precision_refine_point",
            lambda: api.incomplete_bessel_k(jnp.asarray(13.0, dtype=real_dtype), z_fragile, lower_small, mode="point", method="high_precision_refine"),
        ),
        (
            "incomplete_bessel_i",
            "quadrature_point",
            lambda: api.incomplete_bessel_i(jnp.asarray(1.0, dtype=real_dtype), jnp.asarray(0.8, dtype=real_dtype), upper, mode="point", method="quadrature"),
        ),
        (
            "incomplete_bessel_i",
            "high_precision_refine_point",
            lambda: api.incomplete_bessel_i(jnp.asarray(1.0, dtype=real_dtype), jnp.asarray(0.8, dtype=real_dtype), jnp.asarray(jnp.pi, dtype=real_dtype), mode="point", method="high_precision_refine"),
        ),
        (
            "incomplete_bessel_i",
            "batch_point",
            lambda: api.incomplete_bessel_i_batch(
                jnp.asarray([0.0, 1.0, 2.0], dtype=real_dtype),
                jnp.asarray([1.0, 0.8, 0.6], dtype=real_dtype),
                jnp.asarray([jnp.pi, 1.2, 0.9], dtype=real_dtype),
                mode="point",
            ),
        ),
    ]
    if args.smoke:
        rows = rows[:4]

    records = []
    for family, operation, fn in rows:
        cold_s, warm_s, out = _timer(fn, iters=effective_iters)
        sample = float(jnp.ravel(jnp.asarray(out))[0])
        print(f"{family:20s} {operation:28s} {warm_s * 1e6:10.2f} us sample={sample:.6g}")
        records.append(
            BenchmarkRecord(
                benchmark_name="benchmark_incomplete_bessel.py",
                concern="special_speed",
                category="special",
                implementation=family,
                operation=operation,
                device=jax.default_backend(),
                dtype=args.dtype,
                cold_time_s=float(cold_s),
                warm_time_s=float(warm_s),
                measurements=(
                    BenchmarkMeasurement(name="iters", value=effective_iters),
                    BenchmarkMeasurement(name="requested_dtype", value=args.dtype),
                    BenchmarkMeasurement(name="smoke", value=args.smoke),
                ),
                notes="Incomplete-Bessel benchmark normalized onto the shared schema with explicit method/regime operations.",
            )
        )

    report = BenchmarkReport(
        benchmark_name="benchmark_incomplete_bessel.py",
        concern="special_speed",
        category="special",
        records=tuple(records),
        environment=collect_runtime_manifest(Path(__file__).resolve().parents[1], jax_mode=args.jax_mode),
        notes="Normalized incomplete-Bessel benchmark for provider-surface and diagnostics-aware method families.",
    )
    write_benchmark_report(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
