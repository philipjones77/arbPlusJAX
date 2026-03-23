from __future__ import annotations

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


import argparse
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from arbplusjax import api
from benchmarks.schema import BenchmarkMeasurement
from benchmarks.schema import BenchmarkRecord
from benchmarks.schema import BenchmarkReport
from benchmarks.schema import write_benchmark_report
from tools.runtime_manifest import collect_runtime_manifest


def _block(x):
    if isinstance(x, tuple):
        for item in x:
            jax.block_until_ready(item)
        return x
    return jax.block_until_ready(x)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _profile(fn, args, alt_args, *, iterations: int) -> tuple[dict[str, float], list[float]]:
    t0 = time.perf_counter()
    _block(fn(*args))
    t1 = time.perf_counter()
    steady: list[float] = []
    for _ in range(iterations):
        s0 = time.perf_counter()
        _block(fn(*args))
        steady.append(time.perf_counter() - s0)
    t2 = time.perf_counter()
    _block(fn(*alt_args))
    t3 = time.perf_counter()
    return {
        "cold_time_s": t1 - t0,
        "warm_time_s": statistics.mean(steady),
        "recompile_time_s": t3 - t2,
    }, steady


def _next_multiple(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


def _real_case(samples: int, dtype: str) -> tuple[jax.Array, jax.Array]:
    rng = np.random.default_rng(2411)
    dt = jnp.float32 if dtype == "float32" else jnp.float64
    return (
        jnp.asarray(rng.normal(size=samples), dtype=dt),
        jnp.asarray(rng.normal(size=samples), dtype=dt),
    )


def _complex_case(samples: int, dtype: str) -> tuple[jax.Array, jax.Array]:
    rng = np.random.default_rng(2413)
    dt = jnp.complex64 if dtype == "float32" else jnp.complex128
    return (
        jnp.asarray(rng.normal(size=samples) + 1j * rng.normal(size=samples), dtype=dt),
        jnp.asarray(rng.normal(size=samples) + 1j * rng.normal(size=samples), dtype=dt),
    )


def _int_case(samples: int) -> tuple[jax.Array, jax.Array]:
    rng = np.random.default_rng(2417)
    lo = rng.integers(-50, 50, size=samples, dtype=np.int64)
    hi = lo + rng.integers(0, 50, size=samples, dtype=np.int64)
    lo2 = rng.integers(-50, 50, size=samples, dtype=np.int64)
    hi2 = lo2 + rng.integers(0, 50, size=samples, dtype=np.int64)
    return (
        jnp.asarray(np.stack([lo, hi], axis=-1), dtype=jnp.int64),
        jnp.asarray(np.stack([lo2, hi2], axis=-1), dtype=jnp.int64),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark real service-style repeated calls through the public core scalar APIs.")
    parser.add_argument("--samples", type=int, default=4099)
    parser.add_argument("--pad-multiple", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/benchmarks/outputs/scalar/benchmark_core_scalar_service_api.json"),
    )
    args = parser.parse_args()

    pad_to = _next_multiple(args.samples, args.pad_multiple)
    alt_n = max(8, args.samples // 2 + 3)
    records: list[BenchmarkRecord] = []

    cases: list[tuple[str, str, tuple[object, ...], tuple[object, ...]]] = []
    for dtype in ("float32", "float64"):
        a, b = _real_case(args.samples, dtype)
        aa, bb = a[:alt_n], b[:alt_n]
        cases.append(("arf_add", dtype, (a, b), (aa, bb)))
        cases.append(("fmpr_mul", dtype, (a, b), (aa, bb)))
        cases.append(("arb_fpwrap_double_exp", dtype, (a,), (aa,)))
        z1, z2 = _complex_case(args.samples, dtype)
        zz1, zz2 = z1[:alt_n], z2[:alt_n]
        cases.append(("acf_mul", dtype, (z1, z2), (zz1, zz2)))
    ia, ib = _int_case(args.samples)
    ia2, ib2 = ia[:alt_n], ib[:alt_n]
    cases.append(("fmpzi_add", "int64", (ia, ib), (ia2, ib2)))

    for name, dtype, call_args, alt_args in cases:
        for implementation, pad_target in (("service_api_unpadded", None), ("service_api_padded", pad_to)):
            fn = api.bind_point_batch(name, dtype=None if dtype == "int64" else dtype, pad_to=pad_target)
            stats, steady = _profile(fn, call_args, alt_args, iterations=args.iterations)
            records.append(
                BenchmarkRecord(
                    benchmark_name="benchmark_core_scalar_service_api.py",
                    concern="service_api_speed",
                    category="scalar",
                    implementation=implementation,
                    operation=name,
                    device=jax.default_backend(),
                    dtype=dtype,
                    cold_time_s=stats["cold_time_s"],
                    warm_time_s=stats["warm_time_s"],
                    recompile_time_s=stats["recompile_time_s"],
                    measurements=(
                        BenchmarkMeasurement(name="samples", value=args.samples, unit="rows"),
                        BenchmarkMeasurement(name="pad_to", value=pad_target if pad_target is not None else 0, unit="rows"),
                        BenchmarkMeasurement(name="iterations", value=args.iterations, unit="calls"),
                        BenchmarkMeasurement(name="p50_latency_s", value=_percentile(steady, 50), unit="s"),
                        BenchmarkMeasurement(name="p95_latency_s", value=_percentile(steady, 95), unit="s"),
                        BenchmarkMeasurement(name="p99_latency_s", value=_percentile(steady, 99), unit="s"),
                    ),
                    notes="Public API service-style repeated-call benchmark through api.bind_point_batch.",
                )
            )

    report = BenchmarkReport(
        benchmark_name="benchmark_core_scalar_service_api.py",
        concern="service_api_speed",
        category="scalar",
        records=tuple(records),
        environment=collect_runtime_manifest(Path(__file__).resolve().parents[1], jax_mode="auto"),
        notes="Repeated-call public API benchmark for the core numeric scalar helper families, including padded vs unpadded service use.",
    )
    path = write_benchmark_report(args.output, report)
    print(f"report: {path}")
    print(f"samples: {args.samples}")
    print(f"pad_to: {pad_to}")
    for record in records:
        print(f"{record.operation} | {record.dtype} | {record.implementation} | warm_time_ms={1000.0 * float(record.warm_time_s or 0.0):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
