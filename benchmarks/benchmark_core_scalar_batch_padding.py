from __future__ import annotations

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


import argparse
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


def _profile_case(fn, args, alt_args, *, runs: int) -> dict[str, float]:
    t0 = time.perf_counter()
    _block(fn(*args))
    t1 = time.perf_counter()

    warm_times: list[float] = []
    for _ in range(runs):
        s0 = time.perf_counter()
        _block(fn(*args))
        warm_times.append(time.perf_counter() - s0)

    t2 = time.perf_counter()
    _block(fn(*alt_args))
    t3 = time.perf_counter()
    return {
        "cold_time_s": t1 - t0,
        "warm_time_s": sum(warm_times) / len(warm_times),
        "recompile_time_s": t3 - t2,
    }


def _next_multiple(n: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    return ((n + multiple - 1) // multiple) * multiple


def _build_cases(samples: int, pad_to: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(2401)

    real_a = jnp.asarray(rng.normal(size=samples), dtype=jnp.float32)
    real_b = jnp.asarray(rng.normal(size=samples), dtype=jnp.float32)
    complex_a = jnp.asarray(rng.normal(size=samples) + 1j * rng.normal(size=samples), dtype=jnp.complex64)
    complex_b = jnp.asarray(rng.normal(size=samples) + 1j * rng.normal(size=samples), dtype=jnp.complex64)
    ints_lo = rng.integers(-50, 50, size=samples, dtype=np.int64)
    ints_hi = ints_lo + rng.integers(0, 50, size=samples, dtype=np.int64)
    ints2_lo = rng.integers(-50, 50, size=samples, dtype=np.int64)
    ints2_hi = ints2_lo + rng.integers(0, 50, size=samples, dtype=np.int64)
    int_a = jnp.asarray(np.stack([ints_lo, ints_hi], axis=-1), dtype=jnp.int64)
    int_b = jnp.asarray(np.stack([ints2_lo, ints2_hi], axis=-1), dtype=jnp.int64)

    alt_n = max(8, samples // 2 + 3)
    alt_real_a = real_a[:alt_n]
    alt_real_b = real_b[:alt_n]
    alt_complex_a = complex_a[:alt_n]
    alt_complex_b = complex_b[:alt_n]
    alt_int_a = int_a[:alt_n]
    alt_int_b = int_b[:alt_n]

    return [
        {
            "operation": "arf_add",
            "dtype": "float32",
            "args": (real_a, real_b),
            "alt_args": (alt_real_a, alt_real_b),
        },
        {
            "operation": "acf_mul",
            "dtype": "complex64",
            "args": (complex_a, complex_b),
            "alt_args": (alt_complex_a, alt_complex_b),
        },
        {
            "operation": "fmpr_mul",
            "dtype": "float32",
            "args": (real_a, real_b),
            "alt_args": (alt_real_a, alt_real_b),
        },
        {
            "operation": "fmpzi_add",
            "dtype": "int64",
            "args": (int_a, int_b),
            "alt_args": (alt_int_a, alt_int_b),
        },
        {
            "operation": "arb_fpwrap_double_exp",
            "dtype": "float32",
            "args": (real_a,),
            "alt_args": (alt_real_a,),
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark scalar batch padding overhead for the canonical core numeric helper families.")
    parser.add_argument("--samples", type=int, default=4099)
    parser.add_argument("--pad-multiple", type=int, default=128)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/benchmarks/outputs/scalar/benchmark_core_scalar_batch_padding.json"),
    )
    args = parser.parse_args()

    pad_to = _next_multiple(args.samples, args.pad_multiple)
    cases = _build_cases(args.samples, pad_to)
    records: list[BenchmarkRecord] = []

    for case in cases:
        name = str(case["operation"])
        dtype = str(case["dtype"])
        call_args = tuple(case["args"])
        alt_args = tuple(case["alt_args"])

        unpadded = jax.jit(lambda *aa, op=name: api.eval_point_batch(op, *aa))
        padded = jax.jit(lambda *aa, op=name, target=pad_to: api.eval_point_batch(op, *aa, pad_to=target))

        unpadded_stats = _profile_case(unpadded, call_args, alt_args, runs=args.runs)
        padded_stats = _profile_case(padded, call_args, alt_args, runs=args.runs)

        for impl, stats in (("api_batch_unpadded", unpadded_stats), ("api_batch_padded", padded_stats)):
            records.append(
                BenchmarkRecord(
                    benchmark_name="benchmark_core_scalar_batch_padding.py",
                    concern="batch_padding_speed",
                    category="scalar",
                    implementation=impl,
                    operation=name,
                    device=jax.default_backend(),
                    dtype=dtype,
                    cold_time_s=stats["cold_time_s"],
                    warm_time_s=stats["warm_time_s"],
                    recompile_time_s=stats["recompile_time_s"],
                    measurements=(
                        BenchmarkMeasurement(name="samples", value=args.samples, unit="rows"),
                        BenchmarkMeasurement(name="pad_to", value=pad_to, unit="rows"),
                        BenchmarkMeasurement(name="alt_samples", value=len(alt_args[0]), unit="rows"),
                    ),
                    notes="Compares api.eval_point_batch without padding vs explicit caller padding.",
                )
            )

    report = BenchmarkReport(
        benchmark_name="benchmark_core_scalar_batch_padding.py",
        concern="batch_padding_speed",
        category="scalar",
        records=tuple(records),
        environment=collect_runtime_manifest(Path(__file__).resolve().parents[1], jax_mode="auto"),
        notes="Batch/padding speed split for the canonical core numeric scalar helper families.",
    )
    path = write_benchmark_report(args.output, report)
    print(f"report: {path}")
    print(f"samples: {args.samples}")
    print(f"pad_to: {pad_to}")
    for record in records:
        print(
            f"{record.operation} | {record.implementation} | warm_time_ms={1000.0 * float(record.warm_time_s or 0.0):.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
