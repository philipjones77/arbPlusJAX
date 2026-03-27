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

from benchmarks.schema import BenchmarkMeasurement
from benchmarks.schema import BenchmarkRecord
from benchmarks.schema import BenchmarkReport
from benchmarks.schema import write_benchmark_report
from tools.runtime_manifest import collect_runtime_manifest

api = None
di = None


def _load_matrix_service_modules():
    global api, di
    if api is not None:
        return
    from arbplusjax import api as _api
    from arbplusjax import double_interval as _di

    api = _api
    di = _di


def _block(x):
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


def _interval_batch(values: np.ndarray, dtype: str) -> jax.Array:
    _load_matrix_service_modules()
    dt = jnp.float32 if dtype == "float32" else jnp.float64
    arr = jnp.asarray(values, dtype=dt)
    return di.interval(arr, arr)


def _acb_batch(real: np.ndarray, imag: np.ndarray, dtype: str) -> jax.Array:
    _load_matrix_service_modules()
    from arbplusjax import acb_core

    dt = jnp.float32 if dtype == "float32" else jnp.float64
    re = jnp.asarray(real, dtype=dt)
    im = jnp.asarray(imag, dtype=dt)
    return acb_core.acb_box(di.interval(re, re), di.interval(im, im))


def _spd_matrix_case(samples: int, dtype: str) -> tuple[jax.Array, jax.Array]:
    rng = np.random.default_rng(4211)
    mats = []
    rhs = []
    for _ in range(samples):
        base = rng.normal(size=(2, 2))
        mat = base @ base.T + 2.0 * np.eye(2)
        mats.append(mat)
        rhs.append(rng.normal(size=(2,)))
    return _interval_batch(np.stack(mats, axis=0), dtype), _interval_batch(np.stack(rhs, axis=0), dtype)


def _acb_matrix_case(samples: int, dtype: str) -> tuple[jax.Array, jax.Array]:
    rng = np.random.default_rng(4213)
    mats_re = rng.normal(size=(samples, 2, 2))
    mats_im = rng.normal(scale=0.25, size=(samples, 2, 2))
    vec_re = rng.normal(size=(samples, 2))
    vec_im = rng.normal(scale=0.25, size=(samples, 2))
    return _acb_batch(mats_re, mats_im, dtype), _acb_batch(vec_re, vec_im, dtype)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark repeated service-style calls through representative matrix public APIs.")
    parser.add_argument("--samples", type=int, default=2049)
    parser.add_argument("--pad-multiple", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/benchmarks/outputs/matrix/benchmark_matrix_service_api.json"),
    )
    args = parser.parse_args()
    _load_matrix_service_modules()

    pad_to = _next_multiple(args.samples, args.pad_multiple)
    alt_n = max(8, args.samples // 2 + 3)
    records: list[BenchmarkRecord] = []

    cases: list[tuple[str, str, str, object, tuple[object, ...], tuple[object, ...], dict[str, object]]] = []
    for dtype in ("float32", "float64"):
        dense, rhs = _spd_matrix_case(args.samples, dtype)
        dense_alt, rhs_alt = dense[:alt_n], rhs[:alt_n]
        cases.append(("point", "arb_mat_det", dtype, api.bind_point_batch, (dense,), (dense_alt,), {}))
        cases.append(("basic", "arb_mat_solve", dtype, api.bind_interval_batch, (dense, rhs), (dense_alt, rhs_alt), {"prec_bits": 53}))
        acb_dense, acb_rhs = _acb_matrix_case(args.samples, dtype)
        acb_dense_alt, acb_rhs_alt = acb_dense[:alt_n], acb_rhs[:alt_n]
        cases.append(("point", "acb_mat_matvec", dtype, api.bind_point_batch, (acb_dense, acb_rhs), (acb_dense_alt, acb_rhs_alt), {}))

    for mode, name, dtype, binder, call_args, alt_args, extra_kwargs in cases:
        for implementation, pad_target in (("service_api_unpadded", None), ("service_api_padded", pad_to)):
            if binder is api.bind_point_batch:
                fn = binder(name, dtype=dtype, pad_to=pad_target, **extra_kwargs)
            else:
                fn = binder(name, mode=mode, dtype=dtype, pad_to=pad_target, **extra_kwargs)
            stats, steady = _profile(fn, call_args, alt_args, iterations=args.iterations)
            records.append(
                BenchmarkRecord(
                    benchmark_name="benchmark_matrix_service_api.py",
                    concern="service_api_speed",
                    category="matrix",
                    implementation=implementation,
                    operation=name,
                    device=jax.default_backend(),
                    dtype=dtype,
                    cold_time_s=stats["cold_time_s"],
                    warm_time_s=stats["warm_time_s"],
                    recompile_time_s=stats["recompile_time_s"],
                    measurements=(
                        BenchmarkMeasurement(name="mode", value=mode),
                        BenchmarkMeasurement(name="samples", value=args.samples, unit="batches"),
                        BenchmarkMeasurement(name="pad_to", value=pad_target if pad_target is not None else 0, unit="batches"),
                        BenchmarkMeasurement(name="matrix_dim", value=2, unit="n"),
                        BenchmarkMeasurement(name="iterations", value=args.iterations, unit="calls"),
                        BenchmarkMeasurement(name="p50_latency_s", value=_percentile(steady, 50), unit="s"),
                        BenchmarkMeasurement(name="p95_latency_s", value=_percentile(steady, 95), unit="s"),
                        BenchmarkMeasurement(name="p99_latency_s", value=_percentile(steady, 99), unit="s"),
                    ),
                    notes="Public API service-style repeated-call benchmark through bind_point_batch/bind_interval_batch for representative matrix operations.",
                )
            )

    report = BenchmarkReport(
        benchmark_name="benchmark_matrix_service_api.py",
        concern="service_api_speed",
        category="matrix",
        records=tuple(records),
        environment=collect_runtime_manifest(
            Path(__file__).resolve().parents[1],
            jax_mode="gpu" if jax.default_backend() in {"gpu", "cuda"} else "cpu",
        ),
        notes="Repeated-call public API benchmark for representative matrix families, including padded vs unpadded service use.",
    )
    path = write_benchmark_report(args.output, report)
    print(f"report: {path}")
    print(f"samples: {args.samples}")
    print(f"pad_to: {pad_to}")
    for record in records:
        print(
            f"{record.operation} | {record.dtype} | {record.implementation} | "
            f"{next(m.value for m in record.measurements if m.name == 'mode')} | "
            f"warm_time_ms={1000.0 * float(record.warm_time_s or 0.0):.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
