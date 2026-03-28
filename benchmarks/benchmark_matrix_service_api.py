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
stable_kernels = None


def _load_matrix_service_modules():
    global api, di, stable_kernels
    if api is not None:
        return
    from arbplusjax import api as _api
    from arbplusjax import double_interval as _di
    from arbplusjax import stable_kernels as _stable_kernels

    api = _api
    di = _di
    stable_kernels = _stable_kernels


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


def _profile_alt_fn(
    fn,
    args,
    alt_fn,
    alt_args,
    *,
    iterations: int,
) -> tuple[dict[str, float], list[float]]:
    t0 = time.perf_counter()
    _block(fn(*args))
    t1 = time.perf_counter()
    steady: list[float] = []
    for _ in range(iterations):
        s0 = time.perf_counter()
        _block(fn(*args))
        steady.append(time.perf_counter() - s0)
    t2 = time.perf_counter()
    _block(alt_fn(*alt_args))
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


def _warm_matrix_service_kernels(samples: int, pad_to: int) -> dict[str, float]:
    _load_matrix_service_modules()
    timings: dict[str, float] = {}
    for dtype in ("float32", "float64"):
        dense, rhs = _spd_matrix_case(max(8, min(samples, 64)), dtype)
        acb_dense, acb_rhs = _acb_matrix_case(max(8, min(samples, 64)), dtype)
        warmers = (
            (f"arb_mat_det_{dtype}", api.bind_point_batch("arb_mat_det", dtype=dtype, pad_to=pad_to), (dense,)),
            (
                f"arb_mat_solve_basic_{dtype}",
                api.bind_interval_batch("arb_mat_solve", mode="basic", dtype=dtype, pad_to=pad_to, prec_bits=53),
                (dense, rhs),
            ),
            (f"acb_mat_matvec_{dtype}", api.bind_point_batch("acb_mat_matvec", dtype=dtype, pad_to=pad_to), (acb_dense, acb_rhs)),
            (
                f"arb_mat_dense_spd_solve_plan_prepare_{dtype}",
                api.bind_point_batch("arb_mat_dense_spd_solve_plan_prepare", dtype=dtype, pad_to=pad_to),
                (dense,),
            ),
            (
                f"acb_mat_dense_matvec_plan_prepare_{dtype}",
                api.bind_point_batch("acb_mat_dense_matvec_plan_prepare", dtype=dtype, pad_to=pad_to),
                (acb_dense,),
            ),
        )
        for name, fn, args in warmers:
            started = time.perf_counter()
            _block(fn(*args))
            timings[name] = time.perf_counter() - started
        real_plan = api.bind_point_batch("arb_mat_dense_spd_solve_plan_prepare", dtype=dtype, pad_to=pad_to)(dense)
        complex_plan = api.bind_point_batch("acb_mat_dense_matvec_plan_prepare", dtype=dtype, pad_to=pad_to)(acb_dense)
        apply_warmers = (
            (
                f"arb_mat_dense_spd_solve_plan_apply_{dtype}",
                api.bind_point_batch("arb_mat_dense_spd_solve_plan_apply", dtype=dtype, pad_to=pad_to),
                (real_plan, rhs),
            ),
            (
                f"acb_mat_dense_matvec_plan_apply_{dtype}",
                api.bind_point_batch("acb_mat_dense_matvec_plan_apply", dtype=dtype, pad_to=pad_to),
                (complex_plan, acb_rhs),
            ),
        )
        for name, fn, args in apply_warmers:
            started = time.perf_counter()
            _block(fn(*args))
            timings[name] = time.perf_counter() - started
    return timings


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark repeated service-style calls through representative matrix public APIs.")
    parser.add_argument("--samples", type=int, default=2049)
    parser.add_argument("--pad-multiple", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--startup-prewarm", action="store_true", help="Compile representative matrix service kernels before repeated-call measurement.")
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
    prewarm_timings: dict[str, float] = {}
    if args.startup_prewarm:
        prewarm_timings = _warm_matrix_service_kernels(args.samples, pad_to)

    cases: list[tuple[str, str, str, object, tuple[object, ...], tuple[object, ...], dict[str, object]]] = []
    prepared_apply_cases: list[tuple[str, str, str, str, tuple[object, ...], tuple[object, ...]]] = []
    helper_service_cases: list[tuple[str, str, str, tuple[object, ...], tuple[object, ...]]] = []
    for dtype in ("float32", "float64"):
        dense, rhs = _spd_matrix_case(args.samples, dtype)
        dense_alt, rhs_alt = dense[:alt_n], rhs[:alt_n]
        cases.append(("point", "arb_mat_det", dtype, api.bind_point_batch, (dense,), (dense_alt,), {}))
        cases.append(("point", "arb_mat_dense_spd_solve_plan_prepare", dtype, api.bind_point_batch, (dense,), (dense_alt,), {}))
        prepared_apply_cases.append(
            ("point", "arb_mat_dense_spd_solve_plan_prepare", "arb_mat_dense_spd_solve_plan_apply", dtype, (dense, rhs), (dense_alt, rhs_alt))
        )
        helper_service_cases.append(("point", "arb_mat_dense_spd_solve_service", dtype, (dense, rhs), (dense_alt, rhs_alt)))
        cases.append(("basic", "arb_mat_solve", dtype, api.bind_interval_batch, (dense, rhs), (dense_alt, rhs_alt), {"prec_bits": 53}))
        acb_dense, acb_rhs = _acb_matrix_case(args.samples, dtype)
        acb_dense_alt, acb_rhs_alt = acb_dense[:alt_n], acb_rhs[:alt_n]
        cases.append(("point", "acb_mat_matvec", dtype, api.bind_point_batch, (acb_dense, acb_rhs), (acb_dense_alt, acb_rhs_alt), {}))
        cases.append(("point", "acb_mat_dense_matvec_plan_prepare", dtype, api.bind_point_batch, (acb_dense,), (acb_dense_alt,), {}))
        prepared_apply_cases.append(
            ("point", "acb_mat_dense_matvec_plan_prepare", "acb_mat_dense_matvec_plan_apply", dtype, (acb_dense, acb_rhs), (acb_dense_alt, acb_rhs_alt))
        )
        helper_service_cases.append(("point", "acb_mat_dense_matvec_service", dtype, (acb_dense, acb_rhs), (acb_dense_alt, acb_rhs_alt)))

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

    for mode, name, dtype, call_args, alt_args in helper_service_cases:
        matrix_args, rhs_args = call_args
        matrix_args_alt, rhs_args_alt = alt_args
        for implementation, pad_target in (("service_api_unpadded", None), ("service_api_padded", pad_to)):
            if name == "arb_mat_dense_spd_solve_service":
                fn = stable_kernels.prepare_arb_dense_spd_solve_service(matrix_args, dtype=dtype, pad_to=pad_target)
                fn_alt = stable_kernels.prepare_arb_dense_spd_solve_service(matrix_args_alt, dtype=dtype, pad_to=pad_target)
            else:
                fn = stable_kernels.prepare_acb_dense_matvec_service(matrix_args, dtype=dtype, pad_to=pad_target)
                fn_alt = stable_kernels.prepare_acb_dense_matvec_service(matrix_args_alt, dtype=dtype, pad_to=pad_target)
            stats, steady = _profile_alt_fn(fn, (rhs_args,), fn_alt, (rhs_args_alt,), iterations=args.iterations)
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
                    notes="Prepared dense-matrix service helper benchmark using stable-kernel service closures.",
                )
            )

    for mode, prepare_name, apply_name, dtype, call_args, alt_args in prepared_apply_cases:
        matrix_args, rhs_args = call_args[0], call_args[1]
        matrix_args_alt, rhs_args_alt = alt_args[0], alt_args[1]
        for implementation, pad_target in (("service_api_unpadded", None), ("service_api_padded", pad_to)):
            prepare_fn = api.bind_point_batch(prepare_name, dtype=dtype, pad_to=pad_target)
            apply_fn = api.bind_point_batch(apply_name, dtype=dtype, pad_to=pad_target)
            plan = prepare_fn(matrix_args)
            plan_alt = prepare_fn(matrix_args_alt)
            stats, steady = _profile(apply_fn, (plan, rhs_args), (plan_alt, rhs_args_alt), iterations=args.iterations)
            records.append(
                BenchmarkRecord(
                    benchmark_name="benchmark_matrix_service_api.py",
                    concern="service_api_speed",
                    category="matrix",
                    implementation=implementation,
                    operation=apply_name,
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
                    notes="Prepared-plan public API service benchmark with plan prepare matched to the apply batch shape.",
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
    if prewarm_timings:
        for name, value in sorted(prewarm_timings.items()):
            print(f"prewarm_{name}: {value:.6f}")
    for record in records:
        print(
            f"{record.operation} | {record.dtype} | {record.implementation} | "
            f"{next(m.value for m in record.measurements if m.name == 'mode')} | "
            f"warm_time_ms={1000.0 * float(record.warm_time_s or 0.0):.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
