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
stable_kernels = None


def _load_special_api():
    global api, stable_kernels
    if api is not None:
        return
    from arbplusjax import api as _api
    from arbplusjax import stable_kernels as _stable_kernels

    api = _api
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


def _next_multiple(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


def _positive_case(samples: int, dtype: str) -> tuple[jax.Array, jax.Array]:
    rng = np.random.default_rng(3201)
    dt = jnp.float32 if dtype == "float32" else jnp.float64
    return (
        jnp.asarray(rng.uniform(1.1, 3.5, size=samples), dtype=dt),
        jnp.asarray(rng.uniform(0.2, 2.0, size=samples), dtype=dt),
    )


def _bessel_case(samples: int, dtype: str) -> tuple[jax.Array, jax.Array, jax.Array]:
    rng = np.random.default_rng(3203)
    dt = jnp.float32 if dtype == "float32" else jnp.float64
    return (
        jnp.asarray(rng.uniform(0.2, 1.2, size=samples), dtype=dt),
        jnp.asarray(rng.uniform(0.8, 3.0, size=samples), dtype=dt),
        jnp.asarray(rng.uniform(0.05, 0.6, size=samples), dtype=dt),
    )


def _barnes_case(samples: int, dtype: str) -> tuple[jax.Array, jax.Array]:
    rng = np.random.default_rng(3207)
    real_dt = jnp.float32 if dtype == "float32" else jnp.float64
    complex_dt = jnp.complex64 if dtype == "float32" else jnp.complex128
    z_re = jnp.asarray(rng.uniform(1.1, 2.2, size=samples), dtype=real_dt)
    z_im = jnp.asarray(rng.uniform(0.02, 0.2, size=samples), dtype=real_dt)
    tau = jnp.asarray(rng.uniform(0.4, 0.8, size=samples), dtype=real_dt)
    return z_re.astype(complex_dt) + 1j * z_im.astype(complex_dt), tau


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark repeated service-style calls through representative special-function public APIs.")
    parser.add_argument("--samples", type=int, default=4099)
    parser.add_argument("--pad-multiple", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--startup-prewarm", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/benchmarks/outputs/special/benchmark_special_function_service_api.json"),
    )
    args = parser.parse_args()
    _load_special_api()

    if args.startup_prewarm:
        stable_kernels.warm_special_function_point_kernels(dtype="float64", pad_to=_next_multiple(max(8, args.samples // 32), 8))

    pad_to = _next_multiple(args.samples, args.pad_multiple)
    alt_n = max(8, args.samples // 2 + 3)
    records: list[BenchmarkRecord] = []

    cases: list[tuple[str, str, str, object, tuple[object, ...], tuple[object, ...], dict[str, object]]] = []
    service_cases: list[tuple[str, str, str, object, tuple[object, ...], tuple[object, ...], dict[str, object]]] = []
    for dtype in ("float32", "float64"):
        s, z = _positive_case(args.samples, dtype)
        ss, zz = s[:alt_n], z[:alt_n]
        cases.append(
            (
                "point",
                "incomplete_gamma_upper",
                dtype,
                api.bind_point_batch,
                (s, z),
                (ss, zz),
                {"method": "quadrature", "regularized": True},
            )
        )
        cases.append(
            (
                "basic",
                "incomplete_gamma_upper",
                dtype,
                api.bind_interval_batch,
                (s, z),
                (ss, zz),
                {"method": "quadrature", "regularized": True, "prec_bits": 53},
            )
        )
        nu, bz, lower = _bessel_case(args.samples, dtype)
        nnu, bbz, llower = nu[:alt_n], bz[:alt_n], lower[:alt_n]
        cases.append(
            (
                "point",
                "incomplete_bessel_k",
                dtype,
                api.bind_point_batch,
                (nu, bz, lower),
                (nnu, bbz, llower),
                {"method": "quadrature"},
            )
        )
        cases.append(
            (
                "basic",
                "incomplete_bessel_k",
                dtype,
                api.bind_interval_batch,
                (nu, bz, lower),
                (nnu, bbz, llower),
                {"method": "quadrature", "prec_bits": 53},
            )
        )
        bz, btau = _barnes_case(args.samples, dtype)
        bbz, bbtau = bz[:alt_n], btau[:alt_n]
        service_cases.append(
            (
                "point",
                "provider_barnesdoublegamma",
                dtype,
                stable_kernels.provider_barnesdoublegamma_batch,
                (bz, btau),
                (bbz, bbtau),
                {"pad_to": None, "dps": 60},
            )
        )
        service_cases.append(
            (
                "point",
                "provider_log_barnesdoublegamma",
                dtype,
                stable_kernels.provider_log_barnesdoublegamma_batch,
                (bz, btau),
                (bbz, bbtau),
                {"pad_to": None, "dps": 60},
            )
        )

    for mode, name, dtype, binder, call_args, alt_args, extra_kwargs in cases:
        for implementation, pad_target in (("service_api_unpadded", None), ("service_api_padded", pad_to)):
            if binder is api.bind_point_batch:
                fn = binder(name, dtype=dtype, pad_to=pad_target, **extra_kwargs)
            else:
                fn = binder(name, mode=mode, dtype=dtype, pad_to=pad_target, **extra_kwargs)
            stats, steady = _profile(fn, call_args, alt_args, iterations=args.iterations)
            records.append(
                BenchmarkRecord(
                    benchmark_name="benchmark_special_function_service_api.py",
                    concern="service_api_speed",
                    category="special",
                    implementation=implementation,
                    operation=name,
                    device=jax.default_backend(),
                    dtype=dtype,
                    cold_time_s=stats["cold_time_s"],
                    warm_time_s=stats["warm_time_s"],
                    recompile_time_s=stats["recompile_time_s"],
                    measurements=(
                        BenchmarkMeasurement(name="mode", value=mode),
                        BenchmarkMeasurement(name="samples", value=args.samples, unit="rows"),
                        BenchmarkMeasurement(name="pad_to", value=pad_target if pad_target is not None else 0, unit="rows"),
                        BenchmarkMeasurement(name="iterations", value=args.iterations, unit="calls"),
                        BenchmarkMeasurement(name="p50_latency_s", value=_percentile(steady, 50), unit="s"),
                        BenchmarkMeasurement(name="p95_latency_s", value=_percentile(steady, 95), unit="s"),
                        BenchmarkMeasurement(name="p99_latency_s", value=_percentile(steady, 99), unit="s"),
                    ),
                    notes="Public API service-style repeated-call benchmark through bind_point_batch/bind_interval_batch for representative special functions.",
                )
            )

    for mode, name, dtype, fn, call_args, alt_args, extra_kwargs in service_cases:
        for implementation, pad_target in (("service_api_unpadded", None), ("service_api_padded", pad_to)):
            kwargs = dict(extra_kwargs)
            kwargs["pad_to"] = pad_target
            bound = lambda *aa, _fn=fn, _kwargs=kwargs: _fn(*aa, dtype=dtype, **_kwargs)
            stats, steady = _profile(bound, call_args, alt_args, iterations=args.iterations)
            records.append(
                BenchmarkRecord(
                    benchmark_name="benchmark_special_function_service_api.py",
                    concern="service_api_speed",
                    category="special",
                    implementation=implementation,
                    operation=name,
                    device=jax.default_backend(),
                    dtype=dtype,
                    cold_time_s=stats["cold_time_s"],
                    warm_time_s=stats["warm_time_s"],
                    recompile_time_s=stats["recompile_time_s"],
                    measurements=(
                        BenchmarkMeasurement(name="mode", value=mode),
                        BenchmarkMeasurement(name="samples", value=args.samples, unit="rows"),
                        BenchmarkMeasurement(name="pad_to", value=pad_target if pad_target is not None else 0, unit="rows"),
                        BenchmarkMeasurement(name="iterations", value=args.iterations, unit="calls"),
                        BenchmarkMeasurement(name="p50_latency_s", value=_percentile(steady, 50), unit="s"),
                        BenchmarkMeasurement(name="p95_latency_s", value=_percentile(steady, 95), unit="s"),
                        BenchmarkMeasurement(name="p99_latency_s", value=_percentile(steady, 99), unit="s"),
                    ),
                    notes="Service-facing stable-kernel benchmark for Barnes provider batch surfaces.",
                )
            )

    report = BenchmarkReport(
        benchmark_name="benchmark_special_function_service_api.py",
        concern="service_api_speed",
        category="special",
        records=tuple(records),
        environment=collect_runtime_manifest(
            Path(__file__).resolve().parents[1],
            jax_mode="gpu" if jax.default_backend() in {"gpu", "cuda"} else "cpu",
        ),
        notes="Repeated-call public API benchmark for representative special-function families, including padded vs unpadded service use.",
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
