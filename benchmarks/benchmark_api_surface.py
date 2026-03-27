from __future__ import annotations

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


import argparse
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from benchmarks.schema import BenchmarkMeasurement
from benchmarks.schema import BenchmarkRecord
from benchmarks.schema import BenchmarkReport
from benchmarks.schema import write_benchmark_report
from tools.runtime_manifest import collect_runtime_manifest

api = None
di = None


def _load_api_modules():
    global api, di
    if api is not None:
        return
    from arbplusjax import api as _api
    from arbplusjax import double_interval as _di

    api = _api
    di = _di


def _interval_matrix_from_point(a: jax.Array) -> jax.Array:
    _load_api_modules()
    return di.interval(a, a)


def _box_matrix_from_point(a: jax.Array) -> jax.Array:
    _load_api_modules()
    from arbplusjax import acb_core

    return acb_core.acb_box(
        di.interval(jnp.real(a), jnp.real(a)),
        di.interval(jnp.imag(a), jnp.imag(a)),
    )


def _block(out):
    if isinstance(out, tuple):
        for item in out:
            jax.block_until_ready(item)
        return out
    return jax.block_until_ready(out)


def _profile_case(fn, arg, alt_arg, *, warmup: int, runs: int) -> dict[str, float]:
    started = time.perf_counter()
    _block(fn(*arg))
    compile_plus_first = time.perf_counter() - started

    for _ in range(warmup):
        _block(fn(*arg))

    started = time.perf_counter()
    for _ in range(runs):
        _block(fn(*arg))
    warm = (time.perf_counter() - started) / float(runs)

    started = time.perf_counter()
    _block(fn(*alt_arg))
    recompile = time.perf_counter() - started

    return {
        "cold_time_s": compile_plus_first,
        "warm_time_s": warm,
        "recompile_time_s": recompile,
        "python_overhead_s": max(compile_plus_first - warm, 0.0),
    }


def _record(operation: str, implementation: str, dtype: str, timings: dict[str, float], *, notes: str = "") -> BenchmarkRecord:
    return BenchmarkRecord(
        benchmark_name="benchmark_api_surface.py",
        concern="api_speed",
        category="api",
        implementation=implementation,
        operation=operation,
        device=jax.default_backend(),
        dtype=dtype,
        cold_time_s=timings["cold_time_s"],
        warm_time_s=timings["warm_time_s"],
        recompile_time_s=timings["recompile_time_s"],
        python_overhead_s=timings["python_overhead_s"],
        measurements=(
            BenchmarkMeasurement(name="compile_to_warm_ratio", value=timings["cold_time_s"] / max(timings["warm_time_s"], 1e-12), unit="ratio"),
        ),
        notes=notes,
    )


def run_scalar_case(warmup: int, runs: int) -> tuple[dict[str, float], tuple[BenchmarkRecord, BenchmarkRecord]]:
    _load_api_modules()
    x = jnp.asarray(0.5, dtype=jnp.float64)
    y = jnp.asarray(2.0, dtype=jnp.float64)
    direct = jax.jit(lambda a, b: api.eval_point("cuda_besselk", a, b))
    routed = jax.jit(lambda a, b: api.evaluate("besselk", a, b, implementation="cuda_besselk", value_kind="real"))
    direct_stats = _profile_case(direct, (x, y), (x + 0.1, y), warmup=warmup, runs=runs)
    routed_stats = _profile_case(routed, (x, y), (x + 0.1, y), warmup=warmup, runs=runs)
    stats = {
        "api_scalar_direct_cuda_besselk_s": direct_stats["warm_time_s"],
        "api_scalar_routed_cuda_besselk_s": routed_stats["warm_time_s"],
    }
    return stats, (
        _record("besselk", "direct_cuda_besselk", "float64", direct_stats, notes="point direct implementation"),
        _record("besselk", "routed_cuda_besselk", "float64", routed_stats, notes="capability-routed API path"),
    )


def run_incomplete_gamma_case(warmup: int, runs: int) -> tuple[dict[str, float], tuple[BenchmarkRecord, BenchmarkRecord]]:
    _load_api_modules()
    s = jnp.asarray(2.5, dtype=jnp.float64)
    z = jnp.asarray(1.0, dtype=jnp.float64)
    direct = jax.jit(
        lambda a, b: api.incomplete_gamma_upper(a, b, method="quadrature", samples_per_panel=8, max_panels=16)
    )
    routed = jax.jit(
        lambda a, b: api.evaluate(
            "incomplete_gamma_upper",
            a,
            b,
            method="quadrature",
            method_params={"samples_per_panel": 8, "max_panels": 16},
        )
    )
    direct_stats = _profile_case(direct, (s, z), (s, z + 0.2), warmup=warmup, runs=runs)
    routed_stats = _profile_case(routed, (s, z), (s, z + 0.2), warmup=warmup, runs=runs)
    stats = {
        "api_incgamma_direct_s": direct_stats["warm_time_s"],
        "api_incgamma_routed_s": routed_stats["warm_time_s"],
    }
    return stats, (
        _record("incomplete_gamma_upper", "direct", "float64", direct_stats, notes="explicit direct API"),
        _record("incomplete_gamma_upper", "routed", "float64", routed_stats, notes="general evaluate() routing path"),
    )


def run_matrix_case(warmup: int, runs: int) -> tuple[dict[str, float], tuple[BenchmarkRecord, ...]]:
    _load_api_modules()
    a_mid = jnp.array([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
    rhs_mid = jnp.array([[1.0], [2.0]], dtype=jnp.float64)
    a = _interval_matrix_from_point(a_mid)
    rhs = _interval_matrix_from_point(rhs_mid)

    c_mid = jnp.array([[4.0 + 0.0j, 1.0 + 1.0j], [1.0 - 1.0j, 5.0 + 0.0j]], dtype=jnp.complex128)
    c_rhs_mid = jnp.array([[1.0 + 0.5j], [2.0 - 0.25j]], dtype=jnp.complex128)
    c_a = _box_matrix_from_point(c_mid)
    c_rhs = _box_matrix_from_point(c_rhs_mid)

    direct_real = jax.jit(lambda aa, bb: api.eval_interval("arb_mat_solve", aa, bb, mode="basic"))
    routed_real = jax.jit(
        lambda aa, bb: api.evaluate("arb_mat_solve", aa, bb, mode="basic", value_kind="real_matrix", dtype="float64")
    )
    direct_complex = jax.jit(lambda aa, bb: api.eval_interval("acb_mat_solve", aa, bb, mode="basic"))
    routed_complex = jax.jit(
        lambda aa, bb: api.evaluate("acb_mat_solve", aa, bb, mode="basic", value_kind="complex_matrix", dtype="float64")
    )
    direct_real_stats = _profile_case(direct_real, (a, rhs), (_interval_matrix_from_point(a_mid + jnp.eye(2)), rhs), warmup=warmup, runs=runs)
    routed_real_stats = _profile_case(routed_real, (a, rhs), (_interval_matrix_from_point(a_mid + jnp.eye(2)), rhs), warmup=warmup, runs=runs)
    direct_complex_stats = _profile_case(direct_complex, (c_a, c_rhs), (_box_matrix_from_point(c_mid + (0.5 + 0.0j) * jnp.eye(2)), c_rhs), warmup=warmup, runs=runs)
    routed_complex_stats = _profile_case(routed_complex, (c_a, c_rhs), (_box_matrix_from_point(c_mid + (0.5 + 0.0j) * jnp.eye(2)), c_rhs), warmup=warmup, runs=runs)
    stats = {
        "api_matrix_real_direct_s": direct_real_stats["warm_time_s"],
        "api_matrix_real_routed_s": routed_real_stats["warm_time_s"],
        "api_matrix_complex_direct_s": direct_complex_stats["warm_time_s"],
        "api_matrix_complex_routed_s": routed_complex_stats["warm_time_s"],
    }
    return stats, (
        _record("arb_mat_solve", "direct", "float64", direct_real_stats, notes="real interval matrix solve"),
        _record("arb_mat_solve", "routed", "float64", routed_real_stats, notes="real interval matrix solve through evaluate()"),
        _record("acb_mat_solve", "direct", "complex128", direct_complex_stats, notes="complex box matrix solve"),
        _record("acb_mat_solve", "routed", "complex128", routed_complex_stats, notes="complex box matrix solve through evaluate()"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark routed public API overhead against direct API calls.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/benchmarks/outputs/api_surface/api_surface_report.json"),
        help="Optional JSON benchmark report output path.",
    )
    args = parser.parse_args()

    print(f"platform: {platform.platform()}")
    print(f"jax_backend: {jax.default_backend()}")
    print(f"warmup: {args.warmup}")
    print(f"runs: {args.runs}")

    stats = {}
    records: list[BenchmarkRecord] = []
    scalar_stats, scalar_records = run_scalar_case(args.warmup, args.runs)
    incgamma_stats, incgamma_records = run_incomplete_gamma_case(args.warmup, args.runs)
    matrix_stats, matrix_records = run_matrix_case(args.warmup, args.runs)
    stats.update(scalar_stats)
    stats.update(incgamma_stats)
    stats.update(matrix_stats)
    records.extend(scalar_records)
    records.extend(incgamma_records)
    records.extend(matrix_records)

    report = BenchmarkReport(
        benchmark_name="benchmark_api_surface.py",
        concern="api_speed",
        category="api",
        records=tuple(records),
        environment=collect_runtime_manifest(Path(__file__).resolve().parents[1], jax_mode="auto"),
        notes="Direct-vs-routed API timing for representative scalar, special, and matrix surfaces.",
    )
    path = write_benchmark_report(args.output, report)
    print(f"report: {path}")
    for key in sorted(stats):
        print(f"{key}: {stats[key]:.6e}")


if __name__ == "__main__":
    main()
