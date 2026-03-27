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


scb_vblock_mat = None
srb_vblock_mat = None


def _load_vblock_matrix_modules() -> None:
    global scb_vblock_mat, srb_vblock_mat
    if scb_vblock_mat is None:
        from arbplusjax import scb_vblock_mat as _scb_vblock_mat
        from arbplusjax import srb_vblock_mat as _srb_vblock_mat

        scb_vblock_mat = _scb_vblock_mat
        srb_vblock_mat = _srb_vblock_mat


def _time_call(fn, *args, warmup: int, runs: int) -> float:
    out = None
    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)
    started = time.perf_counter()
    for _ in range(runs):
        out = fn(*args)
        jax.block_until_ready(out)
    ended = time.perf_counter()
    return (ended - started) / float(runs)


def _row_col_sizes(total_n: int) -> jax.Array:
    if total_n < 4:
        return jnp.array([total_n], dtype=jnp.int32)
    return jnp.array([1, 1, total_n - 2], dtype=jnp.int32)


def _real_case(n: int, real_dtype):
    _load_vblock_matrix_modules()
    dense = jnp.zeros((n, n), dtype=real_dtype)
    for i in range(n):
        dense = dense.at[i, i].set(2.0 + 0.25 * i)
        if i + 1 < n:
            dense = dense.at[i + 1, i].set(0.5)
            dense = dense.at[i, i + 1].set(0.25)
    vec = jnp.linspace(-1.0, 1.0, n, dtype=real_dtype)
    rhs = jnp.stack([vec, vec + 1.0], axis=-1)
    sizes = _row_col_sizes(n)
    x = srb_vblock_mat.srb_vblock_mat_from_dense_csr(dense, row_block_sizes=sizes, col_block_sizes=sizes)
    plan = srb_vblock_mat.srb_vblock_mat_matvec_cached_prepare(x)
    return x, plan, vec, rhs


def _complex_case(n: int, real_dtype, complex_dtype):
    _load_vblock_matrix_modules()
    dense = jnp.zeros((n, n), dtype=complex_dtype)
    for i in range(n):
        dense = dense.at[i, i].set(2.0 + 0.25j + 0.2 * i)
        if i + 1 < n:
            dense = dense.at[i + 1, i].set(0.5 - 0.1j)
            dense = dense.at[i, i + 1].set(0.25 + 0.05j)
    vec = jnp.linspace(-1.0, 1.0, n, dtype=real_dtype).astype(complex_dtype)
    rhs = jnp.stack([vec, vec + (1.0 - 0.5j)], axis=-1)
    sizes = _row_col_sizes(n)
    x = scb_vblock_mat.scb_vblock_mat_from_dense_csr(dense, row_block_sizes=sizes, col_block_sizes=sizes)
    plan = scb_vblock_mat.scb_vblock_mat_matvec_cached_prepare(x)
    return x, plan, vec, rhs


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark variable-block sparse matrix surface.")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--smoke", action="store_true", help="Run only the fast matvec/cached subset for pytest-owned schema checks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/benchmarks/outputs/matrix/benchmark_vblock_sparse_matrix_surface.json"),
    )
    args = parser.parse_args()
    _load_vblock_matrix_modules()
    real_dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    complex_dtype = jnp.complex128 if args.dtype == "float64" else jnp.complex64

    print(f"platform: {platform.platform()}")
    print(f"jax_backend: {jax.default_backend()}")
    print(f"n: {args.n}")
    print(f"warmup: {args.warmup}")
    print(f"runs: {args.runs}")
    print(f"dtype: {args.dtype}")
    print(f"smoke: {args.smoke}")

    rmat, rplan, rvec, rrhs = _real_case(args.n, real_dtype)
    cmat, cplan, cvec, crhs = _complex_case(args.n, real_dtype, complex_dtype)
    stats = {
        "srb_vblock_storage_prepare_s": _time_call(
            lambda: srb_vblock_mat.srb_vblock_mat_from_dense_csr(
                srb_vblock_mat.srb_vblock_mat_to_dense(rmat),
                row_block_sizes=rmat.row_block_sizes,
                col_block_sizes=rmat.col_block_sizes,
            ),
            warmup=args.warmup,
            runs=args.runs,
        ),
        "scb_vblock_storage_prepare_s": _time_call(
            lambda: scb_vblock_mat.scb_vblock_mat_from_dense_csr(
                scb_vblock_mat.scb_vblock_mat_to_dense(cmat),
                row_block_sizes=cmat.row_block_sizes,
                col_block_sizes=cmat.col_block_sizes,
            ),
            warmup=args.warmup,
            runs=args.runs,
        ),
        "srb_vblock_cached_prepare_s": _time_call(
            srb_vblock_mat.srb_vblock_mat_matvec_cached_prepare,
            rmat,
            warmup=args.warmup,
            runs=args.runs,
        ),
        "scb_vblock_cached_prepare_s": _time_call(
            scb_vblock_mat.scb_vblock_mat_matvec_cached_prepare,
            cmat,
            warmup=args.warmup,
            runs=args.runs,
        ),
        "srb_vblock_matvec_s": _time_call(jax.jit(srb_vblock_mat.srb_vblock_mat_matvec), rmat, rvec, warmup=args.warmup, runs=args.runs),
        "srb_vblock_matvec_cached_s": _time_call(jax.jit(srb_vblock_mat.srb_vblock_mat_matvec_cached_apply), rplan, rvec, warmup=args.warmup, runs=args.runs),
        "scb_vblock_matvec_s": _time_call(jax.jit(scb_vblock_mat.scb_vblock_mat_matvec), cmat, cvec, warmup=args.warmup, runs=args.runs),
        "scb_vblock_matvec_cached_s": _time_call(jax.jit(scb_vblock_mat.scb_vblock_mat_matvec_cached_apply), cplan, cvec, warmup=args.warmup, runs=args.runs),
    }
    if not args.smoke:
        stats.update(
            {
                "srb_vblock_solve_s": _time_call(
                    lambda a, b: srb_vblock_mat.srb_vblock_mat_solve(a, b, method="lu"),
                    rmat,
                    rvec,
                    warmup=args.warmup,
                    runs=args.runs,
                ),
                "scb_vblock_solve_s": _time_call(
                    lambda a, b: scb_vblock_mat.scb_vblock_mat_solve(a, b, method="lu"),
                    cmat,
                    cvec,
                    warmup=args.warmup,
                    runs=args.runs,
                ),
                "srb_vblock_matmul_dense_rhs_s": _time_call(jax.jit(srb_vblock_mat.srb_vblock_mat_matmul_dense_rhs), rmat, rrhs, warmup=args.warmup, runs=args.runs),
                "scb_vblock_matmul_dense_rhs_s": _time_call(jax.jit(scb_vblock_mat.scb_vblock_mat_matmul_dense_rhs), cmat, crhs, warmup=args.warmup, runs=args.runs),
            }
        )
    for key, value in stats.items():
        print(f"{key}: {value:.6f}")
    records: list[BenchmarkRecord] = []
    for key, value in sorted(stats.items()):
        parts = key.split("_")
        implementation = "_".join(parts[:2])
        operation = "_".join(parts[2:-1])
        is_complex = implementation.startswith("scb")
        records.append(
            BenchmarkRecord(
                benchmark_name="benchmark_vblock_sparse_matrix_surface.py",
                concern="matrix_speed",
                category="matrix_vblock_sparse",
                implementation=implementation,
                operation=operation,
                device=jax.default_backend(),
                dtype=("complex128" if args.dtype == "float64" else "complex64") if is_complex else args.dtype,
                warm_time_s=float(value),
                measurements=(
                    BenchmarkMeasurement(name="n", value=args.n, unit="rows"),
                    BenchmarkMeasurement(name="warmup", value=args.warmup, unit="calls"),
                    BenchmarkMeasurement(name="runs", value=args.runs, unit="calls"),
                    BenchmarkMeasurement(name="requested_dtype", value=args.dtype),
                    BenchmarkMeasurement(name="smoke", value=args.smoke),
                ),
                notes="Variable-block sparse matrix benchmark normalized onto the shared benchmark schema; storage-format preparation and cached-plan preparation are reported separately from solve-quality kernels.",
            )
        )
    report = BenchmarkReport(
        benchmark_name="benchmark_vblock_sparse_matrix_surface.py",
        concern="matrix_speed",
        category="matrix_vblock_sparse",
        records=tuple(records),
        environment=collect_runtime_manifest(
            Path(__file__).resolve().parents[1],
            jax_mode="gpu" if jax.default_backend() in {"gpu", "cuda"} else "cpu",
        ),
        notes="Variable-block sparse matrix benchmark. Stdout preserves metric-style lines for notebook/report compatibility and separates storage/plan preparation from solver-quality kernels.",
    )
    write_benchmark_report(args.output, report)


if __name__ == "__main__":
    main()
