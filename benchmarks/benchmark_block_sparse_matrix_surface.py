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


scb_block_mat = None
scb_vblock_mat = None
srb_block_mat = None
srb_vblock_mat = None


def _load_block_matrix_modules() -> None:
    global scb_block_mat, scb_vblock_mat, srb_block_mat, srb_vblock_mat
    if scb_block_mat is None:
        from arbplusjax import scb_block_mat as _scb_block_mat
        from arbplusjax import scb_vblock_mat as _scb_vblock_mat
        from arbplusjax import srb_block_mat as _srb_block_mat
        from arbplusjax import srb_vblock_mat as _srb_vblock_mat

        scb_block_mat = _scb_block_mat
        scb_vblock_mat = _scb_vblock_mat
        srb_block_mat = _srb_block_mat
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


def _real_case(n_blocks: int, block_size: int, real_dtype):
    _load_block_matrix_modules()
    n = n_blocks * block_size
    dense = jnp.zeros((n, n), dtype=real_dtype)
    eye_block = jnp.eye(block_size, dtype=real_dtype)
    for i in range(n_blocks):
        dense = dense.at[i * block_size : (i + 1) * block_size, i * block_size : (i + 1) * block_size].set((i + 2.0) * eye_block)
        if i + 1 < n_blocks:
            dense = dense.at[(i + 1) * block_size : (i + 2) * block_size, i * block_size : (i + 1) * block_size].set(0.25 * eye_block)
    vec = jnp.linspace(-1.0, 1.0, n, dtype=real_dtype)
    rhs = jnp.stack([vec, vec + 1.0], axis=-1)
    return srb_block_mat.srb_block_mat_from_dense_csr(dense, block_shape=(block_size, block_size)), vec, rhs


def _complex_case(n_blocks: int, block_size: int, real_dtype, complex_dtype):
    _load_block_matrix_modules()
    n = n_blocks * block_size
    dense = jnp.zeros((n, n), dtype=complex_dtype)
    eye_block = jnp.eye(block_size, dtype=complex_dtype)
    for i in range(n_blocks):
        dense = dense.at[i * block_size : (i + 1) * block_size, i * block_size : (i + 1) * block_size].set((i + 2.0 + 0.25j) * eye_block)
        if i + 1 < n_blocks:
            dense = dense.at[(i + 1) * block_size : (i + 2) * block_size, i * block_size : (i + 1) * block_size].set((0.25 - 0.1j) * eye_block)
    vec = jnp.linspace(-1.0, 1.0, n, dtype=real_dtype).astype(complex_dtype)
    rhs = jnp.stack([vec, vec + (1.0 - 0.25j)], axis=-1)
    return scb_block_mat.scb_block_mat_from_dense_csr(dense, block_shape=(block_size, block_size)), vec, rhs


def _real_vblock_case(real_dtype):
    _load_block_matrix_modules()
    dense = jnp.asarray(
        [[2.0, 1.0, 0.0], [1.0, 3.0, -1.0], [0.0, -1.0, 4.0]],
        dtype=real_dtype,
    )
    row_sizes = jnp.asarray([1, 2], dtype=jnp.int32)
    col_sizes = jnp.asarray([1, 2], dtype=jnp.int32)
    vec = jnp.asarray([1.0, -1.0, 0.5], dtype=real_dtype)
    return srb_vblock_mat.srb_vblock_mat_from_dense_csr(dense, row_block_sizes=row_sizes, col_block_sizes=col_sizes), vec


def _complex_vblock_case(complex_dtype):
    _load_block_matrix_modules()
    dense = jnp.asarray(
        [[2.0 + 0.0j, 1.0 - 0.5j, 0.0 + 0.0j], [1.0 + 0.5j, 3.0 + 0.0j, -1.0j], [0.0 + 0.0j, 1.0j, 4.0 + 0.0j]],
        dtype=complex_dtype,
    )
    row_sizes = jnp.asarray([1, 2], dtype=jnp.int32)
    col_sizes = jnp.asarray([1, 2], dtype=jnp.int32)
    vec = jnp.asarray([1.0 + 0.0j, -1.0 + 0.5j, 0.5 - 0.25j], dtype=complex_dtype)
    return scb_vblock_mat.scb_vblock_mat_from_dense_csr(dense, row_block_sizes=row_sizes, col_block_sizes=col_sizes), vec


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark block-sparse BSR-like matrix surface.")
    parser.add_argument("--n-blocks", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--smoke", action="store_true", help="Run only the fast matvec/cached subset for pytest-owned schema checks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/benchmarks/outputs/matrix/benchmark_block_sparse_matrix_surface.json"),
    )
    args = parser.parse_args()
    _load_block_matrix_modules()
    real_dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    complex_dtype = jnp.complex128 if args.dtype == "float64" else jnp.complex64

    print(f"platform: {platform.platform()}")
    print(f"jax_backend: {jax.default_backend()}")
    print(f"n_blocks: {args.n_blocks}")
    print(f"block_size: {args.block_size}")
    print(f"warmup: {args.warmup}")
    print(f"runs: {args.runs}")
    print(f"dtype: {args.dtype}")
    print(f"smoke: {args.smoke}")

    rmat, rvec, rrhs = _real_case(args.n_blocks, args.block_size, real_dtype)
    cmat, cvec, crhs = _complex_case(args.n_blocks, args.block_size, real_dtype, complex_dtype)
    rvmat, rvvec = _real_vblock_case(real_dtype)
    cvmat, cvvec = _complex_vblock_case(complex_dtype)
    rrplan = srb_block_mat.srb_block_mat_rmatvec_cached_prepare(rmat)
    carplan = scb_block_mat.scb_block_mat_adjoint_matvec_cached_prepare(cmat)
    rvrplan = srb_vblock_mat.srb_vblock_mat_rmatvec_cached_prepare(rvmat)
    cvarplan = scb_vblock_mat.scb_vblock_mat_adjoint_matvec_cached_prepare(cvmat)
    stats = {
        "srb_block_storage_prepare_s": _time_call(
            lambda: srb_block_mat.srb_block_mat_from_dense_csr(
                srb_block_mat.srb_block_mat_to_dense(rmat),
                block_shape=(rmat.block_rows, rmat.block_cols),
            ),
            warmup=args.warmup,
            runs=args.runs,
        ),
        "scb_block_storage_prepare_s": _time_call(
            lambda: scb_block_mat.scb_block_mat_from_dense_csr(
                scb_block_mat.scb_block_mat_to_dense(cmat),
                block_shape=(cmat.block_rows, cmat.block_cols),
            ),
            warmup=args.warmup,
            runs=args.runs,
        ),
        "srb_vblock_storage_prepare_s": _time_call(
            lambda: srb_vblock_mat.srb_vblock_mat_from_dense_csr(
                srb_vblock_mat.srb_vblock_mat_to_dense(rvmat),
                row_block_sizes=rvmat.row_block_sizes,
                col_block_sizes=rvmat.col_block_sizes,
            ),
            warmup=args.warmup,
            runs=args.runs,
        ),
        "scb_vblock_storage_prepare_s": _time_call(
            lambda: scb_vblock_mat.scb_vblock_mat_from_dense_csr(
                scb_vblock_mat.scb_vblock_mat_to_dense(cvmat),
                row_block_sizes=cvmat.row_block_sizes,
                col_block_sizes=cvmat.col_block_sizes,
            ),
            warmup=args.warmup,
            runs=args.runs,
        ),
        "srb_block_cached_prepare_s": _time_call(
            srb_block_mat.srb_block_mat_rmatvec_cached_prepare,
            rmat,
            warmup=args.warmup,
            runs=args.runs,
        ),
        "scb_block_cached_prepare_s": _time_call(
            scb_block_mat.scb_block_mat_adjoint_matvec_cached_prepare,
            cmat,
            warmup=args.warmup,
            runs=args.runs,
        ),
        "srb_vblock_cached_prepare_s": _time_call(
            srb_vblock_mat.srb_vblock_mat_rmatvec_cached_prepare,
            rvmat,
            warmup=args.warmup,
            runs=args.runs,
        ),
        "scb_vblock_cached_prepare_s": _time_call(
            scb_vblock_mat.scb_vblock_mat_adjoint_matvec_cached_prepare,
            cvmat,
            warmup=args.warmup,
            runs=args.runs,
        ),
        "srb_block_matvec_s": _time_call(jax.jit(srb_block_mat.srb_block_mat_matvec), rmat, rvec, warmup=args.warmup, runs=args.runs),
        "srb_block_rmatvec_cached_s": _time_call(jax.jit(srb_block_mat.srb_block_mat_rmatvec_cached_apply), rrplan, rvec, warmup=args.warmup, runs=args.runs),
        "srb_vblock_matvec_s": _time_call(jax.jit(srb_vblock_mat.srb_vblock_mat_matvec), rvmat, rvvec, warmup=args.warmup, runs=args.runs),
        "srb_vblock_rmatvec_cached_s": _time_call(jax.jit(srb_vblock_mat.srb_vblock_mat_rmatvec_cached_apply), rvrplan, rvvec, warmup=args.warmup, runs=args.runs),
        "scb_block_matvec_s": _time_call(jax.jit(scb_block_mat.scb_block_mat_matvec), cmat, cvec, warmup=args.warmup, runs=args.runs),
        "scb_block_adjoint_cached_s": _time_call(jax.jit(scb_block_mat.scb_block_mat_adjoint_matvec_cached_apply), carplan, cvec, warmup=args.warmup, runs=args.runs),
        "scb_vblock_matvec_s": _time_call(jax.jit(scb_vblock_mat.scb_vblock_mat_matvec), cvmat, cvvec, warmup=args.warmup, runs=args.runs),
        "scb_vblock_adjoint_cached_s": _time_call(jax.jit(scb_vblock_mat.scb_vblock_mat_adjoint_matvec_cached_apply), cvarplan, cvvec, warmup=args.warmup, runs=args.runs),
    }
    if not args.smoke:
        stats.update(
            {
                "srb_block_matmul_dense_rhs_s": _time_call(jax.jit(srb_block_mat.srb_block_mat_matmul_dense_rhs), rmat, rrhs, warmup=args.warmup, runs=args.runs),
                "scb_block_matmul_dense_rhs_s": _time_call(jax.jit(scb_block_mat.scb_block_mat_matmul_dense_rhs), cmat, crhs, warmup=args.warmup, runs=args.runs),
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
                benchmark_name="benchmark_block_sparse_matrix_surface.py",
                concern="matrix_speed",
                category="matrix_block_sparse",
                implementation=implementation,
                operation=operation,
                device=jax.default_backend(),
                dtype=("complex128" if args.dtype == "float64" else "complex64") if is_complex else args.dtype,
                warm_time_s=float(value),
                measurements=(
                    BenchmarkMeasurement(name="n_blocks", value=args.n_blocks, unit="blocks"),
                    BenchmarkMeasurement(name="block_size", value=args.block_size, unit="rows"),
                    BenchmarkMeasurement(name="warmup", value=args.warmup, unit="calls"),
                    BenchmarkMeasurement(name="runs", value=args.runs, unit="calls"),
                    BenchmarkMeasurement(name="requested_dtype", value=args.dtype),
                    BenchmarkMeasurement(name="smoke", value=args.smoke),
                ),
                notes="Block/vblock sparse matrix benchmark normalized onto the shared benchmark schema; storage-format preparation and cached-plan preparation are reported separately from solve-quality kernels.",
            )
        )
    report = BenchmarkReport(
        benchmark_name="benchmark_block_sparse_matrix_surface.py",
        concern="matrix_speed",
        category="matrix_block_sparse",
        records=tuple(records),
        environment=collect_runtime_manifest(
            Path(__file__).resolve().parents[1],
            jax_mode="gpu" if jax.default_backend() in {"gpu", "cuda"} else "cpu",
        ),
        notes="Block-sparse matrix benchmark. Stdout preserves metric-style lines for notebook/report compatibility and separates storage/plan preparation from solver-quality kernels.",
    )
    write_benchmark_report(args.output, report)


if __name__ == "__main__":
    main()
