from __future__ import annotations

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


import argparse
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from arbplusjax import mat_wrappers
from arbplusjax import scb_mat
from arbplusjax import srb_mat
from benchmarks.schema import BenchmarkMeasurement
from benchmarks.schema import BenchmarkRecord
from benchmarks.schema import BenchmarkReport
from benchmarks.schema import write_benchmark_report
from tools.runtime_manifest import collect_runtime_manifest


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


def _srb_case(n: int, real_dtype):
    base = jnp.reshape(jnp.linspace(0.2, 1.2, n * n, dtype=real_dtype), (n, n))
    dense = base.T @ base + jnp.eye(n, dtype=real_dtype) * (n + 1.0)
    sparse = {
        "coo": srb_mat.srb_mat_from_dense_coo(dense),
        "csr": srb_mat.srb_mat_from_dense_csr(dense),
        "bcoo": srb_mat.srb_mat_from_dense_bcoo(dense),
    }
    rhs = jnp.linspace(-0.5, 0.75, n, dtype=real_dtype)
    rhs_batch = jnp.stack([rhs, rhs + 1.0], axis=0)
    rhs_cols = jnp.stack([rhs, rhs + 1.0], axis=1)
    return dense, sparse, rhs, rhs_batch, rhs_cols


def _scb_case(n: int, real_dtype, complex_dtype):
    real = jnp.reshape(jnp.linspace(0.2, 1.1, n * n, dtype=real_dtype), (n, n))
    imag = jnp.reshape(jnp.linspace(-0.3, 0.3, n * n, dtype=real_dtype), (n, n))
    base = real + 1j * imag
    dense = jnp.conj(base.T) @ base + jnp.eye(n, dtype=complex_dtype) * (n + 1.0)
    sparse = {
        "coo": scb_mat.scb_mat_from_dense_coo(dense),
        "csr": scb_mat.scb_mat_from_dense_csr(dense),
        "bcoo": scb_mat.scb_mat_from_dense_bcoo(dense),
    }
    rhs = jnp.linspace(-0.5, 0.75, n, dtype=real_dtype) + 1j * jnp.linspace(0.1, 0.4, n, dtype=real_dtype)
    rhs_batch = jnp.stack([rhs, rhs + (0.25 - 0.1j)], axis=0)
    rhs_cols = jnp.stack([rhs, rhs + (0.25 - 0.1j)], axis=1)
    return dense, sparse, rhs, rhs_batch, rhs_cols


def _srb_storage_prepare(dense: jax.Array, storage: str):
    if storage == "coo":
        return srb_mat.srb_mat_from_dense_coo(dense)
    if storage == "csr":
        return srb_mat.srb_mat_from_dense_csr(dense)
    if storage == "bcoo":
        return srb_mat.srb_mat_from_dense_bcoo(dense)
    raise ValueError(f"unsupported storage kind: {storage}")


def _scb_storage_prepare(dense: jax.Array, storage: str):
    if storage == "coo":
        return scb_mat.scb_mat_from_dense_coo(dense)
    if storage == "csr":
        return scb_mat.scb_mat_from_dense_csr(dense)
    if storage == "bcoo":
        return scb_mat.scb_mat_from_dense_bcoo(dense)
    raise ValueError(f"unsupported storage kind: {storage}")


def run_srb_case(n: int, warmup: int, runs: int, real_dtype, *, smoke: bool) -> dict[str, float]:
    dense, sparse_by_format, rhs, rhs_batch, rhs_cols = _srb_case(n, real_dtype)
    results: dict[str, float] = {}
    items = sparse_by_format.items()
    if smoke:
        items = (("csr", sparse_by_format["csr"]),)
    for storage, sparse in items:
        for impl in ("point", "basic"):
            prefix = f"srb_{storage}_{impl}"
            results[f"{prefix}_storage_prepare_s"] = _time_call(
                _srb_storage_prepare,
                dense,
                storage,
                warmup=warmup,
                runs=runs,
            )
            cache = mat_wrappers.srb_mat_matvec_cached_prepare_mode(sparse, impl=impl)
            rcache = mat_wrappers.srb_mat_rmatvec_cached_prepare_mode(sparse, impl=impl)
            lu_plan = mat_wrappers.srb_mat_lu_solve_plan_prepare_mode(sparse, impl=impl)
            spd_plan = mat_wrappers.srb_mat_spd_solve_plan_prepare_mode(sparse, impl=impl)
            results[f"{prefix}_cached_prepare_s"] = _time_call(
                lambda a: mat_wrappers.srb_mat_matvec_cached_prepare_mode(a, impl=impl),
                sparse,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_lu_prepare_s"] = _time_call(
                lambda a: mat_wrappers.srb_mat_lu_solve_plan_prepare_mode(a, impl=impl),
                sparse,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_spd_prepare_s"] = _time_call(
                lambda a: mat_wrappers.srb_mat_spd_solve_plan_prepare_mode(a, impl=impl),
                sparse,
                warmup=warmup,
                runs=runs,
            )
            if smoke:
                results[f"{prefix}_matvec_s"] = _time_call(
                    lambda a, v: mat_wrappers.srb_mat_matvec_mode(a, v, impl=impl),
                    sparse,
                    rhs,
                    warmup=warmup,
                    runs=runs,
                )
                results[f"{prefix}_cached_matvec_s"] = _time_call(
                    lambda plan, v: mat_wrappers.srb_mat_matvec_cached_apply_mode(plan, v, impl=impl),
                    cache,
                    rhs,
                    warmup=warmup,
                    runs=runs,
                )
                continue
            results[f"{prefix}_solve_s"] = _time_call(
                lambda a, b: mat_wrappers.srb_mat_spd_solve_mode(a, b, impl=impl),
                sparse,
                rhs,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_lu_plan_solve_s"] = _time_call(
                lambda plan, b: mat_wrappers.srb_mat_solve_lu_mode(plan, b, impl=impl),
                lu_plan,
                rhs,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_transpose_solve_s"] = _time_call(
                lambda plan, b: mat_wrappers.srb_mat_solve_transpose_mode(plan, b, impl=impl),
                lu_plan,
                rhs,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_mat_solve_s"] = _time_call(
                lambda plan, b: mat_wrappers.srb_mat_mat_solve_mode(plan, b, impl=impl),
                spd_plan,
                rhs_cols,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_matvec_s"] = _time_call(
                lambda a, v: mat_wrappers.srb_mat_matvec_mode(a, v, impl=impl),
                sparse,
                rhs,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_cached_matvec_s"] = _time_call(
                lambda plan, v: mat_wrappers.srb_mat_matvec_cached_apply_mode(plan, v, impl=impl),
                cache,
                rhs,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_rmatvec_s"] = _time_call(
                lambda a, v: mat_wrappers.srb_mat_rmatvec_mode(a, v, impl=impl),
                sparse,
                rhs,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_cached_rmatvec_s"] = _time_call(
                lambda plan, v: mat_wrappers.srb_mat_rmatvec_cached_apply_mode(plan, v, impl=impl),
                rcache,
                rhs,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_eigsh_s"] = _time_call(
                lambda a: mat_wrappers.srb_mat_eigsh_mode(a, k=min(2, n), which="smallest", steps=min(n, max(4, 2 * min(2, n))), impl=impl),
                sparse,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_padded_matvec_s"] = _time_call(
                lambda a, vs: mat_wrappers.srb_mat_matvec_batch_mode_padded(a, vs, pad_to=8, impl=impl),
                sparse,
                rhs_batch,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_padded_spd_solve_s"] = _time_call(
                lambda plan, bs: mat_wrappers.srb_mat_spd_solve_plan_apply_batch_mode_padded(plan, bs, pad_to=8, impl=impl),
                spd_plan,
                rhs_batch,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_det_s"] = _time_call(
                lambda a: mat_wrappers.srb_mat_det_mode(a, impl=impl),
                sparse,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_inv_s"] = _time_call(
                lambda a: mat_wrappers.srb_mat_inv_mode(a, impl=impl),
                sparse,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_sqr_s"] = _time_call(
                lambda a: mat_wrappers.srb_mat_sqr_mode(a, impl=impl),
                sparse,
                warmup=warmup,
                runs=runs,
            )
    return results


def run_scb_case(n: int, warmup: int, runs: int, real_dtype, complex_dtype, *, smoke: bool) -> dict[str, float]:
    dense, sparse_by_format, rhs, rhs_batch, rhs_cols = _scb_case(n, real_dtype, complex_dtype)
    results: dict[str, float] = {}
    items = sparse_by_format.items()
    if smoke:
        items = (("csr", sparse_by_format["csr"]),)
    for storage, sparse in items:
        for impl in ("point", "basic"):
            prefix = f"scb_{storage}_{impl}"
            results[f"{prefix}_storage_prepare_s"] = _time_call(
                _scb_storage_prepare,
                dense,
                storage,
                warmup=warmup,
                runs=runs,
            )
            cache = mat_wrappers.scb_mat_matvec_cached_prepare_mode(sparse, impl=impl)
            rcache = mat_wrappers.scb_mat_rmatvec_cached_prepare_mode(sparse, impl=impl)
            lu_plan = mat_wrappers.scb_mat_lu_solve_plan_prepare_mode(sparse, impl=impl)
            hpd_plan = mat_wrappers.scb_mat_hpd_solve_plan_prepare_mode(sparse, impl=impl)
            results[f"{prefix}_cached_prepare_s"] = _time_call(
                lambda a: mat_wrappers.scb_mat_matvec_cached_prepare_mode(a, impl=impl),
                sparse,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_lu_prepare_s"] = _time_call(
                lambda a: mat_wrappers.scb_mat_lu_solve_plan_prepare_mode(a, impl=impl),
                sparse,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_hpd_prepare_s"] = _time_call(
                lambda a: mat_wrappers.scb_mat_hpd_solve_plan_prepare_mode(a, impl=impl),
                sparse,
                warmup=warmup,
                runs=runs,
            )
            if smoke:
                results[f"{prefix}_matvec_s"] = _time_call(
                    lambda a, v: mat_wrappers.scb_mat_matvec_mode(a, v, impl=impl),
                    sparse,
                    rhs,
                    warmup=warmup,
                    runs=runs,
                )
                results[f"{prefix}_cached_matvec_s"] = _time_call(
                    lambda plan, v: mat_wrappers.scb_mat_matvec_cached_apply_mode(plan, v, impl=impl),
                    cache,
                    rhs,
                    warmup=warmup,
                    runs=runs,
                )
                continue
            results[f"{prefix}_solve_s"] = _time_call(
                lambda a, b: mat_wrappers.scb_mat_hpd_solve_mode(a, b, impl=impl),
                sparse,
                rhs,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_lu_plan_solve_s"] = _time_call(
                lambda plan, b: mat_wrappers.scb_mat_solve_lu_mode(plan, b, impl=impl),
                lu_plan,
                rhs,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_transpose_solve_s"] = _time_call(
                lambda plan, b: mat_wrappers.scb_mat_solve_transpose_mode(plan, b, impl=impl),
                lu_plan,
                rhs,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_mat_solve_s"] = _time_call(
                lambda plan, b: mat_wrappers.scb_mat_mat_solve_mode(plan, b, impl=impl),
                hpd_plan,
                rhs_cols,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_matvec_s"] = _time_call(
                lambda a, v: mat_wrappers.scb_mat_matvec_mode(a, v, impl=impl),
                sparse,
                rhs,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_cached_matvec_s"] = _time_call(
                lambda plan, v: mat_wrappers.scb_mat_matvec_cached_apply_mode(plan, v, impl=impl),
                cache,
                rhs,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_rmatvec_s"] = _time_call(
                lambda a, v: mat_wrappers.scb_mat_rmatvec_mode(a, v, impl=impl),
                sparse,
                rhs,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_cached_rmatvec_s"] = _time_call(
                lambda plan, v: mat_wrappers.scb_mat_rmatvec_cached_apply_mode(plan, v, impl=impl),
                rcache,
                rhs,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_eigsh_s"] = _time_call(
                lambda a: mat_wrappers.scb_mat_eigsh_mode(a, k=min(2, n), which="largest", steps=min(n, max(4, 2 * min(2, n))), impl=impl),
                sparse,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_padded_matvec_s"] = _time_call(
                lambda a, vs: mat_wrappers.scb_mat_matvec_batch_mode_padded(a, vs, pad_to=8, impl=impl),
                sparse,
                rhs_batch,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_padded_hpd_solve_s"] = _time_call(
                lambda plan, bs: mat_wrappers.scb_mat_hpd_solve_plan_apply_batch_mode_padded(plan, bs, pad_to=8, impl=impl),
                hpd_plan,
                rhs_batch,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_det_s"] = _time_call(
                lambda a: mat_wrappers.scb_mat_det_mode(a, impl=impl),
                sparse,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_inv_s"] = _time_call(
                lambda a: mat_wrappers.scb_mat_inv_mode(a, impl=impl),
                sparse,
                warmup=warmup,
                runs=runs,
            )
            results[f"{prefix}_sqr_s"] = _time_call(
                lambda a: mat_wrappers.scb_mat_sqr_mode(a, impl=impl),
                sparse,
                warmup=warmup,
                runs=runs,
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sparse real/complex matrix surface in pure JAX.")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--smoke", action="store_true", help="Run only the fast matvec/cached-matvec subset for pytest-owned schema checks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/benchmarks/outputs/matrix/benchmark_sparse_matrix_surface.json"),
    )
    args = parser.parse_args()
    real_dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    complex_dtype = jnp.complex128 if args.dtype == "float64" else jnp.complex64

    print(f"platform: {platform.platform()}")
    print(f"jax: {jax.__version__}")
    print(f"n: {args.n}, warmup: {args.warmup}, runs: {args.runs}, dtype: {args.dtype}, smoke: {args.smoke}")

    results = {}
    results.update(run_srb_case(args.n, args.warmup, args.runs, real_dtype, smoke=args.smoke))
    results.update(run_scb_case(args.n, args.warmup, args.runs, real_dtype, complex_dtype, smoke=args.smoke))
    records: list[BenchmarkRecord] = []
    for key in sorted(results):
        print(f"{key}: {results[key]:.6e}")
        parts = key.split("_")
        algebra = parts[0]
        storage = parts[1]
        implementation = f"{algebra}_{storage}"
        mode = parts[2]
        operation = "_".join(parts[3:-1])
        dtype = args.dtype if algebra == "srb" else ("complex128" if args.dtype == "float64" else "complex64")
        records.append(
            BenchmarkRecord(
                benchmark_name="benchmark_sparse_matrix_surface.py",
                concern="matrix_speed",
                category="matrix_sparse",
                implementation=implementation,
                operation=operation,
                device=jax.default_backend(),
                dtype=dtype,
                warm_time_s=float(results[key]),
                measurements=(
                    BenchmarkMeasurement(name="mode", value=mode),
                    BenchmarkMeasurement(name="n", value=args.n, unit="rows"),
                    BenchmarkMeasurement(name="warmup", value=args.warmup, unit="calls"),
                    BenchmarkMeasurement(name="runs", value=args.runs, unit="calls"),
                    BenchmarkMeasurement(name="requested_dtype", value=args.dtype),
                    BenchmarkMeasurement(name="smoke", value=args.smoke),
                ),
                notes="Legacy sparse benchmark normalized onto the shared benchmark schema; stdout remains metric-style for notebook compatibility.",
            )
        )
    report = BenchmarkReport(
        benchmark_name="benchmark_sparse_matrix_surface.py",
        concern="matrix_speed",
        category="matrix_sparse",
        records=tuple(records),
        environment=collect_runtime_manifest(
            Path(__file__).resolve().parents[1],
            jax_mode="gpu" if jax.default_backend() in {"gpu", "cuda"} else "cpu",
        ),
        notes="Sparse real/complex matrix benchmark. Stdout preserves metric-style lines for notebook compatibility.",
    )
    write_benchmark_report(args.output, report)


if __name__ == "__main__":
    main()
