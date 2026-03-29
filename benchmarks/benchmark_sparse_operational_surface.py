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
mat_wrappers = None
scb_mat = None
srb_mat = None


def _load_modules() -> None:
    global api, mat_wrappers, scb_mat, srb_mat
    if api is None:
        from arbplusjax import api as _api
        from arbplusjax import mat_wrappers as _mat_wrappers
        from arbplusjax import scb_mat as _scb_mat
        from arbplusjax import srb_mat as _srb_mat

        api = _api
        mat_wrappers = _mat_wrappers
        scb_mat = _scb_mat
        srb_mat = _srb_mat


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


def _backend_name() -> str:
    backend = jax.default_backend()
    if backend == "gpu":
        return "gpu"
    return "cpu"


def _real_case(n: int, dtype):
    _load_modules()
    dense = jnp.diag(jnp.linspace(2.0, 2.0 + n - 1, n, dtype=dtype))
    dense = dense + jnp.diag(jnp.linspace(0.5, 1.0, n - 1, dtype=dtype), 1)
    dense = dense + jnp.diag(jnp.linspace(-0.25, 0.25, n - 1, dtype=dtype), -1)
    sparse = {
        "coo": srb_mat.srb_mat_from_dense_coo(dense),
        "csr": srb_mat.srb_mat_from_dense_csr(dense),
        "bcoo": srb_mat.srb_mat_from_dense_bcoo(dense),
    }
    vec = jnp.linspace(-0.5, 0.75, n, dtype=dtype)
    batch = jnp.stack(
        [
            vec,
            vec + 0.25,
            vec + 0.5,
            vec + 0.75,
            vec - 0.25,
            vec - 0.5,
            vec * 0.5,
            vec * -1.0,
        ],
        axis=0,
    )
    return dense, sparse, vec, batch


def _complex_case(n: int, real_dtype, complex_dtype):
    _load_modules()
    dense = jnp.diag(jnp.linspace(2.0, 2.0 + n - 1, n, dtype=real_dtype)).astype(complex_dtype)
    dense = dense + jnp.diag(jnp.linspace(0.5, 1.0, n - 1, dtype=real_dtype) + 1j * jnp.linspace(0.1, 0.2, n - 1, dtype=real_dtype), 1)
    dense = dense + jnp.diag(jnp.linspace(-0.25, 0.25, n - 1, dtype=real_dtype) - 1j * jnp.linspace(0.05, 0.15, n - 1, dtype=real_dtype), -1)
    sparse = {
        "coo": scb_mat.scb_mat_from_dense_coo(dense),
        "csr": scb_mat.scb_mat_from_dense_csr(dense),
        "bcoo": scb_mat.scb_mat_from_dense_bcoo(dense),
    }
    vec = jnp.linspace(-0.5, 0.75, n, dtype=real_dtype) + 1j * jnp.linspace(0.1, 0.4, n, dtype=real_dtype)
    batch = jnp.stack(
        [
            vec,
            vec + (0.25 - 0.1j),
            vec + (0.5 - 0.2j),
            vec + (0.75 - 0.3j),
            vec - (0.25 - 0.1j),
            vec - (0.5 - 0.2j),
            vec * (0.5 + 0.0j),
            vec * (-1.0 + 0.0j),
        ],
        axis=0,
    )
    return dense, sparse, vec, batch


def _record(results: list[BenchmarkRecord], *, category: str, case: str, backend: str, n: int, dtype: str, metric: str, seconds: float):
    results.append(
        BenchmarkRecord(
            benchmark_name="benchmark_sparse_operational_surface",
            concern="sparse operational point/basic CPU/GPU",
            category=category,
            implementation=case,
            operation=metric,
            device=backend,
            dtype=dtype,
            warm_time_s=float(seconds),
            measurements=(BenchmarkMeasurement(name="n", value=n, unit=""),),
        )
    )


def _run_real(results: list[BenchmarkRecord], *, n: int, warmup: int, runs: int, dtype_name: str) -> None:
    real_dtype = jnp.float64 if dtype_name == "float64" else jnp.float32
    dense, sparse_by_format, vec, batch = _real_case(n, real_dtype)
    backend = _backend_name()
    for storage, sparse in sparse_by_format.items():
        point_plan = mat_wrappers.srb_mat_matvec_cached_prepare_mode(sparse, impl="point")
        point_rplan = mat_wrappers.srb_mat_rmatvec_cached_prepare_mode(sparse, impl="point")
        basic_plan = mat_wrappers.srb_mat_matvec_cached_prepare_mode(sparse, impl="basic")
        basic_rplan = mat_wrappers.srb_mat_rmatvec_cached_prepare_mode(sparse, impl="basic")
        point_bound = api.bind_point_batch_jit("srb_mat_matvec_cached_apply", dtype=dtype_name, pad_to=8, backend=backend)
        basic_bound = api.bind_interval_batch_jit("srb_mat_matvec_cached_apply", mode="basic", dtype=dtype_name, pad_to=8, backend=backend)

        for impl, plan, rplan in (("point", point_plan, point_rplan), ("basic", basic_plan, basic_rplan)):
            prefix = f"srb_{storage}_{impl}"
            _record(results, category="sparse_operational", case=prefix, backend=backend, n=n, dtype=dtype_name, metric="matvec_s", seconds=_time_call(lambda a, v: mat_wrappers.srb_mat_matvec_mode(a, v, impl=impl), sparse, vec, warmup=warmup, runs=runs))
            _record(results, category="sparse_operational", case=prefix, backend=backend, n=n, dtype=dtype_name, metric="rmatvec_s", seconds=_time_call(lambda a, v: mat_wrappers.srb_mat_rmatvec_mode(a, v, impl=impl), sparse, vec, warmup=warmup, runs=runs))
            _record(results, category="sparse_operational", case=prefix, backend=backend, n=n, dtype=dtype_name, metric="cached_prepare_s", seconds=_time_call(lambda a: mat_wrappers.srb_mat_matvec_cached_prepare_mode(a, impl=impl), sparse, warmup=warmup, runs=runs))
            _record(results, category="sparse_operational", case=prefix, backend=backend, n=n, dtype=dtype_name, metric="cached_apply_s", seconds=_time_call(lambda p, v: mat_wrappers.srb_mat_matvec_cached_apply_mode(p, v, impl=impl), plan, vec, warmup=warmup, runs=runs))
            _record(results, category="sparse_operational", case=prefix, backend=backend, n=n, dtype=dtype_name, metric="cached_rapply_s", seconds=_time_call(lambda p, v: mat_wrappers.srb_mat_rmatvec_cached_apply_mode(p, v, impl=impl), rplan, vec, warmup=warmup, runs=runs))

        _record(results, category="sparse_operational", case=f"srb_{storage}_point", backend=backend, n=n, dtype=dtype_name, metric="jit_cached_apply_batch_s", seconds=_time_call(point_bound, point_plan, batch, warmup=warmup, runs=runs))
        _record(results, category="sparse_operational", case=f"srb_{storage}_basic", backend=backend, n=n, dtype=dtype_name, metric="jit_cached_apply_batch_s", seconds=_time_call(basic_bound, basic_plan, batch, warmup=warmup, runs=runs))


def _run_complex(results: list[BenchmarkRecord], *, n: int, warmup: int, runs: int, dtype_name: str) -> None:
    real_dtype = jnp.float64 if dtype_name == "float64" else jnp.float32
    complex_dtype = jnp.complex128 if dtype_name == "float64" else jnp.complex64
    _dense, sparse_by_format, vec, batch = _complex_case(n, real_dtype, complex_dtype)
    backend = _backend_name()
    for storage, sparse in sparse_by_format.items():
        point_plan = mat_wrappers.scb_mat_matvec_cached_prepare_mode(sparse, impl="point")
        point_rplan = mat_wrappers.scb_mat_rmatvec_cached_prepare_mode(sparse, impl="point")
        basic_plan = mat_wrappers.scb_mat_matvec_cached_prepare_mode(sparse, impl="basic")
        basic_rplan = mat_wrappers.scb_mat_rmatvec_cached_prepare_mode(sparse, impl="basic")
        point_bound = api.bind_point_batch_jit("scb_mat_matvec_cached_apply", dtype=dtype_name, pad_to=8, backend=backend)
        basic_bound = api.bind_interval_batch_jit("scb_mat_matvec_cached_apply", mode="basic", dtype=dtype_name, pad_to=8, backend=backend)

        for impl, plan, rplan in (("point", point_plan, point_rplan), ("basic", basic_plan, basic_rplan)):
            prefix = f"scb_{storage}_{impl}"
            _record(results, category="sparse_operational", case=prefix, backend=backend, n=n, dtype=dtype_name, metric="matvec_s", seconds=_time_call(lambda a, v: mat_wrappers.scb_mat_matvec_mode(a, v, impl=impl), sparse, vec, warmup=warmup, runs=runs))
            _record(results, category="sparse_operational", case=prefix, backend=backend, n=n, dtype=dtype_name, metric="rmatvec_s", seconds=_time_call(lambda a, v: mat_wrappers.scb_mat_rmatvec_mode(a, v, impl=impl), sparse, vec, warmup=warmup, runs=runs))
            _record(results, category="sparse_operational", case=prefix, backend=backend, n=n, dtype=dtype_name, metric="cached_prepare_s", seconds=_time_call(lambda a: mat_wrappers.scb_mat_matvec_cached_prepare_mode(a, impl=impl), sparse, warmup=warmup, runs=runs))
            _record(results, category="sparse_operational", case=prefix, backend=backend, n=n, dtype=dtype_name, metric="cached_apply_s", seconds=_time_call(lambda p, v: mat_wrappers.scb_mat_matvec_cached_apply_mode(p, v, impl=impl), plan, vec, warmup=warmup, runs=runs))
            _record(results, category="sparse_operational", case=prefix, backend=backend, n=n, dtype=dtype_name, metric="cached_rapply_s", seconds=_time_call(lambda p, v: mat_wrappers.scb_mat_rmatvec_cached_apply_mode(p, v, impl=impl), rplan, vec, warmup=warmup, runs=runs))

        _record(results, category="sparse_operational", case=f"scb_{storage}_point", backend=backend, n=n, dtype=dtype_name, metric="jit_cached_apply_batch_s", seconds=_time_call(point_bound, point_plan, batch, warmup=warmup, runs=runs))
        _record(results, category="sparse_operational", case=f"scb_{storage}_basic", backend=backend, n=n, dtype=dtype_name, metric="jit_cached_apply_batch_s", seconds=_time_call(basic_bound, basic_plan, batch, warmup=warmup, runs=runs))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark sparse point/basic operational surfaces without dense-fallback-owned kernels.")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    _load_modules()
    results: list[BenchmarkRecord] = []
    _run_real(results, n=args.n, warmup=args.warmup, runs=args.runs, dtype_name=args.dtype)
    _run_complex(results, n=args.n, warmup=args.warmup, runs=args.runs, dtype_name=args.dtype)

    report = BenchmarkReport(
        benchmark_name="benchmark_sparse_operational_surface",
        concern="sparse operational point/basic CPU/GPU",
        category="sparse_operational",
        records=tuple(results),
        environment=collect_runtime_manifest(Path(__file__).resolve().parents[1], jax_mode=_backend_name(), python_path=None),
        notes=f"generated_at={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
    )
    if args.output is not None:
        write_benchmark_report(args.output, report)
    else:
        for record in results:
            print(f"{record.case}.{record.measurement.name}: {record.measurement.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
