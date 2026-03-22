from __future__ import annotations

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


import argparse
import platform
import time

import jax
import jax.numpy as jnp

from arbplusjax import mat_wrappers
from arbplusjax import scb_mat
from arbplusjax import srb_mat


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


def _srb_case(n: int):
    base = jnp.reshape(jnp.linspace(0.2, 1.2, n * n, dtype=jnp.float64), (n, n))
    dense = base.T @ base + jnp.eye(n, dtype=jnp.float64) * (n + 1.0)
    sparse = {
        "coo": srb_mat.srb_mat_from_dense_coo(dense),
        "csr": srb_mat.srb_mat_from_dense_csr(dense),
        "bcoo": srb_mat.srb_mat_from_dense_bcoo(dense),
    }
    rhs = jnp.linspace(-0.5, 0.75, n, dtype=jnp.float64)
    rhs_batch = jnp.stack([rhs, rhs + 1.0], axis=0)
    rhs_cols = jnp.stack([rhs, rhs + 1.0], axis=1)
    return dense, sparse, rhs, rhs_batch, rhs_cols


def _scb_case(n: int):
    real = jnp.reshape(jnp.linspace(0.2, 1.1, n * n, dtype=jnp.float64), (n, n))
    imag = jnp.reshape(jnp.linspace(-0.3, 0.3, n * n, dtype=jnp.float64), (n, n))
    base = real + 1j * imag
    dense = jnp.conj(base.T) @ base + jnp.eye(n, dtype=jnp.complex128) * (n + 1.0)
    sparse = {
        "coo": scb_mat.scb_mat_from_dense_coo(dense),
        "csr": scb_mat.scb_mat_from_dense_csr(dense),
        "bcoo": scb_mat.scb_mat_from_dense_bcoo(dense),
    }
    rhs = jnp.linspace(-0.5, 0.75, n, dtype=jnp.float64) + 1j * jnp.linspace(0.1, 0.4, n, dtype=jnp.float64)
    rhs_batch = jnp.stack([rhs, rhs + (0.25 - 0.1j)], axis=0)
    rhs_cols = jnp.stack([rhs, rhs + (0.25 - 0.1j)], axis=1)
    return dense, sparse, rhs, rhs_batch, rhs_cols


def run_srb_case(n: int, warmup: int, runs: int) -> dict[str, float]:
    _, sparse_by_format, rhs, rhs_batch, rhs_cols = _srb_case(n)
    results: dict[str, float] = {}
    for storage, sparse in sparse_by_format.items():
        for impl in ("point", "basic"):
            cache = mat_wrappers.srb_mat_matvec_cached_prepare_mode(sparse, impl=impl)
            rcache = mat_wrappers.srb_mat_rmatvec_cached_prepare_mode(sparse, impl=impl)
            lu_plan = mat_wrappers.srb_mat_lu_solve_plan_prepare_mode(sparse, impl=impl)
            spd_plan = mat_wrappers.srb_mat_spd_solve_plan_prepare_mode(sparse, impl=impl)
            prefix = f"srb_{storage}_{impl}"
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


def run_scb_case(n: int, warmup: int, runs: int) -> dict[str, float]:
    _, sparse_by_format, rhs, rhs_batch, rhs_cols = _scb_case(n)
    results: dict[str, float] = {}
    for storage, sparse in sparse_by_format.items():
        for impl in ("point", "basic"):
            cache = mat_wrappers.scb_mat_matvec_cached_prepare_mode(sparse, impl=impl)
            rcache = mat_wrappers.scb_mat_rmatvec_cached_prepare_mode(sparse, impl=impl)
            lu_plan = mat_wrappers.scb_mat_lu_solve_plan_prepare_mode(sparse, impl=impl)
            hpd_plan = mat_wrappers.scb_mat_hpd_solve_plan_prepare_mode(sparse, impl=impl)
            prefix = f"scb_{storage}_{impl}"
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
    args = parser.parse_args()

    print(f"platform: {platform.platform()}")
    print(f"jax: {jax.__version__}")
    print(f"n: {args.n}, warmup: {args.warmup}, runs: {args.runs}")

    results = {}
    results.update(run_srb_case(args.n, args.warmup, args.runs))
    results.update(run_scb_case(args.n, args.warmup, args.runs))
    for key in sorted(results):
        print(f"{key}: {results[key]:.6e}")


if __name__ == "__main__":
    main()
