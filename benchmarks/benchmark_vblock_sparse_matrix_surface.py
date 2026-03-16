from __future__ import annotations

import argparse
import platform
import time

import jax
import jax.numpy as jnp

from arbplusjax import scb_vblock_mat
from arbplusjax import srb_vblock_mat


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


def _real_case(n: int):
    dense = jnp.zeros((n, n), dtype=jnp.float64)
    for i in range(n):
        dense = dense.at[i, i].set(2.0 + 0.25 * i)
        if i + 1 < n:
            dense = dense.at[i + 1, i].set(0.5)
            dense = dense.at[i, i + 1].set(0.25)
    vec = jnp.linspace(-1.0, 1.0, n, dtype=jnp.float64)
    rhs = jnp.stack([vec, vec + 1.0], axis=-1)
    sizes = _row_col_sizes(n)
    x = srb_vblock_mat.srb_vblock_mat_from_dense_csr(dense, row_block_sizes=sizes, col_block_sizes=sizes)
    plan = srb_vblock_mat.srb_vblock_mat_matvec_cached_prepare(x)
    return x, plan, vec, rhs


def _complex_case(n: int):
    dense = jnp.zeros((n, n), dtype=jnp.complex128)
    for i in range(n):
        dense = dense.at[i, i].set(2.0 + 0.25j + 0.2 * i)
        if i + 1 < n:
            dense = dense.at[i + 1, i].set(0.5 - 0.1j)
            dense = dense.at[i, i + 1].set(0.25 + 0.05j)
    vec = jnp.linspace(-1.0, 1.0, n, dtype=jnp.float64).astype(jnp.complex128)
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
    args = parser.parse_args()

    print(f"platform: {platform.platform()}")
    print(f"jax_backend: {jax.default_backend()}")
    print(f"n: {args.n}")
    print(f"warmup: {args.warmup}")
    print(f"runs: {args.runs}")

    rmat, rplan, rvec, rrhs = _real_case(args.n)
    cmat, cplan, cvec, crhs = _complex_case(args.n)
    stats = {
        "srb_vblock_matvec_s": _time_call(jax.jit(srb_vblock_mat.srb_vblock_mat_matvec), rmat, rvec, warmup=args.warmup, runs=args.runs),
        "srb_vblock_matvec_cached_s": _time_call(jax.jit(srb_vblock_mat.srb_vblock_mat_matvec_cached_apply), rplan, rvec, warmup=args.warmup, runs=args.runs),
        "srb_vblock_solve_s": _time_call(
            lambda a, b: srb_vblock_mat.srb_vblock_mat_solve(a, b, method="lu"),
            rmat,
            rvec,
            warmup=args.warmup,
            runs=args.runs,
        ),
        "scb_vblock_matvec_s": _time_call(jax.jit(scb_vblock_mat.scb_vblock_mat_matvec), cmat, cvec, warmup=args.warmup, runs=args.runs),
        "scb_vblock_matvec_cached_s": _time_call(jax.jit(scb_vblock_mat.scb_vblock_mat_matvec_cached_apply), cplan, cvec, warmup=args.warmup, runs=args.runs),
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
    for key, value in stats.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
