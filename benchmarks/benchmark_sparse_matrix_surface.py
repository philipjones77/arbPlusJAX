from __future__ import annotations

import argparse
import platform
import time

import jax
import jax.numpy as jnp

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


def _real_case(n: int):
    dense = jnp.eye(n, dtype=jnp.float64) * 3.0
    dense = dense + jnp.diag(jnp.full((n - 1,), 0.25, dtype=jnp.float64), k=1)
    dense = dense + jnp.diag(jnp.full((n - 1,), -0.5, dtype=jnp.float64), k=-1)
    vec = jnp.linspace(-1.0, 1.0, n, dtype=jnp.float64)
    rhs = jnp.stack([vec, vec + 1.0], axis=-1)
    return srb_mat.srb_mat_from_dense_coo(dense), srb_mat.srb_mat_from_dense_csr(dense), srb_mat.srb_mat_from_dense_bcoo(dense), vec, rhs


def _complex_case(n: int):
    dense = jnp.eye(n, dtype=jnp.complex128) * (2.0 + 0.5j)
    dense = dense + jnp.diag(jnp.full((n - 1,), 0.25 - 0.1j, dtype=jnp.complex128), k=1)
    dense = dense + jnp.diag(jnp.full((n - 1,), -0.5 + 0.2j, dtype=jnp.complex128), k=-1)
    vec = jnp.linspace(-1.0, 1.0, n, dtype=jnp.float64) + 1j * jnp.linspace(0.25, -0.25, n, dtype=jnp.float64)
    rhs = jnp.stack([vec, vec + (1.0 - 0.25j)], axis=-1)
    return scb_mat.scb_mat_from_dense_coo(dense), scb_mat.scb_mat_from_dense_csr(dense), scb_mat.scb_mat_from_dense_bcoo(dense), vec, rhs


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sparse srb_mat/scb_mat matvec and cached matvec paths.")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    print(f"platform: {platform.platform()}")
    print(f"jax_backend: {jax.default_backend()}")
    print(f"n: {args.n}")
    print(f"warmup: {args.warmup}")
    print(f"runs: {args.runs}")

    rcoo, rcsr, rbcoo, rx, rrhs = _real_case(args.n)
    ccoo, ccsr, cbcoo, cx, crhs = _complex_case(args.n)
    rplan = srb_mat.srb_mat_matvec_cached_prepare(rcsr)
    cplan = scb_mat.scb_mat_matvec_cached_prepare(ccsr)

    stats = {
        "srb_coo_matvec_s": _time_call(jax.jit(srb_mat.srb_mat_matvec), rcoo, rx, warmup=args.warmup, runs=args.runs),
        "srb_csr_matvec_s": _time_call(jax.jit(srb_mat.srb_mat_matvec), rcsr, rx, warmup=args.warmup, runs=args.runs),
        "srb_bcoo_matvec_s": _time_call(jax.jit(srb_mat.srb_mat_matvec), rbcoo, rx, warmup=args.warmup, runs=args.runs),
        "srb_cached_matvec_s": _time_call(jax.jit(srb_mat.srb_mat_matvec_cached_apply), rplan, rx, warmup=args.warmup, runs=args.runs),
        "srb_csr_matmul_dense_rhs_s": _time_call(jax.jit(srb_mat.srb_mat_matmul_dense_rhs), rcsr, rrhs, warmup=args.warmup, runs=args.runs),
        "srb_sparse_sparse_matmul_s": _time_call(jax.jit(srb_mat.srb_mat_matmul_sparse), rcsr, rbcoo, warmup=args.warmup, runs=args.runs),
        "srb_sparse_lu_s": _time_call(srb_mat.srb_mat_lu, rcsr, warmup=args.warmup, runs=args.runs),
        "srb_sparse_qr_factor_s": _time_call(srb_mat.srb_mat_qr, rcsr, warmup=args.warmup, runs=args.runs),
        "scb_coo_matvec_s": _time_call(jax.jit(scb_mat.scb_mat_matvec), ccoo, cx, warmup=args.warmup, runs=args.runs),
        "scb_csr_matvec_s": _time_call(jax.jit(scb_mat.scb_mat_matvec), ccsr, cx, warmup=args.warmup, runs=args.runs),
        "scb_bcoo_matvec_s": _time_call(jax.jit(scb_mat.scb_mat_matvec), cbcoo, cx, warmup=args.warmup, runs=args.runs),
        "scb_cached_matvec_s": _time_call(jax.jit(scb_mat.scb_mat_matvec_cached_apply), cplan, cx, warmup=args.warmup, runs=args.runs),
        "scb_csr_matmul_dense_rhs_s": _time_call(jax.jit(scb_mat.scb_mat_matmul_dense_rhs), ccsr, crhs, warmup=args.warmup, runs=args.runs),
        "scb_sparse_sparse_matmul_s": _time_call(jax.jit(scb_mat.scb_mat_matmul_sparse), ccsr, cbcoo, warmup=args.warmup, runs=args.runs),
        "scb_sparse_lu_s": _time_call(scb_mat.scb_mat_lu, ccsr, warmup=args.warmup, runs=args.runs),
        "scb_sparse_qr_factor_s": _time_call(scb_mat.scb_mat_qr, ccsr, warmup=args.warmup, runs=args.runs),
    }
    for key, value in stats.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
