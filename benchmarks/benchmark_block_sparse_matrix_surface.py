from __future__ import annotations

import argparse
import platform
import time

import jax
import jax.numpy as jnp

from arbplusjax import scb_block_mat
from arbplusjax import srb_block_mat


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


def _real_case(n_blocks: int, block_size: int):
    n = n_blocks * block_size
    dense = jnp.zeros((n, n), dtype=jnp.float64)
    eye_block = jnp.eye(block_size, dtype=jnp.float64)
    for i in range(n_blocks):
        dense = dense.at[i * block_size : (i + 1) * block_size, i * block_size : (i + 1) * block_size].set((i + 2.0) * eye_block)
        if i + 1 < n_blocks:
            dense = dense.at[(i + 1) * block_size : (i + 2) * block_size, i * block_size : (i + 1) * block_size].set(0.25 * eye_block)
    vec = jnp.linspace(-1.0, 1.0, n, dtype=jnp.float64)
    rhs = jnp.stack([vec, vec + 1.0], axis=-1)
    return srb_block_mat.srb_block_mat_from_dense_csr(dense, block_shape=(block_size, block_size)), vec, rhs


def _complex_case(n_blocks: int, block_size: int):
    n = n_blocks * block_size
    dense = jnp.zeros((n, n), dtype=jnp.complex128)
    eye_block = jnp.eye(block_size, dtype=jnp.complex128)
    for i in range(n_blocks):
        dense = dense.at[i * block_size : (i + 1) * block_size, i * block_size : (i + 1) * block_size].set((i + 2.0 + 0.25j) * eye_block)
        if i + 1 < n_blocks:
            dense = dense.at[(i + 1) * block_size : (i + 2) * block_size, i * block_size : (i + 1) * block_size].set((0.25 - 0.1j) * eye_block)
    vec = jnp.linspace(-1.0, 1.0, n, dtype=jnp.float64).astype(jnp.complex128)
    rhs = jnp.stack([vec, vec + (1.0 - 0.25j)], axis=-1)
    return scb_block_mat.scb_block_mat_from_dense_csr(dense, block_shape=(block_size, block_size)), vec, rhs


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark block-sparse BSR-like matrix surface.")
    parser.add_argument("--n-blocks", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    print(f"platform: {platform.platform()}")
    print(f"jax_backend: {jax.default_backend()}")
    print(f"n_blocks: {args.n_blocks}")
    print(f"block_size: {args.block_size}")
    print(f"warmup: {args.warmup}")
    print(f"runs: {args.runs}")

    rmat, rvec, rrhs = _real_case(args.n_blocks, args.block_size)
    cmat, cvec, crhs = _complex_case(args.n_blocks, args.block_size)
    stats = {
        "srb_block_matvec_s": _time_call(jax.jit(srb_block_mat.srb_block_mat_matvec), rmat, rvec, warmup=args.warmup, runs=args.runs),
        "srb_block_matmul_dense_rhs_s": _time_call(jax.jit(srb_block_mat.srb_block_mat_matmul_dense_rhs), rmat, rrhs, warmup=args.warmup, runs=args.runs),
        "scb_block_matvec_s": _time_call(jax.jit(scb_block_mat.scb_block_mat_matvec), cmat, cvec, warmup=args.warmup, runs=args.runs),
        "scb_block_matmul_dense_rhs_s": _time_call(jax.jit(scb_block_mat.scb_block_mat_matmul_dense_rhs), cmat, crhs, warmup=args.warmup, runs=args.runs),
    }
    for key, value in stats.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
