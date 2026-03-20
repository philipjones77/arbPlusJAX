from __future__ import annotations

import argparse
import platform
import time

import jax
import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import acb_mat
from arbplusjax import arb_mat
from arbplusjax import double_interval as di


def _interval_matrix_from_point(a: jax.Array) -> jax.Array:
    return di.interval(a, a)


def _interval_vector_from_point(x: jax.Array) -> jax.Array:
    return di.interval(x, x)


def _box_matrix_from_point(a: jax.Array) -> jax.Array:
    return acb_core.acb_box(
        di.interval(jnp.real(a), jnp.real(a)),
        di.interval(jnp.imag(a), jnp.imag(a)),
    )


def _box_vector_from_point(x: jax.Array) -> jax.Array:
    return acb_core.acb_box(
        di.interval(jnp.real(x), jnp.real(x)),
        di.interval(jnp.imag(x), jnp.imag(x)),
    )


def _arb_dense_case(n: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    a_mid = jnp.eye(n, dtype=jnp.float64) * 3.0 + jnp.triu(jnp.ones((n, n), dtype=jnp.float64), k=1) * 0.1
    x_mid = jnp.reshape(jnp.linspace(0.5, 1.5, n * 2, dtype=jnp.float64), (n, 2))
    v_mid = jnp.linspace(-0.25, 0.75, n, dtype=jnp.float64)
    return _interval_matrix_from_point(a_mid), _interval_matrix_from_point(x_mid), _interval_vector_from_point(v_mid)


def _arb_spd_case(n: int) -> tuple[jax.Array, jax.Array]:
    base = jnp.reshape(jnp.linspace(0.2, 1.2, n * n, dtype=jnp.float64), (n, n))
    mid = base.T @ base + jnp.eye(n, dtype=jnp.float64) * (n + 1.0)
    rhs = jnp.reshape(jnp.linspace(-0.5, 0.75, n * 2, dtype=jnp.float64), (n, 2))
    return _interval_matrix_from_point(mid), _interval_matrix_from_point(rhs)


def _acb_dense_case(n: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    real = jnp.eye(n, dtype=jnp.float64) * 2.5 + jnp.triu(jnp.ones((n, n), dtype=jnp.float64), k=1) * 0.1
    imag = jnp.tril(jnp.ones((n, n), dtype=jnp.float64), k=-1) * 0.05
    a_mid = real + 1j * imag
    x_mid = jnp.reshape(
        jnp.linspace(0.5, 1.5, n * 2, dtype=jnp.float64) + 1j * jnp.linspace(-0.2, 0.2, n * 2, dtype=jnp.float64),
        (n, 2),
    )
    v_mid = jnp.linspace(-0.25, 0.75, n, dtype=jnp.float64) + 1j * jnp.linspace(0.1, 0.3, n, dtype=jnp.float64)
    return _box_matrix_from_point(a_mid), _box_matrix_from_point(x_mid), _box_vector_from_point(v_mid)


def _acb_hpd_case(n: int) -> tuple[jax.Array, jax.Array]:
    real = jnp.reshape(jnp.linspace(0.2, 1.1, n * n, dtype=jnp.float64), (n, n))
    imag = jnp.reshape(jnp.linspace(-0.3, 0.3, n * n, dtype=jnp.float64), (n, n))
    base = real + 1j * imag
    mid = jnp.conj(base.T) @ base + jnp.eye(n, dtype=jnp.complex128) * (n + 1.0)
    rhs = jnp.reshape(
        jnp.linspace(-0.5, 0.75, n * 2, dtype=jnp.float64) + 1j * jnp.linspace(0.1, 0.4, n * 2, dtype=jnp.float64),
        (n, 2),
    )
    return _box_matrix_from_point(mid), _box_matrix_from_point(rhs)


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


def run_arb_case(n: int, warmup: int, runs: int) -> dict[str, float]:
    a, x, vec = _arb_dense_case(n)
    spd_a, spd_rhs = _arb_spd_case(n)
    rhs = _interval_matrix_from_point(di.midpoint(a) @ di.midpoint(x))
    lu_plan = arb_mat.arb_mat_dense_lu_solve_plan_prepare(a)
    cache = arb_mat.arb_mat_dense_matvec_plan_prepare(a)
    spd_plan = arb_mat.arb_mat_dense_spd_solve_plan_prepare(spd_a)
    vec_batch = jnp.stack([vec, vec + _interval_vector_from_point(jnp.ones_like(di.midpoint(vec)))], axis=0)
    split = max(1, n // 2)
    arb_blocks = (
        (arb_mat.arb_mat_submatrix(a, 0, split, 0, split), arb_mat.arb_mat_submatrix(a, 0, split, split, n)),
        (arb_mat.arb_mat_submatrix(a, split, n, 0, split), arb_mat.arb_mat_submatrix(a, split, n, split, n)),
    )

    direct_solve = jax.jit(lambda aa, bb: arb_mat.arb_mat_solve(aa, bb))
    lu_solve = jax.jit(lambda plan, bb: arb_mat.arb_mat_dense_lu_solve_plan_apply(plan, bb))
    lu_solve_transpose = jax.jit(lambda plan, bb: arb_mat.arb_mat_solve_transpose(plan, bb))
    lu_solve_add = jax.jit(lambda plan, bb, yy: arb_mat.arb_mat_solve_add(plan, bb, yy))
    cached_matvec = jax.jit(lambda plan, xx: arb_mat.arb_mat_dense_matvec_plan_apply(plan, xx))
    prepare_plan = lambda aa: arb_mat.arb_mat_dense_matvec_plan_prepare(aa)
    cached_matvec_padded = lambda plan, xx: arb_mat.arb_mat_dense_matvec_plan_apply_batch_padded(plan, xx, pad_to=8)
    add = jax.jit(lambda aa, bb: arb_mat.arb_mat_add(aa, bb))
    entrywise = jax.jit(lambda aa, bb: arb_mat.arb_mat_mul_entrywise(aa, bb))
    charpoly = jax.jit(lambda aa: arb_mat.arb_mat_charpoly(aa))
    power2 = jax.jit(lambda aa: arb_mat.arb_mat_pow_ui(aa, 2))
    expm = jax.jit(lambda aa: arb_mat.arb_mat_exp(aa))
    spd_solve = jax.jit(lambda aa, bb: arb_mat.arb_mat_spd_solve(aa, bb))
    spd_plan_solve = jax.jit(lambda plan, bb: arb_mat.arb_mat_dense_spd_solve_plan_apply(plan, bb))
    spd_eigh = jax.jit(lambda aa: arb_mat.arb_mat_eigh(aa))
    solve_tril = jax.jit(lambda aa, bb: arb_mat.arb_mat_solve_tril(aa, bb))
    solve_lu = jax.jit(lambda aa, bb: arb_mat.arb_mat_solve_lu(aa, bb))
    transpose = jax.jit(arb_mat.arb_mat_transpose)
    diag = jax.jit(arb_mat.arb_mat_diag)
    block_assemble = lambda blocks: arb_mat.arb_mat_block_assemble(blocks)
    block_diag = lambda blocks: arb_mat.arb_mat_block_diag((blocks[0][0], blocks[1][1]))

    return {
        "arb_direct_solve_s": _time_call(direct_solve, a, rhs, warmup=warmup, runs=runs),
        "arb_lu_reuse_s": _time_call(lu_solve, lu_plan, rhs, warmup=warmup, runs=runs),
        "arb_lu_reuse_transpose_s": _time_call(lu_solve_transpose, lu_plan, rhs, warmup=warmup, runs=runs),
        "arb_lu_reuse_add_s": _time_call(lu_solve_add, lu_plan, rhs, rhs, warmup=warmup, runs=runs),
        "arb_dense_plan_prepare_s": _time_call(prepare_plan, a, warmup=warmup, runs=runs),
        "arb_cached_matvec_s": _time_call(cached_matvec, cache, vec, warmup=warmup, runs=runs),
        "arb_cached_matvec_padded_s": _time_call(cached_matvec_padded, cache, vec_batch, warmup=warmup, runs=runs),
        "arb_add_s": _time_call(add, a, a, warmup=warmup, runs=runs),
        "arb_mul_entrywise_s": _time_call(entrywise, a, a, warmup=warmup, runs=runs),
        "arb_charpoly_s": _time_call(charpoly, a, warmup=warmup, runs=runs),
        "arb_pow_ui_s": _time_call(power2, a, warmup=warmup, runs=runs),
        "arb_exp_s": _time_call(expm, spd_a, warmup=warmup, runs=runs),
        "arb_spd_solve_s": _time_call(spd_solve, spd_a, spd_rhs, warmup=warmup, runs=runs),
        "arb_spd_plan_solve_s": _time_call(spd_plan_solve, spd_plan, spd_rhs, warmup=warmup, runs=runs),
        "arb_spd_eigh_s": _time_call(spd_eigh, spd_a, warmup=warmup, runs=runs),
        "arb_solve_tril_s": _time_call(solve_tril, arb_mat.arb_mat_cho(spd_a), spd_rhs, warmup=warmup, runs=runs),
        "arb_solve_lu_s": _time_call(solve_lu, a, rhs, warmup=warmup, runs=runs),
        "arb_transpose_s": _time_call(transpose, a, warmup=warmup, runs=runs),
        "arb_diag_s": _time_call(diag, a, warmup=warmup, runs=runs),
        "arb_block_assemble_s": _time_call(block_assemble, arb_blocks, warmup=warmup, runs=runs),
        "arb_block_diag_s": _time_call(block_diag, arb_blocks, warmup=warmup, runs=runs),
    }


def run_acb_case(n: int, warmup: int, runs: int) -> dict[str, float]:
    a, x, vec = _acb_dense_case(n)
    hpd_a, hpd_rhs = _acb_hpd_case(n)
    rhs = _box_matrix_from_point(acb_core.acb_midpoint(a) @ acb_core.acb_midpoint(x))
    lu_plan = acb_mat.acb_mat_dense_lu_solve_plan_prepare(a)
    cache = acb_mat.acb_mat_dense_matvec_plan_prepare(a)
    hpd_plan = acb_mat.acb_mat_dense_hpd_solve_plan_prepare(hpd_a)
    vec_batch = jnp.stack([vec, vec], axis=0)
    split = max(1, n // 2)
    acb_blocks = (
        (acb_mat.acb_mat_submatrix(a, 0, split, 0, split), acb_mat.acb_mat_submatrix(a, 0, split, split, n)),
        (acb_mat.acb_mat_submatrix(a, split, n, 0, split), acb_mat.acb_mat_submatrix(a, split, n, split, n)),
    )

    direct_solve = jax.jit(lambda aa, bb: acb_mat.acb_mat_solve(aa, bb))
    lu_solve = jax.jit(lambda plan, bb: acb_mat.acb_mat_dense_lu_solve_plan_apply(plan, bb))
    lu_solve_transpose = jax.jit(lambda plan, bb: acb_mat.acb_mat_solve_transpose(plan, bb))
    lu_solve_add = jax.jit(lambda plan, bb, yy: acb_mat.acb_mat_solve_add(plan, bb, yy))
    cached_matvec = jax.jit(lambda plan, xx: acb_mat.acb_mat_dense_matvec_plan_apply(plan, xx))
    prepare_plan = lambda aa: acb_mat.acb_mat_dense_matvec_plan_prepare(aa)
    cached_matvec_padded = lambda plan, xx: acb_mat.acb_mat_dense_matvec_plan_apply_batch_padded(plan, xx, pad_to=8)
    add = jax.jit(lambda aa, bb: acb_mat.acb_mat_add(aa, bb))
    entrywise = jax.jit(lambda aa, bb: acb_mat.acb_mat_mul_entrywise(aa, bb))
    charpoly = jax.jit(lambda aa: acb_mat.acb_mat_charpoly(aa))
    power2 = jax.jit(lambda aa: acb_mat.acb_mat_pow_ui(aa, 2))
    expm = jax.jit(lambda aa: acb_mat.acb_mat_exp(aa))
    hpd_solve = jax.jit(lambda aa, bb: acb_mat.acb_mat_hpd_solve(aa, bb))
    hpd_plan_solve = jax.jit(lambda plan, bb: acb_mat.acb_mat_dense_hpd_solve_plan_apply(plan, bb))
    hpd_eigh = jax.jit(lambda aa: acb_mat.acb_mat_eigh(aa))
    solve_tril = jax.jit(lambda aa, bb: acb_mat.acb_mat_solve_tril(aa, bb))
    solve_lu = jax.jit(lambda aa, bb: acb_mat.acb_mat_solve_lu(aa, bb))
    transpose = jax.jit(acb_mat.acb_mat_transpose)
    ctranspose = jax.jit(acb_mat.acb_mat_conjugate_transpose)
    diag = jax.jit(acb_mat.acb_mat_diag)
    block_assemble = lambda blocks: acb_mat.acb_mat_block_assemble(blocks)
    block_diag = lambda blocks: acb_mat.acb_mat_block_diag((blocks[0][0], blocks[1][1]))

    return {
        "acb_direct_solve_s": _time_call(direct_solve, a, rhs, warmup=warmup, runs=runs),
        "acb_lu_reuse_s": _time_call(lu_solve, lu_plan, rhs, warmup=warmup, runs=runs),
        "acb_lu_reuse_transpose_s": _time_call(lu_solve_transpose, lu_plan, rhs, warmup=warmup, runs=runs),
        "acb_lu_reuse_add_s": _time_call(lu_solve_add, lu_plan, rhs, rhs, warmup=warmup, runs=runs),
        "acb_dense_plan_prepare_s": _time_call(prepare_plan, a, warmup=warmup, runs=runs),
        "acb_cached_matvec_s": _time_call(cached_matvec, cache, vec, warmup=warmup, runs=runs),
        "acb_cached_matvec_padded_s": _time_call(cached_matvec_padded, cache, vec_batch, warmup=warmup, runs=runs),
        "acb_add_s": _time_call(add, a, a, warmup=warmup, runs=runs),
        "acb_mul_entrywise_s": _time_call(entrywise, a, a, warmup=warmup, runs=runs),
        "acb_charpoly_s": _time_call(charpoly, a, warmup=warmup, runs=runs),
        "acb_pow_ui_s": _time_call(power2, a, warmup=warmup, runs=runs),
        "acb_exp_s": _time_call(expm, hpd_a, warmup=warmup, runs=runs),
        "acb_hpd_solve_s": _time_call(hpd_solve, hpd_a, hpd_rhs, warmup=warmup, runs=runs),
        "acb_hpd_plan_solve_s": _time_call(hpd_plan_solve, hpd_plan, hpd_rhs, warmup=warmup, runs=runs),
        "acb_hpd_eigh_s": _time_call(hpd_eigh, hpd_a, warmup=warmup, runs=runs),
        "acb_solve_tril_s": _time_call(solve_tril, acb_mat.acb_mat_cho(hpd_a), hpd_rhs, warmup=warmup, runs=runs),
        "acb_solve_lu_s": _time_call(solve_lu, a, rhs, warmup=warmup, runs=runs),
        "acb_transpose_s": _time_call(transpose, a, warmup=warmup, runs=runs),
        "acb_conjugate_transpose_s": _time_call(ctranspose, a, warmup=warmup, runs=runs),
        "acb_diag_s": _time_call(diag, a, warmup=warmup, runs=runs),
        "acb_block_assemble_s": _time_call(block_assemble, acb_blocks, warmup=warmup, runs=runs),
        "acb_block_diag_s": _time_call(block_diag, acb_blocks, warmup=warmup, runs=runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dense arb_mat/acb_mat matrix surface in pure JAX.")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    print(f"platform: {platform.platform()}")
    print(f"jax_backend: {jax.default_backend()}")
    print(f"n: {args.n}")
    print(f"warmup: {args.warmup}")
    print(f"runs: {args.runs}")

    stats = {}
    stats.update(run_arb_case(args.n, args.warmup, args.runs))
    stats.update(run_acb_case(args.n, args.warmup, args.runs))
    for key, value in stats.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
