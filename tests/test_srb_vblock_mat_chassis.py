import jax.numpy as jnp

from arbplusjax import srb_vblock_mat

from tests._test_checks import _check


def test_variable_block_real_roundtrip_and_factors():
    dense = jnp.array(
        [
            [2.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 4.0, 0.0],
            [0.0, 5.0, 6.0, 7.0],
            [0.0, 0.0, 8.0, 9.0],
        ],
        dtype=jnp.float64,
    )
    row_sizes = jnp.array([1, 1, 2], dtype=jnp.int32)
    col_sizes = jnp.array([1, 1, 2], dtype=jnp.int32)
    rhs = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    x = srb_vblock_mat.srb_vblock_mat_from_dense_csr(dense, row_block_sizes=row_sizes, col_block_sizes=col_sizes)
    lu = srb_vblock_mat.srb_vblock_mat_lu(x)
    qr = srb_vblock_mat.srb_vblock_mat_qr(x)

    _check(srb_vblock_mat.srb_vblock_mat_shape(x) == (4, 4))
    _check(bool(jnp.allclose(srb_vblock_mat.srb_vblock_mat_to_dense(x), dense)))
    _check(bool(jnp.allclose(srb_vblock_mat.srb_vblock_mat_matvec(x, rhs), dense @ rhs)))
    _check(bool(jnp.allclose(srb_vblock_mat.srb_vblock_mat_lu_solve(lu, rhs), jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))
    q, r = qr
    _check(bool(jnp.allclose(q @ srb_vblock_mat.srb_vblock_mat_to_dense(r), dense, rtol=1e-8, atol=1e-8)))


def test_variable_block_real_cached_batch_and_diagnostics():
    dense = jnp.array(
        [
            [2.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 4.0, 0.0],
            [0.0, 5.0, 6.0, 7.0],
            [0.0, 0.0, 8.0, 9.0],
        ],
        dtype=jnp.float64,
    )
    row_sizes = jnp.array([1, 1, 2], dtype=jnp.int32)
    col_sizes = jnp.array([1, 1, 2], dtype=jnp.int32)
    vec = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    x = srb_vblock_mat.srb_vblock_mat_from_dense_csr(dense, row_block_sizes=row_sizes, col_block_sizes=col_sizes)
    plan = srb_vblock_mat.srb_vblock_mat_matvec_cached_prepare(x)
    rplan = srb_vblock_mat.srb_vblock_mat_rmatvec_cached_prepare(x)
    xs = jnp.stack([vec, vec + 1.0], axis=0)

    cached = srb_vblock_mat.srb_vblock_mat_matvec_cached_apply(plan, vec)
    rcached = srb_vblock_mat.srb_vblock_mat_rmatvec_cached_apply(rplan, vec)
    batch = srb_vblock_mat.srb_vblock_mat_matvec_batch_fixed(x, xs)
    cached_batch = srb_vblock_mat.srb_vblock_mat_matvec_cached_apply_batch_fixed(plan, xs)
    rbatch = srb_vblock_mat.srb_vblock_mat_rmatvec_batch_fixed(x, xs)
    rcached_batch = srb_vblock_mat.srb_vblock_mat_rmatvec_cached_apply_batch_fixed(rplan, xs)
    mv, mv_diag = srb_vblock_mat.srb_vblock_mat_matvec_with_diagnostics(x, vec)
    _, cached_diag = srb_vblock_mat.srb_vblock_mat_matvec_cached_apply_with_diagnostics(plan, vec)
    sol, solve_diag = srb_vblock_mat.srb_vblock_mat_solve_with_diagnostics(x, vec, method="lu")
    (_, _, _), lu_diag = srb_vblock_mat.srb_vblock_mat_lu_with_diagnostics(x)
    (_, _), qr_diag = srb_vblock_mat.srb_vblock_mat_qr_with_diagnostics(x)

    _check(bool(jnp.allclose(cached, dense @ vec)))
    _check(bool(jnp.allclose(rcached, dense.T @ vec)))
    _check(bool(jnp.allclose(batch, xs @ dense.T)))
    _check(bool(jnp.allclose(cached_batch, xs @ dense.T)))
    _check(bool(jnp.allclose(rbatch, xs @ dense)))
    _check(bool(jnp.allclose(rcached_batch, xs @ dense)))
    _check(bool(jnp.allclose(mv, dense @ vec)))
    _check(bool(jnp.allclose(sol, jnp.linalg.solve(dense, vec), rtol=1e-8, atol=1e-8)))
    _check(mv_diag.row_blocks == 3)
    _check(cached_diag.cached)
    _check(solve_diag.method == "lu")
    _check(lu_diag.direct)
    _check(qr_diag.direct)
