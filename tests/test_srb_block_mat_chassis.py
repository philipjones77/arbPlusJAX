import jax.numpy as jnp

from arbplusjax import srb_block_mat

from tests._test_checks import _check


def test_block_sparse_real_roundtrip_and_kernels():
    dense = jnp.array(
        [
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [5.0, 6.0, 7.0, 8.0],
            [0.0, 0.0, 9.0, 10.0],
        ],
        dtype=jnp.float64,
    )
    vec = jnp.array([1.0, -1.0, 0.5, 2.0], dtype=jnp.float64)
    rhs = jnp.stack([vec, vec + 1.0], axis=-1)
    coo = srb_block_mat.srb_block_mat_from_dense_coo(dense, block_shape=(2, 2))
    csr = srb_block_mat.srb_block_mat_coo_to_csr(coo)
    trans = srb_block_mat.srb_block_mat_transpose(csr)

    _check(srb_block_mat.srb_block_mat_shape(coo) == (4, 4))
    _check(srb_block_mat.srb_block_mat_block_shape(csr) == (2, 2))
    _check(srb_block_mat.srb_block_mat_nnzb(csr) == 3)
    _check(bool(jnp.allclose(srb_block_mat.srb_block_mat_to_dense(coo), dense)))
    _check(bool(jnp.allclose(srb_block_mat.srb_block_mat_to_dense(csr), dense)))
    _check(bool(jnp.allclose(srb_block_mat.srb_block_mat_to_dense(trans), dense.T)))
    _check(bool(jnp.allclose(srb_block_mat.srb_block_mat_matvec(csr, vec), dense @ vec)))
    _check(bool(jnp.allclose(srb_block_mat.srb_block_mat_matmul_dense_rhs(csr, rhs), dense @ rhs)))
    plan = srb_block_mat.srb_block_mat_matvec_cached_prepare(csr)
    _check(bool(jnp.allclose(srb_block_mat.srb_block_mat_matvec_cached_apply(plan, vec), dense @ vec)))
    vs = jnp.stack([vec, vec + 1.0], axis=0)
    _check(bool(jnp.allclose(srb_block_mat.srb_block_mat_matvec_batch_fixed(csr, vs), vs @ dense.T)))
    _check(bool(jnp.allclose(srb_block_mat.srb_block_mat_matvec_cached_apply_batch_fixed(plan, vs), vs @ dense.T)))


def test_block_sparse_real_triangular_and_iterative_solve():
    dense = jnp.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [1.0, 3.0, 0.0, 0.0],
            [0.5, -1.0, 4.0, 0.0],
            [0.0, 0.25, 2.0, 5.0],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.array([2.0, 7.0, 8.0, 11.0], dtype=jnp.float64)
    diag = jnp.diag(jnp.array([2.0, 3.0, 5.0, 7.0], dtype=jnp.float64))
    tri = srb_block_mat.srb_block_mat_from_dense_csr(dense, block_shape=(2, 2))
    dmat = srb_block_mat.srb_block_mat_from_dense_csr(diag, block_shape=(2, 2))

    tri_sol = srb_block_mat.srb_block_mat_triangular_solve(tri, rhs, lower=True)
    gmres = srb_block_mat.srb_block_mat_solve(dmat, rhs, method="gmres", tol=1e-10, atol=1e-10, maxiter=20)

    _check(bool(jnp.allclose(tri_sol, jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(gmres, jnp.linalg.solve(diag, rhs), rtol=1e-8, atol=1e-8)))


def test_block_sparse_real_diagnostics():
    dense = jnp.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [1.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 2.0, 5.0],
        ],
        dtype=jnp.float64,
    )
    vec = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    csr = srb_block_mat.srb_block_mat_from_dense_csr(dense, block_shape=(2, 2))
    plan = srb_block_mat.srb_block_mat_matvec_cached_prepare(csr)

    mv, mv_diag = srb_block_mat.srb_block_mat_matvec_with_diagnostics(csr, vec)
    cached, cached_diag = srb_block_mat.srb_block_mat_matvec_cached_apply_with_diagnostics(plan, vec)
    sol, solve_diag = srb_block_mat.srb_block_mat_solve_with_diagnostics(csr, vec, method="gmres", tol=1e-10, atol=1e-10, maxiter=10)

    _check(bool(jnp.allclose(mv, dense @ vec)))
    _check(bool(jnp.allclose(cached, dense @ vec)))
    _check(bool(jnp.allclose(sol, jnp.linalg.solve(dense, vec), rtol=1e-8, atol=1e-8)))
    _check(mv_diag.block_rows == 2)
    _check(cached_diag.cached)
    _check(solve_diag.method == "gmres")
