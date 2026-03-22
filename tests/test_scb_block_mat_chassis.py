import jax.numpy as jnp

from arbplusjax import scb_block_mat

from tests._test_checks import _check


def test_block_sparse_complex_roundtrip_and_kernels():
    dense = jnp.array(
        [
            [1.0 + 0.0j, 2.0 - 0.5j, 0.0 + 0.0j, 0.0 + 0.0j],
            [3.0 + 0.25j, 4.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [5.0 - 0.75j, 6.0 + 0.0j, 7.0 + 0.5j, 8.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 9.0 - 0.25j, 10.0 + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    vec = jnp.array([1.0 + 0.0j, -1.0 + 0.5j, 0.5 - 0.25j, 2.0 + 0.0j], dtype=jnp.complex128)
    rhs = jnp.stack([vec, vec + (1.0 - 0.25j)], axis=-1)
    coo = scb_block_mat.scb_block_mat_from_dense_coo(dense, block_shape=(2, 2))
    csr = scb_block_mat.scb_block_mat_coo_to_csr(coo)
    trans = scb_block_mat.scb_block_mat_transpose(csr)

    _check(scb_block_mat.scb_block_mat_shape(coo) == (4, 4))
    _check(scb_block_mat.scb_block_mat_block_shape(csr) == (2, 2))
    _check(scb_block_mat.scb_block_mat_nnzb(csr) == 3)
    _check(bool(jnp.allclose(scb_block_mat.scb_block_mat_to_dense(coo), dense)))
    _check(bool(jnp.allclose(scb_block_mat.scb_block_mat_to_dense(csr), dense)))
    _check(bool(jnp.allclose(scb_block_mat.scb_block_mat_to_dense(trans), dense.T)))
    _check(bool(jnp.allclose(scb_block_mat.scb_block_mat_matvec(csr, vec), dense @ vec)))
    _check(bool(jnp.allclose(scb_block_mat.scb_block_mat_matmul_dense_rhs(csr, rhs), dense @ rhs)))
    plan = scb_block_mat.scb_block_mat_matvec_cached_prepare(csr)
    _check(bool(jnp.allclose(scb_block_mat.scb_block_mat_matvec_cached_apply(plan, vec), dense @ vec)))
    rplan = scb_block_mat.scb_block_mat_rmatvec_cached_prepare(csr)
    aplan = scb_block_mat.scb_block_mat_adjoint_matvec_cached_prepare(csr)
    _check(bool(jnp.allclose(scb_block_mat.scb_block_mat_rmatvec(csr, vec), dense.T @ vec)))
    _check(bool(jnp.allclose(scb_block_mat.scb_block_mat_adjoint_matvec(csr, vec), jnp.conjugate(dense).T @ vec)))
    _check(bool(jnp.allclose(scb_block_mat.scb_block_mat_rmatvec_cached_apply(rplan, vec), dense.T @ vec)))
    _check(bool(jnp.allclose(scb_block_mat.scb_block_mat_adjoint_matvec_cached_apply(aplan, vec), jnp.conjugate(dense).T @ vec)))
    vs = jnp.stack([vec, vec + (1.0 - 0.25j)], axis=0)
    _check(bool(jnp.allclose(scb_block_mat.scb_block_mat_matvec_batch_fixed(csr, vs), vs @ dense.T)))
    _check(bool(jnp.allclose(scb_block_mat.scb_block_mat_matvec_cached_apply_batch_fixed(plan, vs), vs @ dense.T)))
    _check(bool(jnp.allclose(scb_block_mat.scb_block_mat_rmatvec_batch_fixed(csr, vs), vs @ dense)))
    _check(bool(jnp.allclose(scb_block_mat.scb_block_mat_rmatvec_cached_apply_batch_fixed(rplan, vs), vs @ dense)))
    _check(bool(jnp.allclose(scb_block_mat.scb_block_mat_adjoint_matvec_batch_fixed(csr, vs), vs @ jnp.conjugate(dense))))
    _check(bool(jnp.allclose(scb_block_mat.scb_block_mat_adjoint_matvec_cached_apply_batch_fixed(aplan, vs), vs @ jnp.conjugate(dense))))


def test_block_sparse_complex_triangular_and_iterative_solve():
    dense = jnp.array(
        [
            [2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 0.5j, 3.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.5 - 0.25j, -1.0 + 0.0j, 4.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.25 + 0.0j, 2.0 - 0.5j, 5.0 + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    rhs = jnp.array([2.0 + 0.0j, 7.0 - 0.5j, 8.0 + 0.25j, 11.0 + 0.0j], dtype=jnp.complex128)
    diag = jnp.diag(jnp.array([2.0 + 0.5j, 3.0 - 0.25j, 5.0 + 0.0j, 7.0 + 0.0j], dtype=jnp.complex128))
    tri = scb_block_mat.scb_block_mat_from_dense_csr(dense, block_shape=(2, 2))
    dmat = scb_block_mat.scb_block_mat_from_dense_csr(diag, block_shape=(2, 2))

    tri_sol = scb_block_mat.scb_block_mat_triangular_solve(tri, rhs, lower=True)
    gmres = scb_block_mat.scb_block_mat_solve(dmat, rhs, method="gmres", tol=1e-10, atol=1e-10, maxiter=20)
    lu = scb_block_mat.scb_block_mat_lu(dmat)
    lu_sol = scb_block_mat.scb_block_mat_lu_solve(lu, rhs)
    qr = scb_block_mat.scb_block_mat_qr(dmat)
    qr_sol = scb_block_mat.scb_block_mat_qr_solve(qr, rhs)

    _check(bool(jnp.allclose(tri_sol, jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(gmres, jnp.linalg.solve(diag, rhs), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(lu_sol, jnp.linalg.solve(diag, rhs), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(qr_sol, jnp.linalg.solve(diag, rhs), rtol=1e-8, atol=1e-8)))


def test_block_sparse_complex_diagnostics():
    dense = jnp.array(
        [
            [2.0 + 0.25j, 0.0, 0.0, 0.0],
            [1.0 - 0.1j, 3.0 + 0.0j, 0.0, 0.0],
            [0.0, 0.0, 4.0 + 0.5j, 0.0],
            [0.0, 0.0, 2.0 - 0.2j, 5.0 + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    vec = jnp.array([1.0 + 0.0j, 2.0 - 0.5j, 3.0 + 0.25j, 4.0 + 0.0j], dtype=jnp.complex128)
    csr = scb_block_mat.scb_block_mat_from_dense_csr(dense, block_shape=(2, 2))
    plan = scb_block_mat.scb_block_mat_matvec_cached_prepare(csr)

    mv, mv_diag = scb_block_mat.scb_block_mat_matvec_with_diagnostics(csr, vec)
    cached, cached_diag = scb_block_mat.scb_block_mat_matvec_cached_apply_with_diagnostics(plan, vec)
    sol, solve_diag = scb_block_mat.scb_block_mat_solve_with_diagnostics(csr, vec, method="gmres", tol=1e-10, atol=1e-10, maxiter=10)

    _check(bool(jnp.allclose(mv, dense @ vec)))
    _check(bool(jnp.allclose(cached, dense @ vec)))
    _check(bool(jnp.allclose(sol, jnp.linalg.solve(dense, vec), rtol=1e-8, atol=1e-8)))
    _check(mv_diag.block_rows == 2)
    _check(cached_diag.cached)
    _check(solve_diag.method == "gmres")
