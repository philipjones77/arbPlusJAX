import jax.numpy as jnp

from arbplusjax import scb_vblock_mat

from tests._test_checks import _check


def test_variable_block_complex_roundtrip_and_factors():
    dense = jnp.array(
        [
            [2.0 + 0.0j, 1.0 - 0.25j, 0.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 0.25j, 3.0 + 0.0j, 4.0 + 0.5j, 0.0 + 0.0j],
            [0.0 + 0.0j, 5.0 - 0.5j, 6.0 + 0.0j, 7.0 + 0.25j],
            [0.0 + 0.0j, 0.0 + 0.0j, 8.0 - 0.25j, 9.0 + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    row_sizes = jnp.array([1, 1, 2], dtype=jnp.int32)
    col_sizes = jnp.array([1, 1, 2], dtype=jnp.int32)
    rhs = jnp.array([1.0 + 0.0j, 2.0 - 0.5j, 3.0 + 0.25j, 4.0 + 0.0j], dtype=jnp.complex128)
    x = scb_vblock_mat.scb_vblock_mat_from_dense_csr(dense, row_block_sizes=row_sizes, col_block_sizes=col_sizes)
    lu = scb_vblock_mat.scb_vblock_mat_lu(x)
    qr = scb_vblock_mat.scb_vblock_mat_qr(x)

    _check(scb_vblock_mat.scb_vblock_mat_shape(x) == (4, 4))
    _check(bool(jnp.allclose(scb_vblock_mat.scb_vblock_mat_to_dense(x), dense)))
    _check(bool(jnp.allclose(scb_vblock_mat.scb_vblock_mat_matvec(x, rhs), dense @ rhs)))
    _check(bool(jnp.allclose(scb_vblock_mat.scb_vblock_mat_lu_solve(lu, rhs), jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))
    q, r = qr
    _check(bool(jnp.allclose(q @ scb_vblock_mat.scb_vblock_mat_to_dense(r), dense, rtol=1e-8, atol=1e-8)))


def test_variable_block_complex_cached_batch_and_diagnostics():
    dense = jnp.array(
        [
            [2.0 + 0.0j, 1.0 - 0.25j, 0.0, 0.0],
            [1.0 + 0.25j, 3.0 + 0.0j, 4.0 - 0.5j, 0.0],
            [0.0, 5.0 + 0.25j, 6.0 + 0.0j, 7.0 - 0.25j],
            [0.0, 0.0, 8.0 + 0.5j, 9.0 + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    row_sizes = jnp.array([1, 1, 2], dtype=jnp.int32)
    col_sizes = jnp.array([1, 1, 2], dtype=jnp.int32)
    vec = jnp.array([1.0 + 0.0j, 2.0 - 0.25j, 3.0 + 0.5j, 4.0 + 0.0j], dtype=jnp.complex128)
    x = scb_vblock_mat.scb_vblock_mat_from_dense_csr(dense, row_block_sizes=row_sizes, col_block_sizes=col_sizes)
    plan = scb_vblock_mat.scb_vblock_mat_matvec_cached_prepare(x)
    rplan = scb_vblock_mat.scb_vblock_mat_rmatvec_cached_prepare(x)
    aplan = scb_vblock_mat.scb_vblock_mat_adjoint_matvec_cached_prepare(x)
    xs = jnp.stack([vec, vec + (1.0 - 0.5j)], axis=0)

    cached = scb_vblock_mat.scb_vblock_mat_matvec_cached_apply(plan, vec)
    rcached = scb_vblock_mat.scb_vblock_mat_rmatvec_cached_apply(rplan, vec)
    acached = scb_vblock_mat.scb_vblock_mat_adjoint_matvec_cached_apply(aplan, vec)
    batch = scb_vblock_mat.scb_vblock_mat_matvec_batch_fixed(x, xs)
    cached_batch = scb_vblock_mat.scb_vblock_mat_matvec_cached_apply_batch_fixed(plan, xs)
    rbatch = scb_vblock_mat.scb_vblock_mat_rmatvec_batch_fixed(x, xs)
    rcached_batch = scb_vblock_mat.scb_vblock_mat_rmatvec_cached_apply_batch_fixed(rplan, xs)
    abatch = scb_vblock_mat.scb_vblock_mat_adjoint_matvec_batch_fixed(x, xs)
    acached_batch = scb_vblock_mat.scb_vblock_mat_adjoint_matvec_cached_apply_batch_fixed(aplan, xs)
    mv, mv_diag = scb_vblock_mat.scb_vblock_mat_matvec_with_diagnostics(x, vec)
    _, cached_diag = scb_vblock_mat.scb_vblock_mat_matvec_cached_apply_with_diagnostics(plan, vec)
    sol, solve_diag = scb_vblock_mat.scb_vblock_mat_solve_with_diagnostics(x, vec, method="lu")
    (_, _, _), lu_diag = scb_vblock_mat.scb_vblock_mat_lu_with_diagnostics(x)
    (_, _), qr_diag = scb_vblock_mat.scb_vblock_mat_qr_with_diagnostics(x)

    _check(bool(jnp.allclose(cached, dense @ vec)))
    _check(bool(jnp.allclose(rcached, dense.T @ vec)))
    _check(bool(jnp.allclose(acached, jnp.conjugate(dense).T @ vec)))
    _check(bool(jnp.allclose(batch, xs @ dense.T)))
    _check(bool(jnp.allclose(cached_batch, xs @ dense.T)))
    _check(bool(jnp.allclose(rbatch, xs @ dense)))
    _check(bool(jnp.allclose(rcached_batch, xs @ dense)))
    _check(bool(jnp.allclose(abatch, xs @ jnp.conjugate(dense))))
    _check(bool(jnp.allclose(acached_batch, xs @ jnp.conjugate(dense))))
    _check(bool(jnp.allclose(mv, dense @ vec)))
    _check(bool(jnp.allclose(sol, jnp.linalg.solve(dense, vec), rtol=1e-8, atol=1e-8)))
    _check(mv_diag.row_blocks == 3)
    _check(cached_diag.cached)
    _check(solve_diag.method == "lu")
    _check(lu_diag.direct)
    _check(qr_diag.direct)


def test_variable_block_complex_basic_det_inv_and_square():
    dense = jnp.array(
        [
            [4.0 + 0.0j, 1.0 - 0.25j, 0.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 0.25j, 5.0 + 0.0j, 0.5 - 0.1j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.5 + 0.1j, 3.5 + 0.0j, 1.0 - 0.2j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.2j, 2.5 + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    row_sizes = jnp.array([1, 3], dtype=jnp.int32)
    col_sizes = jnp.array([2, 2], dtype=jnp.int32)
    csr = scb_vblock_mat.scb_vblock_mat_from_dense_csr(dense, row_block_sizes=row_sizes, col_block_sizes=col_sizes)

    det_basic = scb_vblock_mat.scb_vblock_mat_det_basic(csr)
    inv_basic = scb_vblock_mat.scb_vblock_mat_inv_basic(csr)
    sqr_basic = scb_vblock_mat.scb_vblock_mat_sqr_basic(csr)

    _check(bool(jnp.allclose(det_basic, jnp.linalg.det(dense), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(inv_basic, jnp.linalg.inv(dense), rtol=1e-6, atol=1e-6)))
    _check(bool(jnp.allclose(scb_vblock_mat.scb_vblock_mat_to_dense(sqr_basic), dense @ dense, rtol=1e-8, atol=1e-8)))
