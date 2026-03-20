import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import scb_mat

from tests._test_checks import _check


def test_coo_csr_bcoo_dense_roundtrip_and_conjugate_transpose():
    dense = jnp.array(
        [
            [3.0 + 0.0j, 0.0 + 0.0j, 1.5 - 0.5j],
            [0.0 + 0.0j, -2.0 + 1.0j, 0.0 + 0.0j],
            [4.0 + 0.25j, 0.0 + 0.0j, 5.0 - 2.0j],
        ],
        dtype=jnp.complex128,
    )
    coo = scb_mat.scb_mat_from_dense_coo(dense)
    csr = scb_mat.scb_mat_from_dense_csr(dense)
    bcoo = scb_mat.scb_mat_from_dense_bcoo(dense)

    _check(bool(jnp.allclose(scb_mat.scb_mat_coo_to_dense(coo), dense)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_csr_to_dense(csr), dense)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_bcoo_to_dense(bcoo), dense)))

    ct = scb_mat.scb_mat_conjugate_transpose(csr)
    _check(bool(jnp.allclose(scb_mat.scb_mat_to_dense(ct), jnp.conj(dense).T)))
    _check(scb_mat.scb_mat_shape(coo) == (3, 3))
    _check(scb_mat.scb_mat_nnz(coo) == 5)


def test_matvec_cached_matvec_and_batch_helpers():
    dense = jnp.array(
        [
            [3.0 + 0.0j, 0.0 + 0.0j, 1.5 - 0.5j],
            [0.0 + 0.0j, -2.0 + 1.0j, 0.0 + 0.0j],
            [4.0 + 0.25j, 0.0 + 0.0j, 5.0 - 2.0j],
        ],
        dtype=jnp.complex128,
    )
    x = jnp.array([1.0 + 0.0j, -2.0 + 0.5j, 0.5 - 1.0j], dtype=jnp.complex128)
    xs = jnp.stack([x, x + (1.0 - 0.25j)], axis=0)
    csr = scb_mat.scb_mat_from_dense_csr(dense)
    plan = scb_mat.scb_mat_matvec_cached_prepare(csr)
    expected = dense @ x

    _check(bool(jnp.allclose(scb_mat.scb_mat_matvec(csr, x), expected)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_matvec_cached_apply(plan, x), expected)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_matvec_jit(csr, x), expected)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_matvec_cached_apply_jit(plan, x), expected)))
    rvec0 = jnp.array([1.0 + 0.5j, -1.0 + 0.25j, 0.5 - 0.75j], dtype=jnp.complex128)
    _check(bool(jnp.allclose(scb_mat.scb_mat_rmatvec(csr, rvec0), dense.T @ rvec0)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(scb_mat.scb_mat_rmatvec_basic(csr, rvec0)), dense.T @ rvec0)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_matvec_batch_fixed(csr, xs), xs @ dense.T)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_matvec_cached_apply_batch_fixed(plan, xs), xs @ dense.T)))
    _check(scb_mat.scb_mat_matvec_batch_padded(csr, xs, pad_to=4).shape == (4, 3))
    rvec = jnp.array([2.0 - 0.25j, -1.0 + 0.75j, 0.25 + 0.5j], dtype=jnp.complex128)
    rplan = scb_mat.scb_mat_rmatvec_cached_prepare(csr)
    rbasic_plan = scb_mat.scb_mat_rmatvec_cached_prepare_basic(csr)
    _check(bool(jnp.allclose(scb_mat.scb_mat_rmatvec_cached_apply(rplan, rvec), dense.T @ rvec)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_rmatvec_cached_apply_jit(rplan, rvec), dense.T @ rvec)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(scb_mat.scb_mat_rmatvec_cached_apply_basic(rbasic_plan, rvec)), dense.T @ rvec)))


def test_dense_rhs_multiply_and_format_conversions():
    dense = jnp.array(
        [
            [3.0 + 0.0j, 0.0 + 0.0j, 1.5 - 0.5j],
            [0.0 + 0.0j, -2.0 + 1.0j, 0.0 + 0.0j],
            [4.0 + 0.25j, 0.0 + 0.0j, 5.0 - 2.0j],
        ],
        dtype=jnp.complex128,
    )
    rhs = jnp.array(
        [
            [1.0 + 0.0j, 0.0 + 1.0j],
            [-2.0 + 1.0j, 1.0 + 0.0j],
            [0.5 - 1.0j, 3.0 + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    coo = scb_mat.scb_mat_from_dense_coo(dense)
    csr = scb_mat.scb_mat_coo_to_csr(coo)
    bcoo = scb_mat.scb_mat_coo_to_bcoo(coo)

    _check(bool(jnp.allclose(scb_mat.scb_mat_matmul_dense_rhs(csr, rhs), dense @ rhs)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_matmul_dense_rhs_jit(bcoo, rhs), dense @ rhs)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_to_dense(scb_mat.scb_mat_bcoo_to_coo(bcoo)), dense)))


def test_sparse_add_sub_scale_and_sparse_matmul():
    a = jnp.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j, 2.0 - 1.0j],
            [0.0 + 0.0j, 3.0 + 2.0j, 0.0 + 0.0j],
            [4.0 - 0.5j, 0.0 + 0.0j, 5.0 + 0.25j],
        ],
        dtype=jnp.complex128,
    )
    b = jnp.array(
        [
            [0.0 + 0.0j, 1.0 + 0.5j, 0.0 + 0.0j],
            [2.0 - 1.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 3.0 - 0.25j],
        ],
        dtype=jnp.complex128,
    )
    sa = scb_mat.scb_mat_from_dense_csr(a)
    sb = scb_mat.scb_mat_from_dense_bcoo(b)

    _check(bool(jnp.allclose(scb_mat.scb_mat_to_dense(scb_mat.scb_mat_scale(sa, 2.0 - 1.0j)), (2.0 - 1.0j) * a)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_to_dense(scb_mat.scb_mat_add(sa, sb)), a + b)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_to_dense(scb_mat.scb_mat_sub(sa, sb)), a - b)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_to_dense(scb_mat.scb_mat_matmul_sparse(sa, sb)), a @ b)))


def test_sparse_triangular_solve_and_iterative_solve():
    lower = jnp.array(
        [
            [2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 1.0j, 3.0 + 0.0j, 0.0 + 0.0j],
            [-1.0 + 0.5j, 2.0 - 1.0j, 4.0 + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    b = jnp.array([4.0 + 0.0j, 10.0 + 1.0j, 9.0 - 1.0j], dtype=jnp.complex128)
    rhs = jnp.array([2.0 + 1.0j, 6.0 - 1.0j, 12.0 + 0.5j], dtype=jnp.complex128)
    diag = jnp.diag(jnp.array([2.0 + 0.5j, 3.0 - 0.5j, 5.0 + 0.0j], dtype=jnp.complex128))

    sl = scb_mat.scb_mat_from_dense_csr(lower)
    sd = scb_mat.scb_mat_from_dense_bcoo(diag)

    tri = scb_mat.scb_mat_triangular_solve(sl, b, lower=True)
    bicgstab = scb_mat.scb_mat_solve(sd, rhs, method="bicgstab", tol=1e-10, atol=1e-10, maxiter=30)
    gmres = scb_mat.scb_mat_solve(sd, rhs, method="gmres", tol=1e-10, atol=1e-10, maxiter=30)

    _check(bool(jnp.allclose(tri, jnp.linalg.solve(lower, b), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(bicgstab, jnp.linalg.solve(diag, rhs), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(gmres, jnp.linalg.solve(diag, rhs), rtol=1e-8, atol=1e-8)))


def test_sparse_lu_qr_trace_and_norms():
    dense = jnp.array(
        [
            [0.0 + 0.0j, 1.0 + 0.5j, 2.0 - 0.25j],
            [2.0 - 0.5j, 3.0 + 0.0j, 1.0 + 0.25j],
            [4.0 + 0.0j, 1.0 - 0.25j, 0.0 + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    rhs = jnp.array([1.0 + 0.0j, 2.0 - 0.5j, 3.0 + 0.25j], dtype=jnp.complex128)
    sparse = scb_mat.scb_mat_from_dense_csr(dense)
    p, l, u = scb_mat.scb_mat_lu(sparse)
    qr = scb_mat.scb_mat_qr(sparse)
    q = scb_mat.scb_mat_qr_explicit_q(qr)
    r = scb_mat.scb_mat_to_dense(scb_mat.scb_mat_qr_r(qr))
    sol = scb_mat.scb_mat_lu_solve((p, l, u), rhs)
    qr_sol = scb_mat.scb_mat_qr_solve(qr, rhs)

    _check(bool(jnp.allclose(scb_mat.scb_mat_to_dense(p) @ dense, scb_mat.scb_mat_to_dense(l) @ scb_mat.scb_mat_to_dense(u), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(sol, jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(q @ r, dense, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(qr_sol, jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_trace(sparse), jnp.trace(dense))))
    _check(bool(jnp.allclose(scb_mat.scb_mat_norm_fro(sparse), jnp.linalg.norm(dense, ord="fro"))))
    _check(bool(jnp.allclose(scb_mat.scb_mat_norm_1(sparse), jnp.linalg.norm(dense, ord=1))))
    _check(bool(jnp.allclose(scb_mat.scb_mat_norm_inf(sparse), jnp.linalg.norm(dense, ord=jnp.inf))))


def test_sparse_point_diagnostics():
    dense = jnp.array([[2.0 + 0.5j, 0.0], [1.0 - 0.25j, 3.0 + 0.0j]], dtype=jnp.complex128)
    rhs = jnp.array([2.0 + 1.0j, 7.0 - 0.5j], dtype=jnp.complex128)
    sparse = scb_mat.scb_mat_from_dense_csr(dense)
    plan = scb_mat.scb_mat_matvec_cached_prepare(sparse)

    mv, mv_diag = scb_mat.scb_mat_matvec_with_diagnostics(sparse, rhs)
    cached, cached_diag = scb_mat.scb_mat_matvec_cached_apply_with_diagnostics(plan, rhs)
    sol, solve_diag = scb_mat.scb_mat_solve_with_diagnostics(sparse, rhs, method="gmres", tol=1e-10, atol=1e-10, maxiter=10)
    (_, _, _), lu_diag = scb_mat.scb_mat_lu_with_diagnostics(sparse)
    _, qr_diag = scb_mat.scb_mat_qr_with_diagnostics(sparse)

    _check(bool(jnp.allclose(mv, dense @ rhs)))
    _check(bool(jnp.allclose(cached, dense @ rhs)))
    _check(bool(jnp.allclose(sol, jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))
    _check(mv_diag.storage == "csr")
    _check(cached_diag.cached)
    _check(solve_diag.method == "gmres")
    _check(lu_diag.direct)
    _check(qr_diag.direct)


def test_sparse_structural_helpers():
    d = jnp.array([1.0 + 0.5j, 2.0 - 1.0j, 3.0 + 0.0j], dtype=jnp.complex128)
    dm = scb_mat.scb_mat_diag_matrix(d)
    eye = scb_mat.scb_mat_identity(3)
    perm = scb_mat.scb_mat_permutation_matrix(jnp.array([2, 0, 1], dtype=jnp.int32))
    sub = scb_mat.scb_mat_submatrix(dm, 1, 3, 1, 3)

    _check(bool(jnp.allclose(scb_mat.scb_mat_to_dense(dm), jnp.diag(d))))
    _check(bool(jnp.allclose(scb_mat.scb_mat_diag(dm), d)))
    _check(bool(jnp.allclose(scb_mat.scb_mat_to_dense(eye), jnp.eye(3, dtype=jnp.complex128))))
    _check(bool(jnp.allclose(scb_mat.scb_mat_to_dense(perm), jnp.eye(3, dtype=jnp.complex128)[jnp.array([2, 0, 1])])))
    _check(bool(jnp.allclose(scb_mat.scb_mat_to_dense(sub), jnp.diag(jnp.array([2.0 - 1.0j, 3.0 + 0.0j], dtype=jnp.complex128)))))
