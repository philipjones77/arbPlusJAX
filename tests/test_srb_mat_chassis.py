import jax.numpy as jnp

from arbplusjax import srb_mat

from tests._test_checks import _check


def test_coo_csr_bcoo_dense_roundtrip_and_transpose():
    dense = jnp.array(
        [
            [3.0, 0.0, 1.5],
            [0.0, -2.0, 0.0],
            [4.0, 0.0, 5.0],
        ],
        dtype=jnp.float64,
    )
    coo = srb_mat.srb_mat_from_dense_coo(dense)
    csr = srb_mat.srb_mat_from_dense_csr(dense)
    bcoo = srb_mat.srb_mat_from_dense_bcoo(dense)

    _check(bool(jnp.allclose(srb_mat.srb_mat_coo_to_dense(coo), dense)))
    _check(bool(jnp.allclose(srb_mat.srb_mat_csr_to_dense(csr), dense)))
    _check(bool(jnp.allclose(srb_mat.srb_mat_bcoo_to_dense(bcoo), dense)))

    t = srb_mat.srb_mat_transpose(csr)
    _check(bool(jnp.allclose(srb_mat.srb_mat_to_dense(t), dense.T)))

    _check(srb_mat.srb_mat_shape(coo) == (3, 3))
    _check(srb_mat.srb_mat_nnz(coo) == 5)


def test_matvec_cached_matvec_and_batch_helpers():
    dense = jnp.array(
        [
            [3.0, 0.0, 1.5],
            [0.0, -2.0, 0.0],
            [4.0, 0.0, 5.0],
        ],
        dtype=jnp.float64,
    )
    x = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64)
    xs = jnp.stack([x, x + 1.0], axis=0)
    coo = srb_mat.srb_mat_from_dense_coo(dense)
    csr = srb_mat.srb_mat_from_dense_csr(dense)
    bcoo = srb_mat.srb_mat_from_dense_bcoo(dense)
    plan = srb_mat.srb_mat_matvec_cached_prepare(csr)

    expected = dense @ x

    _check(bool(jnp.allclose(srb_mat.srb_mat_matvec(coo, x), expected)))
    _check(bool(jnp.allclose(srb_mat.srb_mat_matvec(csr, x), expected)))
    _check(bool(jnp.allclose(srb_mat.srb_mat_matvec(bcoo, x), expected)))
    _check(bool(jnp.allclose(srb_mat.srb_mat_matvec_cached_apply(plan, x), expected)))
    _check(bool(jnp.allclose(srb_mat.srb_mat_matvec_jit(csr, x), expected)))
    _check(bool(jnp.allclose(srb_mat.srb_mat_matvec_cached_apply_jit(plan, x), expected)))
    _check(bool(jnp.allclose(srb_mat.srb_mat_matvec_batch_fixed(csr, xs), xs @ dense.T)))
    _check(bool(jnp.allclose(srb_mat.srb_mat_matvec_cached_apply_batch_fixed(plan, xs), xs @ dense.T)))
    _check(srb_mat.srb_mat_matvec_batch_padded(csr, xs, pad_to=4).shape == (4, 3))


def test_dense_rhs_multiply_and_format_conversions():
    dense = jnp.array(
        [
            [3.0, 0.0, 1.5],
            [0.0, -2.0, 0.0],
            [4.0, 0.0, 5.0],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.array(
        [
            [1.0, 0.0],
            [-2.0, 1.0],
            [0.5, 3.0],
        ],
        dtype=jnp.float64,
    )
    coo = srb_mat.srb_mat_from_dense_coo(dense)
    csr = srb_mat.srb_mat_coo_to_csr(coo)
    bcoo = srb_mat.srb_mat_coo_to_bcoo(coo)

    _check(bool(jnp.allclose(srb_mat.srb_mat_matmul_dense_rhs(csr, rhs), dense @ rhs)))
    _check(bool(jnp.allclose(srb_mat.srb_mat_matmul_dense_rhs_jit(bcoo, rhs), dense @ rhs)))
    _check(bool(jnp.allclose(srb_mat.srb_mat_to_dense(srb_mat.srb_mat_bcoo_to_coo(bcoo)), dense)))


def test_sparse_add_sub_scale_and_sparse_matmul():
    a = jnp.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0],
        ],
        dtype=jnp.float64,
    )
    b = jnp.array(
        [
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=jnp.float64,
    )
    sa = srb_mat.srb_mat_from_dense_csr(a)
    sb = srb_mat.srb_mat_from_dense_bcoo(b)

    _check(bool(jnp.allclose(srb_mat.srb_mat_to_dense(srb_mat.srb_mat_scale(sa, 2.0)), 2.0 * a)))
    _check(bool(jnp.allclose(srb_mat.srb_mat_to_dense(srb_mat.srb_mat_add(sa, sb)), a + b)))
    _check(bool(jnp.allclose(srb_mat.srb_mat_to_dense(srb_mat.srb_mat_sub(sa, sb)), a - b)))
    _check(bool(jnp.allclose(srb_mat.srb_mat_to_dense(srb_mat.srb_mat_matmul_sparse(sa, sb)), a @ b)))


def test_sparse_triangular_solve_and_iterative_solve():
    lower = jnp.array(
        [
            [2.0, 0.0, 0.0],
            [1.0, 3.0, 0.0],
            [-1.0, 2.0, 4.0],
        ],
        dtype=jnp.float64,
    )
    b = jnp.array([4.0, 10.0, 9.0], dtype=jnp.float64)
    rhs = jnp.array([2.0, 6.0, 12.0], dtype=jnp.float64)
    diag = jnp.diag(jnp.array([2.0, 3.0, 5.0], dtype=jnp.float64))

    sl = srb_mat.srb_mat_from_dense_csr(lower)
    sd = srb_mat.srb_mat_from_dense_bcoo(diag)

    tri = srb_mat.srb_mat_triangular_solve(sl, b, lower=True)
    cg = srb_mat.srb_mat_solve(sd, rhs, method="cg", tol=1e-10, atol=1e-10, maxiter=20)
    gmres = srb_mat.srb_mat_solve(sd, rhs, method="gmres", tol=1e-10, atol=1e-10, maxiter=20)

    _check(bool(jnp.allclose(tri, jnp.linalg.solve(lower, b), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(cg, jnp.linalg.solve(diag, rhs), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(gmres, jnp.linalg.solve(diag, rhs), rtol=1e-8, atol=1e-8)))


def test_sparse_cg_accepts_callable_preconditioner():
    dense = jnp.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 0.5],
            [0.0, 0.5, 2.0],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    sparse = srb_mat.srb_mat_from_dense_bcoo(dense)
    diag_inv = 1.0 / jnp.diag(dense)
    jacobi = lambda v: diag_inv * v

    sol = srb_mat.srb_mat_solve(sparse, rhs, method="cg", tol=1e-10, atol=1e-10, maxiter=20, M=jacobi)
    _check(bool(jnp.allclose(sol, jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))


def test_sparse_lu_qr_trace_and_norms():
    dense = jnp.array(
        [
            [0.0, 1.0, 2.0],
            [2.0, 3.0, 1.0],
            [4.0, 1.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    sparse = srb_mat.srb_mat_from_dense_csr(dense)
    p, l, u = srb_mat.srb_mat_lu(sparse)
    qr = srb_mat.srb_mat_qr(sparse)
    q = srb_mat.srb_mat_qr_explicit_q(qr)
    r = srb_mat.srb_mat_to_dense(srb_mat.srb_mat_qr_r(qr))
    sol = srb_mat.srb_mat_lu_solve((p, l, u), rhs)
    qr_sol = srb_mat.srb_mat_qr_solve(qr, rhs)

    _check(bool(jnp.allclose(srb_mat.srb_mat_to_dense(p) @ dense, srb_mat.srb_mat_to_dense(l) @ srb_mat.srb_mat_to_dense(u), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(sol, jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(q @ r, dense, rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(qr_sol, jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))
    _check(bool(jnp.allclose(srb_mat.srb_mat_trace(sparse), jnp.trace(dense))))
    _check(bool(jnp.allclose(srb_mat.srb_mat_norm_fro(sparse), jnp.linalg.norm(dense, ord="fro"))))
    _check(bool(jnp.allclose(srb_mat.srb_mat_norm_1(sparse), jnp.linalg.norm(dense, ord=1))))
    _check(bool(jnp.allclose(srb_mat.srb_mat_norm_inf(sparse), jnp.linalg.norm(dense, ord=jnp.inf))))


def test_sparse_point_diagnostics():
    dense = jnp.array([[2.0, 0.0], [1.0, 3.0]], dtype=jnp.float64)
    rhs = jnp.array([2.0, 7.0], dtype=jnp.float64)
    sparse = srb_mat.srb_mat_from_dense_csr(dense)
    plan = srb_mat.srb_mat_matvec_cached_prepare(sparse)

    mv, mv_diag = srb_mat.srb_mat_matvec_with_diagnostics(sparse, rhs)
    cached, cached_diag = srb_mat.srb_mat_matvec_cached_apply_with_diagnostics(plan, rhs)
    sol, solve_diag = srb_mat.srb_mat_solve_with_diagnostics(sparse, rhs, method="gmres", tol=1e-10, atol=1e-10, maxiter=10)
    (_, _, _), lu_diag = srb_mat.srb_mat_lu_with_diagnostics(sparse)
    _, qr_diag = srb_mat.srb_mat_qr_with_diagnostics(sparse)

    _check(bool(jnp.allclose(mv, dense @ rhs)))
    _check(bool(jnp.allclose(cached, dense @ rhs)))
    _check(bool(jnp.allclose(sol, jnp.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-8)))
    _check(mv_diag.storage == "csr")
    _check(cached_diag.cached)
    _check(solve_diag.method == "gmres")
    _check(lu_diag.direct)
    _check(qr_diag.direct)


def test_sparse_structural_helpers():
    d = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    dm = srb_mat.srb_mat_diag_matrix(d)
    eye = srb_mat.srb_mat_identity(3)
    perm = srb_mat.srb_mat_permutation_matrix(jnp.array([2, 0, 1], dtype=jnp.int32))
    sub = srb_mat.srb_mat_submatrix(dm, 1, 3, 1, 3)

    _check(bool(jnp.allclose(srb_mat.srb_mat_to_dense(dm), jnp.diag(d))))
    _check(bool(jnp.allclose(srb_mat.srb_mat_diag(dm), d)))
    _check(bool(jnp.allclose(srb_mat.srb_mat_to_dense(eye), jnp.eye(3))))
    _check(bool(jnp.allclose(srb_mat.srb_mat_to_dense(perm), jnp.eye(3)[jnp.array([2, 0, 1])])))
    _check(bool(jnp.allclose(srb_mat.srb_mat_to_dense(sub), jnp.diag(jnp.array([2.0, 3.0])))))
