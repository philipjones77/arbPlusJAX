import jax
import jax.numpy as jnp

from arbplusjax import api
from arbplusjax import arb_mat
from arbplusjax import double_interval as di


from tests._test_checks import _check
def _interval(lo: float, hi: float) -> jnp.ndarray:
    return di.interval(jnp.float64(lo), jnp.float64(hi))


def test_jit_compiles_and_keeps_interval_order():
    m = jnp.array(
        [
            [[[0.1, 0.2], [0.0, 0.1]], [[0.2, 0.3], [0.4, 0.5]]],
            [[[-0.2, -0.1], [0.3, 0.4]], [[0.1, 0.2], [0.6, 0.7]]],
        ],
        dtype=jnp.float64,
    )
    out = arb_mat.arb_mat_2x2_det_batch_jit(m)
    _check(out.shape == (2, 2))
    _check(bool(jnp.all(out[:, 0] <= out[:, 1])))


def test_grad_path_for_midpoint():
    def loss(t):
        tt = jnp.asarray(t, dtype=jnp.float64)
        a00 = _interval(tt, tt)
        a01 = _interval(0.1, 0.2)
        a10 = _interval(0.0, 0.1)
        a11 = _interval(0.3, 0.4)
        mat = jnp.stack([jnp.stack([a00, a01], axis=0), jnp.stack([a10, a11], axis=0)], axis=0)
        out = arb_mat.arb_mat_2x2_trace(mat)
        return di.midpoint(out)

    g = jax.grad(loss)(jnp.float64(0.2))
    _check(bool(jnp.isfinite(g)))


def test_precision_semantics_wider_at_lower_precision():
    a00 = _interval(0.1, 0.2)
    a01 = _interval(0.0, 0.1)
    a10 = _interval(0.2, 0.3)
    a11 = _interval(0.4, 0.5)
    mat = jnp.stack([jnp.stack([a00, a01], axis=0), jnp.stack([a10, a11], axis=0)], axis=0)
    hi = arb_mat.arb_mat_2x2_det_prec(mat, prec_bits=53)
    lo = arb_mat.arb_mat_2x2_det_prec(mat, prec_bits=20)
    _check(bool(di.contains(lo, hi)))


def test_nxn_matmul_matvec_solve():
    a = jnp.array(
        [
            [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
            [[0.0, 0.0], [3.0, 3.0], [1.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0], [4.0, 4.0]],
        ],
        dtype=jnp.float64,
    )
    b = jnp.array(
        [
            [[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]],
        ],
        dtype=jnp.float64,
    )
    x = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=jnp.float64)

    mm = arb_mat.arb_mat_matmul_basic(a, b)
    mv = arb_mat.arb_mat_matvec_basic(a, x)
    sol = arb_mat.arb_mat_solve_jit(a, mv)

    a_mid = di.midpoint(a)
    b_mid = di.midpoint(b)
    x_mid = di.midpoint(x)

    _check(mm.shape == (3, 3, 2))
    _check(mv.shape == (3, 2))
    _check(sol.shape == (3, 2))
    _check(bool(jnp.allclose(di.midpoint(mm), a_mid @ b_mid)))
    _check(bool(jnp.allclose(di.midpoint(mv), a_mid @ x_mid)))
    _check(bool(jnp.allclose(di.midpoint(sol), x_mid)))


def test_nxn_matvec_cached_and_sqr():
    a = jnp.array(
        [
            [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
            [[0.0, 0.0], [3.0, 3.0], [1.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0], [4.0, 4.0]],
        ],
        dtype=jnp.float64,
    )
    x = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=jnp.float64)

    cache = arb_mat.arb_mat_matvec_cached_prepare(a)
    mv = arb_mat.arb_mat_matvec_cached_apply_jit(cache, x)
    sq = arb_mat.arb_mat_sqr_jit(a)

    a_mid = di.midpoint(a)
    x_mid = di.midpoint(x)

    _check(mv.shape == (3, 2))
    _check(sq.shape == (3, 3, 2))
    _check(bool(jnp.allclose(di.midpoint(mv), a_mid @ x_mid)))
    _check(bool(jnp.allclose(di.midpoint(sq), a_mid @ a_mid)))


def test_rmatvec_and_cached_rmatvec():
    a = jnp.array(
        [
            [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
            [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
        ],
        dtype=jnp.float64,
    )
    x = jnp.array([[1.0, 1.0], [-1.0, -1.0]], dtype=jnp.float64)
    cache = arb_mat.arb_mat_rmatvec_cached_prepare(a)
    got = arb_mat.arb_mat_rmatvec_basic(a, x)
    cached = arb_mat.arb_mat_rmatvec_cached_apply(cache, x)
    expected = di.midpoint(a).T @ di.midpoint(x)

    _check(bool(jnp.allclose(di.midpoint(got), expected)))
    _check(bool(jnp.allclose(di.midpoint(cached), expected)))
    _check(bool(jnp.allclose(di.midpoint(arb_mat.arb_mat_rmatvec_cached_apply_jit(cache, x)), expected)))


def test_cached_prepare_prec_and_batch_helpers():
    a = jnp.array(
        [
            [
                [[1.0, 1.0], [2.0, 2.0]],
                [[0.0, 0.0], [3.0, 3.0]],
            ],
            [
                [[2.0, 2.0], [0.0, 0.0]],
                [[1.0, 1.0], [4.0, 4.0]],
            ],
        ],
        dtype=jnp.float64,
    )

    fixed = arb_mat.arb_mat_matvec_cached_prepare_batch_fixed(a)
    padded = arb_mat.arb_mat_matvec_cached_prepare_batch_padded(a, pad_to=4)
    fixed_prec = arb_mat.arb_mat_matvec_cached_prepare_batch_fixed_prec(a, prec_bits=53)

    _check(fixed.shape == (2, 2, 2, 2))
    _check(padded.shape == (4, 2, 2, 2))
    _check(fixed_prec.shape == (2, 2, 2, 2))
    _check(bool(jnp.allclose(di.midpoint(fixed), di.midpoint(a))))


def test_dense_matvec_plan_prepare_and_apply():
    a = jnp.array(
        [
            [[1.0, 1.0], [2.0, 2.0]],
            [[0.0, 0.0], [3.0, 3.0]],
        ],
        dtype=jnp.float64,
    )
    x = jnp.array([[1.0, 1.0], [2.0, 2.0]], dtype=jnp.float64)

    plan = arb_mat.arb_mat_dense_matvec_plan_prepare(a)
    out = arb_mat.arb_mat_dense_matvec_plan_apply(plan, x)

    _check(plan.rows == 2)
    _check(plan.cols == 2)
    _check(plan.algebra == "arb")
    _check(bool(jnp.allclose(di.midpoint(out), di.midpoint(a) @ di.midpoint(x))))


def test_dense_lu_plan_matrix_rhs_and_structure_helpers():
    a = jnp.array(
        [
            [[4.0, 4.0], [1.0, 1.0], [0.0, 0.0]],
            [[2.0, 2.0], [3.0, 3.0], [1.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        ],
        dtype=jnp.float64,
    )
    x = jnp.array(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[2.0, 2.0], [1.0, 1.0]],
            [[-1.0, -1.0], [3.0, 3.0]],
        ],
        dtype=jnp.float64,
    )
    rhs = di.interval(di.midpoint(a) @ di.midpoint(x), di.midpoint(a) @ di.midpoint(x))
    plan = arb_mat.arb_mat_dense_lu_solve_plan_prepare(a)
    sol = arb_mat.arb_mat_dense_lu_solve_plan_apply(plan, rhs)
    p = arb_mat.arb_mat_permutation_matrix(jnp.array([2, 0, 1], dtype=jnp.int32))
    t = arb_mat.arb_mat_transpose(a)
    d = arb_mat.arb_mat_diag(a)
    dm = arb_mat.arb_mat_diag_matrix(d)
    sub = arb_mat.arb_mat_submatrix(a, 0, 2, 1, 3)

    a_mid = di.midpoint(a)
    x_mid = di.midpoint(x)

    _check(sol.shape == (3, 2, 2))
    _check(plan.rows == 3)
    _check(plan.algebra == "arb")
    _check(bool(jnp.allclose(di.midpoint(sol), x_mid)))
    _check(bool(jnp.allclose(di.midpoint(p), jnp.eye(3)[jnp.array([2, 0, 1])])))
    _check(bool(jnp.allclose(di.midpoint(t), a_mid.T)))
    _check(bool(jnp.allclose(di.midpoint(d), jnp.diag(a_mid))))
    _check(bool(jnp.allclose(di.midpoint(dm), jnp.diag(jnp.diag(a_mid)))))
    _check(bool(jnp.allclose(di.midpoint(sub), a_mid[:2, 1:3])))


def test_diag_matrix_preserves_interval_width():
    d = jnp.array(
        [
            [0.9, 1.1],
            [1.8, 2.2],
            [-0.3, 0.4],
        ],
        dtype=jnp.float64,
    )
    dm = arb_mat.arb_mat_diag_matrix(d)
    idx = jnp.arange(3)

    _check(dm.shape == (3, 3, 2))
    _check(bool(jnp.allclose(dm[idx, idx, :], d)))
    _check(bool(jnp.allclose(dm[..., 0, 1, :], jnp.zeros((2,), dtype=jnp.float64))))


def test_block_structure_helpers():
    a11 = di.interval(jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64), jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64))
    a12 = di.interval(jnp.array([[5.0], [6.0]], dtype=jnp.float64), jnp.array([[5.0], [6.0]], dtype=jnp.float64))
    a21 = di.interval(jnp.array([[7.0, 8.0]], dtype=jnp.float64), jnp.array([[7.0, 8.0]], dtype=jnp.float64))
    a22 = di.interval(jnp.array([[9.0]], dtype=jnp.float64), jnp.array([[9.0]], dtype=jnp.float64))
    assembled = arb_mat.arb_mat_block_assemble(((a11, a12), (a21, a22)))
    block_diag = arb_mat.arb_mat_block_diag((a11, a22))
    extracted = arb_mat.arb_mat_block_extract(assembled, (2, 1), (2, 1), 1, 0)
    row = arb_mat.arb_mat_block_row(assembled, (2, 1), 0)
    col = arb_mat.arb_mat_block_col(assembled, (2, 1), 1)
    product = arb_mat.arb_mat_block_matmul(((a11, a12),), ((a11, a12), (a21, a22)))

    assembled_mid = jnp.array([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0], [7.0, 8.0, 9.0]], dtype=jnp.float64)
    block_diag_mid = jnp.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 9.0]], dtype=jnp.float64)

    _check(bool(jnp.allclose(di.midpoint(assembled), assembled_mid)))
    _check(bool(jnp.allclose(di.midpoint(block_diag), block_diag_mid)))
    _check(bool(jnp.allclose(di.midpoint(extracted), jnp.array([[7.0, 8.0]], dtype=jnp.float64))))
    _check(bool(jnp.allclose(di.midpoint(row), assembled_mid[:2, :])))
    _check(bool(jnp.allclose(di.midpoint(col), assembled_mid[:, 2:3])))
    _check(bool(jnp.allclose(di.midpoint(product), assembled_mid[:2, :] @ assembled_mid)))


def test_nxn_triangular_solve_and_lu():
    a = jnp.array(
        [
            [[2.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [3.0, 3.0], [0.0, 0.0]],
            [[-1.0, -1.0], [2.0, 2.0], [4.0, 4.0]],
        ],
        dtype=jnp.float64,
    )
    b = jnp.array([[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]], dtype=jnp.float64)

    sol = arb_mat.arb_mat_triangular_solve_jit(a, b, lower=True)
    p, l, u = arb_mat.arb_mat_lu_jit(a)

    a_mid = di.midpoint(a)
    b_mid = di.midpoint(b)

    _check(sol.shape == (3, 2))
    _check(p.shape == (3, 3, 2))
    _check(l.shape == (3, 3, 2))
    _check(u.shape == (3, 3, 2))
    ref = jax.lax.linalg.triangular_solve(a_mid, b_mid[..., None], left_side=True, lower=True)
    _check(bool(jnp.allclose(di.midpoint(sol), jnp.squeeze(ref, axis=-1))))
    _check(bool(jnp.allclose(di.midpoint(p) @ a_mid, di.midpoint(l) @ di.midpoint(u))))


def test_nxn_inv_and_qr():
    a = jnp.array(
        [
            [[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [3.0, 3.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 0.0], [4.0, 4.0]],
        ],
        dtype=jnp.float64,
    )

    inv = arb_mat.arb_mat_inv_jit(a)
    q, r = arb_mat.arb_mat_qr_jit(a)

    a_mid = di.midpoint(a)
    inv_mid = jnp.linalg.inv(a_mid)
    q_mid, r_mid = jnp.linalg.qr(a_mid)

    _check(inv.shape == (3, 3, 2))
    _check(q.shape == (3, 3, 2))
    _check(r.shape == (3, 3, 2))
    _check(bool(jnp.allclose(di.midpoint(inv), inv_mid)))
    _check(bool(jnp.allclose(di.midpoint(q), q_mid)))
    _check(bool(jnp.allclose(di.midpoint(r), r_mid)))


def test_nxn_det_and_trace():
    a = jnp.array(
        [
            [[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [3.0, 3.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 0.0], [4.0, 4.0]],
        ],
        dtype=jnp.float64,
    )
    det_point = arb_mat.arb_mat_det_jit(a)
    det_basic = arb_mat.arb_mat_det_basic(a)
    tr_point = arb_mat.arb_mat_trace_jit(a)
    tr_basic = arb_mat.arb_mat_trace_basic(a)

    a_mid = di.midpoint(a)

    _check(det_point.shape == (2,))
    _check(det_basic.shape == (2,))
    _check(tr_point.shape == (2,))
    _check(tr_basic.shape == (2,))
    _check(bool(jnp.allclose(di.midpoint(det_point), jnp.linalg.det(a_mid))))
    _check(bool(jnp.allclose(di.midpoint(det_basic), jnp.linalg.det(a_mid))))
    _check(bool(jnp.allclose(di.midpoint(tr_point), jnp.trace(a_mid))))
    _check(bool(jnp.allclose(di.midpoint(tr_basic), jnp.trace(a_mid))))


def test_norms_and_batch_helpers():
    a = jnp.array(
        [
            [[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]],
            [[-1.0, -1.0], [3.0, 3.0], [2.0, 2.0]],
            [[0.5, 0.5], [0.0, 0.0], [4.0, 4.0]],
        ],
        dtype=jnp.float64,
    )
    batch = jnp.stack([a, a + 1.0], axis=0)
    mid = di.midpoint(a)

    eye = arb_mat.arb_mat_identity(3)
    zeros = arb_mat.arb_mat_zero(3)
    fro = arb_mat.arb_mat_norm_fro(a)
    one = arb_mat.arb_mat_norm_1(a)
    infn = arb_mat.arb_mat_norm_inf(a)
    batch_fro = arb_mat.arb_mat_norm_fro_batch_fixed(batch)

    _check(bool(jnp.allclose(di.midpoint(eye), jnp.eye(3))))
    _check(bool(jnp.allclose(di.midpoint(zeros), jnp.zeros((3, 3)))))
    _check(bool(jnp.allclose(di.midpoint(fro), jnp.linalg.norm(mid, ord="fro"))))
    _check(bool(jnp.allclose(di.midpoint(one), jnp.linalg.norm(mid, ord=1))))
    _check(bool(jnp.allclose(di.midpoint(infn), jnp.linalg.norm(mid, ord=jnp.inf))))
    _check(batch_fro.shape == (2, 2))


def test_batch_solve_inv_triangular_lu_qr_helpers_and_api():
    a = jnp.array(
        [
            [
                [[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [3.0, 3.0], [1.0, 1.0]],
                [[1.0, 1.0], [0.0, 0.0], [4.0, 4.0]],
            ],
            [
                [[3.0, 3.0], [0.0, 0.0], [1.0, 1.0]],
                [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            ],
        ],
        dtype=jnp.float64,
    )
    x = jnp.array(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[-1.0, -1.0], [0.5, 0.5], [2.0, 2.0]],
        ],
        dtype=jnp.float64,
    )
    rhs = arb_mat.arb_mat_matvec_batch_fixed(a, x)
    lower_a = jnp.tril(a)

    solve = arb_mat.arb_mat_solve_batch_fixed(a, rhs)
    inv = arb_mat.arb_mat_inv_batch_fixed(a)
    tri = arb_mat.arb_mat_triangular_solve_batch_fixed(lower_a, rhs, lower=True)
    p, l, u = arb_mat.arb_mat_lu_batch_fixed(a)
    q, r = arb_mat.arb_mat_qr_batch_fixed(a)

    _check(solve.shape == (2, 3, 2))
    _check(inv.shape == (2, 3, 3, 2))
    _check(tri.shape == (2, 3, 2))
    _check(p.shape == (2, 3, 3, 2))
    _check(l.shape == (2, 3, 3, 2))
    _check(u.shape == (2, 3, 3, 2))
    _check(q.shape == (2, 3, 3, 2))
    _check(r.shape == (2, 3, 3, 2))
    _check(bool(jnp.allclose(di.midpoint(solve), di.midpoint(x))))

    solve_api = api.eval_interval_batch("arb_mat_solve", a, rhs, mode="basic")
    inv_api = api.eval_interval_batch("arb_mat_inv", a, mode="basic")
    tri_api = api.eval_interval_batch("arb_mat_triangular_solve", lower_a, rhs, mode="basic", lower=True)
    lu_api = api.eval_interval_batch("arb_mat_lu", a, mode="basic")
    qr_api = api.eval_interval_batch("arb_mat_qr", a, mode="basic")

    _check(bool(jnp.allclose(di.midpoint(solve_api), di.midpoint(solve))))
    _check(bool(jnp.allclose(di.midpoint(inv_api), di.midpoint(inv))))
    _check(bool(jnp.allclose(di.midpoint(tri_api), di.midpoint(tri))))
    _check(bool(jnp.allclose(di.midpoint(lu_api[0]), di.midpoint(p))))
    _check(bool(jnp.allclose(di.midpoint(lu_api[1]), di.midpoint(l))))
    _check(bool(jnp.allclose(di.midpoint(lu_api[2]), di.midpoint(u))))
    _check(bool(jnp.allclose(di.midpoint(qr_api[0]), di.midpoint(q))))
    _check(bool(jnp.allclose(di.midpoint(qr_api[1]), di.midpoint(r))))


def test_dense_exact_reference_paths_cover_cached_qr_inv_det_trace_and_norms():
    a_mid = jnp.array(
        [
            [3.0, -1.0, 0.5, 2.0],
            [1.0, 4.0, -2.0, 0.0],
            [0.0, 2.0, 5.0, 1.0],
            [2.0, 0.0, 1.0, 3.0],
        ],
        dtype=jnp.float64,
    )
    x_mid = jnp.array([1.0, -2.0, 0.5, 3.0], dtype=jnp.float64)
    a = di.interval(a_mid, a_mid)
    x = di.interval(x_mid, x_mid)

    cache = arb_mat.arb_mat_matvec_cached_prepare(a)
    cached = arb_mat.arb_mat_matvec_cached_apply_jit(cache, x)
    inv = arb_mat.arb_mat_inv_jit(a)
    q, r = arb_mat.arb_mat_qr_jit(a)
    det = arb_mat.arb_mat_det_jit(a)
    tr = arb_mat.arb_mat_trace_jit(a)
    fro = arb_mat.arb_mat_norm_fro_jit(a)
    one = arb_mat.arb_mat_norm_1_jit(a)
    infn = arb_mat.arb_mat_norm_inf_jit(a)

    inv_mid = jnp.linalg.inv(a_mid)
    q_mid, r_mid = jnp.linalg.qr(a_mid)

    _check(bool(jnp.allclose(di.midpoint(cached), a_mid @ x_mid)))
    _check(bool(jnp.allclose(di.midpoint(inv), inv_mid)))
    _check(bool(jnp.allclose(di.midpoint(q), q_mid)))
    _check(bool(jnp.allclose(di.midpoint(r), r_mid)))
    _check(bool(jnp.allclose(di.midpoint(det), jnp.linalg.det(a_mid))))
    _check(bool(jnp.allclose(di.midpoint(tr), jnp.trace(a_mid))))
    _check(bool(jnp.allclose(di.midpoint(fro), jnp.linalg.norm(a_mid, ord="fro"))))
    _check(bool(jnp.allclose(di.midpoint(one), jnp.linalg.norm(a_mid, ord=1))))
    _check(bool(jnp.allclose(di.midpoint(infn), jnp.linalg.norm(a_mid, ord=jnp.inf))))


def test_large_n_det_enclosure_contains_midpoint_reference_and_precision_nesting():
    a_mid = jnp.array(
        [
            [4.0, 1.0, 0.0, -1.0, 2.0],
            [0.5, 3.5, 1.5, 0.0, -0.5],
            [0.0, 1.0, 5.0, 1.0, 0.25],
            [1.0, 0.0, 0.5, 2.5, 1.0],
            [2.0, -0.5, 0.0, 1.0, 4.5],
        ],
        dtype=jnp.float64,
    )
    radius = jnp.full_like(a_mid, 1e-6)
    a = di.interval(a_mid - radius, a_mid + radius)

    det_basic = arb_mat.arb_mat_det_basic(a)
    det_rigorous = arb_mat.arb_mat_det_rigorous(a)
    det_hi = arb_mat.arb_mat_det_prec(a, prec_bits=53)
    det_lo = arb_mat.arb_mat_det_prec(a, prec_bits=20)
    ref = jnp.linalg.det(a_mid)

    _check(bool(di.contains(det_basic, di.interval(ref, ref))))
    _check(bool(di.contains(det_rigorous, di.interval(ref, ref))))
    _check(bool(di.contains(det_lo, det_hi)))
