import jax
import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import api
from arbplusjax import acb_mat
from arbplusjax import double_interval as di


from tests._test_checks import _check
def _interval(lo: float, hi: float) -> jnp.ndarray:
    return di.interval(jnp.float64(lo), jnp.float64(hi))


def _box(re_lo: float, re_hi: float, im_lo: float, im_hi: float) -> jnp.ndarray:
    return acb_core.acb_box(_interval(re_lo, re_hi), _interval(im_lo, im_hi))


def test_jit_compiles_and_keeps_interval_order():
    m = jnp.array(
        [
            [
                [[0.1, 0.2, 0.0, 0.1], [0.0, 0.1, -0.1, 0.0]],
                [[0.2, 0.3, 0.05, 0.1], [0.4, 0.5, -0.2, -0.1]],
            ],
            [
                [[-0.2, -0.1, 0.1, 0.2], [0.3, 0.4, 0.0, 0.1]],
                [[0.1, 0.2, -0.1, 0.0], [0.6, 0.7, 0.2, 0.3]],
            ],
        ],
        dtype=jnp.float64,
    )
    out = acb_mat.acb_mat_2x2_det_batch_jit(m)
    _check(out.shape == (2, 4))
    _check(bool(jnp.all(out[:, 0] <= out[:, 1])))
    _check(bool(jnp.all(out[:, 2] <= out[:, 3])))


def test_grad_path_for_midpoint():
    def loss(t):
        tt = jnp.asarray(t, dtype=jnp.float64)
        a00 = acb_core.acb_box(di.interval(tt, tt), di.interval(jnp.float64(0.1), jnp.float64(0.1)))
        a01 = _box(0.1, 0.2, 0.0, 0.1)
        a10 = _box(0.0, 0.1, -0.1, 0.0)
        a11 = _box(0.3, 0.4, 0.0, 0.1)
        mat = jnp.stack([jnp.stack([a00, a01], axis=0), jnp.stack([a10, a11], axis=0)], axis=0)
        out = acb_mat.acb_mat_2x2_trace(mat)
        return jnp.real(acb_core.acb_midpoint(out))

    g = jax.grad(loss)(jnp.float64(0.2))
    _check(bool(jnp.isfinite(g)))


def test_precision_semantics_wider_at_lower_precision():
    a00 = _box(0.1, 0.2, 0.0, 0.1)
    a01 = _box(0.0, 0.1, -0.1, 0.0)
    a10 = _box(0.2, 0.3, 0.05, 0.1)
    a11 = _box(0.4, 0.5, -0.2, -0.1)
    mat = jnp.stack([jnp.stack([a00, a01], axis=0), jnp.stack([a10, a11], axis=0)], axis=0)
    hi = acb_mat.acb_mat_2x2_det_prec(mat, prec_bits=53)
    lo = acb_mat.acb_mat_2x2_det_prec(mat, prec_bits=20)
    _check(bool(di.contains(acb_core.acb_real(lo), acb_core.acb_real(hi))))
    _check(bool(di.contains(acb_core.acb_imag(lo), acb_core.acb_imag(hi))))


def test_nxn_matmul_matvec_solve():
    a = jnp.array(
        [
            [[1.0, 1.0, 0.0, 0.0], [2.0, 2.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 0.0, 0.0], [1.0, 1.0, -1.0, -1.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 0.0, 0.0]],
        ],
        dtype=jnp.float64,
    )
    b = jnp.array(
        [
            [[2.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]],
            [[1.0, 1.0, -1.0, -1.0], [2.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [3.0, 3.0, 0.0, 0.0]],
        ],
        dtype=jnp.float64,
    )
    x = jnp.array(
        [[1.0, 1.0, 0.0, 0.0], [2.0, 2.0, -1.0, -1.0], [3.0, 3.0, 1.0, 1.0]],
        dtype=jnp.float64,
    )

    mm = acb_mat.acb_mat_matmul_basic(a, b)
    mv = acb_mat.acb_mat_matvec_basic(a, x)
    sol = acb_mat.acb_mat_solve_jit(a, mv)

    a_mid = acb_core.acb_midpoint(a)
    b_mid = acb_core.acb_midpoint(b)
    x_mid = acb_core.acb_midpoint(x)

    _check(mm.shape == (3, 3, 4))
    _check(mv.shape == (3, 4))
    _check(sol.shape == (3, 4))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(mm), a_mid @ b_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(mv), a_mid @ x_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(sol), x_mid)))


def test_nxn_matvec_cached_and_sqr():
    a = jnp.array(
        [
            [[1.0, 1.0, 0.0, 0.0], [2.0, 2.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 0.0, 0.0], [1.0, 1.0, -1.0, -1.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 0.0, 0.0]],
        ],
        dtype=jnp.float64,
    )
    x = jnp.array(
        [[1.0, 1.0, 0.0, 0.0], [2.0, 2.0, -1.0, -1.0], [3.0, 3.0, 1.0, 1.0]],
        dtype=jnp.float64,
    )

    cache = acb_mat.acb_mat_matvec_cached_prepare(a)
    mv = acb_mat.acb_mat_matvec_cached_apply_jit(cache, x)
    sq = acb_mat.acb_mat_sqr_jit(a)

    a_mid = acb_core.acb_midpoint(a)
    x_mid = acb_core.acb_midpoint(x)

    _check(mv.shape == (3, 4))
    _check(sq.shape == (3, 3, 4))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(mv), a_mid @ x_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(sq), a_mid @ a_mid)))


def test_rmatvec_and_cached_rmatvec():
    a = jnp.array(
        [
            [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, -0.5, -0.5], [0.0, 0.0, 0.0, 0.0]],
            [[3.0, 3.0, 0.25, 0.25], [4.0, 4.0, 0.0, 0.0], [5.0, 5.0, -1.0, -1.0]],
        ],
        dtype=jnp.float64,
    )
    x = jnp.array([[1.0, 1.0, -0.5, -0.5], [-1.0, -1.0, 0.25, 0.25]], dtype=jnp.float64)
    cache = acb_mat.acb_mat_rmatvec_cached_prepare(a)
    got = acb_mat.acb_mat_rmatvec_basic(a, x)
    cached = acb_mat.acb_mat_rmatvec_cached_apply(cache, x)
    expected = acb_core.acb_midpoint(a).T @ acb_core.acb_midpoint(x)

    _check(bool(jnp.allclose(acb_core.acb_midpoint(got), expected)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(cached), expected)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(acb_mat.acb_mat_rmatvec_cached_apply_jit(cache, x)), expected)))


def test_cached_prepare_prec_and_batch_helpers():
    a = jnp.array(
        [
            [
                [[1.0, 1.0, 0.0, 0.0], [2.0, 2.0, 1.0, 1.0]],
                [[0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 0.0, 0.0]],
            ],
            [
                [[2.0, 2.0, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]],
                [[1.0, 1.0, -0.5, -0.5], [4.0, 4.0, 0.0, 0.0]],
            ],
        ],
        dtype=jnp.float64,
    )

    fixed = acb_mat.acb_mat_matvec_cached_prepare_batch_fixed(a)
    padded = acb_mat.acb_mat_matvec_cached_prepare_batch_padded(a, pad_to=4)
    fixed_prec = acb_mat.acb_mat_matvec_cached_prepare_batch_fixed_prec(a, prec_bits=53)

    _check(fixed.shape == (2, 2, 2, 4))
    _check(padded.shape == (4, 2, 2, 4))
    _check(fixed_prec.shape == (2, 2, 2, 4))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(fixed), acb_core.acb_midpoint(a))))


def test_dense_matvec_plan_prepare_and_apply():
    a = jnp.array(
        [
            [[1.0, 1.0, 0.0, 0.0], [2.0, 2.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 0.0, 0.0]],
        ],
        dtype=jnp.float64,
    )
    x = jnp.array(
        [[1.0, 1.0, 0.0, 0.0], [2.0, 2.0, -1.0, -1.0]],
        dtype=jnp.float64,
    )

    plan = acb_mat.acb_mat_dense_matvec_plan_prepare(a)
    out = acb_mat.acb_mat_dense_matvec_plan_apply(plan, x)

    _check(plan.rows == 2)
    _check(plan.cols == 2)
    _check(plan.algebra == "acb")
    _check(bool(jnp.allclose(acb_core.acb_midpoint(out), acb_core.acb_midpoint(a) @ acb_core.acb_midpoint(x))))


def test_dense_lu_plan_matrix_rhs_and_structure_helpers():
    a = jnp.array(
        [
            [[4.0, 4.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
            [[2.0, 2.0, -1.0, -1.0], [3.0, 3.0, 0.0, 0.0], [1.0, 1.0, 0.5, 0.5]],
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, -0.5, -0.5], [2.0, 2.0, 0.0, 0.0]],
        ],
        dtype=jnp.float64,
    )
    x = jnp.array(
        [
            [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
            [[2.0, 2.0, -1.0, -1.0], [1.0, 1.0, 0.0, 0.0]],
            [[-1.0, -1.0, 0.5, 0.5], [3.0, 3.0, -1.0, -1.0]],
        ],
        dtype=jnp.float64,
    )
    rhs_mid = acb_core.acb_midpoint(a) @ acb_core.acb_midpoint(x)
    rhs = acb_core.acb_box(di.interval(jnp.real(rhs_mid), jnp.real(rhs_mid)), di.interval(jnp.imag(rhs_mid), jnp.imag(rhs_mid)))
    plan = acb_mat.acb_mat_dense_lu_solve_plan_prepare(a)
    sol = acb_mat.acb_mat_dense_lu_solve_plan_apply(plan, rhs)
    p = acb_mat.acb_mat_permutation_matrix(jnp.array([2, 0, 1], dtype=jnp.int32))
    t = acb_mat.acb_mat_transpose(a)
    ct = acb_mat.acb_mat_conjugate_transpose(a)
    d = acb_mat.acb_mat_diag(a)
    dm = acb_mat.acb_mat_diag_matrix(d)
    sub = acb_mat.acb_mat_submatrix(a, 0, 2, 1, 3)

    a_mid = acb_core.acb_midpoint(a)
    x_mid = acb_core.acb_midpoint(x)

    _check(sol.shape == (3, 2, 4))
    _check(plan.rows == 3)
    _check(plan.algebra == "acb")
    _check(bool(jnp.allclose(acb_core.acb_midpoint(sol), x_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(p), jnp.eye(3)[jnp.array([2, 0, 1])])))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(t), a_mid.T)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(ct), jnp.conj(a_mid).T)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(d), jnp.diag(a_mid))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(dm), jnp.diag(jnp.diag(a_mid)))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(sub), a_mid[:2, 1:3])))


def test_diag_matrix_preserves_box_width():
    d = jnp.array(
        [
            [0.9, 1.1, -0.2, 0.3],
            [1.8, 2.2, 0.0, 0.4],
            [-0.3, 0.4, -1.0, -0.5],
        ],
        dtype=jnp.float64,
    )
    dm = acb_mat.acb_mat_diag_matrix(d)
    idx = jnp.arange(3)

    _check(dm.shape == (3, 3, 4))
    _check(bool(jnp.allclose(dm[idx, idx, :], d)))
    _check(bool(jnp.allclose(dm[..., 0, 1, :], jnp.zeros((4,), dtype=jnp.float64))))


def test_block_structure_helpers():
    a11 = acb_core.acb_box(di.interval(jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64), jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64)), di.interval(jnp.zeros((2, 2), dtype=jnp.float64), jnp.zeros((2, 2), dtype=jnp.float64)))
    a12 = acb_core.acb_box(di.interval(jnp.array([[5.0], [6.0]], dtype=jnp.float64), jnp.array([[5.0], [6.0]], dtype=jnp.float64)), di.interval(jnp.zeros((2, 1), dtype=jnp.float64), jnp.zeros((2, 1), dtype=jnp.float64)))
    a21 = acb_core.acb_box(di.interval(jnp.array([[7.0, 8.0]], dtype=jnp.float64), jnp.array([[7.0, 8.0]], dtype=jnp.float64)), di.interval(jnp.zeros((1, 2), dtype=jnp.float64), jnp.zeros((1, 2), dtype=jnp.float64)))
    a22 = acb_core.acb_box(di.interval(jnp.array([[9.0]], dtype=jnp.float64), jnp.array([[9.0]], dtype=jnp.float64)), di.interval(jnp.zeros((1, 1), dtype=jnp.float64), jnp.zeros((1, 1), dtype=jnp.float64)))
    assembled = acb_mat.acb_mat_block_assemble(((a11, a12), (a21, a22)))
    block_diag = acb_mat.acb_mat_block_diag((a11, a22))
    extracted = acb_mat.acb_mat_block_extract(assembled, (2, 1), (2, 1), 1, 0)
    row = acb_mat.acb_mat_block_row(assembled, (2, 1), 0)
    col = acb_mat.acb_mat_block_col(assembled, (2, 1), 1)
    product = acb_mat.acb_mat_block_matmul(((a11, a12),), ((a11, a12), (a21, a22)))

    assembled_mid = jnp.array([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0], [7.0, 8.0, 9.0]], dtype=jnp.complex128)
    block_diag_mid = jnp.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 9.0]], dtype=jnp.complex128)

    _check(bool(jnp.allclose(acb_core.acb_midpoint(assembled), assembled_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(block_diag), block_diag_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(extracted), jnp.array([[7.0, 8.0]], dtype=jnp.complex128))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(row), assembled_mid[:2, :])))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(col), assembled_mid[:, 2:3])))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(product), assembled_mid[:2, :] @ assembled_mid)))


def test_nxn_triangular_solve_and_lu():
    a = jnp.array(
        [
            [[2.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[-1.0, -1.0, 0.5, 0.5], [2.0, 2.0, -1.0, -1.0], [4.0, 4.0, 0.0, 0.0]],
        ],
        dtype=jnp.float64,
    )
    b = jnp.array(
        [[1.0, 1.0, 0.0, 0.0], [5.0, 5.0, 1.0, 1.0], [9.0, 9.0, -1.0, -1.0]],
        dtype=jnp.float64,
    )

    sol = acb_mat.acb_mat_triangular_solve_jit(a, b, lower=True)
    p, l, u = acb_mat.acb_mat_lu_jit(a)

    a_mid = acb_core.acb_midpoint(a)
    b_mid = acb_core.acb_midpoint(b)

    _check(sol.shape == (3, 4))
    _check(p.shape == (3, 3, 4))
    _check(l.shape == (3, 3, 4))
    _check(u.shape == (3, 3, 4))
    ref = jax.lax.linalg.triangular_solve(a_mid, b_mid[..., None], left_side=True, lower=True)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(sol), jnp.squeeze(ref, axis=-1))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(p) @ a_mid, acb_core.acb_midpoint(l) @ acb_core.acb_midpoint(u))))


def test_nxn_inv_and_qr():
    a = jnp.array(
        [
            [[2.0, 2.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 1.0], [3.0, 3.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
            [[1.0, 1.0, -1.0, -1.0], [0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 0.5, 0.5]],
        ],
        dtype=jnp.float64,
    )

    inv = acb_mat.acb_mat_inv_jit(a)
    q, r = acb_mat.acb_mat_qr_jit(a)

    a_mid = acb_core.acb_midpoint(a)
    inv_mid = jnp.linalg.inv(a_mid)
    q_mid, r_mid = jnp.linalg.qr(a_mid)

    _check(inv.shape == (3, 3, 4))
    _check(q.shape == (3, 3, 4))
    _check(r.shape == (3, 3, 4))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(inv), inv_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(q), q_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(r), r_mid)))


def test_nxn_det_and_trace():
    a = jnp.array(
        [
            [[2.0, 2.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 1.0], [3.0, 3.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
            [[1.0, 1.0, -1.0, -1.0], [0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 0.5, 0.5]],
        ],
        dtype=jnp.float64,
    )
    det_point = acb_mat.acb_mat_det_jit(a)
    det_basic = acb_mat.acb_mat_det_basic(a)
    tr_point = acb_mat.acb_mat_trace_jit(a)
    tr_basic = acb_mat.acb_mat_trace_basic(a)

    a_mid = acb_core.acb_midpoint(a)

    _check(det_point.shape == (4,))
    _check(det_basic.shape == (4,))
    _check(tr_point.shape == (4,))
    _check(tr_basic.shape == (4,))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(det_point), jnp.linalg.det(a_mid))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(det_basic), jnp.linalg.det(a_mid))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(tr_point), jnp.trace(a_mid))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(tr_basic), jnp.trace(a_mid))))


def test_norms_and_batch_helpers():
    a = jnp.array(
        [
            [[2.0, 2.0, 0.0, 0.0], [1.0, 1.0, -1.0, -1.0], [0.0, 0.0, 0.5, 0.5]],
            [[-1.0, -1.0, 0.5, 0.5], [3.0, 3.0, 0.0, 0.0], [2.0, 2.0, -0.5, -0.5]],
            [[0.5, 0.5, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 0.0, 0.0]],
        ],
        dtype=jnp.float64,
    )
    batch = jnp.stack([a, a + 1.0], axis=0)
    mid = acb_core.acb_midpoint(a)

    eye = acb_mat.acb_mat_identity(3)
    zeros = acb_mat.acb_mat_zero(3)
    fro = acb_mat.acb_mat_norm_fro(a)
    one = acb_mat.acb_mat_norm_1(a)
    infn = acb_mat.acb_mat_norm_inf(a)
    batch_fro = acb_mat.acb_mat_norm_fro_batch_fixed(batch)

    _check(bool(jnp.allclose(acb_core.acb_midpoint(eye), jnp.eye(3, dtype=mid.dtype))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(zeros), jnp.zeros((3, 3), dtype=mid.dtype))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(fro), jnp.linalg.norm(mid, ord="fro"))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(one), jnp.linalg.norm(mid, ord=1))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(infn), jnp.linalg.norm(mid, ord=jnp.inf))))
    _check(batch_fro.shape == (2, 4))


def test_batch_solve_inv_triangular_lu_qr_helpers_and_api():
    a = jnp.array(
        [
            [
                [[2.0, 2.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0, 1.0], [3.0, 3.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
                [[1.0, 1.0, -1.0, -1.0], [0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 0.5, 0.5]],
            ],
            [
                [[3.0, 3.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]],
                [[1.0, 1.0, -1.0, -1.0], [2.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.5, 0.5], [2.0, 2.0, 0.0, 0.0]],
            ],
        ],
        dtype=jnp.float64,
    )
    x = jnp.array(
        [
            [[1.0, 1.0, 0.0, 0.0], [2.0, 2.0, -1.0, -1.0], [3.0, 3.0, 1.0, 1.0]],
            [[-1.0, -1.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0], [2.0, 2.0, -0.5, -0.5]],
        ],
        dtype=jnp.float64,
    )
    rhs = acb_mat.acb_mat_matvec_batch_fixed(a, x)
    lower_a = jnp.tril(a)

    solve = acb_mat.acb_mat_solve_batch_fixed(a, rhs)
    inv = acb_mat.acb_mat_inv_batch_fixed(a)
    tri = acb_mat.acb_mat_triangular_solve_batch_fixed(lower_a, rhs, lower=True)
    p, l, u = acb_mat.acb_mat_lu_batch_fixed(a)
    q, r = acb_mat.acb_mat_qr_batch_fixed(a)

    _check(solve.shape == (2, 3, 4))
    _check(inv.shape == (2, 3, 3, 4))
    _check(tri.shape == (2, 3, 4))
    _check(p.shape == (2, 3, 3, 4))
    _check(l.shape == (2, 3, 3, 4))
    _check(u.shape == (2, 3, 3, 4))
    _check(q.shape == (2, 3, 3, 4))
    _check(r.shape == (2, 3, 3, 4))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(solve), acb_core.acb_midpoint(x))))

    solve_api = api.eval_interval_batch("acb_mat_solve", a, rhs, mode="basic")
    inv_api = api.eval_interval_batch("acb_mat_inv", a, mode="basic")
    tri_api = api.eval_interval_batch("acb_mat_triangular_solve", lower_a, rhs, mode="basic", lower=True)
    lu_api = api.eval_interval_batch("acb_mat_lu", a, mode="basic")
    qr_api = api.eval_interval_batch("acb_mat_qr", a, mode="basic")

    _check(bool(jnp.allclose(acb_core.acb_midpoint(solve_api), acb_core.acb_midpoint(solve))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(inv_api), acb_core.acb_midpoint(inv))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(tri_api), acb_core.acb_midpoint(tri))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(lu_api[0]), acb_core.acb_midpoint(p))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(lu_api[1]), acb_core.acb_midpoint(l))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(lu_api[2]), acb_core.acb_midpoint(u))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(qr_api[0]), acb_core.acb_midpoint(q))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(qr_api[1]), acb_core.acb_midpoint(r))))


def test_dense_exact_reference_paths_cover_cached_qr_inv_det_trace_and_norms():
    a_mid = jnp.array(
        [
            [3.0 + 0.5j, -1.0 + 0.0j, 0.5 - 0.25j, 2.0 + 1.0j],
            [1.0 - 1.0j, 4.0 + 0.0j, -2.0 + 0.5j, 0.0 + 0.0j],
            [0.0 + 0.0j, 2.0 + 0.5j, 5.0 + 0.0j, 1.0 - 1.0j],
            [2.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.25j, 3.0 + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    x_mid = jnp.array([1.0 + 0.0j, -2.0 + 1.0j, 0.5 - 0.5j, 3.0 + 0.25j], dtype=jnp.complex128)
    a = acb_core.acb_box(di.interval(jnp.real(a_mid), jnp.real(a_mid)), di.interval(jnp.imag(a_mid), jnp.imag(a_mid)))
    x = acb_core.acb_box(di.interval(jnp.real(x_mid), jnp.real(x_mid)), di.interval(jnp.imag(x_mid), jnp.imag(x_mid)))

    cache = acb_mat.acb_mat_matvec_cached_prepare(a)
    cached = acb_mat.acb_mat_matvec_cached_apply_jit(cache, x)
    inv = acb_mat.acb_mat_inv_jit(a)
    q, r = acb_mat.acb_mat_qr_jit(a)
    det = acb_mat.acb_mat_det_jit(a)
    tr = acb_mat.acb_mat_trace_jit(a)
    fro = acb_mat.acb_mat_norm_fro_jit(a)
    one = acb_mat.acb_mat_norm_1_jit(a)
    infn = acb_mat.acb_mat_norm_inf_jit(a)

    inv_mid = jnp.linalg.inv(a_mid)
    q_mid, r_mid = jnp.linalg.qr(a_mid)

    _check(bool(jnp.allclose(acb_core.acb_midpoint(cached), a_mid @ x_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(inv), inv_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(q), q_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(r), r_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(det), jnp.linalg.det(a_mid))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(tr), jnp.trace(a_mid))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(fro), jnp.linalg.norm(a_mid, ord="fro"))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(one), jnp.linalg.norm(a_mid, ord=1))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(infn), jnp.linalg.norm(a_mid, ord=jnp.inf))))


def test_large_n_det_enclosure_contains_midpoint_reference_and_precision_nesting():
    a_mid = jnp.array(
        [
            [4.0 + 0.0j, 1.0 - 0.5j, 0.0 + 0.0j, -1.0 + 0.25j, 2.0 + 0.0j],
            [0.5 + 0.5j, 3.5 + 0.0j, 1.5 - 0.25j, 0.0 + 0.0j, -0.5 + 0.75j],
            [0.0 + 0.0j, 1.0 + 0.0j, 5.0 + 0.0j, 1.0 - 1.0j, 0.25 + 0.0j],
            [1.0 - 0.25j, 0.0 + 0.0j, 0.5 + 1.0j, 2.5 + 0.0j, 1.0 + 0.0j],
            [2.0 + 0.0j, -0.5 - 0.75j, 0.0 + 0.0j, 1.0 + 0.5j, 4.5 + 0.0j],
        ],
        dtype=jnp.complex128,
    )
    radius = jnp.full(a_mid.shape, 1e-6, dtype=jnp.float64)
    a = acb_core.acb_box(
        di.interval(jnp.real(a_mid) - radius, jnp.real(a_mid) + radius),
        di.interval(jnp.imag(a_mid) - radius, jnp.imag(a_mid) + radius),
    )

    det_basic = acb_mat.acb_mat_det_basic(a)
    det_rigorous = acb_mat.acb_mat_det_rigorous(a)
    det_hi = acb_mat.acb_mat_det_prec(a, prec_bits=53)
    det_lo = acb_mat.acb_mat_det_prec(a, prec_bits=20)
    ref = jnp.linalg.det(a_mid)
    ref_box = acb_core.acb_box(di.interval(jnp.real(ref), jnp.real(ref)), di.interval(jnp.imag(ref), jnp.imag(ref)))

    _check(bool(di.contains(acb_core.acb_real(det_basic), acb_core.acb_real(ref_box))))
    _check(bool(di.contains(acb_core.acb_imag(det_basic), acb_core.acb_imag(ref_box))))
    _check(bool(di.contains(acb_core.acb_real(det_rigorous), acb_core.acb_real(ref_box))))
    _check(bool(di.contains(acb_core.acb_imag(det_rigorous), acb_core.acb_imag(ref_box))))
    _check(bool(di.contains(acb_core.acb_real(det_lo), acb_core.acb_real(det_hi))))
    _check(bool(di.contains(acb_core.acb_imag(det_lo), acb_core.acb_imag(det_hi))))
