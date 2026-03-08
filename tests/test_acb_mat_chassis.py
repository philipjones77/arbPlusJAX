import jax
import jax.numpy as jnp

from arbplusjax import acb_core
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
    _check(bool(jnp.allclose(acb_core.acb_midpoint(sol), jax.scipy.linalg.solve_triangular(a_mid, b_mid, lower=True))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(p) @ a_mid, acb_core.acb_midpoint(l) @ acb_core.acb_midpoint(u))))


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
