import jax
import jax.numpy as jnp

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
    _check(bool(jnp.allclose(di.midpoint(sol), jax.scipy.linalg.solve_triangular(a_mid, b_mid, lower=True))))
    _check(bool(jnp.allclose(di.midpoint(p) @ a_mid, di.midpoint(l) @ di.midpoint(u))))


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
