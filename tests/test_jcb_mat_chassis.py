import jax
import jax.numpy as jnp
import pytest

from arbplusjax import acb_core
from arbplusjax import double_interval as di
from arbplusjax import jcb_mat

from tests._test_checks import _check


def _interval(lo: float, hi: float) -> jnp.ndarray:
    return di.interval(jnp.asarray(lo, dtype=jnp.float64), jnp.asarray(hi, dtype=jnp.float64))


def _box(re: float, im: float) -> jnp.ndarray:
    return acb_core.acb_box(_interval(re, re), _interval(im, im))


def _mat2(a00: complex, a01: complex, a10: complex, a11: complex) -> jnp.ndarray:
    return jnp.stack(
        [
            jnp.stack([_box(a00.real, a00.imag), _box(a01.real, a01.imag)], axis=0),
            jnp.stack([_box(a10.real, a10.imag), _box(a11.real, a11.imag)], axis=0),
        ],
        axis=0,
    )


def _vec2(x0: complex, x1: complex) -> jnp.ndarray:
    return jnp.stack([_box(x0.real, x0.imag), _box(x1.real, x1.imag)], axis=0)


def test_layout_contracts_enforced():
    with pytest.raises(ValueError):
        jcb_mat.jcb_mat_as_box_matrix(jnp.zeros((2, 3, 4), dtype=jnp.float64))
    with pytest.raises(ValueError):
        jcb_mat.jcb_mat_as_box_vector(jnp.zeros((2, 3), dtype=jnp.float64))


def test_matmul_point_and_basic_exact_inputs():
    a = _mat2(1.0 + 1.0j, 2.0 + 0.0j, 0.0 + 1.0j, 3.0 - 1.0j)
    b = _mat2(2.0 + 0.0j, 0.0 + 1.0j, 1.0 - 1.0j, 2.0 + 0.0j)
    expected = jnp.asarray(
        [[(1.0 + 1.0j) * (2.0 + 0.0j) + (2.0 + 0.0j) * (1.0 - 1.0j), (1.0 + 1.0j) * 1.0j + 4.0],
         [(0.0 + 1.0j) * (2.0 + 0.0j) + (3.0 - 1.0j) * (1.0 - 1.0j), (0.0 + 1.0j) * 1.0j + (3.0 - 1.0j) * 2.0]],
        dtype=jnp.complex128,
    )

    point = jcb_mat.jcb_mat_matmul_point(a, b)
    basic = jcb_mat.jcb_mat_matmul_basic(a, b)

    _check(point.shape == (2, 2, 4))
    _check(basic.shape == (2, 2, 4))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(point), expected)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(basic), expected)))
    _check(bool(jnp.all(di.contains(acb_core.acb_real(basic), acb_core.acb_real(point)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_imag(basic), acb_core.acb_imag(point)))))


def test_matvec_solve_jit_grad_and_precision():
    a = _mat2(2.0 + 0.0j, 1.0 + 1.0j, 0.0 + 0.0j, 3.0 - 1.0j)
    x = _vec2(1.0 + 2.0j, -1.0 + 1.0j)
    rhs_expected = jnp.asarray(
        [
            (2.0 + 0.0j) * (1.0 + 2.0j) + (1.0 + 1.0j) * (-1.0 + 1.0j),
            (3.0 - 1.0j) * (-1.0 + 1.0j),
        ],
        dtype=jnp.complex128,
    )

    mv = jcb_mat.jcb_mat_matvec_basic_jit(a, x)
    _check(mv.shape == (2, 4))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(mv), rhs_expected)))

    rhs = _vec2(rhs_expected[0], rhs_expected[1])
    sol = jcb_mat.jcb_mat_solve_basic_jit(a, rhs)
    _check(sol.shape == (2, 4))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(sol), jnp.asarray([1.0 + 2.0j, -1.0 + 1.0j]))))

    hi = jcb_mat.jcb_mat_matvec_basic_prec(a, x, prec_bits=53)
    lo = jcb_mat.jcb_mat_matvec_basic_prec(a, x, prec_bits=20)
    _check(bool(jnp.all(di.contains(acb_core.acb_real(lo), acb_core.acb_real(hi)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_imag(lo), acb_core.acb_imag(hi)))))

    def loss(t):
        tt = _box(t, 0.0)
        mat = jnp.stack(
            [
                jnp.stack([tt, _box(1.0, 0.0)], axis=0),
                jnp.stack([_box(0.0, 0.0), _box(2.0, 0.0)], axis=0),
            ],
            axis=0,
        )
        out = jcb_mat.jcb_mat_matvec_point(mat, _vec2(1.0 + 0.0j, 2.0 + 0.0j))
        return jnp.real(jnp.sum(acb_core.acb_midpoint(out)))

    g = jax.grad(loss)(jnp.asarray(3.0, dtype=jnp.float64))
    _check(bool(jnp.isfinite(g)))


def test_triangular_solve_and_lu_substrate():
    a = _mat2(2.0 + 0.0j, 0.0 + 0.0j, 1.0 + 1.0j, 3.0 + 0.0j)
    rhs = _vec2(4.0 + 0.0j, 10.0 + 2.0j)
    sol = jcb_mat.jcb_mat_triangular_solve_basic_jit(a, rhs, lower=True)
    _check(sol.shape == (2, 4))
    expected = jnp.asarray([2.0 + 0.0j, (8.0 + 0.0j) / 3.0], dtype=jnp.complex128)
    _check(bool(jnp.allclose(acb_core.acb_midpoint(sol), expected)))

    full = _mat2(2.0 + 0.0j, 1.0 + 0.0j, 4.0 + 0.0j, 3.0 + 1.0j)
    p, l, u = jcb_mat.jcb_mat_lu_basic_jit(full)
    p_mid = acb_core.acb_midpoint(p)
    l_mid = acb_core.acb_midpoint(l)
    u_mid = acb_core.acb_midpoint(u)
    a_mid = acb_core.acb_midpoint(full)
    _check(p.shape == (2, 2, 4))
    _check(l.shape == (2, 2, 4))
    _check(u.shape == (2, 2, 4))
    _check(bool(jnp.allclose(p_mid @ a_mid, l_mid @ u_mid)))
