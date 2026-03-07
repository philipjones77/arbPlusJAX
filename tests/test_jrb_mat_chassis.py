import jax
import jax.numpy as jnp
import pytest

from arbplusjax import double_interval as di
from arbplusjax import jrb_mat

from tests._test_checks import _check


def _interval(lo: float, hi: float) -> jnp.ndarray:
    return di.interval(jnp.asarray(lo, dtype=jnp.float64), jnp.asarray(hi, dtype=jnp.float64))


def _exact_interval(x: float) -> jnp.ndarray:
    return _interval(x, x)


def _mat2(a00: float, a01: float, a10: float, a11: float) -> jnp.ndarray:
    return jnp.stack(
        [
            jnp.stack([_exact_interval(a00), _exact_interval(a01)], axis=0),
            jnp.stack([_exact_interval(a10), _exact_interval(a11)], axis=0),
        ],
        axis=0,
    )


def _vec2(x0: float, x1: float) -> jnp.ndarray:
    return jnp.stack([_exact_interval(x0), _exact_interval(x1)], axis=0)


def test_layout_contracts_enforced():
    with pytest.raises(ValueError):
        jrb_mat.jrb_mat_as_interval_matrix(jnp.zeros((2, 3, 2), dtype=jnp.float64))
    with pytest.raises(ValueError):
        jrb_mat.jrb_mat_as_interval_vector(jnp.zeros((2, 3), dtype=jnp.float64))


def test_matmul_point_and_basic_exact_inputs():
    a = _mat2(1.0, 2.0, 3.0, 4.0)
    b = _mat2(2.0, 0.0, 1.0, 2.0)
    expected = jnp.asarray([[4.0, 4.0], [10.0, 8.0]], dtype=jnp.float64)

    point = jrb_mat.jrb_mat_matmul_point(a, b)
    basic = jrb_mat.jrb_mat_matmul_basic(a, b)

    _check(point.shape == (2, 2, 2))
    _check(basic.shape == (2, 2, 2))
    _check(bool(jnp.allclose(di.midpoint(point), expected)))
    _check(bool(jnp.allclose(di.midpoint(basic), expected)))
    _check(bool(jnp.all(di.contains(basic, point))))


def test_matvec_solve_jit_grad_and_precision():
    a = _mat2(3.0, 1.0, 0.0, 2.0)
    x = _vec2(5.0, 7.0)
    rhs_expected = jnp.asarray([22.0, 14.0], dtype=jnp.float64)

    mv = jrb_mat.jrb_mat_matvec_basic_jit(a, x)
    _check(mv.shape == (2, 2))
    _check(bool(jnp.allclose(di.midpoint(mv), rhs_expected)))

    rhs = _vec2(22.0, 14.0)
    sol = jrb_mat.jrb_mat_solve_basic_jit(a, rhs)
    _check(sol.shape == (2, 2))
    _check(bool(jnp.allclose(di.midpoint(sol), jnp.asarray([5.0, 7.0], dtype=jnp.float64))))

    hi = jrb_mat.jrb_mat_matvec_basic_prec(a, x, prec_bits=53)
    lo = jrb_mat.jrb_mat_matvec_basic_prec(a, x, prec_bits=20)
    _check(bool(jnp.all(di.contains(lo, hi))))

    def loss(t):
        tt = _exact_interval(t)
        mat = jnp.stack(
            [
                jnp.stack([tt, _exact_interval(1.0)], axis=0),
                jnp.stack([_exact_interval(0.0), _exact_interval(2.0)], axis=0),
            ],
            axis=0,
        )
        out = jrb_mat.jrb_mat_matvec_point(mat, _vec2(2.0, 1.0))
        return jnp.sum(di.midpoint(out))

    g = jax.grad(loss)(jnp.asarray(3.0, dtype=jnp.float64))
    _check(bool(jnp.isfinite(g)))
