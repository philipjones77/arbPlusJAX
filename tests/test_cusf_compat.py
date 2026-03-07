import jax.numpy as jnp

from arbplusjax import cusf_compat
from arbplusjax import double_interval as di
from arbplusjax import hypgeom

from tests._test_checks import _check


def test_cusf_point_bessel_matches_internal_eval():
    v = jnp.asarray([0.0, 0.5, 2.0], dtype=jnp.float64)
    x = jnp.asarray([0.2, 1.3, 3.0], dtype=jnp.float64)
    y = cusf_compat.cusf_besselj(v, x, mode="point")
    ref = hypgeom._real_bessel_eval_j(v, x)
    _check(bool(jnp.allclose(y, ref, rtol=1e-12, atol=1e-12)))


def test_cusf_four_modes_kv_and_derivatives():
    v = di.interval(jnp.float64(0.7), jnp.float64(0.9))
    x = di.interval(jnp.float64(1.1), jnp.float64(1.2))
    b = cusf_compat.cusf_besselk(v, x, mode="basic", prec_bits=80)
    r = cusf_compat.cusf_besselk(v, x, mode="rigorous", prec_bits=80)
    a = cusf_compat.cusf_besselk(v, x, mode="adaptive", prec_bits=80)
    _check(bool(di.contains(r, b)))
    _check(bool(di.contains(a, b)))

    db = cusf_compat.cusf_besselk_deriv(v, x, mode="basic", prec_bits=80)
    dr = cusf_compat.cusf_besselk_deriv(v, x, mode="rigorous", prec_bits=80)
    da = cusf_compat.cusf_besselk_deriv(v, x, mode="adaptive", prec_bits=80)
    _check(bool(di.contains(dr, db)))
    _check(bool(di.contains(da, db)))


def test_cusf_helpers_and_hyp_modes():
    x = jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float64)
    c = jnp.asarray([1.0, 0.5, -0.25], dtype=jnp.float64)
    _check(bool(jnp.all(jnp.isfinite(cusf_compat.cusf_digamma(x, mode="point")))))
    _check(bool(jnp.all(jnp.isfinite(cusf_compat.cusf_tgamma1pmv(x, mode="point")))))
    _check(bool(jnp.all(jnp.isfinite(cusf_compat.cusf_chebyshev(x, c, mode="point")))))
    _check(bool(jnp.all(jnp.isfinite(cusf_compat.cusf_polynomial(x, c, mode="point")))))
    _check(bool(jnp.all(jnp.isfinite(cusf_compat.cusf_poly_rational(x, c, c + 2.0, mode="point")))))

    a = di.interval(jnp.float64(1.0), jnp.float64(1.01))
    b = di.interval(jnp.float64(2.0), jnp.float64(2.01))
    z = di.interval(jnp.float64(0.2), jnp.float64(0.21))
    h1 = cusf_compat.cusf_hyp1f1(a, b, z, mode="basic", prec_bits=80)
    h2 = cusf_compat.cusf_hyp2f1(a, b, b, z, mode="basic", prec_bits=80)
    _check(h1.shape == (2,))
    _check(h2.shape == (2,))
