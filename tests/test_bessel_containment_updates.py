import jax
import jax.numpy as jnp

from arbplusjax import ball_wrappers
from arbplusjax import double_interval as di

from tests._test_checks import _check


def _contains_point(iv: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    return (iv[0] <= x) & (x <= iv[1])


def test_arb_bessel_yk_adaptive_integer_crossing_returns_full_interval():
    nu = di.interval(jnp.float64(-0.1), jnp.float64(0.1))
    z = di.interval(jnp.float64(2.0), jnp.float64(2.5))
    y = ball_wrappers.arb_ball_bessel_y_adaptive(nu, z, prec_bits=80)
    k = ball_wrappers.arb_ball_bessel_k_adaptive(nu, z, prec_bits=80)
    ks = ball_wrappers.arb_ball_bessel_k_scaled_adaptive(nu, z, prec_bits=80)
    _check(bool(jnp.isneginf(y[0]) & jnp.isposinf(y[1])))
    _check(bool(jnp.isneginf(k[0]) & jnp.isposinf(k[1])))
    _check(bool(jnp.isneginf(ks[0]) & jnp.isposinf(ks[1])))


def test_acb_bessel_yk_integer_crossing_returns_full_box():
    nu = jnp.array([-0.1, 0.1, -0.01, 0.01], dtype=jnp.float64)
    z = jnp.array([2.0, 2.5, -0.1, 0.1], dtype=jnp.float64)
    y = ball_wrappers.acb_ball_bessel_y(nu, z, prec_bits=80)
    k = ball_wrappers.acb_ball_bessel_k(nu, z, prec_bits=80)
    ks = ball_wrappers.acb_ball_bessel_k_scaled(nu, z, prec_bits=80)
    _check(bool(jnp.all(jnp.isinf(y))))
    _check(bool(jnp.all(jnp.isinf(k))))
    _check(bool(jnp.all(jnp.isinf(ks))))


def test_arb_bessel_asymptotic_bound_contains_samples():
    nu = di.interval(jnp.float64(0.2), jnp.float64(0.25))
    z = di.interval(jnp.float64(30.0), jnp.float64(34.0))
    out = ball_wrappers.arb_ball_bessel_j(nu, z, prec_bits=80)
    nus = jnp.linspace(nu[0], nu[1], 3)
    zs = jnp.linspace(z[0], z[1], 5)
    vals = jnp.array([ball_wrappers._real_bessel_eval_j(u, v) for u in nus for v in zs], dtype=jnp.float64)
    _check(bool(jnp.all(jnp.isfinite(vals))))
    _check(bool(jnp.all(jax.vmap(lambda t: _contains_point(out, t))(vals))))
