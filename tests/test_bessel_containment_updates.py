import jax
import jax.numpy as jnp

from arbplusjax import ball_wrappers
from arbplusjax import cubesselk
from arbplusjax import cusf_compat
from arbplusjax import double_interval as di
from arbplusjax import hypgeom

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


def test_arb_bessel_k_scaled_asymptotic_bound_contains_samples():
    nu = di.interval(jnp.float64(0.2), jnp.float64(0.25))
    z = di.interval(jnp.float64(30.0), jnp.float64(34.0))
    out = ball_wrappers.arb_ball_bessel_k_scaled(nu, z, prec_bits=80)
    nus = jnp.linspace(nu[0], nu[1], 3)
    zs = jnp.linspace(z[0], z[1], 5)
    vals = jnp.array([jnp.exp(v) * hypgeom._real_bessel_eval_k(u, v) for u in nus for v in zs], dtype=jnp.float64)
    _check(bool(jnp.all(jnp.isfinite(vals))))
    _check(bool(jnp.all(jax.vmap(lambda t: _contains_point(out, t))(vals))))


def test_acb_bessel_k_scaled_asymptotic_bound_contains_samples():
    nu = jnp.array([0.2, 0.25, -0.05, 0.05], dtype=jnp.float64)
    z = jnp.array([30.0, 34.0, -0.2, 0.2], dtype=jnp.float64)
    out = ball_wrappers.acb_ball_bessel_k_scaled(nu, z, prec_bits=80)
    nus_re = jnp.linspace(nu[0], nu[1], 3)
    nus_im = jnp.linspace(nu[2], nu[3], 3)
    zs_re = jnp.linspace(z[0], z[1], 4)
    zs_im = jnp.linspace(z[2], z[3], 3)
    vals = jnp.array(
        [
            jnp.exp(zz) * hypgeom._complex_bessel_k(nn, zz)
            for nn in (nr + 1j * ni for nr in nus_re for ni in nus_im)
            for zz in (zr + 1j * zi for zr in zs_re for zi in zs_im)
        ],
        dtype=jnp.complex128,
    )
    re_ok = jnp.all((out[0] <= jnp.real(vals)) & (jnp.real(vals) <= out[1]))
    im_ok = jnp.all((out[2] <= jnp.imag(vals)) & (jnp.imag(vals) <= out[3]))
    _check(bool(jnp.all(jnp.isfinite(jnp.real(vals))) & jnp.all(jnp.isfinite(jnp.imag(vals)))))
    _check(bool(re_ok & im_ok))


def test_custom_besselk_alternatives_inherit_tightened_k_bounds():
    nu = di.interval(jnp.float64(0.9), jnp.float64(1.1))
    z = di.interval(jnp.float64(18.0), jnp.float64(20.0))

    cuda_basic = cubesselk.cuda_besselk(nu, z, mode="basic", prec_bits=80)
    cuda_rig = cubesselk.cuda_besselk(nu, z, mode="rigorous", prec_bits=80)
    cuda_adp = cubesselk.cuda_besselk(nu, z, mode="adaptive", prec_bits=80)
    _check(bool(di.contains(cuda_rig, cuda_basic)))
    _check(bool(di.contains(cuda_adp, cuda_basic)))

    cusf_basic = cusf_compat.cusf_besselk(nu, z, mode="basic", prec_bits=80)
    cusf_rig = cusf_compat.cusf_besselk(nu, z, mode="rigorous", prec_bits=80)
    cusf_adp = cusf_compat.cusf_besselk(nu, z, mode="adaptive", prec_bits=80)
    _check(bool(di.contains(cusf_rig, cusf_basic)))
    _check(bool(di.contains(cusf_adp, cusf_basic)))
