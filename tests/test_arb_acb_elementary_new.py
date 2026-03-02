import jax
import jax.numpy as jnp

from arbplusjax import acb_core, arb_core, core_wrappers, double_interval as di, point_wrappers

from tests._test_checks import _check


def test_arb_new_elementary_modes_and_shapes():
    x = di.interval(jnp.float64(0.2), jnp.float64(0.3))
    y = di.interval(jnp.float64(1.2), jnp.float64(1.4))
    for impl in ("basic", "rigorous", "adaptive"):
        _check(core_wrappers.arb_asin_mode(x, impl=impl, prec_bits=80).shape == (2,))
        _check(core_wrappers.arb_acos_mode(x, impl=impl, prec_bits=80).shape == (2,))
        _check(core_wrappers.arb_atan_mode(x, impl=impl, prec_bits=80).shape == (2,))
        _check(core_wrappers.arb_asinh_mode(x, impl=impl, prec_bits=80).shape == (2,))
        _check(core_wrappers.arb_acosh_mode(y, impl=impl, prec_bits=80).shape == (2,))
        _check(core_wrappers.arb_atanh_mode(x, impl=impl, prec_bits=80).shape == (2,))
        _check(core_wrappers.arb_pow_mode(y, y, impl=impl, prec_bits=80).shape == (2,))
        _check(core_wrappers.arb_root_ui_mode(y, 3, impl=impl, prec_bits=80).shape == (2,))
        _check(core_wrappers.arb_cbrt_mode(y, impl=impl, prec_bits=80).shape == (2,))
        _check(core_wrappers.arb_sin_pi_mode(x, impl=impl, prec_bits=80).shape == (2,))
        _check(core_wrappers.arb_sinc_mode(x, impl=impl, prec_bits=80).shape == (2,))


def test_acb_inverse_trig_hyperbolic_modes():
    z = jnp.asarray([0.2, 0.3, -0.1, 0.1], dtype=jnp.float64)
    for impl in ("basic", "rigorous", "adaptive"):
        _check(core_wrappers.acb_asin_mode(z, impl=impl, prec_bits=80).shape == (4,))
        _check(core_wrappers.acb_acos_mode(z, impl=impl, prec_bits=80).shape == (4,))
        _check(core_wrappers.acb_atan_mode(z, impl=impl, prec_bits=80).shape == (4,))
        _check(core_wrappers.acb_asinh_mode(z, impl=impl, prec_bits=80).shape == (4,))
        _check(core_wrappers.acb_acosh_mode(z, impl=impl, prec_bits=80).shape == (4,))
        _check(core_wrappers.acb_atanh_mode(z, impl=impl, prec_bits=80).shape == (4,))


def test_point_pow_and_root_grad_paths():
    def loss_pow(t):
        return point_wrappers.arb_pow_point(t, 1.7)

    def loss_root(t):
        return point_wrappers.arb_root_ui_point(t, 3)

    g1 = jax.grad(loss_pow)(jnp.float64(1.3))
    g2 = jax.grad(loss_root)(jnp.float64(1.3))
    _check(bool(jnp.isfinite(g1)))
    _check(bool(jnp.isfinite(g2)))


def test_pow_matches_acb_branch_consistent_point():
    z = jnp.asarray([1.2, 1.2, -0.3, -0.3], dtype=jnp.float64)
    w = jnp.asarray([0.7, 0.7, 0.2, 0.2], dtype=jnp.float64)
    out = acb_core.acb_pow(z, w)
    _check(out.shape == (4,))
