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


def test_real_core_mode_coverage_for_recently_basic_only_functions():
    x = di.interval(jnp.float64(0.2), jnp.float64(0.3))
    y = di.interval(jnp.float64(1.2), jnp.float64(1.4))
    z = di.interval(jnp.float64(-0.2), jnp.float64(-0.1))
    positive = di.interval(jnp.float64(1.4), jnp.float64(1.6))

    cases = [
        (core_wrappers.arb_abs_mode, (x,)),
        (core_wrappers.arb_acos_mode, (x,)),
        (core_wrappers.arb_acosh_mode, (positive,)),
        (core_wrappers.arb_add_mode, (x, y)),
        (core_wrappers.arb_asin_mode, (x,)),
        (core_wrappers.arb_asinh_mode, (x,)),
        (core_wrappers.arb_atan_mode, (x,)),
        (core_wrappers.arb_atanh_mode, (x,)),
        (core_wrappers.arb_cbrt_mode, (y,)),
        (core_wrappers.arb_cos_mode, (x,)),
        (core_wrappers.arb_cos_pi_mode, (x,)),
        (core_wrappers.arb_cosh_mode, (x,)),
        (core_wrappers.arb_div_mode, (y, positive)),
        (core_wrappers.arb_expm1_mode, (x,)),
        (core_wrappers.arb_fma_mode, (x, y, z)),
        (core_wrappers.arb_gamma_mode, (y,)),
        (core_wrappers.arb_inv_mode, (y,)),
        (core_wrappers.arb_lgamma_mode, (y,)),
        (core_wrappers.arb_log1p_mode, (x,)),
        (core_wrappers.arb_mul_mode, (x, y)),
        (core_wrappers.arb_pow_mode, (positive, y)),
        (core_wrappers.arb_pow_fmpq_mode, (positive, jnp.asarray(3.0), jnp.asarray(2.0))),
        (core_wrappers.arb_pow_fmpz_mode, (positive, jnp.asarray(3.0))),
        (core_wrappers.arb_pow_ui_mode, (positive, 3)),
        (core_wrappers.arb_rgamma_mode, (y,)),
        (core_wrappers.arb_root_mode, (positive, 3)),
        (core_wrappers.arb_root_ui_mode, (positive, 3)),
        (core_wrappers.arb_sign_mode, (x,)),
    ]

    for fn, args in cases:
        basic = fn(*args, impl="basic", prec_bits=80)
        rig = fn(*args, impl="rigorous", prec_bits=80)
        adapt = fn(*args, impl="adaptive", prec_bits=80)
        _check(basic.shape == (2,))
        _check(rig.shape == (2,))
        _check(adapt.shape == (2,))
        _check(bool(di.contains(rig, basic)))
        _check(bool(di.contains(adapt, basic)))


def test_acb_inverse_trig_hyperbolic_modes():
    z = jnp.asarray([0.2, 0.3, -0.1, 0.1], dtype=jnp.float64)
    for impl in ("basic", "rigorous", "adaptive"):
        _check(core_wrappers.acb_asin_mode(z, impl=impl, prec_bits=80).shape == (4,))
        _check(core_wrappers.acb_acos_mode(z, impl=impl, prec_bits=80).shape == (4,))
        _check(core_wrappers.acb_atan_mode(z, impl=impl, prec_bits=80).shape == (4,))
        _check(core_wrappers.acb_asinh_mode(z, impl=impl, prec_bits=80).shape == (4,))
        _check(core_wrappers.acb_acosh_mode(z, impl=impl, prec_bits=80).shape == (4,))
        _check(core_wrappers.acb_atanh_mode(z, impl=impl, prec_bits=80).shape == (4,))


def test_complex_core_mode_coverage_for_real_list_equivalents():
    z = jnp.asarray([0.2, 0.3, -0.1, 0.1], dtype=jnp.float64)
    w = jnp.asarray([1.2, 1.3, -0.2, -0.1], dtype=jnp.float64)
    positive = jnp.asarray([1.4, 1.5, 0.1, 0.2], dtype=jnp.float64)
    exponent = jnp.asarray([0.7, 0.8, 0.1, 0.2], dtype=jnp.float64)

    cases = [
        (core_wrappers.acb_abs_mode, (z,), "real"),
        (core_wrappers.acb_acos_mode, (z,), "complex"),
        (core_wrappers.acb_acosh_mode, (positive,), "complex"),
        (core_wrappers.acb_add_mode, (z, w), "complex"),
        (core_wrappers.acb_asin_mode, (z,), "complex"),
        (core_wrappers.acb_asinh_mode, (z,), "complex"),
        (core_wrappers.acb_atan_mode, (z,), "complex"),
        (core_wrappers.acb_atanh_mode, (z,), "complex"),
        (core_wrappers.acb_cos_mode, (z,), "complex"),
        (core_wrappers.acb_cosh_mode, (z,), "complex"),
        (core_wrappers.acb_div_mode, (w, positive), "complex"),
        (core_wrappers.acb_exp_mode, (z,), "complex"),
        (core_wrappers.acb_expm1_mode, (z,), "complex"),
        (core_wrappers.acb_gamma_mode, (positive,), "complex"),
        (core_wrappers.acb_inv_mode, (positive,), "complex"),
        (core_wrappers.acb_lgamma_mode, (positive,), "complex"),
        (core_wrappers.acb_log_mode, (positive,), "complex"),
        (core_wrappers.acb_log1p_mode, (z,), "complex"),
        (core_wrappers.acb_mul_mode, (z, w), "complex"),
        (core_wrappers.acb_pow_mode, (positive, exponent), "complex"),
        (core_wrappers.acb_pow_fmpz_mode, (positive, 3), "complex"),
        (core_wrappers.acb_pow_si_mode, (positive, 3), "complex"),
        (core_wrappers.acb_pow_ui_mode, (positive, 3), "complex"),
        (core_wrappers.acb_rgamma_mode, (positive,), "complex"),
        (core_wrappers.acb_root_ui_mode, (positive, 3), "complex"),
        (core_wrappers.acb_sin_mode, (z,), "complex"),
        (core_wrappers.acb_sinh_mode, (z,), "complex"),
        (core_wrappers.acb_sqrt_mode, (positive,), "complex"),
        (core_wrappers.acb_sub_mode, (z, w), "complex"),
        (core_wrappers.acb_tan_mode, (z,), "complex"),
        (core_wrappers.acb_tanh_mode, (z,), "complex"),
    ]

    for fn, args, kind in cases:
        basic = fn(*args, impl="basic", prec_bits=80)
        rig = fn(*args, impl="rigorous", prec_bits=80)
        adapt = fn(*args, impl="adaptive", prec_bits=80)
        if kind == "real":
            _check(basic.shape == (2,))
            _check(rig.shape == (2,))
            _check(adapt.shape == (2,))
            _check(bool(di.contains(rig, basic)))
            _check(bool(di.contains(adapt, basic)))
        else:
            _check(basic.shape == (4,))
            _check(rig.shape == (4,))
            _check(adapt.shape == (4,))
            _check(bool(di.contains(acb_core.acb_real(rig), acb_core.acb_real(basic))))
            _check(bool(di.contains(acb_core.acb_imag(rig), acb_core.acb_imag(basic))))
            _check(bool(di.contains(acb_core.acb_real(adapt), acb_core.acb_real(basic))))
            _check(bool(di.contains(acb_core.acb_imag(adapt), acb_core.acb_imag(basic))))


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
