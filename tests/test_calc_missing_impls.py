import jax.numpy as jnp
import jax

from arbplusjax import acb_calc, acb_core, api, arb_calc, calc_wrappers, double_interval as di

from tests._test_checks import _check


def _iv(lo: float, hi: float) -> jax.Array:
    return di.interval(jnp.float64(lo), jnp.float64(hi))


def _box(re_lo: float, re_hi: float, im_lo: float, im_hi: float) -> jax.Array:
    return acb_core.acb_box(_iv(re_lo, re_hi), _iv(im_lo, im_hi))


def test_arb_calc_missing_functions_basic_shapes() -> None:
    a = _iv(-0.2, 0.1)
    b = _iv(0.7, 0.9)
    x = _iv(0.2, 0.4)

    p = arb_calc.arb_calc_partition(a, b, parts=6)
    _check(p.shape == (7, 2))
    _check(bool(jnp.all(p[..., 0] <= p[..., 1])))

    _check(arb_calc.arb_calc_newton_conv_factor(x).shape == (2,))
    _check(arb_calc.arb_calc_newton_step(x).shape == (2,))
    _check(arb_calc.arb_calc_refine_root_bisect(x).shape == (2,))
    _check(arb_calc.arb_calc_refine_root_newton(x).shape == (2,))
    roots = arb_calc.arb_calc_isolate_roots(a, b, max_roots=6)
    _check(roots.shape == (6, 2))
    _check(bool(jnp.all(roots[..., 0] <= roots[..., 1])))


def test_acb_calc_missing_functions_basic_shapes() -> None:
    a = _box(-0.2, 0.1, -0.05, 0.05)
    b = _box(0.7, 0.9, 0.1, 0.2)

    _check(acb_calc.acb_calc_integrate(a, b).shape == (4,))
    _check(acb_calc.acb_calc_integrate_gl_auto_deg(a, b).shape == (4,))
    _check(acb_calc.acb_calc_integrate_taylor(a, b).shape == (4,))
    _check(acb_calc.acb_calc_integrate_opt_init(a, b).shape == (4,))
    cb = acb_calc.acb_calc_cauchy_bound(a, b)
    _check(cb.shape == (2,))
    _check(bool(cb[0] <= cb[1]))


def test_calc_modes_and_api_access() -> None:
    a = _iv(-0.2, 0.1)
    b = _iv(0.7, 0.9)
    x = _iv(0.2, 0.4)

    # point mode via API (falls back to public registry)
    p = api.eval_point("arb_calc_partition", a, b, 4)
    _check(p.shape == (5, 2))

    # interval modes via generated calc wrappers
    for impl in ("basic", "adaptive", "rigorous"):
        y = calc_wrappers.arb_calc_partition_mode(a, b, 4, impl=impl, dps=50)
        _check(y.shape == (5, 2))
        _check(bool(jnp.all(y[..., 0] <= y[..., 1])))

    # acb function through interval API
    za = _box(-0.2, 0.1, -0.05, 0.05)
    zb = _box(0.7, 0.9, 0.1, 0.2)
    z = api.eval_interval("acb_calc_integrate", za, zb, mode="basic", dps=50)
    _check(z.shape == (4,))
    _check(bool(z[0] <= z[1]))
    _check(bool(z[2] <= z[3]))
