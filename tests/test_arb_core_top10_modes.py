import jax.numpy as jnp

from arbplusjax import core_wrappers
from arbplusjax import double_interval as di
from arbplusjax import point_wrappers

from tests._test_checks import _check


def _iv(x: float):
    v = jnp.float64(x)
    return di.interval(v, v)


def _contains(a, b) -> bool:
    return bool(di.contains(a, b).all())


def test_top10_point_basic_rigorous_adaptive():
    x = _iv(0.3)
    y = _iv(1.7)
    z = _iv(-0.2)
    w = _iv(0.2)

    cases = [
        ("arb_abs", point_wrappers.arb_abs_point, core_wrappers.arb_abs_mode, (x,)),
        ("arb_add", point_wrappers.arb_add_point, core_wrappers.arb_add_mode, (x, y)),
        ("arb_sub", point_wrappers.arb_sub_point, core_wrappers.arb_sub_mode, (x, y)),
        ("arb_mul", point_wrappers.arb_mul_point, core_wrappers.arb_mul_mode, (x, y)),
        ("arb_div", point_wrappers.arb_div_point, core_wrappers.arb_div_mode, (x, y)),
        ("arb_inv", point_wrappers.arb_inv_point, core_wrappers.arb_inv_mode, (y,)),
        ("arb_fma", point_wrappers.arb_fma_point, core_wrappers.arb_fma_mode, (x, y, z)),
        ("arb_log1p", point_wrappers.arb_log1p_point, core_wrappers.arb_log1p_mode, (w,)),
        ("arb_expm1", point_wrappers.arb_expm1_point, core_wrappers.arb_expm1_mode, (x,)),
    ]

    for _, point_fn, mode_fn, args in cases:
        basic = mode_fn(*args, impl="basic", prec_bits=80)
        rig = mode_fn(*args, impl="rigorous", prec_bits=80)
        adapt = mode_fn(*args, impl="adaptive", prec_bits=80)
        _check(_contains(rig, basic))
        _check(_contains(adapt, basic))

        mids = tuple(di.midpoint(a) for a in args)
        p = point_fn(*mids)
        m = di.midpoint(basic)
        _check(bool(jnp.allclose(p, m)))

    s_basic, c_basic = core_wrappers.arb_sin_cos_mode(x, impl="basic", prec_bits=80)
    s_rig, c_rig = core_wrappers.arb_sin_cos_mode(x, impl="rigorous", prec_bits=80)
    s_adapt, c_adapt = core_wrappers.arb_sin_cos_mode(x, impl="adaptive", prec_bits=80)
    _check(_contains(s_rig, s_basic))
    _check(_contains(c_rig, c_basic))
    _check(_contains(s_adapt, s_basic))
    _check(_contains(c_adapt, c_basic))

    ps, pc = point_wrappers.arb_sin_cos_point(di.midpoint(x))
    _check(bool(jnp.allclose(ps, di.midpoint(s_basic))))
    _check(bool(jnp.allclose(pc, di.midpoint(c_basic))))
