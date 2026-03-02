import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import core_wrappers
from arbplusjax import double_interval as di
from arbplusjax import point_wrappers

from tests._test_checks import _check


def _box(re: float, im: float):
    re_v = jnp.float64(re)
    im_v = jnp.float64(im)
    return acb_core.acb_box(di.interval(re_v, re_v), di.interval(im_v, im_v))


def _contains_box(outer, inner) -> bool:
    re_ok = di.contains(acb_core.acb_real(outer), acb_core.acb_real(inner)).all()
    im_ok = di.contains(acb_core.acb_imag(outer), acb_core.acb_imag(inner)).all()
    return bool(re_ok and im_ok)


def test_acb_top10_point_basic_rigorous_adaptive():
    x = _box(0.3, 0.2)
    y = _box(1.7, -0.1)
    z = _box(-0.2, 0.4)
    w = _box(0.2, -0.1)

    abs_basic = core_wrappers.acb_abs_mode(x, impl="basic", prec_bits=80)
    abs_rig = core_wrappers.acb_abs_mode(x, impl="rigorous", prec_bits=80)
    abs_adapt = core_wrappers.acb_abs_mode(x, impl="adaptive", prec_bits=80)
    _check(bool(di.contains(abs_rig, abs_basic).all()))
    _check(bool(di.contains(abs_adapt, abs_basic).all()))
    _check(bool(jnp.allclose(point_wrappers.acb_abs_point(acb_core.acb_midpoint(x)), di.midpoint(abs_basic))))

    cases = [
        (point_wrappers.acb_add_point, core_wrappers.acb_add_mode, (x, y)),
        (point_wrappers.acb_sub_point, core_wrappers.acb_sub_mode, (x, y)),
        (point_wrappers.acb_mul_point, core_wrappers.acb_mul_mode, (x, y)),
        (point_wrappers.acb_div_point, core_wrappers.acb_div_mode, (x, y)),
        (point_wrappers.acb_inv_point, core_wrappers.acb_inv_mode, (y,)),
        (point_wrappers.acb_fma_point, core_wrappers.acb_fma_mode, (x, y, z)),
        (point_wrappers.acb_log1p_point, core_wrappers.acb_log1p_mode, (w,)),
        (point_wrappers.acb_expm1_point, core_wrappers.acb_expm1_mode, (x,)),
    ]

    for point_fn, mode_fn, args in cases:
        basic = mode_fn(*args, impl="basic", prec_bits=80)
        rig = mode_fn(*args, impl="rigorous", prec_bits=80)
        adapt = mode_fn(*args, impl="adaptive", prec_bits=80)
        _check(_contains_box(rig, basic))
        _check(_contains_box(adapt, basic))

        mids = tuple(acb_core.acb_midpoint(a) for a in args)
        p = point_fn(*mids)
        m = acb_core.acb_midpoint(basic)
        _check(bool(jnp.allclose(p, m)))

    s_basic, c_basic = core_wrappers.acb_sin_cos_mode(x, impl="basic", prec_bits=80)
    s_rig, c_rig = core_wrappers.acb_sin_cos_mode(x, impl="rigorous", prec_bits=80)
    s_adapt, c_adapt = core_wrappers.acb_sin_cos_mode(x, impl="adaptive", prec_bits=80)
    _check(_contains_box(s_rig, s_basic))
    _check(_contains_box(c_rig, c_basic))
    _check(_contains_box(s_adapt, s_basic))
    _check(_contains_box(c_adapt, c_basic))

    ps, pc = point_wrappers.acb_sin_cos_point(acb_core.acb_midpoint(x))
    _check(bool(jnp.allclose(ps, acb_core.acb_midpoint(s_basic))))
    _check(bool(jnp.allclose(pc, acb_core.acb_midpoint(c_basic))))
