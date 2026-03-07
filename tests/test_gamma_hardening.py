from __future__ import annotations

import jax
import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import arb_core
from arbplusjax import ball_wrappers
from arbplusjax import double_gamma
from arbplusjax import double_interval as di

from tests._test_checks import _check


def _contains_all(outer: jax.Array, inner: jax.Array) -> bool:
    return bool(jnp.all(di.contains(outer, inner)))


def _box_contains_all(outer: jax.Array, inner: jax.Array) -> bool:
    outer = acb_core.as_acb_box(outer)
    inner = acb_core.as_acb_box(inner)
    real_ok = jnp.all(di.contains(acb_core.acb_real(outer), acb_core.acb_real(inner)))
    imag_ok = jnp.all(di.contains(acb_core.acb_imag(outer), acb_core.acb_imag(inner)))
    return bool(real_ok & imag_ok)


def test_real_gamma_family_ball_wrappers_contain_basic():
    x = di.interval(jnp.float64(1.15), jnp.float64(1.25))
    y = di.interval(jnp.float64(1.4), jnp.float64(1.5))

    gamma_basic = arb_core.arb_gamma_prec(x, prec_bits=80)
    gamma_rig = ball_wrappers.arb_ball_gamma(x, prec_bits=80)
    gamma_adp = ball_wrappers.arb_ball_gamma_adaptive(x, prec_bits=80)
    _check(_contains_all(gamma_rig, gamma_basic))
    _check(bool(jnp.all(jnp.isfinite(gamma_adp))))
    _check(bool(jnp.all(di.ubound_radius(gamma_adp) <= di.ubound_radius(gamma_rig))))

    lgamma_basic = arb_core.arb_lgamma_prec(x, prec_bits=80)
    lgamma_rig = ball_wrappers.arb_ball_lgamma(x, prec_bits=80)
    lgamma_adp = ball_wrappers.arb_ball_lgamma_adaptive(x, prec_bits=80)
    _check(_contains_all(lgamma_rig, lgamma_basic))
    _check(bool(jnp.all(jnp.isfinite(lgamma_adp))))
    _check(bool(jnp.all(di.ubound_radius(lgamma_adp) <= di.ubound_radius(lgamma_rig))))

    rgamma_basic = arb_core.arb_rgamma_prec(x, prec_bits=80)
    rgamma_rig = ball_wrappers.arb_ball_rgamma(x, prec_bits=80)
    rgamma_adp = ball_wrappers.arb_ball_rgamma_adaptive(x, prec_bits=80)
    _check(_contains_all(rgamma_rig, rgamma_basic))
    _check(bool(jnp.all(jnp.isfinite(rgamma_adp))))
    _check(bool(jnp.all(di.ubound_radius(rgamma_adp) <= di.ubound_radius(rgamma_rig))))

    pow_basic = arb_core.arb_pow_prec(x, y, prec_bits=80)
    pow_rig = ball_wrappers.arb_ball_pow(x, y, prec_bits=80)
    pow_adp = ball_wrappers.arb_ball_pow_adaptive(x, y, prec_bits=80)
    _check(_contains_all(pow_rig, pow_basic))
    _check(bool(jnp.all(jnp.isfinite(pow_adp))))
    _check(_contains_all(pow_adp, pow_basic))

    root_basic = arb_core.arb_root_ui_prec(x, 3, prec_bits=80)
    root_rig = ball_wrappers.arb_ball_root_ui(x, 3, prec_bits=80)
    root_adp = ball_wrappers.arb_ball_root_ui_adaptive(x, 3, prec_bits=80)
    _check(_contains_all(root_rig, root_basic))
    _check(bool(jnp.all(jnp.isfinite(root_adp))))
    _check(bool(jnp.all(di.ubound_radius(root_adp) <= di.ubound_radius(root_rig))))


def test_bdg_interval_modes_contain_basic():
    w = di.interval(jnp.float64(1.2), jnp.float64(1.25))
    beta = di.interval(jnp.float64(1.35), jnp.float64(1.4))

    log_basic = double_gamma.bdg_interval_log_barnesgamma2_mode(w, beta, impl="basic", prec_bits=80)
    log_rig = double_gamma.bdg_interval_log_barnesgamma2_mode(w, beta, impl="rigorous", prec_bits=80)
    log_adp = double_gamma.bdg_interval_log_barnesgamma2_mode(w, beta, impl="adaptive", prec_bits=80)
    _check(_contains_all(log_rig, log_basic))
    _check(_contains_all(log_adp, log_basic))

    norm_basic = double_gamma.bdg_interval_normalizeddoublegamma_mode(w, beta, impl="basic", prec_bits=80)
    norm_rig = double_gamma.bdg_interval_normalizeddoublegamma_mode(w, beta, impl="rigorous", prec_bits=80)
    norm_adp = double_gamma.bdg_interval_normalizeddoublegamma_mode(w, beta, impl="adaptive", prec_bits=80)
    _check(_contains_all(norm_rig, norm_basic))
    _check(_contains_all(norm_adp, norm_basic))


def test_bdg_complex_modes_contain_basic():
    w = acb_core.acb_box(
        di.interval(jnp.float64(1.2), jnp.float64(1.23)),
        di.interval(jnp.float64(0.05), jnp.float64(0.08)),
    )
    beta = acb_core.acb_box(
        di.interval(jnp.float64(1.35), jnp.float64(1.38)),
        di.interval(jnp.float64(0.02), jnp.float64(0.04)),
    )

    basic = double_gamma.bdg_complex_barnesgamma2_mode(w, beta, impl="basic", prec_bits=80)
    rig = double_gamma.bdg_complex_barnesgamma2_mode(w, beta, impl="rigorous", prec_bits=80)
    adp = double_gamma.bdg_complex_barnesgamma2_mode(w, beta, impl="adaptive", prec_bits=80)
    _check(_box_contains_all(rig, basic))
    _check(_box_contains_all(adp, basic))

    basic_log = double_gamma.bdg_complex_log_normalizeddoublegamma_mode(w, beta, impl="basic", prec_bits=80)
    rig_log = double_gamma.bdg_complex_log_normalizeddoublegamma_mode(w, beta, impl="rigorous", prec_bits=80)
    adp_log = double_gamma.bdg_complex_log_normalizeddoublegamma_mode(w, beta, impl="adaptive", prec_bits=80)
    _check(_box_contains_all(rig_log, basic_log))
    _check(_box_contains_all(adp_log, basic_log))


def test_bdg_point_and_basic_respect_float32_complex64_dtypes():
    w32 = jnp.asarray(1.2, dtype=jnp.float32)
    beta32 = jnp.asarray(1.35, dtype=jnp.float32)
    log_pt = double_gamma.bdg_log_barnesgamma2(w32, beta32, prec_bits=53)
    val_pt = double_gamma.bdg_barnesgamma2(w32, beta32, prec_bits=53)
    _check(log_pt.dtype == jnp.dtype(jnp.complex64))
    _check(val_pt.dtype == jnp.dtype(jnp.complex64))

    wi32 = di.interval(jnp.float32(1.2), jnp.float32(1.25))
    bi32 = di.interval(jnp.float32(1.35), jnp.float32(1.4))
    out_basic = double_gamma.bdg_interval_barnesgamma2_mode(wi32, bi32, impl="basic", prec_bits=53)
    out_log_basic = double_gamma.bdg_interval_log_barnesgamma2_mode(wi32, bi32, impl="basic", prec_bits=53)
    _check(out_basic.dtype == jnp.dtype(jnp.float32))
    _check(out_log_basic.dtype == jnp.dtype(jnp.float32))

    wc64 = acb_core.acb_box(
        di.interval(jnp.float32(1.2), jnp.float32(1.23)),
        di.interval(jnp.float32(0.05), jnp.float32(0.08)),
    )
    bc64 = acb_core.acb_box(
        di.interval(jnp.float32(1.35), jnp.float32(1.38)),
        di.interval(jnp.float32(0.02), jnp.float32(0.04)),
    )
    out_c_basic = double_gamma.bdg_complex_barnesgamma2_mode(wc64, bc64, impl="basic", prec_bits=53)
    out_c_log_basic = double_gamma.bdg_complex_log_normalizeddoublegamma_mode(wc64, bc64, impl="basic", prec_bits=53)
    _check(out_c_basic.dtype == jnp.dtype(jnp.float32))
    _check(out_c_log_basic.dtype == jnp.dtype(jnp.float32))
