from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from arbplusjax import dft
from arbplusjax import dft_wrappers
from arbplusjax import acb_core
from arbplusjax import double_interval as di

from tests._test_checks import _check


def _acb_vec(n: int):
    t = jnp.arange(n, dtype=jnp.float64)
    re = di.interval(0.1 * t, 0.1 * t + 0.01)
    im = di.interval(-0.05 * t, -0.05 * t + 0.01)
    return acb_core.acb_box(re, im)


def _contains_box(outer, inner) -> bool:
    re_ok = di.contains(acb_core.acb_real(outer), acb_core.acb_real(inner)).all()
    im_ok = di.contains(acb_core.acb_imag(outer), acb_core.acb_imag(inner)).all()
    return bool(re_ok and im_ok)


def test_acb_dft_missing_api_equivalences():
    x = _acb_vec(8)
    y = _acb_vec(8) * jnp.array([1.0, 1.0, -1.0, -1.0], dtype=jnp.float64)
    cyc = (2, 4)

    pairs = [
        ("acb_dft_bluestein", lambda: dft.acb_dft(x)),
        ("acb_dft_bluestein_precomp", lambda: dft.acb_dft(x)),
        ("acb_dft_convol", lambda: dft.acb_convol_circular(x, y)),
        ("acb_dft_convol_dft", lambda: dft.acb_convol_circular_dft(x, y)),
        ("acb_dft_convol_mullow", lambda: dft.acb_convol_circular_naive(x, y)),
        ("acb_dft_convol_naive", lambda: dft.acb_convol_circular_naive(x, y)),
        ("acb_dft_convol_rad2", lambda: dft.acb_convol_circular_rad2(x, y)),
        ("acb_dft_convol_rad2_precomp", lambda: dft.acb_convol_circular_rad2(x, y)),
        ("acb_dft_crt", lambda: dft.acb_dft_prod(x, cyc)),
        ("acb_dft_crt_precomp", lambda: dft.acb_dft_prod(x, cyc)),
        ("acb_dft_cyc", lambda: dft.acb_dft_prod(x, cyc)),
        ("acb_dft_cyc_precomp", lambda: dft.acb_dft_prod(x, cyc)),
        ("acb_dft_inverse", lambda: dft.acb_idft(x)),
        ("acb_dft_inverse_precomp", lambda: dft.acb_idft(x)),
        ("acb_dft_inverse_rad2_precomp", lambda: dft.acb_idft_rad2(x)),
        ("acb_dft_inverse_rad2_precomp_inplace", lambda: dft.acb_idft_rad2(x)),
        ("acb_dft_naive_precomp", lambda: dft.acb_dft_naive(x, inverse=False)),
        ("acb_dft_precomp", lambda: dft.acb_dft(x)),
        ("acb_dft_prod_precomp", lambda: dft.acb_dft_prod(x, cyc)),
        ("acb_dft_rad2_inplace", lambda: dft.acb_dft_rad2(x)),
        ("acb_dft_rad2_inplace_threaded", lambda: dft.acb_dft_rad2(x)),
        ("acb_dft_rad2_precomp", lambda: dft.acb_dft_rad2(x)),
        ("acb_dft_rad2_precomp_inplace", lambda: dft.acb_dft_rad2(x)),
        ("acb_dft_rad2_precomp_inplace_threaded", lambda: dft.acb_dft_rad2(x)),
        ("acb_dft_step", lambda: dft.acb_dft(x)),
    ]

    for name, ref_fn in pairs:
        fn = getattr(dft, name)
        if "convol" in name:
            got = fn(x, y)
        elif name in ("acb_dft_crt", "acb_dft_cyc", "acb_dft_prod_precomp"):
            got = fn(x, cyc)
        elif name in ("acb_dft_crt_precomp", "acb_dft_cyc_precomp"):
            got = fn(x, cyc)
        elif name == "acb_dft_naive_precomp":
            got = fn(x, False)
        elif "precomp" in name:
            got = fn(x)
        else:
            got = fn(x)
        ref = ref_fn()
        _check(bool(jnp.allclose(got, ref)))


def test_acb_dft_missing_mode_wrappers():
    x = _acb_vec(8)
    y = _acb_vec(8) * jnp.array([1.0, 1.0, -1.0, -1.0], dtype=jnp.float64)
    cyc = (2, 4)

    mode_cases = [
        ("acb_dft_bluestein_mode", (x,)),
        ("acb_dft_bluestein_precomp_mode", (x,)),
        ("acb_dft_convol_mode", (x, y)),
        ("acb_dft_convol_dft_mode", (x, y)),
        ("acb_dft_convol_mullow_mode", (x, y)),
        ("acb_dft_convol_naive_mode", (x, y)),
        ("acb_dft_convol_rad2_mode", (x, y)),
        ("acb_dft_convol_rad2_precomp_mode", (x, y)),
        ("acb_dft_crt_mode", (x, cyc)),
        ("acb_dft_crt_precomp_mode", (x, cyc)),
        ("acb_dft_cyc_mode", (x, cyc)),
        ("acb_dft_cyc_precomp_mode", (x, cyc)),
        ("acb_dft_inverse_mode", (x,)),
        ("acb_dft_inverse_precomp_mode", (x,)),
        ("acb_dft_inverse_rad2_precomp_mode", (x,)),
        ("acb_dft_inverse_rad2_precomp_inplace_mode", (x,)),
        ("acb_dft_naive_precomp_mode", (x, False)),
        ("acb_dft_precomp_mode", (x,)),
        ("acb_dft_prod_precomp_mode", (x, cyc)),
        ("acb_dft_rad2_inplace_mode", (x,)),
        ("acb_dft_rad2_inplace_threaded_mode", (x,)),
        ("acb_dft_rad2_precomp_mode", (x,)),
        ("acb_dft_rad2_precomp_inplace_mode", (x,)),
        ("acb_dft_rad2_precomp_inplace_threaded_mode", (x,)),
        ("acb_dft_step_mode", (x,)),
    ]

    for name, args in mode_cases:
        fn = getattr(dft_wrappers, name)
        basic = fn(*args, impl="basic", prec_bits=80)
        rig = fn(*args, impl="rigorous", prec_bits=80)
        adapt = fn(*args, impl="adaptive", prec_bits=80)
        _check(_contains_box(rig, basic), f"{name} rigorous contains basic")
        _check(_contains_box(adapt, basic), f"{name} adaptive contains basic")


def test_acb_prime_length_precomp_paths_match_main_fft():
    x = _acb_vec(11)
    precomp = dft.make_dft_precomp(11)
    ref = dft.acb_dft(x)
    got_bluestein = dft.acb_dft_bluestein_precomp(x, precomp=precomp)
    got_main = dft.acb_dft_precomp(x, precomp=precomp)
    got_rad2 = dft.acb_dft_rad2_precomp(x, precomp=precomp)
    np.testing.assert_allclose(np.asarray(got_bluestein), np.asarray(ref), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(got_main), np.asarray(ref), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(got_rad2), np.asarray(ref), rtol=1e-12, atol=1e-12)
