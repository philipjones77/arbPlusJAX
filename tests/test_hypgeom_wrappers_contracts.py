from __future__ import annotations

import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import double_interval as di
from arbplusjax import hypgeom_wrappers as hw


def _contains_interval(outer, inner) -> bool:
    return bool(jnp.all(outer[..., 0] <= inner[..., 0]) and jnp.all(outer[..., 1] >= inner[..., 1]))


def _contains_box(outer, inner) -> bool:
    return _contains_interval(acb_core.acb_real(outer), acb_core.acb_real(inner)) and _contains_interval(
        acb_core.acb_imag(outer), acb_core.acb_imag(inner)
    )


def test_hypgeom_wrappers_real_mode_dispatch_contracts() -> None:
    x = di.interval(jnp.float64(0.5), jnp.float64(0.75))
    a = di.interval(jnp.float64(0.75), jnp.float64(1.0))
    b = di.interval(jnp.float64(1.5), jnp.float64(1.75))
    c = di.interval(jnp.float64(2.0), jnp.float64(2.25))
    z = di.interval(jnp.float64(0.1), jnp.float64(0.2))

    gamma_basic = hw.arb_hypgeom_gamma_mode(x, impl="basic", prec_bits=80)
    gamma_rigorous = hw.arb_hypgeom_gamma_mode(x, impl="rigorous", prec_bits=80)
    gamma_adaptive = hw.arb_hypgeom_gamma_mode(x, impl="adaptive", prec_bits=80)
    gamma_baseline = hw.arb_hypgeom_gamma_mode(x, impl="baseline", prec_bits=80)
    onef1_basic = hw.arb_hypgeom_1f1_mode(a, b, z, impl="basic", prec_bits=80)
    gamma_lower = hw.arb_hypgeom_gamma_lower_mode(a, z, impl="rigorous", regularized=False, prec_bits=80)

    assert gamma_basic.shape == (2,)
    assert gamma_rigorous.shape == (2,)
    assert gamma_adaptive.shape == (2,)
    assert jnp.allclose(gamma_baseline, gamma_basic)
    assert _contains_interval(gamma_rigorous, gamma_basic)
    assert onef1_basic.shape == (2,)
    assert gamma_lower.shape == (2,)


def test_hypgeom_wrappers_complex_mode_dispatch_contracts() -> None:
    def box(re_lo, re_hi, im_lo, im_hi):
        return acb_core.acb_box(di.interval(jnp.float64(re_lo), jnp.float64(re_hi)), di.interval(jnp.float64(im_lo), jnp.float64(im_hi)))

    x = box(0.5, 0.75, 0.1, 0.2)
    a = box(0.75, 1.0, -0.1, 0.1)
    b = box(1.5, 1.75, 0.0, 0.2)
    c = box(2.0, 2.25, -0.15, 0.15)
    z = box(0.1, 0.2, -0.05, 0.05)

    gamma_basic = hw.acb_hypgeom_gamma_mode(x, impl="basic", prec_bits=80)
    gamma_rigorous = hw.acb_hypgeom_gamma_mode(x, impl="rigorous", prec_bits=80)
    twof1_basic = hw.acb_hypgeom_2f1_mode(a, b, c, z, impl="basic", prec_bits=80)
    gamma_upper = hw.acb_hypgeom_gamma_upper_mode(a, z, impl="adaptive", regularized=False, prec_bits=80)

    assert gamma_basic.shape == (4,)
    assert _contains_box(gamma_rigorous, gamma_basic)
    assert twof1_basic.shape == (4,)
    assert gamma_upper.shape == (4,)
