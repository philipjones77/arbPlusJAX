from __future__ import annotations

import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import core_wrappers as cw
from arbplusjax import double_interval as di


def _contains_interval(outer, inner) -> bool:
    return bool(jnp.all(outer[..., 0] <= inner[..., 0]) and jnp.all(outer[..., 1] >= inner[..., 1]))


def _contains_box(outer, inner) -> bool:
    return _contains_interval(acb_core.acb_real(outer), acb_core.acb_real(inner)) and _contains_interval(
        acb_core.acb_imag(outer), acb_core.acb_imag(inner)
    )


def test_core_wrappers_real_modes_expose_point_basic_and_adaptive_paths() -> None:
    x = di.interval(jnp.float64(1.0), jnp.float64(1.5))
    y = di.interval(jnp.float64(2.0), jnp.float64(2.5))

    basic = cw.arb_add_mode(x, y, impl="basic")
    rigorous = cw.arb_add_mode(x, y, impl="rigorous", prec_bits=80)
    baseline = cw.arb_add_mode(x, y, impl="baseline", prec_bits=80)
    adaptive = cw.arb_exp_mode(x, impl="adaptive", prec_bits=80)
    gamma_basic = cw.arb_gamma_mode(x, impl="basic", prec_bits=80)

    assert jnp.allclose(baseline, basic)
    assert rigorous.shape == basic.shape
    assert _contains_interval(adaptive, cw.arb_exp_mode(x, impl="basic", prec_bits=80))
    assert gamma_basic.shape == (2,)


def test_core_wrappers_complex_modes_preserve_box_structure_and_basic_enclosure() -> None:
    re = di.interval(jnp.float64(0.5), jnp.float64(0.75))
    im = di.interval(jnp.float64(0.1), jnp.float64(0.2))
    z = acb_core.acb_box(re, im)

    basic = cw.acb_exp_mode(z, impl="basic", prec_bits=80)
    rigorous = cw.acb_exp_mode(z, impl="rigorous", prec_bits=80)
    baseline = cw.acb_exp_mode(z, impl="baseline", prec_bits=80)
    adaptive = cw.acb_log_mode(z, impl="adaptive", prec_bits=80)

    assert basic.shape == (4,)
    assert jnp.array_equal(baseline, basic)
    assert _contains_box(rigorous, basic)
    assert adaptive.shape == (4,)
