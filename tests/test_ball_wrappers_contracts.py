from __future__ import annotations

import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import ball_wrappers as bw
from arbplusjax import double_interval as di


def _contains_interval(outer, inner) -> bool:
    return bool(jnp.all(outer[..., 0] <= inner[..., 0]) and jnp.all(outer[..., 1] >= inner[..., 1]))


def _contains_box(outer, inner) -> bool:
    return _contains_interval(acb_core.acb_real(outer), acb_core.acb_real(inner)) and _contains_interval(
        acb_core.acb_imag(outer), acb_core.acb_imag(inner)
    )


def test_ball_wrappers_real_surfaces_return_interval_containers() -> None:
    x = di.interval(jnp.float64(0.5), jnp.float64(0.75))

    exp_rigorous = bw.arb_ball_exp(x, prec_bits=80)
    exp_adaptive = bw.arb_ball_exp_adaptive(x, prec_bits=80, samples=7)
    gamma_rigorous = bw.arb_ball_gamma(x, prec_bits=80)

    assert exp_rigorous.shape == (2,)
    assert exp_adaptive.shape == (2,)
    assert gamma_rigorous.shape == (2,)
    assert _contains_interval(exp_adaptive, bw.arb_ball_exp(di.interval(di.midpoint(x), di.midpoint(x)), prec_bits=80))


def test_ball_wrappers_complex_surfaces_return_box_containers() -> None:
    z = acb_core.acb_box(di.interval(jnp.float64(0.5), jnp.float64(0.75)), di.interval(jnp.float64(0.1), jnp.float64(0.2)))

    exp_rigorous = bw.acb_ball_exp(z, prec_bits=80)
    log_adaptive = bw.acb_ball_log_adaptive(z, prec_bits=80, samples=7)

    assert exp_rigorous.shape == (4,)
    assert log_adaptive.shape == (4,)
    assert _contains_box(log_adaptive, bw.acb_ball_log(acb_core.acb_box(di.interval(0.6, 0.6), di.interval(0.15, 0.15)), prec_bits=80))
