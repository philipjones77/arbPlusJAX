from __future__ import annotations

import jax.numpy as jnp
import pytest

from arbplusjax import acb_core
from arbplusjax import baseline_wrappers as bw
from arbplusjax import double_interval as di


def _contains_interval(outer, inner) -> bool:
    return bool(jnp.all(outer[..., 0] <= inner[..., 0]) and jnp.all(outer[..., 1] >= inner[..., 1]))


def _contains_box(outer, inner) -> bool:
    return _contains_interval(acb_core.acb_real(outer), acb_core.acb_real(inner)) and _contains_interval(
        acb_core.acb_imag(outer), acb_core.acb_imag(inner)
    )


def test_baseline_wrappers_real_mode_dispatch_contracts() -> None:
    x = di.interval(jnp.float64(0.5), jnp.float64(0.75))
    y = di.interval(jnp.float64(1.0), jnp.float64(1.25))

    exp_basic = bw.arb_exp_mp(x, mode="basic", prec_bits=80)
    exp_baseline = bw.arb_exp_mp(x, mode="baseline", prec_bits=80)
    exp_adaptive = bw.arb_exp_mp(x, mode="adaptive", prec_bits=80)
    add_basic = bw.arb_add_mp(x, y, mode="basic", prec_bits=80)
    gamma_rigorous = bw.arb_gamma_mp(x, mode="rigorous", prec_bits=80)

    assert jnp.allclose(exp_basic, exp_baseline)
    assert exp_adaptive.shape == exp_basic.shape
    assert add_basic.shape == (2,)
    assert gamma_rigorous.shape == (2,)


def test_baseline_wrappers_complex_mode_dispatch_contracts() -> None:
    z = acb_core.acb_box(di.interval(jnp.float64(0.5), jnp.float64(0.75)), di.interval(jnp.float64(0.1), jnp.float64(0.2)))

    exp_basic = bw.acb_exp_mp(z, mode="basic", prec_bits=80)
    exp_rigorous = bw.acb_exp_mp(z, mode="rigorous", prec_bits=80)
    gamma_adaptive = bw.acb_gamma_mp(z, mode="adaptive", prec_bits=80)

    assert exp_basic.shape == (4,)
    assert _contains_box(exp_rigorous, exp_basic)
    assert gamma_adaptive.shape == (4,)


def test_baseline_wrappers_reject_invalid_mode() -> None:
    x = di.interval(jnp.float64(0.5), jnp.float64(0.75))
    with pytest.raises(ValueError):
        bw.arb_exp_mp(x, mode="bad", prec_bits=80)
