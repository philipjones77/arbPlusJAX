from __future__ import annotations

import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import double_interval as di
from arbplusjax import modular_elliptic_wrappers as mw


def _contains_interval(outer, inner) -> bool:
    return bool(jnp.all(outer[..., 0] <= inner[..., 0]) and jnp.all(outer[..., 1] >= inner[..., 1]))


def _contains_box(outer, inner) -> bool:
    return _contains_interval(acb_core.acb_real(outer), acb_core.acb_real(inner)) and _contains_interval(
        acb_core.acb_imag(outer), acb_core.acb_imag(inner)
    )


def test_modular_elliptic_wrappers_mode_dispatch_contracts() -> None:
    tau = acb_core.acb_box(di.interval(jnp.float64(0.25), jnp.float64(0.3)), di.interval(jnp.float64(0.8), jnp.float64(0.9)))
    m = acb_core.acb_box(di.interval(jnp.float64(0.1), jnp.float64(0.2)), di.interval(jnp.float64(0.0), jnp.float64(0.05)))

    j_basic = mw.acb_modular_j_mode(tau, impl="basic", prec_bits=80)
    j_rigorous = mw.acb_modular_j_mode(tau, impl="rigorous", prec_bits=80)
    k_basic = mw.acb_elliptic_k_mode(m, impl="basic", prec_bits=80)
    e_adaptive = mw.acb_elliptic_e_mode(m, impl="adaptive", prec_bits=80)

    assert j_basic.shape == (4,)
    assert _contains_box(j_rigorous, j_basic)
    assert k_basic.shape == (4,)
    assert e_adaptive.shape == (4,)
