from __future__ import annotations

import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import dirichlet_wrappers as dw
from arbplusjax import double_interval as di


def _contains_interval(outer, inner) -> bool:
    return bool(jnp.all(outer[..., 0] <= inner[..., 0]) and jnp.all(outer[..., 1] >= inner[..., 1]))


def _contains_box(outer, inner) -> bool:
    return _contains_interval(acb_core.acb_real(outer), acb_core.acb_real(inner)) and _contains_interval(
        acb_core.acb_imag(outer), acb_core.acb_imag(inner)
    )


def test_dirichlet_wrappers_real_mode_dispatch_contracts() -> None:
    s = di.interval(jnp.float64(2.0), jnp.float64(2.25))

    zeta_basic = dw.dirichlet_zeta_mode(s, impl="basic", n_terms=32, prec_bits=80)
    zeta_rigorous = dw.dirichlet_zeta_mode(s, impl="rigorous", n_terms=32, prec_bits=80)
    eta_adaptive = dw.dirichlet_eta_mode(s, impl="adaptive", n_terms=32, prec_bits=80)

    assert zeta_basic.shape == (2,)
    assert _contains_interval(zeta_rigorous, zeta_basic)
    assert eta_adaptive.shape == (2,)


def test_dirichlet_wrappers_complex_mode_dispatch_contracts() -> None:
    s = acb_core.acb_box(di.interval(jnp.float64(2.0), jnp.float64(2.25)), di.interval(jnp.float64(0.1), jnp.float64(0.2)))

    zeta_basic = dw.acb_dirichlet_zeta_mode(s, impl="basic", n_terms=32, prec_bits=80)
    eta_adaptive = dw.acb_dirichlet_eta_mode(s, impl="adaptive", n_terms=32, prec_bits=80)

    assert zeta_basic.shape == (4,)
    assert eta_adaptive.shape == (4,)
    assert _contains_box(eta_adaptive, dw.acb_dirichlet_eta_mode(s, impl="basic", n_terms=32, prec_bits=80))
