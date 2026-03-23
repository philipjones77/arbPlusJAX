from __future__ import annotations

import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import dft_wrappers as dw
from arbplusjax import double_interval as di


def _contains_interval(outer, inner) -> bool:
    return bool(jnp.all(outer[..., 0] <= inner[..., 0]) and jnp.all(outer[..., 1] >= inner[..., 1]))


def _contains_box(outer, inner) -> bool:
    return _contains_interval(acb_core.acb_real(outer), acb_core.acb_real(inner)) and _contains_interval(
        acb_core.acb_imag(outer), acb_core.acb_imag(inner)
    )


def test_dft_wrappers_complex_mode_dispatch_contracts() -> None:
    x = acb_core.acb_box(
        di.interval(jnp.asarray([1.0, 0.5]), jnp.asarray([1.0, 0.5])),
        di.interval(jnp.asarray([0.0, -0.25]), jnp.asarray([0.0, -0.25])),
    )

    basic = dw.acb_dft_mode(x, impl="basic", prec_bits=80)
    rigorous = dw.acb_dft_mode(x, impl="rigorous", prec_bits=80)
    inverse = dw.acb_dft_inverse_mode(x, impl="basic", prec_bits=80)
    bluestein = dw.acb_dft_bluestein_mode(x, impl="basic", prec_bits=80)

    assert basic.shape == (2, 4)
    assert rigorous.shape == (2, 4)
    assert _contains_box(rigorous, basic)
    assert inverse.shape == (2, 4)
    assert bluestein.shape == (2, 4)
