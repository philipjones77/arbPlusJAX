from __future__ import annotations

import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import double_interval as di
from arbplusjax import poly_wrappers as pw


def _contains_interval(outer, inner) -> bool:
    return bool(jnp.all(outer[..., 0] <= inner[..., 0]) and jnp.all(outer[..., 1] >= inner[..., 1]))


def _contains_box(outer, inner) -> bool:
    return _contains_interval(acb_core.acb_real(outer), acb_core.acb_real(inner)) and _contains_interval(
        acb_core.acb_imag(outer), acb_core.acb_imag(inner)
    )


def test_poly_wrappers_real_and_integer_coeff_surfaces() -> None:
    coeffs = di.interval(
        jnp.asarray([0.5, -1.0, 0.25, 2.0], dtype=jnp.float64),
        jnp.asarray([0.6, -0.9, 0.35, 2.1], dtype=jnp.float64),
    )
    x = di.interval(jnp.float64(-0.2), jnp.float64(0.3))
    int_coeffs = di.interval(
        jnp.asarray([1.0, -2.0, 3.0, -1.0], dtype=jnp.float64),
        jnp.asarray([1.0, -2.0, 3.0, -1.0], dtype=jnp.float64),
    )

    basic = pw.arb_poly_eval_cubic_mode(coeffs, x, impl="basic", prec_bits=80)
    baseline = pw.arb_poly_eval_cubic_mode(coeffs, x, impl="baseline", prec_bits=80)
    rigorous = pw.arb_poly_eval_cubic_mode(coeffs, x, impl="rigorous", prec_bits=80)
    adaptive = pw.arb_poly_eval_cubic_mode(coeffs, x, impl="adaptive", prec_bits=80)
    fmpz_basic = pw.arb_fmpz_poly_eval_cubic_mode(int_coeffs, x, impl="basic", prec_bits=80)

    assert basic.shape == (2,)
    assert jnp.allclose(basic, baseline)
    assert _contains_interval(rigorous, basic)
    assert adaptive.shape == (2,)
    assert fmpz_basic.shape == (2,)


def test_poly_wrappers_complex_surface_contracts() -> None:
    coeffs = acb_core.acb_box(
        di.interval(jnp.asarray([0.5, -1.0, 0.25, 2.0], dtype=jnp.float64), jnp.asarray([0.6, -0.9, 0.35, 2.1], dtype=jnp.float64)),
        di.interval(jnp.asarray([0.1, -0.2, 0.0, 0.3], dtype=jnp.float64), jnp.asarray([0.2, -0.1, 0.1, 0.4], dtype=jnp.float64)),
    )
    z = acb_core.acb_box(di.interval(jnp.float64(-0.2), jnp.float64(0.3)), di.interval(jnp.float64(0.1), jnp.float64(0.2)))

    basic = pw.acb_poly_eval_cubic_mode(coeffs, z, impl="basic", prec_bits=80)
    rigorous = pw.acb_poly_eval_cubic_mode(coeffs, z, impl="rigorous", prec_bits=80)
    adaptive = pw.acb_poly_eval_cubic_mode(coeffs, z, impl="adaptive", prec_bits=80)

    assert basic.shape == (4,)
    assert _contains_box(rigorous, basic)
    assert adaptive.shape == (4,)
