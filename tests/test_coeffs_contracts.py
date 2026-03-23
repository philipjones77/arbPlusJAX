from __future__ import annotations

import jax.numpy as jnp

from arbplusjax import coeffs


def test_coeff_tables_have_expected_shape_dtype_and_anchor_values() -> None:
    assert coeffs.LANCZOS.shape == (9,)
    assert coeffs.STIRLING_COEFFS.shape == (8,)
    assert coeffs.LANCZOS.dtype == jnp.float64
    assert coeffs.STIRLING_COEFFS.dtype == jnp.float64

    assert jnp.isclose(coeffs.LANCZOS[0], jnp.float64(0.9999999999998099))
    assert jnp.isclose(coeffs.LANCZOS[1], jnp.float64(676.5203681218851))
    assert jnp.isclose(coeffs.STIRLING_COEFFS[0], jnp.float64(1.0 / 12.0))
    assert jnp.isclose(coeffs.STIRLING_COEFFS[1], jnp.float64(-1.0 / 360.0))


def test_coeff_tables_match_expected_sign_patterns() -> None:
    lanczos_tail = jnp.sign(coeffs.LANCZOS[1:])
    stirling_signs = jnp.sign(coeffs.STIRLING_COEFFS)

    assert jnp.array_equal(lanczos_tail, jnp.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0]))
    assert jnp.array_equal(stirling_signs, jnp.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]))
