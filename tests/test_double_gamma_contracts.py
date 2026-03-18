from __future__ import annotations

import jax.numpy as jnp

from arbplusjax import double_gamma


def test_bdg_normalizeddoublegamma_is_normalized_at_q_over_2():
    beta = jnp.asarray(1.3, dtype=jnp.float64)
    q_half = 0.5 * (beta + 1.0 / beta)
    value = double_gamma.bdg_normalizeddoublegamma(q_half, beta, prec_bits=80)
    assert jnp.allclose(value, jnp.asarray(1.0 + 0.0j, dtype=jnp.complex128), rtol=1e-10, atol=1e-10)


def test_bdg_barnesgamma2_uses_beta_reciprocal_canonicalization():
    beta = jnp.asarray(1.3, dtype=jnp.float64)
    w = jnp.asarray(1.7, dtype=jnp.float64)
    direct = double_gamma.bdg_barnesgamma2(w, beta, prec_bits=80)
    reciprocal = double_gamma.bdg_barnesgamma2(w, 1.0 / beta, prec_bits=80)
    assert jnp.allclose(direct, reciprocal, rtol=1e-10, atol=1e-10)


def test_bdg_normalizeddoublegamma_is_constant_shift_of_barnesgamma2():
    beta = jnp.asarray(1.3, dtype=jnp.float64)
    w = jnp.asarray(1.7, dtype=jnp.float64)
    q_half = 0.5 * (beta + 1.0 / beta)
    expected = double_gamma.bdg_barnesgamma2(w, beta, prec_bits=80) / double_gamma.bdg_barnesgamma2(q_half, beta, prec_bits=80)
    actual = double_gamma.bdg_normalizeddoublegamma(w, beta, prec_bits=80)
    assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_bdg_shift_ratios_match_between_barnesgamma2_and_normalizeddoublegamma():
    beta = jnp.asarray(1.3, dtype=jnp.float64)
    w = jnp.asarray(1.7, dtype=jnp.float64)
    shift_b = beta
    shift_invb = 1.0 / beta

    raw_ratio_b = double_gamma.bdg_barnesgamma2(w + shift_b, beta, prec_bits=80) / double_gamma.bdg_barnesgamma2(w, beta, prec_bits=80)
    norm_ratio_b = double_gamma.bdg_normalizeddoublegamma(w + shift_b, beta, prec_bits=80) / double_gamma.bdg_normalizeddoublegamma(w, beta, prec_bits=80)
    raw_ratio_invb = double_gamma.bdg_barnesgamma2(w + shift_invb, beta, prec_bits=80) / double_gamma.bdg_barnesgamma2(w, beta, prec_bits=80)
    norm_ratio_invb = double_gamma.bdg_normalizeddoublegamma(w + shift_invb, beta, prec_bits=80) / double_gamma.bdg_normalizeddoublegamma(w, beta, prec_bits=80)

    assert jnp.allclose(raw_ratio_b, norm_ratio_b, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(raw_ratio_invb, norm_ratio_invb, rtol=1e-10, atol=1e-10)


def test_bdg_log_and_value_pairs_agree_for_core_family_members():
    beta = jnp.asarray(1.3, dtype=jnp.float64)
    w = jnp.asarray(1.7, dtype=jnp.float64)
    tau = jnp.asarray(0.8, dtype=jnp.float64)
    z = jnp.asarray(1.2, dtype=jnp.float64)

    gamma2 = double_gamma.bdg_barnesgamma2(w, beta, prec_bits=80)
    gamma2_log = double_gamma.bdg_log_barnesgamma2(w, beta, prec_bits=80)
    doublegamma = double_gamma.bdg_barnesdoublegamma(z, tau, prec_bits=80)
    doublegamma_log = double_gamma.bdg_log_barnesdoublegamma(z, tau, prec_bits=80)

    assert jnp.allclose(gamma2, jnp.exp(gamma2_log), rtol=1e-10, atol=1e-10)
    assert jnp.allclose(doublegamma, jnp.exp(doublegamma_log), rtol=1e-10, atol=1e-10)
