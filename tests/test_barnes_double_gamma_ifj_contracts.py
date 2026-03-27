from __future__ import annotations

import pytest
import jax.numpy as jnp

from arbplusjax import double_gamma


def test_ifj_scalar_diagnostics_expose_provider_contract_fields() -> None:
    value, diagnostics = double_gamma.ifj_barnesdoublegamma(
        0.2 + 0.05j,
        1.0,
        dps=60,
        max_m_cap=96,
        return_diagnostics=True,
    )

    assert jnp.isfinite(jnp.real(value))
    assert jnp.isfinite(jnp.imag(value))
    assert diagnostics.dps == 60
    assert diagnostics.tau == 1.0
    assert diagnostics.m_base >= 64
    assert diagnostics.m_used <= diagnostics.max_m_cap
    assert diagnostics.max_m_cap == 96
    assert diagnostics.m_capped is True
    assert diagnostics.n_shift >= 1
    assert diagnostics.richardson_levels == 3


def test_ifj_vectorized_values_preserve_shape_and_anchor_normalization() -> None:
    zs = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    values = double_gamma.ifj_barnesdoublegamma(zs, 1.0, dps=60)

    assert values.shape == zs.shape
    assert jnp.all(jnp.isfinite(jnp.real(values)))
    assert jnp.all(jnp.isfinite(jnp.imag(values)))
    assert abs(complex(values[0]) - 1.0) < 2e-3
    assert abs(complex(values[1]) - 1.0) < 2e-3
    assert abs(complex(values[2]) - 1.0) < 2e-3
    assert abs(complex(values[3]) - 2.0) < 2e-5


def test_ifj_vectorized_diagnostics_are_rejected_explicitly() -> None:
    zs = jnp.asarray([1.0, 2.0], dtype=jnp.float64)

    with pytest.raises(ValueError, match="scalar z"):
        double_gamma.ifj_barnesdoublegamma(zs, 1.0, dps=60, return_diagnostics=True)


def test_ifj_log_and_value_diagnostics_agree_on_truncation_choices() -> None:
    _, log_diag = double_gamma.ifj_log_barnesdoublegamma(
        0.3 + 0.1j,
        0.5,
        dps=50,
        max_m_cap=128,
        return_diagnostics=True,
    )
    _, value_diag = double_gamma.ifj_barnesdoublegamma(
        0.3 + 0.1j,
        0.5,
        dps=50,
        max_m_cap=128,
        return_diagnostics=True,
    )

    assert log_diag.tau == value_diag.tau
    assert log_diag.m_base == value_diag.m_base
    assert log_diag.m_used == value_diag.m_used
    assert log_diag.m_capped == value_diag.m_capped
    assert log_diag.n_shift == value_diag.n_shift

