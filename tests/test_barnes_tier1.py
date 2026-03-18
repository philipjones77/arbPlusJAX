from __future__ import annotations

import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import barnesg
from arbplusjax import double_gamma
from arbplusjax import double_interval as di


def _exact_box(z: complex) -> jnp.ndarray:
    zz = jnp.asarray(z, dtype=jnp.complex128)
    return acb_core.acb_box(
        di.interval(jnp.real(zz), jnp.real(zz)),
        di.interval(jnp.imag(zz), jnp.imag(zz)),
    )


def _to_complex(x) -> complex:
    return complex(jnp.asarray(x))


def test_ifj_double_gamma_tau1_matches_small_integer_values():
    tol = 2e-3
    assert abs(_to_complex(double_gamma.ifj_barnesdoublegamma(1.0, 1.0)) - 1.0) < tol
    assert abs(_to_complex(double_gamma.ifj_barnesdoublegamma(2.0, 1.0)) - 1.0) < tol
    assert abs(_to_complex(double_gamma.ifj_barnesdoublegamma(3.0, 1.0)) - 1.0) < tol
    assert abs(_to_complex(double_gamma.ifj_barnesdoublegamma(4.0, 1.0, dps=60)) - 2.0) < 2e-5


def test_ifj_double_gamma_shift_equation_tau1_and_nonunit_tau():
    z1 = 2.3 + 0.1j
    lhs1 = double_gamma.ifj_barnesdoublegamma(z1 + 1.0, 1.0, dps=50) / double_gamma.ifj_barnesdoublegamma(z1, 1.0, dps=50)
    rhs1 = jnp.exp(barnesg._complex_loggamma(jnp.asarray(z1, dtype=jnp.complex128)))
    assert abs(_to_complex(lhs1) - _to_complex(rhs1)) < 5e-5

    z2 = 1.7 + 0.1j
    tau = 0.5
    lhs2 = double_gamma.ifj_barnesdoublegamma(z2 + 1.0, tau, dps=60) / double_gamma.ifj_barnesdoublegamma(z2, tau, dps=60)
    rhs2 = jnp.exp(barnesg._complex_loggamma(jnp.asarray(z2 / tau, dtype=jnp.complex128)))
    assert abs(_to_complex(lhs2) - _to_complex(rhs2)) < 5e-3


def test_ifj_log_and_value_are_consistent_and_finite_on_positive_slice():
    z = jnp.asarray(2.5 + 0.2j, dtype=jnp.complex128)
    tau = jnp.asarray(0.5, dtype=jnp.float64)
    value = double_gamma.ifj_barnesdoublegamma(z, tau, dps=50)
    log_value = double_gamma.ifj_log_barnesdoublegamma(z, tau, dps=50)
    assert abs(_to_complex(jnp.log(value)) - _to_complex(log_value)) < 5e-8

    real_slice = jnp.asarray([1.5, 2.0, 2.5, 3.0], dtype=jnp.float64)
    vals = double_gamma.ifj_barnesdoublegamma(real_slice, 1.0, dps=50)
    assert bool(jnp.all(jnp.isfinite(jnp.real(vals))))
    assert bool(jnp.all(jnp.isfinite(jnp.imag(vals))))


def test_ifj_diagnostics_report_shift_and_truncation_choices():
    diagnostics = double_gamma.ifj_barnesdoublegamma_diagnostics(0.2 + 0.05j, 1.0, dps=60, max_m_cap=256)
    assert diagnostics.dps == 60
    assert diagnostics.tau == 1.0
    assert diagnostics.m_base >= 64
    assert diagnostics.m_used <= diagnostics.max_m_cap
    assert diagnostics.n_shift >= 1
    assert diagnostics.richardson_levels == 3


def test_ifj_provider_beats_legacy_tau1_anchor_error():
    anchors = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    expected = jnp.asarray([1.0, 1.0, 1.0, 2.0], dtype=jnp.complex128)
    legacy = jnp.asarray([double_gamma.bdg_barnesdoublegamma(x, 1.0, prec_bits=80) for x in anchors], dtype=jnp.complex128)
    ifj = jnp.asarray([double_gamma.ifj_barnesdoublegamma(x, 1.0, dps=60) for x in anchors], dtype=jnp.complex128)
    legacy_err = jnp.max(jnp.abs(legacy - expected))
    ifj_err = jnp.max(jnp.abs(ifj - expected))
    assert float(ifj_err) < float(legacy_err)


def test_acb_barnes_g_and_log_aliases_match_exact_point_kernels_on_exact_boxes():
    z = 1.3 + 0.2j
    box = _exact_box(z)

    g_box = acb_core.acb_barnes_g(box)
    log_box = acb_core.acb_log_barnes_g(box)

    g_mid = acb_core.acb_midpoint(g_box)
    log_mid = acb_core.acb_midpoint(log_box)
    g_expected = barnesg.barnesg_complex(jnp.asarray(z, dtype=jnp.complex128))
    log_expected = barnesg.log_barnesg(jnp.asarray(z, dtype=jnp.complex128))

    assert abs(_to_complex(g_mid) - _to_complex(g_expected)) < 1e-10
    assert abs(_to_complex(log_mid) - _to_complex(log_expected)) < 1e-10
    assert abs(_to_complex(jnp.exp(log_mid)) - _to_complex(g_mid)) < 1e-10
