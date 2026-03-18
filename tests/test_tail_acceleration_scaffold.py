from jax import lax
import jax.numpy as jnp

from arbplusjax import api
from arbplusjax.special.tail_acceleration.quadrature import finite_interval_quadrature
from arbplusjax.special.tail_acceleration.recurrence import ratio_recurrence_terms


def test_tail_integral_quadrature_matches_exp_tail():
    value, diagnostics = api.tail_integral(
        lambda t: jnp.exp(-t),
        1.25,
        panel_width=0.1,
        max_panels=120,
        samples_per_panel=16,
        return_diagnostics=True,
    )

    assert jnp.isclose(value, jnp.exp(-1.25), rtol=5e-3, atol=5e-4)
    assert diagnostics.method == "quadrature"
    assert diagnostics.panel_count == 120


def test_tail_integral_accelerated_auto_returns_sequence_method_for_slow_decay():
    problem = api.TailIntegralProblem(
        integrand=lambda t: jnp.exp(-t),
        lower_limit=0.5,
        panel_width=0.2,
        max_panels=24,
        samples_per_panel=12,
        regime_metadata=api.TailRegimeMetadata(decay_rate=0.1, oscillation_level=0.0),
    )

    value, diagnostics = api.tail_integral_accelerated(problem, method="auto", return_diagnostics=True)

    assert jnp.isfinite(value)
    assert diagnostics.method == "aitken"


def test_tail_integral_accelerated_high_precision_refine_is_exposed():
    value, diagnostics = api.tail_integral_accelerated(
        lambda t: jnp.exp(-t),
        0.75,
        method="high_precision_refine",
        return_diagnostics=True,
    )

    assert jnp.isfinite(value)
    assert diagnostics.method == "high_precision_refine"
    assert diagnostics.fallback_used is True


def test_tail_integral_accelerated_mpfallback_alias_still_works():
    value, diagnostics = api.tail_integral_accelerated(
        lambda t: jnp.exp(-t),
        0.75,
        method="mpfallback",
        return_diagnostics=True,
    )

    assert jnp.isfinite(value)
    assert diagnostics.method == "high_precision_refine"


def test_tail_integral_gaussian_tail_matches_erfc_reference():
    lower = jnp.float64(0.8)
    value = api.tail_integral(
        lambda t: jnp.exp(-(t * t)),
        lower,
        panel_width=0.05,
        max_panels=180,
        samples_per_panel=16,
    )
    expected = 0.5 * jnp.sqrt(jnp.pi) * lax.erfc(lower)

    assert jnp.isclose(value, expected, rtol=5e-3, atol=5e-4)


def test_tail_integral_accelerated_wynn_tracks_damped_oscillatory_tail():
    lower = jnp.float64(0.4)
    problem = api.TailIntegralProblem(
        integrand=lambda t: jnp.exp(-t) * jnp.cos(t),
        lower_limit=lower,
        panel_width=0.1,
        max_panels=80,
        samples_per_panel=16,
        regime_metadata=api.TailRegimeMetadata(decay_rate=0.3, oscillation_level=1.5),
    )

    value, diagnostics = api.tail_integral_accelerated(problem, method="auto", return_diagnostics=True)
    expected = 0.5 * jnp.exp(-lower) * (jnp.cos(lower) - jnp.sin(lower))

    assert diagnostics.method == "wynn"
    assert jnp.isclose(value, expected, rtol=1e-1, atol=1e-2)


def test_tail_integral_batch_matches_scalar_calls():
    lowers = jnp.asarray([0.2, 0.6, 1.0], dtype=jnp.float64)
    batch = api.tail_integral_batch(
        lambda t: jnp.exp(-t),
        lowers,
        panel_width=0.1,
        max_panels=120,
        samples_per_panel=16,
    )
    scalar = jnp.asarray(
        [
            api.tail_integral(lambda t: jnp.exp(-t), lowers[0], panel_width=0.1, max_panels=120, samples_per_panel=16),
            api.tail_integral(lambda t: jnp.exp(-t), lowers[1], panel_width=0.1, max_panels=120, samples_per_panel=16),
            api.tail_integral(lambda t: jnp.exp(-t), lowers[2], panel_width=0.1, max_panels=120, samples_per_panel=16),
        ]
    )

    assert jnp.allclose(batch, scalar, rtol=1e-10, atol=1e-10)


def test_tail_integral_metadata_is_public():
    metadata = api.get_public_function_metadata("tail_integral")

    assert metadata.family == "integration"
    assert metadata.stability == "stable"


def test_tail_ratio_recurrence_supports_generalized_order_two_coeff_interface():
    spec = api.TailRatioRecurrence(
        a_init=(1.0, 1.0),
        b_init=(1.0, 2.0),
        a_coeffs=lambda n: (1.0, 1.0),
        b_coeffs=lambda n: (1.0, 1.0),
        order=2,
        note="Fibonacci-style generalized coefficient recurrence.",
    )

    a_terms, b_terms = ratio_recurrence_terms(spec, n_terms=6)

    assert jnp.allclose(a_terms[:6], jnp.asarray([1.0, 1.0, 2.0, 3.0, 5.0, 8.0]))
    assert jnp.allclose(b_terms[:6], jnp.asarray([1.0, 2.0, 3.0, 5.0, 8.0, 13.0]))


def test_tail_ratio_recurrence_supports_order_four_history():
    spec = api.TailRatioRecurrence(
        a_init=(1.0, 2.0, 3.0, 4.0),
        b_init=(2.0, 3.0, 4.0, 5.0),
        a_coeffs=lambda n: (1.0, 1.0, 0.0, 0.0),
        b_coeffs=lambda n: (1.0, 1.0, 0.0, 0.0),
        order=4,
        note="Order-four interface using only the two most recent terms.",
    )

    a_terms, b_terms = ratio_recurrence_terms(spec, n_terms=6)

    assert jnp.allclose(a_terms[:6], jnp.asarray([1.0, 2.0, 3.0, 4.0, 7.0, 11.0]))
    assert jnp.allclose(b_terms[:6], jnp.asarray([2.0, 3.0, 4.0, 5.0, 9.0, 14.0]))


def test_tail_integral_accelerated_recurrence_reports_instability_flags():
    problem = api.TailIntegralProblem(
        integrand=lambda t: jnp.exp(-t),
        lower_limit=0.5,
        recurrence=api.TailRatioRecurrence(
            a_init=(1.0, 1.0),
            b_init=(1.0, 1e-16),
            a_coeffs=lambda n: (1.0, 0.0),
            b_coeffs=lambda n: (0.0, 0.0),
            order=2,
            note="Deliberately singular denominator recurrence.",
        ),
        max_panels=6,
    )

    value, diagnostics = api.tail_integral_accelerated(problem, method="recurrence", return_diagnostics=True)

    assert jnp.isfinite(value)
    assert diagnostics.method == "recurrence"
    assert "small_denominator" in diagnostics.instability_flags
    assert diagnostics.precision_warning is True


def test_tail_integral_quadrature_supports_gauss_legendre_rule():
    problem = api.TailIntegralProblem(
        integrand=lambda t: jnp.exp(-t),
        lower_limit=0.5,
        panel_width=0.1,
        max_panels=100,
        samples_per_panel=8,
        quadrature_rule="gauss_legendre",
        regime_metadata=api.TailRegimeMetadata(decay_rate=1.0, oscillation_level=0.0),
    )

    value, diagnostics = api.tail_integral_accelerated(problem, method="quadrature", return_diagnostics=True)

    assert jnp.isclose(value, jnp.exp(-0.5), rtol=5e-3, atol=5e-4)
    assert "quadrature_rule=gauss_legendre" in diagnostics.note


def test_finite_interval_quadrature_supports_simpson_rule():
    value, diagnostics = finite_interval_quadrature(
        lambda t: jnp.sin(t),
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(jnp.pi, dtype=jnp.float64),
        panel_count=32,
        samples_per_panel=7,
        quadrature_rule="simpson",
    )

    assert jnp.isclose(value, 2.0, rtol=1e-5, atol=1e-6)
    assert "quadrature_rule=simpson" in diagnostics.note
