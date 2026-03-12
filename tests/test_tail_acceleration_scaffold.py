import jax.numpy as jnp

from arbplusjax import api


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


def test_tail_integral_accelerated_mpfallback_is_exposed():
    value, diagnostics = api.tail_integral_accelerated(
        lambda t: jnp.exp(-t),
        0.75,
        method="mpfallback",
        return_diagnostics=True,
    )

    assert jnp.isfinite(value)
    assert diagnostics.method == "mpfallback"
    assert diagnostics.fallback_used is True


def test_tail_integral_metadata_is_public():
    metadata = api.get_public_function_metadata("tail_integral")

    assert metadata.family == "integration"
    assert metadata.stability == "stable"
