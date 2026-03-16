import jax
import jax.numpy as jnp

from arbplusjax import api
from arbplusjax import double_interval as di


def test_incomplete_bessel_k_matches_full_besselk_at_zero_lower_limit():
    nu = jnp.float64(0.5)
    z = jnp.float64(1.75)

    inc = api.incomplete_bessel_k(nu, z, jnp.float64(0.0), mode="point", method="quadrature")
    full = api.eval_point("besselk", nu, z)

    assert jnp.isclose(inc, full, rtol=2e-2, atol=2e-3)


def test_incomplete_bessel_k_four_modes_are_shaped_consistently():
    nu = jnp.float64(0.25)
    z = jnp.float64(1.1)
    lower = jnp.float64(0.4)

    point = api.incomplete_bessel_k(nu, z, lower, mode="point")
    basic = api.incomplete_bessel_k(nu, z, lower, mode="basic")
    adaptive = api.incomplete_bessel_k(nu, z, lower, mode="adaptive")
    rigorous = api.incomplete_bessel_k(nu, z, lower, mode="rigorous")

    assert jnp.ndim(point) == 0
    assert basic.shape == (2,)
    assert adaptive.shape == (2,)
    assert rigorous.shape == (2,)
    assert bool((basic[0] <= point) & (point <= basic[1]))
    assert bool(di.contains(adaptive, basic))
    assert bool(di.contains(rigorous, adaptive))


def test_incomplete_bessel_k_lower_limit_derivative_matches_kernel():
    nu = jnp.float64(0.5)
    z = jnp.float64(1.3)
    lower = jnp.float64(0.7)

    deriv = api.incomplete_bessel_k_lower_limit_derivative(nu, z, lower)
    expected = -jnp.exp(-z * jnp.cosh(lower)) * jnp.cosh(nu * lower)

    assert jnp.isclose(deriv, expected, rtol=1e-10, atol=1e-10)


def test_incomplete_bessel_k_argument_derivative_matches_finite_difference():
    nu = jnp.float64(0.5)
    z = jnp.float64(1.6)
    lower = jnp.float64(0.3)
    step = jnp.float64(1e-3)

    explicit = api.incomplete_bessel_k_argument_derivative(nu, z, lower, method="quadrature")
    forward = api.incomplete_bessel_k(nu, z + step, lower, mode="point", method="quadrature")
    backward = api.incomplete_bessel_k(nu, z - step, lower, mode="point", method="quadrature")
    finite_diff = (forward - backward) / (2.0 * step)

    assert jnp.isclose(explicit, finite_diff, rtol=5e-2, atol=5e-3)


def test_incomplete_bessel_k_batch_matches_scalar_calls():
    nu = jnp.asarray([0.25, 0.5], dtype=jnp.float64)
    z = jnp.asarray([1.1, 1.3], dtype=jnp.float64)
    lower = jnp.asarray([0.2, 0.4], dtype=jnp.float64)

    batch = api.incomplete_bessel_k_batch(nu, z, lower, mode="point", method="quadrature")
    scalar = jnp.asarray(
        [
            api.incomplete_bessel_k(nu[0], z[0], lower[0], mode="point", method="quadrature"),
            api.incomplete_bessel_k(nu[1], z[1], lower[1], mode="point", method="quadrature"),
        ]
    )

    assert jnp.allclose(batch, scalar, rtol=1e-10, atol=1e-10)


def test_incomplete_bessel_k_auto_chooses_asymptotic_in_large_decay_regime():
    nu = jnp.float64(0.5)
    z = jnp.float64(20.0)
    lower = jnp.float64(0.6)

    value, diagnostics = api.incomplete_bessel_k(
        nu,
        z,
        lower,
        mode="point",
        method="auto",
        return_diagnostics=True,
    )
    reference = api.incomplete_bessel_k(nu, z, lower, mode="point", method="quadrature")

    assert diagnostics.method == "asymptotic"
    assert jnp.isclose(value, reference, rtol=1.5e-1, atol=1e-8)


def test_incomplete_bessel_k_auto_can_trigger_high_precision_refine_for_fragile_regime():
    nu = jnp.float64(13.0)
    z = jnp.float64(0.5)
    lower = jnp.float64(0.05)

    value, diagnostics = api.incomplete_bessel_k(
        nu,
        z,
        lower,
        mode="point",
        method="auto",
        return_diagnostics=True,
    )

    assert diagnostics.method == "high_precision_refine"
    assert diagnostics.fallback_used is True
    assert jnp.isfinite(value)


def test_incomplete_bessel_k_recurrence_reports_instability_flags_in_fragile_regime():
    _, diagnostics = api.incomplete_bessel_k(
        jnp.float64(9.0),
        jnp.float64(0.8),
        jnp.float64(0.15),
        mode="point",
        method="recurrence",
        return_diagnostics=True,
    )

    assert diagnostics.method == "recurrence"
    assert "small_phi_prime" in diagnostics.instability_flags
    assert diagnostics.precision_warning is True


def test_incomplete_bessel_k_asymptotic_reports_small_lower_flag():
    _, diagnostics = api.incomplete_bessel_k(
        jnp.float64(0.5),
        jnp.float64(25.0),
        jnp.float64(0.1),
        mode="point",
        method="asymptotic",
        return_diagnostics=True,
    )

    assert diagnostics.method == "asymptotic"
    assert "small_lower_limit" in diagnostics.instability_flags


def test_incomplete_bessel_k_explicit_recurrence_tracks_large_lower_limit_regime():
    nu = jnp.float64(0.5)
    z = jnp.float64(20.0)
    lower = jnp.float64(1.2)

    value, diagnostics = api.incomplete_bessel_k(
        nu,
        z,
        lower,
        mode="point",
        method="recurrence",
        return_diagnostics=True,
    )
    reference = api.incomplete_bessel_k(nu, z, lower, mode="point", method="quadrature")

    assert diagnostics.method == "recurrence"
    assert jnp.isclose(value, reference, rtol=1.5e-1, atol=1e-8)


def test_incomplete_bessel_k_custom_jvp_matches_explicit_derivatives():
    nu = jnp.float64(0.7)
    z = jnp.float64(1.4)
    lower = jnp.float64(0.3)

    def target(nu_v, z_v, lower_v):
        return api.incomplete_bessel_k(nu_v, z_v, lower_v, mode="point", method="quadrature")

    _, tangent_out = jax.jvp(
        target,
        (nu, z, lower),
        (jnp.float64(0.0), jnp.float64(1.0), jnp.float64(1.0)),
    )
    expected = api.incomplete_bessel_k_argument_derivative(nu, z, lower, method="quadrature") + api.incomplete_bessel_k_lower_limit_derivative(
        nu, z, lower
    )

    assert jnp.isclose(tangent_out, expected, rtol=5e-2, atol=5e-3)
