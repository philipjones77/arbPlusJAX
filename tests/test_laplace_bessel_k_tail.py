import jax
import jax.numpy as jnp

from arbplusjax import api


def test_laplace_bessel_k_tail_matches_generic_tail_integral():
    nu = jnp.float64(0.5)
    z = jnp.float64(1.5)
    lam = jnp.float64(0.75)
    lower = jnp.float64(0.5)

    specialized = api.laplace_bessel_k_tail(nu, z, lam, lower, mode="point", method="quadrature")
    generic = api.tail_integral(
        lambda t: jnp.exp(-lam * t) * api.eval_point("besselk", nu, z * t),
        lower,
        panel_width=0.125,
        max_panels=160,
        samples_per_panel=24,
    )

    assert jnp.isclose(specialized, generic, rtol=5e-3, atol=5e-4)


def test_laplace_bessel_k_tail_auto_can_trigger_nonquadrature_path():
    value, diagnostics = api.laplace_bessel_k_tail(
        0.5,
        0.4,
        0.1,
        0.5,
        mode="point",
        method="auto",
        return_diagnostics=True,
    )

    assert jnp.isfinite(value)
    assert diagnostics.method in {"aitken", "high_precision_refine", "quadrature"}


def test_laplace_bessel_k_tail_lower_limit_derivative_matches_integrand():
    nu = jnp.float64(0.5)
    z = jnp.float64(1.5)
    lam = jnp.float64(0.75)
    lower = jnp.float64(0.5)
    step = jnp.float64(1e-4)

    explicit = api.laplace_bessel_k_tail_lower_limit_derivative(nu, z, lam, lower)
    forward = api.laplace_bessel_k_tail(nu, z, lam, lower + step, mode="point", method="quadrature")
    backward = api.laplace_bessel_k_tail(nu, z, lam, lower - step, mode="point", method="quadrature")
    finite_diff = (forward - backward) / (2.0 * step)

    assert jnp.isclose(explicit, finite_diff, rtol=5e-3, atol=5e-4)


def test_laplace_bessel_k_tail_lambda_derivative_matches_finite_difference():
    nu = jnp.float64(0.5)
    z = jnp.float64(1.5)
    lam = jnp.float64(0.75)
    lower = jnp.float64(0.5)
    step = jnp.float64(1e-4)

    explicit = api.laplace_bessel_k_tail_lambda_derivative(nu, z, lam, lower, method="quadrature")
    forward = api.laplace_bessel_k_tail(nu, z, lam + step, lower, mode="point", method="quadrature")
    backward = api.laplace_bessel_k_tail(nu, z, lam - step, lower, mode="point", method="quadrature")
    finite_diff = (forward - backward) / (2.0 * step)

    assert jnp.isclose(explicit, finite_diff, rtol=5e-3, atol=5e-4)


def test_laplace_bessel_k_tail_batch_and_interval_batch_work():
    nu = jnp.asarray([0.5, 1.0], dtype=jnp.float64)
    z = jnp.asarray([1.5, 2.0], dtype=jnp.float64)
    lam = jnp.asarray([0.75, 1.0], dtype=jnp.float64)
    lower = jnp.asarray([0.5, 0.75], dtype=jnp.float64)

    point_batch = api.laplace_bessel_k_tail_batch(nu, z, lam, lower, mode="point", method="quadrature")
    basic_batch = api.eval_interval_batch("laplace_bessel_k_tail", nu, z, lam, lower, mode="basic")

    assert point_batch.shape == nu.shape
    assert basic_batch.shape == nu.shape + (2,)


def test_laplace_bessel_k_tail_custom_jvp_matches_explicit_derivatives():
    nu = jnp.float64(0.5)
    z = jnp.float64(1.5)
    lam = jnp.float64(0.75)
    lower = jnp.float64(0.5)

    def fn(nu_v, z_v, lam_v, lower_v):
        return api.laplace_bessel_k_tail(nu_v, z_v, lam_v, lower_v, mode="point", method="quadrature")

    _, tangent = jax.jvp(
        fn,
        (nu, z, lam, lower),
        (jnp.float64(0.0), jnp.float64(0.0), jnp.float64(1.0), jnp.float64(1.0)),
    )
    expected = api.laplace_bessel_k_tail_lambda_derivative(
        nu,
        z,
        lam,
        lower,
        method="quadrature",
    ) + api.laplace_bessel_k_tail_lower_limit_derivative(nu, z, lam, lower)

    assert jnp.isclose(tangent, expected, rtol=5e-3, atol=5e-4)
