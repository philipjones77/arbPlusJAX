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
