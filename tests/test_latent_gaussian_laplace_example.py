import jax
import jax.numpy as jnp

from examples import example_latent_gaussian_laplace as ex


def test_latent_gaussian_laplace_example_returns_finite_mode_and_gradient():
    y = jnp.asarray([1.0, 0.0, 1.0, 1.0], dtype=jnp.float64)
    theta = jnp.asarray(-0.2, dtype=jnp.float64)

    value, aux = ex.laplace_log_marginal(y, theta)
    grad = jax.grad(lambda t: ex.laplace_log_marginal(y, t)[0])(theta)

    assert jnp.isfinite(value)
    assert jnp.isfinite(grad)
    assert aux["spd_safe"]
    assert aux["solve_metadata"].implicit_adjoint
    assert aux["mode"].shape == y.shape
