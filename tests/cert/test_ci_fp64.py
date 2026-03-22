import jax
import jax.numpy as jnp

from arbplusjax.special.gamma.derivatives import incomplete_gamma_upper_argument_derivative
from arbplusjax.special.gamma.incomplete_gamma import incomplete_gamma_upper_point
from arbplusjax.special.gamma.incomplete_gamma_ad import incomplete_gamma_upper_switched_z_vjp


def test_incomplete_gamma_switched_fp64_matches_point_value():
    s = jnp.float64(2.5)
    z = jnp.float64(1.75)
    got = incomplete_gamma_upper_switched_z_vjp(s, z)
    ref = incomplete_gamma_upper_point(s, z)
    assert float(jnp.abs(got - ref) / jnp.maximum(jnp.abs(ref), jnp.asarray(1e-12, dtype=jnp.float64))) <= 1e-7


def test_incomplete_gamma_switched_fp64_grad_matches_analytic_derivative():
    s = jnp.float64(2.5)
    z = jnp.float64(1.75)
    got = jax.grad(lambda zz: incomplete_gamma_upper_switched_z_vjp(s, zz))(z)
    ref = incomplete_gamma_upper_argument_derivative(s, z)
    assert float(jnp.abs(got - ref) / jnp.maximum(jnp.abs(ref), jnp.asarray(1e-12, dtype=jnp.float64))) <= 1e-12


def test_incomplete_gamma_switched_fp64_directional_dot_check():
    s = jnp.float64(3.25)
    z = jnp.float64(2.0)
    direction = jnp.float64(-0.125)
    grad = jax.grad(lambda zz: incomplete_gamma_upper_switched_z_vjp(s, zz))(z)
    lhs = grad * direction

    eps = jnp.float64(1e-6)
    rhs = (
        incomplete_gamma_upper_switched_z_vjp(s, z + eps * direction)
        - incomplete_gamma_upper_switched_z_vjp(s, z - eps * direction)
    ) / (2.0 * eps)
    rhs = rhs * jnp.float64(1.0)

    assert float(jnp.abs(lhs - rhs) / jnp.maximum(jnp.abs(rhs), jnp.asarray(1e-8, dtype=jnp.float64))) <= 1e-6
