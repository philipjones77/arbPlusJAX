import jax
from jax import lax
import jax.numpy as jnp

from arbplusjax import api


def test_incomplete_gamma_upper_matches_regularized_reference():
    s = jnp.float64(2.5)
    z = jnp.float64(1.75)

    value = api.incomplete_gamma_upper(s, z, mode="point", regularized=True, method="quadrature")

    assert jnp.isclose(value, lax.igammac(s, z), rtol=5e-3, atol=5e-4)


def test_incomplete_gamma_lower_matches_regularized_reference():
    s = jnp.float64(2.5)
    z = jnp.float64(1.75)

    value = api.incomplete_gamma_lower(s, z, mode="point", regularized=True, method="quadrature")

    assert jnp.isclose(value, lax.igamma(s, z), rtol=5e-3, atol=5e-4)


def test_incomplete_gamma_auto_can_trigger_high_precision_refine():
    value, diagnostics = api.incomplete_gamma_upper(
        0.75,
        0.05,
        mode="point",
        method="auto",
        return_diagnostics=True,
    )

    assert jnp.isfinite(value)
    assert diagnostics.method == "high_precision_refine"
    assert diagnostics.fallback_used is True


def test_incomplete_gamma_batch_matches_scalar_calls():
    s = jnp.asarray([1.5, 2.5], dtype=jnp.float64)
    z = jnp.asarray([0.75, 1.75], dtype=jnp.float64)

    batch = api.incomplete_gamma_upper_batch(s, z, mode="point", regularized=True, method="quadrature")
    scalar = jnp.asarray(
        [
            api.incomplete_gamma_upper(s[0], z[0], mode="point", regularized=True, method="quadrature"),
            api.incomplete_gamma_upper(s[1], z[1], mode="point", regularized=True, method="quadrature"),
        ]
    )

    assert jnp.allclose(batch, scalar, rtol=1e-10, atol=1e-10)


def test_incomplete_gamma_complement_identity_holds():
    s = jnp.float64(3.25)
    z = jnp.float64(2.0)

    lower = api.incomplete_gamma_lower(s, z, mode="point", method="quadrature")
    upper = api.incomplete_gamma_upper(s, z, mode="point", method="quadrature")
    gamma_s = jnp.exp(lax.lgamma(s))

    assert jnp.isclose(lower + upper, gamma_s, rtol=5e-3, atol=5e-4)


def test_incomplete_gamma_eval_point_and_interval_batch_work():
    s = jnp.asarray([1.5, 2.5], dtype=jnp.float64)
    z = jnp.asarray([0.75, 1.75], dtype=jnp.float64)

    point_batch = api.eval_point_batch("incomplete_gamma_upper", s, z)
    basic_batch = api.eval_interval_batch("incomplete_gamma_upper", s, z, mode="basic")

    assert point_batch.shape == s.shape
    assert basic_batch.shape == s.shape + (2,)


def test_incomplete_gamma_argument_derivative_matches_finite_difference():
    s = jnp.float64(2.5)
    z = jnp.float64(1.75)
    step = jnp.float64(1e-4)

    explicit = api.incomplete_gamma_upper_argument_derivative(s, z)
    forward = api.incomplete_gamma_upper(s, z + step, mode="point", method="quadrature")
    backward = api.incomplete_gamma_upper(s, z - step, mode="point", method="quadrature")
    finite_diff = (forward - backward) / (2.0 * step)

    assert jnp.isclose(explicit, finite_diff, rtol=5e-3, atol=5e-4)


def test_incomplete_gamma_custom_jvp_matches_explicit_derivatives():
    s = jnp.float64(2.5)
    z = jnp.float64(1.75)

    def fn(s_v, z_v):
        return api.incomplete_gamma_upper(s_v, z_v, mode="point", method="quadrature")

    _, tangent = jax.jvp(fn, (s, z), (jnp.float64(1.0), jnp.float64(1.0)))
    expected = api.incomplete_gamma_upper_parameter_derivative(
        s,
        z,
        method="quadrature",
    ) + api.incomplete_gamma_upper_argument_derivative(s, z)

    assert jnp.isclose(tangent, expected, rtol=5e-3, atol=5e-4)
