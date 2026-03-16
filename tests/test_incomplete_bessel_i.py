import jax
import jax.numpy as jnp

from arbplusjax import api
from arbplusjax import double_interval as di


def test_incomplete_bessel_i_matches_integer_order_besseli_at_pi():
    nu = jnp.float64(0.0)
    z = jnp.float64(1.25)
    upper = jnp.pi

    inc = api.incomplete_bessel_i(nu, z, upper, mode="point")
    full = jnp.pi * api.eval_point("besseli", nu, z)

    assert jnp.isclose(inc, full, rtol=2e-2, atol=2e-3)


def test_incomplete_bessel_i_four_modes_are_shaped_consistently():
    nu = jnp.float64(1.0)
    z = jnp.float64(0.8)
    upper = jnp.float64(1.2)

    point = api.incomplete_bessel_i(nu, z, upper, mode="point")
    basic = api.incomplete_bessel_i(nu, z, upper, mode="basic")
    adaptive = api.incomplete_bessel_i(nu, z, upper, mode="adaptive")
    rigorous = api.incomplete_bessel_i(nu, z, upper, mode="rigorous")

    assert jnp.ndim(point) == 0
    assert basic.shape == (2,)
    assert adaptive.shape == (2,)
    assert rigorous.shape == (2,)
    assert bool((basic[0] <= point) & (point <= basic[1]))
    assert bool(di.contains(adaptive, basic))
    assert bool(di.contains(rigorous, adaptive))


def test_incomplete_bessel_i_upper_limit_derivative_matches_integrand():
    nu = jnp.float64(0.5)
    z = jnp.float64(1.1)
    upper = jnp.float64(0.7)

    deriv = api.incomplete_bessel_i_upper_limit_derivative(nu, z, upper)
    expected = jnp.exp(z * jnp.cos(upper)) * jnp.cos(nu * upper)

    assert jnp.isclose(deriv, expected, rtol=1e-10, atol=1e-10)


def test_incomplete_bessel_i_argument_derivative_matches_finite_difference():
    nu = jnp.float64(0.5)
    z = jnp.float64(0.9)
    upper = jnp.float64(0.8)
    step = jnp.float64(1e-3)

    explicit = api.incomplete_bessel_i_argument_derivative(nu, z, upper)
    forward = api.incomplete_bessel_i(nu, z + step, upper, mode="point")
    backward = api.incomplete_bessel_i(nu, z - step, upper, mode="point")
    finite_diff = (forward - backward) / (2.0 * step)

    assert jnp.isclose(explicit, finite_diff, rtol=5e-2, atol=5e-3)


def test_incomplete_bessel_i_batch_matches_scalar_calls():
    nu = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    z = jnp.asarray([1.0, 0.8], dtype=jnp.float64)
    upper = jnp.asarray([jnp.pi, 1.2], dtype=jnp.float64)

    batch = api.incomplete_bessel_i_batch(nu, z, upper, mode="point")
    scalar = jnp.asarray(
        [
            api.incomplete_bessel_i(nu[0], z[0], upper[0], mode="point"),
            api.incomplete_bessel_i(nu[1], z[1], upper[1], mode="point"),
        ]
    )

    assert jnp.allclose(batch, scalar, rtol=1e-10, atol=1e-10)


def test_incomplete_bessel_i_auto_can_trigger_high_precision_refine_in_fragile_regime():
    value, diagnostics = api.incomplete_bessel_i(
        jnp.float64(12.0),
        jnp.float64(9.0),
        jnp.pi - jnp.float64(0.05),
        mode="point",
        method="auto",
        return_diagnostics=True,
    )

    assert diagnostics.method == "high_precision_refine"
    assert diagnostics.fallback_used is True
    assert jnp.isfinite(value)


def test_incomplete_bessel_i_high_precision_refine_alias_keeps_shape_contract():
    point, diagnostics = api.incomplete_bessel_i(
        jnp.float64(1.0),
        jnp.float64(11.0),
        jnp.float64(0.1),
        mode="point",
        method="high_precision_refine",
        return_diagnostics=True,
    )
    basic = api.incomplete_bessel_i(
        jnp.float64(1.0),
        jnp.float64(11.0),
        jnp.float64(0.1),
        mode="basic",
        method="high_precision_refine",
    )

    assert diagnostics.method == "high_precision_refine"
    assert basic.shape == (2,)
    assert bool((basic[0] <= point) & (point <= basic[1]))


def test_incomplete_bessel_i_quadrature_reports_rule_in_diagnostics():
    value, diagnostics = api.incomplete_bessel_i(
        jnp.float64(0.5),
        jnp.float64(1.25),
        jnp.float64(1.0),
        mode="point",
        method="quadrature",
        return_diagnostics=True,
    )

    assert jnp.isfinite(value)
    assert "quadrature_rule=simpson" in diagnostics.note


def test_incomplete_bessel_i_eval_point_batch_and_interval_batch_work():
    nu = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    z = jnp.asarray([1.0, 0.8], dtype=jnp.float64)
    upper = jnp.asarray([jnp.pi, 1.2], dtype=jnp.float64)

    point_batch = api.eval_point_batch("incomplete_bessel_i", nu, z, upper)
    basic_batch = api.eval_interval_batch("incomplete_bessel_i", nu, z, upper, mode="basic")

    assert point_batch.shape == (2,)
    assert basic_batch.shape == (2, 2)
    assert bool((basic_batch[0, 0] <= point_batch[0]) & (point_batch[0] <= basic_batch[0, 1]))
    assert bool((basic_batch[1, 0] <= point_batch[1]) & (point_batch[1] <= basic_batch[1, 1]))


def test_incomplete_bessel_i_custom_jvp_matches_explicit_derivatives():
    nu = jnp.float64(0.4)
    z = jnp.float64(0.9)
    upper = jnp.float64(0.7)

    def target(nu_v, z_v, upper_v):
        return api.incomplete_bessel_i(nu_v, z_v, upper_v, mode="point")

    _, tangent_out = jax.jvp(
        target,
        (nu, z, upper),
        (jnp.float64(0.0), jnp.float64(1.0), jnp.float64(1.0)),
    )
    expected = api.incomplete_bessel_i_argument_derivative(nu, z, upper) + api.incomplete_bessel_i_upper_limit_derivative(
        nu, z, upper
    )

    assert jnp.isclose(tangent_out, expected, rtol=5e-2, atol=5e-3)
