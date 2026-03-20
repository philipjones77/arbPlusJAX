import jax.numpy as jnp

from arbplusjax import bessel_kernels as bk
from arbplusjax.special.bessel import (
    hankel1,
    hankel1_derivative,
    hankel1_order_recurrence,
    hankel2,
    hankel2_derivative,
    hankel2_order_recurrence,
    scaled_hankel1,
    scaled_hankel2,
)


def test_hankel1_matches_j_plus_i_y():
    nu = jnp.asarray(0.4 + 0.2j, dtype=jnp.complex64)
    z = jnp.asarray(2.5 + 0.7j, dtype=jnp.complex64)

    expected = bk.complex_bessel_series(nu, z, -1.0) + 1j * bk.complex_bessel_y(nu, z)

    assert jnp.allclose(hankel1(nu, z, method="direct"), expected, rtol=1e-5, atol=1e-5)


def test_hankel2_matches_j_minus_i_y():
    nu = jnp.asarray(0.4 + 0.2j, dtype=jnp.complex64)
    z = jnp.asarray(2.5 + 0.7j, dtype=jnp.complex64)

    expected = bk.complex_bessel_series(nu, z, -1.0) - 1j * bk.complex_bessel_y(nu, z)

    assert jnp.allclose(hankel2(nu, z, method="direct"), expected, rtol=1e-5, atol=1e-5)


def test_scaled_hankel_definitions_match_unscaled_forms():
    nu = jnp.asarray(0.3 + 0.15j, dtype=jnp.complex64)
    z = jnp.asarray(3.0 + 0.4j, dtype=jnp.complex64)

    assert jnp.allclose(scaled_hankel1(nu, z, method="direct"), jnp.exp(-1j * z) * hankel1(nu, z, method="direct"), rtol=1e-5, atol=1e-5)
    assert jnp.allclose(scaled_hankel2(nu, z, method="direct"), jnp.exp(1j * z) * hankel2(nu, z, method="direct"), rtol=1e-5, atol=1e-5)


def test_hankel_derivative_matches_centered_difference():
    nu = jnp.asarray(0.35 + 0.1j, dtype=jnp.complex64)
    z = jnp.asarray(2.2 + 0.5j, dtype=jnp.complex64)
    step = jnp.asarray(1e-3, dtype=jnp.float32)

    fd1 = (hankel1(nu, z + step, method="direct") - hankel1(nu, z - step, method="direct")) / (2.0 * step)
    fd2 = (hankel2(nu, z + step, method="direct") - hankel2(nu, z - step, method="direct")) / (2.0 * step)

    assert jnp.allclose(hankel1_derivative(nu, z, method="direct"), fd1, rtol=5e-2, atol=5e-3)
    assert jnp.allclose(hankel2_derivative(nu, z, method="direct"), fd2, rtol=5e-2, atol=5e-3)


def test_hankel_order_recurrence_matches_neighbor_value():
    nu = jnp.asarray(0.6 + 0.1j, dtype=jnp.complex64)
    z = jnp.asarray(2.8 + 0.3j, dtype=jnp.complex64)

    assert jnp.allclose(hankel1_order_recurrence(nu, z, method="direct"), hankel1(nu - 1.0, z, method="direct"), rtol=1e-5, atol=1e-5)
    assert jnp.allclose(hankel2_order_recurrence(nu, z, method="direct"), hankel2(nu - 1.0, z, method="direct"), rtol=1e-5, atol=1e-5)


def test_api_exposes_hankel_point_and_batch_paths():
    nu = jnp.asarray([0.25 + 0.1j, 0.5 + 0.2j], dtype=jnp.complex64)
    z = jnp.asarray([2.0 + 0.3j, 2.5 + 0.4j], dtype=jnp.complex64)

    point = hankel1(nu[0], z[0], method="direct")
    batch = jnp.asarray([hankel1(a, b, method="direct") for a, b in zip(nu, z)])

    assert jnp.isfinite(jnp.real(point))
    assert batch.shape == (2,)
    assert jnp.allclose(batch[0], point, rtol=1e-5, atol=1e-5)
