import jax.numpy as jnp

from arbplusjax import bessel_kernels as bk
from arbplusjax.special.bessel import (
    modified_spherical_bessel_i,
    modified_spherical_bessel_i_derivative,
    modified_spherical_bessel_k,
    modified_spherical_bessel_k_derivative,
    spherical_bessel_j,
    spherical_bessel_j_derivative,
    spherical_bessel_y,
    spherical_bessel_y_derivative,
)


def _prefactor(z):
    return jnp.sqrt(jnp.asarray(jnp.pi, dtype=z.dtype) / (2.0 * z))


def test_spherical_bessel_j_matches_half_integer_cylindrical_identity():
    n = jnp.asarray(2, dtype=jnp.int32)
    z = jnp.asarray(1.7 + 0.3j, dtype=jnp.complex64)

    expected = _prefactor(z) * bk.complex_bessel_series(jnp.asarray(n, dtype=z.dtype) + 0.5, z, -1.0)

    assert jnp.allclose(spherical_bessel_j(n, z, method="series"), expected, rtol=1e-5, atol=1e-5)


def test_spherical_bessel_y_matches_half_integer_cylindrical_identity():
    n = jnp.asarray(2, dtype=jnp.int32)
    z = jnp.asarray(2.1 + 0.4j, dtype=jnp.complex64)

    expected = _prefactor(z) * bk.complex_bessel_y(jnp.asarray(n, dtype=z.dtype) + 0.5, z)

    assert jnp.allclose(spherical_bessel_y(n, z, method="recurrence"), expected, rtol=1e-5, atol=1e-5)


def test_modified_spherical_bessel_i_matches_half_integer_cylindrical_identity():
    n = jnp.asarray(3, dtype=jnp.int32)
    z = jnp.asarray(0.5 + 0.2j, dtype=jnp.complex64)

    expected = _prefactor(z) * bk.complex_bessel_series(jnp.asarray(n, dtype=z.dtype) + 0.5, z, 1.0)

    assert jnp.allclose(modified_spherical_bessel_i(n, z, method="series"), expected, rtol=1e-5, atol=1e-5)


def test_modified_spherical_bessel_k_matches_half_integer_cylindrical_identity():
    n = jnp.asarray(2, dtype=jnp.int32)
    z = jnp.asarray(2.4 + 0.1j, dtype=jnp.complex64)

    expected = _prefactor(z) * bk.complex_bessel_k(jnp.asarray(n, dtype=z.dtype) + 0.5, z)

    assert jnp.allclose(modified_spherical_bessel_k(n, z, method="recurrence"), expected, rtol=1e-5, atol=1e-5)


def test_spherical_derivatives_match_centered_difference():
    n = jnp.asarray(2, dtype=jnp.int32)
    z = jnp.asarray(1.8 + 0.25j, dtype=jnp.complex64)
    step = jnp.asarray(1e-3, dtype=jnp.float32)

    fd_j = (spherical_bessel_j(n, z + step, method="recurrence") - spherical_bessel_j(n, z - step, method="recurrence")) / (2.0 * step)
    fd_y = (spherical_bessel_y(n, z + step, method="recurrence") - spherical_bessel_y(n, z - step, method="recurrence")) / (2.0 * step)
    fd_i = (modified_spherical_bessel_i(n, z + step, method="recurrence") - modified_spherical_bessel_i(n, z - step, method="recurrence")) / (2.0 * step)
    fd_k = (modified_spherical_bessel_k(n, z + step, method="recurrence") - modified_spherical_bessel_k(n, z - step, method="recurrence")) / (2.0 * step)

    assert jnp.allclose(spherical_bessel_j_derivative(n, z, method="recurrence"), fd_j, rtol=5e-2, atol=5e-3)
    assert jnp.allclose(spherical_bessel_y_derivative(n, z, method="recurrence"), fd_y, rtol=5e-2, atol=5e-3)
    assert jnp.allclose(modified_spherical_bessel_i_derivative(n, z, method="recurrence"), fd_i, rtol=5e-2, atol=5e-3)
    assert jnp.allclose(modified_spherical_bessel_k_derivative(n, z, method="recurrence"), fd_k, rtol=5e-2, atol=5e-3)


def test_api_exposes_spherical_bessel_point_and_batch_paths():
    n = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    z = jnp.asarray([0.5 + 0.1j, 0.8 + 0.2j, 1.1 + 0.3j], dtype=jnp.complex64)

    point = spherical_bessel_j(n[1], z[1], method="auto")
    batch = jnp.asarray([spherical_bessel_j(a, b, method="auto") for a, b in zip(n, z)])

    assert batch.shape == (3,)
    assert jnp.allclose(batch[1], point, rtol=1e-5, atol=1e-5)
