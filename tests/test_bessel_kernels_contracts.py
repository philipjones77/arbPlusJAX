import jax.numpy as jnp

from arbplusjax import bessel_kernels as bk


def test_real_bessel_eval_switches_to_asymptotic_branch_on_large_positive_inputs():
    nu = jnp.asarray(0.75, dtype=jnp.float64)
    z = jnp.asarray(20.0, dtype=jnp.float64)

    assert jnp.allclose(bk.real_bessel_eval_j(nu, z), bk.real_bessel_asym_j(nu, z))
    assert jnp.allclose(bk.real_bessel_eval_y(nu, z), bk.real_bessel_asym_y(nu, z))
    assert jnp.allclose(bk.real_bessel_eval_i(nu, z), bk.real_bessel_asym_i(nu, z))
    assert jnp.allclose(bk.real_bessel_eval_k(nu, z), bk.real_bessel_asym_k(nu, z))


def test_real_and_complex_bessel_series_agree_on_real_slice():
    nu = jnp.asarray(0.4, dtype=jnp.float64)
    z = jnp.asarray(1.75, dtype=jnp.float64)

    real_j = bk.real_bessel_series(nu, z, -1.0)
    real_i = bk.real_bessel_series(nu, z, 1.0)
    complex_j = bk.complex_bessel_series(nu, z, -1.0)
    complex_i = bk.complex_bessel_series(nu, z, 1.0)

    assert jnp.allclose(jnp.real(complex_j), real_j)
    assert jnp.allclose(jnp.real(complex_i), real_i)
    assert jnp.allclose(jnp.imag(complex_j), 0.0)
    assert jnp.allclose(jnp.imag(complex_i), 0.0)


def test_low_level_bessel_y_and_k_report_singularity_for_integer_orders():
    nu_real = jnp.asarray(1.0, dtype=jnp.float64)
    z_real = jnp.asarray(2.0, dtype=jnp.float64)
    nu_complex = jnp.asarray(1.0 + 0.0j, dtype=jnp.complex128)
    z_complex = jnp.asarray(2.0 + 0.0j, dtype=jnp.complex128)

    assert jnp.isinf(bk.real_bessel_y(nu_real, z_real))
    assert jnp.isinf(bk.real_bessel_k(nu_real, z_real))
    assert jnp.isnan(bk.complex_bessel_y(nu_complex, z_complex))
    assert jnp.isnan(bk.complex_bessel_k(nu_complex, z_complex))


def test_low_level_bessel_kernels_preserve_requested_precision_family():
    out32 = bk.real_bessel_eval_j(jnp.asarray(0.25, dtype=jnp.float32), jnp.asarray(3.0, dtype=jnp.float32))
    out64 = bk.real_bessel_eval_j(jnp.asarray(0.25, dtype=jnp.float64), jnp.asarray(3.0, dtype=jnp.float64))
    outc64 = bk.complex_bessel_k(jnp.asarray(0.5 + 0.1j, dtype=jnp.complex64), jnp.asarray(2.0 + 0.3j, dtype=jnp.complex64))
    outc128 = bk.complex_bessel_k(jnp.asarray(0.5 + 0.1j, dtype=jnp.complex128), jnp.asarray(2.0 + 0.3j, dtype=jnp.complex128))

    assert out32.dtype == jnp.float32
    assert out64.dtype == jnp.float64
    assert outc64.dtype == jnp.complex64
    assert outc128.dtype == jnp.complex128
