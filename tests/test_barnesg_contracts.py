import jax.numpy as jnp

from arbplusjax import barnesg


def test_barnesg_log_and_value_match_on_complex_slice():
    z = jnp.asarray(1.7 + 0.25j, dtype=jnp.complex128)
    log_value = barnesg.log_barnesg(z)
    value = barnesg.barnesg(z)

    assert jnp.allclose(jnp.exp(log_value), value)


def test_barnesg_recurrence_matches_gamma_relation():
    z = jnp.asarray(1.8 + 0.2j, dtype=jnp.complex128)
    lhs = barnesg.barnesg(z + 1.0)
    rhs = jnp.exp(barnesg._complex_loggamma(z)) * barnesg.barnesg(z)

    assert jnp.allclose(lhs, rhs, rtol=1e-8, atol=1e-8)


def test_barnesg_real_and_complex_wrappers_mark_nonpositive_integer_poles():
    assert jnp.isnan(barnesg.barnesg_real(jnp.asarray(0.0, dtype=jnp.float64)))
    assert jnp.isnan(barnesg.barnesg_real(jnp.asarray(-2.0, dtype=jnp.float64)))

    complex_zero = barnesg.barnesg_complex(jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
    complex_neg = barnesg.barnesg_complex(jnp.asarray(-3.0 + 0.0j, dtype=jnp.complex128))

    assert jnp.isnan(complex_zero)
    assert jnp.isnan(complex_neg)


def test_barnesg_preserves_real_and_complex_dtype_families():
    out_real32 = barnesg.barnesg_real(jnp.asarray(2.5, dtype=jnp.float32))
    out_real64 = barnesg.barnesg_real(jnp.asarray(2.5, dtype=jnp.float64))
    out_complex64 = barnesg.barnesg_complex(jnp.asarray(1.5 + 0.1j, dtype=jnp.complex64))
    out_complex128 = barnesg.barnesg_complex(jnp.asarray(1.5 + 0.1j, dtype=jnp.complex128))

    assert out_real32.dtype == jnp.float32
    assert out_real64.dtype == jnp.float64
    assert out_complex64.dtype == jnp.complex64
    assert out_complex128.dtype == jnp.complex128
