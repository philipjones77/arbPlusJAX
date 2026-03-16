import jax.numpy as jnp

from arbplusjax import elementary
from arbplusjax import jax_precision

from tests._test_checks import _check


def test_safe_dot_and_norm_promote_reductions():
    a = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    b = jnp.asarray([4.0, 5.0, 6.0], dtype=jnp.float32)

    dot = jax_precision.safe_dot(a, b)
    norm = jax_precision.safe_norm(a)

    _check(dot.dtype == jnp.float64)
    _check(norm.dtype == jnp.float64)
    _check(bool(jnp.allclose(dot, 32.0, rtol=0.0, atol=0.0)))
    _check(bool(jnp.allclose(norm, jnp.sqrt(14.0), rtol=1e-12, atol=1e-12)))


def test_kahan_sum_reduces_small_tail_loss():
    x = jnp.asarray([1e16, 1.0, -1e16, 1.0], dtype=jnp.float64)
    compensated = jax_precision.kahan_sum(x)
    naive = jnp.sum(x)

    _check(bool(jnp.allclose(compensated, 1.0, rtol=0.0, atol=0.0)))
    _check(bool(jnp.allclose(naive, compensated, rtol=0.0, atol=0.0)))


def test_safe_logsumexp_matches_high_precision_path():
    x = jnp.asarray([1000.0, 999.0, 998.0], dtype=jnp.float32)
    got = elementary.logsumexp(x)
    expected = jnp.asarray(1000.0, dtype=jnp.float64) + jnp.log(
        jnp.asarray(1.0 + jnp.exp(-1.0) + jnp.exp(-2.0), dtype=jnp.float64)
    )

    _check(got.dtype == jnp.float64)
    _check(bool(jnp.allclose(got, expected, rtol=1e-8, atol=1e-8)))
