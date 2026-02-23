import jax
import jax.numpy as jnp

from arbjax import dlog


def test_jit_compiles():
    x = jnp.array([0.1, -0.25, 1.5], dtype=jnp.float64)
    out = dlog.dlog_log1p_batch_jit(x)
    assert out.shape == (3,)


def test_grad_path():
    g = jax.grad(lambda t: dlog.dlog_log1p(t))(jnp.float64(0.2))
    assert bool(jnp.isfinite(g))
