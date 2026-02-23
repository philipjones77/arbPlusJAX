import jax
import jax.numpy as jnp

from arbjax import bernoulli


def test_jit_compiles():
    n = jnp.array([0, 1, 2, 4, 6], dtype=jnp.int64)
    out = bernoulli.bernoulli_number_batch_jit(n)
    assert out.shape == (5,)


def test_grad_path():
    def loss(t):
        n = jnp.asarray(t, dtype=jnp.float64)
        return bernoulli.bernoulli_number(jnp.asarray(0)) + 0.0 * n

    g = jax.grad(loss)(jnp.float64(0.2))
    assert bool(jnp.isfinite(g))
