import jax
import jax.numpy as jnp

from arbjax import partitions


def test_jit_compiles():
    n = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int64)
    out = partitions.partitions_p_batch_jit(n)
    assert out.shape == (5,)


def test_known_values():
    n = jnp.array([0, 1, 2, 3, 4, 5], dtype=jnp.int64)
    out = partitions.partitions_p_batch(n)
    expected = jnp.array([1, 1, 2, 3, 5, 7], dtype=jnp.int64)
    assert bool(jnp.all(out == expected))


def test_grad_path():
    def loss(t):
        return partitions.partitions_p(jnp.int64(6)).astype(jnp.float64) + 0.0 * t

    g = jax.grad(loss)(jnp.float64(0.3))
    assert bool(jnp.isfinite(g))
