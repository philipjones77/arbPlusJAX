import jax
import jax.numpy as jnp

from arbjax import mag


def test_jit_compiles():
    a = jnp.array([0.2, 0.5, -0.1], dtype=jnp.float64)
    b = jnp.array([1.2, -0.3, 0.4], dtype=jnp.float64)
    out = mag.mag_add_batch_jit(a, b)
    out2 = mag.mag_mul_batch_jit(a, b)
    assert out.shape == (3,)
    assert out2.shape == (3,)


def test_grad_path():
    def loss(t):
        return mag.mag_mul(t, 1.5)

    g = jax.grad(loss)(jnp.float64(0.4))
    assert bool(jnp.isfinite(g))
