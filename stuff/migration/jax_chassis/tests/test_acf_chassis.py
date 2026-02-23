import jax
import jax.numpy as jnp

from arbjax import acf


def test_jit_compiles():
    a = jnp.array([1.0 + 2.0j, 0.5 - 0.2j], dtype=jnp.complex128)
    b = jnp.array([0.3 - 0.1j, -0.2 + 0.4j], dtype=jnp.complex128)
    out = acf.acf_add_batch_jit(a, b)
    assert out.shape == (2,)
    out2 = acf.acf_mul_batch_jit(a, b)
    assert out2.shape == (2,)


def test_grad_path():
    def loss(t):
        z = jnp.asarray(t, dtype=jnp.float64) + 0.2j
        return jnp.real(acf.acf_mul(z, 1.5 + 0.3j))

    g = jax.grad(loss)(jnp.float64(0.4))
    assert bool(jnp.isfinite(g))
