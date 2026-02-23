import jax
import jax.numpy as jnp

from arbplusjax import arf


from tests._test_checks import _check
def test_jit_compiles():
    a = jnp.array([0.2, 0.5, -0.1], dtype=jnp.float64)
    b = jnp.array([1.2, -0.3, 0.4], dtype=jnp.float64)
    out = arf.arf_add_batch_jit(a, b)
    _check(out.shape == (3,))
    out2 = arf.arf_mul_batch_jit(a, b)
    _check(out2.shape == (3,))


def test_grad_path():
    def loss(t):
        return arf.arf_mul(t, 1.5)

    g = jax.grad(loss)(jnp.float64(0.4))
    _check(bool(jnp.isfinite(g)))
