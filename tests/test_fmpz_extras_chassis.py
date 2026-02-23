import jax
import jax.numpy as jnp

from arbplusjax import fmpz_extras


from tests._test_checks import _check
def test_jit_compiles():
    a = jnp.array([1, 2, 3], dtype=jnp.int64)
    b = jnp.array([4, 5, 6], dtype=jnp.int64)
    out = fmpz_extras.fmpz_extras_add_batch_jit(a, b)
    out2 = fmpz_extras.fmpz_extras_mul_batch_jit(a, b)
    _check(out.shape == (3,))
    _check(out2.shape == (3,))


def test_grad_path():
    def loss(t):
        a = jnp.array([1, 2], dtype=jnp.int64)
        b = jnp.array([3, 4], dtype=jnp.int64)
        return jnp.sum(fmpz_extras.fmpz_extras_add(a, b)).astype(jnp.float64) + 0.0 * t

    g = jax.grad(loss)(jnp.float64(0.1))
    _check(bool(jnp.isfinite(g)))
