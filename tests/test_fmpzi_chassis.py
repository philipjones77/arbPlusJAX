import jax
import jax.numpy as jnp

from arbplusjax import fmpzi


from tests._test_checks import _check
def test_jit_compiles():
    a = fmpzi.interval(jnp.array([1, 3], dtype=jnp.int64), jnp.array([2, 4], dtype=jnp.int64))
    b = fmpzi.interval(jnp.array([5, 7], dtype=jnp.int64), jnp.array([6, 8], dtype=jnp.int64))
    out = fmpzi.fmpzi_add_batch_jit(a, b)
    out2 = fmpzi.fmpzi_sub_batch_jit(a, b)
    _check(out.shape == (2, 2))
    _check(out2.shape == (2, 2))


def test_grad_path():
    def loss(t):
        a = fmpzi.interval(jnp.array([1], dtype=jnp.int64), jnp.array([2], dtype=jnp.int64))
        b = fmpzi.interval(jnp.array([3], dtype=jnp.int64), jnp.array([4], dtype=jnp.int64))
        return jnp.sum(fmpzi.fmpzi_add(a, b)).astype(jnp.float64) + 0.0 * t

    g = jax.grad(loss)(jnp.float64(0.1))
    _check(bool(jnp.isfinite(g)))
