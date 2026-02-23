import jax
import jax.numpy as jnp

from arbplusjax import dlog


from tests._test_checks import _check
def test_jit_compiles():
    x = jnp.array([0.1, -0.25, 1.5], dtype=jnp.float64)
    out = dlog.dlog_log1p_batch_jit(x)
    _check(out.shape == (3,))


def test_grad_path():
    g = jax.grad(lambda t: dlog.dlog_log1p(t))(jnp.float64(0.2))
    _check(bool(jnp.isfinite(g)))
