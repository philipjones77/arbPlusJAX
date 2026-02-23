import jax
import jax.numpy as jnp

from arbjax import arb_fpwrap


def test_jit_compiles():
    x = jnp.array([0.2, 0.5, 1.0], dtype=jnp.float64)
    out = arb_fpwrap.arb_fpwrap_double_exp_jit(x)
    assert out.shape == (3,)
    z = jnp.array([0.2 + 0.3j, 0.1 - 0.2j], dtype=jnp.complex128)
    out2 = arb_fpwrap.arb_fpwrap_cdouble_log_jit(z)
    assert out2.shape == (2,)


def test_grad_path():
    def loss(t):
        return jnp.real(arb_fpwrap.arb_fpwrap_cdouble_exp(t + 0.2j))

    g = jax.grad(loss)(jnp.float64(0.1))
    assert bool(jnp.isfinite(g))
