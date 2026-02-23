import jax
import jax.numpy as jnp

from arbjax import dirichlet
from arbjax import double_interval as di


def test_jit_compiles():
    s = di.interval(jnp.array([1.2, 2.5, 3.5]), jnp.array([1.3, 2.6, 3.6]))
    out = dirichlet.dirichlet_zeta_batch_jit(s, n_terms=16)
    out_eta = dirichlet.dirichlet_eta_batch_jit(s, n_terms=16)
    assert out.shape == (3, 2)
    assert out_eta.shape == (3, 2)


def test_grad_path():
    def loss(t):
        s = di.interval(t, t + 0.1)
        return di.midpoint(dirichlet.dirichlet_zeta(s, n_terms=8))

    g = jax.grad(loss)(jnp.float64(2.2))
    assert bool(jnp.isfinite(g))


def test_precision_semantics():
    s = di.interval(jnp.float64(2.1), jnp.float64(2.3))
    hi_prec = dirichlet.dirichlet_zeta_prec(s, n_terms=16, prec_bits=53)
    lo_prec = dirichlet.dirichlet_zeta_prec(s, n_terms=16, prec_bits=20)
    assert bool(di.contains(lo_prec, hi_prec))
