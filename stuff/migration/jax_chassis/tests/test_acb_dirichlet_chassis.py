import jax
import jax.numpy as jnp

from arbjax import acb_core
from arbjax import acb_dirichlet
from arbjax import double_interval as di


def _interval(lo: float, hi: float) -> jnp.ndarray:
    return di.interval(jnp.float64(lo), jnp.float64(hi))


def _box(re_lo: float, re_hi: float, im_lo: float, im_hi: float) -> jnp.ndarray:
    return acb_core.acb_box(_interval(re_lo, re_hi), _interval(im_lo, im_hi))


def test_jit_compiles_and_keeps_interval_order():
    s = jnp.array(
        [
            [2.0, 2.1, 0.1, 0.2],
            [1.5, 1.6, -0.3, -0.2],
            [0.8, 1.0, 0.05, 0.1],
        ],
        dtype=jnp.float64,
    )
    out = acb_dirichlet.acb_dirichlet_zeta_batch_jit(s, n_terms=48)
    assert out.shape == (3, 4)
    assert bool(jnp.all(out[:, 0] <= out[:, 1]))
    assert bool(jnp.all(out[:, 2] <= out[:, 3]))


def test_grad_path_for_midpoint():
    def loss(t):
        tt = jnp.asarray(t, dtype=jnp.float64)
        s = acb_core.acb_box(di.interval(tt, tt), di.interval(jnp.float64(0.2), jnp.float64(0.2)))
        out = acb_dirichlet.acb_dirichlet_zeta(s, n_terms=64)
        return jnp.real(acb_core.acb_midpoint(out))

    g = jax.grad(loss)(jnp.float64(2.2))
    assert bool(jnp.isfinite(g))


def test_precision_semantics_wider_at_lower_precision():
    s = _box(1.2, 1.25, 0.1, 0.2)
    hi = acb_dirichlet.acb_dirichlet_eta_prec(s, n_terms=48, prec_bits=53)
    lo = acb_dirichlet.acb_dirichlet_eta_prec(s, n_terms=48, prec_bits=20)
    assert bool(di.contains(acb_core.acb_real(lo), acb_core.acb_real(hi)))
    assert bool(di.contains(acb_core.acb_imag(lo), acb_core.acb_imag(hi)))
