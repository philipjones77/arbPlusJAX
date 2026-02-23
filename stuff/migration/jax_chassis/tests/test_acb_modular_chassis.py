import jax
import jax.numpy as jnp

from arbjax import acb_core
from arbjax import acb_modular
from arbjax import double_interval as di


def _interval(lo: float, hi: float) -> jnp.ndarray:
    return di.interval(jnp.float64(lo), jnp.float64(hi))


def _box(re_lo: float, re_hi: float, im_lo: float, im_hi: float) -> jnp.ndarray:
    return acb_core.acb_box(_interval(re_lo, re_hi), _interval(im_lo, im_hi))


def test_jit_compiles_and_keeps_interval_order():
    tau = jnp.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.6, 0.7],
            [0.4, 0.5, 0.8, 0.9],
        ],
        dtype=jnp.float64,
    )
    out = acb_modular.acb_modular_j_batch_jit(tau)
    assert out.shape == (3, 4)
    assert bool(jnp.all(out[:, 0] <= out[:, 1]))
    assert bool(jnp.all(out[:, 2] <= out[:, 3]))


def test_grad_path_for_midpoint():
    def loss(t):
        tt = jnp.asarray(t, dtype=jnp.float64)
        tau = acb_core.acb_box(di.interval(tt, tt), di.interval(jnp.float64(0.6), jnp.float64(0.6)))
        out = acb_modular.acb_modular_j(tau)
        return jnp.real(acb_core.acb_midpoint(out))

    g = jax.grad(loss)(jnp.float64(0.2))
    assert bool(jnp.isfinite(g))


def test_precision_semantics_wider_at_lower_precision():
    tau = _box(0.1, 0.2, 0.6, 0.7)
    hi = acb_modular.acb_modular_j_prec(tau, prec_bits=53)
    lo = acb_modular.acb_modular_j_prec(tau, prec_bits=20)
    assert bool(di.contains(acb_core.acb_real(lo), acb_core.acb_real(hi)))
    assert bool(di.contains(acb_core.acb_imag(lo), acb_core.acb_imag(hi)))
