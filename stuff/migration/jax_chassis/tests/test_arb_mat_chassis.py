import jax
import jax.numpy as jnp

from arbjax import arb_mat
from arbjax import double_interval as di


def _interval(lo: float, hi: float) -> jnp.ndarray:
    return di.interval(jnp.float64(lo), jnp.float64(hi))


def test_jit_compiles_and_keeps_interval_order():
    m = jnp.array(
        [
            [[[0.1, 0.2], [0.0, 0.1]], [[0.2, 0.3], [0.4, 0.5]]],
            [[[-0.2, -0.1], [0.3, 0.4]], [[0.1, 0.2], [0.6, 0.7]]],
        ],
        dtype=jnp.float64,
    )
    out = arb_mat.arb_mat_2x2_det_batch_jit(m)
    assert out.shape == (2, 2)
    assert bool(jnp.all(out[:, 0] <= out[:, 1]))


def test_grad_path_for_midpoint():
    def loss(t):
        tt = jnp.asarray(t, dtype=jnp.float64)
        a00 = _interval(tt, tt)
        a01 = _interval(0.1, 0.2)
        a10 = _interval(0.0, 0.1)
        a11 = _interval(0.3, 0.4)
        mat = jnp.stack([jnp.stack([a00, a01], axis=0), jnp.stack([a10, a11], axis=0)], axis=0)
        out = arb_mat.arb_mat_2x2_trace(mat)
        return di.midpoint(out)

    g = jax.grad(loss)(jnp.float64(0.2))
    assert bool(jnp.isfinite(g))


def test_precision_semantics_wider_at_lower_precision():
    a00 = _interval(0.1, 0.2)
    a01 = _interval(0.0, 0.1)
    a10 = _interval(0.2, 0.3)
    a11 = _interval(0.4, 0.5)
    mat = jnp.stack([jnp.stack([a00, a01], axis=0), jnp.stack([a10, a11], axis=0)], axis=0)
    hi = arb_mat.arb_mat_2x2_det_prec(mat, prec_bits=53)
    lo = arb_mat.arb_mat_2x2_det_prec(mat, prec_bits=20)
    assert bool(di.contains(lo, hi))
