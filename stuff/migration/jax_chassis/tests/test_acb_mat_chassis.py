import jax
import jax.numpy as jnp

from arbjax import acb_core
from arbjax import acb_mat
from arbjax import double_interval as di


def _interval(lo: float, hi: float) -> jnp.ndarray:
    return di.interval(jnp.float64(lo), jnp.float64(hi))


def _box(re_lo: float, re_hi: float, im_lo: float, im_hi: float) -> jnp.ndarray:
    return acb_core.acb_box(_interval(re_lo, re_hi), _interval(im_lo, im_hi))


def test_jit_compiles_and_keeps_interval_order():
    m = jnp.array(
        [
            [
                [[0.1, 0.2, 0.0, 0.1], [0.0, 0.1, -0.1, 0.0]],
                [[0.2, 0.3, 0.05, 0.1], [0.4, 0.5, -0.2, -0.1]],
            ],
            [
                [[-0.2, -0.1, 0.1, 0.2], [0.3, 0.4, 0.0, 0.1]],
                [[0.1, 0.2, -0.1, 0.0], [0.6, 0.7, 0.2, 0.3]],
            ],
        ],
        dtype=jnp.float64,
    )
    out = acb_mat.acb_mat_2x2_det_batch_jit(m)
    assert out.shape == (2, 4)
    assert bool(jnp.all(out[:, 0] <= out[:, 1]))
    assert bool(jnp.all(out[:, 2] <= out[:, 3]))


def test_grad_path_for_midpoint():
    def loss(t):
        tt = jnp.asarray(t, dtype=jnp.float64)
        a00 = acb_core.acb_box(di.interval(tt, tt), di.interval(jnp.float64(0.1), jnp.float64(0.1)))
        a01 = _box(0.1, 0.2, 0.0, 0.1)
        a10 = _box(0.0, 0.1, -0.1, 0.0)
        a11 = _box(0.3, 0.4, 0.0, 0.1)
        mat = jnp.stack([jnp.stack([a00, a01], axis=0), jnp.stack([a10, a11], axis=0)], axis=0)
        out = acb_mat.acb_mat_2x2_trace(mat)
        return jnp.real(acb_core.acb_midpoint(out))

    g = jax.grad(loss)(jnp.float64(0.2))
    assert bool(jnp.isfinite(g))


def test_precision_semantics_wider_at_lower_precision():
    a00 = _box(0.1, 0.2, 0.0, 0.1)
    a01 = _box(0.0, 0.1, -0.1, 0.0)
    a10 = _box(0.2, 0.3, 0.05, 0.1)
    a11 = _box(0.4, 0.5, -0.2, -0.1)
    mat = jnp.stack([jnp.stack([a00, a01], axis=0), jnp.stack([a10, a11], axis=0)], axis=0)
    hi = acb_mat.acb_mat_2x2_det_prec(mat, prec_bits=53)
    lo = acb_mat.acb_mat_2x2_det_prec(mat, prec_bits=20)
    assert bool(di.contains(acb_core.acb_real(lo), acb_core.acb_real(hi)))
    assert bool(di.contains(acb_core.acb_imag(lo), acb_core.acb_imag(hi)))
