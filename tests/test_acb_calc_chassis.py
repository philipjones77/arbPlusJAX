import jax
import jax.numpy as jnp

from arbplusjax import acb_calc
from arbplusjax import acb_core
from arbplusjax import double_interval as di


from tests._test_checks import _check
def _interval(lo: float, hi: float) -> jnp.ndarray:
    return di.interval(jnp.float64(lo), jnp.float64(hi))


def _box(re_lo: float, re_hi: float, im_lo: float, im_hi: float) -> jnp.ndarray:
    return acb_core.acb_box(_interval(re_lo, re_hi), _interval(im_lo, im_hi))


def test_jit_compiles_and_keeps_interval_order():
    a = jnp.array(
        [
            [0.0, 0.1, 0.0, 0.1],
            [0.2, 0.3, -0.1, 0.0],
            [-0.4, -0.2, 0.1, 0.2],
        ],
        dtype=jnp.float64,
    )
    b = jnp.array(
        [
            [0.9, 1.0, 0.2, 0.3],
            [0.5, 0.6, -0.2, -0.1],
            [0.1, 0.2, 0.4, 0.5],
        ],
        dtype=jnp.float64,
    )
    out = acb_calc.acb_calc_integrate_line_batch_jit(a, b, integrand="exp", n_steps=32)
    _check(out.shape == (3, 4))
    _check(bool(jnp.all(out[:, 0] <= out[:, 1])))
    _check(bool(jnp.all(out[:, 2] <= out[:, 3])))


def test_grad_path_for_midpoint():
    def loss(t):
        tt = jnp.asarray(t, dtype=jnp.float64)
        a = acb_core.acb_box(di.interval(tt, tt), di.interval(jnp.float64(0.0), jnp.float64(0.0)))
        b = _box(1.0, 1.0, 0.0, 0.0)
        out = acb_calc.acb_calc_integrate_line(a, b, integrand="exp", n_steps=32)
        return jnp.real(acb_core.acb_midpoint(out))

    g = jax.grad(loss)(jnp.float64(0.2))
    _check(bool(jnp.isfinite(g)))


def test_precision_semantics_wider_at_lower_precision():
    a = _box(0.0, 0.1, -0.05, 0.05)
    b = _box(0.5, 0.6, 0.1, 0.2)
    hi = acb_calc.acb_calc_integrate_line_prec(a, b, integrand="sin", n_steps=32, prec_bits=53)
    lo = acb_calc.acb_calc_integrate_line_prec(a, b, integrand="sin", n_steps=32, prec_bits=20)
    _check(bool(di.contains(acb_core.acb_real(lo), acb_core.acb_real(hi))))
    _check(bool(di.contains(acb_core.acb_imag(lo), acb_core.acb_imag(hi))))
