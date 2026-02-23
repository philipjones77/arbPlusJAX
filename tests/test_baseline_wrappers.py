import jax
import jax.numpy as jnp

from arbplusjax import baseline_wrappers as bw
from arbplusjax import double_interval as di


from tests._test_checks import _check
def _interval(lo: float, hi: float):
    return di.interval(jnp.float64(lo), jnp.float64(hi))


def _box(re_lo: float, re_hi: float, im_lo: float, im_hi: float):
    return jnp.array([re_lo, re_hi, im_lo, im_hi], dtype=jnp.float64)


def test_baseline_modes_real():
    x = _interval(0.2, 0.3)
    for mode in ("baseline", "rigorous", "adaptive"):
        y = bw.arb_exp_mp(x, mode=mode, dps=50)
        _check(y.shape == (2,))
        _check(bool(y[0] <= y[1]))


def test_baseline_modes_complex():
    x = _box(0.2, 0.3, -0.1, 0.1)
    for mode in ("baseline", "rigorous", "adaptive"):
        y = bw.acb_exp_mp(x, mode=mode, dps=50)
        _check(y.shape == (4,))
        _check(bool(y[0] <= y[1]))
        _check(bool(y[2] <= y[3]))


def test_baseline_grad_path():
    x = _interval(0.2, 0.3)

    def loss(t):
        xt = di.interval(t, t)
        y = bw.arb_log_mp(xt, mode="baseline", dps=50)
        return jnp.sum(y)

    g = jax.grad(loss)(jnp.float64(0.25))
    _check(bool(jnp.isfinite(g)))
