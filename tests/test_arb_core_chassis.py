import jax
import jax.numpy as jnp

from arbplusjax import arb_core
from arbplusjax import double_interval as di


from tests._test_checks import _check
def test_jit_compiles_and_keeps_interval_order():
    x = jnp.array([[0.3, 0.4], [1.2, 1.5], [-0.7, 0.1]], dtype=jnp.float64)
    for fn in (
        arb_core.arb_exp_batch_jit,
        arb_core.arb_log_batch_jit,
        arb_core.arb_sqrt_batch_jit,
        arb_core.arb_sin_batch_jit,
        arb_core.arb_cos_batch_jit,
        arb_core.arb_tan_batch_jit,
        arb_core.arb_sinh_batch_jit,
        arb_core.arb_cosh_batch_jit,
        arb_core.arb_tanh_batch_jit,
    ):
        out = fn(x if fn not in (arb_core.arb_log_batch_jit, arb_core.arb_sqrt_batch_jit) else jnp.array([[0.3, 0.4], [1.2, 1.5], [0.01, 0.1]], dtype=jnp.float64))
        _check(out.shape == (3, 2))
        _check(bool(jnp.all(out[:, 0] <= out[:, 1])))


def test_grad_paths_on_point_intervals():
    def grad_of_midpoint(fn, x0):
        def loss(t):
            y = fn(di.interval(t, t))
            return di.midpoint(y)

        g = jax.grad(loss)(jnp.float64(x0))
        _check(bool(jnp.isfinite(g)))

    grad_of_midpoint(arb_core.arb_exp, 0.4)
    grad_of_midpoint(arb_core.arb_log, 1.3)
    grad_of_midpoint(arb_core.arb_sqrt, 1.8)
    grad_of_midpoint(arb_core.arb_sin, 0.6)
    grad_of_midpoint(arb_core.arb_cos, 0.6)
    grad_of_midpoint(arb_core.arb_tan, 0.2)
    grad_of_midpoint(arb_core.arb_sinh, 0.6)
    grad_of_midpoint(arb_core.arb_cosh, 0.6)
    grad_of_midpoint(arb_core.arb_tanh, 0.6)


def test_precision_semantics_wider_at_lower_precision():
    x = jnp.array([0.123456789, 0.123456799], dtype=jnp.float64)
    hi_exp = arb_core.arb_exp_prec(x, prec_bits=53)
    lo_exp = arb_core.arb_exp_prec(x, prec_bits=20)
    _check(bool(di.contains(lo_exp, hi_exp)))

    y = jnp.array([0.6, 0.65], dtype=jnp.float64)
    hi_sin = arb_core.arb_sin_prec(y, prec_bits=53)
    lo_sin = arb_core.arb_sin_prec(y, prec_bits=20)
    _check(bool(di.contains(lo_sin, hi_sin)))

    z = jnp.array([0.2, 0.25], dtype=jnp.float64)
    hi_tan = arb_core.arb_tan_prec(z, prec_bits=53)
    lo_tan = arb_core.arb_tan_prec(z, prec_bits=20)
    _check(bool(di.contains(lo_tan, hi_tan)))
