import jax
import jax.numpy as jnp

from arbjax import arb_calc
from arbjax import double_interval as di


def test_jit_compiles_and_keeps_interval_order():
    a = jnp.array([[0.0, 0.1], [0.2, 0.3], [-0.2, -0.1]], dtype=jnp.float64)
    b = jnp.array([[1.0, 1.1], [0.4, 0.5], [0.1, 0.2]], dtype=jnp.float64)
    out = arb_calc.arb_calc_integrate_line_batch_jit(a, b, integrand="exp", n_steps=32)
    assert out.shape == (3, 2)
    assert bool(jnp.all(out[:, 0] <= out[:, 1]))


def test_grad_path_for_midpoint():
    def loss(t):
        tt = jnp.asarray(t, dtype=jnp.float64)
        a = di.interval(tt, tt)
        b = di.interval(jnp.float64(1.0), jnp.float64(1.0))
        out = arb_calc.arb_calc_integrate_line(a, b, integrand="sin", n_steps=32)
        return di.midpoint(out)

    g = jax.grad(loss)(jnp.float64(0.2))
    assert bool(jnp.isfinite(g))


def test_precision_semantics_wider_at_lower_precision():
    a = jnp.array([0.1, 0.2], dtype=jnp.float64)
    b = jnp.array([0.8, 0.9], dtype=jnp.float64)
    hi = arb_calc.arb_calc_integrate_line_prec(a, b, integrand="cos", n_steps=32, prec_bits=53)
    lo = arb_calc.arb_calc_integrate_line_prec(a, b, integrand="cos", n_steps=32, prec_bits=20)
    assert bool(di.contains(lo, hi))
