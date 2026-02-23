import jax
import jax.numpy as jnp

from arbplusjax import arb_poly
from arbplusjax import double_interval as di


from tests._test_checks import _check
def _interval(lo: float, hi: float) -> jnp.ndarray:
    return di.interval(jnp.float64(lo), jnp.float64(hi))


def test_jit_compiles_and_keeps_interval_order():
    coeffs = jnp.array(
        [
            [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]],
            [[0.0, 0.1], [0.2, 0.3], [0.1, 0.2], [0.3, 0.4]],
        ],
        dtype=jnp.float64,
    )
    x = jnp.array([[0.2, 0.3], [0.1, 0.2]], dtype=jnp.float64)
    out = arb_poly.arb_poly_eval_cubic_batch_jit(coeffs, x)
    _check(out.shape == (2, 2))
    _check(bool(jnp.all(out[:, 0] <= out[:, 1])))


def test_grad_path_for_midpoint():
    def loss(t):
        tt = jnp.asarray(t, dtype=jnp.float64)
        coeffs = jnp.stack(
            [
                _interval(0.1, 0.2),
                _interval(0.2, 0.3),
                _interval(0.1, 0.2),
                _interval(0.3, 0.4),
            ],
            axis=0,
        )
        x = di.interval(tt, tt)
        out = arb_poly.arb_poly_eval_cubic(coeffs, x)
        return di.midpoint(out)

    g = jax.grad(loss)(jnp.float64(0.2))
    _check(bool(jnp.isfinite(g)))


def test_precision_semantics_wider_at_lower_precision():
    coeffs = jnp.stack(
        [
            _interval(0.1, 0.2),
            _interval(0.2, 0.3),
            _interval(0.1, 0.2),
            _interval(0.3, 0.4),
        ],
        axis=0,
    )
    x = _interval(0.2, 0.25)
    hi = arb_poly.arb_poly_eval_cubic_prec(coeffs, x, prec_bits=53)
    lo = arb_poly.arb_poly_eval_cubic_prec(coeffs, x, prec_bits=20)
    _check(bool(di.contains(lo, hi)))
