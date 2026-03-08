import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import double_gamma
from arbplusjax import shahen_double_gamma as shahen
from arbplusjax import double_interval as di

from tests._test_checks import _check


def _interval(x: float) -> jnp.ndarray:
    return di.interval(jnp.asarray(x, dtype=jnp.float64), jnp.asarray(x, dtype=jnp.float64))


def _box(re: float, im: float) -> jnp.ndarray:
    return acb_core.acb_box(_interval(re), _interval(im))


def test_shahen_matches_bdg_point_and_basic():
    _check(shahen.shahen_barnesgamma2 is double_gamma.bdg_barnesgamma2)
    _check(shahen.shahen_log_normalizeddoublegamma is double_gamma.bdg_log_normalizeddoublegamma)
    _check(shahen.shahen_interval_barnesgamma2_mode is double_gamma.bdg_interval_barnesgamma2_mode)
    _check(shahen.shahen_complex_barnesgamma2_mode is double_gamma.bdg_complex_barnesgamma2_mode)

    wi = di.interval(jnp.asarray(1.2, dtype=jnp.float64), jnp.asarray(1.2, dtype=jnp.float64))
    bi = di.interval(jnp.asarray(1.6, dtype=jnp.float64), jnp.asarray(1.6, dtype=jnp.float64))
    out_a = shahen.shahen_interval_barnesgamma2_mode(wi, bi, impl="basic", prec_bits=80)
    out_b = double_gamma.bdg_interval_barnesgamma2_mode(wi, bi, impl="basic", prec_bits=80)
    _check(bool(jnp.allclose(out_a, out_b)))

    wc = _box(1.2, 0.1)
    bc = _box(1.6, 0.0)
    out_ca = shahen.shahen_complex_barnesgamma2_mode(wc, bc, impl="basic", prec_bits=80)
    out_cb = double_gamma.bdg_complex_barnesgamma2_mode(wc, bc, impl="basic", prec_bits=80)
    _check(bool(jnp.allclose(out_ca, out_cb)))
