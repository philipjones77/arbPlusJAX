import jax.numpy as jnp

from arbplusjax import (
from tests._test_checks import _check
    acb_core,
    arb_core,
    core_wrappers,
    double_interval,
    double_interval_wrappers,
    dirichlet,
    dirichlet_wrappers,
    mp_mode,
    precision,
)


def test_mp_mode_uses_dps() -> None:
    x = jnp.array([[1.0, 1.0]], dtype=jnp.float64)
    with precision.workdps(30):
        expected = arb_core.arb_exp_prec(x, prec_bits=precision.dps_to_bits(30))
        got = mp_mode.arb_exp_mp(x, dps=30)
    _check(jnp.allclose(expected, got))


def test_core_wrapper_baseline_matches_prec() -> None:
    x = jnp.array([[0.25, 0.5]], dtype=jnp.float64)
    expected = arb_core.arb_exp_prec(x, prec_bits=precision.dps_to_bits(40))
    got = core_wrappers.arb_exp_mode(x, impl="baseline", dps=40)
    _check(jnp.allclose(expected, got))


def test_double_interval_modes() -> None:
    x = jnp.array([[1.0, 2.0]], dtype=jnp.float64)
    y = jnp.array([[3.0, 4.0]], dtype=jnp.float64)
    base = double_interval_wrappers.fast_add_mode(x, y, impl="baseline")
    rig = double_interval_wrappers.fast_add_mode(x, y, impl="rigorous", dps=40)
    _check(jnp.allclose(base, double_interval.fast_add(x, y)))
    _check(double_interval.contains(rig, base).all())


def test_core_wrapper_acb_rigorous_contains_baseline() -> None:
    x = jnp.array([[0.1, 0.2, -0.3, -0.1]], dtype=jnp.float64)
    base = acb_core.acb_exp_prec(x, prec_bits=precision.dps_to_bits(40))
    rig = core_wrappers.acb_exp_mode(x, impl="rigorous", dps=40)
    r_ok = double_interval.contains(acb_core.acb_real(rig), acb_core.acb_real(base)).all()
    i_ok = double_interval.contains(acb_core.acb_imag(rig), acb_core.acb_imag(base)).all()
    _check(r_ok and i_ok)


def test_dirichlet_wrapper_baseline_matches_prec() -> None:
    x = jnp.array([[1.1, 1.2]], dtype=jnp.float64)
    expected = dirichlet.dirichlet_zeta_prec(x, prec_bits=precision.dps_to_bits(40), n_terms=32)
    got = dirichlet_wrappers.dirichlet_zeta_mode(x, impl="baseline", dps=40, n_terms=32)
    _check(jnp.allclose(expected, got))
