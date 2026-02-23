import inspect

import jax.numpy as jnp

from arbplusjax import hypgeom
from arbplusjax import hypgeom_wrappers

from tests._test_checks import _check


def _hypgeom_callables():
    out = {}
    for name in dir(hypgeom):
        if not (name.startswith("arb_hypgeom_") or name.startswith("acb_hypgeom_")):
            continue
        obj = getattr(hypgeom, name)
        if callable(obj):
            out[name] = obj
    return out


def test_hypgeom_prec_coverage():
    callables = _hypgeom_callables()
    names = set(callables)
    missing = []
    for name in names:
        if name.endswith("_prec") or name.endswith("_jit") or name.endswith("_rigorous"):
            continue
        if name.endswith("_batch"):
            if name + "_prec" not in names:
                missing.append(name + "_prec")
            continue
        if name + "_prec" not in names:
            missing.append(name + "_prec")
    _check(len(missing) == 0)


def test_hypgeom_mode_wrappers_exist():
    callables = _hypgeom_callables()
    prec_names = [name for name in callables if name.endswith("_prec")]
    missing = []
    for name in prec_names:
        mode_name = name.replace("_prec", "_mode")
        if not hasattr(hypgeom_wrappers, mode_name):
            missing.append(mode_name)
    _check(len(missing) == 0)


def test_hypgeom_lower_level_helpers_shapes():
    xr = jnp.array([0.2, 0.3], dtype=jnp.float64)
    xc = jnp.array([0.2, 0.25, 0.1, 0.15], dtype=jnp.float64)

    coeffs = hypgeom.arb_hypgeom_rising_coeffs_1(5, 6)
    _check(coeffs.shape == (6,))
    coeffs = hypgeom.arb_hypgeom_rising_coeffs_2(5, 6)
    _check(coeffs.shape == (6,))
    coeffs = hypgeom.arb_hypgeom_rising_coeffs_fmpz(5, 6)
    _check(coeffs.shape == (6,))

    shallow = hypgeom.arb_hypgeom_gamma_coeff_shallow(2)
    _check(shallow.shape == (2,))

    bounds = hypgeom.arb_hypgeom_gamma_stirling_term_bounds(jnp.array(0.2, dtype=jnp.float64), 4)
    _check(bounds.shape == (4,))

    n1 = hypgeom.arb_hypgeom_gamma_lower_fmpq_0_choose_N(xr, xr)
    n2 = hypgeom.arb_hypgeom_gamma_upper_fmpq_inf_choose_N(xr, xr)
    n3 = hypgeom.arb_hypgeom_gamma_upper_singular_si_choose_N(2, xr)
    _check(jnp.ndim(n1) == 0 and jnp.ndim(n2) == 0 and jnp.ndim(n3) == 0)

    gl = hypgeom.arb_hypgeom_gamma_lower_fmpq_0_bsplit(xr, xr, n_terms=8)
    gu = hypgeom.arb_hypgeom_gamma_upper_fmpq_inf_bsplit(xr, xr, n_terms=8)
    gs = hypgeom.arb_hypgeom_gamma_upper_singular_si_bsplit(2, xr, n_terms=8)
    _check(gl.shape == (2,) and gu.shape == (2,) and gs.shape == (2,))

    si = hypgeom.arb_hypgeom_si_1f2(xr, n_terms=8)
    ci = hypgeom.arb_hypgeom_ci_2f3(xr, n_terms=8)
    _check(si.shape == (2,) and ci.shape == (2,))

    airy = hypgeom.acb_hypgeom_airy_series(xc, length=4)
    _check(airy.shape == (4, 4))
    ei = hypgeom.acb_hypgeom_ei_series(xc, length=4)
    _check(ei.shape == (4, 4))

    jet = hypgeom.acb_hypgeom_log_rising_ui_jet(xc, n=3, length=4)
    _check(jet.shape == (4, 4))
    jet = hypgeom.acb_hypgeom_rising_ui_jet(xc, n=3, length=4)
    _check(jet.shape == (4, 4))
    jet = hypgeom.acb_hypgeom_rising_ui_jet_powsum(xc, length=4)
    _check(jet.shape == (4, 4))


def test_hypgeom_mode_wrappers_for_helpers():
    xr = jnp.array([0.2, 0.3], dtype=jnp.float64)
    xc = jnp.array([0.2, 0.25, 0.1, 0.15], dtype=jnp.float64)

    out = hypgeom_wrappers.arb_hypgeom_si_1f2_mode(xr, impl="rigorous", prec_bits=40)
    _check(out.shape == (2,))

    n = hypgeom_wrappers.arb_hypgeom_gamma_lower_fmpq_0_choose_N_mode(xr, xr, impl="rigorous", prec_bits=40)
    _check(jnp.ndim(n) == 0)

    ai, aip, bi, bip = hypgeom_wrappers.acb_hypgeom_airy_mode(xc, impl="rigorous", prec_bits=40)
    _check(ai.shape == (4,) and aip.shape == (4,) and bi.shape == (4,) and bip.shape == (4,))

    s, c = hypgeom_wrappers.acb_hypgeom_fresnel_mode(xc, impl="adaptive", prec_bits=40)
    _check(s.shape == (4,) and c.shape == (4,))

    f, g = hypgeom_wrappers.acb_hypgeom_coulomb_mode(xc, xc, xc, impl="rigorous", prec_bits=40)
    _check(f.shape == (4,) and g.shape == (4,))

    a = jnp.array([1.1, 1.2, 0.1, 0.2], dtype=jnp.float64)
    b = jnp.array([2.1, 2.2, 0.1, 0.2], dtype=jnp.float64)
    z = jnp.array([0.2, 0.3, 0.05, 0.1], dtype=jnp.float64)
    s, t = hypgeom_wrappers.acb_hypgeom_pfq_sum_mode(a, b, z, impl="adaptive", prec_bits=40)
    _check(s.shape == (4,) and t.shape == (4,))

    p, dp = hypgeom_wrappers.acb_hypgeom_legendre_p_uiui_rec_mode(3, 0, xc, impl="rigorous", prec_bits=40)
    _check(p.shape == (4,) and dp.shape == (4,))
