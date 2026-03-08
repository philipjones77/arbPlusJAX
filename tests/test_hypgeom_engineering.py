import jax
import jax.numpy as jnp

from arbplusjax import api, boost_hypgeom, cusf_compat, double_interval as di, hypgeom, hypgeom_wrappers, point_wrappers


def _allclose_or_tuple(a, b, rtol=1e-10, atol=1e-10):
    if isinstance(a, tuple):
        return all(_allclose_or_tuple(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b))
    return bool(jnp.allclose(jnp.asarray(a), jnp.asarray(b), rtol=rtol, atol=atol, equal_nan=True))


def _cast_like(ref, value):
    if isinstance(ref, tuple):
        return tuple(_cast_like(r, v) for r, v in zip(ref, value))
    arr = jnp.asarray(value)
    return arr.astype(jnp.asarray(ref).dtype)


def _iv(lo, hi):
    return di.interval(jnp.float64(lo), jnp.float64(hi))


def _iv_batch(lo0, hi0, n):
    lo = jnp.linspace(jnp.float64(lo0), jnp.float64(hi0), n)
    return di.interval(lo, lo + jnp.float64(0.05))


def _manual_pad(arg, pad_to):
    pad_count = pad_to - arg.shape[0]
    return jnp.concatenate([arg, jnp.repeat(arg[-1:], pad_count, axis=0)], axis=0)


def _pfq_real_params(n, p, start):
    base = jnp.linspace(jnp.float64(start), jnp.float64(start + 0.2), n)
    cols = [base + jnp.float64(0.1 * i) for i in range(p)]
    return jnp.stack(cols, axis=1)


def _pfq_box_params(n, p, start):
    base = jnp.linspace(jnp.float32(start), jnp.float32(start + 0.2), n)
    cols = []
    for i in range(p):
        lo = base + jnp.float32(0.1 * i)
        hi = lo + jnp.float32(0.02)
        cols.append(jnp.stack((lo, hi, jnp.zeros_like(lo), jnp.zeros_like(lo)), axis=-1))
    return jnp.stack(cols, axis=1)


def test_hypgeom_padded_batch_equivalence_first_and_second_tranche():
    unary = [
        ("hypgeom.arb_hypgeom_gamma", (_iv_batch(0.9, 1.3, 3),)),
        ("hypgeom.arb_hypgeom_erf", (_iv_batch(0.1, 0.4, 3),)),
        ("hypgeom.arb_hypgeom_ei", (_iv_batch(0.2, 0.6, 3),)),
        ("hypgeom.arb_hypgeom_dilog", (_iv_batch(0.1, 0.4, 3),)),
    ]
    binary = [
        ("hypgeom.arb_hypgeom_0f1", (_iv_batch(1.2, 1.4, 3), _iv_batch(0.1, 0.3, 3))),
        ("hypgeom.arb_hypgeom_gamma_lower", (_iv_batch(1.1, 1.4, 3), _iv_batch(0.2, 0.5, 3))),
        ("hypgeom.arb_hypgeom_gamma_upper", (_iv_batch(1.1, 1.4, 3), _iv_batch(0.2, 0.5, 3))),
    ]
    ternary = [
        ("hypgeom.arb_hypgeom_1f1", (_iv_batch(1.1, 1.3, 3), _iv_batch(2.1, 2.3, 3), _iv_batch(0.1, 0.3, 3))),
        ("hypgeom.arb_hypgeom_u", (_iv_batch(1.1, 1.3, 3), _iv_batch(2.1, 2.3, 3), _iv_batch(0.6, 1.0, 3))),
    ]
    quaternary = [
        (
            "hypgeom.arb_hypgeom_2f1",
            (_iv_batch(1.1, 1.3, 3), _iv_batch(1.2, 1.4, 3), _iv_batch(2.3, 2.5, 3), _iv_batch(0.1, 0.2, 3)),
        ),
    ]
    for name, args in unary + binary + ternary + quaternary:
        unpadded = api.eval_interval_batch(name, *args, mode="basic", pad_to=None)
        padded = api.eval_interval_batch(name, *args, mode="basic", pad_to=8)
        assert _allclose_or_tuple(unpadded, padded)

    z = _iv_batch(0.1, 0.3, 3)
    m = _iv_batch(0.0, 0.0, 3)
    a = _iv_batch(0.1, 0.2, 3)
    b = _iv_batch(0.2, 0.3, 3)
    lam = _iv_batch(0.6, 0.8, 3)
    leg_u = hypgeom.arb_hypgeom_legendre_p_batch(2, m, z)
    leg_p = hypgeom.arb_hypgeom_legendre_p_batch(2, _manual_pad(m, 8), _manual_pad(z, 8))[:3]
    jac_u = hypgeom.arb_hypgeom_jacobi_p_batch(2, a, b, z)
    jac_p = hypgeom.arb_hypgeom_jacobi_p_batch(2, _manual_pad(a, 8), _manual_pad(b, 8), _manual_pad(z, 8))[:3]
    geg_u = hypgeom.arb_hypgeom_gegenbauer_c_batch(2, lam, z)
    geg_p = hypgeom.arb_hypgeom_gegenbauer_c_batch(2, _manual_pad(lam, 8), _manual_pad(z, 8))[:3]
    assert _allclose_or_tuple(leg_u, leg_p)
    assert _allclose_or_tuple(jac_u, jac_p)
    assert _allclose_or_tuple(geg_u, geg_p)


def test_hypgeom_basic_batch_fastpaths_match_direct_entry_points():
    a = _iv_batch(1.1, 1.3, 3)
    b = _iv_batch(2.1, 2.3, 3)
    c = _iv_batch(2.8, 3.0, 3)
    z = _iv_batch(0.1, 0.3, 3)
    lam = _iv_batch(0.6, 0.8, 3)
    m = _iv_batch(0.0, 0.0, 3)
    pfq_a = _pfq_real_params(3, 2, 0.6)
    pfq_b = _pfq_real_params(3, 1, 1.4)

    cases = [
        (
            "hypgeom.arb_hypgeom_0f1",
            (a, z),
            hypgeom.arb_hypgeom_0f1_batch_padded_prec(a, z, pad_to=8),
        ),
        (
            "hypgeom.arb_hypgeom_gamma_lower",
            (a, z),
            hypgeom.arb_hypgeom_gamma_lower_batch_padded_prec(a, z, pad_to=8, prec_bits=53),
        ),
        (
            "hypgeom.arb_hypgeom_gamma_upper",
            (a, z),
            hypgeom.arb_hypgeom_gamma_upper_batch_padded_prec(a, z, pad_to=8, prec_bits=53),
        ),
        (
            "hypgeom.arb_hypgeom_1f1",
            (a, b, z),
            hypgeom.arb_hypgeom_1f1_batch_padded_prec(a, b, z, pad_to=8),
        ),
        (
            "hypgeom.arb_hypgeom_2f1",
            (a, b, c, z),
            hypgeom.arb_hypgeom_2f1_batch_padded_prec(a, b, c, z, pad_to=8),
        ),
        (
            "hypgeom.arb_hypgeom_u",
            (a, b, _iv_batch(0.6, 1.0, 3)),
            hypgeom.arb_hypgeom_u_batch_padded_prec(a, b, _iv_batch(0.6, 1.0, 3), pad_to=8),
        ),
        (
            "hypgeom.arb_hypgeom_legendre_p",
            (2, m, z),
            hypgeom.arb_hypgeom_legendre_p_batch_padded_prec(2, m, z, pad_to=8),
        ),
        (
            "hypgeom.arb_hypgeom_jacobi_p",
            (2, a, b, z),
            hypgeom.arb_hypgeom_jacobi_p_batch_padded_prec(2, a, b, z, pad_to=8),
        ),
        (
            "hypgeom.arb_hypgeom_gegenbauer_c",
            (2, lam, z),
            hypgeom.arb_hypgeom_gegenbauer_c_batch_padded_prec(2, lam, z, pad_to=8),
        ),
        (
            "hypgeom.arb_hypgeom_chebyshev_t",
            (2, z),
            hypgeom.arb_hypgeom_chebyshev_t_batch_padded_prec(2, z, pad_to=8),
        ),
        (
            "hypgeom.arb_hypgeom_chebyshev_u",
            (2, z),
            hypgeom.arb_hypgeom_chebyshev_u_batch_padded_prec(2, z, pad_to=8),
        ),
        (
            "hypgeom.arb_hypgeom_laguerre_l",
            (2, a, z),
            hypgeom.arb_hypgeom_laguerre_l_batch_padded_prec(2, a, z, pad_to=8),
        ),
        (
            "hypgeom.arb_hypgeom_hermite_h",
            (2, z),
            hypgeom.arb_hypgeom_hermite_h_batch_padded_prec(2, z, pad_to=8),
        ),
        (
            "hypgeom.arb_hypgeom_pfq",
            (pfq_a, pfq_b, z),
            hypgeom.arb_hypgeom_pfq_batch_padded_prec(pfq_a, pfq_b, z, pad_to=8),
        ),
    ]

    for name, args, direct in cases:
        via_api = api.eval_interval_batch(name, *args, mode="basic", pad_to=8)
        assert _allclose_or_tuple(via_api, direct)


def test_hypgeom_complex_basic_batch_fastpaths_match_direct_entry_points():
    def _box_batch(lo_re, hi_re, n):
        lo = jnp.linspace(jnp.float32(lo_re), jnp.float32(hi_re), n)
        hi = lo + jnp.float32(0.05)
        return jnp.stack((lo, hi, jnp.zeros_like(lo), jnp.zeros_like(lo)), axis=-1)

    a = _box_batch(1.1, 1.3, 3)
    b = _box_batch(2.1, 2.3, 3)
    c = _box_batch(2.8, 3.0, 3)
    z = _box_batch(0.1, 0.3, 3)
    pfq_a = _pfq_box_params(3, 2, 0.6)
    pfq_b = _pfq_box_params(3, 1, 1.4)

    cases = [
        (
            "hypgeom.acb_hypgeom_0f1",
            (a, z),
            hypgeom.acb_hypgeom_0f1_batch_padded_prec(a, z, pad_to=8, prec_bits=53),
        ),
        (
            "hypgeom.acb_hypgeom_gamma_lower",
            (a, z),
            hypgeom.acb_hypgeom_gamma_lower_batch_padded_prec(a, z, pad_to=8, prec_bits=53),
        ),
        (
            "hypgeom.acb_hypgeom_gamma_upper",
            (a, z),
            hypgeom.acb_hypgeom_gamma_upper_batch_padded_prec(a, z, pad_to=8, prec_bits=53),
        ),
        (
            "hypgeom.acb_hypgeom_1f1",
            (a, b, z),
            hypgeom.acb_hypgeom_1f1_batch_padded_prec(a, b, z, pad_to=8, prec_bits=53),
        ),
        (
            "hypgeom.acb_hypgeom_2f1",
            (a, b, c, z),
            hypgeom.acb_hypgeom_2f1_batch_padded_prec(a, b, c, z, pad_to=8, prec_bits=53),
        ),
        (
            "hypgeom.acb_hypgeom_u",
            (a, b, _box_batch(0.6, 1.0, 3)),
            hypgeom.acb_hypgeom_u_batch_padded_prec(a, b, _box_batch(0.6, 1.0, 3), pad_to=8, prec_bits=53),
        ),
        (
            "hypgeom.acb_hypgeom_chebyshev_t",
            (2, z),
            hypgeom.acb_hypgeom_chebyshev_t_batch_padded_prec(2, z, pad_to=8, prec_bits=53),
        ),
        (
            "hypgeom.acb_hypgeom_chebyshev_u",
            (2, z),
            hypgeom.acb_hypgeom_chebyshev_u_batch_padded_prec(2, z, pad_to=8, prec_bits=53),
        ),
        (
            "hypgeom.acb_hypgeom_laguerre_l",
            (2, a, z),
            hypgeom.acb_hypgeom_laguerre_l_batch_padded_prec(2, a, z, pad_to=8, prec_bits=53),
        ),
        (
            "hypgeom.acb_hypgeom_hermite_h",
            (2, z),
            hypgeom.acb_hypgeom_hermite_h_batch_padded_prec(2, z, pad_to=8, prec_bits=53),
        ),
        (
            "hypgeom.acb_hypgeom_pfq",
            (pfq_a, pfq_b, z),
            hypgeom.acb_hypgeom_pfq_batch_padded_prec(pfq_a, pfq_b, z, pad_to=8, prec_bits=53),
        ),
    ]

    for name, args, direct in cases:
        via_api = api.eval_interval_batch(name, *args, mode="basic", pad_to=8, dtype="float32", prec_bits=53)
        assert _allclose_or_tuple(via_api, _cast_like(via_api, direct))


def test_boost_hypgeom_batch_fastpaths_match_direct_entry_points():
    a = _iv_batch(1.1, 1.3, 3)
    b = _iv_batch(2.1, 2.3, 3)
    c = _iv_batch(2.8, 3.0, 3)
    z = _iv_batch(0.1, 0.3, 3)
    pfq_a = _pfq_real_params(3, 2, 0.6)
    pfq_b = _pfq_real_params(3, 1, 1.4)

    basic_cases = [
        (
            "boost_hypergeometric_0f1",
            (a, z),
            boost_hypgeom.boost_hypergeometric_0f1_batch_padded_prec(a, z, pad_to=8, prec_bits=53),
        ),
        (
            "boost_hypergeometric_1f1",
            (a, b, z),
            boost_hypgeom.boost_hypergeometric_1f1_batch_padded_prec(a, b, z, pad_to=8, prec_bits=53),
        ),
        (
            "boost_hyp2f1_series",
            (a, b, c, z),
            boost_hypgeom.boost_hyp2f1_series_batch_padded_prec(a, b, c, z, pad_to=8, prec_bits=53),
        ),
        (
            "boost_hypergeometric_pfq",
            (pfq_a, pfq_b, z),
            boost_hypgeom.boost_hypergeometric_pfq_batch_padded_prec(pfq_a, pfq_b, z, pad_to=8, prec_bits=53),
        ),
    ]
    for name, args, direct in basic_cases:
        via_api = api.eval_interval_batch(name, *args, mode="basic", pad_to=8, prec_bits=53)
        assert _allclose_or_tuple(via_api, direct)

    mode_cases = [
        (
            "boost_hypergeometric_0f1",
            (a, z),
            boost_hypgeom.boost_hypergeometric_0f1_batch_mode_padded(a, z, pad_to=8, impl="rigorous", prec_bits=53),
        ),
        (
            "boost_hypergeometric_1f1",
            (a, b, z),
            boost_hypgeom.boost_hypergeometric_1f1_batch_mode_padded(a, b, z, pad_to=8, impl="rigorous", prec_bits=53),
        ),
        (
            "boost_hyp2f1_series",
            (a, b, c, z),
            boost_hypgeom.boost_hyp2f1_series_batch_mode_padded(a, b, c, z, pad_to=8, impl="rigorous", prec_bits=53),
        ),
        (
            "boost_hypergeometric_pfq",
            (pfq_a, pfq_b, z),
            boost_hypgeom.boost_hypergeometric_pfq_batch_mode_padded(
                pfq_a, pfq_b, z, pad_to=8, impl="rigorous", prec_bits=53
            ),
        ),
    ]
    for name, args, direct in mode_cases:
        via_api = api.eval_interval_batch(name, *args, mode="rigorous", pad_to=8, prec_bits=53)
        assert _allclose_or_tuple(via_api, direct)


def test_hypgeom_and_boost_point_batch_fastpaths_match_batch_midpoints():
    a = _iv_batch(1.1, 1.3, 3)
    b = _iv_batch(2.1, 2.3, 3)
    c = _iv_batch(2.8, 3.0, 3)
    z = _iv_batch(0.1, 0.3, 3)
    u_z = _iv_batch(0.6, 1.0, 3)
    pfq_a = _pfq_real_params(3, 2, 0.6)
    pfq_b = _pfq_real_params(3, 1, 1.4)

    cases = [
        (
            "hypgeom.arb_hypgeom_1f1",
            (a, b, z),
            point_wrappers.arb_hypgeom_1f1_point(a, b, z),
        ),
        (
            "hypgeom.arb_hypgeom_2f1",
            (a, b, c, z),
            point_wrappers.arb_hypgeom_2f1_point(a, b, c, z),
        ),
        (
            "hypgeom.arb_hypgeom_u",
            (a, b, u_z),
            point_wrappers.arb_hypgeom_u_point(a, b, u_z),
        ),
        (
            "boost_hypergeometric_1f1",
            (a, b, z),
            boost_hypgeom.boost_hypergeometric_1f1(a, b, z, mode="point"),
        ),
        (
            "boost_hyp2f1_series",
            (a, b, c, z),
            boost_hypgeom.boost_hyp2f1_series(a, b, c, z, mode="point"),
        ),
        (
            "boost_hypergeometric_pfq",
            (pfq_a, pfq_b, z),
            boost_hypgeom.boost_hypergeometric_pfq(pfq_a, pfq_b, z, mode="point"),
        ),
    ]
    for name, args, direct in cases:
        via_api = api.eval_point_batch(name, *args, pad_to=8)
        assert _allclose_or_tuple(via_api, direct)


def test_hypgeom_mode_batch_fastpaths_match_api_for_rigorous_and_adaptive():
    a = _iv_batch(1.1, 1.3, 3)
    b = _iv_batch(2.1, 2.3, 3)
    c = _iv_batch(2.8, 3.0, 3)
    z = _iv_batch(0.1, 0.3, 3)
    m = _iv_batch(0.0, 0.0, 3)
    pfq_a = _pfq_real_params(3, 2, 0.6)
    pfq_b = _pfq_real_params(3, 1, 1.4)

    cases = [
        ("hypgeom.arb_hypgeom_0f1", "rigorous", (a, z), hypgeom_wrappers.arb_hypgeom_0f1_batch_mode_padded(a, z, pad_to=8, impl="rigorous", prec_bits=53)),
        ("hypgeom.arb_hypgeom_gamma_lower", "rigorous", (a, z), hypgeom_wrappers.arb_hypgeom_gamma_lower_batch_mode_padded(a, z, pad_to=8, impl="rigorous", prec_bits=53)),
        ("hypgeom.arb_hypgeom_gamma_upper", "adaptive", (a, z), hypgeom_wrappers.arb_hypgeom_gamma_upper_batch_mode_padded(a, z, pad_to=8, impl="adaptive", prec_bits=53)),
        ("hypgeom.arb_hypgeom_1f1", "rigorous", (a, b, z), hypgeom_wrappers.arb_hypgeom_1f1_batch_mode_padded(a, b, z, pad_to=8, impl="rigorous", prec_bits=53)),
        ("hypgeom.arb_hypgeom_2f1", "rigorous", (a, b, c, z), hypgeom_wrappers.arb_hypgeom_2f1_batch_mode_padded(a, b, c, z, pad_to=8, impl="rigorous", prec_bits=53)),
        ("hypgeom.arb_hypgeom_u", "adaptive", (a, b, _iv_batch(0.6, 1.0, 3)), hypgeom_wrappers.arb_hypgeom_u_batch_mode_padded(a, b, _iv_batch(0.6, 1.0, 3), pad_to=8, impl="adaptive", prec_bits=53)),
        ("hypgeom.arb_hypgeom_legendre_p", "rigorous", (2, m, z), hypgeom_wrappers.arb_hypgeom_legendre_p_batch_mode_padded(2, m, z, pad_to=8, impl="rigorous", prec_bits=53)),
        ("hypgeom.arb_hypgeom_jacobi_p", "adaptive", (2, a, b, z), hypgeom_wrappers.arb_hypgeom_jacobi_p_batch_mode_padded(2, a, b, z, pad_to=8, impl="adaptive", prec_bits=53)),
        ("hypgeom.arb_hypgeom_chebyshev_t", "rigorous", (2, z), hypgeom_wrappers.arb_hypgeom_chebyshev_t_batch_mode_padded(2, z, pad_to=8, impl="rigorous", prec_bits=53)),
        ("hypgeom.arb_hypgeom_chebyshev_u", "adaptive", (2, z), hypgeom_wrappers.arb_hypgeom_chebyshev_u_batch_mode_padded(2, z, pad_to=8, impl="adaptive", prec_bits=53)),
        ("hypgeom.arb_hypgeom_laguerre_l", "rigorous", (2, a, z), hypgeom_wrappers.arb_hypgeom_laguerre_l_batch_mode_padded(2, a, z, pad_to=8, impl="rigorous", prec_bits=53)),
        ("hypgeom.arb_hypgeom_hermite_h", "adaptive", (2, z), hypgeom_wrappers.arb_hypgeom_hermite_h_batch_mode_padded(2, z, pad_to=8, impl="adaptive", prec_bits=53)),
        ("hypgeom.arb_hypgeom_pfq", "adaptive", (pfq_a, pfq_b, z), hypgeom_wrappers.arb_hypgeom_pfq_batch_mode_padded(pfq_a, pfq_b, z, pad_to=8, impl="adaptive", prec_bits=53)),
    ]
    for name, mode, args, direct in cases:
        via_api = api.eval_interval_batch(name, *args, mode=mode, pad_to=8, prec_bits=53)
        assert _allclose_or_tuple(via_api, direct)


def test_hypgeom_incomplete_gamma_regularized_mode_fastpaths_match_api():
    a = _iv_batch(1.1, 1.3, 3)
    z = _iv_batch(0.2, 0.5, 3)
    real_cases = [
        (
            "hypgeom.arb_hypgeom_gamma_lower",
            "rigorous",
            hypgeom_wrappers.arb_hypgeom_gamma_lower_batch_mode_padded(
                a, z, pad_to=8, impl="rigorous", prec_bits=53, regularized=True
            ),
        ),
        (
            "hypgeom.arb_hypgeom_gamma_upper",
            "adaptive",
            hypgeom_wrappers.arb_hypgeom_gamma_upper_batch_mode_padded(
                a, z, pad_to=8, impl="adaptive", prec_bits=53, regularized=True
            ),
        ),
    ]
    for name, mode, direct in real_cases:
        via_api = api.eval_interval_batch(name, a, z, mode=mode, pad_to=8, prec_bits=53, regularized=True)
        assert _allclose_or_tuple(via_api, direct)

    def _box_batch(lo_re, hi_re, n):
        lo = jnp.linspace(jnp.float32(lo_re), jnp.float32(hi_re), n)
        hi = lo + jnp.float32(0.05)
        return jnp.stack((lo, hi, jnp.zeros_like(lo), jnp.zeros_like(lo)), axis=-1)

    ac = _box_batch(1.1, 1.3, 3)
    zc = _box_batch(0.2, 0.5, 3)
    complex_cases = [
        (
            "hypgeom.acb_hypgeom_gamma_lower",
            "rigorous",
            hypgeom_wrappers.acb_hypgeom_gamma_lower_batch_mode_padded(
                ac, zc, pad_to=8, impl="rigorous", prec_bits=53, regularized=True
            ),
        ),
        (
            "hypgeom.acb_hypgeom_gamma_upper",
            "adaptive",
            hypgeom_wrappers.acb_hypgeom_gamma_upper_batch_mode_padded(
                ac, zc, pad_to=8, impl="adaptive", prec_bits=53, regularized=True
            ),
        ),
    ]
    for name, mode, direct in complex_cases:
        via_api = api.eval_interval_batch(name, ac, zc, mode=mode, pad_to=8, prec_bits=53, regularized=True, dtype="float32")
        assert _allclose_or_tuple(via_api, _cast_like(via_api, direct))


def test_hypgeom_incomplete_gamma_regularized_complement_consistency():
    s = di.interval(jnp.float64(1.2), jnp.float64(1.3))
    z = di.interval(jnp.float64(0.3), jnp.float64(0.35))
    lower = hypgeom_wrappers.arb_hypgeom_gamma_lower_mode(s, z, impl="rigorous", prec_bits=53, regularized=True)
    upper = hypgeom_wrappers.arb_hypgeom_gamma_upper_mode(s, z, impl="adaptive", prec_bits=53, regularized=True)
    total = di.fast_add(lower, upper)
    assert bool(total[0] <= 1.0 <= total[1])


def test_hypgeom_complex_direct_batch_wrappers_keep_float32_box_dtype():
    def _box_batch(lo_re, hi_re, n):
        lo = jnp.linspace(jnp.float32(lo_re), jnp.float32(hi_re), n)
        hi = lo + jnp.float32(0.05)
        return jnp.stack((lo, hi, jnp.zeros_like(lo), jnp.zeros_like(lo)), axis=-1)

    a = _box_batch(1.1, 1.3, 3)
    b = _box_batch(2.1, 2.3, 3)
    c = _box_batch(2.8, 3.0, 3)
    z = _box_batch(0.1, 0.3, 3)
    pfq_a = _pfq_box_params(3, 2, 0.6)
    pfq_b = _pfq_box_params(3, 1, 1.4)
    outs = [
        hypgeom.acb_hypgeom_0f1_batch_padded_prec(a, z, pad_to=8, prec_bits=53),
        hypgeom.acb_hypgeom_gamma_lower_batch_padded_prec(a, z, pad_to=8, prec_bits=53),
        hypgeom.acb_hypgeom_gamma_upper_batch_padded_prec(a, z, pad_to=8, prec_bits=53),
        hypgeom.acb_hypgeom_1f1_batch_padded_prec(a, b, z, pad_to=8, prec_bits=53),
        hypgeom.acb_hypgeom_2f1_batch_padded_prec(a, b, c, z, pad_to=8, prec_bits=53),
        hypgeom.acb_hypgeom_u_batch_padded_prec(a, b, _box_batch(0.6, 1.0, 3), pad_to=8, prec_bits=53),
        hypgeom.acb_hypgeom_chebyshev_t_batch_padded_prec(2, z, pad_to=8, prec_bits=53),
        hypgeom.acb_hypgeom_chebyshev_u_batch_padded_prec(2, z, pad_to=8, prec_bits=53),
        hypgeom.acb_hypgeom_laguerre_l_batch_padded_prec(2, a, z, pad_to=8, prec_bits=53),
        hypgeom.acb_hypgeom_hermite_h_batch_padded_prec(2, z, pad_to=8, prec_bits=53),
        hypgeom.acb_hypgeom_pfq_batch_padded_prec(pfq_a, pfq_b, z, pad_to=8, prec_bits=53),
    ]
    for out in outs:
        assert jnp.asarray(out).dtype == jnp.float32


def test_hypgeom_point_ad_smoke_across_regime_boundaries():
    si_grad = jax.grad(lambda x: hypgeom._real_si_ci_scalar(x)[0])
    ci_grad = jax.grad(lambda x: hypgeom._real_si_ci_scalar(x)[1])
    for x in (jnp.float64(3.9), jnp.float64(4.1)):
        assert jnp.isfinite(si_grad(x))
        assert jnp.isfinite(ci_grad(x))

    hyp1f1_grad = jax.grad(lambda z: hypgeom._real_hyp1f1_regime(jnp.float64(1.25), jnp.float64(2.5), z))
    hypu_grad = jax.grad(lambda z: hypgeom._real_hypu_regime(jnp.float64(1.25), jnp.float64(2.5), z))
    for z in (jnp.float64(0.5), jnp.float64(5.0)):
        assert jnp.isfinite(hyp1f1_grad(z))
        assert jnp.isfinite(hypu_grad(z))

    boost_grad = jax.grad(lambda z: boost_hypgeom.boost_hypergeometric_1f1(1.25, 2.5, z, mode="point"))
    cusf_grad = jax.grad(lambda z: cusf_compat.cusf_hyp1f1(jnp.float64(1.25), jnp.float64(2.5), z, mode="point"))
    for z in (jnp.float64(0.25), jnp.float64(0.75)):
        assert jnp.isfinite(boost_grad(z))
        assert jnp.isfinite(cusf_grad(z))


def test_hypgeom_orthogonal_point_ad_smoke():
    leg_grad = jax.grad(lambda x: jnp.squeeze(hypgeom._real_legendre_p_scalar(3, x)))
    jac_grad = jax.grad(lambda x: jnp.squeeze(hypgeom._real_jacobi_p_scalar(3, jnp.float64(0.2), jnp.float64(0.3), x)))
    geg_grad = jax.grad(lambda x: jnp.squeeze(hypgeom._real_gegenbauer_c_scalar(3, jnp.float64(0.7), x)))
    for x in (jnp.float64(-0.5), jnp.float64(0.0), jnp.float64(0.5)):
        assert jnp.isfinite(leg_grad(x))
        assert jnp.isfinite(jac_grad(x))
        assert jnp.isfinite(geg_grad(x))


def test_hypgeom_weaker_orthogonal_and_pfq_point_ad_smoke():
    cheb_t_grad = jax.grad(lambda x: jnp.squeeze(hypgeom._interval_midpoint(hypgeom.arb_hypgeom_chebyshev_t(3, di.interval(x, x)))))
    cheb_u_grad = jax.grad(lambda x: jnp.squeeze(hypgeom._interval_midpoint(hypgeom.arb_hypgeom_chebyshev_u(3, di.interval(x, x)))))
    lag_grad = jax.grad(lambda x: jnp.squeeze(hypgeom._interval_midpoint(hypgeom.arb_hypgeom_laguerre_l(3, di.interval(0.2, 0.2), di.interval(x, x)))))
    herm_grad = jax.grad(lambda x: jnp.squeeze(hypgeom._interval_midpoint(hypgeom.arb_hypgeom_hermite_h(3, di.interval(x, x)))))
    pfq_a = jnp.asarray([0.6, 0.9], dtype=jnp.float64)
    pfq_b = jnp.asarray([1.4], dtype=jnp.float64)
    pfq_grad = jax.grad(lambda x: jnp.squeeze(hypgeom._interval_midpoint(hypgeom.arb_hypgeom_pfq(pfq_a, pfq_b, di.interval(x, x)))))
    for x in (jnp.float64(-0.5), jnp.float64(0.1), jnp.float64(0.5)):
        assert jnp.isfinite(cheb_t_grad(x))
        assert jnp.isfinite(cheb_u_grad(x))
        assert jnp.isfinite(lag_grad(x))
        assert jnp.isfinite(herm_grad(x))
    for x in (jnp.float64(0.1), jnp.float64(0.3), jnp.float64(0.5)):
        assert jnp.isfinite(pfq_grad(x))


def test_hypgeom_pfq_and_u_boundary_sweeps():
    pfq_a = jnp.asarray([0.6, 0.9], dtype=jnp.float64)
    pfq_b = jnp.asarray([1.4], dtype=jnp.float64)
    pfq_grad = jax.grad(lambda x: jnp.squeeze(hypgeom._interval_midpoint(hypgeom.arb_hypgeom_pfq(pfq_a, pfq_b, di.interval(x, x)))))
    for x in (jnp.float64(0.85), jnp.float64(0.95), jnp.float64(1.05)):
        assert jnp.isfinite(pfq_grad(x))

    hypu_grad = jax.grad(lambda z: hypgeom._real_hypu_regime(jnp.float64(1.25), jnp.float64(2.5), z))
    for z in (jnp.float64(7.9), jnp.float64(8.0), jnp.float64(8.1)):
        assert jnp.isfinite(hypu_grad(z))


def test_hypgeom_1f1_2f1_and_incomplete_gamma_boundary_sweeps():
    hyp1f1_grad = jax.grad(lambda z: hypgeom._real_hyp1f1_regime(jnp.float64(1.25), jnp.float64(2.5), z))
    for z in (jnp.float64(5.9), jnp.float64(6.0), jnp.float64(6.1)):
        assert jnp.isfinite(hyp1f1_grad(z))

    hyp2f1_grad = jax.grad(lambda z: hypgeom._real_hyp2f1_regime(jnp.float64(0.5), jnp.float64(0.75), jnp.float64(1.5), z))
    for z in (jnp.float64(0.74), jnp.float64(0.75), jnp.float64(0.76)):
        assert jnp.isfinite(hyp2f1_grad(z))

    lower_grad = jax.grad(
        lambda x: jnp.squeeze(hypgeom._interval_midpoint(hypgeom.arb_hypgeom_gamma_lower(di.interval(1.25, 1.25), di.interval(x, x))))
    )
    upper_grad = jax.grad(
        lambda x: jnp.squeeze(hypgeom._interval_midpoint(hypgeom.arb_hypgeom_gamma_upper(di.interval(1.25, 1.25), di.interval(x, x))))
    )
    for x in (jnp.float64(0.1), jnp.float64(0.5), jnp.float64(1.0)):
        assert jnp.isfinite(lower_grad(x))
        assert jnp.isfinite(upper_grad(x))


def test_hypgeom_complex_1f1_2f1_u_and_incomplete_gamma_cut_corner_ad():
    def _box(x, y):
        xr = jnp.asarray(x, dtype=jnp.float64)
        yr = jnp.asarray(y, dtype=jnp.float64)
        return jnp.stack((xr, xr, yr, yr))

    onef1_rx = jax.grad(
        lambda x: hypgeom.acb_midpoint(
            hypgeom.acb_hypgeom_1f1(_box(0.5, 0.0), _box(1.5, 0.0), _box(x, 1e-3))
        ).real
    )
    onef1_ry = jax.grad(
        lambda y: hypgeom.acb_midpoint(
            hypgeom.acb_hypgeom_1f1(_box(0.5, 0.0), _box(1.5, 0.0), _box(6.0, y))
        ).real
    )
    twof1_rx = jax.grad(
        lambda x: hypgeom.acb_midpoint(
            hypgeom.acb_hypgeom_2f1(_box(0.5, 0.0), _box(0.75, 0.0), _box(1.5, 0.0), _box(x, 1e-3))
        ).real
    )
    twof1_ry = jax.grad(
        lambda y: hypgeom.acb_midpoint(
            hypgeom.acb_hypgeom_2f1(_box(0.5, 0.0), _box(0.75, 0.0), _box(1.5, 0.0), _box(1.0, y))
        ).real
    )
    u_rx = jax.grad(
        lambda x: hypgeom.acb_midpoint(hypgeom.acb_hypgeom_u(_box(1.25, 0.0), _box(2.5, 0.0), _box(x, 1e-3))).real
    )
    u_ry = jax.grad(
        lambda y: hypgeom.acb_midpoint(hypgeom.acb_hypgeom_u(_box(1.25, 0.0), _box(2.5, 0.0), _box(-1.0, y))).real
    )
    gl_rx = jax.grad(
        lambda x: hypgeom.acb_midpoint(hypgeom.acb_hypgeom_gamma_lower(_box(1.25, 0.0), _box(x, 1e-3))).real
    )
    gu_rx = jax.grad(
        lambda x: hypgeom.acb_midpoint(hypgeom.acb_hypgeom_gamma_upper(_box(1.25, 0.0), _box(x, 1e-3))).real
    )

    for x in (jnp.float64(5.9), jnp.float64(6.0), jnp.float64(6.1)):
        assert jnp.isfinite(onef1_rx(x))
    for y in (jnp.float64(-0.25), jnp.float64(-1e-3), jnp.float64(1e-3), jnp.float64(0.25)):
        assert jnp.isfinite(onef1_ry(y))
    for x in (jnp.float64(-1.05), jnp.float64(-1.0), jnp.float64(-0.95), jnp.float64(0.95), jnp.float64(1.0), jnp.float64(1.05)):
        assert jnp.isfinite(twof1_rx(x))
    for y in (jnp.float64(-0.25), jnp.float64(-1e-3), jnp.float64(1e-3), jnp.float64(0.25)):
        assert jnp.isfinite(twof1_ry(y))
        assert jnp.isfinite(u_ry(y))
    for x in (jnp.float64(7.5), jnp.float64(7.9), jnp.float64(8.0), jnp.float64(8.1), jnp.float64(8.5)):
        assert jnp.isfinite(u_rx(x))
    for x in (jnp.float64(0.05), jnp.float64(0.1), jnp.float64(1.0), jnp.float64(2.0)):
        assert jnp.isfinite(gl_rx(x))
        assert jnp.isfinite(gu_rx(x))


def test_hypgeom_point_batch_uses_direct_point_kernels():
    a = jnp.linspace(jnp.float64(1.1), jnp.float64(1.3), 4)
    b = jnp.linspace(jnp.float64(2.1), jnp.float64(2.3), 4)
    c = jnp.linspace(jnp.float64(2.8), jnp.float64(3.0), 4)
    z = jnp.linspace(jnp.float64(0.1), jnp.float64(0.3), 4)
    lam = jnp.linspace(jnp.float64(0.6), jnp.float64(0.8), 4)
    zeros = jnp.zeros_like(z)
    cases = [
        ("hypgeom.arb_hypgeom_0f1", (a, z), point_wrappers.arb_hypgeom_0f1_point(a, z)),
        ("hypgeom.arb_hypgeom_1f1", (a, b, z), point_wrappers.arb_hypgeom_1f1_point(a, b, z)),
        ("hypgeom.arb_hypgeom_2f1", (a, b, c, z), point_wrappers.arb_hypgeom_2f1_point(a, b, c, z)),
        ("hypgeom.arb_hypgeom_u", (a, b, z), point_wrappers.arb_hypgeom_u_point(a, b, z)),
        ("hypgeom.arb_hypgeom_gamma_lower", (a, z), point_wrappers.arb_hypgeom_gamma_lower_point(a, z)),
        ("hypgeom.arb_hypgeom_gamma_upper", (a, z), point_wrappers.arb_hypgeom_gamma_upper_point(a, z)),
        ("hypgeom.arb_hypgeom_gamma", (a,), point_wrappers.arb_hypgeom_gamma_point(a)),
        ("hypgeom.arb_hypgeom_erf", (z,), point_wrappers.arb_hypgeom_erf_point(z)),
        ("hypgeom.arb_hypgeom_erfc", (z,), point_wrappers.arb_hypgeom_erfc_point(z)),
        ("hypgeom.arb_hypgeom_erfi", (z,), point_wrappers.arb_hypgeom_erfi_point(z)),
        ("hypgeom.arb_hypgeom_erfinv", (jnp.linspace(jnp.float64(-0.5), jnp.float64(0.5), 4),), point_wrappers.arb_hypgeom_erfinv_point(jnp.linspace(jnp.float64(-0.5), jnp.float64(0.5), 4))),
        ("hypgeom.arb_hypgeom_erfcinv", (jnp.linspace(jnp.float64(0.5), jnp.float64(1.5), 4),), point_wrappers.arb_hypgeom_erfcinv_point(jnp.linspace(jnp.float64(0.5), jnp.float64(1.5), 4))),
        ("hypgeom.arb_hypgeom_ei", (z,), point_wrappers.arb_hypgeom_ei_point(z)),
        ("hypgeom.arb_hypgeom_si", (z,), point_wrappers.arb_hypgeom_si_point(z)),
        ("hypgeom.arb_hypgeom_ci", (z,), point_wrappers.arb_hypgeom_ci_point(z)),
        ("hypgeom.arb_hypgeom_shi", (z,), point_wrappers.arb_hypgeom_shi_point(z)),
        ("hypgeom.arb_hypgeom_chi", (z,), point_wrappers.arb_hypgeom_chi_point(z)),
        ("hypgeom.arb_hypgeom_dilog", (z,), point_wrappers.arb_hypgeom_dilog_point(z)),
        ("hypgeom.arb_hypgeom_legendre_p", (2, zeros, z), point_wrappers.arb_hypgeom_legendre_p_point(2, zeros, z)),
        ("hypgeom.arb_hypgeom_gegenbauer_c", (2, lam, z), point_wrappers.arb_hypgeom_gegenbauer_c_point(2, lam, z)),
        ("hypgeom.arb_hypgeom_fresnel", (z,), point_wrappers.arb_hypgeom_fresnel_point(z)),
    ]
    for name, args, direct in cases:
        via_api = api.eval_point_batch(name, *args)
        assert _allclose_or_tuple(via_api, direct)

    li_direct = point_wrappers.arb_hypgeom_li_point(z, offset=1)
    li_api = api.eval_point_batch("hypgeom.arb_hypgeom_li", z, offset=1)
    assert _allclose_or_tuple(li_api, li_direct)


def test_acb_hypgeom_point_batch_uses_direct_point_kernels():
    z = jnp.linspace(jnp.complex64(0.1 + 0.05j), jnp.complex64(0.4 + 0.2j), 4)
    a = jnp.linspace(jnp.complex64(1.1 + 0.0j), jnp.complex64(1.3 + 0.1j), 4)

    cases = [
        ("hypgeom.acb_hypgeom_gamma", (a,), point_wrappers.acb_hypgeom_gamma_point(a)),
        ("hypgeom.acb_hypgeom_erf", (z,), point_wrappers.acb_hypgeom_erf_point(z)),
        ("hypgeom.acb_hypgeom_erfc", (z,), point_wrappers.acb_hypgeom_erfc_point(z)),
        ("hypgeom.acb_hypgeom_erfi", (z,), point_wrappers.acb_hypgeom_erfi_point(z)),
        ("hypgeom.acb_hypgeom_ei", (z,), point_wrappers.acb_hypgeom_ei_point(z)),
        ("hypgeom.acb_hypgeom_si", (z,), point_wrappers.acb_hypgeom_si_point(z)),
        ("hypgeom.acb_hypgeom_ci", (z,), point_wrappers.acb_hypgeom_ci_point(z)),
        ("hypgeom.acb_hypgeom_shi", (z,), point_wrappers.acb_hypgeom_shi_point(z)),
        ("hypgeom.acb_hypgeom_chi", (z,), point_wrappers.acb_hypgeom_chi_point(z)),
        ("hypgeom.acb_hypgeom_li", (z,), point_wrappers.acb_hypgeom_li_point(z)),
        ("hypgeom.acb_hypgeom_dilog", (z,), point_wrappers.acb_hypgeom_dilog_point(z)),
        ("hypgeom.acb_hypgeom_fresnel", (z,), point_wrappers.acb_hypgeom_fresnel_point(z)),
    ]
    for name, args, direct in cases:
        via_api = api.eval_point_batch(name, *args, dtype="float32", pad_to=8)
        assert _allclose_or_tuple(via_api, direct, rtol=1e-5, atol=1e-5)
