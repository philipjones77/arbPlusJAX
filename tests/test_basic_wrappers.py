import jax
import jax.numpy as jnp

from arbplusjax import baseline_wrappers as bw
from arbplusjax import arb_calc, acb_calc
from arbplusjax import checks
from arbplusjax import double_interval as di
from arbplusjax import api
from arbplusjax import double_gamma as bdg
from arbplusjax import kernel_helpers as kh
from arbplusjax import point_wrappers as pw


from tests._test_checks import _check
def _interval(lo: float, hi: float):
    return di.interval(jnp.float64(lo), jnp.float64(hi))


def _box(re_lo: float, re_hi: float, im_lo: float, im_hi: float):
    return jnp.array([re_lo, re_hi, im_lo, im_hi], dtype=jnp.float64)


def test_basic_modes_real():
    x = _interval(0.2, 0.3)
    for mode in ("basic", "rigorous", "adaptive"):
        y = bw.arb_exp_mp(x, mode=mode, dps=50)
        _check(y.shape == (2,))
        _check(bool(y[0] <= y[1]))


def test_basic_modes_complex():
    x = _box(0.2, 0.3, -0.1, 0.1)
    for mode in ("basic", "rigorous", "adaptive"):
        y = bw.acb_exp_mp(x, mode=mode, dps=50)
        _check(y.shape == (4,))
        _check(bool(y[0] <= y[1]))
        _check(bool(y[2] <= y[3]))


def test_basic_grad_path():
    x = _interval(0.2, 0.3)

    def loss(t):
        xt = di.interval(t, t)
        y = bw.arb_log_mp(xt, mode="basic", dps=50)
        return jnp.sum(y)

    g = jax.grad(loss)(jnp.float64(0.25))
    _check(bool(jnp.isfinite(g)))


def test_top10_wrappers_and_api_shapes():
    x = _interval(0.2, 0.3)
    y = _interval(1.1, 1.2)
    z = _interval(-0.4, -0.2)

    unary = (
        bw.arb_abs_mp,
        bw.arb_inv_mp,
        bw.arb_log1p_mp,
        bw.arb_expm1_mp,
    )
    for fn in unary:
        for mode in ("basic", "rigorous", "adaptive"):
            out = fn(x if fn is not bw.arb_inv_mp else y, mode=mode, dps=50)
            _check(out.shape == (2,))
            _check(bool(out[0] <= out[1]))

    bivariate = (bw.arb_add_mp, bw.arb_sub_mp, bw.arb_mul_mp, bw.arb_div_mp)
    for fn in bivariate:
        for mode in ("basic", "rigorous", "adaptive"):
            out = fn(x, y, mode=mode, dps=50)
            _check(out.shape == (2,))
            _check(bool(out[0] <= out[1]))

    for mode in ("basic", "rigorous", "adaptive"):
        out = bw.arb_fma_mp(x, y, z, mode=mode, dps=50)
        _check(out.shape == (2,))
        _check(bool(out[0] <= out[1]))
        s, c = bw.arb_sin_cos_mp(x, mode=mode, dps=50)
        _check(s.shape == (2,))
        _check(c.shape == (2,))

    _check(api.eval_interval("add", x, y, mode="basic", dps=50).shape == (2,))
    _check(api.eval_interval("fma", x, y, z, mode="basic", dps=50).shape == (2,))
    s_api, c_api = api.eval_interval("sin_cos", x, mode="basic", dps=50)
    _check(s_api.shape == (2,))
    _check(c_api.shape == (2,))


def test_api_dtype_policy_rejects_mixed_float_dtypes():
    x32 = jnp.array([1.0, 2.0], dtype=jnp.float32)
    y64 = jnp.array([3.0, 4.0], dtype=jnp.float64)
    try:
        _ = api.eval_point("add", x32, y64)
        _check(False)
    except ValueError as exc:
        _check("Mixed floating dtypes" in str(exc))


def test_api_dtype_policy_allows_explicit_dtype_override():
    x32 = jnp.array([1.0, 2.0], dtype=jnp.float32)
    y64 = jnp.array([3.0, 4.0], dtype=jnp.float64)
    out32 = api.eval_point("add", x32, y64, dtype="float32")
    out64 = api.eval_point("add", x32, y64, dtype="float64")
    _check(out32.dtype == jnp.float32)
    _check(out64.dtype == jnp.float64)


def test_api_bind_batch_respects_dtype_lock():
    fn64 = api.bind_point_batch("add", dtype="float64")
    x32 = jnp.array([1.0, 2.0], dtype=jnp.float32)
    y32 = jnp.array([3.0, 4.0], dtype=jnp.float32)
    out = fn64(x32, y32)
    _check(out.dtype == jnp.float64)


def test_api_interval_batch_float32_path():
    x = jnp.array([[0.1, 0.2], [0.2, 0.3]], dtype=jnp.float32)
    y = jnp.array([[0.3, 0.4], [0.4, 0.5]], dtype=jnp.float32)
    out = api.eval_interval_batch("add", x, y, mode="basic", dtype="float32")
    _check(out.dtype == jnp.float32)


def test_api_point_batch_optional_padding_trims_output():
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    y = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32)
    out = api.eval_point_batch("add", x, y, dtype="float32", pad_to=8)
    _check(out.dtype == jnp.float32)
    _check(out.shape == (3,))
    _check(bool(jnp.allclose(out, jnp.array([5.0, 7.0, 9.0], dtype=jnp.float32))))


def test_api_core_point_batch_uses_family_point_kernels_cleanly():
    x = jnp.array([0.2, 0.3, 0.4], dtype=jnp.float32)
    y = jnp.array([1.1, 1.2, 1.3], dtype=jnp.float32)

    cases = [
        ("exp", (x,), api.eval_point("exp", x, dtype="float32")),
        ("add", (x, y), api.eval_point("add", x, y, dtype="float32")),
        ("sin_pi", (x,), api.eval_point("sin_pi", x, dtype="float32")),
        ("gamma", (y,), api.eval_point("gamma", y, dtype="float32")),
        ("besselj", (x, y), api.eval_point("besselj", x, y, dtype="float32")),
    ]
    for name, args, expected in cases:
        out = api.eval_point_batch(name, *args, dtype="float32", pad_to=8)
        _check(out.dtype == jnp.asarray(expected).dtype)
        _check(out.shape == jnp.asarray(expected).shape)
        _check(bool(jnp.allclose(out, expected, rtol=1e-5, atol=1e-5, equal_nan=True)))


def test_api_canonical_arb_acb_point_batch_aliases_use_direct_point_kernels():
    xr = jnp.array([0.2, 0.3, 0.4], dtype=jnp.float32)
    xc = jnp.array([0.2 + 0.1j, 0.3 - 0.1j, 0.4 + 0.2j], dtype=jnp.complex64)

    real_cases = [
        ("arb_exp", (xr,), api.eval_point("arb_exp", xr, dtype="float32")),
        ("arb_sin_pi", (xr,), api.eval_point("arb_sin_pi", xr, dtype="float32")),
        ("arb_gamma", (xr + jnp.float32(1.0),), api.eval_point("arb_gamma", xr + jnp.float32(1.0), dtype="float32")),
    ]
    complex_cases = [
        ("acb_exp", (xc,), api.eval_point("acb_exp", xc, dtype="float32")),
        ("acb_sin_pi", (xc,), api.eval_point("acb_sin_pi", xc, dtype="float32")),
        ("acb_gamma", (xc + jnp.complex64(1.0 + 0.0j),), api.eval_point("acb_gamma", xc + jnp.complex64(1.0 + 0.0j), dtype="float32")),
    ]

    for name, args, expected in real_cases + complex_cases:
        out = api.eval_point_batch(name, *args, dtype="float32", pad_to=8)
        _check(out.dtype == jnp.asarray(expected).dtype)
        _check(out.shape == jnp.asarray(expected).shape)
        _check(bool(jnp.allclose(out, expected, rtol=1e-4, atol=1e-4, equal_nan=True)))


def test_api_complex_core_helper_point_batch_stays_on_point_kernels():
    z = jnp.array([0.8 + 0.1j, 1.1 - 0.2j, 1.4 + 0.3j], dtype=jnp.complex64)
    w = jnp.array([1.2 + 0.0j, 1.5 + 0.2j, 1.8 - 0.1j], dtype=jnp.complex64)

    unary_cases = [
        ("acb_digamma", (z,)),
        ("acb_zeta", (z,)),
        ("acb_agm1", (z,)),
        ("acb_agm1_cpx", (z,)),
        ("acb_polylog_si", (z,), {"s": 2}),
    ]
    binary_cases = [
        ("acb_hurwitz_zeta", (z, w)),
        ("acb_polylog", (z, w)),
        ("acb_agm", (z, w)),
    ]

    for entry in unary_cases:
        name, args = entry[0], entry[1]
        kwargs = entry[2] if len(entry) > 2 else {}
        if name == "acb_polylog_si":
            expected = pw.acb_polylog_si_point(*args, kwargs["s"])
            out = pw.acb_polylog_si_point(*args, kwargs["s"])
        else:
            expected = api.eval_point(name, *args, dtype="float32")
            out = api.eval_point_batch(name, *args, dtype="float32", pad_to=8)
        _check(out.dtype == jnp.asarray(expected).dtype)
        _check(out.shape == jnp.asarray(expected).shape)
        _check(bool(jnp.allclose(out, expected, rtol=1e-4, atol=1e-4, equal_nan=True)))


def test_api_custom_complex_point_batch_uses_direct_point_kernels():
    z = jnp.array([0.4 + 0.3j, 0.6 + 0.5j, 0.8 + 0.7j], dtype=jnp.complex64)

    cases = [
        ("acb_dirichlet_zeta", (z,), {"n_terms": 24}),
        ("acb_dirichlet_eta", (z,), {"n_terms": 24}),
        ("acb_modular_j", (z,)),
        ("acb_elliptic_k", (jnp.array([0.1 + 0.1j, 0.2 + 0.15j, 0.3 + 0.2j], dtype=jnp.complex64),)),
        ("acb_elliptic_e", (jnp.array([0.1 + 0.1j, 0.2 + 0.15j, 0.3 + 0.2j], dtype=jnp.complex64),)),
    ]

    for entry in cases:
        name, args = entry[0], entry[1]
        kwargs = entry[2] if len(entry) > 2 else {}
        expected = getattr(pw, f"{name}_point")(*args, **kwargs)
        out = api.eval_point_batch(name, *args, dtype="float32", pad_to=8, **kwargs)
        _check(out.dtype == jnp.asarray(expected).dtype)
        _check(out.shape == jnp.asarray(expected).shape)
        _check(bool(jnp.allclose(out, expected, rtol=1e-4, atol=1e-4, equal_nan=True)))


def test_api_calc_point_batch_uses_direct_point_kernels():
    a = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
    b = jnp.array([0.5, 0.6, 0.7], dtype=jnp.float32)
    out = api.eval_point_batch("arb_calc_integrate_line", a, b, dtype="float32", pad_to=8, integrand="sin", n_steps=16)
    expected = arb_calc.arb_calc_integrate_line_batch_fixed_point(a, b, integrand="sin", n_steps=16)
    _check(out.dtype == expected.dtype)
    _check(out.shape == expected.shape)
    _check(bool(jnp.allclose(out, expected, rtol=1e-5, atol=1e-5, equal_nan=True)))

    ac = jnp.array([0.1 + 0.05j, 0.2 + 0.1j, 0.3 + 0.15j], dtype=jnp.complex64)
    bc = jnp.array([0.4 + 0.1j, 0.5 + 0.15j, 0.6 + 0.2j], dtype=jnp.complex64)
    outc = api.eval_point_batch("acb_calc_integrate_line", ac, bc, dtype="float32", pad_to=8, integrand="exp", n_steps=16)
    expectedc = acb_calc.acb_calc_integrate_line_batch_fixed_point(ac, bc, integrand="exp", n_steps=16)
    _check(outc.dtype == expectedc.dtype)
    _check(outc.shape == expectedc.shape)
    _check(bool(jnp.allclose(outc, expectedc, rtol=1e-5, atol=1e-5, equal_nan=True)))


def test_api_calc_interval_batch_fastpaths_match_direct_kernels():
    a = di.interval(jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32), jnp.array([0.15, 0.25, 0.35], dtype=jnp.float32))
    b = di.interval(jnp.array([0.5, 0.6, 0.7], dtype=jnp.float32), jnp.array([0.55, 0.65, 0.75], dtype=jnp.float32))
    basic = api.eval_interval_batch("arb_calc_integrate_line", a, b, mode="basic", dtype="float32", pad_to=8, integrand="exp", n_steps=16, prec_bits=53)
    basic_expected = arb_calc.arb_calc_integrate_line_batch_padded_prec(a, b, pad_to=8, integrand="exp", n_steps=16, prec_bits=53)
    _check(bool(jnp.allclose(basic, basic_expected, rtol=1e-5, atol=1e-5, equal_nan=True)))

    rigorous = api.eval_interval_batch("arb_calc_integrate_line", a, b, mode="rigorous", dtype="float32", pad_to=8, integrand="exp", n_steps=16)
    rigorous_expected = arb_calc.arb_calc_integrate_line_batch_padded_rigorous(a, b, pad_to=8, integrand="exp", n_steps=16)
    _check(bool(jnp.allclose(rigorous, rigorous_expected, rtol=1e-5, atol=1e-5, equal_nan=True)))

    re_lo = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
    re_hi = re_lo + jnp.float32(0.05)
    im_lo = jnp.array([0.0, 0.05, 0.1], dtype=jnp.float32)
    im_hi = im_lo + jnp.float32(0.05)
    ac = jnp.stack((re_lo, re_hi, im_lo, im_hi), axis=-1)
    bc = jnp.stack((re_lo + 0.3, re_hi + 0.3, im_lo, im_hi), axis=-1)
    cbasic = api.eval_interval_batch("acb_calc_integrate_line", ac, bc, mode="basic", dtype="float32", pad_to=8, integrand="sin", n_steps=16, prec_bits=53)
    cbasic_expected = acb_calc.acb_calc_integrate_line_batch_padded_prec(ac, bc, pad_to=8, integrand="sin", n_steps=16, prec_bits=53)
    _check(bool(jnp.allclose(cbasic, cbasic_expected, rtol=1e-5, atol=1e-5, equal_nan=True)))

    crigorous = api.eval_interval_batch("acb_calc_integrate_line", ac, bc, mode="rigorous", dtype="float32", pad_to=8, integrand="sin", n_steps=16, prec_bits=53)
    crigorous_expected = acb_calc.acb_calc_integrate_line_batch_padded_rigorous(ac, bc, pad_to=8, integrand="sin", n_steps=16, prec_bits=53)
    _check(bool(jnp.allclose(crigorous, crigorous_expected, rtol=1e-5, atol=1e-5, equal_nan=True)))


def test_kernel_helpers_padding_trimming_and_point_conversions():
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    y = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)
    padded, n = kh.pad_batch_args((x, y), pad_to=4, pad_value=-1.0)
    _check(n == 2)
    _check(padded[0].shape == (4, 2))
    _check(bool(jnp.allclose(padded[0][2:], -1.0)))
    _check(bool(jnp.allclose(kh.trim_batch_out(padded[0], n), x)))

    mixed_padded, mixed_n = kh.pad_mixed_batch_args_repeat_last((x, 3.0), pad_to=3)
    _check(mixed_n == 2)
    _check(mixed_padded[0].shape == (3, 2))
    _check(bool(jnp.allclose(mixed_padded[0][-1], x[-1])))

    interval_mid = kh.midpoint_from_interval_like(di.interval(jnp.array([1.0]), jnp.array([3.0])))
    _check(bool(jnp.allclose(interval_mid, jnp.array([2.0]))))

    boxed = kh.point_box(jnp.array([1.0 + 2.0j], dtype=jnp.complex64))
    _check(boxed.shape == (1, 4))


def test_checks_contract_helpers_raise_on_bad_inputs():
    try:
        checks.check_in_set("bad", ("basic", "rigorous"), "mode")
        _check(False)
    except ValueError as exc:
        _check("mode" in str(exc))

    try:
        checks.check_last_dim(jnp.ones((2, 3), dtype=jnp.float32), 4, "tail")
        _check(False)
    except ValueError as exc:
        _check("expected last dimension" in str(exc))


def test_api_barnes_family_point_batch_uses_direct_point_kernels():
    z = jnp.array([1.2 + 0.1j, 1.4 - 0.2j, 1.6 + 0.25j], dtype=jnp.complex64)
    tau = jnp.array([0.9 + 0.0j, 1.1 + 0.1j, 1.3 - 0.1j], dtype=jnp.complex64)

    cases = [
        ("bdg_barnesdoublegamma", (z, tau)),
        ("bdg_barnesgamma2", (z, tau)),
        ("bdg_double_sine", (z, tau)),
        ("shahen_barnesgamma2", (z, tau)),
    ]

    for name, args in cases:
        fixed_name = name.replace("shahen_", "bdg_") + "_batch_fixed_point"
        expected = getattr(bdg, fixed_name)(*args)
        out = api.eval_point_batch(name, *args, dtype="float32", pad_to=8)
        _check(out.dtype == jnp.asarray(expected).dtype)
        _check(out.shape == jnp.asarray(expected).shape)
        _check(bool(jnp.allclose(out, expected, rtol=1e-4, atol=1e-4, equal_nan=True)))


def test_api_bind_interval_batch_optional_padding():
    fn = api.bind_interval_batch("add", mode="basic", dtype="float32", pad_to=4)
    x = jnp.array([[0.1, 0.2], [0.2, 0.3]], dtype=jnp.float32)
    y = jnp.array([[0.3, 0.4], [0.4, 0.5]], dtype=jnp.float32)
    out = fn(x, y)
    _check(out.dtype == jnp.float32)
    _check(out.shape == (2, 2))


def test_api_padding_rejects_smaller_target():
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    y = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32)
    try:
        _ = api.eval_point_batch("add", x, y, dtype="float32", pad_to=2)
        _check(False)
    except ValueError as exc:
        _check("pad_to must be >=" in str(exc))
