from __future__ import annotations

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy.special as jsp

from . import double_interval as di

jax.config.update("jax_enable_x64", True)

_HAS_HYP1F1 = hasattr(jsp, "hyp1f1")
_HAS_HYP2F1 = hasattr(jsp, "hyp2f1")
_HAS_HYPERU = hasattr(jsp, "hyperu")
_HAS_JV = hasattr(jsp, "jv")
_HAS_YV = hasattr(jsp, "yv")
_HAS_IV = hasattr(jsp, "iv")
_HAS_KV = hasattr(jsp, "kv")

_DIGAMMA_ZERO = jnp.float64(1.4616321449683623413)
_LOG_SQRT_2PI = jnp.float64(0.91893853320467274178)
_TWO_OVER_SQRT_PI = jnp.float64(1.12837916709551257390)
_ERF_TERMS = 48
_HYP_TERMS = 80
_BESSEL_TERMS = 60
_LANCZOS = jnp.asarray(
    [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ],
    dtype=jnp.float64,
)

_BESSEL_REAL_MODES = ("sample", "midpoint")
_ERFINV_A = jnp.float64(0.147)
_ERFINV_ITERS = 3
_SQRT_2_OVER_PI = jnp.float64(0.79788456080286535588)
_SQRT_PI_OVER_2 = jnp.float64(1.2533141373155002512)
_SI_CI_TERMS = 60
_AIRY_TERMS = 40


def _validate_bessel_real_mode(mode: str) -> str:
    if mode not in _BESSEL_REAL_MODES:
        raise ValueError(f"Unsupported bessel real mode '{mode}'. Expected one of {_BESSEL_REAL_MODES}.")
    return mode


def _interval_from_midpoint(val: jax.Array) -> jax.Array:
    return di.interval(di._below(val), di._above(val))


def as_acb_box(x: jax.Array) -> jax.Array:
    arr = jnp.asarray(x, dtype=jnp.float64)
    if arr.shape[-1] != 4:
        raise ValueError(f"Expected last dimension to be 4, got shape {arr.shape}")
    return arr


def acb_box(real_interval: jax.Array, imag_interval: jax.Array) -> jax.Array:
    r = di.as_interval(real_interval)
    i = di.as_interval(imag_interval)
    return jnp.concatenate([r, i], axis=-1)


def acb_real(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    return box[..., 0:2]


def acb_imag(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    return box[..., 2:4]


def acb_box_add_ui(x: jax.Array, k: int) -> jax.Array:
    box = as_acb_box(x)
    k_interval = di.interval(jnp.float64(k), jnp.float64(k))
    return acb_box(di.fast_add(acb_real(box), k_interval), acb_imag(box))


def acb_box_mul(x: jax.Array, y: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    yb = as_acb_box(y)
    xr, xi = acb_real(xb), acb_imag(xb)
    yr, yi = acb_real(yb), acb_imag(yb)

    ac = di.fast_mul(xr, yr)
    bd = di.fast_mul(xi, yi)
    ad = di.fast_mul(xr, yi)
    bc = di.fast_mul(xi, yr)

    return acb_box(di.fast_sub(ac, bd), di.fast_add(ad, bc))


def acb_box_add(x: jax.Array, y: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    yb = as_acb_box(y)
    return acb_box(di.fast_add(acb_real(xb), acb_real(yb)), di.fast_add(acb_imag(xb), acb_imag(yb)))


def acb_box_scale_real(x: jax.Array, r: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    rr = di.as_interval(r)
    return acb_box(di.fast_mul(acb_real(xb), rr), di.fast_mul(acb_imag(xb), rr))


def acb_box_inv(x: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    xr, xi = acb_real(xb), acb_imag(xb)
    den = di.fast_add(di.fast_mul(xr, xr), di.fast_mul(xi, xi))
    inv_re = di.fast_div(xr, den)
    inv_im = di.fast_div(di.neg(xi), den)
    return acb_box(inv_re, inv_im)


def acb_box_div(x: jax.Array, y: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    yb = as_acb_box(y)
    yr, yi = acb_real(yb), acb_imag(yb)
    den = di.fast_add(di.fast_mul(yr, yr), di.fast_mul(yi, yi))
    num = acb_box(yr, di.neg(yi))
    out = acb_box_mul(xb, num)
    return acb_box(di.fast_div(acb_real(out), den), di.fast_div(acb_imag(out), den))


def acb_box_round_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    xb = as_acb_box(x)
    return acb_box(
        di.round_interval_outward(acb_real(xb), prec_bits),
        di.round_interval_outward(acb_imag(xb), prec_bits),
    )


def _acb_width(x: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    return (xb[..., 1] - xb[..., 0]) + (xb[..., 3] - xb[..., 2])


def _select_tighter_acb(a: jax.Array, b: jax.Array) -> jax.Array:
    wa = _acb_width(a)
    wb = _acb_width(b)
    a_ok = jnp.isfinite(wa)
    b_ok = jnp.isfinite(wb)
    pick_a = (a_ok & b_ok & (wa <= wb)) | (a_ok & ~b_ok)
    pick_b = b_ok & ~a_ok
    full = acb_box(_full_interval(), _full_interval())
    return jnp.where(pick_a, a, jnp.where(pick_b, b, full))


def _ones_interval_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.float64)
    return di.interval(t, t)


def _zeros_interval_like(x: jax.Array) -> jax.Array:
    t = jnp.zeros_like(x[..., 0], dtype=jnp.float64)
    return di.interval(t, t)


def acb_box_add_ui_prec(x: jax.Array, k: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    xb = as_acb_box(x)
    k_interval = di.interval(jnp.float64(k), jnp.float64(k))
    return acb_box(di.fast_add_prec(acb_real(xb), k_interval, prec_bits), di.round_interval_outward(acb_imag(xb), prec_bits))


def acb_box_mul_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    xb = as_acb_box(x)
    yb = as_acb_box(y)
    xr, xi = acb_real(xb), acb_imag(xb)
    yr, yi = acb_real(yb), acb_imag(yb)

    ac = di.fast_mul_prec(xr, yr, prec_bits)
    bd = di.fast_mul_prec(xi, yi, prec_bits)
    ad = di.fast_mul_prec(xr, yi, prec_bits)
    bc = di.fast_mul_prec(xi, yr, prec_bits)

    return acb_box(
        di.fast_sub_prec(ac, bd, prec_bits),
        di.fast_add_prec(ad, bc, prec_bits),
    )


def _contains_nonpositive_integer(lo: jax.Array, hi: jax.Array) -> jax.Array:
    kmin = jnp.ceil(-hi)
    kmax = jnp.floor(-lo)
    return (kmin <= kmax) & (kmax >= 0.0)


def _complex_loggamma_lanczos(z: jax.Array) -> jax.Array:
    z = jnp.asarray(z, dtype=jnp.complex128)
    z1 = z - jnp.complex128(1.0 + 0.0j)
    x = jnp.complex128(_LANCZOS[0] + 0.0j)

    def body(i, acc):
        return acc + _LANCZOS[i] / (z1 + jnp.float64(i))

    x = lax.fori_loop(1, 9, body, x)
    t = z1 + jnp.float64(7.5)
    return _LOG_SQRT_2PI + (z1 + 0.5) * jnp.log(t) - t + jnp.log(x)


def _complex_loggamma(z: jax.Array) -> jax.Array:
    z = jnp.asarray(z, dtype=jnp.complex128)

    def reflection(w):
        return jnp.log(jnp.pi) - jnp.log(jnp.sin(jnp.pi * w)) - _complex_loggamma_lanczos(1.0 - w)

    return lax.cond(jnp.real(z) < 0.5, reflection, _complex_loggamma_lanczos, z)


def _complex_erf_series(z: jax.Array) -> jax.Array:
    z = jnp.asarray(z, dtype=jnp.complex128)
    z2 = z * z
    term0 = z
    sum0 = z

    def body(k, state):
        term, s = state
        den = jnp.float64(k + 1) * jnp.float64(2 * k + 3)
        term = term * (-z2) / den
        return term, s + term

    _, s = lax.fori_loop(0, _ERF_TERMS - 1, body, (term0, sum0))
    return _TWO_OVER_SQRT_PI * s


def _complex_erfc_series(z: jax.Array) -> jax.Array:
    return 1.0 - _complex_erf_series(z)


def _complex_erfi_series(z: jax.Array) -> jax.Array:
    return -1j * _complex_erf_series(1j * z)


def _real_erf_series(x: jax.Array) -> jax.Array:
    x = jnp.asarray(x, dtype=jnp.float64)
    x2 = x * x
    term0 = x
    sum0 = x

    def body(k, state):
        term, s = state
        den = jnp.float64(k + 1) * jnp.float64(2 * k + 3)
        term = term * (-x2) / den
        return term, s + term

    _, s = lax.fori_loop(0, _ERF_TERMS - 1, body, (term0, sum0))
    return _TWO_OVER_SQRT_PI * s


def _real_erfi(x: jax.Array) -> jax.Array:
    x = jnp.asarray(x, dtype=jnp.float64)
    return jnp.real(_complex_erfi_series(jnp.complex128(x + 0.0j)))


def _real_erfinv_scalar(y: jax.Array) -> jax.Array:
    y = jnp.asarray(y, dtype=jnp.float64)
    ln = jnp.log(1.0 - y * y)
    t = 2.0 / (jnp.pi * _ERFINV_A) + 0.5 * ln
    x0 = jnp.sqrt(jnp.maximum(0.0, jnp.sqrt(t * t - ln / _ERFINV_A) - t))
    x0 = jnp.where(y < 0.0, -x0, x0)

    def body(_, x):
        err = jsp.erf(x) - y
        der = _TWO_OVER_SQRT_PI * jnp.exp(-x * x)
        return x - err / der

    x = lax.fori_loop(0, _ERFINV_ITERS, body, x0)
    valid = jnp.isfinite(y) & (jnp.abs(y) < 1.0)
    return jnp.where(valid, x, jnp.nan)


def _real_hyp1f1_scalar(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)
    term0 = jnp.float64(1.0)
    sum0 = term0

    def body(k, state):
        term, s = state
        kf = jnp.float64(k)
        term = term * (a + kf) / (b + kf)
        term = term * (z / jnp.float64(k + 1))
        return term, s + term

    _, s = lax.fori_loop(0, _HYP_TERMS - 1, body, (term0, sum0))
    return s


def _real_hyp1f1_scalar_tail(a: jax.Array, b: jax.Array, z: jax.Array, n_terms: int) -> tuple[jax.Array, jax.Array]:
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)
    term0 = jnp.float64(1.0)
    sum0 = term0

    def body(k, state):
        term, s = state
        kf = jnp.float64(k)
        step = (a + kf) / (b + kf)
        step = step * (z / jnp.float64(k + 1))
        term = term * step
        return term, s + term

    term, s = lax.fori_loop(0, n_terms - 1, body, (term0, sum0))
    k_last = jnp.float64(n_terms - 1)
    ratio = jnp.abs(z) * jnp.abs(a + k_last) / (jnp.abs(b + k_last) * (k_last + 1.0))
    tail = _series_tail_bound_geom(jnp.abs(term), ratio)
    return s, tail


def _real_hyp0f1_scalar(b: jax.Array, z: jax.Array) -> jax.Array:
    b = jnp.asarray(b, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)
    term0 = jnp.float64(1.0)
    sum0 = term0

    def body(k, state):
        term, s = state
        kf = jnp.float64(k)
        bk = b + kf
        term = term * (z / bk) / jnp.float64(k + 1)
        return term, s + term

    _, s = lax.fori_loop(0, _HYP_TERMS - 1, body, (term0, sum0))
    return s


def _real_hyp0f1_scalar_tail(b: jax.Array, z: jax.Array, n_terms: int) -> tuple[jax.Array, jax.Array]:
    b = jnp.asarray(b, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)
    term0 = jnp.float64(1.0)
    sum0 = term0

    def body(k, state):
        term, s = state
        kf = jnp.float64(k)
        step = (z / (b + kf)) / jnp.float64(k + 1)
        term = term * step
        return term, s + term

    term, s = lax.fori_loop(0, n_terms - 1, body, (term0, sum0))
    k_last = jnp.float64(n_terms - 1)
    ratio = jnp.abs(z) / (jnp.abs(b + k_last) * (k_last + 1.0))
    tail = _series_tail_bound_geom(jnp.abs(term), ratio)
    return s, tail


def _real_hyp2f1_scalar(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array) -> jax.Array:
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    c = jnp.asarray(c, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)
    term0 = jnp.float64(1.0)
    sum0 = term0

    def body(k, state):
        term, s = state
        kf = jnp.float64(k)
        term = term * (a + kf) * (b + kf) / ((c + kf) * jnp.float64(k + 1))
        term = term * z
        return term, s + term

    _, s = lax.fori_loop(0, _HYP_TERMS - 1, body, (term0, sum0))
    return s


def _real_hyp2f1_scalar_tail(
    a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, n_terms: int
) -> tuple[jax.Array, jax.Array]:
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    c = jnp.asarray(c, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)
    term0 = jnp.float64(1.0)
    sum0 = term0

    def body(k, state):
        term, s = state
        kf = jnp.float64(k)
        step = (a + kf) * (b + kf) / ((c + kf) * jnp.float64(k + 1))
        step = step * z
        term = term * step
        return term, s + term

    term, s = lax.fori_loop(0, n_terms - 1, body, (term0, sum0))
    k_last = jnp.float64(n_terms - 1)
    ratio = jnp.abs(z) * jnp.abs(a + k_last) * jnp.abs(b + k_last) / (
        jnp.abs(c + k_last) * (k_last + 1.0)
    )
    tail = _series_tail_bound_geom(jnp.abs(term), ratio)
    return s, tail


def _real_hyp1f1_regime(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    z = jnp.asarray(z, dtype=jnp.float64)
    use_kummer = z > 6.0
    if _HAS_HYP1F1:
        direct = jsp.hyp1f1(a, b, z)
        transformed = jnp.exp(z) * jsp.hyp1f1(b - a, b, -z)
        return jnp.where(use_kummer, transformed, direct)
    transformed = jnp.exp(z) * _real_hyp1f1_scalar(b - a, b, -z)
    return jnp.where(use_kummer, transformed, _real_hyp1f1_scalar(a, b, z))


def _real_hyp0f1_regime(b: jax.Array, z: jax.Array) -> jax.Array:
    z = jnp.asarray(z, dtype=jnp.float64)
    absz = jnp.abs(z)
    use_bessel = absz > 9.0
    if _HAS_IV and _HAS_JV:
        r = jnp.sqrt(jnp.abs(z))
        order = b - 1.0
        iv = jsp.iv(order, 2.0 * r)
        jv = jsp.jv(order, 2.0 * r)
        scale = jsp.gamma(b) / jnp.power(r, order)
        val = jnp.where(z >= 0.0, scale * iv, scale * jv)
        return jnp.where(use_bessel, val, _real_hyp0f1_scalar(b, z))
    return _real_hyp0f1_scalar(b, z)


def _real_hyp2f1_regime(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array) -> jax.Array:
    z = jnp.asarray(z, dtype=jnp.float64)
    absz = jnp.abs(z)
    use_transform = absz > 0.75
    zt = jnp.where(z == 1.0, jnp.nan, z / (z - 1.0))
    if _HAS_HYP2F1:
        direct = jsp.hyp2f1(a, b, c, z)
        transformed = jnp.power(1.0 - z, -a) * jsp.hyp2f1(a, c - b, c, zt)
        return jnp.where(use_transform, transformed, direct)
    transformed = jnp.power(1.0 - z, -a) * _real_hyp2f1_scalar(a, c - b, c, zt)
    return jnp.where(use_transform, transformed, _real_hyp2f1_scalar(a, b, c, z))


def _complex_hyp1f1_scalar(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    a = jnp.asarray(a, dtype=jnp.complex128)
    b = jnp.asarray(b, dtype=jnp.complex128)
    z = jnp.asarray(z, dtype=jnp.complex128)
    term0 = jnp.complex128(1.0 + 0.0j)
    sum0 = term0

    def body(k, state):
        term, s = state
        kf = jnp.float64(k)
        term = term * (a + kf) / (b + kf)
        term = term * (z / jnp.float64(k + 1))
        return term, s + term

    _, s = lax.fori_loop(0, _HYP_TERMS - 1, body, (term0, sum0))
    return s


def _complex_hyp0f1_scalar(b: jax.Array, z: jax.Array) -> jax.Array:
    b = jnp.asarray(b, dtype=jnp.complex128)
    z = jnp.asarray(z, dtype=jnp.complex128)
    term0 = jnp.complex128(1.0 + 0.0j)
    sum0 = term0

    def body(k, state):
        term, s = state
        kf = jnp.complex128(jnp.float64(k))
        bk = b + kf
        term = term * (z / bk) / jnp.complex128(jnp.float64(k + 1))
        return term, s + term

    _, s = lax.fori_loop(0, _HYP_TERMS - 1, body, (term0, sum0))
    return s


def _complex_hyp2f1_scalar(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array) -> jax.Array:
    a = jnp.asarray(a, dtype=jnp.complex128)
    b = jnp.asarray(b, dtype=jnp.complex128)
    c = jnp.asarray(c, dtype=jnp.complex128)
    z = jnp.asarray(z, dtype=jnp.complex128)
    term0 = jnp.complex128(1.0 + 0.0j)
    sum0 = term0

    def body(k, state):
        term, s = state
        kf = jnp.complex128(jnp.float64(k))
        term = term * (a + kf) * (b + kf) / ((c + kf) * jnp.complex128(jnp.float64(k + 1)))
        term = term * z
        return term, s + term

    _, s = lax.fori_loop(0, _HYP_TERMS - 1, body, (term0, sum0))
    return s


def _real_hypu_scalar(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    s = jnp.sin(jnp.pi * b)
    m1 = _real_hyp1f1_scalar(a, b, z)
    m2 = _real_hyp1f1_scalar(1.0 + a - b, 2.0 - b, z)
    t1 = m1 / jnp.exp(jsp.gammaln(1.0 + a - b))
    t2 = jnp.power(z, 1.0 - b) * m2 / jnp.exp(jsp.gammaln(a))
    val = jnp.pi * (t1 - t2) / s
    return jnp.where(jnp.abs(s) < 1e-8, jnp.nan, val)


def _complex_hypu_scalar(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    s = jnp.sin(jnp.pi * b)
    m1 = _complex_hyp1f1_scalar(a, b, z)
    m2 = _complex_hyp1f1_scalar(1.0 + a - b, 2.0 - b, z)
    t1 = m1 / jnp.exp(_complex_loggamma(1.0 + a - b))
    t2 = jnp.exp((1.0 - b) * jnp.log(z)) * m2 / jnp.exp(_complex_loggamma(a))
    val = jnp.pi * (t1 - t2) / s
    return jnp.where(jnp.abs(s) < 1e-8, jnp.nan + 1j * jnp.nan, val)


def _real_bessel_series(nu: jax.Array, z: jax.Array, sign: float) -> jax.Array:
    nu = jnp.asarray(nu, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)
    half = 0.5 * z
    term0 = jnp.power(half, nu) / jnp.exp(jsp.gammaln(nu + 1.0))
    sum0 = term0
    z2 = z * z

    def body(k, state):
        term, s = state
        k1 = jnp.float64(k + 1)
        den = k1 * (k1 + nu)
        num = 0.25 * sign * z2
        term = term * (num / den)
        return term, s + term

    _, s = lax.fori_loop(0, _BESSEL_TERMS - 1, body, (term0, sum0))
    return s


def _real_bessel_eval_j(nu: jax.Array, z: jax.Array) -> jax.Array:
    if _HAS_JV:
        return lax.cond(
            jnp.abs(z) > 12.0,
            lambda _: jsp.jv(nu, z),
            lambda _: _real_bessel_series(nu, z, -1.0),
            operand=None,
        )
    return _real_bessel_series(nu, z, -1.0)


def _real_bessel_eval_i(nu: jax.Array, z: jax.Array) -> jax.Array:
    if _HAS_IV:
        return lax.cond(
            jnp.abs(z) > 12.0,
            lambda _: jsp.iv(nu, z),
            lambda _: _real_bessel_series(nu, z, 1.0),
            operand=None,
        )
    return _real_bessel_series(nu, z, 1.0)


def _real_bessel_eval_y(nu: jax.Array, z: jax.Array) -> jax.Array:
    if _HAS_YV:
        return lax.cond(
            jnp.abs(z) > 12.0,
            lambda _: jsp.yv(nu, z),
            lambda _: _real_bessel_y(nu, z),
            operand=None,
        )
    return _real_bessel_y(nu, z)


def _real_bessel_eval_k(nu: jax.Array, z: jax.Array) -> jax.Array:
    if _HAS_KV:
        use_asym = (jnp.abs(z) > 12.0) & (z > 0.0)
        return lax.cond(
            use_asym,
            lambda _: jsp.kv(nu, z),
            lambda _: _real_bessel_k(nu, z),
            operand=None,
        )
    return _real_bessel_k(nu, z)


def _complex_bessel_series(nu: jax.Array, z: jax.Array, sign: float) -> jax.Array:
    nu = jnp.asarray(nu, dtype=jnp.complex128)
    z = jnp.asarray(z, dtype=jnp.complex128)
    half = 0.5 * z
    pow_half = jnp.exp(nu * jnp.log(half))
    gamma = jnp.exp(_complex_loggamma(nu + 1.0))
    term0 = pow_half / gamma
    sum0 = term0
    z2 = z * z

    def body(k, state):
        term, s = state
        k1 = jnp.float64(k + 1)
        den = k1 * (nu + k1)
        num = (0.25 * sign) * z2
        term = term * (num / den)
        return term, s + term

    _, s = lax.fori_loop(0, _BESSEL_TERMS - 1, body, (term0, sum0))
    return s


def _real_bessel_y(nu: jax.Array, z: jax.Array) -> jax.Array:
    s = jnp.sin(jnp.pi * nu)
    jnu = _real_bessel_series(nu, z, -1.0)
    jneg = _real_bessel_series(-nu, z, -1.0)
    val = (jnu * jnp.cos(jnp.pi * nu) - jneg) / s
    return jnp.where(jnp.abs(s) < 1e-8, jnp.inf, val)


def _real_bessel_k(nu: jax.Array, z: jax.Array) -> jax.Array:
    s = jnp.sin(jnp.pi * nu)
    inu = _real_bessel_series(nu, z, 1.0)
    ineg = _real_bessel_series(-nu, z, 1.0)
    val = 0.5 * jnp.pi * (ineg - inu) / s
    return jnp.where(jnp.abs(s) < 1e-8, jnp.inf, val)


def _complex_bessel_y(nu: jax.Array, z: jax.Array) -> jax.Array:
    s = jnp.sin(jnp.pi * nu)
    jnu = _complex_bessel_series(nu, z, -1.0)
    jneg = _complex_bessel_series(-nu, z, -1.0)
    val = (jnu * jnp.cos(jnp.pi * nu) - jneg) / s
    return jnp.where(jnp.abs(s) < 1e-8, jnp.nan + 1j * jnp.nan, val)


def _complex_bessel_k(nu: jax.Array, z: jax.Array) -> jax.Array:
    s = jnp.sin(jnp.pi * nu)
    inu = _complex_bessel_series(nu, z, 1.0)
    ineg = _complex_bessel_series(-nu, z, 1.0)
    val = 0.5 * jnp.pi * (ineg - inu) / s
    return jnp.where(jnp.abs(s) < 1e-8, jnp.nan + 1j * jnp.nan, val)


def _interval_from_samples(vals: jax.Array) -> jax.Array:
    finite = jnp.all(jnp.isfinite(vals))
    out = di.interval(di._below(jnp.min(vals)), di._above(jnp.max(vals)))
    full = di.interval(-jnp.inf, jnp.inf)
    return jnp.where(finite, out, full)


def _interval_width(x: jax.Array) -> jax.Array:
    return x[1] - x[0]


def _select_tighter_interval(a: jax.Array, b: jax.Array) -> jax.Array:
    wa = _interval_width(a)
    wb = _interval_width(b)
    a_ok = jnp.isfinite(wa)
    b_ok = jnp.isfinite(wb)
    pick_a = (a_ok & b_ok & (wa <= wb)) | (a_ok & ~b_ok)
    pick_b = b_ok & ~a_ok
    full = _full_interval()
    return jnp.where(pick_a, a, jnp.where(pick_b, b, full))


def _interval_is_small(x: jax.Array, tol: float = 1e-6) -> jax.Array:
    return _interval_width(x) <= tol


def _series_tail_bound_geom(term_abs: jax.Array, ratio: jax.Array) -> jax.Array:
    ratio = jnp.abs(ratio)
    safe = (ratio < 0.95) & jnp.isfinite(ratio) & jnp.isfinite(term_abs)
    bound = term_abs * ratio / (1.0 - ratio)
    return jnp.where(safe, bound, jnp.inf)


def _series_interval_from_mid(sum_val: jax.Array, tail: jax.Array) -> jax.Array:
    lo = di._below(sum_val - tail)
    hi = di._above(sum_val + tail)
    return di.interval(lo, hi)


def _interval_midpoint(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    return 0.5 * (x[0] + x[1])


def _full_interval() -> jax.Array:
    return di.interval(-jnp.inf, jnp.inf)


def _taylor_series_unary(fn, x0: jax.Array, length: int) -> jax.Array:
    coeffs = []
    f = fn
    for k in range(length):
        val = f(x0)
        fact = jnp.exp(lax.lgamma(jnp.float64(k + 1)))
        coeffs.append(val / fact)
        f = jax.grad(f)
    return jnp.asarray(coeffs, dtype=jnp.float64)


def _series_intervals_from_midpoint(vals: jax.Array) -> jax.Array:
    finite = jnp.all(jnp.isfinite(vals))
    full = jnp.tile(_full_interval(), (vals.shape[0], 1))
    coeffs = jnp.stack([di._below(vals), di._above(vals)], axis=-1)
    return jnp.where(finite, coeffs, full)


def _fresnel_eval(x: jax.Array, normalized: bool) -> tuple[jax.Array, jax.Array]:
    if normalized:
        s, c = jsp.fresnel(x)
    else:
        t = x * _SQRT_2_OVER_PI
        s, c = jsp.fresnel(t)
        s = _SQRT_PI_OVER_2 * s
        c = _SQRT_PI_OVER_2 * c
    return s, c


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_rising_ui_forward(x: jax.Array, n: int) -> jax.Array:
    x = di.as_interval(x)

    def body(k: int, res: jax.Array) -> jax.Array:
        t = di.fast_add(x, di.interval(jnp.float64(k), jnp.float64(k)))
        return di.fast_mul(res, t)

    return lax.fori_loop(0, n, body, di.interval(1.0, 1.0))


@partial(jax.jit, static_argnames=("n", "prec_bits"))
def arb_hypgeom_rising_ui_forward_prec(x: jax.Array, n: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = di.round_interval_outward(di.as_interval(x), prec_bits)

    def body(k: int, res: jax.Array) -> jax.Array:
        t = di.fast_add_prec(x, di.interval(jnp.float64(k), jnp.float64(k)), prec_bits)
        return di.fast_mul_prec(res, t, prec_bits)

    return lax.fori_loop(0, n, body, di.round_interval_outward(di.interval(1.0, 1.0), prec_bits))


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_rising_ui(x: jax.Array, n: int) -> jax.Array:
    return arb_hypgeom_rising_ui_forward(x, n)


@partial(jax.jit, static_argnames=("n", "prec_bits"))
def arb_hypgeom_rising_ui_prec(x: jax.Array, n: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return arb_hypgeom_rising_ui_forward_prec(x, n, prec_bits)


@jax.jit
def arb_hypgeom_lgamma(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[0]
    b = x[1]
    m = 0.5 * (a + b)
    include_root = (a <= _DIGAMMA_ZERO) & (_DIGAMMA_ZERO <= b)

    vals = jnp.asarray(
        [
            jsp.gammaln(a),
            jsp.gammaln(b),
            jsp.gammaln(m),
            jsp.gammaln(_DIGAMMA_ZERO),
        ],
        dtype=jnp.float64,
    )
    mask = jnp.asarray([True, True, True, include_root], dtype=bool)
    vmin = jnp.min(jnp.where(mask, vals, jnp.inf))
    vmax = jnp.max(jnp.where(mask, vals, -jnp.inf))

    ok = (a > 0.0) & jnp.all(jnp.isfinite(jnp.where(mask, vals, 0.0)))
    out = di.interval(di._below(vmin), di._above(vmax))
    full = di.interval(-jnp.inf, jnp.inf)
    return jnp.where(ok, out, full)


@jax.jit
def arb_hypgeom_gamma(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[0]
    b = x[1]
    m = 0.5 * (a + b)
    include_root = (a <= _DIGAMMA_ZERO) & (_DIGAMMA_ZERO <= b)

    vals = jnp.asarray(
        [
            jnp.exp(jsp.gammaln(a)),
            jnp.exp(jsp.gammaln(b)),
            jnp.exp(jsp.gammaln(m)),
            jnp.exp(jsp.gammaln(_DIGAMMA_ZERO)),
        ],
        dtype=jnp.float64,
    )
    mask = jnp.asarray([True, True, True, include_root], dtype=bool)
    vmin = jnp.min(jnp.where(mask, vals, jnp.inf))
    vmax = jnp.max(jnp.where(mask, vals, -jnp.inf))
    ok = (a > 0.0) & jnp.all(jnp.isfinite(jnp.where(mask, vals, 0.0)))
    out = di.interval(di._below(vmin), di._above(vmax))
    full = di.interval(-jnp.inf, jnp.inf)
    return jnp.where(ok, out, full)


@jax.jit
def arb_hypgeom_rgamma(x: jax.Array) -> jax.Array:
    g = arb_hypgeom_gamma(x)
    return di.fast_div(di.interval(1.0, 1.0), g)


@jax.jit
def arb_hypgeom_erf(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[0]
    b = x[1]
    m = 0.5 * (a + b)
    vals = jnp.asarray([_real_erf_series(a), _real_erf_series(b), _real_erf_series(m)], dtype=jnp.float64)
    return di.interval(di._below(jnp.min(vals)), di._above(jnp.max(vals)))


@jax.jit
def arb_hypgeom_erfc(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[0]
    b = x[1]
    m = 0.5 * (a + b)
    vals = jnp.asarray([1.0 - _real_erf_series(a), 1.0 - _real_erf_series(b), 1.0 - _real_erf_series(m)], dtype=jnp.float64)
    return di.interval(di._below(jnp.min(vals)), di._above(jnp.max(vals)))


@jax.jit
def arb_hypgeom_erfi(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[0]
    b = x[1]
    m = 0.5 * (a + b)
    vals = jnp.asarray(
        [
            jnp.real(_complex_erfi_series(jnp.complex128(a + 0.0j))),
            jnp.real(_complex_erfi_series(jnp.complex128(b + 0.0j))),
            jnp.real(_complex_erfi_series(jnp.complex128(m + 0.0j))),
        ],
        dtype=jnp.float64,
    )
    return di.interval(di._below(jnp.min(vals)), di._above(jnp.max(vals)))


@jax.jit
def arb_hypgeom_erfinv(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[0]
    b = x[1]
    m = 0.5 * (a + b)
    vals = jnp.asarray([_real_erfinv_scalar(a), _real_erfinv_scalar(b), _real_erfinv_scalar(m)], dtype=jnp.float64)
    return _interval_from_samples(vals)


@jax.jit
def arb_hypgeom_erfcinv(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    y = di.interval(1.0 - x[1], 1.0 - x[0])
    return arb_hypgeom_erfinv(y)


@partial(jax.jit, static_argnames=("regularized",))
def arb_hypgeom_0f1(a: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    a = di.as_interval(a)
    z = di.as_interval(z)
    term0 = di.interval(1.0, 1.0)
    sum0 = di.interval(1.0, 1.0)

    def body(k, state):
        term, s = state
        kf = jnp.float64(k)
        ak = di.fast_add(a, di.interval(kf, kf))
        inv_k1 = di.interval(1.0 / jnp.float64(k + 1), 1.0 / jnp.float64(k + 1))
        step = di.fast_div(z, ak)
        step = di.fast_mul(step, inv_k1)
        term = di.fast_mul(term, step)
        return term, di.fast_add(s, term)

    _, s = lax.fori_loop(0, _HYP_TERMS - 1, body, (term0, sum0))
    if regularized:
        s = di.fast_mul(s, arb_hypgeom_rgamma(a))
    a_m = _interval_midpoint(a)
    z_vals = jnp.asarray([z[0], z[1], 0.5 * (z[0] + z[1])], dtype=jnp.float64)
    sample_vals = jnp.asarray([_real_hyp0f1_regime(a_m, zi) for zi in z_vals], dtype=jnp.float64)
    s_samples = _interval_from_samples(sample_vals)
    series_sum, series_tail = _real_hyp0f1_scalar_tail(a_m, _interval_midpoint(z), _HYP_TERMS)
    s_tail = _series_interval_from_mid(series_sum, series_tail)
    if regularized:
        s_samples = di.fast_mul(s_samples, arb_hypgeom_rgamma(a))
        s_tail = di.fast_mul(s_tail, arb_hypgeom_rgamma(a))
    if regularized:
        s_samples = di.fast_mul(s_samples, arb_hypgeom_rgamma(a))
    param_ok = _interval_is_small(a)
    candidate = _select_tighter_interval(s, s_tail)
    candidate = _select_tighter_interval(candidate, s_samples)
    return jnp.where(param_ok, candidate, s)


@partial(jax.jit, static_argnames=("regularized",))
def arb_hypgeom_1f1(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    z = di.as_interval(z)
    term0 = di.interval(1.0, 1.0)
    sum0 = di.interval(1.0, 1.0)

    def body(k, state):
        term, s = state
        kf = jnp.float64(k)
        ak = di.fast_add(a, di.interval(kf, kf))
        bk = di.fast_add(b, di.interval(kf, kf))
        step = di.fast_div(ak, bk)
        inv_k1 = di.interval(1.0 / jnp.float64(k + 1), 1.0 / jnp.float64(k + 1))
        term = di.fast_mul(term, step)
        term = di.fast_mul(term, z)
        term = di.fast_mul(term, inv_k1)
        return term, di.fast_add(s, term)

    _, s = lax.fori_loop(0, _HYP_TERMS - 1, body, (term0, sum0))
    if regularized:
        s = di.fast_mul(s, arb_hypgeom_rgamma(b))
    a_m = _interval_midpoint(a)
    b_m = _interval_midpoint(b)
    z_vals = jnp.asarray([z[0], z[1], 0.5 * (z[0] + z[1])], dtype=jnp.float64)
    sample_vals = jnp.asarray([_real_hyp1f1_regime(a_m, b_m, zi) for zi in z_vals], dtype=jnp.float64)
    s_samples = _interval_from_samples(sample_vals)
    series_sum, series_tail = _real_hyp1f1_scalar_tail(a_m, b_m, _interval_midpoint(z), _HYP_TERMS)
    s_tail = _series_interval_from_mid(series_sum, series_tail)
    if regularized:
        s_samples = di.fast_mul(s_samples, arb_hypgeom_rgamma(b))
        s_tail = di.fast_mul(s_tail, arb_hypgeom_rgamma(b))
    if regularized:
        s_samples = di.fast_mul(s_samples, arb_hypgeom_rgamma(b))
    param_ok = _interval_is_small(a) & _interval_is_small(b)
    candidate = _select_tighter_interval(s, s_tail)
    candidate = _select_tighter_interval(candidate, s_samples)
    return jnp.where(param_ok, candidate, s)


@partial(jax.jit, static_argnames=("regularized",))
def arb_hypgeom_1f1_integration(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_1f1(a, b, z, regularized=regularized)


@partial(jax.jit, static_argnames=("regularized",))
def arb_hypgeom_m(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_1f1(a, b, z, regularized=regularized)


@partial(jax.jit, static_argnames=("regularized",))
def arb_hypgeom_2f1(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    c = di.as_interval(c)
    z = di.as_interval(z)
    term0 = di.interval(1.0, 1.0)
    sum0 = di.interval(1.0, 1.0)

    def body(k, state):
        term, s = state
        kf = jnp.float64(k)
        ak = di.fast_add(a, di.interval(kf, kf))
        bk = di.fast_add(b, di.interval(kf, kf))
        ck = di.fast_add(c, di.interval(kf, kf))
        k1 = di.interval(jnp.float64(k + 1), jnp.float64(k + 1))
        num = di.fast_mul(ak, bk)
        den = di.fast_mul(ck, k1)
        step = di.fast_div(num, den)
        term = di.fast_mul(term, step)
        term = di.fast_mul(term, z)
        return term, di.fast_add(s, term)

    _, s = lax.fori_loop(0, _HYP_TERMS - 1, body, (term0, sum0))
    if regularized:
        s = di.fast_mul(s, arb_hypgeom_rgamma(c))
    a_m = _interval_midpoint(a)
    b_m = _interval_midpoint(b)
    c_m = _interval_midpoint(c)
    z_vals = jnp.asarray([z[0], z[1], 0.5 * (z[0] + z[1])], dtype=jnp.float64)
    sample_vals = jnp.asarray([_real_hyp2f1_regime(a_m, b_m, c_m, zi) for zi in z_vals], dtype=jnp.float64)
    s_samples = _interval_from_samples(sample_vals)
    series_sum, series_tail = _real_hyp2f1_scalar_tail(a_m, b_m, c_m, _interval_midpoint(z), _HYP_TERMS)
    s_tail = _series_interval_from_mid(series_sum, series_tail)
    if regularized:
        s_samples = di.fast_mul(s_samples, arb_hypgeom_rgamma(c))
        s_tail = di.fast_mul(s_tail, arb_hypgeom_rgamma(c))
    if regularized:
        s_samples = di.fast_mul(s_samples, arb_hypgeom_rgamma(c))
    param_ok = _interval_is_small(a) & _interval_is_small(b) & _interval_is_small(c)
    candidate = _select_tighter_interval(s, s_tail)
    candidate = _select_tighter_interval(candidate, s_samples)
    return jnp.where(param_ok, candidate, s)


@partial(jax.jit, static_argnames=("regularized",))
def arb_hypgeom_2f1_integration(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_2f1(a, b, c, z, regularized=regularized)


@jax.jit
def arb_hypgeom_u(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    z = di.as_interval(z)
    a_vals = jnp.asarray([a[0], a[1], 0.5 * (a[0] + a[1])], dtype=jnp.float64)
    b_vals = jnp.asarray([b[0], b[1], 0.5 * (b[0] + b[1])], dtype=jnp.float64)
    z_vals = jnp.asarray([z[0], z[1], 0.5 * (z[0] + z[1])], dtype=jnp.float64)

    def eval_val(aa: jax.Array, bb: jax.Array, zz: jax.Array) -> jax.Array:
        zpos = zz > 0.0
        use_asym = zpos & (jnp.abs(zz) > 8.0)
        if _HAS_HYPERU:
            direct = jsp.hyperu(aa, bb, zz)
            asym = jnp.power(zz, -aa)
            return jnp.where(use_asym, asym, direct)
        asym = jnp.power(zz, -aa)
        return jnp.where(use_asym, asym, _real_hypu_scalar(aa, bb, zz))

    vals = jax.vmap(lambda aa: jax.vmap(lambda bb: jax.vmap(lambda zz: eval_val(aa, bb, zz))(z_vals))(b_vals))(a_vals)
    vals = vals.reshape(-1)
    s_series = _interval_from_samples(vals)
    a_m = _interval_midpoint(a)
    b_m = _interval_midpoint(b)
    z_vals2 = jnp.asarray([z[0], z[1], 0.5 * (z[0] + z[1])], dtype=jnp.float64)
    sample_vals = jnp.asarray([eval_val(a_m, b_m, zi) for zi in z_vals2], dtype=jnp.float64)
    s_samples = _interval_from_samples(sample_vals)
    param_ok = _interval_is_small(a) & _interval_is_small(b)
    return jnp.where(param_ok, _select_tighter_interval(s_series, s_samples), s_series)


@jax.jit
def arb_hypgeom_u_integration(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    return arb_hypgeom_u(a, b, z)


@partial(jax.jit, static_argnames=("normalized",))
def arb_hypgeom_fresnel(z: jax.Array, normalized: bool = False) -> tuple[jax.Array, jax.Array]:
    z = di.as_interval(z)
    a = z[0]
    b = z[1]
    m = 0.5 * (a + b)
    s_vals = []
    c_vals = []
    for x in (a, b, m):
        s, c = _fresnel_eval(x, normalized)
        s_vals.append(s)
        c_vals.append(c)
    s_arr = jnp.asarray(s_vals, dtype=jnp.float64)
    c_arr = jnp.asarray(c_vals, dtype=jnp.float64)
    return _interval_from_samples(s_arr), _interval_from_samples(c_arr)


@jax.jit
def arb_hypgeom_ei(z: jax.Array) -> jax.Array:
    z = di.as_interval(z)
    a = z[0]
    b = z[1]
    m = 0.5 * (a + b)
    vals = jnp.asarray([jsp.expi(a), jsp.expi(b), jsp.expi(m)], dtype=jnp.float64)
    return _interval_from_samples(vals)


def _si_ci_from_series(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    x = jnp.asarray(x, dtype=jnp.float64)
    x2 = x * x

    def si_body(k, state):
        term, acc = state
        denom = jnp.float64((2 * k) * (2 * k + 1))
        term = term * (-x2) / denom
        acc = acc + term / jnp.float64(2 * k + 1)
        return term, acc

    term0 = x
    si0 = term0
    _, si = lax.fori_loop(1, _SI_CI_TERMS, si_body, (term0, si0))

    def ci_body(k, state):
        term, acc = state
        denom = jnp.float64((2 * k - 1) * (2 * k))
        term = term * (-x2) / denom
        acc = acc + term / jnp.float64(2 * k)
        return term, acc

    term1 = -x2 / jnp.float64(2.0)
    ci0 = jnp.euler_gamma + jnp.log(jnp.abs(x)) + term1 / jnp.float64(2.0)
    _, ci = lax.fori_loop(2, _SI_CI_TERMS, ci_body, (term1, ci0))
    ci = jnp.where(x == 0.0, -jnp.inf, ci)
    return si, ci


def _si_ci_asymp(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    x = jnp.asarray(x, dtype=jnp.float64)
    x_abs = jnp.abs(x)
    inv = jnp.where(x_abs == 0.0, jnp.inf, 1.0 / x_abs)
    inv2 = inv * inv
    c = jnp.cos(x_abs)
    s = jnp.sin(x_abs)
    a = 1.0 - 2.0 * inv2 + 24.0 * inv2 * inv2
    b = 1.0 - 6.0 * inv2 + 120.0 * inv2 * inv2
    si_pos = 0.5 * jnp.pi - c * inv * a - s * inv2 * b
    ci_pos = s * inv * a + c * inv2 * b
    si = jnp.where(x < 0.0, -si_pos, si_pos)
    return si, ci_pos


@jax.jit
def arb_hypgeom_si(z: jax.Array) -> jax.Array:
    z = di.as_interval(z)
    a = z[0]
    b = z[1]
    m = 0.5 * (a + b)
    def eval_val(x: jax.Array) -> jax.Array:
        use_series = jnp.abs(x) <= 4.0
        v_series, _ = _si_ci_from_series(x)
        v_asymp, _ = _si_ci_asymp(x)
        return jnp.where(use_series, v_series, v_asymp)

    vals = jnp.asarray([eval_val(a), eval_val(b), eval_val(m)], dtype=jnp.float64)
    return _interval_from_samples(vals)


@jax.jit
def arb_hypgeom_ci(z: jax.Array) -> jax.Array:
    z = di.as_interval(z)
    a = z[0]
    b = z[1]
    m = 0.5 * (a + b)
    def eval_val(x: jax.Array) -> jax.Array:
        use_series = jnp.abs(x) <= 4.0
        _, v_series = _si_ci_from_series(x)
        _, v_asymp = _si_ci_asymp(x)
        return jnp.where(use_series, v_series, v_asymp)

    vals = jnp.asarray([eval_val(a), eval_val(b), eval_val(m)], dtype=jnp.float64)
    return _interval_from_samples(vals)


@jax.jit
def arb_hypgeom_shi(z: jax.Array) -> jax.Array:
    z = di.as_interval(z)
    a = z[0]
    b = z[1]
    m = 0.5 * (a + b)
    vals = jnp.asarray(
        [
            0.5 * (jsp.expi(a) - jsp.expi(-a)),
            0.5 * (jsp.expi(b) - jsp.expi(-b)),
            0.5 * (jsp.expi(m) - jsp.expi(-m)),
        ],
        dtype=jnp.float64,
    )
    return _interval_from_samples(vals)


@jax.jit
def arb_hypgeom_chi(z: jax.Array) -> jax.Array:
    z = di.as_interval(z)
    a = z[0]
    b = z[1]
    m = 0.5 * (a + b)
    vals = jnp.asarray(
        [
            0.5 * (jsp.expi(a) + jsp.expi(-a)),
            0.5 * (jsp.expi(b) + jsp.expi(-b)),
            0.5 * (jsp.expi(m) + jsp.expi(-m)),
        ],
        dtype=jnp.float64,
    )
    return _interval_from_samples(vals)


@partial(jax.jit, static_argnames=("offset",))
def arb_hypgeom_li(z: jax.Array, offset: int = 0) -> jax.Array:
    z = di.as_interval(z)
    a = z[0]
    b = z[1]
    m = 0.5 * (a + b)
    offset_term = jnp.float64(0.0)
    if offset > 0:
        offset_term = jsp.expi(jnp.log(jnp.float64(offset)))

    def eval_val(x: jax.Array) -> jax.Array:
        valid = x > 0.0
        v = jsp.expi(jnp.log(x)) - offset_term
        return jnp.where(valid, v, jnp.nan)

    vals = jnp.asarray([eval_val(a), eval_val(b), eval_val(m)], dtype=jnp.float64)
    return _interval_from_samples(vals)


@jax.jit
def arb_hypgeom_dilog(z: jax.Array) -> jax.Array:
    z = di.as_interval(z)
    a = z[0]
    b = z[1]
    m = 0.5 * (a + b)
    vals = jnp.asarray(
        [
            jsp.spence(1.0 - a),
            jsp.spence(1.0 - b),
            jsp.spence(1.0 - m),
        ],
        dtype=jnp.float64,
    )
    return _interval_from_samples(vals)


def _airy_series(z: jax.Array, sign: float) -> tuple[jax.Array, jax.Array]:
    z = jnp.asarray(z, dtype=jnp.float64)
    z3 = z * z * z
    inv_z = jnp.where(z == 0.0, 0.0, 1.0 / z)

    a0 = 1.0 / jsp.gamma(jnp.float64(2.0 / 3.0))
    b0 = 1.0 / (3.0 * jsp.gamma(jnp.float64(4.0 / 3.0)))

    term_a = a0
    term_b = b0
    sum_a = term_a
    sum_b = term_b
    sum_da = jnp.float64(0.0)
    sum_db = term_b

    def body(k, state):
        term_a, term_b, sum_a, sum_b, sum_da, sum_db = state
        kf = jnp.float64(k)
        term_a = term_a * (sign * z3) / (9.0 * kf * (kf - 1.0 / 3.0))
        term_b = term_b * (sign * z3) / (9.0 * kf * (kf + 1.0 / 3.0))
        sum_a = sum_a + term_a
        sum_b = sum_b + term_b
        sum_da = sum_da + (3.0 * kf) * term_a * inv_z
        sum_db = sum_db + (3.0 * kf + 1.0) * term_b
        return term_a, term_b, sum_a, sum_b, sum_da, sum_db

    term_a, term_b, sum_a, sum_b, sum_da, sum_db = lax.fori_loop(
        1, _AIRY_TERMS, body, (term_a, term_b, sum_a, sum_b, sum_da, sum_db)
    )
    ai = sum_a + z * sum_b
    aip = sum_da + sum_db
    return ai, aip


@jax.jit
def arb_hypgeom_airy(z: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    z = di.as_interval(z)
    a = z[0]
    b = z[1]
    m = 0.5 * (a + b)

    def eval_val(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        ai, aip = _airy_series(x, -1.0)
        bi, bip = _airy_series(x, 1.0)
        return ai, aip, bi, bip

    vals = [eval_val(a), eval_val(b), eval_val(m)]
    ai_vals = jnp.asarray([v[0] for v in vals], dtype=jnp.float64)
    aip_vals = jnp.asarray([v[1] for v in vals], dtype=jnp.float64)
    bi_vals = jnp.asarray([v[2] for v in vals], dtype=jnp.float64)
    bip_vals = jnp.asarray([v[3] for v in vals], dtype=jnp.float64)
    return (
        _interval_from_samples(ai_vals),
        _interval_from_samples(aip_vals),
        _interval_from_samples(bi_vals),
        _interval_from_samples(bip_vals),
    )


@jax.jit
def arb_hypgeom_expint(s: jax.Array, z: jax.Array) -> jax.Array:
    s = di.as_interval(s)
    z = di.as_interval(z)
    s_m = 0.5 * (s[0] + s[1])
    n = jnp.rint(s_m).astype(jnp.int64)
    ok = (jnp.abs(s_m - n) < 1e-6) & (n >= 1)
    a = z[0]
    b = z[1]
    m = 0.5 * (a + b)

    def eval_val(x: jax.Array) -> jax.Array:
        return jsp.expn(n, x)

    vals = jnp.asarray([eval_val(a), eval_val(b), eval_val(m)], dtype=jnp.float64)
    out = _interval_from_samples(vals)
    full = di.interval(-jnp.inf, jnp.inf)
    return jnp.where(ok, out, full)


@jax.jit
def arb_hypgeom_gamma_lower(s: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    s = di.as_interval(s)
    z = di.as_interval(z)
    s_m = 0.5 * (s[0] + s[1])
    a = z[0]
    b = z[1]
    m = 0.5 * (a + b)

    def eval_val(x: jax.Array) -> jax.Array:
        v = jsp.gammainc(s_m, x)
        return v if regularized else jsp.gamma(s_m) * v

    vals = jnp.asarray([eval_val(a), eval_val(b), eval_val(m)], dtype=jnp.float64)
    return _interval_from_samples(vals)


@jax.jit
def arb_hypgeom_gamma_upper(s: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    s = di.as_interval(s)
    z = di.as_interval(z)
    s_m = 0.5 * (s[0] + s[1])
    a = z[0]
    b = z[1]
    m = 0.5 * (a + b)

    def eval_val(x: jax.Array) -> jax.Array:
        v = jsp.gammaincc(s_m, x)
        return v if regularized else jsp.gamma(s_m) * v

    vals = jnp.asarray([eval_val(a), eval_val(b), eval_val(m)], dtype=jnp.float64)
    return _interval_from_samples(vals)


@jax.jit
def arb_hypgeom_beta_lower(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    z = di.as_interval(z)
    a_m = 0.5 * (a[0] + a[1])
    b_m = 0.5 * (b[0] + b[1])
    x0 = z[0]
    x1 = z[1]
    x2 = 0.5 * (x0 + x1)

    def eval_val(x: jax.Array) -> jax.Array:
        v = jsp.betainc(a_m, b_m, x)
        return v if regularized else jnp.exp(jsp.betaln(a_m, b_m)) * v

    vals = jnp.asarray([eval_val(x0), eval_val(x1), eval_val(x2)], dtype=jnp.float64)
    return _interval_from_samples(vals)


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_chebyshev_t(n: int, z: jax.Array) -> jax.Array:
    z = di.as_interval(z)
    x = 0.5 * (z[0] + z[1])
    n = int(n)
    val = jnp.where(jnp.abs(x) <= 1.0, jnp.cos(n * jnp.arccos(x)), jnp.cosh(n * jnp.arccosh(x)))
    return _interval_from_midpoint(val)


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_chebyshev_u(n: int, z: jax.Array) -> jax.Array:
    z = di.as_interval(z)
    x = 0.5 * (z[0] + z[1])
    n = int(n)
    def in_range():
        t = jnp.arccos(x)
        return jnp.sin((n + 1) * t) / jnp.sin(t)

    def out_range():
        t = jnp.arccosh(jnp.abs(x))
        return jnp.sinh((n + 1) * t) / jnp.sinh(t)

    val = jnp.where(jnp.abs(x) <= 1.0, in_range(), out_range())
    return _interval_from_midpoint(val)


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_laguerre_l(n: int, m: jax.Array, z: jax.Array) -> jax.Array:
    n = int(n)
    m = di.as_interval(m)
    z = di.as_interval(z)
    m_m = 0.5 * (m[0] + m[1])
    x = 0.5 * (z[0] + z[1])

    def body(k, acc):
        kf = jnp.float64(k)
        coeff = jnp.exp(jsp.gammaln(n + m_m + 1.0) - jsp.gammaln(n - k + 1.0) - jsp.gammaln(m_m + k + 1.0))
        term = coeff * jnp.power(-x, kf) / jnp.exp(jsp.gammaln(kf + 1.0))
        return acc + term

    val = lax.fori_loop(0, n + 1, body, jnp.float64(0.0))
    return _interval_from_midpoint(val)


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_hermite_h(n: int, z: jax.Array) -> jax.Array:
    n = int(n)
    z = di.as_interval(z)
    x = 0.5 * (z[0] + z[1])
    if n == 0:
        return _interval_from_midpoint(jnp.float64(1.0))
    if n == 1:
        return _interval_from_midpoint(2.0 * x)
    h0 = jnp.float64(1.0)
    h1 = 2.0 * x

    def body(k, state):
        h_prev, h_curr = state
        h_next = 2.0 * x * h_curr - 2.0 * (k - 1) * h_prev
        return h_curr, h_next

    _, hn = lax.fori_loop(2, n + 1, body, (h0, h1))
    return _interval_from_midpoint(hn)


def _legendre_p_scalar(n: int, x: jax.Array) -> jax.Array:
    x = jnp.asarray(x, dtype=jnp.float64)
    if n == 0:
        return jnp.float64(1.0)
    if n == 1:
        return x
    p0 = jnp.float64(1.0)
    p1 = x

    def body(k, state):
        p_prev, p_curr = state
        kf = jnp.float64(k)
        p_next = ((2.0 * kf - 1.0) * x * p_curr - (kf - 1.0) * p_prev) / kf
        return p_curr, p_next

    _, pn = lax.fori_loop(2, n + 1, body, (p0, p1))
    return pn


def _legendre_q_scalar(n: int, x: jax.Array) -> jax.Array:
    x = jnp.asarray(x, dtype=jnp.float64)
    q0 = 0.5 * jnp.log((1.0 + x) / (1.0 - x))
    if n == 0:
        return q0
    q1 = x * q0 - 1.0
    if n == 1:
        return q1

    def body(k, state):
        q_prev, q_curr = state
        kf = jnp.float64(k)
        q_next = ((2.0 * kf - 1.0) * x * q_curr - (kf - 1.0) * q_prev) / kf
        return q_curr, q_next

    _, qn = lax.fori_loop(2, n + 1, body, (q0, q1))
    return qn


def _jacobi_p_scalar(n: int, a: jax.Array, b: jax.Array, x: jax.Array) -> jax.Array:
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    t1 = (x - 1.0) * 0.5
    t2 = (x + 1.0) * 0.5

    def body(k, acc):
        kf = jnp.float64(k)
        c1 = jnp.exp(jsp.gammaln(n + a + 1.0) - jsp.gammaln(n - k + 1.0) - jsp.gammaln(a + k + 1.0))
        c2 = jnp.exp(jsp.gammaln(n + b + 1.0) - jsp.gammaln(k + 1.0) - jsp.gammaln(b + n - k + 1.0))
        term = c1 * c2 * jnp.power(t1, kf) * jnp.power(t2, jnp.float64(n) - kf)
        return acc + term

    return lax.fori_loop(0, n + 1, body, jnp.float64(0.0))


def _gegenbauer_c_scalar(n: int, lam: jax.Array, x: jax.Array) -> jax.Array:
    return _jacobi_p_scalar(n, lam - 0.5, lam - 0.5, x) * jnp.exp(
        jsp.gammaln(n + 2.0 * lam) - jsp.gammaln(2.0 * lam) - jsp.gammaln(n + 1.0) + jsp.gammaln(lam + 0.5)
        - jsp.gammaln(n + lam + 0.5)
    )


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_legendre_p(n: int, m: jax.Array, z: jax.Array, type: int = 0) -> jax.Array:
    n = int(n)
    z = di.as_interval(z)
    m = di.as_interval(m)
    m_m = 0.5 * (m[0] + m[1])
    ok = (jnp.abs(m_m) <= 1e-12) & (n >= 0)
    a = z[0]
    b = z[1]
    x = 0.5 * (a + b)
    vals = jnp.asarray([_legendre_p_scalar(n, a), _legendre_p_scalar(n, b), _legendre_p_scalar(n, x)], dtype=jnp.float64)
    out = _interval_from_samples(vals)
    include_pos = (a <= 1.0) & (b >= 1.0)
    include_neg = (a <= -1.0) & (b >= -1.0)
    neg_val = jnp.float64(1.0) if (n % 2 == 0) else jnp.float64(-1.0)
    out_lo = jnp.where(include_pos, jnp.minimum(out[0], jnp.float64(1.0)), out[0])
    out_hi = jnp.where(include_pos, jnp.maximum(out[1], jnp.float64(1.0)), out[1])
    out_lo = jnp.where(include_neg, jnp.minimum(out_lo, neg_val), out_lo)
    out_hi = jnp.where(include_neg, jnp.maximum(out_hi, neg_val), out_hi)
    out = di.interval(out_lo, out_hi)
    return jnp.where(ok, out, _full_interval())


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_legendre_q(n: int, m: jax.Array, z: jax.Array, type: int = 0) -> jax.Array:
    n = int(n)
    z = di.as_interval(z)
    m = di.as_interval(m)
    m_m = 0.5 * (m[0] + m[1])
    a = z[0]
    b = z[1]
    x = 0.5 * (a + b)
    ok = (jnp.abs(m_m) <= 1e-12) & (n >= 0) & (a > -1.0) & (b < 1.0)
    vals = jnp.asarray([_legendre_q_scalar(n, a), _legendre_q_scalar(n, b), _legendre_q_scalar(n, x)], dtype=jnp.float64)
    out = _interval_from_samples(vals)
    return jnp.where(ok, out, _full_interval())


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_jacobi_p(n: int, a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    n = int(n)
    a = di.as_interval(a)
    b = di.as_interval(b)
    z = di.as_interval(z)
    a_m = 0.5 * (a[0] + a[1])
    b_m = 0.5 * (b[0] + b[1])
    a_z = z[0]
    b_z = z[1]
    x = 0.5 * (a_z + b_z)
    vals = jnp.asarray(
        [
            _jacobi_p_scalar(n, a_m, b_m, a_z),
            _jacobi_p_scalar(n, a_m, b_m, b_z),
            _jacobi_p_scalar(n, a_m, b_m, x),
        ],
        dtype=jnp.float64,
    )
    out = _interval_from_samples(vals)
    include_pos = (a_z <= 1.0) & (b_z >= 1.0)
    include_neg = (a_z <= -1.0) & (b_z >= -1.0)
    pos_val = jnp.exp(jsp.gammaln(n + a_m + 1.0) - jsp.gammaln(a_m + 1.0) - jsp.gammaln(n + 1.0))
    neg_val = jnp.exp(jsp.gammaln(n + b_m + 1.0) - jsp.gammaln(b_m + 1.0) - jsp.gammaln(n + 1.0))
    neg_val = jnp.where(n % 2 == 0, neg_val, -neg_val)
    out_lo = jnp.where(include_pos, jnp.minimum(out[0], pos_val), out[0])
    out_hi = jnp.where(include_pos, jnp.maximum(out[1], pos_val), out[1])
    out_lo = jnp.where(include_neg, jnp.minimum(out_lo, neg_val), out_lo)
    out_hi = jnp.where(include_neg, jnp.maximum(out_hi, neg_val), out_hi)
    return di.interval(out_lo, out_hi)


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_gegenbauer_c(n: int, m: jax.Array, z: jax.Array) -> jax.Array:
    n = int(n)
    m = di.as_interval(m)
    lam = 0.5 * (m[0] + m[1])
    z_int = di.as_interval(z)
    a_z = z_int[0]
    b_z = z_int[1]
    x = 0.5 * (a_z + b_z)
    vals = jnp.asarray(
        [
            _gegenbauer_c_scalar(n, lam, a_z),
            _gegenbauer_c_scalar(n, lam, b_z),
            _gegenbauer_c_scalar(n, lam, x),
        ],
        dtype=jnp.float64,
    )
    out = _interval_from_samples(vals)
    include_pos = (a_z <= 1.0) & (b_z >= 1.0)
    include_neg = (a_z <= -1.0) & (b_z >= -1.0)
    pos_val = jnp.exp(jsp.gammaln(n + 2.0 * lam) - jsp.gammaln(2.0 * lam) - jsp.gammaln(n + 1.0))
    neg_val = jnp.where(n % 2 == 0, pos_val, -pos_val)
    out_lo = jnp.where(include_pos, jnp.minimum(out[0], pos_val), out[0])
    out_hi = jnp.where(include_pos, jnp.maximum(out[1], pos_val), out[1])
    out_lo = jnp.where(include_neg, jnp.minimum(out_lo, neg_val), out_lo)
    out_hi = jnp.where(include_neg, jnp.maximum(out_hi, neg_val), out_hi)
    return di.interval(out_lo, out_hi)


@partial(jax.jit, static_argnames=("n",))
def arb_hypgeom_central_bin_ui(n: int) -> jax.Array:
    n = int(n)
    if n < 0:
        return di.interval(-jnp.inf, jnp.inf)
    val = jnp.exp(jsp.gammaln(2.0 * n + 1.0) - 2.0 * jsp.gammaln(n + 1.0))
    return _interval_from_midpoint(val)


def arb_hypgeom_rising(x: jax.Array, n: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    n = di.as_interval(n)
    n_m = 0.5 * (n[0] + n[1])
    n_int = jnp.rint(n_m).astype(jnp.int64)
    ok = jnp.abs(n_m - n_int) < 1e-6
    val = arb_hypgeom_rising_ui(x, int(n_int))
    return jnp.where(ok, val, _full_interval())


def arb_hypgeom_rising_ui_rs(x: jax.Array, n: int, m: int) -> jax.Array:
    return arb_hypgeom_rising_ui(x, n)


def arb_hypgeom_rising_ui_bs(x: jax.Array, n: int) -> jax.Array:
    return arb_hypgeom_rising_ui(x, n)


def arb_hypgeom_rising_ui_rec(x: jax.Array, n: int) -> jax.Array:
    return arb_hypgeom_rising_ui(x, n)


def arb_hypgeom_rising_ui_jet_powsum(x: jax.Array, n: int, length: int) -> jax.Array:
    x0 = _interval_midpoint(x)
    coeffs = _taylor_series_unary(lambda t: _interval_midpoint(arb_hypgeom_rising_ui(di.interval(t, t), n)), x0, length)
    return _series_intervals_from_midpoint(coeffs)


def arb_hypgeom_rising_ui_jet_rs(x: jax.Array, n: int, m: int, length: int) -> jax.Array:
    return arb_hypgeom_rising_ui_jet_powsum(x, n, length)


def arb_hypgeom_rising_ui_jet_bs(x: jax.Array, n: int, length: int) -> jax.Array:
    return arb_hypgeom_rising_ui_jet_powsum(x, n, length)


def arb_hypgeom_rising_ui_jet(x: jax.Array, n: int, length: int) -> jax.Array:
    return arb_hypgeom_rising_ui_jet_powsum(x, n, length)


def arb_hypgeom_gamma_fmpq(p: int, q: int) -> jax.Array:
    if q == 0:
        return _full_interval()
    val = jsp.gamma(jnp.float64(p) / jnp.float64(q))
    return _interval_from_midpoint(val)


def arb_hypgeom_gamma_fmpz(n: int) -> jax.Array:
    val = jsp.gamma(jnp.float64(n))
    return _interval_from_midpoint(val)


def arb_hypgeom_gamma_stirling(x: jax.Array, reciprocal: bool = False) -> jax.Array:
    val = arb_hypgeom_gamma(x)
    return di.fast_div(di.interval(1.0, 1.0), val) if reciprocal else val


def arb_hypgeom_gamma_stirling_sum_horner(z: jax.Array, n: int) -> jax.Array:
    return _interval_from_midpoint(jnp.float64(0.0))


def arb_hypgeom_gamma_stirling_sum_improved(z: jax.Array, n: int, k: int) -> jax.Array:
    return _interval_from_midpoint(jnp.float64(0.0))


def arb_hypgeom_gamma_lower_series(s: jax.Array, z: jax.Array, length: int, regularized: bool = False) -> jax.Array:
    s = di.as_interval(s)
    s_m = 0.5 * (s[0] + s[1])
    x0 = _interval_midpoint(z)

    def fn(x):
        v = jsp.gammainc(s_m, x)
        return v if regularized else jsp.gamma(s_m) * v

    coeffs = _taylor_series_unary(fn, x0, length)
    return _series_intervals_from_midpoint(coeffs)


def arb_hypgeom_gamma_upper_series(s: jax.Array, z: jax.Array, length: int, regularized: bool = False) -> jax.Array:
    s = di.as_interval(s)
    s_m = 0.5 * (s[0] + s[1])
    x0 = _interval_midpoint(z)

    def fn(x):
        v = jsp.gammaincc(s_m, x)
        return v if regularized else jsp.gamma(s_m) * v

    coeffs = _taylor_series_unary(fn, x0, length)
    return _series_intervals_from_midpoint(coeffs)


def arb_hypgeom_gamma_upper_integration(s: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_gamma_upper(s, z, regularized=regularized)


def arb_hypgeom_beta_lower_series(a: jax.Array, b: jax.Array, z: jax.Array, length: int, regularized: bool = False) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    a_m = 0.5 * (a[0] + a[1])
    b_m = 0.5 * (b[0] + b[1])
    x0 = _interval_midpoint(z)

    def fn(x):
        v = jsp.betainc(a_m, b_m, x)
        return v if regularized else jnp.exp(jsp.betaln(a_m, b_m)) * v

    coeffs = _taylor_series_unary(fn, x0, length)
    return _series_intervals_from_midpoint(coeffs)


def arb_hypgeom_fresnel_series(z: jax.Array, length: int, normalized: bool = False) -> tuple[jax.Array, jax.Array]:
    x0 = _interval_midpoint(z)

    def fn_s(x):
        s, _ = _fresnel_eval(x, normalized)
        return s

    def fn_c(x):
        _, c = _fresnel_eval(x, normalized)
        return c

    s_coeffs = _taylor_series_unary(fn_s, x0, length)
    c_coeffs = _taylor_series_unary(fn_c, x0, length)
    return _series_intervals_from_midpoint(s_coeffs), _series_intervals_from_midpoint(c_coeffs)


def arb_hypgeom_ei_series(z: jax.Array, length: int) -> jax.Array:
    x0 = _interval_midpoint(z)
    coeffs = _taylor_series_unary(jsp.expi, x0, length)
    return _series_intervals_from_midpoint(coeffs)


def arb_hypgeom_si_series(z: jax.Array, length: int) -> jax.Array:
    x0 = _interval_midpoint(z)
    coeffs = _taylor_series_unary(lambda x: _si_ci_from_series(x)[0], x0, length)
    return _series_intervals_from_midpoint(coeffs)


def arb_hypgeom_ci_series(z: jax.Array, length: int) -> jax.Array:
    x0 = _interval_midpoint(z)
    coeffs = _taylor_series_unary(lambda x: _si_ci_from_series(x)[1], x0, length)
    return _series_intervals_from_midpoint(coeffs)


def arb_hypgeom_shi_series(z: jax.Array, length: int) -> jax.Array:
    x0 = _interval_midpoint(z)
    coeffs = _taylor_series_unary(lambda x: 0.5 * (jsp.expi(x) - jsp.expi(-x)), x0, length)
    return _series_intervals_from_midpoint(coeffs)


def arb_hypgeom_chi_series(z: jax.Array, length: int) -> jax.Array:
    x0 = _interval_midpoint(z)
    coeffs = _taylor_series_unary(lambda x: 0.5 * (jsp.expi(x) + jsp.expi(-x)), x0, length)
    return _series_intervals_from_midpoint(coeffs)


def arb_hypgeom_li_series(z: jax.Array, length: int, offset: int = 0) -> jax.Array:
    x0 = _interval_midpoint(z)
    offset_term = jnp.float64(0.0)
    if offset > 0:
        offset_term = jsp.expi(jnp.log(jnp.float64(offset)))

    def fn(x):
        return jsp.expi(jnp.log(x)) - offset_term

    coeffs = _taylor_series_unary(fn, x0, length)
    return _series_intervals_from_midpoint(coeffs)


def arb_hypgeom_erf_series(z: jax.Array, length: int) -> jax.Array:
    x0 = _interval_midpoint(z)
    coeffs = _taylor_series_unary(jsp.erf, x0, length)
    return _series_intervals_from_midpoint(coeffs)


def arb_hypgeom_erfc_series(z: jax.Array, length: int) -> jax.Array:
    x0 = _interval_midpoint(z)
    coeffs = _taylor_series_unary(jsp.erfc, x0, length)
    return _series_intervals_from_midpoint(coeffs)


def arb_hypgeom_erfi_series(z: jax.Array, length: int) -> jax.Array:
    x0 = _interval_midpoint(z)
    coeffs = _taylor_series_unary(lambda x: _real_erfi(x), x0, length)
    return _series_intervals_from_midpoint(coeffs)


def arb_hypgeom_airy_jet(z: jax.Array, length: int) -> tuple[jax.Array, jax.Array]:
    x0 = _interval_midpoint(z)
    ai_coeffs = _taylor_series_unary(lambda x: _airy_series(x, -1.0)[0], x0, length)
    bi_coeffs = _taylor_series_unary(lambda x: _airy_series(x, 1.0)[0], x0, length)
    return _series_intervals_from_midpoint(ai_coeffs), _series_intervals_from_midpoint(bi_coeffs)


def arb_hypgeom_airy_series(z: jax.Array, length: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    x0 = _interval_midpoint(z)
    ai_coeffs = _taylor_series_unary(lambda x: _airy_series(x, -1.0)[0], x0, length)
    aip_coeffs = _taylor_series_unary(lambda x: _airy_series(x, -1.0)[1], x0, length)
    bi_coeffs = _taylor_series_unary(lambda x: _airy_series(x, 1.0)[0], x0, length)
    bip_coeffs = _taylor_series_unary(lambda x: _airy_series(x, 1.0)[1], x0, length)
    return (
        _series_intervals_from_midpoint(ai_coeffs),
        _series_intervals_from_midpoint(aip_coeffs),
        _series_intervals_from_midpoint(bi_coeffs),
        _series_intervals_from_midpoint(bip_coeffs),
    )


def arb_hypgeom_airy_zero(n: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    n = int(n)
    if n <= 0:
        return _full_interval(), _full_interval(), _full_interval(), _full_interval()
    t = (3.0 * jnp.pi * (n - 0.25) / 2.0) ** (2.0 / 3.0)
    z0 = -t
    return arb_hypgeom_airy(di.interval(z0, z0))


def arb_hypgeom_coulomb(l: jax.Array, eta: jax.Array, z: jax.Array) -> tuple[jax.Array, jax.Array]:
    l_m = _interval_midpoint(l)
    x0 = _interval_midpoint(z)
    phase = x0 - 0.5 * jnp.pi * l_m
    f = jnp.sin(phase)
    g = jnp.cos(phase)
    return _interval_from_midpoint(f), _interval_from_midpoint(g)


def arb_hypgeom_coulomb_jet(l: jax.Array, eta: jax.Array, z: jax.Array, length: int) -> tuple[jax.Array, jax.Array]:
    l_m = _interval_midpoint(l)
    x0 = _interval_midpoint(z)
    phase0 = x0 - 0.5 * jnp.pi * l_m
    f_coeffs = _taylor_series_unary(lambda t: jnp.sin(t - 0.5 * jnp.pi * l_m), x0, length)
    g_coeffs = _taylor_series_unary(lambda t: jnp.cos(t - 0.5 * jnp.pi * l_m), x0, length)
    return _series_intervals_from_midpoint(f_coeffs), _series_intervals_from_midpoint(g_coeffs)


def arb_hypgeom_coulomb_series(l: jax.Array, eta: jax.Array, z: jax.Array, length: int) -> tuple[jax.Array, jax.Array]:
    return arb_hypgeom_coulomb_jet(l, eta, z, length)


def arb_hypgeom_pfq(a: jax.Array, b: jax.Array, z: jax.Array, reciprocal: bool = False, n_terms: int = 32) -> jax.Array:
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    z_m = _interval_midpoint(z)

    def body(k, state):
        term, s = state
        k1 = jnp.float64(k + 1)
        num = jnp.prod(a + k) if a.size else 1.0
        den = jnp.prod(b + k) if b.size else 1.0
        step = (num / den) * (z_m / k1)
        term = term * step
        return term, s + term

    term0 = jnp.float64(1.0)
    sum0 = term0
    _, s = lax.fori_loop(0, n_terms - 1, body, (term0, sum0))
    if reciprocal:
        s = jnp.where(s == 0.0, jnp.inf, 1.0 / s)
    return _interval_from_midpoint(s)


def arb_hypgeom_legendre_p_ui(n: int, x: jax.Array) -> jax.Array:
    x0 = _interval_midpoint(x)
    if n < 0:
        return _full_interval()
    if n == 0:
        return _interval_from_midpoint(jnp.float64(1.0))
    if n == 1:
        return _interval_from_midpoint(x0)
    p0 = jnp.float64(1.0)
    p1 = x0

    def body(k, state):
        p_prev, p_curr = state
        kf = jnp.float64(k)
        p_next = ((2.0 * kf - 1.0) * x0 * p_curr - (kf - 1.0) * p_prev) / kf
        return p_curr, p_next

    _, pn = lax.fori_loop(2, n + 1, body, (p0, p1))
    return _interval_from_midpoint(pn)


def arb_hypgeom_legendre_p_ui_rec(n: int, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    x0 = _interval_midpoint(x)
    pn = _interval_midpoint(arb_hypgeom_legendre_p_ui(n, di.interval(x0, x0)))
    pn1 = _interval_midpoint(arb_hypgeom_legendre_p_ui(max(n - 1, 0), di.interval(x0, x0)))
    denom = x0 * x0 - 1.0
    deriv = jnp.where(jnp.abs(denom) < 1e-12, jnp.nan, (n * (x0 * pn - pn1) / denom))
    dval = jnp.where(jnp.isfinite(deriv), _interval_from_midpoint(deriv), _full_interval())
    return _interval_from_midpoint(pn), dval


def arb_hypgeom_legendre_p_ui_asymp(n: int, x: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
    return arb_hypgeom_legendre_p_ui_rec(n, x)


def arb_hypgeom_legendre_p_ui_one(n: int, x: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
    return arb_hypgeom_legendre_p_ui_rec(n, x)


def arb_hypgeom_legendre_p_ui_zero(n: int, x: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
    return arb_hypgeom_legendre_p_ui_rec(n, x)


def arb_hypgeom_legendre_p_ui_deriv_bound(n: int, x: jax.Array, x2sub1: jax.Array) -> tuple[jax.Array, jax.Array]:
    x0 = _interval_midpoint(x)
    pn = _interval_midpoint(arb_hypgeom_legendre_p_ui(n, di.interval(x0, x0)))
    pn1 = _interval_midpoint(arb_hypgeom_legendre_p_ui(max(n - 1, 0), di.interval(x0, x0)))
    denom = x0 * x0 - 1.0
    dp = jnp.where(jnp.abs(denom) < 1e-12, jnp.nan, (n * (x0 * pn - pn1) / denom))
    h = 1e-6
    pph = _interval_midpoint(arb_hypgeom_legendre_p_ui(n, di.interval(x0 + h, x0 + h)))
    pmh = _interval_midpoint(arb_hypgeom_legendre_p_ui(n, di.interval(x0 - h, x0 - h)))
    dp2 = (pph - 2.0 * pn + pmh) / (h * h)
    dp_int = jnp.where(jnp.isfinite(dp), _interval_from_midpoint(jnp.abs(dp)), _full_interval())
    dp2_int = jnp.where(jnp.isfinite(dp2), _interval_from_midpoint(jnp.abs(dp2)), _full_interval())
    return dp_int, dp2_int


def arb_hypgeom_legendre_p_ui_root(n: int, k: int) -> tuple[jax.Array, jax.Array]:
    n = int(n)
    k = int(k)
    if n <= 0 or k <= 0 or k > n:
        return _full_interval(), _full_interval()
    xk = jnp.cos(jnp.pi * (k - 0.25) / (n + 0.5))
    pn = _interval_midpoint(arb_hypgeom_legendre_p_ui(n, di.interval(xk, xk)))
    pn1 = _interval_midpoint(arb_hypgeom_legendre_p_ui(n - 1, di.interval(xk, xk)))
    denom = xk * xk - 1.0
    dp = jnp.where(jnp.abs(denom) < 1e-12, jnp.nan, (n * (xk * pn - pn1) / denom))
    w = jnp.where(jnp.isfinite(dp), 2.0 / ((1.0 - xk * xk) * dp * dp), jnp.nan)
    root_int = jnp.where(jnp.isfinite(xk), _interval_from_midpoint(xk), _full_interval())
    weight_int = jnp.where(jnp.isfinite(w), _interval_from_midpoint(w), _full_interval())
    return root_int, weight_int


def arb_hypgeom_sum_fmpq_arb(a: jax.Array, b: jax.Array, z: jax.Array, reciprocal: bool, n_terms: int) -> jax.Array:
    return arb_hypgeom_pfq(a, b, z, reciprocal=reciprocal, n_terms=n_terms)


def arb_hypgeom_sum_fmpq_arb_forward(a: jax.Array, b: jax.Array, z: jax.Array, reciprocal: bool, n_terms: int) -> jax.Array:
    return arb_hypgeom_sum_fmpq_arb(a, b, z, reciprocal, n_terms)


def arb_hypgeom_sum_fmpq_arb_rs(a: jax.Array, b: jax.Array, z: jax.Array, reciprocal: bool, n_terms: int) -> jax.Array:
    return arb_hypgeom_sum_fmpq_arb(a, b, z, reciprocal, n_terms)


def arb_hypgeom_sum_fmpq_arb_bs(a: jax.Array, b: jax.Array, z: jax.Array, reciprocal: bool, n_terms: int) -> jax.Array:
    return arb_hypgeom_sum_fmpq_arb(a, b, z, reciprocal, n_terms)


def arb_hypgeom_sum_fmpq_imag_arb(a: jax.Array, b: jax.Array, z: jax.Array, reciprocal: bool, n_terms: int) -> tuple[jax.Array, jax.Array]:
    res = arb_hypgeom_pfq(a, b, z, reciprocal=reciprocal, n_terms=n_terms)
    return res, di.interval(0.0, 0.0)


def arb_hypgeom_sum_fmpq_imag_arb_forward(
    a: jax.Array, b: jax.Array, z: jax.Array, reciprocal: bool, n_terms: int
) -> tuple[jax.Array, jax.Array]:
    return arb_hypgeom_sum_fmpq_imag_arb(a, b, z, reciprocal, n_terms)


def arb_hypgeom_sum_fmpq_imag_arb_rs(
    a: jax.Array, b: jax.Array, z: jax.Array, reciprocal: bool, n_terms: int
) -> tuple[jax.Array, jax.Array]:
    return arb_hypgeom_sum_fmpq_imag_arb(a, b, z, reciprocal, n_terms)


def arb_hypgeom_sum_fmpq_imag_arb_bs(
    a: jax.Array, b: jax.Array, z: jax.Array, reciprocal: bool, n_terms: int
) -> tuple[jax.Array, jax.Array]:
    return arb_hypgeom_sum_fmpq_imag_arb(a, b, z, reciprocal, n_terms)


@partial(jax.jit, static_argnames=("mode",))
def arb_hypgeom_bessel_j(nu: jax.Array, z: jax.Array, mode: str = "sample") -> jax.Array:
    mode = _validate_bessel_real_mode(mode)
    nu = di.as_interval(nu)
    z = di.as_interval(z)
    nu_m = 0.5 * (nu[0] + nu[1])
    a = z[0]
    b = z[1]
    m = 0.5 * (a + b)
    if mode == "midpoint":
        val = _real_bessel_eval_j(nu_m, m)
        return jnp.where(jnp.isfinite(val), _interval_from_midpoint(val), di.interval(-jnp.inf, jnp.inf))
    vals = jnp.asarray(
        [
            _real_bessel_eval_j(nu_m, a),
            _real_bessel_eval_j(nu_m, b),
            _real_bessel_eval_j(nu_m, m),
        ]
    )
    return _interval_from_samples(vals)


@partial(jax.jit, static_argnames=("mode",))
def arb_hypgeom_bessel_i(nu: jax.Array, z: jax.Array, mode: str = "sample") -> jax.Array:
    mode = _validate_bessel_real_mode(mode)
    nu = di.as_interval(nu)
    z = di.as_interval(z)
    nu_m = 0.5 * (nu[0] + nu[1])
    a = z[0]
    b = z[1]
    m = 0.5 * (a + b)
    if mode == "midpoint":
        val = _real_bessel_eval_i(nu_m, m)
        return jnp.where(jnp.isfinite(val), _interval_from_midpoint(val), di.interval(-jnp.inf, jnp.inf))
    vals = jnp.asarray(
        [
            _real_bessel_eval_i(nu_m, a),
            _real_bessel_eval_i(nu_m, b),
            _real_bessel_eval_i(nu_m, m),
        ]
    )
    return _interval_from_samples(vals)


@partial(jax.jit, static_argnames=("mode",))
def arb_hypgeom_bessel_y(nu: jax.Array, z: jax.Array, mode: str = "sample") -> jax.Array:
    mode = _validate_bessel_real_mode(mode)
    nu = di.as_interval(nu)
    z = di.as_interval(z)
    if mode == "midpoint":
        nu_m = 0.5 * (nu[0] + nu[1])
        z_m = 0.5 * (z[0] + z[1])
        val = _real_bessel_eval_y(nu_m, z_m)
        return jnp.where(jnp.isfinite(val), _interval_from_midpoint(val), di.interval(-jnp.inf, jnp.inf))
    nu_vals = jnp.asarray([nu[0], nu[1], 0.5 * (nu[0] + nu[1])], dtype=jnp.float64)
    z_vals = jnp.asarray([z[0], z[1], 0.5 * (z[0] + z[1])], dtype=jnp.float64)

    def eval_pair(nu_i, z_i):
        return _real_bessel_eval_y(nu_i, z_i)

    vals = jax.vmap(lambda ni: jax.vmap(lambda zi: eval_pair(ni, zi))(z_vals))(nu_vals).reshape(-1)
    finite = jnp.all(jnp.isfinite(vals))
    out = _interval_from_samples(vals)
    full = di.interval(-jnp.inf, jnp.inf)
    return jnp.where(finite, out, full)


@partial(jax.jit, static_argnames=("mode",))
def arb_hypgeom_bessel_k(nu: jax.Array, z: jax.Array, mode: str = "sample") -> jax.Array:
    mode = _validate_bessel_real_mode(mode)
    nu = di.as_interval(nu)
    z = di.as_interval(z)
    if mode == "midpoint":
        nu_m = 0.5 * (nu[0] + nu[1])
        z_m = 0.5 * (z[0] + z[1])
        val = _real_bessel_eval_k(nu_m, z_m)
        return jnp.where(jnp.isfinite(val), _interval_from_midpoint(val), di.interval(-jnp.inf, jnp.inf))
    nu_vals = jnp.asarray([nu[0], nu[1], 0.5 * (nu[0] + nu[1])], dtype=jnp.float64)
    z_vals = jnp.asarray([z[0], z[1], 0.5 * (z[0] + z[1])], dtype=jnp.float64)

    def eval_pair(nu_i, z_i):
        return _real_bessel_eval_k(nu_i, z_i)

    vals = jax.vmap(lambda ni: jax.vmap(lambda zi: eval_pair(ni, zi))(z_vals))(nu_vals).reshape(-1)
    finite = jnp.all(jnp.isfinite(vals))
    out = _interval_from_samples(vals)
    full = di.interval(-jnp.inf, jnp.inf)
    return jnp.where(finite, out, full)


@partial(jax.jit, static_argnames=("mode",))
def arb_hypgeom_bessel_jy(nu: jax.Array, z: jax.Array, mode: str = "sample") -> tuple[jax.Array, jax.Array]:
    return arb_hypgeom_bessel_j(nu, z, mode=mode), arb_hypgeom_bessel_y(nu, z, mode=mode)


@partial(jax.jit, static_argnames=("prec_bits", "mode"))
def arb_hypgeom_bessel_jy_prec(
    nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, mode: str = "sample"
) -> tuple[jax.Array, jax.Array]:
    return (
        arb_hypgeom_bessel_j_prec(nu, z, prec_bits=prec_bits, mode=mode),
        arb_hypgeom_bessel_y_prec(nu, z, prec_bits=prec_bits, mode=mode),
    )


@partial(jax.jit, static_argnames=("mode",))
def arb_hypgeom_bessel_i_scaled(nu: jax.Array, z: jax.Array, mode: str = "sample") -> jax.Array:
    i = arb_hypgeom_bessel_i(nu, z, mode=mode)
    m = 0.5 * (di.as_interval(z)[0] + di.as_interval(z)[1])
    return di.fast_mul(i, di.interval(jnp.exp(-m), jnp.exp(-m)))


@partial(jax.jit, static_argnames=("mode",))
def arb_hypgeom_bessel_k_scaled(nu: jax.Array, z: jax.Array, mode: str = "sample") -> jax.Array:
    k = arb_hypgeom_bessel_k(nu, z, mode=mode)
    m = 0.5 * (di.as_interval(z)[0] + di.as_interval(z)[1])
    return di.fast_mul(k, di.interval(jnp.exp(m), jnp.exp(m)))


@partial(jax.jit, static_argnames=("scaled", "mode"))
def arb_hypgeom_bessel_i_integration(nu: jax.Array, z: jax.Array, scaled: bool = False, mode: str = "sample") -> jax.Array:
    return arb_hypgeom_bessel_i_scaled(nu, z, mode=mode) if scaled else arb_hypgeom_bessel_i(nu, z, mode=mode)


@partial(jax.jit, static_argnames=("scaled", "mode"))
def arb_hypgeom_bessel_k_integration(nu: jax.Array, z: jax.Array, scaled: bool = False, mode: str = "sample") -> jax.Array:
    return arb_hypgeom_bessel_k_scaled(nu, z, mode=mode) if scaled else arb_hypgeom_bessel_k(nu, z, mode=mode)


@partial(jax.jit, static_argnames=("prec_bits", "mode"))
def arb_hypgeom_bessel_j_prec(
    nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, mode: str = "sample"
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_bessel_j(nu, z, mode=mode), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "mode"))
def arb_hypgeom_bessel_y_prec(
    nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, mode: str = "sample"
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_bessel_y(nu, z, mode=mode), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "mode"))
def arb_hypgeom_bessel_i_prec(
    nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, mode: str = "sample"
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_bessel_i(nu, z, mode=mode), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "mode"))
def arb_hypgeom_bessel_k_prec(
    nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, mode: str = "sample"
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_bessel_k(nu, z, mode=mode), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "mode"))
def arb_hypgeom_bessel_i_scaled_prec(
    nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, mode: str = "sample"
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_bessel_i_scaled(nu, z, mode=mode), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "mode"))
def arb_hypgeom_bessel_k_scaled_prec(
    nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, mode: str = "sample"
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_bessel_k_scaled(nu, z, mode=mode), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "scaled", "mode"))
def arb_hypgeom_bessel_i_integration_prec(
    nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, scaled: bool = False, mode: str = "sample"
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_bessel_i_integration(nu, z, scaled=scaled, mode=mode), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "scaled", "mode"))
def arb_hypgeom_bessel_k_integration_prec(
    nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, scaled: bool = False, mode: str = "sample"
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_bessel_k_integration(nu, z, scaled=scaled, mode=mode), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_hypgeom_lgamma_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_lgamma(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_hypgeom_gamma_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_gamma(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_hypgeom_rgamma_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_rgamma(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_hypgeom_erf_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_erf(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_hypgeom_erfc_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_erfc(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_hypgeom_erfi_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_erfi(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_hypgeom_erfinv_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_erfinv(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_hypgeom_erfcinv_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_erfcinv(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "regularized"))
def arb_hypgeom_0f1_prec(a: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_0f1(a, z, regularized=regularized), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "regularized"))
def arb_hypgeom_1f1_prec(a: jax.Array, b: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_1f1(a, b, z, regularized=regularized), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "regularized"))
def arb_hypgeom_m_prec(a: jax.Array, b: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_m(a, b, z, regularized=regularized), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "regularized"))
def arb_hypgeom_2f1_prec(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_2f1(a, b, c, z, regularized=regularized), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_hypgeom_u_prec(a: jax.Array, b: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_u(a, b, z), prec_bits)


def arb_hypgeom_fresnel_prec(
    z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, normalized: bool = False
) -> tuple[jax.Array, jax.Array]:
    s, c = arb_hypgeom_fresnel(z, normalized=normalized)
    return di.round_interval_outward(s, prec_bits), di.round_interval_outward(c, prec_bits)


def arb_hypgeom_ei_prec(z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_ei(z), prec_bits)


def arb_hypgeom_si_prec(z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_si(z), prec_bits)


def arb_hypgeom_ci_prec(z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_ci(z), prec_bits)


def arb_hypgeom_shi_prec(z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_shi(z), prec_bits)


def arb_hypgeom_chi_prec(z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_chi(z), prec_bits)


def arb_hypgeom_li_prec(z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, offset: int = 0) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_li(z, offset=offset), prec_bits)


def arb_hypgeom_dilog_prec(z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_dilog(z), prec_bits)


def arb_hypgeom_airy_prec(z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    ai, aip, bi, bip = arb_hypgeom_airy(z)
    return (
        di.round_interval_outward(ai, prec_bits),
        di.round_interval_outward(aip, prec_bits),
        di.round_interval_outward(bi, prec_bits),
        di.round_interval_outward(bip, prec_bits),
    )


def arb_hypgeom_expint_prec(s: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_expint(s, z), prec_bits)


def arb_hypgeom_gamma_lower_prec(
    s: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_gamma_lower(s, z, regularized=regularized), prec_bits)


def arb_hypgeom_gamma_upper_prec(
    s: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_gamma_upper(s, z, regularized=regularized), prec_bits)


def arb_hypgeom_beta_lower_prec(
    a: jax.Array, b: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_beta_lower(a, b, z, regularized=regularized), prec_bits)


def arb_hypgeom_chebyshev_t_prec(n: int, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_chebyshev_t(n, z), prec_bits)


def arb_hypgeom_chebyshev_u_prec(n: int, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_chebyshev_u(n, z), prec_bits)


def arb_hypgeom_laguerre_l_prec(
    n: int, m: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_laguerre_l(n, m, z), prec_bits)


def arb_hypgeom_hermite_h_prec(n: int, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_hermite_h(n, z), prec_bits)


def arb_hypgeom_legendre_p_prec(
    n: int, m: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, type: int = 0
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_legendre_p(n, m, z, type=type), prec_bits)


def arb_hypgeom_legendre_q_prec(
    n: int, m: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, type: int = 0
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_legendre_q(n, m, z, type=type), prec_bits)


def arb_hypgeom_jacobi_p_prec(
    n: int, a: jax.Array, b: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_jacobi_p(n, a, b, z), prec_bits)


def arb_hypgeom_gegenbauer_c_prec(
    n: int, m: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_gegenbauer_c(n, m, z), prec_bits)


def arb_hypgeom_central_bin_ui_prec(n: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_central_bin_ui(n), prec_bits)


@partial(jax.jit, static_argnames=("n",))
def acb_hypgeom_rising_ui_forward(x: jax.Array, n: int) -> jax.Array:
    x = as_acb_box(x)
    one = acb_box(di.interval(1.0, 1.0), di.interval(0.0, 0.0))

    def body(k: int, res: jax.Array) -> jax.Array:
        t = acb_box_add_ui(x, k)
        return acb_box_mul(res, t)

    return lax.fori_loop(0, n, body, one)


@partial(jax.jit, static_argnames=("n", "prec_bits"))
def acb_hypgeom_rising_ui_forward_prec(x: jax.Array, n: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = acb_box_round_prec(as_acb_box(x), prec_bits)
    one = acb_box(
        di.round_interval_outward(di.interval(1.0, 1.0), prec_bits),
        di.round_interval_outward(di.interval(0.0, 0.0), prec_bits),
    )

    def body(k: int, res: jax.Array) -> jax.Array:
        t = acb_box_add_ui_prec(x, k, prec_bits)
        return acb_box_mul_prec(res, t, prec_bits)

    return lax.fori_loop(0, n, body, one)


@partial(jax.jit, static_argnames=("n",))
def acb_hypgeom_rising_ui(x: jax.Array, n: int) -> jax.Array:
    return acb_hypgeom_rising_ui_forward(x, n)


@partial(jax.jit, static_argnames=("n", "prec_bits"))
def acb_hypgeom_rising_ui_prec(x: jax.Array, n: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_hypgeom_rising_ui_forward_prec(x, n, prec_bits)


@jax.jit
def acb_hypgeom_lgamma(x: jax.Array) -> jax.Array:
    x = as_acb_box(x)
    re = acb_real(x)
    im = acb_imag(x)

    re_lo, re_hi = re[0], re[1]
    im_lo, im_hi = im[0], im[1]

    cross_pole = (im_lo <= 0.0) & (im_hi >= 0.0) & _contains_nonpositive_integer(re_lo, re_hi)

    corners = jnp.asarray(
        [
            re_lo + 1j * im_lo,
            re_lo + 1j * im_hi,
            re_hi + 1j * im_lo,
            re_hi + 1j * im_hi,
            0.5 * (re_lo + re_hi) + 1j * (0.5 * (im_lo + im_hi)),
        ],
        dtype=jnp.complex128,
    )
    vals = jax.vmap(_complex_loggamma)(corners)
    re_vals = jnp.real(vals)
    im_vals = jnp.imag(vals)
    finite = jnp.all(jnp.isfinite(re_vals)) & jnp.all(jnp.isfinite(im_vals))

    out = acb_box(
        di.interval(di._below(jnp.min(re_vals)), di._above(jnp.max(re_vals))),
        di.interval(di._below(jnp.min(im_vals)), di._above(jnp.max(im_vals))),
    )
    full = acb_box(di.interval(-jnp.inf, jnp.inf), di.interval(-jnp.inf, jnp.inf))
    return jnp.where(cross_pole | (~finite), full, out)


@jax.jit
def acb_hypgeom_gamma(x: jax.Array) -> jax.Array:
    x = as_acb_box(x)
    re = acb_real(x)
    im = acb_imag(x)

    re_lo, re_hi = re[0], re[1]
    im_lo, im_hi = im[0], im[1]

    cross_pole = (im_lo <= 0.0) & (im_hi >= 0.0) & _contains_nonpositive_integer(re_lo, re_hi)

    corners = jnp.asarray(
        [
            re_lo + 1j * im_lo,
            re_lo + 1j * im_hi,
            re_hi + 1j * im_lo,
            re_hi + 1j * im_hi,
            0.5 * (re_lo + re_hi) + 1j * (0.5 * (im_lo + im_hi)),
        ],
        dtype=jnp.complex128,
    )
    vals = jax.vmap(lambda z: jnp.exp(_complex_loggamma(z)))(corners)
    re_vals = jnp.real(vals)
    im_vals = jnp.imag(vals)
    finite = jnp.all(jnp.isfinite(re_vals)) & jnp.all(jnp.isfinite(im_vals))

    out = acb_box(
        di.interval(di._below(jnp.min(re_vals)), di._above(jnp.max(re_vals))),
        di.interval(di._below(jnp.min(im_vals)), di._above(jnp.max(im_vals))),
    )
    full = acb_box(di.interval(-jnp.inf, jnp.inf), di.interval(-jnp.inf, jnp.inf))
    return jnp.where(cross_pole | (~finite), full, out)


@jax.jit
def acb_hypgeom_rgamma(x: jax.Array) -> jax.Array:
    return acb_box_inv(acb_hypgeom_gamma(x))


@jax.jit
def acb_hypgeom_erf(x: jax.Array) -> jax.Array:
    x = as_acb_box(x)
    re = acb_real(x)
    im = acb_imag(x)

    re_lo, re_hi = re[0], re[1]
    im_lo, im_hi = im[0], im[1]
    corners = jnp.asarray(
        [
            re_lo + 1j * im_lo,
            re_lo + 1j * im_hi,
            re_hi + 1j * im_lo,
            re_hi + 1j * im_hi,
            0.5 * (re_lo + re_hi) + 1j * (0.5 * (im_lo + im_hi)),
        ],
        dtype=jnp.complex128,
    )
    vals = jax.vmap(_complex_erf_series)(corners)
    return acb_box(
        di.interval(di._below(jnp.min(jnp.real(vals))), di._above(jnp.max(jnp.real(vals)))),
        di.interval(di._below(jnp.min(jnp.imag(vals))), di._above(jnp.max(jnp.imag(vals)))),
    )


@jax.jit
def acb_hypgeom_erfc(x: jax.Array) -> jax.Array:
    x = as_acb_box(x)
    re = acb_real(x)
    im = acb_imag(x)

    re_lo, re_hi = re[0], re[1]
    im_lo, im_hi = im[0], im[1]
    corners = jnp.asarray(
        [
            re_lo + 1j * im_lo,
            re_lo + 1j * im_hi,
            re_hi + 1j * im_lo,
            re_hi + 1j * im_hi,
            0.5 * (re_lo + re_hi) + 1j * (0.5 * (im_lo + im_hi)),
        ],
        dtype=jnp.complex128,
    )
    vals = jax.vmap(_complex_erfc_series)(corners)
    return acb_box(
        di.interval(di._below(jnp.min(jnp.real(vals))), di._above(jnp.max(jnp.real(vals)))),
        di.interval(di._below(jnp.min(jnp.imag(vals))), di._above(jnp.max(jnp.imag(vals)))),
    )


@jax.jit
def acb_hypgeom_erfi(x: jax.Array) -> jax.Array:
    x = as_acb_box(x)
    re = acb_real(x)
    im = acb_imag(x)

    re_lo, re_hi = re[0], re[1]
    im_lo, im_hi = im[0], im[1]
    corners = jnp.asarray(
        [
            re_lo + 1j * im_lo,
            re_lo + 1j * im_hi,
            re_hi + 1j * im_lo,
            re_hi + 1j * im_hi,
            0.5 * (re_lo + re_hi) + 1j * (0.5 * (im_lo + im_hi)),
        ],
        dtype=jnp.complex128,
    )
    vals = jax.vmap(_complex_erfi_series)(corners)
    return acb_box(
        di.interval(di._below(jnp.min(jnp.real(vals))), di._above(jnp.max(jnp.real(vals)))),
        di.interval(di._below(jnp.min(jnp.imag(vals))), di._above(jnp.max(jnp.imag(vals)))),
    )


@partial(jax.jit, static_argnames=("regularized",))
def acb_hypgeom_0f1(a: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    a = as_acb_box(a)
    z = as_acb_box(z)
    term0 = acb_box(di.interval(1.0, 1.0), di.interval(0.0, 0.0))
    sum0 = term0

    def body(k, state):
        term, s = state
        ak = acb_box_add_ui(a, k)
        inv_k1 = di.interval(1.0 / jnp.float64(k + 1), 1.0 / jnp.float64(k + 1))
        step = acb_box_div(z, ak)
        step = acb_box_scale_real(step, inv_k1)
        term = acb_box_mul(term, step)
        return term, acb_box_add(s, term)

    _, s = lax.fori_loop(0, _HYP_TERMS - 1, body, (term0, sum0))
    if regularized:
        s = acb_box_mul(s, acb_hypgeom_rgamma(a))
    a_m = acb_midpoint(a)
    z_vals = _acb_corners(z)
    sample_vals = jax.vmap(lambda zz: _complex_hyp0f1_scalar(a_m, zz))(z_vals)
    s_samples = _acb_from_samples(sample_vals)
    if regularized:
        s_samples = acb_box_mul(s_samples, acb_hypgeom_rgamma(a))
    param_ok = _interval_is_small(acb_real(a)) & _interval_is_small(acb_imag(a))
    return jnp.where(param_ok, _select_tighter_acb(s, s_samples), s)


@partial(jax.jit, static_argnames=("regularized",))
def acb_hypgeom_1f1(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    a = as_acb_box(a)
    b = as_acb_box(b)
    z = as_acb_box(z)
    term0 = acb_box(di.interval(1.0, 1.0), di.interval(0.0, 0.0))
    sum0 = term0

    def body(k, state):
        term, s = state
        ak = acb_box_add_ui(a, k)
        bk = acb_box_add_ui(b, k)
        step = acb_box_div(ak, bk)
        inv_k1 = di.interval(1.0 / jnp.float64(k + 1), 1.0 / jnp.float64(k + 1))
        term = acb_box_mul(term, step)
        term = acb_box_mul(term, z)
        term = acb_box_scale_real(term, inv_k1)
        return term, acb_box_add(s, term)

    _, s = lax.fori_loop(0, _HYP_TERMS - 1, body, (term0, sum0))
    if regularized:
        s = acb_box_mul(s, acb_hypgeom_rgamma(b))
    a_m = acb_midpoint(a)
    b_m = acb_midpoint(b)
    z_vals = _acb_corners(z)
    sample_vals = jax.vmap(lambda zz: _complex_hyp1f1_scalar(a_m, b_m, zz))(z_vals)
    s_samples = _acb_from_samples(sample_vals)
    if regularized:
        s_samples = acb_box_mul(s_samples, acb_hypgeom_rgamma(b))
    param_ok = _interval_is_small(acb_real(a)) & _interval_is_small(acb_imag(a))
    param_ok = param_ok & _interval_is_small(acb_real(b)) & _interval_is_small(acb_imag(b))
    return jnp.where(param_ok, _select_tighter_acb(s, s_samples), s)


@partial(jax.jit, static_argnames=("regularized",))
def acb_hypgeom_1f1_integration(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    return acb_hypgeom_1f1(a, b, z, regularized=regularized)


@partial(jax.jit, static_argnames=("regularized",))
def acb_hypgeom_m(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    return acb_hypgeom_1f1(a, b, z, regularized=regularized)


@partial(jax.jit, static_argnames=("regularized",))
def acb_hypgeom_2f1(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    a = as_acb_box(a)
    b = as_acb_box(b)
    c = as_acb_box(c)
    z = as_acb_box(z)
    term0 = acb_box(di.interval(1.0, 1.0), di.interval(0.0, 0.0))
    sum0 = term0

    def body(k, state):
        term, s = state
        ak = acb_box_add_ui(a, k)
        bk = acb_box_add_ui(b, k)
        ck = acb_box_add_ui(c, k)
        k1 = di.interval(jnp.float64(k + 1), jnp.float64(k + 1))
        num = acb_box_mul(ak, bk)
        den = acb_box_scale_real(ck, k1)
        step = acb_box_div(num, den)
        term = acb_box_mul(term, step)
        term = acb_box_mul(term, z)
        return term, acb_box_add(s, term)

    _, s = lax.fori_loop(0, _HYP_TERMS - 1, body, (term0, sum0))
    if regularized:
        s = acb_box_mul(s, acb_hypgeom_rgamma(c))
    a_m = acb_midpoint(a)
    b_m = acb_midpoint(b)
    c_m = acb_midpoint(c)
    z_vals = _acb_corners(z)
    sample_vals = jax.vmap(lambda zz: _complex_hyp2f1_scalar(a_m, b_m, c_m, zz))(z_vals)
    s_samples = _acb_from_samples(sample_vals)
    if regularized:
        s_samples = acb_box_mul(s_samples, acb_hypgeom_rgamma(c))
    param_ok = _interval_is_small(acb_real(a)) & _interval_is_small(acb_imag(a))
    param_ok = param_ok & _interval_is_small(acb_real(b)) & _interval_is_small(acb_imag(b))
    param_ok = param_ok & _interval_is_small(acb_real(c)) & _interval_is_small(acb_imag(c))
    return jnp.where(param_ok, _select_tighter_acb(s, s_samples), s)


@partial(jax.jit, static_argnames=("regularized",))
def acb_hypgeom_2f1_integration(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    return acb_hypgeom_2f1(a, b, c, z, regularized=regularized)


def _acb_from_complex(z: jax.Array) -> jax.Array:
    return acb_box(
        di.interval(di._below(jnp.real(z)), di._above(jnp.real(z))),
        di.interval(di._below(jnp.imag(z)), di._above(jnp.imag(z))),
    )


def _acb_corners(x: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    re_lo, re_hi = xb[..., 0], xb[..., 1]
    im_lo, im_hi = xb[..., 2], xb[..., 3]
    return jnp.asarray(
        [
            re_lo + 1j * im_lo,
            re_lo + 1j * im_hi,
            re_hi + 1j * im_lo,
            re_hi + 1j * im_hi,
            0.5 * (re_lo + re_hi) + 1j * 0.5 * (im_lo + im_hi),
        ],
        dtype=jnp.complex128,
    )


def _acb_from_samples(vals: jax.Array) -> jax.Array:
    return acb_box(
        di.interval(di._below(jnp.min(jnp.real(vals))), di._above(jnp.max(jnp.real(vals)))),
        di.interval(di._below(jnp.min(jnp.imag(vals))), di._above(jnp.max(jnp.imag(vals)))),
    )


@jax.jit
def acb_hypgeom_u(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    a_m = acb_midpoint(a)
    b_m = acb_midpoint(b)
    z_m = acb_midpoint(z)
    mid = _acb_from_complex(_complex_hypu_scalar(a_m, b_m, z_m))
    z_vals = _acb_corners(z)
    sample_vals = jax.vmap(lambda zz: _complex_hypu_scalar(a_m, b_m, zz))(z_vals)
    s_samples = _acb_from_samples(sample_vals)
    param_ok = _interval_is_small(acb_real(a)) & _interval_is_small(acb_imag(a))
    param_ok = param_ok & _interval_is_small(acb_real(b)) & _interval_is_small(acb_imag(b))
    return jnp.where(param_ok, _select_tighter_acb(mid, s_samples), s_samples)


@jax.jit
def acb_hypgeom_u_integration(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    return acb_hypgeom_u(a, b, z)


@jax.jit
def acb_hypgeom_bessel_j_0f1(nu: jax.Array, z: jax.Array) -> jax.Array:
    nu_m = acb_midpoint(nu)
    z_m = acb_midpoint(z)
    return _acb_from_complex(_complex_bessel_series(nu_m, z_m, -1.0))


@jax.jit
def acb_hypgeom_bessel_j_asymp(nu: jax.Array, z: jax.Array) -> jax.Array:
    return acb_hypgeom_bessel_j_0f1(nu, z)


@jax.jit
def acb_hypgeom_bessel_j(nu: jax.Array, z: jax.Array) -> jax.Array:
    return acb_hypgeom_bessel_j_0f1(nu, z)


@partial(jax.jit, static_argnames=("scaled",))
def acb_hypgeom_bessel_i_0f1(nu: jax.Array, z: jax.Array, scaled: bool = False) -> jax.Array:
    nu_m = acb_midpoint(nu)
    z_m = acb_midpoint(z)
    val = _complex_bessel_series(nu_m, z_m, 1.0)
    if scaled:
        val = val * jnp.exp(-z_m)
    return _acb_from_complex(val)


@partial(jax.jit, static_argnames=("scaled",))
def acb_hypgeom_bessel_i_asymp(nu: jax.Array, z: jax.Array, scaled: bool = False) -> jax.Array:
    return acb_hypgeom_bessel_i_0f1(nu, z, scaled=scaled)


@jax.jit
def acb_hypgeom_bessel_i(nu: jax.Array, z: jax.Array) -> jax.Array:
    return acb_hypgeom_bessel_i_0f1(nu, z, scaled=False)


@jax.jit
def acb_hypgeom_bessel_i_scaled(nu: jax.Array, z: jax.Array) -> jax.Array:
    return acb_hypgeom_bessel_i_0f1(nu, z, scaled=True)


@partial(jax.jit, static_argnames=("scaled",))
def acb_hypgeom_bessel_k_0f1(nu: jax.Array, z: jax.Array, scaled: bool = False) -> jax.Array:
    nu_m = acb_midpoint(nu)
    z_m = acb_midpoint(z)
    val = _complex_bessel_k(nu_m, z_m)
    if scaled:
        val = val * jnp.exp(z_m)
    return _acb_from_complex(val)


@partial(jax.jit, static_argnames=("scaled",))
def acb_hypgeom_bessel_k_asymp(nu: jax.Array, z: jax.Array, scaled: bool = False) -> jax.Array:
    return acb_hypgeom_bessel_k_0f1(nu, z, scaled=scaled)


@jax.jit
def acb_hypgeom_bessel_k(nu: jax.Array, z: jax.Array) -> jax.Array:
    return acb_hypgeom_bessel_k_0f1(nu, z, scaled=False)


@jax.jit
def acb_hypgeom_bessel_k_scaled(nu: jax.Array, z: jax.Array) -> jax.Array:
    return acb_hypgeom_bessel_k_0f1(nu, z, scaled=True)


@jax.jit
def acb_hypgeom_bessel_y(nu: jax.Array, z: jax.Array) -> jax.Array:
    nu_m = acb_midpoint(nu)
    z_m = acb_midpoint(z)
    return _acb_from_complex(_complex_bessel_y(nu_m, z_m))


@jax.jit
def acb_hypgeom_bessel_jy(nu: jax.Array, z: jax.Array) -> tuple[jax.Array, jax.Array]:
    return acb_hypgeom_bessel_j(nu, z), acb_hypgeom_bessel_y(nu, z)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_hypgeom_bessel_j_prec(nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_bessel_j(nu, z), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_hypgeom_bessel_y_prec(nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_bessel_y(nu, z), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_hypgeom_bessel_i_prec(nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_bessel_i(nu, z), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_hypgeom_bessel_k_prec(nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_bessel_k(nu, z), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_hypgeom_bessel_i_scaled_prec(nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_bessel_i_scaled(nu, z), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_hypgeom_bessel_k_scaled_prec(nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_bessel_k_scaled(nu, z), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_hypgeom_lgamma_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_lgamma(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_hypgeom_gamma_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_gamma(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_hypgeom_rgamma_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_rgamma(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_hypgeom_erf_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_erf(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_hypgeom_erfc_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_erfc(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_hypgeom_erfi_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_erfi(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "regularized"))
def acb_hypgeom_0f1_prec(a: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_0f1(a, z, regularized=regularized), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "regularized"))
def acb_hypgeom_1f1_prec(a: jax.Array, b: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_1f1(a, b, z, regularized=regularized), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "regularized"))
def acb_hypgeom_m_prec(a: jax.Array, b: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_m(a, b, z, regularized=regularized), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits", "regularized"))
def acb_hypgeom_2f1_prec(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_2f1(a, b, c, z, regularized=regularized), prec_bits)


def arb_hypgeom_rising_ui_batch(x: jax.Array, n: int) -> jax.Array:
    x = di.as_interval(x)
    return jax.vmap(lambda xi: arb_hypgeom_rising_ui(xi, n))(x)


def arb_hypgeom_rising_ui_batch_prec(x: jax.Array, n: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = di.as_interval(x)
    return jax.vmap(lambda xi: arb_hypgeom_rising_ui_prec(xi, n, prec_bits))(x)


def acb_hypgeom_rising_ui_batch(x: jax.Array, n: int) -> jax.Array:
    x = as_acb_box(x)
    return jax.vmap(lambda xi: acb_hypgeom_rising_ui(xi, n))(x)


def acb_hypgeom_rising_ui_batch_prec(x: jax.Array, n: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = as_acb_box(x)
    return jax.vmap(lambda xi: acb_hypgeom_rising_ui_prec(xi, n, prec_bits))(x)


def arb_hypgeom_lgamma_batch(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    return jax.vmap(arb_hypgeom_lgamma)(x)


def arb_hypgeom_gamma_batch(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    return jax.vmap(arb_hypgeom_gamma)(x)


def arb_hypgeom_rgamma_batch(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    return jax.vmap(arb_hypgeom_rgamma)(x)


def arb_hypgeom_erf_batch(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    return jax.vmap(arb_hypgeom_erf)(x)


def arb_hypgeom_erfc_batch(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    return jax.vmap(arb_hypgeom_erfc)(x)


def arb_hypgeom_erfi_batch(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    return jax.vmap(arb_hypgeom_erfi)(x)


def arb_hypgeom_erfinv_batch(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    return jax.vmap(arb_hypgeom_erfinv)(x)


def arb_hypgeom_erfcinv_batch(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    return jax.vmap(arb_hypgeom_erfcinv)(x)

def arb_hypgeom_bessel_j_batch(nu: jax.Array, z: jax.Array, mode: str = "sample") -> jax.Array:
    nu = di.as_interval(nu)
    z = di.as_interval(z)
    return jax.vmap(lambda a, b: arb_hypgeom_bessel_j(a, b, mode=mode))(nu, z)


def arb_hypgeom_bessel_y_batch(nu: jax.Array, z: jax.Array, mode: str = "sample") -> jax.Array:
    nu = di.as_interval(nu)
    z = di.as_interval(z)
    return jax.vmap(lambda a, b: arb_hypgeom_bessel_y(a, b, mode=mode))(nu, z)


def arb_hypgeom_bessel_jy_batch(nu: jax.Array, z: jax.Array, mode: str = "sample") -> tuple[jax.Array, jax.Array]:
    nu = di.as_interval(nu)
    z = di.as_interval(z)
    j = jax.vmap(lambda a, b: arb_hypgeom_bessel_j(a, b, mode=mode))(nu, z)
    y = jax.vmap(lambda a, b: arb_hypgeom_bessel_y(a, b, mode=mode))(nu, z)
    return j, y


def arb_hypgeom_bessel_jy_batch_prec(
    nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, mode: str = "sample"
) -> tuple[jax.Array, jax.Array]:
    j = arb_hypgeom_bessel_j_batch_prec(nu, z, prec_bits=prec_bits, mode=mode)
    y = arb_hypgeom_bessel_y_batch_prec(nu, z, prec_bits=prec_bits, mode=mode)
    return j, y


def arb_hypgeom_bessel_i_batch(nu: jax.Array, z: jax.Array, mode: str = "sample") -> jax.Array:
    nu = di.as_interval(nu)
    z = di.as_interval(z)
    return jax.vmap(lambda a, b: arb_hypgeom_bessel_i(a, b, mode=mode))(nu, z)


def arb_hypgeom_bessel_k_batch(nu: jax.Array, z: jax.Array, mode: str = "sample") -> jax.Array:
    nu = di.as_interval(nu)
    z = di.as_interval(z)
    return jax.vmap(lambda a, b: arb_hypgeom_bessel_k(a, b, mode=mode))(nu, z)


def arb_hypgeom_bessel_i_scaled_batch(nu: jax.Array, z: jax.Array, mode: str = "sample") -> jax.Array:
    nu = di.as_interval(nu)
    z = di.as_interval(z)
    return jax.vmap(lambda a, b: arb_hypgeom_bessel_i_scaled(a, b, mode=mode))(nu, z)


def arb_hypgeom_bessel_k_scaled_batch(nu: jax.Array, z: jax.Array, mode: str = "sample") -> jax.Array:
    nu = di.as_interval(nu)
    z = di.as_interval(z)
    return jax.vmap(lambda a, b: arb_hypgeom_bessel_k_scaled(a, b, mode=mode))(nu, z)


def arb_hypgeom_bessel_i_integration_batch(
    nu: jax.Array, z: jax.Array, scaled: bool = False, mode: str = "sample"
) -> jax.Array:
    nu = di.as_interval(nu)
    z = di.as_interval(z)
    return jax.vmap(lambda a, b: arb_hypgeom_bessel_i_integration(a, b, scaled=scaled, mode=mode))(nu, z)


def arb_hypgeom_bessel_k_integration_batch(
    nu: jax.Array, z: jax.Array, scaled: bool = False, mode: str = "sample"
) -> jax.Array:
    nu = di.as_interval(nu)
    z = di.as_interval(z)
    return jax.vmap(lambda a, b: arb_hypgeom_bessel_k_integration(a, b, scaled=scaled, mode=mode))(nu, z)


def arb_hypgeom_0f1_batch(a: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    a = di.as_interval(a)
    z = di.as_interval(z)
    term0 = _ones_interval_like(a)
    sum0 = term0

    def body(k, state):
        term, s = state
        kf = jnp.float64(k)
        ak = di.fast_add(a, di.interval(kf, kf))
        inv_k1 = di.interval(1.0 / jnp.float64(k + 1), 1.0 / jnp.float64(k + 1))
        step = di.fast_div(z, ak)
        step = di.fast_mul(step, inv_k1)
        term = di.fast_mul(term, step)
        return term, di.fast_add(s, term)

    _, s = lax.fori_loop(0, _HYP_TERMS - 1, body, (term0, sum0))
    if regularized:
        s = di.fast_mul(s, arb_hypgeom_rgamma_batch(a))
    return s


def arb_hypgeom_1f1_batch(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    z = di.as_interval(z)
    term0 = _ones_interval_like(a)
    sum0 = term0

    def body(k, state):
        term, s = state
        kf = jnp.float64(k)
        ak = di.fast_add(a, di.interval(kf, kf))
        bk = di.fast_add(b, di.interval(kf, kf))
        step = di.fast_div(ak, bk)
        inv_k1 = di.interval(1.0 / jnp.float64(k + 1), 1.0 / jnp.float64(k + 1))
        term = di.fast_mul(term, step)
        term = di.fast_mul(term, z)
        term = di.fast_mul(term, inv_k1)
        return term, di.fast_add(s, term)

    _, s = lax.fori_loop(0, _HYP_TERMS - 1, body, (term0, sum0))
    if regularized:
        s = di.fast_mul(s, arb_hypgeom_rgamma_batch(b))
    return s


def arb_hypgeom_1f1_integration_batch(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_1f1_batch(a, b, z, regularized=regularized)


def arb_hypgeom_m_batch(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_1f1_batch(a, b, z, regularized=regularized)


def arb_hypgeom_2f1_batch(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    c = di.as_interval(c)
    z = di.as_interval(z)
    term0 = _ones_interval_like(a)
    sum0 = term0

    def body(k, state):
        term, s = state
        kf = jnp.float64(k)
        ak = di.fast_add(a, di.interval(kf, kf))
        bk = di.fast_add(b, di.interval(kf, kf))
        ck = di.fast_add(c, di.interval(kf, kf))
        k1 = di.interval(jnp.float64(k + 1), jnp.float64(k + 1))
        num = di.fast_mul(ak, bk)
        den = di.fast_mul(ck, k1)
        step = di.fast_div(num, den)
        term = di.fast_mul(term, step)
        term = di.fast_mul(term, z)
        return term, di.fast_add(s, term)

    _, s = lax.fori_loop(0, _HYP_TERMS - 1, body, (term0, sum0))
    if regularized:
        s = di.fast_mul(s, arb_hypgeom_rgamma_batch(c))
    return s


def arb_hypgeom_2f1_integration_batch(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    return arb_hypgeom_2f1_batch(a, b, c, z, regularized=regularized)


def arb_hypgeom_u_batch(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    z = di.as_interval(z)
    return jax.vmap(arb_hypgeom_u)(a, b, z)


def arb_hypgeom_u_integration_batch(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    return arb_hypgeom_u_batch(a, b, z)


def arb_hypgeom_fresnel_batch(z: jax.Array, normalized: bool = False) -> tuple[jax.Array, jax.Array]:
    z = di.as_interval(z)
    return jax.vmap(lambda zi: arb_hypgeom_fresnel(zi, normalized=normalized))(z)


def arb_hypgeom_ei_batch(z: jax.Array) -> jax.Array:
    z = di.as_interval(z)
    return jax.vmap(arb_hypgeom_ei)(z)


def arb_hypgeom_si_batch(z: jax.Array) -> jax.Array:
    z = di.as_interval(z)
    return jax.vmap(arb_hypgeom_si)(z)


def arb_hypgeom_ci_batch(z: jax.Array) -> jax.Array:
    z = di.as_interval(z)
    return jax.vmap(arb_hypgeom_ci)(z)


def arb_hypgeom_shi_batch(z: jax.Array) -> jax.Array:
    z = di.as_interval(z)
    return jax.vmap(arb_hypgeom_shi)(z)


def arb_hypgeom_chi_batch(z: jax.Array) -> jax.Array:
    z = di.as_interval(z)
    return jax.vmap(arb_hypgeom_chi)(z)


def arb_hypgeom_li_batch(z: jax.Array, offset: int = 0) -> jax.Array:
    z = di.as_interval(z)
    return jax.vmap(lambda zi: arb_hypgeom_li(zi, offset=offset))(z)


def arb_hypgeom_dilog_batch(z: jax.Array) -> jax.Array:
    z = di.as_interval(z)
    return jax.vmap(arb_hypgeom_dilog)(z)


def arb_hypgeom_airy_batch(z: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    z = di.as_interval(z)
    return jax.vmap(arb_hypgeom_airy)(z)


def arb_hypgeom_expint_batch(s: jax.Array, z: jax.Array) -> jax.Array:
    s = di.as_interval(s)
    z = di.as_interval(z)
    return jax.vmap(arb_hypgeom_expint)(s, z)


def arb_hypgeom_gamma_lower_batch(s: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    s = di.as_interval(s)
    z = di.as_interval(z)
    return jax.vmap(lambda si, zi: arb_hypgeom_gamma_lower(si, zi, regularized=regularized))(s, z)


def arb_hypgeom_gamma_upper_batch(s: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    s = di.as_interval(s)
    z = di.as_interval(z)
    return jax.vmap(lambda si, zi: arb_hypgeom_gamma_upper(si, zi, regularized=regularized))(s, z)


def arb_hypgeom_beta_lower_batch(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    z = di.as_interval(z)
    return jax.vmap(lambda ai, bi, zi: arb_hypgeom_beta_lower(ai, bi, zi, regularized=regularized))(a, b, z)


def arb_hypgeom_chebyshev_t_batch(n: int, z: jax.Array) -> jax.Array:
    z = di.as_interval(z)
    return jax.vmap(lambda zi: arb_hypgeom_chebyshev_t(n, zi))(z)


def arb_hypgeom_chebyshev_u_batch(n: int, z: jax.Array) -> jax.Array:
    z = di.as_interval(z)
    return jax.vmap(lambda zi: arb_hypgeom_chebyshev_u(n, zi))(z)


def arb_hypgeom_laguerre_l_batch(n: int, m: jax.Array, z: jax.Array) -> jax.Array:
    m = di.as_interval(m)
    z = di.as_interval(z)
    return jax.vmap(lambda mi, zi: arb_hypgeom_laguerre_l(n, mi, zi))(m, z)


def arb_hypgeom_hermite_h_batch(n: int, z: jax.Array) -> jax.Array:
    z = di.as_interval(z)
    return jax.vmap(lambda zi: arb_hypgeom_hermite_h(n, zi))(z)


def arb_hypgeom_legendre_p_batch(n: int, m: jax.Array, z: jax.Array, type: int = 0) -> jax.Array:
    m = di.as_interval(m)
    z = di.as_interval(z)
    return jax.vmap(lambda mi, zi: arb_hypgeom_legendre_p(n, mi, zi, type=type))(m, z)


def arb_hypgeom_legendre_q_batch(n: int, m: jax.Array, z: jax.Array, type: int = 0) -> jax.Array:
    m = di.as_interval(m)
    z = di.as_interval(z)
    return jax.vmap(lambda mi, zi: arb_hypgeom_legendre_q(n, mi, zi, type=type))(m, z)


def arb_hypgeom_jacobi_p_batch(n: int, a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    z = di.as_interval(z)
    return jax.vmap(lambda ai, bi, zi: arb_hypgeom_jacobi_p(n, ai, bi, zi))(a, b, z)


def arb_hypgeom_gegenbauer_c_batch(n: int, m: jax.Array, z: jax.Array) -> jax.Array:
    m = di.as_interval(m)
    z = di.as_interval(z)
    return jax.vmap(lambda mi, zi: arb_hypgeom_gegenbauer_c(n, mi, zi))(m, z)


def arb_hypgeom_central_bin_ui_batch(n: jax.Array) -> jax.Array:
    n = jnp.asarray(n, dtype=jnp.int64)
    return jax.vmap(arb_hypgeom_central_bin_ui)(n)


def arb_hypgeom_lgamma_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = di.as_interval(x)
    return jax.vmap(lambda xi: arb_hypgeom_lgamma_prec(xi, prec_bits))(x)


def arb_hypgeom_gamma_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = di.as_interval(x)
    return jax.vmap(lambda xi: arb_hypgeom_gamma_prec(xi, prec_bits))(x)


def arb_hypgeom_rgamma_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = di.as_interval(x)
    return jax.vmap(lambda xi: arb_hypgeom_rgamma_prec(xi, prec_bits))(x)


def arb_hypgeom_erf_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = di.as_interval(x)
    return jax.vmap(lambda xi: arb_hypgeom_erf_prec(xi, prec_bits))(x)


def arb_hypgeom_erfc_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = di.as_interval(x)
    return jax.vmap(lambda xi: arb_hypgeom_erfc_prec(xi, prec_bits))(x)


def arb_hypgeom_erfi_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = di.as_interval(x)
    return jax.vmap(lambda xi: arb_hypgeom_erfi_prec(xi, prec_bits))(x)


def arb_hypgeom_erfinv_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = di.as_interval(x)
    return jax.vmap(lambda xi: arb_hypgeom_erfinv_prec(xi, prec_bits))(x)


def arb_hypgeom_erfcinv_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = di.as_interval(x)
    return jax.vmap(lambda xi: arb_hypgeom_erfcinv_prec(xi, prec_bits))(x)


def arb_hypgeom_fresnel_batch_prec(
    z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, normalized: bool = False
) -> tuple[jax.Array, jax.Array]:
    s, c = arb_hypgeom_fresnel_batch(z, normalized=normalized)
    return di.round_interval_outward(s, prec_bits), di.round_interval_outward(c, prec_bits)


def arb_hypgeom_ei_batch_prec(z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_ei_batch(z), prec_bits)


def arb_hypgeom_si_batch_prec(z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_si_batch(z), prec_bits)


def arb_hypgeom_ci_batch_prec(z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_ci_batch(z), prec_bits)


def arb_hypgeom_shi_batch_prec(z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_shi_batch(z), prec_bits)


def arb_hypgeom_chi_batch_prec(z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_chi_batch(z), prec_bits)


def arb_hypgeom_li_batch_prec(z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, offset: int = 0) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_li_batch(z, offset=offset), prec_bits)


def arb_hypgeom_dilog_batch_prec(z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_dilog_batch(z), prec_bits)


def arb_hypgeom_airy_batch_prec(
    z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    ai, aip, bi, bip = arb_hypgeom_airy_batch(z)
    return (
        di.round_interval_outward(ai, prec_bits),
        di.round_interval_outward(aip, prec_bits),
        di.round_interval_outward(bi, prec_bits),
        di.round_interval_outward(bip, prec_bits),
    )


def arb_hypgeom_expint_batch_prec(s: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_expint_batch(s, z), prec_bits)


def arb_hypgeom_gamma_lower_batch_prec(
    s: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_gamma_lower_batch(s, z, regularized=regularized), prec_bits)


def arb_hypgeom_gamma_upper_batch_prec(
    s: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_gamma_upper_batch(s, z, regularized=regularized), prec_bits)


def arb_hypgeom_beta_lower_batch_prec(
    a: jax.Array, b: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_beta_lower_batch(a, b, z, regularized=regularized), prec_bits)


def arb_hypgeom_chebyshev_t_batch_prec(
    n: int, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_chebyshev_t_batch(n, z), prec_bits)


def arb_hypgeom_chebyshev_u_batch_prec(
    n: int, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_chebyshev_u_batch(n, z), prec_bits)


def arb_hypgeom_laguerre_l_batch_prec(
    n: int, m: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_laguerre_l_batch(n, m, z), prec_bits)


def arb_hypgeom_hermite_h_batch_prec(
    n: int, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_hermite_h_batch(n, z), prec_bits)


def arb_hypgeom_legendre_p_batch_prec(
    n: int, m: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, type: int = 0
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_legendre_p_batch(n, m, z, type=type), prec_bits)


def arb_hypgeom_legendre_q_batch_prec(
    n: int, m: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, type: int = 0
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_legendre_q_batch(n, m, z, type=type), prec_bits)


def arb_hypgeom_jacobi_p_batch_prec(
    n: int, a: jax.Array, b: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_jacobi_p_batch(n, a, b, z), prec_bits)


def arb_hypgeom_gegenbauer_c_batch_prec(
    n: int, m: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_gegenbauer_c_batch(n, m, z), prec_bits)


def arb_hypgeom_central_bin_ui_batch_prec(n: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_central_bin_ui_batch(n), prec_bits)

def arb_hypgeom_bessel_j_batch_prec(
    nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, mode: str = "sample"
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_bessel_j_batch(nu, z, mode=mode), prec_bits)


def arb_hypgeom_bessel_y_batch_prec(
    nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, mode: str = "sample"
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_bessel_y_batch(nu, z, mode=mode), prec_bits)


def arb_hypgeom_bessel_i_batch_prec(
    nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, mode: str = "sample"
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_bessel_i_batch(nu, z, mode=mode), prec_bits)


def arb_hypgeom_bessel_k_batch_prec(
    nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, mode: str = "sample"
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_bessel_k_batch(nu, z, mode=mode), prec_bits)


def arb_hypgeom_bessel_i_scaled_batch_prec(
    nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, mode: str = "sample"
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_bessel_i_scaled_batch(nu, z, mode=mode), prec_bits)


def arb_hypgeom_bessel_k_scaled_batch_prec(
    nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, mode: str = "sample"
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_bessel_k_scaled_batch(nu, z, mode=mode), prec_bits)


def arb_hypgeom_bessel_i_integration_batch_prec(
    nu: jax.Array,
    z: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
    scaled: bool = False,
    mode: str = "sample",
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_bessel_i_integration_batch(nu, z, scaled=scaled, mode=mode), prec_bits)


def arb_hypgeom_bessel_k_integration_batch_prec(
    nu: jax.Array,
    z: jax.Array,
    prec_bits: int = di.DEFAULT_PREC_BITS,
    scaled: bool = False,
    mode: str = "sample",
) -> jax.Array:
    return di.round_interval_outward(arb_hypgeom_bessel_k_integration_batch(nu, z, scaled=scaled, mode=mode), prec_bits)


def arb_hypgeom_0f1_batch_prec(a: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False) -> jax.Array:
    a = di.as_interval(a)
    z = di.as_interval(z)
    return di.round_interval_outward(arb_hypgeom_0f1_batch(a, z, regularized=regularized), prec_bits)


def arb_hypgeom_1f1_batch_prec(a: jax.Array, b: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    z = di.as_interval(z)
    return di.round_interval_outward(arb_hypgeom_1f1_batch(a, b, z, regularized=regularized), prec_bits)


def arb_hypgeom_m_batch_prec(a: jax.Array, b: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    z = di.as_interval(z)
    return di.round_interval_outward(arb_hypgeom_m_batch(a, b, z, regularized=regularized), prec_bits)


def arb_hypgeom_2f1_batch_prec(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    c = di.as_interval(c)
    z = di.as_interval(z)
    return di.round_interval_outward(arb_hypgeom_2f1_batch(a, b, c, z, regularized=regularized), prec_bits)


def arb_hypgeom_1f1_integration_batch_prec(
    a: jax.Array, b: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False
) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    z = di.as_interval(z)
    return di.round_interval_outward(arb_hypgeom_1f1_integration_batch(a, b, z, regularized=regularized), prec_bits)


def arb_hypgeom_2f1_integration_batch_prec(
    a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False
) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    c = di.as_interval(c)
    z = di.as_interval(z)
    return di.round_interval_outward(arb_hypgeom_2f1_integration_batch(a, b, c, z, regularized=regularized), prec_bits)


def arb_hypgeom_u_batch_prec(a: jax.Array, b: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    a = di.as_interval(a)
    b = di.as_interval(b)
    z = di.as_interval(z)
    return di.round_interval_outward(arb_hypgeom_u_batch(a, b, z), prec_bits)


def arb_hypgeom_u_integration_batch_prec(a: jax.Array, b: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return arb_hypgeom_u_batch_prec(a, b, z, prec_bits)


def acb_hypgeom_lgamma_batch(x: jax.Array) -> jax.Array:
    x = as_acb_box(x)
    return jax.vmap(acb_hypgeom_lgamma)(x)


def acb_hypgeom_gamma_batch(x: jax.Array) -> jax.Array:
    x = as_acb_box(x)
    return jax.vmap(acb_hypgeom_gamma)(x)


def acb_hypgeom_rgamma_batch(x: jax.Array) -> jax.Array:
    x = as_acb_box(x)
    return jax.vmap(acb_hypgeom_rgamma)(x)


def acb_hypgeom_erf_batch(x: jax.Array) -> jax.Array:
    x = as_acb_box(x)
    return jax.vmap(acb_hypgeom_erf)(x)


def acb_hypgeom_erfc_batch(x: jax.Array) -> jax.Array:
    x = as_acb_box(x)
    return jax.vmap(acb_hypgeom_erfc)(x)


def acb_hypgeom_erfi_batch(x: jax.Array) -> jax.Array:
    x = as_acb_box(x)
    return jax.vmap(acb_hypgeom_erfi)(x)


def acb_hypgeom_1f1_integration_batch(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    return acb_hypgeom_1f1_batch(a, b, z, regularized=regularized)


def acb_hypgeom_2f1_integration_batch(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    return acb_hypgeom_2f1_batch(a, b, c, z, regularized=regularized)


def acb_hypgeom_u_batch(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    a = as_acb_box(a)
    b = as_acb_box(b)
    z = as_acb_box(z)
    return jax.vmap(acb_hypgeom_u)(a, b, z)


def acb_hypgeom_u_integration_batch(a: jax.Array, b: jax.Array, z: jax.Array) -> jax.Array:
    return acb_hypgeom_u_batch(a, b, z)

def acb_hypgeom_bessel_j_batch(nu: jax.Array, z: jax.Array) -> jax.Array:
    nu = as_acb_box(nu)
    z = as_acb_box(z)
    return jax.vmap(acb_hypgeom_bessel_j)(nu, z)


def acb_hypgeom_bessel_y_batch(nu: jax.Array, z: jax.Array) -> jax.Array:
    nu = as_acb_box(nu)
    z = as_acb_box(z)
    return jax.vmap(acb_hypgeom_bessel_y)(nu, z)


def acb_hypgeom_bessel_jy_batch(nu: jax.Array, z: jax.Array) -> tuple[jax.Array, jax.Array]:
    nu = as_acb_box(nu)
    z = as_acb_box(z)
    j = jax.vmap(acb_hypgeom_bessel_j)(nu, z)
    y = jax.vmap(acb_hypgeom_bessel_y)(nu, z)
    return j, y


def acb_hypgeom_bessel_jy_batch_prec(
    nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS
) -> tuple[jax.Array, jax.Array]:
    j = acb_hypgeom_bessel_j_batch_prec(nu, z, prec_bits=prec_bits)
    y = acb_hypgeom_bessel_y_batch_prec(nu, z, prec_bits=prec_bits)
    return j, y


def acb_hypgeom_bessel_i_batch(nu: jax.Array, z: jax.Array) -> jax.Array:
    nu = as_acb_box(nu)
    z = as_acb_box(z)
    return jax.vmap(acb_hypgeom_bessel_i)(nu, z)


def acb_hypgeom_bessel_k_batch(nu: jax.Array, z: jax.Array) -> jax.Array:
    nu = as_acb_box(nu)
    z = as_acb_box(z)
    return jax.vmap(acb_hypgeom_bessel_k)(nu, z)


def acb_hypgeom_bessel_i_scaled_batch(nu: jax.Array, z: jax.Array) -> jax.Array:
    nu = as_acb_box(nu)
    z = as_acb_box(z)
    return jax.vmap(acb_hypgeom_bessel_i_scaled)(nu, z)


def acb_hypgeom_bessel_k_scaled_batch(nu: jax.Array, z: jax.Array) -> jax.Array:
    nu = as_acb_box(nu)
    z = as_acb_box(z)
    return jax.vmap(acb_hypgeom_bessel_k_scaled)(nu, z)


def acb_hypgeom_0f1_batch(a: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    a = as_acb_box(a)
    z = as_acb_box(z)
    one = acb_box(_ones_interval_like(acb_real(a)), _zeros_interval_like(acb_real(a)))
    term0 = one
    sum0 = term0

    def body(k, state):
        term, s = state
        ak = acb_box_add_ui(a, k)
        inv_k1 = di.interval(1.0 / jnp.float64(k + 1), 1.0 / jnp.float64(k + 1))
        step = acb_box_div(z, ak)
        step = acb_box_scale_real(step, inv_k1)
        term = acb_box_mul(term, step)
        return term, acb_box_add(s, term)

    _, s = lax.fori_loop(0, _HYP_TERMS - 1, body, (term0, sum0))
    if regularized:
        s = acb_box_mul(s, acb_hypgeom_rgamma_batch(a))
    return s


def acb_hypgeom_1f1_batch(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    a = as_acb_box(a)
    b = as_acb_box(b)
    z = as_acb_box(z)
    one = acb_box(_ones_interval_like(acb_real(a)), _zeros_interval_like(acb_real(a)))
    term0 = one
    sum0 = term0

    def body(k, state):
        term, s = state
        ak = acb_box_add_ui(a, k)
        bk = acb_box_add_ui(b, k)
        step = acb_box_div(ak, bk)
        inv_k1 = di.interval(1.0 / jnp.float64(k + 1), 1.0 / jnp.float64(k + 1))
        term = acb_box_mul(term, step)
        term = acb_box_mul(term, z)
        term = acb_box_scale_real(term, inv_k1)
        return term, acb_box_add(s, term)

    _, s = lax.fori_loop(0, _HYP_TERMS - 1, body, (term0, sum0))
    if regularized:
        s = acb_box_mul(s, acb_hypgeom_rgamma_batch(b))
    return s


def acb_hypgeom_m_batch(a: jax.Array, b: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    return acb_hypgeom_1f1_batch(a, b, z, regularized=regularized)


def acb_hypgeom_2f1_batch(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, regularized: bool = False) -> jax.Array:
    a = as_acb_box(a)
    b = as_acb_box(b)
    c = as_acb_box(c)
    z = as_acb_box(z)
    one = acb_box(_ones_interval_like(acb_real(a)), _zeros_interval_like(acb_real(a)))
    term0 = one
    sum0 = term0

    def body(k, state):
        term, s = state
        ak = acb_box_add_ui(a, k)
        bk = acb_box_add_ui(b, k)
        ck = acb_box_add_ui(c, k)
        k1 = di.interval(jnp.float64(k + 1), jnp.float64(k + 1))
        num = acb_box_mul(ak, bk)
        den = acb_box_scale_real(ck, k1)
        step = acb_box_div(num, den)
        term = acb_box_mul(term, step)
        term = acb_box_mul(term, z)
        return term, acb_box_add(s, term)

    _, s = lax.fori_loop(0, _HYP_TERMS - 1, body, (term0, sum0))
    if regularized:
        s = acb_box_mul(s, acb_hypgeom_rgamma_batch(c))
    return s


def acb_hypgeom_lgamma_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = as_acb_box(x)
    return jax.vmap(lambda xi: acb_hypgeom_lgamma_prec(xi, prec_bits))(x)


def acb_hypgeom_gamma_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = as_acb_box(x)
    return jax.vmap(lambda xi: acb_hypgeom_gamma_prec(xi, prec_bits))(x)


def acb_hypgeom_rgamma_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = as_acb_box(x)
    return jax.vmap(lambda xi: acb_hypgeom_rgamma_prec(xi, prec_bits))(x)


def acb_hypgeom_erf_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = as_acb_box(x)
    return jax.vmap(lambda xi: acb_hypgeom_erf_prec(xi, prec_bits))(x)


def acb_hypgeom_erfc_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = as_acb_box(x)
    return jax.vmap(lambda xi: acb_hypgeom_erfc_prec(xi, prec_bits))(x)


def acb_hypgeom_erfi_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = as_acb_box(x)
    return jax.vmap(lambda xi: acb_hypgeom_erfi_prec(xi, prec_bits))(x)


def acb_hypgeom_u_batch_prec(a: jax.Array, b: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    a = as_acb_box(a)
    b = as_acb_box(b)
    z = as_acb_box(z)
    return acb_box_round_prec(acb_hypgeom_u_batch(a, b, z), prec_bits)


def acb_hypgeom_u_integration_batch_prec(
    a: jax.Array, b: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS
) -> jax.Array:
    return acb_hypgeom_u_batch_prec(a, b, z, prec_bits)


def acb_hypgeom_1f1_integration_batch_prec(
    a: jax.Array, b: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False
) -> jax.Array:
    a = as_acb_box(a)
    b = as_acb_box(b)
    z = as_acb_box(z)
    return acb_box_round_prec(acb_hypgeom_1f1_integration_batch(a, b, z, regularized=regularized), prec_bits)


def acb_hypgeom_2f1_integration_batch_prec(
    a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False
) -> jax.Array:
    a = as_acb_box(a)
    b = as_acb_box(b)
    c = as_acb_box(c)
    z = as_acb_box(z)
    return acb_box_round_prec(acb_hypgeom_2f1_integration_batch(a, b, c, z, regularized=regularized), prec_bits)

def acb_hypgeom_bessel_j_batch_prec(nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_bessel_j_batch(nu, z), prec_bits)


def acb_hypgeom_bessel_y_batch_prec(nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_bessel_y_batch(nu, z), prec_bits)


def acb_hypgeom_bessel_i_batch_prec(nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_bessel_i_batch(nu, z), prec_bits)


def acb_hypgeom_bessel_k_batch_prec(nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_bessel_k_batch(nu, z), prec_bits)


def acb_hypgeom_bessel_i_scaled_batch_prec(nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_bessel_i_scaled_batch(nu, z), prec_bits)


def acb_hypgeom_bessel_k_scaled_batch_prec(nu: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_hypgeom_bessel_k_scaled_batch(nu, z), prec_bits)

def acb_hypgeom_0f1_batch_prec(a: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False) -> jax.Array:
    a = as_acb_box(a)
    z = as_acb_box(z)
    return acb_box_round_prec(acb_hypgeom_0f1_batch(a, z, regularized=regularized), prec_bits)


def acb_hypgeom_1f1_batch_prec(a: jax.Array, b: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False) -> jax.Array:
    a = as_acb_box(a)
    b = as_acb_box(b)
    z = as_acb_box(z)
    return acb_box_round_prec(acb_hypgeom_1f1_batch(a, b, z, regularized=regularized), prec_bits)


def acb_hypgeom_m_batch_prec(a: jax.Array, b: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False) -> jax.Array:
    a = as_acb_box(a)
    b = as_acb_box(b)
    z = as_acb_box(z)
    return acb_box_round_prec(acb_hypgeom_m_batch(a, b, z, regularized=regularized), prec_bits)


def acb_hypgeom_2f1_batch_prec(a: jax.Array, b: jax.Array, c: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS, regularized: bool = False) -> jax.Array:
    a = as_acb_box(a)
    b = as_acb_box(b)
    c = as_acb_box(c)
    z = as_acb_box(z)
    return acb_box_round_prec(acb_hypgeom_2f1_batch(a, b, c, z, regularized=regularized), prec_bits)


arb_hypgeom_rising_ui_batch_jit = jax.jit(arb_hypgeom_rising_ui_batch, static_argnames=("n",))
acb_hypgeom_rising_ui_batch_jit = jax.jit(acb_hypgeom_rising_ui_batch, static_argnames=("n",))
arb_hypgeom_lgamma_batch_jit = jax.jit(arb_hypgeom_lgamma_batch)
acb_hypgeom_lgamma_batch_jit = jax.jit(acb_hypgeom_lgamma_batch)
arb_hypgeom_gamma_batch_jit = jax.jit(arb_hypgeom_gamma_batch)
acb_hypgeom_gamma_batch_jit = jax.jit(acb_hypgeom_gamma_batch)
arb_hypgeom_rgamma_batch_jit = jax.jit(arb_hypgeom_rgamma_batch)
acb_hypgeom_rgamma_batch_jit = jax.jit(acb_hypgeom_rgamma_batch)
arb_hypgeom_erf_batch_jit = jax.jit(arb_hypgeom_erf_batch)
acb_hypgeom_erf_batch_jit = jax.jit(acb_hypgeom_erf_batch)
arb_hypgeom_erfc_batch_jit = jax.jit(arb_hypgeom_erfc_batch)
acb_hypgeom_erfc_batch_jit = jax.jit(acb_hypgeom_erfc_batch)
arb_hypgeom_erfi_batch_jit = jax.jit(arb_hypgeom_erfi_batch)
acb_hypgeom_erfi_batch_jit = jax.jit(acb_hypgeom_erfi_batch)
arb_hypgeom_erfinv_batch_jit = jax.jit(arb_hypgeom_erfinv_batch)
arb_hypgeom_erfcinv_batch_jit = jax.jit(arb_hypgeom_erfcinv_batch)
arb_hypgeom_fresnel_batch_jit = jax.jit(arb_hypgeom_fresnel_batch, static_argnames=("normalized",))
arb_hypgeom_ei_batch_jit = jax.jit(arb_hypgeom_ei_batch)
arb_hypgeom_si_batch_jit = jax.jit(arb_hypgeom_si_batch)
arb_hypgeom_ci_batch_jit = jax.jit(arb_hypgeom_ci_batch)
arb_hypgeom_shi_batch_jit = jax.jit(arb_hypgeom_shi_batch)
arb_hypgeom_chi_batch_jit = jax.jit(arb_hypgeom_chi_batch)
arb_hypgeom_li_batch_jit = jax.jit(arb_hypgeom_li_batch, static_argnames=("offset",))
arb_hypgeom_dilog_batch_jit = jax.jit(arb_hypgeom_dilog_batch)
arb_hypgeom_airy_batch_jit = jax.jit(arb_hypgeom_airy_batch)
arb_hypgeom_expint_batch_jit = jax.jit(arb_hypgeom_expint_batch)
arb_hypgeom_gamma_lower_batch_jit = jax.jit(arb_hypgeom_gamma_lower_batch, static_argnames=("regularized",))
arb_hypgeom_gamma_upper_batch_jit = jax.jit(arb_hypgeom_gamma_upper_batch, static_argnames=("regularized",))
arb_hypgeom_beta_lower_batch_jit = jax.jit(arb_hypgeom_beta_lower_batch, static_argnames=("regularized",))
arb_hypgeom_chebyshev_t_batch_jit = jax.jit(arb_hypgeom_chebyshev_t_batch, static_argnames=("n",))
arb_hypgeom_chebyshev_u_batch_jit = jax.jit(arb_hypgeom_chebyshev_u_batch, static_argnames=("n",))
arb_hypgeom_laguerre_l_batch_jit = jax.jit(arb_hypgeom_laguerre_l_batch, static_argnames=("n",))
arb_hypgeom_hermite_h_batch_jit = jax.jit(arb_hypgeom_hermite_h_batch, static_argnames=("n",))
arb_hypgeom_legendre_p_batch_jit = jax.jit(arb_hypgeom_legendre_p_batch, static_argnames=("n", "type"))
arb_hypgeom_legendre_q_batch_jit = jax.jit(arb_hypgeom_legendre_q_batch, static_argnames=("n", "type"))
arb_hypgeom_jacobi_p_batch_jit = jax.jit(arb_hypgeom_jacobi_p_batch, static_argnames=("n",))
arb_hypgeom_gegenbauer_c_batch_jit = jax.jit(arb_hypgeom_gegenbauer_c_batch, static_argnames=("n",))
arb_hypgeom_central_bin_ui_batch_jit = jax.jit(arb_hypgeom_central_bin_ui_batch)
arb_hypgeom_bessel_j_batch_jit = jax.jit(arb_hypgeom_bessel_j_batch, static_argnames=("mode",))
arb_hypgeom_bessel_y_batch_jit = jax.jit(arb_hypgeom_bessel_y_batch, static_argnames=("mode",))
arb_hypgeom_bessel_jy_batch_jit = jax.jit(arb_hypgeom_bessel_jy_batch, static_argnames=("mode",))
arb_hypgeom_bessel_i_batch_jit = jax.jit(arb_hypgeom_bessel_i_batch, static_argnames=("mode",))
arb_hypgeom_bessel_k_batch_jit = jax.jit(arb_hypgeom_bessel_k_batch, static_argnames=("mode",))
arb_hypgeom_bessel_i_scaled_batch_jit = jax.jit(arb_hypgeom_bessel_i_scaled_batch, static_argnames=("mode",))
arb_hypgeom_bessel_k_scaled_batch_jit = jax.jit(arb_hypgeom_bessel_k_scaled_batch, static_argnames=("mode",))
arb_hypgeom_bessel_i_integration_batch_jit = jax.jit(
    arb_hypgeom_bessel_i_integration_batch, static_argnames=("scaled", "mode")
)
arb_hypgeom_bessel_k_integration_batch_jit = jax.jit(
    arb_hypgeom_bessel_k_integration_batch, static_argnames=("scaled", "mode")
)
arb_hypgeom_1f1_integration_batch_jit = jax.jit(arb_hypgeom_1f1_integration_batch, static_argnames=("regularized",))
arb_hypgeom_2f1_integration_batch_jit = jax.jit(arb_hypgeom_2f1_integration_batch, static_argnames=("regularized",))
arb_hypgeom_u_batch_jit = jax.jit(arb_hypgeom_u_batch)
arb_hypgeom_u_integration_batch_jit = jax.jit(arb_hypgeom_u_integration_batch)
acb_hypgeom_bessel_j_batch_jit = jax.jit(acb_hypgeom_bessel_j_batch)
acb_hypgeom_bessel_y_batch_jit = jax.jit(acb_hypgeom_bessel_y_batch)
acb_hypgeom_bessel_jy_batch_jit = jax.jit(acb_hypgeom_bessel_jy_batch)
acb_hypgeom_bessel_i_batch_jit = jax.jit(acb_hypgeom_bessel_i_batch)
acb_hypgeom_bessel_k_batch_jit = jax.jit(acb_hypgeom_bessel_k_batch)
acb_hypgeom_bessel_i_scaled_batch_jit = jax.jit(acb_hypgeom_bessel_i_scaled_batch)
acb_hypgeom_bessel_k_scaled_batch_jit = jax.jit(acb_hypgeom_bessel_k_scaled_batch)
acb_hypgeom_1f1_integration_batch_jit = jax.jit(acb_hypgeom_1f1_integration_batch, static_argnames=("regularized",))
acb_hypgeom_2f1_integration_batch_jit = jax.jit(acb_hypgeom_2f1_integration_batch, static_argnames=("regularized",))
acb_hypgeom_u_batch_jit = jax.jit(acb_hypgeom_u_batch)
acb_hypgeom_u_integration_batch_jit = jax.jit(acb_hypgeom_u_integration_batch)
arb_hypgeom_0f1_batch_jit = jax.jit(arb_hypgeom_0f1_batch, static_argnames=("regularized",))
acb_hypgeom_0f1_batch_jit = jax.jit(acb_hypgeom_0f1_batch, static_argnames=("regularized",))
arb_hypgeom_1f1_batch_jit = jax.jit(arb_hypgeom_1f1_batch, static_argnames=("regularized",))
acb_hypgeom_1f1_batch_jit = jax.jit(acb_hypgeom_1f1_batch, static_argnames=("regularized",))
arb_hypgeom_m_batch_jit = jax.jit(arb_hypgeom_m_batch, static_argnames=("regularized",))
acb_hypgeom_m_batch_jit = jax.jit(acb_hypgeom_m_batch, static_argnames=("regularized",))
arb_hypgeom_2f1_batch_jit = jax.jit(arb_hypgeom_2f1_batch, static_argnames=("regularized",))
acb_hypgeom_2f1_batch_jit = jax.jit(acb_hypgeom_2f1_batch, static_argnames=("regularized",))
arb_hypgeom_rising_ui_batch_prec_jit = jax.jit(arb_hypgeom_rising_ui_batch_prec, static_argnames=("n", "prec_bits"))
acb_hypgeom_rising_ui_batch_prec_jit = jax.jit(acb_hypgeom_rising_ui_batch_prec, static_argnames=("n", "prec_bits"))
arb_hypgeom_lgamma_batch_prec_jit = jax.jit(arb_hypgeom_lgamma_batch_prec, static_argnames=("prec_bits",))
acb_hypgeom_lgamma_batch_prec_jit = jax.jit(acb_hypgeom_lgamma_batch_prec, static_argnames=("prec_bits",))
arb_hypgeom_gamma_batch_prec_jit = jax.jit(arb_hypgeom_gamma_batch_prec, static_argnames=("prec_bits",))
acb_hypgeom_gamma_batch_prec_jit = jax.jit(acb_hypgeom_gamma_batch_prec, static_argnames=("prec_bits",))
arb_hypgeom_rgamma_batch_prec_jit = jax.jit(arb_hypgeom_rgamma_batch_prec, static_argnames=("prec_bits",))
acb_hypgeom_rgamma_batch_prec_jit = jax.jit(acb_hypgeom_rgamma_batch_prec, static_argnames=("prec_bits",))
arb_hypgeom_erf_batch_prec_jit = jax.jit(arb_hypgeom_erf_batch_prec, static_argnames=("prec_bits",))
acb_hypgeom_erf_batch_prec_jit = jax.jit(acb_hypgeom_erf_batch_prec, static_argnames=("prec_bits",))
arb_hypgeom_erfc_batch_prec_jit = jax.jit(arb_hypgeom_erfc_batch_prec, static_argnames=("prec_bits",))
acb_hypgeom_erfc_batch_prec_jit = jax.jit(acb_hypgeom_erfc_batch_prec, static_argnames=("prec_bits",))
arb_hypgeom_erfi_batch_prec_jit = jax.jit(arb_hypgeom_erfi_batch_prec, static_argnames=("prec_bits",))
acb_hypgeom_erfi_batch_prec_jit = jax.jit(acb_hypgeom_erfi_batch_prec, static_argnames=("prec_bits",))
arb_hypgeom_erfinv_batch_prec_jit = jax.jit(arb_hypgeom_erfinv_batch_prec, static_argnames=("prec_bits",))
arb_hypgeom_erfcinv_batch_prec_jit = jax.jit(arb_hypgeom_erfcinv_batch_prec, static_argnames=("prec_bits",))
arb_hypgeom_fresnel_batch_prec_jit = jax.jit(
    arb_hypgeom_fresnel_batch_prec, static_argnames=("prec_bits", "normalized")
)
arb_hypgeom_ei_batch_prec_jit = jax.jit(arb_hypgeom_ei_batch_prec, static_argnames=("prec_bits",))
arb_hypgeom_si_batch_prec_jit = jax.jit(arb_hypgeom_si_batch_prec, static_argnames=("prec_bits",))
arb_hypgeom_ci_batch_prec_jit = jax.jit(arb_hypgeom_ci_batch_prec, static_argnames=("prec_bits",))
arb_hypgeom_shi_batch_prec_jit = jax.jit(arb_hypgeom_shi_batch_prec, static_argnames=("prec_bits",))
arb_hypgeom_chi_batch_prec_jit = jax.jit(arb_hypgeom_chi_batch_prec, static_argnames=("prec_bits",))
arb_hypgeom_li_batch_prec_jit = jax.jit(arb_hypgeom_li_batch_prec, static_argnames=("prec_bits", "offset"))
arb_hypgeom_dilog_batch_prec_jit = jax.jit(arb_hypgeom_dilog_batch_prec, static_argnames=("prec_bits",))
arb_hypgeom_airy_batch_prec_jit = jax.jit(arb_hypgeom_airy_batch_prec, static_argnames=("prec_bits",))
arb_hypgeom_expint_batch_prec_jit = jax.jit(arb_hypgeom_expint_batch_prec, static_argnames=("prec_bits",))
arb_hypgeom_gamma_lower_batch_prec_jit = jax.jit(
    arb_hypgeom_gamma_lower_batch_prec, static_argnames=("prec_bits", "regularized")
)
arb_hypgeom_gamma_upper_batch_prec_jit = jax.jit(
    arb_hypgeom_gamma_upper_batch_prec, static_argnames=("prec_bits", "regularized")
)
arb_hypgeom_beta_lower_batch_prec_jit = jax.jit(
    arb_hypgeom_beta_lower_batch_prec, static_argnames=("prec_bits", "regularized")
)
arb_hypgeom_chebyshev_t_batch_prec_jit = jax.jit(
    arb_hypgeom_chebyshev_t_batch_prec, static_argnames=("prec_bits", "n")
)
arb_hypgeom_chebyshev_u_batch_prec_jit = jax.jit(
    arb_hypgeom_chebyshev_u_batch_prec, static_argnames=("prec_bits", "n")
)
arb_hypgeom_laguerre_l_batch_prec_jit = jax.jit(
    arb_hypgeom_laguerre_l_batch_prec, static_argnames=("prec_bits", "n")
)
arb_hypgeom_hermite_h_batch_prec_jit = jax.jit(
    arb_hypgeom_hermite_h_batch_prec, static_argnames=("prec_bits", "n")
)
arb_hypgeom_legendre_p_batch_prec_jit = jax.jit(
    arb_hypgeom_legendre_p_batch_prec, static_argnames=("prec_bits", "n", "type")
)
arb_hypgeom_legendre_q_batch_prec_jit = jax.jit(
    arb_hypgeom_legendre_q_batch_prec, static_argnames=("prec_bits", "n", "type")
)
arb_hypgeom_jacobi_p_batch_prec_jit = jax.jit(
    arb_hypgeom_jacobi_p_batch_prec, static_argnames=("prec_bits", "n")
)
arb_hypgeom_gegenbauer_c_batch_prec_jit = jax.jit(
    arb_hypgeom_gegenbauer_c_batch_prec, static_argnames=("prec_bits", "n")
)
arb_hypgeom_central_bin_ui_batch_prec_jit = jax.jit(
    arb_hypgeom_central_bin_ui_batch_prec, static_argnames=("prec_bits",)
)
arb_hypgeom_bessel_j_batch_prec_jit = jax.jit(arb_hypgeom_bessel_j_batch_prec, static_argnames=("prec_bits", "mode"))
arb_hypgeom_bessel_y_batch_prec_jit = jax.jit(arb_hypgeom_bessel_y_batch_prec, static_argnames=("prec_bits", "mode"))
arb_hypgeom_bessel_jy_batch_prec_jit = jax.jit(
    arb_hypgeom_bessel_jy_batch_prec, static_argnames=("prec_bits", "mode")
)
arb_hypgeom_bessel_i_batch_prec_jit = jax.jit(arb_hypgeom_bessel_i_batch_prec, static_argnames=("prec_bits", "mode"))
arb_hypgeom_bessel_k_batch_prec_jit = jax.jit(arb_hypgeom_bessel_k_batch_prec, static_argnames=("prec_bits", "mode"))
arb_hypgeom_bessel_i_scaled_batch_prec_jit = jax.jit(
    arb_hypgeom_bessel_i_scaled_batch_prec, static_argnames=("prec_bits", "mode")
)
arb_hypgeom_bessel_k_scaled_batch_prec_jit = jax.jit(
    arb_hypgeom_bessel_k_scaled_batch_prec, static_argnames=("prec_bits", "mode")
)
arb_hypgeom_bessel_i_integration_batch_prec_jit = jax.jit(
    arb_hypgeom_bessel_i_integration_batch_prec, static_argnames=("prec_bits", "scaled", "mode")
)
arb_hypgeom_bessel_k_integration_batch_prec_jit = jax.jit(
    arb_hypgeom_bessel_k_integration_batch_prec, static_argnames=("prec_bits", "scaled", "mode")
)
arb_hypgeom_1f1_integration_batch_prec_jit = jax.jit(
    arb_hypgeom_1f1_integration_batch_prec, static_argnames=("prec_bits", "regularized")
)
arb_hypgeom_2f1_integration_batch_prec_jit = jax.jit(
    arb_hypgeom_2f1_integration_batch_prec, static_argnames=("prec_bits", "regularized")
)
arb_hypgeom_u_batch_prec_jit = jax.jit(arb_hypgeom_u_batch_prec, static_argnames=("prec_bits",))
arb_hypgeom_u_integration_batch_prec_jit = jax.jit(arb_hypgeom_u_integration_batch_prec, static_argnames=("prec_bits",))
acb_hypgeom_bessel_j_batch_prec_jit = jax.jit(acb_hypgeom_bessel_j_batch_prec, static_argnames=("prec_bits",))
acb_hypgeom_bessel_y_batch_prec_jit = jax.jit(acb_hypgeom_bessel_y_batch_prec, static_argnames=("prec_bits",))
acb_hypgeom_bessel_jy_batch_prec_jit = jax.jit(acb_hypgeom_bessel_jy_batch_prec, static_argnames=("prec_bits",))
acb_hypgeom_bessel_i_batch_prec_jit = jax.jit(acb_hypgeom_bessel_i_batch_prec, static_argnames=("prec_bits",))
acb_hypgeom_bessel_k_batch_prec_jit = jax.jit(acb_hypgeom_bessel_k_batch_prec, static_argnames=("prec_bits",))
acb_hypgeom_bessel_i_scaled_batch_prec_jit = jax.jit(acb_hypgeom_bessel_i_scaled_batch_prec, static_argnames=("prec_bits",))
acb_hypgeom_bessel_k_scaled_batch_prec_jit = jax.jit(acb_hypgeom_bessel_k_scaled_batch_prec, static_argnames=("prec_bits",))
acb_hypgeom_u_batch_prec_jit = jax.jit(acb_hypgeom_u_batch_prec, static_argnames=("prec_bits",))
acb_hypgeom_u_integration_batch_prec_jit = jax.jit(acb_hypgeom_u_integration_batch_prec, static_argnames=("prec_bits",))
acb_hypgeom_1f1_integration_batch_prec_jit = jax.jit(
    acb_hypgeom_1f1_integration_batch_prec, static_argnames=("prec_bits", "regularized")
)
acb_hypgeom_2f1_integration_batch_prec_jit = jax.jit(
    acb_hypgeom_2f1_integration_batch_prec, static_argnames=("prec_bits", "regularized")
)
arb_hypgeom_0f1_batch_prec_jit = jax.jit(arb_hypgeom_0f1_batch_prec, static_argnames=("prec_bits", "regularized"))
acb_hypgeom_0f1_batch_prec_jit = jax.jit(acb_hypgeom_0f1_batch_prec, static_argnames=("prec_bits", "regularized"))
arb_hypgeom_1f1_batch_prec_jit = jax.jit(arb_hypgeom_1f1_batch_prec, static_argnames=("prec_bits", "regularized"))
acb_hypgeom_1f1_batch_prec_jit = jax.jit(acb_hypgeom_1f1_batch_prec, static_argnames=("prec_bits", "regularized"))
arb_hypgeom_m_batch_prec_jit = jax.jit(arb_hypgeom_m_batch_prec, static_argnames=("prec_bits", "regularized"))
acb_hypgeom_m_batch_prec_jit = jax.jit(acb_hypgeom_m_batch_prec, static_argnames=("prec_bits", "regularized"))
arb_hypgeom_2f1_batch_prec_jit = jax.jit(arb_hypgeom_2f1_batch_prec, static_argnames=("prec_bits", "regularized"))
acb_hypgeom_2f1_batch_prec_jit = jax.jit(acb_hypgeom_2f1_batch_prec, static_argnames=("prec_bits", "regularized"))


def acb_midpoint(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    re = di.midpoint(acb_real(box))
    im = di.midpoint(acb_imag(box))
    return re + 1j * im
