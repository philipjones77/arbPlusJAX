from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp

from . import baseline_wrappers
from . import double_interval as di
from . import hypgeom
from . import hypgeom_wrappers

jax.config.update("jax_enable_x64", True)


def _as_f64(x: jax.Array) -> jax.Array:
    return jnp.asarray(x, dtype=jnp.float64)


def _as_c128(z: jax.Array) -> jax.Array:
    return jnp.asarray(z, dtype=jnp.complex128)


def _check_mode(mode: str) -> str:
    allowed = ("point", "basic", "rigorous", "adaptive")
    if mode not in allowed:
        raise ValueError(f"Cusf mode must be one of {allowed}, got {mode!r}")
    return mode


def _as_interval(x: jax.Array) -> jax.Array:
    x = jnp.asarray(x, dtype=jnp.float64)
    if x.ndim >= 1 and x.shape[-1] == 2:
        return di.as_interval(x)
    return jnp.stack([x, x], axis=-1)


def _interval_from_point(x: jax.Array, prec_bits: int, adaptive: bool = False) -> jax.Array:
    xx = jnp.asarray(x, dtype=jnp.float64)
    base = jnp.stack([xx, xx], axis=-1)
    out = di.round_interval_outward(base, prec_bits)
    if not adaptive:
        return out
    eps = jnp.exp2(-jnp.float64(prec_bits)) * (1.0 + jnp.abs(xx))
    return di.interval(out[..., 0] - eps, out[..., 1] + eps)


def _iadd(a: jax.Array, b: jax.Array) -> jax.Array:
    return di.fast_add(a, b)


def _isub(a: jax.Array, b: jax.Array) -> jax.Array:
    return di.fast_sub(a, b)


def _imul_scalar(a: jax.Array, c: float) -> jax.Array:
    cc = di.interval(jnp.float64(c), jnp.float64(c))
    return di.fast_mul(a, cc)


@jax.jit
def _Cusf_Hyp1f1_point(a: jax.Array, b: jax.Array, x: jax.Array) -> jax.Array:
    return hypgeom._real_hyp1f1_regime(_as_f64(a), _as_f64(b), _as_f64(x))


@jax.jit
def _Cusf_Hyp2f1_point(a: jax.Array, b: jax.Array, c: jax.Array, x: jax.Array) -> jax.Array:
    return hypgeom._real_hyp2f1_regime(_as_f64(a), _as_f64(b), _as_f64(c), _as_f64(x))


@jax.jit
def _Cusf_faddeeva_w_point(z: jax.Array) -> jax.Array:
    zc = _as_c128(z)
    return jnp.exp(-(zc * zc)) * hypgeom._complex_erfc_series(-1j * zc)


@jax.jit
def _Cusf_erf_point(x: jax.Array) -> jax.Array:
    xr = _as_f64(x)
    return hypgeom._real_erf_series(xr)


@jax.jit
def _Cusf_jv_point(v: jax.Array, x: jax.Array) -> jax.Array:
    return hypgeom._real_bessel_eval_j(_as_f64(v), _as_f64(x))


@jax.jit
def _Cusf_yv_point(v: jax.Array, x: jax.Array) -> jax.Array:
    return hypgeom._real_bessel_eval_y(_as_f64(v), _as_f64(x))


@jax.jit
def _Cusf_iv_point(v: jax.Array, x: jax.Array) -> jax.Array:
    return hypgeom._real_bessel_eval_i(_as_f64(v), _as_f64(x))


@jax.jit
def _Cusf_kv_point(v: jax.Array, x: jax.Array) -> jax.Array:
    return hypgeom._real_bessel_eval_k(_as_f64(v), _as_f64(x))


@jax.jit
def _Cusf_jy_point(v: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return _Cusf_jv_point(v, x), _Cusf_yv_point(v, x)


@jax.jit
def _Cusf_j0_point(x: jax.Array) -> jax.Array:
    return _Cusf_jv_point(jnp.float64(0.0), x)


@jax.jit
def _Cusf_y0_point(x: jax.Array) -> jax.Array:
    return _Cusf_yv_point(jnp.float64(0.0), x)


@jax.jit
def _Cusf_i0_point(x: jax.Array) -> jax.Array:
    return _Cusf_iv_point(jnp.float64(0.0), x)


@jax.jit
def _Cusf_k0_point(x: jax.Array) -> jax.Array:
    return _Cusf_kv_point(jnp.float64(0.0), x)


@jax.jit
def _Cusf_jvp_point(v: jax.Array, x: jax.Array) -> jax.Array:
    v0 = _as_f64(v)
    x0 = _as_f64(x)
    return 0.5 * (_Cusf_jv_point(v0 - 1.0, x0) - _Cusf_jv_point(v0 + 1.0, x0))


@jax.jit
def _Cusf_yvp_point(v: jax.Array, x: jax.Array) -> jax.Array:
    v0 = _as_f64(v)
    x0 = _as_f64(x)
    return 0.5 * (_Cusf_yv_point(v0 - 1.0, x0) - _Cusf_yv_point(v0 + 1.0, x0))


@jax.jit
def _Cusf_ivp_point(v: jax.Array, x: jax.Array) -> jax.Array:
    v0 = _as_f64(v)
    x0 = _as_f64(x)
    return 0.5 * (_Cusf_iv_point(v0 - 1.0, x0) + _Cusf_iv_point(v0 + 1.0, x0))


@jax.jit
def _Cusf_kvp_point(v: jax.Array, x: jax.Array) -> jax.Array:
    v0 = _as_f64(v)
    x0 = _as_f64(x)
    return -0.5 * (_Cusf_kv_point(v0 - 1.0, x0) + _Cusf_kv_point(v0 + 1.0, x0))


@jax.jit
def _Cusf_digamma_point(x: jax.Array) -> jax.Array:
    return lax.digamma(_as_f64(x))


@jax.jit
def _Cusf_gamma_point(x: jax.Array) -> jax.Array:
    return jnp.exp(lax.lgamma(_as_f64(x)))


@jax.jit
def _Cusf_tgamma1pmv_point(v: jax.Array) -> jax.Array:
    x = _as_f64(v)
    return jnp.expm1(lax.lgamma(1.0 + x))


@jax.jit
def _Cusf_polynomial_point(x: jax.Array, coeffs: jax.Array) -> jax.Array:
    xx = _as_f64(x)
    cc = _as_f64(coeffs)
    y = jnp.zeros_like(xx)
    for c in cc:
        y = y * xx + c
    return y


@jax.jit
def _Cusf_poly_rational_point(x: jax.Array, p: jax.Array, q: jax.Array) -> jax.Array:
    num = _Cusf_polynomial_point(x, p)
    den = _Cusf_polynomial_point(x, q)
    return num / den


@jax.jit
def _Cusf_chebyshev_point(x: jax.Array, coeffs: jax.Array) -> jax.Array:
    xx = _as_f64(x)
    cc = _as_f64(coeffs)
    b_kplus1 = jnp.zeros_like(xx)
    b_kplus2 = jnp.zeros_like(xx)
    two_x = 2.0 * xx
    for c in cc[::-1]:
        b_k = c + two_x * b_kplus1 - b_kplus2
        b_kplus2 = b_kplus1
        b_kplus1 = b_k
    return b_kplus1 - xx * b_kplus2


def Cusf_Hyp1f1(a: jax.Array, b: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _Cusf_Hyp1f1_point(a, b, x)
    return hypgeom_wrappers.arb_hypgeom_1f1_mode(_as_interval(a), _as_interval(b), _as_interval(x), impl=mode, prec_bits=prec_bits)


def Cusf_Hyp2f1(a: jax.Array, b: jax.Array, c: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _Cusf_Hyp2f1_point(a, b, c, x)
    return hypgeom_wrappers.arb_hypgeom_2f1_mode(
        _as_interval(a), _as_interval(b), _as_interval(c), _as_interval(x), impl=mode, prec_bits=prec_bits
    )


def Cusf_faddeeva_w(z: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    v = _Cusf_faddeeva_w_point(z)
    if mode == "point":
        return v
    return _interval_from_point(jnp.real(v), prec_bits, adaptive=(mode == "adaptive"))


def Cusf_erf(x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _Cusf_erf_point(x)
    return baseline_wrappers.arb_erf_mp(_as_interval(x), mode=mode, prec_bits=prec_bits)


def Cusf_jv(v: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _Cusf_jv_point(v, x)
    return baseline_wrappers.arb_bessel_j_mp(_as_interval(v), _as_interval(x), mode=mode, prec_bits=prec_bits)


def Cusf_yv(v: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _Cusf_yv_point(v, x)
    return baseline_wrappers.arb_bessel_y_mp(_as_interval(v), _as_interval(x), mode=mode, prec_bits=prec_bits)


def Cusf_iv(v: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _Cusf_iv_point(v, x)
    return baseline_wrappers.arb_bessel_i_mp(_as_interval(v), _as_interval(x), mode=mode, prec_bits=prec_bits)


def Cusf_kv(v: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _Cusf_kv_point(v, x)
    return baseline_wrappers.arb_bessel_k_mp(_as_interval(v), _as_interval(x), mode=mode, prec_bits=prec_bits)


def Cusf_jy(v: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> tuple[jax.Array, jax.Array]:
    return Cusf_jv(v, x, mode=mode, prec_bits=prec_bits), Cusf_yv(v, x, mode=mode, prec_bits=prec_bits)


def Cusf_j0(x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    return Cusf_jv(jnp.float64(0.0), x, mode=mode, prec_bits=prec_bits)


def Cusf_y0(x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    return Cusf_yv(jnp.float64(0.0), x, mode=mode, prec_bits=prec_bits)


def Cusf_i0(x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    return Cusf_iv(jnp.float64(0.0), x, mode=mode, prec_bits=prec_bits)


def Cusf_k0(x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    return Cusf_kv(jnp.float64(0.0), x, mode=mode, prec_bits=prec_bits)


def Cusf_jvp(v: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _Cusf_jvp_point(v, x)
    return _imul_scalar(_isub(Cusf_jv(v - 1.0, x, mode=mode, prec_bits=prec_bits), Cusf_jv(v + 1.0, x, mode=mode, prec_bits=prec_bits)), 0.5)


def Cusf_yvp(v: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _Cusf_yvp_point(v, x)
    return _imul_scalar(_isub(Cusf_yv(v - 1.0, x, mode=mode, prec_bits=prec_bits), Cusf_yv(v + 1.0, x, mode=mode, prec_bits=prec_bits)), 0.5)


def Cusf_ivp(v: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _Cusf_ivp_point(v, x)
    return _imul_scalar(_iadd(Cusf_iv(v - 1.0, x, mode=mode, prec_bits=prec_bits), Cusf_iv(v + 1.0, x, mode=mode, prec_bits=prec_bits)), 0.5)


def Cusf_kvp(v: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _Cusf_kvp_point(v, x)
    return _imul_scalar(_iadd(Cusf_kv(v - 1.0, x, mode=mode, prec_bits=prec_bits), Cusf_kv(v + 1.0, x, mode=mode, prec_bits=prec_bits)), -0.5)


def Cusf_digamma(x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    v = _Cusf_digamma_point(x)
    if mode == "point":
        return v
    return _interval_from_point(v, prec_bits, adaptive=(mode == "adaptive"))


def Cusf_gamma(x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _Cusf_gamma_point(x)
    return baseline_wrappers.arb_gamma_mp(_as_interval(x), mode=mode, prec_bits=prec_bits)


def Cusf_tgamma1pmv(v: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _Cusf_tgamma1pmv_point(v)
    g = Cusf_gamma(1.0 + v, mode=mode, prec_bits=prec_bits)
    return _isub(g, di.interval(1.0, 1.0))


def Cusf_polynomial(x: jax.Array, coeffs: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _Cusf_polynomial_point(x, coeffs)
    iv = _as_interval(x)
    lo = _Cusf_polynomial_point(iv[..., 0], coeffs)
    hi = _Cusf_polynomial_point(iv[..., 1], coeffs)
    mid = _Cusf_polynomial_point(0.5 * (iv[..., 0] + iv[..., 1]), coeffs)
    out = di.interval(jnp.minimum(jnp.minimum(lo, hi), mid), jnp.maximum(jnp.maximum(lo, hi), mid))
    return di.round_interval_outward(out, prec_bits)


def Cusf_poly_rational(x: jax.Array, p: jax.Array, q: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _Cusf_poly_rational_point(x, p, q)
    num = Cusf_polynomial(x, p, mode=mode, prec_bits=prec_bits)
    den = Cusf_polynomial(x, q, mode=mode, prec_bits=prec_bits)
    return di.fast_div(num, den)


def Cusf_chebyshev(x: jax.Array, coeffs: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _Cusf_chebyshev_point(x, coeffs)
    iv = _as_interval(x)
    lo = _Cusf_chebyshev_point(iv[..., 0], coeffs)
    hi = _Cusf_chebyshev_point(iv[..., 1], coeffs)
    mid = _Cusf_chebyshev_point(0.5 * (iv[..., 0] + iv[..., 1]), coeffs)
    out = di.interval(jnp.minimum(jnp.minimum(lo, hi), mid), jnp.maximum(jnp.maximum(lo, hi), mid))
    return di.round_interval_outward(out, prec_bits)


__all__ = [
    "Cusf_Hyp2f1",
    "Cusf_Hyp1f1",
    "Cusf_faddeeva_w",
    "Cusf_erf",
    "Cusf_jy",
    "Cusf_kv",
    "Cusf_iv",
    "Cusf_j0",
    "Cusf_yv",
    "Cusf_jvp",
    "Cusf_y0",
    "Cusf_jv",
    "Cusf_k0",
    "Cusf_kvp",
    "Cusf_i0",
    "Cusf_ivp",
    "Cusf_yvp",
    "Cusf_digamma",
    "Cusf_gamma",
    "Cusf_tgamma1pmv",
    "Cusf_chebyshev",
    "Cusf_polynomial",
    "Cusf_poly_rational",
]
