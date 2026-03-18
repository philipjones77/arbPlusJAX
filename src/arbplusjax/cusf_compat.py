from __future__ import annotations

"""CuSF/CUDA-lineage alternative implementations.

These functions are alternatives to canonical Arb-like public names and should
be surfaced through provenance-prefixed names such as `cusf_besselk`.

Provenance:
- classification: alternative
- module lineage: CuSF/CUDA-style implementation family
- naming policy: see docs/standards/function_naming.md
- registry report: see docs/status/reports/function_implementation_index.md
"""

import jax
from jax import lax
import jax.numpy as jnp

from . import baseline_wrappers
from . import double_interval as di
from . import hypgeom
from . import hypgeom_wrappers


PROVENANCE = {
    "classification": "alternative",
    "module_lineage": "CuSF/CUDA-style implementation family",
    "preferred_prefix": "cusf",
    "naming_policy": "docs/standards/function_naming.md",
    "registry_report": "docs/status/reports/function_implementation_index.md",
}


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
def _cusf_hyp1f1_point(a: jax.Array, b: jax.Array, x: jax.Array) -> jax.Array:
    return hypgeom._real_hyp1f1_regime(_as_f64(a), _as_f64(b), _as_f64(x))


@jax.jit
def _cusf_hyp2f1_point(a: jax.Array, b: jax.Array, c: jax.Array, x: jax.Array) -> jax.Array:
    return hypgeom._real_hyp2f1_regime(_as_f64(a), _as_f64(b), _as_f64(c), _as_f64(x))


@jax.jit
def _cusf_faddeeva_w_point(z: jax.Array) -> jax.Array:
    zc = _as_c128(z)
    return jnp.exp(-(zc * zc)) * hypgeom._complex_erfc_series(-1j * zc)


@jax.jit
def _cusf_erf_point(x: jax.Array) -> jax.Array:
    xr = _as_f64(x)
    return hypgeom._real_erf_series(xr)


@jax.jit
def _cusf_besselj_point(v: jax.Array, x: jax.Array) -> jax.Array:
    return hypgeom._real_bessel_eval_j(_as_f64(v), _as_f64(x))


@jax.jit
def _cusf_bessely_point(v: jax.Array, x: jax.Array) -> jax.Array:
    return hypgeom._real_bessel_eval_y(_as_f64(v), _as_f64(x))


@jax.jit
def _cusf_besseli_point(v: jax.Array, x: jax.Array) -> jax.Array:
    return hypgeom._real_bessel_eval_i(_as_f64(v), _as_f64(x))


@jax.jit
def _cusf_besselk_point(v: jax.Array, x: jax.Array) -> jax.Array:
    return hypgeom._real_bessel_eval_k(_as_f64(v), _as_f64(x))


@jax.jit
def _cusf_besseljy_point(v: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return _cusf_besselj_point(v, x), _cusf_bessely_point(v, x)


@jax.jit
def _cusf_besselj0_point(x: jax.Array) -> jax.Array:
    return _cusf_besselj_point(jnp.float64(0.0), x)


@jax.jit
def _cusf_bessely0_point(x: jax.Array) -> jax.Array:
    return _cusf_bessely_point(jnp.float64(0.0), x)


@jax.jit
def _cusf_besseli0_point(x: jax.Array) -> jax.Array:
    return _cusf_besseli_point(jnp.float64(0.0), x)


@jax.jit
def _cusf_besselk0_point(x: jax.Array) -> jax.Array:
    return _cusf_besselk_point(jnp.float64(0.0), x)


@jax.jit
def _cusf_besselj_deriv_point(v: jax.Array, x: jax.Array) -> jax.Array:
    v0 = _as_f64(v)
    x0 = _as_f64(x)
    return 0.5 * (_cusf_besselj_point(v0 - 1.0, x0) - _cusf_besselj_point(v0 + 1.0, x0))


@jax.jit
def _cusf_bessely_deriv_point(v: jax.Array, x: jax.Array) -> jax.Array:
    v0 = _as_f64(v)
    x0 = _as_f64(x)
    return 0.5 * (_cusf_bessely_point(v0 - 1.0, x0) - _cusf_bessely_point(v0 + 1.0, x0))


@jax.jit
def _cusf_besseli_deriv_point(v: jax.Array, x: jax.Array) -> jax.Array:
    v0 = _as_f64(v)
    x0 = _as_f64(x)
    return 0.5 * (_cusf_besseli_point(v0 - 1.0, x0) + _cusf_besseli_point(v0 + 1.0, x0))


@jax.jit
def _cusf_besselk_deriv_point(v: jax.Array, x: jax.Array) -> jax.Array:
    v0 = _as_f64(v)
    x0 = _as_f64(x)
    return -0.5 * (_cusf_besselk_point(v0 - 1.0, x0) + _cusf_besselk_point(v0 + 1.0, x0))


@jax.jit
def _cusf_digamma_point(x: jax.Array) -> jax.Array:
    return lax.digamma(_as_f64(x))


@jax.jit
def _cusf_gamma_point(x: jax.Array) -> jax.Array:
    return jnp.exp(lax.lgamma(_as_f64(x)))


@jax.jit
def _cusf_tgamma1pmv_point(v: jax.Array) -> jax.Array:
    x = _as_f64(v)
    return jnp.expm1(lax.lgamma(1.0 + x))


@jax.jit
def _cusf_polynomial_point(x: jax.Array, coeffs: jax.Array) -> jax.Array:
    xx = _as_f64(x)
    cc = _as_f64(coeffs)
    y = jnp.zeros_like(xx)
    for c in cc:
        y = y * xx + c
    return y


@jax.jit
def _cusf_poly_rational_point(x: jax.Array, p: jax.Array, q: jax.Array) -> jax.Array:
    num = _cusf_polynomial_point(x, p)
    den = _cusf_polynomial_point(x, q)
    return num / den


@jax.jit
def _cusf_chebyshev_point(x: jax.Array, coeffs: jax.Array) -> jax.Array:
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


def cusf_hyp1f1(a: jax.Array, b: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _cusf_hyp1f1_point(a, b, x)
    return hypgeom_wrappers.arb_hypgeom_1f1_mode(_as_interval(a), _as_interval(b), _as_interval(x), impl=mode, prec_bits=prec_bits)


def cusf_hyp2f1(a: jax.Array, b: jax.Array, c: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _cusf_hyp2f1_point(a, b, c, x)
    return hypgeom_wrappers.arb_hypgeom_2f1_mode(
        _as_interval(a), _as_interval(b), _as_interval(c), _as_interval(x), impl=mode, prec_bits=prec_bits
    )


def cusf_faddeeva_w(z: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    v = _cusf_faddeeva_w_point(z)
    if mode == "point":
        return v
    return _interval_from_point(jnp.real(v), prec_bits, adaptive=(mode == "adaptive"))


def cusf_erf(x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _cusf_erf_point(x)
    return baseline_wrappers.arb_erf_mp(_as_interval(x), mode=mode, prec_bits=prec_bits)


def cusf_besselj(v: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _cusf_besselj_point(v, x)
    return baseline_wrappers.arb_bessel_j_mp(_as_interval(v), _as_interval(x), mode=mode, prec_bits=prec_bits)


def cusf_bessely(v: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _cusf_bessely_point(v, x)
    return baseline_wrappers.arb_bessel_y_mp(_as_interval(v), _as_interval(x), mode=mode, prec_bits=prec_bits)


def cusf_besseli(v: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _cusf_besseli_point(v, x)
    return baseline_wrappers.arb_bessel_i_mp(_as_interval(v), _as_interval(x), mode=mode, prec_bits=prec_bits)


def cusf_besselk(v: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _cusf_besselk_point(v, x)
    return baseline_wrappers.arb_bessel_k_mp(_as_interval(v), _as_interval(x), mode=mode, prec_bits=prec_bits)


def cusf_besseljy(v: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> tuple[jax.Array, jax.Array]:
    return cusf_besselj(v, x, mode=mode, prec_bits=prec_bits), cusf_bessely(v, x, mode=mode, prec_bits=prec_bits)


def cusf_besselj0(x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    return cusf_besselj(jnp.float64(0.0), x, mode=mode, prec_bits=prec_bits)


def cusf_bessely0(x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    return cusf_bessely(jnp.float64(0.0), x, mode=mode, prec_bits=prec_bits)


def cusf_besseli0(x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    return cusf_besseli(jnp.float64(0.0), x, mode=mode, prec_bits=prec_bits)


def cusf_besselk0(x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    return cusf_besselk(jnp.float64(0.0), x, mode=mode, prec_bits=prec_bits)


def cusf_besselj_deriv(v: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _cusf_besselj_deriv_point(v, x)
    return _imul_scalar(_isub(cusf_besselj(v - 1.0, x, mode=mode, prec_bits=prec_bits), cusf_besselj(v + 1.0, x, mode=mode, prec_bits=prec_bits)), 0.5)


def cusf_bessely_deriv(v: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _cusf_bessely_deriv_point(v, x)
    return _imul_scalar(_isub(cusf_bessely(v - 1.0, x, mode=mode, prec_bits=prec_bits), cusf_bessely(v + 1.0, x, mode=mode, prec_bits=prec_bits)), 0.5)


def cusf_besseli_deriv(v: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _cusf_besseli_deriv_point(v, x)
    return _imul_scalar(_iadd(cusf_besseli(v - 1.0, x, mode=mode, prec_bits=prec_bits), cusf_besseli(v + 1.0, x, mode=mode, prec_bits=prec_bits)), 0.5)


def cusf_besselk_deriv(v: jax.Array, x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _cusf_besselk_deriv_point(v, x)
    return _imul_scalar(_iadd(cusf_besselk(v - 1.0, x, mode=mode, prec_bits=prec_bits), cusf_besselk(v + 1.0, x, mode=mode, prec_bits=prec_bits)), -0.5)


def cusf_digamma(x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    v = _cusf_digamma_point(x)
    if mode == "point":
        return v
    return _interval_from_point(v, prec_bits, adaptive=(mode == "adaptive"))


def cusf_gamma(x: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _cusf_gamma_point(x)
    return baseline_wrappers.arb_gamma_mp(_as_interval(x), mode=mode, prec_bits=prec_bits)


def cusf_tgamma1pmv(v: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _cusf_tgamma1pmv_point(v)
    g = cusf_gamma(1.0 + v, mode=mode, prec_bits=prec_bits)
    return _isub(g, di.interval(1.0, 1.0))


def cusf_polynomial(x: jax.Array, coeffs: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _cusf_polynomial_point(x, coeffs)
    iv = _as_interval(x)
    lo = _cusf_polynomial_point(iv[..., 0], coeffs)
    hi = _cusf_polynomial_point(iv[..., 1], coeffs)
    mid = _cusf_polynomial_point(0.5 * (iv[..., 0] + iv[..., 1]), coeffs)
    out = di.interval(jnp.minimum(jnp.minimum(lo, hi), mid), jnp.maximum(jnp.maximum(lo, hi), mid))
    return di.round_interval_outward(out, prec_bits)


def cusf_poly_rational(x: jax.Array, p: jax.Array, q: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _cusf_poly_rational_point(x, p, q)
    num = cusf_polynomial(x, p, mode=mode, prec_bits=prec_bits)
    den = cusf_polynomial(x, q, mode=mode, prec_bits=prec_bits)
    return di.fast_div(num, den)


def cusf_chebyshev(x: jax.Array, coeffs: jax.Array, mode: str = "point", prec_bits: int = 80) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return _cusf_chebyshev_point(x, coeffs)
    iv = _as_interval(x)
    lo = _cusf_chebyshev_point(iv[..., 0], coeffs)
    hi = _cusf_chebyshev_point(iv[..., 1], coeffs)
    mid = _cusf_chebyshev_point(0.5 * (iv[..., 0] + iv[..., 1]), coeffs)
    out = di.interval(jnp.minimum(jnp.minimum(lo, hi), mid), jnp.maximum(jnp.maximum(lo, hi), mid))
    return di.round_interval_outward(out, prec_bits)


__all__ = [
    "cusf_hyp2f1",
    "cusf_hyp1f1",
    "cusf_faddeeva_w",
    "cusf_erf",
    "cusf_besseljy",
    "cusf_besselk",
    "cusf_besseli",
    "cusf_besselj0",
    "cusf_bessely",
    "cusf_besselj_deriv",
    "cusf_bessely0",
    "cusf_besselj",
    "cusf_besselk0",
    "cusf_besselk_deriv",
    "cusf_besseli0",
    "cusf_besseli_deriv",
    "cusf_bessely_deriv",
    "cusf_digamma",
    "cusf_gamma",
    "cusf_tgamma1pmv",
    "cusf_chebyshev",
    "cusf_polynomial",
    "cusf_poly_rational",
]
