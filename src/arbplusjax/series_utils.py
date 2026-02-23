from __future__ import annotations

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp

from . import double_interval as di
from . import acb_core

jax.config.update("jax_enable_x64", True)


def taylor_series_unary_real(fn, x0: jax.Array, length: int) -> jax.Array:
    coeffs = []
    f = fn
    for k in range(length):
        val = f(x0)
        fact = jnp.exp(lax.lgamma(jnp.float64(k + 1)))
        coeffs.append(val / fact)
        f = jax.grad(f)
    return jnp.asarray(coeffs, dtype=jnp.float64)


def taylor_series_unary_complex(fn, z0: jax.Array, length: int) -> jax.Array:
    z0 = jnp.asarray(z0, dtype=jnp.complex128)

    def f_real(t):
        return jnp.real(fn(z0 + t))

    def f_imag(t):
        return jnp.imag(fn(z0 + t))

    fr = f_real
    fi = f_imag
    coeffs = []
    for k in range(length):
        val = fr(0.0) + 1j * fi(0.0)
        fact = jnp.exp(lax.lgamma(jnp.float64(k + 1)))
        coeffs.append(val / fact)
        fr = jax.grad(fr)
        fi = jax.grad(fi)
    return jnp.asarray(coeffs, dtype=jnp.complex128)


def poly_eval(coeffs: jax.Array, x: jax.Array) -> jax.Array:
    coeffs = jnp.asarray(coeffs)
    def body(i, acc):
        return acc + coeffs[i] * (x ** jnp.float64(i))
    return lax.fori_loop(0, coeffs.shape[0], body, jnp.zeros_like(x, dtype=coeffs.dtype))


def series_from_poly_real(fn, coeffs: jax.Array, length: int) -> jax.Array:
    coeffs = di.as_interval(coeffs)
    mid = di.midpoint(coeffs)

    def h(t):
        return poly_eval(mid, t)

    def g(t):
        return fn(h(t))

    return taylor_series_unary_real(g, jnp.float64(0.0), length)


def series_from_poly_complex(fn, coeffs: jax.Array, length: int) -> jax.Array:
    coeffs = acb_core.as_acb_box(coeffs)
    re = acb_core.acb_real(coeffs)
    im = acb_core.acb_imag(coeffs)
    mid = di.midpoint(re) + 1j * di.midpoint(im)

    def h(t):
        return poly_eval(mid, t)

    def g(t):
        return fn(h(t))

    return taylor_series_unary_complex(g, jnp.float64(0.0), length)


def coeffs_to_intervals(vals: jax.Array, prec_bits: int = 53) -> jax.Array:
    vals = jnp.asarray(vals, dtype=jnp.float64)
    eps = jnp.exp2(-jnp.float64(prec_bits)) * (1.0 + jnp.abs(vals))
    return jnp.stack([di._below(vals - eps), di._above(vals + eps)], axis=-1)


def coeffs_to_boxes(vals: jax.Array, prec_bits: int = 53) -> jax.Array:
    vals = jnp.asarray(vals, dtype=jnp.complex128)
    re = jnp.real(vals)
    im = jnp.imag(vals)
    eps = jnp.exp2(-jnp.float64(prec_bits)) * (1.0 + jnp.abs(vals))
    return jnp.stack([
        di._below(re - eps), di._above(re + eps), di._below(im - eps), di._above(im + eps)
    ], axis=-1)


def series_add(a: jax.Array, b: jax.Array, length: int) -> jax.Array:
    a = a[:length]
    b = b[:length]
    return a + b


def series_mul(a: jax.Array, b: jax.Array, length: int) -> jax.Array:
    a = a[:length]
    b = b[:length]
    out = jnp.zeros((length,), dtype=a.dtype)
    for k in range(length):
        s = jnp.zeros((), dtype=a.dtype)
        for j in range(k + 1):
            s = s + a[j] * b[k - j]
        out = out.at[k].set(s)
    return out


def series_inv(a: jax.Array, length: int) -> jax.Array:
    a = a[:length]
    out = jnp.zeros((length,), dtype=a.dtype)
    out = out.at[0].set(1.0 / a[0])
    for k in range(1, length):
        s = jnp.zeros((), dtype=a.dtype)
        for j in range(1, k + 1):
            s = s + a[j] * out[k - j]
        out = out.at[k].set(-s / a[0])
    return out


def series_div(a: jax.Array, b: jax.Array, length: int) -> jax.Array:
    return series_mul(a, series_inv(b, length), length)


def series_compose(f: jax.Array, g: jax.Array, length: int) -> jax.Array:
    f = f[:length]
    g = g[:length]
    out = jnp.zeros((length,), dtype=f.dtype)
    out = out.at[0].set(f[0])
    g_pow = jnp.zeros((length,), dtype=f.dtype)
    g_pow = g_pow.at[0].set(1.0)
    for k in range(1, length):
        g_pow = series_mul(g_pow, g, length)
        out = out + f[k] * g_pow
    return out


def series_revert(f: jax.Array, length: int) -> jax.Array:
    # Lagrange inversion for f with f[0]=0 and f[1]!=0
    f = f[:length]
    f1 = f[1]
    zero = jnp.abs(f1) == 0
    f1_safe = jnp.where(zero, jnp.array(1.0, dtype=f.dtype), f1)
    g = jnp.zeros((length,), dtype=f.dtype)
    g = g.at[1].set(jnp.where(zero, jnp.array(0.0, dtype=f.dtype), 1.0 / f1_safe))
    for k in range(2, length):
        s = jnp.zeros((), dtype=f.dtype)
        for j in range(1, k):
            s = s + f[j + 1] * g[k - j]
        g = g.at[k].set(-s / f1_safe)
    return jnp.where(zero, jnp.zeros((length,), dtype=f.dtype), g)


def series_pow_scalar(f: jax.Array, p: jax.Array, length: int) -> jax.Array:
    # Use exp(p * log f)
    logf = series_log(f, length)
    return series_exp(logf * p, length)


def series_exp(f: jax.Array, length: int) -> jax.Array:
    f = f[:length]
    out = jnp.zeros((length,), dtype=f.dtype)
    out = out.at[0].set(jnp.exp(f[0]))
    for k in range(1, length):
        s = jnp.zeros((), dtype=f.dtype)
        for j in range(1, k + 1):
            s = s + jnp.float64(j) * f[j] * out[k - j]
        out = out.at[k].set(s / jnp.float64(k))
    return out


def series_log(f: jax.Array, length: int) -> jax.Array:
    f = f[:length]
    out = jnp.zeros((length,), dtype=f.dtype)
    out = out.at[0].set(jnp.log(f[0]))
    for k in range(1, length):
        s = f[k]
        for j in range(1, k):
            s = s - jnp.float64(j) * out[j] * f[k - j]
        out = out.at[k].set(s / (jnp.float64(k) * f[0]))
    return out


def series_sin_cos(f: jax.Array, length: int) -> tuple[jax.Array, jax.Array]:
    f = f[:length]
    s = jnp.zeros((length,), dtype=f.dtype)
    c = jnp.zeros((length,), dtype=f.dtype)
    s = s.at[0].set(jnp.sin(f[0]))
    c = c.at[0].set(jnp.cos(f[0]))
    for k in range(1, length):
        s_sum = jnp.zeros((), dtype=f.dtype)
        c_sum = jnp.zeros((), dtype=f.dtype)
        for j in range(1, k + 1):
            s_sum = s_sum + jnp.float64(j) * f[j] * c[k - j]
            c_sum = c_sum - jnp.float64(j) * f[j] * s[k - j]
        s = s.at[k].set(s_sum / jnp.float64(k))
        c = c.at[k].set(c_sum / jnp.float64(k))
    return s, c


def series_tan(f: jax.Array, length: int) -> jax.Array:
    s, c = series_sin_cos(f, length)
    return series_div(s, c, length)


def series_sinh_cosh(f: jax.Array, length: int) -> tuple[jax.Array, jax.Array]:
    f = f[:length]
    s = jnp.zeros((length,), dtype=f.dtype)
    c = jnp.zeros((length,), dtype=f.dtype)
    s = s.at[0].set(jnp.sinh(f[0]))
    c = c.at[0].set(jnp.cosh(f[0]))
    for k in range(1, length):
        s_sum = jnp.zeros((), dtype=f.dtype)
        c_sum = jnp.zeros((), dtype=f.dtype)
        for j in range(1, k + 1):
            s_sum = s_sum + jnp.float64(j) * f[j] * c[k - j]
            c_sum = c_sum + jnp.float64(j) * f[j] * s[k - j]
        s = s.at[k].set(s_sum / jnp.float64(k))
        c = c.at[k].set(c_sum / jnp.float64(k))
    return s, c


def series_sqrt(f: jax.Array, length: int) -> jax.Array:
    f = f[:length]
    out = jnp.zeros((length,), dtype=f.dtype)
    out = out.at[0].set(jnp.sqrt(f[0]))
    for k in range(1, length):
        s = f[k]
        for j in range(1, k):
            s = s - out[j] * out[k - j]
        out = out.at[k].set(s / (2.0 * out[0]))
    return out


def series_rsqrt(f: jax.Array, length: int) -> jax.Array:
    return series_inv(series_sqrt(f, length), length)


# Interval/box series operations for rigorous bounds

from . import arb_core


def _acb_scale_real(x: jax.Array, s: jax.Array) -> jax.Array:
    box = acb_core.acb_box(di.interval(s, s), di.interval(0.0, 0.0))
    return acb_core.acb_mul(x, box)



def interval_exp(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    return di.interval(jnp.exp(x[0]), jnp.exp(x[1]))

def interval_log(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    bad = x[0] <= 0.0
    lo = jnp.where(bad, -jnp.inf, jnp.log(x[0]))
    hi = jnp.where(bad, jnp.inf, jnp.log(x[1]))
    return di.interval(lo, hi)

def interval_sqrt(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    bad = x[0] < 0.0
    lo = jnp.where(bad, -jnp.inf, jnp.sqrt(x[0]))
    hi = jnp.where(bad, jnp.inf, jnp.sqrt(x[1]))
    return di.interval(lo, hi)

def series_mul_interval(a: jax.Array, b: jax.Array, length: int) -> jax.Array:
    out = jnp.zeros((length, 2), dtype=jnp.float64)
    for k in range(length):
        s = di.interval(0.0, 0.0)
        for j in range(k + 1):
            s = di.fast_add(s, di.fast_mul(a[j], b[k - j]))
        out = out.at[k].set(s)
    return out

def series_inv_interval(a: jax.Array, length: int) -> jax.Array:
    out = jnp.zeros((length, 2), dtype=jnp.float64)
    out = out.at[0].set(di.fast_div(di.interval(1.0, 1.0), a[0]))
    for k in range(1, length):
        s = di.interval(0.0, 0.0)
        for j in range(1, k + 1):
            s = di.fast_add(s, di.fast_mul(a[j], out[k - j]))
        out = out.at[k].set(di.fast_div(di.fast_neg(s), a[0]))
    return out

def series_div_interval(a: jax.Array, b: jax.Array, length: int) -> jax.Array:
    return series_mul_interval(a, series_inv_interval(b, length), length)

def series_exp_interval(f: jax.Array, length: int) -> jax.Array:
    out = jnp.zeros((length, 2), dtype=jnp.float64)
    out = out.at[0].set(interval_exp(f[0]))
    for k in range(1, length):
        s = di.interval(0.0, 0.0)
        for j in range(1, k + 1):
            s = di.fast_add(s, di.fast_mul(di.interval(j, j), di.fast_mul(f[j], out[k - j])))
        out = out.at[k].set(di.fast_div(s, di.interval(k, k)))
    return out

def series_log_interval(f: jax.Array, length: int) -> jax.Array:
    out = jnp.zeros((length, 2), dtype=jnp.float64)
    out = out.at[0].set(interval_log(f[0]))
    for k in range(1, length):
        s = f[k]
        for j in range(1, k):
            s = di.fast_sub(s, di.fast_mul(di.interval(j, j), di.fast_mul(out[j], f[k - j])))
        out = out.at[k].set(di.fast_div(s, di.fast_mul(di.interval(k, k), f[0])))
    return out

def series_sin_cos_interval(f: jax.Array, length: int) -> tuple[jax.Array, jax.Array]:
    s = jnp.zeros((length, 2), dtype=jnp.float64)
    c = jnp.zeros((length, 2), dtype=jnp.float64)
    s = s.at[0].set(arb_core.arb_sin(f[0]))
    c = c.at[0].set(arb_core.arb_cos(f[0]))
    for k in range(1, length):
        s_sum = di.interval(0.0, 0.0)
        c_sum = di.interval(0.0, 0.0)
        for j in range(1, k + 1):
            s_sum = di.fast_add(s_sum, di.fast_mul(di.interval(j, j), di.fast_mul(f[j], c[k - j])))
            c_sum = di.fast_sub(c_sum, di.fast_mul(di.interval(j, j), di.fast_mul(f[j], s[k - j])))
        s = s.at[k].set(di.fast_div(s_sum, di.interval(k, k)))
        c = c.at[k].set(di.fast_div(c_sum, di.interval(k, k)))
    return s, c

def series_sinh_cosh_interval(f: jax.Array, length: int) -> tuple[jax.Array, jax.Array]:
    s = jnp.zeros((length, 2), dtype=jnp.float64)
    c = jnp.zeros((length, 2), dtype=jnp.float64)
    s = s.at[0].set(arb_core.arb_sinh(f[0]))
    c = c.at[0].set(arb_core.arb_cosh(f[0]))
    for k in range(1, length):
        s_sum = di.interval(0.0, 0.0)
        c_sum = di.interval(0.0, 0.0)
        for j in range(1, k + 1):
            s_sum = di.fast_add(s_sum, di.fast_mul(di.interval(j, j), di.fast_mul(f[j], c[k - j])))
            c_sum = di.fast_add(c_sum, di.fast_mul(di.interval(j, j), di.fast_mul(f[j], s[k - j])))
        s = s.at[k].set(di.fast_div(s_sum, di.interval(k, k)))
        c = c.at[k].set(di.fast_div(c_sum, di.interval(k, k)))
    return s, c

def series_sqrt_interval(f: jax.Array, length: int) -> jax.Array:
    out = jnp.zeros((length, 2), dtype=jnp.float64)
    out = out.at[0].set(interval_sqrt(f[0]))
    for k in range(1, length):
        s = f[k]
        for j in range(1, k):
            s = di.fast_sub(s, di.fast_mul(out[j], out[k - j]))
        out = out.at[k].set(di.fast_div(s, di.fast_mul(di.interval(2.0, 2.0), out[0])))
    return out

def series_rsqrt_interval(f: jax.Array, length: int) -> jax.Array:
    return series_inv_interval(series_sqrt_interval(f, length), length)

def series_tan_interval(f: jax.Array, length: int) -> jax.Array:
    s, c = series_sin_cos_interval(f, length)
    return series_div_interval(s, c, length)

def series_mul_box(a: jax.Array, b: jax.Array, length: int) -> jax.Array:
    out = jnp.zeros((length, 4), dtype=jnp.float64)
    for k in range(length):
        s = acb_core.acb_zero()
        for j in range(k + 1):
            s = acb_core.acb_add(s, acb_core.acb_mul(a[j], b[k - j]))
        out = out.at[k].set(s)
    return out

def series_inv_box(a: jax.Array, length: int) -> jax.Array:
    out = jnp.zeros((length, 4), dtype=jnp.float64)
    out = out.at[0].set(acb_core.acb_div(acb_core.acb_one(), a[0]))
    for k in range(1, length):
        s = acb_core.acb_zero()
        for j in range(1, k + 1):
            s = acb_core.acb_add(s, acb_core.acb_mul(a[j], out[k - j]))
        out = out.at[k].set(acb_core.acb_div(acb_core.acb_neg(s), a[0]))
    return out

def series_div_box(a: jax.Array, b: jax.Array, length: int) -> jax.Array:
    return series_mul_box(a, series_inv_box(b, length), length)

def series_exp_box(f: jax.Array, length: int) -> jax.Array:
    out = jnp.zeros((length, 4), dtype=jnp.float64)
    out = out.at[0].set(acb_core.acb_exp(f[0]))
    for k in range(1, length):
        s = acb_core.acb_zero()
        for j in range(1, k + 1):
            s = acb_core.acb_add(s, acb_core.acb_mul(_acb_scale_real(f[j], jnp.float64(j)), out[k - j]))
        out = out.at[k].set(_acb_scale_real(s, 1.0 / jnp.float64(k)))
    return out

def series_log_box(f: jax.Array, length: int) -> jax.Array:
    out = jnp.zeros((length, 4), dtype=jnp.float64)
    out = out.at[0].set(acb_core.acb_log(f[0]))
    for k in range(1, length):
        s = f[k]
        for j in range(1, k):
            s = acb_core.acb_sub(s, acb_core.acb_mul(_acb_scale_real(out[j], jnp.float64(j)), f[k - j]))
        denom = acb_core.acb_mul(_acb_scale_real(acb_core.acb_one(), jnp.float64(k)), f[0])
        out = out.at[k].set(acb_core.acb_div(s, denom))
    return out

def series_sin_cos_box(f: jax.Array, length: int) -> tuple[jax.Array, jax.Array]:
    s = jnp.zeros((length, 4), dtype=jnp.float64)
    c = jnp.zeros((length, 4), dtype=jnp.float64)
    s = s.at[0].set(acb_core.acb_sin(f[0]))
    c = c.at[0].set(acb_core.acb_cos(f[0]))
    for k in range(1, length):
        s_sum = acb_core.acb_zero()
        c_sum = acb_core.acb_zero()
        for j in range(1, k + 1):
            s_sum = acb_core.acb_add(s_sum, acb_core.acb_mul(_acb_scale_real(f[j], jnp.float64(j)), c[k - j]))
            c_sum = acb_core.acb_sub(c_sum, acb_core.acb_mul(_acb_scale_real(f[j], jnp.float64(j)), s[k - j]))
        s = s.at[k].set(_acb_scale_real(s_sum, 1.0 / jnp.float64(k)))
        c = c.at[k].set(_acb_scale_real(c_sum, 1.0 / jnp.float64(k)))
    return s, c

def series_sinh_cosh_box(f: jax.Array, length: int) -> tuple[jax.Array, jax.Array]:
    s = jnp.zeros((length, 4), dtype=jnp.float64)
    c = jnp.zeros((length, 4), dtype=jnp.float64)
    s = s.at[0].set(acb_core.acb_sinh(f[0]))
    c = c.at[0].set(acb_core.acb_cosh(f[0]))
    for k in range(1, length):
        s_sum = acb_core.acb_zero()
        c_sum = acb_core.acb_zero()
        for j in range(1, k + 1):
            s_sum = acb_core.acb_add(s_sum, acb_core.acb_mul(_acb_scale_real(f[j], jnp.float64(j)), c[k - j]))
            c_sum = acb_core.acb_add(c_sum, acb_core.acb_mul(_acb_scale_real(f[j], jnp.float64(j)), s[k - j]))
        s = s.at[k].set(_acb_scale_real(s_sum, 1.0 / jnp.float64(k)))
        c = c.at[k].set(_acb_scale_real(c_sum, 1.0 / jnp.float64(k)))
    return s, c

def series_sqrt_box(f: jax.Array, length: int) -> jax.Array:
    out = jnp.zeros((length, 4), dtype=jnp.float64)
    out = out.at[0].set(acb_core.acb_sqrt(f[0]))
    for k in range(1, length):
        s = f[k]
        for j in range(1, k):
            s = acb_core.acb_sub(s, acb_core.acb_mul(out[j], out[k - j]))
        denom = _acb_scale_real(out[0], 2.0)
        out = out.at[k].set(acb_core.acb_div(s, denom))
    return out

def series_rsqrt_box(f: jax.Array, length: int) -> jax.Array:
    return series_inv_box(series_sqrt_box(f, length), length)

def series_tan_box(f: jax.Array, length: int) -> jax.Array:
    s, c = series_sin_cos_box(f, length)
    return series_div_box(s, c, length)



def series_compose_interval(f: jax.Array, g: jax.Array, length: int) -> jax.Array:
    out = jnp.zeros((length, 2), dtype=jnp.float64)
    out = out.at[0].set(f[0])
    g_pow = jnp.zeros((length, 2), dtype=jnp.float64)
    g_pow = g_pow.at[0].set(di.interval(1.0, 1.0))
    for k in range(1, length):
        g_pow = series_mul_interval(g_pow, g, length)
        out = out.at[:].set(di.as_interval(out))
        out = out.at[k].set(di.fast_add(out[k], di.fast_mul(f[k], g_pow[0])))
        for i in range(k):
            out = out.at[k].set(di.fast_add(out[k], di.fast_mul(f[k], g_pow[i])))
    return out

def series_compose_box(f: jax.Array, g: jax.Array, length: int) -> jax.Array:
    out = jnp.zeros((length, 4), dtype=jnp.float64)
    out = out.at[0].set(f[0])
    g_pow = jnp.zeros((length, 4), dtype=jnp.float64)
    g_pow = g_pow.at[0].set(acb_core.acb_one())
    for k in range(1, length):
        g_pow = series_mul_box(g_pow, g, length)
        out = out.at[k].set(acb_core.acb_add(out[k], acb_core.acb_mul(f[k], g_pow[0])))
        for i in range(k):
            out = out.at[k].set(acb_core.acb_add(out[k], acb_core.acb_mul(f[k], g_pow[i])))
    return out

def series_scale_interval(f: jax.Array, c: jax.Array, length: int) -> jax.Array:
    out = jnp.zeros((length, 2), dtype=jnp.float64)
    for k in range(length):
        out = out.at[k].set(di.fast_mul(f[k], c))
    return out

def series_scale_box(f: jax.Array, c: jax.Array, length: int) -> jax.Array:
    out = jnp.zeros((length, 4), dtype=jnp.float64)
    for k in range(length):
        out = out.at[k].set(acb_core.acb_mul(f[k], c))
    return out

def series_pow_interval(f: jax.Array, p: jax.Array, length: int) -> jax.Array:
    logf = series_log_interval(f, length)
    scaled = series_scale_interval(logf, p, length)
    return series_exp_interval(scaled, length)

def series_pow_box(f: jax.Array, p: jax.Array, length: int) -> jax.Array:
    logf = series_log_box(f, length)
    scaled = series_scale_box(logf, p, length)
    return series_exp_box(scaled, length)

