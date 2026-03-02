from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import double_interval as di
from . import elementary as el

jax.config.update("jax_enable_x64", True)

_PI = el.PI
_HALF_PI = el.HALF_PI
_TWO_PI = el.TWO_PI


def _full_interval_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.float64)
    return di.interval(-jnp.inf * t, jnp.inf * t)


@jax.jit
def arb_exp(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[..., 0]
    b = x[..., 1]
    finite = jnp.isfinite(a) & jnp.isfinite(b)
    out = di.interval(di._below(jnp.exp(a)), di._above(jnp.exp(b)))
    return jnp.where(finite[..., None], out, _full_interval_like(x))


@jax.jit
def arb_log(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[..., 0]
    b = x[..., 1]
    valid = (a > 0.0) & jnp.isfinite(a) & jnp.isfinite(b)
    out = di.interval(di._below(jnp.log(a)), di._above(jnp.log(b)))
    return jnp.where(valid[..., None], out, _full_interval_like(x))


@jax.jit
def arb_sqrt(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[..., 0]
    b = x[..., 1]
    valid = (b >= 0.0) & jnp.isfinite(a) & jnp.isfinite(b)
    aa = jnp.maximum(a, 0.0)
    bb = jnp.maximum(b, 0.0)
    out = di.interval(di._below(jnp.sqrt(aa)), di._above(jnp.sqrt(bb)))
    return jnp.where(valid[..., None], out, _full_interval_like(x))


@jax.jit
def arb_sin(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[..., 0]
    b = x[..., 1]
    finite = jnp.isfinite(a) & jnp.isfinite(b)
    width = b - a
    full_cycle = width >= _TWO_PI

    sa = jnp.sin(a)
    sb = jnp.sin(b)
    lo = jnp.minimum(sa, sb)
    hi = jnp.maximum(sa, sb)

    kmax_lo = jnp.ceil((a - 0.5 * _PI) / _TWO_PI)
    kmax_hi = jnp.floor((b - 0.5 * _PI) / _TWO_PI)
    has_max = kmax_lo <= kmax_hi

    kmin_lo = jnp.ceil((a - 1.5 * _PI) / _TWO_PI)
    kmin_hi = jnp.floor((b - 1.5 * _PI) / _TWO_PI)
    has_min = kmin_lo <= kmin_hi

    hi = jnp.where(has_max, 1.0, hi)
    lo = jnp.where(has_min, -1.0, lo)
    lo = jnp.where(full_cycle, -1.0, lo)
    hi = jnp.where(full_cycle, 1.0, hi)

    out = di.interval(di._below(lo), di._above(hi))
    return jnp.where(finite[..., None], out, _full_interval_like(x))


@jax.jit
def arb_cos(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[..., 0]
    b = x[..., 1]
    finite = jnp.isfinite(a) & jnp.isfinite(b)
    width = b - a
    full_cycle = width >= _TWO_PI

    ca = jnp.cos(a)
    cb = jnp.cos(b)
    lo = jnp.minimum(ca, cb)
    hi = jnp.maximum(ca, cb)

    kmax_lo = jnp.ceil(a / _TWO_PI)
    kmax_hi = jnp.floor(b / _TWO_PI)
    has_max = kmax_lo <= kmax_hi

    kmin_lo = jnp.ceil((a - _PI) / _TWO_PI)
    kmin_hi = jnp.floor((b - _PI) / _TWO_PI)
    has_min = kmin_lo <= kmin_hi

    hi = jnp.where(has_max, 1.0, hi)
    lo = jnp.where(has_min, -1.0, lo)
    lo = jnp.where(full_cycle, -1.0, lo)
    hi = jnp.where(full_cycle, 1.0, hi)

    out = di.interval(di._below(lo), di._above(hi))
    return jnp.where(finite[..., None], out, _full_interval_like(x))


@jax.jit
def arb_tan(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[..., 0]
    b = x[..., 1]
    finite = jnp.isfinite(a) & jnp.isfinite(b)

    kmin = jnp.ceil((a - _HALF_PI) / _PI)
    kmax = jnp.floor((b - _HALF_PI) / _PI)
    has_pole = kmin <= kmax

    ta = jnp.tan(a)
    tb = jnp.tan(b)
    lo = jnp.minimum(ta, tb)
    hi = jnp.maximum(ta, tb)
    out = di.interval(di._below(lo), di._above(hi))
    out = jnp.where(has_pole[..., None], _full_interval_like(x), out)
    return jnp.where(finite[..., None], out, _full_interval_like(x))


@jax.jit
def arb_sinh(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[..., 0]
    b = x[..., 1]
    finite = jnp.isfinite(a) & jnp.isfinite(b)
    out = di.interval(di._below(jnp.sinh(a)), di._above(jnp.sinh(b)))
    return jnp.where(finite[..., None], out, _full_interval_like(x))


@jax.jit
def arb_cosh(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[..., 0]
    b = x[..., 1]
    finite = jnp.isfinite(a) & jnp.isfinite(b)

    ca = jnp.cosh(a)
    cb = jnp.cosh(b)
    lo = jnp.minimum(ca, cb)
    lo = jnp.where((a <= 0.0) & (b >= 0.0), 1.0, lo)
    hi = jnp.maximum(ca, cb)

    out = di.interval(di._below(lo), di._above(hi))
    return jnp.where(finite[..., None], out, _full_interval_like(x))


@jax.jit
def arb_tanh(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[..., 0]
    b = x[..., 1]
    finite = jnp.isfinite(a) & jnp.isfinite(b)
    out = di.interval(di._below(jnp.tanh(a)), di._above(jnp.tanh(b)))
    return jnp.where(finite[..., None], out, _full_interval_like(x))


@jax.jit
def arb_abs(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[..., 0]
    b = x[..., 1]
    finite = jnp.isfinite(a) & jnp.isfinite(b)
    lo = jnp.where((a <= 0.0) & (b >= 0.0), 0.0, jnp.minimum(jnp.abs(a), jnp.abs(b)))
    hi = jnp.maximum(jnp.abs(a), jnp.abs(b))
    out = di.interval(di._below(lo), di._above(hi))
    return jnp.where(finite[..., None], out, _full_interval_like(x))


@jax.jit
def arb_add(x: jax.Array, y: jax.Array) -> jax.Array:
    return di.fast_add(di.as_interval(x), di.as_interval(y))


@jax.jit
def arb_sub(x: jax.Array, y: jax.Array) -> jax.Array:
    return di.fast_sub(di.as_interval(x), di.as_interval(y))


@jax.jit
def arb_mul(x: jax.Array, y: jax.Array) -> jax.Array:
    return di.fast_mul(di.as_interval(x), di.as_interval(y))


@jax.jit
def arb_div(x: jax.Array, y: jax.Array) -> jax.Array:
    return di.fast_div(di.as_interval(x), di.as_interval(y))


@jax.jit
def arb_inv(x: jax.Array) -> jax.Array:
    one = di.interval(jnp.ones_like(di.lower(x)), jnp.ones_like(di.upper(x)))
    return di.fast_div(one, di.as_interval(x))


@jax.jit
def arb_fma(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    return di.fast_add(di.fast_mul(di.as_interval(x), di.as_interval(y)), di.as_interval(z))


@jax.jit
def arb_log1p(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[..., 0]
    b = x[..., 1]
    valid = (a > -1.0) & jnp.isfinite(a) & jnp.isfinite(b)
    out = di.interval(di._below(jnp.log1p(a)), di._above(jnp.log1p(b)))
    return jnp.where(valid[..., None], out, _full_interval_like(x))


@jax.jit
def arb_expm1(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[..., 0]
    b = x[..., 1]
    finite = jnp.isfinite(a) & jnp.isfinite(b)
    out = di.interval(di._below(jnp.expm1(a)), di._above(jnp.expm1(b)))
    return jnp.where(finite[..., None], out, _full_interval_like(x))


@jax.jit
def arb_sin_cos(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return arb_sin(x), arb_cos(x)


@jax.jit
def arb_sinh_cosh(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return arb_sinh(x), arb_cosh(x)


@jax.jit
def arb_sin_pi(x: jax.Array) -> jax.Array:
    return arb_sin(di.fast_mul(di.as_interval(x), di.interval(_PI, _PI)))


@jax.jit
def arb_cos_pi(x: jax.Array) -> jax.Array:
    return arb_cos(di.fast_mul(di.as_interval(x), di.interval(_PI, _PI)))


@jax.jit
def arb_tan_pi(x: jax.Array) -> jax.Array:
    return arb_tan(di.fast_mul(di.as_interval(x), di.interval(_PI, _PI)))


@jax.jit
def arb_sinc(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a, b = x[..., 0], x[..., 1]
    m = 0.5 * (a + b)
    v = jnp.stack(
        [
            jnp.where(jnp.abs(a) < 1e-15, 1.0, jnp.sin(a) / a),
            jnp.where(jnp.abs(b) < 1e-15, 1.0, jnp.sin(b) / b),
            jnp.where(jnp.abs(m) < 1e-15, 1.0, jnp.sin(m) / m),
            jnp.ones_like(a),
        ],
        axis=-1,
    )
    include_one = (a <= 0.0) & (b >= 0.0)
    lo = jnp.where(include_one, jnp.min(v, axis=-1), jnp.min(v[..., 0:3], axis=-1))
    hi = jnp.where(include_one, jnp.max(v, axis=-1), jnp.max(v[..., 0:3], axis=-1))
    out = di.interval(di._below(lo), di._above(hi))
    finite = jnp.isfinite(a) & jnp.isfinite(b)
    return jnp.where(finite[..., None], out, _full_interval_like(x))


@jax.jit
def arb_sinc_pi(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a, b = x[..., 0], x[..., 1]
    m = 0.5 * (a + b)

    def _s(v):
        t = _PI * v
        return jnp.where(jnp.abs(v) < 1e-15, 1.0, jnp.sin(t) / t)

    vals = jnp.stack([_s(a), _s(b), _s(m), jnp.ones_like(a)], axis=-1)
    include_one = (a <= 0.0) & (b >= 0.0)
    lo = jnp.where(include_one, jnp.min(vals, axis=-1), jnp.min(vals[..., 0:3], axis=-1))
    hi = jnp.where(include_one, jnp.max(vals, axis=-1), jnp.max(vals[..., 0:3], axis=-1))
    out = di.interval(di._below(lo), di._above(hi))
    finite = jnp.isfinite(a) & jnp.isfinite(b)
    return jnp.where(finite[..., None], out, _full_interval_like(x))


@jax.jit
def arb_asin(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[..., 0]
    b = x[..., 1]
    valid = (a >= -1.0) & (b <= 1.0) & jnp.isfinite(a) & jnp.isfinite(b)
    out = di.interval(di._below(jnp.arcsin(a)), di._above(jnp.arcsin(b)))
    return jnp.where(valid[..., None], out, _full_interval_like(x))


@jax.jit
def arb_acos(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[..., 0]
    b = x[..., 1]
    valid = (a >= -1.0) & (b <= 1.0) & jnp.isfinite(a) & jnp.isfinite(b)
    out = di.interval(di._below(jnp.arccos(b)), di._above(jnp.arccos(a)))
    return jnp.where(valid[..., None], out, _full_interval_like(x))


@jax.jit
def arb_atan(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[..., 0]
    b = x[..., 1]
    finite = jnp.isfinite(a) & jnp.isfinite(b)
    out = di.interval(di._below(jnp.arctan(a)), di._above(jnp.arctan(b)))
    return jnp.where(finite[..., None], out, _full_interval_like(x))


@jax.jit
def arb_asinh(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[..., 0]
    b = x[..., 1]
    finite = jnp.isfinite(a) & jnp.isfinite(b)
    out = di.interval(di._below(jnp.arcsinh(a)), di._above(jnp.arcsinh(b)))
    return jnp.where(finite[..., None], out, _full_interval_like(x))


@jax.jit
def arb_acosh(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[..., 0]
    b = x[..., 1]
    valid = (a >= 1.0) & jnp.isfinite(a) & jnp.isfinite(b)
    out = di.interval(di._below(jnp.arccosh(a)), di._above(jnp.arccosh(b)))
    return jnp.where(valid[..., None], out, _full_interval_like(x))


@jax.jit
def arb_atanh(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[..., 0]
    b = x[..., 1]
    valid = (a > -1.0) & (b < 1.0) & jnp.isfinite(a) & jnp.isfinite(b)
    out = di.interval(di._below(jnp.arctanh(a)), di._above(jnp.arctanh(b)))
    return jnp.where(valid[..., None], out, _full_interval_like(x))


@jax.jit
def arb_sign(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a = x[..., 0]
    b = x[..., 1]
    lo = jnp.where(b < 0.0, -1.0, jnp.where((a == 0.0) & (b == 0.0), 0.0, -1.0))
    hi = jnp.where(a > 0.0, 1.0, jnp.where((a == 0.0) & (b == 0.0), 0.0, 1.0))
    return di.interval(di._below(lo), di._above(hi))


@jax.jit
def arb_pow(x: jax.Array, y: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    y = di.as_interval(y)
    xa, xb = x[..., 0], x[..., 1]
    ya, yb = y[..., 0], y[..., 1]
    xm = 0.5 * (xa + xb)
    ym = 0.5 * (ya + yb)
    xv = jnp.stack([xa, xb, xm], axis=-1)
    yv = jnp.stack([ya, yb, ym], axis=-1)
    vv = jnp.power(xv[..., :, None], yv[..., None, :])
    vv = jnp.reshape(vv, vv.shape[:-2] + (9,))
    finite = jnp.isfinite(vv)
    any_finite = jnp.any(finite, axis=-1)
    vmin = jnp.min(jnp.where(finite, vv, jnp.inf), axis=-1)
    vmax = jnp.max(jnp.where(finite, vv, -jnp.inf), axis=-1)
    out = di.interval(di._below(vmin), di._above(vmax))
    return jnp.where(any_finite[..., None], out, _full_interval_like(x))


@partial(jax.jit, static_argnames=("n",))
def arb_pow_ui(x: jax.Array, n: int) -> jax.Array:
    x = di.as_interval(x)
    xa, xb = x[..., 0], x[..., 1]
    xm = 0.5 * (xa + xb)
    xv = jnp.stack([xa, xb, xm, jnp.zeros_like(xa)], axis=-1)
    include_zero = (xa <= 0.0) & (xb >= 0.0)
    vals = jnp.power(xv, jnp.float64(n))
    lo = jnp.where(include_zero, jnp.min(vals, axis=-1), jnp.min(vals[..., 0:3], axis=-1))
    hi = jnp.where(include_zero, jnp.max(vals, axis=-1), jnp.max(vals[..., 0:3], axis=-1))
    return di.interval(di._below(lo), di._above(hi))


@jax.jit
def arb_pow_fmpz(x: jax.Array, n: jax.Array) -> jax.Array:
    n_f = jnp.asarray(n, dtype=jnp.float64)
    return arb_pow(x, di.interval(n_f, n_f))


@jax.jit
def arb_pow_fmpq(x: jax.Array, p: jax.Array, q: jax.Array) -> jax.Array:
    pp = jnp.asarray(p, dtype=jnp.float64)
    qq = jnp.asarray(q, dtype=jnp.float64)
    return arb_pow(x, di.interval(pp / qq, pp / qq))


@partial(jax.jit, static_argnames=("k",))
def arb_root_ui(x: jax.Array, k: int) -> jax.Array:
    x = di.as_interval(x)
    a, b = x[..., 0], x[..., 1]
    kk = jnp.float64(k)
    odd = (k % 2) == 1
    if odd:
        ra = jnp.sign(a) * jnp.power(jnp.abs(a), 1.0 / kk)
        rb = jnp.sign(b) * jnp.power(jnp.abs(b), 1.0 / kk)
        lo = jnp.minimum(ra, rb)
        hi = jnp.maximum(ra, rb)
        return di.interval(di._below(lo), di._above(hi))
    valid = b >= 0.0
    aa = jnp.maximum(a, 0.0)
    bb = jnp.maximum(b, 0.0)
    out = di.interval(di._below(jnp.power(aa, 1.0 / kk)), di._above(jnp.power(bb, 1.0 / kk)))
    return jnp.where(valid[..., None], out, _full_interval_like(x))


@partial(jax.jit, static_argnames=("k",))
def arb_root(x: jax.Array, k: int) -> jax.Array:
    return arb_root_ui(x, k)


@jax.jit
def arb_cbrt(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    a, b = x[..., 0], x[..., 1]
    ra = jnp.sign(a) * jnp.power(jnp.abs(a), 1.0 / 3.0)
    rb = jnp.sign(b) * jnp.power(jnp.abs(b), 1.0 / 3.0)
    return di.interval(di._below(jnp.minimum(ra, rb)), di._above(jnp.maximum(ra, rb)))


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_exp_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_exp(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_log_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_log(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_sqrt_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_sqrt(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_sin_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_sin(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_cos_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_cos(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_tan_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_tan(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_sinh_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_sinh(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_cosh_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_cosh(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_tanh_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_tanh(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_abs_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_abs(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_add_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_add(x, y), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_sub_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_sub(x, y), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_mul_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mul(x, y), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_div_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_div(x, y), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_inv_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_inv(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_fma_prec(x: jax.Array, y: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_fma(x, y, z), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_log1p_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_log1p(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_expm1_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_expm1(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_sin_cos_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array]:
    s, c = arb_sin_cos(x)
    return di.round_interval_outward(s, prec_bits), di.round_interval_outward(c, prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_sinh_cosh_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array]:
    s, c = arb_sinh_cosh(x)
    return di.round_interval_outward(s, prec_bits), di.round_interval_outward(c, prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_sin_pi_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_sin_pi(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_cos_pi_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_cos_pi(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_tan_pi_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_tan_pi(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_sinc_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_sinc(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_sinc_pi_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_sinc_pi(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_asin_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_asin(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_acos_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_acos(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_atan_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_atan(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_asinh_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_asinh(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_acosh_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_acosh(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_atanh_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_atanh(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_sign_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_sign(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_pow_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_pow(x, y), prec_bits)


@partial(jax.jit, static_argnames=("n", "prec_bits"))
def arb_pow_ui_prec(x: jax.Array, n: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_pow_ui(x, n), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_pow_fmpz_prec(x: jax.Array, n: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_pow_fmpz(x, n), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_pow_fmpq_prec(x: jax.Array, p: jax.Array, q: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_pow_fmpq(x, p, q), prec_bits)


@partial(jax.jit, static_argnames=("k", "prec_bits"))
def arb_root_ui_prec(x: jax.Array, k: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_root_ui(x, k), prec_bits)


@partial(jax.jit, static_argnames=("k", "prec_bits"))
def arb_root_prec(x: jax.Array, k: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_root(x, k), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_cbrt_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_cbrt(x), prec_bits)


def arb_exp_batch(x: jax.Array) -> jax.Array:
    return arb_exp(di.as_interval(x))


def arb_log_batch(x: jax.Array) -> jax.Array:
    return arb_log(di.as_interval(x))


def arb_sqrt_batch(x: jax.Array) -> jax.Array:
    return arb_sqrt(di.as_interval(x))


def arb_sin_batch(x: jax.Array) -> jax.Array:
    return arb_sin(di.as_interval(x))


def arb_cos_batch(x: jax.Array) -> jax.Array:
    return arb_cos(di.as_interval(x))


def arb_tan_batch(x: jax.Array) -> jax.Array:
    return arb_tan(di.as_interval(x))


def arb_sinh_batch(x: jax.Array) -> jax.Array:
    return arb_sinh(di.as_interval(x))


def arb_cosh_batch(x: jax.Array) -> jax.Array:
    return arb_cosh(di.as_interval(x))


def arb_tanh_batch(x: jax.Array) -> jax.Array:
    return arb_tanh(di.as_interval(x))


def arb_abs_batch(x: jax.Array) -> jax.Array:
    return arb_abs(di.as_interval(x))


def arb_add_batch(x: jax.Array, y: jax.Array) -> jax.Array:
    return arb_add(di.as_interval(x), di.as_interval(y))


def arb_sub_batch(x: jax.Array, y: jax.Array) -> jax.Array:
    return arb_sub(di.as_interval(x), di.as_interval(y))


def arb_mul_batch(x: jax.Array, y: jax.Array) -> jax.Array:
    return arb_mul(di.as_interval(x), di.as_interval(y))


def arb_div_batch(x: jax.Array, y: jax.Array) -> jax.Array:
    return arb_div(di.as_interval(x), di.as_interval(y))


def arb_inv_batch(x: jax.Array) -> jax.Array:
    return arb_inv(di.as_interval(x))


def arb_fma_batch(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    return arb_fma(di.as_interval(x), di.as_interval(y), di.as_interval(z))


def arb_log1p_batch(x: jax.Array) -> jax.Array:
    return arb_log1p(di.as_interval(x))


def arb_expm1_batch(x: jax.Array) -> jax.Array:
    return arb_expm1(di.as_interval(x))


def arb_sin_cos_batch(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return arb_sin_cos(di.as_interval(x))


def arb_sinh_cosh_batch(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return arb_sinh_cosh(di.as_interval(x))


def arb_sin_pi_batch(x: jax.Array) -> jax.Array:
    return arb_sin_pi(di.as_interval(x))


def arb_cos_pi_batch(x: jax.Array) -> jax.Array:
    return arb_cos_pi(di.as_interval(x))


def arb_tan_pi_batch(x: jax.Array) -> jax.Array:
    return arb_tan_pi(di.as_interval(x))


def arb_sinc_batch(x: jax.Array) -> jax.Array:
    return arb_sinc(di.as_interval(x))


def arb_sinc_pi_batch(x: jax.Array) -> jax.Array:
    return arb_sinc_pi(di.as_interval(x))


def arb_asin_batch(x: jax.Array) -> jax.Array:
    return arb_asin(di.as_interval(x))


def arb_acos_batch(x: jax.Array) -> jax.Array:
    return arb_acos(di.as_interval(x))


def arb_atan_batch(x: jax.Array) -> jax.Array:
    return arb_atan(di.as_interval(x))


def arb_asinh_batch(x: jax.Array) -> jax.Array:
    return arb_asinh(di.as_interval(x))


def arb_acosh_batch(x: jax.Array) -> jax.Array:
    return arb_acosh(di.as_interval(x))


def arb_atanh_batch(x: jax.Array) -> jax.Array:
    return arb_atanh(di.as_interval(x))


def arb_sign_batch(x: jax.Array) -> jax.Array:
    return arb_sign(di.as_interval(x))


def arb_pow_batch(x: jax.Array, y: jax.Array) -> jax.Array:
    return arb_pow(di.as_interval(x), di.as_interval(y))


def arb_pow_ui_batch(x: jax.Array, n: int) -> jax.Array:
    return arb_pow_ui(di.as_interval(x), n)


def arb_pow_fmpz_batch(x: jax.Array, n: jax.Array) -> jax.Array:
    return arb_pow_fmpz(di.as_interval(x), n)


def arb_pow_fmpq_batch(x: jax.Array, p: jax.Array, q: jax.Array) -> jax.Array:
    return arb_pow_fmpq(di.as_interval(x), p, q)


def arb_root_ui_batch(x: jax.Array, k: int) -> jax.Array:
    return arb_root_ui(di.as_interval(x), k)


def arb_root_batch(x: jax.Array, k: int) -> jax.Array:
    return arb_root(di.as_interval(x), k)


def arb_cbrt_batch(x: jax.Array) -> jax.Array:
    return arb_cbrt(di.as_interval(x))


def arb_exp_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_exp_batch(x), prec_bits)


def arb_log_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_log_batch(x), prec_bits)


def arb_sqrt_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_sqrt_batch(x), prec_bits)


def arb_sin_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_sin_batch(x), prec_bits)


def arb_cos_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_cos_batch(x), prec_bits)


def arb_tan_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_tan_batch(x), prec_bits)


def arb_sinh_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_sinh_batch(x), prec_bits)


def arb_cosh_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_cosh_batch(x), prec_bits)


def arb_tanh_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_tanh_batch(x), prec_bits)


def arb_abs_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_abs_batch(x), prec_bits)


def arb_add_batch_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_add_batch(x, y), prec_bits)


def arb_sub_batch_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_sub_batch(x, y), prec_bits)


def arb_mul_batch_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_mul_batch(x, y), prec_bits)


def arb_div_batch_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_div_batch(x, y), prec_bits)


def arb_inv_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_inv_batch(x), prec_bits)


def arb_fma_batch_prec(x: jax.Array, y: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_fma_batch(x, y, z), prec_bits)


def arb_log1p_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_log1p_batch(x), prec_bits)


def arb_expm1_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_expm1_batch(x), prec_bits)


def arb_sin_cos_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array]:
    s, c = arb_sin_cos_batch(x)
    return di.round_interval_outward(s, prec_bits), di.round_interval_outward(c, prec_bits)


def arb_sinh_cosh_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> tuple[jax.Array, jax.Array]:
    s, c = arb_sinh_cosh_batch(x)
    return di.round_interval_outward(s, prec_bits), di.round_interval_outward(c, prec_bits)


def arb_sin_pi_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_sin_pi_batch(x), prec_bits)


def arb_cos_pi_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_cos_pi_batch(x), prec_bits)


def arb_tan_pi_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_tan_pi_batch(x), prec_bits)


def arb_sinc_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_sinc_batch(x), prec_bits)


def arb_sinc_pi_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_sinc_pi_batch(x), prec_bits)


def arb_asin_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_asin_batch(x), prec_bits)


def arb_acos_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_acos_batch(x), prec_bits)


def arb_atan_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_atan_batch(x), prec_bits)


def arb_asinh_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_asinh_batch(x), prec_bits)


def arb_acosh_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_acosh_batch(x), prec_bits)


def arb_atanh_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_atanh_batch(x), prec_bits)


def arb_sign_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_sign_batch(x), prec_bits)


def arb_pow_batch_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_pow_batch(x, y), prec_bits)


def arb_pow_ui_batch_prec(x: jax.Array, n: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_pow_ui_batch(x, n), prec_bits)


def arb_pow_fmpz_batch_prec(x: jax.Array, n: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_pow_fmpz_batch(x, n), prec_bits)


def arb_pow_fmpq_batch_prec(x: jax.Array, p: jax.Array, q: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_pow_fmpq_batch(x, p, q), prec_bits)


def arb_root_ui_batch_prec(x: jax.Array, k: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_root_ui_batch(x, k), prec_bits)


def arb_root_batch_prec(x: jax.Array, k: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_root_batch(x, k), prec_bits)


def arb_cbrt_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(arb_cbrt_batch(x), prec_bits)


arb_exp_batch_jit = jax.jit(arb_exp_batch)
arb_log_batch_jit = jax.jit(arb_log_batch)
arb_sqrt_batch_jit = jax.jit(arb_sqrt_batch)
arb_sin_batch_jit = jax.jit(arb_sin_batch)
arb_cos_batch_jit = jax.jit(arb_cos_batch)
arb_tan_batch_jit = jax.jit(arb_tan_batch)
arb_sinh_batch_jit = jax.jit(arb_sinh_batch)
arb_cosh_batch_jit = jax.jit(arb_cosh_batch)
arb_tanh_batch_jit = jax.jit(arb_tanh_batch)
arb_abs_batch_jit = jax.jit(arb_abs_batch)
arb_add_batch_jit = jax.jit(arb_add_batch)
arb_sub_batch_jit = jax.jit(arb_sub_batch)
arb_mul_batch_jit = jax.jit(arb_mul_batch)
arb_div_batch_jit = jax.jit(arb_div_batch)
arb_inv_batch_jit = jax.jit(arb_inv_batch)
arb_fma_batch_jit = jax.jit(arb_fma_batch)
arb_log1p_batch_jit = jax.jit(arb_log1p_batch)
arb_expm1_batch_jit = jax.jit(arb_expm1_batch)
arb_sin_cos_batch_jit = jax.jit(arb_sin_cos_batch)
arb_sinh_cosh_batch_jit = jax.jit(arb_sinh_cosh_batch)
arb_sin_pi_batch_jit = jax.jit(arb_sin_pi_batch)
arb_cos_pi_batch_jit = jax.jit(arb_cos_pi_batch)
arb_tan_pi_batch_jit = jax.jit(arb_tan_pi_batch)
arb_sinc_batch_jit = jax.jit(arb_sinc_batch)
arb_sinc_pi_batch_jit = jax.jit(arb_sinc_pi_batch)
arb_asin_batch_jit = jax.jit(arb_asin_batch)
arb_acos_batch_jit = jax.jit(arb_acos_batch)
arb_atan_batch_jit = jax.jit(arb_atan_batch)
arb_asinh_batch_jit = jax.jit(arb_asinh_batch)
arb_acosh_batch_jit = jax.jit(arb_acosh_batch)
arb_atanh_batch_jit = jax.jit(arb_atanh_batch)
arb_sign_batch_jit = jax.jit(arb_sign_batch)
arb_pow_batch_jit = jax.jit(arb_pow_batch)
arb_pow_ui_batch_jit = jax.jit(arb_pow_ui_batch, static_argnames=("n",))
arb_pow_fmpz_batch_jit = jax.jit(arb_pow_fmpz_batch)
arb_pow_fmpq_batch_jit = jax.jit(arb_pow_fmpq_batch)
arb_root_ui_batch_jit = jax.jit(arb_root_ui_batch, static_argnames=("k",))
arb_root_batch_jit = jax.jit(arb_root_batch, static_argnames=("k",))
arb_cbrt_batch_jit = jax.jit(arb_cbrt_batch)

arb_exp_batch_prec_jit = jax.jit(arb_exp_batch_prec, static_argnames=("prec_bits",))
arb_log_batch_prec_jit = jax.jit(arb_log_batch_prec, static_argnames=("prec_bits",))
arb_sqrt_batch_prec_jit = jax.jit(arb_sqrt_batch_prec, static_argnames=("prec_bits",))
arb_sin_batch_prec_jit = jax.jit(arb_sin_batch_prec, static_argnames=("prec_bits",))
arb_cos_batch_prec_jit = jax.jit(arb_cos_batch_prec, static_argnames=("prec_bits",))
arb_tan_batch_prec_jit = jax.jit(arb_tan_batch_prec, static_argnames=("prec_bits",))
arb_sinh_batch_prec_jit = jax.jit(arb_sinh_batch_prec, static_argnames=("prec_bits",))
arb_cosh_batch_prec_jit = jax.jit(arb_cosh_batch_prec, static_argnames=("prec_bits",))
arb_tanh_batch_prec_jit = jax.jit(arb_tanh_batch_prec, static_argnames=("prec_bits",))
arb_abs_batch_prec_jit = jax.jit(arb_abs_batch_prec, static_argnames=("prec_bits",))
arb_add_batch_prec_jit = jax.jit(arb_add_batch_prec, static_argnames=("prec_bits",))
arb_sub_batch_prec_jit = jax.jit(arb_sub_batch_prec, static_argnames=("prec_bits",))
arb_mul_batch_prec_jit = jax.jit(arb_mul_batch_prec, static_argnames=("prec_bits",))
arb_div_batch_prec_jit = jax.jit(arb_div_batch_prec, static_argnames=("prec_bits",))
arb_inv_batch_prec_jit = jax.jit(arb_inv_batch_prec, static_argnames=("prec_bits",))
arb_fma_batch_prec_jit = jax.jit(arb_fma_batch_prec, static_argnames=("prec_bits",))
arb_log1p_batch_prec_jit = jax.jit(arb_log1p_batch_prec, static_argnames=("prec_bits",))
arb_expm1_batch_prec_jit = jax.jit(arb_expm1_batch_prec, static_argnames=("prec_bits",))
arb_sin_cos_batch_prec_jit = jax.jit(arb_sin_cos_batch_prec, static_argnames=("prec_bits",))
arb_sinh_cosh_batch_prec_jit = jax.jit(arb_sinh_cosh_batch_prec, static_argnames=("prec_bits",))
arb_sin_pi_batch_prec_jit = jax.jit(arb_sin_pi_batch_prec, static_argnames=("prec_bits",))
arb_cos_pi_batch_prec_jit = jax.jit(arb_cos_pi_batch_prec, static_argnames=("prec_bits",))
arb_tan_pi_batch_prec_jit = jax.jit(arb_tan_pi_batch_prec, static_argnames=("prec_bits",))
arb_sinc_batch_prec_jit = jax.jit(arb_sinc_batch_prec, static_argnames=("prec_bits",))
arb_sinc_pi_batch_prec_jit = jax.jit(arb_sinc_pi_batch_prec, static_argnames=("prec_bits",))
arb_asin_batch_prec_jit = jax.jit(arb_asin_batch_prec, static_argnames=("prec_bits",))
arb_acos_batch_prec_jit = jax.jit(arb_acos_batch_prec, static_argnames=("prec_bits",))
arb_atan_batch_prec_jit = jax.jit(arb_atan_batch_prec, static_argnames=("prec_bits",))
arb_asinh_batch_prec_jit = jax.jit(arb_asinh_batch_prec, static_argnames=("prec_bits",))
arb_acosh_batch_prec_jit = jax.jit(arb_acosh_batch_prec, static_argnames=("prec_bits",))
arb_atanh_batch_prec_jit = jax.jit(arb_atanh_batch_prec, static_argnames=("prec_bits",))
arb_sign_batch_prec_jit = jax.jit(arb_sign_batch_prec, static_argnames=("prec_bits",))
arb_pow_batch_prec_jit = jax.jit(arb_pow_batch_prec, static_argnames=("prec_bits",))
arb_pow_ui_batch_prec_jit = jax.jit(arb_pow_ui_batch_prec, static_argnames=("n", "prec_bits"))
arb_pow_fmpz_batch_prec_jit = jax.jit(arb_pow_fmpz_batch_prec, static_argnames=("prec_bits",))
arb_pow_fmpq_batch_prec_jit = jax.jit(arb_pow_fmpq_batch_prec, static_argnames=("prec_bits",))
arb_root_ui_batch_prec_jit = jax.jit(arb_root_ui_batch_prec, static_argnames=("k", "prec_bits"))
arb_root_batch_prec_jit = jax.jit(arb_root_batch_prec, static_argnames=("k", "prec_bits"))
arb_cbrt_batch_prec_jit = jax.jit(arb_cbrt_batch_prec, static_argnames=("prec_bits",))


__all__ = [
    "arb_exp",
    "arb_log",
    "arb_sqrt",
    "arb_sin",
    "arb_cos",
    "arb_tan",
    "arb_sinh",
    "arb_cosh",
    "arb_tanh",
    "arb_abs",
    "arb_add",
    "arb_sub",
    "arb_mul",
    "arb_div",
    "arb_inv",
    "arb_fma",
    "arb_log1p",
    "arb_expm1",
    "arb_sin_cos",
    "arb_exp_prec",
    "arb_log_prec",
    "arb_sqrt_prec",
    "arb_sin_prec",
    "arb_cos_prec",
    "arb_tan_prec",
    "arb_sinh_prec",
    "arb_cosh_prec",
    "arb_tanh_prec",
    "arb_abs_prec",
    "arb_add_prec",
    "arb_sub_prec",
    "arb_mul_prec",
    "arb_div_prec",
    "arb_inv_prec",
    "arb_fma_prec",
    "arb_log1p_prec",
    "arb_expm1_prec",
    "arb_sin_cos_prec",
    "arb_exp_batch",
    "arb_log_batch",
    "arb_sqrt_batch",
    "arb_sin_batch",
    "arb_cos_batch",
    "arb_tan_batch",
    "arb_sinh_batch",
    "arb_cosh_batch",
    "arb_tanh_batch",
    "arb_abs_batch",
    "arb_add_batch",
    "arb_sub_batch",
    "arb_mul_batch",
    "arb_div_batch",
    "arb_inv_batch",
    "arb_fma_batch",
    "arb_log1p_batch",
    "arb_expm1_batch",
    "arb_sin_cos_batch",
    "arb_exp_batch_prec",
    "arb_log_batch_prec",
    "arb_sqrt_batch_prec",
    "arb_sin_batch_prec",
    "arb_cos_batch_prec",
    "arb_tan_batch_prec",
    "arb_sinh_batch_prec",
    "arb_cosh_batch_prec",
    "arb_tanh_batch_prec",
    "arb_abs_batch_prec",
    "arb_add_batch_prec",
    "arb_sub_batch_prec",
    "arb_mul_batch_prec",
    "arb_div_batch_prec",
    "arb_inv_batch_prec",
    "arb_fma_batch_prec",
    "arb_log1p_batch_prec",
    "arb_expm1_batch_prec",
    "arb_sin_cos_batch_prec",
    "arb_exp_batch_jit",
    "arb_log_batch_jit",
    "arb_sqrt_batch_jit",
    "arb_sin_batch_jit",
    "arb_cos_batch_jit",
    "arb_tan_batch_jit",
    "arb_sinh_batch_jit",
    "arb_cosh_batch_jit",
    "arb_tanh_batch_jit",
    "arb_abs_batch_jit",
    "arb_add_batch_jit",
    "arb_sub_batch_jit",
    "arb_mul_batch_jit",
    "arb_div_batch_jit",
    "arb_inv_batch_jit",
    "arb_fma_batch_jit",
    "arb_log1p_batch_jit",
    "arb_expm1_batch_jit",
    "arb_sin_cos_batch_jit",
    "arb_exp_batch_prec_jit",
    "arb_log_batch_prec_jit",
    "arb_sqrt_batch_prec_jit",
    "arb_sin_batch_prec_jit",
    "arb_cos_batch_prec_jit",
    "arb_tan_batch_prec_jit",
    "arb_sinh_batch_prec_jit",
    "arb_cosh_batch_prec_jit",
    "arb_tanh_batch_prec_jit",
    "arb_abs_batch_prec_jit",
    "arb_add_batch_prec_jit",
    "arb_sub_batch_prec_jit",
    "arb_mul_batch_prec_jit",
    "arb_div_batch_prec_jit",
    "arb_inv_batch_prec_jit",
    "arb_fma_batch_prec_jit",
    "arb_log1p_batch_prec_jit",
    "arb_expm1_batch_prec_jit",
    "arb_sin_cos_batch_prec_jit",
]

__all__.extend(
    [
        "arb_sinh_cosh",
        "arb_sin_pi",
        "arb_cos_pi",
        "arb_tan_pi",
        "arb_sinc",
        "arb_sinc_pi",
        "arb_asin",
        "arb_acos",
        "arb_atan",
        "arb_asinh",
        "arb_acosh",
        "arb_atanh",
        "arb_sign",
        "arb_pow",
        "arb_pow_ui",
        "arb_pow_fmpz",
        "arb_pow_fmpq",
        "arb_root_ui",
        "arb_root",
        "arb_cbrt",
        "arb_sinh_cosh_prec",
        "arb_sin_pi_prec",
        "arb_cos_pi_prec",
        "arb_tan_pi_prec",
        "arb_sinc_prec",
        "arb_sinc_pi_prec",
        "arb_asin_prec",
        "arb_acos_prec",
        "arb_atan_prec",
        "arb_asinh_prec",
        "arb_acosh_prec",
        "arb_atanh_prec",
        "arb_sign_prec",
        "arb_pow_prec",
        "arb_pow_ui_prec",
        "arb_pow_fmpz_prec",
        "arb_pow_fmpq_prec",
        "arb_root_ui_prec",
        "arb_root_prec",
        "arb_cbrt_prec",
        "arb_sinh_cosh_batch",
        "arb_sin_pi_batch",
        "arb_cos_pi_batch",
        "arb_tan_pi_batch",
        "arb_sinc_batch",
        "arb_sinc_pi_batch",
        "arb_asin_batch",
        "arb_acos_batch",
        "arb_atan_batch",
        "arb_asinh_batch",
        "arb_acosh_batch",
        "arb_atanh_batch",
        "arb_sign_batch",
        "arb_pow_batch",
        "arb_pow_ui_batch",
        "arb_pow_fmpz_batch",
        "arb_pow_fmpq_batch",
        "arb_root_ui_batch",
        "arb_root_batch",
        "arb_cbrt_batch",
        "arb_sinh_cosh_batch_prec",
        "arb_sin_pi_batch_prec",
        "arb_cos_pi_batch_prec",
        "arb_tan_pi_batch_prec",
        "arb_sinc_batch_prec",
        "arb_sinc_pi_batch_prec",
        "arb_asin_batch_prec",
        "arb_acos_batch_prec",
        "arb_atan_batch_prec",
        "arb_asinh_batch_prec",
        "arb_acosh_batch_prec",
        "arb_atanh_batch_prec",
        "arb_sign_batch_prec",
        "arb_pow_batch_prec",
        "arb_pow_ui_batch_prec",
        "arb_pow_fmpz_batch_prec",
        "arb_pow_fmpq_batch_prec",
        "arb_root_ui_batch_prec",
        "arb_root_batch_prec",
        "arb_cbrt_batch_prec",
        "arb_sinh_cosh_batch_jit",
        "arb_sin_pi_batch_jit",
        "arb_cos_pi_batch_jit",
        "arb_tan_pi_batch_jit",
        "arb_sinc_batch_jit",
        "arb_sinc_pi_batch_jit",
        "arb_asin_batch_jit",
        "arb_acos_batch_jit",
        "arb_atan_batch_jit",
        "arb_asinh_batch_jit",
        "arb_acosh_batch_jit",
        "arb_atanh_batch_jit",
        "arb_sign_batch_jit",
        "arb_pow_batch_jit",
        "arb_pow_ui_batch_jit",
        "arb_pow_fmpz_batch_jit",
        "arb_pow_fmpq_batch_jit",
        "arb_root_ui_batch_jit",
        "arb_root_batch_jit",
        "arb_cbrt_batch_jit",
        "arb_sinh_cosh_batch_prec_jit",
        "arb_sin_pi_batch_prec_jit",
        "arb_cos_pi_batch_prec_jit",
        "arb_tan_pi_batch_prec_jit",
        "arb_sinc_batch_prec_jit",
        "arb_sinc_pi_batch_prec_jit",
        "arb_asin_batch_prec_jit",
        "arb_acos_batch_prec_jit",
        "arb_atan_batch_prec_jit",
        "arb_asinh_batch_prec_jit",
        "arb_acosh_batch_prec_jit",
        "arb_atanh_batch_prec_jit",
        "arb_sign_batch_prec_jit",
        "arb_pow_batch_prec_jit",
        "arb_pow_ui_batch_prec_jit",
        "arb_pow_fmpz_batch_prec_jit",
        "arb_pow_fmpq_batch_prec_jit",
        "arb_root_ui_batch_prec_jit",
        "arb_root_batch_prec_jit",
        "arb_cbrt_batch_prec_jit",
    ]
)
