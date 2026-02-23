from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import double_interval as di

jax.config.update("jax_enable_x64", True)

_PI = jnp.float64(3.14159265358979323846)
_HALF_PI = jnp.float64(1.57079632679489661923)
_TWO_PI = jnp.float64(6.28318530717958647692)


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


arb_exp_batch_jit = jax.jit(arb_exp_batch)
arb_log_batch_jit = jax.jit(arb_log_batch)
arb_sqrt_batch_jit = jax.jit(arb_sqrt_batch)
arb_sin_batch_jit = jax.jit(arb_sin_batch)
arb_cos_batch_jit = jax.jit(arb_cos_batch)
arb_tan_batch_jit = jax.jit(arb_tan_batch)
arb_sinh_batch_jit = jax.jit(arb_sinh_batch)
arb_cosh_batch_jit = jax.jit(arb_cosh_batch)
arb_tanh_batch_jit = jax.jit(arb_tanh_batch)

arb_exp_batch_prec_jit = jax.jit(arb_exp_batch_prec, static_argnames=("prec_bits",))
arb_log_batch_prec_jit = jax.jit(arb_log_batch_prec, static_argnames=("prec_bits",))
arb_sqrt_batch_prec_jit = jax.jit(arb_sqrt_batch_prec, static_argnames=("prec_bits",))
arb_sin_batch_prec_jit = jax.jit(arb_sin_batch_prec, static_argnames=("prec_bits",))
arb_cos_batch_prec_jit = jax.jit(arb_cos_batch_prec, static_argnames=("prec_bits",))
arb_tan_batch_prec_jit = jax.jit(arb_tan_batch_prec, static_argnames=("prec_bits",))
arb_sinh_batch_prec_jit = jax.jit(arb_sinh_batch_prec, static_argnames=("prec_bits",))
arb_cosh_batch_prec_jit = jax.jit(arb_cosh_batch_prec, static_argnames=("prec_bits",))
arb_tanh_batch_prec_jit = jax.jit(arb_tanh_batch_prec, static_argnames=("prec_bits",))


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
    "arb_exp_prec",
    "arb_log_prec",
    "arb_sqrt_prec",
    "arb_sin_prec",
    "arb_cos_prec",
    "arb_tan_prec",
    "arb_sinh_prec",
    "arb_cosh_prec",
    "arb_tanh_prec",
    "arb_exp_batch",
    "arb_log_batch",
    "arb_sqrt_batch",
    "arb_sin_batch",
    "arb_cos_batch",
    "arb_tan_batch",
    "arb_sinh_batch",
    "arb_cosh_batch",
    "arb_tanh_batch",
    "arb_exp_batch_prec",
    "arb_log_batch_prec",
    "arb_sqrt_batch_prec",
    "arb_sin_batch_prec",
    "arb_cos_batch_prec",
    "arb_tan_batch_prec",
    "arb_sinh_batch_prec",
    "arb_cosh_batch_prec",
    "arb_tanh_batch_prec",
    "arb_exp_batch_jit",
    "arb_log_batch_jit",
    "arb_sqrt_batch_jit",
    "arb_sin_batch_jit",
    "arb_cos_batch_jit",
    "arb_tan_batch_jit",
    "arb_sinh_batch_jit",
    "arb_cosh_batch_jit",
    "arb_tanh_batch_jit",
    "arb_exp_batch_prec_jit",
    "arb_log_batch_prec_jit",
    "arb_sqrt_batch_prec_jit",
    "arb_sin_batch_prec_jit",
    "arb_cos_batch_prec_jit",
    "arb_tan_batch_prec_jit",
    "arb_sinh_batch_prec_jit",
    "arb_cosh_batch_prec_jit",
    "arb_tanh_batch_prec_jit",
]
